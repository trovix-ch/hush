//! Every worker turns a panic into a `Failed` event; the pipeline would otherwise wait
//! forever for a result.

use std::collections::VecDeque;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::thread::JoinHandle;
use std::time::{Duration, Instant, SystemTime};

use hush_audio::vad::{EnergyVad, Vad, has_speech, trim_silence};
use hush_core::UtteranceId;
use hush_core::cancel::CancelToken;
use hush_core::config::Config;
use hush_core::context::FocusContext;
use hush_core::insert::{InsertError, InsertOutcome, InsertPolicy, Inserter, StrategyChain};
use hush_core::normalize::{NormalizeError, NormalizeRequest, Normalizer};
use hush_core::pipeline::{Event, Failure, NormalizeContext, Timer};
use hush_core::stt::{DecodeOptions, SttEngine, SttError, Transcript};
use hush_normalize::RuleNormalizer;
use hush_platform_windows::clipboard::WinClipboard;
use hush_platform_windows::focus::WinFocus;
use hush_platform_windows::input::WinInput;

use crate::driver::Msg;
use crate::engines::EngineSummary;

const VAD_PAD: Duration = Duration::from_millis(200);

fn panic_text(p: &(dyn std::any::Any + Send)) -> String {
    p.downcast_ref::<&str>()
        .map(|s| s.to_string())
        .or_else(|| p.downcast_ref::<String>().cloned())
        .unwrap_or_else(|| "unknown panic".into())
}

fn spawn(name: &str, f: impl FnOnce() + Send + 'static) -> std::io::Result<JoinHandle<()>> {
    std::thread::Builder::new().name(name.into()).spawn(f)
}

pub struct SttJob {
    pub id: UtteranceId,
    pub segment_index: u32,
    pub pcm: Vec<f32>,
    pub cancel: CancelToken,
}

pub enum SttCmd {
    Job(SttJob),
    Nudge,
}

fn no_speech(id: UtteranceId, segment_index: u32) -> Event {
    Event::SegmentReady {
        id,
        segment_index,
        transcript: Transcript {
            utterance: id,
            ..Default::default()
        },
    }
}

pub type EngineLoader =
    Box<dyn FnOnce() -> anyhow::Result<(Box<dyn SttEngine>, EngineSummary)> + Send>;

/// Jobs sent while the engine loads wait in the channel, so the first dictation is only
/// slower, not lost.
pub fn spawn_stt(
    load: EngineLoader,
    out: Sender<Msg>,
) -> std::io::Result<(Sender<SttCmd>, JoinHandle<()>)> {
    let (tx, rx) = mpsc::channel::<SttCmd>();
    let join = spawn("hush-stt", move || {
        let loaded = match catch_unwind(AssertUnwindSafe(load)) {
            Ok(Ok(loaded)) => Ok(loaded),
            Ok(Err(e)) => Err(format!("{e:#}")),
            Err(p) => Err(format!("engine load panicked: {}", panic_text(p.as_ref()))),
        };
        let mut engine: Result<Box<dyn SttEngine>, String> = match loaded {
            Ok((engine, summary)) => {
                let _ = out.send(Msg::EngineReady(Ok(summary)));
                Ok(engine)
            }
            Err(reason) => {
                let _ = out.send(Msg::EngineReady(Err(reason.clone())));
                Err(reason)
            }
        };
        for cmd in rx {
            let job = match cmd {
                SttCmd::Job(job) => job,
                SttCmd::Nudge => {
                    if let Ok(e) = engine.as_mut() {
                        match catch_unwind(AssertUnwindSafe(|| e.nudge())) {
                            Ok(Ok(())) => {}
                            Ok(Err(err)) => tracing::debug!(error = %err, "gpu nudge failed"),
                            Err(p) => {
                                engine =
                                    Err(format!("engine panicked: {}", panic_text(p.as_ref())));
                            }
                        }
                    }
                    continue;
                }
            };
            // Escape already moved the pipeline on; nobody is waiting for this answer.
            if job.cancel.is_cancelled() {
                continue;
            }
            let (id, segment_index) = (job.id, job.segment_index);
            let vad_started = Instant::now();
            let mut vad = EnergyVad::default();
            let mut events = vad.push(&job.pcm);
            events.extend(vad.finish());
            if !has_speech(&events) {
                if out.send(Msg::Event(no_speech(id, segment_index))).is_err() {
                    break;
                }
                continue;
            }
            let pcm = trim_silence(&job.pcm, &events, VAD_PAD);
            tracing::debug!(%id, samples = job.pcm.len(), trimmed = pcm.len(), vad_ms = vad_started.elapsed().as_secs_f64() * 1e3, "vad");
            let result = match engine.as_mut() {
                Err(reason) => Err(SttError::Load(reason.clone())),
                Ok(e) => {
                    let opts = DecodeOptions {
                        utterance: id,
                        cancel: job.cancel.clone(),
                        ..Default::default()
                    };
                    match catch_unwind(AssertUnwindSafe(|| e.transcribe(&pcm, &opts))) {
                        Ok(r) => r,
                        Err(p) => {
                            // The native session may be half torn down; never reuse it.
                            let msg = format!("engine panicked: {}", panic_text(p.as_ref()));
                            engine = Err(msg.clone());
                            Err(SttError::BackendDied(msg))
                        }
                    }
                }
            };
            let ev = match result {
                Ok(transcript) => Event::SegmentReady {
                    id,
                    segment_index,
                    transcript,
                },
                Err(SttError::EmptyAudio) => no_speech(id, segment_index),
                Err(e) => Event::Failed(id, Failure::Stt(e)),
            };
            if out.send(Msg::Event(ev)).is_err() {
                break;
            }
        }
    })?;
    Ok((tx, join))
}

pub struct NormJob {
    pub id: UtteranceId,
    pub transcript: String,
    pub ctx: NormalizeContext,
    pub cancel: CancelToken,
}

pub enum NormCmd {
    Job(NormJob),
    Upgrade(Box<dyn Normalizer>),
}

/// The inline list plus the file's entries, re-read when the file's mtime changes.
pub struct Vocabulary {
    inline: Vec<String>,
    file: Option<PathBuf>,
    stamp: Option<Result<SystemTime, String>>,
    error: Option<String>,
    merged: Vec<String>,
}

impl Vocabulary {
    /// A relative `vocabulary_file` is taken relative to the config file.
    pub fn from_config(config: &Config, config_file: &Path) -> Self {
        let file = config.vocabulary_file.as_ref().map(|f| {
            config_file
                .parent()
                .map_or_else(|| f.clone(), |dir| dir.join(f))
        });
        Self::new(config.vocabulary.clone(), file)
    }

    pub fn new(inline: Vec<String>, file: Option<PathBuf>) -> Self {
        let merged = merge_vocabulary(&inline, &[]);
        let mut v = Self {
            inline,
            file,
            stamp: None,
            error: None,
            merged,
        };
        v.refresh();
        v
    }

    pub fn file(&self) -> Option<&Path> {
        self.file.as_deref()
    }

    /// The error reading the file, if the last attempt failed.
    pub fn file_error(&self) -> Option<&str> {
        self.error.as_deref()
    }

    pub fn entries(&self) -> &[String] {
        &self.merged
    }

    pub fn refresh(&mut self) -> &[String] {
        let Some(path) = &self.file else {
            return &self.merged;
        };
        let stamp = std::fs::metadata(path)
            .and_then(|m| m.modified())
            .map_err(|e| e.to_string());
        if self.stamp.as_ref() == Some(&stamp) {
            return &self.merged;
        }
        let read = stamp
            .clone()
            .and_then(|_| std::fs::read_to_string(path).map_err(|e| e.to_string()));
        let from_file = match &read {
            Ok(text) => parse_vocabulary_file(text),
            Err(e) => {
                tracing::warn!(path = %path.display(), error = %e, "cannot read the vocabulary file");
                Vec::new()
            }
        };
        self.merged = merge_vocabulary(&self.inline, &from_file);
        tracing::info!(path = %path.display(), from_file = from_file.len(), total = self.merged.len(), "vocabulary loaded");
        self.error = read.err();
        self.stamp = Some(stamp);
        &self.merged
    }
}

pub fn parse_vocabulary_file(text: &str) -> Vec<String> {
    text.lines()
        .map(|l| l.split_once('#').map_or(l, |(before, _)| before).trim())
        .filter(|l| !l.is_empty())
        .map(str::to_string)
        .collect()
}

/// Inline entries first; an exact duplicate is kept once.
pub fn merge_vocabulary(inline: &[String], from_file: &[String]) -> Vec<String> {
    let mut out: Vec<String> = Vec::with_capacity(inline.len() + from_file.len());
    for e in inline.iter().chain(from_file).map(|e| e.trim()) {
        if !e.is_empty() && !out.iter().any(|o| o == e) {
            out.push(e.to_string());
        }
    }
    out
}

/// Starts rules-only, so dictation works before (or without) the LLM.
pub fn spawn_normalizer(
    mut vocabulary: Vocabulary,
    out: Sender<Msg>,
) -> std::io::Result<(Sender<NormCmd>, JoinHandle<()>)> {
    let (tx, rx) = mpsc::channel::<NormCmd>();
    let join = spawn("hush-normalize", move || {
        let mut rules = RuleNormalizer::new();
        let mut full: Option<Box<dyn Normalizer>> = None;
        for cmd in rx {
            let job = match cmd {
                NormCmd::Upgrade(n) => {
                    tracing::info!(normalizer = n.id(), "LLM normalizer active");
                    full = Some(n);
                    continue;
                }
                NormCmd::Job(j) => j,
            };
            if job.cancel.is_cancelled() {
                continue;
            }
            let id = job.id;
            let n: &mut dyn Normalizer = match full.as_mut() {
                Some(f) if !job.ctx.rules_only => f.as_mut(),
                _ => &mut rules,
            };
            let req = NormalizeRequest {
                transcript: &job.transcript,
                language: job.ctx.language.as_deref(),
                vocabulary: vocabulary.refresh(),
                app: &job.ctx.app,
                previous: job.ctx.previous.as_deref(),
                utterance: id,
                cancel: job.cancel.clone(),
            };
            let ev = match catch_unwind(AssertUnwindSafe(|| n.normalize(&req))) {
                Ok(Ok(o)) => Event::NormalizedReady(id, o),
                Ok(Err(e)) => Event::Failed(id, Failure::Normalize(e)),
                Err(p) => {
                    // The retry is rules-only anyway; dropping `full` spares the next
                    // utterance.
                    full = None;
                    Event::Failed(
                        id,
                        Failure::Normalize(NormalizeError::BackendDied(panic_text(p.as_ref()))),
                    )
                }
            };
            if out.send(Msg::Event(ev)).is_err() {
                break;
            }
        }
    })?;
    Ok((tx, join))
}

pub enum InsertCmd {
    Insert {
        id: UtteranceId,
        text: String,
        target: FocusContext,
        cancel: CancelToken,
    },
    PasteLast {
        text: String,
    },
}

/// The tray menu stays our foreground window until it closes, so focus is captured only
/// after it is gone.
const PASTE_LAST_SETTLE: Duration = Duration::from_millis(250);

pub fn spawn_inserter(
    clipboard: WinClipboard,
    focus: WinFocus,
    policy: InsertPolicy,
    out: Sender<Msg>,
) -> std::io::Result<(Sender<InsertCmd>, JoinHandle<()>)> {
    let (tx, rx) = mpsc::channel::<InsertCmd>();
    let join = spawn("hush-insert", move || {
        let mut chain = StrategyChain::new(clipboard, WinInput::new(), focus.clone(), policy);
        let mut insert = |target: &FocusContext, text: &str, cancel: &CancelToken| {
            catch_unwind(AssertUnwindSafe(|| chain.insert(target, text, cancel))).unwrap_or_else(
                |p| {
                    Err(InsertError::Input(format!(
                        "panicked: {}",
                        panic_text(p.as_ref())
                    )))
                },
            )
        };
        for cmd in rx {
            let msg = match cmd {
                InsertCmd::Insert {
                    id,
                    text,
                    target,
                    cancel,
                } => match insert(&target, &text, &cancel) {
                    Ok(o) => Msg::Event(Event::InsertDone(id, o)),
                    Err(e) => Msg::Event(Event::Failed(id, Failure::Insert(e))),
                },
                InsertCmd::PasteLast { text } => {
                    std::thread::sleep(PASTE_LAST_SETTLE);
                    let target = focus.capture().to_core();
                    Msg::PasteLastDone(insert(&target, &text, &CancelToken::new()))
                }
            };
            if out.send(msg).is_err() {
                break;
            }
        }
    })?;
    Ok((tx, join))
}

/// Without the dictated text.
pub fn outcome_label(o: &Result<InsertOutcome, InsertError>) -> String {
    match o {
        Ok(o) => format!("{o:?}"),
        Err(e) => format!("error: {e}"),
    }
}

struct Armed {
    at: Instant,
    id: UtteranceId,
    timer: Timer,
}

#[derive(Clone)]
pub struct Timers {
    tx: Sender<Armed>,
}

impl Timers {
    /// Timers are never disarmed: the pipeline ignores one that fires for an utterance
    /// it has moved past.
    pub fn arm(&self, id: UtteranceId, timer: Timer, after: Duration) {
        let _ = self.tx.send(Armed {
            at: Instant::now() + after,
            id,
            timer,
        });
    }
}

pub fn spawn_timers(out: Sender<Msg>) -> std::io::Result<(Timers, JoinHandle<()>)> {
    let (tx, rx) = mpsc::channel::<Armed>();
    let join = spawn("hush-timers", move || timer_loop(&rx, &out))?;
    Ok((Timers { tx }, join))
}

fn timer_loop(rx: &Receiver<Armed>, out: &Sender<Msg>) {
    // A handful at most, so a sorted deque beats a heap for clarity.
    let mut pending: VecDeque<Armed> = VecDeque::new();
    loop {
        let now = Instant::now();
        while pending.front().is_some_and(|a| a.at <= now) {
            let Some(a) = pending.pop_front() else { break };
            let ev = match a.timer {
                Timer::TapWindow => Event::TapWindowElapsed(a.id),
                Timer::MaxDuration => Event::MaxDurationReached(a.id),
            };
            if out.send(Msg::Event(ev)).is_err() {
                return;
            }
        }
        let next = match pending.front() {
            Some(a) => rx.recv_timeout(a.at.saturating_duration_since(now)),
            None => rx.recv().map_err(|_| RecvTimeoutError::Disconnected),
        };
        match next {
            Ok(a) => {
                let pos = pending
                    .iter()
                    .position(|p| p.at > a.at)
                    .unwrap_or(pending.len());
                pending.insert(pos, a);
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => return,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn timers_fire_in_deadline_order() {
        let (out, rx) = mpsc::channel();
        let (timers, join) = spawn_timers(out).unwrap();
        timers.arm(
            UtteranceId(1),
            Timer::MaxDuration,
            Duration::from_millis(60),
        );
        timers.arm(UtteranceId(2), Timer::TapWindow, Duration::from_millis(10));
        let got: Vec<_> = (0..2)
            .map(|_| match rx.recv_timeout(Duration::from_secs(2)).unwrap() {
                Msg::Event(Event::TapWindowElapsed(id)) => (id, "tap"),
                Msg::Event(Event::MaxDurationReached(id)) => (id, "max"),
                _ => panic!("unexpected message"),
            })
            .collect();
        assert_eq!(got, vec![(UtteranceId(2), "tap"), (UtteranceId(1), "max")]);
        drop(timers);
        join.join().unwrap();
    }

    #[test]
    fn silence_never_reaches_the_engine_and_load_failure_is_a_failed_event() {
        let (out, rx) = mpsc::channel();
        let loader: EngineLoader = Box::new(|| anyhow::bail!("no model here"));
        let (tx, join) = spawn_stt(loader, out).unwrap();
        assert!(matches!(
            rx.recv_timeout(Duration::from_secs(2)).unwrap(),
            Msg::EngineReady(Err(_))
        ));
        // A nudge answers nothing, even with no engine.
        tx.send(SttCmd::Nudge).unwrap();
        tx.send(SttCmd::Job(SttJob {
            id: UtteranceId(1),
            segment_index: 3,
            pcm: vec![0.0; 16_000],
            cancel: CancelToken::new(),
        }))
        .unwrap();
        match rx.recv_timeout(Duration::from_secs(2)).unwrap() {
            Msg::Event(Event::SegmentReady {
                id: UtteranceId(1),
                segment_index: 3,
                transcript,
            }) => assert!(transcript.text.is_empty()),
            _ => panic!("unexpected message"),
        }
        let tone: Vec<f32> = (0..16_000).map(|i| (i as f32 * 0.05).sin() * 0.3).collect();
        for id in [UtteranceId(2), UtteranceId(3)] {
            tx.send(SttCmd::Job(SttJob {
                id,
                segment_index: 0,
                pcm: tone.clone(),
                cancel: CancelToken::new(),
            }))
            .unwrap();
            match rx.recv_timeout(Duration::from_secs(2)).unwrap() {
                Msg::Event(Event::Failed(got, Failure::Stt(SttError::Load(reason)))) => {
                    assert_eq!(got, id);
                    assert!(reason.contains("no model here"), "{reason}");
                }
                _ => panic!("unexpected message"),
            }
        }
        drop(tx);
        join.join().unwrap();
    }

    #[test]
    fn rules_only_normalizer_answers_and_skips_cancelled_jobs() {
        let (out, rx) = mpsc::channel();
        let (tx, join) = spawn_normalizer(Vocabulary::new(vec![], None), out).unwrap();
        let ctx = NormalizeContext {
            app: Default::default(),
            language: Some("en".into()),
            previous: None,
            rules_only: false,
        };
        let cancelled = CancelToken::new();
        cancelled.cancel();
        tx.send(NormCmd::Job(NormJob {
            id: UtteranceId(1),
            transcript: "um hello there".into(),
            ctx: ctx.clone(),
            cancel: cancelled,
        }))
        .unwrap();
        tx.send(NormCmd::Job(NormJob {
            id: UtteranceId(2),
            transcript: "um hello there".into(),
            ctx,
            cancel: CancelToken::new(),
        }))
        .unwrap();
        match rx.recv_timeout(Duration::from_secs(2)).unwrap() {
            Msg::Event(Event::NormalizedReady(id, o)) => {
                assert_eq!(id, UtteranceId(2));
                assert!(!o.text.to_lowercase().contains("um"), "{}", o.text);
            }
            _ => panic!("unexpected message"),
        }
        drop(tx);
        join.join().unwrap();
    }

    #[test]
    fn vocabulary_file_parses_lines_and_comments() {
        let text = "# names\nKubernetes\n  gRPC  # the RPC one\n\n#Ignored\nJane Doe\n";
        assert_eq!(
            parse_vocabulary_file(text),
            vec!["Kubernetes", "gRPC", "Jane Doe"]
        );
    }

    #[test]
    fn vocabulary_merges_inline_first_without_duplicates() {
        let inline = vec!["gRPC".to_string(), " ".to_string()];
        let file = vec!["Kubernetes".to_string(), "gRPC".to_string()];
        assert_eq!(merge_vocabulary(&inline, &file), vec!["gRPC", "Kubernetes"]);
    }

    #[test]
    fn vocabulary_file_is_reread_when_its_mtime_changes() {
        let dir = std::env::temp_dir().join(format!("hush-vocab-test-{}", std::process::id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("words.txt");
        let write = |text: &str, secs: u64| {
            std::fs::write(&path, text).unwrap();
            let f = std::fs::File::options().write(true).open(&path).unwrap();
            f.set_modified(SystemTime::UNIX_EPOCH + Duration::from_secs(secs))
                .unwrap();
        };
        write("Kubernetes\n", 1_000_000);

        let config = Config {
            vocabulary: vec!["gRPC".into()],
            vocabulary_file: Some("words.txt".into()),
            ..Config::default()
        };
        let mut v = Vocabulary::from_config(&config, &dir.join("config.toml"));
        assert_eq!(v.file(), Some(path.as_path()));
        assert_eq!(v.entries(), ["gRPC", "Kubernetes"]);

        write("Kubernetes\nTailscale\n", 2_000_000);
        assert_eq!(v.refresh(), ["gRPC", "Kubernetes", "Tailscale"]);

        std::fs::remove_file(&path).unwrap();
        assert_eq!(v.refresh(), ["gRPC"]);
        assert!(v.file_error().is_some());
        let _ = std::fs::remove_dir_all(&dir);
    }
}
