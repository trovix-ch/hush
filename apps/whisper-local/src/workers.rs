//! Every worker turns a panic into a `Failed` event; the pipeline would otherwise wait
//! forever for a result.

use std::collections::VecDeque;
use std::panic::{AssertUnwindSafe, catch_unwind};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use wl_audio::vad::{EnergyVad, Vad, has_speech, trim_silence};
use wl_core::UtteranceId;
use wl_core::cancel::CancelToken;
use wl_core::context::FocusContext;
use wl_core::insert::{InsertError, InsertOutcome, InsertPolicy, Inserter, StrategyChain};
use wl_core::normalize::{NormalizeError, NormalizeRequest, Normalizer};
use wl_core::pipeline::{Event, Failure, NormalizeContext, Timer};
use wl_core::stt::{DecodeOptions, SttEngine, SttError};
use wl_normalize::RuleNormalizer;
use wl_platform_windows::clipboard::WinClipboard;
use wl_platform_windows::focus::WinFocus;
use wl_platform_windows::input::WinInput;

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
    pub pcm: Vec<f32>,
    pub cancel: CancelToken,
}

pub type EngineLoader =
    Box<dyn FnOnce() -> anyhow::Result<(Box<dyn SttEngine>, EngineSummary)> + Send>;

/// Jobs sent while the engine loads wait in the channel, so the first dictation is only
/// slower, not lost.
pub fn spawn_stt(
    load: EngineLoader,
    out: Sender<Msg>,
) -> std::io::Result<(Sender<SttJob>, JoinHandle<()>)> {
    let (tx, rx) = mpsc::channel::<SttJob>();
    let join = spawn("wl-stt", move || {
        let mut engine = match catch_unwind(AssertUnwindSafe(load)) {
            Ok(Ok((engine, summary))) => {
                let _ = out.send(Msg::EngineReady(Ok(summary)));
                Some(engine)
            }
            Ok(Err(e)) => {
                let _ = out.send(Msg::EngineReady(Err(format!("{e:#}"))));
                None
            }
            Err(p) => {
                let _ = out.send(Msg::EngineReady(Err(format!(
                    "engine load panicked: {}",
                    panic_text(p.as_ref())
                ))));
                None
            }
        };
        let mut load_error = None::<String>;
        for job in rx {
            // Escape already moved the pipeline on; nobody is waiting for this answer.
            if job.cancel.is_cancelled() {
                continue;
            }
            let id = job.id;
            let vad_started = Instant::now();
            let mut vad = EnergyVad::default();
            let mut events = vad.push(&job.pcm);
            events.extend(vad.finish());
            if !has_speech(&events) {
                let _ = out.send(Msg::NoSpeech(id));
                continue;
            }
            let pcm = trim_silence(&job.pcm, &events, VAD_PAD);
            tracing::debug!(%id, samples = job.pcm.len(), trimmed = pcm.len(), vad_ms = vad_started.elapsed().as_secs_f64() * 1e3, "vad");
            let result = match engine.as_mut() {
                None => Err(SttError::Load(
                    load_error
                        .get_or_insert_with(|| "the speech engine did not load".into())
                        .clone(),
                )),
                Some(e) => {
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
                            engine = None;
                            load_error = Some(msg.clone());
                            Err(SttError::BackendDied(msg))
                        }
                    }
                }
            };
            let ev = match result {
                Ok(t) => Event::TranscriptReady(id, t),
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

/// Starts rules-only, so dictation works before (or without) the LLM.
pub fn spawn_normalizer(
    vocabulary: Vec<String>,
    out: Sender<Msg>,
) -> std::io::Result<(Sender<NormCmd>, JoinHandle<()>)> {
    let (tx, rx) = mpsc::channel::<NormCmd>();
    let join = spawn("wl-normalize", move || {
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
                vocabulary: &vocabulary,
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
    let join = spawn("wl-insert", move || {
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
    let join = spawn("wl-timers", move || timer_loop(&rx, &out))?;
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
        tx.send(SttJob {
            id: UtteranceId(1),
            pcm: vec![0.0; 16_000],
            cancel: CancelToken::new(),
        })
        .unwrap();
        assert!(matches!(
            rx.recv_timeout(Duration::from_secs(2)).unwrap(),
            Msg::NoSpeech(UtteranceId(1))
        ));
        let tone: Vec<f32> = (0..16_000).map(|i| (i as f32 * 0.05).sin() * 0.3).collect();
        tx.send(SttJob {
            id: UtteranceId(2),
            pcm: tone,
            cancel: CancelToken::new(),
        })
        .unwrap();
        assert!(matches!(
            rx.recv_timeout(Duration::from_secs(2)).unwrap(),
            Msg::Event(Event::Failed(
                UtteranceId(2),
                Failure::Stt(SttError::Load(_))
            ))
        ));
        drop(tx);
        join.join().unwrap();
    }

    #[test]
    fn rules_only_normalizer_answers_and_skips_cancelled_jobs() {
        let (out, rx) = mpsc::channel();
        let (tx, join) = spawn_normalizer(vec![], out).unwrap();
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
}
