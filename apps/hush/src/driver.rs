//! Recorder start/stop run here rather than on a worker: they measured microseconds warm
//! and ~25 ms cold, and their order relative to the next hotkey event matters.

use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use hush_audio::CpalRecorder;
use hush_audio::display::LevelBallistics;
use hush_audio::vad::Vad;
use hush_core::UtteranceId;
use hush_core::cancel::CancelToken;
use hush_core::config::{Config, GpuPolicy};
use hush_core::insert::{InsertError, InsertOutcome, InsertPolicy};
use hush_core::normalize::{Provenance, Style};
use hush_core::notify::{Notifier, OverlayState as CoreOverlay, Sound};
use hush_core::pipeline::{Effect, Event, Failure, Pipeline, PipelineConfig, Stage};
use hush_core::recorder::{Recorder, RecorderConfig};
use hush_platform_windows::focus::WinFocus;
use hush_platform_windows::hook::{HookHandle, HotkeyConfig, HotkeyEvent, HotkeyHook};
use hush_platform_windows::overlay::OverlayState;
use hush_platform_windows::tray::TrayEvent;
use hush_platform_windows::ui_thread::{
    self, INSTANCE_MUTEX, UiError, UiHandle, UiOptions, WinNotifier,
};

use crate::engines::{self, EngineSummary, NormalizerReady};
use crate::setup::{self, Paths};
use crate::workers::{
    self, EngineLoader, InsertCmd, NormCmd, NormJob, SttJob, Timers, Vocabulary, outcome_label,
};

const LEVEL_PERIOD: Duration = Duration::from_millis(50);

/// Older text is more likely a different thought than the start of this one.
const PREVIOUS_MAX_AGE: Duration = Duration::from_secs(60);

/// The pipeline has already required the same app and a delivered insertion; the age
/// check is here because only the driver sees the clock when the insertion landed.
fn recent_previous(
    previous: Option<String>,
    delivered_at: Option<Instant>,
    now: Instant,
) -> Option<String> {
    previous.filter(|_| {
        delivered_at.is_some_and(|at| now.saturating_duration_since(at) < PREVIOUS_MAX_AGE)
    })
}

/// Normalize as if dictating into this app, whatever window really has focus.
#[derive(Debug, Clone)]
pub struct AppOverride {
    pub exe: Option<String>,
    pub style: Style,
}

pub enum Msg {
    Hotkey(HotkeyEvent),
    Tray(TrayEvent),
    Event(Event),
    EngineReady(std::result::Result<EngineSummary, String>),
    /// `Ok(None)` is rules-only by configuration; `Err` is rules-only because the LLM
    /// could not be had.
    NormalizerReady(std::result::Result<Option<NormalizerReady>, String>),
    PasteLastDone(std::result::Result<InsertOutcome, InsertError>),
    /// Shown until replaced, unlike a toast.
    Status(String),
    Quit,
}

/// Carries no ids because a simulation runs one utterance at a time.
#[derive(Debug, Clone)]
pub enum Observed {
    /// The recording is already stopped when this is sent; `at` is the key-up.
    Released {
        at: Instant,
    },
    /// Audio handed to the speech worker: a closed segment, the tail, or the whole
    /// recording.
    SegmentSent {
        samples: usize,
        while_recording: bool,
    },
    SegmentDone {
        inference: Duration,
    },
    /// The stitched transcript, as it goes to the normalizer.
    Transcript {
        text: String,
    },
    Normalized {
        text: String,
        provenance: Provenance,
    },
    Inserted {
        outcome: InsertOutcome,
    },
    /// Not terminal for a normalizer failure: the pipeline retries rules-only.
    Failed {
        stage: Stage,
        error: String,
    },
    NoSpeech,
}

pub struct Workers {
    pub stt: Sender<SttJob>,
    pub norm: Sender<NormCmd>,
    pub insert: Sender<InsertCmd>,
    pub timers: Timers,
    pub stt_join: JoinHandle<()>,
    pub joins: Vec<JoinHandle<()>>,
}

impl Workers {
    /// `load` runs on the speech worker before its first job.
    pub fn spawn(
        config: &Config,
        vocabulary: Vocabulary,
        ui: &UiHandle,
        focus: &WinFocus,
        load: EngineLoader,
        tx: &Sender<Msg>,
    ) -> Result<Self> {
        let (stt, stt_join) = workers::spawn_stt(load, tx.clone())?;
        let (norm, j2) = workers::spawn_normalizer(vocabulary, tx.clone())?;
        let policy = InsertPolicy {
            apps: config.app_policies(),
            ..InsertPolicy::default()
        };
        let (insert, j3) =
            workers::spawn_inserter(ui.clipboard().clone(), focus.clone(), policy, tx.clone())?;
        let (timers, j4) = workers::spawn_timers(tx.clone())?;
        Ok(Self {
            stt,
            norm,
            insert,
            timers,
            stt_join,
            joins: vec![j2, j3, j4],
        })
    }
}

/// The app is rules-only until (and unless) this reports ready.
pub fn spawn_normalizer_upgrade(config: &Config, norm: Sender<NormCmd>, tx: Sender<Msg>) {
    let config = config.clone();
    let spawned = std::thread::Builder::new()
        .name("hush-normalizer-warm".into())
        .spawn(move || {
            let mut status = |s: &str| {
                let _ = tx.send(Msg::Status(s.into()));
            };
            let msg = match engines::build_normalizer(&config, &mut status) {
                Ok(Some((n, ready))) => {
                    let _ = norm.send(NormCmd::Upgrade(n));
                    Ok(Some(ready))
                }
                Ok(None) => Ok(None),
                Err(e) => Err(format!("{e:#}")),
            };
            let _ = tx.send(Msg::NormalizerReady(msg));
        });
    if let Err(e) = spawned {
        tracing::warn!(error = %e, "could not start the normalizer warm-up; rules only");
    }
}

pub struct Driver {
    pipeline: Pipeline,
    recorder: Box<dyn Recorder>,
    vad: Box<dyn Vad>,
    pre_transcribe: bool,
    /// The utterance the recorder is capturing, set only once `start()` succeeded.
    recording: Option<UtteranceId>,
    /// `Transcribe` effects sent per utterance, which is the next segment's index.
    segments_sent: HashMap<UtteranceId, u32>,
    /// A live segment came back empty; reported only if nothing is inserted before the
    /// pipeline goes idle, since other segments of the same utterance may carry text.
    speechless: bool,
    inserted: bool,
    notifier: WinNotifier,
    ui: UiHandle,
    focus: WinFocus,
    hook: Option<HookHandle>,
    workers: Workers,
    rx: Receiver<Msg>,
    tokens: HashMap<UtteranceId, CancelToken>,
    paused: bool,
    config_path: PathBuf,
    gpu_policy: GpuPolicy,
    engine: Option<String>,
    normalizer: String,
    observer: Option<Sender<Observed>>,
    last_level: Instant,
    level: LevelBallistics,
    last_delivered: Option<Instant>,
    app_override: Option<AppOverride>,
}

pub struct DriverParts {
    pub config: Config,
    pub config_path: PathBuf,
    pub recorder: Box<dyn Recorder>,
    pub vad: Box<dyn Vad>,
    pub ui: UiHandle,
    pub focus: WinFocus,
    pub hook: Option<HookHandle>,
    pub workers: Workers,
    pub rx: Receiver<Msg>,
    pub observer: Option<Sender<Observed>>,
    pub app_override: Option<AppOverride>,
}

impl Driver {
    pub fn new(p: DriverParts) -> Self {
        Self {
            pipeline: Pipeline::new(PipelineConfig::from_config(&p.config)),
            recorder: p.recorder,
            vad: p.vad,
            pre_transcribe: p.config.pipeline.pre_transcribe,
            recording: None,
            segments_sent: HashMap::new(),
            speechless: false,
            inserted: false,
            notifier: WinNotifier::new(p.ui.clone()),
            ui: p.ui,
            focus: p.focus,
            hook: p.hook,
            workers: p.workers,
            rx: p.rx,
            tokens: HashMap::new(),
            paused: false,
            config_path: p.config_path,
            gpu_policy: p.config.engine.gpu,
            engine: None,
            normalizer: "rules".into(),
            observer: p.observer,
            last_level: Instant::now(),
            level: LevelBallistics::default(),
            last_delivered: None,
            app_override: p.app_override,
        }
    }

    pub fn run(mut self) {
        loop {
            let msg = if self.pipeline.is_recording() {
                let wait = LEVEL_PERIOD.saturating_sub(self.last_level.elapsed());
                match self.rx.recv_timeout(wait) {
                    Ok(m) => Some(m),
                    Err(RecvTimeoutError::Timeout) => None,
                    Err(RecvTimeoutError::Disconnected) => break,
                }
            } else {
                match self.rx.recv() {
                    Ok(m) => Some(m),
                    Err(_) => break,
                }
            };
            if let Some(m) = msg
                && !self.on_msg(m)
            {
                break;
            }
            if self.pipeline.is_recording() && self.last_level.elapsed() >= LEVEL_PERIOD {
                let dt = self.last_level.elapsed();
                self.last_level = Instant::now();
                let level = self.level.update(self.recorder.level(), dt);
                self.notifier.set_state(CoreOverlay::Listening { level });
                self.stream_audio();
            }
            if let Some(h) = &self.hook {
                h.set_escape_armed(self.pipeline.is_active());
            }
        }
        self.shutdown();
    }

    fn observe(&self, o: Observed) {
        if let Some(obs) = &self.observer {
            let _ = obs.send(o);
        }
    }

    /// Returns false to quit.
    fn on_msg(&mut self, msg: Msg) -> bool {
        match msg {
            Msg::Quit => return false,
            Msg::Hotkey(h) => self.on_hotkey(h),
            Msg::Tray(t) => return self.on_tray(t),
            Msg::Event(ev) => {
                self.observe_event(&ev);
                if let Event::InsertDone(_, o) = &ev
                    && o.delivered()
                {
                    self.last_delivered = Some(Instant::now());
                }
                if let Event::SegmentReady { id, transcript, .. } = &ev
                    && transcript.text.trim().is_empty()
                    // Escape removed the token; the user has moved on, so say nothing.
                    && self.tokens.contains_key(id)
                {
                    self.speechless = true;
                }
                self.feed(ev);
            }
            Msg::EngineReady(r) => self.on_engine_ready(r),
            Msg::NormalizerReady(r) => {
                let notice = match r {
                    Ok(Some(ready)) => {
                        tracing::info!(normalizer = %ready.label, "LLM normalizer ready");
                        self.normalizer = "LLM".into();
                        match ready.cpu_fallback {
                            // A GPU product never silently becomes a CPU product.
                            Some(why) => {
                                tracing::warn!(reason = %why, "language model runs on the CPU");
                                Some("GPU unavailable; language model runs on the CPU")
                            }
                            None => Some("Language model ready"),
                        }
                    }
                    Ok(None) => {
                        tracing::info!("normalizer: rules only (normalizer.kind = \"rules\")");
                        self.normalizer = "rules".into();
                        None
                    }
                    Err(why) => {
                        tracing::warn!(reason = %why, "language model unavailable; rules only");
                        self.normalizer = "rules".into();
                        Some("Language model unavailable; cleanup is rules-only")
                    }
                };
                if let Some(message) = notice
                    && !self.pipeline.is_active()
                    && !self.paused
                {
                    self.ui.set_overlay(OverlayState::Notice {
                        message: message.into(),
                    });
                }
                self.update_tooltip();
            }
            Msg::PasteLastDone(r) => {
                tracing::info!(outcome = %outcome_label(&r), "paste last");
                match r {
                    Ok(o) => match o.message() {
                        None => self.ui.set_overlay(OverlayState::Done { message: None }),
                        Some(m) => self.ui.set_overlay(OverlayState::Error { message: m }),
                    },
                    Err(e) => self.ui.set_overlay(OverlayState::Error {
                        message: format!("Paste last failed: {e}"),
                    }),
                }
            }
            Msg::Status(s) => {
                if !self.pipeline.is_active() {
                    self.ui.set_overlay(OverlayState::Status { message: s });
                }
            }
        }
        true
    }

    fn observe_event(&self, ev: &Event) {
        if self.observer.is_none() {
            return;
        }
        let o = match ev {
            Event::SegmentReady { transcript, .. } => Observed::SegmentDone {
                inference: transcript.inference_time,
            },
            Event::NormalizedReady(_, n) => Observed::Normalized {
                text: n.text.clone(),
                provenance: n.provenance.clone(),
            },
            Event::InsertDone(_, o) => Observed::Inserted { outcome: o.clone() },
            Event::Failed(_, f) => Observed::Failed {
                stage: f.stage(),
                error: f.to_string(),
            },
            _ => return,
        };
        self.observe(o);
    }

    fn on_hotkey(&mut self, h: HotkeyEvent) {
        match h {
            HotkeyEvent::Down { at } => {
                // The hook stays installed while paused so the key is still swallowed
                // rather than typing a stray Ctrl into the target.
                if self.paused {
                    return;
                }
                let t = Instant::now();
                let snap = self.focus.capture();
                tracing::debug!(
                    exe = snap.exe.as_deref().unwrap_or("?"),
                    elevated = snap.elevated,
                    password = snap.is_password,
                    capture_ms = t.elapsed().as_secs_f64() * 1e3,
                    queued_ms = t.saturating_duration_since(at).as_secs_f64() * 1e3,
                    "hotkey down"
                );
                if snap.elevated {
                    tracing::warn!(exe = ?snap.exe, "target runs elevated; its text will go to the clipboard");
                }
                self.feed(Event::HotkeyDown {
                    at,
                    focus: snap.to_core(),
                });
            }
            HotkeyEvent::Up { at } => {
                let was = self.pipeline.is_recording();
                self.stream_audio();
                self.feed(Event::HotkeyUp { at });
                if was && !self.pipeline.is_recording() {
                    self.observe(Observed::Released { at });
                }
            }
            HotkeyEvent::Cancel { .. } => self.feed(Event::Escape),
            HotkeyEvent::HookReinstalled { reason, .. } => {
                tracing::warn!(
                    ?reason,
                    "keyboard hook was removed by Windows and reinstalled"
                );
            }
        }
    }

    fn on_tray(&mut self, t: TrayEvent) -> bool {
        match t {
            TrayEvent::Quit => return false,
            TrayEvent::TogglePause => {
                self.paused = !self.paused;
                tracing::info!(paused = self.paused, "pause toggled");
                if self.paused && self.pipeline.is_recording() {
                    self.feed(Event::Escape);
                }
                self.ui.set_paused(self.paused);
                self.ui.set_overlay(if self.paused {
                    OverlayState::Status {
                        message: "Paused".into(),
                    }
                } else {
                    OverlayState::Notice {
                        message: "Resumed".into(),
                    }
                });
            }
            TrayEvent::PasteLast => match self.pipeline.last() {
                Some(e) => {
                    let _ = self.workers.insert.send(InsertCmd::PasteLast {
                        text: e.text.clone(),
                    });
                }
                None => self.notifier.toast("Nothing dictated yet"),
            },
            TrayEvent::CopyLast => match self.pipeline.last() {
                Some(e) => match self.ui.clipboard().write_text(&e.text) {
                    Ok(_) => self.notifier.toast("Last transcript copied"),
                    Err(err) => self.ui.set_overlay(OverlayState::Error {
                        message: format!("Copy failed: {err}"),
                    }),
                },
                None => self.notifier.toast("Nothing dictated yet"),
            },
            TrayEvent::OpenConfig => {
                if let Err(e) = ui_thread::open_path(&self.config_path) {
                    tracing::warn!(error = %e, path = %self.config_path.display(), "could not open the config");
                }
            }
            TrayEvent::About => self.ui.show_about(),
        }
        true
    }

    fn on_engine_ready(&mut self, r: std::result::Result<EngineSummary, String>) {
        match r {
            Ok(s) => {
                tracing::info!(
                    model = %s.model,
                    backend = ?s.backend,
                    device = s.device.as_deref().unwrap_or("?"),
                    load_ms = s.load.as_millis() as u64,
                    warm_up_ms = s.warm_up.as_millis() as u64,
                    "speech engine ready"
                );
                self.engine = Some(s.short());
                let message = match &s.fallback {
                    // A GPU product never silently becomes a CPU product.
                    Some(why) => {
                        tracing::warn!(reason = %why, "speech runs on the CPU");
                        OverlayState::Notice {
                            message: "GPU unavailable; speech runs on the CPU".into(),
                        }
                    }
                    None => OverlayState::Notice {
                        message: format!("Ready · {}", s.short()),
                    },
                };
                if !self.pipeline.is_active() && !self.paused {
                    self.ui.set_overlay(message);
                }
            }
            Err(e) => {
                let loud = self.gpu_policy == GpuPolicy::RequireGpu;
                tracing::error!(error = %e, require_gpu = loud, "speech engine failed to load");
                self.engine = Some("speech engine failed".into());
                self.notifier.play(Sound::Error);
                self.ui.set_overlay(OverlayState::Error {
                    message: format!("Speech engine failed: {e}"),
                });
            }
        }
        self.update_tooltip();
    }

    fn update_tooltip(&self) {
        let engine = self.engine.as_deref().unwrap_or("loading model…");
        self.ui
            .set_tooltip(format!("hush · {engine} · {}", self.normalizer));
    }

    fn feed(&mut self, ev: Event) {
        let mut queue = VecDeque::from([ev]);
        while let Some(ev) = queue.pop_front() {
            let effects = self.pipeline.handle(ev);
            for fx in effects {
                if let Some(back) = self.execute(fx) {
                    queue.push_back(back);
                }
            }
        }
        if !self.pipeline.is_active() {
            if std::mem::take(&mut self.speechless) && !self.inserted {
                tracing::info!("no speech detected; nothing to insert");
                self.observe(Observed::NoSpeech);
                self.notifier.toast("No speech detected");
            }
            self.inserted = false;
        }
    }

    fn stream_audio(&mut self) {
        let Some(id) = self.recording.filter(|_| self.pre_transcribe) else {
            return;
        };
        let pcm = self.recorder.take_chunks();
        if pcm.is_empty() {
            return;
        }
        let vad = self.vad.push(&pcm);
        self.feed(Event::Audio {
            id,
            at: Instant::now(),
            pcm,
            vad,
        });
    }

    fn token(&mut self, id: UtteranceId) -> CancelToken {
        self.tokens.entry(id).or_default().clone()
    }

    fn execute(&mut self, fx: Effect) -> Option<Event> {
        match fx {
            Effect::StartRecording(id) => {
                self.level.reset();
                self.last_level = Instant::now();
                match self.recorder.start() {
                    Ok(()) => {
                        self.recording = Some(id);
                        self.vad.reset();
                        None
                    }
                    Err(e) => {
                        self.recording = None;
                        Some(Event::Failed(id, Failure::Record(e)))
                    }
                }
            }
            Effect::StopRecording(id) => {
                self.recording = None;
                match self.recorder.stop() {
                    Ok(rec) => {
                        if rec.dropped_frames > 0 || rec.device_lost {
                            tracing::warn!(%id, dropped = rec.dropped_frames, device_lost = rec.device_lost, "recording has gaps");
                        }
                        tracing::debug!(%id, secs = rec.duration.as_secs_f64(), "recorded");
                        Some(Event::Recorded(id, rec))
                    }
                    Err(e) => Some(Event::Failed(id, Failure::Record(e))),
                }
            }
            Effect::DiscardRecording(_) => {
                self.recording = None;
                self.recorder.cancel();
                None
            }
            Effect::ArmTimer { id, timer, after } => {
                self.workers.timers.arm(id, timer, after);
                None
            }
            Effect::Transcribe { id, pcm } => {
                // Utterances are transcribed in id order, so older tokens are never needed
                // again.
                self.tokens.retain(|k, _| *k >= id);
                self.segments_sent.retain(|k, _| *k >= id);
                let next = self.segments_sent.entry(id).or_default();
                let segment_index = *next;
                *next += 1;
                let cancel = self.token(id);
                self.observe(Observed::SegmentSent {
                    samples: pcm.len(),
                    while_recording: self.pipeline.is_recording(),
                });
                self.workers
                    .stt
                    .send(SttJob {
                        id,
                        segment_index,
                        pcm,
                        cancel,
                    })
                    .err()
                    .map(|_| {
                        Event::Failed(
                            id,
                            Failure::Stt(hush_core::stt::SttError::BackendDied(
                                "speech worker is gone".into(),
                            )),
                        )
                    })
            }
            Effect::Normalize {
                id,
                transcript,
                mut ctx,
            } => {
                ctx.previous = recent_previous(ctx.previous, self.last_delivered, Instant::now());
                if let Some(o) = &self.app_override {
                    ctx.app.exe.clone_from(&o.exe);
                    ctx.app.style = o.style;
                }
                if !ctx.rules_only {
                    self.observe(Observed::Transcript {
                        text: transcript.clone(),
                    });
                }
                let cancel = self.token(id);
                self.workers
                    .norm
                    .send(NormCmd::Job(NormJob {
                        id,
                        transcript,
                        ctx,
                        cancel,
                    }))
                    .err()
                    .map(|_| {
                        Event::Failed(
                            id,
                            Failure::Normalize(hush_core::normalize::NormalizeError::BackendDied(
                                "normalize worker is gone".into(),
                            )),
                        )
                    })
            }
            Effect::Insert { id, text, target } => {
                self.inserted = true;
                let cancel = self.token(id);
                self.workers
                    .insert
                    .send(InsertCmd::Insert {
                        id,
                        text,
                        target,
                        cancel,
                    })
                    .err()
                    .map(|_| {
                        Event::Failed(
                            id,
                            Failure::Insert(InsertError::Input("insert worker is gone".into())),
                        )
                    })
            }
            Effect::CancelInflight(id) => {
                if let Some(t) = self.tokens.remove(&id) {
                    t.cancel();
                }
                // A hands-free restart keeps the id but numbers its segments from 0 again.
                self.segments_sent.remove(&id);
                self.speechless = false;
                None
            }
            Effect::Notify(state) => {
                if let CoreOverlay::Error { message } = &state {
                    tracing::warn!(%message, "shown to the user");
                    // The error says more than "no speech" would.
                    self.speechless = false;
                }
                self.notifier.set_state(state);
                None
            }
            Effect::Play(sound) => {
                self.notifier.play(sound);
                None
            }
        }
    }

    /// The order is deliberate: unhook first so no new events arrive, the UI last.
    fn shutdown(self) {
        let Self {
            hook,
            tokens,
            mut recorder,
            workers,
            ui,
            engine,
            ..
        } = self;
        drop(hook);
        for t in tokens.values() {
            t.cancel();
        }
        recorder.cancel();
        let Workers {
            stt,
            norm,
            insert,
            timers,
            stt_join,
            joins,
        } = workers;
        drop((stt, norm, insert, timers));
        for j in joins {
            let _ = j.join();
        }
        // A speech worker still loading or downloading cannot be interrupted; waiting for
        // it would make Quit hang, and process exit reclaims it anyway.
        if engine.is_some() {
            let _ = stt_join.join();
        }
        drop(recorder);
        ui.shutdown();
        tracing::info!("shut down");
    }
}

pub fn run_app(paths: &Paths) -> Result<std::process::ExitCode> {
    let (config, created) = setup::load_config(&paths.config_file)?;
    let _log = setup::init_logging(&paths.logs_dir, "info,transcribe_cpp=warn")?;
    if created {
        tracing::info!(path = %paths.config_file.display(), "wrote the default config");
    }
    let _instance = match ui_thread::acquire_single_instance(INSTANCE_MUTEX) {
        Ok(g) => g,
        Err(UiError::AlreadyRunning) => {
            eprintln!("hush is already running in this session (see the tray).");
            return Ok(std::process::ExitCode::FAILURE);
        }
        Err(e) => return Err(e).context("single-instance check"),
    };
    let hotkey = HotkeyConfig::parse(&config.hotkey).with_context(|| {
        format!(
            "hotkey {:?} in {}",
            config.hotkey,
            paths.config_file.display()
        )
    })?;
    let (model, model_dir) = engines::resolve_model(&config.engine)?;
    tracing::info!(
        config = %paths.config_file.display(),
        hotkey = %config.hotkey,
        model = %model.id,
        gpu = ?config.engine.gpu,
        gpu_device = ?config.engine.gpu_device,
        remote_session = hush_platform_windows::focus::is_remote_session(),
        elevated = hush_platform_windows::focus::self_elevated(),
        "starting"
    );

    let (ui, tray_rx) = UiHandle::start(UiOptions {
        tray: true,
        ..Default::default()
    })?;
    ui.set_tooltip("hush · loading model…");
    ui.set_overlay(OverlayState::Status {
        message: "Loading speech model…".into(),
    });

    let (tx, rx) = mpsc::channel::<Msg>();
    {
        // The default handler would kill the process with the microphone and hook open.
        let tx = tx.clone();
        if let Err(e) = ctrlc::set_handler(move || {
            let _ = tx.send(Msg::Quit);
        }) {
            tracing::warn!(error = %e, "no Ctrl+C handler; use tray -> Quit");
        }
    }
    forward(tray_rx, tx.clone(), Msg::Tray, "hush-tray-fwd")?;
    // Bounded because the hook callback must never block: it drops on a full channel.
    let (hk_tx, hk_rx) = mpsc::sync_channel::<HotkeyEvent>(64);
    forward(hk_rx, tx.clone(), Msg::Hotkey, "hush-hotkey-fwd")?;

    let focus = WinFocus::new();
    let recorder = CpalRecorder::new(RecorderConfig {
        max_duration: config.max_recording,
        ..RecorderConfig::default()
    })
    .context("starting the audio thread")?;
    let (vad, vad_label) = engines::load_vad();
    tracing::info!(
        vad = %vad_label,
        pre_transcribe = config.pipeline.pre_transcribe,
        segmenter = ?config.pipeline.segmenter,
        "voice activity detection"
    );

    let engine_choice = config.engine.clone();
    let status = tx.clone();
    let load: EngineLoader = Box::new(move || {
        if !model.is_present(&model_dir) {
            let _ = status.send(Msg::Status("Downloading speech model…".into()));
            hush_stt::models::ensure_downloaded(model, &model_dir)?;
        }
        engines::load_engine(&engine_choice, &model.load_path(&model_dir))
    });
    let vocabulary = Vocabulary::from_config(&config, &paths.config_file);
    let workers = Workers::spawn(&config, vocabulary, &ui, &focus, load, &tx)?;
    spawn_normalizer_upgrade(&config, workers.norm.clone(), tx.clone());

    let hook = match HotkeyHook::install(hotkey, hk_tx) {
        Ok(h) => h,
        Err(e) => {
            ui.shutdown();
            return Err(e).context("installing the keyboard hook");
        }
    };
    tracing::info!(hotkey = %config.hotkey, "hold the hotkey and speak; tray -> Quit to exit");

    let driver = Driver::new(DriverParts {
        config,
        config_path: paths.config_file.clone(),
        recorder: Box::new(recorder),
        vad,
        ui,
        focus,
        hook: Some(hook),
        workers,
        rx,
        observer: None,
        app_override: None,
    });
    drop(tx);
    std::thread::Builder::new()
        .name("hush-driver".into())
        .spawn(move || driver.run())
        .context("starting the driver thread")?
        .join()
        .map_err(|_| anyhow::anyhow!("driver thread panicked"))?;
    Ok(std::process::ExitCode::SUCCESS)
}

pub fn forward<T: Send + 'static>(
    rx: Receiver<T>,
    tx: Sender<Msg>,
    wrap: fn(T) -> Msg,
    name: &str,
) -> Result<()> {
    std::thread::Builder::new()
        .name(name.into())
        .spawn(move || {
            while let Ok(v) = rx.recv() {
                if tx.send(wrap(v)).is_err() {
                    break;
                }
            }
        })
        .map(|_| ())
        .context("starting a forwarder thread")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn previous_sentence_is_kept_only_within_the_age_limit() {
        let now = Instant::now() + Duration::from_secs(600);
        let prev = || Some("Hello there.".to_string());
        let ago = |s: u64| Some(now - Duration::from_secs(s));
        assert_eq!(recent_previous(prev(), ago(5), now), prev());
        assert_eq!(recent_previous(prev(), ago(59), now), prev());
        assert_eq!(recent_previous(prev(), ago(60), now), None);
        assert_eq!(recent_previous(prev(), ago(300), now), None);
        assert_eq!(recent_previous(prev(), None, now), None);
        assert_eq!(recent_previous(None, ago(1), now), None);
    }
}
