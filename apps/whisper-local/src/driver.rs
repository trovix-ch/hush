//! Recorder start/stop run here rather than on a worker: they measured microseconds warm
//! and ~25 ms cold, and their order relative to the next hotkey event matters.

use std::collections::{HashMap, VecDeque};
use std::path::PathBuf;
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, Sender};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use anyhow::{Context, Result};
use wl_audio::CpalRecorder;
use wl_core::UtteranceId;
use wl_core::cancel::CancelToken;
use wl_core::config::{Config, GpuPolicy};
use wl_core::insert::{InsertError, InsertOutcome, InsertPolicy};
use wl_core::normalize::Provenance;
use wl_core::notify::{Notifier, OverlayState as CoreOverlay, Sound};
use wl_core::pipeline::{Effect, Event, Failure, Pipeline, PipelineConfig, Stage};
use wl_core::recorder::{Recorder, RecorderConfig};
use wl_platform_windows::focus::WinFocus;
use wl_platform_windows::hook::{HookHandle, HotkeyConfig, HotkeyEvent, HotkeyHook};
use wl_platform_windows::overlay::OverlayState;
use wl_platform_windows::tray::TrayEvent;
use wl_platform_windows::ui_thread::{
    self, INSTANCE_MUTEX, UiError, UiHandle, UiOptions, WinNotifier,
};

use crate::engines::{self, EngineSummary};
use crate::setup::{self, Paths};
use crate::workers::{
    self, EngineLoader, InsertCmd, NormCmd, NormJob, SttJob, Timers, outcome_label,
};

const LEVEL_PERIOD: Duration = Duration::from_millis(50);

pub enum Msg {
    Hotkey(HotkeyEvent),
    Tray(TrayEvent),
    Event(Event),
    NoSpeech(UtteranceId),
    EngineReady(std::result::Result<EngineSummary, String>),
    /// `Err` means the app stays rules-only.
    NormalizerReady(std::result::Result<String, String>),
    PasteLastDone(std::result::Result<InsertOutcome, InsertError>),
    /// Shown until replaced, unlike a toast.
    Status(String),
    Quit,
}

/// Carries no ids because a simulation runs one utterance at a time.
#[derive(Debug, Clone)]
pub enum Observed {
    /// The recording is already stopped when this is sent.
    Released,
    Transcript {
        text: String,
        inference: Duration,
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
        ui: &UiHandle,
        focus: &WinFocus,
        load: EngineLoader,
        tx: &Sender<Msg>,
    ) -> Result<Self> {
        let (stt, stt_join) = workers::spawn_stt(load, tx.clone())?;
        let (norm, j2) = workers::spawn_normalizer(config.vocabulary.clone(), tx.clone())?;
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
    let Some(http) = engines::http_config(&config.normalizer) else {
        let _ = tx.send(Msg::NormalizerReady(Err(
            "rules only (normalizer.kind = \"rules\")".into(),
        )));
        return;
    };
    let spawned = std::thread::Builder::new()
        .name("wl-normalizer-warm".into())
        .spawn(move || {
            let label = format!("{} via {}", http.model, http.base_url);
            let msg = match engines::build_http_normalizer(http) {
                Ok((n, warm)) => {
                    let _ = norm.send(NormCmd::Upgrade(n));
                    Ok(format!("{label} (warm-up {} ms)", warm.as_millis()))
                }
                Err(e) => Err(format!("{e:#}; continuing rules-only")),
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
}

pub struct DriverParts {
    pub config: Config,
    pub config_path: PathBuf,
    pub recorder: Box<dyn Recorder>,
    pub ui: UiHandle,
    pub focus: WinFocus,
    pub hook: Option<HookHandle>,
    pub workers: Workers,
    pub rx: Receiver<Msg>,
    pub observer: Option<Sender<Observed>>,
}

impl Driver {
    pub fn new(p: DriverParts) -> Self {
        Self {
            pipeline: Pipeline::new(PipelineConfig::from_config(&p.config)),
            recorder: p.recorder,
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
                self.last_level = Instant::now();
                self.notifier.set_state(CoreOverlay::Listening {
                    level: self.recorder.level(),
                });
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
                self.feed(ev);
            }
            Msg::NoSpeech(id) => {
                tracing::info!(%id, "no speech detected; nothing to insert");
                self.observe(Observed::NoSpeech);
                // Escape removed the token; the user has moved on, so say nothing.
                let live = self.tokens.contains_key(&id);
                self.feed(Event::TranscriptReady(
                    id,
                    wl_core::stt::Transcript {
                        utterance: id,
                        ..Default::default()
                    },
                ));
                if live && !self.pipeline.is_recording() {
                    self.notifier.toast("No speech detected");
                }
            }
            Msg::EngineReady(r) => self.on_engine_ready(r),
            Msg::NormalizerReady(r) => {
                match r {
                    Ok(desc) => {
                        tracing::info!(normalizer = %desc, "LLM normalizer ready");
                        self.normalizer = "LLM".into();
                    }
                    Err(why) => {
                        tracing::info!(reason = %why, "normalizer: rules only");
                        self.normalizer = "rules".into();
                    }
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
            Event::TranscriptReady(_, t) => Observed::Transcript {
                text: t.text.clone(),
                inference: t.inference_time,
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
                self.feed(Event::HotkeyUp { at });
                if was && !self.pipeline.is_recording() {
                    self.observe(Observed::Released);
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
            .set_tooltip(format!("whisper-local · {engine} · {}", self.normalizer));
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
    }

    fn token(&mut self, id: UtteranceId) -> CancelToken {
        self.tokens.entry(id).or_default().clone()
    }

    fn execute(&mut self, fx: Effect) -> Option<Event> {
        match fx {
            Effect::StartRecording(id) => self
                .recorder
                .start()
                .err()
                .map(|e| Event::Failed(id, Failure::Record(e))),
            Effect::StopRecording(id) => match self.recorder.stop() {
                Ok(rec) => {
                    if rec.dropped_frames > 0 || rec.device_lost {
                        tracing::warn!(%id, dropped = rec.dropped_frames, device_lost = rec.device_lost, "recording has gaps");
                    }
                    tracing::debug!(%id, secs = rec.duration.as_secs_f64(), "recorded");
                    Some(Event::Recorded(id, rec))
                }
                Err(e) => Some(Event::Failed(id, Failure::Record(e))),
            },
            Effect::DiscardRecording(_) => {
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
                let cancel = self.token(id);
                self.workers
                    .stt
                    .send(SttJob { id, pcm, cancel })
                    .err()
                    .map(|_| {
                        Event::Failed(
                            id,
                            Failure::Stt(wl_core::stt::SttError::BackendDied(
                                "speech worker is gone".into(),
                            )),
                        )
                    })
            }
            Effect::Normalize {
                id,
                transcript,
                ctx,
            } => {
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
                            Failure::Normalize(wl_core::normalize::NormalizeError::BackendDied(
                                "normalize worker is gone".into(),
                            )),
                        )
                    })
            }
            Effect::Insert { id, text, target } => {
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
                None
            }
            Effect::Notify(state) => {
                if let CoreOverlay::Error { message } = &state {
                    tracing::warn!(%message, "shown to the user");
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
            eprintln!("whisper-local is already running in this session (see the tray).");
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
        remote_session = wl_platform_windows::focus::is_remote_session(),
        elevated = wl_platform_windows::focus::self_elevated(),
        "starting"
    );

    let (ui, tray_rx) = UiHandle::start(UiOptions {
        tray: true,
        ..Default::default()
    })?;
    ui.set_tooltip("whisper-local · loading model…");
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
    forward(tray_rx, tx.clone(), Msg::Tray, "wl-tray-fwd")?;
    // Bounded because the hook callback must never block: it drops on a full channel.
    let (hk_tx, hk_rx) = mpsc::sync_channel::<HotkeyEvent>(64);
    forward(hk_rx, tx.clone(), Msg::Hotkey, "wl-hotkey-fwd")?;

    let focus = WinFocus::new();
    let recorder = CpalRecorder::new(RecorderConfig {
        max_duration: config.max_recording,
        ..RecorderConfig::default()
    })
    .context("starting the audio thread")?;

    let engine_choice = config.engine.clone();
    let status = tx.clone();
    let load: EngineLoader = Box::new(move || {
        if !model.is_present(&model_dir) {
            let _ = status.send(Msg::Status("Downloading speech model…".into()));
            wl_stt::models::ensure_downloaded(model, &model_dir)?;
        }
        engines::load_engine(&engine_choice, &model.load_path(&model_dir))
    });
    let workers = Workers::spawn(&config, &ui, &focus, load, &tx)?;
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
        ui,
        focus,
        hook: Some(hook),
        workers,
        rx,
        observer: None,
    });
    drop(tx);
    std::thread::Builder::new()
        .name("wl-driver".into())
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
