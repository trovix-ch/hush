//! The dictation state machine: owns no threads, clocks or devices, takes events and
//! returns effects for the caller to execute in order.

use std::collections::VecDeque;
use std::time::{Duration, Instant};

use crate::UtteranceId;
use crate::config::Config;
use crate::context::FocusContext;
use crate::history::{History, HistoryEntry};
use crate::insert::{AppPolicies, InsertError, InsertOutcome};
use crate::normalize::{AppContext, NormalizeError, NormalizeOutput, Provenance};
use crate::notify::{OverlayState, ProvenanceHint, Sound};
use crate::recorder::{RecorderError, Recording};
use crate::stt::{SttError, Transcript};

#[derive(Debug, Clone, PartialEq)]
pub struct PipelineConfig {
    pub hands_free_double_tap: bool,
    pub tap_max: Duration,
    /// Measured from the first tap's release to the second tap's key-down.
    pub tap_pair_window: Duration,
    /// Keep at or above `tap_max`, so a genuine short utterance never waits out the tap
    /// window.
    pub min_press: Duration,
    pub max_recording: Duration,
    pub history_len: usize,
    pub apps: AppPolicies,
}

impl Default for PipelineConfig {
    fn default() -> Self {
        Self {
            hands_free_double_tap: true,
            tap_max: Duration::from_millis(250),
            tap_pair_window: Duration::from_millis(250),
            min_press: Duration::from_millis(250),
            max_recording: Duration::from_secs(120),
            history_len: 10,
            apps: AppPolicies::default(),
        }
    }
}

impl PipelineConfig {
    pub fn from_config(c: &Config) -> Self {
        Self {
            hands_free_double_tap: c.hands_free_double_tap,
            max_recording: c.max_recording,
            history_len: c.history_len,
            apps: c.app_policies(),
            ..Self::default()
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum State {
    Idle,
    Recording,
    Transcribing,
    Normalizing,
    Inserting,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Stage {
    Recording,
    Transcribing,
    Normalizing,
    Inserting,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Timer {
    TapWindow,
    MaxDuration,
}

#[derive(Debug)]
pub enum Failure {
    Record(RecorderError),
    Stt(SttError),
    Normalize(NormalizeError),
    Insert(InsertError),
}

impl Failure {
    pub fn stage(&self) -> Stage {
        match self {
            Self::Record(_) => Stage::Recording,
            Self::Stt(_) => Stage::Transcribing,
            Self::Normalize(_) => Stage::Normalizing,
            Self::Insert(_) => Stage::Inserting,
        }
    }
}

impl std::fmt::Display for Failure {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Record(e) => write!(f, "recording failed: {e}"),
            Self::Stt(e) => write!(f, "transcription failed: {e}"),
            Self::Normalize(e) => write!(f, "cleanup failed: {e}"),
            Self::Insert(e) => write!(f, "insertion failed: {e}"),
        }
    }
}

#[derive(Debug)]
pub enum Event {
    /// The target is where the cursor was when the user started speaking, not where it is
    /// when the text is ready.
    HotkeyDown {
        at: Instant,
        focus: FocusContext,
    },
    HotkeyUp {
        at: Instant,
    },
    Escape,
    MaxDurationReached(UtteranceId),
    TapWindowElapsed(UtteranceId),
    Recorded(UtteranceId, Recording),
    TranscriptReady(UtteranceId, Transcript),
    NormalizedReady(UtteranceId, NormalizeOutput),
    InsertDone(UtteranceId, InsertOutcome),
    Failed(UtteranceId, Failure),
}

#[derive(Debug, Clone, PartialEq)]
pub struct NormalizeContext {
    pub app: AppContext,
    pub language: Option<String>,
    pub previous: Option<String>,
    /// Set on the retry after a failure, so it cannot fail the same way.
    pub rules_only: bool,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Effect {
    StartRecording(UtteranceId),
    StopRecording(UtteranceId),
    /// No `Recorded` event follows.
    DiscardRecording(UtteranceId),
    /// Never disarmed: a timer for an utterance the machine has moved past is ignored by
    /// id when it fires.
    ArmTimer {
        id: UtteranceId,
        timer: Timer,
        after: Duration,
    },
    Transcribe {
        id: UtteranceId,
        pcm: Vec<f32>,
    },
    Normalize {
        id: UtteranceId,
        transcript: String,
        ctx: NormalizeContext,
    },
    Insert {
        id: UtteranceId,
        text: String,
        target: FocusContext,
    },
    /// A result that still arrives afterwards is ignored.
    CancelInflight(UtteranceId),
    Notify(OverlayState),
    Play(Sound),
}

#[derive(Debug, Clone)]
enum Capture {
    Idle,
    Holding {
        id: UtteranceId,
        down_at: Instant,
        focus: FocusContext,
    },
    TapPending {
        id: UtteranceId,
        down_at: Instant,
        up_at: Instant,
        focus: FocusContext,
    },
    HandsFree {
        id: UtteranceId,
        focus: FocusContext,
    },
}

impl Capture {
    fn id(&self) -> Option<UtteranceId> {
        match self {
            Self::Idle => None,
            Self::Holding { id, .. } | Self::TapPending { id, .. } | Self::HandsFree { id, .. } => {
                Some(*id)
            }
        }
    }
}

#[derive(Debug)]
struct Waiting {
    id: UtteranceId,
    focus: FocusContext,
}

#[derive(Debug)]
struct Queued {
    id: UtteranceId,
    focus: FocusContext,
    pcm: Vec<f32>,
}

#[derive(Debug)]
struct Inflight {
    id: UtteranceId,
    focus: FocusContext,
    stage: Stage,
    raw: String,
    language: Option<String>,
    rules_only: bool,
    llm_error: Option<String>,
    hint: ProvenanceHint,
}

pub struct Pipeline {
    cfg: PipelineConfig,
    next_id: UtteranceId,
    capture: Capture,
    awaiting_recorded: Vec<Waiting>,
    queue: VecDeque<Queued>,
    inflight: Option<Inflight>,
    history: History,
    shown: OverlayState,
}

/// Takes the field rather than the pipeline, so the caller can still read the other fields.
fn matching(
    inflight: &mut Option<Inflight>,
    id: UtteranceId,
    stage: Stage,
) -> Option<&mut Inflight> {
    inflight.as_mut().filter(|i| i.id == id && i.stage == stage)
}

impl Pipeline {
    pub fn new(cfg: PipelineConfig) -> Self {
        Self {
            history: History::new(cfg.history_len),
            cfg,
            next_id: UtteranceId::FIRST,
            capture: Capture::Idle,
            awaiting_recorded: Vec::new(),
            queue: VecDeque::new(),
            inflight: None,
            shown: OverlayState::Idle,
        }
    }

    pub fn state(&self) -> State {
        match &self.inflight {
            Some(i) => match i.stage {
                Stage::Recording | Stage::Transcribing => State::Transcribing,
                Stage::Normalizing => State::Normalizing,
                Stage::Inserting => State::Inserting,
            },
            None if self.is_recording() => State::Recording,
            None if !self.awaiting_recorded.is_empty() || !self.queue.is_empty() => {
                State::Transcribing
            }
            None => State::Idle,
        }
    }

    pub fn is_recording(&self) -> bool {
        !matches!(self.capture, Capture::Idle)
    }

    /// Escape is swallowed only while this holds, so an idle app never steals it.
    pub fn is_active(&self) -> bool {
        self.state() != State::Idle
    }

    pub fn history(&self) -> &History {
        &self.history
    }

    pub fn last(&self) -> Option<&HistoryEntry> {
        self.history.last()
    }

    pub fn handle(&mut self, event: Event) -> Vec<Effect> {
        let mut fx = Vec::new();
        match event {
            Event::HotkeyDown { at, focus } => self.on_down(at, focus, &mut fx),
            Event::HotkeyUp { at } => self.on_up(at, &mut fx),
            Event::Escape => self.on_escape(&mut fx),
            Event::MaxDurationReached(id) => self.on_max_duration(id, &mut fx),
            Event::TapWindowElapsed(id) => {
                if matches!(self.capture, Capture::TapPending { id: c, .. } if c == id) {
                    self.resolve_lone_tap(&mut fx);
                }
            }
            Event::Recorded(id, rec) => self.on_recorded(id, rec, &mut fx),
            Event::TranscriptReady(id, t) => self.on_transcript(id, t, &mut fx),
            Event::NormalizedReady(id, out) => self.on_normalized(id, out, &mut fx),
            Event::InsertDone(id, outcome) => self.on_insert_done(id, outcome, &mut fx),
            Event::Failed(id, failure) => self.on_failed(id, failure, &mut fx),
        }
        fx
    }

    fn notify(&mut self, fx: &mut Vec<Effect>, state: OverlayState) {
        if self.shown != state {
            self.shown = state.clone();
            fx.push(Effect::Notify(state));
        }
    }

    /// An active recording matters more to the user than a stage change behind it.
    fn show_stage(&mut self, fx: &mut Vec<Effect>, state: OverlayState) {
        if !self.is_recording() {
            self.notify(fx, state);
        }
    }

    fn inflight_overlay(&self) -> OverlayState {
        match self.inflight.as_ref().map(|i| i.stage) {
            Some(Stage::Normalizing) => OverlayState::Normalizing,
            Some(Stage::Inserting) => OverlayState::Inserting,
            Some(_) => OverlayState::Transcribing,
            None if !self.awaiting_recorded.is_empty() || !self.queue.is_empty() => {
                OverlayState::Transcribing
            }
            None => OverlayState::Idle,
        }
    }

    fn inflight_mut(&mut self, id: UtteranceId, stage: Stage) -> Option<&mut Inflight> {
        matching(&mut self.inflight, id, stage)
    }

    fn on_down(&mut self, at: Instant, focus: FocusContext, fx: &mut Vec<Effect>) {
        match self.capture.clone() {
            Capture::Idle => self.start_capture(at, focus, fx),
            // Auto-repeat must never restart a recording.
            Capture::Holding { .. } => {}
            Capture::TapPending {
                id,
                up_at,
                focus: first,
                ..
            } => {
                if at.saturating_duration_since(up_at) <= self.cfg.tap_pair_window {
                    // The taps themselves are not speech; hands-free audio starts now.
                    fx.push(Effect::DiscardRecording(id));
                    fx.push(Effect::StartRecording(id));
                    self.capture = Capture::HandsFree { id, focus: first };
                } else {
                    // The window closed but its timer has not fired yet.
                    self.resolve_lone_tap(fx);
                    self.start_capture(at, focus, fx);
                }
            }
            Capture::HandsFree { id, focus } => self.stop_capture(id, focus, fx),
        }
    }

    fn on_up(&mut self, at: Instant, fx: &mut Vec<Effect>) {
        // Any other key-up is the second tap of a pair or the tap that ended hands-free.
        let Capture::Holding { id, down_at, focus } = self.capture.clone() else {
            return;
        };
        let held = at.saturating_duration_since(down_at);
        if self.cfg.hands_free_double_tap && held < self.cfg.tap_max {
            self.capture = Capture::TapPending {
                id,
                down_at,
                up_at: at,
                focus,
            };
            fx.push(Effect::ArmTimer {
                id,
                timer: Timer::TapWindow,
                after: self.cfg.tap_pair_window,
            });
        } else if held < self.cfg.min_press {
            self.discard_capture(id, fx);
        } else {
            self.stop_capture(id, focus, fx);
        }
    }

    fn on_max_duration(&mut self, id: UtteranceId, fx: &mut Vec<Effect>) {
        match self.capture.clone() {
            Capture::Holding { id: c, focus, .. } | Capture::HandsFree { id: c, focus }
                if c == id =>
            {
                self.stop_capture(id, focus, fx)
            }
            _ => {}
        }
    }

    fn start_capture(&mut self, at: Instant, focus: FocusContext, fx: &mut Vec<Effect>) {
        let id = self.next_id;
        self.next_id = id.next();
        // The cue plays before the microphone opens so it is not recorded.
        fx.push(Effect::Play(Sound::Start));
        fx.push(Effect::StartRecording(id));
        fx.push(Effect::ArmTimer {
            id,
            timer: Timer::MaxDuration,
            after: self.cfg.max_recording,
        });
        self.capture = Capture::Holding {
            id,
            down_at: at,
            focus,
        };
        self.notify(fx, OverlayState::Listening { level: 0.0 });
    }

    fn resolve_lone_tap(&mut self, fx: &mut Vec<Effect>) {
        let Capture::TapPending {
            id,
            down_at,
            up_at,
            focus,
        } = self.capture.clone()
        else {
            return;
        };
        if up_at.saturating_duration_since(down_at) < self.cfg.min_press {
            self.discard_capture(id, fx);
        } else {
            self.stop_capture(id, focus, fx);
        }
    }

    fn discard_capture(&mut self, id: UtteranceId, fx: &mut Vec<Effect>) {
        fx.push(Effect::DiscardRecording(id));
        self.capture = Capture::Idle;
        let s = self.inflight_overlay();
        self.notify(fx, s);
    }

    fn stop_capture(&mut self, id: UtteranceId, focus: FocusContext, fx: &mut Vec<Effect>) {
        fx.push(Effect::Play(Sound::Stop));
        fx.push(Effect::StopRecording(id));
        self.capture = Capture::Idle;
        self.awaiting_recorded.push(Waiting { id, focus });
        let s = self.inflight_overlay();
        self.notify(fx, s);
    }

    fn on_recorded(&mut self, id: UtteranceId, rec: Recording, fx: &mut Vec<Effect>) {
        let Some(pos) = self.awaiting_recorded.iter().position(|w| w.id == id) else {
            return;
        };
        let Waiting { id, focus } = self.awaiting_recorded.remove(pos);
        if rec.pcm.is_empty() {
            let s = self.inflight_overlay();
            self.show_stage(fx, s);
            return;
        }
        self.queue.push_back(Queued {
            id,
            focus,
            pcm: rec.pcm,
        });
        self.pump(fx);
    }

    fn pump(&mut self, fx: &mut Vec<Effect>) {
        if self.inflight.is_some() {
            return;
        }
        let Some(q) = self.queue.pop_front() else {
            return;
        };
        self.inflight = Some(Inflight {
            id: q.id,
            focus: q.focus,
            stage: Stage::Transcribing,
            raw: String::new(),
            language: None,
            rules_only: false,
            llm_error: None,
            hint: ProvenanceHint::Rules,
        });
        fx.push(Effect::Transcribe {
            id: q.id,
            pcm: q.pcm,
        });
        self.show_stage(fx, OverlayState::Transcribing);
    }

    fn normalize_effect(cfg: &PipelineConfig, history: &History, i: &Inflight) -> Effect {
        let exe = i.focus.exe.clone();
        let previous = history
            .last()
            .filter(|e| {
                e.exe.is_some()
                    && e.exe == exe
                    && e.outcome.as_ref().is_some_and(InsertOutcome::delivered)
            })
            .map(|e| e.text.clone());
        Effect::Normalize {
            id: i.id,
            transcript: i.raw.clone(),
            ctx: NormalizeContext {
                app: AppContext {
                    style: cfg.apps.lookup(exe.as_deref()).style,
                    exe,
                    window_title: i.focus.title.clone(),
                },
                language: i.language.clone(),
                previous,
                rules_only: i.rules_only,
            },
        }
    }

    fn on_transcript(&mut self, id: UtteranceId, t: Transcript, fx: &mut Vec<Effect>) {
        let Some(i) = matching(&mut self.inflight, id, Stage::Transcribing) else {
            return;
        };
        if t.text.trim().is_empty() {
            self.finish(fx);
            return;
        }
        i.raw = t.text;
        i.language = t.language;
        i.stage = Stage::Normalizing;
        fx.push(Self::normalize_effect(&self.cfg, &self.history, i));
        self.show_stage(fx, OverlayState::Normalizing);
    }

    fn on_normalized(&mut self, id: UtteranceId, out: NormalizeOutput, fx: &mut Vec<Effect>) {
        let Some(i) = self.inflight_mut(id, Stage::Normalizing) else {
            return;
        };
        let provenance = match (&i.llm_error, out.provenance) {
            (Some(error), Provenance::Rules) => Provenance::LlmFailed {
                stage: "normalizer".into(),
                error: error.clone(),
            },
            (_, p) => p,
        };
        self.begin_insert(out.text, provenance, fx);
    }

    fn begin_insert(&mut self, text: String, provenance: Provenance, fx: &mut Vec<Effect>) {
        let Some(i) = self.inflight.as_mut() else {
            return;
        };
        if text.trim().is_empty() {
            self.finish(fx);
            return;
        }
        i.stage = Stage::Inserting;
        i.hint = ProvenanceHint::from(&provenance);
        let (id, target, raw) = (i.id, i.focus.clone(), i.raw.clone());
        // Before the attempt, so a refused or failed insertion is still recoverable.
        self.history.push(HistoryEntry {
            id,
            raw,
            text: text.clone(),
            provenance,
            exe: target.exe.clone(),
            outcome: None,
        });
        fx.push(Effect::Insert { id, text, target });
        self.show_stage(fx, OverlayState::Inserting);
    }

    fn on_insert_done(&mut self, id: UtteranceId, outcome: InsertOutcome, fx: &mut Vec<Effect>) {
        let Some(i) = self.inflight_mut(id, Stage::Inserting) else {
            return;
        };
        let hint = i.hint;
        let message = outcome.message();
        self.history.set_outcome(id, outcome);
        match message {
            None => self.notify(
                fx,
                OverlayState::Done {
                    provenance_hint: hint,
                },
            ),
            Some(message) => {
                fx.push(Effect::Play(Sound::Error));
                self.notify(fx, OverlayState::Error { message });
            }
        }
        self.finish(fx);
    }

    fn on_failed(&mut self, id: UtteranceId, failure: Failure, fx: &mut Vec<Effect>) {
        match failure {
            Failure::Record(e) => {
                let ours = self.capture.id() == Some(id)
                    || self.awaiting_recorded.iter().any(|w| w.id == id);
                if !ours {
                    return;
                }
                if self.capture.id() == Some(id) {
                    self.capture = Capture::Idle;
                }
                self.awaiting_recorded.retain(|w| w.id != id);
                self.error(fx, Failure::Record(e).to_string());
                if self.inflight.is_none() {
                    self.pump(fx);
                }
            }
            Failure::Stt(e) => {
                if self.inflight_mut(id, Stage::Transcribing).is_none() {
                    return;
                }
                // Silence dropped before or inside the engine is not a failure to report.
                if !matches!(e, SttError::EmptyAudio) {
                    self.error(fx, Failure::Stt(e).to_string());
                }
                self.finish(fx);
            }
            Failure::Normalize(e) => {
                let Some(i) = matching(&mut self.inflight, id, Stage::Normalizing) else {
                    return;
                };
                if i.rules_only {
                    // The raw transcript beats nothing reaching the target.
                    let raw = i.raw.clone();
                    let error = i.llm_error.clone().unwrap_or_else(|| e.to_string());
                    self.begin_insert(
                        raw,
                        Provenance::LlmFailed {
                            stage: "rules".into(),
                            error,
                        },
                        fx,
                    );
                } else {
                    tracing::info!(%id, error = %e, "normalizer failed; retrying with rules only");
                    i.rules_only = true;
                    i.llm_error = Some(e.to_string());
                    fx.push(Self::normalize_effect(&self.cfg, &self.history, i));
                }
            }
            Failure::Insert(e) => {
                if self.inflight_mut(id, Stage::Inserting).is_none() {
                    return;
                }
                self.error(
                    fx,
                    format!("Could not insert ({e}); use \"paste last\" from the tray"),
                );
                self.finish(fx);
            }
        }
    }

    fn error(&mut self, fx: &mut Vec<Effect>, message: String) {
        fx.push(Effect::Play(Sound::Error));
        self.notify(fx, OverlayState::Error { message });
    }

    fn finish(&mut self, fx: &mut Vec<Effect>) {
        self.inflight = None;
        self.pump(fx);
        if self.is_recording() {
            self.notify(fx, OverlayState::Listening { level: 0.0 });
        } else if self.inflight.is_none() && self.awaiting_recorded.is_empty() {
            // Done and Error stay up until the overlay fades them itself.
            if !matches!(
                self.shown,
                OverlayState::Done { .. } | OverlayState::Error { .. }
            ) {
                self.notify(fx, OverlayState::Idle);
            }
        }
    }

    fn on_escape(&mut self, fx: &mut Vec<Effect>) {
        let mut any = false;
        if let Some(id) = self.capture.id() {
            fx.push(Effect::DiscardRecording(id));
            self.capture = Capture::Idle;
            any = true;
        }
        any |= !self.awaiting_recorded.is_empty() || !self.queue.is_empty();
        self.awaiting_recorded.clear();
        self.queue.clear();
        if let Some(i) = self.inflight.take() {
            fx.push(Effect::CancelInflight(i.id));
            any = true;
        }
        if any {
            fx.push(Effect::Play(Sound::Cancel));
            self.notify(fx, OverlayState::Idle);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::insert::{AppPolicy, RefuseReason};
    use crate::normalize::{Rejection, Scores, Style};

    const MAX: Duration = Duration::from_secs(120);

    struct T {
        p: Pipeline,
        t0: Instant,
    }

    impl T {
        fn new() -> Self {
            Self::with(PipelineConfig::default())
        }

        fn with(cfg: PipelineConfig) -> Self {
            Self {
                p: Pipeline::new(cfg),
                t0: Instant::now(),
            }
        }

        fn at(&self, ms: u64) -> Instant {
            self.t0 + Duration::from_millis(ms)
        }

        fn down(&mut self, ms: u64) -> Vec<Effect> {
            let at = self.at(ms);
            self.p.handle(Event::HotkeyDown {
                at,
                focus: notepad(),
            })
        }

        fn up(&mut self, ms: u64) -> Vec<Effect> {
            let at = self.at(ms);
            self.p.handle(Event::HotkeyUp { at })
        }

        fn recorded(&mut self, id: u64) -> Vec<Effect> {
            self.p.handle(Event::Recorded(UtteranceId(id), recording()))
        }

        fn transcript(&mut self, id: u64, text: &str) -> Vec<Effect> {
            self.p.handle(Event::TranscriptReady(
                UtteranceId(id),
                Transcript {
                    utterance: UtteranceId(id),
                    text: text.into(),
                    language: Some("en".into()),
                    ..Default::default()
                },
            ))
        }

        fn normalized(&mut self, id: u64, text: &str, provenance: Provenance) -> Vec<Effect> {
            self.p.handle(Event::NormalizedReady(
                UtteranceId(id),
                NormalizeOutput {
                    utterance: UtteranceId(id),
                    text: text.into(),
                    provenance,
                    elapsed: Duration::ZERO,
                },
            ))
        }

        fn inserted(&mut self, id: u64, outcome: InsertOutcome) -> Vec<Effect> {
            self.p.handle(Event::InsertDone(UtteranceId(id), outcome))
        }

        fn reach_transcribing(&mut self) {
            self.down(0);
            self.up(1000);
            self.recorded(1);
            assert_eq!(self.p.state(), State::Transcribing);
        }

        fn reach_normalizing(&mut self) {
            self.reach_transcribing();
            self.transcript(1, "um hello there");
            assert_eq!(self.p.state(), State::Normalizing);
        }

        fn reach_inserting(&mut self) {
            self.reach_normalizing();
            self.normalized(1, "Hello there.", Provenance::Rules);
            assert_eq!(self.p.state(), State::Inserting);
        }
    }

    fn notepad() -> FocusContext {
        FocusContext {
            window: 42,
            exe: Some("notepad.exe".into()),
            title: Some("Untitled - Notepad".into()),
            ..Default::default()
        }
    }

    fn recording() -> Recording {
        Recording {
            pcm: vec![0.1; 16_000],
            sample_rate: 16_000,
            duration: Duration::from_secs(1),
            ..Default::default()
        }
    }

    fn id(n: u64) -> UtteranceId {
        UtteranceId(n)
    }

    fn has(fx: &[Effect], pred: impl Fn(&Effect) -> bool) -> bool {
        fx.iter().any(pred)
    }

    #[test]
    fn happy_path() {
        let mut t = T::new();
        assert_eq!(
            t.down(0),
            vec![
                Effect::Play(Sound::Start),
                Effect::StartRecording(id(1)),
                Effect::ArmTimer {
                    id: id(1),
                    timer: Timer::MaxDuration,
                    after: MAX
                },
                Effect::Notify(OverlayState::Listening { level: 0.0 }),
            ]
        );
        assert_eq!(t.p.state(), State::Recording);
        assert_eq!(
            t.up(1500),
            vec![
                Effect::Play(Sound::Stop),
                Effect::StopRecording(id(1)),
                Effect::Notify(OverlayState::Transcribing),
            ]
        );
        assert_eq!(
            t.recorded(1),
            vec![Effect::Transcribe {
                id: id(1),
                pcm: recording().pcm
            }]
        );
        assert_eq!(
            t.transcript(1, "um hello there"),
            vec![
                Effect::Normalize {
                    id: id(1),
                    transcript: "um hello there".into(),
                    ctx: NormalizeContext {
                        app: AppContext {
                            exe: Some("notepad.exe".into()),
                            window_title: Some("Untitled - Notepad".into()),
                            style: Style::Casual,
                        },
                        language: Some("en".into()),
                        previous: None,
                        rules_only: false,
                    },
                },
                Effect::Notify(OverlayState::Normalizing),
            ]
        );
        let llm = Provenance::Llm {
            model: "m".into(),
            scores: Scores::default(),
        };
        assert_eq!(
            t.normalized(1, "Hello there.", llm.clone()),
            vec![
                Effect::Insert {
                    id: id(1),
                    text: "Hello there.".into(),
                    target: notepad()
                },
                Effect::Notify(OverlayState::Inserting),
            ]
        );
        assert_eq!(
            t.inserted(1, InsertOutcome::TargetRead),
            vec![Effect::Notify(OverlayState::Done {
                provenance_hint: ProvenanceHint::Llm
            })]
        );
        assert_eq!(t.p.state(), State::Idle);
        let last = t.p.last().unwrap();
        assert_eq!(last.raw, "um hello there");
        assert_eq!(last.text, "Hello there.");
        assert_eq!(last.provenance, llm);
        assert_eq!(last.outcome, Some(InsertOutcome::TargetRead));
    }

    #[test]
    fn previous_sentence_goes_to_the_same_app_only() {
        let mut t = T::new();
        t.reach_inserting();
        t.inserted(1, InsertOutcome::Typed);
        t.down(5000);
        t.up(6000);
        t.recorded(2);
        let fx = t.transcript(2, "and more");
        let Some(Effect::Normalize { ctx, .. }) = fx.first() else {
            panic!("{fx:?}");
        };
        assert_eq!(ctx.previous.as_deref(), Some("Hello there."));
    }

    #[test]
    fn escape_while_recording_discards() {
        let mut t = T::new();
        t.down(0);
        let fx = t.p.handle(Event::Escape);
        assert_eq!(
            fx,
            vec![
                Effect::DiscardRecording(id(1)),
                Effect::Play(Sound::Cancel),
                Effect::Notify(OverlayState::Idle),
            ]
        );
        assert_eq!(t.p.state(), State::Idle);
        assert!(
            t.up(1000).is_empty(),
            "the key-up of a cancelled press is inert"
        );
        assert!(t.recorded(1).is_empty());
    }

    #[test]
    fn escape_between_stop_and_recorded_drops_the_audio() {
        let mut t = T::new();
        t.down(0);
        t.up(1000);
        let fx = t.p.handle(Event::Escape);
        assert!(has(&fx, |e| *e == Effect::Play(Sound::Cancel)));
        assert!(t.recorded(1).is_empty());
        assert_eq!(t.p.state(), State::Idle);
    }

    #[test]
    fn escape_during_each_processing_stage_cancels_and_ignores_the_late_result() {
        type Setup = fn(&mut T);
        type Late = fn(&mut T) -> Vec<Effect>;
        let stages: [(Setup, Late); 3] = [
            (T::reach_transcribing, |t| t.transcript(1, "late")),
            (T::reach_normalizing, |t| {
                t.normalized(1, "Late.", Provenance::Rules)
            }),
            (T::reach_inserting, |t| t.inserted(1, InsertOutcome::Typed)),
        ];
        for (setup, late) in stages {
            let mut t = T::new();
            setup(&mut t);
            let fx = t.p.handle(Event::Escape);
            assert_eq!(
                fx,
                vec![
                    Effect::CancelInflight(id(1)),
                    Effect::Play(Sound::Cancel),
                    Effect::Notify(OverlayState::Idle),
                ]
            );
            assert_eq!(t.p.state(), State::Idle);
            assert!(late(&mut t).is_empty());
            assert!(!t.p.is_active());
            assert!(
                t.p.handle(Event::Escape).is_empty(),
                "idle Escape is not ours"
            );
        }
    }

    #[test]
    fn press_during_transcribing_queues_the_next_recording() {
        let mut t = T::new();
        t.reach_transcribing();
        let fx = t.down(1200);
        assert!(has(&fx, |e| *e == Effect::StartRecording(id(2))));
        assert_eq!(t.p.state(), State::Transcribing);
        t.up(2500);
        assert!(
            t.recorded(2).is_empty(),
            "second utterance waits for the first"
        );

        t.transcript(1, "first");
        t.normalized(1, "First.", Provenance::Rules);
        let fx = t.inserted(1, InsertOutcome::TargetRead);
        assert!(has(
            &fx,
            |e| matches!(e, Effect::Transcribe { id: i, .. } if *i == id(2))
        ));
        assert_eq!(t.p.state(), State::Transcribing);
        t.transcript(2, "second");
        t.normalized(2, "Second.", Provenance::Rules);
        t.inserted(2, InsertOutcome::Typed);
        assert_eq!(t.p.state(), State::Idle);
        assert_eq!(t.p.history().len(), 2);
    }

    #[test]
    fn stage_notifications_do_not_cover_an_active_recording() {
        let mut t = T::new();
        t.reach_transcribing();
        t.down(1200);
        let fx = t.transcript(1, "first");
        assert!(!has(&fx, |e| matches!(e, Effect::Notify(_))), "{fx:?}");
    }

    #[test]
    fn stale_results_are_ignored() {
        let mut t = T::new();
        t.reach_transcribing();
        assert!(t.transcript(99, "stale").is_empty());
        assert!(
            t.normalized(1, "Wrong stage.", Provenance::Rules)
                .is_empty()
        );
        assert!(t.inserted(1, InsertOutcome::Typed).is_empty());
        assert!(
            t.p.handle(Event::MaxDurationReached(id(1))).is_empty(),
            "timer of a finished capture"
        );
        assert!(
            t.p.handle(Event::Failed(id(7), Failure::Stt(SttError::Cancelled)))
                .is_empty()
        );
        assert_eq!(t.p.state(), State::Transcribing);
    }

    #[test]
    fn tap_pair_enters_hands_free_and_a_tap_stops_it() {
        let mut t = T::new();
        t.down(0);
        assert_eq!(
            t.up(80),
            vec![Effect::ArmTimer {
                id: id(1),
                timer: Timer::TapWindow,
                after: Duration::from_millis(250)
            }]
        );
        assert_eq!(
            t.down(200),
            vec![
                Effect::DiscardRecording(id(1)),
                Effect::StartRecording(id(1))
            ]
        );
        assert!(t.up(260).is_empty());
        assert!(
            t.p.handle(Event::TapWindowElapsed(id(1))).is_empty(),
            "late tap timer"
        );
        assert_eq!(t.p.state(), State::Recording);

        assert_eq!(
            t.down(9000),
            vec![
                Effect::Play(Sound::Stop),
                Effect::StopRecording(id(1)),
                Effect::Notify(OverlayState::Transcribing),
            ]
        );
        assert!(t.up(9080).is_empty());
        assert!(has(&t.recorded(1), |e| matches!(
            e,
            Effect::Transcribe { .. }
        )));
    }

    #[test]
    fn lone_short_tap_is_discarded_when_the_window_closes() {
        let mut t = T::new();
        t.down(0);
        t.up(80);
        assert_eq!(t.p.state(), State::Recording);
        assert_eq!(
            t.p.handle(Event::TapWindowElapsed(id(1))),
            vec![
                Effect::DiscardRecording(id(1)),
                Effect::Notify(OverlayState::Idle)
            ]
        );
        assert_eq!(t.p.state(), State::Idle);
    }

    #[test]
    fn late_second_press_is_a_new_recording() {
        let mut t = T::new();
        t.down(0);
        t.up(80);
        let fx = t.down(700);
        assert_eq!(fx[0], Effect::DiscardRecording(id(1)));
        assert!(has(&fx, |e| *e == Effect::StartRecording(id(2))));
    }

    #[test]
    fn without_double_tap_a_short_press_is_discarded_at_once() {
        let mut t = T::with(PipelineConfig {
            hands_free_double_tap: false,
            ..Default::default()
        });
        t.down(0);
        assert_eq!(
            t.up(80),
            vec![
                Effect::DiscardRecording(id(1)),
                Effect::Notify(OverlayState::Idle)
            ]
        );
    }

    #[test]
    fn max_duration_stops_the_recording() {
        let mut t = T::new();
        t.down(0);
        assert_eq!(
            t.p.handle(Event::MaxDurationReached(id(1))),
            vec![
                Effect::Play(Sound::Stop),
                Effect::StopRecording(id(1)),
                Effect::Notify(OverlayState::Transcribing),
            ]
        );
        assert!(
            t.up(130_000).is_empty(),
            "the key-up that never came, arriving late"
        );
        assert!(has(&t.recorded(1), |e| matches!(
            e,
            Effect::Transcribe { .. }
        )));
    }

    #[test]
    fn max_duration_also_ends_hands_free() {
        let mut t = T::new();
        t.down(0);
        t.up(80);
        t.down(200);
        let fx = t.p.handle(Event::MaxDurationReached(id(1)));
        assert!(has(&fx, |e| *e == Effect::StopRecording(id(1))));
    }

    #[test]
    fn llm_failure_falls_back_to_the_rule_pass() {
        let mut t = T::new();
        t.reach_normalizing();
        let fx = t.p.handle(Event::Failed(
            id(1),
            Failure::Normalize(NormalizeError::Deadline),
        ));
        let [
            Effect::Normalize {
                ctx, transcript, ..
            },
        ] = fx.as_slice()
        else {
            panic!("{fx:?}");
        };
        assert!(ctx.rules_only);
        assert_eq!(transcript, "um hello there");

        let fx = t.normalized(1, "Hello there.", Provenance::Rules);
        assert!(has(
            &fx,
            |e| matches!(e, Effect::Insert { text, .. } if text == "Hello there.")
        ));
        assert!(matches!(
            t.p.last().unwrap().provenance,
            Provenance::LlmFailed { .. }
        ));
        let fx = t.inserted(1, InsertOutcome::TargetRead);
        assert_eq!(
            fx,
            vec![Effect::Notify(OverlayState::Done {
                provenance_hint: ProvenanceHint::LlmFallback
            })]
        );
    }

    #[test]
    fn failure_of_the_rule_retry_inserts_the_raw_transcript() {
        let mut t = T::new();
        t.reach_normalizing();
        t.p.handle(Event::Failed(
            id(1),
            Failure::Normalize(NormalizeError::Unavailable("down".into())),
        ));
        let fx = t.p.handle(Event::Failed(
            id(1),
            Failure::Normalize(NormalizeError::BackendDied("gone".into())),
        ));
        assert!(has(
            &fx,
            |e| matches!(e, Effect::Insert { text, .. } if text == "um hello there")
        ));
    }

    #[test]
    fn rejected_llm_output_shows_the_fallback_hint() {
        let mut t = T::new();
        t.reach_normalizing();
        t.normalized(
            1,
            "Hello there.",
            Provenance::LlmRejected {
                model: "m".into(),
                rejection: Rejection::Empty,
            },
        );
        let fx = t.inserted(1, InsertOutcome::Typed);
        assert_eq!(
            fx,
            vec![Effect::Notify(OverlayState::Done {
                provenance_hint: ProvenanceHint::LlmFallback
            })]
        );
    }

    #[test]
    fn refused_insert_still_lands_in_history() {
        let mut t = T::new();
        t.reach_inserting();
        let fx = t.inserted(
            1,
            InsertOutcome::Refused {
                reason: RefuseReason::Elevated,
                on_clipboard: true,
            },
        );
        assert_eq!(fx[0], Effect::Play(Sound::Error));
        assert!(
            matches!(&fx[1], Effect::Notify(OverlayState::Error { message }) if message.contains("administrator"))
        );
        let last = t.p.last().unwrap();
        assert_eq!(last.text, "Hello there.");
        assert!(matches!(last.outcome, Some(InsertOutcome::Refused { .. })));
        assert_eq!(t.p.state(), State::Idle);
    }

    #[test]
    fn insert_error_keeps_the_text_in_history() {
        let mut t = T::new();
        t.reach_inserting();
        let fx = t.p.handle(Event::Failed(
            id(1),
            Failure::Insert(InsertError::Clipboard("locked".into())),
        ));
        assert_eq!(fx[0], Effect::Play(Sound::Error));
        assert_eq!(t.p.last().unwrap().text, "Hello there.");
        assert_eq!(t.p.last().unwrap().outcome, None);
        assert_eq!(t.p.state(), State::Idle);
    }

    #[test]
    fn silence_ends_quietly() {
        let mut t = T::new();
        t.reach_transcribing();
        let fx = t.transcript(1, "  ");
        assert_eq!(fx, vec![Effect::Notify(OverlayState::Idle)]);
        let mut t = T::new();
        t.reach_transcribing();
        let fx =
            t.p.handle(Event::Failed(id(1), Failure::Stt(SttError::EmptyAudio)));
        assert_eq!(fx, vec![Effect::Notify(OverlayState::Idle)]);
        assert!(t.p.history().is_empty());
    }

    #[test]
    fn transcription_error_is_reported() {
        let mut t = T::new();
        t.reach_transcribing();
        let fx = t.p.handle(Event::Failed(
            id(1),
            Failure::Stt(SttError::BackendDied("device lost".into())),
        ));
        assert_eq!(fx[0], Effect::Play(Sound::Error));
        assert_eq!(t.p.state(), State::Idle);
    }

    #[test]
    fn recorder_failure_resets_capture() {
        let mut t = T::new();
        t.down(0);
        let fx = t.p.handle(Event::Failed(
            id(1),
            Failure::Record(RecorderError::NoInputDevice),
        ));
        assert_eq!(fx[0], Effect::Play(Sound::Error));
        assert_eq!(t.p.state(), State::Idle);
        assert!(has(&t.down(3000), |e| *e == Effect::StartRecording(id(2))));
    }

    #[test]
    fn app_style_reaches_the_normalizer() {
        let mut cfg = PipelineConfig::default();
        cfg.apps.by_exe.push((
            "notepad.exe".into(),
            AppPolicy {
                style: Style::Formal,
                ..Default::default()
            },
        ));
        let mut t = T::with(cfg);
        t.reach_transcribing();
        let fx = t.transcript(1, "hi there");
        let Some(Effect::Normalize { ctx, .. }) = fx.first() else {
            panic!("{fx:?}");
        };
        assert_eq!(ctx.app.style, Style::Formal);
    }

    #[test]
    fn from_config_carries_user_settings() {
        let c = Config {
            hands_free_double_tap: false,
            max_recording: Duration::from_secs(30),
            ..Config::default()
        };
        let p = PipelineConfig::from_config(&c);
        assert!(!p.hands_free_double_tap);
        assert_eq!(p.max_recording, Duration::from_secs(30));
        assert_eq!(p.apps, c.app_policies());
    }
}
