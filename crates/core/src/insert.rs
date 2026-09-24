//! Insertion contract and the D8 strategy chain as pure logic.
//!
//! The chain decides chord, reads the clipboard receipt, picks the fallback and the
//! restore delay. The Win32 calls sit behind three small ports implemented in the
//! platform crate, so every branch here runs in unit tests against fakes.

use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::cancel::CancelToken;
use crate::context::FocusContext;
use crate::normalize::Style;

/// Paste chord sent to the target. Per app because conhost ignores Ctrl+Shift+V and some
/// terminals bind Ctrl+V to something else.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Chord {
    #[default]
    CtrlV,
    CtrlShiftV,
    ShiftInsert,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RefuseReason {
    /// Target runs at a higher integrity level; injected input is dropped by the OS.
    Elevated,
    Password,
    /// The window captured at hotkey-down is no longer the foreground.
    FocusChanged,
    /// Remote desktop, Citrix or WSL client window: typing would land in the wrong
    /// session and the paste receipt is meaningless there.
    UnsupportedRemote,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum LeaveReason {
    /// Nothing read the clipboard after the chord and this app may not be typed into.
    NeverType,
    ChordFailed(String),
    TypingFailed(String),
}

/// The D8 receipt, and what the chain finally did.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum InsertOutcome {
    /// The render came after the chord from the target's own process.
    TargetRead,
    /// Something else read first (before the chord, or another process after it); the
    /// target's own read is invisible. Most likely inserted, not confirmed.
    ThirdPartyRead {
        reader: Option<String>,
    },
    Typed,
    /// Nothing read the clipboard within the bound. A classification the chain turns
    /// into `Typed` or `LeftOnClipboard`; it never returns it.
    NotRequested,
    /// Someone else wrote the clipboard meanwhile. Not restored: that write is theirs.
    ClipboardChanged,
    LeftOnClipboard {
        reason: LeaveReason,
    },
    /// Not attempted. `on_clipboard` says whether the text was left there instead.
    Refused {
        reason: RefuseReason,
        on_clipboard: bool,
    },
}

impl InsertOutcome {
    /// The text reached the target, as far as anything here can tell.
    pub fn delivered(&self) -> bool {
        matches!(
            self,
            Self::TargetRead | Self::ThirdPartyRead { .. } | Self::Typed
        )
    }

    /// What to tell the user when the text did not arrive. `None` when it did.
    pub fn message(&self) -> Option<String> {
        let where_ = |on_clipboard: bool| {
            if on_clipboard {
                "the text is on the clipboard"
            } else {
                "use \"paste last\" from the tray"
            }
        };
        match self {
            Self::TargetRead | Self::ThirdPartyRead { .. } | Self::Typed => None,
            Self::NotRequested => Some(format!("Not pasted; {}", where_(true))),
            Self::ClipboardChanged => Some(format!(
                "Another program changed the clipboard; {}",
                where_(false)
            )),
            Self::LeftOnClipboard { reason } => Some(match reason {
                LeaveReason::NeverType => format!("This app did not paste; {}", where_(true)),
                LeaveReason::ChordFailed(e) | LeaveReason::TypingFailed(e) => {
                    format!("Could not insert ({e}); {}", where_(true))
                }
            }),
            Self::Refused {
                reason,
                on_clipboard,
            } => {
                let why = match reason {
                    RefuseReason::Elevated => "The target runs as administrator",
                    RefuseReason::Password => "Not inserted into a password field",
                    RefuseReason::FocusChanged => "The focused window changed",
                    RefuseReason::UnsupportedRemote => "Remote windows are not supported",
                };
                Some(format!("{why}; {}", where_(*on_clipboard)))
            }
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum InsertError {
    #[error("cancelled")]
    Cancelled,
    #[error("clipboard: {0}")]
    Clipboard(String),
    #[error("input: {0}")]
    Input(String),
}

pub trait Inserter: Send {
    /// Insert `text` into `target`. Cancellation is honoured until the paste chord is
    /// sent; after that the paste is in the target's hands and is only observed.
    fn insert(
        &mut self,
        target: &FocusContext,
        text: &str,
        cancel: &CancelToken,
    ) -> Result<InsertOutcome, InsertError>;
}

/// The process that opened the clipboard to render our data.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct Reader {
    pub pid: Option<u32>,
    /// Lower-case executable file name.
    pub exe: Option<String>,
}

impl Reader {
    /// Matched on the executable: `FocusContext` carries no process id, so two instances
    /// of one program are indistinguishable here. An unknown reader never matches, which
    /// errs towards "third party": a later restore, and no typing.
    fn is_target(&self, target: &FocusContext) -> bool {
        match (&self.exe, &target.exe) {
            (Some(r), Some(t)) => r.eq_ignore_ascii_case(t),
            _ => false,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RenderWait {
    /// First render request since the last write. Windows asks the owner exactly once per
    /// write, so there is never a second one to wait for.
    Read(Reader),
    TimedOut,
    /// The sequence number moved: a foreign write replaced ours.
    Changed,
}

pub trait ClipboardPort: Send {
    type Snapshot: Send;

    /// Bounded, best-effort copy of the current contents.
    fn snapshot(&mut self) -> Result<Self::Snapshot, InsertError>;

    /// Offer `text` as delayed-rendered `CF_UNICODETEXT`, marked for exclusion from
    /// history and cloud sync. Returns the sequence number after the write.
    fn write_delayed(&mut self, text: &str) -> Result<u64, InsertError>;

    fn sequence_number(&self) -> u64;

    fn wait_for_render(&mut self, timeout: Duration) -> RenderWait;

    /// Put `snapshot` back after `after`, but only if the sequence number is still
    /// `if_sequence` then. Must not block: the ≈1 s third-party delay would otherwise sit
    /// on the release-to-text path of the next utterance.
    fn restore(&mut self, snapshot: Self::Snapshot, after: Duration, if_sequence: u64);
}

pub trait InputPort: Send {
    /// Release modifiers the user still holds, so the chord is not Ctrl+Alt+V.
    fn release_modifiers(&mut self) -> Result<(), InsertError>;
    fn send_chord(&mut self, chord: Chord) -> Result<(), InsertError>;
    /// One batch of Unicode key events.
    fn type_text(&mut self, text: &str) -> Result<(), InsertError>;
}

pub trait FocusPort: Send {
    fn foreground_window(&self) -> usize;

    /// Whether `target` is (again) the foreground window. An implementation may try to
    /// refocus it first; it must never steal focus from a window the user moved to.
    fn is_still(&mut self, target: &FocusContext) -> bool {
        target.window != 0 && self.foreground_window() == target.window
    }

    fn is_remote_session(&self) -> bool;
}

/// Per-app insertion and style settings.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AppPolicy {
    pub chord: Chord,
    /// Never fall back to typing: apps where autocomplete or Enter-to-send would mangle
    /// typed text, or that paste late and would get the text twice.
    pub never_type: bool,
    pub style: Style,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AppPolicies {
    pub default: AppPolicy,
    /// Lower-case executable name and its policy; the first match wins.
    pub by_exe: Vec<(String, AppPolicy)>,
}

impl AppPolicies {
    pub fn lookup(&self, exe: Option<&str>) -> &AppPolicy {
        exe.and_then(|e| {
            self.by_exe
                .iter()
                .find(|(x, _)| x.eq_ignore_ascii_case(e))
                .map(|(_, p)| p)
        })
        .unwrap_or(&self.default)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InsertTiming {
    /// Between the clipboard write and the chord; any read in it is a third party.
    /// D8 measured the RDP clipboard reading within 1 ms of every write.
    pub early_read_gap: Duration,
    /// How long after the chord a missing read means the chord did not paste. Targets
    /// were measured reading within 2.3 ms; the bound is for slow apps under load, and it
    /// is only ever paid in full on the typing fallback path.
    pub render_timeout: Duration,
    /// After a target read: the app already holds the data.
    pub restore_margin: Duration,
    /// After a third-party read the target's own read cannot be seen, so wait long enough
    /// for a late paste; history covers the rare miss.
    pub third_party_restore_delay: Duration,
}

impl Default for InsertTiming {
    fn default() -> Self {
        Self {
            early_read_gap: Duration::from_millis(30),
            render_timeout: Duration::from_millis(500),
            restore_margin: Duration::from_millis(100),
            third_party_restore_delay: Duration::from_secs(1),
        }
    }
}

/// Executables whose windows forward input to another machine or session.
pub const DEFAULT_UNSUPPORTED_TARGETS: &[&str] = &[
    "mstsc.exe",
    "msrdc.exe",
    "wfica32.exe",
    "cdviewer.exe",
    "vmconnect.exe",
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct InsertPolicy {
    pub timing: InsertTiming,
    pub apps: AppPolicies,
    /// Lower-case executable names refused as `UnsupportedRemote`.
    pub unsupported_targets: Vec<String>,
    /// Inside a remote session, type instead of pasting so the text never reaches the
    /// clipboard that RDP forwards to the client machine.
    pub type_only_in_remote: bool,
}

impl Default for InsertPolicy {
    fn default() -> Self {
        Self {
            timing: InsertTiming::default(),
            apps: AppPolicies::default(),
            unsupported_targets: DEFAULT_UNSUPPORTED_TARGETS
                .iter()
                .map(|s| s.to_string())
                .collect(),
            type_only_in_remote: false,
        }
    }
}

/// D8: delayed-render paste with a read receipt, typing as fallback, clipboard as the
/// last resort.
pub struct StrategyChain<C, I, F> {
    clipboard: C,
    input: I,
    focus: F,
    policy: InsertPolicy,
}

impl<C: ClipboardPort, I: InputPort, F: FocusPort> StrategyChain<C, I, F> {
    pub fn new(clipboard: C, input: I, focus: F, policy: InsertPolicy) -> Self {
        Self {
            clipboard,
            input,
            focus,
            policy,
        }
    }

    pub fn policy(&self) -> &InsertPolicy {
        &self.policy
    }

    pub fn into_parts(self) -> (C, I, F) {
        (self.clipboard, self.input, self.focus)
    }

    fn unsupported(&self, target: &FocusContext) -> bool {
        target.exe.as_deref().is_some_and(|e| {
            self.policy
                .unsupported_targets
                .iter()
                .any(|u| u.eq_ignore_ascii_case(e))
        })
    }

    fn refuse_to_clipboard(&mut self, text: &str, reason: RefuseReason) -> InsertOutcome {
        let on_clipboard = match self.clipboard.write_delayed(text) {
            Ok(_) => true,
            Err(e) => {
                tracing::warn!(error = %e, "could not leave refused text on the clipboard");
                false
            }
        };
        InsertOutcome::Refused {
            reason,
            on_clipboard,
        }
    }

    /// Restores only while our write is still the latest; a foreign write is the user's
    /// newer data and must survive.
    fn restore(&mut self, snapshot: Option<C::Snapshot>, after: Duration, ours: u64) {
        if let Some(s) = snapshot
            && self.clipboard.sequence_number() == ours
        {
            self.clipboard.restore(s, after, ours);
        }
    }

    fn type_only(
        &mut self,
        target: &FocusContext,
        text: &str,
        app: &AppPolicy,
        cancel: &CancelToken,
    ) -> Result<InsertOutcome, InsertError> {
        if app.never_type {
            return Ok(InsertOutcome::Refused {
                reason: RefuseReason::UnsupportedRemote,
                on_clipboard: false,
            });
        }
        if cancel.is_cancelled() {
            return Err(InsertError::Cancelled);
        }
        if !self.focus.is_still(target) {
            return Ok(InsertOutcome::Refused {
                reason: RefuseReason::FocusChanged,
                on_clipboard: false,
            });
        }
        self.input.release_modifiers()?;
        self.input.type_text(text)?;
        Ok(InsertOutcome::Typed)
    }

    fn paste(
        &mut self,
        target: &FocusContext,
        text: &str,
        app: &AppPolicy,
        cancel: &CancelToken,
    ) -> Result<InsertOutcome, InsertError> {
        let timing = self.policy.timing.clone();
        let snapshot = match self.clipboard.snapshot() {
            Ok(s) => Some(s),
            Err(e) => {
                tracing::warn!(error = %e, "clipboard snapshot failed; previous contents will not be restored");
                None
            }
        };
        let ours = self.clipboard.write_delayed(text)?;

        let early = match self.clipboard.wait_for_render(timing.early_read_gap) {
            RenderWait::Read(r) => Some(r),
            RenderWait::TimedOut => None,
            RenderWait::Changed => return Ok(InsertOutcome::ClipboardChanged),
        };
        if cancel.is_cancelled() {
            self.restore(snapshot, Duration::ZERO, ours);
            return Err(InsertError::Cancelled);
        }
        // Text stays on the clipboard: the user moved on, and it must not be lost.
        if !self.focus.is_still(target) {
            return Ok(InsertOutcome::Refused {
                reason: RefuseReason::FocusChanged,
                on_clipboard: true,
            });
        }
        if let Err(e) = self
            .input
            .release_modifiers()
            .and_then(|()| self.input.send_chord(app.chord))
        {
            return Ok(InsertOutcome::LeftOnClipboard {
                reason: LeaveReason::ChordFailed(e.to_string()),
            });
        }

        // After an early read Windows will not ask again, so there is nothing to wait for.
        let outcome = match early {
            Some(r) => InsertOutcome::ThirdPartyRead { reader: r.exe },
            None => match self.clipboard.wait_for_render(timing.render_timeout) {
                RenderWait::Read(r) if r.is_target(target) => InsertOutcome::TargetRead,
                RenderWait::Read(r) => InsertOutcome::ThirdPartyRead { reader: r.exe },
                RenderWait::TimedOut => InsertOutcome::NotRequested,
                RenderWait::Changed => InsertOutcome::ClipboardChanged,
            },
        };

        match outcome {
            InsertOutcome::TargetRead => {
                self.restore(snapshot, timing.restore_margin, ours);
                Ok(outcome)
            }
            InsertOutcome::ThirdPartyRead { .. } => {
                self.restore(snapshot, timing.third_party_restore_delay, ours);
                Ok(outcome)
            }
            InsertOutcome::NotRequested => {
                self.type_fallback(target, text, app, cancel, snapshot, ours)
            }
            other => Ok(other),
        }
    }

    /// Only reached on `NotRequested`: typing after any read would duplicate the text in
    /// apps that read the clipboard but paste late.
    fn type_fallback(
        &mut self,
        target: &FocusContext,
        text: &str,
        app: &AppPolicy,
        cancel: &CancelToken,
        snapshot: Option<C::Snapshot>,
        ours: u64,
    ) -> Result<InsertOutcome, InsertError> {
        if app.never_type {
            return Ok(InsertOutcome::LeftOnClipboard {
                reason: LeaveReason::NeverType,
            });
        }
        if cancel.is_cancelled() {
            self.restore(snapshot, Duration::ZERO, ours);
            return Err(InsertError::Cancelled);
        }
        if !self.focus.is_still(target) {
            return Ok(InsertOutcome::Refused {
                reason: RefuseReason::FocusChanged,
                on_clipboard: true,
            });
        }
        match self.input.type_text(text) {
            Ok(()) => {
                self.restore(snapshot, Duration::ZERO, ours);
                Ok(InsertOutcome::Typed)
            }
            Err(e) => Ok(InsertOutcome::LeftOnClipboard {
                reason: LeaveReason::TypingFailed(e.to_string()),
            }),
        }
    }
}

impl<C: ClipboardPort, I: InputPort, F: FocusPort> Inserter for StrategyChain<C, I, F> {
    fn insert(
        &mut self,
        target: &FocusContext,
        text: &str,
        cancel: &CancelToken,
    ) -> Result<InsertOutcome, InsertError> {
        if cancel.is_cancelled() {
            return Err(InsertError::Cancelled);
        }
        let app = self.policy.apps.lookup(target.exe.as_deref()).clone();
        // A dictated secret on the clipboard is readable by every process and forwarded
        // by RDP; history keeps it recoverable instead.
        if target.is_password {
            return Ok(InsertOutcome::Refused {
                reason: RefuseReason::Password,
                on_clipboard: false,
            });
        }
        if target.elevated {
            return Ok(self.refuse_to_clipboard(text, RefuseReason::Elevated));
        }
        if self.unsupported(target) {
            return Ok(self.refuse_to_clipboard(text, RefuseReason::UnsupportedRemote));
        }
        if self.policy.type_only_in_remote && self.focus.is_remote_session() {
            return self.type_only(target, text, &app, cancel);
        }
        self.paste(target, text, &app, cancel)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::VecDeque;
    use std::sync::{Arc, Mutex};

    #[derive(Debug, Clone, PartialEq)]
    enum Call {
        Snapshot,
        Write(String),
        Wait(Duration),
        Restore(Duration),
        IsStill,
        Release,
        Chord(Chord),
        Type(String),
    }

    type Log = Arc<Mutex<Vec<Call>>>;

    fn push(log: &Log, c: Call) {
        log.lock().unwrap().push(c);
    }

    struct FakeClipboard {
        log: Log,
        seq: u64,
        waits: VecDeque<RenderWait>,
        cancel_on_wait: Option<CancelToken>,
    }

    impl ClipboardPort for FakeClipboard {
        type Snapshot = &'static str;
        fn snapshot(&mut self) -> Result<Self::Snapshot, InsertError> {
            push(&self.log, Call::Snapshot);
            Ok("previous")
        }
        fn write_delayed(&mut self, text: &str) -> Result<u64, InsertError> {
            push(&self.log, Call::Write(text.into()));
            self.seq += 1;
            Ok(self.seq)
        }
        fn sequence_number(&self) -> u64 {
            self.seq
        }
        fn wait_for_render(&mut self, timeout: Duration) -> RenderWait {
            push(&self.log, Call::Wait(timeout));
            if let Some(c) = &self.cancel_on_wait {
                c.cancel();
            }
            let w = self.waits.pop_front().unwrap_or(RenderWait::TimedOut);
            if w == RenderWait::Changed {
                self.seq += 1;
            }
            w
        }
        fn restore(&mut self, snapshot: Self::Snapshot, after: Duration, if_sequence: u64) {
            assert_eq!(snapshot, "previous");
            assert_eq!(if_sequence, self.seq);
            push(&self.log, Call::Restore(after));
        }
    }

    struct FakeInput {
        log: Log,
        fail_typing: bool,
    }

    impl InputPort for FakeInput {
        fn release_modifiers(&mut self) -> Result<(), InsertError> {
            push(&self.log, Call::Release);
            Ok(())
        }
        fn send_chord(&mut self, chord: Chord) -> Result<(), InsertError> {
            push(&self.log, Call::Chord(chord));
            Ok(())
        }
        fn type_text(&mut self, text: &str) -> Result<(), InsertError> {
            push(&self.log, Call::Type(text.into()));
            if self.fail_typing {
                Err(InsertError::Input("blocked".into()))
            } else {
                Ok(())
            }
        }
    }

    struct FakeFocus {
        log: Log,
        still: bool,
        remote: bool,
    }

    impl FocusPort for FakeFocus {
        fn foreground_window(&self) -> usize {
            0
        }
        fn is_still(&mut self, _: &FocusContext) -> bool {
            push(&self.log, Call::IsStill);
            self.still
        }
        fn is_remote_session(&self) -> bool {
            self.remote
        }
    }

    struct Rig {
        log: Log,
        waits: Vec<RenderWait>,
        still: bool,
        remote: bool,
        fail_typing: bool,
        cancel_on_wait: Option<CancelToken>,
        policy: InsertPolicy,
    }

    impl Rig {
        fn new(waits: Vec<RenderWait>) -> Self {
            Self {
                log: Log::default(),
                waits,
                still: true,
                remote: false,
                fail_typing: false,
                cancel_on_wait: None,
                policy: InsertPolicy::default(),
            }
        }

        fn run(self, target: &FocusContext) -> (Result<InsertOutcome, InsertError>, Vec<Call>) {
            self.run_with(target, &CancelToken::new())
        }

        fn run_with(
            self,
            target: &FocusContext,
            cancel: &CancelToken,
        ) -> (Result<InsertOutcome, InsertError>, Vec<Call>) {
            let log = self.log.clone();
            let mut chain = StrategyChain::new(
                FakeClipboard {
                    log: log.clone(),
                    seq: 100,
                    waits: self.waits.into(),
                    cancel_on_wait: self.cancel_on_wait,
                },
                FakeInput {
                    log: log.clone(),
                    fail_typing: self.fail_typing,
                },
                FakeFocus {
                    log: log.clone(),
                    still: self.still,
                    remote: self.remote,
                },
                self.policy,
            );
            let r = chain.insert(target, "hello", cancel);
            let calls = log.lock().unwrap().clone();
            (r, calls)
        }
    }

    fn notepad() -> FocusContext {
        FocusContext {
            window: 42,
            exe: Some("notepad.exe".into()),
            ..Default::default()
        }
    }

    fn read_by(exe: &str) -> RenderWait {
        RenderWait::Read(Reader {
            pid: Some(7),
            exe: Some(exe.into()),
        })
    }

    const GAP: Duration = Duration::from_millis(30);
    const RENDER: Duration = Duration::from_millis(500);

    #[test]
    fn target_read_restores_after_the_margin() {
        let (r, calls) =
            Rig::new(vec![RenderWait::TimedOut, read_by("Notepad.exe")]).run(&notepad());
        assert_eq!(r, Ok(InsertOutcome::TargetRead));
        assert_eq!(
            calls,
            vec![
                Call::Snapshot,
                Call::Write("hello".into()),
                Call::Wait(GAP),
                Call::IsStill,
                Call::Release,
                Call::Chord(Chord::CtrlV),
                Call::Wait(RENDER),
                Call::Restore(Duration::from_millis(100)),
            ]
        );
    }

    #[test]
    fn early_reader_is_third_party_and_nothing_more_is_awaited() {
        let (r, calls) = Rig::new(vec![read_by("rdpclip.exe")]).run(&notepad());
        assert_eq!(
            r,
            Ok(InsertOutcome::ThirdPartyRead {
                reader: Some("rdpclip.exe".into())
            })
        );
        let waits = calls.iter().filter(|c| matches!(c, Call::Wait(_))).count();
        assert_eq!(waits, 1);
        assert!(calls.contains(&Call::Chord(Chord::CtrlV)));
        assert_eq!(calls.last(), Some(&Call::Restore(Duration::from_secs(1))));
        assert!(!calls.iter().any(|c| matches!(c, Call::Type(_))));
    }

    #[test]
    fn early_read_by_the_target_process_still_counts_as_third_party() {
        let (r, _) = Rig::new(vec![read_by("notepad.exe")]).run(&notepad());
        assert!(matches!(r, Ok(InsertOutcome::ThirdPartyRead { .. })));
    }

    #[test]
    fn late_read_by_another_process_is_third_party_and_never_typed() {
        let (r, calls) = Rig::new(vec![RenderWait::TimedOut, read_by("ditto.exe")]).run(&notepad());
        assert_eq!(
            r,
            Ok(InsertOutcome::ThirdPartyRead {
                reader: Some("ditto.exe".into())
            })
        );
        assert!(!calls.iter().any(|c| matches!(c, Call::Type(_))));
        assert_eq!(calls.last(), Some(&Call::Restore(Duration::from_secs(1))));
    }

    #[test]
    fn unknown_reader_is_third_party() {
        let (r, _) = Rig::new(vec![
            RenderWait::TimedOut,
            RenderWait::Read(Reader::default()),
        ])
        .run(&notepad());
        assert_eq!(r, Ok(InsertOutcome::ThirdPartyRead { reader: None }));
    }

    #[test]
    fn not_requested_falls_back_to_typing_then_restores() {
        let (r, calls) = Rig::new(vec![]).run(&notepad());
        assert_eq!(r, Ok(InsertOutcome::Typed));
        let chord = calls
            .iter()
            .position(|c| *c == Call::Chord(Chord::CtrlV))
            .unwrap();
        let typed = calls
            .iter()
            .position(|c| *c == Call::Type("hello".into()))
            .unwrap();
        assert!(chord < typed);
        assert_eq!(calls.last(), Some(&Call::Restore(Duration::ZERO)));
    }

    #[test]
    fn never_type_list_leaves_text_on_the_clipboard() {
        let mut rig = Rig::new(vec![]);
        rig.policy.apps.by_exe.push((
            "notepad.exe".into(),
            AppPolicy {
                never_type: true,
                ..Default::default()
            },
        ));
        let (r, calls) = rig.run(&notepad());
        assert_eq!(
            r,
            Ok(InsertOutcome::LeftOnClipboard {
                reason: LeaveReason::NeverType
            })
        );
        assert!(!calls.iter().any(|c| matches!(c, Call::Type(_))));
        assert!(!calls.iter().any(|c| matches!(c, Call::Restore(_))));
    }

    #[test]
    fn typing_failure_leaves_text_on_the_clipboard() {
        let mut rig = Rig::new(vec![]);
        rig.fail_typing = true;
        let (r, calls) = rig.run(&notepad());
        assert!(matches!(
            r,
            Ok(InsertOutcome::LeftOnClipboard {
                reason: LeaveReason::TypingFailed(_)
            })
        ));
        assert!(!calls.iter().any(|c| matches!(c, Call::Restore(_))));
    }

    #[test]
    fn focus_change_aborts_before_the_chord() {
        let mut rig = Rig::new(vec![]);
        rig.still = false;
        let (r, calls) = rig.run(&notepad());
        assert_eq!(
            r,
            Ok(InsertOutcome::Refused {
                reason: RefuseReason::FocusChanged,
                on_clipboard: true
            })
        );
        assert!(!calls.iter().any(|c| matches!(c, Call::Chord(_))));
        assert!(!calls.iter().any(|c| matches!(c, Call::Restore(_))));
    }

    #[test]
    fn elevated_target_is_refused_with_text_on_the_clipboard() {
        let target = FocusContext {
            elevated: true,
            ..notepad()
        };
        let (r, calls) = Rig::new(vec![]).run(&target);
        assert_eq!(
            r,
            Ok(InsertOutcome::Refused {
                reason: RefuseReason::Elevated,
                on_clipboard: true
            })
        );
        assert_eq!(calls, vec![Call::Write("hello".into())]);
    }

    #[test]
    fn password_field_is_refused_and_the_clipboard_untouched() {
        let target = FocusContext {
            is_password: true,
            ..notepad()
        };
        let (r, calls) = Rig::new(vec![]).run(&target);
        assert_eq!(
            r,
            Ok(InsertOutcome::Refused {
                reason: RefuseReason::Password,
                on_clipboard: false
            })
        );
        assert!(calls.is_empty());
    }

    #[test]
    fn remote_client_window_is_refused() {
        let target = FocusContext {
            exe: Some("MSTSC.EXE".into()),
            ..notepad()
        };
        let (r, calls) = Rig::new(vec![]).run(&target);
        assert_eq!(
            r,
            Ok(InsertOutcome::Refused {
                reason: RefuseReason::UnsupportedRemote,
                on_clipboard: true
            })
        );
        assert!(!calls.iter().any(|c| matches!(c, Call::Chord(_))));
    }

    #[test]
    fn type_only_in_remote_session_keeps_the_clipboard_clean() {
        let mut rig = Rig::new(vec![]);
        rig.remote = true;
        rig.policy.type_only_in_remote = true;
        let (r, calls) = rig.run(&notepad());
        assert_eq!(r, Ok(InsertOutcome::Typed));
        assert!(
            !calls
                .iter()
                .any(|c| matches!(c, Call::Write(_) | Call::Snapshot))
        );
    }

    #[test]
    fn foreign_write_in_the_gap_aborts_without_restoring() {
        let (r, calls) = Rig::new(vec![RenderWait::Changed]).run(&notepad());
        assert_eq!(r, Ok(InsertOutcome::ClipboardChanged));
        assert!(!calls.iter().any(|c| matches!(c, Call::Chord(_))));
        assert!(!calls.iter().any(|c| matches!(c, Call::Restore(_))));
    }

    #[test]
    fn foreign_write_after_the_chord_is_not_restored_over() {
        let (r, calls) = Rig::new(vec![RenderWait::TimedOut, RenderWait::Changed]).run(&notepad());
        assert_eq!(r, Ok(InsertOutcome::ClipboardChanged));
        assert!(!calls.iter().any(|c| matches!(c, Call::Restore(_))));
        assert!(!calls.iter().any(|c| matches!(c, Call::Type(_))));
    }

    #[test]
    fn per_app_chord_is_used() {
        let mut rig = Rig::new(vec![RenderWait::TimedOut, read_by("conhost.exe")]);
        rig.policy.apps.by_exe.push((
            "conhost.exe".into(),
            AppPolicy {
                chord: Chord::ShiftInsert,
                ..Default::default()
            },
        ));
        let target = FocusContext {
            exe: Some("conhost.exe".into()),
            ..notepad()
        };
        let (r, calls) = rig.run(&target);
        assert_eq!(r, Ok(InsertOutcome::TargetRead));
        assert!(calls.contains(&Call::Chord(Chord::ShiftInsert)));
    }

    #[test]
    fn cancel_before_start_touches_nothing() {
        let c = CancelToken::new();
        c.cancel();
        let (r, calls) = Rig::new(vec![]).run_with(&notepad(), &c);
        assert_eq!(r, Err(InsertError::Cancelled));
        assert!(calls.is_empty());
    }

    #[test]
    fn cancel_during_the_gap_restores_and_sends_no_chord() {
        let c = CancelToken::new();
        let mut rig = Rig::new(vec![]);
        rig.cancel_on_wait = Some(c.clone());
        let (r, calls) = rig.run_with(&notepad(), &c);
        assert_eq!(r, Err(InsertError::Cancelled));
        assert!(!calls.iter().any(|c| matches!(c, Call::Chord(_))));
        assert_eq!(calls.last(), Some(&Call::Restore(Duration::ZERO)));
    }

    #[test]
    fn outcome_messages() {
        assert!(InsertOutcome::TargetRead.message().is_none());
        assert!(InsertOutcome::Typed.delivered());
        let m = InsertOutcome::Refused {
            reason: RefuseReason::Elevated,
            on_clipboard: true,
        }
        .message()
        .unwrap();
        assert!(m.contains("clipboard"), "{m}");
    }
}
