//! The port for what the user sees and hears: overlay state, sounds, toasts.

use crate::normalize::Provenance;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProvenanceHint {
    Rules,
    Llm,
    /// Rule-pass text was inserted because the LLM failed or was rejected. Shown as a
    /// subtle indicator, never an error.
    LlmFallback,
}

impl From<&Provenance> for ProvenanceHint {
    fn from(p: &Provenance) -> Self {
        match p {
            Provenance::Rules => Self::Rules,
            Provenance::Llm { .. } => Self::Llm,
            Provenance::LlmRejected { .. } | Provenance::LlmFailed { .. } => Self::LlmFallback,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum OverlayState {
    Idle,
    /// `level` in `0.0..=1.0`. The pipeline only ever sends `0.0`; the live meter goes
    /// straight from the driver to the overlay.
    Listening {
        level: f32,
    },
    Transcribing,
    Normalizing,
    Inserting,
    Done {
        provenance_hint: ProvenanceHint,
    },
    Error {
        message: String,
    },
    /// Long-running work outside dictation, shown until replaced. `fraction` in
    /// `0.0..=1.0`.
    Progress {
        message: String,
        fraction: f32,
    },
    /// An error that stays up until replaced, because the user has to act on it.
    Alert {
        message: String,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Sound {
    Start,
    Stop,
    Cancel,
    Error,
}

/// Every method must return quickly: implementations post to the UI thread and never
/// block the pipeline on rendering or audio playback.
pub trait Notifier: Send {
    fn set_state(&mut self, state: OverlayState);
    fn play(&mut self, sound: Sound);
    /// A message that outlives the overlay.
    fn toast(&mut self, message: &str);
    /// Shown whenever the overlay would otherwise be hidden, so work outside dictation (a
    /// model download, a failure the user must act on) survives the transient states
    /// that pass over it. `None` clears it.
    fn set_background(&mut self, state: Option<OverlayState>);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::normalize::{Rejection, Scores};

    #[test]
    fn provenance_maps_to_a_hint() {
        assert_eq!(
            ProvenanceHint::from(&Provenance::Rules),
            ProvenanceHint::Rules
        );
        assert_eq!(
            ProvenanceHint::from(&Provenance::Llm {
                model: "m".into(),
                scores: Scores::default()
            }),
            ProvenanceHint::Llm
        );
        assert_eq!(
            ProvenanceHint::from(&Provenance::LlmRejected {
                model: "m".into(),
                rejection: Rejection::Empty
            }),
            ProvenanceHint::LlmFallback
        );
    }

    #[test]
    fn notifier_is_object_safe() {
        fn _takes(_: &mut dyn Notifier) {}
    }
}
