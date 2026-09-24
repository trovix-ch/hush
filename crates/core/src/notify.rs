//! What the user sees and hears: overlay state, sounds, toasts.
//!
//! A port only. The overlay, tray and sound player live in the platform crate; the
//! pipeline talks to them through this trait so it can be tested with a recorder of calls.

use crate::normalize::Provenance;

/// Where the final text came from, reduced to what the overlay shows.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProvenanceHint {
    Rules,
    Llm,
    /// The LLM was wanted but its output was rejected or it failed; rule-pass text was
    /// inserted. D4 asks for a subtle indicator, never an error.
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
    /// `level` in `0.0..=1.0`. The pipeline only ever sends `0.0`; the live meter is fed
    /// by the driver straight from the recorder, because routing 30 updates a second
    /// through the state machine would buy nothing.
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
    /// A message that outlives the overlay, e.g. "text is on the clipboard".
    fn toast(&mut self, message: &str);
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
