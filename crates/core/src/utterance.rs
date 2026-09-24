//! Identity of one dictation, carried on every request and result so a reply for an
//! utterance the pipeline has already moved past is recognised and dropped.

use serde::{Deserialize, Serialize};

/// Monotonic within one pipeline. Zero is never issued, so a default-constructed id in a
/// request that forgot to set it cannot match a live utterance.
#[derive(
    Debug, Clone, Copy, Default, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize,
)]
#[serde(transparent)]
pub struct UtteranceId(pub u64);

impl UtteranceId {
    pub const FIRST: Self = Self(1);

    pub fn next(self) -> Self {
        Self(self.0 + 1)
    }
}

impl std::fmt::Display for UtteranceId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "#{}", self.0)
    }
}
