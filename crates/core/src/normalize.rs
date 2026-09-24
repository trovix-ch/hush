//! Transcript normalization contract. A normalizer is a cleaner, never an assistant: its
//! output is validated against the source transcript before it can reach the target.

use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::UtteranceId;
use crate::cancel::{CancelToken, Cancelled};

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Style {
    Formal,
    /// Trailing period dropped, contractions kept.
    #[default]
    Casual,
    /// No smart punctuation, no capitalisation changes, no rewording: only fillers and
    /// explicit corrections are touched, by the rule pass alone. The LLM is skipped
    /// because nothing it may do there is beyond the rules.
    Code,
    /// Rule pass only; the LLM is skipped.
    None,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AppContext {
    /// Lower-case, e.g. `code.exe`.
    pub exe: Option<String>,
    pub window_title: Option<String>,
    pub style: Style,
}

#[derive(Debug, Clone)]
pub struct NormalizeRequest<'a> {
    pub transcript: &'a str,
    /// BCP-47 primary tag, if known.
    pub language: Option<&'a str>,
    /// Words and phrases spelled exactly as the user wants them.
    pub vocabulary: &'a [String],
    pub app: &'a AppContext,
    /// The previous utterance's inserted text, for continuity.
    pub previous: Option<&'a str>,
    pub utterance: UtteranceId,
    /// A backend applies the deadline as its own timeout.
    pub cancel: CancelToken,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Check {
    Empty,
    Preamble,
    TrailingExplanation,
    LengthRatio,
    Containment,
    Language,
    Number,
    Verbatim,
    Truncated,
}

impl Check {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Empty => "empty",
            Self::Preamble => "preamble",
            Self::TrailingExplanation => "trailing-explanation",
            Self::LengthRatio => "length-ratio",
            Self::Containment => "containment",
            Self::Language => "language",
            Self::Number => "number",
            Self::Verbatim => "verbatim",
            Self::Truncated => "truncated",
        }
    }
}

#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum Rejection {
    #[error("empty output")]
    Empty,
    #[error("starts with a preamble: {0:?}")]
    Preamble(String),
    #[error("adds text after a line break")]
    TrailingExplanation,
    #[error("length ratio {ratio:.2} outside [{min:.2}, {max:.2}]")]
    LengthRatio { ratio: f64, min: f64, max: f64 },
    #[error("word containment {ratio:.2} below {threshold:.2}")]
    LowContainment { ratio: f64, threshold: f64 },
    #[error("none of the source's longest words survived; language changed?")]
    LanguageChanged,
    /// A small model turned "four thirty" into "forty five" at 0.80 containment, passing
    /// every ratio check. Numbers must come from the source even though that also rejects
    /// a legitimate "four thirty" → "4:30".
    #[error("number {0:?} is not in the source")]
    NumberChanged(String),
    /// In a terminal or editor the model may only delete; "git" → "Git" or an added
    /// period breaks the command.
    #[error("code style: {0:?} is not verbatim from the source")]
    NotVerbatim(String),
    /// Set by the backend, not by validation: a cut-off output can still look faithful.
    #[error("output hit the token cap")]
    Truncated,
}

impl Rejection {
    pub fn check(&self) -> Check {
        match self {
            Self::Empty => Check::Empty,
            Self::Preamble(_) => Check::Preamble,
            Self::TrailingExplanation => Check::TrailingExplanation,
            Self::LengthRatio { .. } => Check::LengthRatio,
            Self::LowContainment { .. } => Check::Containment,
            Self::LanguageChanged => Check::Language,
            Self::NumberChanged(_) => Check::Number,
            Self::NotVerbatim(_) => Check::Verbatim,
            Self::Truncated => Check::Truncated,
        }
    }

    pub fn score(&self) -> Option<f64> {
        match self {
            Self::LengthRatio { ratio, .. } | Self::LowContainment { ratio, .. } => Some(*ratio),
            _ => None,
        }
    }

    pub fn threshold(&self) -> Option<f64> {
        match self {
            Self::LengthRatio { ratio, min, max } => Some(if ratio < min { *min } else { *max }),
            Self::LowContainment { threshold, .. } => Some(*threshold),
            _ => None,
        }
    }

    pub fn detail(&self) -> Option<&str> {
        match self {
            Self::Preamble(s) | Self::NumberChanged(s) | Self::NotVerbatim(s) => Some(s),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Measured {
    pub value: f64,
    pub min: f64,
    pub max: Option<f64>,
}

impl Measured {
    /// How close a pass came to being a rejection.
    pub fn margin(&self) -> f64 {
        let low = self.value - self.min;
        self.max.map_or(low, |m| low.min(m - self.value))
    }
}

/// Scores of a validation that passed. `None` for a check that did not apply (too few
/// words for a ratio to mean anything).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Scores {
    pub length_ratio: Option<Measured>,
    pub containment: Option<Measured>,
}

/// In both fallback variants the text is the rule-pass output.
#[derive(Debug, Clone, PartialEq)]
pub enum Provenance {
    Rules,
    Llm { model: String, scores: Scores },
    LlmRejected { model: String, rejection: Rejection },
    LlmFailed { stage: String, error: String },
}

impl Provenance {
    pub fn is_fallback(&self) -> bool {
        matches!(self, Self::LlmRejected { .. } | Self::LlmFailed { .. })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct NormalizeOutput {
    pub utterance: UtteranceId,
    pub text: String,
    pub provenance: Provenance,
    pub elapsed: Duration,
}

#[derive(Debug, thiserror::Error)]
pub enum NormalizeError {
    #[error("backend unavailable: {0}")]
    Unavailable(String),
    #[error("request failed: {0}")]
    Request(String),
    /// Not a reply at all (not JSON, no choices), as opposed to a well-formed reply that
    /// validation rejects.
    #[error("malformed response: {0}")]
    Malformed(String),
    #[error("cancelled")]
    Cancelled,
    #[error("deadline passed")]
    Deadline,
    /// The normalizer must be rebuilt.
    #[error("backend died: {0}")]
    BackendDied(String),
}

impl From<Cancelled> for NormalizeError {
    fn from(c: Cancelled) -> Self {
        match c {
            Cancelled::Requested => Self::Cancelled,
            Cancelled::Deadline => Self::Deadline,
        }
    }
}

/// Takes `&mut self`: an embedded model's resident context is not shareable, and a
/// `Mutex` behind `&self` would only hide that.
pub trait Normalizer: Send {
    fn id(&self) -> &str;

    /// Load or connect so the first real call pays no warm-up cost.
    fn warm(&mut self) -> Result<(), NormalizeError> {
        Ok(())
    }

    fn normalize(&mut self, req: &NormalizeRequest<'_>) -> Result<NormalizeOutput, NormalizeError>;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejection_reports_check_score_and_bound() {
        let r = Rejection::LengthRatio {
            ratio: 0.2,
            min: 0.3,
            max: 1.5,
        };
        assert_eq!(r.check(), Check::LengthRatio);
        assert_eq!(r.score(), Some(0.2));
        assert_eq!(r.threshold(), Some(0.3));
        let r = Rejection::LengthRatio {
            ratio: 2.0,
            min: 0.3,
            max: 1.5,
        };
        assert_eq!(r.threshold(), Some(1.5));
        let r = Rejection::NumberChanged("forty".into());
        assert_eq!(r.check().as_str(), "number");
        assert_eq!(r.score(), None);
        assert_eq!(r.detail(), Some("forty"));
    }

    #[test]
    fn margin_is_distance_to_the_nearer_bound() {
        let m = Measured {
            value: 0.8,
            min: 0.7,
            max: None,
        };
        assert!((m.margin() - 0.1).abs() < 1e-9);
        let m = Measured {
            value: 1.4,
            min: 0.3,
            max: Some(1.5),
        };
        assert!((m.margin() - 0.1).abs() < 1e-9);
    }

    #[test]
    fn normalizer_is_object_safe() {
        fn _takes(_: &mut dyn Normalizer) {}
    }
}
