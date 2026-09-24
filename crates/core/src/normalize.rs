//! Transcript normalization contract.
//!
//! A normalizer turns a raw transcript into the text that gets inserted: fillers gone,
//! self-corrections applied, punctuation fixed, vocabulary respected. Implementations
//! range from a zero-latency rule pass to a local language model. Whatever the
//! implementation, the pipeline validates the output against the source transcript
//! before it can reach the target app; a normalizer is a cleaner, never an assistant.

use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::UtteranceId;
use crate::cancel::{CancelToken, Cancelled};

/// Writing style requested for the target app.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum Style {
    /// Full sentences, proper punctuation, no slang.
    Formal,
    /// Chat-style: trailing period dropped, contractions kept.
    #[default]
    Casual,
    /// Code editor or terminal: no smart punctuation, no capitalisation changes, no
    /// rewording. Only fillers and explicit corrections are touched.
    Code,
    /// Rule pass only; the LLM is skipped.
    None,
}

/// What the pipeline knows about where the text is going.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct AppContext {
    /// Executable name of the foreground process, lower-case, e.g. `code.exe`.
    pub exe: Option<String>,
    pub window_title: Option<String>,
    pub style: Style,
}

#[derive(Debug, Clone)]
pub struct NormalizeRequest<'a> {
    pub transcript: &'a str,
    /// BCP-47 primary tag of the transcript, if known.
    pub language: Option<&'a str>,
    /// User's personal dictionary: words and phrases spelled exactly as wanted.
    pub vocabulary: &'a [String],
    pub app: &'a AppContext,
    /// The sentence inserted by the previous utterance, for continuity.
    pub previous: Option<&'a str>,
    /// Echoed into [`NormalizeOutput::utterance`].
    pub utterance: UtteranceId,
    /// Cancellation and deadline. A backend applies the deadline as its own timeout.
    pub cancel: CancelToken,
}

/// Which validation check turned an LLM output into a fallback.
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
    /// Stable name for logs and metrics.
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

/// Why validation rejected an LLM output, with the measured score and the bound it
/// crossed where the check has one.
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
    /// A small model turned "four thirty" into "forty five" at 0.80 containment, which
    /// passes every ratio check. A wrong number is the costliest silent error, so
    /// numbers must come from the source even when that also rejects a legitimate
    /// "four thirty" → "4:30".
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

    /// The measured value, for checks that measure one.
    pub fn score(&self) -> Option<f64> {
        match self {
            Self::LengthRatio { ratio, .. } | Self::LowContainment { ratio, .. } => Some(*ratio),
            _ => None,
        }
    }

    /// The bound the score crossed: the nearer of the two for the length ratio.
    pub fn threshold(&self) -> Option<f64> {
        match self {
            Self::LengthRatio { ratio, min, max } => Some(if ratio < min { *min } else { *max }),
            Self::LowContainment { threshold, .. } => Some(*threshold),
            _ => None,
        }
    }

    /// The offending text, for checks that point at one.
    pub fn detail(&self) -> Option<&str> {
        match self {
            Self::Preamble(s) | Self::NumberChanged(s) | Self::NotVerbatim(s) => Some(s),
            _ => None,
        }
    }
}

/// A measured value and the bounds it had to fall within.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Measured {
    pub value: f64,
    pub min: f64,
    pub max: Option<f64>,
}

impl Measured {
    /// Distance to the nearest bound; how close a pass came to being a rejection.
    pub fn margin(&self) -> f64 {
        let low = self.value - self.min;
        self.max.map_or(low, |m| low.min(m - self.value))
    }
}

/// Scores of a validation that passed, so logs can show how close it was. `None` for a
/// check that did not apply (too few words for a ratio to mean anything).
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct Scores {
    pub length_ratio: Option<Measured>,
    pub containment: Option<Measured>,
}

/// Which stage produced the final text, so the overlay and logs can say so.
#[derive(Debug, Clone, PartialEq)]
pub enum Provenance {
    Rules,
    Llm {
        model: String,
        scores: Scores,
    },
    /// The LLM answered but validation rejected it; the text is the rule-pass output.
    LlmRejected {
        model: String,
        rejection: Rejection,
    },
    /// The LLM stage failed (timeout, unreachable, crashed); the text is the rule-pass
    /// output.
    LlmFailed {
        stage: String,
        error: String,
    },
}

impl Provenance {
    /// Whether an LLM was wanted but its output did not reach the user.
    pub fn is_fallback(&self) -> bool {
        matches!(self, Self::LlmRejected { .. } | Self::LlmFailed { .. })
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct NormalizeOutput {
    /// The utterance this answers, from [`NormalizeRequest::utterance`].
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
    /// The backend answered with something that is not a reply at all (not JSON, no
    /// choices). Distinct from a validation rejection, which is a well-formed reply with
    /// unacceptable text.
    #[error("malformed response: {0}")]
    Malformed(String),
    #[error("cancelled")]
    Cancelled,
    #[error("deadline passed")]
    Deadline,
    /// The backend's process or device is gone; the normalizer must be rebuilt.
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

/// Takes `&mut self` like `SttEngine`: an embedded model's resident context is not
/// shareable, and a `Mutex` behind `&self` would only hide that.
pub trait Normalizer: Send {
    /// Stable identifier used in config and logs.
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
