//! Speech-to-text engine contract. Resampling, voice activity detection and segment
//! stitching happen outside the engine so every engine benefits equally.

use std::time::Duration;

use crate::UtteranceId;
use crate::cancel::{CancelToken, Cancelled};

pub const SAMPLE_RATE: u32 = 16_000;

/// The backend an engine actually loaded on. Reported, never assumed: some runtimes fall
/// back to CPU silently.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Cpu,
    DirectMl,
    Cuda,
    Vulkan,
}

/// Capabilities as data so callers can adapt instead of probing.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Caps {
    pub prompt: bool,
    pub hotwords: bool,
    pub word_timestamps: bool,
    /// Output already carries punctuation and casing.
    pub punctuation: bool,
    /// An in-flight `transcribe` stops within one encoder pass plus one decode step;
    /// without it the token is checked before the call only. Either way a cancelled call
    /// returns `SttError::Cancelled`, never a result.
    pub cancel: bool,
}

#[derive(Debug, Clone)]
pub struct EngineInfo {
    /// Stable config identifier, e.g. `parakeet-tdt-0.6b-v3`.
    pub id: String,
    pub backend: Backend,
    /// Which GPU matters as much as GPU-or-CPU: a card shared with the LLM runs several
    /// times slower. `None` when the runtime does not say.
    pub device: Option<String>,
    pub caps: Caps,
    /// BCP-47 primary tags. Empty means "any".
    pub languages: Vec<String>,
}

#[derive(Debug, Clone, Default)]
pub struct DecodeOptions {
    /// BCP-47 primary tag. `None` lets the engine detect.
    pub language: Option<String>,
    pub prompt: Option<String>,
    pub hotwords: Vec<String>,
    pub utterance: UtteranceId,
    pub cancel: CancelToken,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Segment {
    pub text: String,
    pub start: Duration,
    pub end: Duration,
}

#[derive(Debug, Clone, Default)]
pub struct Transcript {
    pub utterance: UtteranceId,
    /// Segments joined, whitespace normalised.
    pub text: String,
    pub segments: Vec<Segment>,
    /// One entry per word, relative to the start of the audio; empty when the engine
    /// does not time words.
    pub words: Vec<Segment>,
    pub language: Option<String>,
    pub inference_time: Duration,
}

#[derive(Debug, thiserror::Error)]
pub enum SttError {
    #[error("model could not be loaded: {0}")]
    Load(String),
    #[error("requested backend is unavailable: {0}")]
    Backend(String),
    #[error("inference failed: {0}")]
    Inference(String),
    #[error("audio is empty or too short")]
    EmptyAudio,
    #[error("cancelled")]
    Cancelled,
    #[error("deadline passed")]
    Deadline,
    /// The in-flight request is lost. An engine that owns a child process restarts it
    /// and serves the next request; an in-process engine has to be rebuilt.
    #[error("backend died: {0}")]
    BackendDied(String),
}

impl From<Cancelled> for SttError {
    fn from(c: Cancelled) -> Self {
        match c {
            Cancelled::Requested => Self::Cancelled,
            Cancelled::Deadline => Self::Deadline,
        }
    }
}

pub trait SttEngine: Send {
    fn info(&self) -> &EngineInfo;

    /// Load weights and run one dummy pass so the first real call pays no warm-up cost.
    fn warm_up(&mut self) -> Result<(), SttError>;

    /// A cheap pass that raises an idle GPU's clocks ahead of a real call. Called at
    /// key-down, while the user is still speaking; engines with nothing to ramp do nothing.
    fn nudge(&mut self) -> Result<(), SttError> {
        Ok(())
    }

    /// `pcm` is 16 kHz mono in [-1, 1].
    fn transcribe(&mut self, pcm: &[f32], opts: &DecodeOptions) -> Result<Transcript, SttError>;
}
