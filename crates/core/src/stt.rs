//! Speech-to-text engine contract.
//!
//! Engines receive 16 kHz mono f32 PCM and return a transcript. Resampling, voice
//! activity detection and segment stitching happen outside the engine so every engine
//! benefits from them equally.

use std::time::Duration;

use crate::UtteranceId;
use crate::cancel::{CancelToken, Cancelled};

/// Sample rate every engine receives. Capture is resampled to this before inference.
pub const SAMPLE_RATE: u32 = 16_000;

/// Compute backend an engine actually loaded on. Reported, never assumed: some runtimes
/// fall back to CPU silently and the app must be able to tell the user.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Backend {
    Cpu,
    DirectMl,
    Cuda,
    Vulkan,
}

/// What an engine can do, expressed as data so callers can adapt instead of probing.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Caps {
    /// Accepts a free-text prompt that biases decoding (Whisper's initial prompt).
    pub prompt: bool,
    /// Accepts a list of words to bias towards.
    pub hotwords: bool,
    /// Returns per-word timestamps.
    pub word_timestamps: bool,
    /// Output already carries punctuation and casing.
    pub punctuation: bool,
    /// An in-flight `transcribe` stops at the next decode step when the token is
    /// cancelled. Encoders are not interruptible, so the bound is one encoder pass plus
    /// one step, not a fixed latency. Without it, the token is checked before the call
    /// only. Either way a cancelled call returns `SttError::Cancelled`, never a result.
    pub cancel: bool,
}

#[derive(Debug, Clone)]
pub struct EngineInfo {
    /// Stable identifier used in config, e.g. `parakeet-tdt-0.6b-v3`.
    pub id: String,
    pub backend: Backend,
    /// Human-readable name of the device the weights actually live on, e.g. the GPU
    /// model. `None` when the runtime does not say. With two GPUs, "which one" matters as
    /// much as "GPU or CPU": a card shared with the LLM runs several times slower.
    pub device: Option<String>,
    pub caps: Caps,
    /// Languages the engine claims, as BCP-47 primary tags. Empty means "any".
    pub languages: Vec<String>,
}

#[derive(Debug, Clone, Default)]
pub struct DecodeOptions {
    /// BCP-47 primary tag, e.g. `en`, `de`. `None` lets the engine detect.
    pub language: Option<String>,
    /// Free-text bias, honoured only if `Caps::prompt`.
    pub prompt: Option<String>,
    /// Vocabulary bias, honoured only if `Caps::hotwords`.
    pub hotwords: Vec<String>,
    /// Echoed into [`Transcript::utterance`].
    pub utterance: UtteranceId,
    /// Cancellation and deadline for this call.
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
    /// The utterance this answers, from [`DecodeOptions::utterance`].
    pub utterance: UtteranceId,
    /// Full text, segments joined, whitespace normalised.
    pub text: String,
    pub segments: Vec<Segment>,
    /// Detected or requested language.
    pub language: Option<String>,
    /// Wall-clock time spent inside the engine for this call.
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
    /// The engine's process or device is gone (child exited, GPU device lost). The engine
    /// must be rebuilt; retrying on the same instance is pointless.
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

    /// Transcribe one utterance. `pcm` is 16 kHz mono f32 in [-1, 1].
    fn transcribe(&mut self, pcm: &[f32], opts: &DecodeOptions) -> Result<Transcript, SttError>;
}
