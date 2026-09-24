//! Platform-neutral core of whisper-local: the contracts every engine, normalizer and
//! inserter implements, plus the data that flows between them.
//!
//! Nothing here may depend on Win32, ONNX Runtime or any inference library. Those live
//! in sibling crates and plug in through the traits defined here.

pub mod cancel;
pub mod config;
pub mod context;
pub mod history;
pub mod insert;
pub mod normalize;
pub mod notify;
pub mod pipeline;
pub mod recorder;
pub mod stt;
mod utterance;

pub use cancel::{CancelToken, Cancelled};
pub use utterance::UtteranceId;
