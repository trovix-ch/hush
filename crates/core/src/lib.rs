//! Platform-neutral contracts and pipeline logic for whisper-local. Nothing here may
//! depend on Win32 or an inference library.

pub mod cancel;
pub mod config;
pub mod context;
pub mod history;
pub mod insert;
pub mod normalize;
pub mod notify;
pub mod pipeline;
pub mod recorder;
pub mod segment;
pub mod stt;
mod utterance;

pub use cancel::{CancelToken, Cancelled};
pub use utterance::UtteranceId;
