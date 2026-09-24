//! Speech-to-text engines implementing `wl_core::stt::SttEngine`.
//!
//! The default engine is transcribe.cpp (ggml, Vulkan or CPU). The ONNX Runtime Parakeet
//! engine is compiled only with the `ort-engine` feature.

pub mod models;
#[cfg(feature = "ort-engine")]
pub mod parakeet;
pub mod transcribe_cpp;

#[cfg(feature = "ort-engine")]
pub use parakeet::{ParakeetEngine, ParakeetOptions};
pub use transcribe_cpp::{TranscribeCppEngine, TranscribeCppOptions};

/// Version string of the ONNX Runtime build linked into this binary.
#[cfg(feature = "ort-engine")]
pub fn onnxruntime_build_info() -> String {
    ort::info().to_string()
}
