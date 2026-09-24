pub mod models;
#[cfg(feature = "ort-engine")]
pub mod parakeet;
pub mod transcribe_cpp;

#[cfg(feature = "ort-engine")]
pub use parakeet::{ParakeetEngine, ParakeetOptions};
pub use transcribe_cpp::{TranscribeCppEngine, TranscribeCppOptions};

#[cfg(feature = "ort-engine")]
pub fn onnxruntime_build_info() -> String {
    ort::info().to_string()
}
