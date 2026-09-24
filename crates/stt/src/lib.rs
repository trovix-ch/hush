pub mod models;
#[cfg(feature = "ort-engine")]
pub mod parakeet;
pub mod protocol;
pub mod remote;

#[cfg(feature = "ort-engine")]
pub use parakeet::{ParakeetEngine, ParakeetOptions};
pub use remote::{RemoteEngine, RemoteOptions};

#[cfg(feature = "ort-engine")]
pub fn onnxruntime_build_info() -> String {
    ort::info().to_string()
}
