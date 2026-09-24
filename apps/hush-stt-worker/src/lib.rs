//! The transcribe.cpp engine. It lives in its own package because Cargo unifies features
//! across everything built together: were it an optional part of `hush-stt`, building the
//! workspace would link its static ggml into hush.exe beside llama.cpp's (LNK2005).

pub mod transcribe_cpp;

pub use transcribe_cpp::{TranscribeCppEngine, TranscribeCppOptions};
