//! Microphone capture contract.
//!
//! A recorder hands the pipeline 16 kHz mono f32 PCM, already including a short pre-roll
//! when the stream was warm. Device handling, resampling and the warm window live behind
//! the trait so the state machine can be tested with a scripted recorder.

use std::time::Duration;

/// How capture behaves. Defaults follow the warm-window and pre-roll figures in the design.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecorderConfig {
    /// Input device by name or id. `None` follows the system default.
    pub device: Option<String>,
    /// Audio kept from before `start()` while the stream is warm, so the first syllable
    /// survives the key press racing the speaker.
    pub pre_roll: Duration,
    /// How long the stream stays open after the last utterance. Longer keeps the Windows
    /// mic-in-use indicator lit for no benefit; zero makes every dictation a cold start.
    pub warm_window: Duration,
    /// Hard cap on one recording. Reaching it is not an error: capture stops growing and
    /// `stop()` returns what it has with [`Recording::max_duration_reached`] set.
    pub max_duration: Duration,
}

impl Default for RecorderConfig {
    fn default() -> Self {
        Self {
            device: None,
            pre_roll: Duration::from_millis(300),
            warm_window: Duration::from_secs(30),
            max_duration: Duration::from_secs(120),
        }
    }
}

/// One finished capture.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Recording {
    /// Mono samples at [`Recording::sample_rate`], pre-roll first.
    pub pcm: Vec<f32>,
    /// Always [`crate::stt::SAMPLE_RATE`] for recorders in this workspace; carried so a
    /// consumer never has to assume it.
    pub sample_rate: u32,
    /// Length of `pcm`, not wall-clock time between `start()` and `stop()`: those differ
    /// by the pre-roll and by anything dropped.
    pub duration: Duration,
    /// Frames (at `sample_rate`) known to be missing: ring-buffer overruns and time spent
    /// without a device after the input went away mid-recording. Non-zero means the
    /// transcript may have holes and the user should be told.
    pub dropped_frames: usize,
    /// Capture hit `RecorderConfig::max_duration`; audio after the cap was discarded.
    pub max_duration_reached: bool,
    /// The input device went away or changed during this recording. What was captured
    /// before that is kept.
    pub device_lost: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, thiserror::Error)]
pub enum RecorderError {
    #[error("no input device is available")]
    NoInputDevice,
    #[error("input device error: {0}")]
    Device(String),
    #[error("capture stream error: {0}")]
    Stream(String),
    /// A call that does not fit the current state, e.g. `stop()` without `start()`.
    /// A bug in the caller, reported rather than silently ignored.
    #[error("recorder is {0}")]
    State(&'static str),
}

/// Start/stop capture with pre-roll, returning 16 kHz mono PCM.
///
/// Takes `&mut self` because one recorder owns one capture at a time; sharing it behind a
/// `Mutex` would only hide that.
pub trait Recorder: Send {
    /// Begin accumulating. Opens the stream if it is cold, otherwise reuses it and seeds
    /// the recording with the pre-roll.
    fn start(&mut self) -> Result<(), RecorderError>;

    /// Finish the current recording. The stream stays warm for the configured window.
    fn stop(&mut self) -> Result<Recording, RecorderError>;

    /// Discard the current recording, if any. The stream stays warm as after `stop()`.
    fn cancel(&mut self);

    /// Recent input level in `0.0..=1.0` for the overlay meter; `0.0` when the stream is
    /// closed.
    fn level(&self) -> f32;

    /// Whether a capture stream is currently open.
    fn is_warm(&self) -> bool;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_the_design() {
        let c = RecorderConfig::default();
        assert_eq!(c.pre_roll, Duration::from_millis(300));
        assert_eq!(c.warm_window, Duration::from_secs(30));
        assert_eq!(c.max_duration, Duration::from_secs(120));
        assert!(c.device.is_none());
    }

    #[test]
    fn recorder_is_object_safe() {
        fn _takes(_: &mut dyn Recorder) {}
    }
}
