//! Microphone capture contract.

use std::time::Duration;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RecorderConfig {
    /// Name or id; `None` follows the system default.
    pub device: Option<String>,
    /// Audio kept from before `start()` while warm, so the first syllable survives the key
    /// press racing the speaker.
    pub pre_roll: Duration,
    /// Longer keeps the Windows mic-in-use indicator lit for no benefit; zero makes every
    /// dictation a cold start.
    pub warm_window: Duration,
    /// Reaching it is not an error: `stop()` returns what it has with the flag set.
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

#[derive(Debug, Clone, Default, PartialEq)]
pub struct Recording {
    /// Mono, pre-roll first.
    pub pcm: Vec<f32>,
    pub sample_rate: u32,
    /// Length of `pcm`, not wall-clock time: those differ by the pre-roll and by anything
    /// dropped.
    pub duration: Duration,
    /// Ring overruns plus time without a device. Non-zero means the transcript may have
    /// holes and the user should be told.
    pub dropped_frames: usize,
    pub max_duration_reached: bool,
    /// What was captured before the device went away is kept.
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
    /// A call that does not fit the current state, e.g. `stop()` without `start()`: a
    /// caller bug, reported rather than ignored.
    #[error("recorder is {0}")]
    State(&'static str),
}

/// Start/stop capture with pre-roll, returning 16 kHz mono PCM.
pub trait Recorder: Send {
    fn start(&mut self) -> Result<(), RecorderError>;

    /// The stream stays warm for the configured window.
    fn stop(&mut self) -> Result<Recording, RecorderError>;

    /// The stream stays warm as after `stop()`.
    fn cancel(&mut self);

    /// `0.0..=1.0`; `0.0` when the stream is closed.
    fn level(&self) -> f32;

    fn is_warm(&self) -> bool;
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_values() {
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
