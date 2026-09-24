//! Microphone capture, resampling to 16 kHz mono, and voice activity detection.
//!
//! The device callback only downmixes into a lock-free ring. Resampling, pre-roll and the
//! warm window run on a dedicated worker thread, see [`WorkerRecorder`].

pub mod capture;
mod cpal_backend;
pub mod engine;
pub mod resample;
pub mod vad;

pub use cpal_backend::{CpalOpener, InputDeviceInfo, list_input_devices};
pub use engine::{Clock, SystemClock, WorkerRecorder};
pub use wl_core::recorder::{Recorder, RecorderConfig, RecorderError, Recording};

/// The production recorder: cpal capture on its own worker thread.
pub struct CpalRecorder(WorkerRecorder);

impl CpalRecorder {
    /// Spawns the worker; no device is opened until the first [`Recorder::start`].
    pub fn new(cfg: RecorderConfig) -> Result<Self, RecorderError> {
        WorkerRecorder::spawn(cfg, CpalOpener, SystemClock).map(Self)
    }
}

impl Recorder for CpalRecorder {
    fn start(&mut self) -> Result<(), RecorderError> {
        self.0.start()
    }
    fn stop(&mut self) -> Result<Recording, RecorderError> {
        self.0.stop()
    }
    fn cancel(&mut self) {
        self.0.cancel()
    }
    fn level(&self) -> f32 {
        self.0.level()
    }
    fn is_warm(&self) -> bool {
        self.0.is_warm()
    }
}
