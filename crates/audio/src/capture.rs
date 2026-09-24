//! The device side: opening a stream and the real-time callback.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};

use rtrb::Producer;
use wl_core::recorder::RecorderError;

/// State the real-time callbacks may touch. Atomics only: the callback must not block,
/// allocate or log, and the error callback runs on the same device thread.
#[derive(Debug, Default)]
pub struct CallbackShared {
    /// Device-rate frames the ring had no room for.
    pub dropped: AtomicUsize,
    /// The stream is dead: device removed, or the default changed and WASAPI does not
    /// rebind an open client.
    pub lost: AtomicBool,
}

/// An open capture stream. Dropping it closes the device.
pub struct OpenedStream {
    pub sample_rate: u32,
    pub channels: u16,
    pub device_name: String,
    pub handle: Box<dyn Send>,
}

/// Opens capture streams that deliver mono f32 at the device rate into `producer`.
/// A trait so the worker's warm-window and pre-roll logic runs in tests without hardware.
pub trait StreamOpener: Send + 'static {
    fn open(
        &mut self,
        device: Option<&str>,
        producer: Producer<f32>,
        shared: Arc<CallbackShared>,
    ) -> Result<OpenedStream, RecorderError>;
}

/// Average interleaved frames to mono and push them. Returns frames that did not fit.
///
/// Mono by averaging, not by picking channel 0: array mics and some USB headsets put the
/// signal on one channel only, and an average keeps it at worst 6 dB down instead of gone.
pub fn downmix_into<T, F>(
    data: &[T],
    channels: usize,
    producer: &mut Producer<f32>,
    conv: F,
) -> usize
where
    T: Copy,
    F: Fn(T) -> f32,
{
    let channels = channels.max(1);
    let scale = 1.0 / channels as f32;
    let mut dropped = 0;
    for frame in data.chunks_exact(channels) {
        let mut acc = 0.0;
        for &s in frame {
            acc += conv(s);
        }
        if producer.push(acc * scale).is_err() {
            dropped += 1;
        }
    }
    dropped
}

pub(crate) fn note_dropped(shared: &CallbackShared, n: usize) {
    if n > 0 {
        shared.dropped.fetch_add(n, Ordering::Relaxed);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rtrb::RingBuffer;

    #[test]
    fn stereo_average() {
        let (mut p, mut c) = RingBuffer::new(16);
        let data = [1.0f32, 0.0, 0.5, 0.5, -1.0, 1.0];
        assert_eq!(downmix_into(&data, 2, &mut p, |x| x), 0);
        let got: Vec<f32> = std::iter::from_fn(|| c.pop().ok()).collect();
        assert_eq!(got, vec![0.5, 0.5, 0.0]);
    }

    #[test]
    fn i16_conversion_and_single_channel_signal() {
        let (mut p, mut c) = RingBuffer::new(16);
        let data = [i16::MAX, 0, i16::MIN, 0];
        downmix_into(&data, 2, &mut p, |x| x as f32 / 32768.0);
        let a = c.pop().unwrap();
        let b = c.pop().unwrap();
        assert!((a - 0.5).abs() < 1e-3);
        assert!((b + 0.5).abs() < 1e-3);
    }

    #[test]
    fn full_ring_counts_drops() {
        let (mut p, _c) = RingBuffer::<f32>::new(2);
        let data = [0.1f32; 4];
        assert_eq!(downmix_into(&data, 1, &mut p, |x| x), 2);
    }

    #[test]
    fn partial_frame_is_ignored() {
        let (mut p, c) = RingBuffer::new(8);
        downmix_into(&[1.0f32, 1.0, 1.0], 2, &mut p, |x| x);
        assert_eq!(c.slots(), 1);
    }
}
