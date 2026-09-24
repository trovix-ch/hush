//! Streaming fixed-ratio resampling of mono audio to 16 kHz.

use audioadapter_buffers::direct::InterleavedSlice;
use rubato::{Fft, FixedSync, Indexing, Resampler};
use wl_core::stt::SAMPLE_RATE;

/// Feeds arbitrary-length blocks through a fixed-chunk FFT resampler and keeps the output
/// length equal to `input_len * ratio` over a whole take, with the resampler's start-up
/// delay removed. Getting the length exact matters more than it looks: pre-roll seeding
/// and VAD timestamps are both sample offsets.
pub struct StreamResampler {
    /// `None` when the device already runs at 16 kHz.
    inner: Option<Fft<f32>>,
    rate_in: u32,
    chunk_in: usize,
    pending: Vec<f32>,
    out_buf: Vec<f32>,
    /// Output frames still to discard: the FFT delay after construction or reset.
    skip: usize,
    delay: usize,
    frames_in: u64,
    frames_out: u64,
}

impl StreamResampler {
    pub fn new(rate_in: u32) -> Result<Self, String> {
        if rate_in == 0 {
            return Err("input sample rate is zero".into());
        }
        if rate_in == SAMPLE_RATE {
            return Ok(Self {
                inner: None,
                rate_in,
                chunk_in: 0,
                pending: Vec::new(),
                out_buf: Vec::new(),
                skip: 0,
                delay: 0,
                frames_in: 0,
                frames_out: 0,
            });
        }
        // 20 ms chunks: small enough that the level meter and VAD see fresh audio, large
        // enough that FFT overhead per sample stays negligible. `Both` makes every call
        // consume and produce a constant count, so no per-call size bookkeeping.
        let chunk = (rate_in as usize / 50).max(1);
        let fft = Fft::<f32>::new(
            rate_in as usize,
            SAMPLE_RATE as usize,
            chunk,
            1,
            FixedSync::Both,
        )
        .map_err(|e| e.to_string())?;
        let chunk_in = fft.input_frames_next();
        let delay = fft.output_delay();
        let out_len = fft.output_frames_max();
        Ok(Self {
            inner: Some(fft),
            rate_in,
            chunk_in,
            pending: Vec::with_capacity(chunk_in * 4),
            out_buf: vec![0.0; out_len],
            skip: delay,
            delay,
            frames_in: 0,
            frames_out: 0,
        })
    }

    pub fn rate_in(&self) -> u32 {
        self.rate_in
    }

    /// Output frames the given number of input frames should become.
    fn expected_out(&self, frames_in: u64) -> u64 {
        (frames_in * SAMPLE_RATE as u64 + self.rate_in as u64 / 2) / self.rate_in as u64
    }

    /// Resample `input` and append whatever complete output is ready to `out`.
    pub fn push(&mut self, input: &[f32], out: &mut Vec<f32>) {
        self.frames_in += input.len() as u64;
        let Some(fft) = self.inner.as_mut() else {
            out.extend_from_slice(input);
            self.frames_out += input.len() as u64;
            return;
        };
        self.pending.extend_from_slice(input);
        let mut consumed = 0;
        while self.pending.len() - consumed >= self.chunk_in {
            let block = &self.pending[consumed..consumed + self.chunk_in];
            let produced = run_chunk(fft, block, None, &mut self.out_buf);
            consumed += self.chunk_in;
            emit(
                &self.out_buf[..produced],
                &mut self.skip,
                &mut self.frames_out,
                out,
            );
        }
        self.pending.drain(..consumed);
    }

    /// Emit everything still buffered so the take's output length is exact, then return to
    /// a fresh state. Flushing pads with silence, so the next take must not continue the
    /// old filter state; resetting costs one delay's worth of output, which lands in the
    /// pre-roll where nothing depends on it.
    pub fn flush(&mut self, out: &mut Vec<f32>) {
        let target = self.expected_out(self.frames_in);
        if let Some(fft) = self.inner.as_mut() {
            let mut guard = 0;
            while self.frames_out < target && guard < 64 {
                guard += 1;
                let n = self.pending.len().min(self.chunk_in);
                let idx = Indexing::new().partial_len(n);
                let produced = if n == 0 {
                    let zeros = vec![0.0; self.chunk_in];
                    run_chunk(fft, &zeros, Some(&idx), &mut self.out_buf)
                } else {
                    let block: Vec<f32> = self.pending.drain(..n).collect();
                    run_chunk(fft, &block, Some(&idx), &mut self.out_buf)
                };
                let room = (target - self.frames_out) as usize;
                let take = produced.min(room + self.skip);
                emit(
                    &self.out_buf[..take],
                    &mut self.skip,
                    &mut self.frames_out,
                    out,
                );
            }
            fft.reset();
        }
        self.pending.clear();
        self.skip = self.delay;
        self.frames_in = 0;
        self.frames_out = 0;
    }
}

fn run_chunk(fft: &mut Fft<f32>, block: &[f32], idx: Option<&Indexing>, out: &mut [f32]) -> usize {
    let frames_out = out.len();
    let input = InterleavedSlice::new(block, 1, block.len()).expect("mono slice sized to itself");
    let mut output = InterleavedSlice::new_mut(out, 1, frames_out).expect("mono slice");
    match fft.process_into_buffer(&input, &mut output, idx) {
        Ok((_, produced)) => produced,
        // Buffers are sized from the resampler's own reported maxima, so this cannot fire
        // short of a rubato bug; dropping one chunk beats panicking the audio worker.
        Err(_) => 0,
    }
}

fn emit(block: &[f32], skip: &mut usize, frames_out: &mut u64, out: &mut Vec<f32>) {
    let s = (*skip).min(block.len());
    *skip -= s;
    let rest = &block[s..];
    out.extend_from_slice(rest);
    *frames_out += rest.len() as u64;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sine(rate: u32, hz: f32, secs: f32) -> Vec<f32> {
        let n = (rate as f32 * secs) as usize;
        (0..n)
            .map(|i| (i as f32 * hz * std::f32::consts::TAU / rate as f32).sin() * 0.5)
            .collect()
    }

    fn run(rate: u32, input: &[f32], block: usize) -> Vec<f32> {
        let mut r = StreamResampler::new(rate).unwrap();
        let mut out = Vec::new();
        for b in input.chunks(block) {
            r.push(b, &mut out);
        }
        r.flush(&mut out);
        out
    }

    #[test]
    fn exact_length_48k() {
        let input = sine(48_000, 440.0, 1.0);
        let out = run(48_000, &input, 480);
        assert_eq!(out.len(), 16_000);
    }

    #[test]
    fn exact_length_44k1_with_odd_blocks() {
        let input = sine(44_100, 440.0, 1.37);
        let out = run(44_100, &input, 441 * 3 + 7);
        let expected = (input.len() as f64 * 16_000.0 / 44_100.0).round() as usize;
        assert!(
            out.len().abs_diff(expected) <= 1,
            "{} vs {expected}",
            out.len()
        );
    }

    #[test]
    fn passthrough_at_16k() {
        let input = sine(16_000, 440.0, 0.5);
        assert_eq!(run(16_000, &input, 100), input);
    }

    #[test]
    fn preserves_tone_and_alignment() {
        // A 440 Hz tone must come out as a 440 Hz tone at the same phase: a wrong delay
        // compensation would shift every VAD timestamp.
        let input = sine(48_000, 440.0, 1.0);
        let out = run(48_000, &input, 1000);
        let reference = sine(16_000, 440.0, 1.0);
        let mid = &out[2000..14000];
        let err: f32 = mid
            .iter()
            .zip(&reference[2000..14000])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0, f32::max);
        assert!(err < 0.02, "max error {err}");
    }

    #[test]
    fn flush_resets_for_next_take() {
        let mut r = StreamResampler::new(48_000).unwrap();
        let mut out = Vec::new();
        r.push(&sine(48_000, 440.0, 0.25), &mut out);
        r.flush(&mut out);
        assert_eq!(out.len(), 4000);
        out.clear();
        r.push(&sine(48_000, 440.0, 0.5), &mut out);
        r.flush(&mut out);
        assert_eq!(out.len(), 8000);
    }
}
