use std::path::Path;
use std::time::Duration;

use anyhow::{Context, Result, bail};
use wl_audio::resample::StreamResampler;
use wl_core::recorder::{Recorder, RecorderError, Recording};
use wl_core::stt::SAMPLE_RATE;

pub fn read_16k_mono(path: &Path) -> Result<Vec<f32>> {
    let mut reader =
        hound::WavReader::open(path).with_context(|| format!("opening {}", path.display()))?;
    let spec = reader.spec();
    let interleaved: Vec<f32> = match spec.sample_format {
        hound::SampleFormat::Float => reader
            .samples::<f32>()
            .collect::<Result<_, _>>()
            .context("reading samples")?,
        hound::SampleFormat::Int => {
            let bits = spec.bits_per_sample;
            if !(8..=32).contains(&bits) {
                bail!("unsupported bit depth {bits}");
            }
            let scale = f64::from(1u32 << (bits - 1));
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| (f64::from(v) / scale) as f32))
                .collect::<Result<_, _>>()
                .context("reading samples")?
        }
    };
    let ch = usize::from(spec.channels.max(1));
    let mono: Vec<f32> = interleaved
        .chunks_exact(ch)
        .map(|f| f.iter().sum::<f32>() / ch as f32)
        .collect();
    let mut rs = StreamResampler::new(spec.sample_rate).map_err(anyhow::Error::msg)?;
    let mut out = Vec::with_capacity(mono.len());
    rs.push(&mono, &mut out);
    rs.flush(&mut out);
    Ok(out)
}

/// Hands out the same clip on every `stop()`.
pub struct WavRecorder {
    pcm: Vec<f32>,
    recording: bool,
}

impl WavRecorder {
    pub fn new(pcm: Vec<f32>) -> Self {
        Self {
            pcm,
            recording: false,
        }
    }
}

impl Recorder for WavRecorder {
    fn start(&mut self) -> Result<(), RecorderError> {
        self.recording = true;
        Ok(())
    }

    fn stop(&mut self) -> Result<Recording, RecorderError> {
        if !std::mem::take(&mut self.recording) {
            return Err(RecorderError::State("not recording"));
        }
        Ok(Recording {
            pcm: self.pcm.clone(),
            sample_rate: SAMPLE_RATE,
            duration: Duration::from_secs_f64(self.pcm.len() as f64 / f64::from(SAMPLE_RATE)),
            ..Default::default()
        })
    }

    fn cancel(&mut self) {
        self.recording = false;
    }

    fn level(&self) -> f32 {
        0.0
    }

    fn is_warm(&self) -> bool {
        true
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stereo_48k_becomes_16k_mono_of_the_same_length() {
        let path = std::env::temp_dir().join(format!("wl-wav-test-{}.wav", std::process::id()));
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 48_000,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut w = hound::WavWriter::create(&path, spec).unwrap();
        for i in 0..48_000 {
            let v = ((i as f32 * 0.01).sin() * 8000.0) as i16;
            w.write_sample(v).unwrap();
            w.write_sample(v).unwrap();
        }
        w.finalize().unwrap();
        let pcm = read_16k_mono(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert!((pcm.len() as i64 - 16_000).abs() <= 1, "{}", pcm.len());
    }

    #[test]
    fn recorder_returns_the_clip_once_per_start() {
        let mut r = WavRecorder::new(vec![0.1; 1600]);
        assert!(r.stop().is_err());
        r.start().unwrap();
        let rec = r.stop().unwrap();
        assert_eq!(rec.pcm.len(), 1600);
        assert_eq!(rec.duration, Duration::from_millis(100));
    }
}
