use std::path::Path;

use anyhow::{Context, Result, bail};
use rubato::audioadapter_buffers::direct::InterleavedSlice;
use rubato::{Fft, FixedSync, Resampler};

pub struct Pcm {
    pub samples: Vec<f32>,
    pub sample_rate: u32,
    pub channels: u16,
}

pub fn read_wav(path: &Path) -> Result<Pcm> {
    let mut reader =
        hound::WavReader::open(path).with_context(|| format!("opening {}", path.display()))?;
    let spec = reader.spec();
    let samples = match spec.sample_format {
        hound::SampleFormat::Float => reader.samples::<f32>().collect::<Result<Vec<_>, _>>()?,
        hound::SampleFormat::Int => {
            let bits = spec.bits_per_sample;
            if !(8..=32).contains(&bits) {
                bail!("unsupported bit depth {bits}");
            }
            reader
                .samples::<i32>()
                .map(|s| s.map(|v| int_to_f32(v, bits)))
                .collect::<Result<Vec<_>, _>>()?
        }
    };
    Ok(Pcm {
        samples,
        sample_rate: spec.sample_rate,
        channels: spec.channels,
    })
}

pub fn int_to_f32(v: i32, bits: u16) -> f32 {
    (f64::from(v) / f64::from(1u32 << (bits - 1))) as f32
}

pub fn downmix(interleaved: &[f32], channels: u16) -> Vec<f32> {
    let ch = usize::from(channels.max(1));
    if ch == 1 {
        return interleaved.to_vec();
    }
    interleaved
        .chunks_exact(ch)
        .map(|frame| frame.iter().sum::<f32>() / ch as f32)
        .collect()
}

pub fn resample(mono: &[f32], from: u32, to: u32) -> Result<Vec<f32>> {
    if from == to || mono.is_empty() {
        return Ok(mono.to_vec());
    }
    let mut resampler = Fft::<f32>::new(from as usize, to as usize, 1024, 1, FixedSync::Both)
        .context("building resampler")?;
    let input = InterleavedSlice::new(mono, 1, mono.len()).context("wrapping input")?;
    let out = resampler
        .process_all(&input, mono.len(), None)
        .context("resampling")?;
    Ok(out.take_data())
}

pub fn to_engine_pcm(pcm: &Pcm, target_rate: u32) -> Result<Vec<f32>> {
    let mono = downmix(&pcm.samples, pcm.channels);
    resample(&mono, pcm.sample_rate, target_rate)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn int_scaling() {
        assert_eq!(int_to_f32(0, 16), 0.0);
        assert_eq!(int_to_f32(-32768, 16), -1.0);
        assert_eq!(int_to_f32(16384, 16), 0.5);
        assert_eq!(int_to_f32(-8_388_608, 24), -1.0);
        assert!((int_to_f32(i32::MAX, 32) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn downmix_averages_frames() {
        assert_eq!(downmix(&[1.0, 0.0, 0.5, 0.5], 2), vec![0.5, 0.5]);
        assert_eq!(downmix(&[0.1, 0.2], 1), vec![0.1, 0.2]);
    }

    #[test]
    fn same_rate_is_passthrough() {
        let x = vec![0.1, -0.2, 0.3];
        assert_eq!(resample(&x, 16_000, 16_000).unwrap(), x);
    }

    #[test]
    fn resample_48k_to_16k_keeps_duration_and_tone() {
        let n = 48_000;
        let tone: Vec<f32> = (0..n)
            .map(|i| (i as f32 * 440.0 * std::f32::consts::TAU / 48_000.0).sin() * 0.5)
            .collect();
        let out = resample(&tone, 48_000, 16_000).unwrap();
        assert!((out.len() as i64 - 16_000).abs() <= 2, "len {}", out.len());
        let rms = (out[2000..14000].iter().map(|v| v * v).sum::<f32>() / 12000.0).sqrt();
        assert!((rms - 0.5 / 2f32.sqrt()).abs() < 0.02, "rms {rms}");
    }

    #[test]
    fn round_trip_through_a_wav_file() {
        let dir = std::env::temp_dir().join(format!("bench-stt-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("stereo44k.wav");
        let spec = hound::WavSpec {
            channels: 2,
            sample_rate: 44_100,
            bits_per_sample: 16,
            sample_format: hound::SampleFormat::Int,
        };
        let mut w = hound::WavWriter::create(&path, spec).unwrap();
        for _ in 0..44_100 {
            w.write_sample(16384i16).unwrap();
            w.write_sample(0i16).unwrap();
        }
        w.finalize().unwrap();

        let pcm = read_wav(&path).unwrap();
        assert_eq!((pcm.channels, pcm.sample_rate), (2, 44_100));
        let out = to_engine_pcm(&pcm, 16_000).unwrap();
        assert!((out.len() as i64 - 16_000).abs() <= 2, "len {}", out.len());
        assert!((out[8000] - 0.25).abs() < 0.01, "{}", out[8000]);
        std::fs::remove_dir_all(&dir).ok();
    }
}
