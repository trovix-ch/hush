//! Start, stop and error cues through `PlaySoundW` (§2).
//!
//! Why not an audio crate: three ~80 ms blips do not justify a second audio stack next to
//! the capture path, and `PlaySoundW(SND_MEMORY | SND_ASYNC)` returns immediately. Why
//! the WAVs are synthesised once into statics instead of shipped as files: `SND_ASYNC`
//! keeps reading the buffer after the call returns, so it must live for the whole
//! process, and generating them avoids binary assets in the tree. Why the start cue is
//! played by the caller *before* the microphone opens: otherwise it lands in the
//! transcription.

use std::sync::LazyLock;

use windows::Win32::Media::Audio::{PlaySoundW, SND_ASYNC, SND_MEMORY, SND_NODEFAULT};
use windows::core::PCWSTR;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Cue {
    Start,
    Stop,
    /// Recording discarded (Escape): one soft falling tone.
    Cancel,
    Error,
}

impl From<wl_core::notify::Sound> for Cue {
    fn from(s: wl_core::notify::Sound) -> Self {
        match s {
            wl_core::notify::Sound::Start => Cue::Start,
            wl_core::notify::Sound::Stop => Cue::Stop,
            wl_core::notify::Sound::Cancel => Cue::Cancel,
            wl_core::notify::Sound::Error => Cue::Error,
        }
    }
}

const RATE: u32 = 22_050;

static START: LazyLock<Vec<u8>> =
    LazyLock::new(|| wav(&tones(&[(660.0, 40), (880.0, 40)], 0.16, false)));
static STOP: LazyLock<Vec<u8>> =
    LazyLock::new(|| wav(&tones(&[(880.0, 40), (660.0, 40)], 0.16, false)));
static CANCEL: LazyLock<Vec<u8>> =
    LazyLock::new(|| wav(&tones(&[(523.0, 35), (392.0, 45)], 0.12, false)));
static ERROR: LazyLock<Vec<u8>> = LazyLock::new(|| wav(&tones(&[(196.0, 90)], 0.12, true)));

/// Plays `cue` asynchronously; a new cue cuts off the previous one.
pub fn play(cue: Cue) {
    let data: &'static [u8] = match cue {
        Cue::Start => &START,
        Cue::Stop => &STOP,
        Cue::Cancel => &CANCEL,
        Cue::Error => &ERROR,
    };
    // SAFETY: with SND_MEMORY the "name" is a pointer to a complete WAV image; it is a
    // 'static buffer, so it outlives the asynchronous playback.
    let ok = unsafe {
        PlaySoundW(
            PCWSTR(data.as_ptr() as *const u16),
            None,
            SND_MEMORY | SND_ASYNC | SND_NODEFAULT,
        )
    };
    if !ok.as_bool() {
        tracing::debug!(?cue, "PlaySoundW refused the cue");
    }
}

/// Mono 16-bit samples for a sequence of (frequency, milliseconds) tones.
fn tones(parts: &[(f32, u32)], amplitude: f32, buzz: bool) -> Vec<i16> {
    let mut out = Vec::new();
    for &(freq, ms) in parts {
        let n = (RATE * ms / 1000) as usize;
        let fade = (RATE as usize * 6 / 1000).min(n / 2);
        for i in 0..n {
            let t = i as f32 / RATE as f32;
            let phase = 2.0 * std::f32::consts::PI * freq * t;
            let mut s = phase.sin();
            if buzz {
                // Odd harmonics give a soft buzz without a harsh square wave.
                s = s + (3.0 * phase).sin() / 3.0 + (5.0 * phase).sin() / 5.0;
                s *= 0.8;
            }
            let env = if i < fade {
                i as f32 / fade as f32
            } else if i >= n - fade {
                (n - i) as f32 / fade as f32
            } else {
                1.0
            };
            out.push((s * env * amplitude * i16::MAX as f32) as i16);
        }
    }
    out
}

fn wav(samples: &[i16]) -> Vec<u8> {
    let data_len = (samples.len() * 2) as u32;
    let mut v = Vec::with_capacity(44 + data_len as usize);
    v.extend_from_slice(b"RIFF");
    v.extend_from_slice(&(36 + data_len).to_le_bytes());
    v.extend_from_slice(b"WAVEfmt ");
    v.extend_from_slice(&16u32.to_le_bytes());
    v.extend_from_slice(&1u16.to_le_bytes()); // PCM
    v.extend_from_slice(&1u16.to_le_bytes()); // mono
    v.extend_from_slice(&RATE.to_le_bytes());
    v.extend_from_slice(&(RATE * 2).to_le_bytes());
    v.extend_from_slice(&2u16.to_le_bytes());
    v.extend_from_slice(&16u16.to_le_bytes());
    v.extend_from_slice(b"data");
    v.extend_from_slice(&data_len.to_le_bytes());
    for s in samples {
        v.extend_from_slice(&s.to_le_bytes());
    }
    v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn wav_headers_are_consistent() {
        for w in [&*START, &*STOP, &*CANCEL, &*ERROR] {
            assert_eq!(&w[0..4], b"RIFF");
            assert_eq!(&w[8..16], b"WAVEfmt ");
            let riff = u32::from_le_bytes(w[4..8].try_into().unwrap()) as usize;
            assert_eq!(riff + 8, w.len());
            let data = u32::from_le_bytes(w[40..44].try_into().unwrap()) as usize;
            assert_eq!(data + 44, w.len());
        }
    }

    #[test]
    fn cues_are_short_and_quiet() {
        for (w, max_ms) in [(&*START, 90), (&*STOP, 90), (&*CANCEL, 90), (&*ERROR, 100)] {
            let samples = (w.len() - 44) / 2;
            let ms = samples as u32 * 1000 / RATE;
            assert!((60..=max_ms).contains(&ms), "{ms} ms");
            let peak = w[44..]
                .chunks_exact(2)
                .map(|c| i16::from_le_bytes([c[0], c[1]]).unsigned_abs())
                .max()
                .unwrap();
            assert!(peak < i16::MAX as u16 / 4, "peak {peak}");
        }
    }

    #[test]
    fn ends_fade_to_silence() {
        let s = tones(&[(660.0, 40)], 0.16, false);
        assert_eq!(s[0], 0);
        assert!(s.last().unwrap().unsigned_abs() < 200);
    }

    #[test]
    fn play_does_not_block() {
        let t = std::time::Instant::now();
        play(Cue::Start);
        assert!(t.elapsed() < std::time::Duration::from_millis(200));
    }
}
