//! Voice activity detection on the 16 kHz stream.
//!
//! [`EnergyVad`] is the always-available baseline. A neural detector (Silero) is the
//! intended default once ONNX Runtime is in the build; it is deliberately not here,
//! because this crate must not pull a second native runtime into default builds. It
//! plugs in by implementing [`Vad`]:
//!
//! - it lives next to whichever engine already links ONNX Runtime, behind the same
//!   non-default feature, so the single-runtime rule holds;
//! - Silero consumes fixed 512-sample windows at 16 kHz, while [`Vad::push`] takes any
//!   length, so the implementation buffers the remainder between calls;
//! - it keeps its recurrent state across `push` calls and clears it in [`Vad::reset`];
//! - it reports times with the same origin as here: samples since the last reset.
//!
//! [`trim_silence`] and [`has_speech`] work on the event list alone, so they do not care
//! which detector produced it.

use std::time::Duration;

use wl_core::stt::SAMPLE_RATE;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VadEvent {
    /// Speech begins at this offset from the first sample pushed since the last reset.
    SpeechStart { at: Duration },
    /// Speech ended at this offset (end of the last speech frame, before hangover).
    SpeechEnd { at: Duration },
}

pub trait Vad: Send {
    /// Feed 16 kHz mono samples; returns boundaries that became certain during this call.
    /// A start is reported only after enough speech to confirm it, and an end only after
    /// the hangover, so events lag the audio but their `at` is exact.
    fn push(&mut self, pcm16k: &[f32]) -> Vec<VadEvent>;
    /// Forget all state; the next sample pushed is time zero.
    fn reset(&mut self);
}

/// Tuning for [`EnergyVad`].
#[derive(Debug, Clone, PartialEq)]
pub struct EnergyVadConfig {
    /// Analysis frame length.
    pub frame: Duration,
    /// Speech threshold as a multiple of the tracked noise floor RMS.
    pub ratio: f32,
    /// Absolute RMS below which nothing counts as speech, whatever the floor: a
    /// digitally silent input would otherwise make the ratio trigger on dither.
    pub min_rms: f32,
    /// Speech needed above threshold before a start is declared. Rejects clicks and
    /// keyboard taps.
    pub min_speech: Duration,
    /// Silence needed before an end is declared. Short enough to split sentences for
    /// segment pre-transcription, long enough not to split words at plosive closures.
    pub hangover: Duration,
    /// Per-frame fraction by which the floor rises towards a louder non-speech frame.
    /// Falling is immediate, so a quiet room is learned at once; rising is slow, so a
    /// long utterance is not learned as noise.
    pub floor_rise: f32,
    /// Upper bound on the floor learned from the very first frame.
    pub max_initial_floor: f32,
}

impl Default for EnergyVadConfig {
    fn default() -> Self {
        Self {
            frame: Duration::from_millis(20),
            ratio: 3.0,
            min_rms: 0.004,
            min_speech: Duration::from_millis(60),
            hangover: Duration::from_millis(300),
            floor_rise: 0.02,
            max_initial_floor: 0.01,
        }
    }
}

/// Adaptive-threshold RMS detector with hangover.
#[derive(Debug, Clone)]
pub struct EnergyVad {
    cfg: EnergyVadConfig,
    frame_len: usize,
    min_speech_frames: usize,
    hangover_frames: usize,
    buf: Vec<f32>,
    frames_seen: usize,
    floor: Option<f32>,
    in_speech: bool,
    /// Consecutive loud frames while not in speech, and where they began.
    run: usize,
    run_start: usize,
    /// Consecutive quiet frames while in speech, and the frame after the last loud one.
    quiet: usize,
    last_loud_end: usize,
}

/// About -80 dBFS: below any real microphone's self-noise.
const DIGITAL_SILENCE: f32 = 1e-4;

fn frames_for(d: Duration, frame: Duration) -> usize {
    (d.as_secs_f64() / frame.as_secs_f64()).ceil().max(1.0) as usize
}

impl EnergyVad {
    pub fn new(cfg: EnergyVadConfig) -> Self {
        let frame_len = ((cfg.frame.as_secs_f64() * SAMPLE_RATE as f64) as usize).max(1);
        Self {
            min_speech_frames: frames_for(cfg.min_speech, cfg.frame),
            hangover_frames: frames_for(cfg.hangover, cfg.frame),
            frame_len,
            cfg,
            buf: Vec::new(),
            frames_seen: 0,
            floor: None,
            in_speech: false,
            run: 0,
            run_start: 0,
            quiet: 0,
            last_loud_end: 0,
        }
    }

    /// Close an open segment at the end of input. Not part of [`Vad`] because
    /// [`trim_silence`] already treats an unmatched start as running to the end.
    pub fn finish(&mut self) -> Vec<VadEvent> {
        if self.in_speech {
            self.in_speech = false;
            vec![VadEvent::SpeechEnd {
                at: self.at(self.last_loud_end),
            }]
        } else {
            Vec::new()
        }
    }

    fn at(&self, frame: usize) -> Duration {
        Duration::from_secs_f64((frame * self.frame_len) as f64 / SAMPLE_RATE as f64)
    }

    fn threshold(&self) -> f32 {
        (self.floor.unwrap_or(0.0) * self.cfg.ratio).max(self.cfg.min_rms)
    }

    fn frame(&mut self, rms: f32, out: &mut Vec<VadEvent>) {
        let idx = self.frames_seen;
        self.frames_seen += 1;
        // Digital silence (a stream's first buffers, dropouts on virtual devices) says
        // nothing about the room. Learning it as the floor would pin the threshold to
        // `min_rms` and make ordinary background noise read as speech, observed live on
        // an RDP-redirected microphone.
        let digital_silence = rms < DIGITAL_SILENCE;
        if self.floor.is_none() && !digital_silence {
            // The first real frame seeds the floor, capped so that a recording which opens
            // mid-word (cold start, no pre-roll) does not learn speech as the noise floor.
            self.floor = Some(rms.min(self.cfg.max_initial_floor));
        }
        let loud = rms > self.threshold();
        if self.in_speech {
            if loud {
                self.quiet = 0;
                self.last_loud_end = idx + 1;
            } else {
                self.quiet += 1;
                if self.quiet >= self.hangover_frames {
                    self.in_speech = false;
                    self.quiet = 0;
                    out.push(VadEvent::SpeechEnd {
                        at: self.at(self.last_loud_end),
                    });
                }
            }
            return;
        }
        if loud {
            if self.run == 0 {
                self.run_start = idx;
            }
            self.run += 1;
            if self.run >= self.min_speech_frames {
                self.in_speech = true;
                self.run = 0;
                self.quiet = 0;
                self.last_loud_end = idx + 1;
                out.push(VadEvent::SpeechStart {
                    at: self.at(self.run_start),
                });
            }
            return;
        }
        self.run = 0;
        if digital_silence {
            return;
        }
        let f = self.floor.unwrap_or(rms);
        self.floor = Some(if rms < f {
            rms
        } else {
            f + (rms - f) * self.cfg.floor_rise
        });
    }
}

impl Default for EnergyVad {
    fn default() -> Self {
        Self::new(EnergyVadConfig::default())
    }
}

impl Vad for EnergyVad {
    fn push(&mut self, pcm16k: &[f32]) -> Vec<VadEvent> {
        let mut out = Vec::new();
        let mut rest = pcm16k;
        if !self.buf.is_empty() {
            let need = self.frame_len - self.buf.len();
            let take = need.min(rest.len());
            self.buf.extend_from_slice(&rest[..take]);
            rest = &rest[take..];
            if self.buf.len() < self.frame_len {
                return out;
            }
            let rms = rms(&self.buf);
            self.buf.clear();
            self.frame(rms, &mut out);
        }
        let mut chunks = rest.chunks_exact(self.frame_len);
        for f in &mut chunks {
            self.frame(rms(f), &mut out);
        }
        self.buf.extend_from_slice(chunks.remainder());
        out
    }

    fn reset(&mut self) {
        *self = Self::new(self.cfg.clone());
    }
}

fn rms(x: &[f32]) -> f32 {
    if x.is_empty() {
        return 0.0;
    }
    (x.iter().map(|s| s * s).sum::<f32>() / x.len() as f32).sqrt()
}

/// True if any speech was detected. A recording without speech must not reach an
/// engine: Whisper and the int8 Parakeet variant both invent text on silence.
pub fn has_speech(events: &[VadEvent]) -> bool {
    events
        .iter()
        .any(|e| matches!(e, VadEvent::SpeechStart { .. }))
}

/// Cut leading and trailing silence, keeping `pad` around the speech. Pauses between
/// segments are kept: engines use them as punctuation cues. An unmatched start runs to
/// the end of `pcm`. Returns an empty vector when there is no speech.
pub fn trim_silence(pcm: &[f32], events: &[VadEvent], pad: Duration) -> Vec<f32> {
    let to_idx = |d: Duration| (d.as_secs_f64() * SAMPLE_RATE as f64).round() as usize;
    let first = events.iter().find_map(|e| match e {
        VadEvent::SpeechStart { at } => Some(*at),
        _ => None,
    });
    let Some(first) = first else {
        return Vec::new();
    };
    let mut open = false;
    let mut last_end = None;
    for e in events {
        match e {
            VadEvent::SpeechStart { .. } => open = true,
            VadEvent::SpeechEnd { at } => {
                open = false;
                last_end = Some(*at);
            }
        }
    }
    let pad = to_idx(pad);
    let start = to_idx(first).saturating_sub(pad).min(pcm.len());
    let end = if open {
        pcm.len()
    } else {
        last_end
            .map(|e| to_idx(e).saturating_add(pad))
            .unwrap_or(pcm.len())
            .min(pcm.len())
    };
    pcm[start..end.max(start)].to_vec()
}

#[cfg(test)]
mod tests {
    use super::*;

    const SR: usize = SAMPLE_RATE as usize;

    /// Low-level noise, deterministic.
    fn noise(n: usize, amp: f32, seed: &mut u32) -> Vec<f32> {
        (0..n)
            .map(|_| {
                *seed = seed.wrapping_mul(1_664_525).wrapping_add(1_013_904_223);
                ((*seed >> 8) as f32 / (1u32 << 24) as f32 * 2.0 - 1.0) * amp
            })
            .collect()
    }

    fn tone(n: usize, amp: f32) -> Vec<f32> {
        (0..n)
            .map(|i| (i as f32 * 220.0 * std::f32::consts::TAU / SR as f32).sin() * amp)
            .collect()
    }

    fn ms(n: usize) -> usize {
        n * SR / 1000
    }

    fn d(ms: u64) -> Duration {
        Duration::from_millis(ms)
    }

    /// 500 ms noise, 800 ms tone, 600 ms noise, 400 ms tone, 1000 ms noise.
    fn bursts() -> Vec<f32> {
        let mut s = 7;
        let mut v = noise(ms(500), 0.003, &mut s);
        v.extend(tone(ms(800), 0.2));
        v.extend(noise(ms(600), 0.003, &mut s));
        v.extend(tone(ms(400), 0.2));
        v.extend(noise(ms(1000), 0.003, &mut s));
        v
    }

    fn run(pcm: &[f32], block: usize) -> Vec<VadEvent> {
        let mut vad = EnergyVad::default();
        let mut ev = Vec::new();
        for b in pcm.chunks(block) {
            ev.extend(vad.push(b));
        }
        ev.extend(vad.finish());
        ev
    }

    #[test]
    fn finds_two_bursts_with_exact_times() {
        let ev = run(&bursts(), 512);
        assert_eq!(
            ev,
            vec![
                VadEvent::SpeechStart { at: d(500) },
                VadEvent::SpeechEnd { at: d(1300) },
                VadEvent::SpeechStart { at: d(1900) },
                VadEvent::SpeechEnd { at: d(2300) },
            ]
        );
    }

    #[test]
    fn block_size_does_not_change_events() {
        let pcm = bursts();
        assert_eq!(run(&pcm, 512), run(&pcm, 97));
        assert_eq!(run(&pcm, 512), run(&pcm, pcm.len()));
    }

    #[test]
    fn short_gap_is_bridged_by_hangover() {
        let mut s = 1;
        let mut v = noise(ms(300), 0.003, &mut s);
        v.extend(tone(ms(300), 0.2));
        v.extend(noise(ms(200), 0.003, &mut s));
        v.extend(tone(ms(300), 0.2));
        v.extend(noise(ms(500), 0.003, &mut s));
        let ev = run(&v, 320);
        assert_eq!(
            ev,
            vec![
                VadEvent::SpeechStart { at: d(300) },
                VadEvent::SpeechEnd { at: d(1100) },
            ]
        );
    }

    #[test]
    fn click_is_not_speech() {
        let mut s = 3;
        let mut v = noise(ms(300), 0.003, &mut s);
        v.extend(tone(ms(20), 0.5));
        v.extend(noise(ms(500), 0.003, &mut s));
        assert!(!has_speech(&run(&v, 320)));
    }

    #[test]
    fn silence_and_noise_have_no_speech() {
        assert!(!has_speech(&run(&vec![0.0; SR * 2], 320)));
        let mut s = 5;
        assert!(!has_speech(&run(&noise(SR * 2, 0.01, &mut s), 320)));
    }

    #[test]
    fn leading_digital_silence_does_not_pin_the_floor() {
        // Background just above `min_rms`, after a cold stream's zero buffers.
        let mut s = 11;
        let mut v = vec![0.0; ms(200)];
        v.extend(noise(ms(1500), 0.009, &mut s));
        assert!(!has_speech(&run(&v, 320)));
        v.extend(tone(ms(400), 0.2));
        v.extend(noise(ms(500), 0.009, &mut s));
        let ev = run(&v, 320);
        assert_eq!(ev[0], VadEvent::SpeechStart { at: d(1700) });
    }

    #[test]
    fn floor_adapts_to_louder_room() {
        // Steady fan noise well above min_rms must not read as speech, and speech over it
        // must still be found.
        let mut s = 9;
        let mut v = noise(ms(1000), 0.03, &mut s);
        v.extend(tone(ms(500), 0.3));
        v.extend(noise(ms(800), 0.03, &mut s));
        let ev = run(&v, 320);
        assert_eq!(ev.len(), 2);
        assert_eq!(ev[0], VadEvent::SpeechStart { at: d(1000) });
    }

    #[test]
    fn reset_restarts_time() {
        let pcm = bursts();
        let mut vad = EnergyVad::default();
        vad.push(&pcm[..ms(700)]);
        vad.reset();
        let ev = vad.push(&pcm);
        assert_eq!(ev[0], VadEvent::SpeechStart { at: d(500) });
    }

    #[test]
    fn trim_pads_and_clamps() {
        let pcm: Vec<f32> = (0..SR * 3).map(|i| i as f32).collect();
        let ev = [
            VadEvent::SpeechStart { at: d(500) },
            VadEvent::SpeechEnd { at: d(1300) },
            VadEvent::SpeechStart { at: d(1900) },
            VadEvent::SpeechEnd { at: d(2300) },
        ];
        let t = trim_silence(&pcm, &ev, d(200));
        assert_eq!(t.first().copied(), Some(ms(300) as f32));
        assert_eq!(t.len(), ms(2500) - ms(300));
        // Padding beyond the buffer clamps rather than panicking.
        let t = trim_silence(&pcm, &ev[..2], d(1000));
        assert_eq!(t.len(), ms(2300));
        assert_eq!(t[0], 0.0);
    }

    #[test]
    fn trim_open_segment_runs_to_end() {
        let pcm = vec![0.0; SR];
        let ev = [VadEvent::SpeechStart { at: d(400) }];
        assert_eq!(trim_silence(&pcm, &ev, d(100)).len(), ms(700));
    }

    #[test]
    fn trim_without_speech_is_empty() {
        assert!(trim_silence(&[0.0; 100], &[], d(200)).is_empty());
    }

    #[test]
    fn resampled_sine_through_vad() {
        // The synthetic end-to-end path: 48 kHz capture, resampled, then detected.
        use crate::resample::StreamResampler;
        let rate = 48_000usize;
        let mut input = vec![0.0f32; rate / 2];
        input.extend(
            (0..rate).map(|i| (i as f32 * 300.0 * std::f32::consts::TAU / rate as f32).sin() * 0.3),
        );
        input.extend(vec![0.0f32; rate]);
        let mut r = StreamResampler::new(48_000).unwrap();
        let mut out = Vec::new();
        for b in input.chunks(480) {
            r.push(b, &mut out);
        }
        r.flush(&mut out);
        assert_eq!(out.len(), SR * 5 / 2);
        let ev = run(&out, 160);
        assert_eq!(ev.len(), 2, "{ev:?}");
        let VadEvent::SpeechStart { at } = ev[0] else {
            panic!()
        };
        assert!(at.abs_diff(d(500)) <= d(20), "{at:?}");
        let trimmed = trim_silence(&out, &ev, d(200));
        assert!(
            trimmed.len().abs_diff(ms(1400)) <= ms(40),
            "{}",
            trimmed.len()
        );
    }
}
