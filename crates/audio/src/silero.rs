//! Silero VAD on tract, which is pure Rust, so the neural detector adds no second native
//! runtime to the process.

use std::path::Path;
use std::sync::Arc;
use std::time::Duration;

use hush_core::stt::SAMPLE_RATE;
use tract_onnx::prelude::*;

use crate::vad::{Vad, VadEvent};

mod fixed_if;

/// Tag v6.2 of snakers4/silero-vad; the file on `master` has changed before.
pub const MODEL_FILE: &str = "silero_vad.onnx";
pub const MODEL_URL: &str = "https://raw.githubusercontent.com/snakers4/silero-vad/v6.2/src/silero_vad/data/silero_vad.onnx";
pub const MODEL_SIZE: u64 = 2_327_524;
pub const MODEL_SHA256: &str = "1a153a22f4509e292a94e67d6f9b85e8deb25b4988682b7e174c65279d8788e3";
pub const MODEL_LICENSE: &str = "MIT";
pub const MODEL_ATTRIBUTION: &str =
    "Silero VAD by Silero Team (https://github.com/snakers4/silero-vad), licensed MIT.";

const CHUNK: usize = 512;
/// The v5+ graph expects the last 64 samples of the previous chunk in front of each new
/// one; without them the probabilities drift from the reference.
const CONTEXT: usize = 64;
const CHUNK_TIME: Duration = Duration::from_millis(32);

#[derive(Debug, thiserror::Error)]
#[error("silero VAD: {0}")]
pub struct SileroError(String);

#[derive(Debug, Clone, PartialEq)]
pub struct SileroConfig {
    pub threshold: f32,
    /// Speech continues until the probability falls below this; the gap to `threshold`
    /// keeps a wavering probability from toggling.
    pub neg_threshold: f32,
    pub min_speech: Duration,
    pub hangover: Duration,
}

impl Default for SileroConfig {
    fn default() -> Self {
        Self {
            threshold: 0.5,
            neg_threshold: 0.35,
            min_speech: Duration::from_millis(96),
            hangover: Duration::from_millis(96),
        }
    }
}

pub struct SileroVad {
    plan: Arc<TypedRunnableModel>,
    cfg: SileroConfig,
    min_speech_chunks: usize,
    hangover_chunks: usize,
    state: Tensor,
    context: [f32; CONTEXT],
    buf: Vec<f32>,
    chunks_seen: usize,
    in_speech: bool,
    loud_run: usize,
    loud_run_start: usize,
    quiet_run: usize,
    last_loud_end: usize,
}

fn chunks_for(d: Duration) -> usize {
    (d.as_secs_f64() / CHUNK_TIME.as_secs_f64()).ceil().max(1.0) as usize
}

fn zero_state() -> TractResult<Tensor> {
    Tensor::zero::<f32>(&[2, 1, 128])
}

fn err(e: impl std::fmt::Display) -> SileroError {
    SileroError(e.to_string())
}

impl SileroVad {
    pub fn load(model: &Path, cfg: SileroConfig) -> Result<Self, SileroError> {
        let mut onnx = tract_onnx::onnx();
        fixed_if::register(&mut onnx.op_register);
        let plan = onnx
            .model_for_path(model)
            .and_then(|m| m.with_input_fact(0, f32::fact([1, CHUNK + CONTEXT]).into()))
            .and_then(|m| m.with_input_fact(1, f32::fact([2, 1, 128]).into()))
            // A constant rate lets the graph's 8 kHz/16 kHz branch fold away.
            .and_then(|m| m.with_input_fact(2, InferenceFact::from(tensor0(SAMPLE_RATE as i64))))
            .and_then(|m| m.into_optimized())
            .and_then(|m| m.into_runnable())
            .map_err(|e| err(format!("{}: {e:#}", model.display())))?;
        Ok(Self {
            plan,
            min_speech_chunks: chunks_for(cfg.min_speech),
            hangover_chunks: chunks_for(cfg.hangover),
            cfg,
            state: zero_state().map_err(err)?,
            context: [0.0; CONTEXT],
            buf: Vec::with_capacity(CHUNK),
            chunks_seen: 0,
            in_speech: false,
            loud_run: 0,
            loud_run_start: 0,
            quiet_run: 0,
            last_loud_end: 0,
        })
    }

    /// Speech probability of one 32 ms chunk; advances the recurrent state.
    pub fn probability(&mut self, chunk: &[f32]) -> Result<f32, SileroError> {
        if chunk.len() != CHUNK {
            return Err(err(format!(
                "chunk of {} samples, need {CHUNK}",
                chunk.len()
            )));
        }
        let mut input = Vec::with_capacity(CHUNK + CONTEXT);
        input.extend_from_slice(&self.context);
        input.extend_from_slice(chunk);
        let input = Tensor::from_shape(&[1, CHUNK + CONTEXT], &input).map_err(err)?;
        let state = std::mem::take(&mut self.state);
        let out = self
            .plan
            .run(tvec!(
                input.into(),
                state.into(),
                tensor0(SAMPLE_RATE as i64).into()
            ))
            .map_err(err)?;
        let prob = out[0]
            .try_as_plain_ram()
            .ok()
            .and_then(|t| t.as_slice::<f32>().ok())
            .and_then(|s| s.first().copied())
            .ok_or_else(|| err("no probability in the output"))?;
        self.state = out[1].clone().into_tensor();
        self.context
            .copy_from_slice(&chunk[chunk.len() - CONTEXT..]);
        Ok(prob)
    }

    /// Not on [`Vad`] because an unmatched start already runs to the end of the take.
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

    fn at(&self, chunk: usize) -> Duration {
        CHUNK_TIME * chunk as u32
    }

    fn chunk(&mut self, prob: f32, out: &mut Vec<VadEvent>) {
        let idx = self.chunks_seen;
        self.chunks_seen += 1;
        if self.in_speech {
            if prob >= self.cfg.neg_threshold {
                self.quiet_run = 0;
                self.last_loud_end = idx + 1;
            } else {
                self.quiet_run += 1;
                if self.quiet_run >= self.hangover_chunks {
                    self.in_speech = false;
                    self.quiet_run = 0;
                    out.push(VadEvent::SpeechEnd {
                        at: self.at(self.last_loud_end),
                    });
                }
            }
            return;
        }
        if prob < self.cfg.threshold {
            self.loud_run = 0;
            return;
        }
        if self.loud_run == 0 {
            self.loud_run_start = idx;
        }
        self.loud_run += 1;
        if self.loud_run >= self.min_speech_chunks {
            self.in_speech = true;
            self.loud_run = 0;
            self.quiet_run = 0;
            self.last_loud_end = idx + 1;
            out.push(VadEvent::SpeechStart {
                at: self.at(self.loud_run_start),
            });
        }
    }

    fn classify(&mut self, chunk: &[f32], out: &mut Vec<VadEvent>) {
        // A failed run is counted as speech: a dropped word is worse than a padded segment.
        let prob = self.probability(chunk).unwrap_or_else(|e| {
            tracing::warn!(error = %e, "VAD inference failed; treating the chunk as speech");
            1.0
        });
        self.chunk(prob, out);
    }
}

impl Vad for SileroVad {
    fn push(&mut self, pcm16k: &[f32]) -> Vec<VadEvent> {
        let mut out = Vec::new();
        let mut rest = pcm16k;
        if !self.buf.is_empty() {
            let take = (CHUNK - self.buf.len()).min(rest.len());
            self.buf.extend_from_slice(&rest[..take]);
            rest = &rest[take..];
            if self.buf.len() < CHUNK {
                return out;
            }
            let chunk = std::mem::take(&mut self.buf);
            self.classify(&chunk, &mut out);
            self.buf = chunk;
            self.buf.clear();
        }
        let mut chunks = rest.chunks_exact(CHUNK);
        for c in &mut chunks {
            self.classify(c, &mut out);
        }
        self.buf.extend_from_slice(chunks.remainder());
        out
    }

    fn reset(&mut self) {
        self.state = zero_state().unwrap_or_default();
        self.context = [0.0; CONTEXT];
        self.buf.clear();
        self.chunks_seen = 0;
        self.in_speech = false;
        self.loud_run = 0;
        self.quiet_run = 0;
        self.last_loud_end = 0;
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vad::has_speech;
    use std::path::PathBuf;
    use std::time::Instant;

    /// Set `HUSH_SILERO_MODEL` to the downloaded `silero_vad.onnx` to run these.
    fn model() -> Option<PathBuf> {
        std::env::var_os("HUSH_SILERO_MODEL").map(PathBuf::from)
    }

    fn fixture(name: &str) -> Vec<f32> {
        let path = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../tools/bench-stt/fixtures")
            .join(name);
        let mut r = hound::WavReader::open(&path).unwrap();
        assert_eq!(r.spec().sample_rate, SAMPLE_RATE);
        assert_eq!(r.spec().channels, 1);
        match r.spec().sample_format {
            hound::SampleFormat::Int => r
                .samples::<i16>()
                .map(|s| s.unwrap() as f32 / 32768.0)
                .collect(),
            hound::SampleFormat::Float => r.samples::<f32>().map(Result::unwrap).collect(),
        }
    }

    fn events(vad: &mut SileroVad, pcm: &[f32]) -> Vec<VadEvent> {
        vad.reset();
        let mut ev = Vec::new();
        for b in pcm.chunks(800) {
            ev.extend(vad.push(b));
        }
        ev.extend(vad.finish());
        ev
    }

    #[test]
    fn speech_clips_have_speech_and_silence_has_none() {
        let Some(path) = model() else {
            eprintln!("skipped: HUSH_SILERO_MODEL is not set");
            return;
        };
        let mut vad = SileroVad::load(&path, SileroConfig::default()).unwrap();
        assert!(!has_speech(&events(&mut vad, &fixture("silence-02s.wav"))));
        for clip in [
            "jfk.wav",
            "tts-03s-question.wav",
            "tts-10s-fillers.wav",
            "tts-30s-dictation.wav",
        ] {
            let pcm = fixture(clip);
            let ev = events(&mut vad, &pcm);
            assert!(has_speech(&ev), "{clip}");
            let voiced: Duration = ev
                .chunks(2)
                .filter_map(|p| match p {
                    [
                        VadEvent::SpeechStart { at: a },
                        VadEvent::SpeechEnd { at: b },
                    ] => Some(*b - *a),
                    _ => None,
                })
                .sum();
            let total = Duration::from_secs_f64(pcm.len() as f64 / SAMPLE_RATE as f64);
            eprintln!("{clip}: {} spans, {voiced:?} of {total:?}", ev.len() / 2);
            assert!(voiced > total / 2, "{clip}: {ev:?}");
        }
    }

    #[test]
    fn dictation_closes_segments_before_release() {
        use hush_core::segment::{Segmenter, SegmenterConfig};
        let Some(path) = model() else {
            eprintln!("skipped: HUSH_SILERO_MODEL is not set");
            return;
        };
        let mut vad = SileroVad::load(&path, SileroConfig::default()).unwrap();
        let pcm = fixture("tts-30s-dictation.wav");
        let mut seg = Segmenter::new(&SegmenterConfig::default());
        let mut closed = Vec::new();
        // 50 ms blocks, as a driver polling the recorder would deliver them.
        for block in pcm.chunks(800) {
            let ev = vad.push(block);
            closed.extend(seg.push(block, &ev));
        }
        let tail = seg.finish(&[]);
        let secs = |v: &[hush_core::segment::Cut]| -> Vec<f32> {
            v.iter()
                .map(|s| s.pcm.len() as f32 / SAMPLE_RATE as f32)
                .collect()
        };
        eprintln!("closed {:?} tail {:?}", secs(&closed), secs(&tail));
        assert!(closed.len() >= 2, "{:?}", secs(&closed));
        assert!(tail.iter().map(|c| c.pcm.len()).sum::<usize>() < pcm.len() / 2);
    }

    #[test]
    fn chunk_cost() {
        let Some(path) = model() else {
            eprintln!("skipped: HUSH_SILERO_MODEL is not set");
            return;
        };
        let t = Instant::now();
        let mut vad = SileroVad::load(&path, SileroConfig::default()).unwrap();
        let load = t.elapsed();
        let pcm = fixture("tts-30s-dictation.wav");
        let mut costs: Vec<Duration> = pcm
            .chunks_exact(CHUNK)
            .map(|c| {
                let t = Instant::now();
                vad.probability(c).unwrap();
                t.elapsed()
            })
            .collect();
        costs.sort();
        eprintln!(
            "load {load:?}; per 32 ms chunk over {}: median {:?}, p99 {:?}",
            costs.len(),
            costs[costs.len() / 2],
            costs[costs.len() * 99 / 100]
        );
    }
}
