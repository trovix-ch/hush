use std::path::Path;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use parakeet_rs::{ExecutionConfig, ParakeetTDT, TimedToken, TimestampMode, Transcriber};
use wl_core::stt::{
    Backend, Caps, DecodeOptions, EngineInfo, SAMPLE_RATE, Segment, SttEngine, SttError, Transcript,
};

pub const ENGINE_ID: &str = "parakeet-tdt-0.6b-v3";

/// Per the model card.
const LANGUAGES: [&str; 25] = [
    "bg", "cs", "da", "de", "el", "en", "es", "et", "fi", "fr", "hr", "hu", "it", "lt", "lv", "mt",
    "nl", "pl", "pt", "ro", "ru", "sk", "sl", "sv", "uk",
];

/// Shorter than one STFT window (25 ms) the feature extractor has nothing to work on.
const MIN_SAMPLES: usize = 400;

#[derive(Debug, Clone)]
pub struct ParakeetOptions {
    pub backend: Backend,
    /// Off by default: measured slightly slower on an idle GPU, slightly faster on a GPU
    /// shared with another model.
    pub joint_on_cpu: bool,
    pub intra_threads: Option<usize>,
    /// DXGI adapter index.
    pub gpu_device: Option<i32>,
}

impl ParakeetOptions {
    pub fn new(backend: Backend) -> Self {
        Self {
            backend,
            joint_on_cpu: false,
            intra_threads: None,
            gpu_device: None,
        }
    }
}

pub struct ParakeetEngine {
    model: ParakeetTDT,
    info: EngineInfo,
}

impl ParakeetEngine {
    /// Fails rather than falling back to the CPU when a GPU backend does not load.
    pub fn new(model_dir: &Path, backend: Backend) -> Result<Self, SttError> {
        Self::with_options(model_dir, ParakeetOptions::new(backend))
    }

    pub fn with_options(model_dir: &Path, opts: ParakeetOptions) -> Result<Self, SttError> {
        let threads = opts.intra_threads.unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| (n.get() / 2).max(1))
                .unwrap_or(4)
        });
        let cpu = ExecutionConfig::new().with_intra_threads(threads);

        let backend_error: Arc<Mutex<Option<String>>> = Arc::default();
        let (encoder, joint) = match opts.backend {
            Backend::Cpu => (cpu.clone(), cpu),
            Backend::DirectMl => {
                let gpu = directml_config(cpu.clone(), opts.gpu_device, backend_error.clone())?;
                let joint = if opts.joint_on_cpu { cpu } else { gpu.clone() };
                (gpu, joint)
            }
            other => {
                return Err(SttError::Backend(format!(
                    "{other:?} is not supported by the Parakeet engine"
                )));
            }
        };

        let model =
            ParakeetTDT::from_pretrained_with_joint_config(model_dir, Some(encoder), Some(joint))
                .map_err(
                |e| match backend_error.lock().ok().and_then(|mut g| g.take()) {
                    Some(msg) => SttError::Backend(msg),
                    None => SttError::Load(e.to_string()),
                },
            )?;

        let id = model_dir
            .file_name()
            .and_then(|n| n.to_str())
            .filter(|n| n.starts_with(ENGINE_ID))
            .unwrap_or(ENGINE_ID)
            .to_string();
        tracing::info!(%id, backend = ?opts.backend, joint_on_cpu = opts.joint_on_cpu, threads, "parakeet loaded");

        Ok(Self {
            model,
            info: EngineInfo {
                id,
                // Providers register with `error_on_failure`, so the requested backend is
                // the one that loaded.
                backend: opts.backend,
                // ONNX Runtime does not report which adapter DirectML bound.
                device: match (opts.backend, opts.gpu_device) {
                    (Backend::DirectMl, Some(i)) => Some(format!("DXGI adapter {i}")),
                    _ => None,
                },
                caps: Caps {
                    prompt: false,
                    hotwords: false,
                    word_timestamps: true,
                    punctuation: true,
                    // parakeet-rs exposes no run options, so ORT's terminate flag is
                    // out of reach.
                    cancel: false,
                },
                languages: LANGUAGES.iter().map(|l| l.to_string()).collect(),
            },
        })
    }
}

#[cfg(windows)]
fn directml_config(
    base: ExecutionConfig,
    device: Option<i32>,
    backend_error: Arc<Mutex<Option<String>>>,
) -> Result<ExecutionConfig, SttError> {
    Ok(base.with_custom_configure(move |builder| {
        // ONNX Runtime documents both as required-off for DirectML.
        let builder = builder
            .with_memory_pattern(false)?
            .with_parallel_execution(false)?;
        let mut ep = ort::ep::DirectML::default();
        if let Some(id) = device {
            ep = ep.with_device_id(id);
        }
        builder
            .with_execution_providers([ep.build().error_on_failure()])
            .map_err(|e| {
                if let Ok(mut g) = backend_error.lock() {
                    *g = Some(format!(
                        "DirectML execution provider failed to register: {e}"
                    ));
                }
                e.into()
            })
    }))
}

#[cfg(not(windows))]
fn directml_config(
    _base: ExecutionConfig,
    _device: Option<i32>,
    _backend_error: Arc<Mutex<Option<String>>>,
) -> Result<ExecutionConfig, SttError> {
    Err(SttError::Backend("DirectML exists only on Windows".into()))
}

impl SttEngine for ParakeetEngine {
    fn info(&self) -> &EngineInfo {
        &self.info
    }

    fn warm_up(&mut self) -> Result<(), SttError> {
        let silence = vec![0.0f32; SAMPLE_RATE as usize];
        self.model
            .transcribe_samples(silence, SAMPLE_RATE, 1, Some(TimestampMode::Tokens))
            .map_err(|e| SttError::Inference(e.to_string()))?;
        Ok(())
    }

    fn transcribe(&mut self, pcm: &[f32], opts: &DecodeOptions) -> Result<Transcript, SttError> {
        if pcm.len() < MIN_SAMPLES {
            return Err(SttError::EmptyAudio);
        }
        opts.cancel.checkpoint()?;
        let started = Instant::now();
        // The crate's word and sentence modes drop a repeated word ("that that"); deciding
        // what a disfluency is belongs to the normalizer.
        let result = self
            .model
            .transcribe_samples(pcm.to_vec(), SAMPLE_RATE, 1, Some(TimestampMode::Tokens))
            .map_err(|e| SttError::Inference(e.to_string()))?;
        let inference_time = started.elapsed();
        if opts.cancel.is_cancelled() {
            return Err(SttError::Cancelled);
        }

        let segments = sentence_segments(&result.tokens);
        let text = segments
            .iter()
            .map(|s| s.text.as_str())
            .collect::<Vec<_>>()
            .join(" ");
        Ok(Transcript {
            utterance: opts.utterance,
            text,
            segments,
            language: opts.language.clone(),
            inference_time,
        })
    }
}

fn sentence_segments(tokens: &[TimedToken]) -> Vec<Segment> {
    let mut segments = Vec::new();
    let mut text = String::new();
    let mut start = 0.0f32;
    let mut end = 0.0f32;
    let mut close = |text: &mut String, start: f32, end: f32| {
        let normalised = text.split_whitespace().collect::<Vec<_>>().join(" ");
        if !normalised.is_empty() {
            segments.push(Segment {
                text: normalised,
                start: secs(start),
                end: secs(end),
            });
        }
        text.clear();
    };
    for t in tokens {
        if text.trim().is_empty() {
            start = t.start;
        }
        text.push_str(&t.text);
        end = t.end;
        if t.text.trim_end().ends_with(['.', '?', '!']) {
            close(&mut text, start, end);
        }
    }
    close(&mut text, start, end);
    segments
}

fn secs(s: f32) -> Duration {
    Duration::from_secs_f32(s.max(0.0))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tok(text: &str, start: f32, end: f32) -> TimedToken {
        TimedToken {
            text: text.into(),
            start,
            end,
        }
    }

    #[test]
    fn segments_split_on_sentence_punctuation_and_keep_repeats() {
        let tokens = [
            tok(" Send", 0.0, 0.2),
            tok(" it", 0.2, 0.3),
            tok(" it", 0.3, 0.4),
            tok(".", 0.4, 0.5),
            tok(" Next", 0.8, 1.0),
            tok(" one", 1.0, 1.2),
        ];
        let segs = sentence_segments(&tokens);
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0].text, "Send it it.");
        assert_eq!(segs[0].start, Duration::ZERO);
        assert_eq!(segs[0].end, secs(0.5));
        assert_eq!(segs[1].text, "Next one");
        assert_eq!(segs[1].start, secs(0.8));
        assert_eq!(segs[1].end, secs(1.2));
    }

    #[test]
    fn no_tokens_no_segments() {
        assert!(sentence_segments(&[]).is_empty());
        assert!(sentence_segments(&[tok("  ", 0.0, 0.1)]).is_empty());
    }

    #[test]
    fn unsupported_backend_is_an_error_not_a_fallback() {
        let err = ParakeetEngine::new(Path::new("does-not-exist"), Backend::Vulkan)
            .err()
            .unwrap();
        assert!(matches!(err, SttError::Backend(_)));
    }

    #[test]
    fn missing_model_is_a_load_error() {
        let err = ParakeetEngine::new(Path::new("does-not-exist"), Backend::Cpu)
            .err()
            .unwrap();
        assert!(matches!(err, SttError::Load(_)));
    }
}
