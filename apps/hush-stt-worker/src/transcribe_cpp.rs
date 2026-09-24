use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use hush_core::gpu::{self, GpuDevice, GpuKind, GpuRequest, GpuSelector, PciAddress};
use hush_core::stt::{
    Backend, Caps, DecodeOptions, EngineInfo, SAMPLE_RATE, Segment, SttEngine, SttError, Transcript,
};
use transcribe_cpp as tc;

/// Below one 10 ms feature hop there is nothing to recognise.
const MIN_SAMPLES: usize = 160;

/// The native abort callback reads only the library's own flag, so a watcher copies ours
/// across; this bounds how late a cancel lands.
const CANCEL_POLL: Duration = Duration::from_millis(5);

#[derive(Debug, Clone)]
pub struct TranscribeCppOptions {
    pub backend: Backend,
    /// PCI bus id. `None` on Vulkan applies the auto policy, never ggml's own default,
    /// which may be an integrated GPU or the card the LLM is using.
    pub gpu: Option<String>,
    /// `None` picks half the logical cores: ggml's spin-waiting threads lose throughput
    /// when they share a physical core.
    pub threads: Option<usize>,
}

impl TranscribeCppOptions {
    pub fn new(backend: Backend) -> Self {
        Self {
            backend,
            gpu: None,
            threads: None,
        }
    }
}

/// The one enumeration of GPUs that the worker, the app and `bench-stt` all use.
pub fn vulkan_devices() -> Vec<GpuDevice> {
    tc_vulkan_devices().iter().map(summary).collect()
}

fn tc_vulkan_devices() -> Vec<tc::Device> {
    tc::devices()
        .into_iter()
        .filter(|d| d.kind.eq_ignore_ascii_case("vulkan"))
        .collect()
}

fn summary(d: &tc::Device) -> GpuDevice {
    GpuDevice {
        description: d.description.trim().to_string(),
        pci: d.device_id.clone(),
        kind: match d.device_type {
            tc::DeviceType::Gpu => GpuKind::Discrete,
            tc::DeviceType::Igpu => GpuKind::Integrated,
            _ => GpuKind::Other,
        },
        memory_total: d.memory_total,
        memory_free: d.memory_free,
    }
}

/// The device a PCI id names, or the auto policy's first choice.
fn pick_device(gpu: Option<&str>) -> Result<tc::Device, SttError> {
    let mut devices = tc_vulkan_devices();
    let listed: Vec<GpuDevice> = devices.iter().map(summary).collect();
    let request = match gpu {
        Some(pci) => GpuRequest::Selector(GpuSelector::Pci(
            PciAddress::parse(pci)
                .ok_or_else(|| SttError::Backend(format!("`{pci}` is not a PCI bus id")))?,
        )),
        None => GpuRequest::Selector(GpuSelector::Auto),
    };
    let chosen = gpu::resolve(&request, &listed).map_err(SttError::Backend)?;
    let first = chosen.order[0];
    tracing::debug!(device = %listed[first].label(), why = %chosen.why, "Vulkan device");
    Ok(devices.swap_remove(first))
}

pub fn library_version() -> String {
    format!("{} ({})", tc::version(), tc::version_commit())
}

pub struct TranscribeCppEngine {
    session: tc::Session,
    info: EngineInfo,
    timestamps: tc::TimestampKind,
}

fn load_err(e: tc::Error) -> SttError {
    match e {
        tc::Error::Backend(m) => SttError::Backend(m),
        other => SttError::Load(other.to_string()),
    }
}

impl TranscribeCppEngine {
    /// Fails rather than falling back to the CPU when Vulkan was requested and no Vulkan
    /// device took the weights.
    pub fn new(model_path: &Path, backend: Backend, gpu: Option<String>) -> Result<Self, SttError> {
        Self::with_options(
            model_path,
            TranscribeCppOptions {
                gpu,
                ..TranscribeCppOptions::new(backend)
            },
        )
    }

    pub fn with_options(model_path: &Path, opts: TranscribeCppOptions) -> Result<Self, SttError> {
        // Without this the native library prints every decode's statistics to stderr.
        static ROUTE_LOGS: std::sync::Once = std::sync::Once::new();
        ROUTE_LOGS.call_once(tc::init_logging);

        let tc_backend = match opts.backend {
            Backend::Vulkan => tc::Backend::Vulkan,
            Backend::Cpu => tc::Backend::Cpu,
            other => {
                return Err(SttError::Backend(format!(
                    "{other:?} is not supported by the transcribe.cpp engine"
                )));
            }
        };
        if opts.backend == Backend::Cpu && opts.gpu.is_some() {
            return Err(SttError::Backend(
                "a GPU device was given for the CPU backend".into(),
            ));
        }
        if opts.backend == Backend::Vulkan && !tc::backend_available(tc::Backend::Vulkan) {
            return Err(SttError::Backend("no Vulkan device is available".into()));
        }
        let device = match opts.backend {
            Backend::Vulkan => Some(pick_device(opts.gpu.as_deref())?),
            _ => None,
        };

        let model = tc::Model::load_with(
            model_path,
            &tc::ModelOptions {
                backend: tc_backend,
                device,
            },
        )
        .map_err(load_err)?;

        // A ggml device name such as `Vulkan1` or `CPU`, not the backend enum.
        let loaded = model.backend();
        let lower = loaded.to_ascii_lowercase();
        let loaded_backend = if lower.starts_with("vulkan") {
            Backend::Vulkan
        } else if lower.starts_with("cpu") {
            Backend::Cpu
        } else {
            return Err(SttError::Backend(format!(
                "model bound to unexpected backend `{loaded}`"
            )));
        };
        if loaded_backend != opts.backend {
            return Err(SttError::Backend(format!(
                "requested {:?}, but the model loaded on `{loaded}`",
                opts.backend
            )));
        }
        // The bus id goes in because two identical cards have identical descriptions.
        let device_name = model.device().ok().map(|d| match d.device_id {
            Some(id) => format!("{} [{id}]", d.description.trim()),
            None => d.description.trim().to_string(),
        });

        let threads = opts.threads.unwrap_or_else(|| {
            std::thread::available_parallelism()
                .map(|n| (n.get() / 2).max(1))
                .unwrap_or(4)
        });
        let session = model
            .session_with(&tc::SessionOptions {
                n_threads: i32::try_from(threads).unwrap_or(i32::MAX),
                ..Default::default()
            })
            .map_err(load_err)?;

        let caps = model.capabilities();
        // Word times are what segment stitching measures pauses with.
        let timestamps = match caps.max_timestamp_kind {
            tc::TimestampKind::None => tc::TimestampKind::None,
            tc::TimestampKind::Word | tc::TimestampKind::Token => tc::TimestampKind::Word,
            _ => tc::TimestampKind::Segment,
        };
        let id = model_id(&model.arch(), &model.variant());
        tracing::info!(
            %id,
            backend = %loaded,
            device = device_name.as_deref().unwrap_or("?"),
            threads,
            "transcribe.cpp model loaded"
        );

        Ok(Self {
            session,
            timestamps,
            info: EngineInfo {
                id,
                backend: loaded_backend,
                device: device_name,
                caps: Caps {
                    prompt: model.supports(tc::Feature::InitialPrompt),
                    hotwords: false,
                    word_timestamps: matches!(
                        caps.max_timestamp_kind,
                        tc::TimestampKind::Word | tc::TimestampKind::Token
                    ),
                    // Both supported families (Parakeet TDT v3, Whisper) emit punctuation
                    // and casing; the library exposes no probe for it.
                    punctuation: true,
                    cancel: model.supports(tc::Feature::Cancellation),
                },
                languages: caps.languages,
            },
        })
    }

    fn run(&mut self, pcm: &[f32], opts: &DecodeOptions) -> Result<tc::Transcript, SttError> {
        opts.cancel.checkpoint()?;
        let family = match (&opts.prompt, self.info.caps.prompt) {
            (Some(p), true) => Some(tc::RunExtension::Whisper(tc::WhisperRunOptions {
                initial_prompt: Some(p.clone()),
                ..Default::default()
            })),
            _ => None,
        };
        let run = tc::RunOptions {
            timestamps: self.timestamps,
            language: opts.language.clone(),
            family,
            ..Default::default()
        };
        // A fresh flag per run: the library's flag stays set once raised, and a reset
        // racing a late watcher store could abort the next utterance.
        let abort = tc::CancelToken::new();
        self.session.set_cancel_token(&abort);
        let done = AtomicBool::new(false);
        let session = &mut self.session;
        let result = std::thread::scope(|s| {
            let watcher = s.spawn(|| {
                while !done.load(Ordering::Acquire) {
                    if opts.cancel.checkpoint().is_err() {
                        abort.cancel();
                        return;
                    }
                    std::thread::park_timeout(CANCEL_POLL);
                }
            });
            let r = session.run(pcm, &run);
            done.store(true, Ordering::Release);
            watcher.thread().unpark();
            r
        });
        let tr = result.map_err(|e| match e {
            tc::Error::Aborted { .. } => opts
                .cancel
                .checkpoint()
                .err()
                .map_or_else(|| SttError::Inference(e.to_string()), SttError::from),
            other => SttError::Inference(other.to_string()),
        })?;
        // The encoder cannot be interrupted, so a cancel during it lands only when the run
        // ends. A late deadline still returns the finished work; a cancel never does.
        if opts.cancel.is_cancelled() {
            return Err(SttError::Cancelled);
        }
        Ok(tr)
    }
}

impl SttEngine for TranscribeCppEngine {
    fn info(&self) -> &EngineInfo {
        &self.info
    }

    /// ggml builds a GPU pipeline per graph shape and the driver compiles shaders on first
    /// use (seconds on a fresh cache), so one short clip does not warm the long path.
    fn warm_up(&mut self) -> Result<(), SttError> {
        let opts = DecodeOptions::default();
        for secs in [30, 1] {
            let silence = vec![0.0f32; SAMPLE_RATE as usize * secs];
            self.run(&silence, &opts)?;
        }
        Ok(())
    }

    /// One second because warm-up already built that graph shape; a new length would
    /// compile shaders in the middle of the utterance.
    fn nudge(&mut self) -> Result<(), SttError> {
        if self.info.backend == Backend::Cpu {
            return Ok(());
        }
        let silence = vec![0.0f32; SAMPLE_RATE as usize];
        self.run(&silence, &DecodeOptions::default())?;
        Ok(())
    }

    fn transcribe(&mut self, pcm: &[f32], opts: &DecodeOptions) -> Result<Transcript, SttError> {
        if pcm.len() < MIN_SAMPLES {
            return Err(SttError::EmptyAudio);
        }
        let started = Instant::now();
        let tr = self.run(pcm, opts)?;
        let inference_time = started.elapsed();
        let audio_len = Duration::from_secs_f64(pcm.len() as f64 / f64::from(SAMPLE_RATE));
        let segments = to_segments(&tr, audio_len);
        let words = to_words(&tr);
        let text = normalise_ws(&tr.text);
        Ok(Transcript {
            utterance: opts.utterance,
            text,
            segments,
            words,
            language: tr.language.clone().or_else(|| opts.language.clone()),
            inference_time,
        })
    }
}

/// Some GGUFs put the architecture into the variant (`whisper-large-v3-turbo`), others
/// do not (`tdt-0.6b-v3`).
fn model_id(arch: &str, variant: &str) -> String {
    if variant.is_empty() {
        arch.to_string()
    } else if variant.starts_with(arch) {
        variant.to_string()
    } else {
        format!("{arch}-{variant}")
    }
}

fn normalise_ws(s: &str) -> String {
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

fn ms(v: i64) -> Duration {
    Duration::from_millis(u64::try_from(v).unwrap_or(0))
}

fn to_segments(tr: &tc::Transcript, audio_len: Duration) -> Vec<Segment> {
    let segs: Vec<Segment> = tr
        .segments
        .iter()
        .filter_map(|s| {
            let text = normalise_ws(&s.text);
            (!text.is_empty()).then(|| Segment {
                text,
                start: ms(s.t0_ms),
                end: ms(s.t1_ms),
            })
        })
        .collect();
    if !segs.is_empty() {
        return segs;
    }
    let text = normalise_ws(&tr.text);
    if text.is_empty() {
        return Vec::new();
    }
    vec![Segment {
        text,
        start: Duration::ZERO,
        end: audio_len,
    }]
}

fn to_words(tr: &tc::Transcript) -> Vec<Segment> {
    tr.words
        .iter()
        .filter_map(|w| {
            let text = normalise_ws(&w.text);
            (!text.is_empty()).then(|| Segment {
                text,
                start: ms(w.t0_ms),
                end: ms(w.t1_ms),
            })
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn transcript(text: &str, segs: &[(&str, i64, i64)]) -> tc::Transcript {
        tc::Transcript {
            text: text.into(),
            segments: segs
                .iter()
                .map(|(t, a, b)| tc::Segment {
                    t0_ms: *a,
                    t1_ms: *b,
                    text: (*t).into(),
                    ..Default::default()
                })
                .collect(),
            ..Default::default()
        }
    }

    #[test]
    fn engine_segments_are_used_when_present() {
        let tr = transcript(
            " Hello there. Bye. ",
            &[(" Hello  there.", 0, 900), (" Bye.", 1000, 1400)],
        );
        let segs = to_segments(&tr, Duration::from_secs(2));
        assert_eq!(segs.len(), 2);
        assert_eq!(segs[0].text, "Hello there.");
        assert_eq!(segs[1].start, Duration::from_millis(1000));
        assert_eq!(segs[1].end, Duration::from_millis(1400));
    }

    #[test]
    fn word_times_are_carried_over() {
        let tr = tc::Transcript {
            words: vec![
                tc::Word {
                    t0_ms: 80,
                    t1_ms: 400,
                    text: " Ask".into(),
                    ..Default::default()
                },
                tc::Word {
                    t0_ms: 400,
                    t1_ms: 400,
                    text: " ".into(),
                    ..Default::default()
                },
                tc::Word {
                    t0_ms: 480,
                    t1_ms: 720,
                    text: "not.".into(),
                    ..Default::default()
                },
            ],
            ..Default::default()
        };
        let words = to_words(&tr);
        assert_eq!(words.len(), 2);
        assert_eq!(words[0].text, "Ask");
        assert_eq!(words[1].start, Duration::from_millis(480));
        assert_eq!(words[1].end, Duration::from_millis(720));
    }

    #[test]
    fn falls_back_to_one_segment_spanning_the_audio() {
        let tr = transcript(" Hello  there ", &[]);
        let segs = to_segments(&tr, Duration::from_secs(2));
        assert_eq!(
            segs,
            vec![Segment {
                text: "Hello there".into(),
                start: Duration::ZERO,
                end: Duration::from_secs(2)
            }]
        );
        assert!(to_segments(&transcript("  ", &[]), Duration::from_secs(1)).is_empty());
    }

    #[test]
    fn model_ids_do_not_repeat_the_architecture() {
        assert_eq!(model_id("parakeet", "tdt-0.6b-v3"), "parakeet-tdt-0.6b-v3");
        assert_eq!(
            model_id("whisper", "whisper-large-v3-turbo"),
            "whisper-large-v3-turbo"
        );
        assert_eq!(model_id("moonshine", ""), "moonshine");
    }

    #[test]
    fn unsupported_backend_is_an_error_not_a_fallback() {
        let err = TranscribeCppEngine::new(Path::new("x.gguf"), Backend::DirectMl, None)
            .err()
            .unwrap();
        assert!(matches!(err, SttError::Backend(_)));
        let err = TranscribeCppEngine::new(Path::new("x.gguf"), Backend::Cpu, Some("05:00".into()))
            .err()
            .unwrap();
        assert!(matches!(err, SttError::Backend(_)));
    }

    #[test]
    fn live_cancellation() {
        let Some(path) = std::env::var_os("HUSH_TEST_TC_MODEL") else {
            return;
        };
        let mut engine = TranscribeCppEngine::new(Path::new(&path), Backend::Cpu, None).unwrap();
        let audio = vec![0.0f32; SAMPLE_RATE as usize * 30];

        let expired = DecodeOptions {
            cancel: hush_core::CancelToken::new().with_deadline(Instant::now()),
            ..Default::default()
        };
        assert!(matches!(
            engine.transcribe(&audio, &expired),
            Err(SttError::Deadline)
        ));

        let started = Instant::now();
        let full = engine.transcribe(&audio, &DecodeOptions::default());
        let full_time = started.elapsed();
        assert!(full.is_ok(), "{full:?}");

        let opts = DecodeOptions::default();
        let token = opts.cancel.clone();
        let canceller = std::thread::spawn(move || {
            std::thread::sleep(Duration::from_millis(20));
            token.cancel();
        });
        let started = Instant::now();
        let r = engine.transcribe(&audio, &opts);
        let cancelled_time = started.elapsed();
        canceller.join().unwrap();
        eprintln!(
            "cancel supported: {}; full run {full_time:?}, cancelled run {cancelled_time:?}",
            engine.info().caps.cancel
        );
        assert!(matches!(r, Err(SttError::Cancelled)), "{r:?}");
    }

    #[test]
    fn missing_model_is_a_load_error() {
        let err = TranscribeCppEngine::new(Path::new("does-not-exist.gguf"), Backend::Cpu, None)
            .err()
            .unwrap();
        assert!(matches!(err, SttError::Load(_)));
    }
}
