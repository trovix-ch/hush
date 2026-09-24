use std::path::{Path, PathBuf};
use std::sync::Mutex;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
use hush_audio::silero::{SileroConfig, SileroVad};
use hush_audio::vad::{EnergyVad, Vad};
use hush_core::config::{Config, EngineChoice, GpuPolicy, NormalizerChoice};
use hush_core::gpu::{self, GpuDevice, GpuRequest, GpuSelector};
use hush_core::normalize::Normalizer;
use hush_core::stt::{Backend, SttEngine, SttError};
use hush_normalize::openai_http::Dialect;
use hush_normalize::{HttpConfig, NormalizerChain, OpenAiHttpNormalizer};
use hush_stt::models::{self, ModelManifest};
use hush_stt::{RemoteEngine, RemoteOptions, remote};

/// The only engine family this binary contains.
const ENGINE_FAMILY: &str = "transcribe-cpp";

pub fn resolve_model(choice: &EngineChoice) -> Result<(&'static ModelManifest, PathBuf)> {
    let model = models::find(&choice.model)?;
    if model.engine != ENGINE_FAMILY {
        bail!(
            "model `{}` needs the `{}` engine, which this build does not contain; use a \
             {ENGINE_FAMILY} model such as `{}`",
            model.id,
            model.engine,
            models::DEFAULT_MODEL_ID
        );
    }
    let dir = models::default_model_dir(&model.id)?;
    Ok((model, dir))
}

pub fn vad_model() -> Result<(&'static ModelManifest, PathBuf)> {
    let model = models::find(models::VAD_MODEL_ID)?;
    let dir = models::default_model_dir(&model.id)?;
    Ok((model, dir))
}

/// Silero when its model is on disk, the energy detector otherwise; the label says which
/// one runs and why.
pub fn load_vad() -> (Box<dyn Vad>, String) {
    let silero = vad_model().and_then(|(model, dir)| {
        if !model.is_present(&dir) {
            bail!("{} is not downloaded yet", model.id);
        }
        let started = Instant::now();
        let vad = SileroVad::load(&model.load_path(&dir), SileroConfig::default())?;
        Ok((vad, model.id.as_str(), started.elapsed()))
    });
    match silero {
        Ok((vad, id, load)) => (
            Box::new(vad),
            format!("Silero ({id}, load {} ms)", load.as_millis()),
        ),
        Err(e) => (
            Box::new(EnergyVad::default()),
            format!("energy, because {e:#}"),
        ),
    }
}

#[derive(Debug, Clone)]
pub struct EngineSummary {
    pub model: String,
    pub backend: Backend,
    pub device: Option<String>,
    /// Why the GPU did not load when `prefer-gpu` fell back to the CPU.
    pub fallback: Option<String>,
    /// For the worker: from process start until the model reported loaded.
    pub load: Duration,
    pub warm_up: Duration,
    /// `None` when speech runs in this process.
    pub worker: Option<WorkerSummary>,
}

#[derive(Debug, Clone)]
pub struct WorkerSummary {
    pub pid: u32,
    pub path: PathBuf,
    /// Measured inside the worker: the model load alone.
    pub model_load: Duration,
}

impl EngineSummary {
    pub fn short(&self) -> String {
        let device = self.device.as_deref().unwrap_or("default device");
        match self.backend {
            Backend::Cpu => "CPU".to_string(),
            b => format!("{b:?} · {device}"),
        }
    }
}

/// One enumeration per process: speech and the language model load at the same time, and
/// resolving both against one snapshot, taken before either loads, is what puts them on
/// the same card under `auto`.
static GPU_SNAPSHOT: Mutex<Option<Vec<GpuDevice>>> = Mutex::new(None);

/// Starts the worker for a moment unless speech runs in this process.
pub fn gpu_devices(choice: &EngineChoice) -> Result<Vec<GpuDevice>> {
    let mut snapshot = GPU_SNAPSHOT
        .lock()
        .map_err(|_| anyhow!("GPU list lock poisoned"))?;
    if let Some(d) = snapshot.as_ref() {
        return Ok(d.clone());
    }
    let devices = enumerate_gpus(choice)?;
    *snapshot = Some(devices.clone());
    Ok(devices)
}

fn enumerate_gpus(choice: &EngineChoice) -> Result<Vec<GpuDevice>> {
    #[cfg(feature = "in-process-stt")]
    if choice.in_process {
        return Ok(hush_stt_worker::transcribe_cpp::vulkan_devices());
    }
    Ok(remote::list_devices(&worker_path(choice)?)?)
}

/// Where a stage will try to run: PCI ids in order, and why.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GpuPlan {
    /// Empty when `auto` chose a device without a PCI id; each engine then applies the
    /// same policy to its own list.
    pub candidates: Vec<String>,
    pub why: String,
}

fn plan(key: &str, req: &GpuRequest, devices: &[GpuDevice]) -> Result<GpuPlan, String> {
    let r = gpu::resolve(req, devices)?;
    let candidates: Vec<String> = r
        .order
        .iter()
        .filter_map(|&i| devices[i].pci.clone())
        .collect();
    if candidates.is_empty() {
        let unnamed = devices[r.order[0]].label();
        // Some integrated GPUs report no PCI id (an AMD Radeon iGPU, seen 2026-09-24).
        if *req == GpuRequest::Selector(GpuSelector::Auto) {
            return Ok(GpuPlan {
                candidates,
                why: format!("{}: {unnamed}, which has no PCI bus id", r.why),
            });
        }
        return Err(format!(
            "{unnamed} reports no PCI bus id, so it cannot be pinned; use \"auto\""
        ));
    }
    if let GpuRequest::LegacyIndex(i) = req {
        tracing::warn!(
            "{key}.gpu_device = {i} is deprecated: it names {} in this session, and Vulkan \
             numbers devices differently in console and remote sessions; write {key}.device \
             = \"{}\" or \"auto\"",
            candidates[0],
            candidates[0]
        );
    }
    Ok(GpuPlan {
        candidates,
        why: r.why,
    })
}

/// `Err` says why no GPU can be tried.
pub fn speech_gpu(choice: &EngineChoice) -> Result<GpuPlan, String> {
    let devices = gpu_devices(choice).map_err(|e| format!("cannot list GPUs: {e:#}"))?;
    plan("engine", &choice.gpu_request(), &devices)
}

/// The speech engine's plan unless `normalizer.device` names another card.
pub fn llm_gpu(config: &Config) -> Result<GpuPlan, String> {
    match config.normalizer.gpu_request() {
        None => speech_gpu(&config.engine).map(|p| GpuPlan {
            why: format!("follows speech ({})", p.why),
            ..p
        }),
        Some(req) => {
            let devices =
                gpu_devices(&config.engine).map_err(|e| format!("cannot list GPUs: {e:#}"))?;
            plan("normalizer", &req, &devices)
        }
    }
}

/// The GPU policy applied to the speech engine; the string says why `prefer-gpu` fell
/// back to the CPU.
fn with_policy<E>(
    choice: &EngineChoice,
    build: impl Fn(Backend, Option<String>) -> Result<E, SttError>,
) -> Result<(E, Option<String>)> {
    if choice.gpu == GpuPolicy::CpuOnly {
        return Ok((build(Backend::Cpu, None)?, None));
    }
    let mut failures = Vec::new();
    match speech_gpu(choice) {
        Ok(plan) => {
            let tries: Vec<Option<String>> = if plan.candidates.is_empty() {
                vec![None]
            } else {
                plan.candidates.into_iter().map(Some).collect()
            };
            for pci in tries {
                let gpu = pci.clone().unwrap_or_else(|| "auto".into());
                match build(Backend::Vulkan, pci) {
                    Ok(e) => {
                        tracing::info!(%gpu, why = %plan.why, "speech GPU");
                        return Ok((e, None));
                    }
                    Err(e) => {
                        tracing::warn!(%gpu, error = %e, "speech did not load on this GPU");
                        failures.push(format!("{gpu}: {e}"));
                        if choice.gpu == GpuPolicy::RequireGpu {
                            break;
                        }
                    }
                }
            }
        }
        Err(e) => failures.push(e),
    }
    let gpu_err = failures.join("; ");
    if choice.gpu == GpuPolicy::RequireGpu {
        bail!("engine.gpu = \"require-gpu\" and the GPU backend did not load: {gpu_err}");
    }
    tracing::warn!(error = %gpu_err, "GPU backend did not load; falling back to the CPU");
    let e = build(Backend::Cpu, None)
        .with_context(|| format!("GPU failed ({gpu_err}) and so did the CPU"))?;
    Ok((e, Some(gpu_err)))
}

/// `engine.worker_path`, else the worker next to this executable.
pub fn worker_path(choice: &EngineChoice) -> Result<PathBuf> {
    let path = match &choice.worker_path {
        Some(p) => p.clone(),
        None => remote::default_worker_path().context("locating the speech worker")?,
    };
    if !path.is_file() {
        bail!(
            "the speech worker {} is missing; it ships next to hush.exe (or set \
             engine.worker_path)",
            path.display()
        );
    }
    Ok(path)
}

/// Blocks for seconds.
pub fn load_engine(
    choice: &EngineChoice,
    model_path: &Path,
) -> Result<(Box<dyn SttEngine>, EngineSummary)> {
    if choice.in_process {
        return load_in_process(choice, model_path);
    }
    let worker = worker_path(choice)?;
    let started = Instant::now();
    let (engine, fallback) = with_policy(choice, |backend, gpu| {
        RemoteEngine::new(RemoteOptions {
            gpu,
            ..RemoteOptions::new(worker.clone(), model_path.to_path_buf(), backend)
        })
    })?;
    let info = engine.info();
    let summary = EngineSummary {
        model: info.id.clone(),
        backend: info.backend,
        device: info.device.clone(),
        fallback,
        load: started.elapsed().saturating_sub(engine.warm_up_time()),
        warm_up: engine.warm_up_time(),
        worker: Some(WorkerSummary {
            pid: engine.worker_pid().unwrap_or_default(),
            path: worker,
            model_load: engine.load_time(),
        }),
    };
    Ok((Box::new(engine), summary))
}

#[cfg(feature = "in-process-stt")]
fn load_in_process(
    choice: &EngineChoice,
    model_path: &Path,
) -> Result<(Box<dyn SttEngine>, EngineSummary)> {
    let started = Instant::now();
    let (mut engine, fallback) = with_policy(choice, |backend, gpu| {
        hush_stt_worker::TranscribeCppEngine::new(model_path, backend, gpu)
    })?;
    let load = started.elapsed();
    let started = Instant::now();
    engine.warm_up().context("engine warm-up")?;
    let warm_up = started.elapsed();
    let info = engine.info();
    let summary = EngineSummary {
        model: info.id.clone(),
        backend: info.backend,
        device: info.device.clone(),
        fallback,
        load,
        warm_up,
        worker: None,
    };
    Ok((Box::new(engine), summary))
}

#[cfg(not(feature = "in-process-stt"))]
fn load_in_process(_: &EngineChoice, _: &Path) -> Result<(Box<dyn SttEngine>, EngineSummary)> {
    bail!(
        "engine.in_process = true, but this build has no in-process engine (built without \
         the `in-process-stt` feature)"
    )
}

pub fn llm_model(id: &str) -> Result<(&'static ModelManifest, PathBuf)> {
    let model = models::find(id)?;
    if model.engine != LLM_FAMILY {
        bail!(
            "model `{}` is a {} model; normalizer.model needs a {LLM_FAMILY} model such as `{}`",
            model.id,
            model.engine,
            models::DEFAULT_LLM_ID
        );
    }
    let dir = models::default_model_dir(&model.id)?;
    Ok((model, dir))
}

const LLM_FAMILY: &str = "llama-cpp";

/// What the LLM stage turned out to be once loaded.
#[derive(Debug, Clone)]
pub struct NormalizerReady {
    pub label: String,
    /// Why a `prefer-gpu` language model runs on the CPU.
    pub cpu_fallback: Option<String>,
}

/// Blocks for seconds, or minutes while a model downloads; `status` hears what it is
/// waiting for. `Ok(None)` means the config asks for rules only; `Err` explains why the
/// app stays rules-only.
pub fn build_normalizer(
    config: &Config,
    status: &mut dyn FnMut(&str),
) -> Result<Option<(Box<dyn Normalizer>, NormalizerReady)>> {
    match &config.normalizer {
        NormalizerChoice::Rules => Ok(None),
        NormalizerChoice::Http { .. } => {
            let Some(http) = http_config(&config.normalizer) else {
                return Ok(None);
            };
            let label = format!("{} via {}", http.model, http.base_url);
            status("Connecting to the language model…");
            let (n, warm) = build_http_normalizer(http)?;
            Ok(Some((
                n,
                NormalizerReady {
                    label: format!("{label} (warm-up {} ms)", warm.as_millis()),
                    cpu_fallback: None,
                },
            )))
        }
        NormalizerChoice::LlamaCpp {
            model, timeout_ms, ..
        } => build_llama(config, model, Duration::from_millis(*timeout_ms), status).map(Some),
    }
}

#[cfg(feature = "llama-cpp")]
fn build_llama(
    config: &Config,
    model_id: &str,
    timeout: Duration,
    status: &mut dyn FnMut(&str),
) -> Result<(Box<dyn Normalizer>, NormalizerReady)> {
    use hush_normalize::llama_cpp::{DeviceChoice, LlamaCppConfig, LlamaCppNormalizer};

    let (model, dir) = llm_model(model_id)?;
    if !model.is_present(&dir) {
        status("Downloading language model…");
        models::ensure_downloaded(model, &dir)?;
    }
    status("Loading language model…");
    let device = if config.engine.gpu == GpuPolicy::CpuOnly {
        DeviceChoice::default()
    } else {
        match llm_gpu(config) {
            Ok(p) if p.candidates.is_empty() => DeviceChoice::default(),
            Ok(p) => DeviceChoice::Pci(p.candidates),
            Err(e) => {
                let selector = match &config.normalizer {
                    NormalizerChoice::LlamaCpp { device, .. } if *device != GpuSelector::Auto => {
                        device.clone()
                    }
                    _ => config.engine.device.clone(),
                };
                tracing::warn!(error = %e, %selector, "no GPU plan from the speech engine's list; llama.cpp resolves on its own");
                DeviceChoice::Select(selector)
            }
        }
    };
    let started = Instant::now();
    let mut n = LlamaCppNormalizer::load(LlamaCppConfig {
        model_path: model.load_path(&dir),
        model_id: model.id.clone(),
        device,
        gpu: config.engine.gpu,
        timeout,
    })
    .context("loading the language model")?;
    let load = started.elapsed();
    let started = Instant::now();
    n.warm().context("language model warm-up")?;
    let b = n.backend().clone();
    let label = format!(
        "{} on {:?} ({}), {}/{} layers offloaded, load {} ms, warm-up {} ms",
        model.id,
        b.backend,
        b.device.as_deref().unwrap_or("CPU"),
        b.layers_offloaded,
        b.layers_total,
        load.as_millis(),
        started.elapsed().as_millis()
    );
    Ok((
        Box::new(NormalizerChain::new(vec![Box::new(n)])),
        NormalizerReady {
            label,
            cpu_fallback: b.fallback,
        },
    ))
}

#[cfg(not(feature = "llama-cpp"))]
fn build_llama(
    _: &Config,
    _: &str,
    _: Duration,
    _: &mut dyn FnMut(&str),
) -> Result<(Box<dyn Normalizer>, NormalizerReady)> {
    bail!("this build has no embedded language model (built without the `llama-cpp` feature)")
}

pub fn http_config(choice: &NormalizerChoice) -> Option<HttpConfig> {
    match choice {
        NormalizerChoice::Rules | NormalizerChoice::LlamaCpp { .. } => None,
        NormalizerChoice::Http {
            base_url,
            model,
            timeout_ms,
        } => {
            let mut cfg = HttpConfig::new(base_url.clone(), model.clone());
            cfg.timeout = Duration::from_millis(*timeout_ms);
            Some(cfg)
        }
    }
}

#[derive(Debug, Clone)]
pub struct Probe {
    pub elapsed: Duration,
    pub models: Vec<String>,
    pub has_model: bool,
}

const PROBE_TIMEOUT: Duration = Duration::from_secs(2);

/// Loads nothing, so it answers even when the model is cold; `Err` means the server is
/// not there.
pub fn probe_http(cfg: &HttpConfig) -> Result<Probe> {
    let (url, key) = match cfg.dialect {
        Dialect::Ollama => {
            let root = cfg.base_url.strip_suffix("/v1").unwrap_or(&cfg.base_url);
            (format!("{root}/api/tags"), ("models", "name"))
        }
        Dialect::OpenAi => (format!("{}/models", cfg.base_url), ("data", "id")),
    };
    let agent: ureq::Agent = ureq::Agent::config_builder()
        .timeout_global(Some(PROBE_TIMEOUT))
        .http_status_as_error(false)
        .build()
        .into();
    let started = Instant::now();
    let mut resp = agent
        .get(&url)
        .call()
        .with_context(|| format!("GET {url}"))?;
    let status = resp.status().as_u16();
    if status != 200 {
        bail!("GET {url}: HTTP {status}");
    }
    let body = resp
        .body_mut()
        .read_to_string()
        .with_context(|| format!("reading {url}"))?;
    let elapsed = started.elapsed();
    let json: serde_json::Value =
        serde_json::from_str(&body).with_context(|| format!("{url} did not return JSON"))?;
    let models: Vec<String> = json[key.0]
        .as_array()
        .map(|a| {
            a.iter()
                .filter_map(|m| m[key.1].as_str().map(str::to_string))
                .collect()
        })
        .unwrap_or_default();
    let has_model = models.iter().any(|m| m == &cfg.model);
    Ok(Probe {
        elapsed,
        models,
        has_model,
    })
}

/// `Err` explains why the app stays rules-only.
pub fn build_http_normalizer(cfg: HttpConfig) -> Result<(Box<dyn Normalizer>, Duration)> {
    let probe = probe_http(&cfg).context("normalizer server did not answer")?;
    if !probe.has_model {
        bail!(
            "server at {} does not serve `{}` (it has: {})",
            cfg.base_url,
            cfg.model,
            probe.models.join(", ")
        );
    }
    let started = Instant::now();
    let mut chain = NormalizerChain::new(vec![Box::new(OpenAiHttpNormalizer::new(cfg))]);
    chain.warm().context("normalizer warm-up")?;
    Ok((Box::new(chain), started.elapsed()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use hush_core::config::Config;

    #[test]
    fn default_config_names_a_model_this_build_can_load() {
        let (model, dir) = resolve_model(&Config::default().engine).unwrap();
        assert_eq!(model.id, models::DEFAULT_MODEL_ID);
        assert!(dir.ends_with(model.id.as_str()));
    }

    #[test]
    fn vad_manifest_entry_matches_the_pinned_model() {
        use hush_audio::silero;
        let (model, _) = vad_model().unwrap();
        let [file] = model.files.as_slice() else {
            panic!("{:?}", model.files);
        };
        assert_eq!(file.name, silero::MODEL_FILE);
        assert_eq!(file.url, silero::MODEL_URL);
        assert_eq!(file.size, silero::MODEL_SIZE);
        assert_eq!(file.sha256.as_deref(), Some(silero::MODEL_SHA256));
        assert_eq!(model.license, silero::MODEL_LICENSE);
        assert_eq!(model.attribution, silero::MODEL_ATTRIBUTION);
        assert!(
            resolve_model(&EngineChoice {
                model: model.id.clone(),
                ..Default::default()
            })
            .is_err()
        );
    }

    #[test]
    fn onnx_models_are_refused_in_a_default_build() {
        let choice = EngineChoice {
            model: "parakeet-tdt-0.6b-v3".into(),
            ..Default::default()
        };
        assert!(resolve_model(&choice).is_err());
    }

    #[test]
    fn default_normalizer_names_a_language_model() {
        let NormalizerChoice::LlamaCpp { model, .. } = Config::default().normalizer else {
            panic!("the default normalizer is not the embedded one");
        };
        let (m, dir) = llm_model(&model).unwrap();
        assert_eq!(m.id, models::DEFAULT_LLM_ID);
        assert!(dir.ends_with(m.id.as_str()));
        assert!(llm_model(models::DEFAULT_MODEL_ID).is_err());
        assert!(
            resolve_model(&EngineChoice {
                model: model.clone(),
                ..Default::default()
            })
            .is_err()
        );
    }

    #[test]
    fn rules_config_builds_no_normalizer() {
        let config = Config {
            normalizer: NormalizerChoice::Rules,
            ..Config::default()
        };
        let mut heard = Vec::new();
        let built = build_normalizer(&config, &mut |s| heard.push(s.to_string())).unwrap();
        assert!(built.is_none());
        assert!(heard.is_empty());
    }

    #[test]
    fn a_missing_worker_is_named_rather_than_retried_on_the_cpu() {
        let choice = EngineChoice {
            worker_path: Some("no-such-dir/hush-stt-worker.exe".into()),
            ..Default::default()
        };
        let err = load_engine(&choice, Path::new("x.gguf")).err().unwrap();
        let msg = format!("{err:#}");
        assert!(msg.contains("no-such-dir"), "{msg}");
        assert!(!msg.contains("CPU"), "{msg}");
    }

    #[cfg(not(feature = "in-process-stt"))]
    #[test]
    fn in_process_needs_the_feature() {
        let choice = EngineChoice {
            in_process: true,
            ..Default::default()
        };
        let err = load_engine(&choice, Path::new("x.gguf")).err().unwrap();
        assert!(format!("{err:#}").contains("in-process-stt"), "{err:#}");
    }

    #[test]
    fn the_language_model_follows_speech_unless_pinned_elsewhere() {
        use hush_core::gpu::{GpuKind, GpuSelector};

        let card = |pci: &str, kind, free_gib: u64| GpuDevice {
            description: "card".into(),
            pci: Some(pci.into()),
            kind,
            memory_total: 16 << 30,
            memory_free: free_gib << 30,
        };
        *GPU_SNAPSHOT.lock().unwrap() = Some(vec![
            card("0000:0e:00.0", GpuKind::Integrated, 40),
            card("0000:01:00.0", GpuKind::Discrete, 9),
            card("0000:05:00.0", GpuKind::Discrete, 15),
        ]);
        let mut config = Config::default();
        let speech = speech_gpu(&config.engine).unwrap();
        assert_eq!(speech.candidates, ["0000:05:00.0", "0000:01:00.0"]);
        assert_eq!(llm_gpu(&config).unwrap().candidates, speech.candidates);

        config.engine.device = "01:00".parse().unwrap();
        assert_eq!(llm_gpu(&config).unwrap().candidates, ["0000:01:00.0"]);

        config.normalizer = NormalizerChoice::LlamaCpp {
            model: models::DEFAULT_LLM_ID.into(),
            device: "05:00".parse().unwrap(),
            gpu_device: None,
            timeout_ms: 5000,
        };
        assert_eq!(llm_gpu(&config).unwrap().candidates, ["0000:05:00.0"]);

        config.engine.device = GpuSelector::Auto;
        config.engine.gpu_device = Some(0);
        assert_eq!(
            speech_gpu(&config.engine).unwrap().candidates,
            ["0000:0e:00.0"]
        );

        let mut igpu = card("", GpuKind::Integrated, 8);
        igpu.pci = None;
        *GPU_SNAPSHOT.lock().unwrap() = Some(vec![igpu]);
        let auto = speech_gpu(&EngineChoice::default()).unwrap();
        assert!(auto.candidates.is_empty(), "{auto:?}");
        let pinned = EngineChoice {
            device: "card".parse().unwrap(),
            ..EngineChoice::default()
        };
        assert!(speech_gpu(&pinned).is_err());
        *GPU_SNAPSHOT.lock().unwrap() = None;
    }

    #[test]
    fn unreachable_server_fails_the_probe() {
        // Nothing listens on the discard port on a test machine.
        let cfg = HttpConfig::new("http://127.0.0.1:9/v1", "m");
        assert!(probe_http(&cfg).is_err());
    }
}
