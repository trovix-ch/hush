use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use hush_audio::silero::{SileroConfig, SileroVad};
use hush_audio::vad::{EnergyVad, Vad};
use hush_core::config::{EngineChoice, GpuPolicy, NormalizerChoice};
use hush_core::normalize::Normalizer;
use hush_core::stt::{Backend, SttEngine};
use hush_normalize::openai_http::Dialect;
use hush_normalize::{HttpConfig, NormalizerChain, OpenAiHttpNormalizer};
use hush_stt::models::{self, ModelManifest};
use hush_stt::{TranscribeCppEngine, transcribe_cpp};

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
            bail!("{} is not downloaded; `hush doctor` fetches it", model.id);
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
    pub load: Duration,
    pub warm_up: Duration,
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

/// Blocks for seconds.
pub fn load_engine(
    choice: &EngineChoice,
    model_path: &Path,
) -> Result<(Box<dyn SttEngine>, EngineSummary)> {
    let started = Instant::now();
    let (mut engine, fallback) = match choice.gpu {
        GpuPolicy::CpuOnly => (
            TranscribeCppEngine::new(model_path, Backend::Cpu, None)?,
            None,
        ),
        GpuPolicy::RequireGpu => (
            TranscribeCppEngine::new(model_path, Backend::Vulkan, choice.gpu_device)
                .context("engine.gpu = \"require-gpu\" and the GPU backend did not load")?,
            None,
        ),
        GpuPolicy::PreferGpu => {
            match TranscribeCppEngine::new(model_path, Backend::Vulkan, choice.gpu_device) {
                Ok(e) => (e, None),
                Err(gpu_err) => {
                    tracing::warn!(error = %gpu_err, "GPU backend did not load; falling back to the CPU");
                    let e = TranscribeCppEngine::new(model_path, Backend::Cpu, None)
                        .with_context(|| format!("GPU failed ({gpu_err}) and so did the CPU"))?;
                    (e, Some(gpu_err.to_string()))
                }
            }
        }
    };
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
    };
    Ok((Box::new(engine), summary))
}

pub fn describe_vulkan_devices(selected: Option<usize>) -> Vec<String> {
    transcribe_cpp::vulkan_devices()
        .into_iter()
        .map(|d| {
            let mark = if selected == Some(d.index) {
                "  <- engine.gpu_device"
            } else {
                ""
            };
            format!(
                "vulkan {}: {} [{}] {:.1} GiB{mark}",
                d.index,
                d.description.trim(),
                d.device_id.as_deref().unwrap_or("no bus id"),
                d.memory_total as f64 / (1u64 << 30) as f64
            )
        })
        .collect()
}

pub fn http_config(choice: &NormalizerChoice) -> Option<HttpConfig> {
    match choice {
        NormalizerChoice::Rules => None,
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
    fn unreachable_server_fails_the_probe() {
        // Nothing listens on the discard port on a test machine.
        let cfg = HttpConfig::new("http://127.0.0.1:9/v1", "m");
        assert!(probe_http(&cfg).is_err());
    }
}
