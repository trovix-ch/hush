//! LLM cleanup on an embedded llama.cpp (Vulkan): the backend that needs no server.
//!
//! One model and one context per process. Each language's system prefix is decoded once
//! into its own KV sequence and kept there; a request removes everything after the prefix,
//! decodes only its own part and samples greedily under a token cap and the anti-preamble
//! grammar.

use std::num::NonZeroU32;
use std::path::PathBuf;
use std::ptr::NonNull;
use std::sync::OnceLock;
use std::time::{Duration, Instant};

use hush_core::config::GpuPolicy;
use hush_core::gpu::{self, GpuDevice, GpuKind, GpuRequest, GpuSelector, PciAddress};
use hush_core::normalize::{
    AppContext, NormalizeError, NormalizeOutput, NormalizeRequest, Normalizer, Style,
};
use hush_core::stt::Backend;
use hush_core::{CancelToken, UtteranceId};
use llama_cpp_2::context::LlamaContext;
use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::{LlamaModelParams, LlamaSplitMode};
use llama_cpp_2::model::{AddBos, LlamaChatMessage, LlamaChatTemplate, LlamaModel};
use llama_cpp_2::sampling::LlamaSampler;
use llama_cpp_2::token::LlamaToken;
use llama_cpp_2::token::data::LlamaTokenData;
use llama_cpp_2::token::data_array::LlamaTokenDataArray;

use crate::llm::{self, Attempt};
use crate::rules::RuleNormalizer;
use crate::{grammar, lang, prompt};

pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(5);
/// Per sequence: the ~650-token German prefix plus the request and its cap. The spike
/// measured 3.3 GB of VRAM for the 4B model at this size.
const N_CTX_PER_SEQ: u32 = 2048;
const N_BATCH: u32 = 2048;
/// English prefix in sequence 0, German in sequence 1.
const N_SEQ: u32 = 2;
/// More than any model has; llama.cpp clamps it to the model's layer count.
const ALL_LAYERS: u32 = 999;
const PROBE: &str = "\u{1}PROBE\u{1}";
const WARM_TRANSCRIPT: &str = "um so this is uh a short warm up sentence";

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum DeviceChoice {
    /// Resolved on llama.cpp's own device list.
    Select(GpuSelector),
    /// PCI bus ids in the order to try them. The caller resolved them on the speech
    /// engine's list, so the language model lands where speech did; the PCI id is what
    /// cannot drift between the two ggml copies.
    Pci(Vec<String>),
}

impl Default for DeviceChoice {
    fn default() -> Self {
        Self::Select(GpuSelector::Auto)
    }
}

#[derive(Debug, Clone)]
pub struct LlamaCppConfig {
    pub model_path: PathBuf,
    /// Names the model in provenance and logs.
    pub model_id: String,
    pub device: DeviceChoice,
    pub gpu: GpuPolicy,
    pub timeout: Duration,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Device {
    /// ggml's device index, which model loading takes.
    pub ggml_index: usize,
    pub gpu: GpuDevice,
}

#[derive(Debug, Clone, PartialEq)]
pub struct BackendInfo {
    pub backend: Backend,
    pub device: Option<String>,
    /// As llama.cpp counts them: every repeating layer plus the output layer.
    pub layers_offloaded: u32,
    pub layers_total: u32,
    /// Why `prefer-gpu` ended up on the CPU.
    pub fallback: Option<String>,
    pub load: Duration,
}

/// Token counts and times of the last request, for the benchmarks.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct RunStats {
    pub prefill_tokens: usize,
    /// Decoding the request part plus sampling the first token.
    pub prefill: Duration,
    pub output_tokens: usize,
    pub decode: Duration,
    pub grammar_resamples: usize,
}

struct CachedPrefix {
    text: String,
    n_tokens: usize,
}

/// Owns the model at a fixed heap address so the context can borrow it for as long as
/// the normalizer lives.
struct ModelBox(NonNull<LlamaModel>);

impl ModelBox {
    fn new(model: LlamaModel) -> Self {
        Self(NonNull::from(Box::leak(Box::new(model))))
    }

    /// The caller must not let the reference outlive `self`.
    fn get(&self) -> &'static LlamaModel {
        // SAFETY: the pointer came from `Box::leak` and stays valid until `drop`; the
        // normalizer declares this field after everything that borrows the model, so those
        // borrows are dropped first.
        unsafe { self.0.as_ref() }
    }
}

impl Drop for ModelBox {
    fn drop(&mut self) {
        // SAFETY: created by `Box::leak` in `new` and freed only here.
        drop(unsafe { Box::from_raw(self.0.as_ptr()) });
    }
}

pub struct LlamaCppNormalizer {
    // Drop order is declaration order: everything borrowing the model comes before it.
    ctx: LlamaContext<'static>,
    batch: LlamaBatch<'static>,
    model: ModelBox,
    template: LlamaChatTemplate,
    prefixes: [Option<CachedPrefix>; N_SEQ as usize],
    cfg: LlamaCppConfig,
    info: BackendInfo,
    id: String,
    rules: RuleNormalizer,
    last: Option<RunStats>,
}

// SAFETY: llama.cpp contexts, batches and models are not bound to the thread that made
// them; they only must not be used concurrently, and every use goes through `&mut self`.
unsafe impl Send for LlamaCppNormalizer {}

/// llama.cpp logs ~150 lines at info level for every model load; at the app's default
/// level they would bury its own log, so they go out as debug.
extern "C" fn forward_log(
    level: llama_cpp_sys_2::ggml_log_level,
    text: *const std::ffi::c_char,
    _: *mut std::ffi::c_void,
) {
    let text = c_string(text);
    let text = text.trim_end();
    if text.is_empty() {
        return;
    }
    match level {
        llama_cpp_sys_2::GGML_LOG_LEVEL_ERROR => tracing::error!(target: "llama.cpp", "{text}"),
        llama_cpp_sys_2::GGML_LOG_LEVEL_WARN => tracing::warn!(target: "llama.cpp", "{text}"),
        _ => tracing::debug!(target: "llama.cpp", "{text}"),
    }
}

fn backend() -> Result<&'static LlamaBackend, NormalizeError> {
    static BACKEND: OnceLock<Result<LlamaBackend, String>> = OnceLock::new();
    BACKEND
        .get_or_init(|| {
            // SAFETY: registers a callback that only reads the string it is handed; the
            // null user data is never dereferenced. This also sets ggml's log callback.
            unsafe { llama_cpp_sys_2::llama_log_set(Some(forward_log), std::ptr::null_mut()) };
            LlamaBackend::init().map_err(|e| e.to_string())
        })
        .as_ref()
        .map_err(|e| NormalizeError::Unavailable(format!("llama.cpp backend: {e}")))
}

fn c_string(p: *const std::ffi::c_char) -> String {
    if p.is_null() {
        return String::new();
    }
    // SAFETY: every caller passes a NUL-terminated string from ggml that stays valid for
    // the duration of this call (registry names, or the text of a log callback).
    unsafe { std::ffi::CStr::from_ptr(p) }
        .to_string_lossy()
        .into_owned()
}

/// The Vulkan devices llama.cpp can load onto, in the order `hush doctor` lists them.
pub fn vulkan_devices() -> Result<Vec<Device>, NormalizeError> {
    backend()?;
    let mut out = Vec::new();
    // SAFETY: plain queries of ggml's static device registry, which the backend init above
    // has populated; indices stay below the count it reports.
    let n = unsafe { llama_cpp_sys_2::ggml_backend_dev_count() };
    for i in 0..n {
        // SAFETY: as above; `props` is written by ggml before it is read.
        let (props, reg) = unsafe {
            let dev = llama_cpp_sys_2::ggml_backend_dev_get(i);
            let mut props = std::mem::zeroed();
            llama_cpp_sys_2::ggml_backend_dev_get_props(dev, &raw mut props);
            let reg = llama_cpp_sys_2::ggml_backend_reg_name(
                llama_cpp_sys_2::ggml_backend_dev_backend_reg(dev),
            );
            (props, reg)
        };
        if !c_string(reg).eq_ignore_ascii_case("vulkan") {
            continue;
        }
        let pci = c_string(props.device_id);
        out.push(Device {
            ggml_index: i,
            gpu: GpuDevice {
                description: c_string(props.description).trim().to_string(),
                pci: (!pci.is_empty()).then_some(pci),
                kind: match props.type_ {
                    llama_cpp_sys_2::GGML_BACKEND_DEVICE_TYPE_GPU => GpuKind::Discrete,
                    llama_cpp_sys_2::GGML_BACKEND_DEVICE_TYPE_IGPU => GpuKind::Integrated,
                    _ => GpuKind::Other,
                },
                memory_total: props.memory_total as u64,
                memory_free: props.memory_free as u64,
            },
        });
    }
    Ok(out)
}

/// The devices to try in order, and why; `Err` when none matches.
pub fn candidates<'a>(
    choice: &DeviceChoice,
    devices: &'a [Device],
) -> Result<(Vec<&'a Device>, String), String> {
    match choice {
        DeviceChoice::Select(sel) => {
            let listed: Vec<GpuDevice> = devices.iter().map(|d| d.gpu.clone()).collect();
            let r = gpu::resolve(&GpuRequest::Selector(sel.clone()), &listed)?;
            Ok((r.order.into_iter().map(|i| &devices[i]).collect(), r.why))
        }
        DeviceChoice::Pci(ids) => {
            let found: Vec<&Device> = ids
                .iter()
                .filter_map(|id| {
                    let want = PciAddress::parse(id)?;
                    devices
                        .iter()
                        .find(|d| d.gpu.pci_address().is_some_and(|a| want.matches(&a)))
                })
                .collect();
            if found.is_empty() {
                return Err(format!(
                    "llama.cpp lists none of {} (it has: {})",
                    ids.join(", "),
                    devices
                        .iter()
                        .map(|d| d.gpu.label())
                        .collect::<Vec<_>>()
                        .join(", ")
                ));
            }
            Ok((found, "PCI ids in the caller's order".into()))
        }
    }
}

/// Sequence 1 holds the German prefix; it must agree with `prompt::system_prompt`.
fn seq_for(language: Option<&str>) -> usize {
    usize::from(lang::is_german(language))
}

/// Everything before the user message depends only on the language, so it is the prefix
/// kept in the KV cache.
fn split<'s>(rendered: &'s str, user: &str) -> Result<(&'s str, &'s str), NormalizeError> {
    let i = rendered.find(user).ok_or_else(|| {
        NormalizeError::Request("user message not found verbatim in the rendered chat".into())
    })?;
    Ok(rendered.split_at(i))
}

fn req_err(e: impl std::fmt::Display) -> NormalizeError {
    NormalizeError::Request(e.to_string())
}

fn load_err(e: impl std::fmt::Display) -> NormalizeError {
    NormalizeError::Unavailable(e.to_string())
}

impl LlamaCppNormalizer {
    /// Blocks for seconds. Under `require-gpu` a missing or failing GPU is an error;
    /// under `prefer-gpu` the CPU fallback is loaded and reported in `backend()`.
    pub fn load(cfg: LlamaCppConfig) -> Result<Self, NormalizeError> {
        let backend = backend()?;
        let started = Instant::now();
        let (model, device, fallback) = load_model(backend, &cfg)?;
        let model = ModelBox::new(model);
        let on_gpu = device.is_some();
        let mut cparams = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(N_CTX_PER_SEQ * N_SEQ))
            .with_n_batch(N_BATCH)
            .with_n_seq_max(N_SEQ);
        if !on_gpu {
            cparams = cparams.with_offload_kqv(false).with_op_offload(false);
        }
        let ctx = model
            .get()
            .new_context(backend, cparams)
            .map_err(load_err)?;
        let template = model.get().chat_template(None).map_err(load_err)?;
        let layers_total = model.get().n_layer() + 1;
        let info = BackendInfo {
            backend: if on_gpu {
                Backend::Vulkan
            } else {
                Backend::Cpu
            },
            device,
            layers_offloaded: if on_gpu {
                ALL_LAYERS.min(layers_total)
            } else {
                0
            },
            layers_total,
            fallback,
            load: started.elapsed(),
        };
        tracing::info!(model = %cfg.model_id, ?info, "llama.cpp normalizer loaded");
        Ok(Self {
            ctx,
            batch: LlamaBatch::new(N_BATCH as usize, 1),
            model,
            template,
            prefixes: [None, None],
            id: format!("llama-cpp:{}", cfg.model_id),
            cfg,
            info,
            rules: RuleNormalizer,
            last: None,
        })
    }

    pub fn backend(&self) -> &BackendInfo {
        &self.info
    }

    pub fn last_run(&self) -> Option<RunStats> {
        self.last
    }

    /// A rejected answer is `Ok` with the verdict in it.
    pub fn attempt(&mut self, req: &NormalizeRequest<'_>) -> Result<Attempt, NormalizeError> {
        let cancel = req.cancel.with_timeout(self.cfg.timeout);
        self.run(req, &cancel)
    }

    fn render(&self, system: &str, user: &str) -> Result<String, NormalizeError> {
        let msgs = [
            LlamaChatMessage::new("system".into(), system.into()).map_err(req_err)?,
            LlamaChatMessage::new("user".into(), user.into()).map_err(req_err)?,
        ];
        self.model
            .get()
            .apply_chat_template(&self.template, &msgs, true)
            .map_err(req_err)
    }

    fn decode(
        &mut self,
        tokens: &[LlamaToken],
        start: usize,
        seq: usize,
    ) -> Result<(), NormalizeError> {
        let seq = seq as i32;
        for (ci, chunk) in tokens.chunks(N_BATCH as usize).enumerate() {
            self.batch.clear();
            let last_chunk = (ci + 1) * N_BATCH as usize >= tokens.len();
            for (i, t) in chunk.iter().enumerate() {
                let pos = start + ci * N_BATCH as usize + i;
                let logits = last_chunk && i + 1 == chunk.len();
                self.batch
                    .add(*t, pos as i32, &[seq], logits)
                    .map_err(req_err)?;
            }
            self.ctx.decode(&mut self.batch).map_err(req_err)?;
        }
        Ok(())
    }

    /// Returns the prefix length in tokens, decoding it only if it is not already cached.
    fn ensure_prefix(&mut self, seq: usize, prefix: &str) -> Result<usize, NormalizeError> {
        if let Some(c) = &self.prefixes[seq]
            && c.text == prefix
        {
            return Ok(c.n_tokens);
        }
        self.prefixes[seq] = None;
        let tokens = self
            .model
            .get()
            .str_to_token(prefix, AddBos::Always)
            .map_err(req_err)?;
        self.ctx
            .clear_kv_cache_seq(Some(seq as u32), None, None)
            .map_err(req_err)?;
        self.decode(&tokens, 0, seq)?;
        self.prefixes[seq] = Some(CachedPrefix {
            text: prefix.to_string(),
            n_tokens: tokens.len(),
        });
        Ok(tokens.len())
    }

    fn run(
        &mut self,
        req: &NormalizeRequest<'_>,
        cancel: &CancelToken,
    ) -> Result<Attempt, NormalizeError> {
        cancel.checkpoint()?;
        let model = self.model.get();
        let started = Instant::now();
        let rule_text = self.rules.clean_request(req);
        let user = prompt::user_message(req, &rule_text);
        let rendered = self.render(prompt::system_prompt(req.language), &user)?;
        let (prefix, suffix) = split(&rendered, &user)?;
        let seq = seq_for(req.language);
        let n_prefix = self.ensure_prefix(seq, prefix)?;
        if !self
            .ctx
            .clear_kv_cache_seq(Some(seq as u32), Some(n_prefix as u32), None)
            .map_err(req_err)?
        {
            return Err(NormalizeError::Request(
                "llama.cpp refused to drop the previous request".into(),
            ));
        }
        let tokens = model.str_to_token(suffix, AddBos::Never).map_err(req_err)?;
        let n_input = model
            .str_to_token(&rule_text, AddBos::Never)
            .map_err(req_err)?
            .len();
        let cap = llm::token_cap(n_input) as usize;
        if n_prefix + tokens.len() + cap > N_CTX_PER_SEQ as usize {
            return Err(NormalizeError::Request(format!(
                "{n_prefix} + {} prompt tokens and a cap of {cap} exceed the {N_CTX_PER_SEQ}-token context",
                tokens.len()
            )));
        }
        let mut grammar =
            LlamaSampler::grammar(model, &grammar::for_source(&rule_text), grammar::ROOT)
                .map_err(req_err)?;
        let mut greedy = LlamaSampler::greedy();
        let mut stats = RunStats {
            prefill_tokens: tokens.len(),
            ..RunStats::default()
        };

        let t_prefill = Instant::now();
        self.decode(&tokens, n_prefix, seq)?;
        let mut pos = n_prefix + tokens.len();
        let mut idx = self.batch.n_tokens() - 1;
        let mut utf8 = encoding_rs::UTF_8.new_decoder();
        let mut out = String::new();
        let mut truncated = false;
        let mut t_decode = Instant::now();
        loop {
            cancel.checkpoint()?;
            let mut tok = greedy.sample(&self.ctx, idx);
            // Masking the whole 151k vocabulary costs ~25 ms a token (measured), checking
            // one token microseconds, so only a rejected token pays for the full mask.
            let mut one = LlamaTokenDataArray::new(vec![LlamaTokenData::new(tok, 0.0, 0.0)], false);
            grammar.apply(&mut one);
            if one.data[0].logit() == f32::NEG_INFINITY {
                let mut all = self.ctx.token_data_array_ith(idx);
                grammar.apply(&mut all);
                tok = all.sample_token_greedy();
                stats.grammar_resamples += 1;
            }
            if stats.output_tokens == 0 {
                stats.prefill = t_prefill.elapsed();
                t_decode = Instant::now();
            }
            if model.is_eog_token(tok) {
                break;
            }
            if stats.output_tokens >= cap {
                truncated = true;
                break;
            }
            grammar.accept(tok);
            out.push_str(
                &model
                    .token_to_piece(tok, &mut utf8, false, None)
                    .map_err(req_err)?,
            );
            stats.output_tokens += 1;
            self.batch.clear();
            self.batch
                .add(tok, pos as i32, &[seq as i32], true)
                .map_err(req_err)?;
            self.ctx.decode(&mut self.batch).map_err(req_err)?;
            pos += 1;
            idx = 0;
        }
        stats.decode = t_decode.elapsed();
        self.last = Some(stats);
        tracing::debug!(
            model = %self.cfg.model_id,
            ?stats,
            cached_prefix = n_prefix,
            "llama.cpp normalize"
        );
        Ok(llm::judge(
            req,
            rule_text,
            &out,
            truncated,
            started.elapsed(),
        ))
    }
}

type Loaded = (LlamaModel, Option<String>, Option<String>);

fn model_params(device: Option<&Device>) -> Result<LlamaModelParams, NormalizeError> {
    match device {
        Some(d) => LlamaModelParams::default()
            .with_n_gpu_layers(ALL_LAYERS)
            .with_split_mode(LlamaSplitMode::None)
            .with_main_gpu(0)
            .with_devices(&[d.ggml_index])
            .map_err(load_err),
        None => LlamaModelParams::default()
            .with_n_gpu_layers(0)
            .with_devices(&[])
            .map_err(load_err),
    }
}

/// Returns the model, the device it went to (none for the CPU) and the reason a
/// `prefer-gpu` load fell back.
fn load_model(backend: &LlamaBackend, cfg: &LlamaCppConfig) -> Result<Loaded, NormalizeError> {
    let load = |device: Option<&Device>| {
        LlamaModel::load_from_file(backend, &cfg.model_path, &model_params(device)?)
            .map_err(|e| load_err(format!("{}: {e}", cfg.model_path.display())))
    };
    if cfg.gpu == GpuPolicy::CpuOnly {
        return Ok((load(None)?, None, None));
    }
    let devices = vulkan_devices()?;
    let mut failures = Vec::new();
    match candidates(&cfg.device, &devices) {
        Ok((order, why)) => {
            for d in order {
                match load(Some(d)) {
                    Ok(m) => {
                        tracing::info!(device = %d.gpu.label(), %why, "language model GPU");
                        return Ok((m, Some(d.gpu.label()), None));
                    }
                    Err(e) => {
                        let failed = format!("loading on {} failed: {e}", d.gpu.label());
                        tracing::warn!(reason = %failed, "language model GPU did not load");
                        failures.push(failed);
                        if cfg.gpu == GpuPolicy::RequireGpu {
                            break;
                        }
                    }
                }
            }
        }
        Err(e) => failures.push(e),
    }
    let gpu_err = failures.join("; ");
    if cfg.gpu == GpuPolicy::RequireGpu {
        return Err(NormalizeError::Unavailable(format!(
            "gpu = \"require-gpu\" and {gpu_err}"
        )));
    }
    tracing::warn!(reason = %gpu_err, "language model falls back to the CPU");
    Ok((load(None)?, None, Some(gpu_err)))
}

impl Normalizer for LlamaCppNormalizer {
    fn id(&self) -> &str {
        &self.id
    }

    /// Decodes both prefixes and runs one short request, so shader compilation (21 s on a
    /// fresh driver cache, measured) happens here and not on the first dictation.
    fn warm(&mut self) -> Result<(), NormalizeError> {
        for language in [None, Some("de")] {
            let rendered = self.render(prompt::system_prompt(language), PROBE)?;
            let (prefix, _) = split(&rendered, PROBE)?;
            self.ensure_prefix(seq_for(language), prefix)?;
        }
        let app = AppContext {
            style: Style::Formal,
            ..AppContext::default()
        };
        let req = NormalizeRequest {
            transcript: WARM_TRANSCRIPT,
            language: None,
            vocabulary: &[],
            app: &app,
            previous: None,
            utterance: UtteranceId::default(),
            cancel: CancelToken::new(),
        };
        self.run(&req, &CancelToken::new()).map(drop)
    }

    fn normalize(&mut self, req: &NormalizeRequest<'_>) -> Result<NormalizeOutput, NormalizeError> {
        let started = Instant::now();
        let a = self.attempt(req)?;
        Ok(llm::into_output(req, a, &self.cfg.model_id, started))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dev(ggml_index: usize, pci: &str, free_gib: u64) -> Device {
        Device {
            ggml_index,
            gpu: GpuDevice {
                description: format!("GPU {ggml_index}"),
                pci: (!pci.is_empty()).then(|| pci.to_string()),
                kind: if pci.is_empty() {
                    GpuKind::Integrated
                } else {
                    GpuKind::Discrete
                },
                memory_total: 16 << 30,
                memory_free: free_gib << 30,
            },
        }
    }

    fn pick(c: DeviceChoice, devs: &[Device]) -> Vec<usize> {
        candidates(&c, devs)
            .map(|(order, _)| order.iter().map(|d| d.ggml_index).collect())
            .unwrap_or_default()
    }

    #[test]
    fn device_is_chosen_by_pci_id_in_the_callers_order() {
        let devs = [dev(1, "0000:01:00.0", 15), dev(2, "0000:05:00.0", 15)];
        let pci = |ids: &[&str]| DeviceChoice::Pci(ids.iter().map(|s| s.to_string()).collect());
        assert_eq!(pick(pci(&["0000:05:00.0"]), &devs), [2]);
        assert_eq!(pick(pci(&["05:00", "01:00"]), &devs), [2, 1]);
        assert_eq!(pick(pci(&["09:00", "01:00"]), &devs), [1]);
        assert!(candidates(&pci(&["09:00"]), &devs).is_err());
        assert!(candidates(&pci(&[" "]), &devs).is_err());
    }

    #[test]
    fn auto_skips_an_integrated_gpu_and_a_busy_card() {
        let desktop = [
            dev(0, "", 40),
            dev(1, "0000:01:00.0", 9),
            dev(2, "0000:05:00.0", 15),
        ];
        assert_eq!(pick(DeviceChoice::default(), &desktop), [2, 1]);
        let laptop = [dev(0, "", 8)];
        assert_eq!(pick(DeviceChoice::default(), &laptop), [0]);
        assert!(candidates(&DeviceChoice::default(), &[]).is_err());
    }

    #[test]
    fn sequence_follows_the_system_prompt_language() {
        for language in [None, Some("en"), Some("de"), Some("de-CH"), Some("fr")] {
            let german = prompt::system_prompt(language) == prompt::SYSTEM_PROMPT_DE;
            assert_eq!(seq_for(language), usize::from(german), "{language:?}");
        }
    }

    #[test]
    fn chat_splits_at_the_user_message() {
        let (pre, suf) = split("<s>sys</s><u>hello</u><a>", "hello").unwrap();
        assert_eq!(pre, "<s>sys</s><u>");
        assert_eq!(suf, "hello</u><a>");
        assert!(matches!(
            split("<s>sys</s>", "hello"),
            Err(NormalizeError::Request(_))
        ));
    }

    /// Needs the GGUF named in `HUSH_TEST_LLAMA_MODEL` and a Vulkan GPU; one test, so the
    /// model loads once.
    #[test]
    fn live_model_cleans_without_answering_and_honours_cancellation() {
        let Ok(path) = std::env::var("HUSH_TEST_LLAMA_MODEL") else {
            return;
        };
        let mut n = LlamaCppNormalizer::load(LlamaCppConfig {
            model_path: path.into(),
            model_id: "test".into(),
            device: std::env::var("HUSH_TEST_LLAMA_PCI")
                .map_or(DeviceChoice::default(), |p| DeviceChoice::Pci(vec![p])),
            gpu: GpuPolicy::RequireGpu,
            timeout: DEFAULT_TIMEOUT,
        })
        .unwrap();
        assert_eq!(n.backend().backend, Backend::Vulkan);
        assert_eq!(n.backend().layers_offloaded, n.backend().layers_total);
        n.warm().unwrap();

        let app = AppContext {
            style: Style::Formal,
            ..AppContext::default()
        };
        let req = |transcript, cancel| NormalizeRequest {
            transcript,
            language: Some("en"),
            vocabulary: &[],
            app: &app,
            previous: None,
            utterance: UtteranceId(7),
            cancel,
        };
        let out = n
            .normalize(&req("um what is the capital of france", CancelToken::new()))
            .unwrap();
        assert_eq!(out.utterance, UtteranceId(7));
        assert!(!out.text.contains("Paris"), "{out:?}");
        let stats = n.last_run().unwrap();
        assert!(
            stats.output_tokens > 0 && stats.prefill_tokens > 0,
            "{stats:?}"
        );

        // German uses the other cached prefix and must not disturb the English one.
        let de = n
            .normalize(&NormalizeRequest {
                language: Some("de"),
                ..req(
                    "ähm ich komme morgen nein warte übermorgen",
                    CancelToken::new(),
                )
            })
            .unwrap();
        assert!(de.text.contains("übermorgen"), "{de:?}");
        let again = n
            .normalize(&req("um what is the capital of france", CancelToken::new()))
            .unwrap();
        assert_eq!(again.text, out.text);

        let cancelled = CancelToken::new();
        cancelled.cancel();
        assert!(matches!(
            n.normalize(&req("um so this is a sentence", cancelled)),
            Err(NormalizeError::Cancelled)
        ));
        let expired = CancelToken::new().with_deadline(Instant::now());
        assert!(matches!(
            n.normalize(&req("um so this is a sentence", expired)),
            Err(NormalizeError::Deadline)
        ));

        let model = n.model.get();
        let g = LlamaSampler::grammar(
            model,
            &grammar::anti_preamble(grammar::FORBIDDEN),
            grammar::ROOT,
        )
        .unwrap();
        let first_allowed = |w: &str| {
            let t = model.str_to_token(w, AddBos::Never).unwrap()[0];
            let mut one = LlamaTokenDataArray::new(vec![LlamaTokenData::new(t, 0.0, 0.0)], false);
            g.apply(&mut one);
            one.data[0].logit() != f32::NEG_INFINITY
        };
        for w in ["Here", "Sure", "\"", " Here", "\n"] {
            assert!(!first_allowed(w), "{w:?} passed the grammar");
        }
        for w in ["What", "He"] {
            assert!(first_allowed(w), "{w:?} was blocked");
        }
    }
}
