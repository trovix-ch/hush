//! Milestone-2 gate spike: embedded llama.cpp (Vulkan) with the HTTP path's prompt
//! contract, prefix-state caching, greedy decoding, a hard token cap and an optional
//! anti-preamble grammar. Throwaway; the README is the record.

mod grammar;
mod prompt;

use std::num::NonZeroU32;
use std::path::PathBuf;
use std::process::Command;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, anyhow, bail};
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
use serde::Deserialize;
use wl_core::normalize::{AppContext, NormalizeRequest, Rejection, Style};
use wl_normalize::{rules::RuleNormalizer, validate};

const USAGE: &str = "usage: spike-llama-cpp <model.gguf> [--pci BUS] [--device N] [--runs N] \
[--fixtures PATH] [--case SUBSTRING]

  --pci      pick the Vulkan device whose PCI id contains this (default 05:00)
  --device   ggml device index instead of --pci
  --runs     timed runs per case and mode (default 3)
  --kv-unified  one KV buffer shared by both prefix sequences (default: one per sequence)

Modes run: prefix cache on, cache on + grammar, cache off.";

/// Room for the ~900-token prefix plus the longest request; per sequence.
const N_CTX_PER_SEQ: u32 = 2048;
const N_BATCH: u32 = 2048;

struct Args {
    model: PathBuf,
    pci: String,
    device: Option<usize>,
    runs: usize,
    fixtures: PathBuf,
    case_filter: Option<String>,
    kv_unified: bool,
}

fn parse_args() -> Result<Args> {
    let mut it = std::env::args().skip(1);
    let model = it.next().context(USAGE)?;
    let mut a = Args {
        model: model.into(),
        pci: "05:00".into(),
        device: None,
        runs: 3,
        fixtures: PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tools/bench-normalize/fixtures/transcripts.toml"),
        case_filter: None,
        kv_unified: false,
    };
    while let Some(flag) = it.next() {
        let mut value = || it.next().with_context(|| format!("{flag} needs a value"));
        match flag.as_str() {
            "--pci" => a.pci = value()?,
            "--device" => a.device = Some(value()?.parse()?),
            "--runs" => a.runs = value()?.parse()?,
            "--fixtures" => a.fixtures = value()?.into(),
            "--case" => a.case_filter = Some(value()?),
            "--kv-unified" => a.kv_unified = true,
            other => bail!("unknown argument {other}\n{USAGE}"),
        }
    }
    Ok(a)
}

#[derive(Debug, Deserialize)]
struct Fixtures {
    #[serde(rename = "case")]
    cases: Vec<Case>,
}

#[derive(Debug, Deserialize)]
struct Case {
    name: String,
    input: String,
    expected: Option<String>,
    language: Option<String>,
    #[serde(default)]
    style: Style,
    #[serde(default)]
    vocabulary: Vec<String>,
    #[serde(default)]
    must_not_contain: Vec<String>,
    previous: Option<String>,
    app: Option<String>,
}

/// Everything derived from a case before the model sees it.
struct Prepared<'a> {
    case: &'a Case,
    rule_text: String,
    system: &'static str,
    user: String,
    /// 1.5 x transcript tokens + 20.
    max_tokens: usize,
}

fn prepare<'a>(c: &'a Case, model: &LlamaModel) -> Result<Prepared<'a>> {
    let app = AppContext {
        exe: c.app.clone(),
        window_title: None,
        style: c.style,
    };
    let req = NormalizeRequest {
        transcript: &c.input,
        language: c.language.as_deref(),
        vocabulary: &c.vocabulary,
        app: &app,
        previous: c.previous.as_deref(),
        utterance: wl_core::UtteranceId::FIRST,
        cancel: wl_core::CancelToken::new(),
    };
    let rule_text = RuleNormalizer.clean_request(&req);
    let system = prompt::system_prompt(req.language);
    // The copied constant must be the crate's, byte for byte.
    if system != wl_normalize::prompt::system_prompt(req.language) {
        bail!("system prompt drifted from crates/normalize/src/prompt.rs");
    }
    let user = wl_normalize::prompt::user_message(&req, &rule_text);
    let n_in = model.str_to_token(&rule_text, AddBos::Never)?.len();
    Ok(Prepared {
        case: c,
        max_tokens: (n_in as f64 * 1.5).ceil() as usize + 20,
        rule_text,
        system,
        user,
    })
}

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
struct Mode {
    cache: bool,
    grammar: bool,
}

impl Mode {
    fn label(self) -> String {
        format!(
            "prefix cache {}, grammar {}",
            if self.cache { "ON" } else { "OFF" },
            if self.grammar { "ON" } else { "OFF" }
        )
    }
}

/// A decoded system prefix living in one KV sequence.
struct CachedPrefix {
    text: String,
    n_tokens: usize,
    seq: i32,
}

struct Run {
    text: String,
    truncated: bool,
    prefill_tokens: usize,
    prefill: Duration,
    decode_tokens: usize,
    decode: Duration,
    total: Duration,
    cache_hit: bool,
    grammar_resamples: usize,
}

struct Engine<'m> {
    model: &'m LlamaModel,
    ctx: LlamaContext<'m>,
    template: LlamaChatTemplate,
    batch: LlamaBatch<'m>,
    prefixes: Vec<CachedPrefix>,
}

impl<'m> Engine<'m> {
    fn render(&self, system: &str, user: &str) -> Result<String> {
        let msgs = [
            LlamaChatMessage::new("system".into(), system.into())?,
            LlamaChatMessage::new("user".into(), user.into())?,
        ];
        Ok(self.model.apply_chat_template(&self.template, &msgs, true)?)
    }

    /// Splits the rendered chat at the user message: everything before it depends only on
    /// the language, so it is the cacheable prefix.
    fn split<'s>(rendered: &'s str, user: &str) -> Result<(&'s str, &'s str)> {
        let i = rendered
            .find(user)
            .ok_or_else(|| anyhow!("user message not found verbatim in rendered chat"))?;
        Ok(rendered.split_at(i))
    }

    fn decode_tokens(&mut self, tokens: &[LlamaToken], start_pos: usize, seq: i32) -> Result<()> {
        for (ci, chunk) in tokens.chunks(N_BATCH as usize).enumerate() {
            self.batch.clear();
            let last_chunk = (ci + 1) * N_BATCH as usize >= tokens.len();
            for (i, t) in chunk.iter().enumerate() {
                let pos = start_pos + ci * N_BATCH as usize + i;
                let logits = last_chunk && i == chunk.len() - 1;
                self.batch.add(*t, pos as i32, &[seq], logits)?;
            }
            self.ctx.decode(&mut self.batch)?;
        }
        Ok(())
    }

    /// Decodes the fixed prefix for a system prompt into its own sequence, once.
    fn cache_prefix(&mut self, system: &str, seq: i32) -> Result<Duration> {
        let rendered = self.render(system, "\u{1}PROBE\u{1}")?;
        let (prefix, _) = Self::split(&rendered, "\u{1}PROBE\u{1}")?;
        let toks = self.model.str_to_token(prefix, AddBos::Always)?;
        self.ctx.clear_kv_cache_seq(Some(seq as u32), None, None)?;
        let t = Instant::now();
        self.decode_tokens(&toks, 0, seq)?;
        // Forces the GPU work to finish so the time is real.
        let _ = self.ctx.get_logits_ith(self.batch.n_tokens() - 1);
        let took = t.elapsed();
        self.prefixes.push(CachedPrefix {
            text: prefix.to_string(),
            n_tokens: toks.len(),
            seq,
        });
        Ok(took)
    }

    fn run(&mut self, p: &Prepared<'_>, mode: Mode) -> Result<Run> {
        let start = Instant::now();
        let rendered = self.render(p.system, &p.user)?;
        let (prefix, suffix) = Self::split(&rendered, &p.user)?;

        let hit = if mode.cache {
            self.prefixes.iter().position(|c| c.text == prefix)
        } else {
            None
        };
        let (tokens, start_pos, seq) = match hit {
            Some(i) => {
                let c = &self.prefixes[i];
                let (n, seq) = (c.n_tokens, c.seq);
                // Drop the previous request, keep the prefix.
                if !self
                    .ctx
                    .clear_kv_cache_seq(Some(seq as u32), Some(n as u32), None)?
                {
                    bail!("llama.cpp refused a partial sequence removal");
                }
                (self.model.str_to_token(suffix, AddBos::Never)?, n, seq)
            }
            None => {
                self.ctx.clear_kv_cache();
                (self.model.str_to_token(&rendered, AddBos::Always)?, 0, 0)
            }
        };

        let mut grammar = if mode.grammar {
            let g = grammar::anti_preamble(&grammar::forbidden_for(&p.rule_text));
            Some(LlamaSampler::grammar(self.model, &g, "root")?)
        } else {
            None
        };
        let mut greedy = LlamaSampler::greedy();

        let t_prefill = Instant::now();
        self.decode_tokens(&tokens, start_pos, seq)?;
        let mut pos = start_pos + tokens.len();
        let mut idx = self.batch.n_tokens() - 1;

        let mut decoder = encoding_rs::UTF_8.new_decoder();
        let mut out = String::new();
        let mut n_out = 0usize;
        let mut truncated = false;
        let mut grammar_resamples = 0;
        let mut prefill = Duration::ZERO;
        let mut t_decode = Instant::now();
        loop {
            let mut tok = greedy.sample(&self.ctx, idx);
            if let Some(g) = &mut grammar {
                // Checking one token is cheap; masking the whole 151k vocabulary costs
                // ~25 ms per token (measured), so only a rejected token pays for it.
                // This is what llama.cpp's common sampler does too.
                let mut one = LlamaTokenDataArray::new(vec![LlamaTokenData::new(tok, 0.0, 0.0)], false);
                g.apply(&mut one);
                if one.data[0].logit() == f32::NEG_INFINITY {
                    let mut all = self.ctx.token_data_array_ith(idx);
                    g.apply(&mut all);
                    tok = all.sample_token_greedy();
                    grammar_resamples += 1;
                }
                if !self.model.is_eog_token(tok) {
                    g.accept(tok);
                }
            }
            if n_out == 0 {
                // Time to first token: prefill plus one sample.
                prefill = t_prefill.elapsed();
                t_decode = Instant::now();
            }
            if self.model.is_eog_token(tok) {
                break;
            }
            if n_out >= p.max_tokens {
                truncated = true;
                break;
            }
            out.push_str(&self.model.token_to_piece(tok, &mut decoder, false, None)?);
            n_out += 1;
            self.batch.clear();
            self.batch.add(tok, pos as i32, &[seq], true)?;
            self.ctx.decode(&mut self.batch)?;
            pos += 1;
            idx = 0;
        }
        let decode = t_decode.elapsed();
        Ok(Run {
            text: out,
            truncated,
            prefill_tokens: tokens.len(),
            prefill,
            decode_tokens: n_out,
            decode,
            total: start.elapsed(),
            cache_hit: hit.is_some(),
            grammar_resamples,
        })
    }
}

/// Qwen3 models may emit an empty think block even with thinking off (copied from
/// openai_http.rs).
fn strip_think(s: &str) -> &str {
    let t = s.trim_start();
    if let Some(rest) = t.strip_prefix("<think>")
        && let Some(end) = rest.find("</think>")
    {
        return &rest[end + "</think>".len()..];
    }
    s
}

fn verdict(p: &Prepared<'_>, run: &Run) -> Result<(), Rejection> {
    if run.truncated {
        return Err(Rejection::Truncated);
    }
    let cand = strip_think(&run.text).trim();
    let mut vcfg = validate::Config::new(p.case.language.as_deref(), &p.case.vocabulary);
    vcfg.verbatim = matches!(p.case.style, Style::Code | Style::None);
    validate::validate_with(&p.rule_text, cand, &vcfg).map(drop)
}

fn words(s: &str) -> Vec<String> {
    s.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(str::to_lowercase)
        .collect()
}

/// Nearest-rank, as in bench-normalize.
fn percentile(sorted: &[Duration], q: f64) -> Duration {
    if sorted.is_empty() {
        return Duration::ZERO;
    }
    let rank = ((q * sorted.len() as f64).ceil() as usize).clamp(1, sorted.len());
    sorted[rank - 1]
}

fn ms(d: Duration) -> f64 {
    d.as_secs_f64() * 1000.0
}

fn vram(pci: &str) -> String {
    Command::new("nvidia-smi")
        .args([
            "--query-gpu=pci.bus_id,memory.used",
            "--format=csv,noheader",
        ])
        .output()
        .ok()
        .map(|o| String::from_utf8_lossy(&o.stdout).to_string())
        .unwrap_or_default()
        .lines()
        .filter(|l| l.to_lowercase().contains(&pci.to_lowercase()))
        .collect::<Vec<_>>()
        .join("; ")
}

struct DeviceInfo {
    index: usize,
    name: String,
    description: String,
    pci: String,
    is_gpu: bool,
}

fn devices() -> Vec<DeviceInfo> {
    let mut v = Vec::new();
    let n = unsafe { llama_cpp_sys_2::ggml_backend_dev_count() };
    for i in 0..n {
        let cs = |p: *const std::ffi::c_char| {
            if p.is_null() {
                String::new()
            } else {
                unsafe { std::ffi::CStr::from_ptr(p) }
                    .to_string_lossy()
                    .into_owned()
            }
        };
        let props = unsafe {
            let dev = llama_cpp_sys_2::ggml_backend_dev_get(i);
            let mut props = std::mem::zeroed();
            llama_cpp_sys_2::ggml_backend_dev_get_props(dev, &raw mut props);
            props
        };
        v.push(DeviceInfo {
            index: i,
            name: cs(props.name),
            description: cs(props.description),
            pci: cs(props.device_id),
            is_gpu: props.type_ == llama_cpp_sys_2::GGML_BACKEND_DEVICE_TYPE_GPU,
        });
    }
    v
}

/// The fixtures never tempt the model into a preamble, so prove the grammar separately:
/// which leading tokens it masks, and what it does to a chat that asks for one.
fn grammar_self_test(eng: &mut Engine<'_>) -> Result<()> {
    println!("grammar self-test:");
    let g = grammar::anti_preamble(grammar::FORBIDDEN);
    let s = LlamaSampler::grammar(eng.model, &g, "root")?;
    let mut line = String::new();
    for w in ["Here", "Sure", "Certainly", "\"", "`", " Here", "\n", "What", "Her", "He"] {
        let first = eng.model.str_to_token(w, AddBos::Never)?[0];
        let piece = eng
            .model
            .token_to_piece(first, &mut encoding_rs::UTF_8.new_decoder(), false, None)?;
        let mut one = LlamaTokenDataArray::new(vec![LlamaTokenData::new(first, 0.0, 0.0)], false);
        s.apply(&mut one);
        let allowed = one.data[0].logit() != f32::NEG_INFINITY;
        line.push_str(&format!(
            "{piece:?}={} ",
            if allowed { "allowed" } else { "MASKED" }
        ));
    }
    println!("  first-token mask: {line}");

    let t = Instant::now();
    for _ in 0..20 {
        let _ = LlamaSampler::grammar(eng.model, &g, "root")?;
    }
    let build = t.elapsed() / 20;
    let what = eng.model.str_to_token("What", AddBos::Never)?[0];
    let t = Instant::now();
    for _ in 0..1000 {
        let mut one = LlamaTokenDataArray::new(vec![LlamaTokenData::new(what, 0.0, 0.0)], false);
        s.apply(&mut one);
    }
    let check = t.elapsed() / 1000;
    let mut s2 = LlamaSampler::grammar(eng.model, &g, "root")?;
    let t = Instant::now();
    s2.accept(what);
    let accept = t.elapsed();
    println!(
        "  cost: build {:.2} ms, one-token check {:.1} us, first accept {:.2} ms",
        ms(build),
        check.as_secs_f64() * 1e6,
        ms(accept)
    );
    let bait = Case {
        name: "grammar-bait".into(),
        input: String::new(),
        expected: None,
        language: Some("en".into()),
        style: Style::Formal,
        vocabulary: Vec::new(),
        must_not_contain: Vec::new(),
        previous: None,
        app: None,
    };
    let user = "Start your reply with the exact words \"Here is\" and then write one short sentence about rain.";
    let p = Prepared {
        case: &bait,
        rule_text: user.into(),
        system: "You are a helpful assistant.",
        user: user.into(),
        max_tokens: 40,
    };
    for grammar in [false, true] {
        let r = eng.run(&p, Mode { cache: false, grammar })?;
        println!(
            "  bait, grammar {}: {:?} (resamples {})",
            if grammar { "ON " } else { "OFF" },
            r.text,
            r.grammar_resamples
        );
    }
    println!();
    Ok(())
}

fn main() -> Result<()> {
    let args = parse_args()?;
    let fixtures: Fixtures = toml::from_str(
        &std::fs::read_to_string(&args.fixtures)
            .with_context(|| format!("reading {}", args.fixtures.display()))?,
    )?;
    let cases: Vec<&Case> = fixtures
        .cases
        .iter()
        .filter(|c| {
            args.case_filter
                .as_ref()
                .is_none_or(|f| c.name.contains(f.as_str()))
        })
        .collect();

    #[cfg(feature = "dynamic-backends")]
    {
        println!(
            "dynamic backends from {:?}",
            llama_cpp_2::llama_backend::BACKENDS_DIR
        );
        llama_cpp_2::llama_backend::load_backends();
    }
    let backend = LlamaBackend::init()?;
    println!("# spike-llama-cpp: {}\n", args.model.display());
    let devs = devices();
    for d in &devs {
        println!(
            "device {}: {} [{}] pci={} gpu={}",
            d.index, d.name, d.description, d.pci, d.is_gpu
        );
    }
    let dev = match args.device {
        Some(i) => devs.iter().find(|d| d.index == i),
        None => devs
            .iter()
            .find(|d| d.is_gpu && d.pci.to_lowercase().contains(&args.pci.to_lowercase())),
    }
    .ok_or_else(|| anyhow!("no GPU device matches; refusing to fall back to CPU"))?;
    let pci_filter = if dev.pci.is_empty() {
        args.pci.clone()
    } else {
        dev.pci.clone()
    };
    println!("chosen: device {} ({}, pci {})", dev.index, dev.name, dev.pci);
    let short_pci = pci_filter.trim_start_matches("0000:").to_string();
    println!("VRAM before load: {}", vram(&short_pci));

    let t = Instant::now();
    let mparams = LlamaModelParams::default()
        .with_n_gpu_layers(999)
        .with_split_mode(LlamaSplitMode::None)
        .with_main_gpu(0)
        .with_devices(&[dev.index])?;
    let model = LlamaModel::load_from_file(&backend, &args.model, &mparams)?;
    let load = t.elapsed();
    println!(
        "model: {} layers, {:.2} GB, hybrid={}, recurrent={}; load {:.0} ms",
        model.n_layer(),
        model.size() as f64 / 1e9,
        model.is_hybrid(),
        model.is_recurrent(),
        ms(load)
    );

    let template = model.chat_template(None)?;
    let tmpl_src = template.to_string()?;
    println!(
        "chat template: {} chars, mentions <think>: {}",
        tmpl_src.len(),
        tmpl_src.contains("<think>")
    );

    let cparams = LlamaContextParams::default()
        .with_n_ctx(NonZeroU32::new(N_CTX_PER_SEQ * 2))
        .with_n_batch(N_BATCH)
        .with_n_seq_max(2)
        .with_kv_unified(args.kv_unified);
    let ctx = model.new_context(&backend, cparams)?;
    println!("context: n_ctx {} n_batch {}", ctx.n_ctx(), ctx.n_batch());
    let mut eng = Engine {
        model: &model,
        ctx,
        template,
        batch: LlamaBatch::new(N_BATCH as usize, 2),
        prefixes: Vec::new(),
    };

    let prepared: Vec<Prepared<'_>> = cases
        .iter()
        .map(|c| prepare(c, &model))
        .collect::<Result<_>>()?;

    // Show one rendered prompt, and check that tokenizing prefix and suffix apart gives
    // the same tokens as tokenizing the whole chat (the cache relies on it).
    let mut boundary_mismatch = 0;
    for p in &prepared {
        let rendered = eng.render(p.system, &p.user)?;
        let (pre, suf) = Engine::split(&rendered, &p.user)?;
        let whole = model.str_to_token(&rendered, AddBos::Always)?;
        let mut parts = model.str_to_token(pre, AddBos::Always)?;
        parts.extend(model.str_to_token(suf, AddBos::Never)?);
        if whole != parts {
            boundary_mismatch += 1;
        }
    }
    {
        let p = &prepared[0];
        let r = eng.render(p.system, &p.user)?;
        let (_, suf) = Engine::split(&r, &p.user)?;
        println!("rendered request part of {}: {:?}", p.case.name, suf);
        println!(
            "rendered chat head: {:?}",
            r.chars().take(80).collect::<String>()
        );
    }
    println!("prefix/suffix tokenization mismatches: {boundary_mismatch}\n");

    let t = Instant::now();
    let en = eng.cache_prefix(prompt::SYSTEM_PROMPT_EN, 0)?;
    let first_prefix = t.elapsed();
    let de = eng.cache_prefix(prompt::SYSTEM_PROMPT_DE, 1)?;
    println!(
        "prefix decode (first, includes shader warm-up): EN {} tokens {:.1} ms",
        eng.prefixes[0].n_tokens,
        ms(first_prefix)
    );
    // Again, warm, to get the steady-state cost of decoding the prefix once.
    eng.prefixes.clear();
    let en2 = eng.cache_prefix(prompt::SYSTEM_PROMPT_EN, 0)?;
    let de2 = eng.cache_prefix(prompt::SYSTEM_PROMPT_DE, 1)?;
    println!(
        "prefix decode (warm): EN {} tokens {:.1} ms, DE {} tokens {:.1} ms (first DE {:.1} ms, first EN {:.1} ms)",
        eng.prefixes[0].n_tokens,
        ms(en2),
        eng.prefixes[1].n_tokens,
        ms(de2),
        ms(de),
        ms(en)
    );

    // state_seq_get/set as the alternative to seq_rm, timed once for the record.
    {
        let t = Instant::now();
        let snap = eng
            .ctx
            .state_seq_get(0, llama_cpp_2::context::session::LlamaStateSeqFlags::empty())?;
        let get = t.elapsed();
        eng.ctx.clear_kv_cache_seq(Some(0), None, None)?;
        let t = Instant::now();
        eng.ctx.state_seq_set(&snap, 0)?;
        println!(
            "state_seq_get {:.1} ms / state_seq_set {:.1} ms for {:.1} MB (seq_rm is used instead)",
            ms(get),
            ms(t.elapsed()),
            snap.byte_len() as f64 / 1e6
        );
    }

    // Warm-up: every mode once on the longest case, untimed (shader compiles per shape).
    let longest = prepared
        .iter()
        .max_by_key(|p| p.user.len())
        .expect("fixtures are not empty");
    for mode in [
        Mode { cache: true, grammar: false },
        Mode { cache: true, grammar: true },
        Mode { cache: false, grammar: false },
    ] {
        eng.run(longest, mode)?;
    }
    grammar_self_test(&mut eng)?;
    // Uncached runs trample seq 0; rebuild the prefixes.
    eng.prefixes.clear();
    eng.cache_prefix(prompt::SYSTEM_PROMPT_EN, 0)?;
    eng.cache_prefix(prompt::SYSTEM_PROMPT_DE, 1)?;
    println!("VRAM after load + warm-up: {}\n", vram(&short_pci));

    let modes = [
        Mode { cache: true, grammar: false },
        Mode { cache: true, grammar: true },
        Mode { cache: false, grammar: false },
    ];
    let mut first_outputs: Vec<Vec<String>> = Vec::new();
    let mut summary = Vec::new();
    for mode in modes {
        if mode.cache && eng.prefixes.is_empty() {
            eng.cache_prefix(prompt::SYSTEM_PROMPT_EN, 0)?;
            eng.cache_prefix(prompt::SYSTEM_PROMPT_DE, 1)?;
        }
        println!("## {}\n", mode.label());
        let mut all = Vec::new();
        let mut validated = 0;
        let mut raw_leaks = 0;
        let mut final_leaks = 0;
        let mut expected_hits = 0;
        let mut expected_total = 0;
        let mut unstable = 0;
        let mut misses = 0;
        let mut think = 0;
        let mut resamples = 0;
        let mut outs = Vec::new();
        let (mut pf_tok, mut pf_ms, mut dc_tok, mut dc_ms) = (0usize, 0.0, 0usize, 0.0);
        for p in &prepared {
            let mut runs = Vec::new();
            for _ in 0..args.runs {
                runs.push(eng.run(p, mode)?);
            }
            if !mode.cache {
                // The uncached path decodes into seq 0 and clears everything.
                eng.prefixes.clear();
            }
            let first = &runs[0];
            let v = verdict(p, first);
            let cand = strip_think(&first.text).trim().to_string();
            if first.text.contains("<think>") {
                think += 1;
            }
            resamples += runs.iter().map(|r| r.grammar_resamples).sum::<usize>();
            let final_text = if v.is_ok() { &cand } else { &p.rule_text };
            let leaks = |t: &str| p.case.must_not_contain.iter().any(|m| t.contains(m.as_str()));
            if runs.iter().any(|r| leaks(&r.text)) {
                raw_leaks += 1;
            }
            if leaks(final_text) {
                final_leaks += 1;
            }
            if let Some(e) = &p.case.expected {
                expected_total += 1;
                if words(e) == words(&cand) {
                    expected_hits += 1;
                }
            }
            if runs.iter().any(|r| r.text != first.text) {
                unstable += 1;
            }
            if mode.cache && runs.iter().any(|r| !r.cache_hit) {
                misses += 1;
            }
            if v.is_ok() {
                validated += 1;
            }
            let mut lat: Vec<Duration> = runs.iter().map(|r| r.total).collect();
            lat.sort();
            for r in &runs {
                pf_tok += r.prefill_tokens;
                pf_ms += ms(r.prefill);
                dc_tok += r.decode_tokens;
                dc_ms += ms(r.decode);
            }
            all.extend(&lat);
            println!(
                "- {:<22} prefill {:>4} tok {:>6.1} ms | decode {:>3} tok (cap {:>3}) {:>6.1} ms | p50 {:>6.1} ms | {} | {:?}",
                p.case.name,
                first.prefill_tokens,
                ms(first.prefill),
                first.decode_tokens,
                p.max_tokens,
                ms(first.decode),
                ms(percentile(&lat, 0.5)),
                match &v {
                    Ok(()) => "ok".to_string(),
                    Err(r) => format!("REJECTED [{}] {r}", r.check().as_str()),
                },
                cand
            );
            outs.push(cand);
        }
        all.sort();
        let n = (prepared.len() * args.runs) as f64;
        let line = format!(
            "| {} | {}/{} | {} | {} | {}/{} | {} | {} / {} / {} | {:.0} | {:.0} | {:.1} tok, {:.1} ms | {:.1} tok, {:.1} ms ({:.0} tok/s) |",
            mode.label(),
            validated,
            prepared.len(),
            raw_leaks,
            final_leaks,
            expected_hits,
            expected_total,
            unstable,
            misses,
            think,
            resamples,
            ms(percentile(&all, 0.5)),
            ms(percentile(&all, 0.95)),
            pf_tok as f64 / n,
            pf_ms / n,
            dc_tok as f64 / n,
            dc_ms / n,
            dc_tok as f64 / (dc_ms / 1000.0)
        );
        println!("\n{line}\n");
        summary.push(line);
        first_outputs.push(outs);
    }

    println!("## summary: {}\n", args.model.display());
    println!(
        "| mode | validated | must_not_contain in LLM output | inserted | matches expected | unstable | cache misses / think / grammar resamples | p50 ms | p95 ms | mean prefill | mean decode |"
    );
    println!("|---|---|---|---|---|---|---|---|---|---|---|");
    for l in &summary {
        println!("{l}");
    }
    let differ = |a: usize, b: usize| {
        first_outputs[a]
            .iter()
            .zip(&first_outputs[b])
            .filter(|(x, y)| x != y)
            .count()
    };
    println!(
        "\noutputs differing, cache ON vs OFF: {}; grammar ON vs OFF: {}",
        differ(0, 2),
        differ(0, 1)
    );
    println!("load {:.0} ms; VRAM at end: {}", ms(load), vram(&short_pci));
    Ok(())
}
