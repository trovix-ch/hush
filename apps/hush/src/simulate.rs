//! Drives the real driver, workers and pipeline rather than the engines alone, so the
//! timing covers the channel hops and the insertion chain the user waits on.

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::mpsc::{self, RecvTimeoutError};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use hush_core::config::NormalizerChoice;
use hush_core::normalize::{Provenance, Style};
use hush_core::pipeline::Stage;
use hush_core::stt::SAMPLE_RATE;
use hush_normalize::style_allows_llm;
use hush_platform_windows::focus::{self, WinFocus};
use hush_platform_windows::hook::HotkeyEvent;
use hush_platform_windows::ui_thread::{UiHandle, UiOptions};

use crate::driver::{AppOverride, Driver, DriverParts, Msg, Observed, Workers};
use crate::engines;
use crate::notepad;
use crate::setup::{self, Paths};
use crate::wav::{self, WavRecorder};
use crate::workers::{EngineLoader, NormCmd, Vocabulary};

const RUN_TIMEOUT: Duration = Duration::from_secs(60);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Target {
    Notepad,
    Foreground,
}

pub struct Args {
    wav: PathBuf,
    target: Target,
    runs: usize,
    /// Normalize as if the text went to this exe, whatever the real target is.
    app: Option<String>,
    style: Option<Style>,
    vocab: Vec<String>,
    normalizer: Option<NormalizerChoice>,
}

/// Keeps the config's settings when it already names that kind.
fn parse_normalizer(s: Option<&String>) -> Result<NormalizerChoice> {
    Ok(match s.map(String::as_str) {
        Some("rules") => NormalizerChoice::Rules,
        Some("llama-cpp") => NormalizerChoice::default(),
        Some("http") => NormalizerChoice::Http {
            base_url: "http://127.0.0.1:11434/v1".into(),
            model: "qwen3:4b-instruct-2507-q4_K_M".into(),
            timeout_ms: 5000,
        },
        other => bail!("--normalizer takes rules, llama-cpp or http, got {other:?}"),
    })
}

fn same_kind(a: &NormalizerChoice, b: &NormalizerChoice) -> bool {
    std::mem::discriminant(a) == std::mem::discriminant(b)
}

fn parse_style(s: Option<&String>) -> Result<Style> {
    Ok(match s.map(String::as_str) {
        Some("formal") => Style::Formal,
        Some("casual") => Style::Casual,
        Some("code") => Style::Code,
        Some("none") => Style::None,
        other => bail!("--style takes formal, casual, code or none, got {other:?}"),
    })
}

impl Args {
    pub fn parse(args: &[String]) -> Result<Self> {
        let mut it = args.iter();
        let mut wav = None;
        let mut target = Target::Notepad;
        let mut runs = 1;
        let mut app = None;
        let mut style = None;
        let mut vocab = Vec::new();
        let mut normalizer = None;
        while let Some(a) = it.next() {
            match a.as_str() {
                "--normalizer" => normalizer = Some(parse_normalizer(it.next())?),
                "--app" => {
                    app = Some(
                        it.next()
                            .context("--app needs an executable name")?
                            .trim()
                            .to_ascii_lowercase(),
                    )
                }
                "--style" => style = Some(parse_style(it.next())?),
                "--vocab" => vocab.extend(
                    it.next()
                        .context("--vocab needs comma-separated words")?
                        .split(',')
                        .map(str::trim)
                        .filter(|w| !w.is_empty())
                        .map(str::to_string),
                ),
                "--target" => {
                    target = match it.next().map(String::as_str) {
                        Some("notepad") => Target::Notepad,
                        Some("foreground") => Target::Foreground,
                        other => bail!("--target takes notepad or foreground, got {other:?}"),
                    }
                }
                "--runs" => {
                    runs = it
                        .next()
                        .context("--runs needs a number")?
                        .parse()
                        .context("--runs")?;
                    if runs == 0 {
                        bail!("--runs must be at least 1");
                    }
                }
                s if s.starts_with('-') => bail!("unknown option {s}"),
                s if wav.is_none() => wav = Some(PathBuf::from(s)),
                s => bail!("unexpected argument {s}"),
            }
        }
        Ok(Self {
            wav: wav.context("simulate needs a WAV file")?,
            target,
            runs,
            app,
            style,
            vocab,
            normalizer,
        })
    }
}

#[derive(Default)]
struct RunReport {
    foreground_ok: bool,
    released: Option<Instant>,
    /// Sample counts of the audio sent to the engine before and after the key-up.
    sent_during_hold: Vec<usize>,
    sent_after_release: Vec<usize>,
    done_during_hold: usize,
    inference: Duration,
    transcript: Option<(Instant, String)>,
    normalized: Option<(Instant, String, Provenance)>,
    inserted: Option<(Instant, String)>,
    delivered: bool,
    errors: Vec<String>,
    no_speech: bool,
    notepad: Vec<String>,
}

fn ms(a: Option<Instant>, b: Option<Instant>) -> Option<f64> {
    Some(b?.saturating_duration_since(a?).as_secs_f64() * 1e3)
}

fn ms_between(a: Option<Instant>, b: Option<Instant>) -> String {
    ms(a, b).map_or_else(|| "-".into(), |v| format!("{v:.1}"))
}

fn secs(samples: &[usize]) -> f64 {
    samples.iter().sum::<usize>() as f64 / f64::from(SAMPLE_RATE)
}

/// Nearest rank, so with five runs p95 is the slowest.
fn percentile(sorted: &[f64], p: f64) -> f64 {
    let rank = (p * sorted.len() as f64).ceil().max(1.0) as usize;
    sorted[rank.min(sorted.len()) - 1]
}

fn print_percentiles(label: &str, mut values: Vec<f64>) {
    if values.is_empty() {
        return;
    }
    values.sort_by(f64::total_cmp);
    println!(
        "{label:<22} p50 {:>6.1} ms   p95 {:>6.1} ms   over {} runs",
        percentile(&values, 0.5),
        percentile(&values, 0.95),
        values.len()
    );
}

pub fn run(paths: &Paths, args: Args) -> Result<ExitCode> {
    let (mut config, _) = setup::load_config(&paths.config_file)?;
    config.vocabulary.extend(args.vocab.iter().cloned());
    if let Some(n) = args.normalizer
        && !same_kind(&n, &config.normalizer)
    {
        config.normalizer = n;
    }
    let _log = setup::init_logging(&paths.logs_dir, "warn,hush_stt::models=info")?;
    let pcm = wav::read_16k_mono(&args.wav)?;
    let clip = Duration::from_secs_f64(pcm.len() as f64 / f64::from(SAMPLE_RATE));
    println!(
        "clip        {} ({:.2} s)",
        args.wav.display(),
        clip.as_secs_f64()
    );
    println!("config      {}", paths.config_file.display());
    println!(
        "session     remote desktop: {}, input desktop unlocked: {}",
        focus::is_remote_session(),
        crate::doctor::input_desktop_unlocked()
    );

    // Loaded before the first run so its seconds are not counted as dictation latency.
    let (model, dir) = engines::resolve_model(&config.engine)?;
    if !model.is_present(&dir) {
        println!("model       downloading {} ...", model.id);
        hush_stt::models::ensure_downloaded(model, &dir)?;
    }
    let (engine, summary) = engines::load_engine(&config.engine, &model.load_path(&dir))?;
    println!(
        "engine      {} on {:?} ({}), load {} ms, warm-up {} ms{}",
        summary.model,
        summary.backend,
        summary.device.as_deref().unwrap_or("?"),
        summary.load.as_millis(),
        summary.warm_up.as_millis(),
        summary
            .fallback
            .as_deref()
            .map(|f| format!(", GPU FALLBACK: {f}"))
            .unwrap_or_default()
    );
    let upgrade = match engines::build_normalizer(&config, &mut |s| println!("normalizer  {s}")) {
        Ok(None) => {
            println!("normalizer  rules only");
            None
        }
        Ok(Some((n, ready))) => {
            println!("normalizer  {}", ready.label);
            if let Some(why) = ready.cpu_fallback {
                println!(
                    "            GPU FALLBACK: the language model runs on the CPU because {why}"
                );
            }
            Some(n)
        }
        Err(e) => {
            println!("normalizer  rules only: {e:#}");
            None
        }
    };

    let (vad, vad_label) = engines::load_vad();
    println!("vad         {vad_label}");
    println!(
        "pipeline    pre_transcribe = {}, {:?}",
        config.pipeline.pre_transcribe, config.pipeline.segmenter
    );

    let (ui, _tray) = UiHandle::start(UiOptions {
        tray: false,
        ..Default::default()
    })?;
    let (tx, rx) = mpsc::channel::<Msg>();
    let focus = WinFocus::new();
    let vocabulary = Vocabulary::from_config(&config, &paths.config_file);
    println!(
        "vocabulary  {} entries{}",
        vocabulary.entries().len(),
        vocabulary
            .file()
            .map(|f| format!(" (with {})", f.display()))
            .unwrap_or_default()
    );
    let app_override = (args.app.is_some() || args.style.is_some()).then(|| AppOverride {
        style: args
            .style
            .unwrap_or_else(|| config.app_policies().lookup(args.app.as_deref()).style),
        exe: args.app.clone(),
    });
    if let Some(o) = &app_override {
        println!(
            "app         normalized as {}, style {:?}{}",
            o.exe.as_deref().unwrap_or("the real target"),
            o.style,
            if style_allows_llm(o.style) {
                ""
            } else {
                " (rules only)"
            }
        );
    }
    let loader: EngineLoader = Box::new(move || Ok((engine, summary)));
    let workers = Workers::spawn(&config, vocabulary, &ui, &focus, loader, &tx)?;
    if let Some(n) = upgrade {
        let _ = workers.norm.send(NormCmd::Upgrade(n));
    }
    let (obs_tx, obs_rx) = mpsc::channel::<Observed>();
    let driver = Driver::new(DriverParts {
        config,
        config_path: paths.config_file.clone(),
        recorder: Box::new(WavRecorder::new(pcm)),
        vad,
        ui: ui.clone(),
        focus,
        hook: None,
        workers,
        rx,
        observer: Some(obs_tx),
        app_override,
    });
    let driver = std::thread::Builder::new()
        .name("hush-driver".into())
        .spawn(move || driver.run())?;

    let target = match args.target {
        Target::Notepad => {
            let n = notepad::open()?;
            println!(
                "target      Notepad window {:#x} on {}",
                n.hwnd,
                n.file.display()
            );
            Some(n)
        }
        Target::Foreground => {
            println!("target      whatever window is in the foreground");
            None
        }
    };

    let mut reports = Vec::new();
    for run in 1..=args.runs {
        let mut r = RunReport::default();
        if let Some(n) = &target {
            r.foreground_ok = focus::refocus_raw(n.hwnd);
            // Let the activation settle before the focus snapshot is taken.
            std::thread::sleep(Duration::from_millis(300));
        } else {
            r.foreground_ok = true;
        }
        while obs_rx.try_recv().is_ok() {}
        let down = Instant::now();
        tx.send(Msg::Hotkey(HotkeyEvent::Down { at: down }))
            .context("driver is gone")?;
        // The key is held while the clip plays, at least a second so the pipeline sees a
        // hold rather than a tap.
        std::thread::sleep(clip.max(Duration::from_secs(1)));
        tx.send(Msg::Hotkey(HotkeyEvent::Up { at: Instant::now() }))
            .context("driver is gone")?;
        let deadline = Instant::now() + RUN_TIMEOUT;
        loop {
            let o = match obs_rx.recv_timeout(deadline.saturating_duration_since(Instant::now())) {
                Ok(o) => o,
                Err(RecvTimeoutError::Timeout) => {
                    r.errors.push(format!("no result within {RUN_TIMEOUT:?}"));
                    break;
                }
                Err(RecvTimeoutError::Disconnected) => bail!("driver stopped"),
            };
            let now = Instant::now();
            match o {
                Observed::Released { at } => r.released = Some(at),
                Observed::SegmentSent {
                    samples,
                    while_recording: true,
                } => r.sent_during_hold.push(samples),
                Observed::SegmentSent { samples, .. } => r.sent_after_release.push(samples),
                Observed::SegmentDone { inference } => {
                    r.inference += inference;
                    if r.released.is_none() {
                        r.done_during_hold += 1;
                    }
                }
                Observed::NoSpeech => {
                    r.no_speech = true;
                    break;
                }
                Observed::Transcript { text } => r.transcript = Some((now, text)),
                Observed::Normalized { text, provenance } => {
                    let empty = text.trim().is_empty();
                    r.normalized = Some((now, text, provenance));
                    if empty {
                        break;
                    }
                }
                Observed::Inserted { outcome } => {
                    r.delivered = outcome.delivered();
                    let label = match outcome.message() {
                        None => format!("{outcome:?}"),
                        Some(m) => format!("{outcome:?}: {m}"),
                    };
                    r.inserted = Some((now, label));
                    break;
                }
                Observed::Failed { stage, error } => {
                    r.errors.push(error);
                    if stage != Stage::Normalizing {
                        r.inserted = Some((now, "failed".into()));
                        break;
                    }
                }
            }
        }
        if let Some(n) = &target {
            // Room for the paste to land before reading back.
            std::thread::sleep(Duration::from_millis(400));
            r.notepad = notepad::text(n.hwnd);
        }
        print_run(run, &r);
        reports.push(r);
        // Past the ≈1 s third-party restore, so the next run starts from a quiet clipboard.
        std::thread::sleep(Duration::from_millis(1300));
    }

    let _ = tx.send(Msg::Quit);
    drop(tx);
    let _ = driver.join();
    ui.shutdown();

    println!(
        "\n run | release->transcript | ->normalized | ->inserted | total ms | engine ms | sent in hold | done in hold | tail s | outcome"
    );
    println!(
        "-----|---------------------|--------------|------------|----------|-----------|--------------|--------------|--------|--------"
    );
    for (i, r) in reports.iter().enumerate() {
        let t = r.transcript.as_ref().map(|t| t.0);
        let n = r.normalized.as_ref().map(|n| n.0);
        let ins = r.inserted.as_ref().map(|x| x.0);
        println!(
            " {:>3} | {:>19} | {:>12} | {:>10} | {:>8} | {:>9.1} | {:>12} | {:>12} | {:>6.2} | {}",
            i + 1,
            ms_between(r.released, t),
            ms_between(t, n),
            ms_between(n, ins),
            ms_between(r.released, ins.or(n).or(t)),
            r.inference.as_secs_f64() * 1e3,
            r.sent_during_hold.len(),
            r.done_during_hold,
            secs(&r.sent_after_release),
            r.inserted
                .as_ref()
                .map(|x| x.1.clone())
                .unwrap_or_else(|| if r.no_speech {
                    "no speech".into()
                } else {
                    "-".into()
                })
        );
    }
    println!();
    print_percentiles(
        "release->transcript",
        reports
            .iter()
            .filter_map(|r| ms(r.released, r.transcript.as_ref().map(|t| t.0)))
            .collect(),
    );
    print_percentiles(
        "transcript->normalized",
        reports
            .iter()
            .filter_map(|r| {
                ms(
                    r.transcript.as_ref().map(|t| t.0),
                    r.normalized.as_ref().map(|n| n.0),
                )
            })
            .collect(),
    );
    print_percentiles(
        "normalized->inserted",
        reports
            .iter()
            .filter_map(|r| {
                ms(
                    r.normalized.as_ref().map(|n| n.0),
                    r.inserted.as_ref().map(|x| x.0),
                )
            })
            .collect(),
    );
    print_percentiles(
        "release->inserted",
        reports
            .iter()
            .filter_map(|r| ms(r.released, r.inserted.as_ref().map(|x| x.0)))
            .collect(),
    );
    let all = reports.iter().all(|r| r.delivered);
    Ok(if all {
        ExitCode::SUCCESS
    } else {
        ExitCode::from(2)
    })
}

fn print_run(run: usize, r: &RunReport) {
    println!("\nrun {run}");
    println!("  target foreground: {}", r.foreground_ok);
    if r.no_speech {
        println!("  no speech detected; nothing inserted");
    }
    let lens = |v: &[usize]| {
        v.iter()
            .map(|n| format!("{:.2}", *n as f64 / f64::from(SAMPLE_RATE)))
            .collect::<Vec<_>>()
            .join(", ")
    };
    println!(
        "  segments:   during hold [{}] s, after release [{}] s",
        lens(&r.sent_during_hold),
        lens(&r.sent_after_release)
    );
    if let Some((_, text)) = &r.transcript {
        println!("  transcript: {text:?}");
    }
    if let Some((_, text, p)) = &r.normalized {
        println!("  normalized: {text:?} ({p:?})");
    }
    if let Some((_, o)) = &r.inserted {
        println!("  outcome:    {o}");
    }
    for e in &r.errors {
        println!("  error:      {e}");
    }
    // Windows 11 Notepad keeps other tabs' editors as children too; show ours only.
    let wanted = r.normalized.as_ref().map(|n| n.1.as_str()).unwrap_or("");
    if r.notepad.is_empty() || wanted.is_empty() {
        return;
    }
    match r.notepad.iter().find(|t| t.contains(wanted)) {
        Some(t) => println!("  notepad:    {t:?}"),
        None => println!(
            "  notepad:    the text is NOT in any of its {} editors",
            r.notepad.len()
        ),
    }
}
