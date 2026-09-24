//! Drives the real driver, workers and pipeline rather than the engines alone, so the
//! timing covers the channel hops and the insertion chain the user waits on.

use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::mpsc::{self, RecvTimeoutError};
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use wl_core::normalize::Provenance;
use wl_core::pipeline::Stage;
use wl_core::stt::SAMPLE_RATE;
use wl_platform_windows::focus::{self, WinFocus};
use wl_platform_windows::hook::HotkeyEvent;
use wl_platform_windows::ui_thread::{UiHandle, UiOptions};

use crate::driver::{Driver, DriverParts, Msg, Observed, Workers};
use crate::engines;
use crate::notepad;
use crate::setup::{self, Paths};
use crate::wav::{self, WavRecorder};
use crate::workers::{EngineLoader, NormCmd};

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
}

impl Args {
    pub fn parse(args: &[String]) -> Result<Self> {
        let mut it = args.iter();
        let mut wav = None;
        let mut target = Target::Notepad;
        let mut runs = 1;
        while let Some(a) = it.next() {
            match a.as_str() {
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
        })
    }
}

#[derive(Default)]
struct RunReport {
    foreground_ok: bool,
    released: Option<Instant>,
    transcript: Option<(Instant, String, Duration)>,
    normalized: Option<(Instant, String, Provenance)>,
    inserted: Option<(Instant, String)>,
    delivered: bool,
    errors: Vec<String>,
    no_speech: bool,
    notepad: Vec<String>,
}

fn ms_between(a: Option<Instant>, b: Option<Instant>) -> String {
    match (a, b) {
        (Some(a), Some(b)) => format!("{:.1}", b.saturating_duration_since(a).as_secs_f64() * 1e3),
        _ => "-".into(),
    }
}

pub fn run(paths: &Paths, args: Args) -> Result<ExitCode> {
    let (config, _) = setup::load_config(&paths.config_file)?;
    let _log = setup::init_logging(&paths.logs_dir, "warn,wl_stt::models=info")?;
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
        wl_stt::models::ensure_downloaded(model, &dir)?;
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
    let upgrade = match engines::http_config(&config.normalizer) {
        None => {
            println!("normalizer  rules only");
            None
        }
        Some(http) => {
            let label = format!("{} via {}", http.model, http.base_url);
            match engines::build_http_normalizer(http) {
                Ok((n, warm)) => {
                    println!("normalizer  {label}, warm-up {} ms", warm.as_millis());
                    Some(n)
                }
                Err(e) => {
                    println!("normalizer  rules only: {e:#}");
                    None
                }
            }
        }
    };

    let (ui, _tray) = UiHandle::start(UiOptions {
        tray: false,
        ..Default::default()
    })?;
    let (tx, rx) = mpsc::channel::<Msg>();
    let focus = WinFocus::new();
    let loader: EngineLoader = Box::new(move || Ok((engine, summary)));
    let workers = Workers::spawn(&config, &ui, &focus, loader, &tx)?;
    if let Some(n) = upgrade {
        let _ = workers.norm.send(NormCmd::Upgrade(n));
    }
    let (obs_tx, obs_rx) = mpsc::channel::<Observed>();
    let driver = Driver::new(DriverParts {
        config,
        config_path: paths.config_file.clone(),
        recorder: Box::new(WavRecorder::new(pcm)),
        ui: ui.clone(),
        focus,
        hook: None,
        workers,
        rx,
        observer: Some(obs_tx),
    });
    let driver = std::thread::Builder::new()
        .name("wl-driver".into())
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
        // At least a second, so the pipeline sees a hold rather than a tap.
        let up = down + clip.max(Duration::from_secs(1));
        tx.send(Msg::Hotkey(HotkeyEvent::Down { at: down }))
            .context("driver is gone")?;
        tx.send(Msg::Hotkey(HotkeyEvent::Up { at: up }))
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
                Observed::Released => r.released = Some(now),
                Observed::NoSpeech => {
                    r.no_speech = true;
                    break;
                }
                Observed::Transcript { text, inference } => {
                    let empty = text.trim().is_empty();
                    r.transcript = Some((now, text, inference));
                    if empty {
                        break;
                    }
                }
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
        "\n run | release->transcript | ->normalized | ->inserted | total ms | engine ms | outcome"
    );
    println!(
        "-----|---------------------|--------------|------------|----------|-----------|--------"
    );
    for (i, r) in reports.iter().enumerate() {
        let t = r.transcript.as_ref().map(|t| t.0);
        let n = r.normalized.as_ref().map(|n| n.0);
        let ins = r.inserted.as_ref().map(|x| x.0);
        println!(
            " {:>3} | {:>19} | {:>12} | {:>10} | {:>8} | {:>9} | {}",
            i + 1,
            ms_between(r.released, t),
            ms_between(t, n),
            ms_between(n, ins),
            ms_between(r.released, ins.or(n).or(t)),
            r.transcript
                .as_ref()
                .map(|t| format!("{:.1}", t.2.as_secs_f64() * 1e3))
                .unwrap_or_else(|| "-".into()),
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
    if let Some((_, text, _)) = &r.transcript {
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
