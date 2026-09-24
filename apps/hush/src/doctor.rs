use std::process::ExitCode;
use std::time::Instant;

use anyhow::Result;
use hush_core::stt::{DecodeOptions, SAMPLE_RATE};
use hush_platform_windows::focus;
use hush_platform_windows::hook::HotkeyConfig;
use hush_platform_windows::ui_thread::{self, INSTANCE_MUTEX, UiError};
use windows::Win32::System::StationsAndDesktops::{
    CloseDesktop, DESKTOP_CONTROL_FLAGS, DESKTOP_SWITCHDESKTOP, OpenInputDesktop, SwitchDesktop,
};

use crate::engines;
use crate::setup::{self, Paths};

fn row(label: &str, value: impl std::fmt::Display) {
    println!("{label:<14}{value}");
}

/// Without an unlocked input desktop `SendInput` is refused and no hotkey reaches the
/// hook.
pub fn input_desktop_unlocked() -> bool {
    // SAFETY: plain FFI; the handle is closed below. SwitchDesktop to the current input
    // desktop is a no-op that fails exactly when the workstation is locked.
    unsafe {
        let Ok(desk) = OpenInputDesktop(DESKTOP_CONTROL_FLAGS(0), false, DESKTOP_SWITCHDESKTOP)
        else {
            return false;
        };
        let ok = SwitchDesktop(desk).is_ok();
        let _ = CloseDesktop(desk);
        ok
    }
}

pub fn run(paths: &Paths) -> Result<ExitCode> {
    let (config, created) = setup::load_config(&paths.config_file)?;
    let _log = setup::init_logging(&paths.logs_dir, "warn,hush_stt::models=info")?;
    println!("hush doctor\n");
    row(
        "config",
        format!(
            "{}{}",
            paths.config_file.display(),
            if created { " (created now)" } else { "" }
        ),
    );
    row(
        "hotkey",
        match HotkeyConfig::parse(&config.hotkey) {
            Ok(_) => format!("{} (ok)", config.hotkey),
            Err(e) => format!("{} (INVALID: {e})", config.hotkey),
        },
    );
    row("logs", paths.logs_dir.display());
    row(
        "session",
        format!(
            "remote desktop: {}, elevated: {}, input desktop: {}",
            yes(focus::is_remote_session()),
            yes(focus::self_elevated()),
            if input_desktop_unlocked() {
                "unlocked"
            } else {
                "LOCKED or disconnected (no input can be injected)"
            }
        ),
    );
    row(
        "instance",
        match ui_thread::acquire_single_instance(INSTANCE_MUTEX) {
            Ok(_guard) => "no other instance running".to_string(),
            Err(UiError::AlreadyRunning) => "hush is running in this session".into(),
            Err(e) => format!("check failed: {e}"),
        },
    );

    println!("\ninput devices");
    match hush_audio::list_input_devices() {
        Ok(devs) if devs.is_empty() => println!("  none (dictation needs a microphone)"),
        Ok(devs) => {
            for d in devs {
                println!(
                    "  {} {}  {}",
                    if d.is_default { "*" } else { " " },
                    d.name,
                    d.default_format.as_deref().unwrap_or("?")
                );
            }
        }
        Err(e) => println!("  cannot list: {e}"),
    }

    println!("\nvulkan devices");
    let devices = engines::describe_vulkan_devices(config.engine.gpu_device);
    for d in &devices {
        println!("  {d}");
    }
    if devices.is_empty() {
        println!("  none");
    }
    match config.engine.gpu_device {
        Some(i) if i >= devices.len() => {
            println!("  engine.gpu_device = {i} does not exist")
        }
        Some(_) => {}
        None if devices.len() > 1 => println!(
            "  engine.gpu_device is unset: the runtime picks, which may be an integrated GPU \
             or the card running your LLM"
        ),
        None => {}
    }

    println!();
    let (model, dir) = match engines::resolve_model(&config.engine) {
        Ok(m) => m,
        Err(e) => {
            row("model", format!("UNUSABLE: {e:#}"));
            return Ok(ExitCode::FAILURE);
        }
    };
    if model.is_present(&dir) {
        row(
            "model",
            format!("{} present in {}", model.id, dir.display()),
        );
    } else {
        row(
            "model",
            format!(
                "{} missing; downloading {:.2} GiB to {} (progress on stderr)",
                model.id,
                model.total_size() as f64 / (1u64 << 30) as f64,
                dir.display()
            ),
        );
        if let Err(e) = hush_stt::models::ensure_downloaded(model, &dir) {
            row("", format!("DOWNLOAD FAILED: {e}"));
            return Ok(ExitCode::FAILURE);
        }
        row("", "downloaded and verified");
    }

    match engines::vad_model() {
        Ok((vad, vad_dir)) if !vad.is_present(&vad_dir) => {
            row(
                "vad model",
                format!(
                    "{} missing; downloading {} KiB to {}",
                    vad.id,
                    vad.total_size() / 1024,
                    vad_dir.display()
                ),
            );
            match hush_stt::models::ensure_downloaded(vad, &vad_dir) {
                Ok(()) => row("", "downloaded and verified"),
                Err(e) => row("", format!("DOWNLOAD FAILED: {e}")),
            }
        }
        Ok((vad, vad_dir)) => row(
            "vad model",
            format!("{} present in {}", vad.id, vad_dir.display()),
        ),
        Err(e) => row("vad model", format!("UNUSABLE: {e:#}")),
    }
    let (_, vad_label) = engines::load_vad();
    row("vad", vad_label);
    let seg = &config.pipeline.segmenter;
    row(
        "pipeline",
        format!(
            "pre_transcribe = {}; segmenter min_speech {} ms, min_pause {} ms, pad {} ms, \
             max_segment {} ms",
            config.pipeline.pre_transcribe,
            seg.min_speech.as_millis(),
            seg.min_pause.as_millis(),
            seg.pad.as_millis(),
            seg.max_segment.as_millis()
        ),
    );

    let started = Instant::now();
    let engine_ok = match engines::load_engine(&config.engine, &model.load_path(&dir)) {
        Ok((mut engine, s)) => {
            row(
                "engine",
                format!(
                    "{} on {:?} ({}); load {} ms, warm-up {} ms; policy {:?}",
                    s.model,
                    s.backend,
                    s.device.as_deref().unwrap_or("device not reported"),
                    s.load.as_millis(),
                    s.warm_up.as_millis(),
                    config.engine.gpu
                ),
            );
            if let Some(why) = &s.fallback {
                row(
                    "",
                    format!("GPU FALLBACK: speech runs on the CPU because {why}"),
                );
            }
            let silence = vec![0.0f32; SAMPLE_RATE as usize];
            let t = Instant::now();
            match engine.transcribe(&silence, &DecodeOptions::default()) {
                Ok(tr) => row(
                    "",
                    format!(
                        "1 s of silence -> {:?} in {:.0} ms (the app drops silence before the engine)",
                        tr.text,
                        t.elapsed().as_secs_f64() * 1e3
                    ),
                ),
                Err(e) => row("", format!("test transcription failed: {e}")),
            }
            true
        }
        Err(e) => {
            row(
                "engine",
                format!("FAILED after {} ms: {e:#}", started.elapsed().as_millis()),
            );
            false
        }
    };

    match engines::http_config(&config.normalizer) {
        None => row("normalizer", "rules only (normalizer.kind = \"rules\")"),
        Some(http) => match engines::probe_http(&http) {
            Ok(p) => {
                row(
                    "normalizer",
                    format!(
                        "{} answered in {} ms; model {} {}",
                        http.base_url,
                        p.elapsed.as_millis(),
                        http.model,
                        if p.has_model {
                            "is served"
                        } else {
                            "is NOT served"
                        }
                    ),
                );
                if !p.has_model {
                    row("", format!("served: {}", p.models.join(", ")));
                }
            }
            Err(e) => row("normalizer", format!("{e:#}; the app will run rules-only")),
        },
    }

    println!();
    if engine_ok {
        println!("ok");
        Ok(ExitCode::SUCCESS)
    } else {
        println!("the speech engine cannot load; dictation will not work");
        Ok(ExitCode::FAILURE)
    }
}

fn yes(b: bool) -> &'static str {
    if b { "yes" } else { "no" }
}
