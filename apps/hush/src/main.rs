// No console window for the tray app; the subcommands attach to the terminal that
// started them instead. Tests keep the console so the harness can print.
#![cfg_attr(not(test), windows_subsystem = "windows")]

#[cfg(windows)]
mod doctor;
#[cfg(windows)]
mod driver;
#[cfg(windows)]
mod engines;
#[cfg(windows)]
mod notepad;
#[cfg(windows)]
mod provision;
#[cfg(windows)]
mod setup;
#[cfg(windows)]
mod simulate;
#[cfg(windows)]
mod startup;
#[cfg(windows)]
mod wav;
#[cfg(windows)]
mod workers;

const USAGE: &str = "\
usage: hush [--config <path>]                 run the dictation app (tray + hotkey)
       hush [--config <path>] doctor          check devices, model, engine, normalizer
       hush [--config <path>] simulate <wav> [--target notepad|foreground] [--runs N]
                                   [--app <exe>] [--style formal|casual|code|none]
                                   [--vocab <word,...>] [--normalizer rules|llama-cpp|http]
                                                       run one dictation from a WAV, end to end;
                                                       --app/--style normalize as if for that
                                                       app, --vocab adds dictionary entries,
                                                       --normalizer overrides the config's kind

The default config is %APPDATA%\\hush\\config.toml, created on first run.
Models go to %LOCALAPPDATA%\\hush\\models, or to HUSH_MODELS_DIR when it is set.
Logs go to %LOCALAPPDATA%\\hush\\logs\\. RUST_LOG overrides the level.";

#[cfg(windows)]
struct Cli {
    config_path: Option<std::path::PathBuf>,
    rest: Vec<String>,
    help: bool,
}

#[cfg(windows)]
fn parse_args(args: Vec<String>) -> anyhow::Result<Cli> {
    use anyhow::Context;

    let mut cli = Cli {
        config_path: None,
        rest: Vec::new(),
        help: false,
    };
    let mut args = args.into_iter();
    while let Some(a) = args.next() {
        match a.as_str() {
            "--config" => {
                cli.config_path = Some(args.next().context("--config needs a path")?.into())
            }
            "-h" | "--help" => cli.help = true,
            _ => cli.rest.push(a),
        }
    }
    Ok(cli)
}

#[cfg(windows)]
fn main() -> std::process::ExitCode {
    use std::process::ExitCode;

    let args: Vec<String> = std::env::args().skip(1).collect();
    let cli = parse_args(args);
    // Anything but the bare app prints, so it needs the terminal.
    let prints = cli.as_ref().map_or(true, |c| c.help || !c.rest.is_empty());
    let output = setup::connect_output(prints);
    match cli.and_then(run_cli) {
        Ok(code) => code,
        Err(e) => {
            let message = format!("hush: {e:#}");
            eprintln!("{message}");
            if !prints && output == setup::Output::Nowhere {
                error_box(&message);
            }
            ExitCode::FAILURE
        }
    }
}

/// The tray app has no terminal, and an error that stops it must still be seen.
#[cfg(windows)]
fn error_box(message: &str) {
    use windows::Win32::UI::WindowsAndMessaging::{
        MB_ICONERROR, MB_OK, MB_SETFOREGROUND, MessageBoxW,
    };
    use windows::core::PCWSTR;

    let text: Vec<u16> = message.encode_utf16().chain([0]).collect();
    let title: Vec<u16> = "hush".encode_utf16().chain([0]).collect();
    // SAFETY: both buffers are NUL-terminated and outlive the modal call.
    unsafe {
        MessageBoxW(
            None,
            PCWSTR(text.as_ptr()),
            PCWSTR(title.as_ptr()),
            MB_OK | MB_ICONERROR | MB_SETFOREGROUND,
        );
    }
}

#[cfg(windows)]
fn run_cli(cli: Cli) -> anyhow::Result<std::process::ExitCode> {
    use anyhow::bail;

    if cli.help {
        println!("{USAGE}");
        return Ok(std::process::ExitCode::SUCCESS);
    }
    let paths = setup::Paths::resolve(cli.config_path)?;
    let rest = cli.rest;
    match rest.first().map(String::as_str) {
        None => driver::run_app(&paths),
        Some("doctor") => {
            if rest.len() > 1 {
                bail!("doctor takes no arguments\n{USAGE}");
            }
            doctor::run(&paths)
        }
        Some("simulate") => simulate::run(&paths, simulate::Args::parse(&rest[1..])?),
        Some(other) => bail!("unknown command `{other}`\n{USAGE}"),
    }
}

#[cfg(not(windows))]
fn main() {
    eprintln!("hush runs on Windows only.\n{USAGE}");
    std::process::exit(1);
}
