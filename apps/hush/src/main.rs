#[cfg(windows)]
mod doctor;
#[cfg(windows)]
mod driver;
#[cfg(windows)]
mod engines;
#[cfg(windows)]
mod notepad;
#[cfg(windows)]
mod setup;
#[cfg(windows)]
mod simulate;
#[cfg(windows)]
mod wav;
#[cfg(windows)]
mod workers;

const USAGE: &str = "\
usage: hush [--config <path>]                 run the dictation app (tray + hotkey)
       hush [--config <path>] doctor          check devices, model, engine, normalizer
       hush [--config <path>] simulate <wav> [--target notepad|foreground] [--runs N]
                                   [--app <exe>] [--style formal|casual|code|none]
                                   [--vocab <word,...>]
                                                       run one dictation from a WAV, end to end;
                                                       --app/--style normalize as if for that
                                                       app, --vocab adds dictionary entries

The default config is %APPDATA%\\hush\\config.toml, created on first run.
Logs go to stderr and %LOCALAPPDATA%\\hush\\logs\\. RUST_LOG overrides the level.";

#[cfg(windows)]
fn main() -> std::process::ExitCode {
    use std::process::ExitCode;

    match run_cli() {
        Ok(code) => code,
        Err(e) => {
            eprintln!("hush: {e:#}");
            ExitCode::FAILURE
        }
    }
}

#[cfg(windows)]
fn run_cli() -> anyhow::Result<std::process::ExitCode> {
    use anyhow::{Context, bail};

    let mut args = std::env::args().skip(1).peekable();
    let mut config_path = None;
    let mut rest = Vec::new();
    while let Some(a) = args.next() {
        match a.as_str() {
            "--config" => {
                config_path = Some(std::path::PathBuf::from(
                    args.next().context("--config needs a path")?,
                ))
            }
            "-h" | "--help" => {
                println!("{USAGE}");
                return Ok(std::process::ExitCode::SUCCESS);
            }
            _ => rest.push(a),
        }
    }
    let paths = setup::Paths::resolve(config_path)?;
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
