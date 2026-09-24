use std::os::windows::io::IntoRawHandle;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};

use anyhow::{Context, Result};
use hush_core::config::{Config, DEFAULT_CONFIG};
use hush_platform_windows::ui_thread::AppPaths;
use tracing_appender::non_blocking::WorkerGuard;
use tracing_appender::rolling::{RollingFileAppender, Rotation};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Layer};
use windows::Win32::Foundation::HANDLE;
use windows::Win32::System::Console::{
    ATTACH_PARENT_PROCESS, AttachConsole, GetStdHandle, STD_ERROR_HANDLE, STD_HANDLE,
    STD_OUTPUT_HANDLE, SetStdHandle,
};

const LOG_FILES_KEPT: usize = 7;

#[derive(Debug, Clone)]
pub struct Paths {
    pub config_file: PathBuf,
    pub logs_dir: PathBuf,
}

impl Paths {
    pub fn resolve(config_override: Option<PathBuf>) -> Result<Self> {
        let app = AppPaths::resolve().context("cannot resolve %APPDATA% / %LOCALAPPDATA%")?;
        Ok(Self {
            config_file: config_override.unwrap_or(app.config_file),
            logs_dir: app.data_dir.join("logs"),
        })
    }
}

/// Where stdout and stderr go. hush is a Windows-subsystem program, so it starts with no
/// console at all.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Output {
    /// The terminal hush was started from.
    Console,
    /// Handles the parent passed in, such as a pipe or a file.
    Inherited,
    /// No terminal to attach to; what would have been printed lands in this file.
    File(PathBuf),
    Nowhere,
}

impl Output {
    fn has_stderr(&self) -> bool {
        matches!(self, Output::Console | Output::Inherited)
    }
}

static STDERR_LOGGING: AtomicBool = AtomicBool::new(false);

fn std_handle_set(which: STD_HANDLE) -> bool {
    // SAFETY: plain FFI query.
    unsafe { GetStdHandle(which) }.is_ok_and(|h| !h.is_invalid() && !h.0.is_null())
}

/// `attach` is for the subcommands, which print; the app itself only logs.
pub fn connect_output(attach: bool) -> Output {
    let out = if std_handle_set(STD_OUTPUT_HANDLE) {
        Output::Inherited
    } else if !attach {
        Output::Nowhere
    } else if attach_parent_console() {
        Output::Console
    } else {
        redirect_to_file().map_or(Output::Nowhere, Output::File)
    };
    STDERR_LOGGING.store(out.has_stderr(), Ordering::Relaxed);
    out
}

/// Fails when the parent has no console, as when Explorer or a shortcut started hush.
fn attach_parent_console() -> bool {
    // SAFETY: plain FFI call.
    unsafe { AttachConsole(ATTACH_PARENT_PROCESS) }.is_ok()
}

fn redirect_to_file() -> Option<PathBuf> {
    let logs = Paths::resolve(None).ok()?.logs_dir;
    std::fs::create_dir_all(&logs).ok()?;
    let path = logs.join("console.log");
    let file = std::fs::File::create(&path).ok()?;
    let handle = HANDLE(file.into_raw_handle());
    // SAFETY: the handle was just released by its `File` and stays open for the life of
    // the process, which is what a standard handle needs; std looks the handle up on
    // every write, so later prints follow it.
    unsafe {
        SetStdHandle(STD_OUTPUT_HANDLE, handle).ok()?;
        SetStdHandle(STD_ERROR_HANDLE, handle).ok()?;
    }
    Some(path)
}

/// The bool is true when the default config was just written.
pub fn load_config(path: &Path) -> Result<(Config, bool)> {
    let created = !path.exists();
    if created {
        if let Some(dir) = path.parent() {
            std::fs::create_dir_all(dir).with_context(|| format!("creating {}", dir.display()))?;
        }
        std::fs::write(path, DEFAULT_CONFIG)
            .with_context(|| format!("writing {}", path.display()))?;
    }
    let text =
        std::fs::read_to_string(path).with_context(|| format!("reading {}", path.display()))?;
    let config = Config::from_toml(&text).with_context(|| format!("in {}", path.display()))?;
    Ok((config, created))
}

/// `console_default` applies to stderr only when `RUST_LOG` is unset. The process must
/// not exit before the guard is dropped, or the last lines are lost.
pub fn init_logging(logs_dir: &Path, console_default: &str) -> Result<WorkerGuard> {
    std::fs::create_dir_all(logs_dir)
        .with_context(|| format!("creating {}", logs_dir.display()))?;
    let appender = RollingFileAppender::builder()
        .rotation(Rotation::DAILY)
        .filename_prefix("hush")
        .filename_suffix("log")
        .max_log_files(LOG_FILES_KEPT)
        .build(logs_dir)
        .context("opening the log file")?;
    // Non-blocking: a log line on the release-to-text path must not wait on the disk.
    let (file_writer, guard) = tracing_appender::non_blocking(appender);
    let env = std::env::var("RUST_LOG").ok();
    let filter = |default: &str| {
        env.as_deref()
            .and_then(|e| EnvFilter::try_new(e).ok())
            .unwrap_or_else(|| EnvFilter::new(default))
    };
    let stderr = STDERR_LOGGING.load(Ordering::Relaxed).then(|| {
        tracing_subscriber::fmt::layer()
            .with_writer(std::io::stderr)
            .with_filter(filter(console_default))
    });
    tracing_subscriber::registry()
        .with(stderr)
        .with(
            tracing_subscriber::fmt::layer()
                .with_ansi(false)
                .with_writer(file_writer)
                .with_filter(filter("info")),
        )
        .try_init()
        .context("installing the log subscriber")?;
    Ok(guard)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_run_writes_the_default_config_and_reads_it_back() {
        let dir = std::env::temp_dir().join(format!("hush-setup-test-{}", std::process::id()));
        let path = dir.join("sub").join("config.toml");
        let _ = std::fs::remove_dir_all(&dir);
        let (c, created) = load_config(&path).unwrap();
        assert!(created);
        assert_eq!(c, Config::default());
        let (_, created) = load_config(&path).unwrap();
        assert!(!created);
        std::fs::write(&path, "hot_key = 1").unwrap();
        assert!(load_config(&path).is_err(), "a typo must not be ignored");
        let _ = std::fs::remove_dir_all(&dir);
    }
}
