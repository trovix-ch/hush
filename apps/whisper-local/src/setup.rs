//! Paths, the config file and logging.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use tracing_appender::non_blocking::WorkerGuard;
use tracing_appender::rolling::{RollingFileAppender, Rotation};
use tracing_subscriber::layer::SubscriberExt;
use tracing_subscriber::util::SubscriberInitExt;
use tracing_subscriber::{EnvFilter, Layer};
use wl_core::config::{Config, DEFAULT_CONFIG};
use wl_platform_windows::ui_thread::AppPaths;

/// Log files kept; one per day.
const LOG_FILES_KEPT: usize = 7;

#[derive(Debug, Clone)]
pub struct Paths {
    pub config_file: PathBuf,
    pub logs_dir: PathBuf,
}

impl Paths {
    /// The standard locations, with the config file optionally replaced.
    pub fn resolve(config_override: Option<PathBuf>) -> Result<Self> {
        let app = AppPaths::resolve().context("cannot resolve %APPDATA% / %LOCALAPPDATA%")?;
        Ok(Self {
            config_file: config_override.unwrap_or(app.config_file),
            logs_dir: app.data_dir.join("logs"),
        })
    }
}

/// Reads the config, writing the commented default first if there is none. Returns
/// whether it was created.
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

/// Logs to stderr and to a daily file. `console_default` is the stderr level when
/// `RUST_LOG` is unset: the subcommands print their own report and keep stderr quiet,
/// while the file always gets `info`.
///
/// The returned guard flushes the file writer on drop; the process must not exit
/// before it is dropped or the last lines are lost.
pub fn init_logging(logs_dir: &Path, console_default: &str) -> Result<WorkerGuard> {
    std::fs::create_dir_all(logs_dir)
        .with_context(|| format!("creating {}", logs_dir.display()))?;
    let appender = RollingFileAppender::builder()
        .rotation(Rotation::DAILY)
        .filename_prefix("whisper-local")
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
    tracing_subscriber::registry()
        .with(
            tracing_subscriber::fmt::layer()
                .with_writer(std::io::stderr)
                .with_filter(filter(console_default)),
        )
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
        let dir = std::env::temp_dir().join(format!("wl-setup-test-{}", std::process::id()));
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
