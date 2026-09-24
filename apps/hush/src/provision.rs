//! Missing models are fetched one at a time on a background thread, speech first, and each
//! is handed over the moment it lands: dictation works rules-only as soon as speech can
//! load, long before the language model arrives. One at a time, because two parallel
//! downloads would only split the bandwidth and delay speech.

use std::path::PathBuf;
use std::sync::mpsc::Receiver;
use std::time::{Duration, Instant};

use hush_core::config::{Config, NormalizerChoice};
use hush_core::notify::OverlayState;
use hush_stt::models::{self, ModelError, ModelManifest, Progress};

use crate::engines;

/// Declared in fetch order.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Role {
    Speech,
    Vad,
    Language,
}

impl Role {
    pub fn label(self) -> &'static str {
        match self {
            Role::Speech => "speech model",
            Role::Vad => "voice detection model",
            Role::Language => "language model",
        }
    }
}

#[derive(Debug, Clone)]
pub struct Needed {
    pub role: Role,
    pub model: &'static ModelManifest,
    pub dir: PathBuf,
}

/// Every model this config uses that is not on disk yet, in fetch order.
pub fn missing(config: &Config) -> anyhow::Result<Vec<Needed>> {
    let mut out = Vec::new();
    let mut need = |role, (model, dir): (&'static ModelManifest, PathBuf)| {
        if !model.is_present(&dir) {
            out.push(Needed { role, model, dir });
        }
    };
    need(Role::Speech, engines::resolve_model(&config.engine)?);
    match engines::vad_model() {
        Ok(m) => need(Role::Vad, m),
        Err(e) => tracing::warn!(error = %e, "no voice detection model to fetch"),
    }
    if cfg!(feature = "llama-cpp")
        && let NormalizerChoice::LlamaCpp { model, .. } = &config.normalizer
    {
        // A bad id is reported when the normalizer loads; it must not stop speech.
        if let Ok(m) = engines::llm_model(model) {
            need(Role::Language, m);
        }
    }
    Ok(in_fetch_order(out))
}

pub fn in_fetch_order(mut needed: Vec<Needed>) -> Vec<Needed> {
    needed.sort_by_key(|n| n.role);
    needed
}

#[derive(Debug, Clone, PartialEq)]
pub enum DownloadState {
    Progress(Progress),
    Done,
    Failed(String),
}

#[derive(Debug, Clone, PartialEq)]
pub struct DownloadEvent {
    pub role: Role,
    pub state: DownloadState,
}

const MIB: u64 = 1 << 20;

/// Rounded down, so 100 % means every byte is there.
fn percent(p: &Progress) -> u64 {
    (p.done.min(p.total) * 100)
        .checked_div(p.total)
        .unwrap_or(100)
}

fn capitalized(s: &str) -> String {
    let mut c = s.chars();
    c.next()
        .map(|f| f.to_uppercase().chain(c).collect())
        .unwrap_or_default()
}

impl DownloadEvent {
    pub fn overlay(&self) -> OverlayState {
        let label = self.role.label();
        match &self.state {
            DownloadState::Progress(p) => OverlayState::Progress {
                message: format!(
                    "Downloading {label} {} % · {} of {} MB",
                    percent(p),
                    p.done / MIB,
                    p.total.div_ceil(MIB)
                ),
                fraction: if p.total == 0 {
                    1.0
                } else {
                    (p.done as f64 / p.total as f64).clamp(0.0, 1.0) as f32
                },
            },
            DownloadState::Done => OverlayState::Idle,
            DownloadState::Failed(why) => OverlayState::Alert {
                message: format!("{} download failed: {why}", capitalized(label)),
            },
        }
    }

    /// For the tray tooltip, which has room for little.
    pub fn short(&self) -> String {
        let label = self.role.label();
        match &self.state {
            DownloadState::Progress(p) => format!("downloading {label} {} %", percent(p)),
            DownloadState::Done => format!("{label} downloaded"),
            DownloadState::Failed(_) => format!("{label} download failed, retry from this menu"),
        }
    }
}

/// The URL is in the log; the pill has room for the reason only.
pub fn short_reason(e: &ModelError) -> String {
    match e {
        ModelError::Http { message, .. } => {
            message.strip_prefix("io: ").unwrap_or(message).to_string()
        }
        other => other.to_string(),
    }
}

const PROGRESS_EVERY: Duration = Duration::from_millis(500);

/// A progress report per whole percent or half second, whichever comes first; the
/// downloader calls back for every megabyte.
#[derive(Default)]
struct Throttle {
    last: Option<(u64, Instant)>,
}

impl Throttle {
    fn due(&mut self, p: &Progress, now: Instant) -> bool {
        let pct = percent(p);
        let due = match self.last {
            None => true,
            Some((last_pct, at)) => {
                pct != last_pct || now.saturating_duration_since(at) >= PROGRESS_EVERY
            }
        };
        if due {
            self.last = Some((pct, now));
        }
        due
    }
}

pub type Fetch<'a> = dyn FnMut(&Needed, &mut dyn FnMut(Progress)) -> Result<(), String> + 'a;

/// Blocks until every model has arrived, or until `retry` closes while a failure waits
/// for the user. `arrived` runs on this thread right after a model's `Done` is reported.
pub fn run(
    needed: Vec<Needed>,
    fetch: &mut Fetch<'_>,
    report: &mut dyn FnMut(DownloadEvent),
    arrived: &mut dyn FnMut(Role),
    retry: &Receiver<()>,
) {
    for n in in_fetch_order(needed) {
        loop {
            let mut throttle = Throttle::default();
            let result = fetch(&n, &mut |p| {
                if throttle.due(&p, Instant::now()) {
                    report(DownloadEvent {
                        role: n.role,
                        state: DownloadState::Progress(p),
                    });
                }
            });
            match result {
                Ok(()) => {
                    report(DownloadEvent {
                        role: n.role,
                        state: DownloadState::Done,
                    });
                    arrived(n.role);
                    break;
                }
                Err(why) => {
                    tracing::error!(model = %n.model.id, error = %why, "model download failed");
                    report(DownloadEvent {
                        role: n.role,
                        state: DownloadState::Failed(why),
                    });
                    if retry.recv().is_err() {
                        return;
                    }
                    tracing::info!(model = %n.model.id, "retrying the download");
                }
            }
        }
    }
}

/// The real fetcher: the resumable downloader, with the reason shortened for the pill.
pub fn download(n: &Needed, progress: &mut dyn FnMut(Progress)) -> Result<(), String> {
    tracing::info!(model = %n.model.id, bytes = n.model.total_size(), dir = %n.dir.display(), "first run: fetching");
    models::ensure_downloaded_with(n.model, &n.dir, progress).map_err(|e| {
        tracing::warn!(model = %n.model.id, error = %e, "download error");
        short_reason(&e)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::mpsc;

    fn needed(role: Role) -> Needed {
        let id = match role {
            Role::Speech => models::DEFAULT_MODEL_ID,
            Role::Vad => models::VAD_MODEL_ID,
            Role::Language => models::DEFAULT_LLM_ID,
        };
        Needed {
            role,
            model: models::find(id).unwrap(),
            dir: PathBuf::from("unused"),
        }
    }

    #[test]
    fn download_states_map_to_overlay_states() {
        let ev = |state| DownloadEvent {
            role: Role::Speech,
            state,
        };
        let progress = ev(DownloadState::Progress(Progress {
            done: 530 * MIB,
            total: 1254 * MIB + 5,
        }));
        match progress.overlay() {
            OverlayState::Progress { message, fraction } => {
                assert_eq!(message, "Downloading speech model 42 % · 530 of 1255 MB");
                assert!((fraction - 0.4226).abs() < 0.001, "{fraction}");
            }
            other => panic!("{other:?}"),
        }
        assert_eq!(progress.short(), "downloading speech model 42 %");

        let almost = ev(DownloadState::Progress(Progress {
            done: 999,
            total: 1000,
        }));
        assert!(matches!(
            almost.overlay(),
            OverlayState::Progress { message, .. } if message.contains(" 99 %")
        ));
        let empty = ev(DownloadState::Progress(Progress { done: 0, total: 0 }));
        assert!(matches!(
            empty.overlay(),
            OverlayState::Progress { fraction, .. } if fraction == 1.0
        ));

        assert_eq!(ev(DownloadState::Done).overlay(), OverlayState::Idle);
        let failed = DownloadEvent {
            role: Role::Language,
            state: DownloadState::Failed("connection reset".into()),
        };
        assert_eq!(
            failed.overlay(),
            OverlayState::Alert {
                message: "Language model download failed: connection reset".into()
            }
        );
        assert!(failed.short().contains("retry"), "{}", failed.short());
    }

    #[test]
    fn http_failures_drop_the_url_from_the_reason() {
        let e = ModelError::Http {
            url: "https://example.invalid/a/very/long/path.gguf".into(),
            message: "io: connection refused".into(),
        };
        assert_eq!(short_reason(&e), "connection refused");
    }

    #[test]
    fn progress_is_reported_per_percent_or_half_second() {
        let mut t = Throttle::default();
        let t0 = Instant::now();
        let p = |done| Progress { done, total: 1000 };
        assert!(t.due(&p(0), t0));
        assert!(!t.due(&p(5), t0));
        assert!(t.due(&p(10), t0));
        assert!(!t.due(&p(15), t0 + Duration::from_millis(100)));
        assert!(t.due(&p(15), t0 + Duration::from_millis(600)));
    }

    #[derive(Debug, PartialEq)]
    enum Seen {
        Fetch(Role),
        Report(Role, &'static str),
        Arrived(Role),
    }

    #[test]
    fn speech_arrives_first_and_a_failure_waits_for_retry() {
        let (retry_tx, retry_rx) = mpsc::channel();
        let seen = std::cell::RefCell::new(Vec::new());
        let mut language_attempts = 0;
        let mut fetch = |n: &Needed, progress: &mut dyn FnMut(Progress)| {
            seen.borrow_mut().push(Seen::Fetch(n.role));
            progress(Progress {
                done: 50,
                total: 100,
            });
            if n.role == Role::Language {
                language_attempts += 1;
                if language_attempts == 1 {
                    // The retry is queued before the failure is seen, as a click would be.
                    retry_tx.send(()).unwrap();
                    return Err("reset".into());
                }
            }
            Ok(())
        };
        let mut report = |e: DownloadEvent| {
            let s = match e.state {
                DownloadState::Progress(_) => "progress",
                DownloadState::Done => "done",
                DownloadState::Failed(_) => "failed",
            };
            seen.borrow_mut().push(Seen::Report(e.role, s));
        };
        let mut arrived = |r| seen.borrow_mut().push(Seen::Arrived(r));
        run(
            vec![
                needed(Role::Language),
                needed(Role::Vad),
                needed(Role::Speech),
            ],
            &mut fetch,
            &mut report,
            &mut arrived,
            &retry_rx,
        );
        use Role::*;
        assert_eq!(
            seen.into_inner(),
            vec![
                Seen::Fetch(Speech),
                Seen::Report(Speech, "progress"),
                Seen::Report(Speech, "done"),
                Seen::Arrived(Speech),
                Seen::Fetch(Vad),
                Seen::Report(Vad, "progress"),
                Seen::Report(Vad, "done"),
                Seen::Arrived(Vad),
                Seen::Fetch(Language),
                Seen::Report(Language, "progress"),
                Seen::Report(Language, "failed"),
                Seen::Fetch(Language),
                Seen::Report(Language, "progress"),
                Seen::Report(Language, "done"),
                Seen::Arrived(Language),
            ]
        );
    }

    #[test]
    fn a_failure_with_nobody_to_retry_ends_the_run() {
        let (retry_tx, retry_rx) = mpsc::channel::<()>();
        drop(retry_tx);
        let mut fetched = Vec::new();
        run(
            vec![needed(Role::Speech), needed(Role::Language)],
            &mut |n, _| {
                fetched.push(n.role);
                Err("offline".into())
            },
            &mut |_| {},
            &mut |_| panic!("nothing arrived"),
            &retry_rx,
        );
        assert_eq!(fetched, vec![Role::Speech]);
    }
}
