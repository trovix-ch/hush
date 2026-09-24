//! One transcribe.cpp engine behind the pipe protocol in `hush_stt::protocol`. hush
//! starts and supervises it; a driver crash in here costs one utterance, not the app.

use std::fs::File;
use std::io::{self, BufReader, BufWriter};
use std::path::PathBuf;
use std::process::ExitCode;
use std::sync::mpsc;
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use hush_core::stt::{Backend, DecodeOptions, SttEngine, SttError};
use hush_core::{CancelToken, UtteranceId};
use hush_stt::protocol::{
    Reply, ReplyBody, Request, RequestFrame, WireInfo, bytes_to_pcm, parse_backend, read_frame,
    write_frame,
};
use hush_stt_worker::transcribe_cpp::{self, TranscribeCppEngine, TranscribeCppOptions};

const USAGE: &str = "\
usage: hush-stt-worker --model <path> [--backend vulkan|cpu] [--threads N]
       hush-stt-worker --list-devices

Speaks hush's framed protocol on stdin/stdout and logs to stderr. Started by hush, which
names the GPU by PCI bus id in the load request.";

const HEARTBEAT: Duration = Duration::from_secs(1);

struct Args {
    model: PathBuf,
    engine: TranscribeCppOptions,
}

enum Mode {
    Serve(Args),
    ListDevices,
}

fn parse_args() -> Result<Mode, String> {
    let mut it = std::env::args().skip(1);
    let mut model = None;
    let mut engine = TranscribeCppOptions::new(Backend::Vulkan);
    while let Some(a) = it.next() {
        let mut value = |name: &str| it.next().ok_or(format!("{name} needs a value"));
        match a.as_str() {
            "--list-devices" => return Ok(Mode::ListDevices),
            "--model" => model = Some(PathBuf::from(value("--model")?)),
            "--backend" => {
                let v = value("--backend")?;
                engine.backend = parse_backend(&v).ok_or(format!("unknown backend `{v}`"))?;
            }
            "--threads" => {
                let v = value("--threads")?;
                engine.threads = Some(v.parse().map_err(|_| format!("bad --threads `{v}`"))?);
            }
            other => return Err(format!("unexpected argument `{other}`")),
        }
    }
    let model = model.ok_or("--model is required")?;
    Ok(Mode::Serve(Args { model, engine }))
}

fn main() -> ExitCode {
    let proto = match protocol_output() {
        Ok(f) => f,
        Err(e) => {
            eprintln!("hush-stt-worker: cannot set up the protocol pipe: {e}");
            return ExitCode::FAILURE;
        }
    };
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| "info,transcribe_cpp=warn".into()),
        )
        .with_writer(io::stderr)
        .with_ansi(false)
        .without_time()
        .init();
    match parse_args() {
        Ok(Mode::ListDevices) => list_devices(proto),
        Ok(Mode::Serve(args)) => serve(&args, proto),
        Err(e) => {
            eprintln!("hush-stt-worker: {e}\n{USAGE}");
            ExitCode::from(2)
        }
    }
}

fn list_devices(proto: File) -> ExitCode {
    let mut out = BufWriter::new(proto);
    match serde_json::to_writer(&mut out, &transcribe_cpp::vulkan_devices()) {
        Ok(()) => ExitCode::SUCCESS,
        Err(e) => {
            tracing::error!(error = %e, "writing the device list");
            ExitCode::FAILURE
        }
    }
}

type Out = Arc<Mutex<BufWriter<File>>>;

fn send(out: &Out, id: u64, body: ReplyBody) -> io::Result<()> {
    let mut w = out
        .lock()
        .map_err(|_| io::Error::other("output lock poisoned"))?;
    write_frame(&mut *w, &Reply { id, body }, &[])
}

type InFlight = Arc<Mutex<Option<(u64, CancelToken)>>>;

fn serve(args: &Args, proto: File) -> ExitCode {
    let out: Out = Arc::new(Mutex::new(BufWriter::with_capacity(1 << 16, proto)));
    let in_flight: InFlight = Arc::new(Mutex::new(None));
    let (tx, rx) = mpsc::channel();
    let spawned =
        spawn_heartbeat(Arc::clone(&out)).and_then(|()| spawn_reader(Arc::clone(&in_flight), tx));
    if let Err(e) = spawned {
        tracing::error!(error = %e, "cannot start the worker threads");
        return ExitCode::FAILURE;
    }

    let mut engine: Option<TranscribeCppEngine> = None;
    let mut crash_after: Option<Duration> = None;
    for (frame, blob, token) in rx {
        let RequestFrame { id, request } = frame;
        let body = match request {
            Request::Load { gpu } => load(args, gpu, &mut engine),
            Request::WarmUp => timed(loaded(&mut engine).and_then(|e| {
                let t = Instant::now();
                e.warm_up().map(|()| t.elapsed())
            })),
            Request::Nudge => timed(loaded(&mut engine).and_then(|e| {
                let t = Instant::now();
                e.nudge().map(|()| t.elapsed())
            })),
            Request::Transcribe {
                utterance,
                language,
                prompt,
                hotwords,
                deadline_ms: _,
            } => {
                if let Some(after) = crash_after.take() {
                    arm_crash(after);
                }
                let opts = DecodeOptions {
                    language,
                    prompt,
                    hotwords,
                    utterance: UtteranceId(utterance),
                    cancel: token.unwrap_or_default(),
                };
                let result = bytes_to_pcm(&blob)
                    .map_err(|e| SttError::Inference(e.to_string()))
                    .and_then(|pcm| loaded(&mut engine)?.transcribe(&pcm, &opts));
                if let Ok(mut g) = in_flight.lock()
                    && g.as_ref().is_some_and(|(i, _)| *i == id)
                {
                    *g = None;
                }
                match result {
                    Ok(t) => ReplyBody::Transcript((&t).into()),
                    Err(e) => ReplyBody::Error((&e).into()),
                }
            }
            Request::Info => match loaded(&mut engine) {
                Ok(e) => ReplyBody::Info(e.info().into()),
                Err(e) => ReplyBody::Error((&e).into()),
            },
            Request::Crash { after_ms } => {
                crash_after = Some(Duration::from_millis(after_ms));
                ReplyBody::Done { elapsed_us: 0 }
            }
            Request::Shutdown => break,
            Request::Cancel => continue,
        };
        if let Err(e) = send(&out, id, body) {
            tracing::info!(error = %e, "parent stopped reading; exiting");
            break;
        }
    }
    ExitCode::SUCCESS
}

fn loaded(engine: &mut Option<TranscribeCppEngine>) -> Result<&mut TranscribeCppEngine, SttError> {
    engine
        .as_mut()
        .ok_or_else(|| SttError::Load("no model loaded; send Load first".into()))
}

fn load(args: &Args, gpu: Option<String>, engine: &mut Option<TranscribeCppEngine>) -> ReplyBody {
    *engine = None;
    let started = Instant::now();
    let opts = TranscribeCppOptions {
        gpu,
        ..args.engine.clone()
    };
    match TranscribeCppEngine::with_options(&args.model, opts) {
        Ok(e) => {
            let body = ReplyBody::Loaded {
                info: WireInfo::from(e.info()),
                load_us: micros(started.elapsed()),
            };
            *engine = Some(e);
            body
        }
        Err(e) => ReplyBody::Error((&e).into()),
    }
}

fn timed(r: Result<Duration, SttError>) -> ReplyBody {
    match r {
        Ok(d) => ReplyBody::Done {
            elapsed_us: micros(d),
        },
        Err(e) => ReplyBody::Error((&e).into()),
    }
}

fn micros(d: Duration) -> u64 {
    u64::try_from(d.as_micros()).unwrap_or(u64::MAX)
}

fn arm_crash(after: Duration) {
    let _ = std::thread::Builder::new()
        .name("crash-test".into())
        .spawn(move || {
            std::thread::sleep(after);
            tracing::error!("aborting mid-transcription, as the crash test asked");
            std::process::abort();
        });
}

/// A write failure means the parent is gone; nothing else would stop this process.
fn spawn_heartbeat(out: Out) -> io::Result<()> {
    std::thread::Builder::new()
        .name("heartbeat".into())
        .spawn(move || {
            loop {
                std::thread::sleep(HEARTBEAT);
                if send(&out, 0, ReplyBody::Heartbeat).is_err() {
                    std::process::exit(0);
                }
            }
        })
        .map(|_| ())
}

type Job = (RequestFrame, Vec<u8>, Option<CancelToken>);

/// Registers a transcription's token before the main thread sees the request, so a
/// cancel that follows it on the pipe always finds it.
fn spawn_reader(in_flight: InFlight, tx: mpsc::Sender<Job>) -> io::Result<()> {
    std::thread::Builder::new()
        .name("stdin".into())
        .spawn(move || {
            let mut r = BufReader::new(io::stdin().lock());
            loop {
                let (frame, blob) = match read_frame::<_, RequestFrame>(&mut r) {
                    Ok(Some(f)) => f,
                    // Mid-inference or not: nobody is left to read the answer.
                    Ok(None) => std::process::exit(0),
                    Err(e) => {
                        tracing::error!(error = %e, "unreadable request; exiting");
                        std::process::exit(2);
                    }
                };
                let token = match &frame.request {
                    Request::Cancel => {
                        if let Ok(g) = in_flight.lock()
                            && let Some((id, t)) = g.as_ref()
                            && *id == frame.id
                        {
                            t.cancel();
                        }
                        continue;
                    }
                    Request::Transcribe { deadline_ms, .. } => {
                        let t = CancelToken::new();
                        let t = match deadline_ms {
                            Some(ms) => t.with_timeout(Duration::from_millis(*ms)),
                            None => t,
                        };
                        if let Ok(mut g) = in_flight.lock() {
                            *g = Some((frame.id, t.clone()));
                        }
                        Some(t)
                    }
                    _ => None,
                };
                if tx.send((frame, blob, token)).is_err() {
                    std::process::exit(0);
                }
            }
        })
        .map(|_| ())
}

#[cfg(windows)]
unsafe extern "C" {
    fn _dup2(from: std::ffi::c_int, to: std::ffi::c_int) -> std::ffi::c_int;
}

/// Takes a private copy of stdout for the protocol and points everything else that
/// writes to stdout, the C runtime and native libraries included, at stderr: one stray
/// `printf` into the protocol pipe would corrupt the framing.
#[cfg(windows)]
fn protocol_output() -> io::Result<File> {
    use std::os::windows::io::AsHandle;
    use windows::Win32::System::Console::{
        GetStdHandle, STD_ERROR_HANDLE, STD_OUTPUT_HANDLE, SetStdHandle,
    };

    let proto = File::from(io::stdout().as_handle().try_clone_to_owned()?);
    // SAFETY: plain FFI on the process's standard handles. `_dup2` closes the CRT's
    // descriptor 1 and its original handle, which is why the copy above is taken first.
    unsafe {
        let err = GetStdHandle(STD_ERROR_HANDLE).map_err(io::Error::other)?;
        SetStdHandle(STD_OUTPUT_HANDLE, err).map_err(io::Error::other)?;
        _dup2(2, 1);
    }
    Ok(proto)
}

#[cfg(not(windows))]
fn protocol_output() -> io::Result<File> {
    use std::os::fd::AsFd;
    Ok(File::from(io::stdout().as_fd().try_clone_to_owned()?))
}
