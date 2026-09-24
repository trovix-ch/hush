//! `SttEngine` over a pipe to `hush-stt-worker`. A GPU driver crash then kills a child
//! process instead of the app, and transcribe.cpp's ggml never shares an address space
//! with llama.cpp's.

use std::io::{self, BufRead, BufReader, BufWriter, Read};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver, RecvTimeoutError, TryRecvError};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use hush_core::cancel::CancelToken;
use hush_core::gpu::GpuDevice;
use hush_core::stt::{Backend, DecodeOptions, EngineInfo, SttEngine, SttError, Transcript};

use crate::protocol::{
    ErrorKind, Reply, ReplyBody, Request, RequestFrame, backend_name, pcm_to_bytes, read_frame,
    write_frame,
};

/// The worker sends one every second; five missed means it is wedged, not busy.
const HEARTBEAT_TIMEOUT: Duration = Duration::from_secs(5);
/// Bounds how late a cancel reaches the worker.
const POLL: Duration = Duration::from_millis(5);
/// A worker that has not answered a cancel by then is presumed hung in the driver.
const CANCEL_GRACE: Duration = Duration::from_secs(2);
/// First-ever load compiles shaders for seconds; this only catches a load that never ends.
const START_TIMEOUT: Duration = Duration::from_secs(120);
const SHUTDOWN_GRACE: Duration = Duration::from_millis(500);
const LIST_DEVICES_TIMEOUT: Duration = Duration::from_secs(20);

pub fn worker_file_name() -> String {
    format!("hush-stt-worker{}", std::env::consts::EXE_SUFFIX)
}

/// Next to the running executable.
pub fn default_worker_path() -> io::Result<PathBuf> {
    Ok(std::env::current_exe()?.with_file_name(worker_file_name()))
}

#[derive(Debug, Clone)]
pub struct RemoteOptions {
    pub worker: PathBuf,
    pub model: PathBuf,
    pub backend: Backend,
    /// PCI bus id, sent with every load so a restarted worker lands on the same card.
    pub gpu: Option<String>,
    pub threads: Option<usize>,
}

impl RemoteOptions {
    pub fn new(worker: PathBuf, model: PathBuf, backend: Backend) -> Self {
        Self {
            worker,
            model,
            backend,
            gpu: None,
            threads: None,
        }
    }
}

fn configure(cmd: &mut Command) {
    cmd.stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());
    #[cfg(windows)]
    {
        use std::os::windows::process::CommandExt;
        // The worker is a console program; started from the tray app it would open a
        // console window.
        const CREATE_NO_WINDOW: u32 = 0x0800_0000;
        cmd.creation_flags(CREATE_NO_WINDOW);
    }
}

/// The Vulkan devices as the worker's transcribe.cpp sees them, free memory included.
pub fn list_devices(worker: &Path) -> Result<Vec<GpuDevice>, SttError> {
    let mut cmd = Command::new(worker);
    cmd.arg("--list-devices");
    configure(&mut cmd);
    cmd.stdin(Stdio::null()).stderr(Stdio::null());
    let job = Job::new().map_err(|e| SttError::Backend(format!("job object: {e}")))?;
    let mut child = cmd.spawn().map_err(|e| start_err(worker, &e))?;
    // Dropping the job on a timeout kills a child stuck in the driver.
    let _ = job.assign(&child);
    let mut stdout = child.stdout.take();
    let (tx, rx) = mpsc::channel();
    std::thread::Builder::new()
        .name("hush-stt-devices".into())
        .spawn(move || {
            let mut out = String::new();
            let read = stdout
                .as_mut()
                .map_or(Ok(0), |s| s.read_to_string(&mut out));
            let _ = tx.send(read.map(|_| out));
        })
        .map_err(|e| SttError::Backend(e.to_string()))?;
    let out = rx
        .recv_timeout(LIST_DEVICES_TIMEOUT)
        .map_err(|_| SttError::Backend("the worker did not list devices in time".into()))?
        .map_err(|e| SttError::Backend(e.to_string()))?;
    let _ = child.wait();
    serde_json::from_str(out.trim())
        .map_err(|e| SttError::Backend(format!("worker device list unreadable ({e}): {out:?}")))
}

fn start_err(worker: &Path, e: &io::Error) -> SttError {
    SttError::Load(format!("cannot start {}: {e}", worker.display()))
}

enum Incoming {
    Reply(Reply, Vec<u8>),
    Closed(String),
}

enum Failure {
    /// The worker exited, closed the pipe, stopped heartbeating or wrote garbage.
    Died(String),
    /// Cancelled or out of time and the worker did not answer the cancel; it was killed.
    Abandoned(SttError),
}

/// One worker process and the threads reading its output.
struct Worker {
    child: Child,
    stdin: BufWriter<ChildStdin>,
    replies: Receiver<Incoming>,
    last_seen: Arc<Mutex<Instant>>,
    last_log: Arc<Mutex<String>>,
    next_id: u64,
}

impl Worker {
    fn spawn(opts: &RemoteOptions, job: &Job) -> Result<Self, SttError> {
        let mut cmd = Command::new(&opts.worker);
        cmd.arg("--model")
            .arg(&opts.model)
            .arg("--backend")
            .arg(backend_name(opts.backend));
        if let Some(t) = opts.threads {
            cmd.arg("--threads").arg(t.to_string());
        }
        configure(&mut cmd);
        let mut child = cmd.spawn().map_err(|e| start_err(&opts.worker, &e))?;
        if let Err(e) = job.assign(&child) {
            tracing::warn!(error = %e, "speech worker is not in a kill-on-close job; it may outlive a crashed hush");
        }
        let pid = child.id();
        let (Some(stdin), Some(stdout), Some(stderr)) =
            (child.stdin.take(), child.stdout.take(), child.stderr.take())
        else {
            let _ = child.kill();
            return Err(SttError::Load("worker pipes were not created".into()));
        };
        let last_seen = Arc::new(Mutex::new(Instant::now()));
        let last_log = Arc::new(Mutex::new(String::new()));
        let (tx, replies) = mpsc::channel();
        let seen = Arc::clone(&last_seen);
        let spawned = std::thread::Builder::new()
            .name("hush-stt-rx".into())
            .spawn(move || {
                let mut r = BufReader::new(stdout);
                let closed = loop {
                    match read_frame::<_, Reply>(&mut r) {
                        Ok(Some((reply, blob))) => {
                            if let Ok(mut t) = seen.lock() {
                                *t = Instant::now();
                            }
                            if matches!(reply.body, ReplyBody::Heartbeat) {
                                continue;
                            }
                            if tx.send(Incoming::Reply(reply, blob)).is_err() {
                                return;
                            }
                        }
                        Ok(None) => break "the worker closed its output".to_string(),
                        Err(e) => break format!("worker output unreadable: {e}"),
                    }
                };
                let _ = tx.send(Incoming::Closed(closed));
            })
            .and_then(|_| {
                let log = Arc::clone(&last_log);
                std::thread::Builder::new()
                    .name("hush-stt-log".into())
                    .spawn(move || forward_log(stderr, pid, &log))
            });
        if let Err(e) = spawned {
            let _ = child.kill();
            return Err(SttError::Load(format!("worker threads: {e}")));
        }
        tracing::debug!(pid, worker = %opts.worker.display(), "speech worker started");
        Ok(Self {
            child,
            stdin: BufWriter::new(stdin),
            replies,
            last_seen,
            last_log,
            next_id: 0,
        })
    }

    fn pid(&self) -> u32 {
        self.child.id()
    }

    fn send(&mut self, id: u64, request: Request, blob: &[u8]) -> io::Result<()> {
        write_frame(&mut self.stdin, &RequestFrame { id, request }, blob)
    }

    /// Stale replies (ids from before an abandoned call) are dropped here.
    fn call(
        &mut self,
        request: Request,
        blob: &[u8],
        cancel: Option<&CancelToken>,
    ) -> Result<(ReplyBody, Vec<u8>), Failure> {
        self.next_id += 1;
        let id = self.next_id;
        if let Err(e) = self.send(id, request, blob) {
            return Err(Failure::Died(
                self.reap(&format!("writing to the worker: {e}")),
            ));
        }
        let mut cancel_sent = false;
        let mut stopped_at: Option<Instant> = None;
        loop {
            match self.replies.recv_timeout(POLL) {
                Ok(Incoming::Reply(r, blob)) if r.id == id => return Ok((r.body, blob)),
                Ok(Incoming::Reply(r, _)) => {
                    tracing::debug!(id = r.id, "dropping a stale speech worker reply");
                    continue;
                }
                Ok(Incoming::Closed(why)) => return Err(Failure::Died(self.reap(&why))),
                Err(RecvTimeoutError::Disconnected) => {
                    return Err(Failure::Died(self.reap("worker reader stopped")));
                }
                Err(RecvTimeoutError::Timeout) => {}
            }
            if let Some(why) = self.check_alive() {
                return Err(Failure::Died(self.reap(&why)));
            }
            let Some(c) = cancel else { continue };
            // Only a cancel crosses the pipe: the worker applies the deadline itself, so a
            // late deadline still returns finished work, as it does in process.
            if !cancel_sent && c.is_cancelled() {
                let _ = self.send(id, Request::Cancel, &[]);
                cancel_sent = true;
            }
            if stopped_at.is_none() && c.checkpoint().is_err() {
                stopped_at = Some(Instant::now());
            }
            if stopped_at.is_some_and(|at| at.elapsed() > CANCEL_GRACE) {
                let e = c
                    .checkpoint()
                    .err()
                    .map_or(SttError::Cancelled, SttError::from);
                let why = self.reap("did not stop after a cancel or its deadline");
                tracing::warn!(reason = %why, "speech worker killed");
                return Err(Failure::Abandoned(e));
            }
        }
    }

    fn check_alive(&mut self) -> Option<String> {
        if let Ok(Some(status)) = self.child.try_wait() {
            return Some(format!("worker exited ({status})"));
        }
        let quiet = self.last_seen.lock().map(|t| t.elapsed()).ok()?;
        (quiet > HEARTBEAT_TIMEOUT).then(|| {
            format!(
                "no heartbeat from the worker for {:.1} s",
                quiet.as_secs_f64()
            )
        })
    }

    /// Kills the process and describes how it ended.
    fn reap(&mut self, why: &str) -> String {
        let _ = self.child.kill();
        let status = self
            .child
            .wait()
            .map_or_else(|e| e.to_string(), |s| s.to_string());
        let last = self.last_log.lock().map(|l| l.clone()).unwrap_or_default();
        let mut msg = format!("{why}; worker {} ended ({status})", self.pid());
        if !last.is_empty() {
            msg.push_str("; its last log line: ");
            msg.push_str(&last);
        }
        msg
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        if matches!(self.child.try_wait(), Ok(None)) {
            let id = self.next_id + 1;
            let _ = self.send(id, Request::Shutdown, &[]);
            let until = Instant::now() + SHUTDOWN_GRACE;
            while Instant::now() < until {
                if matches!(self.child.try_wait(), Ok(Some(_))) {
                    return;
                }
                std::thread::sleep(Duration::from_millis(10));
            }
            let _ = self.child.kill();
        }
        let _ = self.child.wait();
    }
}

fn forward_log(stderr: impl Read, pid: u32, last: &Mutex<String>) {
    let mut r = BufReader::new(stderr);
    let mut buf = Vec::new();
    loop {
        buf.clear();
        match r.read_until(b'\n', &mut buf) {
            Ok(0) | Err(_) => return,
            Ok(_) => {}
        }
        let line = String::from_utf8_lossy(&buf);
        let line = line.trim_end();
        if line.is_empty() {
            continue;
        }
        tracing::debug!(target: "hush_stt::worker", "stt-worker[{pid}] {line}");
        if let Ok(mut l) = last.lock() {
            line.clone_into(&mut l);
        }
    }
}

struct Started {
    worker: Worker,
    info: EngineInfo,
    load: Duration,
    warm_up: Duration,
}

fn unexpected(body: &ReplyBody) -> SttError {
    SttError::Inference(format!("unexpected reply from the speech worker: {body:?}"))
}

fn start(opts: &RemoteOptions, job: &Job) -> Result<Started, SttError> {
    let mut worker = Worker::spawn(opts, job)?;
    let limit = CancelToken::new().with_timeout(START_TIMEOUT);
    let call = |w: &mut Worker, req: Request| match w.call(req, &[], Some(&limit)) {
        Ok((ReplyBody::Error(e), _)) => Err(SttError::from(e)),
        Ok((body, _)) => Ok(body),
        Err(Failure::Died(why)) => Err(SttError::Load(why)),
        Err(Failure::Abandoned(_)) => Err(SttError::Load(format!(
            "the worker did not finish starting within {} s",
            START_TIMEOUT.as_secs()
        ))),
    };
    let load = Request::Load {
        gpu: opts.gpu.clone(),
    };
    let (info, load) = match call(&mut worker, load)? {
        ReplyBody::Loaded { info, load_us } => (info.into_info()?, Duration::from_micros(load_us)),
        other => return Err(unexpected(&other)),
    };
    let warm_up = match call(&mut worker, Request::WarmUp)? {
        ReplyBody::Done { elapsed_us } => Duration::from_micros(elapsed_us),
        other => return Err(unexpected(&other)),
    };
    tracing::info!(
        pid = worker.pid(),
        id = %info.id,
        backend = ?info.backend,
        device = info.device.as_deref().unwrap_or("?"),
        load_ms = load.as_millis() as u64,
        warm_up_ms = warm_up.as_millis() as u64,
        "speech worker ready"
    );
    Ok(Started {
        worker,
        info,
        load,
        warm_up,
    })
}

enum State {
    Ready(Worker),
    Starting(Receiver<Result<Started, SttError>>),
    /// The last start failed; the next call tries again.
    Down(String),
}

pub struct RemoteEngine {
    // Before `job`: fields drop in order, and closing the job kills the worker before it
    // could be asked to shut down.
    state: State,
    job: Arc<Job>,
    opts: RemoteOptions,
    info: EngineInfo,
    load: Duration,
    warm_up: Duration,
    restarts: u32,
}

impl RemoteEngine {
    /// Starts the worker, loads the model and warms it up; blocks for seconds. The info
    /// is what the worker reports it loaded, not what was asked for.
    pub fn new(opts: RemoteOptions) -> Result<Self, SttError> {
        let job = Arc::new(Job::new().map_err(|e| SttError::Backend(format!("job object: {e}")))?);
        let started = start(&opts, &job)?;
        Ok(Self {
            opts,
            job,
            info: started.info,
            load: started.load,
            warm_up: started.warm_up,
            state: State::Ready(started.worker),
            restarts: 0,
        })
    }

    /// `None` while a replacement is starting.
    pub fn worker_pid(&self) -> Option<u32> {
        match &self.state {
            State::Ready(w) => Some(w.pid()),
            _ => None,
        }
    }

    /// Measured inside the worker, so process start is not in it.
    pub fn load_time(&self) -> Duration {
        self.load
    }

    pub fn warm_up_time(&self) -> Duration {
        self.warm_up
    }

    pub fn restarts(&self) -> u32 {
        self.restarts
    }

    /// Asks the running worker rather than returning the cached copy.
    pub fn worker_info(&mut self) -> Result<EngineInfo, SttError> {
        match self.call(Request::Info, &[], None)? {
            ReplyBody::Info(i) => i.into_info(),
            other => Err(unexpected(&other)),
        }
    }

    /// Test hook: the worker aborts `after` into the next transcription.
    #[doc(hidden)]
    pub fn arm_crash(&mut self, after: Duration) -> Result<(), SttError> {
        let after_ms = u64::try_from(after.as_millis()).unwrap_or(u64::MAX);
        match self.call(Request::Crash { after_ms }, &[], None)? {
            ReplyBody::Done { .. } => Ok(()),
            other => Err(unexpected(&other)),
        }
    }

    /// Blocks until a replacement is ready when one is starting.
    pub fn wait_ready(&mut self, cancel: Option<&CancelToken>) -> Result<(), SttError> {
        loop {
            match std::mem::replace(&mut self.state, State::Down(String::new())) {
                State::Ready(w) => {
                    self.state = State::Ready(w);
                    return Ok(());
                }
                State::Down(why) => {
                    tracing::info!(reason = %why, "starting the speech worker again");
                    self.restart();
                }
                State::Starting(rx) => match rx.recv_timeout(POLL) {
                    Ok(r) => self.adopt(r)?,
                    Err(RecvTimeoutError::Timeout) => {
                        self.state = State::Starting(rx);
                        if let Some(c) = cancel {
                            c.checkpoint()?;
                        }
                    }
                    Err(RecvTimeoutError::Disconnected) => {
                        let why = "the worker restart thread ended without a result".to_string();
                        self.state = State::Down(why.clone());
                        return Err(SttError::BackendDied(why));
                    }
                },
            }
        }
    }

    fn adopt(&mut self, r: Result<Started, SttError>) -> Result<(), SttError> {
        match r {
            Ok(s) => {
                if s.info.backend != self.info.backend || s.info.device != self.info.device {
                    tracing::warn!(before = ?self.info.backend, after = ?s.info.backend, device = s.info.device.as_deref().unwrap_or("?"), "restarted speech worker loaded elsewhere");
                }
                self.info = s.info;
                self.load = s.load;
                self.warm_up = s.warm_up;
                self.state = State::Ready(s.worker);
                Ok(())
            }
            Err(e) => {
                tracing::warn!(error = %e, "speech worker restart failed");
                self.state = State::Down(e.to_string());
                Err(e)
            }
        }
    }

    /// Replaces the worker in the background; the old one is already dead or killed.
    fn restart(&mut self) {
        self.restarts += 1;
        let (tx, rx) = mpsc::channel();
        let opts = self.opts.clone();
        let job = Arc::clone(&self.job);
        let spawned = std::thread::Builder::new()
            .name("hush-stt-restart".into())
            .spawn(move || {
                let _ = tx.send(start(&opts, &job));
            });
        self.state = match spawned {
            Ok(_) => State::Starting(rx),
            Err(e) => State::Down(format!("cannot start the restart thread: {e}")),
        };
    }

    fn call(
        &mut self,
        request: Request,
        blob: &[u8],
        cancel: Option<&CancelToken>,
    ) -> Result<ReplyBody, SttError> {
        self.wait_ready(cancel)?;
        let State::Ready(worker) = &mut self.state else {
            return Err(SttError::BackendDied("speech worker is not running".into()));
        };
        match worker.call(request, blob, cancel) {
            Ok((ReplyBody::Error(e), _)) if e.kind == ErrorKind::BackendDied => {
                self.restart();
                Err(e.into())
            }
            Ok((ReplyBody::Error(e), _)) => Err(e.into()),
            Ok((body, _)) => Ok(body),
            Err(Failure::Died(why)) => {
                tracing::warn!(reason = %why, "speech worker died; restarting it");
                self.restart();
                Err(SttError::BackendDied(why))
            }
            Err(Failure::Abandoned(e)) => {
                self.restart();
                Err(e)
            }
        }
    }
}

impl SttEngine for RemoteEngine {
    fn info(&self) -> &EngineInfo {
        &self.info
    }

    fn warm_up(&mut self) -> Result<(), SttError> {
        match self.call(Request::WarmUp, &[], None)? {
            ReplyBody::Done { .. } => Ok(()),
            other => Err(unexpected(&other)),
        }
    }

    /// Never waits for a restarting worker: this runs at key-down, and the transcription
    /// that follows waits anyway.
    fn nudge(&mut self) -> Result<(), SttError> {
        match std::mem::replace(&mut self.state, State::Down(String::new())) {
            State::Ready(w) => self.state = State::Ready(w),
            State::Down(why) => {
                self.state = State::Down(why);
                self.restart();
                return Ok(());
            }
            State::Starting(rx) => {
                match rx.try_recv() {
                    Ok(r) => self.adopt(r)?,
                    Err(TryRecvError::Empty) => self.state = State::Starting(rx),
                    Err(TryRecvError::Disconnected) => {
                        self.state = State::Down("restart thread vanished".into());
                    }
                }
                return Ok(());
            }
        }
        match self.call(Request::Nudge, &[], None) {
            Ok(_) => Ok(()),
            Err(SttError::BackendDied(why)) => {
                tracing::warn!(reason = %why, "speech worker died at nudge; it is restarting");
                Ok(())
            }
            Err(e) => Err(e),
        }
    }

    fn transcribe(&mut self, pcm: &[f32], opts: &DecodeOptions) -> Result<Transcript, SttError> {
        opts.cancel.checkpoint()?;
        self.wait_ready(Some(&opts.cancel))?;
        let request = Request::Transcribe {
            utterance: opts.utterance.0,
            language: opts.language.clone(),
            prompt: opts.prompt.clone(),
            hotwords: opts.hotwords.clone(),
            deadline_ms: opts
                .cancel
                .remaining()
                .map(|d| u64::try_from(d.as_millis()).unwrap_or(u64::MAX)),
        };
        match self.call(request, &pcm_to_bytes(pcm), Some(&opts.cancel))? {
            ReplyBody::Transcript(t) => Ok(t.into()),
            other => Err(unexpected(&other)),
        }
    }
}

#[cfg(windows)]
use job::Job;

/// A kill-on-close job: when the last handle closes, including when the parent dies
/// without running any destructor, Windows terminates every process in it.
#[cfg(windows)]
mod job {
    use std::io;
    use std::os::windows::io::AsRawHandle;
    use std::process::Child;

    use windows::Win32::Foundation::{CloseHandle, HANDLE};
    use windows::Win32::System::JobObjects::{
        AssignProcessToJobObject, CreateJobObjectW, JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE,
        JOBOBJECT_EXTENDED_LIMIT_INFORMATION, JobObjectExtendedLimitInformation,
        SetInformationJobObject,
    };
    use windows::core::PCWSTR;

    pub struct Job(HANDLE);

    // SAFETY: a job handle is a kernel object handle, usable from any thread.
    unsafe impl Send for Job {}
    // SAFETY: as above; the handle is only read after construction.
    unsafe impl Sync for Job {}

    impl Job {
        pub fn new() -> io::Result<Self> {
            // SAFETY: no attributes and no name; the returned handle is owned by `Job`.
            let handle =
                unsafe { CreateJobObjectW(None, PCWSTR::null()) }.map_err(io::Error::other)?;
            let job = Self(handle);
            let mut info = JOBOBJECT_EXTENDED_LIMIT_INFORMATION::default();
            info.BasicLimitInformation.LimitFlags = JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE;
            // SAFETY: `info` is a valid JOBOBJECT_EXTENDED_LIMIT_INFORMATION and the size
            // passed is its size.
            unsafe {
                SetInformationJobObject(
                    job.0,
                    JobObjectExtendedLimitInformation,
                    (&raw const info).cast(),
                    size_of::<JOBOBJECT_EXTENDED_LIMIT_INFORMATION>() as u32,
                )
            }
            .map_err(io::Error::other)?;
            Ok(job)
        }

        pub fn assign(&self, child: &Child) -> io::Result<()> {
            // SAFETY: both handles are open for the duration of the call; `Child` owns its
            // process handle until it is dropped.
            unsafe { AssignProcessToJobObject(self.0, HANDLE(child.as_raw_handle())) }
                .map_err(io::Error::other)
        }
    }

    impl Drop for Job {
        fn drop(&mut self) {
            // SAFETY: the handle came from CreateJobObjectW and is closed exactly once.
            let _ = unsafe { CloseHandle(self.0) };
        }
    }
}

#[cfg(not(windows))]
struct Job;

#[cfg(not(windows))]
impl Job {
    fn new() -> io::Result<Self> {
        Ok(Self)
    }

    fn assign(&self, _: &Child) -> io::Result<()> {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_missing_worker_is_a_load_error() {
        let opts = RemoteOptions::new(
            PathBuf::from("definitely-not-here/hush-stt-worker.exe"),
            PathBuf::from("x.gguf"),
            Backend::Cpu,
        );
        let err = RemoteEngine::new(opts).err().unwrap();
        assert!(matches!(err, SttError::Load(_)), "{err}");
        assert!(list_devices(Path::new("definitely-not-here/w.exe")).is_err());
    }
}
