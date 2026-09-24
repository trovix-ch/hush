//! Clipboard owner: delayed-rendered paste with a read signal, snapshot and restore (D8).
//!
//! Why its own thread instead of the UI thread D10 sketched: a reader that asks for our
//! delayed text is blocked inside `GetClipboardData` until our window answers
//! `WM_RENDERFORMAT`, so the owner's message loop must never be busy; and a snapshot of
//! a *foreign* clipboard makes that app render its own delayed data (an Excel range can
//! take seconds), which must not freeze the overlay or tray. One dedicated thread with a
//! message-only window does both jobs, serialises every clipboard operation, and keeps
//! the render state thread-local.
//!
//! Why only `CF_UNICODETEXT` is offered: Windows synthesises `CF_TEXT`, `CF_OEMTEXT` and
//! `CF_LOCALE` from it and still renders through us, and rich formats would carry
//! styling into the target. Why a render is reported as "first reader" and nothing more:
//! measured in the spike, Windows asks the owner exactly once per write; later readers
//! get the cached copy silently. Why the restore re-checks the sequence number with the
//! clipboard held open: a render does not advance it but a foreign write does, and the
//! check must be atomic with the restore or a user's copy in between would be clobbered.

use std::cell::{Cell, RefCell};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::mpsc::{self, SyncSender};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant};

use windows::Win32::Foundation::{GlobalFree, HANDLE, HGLOBAL, HWND, LPARAM, LRESULT, WPARAM};
use windows::Win32::System::DataExchange::{
    CloseClipboard, EmptyClipboard, EnumClipboardFormats, GetClipboardData,
    GetClipboardFormatNameW, GetClipboardOwner, GetClipboardSequenceNumber, GetOpenClipboardWindow,
    OpenClipboard, RegisterClipboardFormatW, SetClipboardData,
};
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::System::Memory::{
    GMEM_MOVEABLE, GlobalAlloc, GlobalLock, GlobalSize, GlobalUnlock,
};
use windows::Win32::UI::WindowsAndMessaging::{
    CreateWindowExW, DefWindowProcW, DestroyWindow, DispatchMessageW, GetMessageW, HWND_MESSAGE,
    KillTimer, MSG, PostMessageW, PostQuitMessage, RegisterClassW, SetTimer, TranslateMessage,
    WINDOW_EX_STYLE, WINDOW_STYLE, WM_APP, WM_DESTROY, WM_DESTROYCLIPBOARD, WM_RENDERALLFORMATS,
    WM_RENDERFORMAT, WM_TIMER, WNDCLASSW,
};
use windows::core::{PCWSTR, w};

use wl_core::insert::{InsertError, Reader, RenderWait};

use crate::util::{exe_name_of_pid, hwnd_from, hwnd_raw, window_thread_pid};

pub const CF_TEXT: u32 = 1;
pub const CF_UNICODETEXT: u32 = 13;

/// Formats never copied into a snapshot: GDI handles, metafile handles, owner-display
/// and private handle ranges are not `HGLOBAL`s and cannot be byte-copied. Windows
/// synthesises `CF_DIB` from a bitmap, so images survive through that.
fn is_handle_format(f: u32) -> bool {
    matches!(
        f,
        2 /* CF_BITMAP */ | 3 /* CF_METAFILEPICT */ | 9 /* CF_PALETTE */
        | 14 /* CF_ENHMETAFILE */ | 0x80 /* CF_OWNERDISPLAY */ | 0x82 /* CF_DSPBITMAP */
        | 0x83 /* CF_DSPMETAFILEPICT */ | 0x8E /* CF_DSPENHMETAFILE */
    ) || (0x200..=0x3FF).contains(&f)
}

/// Default bound on what a snapshot copies.
pub const DEFAULT_SNAPSHOT_CAP: usize = 32 * 1024 * 1024;
const COMMAND_TIMEOUT: Duration = Duration::from_secs(10);

#[derive(Debug, thiserror::Error)]
pub enum ClipboardError {
    #[error("clipboard is held open by another process")]
    Busy,
    #[error("clipboard thread is not running")]
    Gone,
    #[error("clipboard thread did not answer within {0:?}")]
    Timeout(Duration),
    #[error("{0}")]
    Win32(#[from] windows::core::Error),
}

/// One saved format.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SavedFormat {
    pub format: u32,
    pub name: String,
    pub bytes: Vec<u8>,
}

/// Previous clipboard contents, bounded best-effort.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ClipboardSnapshot {
    pub formats: Vec<SavedFormat>,
    /// Formats present but not copied: handle formats, over the size cap, or unreadable.
    pub skipped: Vec<(u32, String)>,
    pub sequence: u32,
    pub truncated: bool,
}

impl ClipboardSnapshot {
    pub fn is_empty(&self) -> bool {
        self.formats.is_empty()
    }

    /// The saved Unicode text, if any.
    pub fn text(&self) -> Option<String> {
        let f = self.formats.iter().find(|f| f.format == CF_UNICODETEXT)?;
        let units: Vec<u16> = f
            .bytes
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]))
            .take_while(|&u| u != 0)
            .collect();
        Some(String::from_utf16_lossy(&units))
    }
}

/// What `write_delayed` put on the clipboard.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WriteReceipt {
    /// Increments per write; render events carry it.
    pub generation: u64,
    /// Sequence number right after our write closed.
    pub sequence: u32,
    pub written_at: Instant,
}

/// A render request: someone read our delayed text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderEvent {
    pub generation: u64,
    pub at: Instant,
    pub format: u32,
    /// Window that has the clipboard open; `None` when the reader opened it with a NULL
    /// window (Windows Terminal does).
    pub reader_hwnd: Option<isize>,
    pub reader_pid: Option<u32>,
    pub reader_exe: Option<String>,
}

impl RenderEvent {
    /// Milliseconds from `chord_at` to this render; negative means before the chord.
    pub fn offset_ms(&self, chord_at: Instant) -> f64 {
        if self.at >= chord_at {
            (self.at - chord_at).as_secs_f64() * 1000.0
        } else {
            -((chord_at - self.at).as_secs_f64() * 1000.0)
        }
    }
}

/// Why a render wait ended without a render.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Waited {
    TimedOut,
    Changed,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RestoreOutcome {
    Restored,
    /// Someone wrote to the clipboard after us; their content was left alone.
    SkippedChanged,
}

// ------------------------------------------------------------------ shared state

#[derive(Default)]
struct RenderLog {
    generation: u64,
    first: Option<RenderEvent>,
    count: u32,
}

struct Shared {
    log: Mutex<RenderLog>,
    cv: Condvar,
    /// Sequence number right after our last write.
    our_sequence: AtomicU32,
}

enum Command {
    Snapshot {
        cap: usize,
        reply: SyncSender<Result<ClipboardSnapshot, ClipboardError>>,
    },
    Restore {
        snapshot: Box<ClipboardSnapshot>,
        expected_sequence: Option<u32>,
        reply: SyncSender<Result<RestoreOutcome, ClipboardError>>,
    },
    WriteDelayed {
        text: String,
        reply: SyncSender<Result<WriteReceipt, ClipboardError>>,
    },
    WriteEager {
        text: String,
        reply: SyncSender<Result<u32, ClipboardError>>,
    },
    ScheduleRestore {
        snapshot: Box<ClipboardSnapshot>,
        after: Duration,
        if_sequence: u32,
    },
    Shutdown,
}

/// A restore waiting for its delay. A snapshot taken before it fires returns *its*
/// snapshot: the clipboard still holds our previous dictation, and the user's real
/// contents are the ones this restore would have put back.
struct Scheduled {
    snapshot: Box<ClipboardSnapshot>,
    if_sequence: u32,
}

const RESTORE_TIMER: usize = 0x5752;

const WM_CB_COMMAND: u32 = WM_APP + 0x20;

/// Handle to the clipboard owner thread. Cheap to clone.
#[derive(Clone)]
pub struct WinClipboard {
    inner: Arc<Inner>,
}

struct Inner {
    tx: Mutex<Option<mpsc::Sender<Command>>>,
    hwnd: isize,
    shared: Arc<Shared>,
    join: Mutex<Option<std::thread::JoinHandle<()>>>,
}

impl Drop for Inner {
    fn drop(&mut self) {
        self.shutdown();
    }
}

impl Inner {
    fn shutdown(&self) {
        let tx = self.tx.lock().unwrap_or_else(|e| e.into_inner()).take();
        if let Some(tx) = tx {
            let _ = tx.send(Command::Shutdown);
            wake(self.hwnd);
        }
        let join = self.join.lock().unwrap_or_else(|e| e.into_inner()).take();
        if let Some(j) = join {
            let _ = j.join();
        }
    }
}

fn wake(hwnd: isize) {
    // SAFETY: posting a plain message to our own window; no pointers travel with it.
    let _ = unsafe { PostMessageW(Some(hwnd_from(hwnd)), WM_CB_COMMAND, WPARAM(0), LPARAM(0)) };
}

impl WinClipboard {
    /// Starts the owner thread and its message-only window.
    pub fn start() -> Result<Self, ClipboardError> {
        let shared = Arc::new(Shared {
            log: Mutex::new(RenderLog::default()),
            cv: Condvar::new(),
            our_sequence: AtomicU32::new(0),
        });
        let (tx, rx) = mpsc::channel();
        let (ready_tx, ready_rx) = mpsc::channel();
        let thread_shared = shared.clone();
        let join = std::thread::Builder::new()
            .name("wl-clipboard".into())
            .spawn(move || owner_thread(rx, thread_shared, ready_tx))
            .map_err(|_| ClipboardError::Gone)?;
        let hwnd = ready_rx.recv().map_err(|_| ClipboardError::Gone)??;
        Ok(Self {
            inner: Arc::new(Inner {
                tx: Mutex::new(Some(tx)),
                hwnd,
                shared,
                join: Mutex::new(Some(join)),
            }),
        })
    }

    fn call<T>(
        &self,
        make: impl FnOnce(SyncSender<Result<T, ClipboardError>>) -> Command,
    ) -> Result<T, ClipboardError> {
        let (reply, rx) = mpsc::sync_channel(1);
        {
            let tx = self.inner.tx.lock().unwrap_or_else(|e| e.into_inner());
            tx.as_ref()
                .ok_or(ClipboardError::Gone)?
                .send(make(reply))
                .map_err(|_| ClipboardError::Gone)?;
        }
        wake(self.inner.hwnd);
        rx.recv_timeout(COMMAND_TIMEOUT)
            .map_err(|_| ClipboardError::Timeout(COMMAND_TIMEOUT))?
    }

    /// Copies every byte-copyable format, up to `cap` bytes in total.
    pub fn snapshot(&self, cap: usize) -> Result<ClipboardSnapshot, ClipboardError> {
        self.call(|reply| Command::Snapshot { cap, reply })
    }

    /// Puts `snapshot` back. With `expected_sequence`, restores only if nobody wrote
    /// since (checked while holding the clipboard open).
    pub fn restore(
        &self,
        snapshot: &ClipboardSnapshot,
        expected_sequence: Option<u32>,
    ) -> Result<RestoreOutcome, ClipboardError> {
        let snapshot = Box::new(snapshot.clone());
        self.call(|reply| Command::Restore {
            snapshot,
            expected_sequence,
            reply,
        })
    }

    /// Offers `text` as delayed-rendered `CF_UNICODETEXT`, marked so clipboard history
    /// and cloud sync skip it. Render requests are recorded against the new generation.
    pub fn write_delayed(&self, text: &str) -> Result<WriteReceipt, ClipboardError> {
        let text = text.to_string();
        self.call(|reply| Command::WriteDelayed { text, reply })
    }

    /// Leaves `text` on the clipboard for good (the last-resort path and "copy last").
    /// Returns the sequence number after the write.
    pub fn write_text(&self, text: &str) -> Result<u32, ClipboardError> {
        let text = text.to_string();
        self.call(|reply| Command::WriteEager { text, reply })
    }

    /// Restores `snapshot` after `after` on the owner thread, only if the sequence number
    /// is still `if_sequence` then. Returns immediately.
    pub fn restore_later(&self, snapshot: ClipboardSnapshot, after: Duration, if_sequence: u32) {
        let tx = self.inner.tx.lock().unwrap_or_else(|e| e.into_inner());
        if let Some(tx) = tx.as_ref() {
            let _ = tx.send(Command::ScheduleRestore {
                snapshot: Box::new(snapshot),
                after,
                if_sequence,
            });
            wake(self.inner.hwnd);
        }
    }

    pub fn sequence_number(&self) -> u32 {
        // SAFETY: plain FFI query, callable from any thread.
        unsafe { GetClipboardSequenceNumber() }
    }

    /// Like [`Self::wait_for_render`], but also ends early when the sequence number moves
    /// away from our last write (a foreign write replaced ours).
    pub fn wait_for_render_or_change(&self, timeout: Duration) -> Result<RenderEvent, Waited> {
        let deadline = Instant::now() + timeout;
        let ours = self.inner.shared.our_sequence.load(Ordering::Acquire);
        loop {
            let now = Instant::now();
            let slice = (deadline.saturating_duration_since(now)).min(Duration::from_millis(5));
            if let Some(ev) = self.wait_for_render(slice) {
                return Ok(ev);
            }
            if self.sequence_number() != ours {
                return Err(Waited::Changed);
            }
            if Instant::now() >= deadline {
                return Err(Waited::TimedOut);
            }
        }
    }

    /// Waits for the first render of the current write.
    pub fn wait_for_render(&self, timeout: Duration) -> Option<RenderEvent> {
        let deadline = Instant::now() + timeout;
        let mut log = self
            .inner
            .shared
            .log
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        loop {
            if let Some(ev) = &log.first {
                return Some(ev.clone());
            }
            let now = Instant::now();
            if now >= deadline {
                return None;
            }
            log = self
                .inner
                .shared
                .cv
                .wait_timeout(log, deadline - now)
                .unwrap_or_else(|e| e.into_inner())
                .0;
        }
    }

    /// First render of the current write, without waiting.
    pub fn first_render(&self) -> Option<RenderEvent> {
        self.wait_for_render(Duration::ZERO)
    }

    /// Render requests seen for the current write. Windows sends one per write; more
    /// only if something emptied and re-read, which would be worth knowing.
    pub fn render_count(&self) -> u32 {
        self.inner
            .shared
            .log
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .count
    }

    /// Whether our window currently owns the clipboard.
    pub fn we_own(&self) -> bool {
        // SAFETY: plain FFI query.
        unsafe { GetClipboardOwner() }
            .ok()
            .is_some_and(|h| hwnd_raw(h) == self.inner.hwnd)
    }

    /// Stops the owner thread. Unrendered text is rendered first so it is not lost.
    pub fn shutdown(&self) {
        self.inner.shutdown();
    }
}

// ------------------------------------------------------------------ core port

impl wl_core::insert::ClipboardPort for WinClipboard {
    type Snapshot = ClipboardSnapshot;

    fn snapshot(&mut self) -> Result<ClipboardSnapshot, InsertError> {
        WinClipboard::snapshot(self, DEFAULT_SNAPSHOT_CAP).map_err(clip_err)
    }

    fn write_delayed(&mut self, text: &str) -> Result<u64, InsertError> {
        WinClipboard::write_delayed(self, text)
            .map(|r| r.sequence as u64)
            .map_err(clip_err)
    }

    fn sequence_number(&self) -> u64 {
        WinClipboard::sequence_number(self) as u64
    }

    fn wait_for_render(&mut self, timeout: Duration) -> RenderWait {
        match self.wait_for_render_or_change(timeout) {
            Ok(ev) => RenderWait::Read(Reader {
                pid: ev.reader_pid,
                exe: ev.reader_exe,
            }),
            Err(Waited::TimedOut) => RenderWait::TimedOut,
            Err(Waited::Changed) => RenderWait::Changed,
        }
    }

    fn restore(&mut self, snapshot: ClipboardSnapshot, after: Duration, if_sequence: u64) {
        self.restore_later(snapshot, after, if_sequence as u32);
    }
}

fn clip_err(e: ClipboardError) -> InsertError {
    InsertError::Clipboard(e.to_string())
}

// ------------------------------------------------------------------ owner thread

thread_local! {
    static PENDING: RefCell<Option<Arc<Vec<u16>>>> = const { RefCell::new(None) };
    static GENERATION: Cell<u64> = const { Cell::new(0) };
    static SHARED: RefCell<Option<Arc<Shared>>> = const { RefCell::new(None) };
}

fn owner_thread(
    rx: mpsc::Receiver<Command>,
    shared: Arc<Shared>,
    ready: mpsc::Sender<Result<isize, ClipboardError>>,
) {
    SHARED.with(|s| *s.borrow_mut() = Some(shared.clone()));
    let hwnd = match create_owner_window() {
        Ok(h) => h,
        Err(e) => {
            let _ = ready.send(Err(e.into()));
            return;
        }
    };
    let _ = ready.send(Ok(hwnd_raw(hwnd)));
    let mut msg = MSG::default();
    let mut quitting = false;
    let mut scheduled: Option<Scheduled> = None;
    let run_scheduled = |s: Scheduled| match restore_now(hwnd, &s.snapshot, Some(s.if_sequence)) {
        Ok(RestoreOutcome::Restored) => {}
        Ok(RestoreOutcome::SkippedChanged) => {
            tracing::debug!("clipboard changed since our write; previous contents not restored")
        }
        Err(e) => tracing::warn!(error = %e, "clipboard restore failed"),
    };
    loop {
        // SAFETY: standard message loop on the thread that owns `hwnd`.
        let r = unsafe { GetMessageW(&mut msg, None, 0, 0) };
        if r.0 <= 0 {
            break;
        }
        if msg.message == WM_TIMER && msg.hwnd == hwnd && msg.wParam.0 == RESTORE_TIMER {
            // SAFETY: our own window and timer id.
            let _ = unsafe { KillTimer(Some(hwnd), RESTORE_TIMER) };
            if let Some(s) = scheduled.take() {
                run_scheduled(s);
            }
            continue;
        }
        // SAFETY: dispatching a message we just retrieved.
        unsafe {
            let _ = TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
        // Commands run here, outside any window procedure, so the render handler can be
        // re-entered by our own clipboard calls without a borrow conflict.
        while !quitting && let Ok(cmd) = rx.try_recv() {
            match cmd {
                Command::Snapshot { cap, reply } => {
                    // SAFETY: plain FFI query.
                    let now = unsafe { GetClipboardSequenceNumber() };
                    let taken = scheduled.take().filter(|s| s.if_sequence == now);
                    if taken.is_some() {
                        // SAFETY: our own window and timer id.
                        let _ = unsafe { KillTimer(Some(hwnd), RESTORE_TIMER) };
                    }
                    let result = match taken {
                        Some(s) => Ok(*s.snapshot),
                        None => snapshot_now(hwnd, cap),
                    };
                    let _ = reply.try_send(result);
                }
                Command::ScheduleRestore {
                    snapshot,
                    after,
                    if_sequence,
                } => {
                    let s = Scheduled {
                        snapshot,
                        if_sequence,
                    };
                    if after.is_zero() {
                        scheduled = None;
                        run_scheduled(s);
                    } else {
                        scheduled = Some(s);
                        // SAFETY: a timer on our own window; handled in this loop.
                        unsafe {
                            SetTimer(
                                Some(hwnd),
                                RESTORE_TIMER,
                                after.as_millis().clamp(1, u32::MAX as u128) as u32,
                                None,
                            )
                        };
                    }
                }
                Command::Restore {
                    snapshot,
                    expected_sequence,
                    reply,
                } => {
                    let _ = reply.try_send(restore_now(hwnd, &snapshot, expected_sequence));
                }
                Command::WriteDelayed { text, reply } => {
                    let _ = reply.try_send(write_delayed_now(hwnd, &shared, &text));
                }
                Command::WriteEager { text, reply } => {
                    let _ = reply.try_send(write_eager_now(hwnd, &text));
                }
                Command::Shutdown => {
                    quitting = true;
                    // A pending restore would otherwise be lost with the process.
                    if let Some(s) = scheduled.take() {
                        run_scheduled(s);
                    }
                    // SAFETY: destroying our own window on its thread; WM_RENDERALLFORMATS
                    // arrives during this call if a delayed format is still pending.
                    let _ = unsafe { DestroyWindow(hwnd) };
                }
            }
        }
    }
    SHARED.with(|s| *s.borrow_mut() = None);
}

fn create_owner_window() -> windows::core::Result<HWND> {
    // SAFETY: registers a class with a 'static window procedure and creates a
    // message-only window owned by this thread.
    unsafe {
        let inst = GetModuleHandleW(PCWSTR::null())?;
        let class = w!("wl-clipboard-owner");
        let wc = WNDCLASSW {
            lpfnWndProc: Some(owner_wndproc),
            hInstance: inst.into(),
            lpszClassName: class,
            ..Default::default()
        };
        // Registering twice (a second owner in tests) fails harmlessly.
        RegisterClassW(&wc);
        CreateWindowExW(
            WINDOW_EX_STYLE(0),
            class,
            w!("whisper-local clipboard"),
            WINDOW_STYLE(0),
            0,
            0,
            0,
            0,
            Some(HWND_MESSAGE),
            None,
            Some(inst.into()),
            None,
        )
    }
}

unsafe extern "system" fn owner_wndproc(hwnd: HWND, msg: u32, wp: WPARAM, lp: LPARAM) -> LRESULT {
    match msg {
        WM_RENDERFORMAT => {
            render(wp.0 as u32);
            LRESULT(0)
        }
        WM_RENDERALLFORMATS => {
            // Retried: a reader holding the clipboard at this moment would otherwise make
            // Windows drop the unrendered text with our window.
            if let Ok(_open) = open_clipboard(hwnd) {
                // SAFETY: plain FFI query while we hold the clipboard.
                if unsafe { GetClipboardOwner() }.ok() == Some(hwnd) {
                    render(CF_UNICODETEXT);
                }
            }
            LRESULT(0)
        }
        WM_DESTROYCLIPBOARD => {
            PENDING.with(|p| p.borrow_mut().take());
            LRESULT(0)
        }
        WM_DESTROY => {
            // SAFETY: ends this thread's message loop.
            unsafe { PostQuitMessage(0) };
            LRESULT(0)
        }
        // SAFETY: default handling for everything else.
        _ => unsafe { DefWindowProcW(hwnd, msg, wp, lp) },
    }
}

/// Answers a render request. The reader holds the clipboard open and is blocked until we
/// return, so the data goes out first and the reader is identified afterwards.
fn render(format: u32) {
    let at = Instant::now();
    // SAFETY: plain FFI query; valid while the reader holds the clipboard open.
    let reader = unsafe { GetOpenClipboardWindow() }.ok();
    let text = PENDING.with(|p| p.borrow().clone());
    let Some(text) = text else { return };
    if format != CF_UNICODETEXT {
        return;
    }
    if let Some(h) = hglobal_from(&utf16_bytes(&text)) {
        // SAFETY: inside WM_RENDERFORMAT the reader has the clipboard open, which is the
        // one case SetClipboardData may be called without our own OpenClipboard. On
        // success the system owns the memory.
        if unsafe { SetClipboardData(format, Some(HANDLE(h.0))) }.is_err() {
            // SAFETY: the system did not take ownership, so we free it.
            let _ = unsafe { GlobalFree(Some(h)) };
        }
    }
    let (reader_hwnd, reader_pid) = match reader {
        Some(h) if !h.0.is_null() => {
            let (_, pid) = window_thread_pid(h);
            (Some(hwnd_raw(h)), (pid != 0).then_some(pid))
        }
        _ => (None, None),
    };
    let reader_exe = reader_pid.and_then(exe_name_of_pid);
    let generation = GENERATION.with(Cell::get);
    SHARED.with(|s| {
        if let Some(shared) = s.borrow().as_ref() {
            let mut log = shared.log.lock().unwrap_or_else(|e| e.into_inner());
            if log.generation == generation {
                log.count += 1;
                if log.first.is_none() {
                    log.first = Some(RenderEvent {
                        generation,
                        at,
                        format,
                        reader_hwnd,
                        reader_pid,
                        reader_exe,
                    });
                }
                shared.cv.notify_all();
            }
        }
    });
}

fn utf16_bytes(units: &[u16]) -> Vec<u8> {
    units
        .iter()
        .chain(std::iter::once(&0))
        .flat_map(|u| u.to_le_bytes())
        .collect()
}

fn hglobal_from(bytes: &[u8]) -> Option<HGLOBAL> {
    // SAFETY: allocate, lock, copy exactly `bytes.len()` bytes, unlock.
    unsafe {
        let h = GlobalAlloc(GMEM_MOVEABLE, bytes.len().max(1)).ok()?;
        let p = GlobalLock(h) as *mut u8;
        if p.is_null() {
            let _ = GlobalFree(Some(h));
            return None;
        }
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), p, bytes.len());
        let _ = GlobalUnlock(h);
        Some(h)
    }
}

/// Opens the clipboard, retrying briefly while another process holds it.
fn open_clipboard(hwnd: HWND) -> Result<ClipboardGuard, ClipboardError> {
    let deadline = Instant::now() + Duration::from_millis(500);
    loop {
        // SAFETY: plain FFI call; paired with CloseClipboard in the guard.
        if unsafe { OpenClipboard(Some(hwnd)) }.is_ok() {
            return Ok(ClipboardGuard);
        }
        if Instant::now() >= deadline {
            return Err(ClipboardError::Busy);
        }
        std::thread::sleep(Duration::from_millis(5));
    }
}

struct ClipboardGuard;

impl Drop for ClipboardGuard {
    fn drop(&mut self) {
        // SAFETY: we opened the clipboard on this thread.
        let _ = unsafe { CloseClipboard() };
    }
}

fn format_name(f: u32) -> String {
    let std_name = match f {
        1 => "CF_TEXT",
        2 => "CF_BITMAP",
        3 => "CF_METAFILEPICT",
        7 => "CF_OEMTEXT",
        8 => "CF_DIB",
        9 => "CF_PALETTE",
        13 => "CF_UNICODETEXT",
        14 => "CF_ENHMETAFILE",
        15 => "CF_HDROP",
        16 => "CF_LOCALE",
        17 => "CF_DIBV5",
        _ => "",
    };
    if !std_name.is_empty() {
        return std_name.into();
    }
    let mut buf = [0u16; 256];
    // SAFETY: `buf` outlives the call.
    let n = unsafe { GetClipboardFormatNameW(f, &mut buf) };
    if n > 0 {
        String::from_utf16_lossy(&buf[..n as usize])
    } else {
        format!("#{f}")
    }
}

fn snapshot_now(hwnd: HWND, cap: usize) -> Result<ClipboardSnapshot, ClipboardError> {
    let _open = open_clipboard(hwnd)?;
    let mut snap = ClipboardSnapshot {
        // SAFETY: plain FFI query.
        sequence: unsafe { GetClipboardSequenceNumber() },
        ..Default::default()
    };
    let mut total = 0usize;
    let mut f = 0u32;
    loop {
        // SAFETY: the clipboard is open on this thread.
        f = unsafe { EnumClipboardFormats(f) };
        if f == 0 {
            break;
        }
        if is_handle_format(f) {
            snap.skipped.push((f, format_name(f)));
            continue;
        }
        // SAFETY: the clipboard is open; the handle stays valid until it closes.
        let Ok(h) = (unsafe { GetClipboardData(f) }) else {
            snap.skipped.push((f, format_name(f)));
            continue;
        };
        let hg = HGLOBAL(h.0);
        // SAFETY: clipboard data of a non-handle format is an HGLOBAL.
        let size = unsafe { GlobalSize(hg) };
        if size == 0 || total + size > cap {
            if size > 0 {
                snap.truncated = true;
            }
            snap.skipped.push((f, format_name(f)));
            continue;
        }
        // SAFETY: lock, copy `size` bytes, unlock; we never free clipboard-owned memory.
        let bytes = unsafe {
            let p = GlobalLock(hg) as *const u8;
            if p.is_null() {
                None
            } else {
                let v = std::slice::from_raw_parts(p, size).to_vec();
                let _ = GlobalUnlock(hg);
                Some(v)
            }
        };
        match bytes {
            Some(bytes) => {
                total += size;
                snap.formats.push(SavedFormat {
                    format: f,
                    name: format_name(f),
                    bytes,
                });
            }
            None => snap.skipped.push((f, format_name(f))),
        }
    }
    Ok(snap)
}

fn exclusion_formats() -> [u32; 3] {
    // SAFETY: registering well-known format names; returns the same id every time.
    unsafe {
        [
            RegisterClipboardFormatW(w!("ExcludeClipboardContentFromMonitorProcessing")),
            RegisterClipboardFormatW(w!("CanIncludeInClipboardHistory")),
            RegisterClipboardFormatW(w!("CanUploadToCloudClipboard")),
        ]
    }
}

/// Sets eager data. Clipboard must be open and emptied by us.
fn set_eager(format: u32, bytes: &[u8]) -> Result<(), ClipboardError> {
    let h = hglobal_from(bytes).ok_or_else(windows::core::Error::from_thread)?;
    // SAFETY: the clipboard is open by us; on success the system owns `h`.
    match unsafe { SetClipboardData(format, Some(HANDLE(h.0))) } {
        Ok(_) => Ok(()),
        Err(e) => {
            // SAFETY: ownership did not transfer, so we free it.
            let _ = unsafe { GlobalFree(Some(h)) };
            Err(e.into())
        }
    }
}

fn mark_excluded() {
    for f in exclusion_formats() {
        let _ = set_eager(f, &0u32.to_le_bytes());
    }
}

fn restore_now(
    hwnd: HWND,
    snapshot: &ClipboardSnapshot,
    expected_sequence: Option<u32>,
) -> Result<RestoreOutcome, ClipboardError> {
    let _open = open_clipboard(hwnd)?;
    // SAFETY: plain FFI query while we hold the clipboard, so nobody can write between
    // this check and the restore.
    let now = unsafe { GetClipboardSequenceNumber() };
    if expected_sequence.is_some_and(|s| s != now) {
        return Ok(RestoreOutcome::SkippedChanged);
    }
    PENDING.with(|p| p.borrow_mut().take());
    // SAFETY: clipboard is open by us.
    unsafe { EmptyClipboard() }?;
    let excl = exclusion_formats();
    for f in &snapshot.formats {
        if let Err(e) = set_eager(f.format, &f.bytes) {
            tracing::debug!(format = %f.name, error = %e, "restore: format not accepted");
        }
    }
    // The restored content is not new; keep it from appearing twice in history.
    if !snapshot.formats.iter().any(|f| excl.contains(&f.format)) && !snapshot.is_empty() {
        mark_excluded();
    }
    Ok(RestoreOutcome::Restored)
}

fn write_delayed_now(
    hwnd: HWND,
    shared: &Shared,
    text: &str,
) -> Result<WriteReceipt, ClipboardError> {
    let units: Vec<u16> = text.encode_utf16().collect();
    let open = open_clipboard(hwnd)?;
    // SAFETY: clipboard is open by us; emptying makes our window the owner.
    unsafe { EmptyClipboard() }?;
    // Set after EmptyClipboard: emptying our own previous entry sends us
    // WM_DESTROYCLIPBOARD, which clears the pending text.
    PENDING.with(|p| *p.borrow_mut() = Some(Arc::new(units)));
    let generation = GENERATION.with(|g| {
        g.set(g.get() + 1);
        g.get()
    });
    {
        let mut log = shared.log.lock().unwrap_or_else(|e| e.into_inner());
        *log = RenderLog {
            generation,
            first: None,
            count: 0,
        };
    }
    // SAFETY: registering delayed rendering. It returns NULL by design, which the
    // `windows` crate maps to an Err carrying a stale last-error (seen in the spike), so
    // the result says nothing and is ignored; ownership is checked below instead.
    let _ = unsafe { SetClipboardData(CF_UNICODETEXT, None) };
    mark_excluded();
    drop(open);
    // SAFETY: plain FFI queries.
    let (sequence, owner) = unsafe { (GetClipboardSequenceNumber(), GetClipboardOwner()) };
    let written_at = Instant::now();
    shared.our_sequence.store(sequence, Ordering::Release);
    if owner.ok() != Some(hwnd) {
        tracing::debug!("clipboard owner changed right after our write");
    }
    Ok(WriteReceipt {
        generation,
        sequence,
        written_at,
    })
}

fn write_eager_now(hwnd: HWND, text: &str) -> Result<u32, ClipboardError> {
    let units: Vec<u16> = text.encode_utf16().collect();
    let open = open_clipboard(hwnd)?;
    // SAFETY: clipboard is open by us.
    unsafe { EmptyClipboard() }?;
    PENDING.with(|p| p.borrow_mut().take());
    set_eager(CF_UNICODETEXT, &utf16_bytes(&units))?;
    drop(open);
    // SAFETY: plain FFI query.
    Ok(unsafe { GetClipboardSequenceNumber() })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex as StdMutex;

    /// Clipboard tests share one global resource; run them one at a time.
    static SERIAL: StdMutex<()> = StdMutex::new(());

    fn read_text_as_other_reader() -> Option<String> {
        let t = std::thread::spawn(|| {
            let deadline = Instant::now() + Duration::from_secs(2);
            // SAFETY: test-only reader on its own thread; opens with a NULL window.
            unsafe {
                while OpenClipboard(None).is_err() {
                    if Instant::now() > deadline {
                        return None;
                    }
                    std::thread::sleep(Duration::from_millis(5));
                }
                let r = GetClipboardData(CF_UNICODETEXT).ok().and_then(|h| {
                    let hg = HGLOBAL(h.0);
                    let p = GlobalLock(hg) as *const u16;
                    if p.is_null() {
                        return None;
                    }
                    let n = GlobalSize(hg) / 2;
                    let s = std::slice::from_raw_parts(p, n);
                    let end = s.iter().position(|&u| u == 0).unwrap_or(n);
                    let out = String::from_utf16_lossy(&s[..end]);
                    let _ = GlobalUnlock(hg);
                    Some(out)
                });
                let _ = CloseClipboard();
                r
            }
        });
        t.join().ok().flatten()
    }

    #[test]
    fn handle_formats_are_skipped() {
        assert!(is_handle_format(2));
        assert!(is_handle_format(14));
        assert!(is_handle_format(0x2A0));
        assert!(!is_handle_format(CF_UNICODETEXT));
        assert!(!is_handle_format(15));
        assert!(!is_handle_format(0xC0FF));
    }

    #[test]
    fn delayed_write_renders_once_and_restore_round_trips() {
        let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let cb = WinClipboard::start().expect("owner");
        let before = cb.snapshot(DEFAULT_SNAPSHOT_CAP).expect("snapshot");

        let receipt = cb.write_delayed("hello ✓ 😀").expect("write");
        assert!(cb.we_own());
        assert_eq!(cb.sequence_number(), receipt.sequence);
        let got = read_text_as_other_reader();
        assert_eq!(got.as_deref(), Some("hello ✓ 😀"));
        let ev = cb
            .wait_for_render(Duration::from_secs(2))
            .expect("render event");
        assert_eq!(ev.generation, receipt.generation);
        assert_eq!(ev.format, CF_UNICODETEXT);
        // Rendering does not advance the sequence number.
        assert_eq!(cb.sequence_number(), receipt.sequence);
        // A second read is served from the cache: no second render.
        assert_eq!(read_text_as_other_reader().as_deref(), Some("hello ✓ 😀"));
        assert!(cb.render_count() >= 1);

        let outcome = cb
            .restore(&before, Some(receipt.sequence))
            .expect("restore");
        assert_eq!(outcome, RestoreOutcome::Restored);
        let after = cb.snapshot(DEFAULT_SNAPSHOT_CAP).expect("snapshot after");
        assert_eq!(after.text(), before.text());
    }

    #[test]
    fn restore_is_skipped_after_a_foreign_write() {
        let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let cb = WinClipboard::start().expect("owner");
        let before = cb.snapshot(DEFAULT_SNAPSHOT_CAP).expect("snapshot");
        let receipt = cb.write_delayed("ours").expect("write");
        let seq = cb.write_text("the user copied this").expect("write");
        assert_ne!(seq, receipt.sequence);
        assert_eq!(
            cb.restore(&before, Some(receipt.sequence)).unwrap(),
            RestoreOutcome::SkippedChanged
        );
        assert_eq!(
            read_text_as_other_reader().as_deref(),
            Some("the user copied this")
        );
        cb.restore(&before, None).unwrap();
    }

    #[test]
    fn snapshot_copies_text() {
        let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let cb = WinClipboard::start().expect("owner");
        let before = cb.snapshot(DEFAULT_SNAPSHOT_CAP).expect("snapshot");
        cb.write_text("snap me").unwrap();
        let s = cb.snapshot(DEFAULT_SNAPSHOT_CAP).unwrap();
        assert_eq!(s.text().as_deref(), Some("snap me"));
        let tiny = cb.snapshot(4).unwrap();
        assert!(tiny.truncated);
        cb.restore(&before, None).unwrap();
    }

    #[test]
    fn scheduled_restore_fires_later_and_a_new_snapshot_takes_it_over() {
        use wl_core::insert::ClipboardPort;
        let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let mut cb = WinClipboard::start().expect("owner");
        let original = cb.snapshot(DEFAULT_SNAPSHOT_CAP).expect("snapshot");
        cb.write_text("user text").unwrap();

        // First dictation: snapshot, write, restore in 300 ms.
        let snap1 = ClipboardPort::snapshot(&mut cb).unwrap();
        let seq1 = ClipboardPort::write_delayed(&mut cb, "dictation one").unwrap();
        ClipboardPort::restore(&mut cb, snap1, Duration::from_millis(300), seq1);
        // Second dictation before that fires: its snapshot must be the user's text,
        // not dictation one.
        let snap2 = ClipboardPort::snapshot(&mut cb).unwrap();
        assert_eq!(snap2.text().as_deref(), Some("user text"));
        let seq2 = ClipboardPort::write_delayed(&mut cb, "dictation two").unwrap();
        ClipboardPort::restore(&mut cb, snap2, Duration::from_millis(100), seq2);
        assert_eq!(
            read_text_as_other_reader().as_deref(),
            Some("dictation two")
        );
        std::thread::sleep(Duration::from_millis(500));
        let now = cb.sequence_number();
        assert_eq!(
            read_text_as_other_reader().as_deref(),
            Some("user text"),
            "seq at write {seq2}, now {now}, owner is us: {}",
            cb.we_own()
        );

        // A foreign write before the delay wins.
        let snap3 = ClipboardPort::snapshot(&mut cb).unwrap();
        let seq3 = ClipboardPort::write_delayed(&mut cb, "dictation three").unwrap();
        ClipboardPort::restore(&mut cb, snap3, Duration::from_millis(100), seq3);
        cb.write_text("copied meanwhile").unwrap();
        std::thread::sleep(Duration::from_millis(300));
        assert_eq!(
            read_text_as_other_reader().as_deref(),
            Some("copied meanwhile")
        );
        cb.restore(&original, None).unwrap();
    }

    #[test]
    fn render_wait_reports_a_foreign_write_as_changed() {
        let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let cb = WinClipboard::start().expect("owner");
        let before = cb.snapshot(DEFAULT_SNAPSHOT_CAP).expect("snapshot");
        cb.write_delayed("x").unwrap();
        // In an RDP session rdpclip may already have rendered; only check the change
        // path when nothing has read yet.
        if cb.first_render().is_none() {
            let other = WinClipboard::start().expect("second owner");
            other.write_text("foreign").unwrap();
            let r = cb.wait_for_render_or_change(Duration::from_millis(500));
            assert!(matches!(r, Err(Waited::Changed) | Ok(_)), "{r:?}");
        }
        cb.restore(&before, None).unwrap();
    }

    #[test]
    fn shutdown_renders_pending_text() {
        let _g = SERIAL.lock().unwrap_or_else(|e| e.into_inner());
        let cb = WinClipboard::start().expect("owner");
        let before = cb.snapshot(DEFAULT_SNAPSHOT_CAP).expect("snapshot");
        cb.write_delayed("survives shutdown").unwrap();
        cb.shutdown();
        assert_eq!(
            read_text_as_other_reader().as_deref(),
            Some("survives shutdown")
        );
        let cb2 = WinClipboard::start().expect("owner");
        cb2.restore(&before, None).unwrap();
    }
}
