//! The owner gets a thread of its own because a reader blocks inside `GetClipboardData`
//! until we answer `WM_RENDERFORMAT`, and snapshotting a foreign clipboard can block for
//! seconds while that app renders. Windows asks for a render once per write (again for
//! each reader that raced the first) and serves later readers the cached copy silently, so a render
//! identifies only the first reader.

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
    KillTimer, MSG, MWMO_INPUTAVAILABLE, MsgWaitForMultipleObjectsEx, PM_NOREMOVE,
    PM_QS_SENDMESSAGE, PeekMessageW, PostMessageW, PostQuitMessage, QS_SENDMESSAGE, RegisterClassW,
    SetTimer, TranslateMessage, WINDOW_EX_STYLE, WINDOW_STYLE, WM_APP, WM_DESTROY,
    WM_DESTROYCLIPBOARD, WM_RENDERALLFORMATS, WM_RENDERFORMAT, WM_TIMER, WNDCLASSW,
};
use windows::core::{PCWSTR, w};

use hush_core::insert::{InsertError, Reader, RenderWait};

use crate::util::{exe_name_of_pid, hwnd_from, hwnd_raw, window_thread_pid};

pub const CF_TEXT: u32 = 1;
pub const CF_UNICODETEXT: u32 = 13;

/// Not `HGLOBAL`s, so they cannot be byte-copied; images still survive through the
/// `CF_DIB` Windows synthesises from a bitmap.
fn is_handle_format(f: u32) -> bool {
    matches!(
        f,
        2 /* CF_BITMAP */ | 3 /* CF_METAFILEPICT */ | 9 /* CF_PALETTE */
        | 14 /* CF_ENHMETAFILE */ | 0x80 /* CF_OWNERDISPLAY */ | 0x82 /* CF_DSPBITMAP */
        | 0x83 /* CF_DSPMETAFILEPICT */ | 0x8E /* CF_DSPENHMETAFILE */
    ) || (0x200..=0x3FF).contains(&f)
}

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SavedFormat {
    pub format: u32,
    pub name: String,
    pub bytes: Vec<u8>,
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ClipboardSnapshot {
    pub formats: Vec<SavedFormat>,
    pub skipped: Vec<(u32, String)>,
    pub sequence: u32,
    pub truncated: bool,
}

impl ClipboardSnapshot {
    pub fn is_empty(&self) -> bool {
        self.formats.is_empty()
    }

    pub fn text(&self) -> Option<String> {
        let f = self.formats.iter().find(|f| f.format == CF_UNICODETEXT)?;
        let (pairs, _) = f.bytes.as_chunks::<2>();
        let units: Vec<u16> = pairs
            .iter()
            .map(|c| u16::from_le_bytes(*c))
            .take_while(|&u| u != 0)
            .collect();
        Some(String::from_utf16_lossy(&units))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct WriteReceipt {
    pub generation: u64,
    pub sequence: u32,
    pub written_at: Instant,
}

/// Someone read our delayed text.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RenderEvent {
    pub generation: u64,
    pub at: Instant,
    pub format: u32,
    /// `None` when the reader opened the clipboard with a NULL window (Windows Terminal
    /// does).
    pub reader_hwnd: Option<isize>,
    pub reader_pid: Option<u32>,
    pub reader_exe: Option<String>,
}

impl RenderEvent {
    /// Negative means the render came before the chord.
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
    SkippedChanged,
}

#[derive(Default)]
struct RenderLog {
    generation: u64,
    first: Option<RenderEvent>,
    count: u32,
}

struct Shared {
    log: Mutex<RenderLog>,
    cv: Condvar,
    /// The sequence number our last delayed write returned.
    written_sequence: AtomicU32,
    /// The same, advanced by our own repeat renders of that write. Answering a second
    /// request replaces the data, which advances the number although nobody else wrote.
    our_sequence: AtomicU32,
}

impl Shared {
    /// The clipboard's sequence number, with advances made only by our own renders folded
    /// back into the number of the write they rendered.
    fn effective_sequence(&self) -> u32 {
        // SAFETY: plain FFI query, callable from any thread.
        let now = unsafe { GetClipboardSequenceNumber() };
        if now == self.our_sequence.load(Ordering::Acquire) {
            self.written_sequence.load(Ordering::Acquire)
        } else {
            now
        }
    }
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

/// A snapshot taken before this fires must return this snapshot instead: the clipboard
/// still holds our previous dictation, not the user's contents.
struct Scheduled {
    snapshot: Box<ClipboardSnapshot>,
    if_sequence: u32,
}

const RESTORE_TIMER: usize = 0x5752;

const WM_CB_COMMAND: u32 = WM_APP + 0x20;

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
    pub fn start() -> Result<Self, ClipboardError> {
        let shared = Arc::new(Shared {
            log: Mutex::new(RenderLog::default()),
            cv: Condvar::new(),
            written_sequence: AtomicU32::new(0),
            our_sequence: AtomicU32::new(0),
        });
        let (tx, rx) = mpsc::channel();
        let (ready_tx, ready_rx) = mpsc::channel();
        let thread_shared = shared.clone();
        let join = std::thread::Builder::new()
            .name("hush-clipboard".into())
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

    /// `cap` bounds the total bytes copied.
    pub fn snapshot(&self, cap: usize) -> Result<ClipboardSnapshot, ClipboardError> {
        self.call(|reply| Command::Snapshot { cap, reply })
    }

    /// With `expected_sequence`, restores only if nobody wrote since.
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

    /// Marked so clipboard history and cloud sync skip it.
    pub fn write_delayed(&self, text: &str) -> Result<WriteReceipt, ClipboardError> {
        let text = text.to_string();
        self.call(|reply| Command::WriteDelayed { text, reply })
    }

    /// Returns the sequence number after the write.
    pub fn write_text(&self, text: &str) -> Result<u32, ClipboardError> {
        let text = text.to_string();
        self.call(|reply| Command::WriteEager { text, reply })
    }

    /// Returns immediately; the restore happens only if the sequence number is still
    /// `if_sequence` by then.
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

    /// Our own renders do not advance it: until someone else writes, it stays what our
    /// last write returned.
    pub fn sequence_number(&self) -> u32 {
        self.inner.shared.effective_sequence()
    }

    /// Also ends early when a foreign write replaces ours.
    pub fn wait_for_render_or_change(&self, timeout: Duration) -> Result<RenderEvent, Waited> {
        let deadline = Instant::now() + timeout;
        let ours = self.inner.shared.written_sequence.load(Ordering::Acquire);
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

    pub fn first_render(&self) -> Option<RenderEvent> {
        self.wait_for_render(Duration::ZERO)
    }

    /// Windows asks once per write, plus once for each reader that raced the first; a
    /// reader after the first render is served the cached copy and adds none.
    pub fn render_count(&self) -> u32 {
        self.inner
            .shared
            .log
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .count
    }

    pub fn we_own(&self) -> bool {
        // SAFETY: plain FFI query.
        unsafe { GetClipboardOwner() }
            .ok()
            .is_some_and(|h| hwnd_raw(h) == self.inner.hwnd)
    }

    /// Unrendered text is rendered first so it is not lost.
    pub fn shutdown(&self) {
        self.inner.shutdown();
    }
}

impl hush_core::insert::ClipboardPort for WinClipboard {
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

thread_local! {
    static PENDING: RefCell<Option<Arc<Vec<u16>>>> = const { RefCell::new(None) };
    static GENERATION: Cell<u64> = const { Cell::new(0) };
    /// The generation whose text is on the clipboard; 0 (never a write) when none is.
    static RENDERED: Cell<u64> = const { Cell::new(0) };
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
    let run_scheduled =
        |s: Scheduled| match restore_now(hwnd, &shared, &s.snapshot, Some(s.if_sequence)) {
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
        // Outside any window procedure, so our own clipboard calls can re-enter the render
        // handler without a borrow conflict.
        while !quitting && let Ok(cmd) = rx.try_recv() {
            match cmd {
                Command::Snapshot { cap, reply } => {
                    let now = shared.effective_sequence();
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
                    let _ =
                        reply.try_send(restore_now(hwnd, &shared, &snapshot, expected_sequence));
                }
                Command::WriteDelayed { text, reply } => {
                    let _ = reply.try_send(write_delayed_now(hwnd, &shared, &text));
                }
                Command::WriteEager { text, reply } => {
                    let _ = reply.try_send(write_eager_now(hwnd, &text));
                }
                Command::Shutdown => {
                    quitting = true;
                    if let Some(s) = scheduled.take() {
                        run_scheduled(s);
                    }
                    // SAFETY: our own window on its thread; WM_RENDERALLFORMATS arrives inside.
                    let _ = unsafe { DestroyWindow(hwnd) };
                }
            }
        }
    }
    SHARED.with(|s| *s.borrow_mut() = None);
}

fn create_owner_window() -> windows::core::Result<HWND> {
    // SAFETY: a 'static window procedure and a message-only window owned by this thread.
    unsafe {
        let inst = GetModuleHandleW(PCWSTR::null())?;
        let class = w!("hush-clipboard-owner");
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
            w!("hush clipboard"),
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
                // Text already rendered is on the clipboard and outlives our window;
                // rendering it again would only advance the sequence number.
                let rendered = RENDERED.with(Cell::get) == GENERATION.with(Cell::get);
                // SAFETY: plain FFI query while we hold the clipboard.
                if !rendered && unsafe { GetClipboardOwner() }.ok() == Some(hwnd) {
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

/// The reader is blocked until we return, so the data goes out before the reader is
/// identified.
fn render(format: u32) {
    let at = Instant::now();
    // SAFETY: plain FFI query; valid while the reader holds the clipboard open.
    let reader = unsafe { GetOpenClipboardWindow() }.ok();
    let text = PENDING.with(|p| p.borrow().clone());
    let Some(text) = text else { return };
    if format != CF_UNICODETEXT {
        return;
    }
    let generation = GENERATION.with(Cell::get);
    // A repeat request (readers raced for the first) must still be answered: measured, the
    // racing reader gets nothing otherwise. The answer replaces the data and advances the
    // sequence number, so that advance is recorded as ours.
    if let Some(h) = hglobal_from(&utf16_bytes(&text)) {
        // SAFETY: WM_RENDERFORMAT permits this without our own open; success hands over `h`.
        if unsafe { SetClipboardData(format, Some(HANDLE(h.0))) }.is_ok() {
            RENDERED.with(|r| r.set(generation));
            SHARED.with(|s| {
                if let Some(shared) = s.borrow().as_ref() {
                    // SAFETY: plain FFI query; the reader still holds the clipboard, so no
                    // foreign write can land between our write and this read.
                    let now = unsafe { GetClipboardSequenceNumber() };
                    shared.our_sequence.store(now, Ordering::Release);
                }
            });
        } else {
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
        // Not a plain sleep: the holder may be a reader blocked until we answer its
        // WM_RENDERFORMAT, which only arrives while this thread takes sent messages.
        // Posted messages stay queued, so no command runs re-entrantly.
        let mut msg = MSG::default();
        // SAFETY: `msg` is a valid out-parameter; only sent messages are dispatched.
        unsafe {
            let _ = PeekMessageW(&mut msg, None, 0, 0, PM_NOREMOVE | PM_QS_SENDMESSAGE);
            MsgWaitForMultipleObjectsEx(None, 5, QS_SENDMESSAGE, MWMO_INPUTAVAILABLE);
        }
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

/// The clipboard must be open and emptied by us.
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
    shared: &Shared,
    snapshot: &ClipboardSnapshot,
    expected_sequence: Option<u32>,
) -> Result<RestoreOutcome, ClipboardError> {
    let _open = open_clipboard(hwnd)?;
    // Checked while we hold the clipboard, so a user's copy cannot land between the check
    // and the restore.
    let now = shared.effective_sequence();
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
    // Only CF_UNICODETEXT: Windows synthesises the other text formats through our render,
    // and rich formats would carry styling into the target. Delayed registration returns
    // NULL by design, which `windows` maps to an Err with a stale last-error, so it is
    // ignored.
    // SAFETY: the clipboard is open and emptied by us.
    let _ = unsafe { SetClipboardData(CF_UNICODETEXT, None) };
    mark_excluded();
    drop(open);
    // SAFETY: plain FFI queries.
    let (sequence, owner) = unsafe { (GetClipboardSequenceNumber(), GetClipboardOwner()) };
    let written_at = Instant::now();
    shared.written_sequence.store(sequence, Ordering::Release);
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

    /// Held by every test that touches the real clipboard, whatever the thread count.
    static SERIAL: StdMutex<()> = StdMutex::new(());

    fn serial() -> std::sync::MutexGuard<'static, ()> {
        SERIAL.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Another process holding the clipboard past the retry bound is the environment, not
    /// a defect, so the test reports SKIPPED and returns.
    macro_rules! live {
        ($e:expr) => {
            match $e {
                Ok(v) => v,
                Err(ClipboardError::Busy) => {
                    eprintln!(
                        "SKIPPED: clipboard held by another process at `{}`",
                        stringify!($e)
                    );
                    return;
                }
                Err(e) => panic!("{}: {e}", stringify!($e)),
            }
        };
    }

    /// Puts the clipboard back however the test ends, through an owner of its own so it
    /// still works after the test's owner has shut down.
    struct RestoreOnDrop(ClipboardSnapshot);

    impl Drop for RestoreOnDrop {
        fn drop(&mut self) {
            match WinClipboard::start().map(|cb| cb.restore(&self.0, None)) {
                Ok(Ok(_)) => {}
                other => eprintln!("clipboard not restored after the test: {other:?}"),
            }
        }
    }

    fn read_text_as_other_reader() -> Result<Option<String>, ClipboardError> {
        let t = std::thread::spawn(|| {
            let deadline = Instant::now() + Duration::from_secs(2);
            // SAFETY: test-only reader on its own thread; opens with a NULL window.
            unsafe {
                while OpenClipboard(None).is_err() {
                    if Instant::now() > deadline {
                        return Err(ClipboardError::Busy);
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
                Ok(r)
            }
        });
        t.join().expect("reader thread")
    }

    /// Polls until `want` is on the clipboard or `within` has passed; returns the last read.
    fn wait_for_text(want: &str, within: Duration) -> Result<Option<String>, ClipboardError> {
        let deadline = Instant::now() + within;
        loop {
            let got = read_text_as_other_reader()?;
            if got.as_deref() == Some(want) || Instant::now() >= deadline {
                return Ok(got);
            }
            std::thread::sleep(Duration::from_millis(20));
        }
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
        const TEXT: &str = "hello ✓ 😀";
        let _g = serial();
        let cb = WinClipboard::start().expect("owner");
        let before = live!(cb.snapshot(DEFAULT_SNAPSHOT_CAP));
        let _restore = RestoreOnDrop(before.clone());

        let receipt = live!(cb.write_delayed(TEXT));
        assert!(cb.we_own());
        assert_eq!(cb.sequence_number(), receipt.sequence);
        assert_eq!(live!(read_text_as_other_reader()).as_deref(), Some(TEXT));
        let ev = cb
            .wait_for_render(Duration::from_secs(2))
            .expect("render event");
        assert_eq!(ev.generation, receipt.generation);
        assert_eq!(ev.format, CF_UNICODETEXT);
        // Not exactly 1: a clipboard monitor racing our reader makes Windows ask twice.
        let renders = cb.render_count();
        assert!(renders >= 1);
        // The conditional restore below relies on our renders leaving the number alone.
        assert_eq!(cb.sequence_number(), receipt.sequence);
        assert_eq!(live!(read_text_as_other_reader()).as_deref(), Some(TEXT));
        assert_eq!(
            cb.render_count(),
            renders,
            "a reader after the first render must get the cached copy"
        );
        assert_eq!(cb.sequence_number(), receipt.sequence);

        let outcome = live!(cb.restore(&before, Some(receipt.sequence)));
        assert_eq!(outcome, RestoreOutcome::Restored);
        assert!(
            cb.sequence_number() > receipt.sequence,
            "restore wrote nothing"
        );
        let after = live!(cb.snapshot(DEFAULT_SNAPSHOT_CAP));
        assert_eq!(after.text(), before.text());
    }

    /// What Windows does when readers race for a fresh write: a second WM_RENDERFORMAT for
    /// text already rendered. Answering it replaces the data and advances the raw number.
    #[test]
    fn a_repeat_render_is_not_mistaken_for_a_foreign_write() {
        use windows::Win32::UI::WindowsAndMessaging::SendMessageW;
        let _g = serial();
        let cb = WinClipboard::start().expect("owner");
        let before = live!(cb.snapshot(DEFAULT_SNAPSHOT_CAP));
        let _restore = RestoreOnDrop(before.clone());
        let receipt = live!(cb.write_delayed("asked twice"));
        assert_eq!(
            live!(read_text_as_other_reader()).as_deref(),
            Some("asked twice")
        );
        let hwnd = cb.inner.hwnd;
        let asked = std::thread::spawn(move || {
            // SAFETY: test-only reader thread; the owner answers WM_RENDERFORMAT while we
            // hold the clipboard open, as it would for a real racing reader.
            unsafe {
                if OpenClipboard(None).is_err() {
                    return false;
                }
                SendMessageW(
                    hwnd_from(hwnd),
                    WM_RENDERFORMAT,
                    Some(WPARAM(CF_UNICODETEXT as usize)),
                    None,
                );
                let _ = CloseClipboard();
                true
            }
        })
        .join()
        .expect("reader");
        if !asked {
            eprintln!("SKIPPED: clipboard held by another process");
            return;
        }
        // SAFETY: plain FFI query.
        let raw = unsafe { GetClipboardSequenceNumber() };
        assert_ne!(raw, receipt.sequence, "the repeat render replaced nothing");
        assert_eq!(cb.sequence_number(), receipt.sequence);
        assert!(cb.wait_for_render_or_change(Duration::ZERO).is_ok());
        assert_eq!(
            live!(read_text_as_other_reader()).as_deref(),
            Some("asked twice")
        );
        assert_eq!(
            live!(cb.restore(&before, Some(receipt.sequence))),
            RestoreOutcome::Restored
        );
        assert_eq!(
            live!(cb.snapshot(DEFAULT_SNAPSHOT_CAP)).text(),
            before.text()
        );
    }

    #[test]
    fn restore_is_skipped_after_a_foreign_write() {
        let _g = serial();
        let cb = WinClipboard::start().expect("owner");
        let before = live!(cb.snapshot(DEFAULT_SNAPSHOT_CAP));
        let _restore = RestoreOnDrop(before.clone());
        let receipt = live!(cb.write_delayed("ours"));
        let seq = live!(cb.write_text("the user copied this"));
        assert_ne!(seq, receipt.sequence);
        assert_eq!(
            live!(cb.restore(&before, Some(receipt.sequence))),
            RestoreOutcome::SkippedChanged
        );
        assert_eq!(cb.sequence_number(), seq, "a refused restore wrote");
        assert_eq!(
            live!(read_text_as_other_reader()).as_deref(),
            Some("the user copied this")
        );
    }

    #[test]
    fn snapshot_copies_text() {
        let _g = serial();
        let cb = WinClipboard::start().expect("owner");
        let _restore = RestoreOnDrop(live!(cb.snapshot(DEFAULT_SNAPSHOT_CAP)));
        live!(cb.write_text("snap me"));
        let s = live!(cb.snapshot(DEFAULT_SNAPSHOT_CAP));
        assert_eq!(s.text().as_deref(), Some("snap me"));
        let tiny = live!(cb.snapshot(4));
        assert!(tiny.truncated);
    }

    #[test]
    fn scheduled_restore_fires_later_and_a_new_snapshot_takes_it_over() {
        // Long enough that the reads and writes between scheduling and firing finish first
        // even when another reader holds the clipboard for a while.
        const DELAY: Duration = Duration::from_millis(1000);
        let _g = serial();
        let cb = WinClipboard::start().expect("owner");
        let _restore = RestoreOnDrop(live!(cb.snapshot(DEFAULT_SNAPSHOT_CAP)));
        live!(cb.write_text("user text"));

        let snap1 = live!(cb.snapshot(DEFAULT_SNAPSHOT_CAP));
        let seq1 = live!(cb.write_delayed("dictation one")).sequence;
        cb.restore_later(snap1, DELAY, seq1);
        let snap2 = live!(cb.snapshot(DEFAULT_SNAPSHOT_CAP));
        assert_eq!(
            snap2.text().as_deref(),
            Some("user text"),
            "a snapshot taken while a restore is pending must take that restore over"
        );
        let seq2 = live!(cb.write_delayed("dictation two")).sequence;
        let scheduled_at = Instant::now();
        cb.restore_later(snap2, DELAY, seq2);
        assert_eq!(
            live!(read_text_as_other_reader()).as_deref(),
            Some("dictation two")
        );
        assert!(
            scheduled_at.elapsed() < DELAY,
            "too slow to observe the delay"
        );
        let got = live!(wait_for_text("user text", DELAY * 3));
        assert_eq!(
            got.as_deref(),
            Some("user text"),
            "seq at write {seq2}, now {}, owner is us: {}",
            cb.sequence_number(),
            cb.we_own()
        );

        let snap3 = live!(cb.snapshot(DEFAULT_SNAPSHOT_CAP));
        assert_eq!(snap3.text().as_deref(), Some("user text"));
        let seq3 = live!(cb.write_delayed("dictation three")).sequence;
        let scheduled_at = Instant::now();
        cb.restore_later(snap3, DELAY, seq3);
        let meanwhile = live!(cb.write_text("copied meanwhile"));
        std::thread::sleep(
            (scheduled_at + DELAY + Duration::from_millis(300))
                .saturating_duration_since(Instant::now()),
        );
        assert_eq!(
            live!(read_text_as_other_reader()).as_deref(),
            Some("copied meanwhile")
        );
        assert_eq!(cb.sequence_number(), meanwhile, "a refused restore wrote");
    }

    #[test]
    fn render_wait_reports_a_foreign_write_as_changed() {
        let _g = serial();
        let cb = WinClipboard::start().expect("owner");
        let _restore = RestoreOnDrop(live!(cb.snapshot(DEFAULT_SNAPSHOT_CAP)));
        live!(cb.write_delayed("x"));
        let other = WinClipboard::start().expect("second owner");
        live!(other.write_text("foreign"));
        // Once the foreign write has landed nothing can render ours any more, so this check
        // is final. A clipboard monitor (rdpclip under RDP) has often rendered before it,
        // and a render legitimately outranks the change.
        if cb.first_render().is_none() {
            let r = cb.wait_for_render_or_change(Duration::from_millis(500));
            assert!(matches!(r, Err(Waited::Changed)), "{r:?}");
        }
    }

    #[test]
    fn shutdown_renders_pending_text() {
        let _g = serial();
        let cb = WinClipboard::start().expect("owner");
        let _restore = RestoreOnDrop(live!(cb.snapshot(DEFAULT_SNAPSHOT_CAP)));
        live!(cb.write_delayed("survives shutdown"));
        cb.shutdown();
        assert_eq!(
            live!(read_text_as_other_reader()).as_deref(),
            Some("survives shutdown")
        );
    }
}
