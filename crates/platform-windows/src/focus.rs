//! Focus context at hotkey-down, and getting back to that window before inserting (D9).
//!
//! Why not UI Automation on the calling thread: UIA calls marshal into the target
//! process and block for seconds on a hung app, so the password query runs on its own
//! multithreaded-apartment worker and the caller waits at most `UIA_TIMEOUT`; a late
//! answer is dropped. Why "cannot open the token" counts as elevated: the only processes
//! a medium-integrity caller cannot query are the ones that would also ignore its input,
//! so assuming the opposite would send keystrokes into the void. Why the
//! `AttachThreadInput` dance is only a fallback: attaching input queues merges key state
//! with another thread for the duration and has caused stuck-modifier bugs in prior
//! art, so it is used only when a plain `SetForegroundWindow` was refused.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, SyncSender};
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use windows::Win32::Foundation::{CloseHandle, HANDLE, HWND};
use windows::Win32::Security::{GetTokenInformation, TOKEN_ELEVATION, TOKEN_QUERY, TokenElevation};
use windows::Win32::System::Com::{
    CLSCTX_INPROC_SERVER, COINIT_MULTITHREADED, CoCreateInstance, CoInitializeEx, CoUninitialize,
};
use windows::Win32::System::Threading::{
    AttachThreadInput, GetCurrentProcess, GetCurrentThreadId, OpenProcess, OpenProcessToken,
    PROCESS_QUERY_LIMITED_INFORMATION,
};
use windows::Win32::UI::Accessibility::{CUIAutomation, IUIAutomation};
use windows::Win32::UI::WindowsAndMessaging::{
    ASFW_ANY, AllowSetForegroundWindow, BringWindowToTop, GetForegroundWindow, GetSystemMetrics,
    GetWindowTextW, IsIconic, IsWindow, SM_REMOTESESSION, SW_RESTORE, SetForegroundWindow,
    ShowWindow,
};

use crate::util::{exe_name_of_pid, hwnd_from, hwnd_raw, window_thread_pid};

/// How long the password query may take before we give up and assume "not a password".
pub const UIA_TIMEOUT: Duration = Duration::from_millis(150);

/// Everything captured about the foreground window at hotkey-down.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FocusSnapshot {
    /// Raw HWND; zero when there was no foreground window (desktop switch, lock screen).
    pub hwnd: isize,
    pub thread_id: u32,
    pub pid: u32,
    /// Lower-cased executable file name.
    pub exe: Option<String>,
    pub title: Option<String>,
    /// Target runs elevated (or cannot be inspected) while we do not: the OS will drop
    /// our input and hide its keys from our hook.
    pub elevated: bool,
    pub self_elevated: bool,
    /// We are inside a Remote Desktop session (`SM_REMOTESESSION`).
    pub remote_session: bool,
    /// The focused element reports `IsPassword`. False on UIA timeout or error.
    pub is_password: bool,
    /// The UIA query did not answer in time; `is_password` is a guess.
    pub uia_timed_out: bool,
}

impl FocusSnapshot {
    /// The platform-neutral view the pipeline carries around.
    pub fn to_core(&self) -> wl_core::context::FocusContext {
        wl_core::context::FocusContext {
            window: self.hwnd as usize,
            exe: self.exe.clone(),
            title: self.title.clone(),
            elevated: self.elevated,
            is_password: self.is_password,
        }
    }
}

/// Focus capture and refocus. Cheap to clone; all clones share one UIA worker.
#[derive(Clone)]
pub struct WinFocus {
    uia: Arc<UiaWorker>,
}

impl Default for WinFocus {
    fn default() -> Self {
        Self::new()
    }
}

impl WinFocus {
    pub fn new() -> Self {
        Self {
            uia: Arc::new(UiaWorker::spawn()),
        }
    }

    /// Captures the foreground window. Never blocks longer than about `UIA_TIMEOUT`.
    pub fn capture(&self) -> FocusSnapshot {
        let mut snap = capture_without_uia();
        if snap.hwnd != 0 {
            match self.uia.is_password(UIA_TIMEOUT) {
                Some(p) => snap.is_password = p,
                None => {
                    snap.uia_timed_out = true;
                    tracing::warn!(
                        exe = snap.exe.as_deref().unwrap_or("?"),
                        "UI Automation did not answer within {:?}; assuming not a password field",
                        UIA_TIMEOUT
                    );
                }
            }
        }
        snap
    }

    /// True when the foreground window is still the captured one.
    pub fn is_still(&self, target: &FocusSnapshot) -> bool {
        // SAFETY: plain FFI query.
        target.hwnd != 0 && hwnd_raw(unsafe { GetForegroundWindow() }) == target.hwnd
    }

    /// Brings the captured window back to the foreground. Returns whether it is
    /// foreground afterwards.
    pub fn refocus(&self, target: &FocusSnapshot) -> bool {
        refocus_hwnd(target.hwnd)
    }
}

/// The part of the capture that needs no COM: handle, process, title, elevation.
pub fn capture_without_uia() -> FocusSnapshot {
    // SAFETY: plain FFI query.
    let hwnd = unsafe { GetForegroundWindow() };
    let self_elevated = self_elevated();
    let remote_session = is_remote_session();
    if hwnd.0.is_null() {
        return FocusSnapshot {
            self_elevated,
            remote_session,
            ..Default::default()
        };
    }
    let (thread_id, pid) = window_thread_pid(hwnd);
    let target_elevated = process_elevated(pid).unwrap_or(true);
    FocusSnapshot {
        hwnd: hwnd_raw(hwnd),
        thread_id,
        pid,
        exe: exe_name_of_pid(pid),
        title: window_title(hwnd),
        elevated: target_elevated && !self_elevated,
        self_elevated,
        remote_session,
        is_password: false,
        uia_timed_out: false,
    }
}

pub fn is_remote_session() -> bool {
    // SAFETY: plain FFI query.
    unsafe { GetSystemMetrics(SM_REMOTESESSION) != 0 }
}

fn window_title(hwnd: HWND) -> Option<String> {
    let mut buf = [0u16; 512];
    // SAFETY: `buf` outlives the call; for other processes' windows this reads the
    // cached caption and sends no message, so a hung target cannot block us.
    let n = unsafe { GetWindowTextW(hwnd, &mut buf) };
    (n > 0).then(|| String::from_utf16_lossy(&buf[..n as usize]))
}

/// `Some(elevated)` when the token could be read, `None` when the process or its token
/// cannot be opened.
pub fn process_elevated(pid: u32) -> Option<bool> {
    // SAFETY: plain FFI call; the handle is closed below.
    let process = unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false, pid) }.ok()?;
    let r = token_elevated(process);
    // SAFETY: `process` came from OpenProcess and is closed exactly once.
    let _ = unsafe { CloseHandle(process) };
    r
}

fn token_elevated(process: HANDLE) -> Option<bool> {
    let mut token = HANDLE::default();
    // SAFETY: `token` is a valid out-parameter; closed below on success.
    unsafe { OpenProcessToken(process, TOKEN_QUERY, &mut token) }.ok()?;
    let mut elevation = TOKEN_ELEVATION::default();
    let mut len = 0u32;
    // SAFETY: the buffer is a TOKEN_ELEVATION of exactly the size passed.
    let r = unsafe {
        GetTokenInformation(
            token,
            TokenElevation,
            Some(&mut elevation as *mut _ as *mut core::ffi::c_void),
            std::mem::size_of::<TOKEN_ELEVATION>() as u32,
            &mut len,
        )
    };
    // SAFETY: `token` came from OpenProcessToken and is closed exactly once.
    let _ = unsafe { CloseHandle(token) };
    r.ok()?;
    Some(elevation.TokenIsElevated != 0)
}

pub fn self_elevated() -> bool {
    static SELF: OnceLock<bool> = OnceLock::new();
    // SAFETY: the pseudo-handle of the current process needs no closing.
    *SELF.get_or_init(|| token_elevated(unsafe { GetCurrentProcess() }).unwrap_or(false))
}

/// Whether the foreground window belongs to a process that would hide its input from
/// our hook. Used by the hook watchdog so UIPI blindness is not mistaken for a dead hook.
pub(crate) fn foreground_is_elevated_over_us() -> bool {
    if self_elevated() {
        return false;
    }
    // SAFETY: plain FFI query.
    let hwnd = unsafe { GetForegroundWindow() };
    if hwnd.0.is_null() {
        return false;
    }
    let (_, pid) = window_thread_pid(hwnd);
    process_elevated(pid).unwrap_or(true)
}

/// [`WinFocus::refocus`] for a raw window handle.
pub fn refocus_raw(raw: isize) -> bool {
    refocus_hwnd(raw)
}

pub(crate) fn refocus_hwnd(raw: isize) -> bool {
    if raw == 0 {
        return false;
    }
    let target = hwnd_from(raw);
    // SAFETY: all calls below are plain FFI on a window handle that may be stale; stale
    // handles make them fail, not misbehave.
    unsafe {
        if !IsWindow(Some(target)).as_bool() {
            return false;
        }
        if GetForegroundWindow() == target {
            return true;
        }
        if IsIconic(target).as_bool() {
            let _ = ShowWindow(target, SW_RESTORE);
        }
        if SetForegroundWindow(target).as_bool() && GetForegroundWindow() == target {
            return true;
        }
        let fg = GetForegroundWindow();
        let (fg_tid, _) = window_thread_pid(fg);
        let me = GetCurrentThreadId();
        let attached = fg_tid != 0 && fg_tid != me && AttachThreadInput(me, fg_tid, true).as_bool();
        let _ = AllowSetForegroundWindow(ASFW_ANY);
        let _ = BringWindowToTop(target);
        let _ = SetForegroundWindow(target);
        if attached {
            let _ = AttachThreadInput(me, fg_tid, false);
        }
        GetForegroundWindow() == target
    }
}

impl wl_core::insert::FocusPort for WinFocus {
    fn foreground_window(&self) -> usize {
        // SAFETY: plain FFI query.
        hwnd_raw(unsafe { GetForegroundWindow() }) as usize
    }

    /// Refocuses only when nothing else took the foreground: no foreground window at
    /// all, or one of ours (a tray menu, a message box). A window the user moved to is
    /// never taken away from them.
    fn is_still(&mut self, target: &wl_core::context::FocusContext) -> bool {
        if target.window == 0 {
            return false;
        }
        let fg = self.foreground_window();
        if fg == target.window {
            return true;
        }
        let ours = fg == 0 || window_thread_pid(hwnd_from(fg as isize)).1 == std::process::id();
        ours && refocus_hwnd(target.window as isize)
    }

    fn is_remote_session(&self) -> bool {
        is_remote_session()
    }
}

// ------------------------------------------------------------------ UIA worker

struct UiaRequest {
    reply: SyncSender<bool>,
}

struct UiaWorker {
    tx: Option<mpsc::Sender<UiaRequest>>,
    /// A query that timed out may still be running; queuing behind it would only time
    /// out again.
    busy: Arc<AtomicBool>,
    join: Option<std::thread::JoinHandle<()>>,
}

impl UiaWorker {
    fn spawn() -> Self {
        let (tx, rx) = mpsc::channel::<UiaRequest>();
        let busy = Arc::new(AtomicBool::new(false));
        let busy_w = busy.clone();
        let join = std::thread::Builder::new()
            .name("wl-uia".into())
            .spawn(move || uia_thread(rx, busy_w))
            .ok();
        Self {
            tx: Some(tx),
            busy,
            join,
        }
    }

    /// `None` on timeout or when a previous query is still stuck.
    fn is_password(&self, timeout: Duration) -> Option<bool> {
        if self.busy.load(Ordering::Acquire) {
            return None;
        }
        let (reply, rx) = mpsc::sync_channel(1);
        self.tx.as_ref()?.send(UiaRequest { reply }).ok()?;
        rx.recv_timeout(timeout).ok()
    }
}

impl Drop for UiaWorker {
    fn drop(&mut self) {
        drop(self.tx.take());
        // A worker stuck inside a hung target cannot be interrupted; do not wait for it.
        if let Some(j) = self.join.take()
            && !self.busy.load(Ordering::Acquire)
        {
            let _ = j.join();
        }
    }
}

fn uia_thread(rx: mpsc::Receiver<UiaRequest>, busy: Arc<AtomicBool>) {
    // SAFETY: initialises COM for this thread only; balanced by CoUninitialize below.
    let init = unsafe { CoInitializeEx(None, COINIT_MULTITHREADED) };
    if init.is_err() {
        tracing::warn!(?init, "CoInitializeEx failed; password detection disabled");
    }
    // SAFETY: standard in-process creation of the UIA client object.
    let automation: Option<IUIAutomation> =
        unsafe { CoCreateInstance(&CUIAutomation, None, CLSCTX_INPROC_SERVER) }
            .inspect_err(|e| tracing::warn!(error = %e, "UI Automation unavailable"))
            .ok();
    while let Ok(req) = rx.recv() {
        busy.store(true, Ordering::Release);
        let answer = automation.as_ref().map(|a| {
            // SAFETY: COM calls on an interface created on this MTA thread.
            unsafe { a.GetFocusedElement() }
                // SAFETY: as above.
                .and_then(|el| unsafe { el.CurrentIsPassword() })
                .map(|b| b.as_bool())
                .unwrap_or(false)
        });
        busy.store(false, Ordering::Release);
        let _ = req.reply.try_send(answer.unwrap_or(false));
    }
    drop(automation);
    if init.is_ok() {
        // SAFETY: balances the successful CoInitializeEx above on the same thread.
        unsafe { CoUninitialize() };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn own_process_elevation_is_readable() {
        let ours = process_elevated(std::process::id());
        assert_eq!(ours, Some(self_elevated()));
    }

    #[test]
    fn unopenable_process_is_unknown() {
        // PID 4 is the System process; a limited query token open is refused.
        assert_eq!(process_elevated(4), None);
    }

    #[test]
    fn remote_session_flag_matches_session_name() {
        let session = std::env::var("SESSIONNAME").unwrap_or_default();
        if session.starts_with("RDP-") {
            assert!(is_remote_session());
        } else if session.eq_ignore_ascii_case("console") {
            assert!(!is_remote_session());
        }
    }

    #[test]
    fn capture_answers_quickly_and_uia_is_bounded() {
        let f = WinFocus::new();
        let t0 = std::time::Instant::now();
        let s = f.capture();
        assert!(t0.elapsed() < Duration::from_secs(2), "{:?}", t0.elapsed());
        if s.hwnd != 0 {
            assert!(s.pid != 0);
            assert!(s.exe.is_some() || s.elevated);
        }
        assert!(!f.is_still(&FocusSnapshot::default()));
        assert_eq!(s.remote_session, is_remote_session());
    }

    #[test]
    fn refocus_rejects_null_and_dead_handles() {
        assert!(!refocus_hwnd(0));
        assert!(!refocus_hwnd(0x7fff_fff0));
    }
}
