//! Commands travel on a channel and are drained inside the window procedure: a boxed
//! pointer per posted message leaks once the window is gone, and draining in our own loop
//! would stall while a tray menu runs its modal loop.

use std::cell::RefCell;
use std::path::{Path, PathBuf};
use std::sync::mpsc::{self, Receiver, Sender};
use std::sync::{Arc, Mutex};

use windows::Win32::Foundation::{
    CloseHandle, ERROR_ALREADY_EXISTS, GetLastError, HANDLE, HWND, LPARAM, LRESULT, WPARAM,
};
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::System::Threading::{CreateMutexW, ReleaseMutex};
use windows::Win32::UI::Shell::ShellExecuteW;
use windows::Win32::UI::WindowsAndMessaging::{
    CreateWindowExW, DefWindowProcW, DestroyWindow, DispatchMessageW, GetForegroundWindow,
    GetMessageW, HWND_MESSAGE, KillTimer, MSG, PostMessageW, PostQuitMessage, RegisterClassW,
    SW_SHOWNORMAL, SetTimer, TranslateMessage, WINDOW_EX_STYLE, WINDOW_STYLE, WM_APP, WM_DESTROY,
    WM_TIMER, WNDCLASSW,
};
use windows::core::{PCWSTR, w};

use crate::clipboard::{ClipboardError, WinClipboard};
use crate::overlay::{Overlay, OverlayConfig, OverlayState};
use crate::tray::{Tray, TrayEvent};
use crate::util::{hwnd_from, hwnd_raw, wide};

/// Per session: a second hook in one session would double every hotkey, but another
/// user's session may run its own instance.
pub const INSTANCE_MUTEX: &str = r"Local\hush";
const APP_DIR: &str = "hush";

#[derive(Debug, thiserror::Error)]
pub enum UiError {
    #[error("another hush is already running in this session")]
    AlreadyRunning,
    #[error("UI thread failed to start: {0}")]
    Start(String),
    #[error(transparent)]
    Clipboard(#[from] ClipboardError),
    #[error("{0}")]
    Win32(#[from] windows::core::Error),
}

pub struct InstanceGuard(HANDLE);

// SAFETY: a mutex handle is a process-wide kernel object reference usable from any thread.
unsafe impl Send for InstanceGuard {}

impl Drop for InstanceGuard {
    fn drop(&mut self) {
        // SAFETY: we own the handle and created the mutex with initial ownership.
        unsafe {
            let _ = ReleaseMutex(self.0);
            let _ = CloseHandle(self.0);
        }
    }
}

pub fn acquire_single_instance(name: &str) -> Result<InstanceGuard, UiError> {
    let wname = wide(name);
    // SAFETY: `wname` outlives the call; the handle is closed here or owned by the guard.
    unsafe {
        let h = CreateMutexW(None, true, PCWSTR(wname.as_ptr()))?;
        if GetLastError() == ERROR_ALREADY_EXISTS {
            let _ = CloseHandle(h);
            return Err(UiError::AlreadyRunning);
        }
        Ok(InstanceGuard(h))
    }
}

/// Config roams; models and history do not.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AppPaths {
    pub config_dir: PathBuf,
    pub config_file: PathBuf,
    pub data_dir: PathBuf,
    pub models_dir: PathBuf,
}

impl AppPaths {
    pub fn resolve() -> Option<Self> {
        let base = directories::BaseDirs::new()?;
        Some(Self::under(base.config_dir(), base.data_local_dir()))
    }

    fn under(roaming: &Path, local: &Path) -> Self {
        let config_dir = roaming.join(APP_DIR);
        let data_dir = local.join(APP_DIR);
        Self {
            config_file: config_dir.join("config.toml"),
            models_dir: data_dir.join("models"),
            config_dir,
            data_dir,
        }
    }
}

/// Falls back to Notepad because a fresh machine has no association for `.toml`.
pub fn open_path(path: &Path) -> std::io::Result<()> {
    let file = wide(&path.to_string_lossy());
    // SAFETY: NUL-terminated buffers that outlive the call.
    let r = unsafe {
        ShellExecuteW(
            None,
            w!("open"),
            PCWSTR(file.as_ptr()),
            PCWSTR::null(),
            PCWSTR::null(),
            SW_SHOWNORMAL,
        )
    };
    if r.0 as usize > 32 {
        return Ok(());
    }
    std::process::Command::new("notepad.exe")
        .arg(path)
        .spawn()
        .map(|_| ())
}

enum UiCommand {
    Overlay(OverlayState),
    Paused(bool),
    Tooltip(String),
    About,
    Shutdown,
}

const WM_UI_WAKE: u32 = WM_APP + 0x40;

#[derive(Debug, Clone, Copy, Default)]
pub struct UiOptions {
    pub overlay: OverlayConfig,
    pub tray: bool,
}

#[derive(Clone)]
pub struct UiHandle {
    inner: Arc<UiInner>,
}

struct UiInner {
    tx: Sender<UiCommand>,
    hwnd: isize,
    clipboard: WinClipboard,
    join: Mutex<Option<std::thread::JoinHandle<()>>>,
}

impl UiHandle {
    pub fn start(options: UiOptions) -> Result<(Self, Receiver<TrayEvent>), UiError> {
        let clipboard = WinClipboard::start()?;
        let (tx, rx) = mpsc::channel();
        let (tray_tx, tray_rx) = mpsc::channel();
        let (ready_tx, ready_rx) = mpsc::channel();
        let join = std::thread::Builder::new()
            .name("hush-ui".into())
            .spawn(move || ui_thread(options, rx, tray_tx, ready_tx))
            .map_err(|e| UiError::Start(e.to_string()))?;
        let hwnd = ready_rx
            .recv()
            .map_err(|_| UiError::Start("UI thread exited".into()))?
            .map_err(UiError::Start)?;
        Ok((
            Self {
                inner: Arc::new(UiInner {
                    tx,
                    hwnd,
                    clipboard,
                    join: Mutex::new(Some(join)),
                }),
            },
            tray_rx,
        ))
    }

    fn post(&self, cmd: UiCommand) {
        if self.inner.tx.send(cmd).is_ok() {
            // SAFETY: posting a plain wake-up to the UI thread's window.
            let _ = unsafe {
                PostMessageW(
                    Some(hwnd_from(self.inner.hwnd)),
                    WM_UI_WAKE,
                    WPARAM(0),
                    LPARAM(0),
                )
            };
        }
    }

    pub fn set_overlay(&self, state: OverlayState) {
        self.post(UiCommand::Overlay(state));
    }

    pub fn set_paused(&self, paused: bool) {
        self.post(UiCommand::Paused(paused));
    }

    /// " (paused)" is appended while paused.
    pub fn set_tooltip(&self, text: impl Into<String>) {
        self.post(UiCommand::Tooltip(text.into()));
    }

    pub fn show_about(&self) {
        self.post(UiCommand::About);
    }

    pub fn clipboard(&self) -> &WinClipboard {
        &self.inner.clipboard
    }

    pub fn shutdown(&self) {
        self.post(UiCommand::Shutdown);
        let join = self
            .inner
            .join
            .lock()
            .unwrap_or_else(|e| e.into_inner())
            .take();
        if let Some(j) = join {
            let _ = j.join();
        }
        self.inner.clipboard.shutdown();
    }
}

/// Every call returns without waiting.
pub struct WinNotifier {
    ui: UiHandle,
}

impl WinNotifier {
    pub fn new(ui: UiHandle) -> Self {
        Self { ui }
    }
}

impl hush_core::notify::Notifier for WinNotifier {
    fn set_state(&mut self, state: hush_core::notify::OverlayState) {
        self.ui.set_overlay(state.into());
    }

    fn play(&mut self, sound: hush_core::notify::Sound) {
        crate::sound::play(sound.into());
    }

    fn toast(&mut self, message: &str) {
        self.ui.set_overlay(OverlayState::Notice {
            message: message.to_string(),
        });
    }
}

pub const ADMIN_WINDOW_MESSAGE: &str = "Admin window: dictation unavailable here";
const ADMIN_POLL_TIMER: usize = 0x5718;
const ADMIN_POLL_MS: u32 = 500;

struct UiState {
    rx: Receiver<UiCommand>,
    hwnd: HWND,
    overlay: Option<Overlay>,
    tray: Option<Tray>,
    tooltip: String,
    paused: bool,
    /// What the driver last asked for; the admin warning is laid over it, never stored
    /// in its place, so the request comes back when focus leaves the admin window.
    requested: OverlayState,
    admin: AdminWatch,
}

#[derive(Default)]
struct AdminWatch {
    polling: bool,
    last_foreground: Option<isize>,
    elevated_foreground: bool,
}

impl UiState {
    fn show(&mut self, requested: OverlayState) {
        let dictating = requested.is_dictating();
        self.requested = requested;
        // Dictation can only have started in a window the hook sees, so until it ends
        // there is nothing to warn about.
        self.set_admin_polling(!dictating);
        self.render();
    }

    fn admin_warning(&self) -> bool {
        self.admin.elevated_foreground
            && !self.paused
            && matches!(
                self.requested,
                OverlayState::Hidden | OverlayState::Status { .. }
            )
    }

    fn render(&mut self) {
        let shown = if self.admin_warning() {
            OverlayState::Status {
                message: ADMIN_WINDOW_MESSAGE.into(),
            }
        } else {
            self.requested.clone()
        };
        if let Some(t) = self.tray.as_mut() {
            t.set_indicator(shown.tray_indicator());
        }
        if let Some(o) = self.overlay.as_mut() {
            o.set(shown);
        }
        self.apply_tooltip();
    }

    fn apply_tooltip(&self) {
        let Some(t) = self.tray.as_ref() else { return };
        if self.admin_warning() {
            t.set_tooltip(ADMIN_WINDOW_MESSAGE);
        } else if self.paused {
            t.set_tooltip(&format!("{} (paused)", self.tooltip));
        } else {
            t.set_tooltip(&self.tooltip);
        }
    }

    fn set_admin_polling(&mut self, on: bool) {
        let on = on && !crate::focus::self_elevated();
        if on == self.admin.polling {
            return;
        }
        self.admin.polling = on;
        self.admin.last_foreground = None;
        if on {
            // SAFETY: a timer on our own window; WM_TIMER is handled by its procedure.
            unsafe { SetTimer(Some(self.hwnd), ADMIN_POLL_TIMER, ADMIN_POLL_MS, None) };
        } else {
            self.admin.elevated_foreground = false;
            // SAFETY: our own window and timer id.
            let _ = unsafe { KillTimer(Some(self.hwnd), ADMIN_POLL_TIMER) };
        }
    }

    /// The elevation check opens the target's process token, so it runs only when the
    /// foreground window changes.
    fn poll_foreground(&mut self) {
        // SAFETY: plain FFI query.
        let fg = unsafe { GetForegroundWindow() };
        if self.admin.last_foreground == Some(hwnd_raw(fg)) {
            return;
        }
        self.admin.last_foreground = Some(hwnd_raw(fg));
        let elevated = crate::focus::window_is_elevated_over_us(fg);
        if elevated != self.admin.elevated_foreground {
            tracing::info!(elevated, "foreground window elevation changed");
            self.admin.elevated_foreground = elevated;
            self.render();
        }
    }

    fn on_timer(&mut self, id: usize) {
        match id {
            Overlay::HIDE_TIMER => self.show(OverlayState::Hidden),
            ADMIN_POLL_TIMER => self.poll_foreground(),
            _ => {}
        }
    }
}

thread_local! {
    static UI: RefCell<Option<UiState>> = const { RefCell::new(None) };
}

fn ui_thread(
    options: UiOptions,
    rx: Receiver<UiCommand>,
    tray_tx: Sender<TrayEvent>,
    ready: Sender<Result<isize, String>>,
) {
    let hwnd = match create_control_window() {
        Ok(h) => h,
        Err(e) => {
            let _ = ready.send(Err(e.to_string()));
            return;
        }
    };
    let overlay = match Overlay::create(options.overlay, Some(hwnd)) {
        Ok(o) => Some(o),
        Err(e) => {
            tracing::warn!(error = %e, "overlay unavailable");
            None
        }
    };
    let tray = if options.tray {
        match Tray::create(tray_tx) {
            Ok(t) => Some(t),
            Err(e) => {
                tracing::warn!(error = %e, "tray icon unavailable");
                None
            }
        }
    } else {
        None
    };
    let mut state = UiState {
        rx,
        hwnd,
        overlay,
        tray,
        tooltip: "hush".into(),
        paused: false,
        requested: OverlayState::Hidden,
        admin: AdminWatch::default(),
    };
    state.set_admin_polling(true);
    UI.with(|u| *u.borrow_mut() = Some(state));
    let _ = ready.send(Ok(hwnd_raw(hwnd)));
    let mut msg = MSG::default();
    loop {
        // SAFETY: standard message loop on the thread that owns the windows.
        let r = unsafe { GetMessageW(&mut msg, None, 0, 0) };
        if r.0 <= 0 {
            break;
        }
        // SAFETY: dispatching a message we just retrieved.
        unsafe {
            let _ = TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
    }
    UI.with(|u| drop(u.borrow_mut().take()));
}

fn create_control_window() -> windows::core::Result<HWND> {
    // SAFETY: a 'static window procedure and a message-only window owned by this thread.
    unsafe {
        let inst = GetModuleHandleW(PCWSTR::null())?;
        let class = w!("hush-ui-control");
        let wc = WNDCLASSW {
            lpfnWndProc: Some(control_wndproc),
            hInstance: inst.into(),
            lpszClassName: class,
            ..Default::default()
        };
        RegisterClassW(&wc);
        CreateWindowExW(
            WINDOW_EX_STYLE(0),
            class,
            w!("hush ui"),
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

unsafe extern "system" fn control_wndproc(hwnd: HWND, msg: u32, wp: WPARAM, lp: LPARAM) -> LRESULT {
    match msg {
        WM_UI_WAKE => {
            drain_commands(hwnd);
            LRESULT(0)
        }
        WM_TIMER => {
            UI.with(|u| {
                if let Ok(mut u) = u.try_borrow_mut()
                    && let Some(s) = u.as_mut()
                {
                    s.on_timer(wp.0);
                }
            });
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

fn drain_commands(hwnd: HWND) {
    let mut shutdown = false;
    UI.with(|u| {
        let Ok(mut guard) = u.try_borrow_mut() else {
            // Re-entered from inside a command; the outer drain picks the rest up.
            return;
        };
        let Some(state) = guard.as_mut() else { return };
        while let Ok(cmd) = state.rx.try_recv() {
            match cmd {
                UiCommand::Overlay(s) => state.show(s),
                UiCommand::Paused(p) => {
                    state.paused = p;
                    if let Some(t) = state.tray.as_ref() {
                        t.set_paused(p);
                    }
                    state.render();
                }
                UiCommand::Tooltip(s) => {
                    state.tooltip = s;
                    state.apply_tooltip();
                }
                UiCommand::About => crate::tray::show_about(),
                UiCommand::Shutdown => {
                    state.set_admin_polling(false);
                    state.tray = None;
                    state.overlay = None;
                    shutdown = true;
                    break;
                }
            }
        }
    });
    if shutdown {
        // SAFETY: destroying our own window on its thread; WM_DESTROY ends the loop.
        let _ = unsafe { DestroyWindow(hwnd) };
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn paths_follow_the_layout() {
        let p = AppPaths::under(Path::new(r"C:\R"), Path::new(r"C:\L"));
        assert_eq!(p.config_file, Path::new(r"C:\R\hush\config.toml"));
        assert_eq!(p.models_dir, Path::new(r"C:\L\hush\models"));
        let real = AppPaths::resolve().expect("known folders");
        let appdata = std::env::var("APPDATA").unwrap();
        assert!(real.config_file.starts_with(appdata));
        let local = std::env::var("LOCALAPPDATA").unwrap();
        assert!(real.models_dir.starts_with(local));
    }

    #[test]
    fn second_instance_is_refused() {
        let name = format!(r"Local\hush-test-{}", std::process::id());
        let first = acquire_single_instance(&name).expect("first");
        assert!(matches!(
            acquire_single_instance(&name),
            Err(UiError::AlreadyRunning)
        ));
        drop(first);
        assert!(acquire_single_instance(&name).is_ok());
    }

    #[test]
    fn admin_warning_lies_over_idle_states_only_and_restores_the_request() {
        let (_tx, rx) = mpsc::channel();
        let mut s = UiState {
            rx,
            hwnd: HWND::default(),
            overlay: None,
            tray: None,
            tooltip: String::new(),
            paused: false,
            requested: OverlayState::Status {
                message: "Loading".into(),
            },
            admin: AdminWatch {
                polling: true,
                last_foreground: Some(1),
                elevated_foreground: true,
            },
        };
        assert!(s.admin_warning());
        s.paused = true;
        assert!(!s.admin_warning());
        s.paused = false;
        s.requested = OverlayState::Done { message: None };
        assert!(!s.admin_warning());
        s.show(OverlayState::Listening { level: 0.0 });
        assert!(!s.admin.polling && !s.admin.elevated_foreground);
        assert!(!s.admin_warning());
    }

    #[test]
    fn ui_thread_starts_drives_overlay_and_shuts_down() {
        let (ui, _tray) = UiHandle::start(UiOptions {
            tray: false,
            ..Default::default()
        })
        .expect("ui");
        ui.set_overlay(OverlayState::Listening { level: 0.3 });
        ui.set_overlay(OverlayState::Done { message: None });
        ui.set_overlay(OverlayState::Status {
            message: "Loading".into(),
        });
        ui.set_tooltip("hush · test");
        ui.set_paused(true);
        let seq = ui.clipboard().sequence_number();
        assert!(seq > 0);
        std::thread::sleep(std::time::Duration::from_millis(100));
        ui.shutdown();
        ui.shutdown();
    }
}
