//! Windows silently removes a low-level hook whose callback overruns
//! `LowLevelHooksTimeout`, so the callback only compares, `try_send`s and returns; all
//! other work is posted back to the hook thread, the only thread allowed to unhook.
//! A key the hook swallows never reaches `GetAsyncKeyState`, so the watchdog reads the
//! hotkey showing up there as proof that nothing swallowed it.

use std::cell::{Cell, RefCell};
use std::sync::Arc;
use std::sync::atomic::{AtomicBool, AtomicU32, AtomicU64, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError, SyncSender, TrySendError};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use windows::Win32::Foundation::{LPARAM, LRESULT, WPARAM};
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::System::Threading::GetCurrentThreadId;
use windows::Win32::UI::Input::KeyboardAndMouse::{
    GetAsyncKeyState, INPUT, INPUT_0, INPUT_KEYBOARD, KEYBD_EVENT_FLAGS, KEYBDINPUT,
    KEYEVENTF_KEYUP, SendInput, VIRTUAL_KEY,
};
use windows::Win32::UI::WindowsAndMessaging::{
    CallNextHookEx, DispatchMessageW, GetMessageW, HHOOK, KBDLLHOOKSTRUCT, LLKHF_INJECTED, MSG,
    PM_NOREMOVE, PeekMessageW, PostThreadMessageW, SetWindowsHookExW, TranslateMessage,
    UnhookWindowsHookEx, WH_KEYBOARD_LL, WM_APP, WM_KEYDOWN, WM_QUIT, WM_SYSKEYDOWN,
};

use crate::INJECTED_TAG;

/// Timestamps are taken inside the callback, before any queueing.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HotkeyEvent {
    Down {
        at: Instant,
    },
    Up {
        at: Instant,
    },
    /// Escape while the hotkey is held or Escape is armed; the Escape itself is swallowed.
    Cancel {
        at: Instant,
    },
    /// If the hotkey was held, a synthetic `Up` precedes this.
    HookReinstalled {
        at: Instant,
        reason: ReinstallReason,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ReinstallReason {
    ProbeUnanswered,
    KeyNotSwallowed,
    HeartbeatStopped,
    Requested,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum HotkeyParseError {
    #[error("empty hotkey")]
    Empty,
    #[error("unknown key name {0:?}")]
    UnknownKey(String),
    #[error("{0:?} cannot be used as a modifier; use Ctrl, Shift, Alt or Win")]
    NotAModifier(String),
    #[error("modifier {0:?} given twice")]
    DuplicateModifier(String),
}

#[derive(Debug, thiserror::Error)]
pub enum HookError {
    #[error("SetWindowsHookExW failed: {0}")]
    Install(windows::core::Error),
    #[error("hook thread did not start: {0}")]
    Thread(String),
}

const MOD_CTRL: u8 = 1;
const MOD_SHIFT: u8 = 2;
const MOD_ALT: u8 = 4;
const MOD_WIN: u8 = 8;

const VK_SHIFT: u8 = 0x10;
const VK_CONTROL: u8 = 0x11;
const VK_MENU: u8 = 0x12;
const VK_ESCAPE: u8 = 0x1B;
const VK_LWIN: u8 = 0x5B;
const VK_RWIN: u8 = 0x5C;
const VK_LSHIFT: u8 = 0xA0;
const VK_RSHIFT: u8 = 0xA1;
const VK_LCONTROL: u8 = 0xA2;
const VK_RCONTROL: u8 = 0xA3;
const VK_LMENU: u8 = 0xA4;
const VK_RMENU: u8 = 0xA5;
/// Unassigned, so injecting it has no effect of its own.
const VK_MASK: u16 = 0xE8;

/// Modifiers must be held when the key goes down. In a modifier-only chord (`Ctrl+Win`)
/// the last one named is the key.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KeySpec {
    /// Side-neutral for `any_side` keys, with VK_LWIN standing for both Win keys.
    pub vk: u8,
    pub any_side: bool,
    /// Bit set of MOD_* flags.
    pub modifiers: u8,
}

impl KeySpec {
    const VALID: u32 = 1 << 31;

    fn pack(self) -> u32 {
        Self::VALID | self.vk as u32 | (self.any_side as u32) << 8 | (self.modifiers as u32) << 16
    }

    fn unpack(v: u32) -> Option<Self> {
        (v & Self::VALID != 0).then_some(Self {
            vk: (v & 0xFF) as u8,
            any_side: v & (1 << 8) != 0,
            modifiers: ((v >> 16) & 0xFF) as u8,
        })
    }

    fn matches(self, vk: u8) -> bool {
        if self.any_side {
            side_neutral(vk) == self.vk
        } else {
            vk == self.vk
        }
    }

    fn concrete_vks(self) -> &'static [u8] {
        match (self.any_side, self.vk) {
            (true, VK_CONTROL) => &[VK_LCONTROL, VK_RCONTROL],
            (true, VK_SHIFT) => &[VK_LSHIFT, VK_RSHIFT],
            (true, VK_MENU) => &[VK_LMENU, VK_RMENU],
            (true, VK_LWIN) => &[VK_LWIN, VK_RWIN],
            _ => std::slice::from_ref(&VK_TABLE[self.vk as usize]),
        }
    }
}

static VK_TABLE: [u8; 256] = {
    let mut t = [0u8; 256];
    let mut i = 0;
    while i < 256 {
        t[i] = i as u8;
        i += 1;
    }
    t
};

fn side_neutral(vk: u8) -> u8 {
    match vk {
        VK_LSHIFT | VK_RSHIFT => VK_SHIFT,
        VK_LCONTROL | VK_RCONTROL => VK_CONTROL,
        VK_LMENU | VK_RMENU => VK_MENU,
        VK_RWIN => VK_LWIN,
        other => other,
    }
}

fn modifier_bit(vk: u8) -> u8 {
    match side_neutral(vk) {
        VK_CONTROL => MOD_CTRL,
        VK_SHIFT => MOD_SHIFT,
        VK_MENU => MOD_ALT,
        VK_LWIN => MOD_WIN,
        _ => 0,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HotkeyConfig {
    pub key: KeySpec,
    pub watchdog_period: Duration,
    /// Liveness probe while the hotkey is held; zero disables it.
    pub probe_interval: Duration,
    accept_injected: bool,
}

impl HotkeyConfig {
    /// `RightCtrl`, `F13`, `Ctrl+Shift+Space`: case-insensitive, `+` separated, modifiers
    /// first.
    pub fn parse(s: &str) -> Result<Self, HotkeyParseError> {
        Ok(Self::new(parse_key_spec(s)?))
    }

    pub fn new(key: KeySpec) -> Self {
        Self {
            key,
            watchdog_period: Duration::from_millis(100),
            probe_interval: Duration::from_millis(500),
            accept_injected: false,
        }
    }

    /// Tests cannot press physical keys; in normal use another program's injected input
    /// must never start a recording.
    #[doc(hidden)]
    pub fn accept_injected_for_tests(mut self) -> Self {
        self.accept_injected = true;
        self
    }
}

fn parse_key_spec(s: &str) -> Result<KeySpec, HotkeyParseError> {
    let parts: Vec<&str> = s
        .split('+')
        .map(str::trim)
        .filter(|p| !p.is_empty())
        .collect();
    let (last, mods) = parts.split_last().ok_or(HotkeyParseError::Empty)?;
    let mut modifiers = 0u8;
    for m in mods {
        let bit = match m.to_ascii_lowercase().as_str() {
            "ctrl" | "control" => MOD_CTRL,
            "shift" => MOD_SHIFT,
            "alt" => MOD_ALT,
            "win" | "windows" | "super" => MOD_WIN,
            _ => {
                return Err(if key_by_name(m).is_some() {
                    HotkeyParseError::NotAModifier(m.to_string())
                } else {
                    HotkeyParseError::UnknownKey(m.to_string())
                });
            }
        };
        if modifiers & bit != 0 {
            return Err(HotkeyParseError::DuplicateModifier(m.to_string()));
        }
        modifiers |= bit;
    }
    let (vk, any_side) =
        key_by_name(last).ok_or_else(|| HotkeyParseError::UnknownKey(last.to_string()))?;
    let own_bit = if any_side { modifier_bit(vk) } else { 0 };
    if own_bit != 0 && modifiers & own_bit != 0 {
        return Err(HotkeyParseError::DuplicateModifier(last.to_string()));
    }
    Ok(KeySpec {
        vk,
        any_side,
        modifiers,
    })
}

/// (virtual key, any side).
fn key_by_name(name: &str) -> Option<(u8, bool)> {
    let n = name.to_ascii_lowercase().replace(['_', '-', ' '], "");
    let fixed = match n.as_str() {
        "rightctrl" | "rctrl" | "rightcontrol" | "rcontrol" => (VK_RCONTROL, false),
        "leftctrl" | "lctrl" | "leftcontrol" | "lcontrol" => (VK_LCONTROL, false),
        "ctrl" | "control" => (VK_CONTROL, true),
        "rightshift" | "rshift" => (VK_RSHIFT, false),
        "leftshift" | "lshift" => (VK_LSHIFT, false),
        "shift" => (VK_SHIFT, true),
        "rightalt" | "ralt" | "altgr" => (VK_RMENU, false),
        "leftalt" | "lalt" => (VK_LMENU, false),
        "alt" => (VK_MENU, true),
        "win" | "windows" | "super" => (VK_LWIN, true),
        "leftwin" | "lwin" => (VK_LWIN, false),
        "rightwin" | "rwin" => (VK_RWIN, false),
        "capslock" | "caps" => (0x14, false),
        "scrolllock" => (0x91, false),
        "pause" => (0x13, false),
        "insert" | "ins" => (0x2D, false),
        "space" => (0x20, false),
        "apps" | "menu" | "contextmenu" => (0x5D, false),
        _ => {
            if let Some(num) = n.strip_prefix('f')
                && let Ok(i) = num.parse::<u8>()
                && (1..=24).contains(&i)
            {
                return Some((0x70 + i - 1, false));
            }
            let mut chars = n.chars();
            return match (chars.next(), chars.next()) {
                (Some(c), None) if c.is_ascii_alphanumeric() => {
                    Some((c.to_ascii_uppercase() as u8, false))
                }
                _ => None,
            };
        }
    };
    Some(fixed)
}

struct Shared {
    spec: AtomicU32,
    accept_injected: AtomicBool,
    heartbeat: AtomicU64,
    held: AtomicBool,
    /// Off when idle, so an idle app never steals Escape from the user.
    escape_armed: AtomicBool,
    stall_ms: AtomicU32,
    stop: AtomicBool,
}

const WM_HOOK_REINSTALL: u32 = WM_APP + 1;
const WM_HOOK_MASK: u32 = WM_APP + 2;

struct CallbackState {
    shared: Option<Arc<Shared>>,
    tx: Option<SyncSender<HotkeyEvent>>,
}

thread_local! {
    // Const-initialised so first access from the callback allocates nothing.
    static CB: RefCell<CallbackState> = const {
        RefCell::new(CallbackState { shared: None, tx: None })
    };
    /// 0 when not held.
    static SWALLOWED: Cell<u8> = const { Cell::new(0) };
    static ESC_SWALLOWED: Cell<bool> = const { Cell::new(false) };
    static MODS_DOWN: Cell<u16> = const { Cell::new(0) };
    static THREAD_ID: Cell<u32> = const { Cell::new(0) };
}

fn side_bit(vk: u8) -> u16 {
    match vk {
        VK_LCONTROL => 1,
        VK_RCONTROL => 2,
        VK_LSHIFT => 4,
        VK_RSHIFT => 8,
        VK_LMENU => 16,
        VK_RMENU => 32,
        VK_LWIN => 64,
        VK_RWIN => 128,
        _ => 0,
    }
}

fn held_modifier_mask(sides: u16) -> u8 {
    let mut m = 0;
    if sides & 3 != 0 {
        m |= MOD_CTRL;
    }
    if sides & 12 != 0 {
        m |= MOD_SHIFT;
    }
    if sides & 48 != 0 {
        m |= MOD_ALT;
    }
    if sides & 192 != 0 {
        m |= MOD_WIN;
    }
    m
}

unsafe extern "system" fn ll_keyboard_proc(code: i32, wparam: WPARAM, lparam: LPARAM) -> LRESULT {
    // SAFETY: forwarding the unmodified arguments is the documented contract.
    let pass = || unsafe { CallNextHookEx(None, code, wparam, lparam) };
    if code < 0 {
        return pass();
    }
    // SAFETY: for HC_ACTION, lParam points to a KBDLLHOOKSTRUCT valid for this call.
    let kb = unsafe { &*(lparam.0 as *const KBDLLHOOKSTRUCT) };
    let swallow = CB.with(|cb| {
        let Ok(cb) = cb.try_borrow() else {
            return false;
        };
        let (Some(shared), Some(tx)) = (cb.shared.as_ref(), cb.tx.as_ref()) else {
            return false;
        };
        shared.heartbeat.fetch_add(1, Ordering::Relaxed);
        let stall = shared.stall_ms.swap(0, Ordering::Relaxed);
        if stall != 0 {
            std::thread::sleep(Duration::from_millis(stall as u64));
        }
        if kb.dwExtraInfo == INJECTED_TAG {
            return false;
        }
        decide(shared, tx, kb, wparam.0 as u32)
    });
    if swallow { LRESULT(1) } else { pass() }
}

/// Returns true to swallow.
fn decide(shared: &Shared, tx: &SyncSender<HotkeyEvent>, kb: &KBDLLHOOKSTRUCT, msg: u32) -> bool {
    let vk = (kb.vkCode & 0xFF) as u8;
    let down = msg == WM_KEYDOWN || msg == WM_SYSKEYDOWN;
    let side = side_bit(vk);
    if side != 0 {
        MODS_DOWN.with(|m| {
            m.set(if down {
                m.get() | side
            } else {
                m.get() & !side
            });
        });
    }
    let injected = kb.flags.0 & LLKHF_INJECTED.0 != 0;
    if injected && !shared.accept_injected.load(Ordering::Relaxed) {
        return false;
    }
    let at = Instant::now();
    let held_vk = SWALLOWED.with(Cell::get);
    if held_vk != 0 {
        if vk == held_vk {
            if !down {
                SWALLOWED.with(|s| s.set(0));
                shared.held.store(false, Ordering::Release);
                send(tx, HotkeyEvent::Up { at });
            }
            // A second down while held is auto-repeat: swallowed, not reported.
            return true;
        }
        if vk == VK_ESCAPE {
            if down {
                ESC_SWALLOWED.with(|e| e.set(true));
                send(tx, HotkeyEvent::Cancel { at });
            } else {
                ESC_SWALLOWED.with(|e| e.set(false));
            }
            return true;
        }
        return false;
    }
    if vk == VK_ESCAPE && !down && ESC_SWALLOWED.with(|e| e.replace(false)) {
        return true;
    }
    if vk == VK_ESCAPE && down && shared.escape_armed.load(Ordering::Relaxed) {
        ESC_SWALLOWED.with(|e| e.set(true));
        send(tx, HotkeyEvent::Cancel { at });
        return true;
    }
    if !down {
        return false;
    }
    let Some(spec) = KeySpec::unpack(shared.spec.load(Ordering::Relaxed)) else {
        return false;
    };
    if !spec.matches(vk) {
        return false;
    }
    let held_mods = MODS_DOWN.with(|m| held_modifier_mask(m.get() & !side_bit(vk)));
    if held_mods & spec.modifiers != spec.modifiers {
        return false;
    }
    SWALLOWED.with(|s| s.set(vk));
    shared.held.store(true, Ordering::Release);
    send(tx, HotkeyEvent::Down { at });
    // Releasing Alt or Win with nothing pressed in between opens a menu or Start; an
    // unassigned key in between prevents it.
    if held_mods & (MOD_ALT | MOD_WIN) != 0 || matches!(side_neutral(vk), VK_MENU | VK_LWIN) {
        let tid = THREAD_ID.with(Cell::get);
        // SAFETY: posting to our own thread's queue; no pointers travel with it.
        let _ = unsafe { PostThreadMessageW(tid, WM_HOOK_MASK, WPARAM(0), LPARAM(0)) };
    }
    true
}

fn send(tx: &SyncSender<HotkeyEvent>, ev: HotkeyEvent) {
    // A full channel means the consumer is stuck; dropping beats blocking the hook into
    // removal.
    match tx.try_send(ev) {
        Ok(()) | Err(TrySendError::Full(_)) | Err(TrySendError::Disconnected(_)) => {}
    }
}

fn mask_inputs(up_only: bool) -> Vec<INPUT> {
    let key = |flags: KEYBD_EVENT_FLAGS| INPUT {
        r#type: INPUT_KEYBOARD,
        Anonymous: INPUT_0 {
            ki: KEYBDINPUT {
                wVk: VIRTUAL_KEY(VK_MASK),
                wScan: 0,
                dwFlags: flags,
                time: 0,
                dwExtraInfo: INJECTED_TAG,
            },
        },
    };
    if up_only {
        vec![key(KEYEVENTF_KEYUP)]
    } else {
        vec![key(KEYBD_EVENT_FLAGS(0)), key(KEYEVENTF_KEYUP)]
    }
}

fn send_inputs(inputs: &[INPUT]) -> u32 {
    // SAFETY: `inputs` is a valid slice of INPUT for the duration of the call.
    unsafe { SendInput(inputs, std::mem::size_of::<INPUT>() as i32) }
}

fn install_hook() -> windows::core::Result<HHOOK> {
    // SAFETY: our own module handle lives for the process and the callback is 'static.
    unsafe {
        let module = GetModuleHandleW(None)?;
        SetWindowsHookExW(
            WH_KEYBOARD_LL,
            Some(ll_keyboard_proc),
            Some(module.into()),
            0,
        )
    }
}

fn hook_thread(
    shared: Arc<Shared>,
    tx: SyncSender<HotkeyEvent>,
    ready: mpsc::Sender<Result<u32, windows::core::Error>>,
) {
    let mut msg = MSG::default();
    // SAFETY: creates this thread's message queue so posted thread messages are kept.
    let _ = unsafe { PeekMessageW(&mut msg, None, 0, 0, PM_NOREMOVE) };
    // SAFETY: plain FFI query.
    let tid = unsafe { GetCurrentThreadId() };
    THREAD_ID.with(|t| t.set(tid));
    CB.with(|cb| {
        let mut cb = cb.borrow_mut();
        cb.shared = Some(shared.clone());
        cb.tx = Some(tx.clone());
    });
    let mut hook = match install_hook() {
        Ok(h) => h,
        Err(e) => {
            let _ = ready.send(Err(e));
            return;
        }
    };
    let _ = ready.send(Ok(tid));
    loop {
        // SAFETY: `msg` is a valid out-parameter; a thread message loop without a window.
        let r = unsafe { GetMessageW(&mut msg, None, 0, 0) };
        if r.0 <= 0 || msg.message == WM_QUIT {
            break;
        }
        match msg.message {
            WM_HOOK_MASK => {
                send_inputs(&mask_inputs(false));
            }
            WM_HOOK_REINSTALL => {
                let reason = reason_from(msg.wParam.0);
                // SAFETY: `hook` was installed by this thread and is removed once here.
                let _ = unsafe { UnhookWindowsHookEx(hook) };
                let was_held = SWALLOWED.with(|s| s.replace(0)) != 0;
                ESC_SWALLOWED.with(|e| e.set(false));
                MODS_DOWN.with(|m| m.set(0));
                shared.held.store(false, Ordering::Release);
                match install_hook() {
                    Ok(h) => hook = h,
                    Err(e) => {
                        tracing::error!(error = %e, "keyboard hook reinstall failed; hotkey is dead");
                        break;
                    }
                }
                let at = Instant::now();
                if was_held {
                    let _ = tx.try_send(HotkeyEvent::Up { at });
                }
                let _ = tx.try_send(HotkeyEvent::HookReinstalled { at, reason });
                tracing::warn!(?reason, was_held, "keyboard hook reinstalled");
            }
            _ => {
                // SAFETY: standard dispatch of a message we just retrieved.
                unsafe {
                    let _ = TranslateMessage(&msg);
                    DispatchMessageW(&msg);
                }
            }
        }
    }
    // SAFETY: the hook belongs to this thread; a failure means it was already removed.
    let _ = unsafe { UnhookWindowsHookEx(hook) };
    CB.with(|cb| {
        let mut cb = cb.borrow_mut();
        cb.shared = None;
        cb.tx = None;
    });
}

fn reason_code(r: ReinstallReason) -> usize {
    match r {
        ReinstallReason::ProbeUnanswered => 1,
        ReinstallReason::KeyNotSwallowed => 2,
        ReinstallReason::HeartbeatStopped => 3,
        ReinstallReason::Requested => 4,
    }
}

fn reason_from(c: usize) -> ReinstallReason {
    match c {
        1 => ReinstallReason::ProbeUnanswered,
        2 => ReinstallReason::KeyNotSwallowed,
        3 => ReinstallReason::HeartbeatStopped,
        _ => ReinstallReason::Requested,
    }
}

fn key_down_async(vk: u8) -> bool {
    // SAFETY: plain FFI query. Only ever called off the hook thread.
    (unsafe { GetAsyncKeyState(vk as i32) } as u16) & 0x8000 != 0
}

/// Skips the mouse buttons, which never pass through a keyboard hook.
/// Left, right, middle, X1, X2. Not a range: 3 between them is Ctrl+Break.
const MOUSE_BUTTON_VKS: [usize; 5] = [0x01, 0x02, 0x04, 0x05, 0x06];

fn is_sampled_key(vk: usize) -> bool {
    vk != 0 && vk < 0xFF && !MOUSE_BUTTON_VKS.contains(&vk)
}

fn sample_keys(buf: &mut [bool; 256]) {
    for (vk, slot) in buf.iter_mut().enumerate() {
        *slot = is_sampled_key(vk) && key_down_async(vk as u8);
    }
}

fn watchdog_thread(
    shared: Arc<Shared>,
    hook_tid: u32,
    period: Duration,
    probe_interval: Duration,
    stop_rx: mpsc::Receiver<()>,
) {
    let mut prev_keys = [false; 256];
    let mut cur_keys = [false; 256];
    sample_keys(&mut prev_keys);
    let mut hb_hist = [shared.heartbeat.load(Ordering::Relaxed); 2];
    let mut last_probe = Instant::now();
    let mut unswallowed_ticks = 0u32;
    // After a reinstall the hotkey may still be physically down; it counts again only
    // once it has been seen released.
    let mut key_check_armed = true;
    let mut last_reinstall: Option<Instant> = None;

    let reinstall = |reason: ReinstallReason, last: &mut Option<Instant>| {
        if last.is_some_and(|t| t.elapsed() < Duration::from_secs(1)) {
            return;
        }
        *last = Some(Instant::now());
        // SAFETY: posting a plain integer message to the hook thread.
        let _ = unsafe {
            PostThreadMessageW(
                hook_tid,
                WM_HOOK_REINSTALL,
                WPARAM(reason_code(reason)),
                LPARAM(0),
            )
        };
    };

    loop {
        match stop_rx.recv_timeout(period) {
            Ok(()) | Err(RecvTimeoutError::Disconnected) => return,
            Err(RecvTimeoutError::Timeout) => {}
        }
        if shared.stop.load(Ordering::Acquire) {
            return;
        }
        let hb_before = shared.heartbeat.load(Ordering::Relaxed);
        sample_keys(&mut cur_keys);
        let blind = crate::focus::foreground_is_elevated_over_us();
        let held = shared.held.load(Ordering::Acquire);
        let spec = KeySpec::unpack(shared.spec.load(Ordering::Relaxed));

        if held && !blind && !probe_interval.is_zero() && last_probe.elapsed() >= probe_interval {
            last_probe = Instant::now();
            let start = shared.heartbeat.load(Ordering::Relaxed);
            send_inputs(&mask_inputs(true));
            let deadline = Instant::now() + Duration::from_millis(250);
            let mut answered = false;
            while Instant::now() < deadline {
                if shared.heartbeat.load(Ordering::Relaxed) != start {
                    answered = true;
                    break;
                }
                std::thread::sleep(Duration::from_millis(5));
            }
            if !answered && shared.held.load(Ordering::Acquire) {
                reinstall(ReinstallReason::ProbeUnanswered, &mut last_reinstall);
            }
        }

        // With required modifiers a bare press legitimately passes through unswallowed.
        if let Some(spec) = spec
            && spec.modifiers == 0
        {
            let visible = spec.concrete_vks().iter().any(|&vk| cur_keys[vk as usize]);
            if !visible {
                key_check_armed = true;
                unswallowed_ticks = 0;
            } else if key_check_armed && !held && !blind {
                unswallowed_ticks += 1;
                if unswallowed_ticks >= 2 {
                    key_check_armed = false;
                    unswallowed_ticks = 0;
                    reinstall(ReinstallReason::KeyNotSwallowed, &mut last_reinstall);
                }
            }
        }

        // Compared with the heartbeat of two ticks ago, so an event caught between the
        // hook and the key-state update is not misread as a dead hook.
        let changed = cur_keys
            .iter()
            .zip(prev_keys.iter())
            .enumerate()
            .any(|(vk, (a, b))| a != b && vk as u8 != VK_MASK as u8);
        if changed && !blind && hb_before == hb_hist[0] {
            reinstall(ReinstallReason::HeartbeatStopped, &mut last_reinstall);
        }
        hb_hist = [hb_hist[1], hb_before];
        std::mem::swap(&mut prev_keys, &mut cur_keys);
    }
}

pub struct HotkeyHook;

impl HotkeyHook {
    /// The callback never blocks on `tx`; events are dropped while it is full.
    pub fn install(
        config: HotkeyConfig,
        tx: SyncSender<HotkeyEvent>,
    ) -> Result<HookHandle, HookError> {
        let shared = Arc::new(Shared {
            spec: AtomicU32::new(config.key.pack()),
            accept_injected: AtomicBool::new(config.accept_injected),
            heartbeat: AtomicU64::new(0),
            held: AtomicBool::new(false),
            escape_armed: AtomicBool::new(false),
            stall_ms: AtomicU32::new(0),
            stop: AtomicBool::new(false),
        });
        let (ready_tx, ready_rx) = mpsc::channel();
        let hook_shared = shared.clone();
        let hook_join = std::thread::Builder::new()
            .name("wl-hotkey-hook".into())
            .spawn(move || hook_thread(hook_shared, tx, ready_tx))
            .map_err(|e| HookError::Thread(e.to_string()))?;
        let tid = match ready_rx.recv() {
            Ok(Ok(tid)) => tid,
            Ok(Err(e)) => {
                let _ = hook_join.join();
                return Err(HookError::Install(e));
            }
            Err(_) => return Err(HookError::Thread("hook thread exited".into())),
        };
        let (stop_tx, stop_rx) = mpsc::channel();
        let wd_shared = shared.clone();
        let period = config.watchdog_period;
        let probe = config.probe_interval;
        let watchdog_join = std::thread::Builder::new()
            .name("wl-hotkey-watchdog".into())
            .spawn(move || watchdog_thread(wd_shared, tid, period, probe, stop_rx))
            .map_err(|e| HookError::Thread(e.to_string()))?;
        Ok(HookHandle {
            shared,
            hook_tid: tid,
            hook_join: Some(hook_join),
            watchdog_join: Some(watchdog_join),
            watchdog_stop: Some(stop_tx),
        })
    }
}

/// Dropping it unhooks and joins both threads.
pub struct HookHandle {
    shared: Arc<Shared>,
    hook_tid: u32,
    hook_join: Option<JoinHandle<()>>,
    watchdog_join: Option<JoinHandle<()>>,
    watchdog_stop: Option<mpsc::Sender<()>>,
}

impl HookHandle {
    /// A key already held keeps being tracked until its release.
    pub fn update(&self, config: &HotkeyConfig) {
        self.shared.spec.store(config.key.pack(), Ordering::Relaxed);
        self.shared
            .accept_injected
            .store(config.accept_injected, Ordering::Relaxed);
    }

    /// Callback invocations so far (every key event on the desktop, ours included).
    pub fn heartbeat(&self) -> u64 {
        self.shared.heartbeat.load(Ordering::Relaxed)
    }

    /// Swallow Escape and report it as [`HotkeyEvent::Cancel`] even while the hotkey is up.
    pub fn set_escape_armed(&self, armed: bool) {
        self.shared.escape_armed.store(armed, Ordering::Relaxed);
    }

    pub fn is_held(&self) -> bool {
        self.shared.held.load(Ordering::Acquire)
    }

    pub fn reinstall(&self) {
        // SAFETY: posting a plain integer message to the hook thread.
        let _ = unsafe {
            PostThreadMessageW(
                self.hook_tid,
                WM_HOOK_REINSTALL,
                WPARAM(reason_code(ReinstallReason::Requested)),
                LPARAM(0),
            )
        };
    }

    /// Past `LowLevelHooksTimeout`, Windows silently removes the hook.
    #[doc(hidden)]
    pub fn debug_stall_next_callback(&self, ms: u32) {
        self.shared.stall_ms.store(ms, Ordering::Relaxed);
    }
}

impl Drop for HookHandle {
    fn drop(&mut self) {
        self.shared.stop.store(true, Ordering::Release);
        drop(self.watchdog_stop.take());
        if let Some(j) = self.watchdog_join.take() {
            let _ = j.join();
        }
        // SAFETY: posting WM_QUIT to the hook thread; it unhooks itself before exiting.
        let _ = unsafe { PostThreadMessageW(self.hook_tid, WM_QUIT, WPARAM(0), LPARAM(0)) };
        if let Some(j) = self.hook_join.take() {
            let _ = j.join();
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Mutex;

    #[test]
    fn watchdog_samples_ctrl_break_but_not_mouse_buttons() {
        assert!(is_sampled_key(0x03));
        for vk in [0x01, 0x02, 0x04, 0x05, 0x06] {
            assert!(!is_sampled_key(vk), "{vk:#x}");
        }
        assert!(!is_sampled_key(0x00));
        assert!(!is_sampled_key(0xFF));
        assert!(is_sampled_key(0x08));
    }

    #[test]
    fn parse_right_ctrl() {
        let c = HotkeyConfig::parse("RightCtrl").unwrap();
        assert_eq!(
            c.key,
            KeySpec {
                vk: VK_RCONTROL,
                any_side: false,
                modifiers: 0
            }
        );
        assert_eq!(HotkeyConfig::parse("rctrl").unwrap().key, c.key);
    }

    #[test]
    fn parse_caps_lock_and_f13() {
        assert_eq!(HotkeyConfig::parse("CapsLock").unwrap().key.vk, 0x14);
        assert_eq!(HotkeyConfig::parse("F13").unwrap().key.vk, 0x7C);
        assert_eq!(HotkeyConfig::parse("f24").unwrap().key.vk, 0x87);
        assert!(HotkeyConfig::parse("F25").is_err());
    }

    #[test]
    fn parse_ctrl_win_chord() {
        let k = HotkeyConfig::parse("Ctrl+Win").unwrap().key;
        assert_eq!(k.vk, VK_LWIN);
        assert!(k.any_side);
        assert_eq!(k.modifiers, MOD_CTRL);
        assert!(k.matches(VK_RWIN) && k.matches(VK_LWIN));
    }

    #[test]
    fn parse_errors() {
        assert_eq!(HotkeyConfig::parse(""), Err(HotkeyParseError::Empty));
        assert!(matches!(
            HotkeyConfig::parse("Hyper"),
            Err(HotkeyParseError::UnknownKey(_))
        ));
        assert!(matches!(
            HotkeyConfig::parse("F1+A"),
            Err(HotkeyParseError::NotAModifier(_))
        ));
        assert!(matches!(
            HotkeyConfig::parse("Ctrl+Ctrl"),
            Err(HotkeyParseError::DuplicateModifier(_))
        ));
    }

    #[test]
    fn pack_roundtrip() {
        for s in [
            "RightCtrl",
            "Ctrl+Shift+Space",
            "Ctrl+Win",
            "CapsLock",
            "Alt+F13",
        ] {
            let k = HotkeyConfig::parse(s).unwrap().key;
            assert_eq!(KeySpec::unpack(k.pack()), Some(k), "{s}");
        }
        assert_eq!(KeySpec::unpack(0), None);
    }

    #[test]
    fn armed_escape_is_swallowed_without_the_hotkey() {
        use windows::Win32::UI::WindowsAndMessaging::WM_KEYUP;
        let shared = Shared {
            spec: AtomicU32::new(HotkeyConfig::parse("RightCtrl").unwrap().key.pack()),
            accept_injected: AtomicBool::new(false),
            heartbeat: AtomicU64::new(0),
            held: AtomicBool::new(false),
            escape_armed: AtomicBool::new(false),
            stall_ms: AtomicU32::new(0),
            stop: AtomicBool::new(false),
        };
        let (tx, rx) = mpsc::sync_channel(8);
        let esc = KBDLLHOOKSTRUCT {
            vkCode: VK_ESCAPE as u32,
            ..Default::default()
        };
        assert!(!decide(&shared, &tx, &esc, WM_KEYDOWN));
        assert!(rx.try_recv().is_err(), "an idle app must not steal Escape");

        shared.escape_armed.store(true, Ordering::Relaxed);
        assert!(decide(&shared, &tx, &esc, WM_KEYDOWN));
        assert!(matches!(rx.try_recv(), Ok(HotkeyEvent::Cancel { .. })));
        shared.escape_armed.store(false, Ordering::Relaxed);
        assert!(
            decide(&shared, &tx, &esc, WM_KEYUP),
            "the up of a swallowed down is ours even after disarming"
        );
    }

    #[test]
    fn modifier_mask_from_sides() {
        assert_eq!(held_modifier_mask(side_bit(VK_RCONTROL)), MOD_CTRL);
        assert_eq!(
            held_modifier_mask(side_bit(VK_LWIN) | side_bit(VK_RSHIFT)),
            MOD_WIN | MOD_SHIFT
        );
    }

    static LIVE: Mutex<()> = Mutex::new(());

    fn key_event(vk: u8, up: bool) -> INPUT {
        INPUT {
            r#type: INPUT_KEYBOARD,
            Anonymous: INPUT_0 {
                ki: KEYBDINPUT {
                    wVk: VIRTUAL_KEY(vk as u16),
                    wScan: 0,
                    dwFlags: if up {
                        KEYEVENTF_KEYUP
                    } else {
                        KEYBD_EVENT_FLAGS(0)
                    },
                    time: 0,
                    dwExtraInfo: 0,
                },
            },
        }
    }

    /// A locked or disconnected session refuses every injection, so live tests report
    /// SKIPPED there instead of failing.
    pub(crate) fn input_desktop_available() -> bool {
        let n = super::send_inputs(&mask_inputs(true));
        if n == 0 {
            eprintln!(
                "SKIPPED live input test: SendInput refused ({}); session disconnected or locked?",
                windows::core::Error::from_thread()
            );
        }
        n == 1
    }

    fn live_hook(key: &str) -> (HookHandle, mpsc::Receiver<HotkeyEvent>) {
        let (tx, rx) = mpsc::sync_channel(64);
        let cfg = HotkeyConfig::parse(key)
            .unwrap()
            .accept_injected_for_tests();
        (HotkeyHook::install(cfg, tx).expect("install hook"), rx)
    }

    fn next(rx: &mpsc::Receiver<HotkeyEvent>) -> Option<HotkeyEvent> {
        rx.recv_timeout(Duration::from_millis(1000)).ok()
    }

    /// Asserts, so a refused injection fails loudly instead of looking like a hook that
    /// saw nothing.
    fn send_inputs(inputs: &[INPUT]) -> u32 {
        let n = super::send_inputs(inputs);
        assert_eq!(
            n as usize,
            inputs.len(),
            "SendInput refused: {}",
            windows::core::Error::from_thread()
        );
        n
    }

    #[test]
    fn down_repeat_escape_up() {
        let _g = LIVE.lock().unwrap_or_else(|e| e.into_inner());
        if !input_desktop_available() {
            return;
        }
        let (h, rx) = live_hook("F13");
        send_inputs(&[key_event(0x7C, false)]);
        assert!(matches!(next(&rx), Some(HotkeyEvent::Down { .. })));
        assert!(h.is_held());
        send_inputs(&[key_event(0x7C, false), key_event(0x7C, false)]);
        send_inputs(&[key_event(VK_ESCAPE, false), key_event(VK_ESCAPE, true)]);
        assert!(matches!(next(&rx), Some(HotkeyEvent::Cancel { .. })));
        send_inputs(&[key_event(0x7C, true)]);
        assert!(matches!(next(&rx), Some(HotkeyEvent::Up { .. })));
        assert!(!h.is_held());
        let mut own = key_event(0x7C, false);
        own.Anonymous.ki.dwExtraInfo = INJECTED_TAG;
        let mut own_up = key_event(0x7C, true);
        own_up.Anonymous.ki.dwExtraInfo = INJECTED_TAG;
        send_inputs(&[own, own_up]);
        let stray = rx.recv_timeout(Duration::from_millis(300)).ok();
        assert_eq!(stray, None, "own-tagged events produced {stray:?}");
    }

    #[test]
    fn modifiers_required_for_chord() {
        let _g = LIVE.lock().unwrap_or_else(|e| e.into_inner());
        if !input_desktop_available() {
            return;
        }
        let (h, rx) = live_hook("Shift+F14");
        send_inputs(&[key_event(0x7D, false), key_event(0x7D, true)]);
        assert_eq!(rx.recv_timeout(Duration::from_millis(300)).ok(), None);
        send_inputs(&[key_event(VK_LSHIFT, false), key_event(0x7D, false)]);
        assert!(matches!(next(&rx), Some(HotkeyEvent::Down { .. })));
        send_inputs(&[key_event(VK_LSHIFT, true), key_event(0x7D, true)]);
        assert!(matches!(next(&rx), Some(HotkeyEvent::Up { .. })));
        h.update(
            &HotkeyConfig::parse("F15")
                .unwrap()
                .accept_injected_for_tests(),
        );
        send_inputs(&[key_event(0x7E, false), key_event(0x7E, true)]);
        assert!(matches!(next(&rx), Some(HotkeyEvent::Down { .. })));
        assert!(matches!(next(&rx), Some(HotkeyEvent::Up { .. })));
    }

    #[test]
    fn swallowed_key_never_reaches_async_state() {
        let _g = LIVE.lock().unwrap_or_else(|e| e.into_inner());
        if !input_desktop_available() {
            return;
        }
        let (_h, rx) = live_hook("F16");
        send_inputs(&[key_event(0x7F, false)]);
        assert!(matches!(next(&rx), Some(HotkeyEvent::Down { .. })));
        std::thread::sleep(Duration::from_millis(50));
        let swallowed_visible = key_down_async(0x7F);
        send_inputs(&[key_event(0x7F, true)]);
        assert!(matches!(next(&rx), Some(HotkeyEvent::Up { .. })));
        send_inputs(&[key_event(0x80, false)]);
        std::thread::sleep(Duration::from_millis(50));
        let passed_visible = key_down_async(0x80);
        send_inputs(&[key_event(0x80, true)]);
        assert!(!swallowed_visible, "swallowed F16 visible in async state");
        assert!(passed_visible, "passed F17 not visible in async state");
    }

    #[test]
    fn watchdog_recovers_from_stalled_callback() {
        let _g = LIVE.lock().unwrap_or_else(|e| e.into_inner());
        if !input_desktop_available() {
            return;
        }
        let (h, rx) = live_hook("F18");
        send_inputs(&[key_event(0x81, false)]);
        assert!(matches!(next(&rx), Some(HotkeyEvent::Down { .. })));
        h.debug_stall_next_callback(1500);
        let t0 = Instant::now();
        send_inputs(&[key_event(0x82, false), key_event(0x82, true)]);
        let mut got_up = false;
        let mut reinstalled = None;
        while t0.elapsed() < Duration::from_secs(6) && reinstalled.is_none() {
            match rx.recv_timeout(Duration::from_millis(200)) {
                Ok(HotkeyEvent::Up { .. }) => got_up = true,
                Ok(HotkeyEvent::HookReinstalled { reason, .. }) => reinstalled = Some(reason),
                _ => {}
            }
        }
        send_inputs(&[key_event(0x81, true)]);
        eprintln!("watchdog: {reinstalled:?} after {:?}", t0.elapsed());
        assert!(reinstalled.is_some(), "watchdog never reinstalled");
        assert!(got_up, "no synthetic Up for the held key");
        while rx.try_recv().is_ok() {}
        send_inputs(&[key_event(0x81, false), key_event(0x81, true)]);
        let after = next(&rx);
        assert!(
            matches!(after, Some(HotkeyEvent::Down { .. })),
            "after reinstall: {after:?}"
        );
    }
}
