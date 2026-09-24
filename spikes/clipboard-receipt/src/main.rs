//! Milestone-0 spike: which clipboard read signals arrive when we paste through
//! delayed rendering into real apps. Throwaway code; findings live in README.md.

use std::cell::RefCell;
use std::sync::Mutex;
use std::sync::OnceLock;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use windows::Win32::Foundation::{CloseHandle, HANDLE, HGLOBAL, HWND, LPARAM, LRESULT, WPARAM};
use windows::Win32::System::DataExchange::{
    AddClipboardFormatListener, CloseClipboard, EmptyClipboard, EnumClipboardFormats,
    GetClipboardData, GetClipboardFormatNameW, GetClipboardOwner, GetClipboardSequenceNumber,
    GetOpenClipboardWindow, IsClipboardFormatAvailable, OpenClipboard, RegisterClipboardFormatW,
    SetClipboardData,
};
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::System::Memory::{GMEM_MOVEABLE, GlobalAlloc, GlobalLock, GlobalUnlock};
use windows::Win32::System::Threading::{
    AttachThreadInput, GetCurrentThreadId, OpenProcess, PROCESS_NAME_WIN32,
    PROCESS_QUERY_LIMITED_INFORMATION, QueryFullProcessImageNameW, Sleep,
};
use windows::Win32::UI::Input::KeyboardAndMouse::{
    GetAsyncKeyState, INPUT, INPUT_0, INPUT_KEYBOARD, KEYBD_EVENT_FLAGS, KEYBDINPUT,
    KEYEVENTF_EXTENDEDKEY, KEYEVENTF_KEYUP, KEYEVENTF_UNICODE, MAPVK_VK_TO_VSC, MapVirtualKeyW,
    SendInput, VIRTUAL_KEY,
};
use windows::Win32::UI::WindowsAndMessaging::{
    BringWindowToTop, CreateWindowExW, DefWindowProcW, DestroyWindow, DispatchMessageW,
    EnumChildWindows, EnumWindows, GetClassNameW, GetForegroundWindow, GetWindowTextW,
    GetWindowThreadProcessId, HWND_MESSAGE, IsIconic, IsWindowVisible, MSG, MWMO_INPUTAVAILABLE,
    MsgWaitForMultipleObjectsEx, PM_REMOVE, PeekMessageW, PostMessageW, QS_ALLINPUT,
    RegisterClassW, SMTO_ABORTIFHUNG, SW_MINIMIZE, SW_RESTORE, SendMessageTimeoutW,
    SetForegroundWindow, ShowWindow, TranslateMessage, WINDOW_EX_STYLE, WINDOW_STYLE, WM_APP,
    WM_CLIPBOARDUPDATE, WM_DESTROYCLIPBOARD, WM_GETTEXT, WM_GETTEXTLENGTH, WM_RENDERALLFORMATS,
    WM_RENDERFORMAT, WNDCLASSW,
};
use windows::core::{BOOL, PCWSTR, PWSTR, w};

const CF_TEXT: u32 = 1;
const CF_OEMTEXT: u32 = 7;
const CF_UNICODETEXT: u32 = 13;
const CF_LOCALE: u32 = 16;
const WM_REARM: u32 = WM_APP + 1;
const INJECT_TAG: usize = 0x5350_494B; // "SPIK"

// ---------------------------------------------------------------- options

#[derive(Clone, Copy, PartialEq, Debug)]
enum Chord {
    CtrlV,
    CtrlShiftV,
    ShiftInsert,
    None,
    /// Control: type the token with KEYEVENTF_UNICODE instead of pasting.
    Type,
}

#[derive(Debug)]
struct Opts {
    countdown: u32,
    observe: f64,
    pre_chord_ms: u64,
    chord: Chord,
    eager: bool,
    also_cf_text: bool,
    probe_cf_text: bool,
    no_exclude: bool,
    rearm: bool,
    focus_class: Option<String>,
    focus_exe: Option<String>,
    focus_title: Option<String>,
    write_before_focus: bool,
    verify: bool,
    label: String,
}

fn parse_args() -> Opts {
    let mut o = Opts {
        countdown: 4,
        observe: 10.0,
        pre_chord_ms: 300,
        chord: Chord::CtrlV,
        eager: false,
        also_cf_text: false,
        probe_cf_text: false,
        no_exclude: false,
        rearm: false,
        focus_class: None,
        focus_exe: None,
        focus_title: None,
        write_before_focus: false,
        verify: false,
        label: String::new(),
    };
    let mut it = std::env::args().skip(1);
    while let Some(a) = it.next() {
        let mut val = || {
            it.next()
                .unwrap_or_else(|| die(&format!("{a} needs a value")))
        };
        match a.as_str() {
            "--countdown" => o.countdown = val().parse().unwrap_or_else(|_| die("bad --countdown")),
            "--observe" => o.observe = val().parse().unwrap_or_else(|_| die("bad --observe")),
            "--pre-chord-ms" => {
                o.pre_chord_ms = val().parse().unwrap_or_else(|_| die("bad --pre-chord-ms"))
            }
            "--chord" => {
                o.chord = match val().as_str() {
                    "ctrl-v" => Chord::CtrlV,
                    "ctrl-shift-v" => Chord::CtrlShiftV,
                    "shift-insert" => Chord::ShiftInsert,
                    "none" => Chord::None,
                    "type" => Chord::Type,
                    other => die(&format!("unknown chord {other}")),
                }
            }
            "--eager" => o.eager = true,
            "--also-cf-text" => o.also_cf_text = true,
            "--probe-cf-text" => o.probe_cf_text = true,
            "--no-exclude" => o.no_exclude = true,
            "--rearm" => o.rearm = true,
            "--focus-class" => o.focus_class = Some(val()),
            "--focus-exe" => o.focus_exe = Some(val().to_lowercase()),
            "--focus-title" => o.focus_title = Some(val()),
            "--verify" => o.verify = true,
            "--write-before-focus" => o.write_before_focus = true,
            "--label" => o.label = val(),
            "-h" | "--help" => {
                println!(
                    "clipboard-receipt [--countdown N] [--observe SECS] [--pre-chord-ms MS]\n  \
                     [--chord ctrl-v|ctrl-shift-v|shift-insert|none|type] [--eager] [--also-cf-text]\n  \
                     [--probe-cf-text] [--no-exclude] [--rearm]\n  \
                     [--focus-class CLASS] [--focus-exe NAME.exe] [--focus-title SUBSTR]\n  \
                     [--write-before-focus] [--verify] [--label TEXT]"
                );
                std::process::exit(0);
            }
            other => die(&format!("unknown argument {other}")),
        }
    }
    o
}

fn die(msg: &str) -> ! {
    eprintln!("error: {msg}");
    std::process::exit(2)
}

// ---------------------------------------------------------------- event log

#[derive(Clone, Debug)]
struct Event {
    at: Instant,
    what: String,
    format: Option<u32>,
    seq: u32,
    reader: String,
}

static EVENTS: Mutex<Vec<Event>> = Mutex::new(Vec::new());
static CHORD_AT: Mutex<Option<Instant>> = Mutex::new(None);
static START: OnceLock<Instant> = OnceLock::new();
static WRITE_CLOSED: AtomicBool = AtomicBool::new(false);

fn log(what: impl Into<String>, format: Option<u32>, reader: String) {
    let e = Event {
        at: Instant::now(),
        what: what.into(),
        format,
        seq: unsafe { GetClipboardSequenceNumber() },
        reader,
    };
    EVENTS.lock().unwrap().push(e);
}

fn reference() -> Instant {
    CHORD_AT
        .lock()
        .unwrap()
        .unwrap_or(*START.get().expect("start set"))
}

fn rel_ms(at: Instant) -> f64 {
    let r = reference();
    if at >= r {
        (at - r).as_secs_f64() * 1000.0
    } else {
        -((r - at).as_secs_f64() * 1000.0)
    }
}

// ---------------------------------------------------------------- owner state

struct State {
    hwnd: HWND,
    nonce: u64,
    eager: bool,
    also_cf_text: bool,
    exclude: bool,
    rearm: bool,
    rearm_count: u32,
    rearm_attempts: u32,
    renders: Vec<(u32, String)>,
    our_seq: u32,
}

thread_local! {
    static STATE: RefCell<Option<State>> = const { RefCell::new(None) };
}

fn with_state<R>(f: impl FnOnce(&mut State) -> R) -> R {
    STATE.with(|s| f(s.borrow_mut().as_mut().expect("state initialised")))
}

fn fmt_name(f: u32) -> String {
    match f {
        CF_TEXT => "CF_TEXT".into(),
        CF_OEMTEXT => "CF_OEMTEXT".into(),
        CF_UNICODETEXT => "CF_UNICODETEXT".into(),
        CF_LOCALE => "CF_LOCALE".into(),
        _ => {
            let mut buf = [0u16; 128];
            let n = unsafe { GetClipboardFormatNameW(f, &mut buf) };
            if n > 0 {
                String::from_utf16_lossy(&buf[..n as usize])
            } else {
                format!("#{f}")
            }
        }
    }
}

fn exe_of_pid(pid: u32) -> String {
    unsafe {
        let Ok(h) = OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false, pid) else {
            return format!("pid{pid}(no access)");
        };
        let mut buf = [0u16; 512];
        let mut len = buf.len() as u32;
        let r =
            QueryFullProcessImageNameW(h, PROCESS_NAME_WIN32, PWSTR(buf.as_mut_ptr()), &mut len);
        let _ = CloseHandle(h);
        match r {
            Ok(()) => {
                let full = String::from_utf16_lossy(&buf[..len as usize]);
                full.rsplit('\\').next().unwrap_or(&full).to_string()
            }
            Err(_) => format!("pid{pid}(?)"),
        }
    }
}

fn window_desc(hwnd: HWND) -> String {
    if hwnd.0.is_null() {
        return "null".into();
    }
    let mut pid = 0u32;
    unsafe { GetWindowThreadProcessId(hwnd, Some(&mut pid)) };
    format!("{} pid{} hwnd{:#x}", exe_of_pid(pid), pid, hwnd.0 as usize)
}

fn open_clipboard_reader() -> String {
    match unsafe { GetOpenClipboardWindow() } {
        Ok(h) => window_desc(h),
        Err(_) => "null (reader opened with NULL hwnd)".into(),
    }
}

fn owner_desc() -> String {
    match unsafe { GetClipboardOwner() } {
        Ok(h) => window_desc(h),
        Err(_) => "null".into(),
    }
}

fn hglobal_from_bytes(bytes: &[u8]) -> HANDLE {
    unsafe {
        let h: HGLOBAL = GlobalAlloc(GMEM_MOVEABLE, bytes.len()).expect("GlobalAlloc");
        let p = GlobalLock(h) as *mut u8;
        std::ptr::copy_nonoverlapping(bytes.as_ptr(), p, bytes.len());
        let _ = GlobalUnlock(h);
        HANDLE(h.0)
    }
}

fn text_bytes(format: u32, text: &str) -> Vec<u8> {
    if format == CF_UNICODETEXT {
        text.encode_utf16()
            .chain(std::iter::once(0))
            .flat_map(|u| u.to_le_bytes())
            .collect()
    } else {
        let mut b: Vec<u8> = text.bytes().collect();
        b.push(0);
        b
    }
}

fn render_text(nonce: u64) -> String {
    format!("spike paste t+{:.0}ms run{nonce}", rel_ms(Instant::now()))
}

/// Clipboard must already be open by us. Writes the entry according to mode.
fn put_entry(st: &State) {
    unsafe {
        EmptyClipboard().expect("EmptyClipboard");
        let formats: &[u32] = if st.also_cf_text {
            &[CF_UNICODETEXT, CF_TEXT]
        } else {
            &[CF_UNICODETEXT]
        };
        for &f in formats {
            let data = if st.eager {
                let t = format!("spike paste eager run{}", st.nonce);
                Some(hglobal_from_bytes(&text_bytes(f, &t)))
            } else {
                None
            };
            let delayed = data.is_none();
            // A delayed-render registration returns NULL by design; the windows crate maps
            // that to Err with a stale last-error, so only eager failures are real.
            if let Err(e) = SetClipboardData(f, data)
                && !delayed
            {
                println!("SetClipboardData({}) failed: {e}", fmt_name(f));
            }
        }
        if st.exclude {
            for name in [
                w!("ExcludeClipboardContentFromMonitorProcessing"),
                w!("CanIncludeInClipboardHistory"),
                w!("CanUploadToCloudClipboard"),
            ] {
                let f = RegisterClipboardFormatW(name);
                let zero = 0u32.to_le_bytes();
                if let Err(e) = SetClipboardData(f, Some(hglobal_from_bytes(&zero))) {
                    println!("SetClipboardData({}) failed: {e}", fmt_name(f));
                }
            }
        }
    }
}

unsafe extern "system" fn wndproc(hwnd: HWND, msg: u32, wp: WPARAM, lp: LPARAM) -> LRESULT {
    match msg {
        WM_RENDERFORMAT => {
            let format = wp.0 as u32;
            let reader = open_clipboard_reader();
            log("WM_RENDERFORMAT", Some(format), reader.clone());
            let (nonce, rearm) = with_state(|s| (s.nonce, s.rearm));
            let text = render_text(nonce);
            // Inside WM_RENDERFORMAT the reader holds the clipboard open; we must not open it.
            let r = unsafe {
                SetClipboardData(format, Some(hglobal_from_bytes(&text_bytes(format, &text))))
            };
            if let Err(e) = r {
                log(format!("render failed: {e}"), Some(format), String::new());
            }
            with_state(|s| s.renders.push((format, text)));
            if rearm {
                let _ = unsafe { PostMessageW(Some(hwnd), WM_REARM, WPARAM(0), LPARAM(0)) };
            }
            LRESULT(0)
        }
        WM_RENDERALLFORMATS => {
            log("WM_RENDERALLFORMATS", None, String::new());
            unsafe {
                if OpenClipboard(Some(hwnd)).is_ok() {
                    if GetClipboardOwner().ok() == Some(hwnd) {
                        let (nonce, cf_text) = with_state(|s| (s.nonce, s.also_cf_text));
                        let text = render_text(nonce);
                        let _ = SetClipboardData(
                            CF_UNICODETEXT,
                            Some(hglobal_from_bytes(&text_bytes(CF_UNICODETEXT, &text))),
                        );
                        if cf_text {
                            let _ = SetClipboardData(
                                CF_TEXT,
                                Some(hglobal_from_bytes(&text_bytes(CF_TEXT, &text))),
                            );
                        }
                    }
                    let _ = CloseClipboard();
                }
            }
            LRESULT(0)
        }
        WM_DESTROYCLIPBOARD => {
            log("WM_DESTROYCLIPBOARD", None, open_clipboard_reader());
            LRESULT(0)
        }
        WM_CLIPBOARDUPDATE => {
            log(
                "WM_CLIPBOARDUPDATE",
                None,
                format!("owner={}", owner_desc()),
            );
            LRESULT(0)
        }
        WM_REARM => {
            let ok = unsafe { OpenClipboard(Some(hwnd)).is_ok() };
            if !ok {
                let attempts = with_state(|s| {
                    s.rearm_attempts += 1;
                    s.rearm_attempts
                });
                if attempts < 500 {
                    unsafe { Sleep(2) };
                    let _ = unsafe { PostMessageW(Some(hwnd), WM_REARM, WPARAM(0), LPARAM(0)) };
                }
                return LRESULT(0);
            }
            with_state(|s| {
                s.rearm_attempts = 0;
                if s.rearm_count < 20 {
                    put_entry(s);
                    s.rearm_count += 1;
                }
            });
            unsafe {
                let _ = CloseClipboard();
            }
            let seq = unsafe { GetClipboardSequenceNumber() };
            with_state(|s| s.our_seq = seq);
            log("rearmed (delayed entry re-set)", None, String::new());
            LRESULT(0)
        }
        _ => unsafe { DefWindowProcW(hwnd, msg, wp, lp) },
    }
}

fn pump_for(d: Duration) {
    let deadline = Instant::now() + d;
    loop {
        let now = Instant::now();
        if now >= deadline {
            break;
        }
        let remaining = (deadline - now).as_millis().min(u32::MAX as u128) as u32;
        unsafe {
            MsgWaitForMultipleObjectsEx(None, remaining.max(1), QS_ALLINPUT, MWMO_INPUTAVAILABLE);
            let mut msg = MSG::default();
            while PeekMessageW(&mut msg, None, 0, 0, PM_REMOVE).as_bool() {
                let _ = TranslateMessage(&msg);
                DispatchMessageW(&msg);
            }
        }
    }
}

// ---------------------------------------------------------------- target window

struct FindCtx {
    class: Option<String>,
    exe: Option<String>,
    title: Option<String>,
    found: Option<HWND>,
}

fn class_of(hwnd: HWND) -> String {
    let mut buf = [0u16; 256];
    let n = unsafe { GetClassNameW(hwnd, &mut buf) };
    String::from_utf16_lossy(&buf[..n.max(0) as usize])
}

unsafe extern "system" fn find_cb(hwnd: HWND, lp: LPARAM) -> BOOL {
    let ctx = unsafe { &mut *(lp.0 as *mut FindCtx) };
    if !unsafe { IsWindowVisible(hwnd) }.as_bool() {
        return BOOL(1);
    }
    if let Some(c) = &ctx.class
        && &class_of(hwnd) != c
    {
        return BOOL(1);
    }
    if let Some(e) = &ctx.exe {
        let mut pid = 0u32;
        unsafe { GetWindowThreadProcessId(hwnd, Some(&mut pid)) };
        if exe_of_pid(pid).to_lowercase() != *e {
            return BOOL(1);
        }
    }
    if let Some(t) = &ctx.title {
        let mut buf = [0u16; 512];
        let n = unsafe { GetWindowTextW(hwnd, &mut buf) };
        if !String::from_utf16_lossy(&buf[..n.max(0) as usize]).contains(t.as_str()) {
            return BOOL(1);
        }
    }
    ctx.found = Some(hwnd);
    BOOL(0)
}

fn find_target(class: Option<String>, exe: Option<String>, title: Option<String>) -> Option<HWND> {
    let mut ctx = FindCtx {
        class,
        exe,
        title,
        found: None,
    };
    unsafe {
        let _ = EnumWindows(Some(find_cb), LPARAM(&mut ctx as *mut _ as isize));
    }
    ctx.found
}

fn key(vk: u16, up: bool) -> INPUT {
    let mut flags = KEYBD_EVENT_FLAGS(0);
    if up {
        flags |= KEYEVENTF_KEYUP;
    }
    // Insert and the right-hand modifiers are extended keys; without the flag Insert
    // arrives as numpad 0 on some layouts.
    if matches!(vk, 0x2D | 0xA3 | 0xA5 | 0x5B | 0x5C) {
        flags |= KEYEVENTF_EXTENDEDKEY;
    }
    INPUT {
        r#type: INPUT_KEYBOARD,
        Anonymous: INPUT_0 {
            ki: KEYBDINPUT {
                wVk: VIRTUAL_KEY(vk),
                wScan: unsafe { MapVirtualKeyW(vk as u32, MAPVK_VK_TO_VSC) } as u16,
                dwFlags: flags,
                time: 0,
                dwExtraInfo: INJECT_TAG,
            },
        },
    }
}

fn send(inputs: &[INPUT]) -> u32 {
    unsafe { SendInput(inputs, std::mem::size_of::<INPUT>() as i32) }
}

fn focus(target: HWND) -> String {
    unsafe {
        if GetForegroundWindow() == target {
            return "already foreground".into();
        }
        if IsIconic(target).as_bool() {
            let _ = ShowWindow(target, SW_RESTORE);
        }
        let fg = GetForegroundWindow();
        let fg_tid = GetWindowThreadProcessId(fg, None);
        let me = GetCurrentThreadId();
        let attached = fg_tid != 0 && fg_tid != me && AttachThreadInput(me, fg_tid, true).as_bool();
        let _ = BringWindowToTop(target);
        let _ = SetForegroundWindow(target);
        if attached {
            let _ = AttachThreadInput(me, fg_tid, false);
        }
        if GetForegroundWindow() == target {
            return "AttachThreadInput+SetForegroundWindow".into();
        }
        // Being the last input source lifts the foreground lock.
        send(&[key(0x7B + 12, false), key(0x7B + 12, true)]); // F24
        let _ = SetForegroundWindow(target);
        if GetForegroundWindow() == target {
            return "F24 tap+SetForegroundWindow".into();
        }
        format!(
            "FAILED (foreground is {})",
            window_desc(GetForegroundWindow())
        )
    }
}

fn release_modifiers() -> Vec<String> {
    let mods: [(u16, &str); 8] = [
        (0xA2, "LCtrl"),
        (0xA3, "RCtrl"),
        (0xA0, "LShift"),
        (0xA1, "RShift"),
        (0xA4, "LAlt"),
        (0xA5, "RAlt"),
        (0x5B, "LWin"),
        (0x5C, "RWin"),
    ];
    let mut released = Vec::new();
    let mut ups = Vec::new();
    for (vk, name) in mods {
        if unsafe { GetAsyncKeyState(vk as i32) } as u16 & 0x8000 != 0 {
            ups.push(key(vk, true));
            released.push(name.to_string());
        }
    }
    if !ups.is_empty() {
        send(&ups);
    }
    released
}

/// Pacing between typed code units. Win11 Notepad translates a queued VK_PACKET with the
/// most recently injected character, so a batch of more than one unit comes out as the
/// last unit repeated once it stalls at a word boundary; 10 ms still lost characters.
const TYPE_GAP: Duration = Duration::from_millis(30);

/// One down+up pair per UTF-16 code unit. Sent one pair per SendInput call, never as a
/// batch: see TYPE_GAP.
fn unicode_inputs(text: &str) -> Vec<INPUT> {
    let mut v = Vec::new();
    for u in text.encode_utf16() {
        for up in [false, true] {
            let mut flags = KEYEVENTF_UNICODE;
            if up {
                flags |= KEYEVENTF_KEYUP;
            }
            v.push(INPUT {
                r#type: INPUT_KEYBOARD,
                Anonymous: INPUT_0 {
                    ki: KEYBDINPUT {
                        wVk: VIRTUAL_KEY(0),
                        wScan: u,
                        dwFlags: flags,
                        time: 0,
                        dwExtraInfo: INJECT_TAG,
                    },
                },
            });
        }
    }
    v
}

fn chord_inputs(c: Chord, nonce: u64) -> Vec<INPUT> {
    const CTRL: u16 = 0xA2;
    const SHIFT: u16 = 0xA0;
    const V: u16 = 0x56;
    const INSERT: u16 = 0x2D;
    match c {
        Chord::CtrlV => vec![
            key(CTRL, false),
            key(V, false),
            key(V, true),
            key(CTRL, true),
        ],
        Chord::CtrlShiftV => vec![
            key(CTRL, false),
            key(SHIFT, false),
            key(V, false),
            key(V, true),
            key(SHIFT, true),
            key(CTRL, true),
        ],
        Chord::ShiftInsert => vec![
            key(SHIFT, false),
            key(INSERT, false),
            key(INSERT, true),
            key(SHIFT, true),
        ],
        Chord::None => vec![],
        Chord::Type => unicode_inputs(&format!("spike typed run{nonce}")),
    }
}

// ---------------------------------------------------------------- verification

struct VerifyCtx {
    needle: String,
    hits: Vec<String>,
    seen: Vec<String>,
}

unsafe extern "system" fn verify_cb(hwnd: HWND, lp: LPARAM) -> BOOL {
    let ctx = unsafe { &mut *(lp.0 as *mut VerifyCtx) };
    let class = class_of(hwnd);
    if !class.to_lowercase().contains("edit") {
        return BOOL(1);
    }
    unsafe {
        let mut len = 0usize;
        SendMessageTimeoutW(
            hwnd,
            WM_GETTEXTLENGTH,
            WPARAM(0),
            LPARAM(0),
            SMTO_ABORTIFHUNG,
            1000,
            Some(&mut len),
        );
        let mut buf = vec![0u16; len + 1];
        let mut got = 0usize;
        SendMessageTimeoutW(
            hwnd,
            WM_GETTEXT,
            WPARAM(buf.len()),
            LPARAM(buf.as_mut_ptr() as isize),
            SMTO_ABORTIFHUNG,
            1000,
            Some(&mut got),
        );
        let text = String::from_utf16_lossy(&buf[..got.min(len)]);
        let head: String = text.chars().take(80).collect();
        ctx.seen
            .push(format!("{class}(len {len}, got {got}): {head:?}"));
        for line in text.lines() {
            if line.contains(&ctx.needle) {
                ctx.hits.push(format!("{class}: {:?}", line.trim()));
            }
        }
    }
    BOOL(1)
}

fn verify(target: HWND, nonce: u64) -> (bool, Vec<String>, Vec<String>) {
    let mut ctx = VerifyCtx {
        needle: format!("run{nonce}"),
        hits: vec![],
        seen: vec![],
    };
    unsafe {
        let _ = EnumChildWindows(
            Some(target),
            Some(verify_cb),
            LPARAM(&mut ctx as *mut _ as isize),
        );
    }
    (!ctx.hits.is_empty(), ctx.hits, ctx.seen)
}

// ---------------------------------------------------------------- CF_TEXT probe

fn probe_thread() -> Vec<String> {
    let mut out = Vec::new();
    unsafe {
        // Spin so we open the clipboard before any other listener (rdpclip in an RDP
        // session renders every delayed entry within ~1 ms of the update).
        while !WRITE_CLOSED.load(Ordering::Acquire) {
            std::hint::spin_loop();
        }
        let spin_start = Instant::now();
        let mut opened = false;
        while spin_start.elapsed() < Duration::from_secs(2) {
            if OpenClipboard(None).is_ok() {
                opened = true;
                break;
            }
        }
        if !opened {
            out.push("probe: could not open clipboard".into());
            return out;
        }
        log("probe: opened clipboard", None, String::new());
        let owner_seen_render = EVENTS
            .lock()
            .unwrap()
            .iter()
            .any(|e| e.what == "WM_RENDERFORMAT");
        out.push(format!(
            "probe opened clipboard first (no render yet): {}",
            !owner_seen_render
        ));
        for f in [CF_UNICODETEXT, CF_TEXT, CF_OEMTEXT, CF_LOCALE] {
            out.push(format!(
                "IsClipboardFormatAvailable({}) = {}",
                fmt_name(f),
                IsClipboardFormatAvailable(f).is_ok()
            ));
        }
        let mut fmts = Vec::new();
        let mut f = 0u32;
        loop {
            f = EnumClipboardFormats(f);
            if f == 0 {
                break;
            }
            fmts.push(fmt_name(f));
        }
        out.push(format!(
            "EnumClipboardFormats (before any read) = [{}]",
            fmts.join(", ")
        ));
        log(
            "probe: GetClipboardData(CF_TEXT) call",
            Some(CF_TEXT),
            String::new(),
        );
        match GetClipboardData(CF_TEXT) {
            Ok(h) => {
                let p = GlobalLock(HGLOBAL(h.0)) as *const u8;
                let s = if p.is_null() {
                    "<lock failed>".to_string()
                } else {
                    let s = std::ffi::CStr::from_ptr(p as *const i8)
                        .to_string_lossy()
                        .into_owned();
                    let _ = GlobalUnlock(HGLOBAL(h.0));
                    s
                };
                out.push(format!("GetClipboardData(CF_TEXT) = Ok {s:?}"));
            }
            Err(e) => out.push(format!("GetClipboardData(CF_TEXT) = Err {e}")),
        }
        log(
            "probe: GetClipboardData(CF_TEXT) returned",
            Some(CF_TEXT),
            String::new(),
        );
        match GetClipboardData(CF_UNICODETEXT) {
            Ok(_) => out.push("GetClipboardData(CF_UNICODETEXT) afterwards = Ok".into()),
            Err(e) => out.push(format!(
                "GetClipboardData(CF_UNICODETEXT) afterwards = Err {e}"
            )),
        }
        log(
            "probe: GetClipboardData(CF_UNICODETEXT) returned",
            Some(CF_UNICODETEXT),
            String::new(),
        );
        let _ = CloseClipboard();
    }
    out
}

// ---------------------------------------------------------------- main

fn main() {
    let o = parse_args();
    START.set(Instant::now()).ok();
    let nonce = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
        % 1_000_000;

    let hwnd = unsafe {
        let inst = GetModuleHandleW(PCWSTR::null()).expect("module");
        let class = w!("ClipboardReceiptSpike");
        let wc = WNDCLASSW {
            lpfnWndProc: Some(wndproc),
            hInstance: inst.into(),
            lpszClassName: class,
            ..Default::default()
        };
        RegisterClassW(&wc);
        CreateWindowExW(
            WINDOW_EX_STYLE(0),
            class,
            w!("clipboard-receipt"),
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
        .expect("CreateWindowExW")
    };
    unsafe { AddClipboardFormatListener(hwnd).expect("AddClipboardFormatListener") };
    STATE.with(|s| {
        *s.borrow_mut() = Some(State {
            hwnd,
            nonce,
            eager: o.eager,
            also_cf_text: o.also_cf_text,
            exclude: !o.no_exclude,
            rearm: o.rearm,
            rearm_count: 0,
            rearm_attempts: 0,
            renders: vec![],
            our_seq: 0,
        })
    });

    println!(
        "clipboard-receipt label={:?} run{nonce} opts={o:?}",
        o.label
    );
    for i in (1..=o.countdown).rev() {
        println!("  pasting in {i}...");
        pump_for(Duration::from_secs(1));
    }

    let mut target = None;
    let mut focus_result = String::from("no target requested");
    if o.focus_class.is_some() || o.focus_exe.is_some() || o.focus_title.is_some() {
        target = find_target(
            o.focus_class.clone(),
            o.focus_exe.clone(),
            o.focus_title.clone(),
        );
        match target {
            Some(t) if o.write_before_focus => {
                // Deactivate the target so that its activation happens after our write.
                unsafe {
                    let _ = ShowWindow(t, SW_MINIMIZE);
                }
                pump_for(Duration::from_millis(400));
                focus_result = "deferred until after the write".into();
            }
            Some(t) => {
                focus_result = focus(t);
                pump_for(Duration::from_millis(250));
            }
            None => die("target window not found"),
        }
    }
    println!("focus: {focus_result}");

    let probe_handle = o.probe_cf_text.then(|| std::thread::spawn(probe_thread));
    let seq_before = unsafe { GetClipboardSequenceNumber() };
    log("clipboard write begins", None, String::new());
    unsafe {
        let mut opened = false;
        for _ in 0..100 {
            if OpenClipboard(Some(hwnd)).is_ok() {
                opened = true;
                break;
            }
            Sleep(5);
        }
        if !opened {
            die("could not open clipboard");
        }
        with_state(|s| put_entry(s));
        CloseClipboard().expect("CloseClipboard");
        WRITE_CLOSED.store(true, Ordering::Release);
    }
    let seq_after_write = unsafe { GetClipboardSequenceNumber() };
    with_state(|s| s.our_seq = seq_after_write);
    log("clipboard write closed", None, String::new());
    if let (true, Some(t)) = (o.write_before_focus, target) {
        pump_for(Duration::from_millis(300));
        let r = focus(t);
        log(format!("target focused ({r})"), None, String::new());
        println!("focus after write: {r}");
    }

    let mut probe_out = Vec::new();
    if let Some(h) = probe_handle {
        let deadline = Instant::now() + Duration::from_secs_f64(o.observe);
        while !h.is_finished() && Instant::now() < deadline {
            pump_for(Duration::from_millis(20));
        }
        pump_for(Duration::from_millis(300));
        if h.is_finished() {
            probe_out = h.join().unwrap();
        } else {
            probe_out.push("probe thread did not finish (deadlock?)".into());
        }
    } else {
        pump_for(Duration::from_millis(o.pre_chord_ms));
        let fg_ok = target.map(|t| unsafe { GetForegroundWindow() } == t);
        let released = release_modifiers();
        // Never inject a paste into whatever else happens to be foreground.
        let inputs = if fg_ok == Some(false) {
            vec![]
        } else {
            chord_inputs(o.chord, nonce)
        };
        if fg_ok == Some(false) {
            println!("ABORT chord: target is not foreground");
        }
        *CHORD_AT.lock().unwrap() = Some(Instant::now());
        let sent = if inputs.is_empty() {
            0
        } else if o.chord == Chord::Type {
            let mut n = 0;
            for (i, pair) in inputs.chunks(2).enumerate() {
                if i > 0 {
                    pump_for(TYPE_GAP);
                }
                if target.is_some_and(|t| unsafe { GetForegroundWindow() } != t) {
                    println!("ABORT typing: target lost foreground after {i} units");
                    break;
                }
                n += send(pair);
            }
            n
        } else {
            send(&inputs)
        };
        log(
            format!("chord {:?} sent ({sent}/{} events)", o.chord, inputs.len()),
            None,
            String::new(),
        );
        println!(
            "chord {:?}: foreground==target: {fg_ok:?}; released modifiers: {released:?}",
            o.chord
        );
        pump_for(Duration::from_secs_f64(o.observe));
    }

    // ------------------------------------------------ report
    let final_seq = unsafe { GetClipboardSequenceNumber() };
    let owner_is_us = unsafe { GetClipboardOwner() }.ok() == Some(hwnd);
    let (renders, our_seq, rearms) = with_state(|s| (s.renders.clone(), s.our_seq, s.rearm_count));
    let ucs_avail = unsafe { IsClipboardFormatAvailable(CF_UNICODETEXT) }.is_ok();
    let entry_state = if !owner_is_us {
        format!("replaced (owner now {})", owner_desc())
    } else if final_seq != our_seq {
        "ours but sequence moved (someone rendered-into or re-wrote?)".into()
    } else if o.eager {
        "ours, eager data".into()
    } else if renders.is_empty() || rearms > 0 && renders.len() as u32 == rearms {
        "ours, delayed entry still unrendered".into()
    } else {
        "ours, rendered (no longer delayed)".into()
    };

    let events = EVENTS.lock().unwrap().clone();
    let has_chord = CHORD_AT.lock().unwrap().is_some();
    println!();
    println!(
        "timeline (ms relative to {}):",
        if has_chord {
            "chord SendInput"
        } else {
            "program start"
        }
    );
    println!(
        "{:>10}  {:<48} {:<16} {:>6}  reader/owner",
        "t_ms", "event", "format", "seq"
    );
    for e in &events {
        println!(
            "{:>10.1}  {:<48} {:<16} {:>6}  {}",
            rel_ms(e.at),
            e.what,
            e.format.map(fmt_name).unwrap_or_default(),
            e.seq,
            e.reader
        );
    }
    println!();
    for (f, t) in &renders {
        println!("rendered {}: {t:?}", fmt_name(*f));
    }
    for l in &probe_out {
        println!("probe: {l}");
    }
    let render_events: Vec<&Event> = events
        .iter()
        .filter(|e| e.what == "WM_RENDERFORMAT")
        .collect();
    let first_render = render_events.first().map(|e| rel_ms(e.at));
    let last_render = render_events.last().map(|e| rel_ms(e.at));
    let pre_chord_renders = render_events.iter().filter(|e| rel_ms(e.at) < 0.0).count();
    let mut readers: Vec<String> = render_events.iter().map(|e| e.reader.clone()).collect();
    readers.dedup();
    let formats: Vec<String> = render_events
        .iter()
        .map(|e| e.format.map(fmt_name).unwrap_or_default())
        .collect();

    let landed = if o.verify {
        match target {
            Some(t) => {
                let (hit, hits, seen) = verify(t, nonce);
                for h in &hits {
                    println!("verify hit: {h}");
                }
                println!("verify scanned edit children: {seen:?}");
                if seen.is_empty() {
                    "unverifiable (no edit child)".to_string()
                } else {
                    hit.to_string()
                }
            }
            None => "no target".into(),
        }
    } else {
        "not checked".into()
    };

    println!(
        "seq before write {seq_before}, after write {seq_after_write}, final {final_seq}; rearms {rearms}"
    );
    println!("final clipboard: {entry_state}; CF_UNICODETEXT available: {ucs_avail}");
    println!(
        "SUMMARY label={:?} first_render_ms={} last_render_ms={} renders={} pre_chord_renders={} formats=[{}] readers=[{}] landed={} final={:?}",
        o.label,
        first_render
            .map(|v| format!("{v:.1}"))
            .unwrap_or("-".into()),
        last_render.map(|v| format!("{v:.1}")).unwrap_or("-".into()),
        render_events.len(),
        pre_chord_renders,
        formats.join(","),
        readers.join(" | "),
        landed,
        entry_state
    );

    // Destroying the owner with pending delayed formats triggers WM_RENDERALLFORMATS.
    let before = EVENTS.lock().unwrap().len();
    unsafe {
        let _ = DestroyWindow(with_state(|s| s.hwnd));
    }
    pump_for(Duration::from_millis(50));
    for e in EVENTS.lock().unwrap().iter().skip(before) {
        println!("at exit: {:.1} ms {} {}", rel_ms(e.at), e.what, e.reader);
    }
}
