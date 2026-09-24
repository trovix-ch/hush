//! Spike: type text into a real app with `SendInput` + `KEYEVENTF_UNICODE`, read it back,
//! and report exactly what arrived. Throwaway code; findings live in README.md.

use std::sync::atomic::{AtomicIsize, Ordering};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use windows::Win32::Foundation::{GetLastError, HWND, LPARAM, LRESULT, WPARAM};
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::System::Threading::{AttachThreadInput, GetCurrentThreadId};
use windows::Win32::UI::Input::KeyboardAndMouse::{
    GetAsyncKeyState, INPUT, INPUT_0, INPUT_KEYBOARD, KEYBD_EVENT_FLAGS, KEYBDINPUT,
    KEYEVENTF_EXTENDEDKEY, KEYEVENTF_KEYUP, KEYEVENTF_UNICODE, MAPVK_VK_TO_VSC, MapVirtualKeyW,
    SendInput, SetFocus, VIRTUAL_KEY, VkKeyScanW,
};
use windows::Win32::UI::WindowsAndMessaging::{
    BringWindowToTop, CreateWindowExW, DefWindowProcW, DispatchMessageW, EnumChildWindows,
    EnumWindows, GUITHREADINFO, GetClassNameW, GetForegroundWindow, GetGUIThreadInfo, GetMessageW,
    GetWindowTextW, GetWindowThreadProcessId, IsIconic, IsWindowVisible, MSG, RegisterClassW,
    SMTO_ABORTIFHUNG, SW_RESTORE, SendMessageTimeoutW, SetForegroundWindow, ShowWindow,
    TranslateMessage, WINDOW_EX_STYLE, WINDOW_STYLE, WM_GETTEXT, WM_GETTEXTLENGTH, WM_NULL,
    WM_SETFOCUS, WM_SETTEXT, WNDCLASSW, WS_CHILD, WS_OVERLAPPEDWINDOW, WS_VISIBLE, WS_VSCROLL,
};
use windows::core::{BOOL, PCWSTR, w};

const TAG: usize = 0x5459_5045; // "TYPE": lets our own hook recognise injected input.
const VK_RETURN: u16 = 0x0D;

// ---------------------------------------------------------------- options

#[derive(Clone, Copy, PartialEq, Debug)]
enum Newline {
    /// `\n` becomes a VK_RETURN press; `\r` is dropped so `\r\n` is one Return.
    Return,
    /// `\n` and `\r` are sent as Unicode code units like any other character.
    Unicode,
    /// `\n` and `\r` are not sent at all.
    Skip,
}

#[derive(Clone, Copy, PartialEq, Debug)]
enum Target {
    Notepad,
    /// A plain EDIT control in a window owned by this process.
    Edit,
}

#[derive(Clone, Copy, Debug)]
struct Mode {
    /// Characters per SendInput call; 0 = everything in one call.
    chunk: usize,
    delay_ms: u64,
    /// WM_NULL round trips to the target's focus window after each SendInput call.
    sync: u8,
    /// Send characters the active layout has a key for as that key (+Shift), not VK_PACKET.
    vk: bool,
    /// Read the target back after each SendInput call and wait until the chunk has
    /// arrived. Measurement only: the product cannot read arbitrary targets.
    confirm: bool,
    /// Group per UTF-16 unit instead of per character, so --chunk 1 splits surrogate pairs.
    per_unit: bool,
}

struct Opts {
    text: Option<String>,
    suite: bool,
    mode: Mode,
    newline: Newline,
    target: Target,
    repeat: usize,
}

fn die(msg: &str) -> ! {
    eprintln!("error: {msg}");
    std::process::exit(2);
}

fn usage() -> ! {
    println!(
        "typing <text> [--chunk N] [--delay-ms N] [--newline return|unicode|skip] \
         [--target notepad|edit] [--repeat N]\n\
         typing --suite [--target notepad|edit] [--newline ...]\n\n\
         <text> may contain the escapes \\n, \\r, \\t and \\\\.\n\
         --chunk N     characters (not code units) per SendInput call; 0 = one call (default)\n\
         --delay-ms N  sleep between SendInput calls (default 0)\n\
         --newline     how \\n is sent (default return: a VK_RETURN press)\n\
         --suite       run the built-in cases across the built-in chunk/delay modes\n\
         --case NAME   type one built-in case: repro|ascii|intl|emoji|newlines|long600\n\
         --per-unit    one UTF-16 unit per group, so --chunk 1 splits surrogate pairs\n\
         --confirm     read the target back after each call and wait for the chunk to land\n\
         --sync N      N WM_NULL round trips to the focus window after each call\n\
         --vk          send layout-mapped keys (+Shift) instead of VK_PACKET where possible"
    );
    std::process::exit(0);
}

fn unescape(s: &str) -> String {
    let mut out = String::new();
    let mut it = s.chars();
    while let Some(c) = it.next() {
        if c != '\\' {
            out.push(c);
            continue;
        }
        match it.next() {
            Some('n') => out.push('\n'),
            Some('r') => out.push('\r'),
            Some('t') => out.push('\t'),
            Some(o) => out.push(o),
            None => out.push('\\'),
        }
    }
    out
}

fn parse() -> Opts {
    let mut o = Opts {
        text: None,
        suite: false,
        mode: Mode {
            chunk: 0,
            delay_ms: 0,
            sync: 0,
            vk: false,
            confirm: false,
            per_unit: false,
        },
        newline: Newline::Return,
        target: Target::Notepad,
        repeat: 1,
    };
    let mut args = std::env::args().skip(1);
    let num = |v: Option<String>, name: &str| -> u64 {
        v.and_then(|s| s.parse().ok())
            .unwrap_or_else(|| die(&format!("{name} needs a number")))
    };
    while let Some(a) = args.next() {
        match a.as_str() {
            "-h" | "--help" => usage(),
            "--suite" => o.suite = true,
            "--chunk" => o.mode.chunk = num(args.next(), "--chunk") as usize,
            "--delay-ms" => o.mode.delay_ms = num(args.next(), "--delay-ms"),
            "--sync" => o.mode.sync = num(args.next(), "--sync") as u8,
            "--vk" => o.mode.vk = true,
            "--confirm" => o.mode.confirm = true,
            "--per-unit" => o.mode.per_unit = true,
            "--case" => {
                let name = args.next().unwrap_or_default();
                o.text = Some(
                    builtin_cases()
                        .into_iter()
                        .find(|(n, _)| *n == name)
                        .map(|(_, t)| t)
                        .unwrap_or_else(|| die("--case repro|ascii|intl|emoji|newlines|long600")),
                );
            }
            "--repeat" => o.repeat = num(args.next(), "--repeat").max(1) as usize,
            "--newline" => {
                o.newline = match args.next().as_deref() {
                    Some("return") => Newline::Return,
                    Some("unicode") => Newline::Unicode,
                    Some("skip") => Newline::Skip,
                    _ => die("--newline return|unicode|skip"),
                }
            }
            "--target" => {
                o.target = match args.next().as_deref() {
                    Some("notepad") => Target::Notepad,
                    Some("edit") => Target::Edit,
                    _ => die("--target notepad|edit"),
                }
            }
            s if s.starts_with("--") => die(&format!("unknown flag {s}")),
            _ => {
                if o.text.is_some() {
                    die("only one <text> argument (quote it)");
                }
                o.text = Some(unescape(&a));
            }
        }
    }
    if o.text.is_none() && !o.suite {
        usage();
    }
    o
}

// ---------------------------------------------------------------- input building

fn kbd(vk: u16, scan: u16, flags: KEYBD_EVENT_FLAGS) -> INPUT {
    INPUT {
        r#type: INPUT_KEYBOARD,
        Anonymous: INPUT_0 {
            ki: KEYBDINPUT {
                wVk: VIRTUAL_KEY(vk),
                wScan: scan,
                dwFlags: flags,
                time: 0,
                dwExtraInfo: TAG,
            },
        },
    }
}

fn vk_press(vk: u16, up: bool) -> INPUT {
    let mut flags = KEYBD_EVENT_FLAGS(0);
    if up {
        flags |= KEYEVENTF_KEYUP;
    }
    // Insert and the right-hand modifiers are extended keys; without the flag they arrive
    // as their numpad / left-hand twins.
    if matches!(vk, 0x2D | 0xA3 | 0xA5 | 0x5B | 0x5C) {
        flags |= KEYEVENTF_EXTENDEDKEY;
    }
    let scan = unsafe { MapVirtualKeyW(vk as u32, MAPVK_VK_TO_VSC) } as u16;
    kbd(vk, scan, flags)
}

/// One group per character: the events for a character must never be split across
/// SendInput calls, or a surrogate pair can arrive with other input between its halves.
fn layout_key(c: char) -> Option<Vec<INPUT>> {
    let mut buf = [0u16; 2];
    let units = c.encode_utf16(&mut buf);
    if units.len() != 1 || c.is_control() {
        return None;
    }
    let r = unsafe { VkKeyScanW(units[0]) };
    if r == -1 {
        return None;
    }
    let vk = (r as u16) & 0xFF;
    let shift = (r as u16) >> 8;
    // Ctrl/Alt combinations (AltGr characters) are left to VK_PACKET: injecting AltGr
    // collides with physically held modifiers and with app shortcuts.
    if shift & !1 != 0 {
        return None;
    }
    let mut g = Vec::new();
    if shift & 1 != 0 {
        g.push(vk_press(0xA0, false));
    }
    g.push(vk_press(vk, false));
    g.push(vk_press(vk, true));
    if shift & 1 != 0 {
        g.push(vk_press(0xA0, true));
    }
    Some(g)
}

fn char_groups(text: &str, nl: Newline, vk: bool) -> Vec<Vec<INPUT>> {
    let mut groups = Vec::new();
    for c in text.chars() {
        if vk && let Some(g) = layout_key(c) {
            groups.push(g);
            continue;
        }
        if matches!(c, '\n' | '\r') {
            match nl {
                Newline::Skip => continue,
                Newline::Return if c == '\r' => continue,
                Newline::Return => {
                    groups.push(vec![vk_press(VK_RETURN, false), vk_press(VK_RETURN, true)]);
                    continue;
                }
                Newline::Unicode => {}
            }
        }
        let mut buf = [0u16; 2];
        let mut g = Vec::new();
        // Down then up per code unit, as enigo does: a pair's high unit is pressed and
        // released before the low one.
        for &u in c.encode_utf16(&mut buf).iter() {
            g.push(kbd(0, u, KEYEVENTF_UNICODE));
            g.push(kbd(0, u, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP));
        }
        groups.push(g);
    }
    groups
}

fn expected_text(text: &str, nl: Newline) -> String {
    match nl {
        Newline::Return => text.replace('\r', ""),
        Newline::Skip => text.replace(['\r', '\n'], ""),
        Newline::Unicode => text.to_string(),
    }
}

fn send(inputs: &[INPUT]) -> (u32, u32) {
    let n = unsafe { SendInput(inputs, std::mem::size_of::<INPUT>() as i32) };
    let err = if n as usize != inputs.len() {
        unsafe { GetLastError() }.0
    } else {
        0
    };
    (n, err)
}

fn release_modifiers() -> Vec<&'static str> {
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
            ups.push(vk_press(vk, true));
            released.push(name);
        }
    }
    if !ups.is_empty() {
        send(&ups);
    }
    released
}

// ---------------------------------------------------------------- windows

fn class_of(hwnd: HWND) -> String {
    let mut buf = [0u16; 256];
    let n = unsafe { GetClassNameW(hwnd, &mut buf) };
    String::from_utf16_lossy(&buf[..n.max(0) as usize])
}

fn title_of(hwnd: HWND) -> String {
    let mut buf = [0u16; 512];
    let n = unsafe { GetWindowTextW(hwnd, &mut buf) };
    String::from_utf16_lossy(&buf[..n.max(0) as usize])
}

struct Find {
    needle: String,
    found: Option<HWND>,
}

unsafe extern "system" fn find_top(hwnd: HWND, lp: LPARAM) -> BOOL {
    let ctx = unsafe { &mut *(lp.0 as *mut Find) };
    if unsafe { IsWindowVisible(hwnd) }.as_bool()
        && class_of(hwnd) == "Notepad"
        && title_of(hwnd).contains(&ctx.needle)
    {
        ctx.found = Some(hwnd);
        return BOOL(0);
    }
    BOOL(1)
}

unsafe extern "system" fn find_edit(hwnd: HWND, lp: LPARAM) -> BOOL {
    let ctx = unsafe { &mut *(lp.0 as *mut Find) };
    if unsafe { IsWindowVisible(hwnd) }.as_bool() && class_of(hwnd).contains(&ctx.needle) {
        ctx.found = Some(hwnd);
        return BOOL(0);
    }
    BOOL(1)
}

fn focus(top: HWND) -> bool {
    unsafe {
        if GetForegroundWindow() == top {
            return true;
        }
        if IsIconic(top).as_bool() {
            let _ = ShowWindow(top, SW_RESTORE);
        }
        let fg = GetForegroundWindow();
        let fg_tid = GetWindowThreadProcessId(fg, None);
        let me = GetCurrentThreadId();
        let attached = fg_tid != 0 && fg_tid != me && AttachThreadInput(me, fg_tid, true).as_bool();
        let _ = BringWindowToTop(top);
        let _ = SetForegroundWindow(top);
        if attached {
            let _ = AttachThreadInput(me, fg_tid, false);
        }
        if GetForegroundWindow() != top {
            // Being the last input source lifts the foreground lock.
            send(&[vk_press(0x87, false), vk_press(0x87, true)]); // F24
            let _ = SetForegroundWindow(top);
        }
        std::thread::sleep(Duration::from_millis(150));
        GetForegroundWindow() == top
    }
}

/// Blocks until the foreground thread next pumps messages. A thread that is busy (Notepad
/// after a word boundary) answers only once it is back in its message loop.
fn round_trip() -> bool {
    unsafe {
        let fg = GetForegroundWindow();
        let tid = GetWindowThreadProcessId(fg, None);
        let mut gti = GUITHREADINFO {
            cbSize: std::mem::size_of::<GUITHREADINFO>() as u32,
            ..Default::default()
        };
        let target = if GetGUIThreadInfo(tid, &mut gti).is_ok() && !gti.hwndFocus.0.is_null() {
            gti.hwndFocus
        } else {
            fg
        };
        SendMessageTimeoutW(
            target,
            WM_NULL,
            WPARAM(0),
            LPARAM(0),
            SMTO_ABORTIFHUNG,
            2000,
            None,
        )
        .0 != 0
    }
}

fn read_text(edit: HWND) -> Option<String> {
    unsafe {
        let mut len = 0usize;
        let r = SendMessageTimeoutW(
            edit,
            WM_GETTEXTLENGTH,
            WPARAM(0),
            LPARAM(0),
            SMTO_ABORTIFHUNG,
            2000,
            Some(&mut len),
        );
        if r.0 == 0 {
            return None;
        }
        let mut buf = vec![0u16; len + 1];
        let mut got = 0usize;
        let r = SendMessageTimeoutW(
            edit,
            WM_GETTEXT,
            WPARAM(buf.len()),
            LPARAM(buf.as_mut_ptr() as isize),
            SMTO_ABORTIFHUNG,
            2000,
            Some(&mut got),
        );
        if r.0 == 0 {
            return None;
        }
        Some(String::from_utf16_lossy(&buf[..got.min(len)]))
    }
}

fn clear_text(edit: HWND) {
    let empty = [0u16];
    unsafe {
        let _ = SendMessageTimeoutW(
            edit,
            WM_SETTEXT,
            WPARAM(0),
            LPARAM(empty.as_ptr() as isize),
            SMTO_ABORTIFHUNG,
            2000,
            None,
        );
    }
}

fn normalize(s: &str) -> String {
    s.replace("\r\n", "\n").replace('\r', "\n")
}

struct Tgt {
    top: HWND,
    edit: HWND,
    desc: String,
}

fn open_notepad() -> Tgt {
    // Reuse a target tab from an earlier run rather than piling up tabs.
    let mut reuse = Find {
        needle: "typing-target-".into(),
        found: None,
    };
    unsafe {
        let _ = EnumWindows(Some(find_top), LPARAM(&mut reuse as *mut _ as isize));
    }
    let name = match reuse.found {
        Some(h) => {
            title_of(h)
                .split(".txt")
                .next()
                .unwrap_or_default()
                .trim_start_matches('*')
                .to_string()
                + ".txt"
        }
        None => format!("typing-target-{}.txt", std::process::id()),
    };
    let path = std::env::temp_dir().join(&name);
    if reuse.found.is_none() {
        std::fs::write(&path, b"").unwrap_or_else(|e| die(&format!("temp file: {e}")));
        // Not waited on: the window is found by title, and Notepad stays open after we exit.
        #[allow(clippy::zombie_processes)]
        std::process::Command::new("notepad.exe")
            .arg(&path)
            // Notepad outlives us; an inherited stdout pipe would keep any `| grep` open forever.
            .stdin(std::process::Stdio::null())
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .spawn()
            .unwrap_or_else(|e| die(&format!("launch notepad: {e}")));
    }
    let deadline = Instant::now() + Duration::from_secs(15);
    let stem = name.trim_end_matches(".txt").to_string();
    loop {
        let mut f = Find {
            needle: stem.clone(),
            found: None,
        };
        unsafe {
            let _ = EnumWindows(Some(find_top), LPARAM(&mut f as *mut _ as isize));
        }
        if let Some(top) = f.found {
            std::thread::sleep(Duration::from_millis(500));
            for class in ["RichEdit", "Edit"] {
                let mut e = Find {
                    needle: class.into(),
                    found: None,
                };
                unsafe {
                    let _ = EnumChildWindows(
                        Some(top),
                        Some(find_edit),
                        LPARAM(&mut e as *mut _ as isize),
                    );
                }
                if let Some(edit) = e.found {
                    return Tgt {
                        top,
                        edit,
                        desc: format!("Notepad '{}' / {}", title_of(top), class_of(edit)),
                    };
                }
            }
        }
        if Instant::now() > deadline {
            die("Notepad window with the temp file did not appear");
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}

static OWN_EDIT: AtomicIsize = AtomicIsize::new(0);

unsafe extern "system" fn own_proc(hwnd: HWND, msg: u32, wp: WPARAM, lp: LPARAM) -> LRESULT {
    if msg == WM_SETFOCUS {
        let e = OWN_EDIT.load(Ordering::Acquire);
        if e != 0 {
            let _ = unsafe { SetFocus(Some(HWND(e as *mut _))) };
            return LRESULT(0);
        }
    }
    unsafe { DefWindowProcW(hwnd, msg, wp, lp) }
}

fn open_own_edit() -> Tgt {
    let (tx, rx) = mpsc::channel::<(isize, isize)>();
    std::thread::spawn(move || unsafe {
        let hinst = GetModuleHandleW(None).expect("GetModuleHandleW");
        let class = w!("TypingSpikeHost");
        let wc = WNDCLASSW {
            lpfnWndProc: Some(own_proc),
            hInstance: hinst.into(),
            lpszClassName: class,
            ..Default::default()
        };
        RegisterClassW(&wc);
        let top = CreateWindowExW(
            WINDOW_EX_STYLE(0),
            class,
            w!("typing spike EDIT target"),
            WS_OVERLAPPEDWINDOW | WS_VISIBLE,
            100,
            100,
            800,
            500,
            None,
            None,
            Some(hinst.into()),
            None,
        )
        .expect("CreateWindowExW top");
        // ES_MULTILINE | ES_AUTOVSCROLL | ES_WANTRETURN
        let style = WS_CHILD | WS_VISIBLE | WS_VSCROLL | WINDOW_STYLE(0x0004 | 0x0040 | 0x1000);
        let edit = CreateWindowExW(
            WINDOW_EX_STYLE(0),
            w!("EDIT"),
            PCWSTR::null(),
            style,
            0,
            0,
            780,
            460,
            Some(top),
            None,
            Some(hinst.into()),
            None,
        )
        .expect("CreateWindowExW edit");
        OWN_EDIT.store(edit.0 as isize, Ordering::Release);
        let _ = SetFocus(Some(edit));
        tx.send((top.0 as isize, edit.0 as isize)).unwrap();
        let mut msg = MSG::default();
        while GetMessageW(&mut msg, None, 0, 0).as_bool() {
            let _ = TranslateMessage(&msg);
            DispatchMessageW(&msg);
        }
    });
    let (top, edit) = rx.recv().unwrap();
    Tgt {
        top: HWND(top as *mut _),
        edit: HWND(edit as *mut _),
        desc: "own EDIT control (separate UI thread in this process)".into(),
    }
}

// ---------------------------------------------------------------- one run

struct RunResult {
    pass: bool,
    units: usize,
    events: usize,
    calls: usize,
    send_ms: f64,
    total_ms: f64,
    diff: String,
    arrive: String,
}

fn diff(exp: &str, got: &str) -> String {
    let e: Vec<char> = exp.chars().collect();
    let g: Vec<char> = got.chars().collect();
    let i = e.iter().zip(&g).take_while(|(a, b)| a == b).count();
    let tail = |v: &[char]| -> String { v.iter().skip(i).take(40).collect() };
    format!(
        "first mismatch at char {i} (expected {} chars, got {}): expected {:?} got {:?}",
        e.len(),
        g.len(),
        tail(&e),
        tail(&g)
    )
}

fn run(t: &Tgt, text: &str, mode: Mode, nl: Newline) -> RunResult {
    clear_text(t.edit);
    if !focus(t.top) {
        die("could not make the target foreground");
    }
    let released = release_modifiers();
    if !released.is_empty() {
        println!("  released held modifiers: {released:?}");
    }
    let mut groups = char_groups(text, nl, mode.vk);
    if mode.per_unit {
        groups = groups
            .into_iter()
            .flat_map(|g| {
                if g.len() == 4 && g.iter().all(|i| unsafe { i.Anonymous.ki.wVk.0 } == 0) {
                    vec![g[..2].to_vec(), g[2..].to_vec()]
                } else {
                    vec![g]
                }
            })
            .collect();
    }
    let events: usize = groups.iter().map(|g| g.len()).sum();
    let units = text.encode_utf16().count();
    let chunk = if mode.chunk == 0 {
        groups.len().max(1)
    } else {
        mode.chunk
    };
    let start = Instant::now();
    let mut calls = 0usize;
    let mut arrive: Vec<f64> = Vec::new();
    let mut failure = String::new();
    for (i, batch) in groups.chunks(chunk).enumerate() {
        if i > 0 && mode.delay_ms > 0 {
            std::thread::sleep(Duration::from_millis(mode.delay_ms));
        }
        // Never type into whatever else became foreground mid-run.
        if unsafe { GetForegroundWindow() } != t.top {
            failure = format!("foreground changed before chunk {i}");
            break;
        }
        let flat: Vec<INPUT> = batch.iter().flatten().copied().collect();
        let (n, err) = send(&flat);
        calls += 1;
        if n as usize != flat.len() {
            failure = format!("SendInput inserted {n}/{} (GetLastError {err})", flat.len());
            break;
        }
        for _ in 0..mode.sync {
            if !round_trip() {
                failure = format!("target did not answer WM_NULL within 2 s after chunk {i}");
                break;
            }
        }
        if mode.confirm && failure.is_empty() {
            let want = ((i + 1) * chunk).min(groups.len());
            let t0 = Instant::now();
            loop {
                let have = normalize(&read_text(t.edit).unwrap_or_default())
                    .chars()
                    .count();
                if have >= want {
                    break;
                }
                if t0.elapsed() > Duration::from_secs(3) {
                    failure = format!("chunk {i} did not arrive within 3 s ({have}/{want})");
                    break;
                }
                std::thread::yield_now();
            }
            let ms = t0.elapsed().as_secs_f64() * 1000.0;
            arrive.push(ms);
        }
        if !failure.is_empty() {
            break;
        }
    }
    let send_ms = start.elapsed().as_secs_f64() * 1000.0;
    let expected = expected_text(text, nl);
    let exp_units = expected.encode_utf16().count();
    // Settle: stop once the text matches, or once it has not changed for 1.5 s.
    let mut last = String::new();
    let mut last_change = Instant::now();
    let got = loop {
        let now = normalize(&read_text(t.edit).unwrap_or_default());
        if now == expected {
            break now;
        }
        if now != last {
            last = now.clone();
            last_change = Instant::now();
        }
        if last_change.elapsed() > Duration::from_millis(1500)
            || start.elapsed() > Duration::from_secs(30)
        {
            break now;
        }
        std::thread::sleep(Duration::from_millis(20));
    };
    let total_ms = start.elapsed().as_secs_f64() * 1000.0;
    let pass = failure.is_empty() && got == expected;
    let mut d = String::new();
    if !failure.is_empty() {
        d.push_str(&failure);
        d.push_str("; ");
    }
    if got != expected {
        d.push_str(&diff(&expected, &got));
    }
    let _ = exp_units;
    arrive.sort_by(f64::total_cmp);
    let arrive = if arrive.is_empty() {
        String::new()
    } else {
        format!(
            " arrive p50={:.1} max={:.1}ms",
            arrive[arrive.len() / 2],
            arrive[arrive.len() - 1]
        )
    };
    RunResult {
        pass,
        units,
        events,
        calls,
        send_ms,
        total_ms,
        diff: d,
        arrive,
    }
}

fn report(case: &str, mode: Mode, r: &RunResult) {
    println!(
        "{:<4} {:<9} chunk={:<4} delay={:<3} sync={} vk={:<5} confirm={:<5} units={:<4} events={:<5} calls={:<4} send={:>7.1}ms settled={:>7.1}ms{}",
        if r.pass { "PASS" } else { "FAIL" },
        case,
        if mode.chunk == 0 {
            "all".to_string()
        } else {
            mode.chunk.to_string()
        },
        mode.delay_ms,
        mode.sync,
        mode.vk,
        mode.confirm,
        r.units,
        r.events,
        r.calls,
        r.send_ms,
        r.total_ms,
        r.arrive
    );
    if !r.pass {
        println!("       {}", r.diff);
    }
}

fn long_paragraph() -> String {
    let base = "Local dictation has to put text exactly where the cursor is, without \
                losing a single character, even when the target application is busy \
                spell-checking, autosaving or redrawing. This paragraph exists to measure \
                whether long injected sequences survive intact: it mixes punctuation (commas, \
                semicolons; colons: dashes - and quotes \"like these\"), digits 0123456789, \
                and ordinary words of different lengths. ";
    let mut s = String::new();
    while s.chars().count() < 600 {
        s.push_str(base);
    }
    s.chars()
        .take(600)
        .collect::<String>()
        .trim_end()
        .to_string()
}

fn builtin_cases() -> Vec<(&'static str, String)> {
    vec![
        ("repro", "spike typed run886453".into()),
        (
            "ascii",
            "The quick brown fox jumps over the lazy dog 0123456789 !?@#%&*()[]{}<>;:'\",./".into(),
        ),
        (
            "intl",
            "Grüße aus Zürich: Ärger, Öl, Übermaß, ß; 你好世界，日本語のテキスト。".into(),
        ),
        ("emoji", "emoji 😀 ok 👍🏽 and 🇨🇭 done 𝄞".into()),
        ("newlines", "line one\nline two\n\nline four\n".into()),
        ("long600", long_paragraph()),
    ]
}

fn main() {
    let o = parse();
    let t = match o.target {
        Target::Notepad => open_notepad(),
        Target::Edit => open_own_edit(),
    };
    println!("target: {}", t.desc);
    println!(
        "session: {}",
        std::env::var("SESSIONNAME").unwrap_or_else(|_| "?".into())
    );
    let mut fails = 0;
    if o.suite {
        let cases = builtin_cases();
        let m = |chunk, delay_ms, confirm| Mode {
            chunk,
            delay_ms,
            sync: 0,
            vk: false,
            confirm,
            per_unit: false,
        };
        let modes = [
            m(0, 0, false),
            m(1, 0, false),
            m(1, 10, false),
            m(1, 20, false),
            m(8, 20, false),
            m(1, 0, true),
        ];
        for m in modes {
            for (name, text) in &cases {
                for _ in 0..o.repeat {
                    let r = run(&t, text, m, o.newline);
                    if !r.pass {
                        fails += 1;
                    }
                    report(name, m, &r);
                }
            }
        }
    } else {
        let text = o.text.clone().unwrap();
        for _ in 0..o.repeat {
            let r = run(&t, &text, o.mode, o.newline);
            if !r.pass {
                fails += 1;
            }
            report("arg", o.mode, &r);
        }
    }
    clear_text(t.edit);
    println!(
        "{}",
        if fails == 0 {
            "ALL PASS"
        } else {
            "SOME FAILED"
        }
    );
    std::process::exit(if fails == 0 { 0 } else { 1 });
}
