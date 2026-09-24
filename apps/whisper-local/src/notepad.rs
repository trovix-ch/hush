//! A Notepad window as a simulation target, and reading its text back.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Result, bail};
use windows::Win32::Foundation::{HWND, LPARAM, WPARAM};
use windows::Win32::UI::Shell::ShellExecuteW;
use windows::Win32::UI::WindowsAndMessaging::{
    EnumChildWindows, EnumWindows, GetClassNameW, GetWindowTextW, IsWindowVisible,
    SMTO_ABORTIFHUNG, SW_SHOWNORMAL, SendMessageTimeoutW, WM_GETTEXT, WM_GETTEXTLENGTH,
};
use windows::core::{BOOL, PCWSTR, w};

pub struct Notepad {
    pub hwnd: isize,
    pub file: PathBuf,
}

/// Opens Notepad on an empty scratch file and waits for its window.
pub fn open() -> Result<Notepad> {
    let file = std::env::temp_dir().join(format!("wl-simulate-{}.txt", std::process::id()));
    std::fs::write(&file, "")?;
    let name = file
        .file_name()
        .map(|n| n.to_string_lossy().into_owned())
        .unwrap_or_default();
    // Not `Command`: it creates the child with handle inheritance on, so even with null
    // stdio Notepad inherits our (inheritable) stdout pipe and keeps a piped caller
    // waiting for as long as Notepad stays open. ShellExecute inherits nothing.
    let param: Vec<u16> = format!("\"{}\"", file.display())
        .encode_utf16()
        .chain([0])
        .collect();
    // SAFETY: NUL-terminated buffers that outlive the call.
    let r = unsafe {
        ShellExecuteW(
            None,
            w!("open"),
            w!("notepad.exe"),
            PCWSTR(param.as_ptr()),
            PCWSTR::null(),
            SW_SHOWNORMAL,
        )
    };
    if r.0 as usize <= 32 {
        bail!(
            "could not start Notepad (ShellExecute returned {})",
            r.0 as usize
        );
    }
    let deadline = Instant::now() + Duration::from_secs(10);
    while Instant::now() < deadline {
        if let Some(h) = find_window_titled(&name) {
            return Ok(Notepad {
                hwnd: h.0 as isize,
                file,
            });
        }
        std::thread::sleep(Duration::from_millis(100));
    }
    bail!("no Notepad window titled {name} appeared within 10 s")
}

fn find_window_titled(needle: &str) -> Option<HWND> {
    struct Ctx<'a> {
        needle: &'a str,
        found: Option<HWND>,
    }
    unsafe extern "system" fn cb(hwnd: HWND, lp: LPARAM) -> BOOL {
        // SAFETY: `lp` is the &mut Ctx passed below, alive for the enumeration.
        let ctx = unsafe { &mut *(lp.0 as *mut Ctx) };
        // SAFETY: plain FFI queries on an enumerated window.
        if !unsafe { IsWindowVisible(hwnd) }.as_bool() {
            return BOOL(1);
        }
        let mut buf = [0u16; 512];
        // SAFETY: as above.
        let n = unsafe { GetWindowTextW(hwnd, &mut buf) };
        if String::from_utf16_lossy(&buf[..n.max(0) as usize]).contains(ctx.needle) {
            ctx.found = Some(hwnd);
            return BOOL(0);
        }
        BOOL(1)
    }
    let mut ctx = Ctx {
        needle,
        found: None,
    };
    // SAFETY: the callback only touches `ctx` through the pointer we pass.
    let _ = unsafe { EnumWindows(Some(cb), LPARAM(&mut ctx as *mut _ as isize)) };
    ctx.found
}

/// Text of every edit-like child of `top`. Works for both the classic and the Windows 11
/// Notepad (whose editor is a RichEdit class) without UI Automation.
pub fn text(top: isize) -> Vec<String> {
    unsafe extern "system" fn cb(hwnd: HWND, lp: LPARAM) -> BOOL {
        // SAFETY: `lp` is the &mut Vec passed below.
        let out = unsafe { &mut *(lp.0 as *mut Vec<String>) };
        let mut cls = [0u16; 128];
        // SAFETY: plain FFI query.
        let n = unsafe { GetClassNameW(hwnd, &mut cls) };
        let class = String::from_utf16_lossy(&cls[..n.max(0) as usize]);
        if !class.to_lowercase().contains("edit") {
            return BOOL(1);
        }
        let mut len = 0usize;
        // SAFETY: cross-process WM_GETTEXT with a timeout; the buffer outlives the call.
        unsafe {
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
            out.push(String::from_utf16_lossy(&buf[..got.min(len)]));
        }
        BOOL(1)
    }
    let mut out = Vec::new();
    // SAFETY: the callback only touches `out` through the pointer we pass.
    let _ = unsafe {
        EnumChildWindows(
            Some(HWND(top as *mut _)),
            Some(cb),
            LPARAM(&mut out as *mut _ as isize),
        )
    };
    out
}
