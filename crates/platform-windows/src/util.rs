//! `HWND` is neither `Send` nor `Sync`, but window handles are process-global values any
//! thread may post to, so they cross threads as `isize`.

use windows::Win32::Foundation::{CloseHandle, HWND};
use windows::Win32::System::Threading::{
    OpenProcess, PROCESS_NAME_WIN32, PROCESS_QUERY_LIMITED_INFORMATION, QueryFullProcessImageNameW,
};
use windows::Win32::UI::WindowsAndMessaging::GetWindowThreadProcessId;
use windows::core::PWSTR;

pub(crate) fn hwnd_from(raw: isize) -> HWND {
    HWND(raw as *mut core::ffi::c_void)
}

pub(crate) fn hwnd_raw(h: HWND) -> isize {
    h.0 as isize
}

pub(crate) fn exe_name_of_pid(pid: u32) -> Option<String> {
    if pid == 0 {
        return None;
    }
    // SAFETY: plain FFI call; the returned handle is closed below on every path.
    let handle = unsafe { OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, false, pid) }.ok()?;
    let mut buf = [0u16; 1024];
    let mut len = buf.len() as u32;
    // SAFETY: `buf` outlives the call and `len` holds its capacity in u16 units.
    let r = unsafe {
        QueryFullProcessImageNameW(
            handle,
            PROCESS_NAME_WIN32,
            PWSTR(buf.as_mut_ptr()),
            &mut len,
        )
    };
    // SAFETY: `handle` came from OpenProcess above and is closed exactly once.
    let _ = unsafe { CloseHandle(handle) };
    r.ok()?;
    let full = String::from_utf16_lossy(&buf[..len as usize]);
    Some(basename_lower(&full))
}

pub(crate) fn basename_lower(path: &str) -> String {
    path.rsplit(['\\', '/'])
        .next()
        .unwrap_or(path)
        .to_lowercase()
}

/// (thread id, process id); zeros for a dead window.
pub(crate) fn window_thread_pid(hwnd: HWND) -> (u32, u32) {
    let mut pid = 0u32;
    // SAFETY: plain FFI query; a stale handle yields 0 rather than UB.
    let tid = unsafe { GetWindowThreadProcessId(hwnd, Some(&mut pid)) };
    (tid, pid)
}

pub(crate) fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn basename_is_lowercased_file_name() {
        assert_eq!(
            basename_lower(r"C:\Windows\System32\NOTEPAD.EXE"),
            "notepad.exe"
        );
        assert_eq!(basename_lower("rdpclip.exe"), "rdpclip.exe");
    }

    #[test]
    fn own_process_resolves() {
        let name = exe_name_of_pid(std::process::id()).expect("own process");
        assert!(name.ends_with(".exe"), "{name}");
        assert_eq!(name, name.to_lowercase());
    }
}
