//! Run at sign-in through the per-user Run key: no admin, no scheduled task, and Task
//! Manager's Startup page lists and can disable it like any other app.

use std::path::Path;

use anyhow::{Context, Result, bail};
use windows::Win32::Foundation::{ERROR_FILE_NOT_FOUND, ERROR_SUCCESS, WIN32_ERROR};
use windows::Win32::System::Registry::{
    HKEY, HKEY_CURRENT_USER, KEY_SET_VALUE, REG_OPTION_NON_VOLATILE, REG_SZ, RRF_RT_REG_SZ,
    RegCloseKey, RegCreateKeyExW, RegDeleteValueW, RegGetValueW, RegSetValueExW,
};
use windows::core::PCWSTR;

/// A value in a key under HKEY_CURRENT_USER.
#[derive(Debug, Clone, Copy)]
pub struct RunEntry<'a> {
    pub key: &'a str,
    pub name: &'a str,
}

pub const STARTUP: RunEntry<'static> = RunEntry {
    key: r"Software\Microsoft\Windows\CurrentVersion\Run",
    name: "hush",
};

fn wide(s: &str) -> Vec<u16> {
    s.encode_utf16().chain(std::iter::once(0)).collect()
}

fn check(r: WIN32_ERROR, what: &str) -> Result<()> {
    if r == ERROR_SUCCESS {
        Ok(())
    } else {
        bail!("{what}: {}", windows::core::Error::from(r.to_hresult()))
    }
}

/// Quoted, because Windows splits an unquoted path at its first space.
pub fn command_for(exe: &Path) -> String {
    format!("\"{}\"", exe.display())
}

impl RunEntry<'_> {
    pub fn read(&self) -> Result<Option<String>> {
        let (key, name) = (wide(self.key), wide(self.name));
        let mut bytes = 0u32;
        // SAFETY: NUL-terminated names; a size query with no buffer.
        let r = unsafe {
            RegGetValueW(
                HKEY_CURRENT_USER,
                PCWSTR(key.as_ptr()),
                PCWSTR(name.as_ptr()),
                RRF_RT_REG_SZ,
                None,
                None,
                Some(&mut bytes),
            )
        };
        if r == ERROR_FILE_NOT_FOUND {
            return Ok(None);
        }
        check(r, "reading the Run key")?;
        let mut buf = vec![0u16; (bytes as usize).div_ceil(2)];
        // SAFETY: `buf` holds `bytes` bytes, as the size query asked for.
        let r = unsafe {
            RegGetValueW(
                HKEY_CURRENT_USER,
                PCWSTR(key.as_ptr()),
                PCWSTR(name.as_ptr()),
                RRF_RT_REG_SZ,
                None,
                Some(buf.as_mut_ptr().cast()),
                Some(&mut bytes),
            )
        };
        if r == ERROR_FILE_NOT_FOUND {
            return Ok(None);
        }
        check(r, "reading the Run key")?;
        let len = buf.iter().position(|&c| c == 0).unwrap_or(buf.len());
        Ok(Some(String::from_utf16_lossy(&buf[..len])))
    }

    pub fn write(&self, command: &str) -> Result<()> {
        let key = self.open()?;
        let name = wide(self.name);
        let data: Vec<u8> = wide(command).iter().flat_map(|u| u.to_le_bytes()).collect();
        // SAFETY: an open key we close below; `data` is a NUL-terminated UTF-16 string.
        let r = unsafe { RegSetValueExW(key, PCWSTR(name.as_ptr()), None, REG_SZ, Some(&data)) };
        // SAFETY: closing the key opened above.
        let _ = unsafe { RegCloseKey(key) };
        check(r, "writing the Run key")
    }

    /// Removing a value that is not there succeeds.
    pub fn remove(&self) -> Result<()> {
        let key = self.open()?;
        let name = wide(self.name);
        // SAFETY: an open key we close below.
        let r = unsafe { RegDeleteValueW(key, PCWSTR(name.as_ptr())) };
        // SAFETY: closing the key opened above.
        let _ = unsafe { RegCloseKey(key) };
        if r == ERROR_FILE_NOT_FOUND {
            return Ok(());
        }
        check(r, "removing the Run key value")
    }

    fn open(&self) -> Result<HKEY> {
        let key = wide(self.key);
        let mut h = HKEY::default();
        // SAFETY: NUL-terminated name; `h` receives the handle, which the caller closes.
        let r = unsafe {
            RegCreateKeyExW(
                HKEY_CURRENT_USER,
                PCWSTR(key.as_ptr()),
                None,
                PCWSTR::null(),
                REG_OPTION_NON_VOLATILE,
                KEY_SET_VALUE,
                None,
                &mut h,
                None,
            )
        };
        check(r, "opening the Run key")?;
        Ok(h)
    }

    /// Makes the entry match `want` for this executable, and says what changed.
    pub fn sync(&self, want: bool, exe: &Path) -> Result<Change> {
        let current = self.read()?;
        let change = reconcile(want, current.as_deref(), &command_for(exe));
        match &change {
            Change::None => {}
            Change::Write(command) => self.write(command)?,
            Change::Remove => self.remove()?,
        }
        Ok(change)
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Change {
    None,
    Write(String),
    Remove,
}

/// A stale path (the exe was moved or a new build unpacked elsewhere) is rewritten; the
/// comparison ignores case because Windows paths do.
pub fn reconcile(want: bool, current: Option<&str>, command: &str) -> Change {
    match (want, current) {
        (true, Some(c)) if c.eq_ignore_ascii_case(command) => Change::None,
        (true, _) => Change::Write(command.to_string()),
        (false, Some(_)) => Change::Remove,
        (false, None) => Change::None,
    }
}

/// The tray toggle: the Run key first, then the config line that mirrors it.
pub fn set(want: bool, exe: &Path, config_file: &Path) -> Result<()> {
    STARTUP.sync(want, exe)?;
    let text = std::fs::read_to_string(config_file)
        .with_context(|| format!("reading {}", config_file.display()))?;
    let updated = hush_core::config::with_top_level_bool(&text, "start_with_windows", want);
    std::fs::write(config_file, updated)
        .with_context(|| format!("writing {}", config_file.display()))
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows::Win32::System::Registry::RegDeleteTreeW;

    #[test]
    fn a_stale_or_missing_entry_is_rewritten_and_an_unwanted_one_removed() {
        let cmd = command_for(Path::new(r"C:\Apps\hush\hush.exe"));
        assert_eq!(cmd, r#""C:\Apps\hush\hush.exe""#);
        assert_eq!(reconcile(true, Some(&cmd), &cmd), Change::None);
        assert_eq!(
            reconcile(true, Some(r#""c:\apps\HUSH\hush.exe""#), &cmd),
            Change::None
        );
        assert_eq!(
            reconcile(true, Some(r#""C:\Old\hush.exe""#), &cmd),
            Change::Write(cmd.clone())
        );
        assert_eq!(reconcile(true, None, &cmd), Change::Write(cmd.clone()));
        assert_eq!(reconcile(false, Some(&cmd), &cmd), Change::Remove);
        assert_eq!(reconcile(false, None, &cmd), Change::None);
    }

    #[test]
    fn run_entry_round_trips_through_the_registry() {
        let key = format!(r"Software\hush-test-{}", std::process::id());
        let entry = RunEntry {
            key: &key,
            name: "hush",
        };
        let exe = Path::new(r"C:\Program Files\hush test\hush.exe");
        let result = (|| -> Result<()> {
            assert_eq!(entry.read()?, None);
            assert_eq!(entry.sync(true, exe)?, Change::Write(command_for(exe)));
            assert_eq!(entry.read()?, Some(command_for(exe)));
            assert_eq!(entry.sync(true, exe)?, Change::None);
            let moved = Path::new(r"D:\hush\hush.exe");
            assert_eq!(entry.sync(true, moved)?, Change::Write(command_for(moved)));
            assert_eq!(entry.read()?, Some(command_for(moved)));
            assert_eq!(entry.sync(false, moved)?, Change::Remove);
            assert_eq!(entry.read()?, None);
            entry.remove()?;
            Ok(())
        })();
        let wkey = wide(&key);
        // SAFETY: deletes only the scratch key this test created.
        let _ = unsafe { RegDeleteTreeW(HKEY_CURRENT_USER, PCWSTR(wkey.as_ptr())) };
        result.unwrap();
    }
}
