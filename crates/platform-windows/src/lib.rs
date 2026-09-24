//! Win32 integration. Everything that touches the OS lives here; the pipeline in `core`
//! only sees the port traits this crate implements.
//!
//! Why not one crate-wide event loop: the hook, the clipboard owner and the UI each have
//! a latency contract of their own (the hook is removed by Windows if it is slow, a paste
//! blocks the target app until the render answers, the overlay must never stall input),
//! so each lives on a thread whose only job is that contract.

#![cfg(windows)]

pub mod clipboard;
pub mod focus;
pub mod hook;
pub mod input;
pub mod overlay;
pub mod sound;
pub mod tray;
pub mod ui_thread;

mod util;

/// Marker in `dwExtraInfo` of every event this process injects. The hook passes these
/// through untouched so our own paste chord can never look like the hotkey.
pub const INJECTED_TAG: usize = 0x574C_4F43; // "WLOC"
