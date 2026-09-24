//! No crate-wide event loop: the hook is removed if it is slow, a paste blocks the target
//! until the render answers, and the overlay must never stall input, so each of those
//! lives on a thread whose only job is its own latency contract.

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
pub const INJECTED_TAG: usize = 0x4855_5348; // "HUSH"
