//! What the app captures about the focused target at hotkey-down.
//!
//! Deliberately minimal: enough to choose an insertion strategy and a style, and to
//! refuse unsafe targets. No screenshots, no full text dumps.

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FocusContext {
    /// Opaque native window handle of the foreground window, as an integer so this
    /// type stays platform-neutral. Zero means unknown.
    pub window: usize,
    /// Lower-case executable file name, e.g. `windowsterminal.exe`.
    pub exe: Option<String>,
    pub title: Option<String>,
    /// Target process runs at a higher integrity level than we do; input injection
    /// and hooks will be ignored by the OS.
    pub elevated: bool,
    /// Focused control reports itself as a password field; never insert.
    pub is_password: bool,
}
