//! What the app captures about the focused target at hotkey-down. Deliberately minimal:
//! no screenshots, no full text dumps.

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FocusContext {
    /// Native window handle as an integer to stay platform-neutral. Zero means unknown.
    pub window: usize,
    /// Lower-case, e.g. `windowsterminal.exe`.
    pub exe: Option<String>,
    pub title: Option<String>,
    /// Higher integrity level than ours: the OS silently ignores our injected input.
    pub elevated: bool,
    pub is_password: bool,
}
