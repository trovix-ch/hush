//! Typing is one UTF-16 unit per `SendInput` with a pause, never one batch: Win11 Notepad
//! translates a queued `VK_PACKET` with the most recently injected character, so a
//! batched string comes out as copies of its last character.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use windows::Win32::UI::Input::KeyboardAndMouse::{
    GetAsyncKeyState, INPUT, INPUT_0, INPUT_KEYBOARD, KEYBD_EVENT_FLAGS, KEYBDINPUT,
    KEYEVENTF_EXTENDEDKEY, KEYEVENTF_KEYUP, KEYEVENTF_UNICODE, MAPVK_VK_TO_VSC, MapVirtualKeyW,
    SendInput, VIRTUAL_KEY,
};
use windows::Win32::UI::WindowsAndMessaging::GetForegroundWindow;
use wl_core::insert::InsertError;

use crate::INJECTED_TAG;

const VK_RETURN: u16 = 0x0D;
const VK_INSERT: u16 = 0x2D;
const VK_V: u16 = 0x56;
const VK_LSHIFT: u16 = 0xA0;
const VK_RSHIFT: u16 = 0xA1;
const VK_LCONTROL: u16 = 0xA2;
const VK_RCONTROL: u16 = 0xA3;
const VK_LMENU: u16 = 0xA4;
const VK_RMENU: u16 = 0xA5;
const VK_LWIN: u16 = 0x5B;
const VK_RWIN: u16 = 0x5C;

const MODIFIERS: [u16; 8] = [
    VK_LCONTROL,
    VK_RCONTROL,
    VK_LSHIFT,
    VK_RSHIFT,
    VK_LMENU,
    VK_RMENU,
    VK_LWIN,
    VK_RWIN,
];

/// Measured against Notepad, only 20-30 ms per unit passed repeatedly.
pub const DEFAULT_UNIT_DELAY: Duration = Duration::from_millis(25);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Chord {
    CtrlV,
    CtrlShiftV,
    ShiftInsert,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum InputError {
    /// How UIPI (elevated target) and a locked or disconnected session show up.
    #[error("SendInput inserted {sent} of {expected} events")]
    Blocked { sent: u32, expected: u32 },
    #[error("focus moved after {typed} of {total} units were typed")]
    FocusLost { typed: usize, total: usize },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct TypeReport {
    /// `SendInput` calls: one per UTF-16 unit or Return.
    pub steps: usize,
    pub elapsed: Duration,
}

#[derive(Debug, Clone)]
pub struct WinInput {
    pub unit_delay: Duration,
    last_chord: Arc<Mutex<Option<Instant>>>,
}

impl Default for WinInput {
    fn default() -> Self {
        Self {
            unit_delay: DEFAULT_UNIT_DELAY,
            last_chord: Arc::default(),
        }
    }
}

impl WinInput {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn last_chord_at(&self) -> Option<Instant> {
        *self.last_chord.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Reads the async key state rather than tracking presses, because a modifier held
    /// since before we started watching still turns `v` into something else.
    pub fn release_modifiers(&self) -> Result<Vec<u16>, InputError> {
        let held: Vec<u16> = MODIFIERS
            .iter()
            .copied()
            // SAFETY: plain FFI query.
            .filter(|&vk| (unsafe { GetAsyncKeyState(vk as i32) } as u16) & 0x8000 != 0)
            .collect();
        if !held.is_empty() {
            let ups: Vec<INPUT> = held.iter().map(|&vk| vk_event(vk, true)).collect();
            send_all(&ups)?;
        }
        Ok(held)
    }

    pub fn send_chord(&self, chord: Chord) -> Result<(), InputError> {
        let events = chord_events(chord);
        *self.last_chord.lock().unwrap_or_else(|e| e.into_inner()) = Some(Instant::now());
        send_all(&events)
    }

    /// Stops if the foreground window changes. Blocks for about `unit_delay` per UTF-16
    /// unit; `\n` becomes Return and `\r` is dropped.
    pub fn type_text(&self, text: &str) -> Result<TypeReport, InputError> {
        let steps = typing_steps(text);
        let start = Instant::now();
        // SAFETY: plain FFI query.
        let target = unsafe { GetForegroundWindow() };
        for (i, step) in steps.iter().enumerate() {
            // SAFETY: as above.
            if i > 0 && unsafe { GetForegroundWindow() } != target {
                return Err(InputError::FocusLost {
                    typed: i,
                    total: steps.len(),
                });
            }
            send_all(step)?;
            if i + 1 < steps.len() {
                std::thread::sleep(self.unit_delay);
            }
        }
        Ok(TypeReport {
            steps: steps.len(),
            elapsed: start.elapsed(),
        })
    }
}

impl From<wl_core::insert::Chord> for Chord {
    fn from(c: wl_core::insert::Chord) -> Self {
        match c {
            wl_core::insert::Chord::CtrlV => Chord::CtrlV,
            wl_core::insert::Chord::CtrlShiftV => Chord::CtrlShiftV,
            wl_core::insert::Chord::ShiftInsert => Chord::ShiftInsert,
        }
    }
}

impl wl_core::insert::InputPort for WinInput {
    fn release_modifiers(&mut self) -> Result<(), InsertError> {
        WinInput::release_modifiers(self)
            .map(|_| ())
            .map_err(input_err)
    }

    fn send_chord(&mut self, chord: wl_core::insert::Chord) -> Result<(), InsertError> {
        WinInput::send_chord(self, chord.into()).map_err(input_err)
    }

    fn type_text(&mut self, text: &str) -> Result<(), InsertError> {
        WinInput::type_text(self, text)
            .map(|_| ())
            .map_err(input_err)
    }
}

fn input_err(e: InputError) -> InsertError {
    InsertError::Input(e.to_string())
}

fn send_all(inputs: &[INPUT]) -> Result<(), InputError> {
    if inputs.is_empty() {
        return Ok(());
    }
    // SAFETY: `inputs` is a valid slice for the duration of the call.
    let sent = unsafe { SendInput(inputs, std::mem::size_of::<INPUT>() as i32) };
    if sent as usize == inputs.len() {
        Ok(())
    } else {
        Err(InputError::Blocked {
            sent,
            expected: inputs.len() as u32,
        })
    }
}

fn kbd(vk: u16, scan: u16, flags: KEYBD_EVENT_FLAGS) -> INPUT {
    INPUT {
        r#type: INPUT_KEYBOARD,
        Anonymous: INPUT_0 {
            ki: KEYBDINPUT {
                wVk: VIRTUAL_KEY(vk),
                wScan: scan,
                dwFlags: flags,
                time: 0,
                dwExtraInfo: INJECTED_TAG,
            },
        },
    }
}

pub(crate) fn vk_event(vk: u16, up: bool) -> INPUT {
    let mut flags = KEYBD_EVENT_FLAGS(0);
    if up {
        flags |= KEYEVENTF_KEYUP;
    }
    // Insert and the right-hand modifiers are extended keys; without the flag they
    // arrive as their numpad or left-hand twins.
    if matches!(vk, VK_INSERT | VK_RCONTROL | VK_RMENU | VK_LWIN | VK_RWIN) {
        flags |= KEYEVENTF_EXTENDEDKEY;
    }
    // SAFETY: plain FFI lookup of the scan code for the active layout.
    let scan = unsafe { MapVirtualKeyW(vk as u32, MAPVK_VK_TO_VSC) } as u16;
    kbd(vk, scan, flags)
}

/// Sent as one batch because the OS never interleaves physical keys into a batch.
fn chord_events(chord: Chord) -> Vec<INPUT> {
    let (mods, key): (&[u16], u16) = match chord {
        Chord::CtrlV => (&[VK_LCONTROL], VK_V),
        Chord::CtrlShiftV => (&[VK_LCONTROL, VK_LSHIFT], VK_V),
        Chord::ShiftInsert => (&[VK_LSHIFT], VK_INSERT),
    };
    let mut v: Vec<INPUT> = mods.iter().map(|&m| vk_event(m, false)).collect();
    v.push(vk_event(key, false));
    v.push(vk_event(key, true));
    v.extend(mods.iter().rev().map(|&m| vk_event(m, true)));
    v
}

/// `\t` goes out as a Unicode unit, not as VK_TAB, which would move focus in a dialog.
fn typing_steps(text: &str) -> Vec<[INPUT; 2]> {
    let mut steps = Vec::with_capacity(text.len());
    for c in text.chars() {
        match c {
            '\r' => {}
            '\n' => steps.push([vk_event(VK_RETURN, false), vk_event(VK_RETURN, true)]),
            _ => {
                let mut buf = [0u16; 2];
                for &u in c.encode_utf16(&mut buf).iter() {
                    steps.push([
                        kbd(0, u, KEYEVENTF_UNICODE),
                        kbd(0, u, KEYEVENTF_UNICODE | KEYEVENTF_KEYUP),
                    ]);
                }
            }
        }
    }
    steps
}

#[cfg(test)]
mod tests {
    use super::*;

    fn describe(i: &INPUT) -> (u16, u16, u32) {
        // SAFETY: every INPUT built here is a keyboard input.
        let ki = unsafe { i.Anonymous.ki };
        (ki.wVk.0, ki.wScan, ki.dwFlags.0)
    }

    #[test]
    fn ctrl_v_is_press_press_release_release() {
        let e: Vec<_> = chord_events(Chord::CtrlV).iter().map(describe).collect();
        let vks: Vec<u16> = e.iter().map(|x| x.0).collect();
        assert_eq!(vks, [VK_LCONTROL, VK_V, VK_V, VK_LCONTROL]);
        let ups: Vec<bool> = e.iter().map(|x| x.2 & KEYEVENTF_KEYUP.0 != 0).collect();
        assert_eq!(ups, [false, false, true, true]);
    }

    #[test]
    fn shift_insert_marks_insert_extended() {
        let e: Vec<_> = chord_events(Chord::ShiftInsert)
            .iter()
            .map(describe)
            .collect();
        let insert = e.iter().find(|x| x.0 == VK_INSERT).unwrap();
        assert!(insert.2 & KEYEVENTF_EXTENDEDKEY.0 != 0);
        assert_eq!(chord_events(Chord::CtrlShiftV).len(), 6);
    }

    #[test]
    fn every_event_is_tagged() {
        let typed: Vec<INPUT> = typing_steps("a😀\n").concat();
        for i in chord_events(Chord::CtrlShiftV).iter().chain(typed.iter()) {
            // SAFETY: keyboard inputs only.
            assert_eq!(unsafe { i.Anonymous.ki.dwExtraInfo }, INJECTED_TAG);
        }
    }

    #[test]
    fn each_unit_is_its_own_down_up_step() {
        let s = typing_steps("a b");
        assert_eq!(s.len(), 3);
        let space: Vec<_> = s[1].iter().map(describe).collect();
        assert_eq!(space[0], (0, 0x20, KEYEVENTF_UNICODE.0));
        assert_eq!(space[1], (0, 0x20, (KEYEVENTF_UNICODE | KEYEVENTF_KEYUP).0));
    }

    #[test]
    fn surrogate_halves_go_out_in_separate_steps_in_order() {
        let s = typing_steps("😀");
        assert_eq!(s.len(), 2);
        assert_eq!(describe(&s[0][0]).1, 0xD83D);
        assert_eq!(describe(&s[0][1]).1, 0xD83D);
        assert_eq!(describe(&s[1][0]).1, 0xDE00);
    }

    #[test]
    fn newlines_become_return_and_crlf_is_one() {
        let s = typing_steps("a\r\nb\n\t");
        assert_eq!(s.len(), 5);
        assert_eq!(describe(&s[1][0]).0, VK_RETURN);
        assert_eq!(describe(&s[3][0]).0, VK_RETURN);
        assert_eq!(describe(&s[4][0]), (0, 0x09, KEYEVENTF_UNICODE.0));
    }

    #[test]
    fn empty_text_types_nothing() {
        assert!(typing_steps("").is_empty());
        assert!(typing_steps("\r").is_empty());
    }
}
