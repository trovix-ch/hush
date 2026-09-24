//! The tray only emits events and never changes state itself; a tray that did would
//! disagree with the pipeline after the first race.

use std::sync::mpsc::Sender;

use tray_icon::menu::{Menu, MenuEvent, MenuItem, PredefinedMenuItem};
use tray_icon::{Icon, TrayIcon, TrayIconBuilder};
use windows::Win32::UI::WindowsAndMessaging::{
    MB_ICONINFORMATION, MB_OK, MB_SETFOREGROUND, MessageBoxW,
};
use windows::core::PCWSTR;

use crate::util::wide;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrayEvent {
    TogglePause,
    PasteLast,
    CopyLast,
    OpenConfig,
    About,
    Quit,
}

pub const ABOUT_TEXT: &str = "hush: local-only voice dictation.\n\n\
Nothing you dictate leaves this machine.\n\n\
Speech recognition uses NVIDIA Parakeet TDT 0.6B v3 by NVIDIA Corporation, \
licensed under the Creative Commons Attribution 4.0 International licence \
(CC BY 4.0, https://creativecommons.org/licenses/by/4.0/). \
Source: https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3. \
The model is used in a converted file format; no other changes were made.";

const ID_PAUSE: &str = "hush.pause";
const ID_PASTE_LAST: &str = "hush.paste_last";
const ID_COPY_LAST: &str = "hush.copy_last";
const ID_OPEN_CONFIG: &str = "hush.open_config";
const ID_ABOUT: &str = "hush.about";
const ID_QUIT: &str = "hush.quit";

fn event_for(id: &str) -> Option<TrayEvent> {
    Some(match id {
        ID_PAUSE => TrayEvent::TogglePause,
        ID_PASTE_LAST => TrayEvent::PasteLast,
        ID_COPY_LAST => TrayEvent::CopyLast,
        ID_OPEN_CONFIG => TrayEvent::OpenConfig,
        ID_ABOUT => TrayEvent::About,
        ID_QUIT => TrayEvent::Quit,
        _ => return None,
    })
}

#[derive(Debug, thiserror::Error)]
pub enum TrayError {
    #[error("tray icon: {0}")]
    Icon(String),
    #[error("tray menu: {0}")]
    Menu(String),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TrayIndicator {
    Idle,
    Listening,
    Busy,
    Error,
}

impl TrayIndicator {
    fn rgb(self) -> [u8; 3] {
        match self {
            TrayIndicator::Idle => [0x9A, 0x9A, 0xA0],
            TrayIndicator::Listening => [0x4C, 0xC2, 0x6E],
            TrayIndicator::Busy => [0xF2, 0xB1, 0x3C],
            TrayIndicator::Error => [0xF0, 0x4A, 0x4A],
        }
    }
}

/// Must live on a thread that runs a message loop.
pub struct Tray {
    icon: TrayIcon,
    pause: MenuItem,
    indicator: TrayIndicator,
}

impl Tray {
    pub fn create(tx: Sender<TrayEvent>) -> Result<Self, TrayError> {
        let menu = Menu::new();
        let pause = MenuItem::with_id(ID_PAUSE, "Pause", true, None);
        let paste = MenuItem::with_id(ID_PASTE_LAST, "Paste last transcript", true, None);
        let copy = MenuItem::with_id(ID_COPY_LAST, "Copy last transcript", true, None);
        let config = MenuItem::with_id(ID_OPEN_CONFIG, "Open config", true, None);
        let about = MenuItem::with_id(ID_ABOUT, "About", true, None);
        let quit = MenuItem::with_id(ID_QUIT, "Quit", true, None);
        menu.append_items(&[
            &pause,
            &PredefinedMenuItem::separator(),
            &paste,
            &copy,
            &PredefinedMenuItem::separator(),
            &config,
            &about,
            &PredefinedMenuItem::separator(),
            &quit,
        ])
        .map_err(|e| TrayError::Menu(e.to_string()))?;
        MenuEvent::set_event_handler(Some(move |e: MenuEvent| {
            if let Some(ev) = event_for(e.id().as_ref()) {
                let _ = tx.send(ev);
            }
        }));
        let icon = TrayIconBuilder::new()
            .with_tooltip("hush")
            .with_icon(icon_image(TrayIndicator::Idle)?)
            .with_menu(Box::new(menu))
            .build()
            .map_err(|e| TrayError::Icon(e.to_string()))?;
        Ok(Self {
            icon,
            pause,
            indicator: TrayIndicator::Idle,
        })
    }

    pub fn set_paused(&self, paused: bool) {
        self.pause.set_text(if paused { "Resume" } else { "Pause" });
    }

    /// Replaces the icon only on a change: the level meter repeats the same state
    /// twenty times a second.
    pub fn set_indicator(&mut self, indicator: TrayIndicator) {
        if indicator == self.indicator {
            return;
        }
        match icon_image(indicator) {
            Ok(i) => {
                if self.icon.set_icon(Some(i)).is_ok() {
                    self.indicator = indicator;
                }
            }
            Err(e) => tracing::warn!(error = %e, "tray icon"),
        }
    }

    /// Windows cuts tooltips at 127 UTF-16 units, possibly mid-surrogate, so the cut
    /// happens here on a character boundary.
    pub fn set_tooltip(&self, text: &str) {
        let _ = self.icon.set_tooltip(Some(truncate_utf16(text, 127)));
    }
}

fn truncate_utf16(s: &str, max_units: usize) -> String {
    let mut units = 0;
    s.chars()
        .take_while(|c| {
            units += c.len_utf16();
            units <= max_units
        })
        .collect()
}

impl Drop for Tray {
    fn drop(&mut self) {
        MenuEvent::set_event_handler(None::<fn(MenuEvent)>);
    }
}

/// On its own thread because a message box is a modal loop that would stall the UI
/// thread's command queue.
pub fn show_about() {
    std::thread::spawn(|| {
        let text = wide(ABOUT_TEXT);
        let title = wide("About hush");
        // SAFETY: both buffers are NUL-terminated and outlive the modal call.
        unsafe {
            MessageBoxW(
                None,
                PCWSTR(text.as_ptr()),
                PCWSTR(title.as_ptr()),
                MB_OK | MB_ICONINFORMATION | MB_SETFOREGROUND,
            );
        }
    });
}

fn icon_image(indicator: TrayIndicator) -> Result<Icon, TrayError> {
    Icon::from_rgba(icon_rgba(indicator), 32, 32).map_err(|e| TrayError::Icon(e.to_string()))
}

/// A microphone glyph, straight (not premultiplied) RGBA.
pub(crate) fn icon_rgba(indicator: TrayIndicator) -> Vec<u8> {
    let color = indicator.rgb();
    let mut out = vec![0u8; 32 * 32 * 4];
    for y in 0..32 {
        for x in 0..32 {
            let (fx, fy) = (x as f32 + 0.5, y as f32 + 0.5);
            let cy = fy.clamp(8.0, 15.0);
            let capsule = ((fx - 16.0).powi(2) + (fy - cy).powi(2)).sqrt() - 5.0;
            let ring = (((fx - 16.0).powi(2) + (fy - 14.0).powi(2)).sqrt() - 8.5).abs() - 1.1;
            let cradle = if fy >= 14.0 { ring } else { f32::MAX };
            let stem = ((fx - 16.0).abs() - 1.1).max((fy - 25.5).abs() - 3.0);
            let foot = ((fx - 16.0).abs() - 5.0).max((fy - 28.0).abs() - 1.1);
            let d = capsule.min(cradle).min(stem).min(foot);
            let a = (0.5 - d).clamp(0.0, 1.0);
            let i = (y * 32 + x) * 4;
            out[i..i + 3].copy_from_slice(&color);
            out[i + 3] = (a * 255.0).round() as u8;
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_menu_id_maps_to_an_event() {
        for id in [
            ID_PAUSE,
            ID_PASTE_LAST,
            ID_COPY_LAST,
            ID_OPEN_CONFIG,
            ID_ABOUT,
            ID_QUIT,
        ] {
            assert!(event_for(id).is_some(), "{id}");
        }
        assert_eq!(event_for("other"), None);
    }

    #[test]
    fn icon_has_a_visible_glyph_and_clear_corners() {
        let px = icon_rgba(TrayIndicator::Listening);
        assert_eq!(px.len(), 32 * 32 * 4);
        assert_eq!(px[3], 0);
        let opaque = px.chunks_exact(4).filter(|p| p[3] > 200).count();
        assert!((80..600).contains(&opaque), "{opaque}");
        assert!(Icon::from_rgba(px, 32, 32).is_ok());
    }

    #[test]
    fn each_indicator_has_its_own_colour() {
        let all = [
            TrayIndicator::Idle,
            TrayIndicator::Listening,
            TrayIndicator::Busy,
            TrayIndicator::Error,
        ];
        for (i, a) in all.iter().enumerate() {
            for b in &all[i + 1..] {
                assert_ne!(a.rgb(), b.rgb(), "{a:?} vs {b:?}");
            }
        }
    }

    #[test]
    fn tooltip_is_cut_on_a_character_boundary() {
        assert_eq!(truncate_utf16("abc", 127), "abc");
        assert_eq!(truncate_utf16("ab😀", 3), "ab");
        assert_eq!(truncate_utf16("ab😀", 4), "ab😀");
    }

    #[test]
    fn about_carries_the_attribution() {
        assert!(ABOUT_TEXT.contains("CC BY 4.0"));
        assert!(ABOUT_TEXT.contains("Parakeet"));
        assert!(ABOUT_TEXT.contains("NVIDIA"));
    }
}
