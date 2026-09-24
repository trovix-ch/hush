//! The pill must never take focus from the paste target, and `SW_SHOWNOACTIVATE` alone
//! does not stop a click from activating it, hence the no-activate click-through styles.
//! GDI ignores alpha, so it only draws text into a coverage mask and pixels are composed
//! here.

use std::time::Duration;

use windows::Win32::Foundation::{COLORREF, HWND, LPARAM, LRESULT, POINT, RECT, SIZE, WPARAM};
use windows::Win32::Graphics::Gdi::{
    AC_SRC_ALPHA, AC_SRC_OVER, ANTIALIASED_QUALITY, BI_RGB, BITMAPINFO, BITMAPINFOHEADER,
    BLENDFUNCTION, CLIP_DEFAULT_PRECIS, CreateCompatibleDC, CreateDIBSection, CreateFontW,
    DEFAULT_CHARSET, DIB_RGB_COLORS, DT_END_ELLIPSIS, DT_LEFT, DT_NOPREFIX, DT_SINGLELINE,
    DT_VCENTER, DeleteDC, DeleteObject, DrawTextW, FW_SEMIBOLD, GetDC, HBITMAP, HDC, HFONT,
    OUT_DEFAULT_PRECIS, ReleaseDC, SelectObject, SetBkMode, SetTextColor, TRANSPARENT,
};
use windows::Win32::System::LibraryLoader::GetModuleHandleW;
use windows::Win32::UI::HiDpi::GetDpiForSystem;
use windows::Win32::UI::WindowsAndMessaging::{
    CreateWindowExW, DefWindowProcW, DestroyWindow, HWND_TOPMOST, KillTimer, RegisterClassW,
    SPI_GETWORKAREA, SW_HIDE, SW_SHOWNOACTIVATE, SWP_NOACTIVATE, SWP_NOMOVE, SWP_NOSIZE,
    SYSTEM_PARAMETERS_INFO_UPDATE_FLAGS, SetTimer, SetWindowPos, ShowWindow, SystemParametersInfoW,
    ULW_ALPHA, UpdateLayeredWindow, WM_NCHITTEST, WNDCLASSW, WS_EX_LAYERED, WS_EX_NOACTIVATE,
    WS_EX_TOOLWINDOW, WS_EX_TOPMOST, WS_EX_TRANSPARENT, WS_POPUP,
};
use windows::core::{PCWSTR, w};

use crate::util::wide;

#[derive(Debug, Clone, PartialEq)]
pub enum OverlayState {
    Hidden,
    /// `level` is in 0..=1.
    Listening {
        level: f32,
    },
    Transcribing,
    Normalizing,
    Inserting,
    /// `message` overrides the default text.
    Done {
        message: Option<String>,
    },
    Error {
        message: String,
    },
    /// Not an error, but hides like one.
    Notice {
        message: String,
    },
    /// Does not hide by itself.
    Status {
        message: String,
    },
}

impl OverlayState {
    fn label(&self) -> String {
        match self {
            OverlayState::Hidden => String::new(),
            OverlayState::Listening { .. } => "Listening".into(),
            OverlayState::Transcribing => "Transcribing…".into(),
            OverlayState::Normalizing => "Cleaning up…".into(),
            OverlayState::Inserting => "Inserting…".into(),
            OverlayState::Done { message } => message.clone().unwrap_or_else(|| "Done".into()),
            OverlayState::Error { message }
            | OverlayState::Notice { message }
            | OverlayState::Status { message } => message.clone(),
        }
    }

    fn dot_rgb(&self) -> [u8; 3] {
        match self {
            OverlayState::Hidden => [0, 0, 0],
            OverlayState::Listening { .. } => [0xF0, 0x4A, 0x4A],
            OverlayState::Transcribing => [0xF2, 0xB1, 0x3C],
            OverlayState::Normalizing | OverlayState::Inserting => [0x5B, 0x9C, 0xF5],
            OverlayState::Done { .. } => [0x4C, 0xC2, 0x6E],
            OverlayState::Error { .. } => [0xF0, 0x4A, 0x4A],
            OverlayState::Notice { .. } => [0xF2, 0xB1, 0x3C],
            OverlayState::Status { .. } => [0x9A, 0x9A, 0xA0],
        }
    }
}

impl From<wl_core::notify::OverlayState> for OverlayState {
    fn from(s: wl_core::notify::OverlayState) -> Self {
        use wl_core::notify::{OverlayState as C, ProvenanceHint};
        match s {
            C::Idle => OverlayState::Hidden,
            C::Listening { level } => OverlayState::Listening { level },
            C::Transcribing => OverlayState::Transcribing,
            C::Normalizing => OverlayState::Normalizing,
            C::Inserting => OverlayState::Inserting,
            // An LLM fallback gets a subtle indicator, never an error.
            C::Done { provenance_hint } => OverlayState::Done {
                message: (provenance_hint == ProvenanceHint::LlmFallback)
                    .then(|| "Done · rules only".to_string()),
            },
            C::Error { message } => OverlayState::Error { message },
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct OverlayConfig {
    pub done_timeout: Duration,
    pub error_timeout: Duration,
    /// Logical pixels above the bottom of the work area.
    pub bottom_margin: i32,
}

impl Default for OverlayConfig {
    fn default() -> Self {
        Self {
            done_timeout: Duration::from_millis(900),
            error_timeout: Duration::from_millis(3500),
            bottom_margin: 56,
        }
    }
}

const BASE_W: i32 = 260;
const BASE_H: i32 = 40;

/// Must be created and driven on one thread with a message loop.
pub struct Overlay {
    hwnd: HWND,
    timer_hwnd: HWND,
    config: OverlayConfig,
    scale: f32,
    width: i32,
    height: i32,
    font: HFONT,
    state: OverlayState,
    visible: bool,
}

impl Overlay {
    pub const HIDE_TIMER: usize = 0x5717;

    /// `timer_hwnd` receives the auto-hide `WM_TIMER` and must forward it to
    /// [`Overlay::on_timer`]; `None` uses the pill window, where nothing handles it.
    pub fn create(config: OverlayConfig, timer_hwnd: Option<HWND>) -> windows::core::Result<Self> {
        // SAFETY: plain FFI query.
        let dpi = unsafe { GetDpiForSystem() }.max(96);
        let scale = dpi as f32 / 96.0;
        let width = (BASE_W as f32 * scale).round() as i32;
        let height = (BASE_H as f32 * scale).round() as i32;
        // SAFETY: a 'static window procedure and a hidden popup owned by this thread.
        let hwnd = unsafe {
            let inst = GetModuleHandleW(PCWSTR::null())?;
            let class = w!("wl-overlay-pill");
            let wc = WNDCLASSW {
                lpfnWndProc: Some(overlay_wndproc),
                hInstance: inst.into(),
                lpszClassName: class,
                ..Default::default()
            };
            RegisterClassW(&wc);
            CreateWindowExW(
                WS_EX_LAYERED
                    | WS_EX_NOACTIVATE
                    | WS_EX_TOOLWINDOW
                    | WS_EX_TOPMOST
                    | WS_EX_TRANSPARENT,
                class,
                w!("whisper-local"),
                WS_POPUP,
                0,
                0,
                width,
                height,
                None,
                None,
                Some(inst.into()),
                None,
            )?
        };
        let face = wide("Segoe UI");
        // SAFETY: creates a font object; deleted in Drop.
        let font = unsafe {
            CreateFontW(
                -(14.0 * scale).round() as i32,
                0,
                0,
                0,
                FW_SEMIBOLD.0 as i32,
                0,
                0,
                0,
                DEFAULT_CHARSET,
                OUT_DEFAULT_PRECIS,
                CLIP_DEFAULT_PRECIS,
                ANTIALIASED_QUALITY,
                0,
                PCWSTR(face.as_ptr()),
            )
        };
        Ok(Self {
            hwnd,
            timer_hwnd: timer_hwnd.unwrap_or(hwnd),
            config,
            scale,
            width,
            height,
            font,
            state: OverlayState::Hidden,
            visible: false,
        })
    }

    pub fn hwnd(&self) -> HWND {
        self.hwnd
    }

    pub fn state(&self) -> &OverlayState {
        &self.state
    }

    pub fn set(&mut self, state: OverlayState) {
        // SAFETY: our own window and timer id; killing a timer that is not set is a no-op.
        let _ = unsafe { KillTimer(Some(self.timer_hwnd), Self::HIDE_TIMER) };
        self.state = state;
        if self.state == OverlayState::Hidden {
            self.hide();
            return;
        }
        if let Err(e) = self.paint() {
            tracing::warn!(error = %e, "overlay paint failed");
            return;
        }
        if !self.visible {
            // SAFETY: our own window, shown and raised without activation.
            unsafe {
                let _ = ShowWindow(self.hwnd, SW_SHOWNOACTIVATE);
                let _ = SetWindowPos(
                    self.hwnd,
                    Some(HWND_TOPMOST),
                    0,
                    0,
                    0,
                    0,
                    SWP_NOACTIVATE | SWP_NOMOVE | SWP_NOSIZE,
                );
            }
            self.visible = true;
        }
        let timeout = match self.state {
            OverlayState::Done { .. } => Some(self.config.done_timeout),
            OverlayState::Error { .. } | OverlayState::Notice { .. } => {
                Some(self.config.error_timeout)
            }
            _ => None,
        };
        if let Some(t) = timeout {
            // SAFETY: a timer on our own window; WM_TIMER is routed by the UI loop.
            unsafe {
                SetTimer(
                    Some(self.timer_hwnd),
                    Self::HIDE_TIMER,
                    t.as_millis() as u32,
                    None,
                )
            };
        }
    }

    pub fn on_timer(&mut self, id: usize) {
        if id == Self::HIDE_TIMER {
            // SAFETY: our own window and timer.
            let _ = unsafe { KillTimer(Some(self.timer_hwnd), Self::HIDE_TIMER) };
            self.state = OverlayState::Hidden;
            self.hide();
        }
    }

    fn hide(&mut self) {
        if self.visible {
            // SAFETY: hiding our own window.
            let _ = unsafe { ShowWindow(self.hwnd, SW_HIDE) };
            self.visible = false;
        }
    }

    fn position(&self) -> POINT {
        let mut work = RECT::default();
        // SAFETY: `work` is a valid RECT out-parameter.
        let ok = unsafe {
            SystemParametersInfoW(
                SPI_GETWORKAREA,
                0,
                Some(&mut work as *mut RECT as *mut core::ffi::c_void),
                SYSTEM_PARAMETERS_INFO_UPDATE_FLAGS(0),
            )
        }
        .is_ok();
        if !ok {
            work = RECT {
                left: 0,
                top: 0,
                right: 1920,
                bottom: 1080,
            };
        }
        let margin = (self.config.bottom_margin as f32 * self.scale).round() as i32;
        POINT {
            x: work.left + (work.right - work.left - self.width) / 2,
            y: work.bottom - margin - self.height,
        }
    }

    fn paint(&self) -> windows::core::Result<()> {
        let (w, h) = (self.width, self.height);
        let text_left = (40.0 * self.scale) as i32;
        let text_right = if matches!(self.state, OverlayState::Listening { .. }) {
            w - (100.0 * self.scale) as i32
        } else {
            w - (16.0 * self.scale) as i32
        };
        let mask = self.text_mask(&self.state.label(), text_left, text_right)?;
        let pixels = compose(&self.state, w as usize, h as usize, self.scale, &mask);
        self.blit(&pixels)
    }

    /// Per-pixel coverage (0..=255).
    fn text_mask(&self, text: &str, left: i32, right: i32) -> windows::core::Result<Vec<u8>> {
        let (w, h) = (self.width, self.height);
        let surface = DibSurface::new(w, h)?;
        let mut units: Vec<u16> = text.encode_utf16().collect();
        // SAFETY: our own memory DC and font; the rect and text outlive the calls.
        unsafe {
            let old_font = SelectObject(surface.dc, self.font.into());
            SetBkMode(surface.dc, TRANSPARENT);
            SetTextColor(surface.dc, COLORREF(0x00FF_FFFF));
            let mut rect = RECT {
                left,
                top: 0,
                right,
                bottom: h,
            };
            DrawTextW(
                surface.dc,
                &mut units,
                &mut rect,
                DT_LEFT | DT_VCENTER | DT_SINGLELINE | DT_END_ELLIPSIS | DT_NOPREFIX,
            );
            SelectObject(surface.dc, old_font);
        }
        Ok(surface
            .pixels()
            .iter()
            .map(|p| (p >> 8 & 0xFF) as u8)
            .collect())
    }

    fn blit(&self, pixels: &[u32]) -> windows::core::Result<()> {
        let surface = DibSurface::new(self.width, self.height)?;
        surface.pixels_mut().copy_from_slice(pixels);
        let dst = self.position();
        let size = SIZE {
            cx: self.width,
            cy: self.height,
        };
        let src = POINT { x: 0, y: 0 };
        let blend = BLENDFUNCTION {
            BlendOp: AC_SRC_OVER as u8,
            BlendFlags: 0,
            SourceConstantAlpha: 255,
            AlphaFormat: AC_SRC_ALPHA as u8,
        };
        // SAFETY: every pointer is a local that outlives the call; the screen DC is released.
        unsafe {
            let screen = GetDC(None);
            let r = UpdateLayeredWindow(
                self.hwnd,
                Some(screen),
                Some(&dst),
                Some(&size),
                Some(surface.dc),
                Some(&src),
                COLORREF(0),
                Some(&blend),
                ULW_ALPHA,
            );
            ReleaseDC(None, screen);
            r
        }
    }
}

impl Drop for Overlay {
    fn drop(&mut self) {
        // SAFETY: destroying our own window and font on the creating thread.
        unsafe {
            let _ = DestroyWindow(self.hwnd);
            let _ = DeleteObject(self.font.into());
        }
    }
}

unsafe extern "system" fn overlay_wndproc(hwnd: HWND, msg: u32, wp: WPARAM, lp: LPARAM) -> LRESULT {
    const HTTRANSPARENT: isize = -1;
    if msg == WM_NCHITTEST {
        return LRESULT(HTTRANSPARENT);
    }
    // SAFETY: default handling for everything else.
    unsafe { DefWindowProcW(hwnd, msg, wp, lp) }
}

struct DibSurface {
    dc: HDC,
    bitmap: HBITMAP,
    old: windows::Win32::Graphics::Gdi::HGDIOBJ,
    bits: *mut u32,
    len: usize,
}

impl DibSurface {
    fn new(w: i32, h: i32) -> windows::core::Result<Self> {
        let info = BITMAPINFO {
            bmiHeader: BITMAPINFOHEADER {
                biSize: std::mem::size_of::<BITMAPINFOHEADER>() as u32,
                biWidth: w,
                biHeight: -h,
                biPlanes: 1,
                biBitCount: 32,
                biCompression: BI_RGB.0,
                ..Default::default()
            },
            ..Default::default()
        };
        let mut bits: *mut core::ffi::c_void = std::ptr::null_mut();
        // SAFETY: the DC and the DIB (which owns its memory) are released in Drop.
        unsafe {
            let dc = CreateCompatibleDC(None);
            let bitmap = CreateDIBSection(Some(dc), &info, DIB_RGB_COLORS, &mut bits, None, 0)
                .inspect_err(|_| {
                    let _ = DeleteDC(dc);
                })?;
            let old = SelectObject(dc, bitmap.into());
            let len = (w * h) as usize;
            std::ptr::write_bytes(bits as *mut u32, 0, len);
            Ok(Self {
                dc,
                bitmap,
                old,
                bits: bits as *mut u32,
                len,
            })
        }
    }

    fn pixels(&self) -> &[u32] {
        // SAFETY: the DIB memory holds `len` u32s and lives as long as `self`.
        unsafe { std::slice::from_raw_parts(self.bits, self.len) }
    }

    #[allow(clippy::mut_from_ref)]
    fn pixels_mut(&self) -> &mut [u32] {
        // SAFETY: as above; the surface is private to one call and never aliased.
        unsafe { std::slice::from_raw_parts_mut(self.bits, self.len) }
    }
}

impl Drop for DibSurface {
    fn drop(&mut self) {
        // SAFETY: restores the original bitmap and frees the objects we created.
        unsafe {
            SelectObject(self.dc, self.old);
            let _ = DeleteObject(self.bitmap.into());
            let _ = DeleteDC(self.dc);
        }
    }
}

/// Premultiplied BGRA (`0xAARRGGBB` little-endian), as `UpdateLayeredWindow` requires.
pub(crate) fn compose(
    state: &OverlayState,
    w: usize,
    h: usize,
    scale: f32,
    text: &[u8],
) -> Vec<u32> {
    let mut px = vec![0u32; w * h];
    let (wf, hf) = (w as f32, h as f32);
    let r = hf / 2.0;
    let bg = [24u8, 24, 28];
    let bg_alpha = 0.90;
    let dot_rgb = state.dot_rgb();
    let dot_cx = 20.0 * scale;
    let dot_r = 5.0 * scale;
    let level = match state {
        OverlayState::Listening { level } => Some(level.clamp(0.0, 1.0)),
        _ => None,
    };
    let bar_x0 = wf - 88.0 * scale;
    let bar_x1 = wf - 16.0 * scale;
    let bar_h = 6.0 * scale;
    for y in 0..h {
        for x in 0..w {
            let (fx, fy) = (x as f32 + 0.5, y as f32 + 0.5);
            let cx = fx.clamp(r, wf - r);
            let d = ((fx - cx).powi(2) + (fy - r).powi(2)).sqrt() - (r - 0.5);
            let cover = (0.5 - d).clamp(0.0, 1.0);
            if cover <= 0.0 {
                continue;
            }
            let mut rgb = [bg[0] as f32, bg[1] as f32, bg[2] as f32];
            let mut blend = |c: [u8; 3], a: f32| {
                for i in 0..3 {
                    rgb[i] = rgb[i] * (1.0 - a) + c[i] as f32 * a;
                }
            };
            let dd = ((fx - dot_cx).powi(2) + (fy - r).powi(2)).sqrt() - dot_r;
            let dot_cover = (0.5 - dd).clamp(0.0, 1.0);
            if dot_cover > 0.0 {
                blend(dot_rgb, dot_cover);
            }
            if let Some(level) = level
                && fx >= bar_x0
                && fx <= bar_x1
                && (fy - r).abs() <= bar_h / 2.0
            {
                let filled = bar_x0 + (bar_x1 - bar_x0) * level;
                let c = if fx <= filled {
                    [0x4C, 0xC2, 0x6E]
                } else {
                    [0x44, 0x44, 0x4C]
                };
                blend(c, 1.0);
            }
            let t = text.get(y * w + x).copied().unwrap_or(0) as f32 / 255.0;
            if t > 0.0 {
                blend([0xF2, 0xF2, 0xF2], t);
            }
            let a = cover * bg_alpha;
            let pm = |c: f32| (c * a).round().clamp(0.0, 255.0) as u32;
            let a8 = (a * 255.0).round() as u32;
            px[y * w + x] = a8 << 24 | pm(rgb[0]) << 16 | pm(rgb[1]) << 8 | pm(rgb[2]);
        }
    }
    px
}

#[cfg(test)]
mod tests {
    use super::*;

    fn channels(p: u32) -> (u32, u32, u32, u32) {
        (p >> 24, p >> 16 & 0xFF, p >> 8 & 0xFF, p & 0xFF)
    }

    #[test]
    fn corners_are_transparent_and_centre_is_opaque_enough() {
        let (w, h) = (260, 40);
        let px = compose(&OverlayState::Transcribing, w, h, 1.0, &vec![0; w * h]);
        assert_eq!(px[0], 0);
        assert_eq!(px[w - 1], 0);
        let (a, ..) = channels(px[20 * w + 130]);
        assert!(a > 200);
    }

    #[test]
    fn output_is_premultiplied() {
        let (w, h) = (260, 40);
        let mask: Vec<u8> = (0..w * h).map(|i| (i % 256) as u8).collect();
        let px = compose(&OverlayState::Listening { level: 0.5 }, w, h, 1.0, &mask);
        for p in px {
            let (a, r, g, b) = channels(p);
            assert!(r <= a && g <= a && b <= a, "{p:#x}");
        }
    }

    #[test]
    fn level_bar_fills_proportionally() {
        let (w, h) = (260, 40);
        let blank = vec![0; w * h];
        let low = compose(&OverlayState::Listening { level: 0.1 }, w, h, 1.0, &blank);
        let high = compose(&OverlayState::Listening { level: 0.9 }, w, h, 1.0, &blank);
        let green = |px: &[u32]| {
            px.iter()
                .filter(|&&p| {
                    let (_, r, g, _) = channels(p);
                    g > r + 60
                })
                .count()
        };
        assert!(green(&high) > green(&low) * 3);
    }

    #[test]
    fn overlay_window_is_created_hidden_and_never_activates() {
        let mut o = Overlay::create(OverlayConfig::default(), None).expect("overlay");
        use windows::Win32::UI::WindowsAndMessaging::{
            GWL_EXSTYLE, GetForegroundWindow, GetWindowLongW, IsWindowVisible,
        };
        // SAFETY: plain FFI queries on our own window.
        let ex = unsafe { GetWindowLongW(o.hwnd(), GWL_EXSTYLE) } as u32;
        for flag in [
            WS_EX_NOACTIVATE,
            WS_EX_TRANSPARENT,
            WS_EX_TOPMOST,
            WS_EX_LAYERED,
        ] {
            assert!(ex & flag.0 != 0);
        }
        // SAFETY: as above.
        let fg_before = unsafe { GetForegroundWindow() };
        o.set(OverlayState::Error {
            message: "test".into(),
        });
        // SAFETY: as above.
        assert!(unsafe { IsWindowVisible(o.hwnd()) }.as_bool());
        // SAFETY: as above.
        assert_eq!(unsafe { GetForegroundWindow() }, fg_before);
        o.set(OverlayState::Hidden);
        // SAFETY: as above.
        assert!(!unsafe { IsWindowVisible(o.hwnd()) }.as_bool());
    }
}
