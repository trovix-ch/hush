# Input and output subsystems for a Rust push-to-talk app on Windows 11 (research, 2026-09-24)

Perishable: crate versions from crates.io on the date above. Claims not checked against a source are marked **[unverified]**.

## 1. Global push-to-talk hotkey

| Mechanism | Key release? | Modifier-only (RCtrl)? | Can block the key? | Notes |
|---|---|---|---|---|
| `RegisterHotKey` | No (press only) | No | Yes | `global-hotkey` 0.8.0 fakes release by polling `GetAsyncKeyState` every 50 ms; cannot express modifier-only |
| `WH_KEYBOARD_LL` via `SetWindowsHookExW` | Yes | Yes | Yes: return nonzero instead of `CallNextHookEx` | see constraints |
| Polling `GetAsyncKeyState` | Only as fast as you poll | Yes | No | release watchdog only |
| Raw Input (`RIDEV_INPUTSINK`) | Yes | Yes | No | monitoring only |

LL hook constraints ([LowLevelKeyboardProc](https://learn.microsoft.com/en-us/windows/win32/winmsg/lowlevelkeyboardproc)): callback runs on the installing thread, which **must run a message loop**; exceeding `LowLevelHooksTimeout` (max 1000 ms on Win10 1709+) **silently removes the hook with no notification**; do not call `GetAsyncKeyState` inside the callback.

Crates: **`handy-keys` 0.3.4** (2026-08-07, what Handy uses: LL hook, modifier-only, blocking, down+up); `win-hotkeys` 0.5.1 (little usage); `rdev` 0.5.3 (2023, `unstable_grab`); `inputbot` 0.6.0 (2023); `device_query` 4.0.1 (polling).

**Recommendation:** write the hook yourself (~150 lines on `windows`), or use `handy-keys`. Details:
- **Ignore own injected input:** skip `KBDLLHOOKSTRUCT.flags & LLKHF_INJECTED` or a `dwExtraInfo` tag, or the synthetic Ctrl+V retriggers the hotkey.
- **Swallow key-up with key-down**, and filter auto-repeat key-downs while held.
- **RCtrl / RAlt / CapsLock:** match `vkCode == VK_RCONTROL` (or scan code + `LLKHF_EXTENDED`). AltGr arrives as LCtrl+RAlt **[unverified]**. CapsLock can be swallowed so its state never toggles.
- **Alt or Win as hotkey:** a swallowed press still leaves "Alt down/up with nothing between" → opens menus / Start. AutoHotkey injects a mask key; vkE8 is unassigned ([A_MenuMaskKey](https://www.autohotkey.com/docs/v2/lib/A_MenuMaskKey.htm)).
- **Fn cannot be used** (firmware) **[unverified, widely known]**.
- **Toggle mode:** tap-vs-hold on the same hook; release within ~250 ms toggles, longer hold is PTT.
- **UIPI:** a medium-integrity hook does not see keystrokes while an elevated window has focus. Only running elevated or `uiAccess=true` (signed, under Program Files) helps. Detect the elevated foreground window and tell the user.

## 2. Text insertion

**(a) `SendInput` + `KEYEVENTF_UNICODE`:** `wVk` must be 0; non-BMP chars as two surrogate events. `enigo` 0.6.1 does this, batching into one `SendInput`, mapping `\n`→Return and `\t`→Tab, tagging via `dwExtraInfo`. One `SendInput` call is not interleaved with user typing. **Fails silently under UIPI.** Does not reset held modifiers (a held Ctrl turns "a" into Ctrl+A). Fails in: DirectInput/Raw Input games; apps that mishandle surrogate pairs or treat chars as keystrokes (IDE autocomplete/autoindent, chat apps sending on `\n`); long text (slow, cut off by focus change); some Java/Electron **[unverified]**.

**(b) Clipboard + Ctrl+V:** default in Wispr Flow, Handy and most clones.
- Handy: `Direct`, `CtrlV`, `CtrlShiftV`, `ShiftInsert`, `ExternalScript`, `None`; sleeps before the chord and again before restoring; saves only text or one image ([clipboard.rs](https://raw.githubusercontent.com/cjpais/Handy/main/src-tauri/src/clipboard.rs)).
- **Handy "Reliable Paste" uses delayed rendering:** `SetClipboardData(CF_UNICODETEXT, NULL)`, treats the target's `WM_RENDERFORMAT` as the receipt, counts reads only after the chord, restores 200 ms after the last read with an 8 s ceiling (500 ms if injection failed), skips restore if the user copied meanwhile ([paste_tx](https://raw.githubusercontent.com/cjpais/Handy/main/src-tauri/src/paste_tx/mod.rs)). **Copy this design.**
- **Hide from clipboard history:** registered formats `ExcludeClipboardContentFromMonitorProcessing`, `CanIncludeInClipboardHistory`=0, `CanUploadToCloudClipboard`=0 ([MS](https://learn.microsoft.com/en-us/windows/win32/dataxchg/clipboard-formats)); arboard exposes them as `SetExtWindows::exclude_from_*`.
- **Failures reported:** TUIs binding Ctrl+V to "paste image" (Handy #918, Codex CLI in CMD); app regressions on synthetic Ctrl+V (Claude Code 2.1.83 + Wispr, [#38620](https://github.com/anthropics/claude-code/issues/38620)); terminals/IDEs wanting Ctrl+Shift+V; RDP/Citrix without redirection; rich-text targets pasting formatting (set only `CF_UNICODETEXT`); transcript left on the clipboard after a failed paste.

**(c) UI Automation:** `ValuePattern.SetValue` only on single-line edits and it *replaces* the value; `TextPattern` is read-only ([MS](https://learn.microsoft.com/en-us/dotnet/framework/ui-automation/ui-automation-textpattern-overview)). UIA is for **reading context** (selection, surrounding text, `IsPassword`), not inserting.

**(d) `WM_CHAR`/`WM_SETTEXT`:** classic Win32 edits only; last resort.

**Recommended chain:**
1. Pre-checks: elevated foreground while we are not → error; focused element is a password field → refuse.
2. Default: delayed-render clipboard paste; chord from per-app config (Ctrl+V default; Ctrl+Shift+V / Shift+Insert for known terminals/IDEs); wait for the `WM_RENDERFORMAT` receipt.
3. No receipt within ~500 ms → fall back to `SendInput` Unicode typing, unless the app is on a "never type" list (avoid double insertion).
4. Per-app override keyed on exe: `paste` | `type` | `paste:ctrl+shift+v` | `none` (copy only + toast).
5. Very short texts could default to typing **[design opinion]**.

## 3. Focused app and control
- Process: `GetForegroundWindow` → `GetWindowThreadProcessId` → `OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION)` → `QueryFullProcessImageNameW`.
- Elevation: `OpenProcessToken` + `GetTokenInformation(TokenElevation)`; treat `OpenProcess` failure as "elevated/unknown" **[unverified edge]**.
- Focused control and caret: `GetGUIThreadInfo(tid)` → `hwndFocus`, `hwndCaret`, `rcCaret` (client coords). Chromium/Electron/WPF often report no system caret; position the overlay from UIA or the cursor instead **[unverified]**.
- UIA: `uiautomation` **0.25.1** (2026-09-04, on `windows` 0.62.2): `get_focused_element()`, `is_password()`, `get_control_type()`, `get_process_id()`, `get_name()`, `get_native_window_handle()`. **UIA calls can block for seconds on a hung target: run on a dedicated MTA COM thread with a timeout, never on the hook thread.**

## 4. Crates (crates.io, 2026-09-24)
| crate | version | last update | notes |
|---|---|---|---|
| `windows` / `windows-sys` | 0.62.2 / 0.61.2 | 2025-10-06 | `windows` has COM wrappers (needed for UIA); `windows-sys` raw FFI, faster builds |
| `enigo` | 0.6.1 | 2025-08-28 | Unicode typing, surrogate pairs, one `SendInput` per `text()` |
| `arboard` | 3.6.1 | 2025-08-23 | text/HTML/image/files; history exclusion; **cannot enumerate arbitrary formats** |
| `clipboard-win` | 5.4.1 | 2025-07-17 | `EnumFormats`, `RawData`, `seq_num`, retrying open — **use for full save/restore** |
| `uiautomation` | 0.25.1 | 2026-09-04 | active |
| `global-hotkey` | 0.8.0 | 2026-05-01 | no modifier-only, polled release |
| `handy-keys` | 0.3.4 | 2026-08-07 | LL hook, modifier-only, blocking |
| `tray-icon` / `muda` | 0.25.1 / 0.20.0 | 2026-09 | needs an event loop on its thread |
| `winit` | 0.30.13 | 2026-09-04 | |
| `rodio` | 0.22.2 | 2026-03-05 | |
| `directories` | 6.0.0 | 2025-01 | |
| `auto-launch` | 0.6.0 | 2026-01 | |
| `single-instance` | 0.3.3 | 2021 | stale; use a named mutex yourself |

Clipboard save/restore with `clipboard-win`: enumerate formats and copy bytes; skip GDI handle formats (`CF_BITMAP`, `CF_ENHMETAFILE`, `CF_PALETTE`), rely on synthesized DIB; delayed-rendered data from other apps is materialised on read and can be huge (Excel ranges): cap the size.

## 5. Tray, overlay, single instance, startup, paths
- **Tray:** `tray-icon` + `muda` on the UI thread's message loop.
- **Overlay pill:** layered popup with `WS_EX_NOACTIVATE | WS_EX_TOOLWINDOW | WS_EX_TOPMOST | WS_EX_LAYERED | WS_EX_TRANSPARENT` and `WS_POPUP`; show with `SW_SHOWNOACTIVATE` / `SWP_NOACTIVATE`; draw with `UpdateLayeredWindow` + premultiplied BGRA (tiny-skia or Direct2D). Test over exclusive-fullscreen games **[unverified]**.
- **GUI stack [opinion]:** plain Win32 + tray-icon is smallest; egui for a settings window on demand; Tauri heavier (WebView2); Slint middle; Iced weakest for no-activate topmost. Reasonable split: raw Win32 overlay + tray, separate settings window created on demand.
- **Single instance:** named mutex `Local\<AppId>` via `CreateMutexW` + `ERROR_ALREADY_EXISTS`; signal the running instance via a message-only window (`FindWindowW`) or named pipe.
- **Startup:** `HKCU\...\Run` (what `auto-launch` does) or a scheduled task; no admin needed.
- **Paths:** config in Roaming (`ProjectDirs::config_dir`); **models in `data_local_dir`** (`%LOCALAPPDATA%`) so they don't roam.

## 6. Sound
`PlaySoundW(SND_MEMORY | SND_ASYNC)` with an `include_bytes!` WAV is lightest. `rodio` 0.22.2 is fine too (keep an output stream pre-opened). **Play the start beep before opening the mic, or trim it from capture**, or it ends up in the transcription.

## Recommended design
**Threads:**
1. **Hook thread** (`std::thread`, never Tokio): `SetWindowsHookExW(WH_KEYBOARD_LL, …)` then `GetMessageW` loop. Proc only: compare vk/scan against an atomic hotkey config, ignore `LLKHF_INJECTED`, `try_send` `Event{Down|Up, t}` on a channel, return 1 to swallow or `CallNextHookEx`. No locks, allocation or logging. **Watchdog** for a silently removed hook: heartbeat, or compare `GetAsyncKeyState` against the hook's state while recording; reinstall on disagreement.
2. **UI thread:** tray, overlay, the clipboard owner window (must receive `WM_RENDERFORMAT`), message loop; receives commands via `PostMessageW(WM_APP+n)` with an `Arc` payload queue.
3. **Async runtime:** state machine Idle→Recording→Transcribing→Inserting. On Down: snapshot context (`GetForegroundWindow`, HWND, pid/exe, elevation; UIA lookup on the COM worker with 150 ms timeout), start capture and beep. On Up: stop, transcribe, insert.
4. **COM/UIA worker:** dedicated thread, `CoInitializeEx(COINIT_MULTITHREADED)`.
5. **Insertion:** verify the foreground window is still the one captured at Down (else abort + toast "focus changed; text copied"); release stuck modifiers; run the strategy chain; all `SendInput` as one batch tagged with our `dwExtraInfo`.

**Manual test matrix:** elevated targets (admin PowerShell, Task Manager); a stalled hook callback (`sleep(1200)`) and watchdog recovery; RCtrl, Alt-based and CapsLock hotkeys; auto-repeat and 60 s+ holds; other keys while holding PTT; Windows Terminal, conhost, Claude Code / Codex TUIs, WSL vim, PuTTY; browser fields, contenteditable (Google Docs, Notion), Slack/Discord (Enter sends), VS Code autoindent, Word rich paste; emoji and CJK via typing; clipboard image / Explorer files / Excel range survive restore, Win+V history clean; Ditto running; RDP/Citrix with redirection on and off; fullscreen games; password fields (UIA and web); focus change during transcription, Alt+Tab during typing; lock screen / UAC during recording; sleep/resume; second instance.
