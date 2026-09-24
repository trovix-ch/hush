# clipboard-receipt spike (2026-09-24)

Milestone-0 spike for D8: which clipboard read signals arrive when we paste through a
delayed-rendered `CF_UNICODETEXT` entry. Standalone crate (own `[workspace]`), `windows`
0.62.2. Everything below was observed on one machine on 2026-09-24 and is perishable.

**Machine:** Windows 11 Enterprise 10.0.26200, **used over RDP** (`SESSIONNAME=RDP-Tcp#0`,
`rdpclip.exe` running). Targets: Win11 Notepad 11.2607 (`RichEditD2DPT`), `conhost.exe
cmd.exe`, Windows Terminal (`wt.exe -w new`), PowerShell ISE. Clipboard history was
**off** (no `EnableClipboardHistory` value) except in the one run that enabled it.

```
cargo build --release
target\release\clipboard-receipt.exe --help
# e.g. paste into the Notepad window whose title contains spike-target.txt and read it back:
clipboard-receipt --countdown 0 --observe 4 --focus-class Notepad --focus-title spike-target.txt --verify
```

Flags: `--eager` (real data, control), `--also-cf-text`, `--probe-cf-text`, `--no-exclude`
(skip the history/cloud exclusion formats), `--rearm` (re-set the delayed entry after
each render, max 20), `--write-before-focus` (minimise the target, write, then activate
it), `--chord ctrl-v|ctrl-shift-v|shift-insert|none|type`. The chord is not sent if the
target is not foreground. Each run prints a timeline (ms relative to the chord
`SendInput`) and a `SUMMARY` line.

## Method

Each run: focus the target (AttachThreadInput + SetForegroundWindow), write the entry,
pump 300 ms, send the chord, pump 4 s. Readers identified via `GetOpenClipboardWindow`
during `WM_RENDERFORMAT`. "Landed" = `WM_GETTEXT` on Notepad's `RichEditD2DPT` contains the
run nonce; other targets were not read back. Matrix run twice: once as-is (rdpclip live),
once with rdpclip stopped for the duration (restarted afterwards) to see the target's own
reads.

## Observed signals

Times are ms after the chord `SendInput`; negative = before the chord. Every row is one run.

| Target / chord | Mode | rdpclip | First render | Renders | Reader(s) | Landed |
|---|---|---|---|---|---|---|
| Notepad / Ctrl+V | delayed | live | **-306** | 1 | rdpclip.exe | yes |
| Notepad / Ctrl+V | delayed + CF_TEXT | live | -305 | 1 (UNICODE only) | rdpclip.exe | yes |
| Notepad / Ctrl+V | eager | live | - | 0 | - | yes |
| conhost, WT, ISE / Ctrl+V | delayed, +CF_TEXT | live | -302 to -306 | 1 | rdpclip.exe | not read back |
| Notepad / Ctrl+V | delayed | stopped | +1.9 | 1 | Notepad.exe | yes |
| Notepad / Ctrl+V | delayed + CF_TEXT | stopped | +0.8 | 1 (UNICODE only) | Notepad.exe | yes |
| Notepad / Ctrl+V | eager | stopped | - | 0 | - | yes |
| conhost / Ctrl+V | delayed | stopped | +1.5 | 1 | "cmd.exe" (console window reports its client) | not read back |
| conhost / Ctrl+V | delayed + CF_TEXT | stopped | +0.5 | 1 (UNICODE only) | "cmd.exe" | not read back |
| conhost / Ctrl+Shift+V | delayed | stopped | **none** | 0 | - | not read back |
| conhost / Shift+Insert | delayed | stopped | +0.7 | 1 | "cmd.exe" | not read back |
| WT / Ctrl+V, Ctrl+Shift+V, Shift+Insert | delayed (+CF_TEXT) | stopped | +0.8 to +1.0 | 1 | null (opens with NULL hwnd) | not read back |
| ISE / Ctrl+V | delayed | stopped | +2.3 | 1 | powershell_ise.exe | not read back |
| ISE / Ctrl+V | delayed + CF_TEXT | stopped | +1.0 | 1 (UNICODE only) | powershell_ise.exe | not read back |
| conhost, WT, ISE / Ctrl+V | eager | both | - | 0 | - | not read back |
| Notepad, write then activate | delayed | stopped | +2.3 | 1 | Notepad.exe (nothing on activation) | yes |
| Notepad, write then activate | delayed | live | -657 | 1 | rdpclip.exe | yes |
| Notepad / Ctrl+V, `--rearm` | delayed | live | -307 | 21 (cap) | rdpclip.exe every ~15 ms, incl. +1.4 | yes |
| Notepad / Ctrl+V, `--rearm` | delayed | stopped | +0.9 | 1, then entry stays delayed | Notepad.exe | yes |

**CF_TEXT probe** (`--probe-cf-text`, 3 runs; a second thread wins the open race before
rdpclip): with only delayed `CF_UNICODETEXT` set, `EnumClipboardFormats` already lists
`CF_LOCALE, CF_TEXT, CF_OEMTEXT`; `IsClipboardFormatAvailable(CF_TEXT)` is true;
`GetClipboardData(CF_TEXT)` triggers `WM_RENDERFORMAT(CF_UNICODETEXT)` in the owner and
returns the converted text. No target in the matrix ever asked for `CF_TEXT` even when it
was offered explicitly.

**Clipboard history** (`EnableClipboardHistory=1` for ~15 s, rdpclip stopped, no chord,
then restored to absent): with the exclusion formats, **no read in 4 s**; with
`--no-exclude`, `svchost.exe` (cbdhsvc) rendered **~208 ms after the write**. rdpclip
ignores the exclusion formats.

**Other facts**
- A delayed format renders **once** per write. Later readers get the cached data and
  produce no signal.
- Rendering does not change `GetClipboardSequenceNumber`; the write itself advanced it by
  5-9. `--rearm` advances it per re-arm.
- `WM_RENDERALLFORMATS` arrives when the owner window is destroyed, even after the only
  delayed format was already rendered.
- With the `windows` crate, `SetClipboardData(fmt, None)` returns `Err` (stale last-error)
  on a successful delayed registration.
- `--chord type` (`KEYEVENTF_UNICODE`, 42 events in one `SendInput`) into Notepad produced
  `spike 333333333333333` for `spike typed run886453`. Everything after the space became
  the last character. This happened twice. **Explained in `../typing/README.md`
  (2026-09-24):** the events were built correctly; Win11 Notepad translates a queued
  `VK_PACKET` with the last injected character once it stalls at a word boundary, so any
  batch is unsafe.
- **Changed 2026-09-24:** `--chord type` now sends one code unit (down + up) per
  `SendInput` with a 30 ms message-pumping gap, and stops if the target loses the
  foreground. The timeline's chord time is taken before the first unit; the chord line
  still counts events (two per unit). Not re-run here after the change: the RDP session was disconnected. The same
  pacing passed 10/10 for ASCII and emoji and 5/5 for 600 characters in the typing spike.

## Not tested (needed a human or was out of scope)

- A non-RDP console session. Stopping rdpclip only approximates one.
- Read-back for conhost, WT and ISE, so for these targets a render is not proof the text
  was inserted.
- Third-party clipboard managers (by instruction), Electron/Chromium, Java, UWP and
  elevated targets, and apps that read late or twice.
- Physically held modifiers. None were down, so the release path never ran.
- Clipboard history with a real paste target, and the Win+V UI itself.
- Whether rdpclip's eager read sends the text to the RDP client's clipboard.
- Noise from the live desktop: during one run Firefox took over the clipboard, and in
  another an unidentified `powershell.exe` read our entry 45 ms after a chord that was
  aborted because focus had moved.
