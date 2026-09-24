# typing spike (2026-09-24)

Why the `KEYEVENTF_UNICODE` typing fallback in the clipboard-receipt spike turned
`spike typed run886453` into `spike 333333333333333`, and what a correct typing path has to
do. Standalone crate (own `[workspace]`), `windows` 0.62.2. Everything below was observed
on one machine on 2026-09-24 and is perishable.

**Machine:** Windows 11 Enterprise 10.0.26200, **used over RDP** (`SESSIONNAME=RDP-Tcp#0`),
keyboard layouts Swiss German (0807) and US (0409). Targets: Win11 Notepad 11.2607.14.0
(`RichEditD2DPT`) and a plain `EDIT` control in a window this process owns (on its own
UI thread), used as the control.

```
cargo build --release
target\release\typing.exe --help
target\release\typing.exe "spike typed run886453"            # the original repro, one SendInput
target\release\typing.exe --case long600 --chunk 1 --delay-ms 20 --repeat 5
target\release\typing.exe --suite [--target edit]
```

Each run clears the edit control with `WM_SETTEXT`, focuses the window, releases any
modifier `GetAsyncKeyState` reports down, types, then polls `WM_GETTEXT` until the text
matches or stops changing for 1.5 s. `\r\n` and `\r` in the read-back are normalised to
`\n`. The Notepad target is a tab on an empty temp file (`%TEMP%\typing-target-<pid>.txt`)
and is reused by later runs; it is left open, cleared.

## Root cause

**The INPUT construction was not the defect.** The old code already matched enigo 0.6:
one down and one up `INPUT` per UTF-16 unit, `wVk = 0`, `wScan` = the unit, surrogate
halves as two units, `cbSize = size_of::<INPUT>()`, a `dwExtraInfo` tag, one fresh
struct per event (no aliasing), and the space sent as a Unicode unit like everything else.
The same 42 events in one `SendInput` land correctly in a plain `EDIT` control in the same
RDP session, every time.

**The defect is sending more than one character before the target has consumed the
previous one.** Win11 Notepad translates a queued `VK_PACKET` keystroke with the *most
recently injected* character, not the one that keystroke carried. It keeps up with a
burst until it stalls; the stall comes at a word boundary. Evidence:

- `spiketypedrun886453` (no space) in one batch: PASS 2/2. With a space: every character
  after the first (sometimes second) space is the last character of the batch, 3/3.
- One `SendInput` per character with no gap fails the same way, so it is not about the
  batch: it is about how many characters are queued when Notepad stalls.
- At a 5 ms gap only the 2-3 characters queued during the stall repeat (`ppped uun886453`).
- `WM_NULL` round trips to the focus window after each character do not help
  (`--sync 1/2`: 0/6): Notepad answers sent messages during the stall while its input
  waits, so "the thread is responsive" is not "the input was consumed".
- Sending real layout keys instead of `VK_PACKET` (`--vk`) fixes the repeated character
  but not the underlying problem: the Shift state is read late too, so `Typed RUN` came
  out `typed run`, `(ok)` as `8ok9`, `!` as the Swiss-German dead key `¨`.
- Surrogate pairs are hit by the same thing without any word boundary: both halves in one
  call at a 10 ms gap gave `��🏽` (the high half became a second low half).

That the translation uses late state is inferred from these outputs; which component in
Notepad does it (the RichEdit/TSF autocorrect path is the obvious suspect, since word
boundaries trigger it) was not established, and Notepad's spellcheck/autocorrect settings
were not inspected or toggled.

**Not RDP-specific, as far as can be tested from inside RDP:** the control `EDIT` target
in the same session took 600 characters (1200 events) in one call, and every other case,
with no error; the Notepad failures track word boundaries, not session input. A
console-session run was not possible, so an RDP amplification of the timing cannot be
ruled out.

## Results

`chunk` = characters per `SendInput` call (`all` = one call), `delay` = ms between calls,
`confirm` = wait for the read-back to show the chunk before sending the next (measurement
only). One suite pass per row unless a pass rate is given. Case texts are in
`builtin_cases()`; `long600` is a 600-character paragraph.

**Control: own `EDIT` control** - every case PASS in every mode, including `long600` in one
call (8 ms in `SendInput`, 0.6 s to land) and per character with confirm (arrive p50 1.9 ms,
max 4.9 ms).

**Win11 Notepad**

| mode | repro | ascii | intl (umlauts, CJK) | emoji (pairs) | newlines (Return) | long600 |
|---|---|---|---|---|---|---|
| chunk all, delay 0 | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| chunk 1, delay 0 | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL (and 323 chars lost) |
| chunk 8, delay 20 | FAIL | FAIL | FAIL | FAIL | FAIL | FAIL |
| chunk 1, delay 5 | 0/2 | | | | | |
| chunk 1, delay 10 | 1/1 | 11/11 | 1/1 | **0/11** | 1/1 | **5/6** |
| chunk 1, delay 20 | 3/3 | 11/11 | 1/1 | **10/14** | 1/1 | 6/6 |
| chunk 1, delay 30 | | 10/10 | | 10/10 | 2/2 | 5/5 |
| per unit, delay 20 | | | | 13/13 | | |
| chunk 1, confirm | 1/1 | FAIL (char lost) | 1/1 | FAIL | 1/1 | FAIL (char lost) |

Cells are passes/runs across the suite and the repeated runs (`--repeat`); blank = not run.
The 10-run ASCII rows used the ascii case without its quote characters, which PowerShell
5.1 mangles when passing arguments to a native program.

Failure shapes: at delay 10-20 the failures are a **single dropped character right after
a space** (`busy spell` -> `usy ppell`, `done` -> `one`), sometimes followed by one
repeat. In confirm mode (next character sent as soon as the previous one shows up, about
5-6 ms per character) Notepad twice swallowed the first character after a space
outright: it never appeared within 3 s. So reading back each character is not enough on
its own; the gap after a word boundary matters.

**Newlines:** `\n` as a `VK_RETURN` press (`\r` dropped, so `\r\n` is one Return) PASSes in
both targets. `\n` sent as a Unicode unit is dropped by Notepad (`line oneline two`) but
works in `EDIT`. The product should send Return for `\n`, as planned.

## Limits

- **Batch size:** one `SendInput` of 10,500 events (5,250 chars) into `EDIT` inserted only
  the first ~5,000 characters, and 42,000 events stopped at exactly 5,000; `SendInput`
  reported every event as inserted both times. That is consistent with the 10,000-message
  per-thread queue quota, so the loss is silent. Keep far below it; pacing does that anyway.
- **Rate:** the only settings that passed repeatedly in Notepad were one character per
  call with >= 20 ms between calls (emoji needed 30 ms, or 20 ms with surrogate halves in
  separate calls). That is 33-50 characters per second: a 600-character paragraph takes
  12-18 s.
- **`SendInput` itself sometimes blocked:** 29 calls at a 30 ms gap took 2.3 s instead of
  0.9 s in one Notepad run. Not investigated (a slow low-level keyboard hook somewhere is
  the usual cause).
- **No receipt:** the product cannot read back arbitrary targets, and the one signal that
  is generic (`WM_NULL` round trip) was shown not to mean the input was consumed. Typing
  cannot produce a confirmed receipt.
- **Only two targets tested.** Elevated targets (UIPI rejects injection, which
  `SendInput` does report), consoles, Chromium/Electron, WPF, games and other IMEs were
  not tried.

## Rules for the product's typing path

1. Per character: one `KEYEVENTF_UNICODE` down + up per UTF-16 unit, `wVk = 0`, tagged
   `dwExtraInfo`. `\n` is a `VK_RETURN` press, `\r` is dropped.
2. **Never queue more than one character ahead of the target.** One character per
   `SendInput`; surrogate halves in separate calls; >= 20-30 ms between calls. Paste stays
   the primary path; typing is slow by construction.
3. Release held modifiers first; check the return count of every call; stop if the target
   loses the foreground.
4. Typing is unconfirmed: the clipboard copy and the user-visible notice still apply.

## Not tested

- A non-RDP console session.
- Notepad with spellcheck/autocorrect toggled, and whether the late translation is TSF.
- The `--newline skip` run in `EDIT` and a re-run of the slow-`SendInput` Notepad case:
  the RDP session was disconnected before they could be repeated, so no window could be
  made foreground.
- One unexplained run: the very first `EDIT` repro (launched from Git Bash) read back 220
  characters starting with a space and kept changing for 6 s. Five fresh launches and
  every later run were clean.
