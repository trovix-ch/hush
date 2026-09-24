# whisper-local: design and decisions

Last revised 2026-09-24. Sections marked *(perishable)* name crates, models or machine
state and must be re-verified before being trusted after 2026-12. Sections without that
mark are product decisions and hold until explicitly changed.

Research backing this document is in `docs/research/` (dated).

## 1. What we are building

A Windows dictation tool in the shape of Wispr Flow, running entirely on the user's
machine:

1. The user holds a global hotkey anywhere in the OS and speaks.
2. On release, the speech is transcribed locally.
3. The transcript is normalized locally: filler words removed, self-corrections applied
   ("Tuesday, no wait, Wednesday" → "Wednesday"), punctuation and casing fixed, the
   user's vocabulary spelled correctly, style adapted to the target app.
4. The result is inserted into whatever control had keyboard focus.

Target end-to-end latency from key release to text on screen: **under one second** on
the first-tier hardware, and the user should perceive the tool as instantaneous for
short utterances.

### Non-goals for v1
- macOS and Linux. The core crates stay platform-neutral, but nothing is tested there.
- Cloud backends of any kind. Every byte of audio and text stays on the machine.
- Live streaming preview while speaking (see decision D6).
- Voice *control* of the OS (Talon-style). This is dictation.
- Screenshots or reading full window contents as context. Privacy is the point.

### Hardware tiers
- **Tier 1 (first):** discrete NVIDIA/AMD GPU with 8 GB+ VRAM. Everything on by default.
- **Tier 2 (later):** CPU-only laptop. Same pipeline, smaller speech model, LLM cleanup
  replaced by a rules-plus-punctuation pass by default and offered as an opt-in with
  honest latency.

Nothing in the architecture may assume Tier 1; tiering is a matter of which engine
implementations get selected.

## 2. User experience (v1)

- **Hold-to-talk** on a configurable key. Default: **Right Ctrl**. A modifier-only key
  avoids the Start-menu and menu-bar side effects of Win/Alt chords and rarely conflicts.
  The key is swallowed so the target app never sees it.
- **Double-tap** the key to lock into hands-free mode; tap again to stop. Escape cancels.
- A small **overlay pill** near the bottom of the screen shows listening / thinking
  states and a live level meter. It never takes focus.
- A short **start/stop sound**, played before the microphone opens.
- **Tray icon** with: pause, paste last transcript, copy last transcript, open config,
  quit. No main window in v1; configuration is a TOML file opened in the editor.
- **Personal dictionary**: a list of words and phrases spelled exactly as the user wants.
- **Per-app rules** keyed on executable name: paste method, style (formal / casual /
  code / none), and "never insert here".
- When insertion cannot work (elevated target, password field, focus changed during
  transcription), the text lands on the clipboard and the pill says so. Text is never
  silently lost.

## 3. Pipeline

```
hotkey down ─► snapshot focus context ─► start capture (pre-roll included) ─► overlay: listening
      │                                                   │
      │                          VAD marks speech segments; closed segments are transcribed
      │                          while the key is still held (segment pre-transcription)
      ▼
hotkey up ──► stop capture ─► transcribe remaining tail ─► stitch transcript
                                                              │
                                    ┌─────────────────────────┘
                                    ▼
                    rule pass (spoken punctuation, commands, stutters, obvious fillers)
                                    │
                    LLM pass if enabled and the utterance warrants it
                                    │
                    validation (containment vs. source, length ratio, language kept)
                        failed ─► fall back to rule-pass output
                                    │
                                    ▼
                    insert (strategy chain, see D8) ─► overlay: done / error
```

State machine: `Idle → Recording → Transcribing → Normalizing → Inserting → Idle`, with
`Cancelled` reachable from any active state. Only one utterance is in flight at a time;
a hotkey press during `Transcribing` or later is queued as the next recording, not
dropped.

## 4. Decisions

Each decision records what was chosen, why, and what was rejected. Numbering is stable;
supersede by adding a new decision, not by editing history.

### D1. Rust workspace, no web-view UI
Rust end to end. No Tauri/WebView2 in v1: the app is mostly invisible, and a WebView
costs a few hundred MB of idle memory and a large dependency surface for a settings
page nobody sees. The overlay and tray are raw Win32. A settings window, if we add one,
is a separate on-demand process or an egui window, decided later.

Rejected: Tauri (Handy's choice; heavy), Electron (not Rust), Python (packaging and
latency).

### D2. Speech engine: Parakeet TDT first, Whisper second *(perishable)*
Default engine: **NVIDIA Parakeet TDT 0.6B v3** through ONNX Runtime, using the
**DirectML** execution provider on GPU and the CPU provider otherwise. Reasons: best
measured accuracy-per-millisecond of any local engine (6.3% average WER on the Open ASR
leaderboard; ~0.03 real-time factor on a desktop CPU), built-in punctuation and casing,
25 European languages, and DirectML runs on any DirectX 12 GPU without a CUDA install.

Second engine: **Whisper large-v3-turbo** through whisper.cpp with the **Vulkan**
backend, for languages outside Parakeet's 25 and for vocabulary prompting. It is a
milestone-3 addition, not part of the first vertical slice.

Both sit behind one `SttEngine` trait; the app selects by config and hardware probe and
reports which backend actually loaded (ONNX Runtime falls back to CPU silently unless
told to fail).

Rejected: CUDA execution providers as the default (CUDA 13 + cuDNN 9 runtime to
redistribute; RTX 50-series needs CUDA 12.8+ and explicit architecture flags), candle
Whisper (no evidence it is competitive), Windows built-in speech (cloud or Copilot+
only), Kyutai/Voxtral (servers, not libraries, as of the research date).

To evaluate early: `transcribe-cpp` (one ggml/Vulkan backend covering Parakeet, Whisper
and others). If it holds up it may replace both paths.

### D3. Voice activity detection runs in the app, not in the engine
Silero VAD (v6 ONNX) runs on the 16 kHz stream in the app. It trims silence with ~200 ms
padding, drops recordings with no speech (Whisper hallucinates on silence), marks
segment boundaries for pre-transcription, and can auto-stop hands-free mode after a
configurable silence. Every engine benefits; no engine's built-in VAD is used.

### D4. Normalization is a chain: rules, then LLM, then validation
1. **Rule pass** always runs: spoken punctuation and "new paragraph", stutter collapse,
   language-specific standalone fillers, explicit correction cues handled conservatively.
   Zero latency; it is the entire Tier-2 default together with a punctuation model.
2. **LLM pass** runs when enabled and the utterance is long enough or contains a
   correction cue. The prompt frames the transcript as *data*, never as a request, with
   few-shot pairs including a question that must not be answered. Vocabulary, app style
   and the previously inserted sentence go after the fixed prefix so the prefix state can
   be cached.
3. **Validation** gates the LLM output before it can be inserted. It is order-aware,
   not a bag-of-words check (a set-containment check only catches deletions; an
   appended "I don't know" built from source words passes it). Checks, all cheap at
   the ~100-token scale:
   - token alignment (LCS or token edit distance) of the candidate against the
     rule-pass output; any run of N or more consecutive content tokens with no match in
     the source is an *insertion* and rejects;
   - protected tokens must survive unchanged: numbers, negations, dictionary entries,
     URLs and code-like identifiers;
   - token length ratio within a lower *and* upper bound (heavy cleanup legitimately
     removes half the words; the bounds are tuned on the fixture corpus, not guessed);
   - no preamble, no trailing explanation, same language and script;
   - deletion ratio of content tokens below a bound.
   On rejection the rule-pass output is inserted and the provenance records why.

   This is defence in depth, not a proof. The prompt's few-shot examples are the first
   line against answering; validation is the last. Adversarial fixtures (dictated
   questions, imperatives, prompt injections) are part of the normal test run and the
   residual risk is stated honestly in the docs, not claimed away.

4. **Any LLM failure falls back to the rule pass.** Timeout, unreachable backend,
   malformed output, validation rejection: all of them insert the rule-cleaned text and
   show a subtle indicator. None of them produce an error instead of text.

Spoken commands ("delete that", "new paragraph") are handled by rules, never by the LLM.

### D5. LLM backend: OpenAI-compatible HTTP first, embedded llama.cpp next *(perishable)*
The `Normalizer` trait has three implementations, in build order:
1. `Rules` (always present).
2. `OpenAiHttp`: any local OpenAI-compatible server. This machine already runs Ollama,
   so the vertical slice gets real LLM cleanup with zero build risk, and prompt and
   validation logic is proven before we take on a C++ build.
3. `LlamaCpp` embedded via `llama-cpp-2` with the **Vulkan** backend (no CUDA runtime
   to ship; same backend as the Whisper engine). This is what ships to end users: a
   bundled model, no setup. Keeps one resident context, snapshots the system-prompt
   state, decodes greedily with a hard token cap and an anti-preamble grammar.

Default model for evaluation: **Qwen3-4B-Instruct-2507 Q4_K_M**, A/B against
**Granite 4.0 Micro** (Apache 2.0). Both are dense transformers, so prefix caching works;
hybrid architectures (Qwen3.5 small, Granite-H) had open prompt-cache bugs in llama.cpp
at the research date. Expected 0.5–0.8 s per 100 output tokens on a 4090-class GPU.

Longer term: LoRA fine-tune of a 1.7–2B model on synthetic disfluency data. Every
serious product (Wispr, superwhisper's S1-mini, VoiceInk's fine-tune) went this way.

Two caveats the review panel made explicit:
- The HTTP path proves prompt *correctness*, not latency and not grammar safety. It
  cannot snapshot the prefix state and most servers offer no grammar constraint, so an
  HTTP latency number is a pessimistic bound and the HTTP configuration is the weakest
  safety configuration. It is a development backend, not the shipping default.
- The prompt contract is backend-neutral and versioned: exact message serialization,
  output-only semantics, token cap, stop behaviour, thinking disabled, validation
  outside the backend, deadline and cancellation. The HTTP implementation emulates the
  embedded path rather than defining the semantics. An embedded llama.cpp smoke test
  runs before milestone 2 is called done, so the transfer is verified early.

### D6. No live streaming preview in v1; pre-transcribe closed VAD segments instead
Streaming partial text into the target app is unsafe (unvalidated output) and Whisper
streams badly. Instead, while the key is held, every VAD segment that closes is
transcribed immediately; on release only the tail is left. Tail latency becomes the cost
of one short segment. A live preview in the overlay is a possible later addition.

*Amended 2026-09-24:* the state machine cuts the segments from audio and VAD events the
driver streams to it, and sends them only for the oldest utterance in the pipeline, so
speech is still transcribed in utterance order. If nothing closed before release, the
whole recording goes to the engine in one call as before: a single call is no slower
than several, and it keeps the no-speech path unchanged. A pipeline flag turns
pre-transcription off for A/B measurement; the tail-latency gain is not yet measured.

### D7. Hotkey: our own low-level keyboard hook
A `WH_KEYBOARD_LL` hook on a dedicated thread with its own message loop. It is the only
mechanism that gives key-up, modifier-only keys, and the ability to swallow the key.
Rules the hook must obey (from the Windows contract and prior-art bugs):
- Do almost nothing in the callback: compare, `try_send` on a channel, return. No locks,
  allocation, logging or `GetAsyncKeyState`. Windows silently removes hooks that take
  longer than the low-level hook timeout, and never tells you.
- Ignore injected events (our own Ctrl+V would otherwise retrigger the hotkey).
- Swallow the key-up if the key-down was swallowed; drop auto-repeat.
- A watchdog detects a silently removed hook and reinstalls it. Measured 2026-09-24: a
  key the hook swallows **never appears** in `GetAsyncKeyState`, so the comparison runs
  the other way round. The callback bumps an atomic heartbeat; a timer thread (never
  the hook thread) treats the hotkey *showing up* in the physical state as proof that
  nothing swallowed it, watches for other keys changing while the heartbeat stays
  still, and probes with an unassigned key while the hotkey is held. On any of these:
  unhook on the owning thread, reinstall, and end the current recording as if the key
  was released. Recovery from a stalled callback takes about 1.5 s because only the
  hook thread may reinstall and it is the one stalled. Keystrokes between removal and
  reinstall are lost; that is inherent and accepted.
- If the foreground window is elevated and we are not, the hook sees nothing; detect
  and tell the user rather than fail silently. Confirmed by hand 2026-09-24: the
  hotkey simply does nothing in an admin window. Since no event arrives, the warning
  has to be proactive: the app watches the foreground window and shows "admin window,
  dictation unavailable" in the pill and tray while one has focus. The same blindness
  happens on the UAC secure desktop and the lock screen while a key is held, so a
  **maximum recording duration** (default 120 s) always ends a recording even if
  key-up never arrives.
- Shutdown order: stop producing events, unhook on the hook thread, then join it.

Trade-offs of Right Ctrl as the default, stated so they are not rediscovered: while
the key is swallowed the OS never sees Ctrl-down, so Ctrl chords cannot be typed while
talking; and some compact keyboards have no Right Ctrl. CapsLock is the documented
alternative (present everywhere, swallowing it just suppresses the toggle). The first
run offers the choice; there is no silently assumed universal key.

Double-tap for hands-free must not add latency to every hold: capture starts on
key-down, and if the press resolves into a tap pair the audio is discarded. No
250 ms wait before recording.

Rejected: `RegisterHotKey` (no key-up, no modifier-only), polling (latency, CPU),
`global-hotkey` crate (polls for release, cannot do modifier-only).

### D8. Insertion: delayed-render clipboard paste with a read signal, typing as fallback
Primary: put the text on the clipboard as **delayed-rendered** `CF_UNICODETEXT`, send
the paste chord, and watch for render requests. A render request means *something read
the clipboard*; it is not proof the target inserted the text. Clipboard managers,
history services, RDP layers and apps that snapshot the clipboard on activation all
produce render requests. So:

Measured 2026-09-24 (`spikes/clipboard-receipt/`): **Windows requests a delayed format
from the owner exactly once per write.** The first reader triggers the render; every
later reader gets the stored copy and the owner hears nothing. There is no "last
render" and no quiet window to wait for. The signal is therefore *who read first*, and
the policy follows from that:

- A short gap (20–50 ms) separates the clipboard write from the paste chord. Any read
  in that gap is a third party (measured: the RDP clipboard process reads within 1 ms
  of every write; the Win+V history service reads at about 200 ms unless the exclusion
  formats are set; the target apps themselves read nothing on focus).
- Outcome values: `TargetRead` (the render came after the chord and the reader's
  process, from the open-clipboard window, matches the target), `ThirdPartyRead` (a
  render arrived before the chord or from another process, so the target's own read is
  invisible), `NotRequested` (nothing read within the bound; the chord did not paste),
  `ClipboardChanged` (someone else wrote meanwhile). The strategy chain and the
  user-facing message are decided from that value.
- Restore policy: after `TargetRead`, restore once the render has been served plus a
  small margin; the app already holds the data. After `ThirdPartyRead`, restore on a
  fixed conservative delay (about 1 s) because the target's read cannot be observed;
  the transcript history (D16) covers the rare miss. Always check the clipboard
  sequence number first; a render does not change it, a foreign write does.
- The snapshot of the previous contents is bounded best-effort: all enumerable formats,
  GDI handle formats skipped, huge delayed-rendered data capped.
- Only `CF_UNICODETEXT` is offered. Windows synthesises `CF_TEXT`, `CF_OEMTEXT` and
  `CF_LOCALE` and still renders through us; no target asked for `CF_TEXT`.
- Our text is marked so clipboard history and cloud sync skip it. The RDP clipboard
  ignores the marks.

Remote desktop sessions are detected up front (`SM_REMOTESESSION`). Inside one, paste
works but the receipt is always third-party, and the dictated text is forwarded to the
client machine's clipboard by design of RDP. A "type only in remote sessions" option
keeps the text off the clipboard entirely for users who care. Amended 2026-09-24: RDP
is the maintainer's primary environment, so the third-party-read path and the
fixed-delay restore are the *normal* case for this project, not a fallback, and every
insertion change is tested there first.

Before pasting: refocus the window captured at hotkey-down and abort if the foreground
changed; release any held modifiers; refuse password fields; per-app paste chord
(Ctrl+V default, Ctrl+Shift+V or Shift+Insert for terminals and known editors).

Fallback: on `NotRequested`, type the text with `SendInput` Unicode events in one
batch, unless the app is on a never-type list. Never type after `TargetRead` or
`ThirdPartyRead`: that duplicates text in apps that read the clipboard but paste late.
Chords per app matter: conhost pastes on Ctrl+V and Shift+Insert but ignores
Ctrl+Shift+V, which correctly surfaces as `NotRequested`.

Typing rules, measured 2026-09-24 (`spikes/typing/`): the event construction that
`enigo` uses is correct, but Windows 11 Notepad turns every keystroke queued behind a
stall into the last injected character, and a plain Win32 edit control never does. So
the product sends **one UTF-16 unit per `SendInput` call, surrogate halves in separate
calls, 20–30 ms apart**, `\n` as a Return key, stops when the target loses the
foreground, and checks every call's return value. That is 33–50 characters per second;
a 600-character paragraph takes 12–18 s, which is why paste stays the primary path.
One call of more than 10,000 events silently loses half its characters (message queue
limit). Typing cannot confirm success, so the clipboard copy and the visible notice
still apply.

Last resort: leave the text on the clipboard and say so in the overlay. RDP, Citrix
and WSL windows are declared unsupported for insertion rather than silently falling
through to a typing path that types into the wrong session.

The strategy chain itself (choose chord, interpret outcome, decide fallback) is pure
logic in `core` behind an `Inserter` contract and is unit-tested with a fake clipboard;
only the Win32 calls live in the platform crate.

Rejected: UI Automation for insertion (replaces whole values, single-line only),
`WM_CHAR` posting (classic Win32 edits only), typing as the primary path (slow, mangled
by autocomplete and Enter-sends chat apps).

### D9. Context read from the target app is minimal and local
At hotkey-down we capture: foreground window handle, process executable name, window
title, whether the process is elevated, and via UI Automation on a separate COM thread
with a short timeout: whether the focused element is a password field and, later, its
selected text. Nothing else. No screenshots, no full text dumps.

### D10. Threading model
- **Hook thread**: keyboard hook and message loop only.
- **Audio thread**: owns the capture stream, downmixes and pushes into a lock-free ring
  buffer. The stream is opened on the first key-down and kept warm for a short idle
  window after the last utterance (default 30 s), then closed. A permanently open
  microphone would keep the Windows mic-in-use indicator lit and contradict the
  privacy pitch; a warm window keeps back-to-back dictations free of start-up loss.
  A small pre-roll buffer covers the first syllable while the stream is warm.
- **UI thread**: tray, overlay, and the clipboard-owner window that receives render
  requests; receives commands by posted messages.
- **COM/UIA worker**: multithreaded-apartment thread for UI Automation queries, with
  timeouts.
- **Pipeline (async runtime)**: the state machine, resampling, VAD, inference calls
  (each engine on its own blocking worker), normalization, and the insertion sequence.

Inference never runs on the UI or hook threads, and the hook thread never touches the
async runtime.

### D11. Model files live outside the binary and outside Roaming
Models are downloaded on first run from a manifest (URL, size, SHA-256, license) with
resume and hash verification, into `%LOCALAPPDATA%` so they never roam. Parakeet is
CC-BY-4.0 and the attribution is shown in the tray "about" entry. The binary ships with
no models embedded.

### D12. Crash isolation for GPU inference ships with the first self-contained build
Handy's worst Windows bugs are Vulkan device-lost crashes taking the app down. Running
inference in a child process is the fix (Voicetypr does it). The first build that end
users run without Ollama (milestone 3) is also the first with in-process GPU code on
unknown drivers, so a minimal child-process wrapper (pipe transport, kill-on-crash,
restart, request ids so stale replies are dropped) ships in that milestone, not later.
Milestones 0–2 run in-process. The contracts in D14 are shaped so the boundary is an
adapter, not a rewrite: owned request data, ids, deadlines, cancellation, and error
variants for a dead backend.

### D13. GPU policy is explicit, and milestone 0 is a decision gate
Engines are constructed under a policy: `RequireGpu`, `PreferGpu` or `CpuOnly`. Under
`RequireGpu` a backend that cannot load fails loudly; under `PreferGpu` the fallback is
reported in the overlay and logs. A GPU product never silently becomes a CPU product.

DirectML is the default, not the only path. The Vulkan SDK is installed on the
development machine from the start so milestone 0 can benchmark a ggml/Vulkan
alternative (`transcribe-cpp`) on the same GPU. Written fallback, decided before the
numbers exist: if DirectML fails to initialise on the development GPU, or Vulkan is
within 20 % of DirectML latency on the 10 s clip, the default becomes the Vulkan path
and ONNX Runtime is dropped from the speech side before milestone 1. One GPU runtime
for speech and LLM means one crash surface and one process to isolate.

### D14. Core contracts, completed before the state machine is written *(perishable)*
The engine traits alone cannot drive the pipeline. Before the state machine exists in
`core`, these are added there so it is hermetic and unit-testable:
- `Inserter`: focus context + text → outcome value (see D8), with the Win32 calls
  behind a small `Clipboard`/`Input` port implemented in the platform crate.
- `Recorder`: start/stop/pre-roll, returning 16 kHz mono PCM.
- `Notifier`: overlay state, sounds, toasts.
- Cancellation and deadline on every long call (`transcribe`, `normalize`, insertion
  wait). An `UtteranceId` on requests and results so stale replies are dropped.
- `Normalizer` takes `&mut self` like `SttEngine`; the resident LLM context is not
  shareable and a `Mutex` behind `&self` only hides that.
- Provenance carries validation diagnostics (which check, what score), not just a
  pass/reject label.
- Error variants for timeout, cancelled, backend crashed, malformed output.

### D15. One ONNX Runtime in the process
Speech, voice activity detection and the punctuation model all use ONNX Runtime. They
pin the same `ort` version workspace-wide; a second copy means a second native runtime
shipped. VAD therefore runs on our own `ort` session (the Silero graph is tiny and its
input contract is stable) rather than through a crate that pins a different `ort`.

*Amended 2026-09-24:* Silero VAD no longer needs ONNX Runtime at all. It runs on
`tract`, pure Rust, with a replacement for tract's `If` parser: stock tract checks both
branches of every `If` and rejects the dead, ill-typed branches in the Silero export.
Its probabilities matched onnxruntime to within 2e-6 on the bench fixtures, and it costs
about 35 µs per 32 ms chunk in a release build (method: `chunk_cost` test in the audio crate,
v6.2 model, this machine). It sits behind the audio crate's non-default `silero`
feature until the driver uses it.

### D17. Speech runs on ggml/Vulkan; ONNX Runtime is a fallback behind a feature *(perishable)*
Supersedes the default in D2, by the rule written in D13 before the numbers existed.
Measured 2026-09-24 on the RTX 5060 Ti (see §7): `transcribe-cpp` with Vulkan
transcribes the 10 s clip in about 75 ms against about 180 ms for ONNX Runtime with
DirectML, whose per-call fixed cost (about 140 ms even for 2 s of silence) puts it
behind the CPU for short utterances. Text was identical across paths. Vulkan was
stable over 100 consecutive runs. The same build runs Whisper large-v3-turbo, so the
second engine of D2 costs no extra runtime.

- Default engine: Parakeet TDT 0.6B v3 F16 GGUF via `transcribe-cpp` (pinned exactly)
  with the Vulkan backend. CPU-only machines use the same crate's CPU backend, to be
  measured before Tier 2 is declared.
- The ONNX Runtime engine stays in the tree behind a non-default cargo feature as an
  escape hatch for GPUs whose Vulkan drivers fail (Handy's device-lost reports), so
  D15's single-runtime rule holds for default builds.
- Warm-up uses a 30 s dummy clip: the first run of each new audio length compiles
  shaders (seconds), and the driver caches them on disk afterwards.
- Two GPUs on the development machine: the speech engine and the LLM must be pinned to
  a device explicitly. Sharing one GPU with a busy LLM tripled DirectML latency.
- The int8 Parakeet ONNX variant invented "Thank you." on pure silence; F16 and fp32
  returned empty. VAD dropping silent recordings (D3) is not optional.

### D16. Transcripts are kept in a short local history
The last N (default 10) utterances, with their raw and normalized text, are kept in
memory and on disk in the local data directory. "Paste last" and "copy last" work on
that ring, so an utterance whose insertion was aborted is still recoverable after the
next successful one. Nothing dictated is lost to a later dictation.

## 5. Workspace layout *(perishable)*

```
Cargo.toml                 workspace
crates/
  core/                    types, config, state machine, contracts (SttEngine, Normalizer, Inserter,
                           Recorder, Notifier — the last three arrive with D14), no platform code
  audio/                   capture, ring buffer, resampling, VAD
  stt/                     engines: parakeet (ort/DirectML); whisper behind a feature later
  normalize/               rules, prompt building, validation, OpenAI-HTTP backend; llama.cpp later
  platform-windows/        hook, focus context, clipboard/insert, overlay, tray, sound, paths
apps/
  whisper-local/           the binary: wiring, config loading, tray menu
tools/
  bench/                   CLI: wav → engine → text with timings; transcript → normalizer with timings
docs/
```

`core` compiles on every platform and has tests. `platform-windows` is `cfg(windows)`.

## 6. Milestones

- **M0 — measure and decide.** Workspace builds. `bench-stt` runs WAVs through Parakeet
  on DirectML and CPU; a Vulkan spike runs the same clips through `transcribe-cpp`;
  the D13 gate is applied. `bench-normalize` runs a fixture corpus (including
  adversarial cases) through the rule pass and the HTTP LLM pass and prints latency and
  validation verdicts. Two small Win32 spikes: a delayed-render clipboard owner
  against Notepad, Windows Terminal, VS Code and a clipboard manager, to learn which
  read signals actually arrive; and the hook watchdog, proving recovery from a stalled
  callback. Core contracts per D14 are completed. Numbers go into §7 with dates.
- **M1 — rules-only vertical slice.** Hold the key → record → Parakeet → rule pass →
  insert into Notepad, VS Code, a browser field and Windows Terminal. Tray icon,
  sounds, overlay pill, cancellation with Escape, maximum recording duration, single
  instance, transcript history. Config file with hotkey and dictionary. No LLM
  dependency. This is the first daily-usable build.
- **M2 — pipeline latency and normalization.** VAD segment pre-transcription measured
  against plain release-then-transcribe; hands-free mode. LLM pass over HTTP with
  validation, per-app styles, dictionary in the prompt. Embedded llama.cpp smoke test
  on this GPU proving the prompt contract and prefix caching transfer. Not done until
  that smoke test passes.
- **M3 — self-contained.** Embedded llama.cpp (Vulkan) as the default normalizer,
  model downloader with first-run flow, inference in a child process (D12),
  run-at-startup. The first build for people who are not us.
- **M4 — breadth and Tier 2.** Whisper engine (Vulkan) for the long-tail languages,
  CPU-only defaults with the punctuation model, per-app overrides UI, command mode on
  selected text, installer.

Separate latency budgets are tracked per stage and reported as p50/p95 for 3 s and 10 s
utterances on the development GPU. "Under one second" is judged on those percentiles,
never on throughput or real-time factor, and the LLM pass is skipped for an utterance
when its projected time would break the budget.

## 7. Measurements

### Speech-to-text, 2026-09-24
Method: release builds, 5 runs after warm-up, median in ms, RTX 5060 Ti 16 GB (one of
two; the other card was running an LLM), Ryzen 9 9900X with other sessions active
(CPU numbers are noisy). Clips are Windows-TTS 16 kHz mono plus whisper.cpp's public
domain `jfk.wav`. Fixtures and the exact commands are in `tools/bench-stt/fixtures/`
and `spikes/transcribe-cpp/README.md`.

| clip | audio | Vulkan Parakeet F16 | Vulkan Whisper turbo F16 | DirectML Parakeet fp32 | CPU Parakeet int8 (ORT) |
|---|---|---|---|---|---|
| tts 3 s | 3.4 s | 42 | 145 | 163 | 89 |
| tts 10 s, fillers | 9.5 s | 79 (67 on the idle card) | 195 | 181 | 206 |
| jfk, real speech | 11.0 s | – | – | 229 | 233 |
| tts 30 s, self-correction | 31.0 s | 222 | 401 | 297 | 811 |
| silence | 2.0 s | 24, empty text | 128, "you" | 144, empty | 61, **"Thank you."** |

Load / warm-up: Vulkan Parakeet 1.6 s / 59 ms (5.7 s the very first time on the
machine, shader compile); DirectML 2.3 s / 0.3 s; CPU int8 1.1 s / 0.05 s. Peak VRAM:
Parakeet F16 3.9 GB, Whisper turbo 4.2 GB. 100 runs of the 10 s clip on Vulkan: p50 74,
p95 79, max 87, no errors. Parakeet text was word-perfect and identical on Vulkan,
DirectML and CPU fp32; Whisper turbo appended a spurious "end." to the 30 s clip.

Final engine numbers after the switch (same method, Vulkan device pinned to the card
not running the LLM, ggml CPU backend at 12 threads = half the logical cores, which
measured fastest):

| clip | Vulkan Parakeet F16 | ggml CPU Parakeet F16 | Vulkan Whisper turbo F16 |
|---|---|---|---|
| tts 3 s | 39 | 112 | – |
| tts 10 s | 78 | 285 | 177 |
| jfk 11 s | 77 | 311 | – |
| tts 30 s | 199 | 940 | – |
| silence 2 s | 24, empty | 71, empty | – |

Tier 2 on this CPU is therefore about 0.03× real time with no silence hallucination;
laptop-class CPUs remain unmeasured. `transcribe-cpp` exposes segment/word/token
timestamps, a Whisper-only initial prompt, streaming and cancellation, and no VAD.

Still to measure: Vulkan Parakeet on real microphone audio; a laptop CPU.

### Audio capture, 2026-09-24
Method: `crates/audio/examples/record.rs`, release build, the only input device in this
RDP session ("Remote Audio", 44.1 kHz stereo f32). Cold stream open 23–26 ms over five
runs; warm start under 0.1 ms; resampling 60 s of audio costs 5–7 ms. The RDP-redirected
microphone delivered about 77 % of the nominal sample count with WASAPI discontinuity
flags about once a second, and its own timestamps agree with the short count, so the
loss is invisible from inside the stream. The recorder now compares received frames
against wall-clock time and reports the gap as dropped frames. A physical microphone
has not been measured; whether the loss is specific to RDP audio redirection is
unverified.

### Insertion end to end, 2026-09-24
Method: `crates/platform-windows/examples/platform-demo.rs` against Notepad with
simulated hotkey events, RDP session. With the RDP clipboard running: 5 of 5 pastes
landed, outcome `ThirdPartyRead` every time (rdpclip read 0.1–0.4 ms before the chord),
2.5 ms per insert. With it stopped: `TargetRead`, Notepad read 0.9–1.8 ms after the
chord, 35–43 ms per insert. The previous clipboard (6 formats) was restored
byte-identical, plus the three history-exclusion markers. Paced typing of a 37-unit
string with umlauts, an emoji and a newline landed intact in 918 ms.

### Whole pipeline, 2026-09-24
Method: `whisper-local simulate tools/bench-stt/fixtures/tts-10s-fillers.wav`, release
build, Parakeet F16 on Vulkan device 1 (the card without the LLM), three runs per
configuration, times in ms from key release; RDP session, so every insert is
`ThirdPartyRead`, and Notepad's text was read back after each run.

| normalizer | speech | cleanup | paste | total |
|---|---|---|---|---|
| rules only | 71 / 72 / 179 | 0.1 | 2 | 74 / 75 / 181 |
| Qwen3-4B over Ollama | 69 / 179 / 179 | 220 / 222 / 227 | 2 | 292 / 402 / 407 |

The pipeline outside the engine calls adds under 0.3 ms. Two things are unexplained:
the speech time alternates between about 72 and about 180 ms when runs are 2 s apart
but not when back to back (GPU clock ramp is the obvious suspect, unverified), and the
LLM round trip is about 220 ms here against 100 ms p50 in the isolated benchmark
(the benchmark reused one HTTP agent; the app's first-request cost or the different
prompt size are candidates). Both are inside the budget and both get measured before
milestone 2 is called done.

### First human run, 2026-09-24
Method: the maintainer at the keyboard, over RDP (the primary way this machine is
used), release build. Passed: build and `doctor`; hold-to-talk into Notepad with every
word typed; Escape cancel; double-tap hands-free; max-duration stop; insertion into
VS Code, a browser field, Windows Terminal and cmd; tray icon and "copy last".
Findings: the overlay level bar barely moves at normal speaking volume although
transcription is complete; the hotkey does nothing while an admin window has focus,
which is the UIPI rule (a medium-integrity hook receives no keystrokes then), so the
only possible behaviour is a warning before the user speaks; the tray icon should show
state. Not yet tested: clipboard restore with an image or files, and the LLM pass.

### Clipboard read signals, 2026-09-24
Method: `spikes/clipboard-receipt/`, delayed-rendered `CF_UNICODETEXT` pasted by
synthetic chord into Notepad, conhost, Windows Terminal and PowerShell ISE, inside an
RDP session, then repeated with the RDP clipboard process stopped. One render per
write in every run. With the RDP clipboard running it read the entry about 305 ms
before the chord every time; with it stopped, the target app read 0.5–2.3 ms after the
chord and nothing read before it. The Win+V history service read at 208 ms only when
the exclusion formats were absent. Text landed in Notepad in every mode. Not tested:
a non-RDP session, third-party clipboard managers, Electron/UWP/elevated targets.

### Normalization, 2026-09-24
Method: `bench-normalize`, 30 fixture cases (English and German, questions, imperatives,
prompt injections, self-corrections, code-editor style, very short), 3 runs each,
Ollama 0.34 native chat API with the model kept resident, RTX 5060 Ti shared with the
speech benchmarks, release build. Latency is the full HTTP round trip.

| model | validated | rejected | forbidden text in raw LLM output | forbidden text inserted | matches expected | p50 ms | p95 ms |
|---|---|---|---|---|---|---|---|
| rules only | – | – | – | 1 (multi-word dictionary near-miss) | 23/28 | 0.01 | 0.04 |
| Qwen3-4B-Instruct-2507 Q4_K_M | 29 | 1 | 0 | 0 | 27/28 | 100 | 162 |
| Granite 4.0 Micro | 24 | 6 | 3 | 0 | 21/28 | 119 | 324 |
| Qwen3-1.7B, thinking off | 26 | 4 | 2 | 0 | 25/28 | 50 | 93 |

Qwen3-4B never answered a question or followed a command; its one rejection was
turning a dictated "dash m" into "-m" in code style. Granite answered a translation
request and both smaller runs obeyed a "you are now a pirate" injection; validation
caught every one before insertion. Two failures passed the originally specified checks
and forced validator additions: Qwen3-1.7B changed "three, no wait, four thirty" into
"three forty five" (containment 0.80, now caught by a changed-number check), and two
models capitalised shell commands in code style (now caught by a verbatim check for
code style). Cold model load is 7–11 s; the first connection to `localhost` cost 2 s
until rewritten to the IPv4 loopback.

Implication for the budget: speech ≈ 80 ms plus LLM ≈ 100 ms leaves most of the
one-second target unspent on this GPU, even before the embedded backend and prefix
caching. The 4B model is the default candidate; the 1.7B halves latency but edits
numbers.

Pending: the embedded llama.cpp smoke test (milestone 2), latency under GPU contention
with the speech engine on the same card, prompt-cache hit rate.

## 8. Build requirements on the development machine *(perishable, 2026-09-24)*

Present: Rust 1.97 stable MSVC, Visual Studio 2022 Community with MSVC 14.44, CMake 4.4
and LLVM 23 (both installed today), Vulkan SDK (installing today, for the D13 spike),
Ollama 0.34, two RTX 5060 Ti 16 GB, Ryzen 9 9900X, 93 GB RAM. Absent: CUDA toolkit
(not needed under D2/D5), Node.js (not needed).

The design was reviewed by an outside three-model panel on 2026-09-24
(`docs/research/2026-09-24-design-review-panel.md`); D4, D5, D7, D8, D10 and D12 were
amended and D13–D16 added in response.
