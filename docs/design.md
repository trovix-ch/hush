# hush: design and decisions

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
  start with Windows (per-user Run key, no admin), retry download, about, quit. Its
  colour shows state: gray idle or paused, green listening, amber processing, red on
  error. No main window and no console window; configuration is a TOML file opened in
  the editor, and `doctor`/`simulate` attach to the terminal they were started from.
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

*Measured 2026-09-24 (`spikes/llama-cpp/`):* the embedded path passes the gate. With
Ollama's exact model file it validates 29 of 30 fixtures, the same as HTTP, at about
105 ms p50 with the system prefix cached and 250 ms uncached; the grammar costs
microseconds and blocks only the literal preamble words. Latency did not improve over
HTTP because Ollama already caches the prefix on the same GPU, so the embedded path is
about shipping without a server, not speed. A different Q4_K_M build of the same model
validated one case fewer: the bundled model is pinned by hash and the fixtures re-run
on that exact file. The very first Vulkan run on a fresh driver cache took 21 s, so
start-up warms the LLM the same way it warms the speech engine.

*Amended 2026-09-24 (milestone 3):* the embedded backend is the default normalizer. The
bundled GGUF is Ollama's exact file, fetched from the Ollama registry by its sha256: none
of the 256 Hugging Face GGUF repositories for the model that were checked carries it.
On the current prompt it validates 28 of 30 fixtures, and so does the HTTP path; both
gave 29 before the per-app `<app>` wording changed the prompt (§7). llama.cpp and
transcribe.cpp each vendor a static ggml (0.24 and 0.20), whose symbols collide at link
time, so the default build links transcribe.cpp as a DLL. The process therefore holds two
ggml copies and two Vulkan instances: D13's single GPU runtime is not met until both
build against one ggml. *(2026-09-24: superseded by D12's worker process, which keeps
the two copies in separate processes and needs no DLL.)*

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
pre-transcription off for A/B measurement.

*Amended 2026-09-24:* measured in the driver with Silero (§7, "Pre-transcription"),
release-to-transcript p50 fell from 309 to 34 ms on a 30 s dictation and from 254 to
50 ms on `jfk.wav`, at the cost of segment boundaries turning into sentence breaks in the
transcript and, on the 30 s clip, up to three times the total GPU time per utterance.
Decision: **off by default until the stitched text is as good as the single call.** The
sentence-break damage is user-visible and outranks 250 ms of waiting. The fix to try
first: close segments only on long pauses (about 700 ms, so boundaries fall on real
sentence ends), and stitch with the engine's word timestamps so a segment's trailing
period and the next segment's capital are dropped when the gap between them is short.
The flag stays for measurement and for users who prefer speed.

*Amended 2026-09-24:* the fix was built and measured (§7, "Pre-transcription after the
fixes"), and **the flag stays off**. What changed: segments close only after 700 ms of
silence; a pause counts only once 100 ms more audio has arrived than `min_pause` asks
for, which covers the detector's late speech-start report and makes the cuts the same
on every run; the engine times words, and segments are joined on the real silence
between the last word of one and the first word of the next. Under 400 ms, a trailing
`.` or `!` is removed (a `.` stays before "I" or a vocabulary word), a comma is added
from 300 ms, and the next word is lowercased; a segment the engine left without a
sentence end always has its successor lowercased. On the 30 s dictation the stitched
text is now identical to the single call. On `jfk` it is not: "Americans. Ask not!
What" against "Americans, ask not what". Those two pauses are 1024 and 832 ms of real
silence, longer than the 560–640 ms before the dictation's true sentence ends. Pause
length does not say whether a boundary is a comma or a full stop, so no gap threshold
fixes both clips. The planned 600 ms threshold turned two of the dictation's sentence
ends into commas, which is why the shipped threshold is 400 ms. That threshold now
joins only boundaries cut inside speech. The next thing to try is giving each segment
the previous segment's last second or so of audio as context, so the engine decides
the punctuation at the boundary with the words on both sides.

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
  reinstall are lost; that is inherent and accepted. Amended 2026-09-24: measured here, a
  callback that overruns the timeout (about 300 ms per event) loses only that event and
  the hook stays installed even through a 6 s stall, so under load a disagreement is
  only a suspicion and the watchdog reinstalls only after three tagged probes about
  250 ms apart all go unanswered, which rides out an 800 ms stall and still recovers a
  silently removed hook in about 0.9 s. The remaining edge: a release dropped during
  such a lag is invisible (a swallowed key never shows in the physical key state), so
  the recording runs on until the next press, Escape or the duration cap.
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
from the owner for the first reader only.** Every later reader gets the stored copy and
the owner hears nothing. There is no "last render" and no quiet window to wait for. The
signal is therefore *who read first*, and the policy follows from that. Corrected later
the same day on the console session: readers that race the first one (a remote-access
tool reading every write within 50 ms) each trigger a render, and each render bumps the
clipboard sequence number; the owner must count its own renders as non-writes or it
mistakes them for a foreign write and never restores. The spike could not see this
under RDP because nothing there read that eagerly.

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

*Amended 2026-09-24 (milestone 3):* the app provisions itself. On launch with models
missing it starts anyway, downloads them one at a time with speech first while a
persistent overlay pill shows progress, brings each engine up as its model lands, and
is usable rules-only as soon as speech is ready. A failed download leaves an alert pill
up until acted on and a "Retry download" tray entry; a transient notice may pass over it
but never clears it. `doctor` remains the diagnostic, not a prerequisite.

### D12. Crash isolation for GPU inference ships with the first self-contained build
Handy's worst Windows bugs are Vulkan device-lost crashes taking the app down. Running
inference in a child process is the fix (Voicetypr does it). The first build that end
users run without Ollama (milestone 3) is also the first with in-process GPU code on
unknown drivers, so a minimal child-process wrapper (pipe transport, kill-on-crash,
restart, request ids so stale replies are dropped) ships in that milestone, not later.
Milestones 0–2 run in-process. The contracts in D14 are shaped so the boundary is an
adapter, not a rewrite: owned request data, ids, deadlines, cancellation, and error
variants for a dead backend.

*Amended 2026-09-24 (milestone 3):* built. Speech runs in `hush-stt-worker.exe` next to
`hush.exe`, behind the same `SttEngine` trait. The pipe carries length-prefixed frames,
a JSON header plus raw little-endian `f32` audio (a 30 s utterance is 1.9 MB, never
base64), with request ids, cancel messages and a heartbeat every second. A worker that
exits, closes the pipe, writes a malformed frame or misses heartbeats for 5 s fails the
request in flight with `BackendDied` and is restarted in the background; the next
request waits for the replacement. One that has not stopped 2 s after a cancel or a
passed deadline is killed and restarted, since a hang inside the driver keeps the
heartbeat thread alive. Workers sit in a kill-on-close job object, so none outlives a
crashed hush. The worker points its own stdout at stderr and keeps a private copy for
the protocol: one stray `printf` from native code would otherwise corrupt the framing.
Its log reaches hush's at debug level, and the last line goes into the `BackendDied`
message. `engine.in_process = true` keeps the old path for debugging, in builds with the
`in-process-stt` feature.

This also ends the DLL split in D5: transcribe.cpp links statically into the worker and
llama.cpp statically into hush, and neither exe needs anything beside it except the VC++
runtime and the Vulkan loader (checked with `dumpbin /dependents` and a run from a
directory holding only the two exes). The engine lives in the worker's package rather
than behind a feature of `hush-stt`: Cargo unifies features across everything built in
one invocation, and a workspace build then linked both static ggml copies into hush's
test binary (LNK2005). D13's one-runtime goal is still unmet, but the two ggml copies
no longer share a process. Cost: under 1.3 ms per call, measured in §7, and 20–150 ms of
process start at load. Crash test: the worker aborted 30 ms into a 30 s
transcription, the call returned `BackendDied`, a replacement was ready 1.9 s later and
served the next call, on the CPU and on Vulkan (`hush-stt-worker` integration tests,
gated on `HUSH_TEST_TC_MODEL`).

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

*Amended 2026-09-24:* the device is named by PCI bus id, never by Vulkan index. The
index order differed between the console session (integrated GPU first) and an RDP
session (05:00, integrated, 01:00), so `gpu_device = 1` put speech on the integrated
GPU over RDP at 636 ms for the 10 s clip (§7, "Speech after idle"). `engine.device` and
`normalizer.device` take `auto` (the default), a PCI id, or part of a device name; the
old index is read for one release with a warning. `auto` drops integrated, CPU and
virtual devices when a discrete GPU exists and takes the one with the most free memory
(the driver's memory budget, which counts other processes' allocations), so a card
another model already fills loses; free memory within 1 GiB is a tie broken by PCI id,
so two idle cards do not swap between sessions. The app resolves once per process on
the speech worker's device list and sends the PCI id in the worker's load request;
the language model follows speech unless configured otherwise, and gets the same
ordered list. Under `require-gpu` a card that fails to load is an error; under
`prefer-gpu` the next candidate is tried, then the CPU, and each failure is logged.
Rejected: a second Vulkan loader (`ash`) for enumeration, since both ggml copies
already report type, PCI id and budgeted free memory.

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
v6.2 model, this machine). Since 2026-09-24 it is built by default and the driver uses
it whenever its model file has been downloaded, falling back to the energy detector
otherwise.

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
- *Amended 2026-09-24:* the 75 ms holds only on an awake card; after the idle gap that
  real dictation always has it was 237 ms, so the driver runs a 1 s dummy pass every
  250 ms while the key is held (`engine.gpu_nudge`, on by default, ignored on the CPU),
  which brings release-time speech back to 75 ms for about 5 W during the hold (§7,
  "Speech after idle").
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
  core/                    contracts (SttEngine, Normalizer, Inserter, Recorder, Notifier), the
                           pipeline state machine, segmenter and stitcher, config, history,
                           cancellation; no platform or inference dependencies
  audio/                   capture, resampling, energy and Silero VAD
  stt/                     model manifest and downloader, the worker protocol and the remote
                           engine client; the ONNX engine behind a non-default feature
  normalize/               rules, prompt, validation, the shared LLM path, embedded llama.cpp
                           and OpenAI-HTTP backends
  platform-windows/        hook, focus context, clipboard/insert, input, overlay, tray, sound
apps/
  hush/                    the app: driver, workers, doctor, simulate
  hush-stt-worker/         child process that links transcribe.cpp and runs the speech engine
tools/
  bench-stt/, bench-normalize/   measurement harnesses and fixtures
spikes/                    standalone experiments with their own workspaces and dated READMEs
docs/
```

Updated 2026-09-24 after milestone 3. `core` compiles on every platform and carries
most tests. `platform-windows` is `cfg(windows)`.

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
  run-at-startup. The first build for people who are not us. *Reached 2026-09-24:*
  a release zip holds `hush.exe`, `hush-stt-worker.exe`, licences and attributions;
  both depend only on system DLLs, the VC++ runtime and the Vulkan loader.
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

### Pre-transcription, 2026-09-24
Method: `hush simulate <wav> --runs 5` (3 for the 10 s and silence clips), release
build, rules-only normalizer, Parakeet F16 on Vulkan device 1, Silero v6.2 VAD in the
driver, default segmenter (300 / 400 / 200 ms, 20 s cap), RDP session. The WAV is
streamed at real-time pace and the key is released when the clip ends. Latency is key-up
to the transcript reaching the normalizer; "engine" is the summed inference of every
call for the utterance.

| clip | pre-transcribe | release→transcript p50 / p95 | total p50 / p95 | segments closed in hold | tail | engine |
|---|---|---|---|---|---|---|
| tts 30 s, cut 0.2 s after the last word | off | 309 / 470 | 312 / 475 | – | 30.4 s | 204–466 |
| same | on | 34 / 39 | 37 / 42 | 10–12 | 1.5 s | 489–593 |
| tts 30 s as recorded (0.8 s trailing silence) | off | 420 / 505 | 422 / 507 | – | 31.0 s | 413–504 |
| same | on | 0.5 / 0.5 | 3.0 / 3.7 | 13–14 | none | 1143–1353 |
| jfk 11 s (0.02 s trailing) | off | 254 / 340 | 257 / 342 | – | 11.0 s | 246–340 |
| same | on | 50 / 54 | 52 / 58 | 3–4 | 0 or 3.0 s | 141–184 |

A first `jfk` batch with pre-transcription on gave p50 145 / p95 192 ms from the same
segmentation, so the run-to-run spread is wide; GPU clocks dropping between calls cost
up to 4.5 times (see "Speech after idle" below), not re-checked against this spread.
The 10 s clip closes as one 9.2 s segment during its
trailing silence, and its text is identical with the flag on and off; the silence clip
inserts nothing either way.

Findings. Segment boundaries become sentence breaks: `jfk` turns from "Americans, ask
not what your country can do for you, ask what…" into "Americans. Ask not! What your
country can do for you. Ask what…", and the 30 s clip gains "Also Please…" and
"Tuesday. No wait. Wednesday." (the rule pass still resolves the self-correction). The
same audio does not always segment the same way (10, 12 or 14 segments on the 30 s
clip); the likely cause is that a pause is judged against the audio received so far
while Silero confirms a start about 96 ms late, so a pause within that margin of
`min_pause` closes or not depending on 50 ms delivery timing. Summed engine time grew
up to threefold on the 30 s clip (13–14 calls) yet fell on `jfk` (4 calls); why short
calls cost that much is not established.
Pending: a real microphone; the LLM normalizer over pre-transcribed segments.

### Pre-transcription after the fixes, 2026-09-24
Method: as above, 5 runs per row, both clips as recorded (0.8 s and 0.02 s trailing
silence), RDP session. "Diffs" compares the stitched transcript with the single call
token by token (words and punctuation marks). Gaps are the word-to-word silences at
each boundary, taken from the pipeline's debug log.

| clip | setting | segments (hold+tail) | tail s | release→transcript p50 / p95 | engine ms | diffs vs single call |
|---|---|---|---|---|---|---|
| jfk | off | 0+1 | 11.0 | 250 / 360 | 209–360 | – |
| jfk | min_pause 400 | 4+0, once 3+1 | 0 or 3.0 | 45 / 207 | 283–445 | 3 |
| jfk | 700 | 2+1 | 5.8 | 194 / 194 | 361–397 | 2 |
| jfk | 1000 | 2+1 | 5.8 | 198 / 211 | 394–447 | 2 |
| jfk | 700, word stitch, join 400 ms, latency margin | 2+1 | 5.8 | 197 / 237 | 360–433 | 2 (gaps 1024, 832 ms) |
| 30 s | off | 0+1 | 31.0 | 407 / 538 | 399–538 | – |
| 30 s | min_pause 400 | 13+0, once 14+0 | 0 | 0.6 / 0.9 | 1351–1472 | 7, once 8 |
| 30 s | 700 | 6+0 | 0 | 97 / 202 | 873–1098 | 0 |
| 30 s | 1000 | 2+1 | 8.9 | 225 / 268 | 631–667 | 1 |
| 30 s | 700, word stitch, join 600 ms | 6+0 | 0 | 112 / 198 | 643–1108 | 2 (sentence ends at 560, 592 ms made commas) |
| 30 s | 700, word stitch, join 400 ms, latency margin | 5+1 | 3.3 | 36 / 108 | 993–1103 | 0 |
| 30 s | 400, latency margin, no joining | 7+0 | 0 | 0.7 / 0.9 | 817–1079 | 1 |

With the 100 ms latency margin every configuration segmented identically on all five
runs; without it, 400 ms gave 13 or 14 segments. The margin moves one boundary in the
30 s clip from the hold into the tail, which costs about 3 s of tail audio. Even with
5–7 calls, the summed engine time is still about 2.5× the single call on the 30 s clip.
Why short calls cost that much is still not established.

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
Method: `hush simulate tools/bench-stt/fixtures/tts-10s-fillers.wav`, release
build, Parakeet F16 on Vulkan device 1 (the card without the LLM), three runs per
configuration, times in ms from key release; RDP session, so every insert is
`ThirdPartyRead`, and Notepad's text was read back after each run.

| normalizer | speech | cleanup | paste | total |
|---|---|---|---|---|
| rules only | 71 / 72 / 179 | 0.1 | 2 | 74 / 75 / 181 |
| Qwen3-4B over Ollama | 69 / 179 / 179 | 220 / 222 / 227 | 2 | 292 / 402 / 407 |

The pipeline outside the engine calls adds under 0.3 ms. Two things are unexplained:
the speech time alternates between about 72 and about 180 ms when runs are 2 s apart
but not when back to back (the GPU dropping its clocks; confirmed and mitigated, see
"Speech after idle" below), and the
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

Embedded llama.cpp (Vulkan, `llama-cpp-2`, same fixtures, 3 runs, card not running
Ollama), 2026-09-24:

| model | load | VRAM | prefix cached p50/p95 | uncached p50/p95 | validated |
|---|---|---|---|---|---|
| Qwen3-4B-Instruct-2507 Q4_K_M, Ollama's file | 2.8 s | 3.3 GB | 106 / 174 | 245 / 324 | 29/30 |
| same model, unsloth Q4_K_M build | 2.2 s | 3.3 GB | 109 / 184 | 251 / 436 | 28/30 |
| Granite 4.0 Micro Q4_K_M | 1.8 s | 2.6 GB | 100 / 181 | 213 / 298 | 24/30 |

Per request with the prefix cached: 46 prompt tokens in 29 ms, 10 output tokens in
89 ms (about 115 tokens/s), so generation is 80 % of the cost.

### Embedded normalizer in the app, 2026-09-24
Method: release build, the manifest's Qwen3-4B file (Ollama's, sha256 `85e4a5b7…`),
Vulkan, RDP session. `bench-normalize --normalizer llama-cpp --pci 05:00 --runs 3` and,
for the same prompt revision, `bench-normalize --runs 3` against Ollama 0.34 on the other
card; latency is the call from rule-pass text to verdict.

| normalizer | validated | forbidden text raw / inserted | matches expected | p50 ms | p95 ms |
|---|---|---|---|---|---|
| llama-cpp, embedded | 28/30 | 0 / 0 | 25/28 | 105 | 174 |
| Ollama over HTTP | 28/30 | 0 / 0 | 24/28 | 101 | 166 |

Both reject the same two code-style cases ("dash m" to "-m", "dash dash workspace" to
"-- workspace"), which the app never sends to the LLM. Embedded, a request prefills 56
tokens in 30 ms and decodes 10 in 83 ms (119 tokens/s). Three of 30 outputs differ
between the backends by a word or a comma.

`hush simulate tools/bench-stt/fixtures/tts-10s-fillers.wav --runs 10`, into Notepad,
speech on Vulkan device 1 (PCI 05:00), p50 / p95 ms from key release:

| language model on | speech | cleanup | paste | total |
|---|---|---|---|---|
| the same card as speech | 238 / 267 | 243 / 246 | 1.4 / 1.6 | 480 / 513 |
| the other card (PCI 01:00, Ollama idle on it) | 226 / 294 | 244 / 336 | 1.4 / 1.7 | 480 / 564 |
| rules only (5 runs) | 274 / 314 | 0.1 | 1.5 | 276 / 315 |

Sharing the card costs nothing measurable here, because the pipeline never runs speech
and the LLM at the same time; overlapping utterances were not measured. A first 5-run
pass had the split configuration slower (cleanup 395 against 280 ms p50) and did not
reproduce, so single runs of this benchmark are noise. Speech takes about 230–280 ms in
every configuration including rules-only, against 75 ms back to back in `bench-stt`
(measured again today). The rules-only row shows it is not the LLM; the 11 s between
runs let the card fall to its idle clocks (confirmed later the same day, see "Speech
after idle" below). The Ninja-built and the Visual-Studio-built speech libraries measured the
same in `bench-stt` (72–79 ms medians, interleaved). Cleanup is 240 ms rather
than 105 because this utterance produces about 25 tokens.

### Speech worker process, 2026-09-24
Method: release build, `bench-stt <clip> --engine transcribe-cpp|remote --device 1
--runs 10`, back to back after warm-up, Vulkan on the RTX 5060 Ti at PCI 05:00, console
session; another benchmark was running on the AMD integrated GPU at the time. The pipe
overhead is the wall-clock `transcribe` call minus the inference time the worker
measures, so it covers copying the audio, both pipe transfers and the JSON.

| clip | in process, p50 ms | worker, p50 ms | pipe overhead p50 / max ms |
|---|---|---|---|
| tts 3 s | 35.7 | 34.4 | 0.26 / 0.36 |
| tts 10 s | 72.0 | 71.9 | 0.29 / 0.43 |
| tts 30 s (1.9 MB of audio) | 193.3 | 190.6 | 1.18 / 1.27 |

Text was identical both ways. Model load inside the worker matched in process (1.56–1.58
s against 1.54 s); starting the process added 15–150 ms. `hush simulate
tools/bench-stt/fixtures/tts-10s-fillers.wav --runs 3 --normalizer llama-cpp`, run from a
directory holding only `hush.exe` and `hush-stt-worker.exe` with a system-only PATH:
release to transcript p50 84.5 ms, to inserted text 368 ms, the language model on the
same card as the worker.

### Speech after idle: GPU clocks, 2026-09-24
Method: release build, the 10 s clip, Parakeet F16 on the RTX 5060 Ti at PCI 05:00,
console session, 8 runs per row. `bench-stt <clip> --pci 05:00 --runs 8 --gap-ms N`
idles N ms before each run; `--kick` calls the engine's nudge (one 1 s silent pass),
repeats it every `--kick-every-ms`, and transcribes `--kick-lead-ms` after the first,
standing in for key-down, speaking and release. The card's state is `nvidia-smi
--query-gpu=clocks.sm,clocks.mem,pstate,power.draw -lms 100`, sampled just before each
call; it lags by up to 100 ms.

| idle before the call | p50 ms | runs (ms) | state before the call (of 8) |
|---|---|---|---|
| none, back to back | 76 | 74–78 | P0, memory 14001 MHz: 8 |
| 2 s | 187 | 77–196 | P0/P3: 5, P5: 3 (two with memory at 810 MHz) |
| 5 s | 204 | 75–319 | P8 (memory 405 MHz): 4, P5: 2, P3: 2 |
| 11 s | 237 | 225–335 | P8: 7, P5: 1 |

| after 11 s idle | 1 s before release | 10 s before release |
|---|---|---|
| no nudge | 237 (the row above) | – |
| one nudge at key-down | 76, three runs 183–185 | 237, all P8 again |
| nudge every 1000 ms | same as one | 189 |
| nudge every 500 ms | 73, max 77 | 77, max 79 |
| nudge every 250 ms | 72, max 75 | 76, max 79 |

So the gap was the card's power state: memory clocks drop to 810 MHz within about 2 s
and to 405 MHz (P8) by 11 s, and the next call runs 2.5–4.5 times slower. One nudge
wears off before a 10 s utterance ends. A cold nudge takes 100–320 ms (median about
150), a warm one 22 ms (`hush doctor`), all while the user is still speaking. Power on the card: 6.1 W at P8,
11–12 W while nudging, so about 5–6 W extra and only while the key is held. Keeping the
card awake for the 30 s warm-microphone window instead would cost the same 5–6 W per
use, over the 1 W allowance, and still leave the first utterance after it cold, so it
was rejected.

`hush simulate <clip> --runs 8`, rules only, the same card, speech in the worker
process, console session; without nudges the card idles for the hold plus about 2 s
between runs. p50 / p95 ms from key release to transcript:

| clip | `engine.gpu_nudge = false` | nudge every 500 ms | nudge every 250 ms (shipped) |
|---|---|---|---|
| tts 10 s | 277 / 293 | 75 / 167 (16 runs) | 75 / 114 (16 runs, one over 100) |
| tts 3 s | 113 / 219 | 37 / 97 | 37 / 39 |

At 500 ms the card still fell to P5 for about a quarter of the hold and three of 16
releases landed there; 250 ms halved that for 0.5 W. A nudge still running at release
delays the transcription by up to one warm pass (seen once, 24 ms). The Vulkan device
order also moved during the day: in the RDP session it was the 05:00 card, the
integrated GPU, the 01:00 card; in the console session the integrated GPU came first and
05:00 was index 1 (two observations each way, cause not established). An index in
`engine.gpu_device` therefore names a different card depending on how the user is
logged in, and the integrated GPU takes 636 ms for the 10 s clip.

## 8. Build requirements on the development machine *(perishable, 2026-09-24)*

Present: Rust 1.97 stable MSVC, Visual Studio 2022 Community with MSVC 14.44, CMake 4.4
and LLVM 23 (both installed today), Vulkan SDK (installing today, for the D13 spike),
Ollama 0.34, two RTX 5060 Ti 16 GB, Ryzen 9 9900X, 93 GB RAM. Absent: CUDA toolkit
(not needed under D2/D5), Node.js (not needed). Since the embedded LLM (milestone 3):
Ninja 1.13 (`uv tool install ninja`) and LLVM's libclang are required, and the machine
also has an AMD integrated GPU, which Vulkan lists first in a console session and second
in an RDP session (see "Speech after idle" in §7).

The design was reviewed by an outside three-model panel on 2026-09-24
(`docs/research/2026-09-24-design-review-panel.md`); D4, D5, D7, D8, D10 and D12 were
amended and D13–D16 added in response.
