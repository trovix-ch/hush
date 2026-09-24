# hush

hush is a Windows dictation tool: hold a key anywhere in the OS, speak, let go,
and cleaned-up text appears where the cursor is. Speech recognition, the language-model
cleanup and the insertion all run on the user's machine. It is the local answer to
Wispr Flow, written in Rust, with no web view and no cloud.

The product decisions, numbered and dated, live in `docs/design.md`. Read the sections
that touch what you are changing before you change it. The code is the ground truth for
how things work; the design doc is the ground truth for why.

## What makes hush special

1. **Nothing leaves the machine.** No cloud, no accounts, no telemetry. The only network
   traffic is a one-time model download from a pinned manifest and the user's own local
   inference server. Screenshots and full-screen text as "context" are out, permanently.
2. **Latency is the product.** The target is under one second from key release to text
   on screen, judged on p50/p95 for 3 s and 10 s utterances, never on throughput. Every
   stage reports its own timing. A change that adds latency on that path needs a
   measured justification, and a measurement without its method and date is a rumour.
3. **It never loses your words.** Every insertion ends in a confirmed outcome or a
   visible error, and the text is in the history ring before insertion is attempted. If
   the target cannot take it, the clipboard has it and the overlay says so.
4. **It cleans, never answers.** Dictate a question and you get the question, punctuated.
   The LLM is a cleaner behind an order-aware validator, and every LLM failure falls
   back to the rule pass. Nothing dictated may be answered, executed or summarised.
5. **It is small and mostly invisible.** A tray icon, an overlay pill that never takes
   focus, a config file. Idle cost is a microphone that closes after a warm window and
   a resident model on the GPU.

## How to think here

Prefer the simplest thing that meets the constraint. Do not preserve complexity because
it already exists; delete what a change makes unnecessary. When two designs seem equal,
the one with fewer moving parts wins, and the one that can be unit-tested without
hardware wins over the one that cannot.

The code is the only source that cannot rot. When something is derivable from the code,
it is not written down anywhere else. When a choice cannot be seen from the code, the
reason lives next to the code in one or two sentences, or in a numbered decision in the
design doc when it shapes more than one crate.

Wrong prose is worse than no prose: it retires a worry nobody is carrying, and it reads
exactly like a checked claim. If you find a comment or a decision that the code
contradicts, the code is right and the prose gets fixed or deleted in the same change.

## A small glossary

- **Utterance**: one press-to-release of the hotkey and everything derived from it. It
  carries an id so stale results are dropped.
- **Hold / hands-free**: hold-to-talk is the default; a double tap locks recording on
  until the next tap. Capture starts on key-down either way.
- **Engine**: a speech-to-text implementation behind the `SttEngine` contract. Reports
  the backend and device it actually loaded on; never assumed.
- **Normalizer**: turns a raw transcript into insertable text. The **rule pass** is
  deterministic and always runs; the **LLM pass** is optional; **validation** gates the
  LLM output against the rule-pass text.
- **Inserter / ports**: the strategy chain that gets text into the target lives in
  `core` and talks to the OS through small port traits (clipboard, input, focus) that
  the platform crate implements and the tests fake.
- **Pipeline / driver / effects**: the state machine in `core` takes events and returns
  effects; it owns no threads. The binary's driver executes effects on the right
  threads. Anything the pipeline decides is testable without Windows.
- **Target / focus context**: the window and process that had focus at key-down, with
  elevation, remote-session and password-field flags.
- **Tier 1 / Tier 2**: discrete GPU with the full pipeline on; CPU-only laptop with
  the same pipeline and cheaper defaults. Nothing may assume Tier 1.
- **Spike**: a standalone throwaway project under `spikes/` that answered one question
  with a number. Its README is the record; its code is not a dependency.
- **Bench**: `tools/bench-*`, the harnesses that measured every engine choice.

## The ways to hurt yourself

1. **Doing work in the keyboard hook callback.** Past its timeout Windows gives up on
   the callback: measured here, it passes the key through and never runs the callback
   for that event, so the hotkey leaks to the target app and a release can be lost;
   the documentation says it may also remove the hook silently. The callback compares,
   sends on a channel, returns. No locks, no allocation, no logging, no key-state
   queries. The watchdog exists for the removal case and must never fire on mere lag.
2. **Trusting a clipboard render as proof of insertion, or restoring on a timer.**
   Windows renders a delayed format for the first reader and hands later readers the
   copy silently; readers that race each trigger a render and each render bumps the
   sequence number, which is not a foreign write. Restore policy is decided by who
   read first. A fixed delay is how the reference project pasted the old clipboard for
   two years.
3. **Letting LLM output reach the inserter without validation**, or streaming it into
   the target. The validator is the last line between a dictated question and its
   answer landing in someone's Slack.
4. **Silent GPU-to-CPU fallback.** Some runtimes do it without an error. Engines are
   constructed under an explicit policy and report what loaded. A GPU product must never
   quietly become a CPU product.
5. **Adding ONNX Runtime, CUDA or a web view to a default build.** Speech and the LLM
   share one ggml/Vulkan runtime for one crash surface and one thing to isolate. The
   ONNX engine stays behind a non-default feature as an escape hatch.
6. **Developing over RDP without knowing it.** In a remote session the RDP clipboard
   process reads every clipboard write within a millisecond, the redirected microphone
   drops samples, and synthetic keys are refused while the session is locked. Say which
   session you measured in. Do not stop system services (the RDP clipboard, clipboard
   history) without saying so and restoring them.

## Hit every surface

A change is not done until each of these that it touches is handled:

- The config key, its default in the default-config constant, and the TOML round trip.
- `doctor` prints the new state; `simulate` exercises the new path.
- The overlay state and sound for any new user-visible outcome.
- The transcript history for any new place text can be lost.
- Per-app policy if the behaviour differs by target application.
- The decision in `docs/design.md` that the change touches: amended in place with a
  dated note when the change refines it, or a new numbered decision when it reverses
  it.
- The measurements section, if the change was justified by a number.

## Building, running, verifying

*(2026-09-24: names paths and versions; re-check when it stops working.)*

The build needs MSVC, CMake, Ninja, the Vulkan SDK and libclang (LLVM; the llama.cpp
bindings run bindgen, and clang-sys finds `C:\Program Files\LLVM\bin\libclang.dll` without
`LIBCLANG_PATH`). CUDA is not needed. Ninja must be on PATH (`uv tool install ninja` or
`winget install Ninja-build.Ninja`): `.cargo/config.toml` makes it the CMake generator,
because the Visual Studio generator races the llama.cpp shader-generator build, and
shortens llama.cpp's nested build paths, which otherwise pass MAX_PATH from a checkout a
few characters deeper than `C:\Github\whisper-local`. Enabling Windows long paths does
not help: MSVC failed with C1083 with `LongPathsEnabled` = 1. A cold
`cargo build --release -p hush` took 336 s on the development machine. The maintainer's
shell is PowerShell; a shell opened before the SDK install needs:

```powershell
$env:VULKAN_SDK = 'C:\VulkanSDK\1.4.357.0'
$env:PATH = "C:\VulkanSDK\1.4.357.0\Bin;C:\Program Files\CMake\bin;$env:PATH"
cargo build --release -p hush
```

In Git Bash the same two are `export VULKAN_SDK='C:\VulkanSDK\1.4.357.0'` and
`export PATH="/c/VulkanSDK/1.4.357.0/Bin:/c/Program Files/CMake/bin:$PATH"`.

Build both binaries: `cargo build --release -p hush -p hush-stt-worker`. Speech runs
in `hush-stt-worker.exe`, which must sit next to `hush.exe`; nothing else is needed
beside them. The split exists because transcribe.cpp and llama.cpp each vendor their
own ggml and clash in one process, and because a GPU driver crash then costs one
utterance instead of the app. `--no-default-features` leaves the embedded LLM out (no
libclang needed); cleanup is then rules-only or over HTTP.

- `hush doctor` reports devices, model, engine backend and the normalizer.
- `hush simulate <wav> [--runs N]` runs one dictation from a WAV into Notepad
  and prints per-stage timings. Fixtures are under `tools/bench-stt/fixtures/`.
- `bench-stt` and `bench-normalize` measure engines and normalizers in isolation.
- Config: `%APPDATA%\hush\config.toml`. Models: `%LOCALAPPDATA%\hush\models`.
  Logs: `%LOCALAPPDATA%\hush\logs`.
- GPUs are chosen by `engine.device` and `normalizer.device`: `"auto"` (discrete card
  with the most free memory), a PCI bus id such as `"0000:05:00.0"`, or a name
  substring. Never an index: Vulkan orders devices differently in an RDP session and on
  the console. Sharing one card with a busy external LLM tripled speech latency in
  measurement; sharing it with the embedded LLM cost nothing measurable (design §7).

Gates before a change is done:

```sh
cargo fmt --all --check
cargo clippy --workspace --all-targets -- -D warnings
cargo test --workspace -- --test-threads=1
```

Tests run single-threaded because the live Win32 tests share the one real clipboard and
keyboard hook. Tests that need a model or a server are gated on environment variables
and skip otherwise; they must never fail for lack of hardware.

Verify the thing you changed, not everything. Use `simulate` for pipeline changes and
the bench tools for engine changes. Some things only a human at the keyboard can check:
the physical hotkey, the overlay's look, the tray menu, pasting into real applications.
Say so explicitly instead of claiming them.

## Commits

Commits are checkpoints: each one builds, passes the gates, and is a state worth rolling
back to. One concern per commit. Nothing under `target/`, no models, no scratch output,
no plans or implementation notes.

Conventional-commit subjects, one short sentence of plain prose after the prefix:
`feat(stt): run Parakeet on ggml/Vulkan by default`, `fix(insert): restore only after
the target read`, `docs(design): record the Vulkan gate result`. Types: `feat`, `fix`,
`perf`, `refactor`, `test`, `docs`, `build`, `chore`. Scope is the crate or area
(`core`, `audio`, `stt`, `normalize`, `platform`, `app`, `bench`, `design`, `agents`).
A breaking change carries `!` after the scope and a `BREAKING:` line in the body.

The body, in this order: the important changes first, as short bullets, with anything
breaking or behaviour-changing marked; then the rationale, what was rejected and what
was measured. A commit that changes a decision names it.

## Documentation

- `docs/design.md` is the decision record: numbered decisions with rationale and
  rejected alternatives, and a measurements section where every number carries its
  method and date. A decision is kept at its current truth: refinements are amended in
  place with a dated note, and a reversal gets a new number that names the one it
  supersedes. Sections that name crates, versions, models or machine state are marked
  perishable and dated.
- `docs/research/` holds dated reports with sources. They are evidence for decisions,
  not instructions; verify before relying on one that is months old.
- Nothing else is documentation. No implementation notes, no plans, no TODO files, no
  status logs. The conversation is the buffer; the commit message is the record.
- Agents can read the code. Do not write down what the code says.

## How it works

Key-down snapshots the focus context and starts capture (a short pre-roll is included
while the microphone is warm). Voice activity detection marks speech segments; closed
segments can be transcribed while the key is still held. Key-up stops capture and
transcribes the tail. The rule pass cleans; the LLM pass runs when enabled and the
utterance warrants it; validation accepts or rejects the LLM text. The inserter writes
the text to the clipboard as a delayed render, sends the paste chord, classifies who
read the clipboard first, and restores the previous contents accordingly, falling back
to paced typing only when nothing read the clipboard at all.

Threads: the hook thread (message loop only), the audio thread (device callback into a
lock-free ring), the UI thread (tray, overlay, clipboard-owner window), a COM thread
for UI Automation with timeouts, and worker threads for speech and normalization. The
pipeline state machine runs on the driver thread and touches none of them directly.

## Where code lives

- `crates/core`: contracts, the pipeline state machine, config, history, cancellation.
  No platform or inference dependencies. Most tests live here.
- `crates/audio`: capture, resampling to 16 kHz mono, voice activity detection.
- `crates/stt`: speech engines and the model manifest and downloader.
- `crates/normalize`: rule pass, prompt, validation, LLM backends, the chain.
- `crates/platform-windows`: everything Win32. Hook, focus, clipboard, input, overlay,
  tray, sound, UI thread.
- `apps/hush`: wiring, the driver, `doctor`, `simulate`.
- `tools/bench-*`: measurement harnesses and their fixtures.
- `spikes/`: standalone experiments with their own workspaces and READMEs.
- `docs/`: decisions and research.

## Taste

**Comments.** The default is none. A comment earns its place only when the constraint
it states cannot be seen from the code: why the obvious alternative is wrong, what a
third-party boundary actually does rather than what its docs say, which measurement
forced the odd-looking value. One or two sentences, at the point of the decision. Never
what the code does, never a narration of the steps, never a section banner, never a
reference to another file, module, function or command (those go stale the moment
someone else's change lands, and that person never sees the comment). Doc comments on
public items only when the name and signature do not already say it. When the constraint
a comment explains goes away, the comment goes with it. If you feel the urge to explain
a block, first try renaming or splitting it.

**Rationale at third-party boundaries.** Where the code integrates a crate or an OS API
in a way that looks wrong at first glance, keep the reason: the API contract that forced
it, the bug it works around, the measurement that chose it. Keep it short and keep it
true; a stale rationale compounds into the next wrong decision.

**Rust.** Edition 2024, clippy clean with warnings denied, `thiserror` in libraries and
`anyhow` only in binaries and tools. No `unwrap` or `expect` outside tests. Every
`unsafe` block carries a one-line safety note; that is the one comment that is always
required. Contracts are traits with data-shaped capabilities, not method probes. Prefer
owned data across thread and process boundaries. Pin the exact version of any crate
that does not follow semver (inference bindings, ONNX wrappers, resamplers that broke
twice in a year).

**Structure.** Platform code stays in the platform crate; inference stays behind an
engine trait; decisions stay in the state machine where a fake can exercise them. Small
functions with names that carry the meaning a comment would have carried. No generics
without a second concrete use. No feature flags without a default build that still does
the whole job.

**Logging.** `tracing` everywhere except the hook callback and the audio callback, which
log nothing. Timing spans on every pipeline stage.

## Additional tips

- The reference project for prior art is Handy; its open issues are a map of what goes
  wrong on Windows. Its clipboard and Vulkan bugs are the ones this design was built
  to avoid, so read the relevant decision before reintroducing either.
- Outside reviews of a plan are cheap and worth it before building against it. The
  first design review changed six decisions; the numbers changed one more.
- If a rule here fights the code, the code wins today and the rule gets fixed in the
  same commit.
