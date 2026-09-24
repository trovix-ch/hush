# hush

Hold a key, talk, let go. Clean text appears wherever your cursor is. Nothing leaves your
computer.

hush is a Windows dictation tool in the spirit of Wispr Flow, built in Rust and
running every step locally: speech recognition, the language-model cleanup that removes
"um"s and applies your mid-sentence corrections, and the insertion into the app you are
using.

## Status

*(2026-09-24)* Pre-alpha, milestone 1 wired: hold-to-talk → Parakeet on Vulkan → rule
pass and LLM cleanup (Qwen3-4B on an embedded llama.cpp, or a local server such as
Ollama) → paste into the focused app, with
tray, overlay, sounds, Escape to cancel and a transcript history. Verified end to end
into Notepad with `simulate`; the physical hotkey path still needs a human at the
keyboard.

## Running

*(2026-09-24; names paths and versions, re-check before trusting)*

Build needs Visual Studio's C++ tools, the Vulkan SDK, CMake, Ninja and LLVM (for
libclang). Ninja must be on PATH: `uv tool install ninja` or
`winget install Ninja-build.Ninja`; the repository's `.cargo/config.toml` makes it the
CMake generator. The installers register the rest system-wide; in a shell opened before
the install, set them for the session (PowerShell):

```powershell
$env:VULKAN_SDK = 'C:\VulkanSDK\1.4.357.0'
$env:PATH = "C:\VulkanSDK\1.4.357.0\Bin;C:\Program Files\CMake\bin;$env:PATH"
cargo build --release -p hush -p hush-stt-worker
```

Speech runs in a child process, `hush-stt-worker.exe`, which must sit next to `hush.exe`
(or be named by `engine.worker_path`); copy both. Neither needs a DLL beside it. If the
GPU driver crashes the worker, hush loses that one utterance and restarts it.
`--no-default-features` builds without the embedded language model (and without LLVM).
`--features in-process-stt` adds `engine.in_process = true`, speech inside hush for
debugging, which then needs the `transcribe.dll` and `ggml*.dll` built next to it.

- `hush` runs the app. Hold **Right Ctrl**, speak, release. Tray → Quit (or
  Ctrl+C in its console) exits.
- `hush doctor` checks microphone, GPUs, the speech and language models (downloads
  them if missing), engine load and the normalizer.
- `hush simulate <wav> [--target notepad|foreground] [--runs N]` runs one
  dictation from a WAV through the whole pipeline and prints per-stage timings.
- Config: `%APPDATA%\hush\config.toml`, written on first run with every key
  commented. `engine.device = "auto"` (the default) puts speech on the discrete GPU
  with the most free memory, skipping integrated GPUs; the built-in language model
  follows speech unless `normalizer.device` says otherwise. To pin a card, write its
  PCI bus id as `doctor` prints it (`"0000:05:00.0"` or `"05:00"`) or part of its
  name; with two GPUs, pin speech to the one not running another LLM server. Device
  indexes are not accepted: Windows numbers the GPUs differently in console and
  remote sessions. The old `gpu_device` index still works for now, with a warning.
  Logs: `%LOCALAPPDATA%\hush\logs\`.

## Principles

- **Local only.** No cloud, no accounts, no telemetry. Model files are downloaded once.
- **Fast.** Under a second from letting go of the key to text on screen on a modern GPU.
- **Never loses your words.** If the text cannot be inserted, it is on your clipboard and
  the app says so.
- **Cleans, never answers.** Dictate a question and you get the question, punctuated.
- **Small.** A tray icon and an overlay pill. No main window, no web view.

## Layout

See `docs/design.md` §5. Decisions are numbered in §4; research notes are dated in
`docs/research/`.
