# hush

Hold a key, talk, let go. Clean text appears wherever your cursor is. Nothing leaves your
computer.

hush is a Windows dictation tool in the spirit of Wispr Flow, built in Rust and
running every step locally: speech recognition, the language-model cleanup that removes
"um"s and applies your mid-sentence corrections, and the insertion into the app you are
using.

## Status

*(2026-09-24)* Pre-alpha, milestone 1 wired: hold-to-talk → Parakeet on Vulkan → rule
pass (optional LLM cleanup through a local Ollama) → paste into the focused app, with
tray, overlay, sounds, Escape to cancel and a transcript history. Verified end to end
into Notepad with `simulate`; the physical hotkey path still needs a human at the
keyboard.

## Running

*(2026-09-24; names paths and versions, re-check before trusting)*

Build needs the Vulkan SDK and CMake. Their installers register both system-wide; in a
shell opened before the install, set them for the session (PowerShell):

```powershell
$env:VULKAN_SDK = 'C:\VulkanSDK\1.4.357.0'
$env:PATH = "C:\VulkanSDK\1.4.357.0\Bin;C:\Program Files\CMake\bin;$env:PATH"
cargo build --release -p hush
```

- `hush` runs the app. Hold **Right Ctrl**, speak, release. Tray → Quit (or
  Ctrl+C in its console) exits.
- `hush doctor` checks microphone, GPUs, model (downloads it if missing),
  engine load and the normalizer server.
- `hush simulate <wav> [--target notepad|foreground] [--runs N]` runs one
  dictation from a WAV through the whole pipeline and prints per-stage timings.
- Config: `%APPDATA%\hush\config.toml`, written on first run with every key
  commented. With two GPUs set `engine.gpu_device` to the one not running your LLM
  (`doctor` lists them). Logs: `%LOCALAPPDATA%\hush\logs\`.

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
