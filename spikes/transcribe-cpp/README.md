# Spike: transcribe-cpp on Vulkan (D13 gate) — 2026-09-24

Standalone Cargo project (own `[workspace]` table, not part of the main workspace).
Everything below is perishable: measured 2026-09-24 on the development machine
(Windows 11, Ryzen 9 9900X, 2x RTX 5060 Ti 16 GB, NVIDIA driver as installed that day).

Route that produced the numbers: **the Rust crate** (`transcribe-cpp` =0.2.3 with
`default-features = false, features = ["vulkan"]`). The C++ CLI fallback was not needed.

## Reproduce

Prerequisites: Rust 1.97 MSVC, VS 2022 (MSVC 14.44), CMake 4.4, Vulkan SDK 1.4.357.0
(winget). libclang is not needed (the sys crate ships pre-generated bindings).

```sh
# Git Bash; a fresh shell after the SDK install has VULKAN_SDK already
export VULKAN_SDK='C:\VulkanSDK\1.4.357.0'
export PATH="/c/VulkanSDK/1.4.357.0/Bin:/c/Program Files/CMake/bin:$PATH"
cargo build --release          # cold: 166 s including the native CMake + shader build
M="$LOCALAPPDATA/whisper-local/models/spike"
F=../../tools/bench-stt/fixtures
./target/release/spike-transcribe-cpp.exe "$M/parakeet-tdt-0.6b-v3-F16.gguf" \
  $F/tts-03s-question.wav $F/tts-10s-fillers.wav $F/tts-30s-dictation.wav \
  $F/silence-02s.wav $F/jfk.wav [--device N] [--runs 5]
```

`--device N` indexes the Vulkan devices only (0 = AMD iGPU, 1 = RTX at PCI 05:00,
2 = RTX at PCI 01:00 on this machine). Without it the backend picked device 1.

`build.rs` exists for one reason: transcribe-cpp-sys 0.2.3 asks the linker for
`vulkan-1.lib` by bare name without adding `%VULKAN_SDK%\Lib` to the search path, so
the first link failed with `LNK1181: cannot open input file 'vulkan-1.lib'`. No other
flags or env vars were needed; the sys crate's MAX_PATH junction under `%LOCALAPPDATA%\tcs`
worked without a short `CARGO_TARGET_DIR`.

Method: one process per model; load, then a warm-up run on the 3 s clip, then per clip
5 timed `Session::run` calls (wall clock around the call, `TimestampKind::None`,
language `en`), then one run on the first 90 % of the clip (a length not seen before).

## Models (Hugging Face, `resolve/main/<file>`)

| file | repo | bytes | sha256 (prefix) |
|---|---|---|---|
| parakeet-tdt-0.6b-v3-F16.gguf | handy-computer/parakeet-tdt-0.6b-v3-gguf | 1 255 869 856 | d9ec7e2c39da |
| parakeet-tdt-0.6b-v3-Q8_0.gguf | handy-computer/parakeet-tdt-0.6b-v3-gguf | 739 508 576 | 5859f7794 |
| whisper-large-v3-turbo-F16.gguf | handy-computer/whisper-large-v3-turbo-gguf | 1 625 935 520 | e1d0144e9afc |

## Results (ms, median of 5 after warm-up; device 1 = PCI 05:00 unless noted)

| model | 3.4 s clip | 9.5 s clip | 31 s clip | 2 s silence | jfk 11 s | load | warm-up | peak VRAM |
|---|---|---|---|---|---|---|---|---|
| Parakeet F16 | 41.9 | 79.3 | 221.5 | 24.4 | 77.7 | 1626 | 59 (5699 first ever) | 3.9 GB |
| Parakeet F16, device 2 (PCI 01:00) | 33.2 | 67.2 | 181.8 | – | – | 928 | 45 | – |
| Parakeet Q8_0 | 39.7 | 77.4 | 216.2 | 24.8 | 79.7 | 897 | 2052 | 3.3 GB |
| Whisper large-v3-turbo F16 | 144.6 | 194.6 | 400.8 | 127.7 | 167.5 | 1755 | 2166 | 4.2 GB |

All values, Parakeet F16 device 1: 3 s [42.3, 42.4, 41.9, 41.3, 41.5];
10 s [116.1, 79.3, 78.3, 86.9, 78.7]; 30 s [1778.3, 236.5, 221.5, 214.7, 218.0].
Device 2: 3 s [34.0, 34.8, 33.0, 33.2, 33.2]; 10 s [67.0, 76.1, 67.2, 65.3, 69.7];
30 s [190.5, 176.1, 178.5, 188.6, 181.8]. 100 runs of the 10 s clip on device 2:
p50 74.0, p95 78.9, max 87.4.

RTF (Parakeet F16, device 1): 0.012 / 0.008 / 0.007 for 3 / 10 / 30 s.
Unseen 90 %-length runs cost the same as the full clip (no per-shape penalty once warm).

Transcripts: Parakeet (both quants) was word-for-word identical to the DirectML output on
all three TTS clips, and produced empty text on silence. Whisper turbo matched on 3 s
and 10 s, appended a hallucinated "end." on the 30 s clip, and returned "you" on silence.

VRAM is `nvidia-smi` memory.used on the benchmarked GPU sampled every 200 ms, idle
baseline 12–31 MiB. No crash, device-lost or validation error in any run.

### First-time costs
The first run of each new pipeline variant on this machine pays a driver shader compile:
5.7 s for the very first 3 s run, 1.8–2.8 s for the first 30 s run of each quant. The
NVIDIA driver caches these on disk; a later process ran its 30 s warm-up in 231 ms. A
product should warm up with a long dummy clip at startup, after install or driver update.

## DirectML comparison (same fixtures, same session)

`tools/bench-stt` release binary built 04:07 (ORT 1.28.0, fp32 Parakeet, joint on CPU,
default settings), copied out and run once per clip, 5 runs:

| clip | DirectML median | Vulkan F16 median (dev 1 / dev 2) |
|---|---|---|
| 3.4 s | 172.7 [299.4, 171.2, 172.7, 173.9, 163.5] | 41.9 / 33.2 |
| 9.5 s | 168.3 [284.8, 165.6, 164.3, 168.3, 205.8] | 79.3 / 67.2 |
| 31 s | 305.7 [454.6, 288.2, 376.9, 298.3, 305.7] | 221.5 / 181.8 |

Which GPU DirectML used was not recorded; Vulkan is ahead on either card.
