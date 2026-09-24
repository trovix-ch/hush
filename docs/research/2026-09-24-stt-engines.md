# Local STT engine choice for Rust on Windows (research, 2026-09-24)

Perishable: names crates, versions and models as of the date above. Re-verify after 2026-12.

**Recommendation:** NVIDIA Parakeet TDT 0.6B v3 through ONNX Runtime as the default engine on every tier. Whisper large-v3-turbo through `whisper-rs` as the second engine (99 languages, vocabulary prompting). No true streaming in v1; instead VAD-segmented pre-transcription while the key is held. Evaluate `transcribe-cpp` (one ggml/Vulkan backend for both) early.

**Biggest gap:** no measured single-utterance GPU latency for either engine was found. GPU figures below are estimates.

## 1. whisper.cpp via `whisper-rs`
- `whisper-rs` **0.16.0 (2026-03-12)**; GitHub archived, development on [Codeberg](https://codeberg.org/tazz4843/whisper-rs). Upstream whisper.cpp **v1.9.4 (2026-09-11)**; v1.9.0 (June 2026) added Parakeet support ([PR #3735](https://github.com/ggml-org/whisper.cpp/pull/3735)). 0.16.0 predates v1.9 so probably no Parakeet exposure (unverified).
- Features: `cuda`, `vulkan`, `openblas`, `openmp`, `hipblas`, `intel-sycl`, `raw-api`. API has `WhisperVadContext`, segment/abort/progress callbacks.
- **Windows build:** MSVC + CMake + LLVM/libclang for bindgen (`LIBCLANG_PATH`). `cuda` needs CUDA Toolkit; `vulkan` needs Vulkan SDK ([BUILDING.md](https://github.com/tazz4843/whisper-rs/blob/master/BUILDING.md)). RTX 50-series (Blackwell, `sm_120`) needs CUDA **12.8+** and explicit `CMAKE_CUDA_ARCHITECTURES=120` ([#2892](https://github.com/ggml-org/whisper.cpp/issues/2892)). libclang 22 needs bindgen ≥ 0.72.1.
- **Vulkan vs CUDA for distribution:** Vulkan needs no CUDA runtime on the user's machine; Handy ships Vulkan. Vulkan init reportedly takes 5–30 s on first load ([snailtext](https://snailtext.app/blog/whisper-vs-parakeet-tdt/)); keep the model resident.
- Models ([HF ggerganov/whisper.cpp](https://huggingface.co/ggerganov/whisper.cpp/tree/main)):

  | Model | f16 | q8_0 | q5 |
  |---|---|---|---|
  | large-v3-turbo | 1.62 GB | 874 MB | 574 MB |
  | medium | 1.53 GB | 823 MB | 539 MB |
  | small | 488 MB | 264 MB | 190 MB |
  | base | 148 MB | 82 MB | 60 MB |

  Open ASR Leaderboard (2026-03-27): large-v3-turbo **7.83% WER**, distil-large-v3.5 7.21% ([arXiv 2510.06961](https://arxiv.org/pdf/2510.06961)).
- **Latency:** RTX 4070 CUDA large-v3 ≈ 8× real-time ([promptquorum](https://www.promptquorum.com/power-local-llm/local-whisper-stt-comparison-2026)); 10 s clip 0.3–1.5 s on a laptop CPU with small. Estimate (unverified): large-v3-turbo on a 4090/5090 ≈ 0.2–0.5 s per 10 s.
- **Problems:** hallucination on silence ("Thank you for watching"), repetition loops ([arXiv 2501.11378](https://arxiv.org/html/2501.11378v1)). Mitigate with VAD first, `no_speech_thold`, temperature fallback, `no_context`, `single_segment`, `initial_prompt` for vocabulary.

## 2. Parakeet TDT 0.6B v3 via ONNX Runtime
- Released 2025-08-14, **CC-BY-4.0** (attribution required), 25 European languages, built-in punctuation, capitalisation and word timestamps. **6.34% avg Open ASR WER**, 1.93% LibriSpeech clean ([model card](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)). int8 ONNX ≈ 640–680 MB.
- **CPU:** RTF ≈ 0.033 on i7-12700KF int8 → 10 s clip ≈ 0.33 s, ~4× faster than Whisper small ([snailtext](https://snailtext.app/blog/whisper-vs-parakeet-tdt/)). GPU single-utterance: no measurement; estimate < 100 ms per 10 s.
- **Rust paths:**
  - **`parakeet-rs` 0.3.8 (2026-09-23)**, MIT/Apache, on `ort`, features `cuda`, `tensorrt`, `directml`, `webgpu`, `openvino`. TDT v3, English CTC, Nemotron cache-aware streaming, EOU, Sortformer diarisation ([GitHub](https://github.com/altunenes/parakeet-rs)). Most on-target.
  - **`sherpa-onnx` crate 1.13.8 (2026-09-11)**, official; static link, downloads prebuilt libs for Windows; NeMo transducers, Whisper, Moonshine, SenseVoice, Silero/TEN VAD, punctuation. `sherpa-rs` is deprecated. Hotword biasing only with `modified_beam_search` and has an open hallucination issue ([#3267](https://github.com/k2-fsa/sherpa-onnx/issues/3267)).
  - **`ort` 2.0.0-rc.13 (2026-07-28)**, wraps ONNX Runtime 1.28. Prebuilt Windows binaries include CUDA, DirectML, TensorRT EPs. CUDA EP needs **CUDA ≥ 13.2 and cuDNN ≥ 9.23** ([ort EP docs](https://ort.pyke.io/perf/execution-providers)) — heavy to redistribute. **ort silently falls back to CPU unless `.error_on_failure()` is set.**
  - `transcribe-rs` 0.3.11: Handy's wrapper over ONNX + whisper.cpp.
- **DirectML:** maintenance mode (new work in Windows ML) but supported; runs on any DX12 GPU ([microsoft/DirectML](https://github.com/microsoft/DirectML)).

## 3. Other candidates
- **transcribe.cpp / `transcribe-cpp` 0.2.3 (2026-08-30, MIT):** GGUF on ggml; Whisper, Parakeet (11 variants, streaming), Moonshine, Canary, Granite, Cohere Transcribe, Qwen3-ASR; CUDA/Vulkan/ROCm. Handy uses it. One Vulkan build could cover every model; young.
- **Moonshine v2 streaming (Feb 2026, MIT):** tiny 34M params, 12.01% avg WER. For very weak CPUs only.
- **Kyutai STT:** websocket server, not a library; CUDA 13 build issues. Poor fit.
- **Voxtral Mini 4B Realtime (Apache-2.0):** vLLM only; llama.cpp support pending ([#20914](https://github.com/ggml-org/llama.cpp/issues/20914)). Revisit later.
- **Canary-Qwen 2.5B, Cohere Transcribe 2B (Apache-2.0, 5.42%), Granite Speech 4.1 (5.33%):** accuracy leaders within one WER point; Cohere is supported by parakeet-rs and transcribe.cpp → plausible "max accuracy" option later ([MarkTechPost 2026-07-23](https://www.marktechpost.com/2026/07/23/best-open-speech-recognition-asr-models-in-2026-wer-languages-latency-and-license-compared/)).
- **Whisper via candle:** no benchmarks found; probably slower. **Whisper via ONNX:** hand-written beam search; not worth it. **Windows built-in speech:** old engine / cloud / Copilot+ only. Not competitive.

## 4. VAD
- **Silero v6:** ~16% fewer errors than v5 on noisy data; 512-sample (32 ms) chunks under 1 ms each ([releases](https://github.com/snakers4/silero-vad/releases)).
- Rust: `voice_activity_detector` 0.2.1 (Aug 2025, pins older `ort` — version conflict risk); sherpa-onnx VAD; whisper.cpp built-in Silero since v1.7.6 (`WhisperVadContext`).
- Use in the app, not inside an engine: trim leading/trailing silence with ~200 ms padding; drop no-speech recordings (main defence against Whisper hallucination); optional auto-stop; mark pause boundaries for chunked pre-transcription.

## 5. Streaming vs batch
- Parakeet makes streaming unnecessary for speed (60 s at RTF 0.033 ≈ 2 s on CPU; well under 1 s on GPU, estimate).
- **Better: VAD-segmented pre-transcription.** Transcribe each segment once a pause closes it while the key is held; on release only the tail remains.
- True streaming models (Nemotron Speech Streaming 0.6B, Moonshine v2, Kyutai, Voxtral Realtime) are worth it only for a live preview. Whisper cannot really stream.

## 6. Audio capture
- `cpal` **0.18.2 (2026-08-16)**: auto-reroutes on default device change (`DeviceChanged`), uses windows-rs, real-time thread priority. `rubato` **5.0.0 (2026-08-10)** — pin it; v4 and v5 both shipped in 2026.
- WASAPI shared mode delivers the device mix format (typically 48 kHz stereo f32). Don't request 16 kHz. Take the default input config, downmix to mono in the callback, push into a lock-free ring buffer (Handy uses `rtrb`), resample 48k→16k on a worker (exact 3:1). Handle I16/I24 too.
- COM apartment: cpal switched to STA by default ([PR #597](https://github.com/RustAudio/cpal/pull/597)); own the stream on a dedicated audio thread.
- Keep the stream open while idle, plus a small pre-roll buffer, so the first syllable isn't lost.

## 7. Recommendation
- **GPU tier:** Parakeet TDT 0.6B v3 via `parakeet-rs`/`ort`, **DirectML EP by default** (no CUDA install), CUDA optional. Whisper large-v3-turbo q8_0 via `whisper-rs` + `vulkan` for other languages and `initial_prompt`. Spike `transcribe-cpp` with Vulkan early.
- **CPU tier:** Parakeet v3 int8 on CPU EP (~0.33 s per 10 s on a desktop i7; laptops slower, unverified). Fallbacks: Whisper base/small q5_1, Moonshine.
- **Engine trait sketch:** `info()` (languages, caps: prompt / hotwords / word_timestamps / streaming / punctuation), `warm_up()`, `transcribe(pcm16k_mono, DecodeOptions)`, optional `stream()`. Capabilities as data; VAD/resampling/stitching outside the engine; backend chosen at construction and the one that actually loaded is reported.
- **Model management:** manifest per model (URL, size, SHA-256 = HF LFS oid, license); resumable HTTPS download with hash check; store under `%LOCALAPPDATA%\<app>\models` (Handy's Roaming AppData is wrong for GB files). Show CC-BY attribution for Parakeet.

**Unverified:** every GPU single-utterance latency, candle performance, whisper-rs 0.16 Parakeet exposure, Parakeet CPU speed on laptop chips. First thing to build: a benchmark that runs a 10 s clip through each engine on the target GPU and a laptop CPU.
