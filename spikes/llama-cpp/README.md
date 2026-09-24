# Spike: embedded llama.cpp on Vulkan (milestone-2 gate): 2026-09-24

Standalone Cargo project with its own `[workspace]` table, not part of the main workspace.
Everything below is perishable: it was measured on 2026-09-24 on the development machine
(Windows 11, Ryzen 9 9900X, 2x RTX 5060 Ti 16 GB, NVIDIA driver 591.86, local console
session). The LLM ran on the card at PCI 05:00. Ollama held the card at PCI 01:00 the whole
time.

The numbers come from **the Rust crate**: `llama-cpp-2` =0.1.157 (llama.cpp's ggml 0.24.0),
`default-features = false, features = ["vulkan", "openmp"]`. The CMake fallback was not
needed.

## Question and answer

Can the shipping backend in D5 (embedded llama.cpp, Vulkan, resident context, cached
prefix state, greedy decoding, token cap, anti-preamble grammar) run the HTTP path's
prompt contract with the same results, inside the latency budget? **Yes**, with the
caveats under "What did not transfer".

## Reproduce

Prerequisites: Rust 1.97 MSVC, VS 2022 (MSVC 14.44), CMake 4.4.3, Vulkan SDK 1.4.357.0,
LLVM 23 (libclang for bindgen), and **Ninja** on PATH (`uv tool install ninja` gives
1.13.2).

```sh
# Git Bash
export VULKAN_SDK='C:\VulkanSDK\1.4.357.0'
export PATH="/c/VulkanSDK/1.4.357.0/Bin:/c/Program Files/CMake/bin:$PATH"
export CMAKE_GENERATOR=Ninja        # required, see "Build" below
export CARGO_TARGET_DIR='C:\lct'    # required: a short path, see "Build" below
cargo build --release               # cold: 85 s, including the llama.cpp + shader build
M="$LOCALAPPDATA/hush/models/spike"
/c/lct/release/spike-llama-cpp.exe "$M/Qwen3-4B-Instruct-2507-Q4_K_M.gguf" 2> llama.log
/c/lct/release/spike-llama-cpp.exe "$M/granite-4.0-micro-Q4_K_M.gguf" 2> llama.log
# flags: --pci 05:00 (default) | --device N, --runs 3, --case SUBSTRING, --kv-unified
grep offloaded llama.log            # llama.cpp logs the layer offload to stderr
```

`cargo build --release --features dynamic-backends` also builds (90 s cold). It runs when
`<target>/release/build/llama-cpp-sys-2-*/out/bin` (llama.dll, ggml*.dll) is on PATH. The
backend DLLs (ggml-vulkan plus nine ggml-cpu variants) are loaded from a directory path
that is compiled into the binary as an absolute path. Its latency was the same as the
static build (question cases: p50 104 ms, cached).

### Build: what failed and what fixed it

1. The first build failed with `C1083: Cannot open compiler generated file: ''` inside
   the nested `vulkan-shaders-gen` ExternalProject. The cause was MAX_PATH under
   `spikes\llama-cpp\target\...`. A short `CARGO_TARGET_DIR` (`C:\lct`) fixed it.
2. The build then failed with the Visual Studio generator. The ExternalProject's
   configure, build and install steps ran concurrently (`file INSTALL cannot find
   ...vulkan-shaders-gen.exe`, `MSB8066`). This reproduced on every retry. The sys crate
   forces `TrackFileAccess=false`, and the generated vcxproj builds custom commands in
   parallel (`BuildInParallel`, `UseMultiToolTask`), so the steps race. **The fix was
   `CMAKE_GENERATOR=Ninja`.** No vcvars shell was needed, because cmake-rs passes the
   MSVC environment through.
3. **libclang is required** because the sys crate runs bindgen 0.72 at build time.
   `LIBCLANG_PATH` is not required: a cold build without it took 76 s, and clang-sys found
   `C:\Program Files\LLVM\bin\libclang.dll` by itself. That differs from transcribe-cpp,
   whose sys crate ships pre-generated bindings. A `build.rs` adding `%VULKAN_SDK%\Lib`
   (as in the transcribe-cpp spike) was not needed either, because this sys crate adds
   the path itself.

The main workspace will hit issues 1 and 2 when it takes this crate. Plan on Ninja and a
short target directory, or a junction.

## Method

- The rule pass, the user-message format and validation come from the real crate:
  `hush-normalize` is a path dependency, called as `bench-normalize` calls it. The system
  prompt was **copied verbatim** into the spike for the measurements, and the binary
  asserted byte equality with the crate at startup. Since the embedded backend moved into
  the crate (milestone 3), the spike imports the prompt and the grammar from it.
- The chat is rendered with the model's own template through
  `LlamaModel::apply_chat_template`, which is llama.cpp's built-in template matcher (the
  chatml family for Qwen, granite for Granite). It is split at the user message.
  Everything before the split depends only on the language. It is decoded once into its
  own KV sequence: English in seq 0 (566 tokens), German in seq 1 (651 tokens). Each
  request runs `llama_memory_seq_rm(seq, prefix_len, -1)`, then decodes only the request
  part. Tokenizing the prefix and the suffix apart gave the same tokens as tokenizing the
  whole chat in 30 of 30 cases.
- Decoding is greedy (`LlamaSampler::greedy`), with a hard cap of
  `ceil(1.5 x transcript tokens) + 20`. This uses real token counts; HTTP estimates
  1.3 tokens per word. No case reached the cap.
- Thinking: the Qwen3-2507 generation prompt is a bare `<|im_start|>assistant\n`. The
  template mentions `<think>` only when it re-renders earlier assistant turns. No output
  contained `<think>`: this was checked on each case's first run in every mode, and the
  repeat runs never differed from the first. The Granite template has no `<think>`.
- Grammar: the spike's grammar module (now the crate's) builds a GBNF that is the complement of a trie of forbidden
  leading prefixes: `Here`, `Sure`, `Certainly` (both cases), `"`, a backtick, `“`, `„`,
  `«`, and any leading whitespace. A word is exempt when the source itself starts with it,
  so a dictated "Sure, ..." still works. Applying the grammar to the whole 151k-token
  vocabulary cost ~25 ms per token (measured). So each token is sampled greedily first and
  checked against the grammar alone (5 µs). The run resamples under the grammar only if
  that token is rejected, which is the same approach as llama.cpp's common sampler.
  Building the grammar costs 0.03 ms.
- Timing: the wall clock runs from template rendering to the final token.
  - "Prefill" is time to first token: decoding the request part plus one sample.
  - "Decode" is the remaining tokens.
  - Validation and the rule pass are excluded, as in `bench-normalize`.
- Each model gets a warm-up (one run per mode on the longest case), then 30 fixtures x
  3 runs in each mode. p50 and p95 are nearest-rank over the 90 runs, as in
  `bench-normalize`. The verdict comes from the first run.
- Each model was run in two separate processes. The table below shows run 1 with run 2 in
  parentheses.
- VRAM is `nvidia-smi` memory.used on the card at 05:00. The idle baseline was 12 MiB. The
  reading was taken after load, context creation and warm-up.

## Models (Hugging Face, `resolve/main/<file>`)

| file | repo | bytes | sha256 |
|---|---|---|---|
| Qwen3-4B-Instruct-2507-Q4_K_M.gguf | unsloth/Qwen3-4B-Instruct-2507-GGUF | 2 497 281 120 | 3605803b982cb64aead44f6c1b2ae36e3acdb41d8e46c8a94c6533bc4c67e597 |
| granite-4.0-micro-Q4_K_M.gguf | ibm-granite/granite-4.0-micro-GGUF | 2 099 502 528 | 97c417dcc0534b0737c74016fb2af083cb17c3b51eaac621192d23961b7024eb |
| Ollama's `qwen3:4b-instruct-2507-q4_K_M` blob (the model the HTTP numbers used) | Ollama library | 2 497 280 480 | 85e4a5b7b8ef0e48af0e8658f5aaab9c2324c76c1641493f4d1e25fce54b18b9 |

`Qwen/Qwen3-4B-Instruct-2507-GGUF` answered 401 on the API without a token, so unsloth's
file was used.

## Results

| model | load | VRAM | cached p50 / p95 ms | cached + grammar p50 / p95 | uncached p50 / p95 | validated | raw must_not_contain / inserted | matches expected | grammar |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-4B-2507 Q4_K_M (unsloth) | 2.2 s | 3306 MiB | 109 / 184 (105 / 173) | 113 / 184 (107 / 174) | 251 / 436 (245 / 325) | 28/30 | 0 / 0 | 26/28 | works, 0 resamples |
| Qwen3-4B-2507 Q4_K_M (Ollama's blob) | 2.8 s | 3306 MiB | 106 / 174 | 107 / 175 | 245 / 324 | **29/30** | 0 / 0 | 27/28 | works |
| Granite 4.0 Micro Q4_K_M | 1.8 s | 2571 MiB | 100 / 181 (99 / 177) | 122 / 419 (120 / 270) | 213 / 298 (221 / 315) | 24/30 | 3 / 0 | 21/28 | works, 0 resamples |
| *HTTP, Ollama, §7: Qwen3-4B* | | | *100 / 162 (uncacheable client-side)* | | | *29/30* | *0 / 0* | *27/28* | *n/a* |
| *HTTP, Ollama, §7: Granite* | | | *119 / 324* | | | *24/30* | *3 / 0* | *21/28* | *n/a* |

Breakdown (Qwen, unsloth, cached):
- Prefill: a mean of 46 tokens in 29 ms.
- Decode: a mean of 10.3 tokens in 89 ms, which is **115 tok/s**. Decode is 80 % of the
  request.
- Uncached prefill: 626 tokens in 155 ms, so the prefix cache saves about 130 ms per
  request.
- Granite decodes at 123 tok/s.

Ollama, measured through its API that day with the same weights, decodes on CUDA at
106–127 tok/s. Vulkan decode speed is therefore at parity with CUDA here.

Other measurements:

- **Prefix decode, once per language:** Qwen: English 131 ms, German 163 ms. Granite:
  English 114 ms, German 140 ms.
- **First run ever on this machine:** the first English prefix took **20.9 s** because of
  Vulkan pipeline compilation. The NVIDIA driver caches the result. The next process took
  153 ms (Qwen); Granite's first run took 554 ms.
- **`state_seq_get` / `state_seq_set` as an alternative to `seq_rm`:** 30 / 32 ms for
  83.5 MB (Qwen), 20 / 18 ms for 46 MB (Granite). That is slower than a whole cached
  request's prefill, so `seq_rm` on one sequence per language is the right mechanism.
- **Determinism:** outputs were identical with the cache on and off, and with the grammar
  on and off, on 30 of 30 cases. No output differed across the 3 runs.
- **Layers offloaded:** 37/37 (Qwen) and 41/41 (Granite) to Vulkan1 = PCI 05:00. The
  device was chosen by the PCI id in `ggml_backend_dev_props.device_id`. The ggml device
  order is:
  - 0 = AMD iGPU, of type IGPU
  - 1 = 05:00
  - 2 = 01:00
  - 3 = CPU

  If no GPU matches, the binary fails; it never falls back to the CPU.
- **Unified KV buffer (`--kv-unified`):** no gain (p50 112 against 109 ms).
- **Rejections:**
  - Qwen, unsloth: `code-git-commit` ("-m") and `code-cargo` ("--workspace"), both by the
    verbatim check.
  - Qwen, Ollama's blob: `code-git-commit` only, exactly as over HTTP.
  - Granite: the same six as over HTTP. These were the "4:30" number check, the
    translation answer, "Arr matey!", "Git" twice and "--workspace".

**Grammar proof.** The fixtures never tempt either model into a preamble (0 resamples in
180 grammar runs per model). A self-test proves the grammar instead:
- **Token mask:** `Here`, `Sure`, `Certainly`, `"`, a backtick, ` Here` and `\n` are
  masked as a first token. `What`, `Her` and `He` are allowed.
- **Bait chat:** a chat that asks for a reply starting "Here is" produced "Here is a
  short sentence..." with the grammar off. With the grammar on it produced "_here is_
  rain falling..." (Qwen) and "Rain nourishes the earth..." (Granite).

So the grammar blocks the literal prefix, and a determined model routes around it.

**Granite's grammar-on column.** The slower grammar-on numbers for Granite are run noise,
not grammar cost:
- There were 0 resamples, and the grammar costs 5 µs per token.
- The slow runs were clustered in the second half of that pass.
- The same case in isolation (`--case de-fillers --runs 5`) gave 158 ms with the grammar
  and 158 ms without it.

## Prompt-lookup (n-gram) speculation

Skipped. `llama-cpp-2` 0.1.157 exposes only `MtpSpeculative`, which needs a model with
multi-token-prediction heads. There is no n-gram or prompt-lookup API. It would need a
hand-written draft/verify loop (batch the drafted tokens, compare
`get_logits_ith` argmaxes, `seq_rm` the rejected tail). Decode is 80 % of the request
time, so this is the largest lever left, and it has not been measured.

## Verdict

The embedded path meets the contract and the budget. On Ollama's exact weights it
reproduces the HTTP result:
- 29/30 validated and 27/28 matching expected.
- The same single rejection, 0 forbidden text in the raw output, and none inserted.

Qwen needs 105–109 ms p50 and 173–184 ms p95 with the prefix cached, against the HTTP
path's 100 / 162 ms. Next to about 80 ms of speech, that leaves most of the one-second
budget unspent. The prefix cache is worth about 130 ms per request (uncached p50 is
245–251 ms). Outputs were bit-identical with the cache on and off, and the grammar is free
once it is applied check-first.

What did not transfer from HTTP:
1. **Latency did not improve.** Ollama already caches the prefix and decodes on CUDA at
   the same ~115 tok/s, so the HTTP hop and the lost prefix snapshot were never the cost.
   The embedded backend's gains are the grammar, the lack of an external process and
   keep-alive cold loads, and determinism. Speed is not among them.
2. **The weights are part of the contract.** Unsloth's Q4_K_M of the same model validated
   28/30 instead of 29/30: it rewrote "dash dash workspace" as "--workspace". Ollama's
   blob, run through this same binary, gave 29/30. The bundled GGUF must be pinned by
   sha256, and the fixture gate re-run on that exact file.
3. **Chat templates differ.** Ollama's template appends `\n\n` to the system text;
   llama.cpp's built-in chatml does not. The results were still the same, but the serialized
   bytes are not identical across backends, so "exact message serialization" in D5 holds
   at the message level only.
4. **The grammar does not remove the validator.** It blocks literal preamble tokens only,
   and the bait test shows the model routing around it.
5. **Build requirements that the main workspace does not have yet:** Ninja, a short
   target path, and libclang. The last contradicts the "libclang is not needed" line in
   AGENTS.md once this crate joins the workspace. Also, the first Vulkan run on a fresh
   driver cache costs about 21 s, so a startup warm-up is not optional.
