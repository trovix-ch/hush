# Panel review — 3 reviewers
Root: `C:\Github\whisper-local` · 392.1s · $0.3429 total · 74% prompt cache hit

## deepseek-v4-pro — verification lens  (`deepseek/deepseek-v4-pro`)
*23 tool calls · 152,160 in (90% cached) / 17,059 out (incl. 12,461 reasoning) · $0.0571 · 392.0s*

Here's my review of `docs/design.md` against the code and research.

---

## 1. DirectML as the default STT backend (D2)

**Verdict: sound but under-tested for the claim.**

The research itself flags DirectML as "maintenance mode (new work in Windows ML)" (`docs/research/2026-09-24-stt-engines.md:35`). That's the core tension: the plan bets on a provider Microsoft has stopped actively developing in favor of Windows ML / WinML (used by `foundry-local-sdk`, noted in the LLM research). For a tool targeting end users in 2026, this isn't a trap yet — DirectML is in Windows 11, DX12 GPUs aren't going anywhere, and the `ort` crate ships prebuilt DirectML binaries — but the migration path if Microsoft deprecates it is unclear. The design acknowledges this implicitly by listing `transcribe-cpp` as an early evaluation target (`docs/design.md:121-122`).

The single-ggml/Vulkan-backend question is the better strategic bet the design already identifies: `transcribe-cpp` covers both Parakeet and Whisper under one Vulkan build, and llama.cpp (D5) also uses Vulkan. One Vulkan runtime for both STT and LLM eliminates the DirectML maintenance-mode risk and the ONNX Runtime redistribution dependency entirely. The research says `transcribe-cpp` is young (0.2.3, Aug 2026) but Handy already ships it (`docs/research/2026-09-24-wispr-flow-and-prior-art.md:50`). The M0 measurement plan must A/B DirectML against Vulkan on the same hardware, not just measure DirectML in isolation.

**Bottom line:** DirectML is defensible as a starting point, but the design should commit to a decision point after M0: if Vulkan latency is within 20% of DirectML, drop DirectML to avoid the maintenance-mode dependency and unify on one GPU runtime.

---

## 2. HTTP-first LLM backend risking a design that doesn't transfer (D5)

**Verdict: real risk, partially mitigated.**

The `Normalizer` trait (`crates/core/src/normalize.rs:76-85`) is intentionally minimal: `normalize(&self, req: &NormalizeRequest) -> Result<NormalizeOutput, NormalizeError>`. No token streaming, no KV-cache hooks, no grammar-constraint surface. This is fine for the HTTP path. The problem is what the embedded path needs that this trait *can't express* but the design relies on:

- **Prefix KV caching** (research §3, trick #2): the plan is to snapshot the system-prompt state with `state_seq_get` / `kv_cache_seq_rm` in the embedded path. The HTTP path can't do this — it sends the full prompt every time. The prompt skeleton in the design (`docs/design.md:138`) puts cacheable prefix first, which is the right structure, but the M2 HTTP measurement will show *server-side* caching behavior (Ollama/llama-server may or may not cache), not the embedded path's snapshot approach. The latency number from M2 won't transfer cleanly to M3.

- **Anti-preamble grammar** (trick #5): the embedded path enforces a GBNF grammar forbidding "Here"/"Sure"/etc. The HTTP path can't — Ollama's API supports `grammar` but it's not universal across OpenAI-compatible servers. If the prompt alone suppresses preambles in testing, the grammar constraint may not get real exercise until M3.

- **N-gram speculation** (trick #3): llama-server provides this for free via `--spec-type ngram-*`. The embedded path needs a hand-rolled draft/verify loop. The HTTP path will benefit from server-side speculation; the embedded path won't unless they build it.

These aren't showstoppers. The design is honest that HTTP is a stepping stone to validate *prompt and validation logic* (`docs/design.md:150-151`). But the Normalizer trait needs one addition: an `on_token` callback (present in the research sketch at `docs/research/2026-09-24-llm-normalization.md:22` but absent from the actual trait at `crates/core/src/normalize.rs:76`). Without it, the embedded path can't stream tokens to the overlay for a thinking indicator, which matters for UX when the LLM takes 0.5-0.8s.

The trait also has no `cancel` method. See point 6 below.

---

## 3. Normalization validation (D4) — what the containment check misses

**Verdict: ambiguity in the primary check needs resolving.**

The validation is three checks: word containment against source, length ratio, language unchanged. Here's what each misses:

**Word containment — ambiguous direction.** The design says "word containment against the source drops below a threshold" but doesn't define the direction. If it means *fraction of source words found in output*: catches dropped content ("Turn left at the light" → "Turn at the light" loses "left") but misses hallucinated additions using common words. If it means *fraction of output words found in source*: catches "I went to the store" → "I went to the store and bought milk" (added words not in source), but misses dropped content. The research reference (`docs/research/2026-09-24-llm-normalization.md:47`) cites local-wisprflow using ~70% but doesn't specify direction. **You need both ratios.** The design must state this explicitly.

**Length ratio** (~1.5 bound): catches verbosity blow-ups (VoiceInk's 3.3k → 7.2k words). But won't catch a short hallucinated sentence appended to a long transcript — "The quarterly results show growth across all sectors" → "The quarterly results show growth across all sectors. I hope this helps!" — ratio ~1.15, passes.

**Language check**: catches translation but not subtle register shifts within the same language.

**What a cheap additional check would catch:** edit-distance ratio (Levenshtein / max length). Good transcripts have edit distance dominated by filler removal and punctuation changes, not word substitutions. An "answer" to a question typically adds novel words not near-matches of source words, inflating edit distance.

Also, the **few-shot examples in the prompt** are the real defense against question-answering, not validation. Validation is the last line. The design is correct to lean on the prompt structure first, but the containment ambiguity weakens the safety net.

---

## 4. Delayed-render clipboard receipt assumption (D8)

**Verdict: correct design, several under-acknowledged failure modes.**

The receipt = `WM_RENDERFORMAT` from the target after the paste chord. The design correctly counts reads only after the chord and restores only if the clipboard hasn't changed. But:

**App reads `CF_TEXT` instead of `CF_UNICODETEXT`.** The design says "put the text on the clipboard as delayed-rendered `CF_UNICODETEXT`." If that's the *only* format rendered, and the target requests `CF_TEXT` (some older apps, some cross-platform toolkits), Windows does *not* auto-synthesize `CF_TEXT` from delayed-rendered `CF_UNICODETEXT` — the clipboard owner is responsible for rendering every format it registers. The paste produces nothing, no receipt comes, and the strategy falls back to typing. The clipboard owner window should handle `CF_TEXT` rendering too. This is confirmable in Win32 docs; the research doesn't address it.

**Clipboard managers triggering spurious receipts.** Ditto, ClipboardFusion, Win+V history — they all trigger `WM_RENDERFORMAT` independently of the target app. If one reads *after* our chord, we get a receipt that wasn't the target app. The design doesn't address distinguishing the target's read from a clipboard manager's read. One mitigation: compare the requesting window's process ID to the foreground window's PID, but that's fragile (clipboard manager might be the foreground briefly). The research mentions clipboard history exclusion formats (`docs/research/2026-09-24-windows-input-output.md:34`), which should prevent Win+V from triggering renders, but third-party managers may ignore these. **Unverified risk.**

**The receipt might not arrive even on success.** Some apps (particularly those using the WinRT clipboard API or custom paste handlers) may read the clipboard through a mechanism that doesn't produce `WM_RENDERFORMAT` to the owner window. The 500ms fallback timeout covers this, but it means these apps always pay the timeout before typing fallback — doubling the insertion latency for them.

**RDP, Citrix, WSL:** the research correctly identifies these as broken for clipboard paste (`docs/research/2026-09-24-windows-input-output.md:35`). The fallback chain handles them. Good.

These failure modes don't make the design wrong — delayed-render with receipt is still the best approach — but the test matrix should explicitly include a clipboard manager and an app that requests `CF_TEXT` only (e.g., a WinForms RichTextBox).

---

## 5. Hotkey hook (D7)

**Verdict: the mechanics are right; the watchdog is underspecified.**

The `WH_KEYBOARD_LL` approach is the correct choice. The constraints listed (`docs/design.md:175-182`) match the Windows contract and the research. Specifically:

- "Do almost nothing in the callback" — correct, and the research quantifies the timeout: max 1000ms, default likely 500ms on Win10 1709+ (`docs/research/2026-09-24-windows-input-output.md:14`).
- "Ignore injected events" — correct, needed to prevent our own Ctrl+V from retriggering.
- "Swallow key-up if key-down was swallowed; drop auto-repeat" — correct.

**The watchdog mechanism is vague.** The design says "detects a silently removed hook and reinstalls it" but doesn't say how. The research suggests comparing `GetAsyncKeyState` against known hook state *outside the callback* (`docs/research/2026-09-24-windows-input-output.md:86`). That's safe. But there's a subtlety: when the hook is removed, the user may have released the key, so `GetAsyncKeyState` showing "not pressed" doesn't necessarily mean the hook is gone — it could mean the user released it. A more robust watchdog: a heartbeat counter incremented in the hook callback, checked by a timer every 2 seconds. If the counter hasn't changed but we expect it to (e.g., we're in Recording state and the key should still be held), the hook is likely gone. This needs a concrete design before M1.

**Right Ctrl as default** is a good choice. It avoids the Start-menu side effect of Win, the menu-bar activation of Alt, and is rarely used by apps. The research notes AltGr arrives as LCtrl+RAlt (not RCtrl), so no interference (`docs/research/2026-09-24-windows-input-output.md:21`). The config file allows changing it — essential for keyboards without a Right Ctrl (laptops, compact layouts).

**One gap:** the design doesn't mention the hook reinstatement race. Between detection and reinstallation, keystrokes are lost. The user may need to release and re-press the hotkey. Acnowledge, don't fix — it's inherent to the mechanism.

---

## 6. Threading (D10) and crate split

**Verdict: threading model is sound. Core traits have one real gap: no cancellation.**

The threading model (`docs/design.md:215-227`) is well-structured: hook, audio, UI, COM/UIA, and async runtime for the pipeline, with inference on blocking workers. This matches the research recommendations (`docs/research/2026-09-24-windows-input-output.md:85-90`) and avoids the classic deadlocks.

**Missing from core traits:**

- **No `Inserter` trait.** The design lists `Inserter` as a core trait (`docs/design.md:246`), but `crates/core/src/` has no `inserter.rs`. Only `stt.rs`, `normalize.rs`, `context.rs`, and `lib.rs`. The `FocusContext` type exists (`crates/core/src/context.rs`) but there's no trait for the insertion strategy chain. This should exist — it's the third major pluggable component.

- **No cancellation.** The state machine says "Cancelled reachable from any active state" (`docs/design.md:83`), but neither `SttEngine` nor `Normalizer` has a `cancel()` method. The `SttEngine::transcribe` takes `&mut self` and `pcm: &[f32]` — there's no way to interrupt a long transcription. ONNX Runtime sessions can be interrupted, whisper.cpp has abort callbacks, and llama.cpp has `llama_synchronize`. The traits need a `cancel(&mut self)` method, or the engine must be droppable mid-inference (which is unreliable with C++ FFI). This should be added before M1, not retrofitted.

- **`SttEngine::transcribe` takes `&mut self`** — fine for single-threaded use, but prevents concurrent segment transcription if D6's pre-transcription model ever wants parallelism. Not a v1 concern, but worth noting.

**Process isolation (D12) claim:** "the trait is designed so a process boundary can be inserted without changing callers." The current traits don't evidence this — they use synchronous `&mut self` with borrowed data. An IPC boundary would require owned data (no lifetimes), serialization, and async/blocking with timeouts. The trait *could* be wrapped in an IPC adapter, but it's not *designed* for it — there's no error variant for process death, no timeout parameter, and `NormalizeRequest` borrows `&str` from the caller. This is fine for v1, but the claim in the design doc overstates the current design.

---

## 7. Milestone order and wrong decisions

**Milestone order:** M0 → M1 → M2 → M3 → M4 is correct. Starting with measurement is essential given the research has no GPU latency data. The vertical slice gets the core loop working. Deferring embedded llama.cpp to M3 while proving the LLM logic over HTTP in M2 is pragmatic.

**What I'd reorder:** pull single-instance (named mutex) into M1. It's ~20 lines of Win32 and prevents the common "I launched it twice and now my hotkey is doubled" bug during development. Also, the `cancel` mechanism should be in M1 — if you can't cancel a transcription during development, every bad utterance means waiting or killing the process.

**No decision is outright wrong.** D2 (DirectML) is the closest to questionable, but only because of the maintenance-mode risk, not because it won't work. The design's own escape hatch — evaluating `transcribe-cpp` early — is the right hedge. If transcribe-cpp's Vulkan backend delivers equivalent latency, switch to it and drop ONNX Runtime entirely.

**One thing not in any decision that should be:** the `Inserter` trait is missing from the code. Since insertion has a strategy chain (delayed-render paste → typing → clipboard-only), this is a pluggable component like STT and normalization. It should be defined in `core` alongside the other traits, with a Windows implementation in `platform-windows`.

---

## Bottom line

The design is unusually thorough for a pre-implementation plan. The research is well-sourced, the rejected alternatives are documented with reasons, and the perishable sections are honestly marked. The main risks are: DirectML maintenance-mode exposure (mitigated by the transcribe-cpp evaluation path), the Normalizer trait lacking a token callback and cancellation mechanism that the embedded path will need, ambiguity in the containment-check direction that weakens the validation safety net, and a missing `Inserter` trait that the workspace layout claims exists. None of these are fatal; all are fixable before M1. The milestone order is right, and starting with measurement (M0) is the correct discipline for a latency-critical product with no existing GPU numbers.

<details><summary>investigation trace</summary>

- `read_file(path=docs/design.md)`
- `read_file(path=docs/research/2026-09-24-stt-engines.md)`
- `read_file(path=docs/research/2026-09-24-llm-normalization.md)`
- `read_file(path=docs/research/2026-09-24-windows-input-output.md)`
- `read_file(path=…research/2026-09-24-wispr-flow-and-prior-art.md)`
- `read_file(path=crates/core/src/lib.rs)`
- `read_file(path=crates/core/src/stt.rs)`
- `read_file(path=crates/core/src/normalize.rs)`
- `read_file(path=crates/core/src/context.rs)`
- `read_file(path=Cargo.toml)`
- `read_file(path=apps/whisper-local/src/main.rs)`
- `read_file(path=crates/stt/src/lib.rs)`
- `read_file(path=crates/normalize/src/lib.rs)`
- `read_file(path=crates/platform-windows/src/lib.rs)`
- `read_file(path=crates/audio/src/lib.rs)`
- `read_file(path=crates/core/Cargo.toml)`
- `read_file(path=tools/bench-stt/src/main.rs)`
- `read_file(path=tools/bench-normalize/src/main.rs)`
- `grep(pattern=Inserter)`
- `grep(pattern=cancel|Cancel|abort|Abort, glob=*.rs)`
- `grep(pattern=stream|Stream, path=crates/core/src)`
- `grep(pattern=Inserter|inserter|insertion|Insertion, path=crates)`
- `grep(pattern=trait, path=crates/core/src)`

</details>

---

## glm-5.2 — failure and operations lens  (`z-ai/glm-5.2`)
*20 tool calls · 162,119 in (63% cached) / 32,696 out (incl. 25,432 reasoning) · $0.2543 · 249.5s*

# Review of docs/design.md as a build plan

**Overall: this is a strong, unusually well-researched design.** The product decisions (D1, D6, D9, D11) and the rejected alternatives are sound and well-justified. The problems are concentrated in three places where the design's own safety/correctness claims rest on mechanisms that are weaker than stated: the normalization validation (D4), the paste "receipt" (D8), and the HTTP-first LLM path (D5). There is also a single point of failure in the GPU story for M0/M1 that the milestone order makes worse. None of these are reasons to start over; they are reasons to tighten four sections before M1.

I'll go in priority order.

---

## 1. D4 validation is order-blind and only checks *deletions*, not *additions* — the "never answer" claim is overstated

This is the most important issue because CLAUDE.md and D4 both make an absolute guarantee ("The LLM is a cleaner, not an assistant… Nothing dictated may be *answered*"), and the mechanism backing that guarantee does not deliver it.

`docs/design.md:139-142` says validation rejects when "word containment against the source drops below a threshold, the length ratio exceeds a bound, or the language changed." Word containment is a **set** metric. It verifies the LLM did not *drop* source words. It does not verify the LLM did not *add* an answer, because:

- **Appended answer reusing source words:** "what time is the meeting tomorrow" → "What time is the meeting tomorrow? I don't know." Every source word is present, length ratio ~1.2 (under any plausible 1.5 bound), same language. Passes. This is exactly the failure D4 claims to prevent.
- **Reorder that inverts meaning:** "send the report to Bob not Alice" → "send the report to Alice not Bob." Containment 1.0, length 1.0, same language. Passes. Meaning flipped.
- **Punctuation that flips meaning:** "Let's eat, Grandma" → "Let's eat Grandma." The LLM is *supposed* to change punctuation, so you cannot reject on punctuation delta. Same words. Passes.

The length-ratio bound is the only addition guard and it is coarse: a short appended answer ("No.") or a one-clause addition slips under 1.5. The research's own threshold is "~70% containment" (`2026-09-24-llm-normalization.md:47`), which is a *lower* bound on overlap — also a deletion guard, not an addition guard.

**Better cheap check that actually catches "it answered":** replace set-containment with an **order-aware alignment** — longest-common-subsequence or token Levenshtein over the source. Then flag any run of ≥ N consecutive *content* tokens in the output that has no match in the source (an inserted run), while *allowing* inserted punctuation and deleted fillers. That directly catches appended answers and reorders at ~100-token scale (cheap). Set-containment can stay as a second cheap signal, but it should not be the primary one. The research even sketches an `on_token` streaming hook (`2026-09-24-llm-normalization.md:22`) that the final trait dropped; an alignment check is the offline equivalent.

Also missing from the stated bounds: a **lower** length bound. Heavy legitimate cleanup (filler removal + self-correction) can shorten by >50% ("um uh so like you know I was thinking Tuesday no wait Wednesday" → "I was thinking Wednesday"). D4 only names an *upper* ratio. Without a tuned lower bound you will falsely reject good cleanups, and the research gives no number for it.

One more gap tied to this: **validation against the dictionary is not mentioned.** D4 validates against the *source transcript* only. A cleanup that lowercases "Anneliese Müller" to "anneliese müller" passes containment but violates the user's explicit dictionary spelling. The dictionary is a harder contract than the transcript; validate against it too.

---

## 2. D8 "receipt" is treated as singular and binary — the design reintroduces the exact Handy #502 bug it claims to fix

`docs/design.md:188-194` says "treat the target's render request as the receipt" and "restore the previous clipboard contents… shortly after the receipt." The research it leans on (`2026-09-24-windows-input-output.md:33`) describes Handy's actual implementation as **"restore 200 ms after the *last* read with an 8 s ceiling"** — not after the first receipt. That distinction is load-bearing.

The receipt (`WM_RENDERFORMAT`) fires whenever *anything* reads the clipboard via `GetClipboardData`, not only the target app on paste. Situations that fire a **false or premature receipt**:

- **Eager clipboard managers that ignore exclusion flags.** The design marks the text with `ExcludeClipboardContentFromMonitorProcessing` / `CanIncludeInClipboardHistory=0` (`design.md:192`, `windows-input-output.md:34`). Third-party managers like Ditto are documented to read the clipboard on every change regardless of those flags. Ditto reading between the paste chord and the target's actual paste fires a receipt. If you restore "shortly after the receipt," you restore *before* the target pastes → the target pastes the **old** clipboard. That is Handy bug #502 (`2026-09-24-wispr-flow-and-prior-art.md:52`), the bug this design exists to fix.
- **Apps that snapshot the clipboard on activation.** The design refocuses the target window before pasting (`design.md:196`). An app that reads the clipboard on `WM_ACTIVATE` (some IDEs, some terminals) fires a receipt at refocus, before the paste chord.
- **Terminal paste-warning dialogs.** conhost/Windows Terminal in some configs pop a paste confirmation. The render fires when the dialog reads the clipboard to show a preview; the actual paste happens after the user dismisses. Restore-on-first-receipt clobbers it.

The fix is to make the receipt **"last render + quiet window,"** explicitly: restore only after *N* ms with no further `WM_RENDERFORMAT` requests, capped by a ceiling, and only if the clipboard sequence number is unchanged since our write. The design says "only if nobody else changed the clipboard meanwhile" for the restore *condition* but does not specify last-read-vs-first-read for the *receipt*. Align D8 with the research's proven Handy impl (last read + 200 ms + 8 s ceiling), or the clipboard-restore-correctness story is no better than Handy's old broken path.

Specific apps the question asked about:

- **CF_TEXT vs CF_UNICODETEXT:** likely fine. We register only `CF_UNICODETEXT` for delayed render. When an app requests `CF_TEXT`, Windows *synthesizes* it from our rendered `CF_UNICODETEXT` and does **not** call our render proc for `CF_TEXT` — so the receipt still fires once, for `CF_UNICODETEXT`. (Unverified against current Windows behavior; worth a 10-minute test in M0.)
- **RDP/Citrix:** the research flags this (`windows-input-output.md:35`, `wispr-flow.md:31`). With clipboard redirection on, the local RDP client reads the clipboard to forward → receipt fires locally, but the actual insert is remote and timing-decoupled; a remote copy between local-receipt and remote-paste gets overwritten on restore. Without redirection, no receipt at all → fallback to `SendInput`, but `SendInput` into a fullscreen RDP client types *locally*, not into the remote session. This is a known-broken case; the "last resort: clipboard + tell user" covers it but D8 should call RDP out as "no reliable paste or type" rather than letting it silently fall through to a fallback that doesn't work remotely.

---

## 3. D5 HTTP-first does not validate the two things that make sub-second work, and ships the *least* safe configuration first

Building HTTP first to prove the **prompt and validation logic** (D5, `design.md:150-151`) is sound. What it does **not** validate, and the design doesn't say so:

- **Prefix KV-cache snapshotting.** D4 (`design.md:137-138`) places vocabulary/app-context/previous "after the fixed prefix so the prefix state can be cached," and D5 (`design.md:154`) says "snapshots the system-prompt state." That is an **embedded-only** capability (`llama-cpp-2`'s `state_seq_get`/`kv_cache_seq_rm`, `2026-09-24-llm-normalization.md:12`). Over OpenAI-compatible HTTP you cannot snapshot or force-cache the prefix; Ollama may evict the model (`keep_alive`) and re-prefill every call (`2026-09-24-llm-normalization.md:14`: "no state-snapshot control"). So M2's HTTP latency is a **pessimistic upper bound** (full 600-token prefill each call on a cold/unloaded server), not a predictor of the embedded number. The sub-second target cannot be validated on the HTTP path. The design should state this so nobody treats an M2 HTTP number as evidence the target is met.
- **Anti-preamble grammar.** D5 (`design.md:155`) and the research (`llm-normalization.md:45`) rely on a GBNF grammar forbidding leading "Here"/"Sure"/`"`/fences. **OpenAI-compatible HTTP has no grammar support.** Over HTTP the only anti-answer guard is prompt framing + the (leaky, per issue #1) validation. So M2 is the **weakest** safety configuration: no grammar, order-blind validation. The embedded path (M3) adds the grammar. 

Recommendation: explicitly mark the M2 HTTP path as **dev-machine-only and not safe to ship by default to end users**, precisely because it lacks the grammar guard that the embedded path relies on as defense-in-depth. The validation (issue #1) is not strong enough to stand alone.

A related robustness gap: D4 says validation rejection → rule fallback (`design.md:140`), but `NormalizeError` has `Unavailable`, `Request`, `Timeout` (`crates/core/src/normalize.rs:67-74`). The design does not state that **LLM backend failures also fall back to the rule pass** rather than surfacing as a user error. "Text is never silently lost" is satisfied by rule-fallback-on-any-LLM-failure, but the contract isn't written. A timeout that shows an error toast instead of inserting the rule-cleaned text is a regression. State it.

---

## 4. The GPU story for M0/M1 has a single point of failure, and the milestone order makes it worse

`design.md:279` and `§8` (`design.md:284-289`): the dev machine is RTX 5060 Ti (Blackwell, sm_120), **no CUDA toolkit, no Vulkan SDK (deferred to M3)**. So for M0 and M1 the **only** possible GPU acceleration is DirectML on Blackwell — and whether DirectML initializes on the 5060 Ti with the pinned `ort` is literally the first unverified item in §7 (`design.md:279`). The research calls DirectML "maintenance mode (new work in Windows ML)" (`2026-09-24-stt-engines.md:35`) and CUDA on Blackwell needs 12.8+/sm_120 (`2026-09-24-stt-engines.md:12`), which the dev hasn't installed.

So: if DirectML fails or is slow on the 5060 Ti, M0/M1 have **no GPU path at all** (CPU Parakeet is the Tier-2 story, ~0.33s/10s on a desktop i7 per `stt-engines.md:29`, slower on laptop — not the Tier-1 experience). The design's D2 hedge ("evaluate `transcribe-cpp` early," `design.md:121-122`) is the natural fallback, but `transcribe-cpp` is Vulkan, and the Vulkan SDK is deferred to M3.

Two concrete fixes:

1. **Install the Vulkan SDK now (move it left of M0),** so M0 can benchmark `transcribe-cpp`/Vulkan as a DirectML alternative on the same GPU. §8 deferring it to M3 collapses the fallback. The design's own "evaluate early" (`design.md:121`) wants Vulkan available at M0; the build-requirements section contradicts it.
2. **Make M0 a decision gate with a written fallback decision.** "If DirectML does not init or is >X ms on the 5060 Ti, D2 default becomes `transcribe-cpp`/Vulkan before M1." Right now M0 measures but the design commits to DirectML as default regardless of the result. A measurement with no bound decision is just a number.

Is DirectML a "trap" for a 2026 end-user ship? Not in the removal sense — it's supported, runs on any DX12 GPU, no CUDA install, which is its real value, and the probe-and-report design (`design.md:113-114`) is correct. The trap is narrower: it's **maintained-but-not-advancing**, performance per ORT-version/GPU-driver combo is unverified, and the Windows ML / NPU story (Foundry Local, `llm-normalization.md:18`) is where Microsoft is investing. DirectML is a reasonable *default*; it is not a safe *only path*. Treat it as one of two and keep Vulkan live from M0.

On "single ggml/Vulkan backend for both STT and LLM vs. mixing ONNX Runtime + llama.cpp": architecturally a single Vulkan backend (transcribe-cpp for STT + llama.cpp Vulkan for LLM) is **cleaner** for two reasons the mixed path gets wrong — (a) one GPU backend means one set of device-lost/crash handling and one driver surface, and (b) D12 process isolation is simpler when both engines are the same family. The cost is that `transcribe-cpp` is "young" (`2026-09-24-stt-engines.md:38`, `design.md:122`). The design's "evaluate early" is the right call; my only addition is that the evaluation needs to happen at M0 with Vulkan installed, not hand-waved, and the result should be allowed to *replace* D2's ONNX path, not just supplement it.

---

## 5. `ort` version unification across STT, VAD, and the punctuation model is unaddressed — build/packaging landmine

The design pulls `ort` 2.0.0-rc.13 (an RC) via `parakeet-rs` for STT (`2026-09-24-stt-engines.md:31-33`), Silero VAD via `voice_activity_detector` 0.2.1 which the research flags as pinning an **older `ort`** ("version conflict risk," `stt-engines.md:47`), and a 47-language punctuation ONNX model via `ort` in Tier-2 (`llm-normalization.md:84`). Three `ort` consumers.

Two `ort` versions in one process with prebuilt binaries risks symbol/build conflicts, or — if Cargo resolves to two copies — **two ONNX Runtime native binaries shipped**, each hundreds of MB, which defeats the "no CUDA/heavy redistribution" goal. The design never names this. Resolve it now: pick the VAD path that uses the *same* `ort` (sherpa-onnx VAD, or whisper.cpp's built-in Silero `WhisperVadContext` since whisper.cpp is already an M3 dep — `stt-engines.md:11,47`), and pin one `ort` workspace-wide. Otherwise M3 packaging will hit this late.

Also: `ort` 2.0.0-rc.13 is a **release candidate** for a shipping product. CLAUDE.md says pin exact versions for non-semver crates; `ort` is pre-1.0 so every minor breaks. Acceptable for M0/M1; call out that `ort` 2.0 **stable** is a hard dependency for the shippable M3, and budget for the API churn.

---

## 6. D12 crash isolation is deferred to M4 — that ships the crash class it exists to fix

`design.md:236-239`: "v1 runs in-process behind the engine trait." D12 is the fix for Handy's **worst** Windows bugs — Vulkan device-lost crashes taking the whole app down (`2026-09-24-wispr-flow-and-prior-art.md:55`: #2047, #1775, #1841; `2026-09-24-windows-input-output.md` manual test matrix calls Vulkan crashes Handy's main Windows problem). M3 ships self-contained with **embedded llama.cpp Vulkan in-process**. So the first build that real end users run (self-contained, no Ollama) is the one most likely to hit exactly the crash class D12 exists to fix, on the widest variety of GPUs/drivers.

The trait *is* designed for a process boundary later (D12), which is good, but retrofitting a child-process wrapper at M4 — after M1-M3 are built around in-process `&mut self` synchronous calls — is more work than doing a crude sidecar at M3. Recommend: move a **minimal** child-process STT/LLM wrapper to M3 (even if the IPC is just stdin/stdout JSON over a pipe and the kill-on-crash path is crude), behind a flag, defaulting on for the Vulkan engines. Doing it late means reworking the call sites and the cancellation story together.

---

## 7. The crate split is missing the traits the state machine needs to be built or tested in core

`design.md:246` claims `core` holds "state machine, traits (SttEngine, Normalizer, Inserter)." Grep confirms `SttEngine` and `Normalizer` exist in `crates/core/src/`; **`Inserter` does not exist** (no match in `crates/core/src/`). More importantly, the state machine — which §3 (`design.md:82-85`) makes the central coordinator — needs to drive platform actions: start/stop capture, beep, insert, show overlay state. None of the **platform-capability traits** (Recorder, Inserter, Notifier) exist in core. So the state machine cannot be written or unit-tested in the platform-neutral crate; it would have to live in `platform-windows`, which contradicts D10's threading model and §5's "core compiles on every platform and has tests."

`SttEngine` and `Normalizer` are the *engine* contracts; the orchestration contracts are absent. Adding them later means the state machine is blocked until they exist. Recommend defining in M0/M1:

- `Inserter` in core, with `FocusContext` + text → `InsertResult { inserted | clipboard_with_reason | failed }`, so the **strategy chain** (pick chord, decide fallback, receipt-wait) is pure logic in core and testable, with only the actual Win32 calls (`SetClipboardData`, `SendInput`, `WM_RENDERFORMAT` handling) in `platform-windows`. This is where the D8 correctness work from issue #2 needs to be testable without a real window.
- `Recorder` (start/stop, pre-roll) and `Notifier` (overlay state, beep) as minimal traits, so the state machine is hermetic.

---

## 8. No cancellation in the engine/normalizer traits — needed for Escape and for D12

`SttEngine::transcribe` (`crates/core/src/stt.rs:92`) and `Normalizer::normalize` (`crates/core/src/normalize.rs:85`) are synchronous blocking calls with **no cancellation token, deadline, or abort handle.** §3 (`design.md:82-85`) makes `Cancelled` reachable from any active state, and D12 wants process isolation (kill child = cancel). But there is no way to cancel an in-flight `transcribe`/`normalize` from the state machine — they're blocked in a C++ call.

For STT this is arguably fine: §3 says a press during Transcribing queues the *next* recording, so the in-flight transcription is allowed to finish and be discarded. For the **LLM normalize** (0.4–0.8s on GPU, multi-second on CPU per `llm-normalization.md:36`), an Escape should cancel, and the embedded path's grammar/`max_tokens` cap is the only thing bounding a runaway. For D12, a child process needs a kill handle, which the trait can't express.

The research's own trait sketch included `on_token` (`llm-normalization.md:22`) — the streaming callback that is also the cancellation hook. The final trait dropped it. Recommend adding an optional `CancelToken`/`AbortHandle` parameter (even if M1 impls ignore it) so the D12 child-kill path and the Escape-during-LLM path don't require a trait reshape. Adding it later is additive for impls but **breaking for the state machine's call sites**, so do it before the state machine is written.

---

## 9. `Normalizer: Send + Sync` with `&self` is the wrong shape for an embedded LLM; `SttEngine` got it right

`SttEngine::transcribe(&mut self, …)` (`stt.rs:92`) is correct — an engine holds mutable scratch/KV state. `Normalizer::normalize(&self, …)` with `Send + Sync` (`normalize.rs:76,85`) forces the `LlamaCpp` impl to wrap its resident context in a `Mutex` (interior mutability), since a llama.cpp context is not `Sync`-safe to share. D10 says only one utterance is in flight, so a `Mutex` is functionally fine, but `&self + Sync + Mutex` is an awkward shape vs just `&mut self` like the STT side. The inconsistency suggests the trait was written for the HTTP impl (stateless, `&self` is natural) and not revisited for the embedded one. Pick one: either make `Normalizer` `&mut self` (matches `SttEngine`, drops `Sync`), or keep `Sync` and document the `Mutex`. Minor, but it's a refactor at M3 if left.

---

## 10. Right Ctrl as default has two real UX costs the design doesn't acknowledge

`design.md:43-45` chooses Right Ctrl because "a modifier-only key avoids the Start-menu and menu-bar side effects." True, but:

- **Swallowing RCtrl consumes the modifier while held.** Because the key-down is swallowed, the OS never sees Ctrl-down, so any Ctrl+chord the user types *while talking* arrives without Ctrl. You cannot hold-to-talk and press Ctrl+C mid-utterance. Wispr uses Ctrl+Win (a *chord*) partly for this reason (`wispr-flow.md:9-10`). This is fine if the user only speaks while holding, but the design doesn't note the trade-off.
- **RCtrl doesn't exist on some compact/60% keyboards** (which often have only a left Ctrl). A default that's absent on a non-trivial slice of hardware is a poor default.

CapsLock is the safer default the design mentions only as an option (`2026-09-24-windows-input-output.md:21`): present on all keyboards, swallowing it just suppresses the toggle (which most power users remap away anyway), and it's not a modifier anyone chords while typing. Consider CapsLock as the default and RCtrl as the documented alternative. Not a correctness issue; a default-choice issue.

---

## 11. Same-key double-tap toggle adds a 250 ms ambiguity cost to every hold-to-talk

`design.md:46` and `windows-input-output.md:24`: double-tap within ~250 ms locks hands-free; longer hold is PTT. To disambiguate tap from hold you must wait ~250 ms after key-down to know it's a hold — which adds 250 ms to **every** PTT start, or you start capture optimistically and discard if it resolves to a toggle. The design doesn't say which. Wispr uses a *separate* toggle key (Ctrl+Win+Space, `wispr-flow.md:9`) to avoid exactly this. Recommend: start capture on key-down (use the pre-roll), and if the key resolves to a tap-pair (toggle), discard the captured audio. That preserves the "instantaneous" feel and wastes only the pre-roll. State it, or the latency budget silently eats 250 ms.

---

## 12. No recording max-duration timeout; lock-screen/UAC-during-record is unspecified

The research's manual test matrix includes "lock screen / UAC during recording; sleep/resume" (`windows-input-output.md:92`). If UAC pops while the key is held, the hook stops seeing events (UIPI, `design.md:181-182`), so **key-up never arrives** and recording hangs. §3 makes `Cancelled` reachable from any active state but specifies no **max-record timeout** that auto-stops. Wispr auto-stops very long dictations (`wispr-flow.md:12`). Add: a configurable max-record duration that auto-stops and inserts; and the hook watchdog should detect "hook died mid-record" (the `design.md:180` watchdog detects removal, but not "we're still hooked but stopped receiving events because UIPI"). The watchdog comparing `GetAsyncKeyState` to hook state (`windows-input-output.md:86`) must run on a non-hook thread (the callback must not call `GetAsyncKeyState`, `design.md:177`) — the design should say so explicitly.

---

## 13. Always-on microphone (pre-roll) is a continuous-capture privacy decision not surfaced

D10 (`design.md:218`) keeps the capture stream open while idle with a pre-roll buffer for first-syllable latency. That is a **continuously open microphone** — the Windows mic-in-use indicator stays on permanently. "Nothing leaves the machine" handles exfiltration, but an always-on mic is itself a privacy signal worth a decision, especially for a tool whose entire pitch is local/private. Either make pre-roll opt-in, or make the buffer short and document the always-on indicator. Not raised in the design or the research as a trade-off.

---

## 14. Two smaller data-loss edges

- **Aborted utterance's clipboard text can be overwritten by the next dictation.** D8 aborts on focus change and puts text on the clipboard + toast (`design.md:55-57`). The next utterance inserts via clipboard paste, overwriting that clipboard text. "Paste last transcript" (`design.md:50`) is the recovery net, but only if "last transcript" is retained *per utterance* and not overwritten by the next successful insert. The design doesn't say. If "last" = most recent successful, the aborted utterance is unrecoverable after the next insert. Minor, but it's a "text is never silently lost" edge.
- **`previous: Option<&'a str>`** (`normalize.rs:47`) is the post-normalization previous sentence. Fine across the in-process path. Across D12 process isolation it's passed in the request, so it survives. No issue — noted only because I checked.

---

## Milestone order (question 6)

Reorders I'd make, in priority:

1. **Vulkan SDK + `transcribe-cpp` spike moves into M0** (issue #4). It's the DirectML fallback and the "evaluate early" item; deferring the SDK to M3 strands it.
2. **D12 child-process wrapper moves to M3, not M4** (issue #6). M3 is the first self-contained end-user build and the first in-process-Vulkan-crash-exposed build.
3. **Split M3** into 3a (embedded llama.cpp + model downloader + first-run) and 3b (Whisper Vulkan second engine + run-at-startup + single-instance). The second STT engine is the lowest-value item in M3 — Parakeet covers 25 languages; Whisper is for the long tail + vocab prompting (`design.md:108-110`). It can slip to M4 without hurting the core product; the downloader and embedded LLM cannot.
4. **M0 is a decision gate**, not just a measurement (issue #4): write the fallback decision *before* M0 so the result can change D2's default.

M0→M1→M2→M3-Measure-first is otherwise the right order. M2 (HTTP LLM) is correctly before M3 (embedded) for proving prompt/validation, with the caveat from issue #3 that it validates *correctness*, not *latency* or *grammar safety*.

---

## Things that are right and don't need changes

- D1 (no WebView), D6 (no live streaming; pre-transcribe closed VAD segments), D9 (minimal context, no screenshots), D11 (models in `%LOCALAPPDATA%`, not Roaming — the research correctly flags Handy's Roaming choice as wrong, `windows-input-output.md:79`).
- D7's hook discipline (do-almost-nothing callback, `try_send`, no `GetAsyncKeyState` in proc, injected-event filtering, watchdog) is correct and matches the Windows contract (`windows-input-output.md:14,86`).
- D3 (VAD in the app, not the engine; drop no-speech to fight Whisper hallucination) is the right call and well-supported (`stt-engines.md:25,48`).
- The `Backend` enum + "report which backend actually loaded, never assume" (`stt.rs:12-20`, `design.md:113-114`) is exactly right given `ort`'s silent CPU fallback (`stt-engines.md:33`).
- Threading model D10 (hook thread never touches the async runtime; inference never on UI/hook; UIA on a dedicated MTA thread with timeout) matches the research's recommended design (`windows-input-output.md:84-90`).

---

## Bottom line

The plan is sound and I'd build against it after four fixes: (1) replace order-blind word-containment in D4 with an order-aware alignment check that flags *inserted* token runs — the "never answer" guarantee currently rests on a check that only catches deletions; (2) make the D8 clipboard receipt "restore after the *last* render + quiet window + ceiling," not "shortly after the receipt," or the design reintroduces Handy #502 via eager clipboard managers; (3) move the Vulkan SDK and a ``transcribe-cpp` Vulkan spike into M0 so DirectML on Blackwell is not the only GPU path, and make M0 a decision gate with a written fallback to Vulkan before M1; (4) add the `Inserter`/`Recorder`/`Notifier` traits and a `CancelToken` to `crates/core/src/` before the state machine is written, otherwise the platform-neutral state machine in §3/D10 can't be built or unit-tested in core and the D12 child-process kill path requires a later trait reshape. Beyond those, the long-tail items worth doing soon are D12 process isolation pulled from M4 into M3 (the first end-user build is also the first in-process-Vulkan-crash-exposed build), workspace-wide `ort` version unification (three `ort` consumers risk shipping two ONNX Runtime binaries), and an explicit "LLM backend failure → rule fallback" contract so a timeout or unavailable Ollama never surfaces as an error toast instead of inserting the rule-cleaned text.

<details><summary>investigation trace</summary>

- `read_file(path=docs/design.md)`
- `read_file(path=crates/core/src/lib.rs)`
- `read_file(path=crates/core/src/stt.rs)`
- `read_file(path=crates/core/src/normalize.rs)`
- `read_file(path=crates/core/src/context.rs)`
- `read_file(path=docs/research/2026-09-24-stt-engines.md)`
- `read_file(path=docs/research/2026-09-24-windows-input-output.md)`
- `read_file(path=docs/research/2026-09-24-llm-normalization.md)`
- `read_file(path=…research/2026-09-24-wispr-flow-and-prior-art.md)`
- `read_file(path=Cargo.toml)`
- `read_file(path=crates/core/Cargo.toml)`
- `read_file(path=crates/stt/src/lib.rs)`
- `read_file(path=crates/normalize/src/lib.rs)`
- `read_file(path=crates/audio/src/lib.rs)`
- `read_file(path=crates/platform-windows/src/lib.rs)`
- `read_file(path=apps/whisper-local/src/main.rs)`
- `grep(pattern=Inserter|Insert|Cancel|cancel|stream|Stream, path=crates/core/src)`
- `grep(pattern=Inserter, path=docs/design.md)`
- `grep(path=crates/core/src, pattern=Serialize|Deserialize|derive)`
- `(hit token ceiling — continuing 1/3)`

</details>

---

## gpt-5.6-luna-pro — prior art and fit lens  (`openai/gpt-5.6-luna-pro`)
*13 tool calls · 159,600 in (70% cached) / 16,469 out (incl. 4,797 reasoning) · $0.0315 · 87.6s*

## Conclusion

The plan is directionally sound, but it is not ready to build against unchanged. The largest problems are the insertion “receipt” being treated as proof of successful insertion, validation being too weak to enforce the “never answer” guarantee, and deferring the embedded inference path until M3 despite making latency and caching central design requirements.

## 1. Critical: `WM_RENDERFORMAT` is not a reliable insertion receipt

D8 overstates what delayed rendering proves. A `WM_RENDERFORMAT` means that some clipboard consumer requested the delayed data; it does not prove that the intended foreground control inserted it successfully. A clipboard manager, history service, RDP layer, or another process can request the format. Conversely, an application may have already cached clipboard data, use a different clipboard path, or read `CF_TEXT` rather than `CF_UNICODETEXT`.

The design currently says to restore the clipboard “shortly after the receipt” and use the receipt as confirmation: `docs/design.md:187-203`. That creates two failure modes:

- The intended application has not finished consuming the data when it is restored.
- A third-party reader generates the receipt, the app does not paste, and the code suppresses the typing fallback because it believes insertion succeeded.

The research itself lists clipboard managers, RDP/Citrix, terminal-specific paste behavior, and applications that regress on synthetic paste: `docs/research/2026-09-24-windows-input-output.md:31-45`. Those are not merely test cases; they directly invalidate the receipt assumption.

A safer design is:

- Treat `WM_RENDERFORMAT` as “clipboard data was requested,” not “text was inserted.”
- Keep the delayed clipboard owner alive until the paste transaction is explicitly complete or times out.
- Restore only after a conservative grace period and sequence-number check, with the known risk documented.
- Do not automatically type after a render request unless the application is on a tested/known-safe profile. Typing after a successful paste can duplicate text.
- Make insertion outcomes distinguish `NotRequested`, `Requested`, `ClipboardChanged`, and `Unknown`; do not collapse them into success/failure.

There is also an implementation gap: the requested full clipboard save/restore is materially harder than saving text. The research notes that arbitrary formats, delayed-rendered data, GDI handles, and large Excel-like data need special handling: `docs/research/2026-09-24-windows-input-output.md:59-71`. This should be an explicit bounded best-effort operation, not an assumed atomic restore.

UIPI is another hard failure path. `SendInput` fails silently against higher-integrity targets, and clipboard behavior across RDP/Citrix is not guaranteed: `docs/research/2026-09-24-windows-input-output.md:29-35`. The pre-check helps, but focus or integrity can change between the check and injection.

## 2. Critical: the validation does not enforce “never answer”

Word containment and length ratio are useful anomaly filters, but they do not establish that the output is a cleaned version of the dictation.

They miss, among other cases:

- An answer made entirely from words present in the transcript.
- Reordering that changes meaning.
- Dropping “not”, “never”, names, quantities, or other critical tokens while retaining enough words to pass the threshold.
- Duplicating source words into a plausible but longer answer.
- A response to a question that repeats some source terms.
- Translation or paraphrase that preserves some words but changes the intended content.
- Correction errors where the rule pass intentionally removes the first phrase.

The current guarantee is too strong: `docs/design.md:139-142` says the LLM “can therefore never turn a dictated question into an answer.” The implementation does not exist yet, so this is unverified operationally, but the stated checks cannot provide that guarantee.

Use a validation model with explicit allowed transformations:

- Token multiset accounting against the raw transcript, with weighted protection for numbers, negations, names, URLs, code identifiers, and dictionary entries.
- A correction-aware comparison against the rule-pass output, not only the raw source.
- Rejection on new content words, duplicated spans, changed numbers, changed negation, or unexpected language/script.
- Maximum edit distance and output/input token bounds.
- Explicit preservation of interrogative structure where appropriate.
- A structured output format with only one text field and no explanation. Grammar or JSON constraints reduce preambles, but they do not prove semantic safety.

The rule pass should produce protected spans and a normalized representation that validation can use. The current `NormalizeOutput` exposes only text and provenance: `crates/core/src/normalize.rs:59-64`; there is no representation of deletions, corrections, protected tokens, or validation diagnostics.

Also, `Provenance::Llm` currently allows an LLM result without exposing whether validation passed and why: `crates/core/src/normalize.rs:50-57`. That makes later auditing and debugging harder.

## 3. High: the latency target conflicts with the proposed LLM budget

The stated target is under one second from release to insertion: `docs/design.md:21-23`. The research estimates 0.5–0.8 seconds for a 3–4B model on a 4090-class GPU for roughly 100 output tokens: `docs/research/2026-09-24-llm-normalization.md:5-9`, `docs/research/2026-09-24-llm-normalization.md:38-49`, `docs/research/2026-09-24-llm-normalization.md:92-95`.

That leaves almost no budget for:

- Tail transcription.
- VAD/stitching.
- Prompt construction.
- HTTP or embedded scheduling.
- Validation.
- Clipboard ownership and paste.
- GPU contention between STT and LLM.

The development GPU is an RTX 5060 Ti, not the 4090-class reference used for the estimate. The document explicitly admits that single-utterance GPU timings are not measured: `docs/design.md:275-281` and `docs/research/2026-09-24-stt-engines.md:5-7`.

The plan should define separate latency budgets and percentile targets, for example P50/P95 for 3-second and 10-second utterances, and should make the LLM optional whenever it cannot meet the deadline. “Under one second” should not be accepted based on aggregate throughput or real-time factor.

## 4. High: DirectML is a reasonable default, but not a safe assumption

DirectML is a sensible first GPU backend for a Windows consumer application that must avoid a CUDA toolkit dependency. The research supports its broad DX12 hardware coverage, while also explicitly describing it as being in maintenance mode: `docs/research/2026-09-24-stt-engines.md:27-35`.

The risk is not that DirectML is inherently a trap. The risk is treating it as a stable, universally performant production backend:

- Operator support and performance depend on the exact ONNX model and ONNX Runtime version.
- Provider initialization and fallback behavior must be tested on actual NVIDIA, AMD, and Intel systems.
- Silent CPU fallback directly undermines the latency target. The design recognizes this risk at `docs/design.md:112-114`, but “report the backend” is not enough; the default should fail clearly or switch policy when GPU execution was requested but unavailable.
- Windows ML / newer EP APIs may become the preferred Microsoft path, but the research does not establish that they are a drop-in replacement for this STT model. The Foundry Local material is specifically about local generative models and constrained ONNX catalogs: `docs/research/2026-09-24-llm-normalization.md:18-20`. It should not drive the current STT design without a concrete model and packaging test.

Keep Parakeet through ONNX Runtime/DirectML for the first spike, but add an explicit backend policy such as `RequireGpu`, `PreferGpu`, or `CpuOnly`. Do not silently turn a GPU product into a CPU product.

A single ggml/Vulkan backend is attractive for packaging and crash isolation, but `transcribe-cpp` is identified as young and its model support is still something to evaluate: `docs/research/2026-09-24-stt-engines.md:37-38`. It is not yet a stronger basis for the product than the established Parakeet ONNX path. Benchmark it early; do not unify merely to reduce the number of libraries.

## 5. High: HTTP-first is acceptable, but only if the prompt contract is backend-neutral

Using Ollama for the vertical slice is reasonable. It gets end-to-end product behavior running before taking on llama.cpp build and packaging work. The design correctly identifies the main tradeoff: HTTP cannot directly control prefix KV state or embedded context snapshots: `docs/research/2026-09-24-llm-normalization.md:11-16`.

The danger is allowing the HTTP implementation to define the semantics. Ollama and an embedded llama.cpp backend can differ in:

- Chat template and role formatting.
- Tokenizer behavior.
- Stop conditions.
- Thinking/reasoning modes.
- Grammar or JSON-schema support.
- Context reuse and eviction.
- Maximum output handling.
- Timeout and cancellation behavior.

Define a backend-independent normalization contract now:

- Exact prompt serialization or a versioned prompt schema.
- Output-only text semantics.
- Maximum output tokens.
- Stop behavior.
- No-thinking requirement.
- Validation performed outside the backend.
- Deadline and cancellation behavior.
- Model/backend capability reporting.

Then make the HTTP implementation deliberately emulate the embedded path, even if it is less convenient. Build a minimal embedded smoke test before M2 is considered complete. Deferring llama.cpp to M3 risks discovering that the prompt, token budget, or latency assumptions do not transfer after the application architecture is established.

The “same Vulkan backend as Whisper” argument is also weaker than it appears. `llama.cpp` and `whisper.cpp` may share a graphics API while still having separate build, model, memory, and failure behavior. One Vulkan dependency is not one operational backend.

## 6. High: the core traits do not yet support the stated pipeline

The current traits are clean stubs, but they are too synchronous and too minimal for the design described.

`SttEngine` takes `&mut self` and exposes only one-shot transcription: `crates/core/src/stt.rs:85-92`. That is workable for a single serialized request, but D6 and D10 require:

- Segment transcription while recording.
- Cancellation when Escape is pressed.
- Queuing the next recording.
- Ordering and stitching of overlapping or completed segments.
- Deadlines and stale-result rejection.
- Potential process-boundary execution later.

There is no cancellation token, request ID, deadline, or utterance/segment identity in the contract. Adding a child-process proxy later may preserve the method names but still require changing every caller to handle cancellation, crashes, restart, and stale responses.

`Normalizer` has the same issue: `crates/core/src/normalize.rs:76-85` has no cancellation, deadline, capabilities, or request ID. A blocked HTTP call or embedded decode cannot be reliably cancelled by the state machine.

At minimum, add concepts equivalent to:

- `UtteranceId` and `SegmentId`.
- Cancellation/deadline in request options.
- Backend capabilities including maximum context, structured output, streaming/partial support, and whether cancellation is supported.
- A result carrying backend/model identity and timing for each stage.
- Explicit failure categories for timeout, cancellation, backend crash, malformed output, and validation rejection.

The process boundary needs a protocol, not just a trait-compatible wrapper. Decide how model startup, health, crash restart, request cancellation, model version, and shared-memory/audio transfer work. This is especially relevant because D12 exists specifically to handle GPU device-loss crashes: `docs/design.md:235-239`.

There is also a concrete mismatch between the design and the repository:

- The design says `core` contains configuration, state machine, and `Inserter`: `docs/design.md:241-250`.
- The actual `core` exports only `context`, `normalize`, and `stt`: `crates/core/src/lib.rs:7-9`.
- No `Inserter` trait or state-machine module exists.
- `platform-windows` is only a placeholder: `crates/platform-windows/src/lib.rs:1-3`.
- The planned combined `tools/bench` does not exist; the workspace currently lists separate `bench-stt` and `bench-normalize`: `Cargo.toml:4-12`.

That is fine for a stub repository, but the plan should not describe these as settled integration boundaries until their ownership and APIs exist.

## 7. Hotkey design is mostly right, with important edge cases

A dedicated `WH_KEYBOARD_LL` thread with a message loop is the correct mechanism for key-up, modifier-only keys, and swallowing input. The callback restrictions and injected-event filtering are correctly called out: `docs/design.md:171-185`.

The missing design details are:

- Hook installation failure and reinstall races.
- What happens to the swallowed-key state during hook removal/reinstallation.
- Whether a watchdog can distinguish a removed hook from a deliberately idle hook.
- Shutdown ordering: stop event production, remove the hook on its owning thread, then join it.
- Handling the hook thread dying unexpectedly.
- Configuration changes while the callback is running.
- Multiple physical keyboards and scan-code versus virtual-key matching.
- Lock screen, secure desktop, UAC, sleep/resume, and session changes.

The research’s watchdog proposal is only a recommendation, not a demonstrated mechanism: `docs/research/2026-09-24-windows-input-output.md:84-90`. It needs a test that proves recovery without installing duplicate hooks or leaving Right Ctrl permanently swallowed.

Right Ctrl is a reasonable initial default, but “rarely conflicts” is not enough to make it universal. It conflicts with applications using Right Ctrl, accessibility layouts, remote desktop mappings, and AltGr-related keyboard behavior. The research itself labels AltGr behavior as unverified: `docs/research/2026-09-24-windows-input-output.md:21`. Make the default configurable on first launch or provide a reliable conflict-detection and rebind path.

## 8. Milestone order should change

I would reorder the milestones as follows:

1. **M0: infrastructure and measured spikes**
   - DirectML initialization and explicit CPU/GPU policy.
   - Real Parakeet latency on the target GPU.
   - Basic delayed clipboard owner and insertion matrix.
   - Hook install/remove/watchdog behavior.
   - HTTP and embedded normalization prompt/output contract.
2. **M1: rules-only vertical slice**
   - Hold-to-talk through insertion into Notepad, VS Code, browser, and Terminal.
   - No LLM dependency.
   - Prove focus capture, cancellation, clipboard failure handling, and visible fallback.
3. **M2: VAD and pipeline concurrency**
   - Pre-transcription, segment IDs, ordering, cancellation, and tail latency.
   - Measure whether it actually improves release latency before adding more product behavior.
4. **M3: normalization quality**
   - HTTP backend, validation corpus, adversarial “answer the question” tests, per-app styles, and dictionary behavior.
5. **M4: embedded backend and packaging**
   - llama.cpp on the actual target GPU, model download, warm-up, memory use, prompt-cache behavior, and fallback policy.
   - The embedded smoke test should happen earlier than this if it is the shipping backend.
6. **Later: Whisper, child-process isolation, CPU tier, installer and polish.**

In particular, the current plan puts VAD pre-transcription in M2 even though it is part of the latency-critical pipeline shown in `docs/design.md:61-79`, and puts the intended shipping LLM in M3 after the HTTP design has already been validated. Both should be risk-reduced earlier.

## Decisions I would change outright

1. **Change D8’s terminology and success semantics.** A render request is not an insertion receipt.
2. **Remove the absolute D4 guarantee.** Validation can make answering unlikely and reject many failures; it cannot guarantee that an LLM never answered. Use stronger structural checks and adversarial tests, and describe the remaining risk accurately.
3. **Add cancellation, deadlines, request IDs, and capability reporting to the core contracts before implementing engines.**
4. **Add an earlier embedded-backend spike.** HTTP-first is acceptable; embedded-last is not, given that embedded llama.cpp is the shipping path and prefix caching is part of the latency design.
5. **Make GPU fallback explicit.** Do not silently fall back to CPU when the user selected or the product requires GPU execution.

The DirectML default itself is not wrong. The current evidence supports it as the lowest-packaging-risk first implementation, while the newer Windows ML story is not sufficiently established here to justify switching. The correct next step is an actual initialization and latency spike, not an architectural commitment to Windows ML or to a unified ggml backend.

## Bottom line

Proceed with Parakeet through ONNX Runtime/DirectML and an HTTP normalizer for the first slice, but revise the contracts and failure semantics first. The insertion receipt, validation guarantee, deferred embedded-backend testing, and lack of cancellation/deadline/request identity are substantial correctness and integration risks, not polish items.

<details><summary>investigation trace</summary>

- `read_file(path=docs/design.md, offset=1, limit=400)`
- `read_file(path=crates/core/src/stt.rs, offset=1, limit=240)`
- `read_file(path=crates/core/src/normalize.rs, offset=1, limit=240)`
- `read_file(path=crates/core/src/context.rs, offset=1, limit=200)`
- `read_file(path=crates/core/src/lib.rs, offset=1, limit=160)`
- `read_file(path=docs/research/2026-09-24-windows-input-output.md, offset=1, limit=300)`
- `read_file(path=docs/research/2026-09-24-stt-engines.md, offset=1, limit=280)`
- `read_file(path=docs/research/2026-09-24-llm-normalization.md, offset=1, limit=280)`
- `read_file(path=Cargo.toml, offset=1, limit=200)`
- `read_file(path=crates/normalize/src/lib.rs, offset=1, limit=120)`
- `read_file(path=crates/stt/src/lib.rs, offset=1, limit=120)`
- `read_file(path=crates/platform-windows/src/lib.rs, offset=1, limit=120)`
- `read_file(path=apps/whisper-local/src/main.rs, offset=1, limit=120)`

</details>

---

