# Wispr Flow and open-source dictation apps (research, 2026-09-24)

Perishable: this names products, repos, versions and issue numbers as of the date above.
Re-verify anything here before relying on it after 2026-12.

## Part 1: What Wispr Flow does

### How you use it
- **Default hotkeys:** hold **Ctrl+Win** on Windows and **Fn** on Mac. Hold to talk, release to stop. Toggle mode ("hands-free") starts with **Ctrl+Win+Space** or by double-tapping the hold key within 0.5 s. [first dictation](https://docs.wisprflow.ai/articles/6409258247-starting-your-first-dictation), [hands-free](https://docs.wisprflow.ai/articles/6391241694-use-flow-hands-free)
- **Win key side effect:** holding Ctrl+Win stops the Start menu from opening. Win alone still works. [shortcuts](https://docs.wisprflow.ai/articles/2612050838-supported-unsupported-keyboard-hotkey-shortcuts)
- **Recording indicator:** a small floating "Flow bar" with moving white bars.
- **End of speech:** no automatic stop on silence. Releasing the key (or pressing again in toggle mode) ends the dictation; transcription and cleanup happen after. Very long dictations are stopped automatically. [failures](https://docs.wisprflow.ai/articles/4984532368-fix-taking-longer-than-usual-and-transcription-errors)
- **How text gets in:** Flow "briefly uses the clipboard to insert text, then restores its previous contents". It sends **Shift+Insert** in editors like Cursor. "Paste last transcript" (Alt+Shift+Z) and "Copy last text" (Alt+Shift+X) exist for recovery. [terminals/WSL](https://docs.wisprflow.ai/articles/6478598909-using-flow-with-linux-wsl-and-terminal-applications)
- **Latency:** stated target **700 ms p99** from end of speech to text: ~200 ms ASR + ~200 ms LLM + ~200 ms network. [Wispr blog](https://wisprflow.ai/post/technical-challenges). A competitor review says 1–2 s in practice (unverified). [spokenly](https://spokenly.app/blog/wispr-flow-review)

### What the "AI formatting" does
- **Backtrack:** filler words, false starts, self-corrections. Trigger phrases ("actually", "scratch that", "never mind") or just restating. "coffee at 2 actually 3" → "coffee at 3". [Smart Formatting & Backtrack](https://docs.wisprflow.ai/articles/5373093536-how-do-i-use-smart-formatting-and-backtrack)
- **Smart Formatting:** capitalization, punctuation, 30+ spoken punctuation names, "new line / new paragraph", "one… two…" / "first… second…" into lists.
- **Styles:** per app category (Personal, Work, Email, Other) at Formal / Casual / Very Casual. Casual drops trailing periods in chat apps. English only.
- **Dictionary:** manual entries plus auto-learned words, capped at 60 characters. **Snippets** (voice text expansion) on the free plan. [dictionary](https://docs.wisprflow.ai/articles/4052411709-teach-flow-your-words-with-the-dictionary)
- **Command Mode:** hold **Ctrl+Win+Alt** and speak an instruction. Acts on selected text, or text around the cursor; with neither, does nothing. "ask/search + Google/ChatGPT/…" opens a web search. Paid plan only. [command mode](https://docs.wisprflow.ai/articles/4816967992-how-to-use-command-mode)
- **Languages:** claims 100+.

### What it reads from the active app ("Context Awareness")
- Sent with every dictation unless Privacy Mode is on: app info, textbox contents (before / selected / after cursor), on-screen text, variable and file names in code editors, user ID in that app, apps in the session, **a screenshot**, conversation history. Skips password fields, numeric-only fields, URL bars, banking apps. [how it works](https://docs.wisprflow.ai/articles/5020906721-how-does-context-awareness-work)
- Windows is only partly covered; conversation context, element descriptions and IDE-specific formatting are Mac-only.
- Used for: spelling names correctly, tagging file names in Cursor/Windsurf, picking the style by app category. [context awareness](https://docs.wisprflow.ai/articles/4678293671-feature-context-awareness)

### Pain points
- **Privacy:** no offline mode. A 2025 incident with screenshots and app/URL tracking made news. [HN](https://news.ycombinator.com/item?id=47781148), [ModelPiper](https://modelpiper.com/blog/wispr-flow-privacy-incident) (competitor sources, not checked against a primary).
- **Insertion fails** (Wispr's own docs): apps running as administrator unless Flow also does; RDP/Citrix without clipboard sharing; **WSL windows, Linux VMs and SSH sessions**; terminals that need Ctrl+Shift+V; context formatting "can unexpectedly transform shell commands". [remote desktop](https://docs.wisprflow.ai/articles/7336156466-use-flow-with-remote-desktops-citrix-rdp-vdi), [text not pasting](https://docs.wisprflow.ai/articles/7971211038-fix-text-not-pasting-after-dictation)
- **Windows resource use (unverified, competitor):** ~800 MB RAM, 8% CPU idle, VS Code freezes, re-adds itself to startup. [spokenly](https://spokenly.app/blog/wispr-flow-review)
- **Outages** on its status page.

### Pricing
| Plan | Price | Notes |
|---|---|---|
| Free | $0 | 2,000 words/week desktop, 1,000 mobile |
| Pro | $15/mo, $12/mo annual | Unlimited; Command Mode |
| Growth | $18–23 | SSO, HIPAA |

[pricing](https://wisprflow.ai/pricing). Free competitor on Windows: **Voice Access "Fluid dictation"**, on-device SLM, Copilot+ PC (40+ TOPS NPU) only, English only. [Microsoft](https://support.microsoft.com/en-us/accessibility/windows/voice-access/fluid-dictation)

## Part 2: Open-source prior art

Repo figures read from the GitHub API on 2026-09-24.

### Handy — the main reference ([cjpais/Handy](https://github.com/cjpais/Handy))
- Rust + Tauri 2, MIT, ~32k stars, v0.9.7 released 2026-09-18.
- **STT:** Whisper via `transcribe-cpp` (Vulkan on Windows x64); `transcribe-rs` (ONNX) for Parakeet, Moonshine, SenseVoice, GigaAM, Canary, Cohere. Streaming models since v0.9.0 through [transcribe.cpp](https://github.com/handy-computer/transcribe.cpp) (MIT, "16+ model families"). Silero VAD.
- **Hotkeys:** [`handy-keys`](https://github.com/handy-computer/handy-keys): low-level keyboard hook on Windows, blocks the key, modifier-only hotkeys, key-up detection, hold / toggle / hold-only / toggle-only.
- **Insertion:** `CtrlV` (default on Windows), `ShiftInsert`, `Direct` (enigo typing), `None`. Saves clipboard and restores **after a fixed delay** → long-open critical bug **#502**: on a busy machine it pastes the *old* clipboard. A "Reliable Paste (Beta)" waiting on clipboard-read notifications is hidden in the Debug menu.
- **LLM cleanup:** exists, off by default, separate hotkey. Providers: OpenAI, Z.ai, OpenRouter, Anthropic, Groq, Cerebras, Bedrock, Apple Intelligence, Custom (defaults to `http://localhost:11434/v1`). Single prompt template; no bundled local LLM.
- **Lacks:** any app context (no foreground window / selected / surrounding text), per-app styles, command mode. Dictionary is a fuzzy post-correction list, not a model hint.
- **Open Windows issues:** freeze when target runs as admin (#434), held hotkey modifiers leaking into Ctrl+V (#2051), pastes missing parts of text (#2126), Vulkan device-lost errors/freezes (#2047, #1775, #1841).

### Others
| Project | Stack / licence / activity | Windows insertion | PTT | LLM cleanup |
|---|---|---|---|---|
| **Whispering** ([Epicenter](https://github.com/EpicenterHQ/epicenter)) | Svelte/TS + Tauri, AGPL-3.0, 4.8k stars | enigo Ctrl+V; snapshot + restore 100 ms later; marks text so clipboard managers skip it; moved off `enigo.text` (slow, broken; v7.1.1 notes) | Tauri plugin; key-up unverified | "Transformations": Ollama, LM Studio, OpenAI, OpenRouter, Groq |
| **OpenWhispr** ([repo](https://github.com/OpenWhispr/openwhispr)) | Electron, MIT, 8.5k stars | native helper (`windows-fast-paste.c`) uses **SendInput**; detects terminals by class/exe and sends **Ctrl+Shift+V**; re-focuses the window captured at record start; releases held modifiers; restores clipboard only if unchanged | yes | llama.cpp local or cloud; agent hotkey edits selected text in place; opt-in screenshot context. Closest to Command Mode. |
| **Voicetypr** ([repo](https://github.com/ideaplexa/voicetypr)) | Tauri/Rust, AGPL-3.0, paid licence | at cursor | PTT + toggle | cloud / custom endpoint; Vulkan Whisper in an **isolated sidecar process** so GPU crashes don't kill the app |
| **Tambourine** ([repo](https://github.com/kstonekuan/tambourine-voice)) | Tauri + Python Pipecat server, AGPL-3.0 | types | hold | **per-app formatting** from focused app; voice shortcuts; mutes audio while dictating; local via faster-whisper + Ollama |
| **WhisperWriter** | Python, GPL-3.0, stale since 2024 | pynput typing | hold/toggle/VAD/continuous | none |
| **VoiceInk** ([repo](https://github.com/Beingpax/VoiceInk)) | Swift, macOS only, GPL-3.0 | — | PTT | screen context, per-app "Power Mode", dictionary, Ollama. Best feature reference. |
| **OpenSuperWhisper**, **FreeFlow** | Swift, macOS | — | hold | none / Groq |
| **BlahST** | shell, Linux | xdotool | hotkey pair | whisper.cpp + llama.cpp fully local |
| **Vibe** | Tauri, MIT | file transcription only | — | — |

Closed-source (unverified competitor sources): **superwhisper** Windows 1.0 late 2025, local/cloud per mode, $8.49/mo or $249.99 lifetime; **Typeless** cloud only, $30/mo; **Talon** is a voice-command tool.

Very new Rust clones (0–14 stars, created Jul–Sep 2026): voxable, rustle, openflow (Handy fork).

## What a v1 that beats Handy needs
1. **Local LLM cleanup on by default with no setup** (bundled small model), strict "clean, don't answer" prompt, backtrack, spoken punctuation, lists; ~1 s end to end on mid-range hardware.
2. **Correct clipboard handling:** wait for the target to actually read the clipboard (delayed rendering), mark the temp text so history skips it, restore only if unchanged since, SendInput Unicode fallback. Exactly where Handy #502 fails.
3. **Paste methods that know the target app:** Ctrl+Shift+V for terminals, Shift+Insert where better, re-focus the window captured at record start, release held modifiers before the chord (Handy #2051).
4. **Reliable push-to-talk:** low-level hook with key-up, modifier-only chords, no Start-menu side effect, cancel key, double-tap to lock hands-free.
5. **App-aware styles:** foreground exe and title, optionally selected/surrounding text via UI Automation.
6. **Command mode on selected text, locally.** Only OpenWhispr has something close.
7. **Tell the user when insertion fails:** detect elevated targets; always offer "paste last transcript".
8. **Low idle footprint and isolated GPU work** (Vulkan crashes are Handy's main Windows problem).

Nobody has nailed: local cleanup close to Wispr's cloud latency/quality; clipboard restore reliable under load; insertion into WSL/SSH/RDP; context-aware formatting that doesn't mangle shell commands; a dictionary that learns from post-insertion edits locally.
