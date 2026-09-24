# whisper-local

Local-only voice dictation for Windows in Rust: hold a key, speak, release, cleaned-up
text appears where the cursor is. Product decisions and architecture live in
`docs/design.md`; read it before changing anything structural. Research behind those
decisions is in `docs/research/`, dated and perishable.

## How work is done here

- **Latency is the product.** Every stage reports its own timing; a change that adds
  latency to the release-to-text path needs a measured justification.
- **Text is never silently lost.** If insertion fails, the text goes to the clipboard and
  the user is told. Every insertion path ends in either a confirmed receipt or a visible
  error.
- **The LLM is a cleaner, not an assistant.** Its output is validated against the source
  transcript before it can reach the target app, and it falls back to the rule pass.
  Nothing dictated may be *answered*.
- **Nothing leaves the machine.** No network calls except model downloads from the
  manifest and the user's own local inference server.
- **Platform code stays in the platform crate.** `core`, `audio`, `stt` and `normalize`
  compile without Win32 and carry the unit tests.
- **Engines are swappable.** Anything that talks to a model implements a trait, reports
  which backend actually loaded, and can be replaced by a process boundary later.

## Conventions

- Edition 2024, `cargo fmt` and `cargo clippy -D warnings` clean before a change is
  considered done. Tests run with `cargo test --workspace`.
- Errors: `thiserror` in library crates, `anyhow` only in the binary and tools.
- Logging through `tracing`; the hook callback and audio callback log nothing.
- Pin exact versions for crates that do not follow semver (llama.cpp and whisper.cpp
  bindings, ONNX Runtime wrappers).
- Files are changed with the Edit and Write tools, never by a script.
- Comments say *why not*, never *what*. A comment that names another module or a command
  will rot; prefer a test.
- Anything in docs that names a crate, version, model or path carries a date.
- Measurements go into `docs/design.md` §7 with method and date, or not at all.
- Temporary scripts and scratch output go outside the repo.
