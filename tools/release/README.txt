hush - local voice dictation for Windows
=========================================

Hold a key, talk, let go: the text appears where your cursor is. Speech
recognition and the clean-up run on this computer; nothing you say leaves it.

Install
  Unzip anywhere and keep hush.exe and hush-stt-worker.exe in the same folder.
  Needs Windows 10 or 11 (x64), a GPU driver with Vulkan (a CPU fallback exists
  but is slower), and the Microsoft Visual C++ Redistributable 2015-2022 (x64):
  https://aka.ms/vs/17/release/vc_redist.x64.exe

First run
  Start hush.exe. There is no window: look for the microphone in the tray. On
  the first start it downloads its models (about 1.3 GB for speech, 2.5 GB for
  the language model); the pill at the bottom of the screen shows the progress.
  Dictation works as soon as the speech model is in, and the text clean-up gets
  better once the language model has loaded. If a download fails, use
  "Retry download" in the tray menu; it resumes where it stopped.

Use
  Hold Right Ctrl, speak, release.
  Double-tap Right Ctrl to dictate hands-free; tap again to stop.
  Escape cancels. The tray menu pastes or copies the last transcript, pauses,
  opens the config, and turns "Start with Windows" on or off.

Files
  Config  %APPDATA%\hush\config.toml   (written on first start, every key commented)
  Models  %LOCALAPPDATA%\hush\models   (or the folder named by HUSH_MODELS_DIR)
  Logs    %LOCALAPPDATA%\hush\logs

Trouble
  From a terminal in this folder, run:   .\hush.exe doctor | Out-Host
  It checks the microphone, GPUs, models, speech engine and language model.
  (hush has no console of its own, so the shell does not wait for it; the pipe
  makes it wait.)

Licences: LICENSE (hush, MIT) and THIRD-PARTY.md (models and libraries).
