# bench-stt fixtures

All clips are 16 kHz, mono, 16-bit PCM WAV. Made 2026-09-24.

| file | length | content |
|---|---|---|
| `tts-03s-question.wav` | 3.4 s | "Can you send me the latest numbers before lunch?" |
| `tts-10s-fillers.wav` | 9.5 s | Filler-heavy: "Um, so, like, I was thinking we should, uh, you know, move the meeting to the afternoon, because, um, basically half the team is out in the morning." |
| `tts-30s-dictation.wav` | 31.0 s | Email dictation with a self-correction ("please send it on Tuesday, no wait, Wednesday"). |
| `silence-02s.wav` | 2.0 s | Digital silence (all samples zero). Checks what an engine emits when there is no speech. |
| `jfk.wav` | 11.0 s | Real speech: J. F. Kennedy, 1961 inaugural address, "And so, my fellow Americans...". Public domain. |

## How they were made

The `tts-*` and `silence-*` clips come from Windows' built-in `System.Speech` synthesizer
(voice "Microsoft David Desktop"), written straight to 16 kHz mono:

```powershell
Add-Type -AssemblyName System.Speech
$fmt = New-Object System.Speech.AudioFormat.SpeechAudioFormatInfo(16000,
    [System.Speech.AudioFormat.AudioBitsPerSample]::Sixteen,
    [System.Speech.AudioFormat.AudioChannel]::Mono)
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$s.SetOutputToWaveFile($path, $fmt); $s.Speak($text); $s.SetOutputToNull()
```

`tts-10s-fillers.wav` used `$s.Rate = 2` to land near 10 s; the others the default rate.
The silence clip is `$s.SpeakSsml(...)` of a single `<break time="2000ms"/>`. The full
text of the 30 s clip:

> Hi Sarah, thanks for the update on the quarterly budget. I went through the spreadsheet
> last night and most of it looks right to me. The travel line seems a little high though,
> so could you double check the hotel costs for the Berlin trip? Also, please send it on
> Tuesday, no wait, Wednesday, because I am out of the office on Tuesday. If anything
> changes, just give me a call or drop me a quick message. Talk soon, and have a great
> weekend.

`jfk.wav` is the sample shipped with whisper.cpp
(`https://raw.githubusercontent.com/ggml-org/whisper.cpp/master/samples/jfk.wav`,
SHA-256 `59dfb9a4acb36fe2a2affc14bacbee2920ff435cb13cc314a08c13f66ba7860e`), so results
are not measured only on synthetic speech.

TTS audio is cleaner and more evenly paced than a real microphone, so accuracy on the
`tts-*` clips is an upper bound.
