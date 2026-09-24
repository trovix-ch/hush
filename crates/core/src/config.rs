//! User configuration: one TOML file, every key optional.

use std::path::PathBuf;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::gpu::{GpuRequest, GpuSelector};
use crate::insert::{AppPolicies, AppPolicy, Chord};
use crate::normalize::Style;
use crate::segment::SegmenterConfig;

/// Written on first run. Must parse to `Config::default()`.
pub const DEFAULT_CONFIG: &str = r#"# hush configuration. Every key is optional; a missing key takes the value shown.

# Hold to talk. CapsLock is the alternative for keyboards without a Right Ctrl.
hotkey = "RightCtrl"
# Double-tap the hotkey to dictate hands-free; tap once more to stop.
hands_free_double_tap = true
# A recording always ends after this long, even if the key-up never arrives.
max_recording_secs = 120
# Utterances kept for "paste last" and "copy last".
history_len = 10
# Words and phrases spelled exactly as you want them.
vocabulary = []
# More of them, one per line, `#` starts a comment. Re-read when it changes. A relative
# path is relative to this file.
# vocabulary_file = "vocabulary.txt"
# Style for apps no rule below matches: formal | casual | code | none
default_style = "casual"
# Start hush when you sign in. The tray's "Start with Windows" writes this line.
start_with_windows = false

[engine]
# Model id from the download manifest.
model = "parakeet-tdt-0.6b-v3-f16-gguf"
# require-gpu | prefer-gpu | cpu-only
gpu = "prefer-gpu"
# Which GPU: "auto" takes the discrete card with the most free memory, so not an
# integrated GPU and not a card another model already fills. A PCI bus id from
# `hush doctor` ("0000:05:00.0" or "05:00") or part of the device name pins one.
device = "auto"
# Speech runs in hush-stt-worker.exe, next to hush.exe, so a GPU driver crash restarts
# the worker instead of taking hush down. true runs it inside hush, for debugging; only
# builds with the `in-process-stt` feature can.
in_process = false
# worker_path = 'C:\path\to\hush-stt-worker.exe'
# An idle GPU drops its clocks after a few seconds and the next transcription runs about
# three times slower. true runs short dummy passes while you speak so the card is awake
# at release. Ignored when speech runs on the CPU.
gpu_nudge = true

[normalizer]
# llama-cpp | http | rules. Dictation is rules-only until the language model has loaded,
# and falls back to the rules whenever the model fails or its output is rejected.
kind = "llama-cpp"
# Model id from the download manifest; `hush doctor` downloads it.
model = "qwen3-4b-instruct-2507-q4_k_m"
# "auto" runs on the same GPU as speech; a PCI bus id or part of a name pins another.
device = "auto"
timeout_ms = 5000
# LLM cleanup through a local OpenAI-compatible server instead, e.g. Ollama:
# kind = "http"
# base_url = "http://127.0.0.1:11434/v1"
# model = "qwen3:4b-instruct-2507-q4_K_M"
# timeout_ms = 5000

[pipeline]
# Transcribe each sentence as soon as you pause, while the key is still held, so the
# release waits only for the last one. Off sends the whole recording at release.
# Off by default: a long pause inside a sentence can still come out as a full stop.
pre_transcribe = false

# How speech is cut into sentences for pre-transcription.
[pipeline.segmenter]
# Speech shorter than this does not end on a pause; it joins the next sentence.
min_speech_ms = 300
# A pause at least this long ends a sentence.
min_pause_ms = 700
# Audio kept before and after each sentence.
pad_ms = 200
# Longer speech without a pause is split at its quietest point.
max_segment_ms = 20000
# How late the voice detector reports that speech started; a pause is judged only
# after this much more audio, so the same recording always splits the same way.
vad_latency_ms = 100
# Where the words either side of a split are closer than this, the sentence end the
# engine put there is removed and the next word lowercased.
join_gap_ms = 400
# ... and joined with a comma if they are at least this far apart.
comma_gap_ms = 300

# Per-app rules, matched on the executable name, case-insensitively; exe = "*" sets the
# rule for every other app. Known terminals and editors already get code style from a
# built-in table (`hush doctor` prints it); a rule here for the same exe replaces it.
# chord: ctrl-v | ctrl-shift-v | shift-insert; style: formal | casual | code | none
[[app]]
exe = "windowsterminal.exe"
chord = "shift-insert"
never_type = false
style = "code"

[[app]]
exe = "conhost.exe"
chord = "shift-insert"
never_type = false
style = "code"
"#;

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum GpuPolicy {
    RequireGpu,
    /// Falls back to the CPU and says so; never silently.
    #[default]
    PreferGpu,
    CpuOnly,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct EngineChoice {
    pub model: String,
    pub gpu: GpuPolicy,
    pub device: GpuSelector,
    /// Deprecated: a Vulkan index, whose order differs between console and remote
    /// sessions. Read, used while `device` is auto, and warned about.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_device: Option<usize>,
    /// Speech inside the app process rather than the worker; debugging only.
    pub in_process: bool,
    /// Unset means the worker next to the executable.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub worker_path: Option<PathBuf>,
    /// Wake the GPU at key-down and keep it awake while recording. Ignored on the CPU.
    pub gpu_nudge: bool,
}

impl Default for EngineChoice {
    fn default() -> Self {
        Self {
            model: "parakeet-tdt-0.6b-v3-f16-gguf".into(),
            gpu: GpuPolicy::default(),
            device: GpuSelector::Auto,
            gpu_device: None,
            in_process: false,
            worker_path: None,
            gpu_nudge: true,
        }
    }
}

impl EngineChoice {
    pub fn gpu_request(&self) -> GpuRequest {
        legacy_or(&self.device, self.gpu_device)
    }
}

fn legacy_or(device: &GpuSelector, legacy: Option<usize>) -> GpuRequest {
    match (device, legacy) {
        (GpuSelector::Auto, Some(i)) => GpuRequest::LegacyIndex(i),
        (s, _) => GpuRequest::Selector(s.clone()),
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub enum NormalizerChoice {
    Rules,
    /// llama.cpp inside the process. A build without it runs rules-only and says so.
    LlamaCpp {
        #[serde(default = "default_llm_model")]
        model: String,
        /// Auto means the speech engine's device.
        #[serde(default)]
        device: GpuSelector,
        /// Deprecated, as `EngineChoice::gpu_device`.
        #[serde(default, skip_serializing_if = "Option::is_none")]
        gpu_device: Option<usize>,
        #[serde(default = "default_llm_timeout_ms")]
        timeout_ms: u64,
    },
    /// A local OpenAI-compatible server.
    Http {
        base_url: String,
        model: String,
        #[serde(default = "default_llm_timeout_ms")]
        timeout_ms: u64,
    },
}

impl Default for NormalizerChoice {
    fn default() -> Self {
        Self::LlamaCpp {
            model: default_llm_model(),
            device: GpuSelector::Auto,
            gpu_device: None,
            timeout_ms: default_llm_timeout_ms(),
        }
    }
}

impl NormalizerChoice {
    /// `None` when the language model follows the speech engine's device.
    pub fn gpu_request(&self) -> Option<GpuRequest> {
        match self {
            Self::LlamaCpp {
                device, gpu_device, ..
            } => match legacy_or(device, *gpu_device) {
                GpuRequest::Selector(GpuSelector::Auto) => None,
                r => Some(r),
            },
            Self::Rules | Self::Http { .. } => None,
        }
    }
}

fn default_llm_model() -> String {
    "qwen3-4b-instruct-2507-q4_k_m".into()
}

fn default_llm_timeout_ms() -> u64 {
    5000
}

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct PipelineSettings {
    pub pre_transcribe: bool,
    pub segmenter: SegmenterConfig,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AppRule {
    pub exe: String,
    #[serde(default)]
    pub chord: Chord,
    #[serde(default)]
    pub never_type: bool,
    #[serde(default)]
    pub style: Style,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct Config {
    pub hotkey: String,
    pub hands_free_double_tap: bool,
    #[serde(rename = "max_recording_secs", with = "secs")]
    pub max_recording: Duration,
    pub history_len: usize,
    pub vocabulary: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub vocabulary_file: Option<PathBuf>,
    pub default_style: Style,
    pub start_with_windows: bool,
    pub engine: EngineChoice,
    pub normalizer: NormalizerChoice,
    pub pipeline: PipelineSettings,
    #[serde(rename = "app")]
    pub apps: Vec<AppRule>,
}

impl Default for Config {
    fn default() -> Self {
        let terminal = |exe: &str| AppRule {
            exe: exe.into(),
            chord: Chord::ShiftInsert,
            never_type: false,
            style: Style::Code,
        };
        Self {
            hotkey: "RightCtrl".into(),
            hands_free_double_tap: true,
            max_recording: Duration::from_secs(120),
            history_len: 10,
            vocabulary: Vec::new(),
            vocabulary_file: None,
            default_style: Style::Casual,
            start_with_windows: false,
            engine: EngineChoice::default(),
            normalizer: NormalizerChoice::default(),
            pipeline: PipelineSettings::default(),
            apps: vec![terminal("windowsterminal.exe"), terminal("conhost.exe")],
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("config: {0}")]
    Parse(String),
    #[error("config: {0}")]
    Serialize(String),
}

impl Config {
    pub fn from_toml(s: &str) -> Result<Self, ConfigError> {
        toml::from_str(s).map_err(|e| ConfigError::Parse(e.to_string()))
    }

    pub fn to_toml(&self) -> Result<String, ConfigError> {
        toml::to_string(self).map_err(|e| ConfigError::Serialize(e.to_string()))
    }

    /// Keys still read for one release, each with what to write instead.
    pub fn deprecations(&self) -> Vec<String> {
        let mut out = Vec::new();
        let mut index = |key: &str, i: usize, device: &GpuSelector| {
            let effect = if *device == GpuSelector::Auto {
                "still used for now"
            } else {
                "ignored, because device is set"
            };
            out.push(format!(
                "{key}.gpu_device = {i} is deprecated ({effect}): Vulkan numbers the devices \
                 differently in console and remote sessions. Write {key}.device = \"<PCI bus \
                 id>\" from `hush doctor`, or \"auto\""
            ));
        };
        if let Some(i) = self.engine.gpu_device {
            index("engine", i, &self.engine.device);
        }
        if let NormalizerChoice::LlamaCpp {
            device,
            gpu_device: Some(i),
            ..
        } = &self.normalizer
        {
            index("normalizer", *i, device);
        }
        out
    }

    pub fn app_policies(&self) -> AppPolicies {
        AppPolicies {
            default: self.default_app_policy(),
            by_exe: self
                .effective_app_rules()
                .into_iter()
                .map(|r| (r.exe, r.policy))
                .collect(),
        }
    }

    /// The `*` rule if there is one, else `default_style` with the default chord.
    pub fn default_app_policy(&self) -> AppPolicy {
        self.apps
            .iter()
            .rfind(|r| r.exe.trim() == WILDCARD)
            .map_or_else(
                || AppPolicy {
                    style: self.default_style,
                    ..AppPolicy::default()
                },
                AppRule::policy,
            )
    }

    /// User rules first, then the built-in rows no user rule names. A later user rule for
    /// the same exe wins over an earlier one.
    pub fn effective_app_rules(&self) -> Vec<EffectiveAppRule> {
        let mut out: Vec<EffectiveAppRule> = Vec::new();
        for r in self.apps.iter().rev() {
            let exe = r.exe.trim().to_ascii_lowercase();
            if exe == WILDCARD || out.iter().any(|e| e.exe == exe) {
                continue;
            }
            out.push(EffectiveAppRule {
                exe,
                policy: r.policy(),
                builtin: false,
            });
        }
        out.reverse();
        for &(exe, chord) in BUILTIN_CODE_APPS {
            if !out.iter().any(|e| e.exe == exe) {
                out.push(EffectiveAppRule {
                    exe: exe.into(),
                    policy: AppPolicy {
                        chord,
                        never_type: false,
                        style: Style::Code,
                    },
                    builtin: true,
                });
            }
        }
        out
    }
}

const WILDCARD: &str = "*";

/// `text` with the top-level `key` set to `value`. Rewriting the file through `to_toml`
/// would drop every comment the default config carries, so the one line is replaced in
/// place, or added before the first table.
pub fn with_top_level_bool(text: &str, key: &str, value: bool) -> String {
    let line = format!("{key} = {value}");
    let is_key = |l: &str| {
        l.trim_start()
            .strip_prefix(key)
            .is_some_and(|rest| rest.trim_start().starts_with('='))
    };
    let mut out = String::with_capacity(text.len() + line.len() + 1);
    let mut done = false;
    let mut in_table = false;
    for l in text.split_inclusive('\n') {
        let body = l.trim_end_matches(['\r', '\n']);
        let ending = &l[body.len()..];
        if !in_table && body.trim_start().starts_with('[') {
            in_table = true;
            if !done {
                out.push_str(&line);
                out.push_str(if ending.is_empty() { "\n" } else { ending });
                done = true;
            }
        }
        if !in_table && !done && is_key(body) {
            out.push_str(&line);
            out.push_str(ending);
            done = true;
            continue;
        }
        out.push_str(l);
    }
    if !done {
        if !out.is_empty() && !out.ends_with('\n') {
            out.push('\n');
        }
        out.push_str(&line);
        out.push('\n');
    }
    out
}

/// Terminals and editors, where a capital letter or an added period breaks a command.
/// Terminals get the paste chord they accept without configuration: conhost and PuTTY
/// take Shift+Insert, WezTerm and Alacritty bind Ctrl+Shift+V, and Ctrl+V reaches a shell
/// as a literal ^V in several of them.
pub const BUILTIN_CODE_APPS: &[(&str, Chord)] = &[
    ("windowsterminal.exe", Chord::ShiftInsert),
    ("conhost.exe", Chord::ShiftInsert),
    ("cmd.exe", Chord::ShiftInsert),
    ("powershell.exe", Chord::ShiftInsert),
    ("pwsh.exe", Chord::ShiftInsert),
    ("putty.exe", Chord::ShiftInsert),
    ("wezterm-gui.exe", Chord::CtrlShiftV),
    ("alacritty.exe", Chord::CtrlShiftV),
    ("code.exe", Chord::CtrlV),
    ("cursor.exe", Chord::CtrlV),
    ("idea64.exe", Chord::CtrlV),
    ("rider64.exe", Chord::CtrlV),
];

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EffectiveAppRule {
    /// Lower-case.
    pub exe: String,
    pub policy: AppPolicy,
    pub builtin: bool,
}

impl AppRule {
    fn policy(&self) -> AppPolicy {
        AppPolicy {
            chord: self.chord,
            never_type: self.never_type,
            style: self.style,
        }
    }
}

/// Whole seconds, so a TOML reader never sees serde's `{secs, nanos}` shape.
mod secs {
    use std::time::Duration;

    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(d: &Duration, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_u64(d.as_secs())
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Duration, D::Error> {
        u64::deserialize(d).map(Duration::from_secs)
    }
}

pub(crate) mod millis {
    use std::time::Duration;

    use serde::{Deserialize, Deserializer, Serializer};

    pub fn serialize<S: Serializer>(d: &Duration, s: S) -> Result<S::Ok, S::Error> {
        s.serialize_u64(d.as_millis() as u64)
    }

    pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Duration, D::Error> {
        u64::deserialize(d).map(Duration::from_millis)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_config_text_is_the_default() {
        assert_eq!(
            Config::from_toml(DEFAULT_CONFIG).unwrap(),
            Config::default()
        );
    }

    #[test]
    fn round_trips_through_toml() {
        let mut c = Config {
            vocabulary: vec!["Kubernetes".into(), "gRPC".into()],
            max_recording: Duration::from_secs(45),
            ..Config::default()
        };
        c.engine.gpu = GpuPolicy::RequireGpu;
        c.engine.device = "0000:05:00.0".parse().unwrap();
        c.engine.in_process = true;
        c.engine.worker_path = Some(r"C:\hush\hush-stt-worker.exe".into());
        c.engine.gpu_nudge = false;
        c.normalizer = NormalizerChoice::Http {
            base_url: "http://127.0.0.1:11434/v1".into(),
            model: "qwen3:4b-instruct-2507-q4_K_M".into(),
            timeout_ms: 1500,
        };
        c.apps.push(AppRule {
            exe: "slack.exe".into(),
            chord: Chord::CtrlV,
            never_type: true,
            style: Style::Casual,
        });
        c.pipeline.pre_transcribe = false;
        c.pipeline.segmenter = SegmenterConfig {
            min_speech: Duration::from_millis(250),
            min_pause: Duration::from_millis(550),
            pad: Duration::from_millis(150),
            max_segment: Duration::from_secs(12),
            vad_latency: Duration::from_millis(120),
            join_gap: Duration::from_millis(500),
            comma_gap: Duration::from_millis(250),
        };
        let text = c.to_toml().unwrap();
        assert_eq!(Config::from_toml(&text).unwrap(), c, "{text}");
        assert_eq!(
            Config::from_toml(&Config::default().to_toml().unwrap()).unwrap(),
            Config::default()
        );
    }

    #[test]
    fn missing_keys_take_defaults_and_typos_are_errors() {
        let c = Config::from_toml("hotkey = \"CapsLock\"\n[normalizer]\nkind = \"http\"\nbase_url = \"http://127.0.0.1:8080/v1\"\nmodel = \"m\"\n").unwrap();
        assert_eq!(c.hotkey, "CapsLock");
        assert_eq!(c.history_len, 10);
        assert!(matches!(
            c.normalizer,
            NormalizerChoice::Http {
                timeout_ms: 5000,
                ..
            }
        ));
        assert!(Config::from_toml("hot_key = \"CapsLock\"").is_err());
        let c = Config::from_toml("[pipeline.segmenter]\npad_ms = 100\n").unwrap();
        assert!(!c.pipeline.pre_transcribe);
        assert_eq!(c.pipeline.segmenter.pad, Duration::from_millis(100));
        assert_eq!(c.pipeline.segmenter.min_pause, Duration::from_millis(700));
        assert!(Config::from_toml("[pipeline.segmenter]\npad = 100\n").is_err());
    }

    #[test]
    fn app_rules_become_case_insensitive_policies() {
        let mut c = Config::default();
        c.apps.push(AppRule {
            exe: "Slack.EXE".into(),
            chord: Chord::CtrlV,
            never_type: true,
            style: Style::Casual,
        });
        let p = c.app_policies();
        assert!(p.lookup(Some("slack.exe")).never_type);
        assert_eq!(
            p.lookup(Some("WindowsTerminal.exe")).chord,
            Chord::ShiftInsert
        );
        assert_eq!(p.lookup(None), &AppPolicy::default());
    }

    fn rule(exe: &str, style: Style) -> AppRule {
        AppRule {
            exe: exe.into(),
            chord: Chord::CtrlV,
            never_type: false,
            style,
        }
    }

    #[test]
    fn app_rule_matching_exact_wildcard_case_and_builtin_override() {
        let mut c = Config {
            apps: vec![],
            ..Config::default()
        };
        let p = c.app_policies();
        assert_eq!(p.lookup(Some("Code.EXE")).style, Style::Code);
        assert_eq!(p.lookup(Some("wezterm-gui.exe")).chord, Chord::CtrlShiftV);
        assert_eq!(p.lookup(Some("slack.exe")).style, Style::Casual);
        assert_eq!(p.lookup(None).style, Style::Casual);

        c.default_style = Style::Formal;
        assert_eq!(
            c.app_policies().lookup(Some("slack.exe")).style,
            Style::Formal
        );

        c.apps = vec![
            rule("SLACK.exe", Style::Casual),
            rule("code.exe", Style::Formal),
            rule("*", Style::None),
        ];
        let p = c.app_policies();
        assert_eq!(p.lookup(Some("slack.exe")).style, Style::Casual);
        assert_eq!(p.lookup(Some("CODE.exe")).style, Style::Formal);
        assert_eq!(p.lookup(Some("code.exe")).chord, Chord::CtrlV);
        assert_eq!(
            p.lookup(Some("notepad.exe")).style,
            Style::None,
            "`*` beats default_style"
        );
        assert_eq!(p.lookup(None).style, Style::None);
        assert_eq!(
            p.lookup(Some("pwsh.exe")).style,
            Style::Code,
            "a built-in row is more specific than `*`"
        );

        let rules = c.effective_app_rules();
        assert!(!rules.iter().any(|r| r.exe == "*"));
        let code: Vec<_> = rules.iter().filter(|r| r.exe == "code.exe").collect();
        assert_eq!(code.len(), 1);
        assert!(!code[0].builtin);
        assert_eq!(
            rules.iter().filter(|r| r.builtin).count(),
            BUILTIN_CODE_APPS.len() - 1
        );
    }

    #[test]
    fn a_later_duplicate_user_rule_wins() {
        let c = Config {
            apps: vec![
                rule("slack.exe", Style::Casual),
                rule("Slack.exe", Style::Formal),
            ],
            ..Config::default()
        };
        assert_eq!(
            c.app_policies().lookup(Some("slack.exe")).style,
            Style::Formal
        );
    }

    #[test]
    fn embedded_llm_is_the_default_and_round_trips() {
        assert_eq!(
            Config::default().normalizer,
            NormalizerChoice::LlamaCpp {
                model: "qwen3-4b-instruct-2507-q4_k_m".into(),
                device: GpuSelector::Auto,
                gpu_device: None,
                timeout_ms: 5000,
            }
        );
        let c = Config::from_toml("[normalizer]\nkind = \"llama-cpp\"\n").unwrap();
        assert_eq!(c.normalizer, NormalizerChoice::default());
        let c = Config {
            normalizer: NormalizerChoice::LlamaCpp {
                model: "other-model".into(),
                device: GpuSelector::Name("RTX 5060".into()),
                gpu_device: None,
                timeout_ms: 800,
            },
            ..Config::default()
        };
        let text = c.to_toml().unwrap();
        assert!(text.contains("kind = \"llama-cpp\""), "{text}");
        assert_eq!(Config::from_toml(&text).unwrap(), c);
        let rules = Config {
            normalizer: NormalizerChoice::Rules,
            ..Config::default()
        };
        assert_eq!(Config::from_toml(&rules.to_toml().unwrap()).unwrap(), rules);
        assert!(
            Config::from_toml("[normalizer]\nkind = \"llama-cpp\"\nbase_url = \"x\"\n").is_err()
        );
    }

    #[test]
    fn gpu_device_takes_auto_a_pci_id_or_a_name_and_never_an_index() {
        use crate::gpu::PciAddress;

        let engine = |toml: &str| Config::from_toml(&format!("[engine]\n{toml}\n"));
        assert_eq!(
            engine("").unwrap().engine.gpu_request(),
            GpuRequest::Selector(GpuSelector::Auto)
        );
        assert_eq!(
            engine("device = \"auto\"").unwrap().engine.device,
            GpuSelector::Auto
        );
        let pci = engine("device = \"0000:05:00.0\"").unwrap();
        assert_eq!(
            pci.engine.device,
            GpuSelector::Pci(PciAddress::parse("0000:05:00.0").unwrap())
        );
        assert!(pci.to_toml().unwrap().contains("device = \"0000:05:00.0\""));
        assert!(matches!(
            engine("device = \"05:00\"").unwrap().engine.device,
            GpuSelector::Pci(PciAddress {
                bus: 5,
                domain: None,
                ..
            })
        ));
        assert_eq!(
            engine("device = \"RTX 5060\"").unwrap().engine.device,
            GpuSelector::Name("RTX 5060".into())
        );
        assert!(engine("device = 1").is_err());
        assert!(engine("device = \"1\"").is_err());
        assert!(engine("device = \"\"").is_err());
        assert!(Config::default().deprecations().is_empty());

        let legacy = Config::from_toml(
            "[engine]\ngpu_device = 1\n[normalizer]\nkind = \"llama-cpp\"\ngpu_device = 0\n",
        )
        .unwrap();
        assert_eq!(legacy.engine.gpu_request(), GpuRequest::LegacyIndex(1));
        assert_eq!(
            legacy.normalizer.gpu_request(),
            Some(GpuRequest::LegacyIndex(0))
        );
        let warnings = legacy.deprecations();
        assert_eq!(warnings.len(), 2, "{warnings:?}");
        assert!(warnings[0].contains("engine.device"), "{}", warnings[0]);
        assert_eq!(
            Config::from_toml(&legacy.to_toml().unwrap()).unwrap(),
            legacy
        );

        let both = engine("device = \"05:00\"\ngpu_device = 1").unwrap();
        assert!(matches!(
            both.engine.gpu_request(),
            GpuRequest::Selector(GpuSelector::Pci(_))
        ));
        assert!(both.deprecations()[0].contains("ignored"));
        assert_eq!(Config::default().normalizer.gpu_request(), None);
    }

    #[test]
    fn start_with_windows_is_set_in_place_with_comments_kept() {
        let on = with_top_level_bool(DEFAULT_CONFIG, "start_with_windows", true);
        let c = Config::from_toml(&on).unwrap();
        assert!(c.start_with_windows);
        assert_eq!(
            Config {
                start_with_windows: false,
                ..c.clone()
            },
            Config::default()
        );
        assert_eq!(on.lines().count(), DEFAULT_CONFIG.lines().count());
        assert!(on.contains("# Start hush when you sign in."));
        assert_eq!(
            with_top_level_bool(&on, "start_with_windows", false),
            DEFAULT_CONFIG
        );
        assert_eq!(Config::from_toml(&c.to_toml().unwrap()).unwrap(), c);

        let crlf = "hotkey = \"CapsLock\"\r\n# start_with_windows = false\r\n[engine]\r\ngpu = \"cpu-only\"\r\n";
        let added = with_top_level_bool(crlf, "start_with_windows", true);
        assert_eq!(
            added,
            "hotkey = \"CapsLock\"\r\n# start_with_windows = false\r\nstart_with_windows = true\r\n[engine]\r\ngpu = \"cpu-only\"\r\n"
        );
        let c = Config::from_toml(&added).unwrap();
        assert!(c.start_with_windows);
        assert_eq!(c.engine.gpu, GpuPolicy::CpuOnly);
        assert_eq!(
            with_top_level_bool("hotkey = \"F9\"", "start_with_windows", true),
            "hotkey = \"F9\"\nstart_with_windows = true\n"
        );
        assert_eq!(
            with_top_level_bool("", "start_with_windows", true),
            "start_with_windows = true\n"
        );
    }

    #[test]
    fn vocabulary_file_and_default_style_round_trip() {
        let c = Config::from_toml("vocabulary_file = 'C:/x/words.txt'\ndefault_style = 'code'\n")
            .unwrap();
        assert_eq!(c.vocabulary_file, Some(PathBuf::from("C:/x/words.txt")));
        assert_eq!(c.default_style, Style::Code);
        assert_eq!(Config::from_toml(&c.to_toml().unwrap()).unwrap(), c);
    }
}
