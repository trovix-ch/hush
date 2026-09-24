//! User configuration: one TOML file, every key optional.

use std::path::PathBuf;
use std::time::Duration;

use serde::{Deserialize, Serialize};

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

[engine]
# Model id from the download manifest.
model = "parakeet-tdt-0.6b-v3-f16-gguf"
# require-gpu | prefer-gpu | cpu-only
gpu = "prefer-gpu"
# Vulkan device index, as `hush doctor` lists them. Unset lets the runtime pick,
# which may be an integrated GPU or the card running your LLM.
# gpu_device = 1

[normalizer]
# llama-cpp | http | rules. Dictation is rules-only until the language model has loaded,
# and falls back to the rules whenever the model fails or its output is rejected.
kind = "llama-cpp"
# Model id from the download manifest; `hush doctor` downloads it.
model = "qwen3-4b-instruct-2507-q4_k_m"
# Vulkan device index, as `hush doctor` lists them. Unset uses engine.gpu_device.
# gpu_device = 0
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
    /// Vulkan device index.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub gpu_device: Option<usize>,
}

impl Default for EngineChoice {
    fn default() -> Self {
        Self {
            model: "parakeet-tdt-0.6b-v3-f16-gguf".into(),
            gpu: GpuPolicy::default(),
            gpu_device: None,
        }
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
        /// Vulkan device index; unset means the speech engine's device.
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
            gpu_device: None,
            timeout_ms: default_llm_timeout_ms(),
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
        c.engine.gpu_device = Some(1);
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
                gpu_device: None,
                timeout_ms: 5000,
            }
        );
        let c = Config::from_toml("[normalizer]\nkind = \"llama-cpp\"\n").unwrap();
        assert_eq!(c.normalizer, NormalizerChoice::default());
        let c = Config {
            normalizer: NormalizerChoice::LlamaCpp {
                model: "other-model".into(),
                gpu_device: Some(1),
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
    fn vocabulary_file_and_default_style_round_trip() {
        let c = Config::from_toml("vocabulary_file = 'C:/x/words.txt'\ndefault_style = 'code'\n")
            .unwrap();
        assert_eq!(c.vocabulary_file, Some(PathBuf::from("C:/x/words.txt")));
        assert_eq!(c.default_style, Style::Code);
        assert_eq!(Config::from_toml(&c.to_toml().unwrap()).unwrap(), c);
    }
}
