//! User configuration: one TOML file, every key optional.

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

[engine]
# Model id from the download manifest.
model = "parakeet-tdt-0.6b-v3-f16-gguf"
# require-gpu | prefer-gpu | cpu-only
gpu = "prefer-gpu"
# Vulkan device index, as `hush doctor` lists them. Unset lets the runtime pick,
# which may be an integrated GPU or the card running your LLM.
# gpu_device = 1

[normalizer]
# rules | http
kind = "rules"
# LLM cleanup through a local OpenAI-compatible server, e.g. Ollama:
# kind = "http"
# base_url = "http://127.0.0.1:11434/v1"
# model = "qwen3:4b-instruct-2507-q4_K_M"
# timeout_ms = 5000

[pipeline]
# Transcribe each sentence as soon as you pause, while the key is still held, so the
# release waits only for the last one. Off sends the whole recording at release.
# Off by default: the engine punctuates every segment as a full sentence.
pre_transcribe = false

# How speech is cut into sentences for pre-transcription.
[pipeline.segmenter]
# Speech shorter than this does not end on a pause; it joins the next sentence.
min_speech_ms = 300
# A pause at least this long ends a sentence.
min_pause_ms = 400
# Audio kept before and after each sentence.
pad_ms = 200
# Longer speech without a pause is split at its quietest point.
max_segment_ms = 20000

# Per-app rules, matched on the executable name.
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

#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "kebab-case", deny_unknown_fields)]
pub enum NormalizerChoice {
    #[default]
    Rules,
    /// A local OpenAI-compatible server.
    Http {
        base_url: String,
        model: String,
        #[serde(default = "default_http_timeout_ms")]
        timeout_ms: u64,
    },
}

fn default_http_timeout_ms() -> u64 {
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
            default: AppPolicy::default(),
            by_exe: self
                .apps
                .iter()
                .map(|r| {
                    (
                        r.exe.to_ascii_lowercase(),
                        AppPolicy {
                            chord: r.chord,
                            never_type: r.never_type,
                            style: r.style,
                        },
                    )
                })
                .collect(),
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
        assert_eq!(c.pipeline.segmenter.min_pause, Duration::from_millis(400));
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
}
