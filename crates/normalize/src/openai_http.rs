//! LLM cleanup through a local HTTP inference server.

use std::time::{Duration, Instant};

use hush_core::normalize::{NormalizeError, NormalizeOutput, NormalizeRequest, Normalizer};
use serde::{Deserialize, Serialize};

use crate::llm::{self, Attempt};
use crate::prompt::{self, ChatMessage};
use crate::rules::RuleNormalizer;

pub const DEFAULT_TIMEOUT: Duration = Duration::from_secs(5);
/// Loading a 4B model from disk takes several seconds; only `warm` gets this long.
pub const DEFAULT_WARM_TIMEOUT: Duration = Duration::from_secs(120);
pub const DEFAULT_KEEP_ALIVE: &str = "30m";
const OLLAMA_PORT: &str = ":11434";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Dialect {
    OpenAi,
    /// Ollama's OpenAI endpoint ignores `think`, and `keep_alive` on a cold load
    /// (measured on Ollama 0.34, 2026-09-24), so it gets its native `/api/chat`.
    Ollama,
}

#[derive(Debug, Clone)]
pub struct HttpConfig {
    /// OpenAI-style base, e.g. `http://localhost:11434/v1`.
    pub base_url: String,
    pub model: String,
    pub timeout: Duration,
    pub warm_timeout: Duration,
    pub dialect: Dialect,
    pub keep_alive: String,
}

impl HttpConfig {
    /// `localhost` becomes `127.0.0.1`: Windows tries `::1` first, Ollama listens on IPv4
    /// only, and each new connection waited ~2 s for that to fail (measured 2026-09-24).
    pub fn new(base_url: impl Into<String>, model: impl Into<String>) -> Self {
        let base_url = base_url
            .into()
            .trim_end_matches('/')
            .replacen("://localhost:", "://127.0.0.1:", 1)
            .replacen("://localhost/", "://127.0.0.1/", 1);
        let dialect = if base_url.contains(OLLAMA_PORT) {
            Dialect::Ollama
        } else {
            Dialect::OpenAi
        };
        Self {
            base_url,
            model: model.into(),
            timeout: DEFAULT_TIMEOUT,
            warm_timeout: DEFAULT_WARM_TIMEOUT,
            dialect,
            keep_alive: DEFAULT_KEEP_ALIVE.to_string(),
        }
    }

    fn endpoint(&self) -> String {
        match self.dialect {
            Dialect::OpenAi => format!("{}/chat/completions", self.base_url),
            Dialect::Ollama => {
                let root = self.base_url.strip_suffix("/v1").unwrap_or(&self.base_url);
                format!("{root}/api/chat")
            }
        }
    }
}

pub struct OpenAiHttpNormalizer {
    cfg: HttpConfig,
    id: String,
    agent: ureq::Agent,
    rules: RuleNormalizer,
}

impl OpenAiHttpNormalizer {
    pub fn new(cfg: HttpConfig) -> Self {
        Self {
            id: format!("http:{}", cfg.model),
            // Every request sets its own timeout from the caller's deadline.
            agent: ureq::Agent::config_builder()
                .http_status_as_error(false)
                .build()
                .into(),
            cfg,
            rules: RuleNormalizer,
        }
    }

    pub fn config(&self) -> &HttpConfig {
        &self.cfg
    }

    /// A rejected answer is `Ok` with the verdict in it. The blocking call cannot be
    /// interrupted, so its deadline-capped timeout bounds how long a cancel goes unnoticed.
    pub fn attempt(&self, req: &NormalizeRequest<'_>) -> Result<Attempt, NormalizeError> {
        let timeout = req.cancel.timeout_within(self.cfg.timeout)?;
        let rule_text = self.rules.clean_request(req);
        let messages = prompt::build_messages(req, &rule_text);
        let start = Instant::now();
        let reply = self.chat(&messages, max_tokens(&rule_text), timeout)?;
        let llm_elapsed = start.elapsed();
        if req.cancel.is_cancelled() {
            return Err(NormalizeError::Cancelled);
        }
        let attempt = llm::judge(req, rule_text, &reply.content, reply.truncated, llm_elapsed);
        tracing::debug!(
            model = %self.cfg.model,
            ms = llm_elapsed.as_millis() as u64,
            prompt_tokens = reply.prompt_tokens,
            cached_tokens = reply.cached_tokens,
            verdict = ?attempt.verdict,
            "llm normalize"
        );
        Ok(attempt)
    }

    fn chat(
        &self,
        messages: &[ChatMessage],
        max_tokens: u32,
        timeout: Duration,
    ) -> Result<Reply, NormalizeError> {
        let body = match self.cfg.dialect {
            Dialect::OpenAi => serde_json::to_string(&OpenAiBody {
                model: &self.cfg.model,
                messages,
                temperature: 0.0,
                max_tokens,
                stream: false,
            }),
            Dialect::Ollama => serde_json::to_string(&OllamaBody {
                model: &self.cfg.model,
                messages,
                stream: false,
                think: false,
                keep_alive: &self.cfg.keep_alive,
                options: OllamaOptions {
                    temperature: 0.0,
                    num_predict: max_tokens,
                },
            }),
        }
        .map_err(|e| NormalizeError::Request(e.to_string()))?;

        let mut resp = self
            .agent
            .post(&self.cfg.endpoint())
            .config()
            .timeout_global(Some(timeout))
            .build()
            .header("Content-Type", "application/json")
            .send(body.as_bytes())
            .map_err(map_ureq)?;
        let status = resp.status();
        let text = resp.body_mut().read_to_string().map_err(map_ureq)?;
        if !status.is_success() {
            return Err(NormalizeError::Unavailable(format!(
                "HTTP {status} from {}: {}",
                self.cfg.endpoint(),
                text.chars().take(300).collect::<String>()
            )));
        }
        parse_reply(self.cfg.dialect, &text)
    }
}

impl Normalizer for OpenAiHttpNormalizer {
    fn id(&self) -> &str {
        &self.id
    }

    /// Also primes the server's prompt cache with the system prompt.
    fn warm(&mut self) -> Result<(), NormalizeError> {
        let app = hush_core::normalize::AppContext::default();
        let req = NormalizeRequest {
            transcript: "ok",
            language: None,
            vocabulary: &[],
            app: &app,
            previous: None,
            utterance: hush_core::UtteranceId::default(),
            cancel: hush_core::CancelToken::new(),
        };
        let messages = prompt::build_messages(&req, "ok");
        self.chat(&messages, 1, self.cfg.warm_timeout).map(|_| ())
    }

    fn normalize(&mut self, req: &NormalizeRequest<'_>) -> Result<NormalizeOutput, NormalizeError> {
        let start = Instant::now();
        let a = self.attempt(req)?;
        Ok(llm::into_output(req, a, &self.cfg.model, start))
    }
}

/// A server does not expose its tokenizer, so the input's token count is estimated at 1.3
/// per word.
pub fn max_tokens(text: &str) -> u32 {
    llm::token_cap((text.split_whitespace().count() as f64 * 1.3).ceil() as usize)
}

/// Every request timeout is either the caller's deadline or the configured cap standing in
/// for it, so both surface as `Deadline`.
fn map_ureq(e: ureq::Error) -> NormalizeError {
    match e {
        ureq::Error::Timeout(_) => NormalizeError::Deadline,
        other => NormalizeError::Unavailable(other.to_string()),
    }
}

#[derive(Serialize)]
struct OpenAiBody<'a> {
    model: &'a str,
    messages: &'a [ChatMessage],
    temperature: f32,
    max_tokens: u32,
    stream: bool,
}

#[derive(Serialize)]
struct OllamaBody<'a> {
    model: &'a str,
    messages: &'a [ChatMessage],
    stream: bool,
    think: bool,
    keep_alive: &'a str,
    options: OllamaOptions,
}

#[derive(Serialize)]
struct OllamaOptions {
    temperature: f32,
    num_predict: u32,
}

#[derive(Debug, PartialEq)]
struct Reply {
    content: String,
    truncated: bool,
    prompt_tokens: Option<u64>,
    cached_tokens: Option<u64>,
}

#[derive(Deserialize)]
struct OpenAiResponse {
    choices: Vec<OpenAiChoice>,
    usage: Option<OpenAiUsage>,
}

#[derive(Deserialize)]
struct OpenAiChoice {
    message: OpenAiMessage,
    finish_reason: Option<String>,
}

#[derive(Deserialize)]
struct OpenAiMessage {
    content: Option<String>,
}

#[derive(Deserialize)]
struct OpenAiUsage {
    prompt_tokens: Option<u64>,
    prompt_tokens_details: Option<OpenAiPromptDetails>,
}

#[derive(Deserialize)]
struct OpenAiPromptDetails {
    cached_tokens: Option<u64>,
}

#[derive(Deserialize)]
struct OllamaResponse {
    message: OpenAiMessage,
    done_reason: Option<String>,
    prompt_eval_count: Option<u64>,
    prompt_eval_cached_count: Option<u64>,
}

fn parse_reply(dialect: Dialect, text: &str) -> Result<Reply, NormalizeError> {
    let bad = |e: serde_json::Error| NormalizeError::Malformed(e.to_string());
    match dialect {
        Dialect::OpenAi => {
            let r: OpenAiResponse = serde_json::from_str(text).map_err(bad)?;
            let choice = r
                .choices
                .into_iter()
                .next()
                .ok_or_else(|| NormalizeError::Malformed("response has no choices".into()))?;
            Ok(Reply {
                content: choice.message.content.unwrap_or_default(),
                truncated: choice.finish_reason.as_deref() == Some("length"),
                prompt_tokens: r.usage.as_ref().and_then(|u| u.prompt_tokens),
                cached_tokens: r
                    .usage
                    .and_then(|u| u.prompt_tokens_details)
                    .and_then(|d| d.cached_tokens),
            })
        }
        Dialect::Ollama => {
            let r: OllamaResponse = serde_json::from_str(text).map_err(bad)?;
            Ok(Reply {
                content: r.message.content.unwrap_or_default(),
                truncated: r.done_reason.as_deref() == Some("length"),
                prompt_tokens: r.prompt_eval_count,
                cached_tokens: r.prompt_eval_cached_count,
            })
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hush_core::normalize::{AppContext, Style};
    use hush_core::{CancelToken, UtteranceId};

    #[test]
    fn dialect_and_endpoint_follow_the_base_url() {
        let c = HttpConfig::new("http://localhost:11434/v1/", "m");
        assert_eq!(c.dialect, Dialect::Ollama);
        assert_eq!(c.endpoint(), "http://127.0.0.1:11434/api/chat");
        let c = HttpConfig::new("http://localhost.example:1234/v1", "m");
        assert_eq!(c.base_url, "http://localhost.example:1234/v1");
        let c = HttpConfig::new("http://127.0.0.1:8080/v1", "m");
        assert_eq!(c.dialect, Dialect::OpenAi);
        assert_eq!(c.endpoint(), "http://127.0.0.1:8080/v1/chat/completions");
        assert_eq!(c.timeout, Duration::from_secs(5));
    }

    #[test]
    fn token_budget_is_one_and_a_half_times_the_estimate_plus_slack() {
        assert_eq!(max_tokens("a b c d e f g h i j"), 40);
        assert_eq!(max_tokens(""), 20);
    }

    #[test]
    fn parses_openai_reply() {
        let r = parse_reply(
            Dialect::OpenAi,
            r#"{"choices":[{"index":0,"message":{"role":"assistant","content":"Hi."},"finish_reason":"length"}],"usage":{"prompt_tokens":12,"prompt_tokens_details":{"cached_tokens":8}}}"#,
        )
        .unwrap();
        assert_eq!(
            r,
            Reply {
                content: "Hi.".into(),
                truncated: true,
                prompt_tokens: Some(12),
                cached_tokens: Some(8),
            }
        );
    }

    #[test]
    fn parses_ollama_reply() {
        let r = parse_reply(
            Dialect::Ollama,
            r#"{"model":"m","message":{"role":"assistant","content":"Hi."},"done":true,"done_reason":"stop","prompt_eval_count":18,"prompt_eval_cached_count":17}"#,
        )
        .unwrap();
        assert!(!r.truncated);
        assert_eq!(r.content, "Hi.");
        assert_eq!(r.cached_tokens, Some(17));
    }

    #[test]
    fn malformed_reply_is_a_malformed_error() {
        assert!(matches!(
            parse_reply(Dialect::OpenAi, r#"{"choices":[]}"#),
            Err(NormalizeError::Malformed(_))
        ));
        assert!(matches!(
            parse_reply(Dialect::Ollama, "not json"),
            Err(NormalizeError::Malformed(_))
        ));
    }

    #[test]
    fn request_bodies_have_the_expected_shape() {
        let msgs = [ChatMessage {
            role: "user",
            content: "x".into(),
        }];
        let o: serde_json::Value = serde_json::to_value(OllamaBody {
            model: "m",
            messages: &msgs,
            stream: false,
            think: false,
            keep_alive: "30m",
            options: OllamaOptions {
                temperature: 0.0,
                num_predict: 7,
            },
        })
        .unwrap();
        assert_eq!(o["think"], false);
        assert_eq!(o["keep_alive"], "30m");
        assert_eq!(o["options"]["num_predict"], 7);
        assert_eq!(o["options"]["temperature"], 0.0);
        let a: serde_json::Value = serde_json::to_value(OpenAiBody {
            model: "m",
            messages: &msgs,
            temperature: 0.0,
            max_tokens: 7,
            stream: false,
        })
        .unwrap();
        assert_eq!(a["max_tokens"], 7);
        assert_eq!(a["messages"][0]["role"], "user");
    }

    fn formal() -> AppContext {
        AppContext {
            style: Style::Formal,
            ..Default::default()
        }
    }

    fn request<'a>(app: &'a AppContext, cancel: CancelToken) -> NormalizeRequest<'a> {
        NormalizeRequest {
            transcript: "um hello there my friend",
            language: None,
            vocabulary: &[],
            app,
            previous: None,
            utterance: UtteranceId(5),
            cancel,
        }
    }

    #[test]
    fn unreachable_server_is_unavailable() {
        // Port 9 (discard) is closed on any sane dev machine.
        let mut cfg = HttpConfig::new("http://127.0.0.1:9/v1", "m");
        cfg.timeout = Duration::from_secs(2);
        let mut n = OpenAiHttpNormalizer::new(cfg);
        let app = formal();
        let err = n.normalize(&request(&app, CancelToken::new())).unwrap_err();
        assert!(
            matches!(
                err,
                NormalizeError::Unavailable(_) | NormalizeError::Deadline
            ),
            "{err:?}"
        );
    }

    fn silent_server() -> (std::net::TcpListener, String) {
        let l = std::net::TcpListener::bind("127.0.0.1:0").unwrap();
        let url = format!("http://{}/v1", l.local_addr().unwrap());
        (l, url)
    }

    #[test]
    fn cancelled_token_sends_nothing() {
        let (_l, url) = silent_server();
        let mut n = OpenAiHttpNormalizer::new(HttpConfig::new(url, "m"));
        let app = formal();
        let cancel = CancelToken::new();
        cancel.cancel();
        let started = Instant::now();
        let err = n.normalize(&request(&app, cancel)).unwrap_err();
        assert!(matches!(err, NormalizeError::Cancelled), "{err:?}");
        assert!(started.elapsed() < Duration::from_millis(50));

        let expired = CancelToken::new().with_deadline(Instant::now());
        let err = n.normalize(&request(&app, expired)).unwrap_err();
        assert!(matches!(err, NormalizeError::Deadline), "{err:?}");
    }

    #[test]
    fn deadline_cuts_the_request_timeout() {
        let (_l, url) = silent_server();
        let mut n = OpenAiHttpNormalizer::new(HttpConfig::new(url, "m"));
        let app = formal();
        let started = Instant::now();
        let cancel = CancelToken::new().with_timeout(Duration::from_millis(200));
        let err = n.normalize(&request(&app, cancel)).unwrap_err();
        let took = started.elapsed();
        assert!(matches!(err, NormalizeError::Deadline), "{err:?}");
        assert!(
            took < Duration::from_secs(2),
            "configured 5 s timeout was used: {took:?}"
        );
    }

    /// Needs a running Ollama with the model named in `HUSH_TEST_OLLAMA_MODEL` pulled.
    #[test]
    fn live_ollama_round_trip() {
        let Ok(model) = std::env::var("HUSH_TEST_OLLAMA_MODEL") else {
            return;
        };
        let mut n = OpenAiHttpNormalizer::new(HttpConfig::new("http://localhost:11434/v1", model));
        n.warm().unwrap();
        let app = formal();
        let req = NormalizeRequest {
            transcript: "um what is the capital of france",
            language: Some("en"),
            ..request(&app, CancelToken::new())
        };
        let out = n.normalize(&req).unwrap();
        assert_eq!(out.utterance, UtteranceId(5));
        assert!(!out.text.contains("Paris"), "{out:?}");
    }
}
