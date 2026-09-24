//! What every LLM backend does with a raw reply: cap, strip, validate, and turn a rejection
//! into the rule-pass text.

use std::time::{Duration, Instant};

use hush_core::normalize::{
    NormalizeOutput, NormalizeRequest, Provenance, Rejection, Scores, Style,
};

use crate::validate;

#[derive(Debug, Clone)]
pub struct Attempt {
    /// What the LLM was given, and what is inserted if it is rejected.
    pub rule_text: String,
    pub candidate: String,
    pub verdict: Result<Scores, Rejection>,
    pub llm_elapsed: Duration,
}

/// A faithful cleanup is never longer than its input; the cap exists to stop a runaway
/// repetition loop.
pub fn token_cap(input_tokens: usize) -> u32 {
    (input_tokens as f64 * 1.5).ceil() as u32 + 20
}

/// Qwen3 models may emit an empty `<think></think>` pair even with thinking off.
pub fn strip_think(s: &str) -> &str {
    let t = s.trim_start();
    if let Some(rest) = t.strip_prefix("<think>")
        && let Some(end) = rest.find("</think>")
    {
        return &rest[end + "</think>".len()..];
    }
    s
}

/// `raw` is the model's reply as generated; `truncated` means it hit the token cap.
pub fn judge(
    req: &NormalizeRequest<'_>,
    rule_text: String,
    raw: &str,
    truncated: bool,
    llm_elapsed: Duration,
) -> Attempt {
    let candidate = strip_think(raw).trim().to_string();
    let verdict = if truncated {
        Err(Rejection::Truncated)
    } else {
        let mut vcfg = validate::Config::new(req.language, req.vocabulary);
        vcfg.verbatim = matches!(req.app.style, Style::Code | Style::None);
        validate::validate_with(&rule_text, &candidate, &vcfg)
    };
    Attempt {
        rule_text,
        candidate,
        verdict,
        llm_elapsed,
    }
}

/// A rejected candidate becomes the rule-pass text, with the rejection in the provenance.
pub fn into_output(
    req: &NormalizeRequest<'_>,
    attempt: Attempt,
    model: &str,
    started: Instant,
) -> NormalizeOutput {
    let model = model.to_string();
    let (text, provenance) = match attempt.verdict {
        Ok(scores) => (attempt.candidate, Provenance::Llm { model, scores }),
        Err(rejection) => {
            tracing::info!(
                check = rejection.check().as_str(),
                score = rejection.score(),
                threshold = rejection.threshold(),
                reason = %rejection,
                "llm output rejected; using rule pass"
            );
            (
                attempt.rule_text,
                Provenance::LlmRejected { model, rejection },
            )
        }
    };
    NormalizeOutput {
        utterance: req.utterance,
        text,
        provenance,
        elapsed: started.elapsed(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use hush_core::normalize::AppContext;
    use hush_core::{CancelToken, UtteranceId};

    #[test]
    fn token_cap_is_one_and_a_half_times_the_input_plus_slack() {
        assert_eq!(token_cap(0), 20);
        assert_eq!(token_cap(10), 35);
        assert_eq!(token_cap(13), 40);
        assert_eq!(token_cap(46), 89);
    }

    #[test]
    fn empty_think_block_is_stripped() {
        assert_eq!(strip_think("<think>\n\n</think>\n\nHello."), "\n\nHello.");
        assert_eq!(strip_think("Hello <think>"), "Hello <think>");
    }

    fn request(app: &AppContext) -> NormalizeRequest<'_> {
        NormalizeRequest {
            transcript: "um so what time does the store close",
            language: Some("en"),
            vocabulary: &[],
            app,
            previous: None,
            utterance: UtteranceId(4),
            cancel: CancelToken::new(),
        }
    }

    #[test]
    fn a_faithful_candidate_is_inserted() {
        let app = AppContext {
            style: Style::Formal,
            ..Default::default()
        };
        let req = request(&app);
        let a = judge(
            &req,
            "So what time does the store close.".into(),
            "<think></think>What time does the store close?",
            false,
            Duration::ZERO,
        );
        assert_eq!(a.candidate, "What time does the store close?");
        let out = into_output(&req, a, "m", Instant::now());
        assert_eq!(out.text, "What time does the store close?");
        assert_eq!(out.utterance, UtteranceId(4));
        assert!(matches!(out.provenance, Provenance::Llm { .. }));
    }

    #[test]
    fn an_answer_or_a_truncation_falls_back_to_the_rule_text() {
        let app = AppContext {
            style: Style::Formal,
            ..Default::default()
        };
        let req = request(&app);
        let rules = "So what time does the store close.";
        let answer = judge(
            &req,
            rules.into(),
            "The store closes at nine pm tonight, according to its website.",
            false,
            Duration::ZERO,
        );
        let out = into_output(&req, answer, "m", Instant::now());
        assert_eq!(out.text, rules);
        assert!(out.provenance.is_fallback());

        let cut = judge(
            &req,
            rules.into(),
            "What time does the store",
            true,
            Duration::ZERO,
        );
        assert_eq!(cut.verdict, Err(Rejection::Truncated));
    }
}
