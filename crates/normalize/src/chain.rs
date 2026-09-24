//! Ordered fallback over normalizers, always ending with the rule pass.

use std::time::Instant;

use hush_core::normalize::{
    NormalizeError, NormalizeOutput, NormalizeRequest, Normalizer, Provenance, Style,
};

use crate::rules::{self, RuleNormalizer};

/// At or below this many words the LLM has too little to fix to be worth its latency.
const MIN_WORDS_FOR_LLM: usize = 4;

pub struct NormalizerChain {
    stages: Vec<Box<dyn Normalizer>>,
    rules: RuleNormalizer,
}

impl NormalizerChain {
    /// The rule pass is appended implicitly.
    pub fn new(stages: Vec<Box<dyn Normalizer>>) -> Self {
        Self {
            stages,
            rules: RuleNormalizer,
        }
    }
}

impl Normalizer for NormalizerChain {
    fn id(&self) -> &str {
        "chain"
    }

    /// One failing stage must not stop the others from loading.
    fn warm(&mut self) -> Result<(), NormalizeError> {
        let mut first_err = None;
        for s in &mut self.stages {
            if let Err(e) = s.warm() {
                tracing::warn!(stage = s.id(), error = %e, "normalizer warm-up failed");
                first_err.get_or_insert(e);
            }
        }
        first_err.map_or(Ok(()), Err)
    }

    fn normalize(&mut self, req: &NormalizeRequest<'_>) -> Result<NormalizeOutput, NormalizeError> {
        let start = Instant::now();
        let mut failed = None;
        if req.app.style != Style::None && should_use_llm(req.transcript) {
            for s in &mut self.stages {
                match s.normalize(req) {
                    Ok(mut out) => {
                        out.elapsed = start.elapsed();
                        return Ok(out);
                    }
                    // Rule-pass text would go nowhere. A deadline falls through instead,
                    // because the rule pass is instant.
                    Err(NormalizeError::Cancelled) => return Err(NormalizeError::Cancelled),
                    Err(e) => {
                        tracing::warn!(stage = s.id(), error = %e, "normalizer failed; falling back");
                        failed.get_or_insert_with(|| Provenance::LlmFailed {
                            stage: s.id().to_string(),
                            error: e.to_string(),
                        });
                    }
                }
            }
        }
        let mut out = self.rules.normalize(req)?;
        if let Some(p) = failed {
            out.provenance = p;
        }
        out.elapsed = start.elapsed();
        Ok(out)
    }
}

pub fn should_use_llm(transcript: &str) -> bool {
    let t = transcript.trim();
    if t.split_whitespace().count() <= MIN_WORDS_FOR_LLM {
        return false;
    }
    rules::has_disfluency(t) || !well_formed(t)
}

fn well_formed(t: &str) -> bool {
    let starts_upper = t
        .chars()
        .find(|c| c.is_alphabetic())
        .is_some_and(char::is_uppercase);
    let ends_final = t
        .trim_end_matches(['"', '\'', ')', '”', '’'])
        .ends_with(['.', '?', '!']);
    starts_upper && ends_final
}

#[cfg(test)]
mod tests {
    use super::*;
    use hush_core::normalize::{AppContext, Scores};
    use hush_core::{CancelToken, UtteranceId};
    use std::sync::Arc;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::Duration;

    #[derive(Clone, Copy)]
    enum Behaviour {
        Ok(&'static str),
        Down,
        Cancelled,
    }

    struct Fake {
        behaviour: Behaviour,
        calls: AtomicUsize,
    }

    impl Fake {
        fn new(behaviour: Behaviour) -> Arc<Self> {
            Arc::new(Self {
                behaviour,
                calls: AtomicUsize::new(0),
            })
        }
    }

    struct Stage(Arc<Fake>);

    impl Normalizer for Stage {
        fn id(&self) -> &str {
            "fake"
        }
        fn normalize(
            &mut self,
            req: &NormalizeRequest<'_>,
        ) -> Result<NormalizeOutput, NormalizeError> {
            self.0.calls.fetch_add(1, Ordering::SeqCst);
            match self.0.behaviour {
                Behaviour::Ok(s) => Ok(NormalizeOutput {
                    utterance: req.utterance,
                    text: s.into(),
                    provenance: Provenance::Llm {
                        model: "fake".into(),
                        scores: Scores::default(),
                    },
                    elapsed: Duration::ZERO,
                }),
                Behaviour::Down => Err(NormalizeError::Unavailable("down".into())),
                Behaviour::Cancelled => Err(NormalizeError::Cancelled),
            }
        }
    }

    fn try_run(
        chain: &mut NormalizerChain,
        text: &str,
        style: Style,
    ) -> Result<NormalizeOutput, NormalizeError> {
        let app = AppContext {
            style,
            ..Default::default()
        };
        chain.normalize(&NormalizeRequest {
            transcript: text,
            language: Some("en"),
            vocabulary: &[],
            app: &app,
            previous: None,
            utterance: UtteranceId(9),
            cancel: CancelToken::new(),
        })
    }

    fn run(chain: &mut NormalizerChain, text: &str, style: Style) -> NormalizeOutput {
        try_run(chain, text, style).unwrap()
    }

    const MESSY: &str = "um so I was thinking we could go";

    #[test]
    fn first_success_wins() {
        let a = Fake::new(Behaviour::Ok("A"));
        let b = Fake::new(Behaviour::Ok("B"));
        let mut chain = NormalizerChain::new(vec![Box::new(Stage(a)), Box::new(Stage(b.clone()))]);
        let out = run(&mut chain, MESSY, Style::Formal);
        assert_eq!(out.text, "A");
        assert_eq!(out.utterance, UtteranceId(9));
        assert_eq!(b.calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn failures_fall_through_to_rules_and_say_so() {
        let a = Fake::new(Behaviour::Down);
        let mut chain = NormalizerChain::new(vec![Box::new(Stage(a.clone()))]);
        let out = run(&mut chain, MESSY, Style::Formal);
        assert_eq!(
            out.provenance,
            Provenance::LlmFailed {
                stage: "fake".into(),
                error: "backend unavailable: down".into()
            }
        );
        assert_eq!(out.text, "So I was thinking we could go.");
        assert_eq!(out.utterance, UtteranceId(9));
        assert_eq!(a.calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn cancellation_is_not_papered_over_by_the_rule_pass() {
        let a = Fake::new(Behaviour::Cancelled);
        let b = Fake::new(Behaviour::Ok("B"));
        let mut chain = NormalizerChain::new(vec![Box::new(Stage(a)), Box::new(Stage(b.clone()))]);
        assert!(matches!(
            try_run(&mut chain, MESSY, Style::Formal),
            Err(NormalizeError::Cancelled)
        ));
        assert_eq!(b.calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn style_none_and_clean_text_skip_the_llm() {
        let a = Fake::new(Behaviour::Ok("A"));
        let mut chain = NormalizerChain::new(vec![Box::new(Stage(a.clone()))]);
        assert_eq!(
            run(&mut chain, MESSY, Style::None).provenance,
            Provenance::Rules
        );
        assert_eq!(
            run(
                &mut chain,
                "This sentence is already perfectly fine.",
                Style::Formal
            )
            .provenance,
            Provenance::Rules
        );
        assert_eq!(a.calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn llm_is_skipped_for_short_or_well_formed_text() {
        assert!(!should_use_llm("um yes"));
        assert!(!should_use_llm("uh okay sure thing"));
        assert!(!should_use_llm("This sentence is already perfectly fine."));
        assert!(!should_use_llm("He said \"it is fine.\""));
        assert!(should_use_llm("um I think this sentence is fine."));
        assert!(should_use_llm("Let's meet Tuesday, no wait, Wednesday."));
        assert!(should_use_llm("this one has no punctuation at all"));
        assert!(should_use_llm("This one does not end properly"));
    }
}
