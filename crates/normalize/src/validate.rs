//! Gate between the LLM and the target app. A check can only turn an LLM output into a
//! fallback, never alter it.

use std::collections::HashSet;

pub use wl_core::normalize::Rejection;
use wl_core::normalize::{Measured, Scores};

use crate::lang;

pub const DEFAULT_MIN_CONTAINMENT: f64 = 0.70;
pub const DEFAULT_MIN_LENGTH_RATIO: f64 = 0.3;
pub const DEFAULT_MAX_LENGTH_RATIO: f64 = 1.5;
/// Below this many source words, the length ratio and the language check are noise:
/// "ok" → "Okay." is a ratio of 1 but "yes no" → "Yes." is 0.5.
const MIN_WORDS_FOR_RATIOS: usize = 4;
const LANGUAGE_PROBE_WORDS: usize = 5;

const PREAMBLES: &[&str] = &[
    "here is",
    "here's",
    "here are",
    "sure",
    "certainly",
    "of course",
    "okay, here",
    "the cleaned",
    "cleaned text",
    "cleaned transcript",
    "hier ist",
    "gerne",
    "natürlich",
    "klar,",
    "```",
    "\"",
    "“",
    "„",
    "«",
    "'",
];

#[derive(Debug, Clone)]
pub struct Config<'a> {
    pub language: Option<&'a str>,
    /// May appear in the output without appearing in the transcript: that is what fixing
    /// "cooper netties" to "Kubernetes" looks like.
    pub vocabulary: &'a [String],
    pub min_containment: f64,
    pub min_length_ratio: f64,
    pub max_length_ratio: f64,
    pub verbatim: bool,
}

impl<'a> Config<'a> {
    pub fn new(language: Option<&'a str>, vocabulary: &'a [String]) -> Self {
        Self {
            language,
            vocabulary,
            min_containment: DEFAULT_MIN_CONTAINMENT,
            min_length_ratio: DEFAULT_MIN_LENGTH_RATIO,
            max_length_ratio: DEFAULT_MAX_LENGTH_RATIO,
            verbatim: false,
        }
    }
}

const NUMBER_WORDS: &[&str] = &[
    "zero",
    "one",
    "two",
    "three",
    "four",
    "five",
    "six",
    "seven",
    "eight",
    "nine",
    "ten",
    "eleven",
    "twelve",
    "thirteen",
    "fourteen",
    "fifteen",
    "sixteen",
    "seventeen",
    "eighteen",
    "nineteen",
    "twenty",
    "thirty",
    "forty",
    "fifty",
    "sixty",
    "seventy",
    "eighty",
    "ninety",
    "hundred",
    "thousand",
    "million",
    "billion",
    "null",
    "eins",
    "zwei",
    "drei",
    "vier",
    "fünf",
    "sechs",
    "sieben",
    "acht",
    "neun",
    "zehn",
    "elf",
    "zwölf",
    "zwanzig",
    "dreißig",
    "vierzig",
    "fünfzig",
    "hundert",
    "tausend",
    "milliarde",
];

fn is_number_word(w: &str) -> bool {
    w.chars().all(|c| c.is_ascii_digit()) || NUMBER_WORDS.contains(&w)
}

pub fn validate(source: &str, candidate: &str, lang: Option<&str>) -> Result<Scores, Rejection> {
    validate_with(source, candidate, &Config::new(lang, &[]))
}

pub fn validate_with(source: &str, candidate: &str, cfg: &Config<'_>) -> Result<Scores, Rejection> {
    let mut scores = Scores::default();
    let cand = candidate.trim();
    let src = source.trim();
    let cand_words = words(cand);
    if cand.is_empty() || cand_words.is_empty() {
        return Err(Rejection::Empty);
    }

    let cand_lower = cand.to_lowercase();
    let src_lower = src.to_lowercase();
    if let Some(p) = PREAMBLES
        .iter()
        .find(|p| cand_lower.starts_with(**p) && !src_lower.starts_with(**p))
    {
        return Err(Rejection::Preamble((*p).to_string()));
    }

    // Line breaks the speaker dictated survive the rule pass; any extra one is the model
    // appending a note, an explanation or a second version.
    if cand.matches('\n').count() > src.matches('\n').count() {
        return Err(Rejection::TrailingExplanation);
    }

    let src_words = words(src);
    if src_words.len() >= MIN_WORDS_FOR_RATIOS {
        let ratio = cand_words.len() as f64 / src_words.len() as f64;
        if !(cfg.min_length_ratio..=cfg.max_length_ratio).contains(&ratio) {
            return Err(Rejection::LengthRatio {
                ratio,
                min: cfg.min_length_ratio,
                max: cfg.max_length_ratio,
            });
        }
        scores.length_ratio = Some(Measured {
            value: ratio,
            min: cfg.min_length_ratio,
            max: Some(cfg.max_length_ratio),
        });
    }

    let vocab: HashSet<String> = cfg.vocabulary.iter().flat_map(|v| words(v)).collect();
    let src_set: HashSet<&str> = src_words.iter().map(String::as_str).collect();
    let counted: Vec<&String> = cand_words
        .iter()
        .filter(|w| !vocab.contains(*w) && !lang::is_filler(cfg.language, w))
        .collect();
    if !counted.is_empty() {
        let found = counted
            .iter()
            .filter(|w| src_set.contains(w.as_str()))
            .count();
        let ratio = found as f64 / counted.len() as f64;
        if ratio < cfg.min_containment {
            return Err(Rejection::LowContainment {
                ratio,
                threshold: cfg.min_containment,
            });
        }
        scores.containment = Some(Measured {
            value: ratio,
            min: cfg.min_containment,
            max: None,
        });
    }

    if let Some(n) = cand_words
        .iter()
        .find(|w| is_number_word(w) && !src_set.contains(w.as_str()) && !vocab.contains(*w))
    {
        return Err(Rejection::NumberChanged(n.clone()));
    }

    if cfg.verbatim {
        let src_tokens: HashSet<&str> = src.split_whitespace().collect();
        let vocab_tokens: HashSet<&str> = cfg
            .vocabulary
            .iter()
            .flat_map(|v| v.split_whitespace())
            .collect();
        if let Some(t) = cand
            .split_whitespace()
            .find(|t| !src_tokens.contains(t) && !vocab_tokens.contains(t))
        {
            return Err(Rejection::NotVerbatim(t.to_string()));
        }
    }

    if src_words.len() >= MIN_WORDS_FOR_RATIOS {
        let mut longest: Vec<&str> = src_set
            .iter()
            .copied()
            .filter(|w| !vocab.contains(*w))
            .collect();
        longest.sort_by(|a, b| b.chars().count().cmp(&a.chars().count()).then(a.cmp(b)));
        longest.truncate(LANGUAGE_PROBE_WORDS);
        let cand_set: HashSet<&str> = cand_words.iter().map(String::as_str).collect();
        if !longest.is_empty() && !longest.iter().any(|w| cand_set.contains(w)) {
            return Err(Rejection::LanguageChanged);
        }
    }
    Ok(scores)
}

/// Apostrophes split words on both sides alike, so "don't" compares as "don" + "t".
fn words(s: &str) -> Vec<String> {
    s.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(str::to_lowercase)
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn v(s: &str, c: &str) -> Result<(), Rejection> {
        validate(s, c, Some("en")).map(drop)
    }

    fn vw(s: &str, c: &str, cfg: &Config<'_>) -> Result<(), Rejection> {
        validate_with(s, c, cfg).map(drop)
    }

    #[test]
    fn a_pass_reports_its_margins() {
        let s = validate(
            "um so what time is the uh meeting tomorrow no wait on friday",
            "What time is the meeting on Friday?",
            Some("en"),
        )
        .unwrap();
        let c = s.containment.unwrap();
        assert_eq!(c.value, 1.0);
        assert_eq!(c.min, DEFAULT_MIN_CONTAINMENT);
        let r = s.length_ratio.unwrap();
        assert!(r.margin() > 0.0);
        assert_eq!(r.max, Some(DEFAULT_MAX_LENGTH_RATIO));
        assert_eq!(validate("yes no", "Yes.", None).unwrap().length_ratio, None);
    }

    #[test]
    fn a_rejection_reports_check_score_and_threshold() {
        let r = validate(
            "what is the capital of france",
            "The capital of France is Paris, a city.",
            Some("en"),
        )
        .unwrap_err();
        assert_eq!(r.check(), wl_core::normalize::Check::Containment);
        assert!(r.score().unwrap() < DEFAULT_MIN_CONTAINMENT);
        assert_eq!(r.threshold(), Some(DEFAULT_MIN_CONTAINMENT));
    }

    #[test]
    fn faithful_cleanup_passes() {
        assert_eq!(
            v(
                "um so what time is the uh meeting tomorrow no wait on friday",
                "What time is the meeting on Friday?"
            ),
            Ok(())
        );
        assert_eq!(v("git status", "git status"), Ok(()));
    }

    #[test]
    fn empty_output_is_rejected() {
        assert_eq!(v("hello there", ""), Err(Rejection::Empty));
        assert_eq!(v("hello there", "  ...  "), Err(Rejection::Empty));
    }

    #[test]
    fn preamble_is_rejected() {
        assert!(matches!(
            v("send the file", "Here is the cleaned text: Send the file."),
            Err(Rejection::Preamble(_))
        ));
        assert!(matches!(
            v("send the file", "Sure! Send the file."),
            Err(Rejection::Preamble(_))
        ));
        assert!(matches!(
            v("send the file", "\"Send the file.\""),
            Err(Rejection::Preamble(_))
        ));
        assert!(matches!(
            v("send the file", "```\nSend the file.\n```"),
            Err(Rejection::Preamble(_))
        ));
    }

    #[test]
    fn preamble_the_speaker_said_is_allowed() {
        assert_eq!(
            v("sure let's meet at noon", "Sure, let's meet at noon."),
            Ok(())
        );
        assert_eq!(v("\"quoted\" he said", "\"Quoted,\" he said."), Ok(()));
    }

    #[test]
    fn trailing_explanation_is_rejected() {
        assert_eq!(
            v(
                "what's the capital of france",
                "What's the capital of France?\n\nNote: I did not answer the question."
            ),
            Err(Rejection::TrailingExplanation)
        );
    }

    #[test]
    fn dictated_line_breaks_are_allowed() {
        assert_eq!(
            v(
                "Dear team,\n\nthe release is ready",
                "Dear team,\n\nThe release is ready."
            ),
            Ok(())
        );
    }

    #[test]
    fn length_ratio_out_of_bounds_is_rejected() {
        assert!(matches!(
            v(
                "write a poem about the sea",
                "Write a poem about the sea the sea the sea is wide and the sea is deep"
            ),
            Err(Rejection::LengthRatio { .. })
        ));
        assert!(matches!(
            v(
                "so the thing I wanted to say about the budget is that we are over it",
                "Over budget."
            ),
            Err(Rejection::LengthRatio { .. })
        ));
    }

    #[test]
    fn answer_is_rejected_by_containment() {
        assert!(matches!(
            v(
                "what is the capital of france",
                "The capital of France is Paris, a city."
            ),
            Err(Rejection::LowContainment { .. })
        ));
        assert!(matches!(
            v(
                "can you write me a haiku about autumn",
                "Leaves fall softly down, golden whispers in the wind"
            ),
            Err(Rejection::LowContainment { .. })
        ));
    }

    #[test]
    fn vocabulary_and_fillers_do_not_count_against_containment() {
        let vocab = vec!["Kubernetes".to_string(), "gRPC".to_string()];
        let cfg = Config::new(Some("en"), &vocab);
        assert_eq!(
            vw(
                "we run cooper netties with g r p c",
                "We run Kubernetes with gRPC.",
                &cfg
            ),
            Ok(())
        );
        assert!(
            validate_with(
                "we run cooper netties with g r p c",
                "We run Kubernetes with gRPC.",
                &Config::new(Some("en"), &[])
            )
            .is_err()
        );
    }

    #[test]
    fn translation_is_rejected() {
        let src = "ich glaube wir sollten das Meeting auf Donnerstag verschieben";
        let out = "I think we should move the meeting to Thursday.";
        let r = validate(src, out, Some("de"));
        assert!(
            matches!(
                r,
                Err(Rejection::LowContainment { .. } | Rejection::LanguageChanged)
            ),
            "{r:?}"
        );
    }

    #[test]
    fn language_check_fires_on_its_own() {
        let mut cfg = Config::new(Some("de"), &[]);
        cfg.min_containment = 0.0;
        let r = validate_with(
            "wir verschieben das Meeting auf Donnerstag",
            "we postpone the appointment to thursday",
            &cfg,
        );
        assert_eq!(r, Err(Rejection::LanguageChanged));
    }

    #[test]
    fn changed_number_is_rejected() {
        let src = "The meeting is at three no wait four thirty in room b.";
        assert_eq!(
            v(src, "The meeting is at three forty five in room B."),
            Err(Rejection::NumberChanged("forty".into()))
        );
        assert_eq!(
            v("what is 17 times 3", "What is 17 times 3? 51."),
            Err(Rejection::NumberChanged("51".into()))
        );
        assert_eq!(v(src, "The meeting is at four thirty in room B."), Ok(()));
        assert_eq!(v("meet at 10 tomorrow", "Meet at 10 tomorrow."), Ok(()));
    }

    #[test]
    fn code_style_rejects_any_rewrite() {
        let mut cfg = Config::new(Some("en"), &[]);
        cfg.verbatim = true;
        let src = "git commit dash m fix typo";
        assert_eq!(vw(src, "git commit dash m fix typo", &cfg), Ok(()));
        assert_eq!(vw(src, "git commit dash m typo", &cfg), Ok(()));
        assert_eq!(
            validate_with(src, "Git commit dash m fix typo", &cfg),
            Err(Rejection::NotVerbatim("Git".into()))
        );
        assert_eq!(
            validate_with(src, "git commit dash m fix typo.", &cfg),
            Err(Rejection::NotVerbatim("typo.".into()))
        );
        assert_eq!(v(src, "Git commit dash m fix typo."), Ok(()));
    }

    #[test]
    fn short_inputs_skip_ratio_checks() {
        assert_eq!(v("yes no", "Yes."), Ok(()));
    }
}
