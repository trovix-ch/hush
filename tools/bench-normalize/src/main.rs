//! Runs fixture transcripts through the rule pass and an HTTP LLM normalizer, and
//! reports outputs, validation verdicts, leaks of forbidden text, and latency.

use std::path::PathBuf;
use std::time::{Duration, Instant};

use anyhow::{Context, Result, bail};
use serde::Deserialize;
use wl_core::normalize::{AppContext, NormalizeRequest, Normalizer, Style};
use wl_normalize::openai_http::{Attempt, HttpConfig, OpenAiHttpNormalizer};
use wl_normalize::{rules, should_use_llm};

const USAGE: &str = "usage: bench-normalize [--base-url URL] [--model NAME]... [--runs N] \
[--fixtures PATH] [--timeout-ms MS] [--case SUBSTRING] [--rules-only]

  --base-url   OpenAI-style base URL (default http://localhost:11434/v1)
  --model      model name; repeat or comma-separate for several (default
               qwen3:4b-instruct-2507-q4_K_M)
  --runs       LLM runs per case, for latency percentiles (default 3)
  --fixtures   fixtures TOML (default: the one shipped with this tool)
  --timeout-ms per-request timeout (default 5000)
  --case       only run cases whose name contains this
  --rules-only skip the LLM";

struct Args {
    base_url: String,
    models: Vec<String>,
    runs: usize,
    fixtures: PathBuf,
    timeout: Duration,
    case_filter: Option<String>,
    rules_only: bool,
}

fn parse_args() -> Result<Args> {
    let mut a = Args {
        base_url: "http://localhost:11434/v1".into(),
        models: Vec::new(),
        runs: 3,
        fixtures: PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/transcripts.toml"),
        timeout: Duration::from_secs(5),
        case_filter: None,
        rules_only: false,
    };
    let mut it = std::env::args().skip(1);
    while let Some(flag) = it.next() {
        let mut value = || {
            it.next()
                .with_context(|| format!("{flag} needs a value\n{USAGE}"))
        };
        match flag.as_str() {
            "--base-url" => a.base_url = value()?,
            "--model" => a.models.extend(
                value()?
                    .split(',')
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(String::from),
            ),
            "--runs" => a.runs = value()?.parse().context("--runs")?,
            "--fixtures" => a.fixtures = value()?.into(),
            "--timeout-ms" => {
                a.timeout = Duration::from_millis(value()?.parse().context("--timeout-ms")?)
            }
            "--case" => a.case_filter = Some(value()?),
            "--rules-only" => a.rules_only = true,
            "-h" | "--help" => {
                println!("{USAGE}");
                std::process::exit(0);
            }
            other => bail!("unknown argument {other}\n{USAGE}"),
        }
    }
    if a.models.is_empty() {
        a.models.push("qwen3:4b-instruct-2507-q4_K_M".into());
    }
    if a.runs == 0 {
        bail!("--runs must be at least 1");
    }
    Ok(a)
}

#[derive(Debug, Deserialize)]
struct Fixtures {
    #[serde(rename = "case")]
    cases: Vec<Case>,
}

#[derive(Debug, Deserialize)]
struct Case {
    name: String,
    input: String,
    expected: Option<String>,
    language: Option<String>,
    #[serde(default)]
    style: Style,
    #[serde(default)]
    vocabulary: Vec<String>,
    /// Case-sensitive substrings that must not reach the target app: the answer to a
    /// dictated question, the output of a dictated command, prose punctuation in code.
    #[serde(default)]
    must_not_contain: Vec<String>,
    previous: Option<String>,
    app: Option<String>,
}

impl Case {
    fn with_request<T>(&self, f: impl FnOnce(&NormalizeRequest<'_>) -> T) -> T {
        let app = AppContext {
            exe: self.app.clone(),
            window_title: None,
            style: self.style,
        };
        f(&NormalizeRequest {
            transcript: &self.input,
            language: self.language.as_deref(),
            vocabulary: &self.vocabulary,
            app: &app,
            previous: self.previous.as_deref(),
            utterance: wl_core::UtteranceId::FIRST,
            cancel: wl_core::CancelToken::new(),
        })
    }

    fn violations(&self, text: &str) -> Vec<&str> {
        self.must_not_contain
            .iter()
            .filter(|s| text.contains(s.as_str()))
            .map(String::as_str)
            .collect()
    }

    fn matches_expected(&self, text: &str) -> Option<bool> {
        self.expected.as_ref().map(|e| words(e) == words(text))
    }
}

fn words(s: &str) -> Vec<String> {
    s.split(|c: char| !c.is_alphanumeric())
        .filter(|w| !w.is_empty())
        .map(str::to_lowercase)
        .collect()
}

/// Nearest-rank percentile.
fn percentile(sorted: &[Duration], q: f64) -> Duration {
    if sorted.is_empty() {
        return Duration::ZERO;
    }
    let rank = ((q * sorted.len() as f64).ceil() as usize).clamp(1, sorted.len());
    sorted[rank - 1]
}

fn ms(d: Duration) -> String {
    format!("{:.0}", d.as_secs_f64() * 1000.0)
}

#[derive(Default)]
struct Summary {
    model: String,
    cases: usize,
    validated: usize,
    rejected: usize,
    errors: usize,
    raw_leaks: usize,
    final_leaks: usize,
    expected_total: usize,
    expected_llm: usize,
    expected_final: usize,
    unstable: usize,
    latencies: Vec<Duration>,
    warm_error: Option<String>,
}

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("warn")),
        )
        .with_writer(std::io::stderr)
        .init();
    let args = parse_args()?;
    let raw = std::fs::read_to_string(&args.fixtures)
        .with_context(|| format!("reading {}", args.fixtures.display()))?;
    let fixtures: Fixtures = toml::from_str(&raw).context("parsing fixtures")?;
    let cases: Vec<&Case> = fixtures
        .cases
        .iter()
        .filter(|c| {
            args.case_filter
                .as_ref()
                .is_none_or(|f| c.name.contains(f.as_str()))
        })
        .collect();
    println!(
        "# bench-normalize: {} cases, {} runs each\n",
        cases.len(),
        args.runs
    );

    let rules_summary = run_rules(&cases);
    let mut summaries = Vec::new();
    if !args.rules_only {
        for model in &args.models {
            summaries.push(run_model(&args, model, &cases));
        }
    }
    print_summary(&rules_summary, &summaries);
    Ok(())
}

fn run_rules(cases: &[&Case]) -> Summary {
    println!("## rules\n");
    let mut s = Summary {
        model: "rules only".into(),
        ..Default::default()
    };
    for c in cases {
        let start = Instant::now();
        let out = c.with_request(|r| rules::RuleNormalizer.clean_request(r));
        let took = start.elapsed();
        s.cases += 1;
        s.latencies.push(took);
        let leaks = c.violations(&out);
        if !leaks.is_empty() {
            s.final_leaks += 1;
        }
        if let Some(m) = c.matches_expected(&out) {
            s.expected_total += 1;
            s.expected_final += usize::from(m);
        }
        println!(
            "- {:<24} {:>6} µs  llm-gate={:<5} {:?}{}",
            c.name,
            took.as_micros(),
            should_use_llm(&c.input),
            out,
            if leaks.is_empty() {
                String::new()
            } else {
                format!("  LEAK {leaks:?}")
            }
        );
    }
    println!();
    s
}

fn run_model(args: &Args, model: &str, cases: &[&Case]) -> Summary {
    println!("## {model}\n");
    let mut cfg = HttpConfig::new(&args.base_url, model);
    cfg.timeout = args.timeout;
    let mut n = OpenAiHttpNormalizer::new(cfg);
    let mut s = Summary {
        model: model.into(),
        ..Default::default()
    };
    let warm_start = Instant::now();
    if let Err(e) = n.warm() {
        println!("warm-up failed: {e}\n");
        s.warm_error = Some(e.to_string());
        return s;
    }
    println!("warm-up {} ms\n", ms(warm_start.elapsed()));

    for c in cases {
        s.cases += 1;
        let mut attempts: Vec<Attempt> = Vec::new();
        let mut errors = Vec::new();
        for _ in 0..args.runs {
            match c.with_request(|r| n.attempt(r)) {
                Ok(a) => attempts.push(a),
                Err(e) => errors.push(e.to_string()),
            }
        }
        let mut lat: Vec<Duration> = attempts.iter().map(|a| a.llm_elapsed).collect();
        lat.sort();
        s.latencies.extend(&lat);

        println!(
            "### {} ({}, {:?})",
            c.name,
            c.language.as_deref().unwrap_or("-"),
            c.style
        );
        println!("input:    {:?}", c.input);
        if let Some(e) = &c.expected {
            println!("expected: {e:?}");
        }
        let Some(first) = attempts.first() else {
            s.errors += 1;
            println!("ERROR:    {}\n", errors.join(" | "));
            continue;
        };
        if !errors.is_empty() {
            s.errors += 1;
            println!("errors:   {}", errors.join(" | "));
        }
        let final_text = match first.verdict {
            Ok(_) => &first.candidate,
            Err(_) => &first.rule_text,
        };
        println!("rules:    {:?}", first.rule_text);
        println!("llm:      {:?}", first.candidate);
        match &first.verdict {
            Ok(scores) => {
                s.validated += 1;
                let margin = |m: Option<wl_core::normalize::Measured>| {
                    m.map_or("-".to_string(), |m| format!("{:.2}", m.margin()))
                };
                println!(
                    "verdict:  ok (margins: containment {}, length ratio {})",
                    margin(scores.containment),
                    margin(scores.length_ratio)
                );
            }
            Err(r) => {
                s.rejected += 1;
                println!("verdict:  REJECTED [{}] ({r})", r.check().as_str());
            }
        }
        let raw_leaks: Vec<&str> = attempts
            .iter()
            .flat_map(|a| c.violations(&a.candidate))
            .collect::<std::collections::BTreeSet<_>>()
            .into_iter()
            .collect();
        let final_leaks = c.violations(final_text);
        if !raw_leaks.is_empty() {
            s.raw_leaks += 1;
        }
        if !final_leaks.is_empty() {
            s.final_leaks += 1;
        }
        println!(
            "must_not_contain: llm {}; inserted {}",
            if raw_leaks.is_empty() {
                "clean".to_string()
            } else {
                format!("VIOLATED {raw_leaks:?}")
            },
            if final_leaks.is_empty() {
                "clean".to_string()
            } else {
                format!("VIOLATED {final_leaks:?}")
            },
        );
        if let Some(m) = c.matches_expected(&first.candidate) {
            s.expected_total += 1;
            s.expected_llm += usize::from(m);
            s.expected_final += usize::from(c.matches_expected(final_text) == Some(true));
            println!("matches expected (words): llm {m}");
        }
        if attempts.iter().any(|a| a.candidate != first.candidate) {
            s.unstable += 1;
            println!("NOTE: output differed across runs");
        }
        println!(
            "latency:  p50 {} ms, p95 {} ms ({} runs)\n",
            ms(percentile(&lat, 0.5)),
            ms(percentile(&lat, 0.95)),
            lat.len()
        );
    }
    s
}

fn print_summary(rules: &Summary, models: &[Summary]) {
    println!("## summary\n");
    println!(
        "| normalizer | cases | validated | rejected | errors | must_not_contain in LLM output | must_not_contain inserted | matches expected (LLM / inserted) | unstable | p50 ms | p95 ms |"
    );
    println!("|---|---|---|---|---|---|---|---|---|---|---|");
    let mut rl = rules.latencies.clone();
    rl.sort();
    println!(
        "| {} | {} | - | - | - | - | {} | - / {} of {} | - | {:.3} | {:.3} |",
        rules.model,
        rules.cases,
        rules.final_leaks,
        rules.expected_final,
        rules.expected_total,
        percentile(&rl, 0.5).as_secs_f64() * 1000.0,
        percentile(&rl, 0.95).as_secs_f64() * 1000.0,
    );
    for s in models {
        if let Some(e) = &s.warm_error {
            println!(
                "| {} | warm-up failed: {} |||||||||",
                s.model,
                e.replace('|', "/")
            );
            continue;
        }
        let mut l = s.latencies.clone();
        l.sort();
        println!(
            "| {} | {} | {} | {} | {} | {} | {} | {} / {} of {} | {} | {} | {} |",
            s.model,
            s.cases,
            s.validated,
            s.rejected,
            s.errors,
            s.raw_leaks,
            s.final_leaks,
            s.expected_llm,
            s.expected_final,
            s.expected_total,
            s.unstable,
            ms(percentile(&l, 0.5)),
            ms(percentile(&l, 0.95)),
        );
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shipped_fixtures_parse_and_are_well_formed() {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/transcripts.toml");
        let f: Fixtures = toml::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
        assert!(f.cases.len() >= 20);
        let mut names = std::collections::HashSet::new();
        for c in &f.cases {
            assert!(names.insert(&c.name), "duplicate case {}", c.name);
            // A forbidden string in the good answer would flag every correct output.
            for m in &c.must_not_contain {
                let in_expected = c
                    .expected
                    .as_deref()
                    .is_some_and(|e| e.contains(m.as_str()));
                assert!(!in_expected, "{}: {m:?} is in the expected output", c.name);
            }
        }
        assert!(f.cases.iter().any(|c| c.language.as_deref() == Some("de")));
        assert!(f.cases.iter().any(|c| c.style == Style::Code));
    }

    #[test]
    fn percentile_is_nearest_rank() {
        let d: Vec<Duration> = (1..=10).map(Duration::from_millis).collect();
        assert_eq!(percentile(&d, 0.5), Duration::from_millis(5));
        assert_eq!(percentile(&d, 0.95), Duration::from_millis(10));
        assert_eq!(percentile(&[], 0.5), Duration::ZERO);
    }
}
