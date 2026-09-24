//! Deterministic cleanup. Every rule demands positive evidence before it fires: a missed
//! filler costs the user a keystroke, a deleted content word costs them trust.

use std::time::Instant;

use wl_core::normalize::{
    NormalizeError, NormalizeOutput, NormalizeRequest, Normalizer, Provenance, Style,
};

use crate::lang::{self, Action, CueDelimit, CueKind, Table};

const MAX_CORRECTION_WINDOW: usize = 6;
const FUZZY_MIN_CHARS: usize = 5;
const FUZZY_MIN_SIMILARITY: f64 = 0.92;
/// Jaro-Winkler forgives length differences that turn one word into another.
const FUZZY_MAX_LEN_DIFF: usize = 2;
const STUTTER_MAX_CHARS: usize = 4;

#[derive(Debug, Clone, Copy, Default)]
pub struct RuleNormalizer;

impl RuleNormalizer {
    pub fn new() -> Self {
        Self
    }

    pub fn clean_request(&self, req: &NormalizeRequest<'_>) -> String {
        clean(req.transcript, req.language, req.app.style, req.vocabulary)
    }
}

impl Normalizer for RuleNormalizer {
    fn id(&self) -> &str {
        "rules"
    }

    /// Ignores cancellation and deadlines: it is the fallback that must still produce text
    /// when everything else ran out of time.
    fn normalize(&mut self, req: &NormalizeRequest<'_>) -> Result<NormalizeOutput, NormalizeError> {
        let start = Instant::now();
        let text = self.clean_request(req);
        Ok(NormalizeOutput {
            utterance: req.utterance,
            text,
            provenance: Provenance::Rules,
            elapsed: start.elapsed(),
        })
    }
}

/// Non-prose styles skip stutters, spoken commands and casing: in a terminal "period" may
/// be an argument and capitalisation breaks commands.
pub fn clean(
    transcript: &str,
    language: Option<&str>,
    style: Style,
    vocabulary: &[String],
) -> String {
    let table = lang::table(language);
    let prose = matches!(style, Style::Formal | Style::Casual);
    let mut toks = tokenize(transcript);
    remove_fillers(&mut toks, table);
    if prose {
        collapse_stutters(&mut toks, table);
        apply_commands(&mut toks, table);
    }
    apply_corrections(&mut toks, table);
    apply_vocabulary(&mut toks, vocabulary);
    apply_style(&mut toks, style);
    join(&toks)
}

/// Checks every supported language, because the STT's language guess is not reliable on
/// short utterances.
pub(crate) fn has_disfluency(text: &str) -> bool {
    let toks = tokenize(text);
    let tables = [&lang::EN, &lang::DE];
    (0..toks.len()).any(|i| {
        let k = toks[i].key().unwrap_or_default();
        let stutter = toks.get(i + 1).and_then(Tok::key).is_some_and(|n| {
            n == k
                && !k.is_empty()
                && k.chars().count() <= STUTTER_MAX_CHARS
                && k.chars().all(char::is_alphabetic)
                && !tables.iter().any(|t| t.stutter_keep.contains(&k.as_str()))
        });
        stutter
            || tables.iter().any(|t| {
                t.fillers.contains(&k.as_str())
                    || t.phrase_fillers
                        .iter()
                        .any(|p| matches_words(&toks, i, p, false))
                    || t.cues
                        .iter()
                        .any(|c| matches_words(&toks, i, c.words, true))
                    || t.commands
                        .iter()
                        .any(|c| matches_words(&toks, i, c.words, false))
            })
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum Tok {
    Word(String),
    /// A dictionary entry in its canonical spelling; casing rules must not touch it.
    Vocab(String),
    Break(&'static str),
}

impl Tok {
    fn text(&self) -> Option<&str> {
        match self {
            Tok::Word(w) | Tok::Vocab(w) => Some(w),
            Tok::Break(_) => None,
        }
    }

    fn text_mut(&mut self) -> Option<&mut String> {
        match self {
            Tok::Word(w) | Tok::Vocab(w) => Some(w),
            Tok::Break(_) => None,
        }
    }

    fn key(&self) -> Option<String> {
        self.text().map(key)
    }
}

const LEAD: &[char] = &['"', '\'', '(', '[', '“', '‘', '„', '«', '¿', '¡'];
const TRAIL: &[char] = &[
    '.', ',', ';', ':', '!', '?', '"', '\'', ')', ']', '”', '’', '»', '…',
];
const CLAUSE: &[char] = &['.', ',', ';', ':', '!', '?', '…'];
const FINAL: &[char] = &['.', '!', '?', '…'];
const CLOSERS: &[char] = &['"', '\'', ')', ']', '”', '’', '»'];
const ABBREVIATIONS: &[&str] = &[
    "mr", "mrs", "ms", "dr", "st", "vs", "etc", "approx", "bzw", "usw", "ca", "nr",
];

/// (leading punctuation, core, trailing punctuation). Only sentence punctuation is
/// stripped, so "C++", "C#" and "node.js" keep their shape.
fn parts(s: &str) -> (&str, &str, &str) {
    let rest = s.trim_start_matches(LEAD);
    let lead = &s[..s.len() - rest.len()];
    let core = rest.trim_end_matches(TRAIL);
    (lead, core, &rest[core.len()..])
}

fn key(s: &str) -> String {
    parts(s).1.to_lowercase()
}

fn trail(s: &str) -> &str {
    parts(s).2
}

fn ends_clause(s: &str) -> bool {
    trail(s).contains(CLAUSE)
}

fn ends_sentence(s: &str) -> bool {
    trail(s).contains(FINAL) && !is_abbreviation(s)
}

fn is_abbreviation(s: &str) -> bool {
    let core = parts(s).1;
    core.contains('.')
        || (core.chars().count() == 1 && core.chars().all(char::is_alphabetic) && core != "I")
        || ABBREVIATIONS.contains(&core.to_lowercase().as_str())
}

fn is_number(k: &str) -> bool {
    !k.is_empty()
        && k.chars().any(|c| c.is_ascii_digit())
        && k.chars()
            .all(|c| c.is_ascii_digit() || matches!(c, '.' | ',' | ':'))
}

fn has_alnum(s: &str) -> bool {
    s.chars().any(char::is_alphanumeric)
}

fn tokenize(text: &str) -> Vec<Tok> {
    let mut out: Vec<Tok> = Vec::new();
    for w in text.split_whitespace() {
        // Punctuation the recogniser spaced off ("hello , world") belongs to the word
        // before; a lone quote does not, because it may open the next word.
        if w.chars().all(|c| CLAUSE.contains(&c))
            && let Some(Tok::Word(prev)) = out.last_mut()
        {
            prev.push_str(w);
            continue;
        }
        out.push(Tok::Word(w.to_string()));
    }
    out
}

fn join(toks: &[Tok]) -> String {
    let mut out = String::new();
    for t in toks {
        match t {
            Tok::Break(b) => {
                let n = out.trim_end_matches(' ').len();
                out.truncate(n);
                out.push_str(b);
            }
            Tok::Word(w) | Tok::Vocab(w) => {
                if !out.is_empty() && !out.ends_with('\n') {
                    out.push(' ');
                }
                out.push_str(w);
            }
        }
    }
    out
}

fn starts_sentence(toks: &[Tok], i: usize) -> bool {
    i == 0
        || match &toks[i - 1] {
            Tok::Break(_) => true,
            t => t.text().is_some_and(ends_sentence),
        }
}

/// Inner tokens must carry no punctuation, so "new, line" is two thoughts rather than one
/// command; `inner_commas` relaxes that for cues, which recognisers write as "no, wait,".
fn matches_words(toks: &[Tok], i: usize, words: &[&str], inner_commas: bool) -> bool {
    if i + words.len() > toks.len() {
        return false;
    }
    words.iter().enumerate().all(|(j, w)| {
        let Tok::Word(t) = &toks[i + j] else {
            return false;
        };
        let inner_ok = j + 1 == words.len() || {
            let tr = trail(t);
            tr.is_empty() || (inner_commas && tr == ",")
        };
        inner_ok && key(t) == *w
    })
}

/// Moves the punctuation the removed tokens carried onto the neighbours, so a sentence
/// end or a quote is never lost with the filler.
fn remove_span(toks: &mut Vec<Tok>, at: usize, n: usize, table: &Table) {
    let first = toks[at].text().unwrap_or_default().to_owned();
    let last = toks[at + n - 1].text().unwrap_or_default().to_owned();
    let lead = parts(&first).0.to_owned();
    let tail = trail(&last).to_owned();
    let prev_initial = at >= 1 && starts_sentence(toks, at - 1);
    toks.drain(at..at + n);

    let next_key = toks.get(at).and_then(Tok::key);
    if !lead.is_empty()
        && let Some(next) = toks.get_mut(at).and_then(Tok::text_mut)
    {
        next.insert_str(0, &lead);
    }
    let Some(prev) = at
        .checked_sub(1)
        .and_then(|p| toks.get_mut(p))
        .and_then(Tok::text_mut)
    else {
        return;
    };
    let closers: String = tail.chars().filter(|c| CLOSERS.contains(c)).collect();
    if let Some(f) = tail.chars().find(|c| FINAL.contains(c)) {
        while prev.ends_with([',', ';', ':']) {
            prev.pop();
        }
        if !ends_sentence(prev) {
            prev.push(f);
        }
    } else if tail.contains(',') && prev.ends_with(',') {
        let keep =
            prev_initial || next_key.is_some_and(|k| table.clause_starters.contains(&k.as_str()));
        if !keep {
            prev.pop();
        }
    }
    prev.push_str(&closers);
}

fn remove_fillers(toks: &mut Vec<Tok>, table: &Table) {
    let mut i = 0;
    while i < toks.len() {
        let is_single =
            matches!(&toks[i], Tok::Word(w) if table.fillers.contains(&key(w).as_str()));
        if is_single {
            remove_span(toks, i, 1, table);
            continue;
        }
        if let Some(n) = phrase_filler_at(toks, i, table) {
            remove_span(toks, i, n, table);
            continue;
        }
        i += 1;
    }
}

fn phrase_filler_at(toks: &[Tok], i: usize, table: &Table) -> Option<usize> {
    table.phrase_fillers.iter().find_map(|p| {
        let n = p.len();
        if !matches_words(toks, i, p, false) {
            return None;
        }
        let last = toks[i + n - 1].text()?;
        let prev_punct = i > 0 && matches!(&toks[i - 1], Tok::Break(_))
            || i > 0 && toks[i - 1].text().is_some_and(ends_clause);
        let before_ok = i == 0 || prev_punct;
        let after_ok = i + n == toks.len() || ends_clause(last);
        let explicit = prev_punct || ends_clause(last);
        (before_ok && after_ok && explicit).then_some(n)
    })
}

fn collapse_stutters(toks: &mut Vec<Tok>, table: &Table) {
    let mut i = 0;
    while i + 1 < toks.len() {
        if let (Tok::Word(a), Tok::Word(b)) = (&toks[i], &toks[i + 1]) {
            let ka = key(a);
            let collapses = !ka.is_empty()
                && ka.chars().count() <= STUTTER_MAX_CHARS
                && ka.chars().all(char::is_alphabetic)
                && trail(a).is_empty()
                && parts(b).0.is_empty()
                && ka == key(b)
                && !table.stutter_keep.contains(&ka.as_str());
            if collapses {
                let merged = format!("{a}{}", trail(b));
                toks[i] = Tok::Word(merged);
                toks.remove(i + 1);
                continue;
            }
        }
        i += 1;
    }
}

fn command_at(toks: &[Tok], i: usize, table: &Table) -> Option<(Action, usize)> {
    let cmd = table
        .commands
        .iter()
        .find(|c| matches_words(toks, i, c.words, false))?;
    let n = cmd.words.len();
    let prev = i.checked_sub(1).map(|p| &toks[p]);
    let next = toks.get(i + n);
    let cmd_punct = !trail(toks[i + n - 1].text()?).is_empty();

    let prev_word_ok = prev.and_then(Tok::text).is_some_and(|w| {
        let k = key(w);
        has_alnum(&k) && !table.not_before_command.contains(&k.as_str()) && !is_number(&k)
    });
    let next_blocks = next
        .and_then(Tok::key)
        .is_some_and(|k| table.not_after_command.contains(&k.as_str()) || is_number(&k));
    let next_is_command = next.is_some()
        && table
            .commands
            .iter()
            .any(|c| matches_words(toks, i + n, c.words, false));
    let boundary = next.is_none()
        || cmd_punct
        || matches!(next, Some(Tok::Break(_)))
        || next_is_command
        || (table.capital_marks_sentence
            && next.and_then(Tok::text).is_some_and(|w| {
                let core = parts(w).1;
                core != "I" && core.chars().next().is_some_and(char::is_uppercase)
            }));
    let after_break_or_start = prev.is_none() || matches!(prev, Some(Tok::Break(_)));

    let fires = match cmd.action {
        Action::Punct(_) => prev_word_ok && !next_blocks && (!cmd.strict || boundary),
        Action::Break(_) => (after_break_or_start || prev_word_ok) && !next_blocks,
        Action::OpenQuote => {
            matches!(next, Some(Tok::Word(_) | Tok::Vocab(_)))
                && (after_break_or_start
                    || prev_word_ok
                    || prev.and_then(Tok::text).is_some_and(ends_clause))
        }
        Action::CloseQuote => prev.and_then(Tok::text).is_some_and(has_alnum) && !next_blocks,
    };
    fires.then_some((cmd.action, n))
}

fn apply_commands(toks: &mut Vec<Tok>, table: &Table) {
    let mut i = 0;
    while i < toks.len() {
        let Some((action, n)) = command_at(toks, i, table) else {
            i += 1;
            continue;
        };
        match action {
            Action::Punct(p) => {
                if let Some(prev) = toks.get_mut(i - 1).and_then(Tok::text_mut) {
                    while prev.ends_with(CLAUSE) {
                        prev.pop();
                    }
                    prev.push_str(p);
                }
                toks.drain(i..i + n);
            }
            Action::Break(b) => {
                toks.splice(i..i + n, [Tok::Break(b)]);
                i += 1;
            }
            Action::OpenQuote => {
                toks.drain(i..i + n);
                if let Some(next) = toks.get_mut(i).and_then(Tok::text_mut) {
                    next.insert(0, '"');
                }
            }
            Action::CloseQuote => {
                if let Some(prev) = toks.get_mut(i - 1).and_then(Tok::text_mut) {
                    prev.push('"');
                }
                toks.drain(i..i + n);
            }
        }
    }
}

fn apply_corrections(toks: &mut Vec<Tok>, table: &Table) {
    let mut j = 0;
    while j < toks.len() {
        let span = table
            .cues
            .iter()
            .find(|c| matches_words(toks, j, c.words, true))
            .and_then(|c| correction_span(toks, j, c));
        match span {
            Some((from, to)) => {
                toks.drain(from..to);
                j = from;
            }
            None => j += 1,
        }
    }
}

/// The corrected part plus the cue. `None` whenever the evidence is thin; the LLM or the
/// user can still fix what the rules leave.
fn correction_span(toks: &[Tok], j: usize, cue: &lang::Cue) -> Option<(usize, usize)> {
    let n = cue.words.len();
    let before = toks.get(j.checked_sub(1)?)?.text()?;
    let prev_delim = ends_clause(before);
    let cue_delim = ends_clause(toks[j + n - 1].text()?);
    let delimited = match cue.delimit {
        CueDelimit::Either => prev_delim || cue_delim,
        CueDelimit::Both => prev_delim && cue_delim,
    };
    if !delimited {
        return None;
    }

    let mut ws = j - 1;
    while ws > 0 && j - ws < MAX_CORRECTION_WINDOW {
        match toks[ws - 1].text() {
            Some(t) if !ends_clause(t) => ws -= 1,
            _ => break,
        }
    }

    let r = j + n;
    let repair_first = toks.get(r)?.text()?;
    if !has_alnum(parts(repair_first).1) {
        return None;
    }
    let mut repair_len = 0;
    for t in &toks[r..] {
        let Some(w) = t.text() else { break };
        repair_len += 1;
        if ends_clause(w) {
            break;
        }
    }

    match cue.kind {
        CueKind::Scratch => Some((ws, r)),
        CueKind::Replace => {
            // "to John, I mean, to Maria": the repair restarts at a word the speaker
            // already said, which pins down exactly what it replaces.
            let rk = key(repair_first);
            if let Some(a) = (ws..j)
                .rev()
                .find(|&a| toks[a].key().as_deref() == Some(&rk))
            {
                return Some((a, r));
            }
            // A like-for-like one-word swap ("eggs, I mean, flour"). Longer unanchored
            // repairs are ambiguous about how much they replace.
            if same_kind(before, repair_first) || repair_len == 1 {
                return Some((j - 1, r));
            }
            None
        }
    }
}

fn same_kind(a: &str, b: &str) -> bool {
    let (ka, kb) = (parts(a).1, parts(b).1);
    let capital = |s: &str| s != "I" && s.chars().next().is_some_and(char::is_uppercase);
    (is_number(ka) && is_number(kb)) || (capital(ka) && capital(kb))
}

fn apply_vocabulary(toks: &mut Vec<Tok>, vocabulary: &[String]) {
    for entry in vocabulary {
        let words: Vec<&str> = entry.split_whitespace().collect();
        if words.is_empty() {
            continue;
        }
        let keys: Vec<String> = words.iter().map(|w| key(w)).collect();
        let key_refs: Vec<&str> = keys.iter().map(String::as_str).collect();
        let mut i = 0;
        while i + words.len() <= toks.len() {
            if matches_words(toks, i, &key_refs, false) {
                let first = toks[i].text().unwrap_or_default();
                let last = toks[i + words.len() - 1].text().unwrap_or_default();
                let replaced = format!("{}{}{}", parts(first).0, words.join(" "), trail(last));
                toks.splice(i..i + words.len(), [Tok::Vocab(replaced)]);
            }
            i += 1;
        }
    }

    let singles: Vec<(&str, String)> = vocabulary
        .iter()
        .map(|e| e.trim())
        .filter(|e| !e.contains(char::is_whitespace) && e.chars().count() >= FUZZY_MIN_CHARS)
        .map(|e| (e, e.to_lowercase()))
        .collect();
    if singles.is_empty() {
        return;
    }
    for t in toks.iter_mut() {
        let Tok::Word(w) = t else { continue };
        let (lead, core, tr) = parts(w);
        let len = core.chars().count();
        if len < FUZZY_MIN_CHARS {
            continue;
        }
        let lc = core.to_lowercase();
        let best = singles
            .iter()
            .filter(|(_, el)| el.chars().count().abs_diff(len) <= FUZZY_MAX_LEN_DIFF)
            .map(|(e, el)| (e, strsim::jaro_winkler(&lc, el)))
            .filter(|(_, s)| *s >= FUZZY_MIN_SIMILARITY)
            .max_by(|a, b| a.1.total_cmp(&b.1));
        if let Some((e, _)) = best {
            *t = Tok::Vocab(format!("{lead}{e}{tr}"));
        }
    }
}

/// Mixed-case words ("iPhone", "eBay") are identifiers and keep their spelling.
fn capitalize_first(w: &mut String) {
    let core = parts(w).1;
    if core.chars().any(char::is_uppercase) {
        return;
    }
    let Some((idx, c)) = w.char_indices().find(|(_, c)| c.is_alphabetic()) else {
        return;
    };
    let upper: String = c.to_uppercase().collect();
    w.replace_range(idx..idx + c.len_utf8(), &upper);
}

fn apply_style(toks: &mut [Tok], style: Style) {
    if !matches!(style, Style::Formal | Style::Casual) {
        return;
    }
    let mut at_start = true;
    for t in toks.iter_mut() {
        match t {
            Tok::Break(_) => at_start = true,
            Tok::Vocab(w) => at_start = ends_sentence(w),
            Tok::Word(w) => {
                if at_start {
                    capitalize_first(w);
                }
                at_start = ends_sentence(w);
            }
        }
    }

    let Some(last) = toks.last_mut().and_then(Tok::text_mut) else {
        return;
    };
    match style {
        Style::Formal => {
            if !trail(last).contains(FINAL) {
                while last.ends_with([',', ';', ':']) {
                    last.pop();
                }
                last.push('.');
            }
        }
        Style::Casual => {
            if last.ends_with('.') && !last.ends_with("..") && !is_abbreviation(last) {
                last.pop();
            }
        }
        Style::Code | Style::None => {}
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use wl_core::normalize::AppContext;

    fn formal(s: &str) -> String {
        clean(s, Some("en"), Style::Formal, &[])
    }

    fn casual(s: &str) -> String {
        clean(s, Some("en"), Style::Casual, &[])
    }

    fn code(s: &str) -> String {
        clean(s, Some("en"), Style::Code, &[])
    }

    fn de(s: &str) -> String {
        clean(s, Some("de"), Style::Formal, &[])
    }

    fn vocab(s: &str, v: &[&str]) -> String {
        let v: Vec<String> = v.iter().map(|s| s.to_string()).collect();
        clean(s, Some("en"), Style::Formal, &v)
    }

    #[test]
    fn whitespace_is_collapsed_and_spaced_punctuation_reattached() {
        assert_eq!(code("  hello   world ,  again \t"), "hello world, again");
    }

    #[test]
    fn empty_input_stays_empty() {
        assert_eq!(formal(""), "");
        assert_eq!(formal("   "), "");
    }

    #[test]
    fn stutters_collapse() {
        assert_eq!(formal("the the cat sat"), "The cat sat.");
        assert_eq!(formal("I I think so."), "I think so.");
        assert_eq!(formal("we we we should go."), "We should go.");
    }

    #[test]
    fn grammatical_doubles_are_not_stutters() {
        assert_eq!(
            formal("I know that that is true."),
            "I know that that is true."
        );
        assert_eq!(formal("Call one one two."), "Call one one two.");
        assert_eq!(
            formal("That was really really good."),
            "That was really really good."
        );
        assert_eq!(formal("The, the thing is."), "The, the thing is.");
        assert_eq!(
            de("Die Frau, die die Zeitung liest."),
            "Die Frau, die die Zeitung liest."
        );
    }

    #[test]
    fn stutters_are_left_alone_in_code() {
        assert_eq!(code("echo the the"), "echo the the");
    }

    #[test]
    fn standalone_fillers_are_removed() {
        assert_eq!(formal("Um, I think we should go."), "I think we should go.");
        assert_eq!(
            formal("I was, uh, thinking about it."),
            "I was thinking about it."
        );
        assert_eq!(formal("hmm okay"), "Okay.");
        assert_eq!(formal("It works, um, but slowly."), "It works, but slowly.");
    }

    #[test]
    fn filler_after_sentence_initial_word_keeps_that_comma() {
        assert_eq!(formal("Yes, um, I agree."), "Yes, I agree.");
    }

    #[test]
    fn filler_carrying_the_sentence_end_hands_it_back() {
        assert_eq!(
            formal("We are done, uh. Next item"),
            "We are done. Next item."
        );
    }

    #[test]
    fn fillers_only_match_whole_words() {
        assert_eq!(formal("We need an umbrella."), "We need an umbrella.");
        assert_eq!(formal("Take him to the ER."), "Take him to the ER.");
        assert_eq!(formal("Uhura said hmmm."), "Uhura said hmmm.");
    }

    #[test]
    fn like_and_so_are_never_removed() {
        assert_eq!(
            formal("So I like it, like, a lot."),
            "So I like it, like, a lot."
        );
    }

    #[test]
    fn you_know_is_removed_only_when_set_off() {
        assert_eq!(formal("You know, it was fine."), "It was fine.");
        assert_eq!(formal("It was, you know, fine."), "It was fine.");
        assert_eq!(formal("It's fine, you know."), "It's fine.");
    }

    #[test]
    fn you_know_as_content_stays() {
        assert_eq!(formal("You know the answer."), "You know the answer.");
        assert_eq!(
            formal("Do you know where it is?"),
            "Do you know where it is?"
        );
        assert_eq!(formal("you know"), "You know.");
    }

    #[test]
    fn german_fillers_are_removed() {
        assert_eq!(
            de("Äh, ich glaube, ähm, dass es passt."),
            "Ich glaube, dass es passt."
        );
        assert_eq!(
            de("Wir sollten, ähm, das Meeting verschieben."),
            "Wir sollten das Meeting verschieben."
        );
    }

    #[test]
    fn german_fillers_are_not_english_fillers() {
        assert_eq!(formal("äh"), "Äh.");
    }

    #[test]
    fn fillers_are_removed_in_code_style() {
        assert_eq!(code("Um, git status"), "git status");
    }

    #[test]
    fn spoken_punctuation_and_layout() {
        assert_eq!(
            formal("Dear team comma new paragraph the release is ready period"),
            "Dear team,\n\nThe release is ready."
        );
        assert_eq!(formal("Meet me at noon period"), "Meet me at noon.");
        assert_eq!(
            formal("Hello comma how are you question mark"),
            "Hello, how are you?"
        );
        assert_eq!(formal("That is great exclamation mark"), "That is great!");
        assert_eq!(
            formal("first item new line second item"),
            "First item\nSecond item."
        );
        assert_eq!(formal("done period Then we left"), "Done. Then we left.");
    }

    #[test]
    fn spoken_quotes() {
        assert_eq!(
            formal("He said open quote hello close quote and left"),
            "He said \"hello\" and left."
        );
    }

    #[test]
    fn punctuation_words_used_as_nouns_stay() {
        assert_eq!(
            formal("The period of time was short."),
            "The period of time was short."
        );
        assert_eq!(
            formal("The trial period ends soon."),
            "The trial period ends soon."
        );
        assert_eq!(
            formal("The data period matters."),
            "The data period matters."
        );
        assert_eq!(formal("Put a comma here."), "Put a comma here.");
        assert_eq!(
            formal("Use comma separated values."),
            "Use comma separated values."
        );
        assert_eq!(
            formal("Is there a question mark over the plan?"),
            "Is there a question mark over the plan?"
        );
        assert_eq!(formal("Start a new line here."), "Start a new line here.");
        assert_eq!(
            formal("Insert new line character."),
            "Insert new line character."
        );
        assert_eq!(formal("period"), "Period.");
    }

    #[test]
    fn spoken_commands_are_ignored_in_code() {
        assert_eq!(
            code("git commit dash m fix typo period"),
            "git commit dash m fix typo period"
        );
    }

    #[test]
    fn german_spoken_punctuation() {
        assert_eq!(
            de("Hallo Anna Komma wie geht es dir Fragezeichen"),
            "Hallo Anna, wie geht es dir?"
        );
        assert_eq!(de("Das war es Punkt"), "Das war es.");
    }

    #[test]
    fn german_punctuation_nouns_stay() {
        assert_eq!(
            de("Wir treffen uns um Punkt drei."),
            "Wir treffen uns um Punkt drei."
        );
        assert_eq!(
            de("Das ist ein wichtiger Punkt für uns."),
            "Das ist ein wichtiger Punkt für uns."
        );
        assert_eq!(de("Der Punkt Mitte ist gut."), "Der Punkt Mitte ist gut.");
        assert_eq!(
            de("Es sind drei Komma fünf Prozent."),
            "Es sind drei Komma fünf Prozent."
        );
    }

    #[test]
    fn like_for_like_correction() {
        assert_eq!(
            formal("Let's meet on Tuesday, no wait, Wednesday at 10."),
            "Let's meet on Wednesday at 10."
        );
        assert_eq!(formal("Um, meet at 5, no wait, 6."), "Meet at 6.");
        assert_eq!(formal("We need eggs, I mean, flour."), "We need flour.");
    }

    #[test]
    fn anchored_correction() {
        assert_eq!(
            formal("Send the report to John, I mean, to Maria by Friday."),
            "Send the report to Maria by Friday."
        );
        assert_eq!(
            de("Schick die Datei an Peter, nein warte, an Anna."),
            "Schick die Datei an Anna."
        );
    }

    #[test]
    fn scratch_that_drops_the_previous_clause() {
        assert_eq!(
            formal("Buy eggs and milk. Scratch that, buy bread."),
            "Buy bread."
        );
    }

    #[test]
    fn scratch_that_reaches_back_at_most_six_words() {
        assert_eq!(
            formal("one two three four five six seven eight, scratch that, nine"),
            "One two nine."
        );
    }

    #[test]
    fn corrections_need_delimiters_and_a_repair() {
        assert_eq!(formal("I mean it."), "I mean it.");
        assert_eq!(
            formal("You know I mean what I say."),
            "You know I mean what I say."
        );
        assert_eq!(formal("I actually like it."), "I actually like it.");
        assert_eq!(
            formal("Tuesday is fine, actually Wednesday is better."),
            "Tuesday is fine, actually Wednesday is better."
        );
        assert_eq!(
            formal("I need to scratch that itch."),
            "I need to scratch that itch."
        );
        assert_eq!(
            formal("Send it Monday, no wait."),
            "Send it Monday, no wait."
        );
        assert_eq!(formal("There is no wait time."), "There is no wait time.");
    }

    #[test]
    fn unanchored_long_repair_is_left_alone() {
        let s = "I think we should go, I mean, let's stay home tonight.";
        assert_eq!(formal(s), s);
    }

    #[test]
    fn corrections_apply_in_code_style() {
        assert_eq!(
            code("git checkout main, no wait, develop"),
            "git checkout develop"
        );
    }

    #[test]
    fn vocabulary_exact_match_is_case_insensitive() {
        let v = vec!["Kubernetes".to_string(), "gRPC".to_string()];
        assert_eq!(
            clean("we use kubernetes and GRPC", Some("en"), Style::Casual, &v),
            "We use Kubernetes and gRPC"
        );
    }

    #[test]
    fn vocabulary_spelling_survives_sentence_casing() {
        assert_eq!(vocab("grpc is fast", &["gRPC"]), "gRPC is fast.");
    }

    #[test]
    fn vocabulary_multi_word_entries() {
        assert_eq!(
            vocab("ask anneliese müller today", &["Anneliese Müller"]),
            "Ask Anneliese Müller today."
        );
    }

    #[test]
    fn vocabulary_keeps_surrounding_punctuation() {
        assert_eq!(
            vocab("I love kubernetes, really", &["Kubernetes"]),
            "I love Kubernetes, really."
        );
    }

    #[test]
    fn vocabulary_fuzzy_near_miss() {
        assert_eq!(
            vocab("The Kubernetis cluster", &["Kubernetes"]),
            "The Kubernetes cluster."
        );
    }

    #[test]
    fn vocabulary_does_not_touch_other_words() {
        assert_eq!(vocab("It matters.", &["Mattias"]), "It matters.");
        assert_eq!(
            vocab("The governess left.", &["Kubernetes"]),
            "The governess left."
        );
        assert_eq!(vocab("I trusted it.", &["Rust"]), "I trusted it.");
        assert_eq!(vocab("Ask Joe.", &["Joel"]), "Ask Joe.");
    }

    #[test]
    fn vocabulary_applies_in_code_style() {
        let v = vec!["kubectl".to_string()];
        assert_eq!(
            clean("KUBECTL get pods", None, Style::Code, &v),
            "kubectl get pods"
        );
    }

    #[test]
    fn formal_adds_terminal_period_and_capitalises() {
        assert_eq!(formal("hello there"), "Hello there.");
        assert_eq!(formal("hello there,"), "Hello there.");
        assert_eq!(formal("is it done?"), "Is it done?");
        assert_eq!(formal("first. second"), "First. Second.");
    }

    #[test]
    fn casual_drops_only_a_single_trailing_period() {
        assert_eq!(casual("hello there."), "Hello there");
        assert_eq!(casual("Is it done?"), "Is it done?");
        assert_eq!(casual("Well..."), "Well...");
        assert_eq!(casual("Ask Dr."), "Ask Dr.");
    }

    #[test]
    fn capitalisation_skips_identifiers_and_abbreviations() {
        assert_eq!(formal("iPhone sales rose"), "iPhone sales rose.");
        assert_eq!(formal("see e.g. the docs"), "See e.g. the docs.");
    }

    #[test]
    fn code_and_none_change_no_casing_or_punctuation() {
        assert_eq!(code("git status"), "git status");
        assert_eq!(clean("git status", None, Style::None, &[]), "git status");
        assert_eq!(clean("uh git status", None, Style::None, &[]), "git status");
    }

    #[test]
    fn unknown_language_uses_english_tables() {
        assert_eq!(clean("um hello", Some("fr"), Style::Formal, &[]), "Hello.");
    }

    #[test]
    fn disfluency_detection() {
        assert!(has_disfluency("so um yeah"));
        assert!(has_disfluency("Tuesday, no wait, Wednesday"));
        assert!(has_disfluency("the the cat"));
        assert!(has_disfluency("ich glaube äh ja"));
        assert!(has_disfluency("hello comma world"));
        assert!(!has_disfluency("I know that that is true."));
        assert!(!has_disfluency("A perfectly clean sentence."));
    }

    #[test]
    fn normalizer_trait_reports_rules() {
        let app = AppContext {
            style: Style::Formal,
            ..Default::default()
        };
        let req = NormalizeRequest {
            transcript: "um hello",
            language: None,
            vocabulary: &[],
            app: &app,
            previous: None,
            utterance: wl_core::UtteranceId(3),
            cancel: wl_core::CancelToken::new(),
        };
        let out = RuleNormalizer::new().normalize(&req).unwrap();
        assert_eq!(out.utterance, wl_core::UtteranceId(3));
        assert_eq!(out.text, "Hello.");
        assert_eq!(out.provenance, Provenance::Rules);
        assert_eq!(RuleNormalizer.id(), "rules");
    }
}
