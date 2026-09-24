//! Per-language word lists used by the rule pass, the validator and the LLM gate.
//!
//! Every list is deliberately short. A word belongs here only if removing or acting on it
//! is almost never wrong; ambiguous discourse markers ("like", "so", German "also") are
//! left for the LLM because deleting them from real content is worse than keeping them.

pub(crate) struct Table {
    /// Single tokens that are never content.
    pub fillers: &'static [&'static str],
    /// Multi-word fillers that are also ordinary phrases ("you know what I mean"), so they
    /// are removed only when commas set them off.
    pub phrase_fillers: &'static [&'static [&'static str]],
    pub cues: &'static [Cue],
    pub commands: &'static [Command],
    /// Words whose exact doubling is grammatical ("I know that that is true", German
    /// "die die Zeitung"), or a number being dictated digit by digit.
    pub stutter_keep: &'static [&'static str],
    /// A spoken-punctuation word after one of these is a noun ("the period of time").
    pub not_before_command: &'static [&'static str],
    /// A spoken-punctuation word before one of these is a noun ("period of", "comma
    /// separated").
    pub not_after_command: &'static [&'static str],
    /// German capitalises every noun, so an upper-case next word is no evidence of a
    /// sentence boundary there.
    pub capital_marks_sentence: bool,
    /// When a filler sat between two commas, one of them may be grammatical ("It works,
    /// um, but slowly"; German "Ich glaube, äh, das passt"). A following clause starter is
    /// the cheap signal that the first comma was real.
    pub clause_starters: &'static [&'static str],
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CueKind {
    /// "X, no wait, Y": Y replaces the end of X.
    Replace,
    /// "X. Scratch that, Y": X is dropped.
    Scratch,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum CueDelimit {
    /// A comma or sentence boundary before or after the cue is enough.
    Either,
    /// Common in ordinary speech ("I actually like it"), so it must be set off on both
    /// sides.
    Both,
}

pub(crate) struct Cue {
    pub words: &'static [&'static str],
    pub kind: CueKind,
    pub delimit: CueDelimit,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Action {
    Punct(&'static str),
    Break(&'static str),
    OpenQuote,
    CloseQuote,
}

pub(crate) struct Command {
    pub words: &'static [&'static str],
    pub action: Action,
    /// "period" and "Punkt" are ordinary nouns far more often than commands, so they
    /// need positive evidence of a sentence boundary, not merely the absence of a
    /// determiner.
    pub strict: bool,
}

const fn cmd(words: &'static [&'static str], action: Action, strict: bool) -> Command {
    Command {
        words,
        action,
        strict,
    }
}

const fn cue(words: &'static [&'static str], kind: CueKind, delimit: CueDelimit) -> Cue {
    Cue {
        words,
        kind,
        delimit,
    }
}

pub(crate) static EN: Table = Table {
    // Not "er": it is also "ER", the emergency room.
    fillers: &["um", "umm", "uh", "uhh", "uhm", "erm", "hmm", "hm", "mmm"],
    phrase_fillers: &[&["you", "know"]],
    cues: &[
        cue(&["no", "wait"], CueKind::Replace, CueDelimit::Either),
        cue(&["wait", "no"], CueKind::Replace, CueDelimit::Either),
        cue(&["i", "mean"], CueKind::Replace, CueDelimit::Both),
        cue(&["actually"], CueKind::Replace, CueDelimit::Both),
        cue(&["scratch", "that"], CueKind::Scratch, CueDelimit::Either),
    ],
    // Longest phrases first so "exclamation point" is not read as something shorter.
    commands: &[
        cmd(&["new", "paragraph"], Action::Break("\n\n"), false),
        cmd(&["new", "line"], Action::Break("\n"), false),
        cmd(&["question", "mark"], Action::Punct("?"), false),
        cmd(&["exclamation", "mark"], Action::Punct("!"), false),
        cmd(&["exclamation", "point"], Action::Punct("!"), false),
        cmd(&["full", "stop"], Action::Punct("."), true),
        cmd(&["open", "quote"], Action::OpenQuote, false),
        cmd(&["begin", "quote"], Action::OpenQuote, false),
        cmd(&["close", "quote"], Action::CloseQuote, false),
        cmd(&["end", "quote"], Action::CloseQuote, false),
        cmd(&["period"], Action::Punct("."), true),
        cmd(&["comma"], Action::Punct(","), false),
    ],
    stutter_keep: &[
        "that", "had", "is", "do", "no", "so", "bye", "ha", "very", "yes", "yeah", "well", "oh",
        "zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
    ],
    not_before_command: &[
        "a",
        "an",
        "the",
        "this",
        "that",
        "these",
        "those",
        "my",
        "your",
        "his",
        "her",
        "its",
        "our",
        "their",
        "each",
        "every",
        "no",
        "any",
        "some",
        "which",
        "what",
        "whose",
        "one",
        "per",
        "of",
        "in",
        "for",
        "to",
        "at",
        "on",
        "by",
        "with",
        "from",
        "into",
        "during",
        "over",
        "under",
        "after",
        "before",
        "same",
        "whole",
        "entire",
        "first",
        "last",
        "next",
        "trial",
        "grace",
        "time",
        "waiting",
        "test",
        "notice",
        "cooling",
        "reporting",
        "billing",
        "oxford",
        "serial",
        "brand",
        "big",
        "long",
        "short",
        "open",
        "and",
        "or",
    ],
    not_after_command: &[
        "of",
        "is",
        "was",
        "are",
        "were",
        "in",
        "for",
        "to",
        "at",
        "on",
        "after",
        "before",
        "between",
        "separated",
        "delimited",
        "splice",
        "character",
        "characters",
        "key",
        "sign",
        "symbol",
        "button",
        "icon",
        "emoji",
        "ends",
        "ended",
        "starts",
        "started",
        "over",
    ],
    capital_marks_sentence: true,
    clause_starters: &[
        "and", "but", "or", "so", "because", "which", "who", "whom", "whose", "that", "if", "when",
        "while", "although", "though", "since", "unless", "then", "i", "we", "you", "he", "she",
        "they", "it", "there", "this",
    ],
};

pub(crate) static DE: Table = Table {
    fillers: &["äh", "ähm", "öh", "öhm", "ähh", "hm", "hmm", "mhm"],
    phrase_fillers: &[&["weißt", "du"]],
    cues: &[
        cue(&["nein", "warte"], CueKind::Replace, CueDelimit::Either),
        cue(&["ich", "meine"], CueKind::Replace, CueDelimit::Both),
        cue(&["streich", "das"], CueKind::Scratch, CueDelimit::Either),
    ],
    commands: &[
        cmd(&["neuer", "absatz"], Action::Break("\n\n"), false),
        cmd(&["neue", "zeile"], Action::Break("\n"), false),
        cmd(&["anführungszeichen", "auf"], Action::OpenQuote, false),
        cmd(&["anführungszeichen", "zu"], Action::CloseQuote, false),
        cmd(&["fragezeichen"], Action::Punct("?"), false),
        cmd(&["ausrufezeichen"], Action::Punct("!"), false),
        cmd(&["punkt"], Action::Punct("."), true),
        cmd(&["komma"], Action::Punct(","), false),
    ],
    stutter_keep: &[
        "die", "der", "das", "den", "dem", "ja", "nein", "so", "sehr", "null", "eins", "zwei",
        "drei", "vier", "fünf", "sechs", "acht", "neun", "zehn",
    ],
    not_before_command: &[
        "der",
        "die",
        "das",
        "den",
        "dem",
        "des",
        "ein",
        "eine",
        "einen",
        "einem",
        "einer",
        "eines",
        "kein",
        "keine",
        "keinen",
        "dieser",
        "diese",
        "diesen",
        "diesem",
        "jeder",
        "jede",
        "jeden",
        "mein",
        "meine",
        "dein",
        "deine",
        "sein",
        "seine",
        "ihr",
        "ihre",
        "unser",
        "unsere",
        "welcher",
        "welche",
        "im",
        "am",
        "zum",
        "zur",
        "beim",
        "vom",
        "von",
        "mit",
        "nach",
        "vor",
        "auf",
        "in",
        "an",
        "um",
        "bis",
        "und",
        "oder",
        "wichtiger",
        "wichtige",
        "nächster",
        "letzter",
        "erster",
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
    ],
    not_after_command: &[
        "der", "die", "das", "des", "für", "von", "ist", "war", "sind", "mit", "zeichen", "taste",
        "symbol", "uhr", "null", "eins", "zwei", "drei", "vier", "fünf", "sechs", "sieben", "acht",
        "neun", "zehn",
    ],
    capital_marks_sentence: false,
    // Not "der/die/das": as articles they mostly continue the clause ("wir sollten, äh,
    // das Meeting"), and a wrong comma reads worse than a missing one.
    clause_starters: &[
        "und", "aber", "oder", "denn", "weil", "dass", "wenn", "als", "ob", "obwohl", "ich", "wir",
        "du", "er", "sie", "es", "man", "da", "dann",
    ],
};

/// Unknown and missing languages use English: its fillers never collide with content
/// words of the other supported languages.
pub(crate) fn table(lang: Option<&str>) -> &'static Table {
    match primary(lang).as_deref() {
        Some("de") => &DE,
        _ => &EN,
    }
}

/// Primary subtag of a BCP-47 tag, lower-cased: `de-AT` and `de_at` are both `de`.
pub(crate) fn primary(lang: Option<&str>) -> Option<String> {
    let tag = lang?.split(['-', '_']).next()?.trim();
    (!tag.is_empty()).then(|| tag.to_ascii_lowercase())
}

pub(crate) fn is_filler(lang: Option<&str>, word: &str) -> bool {
    table(lang).fillers.contains(&word)
}

pub(crate) fn is_german(lang: Option<&str>) -> bool {
    primary(lang).as_deref() == Some("de")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn language_tags_resolve_by_primary_subtag() {
        assert!(std::ptr::eq(table(Some("de-AT")), &DE));
        assert!(std::ptr::eq(table(Some("DE")), &DE));
        assert!(std::ptr::eq(table(Some("fr")), &EN));
        assert!(std::ptr::eq(table(None), &EN));
        assert!(std::ptr::eq(table(Some("")), &EN));
    }

    #[test]
    fn command_phrases_are_ordered_longest_first() {
        for t in [&EN, &DE] {
            let lens: Vec<usize> = t.commands.iter().map(|c| c.words.len()).collect();
            let mut sorted = lens.clone();
            sorted.sort_by(|a, b| b.cmp(a));
            assert_eq!(lens, sorted);
        }
    }

    #[test]
    fn table_entries_are_lower_case() {
        for t in [&EN, &DE] {
            let all = t
                .fillers
                .iter()
                .chain(t.stutter_keep)
                .chain(t.not_before_command)
                .chain(t.not_after_command)
                .chain(t.clause_starters)
                .chain(t.phrase_fillers.iter().flat_map(|p| p.iter()))
                .chain(t.cues.iter().flat_map(|c| c.words.iter()))
                .chain(t.commands.iter().flat_map(|c| c.words.iter()));
            for w in all {
                assert_eq!(*w, w.to_lowercase(), "{w}");
            }
        }
    }
}
