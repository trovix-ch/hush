//! Cuts a recording into speech segments as it arrives, so closed segments can be
//! transcribed while the key is still held, and joins their transcripts again.

use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::stt::{SAMPLE_RATE, Transcript};

/// Offsets count from the first sample pushed since the last reset.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VadEvent {
    SpeechStart {
        at: Duration,
    },
    /// End of the last speech frame, before hangover.
    SpeechEnd {
        at: Duration,
    },
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(default, deny_unknown_fields)]
pub struct SegmenterConfig {
    /// Voiced time a segment needs before a pause may close it; shorter speech stays open
    /// and joins the next segment rather than being dropped.
    #[serde(rename = "min_speech_ms", with = "crate::config::millis")]
    pub min_speech: Duration,
    #[serde(rename = "min_pause_ms", with = "crate::config::millis")]
    pub min_pause: Duration,
    #[serde(rename = "pad_ms", with = "crate::config::millis")]
    pub pad: Duration,
    /// Longer speech without a pause is split at the quietest point of its second half.
    #[serde(rename = "max_segment_ms", with = "crate::config::millis")]
    pub max_segment: Duration,
    /// How late the detector reports a speech start. A pause is not judged over until
    /// this much more audio has arrived, so a start already spoken but not yet reported
    /// cannot be mistaken for silence.
    #[serde(rename = "vad_latency_ms", with = "crate::config::millis")]
    pub vad_latency: Duration,
    /// Silence between the last word of one segment and the first word of the next
    /// below which the boundary is read as mid-sentence and the engine's sentence end
    /// there is undone.
    #[serde(rename = "join_gap_ms", with = "crate::config::millis")]
    pub join_gap: Duration,
    /// A mid-sentence boundary with at least this much silence is joined with a comma.
    #[serde(rename = "comma_gap_ms", with = "crate::config::millis")]
    pub comma_gap: Duration,
}

impl Default for SegmenterConfig {
    fn default() -> Self {
        Self {
            min_speech: Duration::from_millis(300),
            min_pause: Duration::from_millis(700),
            pad: Duration::from_millis(200),
            max_segment: Duration::from_secs(20),
            vad_latency: Duration::from_millis(100),
            // Below the shortest word gap a 700 ms pause produced (560 ms), so only
            // boundaries cut inside speech are joined: pause length measured no guide to
            // punctuation, with sentence ends at 560-640 ms and commas at 830-1020 ms.
            join_gap: Duration::from_millis(400),
            comma_gap: Duration::from_millis(300),
        }
    }
}

/// A closed stretch of the recording.
#[derive(Debug, Clone, PartialEq)]
pub struct Cut {
    /// Absolute sample offset of `pcm[0]` in the recording.
    pub offset: usize,
    pub pcm: Vec<f32>,
}

/// A VAD reports a start only after confirming it; audio this far behind the newest
/// sample is kept while silent so a late start still gets its padding.
const LOOKBACK: usize = SAMPLE_RATE as usize;
const QUIET_FRAME: usize = SAMPLE_RATE as usize / 50;

fn samples(d: Duration) -> usize {
    (d.as_secs_f64() * SAMPLE_RATE as f64).round() as usize
}

fn duration(samples: usize) -> Duration {
    Duration::from_secs_f64(samples as f64 / SAMPLE_RATE as f64)
}

/// Positions are absolute sample offsets from the start of the recording.
#[derive(Debug, Clone)]
pub struct Segmenter {
    min_speech: usize,
    min_pause: usize,
    pad: usize,
    max_segment: usize,
    vad_latency: usize,
    /// Absolute offset of `buf[0]`.
    base: usize,
    buf: Vec<f32>,
    /// Everything before this was emitted or discarded as silence.
    cut: usize,
    /// Padded start of the open segment; `Some` once it has any speech.
    begin: Option<usize>,
    run_start: Option<usize>,
    last_end: usize,
    voiced: usize,
}

impl Segmenter {
    pub fn new(cfg: &SegmenterConfig) -> Self {
        Self {
            min_speech: samples(cfg.min_speech),
            min_pause: samples(cfg.min_pause),
            pad: samples(cfg.pad),
            max_segment: samples(cfg.max_segment).max(2 * QUIET_FRAME),
            vad_latency: samples(cfg.vad_latency),
            base: 0,
            buf: Vec::new(),
            cut: 0,
            begin: None,
            run_start: None,
            last_end: 0,
            voiced: 0,
        }
    }

    /// Samples received so far.
    pub fn fed(&self) -> usize {
        self.base + self.buf.len()
    }

    /// `events` are the detector's output for audio up to and including `pcm`. Returns
    /// the segments that closed, oldest first.
    pub fn push(&mut self, pcm: &[f32], events: &[VadEvent]) -> Vec<Cut> {
        self.buf.extend_from_slice(pcm);
        let mut out = Vec::new();
        for e in events {
            match *e {
                VadEvent::SpeechStart { at } => {
                    let s = samples(at).clamp(self.cut, self.fed());
                    self.close_if_paused(s, 0, &mut out);
                    if self.begin.is_none() {
                        self.begin = Some(s.saturating_sub(self.pad).max(self.cut));
                    }
                    self.run_start.get_or_insert(s);
                }
                VadEvent::SpeechEnd { at } => {
                    if let Some(r) = self.run_start.take() {
                        let end = samples(at).clamp(r, self.fed());
                        self.voiced += end - r;
                        self.last_end = end;
                    }
                }
            }
            self.split_long(&mut out);
        }
        // A start inside the pause may not have been reported yet; judging the pause on
        // the audio alone made the same recording segment differently run to run,
        // depending on when its blocks arrived.
        self.close_if_paused(self.fed(), self.vad_latency, &mut out);
        self.split_long(&mut out);
        if self.begin.is_none() {
            self.cut = self.cut.max(self.fed().saturating_sub(self.pad + LOOKBACK));
        }
        self.compact();
        out
    }

    /// `rest` is audio the detector never saw. Returns the tail, split like any segment,
    /// or nothing when no speech started after the last closed segment.
    pub fn finish(&mut self, rest: &[f32]) -> Vec<Cut> {
        self.buf.extend_from_slice(rest);
        let mut out = Vec::new();
        if let Some(begin) = self.begin.take() {
            self.emit(begin, self.fed(), &mut out);
        }
        self.run_start = None;
        self.cut = self.fed();
        self.compact();
        out
    }

    /// `margin` is audio past `min_pause` that must have arrived before the pause counts.
    fn close_if_paused(&mut self, now: usize, margin: usize, out: &mut Vec<Cut>) {
        let Some(begin) = self.begin else {
            return;
        };
        if self.run_start.is_some()
            || self.voiced < self.min_speech
            || now.saturating_sub(self.last_end) < self.min_pause + margin
        {
            return;
        }
        let end = (self.last_end + self.pad).min(now).max(begin);
        self.emit(begin, end, out);
        self.cut = end;
        self.begin = None;
        self.voiced = 0;
    }

    fn split_long(&mut self, out: &mut Vec<Cut>) {
        while let Some(begin) = self.begin
            && self.fed() - begin >= self.max_segment
        {
            let split = self.quietest(begin + self.max_segment / 2, begin + self.max_segment);
            out.push(self.slice(begin, split));
            self.cut = split;
            self.begin = Some(split);
            self.run_start = self.run_start.map(|r| r.max(split));
            self.voiced = self.last_end.saturating_sub(split);
        }
    }

    fn emit(&self, mut begin: usize, end: usize, out: &mut Vec<Cut>) {
        while end - begin > self.max_segment {
            let split = self.quietest(begin + self.max_segment / 2, begin + self.max_segment);
            out.push(self.slice(begin, split));
            begin = split;
        }
        if end > begin {
            out.push(self.slice(begin, end));
        }
    }

    /// Centre of the lowest-energy frame in `[lo, hi)`.
    fn quietest(&self, lo: usize, hi: usize) -> usize {
        let hi = hi.min(self.fed());
        let mut best = (f32::INFINITY, hi);
        let mut at = lo;
        while at + QUIET_FRAME <= hi {
            let energy: f32 = self.buf[at - self.base..at - self.base + QUIET_FRAME]
                .iter()
                .map(|s| s * s)
                .sum();
            if energy < best.0 {
                best = (energy, at + QUIET_FRAME / 2);
            }
            at += QUIET_FRAME;
        }
        best.1
    }

    fn slice(&self, from: usize, to: usize) -> Cut {
        Cut {
            offset: from,
            pcm: self.buf[from - self.base..to - self.base].to_vec(),
        }
    }

    fn compact(&mut self) {
        let keep_from = self.begin.unwrap_or(self.cut).min(self.cut);
        if keep_from > self.base {
            self.buf.drain(..keep_from - self.base);
            self.base = keep_from;
        }
    }
}

/// Results for one utterance's segments, held until every one has arrived.
///
/// The engine ends every segment as a complete sentence because its audio ends there.
/// Where the words on either side of a boundary are closer than `join_gap`, the speaker
/// did not stop, so that sentence end is undone.
#[derive(Debug, Clone)]
pub struct Stitcher {
    join_gap: Duration,
    comma_gap: Duration,
    vocabulary: Vec<String>,
    parts: Vec<Part>,
    language: Option<String>,
    inference: Duration,
}

#[derive(Debug, Clone)]
struct Part {
    offset: Duration,
    heard: Option<Heard>,
}

#[derive(Debug, Clone)]
struct Heard {
    text: String,
    /// First word's start and last word's end, from the start of the recording.
    speech: Option<(Duration, Duration)>,
}

impl Stitcher {
    pub fn new(cfg: &SegmenterConfig, vocabulary: &[String]) -> Self {
        Self {
            join_gap: cfg.join_gap,
            comma_gap: cfg.comma_gap,
            vocabulary: vocabulary.to_vec(),
            parts: Vec::new(),
            language: None,
            inference: Duration::ZERO,
        }
    }

    /// `offset` is the segment's first sample in the recording. Returns its index.
    pub fn expect(&mut self, offset: usize) -> u32 {
        self.parts.push(Part {
            offset: duration(offset),
            heard: None,
        });
        (self.parts.len() - 1) as u32
    }

    pub fn expected(&self) -> usize {
        self.parts.len()
    }

    pub fn oldest_missing(&self) -> Option<u32> {
        self.parts
            .iter()
            .position(|p| p.heard.is_none())
            .map(|i| i as u32)
    }

    /// Unknown indices and repeats are ignored; returns whether the result was taken.
    pub fn set(&mut self, index: u32, t: Transcript) -> bool {
        let Some(part) = self.parts.get_mut(index as usize) else {
            return false;
        };
        if part.heard.is_some() {
            return false;
        }
        let timed = if t.words.is_empty() {
            &t.segments
        } else {
            &t.words
        };
        let speech = timed
            .first()
            .zip(timed.last())
            .map(|(a, b)| (part.offset + a.start, part.offset + b.end));
        part.heard = Some(Heard {
            text: t.text,
            speech,
        });
        if self.language.is_none() {
            self.language = t.language;
        }
        self.inference += t.inference_time;
        true
    }

    pub fn is_complete(&self) -> bool {
        self.parts.iter().all(|p| p.heard.is_some())
    }

    /// Silence at each boundary between non-empty segments, where both sides were timed.
    pub fn gaps(&self) -> Vec<Option<Duration>> {
        let mut out = Vec::new();
        let mut prev: Option<Option<Duration>> = None;
        for h in self.heard() {
            if let Some(end) = prev {
                out.push(end.zip(h.speech).map(|(e, (s, _))| s.saturating_sub(e)));
            }
            prev = Some(h.speech.map(|s| s.1));
        }
        out
    }

    /// Segments in order; empty segments leave no gap.
    pub fn text(&self) -> String {
        let mut out = String::new();
        let mut gaps = self.gaps().into_iter();
        for h in self.heard() {
            let text = h.text.trim();
            if out.is_empty() {
                out.push_str(text);
            } else {
                self.join(&mut out, text, gaps.next().flatten());
            }
        }
        out
    }

    fn heard(&self) -> impl Iterator<Item = &Heard> {
        self.parts
            .iter()
            .filter_map(|p| p.heard.as_ref())
            .filter(|h| !h.text.trim().is_empty())
    }

    fn join(&self, out: &mut String, next: &str, gap: Option<Duration>) {
        let proper = next
            .split_whitespace()
            .next()
            .is_some_and(|w| self.capitalised_anyway(w));
        if let Some(gap) = gap.filter(|g| *g < self.join_gap) {
            let single_stop = out.ends_with('.') && !out.ends_with("..");
            if (single_stop && !proper) || out.ends_with('!') {
                out.pop();
            }
            if gap >= self.comma_gap && out.ends_with(char::is_alphanumeric) {
                out.push(',');
            }
        }
        // A segment the engine did not end as a sentence is continued by the next one,
        // whose capital comes only from starting the engine's input.
        let continues = out.ends_with(|c: char| c.is_alphanumeric() || c == ',');
        out.push(' ');
        if continues && !proper {
            out.push_str(&lowercase_first(next));
        } else {
            out.push_str(next);
        }
    }

    fn capitalised_anyway(&self, word: &str) -> bool {
        let w = word.trim_matches(|c: char| !c.is_alphanumeric() && c != '\'' && c != '’');
        w == "I"
            || w.starts_with("I'")
            || w.starts_with("I’")
            || self.vocabulary.iter().any(|v| {
                v.split_whitespace()
                    .next()
                    .is_some_and(|f| f.eq_ignore_ascii_case(w))
            })
    }

    pub fn language(&self) -> Option<&str> {
        self.language.as_deref()
    }

    pub fn inference(&self) -> Duration {
        self.inference
    }
}

/// Only a word whose one capital is its first letter can owe it to the sentence start;
/// acronyms and names like "gRPC" or "McDonald" keep theirs.
fn lowercase_first(s: &str) -> String {
    let word = s.split_whitespace().next().unwrap_or_default();
    let mut chars = word.chars();
    let Some(first) = chars.next() else {
        return s.to_string();
    };
    if !first.is_uppercase() || chars.any(char::is_uppercase) {
        return s.to_string();
    }
    let mut out: String = first.to_lowercase().collect();
    out.push_str(&s[first.len_utf8()..]);
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::stt::Segment;

    const SR: usize = SAMPLE_RATE as usize;

    fn ms(n: usize) -> usize {
        n * SR / 1000
    }

    fn at(n: usize) -> Duration {
        Duration::from_millis(n as u64)
    }

    fn start(n: usize) -> VadEvent {
        VadEvent::SpeechStart { at: at(n) }
    }

    fn end(n: usize) -> VadEvent {
        VadEvent::SpeechEnd { at: at(n) }
    }

    /// Sample `i` has value `i`, so a segment's first value is its absolute offset.
    fn ramp(from_ms: usize, to_ms: usize) -> Vec<f32> {
        (ms(from_ms)..ms(to_ms)).map(|i| i as f32).collect()
    }

    /// For ramp audio: checks the offset against the samples, returns the span in ms.
    fn span(seg: &Cut) -> (usize, usize) {
        assert_eq!(seg.pcm[0] as usize, seg.offset);
        (
            seg.offset * 1000 / SR,
            (seg.offset + seg.pcm.len()) * 1000 / SR,
        )
    }

    /// Pinned so the tests do not follow the shipped defaults.
    fn cfg() -> SegmenterConfig {
        SegmenterConfig {
            min_speech: Duration::from_millis(300),
            min_pause: Duration::from_millis(400),
            pad: Duration::from_millis(200),
            max_segment: Duration::from_secs(20),
            vad_latency: Duration::from_millis(100),
            join_gap: Duration::from_millis(600),
            comma_gap: Duration::from_millis(300),
        }
    }

    #[test]
    fn a_pause_after_enough_speech_closes_a_padded_segment() {
        let mut s = Segmenter::new(&cfg());
        assert!(s.push(&ramp(0, 1000), &[start(500)]).is_empty());
        assert!(
            s.push(&ramp(1000, 1500), &[end(1200)]).is_empty(),
            "pause 300 ms"
        );
        let out = s.push(&ramp(1500, 1700), &[]);
        assert_eq!(out.len(), 1);
        assert_eq!(span(&out[0]), (300, 1400));
        assert!(
            s.finish(&ramp(1700, 2000)).is_empty(),
            "no speech after the cut"
        );
    }

    #[test]
    fn short_speech_stays_open_and_joins_the_next() {
        let mut s = Segmenter::new(&cfg());
        let out = s.push(&ramp(0, 2000), &[start(500), end(700)]);
        assert!(out.is_empty(), "200 ms of speech is below min_speech");
        let out = s.push(&ramp(2000, 3500), &[start(2000), end(2800)]);
        assert_eq!(out.len(), 1);
        assert_eq!(span(&out[0]), (300, 3000));
    }

    #[test]
    fn a_start_after_the_pause_closes_the_previous_segment_first() {
        let mut s = Segmenter::new(&cfg());
        let out = s.push(
            &ramp(0, 2800),
            &[start(0), end(1000), start(1600), end(2500)],
        );
        assert_eq!(out.len(), 1);
        assert_eq!(span(&out[0]), (0, 1200));
        let tail = s.finish(&ramp(2800, 3200));
        assert_eq!(tail.len(), 1);
        assert_eq!(span(&tail[0]), (1400, 3200));
    }

    #[test]
    fn an_open_segment_becomes_the_tail() {
        let mut s = Segmenter::new(&cfg());
        assert!(s.push(&ramp(0, 1000), &[start(800)]).is_empty());
        let tail = s.finish(&ramp(1000, 1100));
        assert_eq!(tail.len(), 1);
        assert_eq!(span(&tail[0]), (600, 1100));
    }

    #[test]
    fn long_silence_is_not_buffered() {
        let mut s = Segmenter::new(&cfg());
        for i in 0..60 {
            s.push(&vec![0.0; SR], &[]);
            assert!(
                s.buf.len() <= SR + ms(200) + SR,
                "second {i}: {}",
                s.buf.len()
            );
        }
        assert!(s.finish(&[]).is_empty());
    }

    #[test]
    fn late_start_still_gets_its_padding() {
        let mut s = Segmenter::new(&cfg());
        s.push(&ramp(0, 5000), &[]);
        s.push(&ramp(5000, 5300), &[start(4900)]);
        let tail = s.finish(&[]);
        assert_eq!(span(&tail[0]), (4700, 5300));
    }

    #[test]
    fn unbroken_speech_splits_at_the_quietest_point() {
        let cfg = SegmenterConfig {
            max_segment: Duration::from_secs(2),
            ..cfg()
        };
        let mut s = Segmenter::new(&cfg);
        let mut pcm = vec![0.5; ms(3000)];
        pcm[ms(1500)..ms(1520)].fill(0.0);
        let out = s.push(&pcm, &[start(0)]);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].pcm.len(), ms(1510));
        let tail = s.finish(&[]);
        assert_eq!(tail.len(), 1);
        assert_eq!(tail[0].offset, ms(1510));
        assert_eq!(tail[0].pcm.len(), ms(1490));
        assert_eq!(
            out[0].pcm.len() + tail[0].pcm.len(),
            pcm.len(),
            "nothing lost or doubled"
        );
    }

    #[test]
    fn a_long_closed_segment_is_split_too() {
        let cfg = SegmenterConfig {
            max_segment: Duration::from_secs(2),
            ..cfg()
        };
        let mut s = Segmenter::new(&cfg);
        let pcm = vec![0.5; ms(4000)];
        let out = s.push(&pcm, &[start(0), end(3000)]);
        assert!(out.iter().all(|seg| seg.pcm.len() <= ms(2000)), "{out:?}");
        assert_eq!(out.iter().map(|c| c.pcm.len()).sum::<usize>(), ms(3200));
    }

    /// Delivers `events` as a detector would: a start once `confirm` more audio has
    /// arrived, an end once `hangover` has.
    fn feed_in_blocks(cfg: &SegmenterConfig, block_ms: usize, events: &[VadEvent]) -> Vec<Cut> {
        let (confirm, hangover) = (96, 96);
        let total = 4000;
        let mut s = Segmenter::new(cfg);
        let mut out = Vec::new();
        let mut next = 0;
        let mut fed = 0;
        while fed < total {
            let to = (fed + block_ms).min(total);
            let mut due = Vec::new();
            while let Some(e) = events.get(next) {
                let known_at = match *e {
                    VadEvent::SpeechStart { at } => at + Duration::from_millis(confirm),
                    VadEvent::SpeechEnd { at } => at + Duration::from_millis(hangover),
                };
                if known_at > Duration::from_millis(to as u64) {
                    break;
                }
                due.push(*e);
                next += 1;
            }
            out.extend(s.push(&ramp(fed, to), &due));
            fed = to;
        }
        out.extend(s.finish(&[]));
        out
    }

    #[test]
    fn a_pause_just_short_of_min_pause_never_closes_whatever_the_block_size() {
        // 380 ms of silence, below the 400 ms minimum; the next start is reported 96 ms
        // after it happened.
        let events = [start(0), end(1000), start(1380), end(2500)];
        for block in [10, 20, 32, 50, 70, 130] {
            let cuts = feed_in_blocks(&cfg(), block, &events);
            let spans: Vec<_> = cuts.iter().map(span).collect();
            assert_eq!(spans, vec![(0, 2700)], "{block} ms blocks");
        }
        let judged_on_arrival = SegmenterConfig {
            vad_latency: Duration::ZERO,
            ..cfg()
        };
        let counts: Vec<usize> = [20, 300]
            .into_iter()
            .map(|b| feed_in_blocks(&judged_on_arrival, b, &events).len())
            .collect();
        assert_ne!(counts[0], counts[1], "the race the latency margin removes");
    }

    fn word(start_ms: usize, end_ms: usize) -> Segment {
        Segment {
            text: "w".into(),
            start: at(start_ms),
            end: at(end_ms),
        }
    }

    /// `words` are relative to the segment's own audio.
    fn heard(text: &str, words: &[(usize, usize)]) -> Transcript {
        Transcript {
            text: text.into(),
            words: words.iter().map(|&(a, b)| word(a, b)).collect(),
            inference_time: at(5),
            ..Default::default()
        }
    }

    /// Two segments starting at 0 and 3000 ms, whose words are `gap_ms` apart.
    fn stitch(a: &str, b: &str, gap_ms: usize, vocabulary: &[&str]) -> String {
        let vocabulary: Vec<String> = vocabulary.iter().map(|v| v.to_string()).collect();
        let mut st = Stitcher::new(&cfg(), &vocabulary);
        st.expect(0);
        st.expect(ms(3000));
        let end_a = 3000 + 200 - gap_ms;
        st.set(0, heard(a, &[(200, 1000), (1000, end_a)]));
        st.set(1, heard(b, &[(200, 900), (900, 1500)]));
        assert_eq!(st.gaps(), vec![Some(at(gap_ms))]);
        st.text()
    }

    #[test]
    fn stitcher_orders_and_joins() {
        let mut st = Stitcher::new(&cfg(), &[]);
        assert_eq!(st.expect(0), 0);
        assert_eq!(st.expect(ms(2000)), 1);
        assert_eq!(st.expect(ms(4000)), 2);
        assert!(st.set(2, heard("three", &[])));
        assert!(st.set(
            0,
            Transcript {
                language: Some("en".into()),
                ..heard(" one ", &[])
            }
        ));
        assert!(!st.set(0, heard("again", &[])));
        assert!(!st.set(7, heard("unknown", &[])));
        assert!(!st.is_complete());
        assert_eq!(st.oldest_missing(), Some(1));
        assert!(st.set(
            1,
            Transcript {
                language: Some("de".into()),
                ..heard("", &[])
            }
        ));
        assert!(st.is_complete());
        assert_eq!(st.text(), "one three");
        assert_eq!(st.language(), Some("en"));
        assert_eq!(st.inference(), at(15));
    }

    #[test]
    fn a_short_gap_undoes_the_sentence_end() {
        assert_eq!(
            stitch("my fellow Americans.", "Ask not what", 450, &[]),
            "my fellow Americans, ask not what"
        );
        assert_eq!(
            stitch("Tuesday. No wait.", "Make it Wednesday.", 200, &[]),
            "Tuesday. No wait make it Wednesday."
        );
        assert_eq!(
            stitch("Ask not!", "What your", 350, &[]),
            "Ask not, what your"
        );
    }

    #[test]
    fn a_long_gap_keeps_the_engines_punctuation() {
        assert_eq!(
            stitch("Hi Sarah.", "Thanks for", 600, &[]),
            "Hi Sarah. Thanks for"
        );
        assert_eq!(stitch("Talk soon!", "Bye.", 900, &[]), "Talk soon! Bye.");
    }

    #[test]
    fn a_question_mark_always_stays() {
        assert_eq!(
            stitch("the Berlin trip?", "Also please", 100, &[]),
            "the Berlin trip? Also please"
        );
    }

    #[test]
    fn words_capitalised_for_their_own_reason_keep_the_capital_and_the_stop() {
        assert_eq!(
            stitch("on Tuesday.", "I am out", 200, &[]),
            "on Tuesday. I am out"
        );
        assert_eq!(
            stitch("on Tuesday.", "I'm out", 400, &[]),
            "on Tuesday. I'm out"
        );
        assert_eq!(
            stitch("deploy it to.", "Kubernetes today", 200, &["kubernetes"]),
            "deploy it to. Kubernetes today"
        );
        assert_eq!(
            stitch("run it on the", "GPU tonight", 200, &[]),
            "run it on the GPU tonight"
        );
    }

    #[test]
    fn a_segment_without_a_sentence_end_is_continued_whatever_the_gap() {
        assert_eq!(
            stitch("Also", "Please send it", 900, &[]),
            "Also please send it"
        );
        assert_eq!(
            stitch("If anything changes,", "Just give me", 800, &[]),
            "If anything changes, just give me"
        );
    }

    #[test]
    fn untimed_segments_keep_the_engines_punctuation() {
        let mut st = Stitcher::new(&cfg(), &[]);
        st.expect(0);
        st.expect(ms(3000));
        st.set(0, heard("Hello there.", &[]));
        st.set(1, heard("How are you?", &[(100, 800)]));
        assert_eq!(st.gaps(), vec![None]);
        assert_eq!(st.text(), "Hello there. How are you?");
    }

    #[test]
    fn engine_segments_stand_in_for_missing_word_times() {
        let mut st = Stitcher::new(&cfg(), &[]);
        st.expect(0);
        st.expect(ms(2000));
        st.set(
            0,
            Transcript {
                segments: vec![word(100, 1900)],
                ..heard("Americans.", &[])
            },
        );
        st.set(1, heard("Ask not", &[(200, 800)]));
        assert_eq!(st.text(), "Americans, ask not");
    }
}
