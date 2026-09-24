//! Cuts a recording into speech segments as it arrives, so closed segments can be
//! transcribed while the key is still held.

use std::time::Duration;

use crate::stt::SAMPLE_RATE;

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

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SegmenterConfig {
    /// Voiced time a segment needs before a pause may close it; shorter speech stays open
    /// and joins the next segment rather than being dropped.
    pub min_speech: Duration,
    pub min_pause: Duration,
    pub pad: Duration,
    /// Longer speech without a pause is split at the quietest point of its second half.
    pub max_segment: Duration,
}

impl Default for SegmenterConfig {
    fn default() -> Self {
        Self {
            min_speech: Duration::from_millis(300),
            min_pause: Duration::from_millis(400),
            pad: Duration::from_millis(200),
            max_segment: Duration::from_secs(20),
        }
    }
}

/// A VAD reports a start only after confirming it; audio this far behind the newest
/// sample is kept while silent so a late start still gets its padding.
const LOOKBACK: usize = SAMPLE_RATE as usize;
const QUIET_FRAME: usize = SAMPLE_RATE as usize / 50;

fn samples(d: Duration) -> usize {
    (d.as_secs_f64() * SAMPLE_RATE as f64).round() as usize
}

/// Positions are absolute sample offsets from the start of the recording.
#[derive(Debug, Clone)]
pub struct Segmenter {
    min_speech: usize,
    min_pause: usize,
    pad: usize,
    max_segment: usize,
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
    pub fn push(&mut self, pcm: &[f32], events: &[VadEvent]) -> Vec<Vec<f32>> {
        self.buf.extend_from_slice(pcm);
        let mut out = Vec::new();
        for e in events {
            match *e {
                VadEvent::SpeechStart { at } => {
                    let s = samples(at).clamp(self.cut, self.fed());
                    self.close_if_paused(s, &mut out);
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
        self.close_if_paused(self.fed(), &mut out);
        self.split_long(&mut out);
        if self.begin.is_none() {
            self.cut = self.cut.max(self.fed().saturating_sub(self.pad + LOOKBACK));
        }
        self.compact();
        out
    }

    /// `rest` is audio the detector never saw. Returns the tail, split like any segment,
    /// or nothing when no speech started after the last closed segment.
    pub fn finish(&mut self, rest: &[f32]) -> Vec<Vec<f32>> {
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

    fn close_if_paused(&mut self, now: usize, out: &mut Vec<Vec<f32>>) {
        let Some(begin) = self.begin else {
            return;
        };
        if self.run_start.is_some()
            || self.voiced < self.min_speech
            || now.saturating_sub(self.last_end) < self.min_pause
        {
            return;
        }
        let end = (self.last_end + self.pad).min(now).max(begin);
        self.emit(begin, end, out);
        self.cut = end;
        self.begin = None;
        self.voiced = 0;
    }

    fn split_long(&mut self, out: &mut Vec<Vec<f32>>) {
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

    fn emit(&self, mut begin: usize, end: usize, out: &mut Vec<Vec<f32>>) {
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

    fn slice(&self, from: usize, to: usize) -> Vec<f32> {
        self.buf[from - self.base..to - self.base].to_vec()
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
#[derive(Debug, Clone, Default)]
pub struct Stitcher {
    texts: Vec<Option<String>>,
    language: Option<String>,
    inference: Duration,
}

impl Stitcher {
    /// Returns the new segment's index.
    pub fn expect(&mut self) -> u32 {
        self.texts.push(None);
        (self.texts.len() - 1) as u32
    }

    pub fn expected(&self) -> usize {
        self.texts.len()
    }

    pub fn oldest_missing(&self) -> Option<u32> {
        self.texts
            .iter()
            .position(Option::is_none)
            .map(|i| i as u32)
    }

    /// Unknown indices and repeats are ignored; returns whether the result was taken.
    pub fn set(
        &mut self,
        index: u32,
        text: String,
        language: Option<String>,
        inference: Duration,
    ) -> bool {
        let Some(slot @ None) = self.texts.get_mut(index as usize) else {
            return false;
        };
        *slot = Some(text);
        if self.language.is_none() {
            self.language = language;
        }
        self.inference += inference;
        true
    }

    pub fn is_complete(&self) -> bool {
        self.texts.iter().all(Option::is_some)
    }

    /// Segments in order, joined with one space; empty segments leave no gap.
    pub fn text(&self) -> String {
        self.texts
            .iter()
            .flatten()
            .map(|t| t.trim())
            .filter(|t| !t.is_empty())
            .collect::<Vec<_>>()
            .join(" ")
    }

    pub fn language(&self) -> Option<&str> {
        self.language.as_deref()
    }

    pub fn inference(&self) -> Duration {
        self.inference
    }
}

#[cfg(test)]
mod tests {
    use super::*;

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

    fn span(seg: &[f32]) -> (usize, usize) {
        let first = seg[0] as usize;
        (first * 1000 / SR, (first + seg.len()) * 1000 / SR)
    }

    #[test]
    fn a_pause_after_enough_speech_closes_a_padded_segment() {
        let mut s = Segmenter::new(&SegmenterConfig::default());
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
        let mut s = Segmenter::new(&SegmenterConfig::default());
        let out = s.push(&ramp(0, 2000), &[start(500), end(700)]);
        assert!(out.is_empty(), "200 ms of speech is below min_speech");
        let out = s.push(&ramp(2000, 3500), &[start(2000), end(2800)]);
        assert_eq!(out.len(), 1);
        assert_eq!(span(&out[0]), (300, 3000));
    }

    #[test]
    fn a_start_after_the_pause_closes_the_previous_segment_first() {
        let mut s = Segmenter::new(&SegmenterConfig::default());
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
        let mut s = Segmenter::new(&SegmenterConfig::default());
        assert!(s.push(&ramp(0, 1000), &[start(800)]).is_empty());
        let tail = s.finish(&ramp(1000, 1100));
        assert_eq!(tail.len(), 1);
        assert_eq!(span(&tail[0]), (600, 1100));
    }

    #[test]
    fn long_silence_is_not_buffered() {
        let mut s = Segmenter::new(&SegmenterConfig::default());
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
        let mut s = Segmenter::new(&SegmenterConfig::default());
        s.push(&ramp(0, 5000), &[]);
        s.push(&ramp(5000, 5300), &[start(4900)]);
        let tail = s.finish(&[]);
        assert_eq!(span(&tail[0]), (4700, 5300));
    }

    #[test]
    fn unbroken_speech_splits_at_the_quietest_point() {
        let cfg = SegmenterConfig {
            max_segment: Duration::from_secs(2),
            ..Default::default()
        };
        let mut s = Segmenter::new(&cfg);
        let mut pcm = vec![0.5; ms(3000)];
        pcm[ms(1500)..ms(1520)].fill(0.0);
        let out = s.push(&pcm, &[start(0)]);
        assert_eq!(out.len(), 1);
        assert_eq!(out[0].len(), ms(1510));
        let tail = s.finish(&[]);
        assert_eq!(tail.len(), 1);
        assert_eq!(tail[0].len(), ms(1490));
        assert_eq!(
            out[0].len() + tail[0].len(),
            pcm.len(),
            "nothing lost or doubled"
        );
    }

    #[test]
    fn a_long_closed_segment_is_split_too() {
        let cfg = SegmenterConfig {
            max_segment: Duration::from_secs(2),
            ..Default::default()
        };
        let mut s = Segmenter::new(&cfg);
        let pcm = vec![0.5; ms(4000)];
        let out = s.push(&pcm, &[start(0), end(3000)]);
        assert!(out.iter().all(|seg| seg.len() <= ms(2000)), "{out:?}");
        assert_eq!(out.iter().map(Vec::len).sum::<usize>(), ms(3200));
    }

    #[test]
    fn stitcher_orders_and_joins() {
        let mut st = Stitcher::default();
        assert_eq!(st.expect(), 0);
        assert_eq!(st.expect(), 1);
        assert_eq!(st.expect(), 2);
        assert!(st.set(2, "three".into(), None, at(5)));
        assert!(st.set(0, " one ".into(), Some("en".into()), at(5)));
        assert!(!st.set(0, "again".into(), None, at(5)));
        assert!(!st.set(7, "unknown".into(), None, at(5)));
        assert!(!st.is_complete());
        assert_eq!(st.oldest_missing(), Some(1));
        assert!(st.set(1, "".into(), Some("de".into()), at(5)));
        assert!(st.is_complete());
        assert_eq!(st.text(), "one three");
        assert_eq!(st.language(), Some("en"));
        assert_eq!(st.inference(), at(15));
    }
}
