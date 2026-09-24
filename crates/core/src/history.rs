//! The last N utterances, so "paste last" recovers text whose insertion failed even after
//! a later dictation succeeded.

use std::collections::VecDeque;

use crate::UtteranceId;
use crate::insert::InsertOutcome;
use crate::normalize::Provenance;

#[derive(Debug, Clone, PartialEq)]
pub struct HistoryEntry {
    pub id: UtteranceId,
    pub raw: String,
    pub text: String,
    pub provenance: Provenance,
    /// The next utterance only gets this one as context when it goes to the same app.
    pub exe: Option<String>,
    /// `None` while insertion is in flight, or when it ended in an error.
    pub outcome: Option<InsertOutcome>,
}

#[derive(Debug, Clone)]
pub struct History {
    cap: usize,
    entries: VecDeque<HistoryEntry>,
}

impl History {
    /// A capacity of zero is raised to one: "paste last" must always have something.
    pub fn new(cap: usize) -> Self {
        let cap = cap.max(1);
        Self {
            cap,
            entries: VecDeque::with_capacity(cap),
        }
    }

    pub fn push(&mut self, entry: HistoryEntry) {
        if self.entries.len() == self.cap {
            self.entries.pop_front();
        }
        self.entries.push_back(entry);
    }

    pub fn set_outcome(&mut self, id: UtteranceId, outcome: InsertOutcome) {
        if let Some(e) = self.entries.iter_mut().rev().find(|e| e.id == id) {
            e.outcome = Some(outcome);
        }
    }

    pub fn last(&self) -> Option<&HistoryEntry> {
        self.entries.back()
    }

    pub fn iter(&self) -> impl DoubleEndedIterator<Item = &HistoryEntry> {
        self.entries.iter()
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn entry(n: u64) -> HistoryEntry {
        HistoryEntry {
            id: UtteranceId(n),
            raw: format!("raw {n}"),
            text: format!("text {n}"),
            provenance: Provenance::Rules,
            exe: None,
            outcome: None,
        }
    }

    #[test]
    fn ring_keeps_the_newest() {
        let mut h = History::new(2);
        for n in 1..=3 {
            h.push(entry(n));
        }
        assert_eq!(h.len(), 2);
        assert_eq!(h.iter().next().unwrap().id, UtteranceId(2));
        assert_eq!(h.last().unwrap().text, "text 3");
        h.set_outcome(UtteranceId(2), InsertOutcome::Typed);
        assert_eq!(h.iter().next().unwrap().outcome, Some(InsertOutcome::Typed));
        assert_eq!(History::new(0).cap, 1);
    }
}
