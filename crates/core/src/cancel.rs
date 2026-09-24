//! Cooperative cancellation and deadlines for every long call.
//!
//! One token per utterance. The pipeline cancels it on Escape; engines, normalizers and
//! the inserter poll it at their own checkpoints. A deadline travels with the token so a
//! backend applies it as its own timeout instead of every caller wrapping calls in timers.

use std::sync::Arc;
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

/// Why a call stopped early.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
pub enum Cancelled {
    #[error("cancelled")]
    Requested,
    #[error("deadline passed")]
    Deadline,
}

/// Cheap to clone; clones share one flag, so cancelling any clone cancels them all.
/// The deadline is per clone: [`CancelToken::with_deadline`] narrows it for one call
/// without affecting the caller's copy.
#[derive(Debug, Clone, Default)]
pub struct CancelToken {
    flag: Arc<AtomicBool>,
    deadline: Option<Instant>,
}

impl CancelToken {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn cancel(&self) {
        self.flag.store(true, Ordering::Release);
    }

    /// Whether [`CancelToken::cancel`] was called on any clone. A passed deadline does
    /// not count; use [`CancelToken::checkpoint`] to test both.
    pub fn is_cancelled(&self) -> bool {
        self.flag.load(Ordering::Acquire)
    }

    pub fn deadline(&self) -> Option<Instant> {
        self.deadline
    }

    /// A clone sharing the cancel flag whose deadline is the earlier of the existing one
    /// and `at`. Never extends: a callee cannot grant itself more time than its caller had.
    pub fn with_deadline(&self, at: Instant) -> Self {
        Self {
            flag: Arc::clone(&self.flag),
            deadline: Some(self.deadline.map_or(at, |d| d.min(at))),
        }
    }

    pub fn with_timeout(&self, timeout: Duration) -> Self {
        self.with_deadline(Instant::now() + timeout)
    }

    /// Time left before the deadline, `None` without one. Zero once it has passed.
    pub fn remaining(&self) -> Option<Duration> {
        self.deadline
            .map(|d| d.saturating_duration_since(Instant::now()))
    }

    pub fn is_expired(&self) -> bool {
        self.deadline.is_some_and(|d| Instant::now() >= d)
    }

    /// `Err` if cancelled or past the deadline. Cancellation wins when both hold: the user
    /// asked for it, and the message they see should say so.
    pub fn checkpoint(&self) -> Result<(), Cancelled> {
        if self.is_cancelled() {
            Err(Cancelled::Requested)
        } else if self.is_expired() {
            Err(Cancelled::Deadline)
        } else {
            Ok(())
        }
    }

    /// The timeout to give a blocking call: `cap`, shortened to the time left. `Err` when
    /// there is nothing left to give, so the call is never started with a zero timeout
    /// (which some clients read as "no timeout").
    pub fn timeout_within(&self, cap: Duration) -> Result<Duration, Cancelled> {
        self.checkpoint()?;
        let t = self.remaining().map_or(cap, |r| r.min(cap));
        if t.is_zero() {
            Err(Cancelled::Deadline)
        } else {
            Ok(t)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn clones_share_the_flag() {
        let a = CancelToken::new();
        let b = a.clone();
        assert!(!b.is_cancelled());
        assert_eq!(b.checkpoint(), Ok(()));
        a.cancel();
        assert!(b.is_cancelled());
        assert_eq!(b.checkpoint(), Err(Cancelled::Requested));
    }

    #[test]
    fn deadline_narrows_and_never_extends() {
        let base = CancelToken::new();
        let now = Instant::now();
        let near = base.with_deadline(now + Duration::from_secs(1));
        let still_near = near.with_deadline(now + Duration::from_secs(60));
        assert_eq!(still_near.deadline(), Some(now + Duration::from_secs(1)));
        assert_eq!(base.deadline(), None);
        near.cancel();
        assert!(
            base.is_cancelled(),
            "a narrowed clone still shares the flag"
        );
    }

    #[test]
    fn passed_deadline_fails_the_checkpoint() {
        let t = CancelToken::new().with_deadline(Instant::now() - Duration::from_millis(1));
        assert!(t.is_expired());
        assert!(!t.is_cancelled());
        assert_eq!(t.checkpoint(), Err(Cancelled::Deadline));
        assert_eq!(t.remaining(), Some(Duration::ZERO));
        assert_eq!(
            t.timeout_within(Duration::from_secs(5)),
            Err(Cancelled::Deadline)
        );
        t.cancel();
        assert_eq!(t.checkpoint(), Err(Cancelled::Requested));
    }

    #[test]
    fn timeout_is_capped_by_the_deadline() {
        let t = CancelToken::new();
        assert_eq!(
            t.timeout_within(Duration::from_secs(5)),
            Ok(Duration::from_secs(5))
        );
        let t = t.with_timeout(Duration::from_millis(200));
        let got = t.timeout_within(Duration::from_secs(5)).unwrap();
        assert!(got <= Duration::from_millis(200) && got > Duration::ZERO);
    }
}
