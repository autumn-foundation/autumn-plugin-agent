//! Loop detection: stop an agent that repeats the same tool call.
//!
//! A model sometimes polls a tool that never changes, or bounces between two
//! calls. Each repeat costs a full model call. The [`LoopGuard`] remembers a
//! fingerprint of the recent calls — tool name, arguments, *and* result — and
//! acts on repeats:
//!
//! * at `warn_after` identical fingerprints it appends a note to the tool
//!   result, telling the model to change approach;
//! * at `stop_after` it ends the run with
//!   [`AgentOutcome::LoopDetected`](crate::agent::AgentOutcome::LoopDetected).
//!
//! The result is part of the fingerprint, so polling a tool whose answer
//! changes is never a loop. Ping-pong between two calls is caught too: both
//! fingerprints repeat inside the window.

use std::collections::VecDeque;
use std::hash::{Hash, Hasher};

/// Loop-detection thresholds.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LoopGuard {
    /// Identical calls (in the window) before the loop adds a warning note.
    pub warn_after: u32,
    /// Identical calls (in the window) before the run stops.
    pub stop_after: u32,
    /// How many recent calls to remember.
    pub window: usize,
}

impl Default for LoopGuard {
    fn default() -> Self {
        Self {
            warn_after: 3,
            stop_after: 5,
            window: 30,
        }
    }
}

impl LoopGuard {
    /// A guard that never fires.
    #[must_use]
    pub const fn disabled() -> Self {
        Self {
            warn_after: u32::MAX,
            stop_after: u32::MAX,
            window: 0,
        }
    }
}

/// What the guard says about one recorded call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LoopVerdict {
    /// Nothing unusual.
    Ok,
    /// The call repeated `repeats` times: warn the model.
    Warn {
        /// Identical calls in the window, this one included.
        repeats: u32,
    },
    /// The call repeated `repeats` times: stop the run.
    Stop {
        /// Identical calls in the window, this one included.
        repeats: u32,
    },
}

/// Rolling fingerprint history for one run.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub(crate) struct LoopTracker {
    recent: VecDeque<u64>,
}

impl LoopTracker {
    /// Rebuild a tracker from fingerprints saved in a paused run.
    pub(crate) fn from_saved(saved: Vec<u64>) -> Self {
        Self {
            recent: saved.into(),
        }
    }

    /// The fingerprints, oldest first, for saving with a paused run.
    pub(crate) fn saved(&self) -> Vec<u64> {
        self.recent.iter().copied().collect()
    }

    /// Record one executed call and judge it.
    pub(crate) fn record(
        &mut self,
        guard: &LoopGuard,
        name: &str,
        arguments: &serde_json::Value,
        result: &str,
    ) -> LoopVerdict {
        if guard.window == 0 {
            return LoopVerdict::Ok;
        }
        let print = fingerprint(name, arguments, result);
        self.recent.push_back(print);
        while self.recent.len() > guard.window {
            self.recent.pop_front();
        }
        let repeats = u32::try_from(self.recent.iter().filter(|seen| **seen == print).count())
            .unwrap_or(u32::MAX);
        if repeats >= guard.stop_after {
            LoopVerdict::Stop { repeats }
        } else if repeats >= guard.warn_after {
            LoopVerdict::Warn { repeats }
        } else {
            LoopVerdict::Ok
        }
    }
}

/// Hash a call's name, arguments, and result.
///
/// `serde_json` keeps object keys sorted, so equal arguments always print
/// the same. `DefaultHasher::new()` uses fixed keys, so fingerprints stay
/// stable inside one build — enough to resume a paused run on the same
/// deployment.
fn fingerprint(name: &str, arguments: &serde_json::Value, result: &str) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    name.hash(&mut hasher);
    arguments.to_string().hash(&mut hasher);
    result.hash(&mut hasher);
    hasher.finish()
}

/// The note appended to a repeated call's result.
pub(crate) fn warning_note(name: &str, repeats: u32) -> String {
    format!(
        "\n[harness] You called {name} {repeats} times with the same arguments and got the same result. Do not call it again with these arguments: change approach or answer with what you have."
    )
}

#[cfg(test)]
mod tests;
