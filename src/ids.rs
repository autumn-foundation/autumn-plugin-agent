//! Strongly typed identifiers for runs and sessions.
//!
//! Both are thin `String` newtypes. They serialize as plain strings, so they
//! fit a database column or a job payload without a custom codec.

use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{SystemTime, UNIX_EPOCH};

use serde::{Deserialize, Serialize};

/// Identifies one agent run.
///
/// Tools receive it in [`ToolContext`](crate::tools::ToolContext) and can use
/// it, with the call id, as an idempotency key. A paused run keeps its id
/// across [`Agent::resume`](crate::agent::Agent::resume).
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct RunId(String);

impl RunId {
    /// Wrap an id the caller already owns (a job id, a request id).
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// Make a new process-unique id: `run_<unix-nanos-hex>_<counter-hex>`.
    ///
    /// The counter makes ids unique inside one process. Across replicas the
    /// nanosecond clock makes a clash unlikely, not impossible: pass your own
    /// id with [`RunId::new`] when you need a global guarantee.
    #[must_use]
    pub fn generate() -> Self {
        static COUNTER: AtomicU64 = AtomicU64::new(0);
        let nanos = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_or(0, |elapsed| elapsed.as_nanos());
        let count = COUNTER.fetch_add(1, Ordering::Relaxed);
        Self(format!("run_{nanos:x}_{count:x}"))
    }

    /// The id as a string slice.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for RunId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

/// Identifies one persisted conversation.
///
/// The app picks the key. Map a chat thread, an email `Message-ID` chain, or
/// a user id to a `SessionId`, and every run with that id continues the same
/// transcript.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct SessionId(String);

impl SessionId {
    /// Wrap an app-chosen session key.
    #[must_use]
    pub fn new(id: impl Into<String>) -> Self {
        Self(id.into())
    }

    /// The id as a string slice.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

impl std::fmt::Display for SessionId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(&self.0)
    }
}

impl From<&str> for SessionId {
    fn from(id: &str) -> Self {
        Self::new(id)
    }
}

impl From<String> for SessionId {
    fn from(id: String) -> Self {
        Self(id)
    }
}

#[cfg(test)]
mod tests;
