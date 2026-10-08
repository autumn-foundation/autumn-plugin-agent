//! Always-on behaviour: heartbeats, follow-ups, and delivery.
//!
//! An always-on agent wakes up by itself. This module gives it three ways:
//!
//! * **Heartbeat** — a [`Heartbeat`] registers an Autumn scheduled task
//!   (fixed delay or cron). Each tick runs the agent with a check-in prompt.
//!   If nothing needs attention, the agent answers [`HEARTBEAT_OK`] and the
//!   harness stays silent. Heartbeats run with read-only tools by default:
//!   the agent may look and take notes, not act.
//! * **Follow-ups** — the `schedule_followup` tool lets the agent book its
//!   own next wake-up ("check the deploy in 20 minutes") as a delayed
//!   `agent_run` job.
//! * **Precheck** — a cheap app callback can skip a heartbeat tick before any
//!   model call ("no new orders, nothing to look at").
//!
//! Whatever a background run has to say goes to a [`Delivery`]: the app's
//! bridge to mail, chat, push, or an in-app inbox. [`LogDelivery`] is the
//! default.

use futures::future::BoxFuture;
use serde::{Deserialize, Serialize};

use crate::agent::AgentOutcome;
use crate::error::AgentError;
use crate::ids::{RunId, SessionId};

/// The answer that means "nothing needs attention".
pub const HEARTBEAT_OK: &str = "HEARTBEAT_OK";

/// Longest extra text an answer may carry next to [`HEARTBEAT_OK`] and still
/// count as silent.
pub const SILENT_ACK_MAX_CHARS: usize = 300;

/// The default heartbeat prompt.
pub const DEFAULT_HEARTBEAT_PROMPT: &str = "This is a scheduled heartbeat check-in. Review your memory, your standing instructions, and anything your tools can show you. If something needs the user's attention, report it in a few sentences. If nothing needs attention, reply with exactly HEARTBEAT_OK.";

/// `true` when a background answer should not be delivered: it is
/// [`HEARTBEAT_OK`], or starts or ends with it and adds at most
/// [`SILENT_ACK_MAX_CHARS`] characters.
#[must_use]
pub fn is_silent(text: &str) -> bool {
    let text = text.trim();
    let rest = text
        .strip_prefix(HEARTBEAT_OK)
        .or_else(|| text.strip_suffix(HEARTBEAT_OK));
    rest.is_some_and(|rest| rest.trim().chars().count() <= SILENT_ACK_MAX_CHARS)
}

/// What started a background run that has something to say.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ReportSource {
    /// A heartbeat tick.
    Heartbeat,
    /// A follow-up the agent scheduled.
    Followup,
    /// An `agent_run` job the app enqueued with `deliver` set.
    Job,
    /// A resumed run.
    Resume,
}

/// A message from a background run to a person.
#[derive(Debug, Clone, PartialEq)]
pub struct Report {
    /// What started the run.
    pub source: ReportSource,
    /// The run's id.
    pub run_id: RunId,
    /// The session the run belongs to, if any. Use it to route the report
    /// back to the right chat thread.
    pub session_id: Option<SessionId>,
    /// Text for a person: the answer, or what waits for approval.
    pub text: String,
    /// The full outcome. A paused run carries the
    /// [`RunState`](crate::agent::RunState) to store for the reviewer.
    pub outcome: AgentOutcome,
}

/// Where background results go: mail, chat, push, an inbox table.
pub trait Delivery: Send + Sync + std::fmt::Debug {
    /// Deliver one report.
    fn deliver<'a>(&'a self, report: &'a Report) -> BoxFuture<'a, Result<(), AgentError>>;
}

/// Logs reports with `tracing`. The default [`Delivery`].
#[derive(Debug, Clone, Copy, Default)]
pub struct LogDelivery;

impl Delivery for LogDelivery {
    fn deliver<'a>(&'a self, report: &'a Report) -> BoxFuture<'a, Result<(), AgentError>> {
        tracing::info!(
            source = ?report.source,
            run_id = %report.run_id,
            session = report.session_id.as_ref().map(SessionId::as_str),
            text = %report.text,
            "agent report"
        );
        Box::pin(std::future::ready(Ok(())))
    }
}

/// The text a person should see for an outcome, or `None` to stay quiet.
///
/// Final answers are delivered unless [`is_silent`]; paused runs always are
/// (someone has to approve). Budget and loop stops are only logged.
#[must_use]
pub fn report_text(outcome: &AgentOutcome) -> Option<String> {
    match outcome {
        AgentOutcome::Completed { text, .. } if !is_silent(text) => Some(text.clone()),
        AgentOutcome::AwaitingApproval { state } => {
            let lines: Vec<String> = state
                .awaiting()
                .map(|pending| format!("- {} {}", pending.call.name, pending.call.arguments))
                .collect();
            Some(format!(
                "The agent wants to run these calls and needs approval:\n{}",
                lines.join("\n")
            ))
        }
        _ => None,
    }
}

#[cfg(feature = "autumn")]
mod background;

#[cfg(feature = "autumn")]
pub use background::{FollowupTool, Heartbeat, HeartbeatSchedule, Precheck};
#[cfg(feature = "autumn")]
pub(crate) use background::{HeartbeatSettings, deliver_outcome};
