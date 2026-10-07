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

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use autumn_web::task::{Schedule, TaskCoordination, TaskInfo};
use autumn_web::{AppState, AutumnResult};
use futures::future::BoxFuture;
use serde::{Deserialize, Serialize};

use crate::agent::{AgentOutcome, AgentRuntime};
use crate::error::{AgentError, ErrorKind};
use crate::ids::{RunId, SessionId};
use crate::jobs::{AgentRunArgs, RunOrigin, enqueue_agent_run_in, execute_run};
use crate::memory::MemoryScope;
use crate::policy::{Strictest, ToolPolicy, ToolRules};
use crate::tools::{Tool, ToolContext, ToolEffect};

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

/// Deliver an outcome when it has something to say. Delivery failures are
/// logged, not returned: failing the job would re-run the whole agent.
pub(crate) async fn deliver_outcome(
    delivery: &dyn Delivery,
    source: ReportSource,
    run_id: &RunId,
    session_id: Option<&SessionId>,
    outcome: &AgentOutcome,
) {
    let Some(text) = report_text(outcome) else {
        tracing::debug!(run_id = %run_id, ?source, "background run had nothing to report");
        return;
    };
    let report = Report {
        source,
        run_id: run_id.clone(),
        session_id: session_id.cloned(),
        text,
        outcome: outcome.clone(),
    };
    if let Err(err) = delivery.deliver(&report).await {
        tracing::error!(run_id = %run_id, error = %err, "cannot deliver the agent report");
    }
}

/// A cheap app check that can skip a heartbeat tick before any model call.
pub type Precheck = Arc<dyn Fn(AppState) -> BoxFuture<'static, bool> + Send + Sync>;

/// When a heartbeat fires.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum HeartbeatSchedule {
    /// After a fixed delay from the end of the previous tick.
    Every(Duration),
    /// On a 6-field cron expression (`sec min hour day month weekday`).
    Cron {
        /// The expression, for example `"0 */30 8-18 * * Mon-Fri"`.
        expression: String,
        /// IANA time zone, for example `"Europe/Oslo"`. UTC when `None`.
        timezone: Option<String>,
    },
}

/// A periodic check-in for an always-on agent.
///
/// ```rust
/// use std::time::Duration;
/// use autumn_plugin_agent::proactive::Heartbeat;
///
/// // Every 30 minutes during office hours, Oslo time, in the "ops" session.
/// let heartbeat = Heartbeat::cron("0 */30 8-18 * * Mon-Fri")
///     .timezone("Europe/Oslo")
///     .session("ops");
/// # let _ = Heartbeat::every(Duration::from_secs(1_800));
/// ```
#[derive(Clone)]
pub struct Heartbeat {
    schedule: HeartbeatSchedule,
    prompt: String,
    session: Option<SessionId>,
    memory_scope: Option<MemoryScope>,
    read_only: bool,
    coordination: TaskCoordination,
    precheck: Option<Precheck>,
}

impl std::fmt::Debug for Heartbeat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Heartbeat")
            .field("schedule", &self.schedule)
            .field("session", &self.session)
            .field("memory_scope", &self.memory_scope)
            .field("read_only", &self.read_only)
            .field("coordination", &self.coordination)
            .field("precheck", &self.precheck.is_some())
            .finish_non_exhaustive()
    }
}

impl Heartbeat {
    /// The scheduled-task name the heartbeat registers.
    pub const TASK_NAME: &'static str = "agent_heartbeat";

    fn with_schedule(schedule: HeartbeatSchedule) -> Self {
        Self {
            schedule,
            prompt: DEFAULT_HEARTBEAT_PROMPT.to_owned(),
            session: None,
            memory_scope: None,
            read_only: true,
            coordination: TaskCoordination::Fleet,
            precheck: None,
        }
    }

    /// Tick after a fixed delay from the end of the previous tick.
    #[must_use]
    pub fn every(interval: Duration) -> Self {
        Self::with_schedule(HeartbeatSchedule::Every(interval))
    }

    /// Tick on a 6-field cron expression (`sec min hour day month weekday`).
    #[must_use]
    pub fn cron(expression: impl Into<String>) -> Self {
        Self::with_schedule(HeartbeatSchedule::Cron {
            expression: expression.into(),
            timezone: None,
        })
    }

    /// The IANA time zone for a cron schedule. Ignored for fixed delays.
    #[must_use]
    pub fn timezone(mut self, timezone: impl Into<String>) -> Self {
        if let HeartbeatSchedule::Cron { timezone: tz, .. } = &mut self.schedule {
            *tz = Some(timezone.into());
        }
        self
    }

    /// Replace the check-in prompt. Keep the [`HEARTBEAT_OK`] instruction
    /// in it, or every tick will be delivered.
    #[must_use]
    pub fn prompt(mut self, prompt: impl Into<String>) -> Self {
        self.prompt = prompt.into();
        self
    }

    /// Run ticks inside this session, so the agent sees earlier ticks.
    #[must_use]
    pub fn session(mut self, session: impl Into<SessionId>) -> Self {
        self.session = Some(session.into());
        self
    }

    /// Bind memory to this scope instead of [`MemoryScope::agent`].
    #[must_use]
    pub fn memory_scope(mut self, scope: MemoryScope) -> Self {
        self.memory_scope = Some(scope);
        self
    }

    /// Let ticks use every tool the app policy allows, not only read-only
    /// and internal ones.
    #[must_use]
    pub const fn allow_actions(mut self) -> Self {
        self.read_only = false;
        self
    }

    /// Run on every replica instead of once per tick across the fleet.
    #[must_use]
    pub const fn per_replica(mut self) -> Self {
        self.coordination = TaskCoordination::PerReplica;
        self
    }

    /// Skip a tick, before any model call, when `check` answers `false`.
    #[must_use]
    pub fn precheck<F, Fut>(mut self, check: F) -> Self
    where
        F: Fn(AppState) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = bool> + Send + 'static,
    {
        self.precheck = Some(Arc::new(move |state| Box::pin(check(state))));
        self
    }

    /// The configured schedule.
    #[must_use]
    pub const fn schedule(&self) -> &HeartbeatSchedule {
        &self.schedule
    }

    /// Check the schedule and prompt.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::Config`] for an interval under one second, a
    /// blank cron expression, or a blank prompt.
    pub fn validate(&self) -> Result<(), AgentError> {
        match &self.schedule {
            HeartbeatSchedule::Every(interval) if *interval < Duration::from_secs(1) => {
                return Err(AgentError::new(
                    ErrorKind::Config,
                    "heartbeat interval must be at least one second",
                ));
            }
            HeartbeatSchedule::Cron { expression, .. } if expression.trim().is_empty() => {
                return Err(AgentError::new(
                    ErrorKind::Config,
                    "heartbeat cron expression must not be blank",
                ));
            }
            _ => {}
        }
        if self.prompt.trim().is_empty() {
            return Err(AgentError::new(
                ErrorKind::Config,
                "heartbeat prompt must not be blank",
            ));
        }
        Ok(())
    }

    /// The Autumn scheduled task for this heartbeat.
    pub(crate) fn task_info(&self) -> TaskInfo {
        let schedule = match &self.schedule {
            HeartbeatSchedule::Every(interval) => Schedule::FixedDelay(*interval),
            HeartbeatSchedule::Cron {
                expression,
                timezone,
            } => Schedule::Cron {
                expression: expression.clone(),
                timezone: timezone.clone(),
            },
        };
        TaskInfo {
            name: Self::TASK_NAME.to_owned(),
            schedule,
            coordination: self.coordination,
            handler: heartbeat_tick,
        }
    }

    /// Run one tick. Returns `None` when the precheck skipped it.
    ///
    /// # Errors
    ///
    /// Returns [`AgentError`] when the run fails (see
    /// [`Agent::run`](crate::agent::Agent::run)).
    pub async fn tick(
        &self,
        runtime: &AgentRuntime,
        state: &AppState,
    ) -> Result<Option<AgentOutcome>, AgentError> {
        if let Some(check) = &self.precheck
            && !check(state.clone()).await
        {
            tracing::debug!("heartbeat precheck found nothing to do; skipping the tick");
            return Ok(None);
        }
        let args = AgentRunArgs::new(self.prompt.clone())
            .session(self.session.clone())
            .memory_scope(self.memory_scope.clone())
            .deliver(true)
            .origin(RunOrigin::Heartbeat);
        let policy: Option<Arc<dyn ToolPolicy>> = self.read_only.then(|| {
            Arc::new(Strictest::new(vec![
                runtime.policy(),
                Arc::new(ToolRules::read_only()),
            ])) as Arc<dyn ToolPolicy>
        });
        execute_run(runtime, args, policy).await.map(Some)
    }
}

/// The heartbeat as installed on app state by the plugin's startup hook.
#[derive(Debug, Clone)]
pub(crate) struct HeartbeatSettings(pub(crate) Heartbeat);

/// The scheduled-task handler behind [`Heartbeat`].
fn heartbeat_tick(state: AppState) -> Pin<Box<dyn Future<Output = AutumnResult<()>> + Send>> {
    Box::pin(async move {
        let (Some(runtime), Some(settings)) = (
            state.extension::<AgentRuntime>(),
            state.extension::<HeartbeatSettings>(),
        ) else {
            tracing::warn!("agent heartbeat fired before the agent plugin started");
            return Ok(());
        };
        settings
            .0
            .tick(&runtime, &state)
            .await
            .map(|_| ())
            .map_err(|err| {
                tracing::error!(error = %err, "agent heartbeat failed");
                err.into_autumn_error()
            })
    })
}

/// The `schedule_followup` tool: the agent books its own next wake-up.
///
/// The follow-up runs as a delayed `agent_run` job in the same session, with
/// delivery on. Turn it on with
/// [`AgentPlugin::followups`](crate::plugin::AgentPlugin::followups).
#[derive(Debug, Clone)]
pub struct FollowupTool {
    max_delay: Duration,
    memory_scope: Option<MemoryScope>,
    depth: u32,
}

impl FollowupTool {
    /// Longest prompt a follow-up may carry, in characters.
    pub const MAX_PROMPT_CHARS: usize = 2_000;

    /// Longest chain of follow-ups (a follow-up that schedules a follow-up
    /// that ...). Stops an agent from waking itself forever.
    pub const MAX_CHAIN: u32 = 10;

    /// Allow follow-ups up to `max_delay` from now.
    #[must_use]
    pub const fn new(max_delay: Duration) -> Self {
        Self {
            max_delay,
            memory_scope: None,
            depth: 0,
        }
    }

    /// The run this tool serves is `depth` follow-ups deep.
    #[must_use]
    pub const fn depth(mut self, depth: u32) -> Self {
        self.depth = depth;
        self
    }

    /// Run follow-ups with memory bound to this scope.
    #[must_use]
    pub fn memory_scope(mut self, scope: MemoryScope) -> Self {
        self.memory_scope = Some(scope);
        self
    }

    /// Check the input and build the delayed job's arguments.
    fn plan(
        &self,
        input: &serde_json::Value,
        ctx: &ToolContext,
    ) -> Result<(AgentRunArgs, Duration), AgentError> {
        let depth = self.depth.saturating_add(1);
        if depth > Self::MAX_CHAIN {
            return Err(AgentError::new(
                ErrorKind::Tool,
                format!(
                    "this run is already {} follow-ups deep; finish now and report instead of scheduling another",
                    self.depth
                ),
            ));
        }
        let max_minutes = self.max_delay.as_secs() / 60;
        let minutes = input
            .get("delay_minutes")
            .and_then(serde_json::Value::as_u64)
            .filter(|minutes| (1..=max_minutes).contains(minutes))
            .ok_or_else(|| {
                AgentError::new(
                    ErrorKind::Tool,
                    format!("delay_minutes must be a whole number from 1 to {max_minutes}"),
                )
            })?;
        let prompt = input
            .get("prompt")
            .and_then(serde_json::Value::as_str)
            .map(str::trim)
            .filter(|prompt| !prompt.is_empty())
            .ok_or_else(|| {
                AgentError::new(ErrorKind::Tool, "give a `prompt` for the follow-up run")
            })?;
        if prompt.chars().count() > Self::MAX_PROMPT_CHARS {
            return Err(AgentError::new(
                ErrorKind::Tool,
                format!(
                    "the follow-up prompt must be at most {} characters",
                    Self::MAX_PROMPT_CHARS
                ),
            ));
        }
        let mut args = AgentRunArgs::new(prompt)
            .session(ctx.session_id.clone())
            .memory_scope(self.memory_scope.clone())
            .deliver(true)
            .origin(RunOrigin::Followup);
        args.followup_depth = depth;
        Ok((args, Duration::from_secs(minutes.saturating_mul(60))))
    }
}

impl Tool for FollowupTool {
    fn name(&self) -> &'static str {
        "schedule_followup"
    }

    fn description(&self) -> &'static str {
        "Wake yourself up later: after delay_minutes, a new run starts with `prompt` in this conversation, and its answer is sent to the user. Use it to check back on something that is not ready yet."
    }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "delay_minutes": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": self.max_delay.as_secs() / 60,
                },
                "prompt": {
                    "type": "string",
                    "description": "What your future self should do, with every detail it needs.",
                }
            },
            "required": ["delay_minutes", "prompt"]
        })
    }

    fn effect(&self) -> ToolEffect {
        ToolEffect::Internal
    }

    fn execute<'a>(
        &'a self,
        input: serde_json::Value,
        ctx: &'a ToolContext,
    ) -> Pin<Box<dyn Future<Output = Result<serde_json::Value, AgentError>> + Send + 'a>> {
        Box::pin(async move {
            let (args, delay) = self.plan(&input, ctx)?;
            let run_id = args.run_id.clone();
            enqueue_agent_run_in(args, delay).await.map_err(|err| {
                AgentError::new(
                    ErrorKind::Tool,
                    format!("cannot schedule the follow-up: {err}"),
                )
            })?;
            Ok(serde_json::json!({
                "scheduled": true,
                "in_minutes": delay.as_secs() / 60,
                "run_id": run_id,
            }))
        })
    }
}

#[cfg(test)]
mod tests;
