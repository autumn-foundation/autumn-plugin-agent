//! The Autumn half of always-on behaviour: heartbeats on the scheduler,
//! follow-ups as delayed jobs, and delivery from background runs.
//!
//! This module needs `autumn-web`, so the `autumn` feature gates it.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use autumn_web::task::{Schedule, TaskCoordination, TaskInfo};
use autumn_web::{AppState, AutumnResult};
use futures::future::BoxFuture;

use super::{DEFAULT_HEARTBEAT_PROMPT, Delivery, Report, ReportSource, report_text};
use crate::agent::{AgentOutcome, AgentRuntime};
use crate::error::{AgentError, ErrorKind};
use crate::ids::{RunId, SessionId};
use crate::jobs::{AgentRunArgs, RunOrigin, enqueue_agent_run_in, execute_run};
use crate::memory::MemoryScope;
use crate::policy::{Strictest, ToolPolicy, ToolRules};
use crate::tools::{Tool, ToolContext, ToolEffect};

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

    /// Replace the check-in prompt. Keep the [`HEARTBEAT_OK`](super::HEARTBEAT_OK) instruction
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
#[path = "tests.rs"]
mod tests;
