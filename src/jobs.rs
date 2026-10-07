//! Background runs on Autumn's job queue.
//!
//! Two jobs carry agent work off the request path:
//!
//! * `agent_run` — run the loop for one prompt, optionally inside a session.
//!   Enqueue with [`enqueue_agent_run`], or with [`enqueue_agent_run_tracked`]
//!   to get a token the caller can poll at Autumn's job-status route. A
//!   tracked run reports progress per loop step and stores the
//!   [`AgentOutcome`] as its JSON result.
//! * `agent_resume` — continue a run that paused for approval, with a
//!   reviewer's decisions. It never retries: the approved calls may have
//!   side effects.
//!
//! With `deliver` set, a finished run hands its answer to the plugin's
//! [`Delivery`](crate::proactive::Delivery) (heartbeats and follow-ups set
//! it).

use std::sync::Arc;
use std::time::Duration;

use autumn_web::job::{JobContext, TrackedJobHandle};
use autumn_web::{AppState, AutumnError, AutumnResult};
use futures::future::BoxFuture;
use serde::{Deserialize, Serialize};

use crate::agent::{AgentOutcome, AgentRuntime, ApprovalDecision, RunState};
use crate::error::AgentError;
use crate::hooks::{AgentHooks, RunInfo};
use crate::ids::{RunId, SessionId};
use crate::memory::MemoryScope;
use crate::policy::ToolPolicy;
use crate::proactive::{ReportSource, deliver_outcome};

/// Why a background run started.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RunOrigin {
    /// The app enqueued it (a handler, a listener, a task).
    #[default]
    Request,
    /// The agent scheduled it with the `schedule_followup` tool.
    Followup,
    /// The heartbeat started it.
    Heartbeat,
}

impl From<RunOrigin> for ReportSource {
    fn from(origin: RunOrigin) -> Self {
        match origin {
            RunOrigin::Request => Self::Job,
            RunOrigin::Followup => Self::Followup,
            RunOrigin::Heartbeat => Self::Heartbeat,
        }
    }
}

/// Arguments for the `agent_run` background job.
///
/// Build with [`AgentRunArgs::new`] and the builder methods; the struct is
/// `#[non_exhaustive]` so new options do not break callers.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct AgentRunArgs {
    /// The user message that starts the run.
    pub prompt: String,
    /// Overrides the configured system prompt for this run.
    #[serde(default)]
    pub system_prompt: Option<String>,
    /// Overrides the configured step budget for this run.
    #[serde(default)]
    pub max_steps: Option<u32>,
    /// Run inside this session (load, run, save).
    #[serde(default)]
    pub session: Option<SessionId>,
    /// Bind memory to this scope instead of [`MemoryScope::agent`].
    #[serde(default)]
    pub memory_scope: Option<MemoryScope>,
    /// Hand the answer to the plugin's delivery when the run ends.
    #[serde(default)]
    pub deliver: bool,
    /// Why the run started.
    #[serde(default)]
    pub origin: RunOrigin,
    /// The run id. Fixed at enqueue time, so a retried job keeps it and
    /// tools can use it as an idempotency key.
    #[serde(default = "RunId::generate")]
    pub run_id: RunId,
    /// How many follow-ups led to this run (0 for a run the app started).
    /// The `schedule_followup` tool refuses to extend a chain past
    /// [`FollowupTool::MAX_CHAIN`](crate::proactive::FollowupTool::MAX_CHAIN).
    #[serde(default)]
    pub followup_depth: u32,
}

impl AgentRunArgs {
    /// A run for one prompt, with every option at its default.
    #[must_use]
    pub fn new(prompt: impl Into<String>) -> Self {
        Self {
            prompt: prompt.into(),
            system_prompt: None,
            max_steps: None,
            session: None,
            memory_scope: None,
            deliver: false,
            origin: RunOrigin::Request,
            run_id: RunId::generate(),
            followup_depth: 0,
        }
    }

    /// Override the system prompt.
    #[must_use]
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Override the step budget.
    #[must_use]
    pub const fn max_steps(mut self, max_steps: u32) -> Self {
        self.max_steps = Some(max_steps);
        self
    }

    /// Run inside a session.
    #[must_use]
    pub fn session(mut self, session: impl Into<Option<SessionId>>) -> Self {
        self.session = session.into();
        self
    }

    /// Bind memory to a scope.
    #[must_use]
    pub fn memory_scope(mut self, scope: impl Into<Option<MemoryScope>>) -> Self {
        self.memory_scope = scope.into();
        self
    }

    /// Deliver the answer when the run ends.
    #[must_use]
    pub const fn deliver(mut self, deliver: bool) -> Self {
        self.deliver = deliver;
        self
    }

    /// Record why the run started.
    #[must_use]
    pub const fn origin(mut self, origin: RunOrigin) -> Self {
        self.origin = origin;
        self
    }
}

/// Arguments for the `agent_resume` background job.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[non_exhaustive]
pub struct AgentResumeArgs {
    /// The paused run.
    pub state: RunState,
    /// The reviewer's decisions.
    pub decisions: Vec<ApprovalDecision>,
    /// Bind memory to this scope instead of [`MemoryScope::agent`].
    #[serde(default)]
    pub memory_scope: Option<MemoryScope>,
    /// Hand the answer to the plugin's delivery when the run ends.
    #[serde(default)]
    pub deliver: bool,
}

impl AgentResumeArgs {
    /// Resume `state` with `decisions`.
    #[must_use]
    pub const fn new(state: RunState, decisions: Vec<ApprovalDecision>) -> Self {
        Self {
            state,
            decisions,
            memory_scope: None,
            deliver: false,
        }
    }

    /// Bind memory to a scope.
    #[must_use]
    pub fn memory_scope(mut self, scope: impl Into<Option<MemoryScope>>) -> Self {
        self.memory_scope = scope.into();
        self
    }

    /// Deliver the answer when the run ends.
    #[must_use]
    pub const fn deliver(mut self, deliver: bool) -> Self {
        self.deliver = deliver;
        self
    }
}

/// Reports loop progress to Autumn's tracked-job store. A no-op for
/// untracked jobs and outside jobs.
#[derive(Debug)]
struct JobProgress;

impl AgentHooks for JobProgress {
    fn before_model<'a>(
        &'a self,
        _request: &'a mut crate::client::ChatRequest,
        info: &'a RunInfo,
    ) -> BoxFuture<'a, ()> {
        Box::pin(async move {
            let context = JobContext::current();
            if !context.is_tracked() {
                return;
            }
            let max = info.max_steps.max(1);
            let pct = (info.steps_used.min(max).saturating_mul(99) / max).min(99);
            let pct = u8::try_from(pct).unwrap_or(99);
            let message = format!("step {} of {}", info.steps_used, info.max_steps);
            if let Err(err) = context.set_progress(pct, Some(&message)).await {
                tracing::debug!(error = %err, "cannot record agent job progress");
            }
        })
    }
}

/// Store the outcome as the tracked job's result and log it.
fn record_outcome(run_id: &RunId, outcome: &AgentOutcome) {
    match serde_json::to_value(outcome) {
        Ok(value) => JobContext::current().set_result(value),
        Err(err) => tracing::warn!(error = %err, "cannot serialize the agent outcome"),
    }
    let usage = outcome.usage();
    tracing::info!(
        run_id = %run_id,
        status = outcome_status(outcome),
        steps_used = outcome.steps_used(),
        input_tokens = usage.input_tokens,
        output_tokens = usage.output_tokens,
        cache_read_tokens = usage.cache_read_tokens,
        "background agent run finished"
    );
}

const fn outcome_status(outcome: &AgentOutcome) -> &'static str {
    match outcome {
        AgentOutcome::Completed { .. } => "completed",
        AgentOutcome::BudgetExhausted { .. } => "budget_exhausted",
        AgentOutcome::LoopDetected { .. } => "loop_detected",
        AgentOutcome::AwaitingApproval { .. } => "awaiting_approval",
    }
}

/// Run one `agent_run` payload against a runtime.
///
/// Shared by the job, the heartbeat, and tests. `policy` overrides the
/// runtime policy (the heartbeat passes a read-only one).
pub(crate) async fn execute_run(
    runtime: &AgentRuntime,
    args: AgentRunArgs,
    policy: Option<Arc<dyn ToolPolicy>>,
) -> Result<AgentOutcome, AgentError> {
    let scope = args.memory_scope.clone().unwrap_or_else(MemoryScope::agent);
    let mut agent = runtime
        .agent_with(scope, args.followup_depth)
        .run_id(args.run_id.clone())
        .hook(Arc::new(JobProgress));
    if let Some(system) = &args.system_prompt {
        agent = agent.system_prompt(system.clone());
    }
    if let Some(max_steps) = args.max_steps {
        agent = agent.max_steps(max_steps);
    }
    if let Some(policy) = policy {
        agent = agent.policy(policy);
    }
    let turn = match &args.session {
        Some(session) => {
            agent
                .run_in_session(runtime.sessions().as_ref(), session, &args.prompt)
                .await?
        }
        None => agent.run_turn(Vec::new(), &args.prompt).await?,
    };
    record_outcome(&args.run_id, &turn.outcome);
    if args.deliver {
        deliver_outcome(
            runtime.delivery().as_ref(),
            args.origin.into(),
            &args.run_id,
            args.session.as_ref(),
            &turn.outcome,
        )
        .await;
    }
    Ok(turn.outcome)
}

/// Run one `agent_resume` payload against a runtime.
pub(crate) async fn execute_resume(
    runtime: &AgentRuntime,
    args: AgentResumeArgs,
) -> Result<AgentOutcome, AgentError> {
    let scope = args.memory_scope.clone().unwrap_or_else(MemoryScope::agent);
    let run_id = args.state.run_id.clone();
    let session = args.state.session_id.clone();
    let agent = runtime
        .agent_for(scope)
        .run_id(run_id.clone())
        .hook(Arc::new(JobProgress));
    let turn = agent
        .resume_in_session(runtime.sessions().as_ref(), args.state, args.decisions)
        .await?;
    record_outcome(&run_id, &turn.outcome);
    if args.deliver {
        deliver_outcome(
            runtime.delivery().as_ref(),
            ReportSource::Resume,
            &run_id,
            session.as_ref(),
            &turn.outcome,
        )
        .await;
    }
    Ok(turn.outcome)
}

fn runtime_from(state: &AppState) -> AutumnResult<Arc<AgentRuntime>> {
    state
        .extension::<AgentRuntime>()
        .ok_or_else(|| AutumnError::service_unavailable_msg("agent plugin is not installed"))
}

/// The `agent_run` and `agent_resume` background jobs.
///
/// Lives in a `pub(crate)` module: the `#[job]` macro emits undocumented
/// `*Job` handles, and keeping the module crate-internal keeps them out of
/// the public API (and out of `missing_docs`' reach). Callers use the
/// `enqueue_*` functions.
#[doc(hidden)]
#[allow(missing_docs)]
pub(crate) mod agent_jobs {
    use super::{AgentResumeArgs, AgentRunArgs, execute_resume, execute_run, runtime_from};
    use autumn_web::{AppState, AutumnResult};

    /// Run the agent loop as a background job. Provider failures fail the
    /// job, so the queue's retry/backoff policy applies; the run id stays
    /// the same across attempts.
    #[autumn_web::job(name = "agent_run", max_attempts = 3, backoff_ms = 1_000)]
    pub async fn run_agent(state: AppState, args: AgentRunArgs) -> AutumnResult<()> {
        let runtime = runtime_from(&state)?;
        execute_run(&runtime, args, None)
            .await
            .map(|_| ())
            .map_err(|err| {
                tracing::error!(error = %err, "background agent run failed");
                err.into_autumn_error()
            })
    }

    /// Resume a paused run. One attempt only: approved calls may have side
    /// effects that must not repeat.
    #[autumn_web::job(name = "agent_resume", max_attempts = 1)]
    pub async fn resume_agent(state: AppState, args: AgentResumeArgs) -> AutumnResult<()> {
        let runtime = runtime_from(&state)?;
        execute_resume(&runtime, args)
            .await
            .map(|_| ())
            .map_err(|err| {
                tracing::error!(error = %err, "background agent resume failed");
                err.into_autumn_error()
            })
    }
}

/// Enqueue an `agent_run` background job.
///
/// # Errors
///
/// Returns [`AutumnError`] when the arguments fail to serialize or the job
/// cannot be enqueued (for example, the job runtime is not running).
pub async fn enqueue_agent_run(args: AgentRunArgs) -> AutumnResult<()> {
    agent_jobs::RunAgentJob::enqueue(args).await
}

/// Enqueue an `agent_run` job that runs once after `delay`.
///
/// # Errors
///
/// See [`enqueue_agent_run`].
pub async fn enqueue_agent_run_in(args: AgentRunArgs, delay: Duration) -> AutumnResult<()> {
    agent_jobs::RunAgentJob::enqueue_in(args, delay).await
}

/// Enqueue an `agent_run` job and get a pollable handle.
///
/// The handle's token is a capability: anyone holding it may poll
/// [`TrackedJobHandle::status_path`]. The tracked result is the serialized
/// [`AgentOutcome`]; a paused run's result carries the [`RunState`] to
/// resume.
///
/// # Errors
///
/// See [`enqueue_agent_run`].
pub async fn enqueue_agent_run_tracked(args: AgentRunArgs) -> AutumnResult<TrackedJobHandle> {
    agent_jobs::RunAgentJob::enqueue_tracked(args).await
}

/// Enqueue an `agent_resume` job.
///
/// # Errors
///
/// See [`enqueue_agent_run`].
pub async fn enqueue_agent_resume(args: AgentResumeArgs) -> AutumnResult<()> {
    agent_jobs::ResumeAgentJob::enqueue(args).await
}

/// Enqueue an `agent_resume` job and get a pollable handle.
///
/// # Errors
///
/// See [`enqueue_agent_run`].
pub async fn enqueue_agent_resume_tracked(args: AgentResumeArgs) -> AutumnResult<TrackedJobHandle> {
    agent_jobs::ResumeAgentJob::enqueue_tracked(args).await
}

#[cfg(test)]
mod tests;
