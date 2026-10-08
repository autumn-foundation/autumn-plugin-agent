//! The agent loop: prompt the model, run its tool calls, repeat.
//!
//! [`Agent`] drives the loop. It stops on a final answer, when a budget runs
//! out, when the [`LoopGuard`] sees the model repeat itself, or when the
//! [`ToolPolicy`] wants a person to approve a call. Each of those is a normal
//! [`AgentOutcome`], never an error, so callers always get a structured
//! answer back.
//!
//! A run can be one-shot ([`Agent::run`]), continue a transcript
//! ([`Agent::run_turn`]), or live in a persisted session
//! ([`Agent::run_in_session`]). A run paused for approval resumes with
//! [`Agent::resume`].
//!
//! [`enqueue_agent_run`] exposes the same loop as an Autumn background job.

use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};
use tokio::time::Instant;

use crate::client::{
    ChatMessage, ChatRequest, ChatResponse, ChatRole, ContentPart, LlmClient, TokenUsage,
};
use crate::config::AgentConfig;
use crate::error::{AgentError, ErrorKind};
use crate::hooks::{AgentHooks, HookAction, RunInfo, ToolOutput};
use crate::ids::{RunId, SessionId};
use crate::loop_guard::{LoopGuard, LoopTracker, LoopVerdict, warning_note};
use crate::memory::{MemoryScope, MemoryStore, MemoryTool, render_snapshot};
use crate::policy::{AllowAll, ToolDecision, ToolPolicy};
#[cfg(feature = "autumn")]
use crate::proactive::FollowupTool;
use crate::proactive::{Delivery, LogDelivery};
use crate::session::{Compaction, CompactionReport, SessionStore, compact};
use crate::skills::{Skill, SkillTool, render_index};
use crate::tools::{Tool, ToolCall, ToolContext};

/// Rough tokens-per-byte heuristic for budget accounting.
///
/// Providers report exact usage after each call; between calls the loop
/// estimates with `bytes / 4`. It errs on the cheap side for code and the
/// expensive side for CJK text — good enough for a guardrail, never for
/// billing.
#[must_use]
pub fn estimate_tokens(text: &str) -> u32 {
    // `len() / 4` on bytes: ASCII text dominates prompts, and byte length
    // only over-counts multi-byte text, which errs toward safety.
    // Saturate instead of truncating on 16-bit targets.
    u32::try_from(text.len() / 4).unwrap_or(u32::MAX)
}

/// Estimate the tokens in a whole message history.
#[must_use]
pub fn estimate_messages(messages: &[ChatMessage]) -> u32 {
    messages
        .iter()
        .map(|message| {
            message
                .content
                .iter()
                .map(|part| match part {
                    ContentPart::Text(text) => estimate_tokens(text),
                    ContentPart::ToolCall {
                        name, arguments, ..
                    } => estimate_tokens(name)
                        .saturating_add(estimate_tokens(&arguments.to_string())),
                    ContentPart::ToolResult { content, .. } => estimate_tokens(content),
                })
                .fold(0u32, u32::saturating_add)
        })
        .fold(0u32, u32::saturating_add)
        .saturating_add(
            u32::try_from(messages.len())
                .unwrap_or(u32::MAX)
                .saturating_mul(4),
        ) // per-message framing overhead
}

/// Shrink a history to fit a token budget.
///
/// Keeps every system message and, when it fits, the first non-system
/// message if it is a user message: the task the run serves. Then keeps the
/// newest messages that fit. Order is preserved.
///
/// The kept tail never starts with a tool-result message: a result whose
/// call was cut would make providers reject the request, so such leading
/// results are dropped too.
///
/// When even the system prompt alone exceeds the budget, the system messages
/// come back anyway and the caller treats that as budget exhaustion.
#[must_use]
pub fn truncate_history(messages: &[ChatMessage], budget: u32) -> Vec<ChatMessage> {
    let cost = |message: &ChatMessage| estimate_messages(std::slice::from_ref(message));
    let mut kept: Vec<usize> = Vec::new();
    let mut used: u32 = 0;
    for (index, message) in messages.iter().enumerate() {
        if message.role == ChatRole::System {
            used = used.saturating_add(cost(message));
            kept.push(index);
        }
    }
    // The task anchor: the first non-system message, when it is the user's.
    let anchor = messages
        .iter()
        .position(|message| message.role != ChatRole::System)
        .filter(|&index| {
            messages
                .get(index)
                .is_some_and(|message| message.role == ChatRole::User)
        })
        .filter(|&index| {
            messages
                .get(index)
                .is_some_and(|message| used.saturating_add(cost(message)) <= budget)
        });
    if let Some(index) = anchor
        && let Some(message) = messages.get(index)
    {
        used = used.saturating_add(cost(message));
        kept.push(index);
    }
    let mut tail: Vec<usize> = Vec::new();
    for (index, message) in messages.iter().enumerate().rev() {
        if message.role == ChatRole::System || Some(index) == anchor {
            continue;
        }
        let message_cost = cost(message);
        if used.saturating_add(message_cost) <= budget {
            used = used.saturating_add(message_cost);
            tail.push(index);
        } else {
            break;
        }
    }
    tail.reverse();
    // A tool result at the front of the tail lost its call (the assistant
    // message before it was cut): drop it.
    let orphans = tail
        .iter()
        .take_while(|&&index| {
            messages
                .get(index)
                .is_some_and(|message| message.role == ChatRole::Tool)
        })
        .count();
    tail.drain(..orphans);
    kept.extend(tail);
    kept.sort_unstable();
    kept.dedup();
    kept.into_iter()
        .filter_map(|index| messages.get(index).cloned())
        .collect()
}

/// Which budget stopped the run.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum BudgetKind {
    /// `max_steps` iterations ran out.
    Steps,
    /// `max_tokens` ran out.
    Tokens,
    /// The wall-clock limit (`max_duration`) passed.
    Deadline,
}

/// How a paused run treats one tool call when it resumes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
pub enum PendingStatus {
    /// The policy allowed the call; it runs on resume.
    Allowed,
    /// The call waits for a person.
    NeedsApproval {
        /// Why the policy asked, for the reviewer.
        reason: String,
    },
    /// The policy refused the call; the model sees the reason on resume.
    Denied {
        /// Why the call was refused.
        reason: String,
    },
}

/// One tool call in a paused run.
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PendingCall {
    /// The call the model made.
    pub call: ToolCall,
    /// What happens to it on resume.
    pub status: PendingStatus,
}

/// A reviewer's answer for one call that needed approval.
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "verdict", rename_all = "snake_case")]
pub enum Approval {
    /// Run the call as the model asked.
    Approve,
    /// Run the call with these arguments instead.
    Edit {
        /// Replacement arguments.
        arguments: serde_json::Value,
    },
    /// Do not run the call. The model sees the reason.
    Reject {
        /// Why, for the model.
        reason: String,
    },
}

/// A reviewer's answer, addressed to one call id.
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ApprovalDecision {
    /// The `id` of the [`ToolCall`] this answers.
    pub call_id: String,
    /// The answer.
    pub approval: Approval,
}

impl ApprovalDecision {
    /// Approve one call.
    #[must_use]
    pub fn approve(call_id: impl Into<String>) -> Self {
        Self {
            call_id: call_id.into(),
            approval: Approval::Approve,
        }
    }

    /// Reject one call with a reason the model will see.
    #[must_use]
    pub fn reject(call_id: impl Into<String>, reason: impl Into<String>) -> Self {
        Self {
            call_id: call_id.into(),
            approval: Approval::Reject {
                reason: reason.into(),
            },
        }
    }
}

/// Everything needed to continue a paused run, in serializable form.
///
/// Store it as JSON (a database row, a tracked job result), show
/// [`RunState::awaiting`] to a person, then pass it to [`Agent::resume`]
/// with their decisions. The system prompt is not part of the state: the
/// resuming agent rebuilds it.
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct RunState {
    /// The paused run's id. The resumed run keeps it.
    pub run_id: RunId,
    /// The session the run belongs to, if any.
    pub session_id: Option<SessionId>,
    /// The transcript so far, ending with the assistant's tool calls.
    pub messages: Vec<ChatMessage>,
    /// Provider-reported tokens spent so far.
    pub usage: TokenUsage,
    /// Tool-execution rounds used so far, the paused round included.
    pub steps_used: u32,
    /// The calls of the paused round.
    pub pending: Vec<PendingCall>,
    /// Loop-guard fingerprints, so a resumed run keeps its loop history.
    #[serde(default)]
    pub recent_calls: Vec<u64>,
}

impl RunState {
    /// The calls that wait for a person.
    pub fn awaiting(&self) -> impl Iterator<Item = &PendingCall> {
        self.pending
            .iter()
            .filter(|pending| matches!(pending.status, PendingStatus::NeedsApproval { .. }))
    }
}

/// The result of one agent run.
///
/// Marked `#[non_exhaustive]`: match with a wildcard arm so new stop
/// reasons do not break your build.
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "status", rename_all = "snake_case")]
#[non_exhaustive]
pub enum AgentOutcome {
    /// The model produced a final answer.
    Completed {
        /// The model's final text.
        text: String,
        /// Tool-execution rounds used.
        steps_used: u32,
        /// Provider-reported token totals.
        usage: TokenUsage,
    },
    /// A budget ran out before the model finished.
    BudgetExhausted {
        /// Which budget ran out.
        reason: BudgetKind,
        /// Tool-execution rounds used.
        steps_used: u32,
        /// Provider-reported token totals.
        usage: TokenUsage,
    },
    /// The model repeated the same tool call with the same result too often.
    LoopDetected {
        /// The repeated tool.
        tool: String,
        /// How often the call repeated inside the guard window.
        repeats: u32,
        /// Tool-execution rounds used.
        steps_used: u32,
        /// Provider-reported token totals.
        usage: TokenUsage,
    },
    /// The policy wants a person to approve one or more calls.
    AwaitingApproval {
        /// The paused run. Pass it to [`Agent::resume`].
        state: Box<RunState>,
    },
}

impl AgentOutcome {
    /// Tool-execution rounds consumed, whatever the outcome.
    #[must_use]
    pub fn steps_used(&self) -> u32 {
        match self {
            Self::Completed { steps_used, .. }
            | Self::BudgetExhausted { steps_used, .. }
            | Self::LoopDetected { steps_used, .. } => *steps_used,
            Self::AwaitingApproval { state } => state.steps_used,
        }
    }

    /// Provider-reported token totals, whatever the outcome.
    #[must_use]
    pub fn usage(&self) -> TokenUsage {
        match self {
            Self::Completed { usage, .. }
            | Self::BudgetExhausted { usage, .. }
            | Self::LoopDetected { usage, .. } => *usage,
            Self::AwaitingApproval { state } => state.usage,
        }
    }

    /// `true` when the model produced a final answer.
    #[must_use]
    pub const fn is_completed(&self) -> bool {
        matches!(self, Self::Completed { .. })
    }

    /// The final answer, when the model produced one.
    #[must_use]
    pub fn text(&self) -> Option<&str> {
        match self {
            Self::Completed { text, .. } => Some(text),
            _ => None,
        }
    }
}

/// One finished (or paused) run plus its transcript.
#[derive(Debug, Clone, PartialEq)]
pub struct AgentTurn {
    /// How the run ended.
    pub outcome: AgentOutcome,
    /// The transcript after the run, without the system prompt. Feed it to
    /// the next [`Agent::run_turn`] to continue the conversation.
    pub messages: Vec<ChatMessage>,
    /// Set when the session transcript was compacted before the run.
    pub compaction: Option<CompactionReport>,
}

/// How the loop treats one call of a round.
#[derive(Debug, Clone)]
enum Gate {
    Allowed,
    Approved(Option<serde_json::Value>),
    Denied(String),
    Rejected(String),
}

/// Per-run mutable state.
#[derive(Debug)]
struct LoopState {
    run_id: RunId,
    session_id: Option<SessionId>,
    messages: Vec<ChatMessage>,
    usage: TokenUsage,
    steps: u32,
    tracker: LoopTracker,
}

/// Per-run fixed inputs: the rendered system prompt and the tool list.
#[derive(Debug)]
struct Prepared {
    system: Option<String>,
    tools: Vec<Arc<dyn Tool>>,
}

/// The agent loop.
///
/// ```rust,no_run
/// # use autumn_plugin_agent::{Agent, AgentOutcome};
/// # async fn demo(agent: Agent) -> Result<(), autumn_plugin_agent::AgentError> {
/// match agent.run("What is the weather in Chicago?").await? {
///     AgentOutcome::Completed { text, .. } => println!("answer: {text}"),
///     other => println!("stopped: {other:?}"),
/// }
/// # Ok(())
/// # }
/// ```
#[derive(Clone)]
pub struct Agent {
    client: Arc<dyn LlmClient>,
    tools: Vec<Arc<dyn Tool>>,
    system_prompt: Option<String>,
    max_steps: u32,
    max_tokens: u32,
    per_tool_output_limit: usize,
    temperature: Option<f32>,
    hooks: Vec<Arc<dyn AgentHooks>>,
    policy: Arc<dyn ToolPolicy>,
    loop_guard: LoopGuard,
    max_duration: Option<Duration>,
    parallel_tool_calls: bool,
    memory: Option<(Arc<dyn MemoryStore>, MemoryScope)>,
    skills: Arc<[Skill]>,
    compaction: Option<Compaction>,
    run_id: Option<RunId>,
}

impl std::fmt::Debug for Agent {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("Agent")
            .field("client", &self.client)
            .field(
                "tools",
                &self
                    .tools
                    .iter()
                    .map(|tool| tool.name().to_owned())
                    .collect::<Vec<_>>(),
            )
            .field("max_steps", &self.max_steps)
            .field("max_tokens", &self.max_tokens)
            .field("policy", &self.policy)
            .field("loop_guard", &self.loop_guard)
            .field("max_duration", &self.max_duration)
            .field("memory", &self.memory.as_ref().map(|(_, scope)| scope))
            .field("skills", &self.skills.len())
            .finish_non_exhaustive()
    }
}

impl Agent {
    /// Minimum remaining budget (tokens) that still justifies another model call.
    const MIN_USEFUL_REMAINING: u32 = 512;
    /// Per-call output cap ceiling; the loop never asks for more than this.
    const MAX_CALL_TOKENS: u32 = 4096;

    /// Start building an agent around a provider client.
    #[must_use]
    pub fn new(client: Arc<dyn LlmClient>) -> Self {
        Self {
            client,
            tools: Vec::new(),
            system_prompt: None,
            max_steps: 10,
            max_tokens: 32_000,
            per_tool_output_limit: 8_000,
            temperature: None,
            hooks: Vec::new(),
            policy: Arc::new(AllowAll),
            loop_guard: LoopGuard::default(),
            max_duration: None,
            parallel_tool_calls: true,
            memory: None,
            skills: Arc::from(Vec::new()),
            compaction: None,
            run_id: None,
        }
    }

    /// Tools the model may call. Replaces any earlier list.
    #[must_use]
    pub fn tools(mut self, tools: Vec<Arc<dyn Tool>>) -> Self {
        self.tools = tools;
        self
    }

    /// Add one tool to the list.
    #[must_use]
    pub fn tool(mut self, tool: Arc<dyn Tool>) -> Self {
        self.tools.push(tool);
        self
    }

    /// System prompt prepended to every run.
    #[must_use]
    pub fn system_prompt(mut self, prompt: impl Into<String>) -> Self {
        self.system_prompt = Some(prompt.into());
        self
    }

    /// Maximum tool-execution rounds before the run stops with
    /// [`BudgetKind::Steps`].
    ///
    /// A step is one round in which the model requested tool calls; the final
    /// answering call does not consume a step.
    #[must_use]
    pub const fn max_steps(mut self, max_steps: u32) -> Self {
        self.max_steps = max_steps;
        self
    }

    /// Token budget for the whole run (estimated + provider-reported).
    #[must_use]
    pub const fn max_tokens(mut self, max_tokens: u32) -> Self {
        self.max_tokens = max_tokens;
        self
    }

    /// Wall-clock limit for one run (or one resume). Model and tool calls
    /// still in flight at the deadline are cancelled; the run ends with
    /// [`BudgetKind::Deadline`].
    #[must_use]
    pub const fn max_duration(mut self, limit: Duration) -> Self {
        self.max_duration = Some(limit);
        self
    }

    /// Truncate any single tool output to this many characters.
    ///
    /// One chatty tool must not eat the whole context window.
    #[must_use]
    pub const fn per_tool_output_limit(mut self, limit: usize) -> Self {
        self.per_tool_output_limit = limit;
        self
    }

    /// Sampling temperature for model calls.
    #[must_use]
    pub const fn temperature(mut self, temperature: f32) -> Self {
        self.temperature = Some(temperature);
        self
    }

    /// Add a lifecycle hook. Hooks run in the order they were added.
    #[must_use]
    pub fn hook(mut self, hook: Arc<dyn AgentHooks>) -> Self {
        self.hooks.push(hook);
        self
    }

    /// Decide per call whether the agent may act. Defaults to
    /// [`AllowAll`].
    #[must_use]
    pub fn policy(mut self, policy: Arc<dyn ToolPolicy>) -> Self {
        self.policy = policy;
        self
    }

    /// Loop-detection thresholds. Use [`LoopGuard::disabled`] to turn it off.
    #[must_use]
    pub const fn loop_guard(mut self, guard: LoopGuard) -> Self {
        self.loop_guard = guard;
        self
    }

    /// Run the calls of one round concurrently (the default) or one by one.
    #[must_use]
    pub const fn parallel_tool_calls(mut self, parallel: bool) -> Self {
        self.parallel_tool_calls = parallel;
        self
    }

    /// Give the agent persistent memory: a frozen snapshot in the system
    /// prompt plus the `memory` tool, both bound to `scope`.
    #[must_use]
    pub fn memory(mut self, store: Arc<dyn MemoryStore>, scope: MemoryScope) -> Self {
        self.memory = Some((store, scope));
        self
    }

    /// Offer skills: an index in the system prompt plus the `load_skill`
    /// tool.
    #[must_use]
    pub fn skills(mut self, skills: impl Into<Arc<[Skill]>>) -> Self {
        self.skills = skills.into();
        self
    }

    /// Summarize long session transcripts before a run. Applies to
    /// [`Agent::run_in_session`].
    #[must_use]
    pub const fn compaction(mut self, compaction: Compaction) -> Self {
        self.compaction = Some(compaction);
        self
    }

    /// Use this id for the next run instead of generating one. Background
    /// jobs set it so a retried job keeps the same run id.
    #[must_use]
    pub fn run_id(mut self, run_id: RunId) -> Self {
        self.run_id = Some(run_id);
        self
    }

    /// Run the agent loop against one user message, with no prior history.
    ///
    /// # Errors
    ///
    /// Returns [`AgentError`] when a provider call fails (transport, auth,
    /// rate limit, malformed response) or memory cannot load. Budget
    /// exhaustion, loops, and approval pauses are not errors: they come back
    /// as [`AgentOutcome`] variants.
    pub async fn run(&self, user_input: &str) -> Result<AgentOutcome, AgentError> {
        Ok(self.run_turn(Vec::new(), user_input).await?.outcome)
    }

    /// Continue a transcript with one user message.
    ///
    /// `history` is a transcript from an earlier [`AgentTurn::messages`]
    /// (no system messages needed: the agent adds its own).
    ///
    /// # Errors
    ///
    /// See [`Agent::run`].
    pub async fn run_turn(
        &self,
        history: Vec<ChatMessage>,
        user_input: &str,
    ) -> Result<AgentTurn, AgentError> {
        self.start(history, user_input, None).await
    }

    /// Run one turn inside a persisted session.
    ///
    /// Loads the transcript, compacts it when [`Agent::compaction`] is set
    /// and the transcript is long, runs, then saves the new transcript. A run
    /// that pauses for approval saves nothing: [`Agent::resume_in_session`]
    /// saves once the run finishes. A new message sent to the session while
    /// a run waits for approval continues from the last saved transcript.
    ///
    /// # Errors
    ///
    /// See [`Agent::run`]; also fails when the store or the compaction call
    /// fails.
    pub async fn run_in_session(
        &self,
        store: &dyn SessionStore,
        session_id: &SessionId,
        user_input: &str,
    ) -> Result<AgentTurn, AgentError> {
        let mut history = store.load(session_id).await?;
        let compaction = match &self.compaction {
            Some(settings) => compact(self.client.as_ref(), &mut history, settings).await?,
            None => None,
        };
        let mut turn = self
            .start(history, user_input, Some(session_id.clone()))
            .await?;
        if !matches!(turn.outcome, AgentOutcome::AwaitingApproval { .. }) {
            store.save(session_id, &turn.messages).await?;
        }
        turn.compaction = compaction;
        Ok(turn)
    }

    /// Continue a run that paused with [`AgentOutcome::AwaitingApproval`].
    ///
    /// Calls the policy allowed run; calls it denied report their reason;
    /// calls that needed approval follow `decisions` — a call with no
    /// decision counts as rejected. The deadline restarts on resume.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::Config`] when a decision names a call that is
    /// not waiting for approval, plus every error of [`Agent::run`].
    pub async fn resume(
        &self,
        state: RunState,
        decisions: Vec<ApprovalDecision>,
    ) -> Result<AgentTurn, AgentError> {
        for decision in &decisions {
            if !state
                .awaiting()
                .any(|pending| pending.call.id == decision.call_id)
            {
                return Err(AgentError::new(
                    ErrorKind::Config,
                    format!(
                        "approval decision names call {:?}, which is not waiting for approval",
                        decision.call_id
                    ),
                ));
            }
        }
        let prepared = self.prepare().await?;
        let deadline = self.deadline();
        let gates: Vec<(ToolCall, Gate)> = state
            .pending
            .iter()
            .map(|pending| {
                let gate = match &pending.status {
                    PendingStatus::Allowed => Gate::Allowed,
                    PendingStatus::Denied { reason } => Gate::Denied(reason.clone()),
                    PendingStatus::NeedsApproval { .. } => decisions
                        .iter()
                        .find(|decision| decision.call_id == pending.call.id)
                        .map_or_else(
                            || Gate::Rejected("no decision was given".to_owned()),
                            |decision| match &decision.approval {
                                Approval::Approve => Gate::Approved(None),
                                Approval::Edit { arguments } => {
                                    Gate::Approved(Some(arguments.clone()))
                                }
                                Approval::Reject { reason } => Gate::Rejected(reason.clone()),
                            },
                        ),
                };
                (pending.call.clone(), gate)
            })
            .collect();
        let mut state_now = LoopState {
            run_id: state.run_id,
            session_id: state.session_id,
            messages: state.messages,
            usage: state.usage,
            steps: state.steps_used,
            tracker: LoopTracker::from_saved(state.recent_calls),
        };
        let stopped = self
            .execute_round(&prepared, &mut state_now, gates, deadline)
            .await;
        let outcome = match stopped {
            Some(outcome) => outcome,
            None => self.drive(&prepared, &mut state_now, deadline).await?,
        };
        Ok(self.finish(outcome, state_now).await)
    }

    /// [`Agent::resume`], then save the transcript to the run's session.
    ///
    /// # Errors
    ///
    /// See [`Agent::resume`]; also fails when the store fails.
    pub async fn resume_in_session(
        &self,
        store: &dyn SessionStore,
        state: RunState,
        decisions: Vec<ApprovalDecision>,
    ) -> Result<AgentTurn, AgentError> {
        let session_id = state.session_id.clone();
        let turn = self.resume(state, decisions).await?;
        if let Some(session_id) = session_id
            && !matches!(turn.outcome, AgentOutcome::AwaitingApproval { .. })
        {
            store.save(&session_id, &turn.messages).await?;
        }
        Ok(turn)
    }

    fn deadline(&self) -> Option<Instant> {
        self.max_duration.map(|limit| Instant::now() + limit)
    }

    /// Start a fresh run on top of `history`.
    async fn start(
        &self,
        mut history: Vec<ChatMessage>,
        user_input: &str,
        session_id: Option<SessionId>,
    ) -> Result<AgentTurn, AgentError> {
        let prepared = self.prepare().await?;
        let deadline = self.deadline();
        history.retain(|message| message.role != ChatRole::System);
        history.push(ChatMessage::text(ChatRole::User, user_input));
        let mut state = LoopState {
            run_id: self.run_id.clone().unwrap_or_else(RunId::generate),
            session_id,
            messages: history,
            usage: TokenUsage::default(),
            steps: 0,
            tracker: LoopTracker::default(),
        };
        let outcome = self.drive(&prepared, &mut state, deadline).await?;
        Ok(self.finish(outcome, state).await)
    }

    /// Render the system prompt (base + skills + memory snapshot) and
    /// assemble the run's tool list.
    async fn prepare(&self) -> Result<Prepared, AgentError> {
        let mut sections: Vec<String> = Vec::new();
        if let Some(system) = &self.system_prompt {
            sections.push(system.clone());
        }
        let mut tools = self.tools.clone();
        if !self.skills.is_empty() {
            sections.push(render_index(&self.skills));
            tools.push(Arc::new(SkillTool::new(Arc::clone(&self.skills))));
        }
        if let Some((store, scope)) = &self.memory {
            let blocks = store.load(scope).await?;
            sections.push(render_snapshot(&blocks));
            tools.push(Arc::new(MemoryTool::new(Arc::clone(store), scope.clone())));
        }
        let system = if sections.is_empty() {
            None
        } else {
            Some(sections.join("\n\n"))
        };
        Ok(Prepared { system, tools })
    }

    fn info(&self, state: &LoopState) -> RunInfo {
        RunInfo {
            run_id: state.run_id.clone(),
            session_id: state.session_id.clone(),
            steps_used: state.steps,
            max_steps: self.max_steps,
            usage: state.usage,
        }
    }

    /// Fire the `on_outcome` hooks and package the turn.
    async fn finish(&self, outcome: AgentOutcome, state: LoopState) -> AgentTurn {
        let info = self.info(&state);
        for hook in &self.hooks {
            hook.on_outcome(&outcome, &info).await;
        }
        AgentTurn {
            outcome,
            messages: state.messages,
            compaction: None,
        }
    }

    /// Check the step and token budgets before another model call.
    ///
    /// Returns the remaining token budget, or the [`AgentOutcome`] to return
    /// when a budget is already exhausted.
    fn check_budgets(&self, steps: u32, usage: TokenUsage) -> Result<u32, AgentOutcome> {
        let exhausted = |reason| AgentOutcome::BudgetExhausted {
            reason,
            steps_used: steps,
            usage,
        };
        if steps >= self.max_steps {
            return Err(exhausted(BudgetKind::Steps));
        }
        let spent = usage.total();
        if spent >= self.max_tokens {
            return Err(exhausted(BudgetKind::Tokens));
        }
        let remaining = self.max_tokens - spent;
        if remaining < Self::MIN_USEFUL_REMAINING {
            return Err(exhausted(BudgetKind::Tokens));
        }
        Ok(remaining)
    }

    /// The loop: call the model, gate and run its tool calls, repeat.
    async fn drive(
        &self,
        prepared: &Prepared,
        state: &mut LoopState,
        deadline: Option<Instant>,
    ) -> Result<AgentOutcome, AgentError> {
        loop {
            let exhausted = |reason, state: &LoopState| AgentOutcome::BudgetExhausted {
                reason,
                steps_used: state.steps,
                usage: state.usage,
            };
            if deadline.is_some_and(|deadline| Instant::now() >= deadline) {
                return Ok(exhausted(BudgetKind::Deadline, state));
            }
            let remaining = match self.check_budgets(state.steps, state.usage) {
                Ok(remaining) => remaining,
                Err(outcome) => return Ok(outcome),
            };
            let mut full = Vec::with_capacity(state.messages.len().saturating_add(1));
            if let Some(system) = &prepared.system {
                full.push(ChatMessage::text(ChatRole::System, system.clone()));
            }
            full.extend(state.messages.iter().cloned());
            let history = truncate_history(&full, remaining);
            let history_tokens = estimate_messages(&history);
            let has_turns = history
                .iter()
                .any(|message| message.role != ChatRole::System);
            if history_tokens > remaining || !has_turns {
                return Ok(exhausted(BudgetKind::Tokens, state));
            }
            // Reserve the estimated input cost before sizing the output cap:
            // the provider counts the history against the same budget.
            let output_budget = remaining.saturating_sub(history_tokens);
            if output_budget == 0 {
                return Ok(exhausted(BudgetKind::Tokens, state));
            }
            let mut request = ChatRequest {
                messages: history,
                tools: prepared
                    .tools
                    .iter()
                    .map(|tool| tool.definition())
                    .collect(),
                max_tokens: Some(output_budget.min(Self::MAX_CALL_TOKENS)),
                temperature: self.temperature,
            };
            let info = self.info(state);
            for hook in &self.hooks {
                hook.before_model(&mut request, &info).await;
            }
            tracing::debug!(
                run_id = %state.run_id,
                step = state.steps,
                remaining_tokens = remaining,
                "agent loop calling model"
            );
            let response = match deadline {
                Some(deadline) => {
                    match tokio::time::timeout_at(deadline, self.client.chat(&request)).await {
                        Ok(response) => response?,
                        Err(_) => return Ok(exhausted(BudgetKind::Deadline, state)),
                    }
                }
                None => self.client.chat(&request).await?,
            };
            for hook in &self.hooks {
                hook.after_model(&response, &info).await;
            }
            let calls = match self.fold_response(state, response) {
                Ok(calls) => calls,
                Err(outcome) => return Ok(outcome),
            };
            let (gates, pending) = self.gate_calls(prepared, state, calls).await;
            if pending
                .iter()
                .any(|pending| matches!(pending.status, PendingStatus::NeedsApproval { .. }))
            {
                tracing::info!(run_id = %state.run_id, "agent run paused for approval");
                return Ok(AgentOutcome::AwaitingApproval {
                    state: Box::new(RunState {
                        run_id: state.run_id.clone(),
                        session_id: state.session_id.clone(),
                        messages: state.messages.clone(),
                        usage: state.usage,
                        steps_used: state.steps,
                        pending,
                        recent_calls: state.tracker.saved(),
                    }),
                });
            }
            if let Some(outcome) = self.execute_round(prepared, state, gates, deadline).await {
                return Ok(outcome);
            }
        }
    }

    /// Ask the policy about each call of a round.
    ///
    /// Returns the gates to run with now and the same calls in pending form,
    /// for a pause.
    async fn gate_calls(
        &self,
        prepared: &Prepared,
        state: &LoopState,
        calls: Vec<ToolCall>,
    ) -> (Vec<(ToolCall, Gate)>, Vec<PendingCall>) {
        let info = self.info(state);
        let mut gates = Vec::with_capacity(calls.len());
        let mut pending = Vec::with_capacity(calls.len());
        for call in calls {
            let tool = prepared.tools.iter().find(|tool| tool.name() == call.name);
            let decision = self
                .policy
                .decide(&call, tool.map(AsRef::as_ref), &info)
                .await;
            let (status, gate) = match decision {
                ToolDecision::Allow => (PendingStatus::Allowed, Gate::Allowed),
                ToolDecision::RequireApproval { reason } => {
                    (PendingStatus::NeedsApproval { reason }, Gate::Allowed)
                }
                ToolDecision::Deny { reason } => (
                    PendingStatus::Denied {
                        reason: reason.clone(),
                    },
                    Gate::Denied(reason),
                ),
            };
            pending.push(PendingCall {
                call: call.clone(),
                status,
            });
            gates.push((call, gate));
        }
        (gates, pending)
    }

    /// Fold one provider response into the loop state.
    ///
    /// Adds the exact provider-reported usage, then either yields the run's
    /// final [`AgentOutcome`] — a final answer, or a token overshoot, which
    /// is exhaustion even on an otherwise-final response — or records the
    /// assistant turn and returns the tool calls to gate and run.
    fn fold_response(
        &self,
        state: &mut LoopState,
        response: ChatResponse,
    ) -> Result<Vec<ToolCall>, AgentOutcome> {
        state.usage = state.usage.saturating_add(response.usage);
        let calls: Vec<ToolCall> = response
            .content
            .iter()
            .filter_map(|part| match part {
                ContentPart::ToolCall {
                    id,
                    name,
                    arguments,
                } => Some(ToolCall {
                    id: id.clone(),
                    name: name.clone(),
                    arguments: arguments.clone(),
                }),
                ContentPart::Text(_) | ContentPart::ToolResult { .. } => None,
            })
            .collect();
        let overshoot = state.usage.total() > self.max_tokens;
        if calls.is_empty() {
            let text = response
                .content
                .iter()
                .filter_map(|part| match part {
                    ContentPart::Text(text) => Some(text.as_str()),
                    ContentPart::ToolCall { .. } | ContentPart::ToolResult { .. } => None,
                })
                .collect::<Vec<_>>()
                .join("");
            state.messages.push(ChatMessage {
                role: ChatRole::Assistant,
                content: response.content,
            });
            return Err(if overshoot {
                AgentOutcome::BudgetExhausted {
                    reason: BudgetKind::Tokens,
                    steps_used: state.steps,
                    usage: state.usage,
                }
            } else {
                AgentOutcome::Completed {
                    text,
                    steps_used: state.steps,
                    usage: state.usage,
                }
            });
        }
        if overshoot {
            // Tool calls without results would corrupt the transcript, so the
            // overshooting turn is not recorded.
            return Err(AgentOutcome::BudgetExhausted {
                reason: BudgetKind::Tokens,
                steps_used: state.steps,
                usage: state.usage,
            });
        }
        state.steps += 1;
        state.messages.push(ChatMessage {
            role: ChatRole::Assistant,
            content: response.content,
        });
        Ok(calls)
    }

    /// Run one round of gated calls, append the results, and apply the loop
    /// guard. Returns an outcome when the guard stops the run.
    async fn execute_round(
        &self,
        prepared: &Prepared,
        state: &mut LoopState,
        gates: Vec<(ToolCall, Gate)>,
        deadline: Option<Instant>,
    ) -> Option<AgentOutcome> {
        // The round's assistant message was step `steps - 1`.
        let step = state.steps.saturating_sub(1);
        let contexts: Vec<ToolContext> = gates
            .iter()
            .map(|(call, _)| ToolContext {
                run_id: state.run_id.clone(),
                call_id: call.id.clone(),
                session_id: state.session_id.clone(),
                step,
            })
            .collect();
        let outputs: Vec<ToolOutput> = if self.parallel_tool_calls {
            futures::future::join_all(
                gates
                    .iter()
                    .zip(&contexts)
                    .map(|((call, gate), ctx)| self.run_call(prepared, call, gate, ctx, deadline)),
            )
            .await
        } else {
            let mut outputs = Vec::with_capacity(gates.len());
            for ((call, gate), ctx) in gates.iter().zip(&contexts) {
                outputs.push(self.run_call(prepared, call, gate, ctx, deadline).await);
            }
            outputs
        };
        let mut results = Vec::with_capacity(outputs.len());
        let mut stopped = None;
        for ((call, _), output) in gates.iter().zip(outputs) {
            let mut content = output.content;
            match state
                .tracker
                .record(&self.loop_guard, &call.name, &call.arguments, &content)
            {
                LoopVerdict::Ok => {}
                LoopVerdict::Warn { repeats } => {
                    tracing::warn!(run_id = %state.run_id, tool = %call.name, repeats, "agent is repeating a tool call");
                    content.push_str(&warning_note(&call.name, repeats));
                }
                LoopVerdict::Stop { repeats } => {
                    if stopped.is_none() {
                        stopped = Some((call.name.clone(), repeats));
                    }
                }
            }
            results.push(ContentPart::ToolResult {
                tool_call_id: call.id.clone(),
                content,
            });
        }
        state.messages.push(ChatMessage {
            role: ChatRole::Tool,
            content: results,
        });
        stopped.map(|(tool, repeats)| {
            tracing::warn!(run_id = %state.run_id, tool = %tool, repeats, "agent loop detected; stopping the run");
            AgentOutcome::LoopDetected {
                tool,
                repeats,
                steps_used: state.steps,
                usage: state.usage,
            }
        })
    }

    /// Run one gated call through the hooks and render its result.
    ///
    /// Unknown tools, failures, blocks, denials, and rejections become error
    /// payloads the model can see and recover from; they never abort the run.
    async fn run_call(
        &self,
        prepared: &Prepared,
        call: &ToolCall,
        gate: &Gate,
        ctx: &ToolContext,
        deadline: Option<Instant>,
    ) -> ToolOutput {
        let mut effective = call.clone();
        let output = match gate {
            Gate::Denied(reason) => error_output(&format!("denied by policy: {reason}")),
            Gate::Rejected(reason) => {
                error_output(&format!("a reviewer rejected this call: {reason}"))
            }
            Gate::Allowed | Gate::Approved(_) => {
                if let Gate::Approved(Some(arguments)) = gate {
                    effective.arguments = arguments.clone();
                }
                self.invoke(prepared, &mut effective, ctx, deadline).await
            }
        };
        for hook in &self.hooks {
            hook.after_tool(&effective, &output, ctx).await;
        }
        output
    }

    /// Apply the `before_tool` hooks, then execute the tool.
    async fn invoke(
        &self,
        prepared: &Prepared,
        call: &mut ToolCall,
        ctx: &ToolContext,
        deadline: Option<Instant>,
    ) -> ToolOutput {
        for hook in &self.hooks {
            match hook.before_tool(call, ctx).await {
                HookAction::Continue => {}
                HookAction::Modify(arguments) => call.arguments = arguments,
                HookAction::Block(reason) => {
                    return error_output(&format!("blocked by hook: {reason}"));
                }
            }
        }
        let Some(tool) = prepared.tools.iter().find(|tool| tool.name() == call.name) else {
            tracing::warn!(
                tool_call_id = %call.id,
                tool = %call.name,
                "model called an unknown tool"
            );
            return error_output(&format!("unknown tool {:?}", call.name));
        };
        let execution = tool.execute(call.arguments.clone(), ctx);
        let result = match deadline {
            Some(deadline) => match tokio::time::timeout_at(deadline, execution).await {
                Ok(result) => result,
                Err(_) => {
                    return error_output("the tool did not finish before the run deadline");
                }
            },
            None => execution.await,
        };
        match result {
            Ok(output) => ToolOutput {
                content: truncate_chars(&output.to_string(), self.per_tool_output_limit),
                is_error: false,
            },
            Err(err) => {
                tracing::warn!(
                    tool_call_id = %call.id,
                    tool = %call.name,
                    error = %err,
                    "tool execution failed"
                );
                error_output(err.message())
            }
        }
    }
}

fn error_output(message: &str) -> ToolOutput {
    ToolOutput {
        content: serde_json::json!({"error": message}).to_string(),
        is_error: true,
    }
}

/// Truncate a string to a character limit, marking the cut.
fn truncate_chars(text: &str, limit: usize) -> String {
    if text.chars().count() <= limit {
        return text.to_owned();
    }
    let kept: String = text.chars().take(limit).collect();
    format!("{kept}…[truncated]")
}

/// Shared agent state installed on [`AppState`](autumn_web::AppState) by the
/// plugin.
///
/// Handlers reach it through the
/// [`AgentHandle`](crate::plugin::AgentHandle) extractor; the background
/// jobs and the heartbeat read it directly.
#[derive(Debug, Clone)]
pub struct AgentRuntime {
    client: Arc<dyn LlmClient>,
    tools: Vec<Arc<dyn Tool>>,
    config: AgentConfig,
    hooks: Vec<Arc<dyn AgentHooks>>,
    policy: Arc<dyn ToolPolicy>,
    sessions: Arc<dyn SessionStore>,
    memory: Option<Arc<dyn MemoryStore>>,
    skills: Arc<[Skill]>,
    compaction: Option<Compaction>,
    delivery: Arc<dyn Delivery>,
    #[cfg(feature = "autumn")]
    followups: Option<Duration>,
}

impl AgentRuntime {
    /// Bundle a client, tools, and config into the shared runtime.
    ///
    /// Starts with an in-memory session store, no memory, no skills, the
    /// [`AllowAll`] policy, and [`LogDelivery`]. Adjust with the `with_*`
    /// methods.
    #[must_use]
    pub fn new(config: AgentConfig, client: Arc<dyn LlmClient>, tools: Vec<Arc<dyn Tool>>) -> Self {
        Self {
            client,
            tools,
            config,
            hooks: Vec::new(),
            policy: Arc::new(AllowAll),
            sessions: Arc::new(crate::session::InMemorySessionStore::new()),
            memory: None,
            skills: Arc::from(Vec::new()),
            compaction: None,
            delivery: Arc::new(LogDelivery),
            #[cfg(feature = "autumn")]
            followups: None,
        }
    }

    /// Add lifecycle hooks to every agent the runtime builds.
    #[must_use]
    pub fn with_hooks(mut self, hooks: Vec<Arc<dyn AgentHooks>>) -> Self {
        self.hooks.extend(hooks);
        self
    }

    /// Set the tool policy for every agent the runtime builds.
    #[must_use]
    pub fn with_policy(mut self, policy: Arc<dyn ToolPolicy>) -> Self {
        self.policy = policy;
        self
    }

    /// Replace the session store.
    #[must_use]
    pub fn with_sessions(mut self, sessions: Arc<dyn SessionStore>) -> Self {
        self.sessions = sessions;
        self
    }

    /// Give every agent persistent memory in this store.
    #[must_use]
    pub fn with_memory(mut self, memory: Arc<dyn MemoryStore>) -> Self {
        self.memory = Some(memory);
        self
    }

    /// Offer these skills to every agent.
    #[must_use]
    pub fn with_skills(mut self, skills: impl Into<Arc<[Skill]>>) -> Self {
        self.skills = skills.into();
        self
    }

    /// Compact long session transcripts.
    #[must_use]
    pub const fn with_compaction(mut self, compaction: Compaction) -> Self {
        self.compaction = Some(compaction);
        self
    }

    /// Where background results go.
    #[must_use]
    pub fn with_delivery(mut self, delivery: Arc<dyn Delivery>) -> Self {
        self.delivery = delivery;
        self
    }

    /// Give agents the `schedule_followup` tool, with this maximum delay.
    #[cfg(feature = "autumn")]
    #[must_use]
    pub const fn with_followups(mut self, max_delay: Duration) -> Self {
        self.followups = Some(max_delay);
        self
    }

    /// Build an [`Agent`] with the runtime's client, tools, budgets, hooks,
    /// policy, skills, and memory (scope [`MemoryScope::agent`]).
    #[must_use]
    pub fn agent(&self) -> Agent {
        self.agent_for(MemoryScope::agent())
    }

    /// Like [`AgentRuntime::agent`], with memory bound to `scope` (for
    /// example one scope per user).
    #[must_use]
    pub fn agent_for(&self, scope: MemoryScope) -> Agent {
        self.agent_with(scope, 0)
    }

    /// Build an agent for a run that is `followup_depth` follow-ups deep.
    #[cfg_attr(not(feature = "autumn"), allow(unused_variables))]
    pub(crate) fn agent_with(&self, scope: MemoryScope, followup_depth: u32) -> Agent {
        let mut agent = Agent::new(Arc::clone(&self.client))
            .tools(self.tools.clone())
            .max_steps(self.config.max_steps)
            .max_tokens(self.config.max_tokens)
            .policy(Arc::clone(&self.policy))
            .skills(Arc::clone(&self.skills));
        for hook in &self.hooks {
            agent = agent.hook(Arc::clone(hook));
        }
        if let Some(system) = &self.config.system_prompt {
            agent = agent.system_prompt(system.clone());
        }
        if let Some(secs) = self.config.max_run_secs {
            agent = agent.max_duration(Duration::from_secs(secs));
        }
        #[cfg(feature = "autumn")]
        if let Some(max_delay) = self.followups {
            agent = agent.tool(Arc::new(
                FollowupTool::new(max_delay)
                    .memory_scope(scope.clone())
                    .depth(followup_depth),
            ));
        }
        if let Some(memory) = &self.memory {
            agent = agent.memory(Arc::clone(memory), scope);
        }
        if let Some(compaction) = self.compaction {
            agent = agent.compaction(compaction);
        }
        agent
    }

    /// The plugin configuration backing this runtime.
    #[must_use]
    pub const fn config(&self) -> &AgentConfig {
        &self.config
    }

    /// The provider client this runtime runs agents with.
    ///
    /// The health indicator uses the same client, so `/actuator/health`
    /// always reflects the provider the app actually talks to.
    #[must_use]
    pub fn client(&self) -> Arc<dyn LlmClient> {
        Arc::clone(&self.client)
    }

    /// The tool policy agents run under.
    #[must_use]
    pub fn policy(&self) -> Arc<dyn ToolPolicy> {
        Arc::clone(&self.policy)
    }

    /// The session store.
    #[must_use]
    pub fn sessions(&self) -> Arc<dyn SessionStore> {
        Arc::clone(&self.sessions)
    }

    /// The memory store, when memory is on.
    #[must_use]
    pub fn memory(&self) -> Option<Arc<dyn MemoryStore>> {
        self.memory.clone()
    }

    /// Where background results go.
    #[must_use]
    pub fn delivery(&self) -> Arc<dyn Delivery> {
        Arc::clone(&self.delivery)
    }

    /// Names of the registered tools.
    #[must_use]
    pub fn tool_names(&self) -> Vec<&str> {
        self.tools.iter().map(|tool| tool.name()).collect()
    }
}

#[cfg(feature = "autumn")]
pub use crate::jobs::{AgentRunArgs, enqueue_agent_run};

#[cfg(test)]
mod tests;
