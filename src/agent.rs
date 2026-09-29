//! The agent loop: prompt the model, run its tool calls, repeat.
//!
//! [`Agent`] drives the loop. It stops on a final answer or when a budget
//! runs out — [`AgentOutcome::BudgetExhausted`] is a normal result, not an
//! error, so callers always get a structured answer back.
//!
//! [`agent_run_job::run_agent`] exposes the same loop as an Autumn background
//! job: handlers enqueue [`AgentRunArgs`] and a worker runs the loop
//! off-request.

use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::client::{
    ChatMessage, ChatRequest, ChatResponse, ChatRole, ContentPart, LlmClient, TokenUsage,
};
use crate::config::AgentConfig;
use crate::error::AgentError;
use crate::tools::Tool;
use autumn_web::AutumnResult;

/// One model-requested tool call: id, name, arguments.
type PendingToolCall = (String, String, serde_json::Value);

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
                .sum::<u32>()
        })
        .sum::<u32>()
        .saturating_add(
            u32::try_from(messages.len())
                .unwrap_or(u32::MAX)
                .saturating_mul(4),
        ) // per-message framing overhead
}

/// Shrink a history to fit a token budget.
///
/// Keeps every system message, then keeps the newest non-system messages
/// that fit. Order is preserved. When even the system prompt alone exceeds
/// the budget, the system messages come back anyway and the caller treats
/// that as budget exhaustion.
#[must_use]
pub fn truncate_history(messages: &[ChatMessage], budget: u32) -> Vec<ChatMessage> {
    let mut kept: Vec<(usize, &ChatMessage)> = Vec::new();
    let mut used: u32 = 0;
    for (index, message) in messages.iter().enumerate() {
        if message.role == ChatRole::System {
            used = used.saturating_add(estimate_messages(std::slice::from_ref(message)));
            kept.push((index, message));
        }
    }
    let mut tail: Vec<(usize, &ChatMessage)> = Vec::new();
    for (index, message) in messages.iter().enumerate().rev() {
        if message.role == ChatRole::System {
            continue;
        }
        let cost = estimate_messages(std::slice::from_ref(message));
        if used.saturating_add(cost) <= budget {
            used = used.saturating_add(cost);
            tail.push((index, message));
        } else {
            break;
        }
    }
    kept.extend(tail.into_iter().rev());
    kept.sort_by_key(|(index, _)| *index);
    kept.into_iter()
        .map(|(_, message)| message.clone())
        .collect()
}

/// Which budget stopped the run.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BudgetKind {
    /// `max_steps` iterations ran out.
    Steps,
    /// `max_tokens` ran out.
    Tokens,
}

/// The result of one [`Agent::run`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AgentOutcome {
    /// The model produced a final answer.
    Completed {
        /// The model's final text.
        text: String,
        /// Loop iterations used.
        steps_used: u32,
        /// Provider-reported token totals.
        usage: TokenUsage,
    },
    /// A budget ran out before the model finished.
    BudgetExhausted {
        /// Which budget ran out.
        reason: BudgetKind,
        /// Loop iterations used.
        steps_used: u32,
        /// Provider-reported token totals.
        usage: TokenUsage,
    },
}

impl AgentOutcome {
    /// Loop iterations consumed, whatever the outcome.
    #[must_use]
    pub const fn steps_used(&self) -> u32 {
        match self {
            Self::Completed { steps_used, .. } | Self::BudgetExhausted { steps_used, .. } => {
                *steps_used
            }
        }
    }

    /// Provider-reported token totals, whatever the outcome.
    #[must_use]
    pub const fn usage(&self) -> TokenUsage {
        match self {
            Self::Completed { usage, .. } | Self::BudgetExhausted { usage, .. } => *usage,
        }
    }

    /// `true` when the model produced a final answer.
    #[must_use]
    pub const fn is_completed(&self) -> bool {
        matches!(self, Self::Completed { .. })
    }
}

/// The agent loop.
///
/// ```rust,no_run
/// # use autumn_plugin_agent::{Agent, AgentOutcome};
/// # async fn demo(agent: Agent) -> Result<(), autumn_plugin_agent::AgentError> {
/// match agent.run("What is the weather in Chicago?").await? {
///     AgentOutcome::Completed { text, .. } => println!("answer: {text}"),
///     AgentOutcome::BudgetExhausted { reason, .. } => println!("stopped: {reason:?}"),
/// }
/// # Ok(())
/// # }
/// ```
pub struct Agent {
    client: Arc<dyn LlmClient>,
    tools: Vec<Arc<dyn Tool>>,
    system_prompt: Option<String>,
    max_steps: u32,
    max_tokens: u32,
    per_tool_output_limit: usize,
    temperature: Option<f32>,
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
            .finish_non_exhaustive()
    }
}

impl Agent {
    /// Minimum remaining budget (tokens) that still justifies another model call.
    const MIN_USEFUL_REMAINING: u32 = 512;
    /// Per-call output cap ceiling; the loop never asks for more than this.
    const MAX_CALL_TOKENS: u32 = 4096;

    /// Check the step and token budgets before another model call.
    ///
    /// Returns the remaining token budget, or the [`AgentOutcome`] to return
    /// when a budget is already exhausted.
    const fn check_budgets(&self, steps: u32, usage: TokenUsage) -> Result<u32, AgentOutcome> {
        if steps >= self.max_steps {
            return Err(AgentOutcome::BudgetExhausted {
                reason: BudgetKind::Steps,
                steps_used: steps,
                usage,
            });
        }
        let spent = usage.total();
        if spent >= self.max_tokens {
            return Err(AgentOutcome::BudgetExhausted {
                reason: BudgetKind::Tokens,
                steps_used: steps,
                usage,
            });
        }
        let remaining = self.max_tokens - spent;
        if remaining < Self::MIN_USEFUL_REMAINING {
            return Err(AgentOutcome::BudgetExhausted {
                reason: BudgetKind::Tokens,
                steps_used: steps,
                usage,
            });
        }
        Ok(remaining)
    }

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
        }
    }

    /// Tools the model may call.
    #[must_use]
    pub fn tools(mut self, tools: Vec<Arc<dyn Tool>>) -> Self {
        self.tools = tools;
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

    /// Run the agent loop against one user message.
    ///
    /// Each iteration calls the model, executes any tool calls it emits,
    /// appends the results, and repeats until the model answers or a budget
    /// runs out.
    ///
    /// # Errors
    ///
    /// Returns [`AgentError`] when a provider call fails (transport, auth,
    /// rate limit, malformed response). Budget exhaustion is not an error:
    /// it comes back as [`AgentOutcome::BudgetExhausted`].
    pub async fn run(&self, user_input: &str) -> Result<AgentOutcome, AgentError> {
        let mut messages = Vec::new();
        if let Some(system) = &self.system_prompt {
            messages.push(ChatMessage::text(ChatRole::System, system.clone()));
        }
        messages.push(ChatMessage::text(ChatRole::User, user_input));

        let mut usage = TokenUsage::default();
        let mut steps: u32 = 0;

        loop {
            let remaining = match self.check_budgets(steps, usage) {
                Ok(remaining) => remaining,
                Err(outcome) => return Ok(outcome),
            };
            let history = truncate_history(&messages, remaining);
            let history_tokens = estimate_messages(&history);
            if history_tokens > remaining {
                return Ok(AgentOutcome::BudgetExhausted {
                    reason: BudgetKind::Tokens,
                    steps_used: steps,
                    usage,
                });
            }
            // Reserve the estimated input cost before sizing the output cap:
            // the provider counts the history against the same budget.
            let output_budget = remaining.saturating_sub(history_tokens);
            if output_budget == 0 {
                return Ok(AgentOutcome::BudgetExhausted {
                    reason: BudgetKind::Tokens,
                    steps_used: steps,
                    usage,
                });
            }
            let call_cap = output_budget.min(Self::MAX_CALL_TOKENS);
            let request = ChatRequest {
                messages: history,
                tools: self.tools.iter().map(|tool| tool.definition()).collect(),
                max_tokens: Some(call_cap),
                temperature: self.temperature,
            };
            tracing::debug!(
                step = steps,
                remaining_tokens = remaining,
                "agent loop calling model"
            );
            let response = self.client.chat(&request).await?;
            let (new_usage, tool_calls) = match self.finish_turn(steps, usage, &response) {
                Ok(continued) => continued,
                Err(outcome) => return Ok(outcome),
            };
            usage = new_usage;

            steps += 1;
            messages.push(ChatMessage {
                role: ChatRole::Assistant,
                content: response.content,
            });

            let mut results = Vec::with_capacity(tool_calls.len());
            for (id, name, arguments) in tool_calls {
                let output = self.execute_tool(&id, &name, arguments).await;
                results.push(ContentPart::ToolResult {
                    tool_call_id: id,
                    content: output,
                });
            }
            messages.push(ChatMessage {
                role: ChatRole::Tool,
                content: results,
            });
        }
    }

    /// Fold one provider response into the loop state.
    ///
    /// Adds the exact provider-reported usage, then either yields the run's
    /// final [`AgentOutcome`] — a final answer, or a token overshoot, which
    /// is exhaustion even on an otherwise-final response — or the tool calls
    /// to execute before looping again.
    fn finish_turn(
        &self,
        steps: u32,
        usage: TokenUsage,
        response: &ChatResponse,
    ) -> Result<(TokenUsage, Vec<PendingToolCall>), AgentOutcome> {
        let usage = usage.saturating_add(response.usage);
        if usage.total() > self.max_tokens {
            return Err(AgentOutcome::BudgetExhausted {
                reason: BudgetKind::Tokens,
                steps_used: steps,
                usage,
            });
        }
        let tool_calls: Vec<PendingToolCall> = response
            .content
            .iter()
            .filter_map(|part| match part {
                ContentPart::ToolCall {
                    id,
                    name,
                    arguments,
                } => Some((id.clone(), name.clone(), arguments.clone())),
                ContentPart::Text(_) | ContentPart::ToolResult { .. } => None,
            })
            .collect();
        if tool_calls.is_empty() {
            let text = response
                .content
                .iter()
                .filter_map(|part| match part {
                    ContentPart::Text(text) => Some(text.as_str()),
                    ContentPart::ToolCall { .. } | ContentPart::ToolResult { .. } => None,
                })
                .collect::<Vec<_>>()
                .join("");
            return Err(AgentOutcome::Completed {
                text,
                steps_used: steps,
                usage,
            });
        }
        Ok((usage, tool_calls))
    }

    /// Execute one tool call and render its outcome as a result string.
    ///
    /// Unknown tools and tool failures become error payloads the model can
    /// see and recover from; they never abort the run.
    async fn execute_tool(&self, id: &str, name: &str, arguments: serde_json::Value) -> String {
        let Some(tool) = self.tools.iter().find(|tool| tool.name() == name) else {
            tracing::warn!(
                tool_call_id = id,
                tool = name,
                "model called an unknown tool"
            );
            return serde_json::json!({"error": format!("unknown tool {name:?}")}).to_string();
        };
        match tool.execute(arguments).await {
            Ok(output) => truncate_chars(&output.to_string(), self.per_tool_output_limit),
            Err(err) => {
                tracing::warn!(
                    tool_call_id = id,
                    tool = name,
                    error = %err,
                    "tool execution failed"
                );
                serde_json::json!({"error": err.message()}).to_string()
            }
        }
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
/// [`AgentHandle`](crate::plugin::AgentHandle) extractor; the background job
/// reads it directly.
#[derive(Debug, Clone)]
pub struct AgentRuntime {
    client: Arc<dyn LlmClient>,
    tools: Vec<Arc<dyn Tool>>,
    config: AgentConfig,
}

impl AgentRuntime {
    /// Bundle a client, tools, and config into the shared runtime.
    #[must_use]
    pub fn new(config: AgentConfig, client: Arc<dyn LlmClient>, tools: Vec<Arc<dyn Tool>>) -> Self {
        Self {
            client,
            tools,
            config,
        }
    }

    /// Build an [`Agent`] with the runtime's client, tools, and budgets.
    #[must_use]
    pub fn agent(&self) -> Agent {
        let mut agent = Agent::new(Arc::clone(&self.client))
            .tools(self.tools.clone())
            .max_steps(self.config.max_steps)
            .max_tokens(self.config.max_tokens);
        if let Some(system) = &self.config.system_prompt {
            agent = agent.system_prompt(system.clone());
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

    /// Names of the registered tools.
    #[must_use]
    pub fn tool_names(&self) -> Vec<&str> {
        self.tools.iter().map(|tool| tool.name()).collect()
    }
}

/// Arguments for the [`agent_run_job::run_agent`] background job.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AgentRunArgs {
    /// The user message that starts the run.
    pub prompt: String,
    /// Overrides the configured system prompt for this run.
    #[serde(default)]
    pub system_prompt: Option<String>,
    /// Overrides the configured step budget for this run.
    #[serde(default)]
    pub max_steps: Option<u32>,
}

/// The `agent_run` background job.
///
/// Lives in a `pub(crate)` module: the `#[job]` macro emits an undocumented
/// `RunAgentJob` handle, and keeping the module crate-internal keeps that
/// item out of the public API (and out of `missing_docs`' reach). Handlers
/// enqueue through [`enqueue_agent_run`].
#[doc(hidden)]
#[allow(missing_docs)]
pub(crate) mod agent_run_job {
    use super::{AgentOutcome, AgentRunArgs, AgentRuntime};
    use autumn_web::{AppState, AutumnError, AutumnResult};

    /// Run the agent loop as an Autumn background job.
    ///
    /// Enqueue from a handler with [`enqueue_agent_run`].
    /// The job resolves the plugin's [`AgentRuntime`] from app state, runs the
    /// loop with the plugin's registered tools, and logs the outcome. Tool and
    /// provider failures surface as job failures so the queue's retry/backoff
    /// policy applies.
    #[autumn_web::job(name = "agent_run", max_attempts = 3, backoff_ms = 1_000)]
    pub async fn run_agent(state: AppState, args: AgentRunArgs) -> AutumnResult<()> {
        let runtime = state
            .extension::<AgentRuntime>()
            .ok_or_else(|| AutumnError::service_unavailable_msg("agent plugin is not installed"))?;
        let mut agent = runtime.agent();
        if let Some(system) = args.system_prompt {
            agent = agent.system_prompt(system);
        }
        if let Some(max_steps) = args.max_steps {
            agent = agent.max_steps(max_steps);
        }
        match agent.run(&args.prompt).await {
            Ok(AgentOutcome::Completed {
                text,
                steps_used,
                usage,
            }) => {
                tracing::info!(
                    steps_used,
                    input_tokens = usage.input_tokens,
                    output_tokens = usage.output_tokens,
                    answer_chars = text.len(),
                    "background agent run completed"
                );
                Ok(())
            }
            Ok(AgentOutcome::BudgetExhausted {
                reason,
                steps_used,
                usage,
            }) => {
                tracing::warn!(
                    ?reason,
                    steps_used,
                    input_tokens = usage.input_tokens,
                    output_tokens = usage.output_tokens,
                    "background agent run exhausted its budget"
                );
                Ok(())
            }
            Err(err) => {
                tracing::error!(error = %err, "background agent run failed");
                Err(err.into_autumn_error())
            }
        }
    }
}

/// Enqueue an `agent_run` background job.
///
/// The job runs [`Agent::run`] with the plugin's tools off-request; see
/// [`AgentRunArgs`] for the per-run overrides. The queue's retry/backoff
/// policy applies on failure.
///
/// # Errors
///
/// Returns [`autumn_web::AutumnError`] when the arguments fail to serialize
/// or the job cannot be enqueued.
pub async fn enqueue_agent_run(args: AgentRunArgs) -> AutumnResult<()> {
    agent_run_job::RunAgentJob::enqueue(args).await
}

#[cfg(test)]
mod tests;
