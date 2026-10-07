//! Subagents: run a whole [`Agent`] as one tool.
//!
//! A parent agent hands a focused task to a child agent with its own system
//! prompt, tools, and budgets. The child's working context (every tool call
//! and result) never enters the parent's transcript; the parent sees only
//! the child's final answer. That keeps the parent's context small and lets
//! you give the child tools the parent should not hold directly.

use std::future::Future;
use std::pin::Pin;

use crate::agent::{Agent, AgentOutcome};
use crate::error::{AgentError, ErrorKind};
use crate::tools::{Tool, ToolContext, ToolEffect};

/// A [`Tool`] that runs a child [`Agent`] on a task and returns its answer.
///
/// ```rust,no_run
/// # use std::sync::Arc;
/// # use autumn_plugin_agent::{Agent, LlmClient};
/// use autumn_plugin_agent::delegate::AgentTool;
/// use autumn_plugin_agent::tools::ToolEffect;
///
/// # fn demo(client: Arc<dyn LlmClient>) {
/// let researcher = Agent::new(Arc::clone(&client))
///     .system_prompt("You research one question and answer in five lines.")
///     .max_steps(6);
/// let tool = AgentTool::new(
///     "research",
///     "Research one question in depth and return a short answer.",
///     researcher,
/// )
/// .effect(ToolEffect::ReadOnly);
/// let parent = Agent::new(client).tool(Arc::new(tool));
/// # }
/// ```
#[derive(Debug, Clone)]
pub struct AgentTool {
    name: String,
    description: String,
    agent: Agent,
    effect: ToolEffect,
}

impl AgentTool {
    /// Wrap `agent` as a tool. The model calls it with `{"task": "..."}`.
    #[must_use]
    pub fn new(name: impl Into<String>, description: impl Into<String>, agent: Agent) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            agent,
            effect: ToolEffect::Write,
        }
    }

    /// Declare the strongest effect among the child's tools. The default is
    /// [`ToolEffect::Write`].
    #[must_use]
    pub const fn effect(mut self, effect: ToolEffect) -> Self {
        self.effect = effect;
        self
    }
}

impl Tool for AgentTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "task": {
                    "type": "string",
                    "description": "The task, with every detail the helper needs: it cannot see this conversation.",
                }
            },
            "required": ["task"]
        })
    }

    fn effect(&self) -> ToolEffect {
        self.effect
    }

    fn execute<'a>(
        &'a self,
        input: serde_json::Value,
        _ctx: &'a ToolContext,
    ) -> Pin<Box<dyn Future<Output = Result<serde_json::Value, AgentError>> + Send + 'a>> {
        Box::pin(async move {
            let task = input
                .get("task")
                .and_then(serde_json::Value::as_str)
                .map(str::trim)
                .filter(|task| !task.is_empty())
                .ok_or_else(|| AgentError::new(ErrorKind::Tool, "give the `task` to delegate"))?;
            let outcome = self.agent.run(task).await?;
            let usage = outcome.usage();
            let tokens = usage.total();
            Ok(match outcome {
                AgentOutcome::Completed { text, .. } => {
                    serde_json::json!({"answer": text, "tokens": tokens})
                }
                AgentOutcome::AwaitingApproval { .. } => serde_json::json!({
                    "error": "the helper needs a person to approve an action; do it another way",
                }),
                other => serde_json::json!({
                    "error": "the helper stopped before it finished",
                    "outcome": other,
                }),
            })
        })
    }
}

#[cfg(test)]
mod tests;
