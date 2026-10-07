//! Tools: the functions an agent may call.
//!
//! A [`Tool`] has a name, a description, a JSON input schema, and an async
//! `execute`. The agent loop hands the model the [`ToolDefinition`]s and runs
//! the matching tool when the model emits a call.
//!
//! ## Adapting Autumn handlers
//!
//! Autumn route handlers run inside the request lifecycle with extractors;
//! agent tools run in background jobs with no request in scope, so a tool
//! cannot *be* a handler. The native pattern is one canonical logic function
//! shaped `async fn(serde_json::Value) -> Result<serde_json::Value,
//! AgentError>`: the `#[get]` handler calls it with the request body, and
//! [`FnTool::new`] hands the same function to the agent. See
//! `examples/weather_agent.rs`.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::client::ToolDefinition;
use crate::error::AgentError;
use crate::ids::{RunId, SessionId};

/// What a tool can change. Policies use it to gate calls.
///
/// The variants are ordered from least to most impact, so
/// `effect <= ToolEffect::Internal` reads as "safe for an unattended run".
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ToolEffect {
    /// Reads data. Changes nothing.
    ReadOnly,
    /// Changes only the agent's own private state (memory, notes, its own
    /// follow-ups). Nobody else sees the change.
    Internal,
    /// Changes app data other people can see.
    Write,
    /// Acts outside the app: sends mail, calls a third-party API, spends
    /// money.
    External,
}

/// One tool call the model asked for.
// `arguments` is a `serde_json::Value`, which has no `Eq` impl.
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ToolCall {
    /// Provider-assigned call id.
    pub id: String,
    /// Name of the tool to run.
    pub name: String,
    /// Decoded JSON arguments.
    pub arguments: serde_json::Value,
}

/// Per-call context the agent loop hands to [`Tool::execute`].
///
/// Use `run_id` + `call_id` as an idempotency key for tools with side
/// effects: a retried job re-runs the loop, and the key lets the tool detect
/// a duplicate.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolContext {
    /// The run that issued the call.
    pub run_id: RunId,
    /// The provider-assigned id of this tool call.
    pub call_id: String,
    /// The session the run belongs to, if any.
    pub session_id: Option<SessionId>,
    /// The loop step (0-based) that issued the call.
    pub step: u32,
}

impl ToolContext {
    /// A context for calling a tool outside an agent run (tests, handlers).
    #[must_use]
    pub fn detached(call_id: impl Into<String>) -> Self {
        Self {
            run_id: RunId::new("detached"),
            call_id: call_id.into(),
            session_id: None,
            step: 0,
        }
    }
}

/// A function the agent may call.
///
/// Implementors stay `Send + Sync + 'static` because tools live in the shared
/// [`AgentRuntime`](crate::agent::AgentRuntime) and execute on job workers.
pub trait Tool: Send + Sync + std::fmt::Debug {
    /// Tool name as the model calls it. Keep it `snake_case`.
    fn name(&self) -> &str;

    /// What the tool does, shown to the model.
    fn description(&self) -> &str;

    /// JSON Schema for the tool's input object.
    fn input_schema(&self) -> serde_json::Value;

    /// Run the tool against decoded JSON arguments.
    fn execute<'a>(
        &'a self,
        input: serde_json::Value,
        ctx: &'a ToolContext,
    ) -> Pin<Box<dyn Future<Output = Result<serde_json::Value, AgentError>> + Send + 'a>>;

    /// What the tool can change. Defaults to [`ToolEffect::Write`]: a tool
    /// is trusted with less only when it says so.
    fn effect(&self) -> ToolEffect {
        ToolEffect::Write
    }

    /// Bundle the metadata the model sees into a [`ToolDefinition`].
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: self.name().to_owned(),
            description: self.description().to_owned(),
            input_schema: self.input_schema(),
        }
    }
}

/// The boxed closure behind [`FnTool`]: JSON and context in, JSON or an
/// [`AgentError`] out.
type ToolFn = Box<
    dyn Fn(
            serde_json::Value,
            ToolContext,
        ) -> Pin<Box<dyn Future<Output = Result<serde_json::Value, AgentError>> + Send>>
        + Send
        + Sync,
>;

/// A [`Tool`] built from a plain async function.
///
/// Use this to adapt an Autumn handler's core logic:
///
/// ```rust,no_run
/// # use autumn_plugin_agent::{AgentError, FnTool};
/// # use serde_json::{Value, json};
/// async fn get_weather(input: Value) -> Result<Value, AgentError> {
///     let city = input["city"].as_str().unwrap_or("unknown");
///     Ok(json!({"city": city, "temp_f": 72}))
/// }
///
/// let tool = FnTool::new(
///     "get_weather",
///     "Current weather for a city.",
///     serde_json::json!({
///         "type": "object",
///         "properties": {"city": {"type": "string"}},
///         "required": ["city"],
///     }),
///     get_weather,
/// );
/// ```
pub struct FnTool {
    name: String,
    description: String,
    schema: serde_json::Value,
    effect: ToolEffect,
    run: ToolFn,
}

impl std::fmt::Debug for FnTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FnTool")
            .field("name", &self.name)
            .field("description", &self.description)
            .field("effect", &self.effect)
            .finish_non_exhaustive()
    }
}

impl FnTool {
    /// Wrap an async function as a tool.
    ///
    /// The function receives the decoded JSON arguments and returns JSON.
    /// Return [`ErrorKind::Tool`](crate::error::ErrorKind::Tool) errors for domain failures so the agent
    /// loop reports them back to the model instead of aborting the run.
    pub fn new<F, Fut>(
        name: impl Into<String>,
        description: impl Into<String>,
        schema: serde_json::Value,
        run: F,
    ) -> Self
    where
        F: Fn(serde_json::Value) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<serde_json::Value, AgentError>> + Send + 'static,
    {
        Self {
            name: name.into(),
            description: description.into(),
            schema,
            effect: ToolEffect::Write,
            run: Box::new(move |input, _ctx| Box::pin(run(input))),
        }
    }

    /// Wrap an async function that also wants the [`ToolContext`].
    ///
    /// Use it when the tool needs the run id or call id, for example as an
    /// idempotency key.
    pub fn with_context<F, Fut>(
        name: impl Into<String>,
        description: impl Into<String>,
        schema: serde_json::Value,
        run: F,
    ) -> Self
    where
        F: Fn(serde_json::Value, ToolContext) -> Fut + Send + Sync + 'static,
        Fut: Future<Output = Result<serde_json::Value, AgentError>> + Send + 'static,
    {
        Self {
            name: name.into(),
            description: description.into(),
            schema,
            effect: ToolEffect::Write,
            run: Box::new(move |input, ctx| Box::pin(run(input, ctx))),
        }
    }

    /// Declare what the tool can change. The default is
    /// [`ToolEffect::Write`].
    #[must_use]
    pub const fn effect(mut self, effect: ToolEffect) -> Self {
        self.effect = effect;
        self
    }

    /// Share the tool between the plugin registry and direct use.
    #[must_use]
    pub fn shared(self) -> Arc<dyn Tool> {
        Arc::new(self)
    }
}

impl Tool for FnTool {
    fn name(&self) -> &str {
        &self.name
    }

    fn description(&self) -> &str {
        &self.description
    }

    fn input_schema(&self) -> serde_json::Value {
        self.schema.clone()
    }

    fn execute<'a>(
        &'a self,
        input: serde_json::Value,
        ctx: &'a ToolContext,
    ) -> Pin<Box<dyn Future<Output = Result<serde_json::Value, AgentError>> + Send + 'a>> {
        (self.run)(input, ctx.clone())
    }

    fn effect(&self) -> ToolEffect {
        self.effect
    }
}

#[cfg(test)]
mod tests;
