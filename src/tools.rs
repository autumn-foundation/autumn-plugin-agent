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

use crate::client::ToolDefinition;
use crate::error::AgentError;

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
    fn execute(
        &self,
        input: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<serde_json::Value, AgentError>> + Send + '_>>;

    /// Bundle the metadata the model sees into a [`ToolDefinition`].
    fn definition(&self) -> ToolDefinition {
        ToolDefinition {
            name: self.name().to_owned(),
            description: self.description().to_owned(),
            input_schema: self.input_schema(),
        }
    }
}

/// The boxed closure behind [`FnTool`]: JSON in, JSON or an [`AgentError`] out.
type ToolFn = Box<
    dyn Fn(
            serde_json::Value,
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
    run: ToolFn,
}

impl std::fmt::Debug for FnTool {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("FnTool")
            .field("name", &self.name)
            .field("description", &self.description)
            .finish_non_exhaustive()
    }
}

impl FnTool {
    /// Wrap an async function as a tool.
    ///
    /// The function receives the decoded JSON arguments and returns JSON.
    /// Return [`ErrorKind::Tool`] errors for domain failures so the agent
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
            run: Box::new(move |input| Box::pin(run(input))),
        }
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

    fn execute(
        &self,
        input: serde_json::Value,
    ) -> Pin<Box<dyn Future<Output = Result<serde_json::Value, AgentError>> + Send + '_>> {
        (self.run)(input)
    }
}

#[cfg(test)]
mod tests;
