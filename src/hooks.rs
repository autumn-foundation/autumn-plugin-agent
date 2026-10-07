//! Lifecycle hooks: code that runs at fixed points of the agent loop.
//!
//! An [`AgentHooks`] implementation sees every model call, every tool call,
//! and the final outcome. Use hooks for audit logs, metrics, secret
//! redaction, progress reports, or custom guards. Every method has a no-op
//! default, so an implementation overrides only what it needs.
//!
//! Hooks run in registration order. A [`HookAction::Block`] from any
//! `before_tool` hook stops that call; the model sees the reason as the tool
//! result.

use futures::future::BoxFuture;

use crate::agent::AgentOutcome;
use crate::client::{ChatRequest, ChatResponse, TokenUsage};
use crate::ids::{RunId, SessionId};
use crate::tools::{ToolCall, ToolContext};

/// Read-only facts about the run a hook fires in.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RunInfo {
    /// The run's id.
    pub run_id: RunId,
    /// The session the run belongs to, if any.
    pub session_id: Option<SessionId>,
    /// Tool-execution rounds finished so far.
    pub steps_used: u32,
    /// The run's step budget.
    pub max_steps: u32,
    /// Provider-reported tokens spent so far.
    pub usage: TokenUsage,
}

/// What a `before_tool` hook wants the loop to do with a call.
// `Modify` carries a `serde_json::Value`, which has no `Eq` impl.
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Debug, Clone, PartialEq)]
pub enum HookAction {
    /// Run the call as the model asked.
    Continue,
    /// Run the call with these arguments instead.
    Modify(serde_json::Value),
    /// Do not run the call. The model sees the reason as an error result.
    Block(String),
}

/// The result of one tool call, as hooks see it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ToolOutput {
    /// The text the model receives as the tool result.
    pub content: String,
    /// `true` when the call failed, was blocked, or was denied.
    pub is_error: bool,
}

/// Code that runs at fixed points of the agent loop.
///
/// ```rust
/// use autumn_plugin_agent::hooks::{AgentHooks, HookAction, RunInfo};
/// use autumn_plugin_agent::tools::{ToolCall, ToolContext};
/// use futures::future::BoxFuture;
///
/// /// Never let the agent email anyone outside the company.
/// #[derive(Debug)]
/// struct InternalMailOnly;
///
/// impl AgentHooks for InternalMailOnly {
///     fn before_tool<'a>(
///         &'a self,
///         call: &'a ToolCall,
///         _ctx: &'a ToolContext,
///     ) -> BoxFuture<'a, HookAction> {
///         Box::pin(async move {
///             let to = call.arguments["to"].as_str().unwrap_or_default();
///             if call.name == "send_email" && !to.ends_with("@example.com") {
///                 HookAction::Block("external recipients are not allowed".into())
///             } else {
///                 HookAction::Continue
///             }
///         })
///     }
/// }
/// ```
pub trait AgentHooks: Send + Sync + std::fmt::Debug {
    /// Runs before each model call. May edit the request (for example, to
    /// redact text).
    fn before_model<'a>(
        &'a self,
        _request: &'a mut ChatRequest,
        _info: &'a RunInfo,
    ) -> BoxFuture<'a, ()> {
        Box::pin(std::future::ready(()))
    }

    /// Runs after each model call with the raw response.
    fn after_model<'a>(
        &'a self,
        _response: &'a ChatResponse,
        _info: &'a RunInfo,
    ) -> BoxFuture<'a, ()> {
        Box::pin(std::future::ready(()))
    }

    /// Runs before each tool call that the policy allowed.
    fn before_tool<'a>(
        &'a self,
        _call: &'a ToolCall,
        _ctx: &'a ToolContext,
    ) -> BoxFuture<'a, HookAction> {
        Box::pin(std::future::ready(HookAction::Continue))
    }

    /// Runs after each tool call, including blocked and denied ones.
    fn after_tool<'a>(
        &'a self,
        _call: &'a ToolCall,
        _output: &'a ToolOutput,
        _ctx: &'a ToolContext,
    ) -> BoxFuture<'a, ()> {
        Box::pin(std::future::ready(()))
    }

    /// Runs once when the run returns an outcome (including a pause for
    /// approval).
    fn on_outcome<'a>(
        &'a self,
        _outcome: &'a AgentOutcome,
        _info: &'a RunInfo,
    ) -> BoxFuture<'a, ()> {
        Box::pin(std::future::ready(()))
    }
}
