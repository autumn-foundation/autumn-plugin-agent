//! Provider-agnostic, always-on LLM agent harness for Autumn.
//!
//! `autumn-plugin-agent` equips an Autumn app with tool-calling agents that
//! speak to OpenAI-compatible endpoints (OpenAI, Ollama, vLLM)
//! or the Anthropic Messages API — no heavyweight LLM SDK required.
//!
//! Beyond one-shot runs, the crate has the primitives always-on agents
//! share: persisted sessions with compaction, bounded memory, skills,
//! approval gates with resumable state, lifecycle hooks, a loop guard,
//! heartbeats on Autumn's scheduler, agent-scheduled follow-ups, and a
//! delivery channel.
//!
//! # Quick start
//!
//! ```rust,ignore
//! use autumn_plugin_agent::plugin::{AgentHandle, AgentPlugin};
//! use autumn_plugin_agent::{FnTool, AgentError};
//! use autumn_web::prelude::*;
//! use std::sync::Arc;
//!
//! async fn get_weather(input: serde_json::Value) -> Result<serde_json::Value, AgentError> {
//!     Ok(serde_json::json!({"temp_f": 72}))
//! }
//!
//! #[autumn_web::main]
//! async fn main() {
//!     let weather = FnTool::new(
//!         "get_weather",
//!         "Current weather for a city.",
//!         serde_json::json!({"type": "object",
//!             "properties": {"city": {"type": "string"}},
//!             "required": ["city"]}),
//!         get_weather,
//!     )
//!     .shared();
//!
//!     autumn_web::app()
//!         .plugin(AgentPlugin::new().tool(Arc::clone(&weather)))
//!         .run()
//!         .await;
//! }
//! ```
//!
//! ```toml
//! # autumn.toml
//! [agent]
//! provider = "openai-compatible"
//! model = "gpt-4o-mini"
//! ```
//!
//! ```sh
//! export AGENT_API_KEY="sk-..."
//! ```
//!
//! # Module map
//!
//! | Module | Contents |
//! |--------|----------|
//! | [`config`] | Layered `[agent]` configuration; secrets stay in the env |
//! | [`error`] | `AgentError` + `ErrorKind` + HTTP status mapping |
//! | [`client`] | `LlmClient` trait and both provider implementations |
//! | [`tools`] | `Tool` trait and the handler-function adapter |
//! | [`agent`] | The agent loop, budgets, approvals, and `AgentRuntime` |
//! | [`hooks`] | Lifecycle hooks around model and tool calls |
//! | [`policy`] | Per-call allow / ask / deny decisions |
//! | [`loop_guard`] | Repeated-call detection |
//! | [`session`] | Persisted transcripts and compaction |
//! | [`memory`] | Bounded memory blocks and the `memory` tool |
//! | [`skills`] | `SKILL.md` skills and the `load_skill` tool |
//! | [`delegate`] | Subagents as tools |
//! | [`proactive`] | Heartbeats, follow-ups, and delivery |
//! | `jobs` | The `agent_run` / `agent_resume` background jobs |
//! | [`ids`] | `RunId` and `SessionId` |
//! | `plugin` | `AgentPlugin` registration and the `AgentHandle` extractor |
//! | `health` | Provider health indicator for `/actuator/health` |
//!
//! # Features
//!
//! `autumn` (default) adds `plugin`, `jobs`, `health`, the heartbeat
//! and follow-ups. It needs `autumn-web`. Without it, the agent core builds
//! with no `autumn-web` dependency.

#![forbid(unsafe_code)]
#![deny(
    clippy::unwrap_used,
    clippy::expect_used,
    clippy::panic,
    clippy::todo,
    clippy::unimplemented
)]

pub mod agent;
pub mod client;
pub mod config;
pub mod delegate;
pub mod error;
#[cfg(feature = "autumn")]
pub mod health;
pub mod hooks;
pub mod ids;
#[cfg(feature = "autumn")]
pub mod jobs;
pub mod loop_guard;
pub mod memory;
#[cfg(feature = "autumn")]
pub mod plugin;
pub mod policy;
pub mod proactive;
pub mod session;
pub mod skills;
pub mod tools;

#[cfg(test)]
mod test_support;

pub use agent::{
    Agent, AgentOutcome, AgentRuntime, AgentTurn, Approval, ApprovalDecision, BudgetKind,
    PendingCall, PendingStatus, RunState,
};
pub use client::{
    AnthropicClient, ChatMessage, ChatRequest, ChatResponse, ChatRole, ContentPart, LlmClient,
    OpenAiCompatibleClient, StopReason, TokenUsage, ToolDefinition, client_from_config,
};
pub use config::{AgentConfig, ProviderKind};
pub use delegate::AgentTool;
pub use error::{AgentError, ErrorKind};
#[cfg(feature = "autumn")]
pub use health::AgentHealthIndicator;
pub use hooks::{AgentHooks, HookAction, RunInfo, ToolOutput};
pub use ids::{RunId, SessionId};
#[cfg(feature = "autumn")]
pub use jobs::{
    AgentResumeArgs, AgentRunArgs, RunOrigin, enqueue_agent_resume, enqueue_agent_resume_tracked,
    enqueue_agent_run, enqueue_agent_run_in, enqueue_agent_run_tracked,
};
pub use loop_guard::LoopGuard;
pub use memory::{InMemoryMemoryStore, MemoryBlock, MemoryOp, MemoryScope, MemoryStore};
#[cfg(feature = "autumn")]
pub use plugin::{AgentHandle, AgentPlugin};
pub use policy::{AllowAll, Rule, Strictest, ToolDecision, ToolPolicy, ToolRules};
#[cfg(feature = "autumn")]
pub use proactive::Heartbeat;
pub use proactive::{Delivery, HEARTBEAT_OK, LogDelivery, Report, ReportSource};
pub use session::{Compaction, InMemorySessionStore, SessionStore};
pub use skills::Skill;
pub use tools::{FnTool, Tool, ToolCall, ToolContext, ToolEffect};
