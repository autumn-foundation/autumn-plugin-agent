//! Provider-agnostic LLM agent harness for Autumn.
//!
//! `autumn-plugin-agent` equips an Autumn app with tool-calling agents that
//! speak to OpenAI-compatible endpoints (OpenAI, Ollama, vLLM)
//! or the Anthropic Messages API — no heavyweight LLM SDK required.
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
//! | [`agent`] | The agent loop, budgets, and the `agent_run` background job |
//! | [`plugin`] | `AgentPlugin` registration and the `AgentHandle` extractor |
//! | [`health`] | Provider health indicator for `/actuator/health` |

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
pub mod error;
pub mod health;
pub mod plugin;
pub mod tools;

pub use agent::{Agent, AgentOutcome, AgentRunArgs, AgentRuntime, BudgetKind, enqueue_agent_run};
pub use client::{
    AnthropicClient, ChatMessage, ChatRequest, ChatResponse, ChatRole, ContentPart, LlmClient,
    OpenAiCompatibleClient, StopReason, TokenUsage, ToolDefinition, client_from_config,
};
pub use config::{AgentConfig, ProviderKind};
pub use error::{AgentError, ErrorKind};
pub use health::AgentHealthIndicator;
pub use plugin::{AgentHandle, AgentPlugin};
pub use tools::{FnTool, Tool};
