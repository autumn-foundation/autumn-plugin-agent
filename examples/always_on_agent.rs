//! Always-on agent example.
//!
//! Shows the always-on building blocks with a scripted model, so it runs
//! with no API key and no network:
//!
//! 1. a chat session with persistent memory (the agent saves a preference);
//! 2. an approval gate (the agent wants to send an email; a person approves
//!    the paused run after a JSON round trip);
//! 3. two heartbeat ticks (one quiet `HEARTBEAT_OK`, one delivered report).
//!
//! ```sh
//! cargo run --example always_on_agent
//! ```
//!
//! In an Autumn app, the same pieces install through the plugin:
//!
//! ```rust,ignore
//! AgentPlugin::new()
//!     .memory_store(store)
//!     .policy(Arc::new(ToolRules::ask_before_acting()))
//!     .heartbeat(Heartbeat::every(Duration::from_secs(1_800)).session("ops"))
//!     .followups(Duration::from_secs(24 * 3_600))
//!     .delivery(my_delivery)
//! ```

use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};
use std::time::Duration;

use autumn_plugin_agent::{
    AgentConfig, AgentError, AgentOutcome, AgentRuntime, ApprovalDecision, ChatRequest,
    ChatResponse, ContentPart, Delivery, FnTool, Heartbeat, InMemoryMemoryStore, LlmClient,
    MemoryScope, Report, RunState, SessionId, StopReason, TokenUsage, ToolEffect, ToolRules,
};
use futures::future::BoxFuture;
use serde_json::json;

/// A model that plays back a fixed script, one response per call.
#[derive(Debug)]
struct ScriptedModel(Mutex<VecDeque<ChatResponse>>);

impl ScriptedModel {
    fn text(text: &str) -> ChatResponse {
        ChatResponse {
            content: vec![ContentPart::Text(text.to_owned())],
            stop_reason: StopReason::EndTurn,
            usage: TokenUsage::new(120, 20),
        }
    }

    fn call(id: &str, name: &str, arguments: serde_json::Value) -> ChatResponse {
        ChatResponse {
            content: vec![ContentPart::ToolCall {
                id: id.to_owned(),
                name: name.to_owned(),
                arguments,
            }],
            stop_reason: StopReason::ToolUse,
            usage: TokenUsage::new(120, 20),
        }
    }
}

impl LlmClient for ScriptedModel {
    fn chat<'a>(
        &'a self,
        _request: &'a ChatRequest,
    ) -> Pin<Box<dyn Future<Output = Result<ChatResponse, AgentError>> + Send + 'a>> {
        let next = self
            .0
            .lock()
            .ok()
            .and_then(|mut script| script.pop_front())
            .unwrap_or_else(|| Self::text("(script ended)"));
        Box::pin(async move { Ok(next) })
    }

    fn list_models(
        &self,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<String>, AgentError>> + Send + '_>> {
        Box::pin(async { Ok(vec!["scripted".to_owned()]) })
    }

    fn provider_name(&self) -> &'static str {
        "scripted"
    }
}

/// Prints reports. A real app sends mail, posts to chat, or fills an inbox.
#[derive(Debug)]
struct PrintDelivery;

impl Delivery for PrintDelivery {
    fn deliver<'a>(&'a self, report: &'a Report) -> BoxFuture<'a, Result<(), AgentError>> {
        println!("  [delivered from {:?}] {}", report.source, report.text);
        Box::pin(std::future::ready(Ok(())))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let model = Arc::new(ScriptedModel(Mutex::new(VecDeque::from([
        // 1. Chat: save a preference, then answer.
        ScriptedModel::call(
            "m1",
            "memory",
            json!({"action": "add", "block": "user", "text": "Prefers metric units."}),
        ),
        ScriptedModel::text("Got it: metric from now on."),
        // 2. Approval: the agent wants to email; then confirms.
        ScriptedModel::call(
            "e1",
            "send_email",
            json!({"to": "ops@example.com", "body": "Disk at 91%."}),
        ),
        ScriptedModel::text("I emailed ops about the disk."),
        // 3. Heartbeats: one quiet tick, one with news.
        ScriptedModel::text("HEARTBEAT_OK"),
        ScriptedModel::text("Disk usage reached 95%. Someone should look today."),
    ]))));

    let send_email = FnTool::new(
        "send_email",
        "Send an email.",
        json!({"type": "object", "properties": {"to": {"type": "string"}, "body": {"type": "string"}}}),
        |input: serde_json::Value| async move { Ok(json!({"sent_to": input["to"]})) },
    )
    .effect(ToolEffect::External)
    .shared();

    let runtime = AgentRuntime::new(AgentConfig::default(), model, vec![send_email])
        .with_memory(InMemoryMemoryStore::default().shared())
        .with_policy(Arc::new(ToolRules::ask_before_acting()))
        .with_delivery(Arc::new(PrintDelivery));
    let sessions = runtime.sessions();
    let session = SessionId::new("chat-with-mark");

    println!("1. Chat with memory");
    let turn = runtime
        .agent()
        .run_in_session(sessions.as_ref(), &session, "Please use metric units.")
        .await?;
    println!("  agent: {}", turn.outcome.text().unwrap_or_default());
    if let Some(memory) = runtime.memory() {
        let blocks = memory.load(&MemoryScope::agent()).await?;
        println!("  memory[user]: {:?}", blocks[1].entries);
    }

    println!("2. Approval gate");
    let turn = runtime
        .agent()
        .run_in_session(
            sessions.as_ref(),
            &session,
            "Tell ops the disk is filling up.",
        )
        .await?;
    let AgentOutcome::AwaitingApproval { state } = turn.outcome else {
        return Err("expected the run to pause for approval".into());
    };
    for pending in state.awaiting() {
        println!(
            "  needs approval: {} {}",
            pending.call.name, pending.call.arguments
        );
    }
    // Store the paused run anywhere JSON fits, then resume it later.
    let stored = serde_json::to_string(&*state)?;
    let state: RunState = serde_json::from_str(&stored)?;
    let call_id = state.pending[0].call.id.clone();
    let turn = runtime
        .agent()
        .resume_in_session(
            sessions.as_ref(),
            state,
            vec![ApprovalDecision::approve(call_id)],
        )
        .await?;
    println!("  agent: {}", turn.outcome.text().unwrap_or_default());

    println!("3. Heartbeats");
    let heartbeat = Heartbeat::every(Duration::from_secs(1_800)).session("ops");
    let app = autumn_web::AppState::detached();
    for tick in 1..=2 {
        let outcome = heartbeat.tick(&runtime, &app).await?;
        let text = outcome
            .as_ref()
            .and_then(AgentOutcome::text)
            .unwrap_or_default();
        println!("  tick {tick}: model said {text:?}");
    }
    Ok(())
}
