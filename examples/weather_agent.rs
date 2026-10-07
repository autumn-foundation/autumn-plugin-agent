//! Weather agent example.
//!
//! Runs an agent equipped with a `get_weather` tool against a scripted
//! OpenAI-compatible mock server, so the example works with no API key and
//! no network beyond localhost.
//!
//! ```sh
//! cargo run --example weather_agent
//! ```
//!
//! Point it at a real provider instead:
//!
//! ```sh
//! # OpenAI (or any OpenAI-compatible endpoint: Ollama, vLLM, Azure):
//! export AGENT_API_KEY="sk-..."
//! # provider/model/base_url come from autumn.toml [agent] or AGENT_* env:
//! export AGENT_MODEL="gpt-4o-mini"
//!
//! # Anthropic:
//! export AGENT_PROVIDER="anthropic"
//! export AGENT_MODEL="claude-sonnet-4-5"
//! ```
//!
//! With a real provider, replace `mock_client` below with
//! `autumn_plugin_agent::client_from_config(&AgentConfig::load()?)`.

use std::sync::Arc;

use autumn_plugin_agent::{
    Agent, AgentError, AgentOutcome, FnTool, LlmClient, OpenAiCompatibleClient, Tool,
};
use axum::routing::post;
use axum::{Json, Router};
use serde_json::{Value, json};

/// The tool's canonical logic function. In a real Autumn app this same
/// function backs both the `#[get]` handler and the agent tool.
async fn get_weather(input: Value) -> Result<Value, AgentError> {
    let city = input["city"].as_str().unwrap_or("unknown");
    // A real implementation would call a weather API here.
    Ok(json!({
        "city": city,
        "temp_f": 72,
        "conditions": "sunny",
    }))
}

/// Scripted mock: first turn requests the tool, second turn answers.
async fn mock_chat(Json(body): Json<Value>) -> Json<Value> {
    let already_called = body["messages"].as_array().is_some_and(|messages| {
        messages
            .iter()
            .any(|m| m["role"] == Value::String("tool".to_owned()))
    });
    if already_called {
        Json(json!({
            "choices": [{
                "message": {"role": "assistant",
                            "content": "It is sunny and 72F in Chicago."},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 50, "completion_tokens": 12}
        }))
    } else {
        Json(json!({
            "choices": [{
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{\"city\":\"Chicago\"}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 40, "completion_tokens": 10}
        }))
    }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await?;
    let addr = listener.local_addr()?;
    tokio::spawn(async move {
        let router = Router::new().route("/chat/completions", post(mock_chat));
        if let Err(err) = axum::serve(listener, router).await {
            eprintln!("mock server failed: {err}");
        }
    });

    let client: Arc<dyn LlmClient> = Arc::new(OpenAiCompatibleClient::new(
        format!("http://{addr}"),
        "mock-model",
        "not-a-real-key",
    )?);
    let weather: Arc<dyn Tool> = FnTool::new(
        "get_weather",
        "Current weather for a city.",
        json!({
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        }),
        get_weather,
    )
    .shared();

    let agent = Agent::new(client)
        .tools(vec![weather])
        .system_prompt("You are a concise weather assistant.")
        .max_steps(5);

    match agent.run("What is the weather in Chicago?").await? {
        AgentOutcome::Completed {
            text, steps_used, ..
        } => {
            println!("answer ({steps_used} step(s)): {text}");
        }
        AgentOutcome::BudgetExhausted { reason, .. } => {
            println!("stopped early: {reason:?}");
        }
        other => println!("stopped: {other:?}"),
    }
    Ok(())
}
