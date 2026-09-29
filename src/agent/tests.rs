#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::collections::VecDeque;
use std::future::Future;
use std::sync::Mutex;

use proptest::prelude::*;
use serde_json::json;

use super::*;
use crate::client::{ChatResponse, StopReason};
use crate::error::ErrorKind;
use crate::tools::FnTool;

/// A scripted [`LlmClient`]: pops one response per `chat` call.
#[derive(Debug)]
struct ScriptClient {
    responses: Mutex<VecDeque<ChatResponse>>,
}

impl ScriptClient {
    fn new(responses: Vec<ChatResponse>) -> Arc<Self> {
        Arc::new(Self {
            responses: Mutex::new(responses.into()),
        })
    }

    fn answer(text: &str) -> ChatResponse {
        ChatResponse {
            content: vec![ContentPart::Text(text.to_owned())],
            stop_reason: StopReason::EndTurn,
            usage: TokenUsage {
                input_tokens: 10,
                output_tokens: 5,
            },
        }
    }

    fn tool_call(id: &str, name: &str, arguments: serde_json::Value) -> ChatResponse {
        ChatResponse {
            content: vec![ContentPart::ToolCall {
                id: id.to_owned(),
                name: name.to_owned(),
                arguments,
            }],
            stop_reason: StopReason::ToolUse,
            usage: TokenUsage {
                input_tokens: 10,
                output_tokens: 5,
            },
        }
    }
}

impl LlmClient for ScriptClient {
    fn chat<'a>(
        &'a self,
        _request: &'a ChatRequest,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<ChatResponse, AgentError>> + Send + 'a>> {
        let next = self
            .responses
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or_else(|| Self::answer("default"));
        Box::pin(async move { Ok(next) })
    }

    fn list_models(
        &self,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<Vec<String>, AgentError>> + Send + '_>> {
        Box::pin(async move { Ok(vec!["script".to_owned()]) })
    }

    fn provider_name(&self) -> &'static str {
        "script"
    }
}

fn weather_tool() -> Arc<dyn Tool> {
    FnTool::new(
        "get_weather",
        "Current weather for a city.",
        json!({"type": "object"}),
        |input: serde_json::Value| async move {
            let city = input["city"].as_str().unwrap_or("?");
            Ok(json!({"city": city, "temp_f": 72}))
        },
    )
    .shared()
}

#[tokio::test]
async fn tool_call_round_trip_completes() {
    let client = ScriptClient::new(vec![
        ScriptClient::tool_call("call_1", "get_weather", json!({"city": "Chicago"})),
        ScriptClient::answer("Sunny, 72F in Chicago."),
    ]);
    let agent = Agent::new(client).tools(vec![weather_tool()]);
    let outcome = agent.run("Weather in Chicago?").await.unwrap();
    match outcome {
        AgentOutcome::Completed {
            text,
            steps_used,
            usage,
        } => {
            assert_eq!(text, "Sunny, 72F in Chicago.");
            assert_eq!(steps_used, 1);
            assert_eq!(usage.input_tokens, 20);
        }
        AgentOutcome::BudgetExhausted { .. } => panic!("expected completion"),
    }
}

#[tokio::test]
async fn step_budget_exhausts() {
    let client = ScriptClient::new(vec![
        ScriptClient::tool_call("c1", "get_weather", json!({"city": "Chicago"})),
        ScriptClient::tool_call("c2", "get_weather", json!({"city": "Chicago"})),
        ScriptClient::answer("never reached"),
    ]);
    let agent = Agent::new(client).tools(vec![weather_tool()]).max_steps(1);
    let outcome = agent.run("Weather?").await.unwrap();
    match outcome {
        AgentOutcome::BudgetExhausted {
            reason, steps_used, ..
        } => {
            assert_eq!(reason, BudgetKind::Steps);
            assert_eq!(steps_used, 1);
        }
        AgentOutcome::Completed { .. } => panic!("expected budget exhaustion"),
    }
}

#[tokio::test]
async fn token_budget_exhausts() {
    // Each scripted tool call burns 15 tokens; 100 of them overflow a
    // 600-token budget long before the step budget (1000) runs out.
    let responses: Vec<ChatResponse> = (0..100)
        .map(|i| ScriptClient::tool_call(&format!("c{i}"), "get_weather", json!({"city": "X"})))
        .collect();
    let client = ScriptClient::new(responses);
    let agent = Agent::new(client)
        .tools(vec![weather_tool()])
        .max_steps(1000)
        .max_tokens(600);
    let outcome = agent.run("go").await.unwrap();
    assert!(
        matches!(
            outcome,
            AgentOutcome::BudgetExhausted {
                reason: BudgetKind::Tokens,
                ..
            }
        ),
        "{outcome:?}"
    );
}

#[tokio::test]
async fn token_overshoot_on_final_answer_exhausts() {
    // The model answers directly, but the provider-reported usage blows past
    // the token budget: the run is exhausted, not completed.
    let response = ChatResponse {
        content: vec![ContentPart::Text("done".to_owned())],
        stop_reason: StopReason::EndTurn,
        usage: TokenUsage {
            input_tokens: 900,
            output_tokens: 200,
        },
    };
    let client = ScriptClient::new(vec![response]);
    let agent = Agent::new(client).max_tokens(1000);
    let outcome = agent.run("go").await.unwrap();
    match outcome {
        AgentOutcome::BudgetExhausted { reason, usage, .. } => {
            assert_eq!(reason, BudgetKind::Tokens);
            assert_eq!(usage.total(), 1100);
        }
        AgentOutcome::Completed { .. } => panic!("expected exhaustion on token overshoot"),
    }
}

#[tokio::test]
async fn unknown_tool_reaches_the_model() {
    let client = ScriptClient::new(vec![
        ScriptClient::tool_call("c1", "no_such_tool", json!({})),
        ScriptClient::answer("recovered"),
    ]);
    // Capture the tool-result message the loop built.
    let seen = Arc::new(Mutex::new(Vec::<String>::new()));
    let seen_clone = Arc::clone(&seen);
    let spy: Arc<dyn LlmClient> = Arc::new(SpyClient {
        inner: client,
        seen: seen_clone,
    });
    let agent = Agent::new(spy).tools(vec![weather_tool()]);
    let outcome = agent.run("go").await.unwrap();
    assert!(outcome.is_completed());
    let seen = seen.lock().unwrap();
    assert_eq!(seen.len(), 1);
    assert!(seen[0].contains("unknown tool"), "{}", seen[0]);
}

/// Forwards to an inner client while recording tool-result payloads.
#[derive(Debug)]
struct SpyClient {
    inner: Arc<ScriptClient>,
    seen: Arc<Mutex<Vec<String>>>,
}

impl LlmClient for SpyClient {
    fn chat<'a>(
        &'a self,
        request: &'a ChatRequest,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<ChatResponse, AgentError>> + Send + 'a>> {
        for message in &request.messages {
            for part in &message.content {
                if let ContentPart::ToolResult { content, .. } = part {
                    self.seen.lock().unwrap().push(content.clone());
                }
            }
        }
        self.inner.chat(request)
    }

    fn list_models(
        &self,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<Vec<String>, AgentError>> + Send + '_>> {
        self.inner.list_models()
    }

    fn provider_name(&self) -> &'static str {
        "spy"
    }
}

#[tokio::test]
async fn tool_failure_reaches_the_model() {
    let failing: Arc<dyn Tool> = FnTool::new(
        "broken",
        "Always fails.",
        json!({"type": "object"}),
        |_input: serde_json::Value| async move { Err(AgentError::new(ErrorKind::Tool, "kaput")) },
    )
    .shared();
    let client = ScriptClient::new(vec![
        ScriptClient::tool_call("c1", "broken", json!({})),
        ScriptClient::answer("recovered"),
    ]);
    let seen = Arc::new(Mutex::new(Vec::<String>::new()));
    let spy: Arc<dyn LlmClient> = Arc::new(SpyClient {
        inner: client,
        seen: Arc::clone(&seen),
    });
    let agent = Agent::new(spy).tools(vec![failing]);
    let outcome = agent.run("go").await.unwrap();
    assert!(outcome.is_completed());
    let seen = seen.lock().unwrap();
    assert!(seen[0].contains("kaput"), "{}", seen[0]);
}

#[tokio::test]
async fn tool_output_truncation_applies() {
    let chatty: Arc<dyn Tool> = FnTool::new(
        "chatty",
        "Talks too much.",
        json!({"type": "object"}),
        |_input: serde_json::Value| async move { Ok(json!({"text": "x".repeat(100)})) },
    )
    .shared();
    let client = ScriptClient::new(vec![
        ScriptClient::tool_call("c1", "chatty", json!({})),
        ScriptClient::answer("done"),
    ]);
    let seen = Arc::new(Mutex::new(Vec::<String>::new()));
    let spy: Arc<dyn LlmClient> = Arc::new(SpyClient {
        inner: client,
        seen: Arc::clone(&seen),
    });
    let agent = Agent::new(spy)
        .tools(vec![chatty])
        .per_tool_output_limit(10);
    agent.run("go").await.unwrap();
    let seen = seen.lock().unwrap();
    assert!(seen[0].contains("[truncated]"), "{}", seen[0]);
    assert!(seen[0].chars().count() <= 40, "{}", seen[0].len());
}

#[test]
fn truncate_chars_marks_the_cut() {
    assert_eq!(truncate_chars("abc", 10), "abc");
    assert_eq!(truncate_chars("abcdef", 3), "abc…[truncated]");
}

#[test]
fn estimate_tokens_scales_with_length() {
    assert_eq!(estimate_tokens(""), 0);
    assert_eq!(estimate_tokens("abcd"), 1);
    assert!(estimate_tokens(&"x".repeat(400)) >= 90);
}

// --- proptest: truncation invariants ---

fn arb_text() -> impl Strategy<Value = String> {
    proptest::string::string_regex("[a-z ]{0,40}").unwrap()
}

fn arb_message() -> impl Strategy<Value = ChatMessage> {
    (
        prop_oneof![
            Just(ChatRole::System),
            Just(ChatRole::User),
            Just(ChatRole::Assistant),
            Just(ChatRole::Tool),
        ],
        arb_text(),
    )
        .prop_map(|(role, text)| ChatMessage::text(role, text))
}

proptest! {
    #[test]
    fn truncation_never_exceeds_budget(
        messages in proptest::collection::vec(arb_message(), 0..12),
        budget in 64u32..4096,
    ) {
        let out = truncate_history(&messages, budget);
        // System messages always survive, in order; the rest is a subsequence.
        let mut cursor = 0;
        for kept in &out {
            let pos = messages[cursor..].iter().position(|m| m == kept);
            prop_assert!(pos.is_some(), "kept message not found in input");
            cursor += pos.unwrap() + 1;
        }
        for message in &messages {
            if message.role == ChatRole::System {
                prop_assert!(out.contains(message), "system message dropped");
            }
        }
        let system_cost: u32 = messages.iter()
            .filter(|m| m.role == ChatRole::System)
            .map(|m| estimate_messages(std::slice::from_ref(m)))
            .sum();
        if system_cost <= budget {
            prop_assert!(
                estimate_messages(&out) <= budget,
                "estimate {} exceeds budget {budget}",
                estimate_messages(&out),
            );
        }
    }

    #[test]
    fn loop_never_exceeds_step_budget(
        max_steps in 1u32..8,
        tool_calls in 0usize..10,
    ) {
        let responses: Vec<ChatResponse> = (0..tool_calls)
            .map(|i| ScriptClient::tool_call(
                &format!("c{i}"),
                "get_weather",
                json!({"city": "X"}),
            ))
            .chain(std::iter::once(ScriptClient::answer("done")))
            .collect();
        let client = ScriptClient::new(responses);
        let agent = Agent::new(client)
            .tools(vec![weather_tool()])
            .max_steps(max_steps);
        let rt = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .unwrap();
        let outcome = rt.block_on(agent.run("go")).unwrap();
        prop_assert!(outcome.steps_used() <= max_steps,
            "used {} steps with budget {max_steps}", outcome.steps_used());
        if tool_calls >= max_steps as usize {
            let exhausted = matches!(
                outcome,
                AgentOutcome::BudgetExhausted {
                    reason: BudgetKind::Steps,
                    ..
                }
            );
            prop_assert!(exhausted, "expected step-budget exhaustion");
        }
    }
}
