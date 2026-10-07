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
    requests: Mutex<Vec<ChatRequest>>,
}

impl ScriptClient {
    fn new(responses: Vec<ChatResponse>) -> Arc<Self> {
        Arc::new(Self {
            responses: Mutex::new(responses.into()),
            requests: Mutex::new(Vec::new()),
        })
    }

    fn requests(&self) -> Vec<ChatRequest> {
        self.requests.lock().unwrap().clone()
    }

    fn calls(id_names: &[(&str, &str)]) -> ChatResponse {
        ChatResponse {
            content: id_names
                .iter()
                .map(|(id, name)| ContentPart::ToolCall {
                    id: (*id).to_owned(),
                    name: (*name).to_owned(),
                    arguments: json!({"city": *id}),
                })
                .collect(),
            stop_reason: StopReason::ToolUse,
            usage: TokenUsage::new(10, 5),
        }
    }

    fn answer(text: &str) -> ChatResponse {
        ChatResponse {
            content: vec![ContentPart::Text(text.to_owned())],
            stop_reason: StopReason::EndTurn,
            usage: TokenUsage::new(10, 5),
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
            usage: TokenUsage::new(10, 5),
        }
    }
}

impl LlmClient for ScriptClient {
    fn chat<'a>(
        &'a self,
        request: &'a ChatRequest,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<ChatResponse, AgentError>> + Send + 'a>> {
        self.requests.lock().unwrap().push(request.clone());
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
        other => panic!("expected completion, got {other:?}"),
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
        other => panic!("expected budget exhaustion, got {other:?}"),
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
        .max_tokens(600)
        .loop_guard(LoopGuard::disabled());
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
        usage: TokenUsage::new(900, 200),
    };
    let client = ScriptClient::new(vec![response]);
    let agent = Agent::new(client).max_tokens(1000);
    let outcome = agent.run("go").await.unwrap();
    match outcome {
        AgentOutcome::BudgetExhausted { reason, usage, .. } => {
            assert_eq!(reason, BudgetKind::Tokens);
            assert_eq!(usage.total(), 1100);
        }
        other => panic!("expected exhaustion on token overshoot, got {other:?}"),
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
    fn truncation_never_orphans_tool_results(
        messages in proptest::collection::vec(arb_message(), 0..12),
        budget in 0u32..200,
    ) {
        let out = truncate_history(&messages, budget);
        let first_turn = out.iter().find(|m| m.role != ChatRole::System);
        prop_assert!(first_turn.is_none_or(|m| m.role != ChatRole::Tool));
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
            .max_steps(max_steps)
            .loop_guard(LoopGuard::disabled());
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

// --- multi-turn, approvals, hooks, guards, sessions, memory, skills ---

use crate::hooks::{AgentHooks, HookAction, RunInfo, ToolOutput};
use crate::memory::{InMemoryMemoryStore, MemoryScope};
use crate::policy::{Rule, ToolRules};
use crate::session::{InMemorySessionStore, SUMMARY_PREFIX, SessionStore};
use crate::skills::Skill;
use crate::tools::{ToolCall, ToolContext, ToolEffect};
use futures::future::BoxFuture;

/// A tool that counts calls and records the arguments it saw.
fn counting_tool(name: &str, seen: Arc<Mutex<Vec<serde_json::Value>>>) -> Arc<dyn Tool> {
    FnTool::new(
        name,
        "Counts calls.",
        json!({"type": "object"}),
        move |input: serde_json::Value| {
            let seen = Arc::clone(&seen);
            async move {
                seen.lock().unwrap().push(input.clone());
                Ok(json!({"ok": input}))
            }
        },
    )
    .effect(ToolEffect::External)
    .shared()
}

/// Every tool-result payload in a request, in order.
fn tool_results(request: &ChatRequest) -> Vec<String> {
    request
        .messages
        .iter()
        .flat_map(|message| message.content.iter())
        .filter_map(|part| match part {
            ContentPart::ToolResult { content, .. } => Some(content.clone()),
            _ => None,
        })
        .collect()
}

fn system_text(request: &ChatRequest) -> String {
    request
        .messages
        .iter()
        .filter(|message| message.role == ChatRole::System)
        .flat_map(|message| message.content.iter())
        .filter_map(|part| match part {
            ContentPart::Text(text) => Some(text.clone()),
            _ => None,
        })
        .collect()
}

fn paused(outcome: AgentOutcome) -> RunState {
    match outcome {
        AgentOutcome::AwaitingApproval { state } => *state,
        other => panic!("expected a pause, got {other:?}"),
    }
}

#[tokio::test]
async fn run_turn_continues_a_transcript() {
    let client = ScriptClient::new(vec![
        ScriptClient::answer("Hello, Mark."),
        ScriptClient::answer("You said your name is Mark."),
    ]);
    let agent = Agent::new(client.clone()).system_prompt("Be brief.");
    let first = agent.run_turn(Vec::new(), "I am Mark.").await.unwrap();
    assert_eq!(first.messages.len(), 2);
    assert!(first.messages.iter().all(|m| m.role != ChatRole::System));
    let second = agent
        .run_turn(first.messages.clone(), "What is my name?")
        .await
        .unwrap();
    assert_eq!(second.outcome.text(), Some("You said your name is Mark."));
    assert_eq!(second.messages.len(), 4);
    let request = &client.requests()[1];
    // system + user + assistant + user
    assert_eq!(request.messages.len(), 4);
    assert_eq!(request.messages[0].role, ChatRole::System);
    assert_eq!(request.messages[2].role, ChatRole::Assistant);
}

#[tokio::test]
async fn approval_pauses_then_resumes_on_approve() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let client = ScriptClient::new(vec![
        ScriptClient::tool_call("c1", "send_email", json!({"to": "a@b.c"})),
        ScriptClient::answer("Sent."),
    ]);
    let agent = Agent::new(client.clone())
        .tool(counting_tool("send_email", Arc::clone(&seen)))
        .policy(Arc::new(ToolRules::ask_before_acting()));
    let state = paused(agent.run("email a@b.c").await.unwrap());
    assert!(
        seen.lock().unwrap().is_empty(),
        "nothing runs before approval"
    );
    assert_eq!(state.steps_used, 1);
    assert_eq!(state.awaiting().count(), 1);
    assert_eq!(state.awaiting().next().unwrap().call.id, "c1");

    // The state survives a JSON round trip (a database row).
    let stored = serde_json::to_string(&state).unwrap();
    let state: RunState = serde_json::from_str(&stored).unwrap();

    let turn = agent
        .resume(state.clone(), vec![ApprovalDecision::approve("c1")])
        .await
        .unwrap();
    assert_eq!(turn.outcome.text(), Some("Sent."));
    assert_eq!(seen.lock().unwrap().len(), 1);
    assert_eq!(turn.outcome.steps_used(), 1);
    assert_eq!(turn.messages.last().unwrap().role, ChatRole::Assistant);
    let results = tool_results(client.requests().last().unwrap());
    assert_eq!(results.len(), 1);
    assert!(results[0].contains("a@b.c"));
}

#[tokio::test]
async fn rejected_edited_and_undecided_calls() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let client = ScriptClient::new(vec![
        ScriptClient::calls(&[("a", "act"), ("b", "act"), ("c", "act"), ("d", "read")]),
        ScriptClient::answer("done"),
    ]);
    let read = FnTool::new("read", "Reads.", json!({}), |_: serde_json::Value| async {
        Ok(json!("data"))
    })
    .effect(ToolEffect::ReadOnly)
    .shared();
    let agent = Agent::new(client.clone())
        .tool(counting_tool("act", Arc::clone(&seen)))
        .tool(read)
        .policy(Arc::new(ToolRules::ask_before_acting()));
    let state = paused(agent.run("go").await.unwrap());
    assert_eq!(state.awaiting().count(), 3, "the read call is pre-allowed");
    let turn = agent
        .resume(
            state,
            vec![
                ApprovalDecision::reject("a", "not today"),
                ApprovalDecision {
                    call_id: "b".to_owned(),
                    approval: Approval::Edit {
                        arguments: json!({"city": "Oslo"}),
                    },
                },
            ],
        )
        .await
        .unwrap();
    assert!(turn.outcome.is_completed());
    assert_eq!(*seen.lock().unwrap(), vec![json!({"city": "Oslo"})]);
    let results = tool_results(client.requests().last().unwrap());
    assert!(
        results[0].contains("rejected this call: not today"),
        "{}",
        results[0]
    );
    assert!(results[1].contains("Oslo"), "{}", results[1]);
    assert!(
        results[2].contains("no decision was given"),
        "{}",
        results[2]
    );
    assert_eq!(results[3], "\"data\"");
}

#[tokio::test]
async fn resume_rejects_unknown_call_ids() {
    let client = ScriptClient::new(vec![ScriptClient::tool_call("c1", "act", json!({}))]);
    let agent = Agent::new(client)
        .tool(counting_tool("act", Arc::default()))
        .policy(Arc::new(ToolRules::new().tool("act", Rule::Ask)));
    let state = paused(agent.run("go").await.unwrap());
    let err = agent
        .resume(state, vec![ApprovalDecision::approve("nope")])
        .await
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::Config);
}

#[tokio::test]
async fn denied_calls_report_the_reason_and_the_run_continues() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let client = ScriptClient::new(vec![
        ScriptClient::tool_call("c1", "act", json!({})),
        ScriptClient::answer("ok, I won't"),
    ]);
    let agent = Agent::new(client.clone())
        .tool(counting_tool("act", Arc::clone(&seen)))
        .policy(Arc::new(ToolRules::read_only()));
    let outcome = agent.run("go").await.unwrap();
    assert!(outcome.is_completed());
    assert!(seen.lock().unwrap().is_empty());
    let results = tool_results(client.requests().last().unwrap());
    assert!(results[0].contains("denied by policy"), "{}", results[0]);
}

/// Records every hook event and blocks or rewrites chosen tools.
#[derive(Debug, Default)]
struct Recorder {
    events: Mutex<Vec<String>>,
}

impl AgentHooks for Recorder {
    fn before_model<'a>(
        &'a self,
        request: &'a mut ChatRequest,
        info: &'a RunInfo,
    ) -> BoxFuture<'a, ()> {
        request.temperature = Some(0.25);
        self.events
            .lock()
            .unwrap()
            .push(format!("model:{}", info.steps_used));
        Box::pin(std::future::ready(()))
    }

    fn before_tool<'a>(
        &'a self,
        call: &'a ToolCall,
        _ctx: &'a ToolContext,
    ) -> BoxFuture<'a, HookAction> {
        self.events
            .lock()
            .unwrap()
            .push(format!("before:{}", call.name));
        let action = match call.name.as_str() {
            "blocked" => HookAction::Block("not allowed here".to_owned()),
            "rewritten" => HookAction::Modify(json!({"city": "Bergen"})),
            _ => HookAction::Continue,
        };
        Box::pin(std::future::ready(action))
    }

    fn after_tool<'a>(
        &'a self,
        call: &'a ToolCall,
        output: &'a ToolOutput,
        _ctx: &'a ToolContext,
    ) -> BoxFuture<'a, ()> {
        self.events
            .lock()
            .unwrap()
            .push(format!("after:{}:{}", call.name, output.is_error));
        Box::pin(std::future::ready(()))
    }

    fn on_outcome<'a>(
        &'a self,
        outcome: &'a AgentOutcome,
        _info: &'a RunInfo,
    ) -> BoxFuture<'a, ()> {
        self.events
            .lock()
            .unwrap()
            .push(format!("outcome:{}", outcome.is_completed()));
        Box::pin(std::future::ready(()))
    }
}

#[tokio::test]
async fn hooks_observe_block_and_rewrite() {
    let seen = Arc::new(Mutex::new(Vec::new()));
    let client = ScriptClient::new(vec![
        ScriptClient::calls(&[("x", "blocked"), ("y", "rewritten")]),
        ScriptClient::answer("done"),
    ]);
    let recorder = Arc::new(Recorder::default());
    let agent = Agent::new(client.clone())
        .tool(counting_tool("blocked", Arc::clone(&seen)))
        .tool(counting_tool("rewritten", Arc::clone(&seen)))
        .hook(recorder.clone())
        .parallel_tool_calls(false);
    assert!(agent.run("go").await.unwrap().is_completed());
    assert_eq!(*seen.lock().unwrap(), vec![json!({"city": "Bergen"})]);
    assert_eq!(
        *recorder.events.lock().unwrap(),
        vec![
            "model:0",
            "before:blocked",
            "after:blocked:true",
            "before:rewritten",
            "after:rewritten:false",
            "model:1",
            "outcome:true",
        ]
    );
    assert!(
        client
            .requests()
            .iter()
            .all(|r| r.temperature == Some(0.25))
    );
    let results = tool_results(client.requests().last().unwrap());
    assert!(results[0].contains("blocked by hook: not allowed here"));
}

#[tokio::test]
async fn repeated_identical_calls_warn_then_stop() {
    let responses: Vec<ChatResponse> = (0..10)
        .map(|i| ScriptClient::tool_call(&format!("c{i}"), "get_weather", json!({"city": "X"})))
        .collect();
    let client = ScriptClient::new(responses);
    let agent = Agent::new(client.clone())
        .tools(vec![weather_tool()])
        .max_steps(50);
    match agent.run("go").await.unwrap() {
        AgentOutcome::LoopDetected {
            tool,
            repeats,
            steps_used,
            ..
        } => {
            assert_eq!(tool, "get_weather");
            assert_eq!(repeats, 5);
            assert_eq!(steps_used, 5);
        }
        other => panic!("expected a loop stop, got {other:?}"),
    }
    let results = tool_results(client.requests().last().unwrap());
    assert!(!results[1].contains("[harness]"));
    assert!(results[2].contains("[harness] You called get_weather 3 times"));
}

/// A client that never answers in time.
#[derive(Debug)]
struct SlowClient;

impl LlmClient for SlowClient {
    fn chat<'a>(
        &'a self,
        _request: &'a ChatRequest,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<ChatResponse, AgentError>> + Send + 'a>> {
        Box::pin(async {
            tokio::time::sleep(std::time::Duration::from_secs(3_600)).await;
            Ok(ScriptClient::answer("late"))
        })
    }

    fn list_models(
        &self,
    ) -> std::pin::Pin<Box<dyn Future<Output = Result<Vec<String>, AgentError>> + Send + '_>> {
        Box::pin(async { Ok(Vec::new()) })
    }

    fn provider_name(&self) -> &'static str {
        "slow"
    }
}

#[tokio::test(start_paused = true)]
async fn deadline_stops_a_slow_model() {
    let agent = Agent::new(Arc::new(SlowClient)).max_duration(std::time::Duration::from_secs(5));
    let outcome = agent.run("go").await.unwrap();
    assert!(matches!(
        outcome,
        AgentOutcome::BudgetExhausted {
            reason: BudgetKind::Deadline,
            ..
        }
    ));
}

#[tokio::test(start_paused = true)]
async fn deadline_cuts_off_a_slow_tool() {
    let client = ScriptClient::new(vec![
        ScriptClient::tool_call("c1", "slow", json!({})),
        ScriptClient::answer("never"),
    ]);
    let slow = FnTool::new("slow", "Sleeps.", json!({}), |_: serde_json::Value| async {
        tokio::time::sleep(std::time::Duration::from_secs(3_600)).await;
        Ok(json!("late"))
    })
    .shared();
    let agent = Agent::new(client.clone())
        .tool(slow)
        .max_duration(std::time::Duration::from_secs(5));
    let turn = agent.run_turn(Vec::new(), "go").await.unwrap();
    assert!(matches!(
        turn.outcome,
        AgentOutcome::BudgetExhausted {
            reason: BudgetKind::Deadline,
            ..
        }
    ));
    // The cut-off call still has a result, so the transcript stays valid.
    let last = turn.messages.last().unwrap();
    assert_eq!(last.role, ChatRole::Tool);
    assert!(matches!(
        &last.content[0],
        ContentPart::ToolResult { content, .. } if content.contains("did not finish")
    ));
}

#[tokio::test(start_paused = true)]
async fn parallel_calls_overlap_and_keep_order() {
    let make = |name: &'static str, secs: u64| {
        FnTool::new(
            name,
            "Sleeps.",
            json!({}),
            move |_: serde_json::Value| async move {
                tokio::time::sleep(std::time::Duration::from_secs(secs)).await;
                Ok(json!(name))
            },
        )
        .shared()
    };
    for (parallel, expected_secs) in [(true, 10), (false, 15)] {
        let client = ScriptClient::new(vec![
            ScriptClient::calls(&[("a", "ten"), ("b", "five")]),
            ScriptClient::answer("done"),
        ]);
        let agent = Agent::new(client.clone())
            .tool(make("ten", 10))
            .tool(make("five", 5))
            .parallel_tool_calls(parallel);
        let started = tokio::time::Instant::now();
        assert!(agent.run("go").await.unwrap().is_completed());
        assert_eq!(started.elapsed().as_secs(), expected_secs);
        let results = tool_results(client.requests().last().unwrap());
        assert_eq!(results, vec!["\"ten\"", "\"five\""]);
    }
}

#[tokio::test]
async fn memory_is_a_frozen_snapshot_plus_a_tool() {
    let store = InMemoryMemoryStore::default().shared();
    let client = ScriptClient::new(vec![
        ScriptClient::tool_call(
            "m1",
            "memory",
            json!({"action": "add", "block": "user", "text": "prefers metric units"}),
        ),
        ScriptClient::answer("Noted."),
        ScriptClient::answer("Celsius it is."),
    ]);
    let agent = Agent::new(client.clone()).memory(Arc::clone(&store), MemoryScope::agent());
    assert!(agent.run("Use metric.").await.unwrap().is_completed());
    let requests = client.requests();
    assert!(requests[0].tools.iter().any(|tool| tool.name == "memory"));
    assert!(system_text(&requests[0]).contains("## Memory"));
    // Frozen: the write does not show up mid-run...
    assert!(!system_text(&requests[1]).contains("prefers metric units"));
    // ...but it persisted, and the next run sees it.
    let blocks = store.load(&MemoryScope::agent()).await.unwrap();
    assert_eq!(blocks[1].entries, vec!["prefers metric units"]);
    agent.run("Weather?").await.unwrap();
    assert!(system_text(&client.requests()[2]).contains("- prefers metric units"));
}

#[tokio::test]
async fn skills_are_listed_and_loadable() {
    let client = ScriptClient::new(vec![
        ScriptClient::tool_call("s1", "load_skill", json!({"name": "digest"})),
        ScriptClient::answer("Digest written."),
    ]);
    let agent = Agent::new(client.clone()).skills(vec![Skill::new(
        "digest",
        "How to write the digest.",
        "Step one: gather.",
    )]);
    assert!(agent.run("Write the digest.").await.unwrap().is_completed());
    let requests = client.requests();
    let system = system_text(&requests[0]);
    assert!(system.contains("- digest: How to write the digest."));
    assert!(!system.contains("Step one"));
    assert!(tool_results(&requests[1])[0].contains("Step one: gather."));
}

#[tokio::test]
async fn sessions_persist_turns_but_not_pauses() {
    let store = InMemorySessionStore::new();
    let session = SessionId::new("thread-1");
    let client = ScriptClient::new(vec![
        ScriptClient::answer("Hi!"),
        ScriptClient::tool_call("c1", "act", json!({})),
        ScriptClient::answer("Done."),
    ]);
    let agent = Agent::new(client.clone())
        .tool(counting_tool("act", Arc::default()))
        .policy(Arc::new(ToolRules::new().tool("act", Rule::Ask)));
    agent
        .run_in_session(&store, &session, "hello")
        .await
        .unwrap();
    assert_eq!(store.load(&session).await.unwrap().len(), 2);

    let turn = agent
        .run_in_session(&store, &session, "act now")
        .await
        .unwrap();
    let state = paused(turn.outcome);
    assert_eq!(state.session_id.as_ref(), Some(&session));
    assert_eq!(
        store.load(&session).await.unwrap().len(),
        2,
        "a pause saves nothing"
    );

    let turn = agent
        .resume_in_session(&store, state, vec![ApprovalDecision::approve("c1")])
        .await
        .unwrap();
    assert!(turn.outcome.is_completed());
    // hello, Hi!, act now, tool call, tool result, Done.
    assert_eq!(store.load(&session).await.unwrap().len(), 6);
}

#[tokio::test]
async fn long_sessions_compact_before_the_run() {
    let store = InMemorySessionStore::new();
    let session = SessionId::new("long");
    let mut history = Vec::new();
    for i in 0..20 {
        history.push(ChatMessage::text(
            ChatRole::User,
            format!("question {i} {}", "x".repeat(400)),
        ));
        history.push(ChatMessage::text(
            ChatRole::Assistant,
            format!("answer {i} {}", "y".repeat(400)),
        ));
    }
    store.save(&session, &history).await.unwrap();
    let client = ScriptClient::new(vec![
        ScriptClient::answer("The user asked twenty questions."),
        ScriptClient::answer("Fine."),
    ]);
    let agent = Agent::new(client.clone()).compaction(crate::session::Compaction {
        trigger_tokens: 1_000,
        keep_recent_tokens: 500,
        summary_max_tokens: 256,
    });
    let turn = agent
        .run_in_session(&store, &session, "next")
        .await
        .unwrap();
    let report = turn.compaction.expect("compaction ran");
    assert!(report.tokens_after < report.tokens_before);
    let saved = store.load(&session).await.unwrap();
    assert!(matches!(
        &saved[0].content[0],
        ContentPart::Text(text) if text.starts_with(SUMMARY_PREFIX)
            && text.contains("twenty questions")
    ));
    assert!(saved.len() < history.len());
    // The summary call ran without tools.
    assert!(client.requests()[0].tools.is_empty());
}

#[tokio::test]
async fn run_id_reaches_tools_and_hooks() {
    let client = ScriptClient::new(vec![
        ScriptClient::tool_call("c1", "who", json!({})),
        ScriptClient::answer("ok"),
    ]);
    let who = FnTool::with_context(
        "who",
        "Echoes the context.",
        json!({}),
        |_: serde_json::Value, ctx: ToolContext| async move {
            Ok(json!({"run": ctx.run_id.as_str(), "call": ctx.call_id, "step": ctx.step}))
        },
    )
    .shared();
    let agent = Agent::new(client.clone())
        .tool(who)
        .run_id(RunId::new("job-42"));
    agent.run("go").await.unwrap();
    let results = tool_results(client.requests().last().unwrap());
    assert_eq!(results[0], r#"{"call":"c1","run":"job-42","step":0}"#);
}

#[tokio::test]
async fn overshooting_tool_calls_are_not_recorded() {
    let response = ChatResponse {
        content: vec![ContentPart::ToolCall {
            id: "c1".to_owned(),
            name: "get_weather".to_owned(),
            arguments: json!({}),
        }],
        stop_reason: StopReason::ToolUse,
        usage: TokenUsage::new(900, 200),
    };
    let client = ScriptClient::new(vec![response]);
    let agent = Agent::new(client)
        .tools(vec![weather_tool()])
        .max_tokens(1000);
    let turn = agent.run_turn(Vec::new(), "go").await.unwrap();
    assert!(matches!(
        turn.outcome,
        AgentOutcome::BudgetExhausted {
            reason: BudgetKind::Tokens,
            ..
        }
    ));
    assert_eq!(turn.messages.len(), 1, "only the user message");
}

#[test]
fn outcomes_round_trip_through_json() {
    let usage = TokenUsage::new(3, 4);
    let outcomes = vec![
        AgentOutcome::Completed {
            text: "hi".to_owned(),
            steps_used: 1,
            usage,
        },
        AgentOutcome::BudgetExhausted {
            reason: BudgetKind::Deadline,
            steps_used: 2,
            usage,
        },
        AgentOutcome::LoopDetected {
            tool: "t".to_owned(),
            repeats: 5,
            steps_used: 5,
            usage,
        },
    ];
    for outcome in outcomes {
        let value = serde_json::to_value(&outcome).unwrap();
        assert!(value["status"].is_string(), "{value}");
        let back: AgentOutcome = serde_json::from_value(value).unwrap();
        assert_eq!(back, outcome);
    }
    let value = serde_json::to_value(AgentOutcome::Completed {
        text: "x".to_owned(),
        steps_used: 0,
        usage,
    })
    .unwrap();
    assert_eq!(value["status"], json!("completed"));
}

#[test]
fn truncation_keeps_the_task_and_drops_orphaned_results() {
    let call = ChatMessage {
        role: ChatRole::Assistant,
        content: vec![ContentPart::ToolCall {
            id: "c".to_owned(),
            name: "t".to_owned(),
            arguments: json!({}),
        }],
    };
    let result = ChatMessage {
        role: ChatRole::Tool,
        content: vec![ContentPart::ToolResult {
            tool_call_id: "c".to_owned(),
            content: "r".repeat(40),
        }],
    };
    let messages = vec![
        ChatMessage::text(ChatRole::System, "sys"),
        ChatMessage::text(ChatRole::User, "the task"),
        call.clone(),
        ChatMessage {
            role: ChatRole::Tool,
            content: vec![ContentPart::ToolResult {
                tool_call_id: "c".to_owned(),
                content: "z".repeat(4_000),
            }],
        },
        call,
        result,
    ];
    let budget = estimate_messages(&messages[..3]) + estimate_messages(&messages[4..]);
    let kept = truncate_history(&messages, budget);
    assert_eq!(kept[0].role, ChatRole::System);
    assert_eq!(kept[1], messages[1], "the task anchor survives");
    assert_eq!(kept[2].role, ChatRole::Assistant);
    assert_eq!(kept.len(), 4);
}
