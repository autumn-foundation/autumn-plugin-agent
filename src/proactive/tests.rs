#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::Mutex;

use serde_json::json;

use super::*;
use crate::agent::{AgentRuntime, RunState};
use crate::config::AgentConfig;
use crate::policy::{Rule, ToolDecision};
use crate::test_support::Script;
use crate::tools::FnTool;

/// Collects every delivered report.
#[derive(Debug, Default)]
struct Inbox(Mutex<Vec<Report>>);

impl Delivery for Inbox {
    fn deliver<'a>(&'a self, report: &'a Report) -> BoxFuture<'a, Result<(), AgentError>> {
        self.0.lock().unwrap().push(report.clone());
        Box::pin(std::future::ready(Ok(())))
    }
}

/// Fails every delivery.
#[derive(Debug)]
struct Broken;

impl Delivery for Broken {
    fn deliver<'a>(&'a self, _report: &'a Report) -> BoxFuture<'a, Result<(), AgentError>> {
        Box::pin(std::future::ready(Err(AgentError::new(
            ErrorKind::Transport,
            "down",
        ))))
    }
}

fn runtime(client: Arc<Script>, inbox: Arc<Inbox>, tools: Vec<Arc<dyn Tool>>) -> AgentRuntime {
    AgentRuntime::new(AgentConfig::default(), client, tools).with_delivery(inbox)
}

fn writer(count: Arc<Mutex<u32>>) -> Arc<dyn Tool> {
    FnTool::new(
        "restart_server",
        "Restarts.",
        json!({}),
        move |_: serde_json::Value| {
            let count = Arc::clone(&count);
            async move {
                *count.lock().unwrap() += 1;
                Ok(json!("restarted"))
            }
        },
    )
    .effect(ToolEffect::External)
    .shared()
}

#[test]
fn silence_rules() {
    assert!(is_silent("HEARTBEAT_OK"));
    assert!(is_silent("  HEARTBEAT_OK \n"));
    assert!(is_silent("HEARTBEAT_OK - all quiet"));
    assert!(is_silent("All quiet. HEARTBEAT_OK"));
    assert!(!is_silent("The disk is 95% full."));
    assert!(!is_silent(&format!("HEARTBEAT_OK {}", "x".repeat(301))));
    assert!(!is_silent("heartbeat_ok"));
}

#[test]
fn report_text_rules() {
    let usage = crate::client::TokenUsage::default();
    let done = |text: &str| AgentOutcome::Completed {
        text: text.to_owned(),
        steps_used: 0,
        usage,
    };
    assert_eq!(
        report_text(&done("Disk full")),
        Some("Disk full".to_owned())
    );
    assert_eq!(report_text(&done(HEARTBEAT_OK)), None);
    let budget = AgentOutcome::BudgetExhausted {
        reason: crate::agent::BudgetKind::Steps,
        steps_used: 1,
        usage,
    };
    assert_eq!(report_text(&budget), None);
    let state = RunState {
        run_id: RunId::new("r"),
        session_id: None,
        messages: Vec::new(),
        usage,
        steps_used: 1,
        pending: vec![crate::agent::PendingCall {
            call: crate::tools::ToolCall {
                id: "c".to_owned(),
                name: "send".to_owned(),
                arguments: json!({"to": "x"}),
            },
            status: crate::agent::PendingStatus::NeedsApproval {
                reason: "r".to_owned(),
            },
        }],
        recent_calls: Vec::new(),
    };
    let text = report_text(&AgentOutcome::AwaitingApproval {
        state: Box::new(state),
    })
    .unwrap();
    assert!(text.contains("- send {\"to\":\"x\"}"), "{text}");
}

#[test]
fn heartbeat_builder_and_task() {
    let heartbeat = Heartbeat::cron("0 */30 * * * *")
        .timezone("Europe/Oslo")
        .session("ops")
        .memory_scope(MemoryScope::new("ops"))
        .per_replica();
    assert_eq!(
        heartbeat.schedule(),
        &HeartbeatSchedule::Cron {
            expression: "0 */30 * * * *".to_owned(),
            timezone: Some("Europe/Oslo".to_owned()),
        }
    );
    heartbeat.validate().unwrap();
    let task = heartbeat.task_info();
    assert_eq!(task.name, Heartbeat::TASK_NAME);
    assert_eq!(task.coordination, TaskCoordination::PerReplica);
    assert_eq!(task.schedule.to_string(), "cron 0 */30 * * * *");
    assert!(format!("{heartbeat:?}").contains("read_only: true"));

    let every = Heartbeat::every(Duration::from_secs(1_800)).timezone("ignored");
    assert_eq!(every.task_info().schedule.to_string(), "every 1800s");
    assert_eq!(every.task_info().coordination, TaskCoordination::Fleet);
}

#[test]
fn heartbeat_validation() {
    assert!(
        Heartbeat::every(Duration::from_millis(10))
            .validate()
            .is_err()
    );
    assert!(Heartbeat::cron("  ").validate().is_err());
    assert!(
        Heartbeat::every(Duration::from_secs(60))
            .prompt(" ")
            .validate()
            .is_err()
    );
}

#[tokio::test]
async fn quiet_heartbeat_delivers_nothing() {
    let client = Script::new(vec![Script::answer(HEARTBEAT_OK)]);
    let inbox = Arc::new(Inbox::default());
    let runtime = runtime(client.clone(), Arc::clone(&inbox), Vec::new());
    let outcome = Heartbeat::every(Duration::from_secs(60))
        .tick(&runtime, &AppState::detached())
        .await
        .unwrap()
        .unwrap();
    assert!(outcome.is_completed());
    assert!(inbox.0.lock().unwrap().is_empty());
    let requests = client.requests();
    let prompt = &requests[0].messages.last().unwrap().content[0];
    assert!(matches!(prompt, crate::client::ContentPart::Text(t) if t.contains("HEARTBEAT_OK")));
}

#[tokio::test]
async fn noteworthy_heartbeat_is_delivered_and_stays_read_only() {
    let restarts = Arc::new(Mutex::new(0));
    let client = Script::new(vec![
        Script::call("c1", "restart_server", json!({})),
        Script::answer("The server is down; I could not restart it."),
    ]);
    let inbox = Arc::new(Inbox::default());
    let runtime = runtime(
        client.clone(),
        Arc::clone(&inbox),
        vec![writer(Arc::clone(&restarts))],
    );
    Heartbeat::every(Duration::from_secs(60))
        .session("ops")
        .tick(&runtime, &AppState::detached())
        .await
        .unwrap();
    assert_eq!(
        *restarts.lock().unwrap(),
        0,
        "read-only heartbeat must not act"
    );
    let reports = inbox.0.lock().unwrap().clone();
    assert_eq!(reports.len(), 1);
    assert_eq!(reports[0].source, ReportSource::Heartbeat);
    assert_eq!(reports[0].session_id, Some(SessionId::new("ops")));
    assert!(reports[0].text.contains("server is down"));
    // The tick ran in the session.
    let saved = runtime
        .sessions()
        .load(&SessionId::new("ops"))
        .await
        .unwrap();
    assert_eq!(saved.len(), 4);
}

#[tokio::test]
async fn allow_actions_lifts_the_read_only_rule_but_keeps_the_app_policy() {
    let restarts = Arc::new(Mutex::new(0));
    let client = Script::new(vec![
        Script::call("c1", "restart_server", json!({})),
        Script::answer("Restarted."),
    ]);
    let inbox = Arc::new(Inbox::default());
    let runtime = runtime(client, inbox, vec![writer(Arc::clone(&restarts))]);
    Heartbeat::every(Duration::from_secs(60))
        .allow_actions()
        .tick(&runtime, &AppState::detached())
        .await
        .unwrap();
    assert_eq!(*restarts.lock().unwrap(), 1);

    // An app policy that asks first still applies to read-only heartbeats.
    let client = Script::new(vec![Script::call("c1", "restart_server", json!({}))]);
    let inbox = Arc::new(Inbox::default());
    let strict = AgentRuntime::new(AgentConfig::default(), client, vec![writer(restarts)])
        .with_delivery(inbox)
        .with_policy(Arc::new(
            crate::policy::ToolRules::new().tool("restart_server", Rule::Ask),
        ));
    let decision = Strictest::new(vec![strict.policy(), Arc::new(ToolRules::read_only())])
        .decide(
            &crate::tools::ToolCall {
                id: "c".to_owned(),
                name: "restart_server".to_owned(),
                arguments: json!({}),
            },
            None,
            &crate::hooks::RunInfo {
                run_id: RunId::new("r"),
                session_id: None,
                steps_used: 0,
                max_steps: 1,
                usage: crate::client::TokenUsage::default(),
            },
        )
        .await;
    assert!(matches!(decision, ToolDecision::RequireApproval { .. }));
}

#[tokio::test]
async fn precheck_skips_the_model_call() {
    let client = Script::new(Vec::new());
    let runtime = runtime(client.clone(), Arc::default(), Vec::new());
    let heartbeat = Heartbeat::every(Duration::from_secs(60)).precheck(|_state| async { false });
    assert!(format!("{heartbeat:?}").contains("precheck: true"));
    let outcome = heartbeat
        .tick(&runtime, &AppState::detached())
        .await
        .unwrap();
    assert!(outcome.is_none());
    assert!(client.requests().is_empty());
}

#[tokio::test]
async fn failed_delivery_is_logged_not_returned() {
    let client = Script::new(vec![Script::answer("Something happened.")]);
    let runtime = AgentRuntime::new(AgentConfig::default(), client, Vec::new())
        .with_delivery(Arc::new(Broken));
    let outcome = Heartbeat::every(Duration::from_secs(60))
        .tick(&runtime, &AppState::detached())
        .await
        .unwrap();
    assert!(outcome.unwrap().is_completed());
}

#[tokio::test]
async fn heartbeat_task_handler_without_the_plugin_is_a_no_op() {
    heartbeat_tick(AppState::detached()).await.unwrap();
}

#[tokio::test]
async fn heartbeat_task_handler_runs_the_installed_heartbeat() {
    let client = Script::new(vec![Script::answer("Report!")]);
    let inbox = Arc::new(Inbox::default());
    let state = AppState::detached();
    let installed = runtime(client.clone(), Arc::clone(&inbox), Vec::new());
    state.extension_or_insert_with(|| installed);
    state.extension_or_insert_with(|| HeartbeatSettings(Heartbeat::every(Duration::from_secs(60))));
    heartbeat_tick(state).await.unwrap();
    assert_eq!(inbox.0.lock().unwrap().len(), 1);
    assert_eq!(client.requests().len(), 1);
}

#[test]
fn followup_plans_are_validated() {
    let tool = FollowupTool::new(Duration::from_secs(3_600)).memory_scope(MemoryScope::new("u1"));
    let ctx = ToolContext {
        run_id: RunId::new("r"),
        call_id: "c".to_owned(),
        session_id: Some(SessionId::new("chat-9")),
        step: 0,
    };
    let (args, delay) = tool
        .plan(
            &json!({"delay_minutes": 20, "prompt": " check the deploy "}),
            &ctx,
        )
        .unwrap();
    assert_eq!(delay, Duration::from_secs(1_200));
    assert_eq!(args.prompt, "check the deploy");
    assert_eq!(args.session, Some(SessionId::new("chat-9")));
    assert_eq!(args.memory_scope, Some(MemoryScope::new("u1")));
    assert!(args.deliver);
    assert_eq!(args.origin, RunOrigin::Followup);
    assert_eq!(args.followup_depth, 1);

    // Chains are capped.
    let deep = FollowupTool::new(Duration::from_secs(3_600)).depth(FollowupTool::MAX_CHAIN - 1);
    let (args, _) = deep
        .plan(&json!({"delay_minutes": 1, "prompt": "x"}), &ctx)
        .unwrap();
    assert_eq!(args.followup_depth, FollowupTool::MAX_CHAIN);
    let too_deep = FollowupTool::new(Duration::from_secs(3_600)).depth(FollowupTool::MAX_CHAIN);
    let err = too_deep
        .plan(&json!({"delay_minutes": 1, "prompt": "x"}), &ctx)
        .unwrap_err();
    assert!(
        err.message().contains("follow-ups deep"),
        "{}",
        err.message()
    );

    for bad in [
        json!({"delay_minutes": 0, "prompt": "x"}),
        json!({"delay_minutes": 61, "prompt": "x"}),
        json!({"delay_minutes": "5", "prompt": "x"}),
        json!({"delay_minutes": 5}),
        json!({"delay_minutes": 5, "prompt": "  "}),
        json!({"delay_minutes": 5, "prompt": "x".repeat(2_001)}),
    ] {
        let err = tool.plan(&bad, &ctx).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::Tool, "{bad}");
    }
    assert_eq!(tool.effect(), ToolEffect::Internal);
    assert_eq!(
        tool.input_schema()["properties"]["delay_minutes"]["maximum"],
        json!(60)
    );
}

#[tokio::test]
async fn followup_without_a_job_runtime_reports_a_tool_error() {
    let tool = FollowupTool::new(Duration::from_secs(3_600));
    let err = tool
        .execute(
            json!({"delay_minutes": 5, "prompt": "check"}),
            &ToolContext::detached("c"),
        )
        .await
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::Tool);
    assert!(err.message().contains("cannot schedule the follow-up"));
}

#[tokio::test]
async fn log_delivery_accepts_reports() {
    let report = Report {
        source: ReportSource::Job,
        run_id: RunId::new("r"),
        session_id: None,
        text: "hi".to_owned(),
        outcome: AgentOutcome::Completed {
            text: "hi".to_owned(),
            steps_used: 0,
            usage: crate::client::TokenUsage::default(),
        },
    };
    LogDelivery.deliver(&report).await.unwrap();
}
