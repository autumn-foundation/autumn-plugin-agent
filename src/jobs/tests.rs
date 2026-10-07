#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::Mutex;

use serde_json::json;

use super::*;
use crate::agent::ApprovalDecision;
use crate::config::AgentConfig;
use crate::policy::{Rule, ToolRules};
use crate::proactive::{Delivery, Report};
use crate::test_support::Script;
use crate::tools::FnTool;

#[derive(Debug, Default)]
struct Inbox(Mutex<Vec<Report>>);

impl Delivery for Inbox {
    fn deliver<'a>(&'a self, report: &'a Report) -> BoxFuture<'a, Result<(), AgentError>> {
        self.0.lock().unwrap().push(report.clone());
        Box::pin(std::future::ready(Ok(())))
    }
}

#[test]
fn args_builder_and_serde_defaults() {
    let args = AgentRunArgs::new("hi")
        .system_prompt("sys")
        .max_steps(3)
        .session(SessionId::new("s"))
        .memory_scope(MemoryScope::new("u"))
        .deliver(true)
        .origin(RunOrigin::Heartbeat);
    assert_eq!(args.max_steps, Some(3));
    let value = serde_json::to_value(&args).unwrap();
    let back: AgentRunArgs = serde_json::from_value(value).unwrap();
    assert_eq!(back, args);

    // Old 0.1 payloads still decode; a run id is minted.
    let old: AgentRunArgs = serde_json::from_value(json!({"prompt": "x"})).unwrap();
    assert!(!old.deliver);
    assert_eq!(old.origin, RunOrigin::Request);
    assert!(old.run_id.as_str().starts_with("run_"));
    assert_eq!(ReportSource::from(RunOrigin::Request), ReportSource::Job);
    assert_eq!(
        ReportSource::from(RunOrigin::Followup),
        ReportSource::Followup
    );
    assert_eq!(
        ReportSource::from(RunOrigin::Heartbeat),
        ReportSource::Heartbeat
    );
}

#[tokio::test]
async fn execute_run_uses_the_session_and_delivers() {
    let client = Script::new(vec![Script::answer("Done: 3 signups.")]);
    let inbox = Arc::new(Inbox::default());
    let runtime = AgentRuntime::new(AgentConfig::default(), client.clone(), Vec::new())
        .with_delivery(Arc::clone(&inbox) as Arc<dyn Delivery>);
    let args = AgentRunArgs::new("Summarize signups.")
        .system_prompt("Override.")
        .max_steps(2)
        .session(SessionId::new("daily"))
        .deliver(true);
    let run_id = args.run_id.clone();
    let outcome = execute_run(&runtime, args, None).await.unwrap();
    assert_eq!(outcome.text(), Some("Done: 3 signups."));
    let reports = inbox.0.lock().unwrap().clone();
    assert_eq!(reports.len(), 1);
    assert_eq!(reports[0].run_id, run_id);
    assert_eq!(reports[0].source, ReportSource::Job);
    let saved = runtime
        .sessions()
        .load(&SessionId::new("daily"))
        .await
        .unwrap();
    assert_eq!(saved.len(), 2);
    let system = &client.requests()[0].messages[0];
    assert!(matches!(&system.content[0], crate::client::ContentPart::Text(t) if t == "Override."));
}

#[tokio::test]
async fn execute_resume_finishes_a_paused_run() {
    let deliveries = Arc::new(Mutex::new(0));
    let counter = Arc::clone(&deliveries);
    let send_tool = FnTool::new("send", "Sends.", json!({}), move |_: serde_json::Value| {
        let counter = Arc::clone(&counter);
        async move {
            *counter.lock().unwrap() += 1;
            Ok(json!("sent"))
        }
    })
    .shared();
    let client = Script::new(vec![
        Script::call("c1", "send", json!({})),
        Script::answer("Sent it."),
    ]);
    let inbox = Arc::new(Inbox::default());
    let runtime = AgentRuntime::new(AgentConfig::default(), client, vec![send_tool])
        .with_policy(Arc::new(ToolRules::new().tool("send", Rule::Ask)))
        .with_delivery(Arc::clone(&inbox) as Arc<dyn Delivery>);
    let outcome = execute_run(
        &runtime,
        AgentRunArgs::new("send it")
            .session(SessionId::new("s"))
            .deliver(true),
        None,
    )
    .await
    .unwrap();
    let AgentOutcome::AwaitingApproval { state } = outcome else {
        panic!("expected a pause");
    };
    // The pause is delivered so a person can approve it.
    assert!(inbox.0.lock().unwrap()[0].text.contains("needs approval"));
    let args = AgentResumeArgs::new(*state, vec![ApprovalDecision::approve("c1")])
        .memory_scope(MemoryScope::agent())
        .deliver(true);
    let round_trip: AgentResumeArgs =
        serde_json::from_value(serde_json::to_value(&args).unwrap()).unwrap();
    let outcome = execute_resume(&runtime, round_trip).await.unwrap();
    assert_eq!(outcome.text(), Some("Sent it."));
    assert_eq!(*deliveries.lock().unwrap(), 1);
    let reports = inbox.0.lock().unwrap().clone();
    assert_eq!(reports.len(), 2);
    assert_eq!(reports[1].source, ReportSource::Resume);
    assert_eq!(
        runtime
            .sessions()
            .load(&SessionId::new("s"))
            .await
            .unwrap()
            .len(),
        4
    );
}

#[tokio::test]
async fn progress_hook_is_a_no_op_outside_tracked_jobs() {
    let mut request = crate::client::ChatRequest {
        messages: Vec::new(),
        tools: Vec::new(),
        max_tokens: None,
        temperature: None,
    };
    let info = RunInfo {
        run_id: RunId::new("r"),
        session_id: None,
        steps_used: 3,
        max_steps: 0,
        usage: crate::client::TokenUsage::default(),
    };
    JobProgress.before_model(&mut request, &info).await;
}

#[tokio::test]
async fn enqueue_without_a_job_runtime_fails_cleanly() {
    assert!(enqueue_agent_run(AgentRunArgs::new("x")).await.is_err());
    assert!(
        enqueue_agent_run_tracked(AgentRunArgs::new("x"))
            .await
            .is_err()
    );
    assert!(
        enqueue_agent_run_in(AgentRunArgs::new("x"), Duration::from_secs(60))
            .await
            .is_err()
    );
}

#[tokio::test]
async fn jobs_without_the_plugin_are_unavailable() {
    let err = runtime_from(&AppState::detached()).unwrap_err();
    assert_eq!(err.status(), http::StatusCode::SERVICE_UNAVAILABLE);
}
