#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

//! Tests of the core half of `proactive`. They run with and without the
//! `autumn` feature.

use serde_json::json;

use super::*;
use crate::agent::RunState;

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
