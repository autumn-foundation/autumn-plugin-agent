#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use serde_json::json;

use super::*;
use crate::client::TokenUsage;
use crate::ids::RunId;
use crate::tools::FnTool;

fn call(name: &str) -> ToolCall {
    ToolCall {
        id: "c1".to_owned(),
        name: name.to_owned(),
        arguments: json!({}),
    }
}

fn tool(name: &str, effect: ToolEffect) -> FnTool {
    FnTool::new(
        name,
        "d",
        json!({}),
        |input: serde_json::Value| async move { Ok(input) },
    )
    .effect(effect)
}

fn info() -> RunInfo {
    RunInfo {
        run_id: RunId::new("r"),
        session_id: None,
        steps_used: 0,
        max_steps: 10,
        usage: TokenUsage::default(),
    }
}

#[tokio::test]
async fn allow_all_allows() {
    let decision = AllowAll.decide(&call("x"), None, &info()).await;
    assert_eq!(decision, ToolDecision::Allow);
}

#[test]
fn read_only_rules_gate_by_effect() {
    let rules = ToolRules::read_only();
    let read = tool("lookup", ToolEffect::ReadOnly);
    let note = tool("memory", ToolEffect::Internal);
    let write = tool("save", ToolEffect::Write);
    let send = tool("send", ToolEffect::External);
    assert_eq!(
        rules.decide_now(&call("lookup"), Some(&read)),
        ToolDecision::Allow
    );
    assert_eq!(
        rules.decide_now(&call("memory"), Some(&note)),
        ToolDecision::Allow
    );
    assert!(matches!(
        rules.decide_now(&call("save"), Some(&write)),
        ToolDecision::Deny { .. }
    ));
    assert!(matches!(
        rules.decide_now(&call("send"), Some(&send)),
        ToolDecision::Deny { .. }
    ));
}

#[test]
fn name_rules_win_over_effect_rules() {
    let rules = ToolRules::ask_before_acting().tool("send_digest", Rule::Allow);
    let digest = tool("send_digest", ToolEffect::External);
    let email = tool("send_email", ToolEffect::External);
    assert_eq!(
        rules.decide_now(&call("send_digest"), Some(&digest)),
        ToolDecision::Allow
    );
    assert_eq!(
        rules.decide_now(&call("send_email"), Some(&email)),
        ToolDecision::RequireApproval {
            reason: "send_email needs approval before it runs".to_owned()
        }
    );
}

#[test]
fn unknown_tools_fall_back_to_name_rules_then_allow() {
    let rules = ToolRules::read_only().tool("ghost", Rule::Deny("no".to_owned()));
    assert_eq!(
        rules.decide_now(&call("ghost"), None),
        ToolDecision::Deny {
            reason: "no".to_owned()
        }
    );
    assert_eq!(rules.decide_now(&call("other"), None), ToolDecision::Allow);
}

#[test]
fn decisions_serialize_with_a_tag() {
    let value = serde_json::to_value(ToolDecision::Deny {
        reason: "r".to_owned(),
    })
    .unwrap();
    assert_eq!(value, json!({"decision": "deny", "reason": "r"}));
}

#[tokio::test]
async fn strictest_keeps_the_strictest_answer() {
    let ask: Arc<dyn ToolPolicy> = Arc::new(ToolRules::new().tool("send", Rule::Ask));
    let deny: Arc<dyn ToolPolicy> =
        Arc::new(ToolRules::new().tool("wipe", Rule::Deny("no".to_owned())));
    let combined = Strictest::new(vec![Arc::new(AllowAll), ask, deny]);
    assert_eq!(
        combined.decide(&call("read"), None, &info()).await,
        ToolDecision::Allow
    );
    assert!(matches!(
        combined.decide(&call("send"), None, &info()).await,
        ToolDecision::RequireApproval { .. }
    ));
    assert!(matches!(
        combined.decide(&call("wipe"), None, &info()).await,
        ToolDecision::Deny { .. }
    ));
    let empty = Strictest::new(Vec::new());
    assert_eq!(
        empty.decide(&call("x"), None, &info()).await,
        ToolDecision::Allow
    );
}
