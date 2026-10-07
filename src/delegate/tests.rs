#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::sync::Arc;

use serde_json::json;

use super::*;
use crate::test_support::Script;

#[tokio::test]
async fn child_answer_comes_back_as_the_tool_result() {
    let child_client = Script::new(vec![Script::answer("Oslo is 3C.")]);
    let child = Agent::new(child_client.clone()).system_prompt("You are a researcher.");
    let tool =
        AgentTool::new("research", "Research a question.", child).effect(ToolEffect::ReadOnly);
    assert_eq!(tool.name(), "research");
    assert_eq!(Tool::effect(&tool), ToolEffect::ReadOnly);
    assert_eq!(tool.input_schema()["required"], json!(["task"]));

    let parent_client = Script::new(vec![
        Script::call("c1", "research", json!({"task": "Weather in Oslo?"})),
        Script::answer("It is 3C in Oslo."),
    ]);
    let parent = Agent::new(parent_client.clone()).tool(Arc::new(tool));
    let outcome = parent.run("Oslo weather?").await.unwrap();
    assert_eq!(outcome.text(), Some("It is 3C in Oslo."));
    // The child saw only the task, not the parent's conversation.
    let child_request = &child_client.requests()[0];
    assert_eq!(child_request.messages.len(), 2);
    assert!(matches!(
        &child_request.messages[1].content[0],
        crate::client::ContentPart::Text(t) if t == "Weather in Oslo?"
    ));
}

#[tokio::test]
async fn bad_input_and_stopped_children_report_errors() {
    let child = Agent::new(Script::new(vec![Script::call("c", "missing", json!({}))])).max_steps(1);
    let tool = AgentTool::new("helper", "Helps.", child);
    let ctx = ToolContext::detached("c");
    assert!(tool.execute(json!({"task": "  "}), &ctx).await.is_err());
    let out = tool.execute(json!({"task": "go"}), &ctx).await.unwrap();
    assert_eq!(out["error"], json!("the helper stopped before it finished"));
    assert_eq!(out["outcome"]["status"], json!("budget_exhausted"));
}
