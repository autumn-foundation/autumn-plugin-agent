#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use super::*;
use crate::error::ErrorKind;
use serde_json::json;

#[tokio::test]
async fn fn_tool_executes() {
    let tool = FnTool::new(
        "echo",
        "Echoes its input.",
        json!({"type": "object"}),
        |input: serde_json::Value| async move { Ok(input) },
    );
    assert_eq!(tool.name(), "echo");
    assert_eq!(tool.description(), "Echoes its input.");
    assert_eq!(tool.input_schema(), json!({"type": "object"}));
    let out = tool
        .execute(json!({"hi": 1}), &ToolContext::detached("c1"))
        .await
        .unwrap();
    assert_eq!(out, json!({"hi": 1}));

    let def = tool.definition();
    assert_eq!(def.name, "echo");
    assert_eq!(def.description, "Echoes its input.");
}

#[tokio::test]
async fn fn_tool_propagates_domain_errors() {
    let tool = FnTool::new(
        "broken",
        "Always fails.",
        json!({"type": "object"}),
        |_input: serde_json::Value| async move { Err(AgentError::new(ErrorKind::Tool, "kaput")) },
    );
    let err = tool
        .execute(json!({}), &ToolContext::detached("c1"))
        .await
        .unwrap_err();
    assert_eq!(err.kind(), ErrorKind::Tool);
}

#[test]
fn fn_tool_debug_hides_the_closure() {
    let tool = FnTool::new("t", "d", json!({}), |input: serde_json::Value| async move {
        Ok(input)
    });
    let debug = format!("{tool:?}");
    assert!(debug.contains("FnTool"), "{debug}");
    assert!(debug.contains("\"t\""), "{debug}");
}

#[tokio::test]
async fn shared_tool_is_object_safe() {
    let tool: Arc<dyn Tool> =
        FnTool::new("t", "d", json!({}), |input: serde_json::Value| async move {
            Ok(input)
        })
        .shared();
    assert_eq!(tool.definition().name, "t");
}

#[tokio::test]
async fn context_tool_sees_the_call_context() {
    let tool = FnTool::with_context(
        "ctx",
        "Echoes the call id.",
        json!({"type": "object"}),
        |_input: serde_json::Value, ctx: ToolContext| async move {
            Ok(json!({"run": ctx.run_id.as_str(), "call": ctx.call_id, "step": ctx.step}))
        },
    );
    let ctx = ToolContext {
        run_id: RunId::new("run-1"),
        call_id: "call-9".to_owned(),
        session_id: Some(SessionId::new("s")),
        step: 3,
    };
    let out = tool.execute(json!({}), &ctx).await.unwrap();
    assert_eq!(out, json!({"run": "run-1", "call": "call-9", "step": 3}));
}

#[test]
fn effects_default_to_write_and_order_by_impact() {
    let tool = FnTool::new("t", "d", json!({}), |input: serde_json::Value| async move {
        Ok(input)
    });
    assert_eq!(Tool::effect(&tool), ToolEffect::Write);
    let tool = tool.effect(ToolEffect::ReadOnly);
    assert_eq!(Tool::effect(&tool), ToolEffect::ReadOnly);
    assert!(ToolEffect::ReadOnly < ToolEffect::Internal);
    assert!(ToolEffect::Internal < ToolEffect::Write);
    assert!(ToolEffect::Write < ToolEffect::External);
    assert_eq!(
        serde_json::to_value(ToolEffect::ReadOnly).unwrap(),
        json!("read_only")
    );
}
