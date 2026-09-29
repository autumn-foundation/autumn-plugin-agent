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
    let out = tool.execute(json!({"hi": 1})).await.unwrap();
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
    let err = tool.execute(json!({})).await.unwrap_err();
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
