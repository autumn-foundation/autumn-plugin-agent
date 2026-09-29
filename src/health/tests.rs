#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::net::SocketAddr;

use axum::routing::get;
use axum::{Json, Router};
use serde_json::{Value, json};

use super::*;
use crate::client::client_from_config_with_key;
use crate::config::AgentConfig;

/// Spin up an axum mock provider and return its base URL.
async fn mock_server(router: Router) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr: SocketAddr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    format!("http://{addr}")
}

/// Build a runtime around a mock provider, like the startup hook does.
async fn mock_runtime(router: Router) -> Arc<AgentRuntime> {
    let base = mock_server(router).await;
    let config = AgentConfig {
        base_url: Some(base),
        ..AgentConfig::default()
    };
    let client = client_from_config_with_key(&config, "test-key").unwrap();
    Arc::new(AgentRuntime::new(config, client, Vec::new()))
}

#[test]
fn group_is_health_only() {
    let indicator = AgentHealthIndicator::new(Arc::new(OnceLock::new()));
    assert_eq!(indicator.group(), IndicatorGroup::HealthOnly);
}

#[tokio::test]
async fn uninitialized_reports_down() {
    // Before the startup hook installs the runtime there is nothing to ping.
    let output = run_check(None).await;
    assert_eq!(output.status, HealthStatus::Down);
    assert!(output.details.contains_key("error"));
    assert!(!output.details.contains_key("provider"));
}

#[tokio::test]
async fn up_when_models_list() {
    async fn models() -> Json<Value> {
        Json(json!({"data": [{"id": "gpt-4o-mini"}, {"id": "gpt-4o"}]}))
    }
    let runtime = mock_runtime(Router::new().route("/models", get(models))).await;
    let output = run_check(Some(&runtime)).await;
    assert_eq!(output.status, HealthStatus::Up);
    assert_eq!(output.details["models"], serde_json::Value::from(2));
    assert_eq!(
        output.details["provider"],
        serde_json::Value::String("openai-compatible".to_owned())
    );
}

#[tokio::test]
async fn down_when_provider_errors() {
    async fn models() -> (axum::http::StatusCode, Json<Value>) {
        (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({"error": {"message": "boom"}})),
        )
    }
    let runtime = mock_runtime(Router::new().route("/models", get(models))).await;
    let output = run_check(Some(&runtime)).await;
    assert_eq!(output.status, HealthStatus::Down);
    assert!(output.details.contains_key("error"));
}

#[tokio::test]
async fn check_uses_the_shared_runtime() {
    // The indicator reads the same holder the startup hook installs: before
    // installation it is Down, after installation it pings the live client.
    async fn models() -> Json<Value> {
        Json(json!({"data": [{"id": "gpt-4o-mini"}]}))
    }
    let active = Arc::new(OnceLock::new());
    let indicator = AgentHealthIndicator::new(Arc::clone(&active));

    let before = indicator.check().await;
    assert_eq!(before.status, HealthStatus::Down);

    let runtime = mock_runtime(Router::new().route("/models", get(models))).await;
    active.set(runtime).unwrap();
    let after = indicator.check().await;
    assert_eq!(after.status, HealthStatus::Up);
}

#[test]
fn debug_reports_initialization_state() {
    let uninit = AgentHealthIndicator::new(Arc::new(OnceLock::new()));
    let debug = format!("{uninit:?}");
    assert!(debug.contains("AgentHealthIndicator"), "{debug}");
    assert!(debug.contains("initialized"), "{debug}");
}
