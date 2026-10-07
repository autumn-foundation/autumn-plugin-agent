#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::net::SocketAddr;

use axum::routing::{get, post};
use axum::{Json, Router};
use serde_json::{Value, json};

use super::*;

/// Spin up an axum mock provider and return its base URL.
async fn mock_server(router: Router) -> String {
    let listener = tokio::net::TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr: SocketAddr = listener.local_addr().unwrap();
    tokio::spawn(async move {
        axum::serve(listener, router).await.unwrap();
    });
    format!("http://{addr}")
}

/// OpenAI-compatible mock: first call asks for a tool, second answers.
fn openai_router() -> Router {
    async fn chat(Json(_body): Json<Value>) -> Json<Value> {
        Json(json!({
            "id": "chatcmpl-test",
            "choices": [{
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": null,
                    "tool_calls": [{
                        "id": "call_1",
                        "type": "function",
                        "function": {
                            "name": "get_weather",
                            "arguments": "{\"city\":\"Chicago\"}"
                        }
                    }]
                },
                "finish_reason": "tool_calls"
            }],
            "usage": {"prompt_tokens": 42, "completion_tokens": 17}
        }))
    }
    async fn models() -> Json<Value> {
        Json(json!({"data": [{"id": "gpt-4o-mini"}, {"id": "gpt-4o"}]}))
    }
    Router::new()
        .route("/chat/completions", post(chat))
        .route("/models", get(models))
}

/// Anthropic mock: one `tool_use` block, then usage.
fn anthropic_router() -> Router {
    async fn messages(Json(_body): Json<Value>) -> Json<Value> {
        Json(json!({
            "id": "msg_test",
            "type": "message",
            "role": "assistant",
            "content": [
                {"type": "text", "text": "Checking the sky."},
                {"type": "tool_use", "id": "toolu_1", "name": "get_weather",
                 "input": {"city": "Chicago"}}
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 30, "output_tokens": 12}
        }))
    }
    async fn models() -> Json<Value> {
        Json(json!({"data": [{"id": "claude-sonnet-4-5"}]}))
    }
    Router::new()
        .route("/v1/messages", post(messages))
        .route("/v1/models", get(models))
}

fn tool_def() -> ToolDefinition {
    ToolDefinition {
        name: "get_weather".to_owned(),
        description: "Current weather for a city.".to_owned(),
        input_schema: json!({
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"]
        }),
    }
}

fn chat_request() -> ChatRequest {
    ChatRequest {
        messages: vec![
            ChatMessage::text(ChatRole::System, "You are a weather bot."),
            ChatMessage::text(ChatRole::User, "Weather in Chicago?"),
        ],
        tools: vec![tool_def()],
        max_tokens: Some(512),
        temperature: None,
    }
}

#[tokio::test]
async fn openai_tool_call_round_trip() {
    let base = mock_server(openai_router()).await;
    let client = OpenAiCompatibleClient::new(base, "gpt-4o-mini", "key").unwrap();
    let response = client.chat(&chat_request()).await.unwrap();
    assert_eq!(response.stop_reason, StopReason::ToolUse);
    assert_eq!(response.usage.input_tokens, 42);
    assert_eq!(response.usage.output_tokens, 17);
    let call = response
        .content
        .iter()
        .find_map(|part| match part {
            ContentPart::ToolCall {
                id,
                name,
                arguments,
            } => Some((id, name, arguments)),
            _ => None,
        })
        .expect("expected a tool call part");
    assert_eq!(call.0, "call_1");
    assert_eq!(call.1, "get_weather");
    assert_eq!(call.2["city"], Value::String("Chicago".to_owned()));
}

#[tokio::test]
async fn openai_text_answer_decodes() {
    async fn chat(Json(_body): Json<Value>) -> Json<Value> {
        Json(json!({
            "choices": [{
                "message": {"role": "assistant", "content": "Sunny, 72F."},
                "finish_reason": "stop"
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5}
        }))
    }
    let base = mock_server(Router::new().route("/chat/completions", post(chat))).await;
    let client = OpenAiCompatibleClient::new(base, "gpt-4o-mini", "key").unwrap();
    let response = client.chat(&chat_request()).await.unwrap();
    assert_eq!(response.stop_reason, StopReason::EndTurn);
    assert!(matches!(
        response.content.first(),
        Some(ContentPart::Text(text)) if text == "Sunny, 72F."
    ));
}

#[tokio::test]
async fn openai_errors_map_to_kinds() {
    for (status, kind) in [
        (
            axum::http::StatusCode::UNAUTHORIZED,
            ErrorKind::Authentication,
        ),
        (
            axum::http::StatusCode::TOO_MANY_REQUESTS,
            ErrorKind::RateLimited,
        ),
        (
            axum::http::StatusCode::INTERNAL_SERVER_ERROR,
            ErrorKind::Provider,
        ),
    ] {
        let router = Router::new().route(
            "/chat/completions",
            post(move || async move {
                (
                    status,
                    Json(json!({"error": {"message": "nope", "type": "test"}})),
                )
            }),
        );
        let base = mock_server(router).await;
        let client = OpenAiCompatibleClient::new(base, "gpt-4o-mini", "key").unwrap();
        let err = client.chat(&chat_request()).await.unwrap_err();
        assert_eq!(err.kind(), kind, "status {status}");
    }
}

#[tokio::test]
async fn openai_list_models() {
    let base = mock_server(openai_router()).await;
    let client = OpenAiCompatibleClient::new(base, "gpt-4o-mini", "key").unwrap();
    let models = client.list_models().await.unwrap();
    assert_eq!(models, vec!["gpt-4o-mini", "gpt-4o"]);
    assert_eq!(client.provider_name(), "openai-compatible");
}

#[tokio::test]
async fn openai_request_shape() {
    // Capture the wire body and assert the OpenAI protocol shape.
    async fn chat(Json(body): Json<Value>) -> Json<Value> {
        assert_eq!(body["model"], Value::String("gpt-4o-mini".to_owned()));
        assert_eq!(body["tool_choice"], Value::String("auto".to_owned()));
        let tool = &body["tools"][0];
        assert_eq!(tool["type"], Value::String("function".to_owned()));
        assert_eq!(
            tool["function"]["name"],
            Value::String("get_weather".to_owned())
        );
        assert_eq!(
            body["messages"][0]["role"],
            Value::String("system".to_owned())
        );
        Json(json!({
            "choices": [{"message": {"role": "assistant", "content": "ok"},
                         "finish_reason": "stop"}]
        }))
    }
    let base = mock_server(Router::new().route("/chat/completions", post(chat))).await;
    let client = OpenAiCompatibleClient::new(base, "gpt-4o-mini", "key").unwrap();
    client.chat(&chat_request()).await.unwrap();
}

#[tokio::test]
async fn anthropic_tool_use_round_trip() {
    let base = mock_server(anthropic_router()).await;
    let client = AnthropicClient::new(base, "claude-sonnet-4-5", "key").unwrap();
    let response = client.chat(&chat_request()).await.unwrap();
    assert_eq!(response.stop_reason, StopReason::ToolUse);
    assert_eq!(response.usage.input_tokens, 30);
    assert_eq!(response.usage.output_tokens, 12);
    assert!(matches!(
        &response.content[0],
        ContentPart::Text(text) if text == "Checking the sky."
    ));
    assert!(matches!(
        &response.content[1],
        ContentPart::ToolCall { id, name, arguments }
            if id == "toolu_1" && name == "get_weather" && arguments["city"] == json!("Chicago")
    ));
    assert_eq!(client.provider_name(), "anthropic");
}

#[tokio::test]
async fn anthropic_request_shape() {
    // Axum requires handlers to be `async fn`; this one only asserts, so the
    // `async` is structural.
    #[allow(clippy::unused_async)]
    async fn messages(Json(body): Json<Value>) -> Json<Value> {
        assert_eq!(body["model"], Value::String("claude-sonnet-4-5".to_owned()));
        assert!(body["max_tokens"].as_u64().unwrap() > 0);
        assert_eq!(
            body["system"],
            Value::String("You are a weather bot.".to_owned())
        );
        assert_eq!(
            body["messages"][0]["role"],
            Value::String("user".to_owned())
        );
        let tool = &body["tools"][0];
        assert_eq!(tool["name"], Value::String("get_weather".to_owned()));
        assert!(tool["input_schema"].is_object());
        Json(json!({
            "content": [{"type": "text", "text": "done"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 1, "output_tokens": 1}
        }))
    }
    async fn authed(headers: axum::http::HeaderMap, Json(body): Json<Value>) -> Json<Value> {
        assert_eq!(
            headers["x-api-key"],
            axum::http::HeaderValue::from_static("secret-key")
        );
        assert_eq!(
            headers["anthropic-version"],
            axum::http::HeaderValue::from_static(AnthropicClient::API_VERSION)
        );
        messages(Json(body)).await
    }
    let base = mock_server(Router::new().route("/v1/messages", post(authed))).await;
    let client = AnthropicClient::new(base, "claude-sonnet-4-5", "secret-key")
        .unwrap()
        .with_prompt_caching(false);
    let response = client.chat(&chat_request()).await.unwrap();
    assert_eq!(response.stop_reason, StopReason::EndTurn);
}

#[tokio::test]
async fn anthropic_prompt_caching_marks_breakpoints_and_counts_cache_tokens() {
    // Axum requires handlers to be `async fn`; this one only asserts.
    #[allow(clippy::unused_async)]
    async fn messages(Json(body): Json<Value>) -> Json<Value> {
        let ephemeral = json!({"type": "ephemeral"});
        assert_eq!(body["system"][0]["type"], json!("text"));
        assert_eq!(body["system"][0]["text"], json!("You are a weather bot."));
        assert_eq!(body["system"][0]["cache_control"], ephemeral);
        let tools = body["tools"].as_array().unwrap();
        assert_eq!(tools.last().unwrap()["cache_control"], ephemeral);
        let messages = body["messages"].as_array().unwrap();
        let last_block = messages.last().unwrap()["content"]
            .as_array()
            .unwrap()
            .last()
            .unwrap();
        assert_eq!(last_block["cache_control"], ephemeral);
        Json(json!({
            "content": [{"type": "text", "text": "done"}],
            "stop_reason": "end_turn",
            "usage": {
                "input_tokens": 10,
                "cache_creation_input_tokens": 100,
                "cache_read_input_tokens": 1000,
                "output_tokens": 5
            }
        }))
    }
    let base = mock_server(Router::new().route("/v1/messages", post(messages))).await;
    let client = AnthropicClient::new(base, "claude-sonnet-4-5", "key").unwrap();
    assert!(format!("{client:?}").contains("prompt_caching: true"));
    let response = client.chat(&chat_request()).await.unwrap();
    assert_eq!(response.usage.input_tokens, 1_110);
    assert_eq!(response.usage.cache_read_tokens, 1_000);
    assert_eq!(response.usage.cache_write_tokens, 100);
    assert_eq!(response.usage.output_tokens, 5);
}

#[tokio::test]
async fn openai_reports_cached_prompt_tokens() {
    async fn chat() -> Json<Value> {
        Json(json!({
            "choices": [{"message": {"role": "assistant", "content": "ok"},
                         "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 2000, "completion_tokens": 7,
                      "prompt_tokens_details": {"cached_tokens": 1536}}
        }))
    }
    let base = mock_server(Router::new().route("/chat/completions", post(chat))).await;
    let client = OpenAiCompatibleClient::new(base, "gpt-4o-mini", "key").unwrap();
    let response = client.chat(&chat_request()).await.unwrap();
    assert_eq!(response.usage.input_tokens, 2_000);
    assert_eq!(response.usage.cache_read_tokens, 1_536);
    assert_eq!(response.usage.cache_write_tokens, 0);
}

#[test]
fn transcripts_round_trip_through_json() {
    let messages = vec![
        ChatMessage::text(ChatRole::User, "hi"),
        ChatMessage {
            role: ChatRole::Assistant,
            content: vec![
                ContentPart::Text("checking".to_owned()),
                ContentPart::ToolCall {
                    id: "c1".to_owned(),
                    name: "get_weather".to_owned(),
                    arguments: json!({"city": "Oslo"}),
                },
            ],
        },
        ChatMessage {
            role: ChatRole::Tool,
            content: vec![ContentPart::ToolResult {
                tool_call_id: "c1".to_owned(),
                content: "{\"temp\":3}".to_owned(),
            }],
        },
    ];
    let text = serde_json::to_string(&messages).unwrap();
    assert!(text.contains("\"tool_call\""), "{text}");
    let back: Vec<ChatMessage> = serde_json::from_str(&text).unwrap();
    assert_eq!(back, messages);
}

#[tokio::test]
async fn anthropic_auth_failure_maps() {
    let router = Router::new().route(
        "/v1/messages",
        post(|| async {
            (
                axum::http::StatusCode::UNAUTHORIZED,
                Json(json!({"type": "error", "error": {"type": "authentication_error"}})),
            )
        }),
    );
    let base = mock_server(router).await;
    let client = AnthropicClient::new(base, "claude-sonnet-4-5", "bad").unwrap();
    let err = client.chat(&chat_request()).await.unwrap_err();
    assert_eq!(err.kind(), ErrorKind::Authentication);
}

#[tokio::test]
async fn anthropic_list_models() {
    let base = mock_server(anthropic_router()).await;
    let client = AnthropicClient::new(base, "claude-sonnet-4-5", "key").unwrap();
    let models = client.list_models().await.unwrap();
    assert_eq!(models, vec!["claude-sonnet-4-5"]);
}

#[test]
fn debug_redacts_api_keys() {
    let openai = OpenAiCompatibleClient::new("http://x", "m", "sk-live-secret").unwrap();
    let debug = format!("{openai:?}");
    assert!(!debug.contains("sk-live-secret"), "{debug}");
    assert!(debug.contains("<redacted>"), "{debug}");

    let anthropic = AnthropicClient::new("http://x", "m", "sk-ant-secret").unwrap();
    let debug = format!("{anthropic:?}");
    assert!(!debug.contains("sk-ant-secret"), "{debug}");
}

#[test]
fn client_from_config_picks_provider() {
    // The `_with_key` seam takes the key explicitly so the test never reads
    // or mutates the process environment.
    let mut config = AgentConfig::default();
    config.provider = crate::config::ProviderKind::Anthropic;
    let client = client_from_config_with_key(&config, "test-key").unwrap();
    assert_eq!(client.provider_name(), "anthropic");

    config.provider = crate::config::ProviderKind::OpenAiCompatible;
    let client = client_from_config_with_key(&config, "test-key").unwrap();
    assert_eq!(client.provider_name(), "openai-compatible");
}

#[test]
fn token_usage_saturates() {
    let usage = TokenUsage {
        input_tokens: u32::MAX,
        output_tokens: u32::MAX,
        cache_read_tokens: u32::MAX,
        cache_write_tokens: 1,
    };
    let sum = usage.saturating_add(usage);
    assert_eq!(sum.total(), u32::MAX);
    assert_eq!(sum.cache_read_tokens, u32::MAX);
    assert_eq!(sum.cache_write_tokens, 2);
}
