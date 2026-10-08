//! Provider clients: one trait, two HTTP implementations.
//!
//! [`LlmClient`] speaks chat completions with tool calls. The two
//! implementations speak HTTP directly with `reqwest` — no heavyweight LLM
//! SDK sits in between:
//!
//! * [`OpenAiCompatibleClient`] — OpenAI chat-completions protocol. Works
//!   with OpenAI, Ollama, vLLM, LM Studio, and any other
//!   endpoint that answers `POST /chat/completions`.
//! * [`AnthropicClient`] — Anthropic Messages API (`POST /v1/messages`).
//!
//! Build the right one from config with [`client_from_config`]. Both clients
//! redact the API key from their `Debug` output.

use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use serde::{Deserialize, Serialize};

use crate::config::AgentConfig;
use crate::error::{AgentError, ErrorKind};

/// A single turn participant.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ChatRole {
    /// Provider instructions.
    System,
    /// The human (or the harness).
    User,
    /// The model.
    Assistant,
    /// Tool execution results.
    Tool,
}

/// One piece of a [`ChatMessage`].
// `serde_json::Value` has no `Eq` impl (JSON numbers may be floats), so this
// enum cannot be `Eq` either.
//
// Serialized externally tagged (`{"text": "..."}`, `{"tool_call": {..}}`) so
// session stores can persist transcripts as JSON.
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ContentPart {
    /// Plain text.
    Text(String),
    /// The model wants a tool executed.
    ToolCall {
        /// Provider-assigned call id, echoed back with the result.
        id: String,
        /// Tool name.
        name: String,
        /// Decoded JSON arguments.
        arguments: serde_json::Value,
    },
    /// The outcome of one tool execution.
    ToolResult {
        /// The `id` of the [`ContentPart::ToolCall`] this answers.
        tool_call_id: String,
        /// JSON-serialized tool output.
        content: String,
    },
}

/// One chat message: a role plus its content parts.
///
/// Serializable, so a [`SessionStore`](crate::session::SessionStore) can keep
/// transcripts in a database column.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ChatMessage {
    /// Who sent the message.
    pub role: ChatRole,
    /// The message payload.
    pub content: Vec<ContentPart>,
}

impl ChatMessage {
    /// Build a single-text message.
    #[must_use]
    pub fn text(role: ChatRole, text: impl Into<String>) -> Self {
        Self {
            role,
            content: vec![ContentPart::Text(text.into())],
        }
    }
}

/// A tool the model may call, in provider-neutral form.
// `input_schema` is a `serde_json::Value`, which has no `Eq` impl.
#[allow(clippy::derive_partial_eq_without_eq)]
#[derive(Debug, Clone, PartialEq)]
pub struct ToolDefinition {
    /// Tool name as the model calls it.
    pub name: String,
    /// What the tool does, shown to the model.
    pub description: String,
    /// JSON Schema for the tool's input object.
    pub input_schema: serde_json::Value,
}

/// One chat-completions request.
#[derive(Debug, Clone, PartialEq)]
pub struct ChatRequest {
    /// Full conversation history.
    pub messages: Vec<ChatMessage>,
    /// Tools the model may call this turn.
    pub tools: Vec<ToolDefinition>,
    /// Per-call output cap. Providers may require this (Anthropic does).
    pub max_tokens: Option<u32>,
    /// Sampling temperature, when the caller wants one.
    pub temperature: Option<f32>,
}

/// Why the model stopped.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StopReason {
    /// The model finished its turn.
    EndTurn,
    /// The model emitted tool calls.
    ToolUse,
    /// The model hit an output cap.
    MaxTokens,
    /// The provider sent a reason this client does not recognise.
    Unknown,
}

/// Token counts reported by the provider for one call.
///
/// `input_tokens` is the full prompt size, cached or not. The two cache
/// fields are subsets of it: they tell you how much of the prompt the
/// provider served from (or wrote to) its prompt cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub struct TokenUsage {
    /// Prompt tokens consumed, including cached tokens.
    pub input_tokens: u32,
    /// Completion tokens produced.
    pub output_tokens: u32,
    /// Prompt tokens read from the provider's prompt cache.
    #[serde(default)]
    pub cache_read_tokens: u32,
    /// Prompt tokens written to the provider's prompt cache.
    #[serde(default)]
    pub cache_write_tokens: u32,
}

impl TokenUsage {
    /// Usage with uncached input and output counts.
    #[must_use]
    pub const fn new(input_tokens: u32, output_tokens: u32) -> Self {
        Self {
            input_tokens,
            output_tokens,
            cache_read_tokens: 0,
            cache_write_tokens: 0,
        }
    }

    /// Add two usages without overflowing.
    #[must_use]
    pub const fn saturating_add(self, other: Self) -> Self {
        Self {
            input_tokens: self.input_tokens.saturating_add(other.input_tokens),
            output_tokens: self.output_tokens.saturating_add(other.output_tokens),
            cache_read_tokens: self
                .cache_read_tokens
                .saturating_add(other.cache_read_tokens),
            cache_write_tokens: self
                .cache_write_tokens
                .saturating_add(other.cache_write_tokens),
        }
    }

    /// Total tokens both directions.
    #[must_use]
    pub const fn total(self) -> u32 {
        self.input_tokens.saturating_add(self.output_tokens)
    }
}

/// One chat-completions response.
#[derive(Debug, Clone, PartialEq)]
pub struct ChatResponse {
    /// The model's content parts (text and/or tool calls).
    pub content: Vec<ContentPart>,
    /// Why the model stopped.
    pub stop_reason: StopReason,
    /// Provider-reported token counts.
    pub usage: TokenUsage,
}

/// Provider-agnostic chat-completions client.
///
/// Object-safe: implementations box their futures by hand so the trait stays
/// usable behind `Arc<dyn LlmClient>`.
pub trait LlmClient: Send + Sync + std::fmt::Debug {
    /// Send one chat request and decode the response.
    fn chat<'a>(
        &'a self,
        request: &'a ChatRequest,
    ) -> Pin<Box<dyn Future<Output = Result<ChatResponse, AgentError>> + Send + 'a>>;

    /// List available model ids. Cheap and read-only: the health check uses it.
    fn list_models(
        &self,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<String>, AgentError>> + Send + '_>>;

    /// Human-readable provider name for logs and health details.
    fn provider_name(&self) -> &'static str;
}

/// Build the configured provider client.
///
/// Reads the API key from `AGENT_API_KEY` through
/// [`AgentConfig::api_key`]; fails when the key is missing.
///
/// # Errors
///
/// Returns [`AgentError`] with [`ErrorKind::Config`]
/// when `AGENT_API_KEY` is not set, or with
/// [`ErrorKind::Transport`] when the HTTP
/// client cannot be built.
pub fn client_from_config(config: &AgentConfig) -> Result<Arc<dyn LlmClient>, AgentError> {
    let api_key = config.api_key()?;
    client_from_config_with_key(config, &api_key)
}

/// Build the configured client with an explicitly provided API key.
///
/// Test seam for [`client_from_config`]: avoids reading `AGENT_API_KEY` from
/// the process environment.
pub(crate) fn client_from_config_with_key(
    config: &AgentConfig,
    api_key: &str,
) -> Result<Arc<dyn LlmClient>, AgentError> {
    let timeout = Duration::from_secs(config.request_timeout_secs);
    let model = config.resolved_model().to_owned();
    let base_url = config.resolved_base_url().to_owned();
    let client: Arc<dyn LlmClient> = match config.provider {
        crate::config::ProviderKind::OpenAiCompatible => {
            Arc::new(OpenAiCompatibleClient::new(base_url, model, api_key)?.with_timeout(timeout)?)
        }
        crate::config::ProviderKind::Anthropic => Arc::new(
            AnthropicClient::new(base_url, model, api_key)?
                .with_timeout(timeout)?
                .with_prompt_caching(config.prompt_caching),
        ),
    };
    Ok(client)
}

/// Check an HTTP status and translate failures into [`AgentError`].
async fn check_status(
    response: reqwest::Response,
    provider: &'static str,
) -> Result<reqwest::Response, AgentError> {
    let status = response.status();
    if status.is_success() {
        return Ok(response);
    }
    let body: String = response.text().await.unwrap_or_default();
    let snippet: String = body.chars().take(500).collect();
    let message = format!("{provider} answered {status}: {snippet}");
    let kind = match status {
        reqwest::StatusCode::UNAUTHORIZED | reqwest::StatusCode::FORBIDDEN => {
            ErrorKind::Authentication
        }
        reqwest::StatusCode::TOO_MANY_REQUESTS => ErrorKind::RateLimited,
        reqwest::StatusCode::PAYLOAD_TOO_LARGE => ErrorKind::Budget,
        // A timeout, a server fault, or an overload (Anthropic sends 529).
        // The request did not run, so a retry can succeed.
        reqwest::StatusCode::REQUEST_TIMEOUT => ErrorKind::Unavailable,
        status if status.is_server_error() => ErrorKind::Unavailable,
        _ => ErrorKind::Provider,
    };
    Err(AgentError::new(kind, message))
}

/// Extract model ids from a provider's `{"data": [{"id": ...}]}` envelope.
fn model_ids(payload: &serde_json::Value) -> Vec<String> {
    payload
        .get("data")
        .and_then(serde_json::Value::as_array)
        .map(|items| {
            items
                .iter()
                .filter_map(|item| item.get("id"))
                .filter_map(serde_json::Value::as_str)
                .map(str::to_owned)
                .collect()
        })
        .unwrap_or_default()
}

/// Read one non-negative integer usage field, defaulting to 0.
fn usage_field(usage: &serde_json::Value, key: &str) -> u32 {
    usage
        .get(key)
        .and_then(serde_json::Value::as_u64)
        .map_or(0, |n| u32::try_from(n).unwrap_or(u32::MAX))
}

/// Build a `reqwest` client with a timeout.
fn http_client(timeout: Duration) -> Result<reqwest::Client, AgentError> {
    reqwest::Client::builder()
        .timeout(timeout)
        .build()
        .map_err(|err| {
            AgentError::with_source(ErrorKind::Transport, "cannot build HTTP client", err)
        })
}

/// Client for the OpenAI chat-completions protocol.
///
/// Also serves every OpenAI-compatible endpoint (Ollama, vLLM, LM Studio,
/// a compatible proxy) — point `base_url` at it and keep the protocol.
pub struct OpenAiCompatibleClient {
    http: reqwest::Client,
    base_url: String,
    model: String,
    api_key: String,
}

impl std::fmt::Debug for OpenAiCompatibleClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("OpenAiCompatibleClient")
            .field("base_url", &self.base_url)
            .field("model", &self.model)
            .field("api_key", &"<redacted>")
            .finish_non_exhaustive()
    }
}

impl OpenAiCompatibleClient {
    /// Build a client. The key never leaves this struct except as a header.
    ///
    /// # Errors
    ///
    /// Returns [`AgentError`] when the underlying HTTP client cannot be built.
    pub fn new(
        base_url: impl Into<String>,
        model: impl Into<String>,
        api_key: impl Into<String>,
    ) -> Result<Self, AgentError> {
        Ok(Self {
            http: http_client(Duration::from_secs(60))?,
            base_url: base_url.into().trim_end_matches('/').to_owned(),
            model: model.into(),
            api_key: api_key.into(),
        })
    }

    /// Replace the request timeout.
    ///
    /// # Errors
    ///
    /// Returns [`AgentError`] when the underlying HTTP client cannot be rebuilt.
    pub fn with_timeout(mut self, timeout: Duration) -> Result<Self, AgentError> {
        self.http = http_client(timeout)?;
        Ok(self)
    }

    /// Chat endpoint URL.
    #[must_use]
    pub fn chat_url(&self) -> String {
        format!("{}/chat/completions", self.base_url)
    }

    async fn chat_inner(&self, request: &ChatRequest) -> Result<ChatResponse, AgentError> {
        let mut body = serde_json::json!({
            "model": self.model,
            "messages": openai_messages(&request.messages),
            "tools": request.tools.iter().map(openai_tool).collect::<Vec<_>>(),
        });
        if request.tools.is_empty() {
            if let Some(obj) = body.as_object_mut() {
                obj.remove("tools");
            }
        } else {
            body["tool_choice"] = serde_json::Value::String("auto".to_owned());
        }
        if let Some(max_tokens) = request.max_tokens {
            body["max_tokens"] = serde_json::json!(max_tokens);
        }
        if let Some(temperature) = request.temperature {
            body["temperature"] = serde_json::json!(temperature);
        }
        let response = self
            .http
            .post(self.chat_url())
            .bearer_auth(&self.api_key)
            .json(&body)
            .send()
            .await
            .map_err(AgentError::from)?;
        let response = check_status(response, "openai-compatible").await?;
        let payload: serde_json::Value = response.json().await.map_err(AgentError::from)?;
        openai_response(&payload)
    }
}

impl LlmClient for OpenAiCompatibleClient {
    fn chat<'a>(
        &'a self,
        request: &'a ChatRequest,
    ) -> Pin<Box<dyn Future<Output = Result<ChatResponse, AgentError>> + Send + 'a>> {
        Box::pin(self.chat_inner(request))
    }

    fn list_models(
        &self,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<String>, AgentError>> + Send + '_>> {
        Box::pin(async move {
            let response = self
                .http
                .get(format!("{}/models", self.base_url))
                .bearer_auth(&self.api_key)
                .send()
                .await
                .map_err(AgentError::from)?;
            let response = check_status(response, "openai-compatible").await?;
            let payload: serde_json::Value = response.json().await.map_err(AgentError::from)?;
            Ok(model_ids(&payload))
        })
    }

    fn provider_name(&self) -> &'static str {
        "openai-compatible"
    }
}

/// Convert provider-neutral messages to the OpenAI wire format.
fn openai_messages(messages: &[ChatMessage]) -> Vec<serde_json::Value> {
    let mut out = Vec::new();
    for message in messages {
        match message.role {
            ChatRole::System | ChatRole::User => {
                let role = match message.role {
                    ChatRole::System => "system",
                    _ => "user",
                };
                let text = message
                    .content
                    .iter()
                    .filter_map(|part| match part {
                        ContentPart::Text(text) => Some(text.as_str()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("");
                out.push(serde_json::json!({"role": role, "content": text}));
            }
            ChatRole::Assistant => {
                let text = message
                    .content
                    .iter()
                    .filter_map(|part| match part {
                        ContentPart::Text(text) => Some(text.as_str()),
                        _ => None,
                    })
                    .collect::<Vec<_>>()
                    .join("");
                let tool_calls = message
                    .content
                    .iter()
                    .filter_map(|part| match part {
                        ContentPart::ToolCall {
                            id,
                            name,
                            arguments,
                        } => Some(serde_json::json!({
                            "id": id,
                            "type": "function",
                            "function": {
                                "name": name,
                                "arguments": arguments.to_string(),
                            }
                        })),
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                let mut value = serde_json::json!({"role": "assistant", "content": text});
                if !tool_calls.is_empty() {
                    value["tool_calls"] = serde_json::Value::Array(tool_calls);
                }
                out.push(value);
            }
            ChatRole::Tool => {
                for part in &message.content {
                    if let ContentPart::ToolResult {
                        tool_call_id,
                        content,
                    } = part
                    {
                        out.push(serde_json::json!({
                            "role": "tool",
                            "tool_call_id": tool_call_id,
                            "content": content,
                        }));
                    }
                }
            }
        }
    }
    out
}

/// Convert a tool definition to the OpenAI `{"type": "function", ...}` shape.
fn openai_tool(tool: &ToolDefinition) -> serde_json::Value {
    serde_json::json!({
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            "parameters": tool.input_schema,
        }
    })
}

/// Decode an OpenAI chat-completions response.
fn openai_response(payload: &serde_json::Value) -> Result<ChatResponse, AgentError> {
    let choice = payload
        .get("choices")
        .and_then(serde_json::Value::as_array)
        .and_then(|choices| choices.first())
        .ok_or_else(|| {
            AgentError::new(
                ErrorKind::Decode,
                "openai-compatible response has no choices",
            )
        })?;
    let message = choice.get("message").ok_or_else(|| {
        AgentError::new(ErrorKind::Decode, "openai-compatible choice has no message")
    })?;
    let mut content = Vec::new();
    if let Some(text) = message.get("content").and_then(serde_json::Value::as_str)
        && !text.is_empty()
    {
        content.push(ContentPart::Text(text.to_owned()));
    }
    if let Some(calls) = message
        .get("tool_calls")
        .and_then(serde_json::Value::as_array)
    {
        for call in calls {
            let id = call
                .get("id")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default()
                .to_owned();
            let function = call.get("function").ok_or_else(|| {
                AgentError::new(ErrorKind::Decode, "tool_call has no function payload")
            })?;
            let name = function
                .get("name")
                .and_then(serde_json::Value::as_str)
                .unwrap_or_default()
                .to_owned();
            let raw_args = function
                .get("arguments")
                .and_then(serde_json::Value::as_str)
                .unwrap_or("{}");
            let arguments: serde_json::Value = serde_json::from_str(raw_args).map_err(|err| {
                AgentError::with_source(
                    ErrorKind::Decode,
                    "tool_call arguments are not valid JSON",
                    err,
                )
            })?;
            content.push(ContentPart::ToolCall {
                id,
                name,
                arguments,
            });
        }
    }
    let stop_reason = match choice
        .get("finish_reason")
        .and_then(serde_json::Value::as_str)
    {
        Some("tool_calls") => StopReason::ToolUse,
        Some("length") => StopReason::MaxTokens,
        Some("stop") => StopReason::EndTurn,
        _ => StopReason::Unknown,
    };
    // OpenAI caches long prompts on its own and reports the cached share
    // in `prompt_tokens_details.cached_tokens` (a subset of `prompt_tokens`).
    let usage = payload
        .get("usage")
        .map(|usage| TokenUsage {
            input_tokens: usage_field(usage, "prompt_tokens"),
            output_tokens: usage_field(usage, "completion_tokens"),
            cache_read_tokens: usage
                .get("prompt_tokens_details")
                .map_or(0, |details| usage_field(details, "cached_tokens")),
            cache_write_tokens: 0,
        })
        .unwrap_or_default();
    Ok(ChatResponse {
        content,
        stop_reason,
        usage,
    })
}

/// Client for the Anthropic Messages API.
///
/// Prompt caching is on by default: the client marks the tool list, the
/// system prompt, and the newest message as cache breakpoints, so each loop
/// step and each scheduled run re-reads the stable prefix from the cache.
pub struct AnthropicClient {
    http: reqwest::Client,
    base_url: String,
    model: String,
    api_key: String,
    prompt_caching: bool,
}

impl std::fmt::Debug for AnthropicClient {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AnthropicClient")
            .field("base_url", &self.base_url)
            .field("model", &self.model)
            .field("prompt_caching", &self.prompt_caching)
            .field("api_key", &"<redacted>")
            .finish_non_exhaustive()
    }
}

impl AnthropicClient {
    /// Anthropic API version header sent with every request.
    pub const API_VERSION: &'static str = "2023-06-01";

    /// Build a client. The key never leaves this struct except as a header.
    ///
    /// # Errors
    ///
    /// Returns [`AgentError`] when the underlying HTTP client cannot be built.
    pub fn new(
        base_url: impl Into<String>,
        model: impl Into<String>,
        api_key: impl Into<String>,
    ) -> Result<Self, AgentError> {
        Ok(Self {
            http: http_client(Duration::from_secs(60))?,
            base_url: base_url.into().trim_end_matches('/').to_owned(),
            model: model.into(),
            api_key: api_key.into(),
            prompt_caching: true,
        })
    }

    /// Replace the request timeout.
    ///
    /// # Errors
    ///
    /// Returns [`AgentError`] when the underlying HTTP client cannot be rebuilt.
    pub fn with_timeout(mut self, timeout: Duration) -> Result<Self, AgentError> {
        self.http = http_client(timeout)?;
        Ok(self)
    }

    /// Turn the `cache_control` breakpoints on or off.
    #[must_use]
    pub const fn with_prompt_caching(mut self, enabled: bool) -> Self {
        self.prompt_caching = enabled;
        self
    }

    /// Messages endpoint URL.
    #[must_use]
    pub fn messages_url(&self) -> String {
        format!("{}/v1/messages", self.base_url)
    }

    async fn chat_inner(&self, request: &ChatRequest) -> Result<ChatResponse, AgentError> {
        let (system, mut messages) = anthropic_messages(&request.messages);
        let mut tools = request.tools.iter().map(anthropic_tool).collect::<Vec<_>>();
        if self.prompt_caching {
            mark_cache_breakpoints(&mut tools, &mut messages);
        }
        let mut body = serde_json::json!({
            "model": self.model,
            "max_tokens": request.max_tokens.unwrap_or(1024),
            "messages": messages,
            "tools": tools,
        });
        if let Some(system) = system {
            body["system"] = if self.prompt_caching {
                serde_json::json!([{
                    "type": "text",
                    "text": system,
                    "cache_control": {"type": "ephemeral"},
                }])
            } else {
                serde_json::Value::String(system)
            };
        }
        if let Some(temperature) = request.temperature {
            body["temperature"] = serde_json::json!(temperature);
        }
        let response = self
            .http
            .post(self.messages_url())
            .header("x-api-key", &self.api_key)
            .header("anthropic-version", Self::API_VERSION)
            .json(&body)
            .send()
            .await
            .map_err(AgentError::from)?;
        let response = check_status(response, "anthropic").await?;
        let payload: serde_json::Value = response.json().await.map_err(AgentError::from)?;
        anthropic_response(&payload)
    }
}

impl LlmClient for AnthropicClient {
    fn chat<'a>(
        &'a self,
        request: &'a ChatRequest,
    ) -> Pin<Box<dyn Future<Output = Result<ChatResponse, AgentError>> + Send + 'a>> {
        Box::pin(self.chat_inner(request))
    }

    fn list_models(
        &self,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<String>, AgentError>> + Send + '_>> {
        Box::pin(async move {
            let response = self
                .http
                .get(format!("{}/v1/models", self.base_url))
                .header("x-api-key", &self.api_key)
                .header("anthropic-version", Self::API_VERSION)
                .send()
                .await
                .map_err(AgentError::from)?;
            let response = check_status(response, "anthropic").await?;
            let payload: serde_json::Value = response.json().await.map_err(AgentError::from)?;
            Ok(model_ids(&payload))
        })
    }

    fn provider_name(&self) -> &'static str {
        "anthropic"
    }
}

/// Split provider-neutral messages into Anthropic's top-level `system` plus
/// the `messages` array.
fn anthropic_messages(messages: &[ChatMessage]) -> (Option<String>, Vec<serde_json::Value>) {
    let mut system_parts = Vec::new();
    let mut out = Vec::new();
    for message in messages {
        match message.role {
            ChatRole::System => {
                for part in &message.content {
                    if let ContentPart::Text(text) = part {
                        system_parts.push(text.clone());
                    }
                }
            }
            ChatRole::User | ChatRole::Assistant | ChatRole::Tool => {
                let role = match message.role {
                    ChatRole::Assistant => "assistant",
                    _ => "user",
                };
                let blocks = message
                    .content
                    .iter()
                    .map(|part| match part {
                        ContentPart::Text(text) => {
                            serde_json::json!({"type": "text", "text": text})
                        }
                        ContentPart::ToolCall {
                            id,
                            name,
                            arguments,
                        } => serde_json::json!({
                            "type": "tool_use",
                            "id": id,
                            "name": name,
                            "input": arguments,
                        }),
                        ContentPart::ToolResult {
                            tool_call_id,
                            content,
                        } => serde_json::json!({
                            "type": "tool_result",
                            "tool_use_id": tool_call_id,
                            "content": content,
                        }),
                    })
                    .collect::<Vec<_>>();
                if !blocks.is_empty() {
                    out.push(serde_json::json!({"role": role, "content": blocks}));
                }
            }
        }
    }
    let system = if system_parts.is_empty() {
        None
    } else {
        Some(system_parts.join("\n"))
    };
    (system, out)
}

/// Mark the last tool and the last block of the newest message as
/// `cache_control` breakpoints.
///
/// With the system breakpoint that makes three of Anthropic's four allowed
/// breakpoints. The prefix up to each breakpoint is cached, so the next loop
/// step pays full price only for the new turn.
fn mark_cache_breakpoints(tools: &mut [serde_json::Value], messages: &mut [serde_json::Value]) {
    let ephemeral = serde_json::json!({"type": "ephemeral"});
    if let Some(serde_json::Value::Object(tool)) = tools.last_mut() {
        tool.insert("cache_control".to_owned(), ephemeral.clone());
    }
    if let Some(serde_json::Value::Object(block)) = messages
        .last_mut()
        .and_then(|message| message.get_mut("content"))
        .and_then(serde_json::Value::as_array_mut)
        .and_then(|blocks| blocks.last_mut())
    {
        block.insert("cache_control".to_owned(), ephemeral);
    }
}

/// Convert a tool definition to the Anthropic `input_schema` shape.
fn anthropic_tool(tool: &ToolDefinition) -> serde_json::Value {
    serde_json::json!({
        "name": tool.name,
        "description": tool.description,
        "input_schema": tool.input_schema,
    })
}

/// Decode an Anthropic Messages response.
fn anthropic_response(payload: &serde_json::Value) -> Result<ChatResponse, AgentError> {
    let blocks = payload
        .get("content")
        .and_then(serde_json::Value::as_array)
        .ok_or_else(|| {
            AgentError::new(
                ErrorKind::Decode,
                "anthropic response has no content blocks",
            )
        })?;
    let mut content = Vec::new();
    for block in blocks {
        let block_type = block.get("type").and_then(serde_json::Value::as_str);
        match block_type {
            Some("text") => {
                if let Some(text) = block.get("text").and_then(serde_json::Value::as_str) {
                    content.push(ContentPart::Text(text.to_owned()));
                }
            }
            Some("tool_use") => {
                let id = block
                    .get("id")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or_default()
                    .to_owned();
                let name = block
                    .get("name")
                    .and_then(serde_json::Value::as_str)
                    .unwrap_or_default()
                    .to_owned();
                let arguments = block
                    .get("input")
                    .cloned()
                    .unwrap_or(serde_json::Value::Null);
                content.push(ContentPart::ToolCall {
                    id,
                    name,
                    arguments,
                });
            }
            _ => {}
        }
    }
    let stop_reason = match payload
        .get("stop_reason")
        .and_then(serde_json::Value::as_str)
    {
        Some("tool_use") => StopReason::ToolUse,
        Some("max_tokens") => StopReason::MaxTokens,
        Some("end_turn") => StopReason::EndTurn,
        _ => StopReason::Unknown,
    };
    // Anthropic reports `input_tokens` without the cached share; add the
    // cache reads and writes back so `input_tokens` is the full prompt size
    // on both providers and budgets stay honest.
    let usage = payload
        .get("usage")
        .map(|usage| {
            let cache_read_tokens = usage_field(usage, "cache_read_input_tokens");
            let cache_write_tokens = usage_field(usage, "cache_creation_input_tokens");
            TokenUsage {
                input_tokens: usage_field(usage, "input_tokens")
                    .saturating_add(cache_read_tokens)
                    .saturating_add(cache_write_tokens),
                output_tokens: usage_field(usage, "output_tokens"),
                cache_read_tokens,
                cache_write_tokens,
            }
        })
        .unwrap_or_default();
    Ok(ChatResponse {
        content,
        stop_reason,
        usage,
    })
}

#[cfg(test)]
mod tests;
