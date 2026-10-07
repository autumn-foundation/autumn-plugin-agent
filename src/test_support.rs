//! Shared test doubles. Compiled only for tests.

#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use std::collections::VecDeque;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex};

use crate::client::{ChatRequest, ChatResponse, ContentPart, LlmClient, StopReason, TokenUsage};
use crate::error::AgentError;

/// A scripted [`LlmClient`]: pops one response per call and records requests.
#[derive(Debug, Default)]
pub(crate) struct Script {
    responses: Mutex<VecDeque<ChatResponse>>,
    requests: Mutex<Vec<ChatRequest>>,
}

impl Script {
    pub(crate) fn new(responses: Vec<ChatResponse>) -> Arc<Self> {
        Arc::new(Self {
            responses: Mutex::new(responses.into()),
            requests: Mutex::new(Vec::new()),
        })
    }

    pub(crate) fn requests(&self) -> Vec<ChatRequest> {
        self.requests.lock().unwrap().clone()
    }

    pub(crate) fn answer(text: &str) -> ChatResponse {
        ChatResponse {
            content: vec![ContentPart::Text(text.to_owned())],
            stop_reason: StopReason::EndTurn,
            usage: TokenUsage::new(10, 5),
        }
    }

    pub(crate) fn call(id: &str, name: &str, arguments: serde_json::Value) -> ChatResponse {
        ChatResponse {
            content: vec![ContentPart::ToolCall {
                id: id.to_owned(),
                name: name.to_owned(),
                arguments,
            }],
            stop_reason: StopReason::ToolUse,
            usage: TokenUsage::new(10, 5),
        }
    }
}

impl LlmClient for Script {
    fn chat<'a>(
        &'a self,
        request: &'a ChatRequest,
    ) -> Pin<Box<dyn Future<Output = Result<ChatResponse, AgentError>> + Send + 'a>> {
        self.requests.lock().unwrap().push(request.clone());
        let next = self
            .responses
            .lock()
            .unwrap()
            .pop_front()
            .unwrap_or_else(|| Self::answer("default"));
        Box::pin(async move { Ok(next) })
    }

    fn list_models(
        &self,
    ) -> Pin<Box<dyn Future<Output = Result<Vec<String>, AgentError>> + Send + '_>> {
        Box::pin(async { Ok(vec!["script".to_owned()]) })
    }

    fn provider_name(&self) -> &'static str {
        "script"
    }
}
