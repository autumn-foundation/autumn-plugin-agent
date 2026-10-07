//! Sessions: persisted conversations that outlive one run.
//!
//! A session is an ordered transcript keyed by a [`SessionId`]. Each run in
//! a session loads the transcript, adds one turn, and saves it back, so the
//! agent remembers what was said yesterday. The app chooses the key: a chat
//! thread, an email chain, a user, or a standing job.
//!
//! Long sessions outgrow the context window. [`Compaction`] folds the older
//! part of a transcript into one summary message (one extra model call) and
//! keeps the recent turns verbatim.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::sync::{Arc, Mutex, PoisonError};

use futures::future::BoxFuture;

use crate::agent::estimate_messages;
use crate::client::{ChatMessage, ChatRequest, ChatRole, ContentPart, LlmClient, TokenUsage};
use crate::error::AgentError;
use crate::ids::SessionId;

/// Persistence for session transcripts.
///
/// Transcripts exclude the system prompt: the agent rebuilds it on every run
/// so memory and skills stay current. [`ChatMessage`] is `Serialize`, so a
/// database implementation can store the transcript as one JSON column.
pub trait SessionStore: Send + Sync + std::fmt::Debug {
    /// Load a transcript. An unknown session is an empty transcript.
    fn load<'a>(&'a self, id: &'a SessionId)
    -> BoxFuture<'a, Result<Vec<ChatMessage>, AgentError>>;

    /// Replace a transcript.
    fn save<'a>(
        &'a self,
        id: &'a SessionId,
        messages: &'a [ChatMessage],
    ) -> BoxFuture<'a, Result<(), AgentError>>;
}

/// A process-local [`SessionStore`]. Transcripts vanish on restart and are
/// not shared between replicas.
#[derive(Debug, Default)]
pub struct InMemorySessionStore {
    sessions: Mutex<HashMap<SessionId, Vec<ChatMessage>>>,
}

impl InMemorySessionStore {
    /// An empty store.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Share the store.
    #[must_use]
    pub fn shared(self) -> Arc<dyn SessionStore> {
        Arc::new(self)
    }
}

impl SessionStore for InMemorySessionStore {
    fn load<'a>(
        &'a self,
        id: &'a SessionId,
    ) -> BoxFuture<'a, Result<Vec<ChatMessage>, AgentError>> {
        let messages = self
            .sessions
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(id)
            .cloned()
            .unwrap_or_default();
        Box::pin(std::future::ready(Ok(messages)))
    }

    fn save<'a>(
        &'a self,
        id: &'a SessionId,
        messages: &'a [ChatMessage],
    ) -> BoxFuture<'a, Result<(), AgentError>> {
        self.sessions
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .insert(id.clone(), messages.to_vec());
        Box::pin(std::future::ready(Ok(())))
    }
}

/// When and how to summarize a long session transcript.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Compaction {
    /// Compact when the stored transcript estimates above this many tokens.
    pub trigger_tokens: u32,
    /// Keep roughly this many tokens of the newest turns verbatim.
    pub keep_recent_tokens: u32,
    /// Output cap for the summary call.
    pub summary_max_tokens: u32,
}

impl Default for Compaction {
    fn default() -> Self {
        Self {
            trigger_tokens: 24_000,
            keep_recent_tokens: 8_000,
            summary_max_tokens: 1_024,
        }
    }
}

/// What one compaction did.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct CompactionReport {
    /// Messages folded into the summary.
    pub summarized_messages: usize,
    /// Estimated transcript tokens before compaction.
    pub tokens_before: u32,
    /// Estimated transcript tokens after compaction.
    pub tokens_after: u32,
    /// Tokens the summary call spent.
    pub usage: TokenUsage,
}

/// The prefix marking a compaction summary message.
pub const SUMMARY_PREFIX: &str = "[Summary of the earlier conversation]";

const SUMMARY_INSTRUCTIONS: &str = "You compress conversation history for an AI agent that will continue the conversation. Write a dense summary of the transcript: the user's goals and standing instructions, decisions made, facts learned (with exact names, ids, numbers), tool results that still matter, and open tasks. Leave out small talk. Write plain prose or bullets; do not address the user.";

/// Pick where the verbatim tail starts: the oldest user message such that
/// the tail fits `keep_recent_tokens`. Returns `None` when no user message
/// after index 0 qualifies (nothing to fold).
fn split_point(messages: &[ChatMessage], keep_recent_tokens: u32) -> Option<usize> {
    let mut tail_tokens: u32 = 0;
    let mut split = None;
    for index in (1..messages.len()).rev() {
        let cost = estimate_messages(messages.get(index..=index).unwrap_or_default());
        tail_tokens = tail_tokens.saturating_add(cost);
        if tail_tokens > keep_recent_tokens {
            break;
        }
        if messages
            .get(index)
            .is_some_and(|message| message.role == ChatRole::User)
        {
            split = Some(index);
        }
    }
    split
}

/// Render a transcript slice as plain text for the summarizer.
fn render_transcript(messages: &[ChatMessage]) -> String {
    let mut out = String::new();
    for message in messages {
        let role = match message.role {
            ChatRole::System => "system",
            ChatRole::User => "user",
            ChatRole::Assistant => "assistant",
            ChatRole::Tool => "tool",
        };
        for part in &message.content {
            match part {
                ContentPart::Text(text) => {
                    let _ = writeln!(out, "{role}: {text}");
                }
                ContentPart::ToolCall {
                    name, arguments, ..
                } => {
                    let _ = writeln!(out, "{role} called {name} with {arguments}");
                }
                ContentPart::ToolResult { content, .. } => {
                    let clipped: String = content.chars().take(600).collect();
                    let _ = writeln!(out, "{role} result: {clipped}");
                }
            }
        }
    }
    out
}

/// Summarize the older part of a transcript when it exceeds the trigger.
///
/// Returns `None` (and leaves `messages` alone) when the transcript is under
/// the trigger or has no safe split point. Otherwise replaces the older
/// messages with one user message that starts with [`SUMMARY_PREFIX`]. The
/// split always lands on a user message, so no tool result loses its call.
///
/// # Errors
///
/// Returns [`AgentError`] when the summary call fails. The transcript is
/// unchanged in that case.
pub async fn compact(
    client: &dyn LlmClient,
    messages: &mut Vec<ChatMessage>,
    settings: &Compaction,
) -> Result<Option<CompactionReport>, AgentError> {
    let tokens_before = estimate_messages(messages);
    if tokens_before <= settings.trigger_tokens {
        return Ok(None);
    }
    let Some(split) = split_point(messages, settings.keep_recent_tokens) else {
        return Ok(None);
    };
    let head = messages.get(..split).unwrap_or_default();
    let request = ChatRequest {
        messages: vec![
            ChatMessage::text(ChatRole::System, SUMMARY_INSTRUCTIONS),
            ChatMessage::text(ChatRole::User, render_transcript(head)),
        ],
        tools: Vec::new(),
        max_tokens: Some(settings.summary_max_tokens),
        temperature: None,
    };
    let response = client.chat(&request).await?;
    let summary = response
        .content
        .iter()
        .filter_map(|part| match part {
            ContentPart::Text(text) => Some(text.as_str()),
            ContentPart::ToolCall { .. } | ContentPart::ToolResult { .. } => None,
        })
        .collect::<Vec<_>>()
        .join("");
    let mut compacted = Vec::with_capacity(messages.len().saturating_sub(split).saturating_add(1));
    compacted.push(ChatMessage::text(
        ChatRole::User,
        format!("{SUMMARY_PREFIX}\n{}", summary.trim()),
    ));
    compacted.extend(messages.drain(split..));
    *messages = compacted;
    Ok(Some(CompactionReport {
        summarized_messages: split,
        tokens_before,
        tokens_after: estimate_messages(messages),
        usage: response.usage,
    }))
}

#[cfg(test)]
mod tests;
