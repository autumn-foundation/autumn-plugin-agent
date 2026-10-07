#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use super::*;
use crate::test_support::Script;

fn turn(i: usize, size: usize) -> [ChatMessage; 2] {
    [
        ChatMessage::text(ChatRole::User, format!("q{i} {}", "a".repeat(size))),
        ChatMessage::text(ChatRole::Assistant, format!("a{i} {}", "b".repeat(size))),
    ]
}

#[tokio::test]
async fn store_round_trips_and_isolates_sessions() {
    let store = InMemorySessionStore::new().shared();
    let one = SessionId::new("one");
    assert!(store.load(&one).await.unwrap().is_empty());
    store.save(&one, &turn(0, 1)).await.unwrap();
    assert_eq!(store.load(&one).await.unwrap().len(), 2);
    assert!(store.load(&SessionId::new("two")).await.unwrap().is_empty());
}

#[tokio::test]
async fn short_transcripts_are_left_alone() {
    let client = Script::new(Vec::new());
    let mut messages: Vec<ChatMessage> = turn(0, 10).to_vec();
    let report = compact(client.as_ref(), &mut messages, &Compaction::default())
        .await
        .unwrap();
    assert!(report.is_none());
    assert_eq!(messages.len(), 2);
    assert!(
        client.requests().is_empty(),
        "no model call under the trigger"
    );
}

#[tokio::test]
async fn compaction_folds_the_head_into_one_summary() {
    let client = Script::new(vec![Script::answer("  They talked about q0..q9.  ")]);
    let mut messages: Vec<ChatMessage> = (0..10).flat_map(|i| turn(i, 200)).collect();
    let settings = Compaction {
        trigger_tokens: 200,
        keep_recent_tokens: 250,
        summary_max_tokens: 64,
    };
    let report = compact(client.as_ref(), &mut messages, &settings)
        .await
        .unwrap()
        .unwrap();
    assert_eq!(messages[0].role, ChatRole::User);
    assert_eq!(
        messages[0].content[0],
        ContentPart::Text(format!("{SUMMARY_PREFIX}\nThey talked about q0..q9."))
    );
    // The tail starts on a user message and keeps the newest turn.
    assert_eq!(messages[1].role, ChatRole::User);
    assert!(
        matches!(&messages.last().unwrap().content[0], ContentPart::Text(t) if t.starts_with("a9"))
    );
    assert_eq!(report.summarized_messages, 20 - (messages.len() - 1));
    assert_eq!(report.usage, TokenUsage::new(10, 5));
    let request = &client.requests()[0];
    assert_eq!(request.max_tokens, Some(64));
    assert!(request.tools.is_empty());
}

#[tokio::test]
async fn no_split_point_means_no_compaction() {
    // One giant message: no user message fits the recent window.
    let client = Script::new(Vec::new());
    let mut messages = vec![
        ChatMessage::text(ChatRole::User, "x".repeat(10_000)),
        ChatMessage::text(ChatRole::Assistant, "y".repeat(10_000)),
    ];
    let settings = Compaction {
        trigger_tokens: 10,
        keep_recent_tokens: 10,
        summary_max_tokens: 64,
    };
    assert!(
        compact(client.as_ref(), &mut messages, &settings)
            .await
            .unwrap()
            .is_none()
    );
    assert_eq!(messages.len(), 2);
}

#[test]
fn transcript_rendering_covers_every_part() {
    let messages = vec![
        ChatMessage::text(ChatRole::User, "hi"),
        ChatMessage {
            role: ChatRole::Assistant,
            content: vec![ContentPart::ToolCall {
                id: "c".to_owned(),
                name: "t".to_owned(),
                arguments: serde_json::json!({"a": 1}),
            }],
        },
        ChatMessage {
            role: ChatRole::Tool,
            content: vec![ContentPart::ToolResult {
                tool_call_id: "c".to_owned(),
                content: "r".repeat(1_000),
            }],
        },
    ];
    let text = render_transcript(&messages);
    assert!(text.contains("user: hi"));
    assert!(text.contains("assistant called t with {\"a\":1}"));
    assert!(text.contains(&format!("tool result: {}\n", "r".repeat(600))));
}
