#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use super::*;

#[test]
fn generated_run_ids_are_unique() {
    let ids: std::collections::HashSet<RunId> = (0..1_000).map(|_| RunId::generate()).collect();
    assert_eq!(ids.len(), 1_000);
    assert!(RunId::generate().as_str().starts_with("run_"));
}

#[test]
fn ids_serialize_as_plain_strings() {
    let session = SessionId::from("thread-42");
    assert_eq!(serde_json::to_string(&session).unwrap(), "\"thread-42\"");
    let back: SessionId = serde_json::from_str("\"thread-42\"").unwrap();
    assert_eq!(back, session);
    let run = RunId::new("job-7");
    assert_eq!(run.to_string(), "job-7");
    assert_eq!(
        serde_json::to_value(&run).unwrap(),
        serde_json::json!("job-7")
    );
}

#[test]
fn session_ids_convert_and_display() {
    let from_string = SessionId::from("chat-1".to_owned());
    assert_eq!(from_string, SessionId::new("chat-1"));
    assert_eq!(from_string.as_str(), "chat-1");
    assert_eq!(from_string.to_string(), "chat-1");
}
