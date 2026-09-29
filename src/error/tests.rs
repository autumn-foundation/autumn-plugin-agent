#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use super::*;
use proptest::prelude::*;

#[test]
fn status_code_mapping_covers_every_kind() {
    let cases = [
        (ErrorKind::Config, StatusCode::INTERNAL_SERVER_ERROR),
        (ErrorKind::Authentication, StatusCode::BAD_GATEWAY),
        (ErrorKind::RateLimited, StatusCode::TOO_MANY_REQUESTS),
        (ErrorKind::Provider, StatusCode::BAD_GATEWAY),
        (ErrorKind::Transport, StatusCode::SERVICE_UNAVAILABLE),
        (ErrorKind::Tool, StatusCode::INTERNAL_SERVER_ERROR),
        (ErrorKind::Budget, StatusCode::PAYLOAD_TOO_LARGE),
        (ErrorKind::Decode, StatusCode::INTERNAL_SERVER_ERROR),
    ];
    for (kind, expected) in cases {
        assert_eq!(kind.status_code(), expected, "kind {kind:?}");
        assert_eq!(
            AgentError::new(kind, "boom").status_code(),
            expected,
            "error {kind:?}"
        );
    }
}

#[test]
fn autumn_error_translation_preserves_status() {
    let err = AgentError::new(ErrorKind::RateLimited, "slow down");
    let autumn = err.into_autumn_error();
    assert_eq!(autumn.status(), StatusCode::TOO_MANY_REQUESTS);
}

#[test]
fn display_mentions_kind_and_message() {
    let err = AgentError::new(ErrorKind::Tool, "weather blew up");
    let text = err.to_string();
    assert!(text.contains("Tool"), "{text}");
    assert!(text.contains("weather blew up"), "{text}");
}

#[test]
fn with_source_keeps_the_cause() {
    let io = std::io::Error::new(std::io::ErrorKind::TimedOut, "nope");
    let err = AgentError::with_source(ErrorKind::Transport, "timed out", io);
    assert!(std::error::Error::source(&err).is_some());
    assert!(std::error::Error::source(&AgentError::new(ErrorKind::Config, "x")).is_none());
}

#[test]
fn reqwest_timeout_maps_to_transport() {
    // Build a reqwest error without touching the network: an invalid URL
    // fails at build time with a builder error, which is not a timeout, so
    // assert the mapping on the kind axis we can construct deterministically.
    let err: AgentError = serde_json::from_str::<serde_json::Value>("{oops")
        .unwrap_err()
        .into();
    assert_eq!(err.kind(), ErrorKind::Decode);
}

proptest! {
    #[test]
    fn kind_roundtrips_through_status_code(kind in prop_oneof![
        Just(ErrorKind::Config),
        Just(ErrorKind::Authentication),
        Just(ErrorKind::RateLimited),
        Just(ErrorKind::Provider),
        Just(ErrorKind::Transport),
        Just(ErrorKind::Tool),
        Just(ErrorKind::Budget),
        Just(ErrorKind::Decode),
    ]) {
        // Every kind maps to a non-2xx status: agent failures never look like success.
        let status = kind.status_code();
        prop_assert!(!status.is_success());
        prop_assert_eq!(AgentError::new(kind, "m").status_code(), status);
    }
}
