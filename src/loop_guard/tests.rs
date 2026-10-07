#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use proptest::prelude::*;
use serde_json::json;

use super::*;

#[test]
fn identical_calls_warn_then_stop() {
    let guard = LoopGuard::default();
    let mut tracker = LoopTracker::default();
    let args = json!({"q": 1});
    assert_eq!(
        tracker.record(&guard, "poll", &args, "same"),
        LoopVerdict::Ok
    );
    assert_eq!(
        tracker.record(&guard, "poll", &args, "same"),
        LoopVerdict::Ok
    );
    assert_eq!(
        tracker.record(&guard, "poll", &args, "same"),
        LoopVerdict::Warn { repeats: 3 }
    );
    assert_eq!(
        tracker.record(&guard, "poll", &args, "same"),
        LoopVerdict::Warn { repeats: 4 }
    );
    assert_eq!(
        tracker.record(&guard, "poll", &args, "same"),
        LoopVerdict::Stop { repeats: 5 }
    );
}

#[test]
fn changing_results_are_not_a_loop() {
    let guard = LoopGuard::default();
    let mut tracker = LoopTracker::default();
    for n in 0..20 {
        let verdict = tracker.record(&guard, "poll", &json!({}), &format!("progress {n}"));
        assert_eq!(verdict, LoopVerdict::Ok);
    }
}

#[test]
fn ping_pong_is_caught() {
    let guard = LoopGuard::default();
    let mut tracker = LoopTracker::default();
    let mut last = LoopVerdict::Ok;
    for _ in 0..5 {
        tracker.record(&guard, "a", &json!({}), "x");
        last = tracker.record(&guard, "b", &json!({}), "y");
    }
    assert_eq!(last, LoopVerdict::Stop { repeats: 5 });
}

#[test]
fn disabled_guard_never_fires() {
    let guard = LoopGuard::disabled();
    let mut tracker = LoopTracker::default();
    for _ in 0..100 {
        assert_eq!(
            tracker.record(&guard, "a", &json!({}), "x"),
            LoopVerdict::Ok
        );
    }
    assert_eq!(tracker.saved(), Vec::<u64>::new());
}

#[test]
fn saved_fingerprints_round_trip() {
    let guard = LoopGuard::default();
    let mut tracker = LoopTracker::default();
    tracker.record(&guard, "a", &json!({}), "x");
    tracker.record(&guard, "a", &json!({}), "x");
    let mut restored = LoopTracker::from_saved(tracker.saved());
    assert_eq!(
        restored.record(&guard, "a", &json!({}), "x"),
        LoopVerdict::Warn { repeats: 3 }
    );
    assert!(warning_note("a", 3).contains("3 times"));
}

proptest! {
    #[test]
    fn window_bounds_memory(calls in proptest::collection::vec(0u8..4, 0..200)) {
        let guard = LoopGuard { warn_after: 1_000, stop_after: 1_000, window: 7 };
        let mut tracker = LoopTracker::default();
        for call in calls {
            tracker.record(&guard, "t", &json!({"n": call}), "r");
            prop_assert!(tracker.saved().len() <= 7);
        }
    }
}
