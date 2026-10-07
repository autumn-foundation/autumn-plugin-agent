#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use proptest::prelude::*;
use serde_json::json;

use super::*;

fn block(limit: usize) -> MemoryBlock {
    MemoryBlock::new("memory", "notes", limit)
}

fn add(text: &str) -> MemoryOp {
    MemoryOp::Add {
        block: "memory".to_owned(),
        text: text.to_owned(),
    }
}

#[test]
fn add_replace_remove_round_trip() {
    let mut b = block(100);
    apply_op(&mut b, &add("likes tea")).unwrap();
    apply_op(&mut b, &add("lives in Oslo")).unwrap();
    apply_op(
        &mut b,
        &MemoryOp::Replace {
            block: "memory".to_owned(),
            old: "tea".to_owned(),
            text: "likes coffee".to_owned(),
        },
    )
    .unwrap();
    assert_eq!(b.entries, vec!["likes coffee", "lives in Oslo"]);
    apply_op(
        &mut b,
        &MemoryOp::Remove {
            block: "memory".to_owned(),
            old: "Oslo".to_owned(),
        },
    )
    .unwrap();
    assert_eq!(b.entries, vec!["likes coffee"]);
}

#[test]
fn overflow_tells_the_model_to_consolidate() {
    let mut b = block(10);
    apply_op(&mut b, &add("12345")).unwrap();
    let err = apply_op(&mut b, &add("678901")).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::Tool);
    assert!(err.message().contains("is full"), "{}", err.message());
    assert!(err.message().contains("11 of 10"), "{}", err.message());
    // A replace that frees room fits.
    apply_op(
        &mut b,
        &MemoryOp::Replace {
            block: "memory".to_owned(),
            old: "123".to_owned(),
            text: "0123456789".to_owned(),
        },
    )
    .unwrap();
    assert_eq!(b.used_chars(), 10);
}

#[test]
fn ambiguous_and_missing_matches_fail() {
    let mut b = block(100);
    apply_op(&mut b, &add("alpha one")).unwrap();
    apply_op(&mut b, &add("alpha two")).unwrap();
    let remove = |old: &str| MemoryOp::Remove {
        block: "memory".to_owned(),
        old: old.to_owned(),
    };
    assert!(
        apply_op(&mut b, &remove("alpha"))
            .unwrap_err()
            .message()
            .contains("2 entries")
    );
    assert!(
        apply_op(&mut b, &remove("beta"))
            .unwrap_err()
            .message()
            .contains("no entry")
    );
    assert!(apply_op(&mut b, &remove("  ")).is_err());
    assert!(apply_op(&mut b, &add("   ")).is_err());
}

#[tokio::test]
async fn store_scopes_are_isolated_and_start_from_the_template() {
    let store = InMemoryMemoryStore::default();
    let alice = MemoryScope::new("user:alice");
    let bob = MemoryScope::new("user:bob");
    store.apply(&alice, add("prefers mornings")).await.unwrap();
    let alice_blocks = store.load(&alice).await.unwrap();
    let bob_blocks = store.load(&bob).await.unwrap();
    assert_eq!(alice_blocks[0].entries, vec!["prefers mornings"]);
    assert!(bob_blocks[0].entries.is_empty());
    assert_eq!(bob_blocks.len(), 2);
    let err = store
        .apply(
            &alice,
            MemoryOp::Add {
                block: "nope".to_owned(),
                text: "x".to_owned(),
            },
        )
        .await
        .unwrap_err();
    assert!(err.message().contains("no memory block"));
}

#[tokio::test]
async fn memory_tool_edits_the_store() {
    let store = InMemoryMemoryStore::default().shared();
    let tool = MemoryTool::new(Arc::clone(&store), MemoryScope::agent());
    assert_eq!(tool.effect(), ToolEffect::Internal);
    let ctx = ToolContext::detached("c");
    let out = tool
        .execute(
            json!({"action": "add", "block": "user", "text": "name is Mark"}),
            &ctx,
        )
        .await
        .unwrap();
    assert_eq!(out["ok"], json!(true));
    assert_eq!(out["used"], json!(12));
    let blocks = store.load(&MemoryScope::agent()).await.unwrap();
    assert_eq!(blocks[1].entries, vec!["name is Mark"]);
    let err = tool
        .execute(json!({"action": "explode"}), &ctx)
        .await
        .unwrap_err();
    assert!(err.message().contains("invalid memory edit"));
}

#[test]
fn snapshot_renders_usage_and_entries() {
    let mut b = block(50);
    apply_op(&mut b, &add("likes tea")).unwrap();
    let text = render_snapshot(&[b]);
    assert!(
        text.contains("<memory block=\"memory\" used=\"9/50\">"),
        "{text}"
    );
    assert!(text.contains("- likes tea"), "{text}");
}

proptest! {
    #[test]
    fn edits_never_exceed_the_limit(
        texts in proptest::collection::vec("[a-z ]{0,30}", 0..40),
        limit in 0usize..120,
    ) {
        let mut b = block(limit);
        for (index, text) in texts.iter().enumerate() {
            let op = if index % 3 == 2 && !b.entries.is_empty() {
                let old: String = b.entries[0].chars().take(3).collect();
                MemoryOp::Replace { block: "memory".to_owned(), old, text: text.clone() }
            } else {
                add(text)
            };
            let _ = apply_op(&mut b, &op);
            prop_assert!(b.used_chars() <= limit);
        }
    }
}
