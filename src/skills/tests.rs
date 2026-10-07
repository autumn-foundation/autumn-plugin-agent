#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use serde_json::json;

use super::*;

const DIGEST: &str = "---\nname: weekly-digest\ndescription: \"How to write the Monday digest.\"\nversion: 2\n---\n\n1. Pull incidents.\n2. Summarize.\n";

#[test]
fn parse_reads_front_matter_and_body() {
    let skill = Skill::parse(DIGEST).unwrap();
    assert_eq!(skill.name, "weekly-digest");
    assert_eq!(skill.description, "How to write the Monday digest.");
    assert_eq!(skill.body, "1. Pull incidents.\n2. Summarize.");
}

#[test]
fn parse_rejects_bad_documents() {
    assert!(Skill::parse("no front matter").is_err());
    assert!(Skill::parse("---\nname: x\n").is_err());
    assert!(Skill::parse("---\ndescription: d\n---\nbody").is_err());
    assert!(Skill::parse("---\nname: x\n---\nbody").is_err());
    assert!(Skill::parse("---\nname: ''\ndescription: d\n---\n").is_err());
}

#[test]
fn index_lists_names_and_descriptions_only() {
    let skills = vec![Skill::parse(DIGEST).unwrap()];
    let index = render_index(&skills);
    assert!(index.contains("- weekly-digest: How to write the Monday digest."));
    assert!(!index.contains("Pull incidents"));
}

#[tokio::test]
async fn load_skill_returns_the_body() {
    let tool = SkillTool::new(vec![Skill::parse(DIGEST).unwrap()].into());
    assert_eq!(tool.effect(), ToolEffect::ReadOnly);
    let ctx = ToolContext::detached("c");
    let out = tool
        .execute(json!({"name": "weekly-digest"}), &ctx)
        .await
        .unwrap();
    assert_eq!(
        out["instructions"],
        json!("1. Pull incidents.\n2. Summarize.")
    );
    let err = tool
        .execute(json!({"name": "nope"}), &ctx)
        .await
        .unwrap_err();
    assert!(err.message().contains("weekly-digest"));
    assert!(tool.execute(json!({}), &ctx).await.is_err());
}
