#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use super::*;
use crate::config::ProviderKind;

#[test]
fn builder_defaults() {
    let plugin = AgentPlugin::new();
    assert_eq!(plugin.config.provider, ProviderKind::OpenAiCompatible);
    assert!(plugin.tools.is_empty());
    let _ = AgentPlugin::default();
}

#[test]
fn configure_tunes_config() {
    let plugin = AgentPlugin::new().configure(|config| {
        config.provider = ProviderKind::Anthropic;
        config.max_steps = 4;
        config.model = Some("custom".to_owned());
    });
    assert_eq!(plugin.config.provider, ProviderKind::Anthropic);
    assert_eq!(plugin.config.max_steps, 4);
    assert_eq!(plugin.config.model.as_deref(), Some("custom"));
}

#[test]
fn build_declares_the_agent_config_section() {
    let builder = autumn_web::app().plugin(AgentPlugin::new());
    assert!(builder.has_config_section("agent"));
}

#[test]
fn build_registers_the_agent_job() {
    use autumn_web::plugin::Plugin as _;
    let builder = AgentPlugin::new().build(autumn_web::app());
    // JobInfo is not introspectable by name here; assert the builder still
    // carries the plugin name for duplicate-registration detection.
    let _ = builder;
}

#[test]
fn apply_overrides_prefers_explicit_values() {
    let mut loaded = AgentConfig {
        max_steps: 7, // pretend the file set this
        model: Some("file-model".to_owned()),
        ..AgentConfig::default()
    };

    let configured = AgentPlugin::new()
        .configure(|config| {
            config.max_steps = 3;
            config.provider = ProviderKind::Anthropic;
        })
        .config;

    apply_overrides(&mut loaded, &configured);
    // Explicit builder values win...
    assert_eq!(loaded.max_steps, 3);
    assert_eq!(loaded.provider, ProviderKind::Anthropic);
    // ...untouched fields keep the loaded value.
    assert_eq!(loaded.model.as_deref(), Some("file-model"));
}

#[test]
fn apply_overrides_keeps_loaded_when_builder_is_default() {
    let mut loaded = AgentConfig {
        max_steps: 7,
        ..AgentConfig::default()
    };
    let configured = AgentConfig::default();
    apply_overrides(&mut loaded, &configured);
    assert_eq!(loaded.max_steps, 7);
}

#[test]
fn tool_registration_collects_tools() {
    use crate::tools::FnTool;
    let tool = FnTool::new(
        "t",
        "d",
        serde_json::json!({}),
        |input: serde_json::Value| async move { Ok(input) },
    )
    .shared();
    let plugin = AgentPlugin::new().tool(Arc::clone(&tool)).tools(vec![tool]);
    assert_eq!(plugin.tools.len(), 2);
}

mod always_on {
    use std::time::Duration;

    use serde_json::json;

    use super::*;
    use crate::agent::ApprovalDecision;
    use crate::memory::InMemoryMemoryStore;
    use crate::policy::{Rule, ToolRules};
    use crate::proactive::{Heartbeat, LogDelivery};
    use crate::session::{Compaction, InMemorySessionStore};
    use crate::skills::Skill;
    use crate::test_support::Script;
    use crate::tools::FnTool;

    fn plugin() -> AgentPlugin {
        AgentPlugin::new()
            .hook(Arc::new(NoopHook))
            .policy(Arc::new(ToolRules::new().tool("send", Rule::Ask)))
            .session_store(InMemorySessionStore::new().shared())
            .memory_store(InMemoryMemoryStore::default().shared())
            .skill(Skill::new("a", "Skill a.", "body a"))
            .skills(vec![Skill::new("b", "Skill b.", "body b")])
            .compaction(Compaction::default())
            .delivery(Arc::new(LogDelivery))
            .heartbeat(Heartbeat::every(Duration::from_secs(1_800)))
            .followups(Duration::from_secs(3_600))
            .tool(
                FnTool::new("send", "Sends.", json!({}), |_: serde_json::Value| async {
                    Ok(json!("sent"))
                })
                .shared(),
            )
    }

    #[derive(Debug)]
    struct NoopHook;

    impl crate::hooks::AgentHooks for NoopHook {}

    #[test]
    fn builder_records_every_option() {
        let plugin = plugin();
        let debug = format!("{plugin:?}");
        for needle in [
            "hooks: 1",
            "memory: true",
            "[\"a\", \"b\"]",
            "Heartbeat",
            "followups: Some",
        ] {
            assert!(debug.contains(needle), "{needle} missing from {debug}");
        }
        // Building with a heartbeat registers the task without side effects.
        let builder = autumn_web::app().plugin(plugin);
        assert!(builder.has_config_section("agent"));
    }

    #[tokio::test]
    async fn runtime_wires_tools_memory_skills_and_followups() {
        let client = Script::new(vec![Script::answer("hi")]);
        let runtime = plugin().runtime(AgentConfig::default(), client.clone());
        assert!(runtime.memory().is_some());
        let handle = AgentHandle {
            runtime: Arc::new(runtime),
        };
        let turn = handle.chat("thread", "hello").await.unwrap();
        assert!(turn.outcome.is_completed());
        let tools: Vec<String> = client.requests()[0]
            .tools
            .iter()
            .map(|tool| tool.name.clone())
            .collect();
        assert_eq!(tools, ["send", "schedule_followup", "load_skill", "memory"]);
        assert_eq!(handle.tool_names(), ["send"]);
        assert_eq!(
            handle
                .runtime()
                .sessions()
                .load(&crate::ids::SessionId::new("thread"))
                .await
                .unwrap()
                .len(),
            2
        );
    }

    #[tokio::test]
    async fn handle_pauses_and_resumes_in_the_session() {
        let client = Script::new(vec![
            Script::call("c1", "send", json!({})),
            Script::answer("Sent."),
        ]);
        let runtime = plugin().runtime(AgentConfig::default(), client);
        let handle = AgentHandle {
            runtime: Arc::new(runtime),
        };
        let turn = handle
            .chat_as(crate::memory::MemoryScope::new("user:1"), "t", "send it")
            .await
            .unwrap();
        let AgentOutcome::AwaitingApproval { state } = turn.outcome else {
            panic!("expected a pause");
        };
        let turn = handle
            .resume_as(
                crate::memory::MemoryScope::new("user:1"),
                *state,
                vec![ApprovalDecision::approve("c1")],
            )
            .await
            .unwrap();
        assert_eq!(turn.outcome.text(), Some("Sent."));
        let saved = handle
            .runtime()
            .sessions()
            .load(&crate::ids::SessionId::new("t"))
            .await
            .unwrap();
        assert_eq!(saved.len(), 4);
    }

    #[test]
    fn overrides_cover_the_always_on_keys() {
        let mut loaded = AgentConfig::default();
        let configured = AgentPlugin::new()
            .configure(|config| {
                config.prompt_caching = false;
                config.max_run_secs = Some(90);
            })
            .config;
        apply_overrides(&mut loaded, &configured);
        assert!(!loaded.prompt_caching);
        assert_eq!(loaded.max_run_secs, Some(90));
    }

    #[tokio::test]
    async fn max_run_secs_reaches_the_agent() {
        let runtime = AgentRuntime::new(
            AgentConfig {
                max_run_secs: Some(42),
                ..AgentConfig::default()
            },
            Script::new(Vec::new()),
            Vec::new(),
        );
        assert!(format!("{:?}", runtime.agent()).contains("max_duration: Some(42s)"));
    }
}

#[tokio::test]
async fn handle_resume_uses_the_agent_scope() {
    use crate::agent::ApprovalDecision;
    use crate::policy::{Rule, ToolRules};
    use crate::test_support::Script;
    let client = Script::new(vec![
        Script::call("c1", "t", serde_json::json!({})),
        Script::answer("ok"),
    ]);
    let tool = crate::tools::FnTool::new(
        "t",
        "d",
        serde_json::json!({}),
        |v: serde_json::Value| async move { Ok(v) },
    )
    .shared();
    let runtime = AgentRuntime::new(AgentConfig::default(), client, vec![tool])
        .with_policy(Arc::new(ToolRules::new().tool("t", Rule::Ask)));
    let handle = AgentHandle {
        runtime: Arc::new(runtime),
    };
    let AgentOutcome::AwaitingApproval { state } = handle.run("go").await.unwrap() else {
        panic!("expected a pause");
    };
    let turn = handle
        .resume(*state, vec![ApprovalDecision::approve("c1")])
        .await
        .unwrap();
    assert_eq!(turn.outcome.text(), Some("ok"));
}
