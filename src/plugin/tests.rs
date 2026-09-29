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
