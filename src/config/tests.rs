#![allow(clippy::unwrap_used, clippy::expect_used, clippy::panic)]

use super::*;

#[test]
fn missing_section_yields_defaults() {
    let config = AgentConfig::from_toml_str("[server]\nport = 3000\n").unwrap();
    assert_eq!(config.provider, ProviderKind::OpenAiCompatible);
    assert_eq!(config.resolved_model(), "gpt-4o-mini");
    assert_eq!(config.resolved_base_url(), "https://api.openai.com/v1");
    assert_eq!(config.max_steps, 10);
    assert_eq!(config.max_tokens, 32_000);
}

#[test]
fn full_section_parses() {
    let config = AgentConfig::from_toml_str(
        r#"
[agent]
provider = "anthropic"
model = "claude-sonnet-4-5"
base_url = "https://api.anthropic.com"
max_steps = 5
max_tokens = 8000
request_timeout_secs = 30
system_prompt = "Be terse."
"#,
    )
    .unwrap();
    assert_eq!(config.provider, ProviderKind::Anthropic);
    assert_eq!(config.resolved_model(), "claude-sonnet-4-5");
    assert_eq!(config.resolved_base_url(), "https://api.anthropic.com");
    assert_eq!(config.max_steps, 5);
    assert_eq!(config.max_tokens, 8000);
    assert_eq!(config.request_timeout_secs, 30);
    assert_eq!(config.system_prompt.as_deref(), Some("Be terse."));
}

#[test]
fn api_key_in_file_fails_closed() {
    for doc in [
        "[agent]\napi_key = \"sk-secret\"\n",
        "[agent]\napi-key = \"sk-secret\"\n",
    ] {
        let err = AgentConfig::from_toml_str(doc).unwrap_err();
        assert_eq!(err.kind(), ErrorKind::Config);
        assert!(err.message().contains("AGENT_API_KEY"), "{}", err.message());
    }
}

#[test]
fn unknown_provider_rejected() {
    let err = AgentConfig::from_toml_str("[agent]\nprovider = \"gemini\"\n").unwrap_err();
    assert_eq!(err.kind(), ErrorKind::Config);
    assert!(err.message().contains("gemini"), "{}", err.message());
}

#[test]
fn provider_spellings_accepted() {
    for raw in ["openai-compatible", "OpenAI-Compatible", "openai"] {
        let kind = ProviderKind::from_str(raw).unwrap();
        assert_eq!(kind, ProviderKind::OpenAiCompatible);
    }
    assert_eq!(
        ProviderKind::from_str("anthropic").unwrap(),
        ProviderKind::Anthropic
    );
}

#[test]
fn validation_rejects_nonsense() {
    for bad in [
        AgentConfig {
            max_steps: 0,
            ..AgentConfig::default()
        },
        AgentConfig {
            max_tokens: 10,
            ..AgentConfig::default()
        },
        AgentConfig {
            request_timeout_secs: 0,
            ..AgentConfig::default()
        },
        AgentConfig {
            base_url: Some("ftp://example.com".to_owned()),
            ..AgentConfig::default()
        },
        AgentConfig {
            model: Some("   ".to_owned()),
            ..AgentConfig::default()
        },
    ] {
        assert!(bad.validate().is_err());
    }

    AgentConfig {
        base_url: Some("http://localhost:11434/v1".to_owned()),
        ..AgentConfig::default()
    }
    .validate()
    .unwrap();
}

#[test]
fn non_table_section_rejected() {
    let err = AgentConfig::from_toml_str("[agent]\nprovider = 42\n").unwrap_err();
    assert_eq!(err.kind(), ErrorKind::Config);
}

#[test]
fn invalid_toml_rejected() {
    let err = AgentConfig::from_toml_str("[agent\nprovider = ").unwrap_err();
    assert_eq!(err.kind(), ErrorKind::Config);
}

#[test]
fn env_overrides_file() {
    // `apply_env_with` takes a lookup closure so the test never touches the
    // process environment (mutating it is `unsafe` on Rust 2024 and forbidden
    // by this crate's `unsafe_code = "forbid"`).
    let vars = std::collections::HashMap::from([
        ("AGENT_MODEL".to_owned(), "env-model".to_owned()),
        ("AGENT_MAX_STEPS".to_owned(), "3".to_owned()),
    ]);
    let mut config = AgentConfig::from_toml_str("[agent]\nmodel = \"file-model\"\n").unwrap();
    config
        .apply_env_with(|name| vars.get(name).cloned())
        .unwrap();
    assert_eq!(config.resolved_model(), "env-model");
    assert_eq!(config.max_steps, 3);
}

#[test]
fn env_parse_error_names_the_variable() {
    let mut config = AgentConfig::default();
    let err = config
        .apply_env_with(|name| (name == "AGENT_MAX_STEPS").then(|| "many".to_owned()))
        .unwrap_err();
    assert!(
        err.message().contains("AGENT_MAX_STEPS"),
        "{}",
        err.message()
    );
}

#[test]
fn api_key_missing_is_a_config_error() {
    let err = AgentConfig::api_key_with(|_| None).unwrap_err();
    assert_eq!(err.kind(), ErrorKind::Config);
    assert!(err.message().contains("AGENT_API_KEY"), "{}", err.message());
}

#[test]
fn api_key_present_resolves() {
    let key = AgentConfig::api_key_with(|_| Some("sk-test".to_owned())).unwrap();
    assert_eq!(key, "sk-test");
}

#[test]
fn always_on_keys_parse_from_file_and_env() {
    let mut config =
        AgentConfig::from_toml_str("[agent]\nprompt_caching = false\nmax_run_secs = 120\n")
            .unwrap();
    assert!(!config.prompt_caching);
    assert_eq!(config.max_run_secs, Some(120));
    let vars = std::collections::HashMap::from([
        ("AGENT_PROMPT_CACHING".to_owned(), "yes".to_owned()),
        ("AGENT_MAX_RUN_SECS".to_owned(), "30".to_owned()),
    ]);
    config
        .apply_env_with(|name| vars.get(name).cloned())
        .unwrap();
    assert!(config.prompt_caching);
    assert_eq!(config.max_run_secs, Some(30));
    let defaults = AgentConfig::default();
    assert!(defaults.prompt_caching);
    assert_eq!(defaults.max_run_secs, None);
}

#[test]
fn always_on_keys_are_validated() {
    assert!(AgentConfig::from_toml_str("[agent]\nmax_run_secs = 0\n").is_err());
    assert!(AgentConfig::from_toml_str("[agent]\nmax_run_secs = 86401\n").is_err());
    let mut config = AgentConfig::default();
    let err = config
        .apply_env_with(|name| (name == "AGENT_PROMPT_CACHING").then(|| "maybe".to_owned()))
        .unwrap_err();
    assert!(
        err.message().contains("AGENT_PROMPT_CACHING"),
        "{}",
        err.message()
    );
    for (raw, expected) in [
        ("1", true),
        ("ON", true),
        ("0", false),
        (" off ", false),
        ("no", false),
    ] {
        let mut config = AgentConfig::default();
        config
            .apply_env_with(|name| (name == "AGENT_PROMPT_CACHING").then(|| raw.to_owned()))
            .unwrap();
        assert_eq!(config.prompt_caching, expected, "{raw}");
    }
}
