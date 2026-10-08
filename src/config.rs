//! Layered configuration for the agent plugin.
//!
//! The plugin reads its settings from the `[agent]` section of `autumn.toml`
//! (declared strict-config-safe through
//! [`AppBuilder::config_section`](autumn_web::app::AppBuilder::config_section)),
//! then applies `AGENT_*` environment variables on top. Precedence, weakest
//! to strongest: compiled defaults < `autumn.toml` < environment <
//! [`AgentPlugin::configure`](crate::plugin::AgentPlugin::configure).
//!
//! The API key is the deliberate exception: it comes **only** from the
//! `AGENT_API_KEY` environment variable. A key sitting in `autumn.toml`
//! fails config loading outright instead of being silently accepted.
//!
//! ```toml
//! [agent]
//! provider = "openai-compatible" # or "anthropic"
//! model = "gpt-4o-mini"
//! base_url = "http://localhost:11434/v1" # Ollama, vLLM, ...
//! max_steps = 10
//! max_tokens = 32000
//! request_timeout_secs = 60
//! ```

use std::str::FromStr;

use serde::Deserialize;

use crate::error::{AgentError, ErrorKind};

/// Selects the wire protocol the plugin speaks to the provider.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ProviderKind {
    /// OpenAI chat-completions protocol (`POST /chat/completions`).
    ///
    /// Covers OpenAI itself plus OpenAI-compatible endpoints such as Ollama,
    /// vLLM, and LM Studio. Pick this unless the provider only speaks
    /// Anthropic's API.
    #[default]
    OpenAiCompatible,
    /// Anthropic Messages API (`POST /v1/messages`).
    Anthropic,
}

impl ProviderKind {
    /// The config-file spelling of this provider.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::OpenAiCompatible => "openai-compatible",
            Self::Anthropic => "anthropic",
        }
    }

    /// Default model used when the config leaves `model` unset.
    ///
    /// Pin a model in `autumn.toml` for production use; defaults drift as
    /// providers rename models.
    #[must_use]
    pub const fn default_model(self) -> &'static str {
        match self {
            Self::OpenAiCompatible => "gpt-4o-mini",
            Self::Anthropic => "claude-sonnet-4-5",
        }
    }

    /// Default base URL used when the config leaves `base_url` unset.
    #[must_use]
    pub const fn default_base_url(self) -> &'static str {
        match self {
            Self::OpenAiCompatible => "https://api.openai.com/v1",
            Self::Anthropic => "https://api.anthropic.com",
        }
    }
}

impl FromStr for ProviderKind {
    type Err = AgentError;

    fn from_str(s: &str) -> Result<Self, Self::Err> {
        match s.trim().to_ascii_lowercase().as_str() {
            "openai-compatible" | "openai_compatible" | "openai" => Ok(Self::OpenAiCompatible),
            "anthropic" => Ok(Self::Anthropic),
            other => Err(AgentError::new(
                ErrorKind::Config,
                format!(
                    r#"unknown agent provider "{other}": expected "openai-compatible" or "anthropic""#
                ),
            )),
        }
    }
}

impl std::fmt::Display for ProviderKind {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

impl<'de> Deserialize<'de> for ProviderKind {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let raw = String::deserialize(deserializer)?;
        Self::from_str(&raw).map_err(serde::de::Error::custom)
    }
}

/// Configuration for the agent plugin.
///
/// Build it with [`AgentConfig::load`] (file + env), tweak it in tests with
/// [`AgentConfig::from_toml_str`], or construct it by hand and adjust fields.
#[derive(Debug, Clone)]
pub struct AgentConfig {
    /// Which provider protocol to speak.
    pub provider: ProviderKind,
    /// Model name, or `None` for the provider default.
    pub model: Option<String>,
    /// Provider base URL, or `None` for the provider default.
    pub base_url: Option<String>,
    /// Per-request HTTP timeout in seconds.
    pub request_timeout_secs: u64,
    /// Maximum agent-loop iterations before the run stops.
    pub max_steps: u32,
    /// Token budget for one agent run (estimated + provider-reported).
    pub max_tokens: u32,
    /// System prompt prepended to every agent run, if set.
    pub system_prompt: Option<String>,
    /// Mark Anthropic prompt-cache breakpoints. On by default; OpenAI caches
    /// on its own and ignores this.
    pub prompt_caching: bool,
    /// Wall-clock limit for one agent run in seconds, or `None` for no limit.
    pub max_run_secs: Option<u64>,
}

impl Default for AgentConfig {
    fn default() -> Self {
        Self {
            provider: ProviderKind::default(),
            model: None,
            base_url: None,
            request_timeout_secs: 60,
            max_steps: 10,
            max_tokens: 32_000,
            system_prompt: None,
            prompt_caching: true,
            max_run_secs: None,
        }
    }
}

/// Raw `[agent]` table: every field optional.
///
/// The API key is deliberately absent: it lives in `AGENT_API_KEY`, never in
/// the file. [`AgentConfig::from_toml_str`] rejects `api_key`/`api-key`
/// before deserializing, so no field is needed here.
#[derive(Debug, Default, Deserialize)]
#[serde(deny_unknown_fields)]
struct RawAgentConfig {
    provider: Option<ProviderKind>,
    model: Option<String>,
    base_url: Option<String>,
    request_timeout_secs: Option<u64>,
    max_steps: Option<u32>,
    max_tokens: Option<u32>,
    system_prompt: Option<String>,
    prompt_caching: Option<bool>,
    max_run_secs: Option<u64>,
}

impl AgentConfig {
    /// Load configuration from `autumn.toml` plus the environment.
    ///
    /// Reads `./autumn.toml` (or the file named by `AGENT_CONFIG_FILE`),
    /// applies the `[agent]` table, then applies `AGENT_*` overrides.
    /// A missing file is fine: defaults plus environment still work.
    ///
    /// # Errors
    ///
    /// Returns [`AgentError`] with [`ErrorKind::Config`]
    /// when the file cannot be read or parsed, an `AGENT_*` variable has an
    /// invalid value, or the merged configuration fails validation.
    pub fn load() -> Result<Self, AgentError> {
        let path = std::env::var("AGENT_CONFIG_FILE").unwrap_or_else(|_| "autumn.toml".to_owned());
        let mut config = match std::fs::read_to_string(&path) {
            Ok(text) => Self::from_toml_str(&text)?,
            Err(err) if err.kind() == std::io::ErrorKind::NotFound => Self::default(),
            Err(err) => {
                return Err(AgentError::with_source(
                    ErrorKind::Config,
                    format!("cannot read agent config file {path}"),
                    err,
                ));
            }
        };
        config.apply_env()?;
        config.validate()?;
        Ok(config)
    }

    /// Parse the `[agent]` table out of a TOML document.
    ///
    /// Pure and side-effect free: this is the seam unit tests exercise.
    /// A missing `[agent]` table yields defaults.
    ///
    /// # Errors
    ///
    /// Returns [`AgentError`] with [`ErrorKind::Config`]
    /// when the document is not valid TOML, `[agent]` is not a table, the
    /// table sets `api_key`/`api-key` (fail-closed: the key must come from
    /// `AGENT_API_KEY`), or a field fails validation.
    pub fn from_toml_str(document: &str) -> Result<Self, AgentError> {
        let table: toml::Table = document.parse().map_err(|err| {
            AgentError::with_source(ErrorKind::Config, "invalid TOML in agent config", err)
        })?;
        let Some(section) = table.get("agent") else {
            return Ok(Self::default());
        };
        let section_table = section
            .as_table()
            .ok_or_else(|| AgentError::new(ErrorKind::Config, "[agent] must be a TOML table"))?;
        // Fail closed on secrets before touching any other field.
        if section_table.contains_key("api_key") || section_table.contains_key("api-key") {
            return Err(AgentError::new(
                ErrorKind::Config,
                "refusing to load: api_key must come from the AGENT_API_KEY environment variable, never from the config file",
            ));
        }
        let raw: RawAgentConfig = section_table.clone().try_into().map_err(|err| {
            let detail = err.to_string();
            AgentError::with_source(
                ErrorKind::Config,
                format!("invalid [agent] table: {detail}"),
                err,
            )
        })?;
        let mut config = Self::default();
        if let Some(provider) = raw.provider {
            config.provider = provider;
        }
        if let Some(model) = raw.model {
            config.model = Some(model);
        }
        if let Some(base_url) = raw.base_url {
            config.base_url = Some(base_url);
        }
        if let Some(secs) = raw.request_timeout_secs {
            config.request_timeout_secs = secs;
        }
        if let Some(steps) = raw.max_steps {
            config.max_steps = steps;
        }
        if let Some(tokens) = raw.max_tokens {
            config.max_tokens = tokens;
        }
        if let Some(prompt) = raw.system_prompt {
            config.system_prompt = Some(prompt);
        }
        if let Some(caching) = raw.prompt_caching {
            config.prompt_caching = caching;
        }
        if let Some(secs) = raw.max_run_secs {
            config.max_run_secs = Some(secs);
        }
        config.validate()?;
        Ok(config)
    }

    /// Apply `AGENT_*` environment variables over the current values.
    ///
    /// Recognised: `AGENT_PROVIDER`, `AGENT_MODEL`, `AGENT_BASE_URL`,
    /// `AGENT_REQUEST_TIMEOUT_SECS`, `AGENT_MAX_STEPS`, `AGENT_MAX_TOKENS`,
    /// `AGENT_SYSTEM_PROMPT`, `AGENT_PROMPT_CACHING` (`true`/`false`),
    /// `AGENT_MAX_RUN_SECS`. Unset variables leave the value alone.
    ///
    /// # Errors
    ///
    /// Returns [`AgentError`] with [`ErrorKind::Config`]
    /// when a set variable has an invalid value (unknown provider, malformed
    /// number, or a number that does not fit its field).
    pub fn apply_env(&mut self) -> Result<(), AgentError> {
        self.apply_env_with(|name| std::env::var(name).ok())
    }

    /// Apply environment variables resolved through `get`.
    ///
    /// Test seam for [`AgentConfig::apply_env`]: pass a map lookup instead of
    /// touching the process environment, which is `unsafe` to mutate on
    /// Rust 2024 and forbidden by this crate's `unsafe_code = "forbid"`.
    pub(crate) fn apply_env_with(
        &mut self,
        get: impl Fn(&str) -> Option<String>,
    ) -> Result<(), AgentError> {
        if let Some(raw) = get("AGENT_PROVIDER") {
            self.provider = ProviderKind::from_str(&raw)?;
        }
        if let Some(model) = get("AGENT_MODEL") {
            self.model = Some(model);
        }
        if let Some(base_url) = get("AGENT_BASE_URL") {
            self.base_url = Some(base_url);
        }
        if let Some(raw) = get("AGENT_REQUEST_TIMEOUT_SECS") {
            self.request_timeout_secs = parse_env_u64("AGENT_REQUEST_TIMEOUT_SECS", &raw)?;
        }
        if let Some(raw) = get("AGENT_MAX_STEPS") {
            let steps = parse_env_u64("AGENT_MAX_STEPS", &raw)?;
            self.max_steps = u32::try_from(steps).map_err(|_| {
                AgentError::new(ErrorKind::Config, "AGENT_MAX_STEPS does not fit in a u32")
            })?;
        }
        if let Some(raw) = get("AGENT_MAX_TOKENS") {
            let tokens = parse_env_u64("AGENT_MAX_TOKENS", &raw)?;
            self.max_tokens = u32::try_from(tokens).map_err(|_| {
                AgentError::new(ErrorKind::Config, "AGENT_MAX_TOKENS does not fit in a u32")
            })?;
        }
        if let Some(prompt) = get("AGENT_SYSTEM_PROMPT") {
            self.system_prompt = Some(prompt);
        }
        if let Some(raw) = get("AGENT_PROMPT_CACHING") {
            self.prompt_caching = parse_env_bool("AGENT_PROMPT_CACHING", &raw)?;
        }
        if let Some(raw) = get("AGENT_MAX_RUN_SECS") {
            self.max_run_secs = Some(parse_env_u64("AGENT_MAX_RUN_SECS", &raw)?);
        }
        Ok(())
    }

    /// Read the provider API key from the environment.
    ///
    /// The key comes only from `AGENT_API_KEY`. Never log the return value.
    ///
    /// # Errors
    ///
    /// Returns [`AgentError`] with [`ErrorKind::Config`]
    /// when `AGENT_API_KEY` is not set.
    pub fn api_key(&self) -> Result<String, AgentError> {
        Self::api_key_with(|name| std::env::var(name).ok())
    }

    /// Read the API key resolved through `get`.
    ///
    /// Test seam for [`AgentConfig::api_key`]; see
    /// [`AgentConfig::apply_env_with`].
    ///
    /// # Errors
    ///
    /// Returns [`AgentError`] with [`ErrorKind::Config`]
    /// when `get("AGENT_API_KEY")` returns [`None`].
    pub(crate) fn api_key_with(get: impl Fn(&str) -> Option<String>) -> Result<String, AgentError> {
        get("AGENT_API_KEY").ok_or_else(|| {
            AgentError::new(
                ErrorKind::Config,
                "AGENT_API_KEY is not set: export it before starting the app",
            )
        })
    }

    /// Check every field for sanity. Called by [`AgentConfig::load`] and
    /// [`AgentConfig::from_toml_str`]; call it again after hand-editing.
    ///
    /// # Errors
    ///
    /// Returns [`AgentError`] with [`ErrorKind::Config`]
    /// when any field is out of its accepted range or malformed.
    pub fn validate(&self) -> Result<(), AgentError> {
        if self.request_timeout_secs == 0 || self.request_timeout_secs > 3_600 {
            return Err(AgentError::new(
                ErrorKind::Config,
                "request_timeout_secs must be between 1 and 3600",
            ));
        }
        if self.max_steps == 0 || self.max_steps > 1_000 {
            return Err(AgentError::new(
                ErrorKind::Config,
                "max_steps must be between 1 and 1000",
            ));
        }
        if self.max_tokens < 1_000 || self.max_tokens > 1_000_000 {
            return Err(AgentError::new(
                ErrorKind::Config,
                "max_tokens must be between 1000 and 1000000",
            ));
        }
        if let Some(secs) = self.max_run_secs
            && (secs == 0 || secs > 86_400)
        {
            return Err(AgentError::new(
                ErrorKind::Config,
                "max_run_secs must be between 1 and 86400",
            ));
        }
        if let Some(model) = self.model.as_deref()
            && model.trim().is_empty()
        {
            return Err(AgentError::new(
                ErrorKind::Config,
                "model must not be blank",
            ));
        }
        if let Some(base_url) = self.base_url.as_deref() {
            let url = reqwest::Url::parse(base_url).map_err(|err| {
                AgentError::with_source(ErrorKind::Config, "base_url is not a valid URL", err)
            })?;
            if url.scheme() != "http" && url.scheme() != "https" {
                return Err(AgentError::new(
                    ErrorKind::Config,
                    "base_url must use the http or https scheme",
                ));
            }
        }
        Ok(())
    }

    /// Effective model name: the configured one, or the provider default.
    #[must_use]
    pub fn resolved_model(&self) -> &str {
        self.model
            .as_deref()
            .filter(|model| !model.trim().is_empty())
            .unwrap_or_else(|| self.provider.default_model())
    }

    /// Effective base URL: the configured one, or the provider default.
    #[must_use]
    pub fn resolved_base_url(&self) -> &str {
        self.base_url
            .as_deref()
            .unwrap_or_else(|| self.provider.default_base_url())
    }
}

/// Parse a `AGENT_*` integer variable, naming the variable on failure.
fn parse_env_u64(name: &str, raw: &str) -> Result<u64, AgentError> {
    raw.trim().parse::<u64>().map_err(|_| {
        AgentError::new(
            ErrorKind::Config,
            format!("{name} must be a non-negative integer, got {raw:?}"),
        )
    })
}

/// Parse a `AGENT_*` boolean variable, naming the variable on failure.
fn parse_env_bool(name: &str, raw: &str) -> Result<bool, AgentError> {
    match raw.trim().to_ascii_lowercase().as_str() {
        "1" | "true" | "yes" | "on" => Ok(true),
        "0" | "false" | "no" | "off" => Ok(false),
        _ => Err(AgentError::new(
            ErrorKind::Config,
            format!("{name} must be true or false, got {raw:?}"),
        )),
    }
}

#[cfg(test)]
mod tests;
