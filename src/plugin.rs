//! The Autumn plugin: registration, configuration, and the handler extractor.
//!
//! Install with one line:
//!
//! ```rust,ignore
//! use autumn_plugin_agent::plugin::AgentPlugin;
//! use autumn_web::prelude::*;
//!
//! #[autumn_web::main]
//! async fn main() {
//!     autumn_web::app()
//!         .plugin(AgentPlugin::new().configure(|config| {
//!             config.max_steps = 12;
//!         }))
//!         .run()
//!         .await;
//! }
//! ```
//!
//! The plugin declares the `[agent]` config section, registers the
//! `agent_run` background job, installs a `list_models` health indicator,
//! and builds the shared [`AgentRuntime`](crate::agent::AgentRuntime) in a
//! startup hook. Handlers reach the agent through the [`AgentHandle`]
//! extractor.

use std::future::Future;
use std::sync::{Arc, OnceLock};

use autumn_web::app::AppBuilder;
use autumn_web::plugin::Plugin;
use autumn_web::reexports::axum::extract::FromRequestParts;
use autumn_web::{AppState, AutumnError};
use http::request::Parts;

use crate::agent::{Agent, AgentOutcome, AgentRuntime};
use crate::client::client_from_config;
use crate::config::AgentConfig;
use crate::error::AgentError;
use crate::health::AgentHealthIndicator;
use crate::tools::Tool;

/// The plugin. Register it with `AppBuilder::plugin`.
///
/// ```rust,ignore
/// autumn_web::app().plugin(AgentPlugin::new()).run().await;
/// ```
pub struct AgentPlugin {
    config: AgentConfig,
    tools: Vec<Arc<dyn Tool>>,
}

impl std::fmt::Debug for AgentPlugin {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentPlugin")
            .field("config", &self.config)
            .field(
                "tools",
                &self
                    .tools
                    .iter()
                    .map(|tool| tool.name().to_owned())
                    .collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl AgentPlugin {
    /// Create the plugin with default configuration.
    ///
    /// Defaults resolve from `autumn.toml`'s `[agent]` section and `AGENT_*`
    /// environment variables at startup; see [`AgentConfig`].
    #[must_use]
    pub fn new() -> Self {
        Self {
            config: AgentConfig::default(),
            tools: Vec::new(),
        }
    }

    /// Tune the configuration in code. Runs after file and env loading, so
    /// values set here win over both.
    #[must_use]
    pub fn configure(mut self, configure: impl FnOnce(&mut AgentConfig)) -> Self {
        configure(&mut self.config);
        self
    }

    /// Register one tool the agent may call.
    #[must_use]
    pub fn tool(mut self, tool: Arc<dyn Tool>) -> Self {
        self.tools.push(tool);
        self
    }

    /// Register several tools at once.
    #[must_use]
    pub fn tools(mut self, tools: Vec<Arc<dyn Tool>>) -> Self {
        self.tools.extend(tools);
        self
    }
}

impl Default for AgentPlugin {
    fn default() -> Self {
        Self::new()
    }
}

impl Plugin for AgentPlugin {
    /// Wire the plugin into the app builder.
    ///
    /// Declares the `[agent]` config section, registers the `agent_run`
    /// background job, adds the provider health indicator, and installs a
    /// startup hook that fails the boot fast when the config or the API key
    /// is unusable.
    fn build(self, app: AppBuilder) -> AppBuilder {
        let config = Arc::new(self.config);
        let tools = Arc::new(self.tools);
        // Shared with the health indicator: the startup hook installs the
        // real runtime here once the layered config has loaded, so the
        // health check always pings the provider the app actually uses.
        let active = Arc::new(OnceLock::<Arc<AgentRuntime>>::new());
        let indicator = AgentHealthIndicator::new(Arc::clone(&active));
        app.config_section("agent")
            .jobs(autumn_web::jobs![crate::agent::agent_run_job::run_agent])
            .health_indicator("agent", Arc::new(indicator))
            .on_startup(move |state| {
                let config = Arc::clone(&config);
                let tools = Arc::clone(&tools);
                let active = Arc::clone(&active);
                async move {
                    // Layered load: defaults < autumn.toml < AGENT_* env.
                    // `configure()` values were baked in at build time and win.
                    let mut loaded = AgentConfig::load().map_err(AgentError::into_autumn_error)?;
                    apply_overrides(&mut loaded, &config);
                    loaded.validate().map_err(AgentError::into_autumn_error)?;
                    let client =
                        client_from_config(&loaded).map_err(AgentError::into_autumn_error)?;
                    let provider_name = loaded.provider.as_str().to_owned();
                    let runtime = Arc::new(AgentRuntime::new(loaded, client, (*tools).clone()));
                    if active.set(Arc::clone(&runtime)).is_err() {
                        tracing::warn!("agent startup ran twice; keeping the first runtime");
                    }
                    state.extension_or_insert_with(|| (*runtime).clone());
                    tracing::info!(provider = %provider_name, "agent plugin started");
                    Ok(())
                }
            })
    }
}

/// Overlay the `configure()` values on top of the file+env load.
///
/// Every field the builder explicitly set wins; fields left at default keep
/// the loaded value. `Option` fields use `Some` as "explicitly set".
fn apply_overrides(loaded: &mut AgentConfig, configured: &AgentConfig) {
    let defaults = AgentConfig::default();
    if configured.provider != defaults.provider {
        loaded.provider = configured.provider;
    }
    if let Some(model) = &configured.model {
        loaded.model = Some(model.clone());
    }
    if let Some(base_url) = &configured.base_url {
        loaded.base_url = Some(base_url.clone());
    }
    if configured.request_timeout_secs != defaults.request_timeout_secs {
        loaded.request_timeout_secs = configured.request_timeout_secs;
    }
    if configured.max_steps != defaults.max_steps {
        loaded.max_steps = configured.max_steps;
    }
    if configured.max_tokens != defaults.max_tokens {
        loaded.max_tokens = configured.max_tokens;
    }
    if let Some(prompt) = &configured.system_prompt {
        loaded.system_prompt = Some(prompt.clone());
    }
}

/// Request extractor handing handlers a ready-to-run agent.
///
/// Autumn resolves it from app state before the handler runs:
///
/// ```rust,ignore
/// use autumn_plugin_agent::plugin::AgentHandle;
/// use autumn_web::prelude::*;
///
/// #[get("/ask")]
/// async fn ask(agent: AgentHandle) -> String {
///     agent
///         .run("Summarize this quarter's plot trends.")
///         .await
///         .map(|outcome| format!("{outcome:?}"))
///         .unwrap_or_else(|err| err.to_string())
/// }
/// ```
///
/// Fails with 503 when the plugin was never installed.
#[derive(Debug, Clone)]
pub struct AgentHandle {
    runtime: Arc<AgentRuntime>,
}

impl AgentHandle {
    /// Build an [`Agent`] with the plugin's client, tools, and budgets.
    ///
    /// Customize further with the builder methods before calling `run`.
    #[must_use]
    pub fn agent(&self) -> Agent {
        self.runtime.agent()
    }

    /// Run the agent loop with the plugin's default settings.
    ///
    /// # Errors
    ///
    /// Returns [`AgentError`] when a provider call fails. Budget exhaustion
    /// is not an error: see [`Agent::run`].
    pub async fn run(&self, prompt: &str) -> Result<AgentOutcome, AgentError> {
        self.agent().run(prompt).await
    }

    /// Names of the tools registered on the plugin.
    #[must_use]
    pub fn tool_names(&self) -> Vec<&str> {
        self.runtime.tool_names()
    }
}

impl FromRequestParts<AppState> for AgentHandle {
    type Rejection = AutumnError;

    // Axum 0.8 declares this as `-> impl Future`; a plain fn returning an
    // async block satisfies it without a synthetic `async` (there is nothing
    // to await — the impl only reads shared state).
    fn from_request_parts(
        _parts: &mut Parts,
        state: &AppState,
    ) -> impl Future<Output = Result<Self, Self::Rejection>> + Send {
        let runtime = state.extension::<AgentRuntime>();
        async move {
            runtime.map(|runtime| Self { runtime }).ok_or_else(|| {
                AutumnError::service_unavailable_msg("agent plugin is not installed")
            })
        }
    }
}

#[cfg(test)]
mod tests;
