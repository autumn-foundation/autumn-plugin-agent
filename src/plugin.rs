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
//! `agent_run` and `agent_resume` background jobs, installs a `list_models`
//! health indicator, registers the heartbeat task when one is set, and
//! builds the shared [`AgentRuntime`] in a
//! startup hook. Handlers reach the agent through the [`AgentHandle`]
//! extractor.
//!
//! An always-on setup adds memory, sessions, a heartbeat, and a delivery:
//!
//! ```rust,ignore
//! use std::time::Duration;
//! use autumn_plugin_agent::plugin::AgentPlugin;
//! use autumn_plugin_agent::memory::InMemoryMemoryStore;
//! use autumn_plugin_agent::proactive::Heartbeat;
//!
//! autumn_web::app()
//!     .plugin(
//!         AgentPlugin::new()
//!             .memory_store(InMemoryMemoryStore::default().shared())
//!             .heartbeat(Heartbeat::every(Duration::from_secs(1_800)).session("ops"))
//!             .followups(Duration::from_secs(24 * 3_600))
//!             .delivery(my_slack_delivery),
//!     )
//!     .run()
//!     .await;
//! ```

use std::future::Future;
use std::sync::{Arc, OnceLock};
use std::time::Duration;

use autumn_web::app::AppBuilder;
use autumn_web::plugin::Plugin;
use autumn_web::reexports::axum::extract::FromRequestParts;
use autumn_web::{AppState, AutumnError};
use http::request::Parts;

use crate::agent::{Agent, AgentOutcome, AgentRuntime, AgentTurn, ApprovalDecision, RunState};
use crate::client::client_from_config;
use crate::config::AgentConfig;
use crate::error::AgentError;
use crate::health::AgentHealthIndicator;
use crate::hooks::AgentHooks;
use crate::ids::SessionId;
use crate::memory::{MemoryScope, MemoryStore};
use crate::policy::ToolPolicy;
use crate::proactive::{Delivery, Heartbeat, HeartbeatSettings};
use crate::session::{Compaction, SessionStore};
use crate::skills::Skill;
use crate::tools::Tool;

/// The plugin. Register it with `AppBuilder::plugin`.
///
/// ```rust,ignore
/// autumn_web::app().plugin(AgentPlugin::new()).run().await;
/// ```
pub struct AgentPlugin {
    config: AgentConfig,
    tools: Vec<Arc<dyn Tool>>,
    hooks: Vec<Arc<dyn AgentHooks>>,
    policy: Option<Arc<dyn ToolPolicy>>,
    sessions: Option<Arc<dyn SessionStore>>,
    memory: Option<Arc<dyn MemoryStore>>,
    skills: Vec<Skill>,
    compaction: Option<Compaction>,
    delivery: Option<Arc<dyn Delivery>>,
    heartbeat: Option<Heartbeat>,
    followups: Option<Duration>,
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
            .field("hooks", &self.hooks.len())
            .field("policy", &self.policy)
            .field("memory", &self.memory.is_some())
            .field(
                "skills",
                &self
                    .skills
                    .iter()
                    .map(|skill| skill.name.as_str())
                    .collect::<Vec<_>>(),
            )
            .field("compaction", &self.compaction)
            .field("heartbeat", &self.heartbeat)
            .field("followups", &self.followups)
            .finish_non_exhaustive()
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
            hooks: Vec::new(),
            policy: None,
            sessions: None,
            memory: None,
            skills: Vec::new(),
            compaction: None,
            delivery: None,
            heartbeat: None,
            followups: None,
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

    /// Add a lifecycle hook to every agent.
    #[must_use]
    pub fn hook(mut self, hook: Arc<dyn AgentHooks>) -> Self {
        self.hooks.push(hook);
        self
    }

    /// Decide per tool call whether agents may act. Defaults to allowing
    /// every call.
    #[must_use]
    pub fn policy(mut self, policy: Arc<dyn ToolPolicy>) -> Self {
        self.policy = Some(policy);
        self
    }

    /// Persist session transcripts here. Defaults to an in-memory store,
    /// which loses sessions on restart.
    #[must_use]
    pub fn session_store(mut self, store: Arc<dyn SessionStore>) -> Self {
        self.sessions = Some(store);
        self
    }

    /// Give agents persistent memory in this store.
    #[must_use]
    pub fn memory_store(mut self, store: Arc<dyn MemoryStore>) -> Self {
        self.memory = Some(store);
        self
    }

    /// Offer one skill to every agent.
    #[must_use]
    pub fn skill(mut self, skill: Skill) -> Self {
        self.skills.push(skill);
        self
    }

    /// Offer several skills to every agent.
    #[must_use]
    pub fn skills(mut self, skills: Vec<Skill>) -> Self {
        self.skills.extend(skills);
        self
    }

    /// Summarize long session transcripts before a run.
    #[must_use]
    pub const fn compaction(mut self, compaction: Compaction) -> Self {
        self.compaction = Some(compaction);
        self
    }

    /// Where heartbeat, follow-up, and `deliver` job results go. Defaults to
    /// logging them.
    #[must_use]
    pub fn delivery(mut self, delivery: Arc<dyn Delivery>) -> Self {
        self.delivery = Some(delivery);
        self
    }

    /// Wake the agent on a schedule. Registers the `agent_heartbeat` task.
    #[must_use]
    pub fn heartbeat(mut self, heartbeat: Heartbeat) -> Self {
        self.heartbeat = Some(heartbeat);
        self
    }

    /// Give agents the `schedule_followup` tool, so they can book their own
    /// next run up to `max_delay` ahead.
    #[must_use]
    pub const fn followups(mut self, max_delay: Duration) -> Self {
        self.followups = Some(max_delay);
        self
    }

    /// Assemble the runtime from a loaded config and client.
    fn runtime(
        &self,
        config: AgentConfig,
        client: Arc<dyn crate::client::LlmClient>,
    ) -> AgentRuntime {
        let mut runtime = AgentRuntime::new(config, client, self.tools.clone())
            .with_hooks(self.hooks.clone())
            .with_skills(self.skills.clone());
        if let Some(policy) = &self.policy {
            runtime = runtime.with_policy(Arc::clone(policy));
        }
        if let Some(sessions) = &self.sessions {
            runtime = runtime.with_sessions(Arc::clone(sessions));
        }
        if let Some(memory) = &self.memory {
            runtime = runtime.with_memory(Arc::clone(memory));
        }
        if let Some(compaction) = self.compaction {
            runtime = runtime.with_compaction(compaction);
        }
        if let Some(delivery) = &self.delivery {
            runtime = runtime.with_delivery(Arc::clone(delivery));
        }
        if let Some(max_delay) = self.followups {
            runtime = runtime.with_followups(max_delay);
        }
        runtime
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
    /// Declares the `[agent]` config section, registers the `agent_run` and
    /// `agent_resume` background jobs and the heartbeat task, adds the
    /// provider health indicator, and installs a startup hook that fails the
    /// boot fast when the config, the heartbeat, or the API key is unusable.
    fn build(self, app: AppBuilder) -> AppBuilder {
        let tasks: Vec<_> = self.heartbeat.iter().map(Heartbeat::task_info).collect();
        let plugin = Arc::new(self);
        // Shared with the health indicator: the startup hook installs the
        // real runtime here once the layered config has loaded, so the
        // health check always pings the provider the app actually uses.
        let active = Arc::new(OnceLock::<Arc<AgentRuntime>>::new());
        let indicator = AgentHealthIndicator::new(Arc::clone(&active));
        app.config_section("agent")
            .jobs(autumn_web::jobs![
                crate::jobs::agent_jobs::run_agent,
                crate::jobs::agent_jobs::resume_agent
            ])
            .tasks(tasks)
            .health_indicator("agent", Arc::new(indicator))
            .on_startup(move |state| {
                let plugin = Arc::clone(&plugin);
                let active = Arc::clone(&active);
                async move {
                    // Layered load: defaults < autumn.toml < AGENT_* env.
                    // `configure()` values were baked in at build time and win.
                    let mut loaded = AgentConfig::load().map_err(AgentError::into_autumn_error)?;
                    apply_overrides(&mut loaded, &plugin.config);
                    loaded.validate().map_err(AgentError::into_autumn_error)?;
                    if let Some(heartbeat) = &plugin.heartbeat {
                        heartbeat
                            .validate()
                            .map_err(AgentError::into_autumn_error)?;
                        let heartbeat = heartbeat.clone();
                        state.extension_or_insert_with(|| HeartbeatSettings(heartbeat));
                    }
                    let client =
                        client_from_config(&loaded).map_err(AgentError::into_autumn_error)?;
                    let provider_name = loaded.provider.as_str().to_owned();
                    let runtime = Arc::new(plugin.runtime(loaded, client));
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
    if configured.prompt_caching != defaults.prompt_caching {
        loaded.prompt_caching = configured.prompt_caching;
    }
    if let Some(secs) = configured.max_run_secs {
        loaded.max_run_secs = Some(secs);
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

    /// Run one turn in a persisted session with the plugin's settings.
    ///
    /// # Errors
    ///
    /// See [`Agent::run_in_session`].
    pub async fn chat(
        &self,
        session: impl Into<SessionId>,
        input: &str,
    ) -> Result<AgentTurn, AgentError> {
        let session = session.into();
        self.agent()
            .run_in_session(self.runtime.sessions().as_ref(), &session, input)
            .await
    }

    /// Like [`AgentHandle::chat`], with memory bound to `scope` (for
    /// example the signed-in user).
    ///
    /// # Errors
    ///
    /// See [`Agent::run_in_session`].
    pub async fn chat_as(
        &self,
        scope: MemoryScope,
        session: impl Into<SessionId>,
        input: &str,
    ) -> Result<AgentTurn, AgentError> {
        let session = session.into();
        self.runtime
            .agent_for(scope)
            .run_in_session(self.runtime.sessions().as_ref(), &session, input)
            .await
    }

    /// Continue a paused run with a reviewer's decisions, saving the
    /// transcript when the run belongs to a session.
    ///
    /// # Errors
    ///
    /// See [`Agent::resume`].
    pub async fn resume(
        &self,
        state: RunState,
        decisions: Vec<ApprovalDecision>,
    ) -> Result<AgentTurn, AgentError> {
        let run_id = state.run_id.clone();
        self.agent()
            .run_id(run_id)
            .resume_in_session(self.runtime.sessions().as_ref(), state, decisions)
            .await
    }

    /// Like [`AgentHandle::resume`], with memory bound to `scope`. Use the
    /// scope the paused run started with ([`AgentHandle::chat_as`]).
    ///
    /// # Errors
    ///
    /// See [`Agent::resume`].
    pub async fn resume_as(
        &self,
        scope: MemoryScope,
        state: RunState,
        decisions: Vec<ApprovalDecision>,
    ) -> Result<AgentTurn, AgentError> {
        let run_id = state.run_id.clone();
        self.runtime
            .agent_for(scope)
            .run_id(run_id)
            .resume_in_session(self.runtime.sessions().as_ref(), state, decisions)
            .await
    }

    /// The shared runtime, for stores and settings.
    #[must_use]
    pub fn runtime(&self) -> &AgentRuntime {
        &self.runtime
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
