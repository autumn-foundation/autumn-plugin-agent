//! Health indicator for the agent plugin.
//!
//! [`AgentHealthIndicator`] pings the provider with the cheap, read-only
//! `list_models` call and folds the result into Autumn's `/actuator/health`
//! and `/ready` endpoints. It registers in the [`HealthOnly`] group: a
//! degraded LLM provider must not block rolling deploys.
//!
//! [`HealthOnly`]: autumn_web::actuator::IndicatorGroup::HealthOnly

use std::collections::HashMap;
use std::sync::{Arc, OnceLock};

use autumn_web::actuator::{HealthCheckOutput, HealthIndicator, HealthStatus, IndicatorGroup};

use crate::agent::AgentRuntime;

/// Health indicator that pings the configured LLM provider.
///
/// The indicator shares the plugin startup hook's runtime holder, so the
/// check always hits the exact provider the running app uses. Before the
/// startup hook installs the runtime the check reports DOWN (uninitialized).
pub struct AgentHealthIndicator {
    active: Arc<OnceLock<Arc<AgentRuntime>>>,
}

impl std::fmt::Debug for AgentHealthIndicator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AgentHealthIndicator")
            .field("initialized", &self.active.get().is_some())
            .finish_non_exhaustive()
    }
}

impl AgentHealthIndicator {
    /// Build the indicator around the startup hook's shared runtime holder.
    ///
    /// The plugin creates the holder and hands it to both this indicator and
    /// its `on_startup` hook; the hook installs the real [`AgentRuntime`]
    /// once the layered configuration has loaded.
    #[must_use]
    pub const fn new(active: Arc<OnceLock<Arc<AgentRuntime>>>) -> Self {
        Self { active }
    }
}

impl HealthIndicator for AgentHealthIndicator {
    fn check(&self) -> futures::future::BoxFuture<'_, HealthCheckOutput> {
        Box::pin(run_check(self.active.get()))
    }

    /// A sick provider must not gate deploys; it only shows in `/actuator/health`.
    fn group(&self) -> IndicatorGroup {
        IndicatorGroup::HealthOnly
    }
}

/// Run one health check against the active runtime, if startup installed one.
///
/// Splitting the status-mapping logic out keeps it testable without booting a
/// whole Autumn app.
async fn run_check(active: Option<&Arc<AgentRuntime>>) -> HealthCheckOutput {
    let mut details = HashMap::new();
    let Some(runtime) = active else {
        details.insert(
            "error".to_owned(),
            serde_json::Value::String("agent plugin has not finished starting".to_owned()),
        );
        return HealthCheckOutput {
            status: HealthStatus::Down,
            details,
        };
    };
    details.insert(
        "provider".to_owned(),
        serde_json::Value::String(runtime.config().provider.as_str().to_owned()),
    );
    match runtime.client().list_models().await {
        Ok(models) => {
            details.insert(
                "models".to_owned(),
                serde_json::Value::Number(models.len().into()),
            );
            HealthCheckOutput {
                status: HealthStatus::Up,
                details,
            }
        }
        Err(err) => {
            details.insert(
                "error".to_owned(),
                serde_json::Value::String(err.to_string()),
            );
            HealthCheckOutput {
                status: HealthStatus::Down,
                details,
            }
        }
    }
}

#[cfg(test)]
mod tests;
