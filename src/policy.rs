//! Tool policies: decide, per call, whether the agent may act.
//!
//! The loop asks the [`ToolPolicy`] about every tool call before it runs.
//! There are three answers:
//!
//! * [`ToolDecision::Allow`] — run the call.
//! * [`ToolDecision::Deny`] — skip the call; the model sees the reason.
//! * [`ToolDecision::RequireApproval`] — pause the run. The loop returns
//!   [`AgentOutcome::AwaitingApproval`](crate::agent::AgentOutcome::AwaitingApproval)
//!   with a serializable [`RunState`](crate::agent::RunState). Store it, ask a
//!   person, then continue with
//!   [`Agent::resume`](crate::agent::Agent::resume).
//!
//! [`ToolRules`] covers the common cases: per-tool rules plus a default per
//! [`ToolEffect`].

use std::collections::HashMap;
use std::sync::Arc;

use futures::future::BoxFuture;
use serde::{Deserialize, Serialize};

use crate::hooks::RunInfo;
use crate::tools::{Tool, ToolCall, ToolEffect};

/// A policy's answer for one tool call.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "decision", rename_all = "snake_case")]
pub enum ToolDecision {
    /// Run the call.
    Allow,
    /// Pause the run until a person approves or rejects the call.
    RequireApproval {
        /// Why the call needs a person, shown to the reviewer.
        reason: String,
    },
    /// Do not run the call. The model sees the reason as the tool result.
    Deny {
        /// Why the call is refused, shown to the model.
        reason: String,
    },
}

/// Decides whether the agent may run a tool call.
pub trait ToolPolicy: Send + Sync + std::fmt::Debug {
    /// Decide on one call. `tool` is `None` when the model named a tool
    /// that does not exist; the loop reports that to the model either way.
    fn decide<'a>(
        &'a self,
        call: &'a ToolCall,
        tool: Option<&'a dyn Tool>,
        info: &'a RunInfo,
    ) -> BoxFuture<'a, ToolDecision>;
}

/// Allows every call. The default policy.
#[derive(Debug, Clone, Copy, Default)]
pub struct AllowAll;

impl ToolPolicy for AllowAll {
    fn decide<'a>(
        &'a self,
        _call: &'a ToolCall,
        _tool: Option<&'a dyn Tool>,
        _info: &'a RunInfo,
    ) -> BoxFuture<'a, ToolDecision> {
        Box::pin(std::future::ready(ToolDecision::Allow))
    }
}

/// How [`ToolRules`] treats a tool or an effect class.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum Rule {
    /// Run without asking.
    Allow,
    /// Pause for a person.
    Ask,
    /// Refuse, with this reason.
    Deny(String),
}

/// Per-tool rules plus a default per [`ToolEffect`] — the shape of "custom
/// rules" in always-on assistants: act freely on reads, ask before acting
/// on the world.
///
/// Lookup order: the rule for the tool's name, then the rule for its
/// effect, then [`Rule::Allow`].
///
/// ```rust
/// use autumn_plugin_agent::policy::{Rule, ToolRules};
/// use autumn_plugin_agent::tools::ToolEffect;
///
/// let rules = ToolRules::new()
///     .effect(ToolEffect::External, Rule::Ask)
///     .tool("delete_account", Rule::Deny("never delete accounts".into()))
///     .tool("send_digest", Rule::Allow);
/// ```
#[derive(Debug, Clone, Default)]
pub struct ToolRules {
    by_name: HashMap<String, Rule>,
    by_effect: HashMap<ToolEffect, Rule>,
}

impl ToolRules {
    /// Start with no rules: every call is allowed.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Rules for an unattended run: read-only and internal tools run, every
    /// other tool is refused.
    #[must_use]
    pub fn read_only() -> Self {
        let deny = Rule::Deny("this run may only read data and update its own notes".to_owned());
        Self::new()
            .effect(ToolEffect::Write, deny.clone())
            .effect(ToolEffect::External, deny)
    }

    /// Ask a person before any [`ToolEffect::Write`] or
    /// [`ToolEffect::External`] call.
    #[must_use]
    pub fn ask_before_acting() -> Self {
        Self::new()
            .effect(ToolEffect::Write, Rule::Ask)
            .effect(ToolEffect::External, Rule::Ask)
    }

    /// Set the rule for one tool name. It wins over effect rules.
    #[must_use]
    pub fn tool(mut self, name: impl Into<String>, rule: Rule) -> Self {
        self.by_name.insert(name.into(), rule);
        self
    }

    /// Set the default rule for every tool with this effect.
    #[must_use]
    pub fn effect(mut self, effect: ToolEffect, rule: Rule) -> Self {
        self.by_effect.insert(effect, rule);
        self
    }

    /// The decision for a call, without the async wrapper.
    #[must_use]
    pub fn decide_now(&self, call: &ToolCall, tool: Option<&dyn Tool>) -> ToolDecision {
        let rule = self.by_name.get(&call.name).or_else(|| {
            tool.map(Tool::effect)
                .and_then(|effect| self.by_effect.get(&effect))
        });
        match rule {
            None | Some(Rule::Allow) => ToolDecision::Allow,
            Some(Rule::Ask) => ToolDecision::RequireApproval {
                reason: format!("{} needs approval before it runs", call.name),
            },
            Some(Rule::Deny(reason)) => ToolDecision::Deny {
                reason: reason.clone(),
            },
        }
    }
}

impl ToolPolicy for ToolRules {
    fn decide<'a>(
        &'a self,
        call: &'a ToolCall,
        tool: Option<&'a dyn Tool>,
        _info: &'a RunInfo,
    ) -> BoxFuture<'a, ToolDecision> {
        Box::pin(std::future::ready(self.decide_now(call, tool)))
    }
}

/// Combines policies and keeps the strictest answer: any
/// [`ToolDecision::Deny`] wins, then any [`ToolDecision::RequireApproval`],
/// else [`ToolDecision::Allow`].
///
/// The heartbeat uses it to put read-only rules on top of the app's policy.
#[derive(Debug, Clone)]
pub struct Strictest(Vec<Arc<dyn ToolPolicy>>);

impl Strictest {
    /// Combine these policies.
    #[must_use]
    pub const fn new(policies: Vec<Arc<dyn ToolPolicy>>) -> Self {
        Self(policies)
    }
}

impl ToolPolicy for Strictest {
    fn decide<'a>(
        &'a self,
        call: &'a ToolCall,
        tool: Option<&'a dyn Tool>,
        info: &'a RunInfo,
    ) -> BoxFuture<'a, ToolDecision> {
        Box::pin(async move {
            let mut verdict = ToolDecision::Allow;
            for policy in &self.0 {
                match policy.decide(call, tool, info).await {
                    deny @ ToolDecision::Deny { .. } => return deny,
                    ask @ ToolDecision::RequireApproval { .. } => {
                        if verdict == ToolDecision::Allow {
                            verdict = ask;
                        }
                    }
                    ToolDecision::Allow => {}
                }
            }
            verdict
        })
    }
}

#[cfg(test)]
mod tests;
