//! Bounded memory blocks the agent curates itself.
//!
//! The model keeps durable notes in labelled [`MemoryBlock`]s, each with a
//! character limit: `memory` for what it learned, `user` for who it works
//! for. The limit forces the agent to consolidate instead of hoarding.
//!
//! Two rules keep memory cheap and safe:
//!
//! * **Frozen snapshot.** The loop renders the blocks into the system prompt
//!   once, when a run starts. Writes during the run persist at once but show
//!   up in the prompt only on the next run, so the prompt prefix stays stable
//!   and the provider's prompt cache keeps hitting.
//! * **Full means error.** An `add` that would overflow a block fails with a
//!   message that tells the model to remove or merge entries first.
//!
//! [`MemoryStore`] is the persistence seam. [`InMemoryMemoryStore`] serves
//! tests and single-process apps; back it with your database for anything
//! that must survive a restart. [`apply_op`] holds the edit rules, so every
//! store applies them the same way.

use std::collections::HashMap;
use std::fmt::Write as _;
use std::future::Future;
use std::pin::Pin;
use std::sync::{Arc, Mutex, PoisonError};

use futures::future::BoxFuture;
use serde::{Deserialize, Serialize};

use crate::error::{AgentError, ErrorKind};
use crate::tools::{Tool, ToolContext, ToolEffect};

/// Whose memory a run reads and writes.
///
/// One app can keep one memory for the whole agent, or one per user or per
/// tenant: the scope is the key.
#[derive(Debug, Clone, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
#[serde(transparent)]
pub struct MemoryScope(String);

impl MemoryScope {
    /// Wrap an app-chosen scope key.
    #[must_use]
    pub fn new(scope: impl Into<String>) -> Self {
        Self(scope.into())
    }

    /// The agent-wide scope, `"agent"`.
    #[must_use]
    pub fn agent() -> Self {
        Self::new("agent")
    }

    /// The scope as a string slice.
    #[must_use]
    pub fn as_str(&self) -> &str {
        &self.0
    }
}

/// One labelled, size-limited list of memory entries.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct MemoryBlock {
    /// Block label the model uses to address it (`memory`, `user`, ...).
    pub label: String,
    /// What the block is for, shown to the model.
    pub description: String,
    /// The entries, oldest first.
    pub entries: Vec<String>,
    /// Maximum total characters across all entries.
    pub limit_chars: usize,
}

impl MemoryBlock {
    /// An empty block.
    #[must_use]
    pub fn new(
        label: impl Into<String>,
        description: impl Into<String>,
        limit_chars: usize,
    ) -> Self {
        Self {
            label: label.into(),
            description: description.into(),
            entries: Vec::new(),
            limit_chars,
        }
    }

    /// Characters used by all entries.
    #[must_use]
    pub fn used_chars(&self) -> usize {
        self.entries.iter().map(|entry| entry.chars().count()).sum()
    }

    /// The default pair of blocks: `memory` (2200 chars) and `user`
    /// (1375 chars).
    #[must_use]
    pub fn defaults() -> Vec<Self> {
        vec![
            Self::new(
                "memory",
                "Facts, decisions, and lessons you want to remember across runs.",
                2_200,
            ),
            Self::new(
                "user",
                "Who you work for: preferences, style, standing instructions.",
                1_375,
            ),
        ]
    }
}

/// One edit to a memory block.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "action", rename_all = "snake_case")]
pub enum MemoryOp {
    /// Append a new entry.
    Add {
        /// Target block label.
        block: String,
        /// The entry text.
        text: String,
    },
    /// Replace the one entry that contains `old` with `text`.
    Replace {
        /// Target block label.
        block: String,
        /// A substring that identifies exactly one entry.
        old: String,
        /// The new entry text.
        text: String,
    },
    /// Remove the one entry that contains `old`.
    Remove {
        /// Target block label.
        block: String,
        /// A substring that identifies exactly one entry.
        old: String,
    },
}

impl MemoryOp {
    /// The label of the block this op edits.
    #[must_use]
    pub fn block(&self) -> &str {
        match self {
            Self::Add { block, .. } | Self::Replace { block, .. } | Self::Remove { block, .. } => {
                block
            }
        }
    }
}

/// Apply one edit to a block, enforcing the size limit and unique matches.
///
/// # Errors
///
/// Returns [`ErrorKind::Tool`] errors written for the model: an empty
/// entry, an overflow (with the numbers), or a `old` substring that matches
/// no entry or more than one.
pub fn apply_op(block: &mut MemoryBlock, op: &MemoryOp) -> Result<(), AgentError> {
    match op {
        MemoryOp::Add { text, .. } => {
            let text = non_empty(text)?;
            check_fits(block, None, text)?;
            block.entries.push(text.to_owned());
        }
        MemoryOp::Replace { old, text, .. } => {
            let text = non_empty(text)?;
            let index = find_unique(block, old)?;
            check_fits(block, Some(index), text)?;
            if let Some(entry) = block.entries.get_mut(index) {
                text.clone_into(entry);
            }
        }
        MemoryOp::Remove { old, .. } => {
            let index = find_unique(block, old)?;
            block.entries.remove(index);
        }
    }
    Ok(())
}

fn non_empty(text: &str) -> Result<&str, AgentError> {
    let text = text.trim();
    if text.is_empty() {
        return Err(AgentError::new(
            ErrorKind::Tool,
            "memory entries must not be empty",
        ));
    }
    Ok(text)
}

/// Fail when `text` (replacing entry `replacing`, if any) overflows the block.
fn check_fits(block: &MemoryBlock, replacing: Option<usize>, text: &str) -> Result<(), AgentError> {
    let freed = replacing
        .and_then(|index| block.entries.get(index))
        .map_or(0, |entry| entry.chars().count());
    let after = block
        .used_chars()
        .saturating_sub(freed)
        .saturating_add(text.chars().count());
    if after > block.limit_chars {
        return Err(AgentError::new(
            ErrorKind::Tool,
            format!(
                "memory block {:?} is full: this edit needs {after} of {} characters. Remove or merge entries first, then retry.",
                block.label, block.limit_chars
            ),
        ));
    }
    Ok(())
}

fn find_unique(block: &MemoryBlock, old: &str) -> Result<usize, AgentError> {
    let old = old.trim();
    if old.is_empty() {
        return Err(AgentError::new(
            ErrorKind::Tool,
            "give `old`: a substring of the entry to change",
        ));
    }
    let matches: Vec<usize> = block
        .entries
        .iter()
        .enumerate()
        .filter(|(_, entry)| entry.contains(old))
        .map(|(index, _)| index)
        .collect();
    match matches.as_slice() {
        [index] => Ok(*index),
        [] => Err(AgentError::new(
            ErrorKind::Tool,
            format!("no entry in {:?} contains {old:?}", block.label),
        )),
        _ => Err(AgentError::new(
            ErrorKind::Tool,
            format!(
                "{} entries in {:?} contain {old:?}: give a longer substring",
                matches.len(),
                block.label
            ),
        )),
    }
}

/// Render blocks as the frozen system-prompt section.
#[must_use]
pub fn render_snapshot(blocks: &[MemoryBlock]) -> String {
    let mut out = String::from(
        "## Memory\nYour persistent memory, as it was when this run started. Edit it with the `memory` tool; edits show here from the next run.\n",
    );
    for block in blocks {
        let _ = write!(
            out,
            "\n<memory block=\"{}\" used=\"{}/{}\">\n{}\n",
            block.label,
            block.used_chars(),
            block.limit_chars,
            block.description
        );
        for entry in &block.entries {
            out.push_str("- ");
            out.push_str(entry);
            out.push('\n');
        }
        out.push_str("</memory>\n");
    }
    out
}

/// Persistence for memory blocks.
pub trait MemoryStore: Send + Sync + std::fmt::Debug {
    /// Load every block for a scope. A new scope gets the store's default
    /// (empty) blocks.
    fn load<'a>(
        &'a self,
        scope: &'a MemoryScope,
    ) -> BoxFuture<'a, Result<Vec<MemoryBlock>, AgentError>>;

    /// Apply one edit and persist it. Implementations call [`apply_op`] so
    /// the rules match everywhere.
    fn apply<'a>(
        &'a self,
        scope: &'a MemoryScope,
        op: MemoryOp,
    ) -> BoxFuture<'a, Result<MemoryBlock, AgentError>>;
}

/// A process-local [`MemoryStore`]. Contents vanish on restart.
#[derive(Debug)]
pub struct InMemoryMemoryStore {
    template: Vec<MemoryBlock>,
    scopes: Mutex<HashMap<MemoryScope, Vec<MemoryBlock>>>,
}

impl Default for InMemoryMemoryStore {
    fn default() -> Self {
        Self::new(MemoryBlock::defaults())
    }
}

impl InMemoryMemoryStore {
    /// A store whose new scopes start with copies of `template`.
    #[must_use]
    pub fn new(template: Vec<MemoryBlock>) -> Self {
        Self {
            template,
            scopes: Mutex::new(HashMap::new()),
        }
    }

    /// Share the store.
    #[must_use]
    pub fn shared(self) -> Arc<dyn MemoryStore> {
        Arc::new(self)
    }
}

impl MemoryStore for InMemoryMemoryStore {
    fn load<'a>(
        &'a self,
        scope: &'a MemoryScope,
    ) -> BoxFuture<'a, Result<Vec<MemoryBlock>, AgentError>> {
        let blocks = self
            .scopes
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .get(scope)
            .cloned()
            .unwrap_or_else(|| self.template.clone());
        Box::pin(std::future::ready(Ok(blocks)))
    }

    fn apply<'a>(
        &'a self,
        scope: &'a MemoryScope,
        op: MemoryOp,
    ) -> BoxFuture<'a, Result<MemoryBlock, AgentError>> {
        let mut scopes = self.scopes.lock().unwrap_or_else(PoisonError::into_inner);
        let blocks = scopes
            .entry(scope.clone())
            .or_insert_with(|| self.template.clone());
        let result = match blocks.iter().position(|block| block.label == op.block()) {
            Some(index) => blocks
                .get_mut(index)
                .ok_or_else(|| unknown_block(op.block(), &[]))
                .and_then(|block| apply_op(block, &op).map(|()| block.clone())),
            None => Err(unknown_block(op.block(), blocks)),
        };
        drop(scopes);
        Box::pin(std::future::ready(result))
    }
}

fn unknown_block(label: &str, blocks: &[MemoryBlock]) -> AgentError {
    let known: Vec<&str> = blocks.iter().map(|block| block.label.as_str()).collect();
    AgentError::new(
        ErrorKind::Tool,
        format!("no memory block {label:?}; blocks: {known:?}"),
    )
}

/// The built-in `memory` tool, bound to one store and scope.
///
/// The agent adds it to a run on its own when memory is configured; you
/// rarely build one by hand.
#[derive(Debug, Clone)]
pub struct MemoryTool {
    store: Arc<dyn MemoryStore>,
    scope: MemoryScope,
}

impl MemoryTool {
    /// Bind the tool to a store and scope.
    #[must_use]
    pub fn new(store: Arc<dyn MemoryStore>, scope: MemoryScope) -> Self {
        Self { store, scope }
    }
}

impl Tool for MemoryTool {
    fn name(&self) -> &'static str {
        "memory"
    }

    fn description(&self) -> &'static str {
        "Edit your persistent memory. action=add appends `text` to `block`; action=replace swaps the entry containing `old` for `text`; action=remove deletes the entry containing `old`. Save durable facts and lessons, not logs."
    }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": ["add", "replace", "remove"]},
                "block": {"type": "string", "description": "Block label, e.g. memory or user."},
                "text": {"type": "string", "description": "New entry text (add, replace)."},
                "old": {"type": "string", "description": "Substring of the entry to change (replace, remove)."}
            },
            "required": ["action", "block"]
        })
    }

    fn effect(&self) -> ToolEffect {
        ToolEffect::Internal
    }

    fn execute<'a>(
        &'a self,
        input: serde_json::Value,
        _ctx: &'a ToolContext,
    ) -> Pin<Box<dyn Future<Output = Result<serde_json::Value, AgentError>> + Send + 'a>> {
        Box::pin(async move {
            let op: MemoryOp = serde_json::from_value(input).map_err(|err| {
                AgentError::with_source(ErrorKind::Tool, format!("invalid memory edit: {err}"), err)
            })?;
            let block = self.store.apply(&self.scope, op).await?;
            Ok(serde_json::json!({
                "ok": true,
                "block": block.label,
                "used": block.used_chars(),
                "limit": block.limit_chars,
            }))
        })
    }
}

#[cfg(test)]
mod tests;
