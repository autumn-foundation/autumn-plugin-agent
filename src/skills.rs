//! Skills: reusable instructions the agent loads on demand.
//!
//! A [`Skill`] is a named block of instructions (a checklist, a house style,
//! a playbook). Skills use progressive disclosure: the system prompt lists
//! only each skill's name and one-line description, and the model reads the
//! full body through the `load_skill` tool when a task needs it. Ten skills
//! cost ten lines of prompt, not ten documents.
//!
//! Skills parse from `SKILL.md` files with a small front matter block:
//!
//! ```markdown
//! ---
//! name: weekly-digest
//! description: How to write the Monday digest for the ops team.
//! ---
//! 1. Pull last week's incidents.
//! 2. ...
//! ```

use std::fmt::Write as _;
use std::future::Future;
use std::pin::Pin;
use std::sync::Arc;

use serde::{Deserialize, Serialize};

use crate::error::{AgentError, ErrorKind};
use crate::tools::{Tool, ToolContext, ToolEffect};

/// One skill: a name, a one-line description, and the instruction body.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Skill {
    /// Name the model passes to `load_skill`. Keep it `kebab-case`.
    pub name: String,
    /// One line that tells the model when the skill applies.
    pub description: String,
    /// The full instructions.
    pub body: String,
}

impl Skill {
    /// Build a skill from parts.
    #[must_use]
    pub fn new(
        name: impl Into<String>,
        description: impl Into<String>,
        body: impl Into<String>,
    ) -> Self {
        Self {
            name: name.into(),
            description: description.into(),
            body: body.into(),
        }
    }

    /// Parse a `SKILL.md` document: `---` front matter with `name:` and
    /// `description:` lines, then the body.
    ///
    /// Values may be wrapped in single or double quotes. Other front matter
    /// keys are ignored.
    ///
    /// # Errors
    ///
    /// Returns [`ErrorKind::Config`] when the front matter is missing,
    /// unterminated, or lacks a non-empty `name` or `description`.
    pub fn parse(document: &str) -> Result<Self, AgentError> {
        let document = document.trim_start_matches('\u{feff}');
        let mut lines = document.lines();
        if lines.next().map(str::trim) != Some("---") {
            return Err(AgentError::new(
                ErrorKind::Config,
                "skill document must start with a `---` front matter line",
            ));
        }
        let mut name = None;
        let mut description = None;
        let mut closed = false;
        for line in lines.by_ref() {
            if line.trim() == "---" {
                closed = true;
                break;
            }
            if let Some((key, value)) = line.split_once(':') {
                let value = unquote(value.trim());
                match key.trim() {
                    "name" => name = Some(value.to_owned()),
                    "description" => description = Some(value.to_owned()),
                    _ => {}
                }
            }
        }
        if !closed {
            return Err(AgentError::new(
                ErrorKind::Config,
                "skill front matter is not closed with `---`",
            ));
        }
        let body = lines.collect::<Vec<_>>().join("\n").trim().to_owned();
        let name = name.filter(|name| !name.is_empty()).ok_or_else(|| {
            AgentError::new(ErrorKind::Config, "skill front matter needs a `name`")
        })?;
        let description = description
            .filter(|description| !description.is_empty())
            .ok_or_else(|| {
                AgentError::new(
                    ErrorKind::Config,
                    format!("skill {name:?} needs a `description`"),
                )
            })?;
        Ok(Self {
            name,
            description,
            body,
        })
    }
}

fn unquote(value: &str) -> &str {
    for quote in ['"', '\''] {
        if let Some(inner) = value
            .strip_prefix(quote)
            .and_then(|rest| rest.strip_suffix(quote))
        {
            return inner;
        }
    }
    value
}

/// Render the skill index for the system prompt.
#[must_use]
pub fn render_index(skills: &[Skill]) -> String {
    let mut out = String::from(
        "## Skills\nBefore you do a task that a skill covers, read it with the `load_skill` tool.\n",
    );
    for skill in skills {
        let _ = writeln!(out, "- {}: {}", skill.name, skill.description);
    }
    out
}

/// The built-in `load_skill` tool. The agent adds it when skills are set.
#[derive(Debug, Clone)]
pub struct SkillTool {
    skills: Arc<[Skill]>,
}

impl SkillTool {
    /// Serve these skills.
    #[must_use]
    pub const fn new(skills: Arc<[Skill]>) -> Self {
        Self { skills }
    }
}

impl SkillTool {
    fn lookup(&self, name: &str) -> Result<serde_json::Value, AgentError> {
        self.skills
            .iter()
            .find(|skill| skill.name == name)
            .map(|skill| serde_json::json!({"name": skill.name, "instructions": skill.body}))
            .ok_or_else(|| {
                let known: Vec<&str> = self
                    .skills
                    .iter()
                    .map(|skill| skill.name.as_str())
                    .collect();
                AgentError::new(
                    ErrorKind::Tool,
                    format!("no skill {name:?}; skills: {known:?}"),
                )
            })
    }
}

impl Tool for SkillTool {
    fn name(&self) -> &'static str {
        "load_skill"
    }

    fn description(&self) -> &'static str {
        "Read the full instructions of one skill from the Skills list."
    }

    fn input_schema(&self) -> serde_json::Value {
        serde_json::json!({
            "type": "object",
            "properties": {"name": {"type": "string", "description": "Skill name."}},
            "required": ["name"]
        })
    }

    fn effect(&self) -> ToolEffect {
        ToolEffect::ReadOnly
    }

    fn execute<'a>(
        &'a self,
        input: serde_json::Value,
        _ctx: &'a ToolContext,
    ) -> Pin<Box<dyn Future<Output = Result<serde_json::Value, AgentError>> + Send + 'a>> {
        let result = input
            .get("name")
            .and_then(serde_json::Value::as_str)
            .ok_or_else(|| AgentError::new(ErrorKind::Tool, "give the skill `name`"))
            .and_then(|name| self.lookup(name));
        Box::pin(std::future::ready(result))
    }
}

#[cfg(test)]
mod tests;
