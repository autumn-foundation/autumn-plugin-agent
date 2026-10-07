# Always-on agent harnesses — research notes (2026-10-07)

This note records what the current always-on agents do and what this crate
takes from them. It drives the 0.2 release (see
[ADR 0001](../adr/0001-always-on-harness.md)).

Muse and Dots launched in September 2026. The facts about them come from
launch coverage and vendor docs, cited inline. Read them as public claims,
not verified internals.

## The three named agents

### Muse (Meta, launched 2026-09-08)

A consumer personal agent. It sends email, books travel, fills forms, and
"keeps working after people close the app". It runs in the app, on the web,
and in WhatsApp.
([Meta](https://about.fb.com/news/2026/09/introducing-muse),
[Vellum breakdown](https://www.vellum.ai/blog/official-muse-breakdown))

- **Isolation.** Each user gets a cloud VM ("Muse Secure VM"). The harness
  runs in an unprivileged runtime cell, apart from credential storage.
- **Egress gate.** A separate *Sentinel* agent approves all outbound
  network traffic.
- **Secret indirection.** The agent holds only surrogate tokens. Sentinel
  swaps in the real credential at the network boundary, so the model never
  sees a secret.
- **Taint tracking.** After the agent reads untrusted content, it loses
  autonomous network access until a person approves.
- **Confirmations.** Irreversible actions (email, payment, high-risk forms)
  pause the run and show a structured confirmation. A full audit trail lists
  actions taken and planned.
- **Reporting.** It reports back only at meaningful milestones.

### Dots (OpenAI, DevDay 2026-09-29)

Always-on ChatGPT agents. Each dot has its own cloud computer, a browser,
and app connectors. Users give a dot standing goals.
([SiliconANGLE](https://siliconangle.com/2026/09/29/openai-launches-dots-always-on-ai-agents-in-chatgpt-with-their-own-cloud-computers/),
[deep dive](https://flaviocopes.com/openai-dots.md))

- **Harness.** The open-source Codex harness powers dots
  ([Decrypt](https://decrypt.co/379584/openai-ai-agents-computers-devday-2026-everything-announced)).
  The Codex app server models work as *Thread → Turn → Item*, and persisted
  items are the context for later turns
  ([Codex app server](https://developers.openai.com/codex/app-server)).
- **Wake-ups.** The dot schedules itself, follows a fixed schedule, or
  reacts to an event trigger. It watches only what you tell it to watch.
- **Proactive research.** While idle, a dot reads connected apps with
  read-only tools and keeps private notes.
- **Memory.** Three layers: live context, the user's ChatGPT memory, and
  the dot's own notes ("not a transcript").
- **Custom Rules.** Each action is one of: act without asking, act if
  pre-approved, ask first, or hand off.
- **Auto-review.** A separate review checks consequential actions against
  instructions, permissions, and rules.

### Hermes Agent (Nous Research, open source, MIT)

A self-hosted agent in Python.
([GitHub](https://github.com/NousResearch/hermes-agent),
[architecture](https://hermes-agent.nousresearch.com/docs/developer-guide/architecture/))

- **One core loop** serves the CLI, the chat gateway, cron, and the API.
  The prompt has tiers: a stable identity and tools first, then context
  files, then volatile memory.
- **Compaction.** A pluggable context engine summarizes the middle turns
  when the context passes a threshold.
- **Prompt caching.** Anthropic cache breakpoints, plus one rule: the system
  prompt does not change during a conversation.
- **Approval.** Dangerous commands need approval.
- **Gateway.** One process with adapters for Telegram, Slack, Discord,
  email, and more. A platform event maps to a session key, then to an agent
  run with that session's history.
- **Cron**
  ([docs](https://hermes-agent.nousresearch.com/docs/user-guide/features/cron)):
  - A 60-second tick with a file lock, so no job runs twice.
  - `[SILENT]` suppresses delivery.
  - A pre-run script can skip the model call.
  - Transient failures retry with backoff.
- **Memory**
  ([docs](https://hermes-agent.nousresearch.com/docs/user-guide/features/memory)):
  - Two bounded files: `MEMORY.md` (~2,200 characters) and `USER.md`
    (~1,375 characters).
  - Both inject as a frozen snapshot at session start.
  - The memory tool has three actions: `add`, `replace`, and `remove`.
  - A full block returns an error that tells the agent to consolidate.
- **Skills**
  ([docs](https://hermes-agent.nousresearch.com/docs/user-guide/features/skills)):
  - `SKILL.md` files load with progressive disclosure: the list first, then
    the body on demand.
  - The agent can write its own skills.

## The wider state of the art

- **Claude Agent SDK / Claude Code.** Hooks run deterministic code at fixed
  points (`PreToolUse`, `PostToolUse`, `Stop`, ...). Subagents act as
  context firewalls. Skills use progressive disclosure.
- **Anthropic context management.** Server-side tool-result clearing and
  compaction (beta) are now preferred over SDK-side compaction. Clearing
  invalidates the prompt cache, so clear in large steps
  ([docs](https://platform.claude.com/docs/en/build-with-claude/context-editing)).
- **OpenAI Agents SDK.** Tools declare `needsApproval`. The run returns
  *interruptions* and a serializable `RunState` that resumes after approve or
  reject ([docs](https://developers.openai.com/api/docs/guides/agents/running-agents.md)).
- **Durable execution.** Temporal runs the agent loop as a workflow and each
  model call as an activity, so steps do not repeat after a crash
  ([Temporal](https://temporal.io/blog/announcing-openai-agents-sdk-integration)).
- **OpenClaw heartbeat.** A check-in prompt runs every 30 minutes by
  default. The agent replies `HEARTBEAT_OK` when nothing needs attention.
  The heartbeat is the largest cost driver
  ([docs](https://clawdocs.org/architecture/heartbeat)).
- **OpenClaw loop detection.** It keeps a rolling history of calls, warns
  on repeats, and blocks after more. It also catches ping-pong between two
  calls ([docs](https://docs.openclaw.ai/tools/loop-detection)).
- **Letta.** Memory is labelled, size-limited blocks. *Sleep-time* agents
  rewrite memory while the main agent is idle
  ([docs](https://docs.letta.com/guides/agents/sleep-time-agents)).

## Patterns every always-on harness shares

1. A persisted session (thread → turn → item).
2. Wake-ups from schedules, heartbeats, events, and the agent itself, with
   a silent "nothing to do" path.
3. A per-action policy (allow / ask / deny) with resumable pauses.
4. Read-only autonomy while proactive.
5. Bounded memory, injected as a cache-friendly frozen snapshot.
6. Compaction and loop guards.
7. Procedural memory as skills.
8. Delegation to subagents with their own context.
9. Prompt caching, because the same prefix runs again and again.

## What this crate adopts

| Pattern | Source | In 0.2 |
|---------|--------|--------|
| Persisted sessions | Codex threads, Hermes sessions | `SessionStore`, `Agent::run_in_session`, `AgentHandle::chat` |
| Summarizing compaction | Hermes context engine | `session::Compaction` |
| Approval gates with resumable state | OpenAI `RunState`, Dots Custom Rules, Muse confirmations | `ToolPolicy`, `ToolRules`, `AgentOutcome::AwaitingApproval`, `Agent::resume`, `agent_resume` job |
| Effect classes for tools | Dots read-only research, Muse taint | `ToolEffect::{ReadOnly, Internal, Write, External}` |
| Lifecycle hooks | Claude Code hooks | `AgentHooks` (`before_model`, `before_tool` → `Block`/`Modify`, `after_tool`, `on_outcome`) |
| Loop detection | OpenClaw | `LoopGuard` → `AgentOutcome::LoopDetected` |
| Bounded memory with a frozen snapshot | Hermes `MEMORY.md`/`USER.md`, Letta blocks | `MemoryStore`, `MemoryBlock`, `memory` tool |
| Skills with progressive disclosure | Claude Skills, Hermes | `Skill::parse` (`SKILL.md`), `load_skill` tool |
| Subagents | Claude Code, Hermes `delegate_tool` | `delegate::AgentTool` |
| Heartbeat with a silent ack | OpenClaw `HEARTBEAT_OK`, Hermes `[SILENT]` | `proactive::Heartbeat` (Autumn scheduled task, fleet-coordinated) |
| Skip the model call when idle | Hermes `wakeAgent:false` | `Heartbeat::precheck` |
| Agent-scheduled wake-ups | Dots self-scheduling | `schedule_followup` tool (Autumn `enqueue_in`) with a chain cap |
| Delivery channel | Hermes gateway, Dots multi-channel | `Delivery` trait + `Report` |
| Prompt caching + cache accounting | Hermes, Anthropic docs | Anthropic `cache_control` breakpoints; `TokenUsage::{cache_read_tokens, cache_write_tokens}` |
| Wall-clock budget | Durable-run practice | `Agent::max_duration`, `BudgetKind::Deadline`, `max_run_secs` |
| Pollable background runs | — | `enqueue_agent_run_tracked` (Autumn 0.8 tracked jobs) |
| Idempotency context for tools | Temporal / durable runs | `ToolContext { run_id, call_id, session_id, step }`; run id fixed at enqueue |

## Deferred, with reasons

- **MCP client toolset.** It needs a JSON-RPC transport and the `rmcp`
  dependency. It belongs behind an `mcp` feature in its own change.
- **Durable step checkpoints / Harvest integration.** Checkpointing each
  step needs a run store. Harvest activities are the natural Autumn fit. The
  `RunState` and `ToolContext` added here are the groundwork.
- **Cost budgets and per-tenant ledgers.** Price tables drift. A
  `BudgetLedger` trait backed by the database is the next step.
- **Sleep-time reflection.** It needs sessions and memory, which now exist.
  An idle-triggered `reflect` job can follow.
- **Agent-written skills (`skill_manage`).** This is close to a product
  feature. It needs an approval gate (now available) and a skill store.
- **Server-side Anthropic context editing.** It is a beta API, and it
  invalidates the cache. Client-side compaction ships first.
- **Streaming events, secret indirection, sandboxing.** Streaming stays on
  the 0.1 follow-up list. Sandboxing stays the tool author's job.
