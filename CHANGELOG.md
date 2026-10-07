# Changelog

All notable changes to this project follow [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

Version 0.2.0: always-on agent primitives. Research and rationale:
`docs/research/always-on-agents.md`, `docs/adr/0001-always-on-harness.md`.

### Added

- Multi-turn runs: `Agent::run_turn(history, input)` returns an `AgentTurn`
  with the transcript.
- Sessions: `SessionStore` trait + `InMemorySessionStore`,
  `Agent::run_in_session`, `AgentHandle::chat` / `chat_as`. Transcripts
  (`ChatMessage`, `ContentPart`) are now `Serialize`/`Deserialize`.
- Compaction: `session::Compaction` summarizes old turns with one model call
  when a session passes `trigger_tokens`; the split lands on a user turn.
- Approvals: `ToolPolicy`, `ToolDecision`, `ToolRules` (per-tool and
  per-effect rules: allow / ask / deny), `Strictest`. An "ask" pauses the run
  with `AgentOutcome::AwaitingApproval { state: RunState }`; resume with
  `Agent::resume` / `resume_in_session` (approve, edit, reject), the
  `agent_resume` job, or `AgentHandle::resume` / `resume_as`.
- `ToolEffect` (`ReadOnly`, `Internal`, `Write`, `External`) on every tool;
  `FnTool::effect`.
- `ToolContext` (`run_id`, `call_id`, `session_id`, `step`) passed to every
  tool; `FnTool::with_context`.
- Lifecycle hooks: `AgentHooks` (`before_model`, `after_model`,
  `before_tool` → `Continue`/`Modify`/`Block`, `after_tool`, `on_outcome`).
- Loop guard: `LoopGuard` warns the model after repeated identical calls
  (name + arguments + result) and stops with `AgentOutcome::LoopDetected`.
- Wall-clock deadline: `Agent::max_duration`, `[agent] max_run_secs`,
  `BudgetKind::Deadline`. In-flight model and tool calls are cancelled.
- Parallel tool calls within a round (default on;
  `Agent::parallel_tool_calls(false)` to opt out). Result order is kept.
- Memory: `MemoryStore`, `MemoryBlock` (default `memory` 2200 chars and
  `user` 1375 chars), `MemoryOp`, `apply_op`, `InMemoryMemoryStore`, and the
  built-in `memory` tool. Blocks render into the system prompt as a frozen
  snapshot per run.
- Skills: `Skill::parse` for `SKILL.md` front matter, an index in the system
  prompt, and the built-in `load_skill` tool.
- Subagents: `delegate::AgentTool` runs a child `Agent` as one tool.
- Heartbeat: `proactive::Heartbeat` (`every` / `cron`, time zone, session,
  memory scope, `precheck`, `per_replica`, `allow_actions`) registers the
  fleet-coordinated `agent_heartbeat` Autumn task. Ticks run read-only by
  default; a `HEARTBEAT_OK` answer is not delivered.
- Follow-ups: the `schedule_followup` tool (enable with
  `AgentPlugin::followups(max_delay)`) enqueues a delayed `agent_run` in the
  same session; chains stop at `FollowupTool::MAX_CHAIN`.
- Delivery: `Delivery` trait, `Report`, `ReportSource`, `LogDelivery`.
- Tracked background runs: `enqueue_agent_run_tracked`,
  `enqueue_agent_resume[_tracked]`, `enqueue_agent_run_in`. Tracked runs
  report per-step progress and store the `AgentOutcome` JSON as the result.
- Anthropic prompt caching (`cache_control` on the system prompt, the last
  tool, and the newest message; `[agent] prompt_caching`, default on).
  `TokenUsage` gains `cache_read_tokens` / `cache_write_tokens`; OpenAI's
  `prompt_tokens_details.cached_tokens` is parsed too.
- `RunId` / `SessionId` newtypes; the run id is fixed at enqueue time so job
  retries keep it.
- `AgentPlugin` builder: `hook`, `policy`, `session_store`, `memory_store`,
  `skill(s)`, `compaction`, `delivery`, `heartbeat`, `followups`.
- `examples/always_on_agent.rs`.

### Changed

- **Breaking:** `Tool::execute(input, ctx: &ToolContext)`.
- **Breaking:** `AgentOutcome` and `BudgetKind` are `#[non_exhaustive]`,
  serializable, and no longer `Eq`; `AgentOutcome::steps_used` / `usage` are
  no longer `const`.
- **Breaking:** `AgentRunArgs` is `#[non_exhaustive]`; build it with
  `AgentRunArgs::new(prompt)` and the builder methods. Old JSON payloads
  still decode.
- **Breaking:** `TokenUsage` has two new fields; use `TokenUsage::new`.
- **Breaking:** `AgentRunArgs` and `enqueue_agent_run` moved to the new
  `jobs` module (still re-exported at the crate root and from `agent`).
- `truncate_history` keeps the first user message (the task) when it fits
  and drops tool results whose call was cut, so providers never see an
  orphaned `tool_result`.
- A final answer is now recorded in the transcript; an overshooting
  tool-call turn is not.

- Upgrade to Autumn 0.8: `autumn-web` requirement moves from `>=0.7, <0.8`
  to `>=0.8, <0.9`. The plugin surface it uses (`Plugin`, `config_section`,
  `on_startup`, `#[job]`, `HealthIndicator`, `FromRequestParts<AppState>`)
  compiles unchanged. Apps on Autumn 0.7 must stay on the 0.1 line.
- Background-job enqueue API: handlers now call the documented
  `pub async fn enqueue_agent_run(AgentRunArgs) -> AutumnResult<()>`
  (re-exported at the crate root). The `#[job]`-generated `RunAgentJob`
  stays crate-internal because the macro emits it without documentation.

### Fixed

- Every rustdoc intra-doc link resolves (`cargo doc` with `-D warnings`).
- Health indicator now checks the exact provider client the startup hook
  installed in the shared `AgentRuntime` (via a `OnceLock` holder), instead
  of building a second client from the builder's pre-layering config. Before
  startup completes it reports DOWN (uninitialized).
- Token budget enforcement: the per-call output cap now reserves the
  estimated input-history cost first, and a final answer whose
  provider-reported usage overshoots `max_tokens` returns
  `BudgetExhausted { Tokens }` instead of `Completed`.
- Documented `max_steps` semantics: it counts tool-execution rounds; the
  final answering call does not consume a step.

## [0.1.0] - 2026-09-29

### Added

- `LlmClient` trait (chat completions with tools, `list_models`) with two
  HTTP implementations: `OpenAiCompatibleClient` (`POST /chat/completions`,
  Bearer auth) and `AnthropicClient` (`POST /v1/messages`, `x-api-key` +
  `anthropic-version` headers). No LLM SDK dependencies.
- `Tool` trait (`name`, `description`, JSON input schema, async `execute`)
  plus `FnTool`, an adapter that wraps a plain async function — the pattern
  for sharing logic between an Autumn handler and an agent tool.
- Agent loop (`Agent::run`): system prompt + message history + tools;
  iterates model → tool calls → results until a final answer or budget
  exhaustion. `max_steps` and `max_tokens` budgets enforced;
  `AgentOutcome::BudgetExhausted` is a normal result, not an error.
- `agent_run` Autumn background job (`#[job]`): enqueue agent runs from
  handlers via `enqueue_agent_run(AgentRunArgs { .. })`.
- `AgentPlugin` (`::new()`, `.configure(|c| ...)`, `.tool(...)`) implementing
  Autumn's `Plugin` trait: declares the `[agent]` config section, registers
  the job, installs the health indicator, fail-fast startup hook.
- `AgentHandle` extractor (`FromRequestParts<AppState>`) for handlers.
- `[agent]` config: `provider` (`openai-compatible` | `anthropic`), `model`,
  `base_url`, `request_timeout_secs`, `max_steps`, `max_tokens`,
  `system_prompt`. Layered: defaults < `autumn.toml` < `AGENT_*` env <
  `.configure()`. `AGENT_API_KEY` is env-only; a key in the config file
  fails loading.
- Health indicator: `list_models` ping, health-only group (never gates
  deploys).
- `examples/weather_agent.rs`: agent + `get_weather` tool against a scripted
  mock; documents pointing at real providers.
- Tests: axum mock providers, scripted-client loop tests (round trip, step
  and token budget exhaustion, unknown tools, tool failures, output
  truncation), error-kind mapping, config layering, proptest truncation and
  budget invariants.

### Follow-ups (explicitly deferred)

- **Streaming**: `chat_stream` on `LlmClient` yielding SSE content deltas, and
  a streaming agent loop. Non-streaming is the complete v0.1 surface.
- **Tracked job results**: persist `agent_run` outcomes to Autumn's tracked
  job result store so handlers can poll for answers.
- **MCP client tools**: expose remote MCP servers as `Tool` implementations.
- **Usage metering hook**: per-run token callback for billing.
