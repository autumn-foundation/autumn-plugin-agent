# Changelog

All notable changes to this project follow [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Changed

- Background-job enqueue API: handlers now call the documented
  `pub async fn enqueue_agent_run(AgentRunArgs) -> AutumnResult<()>`
  (re-exported at the crate root). The `#[job]`-generated `RunAgentJob`
  stays crate-internal because the macro emits it without documentation.

### Fixed

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
