# CLAUDE.md — autumn-plugin-agent

Agent guidance for working in this repo.

## Commands

```sh
export PATH="$HOME/.cargo/bin:$PATH"
export CARGO_BUILD_JOBS=2
export CARGO_TARGET_DIR=~/workspace/autumn-arena/target   # shared arena dir; never cargo clean
export TMPDIR=~/workspace/.tmp-cargo                      # /tmp is a 512MB tmpfs

cargo fmt --all -- --check
cargo clippy --locked --all-targets --all-features -- -D warnings
cargo test --locked --all-targets --all-features
cargo clippy --locked --all-targets --no-default-features -- -D warnings
cargo test --locked --all-targets --no-default-features
cargo run --example weather_agent
```

All gates must be green before push. `Cargo.lock` is committed — every
command below runs `--locked`. The `autumn` feature (default) gates every
module that needs `autumn-web`. Keep the rest free of it.

## Architecture

| Module | Owns |
|--------|------|
| `config` | `AgentConfig`, `ProviderKind`; `[agent]` TOML + `AGENT_*` env layering; `AGENT_API_KEY` env-only, rejected from files |
| `error` | `AgentError` (thiserror struct) + `ErrorKind`; `status_code()`; `into_autumn_error()` method (a `From` impl would clash with Autumn's blanket impl) |
| `client` | `LlmClient` trait (boxed futures, object-safe); `OpenAiCompatibleClient`; `AnthropicClient`; `client_from_config` |
| `tools` | `Tool` trait; `FnTool` closure adapter (the Autumn-handler bridge) |
| `agent` | `Agent` loop, `AgentOutcome`, budgets, `truncate_history`, approvals (`RunState`, `resume`), `AgentRuntime` |
| `hooks` | `AgentHooks` lifecycle hooks |
| `policy` | `ToolPolicy`, `ToolRules`, `Strictest` |
| `loop_guard` | `LoopGuard` repeated-call detection |
| `session` | `SessionStore`, `InMemorySessionStore`, `Compaction` |
| `memory` | `MemoryStore`, `MemoryBlock`, `apply_op`, `MemoryTool` (frozen snapshot) |
| `skills` | `Skill::parse`, `SkillTool` (`load_skill`) |
| `delegate` | `AgentTool` subagents |
| `proactive` | `Heartbeat` (hand-built Autumn `TaskInfo`), `Delivery`, `FollowupTool` |
| `jobs` | `#[job] agent_run` / `agent_resume`, `AgentRunArgs`, tracked enqueue |
| `ids` | `RunId`, `SessionId` |
| `plugin` | `AgentPlugin` (`Plugin` impl), `apply_overrides`, `AgentHandle` extractor |
| `health` | `AgentHealthIndicator` (`list_models` ping, health-only group) |

Tests live in `src/<module>/tests.rs`. Mock providers use axum in
`src/client/tests.rs`; the loop tests use a scripted `LlmClient` in
`src/agent/tests.rs`; other modules share `test_support::Script`. Config
tests never touch the process env: they pass a lookup closure to
`apply_env_with`. Background-run logic (`jobs::execute_run`,
`Heartbeat::tick`) is testable with `AppState::detached()` and no job
runtime. Research and design records: `docs/research/`, `docs/adr/`.

## Rules

- No `unwrap` / `expect` / `panic` / `todo` / `unimplemented` in production
  code (deny lints). Tests may use them via the module-level `allow`.
- Docs and comments in ASD-STE100 style: short sentences, active voice,
  simple present tense. `missing_docs` warns — document every public item.
- API keys never touch logs, `Debug` output, files, or chat. Both clients
  redact the key in `Debug`.
- Never invent Autumn APIs from memory. Ground via
  `~/workspace/skills/autumn-mcp/bin/mcp.py` against
  `https://autumn-web.app/mcp` (see `docs/planning.md` for what was grounded).
- `Plugin::build` must stay side-effect free: all IO (config load, client
  build) happens in the `on_startup` hook, which fails the boot fast.
- Budget exhaustion, deadlines, loops, and approval pauses are normal
  `AgentOutcome`s, never an `Err`.
- Unattended runs (heartbeats) stay read-only unless the app opts in.
- Memory renders once per run (frozen snapshot); never re-render the system
  prompt mid-run — it breaks prompt caching.
