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
cargo run --example weather_agent
```

All three gates must be green before push. `Cargo.lock` is committed — every
command below runs `--locked`.

## Architecture

| Module | Owns |
|--------|------|
| `config` | `AgentConfig`, `ProviderKind`; `[agent]` TOML + `AGENT_*` env layering; `AGENT_API_KEY` env-only, rejected from files |
| `error` | `AgentError` (thiserror struct) + `ErrorKind`; `status_code()`; `into_autumn_error()` method (a `From` impl would clash with Autumn's blanket impl) |
| `client` | `LlmClient` trait (boxed futures, object-safe); `OpenAiCompatibleClient`; `AnthropicClient`; `client_from_config` |
| `tools` | `Tool` trait; `FnTool` closure adapter (the Autumn-handler bridge) |
| `agent` | `Agent` loop, `AgentOutcome`, budgets, `truncate_history`, `AgentRuntime`, `#[job] agent_run` |
| `plugin` | `AgentPlugin` (`Plugin` impl), `apply_overrides`, `AgentHandle` extractor |
| `health` | `AgentHealthIndicator` (`list_models` ping, health-only group) |

Tests live in `src/<module>/tests.rs`. Mock providers use axum in
`src/client/tests.rs`; the loop tests use a scripted `LlmClient` in
`src/agent/tests.rs`. Env-touching tests hold `test_support::ENV_LOCK`.

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
- Budget exhaustion is a normal `AgentOutcome`, never an `Err`.
