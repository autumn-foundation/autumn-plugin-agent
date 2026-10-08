# autumn-plugin-agent

Provider-agnostic LLM agent harness with tool calling for [Autumn](https://autumn-web.app) apps.

Requires `autumn-web` 0.8.

One trait, two providers, zero heavyweight LLM SDKs. The plugin speaks HTTP
directly:

- **OpenAI-compatible** chat-completions (`POST /chat/completions`) — works
  with OpenAI, Ollama, vLLM, LM Studio, and any proxy that exposes the
  standard OpenAI protocol.
- **Anthropic** Messages API (`POST /v1/messages`).

## Quickstart

Add the plugin and a tool:

```rust,ignore
use autumn_plugin_agent::plugin::{AgentHandle, AgentPlugin};
use autumn_plugin_agent::{AgentError, FnTool};
use autumn_web::prelude::*;
use std::sync::Arc;

async fn get_weather(input: serde_json::Value) -> Result<serde_json::Value, AgentError> {
    let city = input["city"].as_str().unwrap_or("unknown");
    Ok(serde_json::json!({"city": city, "temp_f": 72}))
}

#[get("/ask")]
async fn ask(agent: AgentHandle) -> String {
    agent
        .run("What is the weather in Chicago?")
        .await
        .map(|o| format!("{o:?}"))
        .unwrap_or_else(|e| e.to_string())
}

#[autumn_web::main]
async fn main() {
    let weather = FnTool::new(
        "get_weather",
        "Current weather for a city.",
        serde_json::json!({
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
        }),
        get_weather,
    )
    .shared();

    autumn_web::app()
        .plugin(AgentPlugin::new().tool(weather))
        .routes(routes![ask])
        .run()
        .await;
}
```

Configure in `autumn.toml`:

```toml
[agent]
provider = "openai-compatible" # or "anthropic"
model = "gpt-4o-mini"
# base_url = "http://localhost:11434/v1"  # Ollama, vLLM, ...
max_steps = 10
max_tokens = 32000
# max_run_secs = 300        # wall-clock limit per run
# prompt_caching = true     # Anthropic cache breakpoints (default on)
```

The API key comes **only** from the environment — never from the config file:

```sh
export AGENT_API_KEY="sk-..."
```

## Always-on agents

The crate has the primitives that today's always-on agents (Hermes,
OpenClaw, Dots, Muse) share. See
[`docs/research/always-on-agents.md`](docs/research/always-on-agents.md).

```rust,ignore
use std::sync::Arc;
use std::time::Duration;
use autumn_plugin_agent::{AgentPlugin, Heartbeat, InMemoryMemoryStore, ToolRules};

AgentPlugin::new()
    .tool(send_email)                                  // declares ToolEffect::External
    .memory_store(InMemoryMemoryStore::default().shared())
    .policy(Arc::new(ToolRules::ask_before_acting()))  // pause before writes
    .heartbeat(Heartbeat::every(Duration::from_secs(1_800)).session("ops"))
    .followups(Duration::from_secs(24 * 3_600))        // agent books its own wake-ups
    .delivery(my_slack_delivery);                      // where reports go
```

| Feature | API |
|---------|-----|
| Multi-turn and sessions | `Agent::run_turn`, `Agent::run_in_session`, `AgentHandle::chat`, `SessionStore` |
| Compaction | `Compaction`: summarizes old turns once a session passes a token threshold |
| Memory | `MemoryStore` + the built-in `memory` tool; bounded blocks, frozen snapshot per run |
| Skills | `Skill::parse` (`SKILL.md` front matter) + the `load_skill` tool |
| Approvals | `ToolPolicy` / `ToolRules` → `AgentOutcome::AwaitingApproval { state }` → `Agent::resume` or `enqueue_agent_resume` |
| Hooks | `AgentHooks`: `before_model`, `before_tool` (`Block`/`Modify`), `after_tool`, `on_outcome` |
| Loop guard | `LoopGuard` → `AgentOutcome::LoopDetected` |
| Deadline | `Agent::max_duration` / `max_run_secs` → `BudgetKind::Deadline` |
| Heartbeat | `Heartbeat::every` / `Heartbeat::cron`; `HEARTBEAT_OK` stays silent; read-only by default; `precheck` |
| Follow-ups | the `schedule_followup` tool (delayed `agent_run`), capped chain depth |
| Subagents | `AgentTool` wraps a child `Agent` as a tool |
| Tracked runs | `enqueue_agent_run_tracked` → poll Autumn's job-status route; the result is the outcome JSON |
| Prompt caching | Anthropic `cache_control` breakpoints; `TokenUsage::cache_read_tokens` |

Run `cargo run --example always_on_agent` to see memory, an approval round
trip, and two heartbeat ticks against a scripted model.

## What the plugin installs

- `[agent]` config section (declared strict-config-safe).
- `agent_run` and `agent_resume` background jobs — enqueue with
  `enqueue_agent_run(AgentRunArgs::new(prompt))`,
  `enqueue_agent_run_tracked`, or `enqueue_agent_resume` (re-exported at the
  crate root).
- `agent_heartbeat` scheduled task, when a `Heartbeat` is set.
- `agent` health indicator on `/actuator/health` — a cheap `list_models`
  ping against the same provider client the running app uses (shared with
  the startup hook; DOWN until startup completes). It sits in the
  health-only group, so a sick provider never blocks deploys.
- `AgentHandle` extractor for handlers — a ready-to-run agent from app state.

## The agent loop

`Agent::run` iterates: call model → execute tool calls → append results →
repeat, until the model answers, a budget runs out, the loop guard fires,
or the policy pauses for approval. `max_steps` counts
tool-execution rounds (the final answer is free) and `max_tokens` covers
estimated plus provider-reported usage; the per-call output cap reserves the
estimated history cost first, and a final answer that overshoots the token
budget still returns `AgentOutcome::BudgetExhausted`, not an error. Unknown
tools, tool failures, denials, and rejections reach the model as error
payloads so it can recover. Calls in one round run concurrently
(`parallel_tool_calls(false)` runs them one by one). When the history must
shrink, the loop keeps the system prompt and the task message and never
leaves a tool result without its call.

## Background runs

```rust,ignore
use autumn_plugin_agent::{AgentRunArgs, SessionId, enqueue_agent_run_tracked};

let handle = enqueue_agent_run_tracked(
    AgentRunArgs::new("Summarize today's signups.")
        .session(SessionId::new("daily-report"))
        .deliver(true),
)
.await?;
// Poll handle.status_path(); the result is the AgentOutcome as JSON.
```

The run id is fixed at enqueue time, so a retried job keeps it. Tools read
it from `ToolContext` to build idempotency keys.

## Crate layout

| Module | Contents |
|--------|----------|
| `config` | Layered `[agent]` config; secrets stay in the env |
| `error` | `AgentError` + `ErrorKind` + HTTP status mapping |
| `client` | `LlmClient` trait, both provider implementations, prompt caching |
| `tools` | `Tool` trait, `ToolContext`, `ToolEffect`, `FnTool` adapter |
| `agent` | Agent loop, budgets, approvals (`RunState`), `AgentRuntime` |
| `hooks` | `AgentHooks` lifecycle hooks |
| `policy` | `ToolPolicy`, `ToolRules`, `Strictest` |
| `loop_guard` | Repeated-call detection |
| `session` | `SessionStore`, compaction |
| `memory` | `MemoryStore`, bounded blocks, the `memory` tool |
| `skills` | `Skill` (`SKILL.md`), the `load_skill` tool |
| `delegate` | `AgentTool` subagents |
| `proactive` | `Delivery`; `Heartbeat` and `schedule_followup` need `autumn` |
| `jobs` | `agent_run` / `agent_resume` jobs, tracked enqueue (`autumn`) |
| `ids` | `RunId`, `SessionId` |
| `plugin` | `AgentPlugin`, `AgentHandle` extractor (`autumn`) |
| `health` | Provider health indicator (`autumn`) |

## Cargo features

| Feature | Default | Contents |
|---------|---------|----------|
| `autumn` | on | `plugin`, `jobs`, `health`, `Heartbeat`, follow-ups, `into_autumn_error`. Needs `autumn-web`. |
| `native-tls` | off | Platform certificate roots for the provider HTTP client. |

Turn off default features to use the agent core without `autumn-web`:

```toml
autumn-plugin-agent = { version = "0.3", default-features = false }
```

## Known issues

- The default session store and the in-memory memory store lose data on
  restart. Supply database-backed `SessionStore` / `MemoryStore`
  implementations in production.
- A retried `agent_run` job re-runs the whole loop. Step checkpoints are a
  follow-up; tools can deduplicate on `ToolContext::run_id` meanwhile.
- Non-streaming only. Streaming chat completions is an explicit follow-up
  (see CHANGELOG).
- Token accounting mixes provider-reported usage with a `bytes / 4`
  heuristic between calls. Good enough for a guardrail, never for billing.
- The plugin reads `[agent]` from `autumn.toml` itself with the `toml`
  crate; Autumn 0.8 still exposes no per-section accessor to plugins, so the
  framework's own profile layering does not apply to plugin keys yet.

## License

Apache-2.0.
