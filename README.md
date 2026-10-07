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
```

The API key comes **only** from the environment — never from the config file:

```sh
export AGENT_API_KEY="sk-..."
```

## What the plugin installs

- `[agent]` config section (declared strict-config-safe).
- `agent_run` background job — enqueue long runs with
  `enqueue_agent_run(AgentRunArgs { prompt, .. })` (re-exported at the crate
  root).
- `agent` health indicator on `/actuator/health` — a cheap `list_models`
  ping against the same provider client the running app uses (shared with
  the startup hook; DOWN until startup completes). It sits in the
  health-only group, so a sick provider never blocks deploys.
- `AgentHandle` extractor for handlers — a ready-to-run agent from app state.

## The agent loop

`Agent::run` iterates: call model → execute tool calls → append results →
repeat, until the model answers or a budget runs out. `max_steps` counts
tool-execution rounds (the final answer is free) and `max_tokens` covers
estimated plus provider-reported usage; the per-call output cap reserves the
estimated history cost first, and a final answer that overshoots the token
budget still returns `AgentOutcome::BudgetExhausted`, not an error. Unknown
tools and tool failures reach the model as error payloads so it can recover.

## Background runs

```rust,ignore
use autumn_plugin_agent::agent::{AgentRunArgs, enqueue_agent_run};

enqueue_agent_run(AgentRunArgs {
    prompt: "Summarize today's signups.".into(),
    system_prompt: None,
    max_steps: None,
})
.await?;
```

## Crate layout

| Module | Contents |
|--------|----------|
| `config` | Layered `[agent]` config; secrets stay in the env |
| `error` | `AgentError` + `ErrorKind` + HTTP status mapping |
| `client` | `LlmClient` trait, both provider implementations |
| `tools` | `Tool` trait, `FnTool` handler-function adapter |
| `agent` | Agent loop, budgets, `AgentRuntime`, `agent_run` job |
| `plugin` | `AgentPlugin`, `AgentHandle` extractor |
| `health` | Provider health indicator |

## Known issues

- Non-streaming only. Streaming chat completions is an explicit follow-up
  (see CHANGELOG).
- Token accounting mixes provider-reported usage with a `bytes / 4`
  heuristic between calls. Good enough for a guardrail, never for billing.
- The plugin reads `[agent]` from `autumn.toml` itself with the `toml`
  crate; Autumn 0.8 still exposes no per-section accessor to plugins, so the
  framework's own profile layering does not apply to plugin keys yet.

## License

Apache-2.0.
