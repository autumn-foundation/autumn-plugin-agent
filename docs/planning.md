# Planning notes — autumn-plugin-agent

## Autumn API grounding (2026-09-29, via the docs MCP + autumn-web 0.7.0 source)

Every framework touchpoint below was verified against the MCP docs or the
cached `autumn-web-0.7.0` source — never from training memory.

| Need | Finding |
|------|---------|
| Plugin shape | `autumn_web::plugin::Plugin`: `fn build(self, app: AppBuilder) -> AppBuilder` (`#[must_use]`); optional `name()` (defaults to `type_name`). Third-party crates are named `autumn-plugin-<name>` and expose `<Name>Plugin` with `::new()` + `#[must_use]` fluent methods. (source: `src/plugin.rs`) |
| Install | Users call `.plugin(AgentPlugin::new(...))`; duplicate names warn and no-op. (docs: `extensibility`, tier 3) |
| Config section | `AppBuilder::config_section("agent")` declares `[agent]` strict-config-safe. The media plugin fail-fast-validates its own section in its startup hook — the established pattern this plugin follows. (source: `src/app.rs`) |
| Plugin config access | Autumn 0.7 exposes **no** per-section accessor to plugins (`AutumnConfig` has no plugin-section reader). The plugin therefore reads `[agent]` from `autumn.toml` itself with the `toml` crate. Documented as a known issue in README. |
| Runtime state | `AppBuilder::on_startup(\|state: AppState\| -> AutumnResult<()>)` runs after `AppState` exists; `AppState::extension_or_insert_with` installs typed state. (source: `src/app.rs`, `src/state.rs`) |
| Extractor | `impl FromRequestParts<AppState> for X` with `type Rejection = AutumnError`; `state.extension::<T>()` reads typed state. (docs: `what-happens-when`) |
| Background job | `#[autumn_web::job(name = "...", max_attempts = N, backoff_ms = M)] async fn(state: AppState, args: Args) -> AutumnResult<()>`; registration via `app.jobs(autumn_web::jobs![fn_name])`; enqueue via generated `FnNameJob::enqueue(args)`. (docs: `jobs`; source: `autumn-macros-0.7.0/src/job.rs`) |
| Health | `autumn_web::actuator::{HealthIndicator, HealthCheckOutput, HealthStatus, IndicatorGroup}`; `check(&self) -> futures::future::BoxFuture<'_, HealthCheckOutput>`; `AppBuilder::health_indicator(name, Arc::new(...))`. `IndicatorGroup::HealthOnly` keeps a sick provider out of `/ready`. (docs: `health-indicators`; source: `src/actuator.rs`) |
| Errors | `autumn_web::{AutumnError, AutumnResult}` at crate root; `AutumnError::{internal_server_error_msg, service_unavailable_msg, ...}` + `.with_status(StatusCode)`. (source: `src/error.rs`) |
| API docs | Autumn ships `#[apidoc]` + swagger-ui built in — not reinvented here. |

## Design decisions

- **No LLM SDKs**: both providers speak HTTP via `reqwest` directly.
- **Object-safe `LlmClient`**: hand-boxed futures (`Pin<Box<dyn Future>>`)
  instead of `async_trait`, so `Arc<dyn LlmClient>` works with no extra dep.
- **Tool adapter honesty**: an Autumn handler cannot *be* a tool (handlers
  need the request lifecycle; tools run in jobs). `FnTool` wraps the
  handler's core logic fn — one canonical `async fn(Value) ->
  Result<Value, AgentError>` called from both.
- **Secrets**: `AGENT_API_KEY` env-only; file keys fail closed; both clients
  redact the key in `Debug`.
- **Budget exhaustion is data**: `AgentOutcome::BudgetExhausted`, never `Err`.
- **Token accounting**: provider-reported usage + `bytes/4` heuristic between
  calls. Guardrail-grade, not billing-grade.

## Deferred (in CHANGELOG)

Streaming (`chat_stream` + SSE), tracked job result persistence, MCP client
tools, usage metering hook.
