# AI Observability (WS-7)

> Opt-in LLM tracing + product analytics behind one dispatcher. Both
> sinks are off by default; enable via `settings.observability`.

## Two sinks behind one dispatcher

```
                         ┌─ PhoenixSink   (LLM traces)
ObservabilityDispatcher ─┤
                         └─ PostHogSink   (product analytics)
```

The dispatcher fans out:

* **`emit_event(name, properties)`** — high-level pipeline events
  (`extraction_started`, `extraction_completed`,
  `human_review_triggered`, …)
* **`start_span(name, **attrs)`** — context-managed traces around
  VLM calls and pipeline nodes
* **`record_llm_call(**attrs)`** — token counts, latency, model name
  per call

Sink failures are isolated: a Phoenix outage doesn't affect PostHog
event capture, and neither blocks the extraction pipeline.

## Phoenix (LLM tracing)

[Arize Phoenix](https://github.com/Arize-ai/phoenix) is OpenInference
+ OpenTelemetry. Self-hosted, runs on `http://localhost:6006` by
default. No data leaves the host.

Auto-instruments two layers via `openinference-instrumentation-*`:

* **`LangChainInstrumentor`** — every node in the LangGraph state
  machine emits a span (preprocess / analyze / extract / validate /
  route / etc.)
* **`OpenAIInstrumentor`** — every VLM call from `LMStudioClient`
  (which uses the OpenAI SDK) gets full request / response /
  token-count attribution.

### Enable

```bash
pip install -e ".[dev,observability]"
```

```python
# src/config/settings.py → ObservabilitySettings
phoenix_enabled: bool = True               # OBSERVABILITY_PHOENIX_ENABLED
phoenix_endpoint: str = "http://localhost:6006"
phoenix_project_name: str = "doc-extraction"
```

### Run the local Phoenix UI

```bash
python -c "import phoenix; phoenix.launch_app()"
# Phoenix UI at http://localhost:6006
```

Process a document with the API or CLI; spans appear live in the UI,
including a tree view of the LangGraph traversal and per-VLM-call
token / latency / cost.

## PostHog (product analytics)

[PostHog](https://posthog.com/) gives you dashboards, funnels, cohort
analyses on the system itself — "what's the success rate by document
type?", "how often does the splitter fire?", "what's the human-review
rate this week?". Self-hostable; the SDK supports either PostHog
Cloud or a private instance via `posthog_host`.

### Enable

```python
# src/config/settings.py → ObservabilitySettings
posthog_enabled: bool = True               # OBSERVABILITY_POSTHOG_ENABLED
posthog_api_key: str  = "phc_..."          # OBSERVABILITY_POSTHOG_API_KEY
posthog_host: str = "https://us.posthog.com"  # OBSERVABILITY_POSTHOG_HOST
                                              # — point at your self-hosted host
```

### Events captured

The dispatcher emits these events when active. Add to the list as
new pipeline phases come online:

| Event | When | Properties |
|---|---|---|
| `extraction_started` | top of `PipelineRunner.extract_*` | `document_type`, `processing_id`, `page_count` |
| `extraction_completed` | terminal `complete` node | `confidence`, `retry_count`, `vlm_calls`, `processing_ms` |
| `human_review_triggered` | enter `_human_review_node` | `processing_id`, `reason`, `confidence` |
| `vlm_call` | every `BaseAgent.send_vision_request` | `agent`, `model`, `latency_ms`, `request_id` |

## Where it's wired

| Layer | Anchor |
|---|---|
| Dispatcher class | [src/monitoring/observability.py](../src/monitoring/observability.py) |
| Settings | `ObservabilitySettings` in [src/config/settings.py](../src/config/settings.py) |
| VLM-call span wrap | `BaseAgent.send_vision_request` in [src/agents/base.py](../src/agents/base.py) |
| Sink fan-out semantics | `ObservabilityDispatcher.start_span` opens N spans, closes in reverse |
| Failure isolation | `try/except` per-sink in `emit_event` / `record_llm_call` |

## What about the existing `audit.py` / `metrics.py` / structlog?

Untouched. They handle:

* **`audit.py`** — HIPAA-compliant tamper-evident audit log (separate
  from observability; covers regulatory needs)
* **`metrics.py`** — Prometheus exposition for ops dashboards
  (`/metrics` endpoint)
* **`structlog`** — structured app logs with PHI masking

Phoenix + PostHog complement those rather than replace them. Pick
the layers your deployment needs:

| You need | Enable |
|---|---|
| HIPAA audit | `audit.py` (always-on) |
| Ops dashboards | Prometheus scrape on `/metrics` |
| App logs | structlog (always-on) |
| LLM tracing / debugging | Phoenix |
| Product analytics / cost tracking | PostHog |

## Testing the dispatcher off-line

`tests/unit/test_observability.py` covers the dispatcher's contract
(fan-out, isolation, no-op safety) with `_FakeSink` — no Phoenix /
PostHog SDKs are required to validate the wiring. The actual sink
classes degrade to `try_create() → None` when their SDKs aren't
installed, so default-install / air-gapped builds incur no overhead.
