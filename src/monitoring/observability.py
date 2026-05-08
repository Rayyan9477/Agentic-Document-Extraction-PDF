"""
WS-7: AI observability dispatcher (Phoenix + PostHog).

This module provides a single chokepoint for emitting LLM / pipeline
traces and product-analytics events to multiple sinks. Existing
``audit.py``, ``metrics.py``, and structlog stay untouched — they
already handle compliance / Prometheus / structured logs respectively.
This dispatcher fans out the same events to optional, opt-in sinks:

    * **Arize Phoenix** (``arize-phoenix``,
      ``openinference-instrumentation-langchain``,
      ``openinference-instrumentation-openai``) — local LLM tracing
      with token / latency / cost capture, runs on
      ``http://localhost:6006`` by default. Self-hosted, no data
      leaves the host.

    * **PostHog** (``posthog`` Python SDK) — product analytics for
      extraction events. Can target either PostHog Cloud or a
      self-hosted PostHog instance via ``POSTHOG_HOST``.

Both are off by default. Enable per-sink via:

    settings.observability.phoenix_enabled = True
    settings.observability.posthog_enabled  = True
    settings.observability.posthog_api_key  = "phc_..."
    settings.observability.posthog_host     = "https://posthog.example.com"

The dispatcher is a no-op when both flags are off, so plain
``ObservabilityDispatcher.from_settings().emit_event(...)`` calls are
safe in every code path.

Usage::

    obs = ObservabilityDispatcher.from_settings()
    obs.emit_event("extraction_started", {"document_type": "cms1500"})
    with obs.start_span("vlm.request", agent="extractor"):
        response = client.send_vision_request(req)
    obs.record_llm_call(
        model="qwen3-vl",
        latency_ms=843,
        prompt_tokens=412,
        completion_tokens=128,
    )

This module **never** raises into the application code. Sink failures
log a warning and the dispatcher continues.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any

from src.config import get_logger


logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Sink protocol
# ---------------------------------------------------------------------------


class _Sink:
    """Minimal sink interface — implementations override what they support."""

    name: str = "noop"

    def emit_event(self, event_name: str, properties: dict[str, Any]) -> None:
        return None

    def record_llm_call(self, **attrs: Any) -> None:
        return None

    def start_span(self, name: str, **attrs: Any) -> Any:
        # Returns a context manager. Default: a no-op contextmanager that
        # accepts the ``with`` protocol without doing anything.
        @contextmanager
        def _noop() -> Any:
            yield None

        return _noop()

    def shutdown(self) -> None:
        return None


# ---------------------------------------------------------------------------
# Phoenix sink
# ---------------------------------------------------------------------------


class PhoenixSink(_Sink):
    """Arize Phoenix (OpenInference + OpenTelemetry) tracing sink.

    Lazy-initialises the OTLP exporter and OpenInference instrumentors
    on construction so importing this module never costs anything at
    boot. When ``arize-phoenix`` or its OpenInference companions are
    not installed, ``PhoenixSink.try_create`` returns ``None`` and the
    dispatcher silently omits this sink.
    """

    name = "phoenix"

    def __init__(self, tracer: Any) -> None:
        self._tracer = tracer

    @classmethod
    def try_create(
        cls,
        *,
        endpoint: str = "http://localhost:6006",
        project_name: str = "doc-extraction",
    ) -> PhoenixSink | None:
        try:
            from phoenix.otel import register  # type: ignore[import-not-found]
        except ImportError:
            logger.info(
                "observability_phoenix_not_installed",
                hint="install with `pip install -e .[observability]`",
            )
            return None

        try:
            tracer_provider = register(
                project_name=project_name,
                endpoint=f"{endpoint.rstrip('/')}/v1/traces",
                set_global_tracer_provider=True,
            )
        except Exception as exc:  # pragma: no cover - integration path
            logger.warning(
                "observability_phoenix_register_failed",
                error=str(exc),
                endpoint=endpoint,
            )
            return None

        # Best-effort instrumentation of the LLM client + LangGraph
        # state machine. Each instrumentor is independent — failure of
        # one doesn't prevent the others from registering.
        for module_path, attr in (
            ("openinference.instrumentation.openai", "OpenAIInstrumentor"),
            ("openinference.instrumentation.langchain", "LangChainInstrumentor"),
        ):
            try:
                module = __import__(module_path, fromlist=[attr])
                getattr(module, attr)().instrument()
            except ImportError:
                logger.info(
                    "observability_phoenix_instrumentor_missing",
                    instrumentor=attr,
                )
            except Exception as exc:  # pragma: no cover - integration path
                logger.warning(
                    "observability_phoenix_instrumentor_failed",
                    instrumentor=attr,
                    error=str(exc),
                )

        tracer = tracer_provider.get_tracer(__name__) if tracer_provider else None
        logger.info(
            "observability_phoenix_ready",
            endpoint=endpoint,
            project_name=project_name,
        )
        return cls(tracer=tracer)

    @contextmanager
    def start_span(self, name: str, **attrs: Any) -> Any:  # type: ignore[override]
        if self._tracer is None:
            yield None
            return
        with self._tracer.start_as_current_span(name) as span:
            try:
                for k, v in attrs.items():
                    if v is not None:
                        try:
                            span.set_attribute(str(k), v)
                        except Exception:  # pragma: no cover - exporter quirks
                            pass
                yield span
            except Exception as exc:
                # Mark the span as errored so Phoenix shows it red, then
                # re-raise — observability never swallows app errors.
                try:
                    span.record_exception(exc)
                except Exception:  # pragma: no cover
                    pass
                raise

    def record_llm_call(self, **attrs: Any) -> None:  # type: ignore[override]
        # Phoenix collects LLM call attributes via the OpenInference
        # auto-instrumentor (registered above), so we don't need to
        # re-record them here. Method left available for symmetry with
        # the dispatcher contract.
        return None


# ---------------------------------------------------------------------------
# PostHog sink
# ---------------------------------------------------------------------------


class PostHogSink(_Sink):
    """PostHog product-analytics sink for high-level pipeline events.

    Captures things like ``extraction_started`` / ``extraction_completed``
    / ``human_review_triggered`` / ``vlm_call`` so operators can
    answer questions like "what's the success rate by document type?"
    or "how often does the splitter fire?" from the PostHog dashboard.
    """

    name = "posthog"

    def __init__(self, client: Any, default_distinct_id: str = "doc-extraction-system") -> None:
        self._client = client
        self._distinct_id = default_distinct_id

    @classmethod
    def try_create(
        cls,
        *,
        api_key: str,
        host: str = "https://us.posthog.com",
    ) -> PostHogSink | None:
        if not api_key:
            return None
        try:
            from posthog import Posthog  # type: ignore[import-not-found]
        except ImportError:
            logger.info(
                "observability_posthog_not_installed",
                hint="install with `pip install -e .[observability]`",
            )
            return None

        try:
            client = Posthog(project_api_key=api_key, host=host)
        except Exception as exc:  # pragma: no cover - integration path
            logger.warning("observability_posthog_init_failed", error=str(exc))
            return None

        logger.info("observability_posthog_ready", host=host)
        return cls(client=client)

    def emit_event(self, event_name: str, properties: dict[str, Any]) -> None:
        try:
            self._client.capture(
                distinct_id=str(properties.get("user_id") or self._distinct_id),
                event=event_name,
                properties=properties,
            )
        except Exception as exc:  # pragma: no cover - network path
            logger.warning("observability_posthog_emit_failed", error=str(exc))

    def shutdown(self) -> None:
        try:
            self._client.shutdown()
        except Exception:  # pragma: no cover
            pass


# ---------------------------------------------------------------------------
# Dispatcher
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ObservabilityDispatcher:
    """Fans out events / spans / LLM-call records to all configured sinks."""

    sinks: list[_Sink] = field(default_factory=list)

    @classmethod
    def from_settings(cls) -> ObservabilityDispatcher:
        """Construct a dispatcher from ``settings.observability``.

        Sinks are added only when their respective enable flag is set
        AND the underlying SDK is importable AND construction succeeds.
        Otherwise the dispatcher is empty (no-op).
        """
        sinks: list[_Sink] = []
        try:
            from src.config import get_settings

            obs = getattr(get_settings(), "observability", None)
        except Exception:
            obs = None

        if obs is not None:
            if getattr(obs, "phoenix_enabled", False):
                phoenix = PhoenixSink.try_create(
                    endpoint=getattr(obs, "phoenix_endpoint", "http://localhost:6006"),
                    project_name=getattr(obs, "phoenix_project_name", "doc-extraction"),
                )
                if phoenix is not None:
                    sinks.append(phoenix)

            if getattr(obs, "posthog_enabled", False):
                posthog = PostHogSink.try_create(
                    api_key=getattr(obs, "posthog_api_key", ""),
                    host=getattr(obs, "posthog_host", "https://us.posthog.com"),
                )
                if posthog is not None:
                    sinks.append(posthog)

        if not sinks:
            logger.info("observability_dispatcher_noop")

        return cls(sinks=sinks)

    @property
    def is_active(self) -> bool:
        return bool(self.sinks)

    def emit_event(self, event_name: str, properties: dict[str, Any] | None = None) -> None:
        properties = properties or {}
        for sink in self.sinks:
            try:
                sink.emit_event(event_name, properties)
            except Exception as exc:  # pragma: no cover - sink path
                # NOTE: ``event`` is reserved by structlog for the log
                # message itself; use ``event_name`` here to avoid a
                # "multiple values for argument 'event'" TypeError.
                logger.warning(
                    "observability_sink_event_failed",
                    sink=sink.name,
                    event_name=event_name,
                    error=str(exc),
                )

    @contextmanager
    def start_span(self, name: str, **attrs: Any) -> Any:
        """Open a span across every sink that supports tracing.

        Sinks that don't implement spans (e.g. PostHog) yield None;
        ones that do (Phoenix) yield their native span object via the
        first sink to provide one. The contract is: callers ``with
        dispatcher.start_span(...)`` and ignore the yielded value
        unless they need to add custom attributes.
        """
        if not self.sinks:
            yield None
            return

        # Open spans on all sinks. We track each sink's context manager
        # so we can ``__exit__`` them in reverse order regardless of
        # which sink yields what.
        cms = [sink.start_span(name, **attrs) for sink in self.sinks]
        spans = []
        try:
            for cm in cms:
                spans.append(cm.__enter__())
            # Surface the first non-None span (typically Phoenix) so the
            # caller can add custom attributes. Most callers ignore.
            yield next((s for s in spans if s is not None), None)
        finally:
            exc_info = None
            for cm in reversed(cms):
                try:
                    cm.__exit__(None, None, None)
                except Exception as exc:  # pragma: no cover
                    exc_info = exc
            if exc_info is not None:
                logger.warning(
                    "observability_span_close_failed",
                    error=str(exc_info),
                )

    def record_llm_call(self, **attrs: Any) -> None:
        for sink in self.sinks:
            try:
                sink.record_llm_call(**attrs)
            except Exception as exc:  # pragma: no cover
                logger.warning(
                    "observability_sink_llm_failed",
                    sink=sink.name,
                    error=str(exc),
                )

    def shutdown(self) -> None:
        for sink in self.sinks:
            try:
                sink.shutdown()
            except Exception:  # pragma: no cover
                pass


# ---------------------------------------------------------------------------
# Module singleton
# ---------------------------------------------------------------------------


_dispatcher: ObservabilityDispatcher | None = None


def get_dispatcher() -> ObservabilityDispatcher:
    """Lazy-construct the process-wide observability dispatcher."""
    global _dispatcher
    if _dispatcher is None:
        _dispatcher = ObservabilityDispatcher.from_settings()
    return _dispatcher


def set_dispatcher(dispatcher: ObservabilityDispatcher) -> None:
    """Override the singleton (used by tests and bootstrap code)."""
    global _dispatcher
    _dispatcher = dispatcher


__all__ = [
    "ObservabilityDispatcher",
    "PhoenixSink",
    "PostHogSink",
    "get_dispatcher",
    "set_dispatcher",
]
