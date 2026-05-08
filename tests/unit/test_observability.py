"""WS-7: tests for the AI observability dispatcher.

Phoenix and PostHog are optional sinks behind the ``[observability]``
extra. These tests focus on the dispatcher's contract — fan-out,
no-op safety, sink-failure isolation — rather than the SDKs themselves.
"""

from __future__ import annotations

from contextlib import contextmanager
from unittest.mock import MagicMock, patch

from src.monitoring.observability import (
    ObservabilityDispatcher,
    PhoenixSink,
    PostHogSink,
    _Sink,
    get_dispatcher,
    set_dispatcher,
)


class _FakeSink(_Sink):
    """In-memory sink for verifying dispatcher calls."""

    def __init__(self, name: str = "fake", *, raise_on_event: bool = False) -> None:
        self.name = name
        self.events: list[tuple[str, dict]] = []
        self.spans_opened: list[str] = []
        self.spans_closed: list[str] = []
        self.llm_calls: list[dict] = []
        self.shutdowns: int = 0
        self._raise = raise_on_event

    def emit_event(self, event_name: str, properties: dict) -> None:
        if self._raise:
            raise RuntimeError("simulated sink failure")
        self.events.append((event_name, properties))

    def record_llm_call(self, **attrs) -> None:
        self.llm_calls.append(attrs)

    @contextmanager
    def start_span(self, name, **attrs):
        self.spans_opened.append(name)
        try:
            yield {"name": name, "attrs": attrs}
        finally:
            self.spans_closed.append(name)

    def shutdown(self) -> None:
        self.shutdowns += 1


# ---------------------------------------------------------------------------
# Dispatcher contract
# ---------------------------------------------------------------------------


class TestDispatcherFanout:
    def test_no_sinks_is_active_false_and_noop(self) -> None:
        d = ObservabilityDispatcher(sinks=[])
        assert d.is_active is False
        # No-op calls don't raise and don't emit anywhere.
        d.emit_event("x", {})
        d.record_llm_call(model="m")
        with d.start_span("noop_span"):
            pass

    def test_emit_event_fans_out_to_every_sink(self) -> None:
        a, b = _FakeSink("a"), _FakeSink("b")
        d = ObservabilityDispatcher(sinks=[a, b])
        d.emit_event("doc.processed", {"type": "cms1500"})
        assert a.events == [("doc.processed", {"type": "cms1500"})]
        assert b.events == [("doc.processed", {"type": "cms1500"})]

    def test_record_llm_call_fans_out(self) -> None:
        a, b = _FakeSink("a"), _FakeSink("b")
        d = ObservabilityDispatcher(sinks=[a, b])
        d.record_llm_call(model="qwen3-vl", latency_ms=843)
        assert a.llm_calls == [{"model": "qwen3-vl", "latency_ms": 843}]
        assert b.llm_calls == [{"model": "qwen3-vl", "latency_ms": 843}]

    def test_start_span_opens_on_all_sinks_and_closes_in_reverse(self) -> None:
        a, b, c = _FakeSink("a"), _FakeSink("b"), _FakeSink("c")
        d = ObservabilityDispatcher(sinks=[a, b, c])
        with d.start_span("vlm.call"):
            pass
        assert a.spans_opened == ["vlm.call"]
        assert b.spans_opened == ["vlm.call"]
        assert c.spans_opened == ["vlm.call"]
        # Reverse order on close.
        assert [a.spans_closed, b.spans_closed, c.spans_closed] == [
            ["vlm.call"],
            ["vlm.call"],
            ["vlm.call"],
        ]

    def test_start_span_yields_first_non_none_span(self) -> None:
        # First sink's start_span yields a structured object.
        a = _FakeSink("a")
        d = ObservabilityDispatcher(sinks=[a])
        with d.start_span("vlm.call", agent="extractor") as span:
            assert span is not None
            assert span["name"] == "vlm.call"
            assert span["attrs"]["agent"] == "extractor"


class TestDispatcherIsolation:
    def test_sink_emit_failure_does_not_block_other_sinks(self) -> None:
        bad = _FakeSink("bad", raise_on_event=True)
        good = _FakeSink("good")
        d = ObservabilityDispatcher(sinks=[bad, good])
        # Bad sink raises internally; dispatcher logs + continues.
        d.emit_event("x", {"k": "v"})
        # Good sink received the event.
        assert good.events == [("x", {"k": "v"})]

    def test_shutdown_calls_each_sink_once(self) -> None:
        a, b = _FakeSink("a"), _FakeSink("b")
        d = ObservabilityDispatcher(sinks=[a, b])
        d.shutdown()
        assert a.shutdowns == 1
        assert b.shutdowns == 1


# ---------------------------------------------------------------------------
# from_settings construction
# ---------------------------------------------------------------------------


class TestFromSettings:
    def test_disabled_settings_yield_empty_dispatcher(self) -> None:
        # Default settings have both sinks off → no sinks attached.
        d = ObservabilityDispatcher.from_settings()
        assert d.is_active is False

    def test_phoenix_enabled_but_sdk_missing_yields_no_phoenix_sink(self) -> None:

        fake_settings = MagicMock()
        fake_settings.observability.phoenix_enabled = True
        fake_settings.observability.phoenix_endpoint = "http://localhost:6006"
        fake_settings.observability.phoenix_project_name = "test"
        fake_settings.observability.posthog_enabled = False

        # ``get_settings`` is imported lazily inside ``from_settings``,
        # so we patch it at its source module.
        with (
            patch("src.config.get_settings", return_value=fake_settings),
            patch.object(PhoenixSink, "try_create", return_value=None),
        ):
            d = ObservabilityDispatcher.from_settings()
        assert d.is_active is False

    def test_posthog_enabled_with_valid_key_attaches_sink(self) -> None:

        fake_settings = MagicMock()
        fake_settings.observability.phoenix_enabled = False
        fake_settings.observability.posthog_enabled = True
        fake_settings.observability.posthog_api_key = "phc_test"
        fake_settings.observability.posthog_host = "https://test.posthog.com"

        fake_sink = _FakeSink("posthog")
        with (
            patch("src.config.get_settings", return_value=fake_settings),
            patch.object(PostHogSink, "try_create", return_value=fake_sink),
        ):
            d = ObservabilityDispatcher.from_settings()
        assert d.is_active is True
        assert d.sinks[0] is fake_sink


class TestSingleton:
    def test_set_dispatcher_overrides_singleton(self) -> None:
        custom = ObservabilityDispatcher(sinks=[_FakeSink("custom")])
        set_dispatcher(custom)
        try:
            assert get_dispatcher() is custom
        finally:
            # Reset for other tests in the suite.
            set_dispatcher(ObservabilityDispatcher(sinks=[]))


class TestPostHogSinkTryCreate:
    def test_empty_api_key_short_circuits_to_none(self) -> None:
        sink = PostHogSink.try_create(api_key="")
        assert sink is None
