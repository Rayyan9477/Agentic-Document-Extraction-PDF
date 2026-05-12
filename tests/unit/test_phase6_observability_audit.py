"""V3 Phase 6 — observability spans + audit trace_id propagation.

Coverage:

* ``build_pass_span_attrs`` produces canonical attribute dicts and
  drops ``None``s.
* Canonical event names + span names exist as module-level constants.
* ``bind_trace_id`` / ``clear_trace_id`` / ``trace_scope`` round-trip
  via structlog contextvars.
* ``AuditContext`` carries trace_id + tenant_id and serialises them.
* ``AuditLogger.log()`` auto-pulls trace_id from contextvars when
  not explicitly supplied.
* ``emit_export_event`` is a no-op when the dispatcher is inactive.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

import pytest
import structlog

from src.monitoring.observability import (
    EVENT_CRITIC_DISAGREED,
    EVENT_EXPORT_COMPLETED,
    EVENT_EXTRACTION_COMPLETED,
    EVENT_EXTRACTION_STARTED,
    EVENT_HUMAN_REVIEW_TRIGGERED,
    EVENT_VLM_CALLED,
    SPAN_CRITIC,
    SPAN_EXTRACTION_PASS,
    SPAN_RECONCILER,
    SPAN_VLM_REQUEST,
    ObservabilityDispatcher,
    build_pass_span_attrs,
    emit_export_event,
    set_dispatcher,
)
from src.security.audit import (
    AuditContext,
    AuditEventType,
    AuditLogger,
    AuditOutcome,
    AuditSeverity,
    bind_trace_id,
    clear_trace_id,
    get_current_trace_id,
    trace_scope,
)


# ---------------------------------------------------------------------------
# Canonical names
# ---------------------------------------------------------------------------


class TestCanonicalNames:
    def test_event_names_are_stable(self) -> None:
        assert EVENT_EXTRACTION_STARTED == "extraction_started"
        assert EVENT_EXTRACTION_COMPLETED == "extraction_completed"
        assert EVENT_VLM_CALLED == "vlm_called"
        assert EVENT_CRITIC_DISAGREED == "critic_disagreed"
        assert EVENT_HUMAN_REVIEW_TRIGGERED == "human_review_triggered"
        assert EVENT_EXPORT_COMPLETED == "export_completed"

    def test_span_names_are_stable(self) -> None:
        assert SPAN_EXTRACTION_PASS == "extraction.pass"
        assert SPAN_VLM_REQUEST == "vlm.request"
        assert SPAN_RECONCILER == "extraction.reconciler"
        assert SPAN_CRITIC == "extraction.critic"


# ---------------------------------------------------------------------------
# build_pass_span_attrs
# ---------------------------------------------------------------------------


class TestBuildPassSpanAttrs:
    def test_minimum_attrs(self) -> None:
        a = build_pass_span_attrs(pass_name="pass1_vlm")
        assert a == {"pass": "pass1_vlm"}

    def test_full_attrs(self) -> None:
        a = build_pass_span_attrs(
            pass_name="pass2_auditor",
            model_id="gemma-4-31b-vl",
            latency_ms=540.0,
            tokens_in=312,
            tokens_out=128,
            page_number=2,
            profile="medical-rcm",
            tenant_id="acme",
            trace_id="t-123",
            document_type="CMS-1500",
            extra={"reconciler_tier": 2},
        )
        assert a["pass"] == "pass2_auditor"
        assert a["model_id"] == "gemma-4-31b-vl"
        assert a["latency_ms"] == 540.0
        assert a["tokens_in"] == 312
        assert a["tokens_out"] == 128
        assert a["page_number"] == 2
        assert a["profile"] == "medical-rcm"
        assert a["tenant_id"] == "acme"
        assert a["trace_id"] == "t-123"
        assert a["document_type"] == "CMS-1500"
        assert a["reconciler_tier"] == 2

    def test_drops_none_values(self) -> None:
        a = build_pass_span_attrs(
            pass_name="pass1_vlm",
            model_id=None,
            tokens_in=None,
            extra={"x": None, "y": "kept"},
        )
        assert "model_id" not in a
        assert "tokens_in" not in a
        assert "x" not in a
        assert a["y"] == "kept"

    def test_pulls_trace_id_from_contextvars(self) -> None:
        # Bind a trace_id via the helper; build_pass_span_attrs
        # without explicit trace_id should pick it up.
        try:
            bind_trace_id("trace-xyz")
            a = build_pass_span_attrs(pass_name="pass1_vlm")
            assert a.get("trace_id") == "trace-xyz"
        finally:
            clear_trace_id()


# ---------------------------------------------------------------------------
# trace_id helpers
# ---------------------------------------------------------------------------


class TestTraceIdHelpers:
    def test_bind_returns_uuid_when_none(self) -> None:
        try:
            tid = bind_trace_id()
            assert tid
            assert get_current_trace_id() == tid
        finally:
            clear_trace_id()

    def test_bind_uses_provided_value(self) -> None:
        try:
            bind_trace_id("custom-id", tenant_id="acme")
            assert get_current_trace_id() == "custom-id"
            ctx = structlog.contextvars.get_contextvars()
            assert ctx.get("tenant_id") == "acme"
        finally:
            clear_trace_id()

    def test_clear_unbinds(self) -> None:
        bind_trace_id("x")
        clear_trace_id()
        assert get_current_trace_id() is None

    def test_trace_scope_round_trip(self) -> None:
        with trace_scope("scoped-id", tenant_id="acme") as tid:
            assert tid == "scoped-id"
            assert get_current_trace_id() == "scoped-id"
        # Cleared on exit.
        assert get_current_trace_id() is None


# ---------------------------------------------------------------------------
# AuditContext serialisation
# ---------------------------------------------------------------------------


class TestAuditContextSerialisation:
    def test_carries_trace_id(self) -> None:
        ctx = AuditContext(trace_id="t-1")
        d = ctx.to_dict()
        assert d.get("trace_id") == "t-1"

    def test_carries_tenant_id(self) -> None:
        ctx = AuditContext(tenant_id="acme")
        d = ctx.to_dict()
        assert d.get("tenant_id") == "acme"

    def test_omits_unset_fields(self) -> None:
        ctx = AuditContext()
        d = ctx.to_dict()
        # Phase 6 fields not stamped when None.
        assert "trace_id" not in d
        assert "tenant_id" not in d


# ---------------------------------------------------------------------------
# AuditLogger pulls trace_id from contextvars
# ---------------------------------------------------------------------------


class TestAuditLoggerPullsTraceId:
    def test_log_picks_up_bound_trace_id(self, tmp_path: Path) -> None:
        # Force a fresh AuditLogger pointed at a temp dir so this test
        # doesn't depend on prod state.
        AuditLogger._instance = None  # reset singleton
        logger = AuditLogger.get_instance(
            log_dir=str(tmp_path),
            async_logging=False,
            mask_phi=False,
        )
        try:
            with trace_scope("audit-trace-1", tenant_id="audit-tenant"):
                event_id = logger.log(
                    event_type=AuditEventType.SYSTEM_START,
                    message="trace-id propagation test",
                )
                assert event_id

            # Read back the on-disk audit file and confirm trace_id
            # got threaded through.
            log_files = list(Path(tmp_path).glob("audit_*.jsonl"))
            assert log_files, "expected an audit log file to be written"
            content = log_files[0].read_text(encoding="utf-8")
            assert "audit-trace-1" in content
            assert "audit-tenant" in content
        finally:
            AuditLogger._instance = None
            clear_trace_id()


# ---------------------------------------------------------------------------
# emit_export_event noop
# ---------------------------------------------------------------------------


class TestEmitExportEventNoop:
    def test_noop_when_dispatcher_inactive(self) -> None:
        # Replace the dispatcher with an empty one (no sinks).
        set_dispatcher(ObservabilityDispatcher(sinks=[]))
        # Should not raise even with all attrs populated.
        emit_export_event(
            exporter="json",
            style="standard",
            record_count=5,
            success=True,
            duration_ms=12.3,
            profile="medical-rcm",
            document_type="CMS-1500",
        )
