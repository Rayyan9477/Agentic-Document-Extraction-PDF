"""V3 Phase 7 — audit chain verification tests."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.security.audit import (
    AuditEventType,
    AuditLogger,
    AuditOutcome,
    AuditSeverity,
    bind_trace_id,
    clear_trace_id,
    verify_audit_chain,
)


def _write_audit_entries(tmp_path: Path, count: int = 5) -> Path:
    AuditLogger._instance = None
    logger = AuditLogger.get_instance(
        log_dir=str(tmp_path),
        async_logging=False,
        mask_phi=False,
    )
    try:
        bind_trace_id("audit-chain-test", tenant_id="tenant-A")
        for i in range(count):
            logger.log(
                event_type=AuditEventType.SYSTEM_START,
                message=f"chain entry {i}",
                severity=AuditSeverity.INFO,
                outcome=AuditOutcome.SUCCESS,
            )
    finally:
        clear_trace_id()
        AuditLogger._instance = None

    files = list(tmp_path.glob("audit_*.jsonl"))
    assert files, "expected an audit log file"
    return files[0]


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestVerifyAuditChain:
    def test_clean_chain_is_intact(self, tmp_path: Path) -> None:
        log_path = _write_audit_entries(tmp_path, count=4)
        result = verify_audit_chain(log_path)
        assert result.chain_intact is True
        assert result.total_entries == 4
        assert result.verified_entries == 4
        assert result.first_break_at is None

    def test_missing_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            verify_audit_chain(tmp_path / "nope.jsonl")


# ---------------------------------------------------------------------------
# Tamper detection
# ---------------------------------------------------------------------------


class TestTamperDetection:
    def test_modified_message_breaks_chain(self, tmp_path: Path) -> None:
        log_path = _write_audit_entries(tmp_path, count=3)
        # Read all lines, alter one entry's message in-place.
        lines = log_path.read_text(encoding="utf-8").splitlines()
        records = [json.loads(line) for line in lines if line.strip()]
        records[1]["message"] = "TAMPERED MESSAGE"
        log_path.write_text(
            "\n".join(json.dumps(r) for r in records) + "\n",
            encoding="utf-8",
        )

        result = verify_audit_chain(log_path)
        assert result.chain_intact is False
        # The break is detected at entry 2 (1-indexed).
        assert result.first_break_at == 2
        assert result.first_break_reason
        assert "event_hash mismatch" in result.first_break_reason

    def test_swapped_records_break_chain(self, tmp_path: Path) -> None:
        log_path = _write_audit_entries(tmp_path, count=4)
        lines = log_path.read_text(encoding="utf-8").splitlines()
        records = [json.loads(line) for line in lines if line.strip()]
        # Swap entries 2 and 3 — their previous_hash now points at
        # the wrong predecessor.
        records[1], records[2] = records[2], records[1]
        log_path.write_text(
            "\n".join(json.dumps(r) for r in records) + "\n",
            encoding="utf-8",
        )

        result = verify_audit_chain(log_path)
        assert result.chain_intact is False

    def test_malformed_json_line(self, tmp_path: Path) -> None:
        log_path = _write_audit_entries(tmp_path, count=2)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write("{not json\n")
        result = verify_audit_chain(log_path)
        assert result.chain_intact is False
        assert result.first_break_reason == "malformed JSON"
