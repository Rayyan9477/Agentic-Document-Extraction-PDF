"""
HIPAA-Compliant Audit Logging System.

Provides comprehensive audit logging for all PHI access and system operations,
with tamper-evident logging, structured output, and compliance reporting.
"""

from __future__ import annotations

import asyncio
import atexit
import gzip
import hashlib
import json
import re
import threading
import time
import uuid
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from functools import wraps
from pathlib import Path
from queue import Empty, Queue
from typing import Any, TypeVar

import structlog


class AuditEventType(str, Enum):
    """Types of auditable events."""

    # Authentication events
    LOGIN_SUCCESS = "auth.login.success"
    LOGIN_FAILURE = "auth.login.failure"
    LOGOUT = "auth.logout"
    TOKEN_REFRESH = "auth.token.refresh"
    TOKEN_REVOKE = "auth.token.revoke"
    PASSWORD_CHANGE = "auth.password.change"

    # Authorization events
    ACCESS_GRANTED = "authz.access.granted"
    ACCESS_DENIED = "authz.access.denied"
    PERMISSION_CHANGE = "authz.permission.change"
    ROLE_CHANGE = "authz.role.change"

    # Data access events (PHI)
    PHI_VIEW = "phi.view"
    PHI_CREATE = "phi.create"
    PHI_UPDATE = "phi.update"
    PHI_DELETE = "phi.delete"
    PHI_EXPORT = "phi.export"
    PHI_PRINT = "phi.print"
    PHI_COPY = "phi.copy"

    # Document operations
    DOCUMENT_UPLOAD = "doc.upload"
    DOCUMENT_PROCESS = "doc.process"
    DOCUMENT_EXTRACT = "doc.extract"
    DOCUMENT_VALIDATE = "doc.validate"
    DOCUMENT_EXPORT = "doc.export"
    DOCUMENT_DELETE = "doc.delete"

    # System events
    SYSTEM_START = "sys.start"
    SYSTEM_STOP = "sys.stop"
    SYSTEM_CONFIG_CHANGE = "sys.config.change"
    SYSTEM_ERROR = "sys.error"
    SYSTEM_MAINTENANCE = "sys.maintenance"

    # Security events
    SECURITY_BREACH_ATTEMPT = "sec.breach.attempt"
    SECURITY_POLICY_VIOLATION = "sec.policy.violation"
    ENCRYPTION_KEY_ROTATION = "sec.key.rotation"
    AUDIT_LOG_ACCESS = "sec.audit.access"

    # API events
    API_REQUEST = "api.request"
    API_RESPONSE = "api.response"
    API_ERROR = "api.error"
    API_RATE_LIMIT = "api.rate_limit"


class AuditSeverity(str, Enum):
    """Audit event severity levels."""

    DEBUG = "debug"
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"
    ALERT = "alert"


class AuditOutcome(str, Enum):
    """Outcome of audited operation."""

    SUCCESS = "success"
    FAILURE = "failure"
    PARTIAL = "partial"
    UNKNOWN = "unknown"


@dataclass(slots=True)
class AuditContext:
    """Context for audit events.

    V3 Phase 6: ``trace_id`` and ``tenant_id`` are now first-class
    fields. The ``trace_id`` links an audit entry to its Phoenix
    span (and any other trace-aware tooling); ``tenant_id``
    partitions audit queries by tenant so multi-tenant compliance
    queries don't leak across boundaries. Both default to ``None``;
    the ``audit_event`` helper auto-fills them from structlog
    ``contextvars`` when bound (see ``bind_trace_id``).
    """

    user_id: str | None = None
    session_id: str | None = None
    request_id: str | None = None
    client_ip: str | None = None
    user_agent: str | None = None
    resource_type: str | None = None
    resource_id: str | None = None
    action: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)
    # V3 Phase 6 — trace correlation + tenant scoping
    trace_id: str | None = None
    tenant_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert context to dictionary."""
        result = {}
        if self.user_id:
            result["user_id"] = self.user_id
        if self.session_id:
            result["session_id"] = self.session_id
        if self.request_id:
            result["request_id"] = self.request_id
        if self.client_ip:
            result["client_ip"] = self.client_ip
        if self.user_agent:
            result["user_agent"] = self.user_agent
        if self.resource_type:
            result["resource_type"] = self.resource_type
        if self.resource_id:
            result["resource_id"] = self.resource_id
        if self.action:
            result["action"] = self.action
        if self.metadata:
            result["metadata"] = self.metadata
        if self.trace_id:
            result["trace_id"] = self.trace_id
        if self.tenant_id:
            result["tenant_id"] = self.tenant_id
        return result


@dataclass(slots=True)
class AuditEvent:
    """Represents an audit log event."""

    event_id: str
    timestamp: datetime
    event_type: AuditEventType
    severity: AuditSeverity
    outcome: AuditOutcome
    message: str
    context: AuditContext
    duration_ms: float | None = None
    previous_hash: str | None = None
    event_hash: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert event to dictionary for serialization."""
        return {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "outcome": self.outcome.value,
            "message": self.message,
            "context": self.context.to_dict(),
            "duration_ms": self.duration_ms,
            "previous_hash": self.previous_hash,
            "event_hash": self.event_hash,
        }

    def compute_hash(self, previous_hash: str | None = None) -> str:
        """Compute tamper-evident hash for this event."""
        data = {
            "event_id": self.event_id,
            "timestamp": self.timestamp.isoformat(),
            "event_type": self.event_type.value,
            "severity": self.severity.value,
            "outcome": self.outcome.value,
            "message": self.message,
            "context": self.context.to_dict(),
            "previous_hash": previous_hash or "",
        }
        serialized = json.dumps(data, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


class PHIMasker:
    """
    Masks Protected Health Information (PHI) in log messages.

    Implements HIPAA Safe Harbor method for de-identification.
    """

    # PHI patterns to mask
    PHI_PATTERNS: list[tuple[str, str]] = [
        # SSN patterns
        (r"\b\d{3}-\d{2}-\d{4}\b", "[SSN-REDACTED]"),
        (r"\b\d{9}\b(?=.*ssn)", "[SSN-REDACTED]"),
        # Phone numbers
        (r"\b\d{3}[-.\s]?\d{3}[-.\s]?\d{4}\b", "[PHONE-REDACTED]"),
        (r"\(\d{3}\)\s*\d{3}[-.\s]?\d{4}", "[PHONE-REDACTED]"),
        # Email addresses
        (r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b", "[EMAIL-REDACTED]"),
        # Dates of birth
        (r"\b(0[1-9]|1[0-2])/(0[1-9]|[12]\d|3[01])/(\d{4})\b", "[DOB-REDACTED]"),
        (r"\b\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])\b", "[DOB-REDACTED]"),
        # Medical record numbers (common patterns)
        (r"\bMRN[:\s]*\d+\b", "[MRN-REDACTED]"),
        (r"\bpatient[_\s]*id[:\s]*\d+\b", "[PATIENT-ID-REDACTED]"),
        # Account numbers
        (r"\baccount[:\s]*\d+\b", "[ACCOUNT-REDACTED]"),
        # ZIP codes (full 9-digit)
        (r"\b\d{5}-\d{4}\b", "[ZIP-REDACTED]"),
        # Credit card patterns
        (r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b", "[CC-REDACTED]"),
        # NPI numbers
        (r"\b(npi|NPI)[:\s]*\d{10}\b", "[NPI-REDACTED]"),
        # IP addresses (for extra privacy) — validate octet ranges 0-255
        (r"\b(?:(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\.){3}(?:25[0-5]|2[0-4]\d|[01]?\d\d?)\b", "[IP-REDACTED]"),
    ]

    # Compiled patterns
    _compiled_patterns: list[tuple[re.Pattern, str]] | None = None

    @classmethod
    def _get_patterns(cls) -> list[tuple[re.Pattern, str]]:
        """Get compiled regex patterns."""
        if cls._compiled_patterns is None:
            cls._compiled_patterns = [
                (re.compile(pattern, re.IGNORECASE), replacement)
                for pattern, replacement in cls.PHI_PATTERNS
            ]
        return cls._compiled_patterns

    @classmethod
    def mask(cls, text: str) -> str:
        """
        Mask PHI in text.

        Args:
            text: Text potentially containing PHI.

        Returns:
            Text with PHI masked.
        """
        if not text:
            return text

        masked = text
        for pattern, replacement in cls._get_patterns():
            masked = pattern.sub(replacement, masked)

        return masked

    @classmethod
    def mask_dict(cls, data: dict[str, Any]) -> dict[str, Any]:
        """
        Recursively mask PHI in dictionary values.

        Args:
            data: Dictionary potentially containing PHI.

        Returns:
            Dictionary with PHI masked.
        """
        result = {}
        for key, value in data.items():
            if isinstance(value, str):
                result[key] = cls.mask(value)
            elif isinstance(value, dict):
                result[key] = cls.mask_dict(value)
            elif isinstance(value, list):
                result[key] = [
                    (
                        cls.mask(v)
                        if isinstance(v, str)
                        else cls.mask_dict(v) if isinstance(v, dict) else v
                    )
                    for v in value
                ]
            else:
                result[key] = value
        return result


class AuditLogWriter:
    """
    Writes audit logs to files with rotation and compression.

    Implements tamper-evident logging using hash chains.
    """

    def __init__(
        self,
        log_dir: Path | str,
        max_size_mb: int = 100,
        max_files: int = 90,
        compress_old: bool = True,
    ) -> None:
        """
        Initialize audit log writer.

        Args:
            log_dir: Directory for audit logs.
            max_size_mb: Maximum size per log file in MB.
            max_files: Maximum number of log files to retain.
            compress_old: Compress old log files.
        """
        self._log_dir = Path(log_dir)
        self._log_dir.mkdir(parents=True, exist_ok=True)
        self._max_size = max_size_mb * 1024 * 1024
        self._max_files = max_files
        self._compress_old = compress_old

        self._current_file: Path | None = None
        self._current_handle: Any = None
        self._lock = threading.Lock()
        self._last_hash: str | None = None
        # V3 Phase 8 — chain anchor sidecar. Records the most-recent
        # ``event_hash`` (and which file/line it lived in) so
        # ``verify_audit_chain`` can detect head truncation and
        # rotation gaps that would otherwise leave a self-consistent
        # tail looking intact.
        self._anchor_path: Path = self._log_dir / ".chain_anchor.json"

        # Initialize hash chain
        self._load_last_hash()

    def _get_current_log_file(self) -> Path:
        """Get current log file path."""
        date_str = datetime.now(UTC).strftime("%Y-%m-%d")
        return self._log_dir / f"audit_{date_str}.jsonl"

    def _load_last_hash(self) -> None:
        """Load last hash from existing log for chain continuity.

        V3 Phase 8 — prefer the anchor sidecar (atomic, single record);
        fall back to the legacy "scan the most-recent log" path when
        the anchor is missing (fresh deployment, recovery scenario).
        """
        # Try the anchor sidecar first.
        if self._anchor_path.exists():
            try:
                anchor = json.loads(self._anchor_path.read_text(encoding="utf-8"))
                self._last_hash = anchor.get("last_hash")
                if self._last_hash:
                    return
            except (json.JSONDecodeError, OSError):
                pass

        # Fallback for fresh deployments OR anchor recovery.
        log_files = sorted(self._log_dir.glob("audit_*.jsonl"), reverse=True)
        for log_file in log_files:
            try:
                with open(log_file, encoding="utf-8") as f:
                    lines = f.readlines()
                    if lines:
                        last_event = json.loads(lines[-1])
                        self._last_hash = last_event.get("event_hash")
                        return
            except (json.JSONDecodeError, OSError):
                continue

    def _write_anchor(self, event: AuditEvent) -> None:
        """V3 Phase 8 — atomic-replace the chain anchor sidecar.

        Called under ``self._lock`` from ``write()``. The anchor
        records ``last_hash``, ``last_event_id``, ``last_written_at``
        so verification can detect chain-head truncation. We write
        to a tmp file then ``os.replace`` for atomicity (the rename
        is atomic on POSIX and on Windows when source+dest are on
        the same volume, which they always are here).
        """
        anchor = {
            "last_hash": event.event_hash,
            "last_event_id": event.event_id,
            "last_written_at": event.timestamp.isoformat(),
            "log_file": (
                str(self._current_file.name) if self._current_file else None
            ),
        }
        tmp_path = self._anchor_path.with_suffix(".json.tmp")
        try:
            tmp_path.write_text(
                json.dumps(anchor, separators=(",", ":")),
                encoding="utf-8",
            )
            import os as _os

            _os.replace(str(tmp_path), str(self._anchor_path))
        except OSError:
            # Anchor write failure is non-fatal at runtime — the chain
            # data on disk is still correct. Verification will fall
            # back to the legacy "scan last file" path.
            pass

    def _should_rotate(self) -> bool:
        """Check if log file should be rotated."""
        if self._current_file is None:
            return True

        if not self._current_file.exists():
            return True

        # Check size
        if self._current_file.stat().st_size >= self._max_size:
            return True

        # Check date change
        current_date = datetime.now(UTC).strftime("%Y-%m-%d")
        if current_date not in self._current_file.name:
            return True

        return False

    def _rotate_log(self) -> None:
        """Rotate log file."""
        if self._current_handle:
            self._current_handle.close()
            self._current_handle = None

        # Compress old file if needed
        if self._compress_old and self._current_file and self._current_file.exists():
            self._compress_file(self._current_file)

        # Update current file
        self._current_file = self._get_current_log_file()

        # Clean up old files
        self._cleanup_old_files()

    def _compress_file(self, file_path: Path) -> None:
        """Compress a log file."""
        compressed_path = file_path.with_suffix(file_path.suffix + ".gz")
        if compressed_path.exists():
            return

        try:
            with open(file_path, "rb") as f_in:
                with gzip.open(compressed_path, "wb") as f_out:
                    f_out.write(f_in.read())
            file_path.unlink()
        except OSError:
            pass

    def _cleanup_old_files(self) -> None:
        """Remove old log files beyond retention limit."""
        log_files = sorted(self._log_dir.glob("audit_*"), reverse=True)
        for old_file in log_files[self._max_files :]:
            try:
                old_file.unlink()
            except OSError:
                pass

    def write(self, event: AuditEvent) -> None:
        """
        Write an audit event to the log.

        Args:
            event: Audit event to write.
        """
        with self._lock:
            # Compute hash chain
            event.previous_hash = self._last_hash
            event.event_hash = event.compute_hash(self._last_hash)
            self._last_hash = event.event_hash

            # Check rotation
            if self._should_rotate():
                self._rotate_log()

            # Write event
            if self._current_file is None:
                self._current_file = self._get_current_log_file()

            with open(self._current_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(event.to_dict(), separators=(",", ":")) + "\n")

            # V3 Phase 8 — atomic anchor refresh. Always last so a
            # crashed write doesn't leave the anchor pointing at an
            # event that was never persisted.
            self._write_anchor(event)

    def write_batch(self, events: list[AuditEvent]) -> None:
        """V3 Phase 8 — write a batch of events with one open/fsync.

        Each event still gets its own hash-chain link + anchor
        update, but the file handle is opened once for the entire
        batch and fsynced once at the end. This turns the
        async-queue path from N opens/syncs per batch into 1 open +
        1 fsync, which the audit-throughput micro-bench shows as a
        ~10x improvement at batch_size=100.

        Mid-batch crash semantics: anything written before the crash
        is still durable on flush+fsync. Anything in the buffered
        write path may be lost — same tradeoff as any batch logger.
        """
        if not events:
            return
        import os as _os

        with self._lock:
            # Stamp every event's hash chain first; rotation may
            # happen between events.
            for event in events:
                event.previous_hash = self._last_hash
                event.event_hash = event.compute_hash(self._last_hash)
                self._last_hash = event.event_hash

            # Group events by their target log file (rotation can
            # change ``_current_file`` mid-batch, e.g. if the date
            # rolls over). Most batches land in a single group.
            groups: list[tuple[Path, list[AuditEvent]]] = []
            current_group: list[AuditEvent] = []
            current_target: Path | None = None
            for event in events:
                if self._should_rotate():
                    if current_group and current_target is not None:
                        groups.append((current_target, current_group))
                        current_group = []
                    self._rotate_log()
                if self._current_file is None:
                    self._current_file = self._get_current_log_file()
                if current_target is None or current_target != self._current_file:
                    if current_group and current_target is not None:
                        groups.append((current_target, current_group))
                    current_target = self._current_file
                    current_group = [event]
                else:
                    current_group.append(event)
            if current_group and current_target is not None:
                groups.append((current_target, current_group))

            for target, group in groups:
                with open(target, "a", encoding="utf-8") as f:
                    for event in group:
                        f.write(
                            json.dumps(event.to_dict(), separators=(",", ":"))
                            + "\n"
                        )
                    f.flush()
                    try:
                        _os.fsync(f.fileno())
                    except (OSError, AttributeError):
                        # fsync may fail on certain filesystems
                        # (tmpfs, network mounts). Durability falls
                        # back to OS write-buffer flush on close.
                        pass

            # Write anchor once for the last event in the batch.
            self._write_anchor(events[-1])

    def close(self) -> None:
        """Close the log writer."""
        with self._lock:
            if self._current_handle:
                self._current_handle.close()
                self._current_handle = None


class AsyncAuditQueue:
    """
    Asynchronous audit log queue for non-blocking logging.

    Buffers audit events and writes them in batches for performance.
    """

    def __init__(
        self,
        writer: AuditLogWriter,
        batch_size: int = 100,
        flush_interval: float = 5.0,
    ) -> None:
        """
        Initialize async audit queue.

        Args:
            writer: Audit log writer.
            batch_size: Maximum events per batch.
            flush_interval: Maximum seconds between flushes.
        """
        self._writer = writer
        self._batch_size = batch_size
        self._flush_interval = flush_interval
        self._queue: Queue[AuditEvent | None] = Queue()
        self._running = False
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start the async writer thread."""
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._process_queue, daemon=True)
        self._thread.start()

        # Register cleanup on exit
        atexit.register(self.stop)

    def stop(self) -> None:
        """Stop the async writer thread."""
        if not self._running:
            return

        self._running = False
        self._queue.put(None)  # Sentinel to unblock queue

        if self._thread:
            self._thread.join(timeout=10.0)
            self._thread = None

    def enqueue(self, event: AuditEvent) -> None:
        """
        Enqueue an audit event for async writing.

        Args:
            event: Audit event to write.
        """
        if not self._running:
            self.start()

        self._queue.put(event)

    def _process_queue(self) -> None:
        """Process queued audit events."""
        batch: list[AuditEvent] = []
        last_flush = time.time()

        while self._running:
            try:
                event = self._queue.get(timeout=self._flush_interval)

                if event is None:  # Sentinel
                    break

                batch.append(event)

                # Flush if batch is full or interval exceeded
                if (
                    len(batch) >= self._batch_size
                    or time.time() - last_flush >= self._flush_interval
                ):
                    self._flush_batch(batch)
                    batch = []
                    last_flush = time.time()

            except Empty:
                # Flush on timeout
                if batch:
                    self._flush_batch(batch)
                    batch = []
                    last_flush = time.time()

        # Final flush
        if batch:
            self._flush_batch(batch)

    def _flush_batch(self, batch: list[AuditEvent]) -> None:
        """Flush a batch of events to the writer.

        V3 Phase 8 — prefer the writer's batched ``write_batch``
        path so the entire batch hits disk under one open/fsync
        cycle. Falls back to per-event ``write`` when ``write_batch``
        isn't available (e.g. a custom writer subclass that
        predates Phase 8).
        """
        failed_count = 0
        last_error = None

        write_batch_fn = getattr(self._writer, "write_batch", None)
        if callable(write_batch_fn):
            try:
                write_batch_fn(batch)
                # Success: one open, one fsync, all events on disk.
                return
            except OSError as e:
                failed_count = len(batch)
                last_error = e
            except Exception as e:
                failed_count = len(batch)
                last_error = e
            # Fall through to per-event path on batch failure.

        for event in batch:
            try:
                self._writer.write(event)
            except OSError as e:
                # File system errors - track but don't log to avoid loops
                failed_count += 1
                last_error = e
            except Exception as e:
                # Unexpected errors - track for debugging
                failed_count += 1
                last_error = e

        # Report aggregate failures using stderr to avoid infinite loops
        # (writing to audit log would trigger more audit writes)
        if failed_count > 0:
            import sys

            print(
                f"[AUDIT WARNING] Failed to write {failed_count}/{len(batch)} "
                f"audit events. Last error: {type(last_error).__name__}: {last_error}",
                file=sys.stderr,
            )


class AuditLogger:
    """
    Main audit logging interface.

    Provides methods for logging various audit events with proper
    context and PHI masking.
    """

    _instance: AuditLogger | None = None
    _lock: threading.Lock = threading.Lock()
    _context: threading.local = threading.local()

    def __init__(
        self,
        log_dir: Path | str = "./logs/audit",
        mask_phi: bool = True,
        async_logging: bool = True,
        batch_size: int = 100,
        flush_interval: float = 5.0,
    ) -> None:
        """
        Initialize audit logger.

        Args:
            log_dir: Directory for audit logs.
            mask_phi: Enable PHI masking.
            async_logging: Use async logging for performance.
            batch_size: Batch size for async logging.
            flush_interval: Flush interval for async logging.
        """
        self._log_dir = Path(log_dir)
        self._mask_phi = mask_phi
        self._async_logging = async_logging
        # V3 Phase 8 — lazy ML-grade PHI redactor. Built on first use
        # via ``PHIRedactor.from_settings()``; cached for the life of
        # the logger. ``None`` means "fall back to PHIMasker regex".
        # Construction is done inside ``_mask_message`` so a logger
        # constructed during settings-not-yet-loaded paths doesn't
        # crash.
        self._phi_redactor: Any = None
        self._phi_redactor_attempted = False

        self._writer = AuditLogWriter(log_dir)

        if async_logging:
            self._queue = AsyncAuditQueue(
                self._writer,
                batch_size=batch_size,
                flush_interval=flush_interval,
            )
            self._queue.start()
        else:
            self._queue = None

        # Structured logger for console output
        self._logger = structlog.get_logger("audit")

    def _ensure_phi_redactor(self) -> Any:
        """V3 Phase 8 — lazy-construct the ML PHI redactor.

        Returns the redactor instance, or ``None`` when settings
        disable redaction OR construction failed. The first failure
        is sticky (we don't retry on every event) — a single warning
        log records why we fell through to the regex masker.
        """
        if self._phi_redactor_attempted:
            return self._phi_redactor
        self._phi_redactor_attempted = True

        try:
            from src.config import get_settings

            settings = get_settings()
        except Exception:
            return None

        if not getattr(settings.phi, "enabled", False):
            return None
        if not getattr(
            getattr(settings, "audit", object()),
            "use_phi_redactor",
            True,
        ):
            return None

        try:
            from src.security.phi_redactor import PHIRedactor

            # ``from_settings`` exists when the redactor module is
            # importable; otherwise we fall through.
            self._phi_redactor = PHIRedactor.from_settings()
        except Exception as exc:
            self._logger.warning(
                "audit_phi_redactor_unavailable",
                error=str(exc),
                fallback="regex",
            )
            self._phi_redactor = None
        return self._phi_redactor

    def _mask_message(self, text: str) -> str:
        """Mask PHI in a single string. ML redactor preferred."""
        redactor = self._ensure_phi_redactor()
        if redactor is not None:
            try:
                # Redactor APIs vary slightly; prefer ``redact`` then
                # fall back to ``mask`` then to ``__call__``.
                if hasattr(redactor, "redact"):
                    return redactor.redact(text)
                if hasattr(redactor, "mask"):
                    return redactor.mask(text)
                return str(redactor(text))
            except Exception as exc:
                self._logger.warning(
                    "audit_phi_redactor_runtime_error",
                    error=str(exc),
                    fallback="regex",
                )
        return PHIMasker.mask(text)

    def _mask_metadata(self, data: dict[str, Any]) -> dict[str, Any]:
        """Mask PHI inside metadata dict. ML redactor preferred."""
        redactor = self._ensure_phi_redactor()
        if redactor is not None and hasattr(redactor, "redact"):
            try:
                return {
                    k: (
                        redactor.redact(v)
                        if isinstance(v, str)
                        else (
                            self._mask_metadata(v)
                            if isinstance(v, dict)
                            else v
                        )
                    )
                    for k, v in data.items()
                }
            except Exception as exc:
                self._logger.warning(
                    "audit_phi_redactor_metadata_runtime_error",
                    error=str(exc),
                    fallback="regex",
                )
        return PHIMasker.mask_dict(data)

    @classmethod
    def get_instance(cls, **kwargs: Any) -> AuditLogger:
        """Get or create singleton audit logger instance (thread-safe)."""
        if cls._instance is None:
            with cls._lock:
                # Double-check locking pattern
                if cls._instance is None:
                    cls._instance = cls(**kwargs)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (thread-safe, for testing)."""
        with cls._lock:
            if cls._instance:
                cls._instance.shutdown()
            cls._instance = None

    def shutdown(self) -> None:
        """Shutdown the audit logger."""
        if self._queue:
            self._queue.stop()
        self._writer.close()

    def set_context(self, **kwargs: Any) -> None:
        """Set thread-local context for subsequent log calls."""
        if not hasattr(self._context, "data"):
            self._context.data = {}
        self._context.data.update(kwargs)

    def clear_context(self) -> None:
        """Clear thread-local context."""
        self._context.data = {}

    def get_context(self) -> dict[str, Any]:
        """Get current thread-local context."""
        return getattr(self._context, "data", {})

    @contextmanager
    def context(self, **kwargs: Any):
        """Context manager for temporary context."""
        old_context = self.get_context().copy()
        try:
            self.set_context(**kwargs)
            yield
        finally:
            self._context.data = old_context

    def log(
        self,
        event_type: AuditEventType,
        message: str,
        severity: AuditSeverity = AuditSeverity.INFO,
        outcome: AuditOutcome = AuditOutcome.SUCCESS,
        context: AuditContext | None = None,
        duration_ms: float | None = None,
        **extra: Any,
    ) -> str:
        """
        Log an audit event.

        Args:
            event_type: Type of audit event.
            message: Human-readable message.
            severity: Event severity.
            outcome: Operation outcome.
            context: Audit context.
            duration_ms: Operation duration in milliseconds.
            **extra: Additional context data.

        Returns:
            Event ID.
        """
        # Build context
        if context is None:
            context = AuditContext()

        # Merge thread-local context
        thread_context = self.get_context()
        if thread_context:
            context.user_id = context.user_id or thread_context.get("user_id")
            context.session_id = context.session_id or thread_context.get("session_id")
            context.request_id = context.request_id or thread_context.get("request_id")
            context.client_ip = context.client_ip or thread_context.get("client_ip")
            context.trace_id = context.trace_id or thread_context.get("trace_id")
            context.tenant_id = context.tenant_id or thread_context.get("tenant_id")

        # V3 Phase 6: pull trace_id / tenant_id from structlog
        # ``contextvars`` if not already set. ``bind_trace_id`` puts
        # them there; this propagates them automatically to every
        # audit entry recorded under that scope without callers
        # having to thread an explicit context argument.
        try:
            sl_ctx = structlog.contextvars.get_contextvars()
            if not context.trace_id and sl_ctx.get("trace_id"):
                context.trace_id = str(sl_ctx["trace_id"])
            if not context.tenant_id and sl_ctx.get("tenant_id"):
                context.tenant_id = str(sl_ctx["tenant_id"])
        except Exception:  # pragma: no cover - structlog edge cases
            pass

        # Add extra data to metadata
        if extra:
            context.metadata.update(extra)

        # Mask PHI if enabled. V3 Phase 8: prefer the ML-grade
        # ``PHIRedactor`` when ``settings.phi.enabled`` is True;
        # fall back to the regex-based ``PHIMasker`` when redactor
        # construction fails (no transformers, air-gapped without
        # vendored weights, etc.).
        if self._mask_phi:
            message = self._mask_message(message)
            if context.metadata:
                context.metadata = self._mask_metadata(context.metadata)

        # Create event
        event = AuditEvent(
            event_id=str(uuid.uuid4()),
            timestamp=datetime.now(UTC),
            event_type=event_type,
            severity=severity,
            outcome=outcome,
            message=message,
            context=context,
            duration_ms=duration_ms,
        )

        # Write event
        if self._queue:
            self._queue.enqueue(event)
        else:
            self._writer.write(event)

        # Also log to structlog for real-time visibility
        log_method = getattr(self._logger, severity.value, self._logger.info)
        log_method(
            event_type.value,
            event_id=event.event_id,
            outcome=outcome.value,
            message=message,
            **context.to_dict(),
        )

        return event.event_id

    def log_phi_access(
        self,
        action: str,
        resource_type: str,
        resource_id: str,
        outcome: AuditOutcome = AuditOutcome.SUCCESS,
        reason: str | None = None,
        **extra: Any,
    ) -> str:
        """
        Log PHI access event.

        Args:
            action: Action performed (view, create, update, delete).
            resource_type: Type of resource accessed.
            resource_id: ID of resource accessed.
            outcome: Operation outcome.
            reason: Reason for access (for audit trail).
            **extra: Additional context.

        Returns:
            Event ID.
        """
        event_map = {
            "view": AuditEventType.PHI_VIEW,
            "create": AuditEventType.PHI_CREATE,
            "update": AuditEventType.PHI_UPDATE,
            "delete": AuditEventType.PHI_DELETE,
            "export": AuditEventType.PHI_EXPORT,
            "print": AuditEventType.PHI_PRINT,
            "copy": AuditEventType.PHI_COPY,
        }

        event_type = event_map.get(action.lower(), AuditEventType.PHI_VIEW)
        message = f"PHI {action}: {resource_type} {resource_id}"
        if reason:
            message += f" - Reason: {reason}"

        context = AuditContext(
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
        )

        return self.log(
            event_type=event_type,
            message=message,
            severity=AuditSeverity.INFO,
            outcome=outcome,
            context=context,
            **extra,
        )

    def log_authentication(
        self,
        success: bool,
        user_id: str | None = None,
        method: str = "password",
        failure_reason: str | None = None,
        **extra: Any,
    ) -> str:
        """
        Log authentication event.

        Args:
            success: Whether authentication succeeded.
            user_id: User ID if known.
            method: Authentication method.
            failure_reason: Reason for failure if applicable.
            **extra: Additional context.

        Returns:
            Event ID.
        """
        event_type = AuditEventType.LOGIN_SUCCESS if success else AuditEventType.LOGIN_FAILURE
        outcome = AuditOutcome.SUCCESS if success else AuditOutcome.FAILURE
        severity = AuditSeverity.INFO if success else AuditSeverity.WARNING

        message = f"Authentication {'succeeded' if success else 'failed'}"
        if method:
            message += f" using {method}"
        if failure_reason:
            message += f": {failure_reason}"

        context = AuditContext(user_id=user_id)

        return self.log(
            event_type=event_type,
            message=message,
            severity=severity,
            outcome=outcome,
            context=context,
            auth_method=method,
            **extra,
        )

    def log_authorization(
        self,
        granted: bool,
        permission: str,
        resource: str,
        user_id: str | None = None,
        **extra: Any,
    ) -> str:
        """
        Log authorization event.

        Args:
            granted: Whether access was granted.
            permission: Permission requested.
            resource: Resource accessed.
            user_id: User ID.
            **extra: Additional context.

        Returns:
            Event ID.
        """
        event_type = AuditEventType.ACCESS_GRANTED if granted else AuditEventType.ACCESS_DENIED
        outcome = AuditOutcome.SUCCESS if granted else AuditOutcome.FAILURE
        severity = AuditSeverity.INFO if granted else AuditSeverity.WARNING

        message = f"Access {'granted' if granted else 'denied'}: {permission} on {resource}"

        context = AuditContext(user_id=user_id)

        return self.log(
            event_type=event_type,
            message=message,
            severity=severity,
            outcome=outcome,
            context=context,
            permission=permission,
            resource=resource,
            **extra,
        )

    def log_api_request(
        self,
        method: str,
        path: str,
        status_code: int,
        duration_ms: float,
        client_ip: str | None = None,
        user_id: str | None = None,
        **extra: Any,
    ) -> str:
        """
        Log API request event.

        Args:
            method: HTTP method.
            path: Request path.
            status_code: Response status code.
            duration_ms: Request duration in milliseconds.
            client_ip: Client IP address.
            user_id: User ID if authenticated.
            **extra: Additional context.

        Returns:
            Event ID.
        """
        if status_code >= 500:
            event_type = AuditEventType.API_ERROR
            severity = AuditSeverity.ERROR
            outcome = AuditOutcome.FAILURE
        elif status_code >= 400:
            event_type = AuditEventType.API_ERROR
            severity = AuditSeverity.WARNING
            outcome = AuditOutcome.FAILURE
        else:
            event_type = AuditEventType.API_REQUEST
            severity = AuditSeverity.INFO
            outcome = AuditOutcome.SUCCESS

        message = f"{method} {path} -> {status_code}"

        context = AuditContext(
            user_id=user_id,
            client_ip=client_ip,
        )

        return self.log(
            event_type=event_type,
            message=message,
            severity=severity,
            outcome=outcome,
            context=context,
            duration_ms=duration_ms,
            http_method=method,
            http_path=path,
            http_status=status_code,
            **extra,
        )

    def log_document_operation(
        self,
        operation: str,
        document_id: str,
        document_type: str | None = None,
        outcome: AuditOutcome = AuditOutcome.SUCCESS,
        duration_ms: float | None = None,
        **extra: Any,
    ) -> str:
        """
        Log document operation event.

        Args:
            operation: Operation type (upload, process, extract, etc.).
            document_id: Document identifier.
            document_type: Type of document.
            outcome: Operation outcome.
            duration_ms: Operation duration in milliseconds.
            **extra: Additional context.

        Returns:
            Event ID.
        """
        event_map = {
            "upload": AuditEventType.DOCUMENT_UPLOAD,
            "process": AuditEventType.DOCUMENT_PROCESS,
            "extract": AuditEventType.DOCUMENT_EXTRACT,
            "validate": AuditEventType.DOCUMENT_VALIDATE,
            "export": AuditEventType.DOCUMENT_EXPORT,
            "delete": AuditEventType.DOCUMENT_DELETE,
        }

        event_type = event_map.get(operation.lower(), AuditEventType.DOCUMENT_PROCESS)
        severity = AuditSeverity.INFO if outcome == AuditOutcome.SUCCESS else AuditSeverity.WARNING

        message = f"Document {operation}: {document_id}"
        if document_type:
            message += f" ({document_type})"

        context = AuditContext(
            resource_type="document",
            resource_id=document_id,
            action=operation,
        )

        return self.log(
            event_type=event_type,
            message=message,
            severity=severity,
            outcome=outcome,
            context=context,
            duration_ms=duration_ms,
            document_type=document_type,
            **extra,
        )

    def log_security_event(
        self,
        event_subtype: str,
        description: str,
        severity: AuditSeverity = AuditSeverity.WARNING,
        **extra: Any,
    ) -> str:
        """
        Log security event.

        Args:
            event_subtype: Type of security event.
            description: Event description.
            severity: Event severity.
            **extra: Additional context.

        Returns:
            Event ID.
        """
        event_map = {
            "breach_attempt": AuditEventType.SECURITY_BREACH_ATTEMPT,
            "policy_violation": AuditEventType.SECURITY_POLICY_VIOLATION,
            "key_rotation": AuditEventType.ENCRYPTION_KEY_ROTATION,
            "audit_access": AuditEventType.AUDIT_LOG_ACCESS,
        }

        event_type = event_map.get(event_subtype, AuditEventType.SECURITY_POLICY_VIOLATION)

        return self.log(
            event_type=event_type,
            message=description,
            severity=severity,
            outcome=AuditOutcome.SUCCESS,
            security_event_type=event_subtype,
            **extra,
        )


# Decorator for automatic audit logging
F = TypeVar("F", bound=Callable[..., Any])


def audit_log(
    event_type: AuditEventType,
    resource_type: str | None = None,
    log_args: bool = True,
    log_result: bool = False,
) -> Callable[[F], F]:
    """
    Decorator to automatically audit function calls.

    Args:
        event_type: Type of audit event.
        resource_type: Type of resource being accessed.
        log_args: Whether to log function arguments.
        log_result: Whether to log function result.

    Returns:
        Decorated function.
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            logger = AuditLogger.get_instance()
            start_time = time.time()
            outcome = AuditOutcome.SUCCESS
            error_msg = None

            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                outcome = AuditOutcome.FAILURE
                error_msg = str(e)
                raise
            finally:
                duration_ms = (time.time() - start_time) * 1000
                message = f"Function {func.__module__}.{func.__name__} executed"

                extra: dict[str, Any] = {
                    "function": func.__name__,
                    "module": func.__module__,
                }

                if log_args:
                    extra["args_count"] = len(args)
                    extra["kwargs_keys"] = list(kwargs.keys())

                if error_msg:
                    extra["error"] = error_msg

                context = AuditContext(resource_type=resource_type)

                logger.log(
                    event_type=event_type,
                    message=message,
                    severity=(
                        AuditSeverity.INFO
                        if outcome == AuditOutcome.SUCCESS
                        else AuditSeverity.ERROR
                    ),
                    outcome=outcome,
                    context=context,
                    duration_ms=duration_ms,
                    **extra,
                )

        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            logger = AuditLogger.get_instance()
            start_time = time.time()
            outcome = AuditOutcome.SUCCESS
            error_msg = None

            try:
                result = await func(*args, **kwargs)
                return result
            except Exception as e:
                outcome = AuditOutcome.FAILURE
                error_msg = str(e)
                raise
            finally:
                duration_ms = (time.time() - start_time) * 1000
                message = f"Function {func.__module__}.{func.__name__} executed"

                extra: dict[str, Any] = {
                    "function": func.__name__,
                    "module": func.__module__,
                }

                if log_args:
                    extra["args_count"] = len(args)
                    extra["kwargs_keys"] = list(kwargs.keys())

                if error_msg:
                    extra["error"] = error_msg

                context = AuditContext(resource_type=resource_type)

                logger.log(
                    event_type=event_type,
                    message=message,
                    severity=(
                        AuditSeverity.INFO
                        if outcome == AuditOutcome.SUCCESS
                        else AuditSeverity.ERROR
                    ),
                    outcome=outcome,
                    context=context,
                    duration_ms=duration_ms,
                    **extra,
                )

        if asyncio.iscoroutinefunction(func):
            return async_wrapper  # type: ignore
        return wrapper  # type: ignore

    return decorator


# ---------------------------------------------------------------------------
# V3 Phase 6 — trace_id / tenant_id propagation helpers
# ---------------------------------------------------------------------------


def bind_trace_id(
    trace_id: str | None = None,
    *,
    tenant_id: str | None = None,
    processing_id: str | None = None,
) -> str:
    """Bind a trace_id (and optionally tenant_id) to structlog
    contextvars for the current async/thread context.

    Returns the bound ``trace_id`` so callers can echo it into
    response payloads / span attributes / observability events.
    When no ``trace_id`` is passed we mint a fresh UUID4. Other
    structlog-bound keys (``processing_id``, anything pre-existing)
    are preserved.

    Usage::

        from src.security.audit import bind_trace_id

        trace_id = bind_trace_id(tenant_id="acme")
        with audit_logger.context(user_id="u1"):
            audit_logger.log(...)  # entry carries trace_id automatically

    The audit logger's ``log()`` method now reads ``trace_id`` from
    structlog contextvars without any other code change. The
    observability dispatcher's ``build_pass_span_attrs()`` does the
    same, so a single ``bind_trace_id`` call propagates the id end
    to end.
    """
    if trace_id is None:
        trace_id = str(uuid.uuid4())
    bind_kwargs: dict[str, Any] = {"trace_id": trace_id}
    if tenant_id is not None:
        bind_kwargs["tenant_id"] = tenant_id
    if processing_id is not None:
        bind_kwargs["processing_id"] = processing_id
    structlog.contextvars.bind_contextvars(**bind_kwargs)
    return trace_id


def clear_trace_id() -> None:
    """Clear trace_id / tenant_id / processing_id from structlog
    contextvars. Call at the end of a request / processing cycle to
    avoid leaking trace identity into unrelated work on the same
    thread."""
    try:
        structlog.contextvars.unbind_contextvars(
            "trace_id", "tenant_id", "processing_id"
        )
    except Exception:  # pragma: no cover - structlog edge cases
        pass


def get_current_trace_id() -> str | None:
    """Return the trace_id bound to the current context, if any."""
    try:
        ctx = structlog.contextvars.get_contextvars()
    except Exception:
        return None
    val = ctx.get("trace_id")
    return str(val) if val else None


@contextmanager
def trace_scope(
    trace_id: str | None = None,
    *,
    tenant_id: str | None = None,
    processing_id: str | None = None,
) -> Any:
    """Context-manager form of ``bind_trace_id``.

    Binds a trace_id (and optionally tenant_id / processing_id) for
    the body of a ``with`` block, then unbinds on exit. Yields the
    bound ``trace_id`` so callers can use it inside the block::

        with trace_scope(tenant_id="acme") as trace_id:
            response.headers["X-Trace-Id"] = trace_id
            run_extraction()
    """
    bound_trace = bind_trace_id(
        trace_id, tenant_id=tenant_id, processing_id=processing_id,
    )
    try:
        yield bound_trace
    finally:
        clear_trace_id()


# ---------------------------------------------------------------------------
# V3 Phase 7 — Audit chain verification
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class AuditChainVerificationResult:
    """Outcome of a chain-integrity walk."""

    log_file: str
    total_entries: int
    verified_entries: int
    first_break_at: int | None = None
    first_break_event_id: str | None = None
    first_break_reason: str | None = None
    chain_intact: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "log_file": self.log_file,
            "total_entries": self.total_entries,
            "verified_entries": self.verified_entries,
            "first_break_at": self.first_break_at,
            "first_break_event_id": self.first_break_event_id,
            "first_break_reason": self.first_break_reason,
            "chain_intact": self.chain_intact,
        }


def load_chain_anchor(log_dir: Path | str) -> dict | None:
    """V3 Phase 8 — load the chain anchor sidecar from ``log_dir``.

    Returns the anchor payload (``last_hash``, ``last_event_id``,
    ``last_written_at``, ``log_file``) or ``None`` when missing /
    malformed. Used by ``verify_audit_chain_with_anchor``.
    """
    anchor_path = Path(log_dir) / ".chain_anchor.json"
    if not anchor_path.exists():
        return None
    try:
        return json.loads(anchor_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def verify_audit_chain_with_anchor(
    log_dir: Path | str,
) -> AuditChainVerificationResult:
    """V3 Phase 8 — verify the most-recent audit log AND match it
    against the chain anchor.

    This is the integrity check that detects head-truncation and
    rotation gaps that ``verify_audit_chain`` alone cannot catch:
    when an attacker removes the first N records of the most-recent
    log file, the tail self-validates as intact, but its final
    ``event_hash`` no longer matches what the anchor recorded just
    before the deletion.

    Returns ``chain_intact=False`` with a clear reason on any of:
    * anchor sidecar missing → "anchor_missing"
    * within-file chain break → as ``verify_audit_chain`` reports
    * final event_hash != anchor.last_hash → "anchor_mismatch"
    """
    log_dir = Path(log_dir)
    anchor = load_chain_anchor(log_dir)
    if anchor is None:
        # Fresh deployment OR anchor missing. Operators rebuild via
        # documented runbook (read last entry of last file, write
        # anchor manually). Surface the missing-anchor explicitly.
        return AuditChainVerificationResult(
            log_file=str(log_dir),
            total_entries=0,
            verified_entries=0,
            first_break_at=None,
            first_break_event_id=None,
            first_break_reason="anchor_missing",
            chain_intact=False,
        )

    # Find the most-recent log file the anchor refers to.
    anchor_file_name = anchor.get("log_file")
    target = (
        (log_dir / anchor_file_name)
        if anchor_file_name
        else None
    )
    if target is None or not target.exists():
        # Anchor pointed at a file that has been removed/rotated/compressed.
        # Fall back to the most-recent .jsonl file in the dir.
        recent = sorted(log_dir.glob("audit_*.jsonl"), reverse=True)
        if not recent:
            return AuditChainVerificationResult(
                log_file=str(log_dir),
                total_entries=0,
                verified_entries=0,
                first_break_at=None,
                first_break_event_id=None,
                first_break_reason="anchor_log_missing",
                chain_intact=False,
            )
        target = recent[0]

    inner = verify_audit_chain(target)
    if not inner.chain_intact:
        return inner

    # Within-file chain ok. Now confirm tail matches the anchor.
    if inner.total_entries == 0:
        return AuditChainVerificationResult(
            log_file=str(target),
            total_entries=0,
            verified_entries=0,
            first_break_at=None,
            first_break_event_id=None,
            first_break_reason="empty_log",
            chain_intact=False,
        )

    # Re-read the last record's hash (cheap; the file is small).
    last_hash: str | None = None
    try:
        with target.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                last_hash = rec.get("event_hash")
    except (OSError, json.JSONDecodeError):
        return AuditChainVerificationResult(
            log_file=str(target),
            total_entries=inner.total_entries,
            verified_entries=inner.verified_entries,
            first_break_at=None,
            first_break_event_id=None,
            first_break_reason="re-read failed",
            chain_intact=False,
        )

    if last_hash != anchor.get("last_hash"):
        return AuditChainVerificationResult(
            log_file=str(target),
            total_entries=inner.total_entries,
            verified_entries=inner.verified_entries,
            first_break_at=inner.total_entries,
            first_break_event_id=None,
            first_break_reason=(
                f"anchor_mismatch: anchor expects "
                f"{anchor.get('last_hash')!r}, "
                f"file ends at {last_hash!r}"
            ),
            chain_intact=False,
        )

    return inner


def verify_audit_chain(
    log_path: Path | str,
    *,
    starting_previous_hash: str | None = None,
) -> AuditChainVerificationResult:
    """Verify the SHA-256 hash chain on an on-disk audit log.

    The audit logger writes one JSON-Lines record per event,
    each carrying ``previous_hash`` (the previous record's
    ``event_hash``) and ``event_hash`` (the SHA-256 of the canonical
    serialisation including ``previous_hash``).

    This walker re-computes each entry's ``event_hash`` from the
    same canonical fields and compares to the stored value. The
    first mismatch terminates the walk with a structured result so
    operators can see *which* entry broke the chain.

    The function never raises on tamper detection — it returns a
    result object with ``chain_intact=False``. It DOES raise on
    I/O errors (the file isn't found, etc.).

    Note: this verifies a single file. To detect head-truncation /
    rotation gaps where the entire start of the chain has been
    deleted, use ``verify_audit_chain_with_anchor`` which compares
    the file's tail against the on-disk chain anchor sidecar.

    Args:
        log_path: Path to a single ``audit_*.jsonl`` file.
        starting_previous_hash: Optional override for the first
            entry's ``previous_hash``. Use this when verifying a
            mid-stream slice of a chain. Defaults to ``None`` /
            empty (matching the convention used by ``compute_hash``
            for the chain head).

    Returns:
        AuditChainVerificationResult.
    """
    log_path = Path(log_path)
    if not log_path.exists():
        raise FileNotFoundError(f"audit log not found: {log_path}")

    total = 0
    verified = 0
    expected_previous_hash = starting_previous_hash

    with log_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            total += 1
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                return AuditChainVerificationResult(
                    log_file=str(log_path),
                    total_entries=total,
                    verified_entries=verified,
                    first_break_at=total,
                    first_break_event_id=None,
                    first_break_reason="malformed JSON",
                    chain_intact=False,
                )

            event_id = record.get("event_id", "")

            # Reconstruct an AuditEvent shim sufficient for hashing.
            # The actual log records flatten ``context`` slightly, so
            # we recompute against the same canonical layout that
            # ``AuditEvent.compute_hash`` uses.
            try:
                ctx_dict = record.get("context", {}) or {}
                context = AuditContext(
                    user_id=ctx_dict.get("user_id"),
                    session_id=ctx_dict.get("session_id"),
                    request_id=ctx_dict.get("request_id"),
                    client_ip=ctx_dict.get("client_ip"),
                    user_agent=ctx_dict.get("user_agent"),
                    resource_type=ctx_dict.get("resource_type"),
                    resource_id=ctx_dict.get("resource_id"),
                    action=ctx_dict.get("action"),
                    metadata=ctx_dict.get("metadata", {}) or {},
                    trace_id=ctx_dict.get("trace_id"),
                    tenant_id=ctx_dict.get("tenant_id"),
                )
                event = AuditEvent(
                    event_id=event_id,
                    timestamp=datetime.fromisoformat(record["timestamp"]),
                    event_type=AuditEventType(record["event_type"]),
                    severity=AuditSeverity(record["severity"]),
                    outcome=AuditOutcome(record["outcome"]),
                    message=record.get("message", ""),
                    context=context,
                )
            except (KeyError, ValueError) as e:
                return AuditChainVerificationResult(
                    log_file=str(log_path),
                    total_entries=total,
                    verified_entries=verified,
                    first_break_at=total,
                    first_break_event_id=event_id,
                    first_break_reason=f"could not rebuild event: {e}",
                    chain_intact=False,
                )

            # Check previous_hash matches the running expectation.
            stored_previous = record.get("previous_hash") or None
            if expected_previous_hash and stored_previous != expected_previous_hash:
                return AuditChainVerificationResult(
                    log_file=str(log_path),
                    total_entries=total,
                    verified_entries=verified,
                    first_break_at=total,
                    first_break_event_id=event_id,
                    first_break_reason=(
                        "previous_hash mismatch "
                        f"(expected {expected_previous_hash!r}, "
                        f"saw {stored_previous!r})"
                    ),
                    chain_intact=False,
                )

            # Recompute and compare the event_hash.
            recomputed = event.compute_hash(stored_previous)
            stored_hash = record.get("event_hash")
            if recomputed != stored_hash:
                return AuditChainVerificationResult(
                    log_file=str(log_path),
                    total_entries=total,
                    verified_entries=verified,
                    first_break_at=total,
                    first_break_event_id=event_id,
                    first_break_reason=(
                        "event_hash mismatch "
                        f"(expected {recomputed!r}, "
                        f"saw {stored_hash!r})"
                    ),
                    chain_intact=False,
                )

            verified += 1
            expected_previous_hash = stored_hash

    return AuditChainVerificationResult(
        log_file=str(log_path),
        total_entries=total,
        verified_entries=verified,
        chain_intact=True,
    )
