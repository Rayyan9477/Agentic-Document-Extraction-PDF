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
    """Context for audit events."""

    user_id: str | None = None
    session_id: str | None = None
    request_id: str | None = None
    client_ip: str | None = None
    user_agent: str | None = None
    resource_type: str | None = None
    resource_id: str | None = None
    action: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)

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
        # IP addresses (for extra privacy)
        (r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b", "[IP-REDACTED]"),
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

        # Initialize hash chain
        self._load_last_hash()

    def _get_current_log_file(self) -> Path:
        """Get current log file path."""
        date_str = datetime.now(UTC).strftime("%Y-%m-%d")
        return self._log_dir / f"audit_{date_str}.jsonl"

    def _load_last_hash(self) -> None:
        """Load last hash from existing log for chain continuity."""
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
        """Flush a batch of events to the writer."""
        failed_count = 0
        last_error = None

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

        # Add extra data to metadata
        if extra:
            context.metadata.update(extra)

        # Mask PHI if enabled
        if self._mask_phi:
            message = PHIMasker.mask(message)
            if context.metadata:
                context.metadata = PHIMasker.mask_dict(context.metadata)

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
