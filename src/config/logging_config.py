"""
Structured logging configuration using structlog.

Provides JSON-formatted logging with contextual information,
PHI masking for HIPAA compliance, and integration with Python's
standard logging module.
"""

import logging
import logging.handlers
import re
import sys
from datetime import UTC, datetime
from typing import Any

import structlog
from structlog.types import EventDict, Processor

from src.config.settings import LogFormat, get_settings


# PHI patterns for masking sensitive information
PHI_PATTERNS: list[tuple[re.Pattern[str], str]] = [
    # SSN patterns
    (re.compile(r"\b\d{3}-\d{2}-\d{4}\b"), "[SSN-MASKED]"),
    (re.compile(r"\b\d{9}\b(?=.*ssn)", re.IGNORECASE), "[SSN-MASKED]"),
    # Phone numbers
    (re.compile(r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b"), "[PHONE-MASKED]"),
    # Email addresses
    (re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"), "[EMAIL-MASKED]"),
    # Date of birth patterns
    (re.compile(r"\b\d{2}/\d{2}/\d{4}\b(?=.*(?:dob|birth))", re.IGNORECASE), "[DOB-MASKED]"),
    (re.compile(r"\b\d{4}-\d{2}-\d{2}\b(?=.*(?:dob|birth))", re.IGNORECASE), "[DOB-MASKED]"),
    # Medicare/Medicaid IDs
    (re.compile(r"\b[A-Za-z]{1}\d{4}[A-Za-z]{1}\d{4}\b"), "[MEDICARE-ID-MASKED]"),
    # Medical Record Numbers (common patterns)
    (re.compile(r"\bMRN[:\s]*\d{6,10}\b", re.IGNORECASE), "[MRN-MASKED]"),
    # NPI numbers
    (re.compile(r"\bNPI[:\s]*\d{10}\b", re.IGNORECASE), "[NPI-MASKED]"),
    # Credit card numbers
    (re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b"), "[CC-MASKED]"),
    # Patient names (common patterns - conservative approach)
    (
        re.compile(r"patient[_\s]?name[:\s]+[A-Za-z]+\s+[A-Za-z]+", re.IGNORECASE),
        "[PATIENT-NAME-MASKED]",
    ),
]


def mask_phi(event_dict: EventDict) -> EventDict:
    """
    Mask Protected Health Information (PHI) in log entries.

    Args:
        event_dict: The event dictionary to process.

    Returns:
        EventDict with PHI masked.
    """
    settings = get_settings()
    if not settings.hipaa.phi_masking_enabled:
        return event_dict

    def mask_value(value: Any) -> Any:
        if isinstance(value, str):
            result = value
            for pattern, replacement in PHI_PATTERNS:
                result = pattern.sub(replacement, result)
            return result
        if isinstance(value, dict):
            return {k: mask_value(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return type(value)(mask_value(item) for item in value)
        return value

    return {key: mask_value(val) for key, val in event_dict.items()}


def add_timestamp(
    logger: logging.Logger,
    method_name: str,
    event_dict: EventDict,
) -> EventDict:
    """
    Add ISO-8601 timestamp to log entries.

    Args:
        logger: Logger instance.
        method_name: Name of the logging method.
        event_dict: Event dictionary.

    Returns:
        EventDict with timestamp added.
    """
    event_dict["timestamp"] = datetime.now(UTC).isoformat()
    return event_dict


def add_service_info(
    logger: logging.Logger,
    method_name: str,
    event_dict: EventDict,
) -> EventDict:
    """
    Add service metadata to log entries.

    Args:
        logger: Logger instance.
        method_name: Name of the logging method.
        event_dict: Event dictionary.

    Returns:
        EventDict with service info added.
    """
    settings = get_settings()
    event_dict["service"] = settings.app_name
    event_dict["version"] = settings.app_version
    event_dict["environment"] = settings.app_env.value
    return event_dict


def add_caller_info(
    logger: logging.Logger,
    method_name: str,
    event_dict: EventDict,
) -> EventDict:
    """
    Add caller information to log entries.

    Args:
        logger: Logger instance.
        method_name: Name of the logging method.
        event_dict: Event dictionary.

    Returns:
        EventDict with caller info added.
    """
    settings = get_settings()
    if not settings.logging.include_caller:
        return event_dict

    # Extract caller info from call stack
    record = event_dict.get("_record")
    if record:
        event_dict["caller"] = {
            "filename": record.filename,
            "function": record.funcName,
            "line": record.lineno,
            "module": record.module,
        }
    return event_dict


def drop_color_message_key(
    logger: logging.Logger,
    method_name: str,
    event_dict: EventDict,
) -> EventDict:
    """
    Remove color_message key that's only used for console output.

    Args:
        logger: Logger instance.
        method_name: Name of the logging method.
        event_dict: Event dictionary.

    Returns:
        EventDict with color_message removed.
    """
    event_dict.pop("color_message", None)
    return event_dict


class PHIFilter(logging.Filter):
    """
    Logging filter that masks PHI in log records.

    Applies PHI masking patterns to all string fields in log records
    before they are emitted.
    """

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Apply PHI masking to the log record.

        Args:
            record: The log record to filter.

        Returns:
            Always True to allow the record through after masking.
        """
        settings = get_settings()
        if not settings.hipaa.phi_masking_enabled:
            return True

        # Mask the main message
        if isinstance(record.msg, str):
            for pattern, replacement in PHI_PATTERNS:
                record.msg = pattern.sub(replacement, record.msg)

        # Mask args if present
        if record.args:
            masked_args = []
            for arg in record.args:
                if isinstance(arg, str):
                    masked = arg
                    for pattern, replacement in PHI_PATTERNS:
                        masked = pattern.sub(replacement, masked)
                    masked_args.append(masked)
                else:
                    masked_args.append(arg)
            record.args = tuple(masked_args)

        return True


def get_json_processors() -> list[Processor]:
    """
    Get processors for JSON log output.

    Returns:
        List of structlog processors for JSON formatting.
    """
    return [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.PositionalArgumentsFormatter(),
        add_timestamp,
        add_service_info,
        add_caller_info,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
        structlog.processors.format_exc_info,
        mask_phi,
        drop_color_message_key,
        structlog.processors.JSONRenderer(),
    ]


def get_console_processors() -> list[Processor]:
    """
    Get processors for console log output.

    Returns:
        List of structlog processors for console formatting.
    """
    return [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.PositionalArgumentsFormatter(),
        add_timestamp,
        structlog.processors.StackInfoRenderer(),
        structlog.processors.UnicodeDecoder(),
        structlog.processors.format_exc_info,
        mask_phi,
        structlog.dev.ConsoleRenderer(
            colors=True,
            exception_formatter=structlog.dev.plain_traceback,
        ),
    ]


def configure_logging() -> None:
    """
    Configure the logging system with structlog.

    Sets up both structlog and standard library logging with:
    - JSON or console output based on settings
    - PHI masking for HIPAA compliance
    - File and console handlers
    - Rotating file logs
    """
    settings = get_settings()

    # Determine log level
    log_level = getattr(logging, settings.logging.level.value)

    # Get processors based on format
    if settings.logging.format == LogFormat.JSON:
        processors = get_json_processors()
    else:
        processors = get_console_processors()

    # Configure structlog
    structlog.configure(
        processors=processors,
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Configure standard library logging
    root_logger = logging.getLogger()
    root_logger.setLevel(log_level)

    # Remove existing handlers
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    # Create formatters
    if settings.logging.format == LogFormat.JSON:
        formatter = structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=[
                structlog.contextvars.merge_contextvars,
                structlog.stdlib.add_log_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.ExtraAdder(),
                add_timestamp,
                add_service_info,
                structlog.processors.StackInfoRenderer(),
                structlog.processors.UnicodeDecoder(),
                structlog.processors.format_exc_info,
                mask_phi,
            ],
            processor=structlog.processors.JSONRenderer(),
        )
    else:
        formatter = structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=[
                structlog.contextvars.merge_contextvars,
                structlog.stdlib.add_log_level,
                structlog.stdlib.add_logger_name,
                structlog.stdlib.ExtraAdder(),
                structlog.processors.StackInfoRenderer(),
                structlog.processors.UnicodeDecoder(),
                structlog.processors.format_exc_info,
                mask_phi,
            ],
            processor=structlog.dev.ConsoleRenderer(colors=True),
        )

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    console_handler.addFilter(PHIFilter())
    root_logger.addHandler(console_handler)

    # File handler with rotation
    log_file = settings.logging.file_path
    log_file.parent.mkdir(parents=True, exist_ok=True)

    file_handler = logging.handlers.RotatingFileHandler(
        filename=str(log_file),
        maxBytes=settings.logging.file_max_size_mb * 1024 * 1024,
        backupCount=settings.logging.file_backup_count,
        encoding="utf-8",
    )
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    file_handler.addFilter(PHIFilter())
    root_logger.addHandler(file_handler)

    # Set levels for noisy third-party loggers
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("openai").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("asyncio").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """
    Get a configured structlog logger instance.

    Args:
        name: Logger name. If None, uses the calling module's name.

    Returns:
        Configured structlog BoundLogger instance.
    """
    return structlog.get_logger(name)


class AuditLogger:
    """
    HIPAA-compliant audit logger for tracking sensitive operations.

    Provides structured logging specifically for audit trail requirements,
    with mandatory fields for compliance tracking.
    """

    def __init__(self) -> None:
        """Initialize the audit logger."""
        self._logger = structlog.get_logger("audit")
        self._settings = get_settings()
        self._setup_audit_handler()

    def _setup_audit_handler(self) -> None:
        """Set up dedicated audit log file handler."""
        audit_log_path = self._settings.hipaa.audit_log_path / "audit.log"
        audit_log_path.parent.mkdir(parents=True, exist_ok=True)

        audit_handler = logging.handlers.RotatingFileHandler(
            filename=str(audit_log_path),
            maxBytes=100 * 1024 * 1024,  # 100 MB
            backupCount=10,
            encoding="utf-8",
        )

        formatter = structlog.stdlib.ProcessorFormatter(
            foreign_pre_chain=[
                structlog.contextvars.merge_contextvars,
                structlog.stdlib.add_log_level,
                add_timestamp,
                add_service_info,
                structlog.processors.format_exc_info,
            ],
            processor=structlog.processors.JSONRenderer(),
        )
        audit_handler.setFormatter(formatter)
        audit_handler.setLevel(logging.INFO)

        audit_logger = logging.getLogger("audit")
        audit_logger.addHandler(audit_handler)
        audit_logger.setLevel(logging.INFO)

    def log_access(
        self,
        user_id: str,
        resource_type: str,
        resource_id: str,
        action: str,
        success: bool,
        details: dict[str, Any] | None = None,
    ) -> None:
        """
        Log a resource access event.

        Args:
            user_id: Identifier of the user performing the action.
            resource_type: Type of resource being accessed.
            resource_id: Identifier of the resource.
            action: Action being performed (e.g., "read", "write", "delete").
            success: Whether the action succeeded.
            details: Additional details about the access.
        """
        self._logger.info(
            "resource_access",
            audit_type="access",
            user_id=user_id,
            resource_type=resource_type,
            resource_id=resource_id,
            action=action,
            success=success,
            details=details or {},
        )

    def log_phi_access(
        self,
        user_id: str,
        document_id: str,
        phi_fields: list[str],
        purpose: str,
    ) -> None:
        """
        Log PHI access for HIPAA compliance.

        Args:
            user_id: Identifier of the user accessing PHI.
            document_id: Identifier of the document containing PHI.
            phi_fields: List of PHI fields accessed.
            purpose: Business purpose for accessing PHI.
        """
        self._logger.info(
            "phi_access",
            audit_type="phi_access",
            user_id=user_id,
            document_id=document_id,
            phi_fields=phi_fields,
            purpose=purpose,
            hipaa_event=True,
        )

    def log_extraction(
        self,
        document_id: str,
        user_id: str,
        extraction_type: str,
        page_count: int,
        field_count: int,
        confidence_score: float,
        processing_time_ms: int,
    ) -> None:
        """
        Log a document extraction event.

        Args:
            document_id: Identifier of the extracted document.
            user_id: Identifier of the user who initiated extraction.
            extraction_type: Type of extraction (e.g., "superbill", "cms1500").
            page_count: Number of pages processed.
            field_count: Number of fields extracted.
            confidence_score: Overall confidence score.
            processing_time_ms: Processing time in milliseconds.
        """
        self._logger.info(
            "document_extraction",
            audit_type="extraction",
            document_id=document_id,
            user_id=user_id,
            extraction_type=extraction_type,
            page_count=page_count,
            field_count=field_count,
            confidence_score=confidence_score,
            processing_time_ms=processing_time_ms,
        )

    def log_security_event(
        self,
        event_type: str,
        severity: str,
        user_id: str | None,
        ip_address: str | None,
        details: dict[str, Any],
    ) -> None:
        """
        Log a security-related event.

        Args:
            event_type: Type of security event.
            severity: Severity level (e.g., "low", "medium", "high", "critical").
            user_id: Identifier of the user involved (if applicable).
            ip_address: IP address of the request (if applicable).
            details: Additional event details.
        """
        self._logger.warning(
            "security_event",
            audit_type="security",
            event_type=event_type,
            severity=severity,
            user_id=user_id,
            ip_address=ip_address,
            details=details,
            security_event=True,
        )


def get_audit_logger() -> AuditLogger:
    """
    Get the HIPAA-compliant audit logger instance.

    Returns:
        Configured AuditLogger instance.
    """
    return AuditLogger()
