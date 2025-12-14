"""
Prometheus Metrics Module for Document Extraction System.

Provides comprehensive metrics collection for monitoring system performance,
resource usage, and extraction accuracy for production observability.
"""

from __future__ import annotations

import functools
import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, TypeVar

from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Gauge,
    Histogram,
    Info,
    Summary,
    generate_latest,
)

import structlog

logger = structlog.get_logger(__name__)


class MetricNamespace(str, Enum):
    """Metric namespace prefixes."""

    EXTRACTION = "extraction"
    API = "api"
    VLM = "vlm"
    PIPELINE = "pipeline"
    VALIDATION = "validation"
    SECURITY = "security"
    SYSTEM = "system"


# Default histogram buckets for different metric types
DURATION_BUCKETS = (0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0)
SIZE_BUCKETS = (1024, 10240, 102400, 1048576, 10485760, 104857600)  # 1KB to 100MB
CONFIDENCE_BUCKETS = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0)
PAGE_BUCKETS = (1, 2, 5, 10, 20, 50, 100, 200)


@dataclass(slots=True)
class MetricLabels:
    """Common labels for metrics."""

    environment: str = "development"
    service: str = "doc-extraction"
    version: str = "2.0.0"
    instance: str = "default"


class MetricsRegistry:
    """
    Central registry for all Prometheus metrics.

    Provides a unified interface for metric creation and management.
    """

    _instance: MetricsRegistry | None = None

    def __init__(self, labels: MetricLabels | None = None) -> None:
        """
        Initialize metrics registry.

        Args:
            labels: Common labels for all metrics.
        """
        self._labels = labels or MetricLabels()
        self._common_labels = {
            "environment": self._labels.environment,
            "service": self._labels.service,
            "version": self._labels.version,
        }

        # Initialize all metrics
        self._init_system_metrics()
        self._init_api_metrics()
        self._init_extraction_metrics()
        self._init_vlm_metrics()
        self._init_validation_metrics()
        self._init_security_metrics()
        self._init_pipeline_metrics()

    @classmethod
    def get_instance(cls, labels: MetricLabels | None = None) -> MetricsRegistry:
        """Get or create singleton instance."""
        if cls._instance is None:
            cls._instance = cls(labels)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (for testing)."""
        cls._instance = None

    def _init_system_metrics(self) -> None:
        """Initialize system-level metrics."""
        # Info metric
        self.system_info = Info(
            "extraction_system",
            "Document extraction system information",
        )
        self.system_info.info({
            "version": self._labels.version,
            "service": self._labels.service,
            "environment": self._labels.environment,
        })

        # System uptime
        self.system_uptime = Gauge(
            "system_uptime_seconds",
            "System uptime in seconds",
        )
        self._start_time = time.time()

        # Resource gauges
        self.memory_usage_bytes = Gauge(
            "system_memory_usage_bytes",
            "Current memory usage in bytes",
            ["type"],  # heap, rss, etc.
        )

        self.cpu_usage_percent = Gauge(
            "system_cpu_usage_percent",
            "Current CPU usage percentage",
        )

        self.disk_usage_bytes = Gauge(
            "system_disk_usage_bytes",
            "Disk usage in bytes",
            ["path", "type"],  # type: used, free, total
        )

        # Active connections/sessions
        self.active_connections = Gauge(
            "system_active_connections",
            "Number of active connections",
            ["type"],  # http, websocket, database
        )

        self.active_tasks = Gauge(
            "system_active_tasks",
            "Number of active tasks",
            ["task_type"],
        )

    def _init_api_metrics(self) -> None:
        """Initialize API-related metrics."""
        # Request counter
        self.api_requests_total = Counter(
            "api_requests_total",
            "Total API requests",
            ["method", "endpoint", "status_code"],
        )

        # Request duration
        self.api_request_duration_seconds = Histogram(
            "api_request_duration_seconds",
            "API request duration in seconds",
            ["method", "endpoint"],
            buckets=DURATION_BUCKETS,
        )

        # Request size
        self.api_request_size_bytes = Histogram(
            "api_request_size_bytes",
            "API request body size in bytes",
            ["method", "endpoint"],
            buckets=SIZE_BUCKETS,
        )

        # Response size
        self.api_response_size_bytes = Histogram(
            "api_response_size_bytes",
            "API response body size in bytes",
            ["method", "endpoint"],
            buckets=SIZE_BUCKETS,
        )

        # Errors
        self.api_errors_total = Counter(
            "api_errors_total",
            "Total API errors",
            ["method", "endpoint", "error_type"],
        )

        # Rate limiting
        self.api_rate_limit_exceeded_total = Counter(
            "api_rate_limit_exceeded_total",
            "Total rate limit exceeded events",
            ["endpoint", "client_id"],
        )

        # Active requests
        self.api_requests_in_progress = Gauge(
            "api_requests_in_progress",
            "Number of API requests in progress",
            ["method", "endpoint"],
        )

    def _init_extraction_metrics(self) -> None:
        """Initialize extraction-related metrics."""
        # Documents processed
        self.documents_processed_total = Counter(
            "extraction_documents_total",
            "Total documents processed",
            ["doc_type", "status"],  # status: success, failure, partial
        )

        # Pages processed
        self.pages_processed_total = Counter(
            "extraction_pages_total",
            "Total pages processed",
            ["doc_type"],
        )

        # Processing duration
        self.extraction_duration_seconds = Histogram(
            "extraction_duration_seconds",
            "Document extraction duration in seconds",
            ["doc_type", "page_count_bucket"],
            buckets=DURATION_BUCKETS,
        )

        # Per-page duration
        self.extraction_page_duration_seconds = Histogram(
            "extraction_page_duration_seconds",
            "Per-page extraction duration in seconds",
            ["doc_type"],
            buckets=DURATION_BUCKETS,
        )

        # Fields extracted
        self.fields_extracted_total = Counter(
            "extraction_fields_total",
            "Total fields extracted",
            ["field_type", "doc_type"],
        )

        # Confidence scores
        self.extraction_confidence = Histogram(
            "extraction_confidence",
            "Extraction confidence scores",
            ["doc_type", "field_type"],
            buckets=CONFIDENCE_BUCKETS,
        )

        # Document size
        self.document_size_bytes = Histogram(
            "extraction_document_size_bytes",
            "Document size in bytes",
            ["doc_type"],
            buckets=SIZE_BUCKETS,
        )

        # Page count
        self.document_page_count = Histogram(
            "extraction_document_page_count",
            "Number of pages per document",
            ["doc_type"],
            buckets=PAGE_BUCKETS,
        )

        # Queue metrics
        self.extraction_queue_size = Gauge(
            "extraction_queue_size",
            "Number of documents in extraction queue",
            ["priority"],
        )

        self.extraction_queue_wait_seconds = Histogram(
            "extraction_queue_wait_seconds",
            "Time spent waiting in queue",
            buckets=DURATION_BUCKETS,
        )

    def _init_vlm_metrics(self) -> None:
        """Initialize VLM-related metrics."""
        # VLM calls
        self.vlm_calls_total = Counter(
            "vlm_calls_total",
            "Total VLM API calls",
            ["agent", "call_type"],  # agent: analyzer, extractor, validator
        )

        # VLM call duration
        self.vlm_call_duration_seconds = Histogram(
            "vlm_call_duration_seconds",
            "VLM call duration in seconds",
            ["agent", "call_type"],
            buckets=DURATION_BUCKETS,
        )

        # VLM errors
        self.vlm_errors_total = Counter(
            "vlm_errors_total",
            "Total VLM errors",
            ["agent", "error_type"],
        )

        # VLM retries
        self.vlm_retries_total = Counter(
            "vlm_retries_total",
            "Total VLM retry attempts",
            ["agent"],
        )

        # Token usage
        self.vlm_tokens_total = Counter(
            "vlm_tokens_total",
            "Total VLM tokens used",
            ["agent", "token_type"],  # token_type: prompt, completion
        )

        # VLM availability
        self.vlm_available = Gauge(
            "vlm_available",
            "VLM service availability (1=available, 0=unavailable)",
        )

        # VLM latency
        self.vlm_latency_seconds = Summary(
            "vlm_latency_seconds",
            "VLM response latency",
            ["agent"],
        )

    def _init_validation_metrics(self) -> None:
        """Initialize validation-related metrics."""
        # Validation results
        self.validation_results_total = Counter(
            "validation_results_total",
            "Total validation results",
            ["validation_type", "result"],  # result: pass, fail, warning
        )

        # Hallucination detection
        self.hallucinations_detected_total = Counter(
            "validation_hallucinations_total",
            "Total hallucinations detected",
            ["pattern_type"],
        )

        # Dual-pass results
        self.dual_pass_agreement_total = Counter(
            "validation_dual_pass_agreement_total",
            "Dual-pass extraction agreement results",
            ["result"],  # match, mismatch
        )

        # Medical code validation
        self.medical_code_validation_total = Counter(
            "validation_medical_code_total",
            "Medical code validation results",
            ["code_type", "result"],  # code_type: CPT, ICD10, NPI
        )

        # Human review
        self.human_review_required_total = Counter(
            "validation_human_review_total",
            "Documents requiring human review",
            ["reason"],
        )

        # Confidence thresholds
        self.confidence_threshold_results = Counter(
            "validation_confidence_threshold_total",
            "Confidence threshold check results",
            ["threshold_level", "result"],  # threshold_level: high, medium, low
        )

    def _init_security_metrics(self) -> None:
        """Initialize security-related metrics."""
        # Authentication
        self.auth_attempts_total = Counter(
            "security_auth_attempts_total",
            "Total authentication attempts",
            ["result", "method"],  # result: success, failure
        )

        # Authorization
        self.authz_checks_total = Counter(
            "security_authz_checks_total",
            "Total authorization checks",
            ["result", "permission"],
        )

        # PHI access
        self.phi_access_total = Counter(
            "security_phi_access_total",
            "Total PHI access events",
            ["action", "resource_type"],
        )

        # Encryption operations
        self.encryption_operations_total = Counter(
            "security_encryption_operations_total",
            "Total encryption operations",
            ["operation"],  # encrypt, decrypt
        )

        # Secure deletion
        self.secure_deletion_total = Counter(
            "security_secure_deletion_total",
            "Total secure deletion operations",
            ["method", "result"],
        )

        # Token operations
        self.token_operations_total = Counter(
            "security_token_operations_total",
            "Total token operations",
            ["operation"],  # create, validate, revoke, refresh
        )

        # Security events
        self.security_events_total = Counter(
            "security_events_total",
            "Total security events",
            ["event_type", "severity"],
        )

    def _init_pipeline_metrics(self) -> None:
        """Initialize pipeline-related metrics."""
        # Pipeline state transitions
        self.pipeline_state_transitions_total = Counter(
            "pipeline_state_transitions_total",
            "Total pipeline state transitions",
            ["from_state", "to_state"],
        )

        # Checkpoint operations
        self.checkpoint_operations_total = Counter(
            "pipeline_checkpoint_operations_total",
            "Total checkpoint operations",
            ["operation"],  # save, restore
        )

        # Pipeline errors
        self.pipeline_errors_total = Counter(
            "pipeline_errors_total",
            "Total pipeline errors",
            ["stage", "error_type"],
        )

        # Retry attempts
        self.pipeline_retries_total = Counter(
            "pipeline_retries_total",
            "Total pipeline retry attempts",
            ["stage"],
        )

        # Stage duration
        self.pipeline_stage_duration_seconds = Histogram(
            "pipeline_stage_duration_seconds",
            "Pipeline stage duration in seconds",
            ["stage"],
            buckets=DURATION_BUCKETS,
        )

        # Active pipelines
        self.active_pipelines = Gauge(
            "pipeline_active_count",
            "Number of active pipelines",
            ["stage"],
        )

    def update_uptime(self) -> None:
        """Update system uptime metric."""
        self.system_uptime.set(time.time() - self._start_time)

    def get_metrics(self) -> bytes:
        """
        Generate Prometheus metrics output.

        Returns:
            Prometheus metrics in exposition format.
        """
        self.update_uptime()
        return generate_latest()

    def get_content_type(self) -> str:
        """Get Prometheus content type header."""
        return CONTENT_TYPE_LATEST


class MetricsCollector:
    """
    High-level metrics collection interface.

    Provides convenient methods for recording metrics throughout
    the application.
    """

    def __init__(self, registry: MetricsRegistry | None = None) -> None:
        """
        Initialize metrics collector.

        Args:
            registry: Metrics registry to use.
        """
        self._registry = registry or MetricsRegistry.get_instance()

    # API Metrics Methods

    def record_api_request(
        self,
        method: str,
        endpoint: str,
        status_code: int,
        duration: float,
        request_size: int = 0,
        response_size: int = 0,
    ) -> None:
        """Record an API request."""
        self._registry.api_requests_total.labels(
            method=method,
            endpoint=endpoint,
            status_code=str(status_code),
        ).inc()

        self._registry.api_request_duration_seconds.labels(
            method=method,
            endpoint=endpoint,
        ).observe(duration)

        if request_size > 0:
            self._registry.api_request_size_bytes.labels(
                method=method,
                endpoint=endpoint,
            ).observe(request_size)

        if response_size > 0:
            self._registry.api_response_size_bytes.labels(
                method=method,
                endpoint=endpoint,
            ).observe(response_size)

        if status_code >= 400:
            error_type = "client_error" if status_code < 500 else "server_error"
            self._registry.api_errors_total.labels(
                method=method,
                endpoint=endpoint,
                error_type=error_type,
            ).inc()

    @contextmanager
    def track_request(self, method: str, endpoint: str):
        """Context manager to track API request duration."""
        self._registry.api_requests_in_progress.labels(
            method=method,
            endpoint=endpoint,
        ).inc()

        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self._registry.api_requests_in_progress.labels(
                method=method,
                endpoint=endpoint,
            ).dec()

    # Extraction Metrics Methods

    def record_document_processed(
        self,
        doc_type: str,
        status: str,
        page_count: int,
        duration: float,
        file_size: int = 0,
    ) -> None:
        """Record a document processing result."""
        self._registry.documents_processed_total.labels(
            doc_type=doc_type,
            status=status,
        ).inc()

        self._registry.pages_processed_total.labels(
            doc_type=doc_type,
        ).inc(page_count)

        # Bucket page count
        page_bucket = "1" if page_count == 1 else "2-5" if page_count <= 5 else "6-10" if page_count <= 10 else "10+"

        self._registry.extraction_duration_seconds.labels(
            doc_type=doc_type,
            page_count_bucket=page_bucket,
        ).observe(duration)

        self._registry.document_page_count.labels(
            doc_type=doc_type,
        ).observe(page_count)

        if file_size > 0:
            self._registry.document_size_bytes.labels(
                doc_type=doc_type,
            ).observe(file_size)

    def record_field_extraction(
        self,
        doc_type: str,
        field_type: str,
        confidence: float,
    ) -> None:
        """Record a field extraction result."""
        self._registry.fields_extracted_total.labels(
            field_type=field_type,
            doc_type=doc_type,
        ).inc()

        self._registry.extraction_confidence.labels(
            doc_type=doc_type,
            field_type=field_type,
        ).observe(confidence)

    # VLM Metrics Methods

    def record_vlm_call(
        self,
        agent: str,
        call_type: str,
        duration: float,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        success: bool = True,
        error_type: str | None = None,
    ) -> None:
        """Record a VLM API call."""
        self._registry.vlm_calls_total.labels(
            agent=agent,
            call_type=call_type,
        ).inc()

        self._registry.vlm_call_duration_seconds.labels(
            agent=agent,
            call_type=call_type,
        ).observe(duration)

        self._registry.vlm_latency_seconds.labels(
            agent=agent,
        ).observe(duration)

        if prompt_tokens > 0:
            self._registry.vlm_tokens_total.labels(
                agent=agent,
                token_type="prompt",
            ).inc(prompt_tokens)

        if completion_tokens > 0:
            self._registry.vlm_tokens_total.labels(
                agent=agent,
                token_type="completion",
            ).inc(completion_tokens)

        if not success and error_type:
            self._registry.vlm_errors_total.labels(
                agent=agent,
                error_type=error_type,
            ).inc()

    def record_vlm_retry(self, agent: str) -> None:
        """Record a VLM retry attempt."""
        self._registry.vlm_retries_total.labels(agent=agent).inc()

    def set_vlm_availability(self, available: bool) -> None:
        """Set VLM availability status."""
        self._registry.vlm_available.set(1 if available else 0)

    # Validation Metrics Methods

    def record_validation_result(
        self,
        validation_type: str,
        result: str,
    ) -> None:
        """Record a validation result."""
        self._registry.validation_results_total.labels(
            validation_type=validation_type,
            result=result,
        ).inc()

    def record_hallucination_detected(self, pattern_type: str) -> None:
        """Record a detected hallucination."""
        self._registry.hallucinations_detected_total.labels(
            pattern_type=pattern_type,
        ).inc()

    def record_dual_pass_result(self, match: bool) -> None:
        """Record dual-pass comparison result."""
        self._registry.dual_pass_agreement_total.labels(
            result="match" if match else "mismatch",
        ).inc()

    def record_medical_code_validation(
        self,
        code_type: str,
        valid: bool,
    ) -> None:
        """Record medical code validation result."""
        self._registry.medical_code_validation_total.labels(
            code_type=code_type,
            result="valid" if valid else "invalid",
        ).inc()

    def record_human_review_required(self, reason: str) -> None:
        """Record document requiring human review."""
        self._registry.human_review_required_total.labels(
            reason=reason,
        ).inc()

    # Security Metrics Methods

    def record_auth_attempt(
        self,
        success: bool,
        method: str = "password",
    ) -> None:
        """Record an authentication attempt."""
        self._registry.auth_attempts_total.labels(
            result="success" if success else "failure",
            method=method,
        ).inc()

    def record_authz_check(
        self,
        granted: bool,
        permission: str,
    ) -> None:
        """Record an authorization check."""
        self._registry.authz_checks_total.labels(
            result="granted" if granted else "denied",
            permission=permission,
        ).inc()

    def record_phi_access(
        self,
        action: str,
        resource_type: str,
    ) -> None:
        """Record PHI access."""
        self._registry.phi_access_total.labels(
            action=action,
            resource_type=resource_type,
        ).inc()

    def record_encryption_operation(self, operation: str) -> None:
        """Record an encryption operation."""
        self._registry.encryption_operations_total.labels(
            operation=operation,
        ).inc()

    def record_secure_deletion(
        self,
        method: str,
        success: bool,
    ) -> None:
        """Record a secure deletion operation."""
        self._registry.secure_deletion_total.labels(
            method=method,
            result="success" if success else "failure",
        ).inc()

    def record_security_event(
        self,
        event_type: str,
        severity: str,
    ) -> None:
        """Record a security event."""
        self._registry.security_events_total.labels(
            event_type=event_type,
            severity=severity,
        ).inc()

    # Pipeline Metrics Methods

    def record_state_transition(
        self,
        from_state: str,
        to_state: str,
    ) -> None:
        """Record a pipeline state transition."""
        self._registry.pipeline_state_transitions_total.labels(
            from_state=from_state,
            to_state=to_state,
        ).inc()

    def record_checkpoint_operation(self, operation: str) -> None:
        """Record a checkpoint operation."""
        self._registry.checkpoint_operations_total.labels(
            operation=operation,
        ).inc()

    def record_pipeline_error(
        self,
        stage: str,
        error_type: str,
    ) -> None:
        """Record a pipeline error."""
        self._registry.pipeline_errors_total.labels(
            stage=stage,
            error_type=error_type,
        ).inc()

    def record_pipeline_retry(self, stage: str) -> None:
        """Record a pipeline retry."""
        self._registry.pipeline_retries_total.labels(stage=stage).inc()

    @contextmanager
    def track_pipeline_stage(self, stage: str):
        """Context manager to track pipeline stage duration."""
        self._registry.active_pipelines.labels(stage=stage).inc()
        start_time = time.time()
        try:
            yield
        finally:
            duration = time.time() - start_time
            self._registry.pipeline_stage_duration_seconds.labels(
                stage=stage,
            ).observe(duration)
            self._registry.active_pipelines.labels(stage=stage).dec()

    # System Metrics Methods

    def update_memory_usage(
        self,
        heap_bytes: int,
        rss_bytes: int,
    ) -> None:
        """Update memory usage metrics."""
        self._registry.memory_usage_bytes.labels(type="heap").set(heap_bytes)
        self._registry.memory_usage_bytes.labels(type="rss").set(rss_bytes)

    def update_cpu_usage(self, percent: float) -> None:
        """Update CPU usage metric."""
        self._registry.cpu_usage_percent.set(percent)

    def update_disk_usage(
        self,
        path: str,
        used: int,
        free: int,
        total: int,
    ) -> None:
        """Update disk usage metrics."""
        self._registry.disk_usage_bytes.labels(path=path, type="used").set(used)
        self._registry.disk_usage_bytes.labels(path=path, type="free").set(free)
        self._registry.disk_usage_bytes.labels(path=path, type="total").set(total)

    def update_active_connections(self, conn_type: str, count: int) -> None:
        """Update active connections count."""
        self._registry.active_connections.labels(type=conn_type).set(count)

    def update_active_tasks(self, task_type: str, count: int) -> None:
        """Update active tasks count."""
        self._registry.active_tasks.labels(task_type=task_type).set(count)

    def update_queue_size(self, priority: str, size: int) -> None:
        """Update extraction queue size."""
        self._registry.extraction_queue_size.labels(priority=priority).set(size)


# Decorator for automatic metric collection
F = TypeVar("F", bound=Callable[..., Any])


# Decorator-specific Prometheus metrics (lazily initialized)
_decorator_duration_histogram: Histogram | None = None
_decorator_calls_counter: Counter | None = None
_decorator_errors_counter: Counter | None = None


def _get_decorator_duration_histogram() -> Histogram:
    """Get or create the decorator duration histogram."""
    global _decorator_duration_histogram
    if _decorator_duration_histogram is None:
        _decorator_duration_histogram = Histogram(
            "function_duration_seconds",
            "Function execution duration in seconds (via @track_duration decorator)",
            ["metric_name", "function_name", "module"],
            buckets=DURATION_BUCKETS,
        )
    return _decorator_duration_histogram


def _get_decorator_calls_counter() -> Counter:
    """Get or create the decorator calls counter."""
    global _decorator_calls_counter
    if _decorator_calls_counter is None:
        _decorator_calls_counter = Counter(
            "function_calls_total",
            "Total function calls (via @count_calls decorator)",
            ["metric_name", "function_name", "module", "status"],
        )
    return _decorator_calls_counter


def _get_decorator_errors_counter() -> Counter:
    """Get or create the decorator errors counter."""
    global _decorator_errors_counter
    if _decorator_errors_counter is None:
        _decorator_errors_counter = Counter(
            "function_errors_total",
            "Total function errors (via decorators)",
            ["metric_name", "function_name", "module", "error_type"],
        )
    return _decorator_errors_counter


def track_duration(
    metric_name: str,
    labels: dict[str, str] | None = None,
) -> Callable[[F], F]:
    """
    Decorator to track function duration.

    Records duration to Prometheus histogram and logs debug info.

    Args:
        metric_name: Name of the duration metric.
        labels: Static labels to apply.

    Returns:
        Decorated function.

    Example:
        @track_duration("extraction_validate")
        def validate_document(doc_id: str) -> bool:
            ...
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            histogram = _get_decorator_duration_histogram()
            error_counter = _get_decorator_errors_counter()
            module_name = func.__module__ or "unknown"

            start_time = time.time()
            error_occurred = False
            error_type = ""

            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_occurred = True
                error_type = type(e).__name__
                raise
            finally:
                duration = time.time() - start_time

                # Record to Prometheus histogram
                histogram.labels(
                    metric_name=metric_name,
                    function_name=func.__name__,
                    module=module_name,
                ).observe(duration)

                # Record error if occurred
                if error_occurred:
                    error_counter.labels(
                        metric_name=metric_name,
                        function_name=func.__name__,
                        module=module_name,
                        error_type=error_type,
                    ).inc()

                # Also log for debugging
                logger.debug(
                    "function_duration",
                    metric=metric_name,
                    function=func.__name__,
                    module=module_name,
                    duration_seconds=duration,
                    error=error_occurred,
                    **(labels or {}),
                )

        return wrapper  # type: ignore

    return decorator


def count_calls(
    metric_name: str,
    labels: dict[str, str] | None = None,
    track_status: bool = True,
) -> Callable[[F], F]:
    """
    Decorator to count function calls.

    Records call count to Prometheus counter and logs debug info.

    Args:
        metric_name: Name of the counter metric.
        labels: Static labels to apply.
        track_status: Whether to track success/error status.

    Returns:
        Decorated function.

    Example:
        @count_calls("api_document_upload")
        def upload_document(file: UploadFile) -> str:
            ...
    """
    def decorator(func: F) -> F:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            counter = _get_decorator_calls_counter()
            module_name = func.__module__ or "unknown"

            try:
                result = func(*args, **kwargs)
                status = "success"
                return result
            except Exception:
                status = "error"
                raise
            finally:
                # Record to Prometheus counter
                counter.labels(
                    metric_name=metric_name,
                    function_name=func.__name__,
                    module=module_name,
                    status=status if track_status else "counted",
                ).inc()

                # Also log for debugging
                logger.debug(
                    "function_called",
                    metric=metric_name,
                    function=func.__name__,
                    module=module_name,
                    status=status if track_status else "counted",
                    **(labels or {}),
                )

        return wrapper  # type: ignore

    return decorator


def track_duration_and_count(
    metric_name: str,
    labels: dict[str, str] | None = None,
) -> Callable[[F], F]:
    """
    Combined decorator to track both duration and call count.

    Convenience decorator that applies both @track_duration and @count_calls.

    Args:
        metric_name: Name of the metric.
        labels: Static labels to apply.

    Returns:
        Decorated function.
    """
    def decorator(func: F) -> F:
        # Apply both decorators
        decorated = track_duration(metric_name, labels)(func)
        decorated = count_calls(metric_name, labels)(decorated)
        return decorated  # type: ignore

    return decorator
