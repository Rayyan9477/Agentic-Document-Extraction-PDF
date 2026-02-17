"""
Celery task definitions for document processing.

Provides async task wrappers for the document extraction pipeline
with comprehensive error handling and status tracking.
"""

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

from celery import Task, current_task
from celery.exceptions import MaxRetriesExceededError, SoftTimeLimitExceeded

from src.config import get_logger
from src.queue.celery_app import celery_app


logger = get_logger(__name__)


class TaskStatus(str, Enum):
    """Task execution status."""

    PENDING = "pending"
    STARTED = "started"
    PROCESSING = "processing"
    VALIDATING = "validating"
    EXPORTING = "exporting"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


@dataclass(slots=True)
class TaskResult:
    """
    Result of a document processing task.

    Attributes:
        task_id: Celery task ID.
        processing_id: Document processing ID.
        status: Task execution status.
        pdf_path: Path to processed PDF.
        output_path: Path to output file(s).
        started_at: Task start timestamp.
        completed_at: Task completion timestamp.
        duration_ms: Processing duration in milliseconds.
        field_count: Number of extracted fields.
        overall_confidence: Overall extraction confidence.
        requires_human_review: Whether human review is needed.
        human_review_reason: Reason for human review.
        errors: List of errors encountered.
        warnings: List of warnings.
        retry_count: Number of retries.
    """

    task_id: str
    processing_id: str
    status: TaskStatus
    pdf_path: str = ""
    output_path: str = ""
    started_at: str = ""
    completed_at: str = ""
    duration_ms: int = 0
    field_count: int = 0
    overall_confidence: float = 0.0
    requires_human_review: bool = False
    human_review_reason: str = ""
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    retry_count: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "task_id": self.task_id,
            "processing_id": self.processing_id,
            "status": self.status.value,
            "pdf_path": self.pdf_path,
            "output_path": self.output_path,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "duration_ms": self.duration_ms,
            "field_count": self.field_count,
            "overall_confidence": self.overall_confidence,
            "requires_human_review": self.requires_human_review,
            "human_review_reason": self.human_review_reason,
            "errors": self.errors,
            "warnings": self.warnings,
            "retry_count": self.retry_count,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "TaskResult":
        """Create from dictionary."""
        status = data.get("status", "pending")
        if isinstance(status, str):
            status = TaskStatus(status)
        return cls(
            task_id=data.get("task_id", ""),
            processing_id=data.get("processing_id", ""),
            status=status,
            pdf_path=data.get("pdf_path", ""),
            output_path=data.get("output_path", ""),
            started_at=data.get("started_at", ""),
            completed_at=data.get("completed_at", ""),
            duration_ms=data.get("duration_ms", 0),
            field_count=data.get("field_count", 0),
            overall_confidence=data.get("overall_confidence", 0.0),
            requires_human_review=data.get("requires_human_review", False),
            human_review_reason=data.get("human_review_reason", ""),
            errors=data.get("errors", []),
            warnings=data.get("warnings", []),
            retry_count=data.get("retry_count", 0),
        )


class DocumentProcessingTask(Task):
    """Base task class with common processing functionality."""

    abstract = True
    autoretry_for = (ConnectionError, TimeoutError)
    retry_backoff = True
    retry_backoff_max = 600
    retry_jitter = True
    max_retries = 3

    def on_failure(
        self,
        exc: Exception,
        task_id: str,
        args: tuple,
        kwargs: dict,
        einfo: Any,
    ) -> None:
        """Handle task failure."""
        logger.error(
            "task_failed",
            task_id=task_id,
            exception=str(exc),
            args=args,
        )

    def on_retry(
        self,
        exc: Exception,
        task_id: str,
        args: tuple,
        kwargs: dict,
        einfo: Any,
    ) -> None:
        """Handle task retry."""
        logger.warning(
            "task_retry",
            task_id=task_id,
            exception=str(exc),
            retry_count=self.request.retries,
        )

    def on_success(
        self,
        retval: Any,
        task_id: str,
        args: tuple,
        kwargs: dict,
    ) -> None:
        """Handle task success."""
        logger.info(
            "task_success",
            task_id=task_id,
        )


def _update_task_state(state: str, meta: dict[str, Any]) -> None:
    """Update current task state with metadata."""
    if current_task:
        current_task.update_state(state=state, meta=meta)


def _store_pipeline_result(
    processing_id: str,
    pipeline_result: dict[str, Any],
) -> None:
    """
    Store pipeline result for later retrieval.

    This enables preview and export operations on completed results.
    Storage failures are logged but do not affect task completion.

    Args:
        processing_id: Document processing ID.
        pipeline_result: Full pipeline extraction result.
    """
    try:
        from src.storage import save_result

        # Store the result (excludes large binary data like page_images)
        storable_result = {
            k: v
            for k, v in pipeline_result.items()
            if k != "page_images"  # Exclude binary image data
        }

        save_result(processing_id, storable_result)

        logger.debug(
            "pipeline_result_stored",
            processing_id=processing_id,
        )

    except Exception as e:
        # Storage failures should not affect task completion
        logger.error(
            "pipeline_result_storage_error",
            processing_id=processing_id,
            error=str(e),
            error_type=type(e).__name__,
        )


def _send_completion_webhook(
    callback_url: str,
    task_id: str,
    processing_id: str,
    status: str,
    result_data: dict[str, Any],
    error: str | None = None,
) -> None:
    """
    Send webhook notification for task completion.

    This function is designed to be non-blocking and failure-tolerant.
    Webhook delivery failures are logged but do not affect task results.

    Args:
        callback_url: URL to send webhook to.
        task_id: Celery task ID.
        processing_id: Document processing ID.
        status: Task status (completed/failed).
        result_data: Full task result data.
        error: Error message if failed.
    """
    try:
        from src.queue.webhook import (
            WebhookEventType,
            send_webhook_notification,
        )

        # Determine event type based on status
        event_type = (
            WebhookEventType.PROCESSING_COMPLETED
            if status == "completed"
            else WebhookEventType.PROCESSING_FAILED
        )

        # Build webhook data payload
        data = {
            "status": status,
            "field_count": result_data.get("field_count", 0),
            "overall_confidence": result_data.get("overall_confidence", 0.0),
            "requires_human_review": result_data.get("requires_human_review", False),
            "duration_ms": result_data.get("duration_ms", 0),
            "output_path": result_data.get("output_path", ""),
        }

        if error:
            data["error"] = error

        if result_data.get("warnings"):
            data["warnings"] = result_data["warnings"]

        # Send webhook (synchronously since we're in a Celery task)
        delivery_result = send_webhook_notification(
            callback_url=callback_url,
            event_type=event_type,
            processing_id=processing_id,
            task_id=task_id,
            data=data,
        )

        logger.info(
            "webhook_delivery_result",
            task_id=task_id,
            processing_id=processing_id,
            callback_url=callback_url,
            delivery_status=delivery_result.status.value,
            attempts=delivery_result.attempts,
        )

        # Fan out to all registered webhook subscriptions
        _fan_out_to_subscriptions(
            event_type=event_type,
            processing_id=processing_id,
            task_id=task_id,
            data=data,
        )

    except Exception as e:
        # Webhook failures should not affect task completion
        logger.error(
            "webhook_send_error",
            task_id=task_id,
            processing_id=processing_id,
            callback_url=callback_url,
            error=str(e),
            error_type=type(e).__name__,
        )


def _fan_out_to_subscriptions(
    event_type: Any,
    processing_id: str,
    task_id: str,
    data: dict[str, Any],
) -> None:
    """
    Fan out a webhook event to all registered subscriptions.

    Uses the WebhookStore to deliver to all matching active subscriptions.
    Failures are logged but never propagated.
    """
    try:
        from src.config import get_settings

        settings = get_settings()
        store_path = settings.webhook.store_path

        if not store_path.exists():
            return  # No subscriptions registered yet

        from src.queue.webhook_store import WebhookStore

        store = WebhookStore(persist_path=store_path)
        fan_out_result = store.fan_out(
            event_type=event_type,
            processing_id=processing_id,
            task_id=task_id,
            data=data,
        )

        if fan_out_result.total_subscriptions > 0:
            logger.info(
                "webhook_fan_out_complete",
                processing_id=processing_id,
                total=fan_out_result.total_subscriptions,
                delivered=fan_out_result.delivered,
                failed=fan_out_result.failed,
            )

    except Exception as e:
        logger.warning(
            "webhook_fan_out_error",
            processing_id=processing_id,
            error=str(e),
        )


@celery_app.task(
    bind=True,
    base=DocumentProcessingTask,
    name="src.queue.tasks.process_document_task",
    queue="document_processing",
)
def process_document_task(
    self: Task,
    pdf_path: str,
    output_dir: str | None = None,
    schema_name: str | None = None,
    export_format: str = "json",
    mask_phi: bool = False,
    priority: str = "normal",
    callback_url: str | None = None,
) -> dict[str, Any]:
    """
    Process a single document asynchronously.

    Args:
        self: Task instance (bound).
        pdf_path: Path to PDF file.
        output_dir: Output directory for results.
        schema_name: Schema to use for extraction.
        export_format: Export format (json/excel/both).
        mask_phi: Whether to mask PHI fields.
        priority: Processing priority (low/normal/high).
        callback_url: URL to call on completion/failure (webhook).

    Returns:
        TaskResult as dictionary.

    Raises:
        FileNotFoundError: If PDF file not found.
        ValueError: If invalid parameters.
    """
    task_id = self.request.id or "unknown"
    started_at = datetime.now(UTC)

    logger.info(
        "document_task_start",
        task_id=task_id,
        pdf_path=pdf_path,
        schema_name=schema_name,
    )

    result = TaskResult(
        task_id=task_id,
        processing_id="",
        status=TaskStatus.STARTED,
        pdf_path=pdf_path,
        started_at=started_at.isoformat(),
        retry_count=self.request.retries or 0,
    )

    try:
        # Validate input file
        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        if not pdf_file.suffix.lower() == ".pdf":
            raise ValueError(f"Invalid file type: {pdf_file.suffix}")

        _update_task_state("PROCESSING", {"status": TaskStatus.PROCESSING.value})

        # Import pipeline here to avoid circular imports
        from src.pipeline.runner import PipelineRunner

        # Run the extraction pipeline
        runner = PipelineRunner(enable_checkpointing=False)
        pipeline_result = runner.extract_from_pdf(
            pdf_path=pdf_path,
            custom_schema=None,
        )

        result.processing_id = pipeline_result.get("processing_id", "")
        result.status = TaskStatus.VALIDATING

        _update_task_state("VALIDATING", {"status": TaskStatus.VALIDATING.value})

        # Handle export
        result.status = TaskStatus.EXPORTING
        _update_task_state("EXPORTING", {"status": TaskStatus.EXPORTING.value})

        output_path = ""
        if output_dir:
            output_base = Path(output_dir) / result.processing_id

            if export_format in ("json", "both"):
                from src.export import ExportFormat, export_to_json

                json_path = output_base.with_suffix(".json")
                export_to_json(
                    pipeline_result,
                    output_path=json_path,
                    format=ExportFormat.DETAILED,
                    include_metadata=True,
                    include_confidence=True,
                )
                output_path = str(json_path)

            if export_format in ("excel", "both"):
                from src.export import export_to_excel

                excel_path = output_base.with_suffix(".xlsx")
                export_to_excel(
                    pipeline_result,
                    output_path=excel_path,
                    mask_phi=mask_phi,
                )
                if export_format == "excel":
                    output_path = str(excel_path)
                else:
                    output_path = f"{json_path}; {excel_path}"

        # Build final result
        completed_at = datetime.now(UTC)
        duration_ms = int((completed_at - started_at).total_seconds() * 1000)

        result.status = TaskStatus.COMPLETED
        result.output_path = output_path
        result.completed_at = completed_at.isoformat()
        result.duration_ms = duration_ms
        result.field_count = len(pipeline_result.get("merged_extraction", {}))
        result.overall_confidence = pipeline_result.get("overall_confidence", 0.0)
        result.requires_human_review = pipeline_result.get("requires_human_review", False)
        result.human_review_reason = pipeline_result.get("human_review_reason", "")
        result.warnings = pipeline_result.get("warnings", [])

        # Store result for later retrieval (preview, export)
        _store_pipeline_result(result.processing_id, pipeline_result)

        logger.info(
            "document_task_complete",
            task_id=task_id,
            processing_id=result.processing_id,
            duration_ms=duration_ms,
            field_count=result.field_count,
        )

        # Send webhook notification on success
        if callback_url:
            _send_completion_webhook(
                callback_url=callback_url,
                task_id=task_id,
                processing_id=result.processing_id,
                status="completed",
                result_data=result.to_dict(),
            )

        return result.to_dict()

    except SoftTimeLimitExceeded:
        result.status = TaskStatus.FAILED
        result.errors = ["Task exceeded time limit"]
        result.completed_at = datetime.now(UTC).isoformat()
        logger.error("task_timeout", task_id=task_id)

        # Send webhook notification on timeout failure
        if callback_url:
            _send_completion_webhook(
                callback_url=callback_url,
                task_id=task_id,
                processing_id=result.processing_id,
                status="failed",
                result_data=result.to_dict(),
                error="Task exceeded time limit",
            )

        return result.to_dict()

    except MaxRetriesExceededError:
        result.status = TaskStatus.FAILED
        result.errors = ["Maximum retries exceeded"]
        result.completed_at = datetime.now(UTC).isoformat()
        logger.error("task_max_retries", task_id=task_id)

        # Send webhook notification on max retries
        if callback_url:
            _send_completion_webhook(
                callback_url=callback_url,
                task_id=task_id,
                processing_id=result.processing_id,
                status="failed",
                result_data=result.to_dict(),
                error="Maximum retries exceeded",
            )

        return result.to_dict()

    except FileNotFoundError as e:
        result.status = TaskStatus.FAILED
        result.errors = [str(e)]
        result.completed_at = datetime.now(UTC).isoformat()
        logger.error("task_file_not_found", task_id=task_id, error=str(e))

        # Send webhook notification on file not found
        if callback_url:
            _send_completion_webhook(
                callback_url=callback_url,
                task_id=task_id,
                processing_id=result.processing_id,
                status="failed",
                result_data=result.to_dict(),
                error=str(e),
            )

        return result.to_dict()

    except Exception as e:
        # Attempt retry for transient errors
        if self.request.retries < self.max_retries:
            result.status = TaskStatus.RETRYING
            logger.warning(
                "task_retrying",
                task_id=task_id,
                error=str(e),
                retry_count=self.request.retries,
            )
            raise self.retry(exc=e, countdown=60 * (2**self.request.retries))

        result.status = TaskStatus.FAILED
        result.errors = [str(e)]
        result.completed_at = datetime.now(UTC).isoformat()
        logger.error("task_error", task_id=task_id, error=str(e))

        # Send webhook notification on general failure
        if callback_url:
            _send_completion_webhook(
                callback_url=callback_url,
                task_id=task_id,
                processing_id=result.processing_id,
                status="failed",
                result_data=result.to_dict(),
                error=str(e),
            )

        return result.to_dict()


# Default timeout per document for result collection (seconds)
DEFAULT_RESULT_TIMEOUT_PER_DOC = 30
# Minimum timeout for batch result collection
MIN_BATCH_RESULT_TIMEOUT = 60
# Maximum timeout for batch result collection
MAX_BATCH_RESULT_TIMEOUT = 3600  # 1 hour


@celery_app.task(
    bind=True,
    base=DocumentProcessingTask,
    name="src.queue.tasks.batch_process_task",
    queue="batch_processing",
)
def batch_process_task(
    self: Task,
    pdf_paths: list[str],
    output_dir: str,
    schema_name: str | None = None,
    export_format: str = "json",
    mask_phi: bool = False,
    stop_on_error: bool = False,
    result_timeout: int | None = None,
) -> dict[str, Any]:
    """
    Process multiple documents in batch.

    Args:
        self: Task instance (bound).
        pdf_paths: List of PDF file paths.
        output_dir: Output directory for results.
        schema_name: Schema to use for extraction.
        export_format: Export format (json/excel/both).
        mask_phi: Whether to mask PHI fields.
        stop_on_error: Stop processing on first error.
        result_timeout: Timeout in seconds for collecting each result.
                        If not provided, calculated based on batch size
                        (30 seconds per document, min 60s, max 3600s).

    Returns:
        Batch processing result with individual task results.
    """
    task_id = self.request.id or "unknown"
    started_at = datetime.now(UTC)

    logger.info(
        "batch_task_start",
        task_id=task_id,
        document_count=len(pdf_paths),
    )

    results: list[dict[str, Any]] = []
    successful = 0
    failed = 0

    # Use Celery group for parallel processing (non-blocking)
    from celery import group

    # Create task signatures for all documents
    task_signatures = [
        process_document_task.s(
            pdf_path,
            output_dir=output_dir,
            schema_name=schema_name,
            export_format=export_format,
            mask_phi=mask_phi,
        )
        for pdf_path in pdf_paths
    ]

    # Execute tasks in parallel
    job = group(task_signatures)
    async_results = job.apply_async()

    # Wait for all tasks to complete with progress updates
    total_tasks = len(pdf_paths)
    while not async_results.ready():
        completed_count = sum(1 for r in async_results.results if r.ready())
        _update_task_state(
            "PROCESSING",
            {
                "current": completed_count,
                "total": total_tasks,
                "status": "processing_batch",
            },
        )
        import time

        time.sleep(1)  # Check every second

    # Calculate result collection timeout if not provided
    if result_timeout is None:
        # Scale timeout based on batch size: 30 seconds per document
        calculated_timeout = len(pdf_paths) * DEFAULT_RESULT_TIMEOUT_PER_DOC
        # Apply min/max bounds
        result_timeout = max(
            MIN_BATCH_RESULT_TIMEOUT, min(calculated_timeout, MAX_BATCH_RESULT_TIMEOUT)
        )

    logger.debug(
        "batch_result_collection_start",
        task_id=task_id,
        result_timeout=result_timeout,
        document_count=len(pdf_paths),
    )

    # Collect results
    for idx, (pdf_path, async_result) in enumerate(
        zip(pdf_paths, async_results.results, strict=False)
    ):
        try:
            if async_result.successful():
                doc_result = async_result.get(timeout=result_timeout)
                results.append(doc_result)

                if doc_result.get("status") == TaskStatus.COMPLETED.value:
                    successful += 1
                else:
                    failed += 1
            else:
                failed += 1
                results.append(
                    {
                        "pdf_path": pdf_path,
                        "status": TaskStatus.FAILED.value,
                        "errors": [
                            str(async_result.result) if async_result.result else "Unknown error"
                        ],
                    }
                )

        except Exception as e:
            failed += 1
            results.append(
                {
                    "pdf_path": pdf_path,
                    "status": TaskStatus.FAILED.value,
                    "errors": [str(e)],
                }
            )

    completed_at = datetime.now(UTC)
    duration_ms = int((completed_at - started_at).total_seconds() * 1000)

    batch_result = {
        "task_id": task_id,
        "status": TaskStatus.COMPLETED.value if failed == 0 else TaskStatus.FAILED.value,
        "started_at": started_at.isoformat(),
        "completed_at": completed_at.isoformat(),
        "duration_ms": duration_ms,
        "total_documents": len(pdf_paths),
        "successful": successful,
        "failed": failed,
        "results": results,
    }

    logger.info(
        "batch_task_complete",
        task_id=task_id,
        successful=successful,
        failed=failed,
        duration_ms=duration_ms,
    )

    return batch_result


@celery_app.task(
    bind=True,
    base=DocumentProcessingTask,
    name="src.queue.tasks.reprocess_failed_task",
    queue="reprocessing",
)
def reprocess_failed_task(
    self: Task,
    original_task_id: str,
    pdf_path: str,
    output_dir: str | None = None,
    schema_name: str | None = None,
    export_format: str = "json",
    mask_phi: bool = False,
) -> dict[str, Any]:
    """
    Reprocess a previously failed document.

    This task performs the reprocessing inline rather than delegating to another
    task to avoid blocking worker threads. The processing logic is identical to
    process_document_task but with reprocessing metadata.

    Args:
        self: Task instance (bound).
        original_task_id: Original failed task ID.
        pdf_path: Path to PDF file.
        output_dir: Output directory for results.
        schema_name: Schema to use for extraction.
        export_format: Export format (json/excel/both).
        mask_phi: Whether to mask PHI fields.

    Returns:
        TaskResult as dictionary with reprocessing metadata.
    """
    task_id = self.request.id or "unknown"
    started_at = datetime.now(UTC)

    logger.info(
        "reprocess_task_start",
        task_id=task_id,
        original_task_id=original_task_id,
        pdf_path=pdf_path,
    )

    result = TaskResult(
        task_id=task_id,
        processing_id="",
        status=TaskStatus.STARTED,
        pdf_path=pdf_path,
        started_at=started_at.isoformat(),
        retry_count=self.request.retries or 0,
    )

    try:
        # Validate input file
        pdf_file = Path(pdf_path)
        if not pdf_file.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        if not pdf_file.suffix.lower() == ".pdf":
            raise ValueError(f"Invalid file type: {pdf_file.suffix}")

        _update_task_state("PROCESSING", {"status": TaskStatus.PROCESSING.value})

        # Import pipeline here to avoid circular imports
        from src.pipeline.runner import PipelineRunner

        # Run the extraction pipeline
        runner = PipelineRunner(enable_checkpointing=False)
        pipeline_result = runner.extract_from_pdf(
            pdf_path=pdf_path,
            custom_schema=None,
        )

        # Validate pipeline result
        if pipeline_result is None:
            raise RuntimeError("Pipeline returned no result")

        result.processing_id = pipeline_result.get("processing_id", "")
        result.status = TaskStatus.VALIDATING

        _update_task_state("VALIDATING", {"status": TaskStatus.VALIDATING.value})

        # Handle export
        result.status = TaskStatus.EXPORTING
        _update_task_state("EXPORTING", {"status": TaskStatus.EXPORTING.value})

        output_path = ""
        if output_dir:
            output_base = Path(output_dir) / result.processing_id

            if export_format in ("json", "both"):
                from src.export import ExportFormat, export_to_json

                json_path = output_base.with_suffix(".json")
                export_to_json(
                    pipeline_result,
                    output_path=json_path,
                    format=ExportFormat.DETAILED,
                    include_metadata=True,
                    include_confidence=True,
                )
                output_path = str(json_path)

            if export_format in ("excel", "both"):
                from src.export import export_to_excel

                excel_path = output_base.with_suffix(".xlsx")
                export_to_excel(
                    pipeline_result,
                    output_path=excel_path,
                    mask_phi=mask_phi,
                )
                if export_format == "excel":
                    output_path = str(excel_path)
                else:
                    output_path = f"{json_path}; {excel_path}"

        # Build final result
        completed_at = datetime.now(UTC)
        duration_ms = int((completed_at - started_at).total_seconds() * 1000)

        result.status = TaskStatus.COMPLETED
        result.output_path = output_path
        result.completed_at = completed_at.isoformat()
        result.duration_ms = duration_ms
        result.field_count = len(pipeline_result.get("merged_extraction", {}))
        result.overall_confidence = pipeline_result.get("overall_confidence", 0.0)
        result.requires_human_review = pipeline_result.get("requires_human_review", False)
        result.human_review_reason = pipeline_result.get("human_review_reason", "")
        result.warnings = pipeline_result.get("warnings", [])

        # Convert to dict and add reprocessing metadata
        result_dict = result.to_dict()
        result_dict["original_task_id"] = original_task_id
        result_dict["reprocess_task_id"] = task_id
        result_dict["is_reprocess"] = True

        logger.info(
            "reprocess_task_complete",
            task_id=task_id,
            original_task_id=original_task_id,
            processing_id=result.processing_id,
            duration_ms=duration_ms,
            field_count=result.field_count,
        )

        return result_dict

    except SoftTimeLimitExceeded:
        result.status = TaskStatus.FAILED
        result.errors = ["Reprocess task exceeded time limit"]
        result.completed_at = datetime.now(UTC).isoformat()
        logger.error("reprocess_task_timeout", task_id=task_id, original_task_id=original_task_id)
        result_dict = result.to_dict()
        result_dict["original_task_id"] = original_task_id
        result_dict["reprocess_task_id"] = task_id
        result_dict["is_reprocess"] = True
        return result_dict

    except MaxRetriesExceededError:
        result.status = TaskStatus.FAILED
        result.errors = ["Reprocess maximum retries exceeded"]
        result.completed_at = datetime.now(UTC).isoformat()
        logger.error(
            "reprocess_task_max_retries", task_id=task_id, original_task_id=original_task_id
        )
        result_dict = result.to_dict()
        result_dict["original_task_id"] = original_task_id
        result_dict["reprocess_task_id"] = task_id
        result_dict["is_reprocess"] = True
        return result_dict

    except FileNotFoundError as e:
        result.status = TaskStatus.FAILED
        result.errors = [str(e)]
        result.completed_at = datetime.now(UTC).isoformat()
        logger.error(
            "reprocess_task_file_not_found",
            task_id=task_id,
            original_task_id=original_task_id,
            error=str(e),
        )
        result_dict = result.to_dict()
        result_dict["original_task_id"] = original_task_id
        result_dict["reprocess_task_id"] = task_id
        result_dict["is_reprocess"] = True
        return result_dict

    except Exception as e:
        # Attempt retry for transient errors
        if self.request.retries < self.max_retries:
            result.status = TaskStatus.RETRYING
            logger.warning(
                "reprocess_task_retrying",
                task_id=task_id,
                original_task_id=original_task_id,
                error=str(e),
                retry_count=self.request.retries,
            )
            raise self.retry(exc=e, countdown=60 * (2**self.request.retries))

        result.status = TaskStatus.FAILED
        result.errors = [str(e)]
        result.completed_at = datetime.now(UTC).isoformat()
        logger.error(
            "reprocess_task_error", task_id=task_id, original_task_id=original_task_id, error=str(e)
        )
        result_dict = result.to_dict()
        result_dict["original_task_id"] = original_task_id
        result_dict["reprocess_task_id"] = task_id
        result_dict["is_reprocess"] = True
        return result_dict


def get_task_status(task_id: str) -> dict[str, Any]:
    """
    Get the status of a task.

    Args:
        task_id: Celery task ID.

    Returns:
        Task status information.
    """
    from celery.result import AsyncResult

    result = AsyncResult(task_id, app=celery_app)

    status_info = {
        "task_id": task_id,
        "status": result.status,
        "ready": result.ready(),
        "successful": result.successful() if result.ready() else None,
        "failed": result.failed() if result.ready() else None,
    }

    if result.ready():
        try:
            status_info["result"] = result.get(timeout=1)
        except Exception as e:
            status_info["error"] = str(e)
    elif result.info:
        status_info["progress"] = result.info

    return status_info


def cancel_task(task_id: str, terminate: bool = False) -> dict[str, Any]:
    """
    Cancel a pending or running task.

    Args:
        task_id: Celery task ID.
        terminate: Whether to terminate the worker process.

    Returns:
        Cancellation result.
    """
    from celery.result import AsyncResult

    result = AsyncResult(task_id, app=celery_app)

    if result.ready():
        return {
            "task_id": task_id,
            "cancelled": False,
            "reason": "Task already completed",
        }

    celery_app.control.revoke(task_id, terminate=terminate)

    logger.info(
        "task_cancelled",
        task_id=task_id,
        terminate=terminate,
    )

    return {
        "task_id": task_id,
        "cancelled": True,
        "terminate": terminate,
    }
