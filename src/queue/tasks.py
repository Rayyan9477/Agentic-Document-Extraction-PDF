"""
Celery task definitions for document processing.

Provides async task wrappers for the document extraction pipeline
with comprehensive error handling and status tracking.
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
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

    Returns:
        TaskResult as dictionary.

    Raises:
        FileNotFoundError: If PDF file not found.
        ValueError: If invalid parameters.
    """
    task_id = self.request.id or "unknown"
    started_at = datetime.now(timezone.utc)

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
        from src.pipeline.graph import run_extraction_pipeline

        # Run the extraction pipeline
        pipeline_result = run_extraction_pipeline(
            pdf_path=pdf_path,
            schema_name=schema_name,
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
                from src.export import export_to_json, ExportFormat

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
        completed_at = datetime.now(timezone.utc)
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

        logger.info(
            "document_task_complete",
            task_id=task_id,
            processing_id=result.processing_id,
            duration_ms=duration_ms,
            field_count=result.field_count,
        )

        return result.to_dict()

    except SoftTimeLimitExceeded:
        result.status = TaskStatus.FAILED
        result.errors = ["Task exceeded time limit"]
        result.completed_at = datetime.now(timezone.utc).isoformat()
        logger.error("task_timeout", task_id=task_id)
        return result.to_dict()

    except MaxRetriesExceededError:
        result.status = TaskStatus.FAILED
        result.errors = ["Maximum retries exceeded"]
        result.completed_at = datetime.now(timezone.utc).isoformat()
        logger.error("task_max_retries", task_id=task_id)
        return result.to_dict()

    except FileNotFoundError as e:
        result.status = TaskStatus.FAILED
        result.errors = [str(e)]
        result.completed_at = datetime.now(timezone.utc).isoformat()
        logger.error("task_file_not_found", task_id=task_id, error=str(e))
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
            raise self.retry(exc=e, countdown=60 * (2 ** self.request.retries))

        result.status = TaskStatus.FAILED
        result.errors = [str(e)]
        result.completed_at = datetime.now(timezone.utc).isoformat()
        logger.error("task_error", task_id=task_id, error=str(e))
        return result.to_dict()


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

    Returns:
        Batch processing result with individual task results.
    """
    task_id = self.request.id or "unknown"
    started_at = datetime.now(timezone.utc)

    logger.info(
        "batch_task_start",
        task_id=task_id,
        document_count=len(pdf_paths),
    )

    results: list[dict[str, Any]] = []
    successful = 0
    failed = 0

    for idx, pdf_path in enumerate(pdf_paths):
        _update_task_state(
            "PROCESSING",
            {
                "current": idx + 1,
                "total": len(pdf_paths),
                "current_file": pdf_path,
            },
        )

        try:
            # Process each document synchronously within the batch
            doc_result = process_document_task.apply(
                args=[pdf_path],
                kwargs={
                    "output_dir": output_dir,
                    "schema_name": schema_name,
                    "export_format": export_format,
                    "mask_phi": mask_phi,
                },
            ).get()

            results.append(doc_result)

            if doc_result.get("status") == TaskStatus.COMPLETED.value:
                successful += 1
            else:
                failed += 1
                if stop_on_error:
                    logger.warning(
                        "batch_stopped_on_error",
                        task_id=task_id,
                        failed_file=pdf_path,
                    )
                    break

        except Exception as e:
            failed += 1
            results.append({
                "pdf_path": pdf_path,
                "status": TaskStatus.FAILED.value,
                "errors": [str(e)],
            })

            if stop_on_error:
                logger.warning(
                    "batch_stopped_on_error",
                    task_id=task_id,
                    failed_file=pdf_path,
                    error=str(e),
                )
                break

    completed_at = datetime.now(timezone.utc)
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

    Args:
        self: Task instance (bound).
        original_task_id: Original failed task ID.
        pdf_path: Path to PDF file.
        output_dir: Output directory for results.
        schema_name: Schema to use for extraction.
        export_format: Export format (json/excel/both).
        mask_phi: Whether to mask PHI fields.

    Returns:
        TaskResult as dictionary.
    """
    task_id = self.request.id or "unknown"

    logger.info(
        "reprocess_task_start",
        task_id=task_id,
        original_task_id=original_task_id,
        pdf_path=pdf_path,
    )

    # Use the main processing task
    result = process_document_task.apply(
        args=[pdf_path],
        kwargs={
            "output_dir": output_dir,
            "schema_name": schema_name,
            "export_format": export_format,
            "mask_phi": mask_phi,
        },
    ).get()

    # Add reprocessing metadata
    result["original_task_id"] = original_task_id
    result["reprocess_task_id"] = task_id

    logger.info(
        "reprocess_task_complete",
        task_id=task_id,
        original_task_id=original_task_id,
        status=result.get("status"),
    )

    return result


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
