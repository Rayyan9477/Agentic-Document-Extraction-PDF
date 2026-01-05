"""
Human review queue system for document extraction.

Manages the workflow for documents that require human intervention:
- Low confidence extractions
- Validation failures
- Hallucination detection flags
- Critical field mismatches

Provides:
- Priority-based review queue
- Review task assignment
- Correction tracking
- Audit logging
"""

import json
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any
import secrets

from src.config import get_logger


logger = get_logger(__name__)


class ReviewPriority(str, Enum):
    """Priority levels for review tasks."""

    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ReviewStatus(str, Enum):
    """Status of a review task."""

    PENDING = "pending"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    REJECTED = "rejected"
    ESCALATED = "escalated"


class ReviewReason(str, Enum):
    """Reasons for requiring human review."""

    LOW_CONFIDENCE = "low_confidence"
    VALIDATION_FAILURE = "validation_failure"
    HALLUCINATION_DETECTED = "hallucination_detected"
    DUAL_PASS_MISMATCH = "dual_pass_mismatch"
    CRITICAL_FIELD_MISSING = "critical_field_missing"
    CROSS_FIELD_VIOLATION = "cross_field_violation"
    RETRY_LIMIT_EXCEEDED = "retry_limit_exceeded"
    MANUAL_REQUEST = "manual_request"
    QUALITY_CHECK = "quality_check"


@dataclass
class ReviewField:
    """
    A field requiring human review.

    Attributes:
        field_name: Name of the field.
        extracted_value: Value extracted by VLM.
        confidence: Extraction confidence.
        reason: Why review is needed.
        pass1_value: Value from Pass 1.
        pass2_value: Value from Pass 2.
        validation_errors: Any validation errors.
        corrected_value: Human-corrected value.
        reviewer_notes: Notes from reviewer.
    """

    field_name: str
    extracted_value: Any
    confidence: float
    reason: str
    pass1_value: Any = None
    pass2_value: Any = None
    validation_errors: list[str] = field(default_factory=list)
    corrected_value: Any = None
    reviewer_notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "field_name": self.field_name,
            "extracted_value": self.extracted_value,
            "confidence": self.confidence,
            "reason": self.reason,
            "pass1_value": self.pass1_value,
            "pass2_value": self.pass2_value,
            "validation_errors": self.validation_errors,
            "corrected_value": self.corrected_value,
            "reviewer_notes": self.reviewer_notes,
        }


@dataclass
class ReviewTask:
    """
    A human review task for a document extraction.

    Attributes:
        task_id: Unique identifier for the task.
        processing_id: ID of the extraction process.
        document_path: Path to the source document.
        document_type: Type of document.
        priority: Review priority level.
        status: Current review status.
        reasons: Reasons for review.
        fields_to_review: Fields requiring review.
        extracted_data: Complete extracted data.
        overall_confidence: Overall extraction confidence.
        created_at: When task was created.
        assigned_to: Assigned reviewer.
        completed_at: When review was completed.
        corrections: Applied corrections.
        reviewer_decision: Final decision.
    """

    task_id: str
    processing_id: str
    document_path: str
    document_type: str
    priority: ReviewPriority
    status: ReviewStatus
    reasons: list[ReviewReason]
    fields_to_review: list[ReviewField]
    extracted_data: dict[str, Any]
    overall_confidence: float
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    assigned_to: str | None = None
    completed_at: datetime | None = None
    corrections: dict[str, Any] = field(default_factory=dict)
    reviewer_decision: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "task_id": self.task_id,
            "processing_id": self.processing_id,
            "document_path": self.document_path,
            "document_type": self.document_type,
            "priority": self.priority.value,
            "status": self.status.value,
            "reasons": [r.value for r in self.reasons],
            "fields_to_review": [f.to_dict() for f in self.fields_to_review],
            "extracted_data": self.extracted_data,
            "overall_confidence": self.overall_confidence,
            "created_at": self.created_at.isoformat(),
            "assigned_to": self.assigned_to,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "corrections": self.corrections,
            "reviewer_decision": self.reviewer_decision,
        }


class HumanReviewQueue:
    """
    Manages human review queue for document extractions.

    Provides FIFO queue with priority-based ordering for review tasks.
    Supports persistence to file system for durability.

    Example:
        queue = HumanReviewQueue(queue_path="/data/review_queue")

        # Add task to queue
        task = queue.create_task(
            processing_id="abc123",
            document_path="/docs/claim.pdf",
            document_type="cms1500",
            extracted_data=extracted,
            fields_to_review=[...],
            reasons=[ReviewReason.LOW_CONFIDENCE],
        )

        # Get next task for review
        next_task = queue.get_next_task()
        if next_task:
            # Complete review
            queue.complete_task(
                task_id=next_task.task_id,
                corrections={"patient_name": "Corrected Name"},
                decision="approved",
            )
    """

    def __init__(
        self,
        queue_path: str | Path | None = None,
        auto_persist: bool = True,
    ) -> None:
        """
        Initialize the review queue.

        Args:
            queue_path: Path for persisting queue to disk.
            auto_persist: Whether to auto-save on changes.
        """
        self.queue_path = Path(queue_path) if queue_path else None
        self.auto_persist = auto_persist
        self._tasks: dict[str, ReviewTask] = {}
        self._priority_order = [
            ReviewPriority.CRITICAL,
            ReviewPriority.HIGH,
            ReviewPriority.MEDIUM,
            ReviewPriority.LOW,
        ]

        # Thread lock for concurrent access protection
        # Protects _tasks dict from race conditions in multi-threaded environments
        self._lock = threading.Lock()

        # Load existing queue if path exists
        if self.queue_path and self.queue_path.exists():
            self._load_queue()

    def create_task(
        self,
        processing_id: str,
        document_path: str,
        document_type: str,
        extracted_data: dict[str, Any],
        fields_to_review: list[dict[str, Any]],
        reasons: list[ReviewReason],
        overall_confidence: float = 0.0,
        priority: ReviewPriority | None = None,
    ) -> ReviewTask:
        """
        Create a new review task.

        Args:
            processing_id: ID of the extraction process.
            document_path: Path to source document.
            document_type: Type of document.
            extracted_data: Complete extracted data.
            fields_to_review: Fields needing review.
            reasons: Reasons for review.
            overall_confidence: Overall extraction confidence.
            priority: Priority level (auto-calculated if None).

        Returns:
            Created ReviewTask.
        """
        # Generate task ID
        task_id = f"review_{secrets.token_hex(8)}"

        # Convert field dicts to ReviewField objects
        review_fields = [
            ReviewField(
                field_name=f.get("field_name", ""),
                extracted_value=f.get("extracted_value"),
                confidence=f.get("confidence", 0.0),
                reason=f.get("reason", ""),
                pass1_value=f.get("pass1_value"),
                pass2_value=f.get("pass2_value"),
                validation_errors=f.get("validation_errors", []),
            )
            for f in fields_to_review
        ]

        # Calculate priority if not provided
        if priority is None:
            priority = self._calculate_priority(reasons, overall_confidence)

        task = ReviewTask(
            task_id=task_id,
            processing_id=processing_id,
            document_path=document_path,
            document_type=document_type,
            priority=priority,
            status=ReviewStatus.PENDING,
            reasons=reasons,
            fields_to_review=review_fields,
            extracted_data=extracted_data,
            overall_confidence=overall_confidence,
        )

        with self._lock:
            self._tasks[task_id] = task

            if self.auto_persist:
                self._save_queue()

        logger.info(
            f"Created review task {task_id} for {document_path} "
            f"with priority {priority.value}"
        )

        return task

    def get_task(self, task_id: str) -> ReviewTask | None:
        """Get a specific task by ID."""
        with self._lock:
            return self._tasks.get(task_id)

    def get_next_task(
        self,
        assignee: str | None = None,
    ) -> ReviewTask | None:
        """
        Get the next pending task for review.

        Tasks are returned in priority order (CRITICAL first).

        Args:
            assignee: Optional assignee to assign the task to.

        Returns:
            Next ReviewTask or None if queue is empty.
        """
        with self._lock:
            for priority in self._priority_order:
                for task in self._tasks.values():
                    if (
                        task.status == ReviewStatus.PENDING
                        and task.priority == priority
                    ):
                        if assignee:
                            task.status = ReviewStatus.IN_PROGRESS
                        task.assigned_to = assignee
                        if self.auto_persist:
                            self._save_queue()
                    return task

        return None

    def assign_task(
        self,
        task_id: str,
        assignee: str,
    ) -> bool:
        """
        Assign a task to a reviewer.

        Args:
            task_id: Task to assign.
            assignee: Person to assign to.

        Returns:
            True if assignment successful.
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return False

            if task.status != ReviewStatus.PENDING:
                logger.warning(f"Cannot assign task {task_id}: status is {task.status.value}")
                return False

            task.status = ReviewStatus.IN_PROGRESS
            task.assigned_to = assignee

            if self.auto_persist:
                self._save_queue()

        logger.info(f"Assigned task {task_id} to {assignee}")
        return True

    def complete_task(
        self,
        task_id: str,
        corrections: dict[str, Any] | None = None,
        decision: str = "approved",
        reviewer_notes: str = "",
    ) -> bool:
        """
        Complete a review task.

        Args:
            task_id: Task to complete.
            corrections: Field corrections applied.
            decision: Final decision (approved, rejected, escalated).
            reviewer_notes: Notes from reviewer.

        Returns:
            True if completion successful.
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return False

            if task.status not in (ReviewStatus.PENDING, ReviewStatus.IN_PROGRESS):
                logger.warning(f"Cannot complete task {task_id}: status is {task.status.value}")
                return False

            task.status = ReviewStatus.COMPLETED
            task.completed_at = datetime.now(timezone.utc)
            task.corrections = corrections or {}
            task.reviewer_decision = decision

            # Update field corrections
            for field in task.fields_to_review:
                if field.field_name in task.corrections:
                    field.corrected_value = task.corrections[field.field_name]
                field.reviewer_notes = reviewer_notes

            if self.auto_persist:
                self._save_queue()

        logger.info(f"Completed task {task_id} with decision: {decision}")
        return True

    def reject_task(
        self,
        task_id: str,
        reason: str = "",
    ) -> bool:
        """
        Reject a review task (document cannot be processed).

        Args:
            task_id: Task to reject.
            reason: Reason for rejection.

        Returns:
            True if rejection successful.
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return False

            task.status = ReviewStatus.REJECTED
            task.completed_at = datetime.now(timezone.utc)
            task.reviewer_decision = f"rejected: {reason}"

            if self.auto_persist:
                self._save_queue()

        logger.info(f"Rejected task {task_id}: {reason}")
        return True

    def escalate_task(
        self,
        task_id: str,
        reason: str = "",
    ) -> bool:
        """
        Escalate a review task to higher authority.

        Args:
            task_id: Task to escalate.
            reason: Reason for escalation.

        Returns:
            True if escalation successful.
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if not task:
                return False

            task.status = ReviewStatus.ESCALATED
            task.priority = ReviewPriority.CRITICAL
            task.reviewer_decision = f"escalated: {reason}"

            if self.auto_persist:
                self._save_queue()

        logger.info(f"Escalated task {task_id}: {reason}")
        return True

    def get_pending_count(self) -> int:
        """Get count of pending tasks."""
        with self._lock:
            return sum(
                1 for t in self._tasks.values()
                if t.status == ReviewStatus.PENDING
            )

    def get_pending_by_priority(self) -> dict[str, int]:
        """Get pending task counts by priority."""
        with self._lock:
            counts = {p.value: 0 for p in ReviewPriority}
            for task in self._tasks.values():
                if task.status == ReviewStatus.PENDING:
                    counts[task.priority.value] += 1
            return counts

    def get_tasks_by_status(
        self,
        status: ReviewStatus,
    ) -> list[ReviewTask]:
        """Get all tasks with a specific status."""
        with self._lock:
            return [
                t for t in self._tasks.values()
            if t.status == status
        ]

    def get_corrected_extraction(
        self,
        task_id: str,
    ) -> dict[str, Any] | None:
        """
        Get the corrected extraction data for a completed task.

        Args:
            task_id: Completed task ID.

        Returns:
            Corrected extraction data or None.
        """
        with self._lock:
            task = self._tasks.get(task_id)
            if not task or task.status != ReviewStatus.COMPLETED:
                return None

            # Start with extracted data and apply corrections
            corrected = dict(task.extracted_data)
            corrected.update(task.corrections)

            return corrected

    def get_queue_statistics(self) -> dict[str, Any]:
        """Get statistics about the review queue."""
        with self._lock:
            total = len(self._tasks)
            by_status = {s.value: 0 for s in ReviewStatus}
            by_priority = {p.value: 0 for p in ReviewPriority}
            by_reason: dict[str, int] = {}

            for task in self._tasks.values():
                by_status[task.status.value] += 1
                by_priority[task.priority.value] += 1
                for reason in task.reasons:
                    by_reason[reason.value] = by_reason.get(reason.value, 0) + 1

            # Calculate average wait time for completed tasks
            wait_times = []
            for task in self._tasks.values():
                if task.status == ReviewStatus.COMPLETED and task.completed_at:
                    wait_time = (task.completed_at - task.created_at).total_seconds()
                    wait_times.append(wait_time)

            avg_wait_time = sum(wait_times) / len(wait_times) if wait_times else 0

            return {
                "total_tasks": total,
                "by_status": by_status,
                "by_priority": by_priority,
                "by_reason": by_reason,
                "average_wait_time_seconds": avg_wait_time,
            }

    def _calculate_priority(
        self,
        reasons: list[ReviewReason],
        confidence: float,
    ) -> ReviewPriority:
        """Calculate priority based on reasons and confidence."""
        # Critical reasons
        critical_reasons = {
            ReviewReason.HALLUCINATION_DETECTED,
            ReviewReason.CRITICAL_FIELD_MISSING,
        }

        # High priority reasons
        high_reasons = {
            ReviewReason.VALIDATION_FAILURE,
            ReviewReason.CROSS_FIELD_VIOLATION,
            ReviewReason.DUAL_PASS_MISMATCH,
        }

        for reason in reasons:
            if reason in critical_reasons:
                return ReviewPriority.CRITICAL
            if reason in high_reasons:
                return ReviewPriority.HIGH

        # Use confidence for remaining cases
        if confidence < 0.30:
            return ReviewPriority.HIGH
        if confidence < 0.50:
            return ReviewPriority.MEDIUM

        return ReviewPriority.LOW

    def _save_queue(self) -> None:
        """Save queue to disk."""
        if not self.queue_path:
            return

        self.queue_path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            task_id: task.to_dict()
            for task_id, task in self._tasks.items()
        }

        with open(self.queue_path, "w") as f:
            json.dump(data, f, indent=2, default=str)

    def _load_queue(self) -> None:
        """Load queue from disk."""
        if not self.queue_path or not self.queue_path.exists():
            return

        with open(self.queue_path) as f:
            data = json.load(f)

        for task_id, task_data in data.items():
            task = self._deserialize_task(task_data)
            if task:
                self._tasks[task_id] = task

        logger.info(f"Loaded {len(self._tasks)} tasks from queue")

    def _deserialize_task(self, data: dict[str, Any]) -> ReviewTask | None:
        """Deserialize a task from dictionary."""
        try:
            return ReviewTask(
                task_id=data["task_id"],
                processing_id=data["processing_id"],
                document_path=data["document_path"],
                document_type=data["document_type"],
                priority=ReviewPriority(data["priority"]),
                status=ReviewStatus(data["status"]),
                reasons=[ReviewReason(r) for r in data["reasons"]],
                fields_to_review=[
                    ReviewField(
                        field_name=f["field_name"],
                        extracted_value=f["extracted_value"],
                        confidence=f["confidence"],
                        reason=f["reason"],
                        pass1_value=f.get("pass1_value"),
                        pass2_value=f.get("pass2_value"),
                        validation_errors=f.get("validation_errors", []),
                        corrected_value=f.get("corrected_value"),
                        reviewer_notes=f.get("reviewer_notes", ""),
                    )
                    for f in data["fields_to_review"]
                ],
                extracted_data=data["extracted_data"],
                overall_confidence=data["overall_confidence"],
                created_at=datetime.fromisoformat(data["created_at"]),
                assigned_to=data.get("assigned_to"),
                completed_at=(
                    datetime.fromisoformat(data["completed_at"])
                    if data.get("completed_at") else None
                ),
                corrections=data.get("corrections", {}),
                reviewer_decision=data.get("reviewer_decision", ""),
            )
        except (KeyError, ValueError) as e:
            logger.error(f"Failed to deserialize task: {e}")
            return None


def create_review_task(
    processing_id: str,
    document_path: str,
    document_type: str,
    extracted_data: dict[str, Any],
    low_confidence_fields: list[str] | None = None,
    validation_errors: dict[str, list[str]] | None = None,
    hallucination_flags: list[str] | None = None,
    dual_pass_mismatches: dict[str, tuple[Any, Any]] | None = None,
    overall_confidence: float = 0.0,
    field_confidences: dict[str, float] | None = None,
) -> ReviewTask:
    """
    Create a review task from extraction results.

    Convenience function that builds ReviewField objects from
    various validation results.

    Args:
        processing_id: Extraction process ID.
        document_path: Source document path.
        document_type: Type of document.
        extracted_data: Extracted field values.
        low_confidence_fields: Fields with low confidence.
        validation_errors: Per-field validation errors.
        hallucination_flags: Fields flagged as hallucinations.
        dual_pass_mismatches: Fields where passes disagreed.
        overall_confidence: Overall extraction confidence.
        field_confidences: Per-field confidence scores.

    Returns:
        Created ReviewTask.

    Example:
        task = create_review_task(
            processing_id="abc123",
            document_path="/docs/claim.pdf",
            document_type="cms1500",
            extracted_data={"patient_name": "John Doe"},
            low_confidence_fields=["patient_name"],
            overall_confidence=0.45,
        )
    """
    low_confidence_fields = low_confidence_fields or []
    validation_errors = validation_errors or {}
    hallucination_flags = hallucination_flags or []
    dual_pass_mismatches = dual_pass_mismatches or {}
    field_confidences = field_confidences or {}

    # Collect all fields needing review
    fields_needing_review: set[str] = set()
    fields_needing_review.update(low_confidence_fields)
    fields_needing_review.update(validation_errors.keys())
    fields_needing_review.update(hallucination_flags)
    fields_needing_review.update(dual_pass_mismatches.keys())

    # Determine reasons
    reasons: list[ReviewReason] = []
    if low_confidence_fields:
        reasons.append(ReviewReason.LOW_CONFIDENCE)
    if validation_errors:
        reasons.append(ReviewReason.VALIDATION_FAILURE)
    if hallucination_flags:
        reasons.append(ReviewReason.HALLUCINATION_DETECTED)
    if dual_pass_mismatches:
        reasons.append(ReviewReason.DUAL_PASS_MISMATCH)

    if not reasons:
        reasons.append(ReviewReason.MANUAL_REQUEST)

    # Build review fields
    review_fields: list[dict[str, Any]] = []
    for field_name in fields_needing_review:
        reason_parts = []
        if field_name in low_confidence_fields:
            reason_parts.append("low confidence")
        if field_name in validation_errors:
            reason_parts.append("validation failed")
        if field_name in hallucination_flags:
            reason_parts.append("potential hallucination")
        if field_name in dual_pass_mismatches:
            reason_parts.append("dual-pass mismatch")

        p1_val, p2_val = None, None
        if field_name in dual_pass_mismatches:
            p1_val, p2_val = dual_pass_mismatches[field_name]

        review_fields.append({
            "field_name": field_name,
            "extracted_value": extracted_data.get(field_name),
            "confidence": field_confidences.get(field_name, 0.0),
            "reason": ", ".join(reason_parts),
            "pass1_value": p1_val,
            "pass2_value": p2_val,
            "validation_errors": validation_errors.get(field_name, []),
        })

    # Create queue and task
    queue = HumanReviewQueue(auto_persist=False)
    return queue.create_task(
        processing_id=processing_id,
        document_path=document_path,
        document_type=document_type,
        extracted_data=extracted_data,
        fields_to_review=review_fields,
        reasons=reasons,
        overall_confidence=overall_confidence,
    )
