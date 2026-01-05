"""
Extraction state definitions for LangGraph workflow.

Defines the TypedDict state that flows through the extraction pipeline,
tracking all extraction data, validation results, and control flow.
"""

import copy
import secrets
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, TypedDict


class ExtractionStatus(str, Enum):
    """Status of the extraction pipeline."""

    PENDING = "pending"
    PREPROCESSING = "preprocessing"
    ANALYZING = "analyzing"
    EXTRACTING = "extracting"
    VALIDATING = "validating"
    RETRYING = "retrying"
    COMPLETED = "completed"
    FAILED = "failed"
    HUMAN_REVIEW = "human_review"


class ConfidenceLevel(str, Enum):
    """Confidence level classification."""

    HIGH = "high"  # >= 0.85
    MEDIUM = "medium"  # 0.50 - 0.84
    LOW = "low"  # < 0.50


@dataclass(frozen=True, slots=True)
class FieldMetadata:
    """
    Metadata for an extracted field.

    Tracks extraction source, confidence, and validation status.

    Attributes:
        field_name: Name of the extracted field.
        value: Extracted value.
        confidence: Confidence score 0.0-1.0.
        pass1_value: Value from first extraction pass.
        pass2_value: Value from second extraction pass.
        passes_agree: Whether both passes agree.
        location_hint: Description of where value was found.
        validation_passed: Whether field passed validation.
        validation_errors: List of validation error messages.
        source_page: Page number where value was found.
        is_hallucination_flag: Whether flagged as potential hallucination.
    """

    field_name: str
    value: Any
    confidence: float
    pass1_value: Any = None
    pass2_value: Any = None
    passes_agree: bool = True
    location_hint: str = ""
    validation_passed: bool = True
    validation_errors: tuple[str, ...] = ()
    source_page: int = 1
    is_hallucination_flag: bool = False

    @property
    def confidence_level(self) -> ConfidenceLevel:
        """Get confidence level classification."""
        if self.confidence >= 0.85:
            return ConfidenceLevel.HIGH
        if self.confidence >= 0.50:
            return ConfidenceLevel.MEDIUM
        return ConfidenceLevel.LOW

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "field_name": self.field_name,
            "value": self.value,
            "confidence": self.confidence,
            "confidence_level": self.confidence_level.value,
            "pass1_value": self.pass1_value,
            "pass2_value": self.pass2_value,
            "passes_agree": self.passes_agree,
            "location_hint": self.location_hint,
            "validation_passed": self.validation_passed,
            "validation_errors": list(self.validation_errors),
            "source_page": self.source_page,
            "is_hallucination_flag": self.is_hallucination_flag,
        }


@dataclass(slots=True)
class PageExtraction:
    """
    Extraction results for a single page.

    Attributes:
        page_number: One-indexed page number.
        pass1_raw: Raw JSON from first extraction pass.
        pass2_raw: Raw JSON from second extraction pass.
        merged_fields: Merged field values with metadata.
        extraction_time_ms: Time taken for extraction.
        vlm_calls: Number of VLM calls made.
        errors: List of extraction errors.
    """

    page_number: int
    pass1_raw: dict[str, Any] = field(default_factory=dict)
    pass2_raw: dict[str, Any] = field(default_factory=dict)
    merged_fields: dict[str, FieldMetadata] = field(default_factory=dict)
    extraction_time_ms: int = 0
    vlm_calls: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def overall_confidence(self) -> float:
        """Calculate average confidence across all fields."""
        if not self.merged_fields:
            return 0.0
        total = sum(f.confidence for f in self.merged_fields.values())
        return total / len(self.merged_fields)

    @property
    def agreement_rate(self) -> float:
        """Calculate rate of agreement between passes."""
        if not self.merged_fields:
            return 1.0
        agreed = sum(1 for f in self.merged_fields.values() if f.passes_agree)
        return agreed / len(self.merged_fields)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "page_number": self.page_number,
            "pass1_raw": self.pass1_raw,
            "pass2_raw": self.pass2_raw,
            "merged_fields": {k: v.to_dict() for k, v in self.merged_fields.items()},
            "extraction_time_ms": self.extraction_time_ms,
            "vlm_calls": self.vlm_calls,
            "errors": self.errors,
            "overall_confidence": self.overall_confidence,
            "agreement_rate": self.agreement_rate,
        }


@dataclass(slots=True)
class ValidationResult:
    """
    Validation results from the Validator agent.

    Attributes:
        is_valid: Whether extraction passed validation.
        overall_confidence: Overall extraction confidence.
        confidence_level: Classification of confidence.
        field_validations: Per-field validation results.
        cross_field_validations: Cross-field rule validations.
        hallucination_flags: Fields flagged as potential hallucinations.
        warnings: Non-fatal validation warnings.
        errors: Fatal validation errors.
        requires_retry: Whether extraction should be retried.
        requires_human_review: Whether human review is needed.
        validation_time_ms: Time taken for validation.
    """

    is_valid: bool = True
    overall_confidence: float = 0.0
    confidence_level: ConfidenceLevel = ConfidenceLevel.LOW
    field_validations: dict[str, bool] = field(default_factory=dict)
    cross_field_validations: list[dict[str, Any]] = field(default_factory=list)
    hallucination_flags: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    requires_retry: bool = False
    requires_human_review: bool = False
    validation_time_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "is_valid": self.is_valid,
            "overall_confidence": self.overall_confidence,
            "confidence_level": self.confidence_level.value,
            "field_validations": self.field_validations,
            "cross_field_validations": self.cross_field_validations,
            "hallucination_flags": self.hallucination_flags,
            "warnings": self.warnings,
            "errors": self.errors,
            "requires_retry": self.requires_retry,
            "requires_human_review": self.requires_human_review,
            "validation_time_ms": self.validation_time_ms,
        }


class DocumentAnalysis(TypedDict, total=False):
    """Analysis results from Analyzer agent."""

    document_type: str
    document_type_confidence: float
    schema_name: str
    detected_structures: list[str]
    has_tables: bool
    has_handwriting: bool
    has_signatures: bool
    page_relationships: dict[int, str]
    regions_of_interest: list[dict[str, Any]]
    analysis_time_ms: int


class ExtractionState(TypedDict, total=False):
    """
    Complete extraction state for LangGraph workflow.

    This TypedDict defines all state that flows through the extraction
    pipeline, tracking input, analysis, extraction, validation, and control.
    """

    # === Input Fields ===
    pdf_path: str
    pdf_hash: str
    page_images: list[dict[str, Any]]  # Serialized PageImage data
    custom_schema: dict[str, Any] | None
    processing_id: str

    # === Analysis Fields ===
    analysis: DocumentAnalysis
    selected_schema_name: str
    document_type: str

    # === Extraction Fields ===
    page_extractions: list[dict[str, Any]]  # Serialized PageExtraction data
    merged_extraction: dict[str, Any]  # Final merged extraction
    field_metadata: dict[str, dict[str, Any]]  # Field name -> FieldMetadata dict

    # === Validation Fields ===
    validation: dict[str, Any]  # Serialized ValidationResult
    overall_confidence: float
    confidence_level: str

    # === Control Fields ===
    status: str
    current_step: str
    retry_count: int
    max_retries: int
    errors: list[str]
    warnings: list[str]

    # === Timing Fields ===
    start_time: str  # ISO format
    end_time: str | None  # ISO format
    total_vlm_calls: int
    total_processing_time_ms: int

    # === Output Fields ===
    final_output: dict[str, Any] | None
    requires_human_review: bool
    human_review_reason: str | None

    # === Memory Fields (Mem0 Integration) ===
    session_id: str | None  # Session identifier for memory grouping
    recovery_checkpoint: (
        str | None
    )  # Checkpoint identifier for recovery (renamed to avoid LangGraph reserved name)
    memory_context: dict[str, Any] | None  # Retrieved context from memory
    similar_docs: list[str]  # IDs of similar previously processed documents
    provider_patterns: dict[str, Any] | None  # Provider-specific extraction patterns
    correction_hints: dict[str, Any] | None  # Hints from past corrections


def create_initial_state(
    pdf_path: str | Path,
    pdf_hash: str | None = None,
    page_images: list[dict[str, Any]] | None = None,
    custom_schema: dict[str, Any] | None = None,
    max_retries: int = 2,
    processing_id: str | None = None,
) -> ExtractionState:
    """
    Create initial extraction state for a new document.

    Args:
        pdf_path: Path to the PDF file.
        pdf_hash: SHA-256 hash of the PDF (optional, can be set later).
        page_images: List of serialized PageImage data (optional, can be set later).
        custom_schema: Optional custom schema for zero-shot extraction.
        max_retries: Maximum extraction retry attempts.
        processing_id: Optional processing ID (auto-generated if not provided).

    Returns:
        Initialized ExtractionState ready for pipeline.
    """
    if processing_id is None:
        processing_id = secrets.token_hex(16)

    return ExtractionState(
        # Input
        pdf_path=str(pdf_path),
        pdf_hash=pdf_hash or "",
        page_images=page_images or [],
        custom_schema=custom_schema,
        processing_id=processing_id,
        # Analysis
        analysis={},
        selected_schema_name="",
        document_type="",
        # Extraction
        page_extractions=[],
        merged_extraction={},
        field_metadata={},
        # Validation
        validation={},
        overall_confidence=0.0,
        confidence_level=ConfidenceLevel.LOW.value,
        # Control
        status=ExtractionStatus.PENDING.value,
        current_step="initialized",
        retry_count=0,
        max_retries=max_retries,
        errors=[],
        warnings=[],
        # Timing
        start_time=datetime.now(UTC).isoformat(),
        end_time=None,
        total_vlm_calls=0,
        total_processing_time_ms=0,
        # Output
        final_output=None,
        requires_human_review=False,
        human_review_reason=None,
        # Memory
        session_id=secrets.token_hex(8),
        recovery_checkpoint=None,
        memory_context=None,
        similar_docs=[],
        provider_patterns=None,
        correction_hints=None,
    )


def update_state(
    state: ExtractionState,
    updates: dict[str, Any],
) -> ExtractionState:
    """
    Create updated state with new values using selective copy.

    This creates a new state dict with updates applied, ensuring
    nested mutable structures (lists, dicts) are not shared between states.
    Used for immutable state updates in LangGraph.

    PERFORMANCE: page_images is treated as immutable (copied by reference) since
    it contains large base64-encoded image data that is expensive to deep copy.
    All other fields are deep copied for safety.

    Args:
        state: Current state.
        updates: Dictionary of updates to apply.

    Returns:
        New ExtractionState with updates applied (selectively copied).
    """
    # Start with a shallow copy of the state dict
    new_state: ExtractionState = {}  # type: ignore

    # Selectively copy state fields
    for key, value in dict(state).items():
        if key == "page_images":
            # page_images is large and treated as immutable - use reference copy
            # This avoids expensive deep copy of base64 image data
            new_state[key] = value  # type: ignore
        else:
            # Deep copy all other mutable fields for safety
            new_state[key] = copy.deepcopy(value)  # type: ignore

    # Apply updates - also selectively copy
    for key, value in updates.items():
        if key == "page_images":
            # page_images updates are also copied by reference
            new_state[key] = value  # type: ignore
        else:
            # Deep copy update values to prevent external mutation
            new_state[key] = copy.deepcopy(value)  # type: ignore

    return new_state


def add_error(state: ExtractionState, error: str) -> ExtractionState:
    """Add an error to the state."""
    errors = list(state.get("errors", []))
    errors.append(error)
    return update_state(state, {"errors": errors})


def add_warning(state: ExtractionState, warning: str) -> ExtractionState:
    """Add a warning to the state."""
    warnings = list(state.get("warnings", []))
    warnings.append(warning)
    return update_state(state, {"warnings": warnings})


def increment_vlm_calls(state: ExtractionState, count: int = 1) -> ExtractionState:
    """Increment the VLM call counter."""
    current = state.get("total_vlm_calls", 0)
    return update_state(state, {"total_vlm_calls": current + count})


def set_status(
    state: ExtractionState,
    status: ExtractionStatus,
    step: str | None = None,
) -> ExtractionState:
    """Update the extraction status."""
    updates: dict[str, Any] = {"status": status.value}
    if step:
        updates["current_step"] = step
    return update_state(state, updates)


def complete_extraction(
    state: ExtractionState,
    final_output: dict[str, Any] | None = None,
    overall_confidence: float | None = None,
) -> ExtractionState:
    """
    Mark extraction as completed with final output.

    Args:
        state: Current extraction state.
        final_output: Optional final output (defaults to merged_extraction).
        overall_confidence: Optional confidence override (defaults to state value).

    Returns:
        Updated state marked as completed.
    """
    # Use provided values or defaults from state
    if final_output is None:
        final_output = state.get("merged_extraction", {})

    if overall_confidence is None:
        overall_confidence = state.get("overall_confidence", 0.0)

    # Determine confidence level
    confidence_level = ConfidenceLevel.HIGH
    if overall_confidence < 0.85:
        confidence_level = ConfidenceLevel.MEDIUM
    if overall_confidence < 0.50:
        confidence_level = ConfidenceLevel.LOW

    return update_state(
        state,
        {
            "status": ExtractionStatus.COMPLETED.value,
            "current_step": "completed",
            "final_output": final_output,
            "overall_confidence": overall_confidence,
            "confidence_level": confidence_level.value,
            "end_time": datetime.now(UTC).isoformat(),
        },
    )


def request_human_review(
    state: ExtractionState,
    reason: str,
) -> ExtractionState:
    """Mark extraction as requiring human review."""
    return update_state(
        state,
        {
            "status": ExtractionStatus.HUMAN_REVIEW.value,
            "current_step": "human_review",
            "requires_human_review": True,
            "human_review_reason": reason,
            "end_time": datetime.now(UTC).isoformat(),
        },
    )


def request_retry(
    state: ExtractionState,
    reason: str | None = None,
) -> ExtractionState:
    """
    Request extraction retry.

    Args:
        state: Current extraction state.
        reason: Optional reason for retry.

    Returns:
        Updated state requesting retry or human review if max retries exceeded.
    """
    retry_count = state.get("retry_count", 0)
    max_retries = state.get("max_retries", 2)

    if retry_count >= max_retries:
        return request_human_review(
            state,
            reason or f"Maximum retries ({max_retries}) exceeded with low confidence",
        )

    updates: dict[str, Any] = {
        "status": ExtractionStatus.RETRYING.value,
        "current_step": "retry_extraction",
        "retry_count": retry_count + 1,
    }

    # Add reason as warning if provided
    if reason:
        warnings = list(state.get("warnings", []))
        warnings.append(f"Retry requested: {reason}")
        updates["warnings"] = warnings

    return update_state(state, updates)


def fail_extraction(state: ExtractionState, error: str) -> ExtractionState:
    """Mark extraction as failed."""
    state_with_error = add_error(state, error)
    return update_state(
        state_with_error,
        {
            "status": ExtractionStatus.FAILED.value,
            "current_step": "failed",
            "end_time": datetime.now(UTC).isoformat(),
        },
    )


def serialize_field_metadata(metadata: FieldMetadata) -> dict[str, Any]:
    """Serialize FieldMetadata for state storage."""
    return metadata.to_dict()


def deserialize_field_metadata(data: dict[str, Any]) -> FieldMetadata:
    """Deserialize FieldMetadata from state storage."""
    return FieldMetadata(
        field_name=data["field_name"],
        value=data["value"],
        confidence=data["confidence"],
        pass1_value=data.get("pass1_value"),
        pass2_value=data.get("pass2_value"),
        passes_agree=data.get("passes_agree", True),
        location_hint=data.get("location_hint", ""),
        validation_passed=data.get("validation_passed", True),
        validation_errors=tuple(data.get("validation_errors", [])),
        source_page=data.get("source_page", 1),
        is_hallucination_flag=data.get("is_hallucination_flag", False),
    )


def serialize_page_extraction(extraction: PageExtraction) -> dict[str, Any]:
    """Serialize PageExtraction for state storage."""
    return extraction.to_dict()


def deserialize_page_extraction(data: dict[str, Any]) -> PageExtraction:
    """Deserialize PageExtraction from state storage."""
    merged_fields = {}
    for field_name, field_data in data.get("merged_fields", {}).items():
        merged_fields[field_name] = deserialize_field_metadata(field_data)

    return PageExtraction(
        page_number=data["page_number"],
        pass1_raw=data.get("pass1_raw", {}),
        pass2_raw=data.get("pass2_raw", {}),
        merged_fields=merged_fields,
        extraction_time_ms=data.get("extraction_time_ms", 0),
        vlm_calls=data.get("vlm_calls", 0),
        errors=data.get("errors", []),
    )


def serialize_validation_result(result: ValidationResult) -> dict[str, Any]:
    """Serialize ValidationResult for state storage."""
    return result.to_dict()


def deserialize_validation_result(data: dict[str, Any]) -> ValidationResult:
    """Deserialize ValidationResult from state storage."""
    return ValidationResult(
        is_valid=data.get("is_valid", True),
        overall_confidence=data.get("overall_confidence", 0.0),
        confidence_level=ConfidenceLevel(data.get("confidence_level", "low")),
        field_validations=data.get("field_validations", {}),
        cross_field_validations=data.get("cross_field_validations", []),
        hallucination_flags=data.get("hallucination_flags", []),
        warnings=data.get("warnings", []),
        errors=data.get("errors", []),
        requires_retry=data.get("requires_retry", False),
        requires_human_review=data.get("requires_human_review", False),
        validation_time_ms=data.get("validation_time_ms", 0),
    )


def serialize_state(state: ExtractionState) -> dict[str, Any]:
    """
    Serialize ExtractionState for persistent storage.

    Converts the TypedDict to a plain dictionary suitable for JSON serialization.

    Args:
        state: ExtractionState to serialize.

    Returns:
        Dictionary representation of the state.
    """
    # ExtractionState is already a dict-like TypedDict, but we ensure
    # all nested structures are properly serializable
    serialized: dict[str, Any] = dict(state)

    # Ensure page_images are serializable (remove any non-serializable data)
    if serialized.get("page_images"):
        serialized["page_images"] = [
            {k: v for k, v in img.items() if k != "image_bytes"}
            for img in serialized["page_images"]
        ]

    return serialized


def deserialize_state(data: dict[str, Any]) -> ExtractionState:
    """
    Deserialize ExtractionState from persistent storage.

    Args:
        data: Dictionary representation of the state.

    Returns:
        Reconstructed ExtractionState.
    """
    # Create ExtractionState from the dictionary
    # TypedDict allows dict casting
    return ExtractionState(**data)  # type: ignore
