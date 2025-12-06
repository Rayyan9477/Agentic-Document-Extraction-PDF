"""
Pipeline module for document extraction workflow.

Provides LangGraph-based extraction pipeline with state management,
checkpointing, and multi-agent coordination.
"""

# State module is always available (no heavy dependencies)
from src.pipeline.state import (
    ExtractionState,
    ExtractionStatus,
    FieldMetadata,
    PageExtraction,
    ValidationResult,
    ConfidenceLevel,
    DocumentAnalysis,
    create_initial_state,
    update_state,
    set_status,
    add_error,
    add_warning,
    increment_vlm_calls,
    complete_extraction,
    request_human_review,
    request_retry,
    fail_extraction,
    serialize_state,
    deserialize_state,
    serialize_field_metadata,
    deserialize_field_metadata,
    serialize_page_extraction,
    deserialize_page_extraction,
    serialize_validation_result,
    deserialize_validation_result,
)

# Runner requires client/preprocessing which may not be available
try:
    from src.pipeline.runner import (
        PipelineRunner,
        extract_document,
        get_extraction_result,
    )
    _RUNNER_AVAILABLE = True
except ImportError:
    PipelineRunner = None  # type: ignore[misc, assignment]
    extract_document = None  # type: ignore[misc, assignment]
    get_extraction_result = None  # type: ignore[misc, assignment]
    _RUNNER_AVAILABLE = False

__all__ = [
    # Availability flag
    "_RUNNER_AVAILABLE",
    # State types
    "ExtractionState",
    "ExtractionStatus",
    "FieldMetadata",
    "PageExtraction",
    "ValidationResult",
    "ConfidenceLevel",
    "DocumentAnalysis",
    # State functions
    "create_initial_state",
    "update_state",
    "set_status",
    "add_error",
    "add_warning",
    "increment_vlm_calls",
    "complete_extraction",
    "request_human_review",
    "request_retry",
    "fail_extraction",
    # Serialization
    "serialize_state",
    "deserialize_state",
    "serialize_field_metadata",
    "deserialize_field_metadata",
    "serialize_page_extraction",
    "deserialize_page_extraction",
    "serialize_validation_result",
    "deserialize_validation_result",
    # Runner (may be None if dependencies unavailable)
    "PipelineRunner",
    "extract_document",
    "get_extraction_result",
]
