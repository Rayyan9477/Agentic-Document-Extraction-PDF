"""
Pydantic models for API request/response schemas.

Defines strongly-typed models for all API endpoints
with comprehensive validation and documentation.
"""

from datetime import datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field, field_validator


class ExtractionModeEnum(str, Enum):
    """Extraction mode options."""

    SINGLE = "single"  # One record per page (legacy pipeline)
    MULTI = "multi"  # Multiple records per page (boundary detection)
    AUTO = "auto"  # Auto-detect (defaults to multi)


class ExportFormatEnum(str, Enum):
    """Export format options."""

    JSON = "json"
    EXCEL = "excel"
    MARKDOWN = "markdown"
    BOTH = "both"  # JSON and Excel
    ALL = "all"  # JSON, Excel, and Markdown


class ProcessingPriority(str, Enum):
    """Processing priority levels."""

    LOW = "low"
    NORMAL = "normal"
    HIGH = "high"


class TaskStatusEnum(str, Enum):
    """Task status values."""

    PENDING = "pending"
    STARTED = "started"
    PROCESSING = "processing"
    VALIDATING = "validating"
    EXPORTING = "exporting"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    CANCELLED = "cancelled"


class ConfidenceLevelEnum(str, Enum):
    """Confidence level categories."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ProcessRequest(BaseModel):
    """
    Request model for document processing.

    Attributes:
        pdf_path: Path to the PDF file to process.
        schema_name: Optional schema name for extraction.
        export_format: Output format (json/excel/both).
        output_dir: Directory for output files.
        mask_phi: Whether to mask PHI fields in output.
        priority: Processing priority level.
        async_processing: Whether to process asynchronously.
        callback_url: URL to call on completion (async only).
    """

    pdf_path: str = Field(
        ...,
        description="Path to the PDF file to process",
        min_length=1,
    )
    schema_name: str | None = Field(
        None,
        description="Schema name for extraction (auto-detected if not specified)",
    )
    export_format: ExportFormatEnum = Field(
        ExportFormatEnum.JSON,
        description="Output format for extraction results",
    )
    output_dir: str | None = Field(
        None,
        description="Directory for output files",
    )
    mask_phi: bool = Field(
        False,
        description="Whether to mask PHI fields in output",
    )
    priority: ProcessingPriority = Field(
        ProcessingPriority.NORMAL,
        description="Processing priority level",
    )
    extraction_mode: ExtractionModeEnum = Field(
        ExtractionModeEnum.MULTI,
        description="Extraction mode: 'multi' for multi-record (per patient), 'single' for legacy pipeline",
    )
    async_processing: bool = Field(
        False,
        description="Whether to process asynchronously",
    )
    callback_url: str | None = Field(
        None,
        description="URL to call on completion (async only)",
    )

    @field_validator("pdf_path")
    @classmethod
    def validate_pdf_path(cls, v: str) -> str:
        """Validate PDF path format."""
        if not v.lower().endswith(".pdf"):
            raise ValueError("File must have .pdf extension")
        return v


class FieldResult(BaseModel):
    """Individual field extraction result."""

    value: Any = Field(None, description="Extracted value")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Confidence score")
    confidence_level: ConfidenceLevelEnum = Field(
        ConfidenceLevelEnum.LOW,
        description="Confidence category",
    )
    location: str | None = Field(None, description="Location in document")
    passes_agree: bool = Field(True, description="Whether dual passes agree")
    validation_passed: bool = Field(True, description="Whether validation passed")


class ValidationResult(BaseModel):
    """Validation result details."""

    is_valid: bool = Field(..., description="Overall validation status")
    field_validations: dict[str, Any] = Field(
        default_factory=dict,
        description="Per-field validation results",
    )
    cross_field_validations: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Cross-field validation results",
    )
    hallucination_flags: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Detected hallucination patterns",
    )
    warnings: list[str] = Field(default_factory=list, description="Validation warnings")
    errors: list[str] = Field(default_factory=list, description="Validation errors")


class ProcessingMetadata(BaseModel):
    """Processing metadata."""

    processing_id: str = Field(..., description="Unique processing ID")
    pdf_path: str = Field(..., description="Source PDF path")
    pdf_hash: str = Field("", description="PDF file hash")
    document_type: str = Field("", description="Detected document type")
    schema_name: str = Field("", description="Schema used for extraction")
    page_count: int = Field(0, ge=0, description="Number of pages processed")
    start_time: str = Field("", description="Processing start timestamp")
    end_time: str | None = Field(None, description="Processing end timestamp")
    processing_time_ms: int = Field(0, ge=0, description="Total processing time")
    total_vlm_calls: int = Field(0, ge=0, description="Number of VLM API calls")
    retry_count: int = Field(0, ge=0, description="Number of retries")


class ProcessResponse(BaseModel):
    """
    Response model for document processing.

    Attributes:
        processing_id: Unique processing ID.
        status: Processing status.
        data: Extracted field values.
        field_metadata: Per-field extraction metadata.
        validation: Validation results.
        metadata: Processing metadata.
        overall_confidence: Overall extraction confidence.
        confidence_level: Overall confidence category.
        requires_human_review: Whether human review is needed.
        human_review_reason: Reason for human review.
        output_path: Path to output file(s).
        errors: List of errors.
        warnings: List of warnings.
    """

    processing_id: str = Field(..., description="Unique processing ID")
    status: TaskStatusEnum = Field(..., description="Processing status")
    data: dict[str, Any] = Field(default_factory=dict, description="Extracted values")
    field_metadata: dict[str, FieldResult] = Field(
        default_factory=dict,
        description="Per-field metadata",
    )
    validation: ValidationResult | None = Field(
        None,
        description="Validation results",
    )
    metadata: ProcessingMetadata | None = Field(
        None,
        description="Processing metadata",
    )
    overall_confidence: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description="Overall confidence score",
    )
    confidence_level: ConfidenceLevelEnum = Field(
        ConfidenceLevelEnum.LOW,
        description="Overall confidence category",
    )
    requires_human_review: bool = Field(
        False,
        description="Whether human review is needed",
    )
    human_review_reason: str = Field(
        "",
        description="Reason for human review",
    )
    output_path: str | None = Field(None, description="Path to output file(s)")
    errors: list[str] = Field(default_factory=list, description="Processing errors")
    warnings: list[str] = Field(default_factory=list, description="Processing warnings")


class AsyncProcessResponse(BaseModel):
    """Response model for async processing request."""

    task_id: str = Field(..., description="Celery task ID")
    status: TaskStatusEnum = Field(TaskStatusEnum.PENDING, description="Initial status")
    message: str = Field("Task queued for processing", description="Status message")
    status_url: str = Field(..., description="URL to check task status")


class BatchProcessRequest(BaseModel):
    """
    Request model for batch document processing.

    Attributes:
        pdf_paths: List of PDF file paths to process.
        schema_name: Optional schema name for extraction.
        export_format: Output format (json/excel/both).
        output_dir: Directory for output files.
        mask_phi: Whether to mask PHI fields in output.
        stop_on_error: Whether to stop on first error.
        async_processing: Whether to process asynchronously.
    """

    pdf_paths: list[str] = Field(
        ...,
        description="List of PDF file paths to process",
        min_length=1,
        max_length=100,
    )
    schema_name: str | None = Field(
        None,
        description="Schema name for extraction",
    )
    export_format: ExportFormatEnum = Field(
        ExportFormatEnum.JSON,
        description="Output format for extraction results",
    )
    output_dir: str = Field(
        ...,
        description="Directory for output files",
    )
    mask_phi: bool = Field(
        False,
        description="Whether to mask PHI fields in output",
    )
    stop_on_error: bool = Field(
        False,
        description="Whether to stop on first error",
    )
    async_processing: bool = Field(
        True,
        description="Whether to process asynchronously (recommended for batch)",
    )

    @field_validator("pdf_paths")
    @classmethod
    def validate_pdf_paths(cls, v: list[str]) -> list[str]:
        """Validate all PDF paths."""
        for path in v:
            if not path.lower().endswith(".pdf"):
                raise ValueError(f"All files must have .pdf extension: {path}")
        return v


class BatchItemResult(BaseModel):
    """Result for a single item in batch processing."""

    pdf_path: str = Field(..., description="PDF file path")
    processing_id: str = Field("", description="Processing ID")
    status: TaskStatusEnum = Field(..., description="Processing status")
    field_count: int = Field(0, ge=0, description="Number of extracted fields")
    overall_confidence: float = Field(0.0, ge=0.0, le=1.0, description="Confidence score")
    output_path: str = Field("", description="Output file path")
    errors: list[str] = Field(default_factory=list, description="Errors")


class BatchProcessResponse(BaseModel):
    """
    Response model for batch document processing.

    Attributes:
        batch_id: Unique batch ID.
        status: Overall batch status.
        total_documents: Total documents in batch.
        successful: Number of successful extractions.
        failed: Number of failed extractions.
        started_at: Batch start timestamp.
        completed_at: Batch completion timestamp.
        duration_ms: Total processing duration.
        results: Individual document results.
    """

    batch_id: str = Field(..., description="Unique batch ID")
    status: TaskStatusEnum = Field(..., description="Overall batch status")
    total_documents: int = Field(..., ge=0, description="Total documents")
    successful: int = Field(0, ge=0, description="Successful extractions")
    failed: int = Field(0, ge=0, description="Failed extractions")
    started_at: str = Field("", description="Start timestamp")
    completed_at: str = Field("", description="Completion timestamp")
    duration_ms: int = Field(0, ge=0, description="Processing duration")
    results: list[BatchItemResult] = Field(
        default_factory=list,
        description="Individual results",
    )


class TaskStatusResponse(BaseModel):
    """
    Response model for task status query.

    Attributes:
        task_id: Celery task ID.
        status: Current task status.
        ready: Whether task is complete.
        successful: Whether task succeeded (if complete).
        progress: Progress information (if in progress).
        result: Task result (if complete).
        error: Error message (if failed).
    """

    task_id: str = Field(..., description="Celery task ID")
    status: str = Field(..., description="Current status")
    ready: bool = Field(False, description="Whether task is complete")
    successful: bool | None = Field(None, description="Whether task succeeded")
    progress: dict[str, Any] | None = Field(
        None,
        description="Progress information",
    )
    result: ProcessResponse | BatchProcessResponse | None = Field(
        None,
        description="Task result",
    )
    error: str | None = Field(None, description="Error message")


class TaskCancelResponse(BaseModel):
    """Response model for task cancellation."""

    task_id: str = Field(..., description="Task ID")
    cancelled: bool = Field(..., description="Whether cancellation succeeded")
    reason: str = Field("", description="Cancellation reason or status")


class HealthResponse(BaseModel):
    """
    Response model for health check.

    Attributes:
        status: Overall health status.
        version: API version.
        timestamp: Current timestamp.
        components: Component health status.
    """

    status: str = Field(..., description="Overall health status")
    version: str = Field(..., description="API version")
    timestamp: str = Field(..., description="Current timestamp")
    components: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Component health",
    )


class SchemaInfo(BaseModel):
    """Information about an extraction schema."""

    name: str = Field(..., description="Schema name")
    description: str = Field("", description="Schema description")
    document_type: str = Field("", description="Target document type")
    field_count: int = Field(0, ge=0, description="Number of fields")
    version: str = Field("1.0.0", description="Schema version")


class SchemaListResponse(BaseModel):
    """Response model for schema list."""

    schemas: list[SchemaInfo] = Field(
        default_factory=list,
        description="Available schemas",
    )
    count: int = Field(0, ge=0, description="Total schema count")


class SchemaSuggestRequest(BaseModel):
    """Request model for schema suggestion wizard."""

    image_base64: str = Field(..., description="Base64-encoded document image")
    context: str = Field("", description="Optional context about the document")


class SchemaRefineRequest(BaseModel):
    """Request model for refining a schema proposal."""

    feedback: str = Field(..., description="Natural language feedback for changes")
    image_base64: str | None = Field(None, description="Optional image for visual context")


class SchemaSaveRequest(BaseModel):
    """Request model for saving a schema proposal."""

    schema_name: str | None = Field(None, description="Override schema name")


class SchemaProposalResponse(BaseModel):
    """Response model for schema proposal operations."""

    proposal_id: str = Field(..., description="Proposal identifier")
    schema_name: str = Field("", description="Proposed schema name")
    document_type_description: str = Field("", description="Document type description")
    fields: list[dict[str, Any]] = Field(default_factory=list, description="Proposed fields")
    field_count: int = Field(0, ge=0, description="Number of fields")
    groups: list[dict[str, Any]] = Field(default_factory=list, description="Field groups")
    cross_field_rules: list[dict[str, Any]] = Field(default_factory=list, description="Rules")
    confidence: float = Field(0.0, description="Proposal confidence")
    revision: int = Field(0, ge=0, description="Revision number")
    status: str = Field("draft", description="Proposal status")


class ErrorDetail(BaseModel):
    """Detailed error information."""

    code: str = Field(..., description="Error code")
    message: str = Field(..., description="Error message")
    field: str | None = Field(None, description="Field causing error")
    details: dict[str, Any] | None = Field(None, description="Additional details")


class ErrorResponse(BaseModel):
    """
    Standard error response model.

    Attributes:
        error: Error type.
        message: Human-readable error message.
        details: Detailed error information.
        request_id: Request tracking ID.
        timestamp: Error timestamp.
    """

    error: str = Field(..., description="Error type")
    message: str = Field(..., description="Error message")
    details: list[ErrorDetail] = Field(
        default_factory=list,
        description="Error details",
    )
    request_id: str = Field("", description="Request ID for tracking")
    timestamp: str = Field(
        default_factory=lambda: datetime.utcnow().isoformat(),
        description="Error timestamp",
    )


class WorkerStatusResponse(BaseModel):
    """Response model for worker status."""

    status: str = Field(..., description="Overall status")
    worker_count: int = Field(0, ge=0, description="Number of workers")
    workers: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Worker details",
    )
    registered_tasks: list[str] = Field(
        default_factory=list,
        description="Registered task names",
    )


class QueueStatsResponse(BaseModel):
    """Response model for queue statistics."""

    status: str = Field(..., description="Status")
    queues: dict[str, dict[str, int]] = Field(
        default_factory=dict,
        description="Queue statistics",
    )


class PreviewStyleEnum(str, Enum):
    """Preview style options."""

    SIMPLE = "simple"
    DETAILED = "detailed"
    SUMMARY = "summary"
    TECHNICAL = "technical"


class PreviewRequest(BaseModel):
    """
    Request model for preview generation.

    Attributes:
        processing_id: Processing ID to preview.
        style: Preview style.
        include_confidence: Include confidence indicators.
        include_validation: Include validation results.
        mask_phi: Mask PHI fields in preview.
    """

    processing_id: str = Field(..., description="Processing ID to preview")
    style: PreviewStyleEnum = Field(
        PreviewStyleEnum.DETAILED,
        description="Preview style",
    )
    include_confidence: bool = Field(
        True,
        description="Include confidence indicators",
    )
    include_validation: bool = Field(
        True,
        description="Include validation results",
    )
    mask_phi: bool = Field(
        False,
        description="Mask PHI fields",
    )


class PreviewResponse(BaseModel):
    """
    Response model for preview generation.

    Attributes:
        processing_id: Processing ID.
        format: Preview format (always markdown).
        content: Markdown formatted preview content.
        generated_at: Preview generation timestamp.
    """

    processing_id: str = Field(..., description="Processing ID")
    format: str = Field("markdown", description="Preview format")
    content: str = Field(..., description="Markdown preview content")
    generated_at: str = Field(..., description="Generation timestamp")


# ─── Multi-Record Models ───


class MultiRecordItem(BaseModel):
    """Single extracted record from multi-record extraction."""

    record_id: int = Field(..., description="Record ID")
    page_number: int = Field(..., ge=1, description="Source page number")
    primary_identifier: str = Field(..., description="Primary identifier (e.g. patient ID)")
    entity_type: str = Field("", description="Entity type (e.g. patient)")
    fields: dict[str, Any] = Field(default_factory=dict, description="Extracted field values")
    confidence: float = Field(0.0, ge=0.0, le=1.0, description="Extraction confidence")
    extraction_time_ms: int = Field(0, ge=0, description="Extraction time in ms")


class MultiRecordDuplicate(BaseModel):
    """Duplicate record detection result."""

    primary_identifier: str = Field(..., description="Identifier with duplicates")
    occurrences: int = Field(0, ge=0, description="Number of occurrences")
    pages: list[int] = Field(default_factory=list, description="Pages where found")
    record_ids: list[int] = Field(default_factory=list, description="Record IDs")


class MultiRecordResponse(BaseModel):
    """Response model for multi-record extraction."""

    pdf_path: str = Field("", description="Source PDF path")
    document_type: str = Field("", description="Detected document type")
    entity_type: str = Field("", description="Entity type (e.g. patient)")
    total_pages: int = Field(0, ge=0, description="Total pages processed")
    total_records: int = Field(0, ge=0, description="Total records extracted")
    unique_records: int = Field(0, ge=0, description="Unique records (after dedup)")
    schema_fields: list[dict[str, Any]] = Field(
        default_factory=list, description="Extraction schema fields"
    )
    records: list[MultiRecordItem] = Field(
        default_factory=list, description="All extracted records"
    )
    duplicates: list[MultiRecordDuplicate] = Field(
        default_factory=list, description="Detected duplicate records"
    )
    total_vlm_calls: int = Field(0, ge=0, description="Total VLM API calls")
    processing_time_ms: int = Field(0, ge=0, description="Total processing time in ms")
    output_paths: dict[str, str] = Field(
        default_factory=dict, description="Export file paths by format"
    )
