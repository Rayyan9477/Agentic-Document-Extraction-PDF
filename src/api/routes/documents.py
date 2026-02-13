"""
Document processing API routes.

Provides endpoints for sync and async document processing,
batch processing, and result retrieval.
"""

import re
import shutil
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile, status
from fastapi import Path as FastAPIPath

from src.api.models import (
    AsyncProcessResponse,
    BatchItemResult,
    BatchProcessRequest,
    BatchProcessResponse,
    ConfidenceLevelEnum,
    ExportFormatEnum,
    ExtractionModeEnum,
    FieldResult,
    MultiRecordDuplicate,
    MultiRecordItem,
    MultiRecordResponse,
    PreviewRequest,
    PreviewResponse,
    PreviewStyleEnum,
    ProcessingMetadata,
    ProcessRequest,
    ProcessResponse,
    TaskStatusEnum,
    ValidationResult,
)
from src.config import get_logger
from src.security.path_validator import (
    PathTraversalError,
    PathValidationError,
    SecurePathValidator,
)


logger = get_logger(__name__)
router = APIRouter()

# SECURITY: Configure allowed directories for file operations
# In production, these should be loaded from configuration
ALLOWED_PDF_DIRECTORIES: list[str] = [
    "./uploads",
    "./data",
    "./input",
    tempfile.gettempdir(),  # For uploaded files
]

ALLOWED_OUTPUT_DIRECTORIES: list[str] = [
    "./output",
    "./exports",
    "./data/output",
]

# Supported document extensions for upload/processing
SUPPORTED_DOCUMENT_EXTENSIONS = [
    ".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp",
    ".docx", ".doc", ".xlsx", ".csv",
    ".dcm", ".dicom", ".edi", ".x12", ".835", ".837",
]

# Initialize validators
_pdf_validator = SecurePathValidator(
    allowed_directories=None,  # Allow any directory but validate for traversal
    allowed_extensions=SUPPORTED_DOCUMENT_EXTENSIONS,
    allow_absolute_paths=True,
    resolve_symlinks=True,
)

_output_validator = SecurePathValidator(
    allowed_directories=None,  # Allow any directory but validate for traversal
    allowed_extensions=[".json", ".xlsx", ".xls", ".md", ".csv"],
    allow_absolute_paths=True,
    resolve_symlinks=True,
)


def _map_confidence_level(confidence: float) -> ConfidenceLevelEnum:
    """Map confidence score to level."""
    if confidence >= 0.85:
        return ConfidenceLevelEnum.HIGH
    if confidence >= 0.50:
        return ConfidenceLevelEnum.MEDIUM
    return ConfidenceLevelEnum.LOW


def _build_process_response(
    state: dict[str, Any],
    output_path: str = "",
) -> ProcessResponse:
    """Build ProcessResponse from extraction state."""
    # Extract field values
    merged = state.get("merged_extraction", {})
    data = {}
    for field_name, field_data in merged.items():
        if isinstance(field_data, dict):
            data[field_name] = field_data.get("value")
        else:
            data[field_name] = field_data

    # Build field metadata
    field_meta = state.get("field_metadata", {})
    field_metadata = {}
    for field_name, meta in field_meta.items():
        if isinstance(meta, dict):
            confidence = meta.get("confidence", 0.0)
            field_metadata[field_name] = FieldResult(
                value=data.get(field_name),
                confidence=confidence,
                confidence_level=_map_confidence_level(confidence),
                location=meta.get("location", ""),
                passes_agree=meta.get("passes_agree", True),
                validation_passed=meta.get("validation_passed", True),
            )

    # Build validation result
    validation_data = state.get("validation", {})
    validation = None
    if validation_data:
        validation = ValidationResult(
            is_valid=validation_data.get("is_valid", False),
            field_validations=validation_data.get("field_validations", {}),
            cross_field_validations=validation_data.get("cross_field_validations", []),
            hallucination_flags=validation_data.get("hallucination_flags", []),
            warnings=validation_data.get("warnings", []),
            errors=validation_data.get("errors", []),
        )

    # Build metadata
    metadata = ProcessingMetadata(
        processing_id=state.get("processing_id", ""),
        pdf_path=state.get("pdf_path", ""),
        pdf_hash=state.get("pdf_hash", ""),
        document_type=state.get("document_type", ""),
        schema_name=state.get("selected_schema_name", ""),
        page_count=len(state.get("page_images", [])),
        start_time=state.get("start_time", ""),
        end_time=state.get("end_time"),
        processing_time_ms=state.get("total_processing_time_ms", 0),
        total_vlm_calls=state.get("total_vlm_calls", 0),
        retry_count=state.get("retry_count", 0),
    )

    # Map status
    status_str = state.get("status", "completed")
    try:
        status = TaskStatusEnum(status_str)
    except ValueError:
        status = TaskStatusEnum.COMPLETED

    overall_confidence = state.get("overall_confidence", 0.0)

    return ProcessResponse(
        processing_id=state.get("processing_id", ""),
        status=status,
        data=data,
        field_metadata=field_metadata,
        validation=validation,
        metadata=metadata,
        overall_confidence=overall_confidence,
        confidence_level=_map_confidence_level(overall_confidence),
        requires_human_review=state.get("requires_human_review", False),
        human_review_reason=state.get("human_review_reason", ""),
        output_path=output_path,
        errors=state.get("errors", []),
        warnings=state.get("warnings", []),
    )


@router.post(
    "/documents/process",
    response_model=ProcessResponse | AsyncProcessResponse | MultiRecordResponse,
    summary="Process a document",
    description="Process a PDF document and extract structured data. Supports single-record (legacy) and multi-record (per-entity) extraction modes.",
)
async def process_document(
    request: ProcessRequest,
    http_request: Request,
) -> ProcessResponse | AsyncProcessResponse | MultiRecordResponse:
    """
    Process a PDF document.

    - **Sync mode**: Returns extracted data immediately.
    - **Async mode**: Queues the document and returns task ID.

    Args:
        request: Processing request parameters.
        http_request: HTTP request object.

    Returns:
        Extraction results or task ID.

    Raises:
        HTTPException: If file not found or processing fails.
    """
    request_id = getattr(http_request.state, "request_id", "")

    logger.info(
        "document_process_request",
        request_id=request_id,
        pdf_path=request.pdf_path,
        async_processing=request.async_processing,
    )

    # SECURITY: Validate path for traversal attacks before any file operations
    try:
        validated_pdf_path = _pdf_validator.validate(request.pdf_path)
    except PathTraversalError as e:
        logger.warning(
            "path_traversal_attempt",
            request_id=request_id,
            path=request.pdf_path[:100],  # Truncate for logging
            error=str(e),
        )
        raise HTTPException(
            status_code=400,
            detail="Invalid file path",  # Generic message to avoid info disclosure
        )
    except PathValidationError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file path: {e}",
        )

    # Validate output_dir if provided
    if request.output_dir:
        try:
            _ = _output_validator.validate(request.output_dir)  # Validate only
        except (PathTraversalError, PathValidationError) as e:
            logger.warning(
                "output_path_validation_failed",
                request_id=request_id,
                path=request.output_dir[:100],
                error=str(e),
            )
            raise HTTPException(
                status_code=400,
                detail="Invalid output directory path",
            )

    # Validate file exists
    pdf_path = validated_pdf_path
    if not pdf_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"PDF file not found: {request.pdf_path}",
        )

    if request.async_processing:
        # Queue for async processing
        from src.queue.tasks import process_document_task

        task = process_document_task.delay(
            pdf_path=request.pdf_path,
            output_dir=request.output_dir,
            schema_name=request.schema_name,
            export_format=request.export_format.value,
            mask_phi=request.mask_phi,
            priority=request.priority.value,
        )

        return AsyncProcessResponse(
            task_id=task.id,
            status=TaskStatusEnum.PENDING,
            message="Document queued for processing",
            status_url=f"/api/v1/tasks/{task.id}",
        )

    # Sync processing
    try:
        use_multi_record = request.extraction_mode in (
            ExtractionModeEnum.MULTI,
            ExtractionModeEnum.AUTO,
        )

        if use_multi_record:
            # Multi-record extraction: returns per-record results
            return _run_multi_record_sync(request, request_id)
        else:
            # Legacy single-record extraction
            return _run_single_record_sync(request, request_id)

    except Exception as e:
        logger.error(
            "document_process_error",
            request_id=request_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {e!s}",
        )


def _run_single_record_sync(
    request: ProcessRequest,
    request_id: str,
) -> ProcessResponse:
    """Run single-record extraction (legacy pipeline)."""
    from src.pipeline.graph import run_extraction_pipeline

    result = run_extraction_pipeline(
        pdf_path=request.pdf_path,
        schema_name=request.schema_name,
    )

    output_path = ""
    if request.output_dir:
        output_base = Path(request.output_dir) / result.get("processing_id", "output")
        output_paths = []

        if request.export_format in (
            ExportFormatEnum.JSON,
            ExportFormatEnum.BOTH,
            ExportFormatEnum.ALL,
        ):
            from src.export import ExportFormat, export_to_json

            json_path = output_base.with_suffix(".json")
            export_to_json(result, output_path=json_path, format=ExportFormat.DETAILED)
            output_paths.append(str(json_path))

        if request.export_format in (
            ExportFormatEnum.EXCEL,
            ExportFormatEnum.BOTH,
            ExportFormatEnum.ALL,
        ):
            from src.export import export_to_excel

            excel_path = output_base.with_suffix(".xlsx")
            export_to_excel(result, output_path=excel_path, mask_phi=request.mask_phi)
            output_paths.append(str(excel_path))

        if request.export_format in (ExportFormatEnum.MARKDOWN, ExportFormatEnum.ALL):
            from src.export import MarkdownStyle, export_to_markdown

            md_path = output_base.with_suffix(".md")
            export_to_markdown(
                result, output_path=md_path, style=MarkdownStyle.DETAILED, mask_phi=request.mask_phi
            )
            output_paths.append(str(md_path))

        output_path = "; ".join(output_paths) if output_paths else ""

    return _build_process_response(result, output_path)


def _run_multi_record_sync(
    request: ProcessRequest,
    request_id: str,
) -> MultiRecordResponse:
    """Run multi-record extraction (per-entity boundary detection)."""
    from src.config import get_extraction_config
    from src.export.consolidated_export import (
        detect_duplicates,
        export_excel,
        export_json,
        export_markdown,
    )
    from src.pipeline.runner import PipelineRunner

    cfg = get_extraction_config()

    runner = PipelineRunner(enable_checkpointing=False)
    result = runner.extract_multi_record(
        pdf_path=request.pdf_path,
        enable_validation=cfg["enable_validation_stage"],
        enable_self_correction=cfg["enable_self_correction"],
        confidence_threshold=cfg["validation_confidence_threshold"],
        enable_consensus=cfg["enable_consensus_for_critical_fields"],
        critical_field_keywords=cfg["critical_field_keywords"],
        max_fields_per_call=cfg["max_fields_per_extraction_call"],
        enable_schema_decomposition=cfg["enable_schema_decomposition"],
        enable_synthetic_examples=cfg["enable_synthetic_few_shot_examples"],
    )

    # Build response records
    records = [
        MultiRecordItem(
            record_id=r.record_id,
            page_number=r.page_number,
            primary_identifier=r.primary_identifier,
            entity_type=r.entity_type,
            fields=r.fields,
            confidence=r.confidence,
            extraction_time_ms=r.extraction_time_ms,
        )
        for r in result.records
    ]

    # Detect duplicates
    dups = detect_duplicates(result.records)
    duplicates = [
        MultiRecordDuplicate(
            primary_identifier=ident.title(),
            occurrences=len(indices),
            pages=[result.records[i].page_number for i in indices],
            record_ids=[result.records[i].record_id for i in indices],
        )
        for ident, indices in dups.items()
    ]

    # Export if output_dir specified
    output_paths: dict[str, str] = {}
    if request.output_dir:
        out_dir = Path(request.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        stem = Path(request.pdf_path).stem

        if request.export_format in (
            ExportFormatEnum.JSON,
            ExportFormatEnum.BOTH,
            ExportFormatEnum.ALL,
        ):
            p = out_dir / f"{stem}_multi_record.json"
            export_json(result, p)
            output_paths["json"] = str(p)

        if request.export_format in (
            ExportFormatEnum.EXCEL,
            ExportFormatEnum.BOTH,
            ExportFormatEnum.ALL,
        ):
            p = out_dir / f"{stem}_multi_record.xlsx"
            export_excel(result, p)
            output_paths["excel"] = str(p)

        if request.export_format in (ExportFormatEnum.MARKDOWN, ExportFormatEnum.ALL):
            p = out_dir / f"{stem}_multi_record.md"
            export_markdown(result, p)
            output_paths["markdown"] = str(p)

    # Count unique
    dup_indices: set[int] = set()
    for indices in dups.values():
        dup_indices.update(indices)
    unique_count = result.total_records - len(dup_indices) + len(dups)

    return MultiRecordResponse(
        pdf_path=request.pdf_path,
        document_type=result.document_type,
        entity_type=result.entity_type,
        total_pages=result.total_pages,
        total_records=result.total_records,
        unique_records=unique_count,
        schema_fields=result.schema.get("fields", []),
        records=records,
        duplicates=duplicates,
        total_vlm_calls=result.total_vlm_calls,
        processing_time_ms=result.total_processing_time_ms,
        output_paths=output_paths,
    )


@router.post(
    "/documents/upload",
    response_model=AsyncProcessResponse,
    summary="Upload and process a document",
    description="Upload a PDF file and queue it for processing.",
)
async def upload_document(
    http_request: Request,
    file: UploadFile = File(...),
    schema_name: str | None = Form(None),
    export_format: ExportFormatEnum = Form(ExportFormatEnum.JSON),
    mask_phi: bool = Form(False),
    priority: str = Form("normal"),
    extraction_mode: ExtractionModeEnum = Form(ExtractionModeEnum.MULTI),
) -> AsyncProcessResponse:
    """
    Upload and process a PDF document.

    - **file**: PDF file to upload and process
    - **schema_name**: Optional schema name for extraction
    - **export_format**: Output format (json/excel/both)
    - **mask_phi**: Whether to mask PHI fields in output
    - **priority**: Processing priority level

    Returns:
        Task ID for async processing.

    Raises:
        HTTPException: If upload or processing fails.
    """
    request_id = getattr(http_request.state, "request_id", "")

    logger.info(
        "document_upload_request",
        request_id=request_id,
        filename=file.filename,
        file_size=file.size,
        schema_name=schema_name,
    )

    # Validate file type
    if not file.filename:
        raise HTTPException(
            status_code=400,
            detail="Filename is required",
        )
    file_ext = Path(file.filename).suffix.lower()
    if file_ext not in SUPPORTED_DOCUMENT_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format: '{file_ext}'. Supported: {', '.join(SUPPORTED_DOCUMENT_EXTENSIONS)}",
        )

    # Validate file size (50MB limit)
    if file.size and file.size > 50 * 1024 * 1024:
        raise HTTPException(
            status_code=400,
            detail="File size exceeds 50MB limit",
        )

    # Create temp directory for uploads
    upload_dir = Path(tempfile.gettempdir()) / "doc_uploads"
    upload_dir.mkdir(exist_ok=True)

    # SECURITY: Sanitize filename to prevent path traversal attacks
    def secure_filename(filename: str) -> str:
        """Sanitize filename to prevent path traversal and other attacks."""
        import os

        # Get only the basename (removes any path components)
        filename = os.path.basename(filename)
        # Remove null bytes
        filename = filename.replace("\x00", "")
        # Allow only safe characters: alphanumeric, dash, underscore, dot
        safe_chars = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_.")
        filename = "".join(c if c in safe_chars else "_" for c in filename)
        # Prevent empty or dangerous filenames
        if not filename or filename.startswith("."):
            filename = f"upload{file_ext}"
        return filename

    safe_name = secure_filename(file.filename or f"upload{file_ext}")
    temp_file_path = upload_dir / f"{request_id}_{safe_name}"
    try:
        with temp_file_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
    except Exception as e:
        logger.error(
            "file_save_error",
            request_id=request_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save uploaded file: {e!s}",
        )

    # Queue for async processing
    try:
        from src.queue.tasks import process_document_task

        task = process_document_task.delay(
            pdf_path=str(temp_file_path),
            output_dir="./output",
            schema_name=schema_name,
            export_format=export_format.value,
            mask_phi=mask_phi,
            priority=priority,
            extraction_mode=extraction_mode.value,
        )

        logger.info(
            "document_upload_queued",
            request_id=request_id,
            task_id=task.id,
            temp_path=str(temp_file_path),
        )

        return AsyncProcessResponse(
            task_id=task.id,
            status=TaskStatusEnum.PENDING,
            message="Document uploaded and queued for processing",
            status_url=f"/api/v1/tasks/{task.id}",
        )

    except Exception as e:
        # Clean up temp file on error
        try:
            temp_file_path.unlink(missing_ok=True)
        except Exception:
            pass

        logger.error(
            "document_upload_queue_error",
            request_id=request_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to queue document for processing: {e!s}",
        )


@router.post(
    "/documents/batch",
    response_model=BatchProcessResponse | AsyncProcessResponse,
    summary="Process multiple documents",
    description="Process a batch of PDF documents.",
)
async def batch_process_documents(
    request: BatchProcessRequest,
    http_request: Request,
) -> BatchProcessResponse | AsyncProcessResponse:
    """
    Process multiple PDF documents in batch.

    Args:
        request: Batch processing request parameters.
        http_request: HTTP request object.

    Returns:
        Batch processing results or task ID.

    Raises:
        HTTPException: If any file not found or processing fails.
    """
    request_id = getattr(http_request.state, "request_id", "")

    logger.info(
        "batch_process_request",
        request_id=request_id,
        document_count=len(request.pdf_paths),
        async_processing=request.async_processing,
    )

    # SECURITY: Validate all paths for traversal attacks
    validated_paths: list[Path] = []
    for pdf_path in request.pdf_paths:
        try:
            validated = _pdf_validator.validate(pdf_path)
            validated_paths.append(validated)
        except PathTraversalError as e:
            logger.warning(
                "batch_path_traversal_attempt",
                request_id=request_id,
                path=pdf_path[:100],
                error=str(e),
            )
            raise HTTPException(
                status_code=400,
                detail="Invalid file path in batch",
            )
        except PathValidationError as e:
            raise HTTPException(
                status_code=400,
                detail=f"Invalid file path: {e}",
            )

    # Validate output_dir if provided
    if request.output_dir:
        try:
            _output_validator.validate(request.output_dir)
        except (PathTraversalError, PathValidationError) as e:
            logger.warning(
                "batch_output_path_validation_failed",
                request_id=request_id,
                path=request.output_dir[:100],
                error=str(e),
            )
            raise HTTPException(
                status_code=400,
                detail="Invalid output directory path",
            )

    # Validate all files exist
    missing_files = []
    for validated_path in validated_paths:
        if not validated_path.exists():
            missing_files.append(str(validated_path))

    if missing_files:
        raise HTTPException(
            status_code=404,
            detail=f"PDF files not found: {', '.join(missing_files)}",
        )

    if request.async_processing:
        # Queue for async processing
        from src.queue.tasks import batch_process_task

        task = batch_process_task.delay(
            pdf_paths=request.pdf_paths,
            output_dir=request.output_dir,
            schema_name=request.schema_name,
            export_format=request.export_format.value,
            mask_phi=request.mask_phi,
            stop_on_error=request.stop_on_error,
        )

        return AsyncProcessResponse(
            task_id=task.id,
            status=TaskStatusEnum.PENDING,
            message=f"Batch of {len(request.pdf_paths)} documents queued for processing",
            status_url=f"/api/v1/tasks/{task.id}",
        )

    # Sync processing
    try:
        from src.export import ExportFormat, export_to_excel, export_to_json
        from src.pipeline.graph import run_extraction_pipeline

        started_at = datetime.now(UTC)
        results: list[BatchItemResult] = []
        successful = 0
        failed = 0

        for pdf_path in request.pdf_paths:
            try:
                result = run_extraction_pipeline(
                    pdf_path=pdf_path,
                    schema_name=request.schema_name,
                )

                output_path = ""
                if request.output_dir:
                    output_base = Path(request.output_dir) / result.get("processing_id", "output")

                    if request.export_format in (ExportFormatEnum.JSON, ExportFormatEnum.BOTH):
                        json_path = output_base.with_suffix(".json")
                        export_to_json(result, output_path=json_path, format=ExportFormat.DETAILED)
                        output_path = str(json_path)

                    if request.export_format in (ExportFormatEnum.EXCEL, ExportFormatEnum.BOTH):
                        excel_path = output_base.with_suffix(".xlsx")
                        export_to_excel(result, output_path=excel_path, mask_phi=request.mask_phi)
                        if request.export_format == ExportFormatEnum.EXCEL:
                            output_path = str(excel_path)

                results.append(
                    BatchItemResult(
                        pdf_path=pdf_path,
                        processing_id=result.get("processing_id", ""),
                        status=TaskStatusEnum.COMPLETED,
                        field_count=len(result.get("merged_extraction", {})),
                        overall_confidence=result.get("overall_confidence", 0.0),
                        output_path=output_path,
                        errors=[],
                    )
                )
                successful += 1

            except Exception as e:
                results.append(
                    BatchItemResult(
                        pdf_path=pdf_path,
                        processing_id="",
                        status=TaskStatusEnum.FAILED,
                        field_count=0,
                        overall_confidence=0.0,
                        output_path="",
                        errors=[str(e)],
                    )
                )
                failed += 1

                if request.stop_on_error:
                    break

        completed_at = datetime.now(UTC)
        duration_ms = int((completed_at - started_at).total_seconds() * 1000)

        return BatchProcessResponse(
            batch_id=request_id,
            status=TaskStatusEnum.COMPLETED if failed == 0 else TaskStatusEnum.FAILED,
            total_documents=len(request.pdf_paths),
            successful=successful,
            failed=failed,
            started_at=started_at.isoformat(),
            completed_at=completed_at.isoformat(),
            duration_ms=duration_ms,
            results=results,
        )

    except Exception as e:
        logger.error(
            "batch_process_error",
            request_id=request_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=500,
            detail=f"Batch processing failed: {e!s}",
        )


@router.get(
    "/documents/{processing_id}",
    response_model=ProcessResponse,
    summary="Get processing result",
    description="Retrieve the result of a previous processing request.",
)
async def get_processing_result(
    processing_id: Annotated[
        str,
        FastAPIPath(
            min_length=16,
            max_length=64,
            pattern=r"^[a-zA-Z0-9\-_]+$",
            description="Unique processing ID (alphanumeric, dashes, underscores only)",
        ),
    ],
    http_request: Request,
) -> ProcessResponse:
    """
    Get the result of a previous processing request.

    Args:
        processing_id: Unique processing ID (validated format).
        http_request: HTTP request object.

    Returns:
        Processing result.

    Raises:
        HTTPException: If processing ID not found or invalid format.
    """
    request_id = getattr(http_request.state, "request_id", "")

    # SECURITY: Additional validation for processing_id format
    if not re.match(r"^[a-zA-Z0-9\-_]{16,64}$", processing_id):
        raise HTTPException(
            status_code=400,
            detail="Invalid processing ID format",
        )

    logger.info(
        "get_result_request",
        request_id=request_id,
        processing_id=processing_id,
    )

    # This would typically retrieve from a database
    # For now, return a not found error
    raise HTTPException(
        status_code=404,
        detail=f"Processing result not found: {processing_id}",
    )


@router.post(
    "/documents/{processing_id}/reprocess",
    response_model=ProcessResponse | AsyncProcessResponse,
    summary="Reprocess a document",
    description="Reprocess a previously failed or completed document.",
)
async def reprocess_document(
    processing_id: str,
    http_request: Request,
    async_processing: bool = True,
) -> ProcessResponse | AsyncProcessResponse:
    """
    Reprocess a document.

    Args:
        processing_id: Original processing ID.
        http_request: HTTP request object.
        async_processing: Whether to process asynchronously.

    Returns:
        New processing result or task ID.

    Raises:
        HTTPException: If original processing not found.
    """
    request_id = getattr(http_request.state, "request_id", "")

    logger.info(
        "reprocess_request",
        request_id=request_id,
        processing_id=processing_id,
    )

    # This would typically retrieve the original request from a database
    # For now, return a not found error
    raise HTTPException(
        status_code=404,
        detail=f"Original processing not found: {processing_id}",
    )


@router.post(
    "/documents/preview",
    response_model=PreviewResponse,
    summary="Generate document preview",
    description="Generate a Markdown preview of extraction results.",
)
async def generate_preview(
    request: PreviewRequest,
    http_request: Request,
) -> PreviewResponse:
    """
    Generate a Markdown preview of extraction results.

    Creates a human-readable preview from stored extraction results.
    Useful for reviewing extractions before final export.

    Args:
        request: Preview request parameters.
        http_request: HTTP request object.

    Returns:
        Markdown formatted preview.

    Raises:
        HTTPException: If processing ID not found.
    """
    from src.export import MarkdownStyle, export_to_markdown

    request_id = getattr(http_request.state, "request_id", "")

    logger.info(
        "preview_request",
        request_id=request_id,
        processing_id=request.processing_id,
        style=request.style.value,
    )

    # Map preview style to markdown style
    style_map = {
        PreviewStyleEnum.SIMPLE: MarkdownStyle.SIMPLE,
        PreviewStyleEnum.DETAILED: MarkdownStyle.DETAILED,
        PreviewStyleEnum.SUMMARY: MarkdownStyle.SUMMARY,
        PreviewStyleEnum.TECHNICAL: MarkdownStyle.TECHNICAL,
    }
    md_style = style_map.get(request.style, MarkdownStyle.DETAILED)

    # Retrieve result from storage
    from src.storage import get_result

    stored_result = get_result(request.processing_id)

    if stored_result is None:
        logger.warning(
            "preview_not_found",
            request_id=request_id,
            processing_id=request.processing_id,
        )
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"No results found for processing ID: {request.processing_id}",
        )

    # Remove storage metadata before processing
    stored_result.pop("_storage_metadata", None)

    # Generate markdown preview
    content = export_to_markdown(
        stored_result,
        style=md_style,
        include_confidence_indicators=request.include_confidence,
        include_validation=request.include_validation,
        mask_phi=request.mask_phi,
    )

    return PreviewResponse(
        processing_id=request.processing_id,
        format="markdown",
        content=content,
        generated_at=datetime.now(UTC).isoformat(),
    )


@router.post(
    "/documents/{processing_id}/preview",
    response_model=PreviewResponse,
    summary="Preview specific document",
    description="Generate preview for a specific processing ID.",
)
async def preview_document(
    processing_id: str,
    http_request: Request,
    style: PreviewStyleEnum = PreviewStyleEnum.DETAILED,
    mask_phi: bool = False,
) -> PreviewResponse:
    """
    Generate preview for a specific document.

    Args:
        processing_id: Processing ID to preview.
        http_request: HTTP request object.
        style: Preview style.
        mask_phi: Whether to mask PHI fields.

    Returns:
        Markdown formatted preview.

    Raises:
        HTTPException: If processing ID not found.
    """
    request = PreviewRequest(
        processing_id=processing_id,
        style=style,
        mask_phi=mask_phi,
    )
    return await generate_preview(request, http_request)
