"""
Document processing API routes.

Provides endpoints for sync and async document processing,
batch processing, and result retrieval.
"""

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fastapi import APIRouter, HTTPException, Request, BackgroundTasks

from src.api.models import (
    ProcessRequest,
    ProcessResponse,
    AsyncProcessResponse,
    BatchProcessRequest,
    BatchProcessResponse,
    BatchItemResult,
    FieldResult,
    ValidationResult,
    ProcessingMetadata,
    TaskStatusEnum,
    ConfidenceLevelEnum,
    ExportFormatEnum,
    PreviewRequest,
    PreviewResponse,
    PreviewStyleEnum,
)
from src.config import get_logger


logger = get_logger(__name__)
router = APIRouter()


def _map_confidence_level(confidence: float) -> ConfidenceLevelEnum:
    """Map confidence score to level."""
    if confidence >= 0.85:
        return ConfidenceLevelEnum.HIGH
    elif confidence >= 0.50:
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
    response_model=ProcessResponse | AsyncProcessResponse,
    summary="Process a document",
    description="Process a PDF document and extract structured data.",
)
async def process_document(
    request: ProcessRequest,
    http_request: Request,
) -> ProcessResponse | AsyncProcessResponse:
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

    # Validate file exists
    pdf_path = Path(request.pdf_path)
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
                from src.export import export_to_json, ExportFormat

                json_path = output_base.with_suffix(".json")
                export_to_json(
                    result,
                    output_path=json_path,
                    format=ExportFormat.DETAILED,
                )
                output_paths.append(str(json_path))

            if request.export_format in (
                ExportFormatEnum.EXCEL,
                ExportFormatEnum.BOTH,
                ExportFormatEnum.ALL,
            ):
                from src.export import export_to_excel

                excel_path = output_base.with_suffix(".xlsx")
                export_to_excel(
                    result,
                    output_path=excel_path,
                    mask_phi=request.mask_phi,
                )
                output_paths.append(str(excel_path))

            if request.export_format in (ExportFormatEnum.MARKDOWN, ExportFormatEnum.ALL):
                from src.export import export_to_markdown, MarkdownStyle

                md_path = output_base.with_suffix(".md")
                export_to_markdown(
                    result,
                    output_path=md_path,
                    style=MarkdownStyle.DETAILED,
                    mask_phi=request.mask_phi,
                )
                output_paths.append(str(md_path))

            output_path = "; ".join(output_paths) if output_paths else ""

        return _build_process_response(result, output_path)

    except Exception as e:
        logger.error(
            "document_process_error",
            request_id=request_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=500,
            detail=f"Processing failed: {str(e)}",
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

    # Validate all files exist
    missing_files = []
    for pdf_path in request.pdf_paths:
        if not Path(pdf_path).exists():
            missing_files.append(pdf_path)

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
        from src.pipeline.graph import run_extraction_pipeline
        from src.export import export_to_json, export_to_excel, ExportFormat

        started_at = datetime.now(timezone.utc)
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

                results.append(BatchItemResult(
                    pdf_path=pdf_path,
                    processing_id=result.get("processing_id", ""),
                    status=TaskStatusEnum.COMPLETED,
                    field_count=len(result.get("merged_extraction", {})),
                    overall_confidence=result.get("overall_confidence", 0.0),
                    output_path=output_path,
                    errors=[],
                ))
                successful += 1

            except Exception as e:
                results.append(BatchItemResult(
                    pdf_path=pdf_path,
                    processing_id="",
                    status=TaskStatusEnum.FAILED,
                    field_count=0,
                    overall_confidence=0.0,
                    output_path="",
                    errors=[str(e)],
                ))
                failed += 1

                if request.stop_on_error:
                    break

        completed_at = datetime.now(timezone.utc)
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
            detail=f"Batch processing failed: {str(e)}",
        )


@router.get(
    "/documents/{processing_id}",
    response_model=ProcessResponse,
    summary="Get processing result",
    description="Retrieve the result of a previous processing request.",
)
async def get_processing_result(
    processing_id: str,
    http_request: Request,
) -> ProcessResponse:
    """
    Get the result of a previous processing request.

    Args:
        processing_id: Unique processing ID.
        http_request: HTTP request object.

    Returns:
        Processing result.

    Raises:
        HTTPException: If processing ID not found.
    """
    request_id = getattr(http_request.state, "request_id", "")

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
    from src.export import export_to_markdown, MarkdownStyle

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

    # In production, this would retrieve from database
    # For demonstration, return a sample preview
    sample_state = {
        "processing_id": request.processing_id,
        "document_type": "CMS-1500",
        "status": "completed",
        "overall_confidence": 0.92,
        "merged_extraction": {
            "patient_name": {"value": "John Doe", "confidence": 0.95},
            "patient_dob": {"value": "01/15/1980", "confidence": 0.93},
            "diagnosis_code_1": {"value": "J06.9", "confidence": 0.91},
        },
        "field_metadata": {
            "patient_name": {
                "confidence": 0.95,
                "confidence_level": "high",
                "passes_agree": True,
                "validation_passed": True,
            },
            "patient_dob": {
                "confidence": 0.93,
                "confidence_level": "high",
                "passes_agree": True,
                "validation_passed": True,
            },
            "diagnosis_code_1": {
                "confidence": 0.91,
                "confidence_level": "high",
                "passes_agree": True,
                "validation_passed": True,
            },
        },
        "validation": {
            "is_valid": True,
            "field_validations": {},
            "cross_field_validations": [],
            "hallucination_flags": [],
            "warnings": [],
            "errors": [],
        },
        "start_time": datetime.now(timezone.utc).isoformat(),
        "end_time": datetime.now(timezone.utc).isoformat(),
        "total_vlm_calls": 4,
        "total_processing_time_ms": 25000,
        "retry_count": 0,
        "page_images": [b"page1"],
        "errors": [],
        "warnings": [],
    }

    content = export_to_markdown(
        sample_state,
        style=md_style,
        include_confidence_indicators=request.include_confidence,
        include_validation=request.include_validation,
        mask_phi=request.mask_phi,
    )

    return PreviewResponse(
        processing_id=request.processing_id,
        format="markdown",
        content=content,
        generated_at=datetime.now(timezone.utc).isoformat(),
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
