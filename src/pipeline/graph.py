"""
Compatibility wrapper for pipeline runner.

Provides the run_extraction_pipeline function expected by API routes.
This module bridges the API layer with the PipelineRunner class.
"""

from pathlib import Path
from typing import Any

from src.pipeline.runner import PipelineRunner, extract_document, get_extraction_result


def run_extraction_pipeline(
    pdf_path: str | Path,
    schema_name: str | None = None,
    enable_checkpointing: bool = False,
    *,
    profile_override: str | None = None,
    modality_override: list[str] | None = None,
) -> dict[str, Any]:
    """
    Run the extraction pipeline on a PDF document.

    This is the main entry point for document extraction,
    designed for use by the API routes.

    Args:
        pdf_path: Path to the PDF file.
        schema_name: Optional schema name for extraction.
        enable_checkpointing: Whether to enable checkpointing.
        profile_override: Phase K — explicit profile id (e.g.
            ``"medical-rcm"``, ``"generic-document"``). Bypasses the
            analyzer's auto-detection. None = auto-detect.
        modality_override: Phase 5 — explicit modality list. Empty
            list / None = auto-detect.

    Returns:
        Dictionary containing extraction results and state.
    """
    runner = PipelineRunner(enable_checkpointing=enable_checkpointing)

    # Build custom schema hint if schema_name provided
    custom_schema = None
    if schema_name:
        custom_schema = {"schema_name": schema_name}

    # Run extraction
    state = runner.extract_from_pdf(
        pdf_path=pdf_path,
        custom_schema=custom_schema,
        profile_override=profile_override,
        modality_override=modality_override,
    )

    # Convert TypedDict to regular dict for JSON serialization
    return dict(state)


__all__ = [
    "PipelineRunner",
    "extract_document",
    "get_extraction_result",
    "run_extraction_pipeline",
]
