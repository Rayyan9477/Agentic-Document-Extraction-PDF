"""
Compatibility wrapper for pipeline runner.

Provides the run_extraction_pipeline function expected by API routes.
This module bridges the API layer with the PipelineRunner class.
"""

from pathlib import Path
from typing import Any

from src.pipeline.runner import PipelineRunner, extract_document, get_extraction_result
from src.pipeline.state import ExtractionState


def run_extraction_pipeline(
    pdf_path: str | Path,
    schema_name: str | None = None,
    enable_checkpointing: bool = False,
) -> dict[str, Any]:
    """
    Run the extraction pipeline on a PDF document.

    This is the main entry point for document extraction,
    designed for use by the API routes.

    Args:
        pdf_path: Path to the PDF file.
        schema_name: Optional schema name for extraction.
        enable_checkpointing: Whether to enable checkpointing.

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
    )

    # Convert TypedDict to regular dict for JSON serialization
    return dict(state)


__all__ = [
    "run_extraction_pipeline",
    "PipelineRunner",
    "extract_document",
    "get_extraction_result",
]
