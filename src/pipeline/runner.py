"""
Pipeline runner for executing document extraction workflows.

Provides the main entry point for running extractions with:
- PDF preprocessing and image conversion
- Workflow execution with checkpointing
- Result formatting and export
- Error handling and recovery
"""

from pathlib import Path
from typing import Any, BinaryIO
from datetime import datetime, timezone
import hashlib
import base64
import io

from src.config import get_logger, get_settings
from src.client.lm_client import LMStudioClient
from src.pipeline.state import (
    ExtractionState,
    ExtractionStatus,
    ConfidenceLevel,
    create_initial_state,
    update_state,
    set_status,
    add_error,
    serialize_state,
    deserialize_state,
)
from src.agents.orchestrator import (
    OrchestratorAgent,
    create_extraction_workflow,
    generate_processing_id,
    generate_thread_id,
)
from src.agents.base import OrchestrationError


logger = get_logger(__name__)


class PipelineRunner:
    """
    Main entry point for running document extraction pipelines.

    Handles:
    - PDF loading and preprocessing
    - Image conversion for VLM
    - Workflow execution
    - Checkpointing and recovery
    - Result formatting
    """

    def __init__(
        self,
        client: LMStudioClient | None = None,
        enable_checkpointing: bool = True,
        max_retries: int = 2,
        dpi: int = 200,
        max_image_dimension: int = 2048,
    ) -> None:
        """
        Initialize the pipeline runner.

        Args:
            client: Optional pre-configured LM Studio client.
            enable_checkpointing: Whether to enable state checkpointing.
            max_retries: Maximum retry attempts for extraction.
            dpi: DPI for PDF to image conversion.
            max_image_dimension: Maximum image dimension for VLM.
        """
        self._client = client or LMStudioClient()
        self._enable_checkpointing = enable_checkpointing
        self._max_retries = max_retries
        self._dpi = dpi
        self._max_image_dimension = max_image_dimension
        self._settings = get_settings()
        self._logger = get_logger("pipeline.runner")

        # Workflow components (lazy initialized)
        self._orchestrator: OrchestratorAgent | None = None
        self._compiled_workflow: Any = None

        self._logger.info(
            "pipeline_runner_initialized",
            checkpointing=enable_checkpointing,
            max_retries=max_retries,
            dpi=dpi,
        )

    def _ensure_workflow_initialized(self) -> None:
        """Ensure the workflow is built and compiled."""
        if self._orchestrator is None or self._compiled_workflow is None:
            self._orchestrator, self._compiled_workflow = create_extraction_workflow(
                preprocess_fn=self._preprocess_node,
                client=self._client,
                enable_checkpointing=self._enable_checkpointing,
                max_retries=self._max_retries,
            )

    def extract_from_pdf(
        self,
        pdf_path: str | Path,
        custom_schema: dict[str, Any] | None = None,
        thread_id: str | None = None,
    ) -> ExtractionState:
        """
        Extract data from a PDF file.

        Args:
            pdf_path: Path to the PDF file.
            custom_schema: Optional custom extraction schema.
            thread_id: Optional thread ID for checkpointing.

        Returns:
            Final extraction state with results.

        Raises:
            FileNotFoundError: If PDF file not found.
            OrchestrationError: If extraction fails.
        """
        pdf_path = Path(pdf_path)

        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        self._logger.info("starting_extraction", pdf_path=str(pdf_path))

        # Generate IDs
        processing_id = generate_processing_id()
        if thread_id is None and self._enable_checkpointing:
            thread_id = generate_thread_id(str(pdf_path), processing_id)

        # Create initial state
        initial_state = create_initial_state(
            pdf_path=str(pdf_path),
            custom_schema=custom_schema,
            processing_id=processing_id,
        )

        # Ensure workflow is ready
        self._ensure_workflow_initialized()

        # Run extraction
        assert self._orchestrator is not None
        final_state = self._orchestrator.run_extraction(
            initial_state=initial_state,
            thread_id=thread_id,
        )

        self._logger.info(
            "extraction_complete",
            processing_id=processing_id,
            status=final_state.get("status"),
            confidence=final_state.get("overall_confidence"),
        )

        return final_state

    def extract_from_bytes(
        self,
        pdf_bytes: bytes,
        filename: str = "document.pdf",
        custom_schema: dict[str, Any] | None = None,
    ) -> ExtractionState:
        """
        Extract data from PDF bytes.

        Args:
            pdf_bytes: PDF file content as bytes.
            filename: Optional filename for logging.
            custom_schema: Optional custom extraction schema.

        Returns:
            Final extraction state with results.
        """
        self._logger.info("starting_extraction_from_bytes", filename=filename)

        # Generate IDs
        processing_id = generate_processing_id()
        pdf_hash = hashlib.sha256(pdf_bytes).hexdigest()

        # Create initial state
        initial_state = create_initial_state(
            pdf_path=filename,
            custom_schema=custom_schema,
            processing_id=processing_id,
        )

        # Add PDF hash
        initial_state = update_state(initial_state, {"pdf_hash": pdf_hash})

        # Convert PDF bytes to images
        try:
            page_images = self._convert_pdf_bytes_to_images(pdf_bytes)
            initial_state = update_state(
                initial_state,
                {
                    "page_images": page_images,
                    "current_step": "preprocessed",
                },
            )
        except Exception as e:
            self._logger.error("pdf_conversion_failed", error=str(e))
            initial_state = add_error(initial_state, f"PDF conversion failed: {e}")
            initial_state = set_status(
                initial_state,
                ExtractionStatus.FAILED,
                "preprocessing_failed",
            )
            return initial_state

        # Ensure workflow is ready
        self._ensure_workflow_initialized()

        # Run extraction (skip preprocess node since we already have images)
        assert self._orchestrator is not None
        final_state = self._orchestrator.run_extraction(
            initial_state=initial_state,
            thread_id=None,  # No checkpointing for byte input
        )

        return final_state

    def resume_extraction(
        self,
        thread_id: str,
        human_corrections: dict[str, Any] | None = None,
    ) -> ExtractionState:
        """
        Resume a checkpointed extraction.

        Args:
            thread_id: Thread ID of the checkpointed extraction.
            human_corrections: Optional human-corrected field values.

        Returns:
            Final extraction state.

        Raises:
            OrchestrationError: If resume fails.
        """
        self._ensure_workflow_initialized()
        assert self._orchestrator is not None

        # Get current checkpoint state
        checkpoint_state = self._orchestrator.get_checkpoint_state(thread_id)

        if checkpoint_state is None:
            raise OrchestrationError(
                f"No checkpoint found for thread: {thread_id}",
                agent_name="runner",
                recoverable=False,
            )

        self._logger.info(
            "resuming_extraction",
            thread_id=thread_id,
            current_status=checkpoint_state.get("status"),
        )

        # Apply human corrections if provided
        if human_corrections:
            checkpoint_state = self._apply_human_corrections(
                checkpoint_state,
                human_corrections,
            )

        return self._orchestrator.resume_extraction(
            thread_id=thread_id,
            updated_state=checkpoint_state,
        )

    def get_checkpoint_status(self, thread_id: str) -> dict[str, Any] | None:
        """
        Get the status of a checkpointed extraction.

        Args:
            thread_id: Thread ID to check.

        Returns:
            Status dictionary or None if not found.
        """
        self._ensure_workflow_initialized()
        assert self._orchestrator is not None

        state = self._orchestrator.get_checkpoint_state(thread_id)

        if state is None:
            return None

        return {
            "thread_id": thread_id,
            "processing_id": state.get("processing_id"),
            "status": state.get("status"),
            "current_step": state.get("current_step"),
            "overall_confidence": state.get("overall_confidence"),
            "retry_count": state.get("retry_count"),
            "errors": state.get("errors", []),
            "warnings": state.get("warnings", []),
        }

    def _preprocess_node(self, state: ExtractionState) -> ExtractionState:
        """
        Preprocess node for the workflow.

        Loads PDF and converts to images.

        Args:
            state: Initial extraction state.

        Returns:
            Updated state with page images.
        """
        start_time = datetime.now(timezone.utc)
        pdf_path = state.get("pdf_path", "")

        self._logger.info("preprocessing_pdf", pdf_path=pdf_path)

        try:
            # Load and convert PDF
            page_images = self._load_and_convert_pdf(pdf_path)

            # Calculate PDF hash
            pdf_hash = self._calculate_file_hash(pdf_path)

            # Update state
            duration_ms = int(
                (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            )

            state = update_state(
                state,
                {
                    "page_images": page_images,
                    "pdf_hash": pdf_hash,
                    "status": ExtractionStatus.ANALYZING.value,
                    "current_step": "preprocessed",
                    "preprocessing_ms": duration_ms,
                },
            )

            self._logger.info(
                "preprocessing_complete",
                page_count=len(page_images),
                duration_ms=duration_ms,
            )

            return state

        except Exception as e:
            self._logger.error("preprocessing_failed", error=str(e))
            state = add_error(state, f"Preprocessing failed: {e}")
            state = set_status(
                state,
                ExtractionStatus.FAILED,
                "preprocessing_failed",
            )
            return state

    def _load_and_convert_pdf(self, pdf_path: str) -> list[dict[str, Any]]:
        """
        Load PDF and convert pages to images.

        Args:
            pdf_path: Path to the PDF file.

        Returns:
            List of page image dictionaries.
        """
        try:
            import fitz  # PyMuPDF
        except ImportError as e:
            raise ImportError(
                "PyMuPDF is required for PDF processing. Install with: pip install pymupdf"
            ) from e

        page_images: list[dict[str, Any]] = []

        with fitz.open(pdf_path) as doc:
            for page_num, page in enumerate(doc, start=1):
                # Calculate scaling for target DPI
                scale = self._dpi / 72.0  # 72 is default PDF DPI
                matrix = fitz.Matrix(scale, scale)

                # Render page to pixmap
                pixmap = page.get_pixmap(matrix=matrix)

                try:
                    # Convert to PNG bytes
                    png_bytes = pixmap.tobytes("png")

                    # Store dimensions before releasing pixmap
                    pix_width = pixmap.width
                    pix_height = pixmap.height

                    # Check if resizing needed
                    if (
                        pix_width > self._max_image_dimension
                        or pix_height > self._max_image_dimension
                    ):
                        png_bytes = self._resize_image(
                            png_bytes,
                            self._max_image_dimension,
                        )

                    # Encode to base64
                    base64_data = base64.b64encode(png_bytes).decode("utf-8")
                    data_uri = f"data:image/png;base64,{base64_data}"

                    page_images.append(
                        {
                            "page_number": page_num,
                            "width": pix_width,
                            "height": pix_height,
                            "data_uri": data_uri,
                            "base64_encoded": base64_data,
                        }
                    )
                finally:
                    # Release PyMuPDF pixmap native memory
                    # Pixmaps hold native memory that must be explicitly freed
                    pixmap = None

        return page_images

    def _convert_pdf_bytes_to_images(
        self,
        pdf_bytes: bytes,
    ) -> list[dict[str, Any]]:
        """
        Convert PDF bytes to page images.

        Args:
            pdf_bytes: PDF file content.

        Returns:
            List of page image dictionaries.
        """
        try:
            import fitz  # PyMuPDF
        except ImportError as e:
            raise ImportError(
                "PyMuPDF is required for PDF processing. Install with: pip install pymupdf"
            ) from e

        page_images: list[dict[str, Any]] = []

        with fitz.open(stream=pdf_bytes, filetype="pdf") as doc:
            for page_num, page in enumerate(doc, start=1):
                # Calculate scaling for target DPI
                scale = self._dpi / 72.0
                matrix = fitz.Matrix(scale, scale)

                # Render page to pixmap
                pixmap = page.get_pixmap(matrix=matrix)

                try:
                    # Convert to PNG bytes
                    png_bytes = pixmap.tobytes("png")

                    # Store dimensions before releasing pixmap
                    pix_width = pixmap.width
                    pix_height = pixmap.height

                    # Check if resizing needed
                    if (
                        pix_width > self._max_image_dimension
                        or pix_height > self._max_image_dimension
                    ):
                        png_bytes = self._resize_image(
                            png_bytes,
                            self._max_image_dimension,
                        )

                    # Encode to base64
                    base64_data = base64.b64encode(png_bytes).decode("utf-8")
                    data_uri = f"data:image/png;base64,{base64_data}"

                    page_images.append(
                        {
                            "page_number": page_num,
                            "width": pix_width,
                            "height": pix_height,
                            "data_uri": data_uri,
                            "base64_encoded": base64_data,
                        }
                    )
                finally:
                    # Release PyMuPDF pixmap native memory
                    pixmap = None

        return page_images

    def _resize_image(self, png_bytes: bytes, max_dimension: int) -> bytes:
        """
        Resize image to fit within max dimension.

        Args:
            png_bytes: PNG image bytes.
            max_dimension: Maximum dimension (width or height).

        Returns:
            Resized PNG bytes.
        """
        try:
            from PIL import Image
        except ImportError as e:
            raise ImportError(
                "Pillow is required for image resizing. Install with: pip install pillow"
            ) from e

        img = Image.open(io.BytesIO(png_bytes))
        img_resized = None
        try:
            # Calculate new size maintaining aspect ratio
            width, height = img.size
            if width > height:
                new_width = max_dimension
                new_height = int(height * max_dimension / width)
            else:
                new_height = max_dimension
                new_width = int(width * max_dimension / height)

            # Resize with high quality
            img_resized = img.resize((new_width, new_height), Image.Resampling.LANCZOS)

            # Convert back to bytes
            output = io.BytesIO()
            img_resized.save(output, format="PNG", optimize=True)
            return output.getvalue()
        finally:
            # Close PIL images to prevent memory/handle leak
            if img_resized is not None:
                img_resized.close()
            img.close()

    def _calculate_file_hash(self, file_path: str) -> str:
        """
        Calculate SHA-256 hash of a file.

        Args:
            file_path: Path to the file.

        Returns:
            Hex digest of file hash.
        """
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256_hash.update(chunk)
        return sha256_hash.hexdigest()

    def _apply_human_corrections(
        self,
        state: ExtractionState,
        corrections: dict[str, Any],
    ) -> ExtractionState:
        """
        Apply human corrections to extraction state.

        Args:
            state: Current extraction state.
            corrections: Dictionary of field corrections.

        Returns:
            Updated state with corrections applied.
        """
        merged_extraction = dict(state.get("merged_extraction", {}))

        for field_name, corrected_value in corrections.items():
            if field_name in merged_extraction:
                # Update the field value
                if isinstance(merged_extraction[field_name], dict):
                    merged_extraction[field_name]["value"] = corrected_value
                    merged_extraction[field_name]["confidence"] = 1.0
                    merged_extraction[field_name]["human_corrected"] = True
                else:
                    merged_extraction[field_name] = {
                        "value": corrected_value,
                        "confidence": 1.0,
                        "human_corrected": True,
                    }
            else:
                # Add new field
                merged_extraction[field_name] = {
                    "value": corrected_value,
                    "confidence": 1.0,
                    "human_corrected": True,
                }

        # Update state
        state = update_state(
            state,
            {
                "merged_extraction": merged_extraction,
                "status": ExtractionStatus.COMPLETED.value,
                "current_step": "human_corrected",
            },
        )

        return state


def extract_document(
    pdf_path: str | Path,
    custom_schema: dict[str, Any] | None = None,
    enable_checkpointing: bool = True,
) -> ExtractionState:
    """
    Convenience function for simple document extraction.

    Args:
        pdf_path: Path to the PDF file.
        custom_schema: Optional custom extraction schema.
        enable_checkpointing: Whether to enable checkpointing.

    Returns:
        Final extraction state.
    """
    runner = PipelineRunner(enable_checkpointing=enable_checkpointing)
    return runner.extract_from_pdf(pdf_path, custom_schema=custom_schema)


def get_extraction_result(state: ExtractionState) -> dict[str, Any]:
    """
    Extract the main result data from an extraction state.

    Args:
        state: Final extraction state.

    Returns:
        Dictionary with extraction results.
    """
    return {
        "success": state.get("status") == ExtractionStatus.COMPLETED.value,
        "document_type": state.get("document_type"),
        "schema_name": state.get("selected_schema_name"),
        "fields": state.get("merged_extraction", {}),
        "confidence": state.get("overall_confidence", 0.0),
        "confidence_level": state.get("confidence_level"),
        "requires_review": state.get("status") == ExtractionStatus.HUMAN_REVIEW.value,
        "errors": state.get("errors", []),
        "warnings": state.get("warnings", []),
        "processing_id": state.get("processing_id"),
        "processing_time_ms": state.get("total_processing_ms", 0),
    }
