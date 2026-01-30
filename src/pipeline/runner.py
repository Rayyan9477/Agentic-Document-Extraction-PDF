"""
Pipeline runner for executing document extraction workflows.

Provides the main entry point for running extractions with:
- PDF preprocessing and image conversion
- Workflow execution with checkpointing
- Result formatting and export
- Error handling and recovery
"""

import base64
import hashlib
import io
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.agents.base import OrchestrationError
from src.agents.orchestrator import (
    OrchestratorAgent,
    create_extraction_workflow,
    generate_processing_id,
    generate_thread_id,
)
from src.client.lm_client import LMStudioClient
from src.config import get_logger, get_settings
from src.pipeline.state import (
    ExtractionState,
    ExtractionStatus,
    add_error,
    create_initial_state,
    set_status,
    update_state,
)


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
        enable_image_enhancement: bool | None = None,
    ) -> None:
        """
        Initialize the pipeline runner.

        Args:
            client: Optional pre-configured LM Studio client.
            enable_checkpointing: Whether to enable state checkpointing.
            max_retries: Maximum retry attempts for extraction.
            dpi: DPI for PDF to image conversion.
            max_image_dimension: Maximum image dimension for VLM.
            enable_image_enhancement: Whether to apply image enhancement
                (deskew, denoise, CLAHE) before VLM extraction. Defaults to
                settings value. Improves quality for scanned/faxed documents.
        """
        self._client = client or LMStudioClient()
        self._enable_checkpointing = enable_checkpointing
        self._max_retries = max_retries
        self._dpi = dpi
        self._max_image_dimension = max_image_dimension
        self._settings = get_settings()
        self._logger = get_logger("pipeline.runner")

        # Image enhancement (optional, improves scanned/faxed doc quality)
        self._enable_enhancement = (
            enable_image_enhancement
            if enable_image_enhancement is not None
            else getattr(self._settings, "image", None) is not None
        )
        self._enhancer = None  # Lazy initialized

        # Workflow components (lazy initialized)
        self._orchestrator: OrchestratorAgent | None = None
        self._compiled_workflow: Any = None

        self._logger.info(
            "pipeline_runner_initialized",
            checkpointing=enable_checkpointing,
            max_retries=max_retries,
            dpi=dpi,
            image_enhancement=self._enable_enhancement,
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
        start_time = datetime.now(UTC)
        pdf_path = state.get("pdf_path", "")

        self._logger.info("preprocessing_pdf", pdf_path=pdf_path)

        try:
            # Load and convert PDF
            page_images = self._load_and_convert_pdf(pdf_path)

            # Calculate PDF hash
            pdf_hash = self._calculate_file_hash(pdf_path)

            # Update state
            duration_ms = int((datetime.now(UTC) - start_time).total_seconds() * 1000)

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

                    # Apply image enhancement for scanned/faxed documents
                    png_bytes = self._enhance_image_bytes(png_bytes, page_num)

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

                    # Extract OCR text layer for hybrid vision+text approach
                    text_content = page.get_text("text").strip()

                    page_images.append(
                        {
                            "page_number": page_num,
                            "width": pix_width,
                            "height": pix_height,
                            "data_uri": data_uri,
                            "base64_encoded": base64_data,
                            "text_content": text_content,
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

                    # Apply image enhancement for scanned/faxed documents
                    png_bytes = self._enhance_image_bytes(png_bytes, page_num)

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

                    # Extract OCR text layer for hybrid vision+text approach
                    text_content = page.get_text("text").strip()

                    page_images.append(
                        {
                            "page_number": page_num,
                            "width": pix_width,
                            "height": pix_height,
                            "data_uri": data_uri,
                            "base64_encoded": base64_data,
                            "text_content": text_content,
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

    def _enhance_image_bytes(self, png_bytes: bytes, page_number: int) -> bytes:
        """
        Apply image enhancement (deskew, denoise, CLAHE) to raw PNG bytes.

        Falls back gracefully to original bytes on any error to avoid
        blocking the extraction pipeline.

        Args:
            png_bytes: Raw PNG image bytes.
            page_number: Page number for logging.

        Returns:
            Enhanced PNG bytes, or original bytes on failure.
        """
        if not self._enable_enhancement:
            return png_bytes

        try:
            import cv2
            import numpy as np

            # Decode PNG bytes to OpenCV image
            nparr = np.frombuffer(png_bytes, np.uint8)
            img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if img is None:
                return png_bytes

            # Lazy-initialize enhancer
            if self._enhancer is None:
                from src.preprocessing.image_enhancer import ImageEnhancer
                self._enhancer = ImageEnhancer()

            # Convert to grayscale for analysis
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Calculate Laplacian variance (sharpness) to decide if enhancement needed
            variance = cv2.Laplacian(gray, cv2.CV_64F).var()

            # Only enhance if image quality is below threshold (likely scanned/faxed)
            if variance > 500:  # High-quality renders don't need enhancement
                self._logger.debug(
                    "skipping_enhancement_high_quality",
                    page=page_number,
                    sharpness_variance=round(variance, 1),
                )
                return png_bytes

            self._logger.info(
                "enhancing_page_image",
                page=page_number,
                sharpness_variance=round(variance, 1),
            )

            # Apply CLAHE (adaptive contrast) on the luminance channel
            lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
            l_channel, a_channel, b_channel = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced_l = clahe.apply(l_channel)
            enhanced_lab = cv2.merge([enhanced_l, a_channel, b_channel])
            img = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)

            # Apply light denoising (preserves text edges)
            img = cv2.fastNlMeansDenoisingColored(img, None, 6, 6, 7, 21)

            # Encode back to PNG
            success, encoded = cv2.imencode(".png", img)
            if success:
                return encoded.tobytes()
            return png_bytes

        except ImportError:
            self._logger.debug("opencv_not_available_skipping_enhancement")
            self._enable_enhancement = False  # Don't try again
            return png_bytes
        except Exception as e:
            self._logger.warning(
                "image_enhancement_failed_using_original",
                page=page_number,
                error=str(e),
            )
            return png_bytes

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

    def extract_multi_record(
        self,
        pdf_path: str | Path,
        start_page: int | None = None,
        end_page: int | None = None,
    ) -> "DocumentExtractionResult":
        """
        Extract multiple records per page from a PDF document.

        Uses a streamlined pipeline that:
        1. Detects document type + entity type (1 VLM call)
        2. Generates adaptive schema (1 VLM call)
        3. Per page: detects record boundaries (1 VLM call)
        4. Per record: extracts fields (1 VLM call)

        This is ideal for documents with multiple entities per page
        (patient lists, invoice batches, employee rosters, etc.).

        Args:
            pdf_path: Path to the PDF file.
            start_page: Optional first page to process (1-indexed).
            end_page: Optional last page to process (1-indexed).

        Returns:
            DocumentExtractionResult with per-record data.

        Raises:
            FileNotFoundError: If PDF file not found.
        """
        from src.extraction.multi_record import (
            DocumentExtractionResult,
            MultiRecordExtractor,
        )

        pdf_path = Path(pdf_path)
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file not found: {pdf_path}")

        self._logger.info(
            "starting_multi_record_extraction",
            pdf_path=str(pdf_path),
            start_page=start_page,
            end_page=end_page,
        )

        # Load and convert PDF pages using existing infrastructure
        page_images = self._load_and_convert_pdf(str(pdf_path))

        # Run multi-record extraction
        extractor = MultiRecordExtractor(client=self._client)
        result = extractor.extract_document(
            page_images=page_images,
            pdf_path=str(pdf_path),
            start_page=start_page,
            end_page=end_page,
        )

        self._logger.info(
            "multi_record_extraction_complete",
            total_records=result.total_records,
            total_pages=result.total_pages,
            vlm_calls=result.total_vlm_calls,
        )

        return result


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
