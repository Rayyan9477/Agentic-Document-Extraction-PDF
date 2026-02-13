"""
Image file processor for PNG, JPG, TIFF, and BMP files.

Converts standalone image files into PageImage objects for
uniform downstream processing through the extraction pipeline.
"""

import base64
import io
import secrets
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image

from src.config import get_logger, get_settings
from src.preprocessing.base_processor import (
    BaseFileProcessor,
    FileValidationError,
)
from src.preprocessing.pdf_processor import (
    DocumentOrientation,
    PDFMetadata,
    PageImage,
    ProcessingResult,
)


logger = get_logger(__name__)

# Maximum image dimensions before resizing
MAX_IMAGE_DIMENSION = 4096
MAX_FILE_SIZE_MB = 50
SUPPORTED_MODES = {"RGB", "RGBA", "L", "1", "P", "CMYK"}


class ImageProcessor(BaseFileProcessor):
    """
    Process image files (PNG, JPG, TIFF, BMP) into PageImage objects.

    TIFF files may contain multiple pages/frames, each converted to a separate PageImage.
    Single-page image formats produce a single PageImage.
    """

    def __init__(self, dpi: int = 300, max_dimension: int = MAX_IMAGE_DIMENSION) -> None:
        self._dpi = dpi
        self._max_dimension = max_dimension

    def validate(self, file_path: Path) -> None:
        """Validate image file exists and is a supported format."""
        if not file_path.exists():
            raise FileValidationError(f"File not found: {file_path}")

        if not file_path.is_file():
            raise FileValidationError(f"Not a file: {file_path}")

        size_mb = file_path.stat().st_size / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            raise FileValidationError(
                f"File size {size_mb:.1f} MB exceeds limit of {MAX_FILE_SIZE_MB} MB"
            )

        try:
            with Image.open(file_path) as img:
                img.verify()
        except Exception as e:
            raise FileValidationError(f"Invalid or corrupted image: {e}") from e

    def process(self, file_path: Path) -> ProcessingResult:
        """Convert image file to ProcessingResult with PageImage(s)."""
        start_time = time.monotonic()

        self.validate(file_path)

        file_hash = self._compute_file_hash(file_path)
        processing_id = f"img_{secrets.token_hex(8)}"
        pages: list[PageImage] = []
        warnings: list[str] = []

        with Image.open(file_path) as img:
            # Handle multi-frame images (TIFF can have multiple pages)
            n_frames = getattr(img, "n_frames", 1)

            for frame_idx in range(n_frames):
                if n_frames > 1:
                    img.seek(frame_idx)

                page_img = self._convert_frame_to_page_image(
                    img, page_number=frame_idx + 1, warnings=warnings,
                )
                pages.append(page_img)

        processing_time_ms = int((time.monotonic() - start_time) * 1000)

        metadata = PDFMetadata(
            file_path=file_path,
            file_name=file_path.name,
            file_size_bytes=file_path.stat().st_size,
            file_hash=file_hash,
            page_count=len(pages),
            title=None,
            author=None,
            subject=None,
            keywords=None,
            creator=f"ImageProcessor ({file_path.suffix})",
            producer="PIL/Pillow",
            creation_date=datetime.now(UTC),
            modification_date=None,
            pdf_version="N/A",
            is_encrypted=False,
            has_forms=False,
            has_annotations=False,
            processing_id=processing_id,
        )

        logger.info(
            "image_processing_complete",
            file=file_path.name,
            pages=len(pages),
            time_ms=processing_time_ms,
        )

        return ProcessingResult(
            metadata=metadata,
            pages=pages,
            processing_time_ms=processing_time_ms,
            warnings=warnings,
        )

    def _convert_frame_to_page_image(
        self,
        img: Image.Image,
        page_number: int,
        warnings: list[str],
    ) -> PageImage:
        """Convert a single image/frame to a PageImage."""
        # Convert to RGB if needed
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")

        original_width, original_height = img.size

        # Resize if too large
        if original_width > self._max_dimension or original_height > self._max_dimension:
            ratio = min(
                self._max_dimension / original_width,
                self._max_dimension / original_height,
            )
            new_size = (int(original_width * ratio), int(original_height * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
            warnings.append(
                f"Page {page_number}: Resized from {original_width}x{original_height} "
                f"to {new_size[0]}x{new_size[1]}"
            )

        # Convert to PNG bytes
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()

        width, height = img.size
        orientation = self._detect_orientation(width, height)

        return PageImage(
            page_number=page_number,
            image_bytes=png_bytes,
            width=width,
            height=height,
            dpi=self._dpi,
            orientation=orientation,
            original_width_pts=original_width * 72.0 / self._dpi,
            original_height_pts=original_height * 72.0 / self._dpi,
            has_text=False,
            has_images=True,
            rotation=0,
            text_content="",
        )
