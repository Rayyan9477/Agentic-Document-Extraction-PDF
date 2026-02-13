"""
DICOM file processor for medical imaging files.

Converts DICOM images into PageImage objects with extracted metadata,
supporting common modalities (X-ray, CT, MRI, Ultrasound).

Requires: pydicom>=2.4.0
"""

import io
import secrets
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from PIL import Image

from src.config import get_logger
from src.preprocessing.base_processor import (
    BaseFileProcessor,
    FileProcessingError,
    FileValidationError,
)
from src.preprocessing.pdf_processor import (
    DocumentOrientation,
    PDFMetadata,
    PageImage,
    ProcessingResult,
)


logger = get_logger(__name__)

MAX_FILE_SIZE_MB = 500  # DICOM files can be large


class DicomProcessor(BaseFileProcessor):
    """
    Process DICOM medical imaging files into PageImage objects.

    Extracts pixel data and patient/study metadata from DICOM files.
    Multi-frame DICOM files produce multiple PageImage objects.
    """

    def __init__(self, dpi: int = 300) -> None:
        self._dpi = dpi

    def validate(self, file_path: Path) -> None:
        """Validate DICOM file."""
        if not file_path.exists():
            raise FileValidationError(f"File not found: {file_path}")

        suffix = file_path.suffix.lower()
        if suffix not in (".dcm", ".dicom"):
            raise FileValidationError(f"Not a DICOM file: {suffix}")

        size_mb = file_path.stat().st_size / (1024 * 1024)
        if size_mb > MAX_FILE_SIZE_MB:
            raise FileValidationError(
                f"File size {size_mb:.1f} MB exceeds limit of {MAX_FILE_SIZE_MB} MB"
            )

    def process(self, file_path: Path) -> ProcessingResult:
        """Convert DICOM to page images with metadata extraction."""
        start_time = time.monotonic()
        self.validate(file_path)

        try:
            import pydicom
            from pydicom.pixel_data_handlers.util import apply_voi_lut
        except ImportError as e:
            raise FileProcessingError(
                "pydicom is required for DICOM processing. "
                "Install with: pip install pydicom>=2.4.0"
            ) from e

        file_hash = self._compute_file_hash(file_path)
        processing_id = f"dcm_{secrets.token_hex(8)}"
        warnings: list[str] = []

        ds = pydicom.dcmread(str(file_path))

        # Extract metadata
        dicom_meta = self._extract_dicom_metadata(ds)

        # Convert pixel data to images
        pages = self._convert_pixel_data(ds, warnings)

        if not pages:
            warnings.append("No pixel data found in DICOM file")

        processing_time_ms = int((time.monotonic() - start_time) * 1000)

        metadata = PDFMetadata(
            file_path=file_path,
            file_name=file_path.name,
            file_size_bytes=file_path.stat().st_size,
            file_hash=file_hash,
            page_count=len(pages),
            title=dicom_meta.get("study_description"),
            author=dicom_meta.get("referring_physician"),
            subject=dicom_meta.get("modality"),
            keywords=dicom_meta.get("body_part"),
            creator="DicomProcessor",
            producer="pydicom",
            creation_date=datetime.now(UTC),
            modification_date=None,
            pdf_version="N/A",
            is_encrypted=False,
            has_forms=False,
            has_annotations=False,
            processing_id=processing_id,
        )

        logger.info(
            "dicom_processing_complete",
            file=file_path.name,
            pages=len(pages),
            modality=dicom_meta.get("modality", "unknown"),
            time_ms=processing_time_ms,
        )

        return ProcessingResult(
            metadata=metadata,
            pages=pages,
            processing_time_ms=processing_time_ms,
            warnings=warnings,
        )

    def _extract_dicom_metadata(self, ds: Any) -> dict[str, str | None]:
        """Extract relevant metadata from DICOM dataset."""
        def safe_str(tag: str) -> str | None:
            val = getattr(ds, tag, None)
            return str(val).strip() if val else None

        return {
            "patient_name": safe_str("PatientName"),
            "patient_id": safe_str("PatientID"),
            "study_description": safe_str("StudyDescription"),
            "series_description": safe_str("SeriesDescription"),
            "modality": safe_str("Modality"),
            "body_part": safe_str("BodyPartExamined"),
            "referring_physician": safe_str("ReferringPhysicianName"),
            "institution": safe_str("InstitutionName"),
            "study_date": safe_str("StudyDate"),
            "accession_number": safe_str("AccessionNumber"),
        }

    def _convert_pixel_data(
        self, ds: Any, warnings: list[str],
    ) -> list[PageImage]:
        """Convert DICOM pixel data to PageImage objects."""
        pages: list[PageImage] = []

        if not hasattr(ds, "pixel_array"):
            return pages

        try:
            from pydicom.pixel_data_handlers.util import apply_voi_lut
            pixel_array = apply_voi_lut(ds.pixel_array, ds)
        except Exception:
            try:
                pixel_array = ds.pixel_array
            except Exception as e:
                warnings.append(f"Could not extract pixel data: {e}")
                return pages

        import numpy as np

        # Handle multi-frame DICOM
        if len(pixel_array.shape) == 3 and pixel_array.shape[2] not in (3, 4):
            # Multi-frame: shape is (frames, height, width)
            for frame_idx in range(pixel_array.shape[0]):
                frame = pixel_array[frame_idx]
                page = self._array_to_page_image(frame, frame_idx + 1)
                if page:
                    pages.append(page)
        else:
            # Single frame
            page = self._array_to_page_image(pixel_array, 1)
            if page:
                pages.append(page)

        return pages

    def _array_to_page_image(
        self, pixel_array: Any, page_number: int,
    ) -> PageImage | None:
        """Convert a numpy array to a PageImage."""
        import numpy as np

        try:
            # Normalize to 8-bit range
            arr = pixel_array.astype(np.float64)
            arr_min, arr_max = arr.min(), arr.max()
            if arr_max > arr_min:
                arr = ((arr - arr_min) / (arr_max - arr_min) * 255).astype(np.uint8)
            else:
                arr = np.zeros_like(arr, dtype=np.uint8)

            # Convert to PIL Image
            if len(arr.shape) == 2:
                img = Image.fromarray(arr, mode="L").convert("RGB")
            elif len(arr.shape) == 3 and arr.shape[2] == 3:
                img = Image.fromarray(arr, mode="RGB")
            else:
                img = Image.fromarray(arr[:, :, 0] if len(arr.shape) == 3 else arr, mode="L").convert("RGB")

            width, height = img.size

            buf = io.BytesIO()
            img.save(buf, format="PNG")
            png_bytes = buf.getvalue()

            orientation = self._detect_orientation(width, height)

            return PageImage(
                page_number=page_number,
                image_bytes=png_bytes,
                width=width,
                height=height,
                dpi=self._dpi,
                orientation=orientation,
                original_width_pts=width * 72.0 / self._dpi,
                original_height_pts=height * 72.0 / self._dpi,
                has_text=False,
                has_images=True,
                rotation=0,
                text_content="",
            )
        except Exception as e:
            logger.warning("dicom_frame_conversion_failed", page=page_number, error=str(e))
            return None
