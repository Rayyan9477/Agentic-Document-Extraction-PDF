"""
PDF processing module using PyMuPDF.

Handles PDF validation, metadata extraction, and high-quality page-to-image
conversion at configurable DPI for VLM processing.
"""

import base64
import hashlib
import io
import secrets
import shutil
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Generator, Iterator

import fitz  # PyMuPDF
from PIL import Image

from src.config import get_logger, get_settings


logger = get_logger(__name__)


class PDFProcessingError(Exception):
    """Base exception for PDF processing errors."""

    pass


class PDFValidationError(PDFProcessingError):
    """Raised when PDF validation fails."""

    pass


class PDFEncryptionError(PDFProcessingError):
    """Raised when PDF is encrypted and cannot be processed."""

    pass


class PDFCorruptionError(PDFProcessingError):
    """Raised when PDF file is corrupted."""

    pass


class PDFSizeError(PDFProcessingError):
    """Raised when PDF exceeds size limits."""

    pass


class PDFPageLimitError(PDFProcessingError):
    """Raised when PDF exceeds page limits."""

    pass


class DocumentOrientation(str, Enum):
    """Document page orientation."""

    PORTRAIT = "portrait"
    LANDSCAPE = "landscape"
    SQUARE = "square"


@dataclass(frozen=True, slots=True)
class PDFMetadata:
    """
    Immutable container for PDF document metadata.

    Attributes:
        file_path: Original file path.
        file_name: Original file name.
        file_size_bytes: File size in bytes.
        file_hash: SHA-256 hash of file contents.
        page_count: Total number of pages.
        title: Document title from metadata.
        author: Document author from metadata.
        subject: Document subject from metadata.
        keywords: Document keywords from metadata.
        creator: PDF creator application.
        producer: PDF producer application.
        creation_date: Document creation date.
        modification_date: Document modification date.
        pdf_version: PDF specification version.
        is_encrypted: Whether document is encrypted.
        has_forms: Whether document contains forms.
        has_annotations: Whether document contains annotations.
        processing_id: Unique identifier for this processing session.
    """

    file_path: Path
    file_name: str
    file_size_bytes: int
    file_hash: str
    page_count: int
    title: str | None
    author: str | None
    subject: str | None
    keywords: str | None
    creator: str | None
    producer: str | None
    creation_date: datetime | None
    modification_date: datetime | None
    pdf_version: str
    is_encrypted: bool
    has_forms: bool
    has_annotations: bool
    processing_id: str

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to dictionary representation."""
        return {
            "file_path": str(self.file_path),
            "file_name": self.file_name,
            "file_size_bytes": self.file_size_bytes,
            "file_hash": self.file_hash,
            "page_count": self.page_count,
            "title": self.title,
            "author": self.author,
            "subject": self.subject,
            "keywords": self.keywords,
            "creator": self.creator,
            "producer": self.producer,
            "creation_date": self.creation_date.isoformat() if self.creation_date else None,
            "modification_date": (
                self.modification_date.isoformat() if self.modification_date else None
            ),
            "pdf_version": self.pdf_version,
            "is_encrypted": self.is_encrypted,
            "has_forms": self.has_forms,
            "has_annotations": self.has_annotations,
            "processing_id": self.processing_id,
        }


@dataclass(frozen=True, slots=True)
class PageImage:
    """
    Immutable container for a processed page image.

    Attributes:
        page_number: One-indexed page number.
        image_bytes: PNG image data as bytes.
        width: Image width in pixels.
        height: Image height in pixels.
        dpi: Resolution in dots per inch.
        orientation: Page orientation.
        original_width_pts: Original page width in points.
        original_height_pts: Original page height in points.
        has_text: Whether page contains extractable text.
        has_images: Whether page contains embedded images.
        rotation: Page rotation in degrees.
    """

    page_number: int
    image_bytes: bytes
    width: int
    height: int
    dpi: int
    orientation: DocumentOrientation
    original_width_pts: float
    original_height_pts: float
    has_text: bool
    has_images: bool
    rotation: int

    @property
    def base64_encoded(self) -> str:
        """Get base64-encoded image data for API transmission."""
        return base64.b64encode(self.image_bytes).decode("utf-8")

    @property
    def data_uri(self) -> str:
        """Get data URI for embedding in HTML or API requests."""
        return f"data:image/png;base64,{self.base64_encoded}"

    @property
    def size_kb(self) -> float:
        """Get image size in kilobytes."""
        return len(self.image_bytes) / 1024

    def to_pil_image(self) -> Image.Image:
        """Convert to PIL Image for further processing."""
        return Image.open(io.BytesIO(self.image_bytes))

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation (without raw bytes)."""
        return {
            "page_number": self.page_number,
            "width": self.width,
            "height": self.height,
            "dpi": self.dpi,
            "orientation": self.orientation.value,
            "original_width_pts": self.original_width_pts,
            "original_height_pts": self.original_height_pts,
            "has_text": self.has_text,
            "has_images": self.has_images,
            "rotation": self.rotation,
            "size_kb": self.size_kb,
        }


@dataclass(slots=True)
class ProcessingResult:
    """
    Container for complete PDF processing results.

    Attributes:
        metadata: PDF document metadata.
        pages: List of processed page images.
        processing_time_ms: Total processing time in milliseconds.
        warnings: List of non-fatal warnings encountered.
        temp_dir: Temporary directory used for processing (for cleanup).
    """

    metadata: PDFMetadata
    pages: list[PageImage] = field(default_factory=list)
    processing_time_ms: int = 0
    warnings: list[str] = field(default_factory=list)
    temp_dir: Path | None = None

    @property
    def page_count(self) -> int:
        """Get number of processed pages."""
        return len(self.pages)

    @property
    def total_size_kb(self) -> float:
        """Get total size of all page images in KB."""
        return sum(page.size_kb for page in self.pages)

    def get_page(self, page_number: int) -> PageImage | None:
        """
        Get a specific page by number.

        Args:
            page_number: One-indexed page number.

        Returns:
            PageImage if found, None otherwise.
        """
        for page in self.pages:
            if page.page_number == page_number:
                return page
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "metadata": self.metadata.to_dict(),
            "pages": [page.to_dict() for page in self.pages],
            "processing_time_ms": self.processing_time_ms,
            "warnings": self.warnings,
            "page_count": self.page_count,
            "total_size_kb": self.total_size_kb,
        }


class PDFProcessor:
    """
    High-performance PDF processor using PyMuPDF.

    Handles PDF validation, metadata extraction, and conversion of pages
    to high-quality images suitable for VLM processing.

    Example:
        processor = PDFProcessor()
        result = processor.process(Path("document.pdf"))
        for page in result.pages:
            # Use page.base64_encoded for VLM API calls
            pass
    """

    def __init__(
        self,
        dpi: int | None = None,
        max_pages: int | None = None,
        max_file_size_mb: int | None = None,
        output_format: str | None = None,
        temp_dir: Path | None = None,
    ) -> None:
        """
        Initialize the PDF processor.

        Args:
            dpi: Resolution for page rendering. Defaults to settings value.
            max_pages: Maximum pages to process. Defaults to settings value.
            max_file_size_mb: Maximum file size in MB. Defaults to settings value.
            output_format: Image output format. Defaults to settings value.
            temp_dir: Temporary directory for processing. Defaults to settings value.
        """
        settings = get_settings()

        self._dpi = dpi or settings.pdf.dpi
        self._max_pages = max_pages or settings.pdf.max_pages
        self._max_file_size_bytes = (
            max_file_size_mb * 1024 * 1024
            if max_file_size_mb
            else settings.pdf.max_file_size_bytes
        )
        self._output_format = output_format or settings.pdf.output_format.value
        self._temp_dir = temp_dir or settings.pdf.temp_dir

        # Calculate zoom factor for target DPI (72 is PDF base DPI)
        self._zoom = self._dpi / 72.0
        self._matrix = fitz.Matrix(self._zoom, self._zoom)

        logger.debug(
            "pdf_processor_initialized",
            dpi=self._dpi,
            max_pages=self._max_pages,
            max_file_size_mb=self._max_file_size_bytes / (1024 * 1024),
            output_format=self._output_format,
        )

    def validate(self, file_path: Path) -> None:
        """
        Validate a PDF file before processing.

        Args:
            file_path: Path to the PDF file.

        Raises:
            PDFValidationError: If file doesn't exist or isn't a PDF.
            PDFSizeError: If file exceeds size limit.
            PDFCorruptionError: If file is corrupted.
            PDFEncryptionError: If file is encrypted.
            PDFPageLimitError: If file exceeds page limit.
        """
        # Check file exists
        if not file_path.exists():
            raise PDFValidationError(f"File not found: {file_path}")

        # Check file extension
        if file_path.suffix.lower() != ".pdf":
            raise PDFValidationError(
                f"Invalid file extension: {file_path.suffix}. Expected .pdf"
            )

        # Check file size
        file_size = file_path.stat().st_size
        if file_size > self._max_file_size_bytes:
            raise PDFSizeError(
                f"File size ({file_size / (1024 * 1024):.2f} MB) exceeds "
                f"limit ({self._max_file_size_bytes / (1024 * 1024):.2f} MB)"
            )

        # Check file magic bytes
        with open(file_path, "rb") as f:
            header = f.read(8)
            if not header.startswith(b"%PDF-"):
                raise PDFCorruptionError(
                    f"Invalid PDF header. File may be corrupted or not a PDF."
                )

        # Try to open and validate with PyMuPDF
        try:
            doc = fitz.open(file_path)
        except fitz.FileDataError as e:
            raise PDFCorruptionError(f"PDF file is corrupted: {e}") from e
        except Exception as e:
            raise PDFProcessingError(f"Failed to open PDF: {e}") from e

        try:
            # Check encryption
            if doc.is_encrypted:
                raise PDFEncryptionError(
                    "PDF is encrypted. Please provide an unencrypted document."
                )

            # Check page count
            if doc.page_count > self._max_pages:
                raise PDFPageLimitError(
                    f"Page count ({doc.page_count}) exceeds limit ({self._max_pages})"
                )

            if doc.page_count == 0:
                raise PDFValidationError("PDF contains no pages")

        finally:
            doc.close()

        logger.debug(
            "pdf_validation_passed",
            file_path=str(file_path),
            file_size_mb=file_size / (1024 * 1024),
        )

    def extract_metadata(self, file_path: Path) -> PDFMetadata:
        """
        Extract comprehensive metadata from a PDF file.

        Args:
            file_path: Path to the PDF file.

        Returns:
            PDFMetadata containing all extracted metadata.

        Raises:
            PDFProcessingError: If metadata extraction fails.
        """
        try:
            # Calculate file hash
            file_hash = self._calculate_file_hash(file_path)

            doc = fitz.open(file_path)

            try:
                # Extract metadata dictionary
                meta = doc.metadata or {}

                # Parse dates if present
                creation_date = self._parse_pdf_date(meta.get("creationDate"))
                mod_date = self._parse_pdf_date(meta.get("modDate"))

                # Check for forms and annotations
                has_forms = any(page.widgets() for page in doc)
                has_annotations = any(
                    len(page.annots()) > 0 for page in doc if page.annots() is not None
                )

                # Generate processing ID
                processing_id = secrets.token_hex(16)

                metadata = PDFMetadata(
                    file_path=file_path.absolute(),
                    file_name=file_path.name,
                    file_size_bytes=file_path.stat().st_size,
                    file_hash=file_hash,
                    page_count=doc.page_count,
                    title=meta.get("title") or None,
                    author=meta.get("author") or None,
                    subject=meta.get("subject") or None,
                    keywords=meta.get("keywords") or None,
                    creator=meta.get("creator") or None,
                    producer=meta.get("producer") or None,
                    creation_date=creation_date,
                    modification_date=mod_date,
                    pdf_version=f"{doc.metadata.get('format', 'PDF 1.4')}",
                    is_encrypted=doc.is_encrypted,
                    has_forms=has_forms,
                    has_annotations=has_annotations,
                    processing_id=processing_id,
                )

                logger.info(
                    "pdf_metadata_extracted",
                    file_name=metadata.file_name,
                    page_count=metadata.page_count,
                    processing_id=processing_id,
                )

                return metadata

            finally:
                doc.close()

        except PDFProcessingError:
            raise
        except Exception as e:
            raise PDFProcessingError(f"Failed to extract metadata: {e}") from e

    def render_page(
        self,
        doc: fitz.Document,
        page_number: int,
    ) -> PageImage:
        """
        Render a single page to a high-quality image.

        Args:
            doc: Open PyMuPDF document.
            page_number: Zero-indexed page number.

        Returns:
            PageImage containing rendered image data.

        Raises:
            PDFProcessingError: If page rendering fails.
        """
        try:
            page = doc[page_number]

            # Get original page dimensions
            rect = page.rect
            original_width = rect.width
            original_height = rect.height

            # Render page to pixmap
            pixmap = page.get_pixmap(
                matrix=self._matrix,
                colorspace=fitz.csRGB,
                alpha=False,
            )

            # Convert to PIL Image
            img = Image.frombytes(
                "RGB",
                (pixmap.width, pixmap.height),
                pixmap.samples,
            )

            # Convert to PNG bytes
            img_buffer = io.BytesIO()
            img.save(img_buffer, format=self._output_format, optimize=True)
            img_bytes = img_buffer.getvalue()

            # Determine orientation
            if pixmap.width > pixmap.height:
                orientation = DocumentOrientation.LANDSCAPE
            elif pixmap.height > pixmap.width:
                orientation = DocumentOrientation.PORTRAIT
            else:
                orientation = DocumentOrientation.SQUARE

            # Check for text and images
            has_text = bool(page.get_text("text").strip())
            has_images = len(page.get_images()) > 0

            page_image = PageImage(
                page_number=page_number + 1,  # Convert to 1-indexed
                image_bytes=img_bytes,
                width=pixmap.width,
                height=pixmap.height,
                dpi=self._dpi,
                orientation=orientation,
                original_width_pts=original_width,
                original_height_pts=original_height,
                has_text=has_text,
                has_images=has_images,
                rotation=int(page.rotation),
            )

            logger.debug(
                "page_rendered",
                page_number=page_number + 1,
                width=pixmap.width,
                height=pixmap.height,
                size_kb=page_image.size_kb,
            )

            return page_image

        except Exception as e:
            raise PDFProcessingError(
                f"Failed to render page {page_number + 1}: {e}"
            ) from e

    def process(
        self,
        file_path: Path,
        page_range: tuple[int, int] | None = None,
    ) -> ProcessingResult:
        """
        Process a complete PDF document.

        Args:
            file_path: Path to the PDF file.
            page_range: Optional tuple of (start, end) page numbers (1-indexed, inclusive).
                       If None, processes all pages.

        Returns:
            ProcessingResult containing metadata and all page images.

        Raises:
            PDFProcessingError: If processing fails.
        """
        start_time = datetime.now(timezone.utc)
        warnings: list[str] = []

        # Validate PDF
        self.validate(file_path)

        # Extract metadata
        metadata = self.extract_metadata(file_path)

        # Create temporary directory for this processing session
        temp_dir = self._temp_dir / metadata.processing_id
        temp_dir.mkdir(parents=True, exist_ok=True)

        try:
            doc = fitz.open(file_path)

            try:
                # Determine page range
                if page_range:
                    start_page = max(0, page_range[0] - 1)  # Convert to 0-indexed
                    end_page = min(doc.page_count, page_range[1])
                else:
                    start_page = 0
                    end_page = min(doc.page_count, self._max_pages)

                if end_page < doc.page_count:
                    warnings.append(
                        f"Only processing pages 1-{end_page} of {doc.page_count} total pages"
                    )

                # Render pages
                pages: list[PageImage] = []
                for page_num in range(start_page, end_page):
                    try:
                        page_image = self.render_page(doc, page_num)
                        pages.append(page_image)
                    except PDFProcessingError as e:
                        warnings.append(f"Page {page_num + 1} rendering failed: {e}")
                        logger.warning(
                            "page_rendering_failed",
                            page_number=page_num + 1,
                            error=str(e),
                        )

                # Calculate processing time
                end_time = datetime.now(timezone.utc)
                processing_time_ms = int(
                    (end_time - start_time).total_seconds() * 1000
                )

                result = ProcessingResult(
                    metadata=metadata,
                    pages=pages,
                    processing_time_ms=processing_time_ms,
                    warnings=warnings,
                    temp_dir=temp_dir,
                )

                logger.info(
                    "pdf_processing_complete",
                    file_name=metadata.file_name,
                    pages_processed=len(pages),
                    processing_time_ms=processing_time_ms,
                    total_size_kb=result.total_size_kb,
                    warnings_count=len(warnings),
                )

                return result

            finally:
                doc.close()

        except PDFProcessingError:
            # Clean up temp directory on failure
            self._cleanup_temp_dir(temp_dir)
            raise
        except Exception as e:
            self._cleanup_temp_dir(temp_dir)
            raise PDFProcessingError(f"PDF processing failed: {e}") from e

    def process_streaming(
        self,
        file_path: Path,
        page_range: tuple[int, int] | None = None,
    ) -> Generator[tuple[PDFMetadata, PageImage], None, None]:
        """
        Process PDF pages as a streaming generator for memory efficiency.

        Yields pages one at a time instead of loading all into memory.
        Useful for large documents or memory-constrained environments.

        Args:
            file_path: Path to the PDF file.
            page_range: Optional tuple of (start, end) page numbers (1-indexed).

        Yields:
            Tuple of (metadata, page_image) for each page.

        Raises:
            PDFProcessingError: If processing fails.
        """
        # Validate PDF
        self.validate(file_path)

        # Extract metadata
        metadata = self.extract_metadata(file_path)

        try:
            doc = fitz.open(file_path)

            try:
                # Determine page range
                if page_range:
                    start_page = max(0, page_range[0] - 1)
                    end_page = min(doc.page_count, page_range[1])
                else:
                    start_page = 0
                    end_page = min(doc.page_count, self._max_pages)

                # Yield pages one at a time
                for page_num in range(start_page, end_page):
                    page_image = self.render_page(doc, page_num)
                    yield metadata, page_image

            finally:
                doc.close()

        except PDFProcessingError:
            raise
        except Exception as e:
            raise PDFProcessingError(f"Streaming processing failed: {e}") from e

    @contextmanager
    def open_document(self, file_path: Path) -> Iterator[fitz.Document]:
        """
        Context manager for safely opening and closing PDF documents.

        Args:
            file_path: Path to the PDF file.

        Yields:
            Open PyMuPDF Document instance.
        """
        self.validate(file_path)
        doc = fitz.open(file_path)
        try:
            yield doc
        finally:
            doc.close()

    def cleanup(self, result: ProcessingResult) -> None:
        """
        Clean up temporary files from processing.

        Args:
            result: ProcessingResult containing temp_dir to clean.
        """
        if result.temp_dir and result.temp_dir.exists():
            self._cleanup_temp_dir(result.temp_dir)

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of file contents."""
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                hasher.update(chunk)
        return hasher.hexdigest()

    def _parse_pdf_date(self, date_str: str | None) -> datetime | None:
        """Parse PDF date string to datetime."""
        if not date_str:
            return None

        # PDF date format: D:YYYYMMDDHHmmSSOHH'mm'
        try:
            # Remove "D:" prefix if present
            if date_str.startswith("D:"):
                date_str = date_str[2:]

            # Extract components
            year = int(date_str[0:4])
            month = int(date_str[4:6]) if len(date_str) >= 6 else 1
            day = int(date_str[6:8]) if len(date_str) >= 8 else 1
            hour = int(date_str[8:10]) if len(date_str) >= 10 else 0
            minute = int(date_str[10:12]) if len(date_str) >= 12 else 0
            second = int(date_str[12:14]) if len(date_str) >= 14 else 0

            return datetime(year, month, day, hour, minute, second, tzinfo=timezone.utc)
        except (ValueError, IndexError):
            logger.debug("failed_to_parse_pdf_date", date_str=date_str)
            return None

    def _cleanup_temp_dir(self, temp_dir: Path) -> None:
        """Securely clean up temporary directory."""
        try:
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
                logger.debug("temp_directory_cleaned", path=str(temp_dir))
        except Exception as e:
            logger.warning(
                "temp_directory_cleanup_failed",
                path=str(temp_dir),
                error=str(e),
            )
