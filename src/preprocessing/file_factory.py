"""
File processor factory for routing files to the correct processor.

Routes incoming documents to the appropriate processor based on file
extension, providing a unified interface for all supported formats.
"""

from pathlib import Path

from src.config import get_logger
from src.preprocessing.base_processor import (
    BaseFileProcessor,
    DICOM_FORMATS,
    EDI_FORMATS,
    FileFormat,
    IMAGE_FORMATS,
    SPREADSHEET_FORMATS,
    SUPPORTED_EXTENSIONS,
    UnsupportedFormatError,
)
from src.preprocessing.pdf_processor import ProcessingResult


logger = get_logger(__name__)


class FileProcessorFactory:
    """
    Factory that routes files to the appropriate processor.

    Usage:
        factory = FileProcessorFactory()
        result = factory.process(Path("document.pdf"))
        result = factory.process(Path("scan.tiff"))
        result = factory.process(Path("report.docx"))
    """

    def __init__(self, dpi: int = 300) -> None:
        self._dpi = dpi
        self._processors: dict[str, BaseFileProcessor] = {}

    def get_processor(self, file_path: Path) -> BaseFileProcessor:
        """
        Get the appropriate processor for a file.

        Args:
            file_path: Path to the file to process.

        Returns:
            Appropriate BaseFileProcessor instance.

        Raises:
            UnsupportedFormatError: If the file format is not supported.
        """
        suffix = file_path.suffix.lower().lstrip(".")

        try:
            file_format = FileFormat(suffix)
        except ValueError:
            raise UnsupportedFormatError(
                f"Unsupported file format: '.{suffix}'. "
                f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}"
            )

        # Return cached processor or create new one
        processor_key = self._get_processor_key(file_format)
        if processor_key not in self._processors:
            self._processors[processor_key] = self._create_processor(file_format)

        return self._processors[processor_key]

    def process(self, file_path: Path) -> ProcessingResult:
        """
        Process a file through the appropriate processor.

        Args:
            file_path: Path to the file to process.

        Returns:
            ProcessingResult with page images and metadata.
        """
        processor = self.get_processor(file_path)

        logger.info(
            "file_processing_start",
            file=file_path.name,
            processor=type(processor).__name__,
        )

        return processor.process(file_path)

    def is_supported(self, file_path: Path) -> bool:
        """Check if a file format is supported."""
        suffix = file_path.suffix.lower()
        return suffix in SUPPORTED_EXTENSIONS

    @staticmethod
    def supported_extensions() -> set[str]:
        """Return all supported file extensions."""
        return SUPPORTED_EXTENSIONS

    def _get_processor_key(self, file_format: FileFormat) -> str:
        """Get a cache key for the processor type."""
        if file_format == FileFormat.PDF:
            return "pdf"
        if file_format in IMAGE_FORMATS:
            return "image"
        if file_format in {FileFormat.DOCX, FileFormat.DOC}:
            return "docx"
        if file_format in SPREADSHEET_FORMATS:
            return "spreadsheet"
        if file_format in DICOM_FORMATS:
            return "dicom"
        if file_format in EDI_FORMATS:
            return "edi"
        return file_format.value

    def _create_processor(self, file_format: FileFormat) -> BaseFileProcessor:
        """Create the appropriate processor for a file format."""
        if file_format == FileFormat.PDF:
            return self._create_pdf_processor()

        if file_format in IMAGE_FORMATS:
            from src.preprocessing.image_processor import ImageProcessor
            return ImageProcessor(dpi=self._dpi)

        if file_format in {FileFormat.DOCX, FileFormat.DOC}:
            from src.preprocessing.docx_processor import DocxProcessor
            return DocxProcessor(dpi=self._dpi)

        if file_format in SPREADSHEET_FORMATS:
            from src.preprocessing.spreadsheet_processor import SpreadsheetProcessor
            return SpreadsheetProcessor(dpi=self._dpi)

        if file_format in DICOM_FORMATS:
            from src.preprocessing.dicom_processor import DicomProcessor
            return DicomProcessor(dpi=self._dpi)

        if file_format in EDI_FORMATS:
            from src.preprocessing.edi_processor import EDIProcessor
            return EDIProcessor(dpi=self._dpi)

        raise UnsupportedFormatError(f"No processor for format: {file_format}")

    def _create_pdf_processor(self) -> BaseFileProcessor:
        """Create PDF processor wrapped as BaseFileProcessor."""
        from src.preprocessing.pdf_processor import PDFProcessor as _PDFProcessor

        # PDFProcessor doesn't extend BaseFileProcessor but has the same interface.
        # We use a lightweight adapter.
        return _PDFProcessorAdapter(dpi=self._dpi)


class _PDFProcessorAdapter(BaseFileProcessor):
    """Adapter wrapping the existing PDFProcessor to the BaseFileProcessor interface."""

    def __init__(self, dpi: int = 300) -> None:
        from src.preprocessing.pdf_processor import PDFProcessor
        self._inner = PDFProcessor(dpi=dpi)

    def validate(self, file_path: Path) -> None:
        self._inner.validate(file_path)

    def process(self, file_path: Path) -> ProcessingResult:
        return self._inner.process(file_path)
