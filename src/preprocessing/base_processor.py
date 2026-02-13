"""
Base file processor abstract class.

Defines the interface that all file format processors must implement,
enabling the FileProcessorFactory to route files to the correct processor.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from src.config import get_logger
from src.preprocessing.pdf_processor import (
    DocumentOrientation,
    PageImage,
    PDFMetadata,
    ProcessingResult,
)


logger = get_logger(__name__)


class FileFormat(str, Enum):
    """Supported file formats for document processing."""

    PDF = "pdf"
    PNG = "png"
    JPG = "jpg"
    JPEG = "jpeg"
    TIFF = "tiff"
    TIF = "tif"
    BMP = "bmp"
    DOCX = "docx"
    DOC = "doc"
    XLSX = "xlsx"
    CSV = "csv"
    DICOM = "dcm"
    DICOM_ALT = "dicom"
    EDI = "edi"
    X12 = "x12"
    EDI_835 = "835"
    EDI_837 = "837"


# Group formats by processor type
IMAGE_FORMATS = {FileFormat.PNG, FileFormat.JPG, FileFormat.JPEG, FileFormat.TIFF, FileFormat.TIF, FileFormat.BMP}
SPREADSHEET_FORMATS = {FileFormat.XLSX, FileFormat.CSV}
DICOM_FORMATS = {FileFormat.DICOM, FileFormat.DICOM_ALT}
EDI_FORMATS = {FileFormat.EDI, FileFormat.X12, FileFormat.EDI_835, FileFormat.EDI_837}

SUPPORTED_EXTENSIONS: set[str] = {f".{fmt.value}" for fmt in FileFormat}


class FileProcessingError(Exception):
    """Base exception for file processing errors."""


class UnsupportedFormatError(FileProcessingError):
    """Raised when file format is not supported."""


class FileValidationError(FileProcessingError):
    """Raised when file validation fails."""


class BaseFileProcessor(ABC):
    """
    Abstract base class for file format processors.

    All processors convert their input format into a list of PageImage objects
    that the downstream extraction pipeline can consume uniformly.
    """

    @abstractmethod
    def validate(self, file_path: Path) -> None:
        """
        Validate the file before processing.

        Args:
            file_path: Path to the file to validate.

        Raises:
            FileValidationError: If the file is invalid.
        """

    @abstractmethod
    def process(self, file_path: Path) -> ProcessingResult:
        """
        Process the file and convert to page images.

        Args:
            file_path: Path to the file to process.

        Returns:
            ProcessingResult with page images and metadata.
        """

    @staticmethod
    def _detect_orientation(width: int, height: int) -> DocumentOrientation:
        """Detect page orientation from dimensions."""
        if width > height:
            return DocumentOrientation.LANDSCAPE
        if height > width:
            return DocumentOrientation.PORTRAIT
        return DocumentOrientation.SQUARE

    @staticmethod
    def _compute_file_hash(file_path: Path) -> str:
        """Compute SHA-256 hash of file contents."""
        import hashlib
        sha256 = hashlib.sha256()
        with file_path.open("rb") as f:
            for chunk in iter(lambda: f.read(8192), b""):
                sha256.update(chunk)
        return sha256.hexdigest()
