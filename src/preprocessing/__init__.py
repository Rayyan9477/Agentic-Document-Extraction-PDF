"""
Preprocessing module for the document extraction system.

Provides multi-format file processing, image enhancement, and batch management
capabilities for preparing documents for VLM extraction.

Supported formats: PDF, PNG, JPG, TIFF, BMP, DOCX, XLSX, CSV, DICOM, EDI X12
"""

from src.preprocessing.base_processor import (
    BaseFileProcessor,
    FileFormat,
    FileProcessingError,
    FileValidationError,
    SUPPORTED_EXTENSIONS,
    UnsupportedFormatError,
)
from src.preprocessing.batch_manager import BatchManager, BatchResult
from src.preprocessing.file_factory import FileProcessorFactory
from src.preprocessing.image_enhancer import EnhancementResult, ImageEnhancer
from src.preprocessing.pdf_processor import (
    PageImage,
    PDFMetadata,
    PDFProcessor,
    ProcessingResult,
)


__all__ = [
    "BaseFileProcessor",
    "BatchManager",
    "BatchResult",
    "EnhancementResult",
    "FileFormat",
    "FileProcessingError",
    "FileProcessorFactory",
    "FileValidationError",
    "ImageEnhancer",
    "PDFMetadata",
    "PDFProcessor",
    "PageImage",
    "ProcessingResult",
    "SUPPORTED_EXTENSIONS",
    "UnsupportedFormatError",
]
