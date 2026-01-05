"""
Preprocessing module for the document extraction system.

Provides PDF processing, image enhancement, and batch management
capabilities for preparing documents for VLM extraction.
"""

from src.preprocessing.batch_manager import BatchManager, BatchResult
from src.preprocessing.image_enhancer import EnhancementResult, ImageEnhancer
from src.preprocessing.pdf_processor import (
    PageImage,
    PDFMetadata,
    PDFProcessor,
    ProcessingResult,
)


__all__ = [
    "BatchManager",
    "BatchResult",
    "EnhancementResult",
    "ImageEnhancer",
    "PDFMetadata",
    "PDFProcessor",
    "PageImage",
    "ProcessingResult",
]
