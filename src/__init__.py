"""
Local Agentic Medical Document Extraction System.

A production-ready, HIPAA-compliant document extraction system using local
Vision Language Models (VLM) with a 4-agent architecture powered by LangChain
and LangGraph.

Copyright 2024-2025. All rights reserved.

Usage:
    from src import get_settings, get_logger
    from src.preprocessing import PDFProcessor, ImageEnhancer, BatchManager
    from src.client import LMStudioClient, ConnectionManager, HealthMonitor
    from src.schemas import DocumentType, get_schema
"""

from importlib.metadata import PackageNotFoundError, version
from typing import TYPE_CHECKING

# Core configuration (always available)
from src.config import AuditLogger, get_logger, get_settings

# Schema components (always available - no heavy dependencies)
from src.schemas import (
    DocumentSchema,
    DocumentType,
    FieldType,
    SchemaRegistry,
    get_all_schemas,
    get_schema,
)

# Utility components (always available)
from src.utils import (
    FileLock,
    atomic_write,
    calculate_age,
    clean_currency,
    compute_sha256,
    ensure_directory,
    format_date,
    fuzzy_match,
    generate_unique_id,
    get_file_hash,
    is_valid_date,
    mask_sensitive_data,
    normalize_name,
    normalize_whitespace,
    parse_date,
    safe_filename,
)


# Preprocessing components (requires cv2/numpy - optional for core functionality)
try:
    from src.preprocessing import (
        BatchManager,
        EnhancementResult,
        ImageEnhancer,
        PageImage,
        PDFMetadata,
        PDFProcessor,
    )

    _PREPROCESSING_AVAILABLE = True
except ImportError:
    PDFProcessor = None  # type: ignore[misc, assignment]
    ImageEnhancer = None  # type: ignore[misc, assignment]
    BatchManager = None  # type: ignore[misc, assignment]
    PageImage = None  # type: ignore[misc, assignment]
    PDFMetadata = None  # type: ignore[misc, assignment]
    EnhancementResult = None  # type: ignore[misc, assignment]
    _PREPROCESSING_AVAILABLE = False

# VLM client components (requires httpx - optional for schema-only usage)
try:
    from src.client import (
        ConnectionManager,
        HealthMonitor,
        LMStudioClient,
        VisionRequest,
        VisionResponse,
    )

    _CLIENT_AVAILABLE = True
except ImportError:
    LMStudioClient = None  # type: ignore[misc, assignment]
    ConnectionManager = None  # type: ignore[misc, assignment]
    HealthMonitor = None  # type: ignore[misc, assignment]
    VisionRequest = None  # type: ignore[misc, assignment]
    VisionResponse = None  # type: ignore[misc, assignment]
    _CLIENT_AVAILABLE = False


try:
    __version__ = version("doc-extraction-system")
except PackageNotFoundError:
    __version__ = "2.0.0"

__author__ = "Rayyan Ahmed"
__email__ = "rayyan@example.com"
__license__ = "Proprietary"

__all__ = [
    # Package info
    "__version__",
    "__author__",
    "__email__",
    "__license__",
    # Availability flags
    "_PREPROCESSING_AVAILABLE",
    "_CLIENT_AVAILABLE",
    # Configuration
    "get_settings",
    "get_logger",
    "AuditLogger",
    # Preprocessing (may be None if cv2 not installed)
    "PDFProcessor",
    "ImageEnhancer",
    "BatchManager",
    "PageImage",
    "PDFMetadata",
    "EnhancementResult",
    # VLM Client (may be None if httpx not installed)
    "LMStudioClient",
    "ConnectionManager",
    "HealthMonitor",
    "VisionRequest",
    "VisionResponse",
    # Schemas (always available)
    "DocumentType",
    "DocumentSchema",
    "SchemaRegistry",
    "FieldType",
    "get_schema",
    "get_all_schemas",
    # Utilities (always available)
    "ensure_directory",
    "get_file_hash",
    "safe_filename",
    "atomic_write",
    "FileLock",
    "compute_sha256",
    "generate_unique_id",
    "mask_sensitive_data",
    "parse_date",
    "format_date",
    "is_valid_date",
    "calculate_age",
    "normalize_whitespace",
    "normalize_name",
    "clean_currency",
    "fuzzy_match",
]
