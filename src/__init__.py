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

# Core configuration
from src.config import get_settings, get_logger, AuditLogger

# Preprocessing components
from src.preprocessing import (
    PDFProcessor,
    ImageEnhancer,
    BatchManager,
    PageImage,
    PDFMetadata,
    EnhancementResult,
)

# VLM client components
from src.client import (
    LMStudioClient,
    ConnectionManager,
    HealthMonitor,
    VisionRequest,
    VisionResponse,
)

# Schema components
from src.schemas import (
    DocumentType,
    DocumentSchema,
    SchemaRegistry,
    FieldType,
    get_schema,
    get_all_schemas,
)

# Utility components
from src.utils import (
    ensure_directory,
    get_file_hash,
    safe_filename,
    atomic_write,
    FileLock,
    compute_sha256,
    generate_unique_id,
    mask_sensitive_data,
    parse_date,
    format_date,
    is_valid_date,
    calculate_age,
    normalize_whitespace,
    normalize_name,
    clean_currency,
    fuzzy_match,
)


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
    # Configuration
    "get_settings",
    "get_logger",
    "AuditLogger",
    # Preprocessing
    "PDFProcessor",
    "ImageEnhancer",
    "BatchManager",
    "PageImage",
    "PDFMetadata",
    "EnhancementResult",
    # VLM Client
    "LMStudioClient",
    "ConnectionManager",
    "HealthMonitor",
    "VisionRequest",
    "VisionResponse",
    # Schemas
    "DocumentType",
    "DocumentSchema",
    "SchemaRegistry",
    "FieldType",
    "get_schema",
    "get_all_schemas",
    # Utilities
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
