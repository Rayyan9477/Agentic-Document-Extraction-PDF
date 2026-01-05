"""
Memory layer for document extraction context management.

Provides Mem0-based persistent memory for:
- Context retrieval from similar documents
- Provider-specific patterns
- User correction tracking
- Self-improving extraction
"""

from src.memory.context_manager import ContextManager, ExtractionContext
from src.memory.correction_tracker import Correction, CorrectionTracker
from src.memory.mem0_client import Mem0Client, MemoryEntry, MemorySearchResult
from src.memory.vector_store import VectorStoreConfig, VectorStoreManager


__all__ = [
    "ContextManager",
    "Correction",
    "CorrectionTracker",
    "ExtractionContext",
    "Mem0Client",
    "MemoryEntry",
    "MemorySearchResult",
    "VectorStoreConfig",
    "VectorStoreManager",
]
