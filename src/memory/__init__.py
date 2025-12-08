"""
Memory layer for document extraction context management.

Provides Mem0-based persistent memory for:
- Context retrieval from similar documents
- Provider-specific patterns
- User correction tracking
- Self-improving extraction
"""

from src.memory.mem0_client import Mem0Client, MemoryEntry, MemorySearchResult
from src.memory.context_manager import ContextManager, ExtractionContext
from src.memory.correction_tracker import CorrectionTracker, Correction
from src.memory.vector_store import VectorStoreManager, VectorStoreConfig

__all__ = [
    "Mem0Client",
    "MemoryEntry",
    "MemorySearchResult",
    "ContextManager",
    "ExtractionContext",
    "CorrectionTracker",
    "Correction",
    "VectorStoreManager",
    "VectorStoreConfig",
]
