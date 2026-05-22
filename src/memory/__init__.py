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
from src.memory.dynamic_prompt import (
    DynamicPromptEnhancer,
    FieldWarning,
    PromptEnhancement,
)
from src.memory.mem0_client import Mem0Client, MemoryEntry, MemorySearchResult
from src.memory.vector_store import VectorStoreConfig, VectorStoreManager


def get_vector_store(
    tenant_id: str | None = None,
    *,
    config: VectorStoreConfig | None = None,
) -> VectorStoreManager:
    """V3 Phase 8 — single canonical entry point for vector-store access.

    When ``tenant_id`` is None / "default" / equals
    ``GLOBAL_SCOPE``, returns a global-scope manager (legacy behaviour).
    Otherwise routes through ``VectorStoreManager.for_tenant`` so the
    on-disk index path is partitioned per tenant.

    Honours ``settings.api.multi_tenant_enabled`` — when off, ignores
    ``tenant_id`` and returns the global manager. This makes the call
    site safe in single-tenant deployments without conditional logic.
    """
    from src.config import get_settings

    settings = get_settings()
    multi_tenant = getattr(settings.api, "multi_tenant_enabled", False)
    default_tenant = getattr(settings.api, "default_tenant_id", "default")

    if (
        not multi_tenant
        or not tenant_id
        or tenant_id == default_tenant
        or tenant_id == VectorStoreManager.GLOBAL_SCOPE
    ):
        return VectorStoreManager(config=config)

    return VectorStoreManager.for_tenant(tenant_id, config=config)


__all__ = [
    "ContextManager",
    "Correction",
    "CorrectionTracker",
    "DynamicPromptEnhancer",
    "ExtractionContext",
    "FieldWarning",
    "Mem0Client",
    "MemoryEntry",
    "MemorySearchResult",
    "PromptEnhancement",
    "VectorStoreConfig",
    "VectorStoreManager",
    "get_vector_store",
]
