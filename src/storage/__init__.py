"""
Storage module for document extraction results.

Provides file-based storage for extraction results with
support for retrieval, listing, and cleanup.
"""

from src.storage.result_store import (
    ResultStore,
    get_result_store,
    save_result,
    get_result,
)

__all__ = [
    "ResultStore",
    "get_result_store",
    "save_result",
    "get_result",
]
