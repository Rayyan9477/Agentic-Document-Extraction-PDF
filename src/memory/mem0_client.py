"""
Mem0 client wrapper for document extraction memory management.

Provides a unified interface for storing and retrieving extraction
context using the Mem0 memory framework with local vector storage.
"""

from __future__ import annotations

import hashlib
import json
import tempfile
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.config import get_logger, get_settings


logger = get_logger(__name__)


@dataclass(slots=True)
class MemoryEntry:
    """
    A single memory entry for extraction context.

    Attributes:
        id: Unique identifier for the memory.
        content: Text content of the memory.
        metadata: Additional metadata about the memory.
        embedding: Vector embedding (if computed).
        created_at: Timestamp when memory was created.
        updated_at: Timestamp when memory was last updated.
        memory_type: Type of memory (document, correction, pattern).
        user_id: User/session identifier.
    """

    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    embedding: list[float] | None = None
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    memory_type: str = "document"
    user_id: str = "default"

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "content": self.content,
            "metadata": self.metadata,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "memory_type": self.memory_type,
            "user_id": self.user_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> MemoryEntry:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            content=data["content"],
            metadata=data.get("metadata", {}),
            embedding=data.get("embedding"),
            created_at=data.get("created_at", datetime.now(UTC).isoformat()),
            updated_at=data.get("updated_at", datetime.now(UTC).isoformat()),
            memory_type=data.get("memory_type", "document"),
            user_id=data.get("user_id", "default"),
        )


@dataclass(slots=True)
class MemorySearchResult:
    """
    Result from a memory search operation.

    Attributes:
        entry: The matching memory entry.
        score: Similarity score (0-1).
        distance: Distance metric value.
    """

    entry: MemoryEntry
    score: float
    distance: float = 0.0


class Mem0Client:
    """
    Client for Mem0 memory operations.

    Provides methods to add, search, update, and delete memories
    for document extraction context management.
    """

    def __init__(
        self,
        data_dir: Path | str | None = None,
        embedding_model: str | None = None,
        user_id: str = "default",
    ) -> None:
        """
        Initialize the Mem0 client.

        Args:
            data_dir: Directory for persistent storage.
            embedding_model: Sentence transformer model name.
            user_id: Default user/session identifier.
        """
        settings = get_settings()
        mem0_settings = settings.mem0

        self._data_dir = Path(data_dir) if data_dir else mem0_settings.data_dir
        self._embedding_model = embedding_model or mem0_settings.embedding_model
        self._user_id = user_id
        self._top_k = mem0_settings.top_k
        self._similarity_threshold = mem0_settings.similarity_threshold

        self._logger = get_logger("memory.mem0_client")

        # Initialize storage
        self._data_dir.mkdir(parents=True, exist_ok=True)
        self._memories_file = self._data_dir / "memories.json"
        self._memories: dict[str, MemoryEntry] = {}

        # File access lock for thread safety
        self._file_lock = threading.RLock()

        # Load existing memories
        self._load_memories()

        # Initialize embedding model (lazy loading)
        self._embedder = None

        self._logger.info(
            "mem0_client_initialized",
            data_dir=str(self._data_dir),
            embedding_model=self._embedding_model,
            memory_count=len(self._memories),
        )

    def _load_memories(self) -> None:
        """Load memories from persistent storage with locking."""
        with self._file_lock:
            if self._memories_file.exists():
                try:
                    with self._memories_file.open("r", encoding="utf-8") as f:
                        data = json.load(f)
                        self._memories = {k: MemoryEntry.from_dict(v) for k, v in data.items()}
                    self._logger.debug("memories_loaded", count=len(self._memories))
                except Exception as e:
                    self._logger.warning("memories_load_failed", error=str(e))
                    self._memories = {}

    def _save_memories(self) -> None:
        """
        Save memories to persistent storage with atomic write.

        Uses atomic write pattern (temp file + rename) to prevent
        data corruption if crash occurs mid-write. Thread-safe
        via locking.
        """
        with self._file_lock:
            try:
                data = {k: v.to_dict() for k, v in self._memories.items()}

                # Atomic write: write to temp file, then rename
                # This prevents data loss if crash occurs mid-write
                fd, temp_path = tempfile.mkstemp(
                    dir=self._data_dir, prefix=".memories_", suffix=".tmp"
                )
                try:
                    with open(fd, "w", encoding="utf-8") as f:
                        json.dump(data, f, indent=2, ensure_ascii=False)
                        f.flush()  # Ensure data is written

                    # Atomic rename (on most filesystems)
                    temp_file = Path(temp_path)
                    temp_file.replace(self._memories_file)
                    self._logger.debug("memories_saved", count=len(self._memories))
                except Exception:
                    # Clean up temp file on failure
                    try:
                        Path(temp_path).unlink(missing_ok=True)
                    except Exception:
                        pass
                    raise
            except Exception as e:
                self._logger.error("memories_save_failed", error=str(e))

    def _get_embedder(self):
        """Lazy load the sentence transformer embedder."""
        if self._embedder is None:
            try:
                from sentence_transformers import SentenceTransformer

                self._embedder = SentenceTransformer(self._embedding_model)
                self._logger.info("embedder_loaded", model=self._embedding_model)
            except ImportError:
                self._logger.warning(
                    "sentence_transformers_not_available",
                    message="Install with: pip install sentence-transformers",
                )
                return None
        return self._embedder

    def unload_embedder(self) -> None:
        """
        Unload the sentence transformer model to free memory.

        Call this when the embedder is no longer needed, especially
        in memory-constrained environments or when switching models.
        The model will be lazily reloaded on next use if needed.
        """
        if self._embedder is not None:
            # Clear reference to allow garbage collection
            self._embedder = None
            self._logger.info("embedder_unloaded", model=self._embedding_model)

            # Force garbage collection to free GPU/CPU memory
            import gc

            gc.collect()

            # If PyTorch is available, clear CUDA cache
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                pass  # PyTorch not installed

    def close(self) -> None:
        """
        Clean up resources held by this client.

        Saves memories and unloads the embedding model.
        Call this when done using the client.
        """
        self._save_memories()
        self.unload_embedder()
        self._logger.info("mem0_client_closed")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit - ensures cleanup."""
        self.close()
        return False

    def _compute_embedding(self, text: str) -> list[float] | None:
        """Compute embedding for text."""
        embedder = self._get_embedder()
        if embedder is None:
            return None

        try:
            embedding = embedder.encode(text, convert_to_numpy=True)
            return embedding.tolist()
        except Exception as e:
            self._logger.warning("embedding_failed", error=str(e))
            return None

    def _generate_id(self, content: str, metadata: dict[str, Any]) -> str:
        """Generate unique ID for a memory entry."""
        data = f"{content}:{json.dumps(metadata, sort_keys=True)}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def add(
        self,
        content: str,
        metadata: dict[str, Any] | None = None,
        memory_type: str = "document",
        user_id: str | None = None,
    ) -> MemoryEntry:
        """
        Add a new memory entry.

        Args:
            content: Text content of the memory.
            metadata: Additional metadata.
            memory_type: Type of memory (document, correction, pattern).
            user_id: User/session identifier.

        Returns:
            The created memory entry.
        """
        metadata = metadata or {}
        user_id = user_id or self._user_id

        memory_id = self._generate_id(content, metadata)

        # Check if memory already exists
        if memory_id in self._memories:
            existing = self._memories[memory_id]
            existing.updated_at = datetime.now(UTC).isoformat()
            self._save_memories()
            return existing

        # Create new entry
        entry = MemoryEntry(
            id=memory_id,
            content=content,
            metadata=metadata,
            embedding=self._compute_embedding(content),
            memory_type=memory_type,
            user_id=user_id,
        )

        self._memories[memory_id] = entry
        self._save_memories()

        self._logger.info(
            "memory_added",
            memory_id=memory_id,
            memory_type=memory_type,
            content_length=len(content),
        )

        return entry

    def search(
        self,
        query: str,
        memory_type: str | None = None,
        user_id: str | None = None,
        top_k: int | None = None,
        threshold: float | None = None,
    ) -> list[MemorySearchResult]:
        """
        Search for relevant memories.

        Args:
            query: Search query text.
            memory_type: Filter by memory type.
            user_id: Filter by user/session.
            top_k: Number of results to return.
            threshold: Minimum similarity threshold.

        Returns:
            List of matching memories with scores.
        """
        top_k = top_k or self._top_k
        threshold = threshold or self._similarity_threshold

        # Compute query embedding
        query_embedding = self._compute_embedding(query)

        results: list[MemorySearchResult] = []

        for memory in self._memories.values():
            # Apply filters
            if memory_type and memory.memory_type != memory_type:
                continue
            if user_id and memory.user_id != user_id:
                continue

            # Calculate similarity
            if query_embedding and memory.embedding:
                score = self._cosine_similarity(query_embedding, memory.embedding)
            else:
                # Fallback to simple text matching
                score = self._text_similarity(query, memory.content)

            if score >= threshold:
                results.append(
                    MemorySearchResult(
                        entry=memory,
                        score=score,
                        distance=1.0 - score,
                    )
                )

        # Sort by score and limit results
        results.sort(key=lambda x: x.score, reverse=True)
        results = results[:top_k]

        self._logger.debug(
            "memory_search",
            query_length=len(query),
            results_count=len(results),
        )

        return results

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity between two vectors."""
        import math

        dot_product = sum(a * b for a, b in zip(vec1, vec2, strict=False))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot_product / (norm1 * norm2)

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity based on word overlap."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())

        if not words1 or not words2:
            return 0.0

        intersection = words1 & words2
        union = words1 | words2

        return len(intersection) / len(union)

    def get(self, memory_id: str) -> MemoryEntry | None:
        """Get a specific memory by ID."""
        return self._memories.get(memory_id)

    def update(
        self,
        memory_id: str,
        content: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> MemoryEntry | None:
        """
        Update an existing memory.

        Args:
            memory_id: ID of memory to update.
            content: New content (optional).
            metadata: New/updated metadata (optional).

        Returns:
            Updated memory entry or None if not found.
        """
        if memory_id not in self._memories:
            return None

        entry = self._memories[memory_id]

        if content is not None:
            entry.content = content
            entry.embedding = self._compute_embedding(content)

        if metadata is not None:
            entry.metadata.update(metadata)

        entry.updated_at = datetime.now(UTC).isoformat()
        self._save_memories()

        self._logger.info("memory_updated", memory_id=memory_id)

        return entry

    def delete(self, memory_id: str) -> bool:
        """
        Delete a memory.

        Args:
            memory_id: ID of memory to delete.

        Returns:
            True if deleted, False if not found.
        """
        if memory_id not in self._memories:
            return False

        del self._memories[memory_id]
        self._save_memories()

        self._logger.info("memory_deleted", memory_id=memory_id)

        return True

    def get_all(
        self,
        memory_type: str | None = None,
        user_id: str | None = None,
    ) -> list[MemoryEntry]:
        """
        Get all memories with optional filtering.

        Args:
            memory_type: Filter by memory type.
            user_id: Filter by user/session.

        Returns:
            List of matching memory entries.
        """
        results = []

        for memory in self._memories.values():
            if memory_type and memory.memory_type != memory_type:
                continue
            if user_id and memory.user_id != user_id:
                continue
            results.append(memory)

        return results

    def clear(self, memory_type: str | None = None, user_id: str | None = None) -> int:
        """
        Clear memories with optional filtering.

        Args:
            memory_type: Filter by memory type (clears all if None).
            user_id: Filter by user/session (clears all if None).

        Returns:
            Number of memories cleared.
        """
        if memory_type is None and user_id is None:
            count = len(self._memories)
            self._memories = {}
        else:
            to_delete = []
            for memory_id, memory in self._memories.items():
                if memory_type and memory.memory_type != memory_type:
                    continue
                if user_id and memory.user_id != user_id:
                    continue
                to_delete.append(memory_id)

            for memory_id in to_delete:
                del self._memories[memory_id]

            count = len(to_delete)

        self._save_memories()

        self._logger.info("memories_cleared", count=count)

        return count

    @property
    def memory_count(self) -> int:
        """Get total number of memories."""
        return len(self._memories)
