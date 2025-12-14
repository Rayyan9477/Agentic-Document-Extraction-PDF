"""
Vector store configuration and management for memory layer.

Provides unified interface for vector storage backends
(FAISS, Qdrant) with local-first HIPAA-compliant deployment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any
import json

from src.config import get_logger, get_settings


logger = get_logger(__name__)


class VectorStoreType(str, Enum):
    """Supported vector store backends."""

    FAISS = "faiss"
    QDRANT = "qdrant"
    SIMPLE = "simple"  # In-memory for testing


@dataclass(slots=True)
class VectorStoreConfig:
    """
    Configuration for vector store.

    Attributes:
        store_type: Type of vector store backend.
        embedding_dim: Dimension of embeddings.
        index_path: Path to index file (for FAISS).
        qdrant_url: URL for Qdrant server.
        qdrant_collection: Collection name for Qdrant.
        similarity_metric: Similarity metric (cosine, l2, ip).
        batch_size: Batch size for indexing.
    """

    store_type: VectorStoreType = VectorStoreType.FAISS
    embedding_dim: int = 384  # all-MiniLM-L6-v2 dimension
    index_path: Path | None = None
    qdrant_url: str = "http://localhost:6333"
    qdrant_collection: str = "document_extractions"
    similarity_metric: str = "cosine"
    batch_size: int = 100

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "store_type": self.store_type.value,
            "embedding_dim": self.embedding_dim,
            "index_path": str(self.index_path) if self.index_path else None,
            "qdrant_url": self.qdrant_url,
            "qdrant_collection": self.qdrant_collection,
            "similarity_metric": self.similarity_metric,
            "batch_size": self.batch_size,
        }


class VectorStoreManager:
    """
    Manages vector store operations.

    Provides a unified interface for different vector store backends
    with support for local FAISS and remote Qdrant deployments.
    """

    def __init__(
        self,
        config: VectorStoreConfig | None = None,
        data_dir: Path | str | None = None,
    ) -> None:
        """
        Initialize the vector store manager.

        Args:
            config: Vector store configuration.
            data_dir: Directory for data storage.
        """
        settings = get_settings()
        mem0_settings = settings.mem0

        self._config = config or VectorStoreConfig(
            store_type=VectorStoreType(mem0_settings.vector_store.value),
        )

        self._data_dir = Path(data_dir) if data_dir else mem0_settings.data_dir
        self._data_dir.mkdir(parents=True, exist_ok=True)

        self._logger = get_logger("memory.vector_store")

        # Initialize the appropriate backend
        self._store: Any = None
        self._initialize_store()

        self._logger.info(
            "vector_store_initialized",
            store_type=self._config.store_type.value,
            data_dir=str(self._data_dir),
        )

    def _initialize_store(self) -> None:
        """Initialize the vector store backend."""
        if self._config.store_type == VectorStoreType.FAISS:
            self._initialize_faiss()
        elif self._config.store_type == VectorStoreType.QDRANT:
            self._initialize_qdrant()
        else:
            self._initialize_simple()

    def _initialize_faiss(self) -> None:
        """Initialize FAISS vector store."""
        try:
            import faiss
            import numpy as np

            index_path = self._config.index_path or (self._data_dir / "faiss.index")

            if index_path.exists():
                self._store = faiss.read_index(str(index_path))
                self._logger.info("faiss_index_loaded", path=str(index_path))
            else:
                # Create new index with cosine similarity (normalized L2)
                self._store = faiss.IndexFlatIP(self._config.embedding_dim)
                self._logger.info("faiss_index_created", dim=self._config.embedding_dim)

            self._index_path = index_path
            self._id_map: dict[int, str] = {}
            self._id_map_file = self._data_dir / "faiss_id_map.json"

            # Load ID mapping
            if self._id_map_file.exists():
                with self._id_map_file.open("r") as f:
                    self._id_map = {int(k): v for k, v in json.load(f).items()}

        except ImportError:
            self._logger.warning(
                "faiss_not_available",
                message="Install with: pip install faiss-cpu"
            )
            self._initialize_simple()

    def _initialize_qdrant(self) -> None:
        """Initialize Qdrant vector store."""
        try:
            from qdrant_client import QdrantClient
            from qdrant_client.models import Distance, VectorParams

            self._store = QdrantClient(url=self._config.qdrant_url)

            # Ensure collection exists
            collections = self._store.get_collections().collections
            collection_names = [c.name for c in collections]

            if self._config.qdrant_collection not in collection_names:
                self._store.create_collection(
                    collection_name=self._config.qdrant_collection,
                    vectors_config=VectorParams(
                        size=self._config.embedding_dim,
                        distance=Distance.COSINE,
                    ),
                )
                self._logger.info(
                    "qdrant_collection_created",
                    collection=self._config.qdrant_collection,
                )

        except ImportError:
            self._logger.warning(
                "qdrant_not_available",
                message="Install with: pip install qdrant-client"
            )
            self._initialize_simple()
        except Exception as e:
            self._logger.warning(
                "qdrant_connection_failed",
                error=str(e),
                message="Falling back to simple store"
            )
            self._initialize_simple()

    def _initialize_simple(self) -> None:
        """Initialize simple in-memory vector store."""
        self._store = []
        self._config.store_type = VectorStoreType.SIMPLE
        self._simple_file = self._data_dir / "simple_vectors.json"

        if self._simple_file.exists():
            try:
                with self._simple_file.open("r") as f:
                    self._store = json.load(f)
            except json.JSONDecodeError as e:
                self._logger.warning(
                    "simple_store_load_failed",
                    error=str(e),
                    file=str(self._simple_file),
                    reason="Invalid JSON format",
                )
                self._store = []
            except (OSError, IOError) as e:
                self._logger.warning(
                    "simple_store_load_failed",
                    error=str(e),
                    file=str(self._simple_file),
                    reason="File read error",
                )
                self._store = []
            except Exception as e:
                self._logger.error(
                    "simple_store_load_unexpected_error",
                    error=str(e),
                    error_type=type(e).__name__,
                    file=str(self._simple_file),
                )
                self._store = []

        self._logger.info("simple_store_initialized")

    def add(
        self,
        id: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> bool:
        """
        Add a vector to the store.

        Args:
            id: Unique identifier for the vector.
            embedding: Vector embedding.
            metadata: Optional metadata.

        Returns:
            True if successful.
        """
        try:
            if self._config.store_type == VectorStoreType.FAISS:
                return self._add_faiss(id, embedding, metadata)
            elif self._config.store_type == VectorStoreType.QDRANT:
                return self._add_qdrant(id, embedding, metadata)
            else:
                return self._add_simple(id, embedding, metadata)
        except Exception as e:
            self._logger.error("vector_add_failed", error=str(e))
            return False

    def _add_faiss(
        self,
        id: str,
        embedding: list[float],
        metadata: dict[str, Any] | None,
    ) -> bool:
        """Add vector to FAISS index."""
        import numpy as np

        # Normalize for cosine similarity
        vec = np.array([embedding], dtype=np.float32)
        faiss_lib = __import__("faiss")
        faiss_lib.normalize_L2(vec)

        # Get next index
        idx = self._store.ntotal
        self._store.add(vec)

        # Store ID mapping
        self._id_map[idx] = id
        self._save_faiss()

        return True

    def _add_qdrant(
        self,
        id: str,
        embedding: list[float],
        metadata: dict[str, Any] | None,
    ) -> bool:
        """Add vector to Qdrant."""
        from qdrant_client.models import PointStruct

        point = PointStruct(
            id=hash(id) % (2**63),  # Qdrant requires int IDs
            vector=embedding,
            payload={"original_id": id, **(metadata or {})},
        )

        self._store.upsert(
            collection_name=self._config.qdrant_collection,
            points=[point],
        )

        return True

    def _add_simple(
        self,
        id: str,
        embedding: list[float],
        metadata: dict[str, Any] | None,
    ) -> bool:
        """Add vector to simple store."""
        # Remove existing entry with same ID
        self._store = [v for v in self._store if v["id"] != id]

        self._store.append({
            "id": id,
            "embedding": embedding,
            "metadata": metadata or {},
        })

        self._save_simple()

        return True

    def search(
        self,
        embedding: list[float],
        top_k: int = 5,
        threshold: float = 0.0,
    ) -> list[dict[str, Any]]:
        """
        Search for similar vectors.

        Args:
            embedding: Query embedding.
            top_k: Number of results.
            threshold: Minimum similarity threshold.

        Returns:
            List of results with id, score, and metadata.
        """
        try:
            if self._config.store_type == VectorStoreType.FAISS:
                return self._search_faiss(embedding, top_k, threshold)
            elif self._config.store_type == VectorStoreType.QDRANT:
                return self._search_qdrant(embedding, top_k, threshold)
            else:
                return self._search_simple(embedding, top_k, threshold)
        except Exception as e:
            self._logger.error("vector_search_failed", error=str(e))
            return []

    def _search_faiss(
        self,
        embedding: list[float],
        top_k: int,
        threshold: float,
    ) -> list[dict[str, Any]]:
        """Search FAISS index."""
        import numpy as np

        if self._store.ntotal == 0:
            return []

        # Normalize query
        vec = np.array([embedding], dtype=np.float32)
        faiss_lib = __import__("faiss")
        faiss_lib.normalize_L2(vec)

        # Search
        k = min(top_k, self._store.ntotal)
        distances, indices = self._store.search(vec, k)

        results = []
        for score, idx in zip(distances[0], indices[0]):
            if idx < 0 or score < threshold:
                continue

            original_id = self._id_map.get(int(idx))
            if original_id:
                results.append({
                    "id": original_id,
                    "score": float(score),
                    "metadata": {},
                })

        return results

    def _search_qdrant(
        self,
        embedding: list[float],
        top_k: int,
        threshold: float,
    ) -> list[dict[str, Any]]:
        """Search Qdrant collection."""
        results = self._store.search(
            collection_name=self._config.qdrant_collection,
            query_vector=embedding,
            limit=top_k,
            score_threshold=threshold,
        )

        return [
            {
                "id": r.payload.get("original_id", str(r.id)),
                "score": r.score,
                "metadata": {k: v for k, v in r.payload.items() if k != "original_id"},
            }
            for r in results
        ]

    def _search_simple(
        self,
        embedding: list[float],
        top_k: int,
        threshold: float,
    ) -> list[dict[str, Any]]:
        """Search simple store."""
        import math

        results = []

        for entry in self._store:
            score = self._cosine_similarity(embedding, entry["embedding"])

            if score >= threshold:
                results.append({
                    "id": entry["id"],
                    "score": score,
                    "metadata": entry.get("metadata", {}),
                })

        # Sort by score
        results.sort(key=lambda x: x["score"], reverse=True)

        return results[:top_k]

    def _cosine_similarity(self, vec1: list[float], vec2: list[float]) -> float:
        """Calculate cosine similarity."""
        import math

        dot = sum(a * b for a, b in zip(vec1, vec2))
        norm1 = math.sqrt(sum(a * a for a in vec1))
        norm2 = math.sqrt(sum(b * b for b in vec2))

        if norm1 == 0 or norm2 == 0:
            return 0.0

        return dot / (norm1 * norm2)

    def delete(self, id: str) -> bool:
        """Delete a vector by ID."""
        try:
            if self._config.store_type == VectorStoreType.SIMPLE:
                original_len = len(self._store)
                self._store = [v for v in self._store if v["id"] != id]
                self._save_simple()
                return len(self._store) < original_len

            # FAISS and Qdrant deletion is more complex
            self._logger.warning(
                "delete_not_fully_supported",
                store_type=self._config.store_type.value,
            )
            return False

        except Exception as e:
            self._logger.error("vector_delete_failed", error=str(e))
            return False

    def _save_faiss(self) -> None:
        """Save FAISS index and ID map."""
        try:
            faiss_lib = __import__("faiss")
            faiss_lib.write_index(self._store, str(self._index_path))

            with self._id_map_file.open("w") as f:
                json.dump({str(k): v for k, v in self._id_map.items()}, f)

        except Exception as e:
            self._logger.error("faiss_save_failed", error=str(e))

    def _save_simple(self) -> None:
        """Save simple store to file."""
        try:
            with self._simple_file.open("w") as f:
                json.dump(self._store, f)
        except Exception as e:
            self._logger.error("simple_save_failed", error=str(e))

    @property
    def count(self) -> int:
        """Get number of vectors in store."""
        if self._config.store_type == VectorStoreType.FAISS:
            return self._store.ntotal if self._store else 0
        elif self._config.store_type == VectorStoreType.QDRANT:
            try:
                info = self._store.get_collection(self._config.qdrant_collection)
                return info.points_count
            except Exception:
                return 0
        else:
            return len(self._store)

    def clear(self) -> int:
        """Clear all vectors from store."""
        count = self.count

        if self._config.store_type == VectorStoreType.FAISS:
            faiss_lib = __import__("faiss")
            self._store = faiss_lib.IndexFlatIP(self._config.embedding_dim)
            self._id_map = {}
            self._save_faiss()

        elif self._config.store_type == VectorStoreType.QDRANT:
            try:
                self._store.delete_collection(self._config.qdrant_collection)
                self._initialize_qdrant()
            except Exception as e:
                self._logger.error("qdrant_clear_failed", error=str(e))

        else:
            self._store = []
            self._save_simple()

        self._logger.info("vector_store_cleared", count=count)

        return count
