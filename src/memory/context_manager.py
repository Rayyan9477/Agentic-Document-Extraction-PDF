"""
Context manager for extraction memory retrieval and storage.

Manages the retrieval of relevant context from memory to enhance
extraction accuracy and the storage of extraction results for
future reference.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from src.config import get_logger, get_settings
from src.memory.mem0_client import Mem0Client


logger = get_logger(__name__)


@dataclass(slots=True)
class ExtractionContext:
    """
    Context retrieved from memory for an extraction.

    Attributes:
        similar_extractions: Previously extracted similar documents.
        provider_patterns: Provider-specific extraction patterns.
        correction_hints: Hints from past corrections.
        schema_hints: Suggested schema modifications.
        confidence_adjustments: Field-level confidence adjustments.
        retrieval_time_ms: Time taken to retrieve context.
    """

    similar_extractions: list[dict[str, Any]] = field(default_factory=list)
    provider_patterns: dict[str, Any] = field(default_factory=dict)
    correction_hints: dict[str, Any] = field(default_factory=dict)
    schema_hints: dict[str, Any] = field(default_factory=dict)
    confidence_adjustments: dict[str, float] = field(default_factory=dict)
    retrieval_time_ms: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "similar_extractions": self.similar_extractions,
            "provider_patterns": self.provider_patterns,
            "correction_hints": self.correction_hints,
            "schema_hints": self.schema_hints,
            "confidence_adjustments": self.confidence_adjustments,
            "retrieval_time_ms": self.retrieval_time_ms,
        }

    @property
    def has_context(self) -> bool:
        """Check if any context was retrieved."""
        return bool(self.similar_extractions or self.provider_patterns or self.correction_hints)


class ContextManager:
    """
    Manages extraction context retrieval and storage.

    Provides methods to:
    - Retrieve relevant context before extraction
    - Store extraction results for future reference
    - Track provider-specific patterns
    - Apply learned corrections
    """

    MEMORY_TYPE_EXTRACTION = "extraction"
    MEMORY_TYPE_PATTERN = "pattern"
    MEMORY_TYPE_PROVIDER = "provider"

    def __init__(
        self,
        mem0_client: Mem0Client | None = None,
        session_id: str | None = None,
    ) -> None:
        """
        Initialize the context manager.

        Args:
            mem0_client: Optional pre-configured Mem0 client.
            session_id: Session identifier for memory grouping.
        """
        self._client = mem0_client or Mem0Client()
        self._session_id = session_id or "default"
        self._logger = get_logger("memory.context_manager")
        self._settings = get_settings()

    def retrieve_context(
        self,
        document_type: str,
        provider_name: str | None = None,
        field_names: list[str] | None = None,
        pdf_hash: str | None = None,
    ) -> ExtractionContext:
        """
        Retrieve relevant context for an extraction.

        Args:
            document_type: Type of document being extracted.
            provider_name: Name of healthcare provider (if known).
            field_names: List of fields being extracted.
            pdf_hash: Hash of the PDF for duplicate detection.

        Returns:
            ExtractionContext with relevant memories.
        """
        start_time = datetime.now(UTC)
        context = ExtractionContext()

        try:
            # Check for exact duplicate
            if pdf_hash:
                duplicate_results = self._client.search(
                    query=f"pdf_hash:{pdf_hash}",
                    memory_type=self.MEMORY_TYPE_EXTRACTION,
                    top_k=1,
                    threshold=0.99,
                )
                if duplicate_results:
                    self._logger.info(
                        "duplicate_document_found",
                        pdf_hash=pdf_hash,
                    )
                    context.similar_extractions = [r.entry.metadata for r in duplicate_results]

            # Search for similar document extractions
            query = f"document_type:{document_type}"
            if field_names:
                query += f" fields:{','.join(field_names[:5])}"

            similar_results = self._client.search(
                query=query,
                memory_type=self.MEMORY_TYPE_EXTRACTION,
                top_k=3,
                threshold=0.6,
            )

            if similar_results:
                context.similar_extractions.extend(
                    [
                        {
                            "id": r.entry.id,
                            "score": r.score,
                            "document_type": r.entry.metadata.get("document_type"),
                            "fields": r.entry.metadata.get("fields", {}),
                            "confidence": r.entry.metadata.get("confidence", 0.0),
                        }
                        for r in similar_results
                    ]
                )

            # Search for provider-specific patterns
            if provider_name:
                provider_results = self._client.search(
                    query=f"provider:{provider_name}",
                    memory_type=self.MEMORY_TYPE_PROVIDER,
                    top_k=1,
                    threshold=0.8,
                )

                if provider_results:
                    best_match = provider_results[0]
                    context.provider_patterns = best_match.entry.metadata.get("patterns", {})

            # Get correction hints for fields
            if field_names:
                for field_name in field_names[:10]:  # Limit to avoid too many queries
                    correction_results = self._client.search(
                        query=f"correction field:{field_name}",
                        memory_type="correction",
                        top_k=1,
                        threshold=0.7,
                    )

                    if correction_results:
                        correction = correction_results[0].entry
                        context.correction_hints[field_name] = {
                            "common_errors": correction.metadata.get("common_errors", []),
                            "suggested_format": correction.metadata.get("format"),
                            "confidence_boost": correction.metadata.get("confidence_boost", 0.0),
                        }

            # Calculate confidence adjustments
            context.confidence_adjustments = self._calculate_confidence_adjustments(context)

        except Exception as e:
            self._logger.warning(
                "context_retrieval_failed",
                error=str(e),
            )

        # Calculate retrieval time
        elapsed = datetime.now(UTC) - start_time
        context.retrieval_time_ms = int(elapsed.total_seconds() * 1000)

        self._logger.info(
            "context_retrieved",
            has_context=context.has_context,
            similar_count=len(context.similar_extractions),
            retrieval_time_ms=context.retrieval_time_ms,
        )

        return context

    def store_extraction(
        self,
        document_type: str,
        extraction_result: dict[str, Any],
        confidence: float,
        pdf_hash: str | None = None,
        provider_name: str | None = None,
    ) -> str:
        """
        Store extraction result in memory.

        Args:
            document_type: Type of document extracted.
            extraction_result: Extracted field values.
            confidence: Overall extraction confidence.
            pdf_hash: Hash of the PDF.
            provider_name: Name of healthcare provider.

        Returns:
            Memory ID of stored extraction.
        """
        # Build content for semantic search
        fields_summary = ", ".join(f"{k}:{v}" for k, v in list(extraction_result.items())[:10])
        content = (
            f"document_type:{document_type} confidence:{confidence:.2f} fields:{fields_summary}"
        )

        # Build metadata
        metadata = {
            "document_type": document_type,
            "fields": extraction_result,
            "confidence": confidence,
            "pdf_hash": pdf_hash,
            "provider_name": provider_name,
            "extracted_at": datetime.now(UTC).isoformat(),
        }

        entry = self._client.add(
            content=content,
            metadata=metadata,
            memory_type=self.MEMORY_TYPE_EXTRACTION,
            user_id=self._session_id,
        )

        self._logger.info(
            "extraction_stored",
            memory_id=entry.id,
            document_type=document_type,
            field_count=len(extraction_result),
        )

        return entry.id

    def store_provider_pattern(
        self,
        provider_name: str,
        patterns: dict[str, Any],
    ) -> str:
        """
        Store provider-specific extraction patterns.

        Args:
            provider_name: Name of the healthcare provider.
            patterns: Dictionary of extraction patterns.

        Returns:
            Memory ID of stored pattern.
        """
        content = f"provider:{provider_name} patterns"

        metadata = {
            "provider_name": provider_name,
            "patterns": patterns,
            "updated_at": datetime.now(UTC).isoformat(),
        }

        entry = self._client.add(
            content=content,
            metadata=metadata,
            memory_type=self.MEMORY_TYPE_PROVIDER,
            user_id=self._session_id,
        )

        self._logger.info(
            "provider_pattern_stored",
            memory_id=entry.id,
            provider_name=provider_name,
        )

        return entry.id

    def _calculate_confidence_adjustments(
        self,
        context: ExtractionContext,
    ) -> dict[str, float]:
        """
        Calculate confidence adjustments based on context.

        Fields with consistent past extractions get a small boost.
        Fields with many corrections get a penalty.
        """
        adjustments: dict[str, float] = {}

        # Boost from similar successful extractions
        if context.similar_extractions:
            high_conf_extractions = [
                e for e in context.similar_extractions if e.get("confidence", 0) >= 0.85
            ]
            if len(high_conf_extractions) >= 2:
                # Extract common fields with high confidence
                all_fields = set()
                for e in high_conf_extractions:
                    all_fields.update(e.get("fields", {}).keys())

                for field in all_fields:
                    adjustments[field] = adjustments.get(field, 0.0) + 0.05

        # Apply correction-based adjustments
        for field_name, hints in context.correction_hints.items():
            boost = hints.get("confidence_boost", 0.0)
            adjustments[field_name] = adjustments.get(field_name, 0.0) + boost

        return adjustments

    def get_similar_documents(
        self,
        document_type: str,
        limit: int = 5,
    ) -> list[dict[str, Any]]:
        """
        Get similar previously processed documents.

        Args:
            document_type: Type of document to search for.
            limit: Maximum number of results.

        Returns:
            List of similar document metadata.
        """
        results = self._client.search(
            query=f"document_type:{document_type}",
            memory_type=self.MEMORY_TYPE_EXTRACTION,
            top_k=limit,
            threshold=0.5,
        )

        return [
            {
                "id": r.entry.id,
                "score": r.score,
                **r.entry.metadata,
            }
            for r in results
        ]

    def clear_session_memory(self) -> int:
        """Clear all memories for the current session."""
        return self._client.clear(user_id=self._session_id)
