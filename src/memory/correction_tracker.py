"""
Correction tracker for learning from user corrections.

Tracks user corrections to extraction results and uses them
to improve future extractions through pattern learning.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from src.config import get_logger, get_settings
from src.memory.mem0_client import Mem0Client


logger = get_logger(__name__)


@dataclass(slots=True)
class Correction:
    """
    A user correction to an extracted value.

    Attributes:
        id: Unique identifier for the correction.
        field_name: Name of the corrected field.
        original_value: Original extracted value.
        corrected_value: User-provided correct value.
        document_type: Type of document.
        confidence_before: Confidence of original extraction.
        correction_type: Type of correction (value, format, missing).
        user_id: ID of user making correction.
        created_at: Timestamp of correction.
        metadata: Additional correction metadata.
    """

    id: str
    field_name: str
    original_value: Any
    corrected_value: Any
    document_type: str
    confidence_before: float = 0.0
    correction_type: str = "value"
    user_id: str = "default"
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "id": self.id,
            "field_name": self.field_name,
            "original_value": self.original_value,
            "corrected_value": self.corrected_value,
            "document_type": self.document_type,
            "confidence_before": self.confidence_before,
            "correction_type": self.correction_type,
            "user_id": self.user_id,
            "created_at": self.created_at,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Correction:
        """Create from dictionary."""
        return cls(
            id=data["id"],
            field_name=data["field_name"],
            original_value=data.get("original_value"),
            corrected_value=data.get("corrected_value"),
            document_type=data.get("document_type", ""),
            confidence_before=data.get("confidence_before", 0.0),
            correction_type=data.get("correction_type", "value"),
            user_id=data.get("user_id", "default"),
            created_at=data.get("created_at", datetime.now(UTC).isoformat()),
            metadata=data.get("metadata", {}),
        )


class CorrectionTracker:
    """
    Tracks and learns from user corrections.

    Provides methods to:
    - Record user corrections
    - Learn patterns from corrections
    - Apply learned patterns to improve extraction
    - Generate correction statistics
    """

    MEMORY_TYPE = "correction"

    def __init__(
        self,
        mem0_client: Mem0Client | None = None,
        data_dir: Path | str | None = None,
    ) -> None:
        """
        Initialize the correction tracker.

        Args:
            mem0_client: Optional pre-configured Mem0 client.
            data_dir: Directory for correction storage.
        """
        self._client = mem0_client or Mem0Client()
        self._logger = get_logger("memory.correction_tracker")

        settings = get_settings()
        self._data_dir = Path(data_dir) if data_dir else settings.mem0.data_dir
        self._corrections_file = self._data_dir / "corrections.json"
        self._corrections: dict[str, Correction] = {}

        # Load existing corrections
        self._load_corrections()

        # Statistics cache
        self._field_stats: dict[str, dict[str, Any]] = {}
        self._update_statistics()

        self._logger.info(
            "correction_tracker_initialized",
            correction_count=len(self._corrections),
        )

    def _load_corrections(self) -> None:
        """Load corrections from persistent storage."""
        if self._corrections_file.exists():
            try:
                with self._corrections_file.open("r", encoding="utf-8") as f:
                    data = json.load(f)
                    self._corrections = {k: Correction.from_dict(v) for k, v in data.items()}
            except Exception as e:
                self._logger.warning("corrections_load_failed", error=str(e))
                self._corrections = {}

    def _save_corrections(self) -> None:
        """Save corrections to persistent storage."""
        try:
            self._data_dir.mkdir(parents=True, exist_ok=True)
            data = {k: v.to_dict() for k, v in self._corrections.items()}
            with self._corrections_file.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        except Exception as e:
            self._logger.error("corrections_save_failed", error=str(e))

    def _generate_id(self, correction: Correction) -> str:
        """Generate unique ID for a correction."""
        import hashlib

        data = f"{correction.field_name}:{correction.document_type}:{correction.created_at}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def record_correction(
        self,
        field_name: str,
        original_value: Any,
        corrected_value: Any,
        document_type: str,
        confidence_before: float = 0.0,
        correction_type: str = "value",
        user_id: str = "default",
        metadata: dict[str, Any] | None = None,
    ) -> Correction:
        """
        Record a user correction.

        Args:
            field_name: Name of the corrected field.
            original_value: Original extracted value.
            corrected_value: User-provided correct value.
            document_type: Type of document.
            confidence_before: Confidence of original extraction.
            correction_type: Type of correction.
            user_id: ID of user making correction.
            metadata: Additional metadata.

        Returns:
            The recorded correction.
        """
        correction = Correction(
            id="",  # Will be generated
            field_name=field_name,
            original_value=original_value,
            corrected_value=corrected_value,
            document_type=document_type,
            confidence_before=confidence_before,
            correction_type=correction_type,
            user_id=user_id,
            metadata=metadata or {},
        )

        correction.id = self._generate_id(correction)
        self._corrections[correction.id] = correction

        # Store in memory for semantic search
        content = (
            f"correction field:{field_name} document_type:{document_type} "
            f"type:{correction_type} original:{original_value} corrected:{corrected_value}"
        )

        self._client.add(
            content=content,
            metadata=correction.to_dict(),
            memory_type=self.MEMORY_TYPE,
            user_id=user_id,
        )

        self._save_corrections()
        self._update_statistics()

        self._logger.info(
            "correction_recorded",
            correction_id=correction.id,
            field_name=field_name,
            document_type=document_type,
        )

        return correction

    def _update_statistics(self) -> None:
        """Update field correction statistics."""
        self._field_stats = {}

        for correction in self._corrections.values():
            field_name = correction.field_name

            if field_name not in self._field_stats:
                self._field_stats[field_name] = {
                    "total_corrections": 0,
                    "correction_types": {},
                    "common_errors": [],
                    "avg_confidence_before": 0.0,
                    "confidence_sum": 0.0,
                }

            stats = self._field_stats[field_name]
            stats["total_corrections"] += 1
            stats["confidence_sum"] += correction.confidence_before

            # Track correction types
            ctype = correction.correction_type
            stats["correction_types"][ctype] = stats["correction_types"].get(ctype, 0) + 1

            # Track common error patterns
            if correction.original_value and correction.corrected_value:
                error_pattern = {
                    "original": str(correction.original_value)[:50],
                    "corrected": str(correction.corrected_value)[:50],
                }
                if error_pattern not in stats["common_errors"]:
                    stats["common_errors"].append(error_pattern)
                    if len(stats["common_errors"]) > 5:
                        stats["common_errors"].pop(0)

            # Update average confidence
            stats["avg_confidence_before"] = stats["confidence_sum"] / stats["total_corrections"]

    def get_field_hints(
        self,
        field_name: str,
        document_type: str | None = None,
    ) -> dict[str, Any]:
        """
        Get correction hints for a field.

        Args:
            field_name: Name of the field.
            document_type: Optional document type filter.

        Returns:
            Dictionary of hints for the field.
        """
        if field_name not in self._field_stats:
            return {}

        stats = self._field_stats[field_name]

        # Calculate confidence boost based on correction history
        # More corrections = lower base confidence boost
        correction_count = stats["total_corrections"]
        if correction_count == 0:
            confidence_boost = 0.0
        elif correction_count < 3:
            confidence_boost = -0.05  # Small penalty
        elif correction_count < 10:
            confidence_boost = -0.10  # Medium penalty
        else:
            confidence_boost = -0.15  # Larger penalty

        return {
            "total_corrections": correction_count,
            "common_errors": stats["common_errors"][:3],
            "avg_confidence": stats["avg_confidence_before"],
            "confidence_boost": confidence_boost,
            "most_common_issue": max(
                stats["correction_types"].items(),
                key=lambda x: x[1],
                default=("none", 0),
            )[0],
        }

    def get_corrections_for_field(
        self,
        field_name: str,
        limit: int = 10,
    ) -> list[Correction]:
        """
        Get recent corrections for a field.

        Args:
            field_name: Name of the field.
            limit: Maximum number of corrections.

        Returns:
            List of corrections for the field.
        """
        field_corrections = [c for c in self._corrections.values() if c.field_name == field_name]

        # Sort by creation time (newest first)
        field_corrections.sort(key=lambda x: x.created_at, reverse=True)

        return field_corrections[:limit]

    def get_statistics(self) -> dict[str, Any]:
        """
        Get overall correction statistics.

        Returns:
            Dictionary of correction statistics.
        """
        total_corrections = len(self._corrections)

        if total_corrections == 0:
            return {
                "total_corrections": 0,
                "fields_with_corrections": 0,
                "top_corrected_fields": [],
                "correction_rate_by_type": {},
            }

        # Get top corrected fields
        top_fields = sorted(
            self._field_stats.items(),
            key=lambda x: x[1]["total_corrections"],
            reverse=True,
        )[:10]

        # Aggregate correction types
        correction_types: dict[str, int] = {}
        for correction in self._corrections.values():
            ctype = correction.correction_type
            correction_types[ctype] = correction_types.get(ctype, 0) + 1

        return {
            "total_corrections": total_corrections,
            "fields_with_corrections": len(self._field_stats),
            "top_corrected_fields": [
                {"field": f, "count": s["total_corrections"]} for f, s in top_fields
            ],
            "correction_rate_by_type": correction_types,
        }

    def apply_learned_patterns(
        self,
        extraction: dict[str, Any],
        document_type: str,
    ) -> dict[str, Any]:
        """
        Apply learned patterns from corrections to improve extraction.

        Args:
            extraction: Original extraction result.
            document_type: Type of document.

        Returns:
            Enhanced extraction with learned patterns applied.
        """
        enhanced = dict(extraction)

        for field_name, value in extraction.items():
            if field_name not in self._field_stats:
                continue

            hints = self.get_field_hints(field_name, document_type)
            common_errors = hints.get("common_errors", [])

            # Check if current value matches known error patterns
            for error in common_errors:
                if str(value) == error.get("original"):
                    self._logger.info(
                        "applying_learned_correction",
                        field_name=field_name,
                        original=value,
                        suggested=error.get("corrected"),
                    )
                    # Flag the field for review rather than auto-correcting
                    if isinstance(enhanced[field_name], dict):
                        enhanced[field_name]["needs_review"] = True
                        enhanced[field_name]["suggested_correction"] = error.get("corrected")
                    break

        return enhanced

    def clear(self) -> int:
        """Clear all corrections."""
        count = len(self._corrections)
        self._corrections = {}
        self._field_stats = {}
        self._save_corrections()
        self._client.clear(memory_type=self.MEMORY_TYPE)

        self._logger.info("corrections_cleared", count=count)

        return count
