"""
Golden dataset management for evaluation and benchmarking.

Provides loading, saving, and versioning of golden datasets —
documents with known ground-truth field values used to measure
extraction accuracy over time.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


@dataclass(slots=True)
class GoldenSample:
    """
    A single ground-truth sample in a golden dataset.

    Attributes:
        sample_id: Unique identifier for this sample.
        document_type: Document type (e.g., "invoice", "w2").
        schema_name: Schema to use for extraction.
        expected_fields: Ground-truth field values.
        source_file: Path or identifier for the source document.
        metadata: Additional sample metadata.
        tags: Tags for filtering (e.g., "easy", "multi-page", "handwritten").
    """

    sample_id: str
    document_type: str
    schema_name: str
    expected_fields: dict[str, Any]
    source_file: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)
    tags: list[str] = field(default_factory=list)

    def field_count(self) -> int:
        """Count of expected non-None fields."""
        return sum(1 for v in self.expected_fields.values() if v is not None)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sample_id": self.sample_id,
            "document_type": self.document_type,
            "schema_name": self.schema_name,
            "expected_fields": self.expected_fields,
            "source_file": self.source_file,
            "metadata": self.metadata,
            "tags": self.tags,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GoldenSample:
        return cls(
            sample_id=data["sample_id"],
            document_type=data["document_type"],
            schema_name=data["schema_name"],
            expected_fields=data.get("expected_fields", {}),
            source_file=data.get("source_file", ""),
            metadata=data.get("metadata", {}),
            tags=data.get("tags", []),
        )


@dataclass
class GoldenDataset:
    """
    A versioned collection of golden samples.

    Attributes:
        name: Dataset name (e.g., "invoice_golden_v1").
        version: Semantic version string.
        description: Human-readable description.
        samples: List of golden samples.
        created_at: Creation timestamp.
        metadata: Additional dataset metadata.
    """

    name: str
    version: str = "1.0.0"
    description: str = ""
    samples: list[GoldenSample] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def sample_count(self) -> int:
        return len(self.samples)

    @property
    def document_types(self) -> list[str]:
        """Unique document types in the dataset."""
        return sorted({s.document_type for s in self.samples})

    @property
    def schema_names(self) -> list[str]:
        """Unique schema names in the dataset."""
        return sorted({s.schema_name for s in self.samples})

    @property
    def all_tags(self) -> list[str]:
        """All unique tags across samples."""
        tags: set[str] = set()
        for s in self.samples:
            tags.update(s.tags)
        return sorted(tags)

    def get_sample(self, sample_id: str) -> GoldenSample | None:
        """Look up a sample by ID."""
        for s in self.samples:
            if s.sample_id == sample_id:
                return s
        return None

    def filter_by_type(self, document_type: str) -> list[GoldenSample]:
        """Get all samples of a given document type."""
        return [s for s in self.samples if s.document_type == document_type]

    def filter_by_tag(self, tag: str) -> list[GoldenSample]:
        """Get all samples with a given tag."""
        return [s for s in self.samples if tag in s.tags]

    def filter_by_schema(self, schema_name: str) -> list[GoldenSample]:
        """Get all samples for a given schema."""
        return [s for s in self.samples if s.schema_name == schema_name]

    def add_sample(self, sample: GoldenSample) -> None:
        """Add a sample to the dataset."""
        existing = self.get_sample(sample.sample_id)
        if existing:
            raise ValueError(f"Sample '{sample.sample_id}' already exists in dataset")
        self.samples.append(sample)

    def remove_sample(self, sample_id: str) -> bool:
        """Remove a sample by ID. Returns True if found and removed."""
        for i, s in enumerate(self.samples):
            if s.sample_id == sample_id:
                self.samples.pop(i)
                return True
        return False

    def content_hash(self) -> str:
        """Compute a SHA-256 hash of the dataset contents for change detection."""
        serialized = json.dumps(
            [s.to_dict() for s in sorted(self.samples, key=lambda x: x.sample_id)],
            sort_keys=True,
        )
        return hashlib.sha256(serialized.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "created_at": self.created_at,
            "metadata": self.metadata,
            "content_hash": self.content_hash(),
            "sample_count": self.sample_count,
            "samples": [s.to_dict() for s in self.samples],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> GoldenDataset:
        samples = [GoldenSample.from_dict(s) for s in data.get("samples", [])]
        return cls(
            name=data["name"],
            version=data.get("version", "1.0.0"),
            description=data.get("description", ""),
            samples=samples,
            created_at=data.get("created_at", datetime.now(UTC).isoformat()),
            metadata=data.get("metadata", {}),
        )


# ──────────────────────────────────────────────────────────────────
# Dataset I/O
# ──────────────────────────────────────────────────────────────────


def save_dataset(dataset: GoldenDataset, path: str | Path) -> Path:
    """
    Save a golden dataset to a JSON file.

    Args:
        dataset: The dataset to save.
        path: File path to write to.

    Returns:
        The resolved path that was written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(dataset.to_dict(), f, indent=2, ensure_ascii=False)
    logger.info("golden_dataset_saved", path=str(path), samples=dataset.sample_count)
    return path


def load_dataset(path: str | Path) -> GoldenDataset:
    """
    Load a golden dataset from a JSON file.

    Args:
        path: File path to read from.

    Returns:
        Loaded GoldenDataset.

    Raises:
        FileNotFoundError: If the file does not exist.
        ValueError: If the file is not valid JSON.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Golden dataset not found: {path}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    dataset = GoldenDataset.from_dict(data)
    logger.info("golden_dataset_loaded", path=str(path), samples=dataset.sample_count)
    return dataset


def create_sample(
    sample_id: str,
    document_type: str,
    schema_name: str,
    expected_fields: dict[str, Any],
    source_file: str = "",
    tags: list[str] | None = None,
) -> GoldenSample:
    """Convenience factory for GoldenSample."""
    return GoldenSample(
        sample_id=sample_id,
        document_type=document_type,
        schema_name=schema_name,
        expected_fields=expected_fields,
        source_file=source_file,
        tags=tags or [],
    )
