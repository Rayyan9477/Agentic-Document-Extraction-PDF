"""
Extraction accuracy metrics for evaluation and benchmarking.

Provides field-level and document-level accuracy computation:
precision, recall, F1, exact match, character error rate, and
aggregate scoring across evaluation datasets.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class MatchLevel(str, Enum):
    """How strictly to compare extracted vs expected values."""

    EXACT = "exact"
    NORMALIZED = "normalized"  # strip whitespace, lowercase
    FUZZY = "fuzzy"  # Levenshtein ratio >= threshold
    NUMERIC = "numeric"  # parse as number, compare within tolerance


# ──────────────────────────────────────────────────────────────────
# Per-Field Comparison Result
# ──────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class FieldMatchResult:
    """Result of comparing a single extracted field to its expected value."""

    field_name: str
    expected: Any
    extracted: Any
    is_match: bool
    match_level: MatchLevel
    similarity: float = 1.0  # 0-1, where 1 = perfect match
    error_message: str | None = None

    @property
    def is_present(self) -> bool:
        """Whether the field was extracted (non-None)."""
        return self.extracted is not None

    @property
    def is_expected(self) -> bool:
        """Whether the field has an expected value."""
        return self.expected is not None


# ──────────────────────────────────────────────────────────────────
# Document-Level Metrics
# ──────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class DocumentMetrics:
    """Accuracy metrics for a single document evaluation."""

    document_id: str
    schema_name: str
    field_results: list[FieldMatchResult] = field(default_factory=list)
    extraction_time_ms: int = 0

    @property
    def total_fields(self) -> int:
        return len(self.field_results)

    @property
    def expected_fields(self) -> int:
        return sum(1 for r in self.field_results if r.is_expected)

    @property
    def extracted_fields(self) -> int:
        return sum(1 for r in self.field_results if r.is_present)

    @property
    def correct_fields(self) -> int:
        return sum(1 for r in self.field_results if r.is_match)

    @property
    def precision(self) -> float:
        """Of extracted fields, how many were correct."""
        if self.extracted_fields == 0:
            return 0.0
        return self.correct_fields / self.extracted_fields

    @property
    def recall(self) -> float:
        """Of expected fields, how many were correctly extracted."""
        if self.expected_fields == 0:
            return 0.0
        return self.correct_fields / self.expected_fields

    @property
    def f1(self) -> float:
        """Harmonic mean of precision and recall."""
        p, r = self.precision, self.recall
        if p + r == 0:
            return 0.0
        return 2 * (p * r) / (p + r)

    @property
    def exact_match(self) -> bool:
        """Whether every expected field was perfectly extracted."""
        return all(r.is_match for r in self.field_results if r.is_expected)

    @property
    def mean_similarity(self) -> float:
        """Average similarity across expected fields."""
        expected = [r for r in self.field_results if r.is_expected]
        if not expected:
            return 0.0
        return sum(r.similarity for r in expected) / len(expected)

    def to_dict(self) -> dict[str, Any]:
        return {
            "document_id": self.document_id,
            "schema_name": self.schema_name,
            "total_fields": self.total_fields,
            "expected_fields": self.expected_fields,
            "extracted_fields": self.extracted_fields,
            "correct_fields": self.correct_fields,
            "precision": round(self.precision, 4),
            "recall": round(self.recall, 4),
            "f1": round(self.f1, 4),
            "exact_match": self.exact_match,
            "mean_similarity": round(self.mean_similarity, 4),
            "extraction_time_ms": self.extraction_time_ms,
        }


# ──────────────────────────────────────────────────────────────────
# Aggregate Metrics (across multiple documents)
# ──────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class AggregateMetrics:
    """Aggregated metrics across an evaluation dataset."""

    document_metrics: list[DocumentMetrics] = field(default_factory=list)
    dataset_name: str = ""

    @property
    def document_count(self) -> int:
        return len(self.document_metrics)

    @property
    def total_fields(self) -> int:
        return sum(d.total_fields for d in self.document_metrics)

    @property
    def total_expected(self) -> int:
        return sum(d.expected_fields for d in self.document_metrics)

    @property
    def total_extracted(self) -> int:
        return sum(d.extracted_fields for d in self.document_metrics)

    @property
    def total_correct(self) -> int:
        return sum(d.correct_fields for d in self.document_metrics)

    @property
    def macro_precision(self) -> float:
        """Average precision across documents."""
        if not self.document_metrics:
            return 0.0
        return sum(d.precision for d in self.document_metrics) / len(self.document_metrics)

    @property
    def macro_recall(self) -> float:
        """Average recall across documents."""
        if not self.document_metrics:
            return 0.0
        return sum(d.recall for d in self.document_metrics) / len(self.document_metrics)

    @property
    def macro_f1(self) -> float:
        """Average F1 across documents."""
        if not self.document_metrics:
            return 0.0
        return sum(d.f1 for d in self.document_metrics) / len(self.document_metrics)

    @property
    def micro_precision(self) -> float:
        """Global precision across all fields."""
        if self.total_extracted == 0:
            return 0.0
        return self.total_correct / self.total_extracted

    @property
    def micro_recall(self) -> float:
        """Global recall across all fields."""
        if self.total_expected == 0:
            return 0.0
        return self.total_correct / self.total_expected

    @property
    def micro_f1(self) -> float:
        """Global F1 across all fields."""
        p, r = self.micro_precision, self.micro_recall
        if p + r == 0:
            return 0.0
        return 2 * (p * r) / (p + r)

    @property
    def exact_match_rate(self) -> float:
        """Fraction of documents with all fields correct."""
        if not self.document_metrics:
            return 0.0
        return sum(1 for d in self.document_metrics if d.exact_match) / len(
            self.document_metrics
        )

    @property
    def mean_extraction_time_ms(self) -> float:
        """Average extraction time across documents."""
        if not self.document_metrics:
            return 0.0
        return sum(d.extraction_time_ms for d in self.document_metrics) / len(
            self.document_metrics
        )

    def per_field_f1(self) -> dict[str, float]:
        """Compute F1 per field name across all documents."""
        field_stats: dict[str, dict[str, int]] = {}
        for doc in self.document_metrics:
            for fr in doc.field_results:
                if fr.field_name not in field_stats:
                    field_stats[fr.field_name] = {
                        "expected": 0,
                        "extracted": 0,
                        "correct": 0,
                    }
                if fr.is_expected:
                    field_stats[fr.field_name]["expected"] += 1
                if fr.is_present:
                    field_stats[fr.field_name]["extracted"] += 1
                if fr.is_match:
                    field_stats[fr.field_name]["correct"] += 1

        result: dict[str, float] = {}
        for name, stats in field_stats.items():
            p = stats["correct"] / stats["extracted"] if stats["extracted"] > 0 else 0.0
            r = stats["correct"] / stats["expected"] if stats["expected"] > 0 else 0.0
            result[name] = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        return result

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset_name": self.dataset_name,
            "document_count": self.document_count,
            "total_fields": self.total_fields,
            "total_expected": self.total_expected,
            "total_extracted": self.total_extracted,
            "total_correct": self.total_correct,
            "macro_precision": round(self.macro_precision, 4),
            "macro_recall": round(self.macro_recall, 4),
            "macro_f1": round(self.macro_f1, 4),
            "micro_precision": round(self.micro_precision, 4),
            "micro_recall": round(self.micro_recall, 4),
            "micro_f1": round(self.micro_f1, 4),
            "exact_match_rate": round(self.exact_match_rate, 4),
            "mean_extraction_time_ms": round(self.mean_extraction_time_ms, 2),
            "per_field_f1": {
                k: round(v, 4) for k, v in self.per_field_f1().items()
            },
        }


# ──────────────────────────────────────────────────────────────────
# Field Comparison Engine
# ──────────────────────────────────────────────────────────────────


def _normalize(value: Any) -> str:
    """Normalize a value to a lowercase stripped string."""
    if value is None:
        return ""
    s = str(value).strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _levenshtein_ratio(a: str, b: str) -> float:
    """Compute normalized Levenshtein similarity (0-1)."""
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    max_len = max(len(a), len(b))
    if max_len == 0:
        return 1.0

    # Simple DP Levenshtein distance
    m, n = len(a), len(b)
    prev = list(range(n + 1))
    for i in range(1, m + 1):
        curr = [i] + [0] * n
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + cost)
        prev = curr

    distance = prev[n]
    return 1.0 - (distance / max_len)


def _parse_number(value: Any) -> float | None:
    """Parse a value as a number, stripping currency symbols."""
    if value is None:
        return None
    s = str(value).strip()
    s = re.sub(r"[$€£¥,]", "", s)
    s = s.strip()
    try:
        return float(s)
    except (ValueError, TypeError):
        return None


def compare_field(
    field_name: str,
    expected: Any,
    extracted: Any,
    match_level: MatchLevel = MatchLevel.NORMALIZED,
    fuzzy_threshold: float = 0.85,
    numeric_tolerance: float = 0.01,
) -> FieldMatchResult:
    """
    Compare an extracted field value against the expected value.

    Args:
        field_name: Name of the field being compared.
        expected: Ground-truth expected value.
        extracted: Value produced by extraction.
        match_level: How strictly to compare.
        fuzzy_threshold: Similarity threshold for fuzzy matching.
        numeric_tolerance: Relative tolerance for numeric matching.

    Returns:
        FieldMatchResult with match status and similarity.
    """
    # Both None → match
    if expected is None and extracted is None:
        return FieldMatchResult(
            field_name=field_name,
            expected=expected,
            extracted=extracted,
            is_match=True,
            match_level=match_level,
            similarity=1.0,
        )

    # One None → no match
    if expected is None or extracted is None:
        return FieldMatchResult(
            field_name=field_name,
            expected=expected,
            extracted=extracted,
            is_match=False,
            match_level=match_level,
            similarity=0.0,
            error_message="Missing value" if extracted is None else "Unexpected value",
        )

    if match_level == MatchLevel.EXACT:
        is_match = str(expected) == str(extracted)
        sim = 1.0 if is_match else _levenshtein_ratio(str(expected), str(extracted))
        return FieldMatchResult(
            field_name=field_name,
            expected=expected,
            extracted=extracted,
            is_match=is_match,
            match_level=match_level,
            similarity=sim,
        )

    if match_level == MatchLevel.NORMALIZED:
        norm_e = _normalize(expected)
        norm_x = _normalize(extracted)
        is_match = norm_e == norm_x
        sim = 1.0 if is_match else _levenshtein_ratio(norm_e, norm_x)
        return FieldMatchResult(
            field_name=field_name,
            expected=expected,
            extracted=extracted,
            is_match=is_match,
            match_level=match_level,
            similarity=sim,
        )

    if match_level == MatchLevel.FUZZY:
        norm_e = _normalize(expected)
        norm_x = _normalize(extracted)
        sim = _levenshtein_ratio(norm_e, norm_x)
        return FieldMatchResult(
            field_name=field_name,
            expected=expected,
            extracted=extracted,
            is_match=sim >= fuzzy_threshold,
            match_level=match_level,
            similarity=sim,
        )

    if match_level == MatchLevel.NUMERIC:
        num_e = _parse_number(expected)
        num_x = _parse_number(extracted)
        if num_e is None or num_x is None:
            # Fall back to normalized string comparison
            norm_e = _normalize(expected)
            norm_x = _normalize(extracted)
            is_match = norm_e == norm_x
            sim = 1.0 if is_match else _levenshtein_ratio(norm_e, norm_x)
            return FieldMatchResult(
                field_name=field_name,
                expected=expected,
                extracted=extracted,
                is_match=is_match,
                match_level=match_level,
                similarity=sim,
                error_message="Could not parse as number",
            )
        if num_e == 0:
            is_match = num_x == 0
            sim = 1.0 if is_match else 0.0
        else:
            rel_diff = abs(num_e - num_x) / abs(num_e)
            is_match = rel_diff <= numeric_tolerance
            sim = max(0.0, 1.0 - rel_diff)
        return FieldMatchResult(
            field_name=field_name,
            expected=expected,
            extracted=extracted,
            is_match=is_match,
            match_level=match_level,
            similarity=sim,
        )

    # Fallback — normalized
    return compare_field(field_name, expected, extracted, MatchLevel.NORMALIZED)


def evaluate_document(
    document_id: str,
    schema_name: str,
    expected: dict[str, Any],
    extracted: dict[str, Any],
    match_level: MatchLevel = MatchLevel.NORMALIZED,
    field_match_levels: dict[str, MatchLevel] | None = None,
    extraction_time_ms: int = 0,
) -> DocumentMetrics:
    """
    Evaluate extraction accuracy for a single document.

    Args:
        document_id: Unique identifier for the document.
        schema_name: Schema used for extraction.
        expected: Ground-truth field values.
        extracted: Extracted field values.
        match_level: Default match level for all fields.
        field_match_levels: Per-field override for match level.
        extraction_time_ms: How long extraction took.

    Returns:
        DocumentMetrics with per-field results.
    """
    field_match_levels = field_match_levels or {}
    all_fields = set(expected.keys()) | set(extracted.keys())

    results: list[FieldMatchResult] = []
    for name in sorted(all_fields):
        level = field_match_levels.get(name, match_level)
        result = compare_field(
            field_name=name,
            expected=expected.get(name),
            extracted=extracted.get(name),
            match_level=level,
        )
        results.append(result)

    return DocumentMetrics(
        document_id=document_id,
        schema_name=schema_name,
        field_results=results,
        extraction_time_ms=extraction_time_ms,
    )
