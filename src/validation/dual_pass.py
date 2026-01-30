"""
Dual-pass extraction comparison for anti-hallucination validation.

Implements Layer 2 of the 3-Layer Anti-Hallucination System:
- Pass 1: Completeness-focused extraction
- Pass 2: Accuracy-focused extraction
- Field-by-field comparison with mismatch detection
- Confidence scoring based on agreement

The dual-pass approach detects hallucinations by requiring agreement
between two independent extraction passes with different prompts.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.config import get_logger
from src.utils import normalize_whitespace
from src.utils.string_utils import similarity_ratio


logger = get_logger(__name__)


class ComparisonResult(str, Enum):
    """Result of comparing two extraction passes."""

    EXACT_MATCH = "exact_match"
    FUZZY_MATCH = "fuzzy_match"
    PARTIAL_MATCH = "partial_match"
    MISMATCH = "mismatch"
    PASS1_ONLY = "pass1_only"
    PASS2_ONLY = "pass2_only"
    BOTH_EMPTY = "both_empty"


class MergeStrategy(str, Enum):
    """Strategy for merging field values from two passes."""

    PREFER_PASS1 = "prefer_pass1"
    PREFER_PASS2 = "prefer_pass2"
    PREFER_LONGER = "prefer_longer"
    PREFER_HIGHER_CONFIDENCE = "prefer_higher_confidence"
    REQUIRE_AGREEMENT = "require_agreement"


@dataclass(frozen=True, slots=True)
class FieldComparison:
    """
    Result of comparing a single field between two extraction passes.

    Attributes:
        field_name: Name of the compared field.
        pass1_value: Value from first extraction pass.
        pass2_value: Value from second extraction pass.
        result: Comparison result category.
        similarity_score: Normalized similarity score 0.0-1.0.
        merged_value: Final merged value for output.
        merge_confidence: Confidence in the merged value.
        requires_review: Whether field requires human review.
        notes: Additional comparison notes.
    """

    field_name: str
    pass1_value: Any
    pass2_value: Any
    result: ComparisonResult
    similarity_score: float
    merged_value: Any
    merge_confidence: float
    requires_review: bool = False
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "field_name": self.field_name,
            "pass1_value": self.pass1_value,
            "pass2_value": self.pass2_value,
            "result": self.result.value,
            "similarity_score": self.similarity_score,
            "merged_value": self.merged_value,
            "merge_confidence": self.merge_confidence,
            "requires_review": self.requires_review,
            "notes": self.notes,
        }


@dataclass(slots=True)
class DualPassResult:
    """
    Complete result of dual-pass comparison.

    Attributes:
        field_comparisons: Individual field comparison results.
        overall_agreement_rate: Percentage of fields that agree.
        overall_confidence: Combined confidence score.
        high_confidence_fields: Fields with high confidence (>=0.85).
        low_confidence_fields: Fields with low confidence (<0.50).
        mismatch_fields: Fields with mismatched values.
        requires_retry: Whether extraction should be retried.
        requires_human_review: Whether human review is needed.
        merged_output: Final merged extraction output.
    """

    field_comparisons: dict[str, FieldComparison] = field(default_factory=dict)
    overall_agreement_rate: float = 0.0
    overall_confidence: float = 0.0
    high_confidence_fields: list[str] = field(default_factory=list)
    low_confidence_fields: list[str] = field(default_factory=list)
    mismatch_fields: list[str] = field(default_factory=list)
    requires_retry: bool = False
    requires_human_review: bool = False
    merged_output: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "field_comparisons": {k: v.to_dict() for k, v in self.field_comparisons.items()},
            "overall_agreement_rate": self.overall_agreement_rate,
            "overall_confidence": self.overall_confidence,
            "high_confidence_fields": self.high_confidence_fields,
            "low_confidence_fields": self.low_confidence_fields,
            "mismatch_fields": self.mismatch_fields,
            "requires_retry": self.requires_retry,
            "requires_human_review": self.requires_human_review,
            "merged_output": self.merged_output,
        }


class DualPassComparator:
    """
    Compares and merges dual-pass extraction results.

    This comparator implements the core dual-pass anti-hallucination logic:
    1. Compares each field between Pass 1 and Pass 2
    2. Calculates similarity scores for each field
    3. Determines merge strategy based on agreement
    4. Computes overall confidence and agreement metrics
    5. Identifies fields requiring human review

    Thresholds:
        - Exact match: similarity >= 0.99
        - Fuzzy match: similarity >= 0.85
        - Partial match: similarity >= 0.50
        - Mismatch: similarity < 0.50

    Example:
        comparator = DualPassComparator()
        result = comparator.compare(pass1_data, pass2_data)

        if result.requires_human_review:
            print(f"Fields needing review: {result.mismatch_fields}")
    """

    EXACT_MATCH_THRESHOLD: float = 0.99
    FUZZY_MATCH_THRESHOLD: float = 0.85
    PARTIAL_MATCH_THRESHOLD: float = 0.50
    HIGH_CONFIDENCE_THRESHOLD: float = 0.85
    LOW_CONFIDENCE_THRESHOLD: float = 0.50
    AGREEMENT_RATE_FOR_RETRY: float = 0.70
    AGREEMENT_RATE_FOR_HUMAN_REVIEW: float = 0.50

    def __init__(
        self,
        default_strategy: MergeStrategy = MergeStrategy.PREFER_HIGHER_CONFIDENCE,
        field_strategies: dict[str, MergeStrategy] | None = None,
        required_fields: list[str] | None = None,
    ) -> None:
        """
        Initialize the dual-pass comparator.

        Args:
            default_strategy: Default merge strategy for fields.
            field_strategies: Per-field merge strategy overrides.
            required_fields: Fields that must have values.
        """
        self.default_strategy = default_strategy
        self.field_strategies = field_strategies or {}
        self.required_fields = set(required_fields or [])

    def compare(
        self,
        pass1_data: dict[str, Any],
        pass2_data: dict[str, Any],
        pass1_confidence: dict[str, float] | None = None,
        pass2_confidence: dict[str, float] | None = None,
    ) -> DualPassResult:
        """
        Compare two extraction passes and produce merged result.

        Args:
            pass1_data: Extracted data from Pass 1 (completeness-focused).
            pass2_data: Extracted data from Pass 2 (accuracy-focused).
            pass1_confidence: Per-field confidence scores from Pass 1.
            pass2_confidence: Per-field confidence scores from Pass 2.

        Returns:
            DualPassResult with comparisons, confidence, and merged output.
        """
        pass1_confidence = pass1_confidence or {}
        pass2_confidence = pass2_confidence or {}

        # Get all unique field names from both passes
        all_fields = set(pass1_data.keys()) | set(pass2_data.keys())

        # Compare each field
        comparisons: dict[str, FieldComparison] = {}
        for field_name in all_fields:
            comparison = self._compare_field(
                field_name=field_name,
                pass1_value=pass1_data.get(field_name),
                pass2_value=pass2_data.get(field_name),
                pass1_conf=pass1_confidence.get(field_name, 0.7),
                pass2_conf=pass2_confidence.get(field_name, 0.7),
            )
            comparisons[field_name] = comparison

        # Build result
        result = self._build_result(comparisons)

        logger.debug(
            f"Dual-pass comparison complete: "
            f"agreement_rate={result.overall_agreement_rate:.2%}, "
            f"confidence={result.overall_confidence:.2f}, "
            f"mismatches={len(result.mismatch_fields)}"
        )

        return result

    def _compare_field(
        self,
        field_name: str,
        pass1_value: Any,
        pass2_value: Any,
        pass1_conf: float,
        pass2_conf: float,
    ) -> FieldComparison:
        """
        Compare a single field between two passes.

        Args:
            field_name: Name of the field.
            pass1_value: Value from Pass 1.
            pass2_value: Value from Pass 2.
            pass1_conf: Confidence from Pass 1.
            pass2_conf: Confidence from Pass 2.

        Returns:
            FieldComparison with result and merged value.
        """
        # Handle empty values
        p1_empty = self._is_empty(pass1_value)
        p2_empty = self._is_empty(pass2_value)

        if p1_empty and p2_empty:
            return FieldComparison(
                field_name=field_name,
                pass1_value=pass1_value,
                pass2_value=pass2_value,
                result=ComparisonResult.BOTH_EMPTY,
                similarity_score=1.0,
                merged_value=None,
                merge_confidence=0.0,
                requires_review=field_name in self.required_fields,
                notes="Both passes returned empty"
                + (" (required field)" if field_name in self.required_fields else ""),
            )

        if p1_empty:
            return FieldComparison(
                field_name=field_name,
                pass1_value=pass1_value,
                pass2_value=pass2_value,
                result=ComparisonResult.PASS2_ONLY,
                similarity_score=0.5,
                merged_value=pass2_value,
                merge_confidence=pass2_conf * 0.8,
                requires_review=True,
                notes="Value found only in Pass 2",
            )

        if p2_empty:
            return FieldComparison(
                field_name=field_name,
                pass1_value=pass1_value,
                pass2_value=pass2_value,
                result=ComparisonResult.PASS1_ONLY,
                similarity_score=0.5,
                merged_value=pass1_value,
                merge_confidence=pass1_conf * 0.8,
                requires_review=True,
                notes="Value found only in Pass 1",
            )

        # Calculate similarity between non-empty values
        similarity = self._calculate_similarity(pass1_value, pass2_value)

        # Determine comparison result based on similarity
        if similarity >= self.EXACT_MATCH_THRESHOLD:
            result_type = ComparisonResult.EXACT_MATCH
            merge_confidence = max(pass1_conf, pass2_conf)
            requires_review = False
        elif similarity >= self.FUZZY_MATCH_THRESHOLD:
            result_type = ComparisonResult.FUZZY_MATCH
            merge_confidence = (pass1_conf + pass2_conf) / 2 * 0.95
            requires_review = False
        elif similarity >= self.PARTIAL_MATCH_THRESHOLD:
            result_type = ComparisonResult.PARTIAL_MATCH
            merge_confidence = (pass1_conf + pass2_conf) / 2 * 0.75
            requires_review = True
        else:
            result_type = ComparisonResult.MISMATCH
            merge_confidence = min(pass1_conf, pass2_conf) * 0.5
            requires_review = True

        # Determine merged value based on strategy
        merged_value = self._merge_values(
            field_name=field_name,
            pass1_value=pass1_value,
            pass2_value=pass2_value,
            pass1_conf=pass1_conf,
            pass2_conf=pass2_conf,
            result_type=result_type,
        )

        notes = self._generate_comparison_notes(result_type, similarity, pass1_conf, pass2_conf)

        return FieldComparison(
            field_name=field_name,
            pass1_value=pass1_value,
            pass2_value=pass2_value,
            result=result_type,
            similarity_score=similarity,
            merged_value=merged_value,
            merge_confidence=merge_confidence,
            requires_review=requires_review,
            notes=notes,
        )

    def _is_empty(self, value: Any) -> bool:
        """Check if a value is considered empty."""
        if value is None:
            return True
        if isinstance(value, str) and not value.strip():
            return True
        if isinstance(value, (list, dict)) and len(value) == 0:
            return True
        return False

    def _calculate_similarity(self, value1: Any, value2: Any) -> float:
        """
        Calculate similarity score between two values.

        Args:
            value1: First value.
            value2: Second value.

        Returns:
            Similarity score between 0.0 and 1.0.
        """
        # Handle identical values
        if value1 == value2:
            return 1.0

        # Convert to strings for comparison
        str1 = self._normalize_value(value1)
        str2 = self._normalize_value(value2)

        # Exact string match after normalization
        if str1 == str2:
            return 0.99

        # Handle numeric comparison
        if isinstance(value1, (int, float)) and isinstance(value2, (int, float)):
            return self._numeric_similarity(float(value1), float(value2))

        # Try to parse as numbers if strings look numeric
        try:
            num1 = self._extract_number(str1)
            num2 = self._extract_number(str2)
            if num1 is not None and num2 is not None:
                return self._numeric_similarity(num1, num2)
        except (ValueError, TypeError):
            pass

        # String similarity (returns float 0.0-1.0, not bool)
        return similarity_ratio(str1, str2)

    def _normalize_value(self, value: Any) -> str:
        """Normalize a value to string for comparison."""
        if value is None:
            return ""
        if isinstance(value, str):
            return normalize_whitespace(value.strip().lower())
        if isinstance(value, (list, tuple)):
            return " ".join(self._normalize_value(v) for v in value)
        if isinstance(value, dict):
            return str(sorted(value.items()))
        return str(value).strip().lower()

    def _extract_number(self, text: str) -> float | None:
        """Extract numeric value from text, handling currency and formatting."""
        import re

        # Remove currency symbols, commas, and whitespace
        cleaned = re.sub(r"[$,\s]", "", text)
        # Handle parentheses for negative numbers
        if cleaned.startswith("(") and cleaned.endswith(")"):
            cleaned = "-" + cleaned[1:-1]
        try:
            return float(cleaned)
        except ValueError:
            return None

    def _numeric_similarity(self, num1: float, num2: float) -> float:
        """Calculate similarity between two numbers."""
        if num1 == num2:
            return 1.0
        if num1 == 0 and num2 == 0:
            return 1.0
        if num1 == 0 or num2 == 0:
            return 0.0

        # Calculate relative difference
        max_val = max(abs(num1), abs(num2))
        diff = abs(num1 - num2)
        relative_diff = diff / max_val

        # Convert to similarity score
        if relative_diff < 0.001:
            return 0.99
        if relative_diff < 0.01:
            return 0.95
        if relative_diff < 0.05:
            return 0.85
        if relative_diff < 0.10:
            return 0.70
        if relative_diff < 0.25:
            return 0.50
        return max(0.0, 1.0 - relative_diff)

    def _merge_values(
        self,
        field_name: str,
        pass1_value: Any,
        pass2_value: Any,
        pass1_conf: float,
        pass2_conf: float,
        result_type: ComparisonResult,
    ) -> Any:
        """
        Merge values from two passes based on strategy.

        Args:
            field_name: Name of the field.
            pass1_value: Value from Pass 1.
            pass2_value: Value from Pass 2.
            pass1_conf: Confidence from Pass 1.
            pass2_conf: Confidence from Pass 2.
            result_type: Type of comparison result.

        Returns:
            Merged value for the field.
        """
        # Get strategy for this field
        strategy = self.field_strategies.get(field_name, self.default_strategy)

        # For exact matches, use either value
        if result_type == ComparisonResult.EXACT_MATCH:
            return pass1_value

        # Apply strategy
        if strategy == MergeStrategy.PREFER_PASS1:
            return pass1_value

        if strategy == MergeStrategy.PREFER_PASS2:
            return pass2_value

        if strategy == MergeStrategy.PREFER_LONGER:
            str1 = str(pass1_value) if pass1_value else ""
            str2 = str(pass2_value) if pass2_value else ""
            return pass1_value if len(str1) >= len(str2) else pass2_value

        if strategy == MergeStrategy.REQUIRE_AGREEMENT:
            if result_type in (
                ComparisonResult.EXACT_MATCH,
                ComparisonResult.FUZZY_MATCH,
            ):
                return pass1_value
            return None

        # Default: PREFER_HIGHER_CONFIDENCE
        return pass1_value if pass1_conf >= pass2_conf else pass2_value

    def _generate_comparison_notes(
        self,
        result_type: ComparisonResult,
        similarity: float,
        pass1_conf: float,
        pass2_conf: float,
    ) -> str:
        """Generate human-readable notes for comparison."""
        notes_parts = []

        if result_type == ComparisonResult.EXACT_MATCH:
            notes_parts.append("Exact match between passes")
        elif result_type == ComparisonResult.FUZZY_MATCH:
            notes_parts.append(f"Fuzzy match (similarity: {similarity:.2%})")
        elif result_type == ComparisonResult.PARTIAL_MATCH:
            notes_parts.append(f"Partial match (similarity: {similarity:.2%})")
        elif result_type == ComparisonResult.MISMATCH:
            notes_parts.append(f"Values differ significantly (similarity: {similarity:.2%})")

        if abs(pass1_conf - pass2_conf) > 0.2:
            higher = "Pass 1" if pass1_conf > pass2_conf else "Pass 2"
            notes_parts.append(f"{higher} has higher confidence")

        return "; ".join(notes_parts)

    def _build_result(
        self,
        comparisons: dict[str, FieldComparison],
    ) -> DualPassResult:
        """
        Build complete comparison result from field comparisons.

        Args:
            comparisons: Dict of field name to FieldComparison.

        Returns:
            Complete DualPassResult with all metrics.
        """
        result = DualPassResult()
        result.field_comparisons = comparisons

        if not comparisons:
            return result

        # Calculate agreement rate
        agreement_results = {
            ComparisonResult.EXACT_MATCH,
            ComparisonResult.FUZZY_MATCH,
            ComparisonResult.BOTH_EMPTY,
        }
        agreed = sum(1 for c in comparisons.values() if c.result in agreement_results)
        result.overall_agreement_rate = agreed / len(comparisons)

        # Calculate overall confidence
        confidences = [c.merge_confidence for c in comparisons.values()]
        result.overall_confidence = sum(confidences) / len(confidences)

        # Categorize fields by confidence
        for name, comp in comparisons.items():
            if comp.merge_confidence >= self.HIGH_CONFIDENCE_THRESHOLD:
                result.high_confidence_fields.append(name)
            elif comp.merge_confidence < self.LOW_CONFIDENCE_THRESHOLD:
                result.low_confidence_fields.append(name)

            if comp.result == ComparisonResult.MISMATCH:
                result.mismatch_fields.append(name)

        # Build merged output
        result.merged_output = {
            name: comp.merged_value
            for name, comp in comparisons.items()
            if comp.merged_value is not None
        }

        # Determine if retry or human review is needed
        if result.overall_agreement_rate < self.AGREEMENT_RATE_FOR_HUMAN_REVIEW:
            result.requires_human_review = True
        elif result.overall_agreement_rate < self.AGREEMENT_RATE_FOR_RETRY:
            result.requires_retry = True

        # Check for required fields with low confidence
        for req_field in self.required_fields:
            if req_field in comparisons:
                comp = comparisons[req_field]
                if comp.merge_confidence < self.LOW_CONFIDENCE_THRESHOLD:
                    result.requires_human_review = True
                    break

        return result


def compare_extractions(
    pass1_data: dict[str, Any],
    pass2_data: dict[str, Any],
    pass1_confidence: dict[str, float] | None = None,
    pass2_confidence: dict[str, float] | None = None,
    required_fields: list[str] | None = None,
) -> DualPassResult:
    """
    Compare two extraction passes and produce merged result.

    Convenience function for one-off comparisons without creating
    a comparator instance.

    Args:
        pass1_data: Extracted data from Pass 1.
        pass2_data: Extracted data from Pass 2.
        pass1_confidence: Per-field confidence from Pass 1.
        pass2_confidence: Per-field confidence from Pass 2.
        required_fields: Fields that must have values.

    Returns:
        DualPassResult with comparisons and merged output.

    Example:
        result = compare_extractions(
            pass1_data={"patient_name": "John Smith", "dob": "1990-01-15"},
            pass2_data={"patient_name": "John Smith", "dob": "01/15/1990"},
        )
        print(f"Agreement: {result.overall_agreement_rate:.0%}")
    """
    comparator = DualPassComparator(required_fields=required_fields)
    return comparator.compare(
        pass1_data=pass1_data,
        pass2_data=pass2_data,
        pass1_confidence=pass1_confidence,
        pass2_confidence=pass2_confidence,
    )
