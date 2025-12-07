"""
Confidence scoring system for document extraction.

Implements comprehensive confidence scoring based on:
- Field-level extraction confidence from VLM
- Dual-pass agreement rates
- Validation results
- Hallucination pattern detection
- Cross-field consistency

Thresholds:
    >= 0.85 (HIGH): Auto-accept extraction
    0.50 - 0.84 (MEDIUM): Retry up to 2 times
    < 0.50 (LOW): Route to human review
"""

from dataclasses import dataclass, field
from enum import Enum
from functools import lru_cache
from typing import Any

from src.config import get_logger


logger = get_logger(__name__)


class ConfidenceLevel(str, Enum):
    """Classification of confidence levels."""

    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class ConfidenceAction(str, Enum):
    """Recommended action based on confidence."""

    AUTO_ACCEPT = "auto_accept"
    RETRY = "retry"
    HUMAN_REVIEW = "human_review"


@dataclass(frozen=True, slots=True)
class FieldConfidence:
    """
    Confidence details for a single field.

    Attributes:
        field_name: Name of the field.
        extraction_confidence: Raw confidence from VLM extraction.
        validation_confidence: Confidence from validation checks.
        agreement_confidence: Confidence from dual-pass agreement.
        pattern_confidence: Confidence after pattern detection.
        combined_confidence: Final weighted confidence score.
        level: Confidence level classification.
        factors: Contributing factors to confidence.
    """

    field_name: str
    extraction_confidence: float
    validation_confidence: float
    agreement_confidence: float
    pattern_confidence: float
    combined_confidence: float
    level: ConfidenceLevel
    factors: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "field_name": self.field_name,
            "extraction_confidence": self.extraction_confidence,
            "validation_confidence": self.validation_confidence,
            "agreement_confidence": self.agreement_confidence,
            "pattern_confidence": self.pattern_confidence,
            "combined_confidence": self.combined_confidence,
            "level": self.level.value,
            "factors": list(self.factors),
        }


@dataclass(slots=True)
class ExtractionConfidence:
    """
    Complete confidence assessment for an extraction.

    Attributes:
        field_confidences: Per-field confidence details.
        overall_confidence: Weighted overall confidence.
        overall_level: Overall confidence level.
        recommended_action: Recommended next action.
        high_confidence_fields: Fields with high confidence.
        medium_confidence_fields: Fields with medium confidence.
        low_confidence_fields: Fields with low confidence.
        critical_fields_status: Status of critical/required fields.
        summary: Human-readable summary.
    """

    field_confidences: dict[str, FieldConfidence] = field(default_factory=dict)
    overall_confidence: float = 0.0
    overall_level: ConfidenceLevel = ConfidenceLevel.LOW
    recommended_action: ConfidenceAction = ConfidenceAction.HUMAN_REVIEW
    high_confidence_fields: list[str] = field(default_factory=list)
    medium_confidence_fields: list[str] = field(default_factory=list)
    low_confidence_fields: list[str] = field(default_factory=list)
    critical_fields_status: dict[str, bool] = field(default_factory=dict)
    summary: str = ""

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "field_confidences": {
                k: v.to_dict() for k, v in self.field_confidences.items()
            },
            "overall_confidence": self.overall_confidence,
            "overall_level": self.overall_level.value,
            "recommended_action": self.recommended_action.value,
            "high_confidence_fields": self.high_confidence_fields,
            "medium_confidence_fields": self.medium_confidence_fields,
            "low_confidence_fields": self.low_confidence_fields,
            "critical_fields_status": self.critical_fields_status,
            "summary": self.summary,
        }


class ConfidenceScorer:
    """
    Calculates comprehensive confidence scores for extractions.

    Combines multiple confidence signals:
    - Raw VLM extraction confidence
    - Dual-pass agreement
    - Validation results
    - Hallucination pattern detection

    Weights can be customized for different document types.

    Example:
        scorer = ConfidenceScorer(critical_fields=["patient_name", "npi"])
        result = scorer.calculate(
            extraction_confidences={"patient_name": 0.95, "npi": 0.87},
            agreement_scores={"patient_name": 1.0, "npi": 0.9},
            validation_results={"patient_name": True, "npi": True},
        )

        if result.recommended_action == ConfidenceAction.AUTO_ACCEPT:
            print("Extraction can be auto-accepted")
    """

    HIGH_THRESHOLD: float = 0.85
    MEDIUM_THRESHOLD: float = 0.50
    MAX_RETRIES: int = 2

    # Default weights for combining confidence sources
    DEFAULT_WEIGHTS = {
        "extraction": 0.35,
        "agreement": 0.30,
        "validation": 0.20,
        "pattern": 0.15,
    }

    # Penalty factors for various issues
    PENALTIES = {
        "validation_failed": 0.30,
        "no_agreement": 0.25,
        "pattern_detected": 0.20,
        "missing_required": 0.40,
        "single_pass_only": 0.15,
    }

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        critical_fields: list[str] | None = None,
        field_weights: dict[str, float] | None = None,
    ) -> None:
        """
        Initialize the confidence scorer.

        Args:
            weights: Custom weights for confidence sources.
            critical_fields: Fields that are critical for the document.
            field_weights: Per-field importance weights.
        """
        self.weights = weights or self.DEFAULT_WEIGHTS.copy()
        self.critical_fields = set(critical_fields or [])
        self.field_weights = field_weights or {}

        # Normalize weights
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}

    def calculate(
        self,
        extraction_confidences: dict[str, float],
        agreement_scores: dict[str, float] | None = None,
        validation_results: dict[str, bool] | None = None,
        pattern_flags: set[str] | None = None,
        retry_count: int = 0,
    ) -> ExtractionConfidence:
        """
        Calculate comprehensive confidence scores.

        Args:
            extraction_confidences: Per-field extraction confidence from VLM.
            agreement_scores: Per-field dual-pass agreement scores.
            validation_results: Per-field validation pass/fail.
            pattern_flags: Fields flagged by pattern detection.
            retry_count: Current retry attempt count.

        Returns:
            ExtractionConfidence with all confidence details.
        """
        agreement_scores = agreement_scores or {}
        validation_results = validation_results or {}
        pattern_flags = pattern_flags or set()

        # Calculate per-field confidence
        field_confs: dict[str, FieldConfidence] = {}
        all_fields = set(extraction_confidences.keys())

        for field_name in all_fields:
            field_conf = self._calculate_field_confidence(
                field_name=field_name,
                extraction_conf=extraction_confidences.get(field_name, 0.5),
                agreement_score=agreement_scores.get(field_name, 1.0),
                validation_passed=validation_results.get(field_name, True),
                has_pattern_flag=field_name in pattern_flags,
            )
            field_confs[field_name] = field_conf

        # Build result
        result = self._build_result(field_confs, retry_count)

        logger.debug(
            f"Confidence calculated: "
            f"overall={result.overall_confidence:.2f}, "
            f"level={result.overall_level.value}, "
            f"action={result.recommended_action.value}"
        )

        return result

    def _calculate_field_confidence(
        self,
        field_name: str,
        extraction_conf: float,
        agreement_score: float,
        validation_passed: bool,
        has_pattern_flag: bool,
    ) -> FieldConfidence:
        """
        Calculate confidence for a single field.

        Args:
            field_name: Name of the field.
            extraction_conf: Raw extraction confidence.
            agreement_score: Dual-pass agreement score.
            validation_passed: Whether validation passed.
            has_pattern_flag: Whether pattern detection flagged this field.

        Returns:
            FieldConfidence with all details.
        """
        factors: list[str] = []

        # Validation confidence
        if validation_passed:
            validation_conf = 1.0
        else:
            validation_conf = 1.0 - self.PENALTIES["validation_failed"]
            factors.append("validation_failed")

        # Pattern confidence
        if has_pattern_flag:
            pattern_conf = 1.0 - self.PENALTIES["pattern_detected"]
            factors.append("pattern_detected")
        else:
            pattern_conf = 1.0

        # Agreement confidence (already a score)
        agreement_conf = agreement_score
        if agreement_score < 0.85:
            factors.append("low_agreement")
        if agreement_score == 1.0 and extraction_conf < 0.8:
            factors.append("single_pass_estimate")

        # Weighted combination
        combined = (
            self.weights["extraction"] * extraction_conf
            + self.weights["agreement"] * agreement_conf
            + self.weights["validation"] * validation_conf
            + self.weights["pattern"] * pattern_conf
        )

        # Apply field-specific weight if exists
        field_weight = self.field_weights.get(field_name, 1.0)
        combined *= field_weight

        # Ensure bounds
        combined = max(0.0, min(1.0, combined))

        # Determine level
        if combined >= self.HIGH_THRESHOLD:
            level = ConfidenceLevel.HIGH
        elif combined >= self.MEDIUM_THRESHOLD:
            level = ConfidenceLevel.MEDIUM
        else:
            level = ConfidenceLevel.LOW

        return FieldConfidence(
            field_name=field_name,
            extraction_confidence=extraction_conf,
            validation_confidence=validation_conf,
            agreement_confidence=agreement_conf,
            pattern_confidence=pattern_conf,
            combined_confidence=combined,
            level=level,
            factors=tuple(factors),
        )

    def _build_result(
        self,
        field_confs: dict[str, FieldConfidence],
        retry_count: int,
    ) -> ExtractionConfidence:
        """
        Build complete confidence result.

        Args:
            field_confs: Per-field confidence details.
            retry_count: Current retry count.

        Returns:
            Complete ExtractionConfidence.
        """
        result = ExtractionConfidence()
        result.field_confidences = field_confs

        if not field_confs:
            result.summary = "No fields to assess"
            return result

        # Categorize fields by confidence level
        for name, conf in field_confs.items():
            if conf.level == ConfidenceLevel.HIGH:
                result.high_confidence_fields.append(name)
            elif conf.level == ConfidenceLevel.MEDIUM:
                result.medium_confidence_fields.append(name)
            else:
                result.low_confidence_fields.append(name)

        # Check critical fields
        for critical_field in self.critical_fields:
            if critical_field in field_confs:
                conf = field_confs[critical_field]
                result.critical_fields_status[critical_field] = (
                    conf.level != ConfidenceLevel.LOW
                )
            else:
                result.critical_fields_status[critical_field] = False

        # Calculate overall confidence with field weights
        total_weight = 0.0
        weighted_sum = 0.0

        for name, conf in field_confs.items():
            # Critical fields have higher weight
            weight = 2.0 if name in self.critical_fields else 1.0
            weight *= self.field_weights.get(name, 1.0)

            weighted_sum += conf.combined_confidence * weight
            total_weight += weight

        result.overall_confidence = weighted_sum / total_weight if total_weight > 0 else 0.0

        # Penalize if critical fields have issues
        critical_issues = sum(
            1 for status in result.critical_fields_status.values()
            if not status
        )
        if critical_issues > 0:
            penalty = self.PENALTIES["missing_required"] * (critical_issues / max(len(self.critical_fields), 1))
            result.overall_confidence = max(0.0, result.overall_confidence - penalty)

        # Determine overall level
        if result.overall_confidence >= self.HIGH_THRESHOLD:
            result.overall_level = ConfidenceLevel.HIGH
        elif result.overall_confidence >= self.MEDIUM_THRESHOLD:
            result.overall_level = ConfidenceLevel.MEDIUM
        else:
            result.overall_level = ConfidenceLevel.LOW

        # Determine recommended action
        result.recommended_action = self._determine_action(
            level=result.overall_level,
            retry_count=retry_count,
            has_critical_issues=critical_issues > 0,
        )

        # Generate summary
        result.summary = self._generate_summary(result)

        return result

    @staticmethod
    @lru_cache(maxsize=32)
    def _determine_action_cached(
        level_value: str,
        retry_count: int,
        has_critical_issues: bool,
        max_retries: int,
    ) -> ConfidenceAction:
        """
        Determine recommended action based on confidence (cached).

        Args:
            level_value: Overall confidence level value (string for hashability).
            retry_count: Current retry count.
            has_critical_issues: Whether critical fields have issues.
            max_retries: Maximum retry count.

        Returns:
            Recommended ConfidenceAction.
        """
        # Critical issues always require human review
        if has_critical_issues:
            return ConfidenceAction.HUMAN_REVIEW

        if level_value == "high":
            return ConfidenceAction.AUTO_ACCEPT

        if level_value == "medium":
            if retry_count < max_retries:
                return ConfidenceAction.RETRY
            return ConfidenceAction.HUMAN_REVIEW

        # LOW confidence
        if retry_count < max_retries:
            return ConfidenceAction.RETRY
        return ConfidenceAction.HUMAN_REVIEW

    def _determine_action(
        self,
        level: ConfidenceLevel,
        retry_count: int,
        has_critical_issues: bool,
    ) -> ConfidenceAction:
        """
        Determine recommended action based on confidence.

        Args:
            level: Overall confidence level.
            retry_count: Current retry count.
            has_critical_issues: Whether critical fields have issues.

        Returns:
            Recommended ConfidenceAction.
        """
        return self._determine_action_cached(
            level.value, retry_count, has_critical_issues, self.MAX_RETRIES
        )

    def _generate_summary(self, result: ExtractionConfidence) -> str:
        """Generate human-readable summary."""
        parts = []

        # Overall assessment
        parts.append(
            f"Overall confidence: {result.overall_confidence:.1%} ({result.overall_level.value})"
        )

        # Field breakdown
        high = len(result.high_confidence_fields)
        medium = len(result.medium_confidence_fields)
        low = len(result.low_confidence_fields)
        total = high + medium + low

        parts.append(
            f"Fields: {high}/{total} high, {medium}/{total} medium, {low}/{total} low"
        )

        # Critical fields status
        if result.critical_fields_status:
            passed = sum(1 for v in result.critical_fields_status.values() if v)
            total_critical = len(result.critical_fields_status)
            parts.append(f"Critical fields: {passed}/{total_critical} passed")

        # Recommended action
        action_descriptions = {
            ConfidenceAction.AUTO_ACCEPT: "Ready for auto-acceptance",
            ConfidenceAction.RETRY: "Retry recommended",
            ConfidenceAction.HUMAN_REVIEW: "Human review required",
        }
        parts.append(f"Action: {action_descriptions[result.recommended_action]}")

        return ". ".join(parts)


class AdaptiveConfidenceScorer(ConfidenceScorer):
    """
    Confidence scorer that adapts based on document type and history.

    Extends ConfidenceScorer with:
    - Document type-specific thresholds
    - Historical accuracy tracking
    - Dynamic weight adjustment

    Example:
        scorer = AdaptiveConfidenceScorer(
            document_type="cms1500",
            historical_accuracy=0.95,
        )
    """

    DOCUMENT_THRESHOLDS = {
        "cms1500": {"high": 0.88, "medium": 0.55},
        "ub04": {"high": 0.90, "medium": 0.60},
        "eob": {"high": 0.85, "medium": 0.50},
        "default": {"high": 0.85, "medium": 0.50},
    }

    DOCUMENT_CRITICAL_FIELDS = {
        "cms1500": [
            "patient_name",
            "subscriber_id",
            "billing_npi",
            "diagnosis_codes",
            "total_charges",
        ],
        "ub04": [
            "patient_name",
            "medical_record_number",
            "total_charges",
            "admission_date",
        ],
        "eob": [
            "patient_name",
            "claim_number",
            "amount_paid",
            "amount_owed",
        ],
    }

    def __init__(
        self,
        document_type: str = "default",
        historical_accuracy: float | None = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize adaptive scorer.

        Args:
            document_type: Type of document being processed.
            historical_accuracy: Historical accuracy for this document type.
            **kwargs: Additional arguments for ConfidenceScorer.
        """
        # Get document-specific settings
        thresholds = self.DOCUMENT_THRESHOLDS.get(
            document_type,
            self.DOCUMENT_THRESHOLDS["default"],
        )
        critical = self.DOCUMENT_CRITICAL_FIELDS.get(document_type, [])

        # Merge with provided critical fields
        provided_critical = kwargs.pop("critical_fields", [])
        all_critical = list(set(critical + provided_critical))

        super().__init__(critical_fields=all_critical, **kwargs)

        # Set document-specific thresholds
        self.HIGH_THRESHOLD = thresholds["high"]
        self.MEDIUM_THRESHOLD = thresholds["medium"]

        self.document_type = document_type
        self.historical_accuracy = historical_accuracy

        # Adjust weights if historical accuracy is known
        if historical_accuracy is not None:
            self._adjust_weights_for_accuracy(historical_accuracy)

    def _adjust_weights_for_accuracy(self, accuracy: float) -> None:
        """
        Adjust weights based on historical accuracy.

        Higher historical accuracy means we can trust extraction more.
        Lower accuracy means we rely more on validation and agreement.
        """
        if accuracy >= 0.95:
            # High historical accuracy - trust extraction more
            self.weights["extraction"] = 0.40
            self.weights["agreement"] = 0.25
            self.weights["validation"] = 0.20
            self.weights["pattern"] = 0.15
        elif accuracy >= 0.85:
            # Good accuracy - balanced weights
            self.weights["extraction"] = 0.35
            self.weights["agreement"] = 0.30
            self.weights["validation"] = 0.20
            self.weights["pattern"] = 0.15
        else:
            # Lower accuracy - rely more on validation
            self.weights["extraction"] = 0.25
            self.weights["agreement"] = 0.30
            self.weights["validation"] = 0.30
            self.weights["pattern"] = 0.15

        # Normalize
        total = sum(self.weights.values())
        self.weights = {k: v / total for k, v in self.weights.items()}


def calculate_confidence(
    extraction_confidences: dict[str, float],
    agreement_scores: dict[str, float] | None = None,
    validation_results: dict[str, bool] | None = None,
    pattern_flags: set[str] | None = None,
    critical_fields: list[str] | None = None,
    retry_count: int = 0,
) -> ExtractionConfidence:
    """
    Calculate comprehensive extraction confidence.

    Convenience function for one-off calculations.

    Args:
        extraction_confidences: Per-field extraction confidence.
        agreement_scores: Per-field dual-pass agreement.
        validation_results: Per-field validation results.
        pattern_flags: Fields flagged by pattern detection.
        critical_fields: Critical/required fields.
        retry_count: Current retry count.

    Returns:
        ExtractionConfidence with all details.

    Example:
        result = calculate_confidence(
            extraction_confidences={"name": 0.95, "dob": 0.88},
            validation_results={"name": True, "dob": True},
        )
        print(f"Action: {result.recommended_action.value}")
    """
    scorer = ConfidenceScorer(critical_fields=critical_fields)
    return scorer.calculate(
        extraction_confidences=extraction_confidences,
        agreement_scores=agreement_scores,
        validation_results=validation_results,
        pattern_flags=pattern_flags,
        retry_count=retry_count,
    )


@lru_cache(maxsize=128)
def get_confidence_level(confidence: float) -> ConfidenceLevel:
    """
    Get confidence level classification for a score (cached).

    Uses standard thresholds: HIGH >= 0.85, MEDIUM >= 0.50, LOW < 0.50

    Args:
        confidence: Confidence score 0.0-1.0 (rounded to 2 decimals for caching).

    Returns:
        ConfidenceLevel classification.
    """
    # Round to 2 decimal places for effective caching
    rounded = round(confidence, 2)
    if rounded >= 0.85:
        return ConfidenceLevel.HIGH
    if rounded >= 0.50:
        return ConfidenceLevel.MEDIUM
    return ConfidenceLevel.LOW
