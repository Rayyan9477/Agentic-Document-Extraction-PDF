"""
Tests for src/validation/confidence.py — confidence scoring system.
"""

import pytest

from src.validation.confidence import (
    AdaptiveConfidenceScorer,
    ConfidenceAction,
    ConfidenceLevel,
    ConfidenceScorer,
    ExtractionConfidence,
    FieldConfidence,
    calculate_confidence,
    get_confidence_level,
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TestConfidenceEnums:

    def test_confidence_level_values(self):
        assert ConfidenceLevel.HIGH == "high"
        assert ConfidenceLevel.MEDIUM == "medium"
        assert ConfidenceLevel.LOW == "low"

    def test_confidence_action_values(self):
        assert ConfidenceAction.AUTO_ACCEPT == "auto_accept"
        assert ConfidenceAction.RETRY == "retry"
        assert ConfidenceAction.HUMAN_REVIEW == "human_review"


# ---------------------------------------------------------------------------
# FieldConfidence
# ---------------------------------------------------------------------------


class TestFieldConfidence:

    def test_creation(self):
        fc = FieldConfidence(
            field_name="patient_name",
            extraction_confidence=0.95,
            validation_confidence=1.0,
            agreement_confidence=0.90,
            pattern_confidence=1.0,
            combined_confidence=0.93,
            level=ConfidenceLevel.HIGH,
        )
        assert fc.field_name == "patient_name"
        assert fc.combined_confidence == 0.93

    def test_to_dict(self):
        fc = FieldConfidence(
            field_name="npi",
            extraction_confidence=0.80,
            validation_confidence=0.70,
            agreement_confidence=0.85,
            pattern_confidence=1.0,
            combined_confidence=0.82,
            level=ConfidenceLevel.MEDIUM,
            factors=("validation_failed",),
        )
        d = fc.to_dict()
        assert d["field_name"] == "npi"
        assert d["level"] == "medium"
        assert "validation_failed" in d["factors"]

    def test_frozen(self):
        fc = FieldConfidence(
            field_name="x",
            extraction_confidence=0.5,
            validation_confidence=0.5,
            agreement_confidence=0.5,
            pattern_confidence=0.5,
            combined_confidence=0.5,
            level=ConfidenceLevel.MEDIUM,
        )
        with pytest.raises(AttributeError):
            fc.field_name = "y"


# ---------------------------------------------------------------------------
# ExtractionConfidence
# ---------------------------------------------------------------------------


class TestExtractionConfidence:

    def test_defaults(self):
        ec = ExtractionConfidence()
        assert ec.overall_confidence == 0.0
        assert ec.overall_level == ConfidenceLevel.LOW
        assert ec.recommended_action == ConfidenceAction.HUMAN_REVIEW

    def test_to_dict(self):
        ec = ExtractionConfidence()
        d = ec.to_dict()
        assert d["overall_level"] == "low"
        assert d["recommended_action"] == "human_review"


# ---------------------------------------------------------------------------
# ConfidenceScorer — field-level calculation
# ---------------------------------------------------------------------------


class TestConfidenceScorerFieldLevel:

    def test_high_confidence_all_pass(self):
        scorer = ConfidenceScorer()
        result = scorer.calculate(
            extraction_confidences={"name": 0.95, "dob": 0.92},
            agreement_scores={"name": 1.0, "dob": 0.95},
            validation_results={"name": True, "dob": True},
        )
        assert result.overall_confidence >= 0.85
        assert result.overall_level == ConfidenceLevel.HIGH

    def test_validation_failure_caps_confidence(self):
        scorer = ConfidenceScorer()
        result = scorer.calculate(
            extraction_confidences={"name": 0.99},
            agreement_scores={"name": 1.0},
            validation_results={"name": False},
        )
        fc = result.field_confidences["name"]
        assert fc.combined_confidence < 0.85
        assert fc.level != ConfidenceLevel.HIGH

    def test_pattern_flag_reduces_confidence(self):
        scorer = ConfidenceScorer()
        clean = scorer.calculate(
            extraction_confidences={"name": 0.90},
            agreement_scores={"name": 0.90},
        )
        flagged = scorer.calculate(
            extraction_confidences={"name": 0.90},
            agreement_scores={"name": 0.90},
            pattern_flags={"name"},
        )
        assert flagged.overall_confidence < clean.overall_confidence

    def test_empty_fields_returns_no_fields(self):
        scorer = ConfidenceScorer()
        result = scorer.calculate(extraction_confidences={})
        assert result.summary == "No fields to assess"
        assert result.overall_confidence == 0.0


# ---------------------------------------------------------------------------
# ConfidenceScorer — actions
# ---------------------------------------------------------------------------


class TestConfidenceScorerActions:

    def test_auto_accept_on_high(self):
        scorer = ConfidenceScorer()
        result = scorer.calculate(
            extraction_confidences={"a": 0.95},
            agreement_scores={"a": 0.95},
            validation_results={"a": True},
        )
        assert result.recommended_action == ConfidenceAction.AUTO_ACCEPT

    def test_retry_on_medium_first_attempt(self):
        scorer = ConfidenceScorer()
        result = scorer.calculate(
            extraction_confidences={"a": 0.60},
            agreement_scores={"a": 0.60},
            retry_count=0,
        )
        if result.overall_level == ConfidenceLevel.MEDIUM:
            assert result.recommended_action == ConfidenceAction.RETRY

    def test_human_review_after_max_retries(self):
        scorer = ConfidenceScorer()
        result = scorer.calculate(
            extraction_confidences={"a": 0.60},
            agreement_scores={"a": 0.60},
            retry_count=3,
        )
        assert result.recommended_action == ConfidenceAction.HUMAN_REVIEW


# ---------------------------------------------------------------------------
# ConfidenceScorer — critical fields
# ---------------------------------------------------------------------------


class TestCriticalFields:

    def test_critical_field_low_triggers_review(self):
        scorer = ConfidenceScorer(critical_fields=["npi"])
        result = scorer.calculate(
            extraction_confidences={"name": 0.95, "npi": 0.20},
            agreement_scores={"name": 0.95, "npi": 0.20},
        )
        assert result.recommended_action == ConfidenceAction.HUMAN_REVIEW
        assert result.critical_fields_status["npi"] is False

    def test_critical_field_missing(self):
        scorer = ConfidenceScorer(critical_fields=["npi"])
        result = scorer.calculate(
            extraction_confidences={"name": 0.95},
        )
        # npi not in extraction_confidences → status False
        assert result.critical_fields_status.get("npi") is False

    def test_critical_field_high_passes(self):
        scorer = ConfidenceScorer(critical_fields=["name"])
        result = scorer.calculate(
            extraction_confidences={"name": 0.95},
            agreement_scores={"name": 0.95},
            validation_results={"name": True},
        )
        assert result.critical_fields_status["name"] is True


# ---------------------------------------------------------------------------
# ConfidenceScorer — weight normalization
# ---------------------------------------------------------------------------


class TestWeightNormalization:

    def test_custom_weights_normalized(self):
        scorer = ConfidenceScorer(
            weights={"extraction": 1.0, "agreement": 1.0, "validation": 1.0, "pattern": 1.0}
        )
        total = sum(scorer.weights.values())
        assert abs(total - 1.0) < 1e-9

    def test_field_weights_affect_result(self):
        base = ConfidenceScorer()
        weighted = ConfidenceScorer(field_weights={"name": 0.5})

        base_result = base.calculate(extraction_confidences={"name": 0.90})
        weighted_result = weighted.calculate(extraction_confidences={"name": 0.90})
        # Field weight < 1.0 should lower the score
        assert weighted_result.overall_confidence < base_result.overall_confidence


# ---------------------------------------------------------------------------
# ConfidenceScorer — summary
# ---------------------------------------------------------------------------


class TestSummary:

    def test_summary_contains_percentage(self):
        scorer = ConfidenceScorer()
        result = scorer.calculate(extraction_confidences={"a": 0.90})
        assert "%" in result.summary

    def test_summary_mentions_action(self):
        scorer = ConfidenceScorer()
        result = scorer.calculate(extraction_confidences={"a": 0.90})
        assert "Action:" in result.summary


# ---------------------------------------------------------------------------
# AdaptiveConfidenceScorer
# ---------------------------------------------------------------------------


class TestAdaptiveConfidenceScorer:

    def test_cms1500_thresholds(self):
        scorer = AdaptiveConfidenceScorer(document_type="cms1500")
        assert scorer.HIGH_THRESHOLD == 0.88
        assert scorer.MEDIUM_THRESHOLD == 0.55

    def test_default_thresholds(self):
        scorer = AdaptiveConfidenceScorer(document_type="unknown_doc")
        assert scorer.HIGH_THRESHOLD == 0.85
        assert scorer.MEDIUM_THRESHOLD == 0.50

    def test_document_specific_critical_fields(self):
        scorer = AdaptiveConfidenceScorer(document_type="cms1500")
        assert "patient_name" in scorer.critical_fields

    def test_high_accuracy_adjusts_weights(self):
        scorer = AdaptiveConfidenceScorer(
            document_type="default", historical_accuracy=0.96
        )
        assert scorer.weights["extraction"] > 0.35  # boosted

    def test_low_accuracy_adjusts_weights(self):
        scorer = AdaptiveConfidenceScorer(
            document_type="default", historical_accuracy=0.70
        )
        assert scorer.weights["validation"] > 0.20  # boosted


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


class TestConvenienceFunctions:

    def test_calculate_confidence_returns_result(self):
        result = calculate_confidence(
            extraction_confidences={"a": 0.90},
            validation_results={"a": True},
        )
        assert isinstance(result, ExtractionConfidence)
        assert result.overall_confidence > 0

    def test_get_confidence_level_high(self):
        assert get_confidence_level(0.90) == ConfidenceLevel.HIGH

    def test_get_confidence_level_medium(self):
        assert get_confidence_level(0.60) == ConfidenceLevel.MEDIUM

    def test_get_confidence_level_low(self):
        assert get_confidence_level(0.30) == ConfidenceLevel.LOW

    def test_get_confidence_level_boundary(self):
        assert get_confidence_level(0.85) == ConfidenceLevel.HIGH
        assert get_confidence_level(0.50) == ConfidenceLevel.MEDIUM
        assert get_confidence_level(0.49) == ConfidenceLevel.LOW
