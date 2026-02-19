"""
Tests for src/validation/pattern_detector.py — hallucination pattern detection.
"""

import pytest

from src.validation.pattern_detector import (
    HallucinationPattern,
    HallucinationPatternDetector,
    PatternDetectionResult,
    PatternMatch,
    PatternSeverity,
    detect_hallucination_patterns,
)


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class TestEnums:

    def test_pattern_types(self):
        assert HallucinationPattern.PLACEHOLDER_TEXT == "placeholder_text"
        assert HallucinationPattern.GENERIC_NAME == "generic_name"
        assert HallucinationPattern.SPATIAL_ANOMALY == "spatial_anomaly"

    def test_severity_levels(self):
        assert PatternSeverity.LOW == "low"
        assert PatternSeverity.CRITICAL == "critical"


# ---------------------------------------------------------------------------
# PatternMatch
# ---------------------------------------------------------------------------


class TestPatternMatch:

    def test_to_dict(self):
        pm = PatternMatch(
            field_name="patient_name",
            value="John Doe",
            pattern=HallucinationPattern.GENERIC_NAME,
            severity=PatternSeverity.CRITICAL,
            confidence=0.92,
            description="Generic name detected",
        )
        d = pm.to_dict()
        assert d["pattern"] == "generic_name"
        assert d["severity"] == "critical"

    def test_frozen(self):
        pm = PatternMatch(
            field_name="x",
            value="y",
            pattern=HallucinationPattern.TEST_DATA,
            severity=PatternSeverity.LOW,
            confidence=0.5,
            description="test",
        )
        with pytest.raises(AttributeError):
            pm.field_name = "z"


# ---------------------------------------------------------------------------
# PatternDetectionResult
# ---------------------------------------------------------------------------


class TestPatternDetectionResult:

    def test_defaults(self):
        r = PatternDetectionResult()
        assert r.overall_suspicion_score == 0.0
        assert r.is_likely_hallucination is False
        assert r.matches == []

    def test_to_dict(self):
        r = PatternDetectionResult()
        d = r.to_dict()
        assert "flagged_fields" in d
        assert "critical_patterns" in d


# ---------------------------------------------------------------------------
# Placeholder text detection
# ---------------------------------------------------------------------------


class TestPlaceholderDetection:

    @pytest.fixture()
    def detector(self):
        return HallucinationPatternDetector()

    @pytest.mark.parametrize("value", ["N/A", "TBD", "xxx", "placeholder", "unknown", "null"])
    def test_placeholder_values_detected(self, detector, value):
        result = detector.detect({f"field_{value}": value})
        patterns = [m.pattern for m in result.matches]
        assert HallucinationPattern.PLACEHOLDER_TEXT in patterns

    def test_bracket_placeholder(self, detector):
        result = detector.detect({"name": "[Enter Name]"})
        patterns = [m.pattern for m in result.matches]
        assert HallucinationPattern.PLACEHOLDER_TEXT in patterns

    def test_normal_value_not_flagged(self, detector):
        result = detector.detect({"name": "John Smith"})
        placeholder_matches = [
            m for m in result.matches
            if m.pattern == HallucinationPattern.PLACEHOLDER_TEXT
        ]
        assert len(placeholder_matches) == 0


# ---------------------------------------------------------------------------
# Test data detection
# ---------------------------------------------------------------------------


class TestTestDataDetection:

    @pytest.fixture()
    def detector(self):
        return HallucinationPatternDetector()

    @pytest.mark.parametrize("value", ["test", "sample", "demo", "example", "fake"])
    def test_test_data_detected(self, detector, value):
        result = detector.detect({"field": value})
        patterns = [m.pattern for m in result.matches]
        assert HallucinationPattern.TEST_DATA in patterns


# ---------------------------------------------------------------------------
# Generic name detection
# ---------------------------------------------------------------------------


class TestGenericNameDetection:

    @pytest.fixture()
    def detector(self):
        return HallucinationPatternDetector()

    def test_john_doe(self, detector):
        result = detector.detect({"patient_name": "John Doe"})
        patterns = [m.pattern for m in result.matches]
        assert HallucinationPattern.GENERIC_NAME in patterns

    def test_jane_smith(self, detector):
        result = detector.detect({"patient_name": "Jane Smith"})
        patterns = [m.pattern for m in result.matches]
        assert HallucinationPattern.GENERIC_NAME in patterns

    def test_real_name_not_flagged(self, detector):
        result = detector.detect({"patient_name": "Robert Johnson"})
        generic_matches = [
            m for m in result.matches
            if m.pattern == HallucinationPattern.GENERIC_NAME
        ]
        assert len(generic_matches) == 0


# ---------------------------------------------------------------------------
# Generic address detection
# ---------------------------------------------------------------------------


class TestGenericAddressDetection:

    @pytest.fixture()
    def detector(self):
        return HallucinationPatternDetector()

    def test_123_main_st(self, detector):
        result = detector.detect({"address": "123 Main St, Anytown"})
        patterns = [m.pattern for m in result.matches]
        assert HallucinationPattern.GENERIC_ADDRESS in patterns


# ---------------------------------------------------------------------------
# Repetitive value detection (cross-field)
# ---------------------------------------------------------------------------


class TestRepetitiveValues:

    @pytest.fixture()
    def detector(self):
        return HallucinationPatternDetector()

    def test_same_value_in_3_fields(self, detector):
        data = {"field_a": "duplicate", "field_b": "duplicate", "field_c": "duplicate"}
        result = detector.detect(data)
        patterns = [m.pattern for m in result.matches]
        assert HallucinationPattern.REPETITIVE_VALUE in patterns

    def test_short_repeated_value_not_flagged(self, detector):
        data = {"a": "x", "b": "x", "c": "x"}
        result = detector.detect(data)
        # len("x") <= 2, should not trigger repetitive
        rep_matches = [
            m for m in result.matches if m.pattern == HallucinationPattern.REPETITIVE_VALUE
        ]
        assert len(rep_matches) == 0


# ---------------------------------------------------------------------------
# Repeated digits detection
# ---------------------------------------------------------------------------


class TestRepeatedDigits:

    @pytest.fixture()
    def detector(self):
        return HallucinationPatternDetector()

    def test_all_zeros(self, detector):
        result = detector.detect({"claim_id": "00000"})
        patterns = [m.pattern for m in result.matches]
        assert HallucinationPattern.REPEATED_DIGITS in patterns or \
               HallucinationPattern.SYNTHETIC_IDENTIFIER in patterns

    def test_all_nines(self, detector):
        result = detector.detect({"member_id": "99999"})
        patterns = [m.pattern for m in result.matches]
        assert HallucinationPattern.REPEATED_DIGITS in patterns


# ---------------------------------------------------------------------------
# Synthetic identifier detection
# ---------------------------------------------------------------------------


class TestSyntheticIdentifier:

    @pytest.fixture()
    def detector(self):
        return HallucinationPatternDetector()

    def test_all_zeros_identifier(self, detector):
        result = detector.detect({"claim_number": "000000000"})
        patterns = [m.pattern for m in result.matches]
        assert HallucinationPattern.SYNTHETIC_IDENTIFIER in patterns

    def test_sequential_identifier(self, detector):
        result = detector.detect({"policy_number": "123456789"})
        patterns = [m.pattern for m in result.matches]
        assert HallucinationPattern.SYNTHETIC_IDENTIFIER in patterns


# ---------------------------------------------------------------------------
# Date pattern detection
# ---------------------------------------------------------------------------


class TestDatePatterns:

    @pytest.fixture()
    def detector(self):
        return HallucinationPatternDetector()

    def test_very_old_date(self, detector):
        result = detector.detect({"service_date": "1800-01-01"})
        date_matches = [
            m for m in result.matches if m.pattern == HallucinationPattern.IMPLAUSIBLE_DATE
        ]
        assert len(date_matches) > 0

    def test_placeholder_date_1900(self, detector):
        result = detector.detect({"admission_date": "01/01/1900"})
        date_matches = [
            m for m in result.matches if m.pattern == HallucinationPattern.IMPLAUSIBLE_DATE
        ]
        assert len(date_matches) > 0

    def test_plausible_date_not_flagged(self, detector):
        result = detector.detect({"service_date": "2024-03-15"})
        date_matches = [
            m for m in result.matches if m.pattern == HallucinationPattern.IMPLAUSIBLE_DATE
        ]
        assert len(date_matches) == 0


# ---------------------------------------------------------------------------
# Truncation detection
# ---------------------------------------------------------------------------


class TestTruncation:

    @pytest.fixture()
    def detector(self):
        return HallucinationPatternDetector()

    def test_ellipsis_truncation(self, detector):
        result = detector.detect({"description": "Long text that was cut off..."})
        patterns = [m.pattern for m in result.matches]
        assert HallucinationPattern.TRUNCATED_VALUE in patterns

    def test_unicode_ellipsis(self, detector):
        result = detector.detect({"notes": "Some text\u2026"})
        patterns = [m.pattern for m in result.matches]
        assert HallucinationPattern.TRUNCATED_VALUE in patterns


# ---------------------------------------------------------------------------
# Spatial anomaly detection
# ---------------------------------------------------------------------------


class TestSpatialAnomalies:

    @pytest.fixture()
    def detector(self):
        return HallucinationPatternDetector()

    def test_zero_area_bbox(self, detector):
        data = {"name": {"bbox": {"x": 0.1, "y": 0.2, "w": 0, "h": 0.05}}}
        result = detector.detect(data)
        spatial = [m for m in result.matches if m.pattern == HallucinationPattern.SPATIAL_ANOMALY]
        assert len(spatial) > 0

    def test_oversized_bbox(self, detector):
        data = {"name": {"bbox": {"x": 0.0, "y": 0.0, "w": 0.9, "h": 0.8}}}
        result = detector.detect(data)
        spatial = [m for m in result.matches if m.pattern == HallucinationPattern.SPATIAL_ANOMALY]
        assert len(spatial) > 0

    def test_out_of_bounds_bbox(self, detector):
        data = {"name": {"bbox": {"x": -0.1, "y": 0.0, "w": 0.5, "h": 0.5}}}
        result = detector.detect(data)
        spatial = [m for m in result.matches if m.pattern == HallucinationPattern.SPATIAL_ANOMALY]
        assert len(spatial) > 0

    def test_normal_bbox_not_flagged(self, detector):
        data = {"name": {"bbox": {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.05}}}
        result = detector.detect(data)
        spatial = [m for m in result.matches if m.pattern == HallucinationPattern.SPATIAL_ANOMALY]
        assert len(spatial) == 0


# ---------------------------------------------------------------------------
# Numeric patterns
# ---------------------------------------------------------------------------


class TestNumericPatterns:

    @pytest.fixture()
    def detector(self):
        return HallucinationPatternDetector()

    def test_round_currency(self, detector):
        result = detector.detect({"total_charges": 1000})
        round_matches = [
            m for m in result.matches if m.pattern == HallucinationPattern.ROUND_NUMBER
        ]
        assert len(round_matches) > 0

    def test_negative_currency(self, detector):
        result = detector.detect({"payment_amount": -500})
        impossible = [
            m for m in result.matches if m.pattern == HallucinationPattern.IMPOSSIBLE_VALUE
        ]
        assert len(impossible) > 0


# ---------------------------------------------------------------------------
# Overall scoring
# ---------------------------------------------------------------------------


class TestOverallScoring:

    @pytest.fixture()
    def detector(self):
        return HallucinationPatternDetector()

    def test_critical_pattern_marks_likely_hallucination(self, detector):
        result = detector.detect({"patient_name": "John Doe"})
        assert result.is_likely_hallucination is True
        assert len(result.critical_patterns) > 0

    def test_no_patterns_clean(self, detector):
        result = detector.detect({"total": 453.21})
        assert result.is_likely_hallucination is False
        assert result.overall_suspicion_score == 0.0

    def test_summary_present(self, detector):
        result = detector.detect({"patient_name": "N/A"})
        assert "pattern" in result.summary.lower() or "detected" in result.summary.lower()

    def test_none_values_skipped(self, detector):
        result = detector.detect({"a": None, "b": None})
        assert len(result.matches) == 0


# ---------------------------------------------------------------------------
# Custom placeholders
# ---------------------------------------------------------------------------


class TestCustomPlaceholders:

    def test_custom_pattern_detected(self):
        detector = HallucinationPatternDetector(
            custom_placeholder_patterns=[r"^REDACTED$"]
        )
        result = detector.detect({"field": "REDACTED"})
        patterns = [m.pattern for m in result.matches]
        assert HallucinationPattern.PLACEHOLDER_TEXT in patterns


# ---------------------------------------------------------------------------
# Cache clearing
# ---------------------------------------------------------------------------


class TestCacheClearing:

    def test_clear_caches_no_error(self):
        HallucinationPatternDetector.clear_caches()


# ---------------------------------------------------------------------------
# Convenience function
# ---------------------------------------------------------------------------


class TestConvenienceFunction:

    def test_detect_hallucination_patterns(self):
        result = detect_hallucination_patterns(
            {"patient_name": "John Doe", "claim_id": "000000000"}
        )
        assert isinstance(result, PatternDetectionResult)
        assert len(result.matches) > 0
        assert result.is_likely_hallucination is True
