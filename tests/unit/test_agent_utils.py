"""
Unit tests for agent utilities.

Tests cover:
- build_custom_schema construction
- RetryConfig delay calculation
- retry_with_backoff success/failure/retry behavior
- identify_low_confidence_fields
- identify_disagreement_fields
- calculate_extraction_quality_score
"""

import time
from unittest.mock import MagicMock, patch

import pytest

from src.agents.utils import (
    RetryConfig,
    build_custom_schema,
    calculate_extraction_quality_score,
    identify_disagreement_fields,
    identify_low_confidence_fields,
    retry_with_backoff,
)


# ---------------------------------------------------------------------------
# TestBuildCustomSchema
# ---------------------------------------------------------------------------


class TestBuildCustomSchema:
    """Tests for build_custom_schema."""

    def test_minimal_schema(self) -> None:
        schema = build_custom_schema({"name": "test_schema"})
        assert schema.name == "test_schema"

    def test_schema_with_fields(self) -> None:
        schema_def = {
            "name": "invoice",
            "description": "Invoice extraction",
            "fields": [
                {"name": "total", "type": "currency", "required": True},
                {"name": "date", "type": "date", "required": False},
            ],
        }
        schema = build_custom_schema(schema_def)
        assert schema.name == "invoice"
        field_names = [f.name for f in schema.fields]
        assert "total" in field_names
        assert "date" in field_names

    def test_schema_with_invalid_field_type_defaults_to_string(self) -> None:
        schema_def = {
            "name": "test",
            "fields": [{"name": "foo", "type": "nonexistent_type"}],
        }
        schema = build_custom_schema(schema_def)
        from src.schemas import FieldType

        assert schema.fields[0].field_type == FieldType.STRING

    def test_schema_with_rules(self) -> None:
        schema_def = {
            "name": "test",
            "fields": [
                {"name": "start_date", "type": "date"},
                {"name": "end_date", "type": "date"},
            ],
            "rules": [
                {
                    "source_field": "start_date",
                    "target_field": "end_date",
                    "operator": "date_before",
                    "error_message": "Start must be before end",
                },
            ],
        }
        schema = build_custom_schema(schema_def)
        assert len(schema.cross_field_rules) == 1

    def test_schema_with_display_name(self) -> None:
        schema_def = {
            "name": "test",
            "display_name": "Test Schema Display",
        }
        schema = build_custom_schema(schema_def)
        assert schema.display_name == "Test Schema Display"

    def test_empty_fields(self) -> None:
        schema = build_custom_schema({"name": "empty", "fields": []})
        assert len(schema.fields) == 0

    def test_field_with_examples(self) -> None:
        schema_def = {
            "name": "test",
            "fields": [
                {"name": "name", "type": "string", "examples": ["Alice", "Bob"]},
            ],
        }
        schema = build_custom_schema(schema_def)
        assert schema.fields[0].examples == ["Alice", "Bob"]

    def test_field_with_pattern(self) -> None:
        schema_def = {
            "name": "test",
            "fields": [
                {"name": "ssn", "type": "string", "pattern": r"\d{3}-\d{2}-\d{4}"},
            ],
        }
        schema = build_custom_schema(schema_def)
        assert schema.fields[0].pattern is not None

    def test_field_with_allowed_values(self) -> None:
        schema_def = {
            "name": "test",
            "fields": [
                {"name": "status", "type": "string", "allowed_values": ["active", "inactive"]},
            ],
        }
        schema = build_custom_schema(schema_def)
        assert schema.fields[0].allowed_values == ["active", "inactive"]


# ---------------------------------------------------------------------------
# TestRetryConfig
# ---------------------------------------------------------------------------


class TestRetryConfig:
    """Tests for RetryConfig delay calculation."""

    def test_default_values(self) -> None:
        config = RetryConfig()
        assert config.max_retries == 3
        assert config.base_delay_ms == 1000
        assert config.max_delay_ms == 30000

    def test_custom_values(self) -> None:
        config = RetryConfig(max_retries=5, base_delay_ms=500, max_delay_ms=10000)
        assert config.max_retries == 5
        assert config.base_delay_ms == 500
        assert config.max_delay_ms == 10000

    def test_delay_increases_with_attempts(self) -> None:
        config = RetryConfig(base_delay_ms=1000, jitter=False)
        d0 = config.get_delay_ms(0)
        d1 = config.get_delay_ms(1)
        d2 = config.get_delay_ms(2)
        assert d0 < d1 < d2

    def test_delay_capped_at_max(self) -> None:
        config = RetryConfig(
            base_delay_ms=1000,
            max_delay_ms=5000,
            exponential_base=10.0,
            jitter=False,
        )
        delay = config.get_delay_ms(10)
        assert delay == 5000

    def test_jitter_adds_variance(self) -> None:
        config = RetryConfig(base_delay_ms=1000, jitter=True)
        delays = {config.get_delay_ms(0) for _ in range(20)}
        # With jitter, should get varied delays
        assert len(delays) > 1

    def test_no_jitter_deterministic(self) -> None:
        config = RetryConfig(base_delay_ms=1000, jitter=False)
        d1 = config.get_delay_ms(0)
        d2 = config.get_delay_ms(0)
        assert d1 == d2


# ---------------------------------------------------------------------------
# TestRetryWithBackoff
# ---------------------------------------------------------------------------


class TestRetryWithBackoff:
    """Tests for retry_with_backoff function."""

    def test_success_on_first_try(self) -> None:
        result = retry_with_backoff(lambda: 42)
        assert result == 42

    def test_retries_on_failure(self) -> None:
        call_count = {"n": 0}

        def flaky():
            call_count["n"] += 1
            if call_count["n"] < 3:
                raise ValueError("fail")
            return "ok"

        config = RetryConfig(max_retries=3, base_delay_ms=1, jitter=False)
        result = retry_with_backoff(flaky, config=config)
        assert result == "ok"
        assert call_count["n"] == 3

    def test_raises_after_max_retries(self) -> None:
        config = RetryConfig(max_retries=2, base_delay_ms=1, jitter=False)

        with pytest.raises(ValueError, match="always fail"):
            retry_with_backoff(
                lambda: (_ for _ in ()).throw(ValueError("always fail")),
                config=config,
            )

    def test_only_retries_recoverable_exceptions(self) -> None:
        config = RetryConfig(max_retries=3, base_delay_ms=1, jitter=False)

        with pytest.raises(TypeError):
            retry_with_backoff(
                lambda: (_ for _ in ()).throw(TypeError("not recoverable")),
                config=config,
                recoverable_exceptions=(ValueError,),
            )

    def test_on_retry_callback(self) -> None:
        call_count = {"n": 0}
        retry_calls = []

        def flaky():
            call_count["n"] += 1
            if call_count["n"] < 2:
                raise ValueError("fail")
            return "ok"

        config = RetryConfig(max_retries=3, base_delay_ms=1, jitter=False)
        retry_with_backoff(
            flaky,
            config=config,
            on_retry=lambda attempt, exc: retry_calls.append(attempt),
        )
        assert len(retry_calls) == 1
        assert retry_calls[0] == 0


# ---------------------------------------------------------------------------
# TestIdentifyLowConfidenceFields
# ---------------------------------------------------------------------------


class TestIdentifyLowConfidenceFields:
    """Tests for identify_low_confidence_fields."""

    def test_all_above_threshold(self) -> None:
        fields = {
            "name": {"value": "Alice", "confidence": 0.9},
            "date": {"value": "2024-01-01", "confidence": 0.8},
        }
        result = identify_low_confidence_fields(fields, threshold=0.7)
        assert result == []

    def test_some_below_threshold(self) -> None:
        fields = {
            "name": {"value": "Alice", "confidence": 0.9},
            "date": {"value": "2024-01-01", "confidence": 0.5},
        }
        result = identify_low_confidence_fields(fields, threshold=0.7)
        assert result == ["date"]

    def test_null_value_not_flagged(self) -> None:
        fields = {
            "name": {"value": None, "confidence": 0.3},
        }
        result = identify_low_confidence_fields(fields, threshold=0.7)
        assert result == []

    def test_empty_fields(self) -> None:
        assert identify_low_confidence_fields({}) == []

    def test_non_dict_metadata_skipped(self) -> None:
        fields = {"name": "raw_string"}
        result = identify_low_confidence_fields(fields)
        assert result == []

    def test_custom_threshold(self) -> None:
        fields = {"a": {"value": "x", "confidence": 0.8}}
        assert identify_low_confidence_fields(fields, threshold=0.9) == ["a"]
        assert identify_low_confidence_fields(fields, threshold=0.7) == []


# ---------------------------------------------------------------------------
# TestIdentifyDisagreementFields
# ---------------------------------------------------------------------------


class TestIdentifyDisagreementFields:
    """Tests for identify_disagreement_fields."""

    def test_all_agree(self) -> None:
        fields = {
            "name": {"passes_agree": True},
            "date": {"passes_agree": True},
        }
        assert identify_disagreement_fields(fields) == []

    def test_some_disagree(self) -> None:
        fields = {
            "name": {"passes_agree": True},
            "amount": {"passes_agree": False},
        }
        result = identify_disagreement_fields(fields)
        assert result == ["amount"]

    def test_empty_fields(self) -> None:
        assert identify_disagreement_fields({}) == []

    def test_missing_passes_agree_defaults_true(self) -> None:
        fields = {"name": {"value": "x"}}
        assert identify_disagreement_fields(fields) == []


# ---------------------------------------------------------------------------
# TestCalculateExtractionQualityScore
# ---------------------------------------------------------------------------


class TestCalculateExtractionQualityScore:
    """Tests for calculate_extraction_quality_score."""

    def test_perfect_score(self) -> None:
        fields = {
            "name": {"value": "Alice", "confidence": 1.0},
            "date": {"value": "2024-01-01", "confidence": 1.0},
        }
        score = calculate_extraction_quality_score(fields, [], [])
        assert score == 1.0

    def test_hallucination_penalty(self) -> None:
        fields = {"name": {"value": "Alice", "confidence": 0.9}}
        score_clean = calculate_extraction_quality_score(fields, [], [])
        score_hallucinated = calculate_extraction_quality_score(
            fields, ["name"], [],
        )
        assert score_hallucinated < score_clean

    def test_validation_error_penalty(self) -> None:
        fields = {"name": {"value": "Alice", "confidence": 0.9}}
        score_clean = calculate_extraction_quality_score(fields, [], [])
        score_errors = calculate_extraction_quality_score(
            fields, [], ["Invalid format"],
        )
        assert score_errors < score_clean

    def test_empty_fields_returns_zero(self) -> None:
        assert calculate_extraction_quality_score({}, [], []) == 0.0

    def test_score_clamped_to_zero(self) -> None:
        fields = {"a": {"value": "x", "confidence": 0.1}}
        # Many penalties should clamp to 0
        score = calculate_extraction_quality_score(
            fields, ["a", "b", "c", "d"], ["e1", "e2", "e3", "e4"],
        )
        assert score == 0.0

    def test_score_clamped_to_one(self) -> None:
        fields = {"a": {"value": "x", "confidence": 1.0}}
        score = calculate_extraction_quality_score(fields, [], [])
        assert score <= 1.0

    def test_null_values_excluded_from_average(self) -> None:
        fields = {
            "name": {"value": "Alice", "confidence": 0.8},
            "empty": {"value": None, "confidence": 0.2},
        }
        score = calculate_extraction_quality_score(fields, [], [])
        # Only "name" is counted (confidence 0.8), not "empty"
        assert score == pytest.approx(0.8, abs=0.01)
