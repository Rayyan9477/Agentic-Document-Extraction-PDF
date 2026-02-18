"""
Unit tests for ValidatorAgent.

Tests cover:
- Initialization (thresholds, calibrator, VLM verification)
- process() state transition
- _validate_extraction comprehensive flow
- _check_hallucination_patterns (placeholders, round amounts, dates)
- _validate_medical_codes (CPT, ICD-10, NPI)
- _check_repetitive_values
- _route_based_on_confidence
- validate_field_standalone
- Calibration integration
"""

from unittest.mock import MagicMock, patch

import pytest

from src.agents.base import ValidationError as AgentValidationError
from src.agents.validator import ValidatorAgent
from src.client.lm_client import VisionResponse
from src.pipeline.state import (
    ConfidenceLevel,
    ExtractionStatus,
    ValidationResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(**overrides) -> dict:
    base = {
        "processing_id": "test-proc",
        "pdf_path": "/tmp/test.pdf",
        "status": "extracting",
        "current_step": "extraction_complete",
        "page_images": [],
        "document_type": "CMS-1500",
        "selected_schema_name": "cms_1500",
        "overall_confidence": 0.0,
        "confidence_level": "low",
        "retry_count": 0,
        "errors": [],
        "warnings": [],
        "merged_extraction": {
            "patient_name": {"value": "Alice Smith", "confidence": 0.9},
            "date_of_service": {"value": "01/15/2024", "confidence": 0.85},
        },
        "field_metadata": {
            "patient_name": {"value": "Alice Smith", "confidence": 0.9},
            "date_of_service": {"value": "01/15/2024", "confidence": 0.85},
        },
        "validation": {},
        "total_vlm_calls": 4,
    }
    base.update(overrides)
    return base


# ---------------------------------------------------------------------------
# TestValidatorInit
# ---------------------------------------------------------------------------


class TestValidatorInit:
    """Tests for ValidatorAgent initialization."""

    def test_default_init(self) -> None:
        agent = ValidatorAgent(client=MagicMock())
        assert agent.name == "validator"
        assert agent._high_threshold == 0.85
        assert agent._low_threshold == 0.50

    def test_custom_thresholds(self) -> None:
        agent = ValidatorAgent(
            client=MagicMock(),
            high_confidence_threshold=0.90,
            low_confidence_threshold=0.40,
        )
        assert agent._high_threshold == 0.90
        assert agent._low_threshold == 0.40

    def test_calibrator_injection(self) -> None:
        calibrator = MagicMock()
        agent = ValidatorAgent(client=MagicMock(), calibrator=calibrator)
        assert agent._calibrator is calibrator

    def test_calibrator_default_none(self) -> None:
        agent = ValidatorAgent(client=MagicMock())
        assert agent._calibrator is None

    def test_vlm_verification_default_false(self) -> None:
        agent = ValidatorAgent(client=MagicMock())
        assert agent._enable_vlm_verification is False


# ---------------------------------------------------------------------------
# TestProcess
# ---------------------------------------------------------------------------


class TestProcess:
    """Tests for process() method."""

    def test_process_no_extraction_raises(self) -> None:
        agent = ValidatorAgent(client=MagicMock())
        state = _make_state(merged_extraction={})
        with pytest.raises(AgentValidationError, match="No extraction results"):
            agent.process(state)

    def test_process_updates_state(self) -> None:
        agent = ValidatorAgent(client=MagicMock())
        state = _make_state()
        result = agent.process(state)
        assert "validation" in result
        assert "overall_confidence" in result
        assert "confidence_level" in result

    def test_process_sets_validation_complete(self) -> None:
        agent = ValidatorAgent(client=MagicMock())
        state = _make_state()
        result = agent.process(state)
        assert result["current_step"] == "validation_complete"


# ---------------------------------------------------------------------------
# TestCheckHallucinationPatterns
# ---------------------------------------------------------------------------


class TestCheckHallucinationPatterns:
    """Tests for _check_hallucination_patterns."""

    def setup_method(self) -> None:
        self.agent = ValidatorAgent(client=MagicMock())

    def test_placeholder_na(self) -> None:
        result = self.agent._check_hallucination_patterns("name", "N/A", 0.9)
        assert result is not None
        assert "Placeholder" in result

    def test_placeholder_tbd(self) -> None:
        result = self.agent._check_hallucination_patterns("name", "TBD", 0.9)
        assert result is not None

    def test_placeholder_john_doe(self) -> None:
        result = self.agent._check_hallucination_patterns("name", "John Doe", 0.9)
        assert result is not None

    def test_valid_value_no_flag(self) -> None:
        result = self.agent._check_hallucination_patterns("name", "Alice Smith", 0.9)
        assert result is None

    def test_suspicious_round_amount(self) -> None:
        result = self.agent._check_hallucination_patterns(
            "total_amount", "$1000.00", 0.95,
        )
        assert result is not None
        assert "round amount" in result.lower()

    def test_normal_amount_not_flagged(self) -> None:
        result = self.agent._check_hallucination_patterns(
            "total_amount", "$127.50", 0.9,
        )
        assert result is None

    def test_suspicious_date(self) -> None:
        result = self.agent._check_hallucination_patterns(
            "date_of_service", "01/01/2000", 0.9,
        )
        assert result is not None
        assert "Suspicious date" in result

    def test_normal_date_not_flagged(self) -> None:
        result = self.agent._check_hallucination_patterns(
            "date_of_service", "03/15/2024", 0.9,
        )
        assert result is None

    def test_none_value_returns_none(self) -> None:
        result = self.agent._check_hallucination_patterns("name", None, 0.9)
        assert result is None


# ---------------------------------------------------------------------------
# TestValidateMedicalCodes
# ---------------------------------------------------------------------------


class TestValidateMedicalCodes:
    """Tests for _validate_medical_codes."""

    def setup_method(self) -> None:
        self.agent = ValidatorAgent(client=MagicMock())

    def test_valid_cpt_code(self) -> None:
        errors = self.agent._validate_medical_codes("cpt_code", "99213")
        assert len(errors) == 0

    def test_invalid_cpt_code(self) -> None:
        errors = self.agent._validate_medical_codes("cpt_code", "XXXXX")
        assert len(errors) > 0

    def test_valid_npi(self) -> None:
        # 1234567893 is a valid NPI (passes Luhn)
        errors = self.agent._validate_medical_codes("npi", "1234567893")
        assert len(errors) == 0

    def test_invalid_npi(self) -> None:
        errors = self.agent._validate_medical_codes("npi", "0000000000")
        assert len(errors) > 0

    def test_none_value(self) -> None:
        errors = self.agent._validate_medical_codes("cpt_code", None)
        assert errors == []

    def test_non_medical_field_no_validation(self) -> None:
        errors = self.agent._validate_medical_codes("patient_name", "Alice")
        assert errors == []


# ---------------------------------------------------------------------------
# TestCheckRepetitiveValues
# ---------------------------------------------------------------------------


class TestCheckRepetitiveValues:
    """Tests for _check_repetitive_values."""

    def setup_method(self) -> None:
        self.agent = ValidatorAgent(client=MagicMock())

    def test_no_repetition(self) -> None:
        extraction = {
            "name": {"value": "Alice"},
            "city": {"value": "New York"},
            "state": {"value": "NY"},
        }
        warnings = self.agent._check_repetitive_values(extraction)
        assert len(warnings) == 0

    def test_repetitive_values_flagged(self) -> None:
        extraction = {
            "field1": {"value": "SAME VALUE HERE"},
            "field2": {"value": "SAME VALUE HERE"},
            "field3": {"value": "SAME VALUE HERE"},
        }
        warnings = self.agent._check_repetitive_values(extraction)
        assert len(warnings) > 0
        assert "Repetitive" in warnings[0]

    def test_short_values_not_flagged(self) -> None:
        extraction = {
            "a": {"value": "NY"},
            "b": {"value": "NY"},
            "c": {"value": "NY"},
        }
        warnings = self.agent._check_repetitive_values(extraction)
        assert len(warnings) == 0  # "NY" is < 3 chars


# ---------------------------------------------------------------------------
# TestCalibrationIntegration
# ---------------------------------------------------------------------------


class TestCalibrationIntegration:
    """Tests for calibration integration."""

    def test_calibrator_applied(self) -> None:
        calibrator = MagicMock()
        cal_result = MagicMock()
        cal_result.calibrated_confidence = 0.75
        calibrator.calibrate.return_value = cal_result

        agent = ValidatorAgent(client=MagicMock(), calibrator=calibrator)
        state = _make_state()
        result = agent.process(state)

        # Calibrator should have been called
        calibrator.calibrate.assert_called_once()

    def test_calibrator_failure_graceful(self) -> None:
        calibrator = MagicMock()
        calibrator.calibrate.side_effect = Exception("calibration error")

        agent = ValidatorAgent(client=MagicMock(), calibrator=calibrator)
        state = _make_state()
        # Should not raise — calibration failure is logged and skipped
        result = agent.process(state)
        assert "overall_confidence" in result


# ---------------------------------------------------------------------------
# TestValidateFieldStandalone
# ---------------------------------------------------------------------------


class TestValidateFieldStandalone:
    """Tests for validate_field_standalone."""

    def test_valid_field(self) -> None:
        agent = ValidatorAgent(client=MagicMock())
        result = agent.validate_field_standalone("patient_name", "Alice Smith")
        assert result.success is True
        assert result.data["valid"] is True

    def test_placeholder_field(self) -> None:
        agent = ValidatorAgent(client=MagicMock())
        result = agent.validate_field_standalone("patient_name", "N/A")
        assert result.success is True
        assert len(result.data["warnings"]) > 0

    def test_invalid_cpt(self) -> None:
        agent = ValidatorAgent(client=MagicMock())
        result = agent.validate_field_standalone("cpt_code", "XXXXX")
        assert result.success is True
        assert len(result.data["errors"]) > 0


# ---------------------------------------------------------------------------
# TestRouteBasedOnConfidence
# ---------------------------------------------------------------------------


class TestRouteBasedOnConfidence:
    """Tests for _route_based_on_confidence."""

    def test_high_confidence_no_flags(self) -> None:
        agent = ValidatorAgent(client=MagicMock())
        state = _make_state()
        validation = ValidationResult(
            overall_confidence=0.95,
            confidence_level=ConfidenceLevel.HIGH,
        )
        result = agent._route_based_on_confidence(state, validation)
        assert result["current_step"] == "validation_complete"

    def test_low_confidence_sets_reasons(self) -> None:
        agent = ValidatorAgent(client=MagicMock())
        state = _make_state()
        validation = ValidationResult(
            overall_confidence=0.30,
            confidence_level=ConfidenceLevel.LOW,
            errors=["invalid field"],
        )
        result = agent._route_based_on_confidence(state, validation)
        assert "validation_reasons" in result
        assert len(result["validation_reasons"]) > 0
