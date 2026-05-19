"""Tests for the Phase K medical-code tool registry (gemma_tools.py).

Each test validates one Python tool implementation in isolation and the
public ``dispatch_tool_call`` entrypoint. The actual Gemma 4 model is not
required — the tools are stateless functions over Python dicts.
"""

from __future__ import annotations

import pytest

from src.client.backends.gemma_tools import (
    TOOL_DISPATCH,
    VERIDOC_TOOLS,
    dispatch_tool_call,
)


class TestRegistryShape:
    """The registry exposes exactly the five tools the plan promised."""

    def test_five_tools_registered(self) -> None:
        assert len(VERIDOC_TOOLS) == 5
        assert len(TOOL_DISPATCH) == 5

    def test_tool_names_match_between_schema_and_dispatch(self) -> None:
        schema_names = {entry["function"]["name"] for entry in VERIDOC_TOOLS}
        dispatch_names = set(TOOL_DISPATCH.keys())
        assert schema_names == dispatch_names

    def test_all_tools_have_required_parameters_field(self) -> None:
        for entry in VERIDOC_TOOLS:
            fn = entry["function"]
            assert "name" in fn
            assert "description" in fn
            assert "parameters" in fn
            assert fn["parameters"]["type"] == "object"
            assert "required" in fn["parameters"]


class TestNPILuhnCheck:
    """``npi_luhn_check`` validates 10-digit NPI numbers via Luhn."""

    def test_valid_individual_npi(self) -> None:
        # 1234567893 is canonically valid (commonly used as CMS example).
        result = TOOL_DISPATCH["npi_luhn_check"](npi="1234567893")
        assert result["valid"] is True
        assert result["entity_type"] == "individual"
        assert result["normalised"] == "1234567893"

    def test_valid_organisation_npi(self) -> None:
        # Find a valid 10-digit NPI starting with '2' (organisation) that
        # passes the same bare Luhn check ``validate_npi`` uses.
        from src.schemas.validators import _luhn_checksum

        for tail in range(0, 10):
            candidate = f"234567890{tail}"
            if _luhn_checksum(candidate):
                result = TOOL_DISPATCH["npi_luhn_check"](npi=candidate)
                assert result["valid"] is True
                assert result["entity_type"] == "organisation"
                return
        pytest.fail("Could not synthesise a valid organisation NPI for the test")

    def test_invalid_npi_fails_luhn(self) -> None:
        # Flip the check digit of a known-valid NPI.
        result = TOOL_DISPATCH["npi_luhn_check"](npi="1234567890")
        assert result["valid"] is False
        assert result["reason"] == "luhn_fail"

    def test_wrong_length_rejected(self) -> None:
        result = TOOL_DISPATCH["npi_luhn_check"](npi="12345")
        assert result["valid"] is False
        assert result["reason"] == "not_10_digits"

    def test_non_digit_rejected(self) -> None:
        result = TOOL_DISPATCH["npi_luhn_check"](npi="abc1234567")
        assert result["valid"] is False


class TestCPTValidate:
    """CPT format validation — Category I (numeric) + II/III (alphanumeric)."""

    def test_category_i_numeric(self) -> None:
        result = TOOL_DISPATCH["cpt_validate"](cpt_code="99213")
        assert result["valid"] is True
        assert result["category"] == "I"

    def test_category_ii_f_suffix(self) -> None:
        result = TOOL_DISPATCH["cpt_validate"](cpt_code="0001F")
        assert result["valid"] is True
        assert result["category"] == "II"

    def test_category_iii_t_suffix(self) -> None:
        result = TOOL_DISPATCH["cpt_validate"](cpt_code="0050T")
        assert result["valid"] is True
        assert result["category"] == "III"

    def test_wrong_length(self) -> None:
        result = TOOL_DISPATCH["cpt_validate"](cpt_code="9921")
        assert result["valid"] is False
        assert result["reason"] == "wrong_length"

    def test_invalid_alpha_suffix(self) -> None:
        result = TOOL_DISPATCH["cpt_validate"](cpt_code="0001X")
        assert result["valid"] is False


class TestICDNormalize:
    """ICD-10-CM normalisation: 'J069' → 'J06.9'; validates structure."""

    def test_adds_period(self) -> None:
        result = TOOL_DISPATCH["icd_normalize"](raw_code="J069")
        assert result["valid"] is True
        assert result["normalised"] == "J06.9"

    def test_preserves_existing_period(self) -> None:
        result = TOOL_DISPATCH["icd_normalize"](raw_code="E11.65")
        assert result["valid"] is True
        assert result["normalised"] == "E11.65"

    def test_lowercase_normalised(self) -> None:
        result = TOOL_DISPATCH["icd_normalize"](raw_code="j069")
        assert result["valid"] is True
        assert result["normalised"] == "J06.9"

    def test_three_char_minimum(self) -> None:
        result = TOOL_DISPATCH["icd_normalize"](raw_code="J0")
        assert result["valid"] is False
        assert result["reason"] == "too_short"

    def test_must_start_with_letter(self) -> None:
        result = TOOL_DISPATCH["icd_normalize"](raw_code="123")
        assert result["valid"] is False
        assert result["reason"] == "must_start_with_letter"


class TestSumReconcile:
    """Line-item amount sum vs. reported total reconciliation."""

    def test_exact_match(self) -> None:
        result = TOOL_DISPATCH["sum_reconcile"](
            line_amounts=[100.00, 50.00, 25.00],
            reported_total=175.00,
        )
        assert result["match"] is True
        assert result["computed_total"] == 175.00
        assert result["delta"] == 0.0

    def test_within_tolerance(self) -> None:
        result = TOOL_DISPATCH["sum_reconcile"](
            line_amounts=[100.00, 50.00],
            reported_total=150.01,
            tolerance_cents=2,
        )
        assert result["match"] is True

    def test_outside_tolerance(self) -> None:
        result = TOOL_DISPATCH["sum_reconcile"](
            line_amounts=[100.00, 50.00],
            reported_total=152.00,
            tolerance_cents=1,
        )
        assert result["match"] is False
        assert result["delta_cents"] > 1

    def test_empty_line_amounts_rejected(self) -> None:
        result = TOOL_DISPATCH["sum_reconcile"](line_amounts=[], reported_total=0.0)
        assert result["match"] is False
        assert result["reason"] == "no_line_amounts"


class TestValidateDateOrdering:
    """Medical-document date invariants."""

    def test_happy_path(self) -> None:
        result = TOOL_DISPATCH["validate_date_ordering"](
            date_of_birth="1980-01-15",
            service_dates=["2024-06-01", "2024-06-15"],
            admission_date="2024-05-30",
            discharge_date="2024-06-30",
        )
        assert result["valid"] is True

    def test_service_before_dob_flagged(self) -> None:
        result = TOOL_DISPATCH["validate_date_ordering"](
            date_of_birth="1980-01-15",
            service_dates=["1975-06-01"],
        )
        assert result["valid"] is False
        rules = [v["rule"] for v in result["violations"]]
        assert "service_before_dob" in rules

    def test_discharge_before_admission_flagged(self) -> None:
        result = TOOL_DISPATCH["validate_date_ordering"](
            date_of_birth="1980-01-15",
            service_dates=["2024-06-15"],
            admission_date="2024-06-30",
            discharge_date="2024-06-01",
        )
        assert result["valid"] is False
        rules = [v["rule"] for v in result["violations"]]
        assert "discharge_before_admission" in rules

    def test_service_outside_inpatient_window_flagged(self) -> None:
        result = TOOL_DISPATCH["validate_date_ordering"](
            date_of_birth="1980-01-15",
            admission_date="2024-06-01",
            discharge_date="2024-06-15",
            service_dates=["2024-07-01"],
        )
        assert result["valid"] is False
        rules = [v["rule"] for v in result["violations"]]
        assert "service_after_discharge" in rules

    def test_invalid_dob_short_circuits(self) -> None:
        result = TOOL_DISPATCH["validate_date_ordering"](
            date_of_birth="not-a-date",
            service_dates=["2024-06-01"],
        )
        assert result["valid"] is False
        assert result["reason"] == "invalid_date_of_birth"


class TestDispatchEntrypoint:
    """``dispatch_tool_call`` is the public surface the orchestrator calls."""

    def test_known_tool_executes(self) -> None:
        result = dispatch_tool_call("cpt_validate", {"cpt_code": "99213"})
        assert result["valid"] is True

    def test_unknown_tool_returns_structured_error(self) -> None:
        result = dispatch_tool_call("hallucinated_tool", {"x": 1})
        assert result["valid"] is False
        assert result["reason"] == "unknown_tool"
        assert result["requested_tool"] == "hallucinated_tool"
        assert "cpt_validate" in result["available_tools"]

    def test_bad_arguments_caught(self) -> None:
        result = dispatch_tool_call("cpt_validate", {"wrong_kwarg": "99213"})
        assert result["valid"] is False
        assert result["reason"] == "bad_arguments"

    def test_non_dict_arguments_rejected(self) -> None:
        result = dispatch_tool_call("cpt_validate", "not a dict")  # type: ignore[arg-type]
        assert result["valid"] is False
        assert result["reason"] == "arguments_must_be_dict"
