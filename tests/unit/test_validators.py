"""
Tests for src/schemas/validators.py — medical code, phone, SSN, date, currency validators.
"""

import pytest
from datetime import datetime

from src.schemas.validators import (
    MedicalCodeValidator,
    ValidationInfo,
    ValidationResult,
    validate_carc_code,
    validate_cpt_code,
    validate_currency,
    validate_date,
    validate_field,
    validate_hcpcs_code,
    validate_icd10_code,
    validate_ndc_code,
    validate_npi,
    validate_phone,
    validate_rarc_code,
    validate_ssn,
    validate_taxonomy_code,
)
from src.schemas.field_types import FieldType


# ---------------------------------------------------------------------------
# ValidationInfo / ValidationResult
# ---------------------------------------------------------------------------


class TestValidationInfo:

    def test_is_valid_for_valid(self):
        info = ValidationInfo(result=ValidationResult.VALID, message="ok")
        assert info.is_valid is True

    def test_is_valid_for_warning(self):
        info = ValidationInfo(result=ValidationResult.WARNING, message="warn")
        assert info.is_valid is True

    def test_is_valid_for_invalid(self):
        info = ValidationInfo(result=ValidationResult.INVALID, message="bad")
        assert info.is_valid is False

    def test_is_valid_for_unknown(self):
        info = ValidationInfo(result=ValidationResult.UNKNOWN, message="?")
        assert info.is_valid is False

    def test_frozen(self):
        info = ValidationInfo(result=ValidationResult.VALID, message="ok")
        with pytest.raises(AttributeError):
            info.message = "new"


# ---------------------------------------------------------------------------
# CPT code validation
# ---------------------------------------------------------------------------


class TestValidateCptCode:

    def test_valid_em_code(self):
        r = validate_cpt_code("99213")
        assert r.is_valid
        assert "E&M" in r.message

    def test_valid_surgery_code(self):
        r = validate_cpt_code("27447")
        assert r.is_valid
        assert "Surgery" in r.message

    def test_valid_with_modifier(self):
        r = validate_cpt_code("99213-25")
        assert r.is_valid
        assert r.details["modifier"] == "25"

    def test_none_invalid(self):
        r = validate_cpt_code(None)
        assert r.is_valid is False

    def test_out_of_range_warning(self):
        r = validate_cpt_code("00100")
        assert r.result == ValidationResult.WARNING

    def test_invalid_format(self):
        r = validate_cpt_code("ABCDE")
        assert r.is_valid is False

    def test_integer_input(self):
        r = validate_cpt_code(99213)
        assert r.is_valid


# ---------------------------------------------------------------------------
# ICD-10 code validation
# ---------------------------------------------------------------------------


class TestValidateIcd10Code:

    def test_valid_cm_with_dot(self):
        r = validate_icd10_code("E11.9")
        assert r.is_valid
        assert "ICD-10-CM" in r.message

    def test_valid_cm_without_dot_normalizes(self):
        r = validate_icd10_code("E119")
        assert r.is_valid
        assert r.normalized_value == "E11.9"

    def test_valid_cm_short(self):
        r = validate_icd10_code("E11")
        assert r.is_valid

    def test_valid_pcs_7char(self):
        r = validate_icd10_code("0BJ08ZZ")
        assert r.is_valid
        assert "ICD-10-PCS" in r.message

    def test_none_invalid(self):
        assert validate_icd10_code(None).is_valid is False

    def test_invalid_format(self):
        assert validate_icd10_code("12345").is_valid is False

    def test_case_insensitive(self):
        r = validate_icd10_code("e11.9")
        assert r.is_valid


# ---------------------------------------------------------------------------
# HCPCS code validation
# ---------------------------------------------------------------------------


class TestValidateHcpcsCode:

    def test_valid_level2(self):
        r = validate_hcpcs_code("A4253")
        assert r.is_valid
        assert r.details["level"] == "II"

    def test_valid_with_modifier(self):
        r = validate_hcpcs_code("E0601-RR")
        assert r.is_valid
        assert r.details["has_modifier"] is True

    def test_numeric_delegates_to_cpt(self):
        """HCPCS Level I is CPT — numeric codes route to CPT validator."""
        r = validate_hcpcs_code("99213")
        assert r.is_valid
        assert "E&M" in r.message or "CPT" in r.message

    def test_none_invalid(self):
        assert validate_hcpcs_code(None).is_valid is False

    def test_empty_invalid(self):
        assert validate_hcpcs_code("").is_valid is False

    def test_invalid_letter(self):
        """Letters W-Z are not valid HCPCS Level II prefixes."""
        r = validate_hcpcs_code("Z1234")
        assert r.is_valid is False


# ---------------------------------------------------------------------------
# NDC code validation
# ---------------------------------------------------------------------------


class TestValidateNdcCode:

    def test_valid_with_hyphens(self):
        r = validate_ndc_code("0002-3227-01")
        assert r.is_valid
        assert "-" in r.normalized_value

    def test_valid_11_digits_no_hyphens(self):
        r = validate_ndc_code("00023227001")
        assert r.is_valid

    def test_valid_10_digits(self):
        r = validate_ndc_code("0002322701")
        assert r.is_valid

    def test_none_invalid(self):
        assert validate_ndc_code(None).is_valid is False

    def test_empty_invalid(self):
        assert validate_ndc_code("").is_valid is False

    def test_too_short(self):
        assert validate_ndc_code("12345").is_valid is False


# ---------------------------------------------------------------------------
# Taxonomy code validation
# ---------------------------------------------------------------------------


class TestValidateTaxonomyCode:

    def test_valid(self):
        r = validate_taxonomy_code("207Q00000X")
        assert r.is_valid
        assert r.details["version"] == "X"

    def test_none_invalid(self):
        assert validate_taxonomy_code(None).is_valid is False

    def test_wrong_length(self):
        assert validate_taxonomy_code("207Q00").is_valid is False

    def test_empty_invalid(self):
        assert validate_taxonomy_code("").is_valid is False

    def test_invalid_format(self):
        assert validate_taxonomy_code("ABCDEFGHIJ").is_valid is False


# ---------------------------------------------------------------------------
# NPI validation
# ---------------------------------------------------------------------------


class TestValidateNpi:

    def test_valid_individual(self):
        r = validate_npi("1234567893")
        assert r.is_valid
        assert r.details["entity_type"] == "Individual"

    def test_none_invalid(self):
        assert validate_npi(None).is_valid is False

    def test_wrong_length(self):
        assert validate_npi("12345").is_valid is False

    def test_bad_start_digit(self):
        """NPI must start with 1 or 2."""
        assert validate_npi("3234567893").is_valid is False

    def test_luhn_failure(self):
        assert validate_npi("1234567890").is_valid is False

    def test_integer_input(self):
        r = validate_npi(1234567893)
        assert r.is_valid

    def test_strips_non_digits(self):
        r = validate_npi("123-456-7893")
        assert r.is_valid


# ---------------------------------------------------------------------------
# Phone validation
# ---------------------------------------------------------------------------


class TestValidatePhone:

    def test_valid_formatted(self):
        r = validate_phone("(555) 123-4567")
        assert r.is_valid
        assert r.normalized_value == "555-123-4567"

    def test_valid_11_digits_country_code(self):
        r = validate_phone("15551234567")
        assert r.is_valid
        assert r.normalized_value == "555-123-4567"

    def test_none_invalid(self):
        assert validate_phone(None).is_valid is False

    def test_too_short(self):
        assert validate_phone("555123").is_valid is False


# ---------------------------------------------------------------------------
# SSN validation
# ---------------------------------------------------------------------------


class TestValidateSsn:

    def test_valid(self):
        r = validate_ssn("123-45-6789")
        assert r.is_valid
        assert r.normalized_value == "123-45-6789"

    def test_valid_no_hyphens(self):
        r = validate_ssn("123456789")
        assert r.is_valid

    def test_none_invalid(self):
        assert validate_ssn(None).is_valid is False

    def test_wrong_length(self):
        assert validate_ssn("12345").is_valid is False

    def test_area_000_invalid(self):
        assert validate_ssn("000-12-3456").is_valid is False

    def test_area_666_invalid(self):
        assert validate_ssn("666-12-3456").is_valid is False

    def test_area_9xx_invalid(self):
        assert validate_ssn("900-12-3456").is_valid is False

    def test_group_00_invalid(self):
        assert validate_ssn("123-00-6789").is_valid is False

    def test_serial_0000_invalid(self):
        assert validate_ssn("123-45-0000").is_valid is False

    def test_masked_in_details(self):
        r = validate_ssn("123-45-6789")
        assert r.details["masked"] == "XXX-XX-6789"


# ---------------------------------------------------------------------------
# CARC code validation
# ---------------------------------------------------------------------------


class TestValidateCarcCode:

    def test_known_standalone(self):
        r = validate_carc_code("45")
        assert r.is_valid
        assert "fee schedule" in r.details["description"].lower()

    def test_with_group(self):
        r = validate_carc_code("CO-45")
        assert r.is_valid
        assert r.details["group_code"] == "CO"

    def test_with_group_no_hyphen(self):
        r = validate_carc_code("PR1")
        assert r.is_valid
        assert r.details["group_code"] == "PR"
        assert r.details["adjustment_code"] == "1"

    def test_unknown_code_warning(self):
        r = validate_carc_code("999")
        assert r.result == ValidationResult.WARNING

    def test_none_invalid(self):
        assert validate_carc_code(None).is_valid is False

    def test_invalid_format(self):
        assert validate_carc_code("ZZZZZ").is_valid is False


# ---------------------------------------------------------------------------
# RARC code validation
# ---------------------------------------------------------------------------


class TestValidateRarcCode:

    def test_known_ma_code(self):
        r = validate_rarc_code("MA01")
        assert r.is_valid
        assert r.details["category"] == "Alert"

    def test_known_n_code(self):
        r = validate_rarc_code("N1")
        assert r.is_valid
        assert r.details["category"] == "Supplemental"

    def test_known_m_code(self):
        r = validate_rarc_code("M1")
        assert r.is_valid
        assert r.details["category"] == "Modified"

    def test_unknown_code_warning(self):
        r = validate_rarc_code("N999")
        assert r.result == ValidationResult.WARNING

    def test_none_invalid(self):
        assert validate_rarc_code(None).is_valid is False

    def test_invalid_format(self):
        assert validate_rarc_code("XYZ").is_valid is False


# ---------------------------------------------------------------------------
# Date validation
# ---------------------------------------------------------------------------


class TestValidateDate:

    def test_iso_format(self):
        r = validate_date("2024-01-15")
        assert r.is_valid
        assert r.normalized_value == "2024-01-15"

    def test_us_format(self):
        r = validate_date("01/15/2024")
        assert r.is_valid

    def test_datetime_object(self):
        r = validate_date(datetime(2024, 1, 15))
        assert r.is_valid

    def test_none_invalid(self):
        assert validate_date(None).is_valid is False

    def test_unparseable(self):
        assert validate_date("not-a-date").is_valid is False

    def test_min_date_violation(self):
        r = validate_date("2020-01-01", min_date=datetime(2023, 1, 1))
        assert r.is_valid is False

    def test_max_date_violation(self):
        r = validate_date("2025-12-31", max_date=datetime(2024, 12, 31))
        assert r.is_valid is False


# ---------------------------------------------------------------------------
# Currency validation
# ---------------------------------------------------------------------------


class TestValidateCurrency:

    def test_numeric_float(self):
        r = validate_currency(123.45)
        assert r.is_valid
        assert r.normalized_value == 123.45

    def test_numeric_int(self):
        r = validate_currency(100)
        assert r.is_valid
        assert r.normalized_value == 100.0

    def test_string_with_dollar(self):
        r = validate_currency("$1,234.56")
        assert r.is_valid

    def test_negative_parentheses(self):
        r = validate_currency("($500.00)")
        assert r.is_valid
        assert r.normalized_value < 0

    def test_none_invalid(self):
        assert validate_currency(None).is_valid is False

    def test_invalid_string(self):
        assert validate_currency("abc").is_valid is False


# ---------------------------------------------------------------------------
# validate_field dispatch
# ---------------------------------------------------------------------------


class TestValidateField:

    def test_required_none(self):
        r = validate_field(None, FieldType.STRING, required=True)
        assert r.is_valid is False

    def test_optional_none(self):
        r = validate_field(None, FieldType.STRING, required=False)
        assert r.is_valid is True

    def test_routes_to_cpt(self):
        r = validate_field("99213", FieldType.CPT_CODE)
        assert r.is_valid

    def test_routes_to_npi(self):
        r = validate_field("1234567893", FieldType.NPI)
        assert r.is_valid

    def test_email_valid(self):
        r = validate_field("user@example.com", FieldType.EMAIL)
        assert r.is_valid
        assert r.normalized_value == "user@example.com"

    def test_email_invalid(self):
        r = validate_field("not-an-email", FieldType.EMAIL)
        assert r.is_valid is False

    def test_zip_valid(self):
        r = validate_field("12345", FieldType.ZIP_CODE)
        assert r.is_valid

    def test_zip_plus4_valid(self):
        r = validate_field("12345-6789", FieldType.ZIP_CODE)
        assert r.is_valid

    def test_zip_invalid(self):
        r = validate_field("1234", FieldType.ZIP_CODE)
        assert r.is_valid is False

    def test_state_valid(self):
        r = validate_field("CA", FieldType.STATE)
        assert r.is_valid

    def test_state_invalid(self):
        r = validate_field("XX", FieldType.STATE)
        assert r.is_valid is False

    def test_generic_type_passes(self):
        r = validate_field("anything", FieldType.STRING)
        assert r.is_valid


# ---------------------------------------------------------------------------
# MedicalCodeValidator class
# ---------------------------------------------------------------------------


class TestMedicalCodeValidator:

    def test_validate_cpt_caches(self):
        v = MedicalCodeValidator()
        r1 = v.validate_cpt("99213")
        r2 = v.validate_cpt("99213")
        assert r1 is r2  # same object from cache

    def test_validate_icd10(self):
        v = MedicalCodeValidator()
        assert v.validate_icd10("E11.9").is_valid

    def test_validate_npi(self):
        v = MedicalCodeValidator()
        assert v.validate_npi("1234567893").is_valid

    def test_validate_carc(self):
        v = MedicalCodeValidator()
        assert v.validate_carc("CO-45").is_valid

    def test_validate_rarc(self):
        v = MedicalCodeValidator()
        assert v.validate_rarc("MA01").is_valid

    def test_validate_codes_batch(self):
        v = MedicalCodeValidator()
        results = v.validate_codes(["99213", "99214"], "cpt")
        assert len(results) == 2
        assert all(r.is_valid for r in results)

    def test_validate_codes_unknown_type(self):
        v = MedicalCodeValidator()
        with pytest.raises(ValueError, match="Unknown code type"):
            v.validate_codes(["X"], "unknown_type")

    def test_clear_cache(self):
        v = MedicalCodeValidator()
        v.validate_cpt("99213")
        assert len(v._cpt_cache) == 1
        v.clear_cache()
        assert len(v._cpt_cache) == 0
