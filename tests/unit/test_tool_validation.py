"""Phase K — post-extraction tool-validation tests.

Verifies that ``validate_record`` flags real-world Gemma 4 extraction
errors observed during live validation:

* Faxed CMS-1500 produced NPI ``1234567890`` — Luhn check digit dropped.
  ``validate_record`` flags this as a ``luhn_fail`` violation.
* NPI concatenated into a longer string (``"Springfield Family Medicine
  — NPI 1234567893"``) is still validated by extracting the 10-digit run.
* Cross-field sum reconciliation flags line / total mismatches.
* DOB-after-service-date is flagged.
"""

from __future__ import annotations

import pytest

from src.validation.tool_validation import (
    validate_extraction_result,
    validate_record,
)


class TestNPIFieldRouting:
    """Fields whose name contains 'npi' route to the Luhn check."""

    def test_clean_npi_passes(self) -> None:
        report = validate_record({"billing_provider_npi": "1234567893"})
        v = report["validations"]["billing_provider_npi"]
        assert v["tool"] == "npi_luhn_check"
        assert v["valid"] is True

    def test_dropped_check_digit_npi_fails(self) -> None:
        """The actual live-run failure case from faxed CMS-1500."""
        report = validate_record({"billing_provider_npi": "1234567890"})
        v = report["validations"]["billing_provider_npi"]
        assert v["valid"] is False
        assert v["reason"] == "luhn_fail"
        assert "billing_provider_npi" in report["summary"]["failed_fields"]

    def test_npi_embedded_in_provider_name_string(self) -> None:
        """The clean-run case where Gemma folded NPI into the provider name."""
        report = validate_record(
            {
                "billing_provider_name": "Springfield Family Medicine - NPI 1234567893",
            }
        )
        # The field name carries 'npi'? No — it's 'billing_provider_name'.
        # The routing should only fire on fields whose name contains 'npi'.
        # So this field is skipped.
        assert "billing_provider_name" not in report["validations"]

        # But if the field IS named '_npi' and the value is a string with
        # embedded NPI, we extract the 10-digit run and validate it.
        report2 = validate_record(
            {"rendering_provider_npi": "NPI 1234567893"}
        )
        assert report2["validations"]["rendering_provider_npi"]["valid"] is True


class TestCPTFieldRouting:
    def test_valid_cpt(self) -> None:
        report = validate_record({"cpt_hcpcs": "99213"})
        v = report["validations"]["cpt_hcpcs"]
        assert v["tool"] == "cpt_validate"
        assert v["valid"] is True

    def test_wrong_length_cpt_fails(self) -> None:
        report = validate_record({"procedure_code": "9921"})
        v = report["validations"]["procedure_code"]
        assert v["valid"] is False


class TestICDFieldRouting:
    def test_valid_icd_normalises_with_period(self) -> None:
        report = validate_record({"diagnosis_code_1": "J069"})
        v = report["validations"]["diagnosis_code_1"]
        assert v["tool"] == "icd_normalize"
        assert v["valid"] is True
        assert v["normalised"] == "J06.9"

    def test_dotted_icd_passes(self) -> None:
        report = validate_record({"diagnosis_code_2": "R05.9"})
        assert report["validations"]["diagnosis_code_2"]["valid"] is True


class TestSumReconcile:
    def test_line_matches_total(self) -> None:
        report = validate_record(
            {"charges": 175.00, "total_charge": 175.00}
        )
        assert report["validations"]["_sum_reconcile"]["match"] is True

    def test_line_mismatch_flagged(self) -> None:
        report = validate_record(
            {"charges": 100.00, "total_charge": 203.50}
        )
        v = report["validations"]["_sum_reconcile"]
        assert v["match"] is False
        assert "_sum_reconcile" in report["summary"]["failed_fields"]

    def test_no_line_no_validation(self) -> None:
        report = validate_record({"total_charge": 175.00})
        assert "_sum_reconcile" not in report["validations"]


class TestDateOrdering:
    def test_service_before_dob_flagged(self) -> None:
        report = validate_record(
            {
                "patient_birth_date": "2000-01-01",
                "service_date_from": "1990-01-01",
            }
        )
        v = report["validations"]["_date_ordering"]
        assert v["valid"] is False
        assert "_date_ordering" in report["summary"]["failed_fields"]

    def test_normal_ordering_passes(self) -> None:
        report = validate_record(
            {
                "patient_birth_date": "1985-03-15",
                "service_date_from": "2026-05-10",
            }
        )
        assert report["validations"]["_date_ordering"]["valid"] is True


class TestEmptyFieldsSkipped:
    """Empty values are not flagged — only routed-and-non-empty fields appear."""

    def test_none_value_skipped(self) -> None:
        report = validate_record({"billing_provider_npi": None})
        assert "billing_provider_npi" not in report["validations"]

    def test_empty_string_skipped(self) -> None:
        report = validate_record({"cpt_hcpcs": ""})
        assert "cpt_hcpcs" not in report["validations"]


class TestExtractionResultLevel:
    """``validate_extraction_result`` walks every record."""

    def test_multi_record_walked(self) -> None:
        extraction = {
            "records": [
                {
                    "record_id": 1,
                    "primary_identifier": "Williams, Mary L",
                    "fields": {"billing_provider_npi": "1234567893"},
                },
                {
                    "record_id": 2,
                    "primary_identifier": "Doe, John",
                    "fields": {"billing_provider_npi": "1234567890"},
                },
            ]
        }
        report = validate_extraction_result(extraction)
        assert report["totals"]["records_processed"] == 2
        assert report["totals"]["total_failed_validations"] == 1
        # Record 1 has 0 failed; record 2 has the bad NPI.
        assert report["records"]["1"]["summary"]["failed"] == 0
        assert report["records"]["2"]["summary"]["failed"] == 1


class TestLiveFaxedScenario:
    """The exact field shape from the faxed CMS-1500 live run."""

    def test_full_faxed_record(self) -> None:
        """The live faxed extraction had a hallucinated NPI + sum match."""
        fields = {
            "patient_name": "Williams, Mary L",
            "patient_sex": "F",
            "patient_zip_code": "97403",
            "member_id": "BCHP4421988",
            "group_number": "GRP-77810",
            "diagnosis_code_1": "J06.9",
            "service_date_from": "2026-05-10",
            "cpt_hcpcs": "99213",
            "modifier": "",
            "dx_ptr": "A",
            "units": 1,
            "charges": 175.00,
            "total_charge": 175.00,
            "tax_id_number": "82-1234567",
            "billing_provider_npi": "1234567890",  # the live hallucination
        }
        report = validate_record(fields)
        # NPI hallucination caught.
        assert "billing_provider_npi" in report["summary"]["failed_fields"]
        # Valid fields aren't false-positive flagged.
        assert "cpt_hcpcs" not in report["summary"]["failed_fields"]
        assert "diagnosis_code_1" not in report["summary"]["failed_fields"]
        # Sum reconciles in this case (line == total).
        assert "_sum_reconcile" not in report["summary"]["failed_fields"]
