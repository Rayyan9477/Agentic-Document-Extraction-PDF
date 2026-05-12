"""V3 Phase 5 — Medical-RCM profile end-to-end behaviour tests.

Covers:

* Schema overlay re-introduces healthcare fields when medical-rcm is
  active.
* Generic fallback no longer carries healthcare fields by default.
* Profile prompt fragment is injected into the extraction prompt.
* RCM-specific validators (POS, modifier) work as documented.
"""

from __future__ import annotations

import pytest

from src.profiles import get_profile
from src.prompts.extraction import build_extraction_prompt
from src.schemas import ENHANCED_GENERIC_SCHEMA, get_schema, DocumentType
from src.schemas.profile_overlays import (
    HEALTHCARE_CORE_FIELDS,
    OVERLAY_BUNDLES,
    apply_overlay,
)
from src.schemas.validators import (
    validate_modifier,
    validate_modifier_combination,
    validate_pos_code,
)


# ---------------------------------------------------------------------------
# Generic fallback no longer carries healthcare fields
# ---------------------------------------------------------------------------


class TestGenericFallbackPurged:
    def test_no_patient_name_in_generic(self) -> None:
        names = {f.name for f in ENHANCED_GENERIC_SCHEMA.fields}
        for medical_name in (
            "patient_name",
            "provider_name",
            "service_date",
            "diagnosis_codes",
            "procedure_codes",
        ):
            assert medical_name not in names, (
                f"{medical_name} should NOT be in the generic fallback "
                "after Phase 5; it now lives in the medical-rcm overlay."
            )


# ---------------------------------------------------------------------------
# Overlay re-introduces healthcare fields
# ---------------------------------------------------------------------------


class TestMedicalRCMOverlay:
    def test_overlay_bundle_registered(self) -> None:
        assert "healthcare_core" in OVERLAY_BUNDLES
        assert HEALTHCARE_CORE_FIELDS

    def test_apply_overlay_adds_missing_fields(self) -> None:
        descriptor = get_profile("medical-rcm")
        overlaid = apply_overlay(ENHANCED_GENERIC_SCHEMA, descriptor)
        names = {f.name for f in overlaid.fields}
        assert "patient_name" in names
        assert "diagnosis_codes" in names
        assert "procedure_codes" in names

    def test_apply_overlay_does_not_duplicate(self) -> None:
        # If the base schema already has a field with the same name,
        # the overlay must not add a duplicate.
        cms = get_schema(DocumentType.CMS_1500)
        descriptor = get_profile("medical-rcm")
        overlaid = apply_overlay(cms, descriptor)
        names = [f.name for f in overlaid.fields]
        for n in set(names):
            assert names.count(n) == 1, f"duplicate field name {n!r}"

    def test_generic_profile_overlay_is_no_op(self) -> None:
        # Generic profile has no overlay; apply_overlay must return
        # the original schema unchanged.
        descriptor = get_profile("generic-document")
        out = apply_overlay(ENHANCED_GENERIC_SCHEMA, descriptor)
        assert out is ENHANCED_GENERIC_SCHEMA

    def test_overlay_does_not_mutate_original(self) -> None:
        descriptor = get_profile("medical-rcm")
        before_fields = list(ENHANCED_GENERIC_SCHEMA.fields)
        apply_overlay(ENHANCED_GENERIC_SCHEMA, descriptor)
        assert list(ENHANCED_GENERIC_SCHEMA.fields) == before_fields


# ---------------------------------------------------------------------------
# Prompt fragment injection
# ---------------------------------------------------------------------------


class TestRCMPromptFragment:
    def test_medical_rcm_fragment_present(self) -> None:
        prompt = build_extraction_prompt(
            schema_fields=[{"name": "x", "field_type": "string"}],
            document_type="CMS-1500",
            page_number=1,
            total_pages=1,
            profile="medical-rcm",
        )
        assert "MEDICAL / RCM PROFILE NOTES" in prompt
        assert "CPT codes" in prompt
        assert "NPI" in prompt

    def test_generic_profile_emits_no_fragment(self) -> None:
        prompt = build_extraction_prompt(
            schema_fields=[{"name": "x", "field_type": "string"}],
            document_type="OTHER",
            page_number=1,
            total_pages=1,
            profile="generic-document",
        )
        assert "MEDICAL / RCM PROFILE NOTES" not in prompt
        assert "FINANCE PROFILE NOTES" not in prompt

    def test_no_profile_emits_no_fragment(self) -> None:
        prompt = build_extraction_prompt(
            schema_fields=[{"name": "x", "field_type": "string"}],
            document_type="OTHER",
            page_number=1,
            total_pages=1,
            profile=None,
        )
        assert "PROFILE NOTES" not in prompt

    def test_unknown_profile_silent_fallback(self) -> None:
        # An unknown profile name must not blow up — it logs and
        # produces a prompt with no fragment (or generic's empty
        # fragment).
        prompt = build_extraction_prompt(
            schema_fields=[{"name": "x", "field_type": "string"}],
            document_type="OTHER",
            page_number=1,
            total_pages=1,
            profile="nope-not-real",
        )
        # Falls through to generic which has empty fragment.
        assert "MEDICAL / RCM PROFILE NOTES" not in prompt


# ---------------------------------------------------------------------------
# POS code validator
# ---------------------------------------------------------------------------


class TestPOSValidator:
    def test_office(self) -> None:
        v = validate_pos_code("11")
        assert v.is_valid
        assert v.details["name"] == "Office"
        assert v.details["type"] == "non-facility"

    def test_inpatient(self) -> None:
        v = validate_pos_code("21")
        assert v.is_valid
        assert v.details["type"] == "facility"

    def test_telehealth_in_home(self) -> None:
        v = validate_pos_code("10")
        assert v.is_valid
        assert "Telehealth" in v.details["name"]

    def test_zero_pads_single_digit(self) -> None:
        v = validate_pos_code("1")
        assert v.normalized_value == "01"

    def test_int_input_zero_padded(self) -> None:
        v = validate_pos_code(11)
        assert v.is_valid
        assert v.normalized_value == "11"

    def test_alpha_rejected(self) -> None:
        v = validate_pos_code("XX")
        assert not v.is_valid

    def test_3_digit_rejected(self) -> None:
        v = validate_pos_code("999")
        assert not v.is_valid

    def test_unknown_2digit_warns(self) -> None:
        # A format-valid but unpublished code warns (not invalid).
        v = validate_pos_code("88")
        assert v.result.value == "warning"
        assert v.normalized_value == "88"

    def test_none_invalid(self) -> None:
        assert not validate_pos_code(None).is_valid


# ---------------------------------------------------------------------------
# Modifier validator
# ---------------------------------------------------------------------------


class TestModifierValidator:
    def test_25_on_em(self) -> None:
        v = validate_modifier("25", cpt_code="99213")
        assert v.is_valid
        assert v.details["cpt_category"] == "E_M"

    def test_lt_on_surgery(self) -> None:
        v = validate_modifier("LT", cpt_code="29881")
        assert v.is_valid

    def test_25_outside_em_warns(self) -> None:
        # 25 is conventionally E/M-only; using it on a radiology
        # code (70010-79999) should warn.
        v = validate_modifier("25", cpt_code="70450")
        assert v.result.value == "warning"

    def test_50_blocks_with_lt(self) -> None:
        v = validate_modifier("50", cpt_code="29881", other_modifiers=["LT"])
        assert v.result.value == "warning"
        assert "LT" in v.details["conflicts"]

    def test_24_blocks_with_25(self) -> None:
        v = validate_modifier("24", cpt_code="99213", other_modifiers=["25"])
        assert v.result.value == "warning"

    def test_alpha_format_rejected(self) -> None:
        v = validate_modifier("X")
        assert not v.is_valid

    def test_unknown_format_valid_warns(self) -> None:
        v = validate_modifier("ZZ")
        assert v.result.value == "warning"

    def test_combination_rolls_up(self) -> None:
        # All-valid combination → VALID.
        result = validate_modifier_combination("99213", ["25", "GA"])
        assert result.is_valid

    def test_combination_warning_propagates(self) -> None:
        # 50 + LT conflict → warning at the combination level.
        result = validate_modifier_combination("29881", ["50", "LT"])
        assert result.result.value == "warning"

    def test_combination_invalid_dominates(self) -> None:
        # An INVALID format trumps everything else.
        result = validate_modifier_combination("99213", ["25", "@@"])
        assert not result.is_valid


# ---------------------------------------------------------------------------
# CPT-ICD pairing schema rule presence
# ---------------------------------------------------------------------------


class TestRCMSchemaRules:
    def test_cms1500_has_sum_equals_rule(self) -> None:
        cms = get_schema(DocumentType.CMS_1500)
        ops = {r.operator.value for r in cms.cross_field_rules}
        assert "sum_equals" in ops

    def test_ub04_has_sum_equals_rule(self) -> None:
        ub04 = get_schema(DocumentType.UB_04)
        ops = {r.operator.value for r in ub04.cross_field_rules}
        assert "sum_equals" in ops

    def test_eob_has_sum_equals_rule(self) -> None:
        eob = get_schema(DocumentType.EOB)
        ops = {r.operator.value for r in eob.cross_field_rules}
        assert "sum_equals" in ops
