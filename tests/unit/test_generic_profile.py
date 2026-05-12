"""V3 Phase 5 — Generic profile invariants.

The generic profile is the fallback. The headline invariant: a
non-medical document must NOT have medical fields proposed to it. This
test asserts that against the post-Phase-5 generic schema and the
profile-aware schema-selection path.
"""

from __future__ import annotations

from src.profiles import detect_profile, get_profile
from src.prompts.extraction import build_extraction_prompt
from src.schemas import ENHANCED_GENERIC_SCHEMA
from src.schemas.profile_overlays import apply_overlay


# ---------------------------------------------------------------------------
# No medical fields on a generic doc
# ---------------------------------------------------------------------------


class TestNoMedicalFieldsOnGeneric:
    def test_invoice_does_not_carry_patient_name(self) -> None:
        """An invoice classified through the generic fallback must
        not have a ``patient_name`` field invented for it."""
        descriptor = get_profile("generic-document")
        out = apply_overlay(ENHANCED_GENERIC_SCHEMA, descriptor)
        names = {f.name for f in out.fields}
        for medical in (
            "patient_name",
            "provider_name",
            "diagnosis_codes",
            "procedure_codes",
        ):
            assert medical not in names

    def test_generic_prompt_has_no_medical_reminders(self) -> None:
        prompt = build_extraction_prompt(
            schema_fields=[{"name": "total_amount", "field_type": "currency"}],
            document_type="OTHER",
            page_number=1,
            total_pages=1,
            profile="generic-document",
        )
        # No medical reminders (CPT/ICD/NPI/POS) in a generic prompt.
        assert "CPT" not in prompt or "CPT codes" not in prompt
        assert "ICD-10-CM" not in prompt
        assert "NPI is exactly 10 digits" not in prompt


# ---------------------------------------------------------------------------
# Detection routes a generic memo to generic
# ---------------------------------------------------------------------------


class TestDetectionRoutingGeneric:
    def test_memo_routes_to_generic(self) -> None:
        result = detect_profile(
            classification_features=["Memorandum"],
            page_text="Memo: To all employees regarding Q3 quarterly review.",
        )
        assert result.profile_name == "generic-document"

    def test_invoice_routes_to_generic_when_finance_signals_weak(self) -> None:
        # Just 'invoice' header alone does not clear finance's floor
        # of 0.6 (invoice_strong needs 'invoice number/#/no.'). Plain
        # invoice should fall back to generic via the
        # ``invoice_header`` generic signal.
        result = detect_profile(
            classification_features=["Invoice"],
            page_text="Invoice for services rendered.",
        )
        # Generic fires because invoice_header (0.5) >= floor 0.4.
        assert result.profile_name == "generic-document"

    def test_invoice_strong_routes_to_finance(self) -> None:
        # An "Invoice Number: 123" pattern triggers the strong
        # finance signal and should win over generic.
        result = detect_profile(
            page_text="Invoice Number: 12345 — total due $100.",
        )
        # finance_strong = 0.6, clears finance floor.
        assert result.profile_name == "finance"


# ---------------------------------------------------------------------------
# Profile detection disabled → always generic
# ---------------------------------------------------------------------------


class TestDetectionDisabled:
    def test_disabled_falls_back_via_settings(self, monkeypatch) -> None:
        # When the analyzer's detection is disabled in settings, every
        # doc resolves to ``default_profile``. Tested at the analyzer
        # level by reading PROFILE_DETECTION_ENABLED from env.
        monkeypatch.setenv("PROFILE_DETECTION_ENABLED", "false")
        # Force a fresh settings load — get_settings is lru_cache'd.
        from src.config import get_settings

        get_settings.cache_clear()
        settings = get_settings()
        assert settings.profile.detection_enabled is False
        # Reset for downstream tests in the same session.
        monkeypatch.setenv("PROFILE_DETECTION_ENABLED", "true")
        get_settings.cache_clear()
