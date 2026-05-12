"""V3 Phase 5 — Profile registry + auto-detection unit tests.

Coverage:

* Registry singleton: register / get / names / reset.
* ``get_profile`` falls back to generic for unknown names.
* ``detect_profile`` scoring, confidence floor, fallback semantics.
* Operator ``profile_override`` short-circuits scoring.
* Profile descriptor ``to_serialisable`` shape.
"""

from __future__ import annotations

import re

import pytest

from src.profiles import (
    ProfileDescriptor,
    ProfileDetectionResult,
    ProfileRegistry,
    ProfileSignal,
    detect_profile,
    get_profile,
)
from src.profiles.descriptor import compile_signal


# ---------------------------------------------------------------------------
# Registry semantics
# ---------------------------------------------------------------------------


class TestProfileRegistry:
    def test_registry_is_singleton(self) -> None:
        a = ProfileRegistry()
        b = ProfileRegistry()
        assert a is b

    def test_built_in_profiles_registered(self) -> None:
        names = ProfileRegistry().names()
        for required in ("generic-document", "medical-rcm", "finance"):
            assert required in names

    def test_get_unknown_falls_back_to_generic(self) -> None:
        descriptor = ProfileRegistry().get("does-not-exist")
        assert descriptor.name == "generic-document"

    def test_get_profile_helper_handles_none(self) -> None:
        assert get_profile(None).name == "generic-document"
        assert get_profile("").name == "generic-document"
        assert get_profile("medical-rcm").name == "medical-rcm"


# ---------------------------------------------------------------------------
# Detection: medical-rcm
# ---------------------------------------------------------------------------


class TestDetectMedicalRCM:
    def test_strong_header_alone_clears_floor(self) -> None:
        # Single strong signal (HCFA header) is enough.
        result = detect_profile(
            classification_features=["HEALTH INSURANCE CLAIM FORM"],
            document_type=None,
        )
        assert result.profile_name == "medical-rcm"
        assert result.confidence >= 0.6
        assert result.fallback_to_generic is False
        assert "hcfa_header" in result.matched_signals["medical-rcm"]

    def test_document_type_cms1500_drives_detection(self) -> None:
        # Just the classifier's ``CMS-1500`` is enough.
        result = detect_profile(
            classification_features=[],
            document_type="CMS-1500",
        )
        assert result.profile_name == "medical-rcm"
        assert "document_type_cms1500" in result.matched_signals["medical-rcm"]

    def test_three_weak_signals_clear_floor(self) -> None:
        result = detect_profile(
            page_text="patient information NPI ICD-10 CPT modifier 25",
            document_type=None,
        )
        # NPI + ICD + CPT + modifier ⇒ ≥ 0.8 score
        assert result.profile_name == "medical-rcm"

    def test_two_weak_signals_do_not_clear_floor(self) -> None:
        # NPI + ICD only = 0.4, below medical-rcm floor of 0.6.
        result = detect_profile(
            page_text="page mentions NPI and ICD-10 codes only",
            document_type=None,
        )
        # Should fall back to generic.
        assert result.profile_name == "generic-document"
        assert result.fallback_to_generic is True


# ---------------------------------------------------------------------------
# Detection: finance
# ---------------------------------------------------------------------------


class TestDetectFinance:
    def test_w2_header(self) -> None:
        result = detect_profile(
            classification_features=["WAGE AND TAX STATEMENT"],
        )
        assert result.profile_name == "finance"

    def test_form_1099(self) -> None:
        result = detect_profile(
            classification_features=["Form 1099-NEC"],
        )
        assert result.profile_name == "finance"


# ---------------------------------------------------------------------------
# Detection: generic fallback
# ---------------------------------------------------------------------------


class TestDetectGenericFallback:
    def test_blank_input_falls_back(self) -> None:
        result = detect_profile(
            classification_features=[],
            page_text="",
            document_type=None,
        )
        assert result.profile_name == "generic-document"
        assert result.fallback_to_generic is True

    def test_clear_memo_picks_generic(self) -> None:
        result = detect_profile(
            classification_features=["Memo"],
            page_text="To: All staff. Memorandum regarding Q3 results.",
        )
        # Generic floor is 0.4; "memo" + "memorandum" both fire.
        assert result.profile_name == "generic-document"
        assert result.fallback_to_generic is False


# ---------------------------------------------------------------------------
# Detection: operator override
# ---------------------------------------------------------------------------


class TestDetectOverride:
    def test_override_short_circuits_scoring(self) -> None:
        result = detect_profile(
            classification_features=["HEALTH INSURANCE CLAIM FORM"],
            profile_override="finance",
        )
        # Despite strong medical signals, the override wins.
        assert result.profile_name == "finance"
        assert result.confidence == 1.0
        assert result.matched_signals["finance"] == ["operator_override"]

    def test_override_unknown_name_falls_back_to_generic(self) -> None:
        # Unknown names resolve via ``ProfileRegistry.get`` which falls
        # back to generic. We still report the override path.
        result = detect_profile(
            classification_features=["WAGE AND TAX STATEMENT"],
            profile_override="not-a-real-profile",
        )
        assert result.profile_name == "generic-document"


# ---------------------------------------------------------------------------
# Descriptor invariants
# ---------------------------------------------------------------------------


class TestProfileDescriptor:
    def test_medical_rcm_has_overlay(self) -> None:
        descriptor = get_profile("medical-rcm")
        assert "healthcare_core" in descriptor.schema_overlay_fields
        assert descriptor.enabled_emitters == ("ccda", "x12_275")

    def test_generic_has_no_overlay(self) -> None:
        descriptor = get_profile("generic-document")
        assert descriptor.schema_overlay_fields == ()
        assert descriptor.prompt_fragment == ""

    def test_to_serialisable_shape(self) -> None:
        s = get_profile("medical-rcm").to_serialisable()
        assert s["name"] == "medical-rcm"
        assert "signals" in s and isinstance(s["signals"], list)
        assert "validator_packs" in s

    def test_signal_match_helper(self) -> None:
        sig = compile_signal(
            name="x", pattern=r"hello", score=0.5, description="test",
        )
        assert sig.matches("Hello world!")
        assert not sig.matches("nope")


# ---------------------------------------------------------------------------
# ProfileDetectionResult shape
# ---------------------------------------------------------------------------


class TestDetectionResultShape:
    def test_result_carries_score_map(self) -> None:
        result = detect_profile(
            classification_features=["HEALTH INSURANCE CLAIM FORM"],
        )
        # Every registered profile should appear in the score map.
        for name in ProfileRegistry().names():
            assert name in result.score_by_profile
        # medical-rcm should outscore everyone.
        assert (
            result.score_by_profile["medical-rcm"]
            > result.score_by_profile["generic-document"]
        )

    def test_dataclass_is_frozen(self) -> None:
        result = detect_profile(classification_features=[])
        with pytest.raises(Exception):  # FrozenInstanceError
            result.profile_name = "x"  # type: ignore[misc]
