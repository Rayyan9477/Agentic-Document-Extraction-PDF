"""Tests for src.security.phi_mask.enforce_mask_phi (WS-1)."""

from __future__ import annotations

import pytest

from src.security.phi_mask import (
    REDACTED_TOKEN,
    _is_phi_field_name,
    enforce_mask_phi,
)


class TestFieldNameDetection:
    """The default substring matcher used when no explicit field set is given."""

    @pytest.mark.parametrize(
        "field_name",
        [
            "patient_name",
            "Patient_DOB",
            "subscriber_id",
            "MEMBER_id",
            "ssn",
            "social_security_number",
            "email_address",
            "phone_number",
            "fax_line",
            "guarantor_address",
            "city_zip",
            "policy_number",
            "ip_address",
        ],
    )
    def test_phi_field_names_detected(self, field_name: str) -> None:
        assert _is_phi_field_name(field_name) is True

    @pytest.mark.parametrize(
        "field_name",
        ["amount", "total", "diagnosis_code", "service_date_count", "place_of_service"],
    )
    def test_non_phi_field_names_pass_through(self, field_name: str) -> None:
        assert _is_phi_field_name(field_name) is False


class TestEnforceMaskPhi:
    """End-to-end behaviour of the export-time redaction primitive."""

    def test_default_pattern_redacts_phi_fields(self) -> None:
        record = {
            "patient_name": "John Doe",
            "patient_dob": "01/01/1990",
            "amount": 250.0,
            "diagnosis_code": "E11.9",
        }
        masked = enforce_mask_phi(record)
        assert masked["patient_name"] == REDACTED_TOKEN
        assert masked["patient_dob"] == REDACTED_TOKEN
        # Non-PHI numeric stays
        assert masked["amount"] == 250.0
        # Diagnosis codes are clinical, not PHI
        assert masked["diagnosis_code"] == "E11.9"

    def test_explicit_field_set_overrides_pattern(self) -> None:
        record = {"foo": "John Doe", "patient_name": "Jane"}
        masked = enforce_mask_phi(record, phi_field_names={"foo"}, redact_values=False)
        # `foo` is explicitly listed; redact it.
        assert masked["foo"] == REDACTED_TOKEN
        # `patient_name` is NOT in the explicit set; pattern path is bypassed.
        assert masked["patient_name"] == "Jane"

    def test_value_level_redaction_catches_ssn_phone_email(self) -> None:
        record = {
            "notes": "Contact at 555-123-4567 or john@example.com; SSN 123-45-6789.",
            "amount": 250.0,
        }
        masked = enforce_mask_phi(record)
        # Whole notes value collapses to REDACTED because PHI shapes are present.
        assert masked["notes"] == REDACTED_TOKEN
        assert masked["amount"] == 250.0

    def test_redact_values_off_preserves_clean_strings(self) -> None:
        record = {"notes": "Routine follow-up"}
        masked = enforce_mask_phi(record, redact_values=False)
        assert masked["notes"] == "Routine follow-up"

    def test_nested_dicts_walked_recursively(self) -> None:
        record = {
            "subscriber": {"name": "John", "id": 42},
            "service_date": "2024-01-01",
        }
        masked = enforce_mask_phi(record)
        # `subscriber.name` is detected by parent_field='name'.
        assert masked["subscriber"]["name"] == REDACTED_TOKEN
        # `subscriber.id` is detected because parent_field='id' isn't a PHI hit
        # but the *parent* key 'subscriber' isn't propagated; this assertion
        # documents that nested non-PHI scalars survive.
        assert masked["subscriber"]["id"] == 42

    def test_lists_walked_recursively(self) -> None:
        record = {
            "patient_emails": ["a@x.com", "b@y.com"],
            "amounts": [10.0, 20.0],
        }
        masked = enforce_mask_phi(record)
        # parent_field='patient_emails' contains 'email' — list values redacted.
        assert masked["patient_emails"] == [REDACTED_TOKEN, REDACTED_TOKEN]
        assert masked["amounts"] == [10.0, 20.0]

    def test_input_is_never_mutated(self) -> None:
        record = {"patient_name": "John Doe"}
        snapshot = dict(record)
        enforce_mask_phi(record)
        assert record == snapshot

    def test_none_values_preserved(self) -> None:
        record = {"patient_name": None, "amount": None}
        masked = enforce_mask_phi(record)
        assert masked["patient_name"] is None
        assert masked["amount"] is None

    def test_extra_patterns_extend_default_set(self) -> None:
        record = {"custom_phi_field": "secret", "amount": 1.0}
        masked = enforce_mask_phi(record, extra_patterns=("custom_phi",))
        assert masked["custom_phi_field"] == REDACTED_TOKEN
        assert masked["amount"] == 1.0


class TestNPILuhn:
    """The rewritten Luhn check still accepts known-valid NPIs and rejects mutations."""

    def test_valid_npi_passes(self) -> None:
        from src.schemas.validators import _luhn_checksum

        # 1234567893 is the canonical valid example NPI used in CMS docs.
        assert _luhn_checksum("1234567893") is True

    def test_mutated_npi_fails(self) -> None:
        from src.schemas.validators import _luhn_checksum

        # Flip the check digit; should now fail.
        assert _luhn_checksum("1234567890") is False
        assert _luhn_checksum("1234567894") is False


class TestTokenMasking:
    """Phase 8.5-A3 — JWT / Bearer / token-query masking.

    These tests verify that bearer tokens, JWTs, and ``refresh_token=``
    query strings are scrubbed from arbitrary text without erasing
    surrounding context. The same patterns also feed the structlog
    PHI processor via ``logging_config.PHI_PATTERNS``.
    """

    def test_jwt_in_text_is_masked_inline(self) -> None:
        from src.security.phi_mask import mask_tokens_in_text

        # Canonical 3-segment JWT shape.
        jwt = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NSJ9.SflKxwRJSMeKKF2QT4"
        text = f"User context: {jwt} (audit trail)"
        result = mask_tokens_in_text(text)
        assert jwt not in result
        assert "[TOKEN-MASKED]" in result
        assert "User context:" in result and "(audit trail)" in result

    def test_bearer_header_value_is_masked(self) -> None:
        from src.security.phi_mask import mask_tokens_in_text

        result = mask_tokens_in_text("Authorization: Bearer abc.def.ghi_secret")
        assert "abc.def.ghi_secret" not in result
        assert "Bearer [TOKEN-MASKED]" in result

    def test_refresh_token_query_string_is_masked(self) -> None:
        from src.security.phi_mask import mask_tokens_in_text

        text = "foo=bar&refresh_token=eyJ.payload.sig&grant_type=refresh"
        result = mask_tokens_in_text(text)
        assert "eyJ.payload.sig" not in result
        assert "refresh_token=[TOKEN-MASKED]" in result
        # Surrounding query params preserved.
        assert "foo=bar" in result and "grant_type=refresh" in result

    def test_access_token_and_api_key_also_masked(self) -> None:
        from src.security.phi_mask import mask_tokens_in_text

        for keyword in ("access_token", "api_key", "secret", "password", "token"):
            text = f"{keyword}=supersecret123"
            result = mask_tokens_in_text(text)
            assert "supersecret123" not in result
            assert f"{keyword}=[TOKEN-MASKED]" in result

    def test_non_token_text_passes_through(self) -> None:
        from src.security.phi_mask import mask_tokens_in_text

        text = "Plain log line with no secrets, just user_id=alice."
        assert mask_tokens_in_text(text) == text

    def test_phi_mask_redacts_whole_value_containing_token(self) -> None:
        """Field values that are themselves tokens redact to ``[REDACTED]``."""
        record = {"access_token": "eyJhbGciOiJSUzI1NiJ9.payload.signature"}
        masked = enforce_mask_phi(record, phi_field_names=set(), redact_values=True)
        # No PHI field set; falls to value-level scan. Token pattern hits → REDACTED.
        assert masked["access_token"] == REDACTED_TOKEN

    def test_logging_config_pattern_list_includes_tokens_first(self) -> None:
        """``PHI_PATTERNS`` prepends token patterns so they run before PHI."""
        from src.config.logging_config import (
            PHI_PATTERNS,
            _TOKEN_PATTERNS_WITH_REPLACEMENTS,
        )

        # The first entries in PHI_PATTERNS must be the token entries, in order.
        for i, (pattern, replacement) in enumerate(_TOKEN_PATTERNS_WITH_REPLACEMENTS):
            assert PHI_PATTERNS[i] == (pattern, replacement)

    def test_logging_config_token_patterns_stay_in_sync(self) -> None:
        """``logging_config`` mirrors ``phi_mask`` token patterns.

        The two lists are intentionally duplicated to break a circular
        import (see logging_config.py docstring); this test catches drift.
        """
        from src.config.logging_config import _TOKEN_PATTERNS_WITH_REPLACEMENTS as L
        from src.security.phi_mask import TOKEN_PATTERNS_WITH_REPLACEMENTS as P

        assert len(L) == len(P), (
            "Token-pattern lists in logging_config and phi_mask diverged. "
            "Sync the two tuples so audit-log masking and PHI redaction stay aligned."
        )
        for (lp, lr), (pp, pr) in zip(L, P, strict=True):
            assert lp.pattern == pp.pattern
            assert lp.flags == pp.flags
            assert lr == pr
