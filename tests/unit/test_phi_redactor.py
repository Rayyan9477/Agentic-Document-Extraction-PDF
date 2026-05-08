"""WS-6: tests for the opt-in PHI redactor.

The HuggingFace ``openai/privacy-filter`` model is not assumed to be
pre-vendored on CI / dev hosts, so these tests force the regex
fallback path by patching ``_get_pipeline`` at the class level (an
autouse fixture). The ML path is tested separately via
``_apply_entity_spans``, the pure-function span builder.
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.security.phi_mask import REDACTED_TOKEN
from src.security.phi_redactor import PHIRedactor, _apply_entity_spans


@pytest.fixture(autouse=True)
def force_regex_fallback() -> object:
    """Disable the ML loader globally for these tests.

    Patches ``PHIRedactor._get_pipeline`` to always return ``None`` so the
    redactor never tries to hit HuggingFace Hub during a test run. The
    ML span-construction logic is covered separately via the pure
    ``_apply_entity_spans`` tests.
    """
    with patch.object(PHIRedactor, "_get_pipeline", return_value=None):
        yield


class TestRedactorBasics:
    def test_empty_input_is_noop(self) -> None:
        r = PHIRedactor(fallback_to_regex=True)
        assert r.redact("").redacted_text == ""
        assert r.redact("").spans == []

    def test_non_string_input_returns_empty_string(self) -> None:
        r = PHIRedactor(fallback_to_regex=True)
        # Defensive: we don't crash on accidental dict / int input.
        result = r.redact(None)  # type: ignore[arg-type]
        assert result.redacted_text == ""

    def test_clean_text_passes_through(self) -> None:
        r = PHIRedactor(fallback_to_regex=True)
        result = r.redact("Routine follow-up.")
        # No PHI shapes -> regex fallback emits no spans, text unchanged.
        assert result.redacted_text == "Routine follow-up."
        assert result.spans == []


class TestRegexFallback:
    """Without transformers installed, the redactor uses the regex layer."""

    def test_ssn_redacted(self) -> None:
        r = PHIRedactor(fallback_to_regex=True)
        result = r.redact("SSN: 123-45-6789.")
        assert result.layer_used == "regex"
        # Regex layer collapses any matched value to a single REDACTED token
        # because it can't reliably map back to a precise span.
        assert result.redacted_text == REDACTED_TOKEN
        assert len(result.spans) == 1
        assert result.spans[0].label == "regex"

    def test_phone_redacted(self) -> None:
        r = PHIRedactor(fallback_to_regex=True)
        result = r.redact("Call 555-123-4567 today")
        assert result.redacted_text == REDACTED_TOKEN

    def test_email_redacted(self) -> None:
        r = PHIRedactor(fallback_to_regex=True)
        result = r.redact("alice@example.com")
        assert result.redacted_text == REDACTED_TOKEN

    def test_no_fallback_when_disabled(self) -> None:
        # With fallback_to_regex=False AND transformers unavailable in
        # test env, the redactor is a no-op (logs a warning).
        r = PHIRedactor(fallback_to_regex=False)
        result = r.redact("SSN: 123-45-6789")
        # Without any layer the text passes through.
        assert result.redacted_text == "SSN: 123-45-6789"
        assert result.layer_used == "noop"


class TestRecordWalking:
    def test_redact_record_walks_nested_dicts(self) -> None:
        r = PHIRedactor(fallback_to_regex=True)
        record = {
            "patient": {
                "name": "Alice Smith",
                "phone": "555-123-4567",
            },
            "amount": 250.0,
            "notes": "Routine.",
        }
        out = r.redact_record(record)
        # Phone matches a regex pattern -> redacted to token.
        assert out["patient"]["phone"] == REDACTED_TOKEN
        # Plain text without PHI shapes passes through.
        assert out["notes"] == "Routine."
        # Non-string scalars untouched.
        assert out["amount"] == 250.0

    def test_redact_record_walks_lists(self) -> None:
        r = PHIRedactor(fallback_to_regex=True)
        record = {"emails": ["a@x.com", "plain text"]}
        out = r.redact_record(record)
        assert out["emails"][0] == REDACTED_TOKEN
        assert out["emails"][1] == "plain text"

    def test_redact_record_does_not_mutate_input(self) -> None:
        r = PHIRedactor(fallback_to_regex=True)
        record = {"phone": "555-123-4567"}
        snapshot = dict(record)
        r.redact_record(record)
        assert record == snapshot


class TestApplyEntitySpans:
    """Pure-function unit tests for the ML-output span builder.

    Exercised without the ML model by feeding pre-built entity dicts
    that mimic what ``transformers.pipeline`` produces with
    ``aggregation_strategy="simple"``.
    """

    def test_single_entity_replaces_with_token(self) -> None:
        text = "Hello Alice Smith, welcome."
        entities = [
            {"start": 6, "end": 17, "entity_group": "private_person"},
        ]
        result = _apply_entity_spans(text, entities)
        assert result.redacted_text == f"Hello {REDACTED_TOKEN}, welcome."
        assert len(result.spans) == 1
        assert result.spans[0].label == "private_person"
        assert result.spans[0].original == "Alice Smith"

    def test_multiple_entities_replaced_in_order(self) -> None:
        text = "Alice (555-1234) at alice@x.com"
        entities = [
            {"start": 0, "end": 5, "entity_group": "private_person"},
            {"start": 7, "end": 15, "entity_group": "private_phone"},
            {"start": 20, "end": 31, "entity_group": "private_email"},
        ]
        result = _apply_entity_spans(text, entities)
        assert "Alice" not in result.redacted_text
        assert "555-1234" not in result.redacted_text
        assert "alice@x.com" not in result.redacted_text
        assert result.redacted_text.count(REDACTED_TOKEN) == 3
        assert len(result.spans) == 3

    def test_overlapping_entities_merged(self) -> None:
        # Two entities overlap by one character; should merge into one
        # span covering [0, 12).
        text = "Alice Smith"
        entities = [
            {"start": 0, "end": 5, "entity_group": "private_person"},
            {"start": 4, "end": 11, "entity_group": "private_person"},
        ]
        result = _apply_entity_spans(text, entities)
        assert result.redacted_text == REDACTED_TOKEN
        assert len(result.spans) == 1

    def test_invalid_offsets_dropped(self) -> None:
        text = "short"
        entities = [
            {"start": -1, "end": 5},  # negative start -> drop
            {"start": 0, "end": 100},  # end past text -> drop
            {"start": 3, "end": 2},  # end <= start -> drop
        ]
        result = _apply_entity_spans(text, entities)
        # All entities invalid → no redaction
        assert result.redacted_text == "short"
        assert result.spans == []

    def test_empty_entities_returns_original(self) -> None:
        text = "no PHI here"
        result = _apply_entity_spans(text, [])
        assert result.redacted_text == text
        assert result.spans == []
        assert result.layer_used == "ml"


class TestFromSettings:
    def test_constructs_disabled_redactor_when_settings_off(self) -> None:
        # Default settings have phi.enabled = False; from_settings still
        # returns a redactor (so per-request override paths work) but
        # with fallback_to_regex from settings (default True).
        r = PHIRedactor.from_settings()
        assert isinstance(r, PHIRedactor)
        assert r._fallback_to_regex is True

    def test_honours_explicit_settings(self) -> None:
        from unittest.mock import MagicMock

        fake_settings = MagicMock()
        fake_settings.phi.enabled = True
        fake_settings.phi.model_id = "openai/privacy-filter"
        fake_settings.phi.local_only = True
        fake_settings.phi.fallback_to_regex = False

        with patch("src.security.phi_redactor.get_settings", return_value=fake_settings):
            r = PHIRedactor.from_settings()
        assert r._fallback_to_regex is False
        assert r._local_only is True
        assert r._model_id == "openai/privacy-filter"
