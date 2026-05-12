"""
Phase 4 — provenance data model unit tests.

Coverage:

* ``Provenance`` schema — required/optional fields, range validation,
  ``append_stage`` immutability, ``to_serialisable`` shape.
* ``FieldValue`` round-trip — wrap, serialise, ``unwrap_value``,
  ``unwrap_provenance``.
* ``PHIFieldValue`` adds redaction fields without breaking the
  ``FieldValue`` invariant.
* ``unwrap_value(strict=True)`` raises ``ProvenanceMissingError`` on
  bare scalars.
* ``is_field_value`` recognises both Pydantic instances and serialised
  dicts.
* ``FieldMetadata.to_provenance()`` bridges legacy → V3 shape.
* ``empty_provenance`` and ``LEGACY_SENTINEL_PROVENANCE`` are valid.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from src.pipeline.provenance import (
    FieldValue,
    LEGACY_SENTINEL_PROVENANCE,
    PHIFieldValue,
    Provenance,
    ProvenanceMissingError,
    empty_provenance,
    is_field_value,
    unwrap_provenance,
    unwrap_value,
    wrap_value,
)
from src.pipeline.state import BoundingBoxCoords, FieldMetadata


# ---------------------------------------------------------------------------
# Provenance schema
# ---------------------------------------------------------------------------


class TestProvenanceSchema:
    def test_minimum_valid_construction(self) -> None:
        p = Provenance(page=1)
        assert p.page == 1
        assert p.bbox is None
        assert p.confidence == 0.0
        assert p.extraction_path == []

    def test_full_construction(self) -> None:
        bbox = BoundingBoxCoords(x=0.1, y=0.2, width=0.3, height=0.04, page=1)
        p = Provenance(
            page=1,
            bbox=bbox,
            source_block_id="blk_p1_003",
            extraction_path=["pass1_vlm", "reconciler"],
            agent_signatures=["extractor", "reconciler"],
            confidence=0.94,
            vlm_model_id="qwen3.6-27b-vl@8001",
            mem0_match="provider_4521_doc_889",
        )
        assert p.confidence == 0.94
        assert p.vlm_model_id == "qwen3.6-27b-vl@8001"
        assert p.mem0_match == "provider_4521_doc_889"

    def test_negative_page_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Provenance(page=-1)

    def test_confidence_out_of_range_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Provenance(page=1, confidence=1.5)

    def test_extra_keys_allowed_for_forward_compat(self) -> None:
        p = Provenance.model_validate(
            {"page": 1, "future_field_we_dont_know_about": "value"}
        )
        assert p.page == 1


class TestProvenanceAppendStage:
    def test_appends_to_path(self) -> None:
        p = Provenance(page=1, extraction_path=["pass1_vlm"])
        p2 = p.append_stage("reconciler")
        assert p2.extraction_path == ["pass1_vlm", "reconciler"]

    def test_immutable_original(self) -> None:
        p = Provenance(page=1, extraction_path=["pass1_vlm"])
        p.append_stage("reconciler")
        # Original unchanged.
        assert p.extraction_path == ["pass1_vlm"]

    def test_appends_agent_when_supplied(self) -> None:
        p = Provenance(page=1, agent_signatures=["extractor"])
        p2 = p.append_stage("validator", agent="validator")
        assert "validator" in p2.agent_signatures

    def test_dedupes_agent(self) -> None:
        p = Provenance(page=1, agent_signatures=["extractor"])
        p2 = p.append_stage("validate", agent="extractor")
        # No duplicate added.
        assert p2.agent_signatures.count("extractor") == 1


class TestProvenanceSerialise:
    def test_to_serialisable_keys(self) -> None:
        p = Provenance(page=1, confidence=0.5)
        d = p.to_serialisable()
        assert set(d.keys()) >= {
            "page", "bbox", "source_block_id", "extraction_path",
            "agent_signatures", "confidence", "vlm_model_id", "mem0_match",
        }

    def test_bbox_serialised_as_dict(self) -> None:
        bbox = BoundingBoxCoords(x=0.1, y=0.2, width=0.3, height=0.04, page=1)
        p = Provenance(page=1, bbox=bbox)
        d = p.to_serialisable()
        assert isinstance(d["bbox"], dict)
        assert d["bbox"].get("x") == pytest.approx(0.1)

    def test_null_bbox_round_trips_as_none(self) -> None:
        p = Provenance(page=1)
        assert p.to_serialisable()["bbox"] is None


# ---------------------------------------------------------------------------
# FieldValue
# ---------------------------------------------------------------------------


class TestFieldValueWrapper:
    def test_construction(self) -> None:
        prov = Provenance(page=1)
        fv = FieldValue(value="Alice", provenance=prov)
        assert fv.value == "Alice"
        assert fv.provenance is prov

    def test_to_serialisable_has_underscore_provenance(self) -> None:
        fv = FieldValue(value="Alice", provenance=Provenance(page=1))
        d = fv.to_serialisable()
        assert d["value"] == "Alice"
        assert "_provenance" in d

    def test_round_trip_unwrap_value(self) -> None:
        fv = FieldValue(value=42, provenance=Provenance(page=1))
        assert unwrap_value(fv) == 42
        assert unwrap_value(fv.to_serialisable()) == 42

    def test_round_trip_unwrap_provenance(self) -> None:
        fv = FieldValue(value=42, provenance=Provenance(page=2, confidence=0.9))
        assert unwrap_provenance(fv).page == 2
        assert unwrap_provenance(fv.to_serialisable()).confidence == 0.9


# ---------------------------------------------------------------------------
# PHIFieldValue
# ---------------------------------------------------------------------------


class TestPHIFieldValue:
    def test_inherits_field_value_invariant(self) -> None:
        prov = Provenance(page=1)
        phi = PHIFieldValue(value="[REDACTED]", provenance=prov)
        assert isinstance(phi, FieldValue)
        assert unwrap_provenance(phi) is prov

    def test_redacted_default(self) -> None:
        phi = PHIFieldValue(value="[REDACTED]", provenance=Provenance(page=1))
        assert phi.redacted_value == "[REDACTED]"

    def test_carries_encrypted_value(self) -> None:
        phi = PHIFieldValue(
            value="[REDACTED]",
            provenance=Provenance(page=1),
            encrypted_value=b"ciphertext",
        )
        assert phi.encrypted_value == b"ciphertext"


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class TestUnwrap:
    def test_unwrap_value_handles_dict_form(self) -> None:
        d = {"value": "x", "_provenance": {"page": 1}}
        assert unwrap_value(d) == "x"

    def test_unwrap_value_handles_legacy_provenance_key(self) -> None:
        d = {"value": "x", "provenance": {"page": 1}}
        assert unwrap_value(d) == "x"

    def test_unwrap_value_passes_through_bare_scalar(self) -> None:
        assert unwrap_value(42) == 42
        assert unwrap_value("hello") == "hello"

    def test_unwrap_value_strict_raises_on_bare(self) -> None:
        with pytest.raises(ProvenanceMissingError):
            unwrap_value(42, strict=True)

    def test_unwrap_value_strict_allows_none(self) -> None:
        # Null is the documented exception to the invariant.
        assert unwrap_value(None, strict=True) is None

    def test_unwrap_provenance_returns_none_for_bare(self) -> None:
        assert unwrap_provenance(42) is None
        assert unwrap_provenance(None) is None

    def test_unwrap_provenance_returns_none_for_dict_without_keys(self) -> None:
        assert unwrap_provenance({"value": "x"}) is None  # no provenance key

    def test_unwrap_provenance_handles_invalid_dict_payload(self) -> None:
        # When the embedded provenance is mal-shaped, unwrap returns None
        # rather than raising — exporters fall back to the no-provenance
        # rendering path.
        assert (
            unwrap_provenance({"value": "x", "_provenance": "not-a-dict"}) is None
        )


class TestIsFieldValue:
    def test_pydantic_instance(self) -> None:
        fv = FieldValue(value=1, provenance=Provenance(page=1))
        assert is_field_value(fv) is True

    def test_serialised_dict_underscore_key(self) -> None:
        assert is_field_value({"value": 1, "_provenance": {}}) is True

    def test_serialised_dict_legacy_key(self) -> None:
        assert is_field_value({"value": 1, "provenance": {}}) is True

    def test_bare_scalar(self) -> None:
        assert is_field_value(42) is False
        assert is_field_value(None) is False
        assert is_field_value("x") is False

    def test_dict_without_value_key(self) -> None:
        assert is_field_value({"foo": "bar"}) is False


class TestWrapValue:
    def test_wrap_with_explicit_provenance(self) -> None:
        prov = Provenance(page=2)
        fv = wrap_value("Alice", provenance=prov)
        assert fv.value == "Alice"
        assert fv.provenance is prov

    def test_wrap_uses_legacy_sentinel_when_omitted(self) -> None:
        fv = wrap_value(42)
        assert fv.provenance.source_block_id == "legacy"
        assert fv.provenance.extraction_path == ["legacy"]


# ---------------------------------------------------------------------------
# Sentinels
# ---------------------------------------------------------------------------


class TestSentinels:
    def test_legacy_sentinel_well_formed(self) -> None:
        # The sentinel should serialise without error.
        d = LEGACY_SENTINEL_PROVENANCE.to_serialisable()
        assert d["source_block_id"] == "legacy"
        assert d["extraction_path"] == ["legacy"]

    def test_empty_provenance_well_formed(self) -> None:
        e = empty_provenance(stage="extraction_failed")
        assert e.extraction_path == ["extraction_failed"]
        assert e.confidence == 0.0
        assert e.bbox is None


# ---------------------------------------------------------------------------
# FieldMetadata.to_provenance bridge
# ---------------------------------------------------------------------------


class TestFieldMetadataToProvenance:
    def test_bridges_legacy_metadata(self) -> None:
        bbox = BoundingBoxCoords(x=0.1, y=0.2, width=0.3, height=0.04, page=2)
        fm = FieldMetadata(
            field_name="patient_name",
            value="Alice",
            confidence=0.9,
            source_page=2,
            bbox=bbox,
        )
        prov = fm.to_provenance(
            extraction_path=["pass1_vlm", "reconciler"],
            agent_signatures=["extractor"],
            vlm_model_id="qwen3.6@8001",
        )
        assert isinstance(prov, Provenance)
        assert prov.page == 2
        assert prov.confidence == 0.9
        assert prov.bbox.x == pytest.approx(0.1)
        assert prov.extraction_path == ["pass1_vlm", "reconciler"]
        assert "extractor" in prov.agent_signatures
        assert prov.vlm_model_id == "qwen3.6@8001"

    def test_no_bbox_metadata(self) -> None:
        fm = FieldMetadata(
            field_name="x", value="y", confidence=0.5, source_page=1
        )
        prov = fm.to_provenance()
        assert prov.bbox is None
