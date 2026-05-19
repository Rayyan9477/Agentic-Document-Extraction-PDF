"""Phase K — ``export_fhir_bundle`` wrapper tests.

Verifies the profile-gating logic and the file-output behaviour of the
new helper at ``src.export.consolidated_export.export_fhir_bundle``.
The underlying ``export_fhir`` is already tested in
``tests/unit/test_ws8_exports.py``; here we only exercise the wrapper.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from src.export.consolidated_export import export_fhir_bundle
from src.extraction.multi_record import DocumentExtractionResult, ExtractedRecord


def _make_result(
    *,
    document_type: str,
    records: list[dict] | None = None,
) -> DocumentExtractionResult:
    """Minimal-shape DocumentExtractionResult for unit tests."""
    return DocumentExtractionResult(
        pdf_path="/tmp/claim.pdf",
        document_type=document_type,
        entity_type="patient",
        records=[
            ExtractedRecord(
                record_id=i,
                page_number=1,
                primary_identifier=f"Patient {i}",
                entity_type="patient",
                fields=fields,
                confidence=0.9,
                extraction_time_ms=100,
            )
            for i, fields in enumerate(records or [], start=1)
        ],
        total_pages=1,
        total_records=len(records or []),
        total_vlm_calls=1,
        total_processing_time_ms=100,
        schema={},
    )


class TestProfileGate:
    """The wrapper short-circuits for non-medical profiles."""

    def test_general_profile_short_circuits(self, tmp_path: Path) -> None:
        result = _make_result(
            document_type="cms1500",
            records=[{"patient_name": "Jane Doe"}],
        )
        out = tmp_path / "x.fhir.json"
        bundle = export_fhir_bundle(result, out, profile="generic-document")
        assert bundle is None
        assert not out.exists()

    def test_medical_profile_emits(self, tmp_path: Path) -> None:
        result = _make_result(
            document_type="cms1500",
            records=[{"patient_name": "Jane Doe"}],
        )
        out = tmp_path / "x.fhir.json"
        bundle = export_fhir_bundle(result, out, profile="medical-rcm")
        assert bundle is not None
        assert out.exists()
        assert bundle["resourceType"] == "Bundle"

    def test_auto_profile_infers_from_document_type(self, tmp_path: Path) -> None:
        """``profile=None`` should still emit when document_type is medical."""
        result = _make_result(
            document_type="ub04",
            records=[{"patient_name": "John Smith"}],
        )
        out = tmp_path / "x.fhir.json"
        bundle = export_fhir_bundle(result, out, profile=None)
        assert bundle is not None
        assert bundle["resourceType"] == "Bundle"

    def test_auto_profile_skips_non_medical_document_type(
        self, tmp_path: Path
    ) -> None:
        """A non-medical document_type with profile=None must not emit FHIR."""
        result = _make_result(
            document_type="invoice",
            records=[{"vendor_name": "Acme Corp"}],
        )
        out = tmp_path / "x.fhir.json"
        bundle = export_fhir_bundle(result, out, profile=None)
        assert bundle is None
        assert not out.exists()


class TestEmptyState:
    """Empty-state behaviour."""

    def test_no_records_returns_none(self, tmp_path: Path) -> None:
        result = _make_result(document_type="cms1500", records=[])
        out = tmp_path / "x.fhir.json"
        bundle = export_fhir_bundle(result, out, profile="medical-rcm")
        assert bundle is None
        assert not out.exists()


class TestOutputShape:
    """The written file matches the returned bundle dict."""

    def test_file_content_matches_return_value(self, tmp_path: Path) -> None:
        result = _make_result(
            document_type="cms1500",
            records=[{"patient_name": "Jane Doe"}],
        )
        out = tmp_path / "claim.fhir.json"
        bundle = export_fhir_bundle(result, out, profile="medical-rcm")
        on_disk = json.loads(out.read_text(encoding="utf-8"))
        assert on_disk == bundle


class TestPHIMasking:
    """``mask_phi=True`` masks before FHIR construction."""

    def test_mask_phi_redacts_patient_name_in_bundle(
        self, tmp_path: Path
    ) -> None:
        result = _make_result(
            document_type="cms1500",
            records=[{"patient_name": "Jane Doe"}],
        )
        out = tmp_path / "x.fhir.json"
        bundle = export_fhir_bundle(
            result, out, profile="medical-rcm", mask_phi=True
        )
        # The bundle is built from masked fields, so the name should
        # not appear anywhere in serialised form.
        as_text = json.dumps(bundle)
        assert "Jane Doe" not in as_text
