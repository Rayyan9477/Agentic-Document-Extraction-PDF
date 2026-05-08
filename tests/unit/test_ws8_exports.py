"""WS-8: tests for the new export surfaces.

Covers:
    * Bbox overlay rendering — renders into a tmp dir without
      hitting OpenCV, asserts colour bands by confidence band, and
      confirms pages without bboxes don't produce empty PNGs.
    * DataFrame-flat JSON export — verifies the row shape so callers
      can drop the result straight into ``pandas``.
    * FHIR R4 exporter — exercises the dict-fallback path (no
      ``fhir.resources`` dependency required) and validates the
      expected Bundle structure.
    * Markdown decision trail — confirms the new section appears in
      detailed style when the relevant state is present.
"""

from __future__ import annotations

import io

import pytest
from PIL import Image

from src.export.bbox_overlay import (
    _COLOR_HIGH,
    _COLOR_LOW,
    _COLOR_MEDIUM,
    _COLOR_UNKNOWN,
    OverlayResult,
    _confidence_color,
    render_overlays,
)
from src.export.fhir_exporter import FHIRBundle, export_fhir
from src.export.json_exporter import (
    ExportFormat,
    JSONExportConfig,
    JSONExporter,
)
from src.pipeline.state import BoundingBoxCoords


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_pil_page(page_number: int, width: int = 200, height: int = 280) -> dict:
    """Build a serialised PageImage dict the overlay renderer accepts."""
    img = Image.new("RGB", (width, height), color=(240, 240, 240))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return {
        "page_number": page_number,
        "width": width,
        "height": height,
        "image_bytes": buf.getvalue(),
    }


def _make_bbox(x=0.1, y=0.1, w=0.3, h=0.05, page=1) -> BoundingBoxCoords:
    return BoundingBoxCoords(
        x=x, y=y, width=w, height=h, page=page,
        pixel_x=0, pixel_y=0, pixel_width=0, pixel_height=0,
    )


# ---------------------------------------------------------------------------
# Bbox overlays
# ---------------------------------------------------------------------------


class TestConfidencePalette:
    @pytest.mark.parametrize(
        "score,expected",
        [
            (0.95, _COLOR_HIGH),
            (0.85, _COLOR_HIGH),
            (0.84999, _COLOR_MEDIUM),
            (0.50, _COLOR_MEDIUM),
            (0.49999, _COLOR_LOW),
            (0.0, _COLOR_LOW),
            (None, _COLOR_UNKNOWN),
        ],
    )
    def test_confidence_band_to_color(self, score, expected) -> None:
        assert _confidence_color(score) == expected


class TestRenderOverlays:
    def test_renders_only_pages_with_bboxes(self, tmp_path) -> None:
        # Two pages but only page 1 has a tagged field.
        pages = [_make_pil_page(1), _make_pil_page(2)]
        field_metadata = {
            "patient_name": {
                "confidence": 0.95,
                "bbox": _make_bbox(page=1).to_dict(),
            },
        }
        result = render_overlays(pages, field_metadata, tmp_path)
        assert isinstance(result, OverlayResult)
        assert len(result.pages) == 1
        assert result.pages[0].page_number == 1
        assert result.pages[0].field_count == 1
        # Only page 1's PNG is on disk; page 2 is correctly skipped.
        assert result.pages[0].output_path.exists()
        assert not (tmp_path / "overlays" / "page_02.png").exists()

    def test_field_without_bbox_is_skipped(self, tmp_path) -> None:
        pages = [_make_pil_page(1)]
        field_metadata = {
            "no_bbox_field": {"confidence": 0.91},  # missing 'bbox' key
        }
        result = render_overlays(pages, field_metadata, tmp_path)
        # No bbox-tagged fields → no overlay PNGs.
        assert result.pages == []
        assert result.total_fields == 0

    def test_overlay_png_is_valid_image(self, tmp_path) -> None:
        pages = [_make_pil_page(1, width=300, height=400)]
        field_metadata = {
            "x": {"confidence": 0.6, "bbox": _make_bbox(0.2, 0.2, 0.4, 0.1).to_dict()},
        }
        result = render_overlays(pages, field_metadata, tmp_path)
        png_path = result.pages[0].output_path
        # Re-open the rendered PNG to confirm it's a valid image of
        # the expected dimensions.
        with Image.open(png_path) as out:
            assert out.format == "PNG"
            assert out.size == (300, 400)

    def test_total_fields_aggregates_across_pages(self, tmp_path) -> None:
        pages = [_make_pil_page(1), _make_pil_page(2)]
        field_metadata = {
            "f1": {"confidence": 0.9, "bbox": _make_bbox(page=1).to_dict()},
            "f2": {"confidence": 0.7, "bbox": _make_bbox(page=1, x=0.5).to_dict()},
            "f3": {"confidence": 0.4, "bbox": _make_bbox(page=2).to_dict()},
        }
        result = render_overlays(pages, field_metadata, tmp_path)
        assert result.total_fields == 3


# ---------------------------------------------------------------------------
# DataFrame-flat JSON export
# ---------------------------------------------------------------------------


class TestDataFrameFlatExport:
    def test_one_row_per_field_with_bbox_columns(self) -> None:
        state = {
            "processing_id": "rec-42",
            "merged_extraction": {
                "patient_name": {"value": "John Doe", "confidence": 0.91},
                "amount": {"value": 250.0, "confidence": 0.83},
            },
            "field_metadata": {
                "patient_name": {
                    "confidence": 0.91,
                    "bbox": {
                        "x": 0.1, "y": 0.2, "width": 0.3, "height": 0.04, "page": 1,
                    },
                },
                "amount": {"confidence": 0.83},
            },
        }
        exporter = JSONExporter(JSONExportConfig(format=ExportFormat.DATAFRAME_FLAT))
        result = exporter.export(state)

        assert result["format"] == "dataframe_flat"
        assert result["row_count"] == 2

        rows = result["rows"]
        rows_by_field = {r["field"]: r for r in rows}

        # Bbox-tagged field has full coordinates.
        pn = rows_by_field["patient_name"]
        assert pn["record_id"] == "rec-42"
        assert pn["value"] == "John Doe"
        assert pn["confidence"] == 0.91
        assert pn["page"] == 1
        assert pn["bbox_x"] == 0.1
        assert pn["bbox_w"] == 0.3

        # Field without bbox metadata still gets the columns (None values).
        amt = rows_by_field["amount"]
        assert amt["bbox_x"] is None
        assert amt["page"] is None

    def test_envelope_fields_unwrapped(self) -> None:
        state = {
            "processing_id": "rec-1",
            "merged_extraction": {
                "f": {"value": "v", "confidence": 0.5, "human_corrected": True},
            },
            "field_metadata": {},
        }
        exporter = JSONExporter(JSONExportConfig(format=ExportFormat.DATAFRAME_FLAT))
        result = exporter.export(state)
        row = result["rows"][0]
        assert row["value"] == "v"
        assert row["human_corrected"] is True

    def test_phi_redacted_fields_get_redacted_value_column(self) -> None:
        state = {
            "processing_id": "rec-1",
            "merged_extraction": {
                "patient_name": "[REDACTED]",
                "amount": 250.0,
            },
            "field_metadata": {},
            "phi_redacted_fields": ["patient_name"],
        }
        exporter = JSONExporter(JSONExportConfig(format=ExportFormat.DATAFRAME_FLAT))
        rows = exporter.export(state)["rows"]
        rows_by_field = {r["field"]: r for r in rows}
        # Redacted field has the dedicated column …
        assert "redacted_value" in rows_by_field["patient_name"]
        # … and the non-redacted field doesn't.
        assert "redacted_value" not in rows_by_field["amount"]


# ---------------------------------------------------------------------------
# FHIR R4 exporter
# ---------------------------------------------------------------------------


class TestFhirExporter:
    def test_cms1500_emits_patient_coverage_claim_bundle(self) -> None:
        record = {
            "patient_name": "Doe, John",
            "patient_dob": "01/15/1980",
            "patient_gender": "M",
            "member_id": "M-1234",
            "insurance_company": "Acme Health",
            "claim_number": "CL-9000",
            "service_date": "03/12/2024",
            "total_charges": "$1,234.56",
        }
        bundle = export_fhir(record, document_type="cms1500", processing_id="proc-cms")
        assert isinstance(bundle, FHIRBundle)
        assert bundle.bundle["resourceType"] == "Bundle"
        assert bundle.bundle["type"] == "collection"
        assert bundle.bundle["id"] == "proc-cms"

        types = [e["resource"]["resourceType"] for e in bundle.bundle["entry"]]
        assert "Patient" in types
        assert "Coverage" in types
        assert "Claim" in types

        # Patient name decomposed correctly.
        patient = next(e["resource"] for e in bundle.bundle["entry"]
                       if e["resource"]["resourceType"] == "Patient")
        name = patient["name"][0]
        assert name["family"] == "Doe"
        assert name["given"] == ["John"]
        assert patient["birthDate"] == "1980-01-15"
        assert patient["gender"] == "male"

        # Claim references the patient + coverage.
        claim = next(e["resource"] for e in bundle.bundle["entry"]
                     if e["resource"]["resourceType"] == "Claim")
        assert claim["type"]["coding"][0]["code"] == "professional"
        assert claim["total"]["value"] == 1234.56
        assert claim["total"]["currency"] == "USD"
        assert claim["insurance"][0]["focal"] is True

    def test_ub04_uses_institutional_claim_type(self) -> None:
        record = {
            "patient_name": "Smith, Jane",
            "member_id": "M-1",
            "claim_number": "C-1",
        }
        bundle = export_fhir(record, document_type="ub04")
        claim = next(e["resource"] for e in bundle.bundle["entry"]
                     if e["resource"]["resourceType"] == "Claim")
        assert claim["type"]["coding"][0]["code"] == "institutional"

    def test_eob_emits_patient_and_explanation_of_benefit(self) -> None:
        record = {
            "patient_name": "John Doe",
            "amount_paid": 100.0,
            "total_charges": 250.0,
            "statement_date": "2024-03-12",
        }
        bundle = export_fhir(record, document_type="eob")
        types = {e["resource"]["resourceType"] for e in bundle.bundle["entry"]}
        assert "Patient" in types
        assert "ExplanationOfBenefit" in types

        eob = next(e["resource"] for e in bundle.bundle["entry"]
                   if e["resource"]["resourceType"] == "ExplanationOfBenefit")
        assert eob["payment"]["amount"]["value"] == 100.0
        assert eob["created"] == "2024-03-12"

    def test_unknown_doc_type_falls_back_to_document_reference(self) -> None:
        bundle = export_fhir({"foo": "bar"}, document_type="unknown_schema")
        types = {e["resource"]["resourceType"] for e in bundle.bundle["entry"]}
        assert "DocumentReference" in types

    def test_value_envelope_unwrapped_before_mapping(self) -> None:
        record = {
            "patient_name": {"value": "Jane Doe", "confidence": 0.9},
            "member_id": {"value": "M-2", "confidence": 0.8},
            "claim_number": "C-9",
        }
        bundle = export_fhir(record, document_type="cms1500")
        patient = next(e["resource"] for e in bundle.bundle["entry"]
                       if e["resource"]["resourceType"] == "Patient")
        # Confirm the envelope was stripped, not embedded as-is.
        assert isinstance(patient["name"][0].get("family"), str)


# ---------------------------------------------------------------------------
# Markdown decision trail
# ---------------------------------------------------------------------------


class TestMarkdownDecisionTrail:
    def test_decision_trail_appears_in_detailed_when_state_populated(self) -> None:
        from src.export.markdown_exporter import (
            MarkdownExportConfig,
            MarkdownExporter,
            MarkdownStyle,
        )

        state = {
            "processing_id": "p-1",
            "document_type": "cms1500",
            "status": "completed",
            "overall_confidence": 0.92,
            "retry_count": 1,
            "modalities": ["printed", "fax"],
            "phi_redacted_fields": ["patient_name", "ssn"],
            "human_corrections": {"diagnosis_code": "E11.9"},
            "merged_extraction": {},
            "field_metadata": {},
        }
        exporter = MarkdownExporter(
            MarkdownExportConfig(style=MarkdownStyle.DETAILED, include_audit_trail=True)
        )
        report = exporter.export(state)

        assert "## Decision Trail" in report
        assert "Final status" in report
        assert "`completed`" in report
        # Modalities list
        assert "`fax`" in report
        # PHI redaction summary
        assert "PHI redaction" in report
        assert "`patient_name`" in report
        # Reviewer corrections
        assert "Reviewer corrections" in report
        assert "`diagnosis_code`" in report

    def test_decision_trail_omitted_when_no_relevant_state(self) -> None:
        from src.export.markdown_exporter import (
            MarkdownExportConfig,
            MarkdownExporter,
            MarkdownStyle,
        )

        state = {
            "processing_id": "p-empty",
            "document_type": "cms1500",
            "merged_extraction": {},
            "field_metadata": {},
        }
        exporter = MarkdownExporter(
            MarkdownExportConfig(style=MarkdownStyle.DETAILED, include_audit_trail=True)
        )
        report = exporter.export(state)
        # Empty state → no Decision Trail heading at all.
        assert "## Decision Trail" not in report
