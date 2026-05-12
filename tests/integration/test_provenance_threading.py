"""
Phase 4 — provenance threading integration tests.

Verifies the end-to-end invariant: every leaf value emerging from the
dual-VLM reconciler carries a populated ``Provenance`` AND the
existing ``merged_extraction`` shape coexists (default flag) OR is
empty (enforce flag).

Coverage:

* Default flag: reconciler dual-writes both shapes.
* Enforce flag: reconciler writes ONLY merged_extraction_v2.
* Provenance carries page, bbox, extraction_path, agent_signatures,
  vlm_model_id when populated.
* JSON exporter surfaces the provenance block at top-level + per-leaf.
* Markdown DETAILED export includes provenance footnotes.
* DataFrame-flat export adds the new columns.
* FHIR exporter accepts and emits provenance_map.

No live VLMs. The reconciler closure runs through orchestrator-built
state; pass1/pass2 results are pre-populated dicts.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from src.client.backends.factory import reset_cache


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    from src.config.settings import get_settings

    reset_cache()
    get_settings.cache_clear()  # type: ignore[attr-defined]
    yield
    get_settings.cache_clear()  # type: ignore[attr-defined]
    reset_cache()


@pytest.fixture
def dual_vlm_default(monkeypatch):
    """Dual-VLM engine with default ``PROVENANCE_ENFORCE_FIELD_VALUE_WRAPPER=False``."""
    monkeypatch.setenv("EXTRACTION_ENGINE", "dual_vlm")
    monkeypatch.setenv("PROVENANCE_ENFORCE_FIELD_VALUE_WRAPPER", "false")


@pytest.fixture
def dual_vlm_enforce(monkeypatch):
    """Dual-VLM engine with the wrapper-enforce flag flipped on."""
    monkeypatch.setenv("EXTRACTION_ENGINE", "dual_vlm")
    monkeypatch.setenv("PROVENANCE_ENFORCE_FIELD_VALUE_WRAPPER", "true")


def _identity(state: dict[str, Any]) -> dict[str, Any]:
    return state


def _state_with_passes() -> dict[str, Any]:
    """Build an ExtractionState with Pass 1 + Pass 2 ready for reconciliation."""
    return {
        "processing_id": "p-prov-test",
        "pdf_path": "/tmp/x.pdf",
        "page_images": [
            {
                "page_number": 1,
                "data_uri": "data:image/png;base64,IM",
            }
        ],
        "document_type": "CMS-1500",
        "selected_schema_name": "cms_1500",
        "modalities": [],
        "errors": [],
        "warnings": [],
        "pass1_model_id": "qwen3.6-27b-vl@8001",
        "pass2_model_id": "gemma-4-31b-vl@8002",
        "pass1_result": {
            1: {
                "fields": {
                    "patient_name": {
                        "value": "Alice Smith",
                        "confidence": 0.92,
                    },
                    "service_date": {
                        "value": "2024-01-15",
                        "confidence": 0.88,
                    },
                }
            }
        },
        "pass2_result": {
            1: {
                "fields": {
                    "patient_name": {
                        "value": "Alice Smith",
                        "confidence": 0.90,
                        "bbox": [0.10, 0.05, 0.32, 0.09],
                    },
                    "service_date": {
                        "value": "2024-01-15",
                        "confidence": 0.86,
                        "bbox": [0.50, 0.05, 0.62, 0.09],
                    },
                }
            }
        },
    }


def _build_reconcile(monkeypatch_fixture):
    """Build the orchestrator + reconciler closure under the active flag."""
    from src.agents.orchestrator import create_extraction_workflow

    orch, _ = create_extraction_workflow(
        preprocess_fn=_identity,
        enable_checkpointing=False,
        enable_vlm_first=False,
        enable_splitter=False,
        enable_table_detection=False,
    )
    return orch


# ---------------------------------------------------------------------------
# Dual-write behavior
# ---------------------------------------------------------------------------


class TestReconcilerDualWrite:
    def test_default_flag_writes_both_shapes(self, dual_vlm_default) -> None:
        orch = _build_reconcile(dual_vlm_default)
        out = orch._reconciler_node(_state_with_passes())  # type: ignore[attr-defined]
        # Legacy shape populated.
        assert "merged_extraction" in out
        assert out["merged_extraction"]["patient_name"] == "Alice Smith"
        # V2 shape populated.
        assert "merged_extraction_v2" in out
        assert "patient_name" in out["merged_extraction_v2"]
        wrapper = out["merged_extraction_v2"]["patient_name"]
        assert wrapper["value"] == "Alice Smith"
        assert "_provenance" in wrapper

    def test_enforce_flag_empties_legacy_shape(self, dual_vlm_enforce) -> None:
        orch = _build_reconcile(dual_vlm_enforce)
        out = orch._reconciler_node(_state_with_passes())  # type: ignore[attr-defined]
        # Legacy shape now empty.
        assert out["merged_extraction"] == {}
        # V2 shape still populated.
        assert out["merged_extraction_v2"]["patient_name"]["value"] == "Alice Smith"

    def test_provenance_carries_lineage(self, dual_vlm_default) -> None:
        orch = _build_reconcile(dual_vlm_default)
        out = orch._reconciler_node(_state_with_passes())  # type: ignore[attr-defined]
        wrapper = out["merged_extraction_v2"]["patient_name"]
        prov = wrapper["_provenance"]
        # Both passes agreed → extraction_path includes both.
        assert "pass1_vlm" in prov["extraction_path"]
        assert "pass2_vlm" in prov["extraction_path"]
        assert "reconciler" in prov["extraction_path"]
        # Bbox came from Pass 2.
        assert prov["bbox"] is not None
        # vlm_model_id pinned to one of the configured passes.
        assert prov["vlm_model_id"] in (
            "qwen3.6-27b-vl@8001",
            "gemma-4-31b-vl@8002",
        )
        # Reconciliation metadata records the agreement.
        assert out["reconciliation_metadata"]["agreement_rate"] == pytest.approx(1.0)

    def test_provenance_index_keyed_by_field(self, dual_vlm_default) -> None:
        orch = _build_reconcile(dual_vlm_default)
        out = orch._reconciler_node(_state_with_passes())  # type: ignore[attr-defined]
        index = out["provenance_index"]
        assert "patient_name" in index
        assert "service_date" in index
        assert "reconciler" in index["patient_name"]


# ---------------------------------------------------------------------------
# JSON exporter
# ---------------------------------------------------------------------------


class TestJSONExporterProvenance:
    def _state_with_v2(self) -> dict[str, Any]:
        from src.pipeline.provenance import Provenance, wrap_value

        prov = Provenance(
            page=1,
            extraction_path=["pass1_vlm", "reconciler"],
            agent_signatures=["extractor"],
            confidence=0.9,
            vlm_model_id="qwen@8001",
        )
        fv = wrap_value("Alice", provenance=prov)
        return {
            "processing_id": "p1",
            "document_type": "CMS-1500",
            "merged_extraction": {"patient_name": "Alice"},
            "merged_extraction_v2": {
                "patient_name": fv.to_serialisable(),
            },
        }

    def test_standard_export_includes_provenance_block(self) -> None:
        from src.export.json_exporter import (
            ExportFormat,
            JSONExportConfig,
            JSONExporter,
        )

        exp = JSONExporter(JSONExportConfig(format=ExportFormat.STANDARD))
        out = exp.export(self._state_with_v2())
        assert "provenance" in out
        assert out["provenance"]["patient_name"]["page"] == 1
        # Data field still has the unwrapped scalar.
        assert out["data"]["patient_name"] == "Alice"

    def test_minimal_export_omits_provenance(self) -> None:
        from src.export.json_exporter import (
            ExportFormat,
            JSONExportConfig,
            JSONExporter,
        )

        exp = JSONExporter(JSONExportConfig(format=ExportFormat.MINIMAL))
        out = exp.export(self._state_with_v2())
        assert "provenance" not in out

    def test_dataframe_flat_includes_provenance_columns(self) -> None:
        from src.export.json_exporter import (
            ExportFormat,
            JSONExportConfig,
            JSONExporter,
        )

        exp = JSONExporter(JSONExportConfig(format=ExportFormat.DATAFRAME_FLAT))
        out = exp.export(self._state_with_v2())
        assert out["row_count"] == 1
        row = out["rows"][0]
        assert row["field"] == "patient_name"
        assert row["extraction_path"] == "pass1_vlm,reconciler"
        assert row["agent_signatures"] == "extractor"
        assert row["vlm_model_id"] == "qwen@8001"
        assert "source_block_id" in row


# ---------------------------------------------------------------------------
# Markdown exporter
# ---------------------------------------------------------------------------


class TestMarkdownProvenanceFootnotes:
    def _state_with_v2(self) -> dict[str, Any]:
        from src.pipeline.provenance import Provenance, wrap_value
        from src.pipeline.state import BoundingBoxCoords

        bbox = BoundingBoxCoords(x=0.1, y=0.05, width=0.2, height=0.04, page=1)
        prov = Provenance(
            page=1,
            bbox=bbox,
            extraction_path=["pass1_vlm", "reconciler"],
            agent_signatures=["extractor", "reconciler"],
            confidence=0.94,
        )
        fv = wrap_value("Alice", provenance=prov)
        return {
            "processing_id": "p1",
            "document_type": "CMS-1500",
            "merged_extraction": {"patient_name": "Alice"},
            "merged_extraction_v2": {"patient_name": fv.to_serialisable()},
            "field_metadata": {},
        }

    def test_detailed_style_renders_footnotes(self) -> None:
        from src.export.markdown_exporter import (
            MarkdownExportConfig,
            MarkdownExporter,
            MarkdownStyle,
        )

        exp = MarkdownExporter(
            MarkdownExportConfig(style=MarkdownStyle.DETAILED)
        )
        out = exp._format_extracted_data_detailed(self._state_with_v2())
        # Footnote marker + footnote body present.
        assert "<sup>1</sup>" in out
        assert "p.1" in out
        assert "pass1_vlm" in out
        assert "reconciler" in out

    def test_explicit_disable(self) -> None:
        from src.export.markdown_exporter import (
            MarkdownExportConfig,
            MarkdownExporter,
            MarkdownStyle,
        )

        exp = MarkdownExporter(
            MarkdownExportConfig(
                style=MarkdownStyle.DETAILED,
                include_provenance_footnotes=False,
            )
        )
        out = exp._format_extracted_data_detailed(self._state_with_v2())
        assert "<sup>1</sup>" not in out


# ---------------------------------------------------------------------------
# FHIR exporter
# ---------------------------------------------------------------------------


class TestFHIRProvenanceExtension:
    def test_provenance_map_attached_to_bundle(self) -> None:
        from src.export.fhir_exporter import export_fhir

        record = {"patient_name": "Alice"}
        prov_map = {
            "patient_name": {
                "page": 1,
                "bbox": {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.04},
                "extraction_path": ["pass1_vlm", "reconciler"],
                "agent_signatures": ["extractor"],
                "confidence": 0.9,
                "vlm_model_id": "qwen@8001",
                "source_block_id": "blk_p1_3",
                "mem0_match": None,
            }
        }
        bundle = export_fhir(
            record,
            document_type="cms1500",
            provenance_map=prov_map,
        )
        meta = bundle.bundle.get("meta")
        assert meta is not None
        ext = meta["extension"]
        assert len(ext) == 1
        assert ext[0]["url"] == "urn:veridoc:provenance:1.0"
        sub = {x["url"]: x for x in ext[0]["extension"]}
        assert sub["fieldName"]["valueString"] == "patient_name"
        assert sub["page"]["valueInteger"] == 1
        assert sub["confidence"]["valueDecimal"] == pytest.approx(0.9)
        assert sub["bbox"]["valueString"] == "0.1000,0.2000,0.3000,0.0400"

    def test_no_provenance_map_omits_meta(self) -> None:
        from src.export.fhir_exporter import export_fhir

        bundle = export_fhir({"patient_name": "Alice"}, document_type="cms1500")
        assert "meta" not in bundle.bundle


# ---------------------------------------------------------------------------
# API endpoints
# ---------------------------------------------------------------------------


class TestProvenanceAPI:
    def test_pages_endpoint_returns_404_without_orchestrator(self) -> None:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from src.api.routes.documents import router

        app = FastAPI()
        app.include_router(router, prefix="/api/v1")
        # No app.state.orchestrator — endpoint returns 404.
        client = TestClient(app)
        # Use a 16+ char processing_id to satisfy the validator.
        resp = client.get("/api/v1/documents/aaaa1111bbbb2222/pages/1")
        assert resp.status_code == 404

    def test_provenance_endpoint_returns_404_without_orchestrator(self) -> None:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from src.api.routes.documents import router

        app = FastAPI()
        app.include_router(router, prefix="/api/v1")
        client = TestClient(app)
        resp = client.get("/api/v1/documents/aaaa1111bbbb2222/provenance")
        assert resp.status_code == 404

    def test_pages_endpoint_serves_png_when_state_present(self) -> None:
        import base64

        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from src.api.routes.documents import router

        # Tiny 1×1 transparent PNG.
        png_bytes = base64.b64decode(
            "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mNkYAAAAAYAAjCB0C8AAAAASUVORK5CYII="
        )
        data_uri = f"data:image/png;base64,{base64.b64encode(png_bytes).decode()}"

        fake_orch = MagicMock()
        fake_orch.get_checkpoint_state.return_value = {
            "processing_id": "aaaa1111bbbb2222",
            "page_images": [
                {"page_number": 1, "data_uri": data_uri},
            ],
        }

        app = FastAPI()
        app.state.orchestrator = fake_orch
        app.include_router(router, prefix="/api/v1")
        client = TestClient(app)
        resp = client.get("/api/v1/documents/aaaa1111bbbb2222/pages/1")
        assert resp.status_code == 200
        assert resp.headers["content-type"] == "image/png"
        assert resp.content == png_bytes

    def test_provenance_endpoint_returns_serialised_map(self) -> None:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from src.api.routes.documents import router
        from src.pipeline.provenance import Provenance, wrap_value

        prov = Provenance(
            page=1,
            extraction_path=["pass1_vlm"],
            confidence=0.85,
            vlm_model_id="qwen@8001",
        )
        fv = wrap_value("Alice", provenance=prov)

        fake_orch = MagicMock()
        fake_orch.get_checkpoint_state.return_value = {
            "processing_id": "aaaa1111bbbb2222",
            "extraction_engine": "dual_vlm",
            "merged_extraction_v2": {
                "patient_name": fv.to_serialisable(),
            },
        }

        app = FastAPI()
        app.state.orchestrator = fake_orch
        app.include_router(router, prefix="/api/v1")
        client = TestClient(app)
        resp = client.get("/api/v1/documents/aaaa1111bbbb2222/provenance")
        assert resp.status_code == 200
        body = resp.json()
        assert body["processing_id"] == "aaaa1111bbbb2222"
        assert body["engine"] == "dual_vlm"
        assert body["field_count"] == 1
        assert body["fields"]["patient_name"]["page"] == 1

    def test_invalid_processing_id_returns_400(self) -> None:
        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from src.api.routes.documents import router

        fake_orch = MagicMock()
        app = FastAPI()
        app.state.orchestrator = fake_orch
        app.include_router(router, prefix="/api/v1")
        client = TestClient(app)
        # 16+ chars but with an invalid char — FastAPI's path constraint
        # rejects with 422 before our explicit 400.
        resp = client.get("/api/v1/documents/aaaa1111bbbb2222!/provenance")
        assert resp.status_code in (400, 422)
