"""Phase K mode-selector wiring tests.

Verifies the Healthcare / General mode toggle reaches every layer:

* ``create_initial_state`` accepts ``profile_override`` + ``modality_override``
  and writes them into the resulting state dict.
* ``run_extraction_pipeline`` (graph.py) forwards both kwargs to
  ``PipelineRunner.extract_from_pdf``.
* ``ProcessRequest.profile_override`` validates the new pydantic field.
* CLI mode → profile mapping (``_MODE_TO_PROFILE``) is bug-for-bug stable.
* The ``MODE_LABELS`` constant on the frontend stays in sync with the
  backend's profile ids (validated indirectly via the mapping table).
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from src.pipeline.state import create_initial_state


class TestCreateInitialStateOverrides:
    """``create_initial_state`` honours Phase K override kwargs."""

    def test_profile_override_propagated(self) -> None:
        state = create_initial_state(
            pdf_path="/tmp/doc.pdf",
            profile_override="medical-rcm",
        )
        assert state["profile_override"] == "medical-rcm"
        # Modality default stays empty (auto-detect).
        assert state["modality_override"] == []

    def test_modality_override_copied_not_shared(self) -> None:
        sentinel = ["fax", "handwritten"]
        state = create_initial_state(
            pdf_path="/tmp/doc.pdf",
            modality_override=sentinel,
        )
        assert state["modality_override"] == ["fax", "handwritten"]
        # Caller's list must not be aliased into state.
        sentinel.clear()
        assert state["modality_override"] == ["fax", "handwritten"]

    def test_no_overrides_keeps_auto_detect_behaviour(self) -> None:
        state = create_initial_state(pdf_path="/tmp/doc.pdf")
        assert state["profile_override"] is None
        assert state["modality_override"] == []

    def test_empty_modality_list_treated_as_auto_detect(self) -> None:
        state = create_initial_state(
            pdf_path="/tmp/doc.pdf",
            modality_override=[],
        )
        assert state["modality_override"] == []


class TestRunExtractionPipelineForwarding:
    """``run_extraction_pipeline`` threads kwargs to ``extract_from_pdf``."""

    def test_kwargs_forwarded(self) -> None:
        from src.pipeline import graph

        captured: dict = {}

        class _FakeRunner:
            def __init__(self, **kwargs):  # noqa: D401 - simple stub
                pass

            def extract_from_pdf(self, **kwargs):
                captured.update(kwargs)
                return {"processing_id": "p1"}

        with patch.object(graph, "PipelineRunner", _FakeRunner):
            graph.run_extraction_pipeline(
                pdf_path="/tmp/doc.pdf",
                profile_override="medical-rcm",
                modality_override=["fax"],
            )

        assert captured["profile_override"] == "medical-rcm"
        assert captured["modality_override"] == ["fax"]

    def test_no_kwargs_passes_none(self) -> None:
        from src.pipeline import graph

        captured: dict = {}

        class _FakeRunner:
            def __init__(self, **kwargs):
                pass

            def extract_from_pdf(self, **kwargs):
                captured.update(kwargs)
                return {"processing_id": "p2"}

        with patch.object(graph, "PipelineRunner", _FakeRunner):
            graph.run_extraction_pipeline(pdf_path="/tmp/doc.pdf")

        assert captured["profile_override"] is None
        assert captured["modality_override"] is None


class TestProcessRequestProfileOverride:
    """``ProcessRequest`` accepts and validates the new field."""

    def test_default_is_none(self) -> None:
        from src.api.models import ProcessRequest

        req = ProcessRequest(pdf_path="claim.pdf")
        assert req.profile_override is None

    def test_explicit_profile_accepted(self) -> None:
        from src.api.models import ProcessRequest

        req = ProcessRequest(
            pdf_path="claim.pdf",
            profile_override="medical-rcm",
        )
        assert req.profile_override == "medical-rcm"

    def test_long_profile_id_rejected(self) -> None:
        from src.api.models import ProcessRequest

        with pytest.raises(Exception):  # noqa: BLE001 - pydantic ValidationError
            ProcessRequest(
                pdf_path="claim.pdf",
                profile_override="a" * 100,  # exceeds max_length=64
            )


class TestCLIModeMapping:
    """``main._MODE_TO_PROFILE`` is the bridge from CLI args to profile ids."""

    def test_healthcare_maps_to_medical_rcm(self) -> None:
        from main import _MODE_TO_PROFILE

        assert _MODE_TO_PROFILE["healthcare"] == "medical-rcm"

    def test_general_maps_to_generic_document(self) -> None:
        from main import _MODE_TO_PROFILE

        assert _MODE_TO_PROFILE["general"] == "generic-document"

    def test_auto_is_not_in_mapping(self) -> None:
        """``auto`` intentionally has no entry — resolved-profile is None."""
        from main import _MODE_TO_PROFILE

        assert "auto" not in _MODE_TO_PROFILE
