"""
Phase 2 — ExtractorPass1Agent + ExtractorPass2Agent unit tests.

Both agents share the same skeleton (load page images, call
``send_vision_request_with_schema`` per page, stash results in state).
We test the per-pass invariants:

* Pass 1 writes ``pass1_result`` keyed by page number, populates
  ``pass1_model_id``, ``pass1_latency_ms``, sets ``extraction_engine``.
* Pass 2 writes ``pass2_result`` and the AUDITOR-side bbox normaliser
  rejects ``(value=null, bbox=[...])`` pairs (bbox hallucination).
* Both agents handle empty pages gracefully (skip + warn, no crash).
* Both agents tolerate per-page VLM failures (stash error, continue).

No live VLMs — we patch ``send_vision_request_with_schema`` on the
agent instance to return canned ``(payload, trace)`` tuples.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.agents.extractor_pass1 import ExtractorPass1Agent
from src.agents.extractor_pass2 import (
    ExtractorPass2Agent,
    Pass2AuditorEnvelope,
)
from src.client.backends.protocol import VLMRole
from src.client.constrained import DecodingTrace
from src.client.lm_client import LMStudioClient


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_state(num_pages: int = 1) -> dict[str, Any]:
    page_images = [
        {
            "page_number": i + 1,
            "data_uri": f"data:image/png;base64,PAGE{i}",
            "width": 1200,
            "height": 1600,
        }
        for i in range(num_pages)
    ]
    return {
        "processing_id": "proc-test",
        "pdf_path": "/tmp/test.pdf",
        "page_images": page_images,
        "document_type": "CMS-1500",
        "selected_schema_name": "cms_1500",
        "modalities": [],
        "errors": [],
        "warnings": [],
    }


def _trace(model_id: str = "stub-model") -> DecodingTrace:
    return DecodingTrace(
        backend_name="stub",
        role=VLMRole.PRIMARY,
        model_id=model_id,
        schema_name="JSONObjectEnvelope",
        latency_ms=12,
        tokens_in=10,
        tokens_out=10,
        schema_enforced=True,
    )


def _mock_client() -> MagicMock:
    return MagicMock(spec_set=LMStudioClient)


# ---------------------------------------------------------------------------
# ExtractorPass1Agent
# ---------------------------------------------------------------------------


class TestExtractorPass1Agent:
    def test_writes_pass1_result_keyed_by_page(self) -> None:
        agent = ExtractorPass1Agent(client=_mock_client())
        agent.send_vision_request_with_schema = MagicMock(
            return_value=({"fields": {"a": "1"}}, _trace())
        )
        state = _make_state(num_pages=3)
        out = agent.process(state)

        assert isinstance(out["pass1_result"], dict)
        assert set(out["pass1_result"].keys()) == {1, 2, 3}
        for v in out["pass1_result"].values():
            assert v["fields"]["a"] == "1"
        assert out["pass1_model_id"] == "stub-model"
        assert out["pass1_latency_ms"] >= 0
        assert out["extraction_engine"] == "dual_vlm"

    def test_skips_blank_pages(self) -> None:
        agent = ExtractorPass1Agent(client=_mock_client())
        agent.send_vision_request_with_schema = MagicMock(
            return_value=({"fields": {}}, _trace())
        )
        state = _make_state(num_pages=2)
        # Blank out page 2's image data
        state["page_images"][1] = {"page_number": 2, "data_uri": ""}
        out = agent.process(state)
        # Only page 1 should be in pass1_result.
        assert set(out["pass1_result"].keys()) == {1}

    def test_per_page_failure_logged_and_continues(self) -> None:
        agent = ExtractorPass1Agent(client=_mock_client())

        call_count = {"i": 0}

        def _send(*args: Any, **kwargs: Any) -> Any:
            call_count["i"] += 1
            if call_count["i"] == 1:
                raise RuntimeError("page 1 boom")
            return ({"fields": {"x": "ok"}}, _trace())

        agent.send_vision_request_with_schema = MagicMock(side_effect=_send)

        state = _make_state(num_pages=2)
        out = agent.process(state)
        # Page 1 stashed an error envelope; page 2 succeeded.
        assert "_pass1_error" in out["pass1_result"][1]
        assert out["pass1_result"][2]["fields"]["x"] == "ok"

    def test_empty_pages_raises(self) -> None:
        from src.agents.base import ExtractionError

        agent = ExtractorPass1Agent(client=_mock_client())
        with pytest.raises(ExtractionError, match="no page images"):
            agent.process({"page_images": []})

    def test_uses_primary_role(self) -> None:
        agent = ExtractorPass1Agent(client=_mock_client())
        send = MagicMock(return_value=({"fields": {}}, _trace()))
        agent.send_vision_request_with_schema = send
        agent.process(_make_state(num_pages=1))
        kwargs = send.call_args.kwargs
        assert kwargs["role"] is VLMRole.PRIMARY


# ---------------------------------------------------------------------------
# ExtractorPass2Agent
# ---------------------------------------------------------------------------


class TestExtractorPass2Agent:
    def test_writes_pass2_result_keyed_by_page(self) -> None:
        agent = ExtractorPass2Agent(client=_mock_client())
        agent.send_vision_request_with_schema = MagicMock(
            return_value=(
                {
                    "fields": {
                        "a": {
                            "value": "1",
                            "confidence": 0.9,
                            "bbox": [0.1, 0.1, 0.2, 0.2],
                        },
                    }
                },
                _trace(),
            )
        )
        state = _make_state(num_pages=2)
        out = agent.process(state)

        assert isinstance(out["pass2_result"], dict)
        assert set(out["pass2_result"].keys()) == {1, 2}
        for page_payload in out["pass2_result"].values():
            record = page_payload["fields"]["a"]
            assert record["bbox"] == [0.1, 0.1, 0.2, 0.2]

    def test_uses_secondary_role(self) -> None:
        agent = ExtractorPass2Agent(client=_mock_client())
        send = MagicMock(return_value=({"fields": {}}, _trace()))
        agent.send_vision_request_with_schema = send
        agent.process(_make_state(num_pages=1))
        kwargs = send.call_args.kwargs
        assert kwargs["role"] is VLMRole.SECONDARY
        # Pass 2 binds the AUDITOR envelope (bbox-mandated)
        assert kwargs["schema"] is Pass2AuditorEnvelope

    def test_normalises_bbox_hallucination(self) -> None:
        """``value=null AND bbox=[...]`` is the AUDITOR's invariant violation;
        the agent must drop the bbox so the reconciler doesn't trust it."""
        agent = ExtractorPass2Agent(client=_mock_client())
        agent.send_vision_request_with_schema = MagicMock(
            return_value=(
                {
                    "fields": {
                        "good": {
                            "value": "x",
                            "confidence": 0.9,
                            "bbox": [0.1, 0.1, 0.2, 0.2],
                        },
                        "hallucinated_bbox": {
                            "value": None,
                            "confidence": 0.4,
                            "bbox": [0.5, 0.5, 0.6, 0.6],  # invariant violation
                        },
                    }
                },
                _trace(),
            )
        )
        out = agent.process(_make_state(num_pages=1))
        records = out["pass2_result"][1]["fields"]
        assert records["good"]["bbox"] == [0.1, 0.1, 0.2, 0.2]
        # bbox dropped because value was null
        assert records["hallucinated_bbox"]["bbox"] is None

    def test_per_page_failure_continues(self) -> None:
        agent = ExtractorPass2Agent(client=_mock_client())

        call_count = {"i": 0}

        def _send(*args: Any, **kwargs: Any) -> Any:
            call_count["i"] += 1
            if call_count["i"] == 1:
                raise RuntimeError("page 1 boom")
            return ({"fields": {}}, _trace())

        agent.send_vision_request_with_schema = MagicMock(side_effect=_send)

        out = agent.process(_make_state(num_pages=2))
        assert "_pass2_error" in out["pass2_result"][1]


# ---------------------------------------------------------------------------
# Pass2AuditorEnvelope schema (Pydantic)
# ---------------------------------------------------------------------------


class TestPass2AuditorEnvelope:
    def test_schema_accepts_well_formed_response(self) -> None:
        env = Pass2AuditorEnvelope.model_validate(
            {
                "fields": {
                    "a": {
                        "value": "x",
                        "confidence": 0.9,
                        "bbox": [0.1, 0.1, 0.2, 0.2],
                        "location": "top-left",
                    }
                }
            }
        )
        assert env.fields["a"].value == "x"
        assert env.fields["a"].bbox == [0.1, 0.1, 0.2, 0.2]

    def test_schema_allows_null_value_and_null_bbox(self) -> None:
        env = Pass2AuditorEnvelope.model_validate(
            {
                "fields": {
                    "missing": {"value": None, "confidence": None, "bbox": None}
                }
            }
        )
        assert env.fields["missing"].value is None

    def test_schema_allows_extra_top_level_keys(self) -> None:
        env = Pass2AuditorEnvelope.model_validate(
            {
                "fields": {},
                "page_number": 7,
                "extraction_notes": "all good",
            }
        )
        assert env.fields == {}

    def test_schema_rejects_invalid_confidence(self) -> None:
        from pydantic import ValidationError

        with pytest.raises(ValidationError):
            Pass2AuditorEnvelope.model_validate(
                {
                    "fields": {
                        "x": {"value": "y", "confidence": 1.5, "bbox": None}
                    }
                }
            )
