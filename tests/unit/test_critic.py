"""
Phase 3 — Critic agent + prompt + schema unit tests.

Coverage:

* ``CriticReport`` schema accepts well-formed responses, rejects
  invalid trust_score / recommendation values.
* ``build_critic_system_prompt`` / ``build_critic_user_prompt``
  produce non-empty strings with the extraction JSON embedded.
* ``CriticAgent.process`` happy path, short-circuits, normalisation,
  failure handling — all mocked, no live VLM.
* Family rotation: agent calls ``send_vision_request_with_schema``
  with ``role=VLMRole.CRITIC``.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest
from pydantic import ValidationError

from src.agents.critic import CriticAgent, CriticConcern, CriticReport
from src.client.backends.protocol import VLMRole
from src.client.constrained import DecodingTrace
from src.client.lm_client import LMStudioClient
from src.prompts.critic import (
    build_critic_system_prompt,
    build_critic_user_prompt,
)


# ---------------------------------------------------------------------------
# CriticReport schema
# ---------------------------------------------------------------------------


class TestCriticReportSchema:
    def test_accepts_well_formed_response(self) -> None:
        report = CriticReport.model_validate(
            {
                "trust_score": 0.92,
                "concerns": [
                    {
                        "field_path": "patient_name",
                        "issue": "supported",
                        "severity": "info",
                        "observed_in_image": True,
                    },
                    {
                        "field_path": "service_lines.0.cpt_code",
                        "issue": "ambiguous",
                        "severity": "warning",
                        "observed_in_image": False,
                        "recommended_bbox": [0.1, 0.2, 0.3, 0.4],
                    },
                ],
                "recommendation": "verify_bbox",
            }
        )
        assert report.trust_score == 0.92
        assert len(report.concerns) == 2
        assert report.recommendation == "verify_bbox"
        assert report.concerns[1].recommended_bbox == [0.1, 0.2, 0.3, 0.4]

    def test_rejects_trust_score_above_one(self) -> None:
        with pytest.raises(ValidationError):
            CriticReport.model_validate(
                {
                    "trust_score": 1.5,
                    "concerns": [],
                    "recommendation": "accept",
                }
            )

    def test_rejects_invalid_recommendation(self) -> None:
        with pytest.raises(ValidationError):
            CriticReport.model_validate(
                {
                    "trust_score": 0.5,
                    "concerns": [],
                    "recommendation": "ignore",  # not in enum
                }
            )

    def test_rejects_invalid_severity(self) -> None:
        with pytest.raises(ValidationError):
            CriticReport.model_validate(
                {
                    "trust_score": 0.5,
                    "concerns": [
                        {
                            "field_path": "x",
                            "issue": "ambiguous",
                            "severity": "critical",  # not in enum
                        }
                    ],
                    "recommendation": "accept",
                }
            )

    def test_concern_minimum_required(self) -> None:
        # field_path + issue are required; everything else optional.
        c = CriticConcern.model_validate(
            {"field_path": "x", "issue": "not_visible"}
        )
        assert c.severity == "warning"  # default
        assert c.observed_in_image is False  # default
        assert c.recommended_bbox is None

    def test_extra_top_level_keys_allowed(self) -> None:
        report = CriticReport.model_validate(
            {
                "trust_score": 0.7,
                "concerns": [],
                "recommendation": "accept",
                "model_specific_extra": "value",
            }
        )
        assert report.trust_score == 0.7


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


class TestCriticPrompts:
    def test_system_prompt_has_critic_header(self) -> None:
        sp = build_critic_system_prompt()
        assert "CRITIC" in sp
        assert "audit" in sp.lower()
        # Forbidden text aimed at extractors must NOT appear in the
        # critic system prompt — the critic doesn't extract.
        # ``forbidden`` block is gated off in ``build_critic_system_prompt``.

    def test_user_prompt_embeds_extraction(self) -> None:
        prompt = build_critic_user_prompt(
            extraction={"patient_name": "Alice"},
            document_type="CMS-1500",
            page_number=1,
            page_count=2,
            modalities=["fax"],
        )
        assert "patient_name" in prompt
        assert "Alice" in prompt
        assert "CMS-1500" in prompt
        assert "fax" in prompt
        assert "Page 1 of 2" in prompt

    def test_user_prompt_handles_empty_extraction(self) -> None:
        prompt = build_critic_user_prompt(
            extraction={},
            document_type="UNKNOWN",
            page_number=1,
            page_count=1,
        )
        # Empty JSON object should still serialise cleanly.
        assert "{}" in prompt

    def test_user_prompt_omits_modalities_block_when_none(self) -> None:
        prompt = build_critic_user_prompt(
            extraction={"x": 1},
            document_type="X",
            page_number=1,
            page_count=1,
            modalities=None,
        )
        assert "Detected modalities" not in prompt


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------


def _trace(model_id: str = "stub-critic") -> DecodingTrace:
    return DecodingTrace(
        backend_name="stub",
        role=VLMRole.CRITIC,
        model_id=model_id,
        schema_name="CriticReport",
        latency_ms=12,
        tokens_in=100,
        tokens_out=80,
        schema_enforced=True,
    )


def _make_state(extraction: dict[str, Any] | None = None, num_pages: int = 1) -> dict[str, Any]:
    return {
        "processing_id": "p",
        "merged_extraction": extraction if extraction is not None else {"name": "Alice"},
        "page_images": [
            {"page_number": i + 1, "data_uri": f"data:image/png;base64,P{i}"}
            for i in range(num_pages)
        ],
        "document_type": "CMS-1500",
        "modalities": [],
        "errors": [],
        "warnings": [],
    }


class TestCriticAgent:
    def test_happy_path_writes_critic_report(self) -> None:
        agent = CriticAgent(client=MagicMock(spec_set=LMStudioClient))
        agent.send_vision_request_with_schema = MagicMock(
            return_value=(
                {
                    "trust_score": 0.92,
                    "concerns": [],
                    "recommendation": "accept",
                },
                _trace(),
            )
        )
        out = agent.process(_make_state())
        assert out["critic_report"]["trust_score"] == 0.92
        assert out["critic_recommendation"] == "accept"
        assert out["critic_model_id"] == "stub-critic"
        assert out["critic_latency_ms"] >= 0

    def test_uses_critic_role(self) -> None:
        agent = CriticAgent(client=MagicMock(spec_set=LMStudioClient))
        send = MagicMock(
            return_value=(
                {"trust_score": 0.5, "concerns": [], "recommendation": "accept"},
                _trace(),
            )
        )
        agent.send_vision_request_with_schema = send
        agent.process(_make_state())
        assert send.call_args.kwargs["role"] is VLMRole.CRITIC

    def test_empty_extraction_short_circuits_to_human_review(self) -> None:
        agent = CriticAgent(client=MagicMock(spec_set=LMStudioClient))
        send = MagicMock()
        agent.send_vision_request_with_schema = send
        out = agent.process(_make_state(extraction={}))
        assert out["critic_recommendation"] == "human_review"
        assert out["critic_report"]["trust_score"] == 0.0
        assert "_short_circuit_reason" in out["critic_report"]
        send.assert_not_called()

    def test_no_page_images_short_circuits_to_accept(self) -> None:
        agent = CriticAgent(client=MagicMock(spec_set=LMStudioClient))
        send = MagicMock()
        agent.send_vision_request_with_schema = send
        out = agent.process(_make_state(num_pages=0))
        assert out["critic_recommendation"] == "accept"
        send.assert_not_called()

    def test_blank_page_short_circuits_to_accept(self) -> None:
        agent = CriticAgent(client=MagicMock(spec_set=LMStudioClient))
        send = MagicMock()
        agent.send_vision_request_with_schema = send
        state = _make_state()
        state["page_images"][0] = {"page_number": 1, "data_uri": ""}
        out = agent.process(state)
        assert out["critic_recommendation"] == "accept"
        send.assert_not_called()

    def test_vlm_failure_falls_back_to_accept(self) -> None:
        agent = CriticAgent(client=MagicMock(spec_set=LMStudioClient))
        agent.send_vision_request_with_schema = MagicMock(
            side_effect=RuntimeError("vlm timeout")
        )
        out = agent.process(_make_state())
        # Critic failure is non-fatal: recommend ``accept`` so pipeline continues.
        assert out["critic_recommendation"] == "accept"
        assert "critic_call_failed" in out["critic_report"]["_short_circuit_reason"]

    def test_normalises_invalid_recommendation(self) -> None:
        agent = CriticAgent(client=MagicMock(spec_set=LMStudioClient))
        agent.send_vision_request_with_schema = MagicMock(
            return_value=(
                {
                    "trust_score": 0.7,
                    "concerns": [],
                    "recommendation": "BOGUS_VALUE",
                },
                _trace(),
            )
        )
        out = agent.process(_make_state())
        # Normalised back to ``accept`` so downstream routing isn't poisoned.
        assert out["critic_recommendation"] == "accept"

    def test_normalises_out_of_range_trust_score(self) -> None:
        agent = CriticAgent(client=MagicMock(spec_set=LMStudioClient))
        agent.send_vision_request_with_schema = MagicMock(
            return_value=(
                {
                    "trust_score": 2.5,  # invalid; will be clamped
                    "concerns": [],
                    "recommendation": "accept",
                },
                _trace(),
            )
        )
        out = agent.process(_make_state())
        assert 0.0 <= out["critic_report"]["trust_score"] <= 1.0
