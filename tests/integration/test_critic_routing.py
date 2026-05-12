"""
Phase 3 — Critic-driven routing integration tests.

Verifies the orchestrator topology under each combination of
``extraction.engine`` and ``extraction.critic_enabled``:

* Critic disabled (default) → no Critic node, no combiner node.
* Critic enabled → ``NODE_CRITIC`` + ``NODE_CRITIC_COMBINER`` land
  in the workflow regardless of engine.
* The Critic's ``recommendation`` drives ``_route_with_reason``:
  ``human_review`` → ROUTE_HUMAN_REVIEW; ``retry`` (with budget) →
  ROUTE_RETRY; ``retry`` (without budget) → ROUTE_HUMAN_REVIEW;
  ``accept``/``verify_bbox`` → fall through to legacy routing.
* Critic combiner writes ``confidence_components`` and recomputes
  ``overall_confidence`` from ``raw_combined``.

No live VLMs. All VLM calls stubbed.
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
def critic_disabled(monkeypatch):
    monkeypatch.setenv("EXTRACTION_CRITIC_ENABLED", "false")


@pytest.fixture
def critic_enabled_legacy(monkeypatch):
    monkeypatch.setenv("EXTRACTION_CRITIC_ENABLED", "true")
    monkeypatch.setenv("EXTRACTION_ENGINE", "legacy")


@pytest.fixture
def critic_enabled_dual(monkeypatch):
    monkeypatch.setenv("EXTRACTION_CRITIC_ENABLED", "true")
    monkeypatch.setenv("EXTRACTION_ENGINE", "dual_vlm")


def _identity(state: dict[str, Any]) -> dict[str, Any]:
    return state


# ---------------------------------------------------------------------------
# Topology under flag combinations
# ---------------------------------------------------------------------------


class TestCriticTopology:
    def test_critic_disabled_no_critic_nodes(self, critic_disabled) -> None:
        from src.agents.orchestrator import (
            NODE_CRITIC,
            NODE_CRITIC_COMBINER,
            create_extraction_workflow,
        )

        orch, _ = create_extraction_workflow(
            preprocess_fn=_identity,
            enable_checkpointing=False,
            enable_vlm_first=False,
            enable_splitter=False,
            enable_table_detection=False,
        )
        nodes = orch._workflow.nodes  # type: ignore[attr-defined]
        assert NODE_CRITIC not in nodes
        assert NODE_CRITIC_COMBINER not in nodes
        assert orch._critic is None  # type: ignore[attr-defined]
        assert orch._critic_combiner_node is None  # type: ignore[attr-defined]

    def test_critic_enabled_legacy_registers_nodes(
        self, critic_enabled_legacy
    ) -> None:
        from src.agents.orchestrator import (
            NODE_CRITIC,
            NODE_CRITIC_COMBINER,
            create_extraction_workflow,
        )

        orch, _ = create_extraction_workflow(
            preprocess_fn=_identity,
            enable_checkpointing=False,
            enable_vlm_first=False,
            enable_splitter=False,
            enable_table_detection=False,
        )
        nodes = orch._workflow.nodes  # type: ignore[attr-defined]
        assert NODE_CRITIC in nodes
        assert NODE_CRITIC_COMBINER in nodes
        assert orch._critic is not None  # type: ignore[attr-defined]
        assert orch._critic_combiner_node is not None  # type: ignore[attr-defined]

    def test_critic_enabled_dual_vlm_registers_all_nodes(
        self, critic_enabled_dual
    ) -> None:
        from src.agents.orchestrator import (
            NODE_CRITIC,
            NODE_CRITIC_COMBINER,
            NODE_EXTRACT_PASS1,
            NODE_EXTRACT_PASS2,
            NODE_RECONCILE,
            create_extraction_workflow,
        )

        orch, _ = create_extraction_workflow(
            preprocess_fn=_identity,
            enable_checkpointing=False,
            enable_vlm_first=False,
            enable_splitter=False,
            enable_table_detection=False,
        )
        nodes = orch._workflow.nodes  # type: ignore[attr-defined]
        # Both Phase 2 (dual-VLM) and Phase 3 (critic) nodes coexist.
        assert NODE_EXTRACT_PASS1 in nodes
        assert NODE_EXTRACT_PASS2 in nodes
        assert NODE_RECONCILE in nodes
        assert NODE_CRITIC in nodes
        assert NODE_CRITIC_COMBINER in nodes


# ---------------------------------------------------------------------------
# Recommendation-driven routing
# ---------------------------------------------------------------------------


class TestCriticRecommendationRouting:
    def _make_orchestrator(self, critic_enabled_legacy):
        from src.agents.orchestrator import create_extraction_workflow

        orch, _ = create_extraction_workflow(
            preprocess_fn=_identity,
            enable_checkpointing=False,
            enable_vlm_first=False,
            enable_splitter=False,
            enable_table_detection=False,
        )
        return orch

    def _state(self, recommendation: str, retry_count: int = 0) -> dict[str, Any]:
        from src.pipeline.state import ConfidenceLevel, ExtractionStatus

        return {
            "status": ExtractionStatus.EXTRACTING.value,
            "confidence_level": ConfidenceLevel.MEDIUM.value,
            "overall_confidence": 0.6,
            "validation_is_valid": True,
            "validation_requires_retry": False,
            "validation_requires_human_review": False,
            "retry_count": retry_count,
            "critic_recommendation": recommendation,
        }

    def test_human_review_recommendation_routes_to_human_review(
        self, critic_enabled_legacy
    ) -> None:
        from src.agents.orchestrator import ROUTE_HUMAN_REVIEW

        orch = self._make_orchestrator(critic_enabled_legacy)
        decision, reason = orch._route_with_reason(  # type: ignore[attr-defined]
            self._state("human_review")
        )
        assert decision == ROUTE_HUMAN_REVIEW
        assert "critic" in reason.lower()

    def test_retry_recommendation_with_budget_routes_to_retry(
        self, critic_enabled_legacy
    ) -> None:
        from src.agents.orchestrator import ROUTE_RETRY

        orch = self._make_orchestrator(critic_enabled_legacy)
        decision, reason = orch._route_with_reason(  # type: ignore[attr-defined]
            self._state("retry", retry_count=0)
        )
        assert decision == ROUTE_RETRY
        assert "critic" in reason.lower()

    def test_retry_recommendation_without_budget_escalates_to_review(
        self, critic_enabled_legacy
    ) -> None:
        from src.agents.orchestrator import ROUTE_HUMAN_REVIEW

        orch = self._make_orchestrator(critic_enabled_legacy)
        # Use a retry_count >= max_retries (default 2 in OrchestratorAgent).
        decision, reason = orch._route_with_reason(  # type: ignore[attr-defined]
            self._state("retry", retry_count=10)
        )
        assert decision == ROUTE_HUMAN_REVIEW
        assert "critic" in reason.lower()

    def test_accept_falls_through_to_legacy_routing(
        self, critic_enabled_legacy
    ) -> None:
        # Accept means the critic does not impose a routing override;
        # legacy confidence-based routing decides.
        orch = self._make_orchestrator(critic_enabled_legacy)
        decision, reason = orch._route_with_reason(  # type: ignore[attr-defined]
            self._state("accept")
        )
        # The routing reason should NOT mention the critic.
        assert "critic" not in reason.lower()

    def test_verify_bbox_falls_through_to_legacy_routing(
        self, critic_enabled_legacy
    ) -> None:
        # verify_bbox is handled in-line by the bbox-roundtrip helper,
        # not at the routing layer; routing falls through.
        orch = self._make_orchestrator(critic_enabled_legacy)
        decision, reason = orch._route_with_reason(  # type: ignore[attr-defined]
            self._state("verify_bbox")
        )
        assert "critic" not in reason.lower()


# ---------------------------------------------------------------------------
# Combiner end-to-end
# ---------------------------------------------------------------------------


class TestCombinerNodeBehavior:
    def test_combiner_writes_confidence_components(
        self, critic_enabled_legacy
    ) -> None:
        from src.agents.orchestrator import create_extraction_workflow

        orch, _ = create_extraction_workflow(
            preprocess_fn=_identity,
            enable_checkpointing=False,
            enable_vlm_first=False,
            enable_splitter=False,
            enable_table_detection=False,
        )
        state: dict[str, Any] = {
            "overall_confidence": 0.8,
            "critic_report": {"trust_score": 0.9},
            "modalities": [],
        }
        out = orch._critic_combiner_node(state)  # type: ignore[attr-defined]
        assert "confidence_components" in out
        components = out["confidence_components"]
        assert components["dual_pass"] == 0.8
        assert components["critic"] == 0.9
        # Combined: 0.5*0.8 + 0.3*0.9 + 0.2*1.0 = 0.4 + 0.27 + 0.2 = 0.87
        assert components["raw_combined"] == pytest.approx(0.87)
        assert out["overall_confidence"] == pytest.approx(0.87)

    def test_combiner_recomputes_confidence_level(
        self, critic_enabled_legacy
    ) -> None:
        from src.agents.orchestrator import create_extraction_workflow
        from src.pipeline.state import ConfidenceLevel

        orch, _ = create_extraction_workflow(
            preprocess_fn=_identity,
            enable_checkpointing=False,
            enable_vlm_first=False,
            enable_splitter=False,
            enable_table_detection=False,
        )
        # Heavy fax penalty → low combined → LOW level.
        state: dict[str, Any] = {
            "overall_confidence": 0.3,
            "critic_report": {"trust_score": 0.3},
            "modalities": ["fax"],
        }
        out = orch._critic_combiner_node(state)  # type: ignore[attr-defined]
        assert out["confidence_level"] == ConfidenceLevel.LOW.value
