"""
Phase 2 — dual-VLM pipeline integration through the orchestrator factory.

Verifies the engine flag actually toggles topology:

* ``EXTRACTION_ENGINE=legacy`` (default) → orchestrator wires the
  legacy single-VLM extractor; no Pass 1/Pass 2/reconcile nodes are
  registered.
* ``EXTRACTION_ENGINE=dual_vlm`` → factory instantiates
  ``ExtractorPass1Agent`` and ``ExtractorPass2Agent``, builds a
  reconciler closure, and the workflow contains the new nodes.

Plus a smoke test that the dual-VLM reconciler closure correctly
fuses two passes through the orchestrator's ``_run_extractor`` ->
pass1 -> pass2 -> reconcile chain when run directly with mocked
``send_vision_request_with_schema``.

No live VLMs. Backends and VLM calls are stubbed.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.client.backends.factory import reset_cache
from src.client.backends.protocol import VLMRole
from src.client.constrained import DecodingTrace


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
def legacy_engine(monkeypatch):
    monkeypatch.setenv("EXTRACTION_ENGINE", "legacy")


@pytest.fixture
def dual_vlm_engine(monkeypatch):
    monkeypatch.setenv("EXTRACTION_ENGINE", "dual_vlm")
    # These Phase 2 tests assert against the legacy ``merged_extraction``
    # shape; explicitly disable the Phase 4 enforce flag so they remain
    # green when the flag is on globally.
    monkeypatch.setenv("PROVENANCE_ENFORCE_FIELD_VALUE_WRAPPER", "false")


# ---------------------------------------------------------------------------
# Engine flag drives orchestrator topology
# ---------------------------------------------------------------------------


class TestEngineFlagWiring:
    def _identity_preprocess(self, state: dict[str, Any]) -> dict[str, Any]:
        return state

    def test_legacy_does_not_register_dual_vlm_nodes(self, legacy_engine) -> None:
        from src.agents.orchestrator import (
            NODE_EXTRACT_PASS1,
            NODE_EXTRACT_PASS2,
            NODE_RECONCILE,
            create_extraction_workflow,
        )

        orch, _ = create_extraction_workflow(
            preprocess_fn=self._identity_preprocess,
            enable_checkpointing=False,
            enable_vlm_first=False,
            enable_splitter=False,
            enable_table_detection=False,
        )
        nodes = orch._workflow.nodes  # type: ignore[attr-defined]
        assert NODE_EXTRACT_PASS1 not in nodes
        assert NODE_EXTRACT_PASS2 not in nodes
        assert NODE_RECONCILE not in nodes
        # And the agent attributes stayed unset.
        assert orch._extractor_pass1 is None  # type: ignore[attr-defined]
        assert orch._extractor_pass2 is None  # type: ignore[attr-defined]
        assert orch._reconciler_node is None  # type: ignore[attr-defined]

    def test_dual_vlm_registers_dual_vlm_nodes(self, dual_vlm_engine) -> None:
        from src.agents.orchestrator import (
            NODE_EXTRACT_PASS1,
            NODE_EXTRACT_PASS2,
            NODE_RECONCILE,
            create_extraction_workflow,
        )

        orch, _ = create_extraction_workflow(
            preprocess_fn=self._identity_preprocess,
            enable_checkpointing=False,
            enable_vlm_first=False,
            enable_splitter=False,
            enable_table_detection=False,
        )
        nodes = orch._workflow.nodes  # type: ignore[attr-defined]
        assert NODE_EXTRACT_PASS1 in nodes
        assert NODE_EXTRACT_PASS2 in nodes
        assert NODE_RECONCILE in nodes
        assert orch._extractor_pass1 is not None  # type: ignore[attr-defined]
        assert orch._extractor_pass2 is not None  # type: ignore[attr-defined]
        assert orch._reconciler_node is not None  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Reconciler closure end-to-end (via orchestrator)
# ---------------------------------------------------------------------------


class TestDualVLMReconcileClosure:
    def test_pass1_pass2_reconcile_chain_fuses_extraction(self, dual_vlm_engine) -> None:
        """End-to-end smoke through pass1 -> pass2 -> reconcile.

        * Stub Pass 1 to emit ``{name: "Alice"}`` + bbox-less.
        * Stub Pass 2 to emit ``{name: "Alice", bbox=[...]}``.
        * Reconciler should fuse to ``{name: "Alice"}`` via tier-1
          exact match, write ``merged_extraction`` and
          ``reconciliation_metadata``.
        """
        from src.agents.orchestrator import create_extraction_workflow

        orch, _ = create_extraction_workflow(
            preprocess_fn=lambda s: s,
            enable_checkpointing=False,
            enable_vlm_first=False,
            enable_splitter=False,
            enable_table_detection=False,
        )

        pass1 = orch._extractor_pass1  # type: ignore[attr-defined]
        pass2 = orch._extractor_pass2  # type: ignore[attr-defined]
        assert pass1 is not None and pass2 is not None

        # Stub VLM calls on each agent.
        trace = DecodingTrace(
            backend_name="stub",
            role=VLMRole.PRIMARY,
            model_id="stub-model",
            schema_name="any",
            latency_ms=5,
            tokens_in=1,
            tokens_out=1,
            schema_enforced=True,
        )
        pass1.send_vision_request_with_schema = MagicMock(
            return_value=(
                {"fields": {"name": {"value": "Alice", "confidence": 0.9}}},
                trace,
            )
        )
        pass2.send_vision_request_with_schema = MagicMock(
            return_value=(
                {
                    "fields": {
                        "name": {
                            "value": "Alice",
                            "confidence": 0.85,
                            "bbox": [0.1, 0.1, 0.3, 0.2],
                        }
                    }
                },
                trace,
            )
        )

        state: dict[str, Any] = {
            "processing_id": "p",
            "pdf_path": "/tmp/x.pdf",
            "page_images": [
                {"page_number": 1, "data_uri": "data:image/png;base64,A"},
            ],
            "document_type": "CMS-1500",
            "selected_schema_name": "cms_1500",
            "modalities": [],
            "errors": [],
            "warnings": [],
        }

        # Drive the chain manually (LangGraph would do this; we exercise
        # the runners directly to keep the test fast and dep-free).
        state = orch._run_extractor(state)  # type: ignore[attr-defined]
        # Engine is dual_vlm so _run_extractor is a pass-through.
        state = orch._run_extractor_pass1(state)  # type: ignore[attr-defined]
        state = orch._run_extractor_pass2(state)  # type: ignore[attr-defined]
        state = orch._reconciler_node(state)  # type: ignore[attr-defined]

        # Reconciler should have fused the matching value into merged_extraction.
        assert state["merged_extraction"] == {"name": "Alice"}
        # Metadata records the agreement.
        meta = state["reconciliation_metadata"]
        assert meta["agreement_rate"] == pytest.approx(1.0)
        assert meta["disagreements"] == 0
        assert meta["total_fields"] == 1
        assert state["extraction_engine"] == "dual_vlm"

    def test_disagreement_routes_through_pattern_detector(
        self, dual_vlm_engine
    ) -> None:
        """Pass 1 says ``N/A`` (placeholder), Pass 2 says ``Alice``.
        Tier-4 (pattern detector) should drop the placeholder; merged
        extraction takes Pass 2's value."""
        from src.agents.orchestrator import create_extraction_workflow

        orch, _ = create_extraction_workflow(
            preprocess_fn=lambda s: s,
            enable_checkpointing=False,
            enable_vlm_first=False,
            enable_splitter=False,
            enable_table_detection=False,
        )

        trace = DecodingTrace(
            backend_name="stub",
            role=VLMRole.PRIMARY,
            model_id="stub",
            schema_name="any",
            latency_ms=5,
            tokens_in=1,
            tokens_out=1,
            schema_enforced=True,
        )
        orch._extractor_pass1.send_vision_request_with_schema = MagicMock(  # type: ignore[attr-defined]
            return_value=(
                {"fields": {"name": {"value": "N/A", "confidence": 0.5}}},
                trace,
            )
        )
        orch._extractor_pass2.send_vision_request_with_schema = MagicMock(  # type: ignore[attr-defined]
            return_value=(
                {
                    "fields": {
                        "name": {
                            "value": "Alice",
                            "confidence": 0.6,
                            "bbox": [0.1, 0.1, 0.2, 0.2],
                        }
                    }
                },
                trace,
            )
        )

        state: dict[str, Any] = {
            "processing_id": "p",
            "pdf_path": "/tmp/x.pdf",
            "page_images": [
                {"page_number": 1, "data_uri": "data:image/png;base64,A"},
            ],
            "document_type": "CMS-1500",
            "selected_schema_name": "cms_1500",
            "modalities": [],
            "errors": [],
            "warnings": [],
        }

        state = orch._run_extractor(state)  # type: ignore[attr-defined]
        state = orch._run_extractor_pass1(state)  # type: ignore[attr-defined]
        state = orch._run_extractor_pass2(state)  # type: ignore[attr-defined]
        state = orch._reconciler_node(state)  # type: ignore[attr-defined]

        assert state["merged_extraction"]["name"] == "Alice"
        meta = state["reconciliation_metadata"]
        assert meta["disagreements"] == 1
        assert meta["tiebreakers_used"].get("pattern_detector") == 1


# ---------------------------------------------------------------------------
# Backend capability matrix sanity (vLLM + LM Studio paths configured)
# ---------------------------------------------------------------------------


class TestBackendCapabilityIntegration:
    """Verify both backend choices wire through the dual-VLM topology
    without requiring live VLMs. Capabilities flow into the agent's
    ``send_vision_request_with_schema`` call as the schema -> response_format
    or schema -> extra_body translation, exercised in Phase 1 tests."""

    def test_lm_studio_dual_instance_advertises_dual_vlm(self, monkeypatch) -> None:
        from src.client.backends.factory import get_backend
        from src.config.settings import get_settings

        monkeypatch.setenv("VLM_BACKEND", "lm_studio")
        monkeypatch.setenv("VLM_LM_STUDIO_DUAL_MODE", "dual_instance")
        monkeypatch.setenv("VLM_LM_STUDIO_PRIMARY_URL", "http://primary/v1")
        monkeypatch.setenv("VLM_LM_STUDIO_PRIMARY_MODEL", "primary-model")
        monkeypatch.setenv("VLM_LM_STUDIO_SECONDARY_URL", "http://secondary/v1")
        monkeypatch.setenv("VLM_LM_STUDIO_SECONDARY_MODEL", "secondary-model")
        get_settings.cache_clear()  # type: ignore[attr-defined]

        backend = get_backend()
        caps = backend.capabilities()
        assert caps.supports_dual_vlm is True
        assert caps.notes == ()

    def test_lm_studio_single_only_does_not_pretend(self, monkeypatch) -> None:
        from src.client.backends.factory import get_backend
        from src.config.settings import get_settings

        monkeypatch.setenv("VLM_BACKEND", "lm_studio")
        monkeypatch.setenv("VLM_LM_STUDIO_DUAL_MODE", "single_only")
        get_settings.cache_clear()  # type: ignore[attr-defined]

        backend = get_backend()
        caps = backend.capabilities()
        assert caps.supports_dual_vlm is False
        assert any("collapse" in note for note in caps.notes)

    def test_vllm_with_secondary_advertises_dual_vlm(self, monkeypatch) -> None:
        from src.client.backends.factory import get_backend
        from src.config.settings import get_settings

        monkeypatch.setenv("VLM_BACKEND", "vllm")
        monkeypatch.setenv("VLLM_PRIMARY_URL", "http://primary:8001/v1")
        monkeypatch.setenv("VLLM_PRIMARY_MODEL", "qwen")
        monkeypatch.setenv("VLLM_SECONDARY_URL", "http://secondary:8002/v1")
        monkeypatch.setenv("VLLM_SECONDARY_MODEL", "gemma")
        get_settings.cache_clear()  # type: ignore[attr-defined]

        backend = get_backend()
        caps = backend.capabilities()
        assert caps.name == "vllm"
        assert caps.supports_dual_vlm is True
        assert caps.supports_logprobs is True
        assert caps.supports_tensor_parallelism is True

    def test_role_resolution_routes_to_correct_endpoint_lm_studio(
        self, monkeypatch
    ) -> None:
        from src.client.backends.factory import get_backend
        from src.client.backends.protocol import VLMRole
        from src.config.settings import get_settings

        monkeypatch.setenv("VLM_BACKEND", "lm_studio")
        monkeypatch.setenv("VLM_LM_STUDIO_DUAL_MODE", "dual_instance")
        monkeypatch.setenv("VLM_LM_STUDIO_PRIMARY_URL", "http://p/v1")
        monkeypatch.setenv("VLM_LM_STUDIO_PRIMARY_MODEL", "p")
        monkeypatch.setenv("VLM_LM_STUDIO_SECONDARY_URL", "http://s/v1")
        monkeypatch.setenv("VLM_LM_STUDIO_SECONDARY_MODEL", "s")
        get_settings.cache_clear()  # type: ignore[attr-defined]

        backend = get_backend()
        assert backend.resolve(VLMRole.PRIMARY) == ("http://p/v1", "p")
        assert backend.resolve(VLMRole.SECONDARY) == ("http://s/v1", "s")
        # Critic also routes to secondary in dual_instance mode.
        assert backend.resolve(VLMRole.CRITIC) == ("http://s/v1", "s")

    def test_role_resolution_routes_to_correct_endpoint_vllm(
        self, monkeypatch
    ) -> None:
        from src.client.backends.factory import get_backend
        from src.client.backends.protocol import VLMRole
        from src.config.settings import get_settings

        monkeypatch.setenv("VLM_BACKEND", "vllm")
        monkeypatch.setenv("VLLM_PRIMARY_URL", "http://p:8001/v1")
        monkeypatch.setenv("VLLM_PRIMARY_MODEL", "qwen")
        monkeypatch.setenv("VLLM_SECONDARY_URL", "http://s:8002/v1")
        monkeypatch.setenv("VLLM_SECONDARY_MODEL", "gemma")
        get_settings.cache_clear()  # type: ignore[attr-defined]

        backend = get_backend()
        assert backend.resolve(VLMRole.PRIMARY) == ("http://p:8001/v1", "qwen")
        assert backend.resolve(VLMRole.SECONDARY) == ("http://s:8002/v1", "gemma")
