"""
Phase 0 tests — VLM backend factory + role routing.

Goals:

* Default ``VLM_BACKEND=lm_studio`` resolves to an ``LMStudioBackend``
  configured from ``LMStudioSettings`` (zero-config compat).
* ``VLM_BACKEND=vllm`` resolves to a ``VLLMBackend`` reading
  ``VLLMBackendSettings``.
* Switching backends between calls (test-time settings injection)
  returns fresh instances after ``reset_cache()``.
* ``ModelRouter.route_for_role`` maps the four ``VLMRole`` values to
  sensible ``ModelTask`` choices, falling back to GENERAL when no
  agent name is supplied.
* Both backends conform to the ``VLMBackend`` runtime-checkable
  Protocol.

These tests do NOT require a live VLM server. Health checks are
patched. The only thing exercised in real is settings construction
and Python-level routing logic.
"""

from __future__ import annotations

import os
from unittest.mock import patch

import pytest

from src.client.backends import (
    BackendCapabilities,
    LMStudioBackend,
    VLLMBackend,
    VLMBackend,
    VLMRole,
    get_backend,
)
from src.client.backends.factory import reset_cache
from src.client.model_router import (
    ModelRouter,
    ModelTask,
    qwen3vl_config,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_backend_cache():
    """Drop the process-wide cache before and after every test."""
    reset_cache()
    yield
    reset_cache()


@pytest.fixture
def lm_studio_settings(monkeypatch):
    """Force ``VLM_BACKEND=lm_studio`` and clear vLLM env."""
    monkeypatch.setenv("VLM_BACKEND", "lm_studio")
    # Clear vLLM env so the vllm settings block defaults stay intact.
    for key in list(os.environ):
        if key.startswith("VLLM_"):
            monkeypatch.delenv(key, raising=False)
    # Force a fresh Settings instance — get_settings() is lru_cached.
    from src.config.settings import get_settings

    get_settings.cache_clear()  # type: ignore[attr-defined]
    yield
    get_settings.cache_clear()  # type: ignore[attr-defined]


@pytest.fixture
def vllm_settings(monkeypatch):
    """Force ``VLM_BACKEND=vllm`` with valid primary URL/model."""
    monkeypatch.setenv("VLM_BACKEND", "vllm")
    monkeypatch.setenv("VLLM_PRIMARY_URL", "http://localhost:8001/v1")
    monkeypatch.setenv("VLLM_PRIMARY_MODEL", "veridoc-primary")
    from src.config.settings import get_settings

    get_settings.cache_clear()  # type: ignore[attr-defined]
    yield
    get_settings.cache_clear()  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Factory selection
# ---------------------------------------------------------------------------


class TestFactorySelection:
    def test_default_resolves_to_lm_studio(self, lm_studio_settings) -> None:
        backend = get_backend()
        assert isinstance(backend, LMStudioBackend)
        assert backend.name == "lm_studio"

    def test_vllm_when_configured(self, vllm_settings) -> None:
        backend = get_backend()
        assert isinstance(backend, VLLMBackend)
        assert backend.name == "vllm"

    def test_factory_caches(self, lm_studio_settings) -> None:
        b1 = get_backend()
        b2 = get_backend()
        assert b1 is b2  # same process-wide instance

    def test_reset_cache_yields_fresh_instance(self, lm_studio_settings) -> None:
        b1 = get_backend()
        reset_cache()
        b2 = get_backend()
        assert b1 is not b2

    def test_unknown_backend_raises(self, lm_studio_settings) -> None:
        # Bypass Pydantic validation by constructing a minimal stand-in
        # that carries the same attribute shape the factory reads from.
        from types import SimpleNamespace

        from src.config.settings import get_settings

        real = get_settings()
        fake_vlm = SimpleNamespace(
            backend="invalid_backend",
            mode=real.vlm.mode,
            lm_studio=real.vlm.lm_studio,
            vllm=real.vlm.vllm,
        )
        fake_settings = SimpleNamespace(vlm=fake_vlm, lm_studio=real.lm_studio)
        with pytest.raises(ValueError, match="Unsupported VLM_BACKEND"):
            get_backend(settings=fake_settings)  # type: ignore[arg-type]

    def test_vllm_requires_primary_url_and_model(self, monkeypatch) -> None:
        monkeypatch.setenv("VLM_BACKEND", "vllm")
        monkeypatch.setenv("VLLM_PRIMARY_URL", "")
        monkeypatch.setenv("VLLM_PRIMARY_MODEL", "")
        from src.config.settings import get_settings

        get_settings.cache_clear()  # type: ignore[attr-defined]
        with pytest.raises(ValueError, match="VLM_BACKEND=vllm requires"):
            get_backend()


# ---------------------------------------------------------------------------
# Protocol conformance (runtime_checkable)
# ---------------------------------------------------------------------------


class TestProtocolConformance:
    def test_lm_studio_backend_isinstance_vlm_backend(self) -> None:
        backend = LMStudioBackend(
            primary_url="http://localhost:1234/v1",
            primary_model="qwen3-vl",
        )
        assert isinstance(backend, VLMBackend)

    def test_vllm_backend_isinstance_vlm_backend(self) -> None:
        backend = VLLMBackend(
            primary_url="http://localhost:8001/v1",
            primary_model="veridoc-primary",
        )
        assert isinstance(backend, VLMBackend)

    def test_capabilities_shape(self) -> None:
        caps = LMStudioBackend(
            primary_url="http://localhost:1234/v1",
            primary_model="qwen3-vl",
        ).capabilities()
        assert isinstance(caps, BackendCapabilities)
        assert caps.name == "lm_studio"
        # single_only by default → no real dual-VLM
        assert caps.supports_dual_vlm is False


# ---------------------------------------------------------------------------
# LM Studio dual-mode role resolution
# ---------------------------------------------------------------------------


class TestLMStudioDualMode:
    def test_single_only_collapses_secondary_to_primary(self) -> None:
        backend = LMStudioBackend(
            primary_url="http://localhost:1234/v1",
            primary_model="qwen3-vl",
            dual_mode="single_only",
        )
        assert backend.resolve(VLMRole.PRIMARY) == (
            "http://localhost:1234/v1",
            "qwen3-vl",
        )
        assert backend.resolve(VLMRole.SECONDARY) == (
            "http://localhost:1234/v1",
            "qwen3-vl",
        )
        assert backend.resolve(VLMRole.CRITIC) == (
            "http://localhost:1234/v1",
            "qwen3-vl",
        )
        # capabilities reflect the degradation
        assert backend.capabilities().supports_dual_vlm is False

    def test_dual_instance_resolves_secondary_to_secondary(self) -> None:
        backend = LMStudioBackend(
            primary_url="http://localhost:1234/v1",
            primary_model="qwen3-vl",
            secondary_url="http://localhost:1235/v1",
            secondary_model="gemma4-vl",
            dual_mode="dual_instance",
        )
        assert backend.resolve(VLMRole.SECONDARY) == (
            "http://localhost:1235/v1",
            "gemma4-vl",
        )
        assert backend.resolve(VLMRole.CRITIC) == (
            "http://localhost:1235/v1",
            "gemma4-vl",
        )
        assert backend.capabilities().supports_dual_vlm is True

    def test_dual_instance_without_secondary_falls_back(self) -> None:
        # Operator declared dual_instance but forgot to set secondary_url.
        # Backend should NOT pretend dual-VLM is active.
        backend = LMStudioBackend(
            primary_url="http://localhost:1234/v1",
            primary_model="qwen3-vl",
            dual_mode="dual_instance",  # but no secondary_*
        )
        assert backend.resolve(VLMRole.SECONDARY) == (
            "http://localhost:1234/v1",
            "qwen3-vl",
        )
        assert backend.capabilities().supports_dual_vlm is False

    def test_unknown_dual_mode_raises_in_settings(self, monkeypatch) -> None:
        from src.config.settings import LMStudioBackendSettings

        with pytest.raises(ValueError, match="dual_mode must be one of"):
            LMStudioBackendSettings(dual_mode="bogus")


# ---------------------------------------------------------------------------
# vLLM backend
# ---------------------------------------------------------------------------


class TestVLLMBackend:
    def test_dual_when_secondary_configured(self) -> None:
        backend = VLLMBackend(
            primary_url="http://localhost:8001/v1",
            primary_model="qwen",
            secondary_url="http://localhost:8002/v1",
            secondary_model="gemma",
        )
        assert backend.capabilities().supports_dual_vlm is True
        assert backend.capabilities().supports_logprobs is True
        assert backend.capabilities().supports_tensor_parallelism is True

    def test_no_dual_without_secondary(self) -> None:
        backend = VLLMBackend(
            primary_url="http://localhost:8001/v1",
            primary_model="qwen",
        )
        assert backend.capabilities().supports_dual_vlm is False

    def test_invalid_guided_decoding_backend(self) -> None:
        with pytest.raises(ValueError, match="guided_decoding_backend must be one of"):
            VLLMBackend(
                primary_url="http://localhost:8001/v1",
                primary_model="qwen",
                guided_decoding_backend="bogus",
            )

    def test_guided_backend_default_xgrammar(self) -> None:
        backend = VLLMBackend(
            primary_url="http://localhost:8001/v1",
            primary_model="qwen",
        )
        assert backend.guided_decoding_backend == "xgrammar"


# ---------------------------------------------------------------------------
# ModelRouter role bridging
# ---------------------------------------------------------------------------


class TestModelRouterRoleBridge:
    def _router(self) -> ModelRouter:
        return ModelRouter(
            models=[qwen3vl_config()],
            default_model_name="qwen3-vl",
        )

    def test_primary_routes_via_agent_name(self) -> None:
        decision = self._router().route_for_role(
            VLMRole.PRIMARY, agent_name="extractor"
        )
        # FIELD_EXTRACTION is in qwen3-vl's capabilities
        assert decision.task is ModelTask.FIELD_EXTRACTION
        assert decision.model.name == "qwen3-vl"

    def test_lite_falls_back_to_general_without_agent(self) -> None:
        decision = self._router().route_for_role(VLMRole.LITE)
        assert decision.task is ModelTask.GENERAL

    def test_secondary_uses_field_extraction(self) -> None:
        decision = self._router().route_for_role(VLMRole.SECONDARY)
        assert decision.task is ModelTask.FIELD_EXTRACTION

    def test_critic_uses_verification(self) -> None:
        decision = self._router().route_for_role(VLMRole.CRITIC)
        assert decision.task is ModelTask.VERIFICATION

    def test_role_for_agent_pass1_is_primary(self) -> None:
        assert (
            self._router().role_for_agent("extractor_pass1") is VLMRole.PRIMARY
        )

    def test_role_for_agent_pass2_is_secondary(self) -> None:
        assert (
            self._router().role_for_agent("extractor_pass2") is VLMRole.SECONDARY
        )

    def test_role_for_agent_critic(self) -> None:
        assert self._router().role_for_agent("critic") is VLMRole.CRITIC

    def test_role_for_unknown_agent_is_primary(self) -> None:
        assert self._router().role_for_agent("anything") is VLMRole.PRIMARY
