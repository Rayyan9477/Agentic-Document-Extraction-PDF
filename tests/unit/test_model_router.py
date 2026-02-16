"""
Unit tests for Phase 3C: Multi-Model Routing.

Tests ModelTask, ModelConfig, RoutingDecision, ModelRouter,
pre-built configs (florence2, qwen3vl), and BaseAgent integration.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import pytest

from src.client.model_router import (
    ModelConfig,
    ModelRouter,
    ModelTask,
    RoutingDecision,
    florence2_config,
    qwen3vl_config,
)


# ──────────────────────────────────────────────────────────────────
# ModelTask Tests
# ──────────────────────────────────────────────────────────────────


class TestModelTask:
    def test_all_tasks_defined(self):
        expected = {
            "classification",
            "layout_analysis",
            "table_detection",
            "component_detection",
            "field_extraction",
            "verification",
            "schema_generation",
            "handwriting_recognition",
            "general",
        }
        actual = {t.value for t in ModelTask}
        assert actual == expected

    def test_task_is_str_enum(self):
        assert isinstance(ModelTask.CLASSIFICATION, str)
        assert ModelTask.CLASSIFICATION == "classification"


# ──────────────────────────────────────────────────────────────────
# ModelConfig Tests
# ──────────────────────────────────────────────────────────────────


class TestModelConfig:
    def test_basic_creation(self):
        cfg = ModelConfig(name="test", model_id="test-model")
        assert cfg.name == "test"
        assert cfg.model_id == "test-model"
        assert cfg.enabled is True
        assert cfg.priority == 0
        assert cfg.capabilities == set()

    def test_supports_task(self):
        cfg = ModelConfig(
            name="test",
            model_id="m",
            capabilities={ModelTask.CLASSIFICATION, ModelTask.LAYOUT_ANALYSIS},
        )
        assert cfg.supports(ModelTask.CLASSIFICATION) is True
        assert cfg.supports(ModelTask.FIELD_EXTRACTION) is False

    def test_to_dict(self):
        cfg = ModelConfig(
            name="test",
            model_id="m",
            capabilities={ModelTask.CLASSIFICATION},
            priority=5,
        )
        d = cfg.to_dict()
        assert d["name"] == "test"
        assert d["model_id"] == "m"
        assert d["priority"] == 5
        assert "classification" in d["capabilities"]
        assert d["enabled"] is True

    def test_to_dict_capabilities_sorted(self):
        cfg = ModelConfig(
            name="test",
            model_id="m",
            capabilities={ModelTask.VERIFICATION, ModelTask.CLASSIFICATION},
        )
        d = cfg.to_dict()
        assert d["capabilities"] == ["classification", "verification"]


# ──────────────────────────────────────────────────────────────────
# Pre-built Configs Tests
# ──────────────────────────────────────────────────────────────────


class TestPrebuiltConfigs:
    def test_florence2_defaults(self):
        cfg = florence2_config()
        assert cfg.name == "florence-2"
        assert cfg.model_id == "florence-2"
        assert ModelTask.LAYOUT_ANALYSIS in cfg.capabilities
        assert ModelTask.TABLE_DETECTION in cfg.capabilities
        assert ModelTask.CLASSIFICATION in cfg.capabilities
        assert ModelTask.COMPONENT_DETECTION in cfg.capabilities
        # Should NOT have extraction tasks
        assert ModelTask.FIELD_EXTRACTION not in cfg.capabilities

    def test_florence2_custom_url(self):
        cfg = florence2_config(base_url="http://other:5000/v1", model_id="fl2-custom")
        assert cfg.base_url == "http://other:5000/v1"
        assert cfg.model_id == "fl2-custom"

    def test_qwen3vl_defaults(self):
        cfg = qwen3vl_config()
        assert cfg.name == "qwen3-vl"
        assert cfg.model_id == "qwen3-vl"
        assert ModelTask.FIELD_EXTRACTION in cfg.capabilities
        assert ModelTask.VERIFICATION in cfg.capabilities
        assert ModelTask.SCHEMA_GENERATION in cfg.capabilities
        assert ModelTask.GENERAL in cfg.capabilities
        # Should NOT have layout tasks
        assert ModelTask.LAYOUT_ANALYSIS not in cfg.capabilities

    def test_qwen3vl_custom(self):
        cfg = qwen3vl_config(model_id="qwen3-vl-8b")
        assert cfg.model_id == "qwen3-vl-8b"

    def test_florence_higher_priority_for_spatial(self):
        f = florence2_config()
        q = qwen3vl_config()
        assert f.priority > q.priority


# ──────────────────────────────────────────────────────────────────
# RoutingDecision Tests
# ──────────────────────────────────────────────────────────────────


class TestRoutingDecision:
    def test_basic_decision(self):
        cfg = ModelConfig(name="test", model_id="m")
        rd = RoutingDecision(
            model=cfg,
            task=ModelTask.GENERAL,
            reason="test",
        )
        assert rd.model.name == "test"
        assert rd.task == ModelTask.GENERAL
        assert rd.is_fallback is False

    def test_fallback_decision(self):
        cfg = ModelConfig(name="default", model_id="m")
        rd = RoutingDecision(
            model=cfg,
            task=ModelTask.CLASSIFICATION,
            reason="fallback",
            is_fallback=True,
        )
        assert rd.is_fallback is True

    def test_frozen(self):
        cfg = ModelConfig(name="test", model_id="m")
        rd = RoutingDecision(model=cfg, task=ModelTask.GENERAL, reason="r")
        with pytest.raises(AttributeError):
            rd.reason = "other"  # type: ignore


# ──────────────────────────────────────────────────────────────────
# ModelRouter — Registration
# ──────────────────────────────────────────────────────────────────


class TestRouterRegistration:
    def test_empty_router(self):
        router = ModelRouter()
        assert router.available_models == []
        assert router.default_model is None

    def test_register_model(self):
        router = ModelRouter()
        cfg = ModelConfig(name="test", model_id="m")
        router.register_model(cfg)
        assert router.get_model("test") is cfg

    def test_register_multiple(self):
        router = ModelRouter(models=[florence2_config(), qwen3vl_config()])
        assert len(router.available_models) == 2

    def test_unregister_model(self):
        router = ModelRouter(models=[florence2_config()])
        assert router.unregister_model("florence-2") is True
        assert router.get_model("florence-2") is None
        assert router.available_models == []

    def test_unregister_nonexistent(self):
        router = ModelRouter()
        assert router.unregister_model("nonexistent") is False

    def test_default_model(self):
        router = ModelRouter(
            models=[qwen3vl_config()],
            default_model_name="qwen3-vl",
        )
        assert router.default_model is not None
        assert router.default_model.name == "qwen3-vl"

    def test_disabled_model_not_in_available(self):
        cfg = ModelConfig(name="disabled", model_id="m", enabled=False)
        router = ModelRouter(models=[cfg])
        assert router.available_models == []

    def test_get_model_returns_none_for_missing(self):
        router = ModelRouter()
        assert router.get_model("nope") is None


# ──────────────────────────────────────────────────────────────────
# ModelRouter — Routing Logic
# ──────────────────────────────────────────────────────────────────


class TestRouterRouting:
    def test_routes_to_specialist(self):
        router = ModelRouter(
            models=[florence2_config(), qwen3vl_config()],
            default_model_name="qwen3-vl",
        )
        decision = router.route(ModelTask.LAYOUT_ANALYSIS)
        assert decision.model.name == "florence-2"
        assert decision.is_fallback is False

    def test_routes_extraction_to_qwen(self):
        router = ModelRouter(
            models=[florence2_config(), qwen3vl_config()],
            default_model_name="qwen3-vl",
        )
        decision = router.route(ModelTask.FIELD_EXTRACTION)
        assert decision.model.name == "qwen3-vl"
        assert decision.is_fallback is False

    def test_fallback_when_no_specialist(self):
        router = ModelRouter(
            models=[qwen3vl_config()],
            default_model_name="qwen3-vl",
        )
        # No florence-2, but layout_analysis requested → fallback to qwen3
        decision = router.route(ModelTask.LAYOUT_ANALYSIS)
        assert decision.model.name == "qwen3-vl"
        assert decision.is_fallback is True

    def test_highest_priority_wins(self):
        low = ModelConfig(
            name="low",
            model_id="low",
            capabilities={ModelTask.CLASSIFICATION},
            priority=1,
        )
        high = ModelConfig(
            name="high",
            model_id="high",
            capabilities={ModelTask.CLASSIFICATION},
            priority=10,
        )
        router = ModelRouter(models=[low, high])
        decision = router.route(ModelTask.CLASSIFICATION)
        assert decision.model.name == "high"

    def test_disabled_model_skipped(self):
        disabled = ModelConfig(
            name="specialist",
            model_id="s",
            capabilities={ModelTask.CLASSIFICATION},
            priority=100,
            enabled=False,
        )
        fallback = ModelConfig(
            name="default",
            model_id="d",
            capabilities={ModelTask.GENERAL},
            priority=1,
        )
        router = ModelRouter(
            models=[disabled, fallback],
            default_model_name="default",
        )
        decision = router.route(ModelTask.CLASSIFICATION)
        assert decision.model.name == "default"
        assert decision.is_fallback is True

    def test_raises_when_no_models(self):
        router = ModelRouter()
        with pytest.raises(ValueError, match="No models available"):
            router.route(ModelTask.GENERAL)

    def test_uses_first_available_when_no_default(self):
        cfg = ModelConfig(
            name="only-model",
            model_id="m",
            capabilities=set(),
        )
        router = ModelRouter(
            models=[cfg],
            default_model_name="nonexistent",
        )
        decision = router.route(ModelTask.GENERAL)
        assert decision.model.name == "only-model"
        assert decision.is_fallback is True


# ──────────────────────────────────────────────────────────────────
# ModelRouter — Agent Routing
# ──────────────────────────────────────────────────────────────────


class TestRouterAgentRouting:
    def _make_router(self) -> ModelRouter:
        return ModelRouter(
            models=[florence2_config(), qwen3vl_config()],
            default_model_name="qwen3-vl",
        )

    def test_analyzer_routes_to_florence(self):
        decision = self._make_router().route_for_agent("analyzer")
        assert decision.model.name == "florence-2"

    def test_layout_routes_to_florence(self):
        decision = self._make_router().route_for_agent("layout")
        assert decision.model.name == "florence-2"

    def test_table_detector_routes_to_florence(self):
        decision = self._make_router().route_for_agent("table_detector")
        assert decision.model.name == "florence-2"

    def test_extractor_routes_to_qwen(self):
        decision = self._make_router().route_for_agent("extractor")
        assert decision.model.name == "qwen3-vl"

    def test_validator_routes_to_qwen(self):
        decision = self._make_router().route_for_agent("validator")
        assert decision.model.name == "qwen3-vl"

    def test_unknown_agent_routes_to_general(self):
        decision = self._make_router().route_for_agent("unknown_agent")
        # GENERAL is handled by qwen3-vl
        assert decision.model.name == "qwen3-vl"

    def test_schema_generator_routes_to_qwen(self):
        decision = self._make_router().route_for_agent("schema_generator")
        assert decision.model.name == "qwen3-vl"

    def test_splitter_routes_to_florence(self):
        decision = self._make_router().route_for_agent("splitter")
        assert decision.model.name == "florence-2"


# ──────────────────────────────────────────────────────────────────
# ModelRouter — Statistics
# ──────────────────────────────────────────────────────────────────


class TestRouterStats:
    def test_initial_stats(self):
        router = ModelRouter(models=[qwen3vl_config()])
        stats = router.get_stats()
        assert stats["total_routes"] == 0
        assert stats["fallback_routes"] == 0
        assert stats["registered_models"] == 1
        assert stats["enabled_models"] == 1

    def test_stats_after_routing(self):
        router = ModelRouter(
            models=[florence2_config(), qwen3vl_config()],
            default_model_name="qwen3-vl",
        )
        router.route(ModelTask.LAYOUT_ANALYSIS)  # specialist
        router.route(ModelTask.FIELD_EXTRACTION)  # specialist
        router.route(ModelTask.HANDWRITING_RECOGNITION)  # fallback for florence-less

        stats = router.get_stats()
        assert stats["total_routes"] == 3

    def test_fallback_count(self):
        router = ModelRouter(
            models=[qwen3vl_config()],
            default_model_name="qwen3-vl",
        )
        router.route(ModelTask.LAYOUT_ANALYSIS)  # no florence → fallback
        router.route(ModelTask.TABLE_DETECTION)  # no florence → fallback
        stats = router.get_stats()
        assert stats["fallback_routes"] == 2

    def test_reset_stats(self):
        router = ModelRouter(
            models=[qwen3vl_config()],
            default_model_name="qwen3-vl",
        )
        router.route(ModelTask.GENERAL)
        router.reset_stats()
        stats = router.get_stats()
        assert stats["total_routes"] == 0
        assert stats["fallback_routes"] == 0


# ──────────────────────────────────────────────────────────────────
# BaseAgent Integration
# ──────────────────────────────────────────────────────────────────


class TestBaseAgentIntegration:
    def test_base_agent_accepts_model_router(self):
        from src.agents.base import BaseAgent

        class DummyAgent(BaseAgent):
            def process(self, state):
                return state

        router = ModelRouter(models=[qwen3vl_config()])
        mock_client = MagicMock()
        agent = DummyAgent(name="test", client=mock_client, model_router=router)
        assert agent.model_router is router

    def test_base_agent_default_no_router(self):
        from src.agents.base import BaseAgent

        class DummyAgent(BaseAgent):
            def process(self, state):
                return state

        mock_client = MagicMock()
        agent = DummyAgent(name="test", client=mock_client)
        assert agent.model_router is None


# ──────────────────────────────────────────────────────────────────
# Module Exports
# ──────────────────────────────────────────────────────────────────


class TestModuleExports:
    def test_imports_from_client(self):
        from src.client import (
            ModelConfig as MC,
            ModelRouter as MR,
            ModelTask as MT,
            RoutingDecision as RD,
            florence2_config as f2,
            qwen3vl_config as q3,
        )
        assert MC is ModelConfig
        assert MR is ModelRouter
        assert MT is ModelTask
        assert RD is RoutingDecision
        assert f2 is florence2_config
        assert q3 is qwen3vl_config
