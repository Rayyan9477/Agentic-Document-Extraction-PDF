"""
Multi-model routing for task-specific model selection.

Routes VLM requests to the most suitable model based on task type.
Supports Florence-2 for layout/detection tasks and Qwen3-VL for
text-heavy extraction, with automatic fallback to the default model.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.config import get_logger


logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Task taxonomy
# ---------------------------------------------------------------------------


class ModelTask(str, Enum):
    """Categories of VLM tasks used for model routing.

    Each task maps to the kind of work the model will perform,
    enabling the router to select the best model.
    """

    # Document-level analysis
    CLASSIFICATION = "classification"
    LAYOUT_ANALYSIS = "layout_analysis"

    # Component-level detection
    TABLE_DETECTION = "table_detection"
    COMPONENT_DETECTION = "component_detection"

    # Field-level extraction
    FIELD_EXTRACTION = "field_extraction"
    VERIFICATION = "verification"

    # Specialised
    SCHEMA_GENERATION = "schema_generation"
    HANDWRITING_RECOGNITION = "handwriting_recognition"

    # Fallback
    GENERAL = "general"


# ---------------------------------------------------------------------------
# Model configuration
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ModelConfig:
    """Configuration for a single model endpoint.

    Attributes:
        name: Human-readable model name (e.g. ``"florence-2"``).
        model_id: Model identifier used in API calls.
        base_url: Base URL for the model's API endpoint.
        capabilities: Set of ``ModelTask`` values this model handles well.
        priority: Higher priority wins when multiple models match.
        max_tokens: Default max tokens for this model.
        temperature: Default temperature for this model.
        enabled: Whether the model is active.
    """

    name: str
    model_id: str
    base_url: str = "http://localhost:1234/v1"
    capabilities: set[ModelTask] = field(default_factory=set)
    priority: int = 0
    max_tokens: int = 4096
    temperature: float = 0.1
    enabled: bool = True

    def supports(self, task: ModelTask) -> bool:
        """Check if this model supports a given task."""
        return task in self.capabilities

    def to_dict(self) -> dict[str, Any]:
        """Serialise to dictionary."""
        return {
            "name": self.name,
            "model_id": self.model_id,
            "base_url": self.base_url,
            "capabilities": sorted(c.value for c in self.capabilities),
            "priority": self.priority,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "enabled": self.enabled,
        }


# ---------------------------------------------------------------------------
# Routing result
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class RoutingDecision:
    """Outcome of a routing decision.

    Attributes:
        model: Selected ``ModelConfig``.
        task: Task that was routed.
        reason: Human-readable reason for the selection.
        is_fallback: Whether the default model was used as a fallback.
    """

    model: ModelConfig
    task: ModelTask
    reason: str
    is_fallback: bool = False


# ---------------------------------------------------------------------------
# Pre-built model profiles
# ---------------------------------------------------------------------------


def florence2_config(
    base_url: str = "http://localhost:1234/v1",
    model_id: str = "florence-2",
) -> ModelConfig:
    """Create a default Florence-2 configuration.

    Florence-2 excels at spatial understanding, layout analysis, and
    visual grounding tasks.
    """
    return ModelConfig(
        name="florence-2",
        model_id=model_id,
        base_url=base_url,
        capabilities={
            ModelTask.CLASSIFICATION,
            ModelTask.LAYOUT_ANALYSIS,
            ModelTask.TABLE_DETECTION,
            ModelTask.COMPONENT_DETECTION,
        },
        priority=10,
        max_tokens=2048,
        temperature=0.1,
    )


def qwen3vl_config(
    base_url: str = "http://localhost:1234/v1",
    model_id: str = "qwen3-vl",
) -> ModelConfig:
    """Create a default Qwen3-VL configuration.

    Qwen3-VL excels at text extraction, JSON generation, complex
    reasoning, and schema generation.
    """
    return ModelConfig(
        name="qwen3-vl",
        model_id=model_id,
        base_url=base_url,
        capabilities={
            ModelTask.FIELD_EXTRACTION,
            ModelTask.VERIFICATION,
            ModelTask.SCHEMA_GENERATION,
            ModelTask.HANDWRITING_RECOGNITION,
            ModelTask.GENERAL,
        },
        priority=5,
        max_tokens=4096,
        temperature=0.1,
    )


# ---------------------------------------------------------------------------
# ModelRouter
# ---------------------------------------------------------------------------


class ModelRouter:
    """Routes VLM tasks to the most suitable model.

    The router maintains a registry of ``ModelConfig`` objects and,
    given a ``ModelTask``, selects the best matching model based on
    capabilities and priority.  If no specialist model matches, the
    router falls back to the configured default model.

    Usage::

        router = ModelRouter(
            models=[florence2_config(), qwen3vl_config()],
            default_model_name="qwen3-vl",
        )
        decision = router.route(ModelTask.LAYOUT_ANALYSIS)
        # decision.model.name == "florence-2"
    """

    def __init__(
        self,
        models: list[ModelConfig] | None = None,
        default_model_name: str = "qwen3-vl",
    ) -> None:
        """Initialise the router.

        Args:
            models: List of available model configurations.
            default_model_name: Name of the fallback model.
        """
        self._models: dict[str, ModelConfig] = {}
        self._default_name = default_model_name
        self._route_count = 0
        self._fallback_count = 0
        self._logger = get_logger("client.model_router")

        for m in models or []:
            self.register_model(m)

    # ------------------------------------------------------------------
    # Model management
    # ------------------------------------------------------------------

    def register_model(self, config: ModelConfig) -> None:
        """Register a model configuration.

        Args:
            config: Model configuration to register.
        """
        self._models[config.name] = config
        self._logger.info(
            "model_registered",
            name=config.name,
            model_id=config.model_id,
            capabilities=sorted(c.value for c in config.capabilities),
        )

    def unregister_model(self, name: str) -> bool:
        """Remove a model from the registry.

        Args:
            name: Model name to remove.

        Returns:
            True if removed, False if not found.
        """
        if name in self._models:
            del self._models[name]
            return True
        return False

    def get_model(self, name: str) -> ModelConfig | None:
        """Get a model by name."""
        return self._models.get(name)

    @property
    def available_models(self) -> list[ModelConfig]:
        """List enabled models."""
        return [m for m in self._models.values() if m.enabled]

    @property
    def default_model(self) -> ModelConfig | None:
        """Get the default fallback model."""
        return self._models.get(self._default_name)

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def route(self, task: ModelTask) -> RoutingDecision:
        """Select the best model for a task.

        Args:
            task: The VLM task to route.

        Returns:
            ``RoutingDecision`` with the selected model.

        Raises:
            ValueError: If no model is available at all.
        """
        self._route_count += 1

        # Find enabled models that support this task
        candidates = [
            m for m in self._models.values()
            if m.enabled and m.supports(task)
        ]

        if candidates:
            # Pick highest priority
            best = max(candidates, key=lambda m: m.priority)
            self._logger.debug(
                "model_routed",
                task=task.value,
                model=best.name,
                candidates=len(candidates),
            )
            return RoutingDecision(
                model=best,
                task=task,
                reason=f"Best match for {task.value} (priority={best.priority})",
            )

        # Fallback to default
        default = self.default_model
        if default and default.enabled:
            self._fallback_count += 1
            self._logger.info(
                "model_fallback",
                task=task.value,
                default_model=default.name,
            )
            return RoutingDecision(
                model=default,
                task=task,
                reason=f"No specialist for {task.value}, using default {default.name}",
                is_fallback=True,
            )

        # No models at all
        enabled = [m for m in self._models.values() if m.enabled]
        if enabled:
            fallback = enabled[0]
            self._fallback_count += 1
            return RoutingDecision(
                model=fallback,
                task=task,
                reason=f"No specialist or default, using first available: {fallback.name}",
                is_fallback=True,
            )

        raise ValueError(
            f"No models available to handle task {task.value}. "
            f"Register at least one model."
        )

    def route_for_agent(self, agent_name: str) -> RoutingDecision:
        """Route based on agent name using a standard mapping.

        Args:
            agent_name: Name of the agent (e.g. ``"analyzer"``, ``"extractor"``).

        Returns:
            ``RoutingDecision`` for the agent's primary task.
        """
        task_map: dict[str, ModelTask] = {
            "analyzer": ModelTask.CLASSIFICATION,
            "layout": ModelTask.LAYOUT_ANALYSIS,
            "component_detector": ModelTask.COMPONENT_DETECTION,
            "table_detector": ModelTask.TABLE_DETECTION,
            "extractor": ModelTask.FIELD_EXTRACTION,
            "validator": ModelTask.VERIFICATION,
            "schema_generator": ModelTask.SCHEMA_GENERATION,
            "schema_proposal": ModelTask.SCHEMA_GENERATION,
            "splitter": ModelTask.CLASSIFICATION,
        }

        task = task_map.get(agent_name, ModelTask.GENERAL)
        return self.route(task)

    # ------------------------------------------------------------------
    # Statistics
    # ------------------------------------------------------------------

    def get_stats(self) -> dict[str, Any]:
        """Get routing statistics."""
        return {
            "total_routes": self._route_count,
            "fallback_routes": self._fallback_count,
            "registered_models": len(self._models),
            "enabled_models": len(self.available_models),
            "default_model": self._default_name,
        }

    def reset_stats(self) -> None:
        """Reset routing counters."""
        self._route_count = 0
        self._fallback_count = 0
