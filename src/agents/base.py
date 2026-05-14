"""
Base agent class for document extraction agents.

Provides common functionality, error handling, and interfaces
that all extraction agents share.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from src.client.backends.protocol import VLMRole
from src.client.lm_client import (
    LMClientError,
    LMStudioClient,
    VisionRequest,
    VisionResponse,
)
from src.config import get_logger, get_settings
from src.pipeline.state import ExtractionState


if TYPE_CHECKING:
    from src.client.constrained import DecodingTrace


logger = get_logger(__name__)

T = TypeVar("T")


class AgentError(Exception):
    """Base exception for agent errors."""

    def __init__(
        self,
        message: str,
        agent_name: str = "",
        recoverable: bool = True,
        details: dict[str, Any] | None = None,
    ) -> None:
        """
        Initialize agent error.

        Args:
            message: Error message.
            agent_name: Name of the agent that raised the error.
            recoverable: Whether the error is recoverable.
            details: Additional error details.
        """
        super().__init__(message)
        self.agent_name = agent_name
        self.recoverable = recoverable
        self.details = details or {}


class AnalysisError(AgentError):
    """Error during document analysis."""


class ExtractionError(AgentError):
    """Error during data extraction."""


class ValidationError(AgentError):
    """Error during validation."""


class OrchestrationError(AgentError):
    """Error during workflow orchestration."""


@dataclass(slots=True)
class AgentResult(Generic[T]):
    """
    Result container for agent operations.

    Attributes:
        success: Whether the operation succeeded.
        data: Result data if successful.
        error: Error message if failed.
        agent_name: Name of the agent.
        operation: Name of the operation.
        vlm_calls: Number of VLM calls made.
        processing_time_ms: Processing time in milliseconds.
        metadata: Additional result metadata.
    """

    success: bool
    data: T | None = None
    error: str | None = None
    agent_name: str = ""
    operation: str = ""
    vlm_calls: int = 0
    processing_time_ms: int = 0
    metadata: dict[str, Any] = field(default_factory=dict)

    @classmethod
    def ok(
        cls,
        data: T,
        agent_name: str = "",
        operation: str = "",
        vlm_calls: int = 0,
        processing_time_ms: int = 0,
        **metadata: Any,
    ) -> "AgentResult[T]":
        """Create a successful result."""
        return cls(
            success=True,
            data=data,
            error=None,
            agent_name=agent_name,
            operation=operation,
            vlm_calls=vlm_calls,
            processing_time_ms=processing_time_ms,
            metadata=metadata,
        )

    @classmethod
    def fail(
        cls,
        error: str,
        agent_name: str = "",
        operation: str = "",
        **metadata: Any,
    ) -> "AgentResult[T]":
        """Create a failed result."""
        return cls(
            success=False,
            data=None,
            error=error,
            agent_name=agent_name,
            operation=operation,
            metadata=metadata,
        )

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary representation."""
        return {
            "success": self.success,
            "data": self.data,
            "error": self.error,
            "agent_name": self.agent_name,
            "operation": self.operation,
            "vlm_calls": self.vlm_calls,
            "processing_time_ms": self.processing_time_ms,
            "metadata": self.metadata,
        }


class BaseAgent(ABC):
    """
    Abstract base class for all extraction agents.

    Provides common functionality including:
    - VLM client management
    - Logging and metrics
    - Error handling
    - State access utilities
    """

    def __init__(
        self,
        name: str,
        client: LMStudioClient | None = None,
        model_router: Any | None = None,
    ) -> None:
        """
        Initialize the base agent.

        Args:
            name: Agent name for logging and identification.
            client: Optional pre-configured LM Studio client.
            model_router: Optional ModelRouter for multi-model routing
                (Phase 3C). When set, ``send_vision_request`` can route
                requests to task-appropriate models.
        """
        self._name = name
        self._client = client or LMStudioClient()
        self._model_router = model_router
        self._logger = get_logger(f"agent.{name}")
        self._settings = get_settings()
        self._vlm_calls = 0
        self._total_processing_ms = 0

        self._logger.info(f"{name}_agent_initialized")

    @property
    def name(self) -> str:
        """Get agent name."""
        return self._name

    @property
    def vlm_calls(self) -> int:
        """Get total VLM calls made by this agent."""
        return self._vlm_calls

    @property
    def total_processing_ms(self) -> int:
        """Get total processing time in milliseconds."""
        return self._total_processing_ms

    @property
    def model_router(self) -> Any | None:
        """Get the model router (Phase 3C), if configured."""
        return self._model_router

    def set_model_router(self, router: Any | None) -> None:
        """
        Attach (or detach) a ``ModelRouter`` after construction.

        The router is consulted in ``send_vision_request`` to choose a
        task-appropriate model per call. Use this from the workflow
        factory so agent subclasses don't have to thread the router
        through their own ``__init__`` signatures.
        """
        self._model_router = router

    @abstractmethod
    def process(self, state: ExtractionState) -> ExtractionState:
        """
        Process the extraction state.

        This is the main entry point for the agent in the LangGraph workflow.
        Each agent must implement this method to define its processing logic.

        Args:
            state: Current extraction state.

        Returns:
            Updated extraction state.
        """
        raise NotImplementedError

    def send_vision_request(
        self,
        image_data: str,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.1,
        *,
        role: VLMRole = VLMRole.PRIMARY,
    ) -> VisionResponse:
        """
        Send a vision request to the VLM.

        Wraps the LM client with agent-specific logging and metrics.

        Args:
            image_data: Base64-encoded image or data URI.
            prompt: User prompt for extraction.
            system_prompt: Optional system prompt.
            max_tokens: Maximum response tokens.
            temperature: Sampling temperature.
            role: Logical VLM role for this call. Defaults to ``PRIMARY``
                so existing call-sites are unchanged. Phase 2 agents
                (``extractor_pass2``) and Phase 3 agents (``critic``)
                pass ``SECONDARY``/``CRITIC`` to drive the dual-VLM
                topology. The role flows through ``ModelRouter`` and is
                resolved to a backend endpoint at request time. When no
                ``VLMBackend`` is wired (legacy default), the call falls
                back to ``self._client`` exactly as before.

        Returns:
            VisionResponse from the VLM.

        Raises:
            AgentError: If the request fails.
        """
        self._vlm_calls += 1

        request = VisionRequest(
            image_data=image_data,
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # WS-2: consult the model router (Phase 3C) when configured.
        # The router maps the calling agent's name to a ``ModelTask`` and
        # returns a ``RoutingDecision`` whose ``model.model_id`` is forwarded
        # to LMStudioClient as a per-request override. When no router is
        # configured, behaviour is unchanged: the client's default model is
        # used.
        model_override: str | None = None
        if self._model_router is not None:
            try:
                decision = self._model_router.route_for_agent(self._name)
                model_override = decision.model.model_id
                self._logger.debug(
                    "model_routed",
                    agent=self._name,
                    selected_model=decision.model.name,
                    is_fallback=decision.is_fallback,
                    reason=decision.reason,
                )
            except Exception as exc:
                # Routing is best-effort — never fail the extraction over a
                # routing decision. Fall through to the default client/model.
                self._logger.warning(
                    "model_routing_failed_using_default",
                    agent=self._name,
                    error=str(exc),
                )

        # WS-7: wrap the VLM call in an observability span. The
        # dispatcher is a no-op when no sinks are configured, so this
        # is safe in every code path — the span context just yields
        # None and the call proceeds unchanged.
        try:
            from src.monitoring.observability import get_dispatcher

            _dispatcher = get_dispatcher()
        except Exception:  # pragma: no cover - defensive
            _dispatcher = None

        # V3 Phase 8 — VLM queue-depth gate. ``vlm_queue_slot`` is a
        # no-op when ``settings.vlm.max_concurrent_requests`` is 0
        # (the default), so existing tests are unaffected. Production
        # deployments set the capacity to bound concurrent VLM calls
        # per Python process and protect the GPU backend from
        # thundering-herd OOMs.
        from src.client.backends.queue_depth import (
            configure_from_settings as _qd_configure,
            vlm_queue_slot,
        )

        _qd_configure()  # idempotent; picks up settings if changed

        try:
            self._logger.debug(
                "sending_vision_request",
                agent=self._name,
                request_id=request.request_id,
                model_override=model_override,
                role=role.value,
            )

            if _dispatcher is not None and _dispatcher.is_active:
                with vlm_queue_slot(), _dispatcher.start_span(
                    "vlm.request",
                    agent=self._name,
                    request_id=request.request_id,
                    model=model_override,
                    role=role.value,
                ):
                    response = self._client.send_vision_request(
                        request, model=model_override
                    )
            else:
                with vlm_queue_slot():
                    response = self._client.send_vision_request(
                        request, model=model_override
                    )

            self._total_processing_ms += response.latency_ms

            if _dispatcher is not None and _dispatcher.is_active:
                _dispatcher.record_llm_call(
                    agent=self._name,
                    model=model_override,
                    latency_ms=response.latency_ms,
                    request_id=request.request_id,
                )
                # V3 Phase 6 — canonical PostHog event for every VLM
                # call. PostHog's ``capture`` is async/batched in the
                # SDK, so emit overhead is negligible. Emit only when
                # the dispatcher has at least one sink configured.
                try:
                    from src.monitoring.observability import (
                        EVENT_VLM_CALLED,
                        _read_trace_id_from_context,
                    )

                    _dispatcher.emit_event(
                        EVENT_VLM_CALLED,
                        {
                            "agent": self._name,
                            "model": model_override,
                            "role": role.value,
                            "latency_ms": response.latency_ms,
                            "has_json": response.has_json,
                            "request_id": request.request_id,
                            "trace_id": _read_trace_id_from_context(),
                        },
                    )
                except Exception:  # pragma: no cover - defensive
                    pass

            self._logger.debug(
                "vision_request_complete",
                agent=self._name,
                request_id=request.request_id,
                latency_ms=response.latency_ms,
                has_json=response.has_json,
            )

            return response

        except LMClientError as e:
            self._logger.error(
                "vision_request_failed",
                agent=self._name,
                error=str(e),
            )
            raise AgentError(
                f"VLM request failed: {e}",
                agent_name=self._name,
                recoverable=True,
            ) from e

    def send_vision_request_with_schema(
        self,
        image_data: str,
        prompt: str,
        schema: type,
        system_prompt: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.1,
        *,
        role: VLMRole = VLMRole.PRIMARY,
    ) -> tuple[dict[str, Any], "DecodingTrace"]:
        """V3 Phase 1: schema-bound vision request via constrained decoding.

        The agent supplies a Pydantic ``BaseModel`` subclass; this
        method converts it to JSON Schema and binds it at decode time
        through the underlying client. The decoder cannot emit JSON
        that violates the schema — the entire class of malformed-output
        failures becomes structurally impossible.

        Implementation routes through ``self._client`` (the
        constructor-injected ``LMStudioClient``) so existing test
        injection points continue to work. The schema → ``response_format``
        translation is the same shape ``LMStudioBackend`` uses; vLLM
        deployments would substitute ``extra_body`` via a different
        backend resolution in Phase 2.

        Args:
            image_data: Base64-encoded image or data URI.
            prompt: User prompt for extraction.
            schema: Pydantic ``BaseModel`` subclass describing the
                expected response shape.
            system_prompt: Optional system prompt.
            max_tokens: Maximum response tokens.
            temperature: Sampling temperature.
            role: Logical VLM role; see :meth:`send_vision_request`.

        Returns:
            ``(parsed_json, trace)`` where ``parsed_json`` is the
            response dict and ``trace`` carries decoding telemetry
            (consumed by ``ConfidenceCalibrator`` and Phoenix spans).

        Raises:
            AgentError: When the constrained call fails or returns
                non-JSON despite the schema constraint.
        """
        # Lazy imports to avoid a circular import via observability.
        from src.client.constrained import DecodingTrace as _DecodingTrace

        self._vlm_calls += 1
        request = VisionRequest(
            image_data=image_data,
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
        )

        # Build the OpenAI-style response_format from the Pydantic schema.
        # Same shape LMStudioBackend would build; centralised here so
        # tests that mock ``self._client.send_vision_request`` directly
        # observe the schema kwarg without needing the backend factory.
        schema_dict = schema.model_json_schema()
        response_format = {
            "type": "json_schema",
            "json_schema": {"name": "veridoc", "schema": schema_dict},
        }

        # Optional model routing — same logic as send_vision_request.
        model_override: str | None = None
        if self._model_router is not None:
            try:
                decision = self._model_router.route_for_agent(self._name)
                model_override = decision.model.model_id
            except Exception as exc:
                self._logger.warning(
                    "model_routing_failed_using_default",
                    agent=self._name,
                    error=str(exc),
                )

        try:
            from src.monitoring.observability import get_dispatcher

            _dispatcher = get_dispatcher()
        except Exception:  # pragma: no cover - defensive
            _dispatcher = None

        # V3 Phase 8 — VLM queue-depth gate (see send_vision_request).
        from src.client.backends.queue_depth import (
            configure_from_settings as _qd_configure,
            vlm_queue_slot,
        )

        _qd_configure()

        try:
            self._logger.debug(
                "sending_constrained_vision_request",
                agent=self._name,
                request_id=request.request_id,
                schema=schema.__name__,
                role=role.value,
                model_override=model_override,
            )
            if _dispatcher is not None and _dispatcher.is_active:
                with vlm_queue_slot(), _dispatcher.start_span(
                    "vlm.request_constrained",
                    agent=self._name,
                    request_id=request.request_id,
                    schema=schema.__name__,
                    role=role.value,
                ):
                    response = self._client.send_vision_request(
                        request,
                        model=model_override,
                        response_format=response_format,
                    )
            else:
                with vlm_queue_slot():
                    response = self._client.send_vision_request(
                        request,
                        model=model_override,
                        response_format=response_format,
                    )

            self._total_processing_ms += response.latency_ms

            if _dispatcher is not None and _dispatcher.is_active:
                _dispatcher.record_llm_call(
                    agent=self._name,
                    model=model_override,
                    latency_ms=response.latency_ms,
                    request_id=request.request_id,
                )

            trace = _DecodingTrace(
                backend_name="lm_studio",
                role=role,
                model_id=model_override or "",
                schema_name=schema.__name__,
                latency_ms=response.latency_ms,
                tokens_in=response.prompt_tokens,
                tokens_out=response.completion_tokens,
                schema_enforced=True,
            )

            self._logger.debug(
                "constrained_vision_request_complete",
                agent=self._name,
                request_id=request.request_id,
                latency_ms=response.latency_ms,
                schema=schema.__name__,
            )

            if response.parsed_json is None:
                # Constrained decoding makes this branch unreachable on
                # cooperating backends; surface as an explicit error so
                # any non-cooperating backend fails loud.
                raise AgentError(
                    "Schema-bound VLM request returned non-JSON content",
                    agent_name=self._name,
                    recoverable=True,
                    details={
                        "schema": schema.__name__,
                        "content_preview": response.content[:200],
                    },
                )
            return response.parsed_json, trace

        except LMClientError as e:
            self._logger.error(
                "constrained_vision_request_failed",
                agent=self._name,
                error=str(e),
            )
            raise AgentError(
                f"VLM request failed: {e}",
                agent_name=self._name,
                recoverable=True,
            ) from e

    def send_vision_request_with_json(
        self,
        image_data: str,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.1,
        *,
        role: VLMRole = VLMRole.PRIMARY,
    ) -> dict[str, Any]:
        """
        Send a vision request and extract JSON response.

        Args:
            image_data: Base64-encoded image or data URI.
            prompt: User prompt for extraction.
            system_prompt: Optional system prompt.
            max_tokens: Maximum response tokens.
            temperature: Sampling temperature.
            role: Logical VLM role; see :meth:`send_vision_request`.

        Returns:
            Parsed JSON from response.

        Raises:
            AgentError: If request fails or JSON extraction fails.
        """
        response = self.send_vision_request(
            image_data=image_data,
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            role=role,
        )

        if not response.has_json:
            self._logger.warning(
                "json_extraction_failed",
                agent=self._name,
                content_length=len(response.content),
            )
            raise AgentError(
                "Failed to extract JSON from VLM response",
                agent_name=self._name,
                recoverable=True,
                details={"content_preview": response.content[:500]},
            )

        return response.parsed_json  # type: ignore

    def log_operation_start(self, operation: str, **context: Any) -> datetime:
        """
        Log the start of an operation.

        Args:
            operation: Name of the operation.
            **context: Additional context to log.

        Returns:
            Start timestamp for duration calculation.
        """
        self._logger.info(
            f"{operation}_started",
            agent=self._name,
            **context,
        )
        return datetime.now(UTC)

    def log_operation_complete(
        self,
        operation: str,
        start_time: datetime,
        success: bool = True,
        **context: Any,
    ) -> int:
        """
        Log the completion of an operation.

        Args:
            operation: Name of the operation.
            start_time: When the operation started.
            success: Whether the operation succeeded.
            **context: Additional context to log.

        Returns:
            Duration in milliseconds.
        """
        end_time = datetime.now(UTC)
        duration_ms = int((end_time - start_time).total_seconds() * 1000)

        log_method = self._logger.info if success else self._logger.error
        log_method(
            f"{operation}_complete",
            agent=self._name,
            success=success,
            duration_ms=duration_ms,
            **context,
        )

        return duration_ms

    def extract_field_value(
        self,
        data: dict[str, Any],
        field_path: str,
        default: Any = None,
    ) -> Any:
        """
        Extract a value from nested dictionary using dot notation.

        Args:
            data: Dictionary to extract from.
            field_path: Dot-separated path (e.g., "fields.patient_name.value").
            default: Default value if path not found.

        Returns:
            Extracted value or default.
        """
        keys = field_path.split(".")
        current = data

        for key in keys:
            if isinstance(current, dict) and key in current:
                current = current[key]
            else:
                return default

        return current

    def merge_field_results(
        self,
        pass1: dict[str, Any],
        pass2: dict[str, Any],
    ) -> dict[str, Any]:
        """
        Merge results from dual-pass extraction.

        Compares field values from both passes and calculates confidence
        based on agreement.

        Args:
            pass1: First pass extraction results.
            pass2: Second pass extraction results.

        Returns:
            Merged results with confidence adjustments.
        """
        merged = {}
        all_fields = set(pass1.keys()) | set(pass2.keys())

        for field_name in all_fields:
            v1 = pass1.get(field_name, {})
            v2 = pass2.get(field_name, {})

            value1 = v1.get("value") if isinstance(v1, dict) else v1
            value2 = v2.get("value") if isinstance(v2, dict) else v2

            conf1 = v1.get("confidence", 0.0) if isinstance(v1, dict) else 0.5
            conf2 = v2.get("confidence", 0.0) if isinstance(v2, dict) else 0.5

            # Determine agreement
            passes_agree = self._values_match(value1, value2)

            # Calculate merged confidence
            if passes_agree:
                # Agreement boosts confidence
                merged_confidence = min(1.0, (conf1 + conf2) / 2 + 0.1)
                merged_value = value1 if value1 is not None else value2
            else:
                # Disagreement reduces confidence
                merged_confidence = min(conf1, conf2) * 0.7
                # Use the higher confidence value
                merged_value = value1 if conf1 >= conf2 else value2

            merged[field_name] = {
                "value": merged_value,
                "confidence": merged_confidence,
                "pass1_value": value1,
                "pass2_value": value2,
                "passes_agree": passes_agree,
                "location": v1.get("location") or v2.get("location", ""),
            }

        return merged

    def _values_match(self, v1: Any, v2: Any) -> bool:
        """
        Check if two values match for dual-pass comparison.

        Handles various value types and allows for minor formatting differences.

        Args:
            v1: First value.
            v2: Second value.

        Returns:
            True if values are considered matching.
        """
        # Both null
        if v1 is None and v2 is None:
            return True

        # One null
        if v1 is None or v2 is None:
            return False

        # String comparison (case-insensitive, whitespace-normalized)
        if isinstance(v1, str) and isinstance(v2, str):
            norm1 = " ".join(v1.lower().split())
            norm2 = " ".join(v2.lower().split())
            return norm1 == norm2

        # Numeric comparison with tolerance.
        # WS-2: tightened from 0.1% to 0.01% — at 0.1% a billing amount of
        # $1,000.00 vs $1,000.99 would be treated as a match, which is not
        # acceptable for medical claim totals. 0.01% is still permissive
        # enough for legitimate float-rounding ($100.00 vs $100.01 ≈ 0.01%
        # boundary), while catching cents-level VLM disagreements.
        if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
            if v1 == 0 and v2 == 0:
                return True
            if v1 == 0 or v2 == 0:
                return False
            return abs(v1 - v2) / max(abs(v1), abs(v2)) < 0.0001

        # List comparison
        if isinstance(v1, list) and isinstance(v2, list):
            if len(v1) != len(v2):
                return False
            return all(self._values_match(a, b) for a, b in zip(v1, v2, strict=False))

        # Direct equality for other types
        return v1 == v2

    def reset_metrics(self) -> None:
        """Reset agent metrics."""
        self._vlm_calls = 0
        self._total_processing_ms = 0

    def get_metrics(self) -> dict[str, Any]:
        """Get agent metrics."""
        return {
            "agent_name": self._name,
            "vlm_calls": self._vlm_calls,
            "total_processing_ms": self._total_processing_ms,
        }
