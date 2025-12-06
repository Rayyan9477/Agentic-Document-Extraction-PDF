"""
Base agent class for document extraction agents.

Provides common functionality, error handling, and interfaces
that all extraction agents share.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Generic, TypeVar

from src.config import get_logger, get_settings
from src.client.lm_client import (
    LMStudioClient,
    VisionRequest,
    VisionResponse,
    LMClientError,
)
from src.pipeline.state import ExtractionState


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

    pass


class ExtractionError(AgentError):
    """Error during data extraction."""

    pass


class ValidationError(AgentError):
    """Error during validation."""

    pass


class OrchestrationError(AgentError):
    """Error during workflow orchestration."""

    pass


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
    ) -> None:
        """
        Initialize the base agent.

        Args:
            name: Agent name for logging and identification.
            client: Optional pre-configured LM Studio client.
        """
        self._name = name
        self._client = client or LMStudioClient()
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

        try:
            self._logger.debug(
                "sending_vision_request",
                agent=self._name,
                request_id=request.request_id,
            )

            response = self._client.send_vision_request(request)
            self._total_processing_ms += response.latency_ms

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

    def send_vision_request_with_json(
        self,
        image_data: str,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.1,
    ) -> dict[str, Any]:
        """
        Send a vision request and extract JSON response.

        Args:
            image_data: Base64-encoded image or data URI.
            prompt: User prompt for extraction.
            system_prompt: Optional system prompt.
            max_tokens: Maximum response tokens.
            temperature: Sampling temperature.

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
        return datetime.now(timezone.utc)

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
        end_time = datetime.now(timezone.utc)
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

        # Numeric comparison with tolerance
        if isinstance(v1, (int, float)) and isinstance(v2, (int, float)):
            if v1 == 0 and v2 == 0:
                return True
            if v1 == 0 or v2 == 0:
                return False
            # Allow 0.1% tolerance for floating point
            return abs(v1 - v2) / max(abs(v1), abs(v2)) < 0.001

        # List comparison
        if isinstance(v1, list) and isinstance(v2, list):
            if len(v1) != len(v2):
                return False
            return all(self._values_match(a, b) for a, b in zip(v1, v2))

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
