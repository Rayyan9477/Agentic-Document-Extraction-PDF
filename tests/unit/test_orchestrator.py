"""
Unit tests for the OrchestratorAgent.

Tests cover:
- Initialization and configuration
- Checkpointer creation (memory, sqlite, postgres)
- Routing logic (_determine_route, _make_routing_decision)
- Failure handling (_handle_failure)
- Workflow building (build_workflow)
- State transitions and edge cases
"""

from unittest.mock import MagicMock, patch

import pytest

from src.agents.orchestrator import (
    NODE_ANALYZE,
    NODE_COMPLETE,
    NODE_EXTRACT,
    NODE_HUMAN_REVIEW,
    NODE_PREPROCESS,
    NODE_RETRY,
    NODE_ROUTE,
    NODE_VALIDATE,
    ROUTE_COMPLETE,
    ROUTE_HUMAN_REVIEW,
    ROUTE_RETRY,
    CheckpointerType,
    OrchestratorAgent,
    create_extraction_workflow,
    generate_processing_id,
    generate_thread_id,
)
from src.pipeline.state import (
    ConfidenceLevel,
    ExtractionState,
    ExtractionStatus,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(**overrides: object) -> ExtractionState:
    """Build a minimal ExtractionState with overrides."""
    base: ExtractionState = {
        "processing_id": "test-proc-id",
        "pdf_path": "/tmp/test.pdf",
        "status": ExtractionStatus.VALIDATING.value,
        "current_step": "validation_complete",
        "overall_confidence": 0.90,
        "confidence_level": ConfidenceLevel.HIGH.value,
        "retry_count": 0,
        "errors": [],
        "warnings": [],
        "merged_extraction": {},
        "validation": {},
    }
    base.update(overrides)  # type: ignore[typeddict-item]
    return base


def _dummy_preprocess(state: ExtractionState) -> ExtractionState:
    return state


# ---------------------------------------------------------------------------
# TestOrchestratorInit
# ---------------------------------------------------------------------------


class TestOrchestratorInit:
    """Tests for OrchestratorAgent initialization."""

    def test_default_init(self) -> None:
        orch = OrchestratorAgent()
        assert orch.name == "orchestrator"
        assert orch._max_retries == 2
        assert orch._enable_checkpointing is True
        assert orch._checkpointer is not None  # memory checkpointer

    def test_disable_checkpointing(self) -> None:
        orch = OrchestratorAgent(enable_checkpointing=False)
        assert orch._checkpointer is None

    def test_custom_thresholds(self) -> None:
        orch = OrchestratorAgent(
            high_confidence_threshold=0.90,
            low_confidence_threshold=0.40,
            max_retries=5,
        )
        assert orch._high_confidence_threshold == 0.90
        assert orch._low_confidence_threshold == 0.40
        assert orch._max_retries == 5

    def test_memory_checkpointer_created(self) -> None:
        orch = OrchestratorAgent(
            enable_checkpointing=True,
            checkpointer_type=CheckpointerType.MEMORY,
        )
        assert orch._checkpointer is not None

    def test_string_checkpointer_type(self) -> None:
        orch = OrchestratorAgent(
            enable_checkpointing=True,
            checkpointer_type="memory",
        )
        assert orch._checkpointer is not None

    def test_postgres_checkpointer_requires_conn_string(self) -> None:
        with pytest.raises(Exception):
            OrchestratorAgent(
                enable_checkpointing=True,
                checkpointer_type=CheckpointerType.POSTGRES,
                postgres_conn_string=None,
            )


# ---------------------------------------------------------------------------
# TestRoutingLogic
# ---------------------------------------------------------------------------


class TestRoutingLogic:
    """Tests for routing decision logic."""

    def setup_method(self) -> None:
        self.orch = OrchestratorAgent(
            enable_checkpointing=False,
            max_retries=2,
            high_confidence_threshold=0.85,
            low_confidence_threshold=0.50,
        )

    def test_high_confidence_routes_to_complete(self) -> None:
        state = _make_state(
            confidence_level=ConfidenceLevel.HIGH.value,
            overall_confidence=0.92,
        )
        route = self.orch._determine_route(state)
        assert route == ROUTE_COMPLETE

    def test_low_confidence_routes_to_retry(self) -> None:
        state = _make_state(
            confidence_level=ConfidenceLevel.LOW.value,
            overall_confidence=0.30,
            retry_count=0,
        )
        route = self.orch._determine_route(state)
        assert route == ROUTE_RETRY

    def test_low_confidence_max_retries_routes_to_human_review(self) -> None:
        state = _make_state(
            confidence_level=ConfidenceLevel.LOW.value,
            overall_confidence=0.30,
            retry_count=2,
        )
        route = self.orch._determine_route(state)
        assert route == ROUTE_HUMAN_REVIEW

    def test_medium_confidence_retry_count_0_routes_to_retry(self) -> None:
        state = _make_state(
            confidence_level=ConfidenceLevel.MEDIUM.value,
            overall_confidence=0.65,
            retry_count=0,
        )
        route = self.orch._determine_route(state)
        assert route == ROUTE_RETRY

    def test_medium_confidence_max_retries_routes_to_complete(self) -> None:
        state = _make_state(
            confidence_level=ConfidenceLevel.MEDIUM.value,
            overall_confidence=0.65,
            retry_count=2,
        )
        route = self.orch._determine_route(state)
        assert route == ROUTE_COMPLETE

    def test_make_routing_decision_high_confidence(self) -> None:
        state = _make_state(
            confidence_level=ConfidenceLevel.HIGH.value,
            overall_confidence=0.95,
        )
        result = self.orch._make_routing_decision(state)
        assert result.get("status") == ExtractionStatus.COMPLETED.value

    def test_make_routing_decision_low_confidence_human_review(self) -> None:
        state = _make_state(
            confidence_level=ConfidenceLevel.LOW.value,
            overall_confidence=0.20,
            retry_count=2,
        )
        result = self.orch._make_routing_decision(state)
        assert result.get("status") == ExtractionStatus.HUMAN_REVIEW.value


# ---------------------------------------------------------------------------
# TestFailureHandling
# ---------------------------------------------------------------------------


class TestFailureHandling:
    """Tests for failure handling logic."""

    def setup_method(self) -> None:
        self.orch = OrchestratorAgent(
            enable_checkpointing=False,
            max_retries=2,
        )

    def test_failure_with_retries_remaining(self) -> None:
        state = _make_state(
            status=ExtractionStatus.FAILED.value,
            retry_count=0,
            errors=["Some error"],
        )
        result = self.orch._handle_failure(state)
        assert result.get("status") == ExtractionStatus.RETRYING.value

    def test_failure_at_max_retries(self) -> None:
        state = _make_state(
            status=ExtractionStatus.FAILED.value,
            retry_count=2,
            errors=["Some error"],
        )
        result = self.orch._handle_failure(state)
        assert result.get("status") == ExtractionStatus.HUMAN_REVIEW.value

    def test_failure_empty_errors(self) -> None:
        state = _make_state(
            status=ExtractionStatus.FAILED.value,
            retry_count=0,
            errors=[],
        )
        result = self.orch._handle_failure(state)
        assert result.get("status") == ExtractionStatus.RETRYING.value


# ---------------------------------------------------------------------------
# TestBuildWorkflow
# ---------------------------------------------------------------------------


class TestBuildWorkflow:
    """Tests for workflow building."""

    def test_build_legacy_workflow(self) -> None:
        orch = OrchestratorAgent(enable_checkpointing=False)
        analyzer = MagicMock()
        analyzer.process = MagicMock(return_value={})
        extractor = MagicMock()
        extractor.process = MagicMock(return_value={})
        validator = MagicMock()
        validator.process = MagicMock(return_value={})

        workflow = orch.build_workflow(
            preprocess_fn=_dummy_preprocess,
            analyzer=analyzer,
            extractor=extractor,
            validator=validator,
        )

        assert workflow is not None
        assert orch._analyzer is analyzer
        assert orch._extractor is extractor
        assert orch._validator is validator

    def test_build_workflow_with_vlm_first_agents(self) -> None:
        orch = OrchestratorAgent(enable_checkpointing=False)
        analyzer = MagicMock()
        extractor = MagicMock()
        validator = MagicMock()
        layout = MagicMock()
        component = MagicMock()
        schema = MagicMock()

        workflow = orch.build_workflow(
            preprocess_fn=_dummy_preprocess,
            analyzer=analyzer,
            extractor=extractor,
            validator=validator,
            layout_agent=layout,
            component_agent=component,
            schema_agent=schema,
        )

        assert workflow is not None
        assert orch._layout_agent is layout
        assert orch._component_agent is component
        assert orch._schema_agent is schema

    def test_compile_workflow(self) -> None:
        orch = OrchestratorAgent(enable_checkpointing=False)
        analyzer = MagicMock()
        extractor = MagicMock()
        validator = MagicMock()

        orch.build_workflow(
            preprocess_fn=_dummy_preprocess,
            analyzer=analyzer,
            extractor=extractor,
            validator=validator,
        )

        compiled = orch.compile_workflow()
        assert compiled is not None


# ---------------------------------------------------------------------------
# TestProcess
# ---------------------------------------------------------------------------


class TestProcess:
    """Tests for the process() method."""

    def test_process_validating_high_confidence(self) -> None:
        orch = OrchestratorAgent(enable_checkpointing=False)
        state = _make_state(
            status=ExtractionStatus.VALIDATING.value,
            confidence_level=ConfidenceLevel.HIGH.value,
            overall_confidence=0.95,
        )
        result = orch.process(state)
        assert result.get("status") == ExtractionStatus.COMPLETED.value

    def test_process_failed_state(self) -> None:
        orch = OrchestratorAgent(enable_checkpointing=False)
        state = _make_state(
            status=ExtractionStatus.FAILED.value,
            retry_count=0,
            errors=["Test error"],
        )
        result = orch.process(state)
        assert result.get("status") == ExtractionStatus.RETRYING.value


# ---------------------------------------------------------------------------
# TestHelperFunctions
# ---------------------------------------------------------------------------


class TestHelperFunctions:
    """Tests for module-level helper functions."""

    def test_generate_processing_id_format(self) -> None:
        pid = generate_processing_id()
        assert isinstance(pid, str)
        assert len(pid) > 10  # UUID-based, should be long

    def test_generate_processing_id_unique(self) -> None:
        ids = {generate_processing_id() for _ in range(50)}
        assert len(ids) == 50  # All unique

    def test_generate_thread_id_deterministic(self) -> None:
        tid1 = generate_thread_id("/tmp/a.pdf", "proc-1")
        tid2 = generate_thread_id("/tmp/a.pdf", "proc-1")
        assert tid1 == tid2

    def test_generate_thread_id_varies_with_inputs(self) -> None:
        tid1 = generate_thread_id("/tmp/a.pdf", "proc-1")
        tid2 = generate_thread_id("/tmp/b.pdf", "proc-1")
        assert tid1 != tid2

    @patch("src.agents.orchestrator.LMStudioClient")
    def test_create_extraction_workflow_returns_tuple(self, mock_client_cls: MagicMock) -> None:
        mock_client_cls.return_value = MagicMock()

        orch, compiled = create_extraction_workflow(
            preprocess_fn=_dummy_preprocess,
            client=MagicMock(),
            enable_checkpointing=False,
        )

        assert isinstance(orch, OrchestratorAgent)
        assert compiled is not None
