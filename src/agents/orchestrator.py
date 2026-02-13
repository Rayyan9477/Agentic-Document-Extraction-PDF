"""
Orchestrator Agent for coordinating the document extraction workflow.

Responsible for:
- Building and managing the LangGraph StateGraph workflow
- Coordinating state transitions between agents
- Handling checkpointing for fault tolerance
- Implementing retry logic for failed extractions
- Routing decisions based on validation confidence

Checkpointing Options:
- memory: In-memory checkpointing (development/testing only)
- sqlite: SQLite-based persistent checkpointing (local production)
- postgres: PostgreSQL-based checkpointing (production at scale)
"""

import hashlib
import uuid
from collections.abc import Callable
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any, Literal

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, StateGraph

from src.agents.base import AgentError, BaseAgent, OrchestrationError
from src.client.lm_client import LMStudioClient
from src.config import get_logger
from src.pipeline.state import (
    ConfidenceLevel,
    ExtractionState,
    ExtractionStatus,
    add_error,
    add_warning,
    complete_extraction,
    request_human_review,
    request_retry,
    set_status,
    update_state,
)


class CheckpointerType(str, Enum):
    """Supported checkpointer backends."""

    MEMORY = "memory"  # In-memory (dev/testing only)
    SQLITE = "sqlite"  # SQLite file-based (local production)
    POSTGRES = "postgres"  # PostgreSQL (production at scale)


logger = get_logger(__name__)


# Node names for the workflow graph
NODE_PREPROCESS = "preprocess"
NODE_SPLIT = "split"  # Document boundary detection (Phase 2A)
NODE_ANALYZE = "analyze"
NODE_EXTRACT = "extract"
NODE_VALIDATE = "validate"
NODE_ROUTE = "route"
NODE_RETRY = "retry"
NODE_HUMAN_REVIEW = "human_review"
NODE_COMPLETE = "complete"

# VLM-first pipeline nodes
NODE_LAYOUT = "layout"
NODE_COMPONENTS = "components"
NODE_SCHEMA = "schema"

# Routing decisions
ROUTE_COMPLETE = "complete"
ROUTE_RETRY = "retry"
ROUTE_HUMAN_REVIEW = "human_review"

# Pipeline routing decisions
ROUTE_ADAPTIVE = "adaptive"  # VLM-first pipeline
ROUTE_LEGACY = "legacy"  # Hardcoded schema pipeline


class OrchestratorAgent(BaseAgent):
    """
    Central orchestrator for the document extraction workflow.

    Manages the LangGraph StateGraph that coordinates:
    1. Preprocessing - PDF loading and image conversion
    2. Analysis - Document classification and schema selection
    3. Extraction - Dual-pass field extraction
    4. Validation - Hallucination detection and confidence scoring
    5. Routing - Decision on completion, retry, or human review

    Supports checkpointing for fault tolerance and resumable workflows.
    """

    def __init__(
        self,
        client: LMStudioClient | None = None,
        enable_checkpointing: bool = True,
        checkpointer_type: CheckpointerType | str = CheckpointerType.MEMORY,
        sqlite_path: str | Path | None = None,
        postgres_conn_string: str | None = None,
        max_retries: int = 2,
        high_confidence_threshold: float = 0.85,
        low_confidence_threshold: float = 0.50,
    ) -> None:
        """
        Initialize the Orchestrator agent.

        Args:
            client: Optional pre-configured LM Studio client.
            enable_checkpointing: Whether to enable state checkpointing.
            checkpointer_type: Type of checkpointer backend (memory, sqlite, postgres).
            sqlite_path: Path to SQLite database file (for sqlite checkpointer).
            postgres_conn_string: PostgreSQL connection string (for postgres checkpointer).
            max_retries: Maximum retry attempts for failed extractions.
            high_confidence_threshold: Threshold for auto-acceptance.
            low_confidence_threshold: Threshold below which human review is required.

        Note:
            - memory: Use for development/testing only (data lost on restart)
            - sqlite: Use for local production (requires langgraph-checkpoint-sqlite)
            - postgres: Use for production at scale (requires langgraph-checkpoint-postgres)
        """
        super().__init__(name="orchestrator", client=client)
        self._enable_checkpointing = enable_checkpointing
        self._checkpointer_type = (
            CheckpointerType(checkpointer_type)
            if isinstance(checkpointer_type, str)
            else checkpointer_type
        )
        self._sqlite_path = sqlite_path
        self._postgres_conn_string = postgres_conn_string
        self._max_retries = max_retries
        self._high_confidence_threshold = high_confidence_threshold
        self._low_confidence_threshold = low_confidence_threshold

        # Agent instances - injected via build_workflow()
        # These are NOT lazy loaded. They're passed in when build_workflow() is called
        # and remain None until that point. The workflow graph owns the lifecycle.

        # Document splitting agent (Phase 2A)
        self._splitter: BaseAgent | None = None

        # Legacy pipeline agents
        self._analyzer: BaseAgent | None = None
        self._extractor: BaseAgent | None = None
        self._validator: BaseAgent | None = None

        # VLM-first pipeline agents
        self._layout_agent: BaseAgent | None = None
        self._component_agent: BaseAgent | None = None
        self._schema_agent: BaseAgent | None = None

        # Workflow graph
        self._workflow: StateGraph | None = None
        self._compiled_workflow: Any = None
        self._checkpointer: Any = None

        if enable_checkpointing:
            self._checkpointer = self._create_checkpointer()

    def _create_checkpointer(self) -> Any:
        """
        Create the appropriate checkpointer based on configuration.

        Returns:
            Configured checkpointer instance.

        Raises:
            ImportError: If required checkpointer package is not installed.
            OrchestrationError: If configuration is invalid.
        """
        if self._checkpointer_type == CheckpointerType.MEMORY:
            self._logger.info("using_memory_checkpointer")
            return MemorySaver()

        if self._checkpointer_type == CheckpointerType.SQLITE:
            try:
                from langgraph.checkpoint.sqlite import SqliteSaver
            except ImportError as e:
                raise ImportError(
                    "SQLite checkpointer requires langgraph-checkpoint-sqlite. "
                    "Install with: pip install langgraph-checkpoint-sqlite"
                ) from e

            if self._sqlite_path:
                db_path = self._sqlite_path
                self._logger.info("using_sqlite_checkpointer", path=str(db_path))
            else:
                # IMPORTANT: Warn about non-persistent storage
                # In-memory SQLite won't survive process restarts
                db_path = ":memory:"
                self._logger.warning(
                    "sqlite_checkpointer_using_memory",
                    message=(
                        "No sqlite_path provided - using in-memory database. "
                        "Checkpoints will NOT persist across restarts. "
                        "Set sqlite_path for persistent checkpointing."
                    ),
                )
            return SqliteSaver.from_conn_string(str(db_path))

        if self._checkpointer_type == CheckpointerType.POSTGRES:
            if not self._postgres_conn_string:
                raise OrchestrationError(
                    "PostgreSQL checkpointer requires postgres_conn_string",
                    agent_name=self.name,
                    recoverable=False,
                )

            try:
                from langgraph.checkpoint.postgres import PostgresSaver
            except ImportError as e:
                raise ImportError(
                    "PostgreSQL checkpointer requires langgraph-checkpoint-postgres. "
                    "Install with: pip install langgraph-checkpoint-postgres"
                ) from e

            self._logger.info("using_postgres_checkpointer")
            return PostgresSaver.from_conn_string(self._postgres_conn_string)

        raise OrchestrationError(
            f"Unknown checkpointer type: {self._checkpointer_type}",
            agent_name=self.name,
            recoverable=False,
        )

    def process(self, state: ExtractionState) -> ExtractionState:
        """
        Process the extraction state through the orchestration workflow.

        This is the main entry point for standalone orchestrator usage.
        For full workflow execution, use run_extraction() instead.

        Args:
            state: Current extraction state.

        Returns:
            Updated state after orchestration decisions.
        """
        # Reset metrics to prevent accumulation across documents
        self.reset_metrics()

        start_time = self.log_operation_start(
            "orchestration",
            processing_id=state.get("processing_id", ""),
        )

        try:
            # Determine current step and route appropriately
            _ = state.get("current_step", "")  # Retrieved for logging context
            status = state.get("status", "")

            # Update state based on routing decision
            # After validation completes, the status is still VALIDATING
            # We check if validation results exist to determine if validation is done
            if status == ExtractionStatus.VALIDATING.value:
                # Validation complete, make routing decision
                state = self._make_routing_decision(state)
            elif status == ExtractionStatus.FAILED.value:
                # Check if retry is possible
                state = self._handle_failure(state)

            _ = self.log_operation_complete(
                "orchestration",
                start_time,
                success=True,
                current_step=state.get("current_step"),
            )

            return state

        except Exception as e:
            self.log_operation_complete("orchestration", start_time, success=False)
            raise OrchestrationError(
                f"Orchestration failed: {e}",
                agent_name=self.name,
                recoverable=True,
            ) from e

    def _make_routing_decision(self, state: ExtractionState) -> ExtractionState:
        """
        Make routing decision based on validation results.

        NOTE: This method delegates to _determine_route() for routing logic
        to ensure single source of truth. Only this method should modify state.

        Args:
            state: Current extraction state with validation results.

        Returns:
            Updated state with routing decision.
        """
        confidence_level = state.get("confidence_level", ConfidenceLevel.LOW.value)
        overall_confidence = state.get("overall_confidence", 0.0)
        retry_count = state.get("retry_count", 0)

        self._logger.info(
            "routing_decision",
            confidence_level=confidence_level,
            overall_confidence=overall_confidence,
            retry_count=retry_count,
        )

        # Use centralized routing logic - single source of truth
        route = self._determine_route(state)

        if route == ROUTE_COMPLETE:
            # Add warning for medium confidence completions at max retries
            if (
                confidence_level == ConfidenceLevel.MEDIUM.value
                and retry_count >= self._max_retries
            ):
                state = add_warning(
                    state,
                    f"Medium confidence ({overall_confidence:.2f}) after {retry_count} retries",
                )
            return complete_extraction(state)

        if route == ROUTE_RETRY:
            reason = (
                f"{confidence_level} confidence extraction - retrying"
                if confidence_level != ConfidenceLevel.HIGH.value
                else "Extraction retry requested"
            )
            return request_retry(state, reason)

        # ROUTE_HUMAN_REVIEW
        return request_human_review(
            state,
            f"Low confidence ({overall_confidence:.2f}) requires human review",
        )

    def _handle_failure(self, state: ExtractionState) -> ExtractionState:
        """
        Handle extraction failure with retry logic.

        Args:
            state: Current extraction state.

        Returns:
            Updated state with failure handling.
        """
        retry_count = state.get("retry_count", 0)
        errors = state.get("errors", [])

        if retry_count < self._max_retries:
            return request_retry(
                state,
                f"Extraction failed: {errors[-1] if errors else 'Unknown error'}",
            )
        return request_human_review(
            state,
            f"Extraction failed after {retry_count} retries",
        )

    def build_workflow(
        self,
        preprocess_fn: Callable[[ExtractionState], ExtractionState],
        analyzer: BaseAgent,
        extractor: BaseAgent,
        validator: BaseAgent,
        layout_agent: BaseAgent | None = None,
        component_agent: BaseAgent | None = None,
        schema_agent: BaseAgent | None = None,
        splitter_agent: BaseAgent | None = None,
    ) -> StateGraph:
        """
        Build the LangGraph workflow with all agent nodes.

        Supports both legacy (hardcoded schema) and VLM-first (adaptive) pipelines.
        If VLM-first agents are provided, the workflow will route conditionally
        based on the use_adaptive_extraction flag in state.

        If a splitter_agent is provided, document boundary detection runs between
        preprocess and pipeline routing. For multi-document PDFs, each segment is
        processed through the full extract→validate pipeline.

        Args:
            preprocess_fn: Function to preprocess PDF and create initial state.
            analyzer: Analyzer agent instance (legacy pipeline).
            extractor: Extractor agent instance (both pipelines).
            validator: Validator agent instance (both pipelines).
            layout_agent: Optional layout analysis agent (VLM-first pipeline).
            component_agent: Optional component detector agent (VLM-first pipeline).
            schema_agent: Optional schema generator agent (VLM-first pipeline).
            splitter_agent: Optional document splitter agent (Phase 2A).

        Returns:
            Configured StateGraph workflow.
        """
        # Store agent references
        self._splitter = splitter_agent
        self._analyzer = analyzer
        self._extractor = extractor
        self._validator = validator
        self._layout_agent = layout_agent
        self._component_agent = component_agent
        self._schema_agent = schema_agent

        # Determine if VLM-first pipeline is available
        has_vlm_first = all([layout_agent, component_agent, schema_agent])
        has_splitter = splitter_agent is not None

        # Create the state graph
        workflow = StateGraph(ExtractionState)

        # Add common nodes
        workflow.add_node(NODE_PREPROCESS, preprocess_fn)
        workflow.add_node(NODE_VALIDATE, self._run_validator)
        workflow.add_node(NODE_ROUTE, self._route_node)
        workflow.add_node(NODE_RETRY, self._retry_node)
        workflow.add_node(NODE_HUMAN_REVIEW, self._human_review_node)
        workflow.add_node(NODE_COMPLETE, self._complete_node)

        # Add splitter node if available
        if has_splitter:
            workflow.add_node(NODE_SPLIT, self._run_splitter)

        # Add legacy pipeline nodes
        workflow.add_node(NODE_ANALYZE, self._run_analyzer)
        workflow.add_node(NODE_EXTRACT, self._run_extractor)

        # Add VLM-first pipeline nodes if available
        if has_vlm_first:
            workflow.add_node(NODE_LAYOUT, self._run_layout)
            workflow.add_node(NODE_COMPONENTS, self._run_components)
            workflow.add_node(NODE_SCHEMA, self._run_schema_generator)

        # Set entry point
        workflow.set_entry_point(NODE_PREPROCESS)

        # Determine the node after preprocess (splitter or pipeline routing)
        post_preprocess_node = NODE_SPLIT if has_splitter else None

        if has_splitter:
            # Preprocess → Split → Pipeline routing
            workflow.add_edge(NODE_PREPROCESS, NODE_SPLIT)

            if has_vlm_first:
                workflow.add_conditional_edges(
                    NODE_SPLIT,
                    self._determine_pipeline,
                    {
                        ROUTE_ADAPTIVE: NODE_LAYOUT,
                        ROUTE_LEGACY: NODE_ANALYZE,
                    },
                )
            else:
                workflow.add_edge(NODE_SPLIT, NODE_ANALYZE)
        else:
            # No splitter: Preprocess → Pipeline routing (original behavior)
            if has_vlm_first:
                workflow.add_conditional_edges(
                    NODE_PREPROCESS,
                    self._determine_pipeline,
                    {
                        ROUTE_ADAPTIVE: NODE_LAYOUT,
                        ROUTE_LEGACY: NODE_ANALYZE,
                    },
                )
            else:
                workflow.add_edge(NODE_PREPROCESS, NODE_ANALYZE)

        # VLM-first pipeline flow
        if has_vlm_first:
            workflow.add_edge(NODE_LAYOUT, NODE_COMPONENTS)
            workflow.add_edge(NODE_COMPONENTS, NODE_SCHEMA)
            workflow.add_edge(NODE_SCHEMA, NODE_EXTRACT)

        # Legacy pipeline flow (after analyze)
        workflow.add_edge(NODE_ANALYZE, NODE_EXTRACT)

        # Both pipelines converge at extract -> validate -> route
        workflow.add_edge(NODE_EXTRACT, NODE_VALIDATE)
        workflow.add_edge(NODE_VALIDATE, NODE_ROUTE)

        # Add conditional edges from route node
        workflow.add_conditional_edges(
            NODE_ROUTE,
            self._determine_route,
            {
                ROUTE_COMPLETE: NODE_COMPLETE,
                ROUTE_RETRY: NODE_RETRY,
                ROUTE_HUMAN_REVIEW: NODE_HUMAN_REVIEW,
            },
        )

        # Retry loops back to extract (works for both pipelines)
        workflow.add_edge(NODE_RETRY, NODE_EXTRACT)

        # Terminal nodes
        workflow.add_edge(NODE_COMPLETE, END)
        workflow.add_edge(NODE_HUMAN_REVIEW, END)

        self._workflow = workflow

        if has_vlm_first:
            self._logger.info(
                "workflow_built_with_vlm_first_support",
                splitter_enabled=has_splitter,
            )
        else:
            self._logger.info(
                "workflow_built_legacy_only",
                splitter_enabled=has_splitter,
            )

        return workflow

    def compile_workflow(self) -> Any:
        """
        Compile the workflow for execution.

        Returns:
            Compiled workflow that can be invoked.

        Raises:
            OrchestrationError: If workflow not built.
        """
        if self._workflow is None:
            raise OrchestrationError(
                "Workflow not built. Call build_workflow() first.",
                agent_name=self.name,
                recoverable=False,
            )

        compile_kwargs: dict[str, Any] = {}
        if self._checkpointer is not None:
            compile_kwargs["checkpointer"] = self._checkpointer

        self._compiled_workflow = self._workflow.compile(**compile_kwargs)
        self._logger.info("workflow_compiled", checkpointing=self._enable_checkpointing)

        return self._compiled_workflow

    def run_extraction(
        self,
        initial_state: ExtractionState,
        thread_id: str | None = None,
    ) -> ExtractionState:
        """
        Run the full extraction workflow.

        Args:
            initial_state: Initial extraction state with PDF data.
            thread_id: Optional thread ID for checkpointing.

        Returns:
            Final extraction state.

        Raises:
            OrchestrationError: If workflow execution fails.
        """
        if self._compiled_workflow is None:
            raise OrchestrationError(
                "Workflow not compiled. Call compile_workflow() first.",
                agent_name=self.name,
                recoverable=False,
            )

        start_time = self.log_operation_start(
            "full_extraction",
            processing_id=initial_state.get("processing_id", ""),
            thread_id=thread_id,
        )

        try:
            # Prepare config for checkpointing
            config: dict[str, Any] = {}
            if thread_id and self._checkpointer:
                config["configurable"] = {"thread_id": thread_id}

            # Run the workflow
            final_state = self._compiled_workflow.invoke(initial_state, config)

            duration_ms = self.log_operation_complete(
                "full_extraction",
                start_time,
                success=True,
                final_status=final_state.get("status"),
                confidence=final_state.get("overall_confidence"),
            )

            # Update timing in final state
            final_state = update_state(
                final_state,
                {"total_processing_ms": duration_ms},
            )

            return final_state

        except Exception as e:
            self.log_operation_complete("full_extraction", start_time, success=False)
            raise OrchestrationError(
                f"Workflow execution failed: {e}",
                agent_name=self.name,
                recoverable=True,
            ) from e

    def resume_extraction(
        self,
        thread_id: str,
        updated_state: ExtractionState | None = None,
    ) -> ExtractionState:
        """
        Resume a checkpointed extraction workflow.

        Args:
            thread_id: Thread ID of the checkpointed workflow.
            updated_state: Optional state updates to apply before resuming.

        Returns:
            Final extraction state.

        Raises:
            OrchestrationError: If resume fails.
        """
        if not self._checkpointer:
            raise OrchestrationError(
                "Checkpointing not enabled. Cannot resume.",
                agent_name=self.name,
                recoverable=False,
            )

        if self._compiled_workflow is None:
            raise OrchestrationError(
                "Workflow not compiled. Call compile_workflow() first.",
                agent_name=self.name,
                recoverable=False,
            )

        self._logger.info("resuming_extraction", thread_id=thread_id)

        config = {"configurable": {"thread_id": thread_id}}

        if updated_state:
            # Update state and continue
            return self._compiled_workflow.invoke(updated_state, config)
        # Resume from last checkpoint
        return self._compiled_workflow.invoke(None, config)

    def get_checkpoint_state(self, thread_id: str) -> ExtractionState | None:
        """
        Get the current state from a checkpoint.

        Args:
            thread_id: Thread ID of the checkpoint.

        Returns:
            Checkpointed state or None if not found.
        """
        if not self._checkpointer or not self._compiled_workflow:
            return None

        config = {"configurable": {"thread_id": thread_id}}

        try:
            state = self._compiled_workflow.get_state(config)
            return state.values if state else None
        except Exception as e:
            self._logger.warning("checkpoint_retrieval_failed", error=str(e))
            return None

    # Node implementations

    def _run_splitter(self, state: ExtractionState) -> ExtractionState:
        """Run the document splitter agent node."""
        if self._splitter is None:
            raise OrchestrationError(
                "Splitter agent not configured",
                agent_name=self.name,
                recoverable=False,
            )

        try:
            return self._splitter.process(state)
        except AgentError as e:
            # Splitter failure is non-fatal — treat as single document
            self._logger.warning(
                "splitter_failed_treating_as_single_document",
                error=str(e),
            )
            state = add_warning(
                state,
                f"Document splitting failed, treating as single document: {e}",
            )
            page_images = state.get("page_images", [])
            n_pages = len(page_images)
            return update_state(state, {
                "document_segments": [{
                    "start_page": 1,
                    "end_page": max(n_pages, 1),
                    "document_type": "unknown",
                    "confidence": 0.3,
                    "page_count": max(n_pages, 1),
                    "title": "",
                }] if n_pages > 0 else [],
                "is_multi_document": False,
                "active_segment_index": 0,
            })

    def _run_analyzer(self, state: ExtractionState) -> ExtractionState:
        """Run the analyzer agent node."""
        if self._analyzer is None:
            raise OrchestrationError(
                "Analyzer agent not configured",
                agent_name=self.name,
                recoverable=False,
            )

        try:
            return self._analyzer.process(state)
        except AgentError as e:
            state = add_error(state, f"Analysis failed: {e}")
            state = set_status(state, ExtractionStatus.FAILED, "analysis_failed")
            return state

    def _run_extractor(self, state: ExtractionState) -> ExtractionState:
        """Run the extractor agent node."""
        if self._extractor is None:
            raise OrchestrationError(
                "Extractor agent not configured",
                agent_name=self.name,
                recoverable=False,
            )

        try:
            return self._extractor.process(state)
        except AgentError as e:
            state = add_error(state, f"Extraction failed: {e}")
            state = set_status(state, ExtractionStatus.FAILED, "extraction_failed")
            return state

    def _run_validator(self, state: ExtractionState) -> ExtractionState:
        """Run the validator agent node."""
        if self._validator is None:
            raise OrchestrationError(
                "Validator agent not configured",
                agent_name=self.name,
                recoverable=False,
            )

        try:
            return self._validator.process(state)
        except AgentError as e:
            state = add_error(state, f"Validation failed: {e}")
            state = set_status(state, ExtractionStatus.FAILED, "validation_failed")
            return state

    # VLM-first pipeline node implementations

    def _run_layout(self, state: ExtractionState) -> ExtractionState:
        """Run the layout analysis agent node."""
        if self._layout_agent is None:
            raise OrchestrationError(
                "Layout agent not configured",
                agent_name=self.name,
                recoverable=False,
            )

        try:
            return self._layout_agent.process(state)
        except AgentError as e:
            # On VLM-first stage failure, fall back to legacy pipeline
            self._logger.warning(
                "layout_analysis_failed_fallback_to_legacy",
                error=str(e),
            )
            state = add_warning(
                state,
                f"Layout analysis failed, using legacy pipeline: {e}",
            )
            # Disable adaptive extraction to force legacy path
            state = update_state(state, {"use_adaptive_extraction": False})
            return state

    def _run_components(self, state: ExtractionState) -> ExtractionState:
        """Run the component detector agent node."""
        if self._component_agent is None:
            raise OrchestrationError(
                "Component detector agent not configured",
                agent_name=self.name,
                recoverable=False,
            )

        try:
            return self._component_agent.process(state)
        except AgentError as e:
            # On VLM-first stage failure, fall back to legacy pipeline
            self._logger.warning(
                "component_detection_failed_fallback_to_legacy",
                error=str(e),
            )
            state = add_warning(
                state,
                f"Component detection failed, using legacy pipeline: {e}",
            )
            # Disable adaptive extraction to force legacy path
            state = update_state(state, {"use_adaptive_extraction": False})
            return state

    def _run_schema_generator(self, state: ExtractionState) -> ExtractionState:
        """Run the schema generator agent node."""
        if self._schema_agent is None:
            raise OrchestrationError(
                "Schema generator agent not configured",
                agent_name=self.name,
                recoverable=False,
            )

        try:
            return self._schema_agent.process(state)
        except AgentError as e:
            # On VLM-first stage failure, fall back to legacy pipeline
            self._logger.warning(
                "schema_generation_failed_fallback_to_legacy",
                error=str(e),
            )
            state = add_warning(
                state,
                f"Schema generation failed, using legacy pipeline: {e}",
            )
            # Disable adaptive extraction to force legacy path
            state = update_state(state, {"use_adaptive_extraction": False})
            return state

    def _route_node(self, state: ExtractionState) -> ExtractionState:
        """
        Route node - determines next step based on validation results.

        Updates state with routing decision but doesn't change status yet.
        """
        # This node just marks that routing is happening
        # Actual routing logic is in _determine_route
        return update_state(state, {"current_step": "routing"})

    def _determine_route(
        self,
        state: ExtractionState,
    ) -> Literal["complete", "retry", "human_review"]:
        """
        Determine the routing path based on state — single source of truth.

        Uses confidence_level, overall_confidence, and validator recommendation
        flags to make the final routing decision. The validator annotates the
        state with recommendations but does NOT set the final status.

        This is the conditional edge function for LangGraph.

        Args:
            state: Current extraction state.

        Returns:
            Route name (complete, retry, or human_review).
        """
        status = state.get("status", "")

        # Check for failures
        if status == ExtractionStatus.FAILED.value:
            retry_count = state.get("retry_count", 0)
            if retry_count < self._max_retries:
                return ROUTE_RETRY
            return ROUTE_HUMAN_REVIEW

        # Gather routing inputs
        confidence_level = state.get("confidence_level", ConfidenceLevel.LOW.value)
        overall_confidence = state.get("overall_confidence", 0.0)
        retry_count = state.get("retry_count", 0)
        validation_is_valid = state.get("validation_is_valid", True)
        validator_wants_retry = state.get("validation_requires_retry", False)
        validator_wants_review = state.get("validation_requires_human_review", False)

        self._logger.debug(
            "determining_route",
            confidence_level=confidence_level,
            overall_confidence=overall_confidence,
            retry_count=retry_count,
            validation_is_valid=validation_is_valid,
            validator_wants_retry=validator_wants_retry,
            validator_wants_review=validator_wants_review,
        )

        # HIGH confidence + valid: auto-complete
        if confidence_level == ConfidenceLevel.HIGH.value and validation_is_valid:
            return ROUTE_COMPLETE

        # Validator explicitly recommends retry and retries remain
        if validator_wants_retry and retry_count < self._max_retries:
            return ROUTE_RETRY

        # Validator explicitly recommends human review
        if validator_wants_review:
            if retry_count < self._max_retries:
                return ROUTE_RETRY  # Try once more before human review
            return ROUTE_HUMAN_REVIEW

        # MEDIUM confidence: retry if possible, else accept
        if confidence_level == ConfidenceLevel.MEDIUM.value:
            if retry_count < self._max_retries:
                return ROUTE_RETRY
            return ROUTE_COMPLETE

        # LOW confidence: retry once, then human review
        if retry_count < 1:
            return ROUTE_RETRY
        return ROUTE_HUMAN_REVIEW

    def _determine_pipeline(
        self,
        state: ExtractionState,
    ) -> Literal["adaptive", "legacy"]:
        """
        Determine which pipeline to use based on state configuration.

        This is the conditional edge function after preprocessing.

        Args:
            state: Current extraction state after preprocessing.

        Returns:
            Pipeline route name (adaptive or legacy).
        """
        use_adaptive = state.get("use_adaptive_extraction", True)

        if use_adaptive:
            self._logger.info("using_vlm_first_adaptive_pipeline")
            return ROUTE_ADAPTIVE

        self._logger.info("using_legacy_hardcoded_schema_pipeline")
        return ROUTE_LEGACY

    def _retry_node(self, state: ExtractionState) -> ExtractionState:
        """
        Retry node - clears stale extraction data and prepares for re-extraction.

        Clears previous extraction results to prevent stale data contamination,
        increments retry counter, and resets status for fresh extraction.
        """
        retry_count = state.get("retry_count", 0) + 1

        self._logger.info(
            "retry_extraction",
            retry_count=retry_count,
            max_retries=self._max_retries,
            clearing_previous="merged_extraction,field_metadata,validation",
        )

        return update_state(
            state,
            {
                "retry_count": retry_count,
                "status": ExtractionStatus.EXTRACTING.value,
                "current_step": f"retry_{retry_count}",
                # Clear stale extraction data to prevent contamination
                "merged_extraction": {},
                "field_metadata": {},
                "page_extractions": [],
                "validation": None,
                "overall_confidence": 0.0,
                "confidence_level": "",
                # Clear validator recommendation flags
                "validation_is_valid": True,
                "validation_requires_retry": False,
                "validation_requires_human_review": False,
                "validation_reasons": "",
            },
        )

    def _human_review_node(self, state: ExtractionState) -> ExtractionState:
        """
        Human review node - marks extraction for human review.

        Sets final status and prepares state for human review queue.
        """
        overall_confidence = state.get("overall_confidence", 0.0)
        errors = state.get("errors", [])

        reason = (
            f"Low confidence ({overall_confidence:.2f})"
            if not errors
            else f"Extraction errors: {len(errors)}"
        )

        self._logger.info(
            "human_review_required",
            reason=reason,
            confidence=overall_confidence,
            error_count=len(errors),
        )

        return request_human_review(state, reason)

    def _complete_node(self, state: ExtractionState) -> ExtractionState:
        """
        Complete node - finalizes successful extraction.

        Sets final status and calculates total processing time.
        """
        overall_confidence = state.get("overall_confidence", 0.0)
        retry_count = state.get("retry_count", 0)

        self._logger.info(
            "extraction_complete",
            confidence=overall_confidence,
            retry_count=retry_count,
        )

        return complete_extraction(state)

    def get_workflow_metrics(self) -> dict[str, Any]:
        """
        Get metrics about the workflow execution.

        Returns:
            Dictionary of workflow metrics.
        """
        return {
            "agent_name": self.name,
            "checkpointing_enabled": self._enable_checkpointing,
            "max_retries": self._max_retries,
            "high_confidence_threshold": self._high_confidence_threshold,
            "low_confidence_threshold": self._low_confidence_threshold,
            "workflow_compiled": self._compiled_workflow is not None,
        }


def create_extraction_workflow(
    preprocess_fn: Callable[[ExtractionState], ExtractionState],
    client: LMStudioClient | None = None,
    enable_checkpointing: bool = True,
    max_retries: int = 2,
    enable_vlm_first: bool = True,
) -> tuple[OrchestratorAgent, Any]:
    """
    Factory function to create a complete extraction workflow.

    Creates all agents and builds the workflow graph.

    Args:
        preprocess_fn: Function to preprocess PDF and create initial state.
        client: Optional pre-configured LM Studio client.
        enable_checkpointing: Whether to enable state checkpointing.
        max_retries: Maximum retry attempts.
        enable_vlm_first: Whether to enable VLM-first adaptive pipeline (default: True).

    Returns:
        Tuple of (orchestrator_agent, compiled_workflow).
    """
    # Import agents here to avoid circular imports
    from src.agents.analyzer import AnalyzerAgent
    from src.agents.extractor import ExtractorAgent
    from src.agents.validator import ValidatorAgent

    # Create shared client
    shared_client = client or LMStudioClient()

    # Create legacy pipeline agents (always needed)
    analyzer = AnalyzerAgent(client=shared_client)
    extractor = ExtractorAgent(client=shared_client)
    validator = ValidatorAgent(client=shared_client)

    # Create VLM-first pipeline agents if enabled
    layout_agent = None
    component_agent = None
    schema_agent = None

    if enable_vlm_first:
        try:
            from src.agents.layout_agent import LayoutAgent
            from src.agents.component_detector import ComponentDetectorAgent
            from src.agents.schema_generator import SchemaGeneratorAgent

            layout_agent = LayoutAgent(client=shared_client)
            component_agent = ComponentDetectorAgent(client=shared_client)
            schema_agent = SchemaGeneratorAgent(client=shared_client)

            logger.info("vlm_first_agents_created")
        except ImportError as e:
            logger.warning(
                "vlm_first_agents_import_failed",
                error=str(e),
                message="Falling back to legacy pipeline only",
            )
            enable_vlm_first = False

    # Create orchestrator
    orchestrator = OrchestratorAgent(
        client=shared_client,
        enable_checkpointing=enable_checkpointing,
        max_retries=max_retries,
    )

    # Build workflow with appropriate agents
    orchestrator.build_workflow(
        preprocess_fn=preprocess_fn,
        analyzer=analyzer,
        extractor=extractor,
        validator=validator,
        layout_agent=layout_agent,
        component_agent=component_agent,
        schema_agent=schema_agent,
    )

    compiled_workflow = orchestrator.compile_workflow()

    logger.info(
        "extraction_workflow_created",
        checkpointing=enable_checkpointing,
        max_retries=max_retries,
        vlm_first_enabled=enable_vlm_first,
    )

    return orchestrator, compiled_workflow


def generate_processing_id() -> str:
    """
    Generate a unique processing ID for a new extraction.

    Returns:
        Unique processing ID string.
    """
    timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
    unique_id = uuid.uuid4().hex[:8]
    return f"extract_{timestamp}_{unique_id}"


def generate_thread_id(pdf_path: str, processing_id: str) -> str:
    """
    Generate a deterministic thread ID for checkpointing.

    Args:
        pdf_path: Path to the PDF file.
        processing_id: Processing ID for this extraction.

    Returns:
        Thread ID string for checkpointing.
    """
    combined = f"{pdf_path}:{processing_id}"
    hash_value = hashlib.sha256(combined.encode()).hexdigest()[:16]
    return f"thread_{hash_value}"
