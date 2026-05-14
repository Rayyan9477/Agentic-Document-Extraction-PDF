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


# WS-5a: LangGraph v3 primitives. ``interrupt`` pauses graph execution so a
# human reviewer can supply corrections; ``Command(resume=...)`` is the
# matching client-side resume primitive used in PipelineRunner.
try:
    from langgraph.types import Command, interrupt
except ImportError:  # pragma: no cover - fallback for older langgraph
    Command = None  # type: ignore[assignment,misc]
    interrupt = None  # type: ignore[assignment]

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
NODE_TABLE_DETECT = "table_detect"  # Table structure detection (Phase 2B)

# V3 Phase 2 — heterogeneous dual-VLM extraction nodes
NODE_EXTRACT_PASS1 = "extract_pass1"
NODE_EXTRACT_PASS2 = "extract_pass2"
NODE_RECONCILE = "reconcile"

# V3 Phase 3 — Critic + bbox-roundtrip routing nodes
NODE_CRITIC = "critic"
NODE_CRITIC_COMBINER = "critic_combiner"

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
        checkpointer_type: CheckpointerType | str = CheckpointerType.SQLITE,
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

        # Table detection agent (Phase 2B)
        self._table_detector: BaseAgent | None = None

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
            self._logger.warning(
                "memory_checkpointer_explicitly_selected",
                message=(
                    "MemorySaver is non-durable: in-flight extractions are lost "
                    "on process crash or restart. Use this only for tests or "
                    "ad-hoc smoke runs. Production deployments should use "
                    "CheckpointerType.SQLITE (default) or CheckpointerType.POSTGRES."
                ),
            )
            return MemorySaver()

        if self._checkpointer_type == CheckpointerType.SQLITE:
            try:
                from langgraph.checkpoint.sqlite import SqliteSaver
            except ImportError:
                # Graceful degradation: SQLite is the new default, but a fresh
                # checkout that hasn't run `pip install -e .` won't have
                # langgraph-checkpoint-sqlite yet. Warn loudly and fall back
                # so the system stays runnable; production installs that pull
                # the manifest will get durable checkpointing automatically.
                self._logger.error(
                    "sqlite_checkpoint_dep_missing_fallback_to_memory",
                    message=(
                        "SQLite checkpointer requested (current default) but "
                        "langgraph-checkpoint-sqlite is not installed. Falling "
                        "back to MemorySaver — checkpoints will NOT survive a "
                        "process restart. Install with: "
                        "`pip install langgraph-checkpoint-sqlite` (or "
                        "`pip install -e .` to pick up the project manifest)."
                    ),
                )
                return MemorySaver()

            if self._sqlite_path:
                db_path = Path(self._sqlite_path)
            else:
                # Persistent default: workspace-local checkpoint store.
                # Survives process restarts; safe to delete to reset state.
                db_path = Path(".extraction_checkpoints") / "checkpoints.db"

            db_path.parent.mkdir(parents=True, exist_ok=True)
            self._logger.info("using_sqlite_checkpointer", path=str(db_path))
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
        table_detector_agent: BaseAgent | None = None,
        extractor_pass1_agent: BaseAgent | None = None,
        extractor_pass2_agent: BaseAgent | None = None,
        reconciler_node: Callable[[ExtractionState], ExtractionState] | None = None,
        critic_agent: BaseAgent | None = None,
        critic_combiner_node: Callable[[ExtractionState], ExtractionState] | None = None,
    ) -> StateGraph:
        """
        Build the LangGraph workflow with all agent nodes.

        Supports both legacy (hardcoded schema) and VLM-first (adaptive) pipelines.
        If VLM-first agents are provided, the workflow will route conditionally
        based on the use_adaptive_extraction flag in state.

        If a splitter_agent is provided, document boundary detection runs between
        preprocess and pipeline routing. For multi-document PDFs, each segment is
        processed through the full extract→validate pipeline.

        If a table_detector_agent is provided, table detection runs between
        component detection and schema generation in the VLM-first pipeline,
        or between analyze and extract in the legacy pipeline.

        Args:
            preprocess_fn: Function to preprocess PDF and create initial state.
            analyzer: Analyzer agent instance (legacy pipeline).
            extractor: Extractor agent instance (both pipelines).
            validator: Validator agent instance (both pipelines).
            layout_agent: Optional layout analysis agent (VLM-first pipeline).
            component_agent: Optional component detector agent (VLM-first pipeline).
            schema_agent: Optional schema generator agent (VLM-first pipeline).
            splitter_agent: Optional document splitter agent (Phase 2A).
            table_detector_agent: Optional table detector agent (Phase 2B).

        Returns:
            Configured StateGraph workflow.
        """
        # Store agent references
        self._splitter = splitter_agent
        self._table_detector = table_detector_agent
        self._analyzer = analyzer
        self._extractor = extractor
        self._validator = validator
        self._layout_agent = layout_agent
        self._component_agent = component_agent
        self._schema_agent = schema_agent
        self._extractor_pass1 = extractor_pass1_agent
        self._extractor_pass2 = extractor_pass2_agent
        self._reconciler_node = reconciler_node
        self._critic = critic_agent
        self._critic_combiner_node = critic_combiner_node

        # Determine if VLM-first pipeline is available
        has_vlm_first = all([layout_agent, component_agent, schema_agent])
        has_splitter = splitter_agent is not None
        has_table_detector = table_detector_agent is not None
        # V3 Phase 2 — heterogeneous dual-VLM is available when both Pass
        # agents AND the reconciler node are wired. Without all three, the
        # legacy single-VLM extractor handles the EXTRACT node.
        has_dual_vlm = bool(
            extractor_pass1_agent
            and extractor_pass2_agent
            and reconciler_node
        )
        # V3 Phase 3 — Critic runs between validate and route when wired.
        # The combiner node always pairs with the Critic; without one we
        # skip the Critic chain entirely.
        has_critic = bool(critic_agent and critic_combiner_node)

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

        # V3 Phase 2 — heterogeneous dual-VLM nodes. ``NODE_EXTRACT`` is
        # always declared (the legacy extractor) so existing edges that
        # target it stay valid. When dual-VLM is wired, an additional
        # 3-node chain (pass1 -> pass2 in parallel -> reconcile) lands in
        # the graph and ``NODE_EXTRACT`` becomes a no-op for the new
        # path. Routing decides which chain runs (see edge wiring below).
        if has_dual_vlm:
            workflow.add_node(NODE_EXTRACT_PASS1, self._run_extractor_pass1)
            workflow.add_node(NODE_EXTRACT_PASS2, self._run_extractor_pass2)
            workflow.add_node(NODE_RECONCILE, self._reconciler_node)

        # V3 Phase 3 — Critic + combiner. Critic runs once per document
        # between ``validate`` and ``route``; the combiner merges its
        # signal with dual-pass agreement and modality penalty into
        # ``confidence_components`` for the calibrator. Both nodes
        # land together; the combiner has no value without the Critic.
        if has_critic:
            workflow.add_node(NODE_CRITIC, self._run_critic)
            workflow.add_node(NODE_CRITIC_COMBINER, self._critic_combiner_node)

        # Add table detector node if available
        if has_table_detector:
            workflow.add_node(NODE_TABLE_DETECT, self._run_table_detector)

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
        # No splitter: Preprocess → Pipeline routing (original behavior)
        elif has_vlm_first:
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
            if has_table_detector:
                # Components → Table Detect → Schema → Extract
                workflow.add_edge(NODE_COMPONENTS, NODE_TABLE_DETECT)
                workflow.add_edge(NODE_TABLE_DETECT, NODE_SCHEMA)
            else:
                workflow.add_edge(NODE_COMPONENTS, NODE_SCHEMA)
            workflow.add_edge(NODE_SCHEMA, NODE_EXTRACT)

        # Legacy pipeline flow (after analyze)
        if has_table_detector and not has_vlm_first:
            # Analyze → Table Detect → Extract
            workflow.add_edge(NODE_ANALYZE, NODE_TABLE_DETECT)
            workflow.add_edge(NODE_TABLE_DETECT, NODE_EXTRACT)
        else:
            workflow.add_edge(NODE_ANALYZE, NODE_EXTRACT)

        # V3 Phase 2 — when dual-VLM is wired AND extraction.engine=dual_vlm,
        # the EXTRACT node fans out to Pass 1 + Pass 2 then reconciles.
        # We route via a conditional edge from a "select_extractor" check
        # baked into the legacy NODE_EXTRACT runner: if dual_vlm is active,
        # NODE_EXTRACT is a pass-through that hands off to NODE_EXTRACT_PASS1.
        # See ``_run_extractor`` for the dispatch logic. The graph wiring
        # below adds the new chain unconditionally when dual-VLM agents
        # exist; the runtime gate is the engine flag.
        if has_dual_vlm:
            # Sequential: pass1 -> pass2 -> reconcile -> validate.
            # (Phase 2 ships sequential; ``Send(...)`` fanout for true
            # parallel pass1 || pass2 lands when LangGraph's parallel-
            # branch story stabilises in the orchestrator's broader
            # refactor.)
            workflow.add_edge(NODE_EXTRACT, NODE_EXTRACT_PASS1)
            workflow.add_edge(NODE_EXTRACT_PASS1, NODE_EXTRACT_PASS2)
            workflow.add_edge(NODE_EXTRACT_PASS2, NODE_RECONCILE)
            workflow.add_edge(NODE_RECONCILE, NODE_VALIDATE)
        else:
            workflow.add_edge(NODE_EXTRACT, NODE_VALIDATE)

        # V3 Phase 3 — splice the Critic chain between validate and route.
        # When wired: validate -> critic -> critic_combiner -> route.
        # When not: validate -> route (legacy / Phase 2 behavior).
        if has_critic:
            workflow.add_edge(NODE_VALIDATE, NODE_CRITIC)
            workflow.add_edge(NODE_CRITIC, NODE_CRITIC_COMBINER)
            workflow.add_edge(NODE_CRITIC_COMBINER, NODE_ROUTE)
        else:
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

        # Retry target depends on confidence and pipeline type.
        # Low-confidence adaptive retries regenerate schema first.
        if has_vlm_first:
            workflow.add_conditional_edges(
                NODE_RETRY,
                self._determine_retry_target,
                {
                    "schema": NODE_SCHEMA,
                    "extract": NODE_EXTRACT,
                },
            )
        else:
            workflow.add_edge(NODE_RETRY, NODE_EXTRACT)

        # Terminal nodes
        workflow.add_edge(NODE_COMPLETE, END)
        workflow.add_edge(NODE_HUMAN_REVIEW, END)

        self._workflow = workflow

        if has_vlm_first:
            self._logger.info(
                "workflow_built_with_vlm_first_support",
                splitter_enabled=has_splitter,
                table_detector_enabled=has_table_detector,
            )
        else:
            self._logger.info(
                "workflow_built_legacy_only",
                splitter_enabled=has_splitter,
                table_detector_enabled=has_table_detector,
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

        # V3 Phase 6 — observability dispatcher for canonical events.
        # Loaded lazily; failures are non-fatal (defensive sink).
        try:
            from src.monitoring.observability import (
                EVENT_EXTRACTION_COMPLETED,
                EVENT_EXTRACTION_STARTED,
                EVENT_HUMAN_REVIEW_TRIGGERED,
                _read_trace_id_from_context,
                get_dispatcher,
            )

            _obs = get_dispatcher()
        except Exception:  # pragma: no cover - defensive
            _obs = None
            EVENT_EXTRACTION_STARTED = "extraction_started"  # type: ignore[assignment]
            EVENT_EXTRACTION_COMPLETED = "extraction_completed"  # type: ignore[assignment]
            EVENT_HUMAN_REVIEW_TRIGGERED = "human_review_triggered"  # type: ignore[assignment]

            def _read_trace_id_from_context() -> str | None:  # type: ignore[no-redef]
                return None

        if _obs is not None and _obs.is_active:
            try:
                _obs.emit_event(
                    EVENT_EXTRACTION_STARTED,
                    {
                        "processing_id": initial_state.get("processing_id"),
                        "thread_id": thread_id,
                        "document_type": initial_state.get("document_type"),
                        "profile": initial_state.get("profile"),
                        "page_count": len(initial_state.get("page_images", [])),
                        "trace_id": _read_trace_id_from_context(),
                    },
                )
            except Exception:  # pragma: no cover - defensive
                pass

        try:
            # Prepare config for checkpointing.
            # V3 Phase 7 — checkpoint namespace partitioned by tenant.
            # When the initial state carries a ``tenant_id`` we scope
            # the checkpoint to ``tenant:<tenant>:proc:<processing_id>``
            # so two simultaneous extractions for different tenants
            # cannot read each other's checkpoints. Falls back to
            # the legacy single-namespace behaviour when no tenant
            # is set.
            config: dict[str, Any] = {}
            if thread_id and self._checkpointer:
                configurable: dict[str, Any] = {"thread_id": thread_id}
                tenant_id = (
                    initial_state.get("tenant_id")
                    or initial_state.get("tenant")
                )
                proc_id = initial_state.get("processing_id")
                if tenant_id and proc_id:
                    configurable["checkpoint_ns"] = (
                        f"tenant:{tenant_id}:proc:{proc_id}"
                    )
                config["configurable"] = configurable

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

            # V3 Phase 6 — emit completion / human-review events.
            if _obs is not None and _obs.is_active:
                try:
                    _obs.emit_event(
                        EVENT_EXTRACTION_COMPLETED,
                        {
                            "processing_id": final_state.get("processing_id"),
                            "thread_id": thread_id,
                            "status": final_state.get("status"),
                            "document_type": final_state.get("document_type"),
                            "profile": final_state.get("profile"),
                            "overall_confidence": final_state.get("overall_confidence"),
                            "duration_ms": duration_ms,
                            "vlm_calls": final_state.get("total_vlm_calls", 0),
                            "trace_id": _read_trace_id_from_context(),
                        },
                    )
                    if final_state.get("requires_human_review"):
                        _obs.emit_event(
                            EVENT_HUMAN_REVIEW_TRIGGERED,
                            {
                                "processing_id": final_state.get("processing_id"),
                                "thread_id": thread_id,
                                "reason": final_state.get("human_review_reason"),
                                "critic_recommendation": final_state.get("critic_recommendation"),
                                "trace_id": _read_trace_id_from_context(),
                            },
                        )
                except Exception:  # pragma: no cover - defensive
                    pass

            return final_state

        except Exception as e:
            self.log_operation_complete("full_extraction", start_time, success=False)
            if _obs is not None and _obs.is_active:
                try:
                    _obs.emit_event(
                        EVENT_EXTRACTION_COMPLETED,
                        {
                            "processing_id": initial_state.get("processing_id"),
                            "thread_id": thread_id,
                            "status": "failed",
                            "error": str(e),
                            "trace_id": _read_trace_id_from_context(),
                        },
                    )
                except Exception:  # pragma: no cover - defensive
                    pass
            raise OrchestrationError(
                f"Workflow execution failed: {e}",
                agent_name=self.name,
                recoverable=True,
            ) from e

    def resume_extraction(
        self,
        thread_id: str,
        updated_state: ExtractionState | None = None,
        *,
        human_corrections: dict[str, Any] | None = None,
        processing_id: str | None = None,
        tenant_id: str | None = None,
    ) -> ExtractionState:
        """
        Resume a checkpointed extraction workflow.

        Three resume modes (mutually exclusive, evaluated in order):

        1. ``human_corrections`` provided → resume from a human-review
           ``interrupt()`` (WS-5a). The graph re-enters the human-review
           node and the corrections are merged into ``merged_extraction``.
        2. ``updated_state`` provided → re-invoke the graph from the
           checkpoint with those state overrides.
        3. Neither → plain checkpoint resume (re-invoke with ``None`` so
           LangGraph picks up where it left off).

        Args:
            thread_id: Thread ID of the checkpointed workflow.
            updated_state: Optional state updates to apply before resuming.
            human_corrections: Optional dict of reviewer corrections used
                with the v3 ``Command(resume=...)`` flow. Pass ``{}`` to
                accept the extraction as-is and continue past the
                ``interrupt``; pass ``{"field": "value", ...}`` to overlay
                corrected values.
            processing_id: Optional processing ID for tenant-isolated
                ``checkpoint_ns``. Defaults to the thread's namespace.

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

        self._logger.info(
            "resuming_extraction",
            thread_id=thread_id,
            mode=(
                "human_corrections"
                if human_corrections is not None
                else "state_override"
                if updated_state
                else "checkpoint_only"
            ),
        )

        config: dict[str, Any] = {"configurable": {"thread_id": thread_id}}
        # WS-5a / V3 Phase 8: per-(tenant, processing) checkpoint
        # namespace. ``run_extraction`` writes under
        # ``tenant:{id}:proc:{id}`` when a tenant_id is in state; we
        # mirror that here so resume reads from the same namespace.
        # Legacy fallback to ``proc:{id}`` covers checkpoints written
        # before Phase 8 (which never carried a tenant prefix).
        if tenant_id and processing_id:
            config["configurable"]["checkpoint_ns"] = (
                f"tenant:{tenant_id}:proc:{processing_id}"
            )
        elif processing_id:
            config["configurable"]["checkpoint_ns"] = f"proc:{processing_id}"

        # Resume from a human-review interrupt (v3 path).
        if human_corrections is not None:
            if Command is None:
                raise OrchestrationError(
                    "LangGraph v3 Command primitive is not available; "
                    "cannot resume from interrupt with corrections.",
                    agent_name=self.name,
                    recoverable=False,
                )
            return self._compiled_workflow.invoke(
                Command(resume=human_corrections),
                config,
            )

        if updated_state:
            return self._compiled_workflow.invoke(updated_state, config)
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

    def _run_table_detector(self, state: ExtractionState) -> ExtractionState:
        """Run the table detector agent node."""
        if self._table_detector is None:
            raise OrchestrationError(
                "Table detector agent not configured",
                agent_name=self.name,
                recoverable=False,
            )

        try:
            return self._table_detector.process(state)
        except AgentError as e:
            # Table detection failure is non-fatal — continue without table data
            self._logger.warning(
                "table_detection_failed_continuing",
                error=str(e),
            )
            state = add_warning(
                state,
                f"Table detection failed, continuing without table data: {e}",
            )
            return update_state(state, {"detected_tables": []})

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
        """Run the legacy extractor — or pass through when dual-VLM is active.

        When ``settings.extraction.engine == "dual_vlm"`` AND the dual-
        VLM agents are wired in ``build_workflow``, this node becomes a
        pass-through and the downstream edges (``NODE_EXTRACT_PASS1`` ->
        ``NODE_EXTRACT_PASS2`` -> ``NODE_RECONCILE``) do the real work.
        Otherwise the legacy single-VLM extractor runs as before.
        """
        from src.config.settings import ExtractionEngine

        engine = self._settings.extraction.engine
        if (
            engine == ExtractionEngine.DUAL_VLM
            and self._extractor_pass1 is not None
            and self._extractor_pass2 is not None
            and self._reconciler_node is not None
        ):
            # Pass-through: the next edge dispatches to NODE_EXTRACT_PASS1.
            self._logger.info(
                "extractor_dispatching_to_dual_vlm",
                processing_id=state.get("processing_id"),
            )
            return state

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

    # ------------------------------------------------------------------
    # V3 Phase 2 — heterogeneous dual-VLM runners
    # ------------------------------------------------------------------

    def _run_extractor_pass1(self, state: ExtractionState) -> ExtractionState:
        """Run Pass 1 (EXTRACTOR / primary VLM)."""
        if self._extractor_pass1 is None:
            raise OrchestrationError(
                "ExtractorPass1Agent not configured",
                agent_name=self.name,
                recoverable=False,
            )
        try:
            return self._extractor_pass1.process(state)
        except AgentError as e:
            state = add_error(state, f"Pass 1 extraction failed: {e}")
            state = set_status(state, ExtractionStatus.FAILED, "pass1_failed")
            return state

    def _run_extractor_pass2(self, state: ExtractionState) -> ExtractionState:
        """Run Pass 2 (AUDITOR / secondary VLM)."""
        if self._extractor_pass2 is None:
            raise OrchestrationError(
                "ExtractorPass2Agent not configured",
                agent_name=self.name,
                recoverable=False,
            )
        try:
            return self._extractor_pass2.process(state)
        except AgentError as e:
            # Pass 2 failure does not abort the extraction — the
            # reconciler can still emit a Pass-1-only result with a
            # confidence penalty. The error surfaces in warnings.
            state = add_warning(state, f"Pass 2 extraction failed: {e}")
            return state

    # ------------------------------------------------------------------
    # V3 Phase 3 — Critic runner
    # ------------------------------------------------------------------

    def _run_critic(self, state: ExtractionState) -> ExtractionState:
        """Run the Critic agent (independent verifier).

        Critic failure is non-fatal: ``CriticAgent.process`` already
        catches its own VLM exceptions and short-circuits to
        ``recommendation=accept``. We add an outer guard for any other
        error class so a misbehaving Critic never blocks emission.
        """
        if self._critic is None:
            raise OrchestrationError(
                "Critic agent not configured",
                agent_name=self.name,
                recoverable=False,
            )
        try:
            return self._critic.process(state)
        except AgentError as e:
            # Even AgentError is non-fatal here — the Critic is an
            # auxiliary signal, not a blocker. Log and recommend
            # ``accept`` so downstream routing proceeds.
            self._logger.warning(
                "critic_agent_error_falling_back_accept",
                error=str(e),
            )
            state = add_warning(state, f"Critic failed: {e}")
            return update_state(
                state,
                {
                    "critic_report": {
                        "trust_score": 0.5,
                        "concerns": [],
                        "recommendation": "accept",
                        "_short_circuit_reason": "critic_agent_error",
                    },
                    "critic_recommendation": "accept",
                    "critic_model_id": "",
                    "critic_latency_ms": 0,
                },
            )

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
        # Skip if adaptive pipeline was disabled (e.g., layout failed)
        if not state.get("use_adaptive_extraction", True):
            self._logger.info("skipping_components_adaptive_disabled")
            return state

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
        # Skip if adaptive pipeline was disabled (e.g., earlier stage failed)
        if not state.get("use_adaptive_extraction", True):
            self._logger.info("skipping_schema_generation_adaptive_disabled")
            return state

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

        Every decision emits a structured audit event so HIPAA reviewers can
        reconstruct *why* a record was auto-accepted vs. retried vs. routed to
        human review. The audit log is the system of record for those choices.

        Args:
            state: Current extraction state.

        Returns:
            Route name (complete, retry, or human_review).
        """
        decision, reason = self._route_with_reason(state)
        self._logger.info(
            "routing_decision",
            decision=decision,
            reason=reason,
            processing_id=state.get("processing_id"),
            confidence=state.get("overall_confidence", 0.0),
            confidence_level=state.get("confidence_level"),
            retry_count=state.get("retry_count", 0),
            validation_is_valid=state.get("validation_is_valid", True),
        )
        return decision

    def _route_with_reason(
        self,
        state: ExtractionState,
    ) -> tuple[Literal["complete", "retry", "human_review"], str]:
        """Compute the route plus a human-readable reason for the audit log."""
        status = state.get("status", "")

        # Check for failures
        if status == ExtractionStatus.FAILED.value:
            retry_count = state.get("retry_count", 0)
            if retry_count < self._max_retries:
                return ROUTE_RETRY, f"failed status with retry budget remaining ({retry_count}/{self._max_retries})"
            return ROUTE_HUMAN_REVIEW, "failed status with retry budget exhausted"

        # V3 Phase 3 — honor the Critic's recommendation when present.
        # The Critic's verdict is a strong external signal; it overrides
        # the legacy confidence-based routing only when its recommendation
        # is more conservative (retry / human_review). ``accept`` and
        # ``verify_bbox`` fall through so the existing confidence rules
        # still apply (verify_bbox is handled in-line by the reconciler
        # / bbox-roundtrip helper, not at the routing layer).
        critic_rec = state.get("critic_recommendation")
        retry_count_for_critic = state.get("retry_count", 0)
        if critic_rec == "human_review":
            return ROUTE_HUMAN_REVIEW, "critic recommended human_review"
        if critic_rec == "retry":
            if retry_count_for_critic < self._max_retries:
                return ROUTE_RETRY, "critic recommended retry"
            return ROUTE_HUMAN_REVIEW, "critic recommended retry but budget exhausted"

        confidence_level = state.get("confidence_level", ConfidenceLevel.LOW.value)
        retry_count = state.get("retry_count", 0)
        validation_is_valid = state.get("validation_is_valid", True)
        validator_wants_retry = state.get("validation_requires_retry", False)
        validator_wants_review = state.get("validation_requires_human_review", False)

        if confidence_level == ConfidenceLevel.HIGH.value and validation_is_valid:
            return ROUTE_COMPLETE, "high confidence + validation passed"

        if validator_wants_retry and retry_count < self._max_retries:
            return ROUTE_RETRY, f"validator requested retry ({retry_count}/{self._max_retries})"

        if validator_wants_review:
            if retry_count < self._max_retries:
                return ROUTE_RETRY, f"validator requested review; retrying first ({retry_count}/{self._max_retries})"
            return ROUTE_HUMAN_REVIEW, "validator requested human review; retry budget exhausted"

        if confidence_level == ConfidenceLevel.MEDIUM.value:
            if retry_count < self._max_retries:
                return ROUTE_RETRY, f"medium confidence; retrying ({retry_count}/{self._max_retries})"
            return ROUTE_COMPLETE, "medium confidence; retry budget exhausted, accepting"

        if retry_count < self._max_retries:
            return ROUTE_RETRY, f"low confidence; retrying ({retry_count}/{self._max_retries})"
        return ROUTE_HUMAN_REVIEW, "low confidence; retry budget exhausted"

    def _determine_retry_target(
        self,
        state: ExtractionState,
    ) -> Literal["schema", "extract"]:
        """
        Determine whether retry should regenerate schema or just re-extract.

        On the first retry of a low-confidence adaptive extraction, loop back
        to schema generation so the VLM can produce a better field list.
        Subsequent retries go straight to extraction.
        """
        use_adaptive = state.get("use_adaptive_extraction", False)
        confidence = state.get("overall_confidence", 1.0)
        retry_count = state.get("retry_count", 0)

        if use_adaptive and confidence < 0.5 and retry_count <= 1:
            self._logger.info(
                "retry_target_schema",
                confidence=confidence,
                retry_count=retry_count,
            )
            return "schema"

        return "extract"

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
        Human review node - pauses the graph for human review when a
        checkpointer is configured (LangGraph v3 ``interrupt`` flow), or
        marks the run for offline review otherwise.

        Resume contract (when paused via ``interrupt``):
            - ``runner.resume_extraction(thread_id)`` with no payload accepts
              the extraction as-is.
            - ``runner.resume_extraction(thread_id, corrections={...})``
              merges the corrections into ``merged_extraction`` and resumes.

        Without a checkpointer (e.g. unit tests with
        ``enable_checkpointing=False``), the node falls back to the legacy
        behaviour of marking the extraction as ``HUMAN_REVIEW`` and ending.
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

        # Mark the state so any external dashboard sees the pending review,
        # then either pause for resume (v3 path) or end (legacy path).
        state = request_human_review(state, reason)

        # WS-5a: LangGraph v3 interrupt-resume path. Only available when:
        #   * a checkpointer is enabled (interrupt() requires durable state)
        #   * the langgraph.types primitives are importable
        # Otherwise fall through and return the human-review-marked state as
        # a terminal node, preserving prior behaviour for unit tests.
        if self._enable_checkpointing and interrupt is not None:
            payload = {
                "reason": reason,
                "processing_id": state.get("processing_id"),
                "overall_confidence": overall_confidence,
                "extracted": state.get("merged_extraction", {}),
                "validation": state.get("validation", {}),
            }
            # interrupt() pauses execution; the value returned here on resume
            # is whatever the caller passes via Command(resume=value).
            corrections = interrupt(payload)

            # Single funnel for both "accept as-is" (empty dict / None) and
            # "apply corrections" cases. _apply_human_corrections handles an
            # empty dict by overlaying nothing and still finalising via
            # complete_extraction, so this also moves the run to status
            # COMPLETED when the reviewer simply approves.
            state = self._apply_human_corrections(state, corrections or {})

        return state

    def _apply_human_corrections(
        self,
        state: ExtractionState,
        corrections: Any,
    ) -> ExtractionState:
        """Merge reviewer corrections into the extraction and complete the run.

        Accepts either:
            * a ``dict`` of ``{field_name: corrected_value}`` to overlay onto
              ``merged_extraction``
            * a ``dict`` containing a ``"fields"`` sub-dict (compatibility
              with pre-v3 callers that wrap corrections in an envelope)

        Corrected fields are wrapped in the same value/confidence/
        human_corrected envelope used by ``PipelineRunner._apply_human_corrections``
        so downstream consumers see consistent shapes regardless of which
        path applied the corrections. The corrected field names are also
        recorded in ``state["human_corrections"]`` for the audit trail.
        """
        from src.pipeline.state import complete_extraction, update_state

        # Normalise envelope shape.
        if isinstance(corrections, dict) and "fields" in corrections and isinstance(
            corrections["fields"], dict
        ):
            field_corrections = corrections["fields"]
        elif isinstance(corrections, dict):
            field_corrections = corrections
        else:
            self._logger.warning(
                "human_corrections_unexpected_shape",
                shape=type(corrections).__name__,
            )
            field_corrections = {}

        merged_extraction = dict(state.get("merged_extraction", {}) or {})
        for field_name, corrected_value in field_corrections.items():
            existing = merged_extraction.get(field_name)
            if isinstance(existing, dict):
                existing["value"] = corrected_value
                existing["confidence"] = 1.0
                existing["human_corrected"] = True
            else:
                merged_extraction[field_name] = {
                    "value": corrected_value,
                    "confidence": 1.0,
                    "human_corrected": True,
                }

        self._logger.info(
            "human_corrections_applied",
            processing_id=state.get("processing_id"),
            corrected_fields=list(field_corrections.keys()),
        )

        state = update_state(
            state,
            {
                "merged_extraction": merged_extraction,
                "human_corrections": field_corrections,
            },
        )
        return complete_extraction(state)

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
    enable_splitter: bool = True,
    enable_table_detection: bool = True,
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
        enable_splitter: Whether to enable the document boundary splitter
            (Phase 2A). Default ``True``. Disable for single-document PDFs to
            save one VLM batch per ~5 pages.
        enable_table_detection: Whether to enable cell-level table detection
            (Phase 2B). Default ``True``. Disable for documents that are
            known to contain no tables (free-text reports, etc.).

    Returns:
        Tuple of (orchestrator_agent, compiled_workflow).
    """
    # Import agents here to avoid circular imports
    from src.agents.analyzer import AnalyzerAgent
    from src.agents.extractor import ExtractorAgent
    from src.agents.validator import ValidatorAgent
    from src.config import get_settings

    settings = get_settings()

    # Create shared client
    shared_client = client or LMStudioClient()

    # --- Multi-model routing (Phase 3C, opt-in via settings) ---
    # When enabled, agents consult the router in BaseAgent.send_vision_request
    # to pick a task-appropriate model. Without a second model registered the
    # router falls through to the default, so enabling this is a no-op until
    # operators populate ``settings.model_routing.task_models``.
    model_router = None
    if getattr(settings, "model_routing", None) and settings.model_routing.enabled:
        try:
            from src.client.model_router import (
                ModelConfig,
                ModelRouter,
                ModelTask,
                qwen3vl_config,
            )

            primary = qwen3vl_config(
                base_url=settings.model_routing.default_base_url,
                model_id=settings.model_routing.default_model,
            )
            extra_models: list[ModelConfig] = []
            for task_name, model_id in settings.model_routing.task_models.items():
                # Only register additional models that aren't the primary.
                if model_id == primary.model_id:
                    continue
                try:
                    task = ModelTask(task_name)
                except ValueError:
                    logger.warning("model_routing_unknown_task", task=task_name)
                    continue
                extra_models.append(
                    ModelConfig(
                        name=model_id,
                        model_id=model_id,
                        base_url=settings.model_routing.default_base_url,
                        capabilities={task},
                        priority=20,
                    )
                )

            model_router = ModelRouter(
                models=[primary, *extra_models],
                default_model_name=primary.name,
            )
            logger.info(
                "model_router_created",
                models=[primary.name] + [m.name for m in extra_models],
            )
        except Exception as e:
            logger.warning("model_router_init_failed", error=str(e))

    # --- Confidence calibration (opt-in via settings) ---
    calibrator = None
    if settings.calibration.enabled:
        try:
            from src.validation.calibration import ConfidenceCalibrator

            calibrator = ConfidenceCalibrator(
                storage_path=settings.calibration.storage_path,
            )
            logger.info(
                "calibration_enabled",
                method=calibrator.active_method,
                samples=calibrator.sample_count,
            )
        except Exception as e:
            logger.warning("calibration_init_failed", error=str(e))

    # --- Dynamic prompt enhancement (always created — lightweight) ---
    prompt_enhancer = None
    try:
        from src.memory.dynamic_prompt import DynamicPromptEnhancer

        prompt_enhancer = DynamicPromptEnhancer()
        logger.info("prompt_enhancer_created")
    except Exception as e:
        logger.warning("prompt_enhancer_init_failed", error=str(e))

    # Create legacy pipeline agents (always needed)
    analyzer = AnalyzerAgent(client=shared_client)
    extractor = ExtractorAgent(
        client=shared_client,
        prompt_enhancer=prompt_enhancer,
    )
    validator = ValidatorAgent(
        client=shared_client,
        calibrator=calibrator,
    )

    # WS-2: attach the model router to every agent that has one. Subclasses
    # don't need to thread router through their __init__ signatures because
    # BaseAgent exposes set_model_router for post-init injection.
    if model_router is not None:
        analyzer.set_model_router(model_router)
        extractor.set_model_router(model_router)
        validator.set_model_router(model_router)

    # Create VLM-first pipeline agents if enabled
    layout_agent = None
    component_agent = None
    schema_agent = None

    if enable_vlm_first:
        try:
            from src.agents.component_detector import ComponentDetectorAgent
            from src.agents.layout_agent import LayoutAgent
            from src.agents.schema_generator import SchemaGeneratorAgent

            layout_agent = LayoutAgent(client=shared_client)
            component_agent = ComponentDetectorAgent(client=shared_client)
            schema_agent = SchemaGeneratorAgent(client=shared_client)

            if model_router is not None:
                layout_agent.set_model_router(model_router)
                component_agent.set_model_router(model_router)
                schema_agent.set_model_router(model_router)

            logger.info("vlm_first_agents_created")
        except ImportError as e:
            logger.warning(
                "vlm_first_agents_import_failed",
                error=str(e),
                message="Falling back to legacy pipeline only",
            )
            enable_vlm_first = False

    # WS-2: Default-enable the Splitter (Phase 2A) and TableDetector (Phase 2B)
    # agents that previously existed but were never wired into the default
    # workflow. They degrade gracefully if their imports fail.
    splitter_agent: BaseAgent | None = None
    if enable_splitter:
        try:
            from src.agents.splitter import SplitterAgent

            splitter_agent = SplitterAgent(client=shared_client)
            if model_router is not None:
                splitter_agent.set_model_router(model_router)
            logger.info("splitter_agent_created")
        except ImportError as e:
            logger.warning(
                "splitter_agent_import_failed",
                error=str(e),
                message="Document boundary detection disabled for this run.",
            )

    table_detector_agent: BaseAgent | None = None
    if enable_table_detection:
        try:
            from src.agents.table_detector import TableDetectorAgent

            table_detector_agent = TableDetectorAgent(client=shared_client)
            if model_router is not None:
                table_detector_agent.set_model_router(model_router)
            logger.info("table_detector_agent_created")
        except ImportError as e:
            logger.warning(
                "table_detector_agent_import_failed",
                error=str(e),
                message="Table structure detection disabled for this run.",
            )

    # V3 Phase 2 — heterogeneous dual-VLM extraction (opt-in via
    # ``settings.extraction.engine == dual_vlm``). When the flag is
    # ``legacy`` (default), Pass 1/Pass 2/reconciler stay unwired and
    # the legacy single-VLM extractor handles everything.
    extractor_pass1_agent: BaseAgent | None = None
    extractor_pass2_agent: BaseAgent | None = None
    reconciler_node: Callable[[ExtractionState], ExtractionState] | None = None
    from src.config.settings import ExtractionEngine

    if settings.extraction.engine == ExtractionEngine.DUAL_VLM:
        try:
            from src.agents.extractor_pass1 import ExtractorPass1Agent
            from src.agents.extractor_pass2 import ExtractorPass2Agent
            from src.agents.reconciler import HeterogeneousReconciler
            from src.validation.bbox_roundtrip import perform_bbox_roundtrip

            extractor_pass1_agent = ExtractorPass1Agent(client=shared_client)
            extractor_pass2_agent = ExtractorPass2Agent(client=shared_client)
            if model_router is not None:
                extractor_pass1_agent.set_model_router(model_router)
                extractor_pass2_agent.set_model_router(model_router)

            # Build a reconciler node closure that runs the field-level
            # fusion across all pages and writes ``merged_extraction`` +
            # ``reconciliation_metadata`` (V2 behaviour) PLUS
            # ``merged_extraction_v2`` + ``provenance_index`` (V3 Phase 4
            # provenance threading) into state. The dual-write keeps
            # legacy exporters working unchanged while new exporters
            # opt into the FieldValue-shaped path.
            def _reconcile_state(state: ExtractionState) -> ExtractionState:
                from src.agents.reconciler import HeterogeneousReconciler  # noqa: F401
                from src.pipeline.provenance import (
                    Provenance,
                    wrap_value,
                )

                reconciler = HeterogeneousReconciler(
                    backend=None,
                    roundtrip_helper=perform_bbox_roundtrip,
                )
                pass1 = state.get("pass1_result", {}) or {}
                pass2 = state.get("pass2_result", {}) or {}
                page_images = state.get("page_images", []) or []
                modalities = list(state.get("modalities", []) or [])
                doc_type = state.get("document_type", "UNKNOWN")
                pass1_model_id = state.get("pass1_model_id", "") or ""
                pass2_model_id = state.get("pass2_model_id", "") or ""
                enforce_wrapper = settings.provenance.enforce_field_value_wrapper

                # Fuse per page, then merge across pages by overwriting
                # nulls. The legacy extractor uses the same shape.
                merged: dict[str, Any] = {}
                field_meta: dict[str, dict[str, Any]] = {}
                merged_v2: dict[str, dict[str, Any]] = {}
                provenance_index: dict[str, list[str]] = {}
                tiebreakers: dict[str, int] = {}
                disagreements = 0
                total_fields = 0
                agreed_fields = 0

                for page in page_images:
                    page_no = page.get("page_number", 0) or 0
                    p1_payload = pass1.get(page_no, {}) or {}
                    p2_payload = pass2.get(page_no, {}) or {}
                    p1_fields = (
                        p1_payload.get("fields", {})
                        if isinstance(p1_payload, dict)
                        else {}
                    )
                    p2_fields = (
                        p2_payload.get("fields", {})
                        if isinstance(p2_payload, dict)
                        else {}
                    )
                    if not isinstance(p1_fields, dict) or not isinstance(p2_fields, dict):
                        continue
                    image_data = page.get("data_uri") or page.get("base64_encoded", "")
                    report = reconciler.reconcile(
                        pass1_fields=p1_fields,
                        pass2_fields=p2_fields,
                        page_image_data=image_data,
                        modalities=modalities,
                        doc_type=doc_type,
                    )
                    total_fields += len(report.fields)
                    if report.fields:
                        agreed_fields += sum(
                            1
                            for f in report.fields.values()
                            if f.tiebreaker in (None, "exact_match")
                        )
                    disagreements += report.disagreement_count
                    for tk, count in report.tiebreakers_used.items():
                        tiebreakers[tk] = tiebreakers.get(tk, 0) + count
                    for name, rf in report.fields.items():
                        # Merge: prefer first non-null value seen.
                        if rf.value is not None and (
                            name not in merged or merged[name] is None
                        ):
                            merged[name] = rf.value
                            field_meta[name] = {
                                "value": rf.value,
                                "confidence": rf.confidence,
                                "bbox": rf.bbox,
                                "source_pass": rf.source_pass,
                                "tiebreaker": rf.tiebreaker,
                                "page_number": page_no,
                            }

                            # V3 Phase 4 — build the FieldValue twin.
                            # Pick the model id according to the winning
                            # pass; fallback to primary when unknown.
                            if rf.source_pass == "pass2":
                                vlm_model_id = pass2_model_id
                            elif rf.source_pass == "pass1":
                                vlm_model_id = pass1_model_id
                            else:
                                # ``both``, ``roundtrip``, ``history``,
                                # ``low_confidence`` — primary as default.
                                vlm_model_id = pass1_model_id

                            extraction_path: list[str] = []
                            if rf.source_pass in ("pass1", "both"):
                                extraction_path.append("pass1_vlm")
                            if rf.source_pass in ("pass2", "both"):
                                extraction_path.append("pass2_vlm")
                            extraction_path.append("reconciler")
                            if rf.tiebreaker and rf.tiebreaker != "exact_match":
                                extraction_path.append(
                                    f"reconciler:{rf.tiebreaker}"
                                )

                            # rf.bbox is a list[float] | None from the
                            # reconciler; wrap into BoundingBoxCoords for
                            # Provenance.
                            from src.pipeline.state import BoundingBoxCoords

                            bbox_obj = None
                            if rf.bbox is not None and len(rf.bbox) >= 4:
                                try:
                                    bbox_obj = BoundingBoxCoords(
                                        x=float(rf.bbox[0]),
                                        y=float(rf.bbox[1]),
                                        width=float(rf.bbox[2] - rf.bbox[0]),
                                        height=float(rf.bbox[3] - rf.bbox[1]),
                                        page=page_no,
                                    )
                                except (TypeError, ValueError):
                                    bbox_obj = None

                            prov = Provenance(
                                page=page_no,
                                bbox=bbox_obj,
                                source_block_id="",
                                extraction_path=extraction_path,
                                agent_signatures=["extractor", "reconciler"],
                                confidence=rf.confidence,
                                vlm_model_id=vlm_model_id,
                                mem0_match=None,
                            )
                            merged_v2[name] = wrap_value(
                                rf.value, provenance=prov
                            ).to_serialisable()
                            provenance_index[name] = list(extraction_path)

                agreement_rate = (
                    agreed_fields / total_fields if total_fields > 0 else 1.0
                )
                metadata = {
                    "agreement_rate": agreement_rate,
                    "disagreements": disagreements,
                    "tiebreakers_used": tiebreakers,
                    "total_fields": total_fields,
                }
                update: dict[str, Any] = {
                    "merged_extraction_v2": merged_v2,
                    "provenance_index": provenance_index,
                    "field_metadata": field_meta,
                    "reconciliation_metadata": metadata,
                    "extraction_engine": "dual_vlm",
                }
                # Dual-write the legacy ``merged_extraction`` only when
                # the wrapper-enforce flag is OFF. When it's ON, the
                # legacy path is empty and downstream consumers must
                # use ``merged_extraction_v2`` (via ``unwrap_value``).
                if enforce_wrapper:
                    update["merged_extraction"] = {}
                else:
                    update["merged_extraction"] = merged
                return update_state(state, update)

            reconciler_node = _reconcile_state
            logger.info("dual_vlm_engine_wired")
        except Exception as exc:  # pragma: no cover - opt-in path
            logger.warning(
                "dual_vlm_engine_init_failed",
                error=str(exc),
                message="Falling back to legacy extractor.",
            )
            extractor_pass1_agent = None
            extractor_pass2_agent = None
            reconciler_node = None

    # Create orchestrator
    orchestrator = OrchestratorAgent(
        client=shared_client,
        enable_checkpointing=enable_checkpointing,
        max_retries=max_retries,
    )

    # V3 Phase 3 — Critic agent + combiner. Independent of dual-VLM:
    # operators can enable the Critic on legacy (same-model second
    # opinion) or on dual-VLM (family-rotated independent verifier).
    critic_agent: BaseAgent | None = None
    critic_combiner_node: Callable[[ExtractionState], ExtractionState] | None = None

    if settings.extraction.critic_enabled:
        try:
            from src.agents.critic import CriticAgent
            from src.validation.critic_combiner import apply_combiner_to_state

            critic_agent = CriticAgent(client=shared_client)
            if model_router is not None:
                critic_agent.set_model_router(model_router)

            def _combiner_node(state: ExtractionState) -> ExtractionState:
                """Apply the combiner; write ``confidence_components``.

                Also rewrites ``overall_confidence`` to the combiner's
                ``raw_combined`` so downstream confidence-based routing
                consumes the calibrator-ready value. Calibration itself
                runs in the validator (legacy) — a follow-up phase will
                call the calibrator here explicitly.
                """
                components = apply_combiner_to_state(state)
                merged_update: dict[str, Any] = {
                    "confidence_components": components,
                    "overall_confidence": components["raw_combined"],
                }
                # Recompute confidence_level so the route node sees a
                # consistent (level, score) pair.
                from src.pipeline.state import ConfidenceLevel

                raw = components["raw_combined"]
                if raw >= settings.extraction.confidence_auto_accept:
                    merged_update["confidence_level"] = ConfidenceLevel.HIGH.value
                elif raw >= settings.extraction.confidence_retry:
                    merged_update["confidence_level"] = ConfidenceLevel.MEDIUM.value
                else:
                    merged_update["confidence_level"] = ConfidenceLevel.LOW.value
                return update_state(state, merged_update)

            critic_combiner_node = _combiner_node
            logger.info("critic_agent_wired")
        except Exception as exc:  # pragma: no cover - opt-in path
            logger.warning(
                "critic_agent_init_failed",
                error=str(exc),
                message="Critic disabled for this run; continuing without it.",
            )
            critic_agent = None
            critic_combiner_node = None

    # Build workflow with appropriate agents
    orchestrator.build_workflow(
        preprocess_fn=preprocess_fn,
        analyzer=analyzer,
        extractor=extractor,
        validator=validator,
        layout_agent=layout_agent,
        component_agent=component_agent,
        schema_agent=schema_agent,
        splitter_agent=splitter_agent,
        table_detector_agent=table_detector_agent,
        extractor_pass1_agent=extractor_pass1_agent,
        extractor_pass2_agent=extractor_pass2_agent,
        reconciler_node=reconciler_node,
        critic_agent=critic_agent,
        critic_combiner_node=critic_combiner_node,
    )

    compiled_workflow = orchestrator.compile_workflow()

    logger.info(
        "extraction_workflow_created",
        checkpointing=enable_checkpointing,
        max_retries=max_retries,
        vlm_first_enabled=enable_vlm_first,
        splitter_enabled=splitter_agent is not None,
        table_detection_enabled=table_detector_agent is not None,
        calibration_enabled=calibrator is not None,
        prompt_enhancement_enabled=prompt_enhancer is not None,
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
