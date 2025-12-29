"""
Optimization Framework Integration Module.

Provides integration utilities to connect the optimization framework
with the existing agent architecture. Enables seamless performance
profiling, caching, and cost optimization.
"""

from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Any, Callable, Generator

from src.agents.base import BaseAgent, AgentError
from src.agents.optimization import (
    get_profiler,
    PerformanceProfiler,
    AgentMetrics,
    PipelineMetrics,
    CostOptimizer,
    IntelligentCache,
    ParallelExecutor,
    PerformanceMonitor,
    OptimizedOrchestrator,
)
from src.config import get_logger
from src.pipeline.state import ExtractionState


logger = get_logger(__name__)


# =============================================================================
# Agent Profiling Integration
# =============================================================================


class ProfiledAgentMixin:
    """
    Mixin class to add profiling capabilities to agents.

    Can be used with any agent that inherits from BaseAgent.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        """Initialize with profiler reference."""
        super().__init__(*args, **kwargs)
        self._profiler = get_profiler()
        self._current_metrics: AgentMetrics | None = None

    def start_profiling(self, operation: str) -> AgentMetrics:
        """
        Start profiling an operation.

        Args:
            operation: Name of the operation being profiled.

        Returns:
            AgentMetrics instance for tracking.
        """
        self._current_metrics = self._profiler.start_agent(
            agent_name=getattr(self, "_name", "unknown"),
            operation=operation,
        )
        return self._current_metrics

    def end_profiling(self) -> None:
        """End current profiling session."""
        if self._current_metrics:
            self._profiler.end_agent(self._current_metrics)
            self._current_metrics = None

    def record_vlm_call_metrics(
        self,
        latency_ms: int,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        """
        Record VLM call metrics.

        Args:
            latency_ms: Call latency in milliseconds.
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.
        """
        if self._current_metrics:
            self._profiler.record_vlm_call(
                self._current_metrics,
                latency_ms=latency_ms,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

    def record_cache_result(self, hit: bool) -> None:
        """
        Record cache hit/miss.

        Args:
            hit: True if cache hit, False if miss.
        """
        if self._current_metrics:
            if hit:
                self._profiler.record_cache_hit(self._current_metrics)
            else:
                self._profiler.record_cache_miss(self._current_metrics)


@contextmanager
def profile_operation(
    agent: BaseAgent,
    operation: str,
    profiler: PerformanceProfiler | None = None,
) -> Generator[AgentMetrics, None, None]:
    """
    Context manager for profiling an agent operation.

    Args:
        agent: Agent being profiled.
        operation: Operation name.
        profiler: Optional profiler instance (uses global if not provided).

    Yields:
        AgentMetrics instance for tracking.

    Example:
        with profile_operation(extractor, "extraction") as metrics:
            result = extractor.process(state)
            metrics.vlm_calls = extractor.vlm_calls
    """
    profiler = profiler or get_profiler()
    metrics = profiler.start_agent(agent.name, operation)

    try:
        yield metrics
    finally:
        # Capture agent metrics
        metrics.vlm_calls = agent.vlm_calls
        metrics.vlm_latency_ms = agent.total_processing_ms
        profiler.end_agent(metrics)


# =============================================================================
# Caching Integration
# =============================================================================


def create_vlm_cache(
    max_size: int = 500,
    ttl_seconds: int = 1800,
) -> IntelligentCache[dict[str, Any]]:
    """
    Create a cache optimized for VLM responses.

    Args:
        max_size: Maximum cache entries.
        ttl_seconds: Time-to-live in seconds.

    Returns:
        Configured IntelligentCache instance.
    """
    return IntelligentCache[dict[str, Any]](
        max_size=max_size,
        default_ttl_seconds=ttl_seconds,
    )


def create_extraction_cache_key(
    image_hash: str,
    schema_name: str,
    pass_number: int,
) -> str:
    """
    Create a cache key for extraction results.

    Args:
        image_hash: Hash of the document image.
        schema_name: Name of the extraction schema.
        pass_number: Extraction pass number.

    Returns:
        Cache key string.
    """
    return f"extract:{image_hash}:{schema_name}:pass{pass_number}"


def create_validation_cache_key(
    extraction_hash: str,
    schema_name: str,
) -> str:
    """
    Create a cache key for validation results.

    Args:
        extraction_hash: Hash of extraction results.
        schema_name: Name of the extraction schema.

    Returns:
        Cache key string.
    """
    return f"validate:{extraction_hash}:{schema_name}"


# =============================================================================
# Cost Optimization Integration
# =============================================================================


class CostAwareAgent:
    """
    Mixin to add cost awareness to agents.

    Tracks token usage and selects optimal models based on budget.
    """

    def __init__(
        self,
        *args: Any,
        cost_optimizer: CostOptimizer | None = None,
        **kwargs: Any,
    ) -> None:
        """Initialize with cost optimizer."""
        super().__init__(*args, **kwargs)
        self._cost_optimizer = cost_optimizer or CostOptimizer()

    def record_token_usage(
        self,
        model_name: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """
        Record token usage and get cost.

        Args:
            model_name: Model used.
            input_tokens: Input token count.
            output_tokens: Output token count.

        Returns:
            Cost in USD.
        """
        return self._cost_optimizer.record_usage(
            model_name=model_name,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
        )

    def get_optimal_model(
        self,
        task_complexity: float,
        quality_threshold: float = 0.7,
    ) -> str:
        """
        Get optimal model for a task.

        Args:
            task_complexity: Estimated complexity (0-1).
            quality_threshold: Minimum acceptable quality.

        Returns:
            Model name.
        """
        config = self._cost_optimizer.select_optimal_model(
            task_complexity=task_complexity,
            quality_threshold=quality_threshold,
        )
        return config.name

    def check_budget(self, warn_threshold: float = 0.8) -> bool:
        """
        Check if budget is okay.

        Args:
            warn_threshold: Utilization threshold for warning.

        Returns:
            True if budget is okay, False if over threshold.
        """
        utilization = self._cost_optimizer.get_budget_utilization()
        if utilization > warn_threshold * 100:
            logger.warning(
                "budget_utilization_high",
                utilization_pct=utilization,
                threshold_pct=warn_threshold * 100,
            )
            return False
        return True


# =============================================================================
# Parallel Execution Integration
# =============================================================================


def extract_pages_parallel(
    pages: list[dict[str, Any]],
    extractor: BaseAgent,
    max_workers: int = 4,
) -> list[dict[str, Any]]:
    """
    Extract from multiple pages in parallel.

    Args:
        pages: List of page data dictionaries.
        extractor: Extractor agent instance.
        max_workers: Maximum parallel workers.

    Returns:
        List of extraction results.
    """
    if len(pages) <= 1:
        # No benefit from parallelization
        return [_extract_single_page(extractor, page) for page in pages]

    with ParallelExecutor(max_workers=max_workers) as executor:
        results = executor.map_parallel(
            lambda page: _extract_single_page(extractor, page),
            pages,
        )

    # Convert exceptions to error dicts
    processed_results = []
    for i, result in enumerate(results):
        if isinstance(result, Exception):
            logger.warning(f"page_{i}_extraction_failed", error=str(result))
            processed_results.append({
                "error": str(result),
                "page_number": i,
                "success": False,
            })
        else:
            processed_results.append(result)

    return processed_results


def _extract_single_page(
    extractor: BaseAgent,
    page: dict[str, Any],
) -> dict[str, Any]:
    """
    Extract from a single page.

    Args:
        extractor: Extractor agent.
        page: Page data dictionary.

    Returns:
        Extraction result.
    """
    try:
        # Create minimal state for single page
        state: ExtractionState = {
            "processing_id": page.get("processing_id", ""),
            "pdf_path": page.get("pdf_path", ""),
            "pdf_images": [page.get("image_data", "")],
            "page_count": 1,
            "current_page": page.get("page_number", 0),
            "document_type": page.get("document_type", ""),
            "schema_name": page.get("schema_name", ""),
            "status": "extracting",
            "current_step": "page_extraction",
        }

        result_state = extractor.process(state)
        return {
            "success": True,
            "page_number": page.get("page_number", 0),
            "extracted_fields": result_state.get("extracted_fields", {}),
            "field_metadata": result_state.get("field_metadata", {}),
        }
    except AgentError as e:
        return {
            "success": False,
            "page_number": page.get("page_number", 0),
            "error": str(e),
        }


# =============================================================================
# Optimized Orchestrator Integration
# =============================================================================


def create_optimized_orchestrator(
    monthly_budget_usd: float = 100.0,
    cache_max_size: int = 500,
    cache_ttl_seconds: int = 1800,
    enable_parallel: bool = True,
    parallel_workers: int = 4,
    alert_latency_ms: int = 5000,
    alert_error_rate: float = 0.1,
    alert_cost_usd: float = 10.0,
) -> OptimizedOrchestrator:
    """
    Create a fully configured optimized orchestrator.

    Args:
        monthly_budget_usd: Monthly cost budget.
        cache_max_size: Maximum cache entries.
        cache_ttl_seconds: Cache TTL.
        enable_parallel: Enable parallel execution.
        parallel_workers: Number of parallel workers.
        alert_latency_ms: Latency alert threshold.
        alert_error_rate: Error rate alert threshold.
        alert_cost_usd: Cost alert threshold.

    Returns:
        Configured OptimizedOrchestrator.
    """
    profiler = PerformanceProfiler()
    cost_optimizer = CostOptimizer(monthly_budget_usd=monthly_budget_usd)
    cache = IntelligentCache[dict[str, Any]](
        max_size=cache_max_size,
        default_ttl_seconds=cache_ttl_seconds,
    )
    monitor = PerformanceMonitor(
        alert_latency_threshold_ms=alert_latency_ms,
        alert_error_rate_threshold=alert_error_rate,
        alert_cost_threshold_usd=alert_cost_usd,
    )

    return OptimizedOrchestrator(
        profiler=profiler,
        cost_optimizer=cost_optimizer,
        cache=cache,
        monitor=monitor,
        enable_parallel=enable_parallel,
        parallel_workers=parallel_workers,
    )


# =============================================================================
# Pipeline Integration
# =============================================================================


class OptimizedPipeline:
    """
    Optimized extraction pipeline with full performance monitoring.

    Wraps the entire extraction workflow with profiling, caching,
    cost tracking, and parallel execution.
    """

    def __init__(
        self,
        orchestrator: OptimizedOrchestrator | None = None,
    ) -> None:
        """
        Initialize optimized pipeline.

        Args:
            orchestrator: Optional pre-configured orchestrator.
        """
        self.orchestrator = orchestrator or create_optimized_orchestrator()
        self._pipeline_count = 0

    def run_extraction(
        self,
        state: ExtractionState,
        analyzer: BaseAgent,
        extractor: BaseAgent,
        validator: BaseAgent,
    ) -> ExtractionState:
        """
        Run optimized extraction pipeline.

        Args:
            state: Initial extraction state.
            analyzer: Analyzer agent.
            extractor: Extractor agent.
            validator: Validator agent.

        Returns:
            Final extraction state with optimization metrics.
        """
        self._pipeline_count += 1
        processing_id = state.get("processing_id", f"pipeline_{self._pipeline_count}")

        # Start pipeline profiling
        pipeline_metrics = self.orchestrator.profiler.start_pipeline(processing_id)

        try:
            # Analysis phase
            with profile_operation(analyzer, "analysis", self.orchestrator.profiler):
                state = analyzer.process(state)

            # Extraction phase (with potential parallel page processing)
            pages = state.get("pdf_images", [])
            if len(pages) > 1 and self.orchestrator.enable_parallel:
                # Prepare page data for parallel processing
                page_data = [
                    {
                        "processing_id": processing_id,
                        "pdf_path": state.get("pdf_path", ""),
                        "image_data": img,
                        "page_number": i,
                        "document_type": state.get("document_type", ""),
                        "schema_name": state.get("schema_name", ""),
                    }
                    for i, img in enumerate(pages)
                ]

                # Extract in parallel
                with profile_operation(extractor, "extraction", self.orchestrator.profiler):
                    results = extract_pages_parallel(
                        page_data,
                        extractor,
                        max_workers=self.orchestrator.parallel_workers,
                    )

                # Merge results into state
                merged_fields = {}
                for result in results:
                    if result.get("success"):
                        merged_fields.update(result.get("extracted_fields", {}))

                state["extracted_fields"] = merged_fields
            else:
                with profile_operation(extractor, "extraction", self.orchestrator.profiler):
                    state = extractor.process(state)

            # Validation phase
            with profile_operation(validator, "validation", self.orchestrator.profiler):
                state = validator.process(state)

            # End pipeline profiling
            pipeline_metrics = self.orchestrator.profiler.end_pipeline()

            # Process metrics for monitoring
            if pipeline_metrics:
                alerts = self.orchestrator.monitor.process_metrics(pipeline_metrics)
                if alerts:
                    logger.warning(
                        "pipeline_alerts_generated",
                        alert_count=len(alerts),
                        alerts=[a["type"] for a in alerts],
                    )

            # Add optimization metrics to state
            state["optimization_metrics"] = self.get_optimization_summary()

            return state

        except Exception as e:
            # End profiling even on error
            self.orchestrator.profiler.end_pipeline()
            raise

    def get_optimization_summary(self) -> dict[str, Any]:
        """
        Get optimization summary.

        Returns:
            Summary of optimization metrics.
        """
        return self.orchestrator.get_optimization_report()


# =============================================================================
# Utility Functions
# =============================================================================


def estimate_task_complexity(state: ExtractionState) -> float:
    """
    Estimate task complexity for model selection.

    Args:
        state: Extraction state.

    Returns:
        Complexity score from 0.0 to 1.0.
    """
    complexity = 0.5  # Base complexity

    # Adjust based on document type
    doc_type = state.get("document_type", "").lower()
    if "medical" in doc_type or "legal" in doc_type or "financial" in doc_type:
        complexity += 0.2

    # Adjust based on page count
    page_count = state.get("page_count", 1)
    if page_count > 5:
        complexity += 0.1
    if page_count > 10:
        complexity += 0.1

    # Adjust based on field count
    schema = state.get("schema")
    if schema:
        field_count = len(schema.get("fields", []))
        if field_count > 20:
            complexity += 0.1

    return min(1.0, complexity)


def format_optimization_report(report: dict[str, Any]) -> str:
    """
    Format optimization report for display.

    Args:
        report: Optimization report dictionary.

    Returns:
        Formatted string report.
    """
    lines = [
        "=" * 60,
        "OPTIMIZATION REPORT",
        "=" * 60,
        "",
        "## Cache Performance",
        f"  Hit Rate: {report.get('cache', {}).get('hit_rate_pct', 0):.1f}%",
        f"  Size: {report.get('cache', {}).get('size', 0)} / {report.get('cache', {}).get('max_size', 0)}",
        "",
        "## Cost Summary",
        f"  Total Cost: ${report.get('cost', {}).get('total_cost_usd', 0):.4f}",
        f"  Budget Used: {report.get('cost', {}).get('budget_utilization_pct', 0):.1f}%",
        f"  Remaining: ${report.get('cost', {}).get('remaining_budget_usd', 0):.2f}",
        "",
        "## Performance",
        f"  Pipelines Run: {report.get('profiler', {}).get('total_pipelines', 0)}",
        f"  Avg Duration: {report.get('profiler', {}).get('avg_duration_ms', 0):.0f}ms",
        f"  Avg VLM Calls: {report.get('profiler', {}).get('avg_vlm_calls', 0):.1f}",
        "",
        "## Recommendations",
    ]

    for rec in report.get("recommendations", []):
        lines.append(f"  - {rec}")

    if not report.get("recommendations"):
        lines.append("  (No recommendations)")

    lines.extend(["", "=" * 60])

    return "\n".join(lines)
