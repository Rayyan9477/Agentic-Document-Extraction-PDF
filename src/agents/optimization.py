"""
Multi-Agent Optimization Framework.

Provides comprehensive optimization utilities for the multi-agent extraction pipeline:
- Performance profiling across all agents
- Parallel execution optimization
- Cost tracking and optimization
- Latency reduction with intelligent caching
- Performance monitoring and dashboards
- Adaptive agent coordination
"""

import hashlib
import json
import threading
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any, Generic, TypeVar

from src.config import get_logger


logger = get_logger(__name__)

T = TypeVar("T")


# =============================================================================
# Performance Profiling
# =============================================================================


@dataclass
class AgentMetrics:
    """Metrics for a single agent execution."""

    agent_name: str
    operation: str
    start_time: datetime
    end_time: datetime | None = None
    vlm_calls: int = 0
    vlm_latency_ms: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def duration_ms(self) -> int:
        """Calculate duration in milliseconds."""
        if self.end_time is None:
            return 0
        return int((self.end_time - self.start_time).total_seconds() * 1000)

    @property
    def avg_vlm_latency_ms(self) -> float:
        """Average VLM call latency."""
        if self.vlm_calls == 0:
            return 0.0
        return self.vlm_latency_ms / self.vlm_calls

    @property
    def cache_hit_rate(self) -> float:
        """Cache hit rate as percentage."""
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return 0.0
        return (self.cache_hits / total) * 100

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "agent_name": self.agent_name,
            "operation": self.operation,
            "duration_ms": self.duration_ms,
            "vlm_calls": self.vlm_calls,
            "vlm_latency_ms": self.vlm_latency_ms,
            "avg_vlm_latency_ms": self.avg_vlm_latency_ms,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "cache_hit_rate": self.cache_hit_rate,
            "errors": self.errors,
        }


@dataclass
class PipelineMetrics:
    """Aggregated metrics for the entire pipeline."""

    processing_id: str
    start_time: datetime
    end_time: datetime | None = None
    agent_metrics: list[AgentMetrics] = field(default_factory=list)
    total_vlm_calls: int = 0
    total_vlm_latency_ms: int = 0
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    estimated_cost_usd: float = 0.0

    @property
    def duration_ms(self) -> int:
        """Total pipeline duration in milliseconds."""
        if self.end_time is None:
            return 0
        return int((self.end_time - self.start_time).total_seconds() * 1000)

    @property
    def vlm_time_percentage(self) -> float:
        """Percentage of time spent on VLM calls."""
        if self.duration_ms == 0:
            return 0.0
        return (self.total_vlm_latency_ms / self.duration_ms) * 100

    def add_agent_metrics(self, metrics: AgentMetrics) -> None:
        """Add agent metrics to pipeline."""
        self.agent_metrics.append(metrics)
        self.total_vlm_calls += metrics.vlm_calls
        self.total_vlm_latency_ms += metrics.vlm_latency_ms
        self.total_input_tokens += metrics.input_tokens
        self.total_output_tokens += metrics.output_tokens

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "processing_id": self.processing_id,
            "duration_ms": self.duration_ms,
            "total_vlm_calls": self.total_vlm_calls,
            "total_vlm_latency_ms": self.total_vlm_latency_ms,
            "vlm_time_percentage": self.vlm_time_percentage,
            "total_input_tokens": self.total_input_tokens,
            "total_output_tokens": self.total_output_tokens,
            "estimated_cost_usd": self.estimated_cost_usd,
            "agents": [m.to_dict() for m in self.agent_metrics],
        }


class PerformanceProfiler:
    """
    Performance profiler for multi-agent pipelines.

    Tracks and aggregates performance metrics across all agents.
    """

    def __init__(self) -> None:
        """Initialize profiler."""
        self._current_pipeline: PipelineMetrics | None = None
        self._current_agent: AgentMetrics | None = None
        self._history: list[PipelineMetrics] = []
        self._lock = threading.Lock()

    def start_pipeline(self, processing_id: str) -> PipelineMetrics:
        """Start profiling a new pipeline execution."""
        with self._lock:
            self._current_pipeline = PipelineMetrics(
                processing_id=processing_id,
                start_time=datetime.now(UTC),
            )
            return self._current_pipeline

    def end_pipeline(self) -> PipelineMetrics | None:
        """End current pipeline profiling."""
        with self._lock:
            if self._current_pipeline:
                self._current_pipeline.end_time = datetime.now(UTC)
                self._history.append(self._current_pipeline)
                result = self._current_pipeline
                self._current_pipeline = None
                return result
            return None

    def start_agent(self, agent_name: str, operation: str) -> AgentMetrics:
        """Start profiling an agent operation."""
        metrics = AgentMetrics(
            agent_name=agent_name,
            operation=operation,
            start_time=datetime.now(UTC),
        )
        self._current_agent = metrics
        return metrics

    def end_agent(self, metrics: AgentMetrics) -> None:
        """End agent profiling and add to pipeline."""
        metrics.end_time = datetime.now(UTC)
        with self._lock:
            if self._current_pipeline:
                self._current_pipeline.add_agent_metrics(metrics)

    def record_vlm_call(
        self,
        metrics: AgentMetrics,
        latency_ms: int,
        input_tokens: int = 0,
        output_tokens: int = 0,
    ) -> None:
        """Record a VLM call for the current agent."""
        metrics.vlm_calls += 1
        metrics.vlm_latency_ms += latency_ms
        metrics.input_tokens += input_tokens
        metrics.output_tokens += output_tokens

    def record_cache_hit(self, metrics: AgentMetrics) -> None:
        """Record a cache hit."""
        metrics.cache_hits += 1

    def record_cache_miss(self, metrics: AgentMetrics) -> None:
        """Record a cache miss."""
        metrics.cache_misses += 1

    def get_history(self, limit: int = 100) -> list[PipelineMetrics]:
        """Get historical pipeline metrics."""
        return self._history[-limit:]

    def get_aggregate_stats(self) -> dict[str, Any]:
        """Get aggregate statistics across all executions."""
        if not self._history:
            return {}

        total_pipelines = len(self._history)
        avg_duration = sum(p.duration_ms for p in self._history) / total_pipelines
        avg_vlm_calls = sum(p.total_vlm_calls for p in self._history) / total_pipelines
        total_cost = sum(p.estimated_cost_usd for p in self._history)

        return {
            "total_pipelines": total_pipelines,
            "avg_duration_ms": avg_duration,
            "avg_vlm_calls": avg_vlm_calls,
            "total_estimated_cost_usd": total_cost,
        }


# Global profiler instance
_profiler = PerformanceProfiler()


def get_profiler() -> PerformanceProfiler:
    """Get the global performance profiler."""
    return _profiler


# =============================================================================
# Cost Optimization
# =============================================================================


class ModelCostTier(str, Enum):
    """Model cost tiers for optimization."""

    PREMIUM = "premium"  # Highest quality, highest cost
    STANDARD = "standard"  # Good balance of quality and cost
    ECONOMY = "economy"  # Lower cost, suitable for simple tasks


@dataclass
class ModelConfig:
    """Configuration for an LLM model."""

    name: str
    tier: ModelCostTier
    input_cost_per_1k: float  # Cost per 1K input tokens
    output_cost_per_1k: float  # Cost per 1K output tokens
    max_tokens: int
    quality_score: float  # 0-1 quality estimate


# Default model configurations
DEFAULT_MODELS = {
    "claude-3-opus": ModelConfig(
        name="claude-3-opus",
        tier=ModelCostTier.PREMIUM,
        input_cost_per_1k=0.015,
        output_cost_per_1k=0.075,
        max_tokens=4096,
        quality_score=1.0,
    ),
    "claude-3-sonnet": ModelConfig(
        name="claude-3-sonnet",
        tier=ModelCostTier.STANDARD,
        input_cost_per_1k=0.003,
        output_cost_per_1k=0.015,
        max_tokens=4096,
        quality_score=0.85,
    ),
    "claude-3-haiku": ModelConfig(
        name="claude-3-haiku",
        tier=ModelCostTier.ECONOMY,
        input_cost_per_1k=0.00025,
        output_cost_per_1k=0.00125,
        max_tokens=4096,
        quality_score=0.7,
    ),
}


class CostOptimizer:
    """
    Cost optimizer for LLM usage.

    Tracks token usage and provides cost-aware model selection.
    """

    def __init__(
        self,
        monthly_budget_usd: float = 100.0,
        models: dict[str, ModelConfig] | None = None,
    ) -> None:
        """
        Initialize cost optimizer.

        Args:
            monthly_budget_usd: Monthly budget in USD.
            models: Model configurations (defaults to DEFAULT_MODELS).
        """
        self.monthly_budget_usd = monthly_budget_usd
        self.models = models or DEFAULT_MODELS
        self._usage: dict[str, dict[str, int]] = defaultdict(lambda: {"input": 0, "output": 0})
        self._cost_history: list[tuple[datetime, str, float]] = []
        self._lock = threading.Lock()

    def record_usage(
        self,
        model_name: str,
        input_tokens: int,
        output_tokens: int,
    ) -> float:
        """
        Record token usage and return cost.

        Args:
            model_name: Name of the model used.
            input_tokens: Number of input tokens.
            output_tokens: Number of output tokens.

        Returns:
            Cost of this usage in USD.
        """
        model = self.models.get(model_name)
        if not model:
            logger.warning(f"Unknown model: {model_name}, using default pricing")
            model = self.models.get("claude-3-sonnet", list(self.models.values())[0])

        input_cost = (input_tokens / 1000) * model.input_cost_per_1k
        output_cost = (output_tokens / 1000) * model.output_cost_per_1k
        total_cost = input_cost + output_cost

        with self._lock:
            self._usage[model_name]["input"] += input_tokens
            self._usage[model_name]["output"] += output_tokens
            self._cost_history.append((datetime.now(UTC), model_name, total_cost))

        return total_cost

    def get_total_cost(self) -> float:
        """Get total cost for the current period."""
        return sum(cost for _, _, cost in self._cost_history)

    def get_remaining_budget(self) -> float:
        """Get remaining budget."""
        return max(0, self.monthly_budget_usd - self.get_total_cost())

    def get_budget_utilization(self) -> float:
        """Get budget utilization as percentage."""
        if self.monthly_budget_usd == 0:
            return 100.0
        return (self.get_total_cost() / self.monthly_budget_usd) * 100

    def select_optimal_model(
        self,
        task_complexity: float,
        quality_threshold: float = 0.7,
    ) -> ModelConfig:
        """
        Select optimal model based on task complexity and budget.

        Args:
            task_complexity: Estimated task complexity (0-1).
            quality_threshold: Minimum acceptable quality score.

        Returns:
            Selected model configuration.
        """
        budget_remaining_pct = self.get_remaining_budget() / max(1, self.monthly_budget_usd)

        # Filter models by quality threshold
        viable_models = [m for m in self.models.values() if m.quality_score >= quality_threshold]

        if not viable_models:
            viable_models = list(self.models.values())

        # Sort by cost efficiency (quality / cost)
        def efficiency_score(m: ModelConfig) -> float:
            avg_cost = (m.input_cost_per_1k + m.output_cost_per_1k) / 2
            if avg_cost == 0:
                return float("inf")
            return m.quality_score / avg_cost

        viable_models.sort(key=efficiency_score, reverse=True)

        # High complexity + good budget = premium model
        if task_complexity > 0.8 and budget_remaining_pct > 0.5:
            premium_models = [m for m in viable_models if m.tier == ModelCostTier.PREMIUM]
            if premium_models:
                return premium_models[0]

        # Low budget = economy model
        if budget_remaining_pct < 0.2:
            economy_models = [m for m in viable_models if m.tier == ModelCostTier.ECONOMY]
            if economy_models:
                return economy_models[0]

        # Default to best efficiency
        return viable_models[0]

    def get_usage_report(self) -> dict[str, Any]:
        """Get detailed usage report."""
        with self._lock:
            return {
                "total_cost_usd": self.get_total_cost(),
                "remaining_budget_usd": self.get_remaining_budget(),
                "budget_utilization_pct": self.get_budget_utilization(),
                "usage_by_model": dict(self._usage),
                "call_count": len(self._cost_history),
            }


# =============================================================================
# Intelligent Caching
# =============================================================================


@dataclass
class CacheEntry(Generic[T]):
    """Cache entry with metadata."""

    key: str
    value: T
    created_at: datetime
    accessed_at: datetime
    access_count: int = 0
    ttl_seconds: int = 3600


class IntelligentCache(Generic[T]):
    """
    Intelligent caching with TTL and LRU eviction.

    Designed for caching VLM responses and extraction results.
    """

    def __init__(
        self,
        max_size: int = 1000,
        default_ttl_seconds: int = 3600,
    ) -> None:
        """
        Initialize cache.

        Args:
            max_size: Maximum number of entries.
            default_ttl_seconds: Default TTL in seconds.
        """
        self._cache: dict[str, CacheEntry[T]] = {}
        self._max_size = max_size
        self._default_ttl = default_ttl_seconds
        self._lock = threading.RLock()
        self._stats = {"hits": 0, "misses": 0, "evictions": 0}

    def _generate_key(self, *args: Any, **kwargs: Any) -> str:
        """Generate cache key from arguments."""
        key_data = json.dumps({"args": args, "kwargs": kwargs}, sort_keys=True, default=str)
        return hashlib.sha256(key_data.encode()).hexdigest()

    def get(self, key: str) -> T | None:
        """
        Get value from cache.

        Args:
            key: Cache key.

        Returns:
            Cached value or None if not found/expired.
        """
        with self._lock:
            entry = self._cache.get(key)

            if entry is None:
                self._stats["misses"] += 1
                return None

            # Check TTL
            now = datetime.now(UTC)
            age = (now - entry.created_at).total_seconds()
            if age > entry.ttl_seconds:
                del self._cache[key]
                self._stats["misses"] += 1
                return None

            # Update access metadata
            entry.accessed_at = now
            entry.access_count += 1
            self._stats["hits"] += 1

            return entry.value

    def set(
        self,
        key: str,
        value: T,
        ttl_seconds: int | None = None,
    ) -> None:
        """
        Set value in cache.

        Args:
            key: Cache key.
            value: Value to cache.
            ttl_seconds: TTL override.
        """
        with self._lock:
            # Evict if at capacity
            if len(self._cache) >= self._max_size:
                self._evict_lru()

            now = datetime.now(UTC)
            self._cache[key] = CacheEntry(
                key=key,
                value=value,
                created_at=now,
                accessed_at=now,
                ttl_seconds=ttl_seconds or self._default_ttl,
            )

    def _evict_lru(self) -> None:
        """Evict least recently used entry."""
        if not self._cache:
            return

        # Find LRU entry
        lru_key = min(
            self._cache.keys(),
            key=lambda k: self._cache[k].accessed_at,
        )
        del self._cache[lru_key]
        self._stats["evictions"] += 1

    def invalidate(self, key: str) -> bool:
        """
        Invalidate a cache entry.

        Args:
            key: Cache key.

        Returns:
            True if entry was found and removed.
        """
        with self._lock:
            if key in self._cache:
                del self._cache[key]
                return True
            return False

    def clear(self) -> None:
        """Clear all cache entries."""
        with self._lock:
            self._cache.clear()

    def get_stats(self) -> dict[str, Any]:
        """Get cache statistics."""
        with self._lock:
            total = self._stats["hits"] + self._stats["misses"]
            hit_rate = (self._stats["hits"] / total * 100) if total > 0 else 0

            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._stats["hits"],
                "misses": self._stats["misses"],
                "evictions": self._stats["evictions"],
                "hit_rate_pct": hit_rate,
            }


def cached_vlm_call(cache: IntelligentCache[dict[str, Any]]):
    """
    Decorator for caching VLM calls.

    Args:
        cache: Cache instance to use.
    """

    def decorator(func: Callable[..., dict[str, Any]]) -> Callable[..., dict[str, Any]]:
        def wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
            # Generate cache key from prompt (excluding image data for now)
            prompt = kwargs.get("prompt", "")
            system_prompt = kwargs.get("system_prompt", "")
            key_data = f"{prompt}:{system_prompt}"
            cache_key = hashlib.sha256(key_data.encode()).hexdigest()

            # Try cache
            cached = cache.get(cache_key)
            if cached is not None:
                logger.debug("vlm_cache_hit", key=cache_key[:8])
                return cached

            # Execute and cache
            result = func(*args, **kwargs)
            cache.set(cache_key, result)
            logger.debug("vlm_cache_miss", key=cache_key[:8])

            return result

        return wrapper

    return decorator


# =============================================================================
# Parallel Execution
# =============================================================================


class ParallelExecutor:
    """
    Parallel executor for independent agent tasks.

    Enables concurrent execution of agents when safe to do so.
    """

    def __init__(self, max_workers: int = 4) -> None:
        """
        Initialize parallel executor.

        Args:
            max_workers: Maximum concurrent workers.
        """
        self._max_workers = max_workers
        self._executor: ThreadPoolExecutor | None = None

    def __enter__(self) -> "ParallelExecutor":
        """Enter context manager."""
        self._executor = ThreadPoolExecutor(max_workers=self._max_workers)
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Exit context manager."""
        if self._executor:
            self._executor.shutdown(wait=True)

    def execute_parallel(
        self,
        tasks: list[tuple[Callable[..., T], tuple[Any, ...], dict[str, Any]]],
        timeout: float | None = None,
    ) -> list[T | Exception]:
        """
        Execute tasks in parallel.

        Args:
            tasks: List of (function, args, kwargs) tuples.
            timeout: Optional timeout in seconds.

        Returns:
            List of results or exceptions in same order as tasks.
        """
        if self._executor is None:
            raise RuntimeError("Executor not initialized. Use context manager.")

        futures = []
        for func, args, kwargs in tasks:
            future = self._executor.submit(func, *args, **kwargs)
            futures.append(future)

        results: list[T | Exception] = []
        for future in futures:
            try:
                result = future.result(timeout=timeout)
                results.append(result)
            except Exception as e:
                results.append(e)

        return results

    def map_parallel(
        self,
        func: Callable[..., T],
        items: list[Any],
        **kwargs: Any,
    ) -> list[T | Exception]:
        """
        Map function over items in parallel.

        Args:
            func: Function to apply.
            items: Items to process.
            **kwargs: Additional kwargs for func.

        Returns:
            List of results.
        """
        tasks = [(func, (item,), kwargs) for item in items]
        return self.execute_parallel(tasks)


# =============================================================================
# Performance Monitoring Dashboard
# =============================================================================


class PerformanceMonitor:
    """
    Real-time performance monitoring for multi-agent pipelines.

    Provides live metrics and alerting capabilities.
    """

    def __init__(
        self,
        alert_latency_threshold_ms: int = 5000,
        alert_error_rate_threshold: float = 0.1,
        alert_cost_threshold_usd: float = 10.0,
    ) -> None:
        """
        Initialize performance monitor.

        Args:
            alert_latency_threshold_ms: Latency alert threshold.
            alert_error_rate_threshold: Error rate alert threshold (0-1).
            alert_cost_threshold_usd: Cost alert threshold per pipeline.
        """
        self._latency_threshold = alert_latency_threshold_ms
        self._error_rate_threshold = alert_error_rate_threshold
        self._cost_threshold = alert_cost_threshold_usd

        self._alerts: list[dict[str, Any]] = []
        self._metrics_buffer: list[PipelineMetrics] = []
        self._lock = threading.Lock()

    def process_metrics(self, metrics: PipelineMetrics) -> list[dict[str, Any]]:
        """
        Process pipeline metrics and generate alerts if needed.

        Args:
            metrics: Pipeline metrics to process.

        Returns:
            List of alerts generated.
        """
        new_alerts: list[dict[str, Any]] = []

        # Check latency
        if metrics.duration_ms > self._latency_threshold:
            alert = {
                "type": "latency",
                "severity": "warning",
                "message": f"Pipeline latency {metrics.duration_ms}ms exceeds threshold {self._latency_threshold}ms",
                "value": metrics.duration_ms,
                "threshold": self._latency_threshold,
                "timestamp": datetime.now(UTC).isoformat(),
            }
            new_alerts.append(alert)

        # Check cost
        if metrics.estimated_cost_usd > self._cost_threshold:
            alert = {
                "type": "cost",
                "severity": "warning",
                "message": f"Pipeline cost ${metrics.estimated_cost_usd:.4f} exceeds threshold ${self._cost_threshold}",
                "value": metrics.estimated_cost_usd,
                "threshold": self._cost_threshold,
                "timestamp": datetime.now(UTC).isoformat(),
            }
            new_alerts.append(alert)

        # Check for errors
        error_count = sum(len(m.errors) for m in metrics.agent_metrics)
        if error_count > 0:
            alert = {
                "type": "error",
                "severity": "error" if error_count > 1 else "warning",
                "message": f"Pipeline had {error_count} errors",
                "value": error_count,
                "timestamp": datetime.now(UTC).isoformat(),
            }
            new_alerts.append(alert)

        with self._lock:
            self._alerts.extend(new_alerts)
            self._metrics_buffer.append(metrics)

            # Keep buffer bounded
            if len(self._metrics_buffer) > 1000:
                self._metrics_buffer = self._metrics_buffer[-500:]

        return new_alerts

    def get_dashboard_data(self) -> dict[str, Any]:
        """
        Get data for performance dashboard.

        Returns:
            Dashboard data including metrics and alerts.
        """
        with self._lock:
            recent_metrics = self._metrics_buffer[-100:]
            recent_alerts = self._alerts[-50:]

        if not recent_metrics:
            return {
                "status": "no_data",
                "alerts": recent_alerts,
            }

        # Calculate aggregates
        avg_duration = sum(m.duration_ms for m in recent_metrics) / len(recent_metrics)
        avg_vlm_calls = sum(m.total_vlm_calls for m in recent_metrics) / len(recent_metrics)
        total_cost = sum(m.estimated_cost_usd for m in recent_metrics)

        # Agent-level breakdown
        agent_stats: dict[str, list[int]] = defaultdict(list)
        for pipeline in recent_metrics:
            for agent in pipeline.agent_metrics:
                agent_stats[agent.agent_name].append(agent.duration_ms)

        agent_summary = {
            name: {
                "avg_duration_ms": sum(durations) / len(durations),
                "count": len(durations),
            }
            for name, durations in agent_stats.items()
        }

        return {
            "status": "healthy" if not recent_alerts else "warning",
            "summary": {
                "pipeline_count": len(recent_metrics),
                "avg_duration_ms": avg_duration,
                "avg_vlm_calls": avg_vlm_calls,
                "total_cost_usd": total_cost,
            },
            "agents": agent_summary,
            "alerts": recent_alerts,
        }

    def clear_alerts(self) -> None:
        """Clear all alerts."""
        with self._lock:
            self._alerts.clear()


# =============================================================================
# Multi-Agent Orchestration Optimization
# =============================================================================


class OptimizedOrchestrator:
    """
    Optimized orchestrator with parallel execution and caching.

    Wraps the existing orchestrator with performance optimizations.
    """

    def __init__(
        self,
        profiler: PerformanceProfiler | None = None,
        cost_optimizer: CostOptimizer | None = None,
        cache: IntelligentCache[dict[str, Any]] | None = None,
        monitor: PerformanceMonitor | None = None,
        enable_parallel: bool = True,
        parallel_workers: int = 4,
    ) -> None:
        """
        Initialize optimized orchestrator.

        Args:
            profiler: Performance profiler instance.
            cost_optimizer: Cost optimizer instance.
            cache: Intelligent cache instance.
            monitor: Performance monitor instance.
            enable_parallel: Whether to enable parallel execution.
            parallel_workers: Number of parallel workers.
        """
        self.profiler = profiler or PerformanceProfiler()
        self.cost_optimizer = cost_optimizer or CostOptimizer()
        self.cache = cache or IntelligentCache()
        self.monitor = monitor or PerformanceMonitor()
        self.enable_parallel = enable_parallel
        self.parallel_workers = parallel_workers

    def optimize_extraction(
        self,
        pages: list[dict[str, Any]],
        extract_fn: Callable[[dict[str, Any]], dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """
        Optimize multi-page extraction with parallel processing.

        Args:
            pages: List of page data dictionaries.
            extract_fn: Function to extract from a single page.

        Returns:
            List of extraction results.
        """
        if not self.enable_parallel or len(pages) == 1:
            # Sequential extraction
            return [extract_fn(page) for page in pages]

        # Parallel extraction
        with ParallelExecutor(max_workers=self.parallel_workers) as executor:
            results = executor.map_parallel(extract_fn, pages)

        # Filter out exceptions
        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.warning(f"Page {i} extraction failed: {result}")
                valid_results.append({"error": str(result), "page": i})
            else:
                valid_results.append(result)

        return valid_results

    def get_optimization_report(self) -> dict[str, Any]:
        """
        Get comprehensive optimization report.

        Returns:
            Optimization report with metrics and recommendations.
        """
        cache_stats = self.cache.get_stats()
        cost_report = self.cost_optimizer.get_usage_report()
        profiler_stats = self.profiler.get_aggregate_stats()
        dashboard_data = self.monitor.get_dashboard_data()

        # Generate recommendations
        recommendations: list[str] = []

        if cache_stats["hit_rate_pct"] < 20:
            recommendations.append("Consider increasing cache TTL or size - low hit rate detected")

        if cost_report["budget_utilization_pct"] > 80:
            recommendations.append(
                "Budget utilization high - consider using more economical models"
            )

        if profiler_stats.get("avg_vlm_calls", 0) > 4:
            recommendations.append("High VLM call count - consider consolidating prompts")

        return {
            "cache": cache_stats,
            "cost": cost_report,
            "profiler": profiler_stats,
            "dashboard": dashboard_data,
            "recommendations": recommendations,
        }
