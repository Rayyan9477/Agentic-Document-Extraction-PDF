"""
Tests for the Multi-Agent Optimization Framework.

Tests cover:
- Performance profiling
- Cost optimization
- Intelligent caching
- Parallel execution
- Performance monitoring
- Integration utilities
"""

import time
import threading
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from src.agents.optimization import (
    AgentMetrics,
    PipelineMetrics,
    PerformanceProfiler,
    ModelCostTier,
    ModelConfig,
    CostOptimizer,
    IntelligentCache,
    CacheEntry,
    ParallelExecutor,
    PerformanceMonitor,
    OptimizedOrchestrator,
    get_profiler,
    DEFAULT_MODELS,
)
from src.agents.optimization_integration import (
    ProfiledAgentMixin,
    profile_operation,
    create_vlm_cache,
    create_extraction_cache_key,
    create_validation_cache_key,
    CostAwareAgent,
    extract_pages_parallel,
    create_optimized_orchestrator,
    OptimizedPipeline,
    estimate_task_complexity,
    format_optimization_report,
)


# =============================================================================
# AgentMetrics Tests
# =============================================================================


class TestAgentMetrics:
    """Tests for AgentMetrics dataclass."""

    def test_metrics_initialization(self) -> None:
        """Test basic metrics initialization."""
        metrics = AgentMetrics(
            agent_name="extractor",
            operation="extract",
            start_time=datetime.now(timezone.utc),
        )

        assert metrics.agent_name == "extractor"
        assert metrics.operation == "extract"
        assert metrics.vlm_calls == 0
        assert metrics.cache_hits == 0
        assert metrics.errors == []

    def test_duration_calculation(self) -> None:
        """Test duration calculation."""
        start = datetime.now(timezone.utc)
        metrics = AgentMetrics(
            agent_name="test",
            operation="test",
            start_time=start,
        )

        # Before end time is set
        assert metrics.duration_ms == 0

        # After end time is set
        time.sleep(0.01)  # 10ms
        metrics.end_time = datetime.now(timezone.utc)
        assert metrics.duration_ms >= 10

    def test_avg_vlm_latency(self) -> None:
        """Test average VLM latency calculation."""
        metrics = AgentMetrics(
            agent_name="test",
            operation="test",
            start_time=datetime.now(timezone.utc),
            vlm_calls=3,
            vlm_latency_ms=300,
        )

        assert metrics.avg_vlm_latency_ms == 100.0

    def test_avg_vlm_latency_zero_calls(self) -> None:
        """Test average VLM latency with no calls."""
        metrics = AgentMetrics(
            agent_name="test",
            operation="test",
            start_time=datetime.now(timezone.utc),
        )

        assert metrics.avg_vlm_latency_ms == 0.0

    def test_cache_hit_rate(self) -> None:
        """Test cache hit rate calculation."""
        metrics = AgentMetrics(
            agent_name="test",
            operation="test",
            start_time=datetime.now(timezone.utc),
            cache_hits=7,
            cache_misses=3,
        )

        assert metrics.cache_hit_rate == 70.0

    def test_cache_hit_rate_no_accesses(self) -> None:
        """Test cache hit rate with no accesses."""
        metrics = AgentMetrics(
            agent_name="test",
            operation="test",
            start_time=datetime.now(timezone.utc),
        )

        assert metrics.cache_hit_rate == 0.0

    def test_to_dict(self) -> None:
        """Test dictionary conversion."""
        metrics = AgentMetrics(
            agent_name="test",
            operation="test",
            start_time=datetime.now(timezone.utc),
            vlm_calls=2,
            input_tokens=100,
            output_tokens=50,
        )
        metrics.end_time = datetime.now(timezone.utc)

        result = metrics.to_dict()

        assert result["agent_name"] == "test"
        assert result["vlm_calls"] == 2
        assert result["input_tokens"] == 100
        assert result["output_tokens"] == 50
        assert "duration_ms" in result


# =============================================================================
# PipelineMetrics Tests
# =============================================================================


class TestPipelineMetrics:
    """Tests for PipelineMetrics dataclass."""

    def test_pipeline_metrics_initialization(self) -> None:
        """Test pipeline metrics initialization."""
        metrics = PipelineMetrics(
            processing_id="test_123",
            start_time=datetime.now(timezone.utc),
        )

        assert metrics.processing_id == "test_123"
        assert metrics.total_vlm_calls == 0
        assert metrics.agent_metrics == []

    def test_add_agent_metrics(self) -> None:
        """Test adding agent metrics to pipeline."""
        pipeline = PipelineMetrics(
            processing_id="test",
            start_time=datetime.now(timezone.utc),
        )

        agent_metrics = AgentMetrics(
            agent_name="extractor",
            operation="extract",
            start_time=datetime.now(timezone.utc),
            vlm_calls=2,
            vlm_latency_ms=500,
            input_tokens=200,
            output_tokens=100,
        )

        pipeline.add_agent_metrics(agent_metrics)

        assert len(pipeline.agent_metrics) == 1
        assert pipeline.total_vlm_calls == 2
        assert pipeline.total_vlm_latency_ms == 500
        assert pipeline.total_input_tokens == 200
        assert pipeline.total_output_tokens == 100

    def test_vlm_time_percentage(self) -> None:
        """Test VLM time percentage calculation."""
        start = datetime.now(timezone.utc)
        pipeline = PipelineMetrics(
            processing_id="test",
            start_time=start,
            total_vlm_latency_ms=500,
        )

        # Set end time to create 1000ms duration
        time.sleep(0.1)
        pipeline.end_time = datetime.now(timezone.utc)

        # VLM percentage should be calculated
        assert pipeline.vlm_time_percentage > 0


# =============================================================================
# PerformanceProfiler Tests
# =============================================================================


class TestPerformanceProfiler:
    """Tests for PerformanceProfiler class."""

    def test_start_and_end_pipeline(self) -> None:
        """Test starting and ending pipeline profiling."""
        profiler = PerformanceProfiler()

        pipeline = profiler.start_pipeline("test_123")
        assert pipeline.processing_id == "test_123"

        result = profiler.end_pipeline()
        assert result is not None
        assert result.end_time is not None

    def test_start_and_end_agent(self) -> None:
        """Test starting and ending agent profiling."""
        profiler = PerformanceProfiler()
        profiler.start_pipeline("test")

        metrics = profiler.start_agent("extractor", "extract")
        assert metrics.agent_name == "extractor"
        assert metrics.operation == "extract"

        profiler.end_agent(metrics)
        assert metrics.end_time is not None

        profiler.end_pipeline()

    def test_record_vlm_call(self) -> None:
        """Test recording VLM calls."""
        profiler = PerformanceProfiler()
        metrics = profiler.start_agent("test", "test")

        profiler.record_vlm_call(metrics, latency_ms=100, input_tokens=50, output_tokens=25)

        assert metrics.vlm_calls == 1
        assert metrics.vlm_latency_ms == 100
        assert metrics.input_tokens == 50
        assert metrics.output_tokens == 25

    def test_record_cache_hit_miss(self) -> None:
        """Test recording cache hits and misses."""
        profiler = PerformanceProfiler()
        metrics = profiler.start_agent("test", "test")

        profiler.record_cache_hit(metrics)
        profiler.record_cache_hit(metrics)
        profiler.record_cache_miss(metrics)

        assert metrics.cache_hits == 2
        assert metrics.cache_misses == 1

    def test_get_history(self) -> None:
        """Test getting profiler history."""
        profiler = PerformanceProfiler()

        # Run a few pipelines
        for i in range(3):
            profiler.start_pipeline(f"test_{i}")
            profiler.end_pipeline()

        history = profiler.get_history()
        assert len(history) == 3

    def test_get_aggregate_stats(self) -> None:
        """Test getting aggregate statistics."""
        profiler = PerformanceProfiler()

        # Run some pipelines
        for i in range(3):
            profiler.start_pipeline(f"test_{i}")
            time.sleep(0.01)
            profiler.end_pipeline()

        stats = profiler.get_aggregate_stats()

        assert stats["total_pipelines"] == 3
        assert stats["avg_duration_ms"] > 0

    def test_get_aggregate_stats_empty(self) -> None:
        """Test aggregate stats with no history."""
        profiler = PerformanceProfiler()
        stats = profiler.get_aggregate_stats()
        assert stats == {}

    def test_global_profiler(self) -> None:
        """Test global profiler instance."""
        profiler1 = get_profiler()
        profiler2 = get_profiler()
        assert profiler1 is profiler2


# =============================================================================
# CostOptimizer Tests
# =============================================================================


class TestCostOptimizer:
    """Tests for CostOptimizer class."""

    def test_record_usage(self) -> None:
        """Test recording token usage."""
        optimizer = CostOptimizer(monthly_budget_usd=100.0)

        cost = optimizer.record_usage(
            model_name="claude-3-sonnet",
            input_tokens=1000,
            output_tokens=500,
        )

        assert cost > 0
        assert optimizer.get_total_cost() == cost

    def test_unknown_model_uses_default(self) -> None:
        """Test unknown model falls back to default pricing."""
        optimizer = CostOptimizer()

        # Should not raise
        cost = optimizer.record_usage(
            model_name="unknown-model",
            input_tokens=1000,
            output_tokens=500,
        )

        assert cost > 0

    def test_remaining_budget(self) -> None:
        """Test remaining budget calculation."""
        optimizer = CostOptimizer(monthly_budget_usd=100.0)

        initial_remaining = optimizer.get_remaining_budget()
        assert initial_remaining == 100.0

        optimizer.record_usage("claude-3-haiku", 10000, 5000)
        assert optimizer.get_remaining_budget() < 100.0

    def test_budget_utilization(self) -> None:
        """Test budget utilization percentage."""
        optimizer = CostOptimizer(monthly_budget_usd=100.0)

        assert optimizer.get_budget_utilization() == 0.0

        # Record some usage
        optimizer.record_usage("claude-3-opus", 10000, 10000)
        assert optimizer.get_budget_utilization() > 0.0

    def test_select_optimal_model_high_complexity(self) -> None:
        """Test model selection for high complexity tasks."""
        optimizer = CostOptimizer(monthly_budget_usd=100.0)

        model = optimizer.select_optimal_model(
            task_complexity=0.9,
            quality_threshold=0.8,
        )

        assert model.tier == ModelCostTier.PREMIUM

    def test_select_optimal_model_low_budget(self) -> None:
        """Test model selection with low budget."""
        optimizer = CostOptimizer(monthly_budget_usd=1.0)

        # Use up most of budget
        optimizer.record_usage("claude-3-opus", 100000, 50000)

        model = optimizer.select_optimal_model(
            task_complexity=0.5,
            quality_threshold=0.5,
        )

        assert model.tier == ModelCostTier.ECONOMY

    def test_get_usage_report(self) -> None:
        """Test getting usage report."""
        optimizer = CostOptimizer(monthly_budget_usd=100.0)

        optimizer.record_usage("claude-3-sonnet", 1000, 500)
        optimizer.record_usage("claude-3-haiku", 2000, 1000)

        report = optimizer.get_usage_report()

        assert "total_cost_usd" in report
        assert "remaining_budget_usd" in report
        assert "usage_by_model" in report
        assert report["call_count"] == 2


# =============================================================================
# IntelligentCache Tests
# =============================================================================


class TestIntelligentCache:
    """Tests for IntelligentCache class."""

    def test_set_and_get(self) -> None:
        """Test basic set and get operations."""
        cache: IntelligentCache[str] = IntelligentCache(max_size=100)

        cache.set("key1", "value1")
        result = cache.get("key1")

        assert result == "value1"

    def test_get_nonexistent(self) -> None:
        """Test getting nonexistent key."""
        cache: IntelligentCache[str] = IntelligentCache()

        result = cache.get("nonexistent")
        assert result is None

    def test_ttl_expiration(self) -> None:
        """Test TTL expiration."""
        cache: IntelligentCache[str] = IntelligentCache(default_ttl_seconds=1)

        cache.set("key1", "value1", ttl_seconds=1)

        # Should exist immediately
        assert cache.get("key1") == "value1"

        # Wait for expiration
        time.sleep(1.1)
        assert cache.get("key1") is None

    def test_lru_eviction(self) -> None:
        """Test LRU eviction when at capacity."""
        cache: IntelligentCache[str] = IntelligentCache(max_size=3)

        cache.set("key1", "value1")
        time.sleep(0.01)  # Ensure distinct timestamps
        cache.set("key2", "value2")
        time.sleep(0.01)
        cache.set("key3", "value3")
        time.sleep(0.01)

        # Access key1 and key3 to make them recently used
        cache.get("key1")
        time.sleep(0.01)
        cache.get("key3")
        time.sleep(0.01)

        # Add key4 - should evict key2 (least recently used)
        cache.set("key4", "value4")

        # Verify cache size is still 3
        stats = cache.get_stats()
        assert stats["size"] == 3

        # key2 should be evicted (oldest access time)
        assert cache.get("key2") is None  # Evicted

        # Others should still exist
        assert cache.get("key1") == "value1"
        assert cache.get("key3") == "value3"
        assert cache.get("key4") == "value4"

    def test_invalidate(self) -> None:
        """Test cache invalidation."""
        cache: IntelligentCache[str] = IntelligentCache()

        cache.set("key1", "value1")
        assert cache.get("key1") == "value1"

        result = cache.invalidate("key1")
        assert result is True
        assert cache.get("key1") is None

        # Invalidating nonexistent key
        result = cache.invalidate("nonexistent")
        assert result is False

    def test_clear(self) -> None:
        """Test clearing cache."""
        cache: IntelligentCache[str] = IntelligentCache()

        cache.set("key1", "value1")
        cache.set("key2", "value2")

        cache.clear()

        assert cache.get("key1") is None
        assert cache.get("key2") is None

    def test_get_stats(self) -> None:
        """Test getting cache statistics."""
        cache: IntelligentCache[str] = IntelligentCache(max_size=100)

        cache.set("key1", "value1")
        cache.get("key1")  # Hit
        cache.get("key1")  # Hit
        cache.get("nonexistent")  # Miss

        stats = cache.get_stats()

        assert stats["size"] == 1
        assert stats["hits"] == 2
        assert stats["misses"] == 1
        assert stats["hit_rate_pct"] == pytest.approx(66.67, rel=0.01)

    def test_thread_safety(self) -> None:
        """Test thread-safe operations."""
        cache: IntelligentCache[int] = IntelligentCache(max_size=1000)
        errors: list[Exception] = []

        def writer(thread_id: int) -> None:
            try:
                for i in range(100):
                    cache.set(f"key_{thread_id}_{i}", i)
            except Exception as e:
                errors.append(e)

        def reader(thread_id: int) -> None:
            try:
                for i in range(100):
                    cache.get(f"key_{thread_id}_{i}")
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(5):
            threads.append(threading.Thread(target=writer, args=(i,)))
            threads.append(threading.Thread(target=reader, args=(i,)))

        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# =============================================================================
# ParallelExecutor Tests
# =============================================================================


class TestParallelExecutor:
    """Tests for ParallelExecutor class."""

    def test_execute_parallel(self) -> None:
        """Test parallel execution of tasks."""
        def slow_task(x: int) -> int:
            time.sleep(0.01)
            return x * 2

        tasks = [
            (slow_task, (1,), {}),
            (slow_task, (2,), {}),
            (slow_task, (3,), {}),
        ]

        with ParallelExecutor(max_workers=3) as executor:
            results = executor.execute_parallel(tasks)

        assert results == [2, 4, 6]

    def test_execute_parallel_with_exceptions(self) -> None:
        """Test handling of exceptions in parallel tasks."""
        def failing_task(x: int) -> int:
            if x == 2:
                raise ValueError("Task failed")
            return x * 2

        tasks = [
            (failing_task, (1,), {}),
            (failing_task, (2,), {}),
            (failing_task, (3,), {}),
        ]

        with ParallelExecutor(max_workers=3) as executor:
            results = executor.execute_parallel(tasks)

        assert results[0] == 2
        assert isinstance(results[1], ValueError)
        assert results[2] == 6

    def test_map_parallel(self) -> None:
        """Test parallel map operation."""
        def double(x: int) -> int:
            return x * 2

        with ParallelExecutor(max_workers=4) as executor:
            results = executor.map_parallel(double, [1, 2, 3, 4, 5])

        assert results == [2, 4, 6, 8, 10]

    def test_context_manager_required(self) -> None:
        """Test that context manager is required."""
        executor = ParallelExecutor()

        with pytest.raises(RuntimeError, match="Executor not initialized"):
            executor.execute_parallel([])


# =============================================================================
# PerformanceMonitor Tests
# =============================================================================


class TestPerformanceMonitor:
    """Tests for PerformanceMonitor class."""

    def test_process_metrics_no_alerts(self) -> None:
        """Test processing metrics without triggering alerts."""
        monitor = PerformanceMonitor(
            alert_latency_threshold_ms=10000,
            alert_cost_threshold_usd=100.0,
        )

        metrics = PipelineMetrics(
            processing_id="test",
            start_time=datetime.now(timezone.utc),
        )
        metrics.end_time = datetime.now(timezone.utc)
        metrics.estimated_cost_usd = 0.01

        alerts = monitor.process_metrics(metrics)
        assert len(alerts) == 0

    def test_process_metrics_latency_alert(self) -> None:
        """Test latency alert generation."""
        monitor = PerformanceMonitor(alert_latency_threshold_ms=100)

        start = datetime.now(timezone.utc)
        time.sleep(0.15)  # 150ms

        metrics = PipelineMetrics(
            processing_id="test",
            start_time=start,
        )
        metrics.end_time = datetime.now(timezone.utc)

        alerts = monitor.process_metrics(metrics)

        latency_alerts = [a for a in alerts if a["type"] == "latency"]
        assert len(latency_alerts) == 1
        assert latency_alerts[0]["severity"] == "warning"

    def test_process_metrics_cost_alert(self) -> None:
        """Test cost alert generation."""
        monitor = PerformanceMonitor(alert_cost_threshold_usd=1.0)

        metrics = PipelineMetrics(
            processing_id="test",
            start_time=datetime.now(timezone.utc),
            estimated_cost_usd=2.0,
        )
        metrics.end_time = datetime.now(timezone.utc)

        alerts = monitor.process_metrics(metrics)

        cost_alerts = [a for a in alerts if a["type"] == "cost"]
        assert len(cost_alerts) == 1

    def test_process_metrics_error_alert(self) -> None:
        """Test error alert generation."""
        monitor = PerformanceMonitor()

        metrics = PipelineMetrics(
            processing_id="test",
            start_time=datetime.now(timezone.utc),
        )
        metrics.end_time = datetime.now(timezone.utc)

        # Add agent with errors
        agent_metrics = AgentMetrics(
            agent_name="test",
            operation="test",
            start_time=datetime.now(timezone.utc),
            errors=["Error 1", "Error 2"],
        )
        metrics.add_agent_metrics(agent_metrics)

        alerts = monitor.process_metrics(metrics)

        error_alerts = [a for a in alerts if a["type"] == "error"]
        assert len(error_alerts) == 1
        assert error_alerts[0]["severity"] == "error"

    def test_get_dashboard_data_no_data(self) -> None:
        """Test dashboard data with no metrics."""
        monitor = PerformanceMonitor()
        data = monitor.get_dashboard_data()

        assert data["status"] == "no_data"

    def test_get_dashboard_data_with_metrics(self) -> None:
        """Test dashboard data with metrics."""
        monitor = PerformanceMonitor()

        # Add some metrics
        for i in range(5):
            metrics = PipelineMetrics(
                processing_id=f"test_{i}",
                start_time=datetime.now(timezone.utc),
            )
            metrics.end_time = datetime.now(timezone.utc)
            monitor.process_metrics(metrics)

        data = monitor.get_dashboard_data()

        assert data["status"] == "healthy"
        assert data["summary"]["pipeline_count"] == 5

    def test_clear_alerts(self) -> None:
        """Test clearing alerts."""
        monitor = PerformanceMonitor(alert_latency_threshold_ms=1)

        # Generate an alert
        start = datetime.now(timezone.utc)
        time.sleep(0.01)
        metrics = PipelineMetrics(
            processing_id="test",
            start_time=start,
        )
        metrics.end_time = datetime.now(timezone.utc)
        monitor.process_metrics(metrics)

        # Should have alerts
        data = monitor.get_dashboard_data()
        assert len(data["alerts"]) > 0

        # Clear and verify
        monitor.clear_alerts()
        data = monitor.get_dashboard_data()
        assert len(data["alerts"]) == 0


# =============================================================================
# OptimizedOrchestrator Tests
# =============================================================================


class TestOptimizedOrchestrator:
    """Tests for OptimizedOrchestrator class."""

    def test_initialization(self) -> None:
        """Test orchestrator initialization."""
        orchestrator = OptimizedOrchestrator()

        assert orchestrator.profiler is not None
        assert orchestrator.cost_optimizer is not None
        assert orchestrator.cache is not None
        assert orchestrator.monitor is not None

    def test_optimize_extraction_sequential(self) -> None:
        """Test sequential extraction optimization."""
        orchestrator = OptimizedOrchestrator(enable_parallel=False)

        pages = [{"id": 1}]
        extract_fn = MagicMock(return_value={"field": "value"})

        results = orchestrator.optimize_extraction(pages, extract_fn)

        assert len(results) == 1
        assert extract_fn.call_count == 1

    def test_optimize_extraction_parallel(self) -> None:
        """Test parallel extraction optimization."""
        orchestrator = OptimizedOrchestrator(
            enable_parallel=True,
            parallel_workers=2,
        )

        pages = [{"id": 1}, {"id": 2}, {"id": 3}]
        extract_fn = MagicMock(side_effect=lambda p: {"field": p["id"]})

        results = orchestrator.optimize_extraction(pages, extract_fn)

        assert len(results) == 3

    def test_get_optimization_report(self) -> None:
        """Test getting optimization report."""
        orchestrator = OptimizedOrchestrator()

        report = orchestrator.get_optimization_report()

        assert "cache" in report
        assert "cost" in report
        assert "profiler" in report
        assert "dashboard" in report
        assert "recommendations" in report


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegrationUtilities:
    """Tests for integration utilities."""

    def test_create_vlm_cache(self) -> None:
        """Test VLM cache creation."""
        cache = create_vlm_cache(max_size=100, ttl_seconds=600)

        assert cache._max_size == 100
        assert cache._default_ttl == 600

    def test_create_extraction_cache_key(self) -> None:
        """Test extraction cache key generation."""
        key = create_extraction_cache_key(
            image_hash="abc123",
            schema_name="invoice",
            pass_number=1,
        )

        assert key == "extract:abc123:invoice:pass1"

    def test_create_validation_cache_key(self) -> None:
        """Test validation cache key generation."""
        key = create_validation_cache_key(
            extraction_hash="def456",
            schema_name="invoice",
        )

        assert key == "validate:def456:invoice"

    def test_create_optimized_orchestrator(self) -> None:
        """Test optimized orchestrator factory."""
        orchestrator = create_optimized_orchestrator(
            monthly_budget_usd=50.0,
            cache_max_size=200,
            enable_parallel=True,
        )

        assert orchestrator.cost_optimizer.monthly_budget_usd == 50.0
        assert orchestrator.cache._max_size == 200
        assert orchestrator.enable_parallel is True

    def test_estimate_task_complexity_base(self) -> None:
        """Test base task complexity estimation."""
        state = {"document_type": "unknown"}
        complexity = estimate_task_complexity(state)
        assert complexity == 0.5

    def test_estimate_task_complexity_medical(self) -> None:
        """Test complexity for medical documents."""
        state = {"document_type": "medical_record"}
        complexity = estimate_task_complexity(state)
        assert complexity == 0.7

    def test_estimate_task_complexity_high_page_count(self) -> None:
        """Test complexity for multi-page documents."""
        state = {"document_type": "report", "page_count": 15}
        complexity = estimate_task_complexity(state)
        assert complexity == 0.7

    def test_format_optimization_report(self) -> None:
        """Test optimization report formatting."""
        report = {
            "cache": {"hit_rate_pct": 75.5, "size": 50, "max_size": 100},
            "cost": {
                "total_cost_usd": 1.23,
                "budget_utilization_pct": 12.3,
                "remaining_budget_usd": 87.67,
            },
            "profiler": {
                "total_pipelines": 10,
                "avg_duration_ms": 1500,
                "avg_vlm_calls": 3.5,
            },
            "recommendations": ["Consider increasing cache size"],
        }

        formatted = format_optimization_report(report)

        assert "Hit Rate: 75.5%" in formatted
        assert "Total Cost: $1.2300" in formatted
        assert "Consider increasing cache size" in formatted


class TestProfileOperation:
    """Tests for profile_operation context manager."""

    def test_profile_operation_context(self) -> None:
        """Test profile operation context manager."""
        mock_agent = MagicMock()
        mock_agent.name = "test_agent"
        mock_agent.vlm_calls = 2
        mock_agent.total_processing_ms = 500

        profiler = PerformanceProfiler()
        profiler.start_pipeline("test")

        with profile_operation(mock_agent, "test_op", profiler) as metrics:
            assert metrics.agent_name == "test_agent"
            assert metrics.operation == "test_op"

        # After context, metrics should be updated
        assert metrics.vlm_calls == 2
        assert metrics.vlm_latency_ms == 500

        profiler.end_pipeline()


# =============================================================================
# Model Configuration Tests
# =============================================================================


class TestModelConfiguration:
    """Tests for model configuration."""

    def test_default_models_exist(self) -> None:
        """Test default model configurations exist."""
        assert "claude-3-opus" in DEFAULT_MODELS
        assert "claude-3-sonnet" in DEFAULT_MODELS
        assert "claude-3-haiku" in DEFAULT_MODELS

    def test_model_tiers_correct(self) -> None:
        """Test model tier assignments."""
        assert DEFAULT_MODELS["claude-3-opus"].tier == ModelCostTier.PREMIUM
        assert DEFAULT_MODELS["claude-3-sonnet"].tier == ModelCostTier.STANDARD
        assert DEFAULT_MODELS["claude-3-haiku"].tier == ModelCostTier.ECONOMY

    def test_quality_scores_ordered(self) -> None:
        """Test quality scores are properly ordered."""
        opus = DEFAULT_MODELS["claude-3-opus"]
        sonnet = DEFAULT_MODELS["claude-3-sonnet"]
        haiku = DEFAULT_MODELS["claude-3-haiku"]

        assert opus.quality_score > sonnet.quality_score > haiku.quality_score
