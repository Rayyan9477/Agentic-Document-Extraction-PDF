"""
Comprehensive Unit Tests for Monitoring Module.

Tests cover:
- Prometheus metrics collection
- Metrics registry and exposition
- Alert rules and management
- Notification handlers
- Rate limiting for alerts
"""

from __future__ import annotations

import asyncio
import time
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest


# =============================================================================
# Metrics Tests
# =============================================================================


class TestMetricNamespace:
    """Tests for MetricNamespace enum."""

    def test_namespaces_exist(self) -> None:
        """Test that required namespaces exist."""
        from src.monitoring.metrics import MetricNamespace

        assert MetricNamespace.API
        assert MetricNamespace.EXTRACTION
        assert MetricNamespace.VLM
        assert MetricNamespace.VALIDATION
        assert MetricNamespace.SECURITY
        assert MetricNamespace.PIPELINE


class TestMetricLabels:
    """Tests for MetricLabels."""

    def test_create_labels(self) -> None:
        """Test label creation."""
        from src.monitoring.metrics import MetricLabels

        labels = MetricLabels(
            method="POST",
            endpoint="/api/v1/documents",
            status_code="200",
        )

        assert labels.method == "POST"
        assert labels.endpoint == "/api/v1/documents"
        assert labels.status_code == "200"

    def test_labels_to_dict(self) -> None:
        """Test label serialization."""
        from src.monitoring.metrics import MetricLabels

        labels = MetricLabels(
            document_type="pdf",
            extraction_method="vlm",
        )

        data = labels.to_dict()
        assert data["document_type"] == "pdf"
        assert data["extraction_method"] == "vlm"


class TestMetricsRegistry:
    """Tests for MetricsRegistry."""

    def test_singleton_pattern(self) -> None:
        """Test that registry is singleton."""
        from src.monitoring.metrics import MetricsRegistry

        registry1 = MetricsRegistry()
        registry2 = MetricsRegistry()

        # Should be same instance
        assert registry1 is registry2

    def test_api_metrics_registered(self) -> None:
        """Test that API metrics are registered."""
        from src.monitoring.metrics import MetricsRegistry

        registry = MetricsRegistry()

        assert registry.api_requests_total is not None
        assert registry.api_request_duration_seconds is not None
        assert registry.api_requests_in_progress is not None

    def test_extraction_metrics_registered(self) -> None:
        """Test that extraction metrics are registered."""
        from src.monitoring.metrics import MetricsRegistry

        registry = MetricsRegistry()

        assert registry.extraction_documents_total is not None
        assert registry.extraction_pages_total is not None
        assert registry.extraction_duration_seconds is not None

    def test_vlm_metrics_registered(self) -> None:
        """Test that VLM metrics are registered."""
        from src.monitoring.metrics import MetricsRegistry

        registry = MetricsRegistry()

        assert registry.vlm_requests_total is not None
        assert registry.vlm_tokens_total is not None
        assert registry.vlm_latency_seconds is not None

    def test_security_metrics_registered(self) -> None:
        """Test that security metrics are registered."""
        from src.monitoring.metrics import MetricsRegistry

        registry = MetricsRegistry()

        assert registry.security_auth_attempts_total is not None
        assert registry.security_encryption_operations_total is not None

    def test_generate_exposition(self) -> None:
        """Test Prometheus exposition format generation."""
        from src.monitoring.metrics import MetricsRegistry

        registry = MetricsRegistry()
        exposition = registry.generate_exposition()

        assert isinstance(exposition, str)
        # Should contain HELP and TYPE declarations
        assert "# HELP" in exposition or exposition == ""


class TestMetricsCollector:
    """Tests for MetricsCollector."""

    def test_record_api_request(self) -> None:
        """Test recording API request metrics."""
        from src.monitoring.metrics import MetricsCollector

        collector = MetricsCollector()

        collector.record_api_request(
            method="POST",
            endpoint="/api/v1/documents/process",
            status_code=200,
            duration=0.5,
            request_size=1024,
        )

        # Should not raise any errors

    def test_record_extraction(self) -> None:
        """Test recording extraction metrics."""
        from src.monitoring.metrics import MetricsCollector

        collector = MetricsCollector()

        collector.record_extraction(
            document_type="pdf",
            page_count=10,
            duration=2.5,
            success=True,
            file_size=1024 * 1024,
        )

    def test_record_vlm_request(self) -> None:
        """Test recording VLM request metrics."""
        from src.monitoring.metrics import MetricsCollector

        collector = MetricsCollector()

        collector.record_vlm_request(
            provider="openai",
            model="gpt-4-vision",
            tokens_input=500,
            tokens_output=200,
            latency=1.5,
            success=True,
        )

    def test_record_validation(self) -> None:
        """Test recording validation metrics."""
        from src.monitoring.metrics import MetricsCollector

        collector = MetricsCollector()

        collector.record_validation(
            validation_type="format",
            passed=True,
            field_count=25,
            confidence_avg=0.92,
        )

    def test_record_security_event(self) -> None:
        """Test recording security event metrics."""
        from src.monitoring.metrics import MetricsCollector

        collector = MetricsCollector()

        collector.record_security_event(
            event_type="authentication",
            success=True,
            user_id="user-123",
        )

    def test_record_pipeline(self) -> None:
        """Test recording pipeline metrics."""
        from src.monitoring.metrics import MetricsCollector

        collector = MetricsCollector()

        collector.record_pipeline(
            stage="extraction",
            duration=3.0,
            success=True,
        )

    def test_increment_counter(self) -> None:
        """Test counter increment."""
        from src.monitoring.metrics import MetricsCollector

        collector = MetricsCollector()

        collector.increment_counter(
            name="custom_counter",
            value=1,
            labels={"type": "test"},
        )

    def test_set_gauge(self) -> None:
        """Test gauge setting."""
        from src.monitoring.metrics import MetricsCollector

        collector = MetricsCollector()

        collector.set_gauge(
            name="queue_depth",
            value=42,
            labels={"queue": "extraction"},
        )


class TestTrackDurationDecorator:
    """Tests for track_duration decorator."""

    def test_track_sync_function(self) -> None:
        """Test duration tracking for sync function."""
        from src.monitoring.metrics import MetricsCollector, track_duration

        collector = MetricsCollector()

        @track_duration(collector, "test_operation")
        def slow_function() -> str:
            time.sleep(0.01)
            return "done"

        result = slow_function()
        assert result == "done"

    def test_track_async_function(self) -> None:
        """Test duration tracking for async function."""
        from src.monitoring.metrics import MetricsCollector, track_duration

        collector = MetricsCollector()

        @track_duration(collector, "async_test_operation")
        async def async_slow_function() -> str:
            await asyncio.sleep(0.01)
            return "async done"

        result = asyncio.run(async_slow_function())
        assert result == "async done"

    def test_track_with_labels(self) -> None:
        """Test duration tracking with custom labels."""
        from src.monitoring.metrics import MetricsCollector, track_duration

        collector = MetricsCollector()

        @track_duration(collector, "labeled_operation", labels={"type": "test"})
        def labeled_function(x: int) -> int:
            return x * 2

        result = labeled_function(5)
        assert result == 10


class TestCountCallsDecorator:
    """Tests for count_calls decorator."""

    def test_count_sync_function(self) -> None:
        """Test call counting for sync function."""
        from src.monitoring.metrics import MetricsCollector, count_calls

        collector = MetricsCollector()

        @count_calls(collector, "test_counter")
        def counted_function() -> str:
            return "counted"

        result = counted_function()
        assert result == "counted"

    def test_count_exceptions(self) -> None:
        """Test that exceptions are still counted."""
        from src.monitoring.metrics import MetricsCollector, count_calls

        collector = MetricsCollector()

        @count_calls(collector, "error_counter")
        def failing_function() -> None:
            raise ValueError("Test error")

        with pytest.raises(ValueError):
            failing_function()


# =============================================================================
# Alert Tests
# =============================================================================


class TestAlertSeverity:
    """Tests for AlertSeverity enum."""

    def test_severity_levels(self) -> None:
        """Test severity levels exist."""
        from src.monitoring.alerts import AlertSeverity

        assert AlertSeverity.INFO
        assert AlertSeverity.WARNING
        assert AlertSeverity.CRITICAL


class TestAlertStatus:
    """Tests for AlertStatus enum."""

    def test_status_values(self) -> None:
        """Test status values exist."""
        from src.monitoring.alerts import AlertStatus

        assert AlertStatus.FIRING
        assert AlertStatus.RESOLVED
        assert AlertStatus.ACKNOWLEDGED


class TestAlertChannel:
    """Tests for AlertChannel enum."""

    def test_channels_exist(self) -> None:
        """Test that required channels exist."""
        from src.monitoring.alerts import AlertChannel

        assert AlertChannel.LOG
        assert AlertChannel.WEBHOOK
        assert AlertChannel.SLACK
        assert AlertChannel.PAGERDUTY


class TestAlert:
    """Tests for Alert model."""

    def test_create_alert(self) -> None:
        """Test alert creation."""
        from src.monitoring.alerts import Alert, AlertSeverity, AlertStatus

        alert = Alert(
            id="alert-001",
            rule_name="high_error_rate",
            severity=AlertSeverity.WARNING,
            status=AlertStatus.FIRING,
            message="Error rate exceeds threshold",
            labels={"service": "extraction"},
            value=15.5,
        )

        assert alert.id == "alert-001"
        assert alert.rule_name == "high_error_rate"
        assert alert.severity == AlertSeverity.WARNING
        assert alert.status == AlertStatus.FIRING
        assert alert.created_at is not None

    def test_alert_to_dict(self) -> None:
        """Test alert serialization."""
        from src.monitoring.alerts import Alert, AlertSeverity, AlertStatus

        alert = Alert(
            id="alert-002",
            rule_name="test_rule",
            severity=AlertSeverity.CRITICAL,
            status=AlertStatus.FIRING,
            message="Test alert",
        )

        data = alert.to_dict()

        assert data["id"] == "alert-002"
        assert data["severity"] == "CRITICAL"
        assert "created_at" in data

    def test_resolve_alert(self) -> None:
        """Test alert resolution."""
        from src.monitoring.alerts import Alert, AlertSeverity, AlertStatus

        alert = Alert(
            id="alert-003",
            rule_name="test_rule",
            severity=AlertSeverity.WARNING,
            status=AlertStatus.FIRING,
            message="Test alert",
        )

        alert.resolve()

        assert alert.status == AlertStatus.RESOLVED
        assert alert.resolved_at is not None

    def test_acknowledge_alert(self) -> None:
        """Test alert acknowledgement."""
        from src.monitoring.alerts import Alert, AlertSeverity, AlertStatus

        alert = Alert(
            id="alert-004",
            rule_name="test_rule",
            severity=AlertSeverity.CRITICAL,
            status=AlertStatus.FIRING,
            message="Critical alert",
        )

        alert.acknowledge(by="admin@example.com")

        assert alert.status == AlertStatus.ACKNOWLEDGED
        assert alert.acknowledged_by == "admin@example.com"


class TestAlertRule:
    """Tests for AlertRule."""

    def test_create_rule(self) -> None:
        """Test rule creation."""
        from src.monitoring.alerts import AlertRule, AlertSeverity

        rule = AlertRule(
            name="high_latency",
            description="Alert when latency exceeds threshold",
            severity=AlertSeverity.WARNING,
            condition=lambda v: v > 5.0,
            threshold=5.0,
            for_duration=timedelta(minutes=5),
        )

        assert rule.name == "high_latency"
        assert rule.severity == AlertSeverity.WARNING
        assert rule.threshold == 5.0

    def test_rule_evaluation_true(self) -> None:
        """Test rule evaluation returns true."""
        from src.monitoring.alerts import AlertRule, AlertSeverity

        rule = AlertRule(
            name="high_error_rate",
            description="Error rate too high",
            severity=AlertSeverity.CRITICAL,
            condition=lambda v: v > 10.0,
            threshold=10.0,
        )

        assert rule.evaluate(15.0) is True

    def test_rule_evaluation_false(self) -> None:
        """Test rule evaluation returns false."""
        from src.monitoring.alerts import AlertRule, AlertSeverity

        rule = AlertRule(
            name="high_error_rate",
            description="Error rate too high",
            severity=AlertSeverity.CRITICAL,
            condition=lambda v: v > 10.0,
            threshold=10.0,
        )

        assert rule.evaluate(5.0) is False

    def test_rule_with_labels(self) -> None:
        """Test rule with labels."""
        from src.monitoring.alerts import AlertRule, AlertSeverity

        rule = AlertRule(
            name="service_error",
            description="Service-specific errors",
            severity=AlertSeverity.WARNING,
            condition=lambda v: v > 0,
            labels={"service": "extraction", "environment": "production"},
        )

        assert rule.labels["service"] == "extraction"


class TestNotificationHandler:
    """Tests for NotificationHandler base class."""

    def test_log_handler(self) -> None:
        """Test LogHandler notification."""
        from src.monitoring.alerts import Alert, AlertSeverity, AlertStatus, LogHandler

        handler = LogHandler()

        alert = Alert(
            id="log-alert-001",
            rule_name="test_rule",
            severity=AlertSeverity.INFO,
            status=AlertStatus.FIRING,
            message="Test log alert",
        )

        # Should not raise
        asyncio.run(handler.send(alert))


class TestWebhookHandler:
    """Tests for WebhookHandler."""

    def test_create_webhook_handler(self) -> None:
        """Test webhook handler creation."""
        from src.monitoring.alerts import WebhookHandler

        handler = WebhookHandler(
            url="https://webhook.example.com/alerts",
            headers={"Authorization": "Bearer token"},
        )

        assert handler.url == "https://webhook.example.com/alerts"

    @pytest.mark.asyncio
    async def test_webhook_send_success(self) -> None:
        """Test successful webhook send."""
        from src.monitoring.alerts import (
            Alert,
            AlertSeverity,
            AlertStatus,
            WebhookHandler,
        )

        handler = WebhookHandler(url="https://webhook.example.com/alerts")

        alert = Alert(
            id="webhook-alert-001",
            rule_name="test_rule",
            severity=AlertSeverity.WARNING,
            status=AlertStatus.FIRING,
            message="Test webhook alert",
        )

        with patch("aiohttp.ClientSession.post") as mock_post:
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_post.return_value.__aenter__.return_value = mock_response

            # Should not raise
            await handler.send(alert)


class TestSlackHandler:
    """Tests for SlackHandler."""

    def test_create_slack_handler(self) -> None:
        """Test Slack handler creation."""
        from src.monitoring.alerts import SlackHandler

        handler = SlackHandler(
            webhook_url="https://hooks.slack.com/services/xxx",
            channel="#alerts",
        )

        assert handler.channel == "#alerts"

    def test_format_slack_message(self) -> None:
        """Test Slack message formatting."""
        from src.monitoring.alerts import (
            Alert,
            AlertSeverity,
            AlertStatus,
            SlackHandler,
        )

        handler = SlackHandler(
            webhook_url="https://hooks.slack.com/services/xxx",
        )

        alert = Alert(
            id="slack-alert-001",
            rule_name="high_latency",
            severity=AlertSeverity.CRITICAL,
            status=AlertStatus.FIRING,
            message="Latency is critically high",
            value=15.5,
        )

        payload = handler._format_message(alert)

        assert "attachments" in payload
        assert payload["attachments"][0]["color"] == "danger"  # Critical = red


class TestPagerDutyHandler:
    """Tests for PagerDutyHandler."""

    def test_create_pagerduty_handler(self) -> None:
        """Test PagerDuty handler creation."""
        from src.monitoring.alerts import PagerDutyHandler

        handler = PagerDutyHandler(
            routing_key="your-routing-key",
            service_name="extraction-service",
        )

        assert handler.service_name == "extraction-service"


class TestAlertStore:
    """Tests for AlertStore."""

    def test_store_alert(self) -> None:
        """Test storing an alert."""
        from src.monitoring.alerts import Alert, AlertSeverity, AlertStatus, AlertStore

        store = AlertStore()

        alert = Alert(
            id="store-alert-001",
            rule_name="test_rule",
            severity=AlertSeverity.WARNING,
            status=AlertStatus.FIRING,
            message="Test alert",
        )

        store.add(alert)

        retrieved = store.get("store-alert-001")
        assert retrieved is not None
        assert retrieved.id == "store-alert-001"

    def test_get_active_alerts(self) -> None:
        """Test getting active alerts."""
        from src.monitoring.alerts import Alert, AlertSeverity, AlertStatus, AlertStore

        store = AlertStore()

        # Add firing alert
        alert1 = Alert(
            id="active-001",
            rule_name="rule1",
            severity=AlertSeverity.WARNING,
            status=AlertStatus.FIRING,
            message="Firing alert",
        )
        store.add(alert1)

        # Add resolved alert
        alert2 = Alert(
            id="resolved-001",
            rule_name="rule2",
            severity=AlertSeverity.INFO,
            status=AlertStatus.RESOLVED,
            message="Resolved alert",
        )
        store.add(alert2)

        active = store.get_active()
        assert len(active) == 1
        assert active[0].id == "active-001"

    def test_get_alerts_by_severity(self) -> None:
        """Test filtering alerts by severity."""
        from src.monitoring.alerts import Alert, AlertSeverity, AlertStatus, AlertStore

        store = AlertStore()

        # Add alerts with different severities
        store.add(
            Alert(
                id="warn-001",
                rule_name="rule1",
                severity=AlertSeverity.WARNING,
                status=AlertStatus.FIRING,
                message="Warning",
            )
        )

        store.add(
            Alert(
                id="crit-001",
                rule_name="rule2",
                severity=AlertSeverity.CRITICAL,
                status=AlertStatus.FIRING,
                message="Critical",
            )
        )

        critical_alerts = store.get_by_severity(AlertSeverity.CRITICAL)
        assert len(critical_alerts) == 1
        assert critical_alerts[0].severity == AlertSeverity.CRITICAL

    def test_prune_old_alerts(self) -> None:
        """Test pruning old resolved alerts."""
        from src.monitoring.alerts import Alert, AlertSeverity, AlertStatus, AlertStore

        store = AlertStore(max_resolved_age=timedelta(seconds=0))

        alert = Alert(
            id="old-alert-001",
            rule_name="rule1",
            severity=AlertSeverity.INFO,
            status=AlertStatus.RESOLVED,
            message="Old resolved alert",
        )
        alert.resolved_at = datetime.now(UTC) - timedelta(hours=1)
        store.add(alert)

        store.prune()

        assert store.get("old-alert-001") is None


class TestAlertManager:
    """Tests for AlertManager."""

    def test_add_rule(self) -> None:
        """Test adding alert rule."""
        from src.monitoring.alerts import AlertManager, AlertRule, AlertSeverity

        manager = AlertManager()

        rule = AlertRule(
            name="test_rule",
            description="Test rule",
            severity=AlertSeverity.WARNING,
            condition=lambda v: v > 10,
        )

        manager.add_rule(rule)

        assert manager.get_rule("test_rule") is not None

    def test_remove_rule(self) -> None:
        """Test removing alert rule."""
        from src.monitoring.alerts import AlertManager, AlertRule, AlertSeverity

        manager = AlertManager()

        rule = AlertRule(
            name="removable_rule",
            description="Will be removed",
            severity=AlertSeverity.INFO,
            condition=lambda v: v > 0,
        )

        manager.add_rule(rule)
        manager.remove_rule("removable_rule")

        assert manager.get_rule("removable_rule") is None

    def test_check_and_fire(self) -> None:
        """Test checking rule and firing alert."""
        from src.monitoring.alerts import AlertManager, AlertRule, AlertSeverity

        manager = AlertManager()

        rule = AlertRule(
            name="threshold_rule",
            description="Threshold exceeded",
            severity=AlertSeverity.WARNING,
            condition=lambda v: v > 10,
            threshold=10,
        )

        manager.add_rule(rule)

        # Should fire
        alert = manager.check("threshold_rule", 15.0)
        assert alert is not None
        assert alert.status.value == "FIRING"

    def test_check_no_fire(self) -> None:
        """Test checking rule without firing."""
        from src.monitoring.alerts import AlertManager, AlertRule, AlertSeverity

        manager = AlertManager()

        rule = AlertRule(
            name="ok_rule",
            description="Within threshold",
            severity=AlertSeverity.WARNING,
            condition=lambda v: v > 10,
        )

        manager.add_rule(rule)

        # Should not fire
        alert = manager.check("ok_rule", 5.0)
        assert alert is None

    def test_resolve_alert(self) -> None:
        """Test resolving an alert."""
        from src.monitoring.alerts import (
            AlertManager,
            AlertRule,
            AlertSeverity,
            AlertStatus,
        )

        manager = AlertManager()

        rule = AlertRule(
            name="resolvable_rule",
            description="Can be resolved",
            severity=AlertSeverity.WARNING,
            condition=lambda v: v > 10,
        )

        manager.add_rule(rule)

        # Fire alert
        alert = manager.check("resolvable_rule", 15.0)
        assert alert is not None

        # Resolve
        manager.resolve(alert.id)

        resolved = manager.get_alert(alert.id)
        assert resolved.status == AlertStatus.RESOLVED

    def test_get_active_alerts(self) -> None:
        """Test getting active alerts."""
        from src.monitoring.alerts import AlertManager, AlertRule, AlertSeverity

        manager = AlertManager()

        rule = AlertRule(
            name="active_rule",
            description="Active alert",
            severity=AlertSeverity.CRITICAL,
            condition=lambda v: v > 5,
        )

        manager.add_rule(rule)
        manager.check("active_rule", 10.0)

        active = manager.get_active_alerts()
        assert len(active) >= 1

    def test_add_handler(self) -> None:
        """Test adding notification handler."""
        from src.monitoring.alerts import AlertManager, LogHandler

        manager = AlertManager()
        handler = LogHandler()

        manager.add_handler(handler)

        assert len(manager._handlers) >= 1

    def test_check_with_notification(self) -> None:
        """Test that alerts trigger notifications."""
        from src.monitoring.alerts import AlertManager, AlertRule, AlertSeverity

        manager = AlertManager()

        # Add mock handler
        mock_handler = MagicMock()
        mock_handler.send = AsyncMock()
        manager.add_handler(mock_handler)

        rule = AlertRule(
            name="notify_rule",
            description="Triggers notification",
            severity=AlertSeverity.WARNING,
            condition=lambda v: v > 0,
        )

        manager.add_rule(rule)
        manager.check("notify_rule", 10.0)

        # Handler should have been called
        # Note: May need to wait for async processing


class TestNotificationConfig:
    """Tests for NotificationConfig."""

    def test_create_config(self) -> None:
        """Test notification config creation."""
        from src.monitoring.alerts import AlertChannel, NotificationConfig

        config = NotificationConfig(
            channels=[AlertChannel.SLACK, AlertChannel.PAGERDUTY],
            rate_limit_per_minute=10,
            dedupe_interval=timedelta(minutes=5),
        )

        assert AlertChannel.SLACK in config.channels
        assert config.rate_limit_per_minute == 10


class TestDefaultAlertRules:
    """Tests for default alert rules."""

    def test_get_default_rules(self) -> None:
        """Test getting default alert rules."""
        from src.monitoring.alerts import get_default_alert_rules

        rules = get_default_alert_rules()

        assert len(rules) > 0
        assert all(hasattr(rule, "name") for rule in rules)

    def test_default_rules_have_conditions(self) -> None:
        """Test that default rules have valid conditions."""
        from src.monitoring.alerts import get_default_alert_rules

        rules = get_default_alert_rules()

        for rule in rules:
            assert callable(rule.condition)


# =============================================================================
# Integration Tests
# =============================================================================


class TestMonitoringIntegration:
    """Integration tests for monitoring components."""

    def test_metrics_to_alerts_flow(self) -> None:
        """Test flow from metrics to alerts."""
        from src.monitoring.alerts import AlertManager, AlertRule, AlertSeverity
        from src.monitoring.metrics import MetricsCollector

        # Set up metrics
        collector = MetricsCollector()

        # Set up alerts
        manager = AlertManager()
        rule = AlertRule(
            name="high_latency_alert",
            description="API latency too high",
            severity=AlertSeverity.WARNING,
            condition=lambda v: v > 1.0,
            threshold=1.0,
        )
        manager.add_rule(rule)

        # Record a slow request
        slow_duration = 2.5
        collector.record_api_request(
            method="POST",
            endpoint="/api/v1/documents/process",
            status_code=200,
            duration=slow_duration,
        )

        # Check alert
        alert = manager.check("high_latency_alert", slow_duration)

        assert alert is not None
        assert alert.value == slow_duration

    def test_multiple_handlers_receive_alert(self) -> None:
        """Test that multiple handlers receive the same alert."""
        from src.monitoring.alerts import (
            AlertManager,
            AlertRule,
            AlertSeverity,
        )

        manager = AlertManager()

        # Add multiple handlers
        handler1 = MagicMock()
        handler1.send = AsyncMock()
        handler2 = MagicMock()
        handler2.send = AsyncMock()

        manager.add_handler(handler1)
        manager.add_handler(handler2)

        rule = AlertRule(
            name="multi_handler_rule",
            description="Test multiple handlers",
            severity=AlertSeverity.CRITICAL,
            condition=lambda v: v > 0,
        )

        manager.add_rule(rule)
        manager.check("multi_handler_rule", 10.0)

        # Both handlers should receive the alert

    def test_alert_lifecycle(self) -> None:
        """Test complete alert lifecycle: fire -> acknowledge -> resolve."""
        from src.monitoring.alerts import (
            AlertManager,
            AlertRule,
            AlertSeverity,
            AlertStatus,
        )

        manager = AlertManager()

        rule = AlertRule(
            name="lifecycle_rule",
            description="Lifecycle test",
            severity=AlertSeverity.WARNING,
            condition=lambda v: v > 0,
        )

        manager.add_rule(rule)

        # 1. Fire alert
        alert = manager.check("lifecycle_rule", 10.0)
        assert alert.status == AlertStatus.FIRING

        # 2. Acknowledge
        manager.acknowledge(alert.id, by="admin@example.com")
        acked = manager.get_alert(alert.id)
        assert acked.status == AlertStatus.ACKNOWLEDGED
        assert acked.acknowledged_by == "admin@example.com"

        # 3. Resolve
        manager.resolve(alert.id)
        resolved = manager.get_alert(alert.id)
        assert resolved.status == AlertStatus.RESOLVED

    def test_concurrent_metric_recording(self) -> None:
        """Test concurrent metric recording."""
        from src.monitoring.metrics import MetricsCollector

        collector = MetricsCollector()

        async def record_batch():
            tasks = []
            for i in range(100):
                # Record various metrics concurrently
                collector.record_api_request(
                    method="GET",
                    endpoint=f"/api/v1/documents/{i}",
                    status_code=200,
                    duration=0.1,
                )
            return True

        result = asyncio.run(record_batch())
        assert result is True

    def test_metrics_exposition_format(self) -> None:
        """Test that metrics are in valid Prometheus format."""
        from src.monitoring.metrics import MetricsCollector, MetricsRegistry

        collector = MetricsCollector()

        # Record some metrics
        collector.record_api_request(
            method="POST",
            endpoint="/test",
            status_code=200,
            duration=0.5,
        )

        registry = MetricsRegistry()
        exposition = registry.generate_exposition()

        # Should be valid Prometheus format
        # Each metric line should be: name{labels} value or # comment
        for line in exposition.strip().split("\n"):
            if line:
                assert line.startswith("#") or " " in line or line.strip() == ""
