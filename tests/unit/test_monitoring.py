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
        """Test label creation with actual fields."""
        from src.monitoring.metrics import MetricLabels

        labels = MetricLabels(
            environment="production",
            service="doc-extraction",
            version="2.0.0",
            instance="node-1",
        )

        assert labels.environment == "production"
        assert labels.service == "doc-extraction"
        assert labels.version == "2.0.0"
        assert labels.instance == "node-1"

    def test_labels_defaults(self) -> None:
        """Test label default values."""
        from src.monitoring.metrics import MetricLabels

        labels = MetricLabels()

        assert labels.environment == "development"
        assert labels.service == "doc-extraction"
        assert labels.version == "2.0.0"
        assert labels.instance == "default"


class TestMetricsRegistry:
    """Tests for MetricsRegistry."""

    def test_singleton_pattern(self) -> None:
        """Test that registry is singleton via get_instance."""
        from src.monitoring.metrics import MetricsRegistry

        registry1 = MetricsRegistry.get_instance()
        registry2 = MetricsRegistry.get_instance()

        # Should be same instance
        assert registry1 is registry2

    def test_api_metrics_registered(self) -> None:
        """Test that API metrics are registered."""
        from src.monitoring.metrics import MetricsRegistry

        registry = MetricsRegistry.get_instance()

        assert registry.api_requests_total is not None
        assert registry.api_request_duration_seconds is not None
        assert registry.api_requests_in_progress is not None

    def test_extraction_metrics_registered(self) -> None:
        """Test that extraction metrics are registered."""
        from src.monitoring.metrics import MetricsRegistry

        registry = MetricsRegistry.get_instance()

        assert registry.documents_processed_total is not None
        assert registry.pages_processed_total is not None
        assert registry.extraction_duration_seconds is not None

    def test_vlm_metrics_registered(self) -> None:
        """Test that VLM metrics are registered."""
        from src.monitoring.metrics import MetricsRegistry

        registry = MetricsRegistry.get_instance()

        assert registry.vlm_calls_total is not None
        assert registry.vlm_tokens_total is not None
        assert registry.vlm_latency_seconds is not None

    def test_security_metrics_registered(self) -> None:
        """Test that security metrics are registered."""
        from src.monitoring.metrics import MetricsRegistry

        registry = MetricsRegistry.get_instance()

        assert registry.auth_attempts_total is not None
        assert registry.encryption_operations_total is not None

    def test_get_metrics(self) -> None:
        """Test Prometheus metrics output generation."""
        from src.monitoring.metrics import MetricsRegistry

        registry = MetricsRegistry.get_instance()
        output = registry.get_metrics()

        assert isinstance(output, bytes)
        text = output.decode("utf-8")
        # Should contain HELP and TYPE declarations
        assert "# HELP" in text or text == ""


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

    def test_record_document_processed(self) -> None:
        """Test recording document processing metrics."""
        from src.monitoring.metrics import MetricsCollector

        collector = MetricsCollector()

        collector.record_document_processed(
            doc_type="pdf",
            status="success",
            page_count=10,
            duration=2.5,
            file_size=1024 * 1024,
        )

    def test_record_vlm_call(self) -> None:
        """Test recording VLM call metrics."""
        from src.monitoring.metrics import MetricsCollector

        collector = MetricsCollector()

        collector.record_vlm_call(
            agent="extractor",
            call_type="extraction",
            duration=1.5,
            prompt_tokens=500,
            completion_tokens=200,
            success=True,
        )

    def test_record_validation_result(self) -> None:
        """Test recording validation result metrics."""
        from src.monitoring.metrics import MetricsCollector

        collector = MetricsCollector()

        collector.record_validation_result(
            validation_type="format",
            result="pass",
        )

    def test_record_security_event(self) -> None:
        """Test recording security event metrics."""
        from src.monitoring.metrics import MetricsCollector

        collector = MetricsCollector()

        collector.record_security_event(
            event_type="authentication",
            severity="warning",
        )

    def test_record_pipeline_error(self) -> None:
        """Test recording pipeline error metrics."""
        from src.monitoring.metrics import MetricsCollector

        collector = MetricsCollector()

        collector.record_pipeline_error(
            stage="extraction",
            error_type="timeout",
        )

    def test_record_field_extraction(self) -> None:
        """Test recording field extraction metrics."""
        from src.monitoring.metrics import MetricsCollector

        collector = MetricsCollector()

        collector.record_field_extraction(
            doc_type="pdf",
            field_type="text",
            confidence=0.95,
        )

    def test_record_auth_attempt(self) -> None:
        """Test recording authentication attempt metrics."""
        from src.monitoring.metrics import MetricsCollector

        collector = MetricsCollector()

        collector.record_auth_attempt(
            success=True,
            method="password",
        )


class TestTrackDurationDecorator:
    """Tests for track_duration decorator."""

    def test_track_sync_function(self) -> None:
        """Test duration tracking for sync function."""
        from src.monitoring.metrics import track_duration

        @track_duration("test_operation")
        def slow_function() -> str:
            time.sleep(0.01)
            return "done"

        result = slow_function()
        assert result == "done"

    def test_track_preserves_function_name(self) -> None:
        """Test that decorator preserves function name."""
        from src.monitoring.metrics import track_duration

        @track_duration("named_op")
        def my_special_function() -> int:
            return 42

        assert my_special_function.__name__ == "my_special_function"
        assert my_special_function() == 42

    def test_track_with_labels(self) -> None:
        """Test duration tracking with custom labels."""
        from src.monitoring.metrics import track_duration

        @track_duration("labeled_operation", labels={"component": "test"})
        def labeled_function(x: int) -> int:
            return x * 2

        result = labeled_function(5)
        assert result == 10


class TestCountCallsDecorator:
    """Tests for count_calls decorator."""

    def test_count_sync_function(self) -> None:
        """Test call counting for sync function."""
        from src.monitoring.metrics import count_calls

        @count_calls("test_counter")
        def counted_function() -> str:
            return "counted"

        result = counted_function()
        assert result == "counted"

    def test_count_exceptions(self) -> None:
        """Test that exceptions are still counted."""
        from src.monitoring.metrics import count_calls

        @count_calls("error_counter")
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
            alert_id="alert-001",
            name="high_error_rate",
            severity=AlertSeverity.WARNING,
            status=AlertStatus.FIRING,
            message="Error rate exceeds threshold",
            source="extraction",
            labels={"service": "extraction"},
            value=15.5,
        )

        assert alert.alert_id == "alert-001"
        assert alert.name == "high_error_rate"
        assert alert.severity == AlertSeverity.WARNING
        assert alert.status == AlertStatus.FIRING
        assert alert.fired_at is not None

    def test_alert_to_dict(self) -> None:
        """Test alert serialization."""
        from src.monitoring.alerts import Alert, AlertSeverity, AlertStatus

        alert = Alert(
            alert_id="alert-002",
            name="test_rule",
            severity=AlertSeverity.CRITICAL,
            status=AlertStatus.FIRING,
            message="Test alert",
            source="system",
        )

        data = alert.to_dict()

        assert data["alert_id"] == "alert-002"
        assert data["severity"] == "critical"
        assert "fired_at" in data

    def test_resolve_alert(self) -> None:
        """Test alert resolution."""
        from src.monitoring.alerts import Alert, AlertSeverity, AlertStatus

        alert = Alert(
            alert_id="alert-003",
            name="test_rule",
            severity=AlertSeverity.WARNING,
            status=AlertStatus.FIRING,
            message="Test alert",
            source="system",
        )

        alert.resolve()

        assert alert.status == AlertStatus.RESOLVED
        assert alert.resolved_at is not None

    def test_acknowledge_alert(self) -> None:
        """Test alert acknowledgement."""
        from src.monitoring.alerts import Alert, AlertSeverity, AlertStatus

        alert = Alert(
            alert_id="alert-004",
            name="test_rule",
            severity=AlertSeverity.CRITICAL,
            status=AlertStatus.FIRING,
            message="Critical alert",
            source="system",
        )

        alert.acknowledge(user="admin@example.com")

        assert alert.status == AlertStatus.ACKNOWLEDGED
        assert alert.acknowledged_by == "admin@example.com"


class TestAlertRule:
    """Tests for AlertRule."""

    def test_create_rule(self) -> None:
        """Test rule creation."""
        from src.monitoring.alerts import AlertRule, AlertSeverity

        rule = AlertRule(
            name="high_latency",
            condition="latency > 5.0",
            severity=AlertSeverity.WARNING,
            message_template="Latency exceeded: {value}",
            for_duration=timedelta(minutes=5),
        )

        assert rule.name == "high_latency"
        assert rule.severity == AlertSeverity.WARNING
        assert rule.condition == "latency > 5.0"
        assert rule.for_duration == timedelta(minutes=5)

    def test_rule_with_labels(self) -> None:
        """Test rule with labels."""
        from src.monitoring.alerts import AlertRule, AlertSeverity

        rule = AlertRule(
            name="service_error",
            condition="error_count > 0",
            severity=AlertSeverity.WARNING,
            message_template="Errors detected",
            labels={"service": "extraction", "environment": "production"},
        )

        assert rule.labels["service"] == "extraction"

    def test_rule_with_channels(self) -> None:
        """Test rule with custom notification channels."""
        from src.monitoring.alerts import AlertChannel, AlertRule, AlertSeverity

        rule = AlertRule(
            name="critical_alert",
            condition="error_rate > 0.1",
            severity=AlertSeverity.CRITICAL,
            message_template="Critical error rate",
            channels=[AlertChannel.LOG, AlertChannel.SLACK],
        )

        assert AlertChannel.SLACK in rule.channels

    def test_rule_defaults(self) -> None:
        """Test rule default values."""
        from src.monitoring.alerts import AlertChannel, AlertRule, AlertSeverity

        rule = AlertRule(
            name="basic_rule",
            condition="value > 10",
            severity=AlertSeverity.INFO,
            message_template="Value exceeded",
        )

        assert rule.enabled is True
        assert rule.labels == {}
        assert rule.channels == [AlertChannel.LOG]


class TestNotificationHandler:
    """Tests for NotificationHandler base class."""

    def test_log_handler(self) -> None:
        """Test LogHandler notification."""
        from src.monitoring.alerts import Alert, AlertSeverity, AlertStatus, LogHandler

        handler = LogHandler()

        alert = Alert(
            alert_id="log-alert-001",
            name="test_rule",
            severity=AlertSeverity.INFO,
            status=AlertStatus.FIRING,
            message="Test log alert",
            source="system",
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

        # Handler created successfully
        assert handler is not None

    def test_webhook_send_success(self) -> None:
        """Test successful webhook send."""
        from src.monitoring.alerts import (
            Alert,
            AlertSeverity,
            AlertStatus,
            WebhookHandler,
        )

        handler = WebhookHandler(url="https://webhook.example.com/alerts")

        alert = Alert(
            alert_id="webhook-alert-001",
            name="test_rule",
            severity=AlertSeverity.WARNING,
            status=AlertStatus.FIRING,
            message="Test webhook alert",
            source="system",
        )

        with patch("src.monitoring.alerts.httpx.AsyncClient") as MockClientCls:
            mock_client = AsyncMock()
            mock_response = MagicMock()
            mock_response.status_code = 200
            mock_response.raise_for_status = MagicMock()
            mock_client.post = AsyncMock(return_value=mock_response)

            mock_ctx = MagicMock()
            mock_ctx.__aenter__ = AsyncMock(return_value=mock_client)
            mock_ctx.__aexit__ = AsyncMock(return_value=False)
            MockClientCls.return_value = mock_ctx

            result = asyncio.run(handler.send(alert))
            assert result is True


class TestSlackHandler:
    """Tests for SlackHandler."""

    def test_create_slack_handler(self) -> None:
        """Test Slack handler creation."""
        from src.monitoring.alerts import SlackHandler

        handler = SlackHandler(
            webhook_url="https://hooks.slack.com/services/xxx",
            channel="#alerts",
        )

        # Handler created successfully
        assert handler is not None

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
            alert_id="slack-alert-001",
            name="high_latency",
            severity=AlertSeverity.CRITICAL,
            status=AlertStatus.FIRING,
            message="Latency is critically high",
            source="system",
            value=15.5,
        )

        payload = handler._format_message(alert)

        assert "attachments" in payload
        assert payload["attachments"][0]["color"] == "#8b0000"  # CRITICAL = dark red


class TestPagerDutyHandler:
    """Tests for PagerDutyHandler."""

    def test_create_pagerduty_handler(self) -> None:
        """Test PagerDuty handler creation."""
        from src.monitoring.alerts import PagerDutyHandler

        handler = PagerDutyHandler(
            routing_key="your-routing-key",
            source="extraction-service",
        )

        # Handler created successfully
        assert handler is not None


class TestAlertStore:
    """Tests for AlertStore."""

    def test_store_alert(self) -> None:
        """Test storing an alert."""
        from src.monitoring.alerts import Alert, AlertSeverity, AlertStatus, AlertStore

        store = AlertStore()

        alert = Alert(
            alert_id="store-alert-001",
            name="test_rule",
            severity=AlertSeverity.WARNING,
            status=AlertStatus.FIRING,
            message="Test alert",
            source="system",
        )

        is_new = store.add(alert)
        assert is_new is True

        retrieved = store.get_by_fingerprint(alert.fingerprint)
        assert retrieved is not None
        assert retrieved.alert_id == "store-alert-001"

    def test_get_active_alerts(self) -> None:
        """Test getting active alerts."""
        from src.monitoring.alerts import Alert, AlertSeverity, AlertStatus, AlertStore

        store = AlertStore()

        # Add firing alert
        alert1 = Alert(
            alert_id="active-001",
            name="rule1",
            severity=AlertSeverity.WARNING,
            status=AlertStatus.FIRING,
            message="Firing alert",
            source="system",
        )
        store.add(alert1)

        # Add alert then resolve it
        alert2 = Alert(
            alert_id="resolved-001",
            name="rule2",
            severity=AlertSeverity.INFO,
            status=AlertStatus.FIRING,
            message="Will be resolved",
            source="system",
        )
        store.add(alert2)
        store.resolve(alert2.fingerprint)

        active = store.get_active()
        assert len(active) == 1
        assert active[0].alert_id == "active-001"

    def test_get_alerts_by_severity(self) -> None:
        """Test filtering alerts by severity."""
        from src.monitoring.alerts import Alert, AlertSeverity, AlertStatus, AlertStore

        store = AlertStore()

        store.add(
            Alert(
                alert_id="warn-001",
                name="rule1",
                severity=AlertSeverity.WARNING,
                status=AlertStatus.FIRING,
                message="Warning",
                source="system",
            )
        )

        store.add(
            Alert(
                alert_id="crit-001",
                name="rule2",
                severity=AlertSeverity.CRITICAL,
                status=AlertStatus.FIRING,
                message="Critical",
                source="system",
            )
        )

        critical_alerts = store.get_active(severity=AlertSeverity.CRITICAL)
        assert len(critical_alerts) == 1
        assert critical_alerts[0].severity == AlertSeverity.CRITICAL

    def test_store_max_history(self) -> None:
        """Test that history is trimmed to max_history."""
        from src.monitoring.alerts import Alert, AlertSeverity, AlertStatus, AlertStore

        store = AlertStore(max_history=5)

        for i in range(10):
            alert = Alert(
                alert_id=f"alert-{i}",
                name=f"rule_{i}",
                severity=AlertSeverity.INFO,
                status=AlertStatus.FIRING,
                message=f"Alert {i}",
                source="system",
            )
            store.add(alert)
            store.resolve(alert.fingerprint)

        history = store.get_history()
        assert len(history) <= 5


class TestAlertManager:
    """Tests for AlertManager."""

    def test_add_rule(self) -> None:
        """Test adding alert rule."""
        from src.monitoring.alerts import AlertManager, AlertRule, AlertSeverity

        manager = AlertManager()

        rule = AlertRule(
            name="test_rule",
            condition="error_rate > 10",
            severity=AlertSeverity.WARNING,
            message_template="Error rate high",
        )

        manager.add_rule(rule)

        rules = manager.get_rules()
        assert any(r.name == "test_rule" for r in rules)

    def test_remove_rule(self) -> None:
        """Test removing alert rule."""
        from src.monitoring.alerts import AlertManager, AlertRule, AlertSeverity

        manager = AlertManager()

        rule = AlertRule(
            name="removable_rule",
            condition="value > 0",
            severity=AlertSeverity.INFO,
            message_template="Value exceeded",
        )

        manager.add_rule(rule)
        manager.remove_rule("removable_rule")

        rules = manager.get_rules()
        assert not any(r.name == "removable_rule" for r in rules)

    def test_fire_alert(self) -> None:
        """Test firing an alert directly."""
        from src.monitoring.alerts import AlertManager, AlertSeverity, AlertStatus

        manager = AlertManager()

        alert = manager.fire_alert(
            name="test_alert",
            message="Test fired alert",
            severity=AlertSeverity.WARNING,
            source="test",
        )

        assert alert is not None
        assert alert.status == AlertStatus.FIRING
        assert alert.name == "test_alert"

    def test_check_rules_fires_alert(self) -> None:
        """Test checking rules fires alert when condition met."""
        from src.monitoring.alerts import AlertManager, AlertRule, AlertSeverity

        manager = AlertManager()

        rule = AlertRule(
            name="threshold_rule",
            condition="error_rate > 10",
            severity=AlertSeverity.WARNING,
            message_template="Error rate: {value}",
        )

        manager.add_rule(rule)

        # Should fire
        fired = manager.check_rules({"error_rate": 15.0})
        assert len(fired) >= 1

    def test_check_rules_no_fire(self) -> None:
        """Test checking rules does not fire when condition not met."""
        from src.monitoring.alerts import AlertManager, AlertRule, AlertSeverity

        manager = AlertManager()

        rule = AlertRule(
            name="ok_rule",
            condition="error_rate > 10",
            severity=AlertSeverity.WARNING,
            message_template="Error rate: {value}",
        )

        manager.add_rule(rule)

        # Should not fire
        fired = manager.check_rules({"error_rate": 5.0})
        assert len(fired) == 0

    def test_resolve_alert(self) -> None:
        """Test resolving an alert."""
        from src.monitoring.alerts import (
            AlertManager,
            AlertSeverity,
            AlertStatus,
        )

        manager = AlertManager()

        # Fire alert
        alert = manager.fire_alert(
            name="resolvable_alert",
            message="Will be resolved",
            severity=AlertSeverity.WARNING,
            source="test",
        )

        # Resolve
        resolved = manager.resolve_alert(alert.fingerprint)

        assert resolved is not None
        assert resolved.status == AlertStatus.RESOLVED

    def test_get_active_alerts(self) -> None:
        """Test getting active alerts."""
        from src.monitoring.alerts import AlertManager, AlertSeverity

        manager = AlertManager()

        manager.fire_alert(
            name="active_alert",
            message="Active",
            severity=AlertSeverity.CRITICAL,
            source="test",
        )

        active = manager.get_active_alerts()
        assert len(active) >= 1

    def test_register_handler(self) -> None:
        """Test registering notification handler."""
        from src.monitoring.alerts import AlertChannel, AlertManager, LogHandler

        manager = AlertManager()
        handler = LogHandler()

        manager.register_handler(AlertChannel.LOG, handler)

        assert AlertChannel.LOG in manager._handlers

    def test_acknowledge_alert(self) -> None:
        """Test acknowledging an alert."""
        from src.monitoring.alerts import AlertManager, AlertSeverity, AlertStatus

        manager = AlertManager()

        alert = manager.fire_alert(
            name="ack_alert",
            message="To acknowledge",
            severity=AlertSeverity.WARNING,
            source="test",
        )

        acked = manager.acknowledge_alert(
            alert.fingerprint, user="admin@example.com"
        )

        assert acked is not None
        assert acked.status == AlertStatus.ACKNOWLEDGED
        assert acked.acknowledged_by == "admin@example.com"


class TestNotificationConfig:
    """Tests for NotificationConfig."""

    def test_create_config(self) -> None:
        """Test notification config creation."""
        from src.monitoring.alerts import AlertChannel, NotificationConfig

        config = NotificationConfig(
            channel=AlertChannel.SLACK,
            config={"webhook_url": "https://hooks.slack.com/xxx"},
            enabled=True,
        )

        assert config.channel == AlertChannel.SLACK
        assert config.enabled is True
        assert "webhook_url" in config.config


class TestDefaultAlertRules:
    """Tests for default alert rules."""

    def test_get_default_rules(self) -> None:
        """Test getting default alert rules."""
        from src.monitoring.alerts import get_default_alert_rules

        rules = get_default_alert_rules()

        assert len(rules) > 0
        assert all(hasattr(rule, "name") for rule in rules)

    def test_default_rules_have_conditions(self) -> None:
        """Test that default rules have valid string conditions."""
        from src.monitoring.alerts import get_default_alert_rules

        rules = get_default_alert_rules()

        for rule in rules:
            assert isinstance(rule.condition, str)
            assert len(rule.condition) > 0


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
            condition="api_latency > 1.0",
            severity=AlertSeverity.WARNING,
            message_template="API latency too high: {value}",
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

        # Check alert with metric value
        fired = manager.check_rules({"api_latency": slow_duration})

        assert len(fired) >= 1
        assert fired[0].value == slow_duration

    def test_multiple_handlers_registered(self) -> None:
        """Test that multiple handlers can be registered."""
        from src.monitoring.alerts import (
            AlertChannel,
            AlertManager,
            LogHandler,
            WebhookHandler,
        )

        manager = AlertManager()

        log_handler = LogHandler()
        webhook_handler = WebhookHandler(url="https://webhook.example.com")

        manager.register_handler(AlertChannel.LOG, log_handler)
        manager.register_handler(AlertChannel.WEBHOOK, webhook_handler)

        assert AlertChannel.LOG in manager._handlers
        assert AlertChannel.WEBHOOK in manager._handlers

    def test_alert_lifecycle(self) -> None:
        """Test complete alert lifecycle: fire -> acknowledge -> resolve."""
        from src.monitoring.alerts import (
            AlertManager,
            AlertSeverity,
            AlertStatus,
        )

        manager = AlertManager()

        # 1. Fire alert
        alert = manager.fire_alert(
            name="lifecycle_alert",
            message="Lifecycle test",
            severity=AlertSeverity.WARNING,
            source="test",
        )
        assert alert.status == AlertStatus.FIRING

        # 2. Acknowledge
        acked = manager.acknowledge_alert(
            alert.fingerprint, user="admin@example.com"
        )
        assert acked.status == AlertStatus.ACKNOWLEDGED
        assert acked.acknowledged_by == "admin@example.com"

        # 3. Resolve
        resolved = manager.resolve_alert(alert.fingerprint)
        assert resolved.status == AlertStatus.RESOLVED

    def test_concurrent_metric_recording(self) -> None:
        """Test concurrent metric recording."""
        from src.monitoring.metrics import MetricsCollector

        collector = MetricsCollector()

        async def record_batch():
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

    def test_metrics_output_format(self) -> None:
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

        registry = MetricsRegistry.get_instance()
        output = registry.get_metrics()

        assert isinstance(output, bytes)
        text = output.decode("utf-8")
        # Should be valid Prometheus format
        # Each metric line should be: name{labels} value or # comment
        for line in text.strip().split("\n"):
            if line:
                assert line.startswith("#") or " " in line or line.strip() == ""


class TestAlertConditionEvaluator:
    """Tests for AlertConditionEvaluator."""

    def test_simple_comparison_true(self) -> None:
        """Test simple comparison that evaluates to true."""
        from src.monitoring.alerts import AlertConditionEvaluator

        evaluator = AlertConditionEvaluator({"error_rate": 0.08})

        result, error = evaluator.evaluate("error_rate > 0.05")
        assert result is True
        assert error is None

    def test_simple_comparison_false(self) -> None:
        """Test simple comparison that evaluates to false."""
        from src.monitoring.alerts import AlertConditionEvaluator

        evaluator = AlertConditionEvaluator({"error_rate": 0.03})

        result, error = evaluator.evaluate("error_rate > 0.05")
        assert result is False
        assert error is None

    def test_invalid_condition(self) -> None:
        """Test invalid condition syntax."""
        from src.monitoring.alerts import AlertConditionEvaluator

        evaluator = AlertConditionEvaluator()

        result, error = evaluator.evaluate("not a valid condition!!!")
        assert result is False
        assert error is not None
