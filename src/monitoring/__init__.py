"""
Monitoring Module for Document Extraction System.

Provides comprehensive monitoring capabilities including:
- Prometheus metrics collection and exposition
- Alerting system with multiple notification channels
- System health monitoring
- Performance tracking
"""

from src.monitoring.alerts import (
    Alert,
    AlertChannel,
    AlertManager,
    AlertRule,
    AlertSeverity,
    AlertStatus,
    AlertStore,
    LogHandler,
    NotificationConfig,
    NotificationHandler,
    PagerDutyHandler,
    SlackHandler,
    WebhookHandler,
    get_default_alert_rules,
)
from src.monitoring.metrics import (
    CONFIDENCE_BUCKETS,
    DURATION_BUCKETS,
    PAGE_BUCKETS,
    SIZE_BUCKETS,
    MetricLabels,
    MetricNamespace,
    MetricsCollector,
    MetricsRegistry,
    count_calls,
    track_duration,
)

__all__ = [
    # Metrics
    "CONFIDENCE_BUCKETS",
    "DURATION_BUCKETS",
    "PAGE_BUCKETS",
    "SIZE_BUCKETS",
    "MetricLabels",
    "MetricNamespace",
    "MetricsCollector",
    "MetricsRegistry",
    "count_calls",
    "track_duration",
    # Alerts
    "Alert",
    "AlertChannel",
    "AlertManager",
    "AlertRule",
    "AlertSeverity",
    "AlertStatus",
    "AlertStore",
    "LogHandler",
    "NotificationConfig",
    "NotificationHandler",
    "PagerDutyHandler",
    "SlackHandler",
    "WebhookHandler",
    "get_default_alert_rules",
]
