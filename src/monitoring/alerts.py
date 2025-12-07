"""
Alerting System Module for Document Extraction System.

Provides comprehensive alerting capabilities with multiple notification
channels, alert rules, and escalation policies.
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from queue import Empty, Queue
from typing import Any, Callable
from urllib.parse import urljoin

import httpx
import structlog

logger = structlog.get_logger(__name__)


class AlertSeverity(str, Enum):
    """Alert severity levels."""

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class AlertStatus(str, Enum):
    """Alert status."""

    FIRING = "firing"
    RESOLVED = "resolved"
    ACKNOWLEDGED = "acknowledged"
    SILENCED = "silenced"


class AlertChannel(str, Enum):
    """Alert notification channels."""

    WEBHOOK = "webhook"
    EMAIL = "email"
    SLACK = "slack"
    PAGERDUTY = "pagerduty"
    LOG = "log"


@dataclass(slots=True)
class Alert:
    """Represents an alert."""

    alert_id: str
    name: str
    severity: AlertSeverity
    status: AlertStatus
    message: str
    source: str
    labels: dict[str, str] = field(default_factory=dict)
    annotations: dict[str, str] = field(default_factory=dict)
    value: float | None = None
    threshold: float | None = None
    fired_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolved_at: datetime | None = None
    acknowledged_at: datetime | None = None
    acknowledged_by: str | None = None
    fingerprint: str | None = None

    def __post_init__(self) -> None:
        """Generate fingerprint if not provided."""
        if self.fingerprint is None:
            self.fingerprint = self._generate_fingerprint()

    def _generate_fingerprint(self) -> str:
        """Generate unique fingerprint for deduplication."""
        import hashlib
        data = f"{self.name}:{self.source}:{sorted(self.labels.items())}"
        return hashlib.sha256(data.encode()).hexdigest()[:16]

    def to_dict(self) -> dict[str, Any]:
        """Convert alert to dictionary."""
        return {
            "alert_id": self.alert_id,
            "name": self.name,
            "severity": self.severity.value,
            "status": self.status.value,
            "message": self.message,
            "source": self.source,
            "labels": self.labels,
            "annotations": self.annotations,
            "value": self.value,
            "threshold": self.threshold,
            "fired_at": self.fired_at.isoformat(),
            "resolved_at": self.resolved_at.isoformat() if self.resolved_at else None,
            "acknowledged_at": self.acknowledged_at.isoformat() if self.acknowledged_at else None,
            "acknowledged_by": self.acknowledged_by,
            "fingerprint": self.fingerprint,
        }

    def resolve(self) -> None:
        """Mark alert as resolved."""
        self.status = AlertStatus.RESOLVED
        self.resolved_at = datetime.now(timezone.utc)

    def acknowledge(self, user: str) -> None:
        """Acknowledge the alert."""
        self.status = AlertStatus.ACKNOWLEDGED
        self.acknowledged_at = datetime.now(timezone.utc)
        self.acknowledged_by = user


@dataclass(slots=True)
class AlertRule:
    """Alert rule configuration."""

    name: str
    condition: str  # Expression to evaluate
    severity: AlertSeverity
    message_template: str
    labels: dict[str, str] = field(default_factory=dict)
    annotations: dict[str, str] = field(default_factory=dict)
    for_duration: timedelta = field(default_factory=lambda: timedelta(seconds=0))
    channels: list[AlertChannel] = field(default_factory=lambda: [AlertChannel.LOG])
    enabled: bool = True

    # Rate limiting
    repeat_interval: timedelta = field(default_factory=lambda: timedelta(minutes=5))
    group_wait: timedelta = field(default_factory=lambda: timedelta(seconds=30))


@dataclass(slots=True)
class NotificationConfig:
    """Configuration for notification channels."""

    channel: AlertChannel
    config: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


class NotificationHandler(ABC):
    """Base class for notification handlers."""

    @abstractmethod
    async def send(self, alert: Alert) -> bool:
        """
        Send notification for an alert.

        Args:
            alert: Alert to notify.

        Returns:
            True if notification was sent successfully.
        """


class WebhookHandler(NotificationHandler):
    """Webhook notification handler."""

    def __init__(
        self,
        url: str,
        headers: dict[str, str] | None = None,
        timeout: float = 30.0,
    ) -> None:
        """
        Initialize webhook handler.

        Args:
            url: Webhook URL.
            headers: Optional headers.
            timeout: Request timeout.
        """
        self._url = url
        self._headers = headers or {"Content-Type": "application/json"}
        self._timeout = timeout

    async def send(self, alert: Alert) -> bool:
        """Send webhook notification."""
        try:
            async with httpx.AsyncClient(timeout=self._timeout) as client:
                response = await client.post(
                    self._url,
                    json=alert.to_dict(),
                    headers=self._headers,
                )
                response.raise_for_status()
                logger.info(
                    "webhook_notification_sent",
                    alert_id=alert.alert_id,
                    url=self._url,
                    status=response.status_code,
                )
                return True
        except Exception as e:
            logger.error(
                "webhook_notification_failed",
                alert_id=alert.alert_id,
                url=self._url,
                error=str(e),
            )
            return False


class SlackHandler(NotificationHandler):
    """Slack notification handler."""

    def __init__(
        self,
        webhook_url: str,
        channel: str | None = None,
        username: str = "Alert Bot",
        icon_emoji: str = ":warning:",
    ) -> None:
        """
        Initialize Slack handler.

        Args:
            webhook_url: Slack webhook URL.
            channel: Override channel.
            username: Bot username.
            icon_emoji: Bot icon emoji.
        """
        self._webhook_url = webhook_url
        self._channel = channel
        self._username = username
        self._icon_emoji = icon_emoji

    def _format_message(self, alert: Alert) -> dict[str, Any]:
        """Format alert as Slack message."""
        color_map = {
            AlertSeverity.INFO: "#36a64f",
            AlertSeverity.WARNING: "#ffa500",
            AlertSeverity.ERROR: "#ff0000",
            AlertSeverity.CRITICAL: "#8b0000",
        }

        status_emoji = {
            AlertStatus.FIRING: ":fire:",
            AlertStatus.RESOLVED: ":white_check_mark:",
            AlertStatus.ACKNOWLEDGED: ":eyes:",
            AlertStatus.SILENCED: ":mute:",
        }

        attachments = [{
            "color": color_map.get(alert.severity, "#808080"),
            "title": f"{status_emoji.get(alert.status, '')} [{alert.severity.value.upper()}] {alert.name}",
            "text": alert.message,
            "fields": [
                {"title": "Source", "value": alert.source, "short": True},
                {"title": "Status", "value": alert.status.value, "short": True},
            ],
            "footer": f"Alert ID: {alert.alert_id}",
            "ts": int(alert.fired_at.timestamp()),
        }]

        if alert.value is not None:
            attachments[0]["fields"].append({
                "title": "Value",
                "value": f"{alert.value:.2f}" if isinstance(alert.value, float) else str(alert.value),
                "short": True,
            })

        if alert.threshold is not None:
            attachments[0]["fields"].append({
                "title": "Threshold",
                "value": f"{alert.threshold:.2f}" if isinstance(alert.threshold, float) else str(alert.threshold),
                "short": True,
            })

        message: dict[str, Any] = {
            "username": self._username,
            "icon_emoji": self._icon_emoji,
            "attachments": attachments,
        }

        if self._channel:
            message["channel"] = self._channel

        return message

    async def send(self, alert: Alert) -> bool:
        """Send Slack notification."""
        try:
            message = self._format_message(alert)

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self._webhook_url,
                    json=message,
                )
                response.raise_for_status()
                logger.info(
                    "slack_notification_sent",
                    alert_id=alert.alert_id,
                )
                return True
        except Exception as e:
            logger.error(
                "slack_notification_failed",
                alert_id=alert.alert_id,
                error=str(e),
            )
            return False


class PagerDutyHandler(NotificationHandler):
    """PagerDuty notification handler."""

    API_URL = "https://events.pagerduty.com/v2/enqueue"

    def __init__(
        self,
        routing_key: str,
        source: str = "doc-extraction-system",
    ) -> None:
        """
        Initialize PagerDuty handler.

        Args:
            routing_key: PagerDuty routing key.
            source: Event source identifier.
        """
        self._routing_key = routing_key
        self._source = source

    def _format_payload(self, alert: Alert) -> dict[str, Any]:
        """Format alert as PagerDuty event."""
        severity_map = {
            AlertSeverity.INFO: "info",
            AlertSeverity.WARNING: "warning",
            AlertSeverity.ERROR: "error",
            AlertSeverity.CRITICAL: "critical",
        }

        event_action = "trigger" if alert.status == AlertStatus.FIRING else "resolve"

        return {
            "routing_key": self._routing_key,
            "event_action": event_action,
            "dedup_key": alert.fingerprint,
            "payload": {
                "summary": f"[{alert.severity.value.upper()}] {alert.name}: {alert.message}",
                "severity": severity_map.get(alert.severity, "warning"),
                "source": self._source,
                "component": alert.source,
                "group": alert.labels.get("group", "default"),
                "class": alert.name,
                "custom_details": {
                    "alert_id": alert.alert_id,
                    "labels": alert.labels,
                    "annotations": alert.annotations,
                    "value": alert.value,
                    "threshold": alert.threshold,
                    "fired_at": alert.fired_at.isoformat(),
                },
            },
        }

    async def send(self, alert: Alert) -> bool:
        """Send PagerDuty notification."""
        try:
            payload = self._format_payload(alert)

            async with httpx.AsyncClient(timeout=30.0) as client:
                response = await client.post(
                    self.API_URL,
                    json=payload,
                    headers={"Content-Type": "application/json"},
                )
                response.raise_for_status()
                logger.info(
                    "pagerduty_notification_sent",
                    alert_id=alert.alert_id,
                    dedup_key=alert.fingerprint,
                )
                return True
        except Exception as e:
            logger.error(
                "pagerduty_notification_failed",
                alert_id=alert.alert_id,
                error=str(e),
            )
            return False


class LogHandler(NotificationHandler):
    """Log-based notification handler."""

    async def send(self, alert: Alert) -> bool:
        """Log the alert."""
        log_method = getattr(logger, alert.severity.value, logger.info)
        log_method(
            "alert_notification",
            alert_id=alert.alert_id,
            name=alert.name,
            status=alert.status.value,
            message=alert.message,
            source=alert.source,
            labels=alert.labels,
            value=alert.value,
            threshold=alert.threshold,
        )
        return True


class AlertStore:
    """
    In-memory alert store with deduplication.

    Tracks active alerts and their history.
    """

    def __init__(self, max_history: int = 10000) -> None:
        """
        Initialize alert store.

        Args:
            max_history: Maximum number of alerts to keep in history.
        """
        self._active: dict[str, Alert] = {}
        self._history: list[Alert] = []
        self._max_history = max_history
        self._lock = threading.Lock()

    def add(self, alert: Alert) -> bool:
        """
        Add or update an alert.

        Args:
            alert: Alert to add.

        Returns:
            True if alert is new, False if updated.
        """
        with self._lock:
            fingerprint = alert.fingerprint

            if fingerprint in self._active:
                # Update existing alert
                existing = self._active[fingerprint]
                existing.status = alert.status
                if alert.status == AlertStatus.RESOLVED:
                    existing.resolved_at = alert.resolved_at
                    self._move_to_history(fingerprint)
                return False
            else:
                # New alert
                self._active[fingerprint] = alert
                return True

    def resolve(self, fingerprint: str) -> Alert | None:
        """
        Resolve an alert by fingerprint.

        Args:
            fingerprint: Alert fingerprint.

        Returns:
            Resolved alert or None.
        """
        with self._lock:
            if fingerprint in self._active:
                alert = self._active[fingerprint]
                alert.resolve()
                self._move_to_history(fingerprint)
                return alert
            return None

    def acknowledge(self, fingerprint: str, user: str) -> Alert | None:
        """
        Acknowledge an alert.

        Args:
            fingerprint: Alert fingerprint.
            user: User acknowledging.

        Returns:
            Acknowledged alert or None.
        """
        with self._lock:
            if fingerprint in self._active:
                alert = self._active[fingerprint]
                alert.acknowledge(user)
                return alert
            return None

    def _move_to_history(self, fingerprint: str) -> None:
        """Move alert from active to history."""
        if fingerprint in self._active:
            alert = self._active.pop(fingerprint)
            self._history.append(alert)

            # Trim history
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]

    def get_active(self, severity: AlertSeverity | None = None) -> list[Alert]:
        """Get active alerts."""
        with self._lock:
            alerts = list(self._active.values())
            if severity:
                alerts = [a for a in alerts if a.severity == severity]
            return alerts

    def get_history(
        self,
        limit: int = 100,
        since: datetime | None = None,
    ) -> list[Alert]:
        """Get alert history."""
        with self._lock:
            alerts = self._history
            if since:
                alerts = [a for a in alerts if a.fired_at >= since]
            return alerts[-limit:]

    def get_by_fingerprint(self, fingerprint: str) -> Alert | None:
        """Get alert by fingerprint."""
        with self._lock:
            return self._active.get(fingerprint)

    def count_by_severity(self) -> dict[AlertSeverity, int]:
        """Count active alerts by severity."""
        with self._lock:
            counts: dict[AlertSeverity, int] = defaultdict(int)
            for alert in self._active.values():
                counts[alert.severity] += 1
            return dict(counts)


class AlertManager:
    """
    Central alert management system.

    Handles alert creation, routing, and notification.
    """

    _instance: AlertManager | None = None

    def __init__(
        self,
        handlers: dict[AlertChannel, NotificationHandler] | None = None,
        store: AlertStore | None = None,
    ) -> None:
        """
        Initialize alert manager.

        Args:
            handlers: Notification handlers by channel.
            store: Alert store.
        """
        self._handlers = handlers or {}
        self._store = store or AlertStore()
        self._rules: dict[str, AlertRule] = {}
        self._silences: dict[str, datetime] = {}  # fingerprint -> silence_until
        self._last_fired: dict[str, datetime] = {}  # fingerprint -> last fired
        self._notification_queue: Queue[tuple[Alert, list[AlertChannel]]] = Queue()
        self._running = False
        self._worker_thread: threading.Thread | None = None

        # Add default log handler
        if AlertChannel.LOG not in self._handlers:
            self._handlers[AlertChannel.LOG] = LogHandler()

    @classmethod
    def get_instance(cls, **kwargs: Any) -> AlertManager:
        """Get or create singleton instance."""
        if cls._instance is None:
            cls._instance = cls(**kwargs)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance."""
        if cls._instance:
            cls._instance.stop()
        cls._instance = None

    def start(self) -> None:
        """Start the alert manager."""
        if self._running:
            return

        self._running = True
        self._worker_thread = threading.Thread(
            target=self._notification_worker,
            daemon=True,
        )
        self._worker_thread.start()
        logger.info("alert_manager_started")

    def stop(self) -> None:
        """Stop the alert manager."""
        self._running = False
        if self._worker_thread:
            self._notification_queue.put((None, []))  # Sentinel
            self._worker_thread.join(timeout=5.0)
        logger.info("alert_manager_stopped")

    def register_handler(
        self,
        channel: AlertChannel,
        handler: NotificationHandler,
    ) -> None:
        """Register a notification handler."""
        self._handlers[channel] = handler
        logger.info("handler_registered", channel=channel.value)

    def add_rule(self, rule: AlertRule) -> None:
        """Add an alert rule."""
        self._rules[rule.name] = rule
        logger.info("rule_added", rule=rule.name)

    def remove_rule(self, name: str) -> None:
        """Remove an alert rule."""
        if name in self._rules:
            del self._rules[name]
            logger.info("rule_removed", rule=name)

    def get_rules(self) -> list[AlertRule]:
        """Get all alert rules."""
        return list(self._rules.values())

    def fire_alert(
        self,
        name: str,
        message: str,
        severity: AlertSeverity = AlertSeverity.WARNING,
        source: str = "system",
        labels: dict[str, str] | None = None,
        annotations: dict[str, str] | None = None,
        value: float | None = None,
        threshold: float | None = None,
        channels: list[AlertChannel] | None = None,
    ) -> Alert:
        """
        Fire an alert.

        Args:
            name: Alert name.
            message: Alert message.
            severity: Alert severity.
            source: Alert source.
            labels: Alert labels.
            annotations: Alert annotations.
            value: Current value.
            threshold: Threshold value.
            channels: Notification channels.

        Returns:
            Created alert.
        """
        import uuid

        alert = Alert(
            alert_id=str(uuid.uuid4()),
            name=name,
            severity=severity,
            status=AlertStatus.FIRING,
            message=message,
            source=source,
            labels=labels or {},
            annotations=annotations or {},
            value=value,
            threshold=threshold,
        )

        # Check if silenced
        if self._is_silenced(alert.fingerprint):
            logger.debug("alert_silenced", fingerprint=alert.fingerprint)
            alert.status = AlertStatus.SILENCED
            return alert

        # Check rate limiting
        if not self._should_fire(alert):
            logger.debug("alert_rate_limited", fingerprint=alert.fingerprint)
            return alert

        # Store alert
        is_new = self._store.add(alert)

        if is_new:
            # Queue for notification
            channels = channels or [AlertChannel.LOG]
            self._notification_queue.put((alert, channels))
            self._last_fired[alert.fingerprint] = datetime.now(timezone.utc)

        return alert

    def resolve_alert(self, fingerprint: str) -> Alert | None:
        """
        Resolve an alert.

        Args:
            fingerprint: Alert fingerprint.

        Returns:
            Resolved alert or None.
        """
        alert = self._store.resolve(fingerprint)
        if alert:
            # Send resolve notification
            channels = [AlertChannel.LOG]
            self._notification_queue.put((alert, channels))
        return alert

    def acknowledge_alert(
        self,
        fingerprint: str,
        user: str,
    ) -> Alert | None:
        """
        Acknowledge an alert.

        Args:
            fingerprint: Alert fingerprint.
            user: User acknowledging.

        Returns:
            Acknowledged alert or None.
        """
        return self._store.acknowledge(fingerprint, user)

    def silence(
        self,
        fingerprint: str,
        duration: timedelta = timedelta(hours=1),
    ) -> None:
        """
        Silence an alert.

        Args:
            fingerprint: Alert fingerprint.
            duration: Silence duration.
        """
        self._silences[fingerprint] = datetime.now(timezone.utc) + duration
        logger.info("alert_silenced", fingerprint=fingerprint, duration=duration)

    def unsilence(self, fingerprint: str) -> None:
        """Unsilence an alert."""
        if fingerprint in self._silences:
            del self._silences[fingerprint]
            logger.info("alert_unsilenced", fingerprint=fingerprint)

    def _is_silenced(self, fingerprint: str) -> bool:
        """Check if alert is silenced."""
        if fingerprint not in self._silences:
            return False

        if datetime.now(timezone.utc) >= self._silences[fingerprint]:
            del self._silences[fingerprint]
            return False

        return True

    def _should_fire(self, alert: Alert) -> bool:
        """Check if alert should fire based on rate limiting."""
        fingerprint = alert.fingerprint

        if fingerprint not in self._last_fired:
            return True

        # Get rule for this alert
        rule = self._rules.get(alert.name)
        repeat_interval = rule.repeat_interval if rule else timedelta(minutes=5)

        last = self._last_fired[fingerprint]
        return datetime.now(timezone.utc) - last >= repeat_interval

    def _notification_worker(self) -> None:
        """Background worker for sending notifications."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            while self._running:
                try:
                    item = self._notification_queue.get(timeout=1.0)
                    alert, channels = item

                    if alert is None:  # Sentinel
                        break

                    loop.run_until_complete(self._send_notifications(alert, channels))

                except Empty:
                    continue
                except Exception as e:
                    logger.error("notification_worker_error", error=str(e))
        finally:
            loop.close()

    async def _send_notifications(
        self,
        alert: Alert,
        channels: list[AlertChannel],
    ) -> None:
        """Send notifications to specified channels."""
        for channel in channels:
            handler = self._handlers.get(channel)
            if handler:
                try:
                    await handler.send(alert)
                except Exception as e:
                    logger.error(
                        "notification_send_error",
                        channel=channel.value,
                        error=str(e),
                    )

    def get_active_alerts(
        self,
        severity: AlertSeverity | None = None,
    ) -> list[Alert]:
        """Get active alerts."""
        return self._store.get_active(severity)

    def get_alert_history(
        self,
        limit: int = 100,
        since: datetime | None = None,
    ) -> list[Alert]:
        """Get alert history."""
        return self._store.get_history(limit, since)

    def get_alert_counts(self) -> dict[AlertSeverity, int]:
        """Get counts of active alerts by severity."""
        return self._store.count_by_severity()


# Pre-defined alert rules for the extraction system
def get_default_alert_rules() -> list[AlertRule]:
    """Get default alert rules for the extraction system."""
    return [
        AlertRule(
            name="high_error_rate",
            condition="error_rate > 0.05",
            severity=AlertSeverity.ERROR,
            message_template="Error rate is {value:.2%}, exceeding {threshold:.2%} threshold",
            labels={"category": "reliability"},
            channels=[AlertChannel.LOG, AlertChannel.SLACK],
        ),
        AlertRule(
            name="low_extraction_accuracy",
            condition="accuracy < 0.90",
            severity=AlertSeverity.WARNING,
            message_template="Extraction accuracy is {value:.2%}, below {threshold:.2%} threshold",
            labels={"category": "quality"},
            channels=[AlertChannel.LOG],
        ),
        AlertRule(
            name="vlm_unavailable",
            condition="vlm_available == 0",
            severity=AlertSeverity.CRITICAL,
            message_template="VLM service is unavailable",
            labels={"category": "availability"},
            channels=[AlertChannel.LOG, AlertChannel.PAGERDUTY],
        ),
        AlertRule(
            name="high_queue_depth",
            condition="queue_depth > 100",
            severity=AlertSeverity.WARNING,
            message_template="Extraction queue depth is {value}, exceeding {threshold} threshold",
            labels={"category": "performance"},
            channels=[AlertChannel.LOG],
        ),
        AlertRule(
            name="slow_extraction",
            condition="avg_extraction_time > 60",
            severity=AlertSeverity.WARNING,
            message_template="Average extraction time is {value:.1f}s, exceeding {threshold}s threshold",
            labels={"category": "performance"},
            channels=[AlertChannel.LOG],
        ),
        AlertRule(
            name="high_hallucination_rate",
            condition="hallucination_rate > 0.02",
            severity=AlertSeverity.ERROR,
            message_template="Hallucination rate is {value:.2%}, exceeding {threshold:.2%} threshold",
            labels={"category": "quality"},
            channels=[AlertChannel.LOG, AlertChannel.SLACK],
        ),
        AlertRule(
            name="security_breach_attempt",
            condition="security_event == 'breach_attempt'",
            severity=AlertSeverity.CRITICAL,
            message_template="Security breach attempt detected from {source}",
            labels={"category": "security"},
            channels=[AlertChannel.LOG, AlertChannel.PAGERDUTY],
        ),
        AlertRule(
            name="disk_space_low",
            condition="disk_free_percent < 10",
            severity=AlertSeverity.WARNING,
            message_template="Disk space is low: {value:.1f}% free",
            labels={"category": "infrastructure"},
            channels=[AlertChannel.LOG],
        ),
        AlertRule(
            name="memory_high",
            condition="memory_percent > 90",
            severity=AlertSeverity.WARNING,
            message_template="Memory usage is high: {value:.1f}%",
            labels={"category": "infrastructure"},
            channels=[AlertChannel.LOG],
        ),
        AlertRule(
            name="phi_access_anomaly",
            condition="phi_access_rate > normal_rate * 2",
            severity=AlertSeverity.WARNING,
            message_template="Unusual PHI access pattern detected",
            labels={"category": "security", "compliance": "hipaa"},
            channels=[AlertChannel.LOG, AlertChannel.SLACK],
        ),
    ]
