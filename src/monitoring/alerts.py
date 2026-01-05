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
import operator
import re
import structlog

logger = structlog.get_logger(__name__)


# ============================================================================
# Alert Rule Evaluation Engine (CRIT-003 Fix)
# ============================================================================

class AlertConditionEvaluator:
    """
    Evaluates alert condition expressions against metric values.

    Supports expressions like:
    - "error_rate > 0.05"
    - "accuracy < 0.90"
    - "vlm_available == 0"
    - "queue_depth > 100"
    - "phi_access_rate > normal_rate * 2"
    - "security_event == 'breach_attempt'"
    """

    # Supported operators with their operator functions
    OPERATORS: dict[str, Callable[[Any, Any], bool]] = {
        ">": operator.gt,
        "<": operator.lt,
        ">=": operator.ge,
        "<=": operator.le,
        "==": operator.eq,
        "!=": operator.ne,
    }

    # Pattern to match condition expressions
    # Matches: metric_name operator value (with optional arithmetic)
    CONDITION_PATTERN = re.compile(
        r"^\s*"
        r"(?P<left>[\w.]+(?:\s*[+\-*/]\s*[\w.]+)*)"  # Left side (metric or expression)
        r"\s*(?P<operator>>=|<=|==|!=|>|<)\s*"        # Operator
        r"(?P<right>[\w.'\"]+(?:\s*[+\-*/]\s*[\w.'\"]+)*)"  # Right side (value or expression)
        r"\s*$"
    )

    # Pattern to match arithmetic expressions
    ARITHMETIC_PATTERN = re.compile(
        r"(?P<var>[\w.]+)\s*(?P<op>[+\-*/])\s*(?P<val>[\w.]+)"
    )

    def __init__(self, metrics: dict[str, Any] | None = None) -> None:
        """
        Initialize the evaluator.

        Args:
            metrics: Dictionary of metric name -> value mappings.
        """
        self._metrics = metrics or {}

    def set_metrics(self, metrics: dict[str, Any]) -> None:
        """Update the metrics dictionary."""
        self._metrics = metrics

    def update_metric(self, name: str, value: Any) -> None:
        """Update a single metric value."""
        self._metrics[name] = value

    def evaluate(self, condition: str) -> tuple[bool, str | None]:
        """
        Evaluate a condition expression.

        Args:
            condition: Condition string to evaluate (e.g., "error_rate > 0.05")

        Returns:
            Tuple of (result: bool, error: str | None)
            - result: True if condition is met, False otherwise
            - error: Error message if evaluation failed, None otherwise
        """
        if not condition or not condition.strip():
            return False, "Empty condition"

        try:
            match = self.CONDITION_PATTERN.match(condition.strip())
            if not match:
                return False, f"Invalid condition syntax: {condition}"

            left_expr = match.group("left")
            op_str = match.group("operator")
            right_expr = match.group("right")

            # Get operator function
            op_func = self.OPERATORS.get(op_str)
            if op_func is None:
                return False, f"Unsupported operator: {op_str}"

            # Evaluate left side
            left_value = self._evaluate_expression(left_expr)
            if left_value is None:
                return False, f"Cannot evaluate left expression: {left_expr}"

            # Evaluate right side
            right_value = self._evaluate_expression(right_expr)
            if right_value is None:
                return False, f"Cannot evaluate right expression: {right_expr}"

            # Perform comparison
            result = op_func(left_value, right_value)
            return bool(result), None

        except Exception as e:
            return False, f"Evaluation error: {str(e)}"

    def _evaluate_expression(self, expr: str) -> Any:
        """
        Evaluate a single expression (variable, literal, or arithmetic).

        Args:
            expr: Expression to evaluate

        Returns:
            Evaluated value or None if evaluation fails
        """
        expr = expr.strip()

        # Check for string literal (single or double quoted)
        if (expr.startswith("'") and expr.endswith("'")) or \
           (expr.startswith('"') and expr.endswith('"')):
            return expr[1:-1]

        # Check for arithmetic expression
        arith_match = self.ARITHMETIC_PATTERN.match(expr)
        if arith_match:
            var_name = arith_match.group("var")
            arith_op = arith_match.group("op")
            arith_val = arith_match.group("val")

            # Get variable value
            var_value = self._get_value(var_name)
            if var_value is None:
                return None

            # Get arithmetic operand value
            arith_operand = self._get_value(arith_val)
            if arith_operand is None:
                return None

            # Perform arithmetic
            try:
                var_value = float(var_value)
                arith_operand = float(arith_operand)

                if arith_op == "+":
                    return var_value + arith_operand
                elif arith_op == "-":
                    return var_value - arith_operand
                elif arith_op == "*":
                    return var_value * arith_operand
                elif arith_op == "/":
                    if arith_operand == 0:
                        return None
                    return var_value / arith_operand
            except (ValueError, TypeError):
                return None

        # Simple value (variable or literal)
        return self._get_value(expr)

    def _get_value(self, name: str) -> Any:
        """
        Get a value by name (from metrics) or parse as literal.

        Args:
            name: Variable name or literal value

        Returns:
            The value or None if not found/parseable
        """
        name = name.strip()

        # Check if it's in metrics
        if name in self._metrics:
            return self._metrics[name]

        # Try to parse as number
        try:
            if "." in name:
                return float(name)
            return int(name)
        except ValueError:
            pass

        # Check for boolean literals
        if name.lower() == "true":
            return True
        if name.lower() == "false":
            return False

        # Not found
        return None

    def get_threshold_from_condition(self, condition: str) -> float | None:
        """
        Extract the threshold value from a condition.

        Args:
            condition: Condition string

        Returns:
            Threshold value or None
        """
        try:
            match = self.CONDITION_PATTERN.match(condition.strip())
            if match:
                right_expr = match.group("right")
                value = self._evaluate_expression(right_expr)
                if isinstance(value, (int, float)):
                    return float(value)
        except Exception:
            pass
        return None

    def get_metric_name_from_condition(self, condition: str) -> str | None:
        """
        Extract the metric name from a condition.

        Args:
            condition: Condition string

        Returns:
            Metric name or None
        """
        try:
            match = self.CONDITION_PATTERN.match(condition.strip())
            if match:
                left_expr = match.group("left").strip()
                # Handle arithmetic expressions
                arith_match = self.ARITHMETIC_PATTERN.match(left_expr)
                if arith_match:
                    return arith_match.group("var")
                return left_expr
        except Exception:
            pass
        return None


class AlertRuleEvaluator:
    """
    Evaluates alert rules against current metrics and fires alerts.
    """

    def __init__(self, alert_manager: "AlertManager") -> None:
        """
        Initialize the rule evaluator.

        Args:
            alert_manager: AlertManager instance to fire alerts.
        """
        self._alert_manager = alert_manager
        self._condition_evaluator = AlertConditionEvaluator()
        self._pending_conditions: dict[str, datetime] = {}  # rule_name -> first_true_time

    def check_rules(self, metrics: dict[str, Any]) -> list[Alert]:
        """
        Check all rules against provided metrics and fire alerts.

        Args:
            metrics: Dictionary of metric name -> value mappings

        Returns:
            List of alerts that were fired
        """
        self._condition_evaluator.set_metrics(metrics)
        fired_alerts: list[Alert] = []

        for rule in self._alert_manager.get_rules():
            if not rule.enabled:
                continue

            try:
                alert = self._evaluate_rule(rule, metrics)
                if alert:
                    fired_alerts.append(alert)
            except Exception as e:
                logger.error(
                    "rule_evaluation_error",
                    rule=rule.name,
                    error=str(e),
                )

        return fired_alerts

    def _evaluate_rule(
        self,
        rule: AlertRule,
        metrics: dict[str, Any],
    ) -> Alert | None:
        """
        Evaluate a single rule.

        Args:
            rule: Alert rule to evaluate
            metrics: Current metrics

        Returns:
            Alert if fired, None otherwise
        """
        # Evaluate condition
        condition_met, error = self._condition_evaluator.evaluate(rule.condition)

        if error:
            logger.debug(
                "rule_condition_error",
                rule=rule.name,
                condition=rule.condition,
                error=error,
            )
            return None

        now = datetime.now(timezone.utc)

        if condition_met:
            # Check for_duration
            if rule.for_duration.total_seconds() > 0:
                if rule.name not in self._pending_conditions:
                    # Start tracking
                    self._pending_conditions[rule.name] = now
                    return None

                elapsed = now - self._pending_conditions[rule.name]
                if elapsed < rule.for_duration:
                    # Not enough time elapsed
                    return None

            # Clear pending state
            self._pending_conditions.pop(rule.name, None)

            # Get metric value and threshold for alert
            metric_name = self._condition_evaluator.get_metric_name_from_condition(
                rule.condition
            )
            current_value = metrics.get(metric_name) if metric_name else None
            threshold = self._condition_evaluator.get_threshold_from_condition(
                rule.condition
            )

            # Format message
            message = rule.message_template
            if current_value is not None:
                message = message.replace("{value}", str(current_value))
                message = message.replace("{value:.2%}", f"{current_value:.2%}")
                message = message.replace("{value:.1f}", f"{current_value:.1f}")
            if threshold is not None:
                message = message.replace("{threshold}", str(threshold))
                message = message.replace("{threshold:.2%}", f"{threshold:.2%}")

            # Add source from metrics if available
            source = metrics.get("source", "metrics")
            if "{source}" in message:
                message = message.replace("{source}", str(source))

            # Fire alert
            alert = self._alert_manager.fire_alert(
                name=rule.name,
                message=message,
                severity=rule.severity,
                source=str(source),
                labels=rule.labels,
                annotations=rule.annotations,
                value=float(current_value) if isinstance(current_value, (int, float)) else None,
                threshold=threshold,
                channels=rule.channels,
            )

            logger.info(
                "alert_rule_fired",
                rule=rule.name,
                value=current_value,
                threshold=threshold,
            )

            return alert
        else:
            # Condition not met - clear pending state and potentially resolve
            self._pending_conditions.pop(rule.name, None)

            # Check if there's an active alert for this rule that should be resolved
            active_alerts = self._alert_manager.get_active_alerts()
            for alert in active_alerts:
                if alert.name == rule.name:
                    self._alert_manager.resolve_alert(alert.fingerprint)
                    logger.info("alert_auto_resolved", rule=rule.name)

        return None


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


class EmailHandler(NotificationHandler):
    """
    Email notification handler using SMTP.

    Sends alert notifications via email with HTML formatting
    and proper severity-based styling.

    Supports:
    - SMTP with TLS/SSL encryption
    - HTML and plain text emails
    - Multiple recipients
    - Custom sender name
    - Severity-based color coding
    """

    def __init__(
        self,
        smtp_host: str,
        smtp_port: int = 587,
        smtp_user: str | None = None,
        smtp_password: str | None = None,
        use_tls: bool = True,
        use_ssl: bool = False,
        sender_email: str = "alerts@example.com",
        sender_name: str = "Document Extraction Alerts",
        recipients: list[str] | None = None,
        timeout: float = 30.0,
    ) -> None:
        """
        Initialize email handler.

        Args:
            smtp_host: SMTP server hostname.
            smtp_port: SMTP server port (587 for TLS, 465 for SSL, 25 for plain).
            smtp_user: SMTP username for authentication.
            smtp_password: SMTP password for authentication.
            use_tls: Use STARTTLS encryption (port 587).
            use_ssl: Use SSL encryption (port 465). Mutually exclusive with use_tls.
            sender_email: Sender email address.
            sender_name: Sender display name.
            recipients: List of recipient email addresses.
            timeout: Connection timeout in seconds.
        """
        self._smtp_host = smtp_host
        self._smtp_port = smtp_port
        self._smtp_user = smtp_user
        self._smtp_password = smtp_password
        self._use_tls = use_tls and not use_ssl  # TLS and SSL are mutually exclusive
        self._use_ssl = use_ssl
        self._sender_email = sender_email
        self._sender_name = sender_name
        self._recipients = recipients or []
        self._timeout = timeout

    def add_recipient(self, email: str) -> None:
        """Add a recipient email address."""
        if email not in self._recipients:
            self._recipients.append(email)

    def remove_recipient(self, email: str) -> None:
        """Remove a recipient email address."""
        if email in self._recipients:
            self._recipients.remove(email)

    def _get_severity_color(self, severity: AlertSeverity) -> str:
        """Get HTML color for severity level."""
        color_map = {
            AlertSeverity.INFO: "#17a2b8",      # Blue
            AlertSeverity.WARNING: "#ffc107",   # Yellow/Amber
            AlertSeverity.ERROR: "#dc3545",     # Red
            AlertSeverity.CRITICAL: "#721c24",  # Dark Red
        }
        return color_map.get(severity, "#6c757d")  # Gray default

    def _get_status_icon(self, status: AlertStatus) -> str:
        """Get text icon for status."""
        icon_map = {
            AlertStatus.FIRING: "🔥",
            AlertStatus.RESOLVED: "✅",
            AlertStatus.ACKNOWLEDGED: "👁",
            AlertStatus.SILENCED: "🔇",
        }
        return icon_map.get(status, "📢")

    def _format_html_message(self, alert: Alert) -> str:
        """Format alert as HTML email body."""
        severity_color = self._get_severity_color(alert.severity)
        status_icon = self._get_status_icon(alert.status)

        # Format labels as a list
        labels_html = ""
        if alert.labels:
            labels_list = "".join(
                f"<li><strong>{k}:</strong> {v}</li>"
                for k, v in alert.labels.items()
            )
            labels_html = f"""
            <tr>
                <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Labels</strong></td>
                <td style="padding: 8px; border-bottom: 1px solid #ddd;">
                    <ul style="margin: 0; padding-left: 20px;">{labels_list}</ul>
                </td>
            </tr>
            """

        # Format value and threshold if present
        value_html = ""
        if alert.value is not None:
            value_str = f"{alert.value:.2f}" if isinstance(alert.value, float) else str(alert.value)
            value_html = f"""
            <tr>
                <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Current Value</strong></td>
                <td style="padding: 8px; border-bottom: 1px solid #ddd;">{value_str}</td>
            </tr>
            """

        threshold_html = ""
        if alert.threshold is not None:
            threshold_str = f"{alert.threshold:.2f}" if isinstance(alert.threshold, float) else str(alert.threshold)
            threshold_html = f"""
            <tr>
                <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Threshold</strong></td>
                <td style="padding: 8px; border-bottom: 1px solid #ddd;">{threshold_str}</td>
            </tr>
            """

        # Resolution info if resolved
        resolution_html = ""
        if alert.status == AlertStatus.RESOLVED and alert.resolved_at:
            duration = alert.resolved_at - alert.fired_at
            resolution_html = f"""
            <tr>
                <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Resolved At</strong></td>
                <td style="padding: 8px; border-bottom: 1px solid #ddd;">{alert.resolved_at.strftime('%Y-%m-%d %H:%M:%S UTC')}</td>
            </tr>
            <tr>
                <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Duration</strong></td>
                <td style="padding: 8px; border-bottom: 1px solid #ddd;">{str(duration).split('.')[0]}</td>
            </tr>
            """

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
</head>
<body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333; max-width: 600px; margin: 0 auto; padding: 20px;">
    <div style="background-color: {severity_color}; color: white; padding: 20px; border-radius: 5px 5px 0 0;">
        <h1 style="margin: 0; font-size: 24px;">
            {status_icon} [{alert.severity.value.upper()}] {alert.name}
        </h1>
        <p style="margin: 10px 0 0 0; opacity: 0.9;">
            Status: {alert.status.value.upper()}
        </p>
    </div>

    <div style="background-color: #f8f9fa; padding: 20px; border: 1px solid #ddd; border-top: none;">
        <h2 style="margin-top: 0; color: #333; font-size: 18px;">Alert Message</h2>
        <p style="background-color: white; padding: 15px; border-radius: 5px; border-left: 4px solid {severity_color};">
            {alert.message}
        </p>

        <h2 style="color: #333; font-size: 18px;">Details</h2>
        <table style="width: 100%; border-collapse: collapse; background-color: white; border-radius: 5px;">
            <tr>
                <td style="padding: 8px; border-bottom: 1px solid #ddd; width: 30%;"><strong>Alert ID</strong></td>
                <td style="padding: 8px; border-bottom: 1px solid #ddd; font-family: monospace;">{alert.alert_id}</td>
            </tr>
            <tr>
                <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Source</strong></td>
                <td style="padding: 8px; border-bottom: 1px solid #ddd;">{alert.source}</td>
            </tr>
            <tr>
                <td style="padding: 8px; border-bottom: 1px solid #ddd;"><strong>Fired At</strong></td>
                <td style="padding: 8px; border-bottom: 1px solid #ddd;">{alert.fired_at.strftime('%Y-%m-%d %H:%M:%S UTC')}</td>
            </tr>
            {value_html}
            {threshold_html}
            {labels_html}
            {resolution_html}
            <tr>
                <td style="padding: 8px;"><strong>Fingerprint</strong></td>
                <td style="padding: 8px; font-family: monospace;">{alert.fingerprint}</td>
            </tr>
        </table>
    </div>

    <div style="background-color: #e9ecef; padding: 15px; text-align: center; font-size: 12px; color: #6c757d; border-radius: 0 0 5px 5px;">
        <p style="margin: 0;">
            This is an automated alert from the Document Extraction System.
            <br>
            Do not reply to this email.
        </p>
    </div>
</body>
</html>
"""
        return html

    def _format_text_message(self, alert: Alert) -> str:
        """Format alert as plain text email body."""
        status_icon = self._get_status_icon(alert.status)

        lines = [
            f"{status_icon} [{alert.severity.value.upper()}] {alert.name}",
            f"Status: {alert.status.value.upper()}",
            "",
            "=" * 50,
            "",
            "ALERT MESSAGE:",
            alert.message,
            "",
            "=" * 50,
            "",
            "DETAILS:",
            f"  Alert ID: {alert.alert_id}",
            f"  Source: {alert.source}",
            f"  Fired At: {alert.fired_at.strftime('%Y-%m-%d %H:%M:%S UTC')}",
        ]

        if alert.value is not None:
            value_str = f"{alert.value:.2f}" if isinstance(alert.value, float) else str(alert.value)
            lines.append(f"  Current Value: {value_str}")

        if alert.threshold is not None:
            threshold_str = f"{alert.threshold:.2f}" if isinstance(alert.threshold, float) else str(alert.threshold)
            lines.append(f"  Threshold: {threshold_str}")

        if alert.labels:
            lines.append("  Labels:")
            for key, value in alert.labels.items():
                lines.append(f"    - {key}: {value}")

        if alert.status == AlertStatus.RESOLVED and alert.resolved_at:
            duration = alert.resolved_at - alert.fired_at
            lines.append(f"  Resolved At: {alert.resolved_at.strftime('%Y-%m-%d %H:%M:%S UTC')}")
            lines.append(f"  Duration: {str(duration).split('.')[0]}")

        lines.extend([
            f"  Fingerprint: {alert.fingerprint}",
            "",
            "=" * 50,
            "",
            "This is an automated alert from the Document Extraction System.",
            "Do not reply to this email.",
        ])

        return "\n".join(lines)

    async def send(self, alert: Alert) -> bool:
        """
        Send email notification.

        Args:
            alert: Alert to send notification for.

        Returns:
            True if email was sent successfully, False otherwise.
        """
        if not self._recipients:
            logger.warning(
                "email_notification_skipped",
                alert_id=alert.alert_id,
                reason="No recipients configured",
            )
            return False

        try:
            # Import email modules here to avoid import overhead when not using email
            import smtplib
            from email.mime.multipart import MIMEMultipart
            from email.mime.text import MIMEText
            from email.utils import formataddr, formatdate

            # Create message
            msg = MIMEMultipart("alternative")
            msg["Subject"] = f"[{alert.severity.value.upper()}] {alert.name} - {alert.status.value}"
            msg["From"] = formataddr((self._sender_name, self._sender_email))
            msg["To"] = ", ".join(self._recipients)
            msg["Date"] = formatdate(localtime=True)
            msg["X-Priority"] = "1" if alert.severity == AlertSeverity.CRITICAL else "3"

            # Add custom headers for tracking
            msg["X-Alert-ID"] = alert.alert_id
            msg["X-Alert-Severity"] = alert.severity.value
            msg["X-Alert-Fingerprint"] = alert.fingerprint or ""

            # Attach plain text version
            text_body = self._format_text_message(alert)
            msg.attach(MIMEText(text_body, "plain", "utf-8"))

            # Attach HTML version
            html_body = self._format_html_message(alert)
            msg.attach(MIMEText(html_body, "html", "utf-8"))

            # Send email asynchronously using run_in_executor
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(
                None,
                self._send_smtp,
                msg,
            )

            logger.info(
                "email_notification_sent",
                alert_id=alert.alert_id,
                recipients=len(self._recipients),
                severity=alert.severity.value,
            )
            return True

        except smtplib.SMTPAuthenticationError as e:
            logger.error(
                "email_authentication_failed",
                alert_id=alert.alert_id,
                error=str(e),
            )
            return False

        except smtplib.SMTPRecipientsRefused as e:
            logger.error(
                "email_recipients_refused",
                alert_id=alert.alert_id,
                recipients=list(e.recipients.keys()),
            )
            return False

        except smtplib.SMTPException as e:
            logger.error(
                "email_smtp_error",
                alert_id=alert.alert_id,
                error=str(e),
            )
            return False

        except Exception as e:
            logger.error(
                "email_notification_failed",
                alert_id=alert.alert_id,
                error=str(e),
                error_type=type(e).__name__,
            )
            return False

    def _send_smtp(self, msg: "MIMEMultipart") -> None:
        """
        Send email via SMTP (synchronous, run in executor).

        Args:
            msg: Email message to send.
        """
        import smtplib
        import ssl

        # Create SSL context for secure connections
        context = ssl.create_default_context()

        if self._use_ssl:
            # Direct SSL connection (port 465)
            with smtplib.SMTP_SSL(
                self._smtp_host,
                self._smtp_port,
                context=context,
                timeout=self._timeout,
            ) as server:
                if self._smtp_user and self._smtp_password:
                    server.login(self._smtp_user, self._smtp_password)
                server.sendmail(
                    self._sender_email,
                    self._recipients,
                    msg.as_string(),
                )
        else:
            # Plain or STARTTLS connection
            with smtplib.SMTP(
                self._smtp_host,
                self._smtp_port,
                timeout=self._timeout,
            ) as server:
                if self._use_tls:
                    server.starttls(context=context)
                if self._smtp_user and self._smtp_password:
                    server.login(self._smtp_user, self._smtp_password)
                server.sendmail(
                    self._sender_email,
                    self._recipients,
                    msg.as_string(),
                )


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
    _lock: threading.Lock = threading.Lock()

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
        """Get or create singleton instance (thread-safe)."""
        if cls._instance is None:
            with cls._lock:
                # Double-check locking pattern
                if cls._instance is None:
                    cls._instance = cls(**kwargs)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (thread-safe)."""
        with cls._lock:
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

    def resolve_alert(
        self,
        fingerprint: str,
        channels: list[AlertChannel] | None = None,
    ) -> Alert | None:
        """
        Resolve an alert and notify all original channels.

        Args:
            fingerprint: Alert fingerprint.
            channels: Optional override channels for notification.
                      If not provided, uses channels from the associated rule.

        Returns:
            Resolved alert or None.
        """
        alert = self._store.resolve(fingerprint)
        if alert:
            # Determine notification channels
            if channels is None:
                # Look up the rule to get the original channels
                rule = self._rules.get(alert.name)
                if rule:
                    channels = rule.channels
                else:
                    # Fallback to LOG if no rule found
                    channels = [AlertChannel.LOG]

            # Ensure LOG is always included for audit trail
            if AlertChannel.LOG not in channels:
                channels = list(channels) + [AlertChannel.LOG]

            # Send resolution notification to all channels
            self._notification_queue.put((alert, channels))

            logger.info(
                "alert_resolved_notification_queued",
                fingerprint=fingerprint,
                alert_name=alert.name,
                channels=[c.value for c in channels],
            )

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

    def cleanup_stale_entries(self, max_age: timedelta | None = None) -> int:
        """
        Clean up stale entries from _silences and _last_fired dicts.

        This prevents unbounded memory growth in long-running processes.
        Removes expired silences and old last_fired entries.

        Args:
            max_age: Maximum age for last_fired entries. Defaults to 24 hours.

        Returns:
            Number of entries cleaned up.
        """
        if max_age is None:
            max_age = timedelta(hours=24)

        now = datetime.now(timezone.utc)
        cleaned = 0

        # Clean up expired silences
        expired_silences = [
            fp for fp, until in self._silences.items()
            if now >= until
        ]
        for fp in expired_silences:
            del self._silences[fp]
            cleaned += 1

        # Clean up old last_fired entries
        stale_fired = [
            fp for fp, fired_at in self._last_fired.items()
            if now - fired_at > max_age
        ]
        for fp in stale_fired:
            del self._last_fired[fp]
            cleaned += 1

        if cleaned > 0:
            logger.debug(
                "alert_manager_cleanup",
                silences_cleaned=len(expired_silences),
                last_fired_cleaned=len(stale_fired),
            )

        return cleaned

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

    def check_rules(self, metrics: dict[str, Any]) -> list[Alert]:
        """
        Check all registered alert rules against provided metrics.

        This is the main entry point for rule-based alerting. Call this method
        periodically with current system metrics to evaluate all rules and
        fire alerts when conditions are met.

        Args:
            metrics: Dictionary of metric name -> value mappings.
                     Example: {
                         "error_rate": 0.08,
                         "accuracy": 0.92,
                         "queue_depth": 150,
                         "vlm_available": 1,
                     }

        Returns:
            List of alerts that were fired.

        Example:
            >>> manager = AlertManager.get_instance()
            >>> manager.add_rule(AlertRule(
            ...     name="high_error_rate",
            ...     condition="error_rate > 0.05",
            ...     severity=AlertSeverity.ERROR,
            ...     message_template="Error rate: {value:.2%}",
            ... ))
            >>> metrics = {"error_rate": 0.08, "source": "extraction"}
            >>> alerts = manager.check_rules(metrics)
            >>> len(alerts)  # Alert fired because 0.08 > 0.05
            1
        """
        evaluator = AlertRuleEvaluator(self)
        return evaluator.check_rules(metrics)

    def load_default_rules(self) -> None:
        """
        Load the default alert rules for the extraction system.

        This registers all pre-defined rules from get_default_alert_rules().
        """
        for rule in get_default_alert_rules():
            self.add_rule(rule)
        logger.info("default_rules_loaded", count=len(self._rules))


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
