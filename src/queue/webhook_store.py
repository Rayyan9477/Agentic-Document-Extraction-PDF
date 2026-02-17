"""
Webhook subscription store and delivery log.

Manages webhook endpoint registrations (CRUD), tracks delivery history,
and provides fan-out delivery to all matching subscriptions.
"""

from __future__ import annotations

import json
import secrets
import threading
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import structlog

from src.queue.webhook import (
    WebhookClient,
    WebhookDeliveryResult,
    WebhookDeliveryStatus,
    WebhookEventType,
    WebhookPayload,
)

logger = structlog.get_logger(__name__)


# ──────────────────────────────────────────────────────────────────
# Subscription Model
# ──────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class WebhookSubscription:
    """
    A registered webhook endpoint subscription.

    Attributes:
        subscription_id: Unique identifier.
        url: Endpoint URL to deliver webhooks to.
        event_types: List of event types to subscribe to (empty = all).
        secret: Shared secret for HMAC signature verification.
        description: Human-readable description.
        active: Whether the subscription is active.
        created_at: Creation timestamp.
        updated_at: Last update timestamp.
        metadata: Additional metadata.
    """

    subscription_id: str
    url: str
    event_types: list[str] = field(default_factory=list)
    secret: str = ""
    description: str = ""
    active: bool = True
    created_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(UTC).isoformat())
    metadata: dict[str, Any] = field(default_factory=dict)

    def matches_event(self, event_type: WebhookEventType) -> bool:
        """Check if this subscription should receive a given event type."""
        if not self.event_types:
            return True  # empty list = subscribe to all
        return event_type.value in self.event_types

    def to_dict(self) -> dict[str, Any]:
        return {
            "subscription_id": self.subscription_id,
            "url": self.url,
            "event_types": self.event_types,
            "secret": self.secret,
            "description": self.description,
            "active": self.active,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "metadata": self.metadata,
        }

    def to_public_dict(self) -> dict[str, Any]:
        """Dict without the secret (safe for API responses)."""
        d = self.to_dict()
        d["secret"] = "***" if self.secret else ""
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> WebhookSubscription:
        return cls(
            subscription_id=data["subscription_id"],
            url=data["url"],
            event_types=data.get("event_types", []),
            secret=data.get("secret", ""),
            description=data.get("description", ""),
            active=data.get("active", True),
            created_at=data.get("created_at", datetime.now(UTC).isoformat()),
            updated_at=data.get("updated_at", datetime.now(UTC).isoformat()),
            metadata=data.get("metadata", {}),
        )


# ──────────────────────────────────────────────────────────────────
# Delivery Log Entry
# ──────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class DeliveryLogEntry:
    """Record of a single webhook delivery attempt."""

    log_id: str
    subscription_id: str
    event_type: str
    processing_id: str
    status: str
    status_code: int | None = None
    attempts: int = 0
    error: str | None = None
    timestamp: str = field(default_factory=lambda: datetime.now(UTC).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return {
            "log_id": self.log_id,
            "subscription_id": self.subscription_id,
            "event_type": self.event_type,
            "processing_id": self.processing_id,
            "status": self.status,
            "status_code": self.status_code,
            "attempts": self.attempts,
            "error": self.error,
            "timestamp": self.timestamp,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> DeliveryLogEntry:
        return cls(
            log_id=data["log_id"],
            subscription_id=data["subscription_id"],
            event_type=data["event_type"],
            processing_id=data["processing_id"],
            status=data["status"],
            status_code=data.get("status_code"),
            attempts=data.get("attempts", 0),
            error=data.get("error"),
            timestamp=data.get("timestamp", datetime.now(UTC).isoformat()),
        )


# ──────────────────────────────────────────────────────────────────
# Fan-Out Result
# ──────────────────────────────────────────────────────────────────


@dataclass(slots=True)
class FanOutResult:
    """Result of delivering a webhook to all matching subscriptions."""

    event_type: str
    processing_id: str
    total_subscriptions: int = 0
    delivered: int = 0
    failed: int = 0
    skipped: int = 0
    results: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_type": self.event_type,
            "processing_id": self.processing_id,
            "total_subscriptions": self.total_subscriptions,
            "delivered": self.delivered,
            "failed": self.failed,
            "skipped": self.skipped,
            "results": self.results,
        }


# ──────────────────────────────────────────────────────────────────
# Webhook Store
# ──────────────────────────────────────────────────────────────────


class WebhookStore:
    """
    In-memory webhook subscription store with optional JSON persistence.

    Provides CRUD for subscriptions, delivery logging, and fan-out
    delivery to all matching subscriptions for a given event.

    Usage:
        store = WebhookStore()
        sub = store.create_subscription("https://example.com/hook")
        result = store.fan_out(event_type, processing_id, task_id, data)
    """

    def __init__(self, persist_path: str | Path | None = None) -> None:
        """
        Initialize the webhook store.

        Args:
            persist_path: Optional JSON file path for persistence.
                If provided, subscriptions are loaded on init and
                saved on every mutation.
        """
        self._subscriptions: dict[str, WebhookSubscription] = {}
        self._delivery_log: list[DeliveryLogEntry] = []
        self._lock = threading.Lock()
        self._persist_path = Path(persist_path) if persist_path else None
        self._max_log_entries = 1000

        if self._persist_path and self._persist_path.exists():
            self._load()

    # ── Subscription CRUD ─────────────────────────────────────────

    def create_subscription(
        self,
        url: str,
        event_types: list[str] | None = None,
        description: str = "",
        secret: str | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> WebhookSubscription:
        """
        Create and register a new webhook subscription.

        Args:
            url: Endpoint URL.
            event_types: Event types to subscribe to (empty = all).
            description: Human-readable description.
            secret: Shared secret for signing. Auto-generated if None.
            metadata: Additional metadata.

        Returns:
            The created WebhookSubscription.

        Raises:
            ValueError: If URL validation fails.
        """
        # Validate URL
        client = WebhookClient()
        is_valid, error = client.validate_url(url)
        if not is_valid:
            raise ValueError(f"Invalid webhook URL: {error}")

        sub_id = f"wh_{secrets.token_hex(8)}"
        generated_secret = secret if secret is not None else secrets.token_hex(16)

        sub = WebhookSubscription(
            subscription_id=sub_id,
            url=url,
            event_types=event_types or [],
            secret=generated_secret,
            description=description,
            metadata=metadata or {},
        )

        with self._lock:
            self._subscriptions[sub_id] = sub
            self._persist()

        logger.info(
            "webhook_subscription_created",
            subscription_id=sub_id,
            url=url,
            event_types=event_types or ["*"],
        )
        return sub

    def get_subscription(self, subscription_id: str) -> WebhookSubscription | None:
        """Get a subscription by ID."""
        return self._subscriptions.get(subscription_id)

    def list_subscriptions(self, active_only: bool = False) -> list[WebhookSubscription]:
        """List all subscriptions, optionally filtering by active status."""
        subs = list(self._subscriptions.values())
        if active_only:
            subs = [s for s in subs if s.active]
        return subs

    def update_subscription(
        self,
        subscription_id: str,
        url: str | None = None,
        event_types: list[str] | None = None,
        description: str | None = None,
        active: bool | None = None,
        metadata: dict[str, Any] | None = None,
    ) -> WebhookSubscription | None:
        """
        Update an existing subscription.

        Returns None if subscription not found.
        """
        with self._lock:
            sub = self._subscriptions.get(subscription_id)
            if sub is None:
                return None

            if url is not None:
                client = WebhookClient()
                is_valid, error = client.validate_url(url)
                if not is_valid:
                    raise ValueError(f"Invalid webhook URL: {error}")
                sub.url = url
            if event_types is not None:
                sub.event_types = event_types
            if description is not None:
                sub.description = description
            if active is not None:
                sub.active = active
            if metadata is not None:
                sub.metadata = metadata

            sub.updated_at = datetime.now(UTC).isoformat()
            self._persist()

        logger.info("webhook_subscription_updated", subscription_id=subscription_id)
        return sub

    def delete_subscription(self, subscription_id: str) -> bool:
        """Delete a subscription. Returns True if found and deleted."""
        with self._lock:
            if subscription_id in self._subscriptions:
                del self._subscriptions[subscription_id]
                self._persist()
                logger.info(
                    "webhook_subscription_deleted",
                    subscription_id=subscription_id,
                )
                return True
        return False

    # ── Delivery ──────────────────────────────────────────────────

    def fan_out(
        self,
        event_type: WebhookEventType,
        processing_id: str,
        task_id: str,
        data: dict[str, Any] | None = None,
    ) -> FanOutResult:
        """
        Deliver a webhook event to all matching active subscriptions.

        Args:
            event_type: The event type being fired.
            processing_id: Document processing ID.
            task_id: Task ID.
            data: Event-specific payload data.

        Returns:
            FanOutResult with per-subscription delivery status.
        """
        payload = WebhookPayload(
            event_type=event_type,
            processing_id=processing_id,
            task_id=task_id,
            timestamp=datetime.now(UTC).isoformat(),
            data=data or {},
        )

        active_subs = self.list_subscriptions(active_only=True)
        matching = [s for s in active_subs if s.matches_event(event_type)]

        result = FanOutResult(
            event_type=event_type.value,
            processing_id=processing_id,
            total_subscriptions=len(matching),
        )

        for sub in matching:
            client = WebhookClient(signing_secret=sub.secret or None)
            delivery = client.send_sync(sub.url, payload)

            log_entry = DeliveryLogEntry(
                log_id=f"dl_{secrets.token_hex(6)}",
                subscription_id=sub.subscription_id,
                event_type=event_type.value,
                processing_id=processing_id,
                status=delivery.status.value,
                status_code=delivery.status_code,
                attempts=delivery.attempts,
                error=delivery.error,
            )
            self._add_log_entry(log_entry)

            if delivery.status == WebhookDeliveryStatus.DELIVERED:
                result.delivered += 1
            else:
                result.failed += 1

            result.results.append({
                "subscription_id": sub.subscription_id,
                "url": sub.url,
                "status": delivery.status.value,
                "status_code": delivery.status_code,
                "attempts": delivery.attempts,
                "error": delivery.error,
            })

        logger.info(
            "webhook_fan_out_complete",
            event_type=event_type.value,
            processing_id=processing_id,
            delivered=result.delivered,
            failed=result.failed,
        )

        return result

    # ── Delivery Log ──────────────────────────────────────────────

    def get_delivery_log(
        self,
        subscription_id: str | None = None,
        limit: int = 50,
    ) -> list[DeliveryLogEntry]:
        """Get recent delivery log entries, optionally filtered by subscription."""
        entries = list(reversed(self._delivery_log))  # newest first
        if subscription_id:
            entries = [e for e in entries if e.subscription_id == subscription_id]
        return entries[:limit]

    def _add_log_entry(self, entry: DeliveryLogEntry) -> None:
        """Add a delivery log entry, trimming if over max."""
        with self._lock:
            self._delivery_log.append(entry)
            if len(self._delivery_log) > self._max_log_entries:
                self._delivery_log = self._delivery_log[-self._max_log_entries :]

    # ── Persistence ───────────────────────────────────────────────

    def _persist(self) -> None:
        """Save subscriptions to disk (called under lock)."""
        if not self._persist_path:
            return
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "subscriptions": [s.to_dict() for s in self._subscriptions.values()],
        }
        with open(self._persist_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    def _load(self) -> None:
        """Load subscriptions from disk."""
        if not self._persist_path or not self._persist_path.exists():
            return
        try:
            with open(self._persist_path, encoding="utf-8") as f:
                data = json.load(f)
            for s_data in data.get("subscriptions", []):
                sub = WebhookSubscription.from_dict(s_data)
                self._subscriptions[sub.subscription_id] = sub
            logger.info(
                "webhook_store_loaded",
                path=str(self._persist_path),
                count=len(self._subscriptions),
            )
        except Exception as e:
            logger.error("webhook_store_load_error", error=str(e))

    # ── Stats ─────────────────────────────────────────────────────

    def stats(self) -> dict[str, Any]:
        """Get store statistics."""
        subs = list(self._subscriptions.values())
        return {
            "total_subscriptions": len(subs),
            "active_subscriptions": sum(1 for s in subs if s.active),
            "inactive_subscriptions": sum(1 for s in subs if not s.active),
            "delivery_log_entries": len(self._delivery_log),
        }
