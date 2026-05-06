"""
Webhook management REST API routes.

Provides endpoints for CRUD operations on webhook subscriptions,
viewing delivery logs, and triggering test deliveries.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field, HttpUrl

from src.queue.webhook_dlq import WebhookDLQ
from src.queue.webhook_store import WebhookStore

router = APIRouter(prefix="/webhooks", tags=["webhooks"])

# Module-level store instance (can be replaced at startup)
_store: WebhookStore | None = None
_dlq: WebhookDLQ | None = None


def get_store() -> WebhookStore:
    """Get or create the global WebhookStore."""
    global _store
    if _store is None:
        _store = WebhookStore()
    return _store


def set_store(store: WebhookStore) -> None:
    """Set the global WebhookStore (for testing or custom configuration)."""
    global _store
    _store = store


def get_dlq() -> WebhookDLQ:
    """Get or create the global WebhookDLQ.

    WS-9: persistent dead-letter queue. Defaults to a SQLite database
    under ``.webhook_dlq/`` in the working directory; override at
    startup by calling ``set_dlq``.
    """
    global _dlq
    if _dlq is None:
        _dlq = WebhookDLQ()
    return _dlq


def set_dlq(dlq: WebhookDLQ) -> None:
    """Set the global WebhookDLQ (for testing or custom configuration)."""
    global _dlq
    _dlq = dlq


# ──────────────────────────────────────────────────────────────────
# Request / Response Models
# ──────────────────────────────────────────────────────────────────


class CreateWebhookRequest(BaseModel):
    """Request body for creating a webhook subscription."""

    url: str = Field(..., description="Webhook endpoint URL")
    event_types: list[str] = Field(
        default_factory=list,
        description="Event types to subscribe to (empty = all)",
    )
    description: str = Field(default="", description="Human-readable description")
    secret: str | None = Field(
        default=None,
        description="Shared secret for signing (auto-generated if omitted)",
    )
    metadata: dict[str, Any] = Field(default_factory=dict)


class UpdateWebhookRequest(BaseModel):
    """Request body for updating a webhook subscription."""

    url: str | None = None
    event_types: list[str] | None = None
    description: str | None = None
    active: bool | None = None
    metadata: dict[str, Any] | None = None


# ──────────────────────────────────────────────────────────────────
# Endpoints
# ──────────────────────────────────────────────────────────────────


@router.post("", status_code=201)
def create_webhook(body: CreateWebhookRequest) -> dict[str, Any]:
    """Create a new webhook subscription."""
    store = get_store()
    try:
        sub = store.create_subscription(
            url=body.url,
            event_types=body.event_types,
            description=body.description,
            secret=body.secret,
            metadata=body.metadata,
        )
        return {
            "status": "created",
            "subscription": sub.to_dict(),  # includes secret for initial creation
        }
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("")
def list_webhooks(
    active_only: bool = Query(False, description="Only return active subscriptions"),
) -> dict[str, Any]:
    """List all webhook subscriptions."""
    store = get_store()
    subs = store.list_subscriptions(active_only=active_only)
    return {
        "subscriptions": [s.to_public_dict() for s in subs],
        "count": len(subs),
    }


@router.get("/stats")
def webhook_stats() -> dict[str, Any]:
    """Get webhook store statistics."""
    store = get_store()
    return store.stats()


@router.get("/{subscription_id}")
def get_webhook(subscription_id: str) -> dict[str, Any]:
    """Get a specific webhook subscription."""
    store = get_store()
    sub = store.get_subscription(subscription_id)
    if sub is None:
        raise HTTPException(status_code=404, detail="Subscription not found")
    return {"subscription": sub.to_public_dict()}


@router.patch("/{subscription_id}")
def update_webhook(
    subscription_id: str,
    body: UpdateWebhookRequest,
) -> dict[str, Any]:
    """Update a webhook subscription."""
    store = get_store()
    try:
        sub = store.update_subscription(
            subscription_id=subscription_id,
            url=body.url,
            event_types=body.event_types,
            description=body.description,
            active=body.active,
            metadata=body.metadata,
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if sub is None:
        raise HTTPException(status_code=404, detail="Subscription not found")
    return {"status": "updated", "subscription": sub.to_public_dict()}


@router.delete("/{subscription_id}", status_code=200)
def delete_webhook(subscription_id: str) -> dict[str, Any]:
    """Delete a webhook subscription."""
    store = get_store()
    deleted = store.delete_subscription(subscription_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="Subscription not found")
    return {"status": "deleted", "subscription_id": subscription_id}


@router.get("/{subscription_id}/log")
def get_delivery_log(
    subscription_id: str,
    limit: int = Query(50, ge=1, le=200),
) -> dict[str, Any]:
    """Get delivery log for a specific subscription."""
    store = get_store()
    sub = store.get_subscription(subscription_id)
    if sub is None:
        raise HTTPException(status_code=404, detail="Subscription not found")
    entries = store.get_delivery_log(subscription_id=subscription_id, limit=limit)
    return {
        "subscription_id": subscription_id,
        "entries": [e.to_dict() for e in entries],
        "count": len(entries),
    }


# ──────────────────────────────────────────────────────────────────
# WS-9: Dead-Letter Queue admin
# ──────────────────────────────────────────────────────────────────


@router.get("/{subscription_id}/dlq")
def list_dlq(
    subscription_id: str,
    status: str | None = Query(
        None,
        description="Filter by entry status: 'pending' | 'delivered' | 'dead'",
    ),
    limit: int = Query(100, ge=1, le=500),
) -> dict[str, Any]:
    """List dead-lettered webhook deliveries for a subscription.

    Returns entries that exhausted the in-line retry budget. Operators
    can inspect the ``last_error`` and ``next_retry_at`` to triage,
    then call ``POST /{id}/dlq/{entry_id}/redeliver`` to force an
    immediate re-attempt.
    """
    store = get_store()
    if store.get_subscription(subscription_id) is None:
        raise HTTPException(status_code=404, detail="Subscription not found")
    dlq = get_dlq()
    entries = dlq.list_for_subscription(subscription_id, status=status, limit=limit)
    return {
        "subscription_id": subscription_id,
        "entries": [e.to_dict() for e in entries],
        "count": len(entries),
    }


@router.post("/{subscription_id}/dlq/{entry_id}/redeliver", status_code=200)
def redeliver_dlq_entry(
    subscription_id: str,
    entry_id: int,
) -> dict[str, Any]:
    """Force-mark a DLQ entry as due-now so the redeliver task picks it up.

    Operator escape hatch: bumps ``next_retry_at`` to the current
    timestamp without resetting ``attempts``, so the entry will be
    claimed on the next sweep of ``redeliver_failed_webhooks``. The
    delivery itself is still subject to the same SSRF / HMAC /
    retry-budget rules as the original send.
    """
    from datetime import UTC, datetime

    dlq = get_dlq()
    entry = dlq.get(entry_id)
    if entry is None or entry.subscription_id != subscription_id:
        raise HTTPException(status_code=404, detail="DLQ entry not found")
    if entry.status != "pending":
        raise HTTPException(
            status_code=409,
            detail=f"Entry is in '{entry.status}' state and cannot be redelivered.",
        )
    # Reschedule for immediate pickup. We don't bump attempts here —
    # only the redeliver task itself increments on actual failure.
    with dlq._connect() as conn:  # noqa: SLF001 — admin escape hatch
        conn.execute(
            "UPDATE webhook_dlq SET next_retry_at=? WHERE id=?",
            (datetime.now(UTC).isoformat(), entry_id),
        )
        conn.commit()
    return {"status": "scheduled", "entry_id": entry_id}
