"""
Tests for the webhook management API routes in src/api/routes/webhooks.py.
"""

import pytest
from unittest.mock import MagicMock
from datetime import datetime, timezone
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes.webhooks import router, get_store, set_store


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_subscription_dict(
    webhook_id="wh_test123",
    url="https://example.com/hook",
    event_types=None,
    description="Test webhook",
    secret="sec_abc",
    active=True,
    metadata=None,
):
    """Return a dict that mirrors WebhookSubscription.to_dict() output."""
    now = datetime.now(timezone.utc).isoformat()
    return {
        "id": webhook_id,
        "url": url,
        "event_types": event_types or ["document.processed"],
        "description": description,
        "secret": secret,
        "active": active,
        "metadata": metadata or {},
        "created_at": now,
        "updated_at": now,
    }


def _make_mock_subscription(
    webhook_id="wh_test123",
    url="https://example.com/hook",
    event_types=None,
    description="Test webhook",
    secret="sec_abc",
    active=True,
    metadata=None,
):
    """Return a MagicMock that behaves like a WebhookSubscription."""
    sub = MagicMock()
    sub.id = webhook_id
    sub.url = url
    sub.event_types = event_types or ["document.processed"]
    sub.description = description
    sub.secret = secret
    sub.active = active
    sub.metadata = metadata or {}
    sub.created_at = datetime.now(timezone.utc).isoformat()
    sub.updated_at = datetime.now(timezone.utc).isoformat()

    d = _make_subscription_dict(
        webhook_id=webhook_id, url=url, event_types=sub.event_types,
        description=description, secret=secret, active=active, metadata=sub.metadata,
    )
    sub.to_dict.return_value = d
    # to_public_dict is used by list/get/update, it omits the secret
    public_d = {k: v for k, v in d.items() if k != "secret"}
    sub.to_public_dict.return_value = public_d
    return sub


app = FastAPI()
app.include_router(router)
client = TestClient(app)


# ---------------------------------------------------------------------------
# Fixture: inject a mock store before each test, restore afterwards
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def mock_store():
    """Inject a fresh MagicMock as the webhook store for every test."""
    store = MagicMock()
    # Routes are SYNC — use regular MagicMock, not AsyncMock
    store.create_subscription = MagicMock()
    store.list_subscriptions = MagicMock(return_value=[])
    store.get_subscription = MagicMock(return_value=None)
    store.update_subscription = MagicMock()
    store.delete_subscription = MagicMock(return_value=True)
    store.get_delivery_log = MagicMock(return_value=[])
    original = get_store()
    set_store(store)
    yield store
    set_store(original)


# ---------------------------------------------------------------------------
# set_store / get_store
# ---------------------------------------------------------------------------


class TestStoreAccessors:
    """Tests for module-level get_store / set_store."""

    def test_set_and_get_store_round_trip(self):
        sentinel = MagicMock()
        original = get_store()
        try:
            set_store(sentinel)
            assert get_store() is sentinel
        finally:
            set_store(original)


# ---------------------------------------------------------------------------
# POST /webhooks
# ---------------------------------------------------------------------------


class TestCreateWebhook:
    """Tests for POST /webhooks."""

    def test_create_webhook_success(self, mock_store):
        sub = _make_mock_subscription()
        mock_store.create_subscription.return_value = sub

        response = client.post(
            "/webhooks",
            json={
                "url": "https://example.com/hook",
                "event_types": ["document.processed"],
                "description": "Test webhook",
            },
        )
        assert response.status_code == 201
        data = response.json()
        assert data["status"] == "created"
        assert "subscription" in data
        mock_store.create_subscription.assert_called_once()

    def test_create_webhook_store_error(self, mock_store):
        mock_store.create_subscription.side_effect = ValueError("Invalid URL")

        response = client.post(
            "/webhooks",
            json={"url": "bad-url"},
        )
        assert response.status_code == 400


# ---------------------------------------------------------------------------
# GET /webhooks
# ---------------------------------------------------------------------------


class TestListWebhooks:
    """Tests for GET /webhooks."""

    def test_list_webhooks_returns_list(self, mock_store):
        sub = _make_mock_subscription()
        mock_store.list_subscriptions.return_value = [sub]

        response = client.get("/webhooks")
        assert response.status_code == 200
        data = response.json()
        assert "subscriptions" in data
        assert data["count"] == 1


# ---------------------------------------------------------------------------
# GET /webhooks/{subscription_id}
# ---------------------------------------------------------------------------


class TestGetWebhook:
    """Tests for GET /webhooks/{subscription_id}."""

    def test_get_webhook_found(self, mock_store):
        sub = _make_mock_subscription(webhook_id="wh_abc")
        mock_store.get_subscription.return_value = sub

        response = client.get("/webhooks/wh_abc")
        assert response.status_code == 200
        data = response.json()
        assert "subscription" in data

    def test_get_webhook_not_found(self, mock_store):
        mock_store.get_subscription.return_value = None

        response = client.get("/webhooks/wh_nonexistent")
        assert response.status_code == 404


# ---------------------------------------------------------------------------
# PATCH /webhooks/{subscription_id}  (note: route uses PATCH, not PUT)
# ---------------------------------------------------------------------------


class TestUpdateWebhook:
    """Tests for PATCH /webhooks/{subscription_id}."""

    def test_update_webhook_success(self, mock_store):
        updated = _make_mock_subscription(webhook_id="wh_upd", description="Updated")
        mock_store.update_subscription.return_value = updated

        response = client.patch(
            "/webhooks/wh_upd",
            json={"description": "Updated"},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "updated"


# ---------------------------------------------------------------------------
# DELETE /webhooks/{subscription_id}  (route returns 200, not 204)
# ---------------------------------------------------------------------------


class TestDeleteWebhook:
    """Tests for DELETE /webhooks/{subscription_id}."""

    def test_delete_webhook_success(self, mock_store):
        mock_store.delete_subscription.return_value = True

        response = client.delete("/webhooks/wh_del")
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "deleted"


# ---------------------------------------------------------------------------
# GET /webhooks/{subscription_id}/log  (route is /log, not /deliveries)
# ---------------------------------------------------------------------------


class TestGetDeliveryLog:
    """Tests for GET /webhooks/{subscription_id}/log."""

    def test_get_log_returns_entries(self, mock_store):
        sub = _make_mock_subscription(webhook_id="wh_abc")
        mock_store.get_subscription.return_value = sub

        entry = MagicMock()
        entry.to_dict.return_value = {"id": "dlv_1", "status": "success"}
        mock_store.get_delivery_log.return_value = [entry]

        response = client.get("/webhooks/wh_abc/log")
        assert response.status_code == 200
        data = response.json()
        assert data["count"] == 1
        assert len(data["entries"]) == 1
