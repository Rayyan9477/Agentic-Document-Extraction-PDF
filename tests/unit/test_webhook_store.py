"""
Unit tests for Phase 5B: Webhook Delivery System.

Tests webhook subscription store (CRUD, fan-out, delivery log,
persistence), API routes, and request models.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from src.queue.webhook import (
    WebhookClient,
    WebhookDeliveryResult,
    WebhookDeliveryStatus,
    WebhookEventType,
    WebhookPayload,
)
from src.queue.webhook_store import (
    DeliveryLogEntry,
    FanOutResult,
    WebhookStore,
    WebhookSubscription,
)


# ──────────────────────────────────────────────────────────────────
# WebhookSubscription Tests
# ──────────────────────────────────────────────────────────────────


class TestWebhookSubscription:
    """Tests for WebhookSubscription model."""

    def test_create_subscription(self):
        sub = WebhookSubscription(
            subscription_id="wh_abc123",
            url="https://example.com/hook",
            event_types=["processing.completed"],
            secret="secret123",
            description="Test hook",
        )
        assert sub.subscription_id == "wh_abc123"
        assert sub.active is True

    def test_matches_event_specific(self):
        sub = WebhookSubscription(
            subscription_id="wh_1",
            url="https://example.com",
            event_types=["processing.completed", "processing.failed"],
        )
        assert sub.matches_event(WebhookEventType.PROCESSING_COMPLETED) is True
        assert sub.matches_event(WebhookEventType.PROCESSING_FAILED) is True
        assert sub.matches_event(WebhookEventType.PROCESSING_STARTED) is False

    def test_matches_event_wildcard(self):
        sub = WebhookSubscription(
            subscription_id="wh_1",
            url="https://example.com",
            event_types=[],  # empty = all
        )
        assert sub.matches_event(WebhookEventType.PROCESSING_COMPLETED) is True
        assert sub.matches_event(WebhookEventType.BATCH_STARTED) is True

    def test_to_dict(self):
        sub = WebhookSubscription(
            subscription_id="wh_1",
            url="https://example.com/hook",
            secret="my_secret",
        )
        d = sub.to_dict()
        assert d["subscription_id"] == "wh_1"
        assert d["secret"] == "my_secret"

    def test_to_public_dict_hides_secret(self):
        sub = WebhookSubscription(
            subscription_id="wh_1",
            url="https://example.com/hook",
            secret="my_secret",
        )
        d = sub.to_public_dict()
        assert d["secret"] == "***"

    def test_to_public_dict_no_secret(self):
        sub = WebhookSubscription(
            subscription_id="wh_1",
            url="https://example.com/hook",
            secret="",
        )
        d = sub.to_public_dict()
        assert d["secret"] == ""

    def test_from_dict(self):
        data = {
            "subscription_id": "wh_1",
            "url": "https://example.com",
            "event_types": ["processing.completed"],
            "secret": "s3cret",
            "active": False,
        }
        sub = WebhookSubscription.from_dict(data)
        assert sub.subscription_id == "wh_1"
        assert sub.active is False
        assert sub.secret == "s3cret"


# ──────────────────────────────────────────────────────────────────
# DeliveryLogEntry Tests
# ──────────────────────────────────────────────────────────────────


class TestDeliveryLogEntry:
    """Tests for DeliveryLogEntry model."""

    def test_create_entry(self):
        entry = DeliveryLogEntry(
            log_id="dl_1",
            subscription_id="wh_1",
            event_type="processing.completed",
            processing_id="p1",
            status="delivered",
            status_code=200,
            attempts=1,
        )
        assert entry.log_id == "dl_1"
        assert entry.status_code == 200

    def test_to_dict(self):
        entry = DeliveryLogEntry(
            log_id="dl_1",
            subscription_id="wh_1",
            event_type="processing.completed",
            processing_id="p1",
            status="failed",
            error="Timeout",
        )
        d = entry.to_dict()
        assert d["status"] == "failed"
        assert d["error"] == "Timeout"

    def test_from_dict(self):
        data = {
            "log_id": "dl_1",
            "subscription_id": "wh_1",
            "event_type": "processing.completed",
            "processing_id": "p1",
            "status": "delivered",
        }
        entry = DeliveryLogEntry.from_dict(data)
        assert entry.log_id == "dl_1"


# ──────────────────────────────────────────────────────────────────
# WebhookStore CRUD Tests
# ──────────────────────────────────────────────────────────────────


class TestWebhookStoreCRUD:
    """Tests for WebhookStore CRUD operations."""

    def test_create_subscription(self):
        store = WebhookStore()
        sub = store.create_subscription(
            url="https://example.com/hook",
            description="Test webhook",
        )
        assert sub.subscription_id.startswith("wh_")
        assert sub.url == "https://example.com/hook"
        assert sub.active is True
        assert len(sub.secret) > 0  # auto-generated

    def test_create_with_custom_secret(self):
        store = WebhookStore()
        sub = store.create_subscription(
            url="https://example.com/hook",
            secret="custom_secret",
        )
        assert sub.secret == "custom_secret"

    def test_create_with_event_types(self):
        store = WebhookStore()
        sub = store.create_subscription(
            url="https://example.com/hook",
            event_types=["processing.completed", "processing.failed"],
        )
        assert len(sub.event_types) == 2

    def test_create_invalid_url_raises(self):
        store = WebhookStore()
        with pytest.raises(ValueError, match="Invalid webhook URL"):
            store.create_subscription(url="ftp://bad-scheme.com")

    def test_create_blocked_host_raises(self):
        store = WebhookStore()
        with pytest.raises(ValueError, match="Invalid webhook URL"):
            store.create_subscription(url="https://localhost/hook")

    def test_get_subscription(self):
        store = WebhookStore()
        sub = store.create_subscription(url="https://example.com/hook")
        found = store.get_subscription(sub.subscription_id)
        assert found is not None
        assert found.url == "https://example.com/hook"

    def test_get_nonexistent(self):
        store = WebhookStore()
        assert store.get_subscription("nonexistent") is None

    def test_list_subscriptions(self):
        store = WebhookStore()
        store.create_subscription(url="https://a.com/hook")
        store.create_subscription(url="https://b.com/hook")
        subs = store.list_subscriptions()
        assert len(subs) == 2

    def test_list_active_only(self):
        store = WebhookStore()
        sub1 = store.create_subscription(url="https://a.com/hook")
        sub2 = store.create_subscription(url="https://b.com/hook")
        store.update_subscription(sub2.subscription_id, active=False)
        active = store.list_subscriptions(active_only=True)
        assert len(active) == 1
        assert active[0].subscription_id == sub1.subscription_id

    def test_update_subscription(self):
        store = WebhookStore()
        sub = store.create_subscription(url="https://example.com/hook")
        updated = store.update_subscription(
            sub.subscription_id,
            description="Updated hook",
            active=False,
        )
        assert updated is not None
        assert updated.description == "Updated hook"
        assert updated.active is False

    def test_update_url(self):
        store = WebhookStore()
        sub = store.create_subscription(url="https://example.com/hook")
        updated = store.update_subscription(
            sub.subscription_id,
            url="https://new-url.com/hook",
        )
        assert updated.url == "https://new-url.com/hook"

    def test_update_invalid_url_raises(self):
        store = WebhookStore()
        sub = store.create_subscription(url="https://example.com/hook")
        with pytest.raises(ValueError):
            store.update_subscription(sub.subscription_id, url="ftp://bad.com")

    def test_update_nonexistent(self):
        store = WebhookStore()
        result = store.update_subscription("nonexistent", description="test")
        assert result is None

    def test_delete_subscription(self):
        store = WebhookStore()
        sub = store.create_subscription(url="https://example.com/hook")
        assert store.delete_subscription(sub.subscription_id) is True
        assert store.get_subscription(sub.subscription_id) is None

    def test_delete_nonexistent(self):
        store = WebhookStore()
        assert store.delete_subscription("nonexistent") is False

    def test_stats(self):
        store = WebhookStore()
        store.create_subscription(url="https://a.com/hook")
        sub2 = store.create_subscription(url="https://b.com/hook")
        store.update_subscription(sub2.subscription_id, active=False)
        stats = store.stats()
        assert stats["total_subscriptions"] == 2
        assert stats["active_subscriptions"] == 1
        assert stats["inactive_subscriptions"] == 1


# ──────────────────────────────────────────────────────────────────
# Persistence Tests
# ──────────────────────────────────────────────────────────────────


class TestWebhookStorePersistence:
    """Tests for WebhookStore JSON persistence."""

    def test_persist_and_load(self, tmp_path):
        path = tmp_path / "webhooks.json"
        store1 = WebhookStore(persist_path=path)
        store1.create_subscription(
            url="https://example.com/hook",
            description="Persisted hook",
        )
        assert path.exists()

        store2 = WebhookStore(persist_path=path)
        subs = store2.list_subscriptions()
        assert len(subs) == 1
        assert subs[0].description == "Persisted hook"

    def test_persist_on_update(self, tmp_path):
        path = tmp_path / "webhooks.json"
        store = WebhookStore(persist_path=path)
        sub = store.create_subscription(url="https://example.com/hook")
        store.update_subscription(sub.subscription_id, description="Updated")

        store2 = WebhookStore(persist_path=path)
        loaded = store2.get_subscription(sub.subscription_id)
        assert loaded.description == "Updated"

    def test_persist_on_delete(self, tmp_path):
        path = tmp_path / "webhooks.json"
        store = WebhookStore(persist_path=path)
        sub = store.create_subscription(url="https://example.com/hook")
        store.delete_subscription(sub.subscription_id)

        store2 = WebhookStore(persist_path=path)
        assert store2.list_subscriptions() == []

    def test_no_persist_path(self):
        store = WebhookStore(persist_path=None)
        store.create_subscription(url="https://example.com/hook")
        # Should not raise — just no file written


# ──────────────────────────────────────────────────────────────────
# Delivery Log Tests
# ──────────────────────────────────────────────────────────────────


class TestDeliveryLog:
    """Tests for webhook delivery log."""

    def test_empty_log(self):
        store = WebhookStore()
        entries = store.get_delivery_log()
        assert entries == []

    def test_log_entries_added_on_fan_out(self):
        store = WebhookStore()
        sub = store.create_subscription(url="https://example.com/hook")

        # Mock the WebhookClient.send_sync to avoid real HTTP calls
        with patch.object(
            WebhookClient,
            "send_sync",
            return_value=WebhookDeliveryResult(
                status=WebhookDeliveryStatus.DELIVERED,
                status_code=200,
                attempts=1,
            ),
        ):
            store.fan_out(
                event_type=WebhookEventType.PROCESSING_COMPLETED,
                processing_id="p1",
                task_id="t1",
                data={"result": "ok"},
            )

        entries = store.get_delivery_log(subscription_id=sub.subscription_id)
        assert len(entries) == 1
        assert entries[0].status == "delivered"

    def test_log_limit(self):
        store = WebhookStore()
        sub = store.create_subscription(url="https://example.com/hook")

        with patch.object(
            WebhookClient,
            "send_sync",
            return_value=WebhookDeliveryResult(
                status=WebhookDeliveryStatus.DELIVERED,
                status_code=200,
                attempts=1,
            ),
        ):
            for i in range(5):
                store.fan_out(
                    event_type=WebhookEventType.PROCESSING_COMPLETED,
                    processing_id=f"p{i}",
                    task_id=f"t{i}",
                )

        entries = store.get_delivery_log(limit=3)
        assert len(entries) == 3

    def test_log_newest_first(self):
        store = WebhookStore()
        sub = store.create_subscription(url="https://example.com/hook")

        with patch.object(
            WebhookClient,
            "send_sync",
            return_value=WebhookDeliveryResult(
                status=WebhookDeliveryStatus.DELIVERED,
                status_code=200,
                attempts=1,
            ),
        ):
            store.fan_out(
                WebhookEventType.PROCESSING_STARTED, "p_first", "t1"
            )
            store.fan_out(
                WebhookEventType.PROCESSING_COMPLETED, "p_second", "t2"
            )

        entries = store.get_delivery_log()
        assert entries[0].processing_id == "p_second"  # newest first


# ──────────────────────────────────────────────────────────────────
# Fan-Out Tests
# ──────────────────────────────────────────────────────────────────


class TestFanOut:
    """Tests for fan-out delivery."""

    def test_fan_out_to_matching_only(self):
        store = WebhookStore()
        store.create_subscription(
            url="https://a.com/hook",
            event_types=["processing.completed"],
        )
        store.create_subscription(
            url="https://b.com/hook",
            event_types=["processing.failed"],
        )

        with patch.object(
            WebhookClient,
            "send_sync",
            return_value=WebhookDeliveryResult(
                status=WebhookDeliveryStatus.DELIVERED,
                status_code=200,
                attempts=1,
            ),
        ) as mock_send:
            result = store.fan_out(
                event_type=WebhookEventType.PROCESSING_COMPLETED,
                processing_id="p1",
                task_id="t1",
            )

        assert result.total_subscriptions == 1
        assert result.delivered == 1
        assert mock_send.call_count == 1

    def test_fan_out_wildcard_subscriptions(self):
        store = WebhookStore()
        store.create_subscription(url="https://a.com/hook", event_types=[])
        store.create_subscription(url="https://b.com/hook", event_types=[])

        with patch.object(
            WebhookClient,
            "send_sync",
            return_value=WebhookDeliveryResult(
                status=WebhookDeliveryStatus.DELIVERED,
                status_code=200,
                attempts=1,
            ),
        ):
            result = store.fan_out(
                event_type=WebhookEventType.BATCH_COMPLETED,
                processing_id="p1",
                task_id="t1",
            )

        assert result.total_subscriptions == 2
        assert result.delivered == 2

    def test_fan_out_skips_inactive(self):
        store = WebhookStore()
        sub = store.create_subscription(url="https://a.com/hook")
        store.update_subscription(sub.subscription_id, active=False)

        with patch.object(WebhookClient, "send_sync") as mock_send:
            result = store.fan_out(
                event_type=WebhookEventType.PROCESSING_COMPLETED,
                processing_id="p1",
                task_id="t1",
            )

        assert result.total_subscriptions == 0
        mock_send.assert_not_called()

    def test_fan_out_tracks_failures(self):
        store = WebhookStore()
        store.create_subscription(url="https://a.com/hook")

        with patch.object(
            WebhookClient,
            "send_sync",
            return_value=WebhookDeliveryResult(
                status=WebhookDeliveryStatus.FAILED,
                attempts=3,
                error="Timeout",
            ),
        ):
            result = store.fan_out(
                event_type=WebhookEventType.PROCESSING_COMPLETED,
                processing_id="p1",
                task_id="t1",
            )

        assert result.failed == 1
        assert result.delivered == 0

    def test_fan_out_result_to_dict(self):
        result = FanOutResult(
            event_type="processing.completed",
            processing_id="p1",
            total_subscriptions=2,
            delivered=1,
            failed=1,
        )
        d = result.to_dict()
        assert d["total_subscriptions"] == 2
        assert d["delivered"] == 1


# ──────────────────────────────────────────────────────────────────
# API Routes Tests
# ──────────────────────────────────────────────────────────────────


class TestWebhookAPIRoutes:
    """Tests for webhook REST API routes."""

    @pytest.fixture(autouse=True)
    def setup_app(self):
        """Set up FastAPI test client with clean store for each test."""
        from fastapi import FastAPI

        from src.api.routes.webhooks import router, set_store

        app = FastAPI()
        app.include_router(router)
        self.store = WebhookStore()
        set_store(self.store)
        self.client = TestClient(app)

    def test_create_webhook(self):
        resp = self.client.post(
            "/webhooks",
            json={"url": "https://example.com/hook", "description": "Test"},
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["status"] == "created"
        assert data["subscription"]["url"] == "https://example.com/hook"

    def test_create_webhook_invalid_url(self):
        resp = self.client.post(
            "/webhooks",
            json={"url": "ftp://invalid.com"},
        )
        assert resp.status_code == 400

    def test_list_webhooks(self):
        self.store.create_subscription(url="https://a.com/hook")
        self.store.create_subscription(url="https://b.com/hook")
        resp = self.client.get("/webhooks")
        assert resp.status_code == 200
        data = resp.json()
        assert data["count"] == 2

    def test_list_webhooks_active_only(self):
        sub = self.store.create_subscription(url="https://a.com/hook")
        self.store.create_subscription(url="https://b.com/hook")
        self.store.update_subscription(sub.subscription_id, active=False)
        resp = self.client.get("/webhooks?active_only=true")
        assert resp.json()["count"] == 1

    def test_get_webhook(self):
        sub = self.store.create_subscription(url="https://example.com/hook")
        resp = self.client.get(f"/webhooks/{sub.subscription_id}")
        assert resp.status_code == 200
        assert resp.json()["subscription"]["url"] == "https://example.com/hook"

    def test_get_webhook_not_found(self):
        resp = self.client.get("/webhooks/nonexistent")
        assert resp.status_code == 404

    def test_update_webhook(self):
        sub = self.store.create_subscription(url="https://example.com/hook")
        resp = self.client.patch(
            f"/webhooks/{sub.subscription_id}",
            json={"description": "Updated", "active": False},
        )
        assert resp.status_code == 200
        assert resp.json()["subscription"]["active"] is False

    def test_update_webhook_not_found(self):
        resp = self.client.patch(
            "/webhooks/nonexistent",
            json={"description": "test"},
        )
        assert resp.status_code == 404

    def test_delete_webhook(self):
        sub = self.store.create_subscription(url="https://example.com/hook")
        resp = self.client.delete(f"/webhooks/{sub.subscription_id}")
        assert resp.status_code == 200
        assert resp.json()["status"] == "deleted"

    def test_delete_webhook_not_found(self):
        resp = self.client.delete("/webhooks/nonexistent")
        assert resp.status_code == 404

    def test_get_delivery_log(self):
        sub = self.store.create_subscription(url="https://example.com/hook")
        resp = self.client.get(f"/webhooks/{sub.subscription_id}/log")
        assert resp.status_code == 200
        assert resp.json()["count"] == 0

    def test_get_delivery_log_not_found(self):
        resp = self.client.get("/webhooks/nonexistent/log")
        assert resp.status_code == 404

    def test_stats_endpoint(self):
        self.store.create_subscription(url="https://example.com/hook")
        resp = self.client.get("/webhooks/stats")
        assert resp.status_code == 200
        stats = resp.json()
        assert stats["total_subscriptions"] == 1

    def test_list_hides_secrets(self):
        self.store.create_subscription(url="https://example.com/hook")
        resp = self.client.get("/webhooks")
        for sub in resp.json()["subscriptions"]:
            assert sub["secret"] == "***"


# ──────────────────────────────────────────────────────────────────
# Module Exports Tests
# ──────────────────────────────────────────────────────────────────


class TestWebhookModuleExports:
    """Verify module import paths work."""

    def test_webhook_store_imports(self):
        from src.queue.webhook_store import (
            DeliveryLogEntry,
            FanOutResult,
            WebhookStore,
            WebhookSubscription,
        )
        assert WebhookStore is not None

    def test_webhook_routes_imports(self):
        from src.api.routes.webhooks import (
            CreateWebhookRequest,
            UpdateWebhookRequest,
            router,
        )
        assert router is not None
