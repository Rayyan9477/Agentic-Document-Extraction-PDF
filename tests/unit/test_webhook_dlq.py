"""WS-9: tests for the SQLite-backed webhook dead-letter queue."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from src.queue.webhook_dlq import WebhookDLQ


@pytest.fixture()
def dlq() -> WebhookDLQ:
    """Per-test in-memory DLQ — fast, isolated, no temp files."""
    return WebhookDLQ(db_path=":memory:", max_attempts=5, backoff_base_seconds=10)


@pytest.fixture()
def file_dlq(tmp_path):
    """Disk-backed DLQ for persistence-across-instance tests."""
    return WebhookDLQ(
        db_path=tmp_path / "dlq.db",
        max_attempts=5,
        backoff_base_seconds=10,
    )


class TestEnqueue:
    def test_enqueue_failed_returns_id_and_persists(self, dlq: WebhookDLQ) -> None:
        entry_id = dlq.enqueue_failed(
            subscription_id="sub-1",
            payload={"event": "extraction.completed", "doc": "abc"},
            last_error="HTTP 500 from receiver",
            attempts=3,
        )
        assert isinstance(entry_id, int) and entry_id > 0

        entry = dlq.get(entry_id)
        assert entry is not None
        assert entry.subscription_id == "sub-1"
        assert entry.payload == {"event": "extraction.completed", "doc": "abc"}
        assert entry.last_error == "HTTP 500 from receiver"
        assert entry.attempts == 3
        assert entry.status == "pending"
        assert entry.delivered_at is None
        assert entry.dead_at is None

    def test_enqueue_schedules_first_retry_after_base_backoff(
        self, dlq: WebhookDLQ
    ) -> None:
        before = datetime.now(UTC)
        entry_id = dlq.enqueue_failed(
            subscription_id="sub-1", payload={}, last_error=None, attempts=1
        )
        entry = dlq.get(entry_id)
        # backoff_base=10, attempts=1 → next_retry ~10s from now
        delta = (entry.next_retry_at - before).total_seconds()
        assert 8 <= delta <= 12

    def test_higher_attempts_use_exponential_backoff(self, dlq: WebhookDLQ) -> None:
        before = datetime.now(UTC)
        entry_id = dlq.enqueue_failed(
            subscription_id="sub-1", payload={}, last_error=None, attempts=4
        )
        entry = dlq.get(entry_id)
        # backoff_base=10, attempts=4 → 10 * 2^3 = 80s
        delta = (entry.next_retry_at - before).total_seconds()
        assert 75 <= delta <= 90


class TestClaimDue:
    def test_only_due_entries_claimed(self, dlq: WebhookDLQ) -> None:
        # Two entries: one already due, one in the future.
        dlq.enqueue_failed(
            subscription_id="sub-A", payload={}, last_error=None, attempts=1
        )
        dlq.enqueue_failed(
            subscription_id="sub-B", payload={}, last_error=None, attempts=10
        )
        # Claim with a "now" far in the future to make both due.
        future = datetime.now(UTC) + timedelta(days=365)
        claimed = dlq.claim_due(now=future, limit=10)
        assert len(claimed) == 2

        # Claim with a "now" before the first scheduled retry → none.
        past = datetime.now(UTC) - timedelta(seconds=1)
        assert dlq.claim_due(now=past, limit=10) == []

    def test_claim_orders_by_next_retry_at(self, dlq: WebhookDLQ) -> None:
        # Enqueue with various attempt counts → various next_retry_at.
        dlq.enqueue_failed(subscription_id="late", payload={}, last_error=None, attempts=4)
        dlq.enqueue_failed(subscription_id="soon", payload={}, last_error=None, attempts=1)
        dlq.enqueue_failed(subscription_id="middle", payload={}, last_error=None, attempts=2)

        future = datetime.now(UTC) + timedelta(days=365)
        claimed = dlq.claim_due(now=future, limit=10)
        order = [e.subscription_id for e in claimed]
        assert order == ["soon", "middle", "late"]

    def test_delivered_and_dead_entries_excluded(self, dlq: WebhookDLQ) -> None:
        delivered_id = dlq.enqueue_failed(
            subscription_id="sub", payload={}, last_error=None, attempts=1
        )
        dead_id = dlq.enqueue_failed(
            subscription_id="sub", payload={}, last_error="boom", attempts=1
        )
        dlq.mark_delivered(delivered_id)
        dlq.mark_dead(dead_id, last_error="exhausted")

        future = datetime.now(UTC) + timedelta(days=365)
        claimed_ids = [e.id for e in dlq.claim_due(now=future, limit=10)]
        assert delivered_id not in claimed_ids
        assert dead_id not in claimed_ids


class TestStateTransitions:
    def test_mark_delivered_sets_status_and_timestamp(self, dlq: WebhookDLQ) -> None:
        entry_id = dlq.enqueue_failed(
            subscription_id="sub", payload={}, last_error=None, attempts=1
        )
        dlq.mark_delivered(entry_id)
        entry = dlq.get(entry_id)
        assert entry.status == "delivered"
        assert entry.delivered_at is not None

    def test_mark_dead_sets_status_and_timestamp(self, dlq: WebhookDLQ) -> None:
        entry_id = dlq.enqueue_failed(
            subscription_id="sub", payload={}, last_error="x", attempts=1
        )
        dlq.mark_dead(entry_id, last_error="explicit override")
        entry = dlq.get(entry_id)
        assert entry.status == "dead"
        assert entry.dead_at is not None
        assert entry.last_error == "explicit override"

    def test_reschedule_failed_attempt_bumps_attempts(self, dlq: WebhookDLQ) -> None:
        entry_id = dlq.enqueue_failed(
            subscription_id="sub", payload={}, last_error=None, attempts=1
        )
        rescheduled = dlq.reschedule_failed_attempt(entry_id, last_error="still failing")
        assert rescheduled is True
        entry = dlq.get(entry_id)
        assert entry.attempts == 2
        assert entry.last_error == "still failing"
        assert entry.status == "pending"

    def test_reschedule_marks_dead_at_max_attempts(self, dlq: WebhookDLQ) -> None:
        # max_attempts is 5 in the fixture — start at 4 so the next bump tips it.
        entry_id = dlq.enqueue_failed(
            subscription_id="sub", payload={}, last_error=None, attempts=4
        )
        rescheduled = dlq.reschedule_failed_attempt(entry_id, last_error="final")
        assert rescheduled is False
        entry = dlq.get(entry_id)
        assert entry.status == "dead"
        assert entry.last_error == "final"


class TestListing:
    def test_list_for_subscription_orders_by_recency(self, dlq: WebhookDLQ) -> None:
        a = dlq.enqueue_failed(subscription_id="s", payload={"a": 1}, last_error=None)
        b = dlq.enqueue_failed(subscription_id="s", payload={"b": 2}, last_error=None)
        c = dlq.enqueue_failed(subscription_id="s", payload={"c": 3}, last_error=None)
        # Cross-subscription noise — must not appear.
        dlq.enqueue_failed(subscription_id="other", payload={}, last_error=None)

        listed = dlq.list_for_subscription("s")
        ids = [e.id for e in listed]
        assert ids == [c, b, a]

    def test_list_status_filter(self, dlq: WebhookDLQ) -> None:
        a = dlq.enqueue_failed(subscription_id="s", payload={}, last_error=None)
        b = dlq.enqueue_failed(subscription_id="s", payload={}, last_error=None)
        dlq.mark_delivered(a)

        pending = dlq.list_for_subscription("s", status="pending")
        delivered = dlq.list_for_subscription("s", status="delivered")
        assert [e.id for e in pending] == [b]
        assert [e.id for e in delivered] == [a]


class TestPersistence:
    def test_disk_backed_dlq_survives_new_instance(self, tmp_path) -> None:
        db = tmp_path / "dlq.db"
        first = WebhookDLQ(db_path=db, max_attempts=5)
        entry_id = first.enqueue_failed(
            subscription_id="sub-persist",
            payload={"event": "x"},
            last_error="503",
            attempts=2,
        )

        # Second instance opens the same file → state must survive.
        second = WebhookDLQ(db_path=db, max_attempts=5)
        entry = second.get(entry_id)
        assert entry is not None
        assert entry.subscription_id == "sub-persist"
        assert entry.last_error == "503"
        assert entry.payload == {"event": "x"}


class TestEntrySerialization:
    def test_to_dict_uses_iso_strings_for_timestamps(self, dlq: WebhookDLQ) -> None:
        entry_id = dlq.enqueue_failed(
            subscription_id="sub", payload={"k": "v"}, last_error="boom", attempts=1
        )
        entry = dlq.get(entry_id)
        d = entry.to_dict()
        # ISO strings, not datetime objects, so the dict is JSON-serialisable
        # straight out of the door.
        assert isinstance(d["first_failed_at"], str)
        assert isinstance(d["next_retry_at"], str)
        assert d["delivered_at"] is None
        assert d["dead_at"] is None
        assert d["status"] == "pending"
