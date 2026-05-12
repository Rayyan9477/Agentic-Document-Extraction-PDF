"""V3 Phase 7 — webhook DLQ poison-message detection."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from src.queue.webhook_dlq import (
    PoisonDetectionResult,
    WebhookDLQ,
    _normalise_error_signature,
)


@pytest.fixture
def dlq() -> WebhookDLQ:
    return WebhookDLQ(db_path=":memory:")


# ---------------------------------------------------------------------------
# Signature normalisation
# ---------------------------------------------------------------------------


class TestNormaliseSignature:
    def test_none_input(self) -> None:
        assert _normalise_error_signature(None) is None
        assert _normalise_error_signature("") is None

    def test_strips_numbers(self) -> None:
        a = _normalise_error_signature("HTTP 500 at 12:34")
        b = _normalise_error_signature("HTTP 502 at 56:78")
        assert a == b

    def test_strips_iso_timestamp(self) -> None:
        a = _normalise_error_signature("Failed 2026-05-09T17:00:00Z")
        b = _normalise_error_signature("Failed 2026-05-09T18:30:00Z")
        assert a == b

    def test_distinct_messages_distinct_signatures(self) -> None:
        a = _normalise_error_signature("connection refused")
        b = _normalise_error_signature("unauthorized")
        assert a != b


# ---------------------------------------------------------------------------
# Detection
# ---------------------------------------------------------------------------


def _seed_failures(
    dlq: WebhookDLQ,
    sub_id: str,
    *,
    error: str,
    count: int,
) -> None:
    """Helper: enqueue N identical-error failures for a subscription."""
    for i in range(count):
        dlq.enqueue_failed(
            subscription_id=sub_id,
            payload={"event": "test", "i": i},
            last_error=error,
        )


class TestDetectPoisonSubscription:
    def test_below_threshold_not_poisoned(self, dlq: WebhookDLQ) -> None:
        _seed_failures(dlq, "sub-1", error="unauthorized 401", count=3)
        result = dlq.detect_poison_subscription(
            "sub-1", consecutive_threshold=5
        )
        assert result.poisoned is False
        assert result.consecutive_failures == 3

    def test_consecutive_identical_errors_poisoned(
        self, dlq: WebhookDLQ
    ) -> None:
        _seed_failures(dlq, "sub-2", error="unauthorized 401", count=5)
        result = dlq.detect_poison_subscription(
            "sub-2", consecutive_threshold=5
        )
        assert result.poisoned is True
        assert result.signature is not None
        assert result.consecutive_failures == 5

    def test_diverse_errors_not_poisoned(self, dlq: WebhookDLQ) -> None:
        for i, err in enumerate(
            ["AAA error", "BBB error", "CCC error", "DDD error", "EEE error"]
        ):
            dlq.enqueue_failed(
                subscription_id="sub-3",
                payload={"i": i},
                last_error=err,
            )
        result = dlq.detect_poison_subscription(
            "sub-3", consecutive_threshold=5
        )
        assert result.poisoned is False

    def test_unrelated_subscriptions_dont_pollute(
        self, dlq: WebhookDLQ
    ) -> None:
        _seed_failures(dlq, "sub-A", error="401 unauthorized", count=5)
        _seed_failures(dlq, "sub-B", error="503 service unavailable", count=2)
        a = dlq.detect_poison_subscription("sub-A", consecutive_threshold=5)
        b = dlq.detect_poison_subscription("sub-B", consecutive_threshold=5)
        assert a.poisoned is True
        assert b.poisoned is False

    def test_to_dict_shape(self, dlq: WebhookDLQ) -> None:
        _seed_failures(dlq, "sub-D", error="unauthorized 401", count=5)
        result = dlq.detect_poison_subscription(
            "sub-D", consecutive_threshold=5
        )
        d = result.to_dict()
        assert set(d.keys()) == {
            "subscription_id",
            "poisoned",
            "consecutive_failures",
            "signature",
            "threshold",
        }
