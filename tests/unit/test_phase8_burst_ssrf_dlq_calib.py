"""V3 Phase 8 — burst rate limit + SSRF + DLQ poison + calibration first-fit + reconciler."""

from __future__ import annotations

import socket
from unittest.mock import patch

import pytest

from src.api.middleware import RateLimiter
from src.queue._url_safety import check_public_url
from src.queue.webhook_dlq import WebhookDLQ
from src.validation.calibration import CalibrationPoint, PartitionedCalibrator


# ---------------------------------------------------------------------------
# 8C.B1 — Rate limiter burst + rpm=0 emergency
# ---------------------------------------------------------------------------


class TestRateLimiterBurst:
    def test_burst_allows_exactly_burst_count(self) -> None:
        limiter = RateLimiter(default_rpm=60, burst_size=3)
        limiter.set_tenant_limit("t", rpm=60, burst=3)
        allowed = sum(
            1
            for _ in range(10)
            if limiter.is_allowed("c", tenant_id="t")[0]
        )
        # First 3 burst tokens consumed; subsequent calls fail
        # because 1/(60s) of refill hasn't accrued.
        assert allowed == 3

    def test_rpm_zero_blocks_first_call(self) -> None:
        limiter = RateLimiter(default_rpm=60, burst_size=10)
        limiter.set_tenant_limit("emergency", rpm=0, burst=5)
        allowed, headers = limiter.is_allowed("c", tenant_id="emergency")
        assert allowed is False
        assert headers["X-RateLimit-Limit"] == 0
        assert headers["X-RateLimit-Remaining"] == 0

    def test_separate_tenants_have_independent_buckets(self) -> None:
        limiter = RateLimiter(default_rpm=60, burst_size=2)
        limiter.set_tenant_limit("a", rpm=60, burst=1)
        limiter.set_tenant_limit("b", rpm=60, burst=1)
        assert limiter.is_allowed("c", tenant_id="a")[0]
        assert not limiter.is_allowed("c", tenant_id="a")[0]
        # b still has its token.
        assert limiter.is_allowed("c", tenant_id="b")[0]


# ---------------------------------------------------------------------------
# 8A.C5 — SSRF defence
# ---------------------------------------------------------------------------


class TestSSRFDefence:
    def test_localhost_blocked(self) -> None:
        result = check_public_url("http://localhost:8080/")
        assert not result.allowed

    def test_loopback_ip_blocked(self) -> None:
        result = check_public_url("http://127.0.0.1/")
        assert not result.allowed
        assert "loopback" in (result.reason or "")

    def test_private_10_blocked(self) -> None:
        result = check_public_url("http://10.0.0.5/")
        assert not result.allowed
        assert "private" in (result.reason or "")

    def test_private_192_168_blocked(self) -> None:
        result = check_public_url("http://192.168.1.1/")
        assert not result.allowed

    def test_aws_metadata_blocked(self) -> None:
        result = check_public_url("http://169.254.169.254/")
        assert not result.allowed
        assert "link_local" in (result.reason or "")

    def test_ipv6_loopback_blocked(self) -> None:
        result = check_public_url("http://[::1]/")
        assert not result.allowed

    def test_dns_rebinding_blocked(self) -> None:
        # Resolve to a private IP via mocked getaddrinfo.
        with patch(
            "src.queue._url_safety.socket.getaddrinfo",
            return_value=[
                (socket.AF_INET, socket.SOCK_STREAM, 0, "", ("127.0.0.1", 0))
            ],
        ):
            result = check_public_url("http://attacker.example.com/")
        assert not result.allowed

    def test_unknown_scheme_blocked(self) -> None:
        result = check_public_url("file:///etc/passwd")
        assert not result.allowed


# ---------------------------------------------------------------------------
# 8C.B2 — DLQ poison detection on timeout-only failures
# ---------------------------------------------------------------------------


class TestDLQPoisonTimeoutOnly:
    def test_all_none_errors_marked_opaque_timeout(self) -> None:
        dlq = WebhookDLQ(db_path=":memory:")
        for _ in range(5):
            dlq.enqueue_failed(
                subscription_id="sub",
                payload={},
                last_error=None,
            )
        result = dlq.detect_poison_subscription("sub", consecutive_threshold=5)
        assert result.poisoned is True
        assert result.signature == "opaque_timeout"

    def test_all_empty_errors_also_caught(self) -> None:
        dlq = WebhookDLQ(db_path=":memory:")
        for _ in range(5):
            dlq.enqueue_failed(
                subscription_id="sub2", payload={}, last_error="",
            )
        result = dlq.detect_poison_subscription("sub2", consecutive_threshold=5)
        assert result.poisoned is True
        assert result.signature == "opaque_timeout"


# ---------------------------------------------------------------------------
# 8C.B3 — Calibration ECE rollback skip on first fit
# ---------------------------------------------------------------------------


class TestCalibrationFirstFit:
    def test_first_fit_accepts_unconditionally(self) -> None:
        pc = PartitionedCalibrator()
        for i in range(25):
            pc.add_point(
                CalibrationPoint(
                    raw_confidence=0.5 + (i % 5) * 0.1,
                    is_correct=(i % 3 != 0),
                )
            )
        results = pc.fit_all()
        gk = pc.GLOBAL_KEY
        assert results[gk].accepted is True
        # Method should be a real fit, not a rollback.
        assert results[gk].method_selected != "rollback_linear"
