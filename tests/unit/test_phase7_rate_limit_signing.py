"""V3 Phase 7 — Per-tenant rate limit + queue depth + RCM signing config."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path

import pytest

from src.api.middleware import RateLimiter
from src.client.backends.queue_depth import (
    configure,
    is_active,
    reset,
    vlm_queue_slot,
    vlm_queue_slot_async,
)
from src.export.rcm_signing import (
    SIGNING_BACKEND_AWS_KMS,
    SIGNING_BACKEND_FILE,
    SIGNING_BACKEND_GCP_KMS,
    SIGNING_BACKEND_UNCONFIGURED,
    SIGNING_BACKEND_VAULT_TRANSIT,
    AwsKmsSigner,
    GcpKmsSigner,
    LocalFileSigner,
    RCMSigningConfig,
    UnconfiguredSigner,
    VaultTransitSigner,
    get_signer,
)


# ---------------------------------------------------------------------------
# Per-tenant rate limit
# ---------------------------------------------------------------------------


class TestPerTenantRateLimit:
    def test_tenant_quota_overrides_default(self) -> None:
        limiter = RateLimiter(default_rpm=1000, burst_size=100)
        limiter.set_tenant_limit("acme", rpm=2, burst=2)
        # First two requests pass.
        a1, _ = limiter.is_allowed("client", tenant_id="acme")
        a2, _ = limiter.is_allowed("client", tenant_id="acme")
        # Burst exhausted; third blocked.
        a3, _ = limiter.is_allowed("client", tenant_id="acme")
        assert a1
        assert a2
        assert not a3

    def test_separate_tenants_have_separate_quotas(self) -> None:
        limiter = RateLimiter(default_rpm=1000)
        limiter.set_tenant_limit("acme", rpm=1, burst=1)
        limiter.set_tenant_limit("globex", rpm=1, burst=1)
        # acme uses its budget.
        assert limiter.is_allowed("c1", tenant_id="acme")[0]
        assert not limiter.is_allowed("c2", tenant_id="acme")[0]
        # globex is unaffected.
        assert limiter.is_allowed("c1", tenant_id="globex")[0]

    def test_no_tenant_id_uses_default(self) -> None:
        limiter = RateLimiter(default_rpm=2, burst_size=2)
        # Without tenant_id, default behaviour applies.
        for _ in range(2):
            assert limiter.is_allowed("ip:1.2.3.4")[0]
        # 3rd within the same minute is rejected.
        assert not limiter.is_allowed("ip:1.2.3.4")[0]

    def test_get_tenant_limit_returns_config(self) -> None:
        limiter = RateLimiter()
        assert limiter.get_tenant_limit("nope") is None
        limiter.set_tenant_limit("acme", rpm=10, burst=2)
        cfg = limiter.get_tenant_limit("acme")
        assert cfg is not None
        assert cfg.requests_per_minute == 10

    def test_negative_rpm_rejected(self) -> None:
        limiter = RateLimiter()
        with pytest.raises(ValueError):
            limiter.set_tenant_limit("acme", rpm=-1)


# ---------------------------------------------------------------------------
# Queue-depth gate
# ---------------------------------------------------------------------------


class TestQueueDepth:
    def teardown_method(self, method) -> None:
        reset()

    def test_disabled_by_default(self) -> None:
        reset()
        assert is_active() is False

    def test_capacity_zero_means_no_op(self) -> None:
        configure(0)
        assert is_active() is False
        with vlm_queue_slot():
            pass

    def test_acquire_release_sync(self) -> None:
        configure(2)
        assert is_active()
        with vlm_queue_slot():
            with vlm_queue_slot():
                pass

    def test_timeout_raises(self) -> None:
        configure(1)
        # Hold the only slot, then attempt a second acquire with a
        # short timeout — should raise.
        with vlm_queue_slot():
            with pytest.raises(TimeoutError):
                with vlm_queue_slot(timeout=0.1):
                    pass

    def test_async_acquire_release(self) -> None:
        configure(2)

        async def _scenario() -> None:
            async with vlm_queue_slot_async():
                async with vlm_queue_slot_async():
                    pass

        asyncio.run(_scenario())

    def test_reset_disables(self) -> None:
        configure(3)
        assert is_active()
        reset()
        assert is_active() is False


# ---------------------------------------------------------------------------
# RCM signing config + factory
# ---------------------------------------------------------------------------


class TestSigningFactory:
    def test_unconfigured_returns_unconfigured(self) -> None:
        signer = get_signer(RCMSigningConfig())
        assert isinstance(signer, UnconfiguredSigner)

    def test_unconfigured_sign_raises(self) -> None:
        signer = get_signer(RCMSigningConfig())
        with pytest.raises(RuntimeError, match="not configured"):
            signer.sign(b"payload")

    def test_unknown_backend_raises(self) -> None:
        with pytest.raises(ValueError, match="unknown signing backend"):
            get_signer(RCMSigningConfig(backend="totally-fake"))

    def test_file_backend_requires_both_paths(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="signing_cert"):
            get_signer(
                RCMSigningConfig(
                    backend=SIGNING_BACKEND_FILE,
                    signing_cert=None,
                    signing_key=tmp_path / "key.pem",
                )
            )

    def test_file_backend_missing_files(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            get_signer(
                RCMSigningConfig(
                    backend=SIGNING_BACKEND_FILE,
                    signing_cert=tmp_path / "cert.pem",
                    signing_key=tmp_path / "key.pem",
                )
            )

    def test_aws_kms_requires_key_id(self) -> None:
        with pytest.raises(ValueError, match="kms_key_id"):
            get_signer(RCMSigningConfig(backend=SIGNING_BACKEND_AWS_KMS))

    def test_aws_kms_signer_is_stub(self) -> None:
        signer = get_signer(
            RCMSigningConfig(backend=SIGNING_BACKEND_AWS_KMS, kms_key_id="arn:x")
        )
        assert isinstance(signer, AwsKmsSigner)
        with pytest.raises(NotImplementedError):
            signer.sign(b"payload")

    def test_gcp_kms_signer_is_stub(self) -> None:
        signer = get_signer(
            RCMSigningConfig(backend=SIGNING_BACKEND_GCP_KMS, kms_key_id="project/...")
        )
        assert isinstance(signer, GcpKmsSigner)
        with pytest.raises(NotImplementedError):
            signer.sign(b"payload")

    def test_vault_signer_is_stub(self) -> None:
        signer = get_signer(
            RCMSigningConfig(
                backend=SIGNING_BACKEND_VAULT_TRANSIT,
                vault_transit_path="transit/sign/rcm",
            )
        )
        assert isinstance(signer, VaultTransitSigner)
        with pytest.raises(NotImplementedError):
            signer.sign(b"payload")

    def test_vault_requires_path(self) -> None:
        with pytest.raises(ValueError, match="vault_transit_path"):
            get_signer(RCMSigningConfig(backend=SIGNING_BACKEND_VAULT_TRANSIT))

    def test_config_to_dict_round_trips(self, tmp_path: Path) -> None:
        cfg = RCMSigningConfig(
            backend=SIGNING_BACKEND_AWS_KMS,
            kms_key_id="arn:aws:kms:us-east-1:123:key/abc",
        )
        d = cfg.to_dict()
        assert d["backend"] == SIGNING_BACKEND_AWS_KMS
        assert d["kms_key_id"] == "arn:aws:kms:us-east-1:123:key/abc"
