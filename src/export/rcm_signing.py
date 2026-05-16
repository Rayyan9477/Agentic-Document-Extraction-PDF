"""V3 Phase 7 — Pluggable signing configuration for Medical-RCM emitters.

The C-CDA / X12N 275 emitters need to attach a PKCS#7 signature to
the bundle they hand to the payer. Production deployments vary
widely on where the signing key lives:

* On-prem / dev: a self-signed PEM file pair on disk.
* AWS deployments: the private key lives in AWS KMS and only the
  certificate is on disk.
* HashiCorp Vault deployments: ``vault.transit.sign`` is the
  signing primitive.
* Google Cloud KMS: similar to AWS, distinct API.

This module defines the **config + protocol** that emitters will
consume. The emitters themselves are not yet built (Phase 5
roadmap deferred them); this lays the foundation so the emitters
land with a well-understood signing interface in Phase 7.

Design choice: the signing backend is selected via
``settings.export.rcm_signing.*`` and resolved by a single factory
``get_signer()``. Emitters never import a specific backend; they
take a ``Signer`` and call ``.sign(payload) -> SignedPayload``.

For Phase 7 we ship two backends:
* ``LocalFileSigner`` — PEM cert + PEM key on disk. Dev/on-prem.
* ``UnconfiguredSigner`` — pass-through that raises a clear error
  on ``.sign()``. Used when no backend is configured so emitters
  fail loudly rather than silently emitting unsigned bundles.

KMS / Vault backends are stub implementations that raise
``NotImplementedError`` with a clear message pointing operators at
the integration. The real KMS code lives in Phase 7+ in the
deployment-specific repo (it depends on the cloud SDK choice).
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


SIGNING_BACKEND_FILE = "file"
SIGNING_BACKEND_AWS_KMS = "aws_kms"
SIGNING_BACKEND_VAULT_TRANSIT = "vault_transit"
SIGNING_BACKEND_GCP_KMS = "gcp_kms"
SIGNING_BACKEND_UNCONFIGURED = "unconfigured"

SUPPORTED_SIGNING_BACKENDS = (
    SIGNING_BACKEND_FILE,
    SIGNING_BACKEND_AWS_KMS,
    SIGNING_BACKEND_VAULT_TRANSIT,
    SIGNING_BACKEND_GCP_KMS,
    SIGNING_BACKEND_UNCONFIGURED,
)


@dataclass(frozen=True, slots=True)
class RCMSigningConfig:
    """Configuration for the C-CDA / X12N 275 signing path.

    All fields are optional so callers can build minimal configs;
    each backend documents which subset of fields it consumes.
    Field validation lives in ``get_signer()`` so the config is a
    pure data carrier.
    """

    backend: str = SIGNING_BACKEND_UNCONFIGURED
    """One of ``SUPPORTED_SIGNING_BACKENDS``."""

    signing_cert: Path | None = None
    """X.509 certificate (PEM-encoded) used by every backend except UNCONFIGURED."""

    signing_key: Path | None = None
    """PEM private key. Required by FILE backend; ignored by KMS/Vault."""

    kms_key_id: str | None = None
    """AWS KMS / GCP KMS key identifier (ARN or resource name)."""

    vault_transit_path: str | None = None
    """HashiCorp Vault Transit engine path (e.g. ``transit/sign/rcm-key``)."""

    notes: tuple[str, ...] = field(default_factory=tuple)
    """Operator-facing notes (deployment-specific overrides etc.)."""

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": self.backend,
            "signing_cert": str(self.signing_cert) if self.signing_cert else None,
            "signing_key": str(self.signing_key) if self.signing_key else None,
            "kms_key_id": self.kms_key_id,
            "vault_transit_path": self.vault_transit_path,
            "notes": list(self.notes),
        }


# ---------------------------------------------------------------------------
# Signed payload
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SignedPayload:
    """The output of ``Signer.sign``.

    ``signature_bytes`` is the raw signature (PKCS#7 DER, or whatever
    the backend produces). ``signature_format`` is a short label
    consumers can use to select a verification path
    (``"pkcs7-der"``, ``"pkcs1-v1_5"``, etc.).
    """

    payload: bytes
    signature_bytes: bytes
    signature_format: str
    signer_identity: str
    """Free-form identity label — the certificate Common Name, KMS key id, etc."""


# ---------------------------------------------------------------------------
# Signer protocol
# ---------------------------------------------------------------------------


class Signer(ABC):
    """Common surface every signing backend implements."""

    backend_name: str = "unknown"

    @abstractmethod
    def sign(self, payload: bytes) -> SignedPayload:
        """Sign ``payload`` and return ``SignedPayload``."""

    def describe(self) -> dict[str, Any]:
        """Diagnostic description of what backend / identity is signing."""
        return {"backend": self.backend_name}


# ---------------------------------------------------------------------------
# Backends
# ---------------------------------------------------------------------------


class UnconfiguredSigner(Signer):
    """Pass-through that raises on ``sign()``.

    The factory returns this when no signing backend is configured.
    Emitters that try to sign with it get a loud error instead of
    silently emitting an unsigned bundle.
    """

    backend_name = SIGNING_BACKEND_UNCONFIGURED

    def sign(self, payload: bytes) -> SignedPayload:
        raise RuntimeError(
            "RCM signing backend is not configured. Set "
            "settings.export.rcm_signing.backend to one of "
            f"{SUPPORTED_SIGNING_BACKENDS} and provide the matching "
            "credentials before emitting C-CDA / X12N 275."
        )


class LocalFileSigner(Signer):
    """PEM cert + key on disk (development / on-prem deployments).

    Lazy-loads the cryptography library so installations without
    the [signing] extra still import this module cleanly.
    """

    backend_name = SIGNING_BACKEND_FILE

    def __init__(
        self,
        signing_cert: Path,
        signing_key: Path,
    ) -> None:
        if not signing_cert.exists():
            raise FileNotFoundError(f"signing_cert not found: {signing_cert}")
        if not signing_key.exists():
            raise FileNotFoundError(f"signing_key not found: {signing_key}")
        self._cert_path = signing_cert
        self._key_path = signing_key

    def sign(self, payload: bytes) -> SignedPayload:
        try:
            from cryptography.hazmat.primitives import hashes, serialization
            from cryptography.hazmat.primitives.asymmetric import padding
            from cryptography.x509 import load_pem_x509_certificate
        except ImportError as e:  # pragma: no cover - optional dep
            raise RuntimeError(
                "Local-file signing requires the ``cryptography`` "
                "package. Install via the [signing] extra."
            ) from e

        cert = load_pem_x509_certificate(self._cert_path.read_bytes())
        key = serialization.load_pem_private_key(
            self._key_path.read_bytes(),
            password=None,
        )
        signature = key.sign(  # type: ignore[union-attr]
            payload,
            padding.PKCS1v15(),
            hashes.SHA256(),
        )
        common_name = ""
        try:
            from cryptography.x509.oid import NameOID

            for attr in cert.subject:
                if attr.oid == NameOID.COMMON_NAME:
                    common_name = str(attr.value)
                    break
        except Exception:  # pragma: no cover - defensive
            pass
        return SignedPayload(
            payload=payload,
            signature_bytes=signature,
            signature_format="pkcs1-v1_5-sha256",
            signer_identity=common_name or str(self._cert_path),
        )


class _KMSSigner(Signer):
    """Common parent for cloud-KMS-style backends.

    The actual signing call is deployment-specific (different SDKs);
    this base raises with a clear message pointing operators at the
    integration. Phase 7+ deployment repos override ``sign``.
    """

    def __init__(self, key_id: str) -> None:
        self._key_id = key_id

    def describe(self) -> dict[str, Any]:
        return {"backend": self.backend_name, "key_id": self._key_id}


class AwsKmsSigner(_KMSSigner):
    backend_name = SIGNING_BACKEND_AWS_KMS

    def sign(self, payload: bytes) -> SignedPayload:
        raise NotImplementedError(
            "AwsKmsSigner is a stub. Production deployments wire "
            "this to ``boto3.client('kms').sign(...)``; see "
            "docs/DEPLOYMENT.md for the integration recipe."
        )


class GcpKmsSigner(_KMSSigner):
    backend_name = SIGNING_BACKEND_GCP_KMS

    def sign(self, payload: bytes) -> SignedPayload:
        raise NotImplementedError(
            "GcpKmsSigner is a stub. Production deployments wire "
            "this to ``google.cloud.kms.KeyManagementServiceClient.asymmetric_sign``."
        )


class VaultTransitSigner(_KMSSigner):
    backend_name = SIGNING_BACKEND_VAULT_TRANSIT

    def sign(self, payload: bytes) -> SignedPayload:
        raise NotImplementedError(
            "VaultTransitSigner is a stub. Production deployments wire "
            "this to ``hvac.Client.secrets.transit.sign_data(...)``."
        )


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_signer(config: RCMSigningConfig) -> Signer:
    """Resolve a config to a concrete ``Signer`` instance.

    Validates required fields per backend and raises a clear
    ``ValueError`` for misconfigurations. The factory never silently
    falls back to ``UnconfiguredSigner`` — when an operator asks for
    KMS but didn't supply ``kms_key_id``, that's a deployment error
    and we say so loudly.
    """
    backend = (config.backend or SIGNING_BACKEND_UNCONFIGURED).lower()

    if backend not in SUPPORTED_SIGNING_BACKENDS:
        raise ValueError(
            f"unknown signing backend {backend!r}; expected one of "
            f"{SUPPORTED_SIGNING_BACKENDS}"
        )

    if backend == SIGNING_BACKEND_UNCONFIGURED:
        return UnconfiguredSigner()

    if backend == SIGNING_BACKEND_FILE:
        if config.signing_cert is None or config.signing_key is None:
            raise ValueError(
                "FILE signing backend requires both signing_cert and "
                "signing_key in RCMSigningConfig."
            )
        return LocalFileSigner(
            signing_cert=Path(config.signing_cert),
            signing_key=Path(config.signing_key),
        )

    if backend == SIGNING_BACKEND_AWS_KMS:
        if not config.kms_key_id:
            raise ValueError("AWS KMS backend requires kms_key_id (ARN).")
        return AwsKmsSigner(key_id=config.kms_key_id)

    if backend == SIGNING_BACKEND_GCP_KMS:
        if not config.kms_key_id:
            raise ValueError("GCP KMS backend requires kms_key_id (resource name).")
        return GcpKmsSigner(key_id=config.kms_key_id)

    if backend == SIGNING_BACKEND_VAULT_TRANSIT:
        if not config.vault_transit_path:
            raise ValueError(
                "Vault Transit backend requires vault_transit_path."
            )
        return VaultTransitSigner(key_id=config.vault_transit_path)

    # Unreachable given the validation above, but keep mypy happy.
    raise ValueError(f"unhandled signing backend {backend!r}")  # pragma: no cover
