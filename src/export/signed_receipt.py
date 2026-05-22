"""
Phase K — signed export receipt.

Every Healthcare-mode export bundle (and optionally General-mode) ships
with a ``receipt.json`` that binds:

* the SHA-256 of every artefact in the bundle (json / md / xlsx / fhir.json /
  bbox-overlay PNGs),
* the audit-chain tail hash (Phase 8 ``verify_audit_chain_with_anchor``)
  when available,
* the processing id, timestamp, key id, and HMAC signature,

into one offline-verifiable JSON object. A reviewer who downloads the
bundle can run ``verify_receipt(receipt, key)`` and confirm:

1. No bundle artefact has been tampered with since emission.
2. The audit chain at the moment of emission is intact (when the audit
   log was available).
3. The signature was produced with the named key.

The cryptographic primitive is HMAC-SHA-256 with a shared secret pulled
from ``settings.api.export_receipt_signing_key`` (env
``EXPORT_RECEIPT_SIGNING_KEY``). A keyless installation still produces a
receipt — the ``signature`` field is ``None`` — so downstream consumers
that don't care about cryptographic provenance still get the artefact
hashes.

This is **not** PKCS#7 / X.509 signing — for production-grade signing
see ``src/export/rcm_signing.py`` (Phase 7), which supports AWS KMS /
Vault Transit. The receipt is a lightweight, dependency-free integrity
attestation that works in air-gapped environments.

Wire-up:
* ``src/export/consolidated_export.py::write_signed_receipt`` calls
  ``mint_receipt`` once all other export files are written.
* ``main.py extract_pdf_cli`` invokes that wrapper as the final step.
"""

from __future__ import annotations

import hashlib
import hmac
import json
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


RECEIPT_SCHEMA_VERSION = "1.0"
"""Versioned schema. Bump when the receipt-fields shape changes."""

SIGNATURE_ALGORITHM = "HMAC-SHA256"


@dataclass(frozen=True, slots=True)
class SignedReceipt:
    """Offline-verifiable integrity attestation for an export bundle."""

    schema_version: str
    processing_id: str
    profile: str | None
    signed_at: str
    artefact_hashes: dict[str, str]
    audit_chain_tail: str | None
    signer_key_id: str | None
    signature_algorithm: str
    signature: str | None
    notes: tuple[str, ...] = field(default_factory=tuple)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["notes"] = list(self.notes)
        return d


# ---------------------------------------------------------------------------
# Hashing helpers
# ---------------------------------------------------------------------------


def _sha256_file(path: Path, chunk_size: int = 65536) -> str:
    """Stream a file through SHA-256, return hex digest."""
    hasher = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def _canonical_signing_payload(receipt_without_signature: dict[str, Any]) -> bytes:
    """Serialise the receipt deterministically for HMAC input.

    Keys are sorted and separators are tight so the byte form is stable
    regardless of dict-insertion order. The signature field is
    explicitly excluded — that's what we're computing.
    """
    payload = {k: v for k, v in receipt_without_signature.items() if k != "signature"}
    return json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def mint_receipt(
    *,
    processing_id: str,
    profile: str | None,
    artefact_paths: list[Path] | list[str],
    audit_chain_tail: str | None = None,
    signing_key: bytes | str | None = None,
    signer_key_id: str | None = None,
    notes: tuple[str, ...] = (),
    now: datetime | None = None,
) -> SignedReceipt:
    """Build a ``SignedReceipt`` from a set of artefact files.

    Args:
        processing_id: The pipeline's processing id (binds the receipt to
            a specific extraction run).
        profile: The active mode profile (``"medical-rcm"`` or
            ``"generic-document"``). Stored verbatim so verifiers can
            distinguish Healthcare-mode receipts.
        artefact_paths: Paths to every file the bundle should attest.
            Each path is opened, streamed through SHA-256, and recorded
            under its filename in ``artefact_hashes``.
        audit_chain_tail: Optional tail hash from
            ``verify_audit_chain_with_anchor``. ``None`` means the audit
            log wasn't available at emission time (e.g. CLI run without
            a configured audit dir) — the receipt is still mint-able.
        signing_key: HMAC shared secret. ``None`` produces an unsigned
            receipt (signature field is ``None``). Tests pass a known
            key; production reads from
            ``settings.api.export_receipt_signing_key``.
        signer_key_id: Operator-chosen identifier for the key (e.g.
            ``"ops-2026-Q2"``). Recorded so a key-rotation event can be
            traced post-hoc.
        notes: Optional free-form notes appended to the receipt
            (operators sometimes want to stamp a build version, deploy
            environment, etc.).
        now: Inject a fixed timestamp for tests. Defaults to
            ``datetime.now(UTC)``.

    Returns:
        A ``SignedReceipt`` ready to serialise via ``write_receipt``.
    """
    now = now or datetime.now(UTC)
    artefact_hashes: dict[str, str] = {}
    for raw in artefact_paths:
        path = Path(raw)
        if not path.exists():
            # Skip silently — missing files don't get attested. The
            # receipt only covers what's actually in the bundle.
            continue
        artefact_hashes[path.name] = _sha256_file(path)

    body: dict[str, Any] = {
        "schema_version": RECEIPT_SCHEMA_VERSION,
        "processing_id": processing_id,
        "profile": profile,
        "signed_at": now.isoformat(timespec="seconds"),
        "artefact_hashes": dict(sorted(artefact_hashes.items())),
        "audit_chain_tail": audit_chain_tail,
        "signer_key_id": signer_key_id,
        "signature_algorithm": SIGNATURE_ALGORITHM,
        "signature": None,
        "notes": list(notes),
    }

    if signing_key:
        key_bytes = signing_key.encode("utf-8") if isinstance(signing_key, str) else signing_key
        message = _canonical_signing_payload(body)
        body["signature"] = hmac.new(key_bytes, message, hashlib.sha256).hexdigest()

    return SignedReceipt(
        schema_version=body["schema_version"],
        processing_id=body["processing_id"],
        profile=body["profile"],
        signed_at=body["signed_at"],
        artefact_hashes=body["artefact_hashes"],
        audit_chain_tail=body["audit_chain_tail"],
        signer_key_id=body["signer_key_id"],
        signature_algorithm=body["signature_algorithm"],
        signature=body["signature"],
        notes=tuple(body["notes"]),
    )


def write_receipt(receipt: SignedReceipt, output_path: Path | str) -> Path:
    """Write a receipt to disk as pretty-printed JSON. Returns the path."""
    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(receipt.to_dict(), indent=2, sort_keys=True),
        encoding="utf-8",
    )
    return out


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class VerificationResult:
    """Outcome of ``verify_receipt``."""

    valid: bool
    reason: str | None = None
    mismatched_artefacts: tuple[str, ...] = field(default_factory=tuple)


def verify_receipt(
    receipt: SignedReceipt | dict[str, Any],
    *,
    bundle_dir: Path | str,
    signing_key: bytes | str | None = None,
) -> VerificationResult:
    """Re-hash bundle artefacts and re-compute the HMAC. Pure-function check.

    Args:
        receipt: Either a ``SignedReceipt`` or the dict form (e.g. loaded
            from ``receipt.json``).
        bundle_dir: Directory containing the artefacts named in
            ``receipt.artefact_hashes``. Files not present in the bundle
            dir are reported as ``"missing_artefact"``.
        signing_key: The key the receipt was minted with. Required when
            the receipt carries a signature. When ``None`` and the
            receipt has a signature, the verifier reports
            ``"key_required"``.

    Returns:
        ``VerificationResult`` with ``valid=True`` only when every
        artefact hash matches AND the signature re-computes correctly
        (when a key is supplied).
    """
    if isinstance(receipt, SignedReceipt):
        receipt_dict: dict[str, Any] = receipt.to_dict()
    else:
        receipt_dict = dict(receipt)

    bundle_dir = Path(bundle_dir)
    mismatches: list[str] = []
    for filename, expected_hash in receipt_dict.get("artefact_hashes", {}).items():
        path = bundle_dir / filename
        if not path.exists():
            mismatches.append(filename)
            continue
        actual = _sha256_file(path)
        if actual != expected_hash:
            mismatches.append(filename)

    if mismatches:
        return VerificationResult(
            valid=False,
            reason="artefact_hash_mismatch",
            mismatched_artefacts=tuple(mismatches),
        )

    sig = receipt_dict.get("signature")
    if sig:
        if signing_key is None:
            return VerificationResult(valid=False, reason="key_required")
        key_bytes = (
            signing_key.encode("utf-8") if isinstance(signing_key, str) else signing_key
        )
        message = _canonical_signing_payload(receipt_dict)
        expected_sig = hmac.new(key_bytes, message, hashlib.sha256).hexdigest()
        if not hmac.compare_digest(sig, expected_sig):
            return VerificationResult(valid=False, reason="signature_mismatch")

    return VerificationResult(valid=True)


__all__ = [
    "RECEIPT_SCHEMA_VERSION",
    "SIGNATURE_ALGORITHM",
    "SignedReceipt",
    "VerificationResult",
    "mint_receipt",
    "verify_receipt",
    "write_receipt",
]
