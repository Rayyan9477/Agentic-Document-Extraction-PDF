"""Phase K — signed export receipt tests.

Covers:
* Receipt construction over a set of artefact files.
* Deterministic HMAC computation.
* Verifier accepts good receipts, rejects tampering.
* Audit-chain-tail field is optional (CLI runs may have no audit log).
* Unsigned receipts are still verifiable for artefact integrity.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from src.export.signed_receipt import (
    RECEIPT_SCHEMA_VERSION,
    SIGNATURE_ALGORITHM,
    SignedReceipt,
    mint_receipt,
    verify_receipt,
    write_receipt,
)


@pytest.fixture
def bundle_dir(tmp_path: Path) -> Path:
    """Create a fake bundle with three artefacts."""
    (tmp_path / "claim.json").write_text('{"patient": "Mary"}', encoding="utf-8")
    (tmp_path / "claim.fhir.json").write_text(
        '{"resourceType": "Bundle"}', encoding="utf-8"
    )
    (tmp_path / "claim_report.md").write_text("# Claim Report\n", encoding="utf-8")
    return tmp_path


@pytest.fixture
def fixed_now() -> datetime:
    return datetime(2026, 5, 17, 12, 0, 0, tzinfo=UTC)


# ---------------------------------------------------------------------------
# Minting
# ---------------------------------------------------------------------------


class TestMintReceipt:
    """``mint_receipt`` builds the canonical receipt object."""

    def test_minted_receipt_carries_schema_version(
        self, bundle_dir: Path, fixed_now: datetime
    ) -> None:
        receipt = mint_receipt(
            processing_id="proc-abc",
            profile="medical-rcm",
            artefact_paths=list(bundle_dir.iterdir()),
            now=fixed_now,
        )
        assert receipt.schema_version == RECEIPT_SCHEMA_VERSION

    def test_hashes_every_artefact(
        self, bundle_dir: Path, fixed_now: datetime
    ) -> None:
        receipt = mint_receipt(
            processing_id="proc-abc",
            profile="medical-rcm",
            artefact_paths=list(bundle_dir.iterdir()),
            now=fixed_now,
        )
        # Three files in, three hashes out — keyed by basename.
        assert set(receipt.artefact_hashes.keys()) == {
            "claim.json",
            "claim.fhir.json",
            "claim_report.md",
        }
        # Hashes are 64-char hex (SHA-256).
        for hexdigest in receipt.artefact_hashes.values():
            assert len(hexdigest) == 64

    def test_missing_files_are_skipped_not_fatal(
        self, bundle_dir: Path, fixed_now: datetime
    ) -> None:
        receipt = mint_receipt(
            processing_id="proc-abc",
            profile="medical-rcm",
            artefact_paths=[
                bundle_dir / "claim.json",
                bundle_dir / "does_not_exist.json",
            ],
            now=fixed_now,
        )
        assert "claim.json" in receipt.artefact_hashes
        assert "does_not_exist.json" not in receipt.artefact_hashes

    def test_unsigned_receipt_when_no_key(
        self, bundle_dir: Path, fixed_now: datetime
    ) -> None:
        receipt = mint_receipt(
            processing_id="proc-abc",
            profile="generic-document",
            artefact_paths=list(bundle_dir.iterdir()),
            signing_key=None,
            now=fixed_now,
        )
        assert receipt.signature is None
        assert receipt.signature_algorithm == SIGNATURE_ALGORITHM

    def test_signed_receipt_has_hex_signature(
        self, bundle_dir: Path, fixed_now: datetime
    ) -> None:
        receipt = mint_receipt(
            processing_id="proc-abc",
            profile="medical-rcm",
            artefact_paths=list(bundle_dir.iterdir()),
            signing_key="test-secret-key",
            signer_key_id="ops-2026-Q2",
            now=fixed_now,
        )
        assert receipt.signature is not None
        assert len(receipt.signature) == 64  # HMAC-SHA256 hex
        assert receipt.signer_key_id == "ops-2026-Q2"

    def test_deterministic_signature(
        self, bundle_dir: Path, fixed_now: datetime
    ) -> None:
        """Same inputs → same signature."""
        kwargs = dict(
            processing_id="proc-abc",
            profile="medical-rcm",
            artefact_paths=list(bundle_dir.iterdir()),
            signing_key="test-secret-key",
            now=fixed_now,
        )
        sig_a = mint_receipt(**kwargs).signature
        sig_b = mint_receipt(**kwargs).signature
        assert sig_a == sig_b

    def test_audit_tail_recorded(
        self, bundle_dir: Path, fixed_now: datetime
    ) -> None:
        receipt = mint_receipt(
            processing_id="proc-abc",
            profile="medical-rcm",
            artefact_paths=[bundle_dir / "claim.json"],
            audit_chain_tail="0123456789abcdef",
            now=fixed_now,
        )
        assert receipt.audit_chain_tail == "0123456789abcdef"

    def test_missing_audit_tail_is_none(
        self, bundle_dir: Path, fixed_now: datetime
    ) -> None:
        receipt = mint_receipt(
            processing_id="proc-abc",
            profile="medical-rcm",
            artefact_paths=[bundle_dir / "claim.json"],
            now=fixed_now,
        )
        assert receipt.audit_chain_tail is None


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------


class TestVerifyReceipt:
    """``verify_receipt`` validates artefact hashes + signature."""

    def test_valid_receipt_passes(self, bundle_dir: Path, fixed_now: datetime) -> None:
        receipt = mint_receipt(
            processing_id="proc-abc",
            profile="medical-rcm",
            artefact_paths=list(bundle_dir.iterdir()),
            signing_key="test-secret-key",
            now=fixed_now,
        )
        result = verify_receipt(
            receipt,
            bundle_dir=bundle_dir,
            signing_key="test-secret-key",
        )
        assert result.valid is True
        assert result.reason is None

    def test_tampered_artefact_detected(
        self, bundle_dir: Path, fixed_now: datetime
    ) -> None:
        receipt = mint_receipt(
            processing_id="proc-abc",
            profile="medical-rcm",
            artefact_paths=list(bundle_dir.iterdir()),
            signing_key="test-secret-key",
            now=fixed_now,
        )
        # Tamper with one artefact after the receipt was minted.
        (bundle_dir / "claim.json").write_text('{"patient": "Jane"}', encoding="utf-8")
        result = verify_receipt(
            receipt,
            bundle_dir=bundle_dir,
            signing_key="test-secret-key",
        )
        assert result.valid is False
        assert result.reason == "artefact_hash_mismatch"
        assert "claim.json" in result.mismatched_artefacts

    def test_missing_artefact_detected(
        self, bundle_dir: Path, fixed_now: datetime
    ) -> None:
        receipt = mint_receipt(
            processing_id="proc-abc",
            profile="medical-rcm",
            artefact_paths=list(bundle_dir.iterdir()),
            signing_key="test-secret-key",
            now=fixed_now,
        )
        (bundle_dir / "claim.fhir.json").unlink()
        result = verify_receipt(
            receipt,
            bundle_dir=bundle_dir,
            signing_key="test-secret-key",
        )
        assert result.valid is False
        assert result.reason == "artefact_hash_mismatch"
        assert "claim.fhir.json" in result.mismatched_artefacts

    def test_signature_required_when_signed(
        self, bundle_dir: Path, fixed_now: datetime
    ) -> None:
        receipt = mint_receipt(
            processing_id="proc-abc",
            profile="medical-rcm",
            artefact_paths=list(bundle_dir.iterdir()),
            signing_key="test-secret-key",
            now=fixed_now,
        )
        result = verify_receipt(receipt, bundle_dir=bundle_dir, signing_key=None)
        assert result.valid is False
        assert result.reason == "key_required"

    def test_wrong_key_rejected(self, bundle_dir: Path, fixed_now: datetime) -> None:
        receipt = mint_receipt(
            processing_id="proc-abc",
            profile="medical-rcm",
            artefact_paths=list(bundle_dir.iterdir()),
            signing_key="real-key",
            now=fixed_now,
        )
        result = verify_receipt(
            receipt, bundle_dir=bundle_dir, signing_key="attacker-key"
        )
        assert result.valid is False
        assert result.reason == "signature_mismatch"

    def test_unsigned_receipt_still_verifies_hashes(
        self, bundle_dir: Path, fixed_now: datetime
    ) -> None:
        receipt = mint_receipt(
            processing_id="proc-abc",
            profile="generic-document",
            artefact_paths=list(bundle_dir.iterdir()),
            signing_key=None,
            now=fixed_now,
        )
        result = verify_receipt(receipt, bundle_dir=bundle_dir, signing_key=None)
        assert result.valid is True

    def test_accepts_dict_form(self, bundle_dir: Path, fixed_now: datetime) -> None:
        """``receipt.json`` round-trip: write, load as dict, verify."""
        receipt = mint_receipt(
            processing_id="proc-abc",
            profile="medical-rcm",
            artefact_paths=list(bundle_dir.iterdir()),
            signing_key="test-secret-key",
            now=fixed_now,
        )
        receipt_path = write_receipt(receipt, bundle_dir / "receipt.json")
        loaded = json.loads(receipt_path.read_text(encoding="utf-8"))
        # Remove the receipt file itself from the bundle dir before
        # verifying — otherwise its hash isn't in the receipt and the
        # verifier's filename loop walks past it cleanly (we don't add
        # the receipt to its own hash list).
        result = verify_receipt(
            loaded,
            bundle_dir=bundle_dir,
            signing_key="test-secret-key",
        )
        assert result.valid is True
