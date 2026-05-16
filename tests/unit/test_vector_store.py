"""Tests for the deterministic Qdrant id projection (Phase 8.5-A4).

The previous ``hash(id) % (2**63)`` was randomised by ``PYTHONHASHSEED``,
causing same-id-different-int across worker restarts. ``safe_query_id``
replaces it with a blake2b-based stable projection.

The subprocess test deliberately spawns a child with a *different*
``PYTHONHASHSEED`` so the determinism guarantee is exercised against
real cross-process behaviour, not just within-test stability.
"""

from __future__ import annotations

import os
import subprocess
import sys
import textwrap

import pytest

from src.memory._qdrant_ids import safe_query_id


class TestSafeQueryIdRange:
    """The output must fit in Qdrant's safe-portable int63 range."""

    def test_output_is_non_negative(self) -> None:
        assert safe_query_id("any-id") >= 0

    def test_output_fits_in_int63(self) -> None:
        """Mask is ``(1 << 63) - 1`` so output is < 2**63."""
        assert safe_query_id("any-id") < (1 << 63)

    @pytest.mark.parametrize(
        "raw_id",
        [
            "",
            "a",
            "x" * 1024,
            "550e8400-e29b-41d4-a716-446655440000",
            "ünıçødé-id-🩺",  # non-ascii survives utf-8 encoding
        ],
    )
    def test_arbitrary_strings_produce_valid_int(self, raw_id: str) -> None:
        out = safe_query_id(raw_id)
        assert isinstance(out, int)
        assert 0 <= out < (1 << 63)


class TestSafeQueryIdDeterminism:
    """The whole point of this helper — same input → same output, forever."""

    def test_idempotent_same_process(self) -> None:
        a = safe_query_id("processing-id-1234")
        b = safe_query_id("processing-id-1234")
        assert a == b

    def test_different_inputs_produce_different_outputs(self) -> None:
        a = safe_query_id("processing-id-1234")
        b = safe_query_id("processing-id-1235")
        assert a != b

    def test_stable_across_pythonhashseed_via_subprocess(self) -> None:
        """The critical property: cross-process determinism.

        Spawns two child processes with deliberately-different
        ``PYTHONHASHSEED`` values. ``hash()`` would have returned
        different ints in each. ``safe_query_id`` must not.
        """
        script = textwrap.dedent(
            """
            import sys
            sys.path.insert(0, r"{root}")
            from src.memory._qdrant_ids import safe_query_id
            print(safe_query_id("phase8.5-A4-stability-canary"))
            """
        ).format(root=os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

        env_a = {**os.environ, "PYTHONHASHSEED": "0"}
        env_b = {**os.environ, "PYTHONHASHSEED": "42"}

        out_a = subprocess.check_output([sys.executable, "-c", script], env=env_a).strip()
        out_b = subprocess.check_output([sys.executable, "-c", script], env=env_b).strip()

        assert out_a == out_b, (
            "safe_query_id is not stable across PYTHONHASHSEED values: "
            f"seed=0 → {out_a!r}, seed=42 → {out_b!r}"
        )

    def test_known_canonical_value(self) -> None:
        """Pin one known-good (input, output) pair.

        If this test ever fails after a refactor, it means the projection
        has changed and every existing Qdrant collection just lost its
        identity-mapping. That's a release-note-worthy regression.
        """
        # Computed via:
        #   blake2b(b"veridoc-canon-id", digest_size=8).digest()
        # then int.from_bytes(.., 'big') & ((1 << 63) - 1)
        expected = int.from_bytes(
            __import__("hashlib")
            .blake2b(b"veridoc-canon-id", digest_size=8)
            .digest(),
            byteorder="big",
        ) & ((1 << 63) - 1)
        assert safe_query_id("veridoc-canon-id") == expected


class TestSafeQueryIdCollisionResistance:
    """Sanity check: 10k random-ish ids should produce 10k distinct outputs."""

    def test_no_collisions_in_10k_uuid_like_ids(self) -> None:
        ids = [f"proc-{i:08x}-tenant-{i % 7}" for i in range(10_000)]
        outputs = {safe_query_id(i) for i in ids}
        # Probability of even one collision in 2**63 over 10k samples is
        # vanishingly small (~5e-15) so any drop signals a bug.
        assert len(outputs) == len(ids)


class TestSafeQueryIdInput:
    def test_non_string_raises_typeerror(self) -> None:
        with pytest.raises(TypeError, match="expects a string id"):
            safe_query_id(12345)  # type: ignore[arg-type]

    def test_bytes_input_raises_typeerror(self) -> None:
        with pytest.raises(TypeError):
            safe_query_id(b"raw-bytes")  # type: ignore[arg-type]
