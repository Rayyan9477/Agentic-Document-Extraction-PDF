"""
Deterministic 63-bit integer projection for Qdrant point ids (Phase 8.5-A4).

Qdrant requires point ids to be unsigned integers (or UUIDs). The previous
implementation used ``hash(string_id) % (2**63)``, which fails subtly:

    * ``hash()`` is non-deterministic across Python processes — every worker
      restart picks a fresh ``PYTHONHASHSEED``, so the *same* external id
      maps to a *different* Qdrant int. Upserts silently duplicate, queries
      miss historical records, and reconciler tier-5 (FAISS field history)
      lookups become unreliable across pod restarts.
    * Cross-process tests that seed deterministically (``PYTHONHASHSEED=0``)
      hide the bug locally.

The fix is a deterministic hash. We use ``hashlib.blake2b`` with an 8-byte
digest, parsed as an unsigned int and masked into the Qdrant int63 range.

Why blake2b:
    * Available in the stdlib (no new dependency).
    * Fast (faster than SHA-256 for short inputs).
    * Cryptographically strong — collision resistance is overkill for a
      63-bit id space but it guarantees no adversarial input can force a
      collision.

Rejected alternatives:
    * ``uuid.uuid5(...)``: returns 128-bit UUIDs that overflow Qdrant's int64
      contract; would force a separate uuid-mode upsert path.
    * ``int(hashlib.md5(...).hexdigest(), 16)``: works but md5 is deprecated.
    * ``zlib.crc32``: only 32 bits — too narrow for the 1B+ vectors at scale.

The ``int63`` range (not int64) is deliberate. Qdrant's documentation says
unsigned 64-bit, but several client versions choke on values > ``2**63 - 1``
when round-tripping through JSON. ``int63`` is the safe portable range.
"""

from __future__ import annotations

import hashlib


# 8 bytes = 64 bits; mask one bit off to stay in the safe int63 range.
_INT63_MASK: int = (1 << 63) - 1
_DIGEST_SIZE_BYTES: int = 8


def safe_query_id(raw_id: str) -> int:
    """Project an arbitrary string id into a deterministic 63-bit integer.

    Args:
        raw_id: The external id (typically a UUID4 string or a content-hash).

    Returns:
        An ``int`` in the inclusive range ``[0, 2**63 - 1]`` that is
        deterministic across processes, machines, and Python versions.

    Raises:
        TypeError: If ``raw_id`` is not a ``str``.
    """
    if not isinstance(raw_id, str):
        raise TypeError(
            f"safe_query_id expects a string id, got {type(raw_id).__name__}"
        )

    digest = hashlib.blake2b(raw_id.encode("utf-8"), digest_size=_DIGEST_SIZE_BYTES)
    return int.from_bytes(digest.digest(), byteorder="big") & _INT63_MASK


__all__ = ["safe_query_id"]
