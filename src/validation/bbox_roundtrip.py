"""
V3 Phase 2 — Bbox round-trip verification.

When the reconciler hits a disputed field with a credible bbox (from
Pass 2's AUDITOR-mandated grounding) but neither pass agrees on the
value, this helper crops the page image to that bbox + padding and
re-queries the **secondary** VLM with a focused "what is the exact
text in this region?" prompt. The crop has no surrounding distractors
so the VLM's read is more accurate than its whole-page extraction
was — at the cost of one additional VLM call per flagged field.

This module is consumed by ``HeterogeneousReconciler`` step 3 and by
the Critic agent's ``verify_bbox`` recommendation in Phase 3.

The function is **sync** because the reconciler is sync; the
underlying backend may itself dispatch async via the OpenAI SDK pool.

This module is dependency-light — it imports PIL lazily so test
mocking is cheap. It does not depend on the orchestrator or any
agent class, only on the VLMBackend protocol.
"""

from __future__ import annotations

import base64
import io
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from src.client.backends.protocol import VLMBackend, VLMRole
from src.client.lm_client import VisionRequest
from src.config import get_logger


if TYPE_CHECKING:
    from PIL import Image as PILImage  # noqa: F401


logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Result
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class BboxRoundtripResult:
    """Outcome of a bbox round-trip verification call."""

    value: str | None
    """The value read from the cropped region, or None on failure."""

    confidence: float
    """Self-reported confidence from the secondary VLM."""

    similarity_to_pass1: float
    """Similarity ∈ [0, 1] between round-trip value and Pass 1 candidate."""

    similarity_to_pass2: float
    """Similarity ∈ [0, 1] between round-trip value and Pass 2 candidate."""

    winning_pass: str
    """Which pass the round-trip ratifies: ``pass1`` | ``pass2`` | ``neither``."""

    crop_size_px: tuple[int, int] = (0, 0)

    backend_name: str = ""
    """Backend that served the call (for trace correlation)."""

    error: str | None = None
    """Populated when the round-trip itself failed (transport, decode)."""


# ---------------------------------------------------------------------------
# Bbox helpers
# ---------------------------------------------------------------------------


def _normalise_bbox(
    bbox: tuple[float, float, float, float] | list[float],
) -> tuple[float, float, float, float]:
    """Coerce a bbox into ``(x1, y1, x2, y2)`` floats clamped to ``[0, 1]``."""
    if len(bbox) != 4:
        raise ValueError(f"bbox must have 4 elements, got {len(bbox)}: {bbox!r}")
    x1, y1, x2, y2 = (float(v) for v in bbox)
    # Some VLMs return [x, y, w, h]; we treat anything where x2 < x1 as that.
    if x2 < x1 or y2 < y1:
        x2, y2 = x1 + x2, y1 + y2
    x1 = max(0.0, min(1.0, x1))
    y1 = max(0.0, min(1.0, y1))
    x2 = max(0.0, min(1.0, x2))
    y2 = max(0.0, min(1.0, y2))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"degenerate bbox after clamp: ({x1}, {y1}, {x2}, {y2})")
    return x1, y1, x2, y2


def _expand_bbox(
    bbox: tuple[float, float, float, float],
    *,
    padding_pct: float = 0.10,
) -> tuple[float, float, float, float]:
    """Pad a normalised bbox outward by ``padding_pct`` of its dimensions.

    The crop performs better when there's a small margin around the
    target text (lets the VLM see leading/trailing whitespace, which
    helps with character disambiguation). Capped at the page edges.
    """
    x1, y1, x2, y2 = bbox
    w, h = x2 - x1, y2 - y1
    px, py = w * padding_pct, h * padding_pct
    return (
        max(0.0, x1 - px),
        max(0.0, y1 - py),
        min(1.0, x2 + px),
        min(1.0, y2 + py),
    )


def crop_image_to_bbox(
    image_data_uri: str,
    bbox: tuple[float, float, float, float] | list[float],
    *,
    padding_pct: float = 0.10,
) -> tuple[str, tuple[int, int]]:
    """Crop a base64 page image to a normalised bbox + padding.

    Args:
        image_data_uri: ``"data:image/png;base64,..."`` or raw base64.
        bbox: normalised ``(x1, y1, x2, y2)`` or ``[x, y, w, h]``.
        padding_pct: outward expansion as a fraction of bbox dimensions.

    Returns:
        ``(cropped_data_uri, (width_px, height_px))``.
    """
    from PIL import Image as PILImage  # lazy import for test cheapness

    if image_data_uri.startswith("data:"):
        # Parse "data:image/png;base64,XXXX"
        _, _, b64 = image_data_uri.partition(",")
    else:
        b64 = image_data_uri

    raw = base64.b64decode(b64)
    img = PILImage.open(io.BytesIO(raw))

    x1, y1, x2, y2 = _expand_bbox(_normalise_bbox(bbox), padding_pct=padding_pct)
    w, h = img.size
    crop_box = (
        int(round(x1 * w)),
        int(round(y1 * h)),
        int(round(x2 * w)),
        int(round(y2 * h)),
    )
    cropped = img.crop(crop_box)

    # Re-encode as PNG (lossless preserves character edges better than JPEG).
    buf = io.BytesIO()
    cropped.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}", cropped.size


# ---------------------------------------------------------------------------
# Similarity (mirrors DualPassComparator._calculate_similarity)
# ---------------------------------------------------------------------------


def _string_similarity(a: str, b: str) -> float:
    """Length-aware token overlap fallback when ``difflib`` is overkill.

    Mirrors the spirit of ``DualPassComparator._calculate_similarity``
    but kept tiny here so this module has no upstream dependency on
    ``dual_pass.py`` (which is a Phase 2 compat shim).
    """
    if a == b:
        return 1.0
    if not a or not b:
        return 0.0
    a_n, b_n = a.strip().lower(), b.strip().lower()
    if a_n == b_n:
        return 0.95
    # Levenshtein-ish via difflib for short strings.
    import difflib

    return difflib.SequenceMatcher(a=a_n, b=b_n).ratio()


def value_similarity(a: Any, b: Any) -> float:
    """Coerce both values to string and compare. Conservative and cheap.

    For numeric fields, callers may want to pre-normalise (drop
    currency symbols, trim leading zeros). The reconciler does that
    upstream; here we just compare strings.
    """
    if a is None or b is None:
        return 0.0
    return _string_similarity(str(a), str(b))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


ROUNDTRIP_PROMPT = (
    "What is the exact text or value visible in this image? "
    "Reply with ONLY the value, no commentary, no quotes, no labels. "
    "If the image is unreadable or empty, reply ``null``."
)


def perform_bbox_roundtrip(
    *,
    backend: VLMBackend,
    image_data_uri: str,
    bbox: tuple[float, float, float, float] | list[float],
    pass1_value: Any,
    pass2_value: Any,
    field_name: str,
    role: VLMRole = VLMRole.SECONDARY,
    padding_pct: float = 0.10,
    max_tokens: int = 64,
) -> BboxRoundtripResult:
    """Crop the page to ``bbox``, re-query the secondary VLM, compare.

    The round-trip ratifies whichever pass-1 / pass-2 candidate the
    cropped re-read agrees with, breaking the deadlock the reconciler
    handed us. When neither agrees (similarity < 0.7 to both), we
    return ``winning_pass="neither"`` and the reconciler downgrades
    the field to ``low_confidence``.

    Args:
        backend: ``VLMBackend`` to dispatch the call. Typically the
            same backend the orchestrator is using; ``role`` selects
            which model serves it (default ``SECONDARY`` so it's a
            different model family from Pass 1).
        image_data_uri: full-page image (data URI or raw base64).
        bbox: normalised bounding box from Pass 2 (or union of Pass 1
            + Pass 2 if both have bboxes).
        pass1_value, pass2_value: candidates to compare against.
        field_name: included in the log for traceability.
        role: VLM role to dispatch to.
        padding_pct: bbox outward padding for the crop.
        max_tokens: cap on response tokens — the round-trip prompt
            asks for the value only; 64 is plenty for any field.

    Returns:
        ``BboxRoundtripResult`` with the round-trip read + similarities
        + which pass it ratifies.
    """
    try:
        cropped_uri, crop_size = crop_image_to_bbox(
            image_data_uri,
            bbox,
            padding_pct=padding_pct,
        )
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning(
            "bbox_roundtrip_crop_failed",
            field=field_name,
            error=str(exc),
        )
        return BboxRoundtripResult(
            value=None,
            confidence=0.0,
            similarity_to_pass1=0.0,
            similarity_to_pass2=0.0,
            winning_pass="neither",
            backend_name=backend.name,
            error=f"crop_failed: {exc}",
        )

    request = VisionRequest(
        image_data=cropped_uri,
        prompt=ROUNDTRIP_PROMPT,
        max_tokens=max_tokens,
        temperature=0.0,
        json_mode=False,
    )

    try:
        response = backend.send_vision_request(request, role=role)
    except Exception as exc:
        logger.warning(
            "bbox_roundtrip_call_failed",
            field=field_name,
            backend=backend.name,
            role=role.value,
            error=str(exc),
        )
        return BboxRoundtripResult(
            value=None,
            confidence=0.0,
            similarity_to_pass1=0.0,
            similarity_to_pass2=0.0,
            winning_pass="neither",
            crop_size_px=crop_size,
            backend_name=backend.name,
            error=f"call_failed: {exc}",
        )

    raw = (response.content or "").strip()
    # The model may quote the value or wrap in markdown — strip lightly.
    if raw.startswith("`") and raw.endswith("`"):
        raw = raw.strip("`").strip()
    if raw.startswith('"') and raw.endswith('"'):
        raw = raw[1:-1]
    if raw.lower() in {"null", "none", "n/a", ""}:
        roundtrip_value: str | None = None
    else:
        roundtrip_value = raw

    sim1 = value_similarity(roundtrip_value, pass1_value)
    sim2 = value_similarity(roundtrip_value, pass2_value)

    # Decision rule:
    #   * If both >= 0.95 (i.e. both passes agreed), the round-trip
    #     ratifies pass1 (pick is arbitrary; reconciler step 1 already
    #     handled exact agreement before we got here).
    #   * If sim1 - sim2 >= 0.1 → pass1 wins.
    #   * If sim2 - sim1 >= 0.1 → pass2 wins.
    #   * Otherwise → neither (low confidence).
    delta = sim1 - sim2
    if max(sim1, sim2) < 0.7:
        winner = "neither"
    elif delta >= 0.1:
        winner = "pass1"
    elif -delta >= 0.1:
        winner = "pass2"
    else:
        winner = "neither"

    logger.debug(
        "bbox_roundtrip_decision",
        field=field_name,
        roundtrip_value=roundtrip_value,
        sim_pass1=sim1,
        sim_pass2=sim2,
        winner=winner,
    )

    return BboxRoundtripResult(
        value=roundtrip_value,
        confidence=0.85 if winner != "neither" else 0.4,
        similarity_to_pass1=sim1,
        similarity_to_pass2=sim2,
        winning_pass=winner,
        crop_size_px=crop_size,
        backend_name=backend.name,
    )
