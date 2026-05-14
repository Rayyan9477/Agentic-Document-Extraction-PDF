"""
V3 Phase 2 — HeterogeneousReconciler.

Fuses Pass 1 (EXTRACTOR / primary VLM) and Pass 2 (AUDITOR / secondary
VLM) outputs field-by-field. The 5-step tiebreaker is the heart of
the dual-VLM extraction guarantee: when two different model families
agree, the agreement carries real signal because their failure modes
are orthogonal. When they disagree, the tiebreaker order tries the
informative tests first (visual ground truth via bbox-overlap, then
focused crop re-read) and falls back to soft signals (pattern-match,
field-history) only when the visual evidence is exhausted.

The reconciler is **deterministic and pure** — given the same Pass 1
output, Pass 2 output, image bytes, mode set, and FAISS state, it
produces the same merged extraction. That property is what lets us
test it exhaustively against synthetic disagreements.

Order of operations per disputed field:

1. **Exact match.** Pass 1 == Pass 2 (string-equal or numeric within
   ``1e-4`` of magnitude). Boost confidence; emit Pass 1's value.
2. **Bbox-overlap test.** Does Pass 1's value lie inside Pass 2's
   reported bbox region (IoU >= ``reconciler_bbox_iou_threshold``)?
   Pass 1 wins — Pass 2's bbox is the visual ground truth.
3. **Bbox round-trip.** Crop the page to the bbox + padding, re-query
   the secondary VLM, compare. The pass whose value matches the crop
   read wins. (Phase 2 wires this; Phase 3's Critic also uses it.)
4. **Pattern detector.** If one value matches a known hallucination
   pattern (placeholder, suspiciously round amount, sequential
   digits) and the other doesn't, drop the flagged value.
5. **Field-history match.** FAISS lookup over historical extractions
   for ``(profile, doc_type, field_name)``. Match wins on similarity
   >= ``reconciler_history_similarity_threshold``. Best-effort —
   FAISS may be empty for new tenants.

If none of the above resolves the dispute, the field is marked
``low_confidence`` with both candidates preserved in the metadata so
the Critic agent (Phase 3) can take another swing.

Mode-aware weighting (``RECONCILER_WEIGHTS_BY_MODE``) shifts which
pass we trust **before** the tiebreaker runs. On fax input, Pass 2's
text precision matters more for numeric fields; on handwritten, Pass
1's vision tower wins on cursive content. Weights are knobs in the
weight table, not new code paths.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Any

from src.client.backends.protocol import VLMBackend, VLMRole
from src.config import get_logger, get_settings


logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Mode-aware weighting
# ---------------------------------------------------------------------------


# Per-modality (pass1, pass2) trust weights for {numeric, text} fields. The
# reconciler uses these to bias confidence boosts when the passes agree, and
# to break ties when neither bbox-grounding nor history can decide.
RECONCILER_WEIGHTS_BY_MODE: dict[str, dict[str, tuple[float, float]]] = {
    "fax": {"numeric": (0.3, 0.7), "text": (0.5, 0.5)},
    "handwritten": {"numeric": (0.5, 0.5), "text": (0.7, 0.3)},
    "printed": {"numeric": (0.5, 0.5), "text": (0.5, 0.5)},
    "table": {"numeric": (0.5, 0.5), "text": (0.5, 0.5)},
    "form": {"numeric": (0.5, 0.5), "text": (0.5, 0.5)},
    "visual": {"numeric": (0.4, 0.6), "text": (0.6, 0.4)},
}

DEFAULT_MODE_WEIGHTS = (0.5, 0.5)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class ReconciledField:
    """One field's reconciliation outcome."""

    field_name: str
    value: Any
    confidence: float
    bbox: list[float] | None
    source_pass: str
    """Which pass(es) the winning value came from: ``pass1``, ``pass2``,
    ``both`` (exact agreement), ``roundtrip`` (round-trip ratified),
    ``history``, or ``low_confidence``."""

    tiebreaker: str | None
    """Which tiebreaker step decided this field, or ``None`` for
    exact-match cases. One of: ``exact_match``, ``bbox_overlap``,
    ``bbox_roundtrip``, ``pattern_detector``, ``field_history``,
    ``low_confidence``."""

    pass1_candidate: Any = None
    pass2_candidate: Any = None


@dataclass(slots=True)
class ReconciliationReport:
    """Summary of one document-level reconciliation pass."""

    fields: dict[str, ReconciledField] = dc_field(default_factory=dict)
    agreement_rate: float = 0.0
    """Fraction of fields where Pass 1 == Pass 2 exactly (step 1 wins)."""

    disagreement_count: int = 0
    tiebreakers_used: dict[str, int] = dc_field(default_factory=dict)
    """Per-tiebreaker counts. Keys mirror ``ReconciledField.tiebreaker``."""


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


_PLACEHOLDER_VALUES = {
    "n/a", "na", "tbd", "xxx", "xxxx", "xxxxx",
    "unknown", "none", "null", "test", "sample",
    "123", "1234", "12345", "123456",
}


def _is_placeholder(value: Any) -> bool:
    """Cheap pattern detector for the reconciler's tier-4 tiebreaker.

    Mirrors a subset of ``src/validation/pattern_detector.py``. Kept
    inline to keep the reconciler dependency-light; the full detector
    runs downstream in the validator.
    """
    if value is None:
        return False
    s = str(value).strip().lower()
    if s in _PLACEHOLDER_VALUES:
        return True
    # Sequential digits like "12345"
    if s.isdigit() and len(s) >= 3:
        digits = [int(c) for c in s]
        deltas = {b - a for a, b in zip(digits, digits[1:], strict=False)}
        if deltas == {1} or deltas == {-1}:
            return True
    return False


def _values_agree(a: Any, b: Any, *, numeric_tol: float = 1e-4) -> bool:
    """Exact-match check for tier-1 tiebreaker.

    Numeric values agree within ``1e-4`` of magnitude (matches the
    legacy ``BaseAgent._values_match`` semantics tightened in Phase 0).
    Strings are case-insensitive, whitespace-stripped.
    """
    if a is None or b is None:
        return a is None and b is None
    if a == b:
        return True
    try:
        fa, fb = float(a), float(b)
        magnitude = max(abs(fa), abs(fb), 1.0)
        return abs(fa - fb) < numeric_tol * magnitude
    except (TypeError, ValueError):
        pass
    return str(a).strip().lower() == str(b).strip().lower()


def _bbox_iou(
    a: list[float] | tuple[float, ...] | None,
    b: list[float] | tuple[float, ...] | None,
) -> float:
    """IoU over normalised ``(x1, y1, x2, y2)`` bboxes. Returns 0.0 if
    either side is missing or degenerate."""
    if not a or not b or len(a) < 4 or len(b) < 4:
        return 0.0
    ax1, ay1, ax2, ay2 = a[0], a[1], a[2], a[3]
    bx1, by1, bx2, by2 = b[0], b[1], b[2], b[3]
    if ax2 <= ax1 or ay2 <= ay1 or bx2 <= bx1 or by2 <= by1:
        return 0.0
    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    a_area = (ax2 - ax1) * (ay2 - ay1)
    b_area = (bx2 - bx1) * (by2 - by1)
    union = a_area + b_area - inter
    if union <= 0.0:
        return 0.0
    return inter / union


def _is_numeric_field(field_name: str, hint: str | None = None) -> bool:
    """Heuristic: numeric fields by name suffix or explicit hint."""
    if hint and hint in {"NUMBER", "INTEGER", "FLOAT", "CURRENCY", "AMOUNT"}:
        return True
    n = field_name.lower()
    return any(
        token in n
        for token in ("amount", "total", "charge", "balance", "qty",
                      "quantity", "count", "rate", "fee")
    )


# ---------------------------------------------------------------------------
# Reconciler
# ---------------------------------------------------------------------------


class HeterogeneousReconciler:
    """Fuse Pass 1 + Pass 2 outputs into a single reconciled extraction.

    The reconciler is constructed per-document; it carries no
    cross-document state. Lookups against FAISS or the bbox-roundtrip
    helper are passed in as collaborators so unit tests can stub them.
    """

    def __init__(
        self,
        *,
        backend: VLMBackend | None = None,
        bbox_iou_threshold: float | None = None,
        history_similarity_threshold: float | None = None,
        history_lookup: Any = None,
        roundtrip_helper: Any = None,
    ) -> None:
        settings = get_settings()
        self._backend = backend
        self._bbox_iou_threshold = (
            bbox_iou_threshold
            if bbox_iou_threshold is not None
            else settings.extraction.reconciler_bbox_iou_threshold
        )
        self._history_threshold = (
            history_similarity_threshold
            if history_similarity_threshold is not None
            else settings.extraction.reconciler_history_similarity_threshold
        )
        # Optional FAISS-backed history; ``history_lookup(field_name,
        # candidate_value, profile, doc_type)`` returns a similarity in
        # [0, 1] for the closest historical match, or 0.0 if absent.
        self._history_lookup = history_lookup
        # Optional bbox-roundtrip helper; ``roundtrip(...)`` returns a
        # ``BboxRoundtripResult``. Allows stubbing in unit tests.
        self._roundtrip_helper = roundtrip_helper

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def reconcile(
        self,
        *,
        pass1_fields: dict[str, dict[str, Any]],
        pass2_fields: dict[str, dict[str, Any]],
        page_image_data: str | None = None,
        modalities: list[str] | None = None,
        profile: str = "generic-document",
        doc_type: str = "UNKNOWN",
    ) -> ReconciliationReport:
        """Reconcile per-field records from two VLM passes.

        Args:
            pass1_fields: ``{field_name: {value, confidence, bbox?}}``
                from Pass 1 (EXTRACTOR / primary VLM).
            pass2_fields: ``{field_name: {value, confidence, bbox}}``
                from Pass 2 (AUDITOR / secondary VLM). Bbox is mandated.
            page_image_data: full-page image (data URI or raw base64)
                for bbox round-trip. ``None`` skips step 3 — the
                reconciler falls through to pattern + history.
            modalities: detected modality labels (``fax``, ...).
            profile: active profile name (used for history lookup key).
            doc_type: document type (used for history lookup key).

        Returns:
            ``ReconciliationReport`` with per-field outcomes and aggregate
            counts. The reconciled extraction is in
            ``report.fields[name].value``.
        """
        modalities = modalities or []
        all_fields = set(pass1_fields) | set(pass2_fields)
        report = ReconciliationReport()

        for name in sorted(all_fields):
            p1 = pass1_fields.get(name) or {}
            p2 = pass2_fields.get(name) or {}
            outcome = self._reconcile_field(
                name=name,
                p1=p1,
                p2=p2,
                page_image_data=page_image_data,
                modalities=modalities,
                profile=profile,
                doc_type=doc_type,
            )
            report.fields[name] = outcome
            if outcome.tiebreaker is None:
                # exact match counts toward agreement
                continue
            if outcome.tiebreaker == "exact_match":
                continue
            # V3 Phase 8 — ``single_pass`` means one pass had the
            # field and the other was silent. That's a coverage
            # gap, not a disagreement; don't inflate the
            # disagreement counter or treat it as a tiebreaker.
            if outcome.tiebreaker == "single_pass":
                continue
            report.disagreement_count += 1
            counts = report.tiebreakers_used
            counts[outcome.tiebreaker] = counts.get(outcome.tiebreaker, 0) + 1

        if all_fields:
            agreed = sum(
                1
                for f in report.fields.values()
                if f.tiebreaker
                in ("exact_match", "single_pass")
                or f.tiebreaker is None
            )
            report.agreement_rate = agreed / len(all_fields)
        return report

    # ------------------------------------------------------------------
    # Per-field tiebreaker
    # ------------------------------------------------------------------

    def _reconcile_field(
        self,
        *,
        name: str,
        p1: dict[str, Any],
        p2: dict[str, Any],
        page_image_data: str | None,
        modalities: list[str],
        profile: str,
        doc_type: str,
    ) -> ReconciledField:
        v1 = p1.get("value")
        v2 = p2.get("value")
        c1 = float(p1.get("confidence") or 0.0)
        c2 = float(p2.get("confidence") or 0.0)
        bbox1 = p1.get("bbox")
        bbox2 = p2.get("bbox")

        # ---- 0. V3 Phase 8 — Single-pass coverage gap ----
        # When one pass returned the field and the other left it
        # blank (different field-coverage between extractor models is
        # common), this is NOT a disagreement. Treat it as
        # uncontested coverage from the present pass: native
        # confidence (no halving) and ``tiebreaker="single_pass"``
        # so the orchestrator's disagreement_count doesn't inflate.
        # Pre-Phase-8 behaviour: every one-pass-only field fell
        # through to the low-confidence tier and got its confidence
        # halved as ``low_confidence``.
        v1_present = v1 is not None and v1 != ""
        v2_present = v2 is not None and v2 != ""
        if v1_present and not v2_present:
            return ReconciledField(
                field_name=name,
                value=v1,
                confidence=c1,
                bbox=bbox1,
                source_pass="pass1",
                tiebreaker="single_pass",
                pass1_candidate=v1,
                pass2_candidate=None,
            )
        if v2_present and not v1_present:
            return ReconciledField(
                field_name=name,
                value=v2,
                confidence=c2,
                bbox=bbox2,
                source_pass="pass2",
                tiebreaker="single_pass",
                pass1_candidate=None,
                pass2_candidate=v2,
            )

        # ---- 1. Exact match ----
        if _values_agree(v1, v2):
            return ReconciledField(
                field_name=name,
                value=v1 if v1 is not None else v2,
                confidence=min(1.0, max(c1, c2) + 0.05),
                bbox=bbox2 or bbox1,
                source_pass="both",
                tiebreaker="exact_match",
                pass1_candidate=v1,
                pass2_candidate=v2,
            )

        # ---- 2. Bbox-overlap ----
        # Pass 1 wins iff its bbox lies inside Pass 2's bbox region by
        # IoU. Without a Pass 1 bbox we fall through; the AUDITOR
        # contract guarantees Pass 2 has a bbox whenever value is set.
        iou = _bbox_iou(bbox1, bbox2)
        if iou >= self._bbox_iou_threshold and v1 is not None and v2 is not None:
            return ReconciledField(
                field_name=name,
                value=v1,
                confidence=max(c1, c2) * 0.95,
                bbox=bbox2 or bbox1,
                source_pass="pass1",
                tiebreaker="bbox_overlap",
                pass1_candidate=v1,
                pass2_candidate=v2,
            )

        # ---- 3. Bbox round-trip ----
        # Re-read the cropped region with the secondary VLM. Prefers
        # whichever pass the round-trip ratifies.
        rt = None
        if (
            self._roundtrip_helper is not None
            and page_image_data is not None
            and bbox2 is not None
        ):
            try:
                rt = self._roundtrip_helper(
                    backend=self._backend,
                    image_data_uri=page_image_data,
                    bbox=bbox2,
                    pass1_value=v1,
                    pass2_value=v2,
                    field_name=name,
                    role=VLMRole.SECONDARY,
                )
            except Exception as exc:  # pragma: no cover - defensive
                logger.warning(
                    "reconciler_roundtrip_call_failed",
                    field=name,
                    error=str(exc),
                )
        if rt is not None and rt.winning_pass in ("pass1", "pass2"):
            winning_value = v1 if rt.winning_pass == "pass1" else v2
            return ReconciledField(
                field_name=name,
                value=winning_value,
                confidence=rt.confidence,
                bbox=bbox2,
                source_pass="roundtrip",
                tiebreaker="bbox_roundtrip",
                pass1_candidate=v1,
                pass2_candidate=v2,
            )

        # ---- 4. Pattern detector ----
        p1_bad = _is_placeholder(v1)
        p2_bad = _is_placeholder(v2)
        if p1_bad and not p2_bad:
            return ReconciledField(
                field_name=name,
                value=v2,
                confidence=c2 * 0.9,
                bbox=bbox2,
                source_pass="pass2",
                tiebreaker="pattern_detector",
                pass1_candidate=v1,
                pass2_candidate=v2,
            )
        if p2_bad and not p1_bad:
            return ReconciledField(
                field_name=name,
                value=v1,
                confidence=c1 * 0.9,
                bbox=bbox1,
                source_pass="pass1",
                tiebreaker="pattern_detector",
                pass1_candidate=v1,
                pass2_candidate=v2,
            )

        # ---- 5. Field-history match ----
        if self._history_lookup is not None:
            try:
                sim1 = float(self._history_lookup(name, v1, profile, doc_type))
            except Exception:
                sim1 = 0.0
            try:
                sim2 = float(self._history_lookup(name, v2, profile, doc_type))
            except Exception:
                sim2 = 0.0
            if max(sim1, sim2) >= self._history_threshold:
                if sim1 >= sim2:
                    return ReconciledField(
                        field_name=name,
                        value=v1,
                        confidence=c1 * 0.9,
                        bbox=bbox1 or bbox2,
                        source_pass="pass1",
                        tiebreaker="field_history",
                        pass1_candidate=v1,
                        pass2_candidate=v2,
                    )
                return ReconciledField(
                    field_name=name,
                    value=v2,
                    confidence=c2 * 0.9,
                    bbox=bbox2 or bbox1,
                    source_pass="pass2",
                    tiebreaker="field_history",
                    pass1_candidate=v1,
                    pass2_candidate=v2,
                )

        # ---- Last resort: low_confidence ----
        # Apply mode-aware weighting to choose which pass to prefer when
        # everything else is exhausted. Numeric/text bias only matters at
        # the margin; the field is flagged either way.
        mode = next(
            (m for m in modalities if m in RECONCILER_WEIGHTS_BY_MODE),
            "printed",
        )
        kind = "numeric" if _is_numeric_field(name) else "text"
        w1, w2 = RECONCILER_WEIGHTS_BY_MODE.get(mode, {}).get(
            kind, DEFAULT_MODE_WEIGHTS
        )
        winning_pass = "pass1" if (c1 * w1) >= (c2 * w2) else "pass2"
        winning_value = v1 if winning_pass == "pass1" else v2
        winning_bbox = bbox1 if winning_pass == "pass1" else bbox2
        winning_conf = (c1 if winning_pass == "pass1" else c2) * 0.5
        return ReconciledField(
            field_name=name,
            value=winning_value,
            confidence=winning_conf,
            bbox=winning_bbox or bbox2,
            source_pass=winning_pass,
            tiebreaker="low_confidence",
            pass1_candidate=v1,
            pass2_candidate=v2,
        )
