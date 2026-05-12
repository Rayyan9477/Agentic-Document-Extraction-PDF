"""
Phase 2 — HeterogeneousReconciler unit tests.

Each test exercises one tier of the 5-step tiebreaker so a regression
in any tier surfaces in isolation:

* Tier 1 — exact match (string-equal or numeric-within-tolerance).
* Tier 2 — bbox-overlap (Pass 1 inside Pass 2's bbox region by IoU).
* Tier 3 — bbox round-trip (focused crop re-read ratifies one pass).
* Tier 4 — pattern detector (one is a hallucination pattern).
* Tier 5 — field-history match (FAISS lookup).
* Last resort — low_confidence with mode-aware weighting.

Plus aggregate-report tests (agreement_rate, tiebreaker counts) and
mode-weighting tests.

These tests are pure; no live VLM, no FAISS, no PIL. Round-trip helper
and history lookup are stubbed via callables.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.agents.reconciler import (
    DEFAULT_MODE_WEIGHTS,
    HeterogeneousReconciler,
    RECONCILER_WEIGHTS_BY_MODE,
    ReconciledField,
    ReconciliationReport,
    _bbox_iou,
    _is_placeholder,
    _values_agree,
)


# ---------------------------------------------------------------------------
# Stand-ins
# ---------------------------------------------------------------------------


@dataclass
class _StubRoundtrip:
    """Lightweight stand-in for ``BboxRoundtripResult``."""

    winning_pass: str
    confidence: float = 0.85
    similarity_to_pass1: float = 0.0
    similarity_to_pass2: float = 0.0


def _stub_roundtrip_helper(winner: str = "neither", confidence: float = 0.85):
    """Build a callable matching ``perform_bbox_roundtrip`` signature."""

    def _call(**kwargs: Any) -> _StubRoundtrip:
        return _StubRoundtrip(winning_pass=winner, confidence=confidence)

    return _call


def _field(value: Any, conf: float, bbox: list[float] | None = None) -> dict[str, Any]:
    return {"value": value, "confidence": conf, "bbox": bbox}


# ---------------------------------------------------------------------------
# Helpers (pure functions in reconciler module)
# ---------------------------------------------------------------------------


class TestPureHelpers:
    def test_values_agree_exact_string(self) -> None:
        assert _values_agree("foo", "foo") is True

    def test_values_agree_case_insensitive(self) -> None:
        assert _values_agree("FOO", "foo") is True

    def test_values_agree_numeric_within_tol(self) -> None:
        assert _values_agree(100.0, 100.00001) is True

    def test_values_agree_numeric_outside_tol(self) -> None:
        assert _values_agree(100.0, 101.0) is False

    def test_values_agree_both_none(self) -> None:
        assert _values_agree(None, None) is True

    def test_values_agree_one_none(self) -> None:
        assert _values_agree(None, "x") is False

    def test_bbox_iou_perfect_overlap(self) -> None:
        b = [0.1, 0.1, 0.3, 0.3]
        assert _bbox_iou(b, b) == pytest.approx(1.0)

    def test_bbox_iou_no_overlap(self) -> None:
        a = [0.0, 0.0, 0.1, 0.1]
        b = [0.5, 0.5, 0.6, 0.6]
        assert _bbox_iou(a, b) == 0.0

    def test_bbox_iou_partial(self) -> None:
        a = [0.0, 0.0, 0.4, 0.4]  # area = 0.16
        b = [0.2, 0.2, 0.6, 0.6]  # area = 0.16
        # intersection = 0.04; union = 0.16 + 0.16 - 0.04 = 0.28; IoU = 1/7
        assert _bbox_iou(a, b) == pytest.approx(0.04 / 0.28)

    def test_bbox_iou_none_returns_zero(self) -> None:
        assert _bbox_iou(None, [0, 0, 1, 1]) == 0.0
        assert _bbox_iou([0, 0, 1, 1], None) == 0.0

    def test_bbox_iou_degenerate_zero(self) -> None:
        assert _bbox_iou([0, 0, 0, 1], [0, 0, 1, 1]) == 0.0  # degenerate width

    def test_is_placeholder_known_strings(self) -> None:
        for val in ["N/A", "TBD", "XXX", "unknown", "test"]:
            assert _is_placeholder(val) is True, f"expected placeholder for {val!r}"

    def test_is_placeholder_sequential_digits(self) -> None:
        assert _is_placeholder("12345") is True
        assert _is_placeholder("54321") is True

    def test_is_placeholder_real_value(self) -> None:
        assert _is_placeholder("Alice Smith") is False
        assert _is_placeholder("99213") is False  # non-sequential

    def test_is_placeholder_none(self) -> None:
        assert _is_placeholder(None) is False


# ---------------------------------------------------------------------------
# Tier 1 — exact match
# ---------------------------------------------------------------------------


class TestTier1ExactMatch:
    def test_exact_string_match_emits_both_source(self) -> None:
        r = HeterogeneousReconciler()
        report = r.reconcile(
            pass1_fields={"name": _field("Alice", 0.8)},
            pass2_fields={"name": _field("Alice", 0.7, [0, 0, 0.1, 0.1])},
        )
        f = report.fields["name"]
        assert f.value == "Alice"
        assert f.source_pass == "both"
        assert f.tiebreaker == "exact_match"
        # Confidence boost: max(c1, c2) + 0.05
        assert f.confidence == pytest.approx(0.85)

    def test_numeric_within_tolerance_counts_as_match(self) -> None:
        r = HeterogeneousReconciler()
        report = r.reconcile(
            pass1_fields={"amt": _field(100.0, 0.9)},
            pass2_fields={"amt": _field(100.00001, 0.9, [0, 0, 0.1, 0.1])},
        )
        assert report.fields["amt"].tiebreaker == "exact_match"

    def test_both_none_counts_as_agreement(self) -> None:
        r = HeterogeneousReconciler()
        report = r.reconcile(
            pass1_fields={"x": _field(None, 0.0)},
            pass2_fields={"x": _field(None, 0.0)},
        )
        assert report.fields["x"].tiebreaker == "exact_match"
        assert report.fields["x"].value is None


# ---------------------------------------------------------------------------
# Tier 2 — bbox overlap
# ---------------------------------------------------------------------------


class TestTier2BboxOverlap:
    def test_high_iou_pass1_wins(self) -> None:
        r = HeterogeneousReconciler(bbox_iou_threshold=0.4)
        report = r.reconcile(
            pass1_fields={
                "amt": _field("100", 0.7, [0.1, 0.1, 0.4, 0.4]),
            },
            pass2_fields={
                "amt": _field("1OO", 0.6, [0.1, 0.1, 0.4, 0.4]),  # OCR error
            },
        )
        f = report.fields["amt"]
        assert f.value == "100"
        assert f.source_pass == "pass1"
        assert f.tiebreaker == "bbox_overlap"

    def test_low_iou_falls_through(self) -> None:
        r = HeterogeneousReconciler(
            bbox_iou_threshold=0.4,
            roundtrip_helper=_stub_roundtrip_helper("neither"),
        )
        report = r.reconcile(
            pass1_fields={"amt": _field("100", 0.7, [0.0, 0.0, 0.1, 0.1])},
            pass2_fields={"amt": _field("200", 0.6, [0.5, 0.5, 0.6, 0.6])},
            page_image_data="data:image/png;base64,AAAA",
        )
        # Neither tier 1 nor tier 2 fires; falls through.
        assert report.fields["amt"].tiebreaker != "bbox_overlap"
        assert report.fields["amt"].tiebreaker != "exact_match"

    def test_no_pass1_bbox_skips_tier_2(self) -> None:
        # Pass 1 lacks a bbox so IoU is 0; tier 2 cannot fire.
        r = HeterogeneousReconciler(bbox_iou_threshold=0.4)
        report = r.reconcile(
            pass1_fields={"amt": _field("100", 0.7)},  # no bbox
            pass2_fields={"amt": _field("200", 0.6, [0.1, 0.1, 0.4, 0.4])},
        )
        assert report.fields["amt"].tiebreaker != "bbox_overlap"


# ---------------------------------------------------------------------------
# Tier 3 — bbox round-trip
# ---------------------------------------------------------------------------


class TestTier3BboxRoundtrip:
    # Bboxes that don't overlap → tier 2 cannot fire → tier 3 tested in isolation.
    _PASS1_BOX = [0.0, 0.0, 0.1, 0.1]
    _PASS2_BOX = [0.5, 0.5, 0.6, 0.6]

    def test_roundtrip_ratifies_pass1(self) -> None:
        r = HeterogeneousReconciler(
            bbox_iou_threshold=0.4,
            roundtrip_helper=_stub_roundtrip_helper("pass1", confidence=0.92),
        )
        report = r.reconcile(
            pass1_fields={"x": _field("A", 0.5, self._PASS1_BOX)},
            pass2_fields={"x": _field("B", 0.5, self._PASS2_BOX)},
            page_image_data="data:image/png;base64,AAAA",
        )
        f = report.fields["x"]
        assert f.value == "A"
        assert f.source_pass == "roundtrip"
        assert f.tiebreaker == "bbox_roundtrip"
        assert f.confidence == pytest.approx(0.92)

    def test_roundtrip_ratifies_pass2(self) -> None:
        r = HeterogeneousReconciler(
            bbox_iou_threshold=0.4,
            roundtrip_helper=_stub_roundtrip_helper("pass2"),
        )
        report = r.reconcile(
            pass1_fields={"x": _field("A", 0.5, self._PASS1_BOX)},
            pass2_fields={"x": _field("B", 0.5, self._PASS2_BOX)},
            page_image_data="data:image/png;base64,AAAA",
        )
        assert report.fields["x"].value == "B"
        assert report.fields["x"].source_pass == "roundtrip"

    def test_roundtrip_neither_falls_through(self) -> None:
        r = HeterogeneousReconciler(
            bbox_iou_threshold=0.4,
            roundtrip_helper=_stub_roundtrip_helper("neither"),
        )
        report = r.reconcile(
            pass1_fields={"x": _field("A", 0.4, self._PASS1_BOX)},
            pass2_fields={"x": _field("B", 0.4, self._PASS2_BOX)},
            page_image_data="data:image/png;base64,AAAA",
        )
        # Falls to either pattern, history, or low_confidence
        assert report.fields["x"].tiebreaker not in ("bbox_roundtrip", "exact_match")

    def test_roundtrip_skipped_without_image_data(self) -> None:
        called = False

        def _helper(**_: Any) -> _StubRoundtrip:
            nonlocal called
            called = True
            return _StubRoundtrip(winning_pass="pass1")

        r = HeterogeneousReconciler(
            bbox_iou_threshold=0.4,
            roundtrip_helper=_helper,
        )
        r.reconcile(
            pass1_fields={"x": _field("A", 0.4, self._PASS1_BOX)},
            pass2_fields={"x": _field("B", 0.4, self._PASS2_BOX)},
            page_image_data=None,
        )
        assert called is False, "roundtrip should not run without image data"

    def test_roundtrip_helper_exception_is_swallowed(self) -> None:
        def _bad_helper(**_: Any) -> Any:
            raise RuntimeError("vlm down")

        r = HeterogeneousReconciler(
            bbox_iou_threshold=0.4,
            roundtrip_helper=_bad_helper,
        )
        # Should not propagate; fall through to pattern/history/low_conf.
        report = r.reconcile(
            pass1_fields={"x": _field("A", 0.4, self._PASS1_BOX)},
            pass2_fields={"x": _field("B", 0.4, self._PASS2_BOX)},
            page_image_data="data:image/png;base64,AAAA",
        )
        assert report.fields["x"].tiebreaker != "bbox_roundtrip"


# ---------------------------------------------------------------------------
# Tier 4 — pattern detector
# ---------------------------------------------------------------------------


class TestTier4PatternDetector:
    def test_pass1_placeholder_drops_pass1(self) -> None:
        r = HeterogeneousReconciler(bbox_iou_threshold=0.99)
        report = r.reconcile(
            pass1_fields={"name": _field("N/A", 0.5)},
            pass2_fields={"name": _field("Alice", 0.5)},
        )
        f = report.fields["name"]
        assert f.value == "Alice"
        assert f.source_pass == "pass2"
        assert f.tiebreaker == "pattern_detector"

    def test_pass2_placeholder_drops_pass2(self) -> None:
        r = HeterogeneousReconciler(bbox_iou_threshold=0.99)
        report = r.reconcile(
            pass1_fields={"name": _field("Alice", 0.5)},
            pass2_fields={"name": _field("XXX", 0.5)},
        )
        assert report.fields["name"].value == "Alice"
        assert report.fields["name"].source_pass == "pass1"
        assert report.fields["name"].tiebreaker == "pattern_detector"

    def test_both_placeholders_falls_through(self) -> None:
        r = HeterogeneousReconciler(bbox_iou_threshold=0.99)
        report = r.reconcile(
            pass1_fields={"name": _field("N/A", 0.5)},
            pass2_fields={"name": _field("XXX", 0.5)},
        )
        # Neither dominates; tier 4 cannot decide.
        assert report.fields["name"].tiebreaker != "pattern_detector"


# ---------------------------------------------------------------------------
# Tier 5 — field-history match (FAISS proxy)
# ---------------------------------------------------------------------------


class TestTier5FieldHistory:
    def test_pass1_history_match_wins(self) -> None:
        # history_lookup returns 0.9 for pass1's value, 0.0 for pass2's.
        def _hist(name: str, value: Any, profile: str, doc_type: str) -> float:
            return 0.9 if value == "Alice" else 0.0

        r = HeterogeneousReconciler(
            bbox_iou_threshold=0.99,
            history_similarity_threshold=0.88,
            history_lookup=_hist,
        )
        report = r.reconcile(
            pass1_fields={"name": _field("Alice", 0.5)},
            pass2_fields={"name": _field("Alyce", 0.5)},
        )
        f = report.fields["name"]
        assert f.value == "Alice"
        assert f.source_pass == "pass1"
        assert f.tiebreaker == "field_history"

    def test_pass2_history_match_wins(self) -> None:
        def _hist(name: str, value: Any, profile: str, doc_type: str) -> float:
            return 0.92 if value == "Alyce" else 0.0

        r = HeterogeneousReconciler(
            bbox_iou_threshold=0.99,
            history_similarity_threshold=0.88,
            history_lookup=_hist,
        )
        report = r.reconcile(
            pass1_fields={"name": _field("Alice", 0.5)},
            pass2_fields={"name": _field("Alyce", 0.5)},
        )
        assert report.fields["name"].value == "Alyce"
        assert report.fields["name"].source_pass == "pass2"
        assert report.fields["name"].tiebreaker == "field_history"

    def test_history_below_threshold_falls_through(self) -> None:
        def _hist(*_: Any) -> float:
            return 0.5  # below 0.88

        r = HeterogeneousReconciler(
            bbox_iou_threshold=0.99,
            history_similarity_threshold=0.88,
            history_lookup=_hist,
        )
        report = r.reconcile(
            pass1_fields={"name": _field("Alice", 0.5)},
            pass2_fields={"name": _field("Alyce", 0.5)},
        )
        assert report.fields["name"].tiebreaker != "field_history"

    def test_history_lookup_exception_does_not_propagate(self) -> None:
        def _bad(*_: Any) -> float:
            raise RuntimeError("faiss exploded")

        r = HeterogeneousReconciler(
            bbox_iou_threshold=0.99,
            history_lookup=_bad,
        )
        # Should fall through to low_confidence, not raise.
        report = r.reconcile(
            pass1_fields={"name": _field("A", 0.5)},
            pass2_fields={"name": _field("B", 0.5)},
        )
        assert report.fields["name"].tiebreaker == "low_confidence"


# ---------------------------------------------------------------------------
# Last resort — low_confidence + mode-weighted choice
# ---------------------------------------------------------------------------


class TestLowConfidenceFallback:
    def test_marks_low_confidence(self) -> None:
        r = HeterogeneousReconciler(bbox_iou_threshold=0.99)
        report = r.reconcile(
            pass1_fields={"x": _field("foo", 0.5)},
            pass2_fields={"x": _field("bar", 0.5)},
        )
        assert report.fields["x"].tiebreaker == "low_confidence"

    def test_fax_mode_prefers_pass2_for_numeric(self) -> None:
        r = HeterogeneousReconciler(bbox_iou_threshold=0.99)
        report = r.reconcile(
            pass1_fields={"total_amount": _field("100", 0.5)},
            pass2_fields={"total_amount": _field("110", 0.5)},
            modalities=["fax"],
        )
        # fax weights numeric (0.3, 0.7) so pass2 wins
        assert report.fields["total_amount"].source_pass == "pass2"

    def test_handwritten_prefers_pass1_for_text(self) -> None:
        r = HeterogeneousReconciler(bbox_iou_threshold=0.99)
        report = r.reconcile(
            pass1_fields={"name": _field("Alice", 0.5)},
            pass2_fields={"name": _field("Alyce", 0.5)},
            modalities=["handwritten"],
        )
        # handwritten weights text (0.7, 0.3) so pass1 wins
        assert report.fields["name"].source_pass == "pass1"

    def test_unknown_modality_uses_default_weights(self) -> None:
        r = HeterogeneousReconciler(bbox_iou_threshold=0.99)
        report = r.reconcile(
            pass1_fields={"x": _field("a", 0.51)},  # higher conf
            pass2_fields={"x": _field("b", 0.49)},
            modalities=["entirely_unknown_mode"],
        )
        # Default weights are (0.5, 0.5); pass1's conf wins by 0.02
        assert report.fields["x"].source_pass == "pass1"


# ---------------------------------------------------------------------------
# Aggregate report
# ---------------------------------------------------------------------------


class TestAggregateReport:
    def test_agreement_rate_all_match(self) -> None:
        r = HeterogeneousReconciler()
        report = r.reconcile(
            pass1_fields={
                "a": _field("x", 0.9),
                "b": _field("y", 0.9),
            },
            pass2_fields={
                "a": _field("x", 0.9),
                "b": _field("y", 0.9),
            },
        )
        assert report.agreement_rate == pytest.approx(1.0)
        assert report.disagreement_count == 0

    def test_agreement_rate_partial(self) -> None:
        r = HeterogeneousReconciler(bbox_iou_threshold=0.99)
        report = r.reconcile(
            pass1_fields={
                "a": _field("x", 0.9),
                "b": _field("y", 0.9),
                "c": _field("z", 0.9),
            },
            pass2_fields={
                "a": _field("x", 0.9),  # match
                "b": _field("DIFFERENT", 0.9),  # disagree
                "c": _field("z", 0.9),  # match
            },
        )
        assert report.agreement_rate == pytest.approx(2 / 3)
        assert report.disagreement_count == 1

    def test_tiebreakers_used_counted(self) -> None:
        r = HeterogeneousReconciler(bbox_iou_threshold=0.99)
        report = r.reconcile(
            pass1_fields={
                "a": _field("N/A", 0.5),  # placeholder → pattern_detector
                "b": _field("foo", 0.5),  # disagree → low_confidence
            },
            pass2_fields={
                "a": _field("Alice", 0.5),
                "b": _field("bar", 0.5),
            },
        )
        assert report.tiebreakers_used.get("pattern_detector") == 1
        assert report.tiebreakers_used.get("low_confidence") == 1
        assert report.disagreement_count == 2

    def test_field_only_in_pass1(self) -> None:
        r = HeterogeneousReconciler(bbox_iou_threshold=0.99)
        report = r.reconcile(
            pass1_fields={"a": _field("x", 0.9)},
            pass2_fields={},
        )
        # Pass 2 has no value; pass 1 alone — falls into low_confidence
        # because there's no agreement and no bbox/pattern signal.
        assert "a" in report.fields

    def test_empty_inputs(self) -> None:
        r = HeterogeneousReconciler()
        report = r.reconcile(pass1_fields={}, pass2_fields={})
        assert report.fields == {}
        assert report.agreement_rate == 0.0


# ---------------------------------------------------------------------------
# Mode-weight constants are sensible
# ---------------------------------------------------------------------------


class TestModeWeights:
    def test_fax_numeric_biases_pass2(self) -> None:
        w1, w2 = RECONCILER_WEIGHTS_BY_MODE["fax"]["numeric"]
        assert w2 > w1

    def test_handwritten_text_biases_pass1(self) -> None:
        w1, w2 = RECONCILER_WEIGHTS_BY_MODE["handwritten"]["text"]
        assert w1 > w2

    def test_default_is_balanced(self) -> None:
        assert DEFAULT_MODE_WEIGHTS == (0.5, 0.5)

    def test_all_modalities_have_numeric_and_text(self) -> None:
        for mode, weights in RECONCILER_WEIGHTS_BY_MODE.items():
            assert "numeric" in weights, f"missing numeric for {mode}"
            assert "text" in weights, f"missing text for {mode}"
