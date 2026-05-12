"""
Phase 3 — critic_combiner unit tests.

Coverage:

* ``combine_confidence`` produces a dict with the four expected keys
  and a ``raw_combined`` value in [0, 1].
* Weight respect: changing weights changes the output proportionally.
* Modality penalty: ``fax`` > ``handwritten`` > ``visual`` > printed.
* Multi-modality: worst-of penalty selected.
* ``apply_combiner_to_state`` reads the right state keys for each
  engine mode (legacy uses ``overall_confidence``; dual_vlm uses
  ``reconciliation_metadata.agreement_rate``).
* No critic ⇒ critic_trust=1.0 (no penalty term).
"""

from __future__ import annotations

import pytest

from src.validation.critic_combiner import (
    apply_combiner_to_state,
    combine_confidence,
    _modality_penalty,
)


# ---------------------------------------------------------------------------
# Modality penalty table
# ---------------------------------------------------------------------------


class TestModalityPenalty:
    def test_no_modalities(self) -> None:
        assert _modality_penalty([]) == 0.0

    def test_fax_is_worst(self) -> None:
        assert _modality_penalty(["fax"]) == 0.7

    def test_handwritten(self) -> None:
        assert _modality_penalty(["handwritten"]) == 0.6

    def test_visual(self) -> None:
        assert _modality_penalty(["visual"]) == 0.4

    def test_printed_zero_penalty(self) -> None:
        assert _modality_penalty(["printed"]) == 0.0

    def test_multi_modality_worst_of(self) -> None:
        # fax + handwritten → take fax's 0.7
        assert _modality_penalty(["handwritten", "fax"]) == 0.7

    def test_unknown_modality_zero_penalty(self) -> None:
        assert _modality_penalty(["custom_mode"]) == 0.0


# ---------------------------------------------------------------------------
# combine_confidence
# ---------------------------------------------------------------------------


class TestCombineConfidence:
    def test_returns_all_four_keys(self) -> None:
        result = combine_confidence(
            dual_pass_agreement=0.9,
            critic_trust=0.85,
            modalities=[],
        )
        assert set(result.keys()) == {
            "dual_pass",
            "critic",
            "modality_penalty",
            "raw_combined",
        }

    def test_all_perfect_yields_one(self) -> None:
        result = combine_confidence(
            dual_pass_agreement=1.0,
            critic_trust=1.0,
            modalities=[],
            weights=(0.5, 0.3, 0.2),
        )
        assert result["raw_combined"] == pytest.approx(1.0)

    def test_clamps_dual_pass_to_unit(self) -> None:
        result = combine_confidence(
            dual_pass_agreement=2.5,  # invalid input
            critic_trust=0.5,
        )
        assert result["dual_pass"] == 1.0

    def test_clamps_critic_to_unit(self) -> None:
        result = combine_confidence(
            dual_pass_agreement=0.5,
            critic_trust=-0.3,  # invalid input
        )
        assert result["critic"] == 0.0

    def test_modality_penalty_drags_combined_down(self) -> None:
        no_pen = combine_confidence(
            dual_pass_agreement=1.0,
            critic_trust=1.0,
            modalities=[],
            weights=(0.5, 0.3, 0.2),
        )
        with_fax = combine_confidence(
            dual_pass_agreement=1.0,
            critic_trust=1.0,
            modalities=["fax"],
            weights=(0.5, 0.3, 0.2),
        )
        # fax penalty 0.7 → modality_term = 0.3 → combined drops by
        # 0.2 * (1.0 - 0.3) = 0.14
        assert with_fax["raw_combined"] < no_pen["raw_combined"]
        assert with_fax["raw_combined"] == pytest.approx(0.86)

    def test_weight_validation_via_settings_path(self) -> None:
        # combine_confidence accepts arbitrary weights; the validator
        # on settings.extraction.critic_combiner_weights catches
        # invalid sums. Here we just sanity-check a custom weight set.
        result = combine_confidence(
            dual_pass_agreement=1.0,
            critic_trust=0.0,
            modalities=[],
            weights=(0.0, 1.0, 0.0),
        )
        # All weight on critic; dual_pass and modality contribute 0.
        assert result["raw_combined"] == 0.0

    def test_combined_clamped_into_unit(self) -> None:
        # Pathological weights still produce a clamped output.
        result = combine_confidence(
            dual_pass_agreement=1.0,
            critic_trust=1.0,
            modalities=[],
            weights=(2.0, 2.0, 2.0),  # Sum 6, not validated here
        )
        assert 0.0 <= result["raw_combined"] <= 1.0


# ---------------------------------------------------------------------------
# apply_combiner_to_state
# ---------------------------------------------------------------------------


class TestApplyCombinerToState:
    def test_reads_overall_confidence_when_no_recon_metadata(self) -> None:
        state: dict = {
            "overall_confidence": 0.8,
            "critic_report": {"trust_score": 0.9},
            "modalities": [],
        }
        result = apply_combiner_to_state(state)
        assert result["dual_pass"] == 0.8
        assert result["critic"] == 0.9

    def test_reads_recon_metadata_when_present(self) -> None:
        state: dict = {
            "reconciliation_metadata": {"agreement_rate": 0.95},
            "overall_confidence": 0.5,  # should be ignored
            "critic_report": {"trust_score": 0.85},
            "modalities": [],
        }
        result = apply_combiner_to_state(state)
        assert result["dual_pass"] == 0.95

    def test_no_critic_treated_as_full_trust(self) -> None:
        state: dict = {
            "overall_confidence": 0.8,
            "critic_report": {},  # no trust_score
            "modalities": [],
        }
        result = apply_combiner_to_state(state)
        assert result["critic"] == 1.0

    def test_no_critic_key_at_all(self) -> None:
        state: dict = {
            "overall_confidence": 0.8,
            "modalities": [],
        }
        result = apply_combiner_to_state(state)
        assert result["critic"] == 1.0

    def test_modality_penalty_propagates(self) -> None:
        state: dict = {
            "overall_confidence": 1.0,
            "critic_report": {"trust_score": 1.0},
            "modalities": ["handwritten"],
        }
        result = apply_combiner_to_state(state)
        assert result["modality_penalty"] == 0.6

    def test_does_not_mutate_state(self) -> None:
        state: dict = {
            "overall_confidence": 0.8,
            "critic_report": {"trust_score": 0.9},
            "modalities": [],
        }
        before = dict(state)
        apply_combiner_to_state(state)
        assert state == before
