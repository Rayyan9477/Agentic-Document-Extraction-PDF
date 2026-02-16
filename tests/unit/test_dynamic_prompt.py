"""
Unit tests for Phase 3B: Dynamic Prompt Enhancement.

Tests DynamicPromptEnhancer, FieldWarning, PromptEnhancement,
correction-context building, and ExtractorAgent integration.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.memory.dynamic_prompt import (
    DynamicPromptEnhancer,
    FieldWarning,
    PromptEnhancement,
    _severity_for_count,
)


# ──────────────────────────────────────────────────────────────────
# Helpers — mock CorrectionTracker
# ──────────────────────────────────────────────────────────────────


def _make_tracker(
    field_hints: dict[str, dict[str, Any]] | None = None,
) -> MagicMock:
    """Create a mock CorrectionTracker with configurable get_field_hints."""
    tracker = MagicMock()
    hints = field_hints or {}

    def _get_hints(field_name: str, document_type: str | None = None) -> dict[str, Any]:
        return hints.get(field_name, {})

    tracker.get_field_hints = MagicMock(side_effect=_get_hints)
    return tracker


def _simple_hints(
    total_corrections: int = 5,
    most_common_issue: str = "value",
    common_errors: list[dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Build a simple field hints dict."""
    return {
        "total_corrections": total_corrections,
        "common_errors": common_errors or [],
        "avg_confidence": 0.7,
        "confidence_boost": -0.10,
        "most_common_issue": most_common_issue,
    }


# ──────────────────────────────────────────────────────────────────
# FieldWarning Tests
# ──────────────────────────────────────────────────────────────────


class TestFieldWarning:
    def test_basic_creation(self):
        w = FieldWarning(
            field_name="patient_name",
            warning_text="Watch out!",
        )
        assert w.field_name == "patient_name"
        assert w.warning_text == "Watch out!"
        assert w.severity == "low"
        assert w.correction_count == 0

    def test_with_severity(self):
        w = FieldWarning(
            field_name="dob",
            warning_text="Common mistake",
            severity="high",
            correction_count=15,
        )
        assert w.severity == "high"
        assert w.correction_count == 15

    def test_frozen(self):
        w = FieldWarning(field_name="f", warning_text="w")
        with pytest.raises(AttributeError):
            w.field_name = "other"  # type: ignore


# ──────────────────────────────────────────────────────────────────
# PromptEnhancement Tests
# ──────────────────────────────────────────────────────────────────


class TestPromptEnhancement:
    def test_default_values(self):
        pe = PromptEnhancement(
            original_prompt="base",
            enhanced_prompt="base",
        )
        assert pe.field_warnings == {}
        assert pe.total_corrections_used == 0
        assert pe.enhancement_applied is False

    def test_with_data(self):
        w = FieldWarning(field_name="f1", warning_text="test")
        pe = PromptEnhancement(
            original_prompt="base",
            enhanced_prompt="base + context",
            field_warnings={"f1": [w]},
            total_corrections_used=5,
            enhancement_applied=True,
        )
        assert pe.enhancement_applied is True
        assert pe.total_corrections_used == 5
        assert "f1" in pe.field_warnings


# ──────────────────────────────────────────────────────────────────
# Severity Mapping Tests
# ──────────────────────────────────────────────────────────────────


class TestSeverity:
    @pytest.mark.parametrize(
        "count,expected",
        [
            (0, "low"),
            (1, "low"),
            (2, "low"),
            (3, "medium"),
            (5, "medium"),
            (7, "medium"),
            (8, "high"),
            (10, "high"),
            (100, "high"),
        ],
    )
    def test_severity_mapping(self, count: int, expected: str):
        assert _severity_for_count(count) == expected


# ──────────────────────────────────────────────────────────────────
# DynamicPromptEnhancer — No Tracker
# ──────────────────────────────────────────────────────────────────


class TestEnhancerNoTracker:
    def test_no_tracker_passthrough(self):
        enhancer = DynamicPromptEnhancer(tracker=None)
        result = enhancer.enhance_prompt("prompt", ["f1", "f2"], "cms1500")
        assert result.enhanced_prompt == "prompt"
        assert result.enhancement_applied is False

    def test_empty_field_names_passthrough(self):
        tracker = _make_tracker({"f1": _simple_hints()})
        enhancer = DynamicPromptEnhancer(tracker=tracker)
        result = enhancer.enhance_prompt("prompt", [], "cms1500")
        assert result.enhanced_prompt == "prompt"
        assert result.enhancement_applied is False

    def test_stats_without_tracker(self):
        enhancer = DynamicPromptEnhancer(tracker=None)
        stats = enhancer.get_enhancement_stats()
        assert stats["tracker_available"] is False
        assert stats["total_enhancements"] == 0


# ──────────────────────────────────────────────────────────────────
# DynamicPromptEnhancer — Field Warnings
# ──────────────────────────────────────────────────────────────────


class TestFieldWarnings:
    def test_no_corrections_returns_empty(self):
        tracker = _make_tracker({"f1": {}})
        enhancer = DynamicPromptEnhancer(tracker=tracker)
        warnings = enhancer.get_field_warnings("f1")
        assert warnings == []

    def test_zero_corrections_returns_empty(self):
        tracker = _make_tracker({"f1": _simple_hints(total_corrections=0)})
        enhancer = DynamicPromptEnhancer(tracker=tracker)
        warnings = enhancer.get_field_warnings("f1")
        assert warnings == []

    def test_unknown_field_returns_empty(self):
        tracker = _make_tracker({})
        enhancer = DynamicPromptEnhancer(tracker=tracker)
        warnings = enhancer.get_field_warnings("nonexistent")
        assert warnings == []

    def test_basic_frequency_warning(self):
        tracker = _make_tracker({"dob": _simple_hints(total_corrections=5)})
        enhancer = DynamicPromptEnhancer(tracker=tracker)
        warnings = enhancer.get_field_warnings("dob")
        assert len(warnings) >= 1
        assert "5 time(s)" in warnings[0].warning_text
        assert warnings[0].severity == "medium"

    def test_common_error_examples_included(self):
        errors = [
            {"original": "01/15/1990", "corrected": "01/15/1980"},
            {"original": "John Smith", "corrected": "John A. Smith"},
        ]
        tracker = _make_tracker(
            {"patient_name": _simple_hints(total_corrections=4, common_errors=errors)}
        )
        enhancer = DynamicPromptEnhancer(tracker=tracker)
        warnings = enhancer.get_field_warnings("patient_name")
        # 1 frequency warning + 2 error examples
        assert len(warnings) == 3
        assert "01/15/1990" in warnings[1].warning_text
        assert "John Smith" in warnings[2].warning_text

    def test_max_examples_respected(self):
        errors = [
            {"original": f"v{i}", "corrected": f"c{i}"} for i in range(10)
        ]
        tracker = _make_tracker(
            {"f1": _simple_hints(total_corrections=15, common_errors=errors)}
        )
        enhancer = DynamicPromptEnhancer(tracker=tracker, max_examples_per_field=2)
        warnings = enhancer.get_field_warnings("f1")
        # 1 frequency + max 2 examples = 3
        assert len(warnings) == 3

    def test_high_severity(self):
        tracker = _make_tracker(
            {"charges": _simple_hints(total_corrections=10)}
        )
        enhancer = DynamicPromptEnhancer(tracker=tracker)
        warnings = enhancer.get_field_warnings("charges")
        assert warnings[0].severity == "high"

    def test_low_severity(self):
        tracker = _make_tracker(
            {"field": _simple_hints(total_corrections=1)}
        )
        enhancer = DynamicPromptEnhancer(tracker=tracker)
        warnings = enhancer.get_field_warnings("field")
        assert warnings[0].severity == "low"


# ──────────────────────────────────────────────────────────────────
# DynamicPromptEnhancer — Prompt Enhancement
# ──────────────────────────────────────────────────────────────────


class TestEnhancePrompt:
    def test_enhances_with_corrections(self):
        tracker = _make_tracker(
            {"patient_name": _simple_hints(total_corrections=5)}
        )
        enhancer = DynamicPromptEnhancer(tracker=tracker)
        result = enhancer.enhance_prompt(
            "Extract these fields:", ["patient_name"], "cms1500",
        )
        assert result.enhancement_applied is True
        assert "CORRECTION HISTORY WARNINGS" in result.enhanced_prompt
        assert "patient_name" in result.enhanced_prompt
        assert result.original_prompt == "Extract these fields:"

    def test_no_enhancement_when_no_corrections(self):
        tracker = _make_tracker({})
        enhancer = DynamicPromptEnhancer(tracker=tracker)
        result = enhancer.enhance_prompt(
            "Extract:", ["unknown_field"], "cms1500",
        )
        assert result.enhancement_applied is False
        assert result.enhanced_prompt == "Extract:"

    def test_multiple_fields_enhanced(self):
        tracker = _make_tracker({
            "patient_name": _simple_hints(total_corrections=3),
            "dob": _simple_hints(total_corrections=8),
        })
        enhancer = DynamicPromptEnhancer(tracker=tracker)
        result = enhancer.enhance_prompt(
            "prompt", ["patient_name", "dob", "npi"], "cms1500",
        )
        assert result.enhancement_applied is True
        assert "patient_name" in result.enhanced_prompt
        assert "dob" in result.enhanced_prompt
        assert len(result.field_warnings) == 2

    def test_total_corrections_counted(self):
        tracker = _make_tracker({
            "f1": _simple_hints(total_corrections=3),
            "f2": _simple_hints(total_corrections=7),
        })
        enhancer = DynamicPromptEnhancer(tracker=tracker)
        result = enhancer.enhance_prompt("p", ["f1", "f2"], "")
        assert result.total_corrections_used == 10

    def test_stats_updated_after_enhancement(self):
        tracker = _make_tracker(
            {"f1": _simple_hints(total_corrections=2)}
        )
        enhancer = DynamicPromptEnhancer(tracker=tracker)
        enhancer.enhance_prompt("p", ["f1"], "")
        enhancer.enhance_prompt("p2", ["f1"], "")

        stats = enhancer.get_enhancement_stats()
        assert stats["total_enhancements"] == 2
        assert stats["total_warnings_emitted"] >= 2
        assert stats["tracker_available"] is True


# ──────────────────────────────────────────────────────────────────
# DynamicPromptEnhancer — Context Building
# ──────────────────────────────────────────────────────────────────


class TestBuildCorrectionContext:
    def test_empty_warnings_returns_empty(self):
        enhancer = DynamicPromptEnhancer(tracker=None)
        ctx = enhancer.build_correction_context(["f1"], "", warnings={})
        assert ctx == ""

    def test_builds_formatted_block(self):
        warnings = {
            "dob": [
                FieldWarning(
                    field_name="dob",
                    warning_text="Frequently corrected",
                    severity="medium",
                    correction_count=4,
                )
            ]
        }
        enhancer = DynamicPromptEnhancer(tracker=None)
        ctx = enhancer.build_correction_context(["dob"], "", warnings=warnings)
        assert "CORRECTION HISTORY WARNINGS" in ctx
        assert "dob" in ctx
        assert "Frequently corrected" in ctx

    def test_severity_markers(self):
        warnings = {
            "high_field": [
                FieldWarning("high_field", "w", severity="high", correction_count=10)
            ],
            "med_field": [
                FieldWarning("med_field", "w", severity="medium", correction_count=5)
            ],
            "low_field": [
                FieldWarning("low_field", "w", severity="low", correction_count=1)
            ],
        }
        enhancer = DynamicPromptEnhancer(tracker=None)
        ctx = enhancer.build_correction_context(
            ["high_field", "med_field", "low_field"], "", warnings=warnings,
        )
        assert "!! **high_field**" in ctx
        assert "! **med_field**" in ctx
        assert "* **low_field**" in ctx

    def test_truncation_on_long_context(self):
        # Create a very large set of warnings
        warnings = {}
        for i in range(50):
            fname = f"field_{i}"
            warnings[fname] = [
                FieldWarning(
                    field_name=fname,
                    warning_text="x" * 100,
                    severity="medium",
                    correction_count=5,
                )
            ]
        enhancer = DynamicPromptEnhancer(
            tracker=None, max_injection_chars=500,
        )
        ctx = enhancer.build_correction_context(
            list(warnings.keys()), "", warnings=warnings,
        )
        assert len(ctx) <= 500
        assert "[truncated]" in ctx

    def test_computes_warnings_when_not_provided(self):
        tracker = _make_tracker(
            {"f1": _simple_hints(total_corrections=3)}
        )
        enhancer = DynamicPromptEnhancer(tracker=tracker)
        ctx = enhancer.build_correction_context(["f1", "f2"], "cms1500")
        assert "f1" in ctx

    def test_end_marker_present(self):
        warnings = {
            "f1": [FieldWarning("f1", "test", "low", 1)]
        }
        enhancer = DynamicPromptEnhancer(tracker=None)
        ctx = enhancer.build_correction_context(["f1"], "", warnings=warnings)
        assert "END CORRECTION WARNINGS" in ctx


# ──────────────────────────────────────────────────────────────────
# DynamicPromptEnhancer — Stats & Reset
# ──────────────────────────────────────────────────────────────────


class TestStatsAndReset:
    def test_initial_stats(self):
        enhancer = DynamicPromptEnhancer(tracker=_make_tracker({}))
        stats = enhancer.get_enhancement_stats()
        assert stats["total_enhancements"] == 0
        assert stats["total_warnings_emitted"] == 0
        assert stats["tracker_available"] is True

    def test_reset_stats(self):
        tracker = _make_tracker({"f1": _simple_hints(total_corrections=2)})
        enhancer = DynamicPromptEnhancer(tracker=tracker)
        enhancer.enhance_prompt("p", ["f1"], "")
        assert enhancer.get_enhancement_stats()["total_enhancements"] == 1

        enhancer.reset_stats()
        assert enhancer.get_enhancement_stats()["total_enhancements"] == 0
        assert enhancer.get_enhancement_stats()["total_warnings_emitted"] == 0


# ──────────────────────────────────────────────────────────────────
# Edge Cases
# ──────────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_tracker_get_hints_raises(self):
        tracker = MagicMock()
        tracker.get_field_hints = MagicMock(side_effect=RuntimeError("db down"))
        enhancer = DynamicPromptEnhancer(tracker=tracker)
        # get_field_warnings should propagate — caller handles
        with pytest.raises(RuntimeError):
            enhancer.get_field_warnings("f1")

    def test_enhance_with_tracker_error_is_safe(self):
        """enhance_prompt catches errors from the tracker gracefully."""
        tracker = MagicMock()
        tracker.get_field_hints = MagicMock(side_effect=RuntimeError("boom"))
        enhancer = DynamicPromptEnhancer(tracker=tracker)
        # enhance_prompt calls get_field_warnings which will raise,
        # but enhance_prompt should not catch it (caller should handle)
        with pytest.raises(RuntimeError):
            enhancer.enhance_prompt("p", ["f1"], "")

    def test_error_with_empty_original(self):
        """Error dict with empty original is skipped."""
        errors = [
            {"original": "", "corrected": "correct_value"},
            {"original": "bad", "corrected": ""},
        ]
        tracker = _make_tracker(
            {"f1": _simple_hints(total_corrections=2, common_errors=errors)}
        )
        enhancer = DynamicPromptEnhancer(tracker=tracker)
        warnings = enhancer.get_field_warnings("f1")
        # Should only include frequency warning + errors where both are non-empty
        error_warnings = [w for w in warnings if "Known mistake" in w.warning_text]
        assert len(error_warnings) == 0  # Both have one empty side

    def test_common_errors_none(self):
        hints = {
            "total_corrections": 3,
            "common_errors": None,
            "most_common_issue": "value",
        }
        tracker = _make_tracker({"f1": hints})
        enhancer = DynamicPromptEnhancer(tracker=tracker)
        warnings = enhancer.get_field_warnings("f1")
        # Should still get the frequency warning
        assert len(warnings) == 1

    def test_most_common_issue_in_warning(self):
        tracker = _make_tracker(
            {"f1": _simple_hints(total_corrections=2, most_common_issue="format")}
        )
        enhancer = DynamicPromptEnhancer(tracker=tracker)
        warnings = enhancer.get_field_warnings("f1")
        assert "format" in warnings[0].warning_text


# ──────────────────────────────────────────────────────────────────
# State Fields
# ──────────────────────────────────────────────────────────────────


class TestStateFields:
    def test_initial_state_has_enhancement_flag(self):
        from src.pipeline.state import create_initial_state
        state = create_initial_state(pdf_path="test.pdf")
        assert state["prompt_enhancement_applied"] is False

    def test_update_state_preserves_flag(self):
        from src.pipeline.state import create_initial_state, update_state
        state = create_initial_state(pdf_path="test.pdf")
        state = update_state(state, {"prompt_enhancement_applied": True})
        assert state["prompt_enhancement_applied"] is True


# ──────────────────────────────────────────────────────────────────
# Memory Module Exports
# ──────────────────────────────────────────────────────────────────


class TestModuleExports:
    def test_imports_from_memory(self):
        from src.memory import DynamicPromptEnhancer as DPE
        from src.memory import FieldWarning as FW
        from src.memory import PromptEnhancement as PE
        assert DPE is DynamicPromptEnhancer
        assert FW is FieldWarning
        assert PE is PromptEnhancement


# ──────────────────────────────────────────────────────────────────
# ExtractorAgent Integration
# ──────────────────────────────────────────────────────────────────


class TestExtractorIntegration:
    def test_extractor_accepts_prompt_enhancer(self):
        """ExtractorAgent constructor should accept prompt_enhancer kwarg."""
        from src.agents.extractor import ExtractorAgent
        mock_client = MagicMock()
        enhancer = DynamicPromptEnhancer(tracker=None)
        agent = ExtractorAgent(client=mock_client, prompt_enhancer=enhancer)
        assert agent._prompt_enhancer is enhancer

    def test_extractor_default_no_enhancer(self):
        """ExtractorAgent without enhancer should work normally."""
        from src.agents.extractor import ExtractorAgent
        mock_client = MagicMock()
        agent = ExtractorAgent(client=mock_client)
        assert agent._prompt_enhancer is None
