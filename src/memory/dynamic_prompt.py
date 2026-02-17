"""
Dynamic prompt enhancement using correction history.

Enhances extraction prompts by injecting field-specific warnings
and guidance derived from past user corrections. This enables the
VLM to avoid repeating known mistakes without retraining.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.config import get_logger


logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class FieldWarning:
    """A single warning for a field based on correction history.

    Attributes:
        field_name: Name of the field.
        warning_text: Human-readable warning for the VLM.
        severity: low / medium / high — based on correction frequency.
        correction_count: How many corrections drive this warning.
    """

    field_name: str
    warning_text: str
    severity: str = "low"
    correction_count: int = 0


@dataclass(slots=True)
class PromptEnhancement:
    """Result of enhancing a prompt with correction history.

    Attributes:
        original_prompt: The unmodified prompt.
        enhanced_prompt: The prompt with correction context injected.
        field_warnings: Per-field warnings that were injected.
        total_corrections_used: Total corrections that informed the enhancement.
        enhancement_applied: Whether any enhancement was actually applied.
    """

    original_prompt: str
    enhanced_prompt: str
    field_warnings: dict[str, list[FieldWarning]] = field(default_factory=dict)
    total_corrections_used: int = 0
    enhancement_applied: bool = False


# ---------------------------------------------------------------------------
# Severity thresholds
# ---------------------------------------------------------------------------

_SEVERITY_LOW = 1
_SEVERITY_MEDIUM = 3
_SEVERITY_HIGH = 8


def _severity_for_count(count: int) -> str:
    """Map correction count to severity label."""
    if count >= _SEVERITY_HIGH:
        return "high"
    if count >= _SEVERITY_MEDIUM:
        return "medium"
    return "low"


# ---------------------------------------------------------------------------
# DynamicPromptEnhancer
# ---------------------------------------------------------------------------


class DynamicPromptEnhancer:
    """Enhances extraction prompts with correction-history context.

    The enhancer pulls field-level correction statistics from a
    ``CorrectionTracker`` and builds concise VLM-readable warnings
    that are appended to the extraction prompt.  This allows the model
    to avoid repeating the same extraction mistakes across documents.

    Usage::

        enhancer = DynamicPromptEnhancer(tracker=correction_tracker)
        result = enhancer.enhance_prompt(
            base_prompt="Extract these fields ...",
            field_names=["patient_name", "dob", "total_charges"],
            document_type="cms1500",
        )
        # result.enhanced_prompt contains the enriched prompt
    """

    # Maximum number of error examples per field to avoid prompt bloat
    MAX_EXAMPLES_PER_FIELD: int = 3

    # Maximum total injection length (characters) to keep prompt compact
    MAX_INJECTION_CHARS: int = 2000

    def __init__(
        self,
        tracker: Any | None = None,
        *,
        max_examples_per_field: int | None = None,
        max_injection_chars: int | None = None,
    ) -> None:
        """Initialise the enhancer.

        Args:
            tracker: A ``CorrectionTracker`` instance (or compatible).
                     If *None*, enhancement is a no-op (passthrough).
            max_examples_per_field: Override default example cap.
            max_injection_chars: Override default injection char limit.
        """
        self._tracker = tracker
        self._max_examples = max_examples_per_field or self.MAX_EXAMPLES_PER_FIELD
        self._max_chars = max_injection_chars or self.MAX_INJECTION_CHARS
        self._enhancement_count = 0
        self._total_warnings_emitted = 0
        self._logger = get_logger("memory.dynamic_prompt")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enhance_prompt(
        self,
        base_prompt: str,
        field_names: list[str],
        document_type: str = "",
    ) -> PromptEnhancement:
        """Enhance an extraction prompt with correction-based warnings.

        Args:
            base_prompt: The original extraction prompt.
            field_names: Field names that will be extracted.
            document_type: Document type for context filtering.

        Returns:
            ``PromptEnhancement`` with the enriched prompt and metadata.
        """
        if not self._tracker or not field_names:
            return PromptEnhancement(
                original_prompt=base_prompt,
                enhanced_prompt=base_prompt,
            )

        # Gather warnings for each field
        all_warnings: dict[str, list[FieldWarning]] = {}
        total_corrections = 0

        for name in field_names:
            warnings = self.get_field_warnings(name, document_type)
            if warnings:
                all_warnings[name] = warnings
                total_corrections += sum(w.correction_count for w in warnings)

        if not all_warnings:
            return PromptEnhancement(
                original_prompt=base_prompt,
                enhanced_prompt=base_prompt,
            )

        # Build the injection block
        injection = self.build_correction_context(
            field_names=field_names,
            document_type=document_type,
            warnings=all_warnings,
        )

        enhanced = base_prompt + injection
        self._enhancement_count += 1
        self._total_warnings_emitted += sum(
            len(ws) for ws in all_warnings.values()
        )

        self._logger.info(
            "prompt_enhanced",
            fields_warned=len(all_warnings),
            total_corrections=total_corrections,
        )

        return PromptEnhancement(
            original_prompt=base_prompt,
            enhanced_prompt=enhanced,
            field_warnings=all_warnings,
            total_corrections_used=total_corrections,
            enhancement_applied=True,
        )

    def get_field_warnings(
        self,
        field_name: str,
        document_type: str = "",
    ) -> list[FieldWarning]:
        """Build warnings for a single field from correction history.

        Args:
            field_name: The field to check.
            document_type: Optional document-type filter.

        Returns:
            List of ``FieldWarning`` objects (may be empty).
        """
        if not self._tracker:
            return []

        hints = self._tracker.get_field_hints(field_name, document_type)
        if not hints:
            return []

        correction_count: int = hints.get("total_corrections", 0)
        if correction_count == 0:
            return []

        severity = _severity_for_count(correction_count)
        warnings: list[FieldWarning] = []

        # 1. General frequency warning
        most_common_issue = hints.get("most_common_issue", "value")
        warnings.append(
            FieldWarning(
                field_name=field_name,
                warning_text=(
                    f"This field has been corrected {correction_count} time(s) "
                    f"(most common issue: {most_common_issue}). "
                    f"Double-check your extraction carefully."
                ),
                severity=severity,
                correction_count=correction_count,
            )
        )

        # 2. Specific error examples
        common_errors: list[dict[str, str]] = hints.get("common_errors") or []
        for error in common_errors[: self._max_examples]:
            original = error.get("original", "")
            corrected = error.get("corrected", "")
            if original and corrected:
                warnings.append(
                    FieldWarning(
                        field_name=field_name,
                        warning_text=(
                            f"Known mistake: '{original}' was corrected to "
                            f"'{corrected}'. Avoid this error."
                        ),
                        severity=severity,
                        correction_count=correction_count,
                    )
                )

        return warnings

    def build_correction_context(
        self,
        field_names: list[str],
        document_type: str = "",
        warnings: dict[str, list[FieldWarning]] | None = None,
    ) -> str:
        """Build a formatted context block for prompt injection.

        Args:
            field_names: Fields to include.
            document_type: Optional document type filter.
            warnings: Pre-computed warnings (if None, will be computed).

        Returns:
            Formatted string ready to append to a prompt.
        """
        if warnings is None:
            warnings = {}
            for name in field_names:
                ws = self.get_field_warnings(name, document_type)
                if ws:
                    warnings[name] = ws

        if not warnings:
            return ""

        lines: list[str] = [
            "",
            "",
            "--- CORRECTION HISTORY WARNINGS ---",
            "The following fields have known extraction issues based on "
            "past corrections. Pay extra attention to these fields:",
            "",
        ]

        for fname, field_warnings in warnings.items():
            severity = field_warnings[0].severity if field_warnings else "low"
            severity_marker = {
                "high": "!!",
                "medium": "!",
                "low": "*",
            }.get(severity, "*")

            lines.append(f"{severity_marker} **{fname}**:")
            for w in field_warnings:
                lines.append(f"  - {w.warning_text}")
            lines.append("")

        lines.append("--- END CORRECTION WARNINGS ---")

        block = "\n".join(lines)

        # Truncate if too long
        if len(block) > self._max_chars:
            block = block[: self._max_chars - 20] + "\n[truncated]\n---"

        return block

    def get_enhancement_stats(self) -> dict[str, Any]:
        """Return statistics about enhancements applied this session.

        Returns:
            Dictionary with enhancement counts.
        """
        return {
            "total_enhancements": self._enhancement_count,
            "total_warnings_emitted": self._total_warnings_emitted,
            "tracker_available": self._tracker is not None,
        }

    def reset_stats(self) -> None:
        """Reset session-level statistics."""
        self._enhancement_count = 0
        self._total_warnings_emitted = 0
