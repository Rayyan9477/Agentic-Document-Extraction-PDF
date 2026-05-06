"""
WS-3: specialized medical input modes.

A *modality* is a tag on the document (or a single page) that tells the
extraction pipeline how to treat the input. Multiple modalities can apply
at once: a fax of a handwritten form is ``{"fax", "handwritten", "form"}``.

Modalities are auto-detected by the analyzer (see ``derive_modalities``)
and can be overridden by the API caller via ``ProcessRequest.modality_override``
or by the CLI. The image enhancer and the prompt builder both consume the
final mode set.

Modality reference table:

==========    ==============================================  ====================================================
Mode          When detected                                   Effect
==========    ==============================================  ====================================================
``printed``   Default. Always set unless explicitly cleared.  Standard preprocessing + base extraction prompt.
``handwritten`` ``analysis.has_handwriting`` is true.         Skip CLAHE; gentle denoise; "treat values as low confidence".
``table``     ``analysis.has_tables`` or ``table_count > 0``. Use ``build_table_extraction_prompt``.
``form``      ``analysis.layout_type == "form"``.             "Extract by label-value pairs".
``fax``       Quality metrics indicate low contrast +         Otsu binarization + morphological opening; "treat as
              low blur score (typical fax characteristics).   1-bit fax; missing strokes are common".
``visual``    Low text density + no handwriting + no tables.  "Describe what you see; do not invent structure".
==========    ==============================================  ====================================================
"""

from __future__ import annotations

from typing import Any


# Canonical mode names — keep in sync with the table above.
MODE_PRINTED: str = "printed"
MODE_HANDWRITTEN: str = "handwritten"
MODE_TABLE: str = "table"
MODE_FORM: str = "form"
MODE_FAX: str = "fax"
MODE_VISUAL: str = "visual"

ALL_MODES: tuple[str, ...] = (
    MODE_PRINTED,
    MODE_HANDWRITTEN,
    MODE_TABLE,
    MODE_FORM,
    MODE_FAX,
    MODE_VISUAL,
)


def derive_modalities(
    analysis: dict[str, Any] | None,
    quality_metrics: list[dict[str, Any]] | None = None,
) -> list[str]:
    """Compute the active modalities from analyzer output and image-quality metrics.

    Pure function — does not consume VLM calls. Heuristics are deliberately
    conservative; ambiguous cases default to ``printed``. Callers can layer
    explicit user overrides on top via ``apply_overrides``.

    Args:
        analysis: ``DocumentAnalysis`` dict (or any mapping with the same
            shape) emitted by ``AnalyzerAgent``. May be ``None`` if the
            analyzer hasn't run yet.
        quality_metrics: Optional list of per-page quality dicts produced by
            ``ImageEnhancer.analyze_quality``. The fax heuristic uses the
            average of these. May be ``None`` or empty.

    Returns:
        Sorted, de-duplicated list of mode names. ``printed`` is always
        included unless an explicit override removes it.
    """
    modes: set[str] = {MODE_PRINTED}

    if analysis:
        if analysis.get("has_handwriting"):
            modes.add(MODE_HANDWRITTEN)
        if analysis.get("has_tables") or int(analysis.get("table_count", 0) or 0) > 0:
            modes.add(MODE_TABLE)
        if (analysis.get("layout_type") or "").lower() == "form":
            modes.add(MODE_FORM)
        # Visual: text-light pages with no structure (radiology, ultrasound,
        # photo-of-screen). Only triggers when the analyzer is confident
        # there's no handwriting or tables — otherwise it'd misfire on,
        # e.g., a faxed CMS-1500 with thin text.
        if (
            (analysis.get("text_density") or "").lower() == "low"
            and not analysis.get("has_handwriting")
            and not analysis.get("has_tables")
        ):
            modes.add(MODE_VISUAL)

    # Fax heuristic: average page quality is bimodal (very low contrast +
    # low blur score) — typical of 1-bit CCITT-compressed fax scans. We
    # don't trust a single noisy page; require the bulk of the document
    # to look like fax before tagging it as such.
    if quality_metrics:
        n = len(quality_metrics)
        low_contrast_pages = sum(1 for q in quality_metrics if q.get("low_contrast"))
        avg_blur = sum(float(q.get("blur_score", 0.0)) for q in quality_metrics) / max(n, 1)
        avg_quality = sum(float(q.get("quality_score", 100.0)) for q in quality_metrics) / max(n, 1)
        if low_contrast_pages >= max(1, n // 2) and avg_blur < 150 and avg_quality < 50:
            modes.add(MODE_FAX)

    return sorted(modes)


def apply_overrides(
    derived: list[str] | None,
    override: list[str] | None,
) -> list[str]:
    """Merge user-supplied mode overrides with the auto-detected set.

    Semantics:
        * ``override is None`` → return ``derived`` unchanged.
        * ``override == []`` → "auto-detect" (same as ``None``).
        * ``override`` non-empty → take it verbatim; ``printed`` is added
          if absent so the printed prompt is always at least baseline.

    Unknown mode names are silently dropped so user input can't smuggle
    arbitrary strings into the prompt. The result is sorted + deduped.
    """
    if not override:
        return list(derived or [MODE_PRINTED])

    valid = {m for m in override if m in ALL_MODES}
    if not valid:
        # All overrides invalid → fall back to derived rather than producing
        # a meaningless empty mode set.
        return list(derived or [MODE_PRINTED])

    valid.add(MODE_PRINTED)
    return sorted(valid)


__all__ = [
    "ALL_MODES",
    "MODE_FAX",
    "MODE_FORM",
    "MODE_HANDWRITTEN",
    "MODE_PRINTED",
    "MODE_TABLE",
    "MODE_VISUAL",
    "apply_overrides",
    "derive_modalities",
]
