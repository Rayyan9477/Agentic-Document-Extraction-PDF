"""WS-3: tests for the specialized-mode detection logic, the
modality-aware prompt builder, and the image enhancer's per-mode
preprocessing branches.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.agents.modality import (
    ALL_MODES,
    MODE_FAX,
    MODE_FORM,
    MODE_HANDWRITTEN,
    MODE_PRINTED,
    MODE_TABLE,
    MODE_VISUAL,
    apply_overrides,
    derive_modalities,
)


class TestDeriveModalities:
    def test_empty_analysis_returns_printed_baseline(self) -> None:
        assert derive_modalities(None) == [MODE_PRINTED]
        assert derive_modalities({}) == [MODE_PRINTED]

    def test_handwriting_signal_adds_handwritten(self) -> None:
        modes = derive_modalities({"has_handwriting": True})
        assert MODE_HANDWRITTEN in modes
        assert MODE_PRINTED in modes  # baseline always present

    def test_tables_signal_adds_table(self) -> None:
        assert MODE_TABLE in derive_modalities({"has_tables": True})
        # table_count alone (without has_tables) also triggers
        assert MODE_TABLE in derive_modalities({"table_count": 3})

    def test_form_layout_adds_form(self) -> None:
        assert MODE_FORM in derive_modalities({"layout_type": "form"})
        # case-insensitive
        assert MODE_FORM in derive_modalities({"layout_type": "FORM"})

    def test_visual_requires_low_text_density_and_no_other_signals(self) -> None:
        # All conditions met
        assert MODE_VISUAL in derive_modalities(
            {"text_density": "low", "has_handwriting": False, "has_tables": False}
        )
        # Visual is suppressed if handwriting present
        assert MODE_VISUAL not in derive_modalities(
            {"text_density": "low", "has_handwriting": True}
        )
        # Visual is suppressed if tables present
        assert MODE_VISUAL not in derive_modalities(
            {"text_density": "low", "has_tables": True}
        )

    def test_fax_heuristic_requires_majority_low_contrast_pages(self) -> None:
        # Strong fax signal: 3 of 3 pages low-contrast + low quality
        fax_metrics = [
            {"low_contrast": True, "blur_score": 80.0, "quality_score": 35.0},
            {"low_contrast": True, "blur_score": 95.0, "quality_score": 30.0},
            {"low_contrast": True, "blur_score": 70.0, "quality_score": 40.0},
        ]
        assert MODE_FAX in derive_modalities({}, fax_metrics)

    def test_fax_heuristic_negative_on_high_quality_input(self) -> None:
        crisp_metrics = [
            {"low_contrast": False, "blur_score": 800.0, "quality_score": 92.0},
        ]
        assert MODE_FAX not in derive_modalities({}, crisp_metrics)

    def test_combined_modes_all_apply(self) -> None:
        # Faxed handwritten form: every mode that should fire does
        modes = derive_modalities(
            {
                "has_handwriting": True,
                "has_tables": False,
                "layout_type": "form",
            },
            quality_metrics=[
                {"low_contrast": True, "blur_score": 60.0, "quality_score": 25.0},
                {"low_contrast": True, "blur_score": 70.0, "quality_score": 30.0},
            ],
        )
        assert MODE_HANDWRITTEN in modes
        assert MODE_FORM in modes
        assert MODE_FAX in modes
        assert MODE_PRINTED in modes  # baseline


class TestApplyOverrides:
    def test_no_override_returns_derived(self) -> None:
        derived = [MODE_PRINTED, MODE_TABLE]
        assert apply_overrides(derived, None) == derived
        assert apply_overrides(derived, []) == derived

    def test_override_replaces_derived(self) -> None:
        result = apply_overrides([MODE_PRINTED, MODE_TABLE], [MODE_FAX])
        assert MODE_FAX in result
        assert MODE_PRINTED in result  # always re-added as baseline
        # Derived modes (TABLE) are NOT carried over — caller is in charge.
        assert MODE_TABLE not in result

    def test_unknown_modes_dropped(self) -> None:
        result = apply_overrides([MODE_PRINTED], ["not-a-mode", "also-fake"])
        # All overrides invalid → fall back to derived
        assert result == [MODE_PRINTED]

    def test_partial_validity_keeps_valid_modes(self) -> None:
        result = apply_overrides(
            [MODE_PRINTED], [MODE_HANDWRITTEN, "garbage", MODE_FAX]
        )
        assert MODE_HANDWRITTEN in result
        assert MODE_FAX in result
        assert "garbage" not in result


class TestModalityPromptIntegration:
    def test_no_modalities_omits_section(self) -> None:
        from src.prompts.extraction import build_extraction_prompt

        prompt = build_extraction_prompt(
            schema_fields=[{"field_name": "patient_name", "field_type": "string"}],
            document_type="cms1500",
            page_number=1,
            total_pages=1,
            modalities=None,
        )
        assert "MODALITY-SPECIFIC RULES" not in prompt

    def test_printed_only_omits_section(self) -> None:
        from src.prompts.extraction import build_extraction_prompt

        prompt = build_extraction_prompt(
            schema_fields=[{"field_name": "patient_name", "field_type": "string"}],
            document_type="cms1500",
            page_number=1,
            total_pages=1,
            modalities=[MODE_PRINTED],
        )
        # baseline-only -> no fragment to emit
        assert "MODALITY-SPECIFIC RULES" not in prompt

    @pytest.mark.parametrize(
        "mode,marker",
        [
            (MODE_FAX, "FAX-GRADE INPUT"),
            (MODE_HANDWRITTEN, "HANDWRITING CAUTION"),
            (MODE_VISUAL, "VISUAL / IMAGE-FIRST PAGE"),
            (MODE_TABLE, "TABLE-AWARE EXTRACTION"),
            (MODE_FORM, "FORM-AWARE EXTRACTION"),
        ],
    )
    def test_each_mode_emits_distinct_guidance(
        self, mode: str, marker: str
    ) -> None:
        from src.prompts.extraction import build_extraction_prompt

        prompt = build_extraction_prompt(
            schema_fields=[{"field_name": "patient_name", "field_type": "string"}],
            document_type="cms1500",
            page_number=1,
            total_pages=1,
            modalities=[MODE_PRINTED, mode],
        )
        assert "MODALITY-SPECIFIC RULES" in prompt
        assert marker in prompt

    def test_combined_modes_emit_all_fragments(self) -> None:
        from src.prompts.extraction import build_extraction_prompt

        prompt = build_extraction_prompt(
            schema_fields=[{"field_name": "patient_name", "field_type": "string"}],
            document_type="cms1500",
            page_number=1,
            total_pages=1,
            modalities=[MODE_PRINTED, MODE_FAX, MODE_HANDWRITTEN, MODE_FORM],
        )
        assert "FAX-GRADE INPUT" in prompt
        assert "HANDWRITING CAUTION" in prompt
        assert "FORM-AWARE EXTRACTION" in prompt


class TestImageEnhancerModes:
    """The fax mode flips the enhancement chain to binarization +
    morphology and skips CLAHE; handwriting mode reduces denoise
    strength and skips CLAHE.
    """

    @pytest.fixture()
    def enhancer(self):
        from src.preprocessing.image_enhancer import ImageEnhancer

        return ImageEnhancer()

    @pytest.fixture()
    def synthetic_page(self):
        """A 64x64 RGB PageImage with a mid-tone gradient.

        Built with the actual PageImage signature from
        ``src/preprocessing/pdf_processor.py`` (no ``mime_type`` kwarg).
        """
        import io

        from PIL import Image

        from src.preprocessing.pdf_processor import PageImage

        arr = np.tile(np.linspace(20, 230, 64, dtype=np.uint8), (64, 1))
        img = Image.fromarray(arr, mode="L").convert("RGB")
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        png_bytes = buf.getvalue()
        return PageImage(
            page_number=1,
            image_bytes=png_bytes,
            width=64,
            height=64,
            dpi=72,
            orientation="portrait",
            original_width_pts=64.0,
            original_height_pts=64.0,
            has_text=True,
            has_images=False,
            rotation=0,
        )

    def test_default_mode_runs_clahe(self, enhancer, synthetic_page) -> None:
        from src.preprocessing.image_enhancer import EnhancementType

        result = enhancer.enhance(synthetic_page)
        assert EnhancementType.CLAHE in result.enhancements_applied
        assert EnhancementType.BINARIZATION not in result.enhancements_applied

    def test_fax_mode_binarizes_and_skips_clahe(
        self, enhancer, synthetic_page
    ) -> None:
        from src.preprocessing.image_enhancer import EnhancementType

        result = enhancer.enhance(synthetic_page, modes=["printed", "fax"])
        assert EnhancementType.BINARIZATION in result.enhancements_applied
        assert EnhancementType.MORPHOLOGICAL in result.enhancements_applied
        # CLAHE is wrong for 1-bit fax — must be skipped.
        assert EnhancementType.CLAHE not in result.enhancements_applied
        # Non-Local-Means denoise is also skipped on fax (binarization
        # collapses noise anyway).
        assert EnhancementType.DENOISE not in result.enhancements_applied

    def test_handwritten_mode_skips_clahe(
        self, enhancer, synthetic_page
    ) -> None:
        from src.preprocessing.image_enhancer import EnhancementType

        result = enhancer.enhance(synthetic_page, modes=["printed", "handwritten"])
        assert EnhancementType.CLAHE not in result.enhancements_applied
        # Denoise still runs (just with gentler strength).
        assert EnhancementType.DENOISE in result.enhancements_applied

    def test_visual_mode_skips_clahe_and_runs_minimal_path(
        self, enhancer, synthetic_page
    ) -> None:
        from src.preprocessing.image_enhancer import EnhancementType

        result = enhancer.enhance(synthetic_page, modes=["printed", "visual"])
        assert EnhancementType.CLAHE not in result.enhancements_applied


class TestAllModesConstant:
    def test_all_modes_includes_every_named_constant(self) -> None:
        # Smoke test: ensure no mode constant slips out of ALL_MODES.
        for m in (
            MODE_PRINTED,
            MODE_HANDWRITTEN,
            MODE_TABLE,
            MODE_FORM,
            MODE_FAX,
            MODE_VISUAL,
        ):
            assert m in ALL_MODES
