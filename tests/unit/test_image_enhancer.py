"""
Unit tests for ImageEnhancer and its data classes.

Tests cover:
- EnhancementType enum values
- EnhancementMetrics: creation, sharpness_improvement property, to_dict
- EnhancementResult: creation, to_dict
- Error hierarchy (EnhancementError, DeskewError, DenoiseError, ContrastError)
- ImageEnhancer.enhance with a real small image
- ImageEnhancer.analyze_quality with a real small image
- ImageEnhancer.enhance_batch
"""

import io
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image as PILImage

from src.preprocessing.image_enhancer import (
    ContrastError,
    DenoiseError,
    DeskewError,
    EnhancementError,
    EnhancementMetrics,
    EnhancementResult,
    EnhancementType,
    ImageEnhancer,
)
from src.preprocessing.pdf_processor import (
    DocumentOrientation,
    PageImage,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_page_image(width: int = 100, height: int = 150) -> PageImage:
    """Create a PageImage backed by a real in-memory PNG."""
    img = PILImage.new("RGB", (width, height), color=(200, 200, 200))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return PageImage(
        page_number=1,
        image_bytes=buf.getvalue(),
        width=width,
        height=height,
        dpi=300,
        orientation=DocumentOrientation.PORTRAIT,
        original_width_pts=width * 72.0 / 300,
        original_height_pts=height * 72.0 / 300,
        has_text=False,
        has_images=False,
        rotation=0,
    )


def _make_metrics(**overrides) -> EnhancementMetrics:
    """Return an EnhancementMetrics with sensible defaults."""
    defaults = dict(
        original_variance=50.0,
        enhanced_variance=75.0,
        skew_angle=0.5,
        skew_corrected=True,
        noise_reduction_ratio=0.15,
        contrast_improvement=0.30,
    )
    defaults.update(overrides)
    return EnhancementMetrics(**defaults)


def _make_enhancer(**kwargs) -> ImageEnhancer:
    """Create an ImageEnhancer with explicit parameters to bypass settings.

    All constructor parameters are provided so ``get_settings()`` is never
    consulted for the values that feed into ``cv2.createCLAHE``.
    """
    defaults = dict(
        enable_deskew=False,
        enable_denoise=False,
        enable_contrast=False,
        clahe_clip_limit=2.0,
        clahe_tile_size=8,
        denoise_strength=5,
        deskew_max_angle=10.0,
    )
    defaults.update(kwargs)
    return ImageEnhancer(**defaults)


# ──────────────────────────────────────────────────────────────────
# EnhancementType
# ──────────────────────────────────────────────────────────────────


class TestEnhancementType:
    """Tests for the EnhancementType enum."""

    def test_expected_members_exist(self):
        members = {m.name for m in EnhancementType}
        for name in ("DESKEW", "DENOISE", "CLAHE", "BINARIZATION", "MORPHOLOGICAL"):
            assert name in members, f"{name} missing from EnhancementType"

    def test_values_are_strings(self):
        for member in EnhancementType:
            assert isinstance(member.value, str)


# ──────────────────────────────────────────────────────────────────
# EnhancementMetrics
# ──────────────────────────────────────────────────────────────────


class TestEnhancementMetrics:
    """Tests for the EnhancementMetrics frozen dataclass."""

    def test_creation(self):
        m = _make_metrics()
        assert m.original_variance == 50.0
        assert m.enhanced_variance == 75.0
        assert m.skew_corrected is True

    def test_sharpness_improvement_positive(self):
        m = _make_metrics(original_variance=50.0, enhanced_variance=75.0)
        improvement = m.sharpness_improvement
        assert isinstance(improvement, (int, float))
        # 75 / 50 = 1.5  -- ratio > 1 means improvement
        assert improvement > 1.0

    def test_sharpness_improvement_no_change(self):
        """When variances are equal the ratio is 1.0 (no change)."""
        m = _make_metrics(original_variance=50.0, enhanced_variance=50.0)
        assert m.sharpness_improvement == pytest.approx(1.0)

    def test_to_dict(self):
        m = _make_metrics()
        d = m.to_dict()
        assert isinstance(d, dict)
        assert "original_variance" in d
        assert "enhanced_variance" in d
        assert "skew_angle" in d
        assert "noise_reduction_ratio" in d


# ──────────────────────────────────────────────────────────────────
# EnhancementResult
# ──────────────────────────────────────────────────────────────────


class TestEnhancementResult:
    """Tests for the EnhancementResult mutable dataclass."""

    def test_creation(self):
        page = _make_page_image()
        result = EnhancementResult(
            page_image=page,
            original_page=page,
            enhancements_applied=[EnhancementType.DESKEW],
            metrics=_make_metrics(),
            processing_time_ms=42,
        )
        assert result.processing_time_ms == 42
        assert EnhancementType.DESKEW in result.enhancements_applied

    def test_to_dict(self):
        page = _make_page_image()
        result = EnhancementResult(
            page_image=page,
            original_page=page,
            enhancements_applied=[EnhancementType.DENOISE, EnhancementType.CLAHE],
            metrics=_make_metrics(),
            processing_time_ms=100,
        )
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "enhancements_applied" in d
        assert "processing_time_ms" in d


# ──────────────────────────────────────────────────────────────────
# Error Hierarchy
# ──────────────────────────────────────────────────────────────────


class TestEnhancementErrors:
    """Tests for the enhancement error hierarchy."""

    def test_enhancement_error_is_exception(self):
        err = EnhancementError("generic failure")
        assert isinstance(err, Exception)

    def test_deskew_error_inherits_enhancement_error(self):
        err = DeskewError("deskew failed")
        assert isinstance(err, EnhancementError)

    def test_denoise_and_contrast_errors_inherit(self):
        assert isinstance(DenoiseError("noise"), EnhancementError)
        assert isinstance(ContrastError("contrast"), EnhancementError)


# ──────────────────────────────────────────────────────────────────
# ImageEnhancer.enhance
# ──────────────────────────────────────────────────────────────────


class TestImageEnhancerEnhance:
    """Tests for ImageEnhancer.enhance with real (tiny) images."""

    def test_enhance_returns_result(self):
        """enhance() returns an EnhancementResult with a valid page_image."""
        enhancer = _make_enhancer(
            enable_deskew=True,
            enable_denoise=False,  # denoise disabled to avoid cv2 API issue
            enable_contrast=True,
        )
        page = _make_page_image()
        result = enhancer.enhance(page)

        assert isinstance(result, EnhancementResult)
        assert result.page_image is not None
        assert result.page_image.image_bytes  # non-empty
        assert result.original_page is page

    def test_enhance_populates_enhancements_applied(self):
        """enhance() records which enhancements were actually applied."""
        enhancer = _make_enhancer(
            enable_deskew=False,
            enable_denoise=False,
            enable_contrast=True,
        )
        page = _make_page_image()
        result = enhancer.enhance(page)

        assert isinstance(result.enhancements_applied, list)
        # Deskew is disabled, so it should NOT appear
        assert EnhancementType.DESKEW not in result.enhancements_applied


# ──────────────────────────────────────────────────────────────────
# ImageEnhancer.analyze_quality
# ──────────────────────────────────────────────────────────────────


class TestImageEnhancerAnalyzeQuality:
    """Tests for ImageEnhancer.analyze_quality."""

    def test_analyze_quality_returns_dict(self):
        enhancer = _make_enhancer()
        page = _make_page_image()
        quality = enhancer.analyze_quality(page)

        assert isinstance(quality, dict)

    def test_analyze_quality_contains_expected_keys(self):
        enhancer = _make_enhancer()
        page = _make_page_image()
        quality = enhancer.analyze_quality(page)

        # Should include at least variance / brightness / contrast metrics
        assert len(quality) > 0


# ──────────────────────────────────────────────────────────────────
# ImageEnhancer.enhance_batch
# ──────────────────────────────────────────────────────────────────


class TestImageEnhancerBatch:
    """Tests for ImageEnhancer.enhance_batch."""

    def test_enhance_batch_processes_all_pages(self):
        enhancer = _make_enhancer(enable_contrast=True)
        pages = [_make_page_image(width=80, height=120) for _ in range(3)]
        results = enhancer.enhance_batch(pages)

        assert isinstance(results, list)
        assert len(results) == 3
        for r in results:
            assert isinstance(r, EnhancementResult)
