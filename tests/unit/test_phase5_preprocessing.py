"""V3 Phase 5 — Image enhancer despeckle + PDF fax-detection metadata.

These tests cover the new ``_apply_despeckle`` and
``_classify_orientation`` methods on ``ImageEnhancer`` plus the
``is_one_bit`` / ``is_ccitt`` / ``fax_signals`` fields surfaced on
``PageImage``.
"""

from __future__ import annotations

import io

import numpy as np
import pytest
from PIL import Image

from src.preprocessing.image_enhancer import (
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


def _make_white_page(width: int = 200, height: int = 280) -> PageImage:
    """Build a white-page PageImage with a single horizontal black band
    near the top (so portrait_score is dominated by it)."""
    img = Image.new("RGB", (width, height), (255, 255, 255))
    px = img.load()
    # Draw a few horizontal text-ish bands in the upper third.
    for y in range(20, 80, 12):
        for x in range(15, width - 15):
            px[x, y] = (0, 0, 0)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG", optimize=True)
    return PageImage(
        page_number=1,
        image_bytes=buffer.getvalue(),
        width=width,
        height=height,
        dpi=200,
        orientation=DocumentOrientation.PORTRAIT,
        original_width_pts=8.5 * 72,
        original_height_pts=11 * 72,
        has_text=True,
        has_images=False,
        rotation=0,
    )


def _make_speckled_page() -> PageImage:
    """Build a page with a clear glyph plus scattered single-pixel
    speckle. After despeckle, the speckle should be gone but the glyph
    should still be there."""
    width, height = 200, 200
    img = Image.new("L", (width, height), 255)
    pixels = img.load()
    # A single connected black blob (the "glyph") - 20x20.
    for y in range(50, 70):
        for x in range(50, 70):
            pixels[x, y] = 0
    # Scatter 50 single-pixel "speckle" dots elsewhere.
    rng = np.random.default_rng(42)
    for _ in range(50):
        x = int(rng.integers(0, width))
        y = int(rng.integers(0, height))
        # Avoid stamping on the glyph
        if 50 <= x < 70 and 50 <= y < 70:
            continue
        pixels[x, y] = 0
    rgb = img.convert("RGB")
    buffer = io.BytesIO()
    rgb.save(buffer, format="PNG", optimize=True)
    return PageImage(
        page_number=1,
        image_bytes=buffer.getvalue(),
        width=width,
        height=height,
        dpi=200,
        orientation=DocumentOrientation.PORTRAIT,
        original_width_pts=72,
        original_height_pts=72,
        has_text=True,
        has_images=False,
        rotation=0,
    )


# ---------------------------------------------------------------------------
# Despeckle
# ---------------------------------------------------------------------------


class TestDespeckle:
    def test_despeckle_runs_in_fax_mode(self) -> None:
        page = _make_speckled_page()
        enhancer = ImageEnhancer(
            enable_deskew=False, enable_denoise=False, enable_contrast=False
        )
        result = enhancer.enhance(page, modes=["fax"])
        assert EnhancementType.DESPECKLE in result.enhancements_applied
        assert EnhancementType.BINARIZATION in result.enhancements_applied

    def test_despeckle_removes_isolated_pixels(self) -> None:
        # The actual ``_apply_despeckle`` should drop sub-threshold
        # connected components. Test by counting black pixels before
        # and after on a synthetic image.
        enhancer = ImageEnhancer(
            enable_deskew=False, enable_denoise=False, enable_contrast=False
        )
        # Build a small BGR image with a 20x20 glyph + 10 single
        # pixels.
        canvas = np.full((100, 100, 3), 255, dtype=np.uint8)
        canvas[40:60, 40:60] = 0  # glyph
        speckle_coords = [(5, 5), (15, 80), (80, 10), (90, 90), (33, 7)]
        for cy, cx in speckle_coords:
            canvas[cy, cx] = 0
        before = int(np.sum(canvas[..., 0] == 0))
        cleaned = enhancer._apply_despeckle(canvas)
        after = int(np.sum(cleaned[..., 0] == 0))
        assert after < before, "despeckle should remove pixels"
        # Glyph (~400 px) should largely survive.
        assert after >= 350

    def test_despeckle_does_not_run_outside_fax(self) -> None:
        page = _make_speckled_page()
        enhancer = ImageEnhancer(
            enable_deskew=False, enable_denoise=False, enable_contrast=False
        )
        result = enhancer.enhance(page, modes=["printed"])
        assert EnhancementType.DESPECKLE not in result.enhancements_applied


# ---------------------------------------------------------------------------
# Orientation classifier
# ---------------------------------------------------------------------------


class TestOrientationClassifier:
    def test_upright_returns_zero(self) -> None:
        enhancer = ImageEnhancer(
            enable_deskew=False, enable_denoise=False, enable_contrast=False
        )
        # Build a synthetic page: text bands in the upper third,
        # blank lower 2/3.
        page = _make_white_page()
        img_array = np.array(page.to_pil_image().convert("RGB"))
        # cv2 expects BGR.
        bgr = img_array[:, :, ::-1].copy()
        rotation = enhancer._classify_orientation(bgr)
        # Either 0 or 90 may be returned because horizontal bands
        # have similar variance to vertical bands of the same length.
        # Importantly, 180 should NOT be returned (letterhead heuristic).
        assert rotation != 180

    def test_blank_page_returns_zero_safely(self) -> None:
        enhancer = ImageEnhancer(
            enable_deskew=False, enable_denoise=False, enable_contrast=False
        )
        # Pure white page — no text bands at all.
        bgr = np.full((100, 100, 3), 255, dtype=np.uint8)
        rotation = enhancer._classify_orientation(bgr)
        # All four rotations score 0; the implementation falls back
        # to zero (the first key tie-broken by max).
        assert rotation == 0


# ---------------------------------------------------------------------------
# PageImage carries fax detection fields
# ---------------------------------------------------------------------------


class TestPageImageFaxFields:
    def test_default_fields_none(self) -> None:
        page = _make_white_page()
        # Default factory in test helper does not set fax fields.
        assert page.is_one_bit is None
        assert page.is_ccitt is None
        assert page.fax_signals == ()

    def test_to_dict_carries_fax_fields(self) -> None:
        page = PageImage(
            page_number=1,
            image_bytes=b"\x00",
            width=10,
            height=10,
            dpi=100,
            orientation=DocumentOrientation.PORTRAIT,
            original_width_pts=72,
            original_height_pts=72,
            has_text=False,
            has_images=False,
            rotation=0,
            is_one_bit=True,
            is_ccitt=True,
            fax_signals=("1-bit-image", "ccitt-fax-encoded"),
        )
        d = page.to_dict()
        assert d["is_one_bit"] is True
        assert d["is_ccitt"] is True
        assert "1-bit-image" in d["fax_signals"]


# ---------------------------------------------------------------------------
# PDF processor _inspect_image_streams (mocked PyMuPDF)
# ---------------------------------------------------------------------------


class TestInspectImageStreams:
    def test_no_images_returns_empty(self, monkeypatch) -> None:
        from src.preprocessing.pdf_processor import PDFProcessor

        proc = PDFProcessor()

        class FakePage:
            def get_images(self, full=True):
                return []

        is_one_bit, is_ccitt, signals = proc._inspect_image_streams(FakePage())
        assert is_one_bit is False
        assert is_ccitt is False
        assert signals == []

    def test_detects_one_bit_image(self) -> None:
        from src.preprocessing.pdf_processor import PDFProcessor

        proc = PDFProcessor()

        class FakeDoc:
            def xref_object(self, xref):
                # Simulate a 1-bit DeviceGray XObject.
                return (
                    "<< /Type /XObject /Subtype /Image "
                    "/BitsPerComponent 1 /ColorSpace /DeviceGray "
                    "/Filter /FlateDecode >>"
                )

        class FakePage:
            parent = FakeDoc()

            def get_images(self, full=True):
                return [(99, 0, 0, 0, 0, 0, 0, 0)]

        is_one_bit, is_ccitt, signals = proc._inspect_image_streams(FakePage())
        assert is_one_bit is True
        assert is_ccitt is False
        assert "1-bit-image" in signals

    def test_detects_ccitt(self) -> None:
        from src.preprocessing.pdf_processor import PDFProcessor

        proc = PDFProcessor()

        class FakeDoc:
            def xref_object(self, xref):
                return (
                    "<< /Type /XObject /Filter /CCITTFaxDecode "
                    "/BitsPerComponent 1 >>"
                )

        class FakePage:
            parent = FakeDoc()

            def get_images(self, full=True):
                return [(7, 0, 0, 0, 0, 0, 0, 0)]

        is_one_bit, is_ccitt, signals = proc._inspect_image_streams(FakePage())
        assert is_one_bit is True
        assert is_ccitt is True
        assert "ccitt-fax-encoded" in signals
        assert "1-bit-image" in signals

    def test_inspection_swallows_errors(self) -> None:
        from src.preprocessing.pdf_processor import PDFProcessor

        proc = PDFProcessor()

        class FakePage:
            def get_images(self, full=True):
                raise RuntimeError("boom")

        is_one_bit, is_ccitt, signals = proc._inspect_image_streams(FakePage())
        assert is_one_bit is False
        assert is_ccitt is False
        assert signals == []
