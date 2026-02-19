"""
Unit tests for preprocessing data classes in pdf_processor.py.

Tests cover:
- DocumentOrientation enum values
- PDFMetadata creation and serialisation (to_dict)
- PageImage creation, computed properties (base64_encoded, data_uri, size_kb),
  and serialisation (to_dict)
- ProcessingResult: page_count, total_size_kb, get_page, to_dict
- Error hierarchy (PDFProcessingError and subclasses)
"""

import base64
import io
from datetime import datetime
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from src.preprocessing.pdf_processor import (
    DocumentOrientation,
    PageImage,
    PDFCorruptionError,
    PDFEncryptionError,
    PDFMetadata,
    PDFPageLimitError,
    PDFProcessingError,
    PDFSizeError,
    PDFValidationError,
    ProcessingResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_metadata(**overrides) -> PDFMetadata:
    """Return a minimal PDFMetadata with sensible defaults."""
    defaults = dict(
        file_path=Path("/tmp/test.pdf"),
        file_name="test.pdf",
        file_size_bytes=1024,
        file_hash="abc123hash",
        page_count=2,
        title=None,
        author=None,
        subject=None,
        keywords=None,
        creator=None,
        producer=None,
        creation_date=None,
        modification_date=None,
        pdf_version="1.7",
        is_encrypted=False,
        has_forms=False,
        has_annotations=False,
        processing_id="proc-001",
    )
    defaults.update(overrides)
    return PDFMetadata(**defaults)


def _make_page_image(**overrides) -> PageImage:
    """Return a minimal PageImage with a tiny PNG payload."""
    # 1x1 white PNG
    img_bytes = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
        b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
        b"\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00"
        b"\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    defaults = dict(
        page_number=1,
        image_bytes=img_bytes,
        width=100,
        height=150,
        dpi=300,
        orientation=DocumentOrientation.PORTRAIT,
        original_width_pts=24.0,
        original_height_pts=36.0,
        has_text=True,
        has_images=False,
        rotation=0,
    )
    defaults.update(overrides)
    return PageImage(**defaults)


# ──────────────────────────────────────────────────────────────────
# DocumentOrientation
# ──────────────────────────────────────────────────────────────────


class TestDocumentOrientation:
    """Tests for DocumentOrientation enum."""

    def test_portrait_value(self):
        assert DocumentOrientation.PORTRAIT.value == "portrait"

    def test_landscape_value(self):
        assert DocumentOrientation.LANDSCAPE.value == "landscape"

    def test_square_value(self):
        assert DocumentOrientation.SQUARE.value == "square"


# ──────────────────────────────────────────────────────────────────
# PDFMetadata
# ──────────────────────────────────────────────────────────────────


class TestPDFMetadata:
    """Tests for PDFMetadata frozen dataclass."""

    def test_creation_defaults(self):
        meta = _make_metadata()
        assert meta.file_name == "test.pdf"
        assert meta.file_size_bytes == 1024
        assert meta.page_count == 2
        assert meta.is_encrypted is False

    def test_creation_with_optional_fields(self):
        now = datetime.now()
        meta = _make_metadata(
            title="Claims Form",
            author="Admin",
            creation_date=now,
        )
        assert meta.title == "Claims Form"
        assert meta.author == "Admin"
        assert meta.creation_date == now

    def test_to_dict_contains_expected_keys(self):
        meta = _make_metadata(title="My PDF")
        d = meta.to_dict()
        assert isinstance(d, dict)
        assert d["file_name"] == "test.pdf"
        assert d["title"] == "My PDF"
        assert "file_hash" in d
        assert "page_count" in d
        assert "processing_id" in d


# ──────────────────────────────────────────────────────────────────
# PageImage
# ──────────────────────────────────────────────────────────────────


class TestPageImage:
    """Tests for PageImage frozen dataclass and its computed properties."""

    def test_creation(self):
        page = _make_page_image()
        assert page.page_number == 1
        assert page.width == 100
        assert page.height == 150
        assert page.dpi == 300
        assert page.orientation == DocumentOrientation.PORTRAIT

    def test_base64_encoded_property(self):
        page = _make_page_image()
        encoded = page.base64_encoded
        # Must be a valid base64 string that decodes back to the original bytes
        decoded = base64.b64decode(encoded)
        assert decoded == page.image_bytes

    def test_data_uri_property(self):
        page = _make_page_image()
        uri = page.data_uri
        assert uri.startswith("data:image/")
        assert ";base64," in uri
        assert page.base64_encoded in uri

    def test_size_kb_property(self):
        page = _make_page_image()
        expected_kb = len(page.image_bytes) / 1024
        assert abs(page.size_kb - expected_kb) < 0.01

    def test_text_content_default(self):
        page = _make_page_image()
        assert page.text_content == ""

    def test_to_dict(self):
        page = _make_page_image()
        d = page.to_dict()
        assert isinstance(d, dict)
        assert d["page_number"] == 1
        assert d["width"] == 100
        assert d["height"] == 150
        assert "orientation" in d


# ──────────────────────────────────────────────────────────────────
# ProcessingResult
# ──────────────────────────────────────────────────────────────────


class TestProcessingResult:
    """Tests for ProcessingResult dataclass."""

    def _make_result(self, num_pages: int = 2) -> ProcessingResult:
        meta = _make_metadata(page_count=num_pages)
        pages = [_make_page_image(page_number=i + 1) for i in range(num_pages)]
        return ProcessingResult(
            metadata=meta,
            pages=pages,
            processing_time_ms=500,
            warnings=[],
            temp_dir=None,
        )

    def test_page_count_property(self):
        result = self._make_result(3)
        assert result.page_count == 3

    def test_total_size_kb_property(self):
        result = self._make_result(2)
        expected = sum(p.size_kb for p in result.pages)
        assert abs(result.total_size_kb - expected) < 0.01

    def test_get_page_valid(self):
        result = self._make_result(3)
        page = result.get_page(2)
        assert page is not None
        assert page.page_number == 2

    def test_get_page_invalid_returns_none(self):
        result = self._make_result(2)
        assert result.get_page(99) is None

    def test_to_dict(self):
        result = self._make_result(1)
        d = result.to_dict()
        assert isinstance(d, dict)
        assert "metadata" in d
        assert "pages" in d
        assert d["processing_time_ms"] == 500


# ──────────────────────────────────────────────────────────────────
# Error Hierarchy
# ──────────────────────────────────────────────────────────────────


class TestPDFErrorHierarchy:
    """Tests for the PDF error class hierarchy."""

    def test_processing_error_is_exception(self):
        err = PDFProcessingError("something broke")
        assert isinstance(err, Exception)

    def test_validation_error_inherits_processing_error(self):
        err = PDFValidationError("bad input")
        assert isinstance(err, PDFProcessingError)

    def test_encryption_error_inherits_processing_error(self):
        err = PDFEncryptionError("encrypted")
        assert isinstance(err, PDFProcessingError)

    def test_corruption_error_inherits_processing_error(self):
        err = PDFCorruptionError("corrupt file")
        assert isinstance(err, PDFProcessingError)

    def test_size_and_page_limit_errors_inherit_processing_error(self):
        size_err = PDFSizeError("too big")
        page_err = PDFPageLimitError("too many pages")
        assert isinstance(size_err, PDFProcessingError)
        assert isinstance(page_err, PDFProcessingError)
