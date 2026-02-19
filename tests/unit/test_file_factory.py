"""
Unit tests for FileProcessorFactory (file_factory.py) and
BaseFileProcessor / FileFormat (base_processor.py).

Tests cover:
- FileFormat enum values
- SUPPORTED_EXTENSIONS set
- BaseFileProcessor._detect_orientation static method
- FileProcessorFactory: is_supported, supported_extensions,
  get_processor routing (mocked), UnsupportedFormatError
- Error hierarchy (UnsupportedFormatError, FileValidationError)
"""

import csv
import hashlib
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from src.preprocessing.base_processor import (
    BaseFileProcessor,
    FileFormat,
    FileValidationError,
    SUPPORTED_EXTENSIONS,
    UnsupportedFormatError,
)
from src.preprocessing.file_factory import FileProcessorFactory
from src.preprocessing.pdf_processor import DocumentOrientation


# ──────────────────────────────────────────────────────────────────
# Fixtures
# ──────────────────────────────────────────────────────────────────


@pytest.fixture
def tmp_dir():
    """Provide a temporary directory."""
    with tempfile.TemporaryDirectory() as d:
        yield Path(d)


@pytest.fixture
def sample_png(tmp_dir: Path) -> Path:
    """Create a sample PNG file."""
    path = tmp_dir / "test.png"
    img = Image.new("RGB", (200, 100), color="white")
    img.save(str(path), format="PNG")
    return path


@pytest.fixture
def sample_jpg(tmp_dir: Path) -> Path:
    """Create a sample JPG file."""
    path = tmp_dir / "test.jpg"
    img = Image.new("RGB", (200, 100), color="blue")
    img.save(str(path), format="JPEG")
    return path


@pytest.fixture
def sample_csv(tmp_dir: Path) -> Path:
    """Create a sample CSV file."""
    path = tmp_dir / "test.csv"
    with path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Name", "Age"])
        writer.writerow(["Alice", "30"])
    return path


# ──────────────────────────────────────────────────────────────────
# FileFormat Enum
# ──────────────────────────────────────────────────────────────────


class TestFileFormat:
    """Tests for the FileFormat string enum."""

    def test_pdf_value(self):
        assert FileFormat.PDF.value == "pdf"

    def test_image_values(self):
        assert FileFormat.PNG.value == "png"
        assert FileFormat.JPG.value == "jpg"
        assert FileFormat.JPEG.value == "jpeg"
        assert FileFormat.TIFF.value == "tiff"
        assert FileFormat.BMP.value == "bmp"

    def test_edi_values(self):
        assert FileFormat.EDI.value == "edi"
        assert FileFormat.X12.value == "x12"
        assert FileFormat.EDI_835.value == "835"
        assert FileFormat.EDI_837.value == "837"


# ──────────────────────────────────────────────────────────────────
# SUPPORTED_EXTENSIONS
# ──────────────────────────────────────────────────────────────────


class TestSupportedExtensions:
    """Tests for the SUPPORTED_EXTENSIONS set."""

    def test_contains_common_extensions(self):
        for ext in (".pdf", ".png", ".jpg", ".csv", ".docx"):
            assert ext in SUPPORTED_EXTENSIONS, f"{ext} missing"

    def test_all_formats_have_extension(self):
        """Every FileFormat value should have a corresponding .ext entry."""
        for fmt in FileFormat:
            assert f".{fmt.value}" in SUPPORTED_EXTENSIONS


# ──────────────────────────────────────────────────────────────────
# BaseFileProcessor._detect_orientation
# ──────────────────────────────────────────────────────────────────


class TestDetectOrientation:
    """Tests for the static orientation detection helper."""

    def test_portrait(self):
        result = BaseFileProcessor._detect_orientation(100, 200)
        assert result == DocumentOrientation.PORTRAIT

    def test_landscape(self):
        result = BaseFileProcessor._detect_orientation(200, 100)
        assert result == DocumentOrientation.LANDSCAPE

    def test_square(self):
        result = BaseFileProcessor._detect_orientation(100, 100)
        assert result == DocumentOrientation.SQUARE


# ──────────────────────────────────────────────────────────────────
# FileProcessorFactory
# ──────────────────────────────────────────────────────────────────


class TestFileProcessorFactory:
    """Tests for FileProcessorFactory routing and helpers."""

    def test_is_supported_true(self, sample_png: Path):
        factory = FileProcessorFactory()
        assert factory.is_supported(sample_png) is True

    def test_is_supported_false(self, tmp_dir: Path):
        path = tmp_dir / "unknown.xyz"
        factory = FileProcessorFactory()
        assert factory.is_supported(path) is False

    def test_supported_extensions_returns_set(self):
        exts = FileProcessorFactory.supported_extensions()
        assert isinstance(exts, set)
        assert ".pdf" in exts
        assert ".png" in exts

    def test_get_processor_routes_image(self, sample_png: Path):
        """Image files get routed to the image processor."""
        factory = FileProcessorFactory()
        processor = factory.get_processor(sample_png)
        # Should be an instance of BaseFileProcessor (any concrete subclass)
        assert isinstance(processor, BaseFileProcessor)

    def test_get_processor_routes_csv(self, sample_csv: Path):
        """CSV files get routed to the spreadsheet processor."""
        factory = FileProcessorFactory()
        processor = factory.get_processor(sample_csv)
        assert isinstance(processor, BaseFileProcessor)

    def test_get_processor_caches(self, sample_png: Path, sample_jpg: Path):
        """Same processor category returns the same cached instance."""
        factory = FileProcessorFactory()
        p1 = factory.get_processor(sample_png)
        p2 = factory.get_processor(sample_jpg)
        assert p1 is p2

    def test_get_processor_raises_for_unsupported(self, tmp_dir: Path):
        path = tmp_dir / "document.xyz"
        path.write_text("unsupported content")
        factory = FileProcessorFactory()
        with pytest.raises(UnsupportedFormatError):
            factory.get_processor(path)

    def test_process_delegates_to_processor(self, sample_png: Path):
        """Factory.process() returns a valid ProcessingResult."""
        factory = FileProcessorFactory()
        result = factory.process(sample_png)
        assert result.page_count >= 1
        assert result.metadata.file_name == "test.png"


# ──────────────────────────────────────────────────────────────────
# Error Hierarchy
# ──────────────────────────────────────────────────────────────────


class TestFileErrors:
    """Tests for UnsupportedFormatError and FileValidationError."""

    def test_unsupported_format_error_is_exception(self):
        err = UnsupportedFormatError(".xyz")
        assert isinstance(err, Exception)
        assert ".xyz" in str(err)

    def test_file_validation_error_is_exception(self):
        err = FileValidationError("file not found")
        assert isinstance(err, Exception)
        assert "file not found" in str(err)
