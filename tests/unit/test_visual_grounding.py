"""
Unit tests for Phase 1A: Visual Grounding & Bounding Boxes.

Tests BoundingBoxCoords dataclass, FieldMetadata bbox integration,
spatial pattern detection, JSON/Excel export of bbox data, and
the extractor _parse_bbox static method.
"""

import json
import tempfile
from pathlib import Path
from typing import Any

import pytest

from src.pipeline.state import (
    BoundingBoxCoords,
    ConfidenceLevel,
    ExtractionStatus,
    FieldMetadata,
    deserialize_field_metadata,
)
from src.validation.pattern_detector import (
    HallucinationPattern,
    HallucinationPatternDetector,
    PatternSeverity,
)
from src.export.json_exporter import (
    ExportFormat,
    JSONExportConfig,
    JSONExporter,
    export_to_json,
)


# ──────────────────────────────────────────────────────────────────
# BoundingBoxCoords Tests
# ──────────────────────────────────────────────────────────────────


class TestBoundingBoxCoords:
    """Tests for BoundingBoxCoords dataclass."""

    def test_basic_construction(self):
        """Test basic construction with required fields."""
        bbox = BoundingBoxCoords(x=0.1, y=0.2, width=0.3, height=0.04)
        assert bbox.x == 0.1
        assert bbox.y == 0.2
        assert bbox.width == 0.3
        assert bbox.height == 0.04
        assert bbox.page == 1  # default
        assert bbox.pixel_x == 0  # default
        assert bbox.pixel_y == 0
        assert bbox.pixel_width == 0
        assert bbox.pixel_height == 0

    def test_full_construction_with_pixels(self):
        """Test construction with all fields including pixel coords."""
        bbox = BoundingBoxCoords(
            x=0.1, y=0.2, width=0.3, height=0.04,
            page=2,
            pixel_x=100, pixel_y=200,
            pixel_width=300, pixel_height=40,
        )
        assert bbox.page == 2
        assert bbox.pixel_x == 100
        assert bbox.pixel_y == 200
        assert bbox.pixel_width == 300
        assert bbox.pixel_height == 40

    def test_frozen_immutability(self):
        """Test that BoundingBoxCoords is frozen (immutable)."""
        bbox = BoundingBoxCoords(x=0.1, y=0.2, width=0.3, height=0.04)
        with pytest.raises(AttributeError):
            bbox.x = 0.5  # type: ignore[misc]

    def test_to_dict(self):
        """Test conversion to dictionary."""
        bbox = BoundingBoxCoords(
            x=0.12, y=0.05, width=0.25, height=0.03, page=1,
        )
        d = bbox.to_dict()
        assert d["x"] == 0.12
        assert d["y"] == 0.05
        assert d["width"] == 0.25
        assert d["height"] == 0.03
        assert d["page"] == 1
        assert "pixel_x" in d
        assert "pixel_y" in d

    def test_from_dict_with_width_height(self):
        """Test creation from dict using width/height keys."""
        data = {"x": 0.1, "y": 0.2, "width": 0.3, "height": 0.04, "page": 2}
        bbox = BoundingBoxCoords.from_dict(data)
        assert bbox.x == 0.1
        assert bbox.y == 0.2
        assert bbox.width == 0.3
        assert bbox.height == 0.04
        assert bbox.page == 2

    def test_from_dict_with_w_h_shorthand(self):
        """Test creation from dict using w/h shorthand keys (VLM format)."""
        data = {"x": 0.15, "y": 0.25, "w": 0.35, "h": 0.05}
        bbox = BoundingBoxCoords.from_dict(data)
        assert bbox.x == 0.15
        assert bbox.y == 0.25
        assert bbox.width == 0.35
        assert bbox.height == 0.05

    def test_from_dict_defaults(self):
        """Test from_dict with missing keys falls back to defaults."""
        data: dict[str, Any] = {}
        bbox = BoundingBoxCoords.from_dict(data)
        assert bbox.x == 0.0
        assert bbox.y == 0.0
        assert bbox.width == 0.0
        assert bbox.height == 0.0
        assert bbox.page == 1

    def test_from_dict_with_pixel_coords(self):
        """Test from_dict preserves pixel coordinates."""
        data = {
            "x": 0.1, "y": 0.2, "width": 0.3, "height": 0.04,
            "pixel_x": 85, "pixel_y": 170, "pixel_width": 255, "pixel_height": 34,
        }
        bbox = BoundingBoxCoords.from_dict(data)
        assert bbox.pixel_x == 85
        assert bbox.pixel_y == 170
        assert bbox.pixel_width == 255
        assert bbox.pixel_height == 34

    def test_from_normalized_basic(self):
        """Test from_normalized without page dimensions."""
        bbox = BoundingBoxCoords.from_normalized(x=0.1, y=0.2, w=0.3, h=0.04)
        assert bbox.x == 0.1
        assert bbox.y == 0.2
        assert bbox.width == 0.3
        assert bbox.height == 0.04
        assert bbox.pixel_x == 0  # No page dimensions provided
        assert bbox.pixel_y == 0

    def test_from_normalized_with_page_dimensions(self):
        """Test from_normalized computes pixel coordinates."""
        bbox = BoundingBoxCoords.from_normalized(
            x=0.1, y=0.2, w=0.3, h=0.04,
            page=1,
            page_width_px=850,
            page_height_px=1100,
        )
        assert bbox.pixel_x == 85  # 0.1 * 850
        assert bbox.pixel_y == 220  # 0.2 * 1100
        assert bbox.pixel_width == 255  # 0.3 * 850
        assert bbox.pixel_height == 44  # 0.04 * 1100

    def test_from_normalized_clamps_to_valid_range(self):
        """Test from_normalized clamps out-of-range values."""
        bbox = BoundingBoxCoords.from_normalized(x=-0.1, y=1.5, w=2.0, h=0.04)
        assert bbox.x == 0.0  # Clamped from -0.1
        assert bbox.y == 1.0  # Clamped from 1.5
        assert bbox.width == 1.0  # Clamped to max available (1.0 - 0.0)
        # h clamped to min(1.0 - 1.0, 0.04) = 0.0 since y was clamped to 1.0
        assert bbox.height == 0.0

    def test_from_normalized_width_does_not_exceed_page(self):
        """Test width is clamped so x + width <= 1.0."""
        bbox = BoundingBoxCoords.from_normalized(x=0.8, y=0.1, w=0.5, h=0.1)
        assert bbox.x == 0.8
        assert bbox.width == pytest.approx(0.2)  # Clamped to 1.0 - 0.8

    def test_is_valid(self):
        """Test is_valid checks for non-zero dimensions."""
        valid = BoundingBoxCoords(x=0.1, y=0.2, width=0.3, height=0.04)
        assert valid.is_valid()

        zero_w = BoundingBoxCoords(x=0.1, y=0.2, width=0.0, height=0.04)
        assert not zero_w.is_valid()

        zero_h = BoundingBoxCoords(x=0.1, y=0.2, width=0.3, height=0.0)
        assert not zero_h.is_valid()

    def test_roundtrip_to_dict_from_dict(self):
        """Test to_dict -> from_dict roundtrip preserves data."""
        original = BoundingBoxCoords(
            x=0.12, y=0.34, width=0.56, height=0.08,
            page=3,
            pixel_x=102, pixel_y=374,
            pixel_width=476, pixel_height=88,
        )
        d = original.to_dict()
        restored = BoundingBoxCoords.from_dict(d)
        assert restored.x == original.x
        assert restored.y == original.y
        assert restored.width == original.width
        assert restored.height == original.height
        assert restored.page == original.page
        assert restored.pixel_x == original.pixel_x
        assert restored.pixel_y == original.pixel_y


# ──────────────────────────────────────────────────────────────────
# FieldMetadata bbox Integration Tests
# ──────────────────────────────────────────────────────────────────


class TestFieldMetadataBbox:
    """Tests for bbox integration in FieldMetadata."""

    def test_field_metadata_without_bbox(self):
        """Test FieldMetadata defaults to no bbox."""
        meta = FieldMetadata(
            field_name="patient_name",
            value="John Doe",
            confidence=0.95,
        )
        assert meta.bbox is None

    def test_field_metadata_with_bbox(self):
        """Test FieldMetadata with bbox attached."""
        bbox = BoundingBoxCoords(x=0.12, y=0.05, width=0.25, height=0.03)
        meta = FieldMetadata(
            field_name="patient_name",
            value="John Doe",
            confidence=0.95,
            bbox=bbox,
        )
        assert meta.bbox is not None
        assert meta.bbox.x == 0.12
        assert meta.bbox.width == 0.25

    def test_to_dict_includes_bbox_when_present(self):
        """Test to_dict includes bbox sub-dict when bbox is set."""
        bbox = BoundingBoxCoords(x=0.1, y=0.2, width=0.3, height=0.04, page=1)
        meta = FieldMetadata(
            field_name="dob",
            value="1990-01-15",
            confidence=0.88,
            bbox=bbox,
        )
        d = meta.to_dict()
        assert "bbox" in d
        assert d["bbox"]["x"] == 0.1
        assert d["bbox"]["width"] == 0.3
        assert d["bbox"]["page"] == 1

    def test_to_dict_excludes_bbox_when_none(self):
        """Test to_dict omits bbox key when bbox is None."""
        meta = FieldMetadata(
            field_name="dob",
            value="1990-01-15",
            confidence=0.88,
        )
        d = meta.to_dict()
        assert "bbox" not in d

    def test_deserialize_with_bbox(self):
        """Test deserialize_field_metadata restores bbox."""
        data = {
            "field_name": "patient_name",
            "value": "Jane Smith",
            "confidence": 0.92,
            "confidence_level": "high",
            "pass1_value": "Jane Smith",
            "pass2_value": "Jane Smith",
            "passes_agree": True,
            "location_hint": "Box 2",
            "validation_passed": True,
            "validation_errors": [],
            "source_page": 1,
            "is_hallucination_flag": False,
            "bbox": {
                "x": 0.15, "y": 0.08, "width": 0.30, "height": 0.03,
                "page": 1, "pixel_x": 128, "pixel_y": 88,
                "pixel_width": 255, "pixel_height": 33,
            },
        }
        meta = deserialize_field_metadata(data)
        assert meta.bbox is not None
        assert meta.bbox.x == 0.15
        assert meta.bbox.width == 0.30
        assert meta.bbox.pixel_x == 128

    def test_deserialize_without_bbox(self):
        """Test deserialize_field_metadata works without bbox."""
        data = {
            "field_name": "total_charges",
            "value": "150.00",
            "confidence": 0.90,
            "confidence_level": "high",
            "pass1_value": "150.00",
            "pass2_value": "150.00",
            "passes_agree": True,
            "location_hint": "",
            "validation_passed": True,
            "validation_errors": [],
            "source_page": 1,
            "is_hallucination_flag": False,
        }
        meta = deserialize_field_metadata(data)
        assert meta.bbox is None

    def test_roundtrip_field_metadata_with_bbox(self):
        """Test to_dict -> deserialize roundtrip with bbox."""
        bbox = BoundingBoxCoords(
            x=0.22, y=0.45, width=0.15, height=0.02, page=2,
            pixel_x=187, pixel_y=495, pixel_width=128, pixel_height=22,
        )
        original = FieldMetadata(
            field_name="claim_id",
            value="CLM-12345",
            confidence=0.91,
            pass1_value="CLM-12345",
            pass2_value="CLM-12345",
            passes_agree=True,
            bbox=bbox,
        )
        d = original.to_dict()
        restored = deserialize_field_metadata(d)
        assert restored.bbox is not None
        assert restored.bbox.x == original.bbox.x
        assert restored.bbox.y == original.bbox.y
        assert restored.bbox.width == original.bbox.width
        assert restored.bbox.page == original.bbox.page


# ──────────────────────────────────────────────────────────────────
# Spatial Pattern Detection Tests
# ──────────────────────────────────────────────────────────────────


class TestSpatialPatternDetection:
    """Tests for spatial anomaly detection in HallucinationPatternDetector."""

    @pytest.fixture
    def detector(self) -> HallucinationPatternDetector:
        return HallucinationPatternDetector()

    def test_no_bboxes_returns_empty(self, detector: HallucinationPatternDetector):
        """No spatial matches when data has no bbox fields."""
        data = {"patient_name": "John", "dob": "1990-01-15"}
        result = detector._check_spatial_patterns(data)
        assert result == []

    def test_single_bbox_returns_empty(self, detector: HallucinationPatternDetector):
        """No spatial anomaly with only one bbox."""
        data = {
            "patient_name": {
                "value": "John",
                "bbox": {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.04},
            },
        }
        result = detector._check_spatial_patterns(data)
        assert result == []

    def test_identical_bboxes_flagged(self, detector: HallucinationPatternDetector):
        """Identical bboxes across 3+ fields are flagged as spatial anomaly."""
        same_bbox = {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.04}
        data = {
            "field_a": {"value": "A", "bbox": same_bbox},
            "field_b": {"value": "B", "bbox": same_bbox},
            "field_c": {"value": "C", "bbox": same_bbox},
        }
        result = detector._check_spatial_patterns(data)
        spatial = [m for m in result if m.pattern == HallucinationPattern.SPATIAL_ANOMALY]
        assert len(spatial) >= 1
        assert "Identical bounding box" in spatial[0].description

    def test_distinct_bboxes_no_flag(self, detector: HallucinationPatternDetector):
        """Distinct bboxes across fields do not trigger spatial anomaly."""
        data = {
            "field_a": {"value": "A", "bbox": {"x": 0.1, "y": 0.1, "w": 0.2, "h": 0.03}},
            "field_b": {"value": "B", "bbox": {"x": 0.5, "y": 0.3, "w": 0.2, "h": 0.03}},
            "field_c": {"value": "C", "bbox": {"x": 0.1, "y": 0.6, "w": 0.2, "h": 0.03}},
        }
        result = detector._check_spatial_patterns(data)
        identical = [m for m in result if "Identical" in m.description]
        assert len(identical) == 0

    def test_oversized_bbox_flagged(self, detector: HallucinationPatternDetector):
        """Bbox covering >60% of page area is flagged."""
        data = {
            "field_a": {
                "value": "A",
                "bbox": {"x": 0.0, "y": 0.0, "w": 0.9, "h": 0.8},  # 72% area
            },
        }
        result = detector._check_spatial_patterns(data)
        oversized = [m for m in result if "covers" in m.description]
        assert len(oversized) == 1
        assert oversized[0].severity == PatternSeverity.MEDIUM

    def test_normal_sized_bbox_no_flag(self, detector: HallucinationPatternDetector):
        """Normal-sized bbox does not trigger oversized flag."""
        data = {
            "field_a": {
                "value": "A",
                "bbox": {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.04},  # 1.2% area
            },
        }
        result = detector._check_spatial_patterns(data)
        oversized = [m for m in result if "covers" in m.description]
        assert len(oversized) == 0

    def test_zero_area_bbox_flagged(self, detector: HallucinationPatternDetector):
        """Zero-area bbox is flagged as degenerate."""
        data = {
            "field_a": {
                "value": "A",
                "bbox": {"x": 0.1, "y": 0.2, "w": 0.0, "h": 0.04},
            },
        }
        result = detector._check_spatial_patterns(data)
        degenerate = [m for m in result if "Degenerate" in m.description]
        assert len(degenerate) == 1
        assert degenerate[0].severity == PatternSeverity.HIGH

    def test_negative_dimension_bbox_flagged(self, detector: HallucinationPatternDetector):
        """Negative dimension bbox is flagged."""
        data = {
            "field_a": {
                "value": "A",
                "bbox": {"x": 0.1, "y": 0.2, "w": -0.1, "h": 0.04},
            },
        }
        result = detector._check_spatial_patterns(data)
        degenerate = [m for m in result if "Degenerate" in m.description]
        assert len(degenerate) == 1

    def test_out_of_bounds_bbox_flagged(self, detector: HallucinationPatternDetector):
        """Bbox extending beyond page bounds is flagged."""
        data = {
            "field_a": {
                "value": "A",
                "bbox": {"x": 0.8, "y": 0.1, "w": 0.3, "h": 0.04},  # x+w = 1.1
            },
        }
        result = detector._check_spatial_patterns(data)
        oob = [m for m in result if "outside page bounds" in m.description]
        assert len(oob) == 1

    def test_negative_origin_bbox_flagged(self, detector: HallucinationPatternDetector):
        """Bbox with negative origin is flagged."""
        data = {
            "field_a": {
                "value": "A",
                "bbox": {"x": -0.1, "y": 0.2, "w": 0.3, "h": 0.04},
            },
        }
        result = detector._check_spatial_patterns(data)
        oob = [m for m in result if "outside page bounds" in m.description]
        assert len(oob) == 1

    def test_spatial_integrated_with_detect(self, detector: HallucinationPatternDetector):
        """Spatial checks are called as part of the full detect() pipeline."""
        same_bbox = {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.04}
        data = {
            "field_a": {"value": "A", "bbox": same_bbox},
            "field_b": {"value": "B", "bbox": same_bbox},
            "field_c": {"value": "C", "bbox": same_bbox},
        }
        result = detector.detect(data)
        spatial_flags = [
            m for m in result.matches
            if m.pattern == HallucinationPattern.SPATIAL_ANOMALY
        ]
        assert len(spatial_flags) >= 1

    def test_multiple_spatial_anomalies(self, detector: HallucinationPatternDetector):
        """Multiple spatial anomalies detected in same data."""
        data = {
            # Zero-area bbox
            "field_a": {"value": "A", "bbox": {"x": 0.1, "y": 0.2, "w": 0.0, "h": 0.04}},
            # Oversized bbox
            "field_b": {"value": "B", "bbox": {"x": 0.0, "y": 0.0, "w": 0.9, "h": 0.9}},
        }
        result = detector._check_spatial_patterns(data)
        assert len(result) >= 2


# ──────────────────────────────────────────────────────────────────
# Extractor _parse_bbox Tests
# ──────────────────────────────────────────────────────────────────


class TestExtractorParseBbox:
    """Tests for ExtractorAgent._parse_bbox static method."""

    @pytest.fixture
    def parse_bbox(self):
        """Import _parse_bbox from ExtractorAgent."""
        from src.agents.extractor import ExtractorAgent
        return ExtractorAgent._parse_bbox

    def test_dict_with_w_h(self, parse_bbox):
        """Parse dict with w/h shorthand keys."""
        result = parse_bbox({"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.04}, page_number=1)
        assert result is not None
        assert result.x == 0.1
        assert result.y == 0.2
        assert result.width == 0.3
        assert result.height == 0.04
        assert result.page == 1

    def test_dict_with_width_height(self, parse_bbox):
        """Parse dict with width/height full keys."""
        result = parse_bbox(
            {"x": 0.15, "y": 0.25, "width": 0.35, "height": 0.05},
            page_number=2,
        )
        assert result is not None
        assert result.width == 0.35
        assert result.height == 0.05
        assert result.page == 2

    def test_list_format(self, parse_bbox):
        """Parse [x, y, w, h] list format."""
        result = parse_bbox([0.1, 0.2, 0.3, 0.04], page_number=1)
        assert result is not None
        assert result.x == 0.1
        assert result.width == 0.3

    def test_tuple_format(self, parse_bbox):
        """Parse (x, y, w, h) tuple format."""
        result = parse_bbox((0.1, 0.2, 0.3, 0.04), page_number=1)
        assert result is not None
        assert result.x == 0.1

    def test_invalid_string_returns_none(self, parse_bbox):
        """Non-dict/list input returns None."""
        assert parse_bbox("invalid", page_number=1) is None

    def test_none_returns_none(self, parse_bbox):
        """None input returns None."""
        assert parse_bbox(None, page_number=1) is None

    def test_out_of_range_x_returns_none(self, parse_bbox):
        """x > 1.0 returns None."""
        assert parse_bbox({"x": 1.5, "y": 0.2, "w": 0.3, "h": 0.04}, page_number=1) is None

    def test_negative_x_returns_none(self, parse_bbox):
        """Negative x returns None."""
        assert parse_bbox({"x": -0.1, "y": 0.2, "w": 0.3, "h": 0.04}, page_number=1) is None

    def test_zero_width_returns_none(self, parse_bbox):
        """Zero width returns None."""
        assert parse_bbox({"x": 0.1, "y": 0.2, "w": 0.0, "h": 0.04}, page_number=1) is None

    def test_zero_height_returns_none(self, parse_bbox):
        """Zero height returns None."""
        assert parse_bbox({"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.0}, page_number=1) is None

    def test_short_list_returns_none(self, parse_bbox):
        """List with < 4 elements returns None."""
        assert parse_bbox([0.1, 0.2], page_number=1) is None

    def test_boundary_values_valid(self, parse_bbox):
        """Edge values (0.0 origin, small dimensions) are valid."""
        result = parse_bbox({"x": 0.0, "y": 0.0, "w": 0.01, "h": 0.01}, page_number=1)
        assert result is not None
        assert result.x == 0.0
        assert result.y == 0.0

    def test_full_page_bbox(self, parse_bbox):
        """Full-page bbox (0,0,1,1) is valid."""
        result = parse_bbox({"x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}, page_number=1)
        assert result is not None
        assert result.width == 1.0
        assert result.height == 1.0


# ──────────────────────────────────────────────────────────────────
# JSON Export Bbox Tests
# ──────────────────────────────────────────────────────────────────


class TestJSONExportBbox:
    """Tests for bbox in JSON export output."""

    @pytest.fixture
    def state_with_bbox(self) -> dict[str, Any]:
        """Sample extraction state with bbox in field metadata."""
        return {
            "processing_id": "bbox-test-001",
            "pdf_path": "/test/sample.pdf",
            "pdf_hash": "abc123",
            "document_type": "CMS-1500",
            "selected_schema_name": "cms1500_v1",
            "status": ExtractionStatus.COMPLETED.value,
            "start_time": "2024-01-15T10:00:00Z",
            "end_time": "2024-01-15T10:00:30Z",
            "total_vlm_calls": 4,
            "total_processing_time_ms": 30000,
            "retry_count": 0,
            "overall_confidence": 0.92,
            "confidence_level": ConfidenceLevel.HIGH.value,
            "requires_human_review": False,
            "page_images": [b"page1"],
            "merged_extraction": {
                "patient_name": {"value": "John Doe", "confidence": 0.95},
                "dob": {"value": "1990-01-15", "confidence": 0.88},
            },
            "field_metadata": {
                "patient_name": {
                    "confidence": 0.95,
                    "confidence_level": "high",
                    "passes_agree": True,
                    "validation_passed": True,
                    "bbox": {
                        "x": 0.12, "y": 0.05, "width": 0.25, "height": 0.03,
                        "page": 1,
                    },
                },
                "dob": {
                    "confidence": 0.88,
                    "confidence_level": "high",
                    "passes_agree": True,
                    "validation_passed": True,
                    # No bbox for this field
                },
            },
        }

    def test_standard_export_includes_bbox(self, state_with_bbox: dict[str, Any]):
        """Standard format includes bbox in field confidence metadata."""
        config = JSONExportConfig(format=ExportFormat.STANDARD)
        exporter = JSONExporter(config)
        result = exporter.export(state_with_bbox)

        confidence = result.get("confidence", {})
        fields = confidence.get("fields", {})

        # patient_name should have bbox
        assert "patient_name" in fields
        assert "bbox" in fields["patient_name"]
        assert fields["patient_name"]["bbox"]["x"] == 0.12

        # dob should not have bbox
        assert "dob" in fields
        assert "bbox" not in fields["dob"]

    def test_minimal_export_no_bbox(self, state_with_bbox: dict[str, Any]):
        """Minimal format does not include field metadata (and thus no bbox)."""
        config = JSONExportConfig(format=ExportFormat.MINIMAL)
        exporter = JSONExporter(config)
        result = exporter.export(state_with_bbox)

        # Minimal only has data, processing_id, status
        assert "confidence" not in result
        assert "data" in result

    def test_detailed_export_includes_bbox(self, state_with_bbox: dict[str, Any]):
        """Detailed format includes bbox in field confidence metadata."""
        state_with_bbox["validation"] = {"is_valid": True}
        state_with_bbox["page_extractions"] = []
        state_with_bbox["errors"] = []
        state_with_bbox["warnings"] = []
        config = JSONExportConfig(format=ExportFormat.DETAILED)
        exporter = JSONExporter(config)
        result = exporter.export(state_with_bbox)

        confidence = result.get("confidence", {})
        fields = confidence.get("fields", {})
        assert "bbox" in fields["patient_name"]

    def test_export_to_file_with_bbox(self, state_with_bbox: dict[str, Any]):
        """Export to file and verify bbox is in JSON output."""
        with tempfile.TemporaryDirectory() as tmpdir:
            out_path = Path(tmpdir) / "output.json"
            config = JSONExportConfig(format=ExportFormat.STANDARD)
            exporter = JSONExporter(config)
            exporter.export(state_with_bbox, output_path=out_path)

            with out_path.open() as f:
                data = json.load(f)

            assert data["confidence"]["fields"]["patient_name"]["bbox"]["x"] == 0.12


# ──────────────────────────────────────────────────────────────────
# Multi-Record Bbox Tests
# ──────────────────────────────────────────────────────────────────


class TestMultiRecordBbox:
    """Tests for bbox in multi-record extraction dataclass."""

    def test_extracted_record_with_bboxes(self):
        """ExtractedRecord stores field_bboxes correctly."""
        from src.extraction.multi_record import ExtractedRecord

        record = ExtractedRecord(
            record_id=1,
            page_number=1,
            primary_identifier="John Doe",
            entity_type="patient",
            fields={"patient_name": "John Doe", "dob": "1990-01-15"},
            confidence=0.92,
            extraction_time_ms=1500,
            field_bboxes={
                "patient_name": {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.04},
                "dob": {"x": 0.5, "y": 0.2, "w": 0.2, "h": 0.03},
            },
        )
        assert record.field_bboxes is not None
        assert "patient_name" in record.field_bboxes
        assert record.field_bboxes["patient_name"]["x"] == 0.1

    def test_extracted_record_without_bboxes(self):
        """ExtractedRecord works without field_bboxes."""
        from src.extraction.multi_record import ExtractedRecord

        record = ExtractedRecord(
            record_id=2,
            page_number=1,
            primary_identifier="Jane Smith",
            entity_type="patient",
            fields={"patient_name": "Jane Smith"},
            confidence=0.88,
            extraction_time_ms=1200,
        )
        assert record.field_bboxes is None

    def test_extracted_record_to_dict_includes_bboxes(self):
        """ExtractedRecord serialization includes field_bboxes."""
        from src.extraction.multi_record import ExtractedRecord
        from dataclasses import asdict

        record = ExtractedRecord(
            record_id=3,
            page_number=1,
            primary_identifier="Test",
            entity_type="patient",
            fields={"patient_name": "Test"},
            confidence=0.90,
            extraction_time_ms=1000,
            field_bboxes={"patient_name": {"x": 0.1, "y": 0.2, "w": 0.3, "h": 0.04}},
        )
        d = asdict(record)
        assert d["field_bboxes"]["patient_name"]["x"] == 0.1


# ──────────────────────────────────────────────────────────────────
# HallucinationPattern Enum Tests
# ──────────────────────────────────────────────────────────────────


class TestHallucinationPatternEnum:
    """Tests for new SPATIAL_ANOMALY enum value."""

    def test_spatial_anomaly_exists(self):
        """SPATIAL_ANOMALY is a valid HallucinationPattern."""
        assert HallucinationPattern.SPATIAL_ANOMALY == "spatial_anomaly"
        assert HallucinationPattern.SPATIAL_ANOMALY.value == "spatial_anomaly"

    def test_all_patterns_accessible(self):
        """All original + new patterns are accessible."""
        patterns = [p.value for p in HallucinationPattern]
        assert "placeholder_text" in patterns
        assert "repetitive_value" in patterns
        assert "spatial_anomaly" in patterns
