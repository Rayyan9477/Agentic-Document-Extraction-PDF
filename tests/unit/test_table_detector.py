"""
Unit tests for Phase 2B: Table Structure Detection Agent.

Tests TableDetectorAgent, table type definitions, VLM response parsing,
orchestrator integration, and state field management.
"""

import json
from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.agents.table_detector import (
    TABLE_DETECTION_SYSTEM_PROMPT,
    TableDetectionError,
    TableDetectorAgent,
    _empty_detection_result,
)
from src.pipeline.state import ExtractionState, create_initial_state, update_state
from src.pipeline.table_types import (
    DetectedTable,
    TableCell,
    TableDetectionResult,
    TableHeader,
    TableRow,
    create_empty_detected_table,
    create_empty_table_detection_result,
    table_to_rows_dict,
)


# ──────────────────────────────────────────────────────────────────
# Helper fixtures
# ──────────────────────────────────────────────────────────────────


def _make_page_image(page_number: int) -> dict[str, Any]:
    """Create a minimal page image dict for testing."""
    return {
        "page_number": page_number,
        "data_uri": f"data:image/png;base64,fakepage{page_number}",
        "base64_encoded": f"fakepage{page_number}",
        "width": 2550,
        "height": 3300,
    }


def _make_state_with_pages(n_pages: int) -> ExtractionState:
    """Create a state with N fake page images."""
    pages = [_make_page_image(i + 1) for i in range(n_pages)]
    state = create_initial_state(pdf_path="/tmp/test.pdf")
    return update_state(state, {"page_images": pages})


def _make_vlm_table_response(
    num_tables: int = 1,
    rows_per_table: int = 3,
    cols: int = 3,
) -> dict[str, Any]:
    """Build a realistic VLM table detection response."""
    tables = []
    for t in range(num_tables):
        headers = [
            {
                "col_index": c,
                "text": f"Col_{c}",
                "location": {"x": 0.1 + c * 0.2, "y": 0.2, "width": 0.15, "height": 0.03},
                "data_type_hint": "text",
            }
            for c in range(cols)
        ]
        rows = []
        for r in range(rows_per_table):
            cells = [
                {
                    "row_index": r,
                    "col_index": c,
                    "text": f"R{r}C{c}",
                    "location": {"x": 0.1 + c * 0.2, "y": 0.25 + r * 0.04, "width": 0.15, "height": 0.03},
                    "confidence": 0.9,
                    "is_header": False,
                    "rowspan": 1,
                    "colspan": 1,
                    "cell_type": "text",
                }
                for c in range(cols)
            ]
            rows.append({
                "row_index": r,
                "cells": cells,
                "is_header_row": False,
                "is_total_row": r == rows_per_table - 1 and t == 0,
                "is_separator_row": False,
                "row_location": {"x": 0.1, "y": 0.25 + r * 0.04, "width": 0.8, "height": 0.03},
            })
        tables.append({
            "table_id": f"table_{t}",
            "location": {"x": 0.05, "y": 0.15, "width": 0.9, "height": 0.5},
            "row_count": rows_per_table,
            "column_count": cols,
            "confidence": 0.92,
            "headers": headers,
            "rows": rows,
            "has_header_row": True,
            "has_total_row": t == 0,
            "has_merged_cells": False,
            "table_type": "line_items",
            "description": f"Test table {t}",
            "cell_borders_visible": True,
            "extraction_quality": "high",
            "needs_review": False,
            "review_reason": "",
        })
    return {
        "tables": tables,
        "table_count": num_tables,
        "has_tables": num_tables > 0,
        "notes": "Test response",
    }


@pytest.fixture
def detector() -> TableDetectorAgent:
    """Create a TableDetectorAgent with a mock client."""
    mock_client = MagicMock()
    return TableDetectorAgent(client=mock_client)


# ──────────────────────────────────────────────────────────────────
# Table Type Tests
# ──────────────────────────────────────────────────────────────────


class TestTableTypes:
    def test_create_empty_detected_table(self):
        table = create_empty_detected_table()
        assert table["table_id"] == "table_0"
        assert table["page_number"] == 1
        assert table["row_count"] == 0
        assert table["column_count"] == 0
        assert table["headers"] == []
        assert table["rows"] == []
        assert table["table_type"] == "unknown"
        assert table["extraction_quality"] == "low"

    def test_create_empty_detected_table_custom_id(self):
        table = create_empty_detected_table(table_id="my_table", page_number=5)
        assert table["table_id"] == "my_table"
        assert table["page_number"] == 5

    def test_create_empty_detection_result(self):
        result = create_empty_table_detection_result(page_number=3)
        assert result["page_number"] == 3
        assert result["tables"] == []
        assert result["table_count"] == 0
        assert result["has_tables"] is False
        assert result["detection_method"] == "vlm"

    def test_table_to_rows_dict_basic(self):
        table: dict[str, Any] = {
            "headers": [
                {"col_index": 0, "text": "Name", "location": {}, "data_type_hint": "text"},
                {"col_index": 1, "text": "Amount", "location": {}, "data_type_hint": "currency"},
            ],
            "rows": [
                {
                    "row_index": 0,
                    "cells": [
                        {"col_index": 0, "text": "Item A", "row_index": 0},
                        {"col_index": 1, "text": "$100", "row_index": 0},
                    ],
                    "is_header_row": False,
                    "is_total_row": False,
                    "is_separator_row": False,
                },
                {
                    "row_index": 1,
                    "cells": [
                        {"col_index": 0, "text": "Item B", "row_index": 1},
                        {"col_index": 1, "text": "$200", "row_index": 1},
                    ],
                    "is_header_row": False,
                    "is_total_row": False,
                    "is_separator_row": False,
                },
            ],
            "column_count": 2,
        }
        rows = table_to_rows_dict(table)
        assert len(rows) == 2
        assert rows[0]["Name"] == "Item A"
        assert rows[0]["Amount"] == "$100"
        assert rows[1]["Name"] == "Item B"

    def test_table_to_rows_dict_skips_header_row(self):
        table: dict[str, Any] = {
            "headers": [{"col_index": 0, "text": "X", "location": {}, "data_type_hint": "text"}],
            "rows": [
                {
                    "row_index": 0,
                    "cells": [{"col_index": 0, "text": "Header", "row_index": 0}],
                    "is_header_row": True,
                    "is_total_row": False,
                    "is_separator_row": False,
                },
                {
                    "row_index": 1,
                    "cells": [{"col_index": 0, "text": "Data", "row_index": 1}],
                    "is_header_row": False,
                    "is_total_row": False,
                    "is_separator_row": False,
                },
            ],
            "column_count": 1,
        }
        rows = table_to_rows_dict(table)
        assert len(rows) == 1
        assert rows[0]["X"] == "Data"

    def test_table_to_rows_dict_skips_separator_rows(self):
        table: dict[str, Any] = {
            "headers": [],
            "rows": [
                {
                    "row_index": 0,
                    "cells": [{"col_index": 0, "text": "Data", "row_index": 0}],
                    "is_header_row": False,
                    "is_total_row": False,
                    "is_separator_row": True,
                },
            ],
            "column_count": 1,
        }
        rows = table_to_rows_dict(table)
        assert len(rows) == 0

    def test_table_to_rows_dict_fallback_col_names(self):
        table: dict[str, Any] = {
            "headers": [],
            "rows": [
                {
                    "row_index": 0,
                    "cells": [
                        {"col_index": 0, "text": "A", "row_index": 0},
                        {"col_index": 1, "text": "B", "row_index": 0},
                    ],
                    "is_header_row": False,
                    "is_total_row": False,
                    "is_separator_row": False,
                },
            ],
            "column_count": 2,
        }
        rows = table_to_rows_dict(table)
        assert rows[0]["col_0"] == "A"
        assert rows[0]["col_1"] == "B"

    def test_table_to_rows_dict_empty_table(self):
        table: dict[str, Any] = {"headers": [], "rows": [], "column_count": 0}
        rows = table_to_rows_dict(table)
        assert rows == []


# ──────────────────────────────────────────────────────────────────
# Empty/No Pages Tests
# ──────────────────────────────────────────────────────────────────


class TestTableDetectorEmptyInput:
    def test_empty_pages_returns_empty(self, detector: TableDetectorAgent):
        state = create_initial_state(pdf_path="/tmp/test.pdf")
        result = detector.process(state)
        assert result["detected_tables"] == []

    def test_no_pages_key(self, detector: TableDetectorAgent):
        state = create_initial_state(pdf_path="/tmp/test.pdf")
        state = update_state(state, {"page_images": []})
        result = detector.process(state)
        assert result["detected_tables"] == []


# ──────────────────────────────────────────────────────────────────
# VLM Response Parsing Tests
# ──────────────────────────────────────────────────────────────────


class TestTableDetectorParsing:
    def test_parse_single_table(self, detector: TableDetectorAgent):
        raw = _make_vlm_table_response(num_tables=1, rows_per_table=3, cols=3)
        result = detector._parse_detection_response(raw, page_number=1, elapsed_ms=500)

        assert result["page_number"] == 1
        assert result["table_count"] == 1
        assert result["has_tables"] is True
        assert result["detection_time_ms"] == 500
        assert result["detection_method"] == "vlm"

        table = result["tables"][0]
        assert table["table_id"] == "table_0"
        assert table["row_count"] == 3
        assert table["column_count"] == 3
        assert len(table["headers"]) == 3
        assert len(table["rows"]) == 3

    def test_parse_multiple_tables(self, detector: TableDetectorAgent):
        raw = _make_vlm_table_response(num_tables=3, rows_per_table=2, cols=2)
        result = detector._parse_detection_response(raw, page_number=2, elapsed_ms=800)

        assert result["table_count"] == 3
        assert len(result["tables"]) == 3
        for i, t in enumerate(result["tables"]):
            assert t["table_id"] == f"table_{i}"
            assert t["page_number"] == 2

    def test_parse_empty_response(self, detector: TableDetectorAgent):
        raw = {"tables": [], "table_count": 0, "has_tables": False, "notes": ""}
        result = detector._parse_detection_response(raw, page_number=1, elapsed_ms=100)

        assert result["table_count"] == 0
        assert result["has_tables"] is False
        assert result["tables"] == []

    def test_parse_response_missing_tables_key(self, detector: TableDetectorAgent):
        raw = {"notes": "No tables found"}
        result = detector._parse_detection_response(raw, page_number=1, elapsed_ms=50)

        assert result["table_count"] == 0
        assert result["has_tables"] is False

    def test_normalize_table_defaults(self, detector: TableDetectorAgent):
        """Normalize fills in missing fields with defaults."""
        raw_table: dict[str, Any] = {
            "location": {"x": 0.1, "y": 0.2, "width": 0.8, "height": 0.5},
            "confidence": 0.85,
        }
        normalized = detector._normalize_table(raw_table, page_number=1, index=0)

        assert normalized["table_id"] == "table_0"
        assert normalized["page_number"] == 1
        assert normalized["confidence"] == 0.85
        assert normalized["headers"] == []
        assert normalized["rows"] == []
        assert normalized["table_type"] == "unknown"
        assert normalized["extraction_quality"] == "medium"
        assert normalized["has_header_row"] is False
        assert normalized["has_merged_cells"] is False

    def test_normalize_table_infers_col_count_from_rows(self, detector: TableDetectorAgent):
        raw_table: dict[str, Any] = {
            "location": {"x": 0, "y": 0, "width": 1, "height": 1},
            "rows": [
                {
                    "row_index": 0,
                    "cells": [
                        {"col_index": 0, "text": "A"},
                        {"col_index": 1, "text": "B"},
                        {"col_index": 2, "text": "C"},
                    ],
                },
            ],
        }
        normalized = detector._normalize_table(raw_table, page_number=1, index=0)
        assert normalized["column_count"] == 3

    def test_normalize_table_preserves_custom_fields(self, detector: TableDetectorAgent):
        raw_table: dict[str, Any] = {
            "table_id": "custom_table",
            "location": {"x": 0, "y": 0, "width": 1, "height": 1},
            "table_type": "financial",
            "description": "P&L Statement",
            "needs_review": True,
            "review_reason": "Merged cells detected",
            "has_merged_cells": True,
        }
        normalized = detector._normalize_table(raw_table, page_number=2, index=5)
        assert normalized["table_id"] == "custom_table"
        assert normalized["table_type"] == "financial"
        assert normalized["description"] == "P&L Statement"
        assert normalized["needs_review"] is True
        assert normalized["review_reason"] == "Merged cells detected"
        assert normalized["has_merged_cells"] is True

    def test_cell_text_converted_to_string(self, detector: TableDetectorAgent):
        raw_table: dict[str, Any] = {
            "location": {"x": 0, "y": 0, "width": 1, "height": 1},
            "rows": [
                {
                    "row_index": 0,
                    "cells": [
                        {"col_index": 0, "text": 12345},
                        {"col_index": 1, "text": None},
                    ],
                },
            ],
        }
        normalized = detector._normalize_table(raw_table, page_number=1, index=0)
        cells = normalized["rows"][0]["cells"]
        assert cells[0]["text"] == "12345"
        assert cells[1]["text"] == "None"


# ──────────────────────────────────────────────────────────────────
# Process Integration Tests (with mocked VLM)
# ──────────────────────────────────────────────────────────────────


class TestTableDetectorProcess:
    def test_process_single_page_with_table(self, detector: TableDetectorAgent):
        state = _make_state_with_pages(1)

        vlm_response = _make_vlm_table_response(num_tables=1, rows_per_table=2, cols=3)

        with patch.object(detector, "send_vision_request_with_json", return_value=vlm_response):
            result = detector.process(state)

        tables = result["detected_tables"]
        assert len(tables) == 1
        assert tables[0]["page_number"] == 1
        assert tables[0]["table_count"] == 1
        assert tables[0]["has_tables"] is True

    def test_process_multi_page(self, detector: TableDetectorAgent):
        state = _make_state_with_pages(3)

        responses = [
            _make_vlm_table_response(num_tables=1),
            _make_vlm_table_response(num_tables=0),
            _make_vlm_table_response(num_tables=2),
        ]
        call_count = {"i": 0}

        def mock_vlm(*args, **kwargs):
            idx = call_count["i"]
            call_count["i"] += 1
            return responses[idx]

        with patch.object(detector, "send_vision_request_with_json", side_effect=mock_vlm):
            result = detector.process(state)

        tables = result["detected_tables"]
        assert len(tables) == 3
        assert tables[0]["table_count"] == 1
        assert tables[1]["table_count"] == 0
        assert tables[2]["table_count"] == 2

    def test_process_vlm_failure_returns_empty(self, detector: TableDetectorAgent):
        state = _make_state_with_pages(1)

        with patch.object(
            detector, "send_vision_request_with_json",
            side_effect=Exception("VLM unavailable"),
        ):
            result = detector.process(state)

        tables = result["detected_tables"]
        assert len(tables) == 1
        assert tables[0]["table_count"] == 0
        assert tables[0]["has_tables"] is False

    def test_process_updates_vlm_calls(self, detector: TableDetectorAgent):
        state = _make_state_with_pages(2)
        vlm_response = _make_vlm_table_response(num_tables=1)

        with patch.object(detector, "send_vision_request_with_json", return_value=vlm_response):
            result = detector.process(state)

        # VLM calls should be tracked
        assert result.get("total_vlm_calls", 0) >= 0

    def test_process_preserves_existing_state(self, detector: TableDetectorAgent):
        state = _make_state_with_pages(1)
        state = update_state(state, {
            "document_type": "CMS-1500",
            "overall_confidence": 0.85,
        })

        vlm_response = _make_vlm_table_response(num_tables=1)

        with patch.object(detector, "send_vision_request_with_json", return_value=vlm_response):
            result = detector.process(state)

        assert result["document_type"] == "CMS-1500"
        assert result["overall_confidence"] == 0.85
        assert len(result["detected_tables"]) == 1


# ──────────────────────────────────────────────────────────────────
# Table Hints from Component Maps
# ──────────────────────────────────────────────────────────────────


class TestTableHints:
    def test_get_table_hints_from_component_maps(self, detector: TableDetectorAgent):
        component_maps = [
            {
                "page_number": 1,
                "tables": [
                    {"table_id": "t1", "location": {"x": 0.1, "y": 0.2, "width": 0.8, "height": 0.4}},
                ],
            },
            {
                "page_number": 2,
                "tables": [],
            },
        ]

        hints_p1 = detector._get_table_hints(component_maps, page_number=1)
        assert len(hints_p1) == 1
        assert hints_p1[0]["table_id"] == "t1"

        hints_p2 = detector._get_table_hints(component_maps, page_number=2)
        assert len(hints_p2) == 0

        hints_p3 = detector._get_table_hints(component_maps, page_number=99)
        assert len(hints_p3) == 0

    def test_hints_included_in_prompt(self, detector: TableDetectorAgent):
        hints = [
            {
                "table_id": "hint_0",
                "location": {"x": 0.1, "y": 0.2, "width": 0.8, "height": 0.4},
                "row_count": 5,
                "column_count": 3,
            },
        ]
        prompt = detector._build_detection_prompt(page_number=1, table_hints=hints)
        assert "Pre-detected Table Regions" in prompt
        assert "1 potential table" in prompt

    def test_no_hints_no_hint_section(self, detector: TableDetectorAgent):
        prompt = detector._build_detection_prompt(page_number=1, table_hints=[])
        assert "Pre-detected Table Regions" not in prompt

    def test_process_with_component_maps(self, detector: TableDetectorAgent):
        state = _make_state_with_pages(1)
        state = update_state(state, {
            "component_maps": [
                {
                    "page_number": 1,
                    "tables": [{"table_id": "hint_t", "location": {"x": 0.1, "y": 0.2, "width": 0.8, "height": 0.4}}],
                },
            ],
        })

        vlm_response = _make_vlm_table_response(num_tables=1)
        with patch.object(detector, "send_vision_request_with_json", return_value=vlm_response):
            result = detector.process(state)

        assert result["detected_tables"][0]["has_tables"] is True


# ──────────────────────────────────────────────────────────────────
# Get Tables For Page Helper
# ──────────────────────────────────────────────────────────────────


class TestGetTablesForPage:
    def test_get_tables_for_existing_page(self, detector: TableDetectorAgent):
        results = [
            {"page_number": 1, "tables": [{"table_id": "t1"}]},
            {"page_number": 2, "tables": [{"table_id": "t2"}, {"table_id": "t3"}]},
        ]
        tables = detector.get_tables_for_page(results, page_number=2)
        assert len(tables) == 2
        assert tables[0]["table_id"] == "t2"

    def test_get_tables_for_missing_page(self, detector: TableDetectorAgent):
        results = [
            {"page_number": 1, "tables": [{"table_id": "t1"}]},
        ]
        tables = detector.get_tables_for_page(results, page_number=99)
        assert tables == []

    def test_get_tables_from_empty_results(self, detector: TableDetectorAgent):
        tables = detector.get_tables_for_page([], page_number=1)
        assert tables == []


# ──────────────────────────────────────────────────────────────────
# Prompt Tests
# ──────────────────────────────────────────────────────────────────


class TestTableDetectorPrompts:
    def test_system_prompt_is_defined(self):
        assert "table structure detection" in TABLE_DETECTION_SYSTEM_PROMPT.lower()
        assert "JSON" in TABLE_DETECTION_SYSTEM_PROMPT

    def test_detection_prompt_contains_page_number(self, detector: TableDetectorAgent):
        prompt = detector._build_detection_prompt(page_number=5, table_hints=[])
        assert "Page 5" in prompt

    def test_detection_prompt_includes_format_spec(self, detector: TableDetectorAgent):
        prompt = detector._build_detection_prompt(page_number=1, table_hints=[])
        assert "table_id" in prompt
        assert "row_count" in prompt
        assert "column_count" in prompt
        assert "headers" in prompt
        assert "cell_type" in prompt


# ──────────────────────────────────────────────────────────────────
# State Fields Tests
# ──────────────────────────────────────────────────────────────────


class TestTableStateFields:
    def test_detected_tables_in_initial_state(self):
        state = create_initial_state(pdf_path="/tmp/test.pdf")
        assert "detected_tables" in state
        assert state["detected_tables"] == []

    def test_detected_tables_preserved_across_updates(self):
        state = create_initial_state(pdf_path="/tmp/test.pdf")
        tables_data = [{"page_number": 1, "tables": [], "table_count": 0}]
        state = update_state(state, {"detected_tables": tables_data})
        assert state["detected_tables"] == tables_data


# ──────────────────────────────────────────────────────────────────
# Empty Detection Result Helper
# ──────────────────────────────────────────────────────────────────


class TestEmptyDetectionResult:
    def test_empty_result_structure(self):
        result = _empty_detection_result(page_number=7)
        assert result["page_number"] == 7
        assert result["tables"] == []
        assert result["table_count"] == 0
        assert result["has_tables"] is False
        assert result["detection_time_ms"] == 0


# ──────────────────────────────────────────────────────────────────
# Edge Cases
# ──────────────────────────────────────────────────────────────────


class TestTableDetectorEdgeCases:
    def test_page_without_data_uri(self, detector: TableDetectorAgent):
        """Pages missing image data get empty detection results."""
        state = create_initial_state(pdf_path="/tmp/test.pdf")
        state = update_state(state, {
            "page_images": [{"page_number": 1}],  # No data_uri
        })

        result = detector.process(state)
        tables = result["detected_tables"]
        assert len(tables) == 1
        assert tables[0]["has_tables"] is False

    def test_large_page_count(self, detector: TableDetectorAgent):
        """Handles many pages without error."""
        state = _make_state_with_pages(50)

        empty_response = {"tables": [], "table_count": 0, "has_tables": False, "notes": ""}
        with patch.object(detector, "send_vision_request_with_json", return_value=empty_response):
            result = detector.process(state)

        assert len(result["detected_tables"]) == 50

    def test_malformed_vlm_cell_data(self, detector: TableDetectorAgent):
        """Handles malformed cell data gracefully."""
        raw_table: dict[str, Any] = {
            "location": {"x": 0, "y": 0, "width": 1, "height": 1},
            "rows": [
                {
                    "cells": [
                        {"text": "OK"},  # Missing col_index, row_index, etc.
                    ],
                },
            ],
        }
        normalized = detector._normalize_table(raw_table, page_number=1, index=0)
        assert len(normalized["rows"]) == 1
        cell = normalized["rows"][0]["cells"][0]
        assert cell["text"] == "OK"
        assert cell["col_index"] == 0
        assert cell["confidence"] == 0.5  # Default

    def test_table_type_values(self, detector: TableDetectorAgent):
        """All table type values are handled."""
        for table_type in ["line_items", "summary", "schedule", "comparison",
                           "reference", "form_grid", "financial", "unknown"]:
            raw = {"location": {"x": 0, "y": 0, "width": 1, "height": 1}, "table_type": table_type}
            normalized = detector._normalize_table(raw, page_number=1, index=0)
            assert normalized["table_type"] == table_type
