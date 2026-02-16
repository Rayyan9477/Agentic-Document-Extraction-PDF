"""
Table Detector Agent for VLM-based table structure detection and extraction.

Detects tables in document pages, extracts their structure (rows, columns,
headers, cells), and provides cell-level content with spatial coordinates.

Works both standalone and as an optional step in the VLM-first pipeline
between component detection and extraction.
"""

import time
from typing import Any

from src.agents.base import AgentError, BaseAgent
from src.agents.utils import RetryConfig, retry_with_backoff
from src.client.lm_client import LMStudioClient
from src.config import get_logger, get_settings
from src.pipeline.state import ExtractionState, update_state
from src.pipeline.table_types import (
    DetectedTable,
    TableCell,
    TableDetectionResult,
    TableHeader,
    TableRow,
)

logger = get_logger(__name__)


class TableDetectionError(AgentError):
    """Error during table detection."""


TABLE_DETECTION_SYSTEM_PROMPT = """You are a table structure detection specialist.

Given a document page image, identify ALL tables present and extract their
complete structure including headers, rows, and cell content.

Rules:
1. Use normalized coordinates (0.0-1.0) for all locations
2. Detect merged cells and spanning correctly
3. Identify header rows vs data rows vs total/summary rows
4. Classify table type (line_items, summary, financial, etc.)
5. Report confidence for each cell and overall table
6. Flag cells that are difficult to read or ambiguous

Respond ONLY with valid JSON matching the requested format.
"""


class TableDetectorAgent(BaseAgent):
    """
    VLM-powered table detection and structure extraction agent.

    Strategy:
    1. For each page, detect if tables exist and their boundaries
    2. For each detected table, extract full structure (headers, rows, cells)
    3. Classify table type and assess extraction quality
    4. Flag tables needing human review

    Integrates with the component_maps from ComponentDetectorAgent when
    available, using pre-detected table regions to focus VLM attention.
    """

    def __init__(self, client: LMStudioClient | None = None) -> None:
        super().__init__(name="table_detector", client=client)
        self._settings = get_settings()

    def process(self, state: ExtractionState) -> ExtractionState:
        """
        Detect tables in all pages and extract their structure.

        Reads from state:
            page_images: list of page image dicts
            component_maps: optional list of component detection results

        Adds to state:
            detected_tables: list of TableDetectionResult dicts
        """
        start_time = time.time()
        page_images = state.get("page_images", [])

        if not page_images:
            return update_state(state, {"detected_tables": []})

        self._logger.info(
            "table_detection_started",
            page_count=len(page_images),
            processing_id=state.get("processing_id"),
        )

        # Get pre-detected table regions from component maps if available
        component_maps = state.get("component_maps", [])

        detection_results: list[dict[str, Any]] = []

        for page_data in page_images:
            page_number = page_data.get("page_number", 1)
            image_data = page_data.get("data_uri", page_data.get("base64_encoded", ""))

            if not image_data:
                self._logger.warning(
                    "page_missing_image_data",
                    page_number=page_number,
                )
                detection_results.append(
                    _empty_detection_result(page_number),
                )
                continue

            # Find pre-detected table hints for this page
            table_hints = self._get_table_hints(component_maps, page_number)

            result = self._detect_page_tables(
                image_data, page_number, table_hints,
            )
            detection_results.append(result)

            self._logger.info(
                "page_tables_detected",
                page_number=page_number,
                table_count=result.get("table_count", 0),
            )

        elapsed_ms = int((time.time() - start_time) * 1000)

        total_tables = sum(r.get("table_count", 0) for r in detection_results)
        self._logger.info(
            "table_detection_completed",
            pages_processed=len(detection_results),
            total_tables=total_tables,
            total_time_ms=elapsed_ms,
        )

        return update_state(state, {
            "detected_tables": detection_results,
            "total_vlm_calls": state.get("total_vlm_calls", 0) + self._vlm_calls,
            "total_processing_time_ms": state.get("total_processing_time_ms", 0) + elapsed_ms,
        })

    def _get_table_hints(
        self,
        component_maps: list[dict[str, Any]],
        page_number: int,
    ) -> list[dict[str, Any]]:
        """Extract table hints from component detection results."""
        for cm in component_maps:
            if cm.get("page_number") == page_number:
                return cm.get("tables", [])
        return []

    def _detect_page_tables(
        self,
        image_data: str,
        page_number: int,
        table_hints: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """
        Detect and extract tables from a single page.

        Falls back to empty result on VLM failure.
        """
        page_start = time.time()

        prompt = self._build_detection_prompt(page_number, table_hints)

        retry_config = RetryConfig(
            max_retries=self._settings.extraction.max_retries,
            base_delay_ms=500,
            max_delay_ms=self._settings.agent.max_retry_delay_ms,
        )

        def make_vlm_call() -> dict[str, Any]:
            return self.send_vision_request_with_json(
                image_data=image_data,
                prompt=prompt,
                system_prompt=TABLE_DETECTION_SYSTEM_PROMPT,
                temperature=0.1,
                max_tokens=4096,
            )

        try:
            raw = retry_with_backoff(
                func=make_vlm_call,
                config=retry_config,
                on_retry=lambda attempt, e: self._logger.warning(
                    "table_detection_retry",
                    page_number=page_number,
                    attempt=attempt + 1,
                    error=str(e),
                ),
            )

            elapsed_ms = int((time.time() - page_start) * 1000)
            return self._parse_detection_response(raw, page_number, elapsed_ms)

        except Exception as e:
            self._logger.warning(
                "page_table_detection_failed",
                page_number=page_number,
                error=str(e),
            )
            return _empty_detection_result(page_number)

    def _build_detection_prompt(
        self,
        page_number: int,
        table_hints: list[dict[str, Any]],
    ) -> str:
        """Build the table detection prompt, optionally with component hints."""
        hint_section = ""
        if table_hints:
            hint_section = f"""
## Pre-detected Table Regions

The component detector found {len(table_hints)} potential table(s) on this page.
Focus on these regions but also check for any tables it may have missed.

Hint regions:
"""
            for i, hint in enumerate(table_hints):
                loc = hint.get("location", {})
                hint_section += (
                    f"- Table {i+1}: x={loc.get('x', '?')}, y={loc.get('y', '?')}, "
                    f"w={loc.get('width', '?')}, h={loc.get('height', '?')}, "
                    f"rows~{hint.get('row_count', '?')}, cols~{hint.get('column_count', '?')}\n"
                )

        return f"""# TASK: Table Detection & Structure Extraction — Page {page_number}

{hint_section}

## Goal

Detect ALL tables on this page and extract their complete structure.

## For each table, provide:

```json
{{
  "table_id": "table_0",
  "location": {{"x": 0.05, "y": 0.2, "width": 0.9, "height": 0.4}},
  "row_count": 10,
  "column_count": 5,
  "confidence": 0.92,
  "headers": [
    {{"col_index": 0, "text": "Date", "location": {{"x":0.05,"y":0.2,"width":0.15,"height":0.03}}, "data_type_hint": "date"}},
    ...
  ],
  "rows": [
    {{
      "row_index": 0,
      "cells": [
        {{"row_index": 0, "col_index": 0, "text": "01/15/2024", "location": {{"x":0.05,"y":0.23,"width":0.15,"height":0.03}}, "confidence": 0.95, "is_header": false, "rowspan": 1, "colspan": 1, "cell_type": "date"}},
        ...
      ],
      "is_header_row": false,
      "is_total_row": false,
      "is_separator_row": false,
      "row_location": {{"x":0.05,"y":0.23,"width":0.9,"height":0.03}}
    }},
    ...
  ],
  "has_header_row": true,
  "has_total_row": true,
  "has_merged_cells": false,
  "table_type": "line_items",
  "description": "Service line items with dates, CPT codes, and charges",
  "cell_borders_visible": true,
  "extraction_quality": "high",
  "needs_review": false,
  "review_reason": ""
}}
```

## Table types
- **line_items**: Billing/invoice line items
- **summary**: Summary or totals table
- **schedule**: Schedule of dates/events
- **comparison**: Side-by-side comparison
- **reference**: Reference codes/lookup table
- **form_grid**: Form fields arranged in grid
- **financial**: Financial data (P&L, balance sheet)
- **unknown**: Cannot determine

## Cell types
- text, number, currency, date, code, empty, checkbox

## Required output format

```json
{{
  "tables": [/* array of table objects */],
  "table_count": 2,
  "has_tables": true,
  "notes": "Observations about table quality or issues"
}}
```

## Critical reminders
1. Use normalized 0-1 coordinates for ALL locations
2. Read EVERY cell — don't skip or summarize
3. Mark total/summary rows explicitly
4. Flag low-confidence cells
5. Detect merged cells (rowspan/colspan > 1)

Begin table detection now."""

    def _parse_detection_response(
        self,
        raw: dict[str, Any],
        page_number: int,
        elapsed_ms: int,
    ) -> dict[str, Any]:
        """Parse and normalize VLM response into TableDetectionResult."""
        tables_raw = raw.get("tables", [])
        tables: list[dict[str, Any]] = []

        for i, t in enumerate(tables_raw):
            table = self._normalize_table(t, page_number, i)
            tables.append(table)

        return {
            "page_number": page_number,
            "tables": tables,
            "table_count": len(tables),
            "has_tables": len(tables) > 0,
            "detection_time_ms": elapsed_ms,
            "detection_method": "vlm",
            "notes": raw.get("notes", ""),
        }

    def _normalize_table(
        self,
        raw_table: dict[str, Any],
        page_number: int,
        index: int,
    ) -> dict[str, Any]:
        """Normalize a raw VLM table response into DetectedTable format."""
        table_id = raw_table.get("table_id", f"table_{index}")
        location = raw_table.get("location", {"x": 0, "y": 0, "width": 0, "height": 0})

        # Normalize headers
        headers = []
        for h in raw_table.get("headers", []):
            headers.append({
                "col_index": h.get("col_index", 0),
                "text": h.get("text", ""),
                "location": h.get("location", {"x": 0, "y": 0, "width": 0, "height": 0}),
                "data_type_hint": h.get("data_type_hint", "text"),
            })

        # Normalize rows
        rows = []
        for r in raw_table.get("rows", []):
            cells = []
            for c in r.get("cells", []):
                cells.append({
                    "row_index": c.get("row_index", 0),
                    "col_index": c.get("col_index", 0),
                    "text": str(c.get("text", "")),
                    "location": c.get("location", {"x": 0, "y": 0, "width": 0, "height": 0}),
                    "confidence": float(c.get("confidence", 0.5)),
                    "is_header": bool(c.get("is_header", False)),
                    "rowspan": int(c.get("rowspan", 1)),
                    "colspan": int(c.get("colspan", 1)),
                    "cell_type": c.get("cell_type", "text"),
                })
            rows.append({
                "row_index": r.get("row_index", len(rows)),
                "cells": cells,
                "is_header_row": bool(r.get("is_header_row", False)),
                "is_total_row": bool(r.get("is_total_row", False)),
                "is_separator_row": bool(r.get("is_separator_row", False)),
                "row_location": r.get("row_location", {"x": 0, "y": 0, "width": 0, "height": 0}),
            })

        # Infer row/col count from actual data if not provided
        row_count = raw_table.get("row_count", len(rows))
        col_count = raw_table.get("column_count", len(headers) or 0)
        if col_count == 0 and rows:
            col_count = max((len(r.get("cells", [])) for r in rows), default=0)

        return {
            "table_id": table_id,
            "page_number": page_number,
            "location": location,
            "row_count": row_count,
            "column_count": col_count,
            "confidence": float(raw_table.get("confidence", 0.5)),
            "headers": headers,
            "rows": rows,
            "has_header_row": bool(raw_table.get("has_header_row", len(headers) > 0)),
            "has_total_row": bool(raw_table.get("has_total_row", False)),
            "has_merged_cells": bool(raw_table.get("has_merged_cells", False)),
            "table_type": raw_table.get("table_type", "unknown"),
            "description": raw_table.get("description", ""),
            "cell_borders_visible": bool(raw_table.get("cell_borders_visible", False)),
            "extraction_quality": raw_table.get("extraction_quality", "medium"),
            "needs_review": bool(raw_table.get("needs_review", False)),
            "review_reason": raw_table.get("review_reason", ""),
        }

    def get_tables_for_page(
        self,
        detection_results: list[dict[str, Any]],
        page_number: int,
    ) -> list[dict[str, Any]]:
        """
        Get detected tables for a specific page.

        Args:
            detection_results: All detection results from state.
            page_number: Page number to look up.

        Returns:
            List of DetectedTable dicts for that page.
        """
        for result in detection_results:
            if result.get("page_number") == page_number:
                return result.get("tables", [])
        return []


def _empty_detection_result(page_number: int) -> dict[str, Any]:
    """Create empty detection result for a page."""
    return {
        "page_number": page_number,
        "tables": [],
        "table_count": 0,
        "has_tables": False,
        "detection_time_ms": 0,
        "detection_method": "vlm",
        "notes": "",
    }
