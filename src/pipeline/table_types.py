"""
Table structure types for VLM-based table detection and extraction.

Defines granular types for detected tables, cells, rows, and headers
that go beyond the basic TableStructure in layout_types.py by capturing
actual cell content and spatial relationships.
"""

from typing import Any, Literal, TypedDict

from src.pipeline.layout_types import BoundingBox


class TableCell(TypedDict):
    """A single cell in a detected table."""

    row_index: int
    col_index: int
    text: str
    location: BoundingBox
    confidence: float
    is_header: bool
    rowspan: int
    colspan: int
    cell_type: Literal["text", "number", "currency", "date", "code", "empty", "checkbox"]


class TableHeader(TypedDict):
    """A column header in a detected table."""

    col_index: int
    text: str
    location: BoundingBox
    data_type_hint: Literal["text", "number", "currency", "date", "code", "mixed"]


class TableRow(TypedDict):
    """A row of cells in a detected table."""

    row_index: int
    cells: list[TableCell]
    is_header_row: bool
    is_total_row: bool
    is_separator_row: bool
    row_location: BoundingBox


class DetectedTable(TypedDict):
    """
    A fully detected table with structure and content.

    This extends the basic TableStructure from layout_types.py by including
    actual cell values, row/column structure, and extraction metadata.
    """

    table_id: str
    page_number: int
    location: BoundingBox
    row_count: int
    column_count: int
    confidence: float

    # Structure
    headers: list[TableHeader]
    rows: list[TableRow]
    has_header_row: bool
    has_total_row: bool
    has_merged_cells: bool

    # Classification
    table_type: Literal[
        "line_items", "summary", "schedule", "comparison",
        "reference", "form_grid", "financial", "unknown",
    ]
    description: str

    # Quality indicators
    cell_borders_visible: bool
    extraction_quality: Literal["high", "medium", "low"]
    needs_review: bool
    review_reason: str


class TableDetectionResult(TypedDict):
    """Result from table detection for a single page."""

    page_number: int
    tables: list[DetectedTable]
    table_count: int
    has_tables: bool
    detection_time_ms: int
    detection_method: Literal["vlm", "heuristic", "hybrid"]
    notes: str


def create_empty_detected_table(
    table_id: str = "table_0",
    page_number: int = 1,
) -> DetectedTable:
    """Create an empty DetectedTable for initialization."""
    return DetectedTable(
        table_id=table_id,
        page_number=page_number,
        location=BoundingBox(x=0.0, y=0.0, width=0.0, height=0.0),
        row_count=0,
        column_count=0,
        confidence=0.0,
        headers=[],
        rows=[],
        has_header_row=False,
        has_total_row=False,
        has_merged_cells=False,
        table_type="unknown",
        description="",
        cell_borders_visible=False,
        extraction_quality="low",
        needs_review=False,
        review_reason="",
    )


def create_empty_table_detection_result(page_number: int = 1) -> TableDetectionResult:
    """Create an empty table detection result for initialization."""
    return TableDetectionResult(
        page_number=page_number,
        tables=[],
        table_count=0,
        has_tables=False,
        detection_time_ms=0,
        detection_method="vlm",
        notes="",
    )


def table_to_rows_dict(table: DetectedTable) -> list[dict[str, Any]]:
    """
    Convert a DetectedTable to a list of row dictionaries.

    Useful for export to DataFrame, CSV, or structured JSON.

    Args:
        table: DetectedTable to convert.

    Returns:
        List of dicts where keys are column headers and values are cell text.
    """
    header_names = [h["text"] for h in table.get("headers", [])]

    # Fallback column names if no headers
    if not header_names:
        col_count = table.get("column_count", 0)
        header_names = [f"col_{i}" for i in range(col_count)]

    result = []
    for row in table.get("rows", []):
        if row.get("is_header_row") or row.get("is_separator_row"):
            continue

        row_dict: dict[str, Any] = {}
        for cell in row.get("cells", []):
            col_idx = cell.get("col_index", 0)
            col_name = header_names[col_idx] if col_idx < len(header_names) else f"col_{col_idx}"
            row_dict[col_name] = cell.get("text", "")

        if row_dict:
            result.append(row_dict)

    return result
