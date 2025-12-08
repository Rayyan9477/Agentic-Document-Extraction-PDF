"""
Markdown exporter for document extraction results.

Provides comprehensive Markdown export with:
- Human-readable formatted reports
- Confidence highlighting with emoji indicators
- Validation status summaries
- Audit trail and metadata sections
- HIPAA-compliant data handling with PHI masking
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any

from src.config import get_logger
from src.pipeline.state import (
    ExtractionState,
    ExtractionStatus,
    ConfidenceLevel,
)


logger = get_logger(__name__)


class MarkdownStyle(str, Enum):
    """Markdown output style options."""

    SIMPLE = "simple"  # Basic text report
    DETAILED = "detailed"  # Full report with all metadata
    SUMMARY = "summary"  # Executive summary only
    TECHNICAL = "technical"  # Technical details for debugging


@dataclass(slots=True)
class MarkdownExportConfig:
    """
    Configuration for Markdown export.

    Attributes:
        style: Output style (simple/detailed/summary/technical).
        include_toc: Include table of contents.
        include_confidence_indicators: Show emoji confidence indicators.
        include_validation_details: Include validation results.
        include_audit_trail: Include audit information.
        include_raw_values: Include raw pass values.
        mask_phi: Apply PHI masking to specified fields.
        phi_fields: Fields considered PHI for masking.
        phi_mask_pattern: Pattern to use for PHI masking.
        header_level: Starting header level (1-6).
    """

    style: MarkdownStyle = MarkdownStyle.DETAILED
    include_toc: bool = True
    include_confidence_indicators: bool = True
    include_validation_details: bool = True
    include_audit_trail: bool = True
    include_raw_values: bool = False
    mask_phi: bool = False
    phi_fields: set[str] = field(default_factory=lambda: {
        "ssn", "social_security", "member_id", "subscriber_id",
        "patient_account", "policy_number", "group_number",
        "patient_name", "patient_dob", "patient_address",
    })
    phi_mask_pattern: str = "[REDACTED]"
    header_level: int = 1


class MarkdownExporter:
    """
    Export extraction results to Markdown format.

    Produces human-readable reports suitable for documentation,
    review workflows, and audit purposes.
    """

    CONFIDENCE_EMOJI = {
        ConfidenceLevel.HIGH: "✅",
        ConfidenceLevel.MEDIUM: "⚠️",
        ConfidenceLevel.LOW: "❌",
    }

    STATUS_EMOJI = {
        "completed": "✅",
        "failed": "❌",
        "pending": "⏳",
        "processing": "🔄",
        "human_review": "👁️",
    }

    def __init__(self, config: MarkdownExportConfig | None = None) -> None:
        """
        Initialize the Markdown exporter.

        Args:
            config: Export configuration (uses defaults if not provided).
        """
        self.config = config or MarkdownExportConfig()
        self._logger = logger

    def export(
        self,
        state: ExtractionState,
        output_path: Path | str | None = None,
    ) -> str:
        """
        Export extraction state to Markdown.

        Args:
            state: Extraction state to export.
            output_path: Optional path to write the file.

        Returns:
            Markdown formatted string.
        """
        self._logger.info(
            "markdown_export_start",
            style=self.config.style.value,
            output_path=str(output_path) if output_path else None,
        )

        sections: list[str] = []

        # Build report based on style
        if self.config.style == MarkdownStyle.SUMMARY:
            sections.extend(self._build_summary_report(state))
        elif self.config.style == MarkdownStyle.SIMPLE:
            sections.extend(self._build_simple_report(state))
        elif self.config.style == MarkdownStyle.TECHNICAL:
            sections.extend(self._build_technical_report(state))
        else:
            sections.extend(self._build_detailed_report(state))

        content = "\n\n".join(sections)

        # Add TOC if requested
        if self.config.include_toc and self.config.style in (
            MarkdownStyle.DETAILED,
            MarkdownStyle.TECHNICAL,
        ):
            content = self._add_toc(content)

        # Write to file if path provided
        if output_path:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            output_file.write_text(content, encoding="utf-8")
            self._logger.info("markdown_export_complete", path=str(output_file))

        return content

    def _h(self, level: int, text: str) -> str:
        """Create a header at the specified level."""
        adjusted_level = min(level + self.config.header_level - 1, 6)
        return f"{'#' * adjusted_level} {text}"

    def _build_summary_report(self, state: ExtractionState) -> list[str]:
        """Build executive summary report."""
        sections = []

        # Title
        doc_type = state.get("document_type", "Document")
        status = state.get("status", "unknown")
        status_emoji = self.STATUS_EMOJI.get(status, "")
        sections.append(self._h(1, f"{status_emoji} {doc_type} Extraction Summary"))

        # Key metrics
        overall_conf = state.get("overall_confidence", 0.0)
        conf_level = self._get_confidence_level(overall_conf)
        conf_emoji = self.CONFIDENCE_EMOJI.get(conf_level, "")

        sections.append(f"**Status**: {status.replace('_', ' ').title()}")
        sections.append(f"**Overall Confidence**: {conf_emoji} {overall_conf:.1%}")

        # Field count
        merged = state.get("merged_extraction", {})
        sections.append(f"**Fields Extracted**: {len(merged)}")

        # Validation summary
        validation = state.get("validation", {})
        if validation:
            is_valid = validation.get("is_valid", False)
            val_status = "" if is_valid else ""
            sections.append(f"**Validation**: {val_status} {'Passed' if is_valid else 'Failed'}")

        # Human review status
        if state.get("requires_human_review"):
            reason = state.get("human_review_reason", "")
            sections.append(f"** Requires Human Review**: {reason}")

        return sections

    def _build_simple_report(self, state: ExtractionState) -> list[str]:
        """Build simple text report."""
        sections = []

        # Title
        doc_type = state.get("document_type", "Document")
        sections.append(self._h(1, f"{doc_type} Extraction Results"))

        # Processing info
        sections.append(self._h(2, "Processing Information"))
        info_lines = [
            f"- **Processing ID**: `{state.get('processing_id', 'N/A')}`",
            f"- **Document Type**: {doc_type}",
            f"- **Status**: {state.get('status', 'unknown').replace('_', ' ').title()}",
            f"- **Confidence**: {state.get('overall_confidence', 0.0):.1%}",
        ]
        sections.append("\n".join(info_lines))

        # Extracted data
        sections.append(self._h(2, "Extracted Data"))
        sections.append(self._format_extracted_data(state))

        return sections

    def _build_detailed_report(self, state: ExtractionState) -> list[str]:
        """Build detailed report with all information."""
        sections = []

        # Title and summary
        doc_type = state.get("document_type", "Document")
        processing_id = state.get("processing_id", "N/A")
        sections.append(self._h(1, f"{doc_type} Extraction Report"))
        sections.append(f"*Processing ID: `{processing_id}`*")

        # Executive summary
        sections.append(self._h(2, "Executive Summary"))
        sections.extend(self._build_summary_report(state)[1:])

        # Processing metadata
        sections.append(self._h(2, "Processing Metadata"))
        sections.append(self._format_metadata(state))

        # Extracted data with confidence
        sections.append(self._h(2, "Extracted Data"))
        sections.append(self._format_extracted_data_detailed(state))

        # Validation results
        if self.config.include_validation_details:
            sections.append(self._h(2, "Validation Results"))
            sections.append(self._format_validation(state))

        # Audit trail
        if self.config.include_audit_trail:
            sections.append(self._h(2, "Audit Trail"))
            sections.append(self._format_audit_trail(state))

        # Errors and warnings
        errors = state.get("errors", [])
        warnings = state.get("warnings", [])
        if errors or warnings:
            sections.append(self._h(2, "Issues"))
            if errors:
                sections.append(self._h(3, "Errors"))
                sections.append("\n".join(f"- {e}" for e in errors))
            if warnings:
                sections.append(self._h(3, "Warnings"))
                sections.append("\n".join(f"- {w}" for w in warnings))

        return sections

    def _build_technical_report(self, state: ExtractionState) -> list[str]:
        """Build technical report for debugging."""
        sections = []

        # Title
        sections.append(self._h(1, "Technical Extraction Report"))

        # Include all from detailed
        sections.extend(self._build_detailed_report(state)[1:])

        # Raw pass data
        if self.config.include_raw_values:
            sections.append(self._h(2, "Raw Pass Data"))
            sections.append(self._format_raw_passes(state))

        # Page extractions
        page_extractions = state.get("page_extractions", [])
        if page_extractions:
            sections.append(self._h(2, "Page-by-Page Extractions"))
            for page_data in page_extractions:
                page_num = page_data.get("page_number", "?")
                sections.append(self._h(3, f"Page {page_num}"))
                sections.append(self._format_page_extraction(page_data))

        # Processing metrics
        sections.append(self._h(2, "Performance Metrics"))
        metrics = {
            "Total VLM Calls": state.get("total_vlm_calls", 0),
            "Processing Time (ms)": state.get("total_processing_time_ms", 0),
            "Retry Count": state.get("retry_count", 0),
            "Page Count": len(state.get("page_images", [])),
        }
        sections.append(self._format_key_value_table(metrics))

        return sections

    def _format_extracted_data(self, state: ExtractionState) -> str:
        """Format extracted data as simple list."""
        merged = state.get("merged_extraction", {})
        if not merged:
            return "*No data extracted*"

        lines = []
        for field_name, field_data in sorted(merged.items()):
            value = self._extract_value(field_data)
            display_value = self._mask_if_phi(field_name, value)
            lines.append(f"- **{self._format_field_name(field_name)}**: {display_value}")

        return "\n".join(lines)

    def _format_extracted_data_detailed(self, state: ExtractionState) -> str:
        """Format extracted data with confidence indicators."""
        merged = state.get("merged_extraction", {})
        field_meta = state.get("field_metadata", {})

        if not merged:
            return "*No data extracted*"

        # Build table
        lines = [
            "| Field | Value | Confidence | Status |",
            "|-------|-------|------------|--------|",
        ]

        for field_name, field_data in sorted(merged.items()):
            value = self._extract_value(field_data)
            display_value = self._mask_if_phi(field_name, value)

            # Get metadata
            meta = field_meta.get(field_name, {})
            if isinstance(meta, dict):
                confidence = meta.get("confidence", 0.0)
                validation_passed = meta.get("validation_passed", True)
                passes_agree = meta.get("passes_agree", True)
            else:
                confidence = 0.0
                validation_passed = True
                passes_agree = True

            # Confidence indicator
            conf_level = self._get_confidence_level(confidence)
            if self.config.include_confidence_indicators:
                conf_emoji = self.CONFIDENCE_EMOJI.get(conf_level, "")
                conf_str = f"{conf_emoji} {confidence:.0%}"
            else:
                conf_str = f"{confidence:.0%}"

            # Status
            status_parts = []
            if validation_passed:
                status_parts.append(" Valid")
            else:
                status_parts.append(" Invalid")
            if not passes_agree:
                status_parts.append(" Mismatch")

            status_str = ", ".join(status_parts)

            lines.append(
                f"| {self._format_field_name(field_name)} | "
                f"{display_value} | {conf_str} | {status_str} |"
            )

        return "\n".join(lines)

    def _format_metadata(self, state: ExtractionState) -> str:
        """Format processing metadata."""
        metadata = {
            "Processing ID": f"`{state.get('processing_id', 'N/A')}`",
            "PDF Path": f"`{state.get('pdf_path', 'N/A')}`",
            "PDF Hash": f"`{state.get('pdf_hash', 'N/A')[:16]}...`" if state.get("pdf_hash") else "N/A",
            "Document Type": state.get("document_type", "Unknown"),
            "Schema": state.get("selected_schema_name", "Auto-detected"),
            "Status": state.get("status", "unknown").replace("_", " ").title(),
            "Start Time": state.get("start_time", "N/A"),
            "End Time": state.get("end_time", "N/A"),
        }
        return self._format_key_value_list(metadata)

    def _format_validation(self, state: ExtractionState) -> str:
        """Format validation results."""
        validation = state.get("validation", {})
        if not validation:
            return "*No validation data available*"

        lines = []

        # Overall status
        is_valid = validation.get("is_valid", False)
        status_emoji = "" if is_valid else ""
        lines.append(f"**Overall Status**: {status_emoji} {'Passed' if is_valid else 'Failed'}")

        # Field validations
        field_validations = validation.get("field_validations", {})
        if field_validations:
            lines.append("")
            lines.append(self._h(3, "Field Validations"))
            for field_name, result in sorted(field_validations.items()):
                if isinstance(result, dict):
                    is_field_valid = result.get("is_valid", True)
                    val_type = result.get("validation_type", "")
                    emoji = "" if is_field_valid else ""
                    type_info = f" ({val_type})" if val_type else ""
                    lines.append(f"- {emoji} **{self._format_field_name(field_name)}**{type_info}")

        # Cross-field validations
        cross_field = validation.get("cross_field_validations", [])
        if cross_field:
            lines.append("")
            lines.append(self._h(3, "Cross-Field Validations"))
            for item in cross_field:
                lines.append(f"- {item}")

        # Hallucination flags
        hallucination_flags = validation.get("hallucination_flags", [])
        if hallucination_flags:
            lines.append("")
            lines.append(self._h(3, " Hallucination Flags"))
            for flag in hallucination_flags:
                lines.append(f"- {flag}")

        return "\n".join(lines)

    def _format_audit_trail(self, state: ExtractionState) -> str:
        """Format audit trail information."""
        lines = []

        # Timestamps
        lines.append(self._h(3, "Timeline"))
        timeline = []
        if state.get("start_time"):
            timeline.append(f"- **Started**: {state.get('start_time')}")
        if state.get("end_time"):
            timeline.append(f"- **Completed**: {state.get('end_time')}")

        processing_time = state.get("total_processing_time_ms", 0)
        if processing_time:
            timeline.append(f"- **Duration**: {processing_time / 1000:.2f} seconds")

        lines.extend(timeline)

        # Processing details
        lines.append("")
        lines.append(self._h(3, "Processing Details"))
        details = [
            f"- **VLM Calls**: {state.get('total_vlm_calls', 0)}",
            f"- **Retry Count**: {state.get('retry_count', 0)}",
            f"- **Pages Processed**: {len(state.get('page_images', []))}",
        ]
        lines.extend(details)

        # Export timestamp
        lines.append("")
        lines.append(f"*Report generated: {datetime.now(timezone.utc).isoformat()}*")

        return "\n".join(lines)

    def _format_raw_passes(self, state: ExtractionState) -> str:
        """Format raw pass data for technical report."""
        page_extractions = state.get("page_extractions", [])
        if not page_extractions:
            return "*No raw pass data available*"

        lines = []
        for page_data in page_extractions:
            page_num = page_data.get("page_number", "?")

            pass1_raw = page_data.get("pass1_raw", {})
            pass2_raw = page_data.get("pass2_raw", {})

            if pass1_raw or pass2_raw:
                lines.append(self._h(3, f"Page {page_num}"))

                if pass1_raw:
                    lines.append("**Pass 1:**")
                    lines.append("```json")
                    lines.append(self._format_json_block(pass1_raw))
                    lines.append("```")

                if pass2_raw:
                    lines.append("**Pass 2:**")
                    lines.append("```json")
                    lines.append(self._format_json_block(pass2_raw))
                    lines.append("```")

        return "\n".join(lines) if lines else "*No raw data*"

    def _format_page_extraction(self, page_data: dict[str, Any]) -> str:
        """Format page extraction data."""
        lines = []

        confidence = page_data.get("overall_confidence", 0.0)
        agreement = page_data.get("agreement_rate", 0.0)
        vlm_calls = page_data.get("vlm_calls", 0)
        time_ms = page_data.get("extraction_time_ms", 0)

        lines.append(f"- **Confidence**: {confidence:.1%}")
        lines.append(f"- **Agreement Rate**: {agreement:.1%}")
        lines.append(f"- **VLM Calls**: {vlm_calls}")
        lines.append(f"- **Extraction Time**: {time_ms}ms")

        merged_fields = page_data.get("merged_fields", {})
        if merged_fields:
            lines.append("")
            lines.append("**Fields:**")
            for field_name, value in sorted(merged_fields.items()):
                display_value = self._mask_if_phi(field_name, str(value))
                lines.append(f"- {self._format_field_name(field_name)}: {display_value}")

        return "\n".join(lines)

    def _format_key_value_list(self, data: dict[str, Any]) -> str:
        """Format key-value pairs as a list."""
        return "\n".join(f"- **{k}**: {v}" for k, v in data.items())

    def _format_key_value_table(self, data: dict[str, Any]) -> str:
        """Format key-value pairs as a table."""
        lines = ["| Metric | Value |", "|--------|-------|"]
        for k, v in data.items():
            lines.append(f"| {k} | {v} |")
        return "\n".join(lines)

    def _format_json_block(self, data: dict[str, Any]) -> str:
        """Format data as JSON for code block."""
        import json
        return json.dumps(data, indent=2, default=str)

    def _format_field_name(self, field_name: str) -> str:
        """Format field name for display."""
        return field_name.replace("_", " ").title()

    def _extract_value(self, field_data: Any) -> str:
        """Extract value from field data."""
        if isinstance(field_data, dict):
            return str(field_data.get("value", ""))
        return str(field_data) if field_data is not None else ""

    def _mask_if_phi(self, field_name: str, value: str) -> str:
        """Mask value if it's a PHI field and masking is enabled."""
        if not self.config.mask_phi:
            return value

        field_lower = field_name.lower()
        for phi_field in self.config.phi_fields:
            if phi_field in field_lower:
                if len(value) > 4:
                    return f"{value[:2]}{self.config.phi_mask_pattern}{value[-2:]}"
                return self.config.phi_mask_pattern
        return value

    def _get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Get confidence level from score."""
        if confidence >= 0.85:
            return ConfidenceLevel.HIGH
        elif confidence >= 0.50:
            return ConfidenceLevel.MEDIUM
        return ConfidenceLevel.LOW

    def _add_toc(self, content: str) -> str:
        """Add table of contents to content."""
        lines = content.split("\n")
        toc_lines = ["## Table of Contents", ""]

        for line in lines:
            if line.startswith("## "):
                title = line[3:].strip()
                anchor = title.lower().replace(" ", "-").replace(":", "")
                toc_lines.append(f"- [{title}](#{anchor})")
            elif line.startswith("### "):
                title = line[4:].strip()
                anchor = title.lower().replace(" ", "-").replace(":", "")
                toc_lines.append(f"  - [{title}](#{anchor})")

        toc_lines.append("")

        # Insert TOC after first header
        result_lines = []
        toc_inserted = False
        for line in lines:
            result_lines.append(line)
            if not toc_inserted and line.startswith("# "):
                result_lines.append("")
                result_lines.extend(toc_lines)
                toc_inserted = True

        return "\n".join(result_lines)


def export_to_markdown(
    state: ExtractionState,
    output_path: Path | str | None = None,
    style: MarkdownStyle = MarkdownStyle.DETAILED,
    include_confidence_indicators: bool = True,
    include_validation: bool = True,
    mask_phi: bool = False,
) -> str:
    """
    Convenience function to export extraction state to Markdown.

    Args:
        state: Extraction state to export.
        output_path: Optional path to write the file.
        style: Output style.
        include_confidence_indicators: Show emoji indicators.
        include_validation: Include validation details.
        mask_phi: Apply PHI masking.

    Returns:
        Markdown formatted string.
    """
    config = MarkdownExportConfig(
        style=style,
        include_confidence_indicators=include_confidence_indicators,
        include_validation_details=include_validation,
        mask_phi=mask_phi,
    )
    exporter = MarkdownExporter(config)
    return exporter.export(state, output_path)
