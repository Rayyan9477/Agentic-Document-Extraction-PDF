"""
JSON exporter for document extraction results.

Provides comprehensive JSON export with:
- Multiple output formats (minimal, standard, detailed)
- Full metadata and audit trail
- HIPAA-compliant data handling
- Configurable field inclusion
"""

import json
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


class ExportFormat(str, Enum):
    """JSON export format options."""

    MINIMAL = "minimal"  # Values only
    STANDARD = "standard"  # Values + confidence + basic metadata
    DETAILED = "detailed"  # Full extraction details + audit trail
    FHIR_COMPATIBLE = "fhir_compatible"  # FHIR-style resource format


@dataclass(slots=True)
class JSONExportConfig:
    """
    Configuration for JSON export.

    Attributes:
        format: Export format (minimal/standard/detailed).
        include_metadata: Include processing metadata.
        include_confidence: Include confidence scores.
        include_validation: Include validation details.
        include_audit_trail: Include audit information.
        include_raw_passes: Include raw pass data.
        pretty_print: Use indented formatting.
        indent_size: Indentation size for pretty print.
        exclude_fields: Fields to exclude from output.
        mask_phi: Apply PHI masking to specified fields.
        phi_mask_pattern: Pattern to use for PHI masking.
    """

    format: ExportFormat = ExportFormat.STANDARD
    include_metadata: bool = True
    include_confidence: bool = True
    include_validation: bool = True
    include_audit_trail: bool = True
    include_raw_passes: bool = False
    pretty_print: bool = True
    indent_size: int = 2
    exclude_fields: set[str] = field(default_factory=set)
    mask_phi: bool = False
    phi_fields: set[str] = field(default_factory=lambda: {
        "ssn", "social_security", "member_id", "subscriber_id",
        "patient_account", "policy_number", "group_number",
    })
    phi_mask_pattern: str = "***MASKED***"


class JSONExporter:
    """
    Export extraction results to JSON format.

    Supports multiple output formats and comprehensive metadata
    for audit trail and compliance requirements.
    """

    def __init__(self, config: JSONExportConfig | None = None) -> None:
        """
        Initialize the JSON exporter.

        Args:
            config: Export configuration (uses defaults if not provided).
        """
        self.config = config or JSONExportConfig()
        self._logger = logger

    def export(
        self,
        state: ExtractionState,
        output_path: Path | str | None = None,
    ) -> dict[str, Any]:
        """
        Export extraction state to JSON.

        Args:
            state: Extraction state to export.
            output_path: Optional file path to write output.

        Returns:
            Exported JSON as dictionary.
        """
        self._logger.debug(
            "json_export_start",
            format=self.config.format.value,
            processing_id=state.get("processing_id", ""),
        )

        # Build export based on format
        if self.config.format == ExportFormat.MINIMAL:
            result = self._build_minimal_export(state)
        elif self.config.format == ExportFormat.STANDARD:
            result = self._build_standard_export(state)
        elif self.config.format == ExportFormat.DETAILED:
            result = self._build_detailed_export(state)
        elif self.config.format == ExportFormat.FHIR_COMPATIBLE:
            result = self._build_fhir_export(state)
        else:
            result = self._build_standard_export(state)

        # Apply PHI masking if enabled
        if self.config.mask_phi:
            result = self._apply_phi_masking(result)

        # Write to file if path provided
        if output_path:
            self._write_to_file(result, Path(output_path))

        self._logger.debug(
            "json_export_complete",
            processing_id=state.get("processing_id", ""),
            field_count=len(result.get("data", {})),
        )

        return result

    def _build_minimal_export(self, state: ExtractionState) -> dict[str, Any]:
        """Build minimal export with values only."""
        data = self._extract_values(state)

        return {
            "data": data,
            "processing_id": state.get("processing_id", ""),
            "status": state.get("status", ExtractionStatus.PENDING.value),
        }

    def _build_standard_export(self, state: ExtractionState) -> dict[str, Any]:
        """Build standard export with values, confidence, and basic metadata."""
        data = self._extract_values(state)
        field_metadata = self._extract_field_metadata(state)

        result: dict[str, Any] = {
            "data": data,
            "processing_id": state.get("processing_id", ""),
            "document_type": state.get("document_type", ""),
            "status": state.get("status", ExtractionStatus.PENDING.value),
        }

        if self.config.include_confidence:
            result["confidence"] = {
                "overall": state.get("overall_confidence", 0.0),
                "level": state.get("confidence_level", ConfidenceLevel.LOW.value),
                "fields": field_metadata,
            }

        if self.config.include_metadata:
            result["metadata"] = self._build_metadata(state)

        return result

    def _build_detailed_export(self, state: ExtractionState) -> dict[str, Any]:
        """Build detailed export with full information."""
        result = self._build_standard_export(state)

        # Add validation details
        if self.config.include_validation:
            result["validation"] = self._build_validation_details(state)

        # Add page-level details
        result["pages"] = self._build_page_details(state)

        # Add audit trail
        if self.config.include_audit_trail:
            result["audit"] = self._build_audit_trail(state)

        # Add raw pass data if requested
        if self.config.include_raw_passes:
            result["raw_passes"] = self._build_raw_passes(state)

        return result

    def _build_fhir_export(self, state: ExtractionState) -> dict[str, Any]:
        """Build FHIR-compatible resource format."""
        data = self._extract_values(state)

        return {
            "resourceType": "DocumentReference",
            "id": state.get("processing_id", ""),
            "status": self._map_status_to_fhir(state.get("status", "")),
            "docStatus": "final" if state.get("status") == ExtractionStatus.COMPLETED.value else "preliminary",
            "type": {
                "coding": [{
                    "system": "http://loinc.org",
                    "code": self._get_loinc_code(state.get("document_type", "")),
                    "display": state.get("document_type", "Unknown"),
                }]
            },
            "date": state.get("end_time") or state.get("start_time", ""),
            "content": [{
                "attachment": {
                    "contentType": "application/pdf",
                    "url": state.get("pdf_path", ""),
                },
                "format": {
                    "system": "urn:oid:1.3.6.1.4.1.19376.1.2.3",
                    "code": "urn:ihe:pcc:xds-ms:2007",
                    "display": "Medical Summary",
                }
            }],
            "context": {
                "related": [{
                    "identifier": {
                        "system": "urn:oid:2.16.840.1.113883.3.88.11.83.8",
                        "value": state.get("processing_id", ""),
                    }
                }]
            },
            "extension": [
                {
                    "url": "http://example.org/fhir/StructureDefinition/extraction-data",
                    "valueString": self._serialize_extraction_data(data),
                },
                {
                    "url": "http://example.org/fhir/StructureDefinition/extraction-confidence",
                    "valueDecimal": state.get("overall_confidence", 0.0),
                },
                {
                    "url": "http://example.org/fhir/StructureDefinition/extraction-status",
                    "valueCode": state.get("status", "unknown"),
                }
            ]
        }

    def _extract_values(self, state: ExtractionState) -> dict[str, Any]:
        """Extract field values from state."""
        merged = state.get("merged_extraction", {})
        values: dict[str, Any] = {}

        for field_name, field_data in merged.items():
            if field_name in self.config.exclude_fields:
                continue

            if isinstance(field_data, dict):
                values[field_name] = field_data.get("value")
            else:
                values[field_name] = field_data

        return values

    def _extract_field_metadata(self, state: ExtractionState) -> dict[str, dict[str, Any]]:
        """Extract per-field metadata."""
        field_meta = state.get("field_metadata", {})
        result: dict[str, dict[str, Any]] = {}

        for field_name, meta in field_meta.items():
            if field_name in self.config.exclude_fields:
                continue

            if isinstance(meta, dict):
                result[field_name] = {
                    "confidence": meta.get("confidence", 0.0),
                    "confidence_level": meta.get("confidence_level", "low"),
                    "passes_agree": meta.get("passes_agree", True),
                    "validation_passed": meta.get("validation_passed", True),
                }

        return result

    def _build_metadata(self, state: ExtractionState) -> dict[str, Any]:
        """Build processing metadata."""
        return {
            "pdf_path": state.get("pdf_path", ""),
            "pdf_hash": state.get("pdf_hash", ""),
            "schema_name": state.get("selected_schema_name", ""),
            "page_count": len(state.get("page_images", [])),
            "start_time": state.get("start_time", ""),
            "end_time": state.get("end_time"),
            "total_vlm_calls": state.get("total_vlm_calls", 0),
            "processing_time_ms": state.get("total_processing_time_ms", 0),
            "retry_count": state.get("retry_count", 0),
        }

    def _build_validation_details(self, state: ExtractionState) -> dict[str, Any]:
        """Build validation details."""
        validation = state.get("validation", {})
        return {
            "is_valid": validation.get("is_valid", False),
            "field_validations": validation.get("field_validations", {}),
            "cross_field_validations": validation.get("cross_field_validations", []),
            "hallucination_flags": validation.get("hallucination_flags", []),
            "warnings": validation.get("warnings", []),
            "errors": validation.get("errors", []),
        }

    def _build_page_details(self, state: ExtractionState) -> list[dict[str, Any]]:
        """Build page-level extraction details."""
        pages = state.get("page_extractions", [])
        return [
            {
                "page_number": page.get("page_number", 0),
                "field_count": len(page.get("merged_fields", {})),
                "confidence": page.get("overall_confidence", 0.0),
                "agreement_rate": page.get("agreement_rate", 0.0),
                "vlm_calls": page.get("vlm_calls", 0),
                "extraction_time_ms": page.get("extraction_time_ms", 0),
                "errors": page.get("errors", []),
            }
            for page in pages
        ]

    def _build_audit_trail(self, state: ExtractionState) -> dict[str, Any]:
        """Build audit trail information."""
        return {
            "processing_id": state.get("processing_id", ""),
            "started_at": state.get("start_time", ""),
            "completed_at": state.get("end_time"),
            "status": state.get("status", ""),
            "requires_human_review": state.get("requires_human_review", False),
            "human_review_reason": state.get("human_review_reason"),
            "errors": state.get("errors", []),
            "warnings": state.get("warnings", []),
            "export_timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def _build_raw_passes(self, state: ExtractionState) -> list[dict[str, Any]]:
        """Build raw pass extraction data."""
        pages = state.get("page_extractions", [])
        return [
            {
                "page_number": page.get("page_number", 0),
                "pass1_raw": page.get("pass1_raw", {}),
                "pass2_raw": page.get("pass2_raw", {}),
            }
            for page in pages
        ]

    def _apply_phi_masking(self, data: dict[str, Any]) -> dict[str, Any]:
        """Apply PHI masking to sensitive fields."""
        return self._mask_dict(data)

    def _mask_dict(self, obj: Any) -> Any:
        """Recursively mask PHI fields in a dictionary."""
        if isinstance(obj, dict):
            return {
                k: self._mask_value(k, v) if k.lower() in self.config.phi_fields
                else self._mask_dict(v)
                for k, v in obj.items()
            }
        elif isinstance(obj, list):
            return [self._mask_dict(item) for item in obj]
        return obj

    def _mask_value(self, field_name: str, value: Any) -> str:
        """
        Mask a PHI value completely for HIPAA compliance.

        IMPORTANT: Never expose any part of PHI data, including:
        - First/last characters (previous implementation was non-compliant)
        - Value length (can enable inference attacks)
        - Any derivable information

        Args:
            field_name: Name of the field being masked.
            value: Value to mask.

        Returns:
            Fully masked value with no PHI exposure.
        """
        # Log PHI masking for audit trail
        logger.debug(
            "phi_field_masked",
            field_name=field_name,
            had_value=value is not None,
        )

        # Return consistent mask - no partial exposure of any kind
        # Do NOT reveal length, first/last chars, or any other derivable info
        return self.config.phi_mask_pattern

    def _serialize_extraction_data(self, data: dict[str, Any]) -> str:
        """
        Serialize extraction data to JSON string for FHIR extension.

        Args:
            data: Extraction data dictionary.

        Returns:
            JSON string representation of the data.
        """
        import json
        return json.dumps(data, default=str, ensure_ascii=False)

    def _write_to_file(self, data: dict[str, Any], path: Path) -> None:
        """Write JSON data to file."""
        indent = self.config.indent_size if self.config.pretty_print else None

        path.parent.mkdir(parents=True, exist_ok=True)

        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f, indent=indent, ensure_ascii=False, default=str)

        self._logger.info(
            "json_file_written",
            path=str(path),
            size_bytes=path.stat().st_size,
        )

    def _map_status_to_fhir(self, status: str) -> str:
        """Map extraction status to FHIR document status."""
        mapping = {
            ExtractionStatus.COMPLETED.value: "current",
            ExtractionStatus.FAILED.value: "entered-in-error",
            ExtractionStatus.HUMAN_REVIEW.value: "preliminary",
            ExtractionStatus.PENDING.value: "preliminary",
        }
        return mapping.get(status, "preliminary")

    def _get_loinc_code(self, document_type: str) -> str:
        """Get LOINC code for document type."""
        loinc_codes = {
            "CMS-1500": "34117-2",  # History and physical note
            "UB-04": "11504-8",  # Surgical operation note
            "EOB": "34108-1",  # Outpatient note
            "SUPERBILL": "34117-2",  # History and physical note
        }
        return loinc_codes.get(document_type.upper(), "11488-4")


def export_to_json(
    state: ExtractionState,
    output_path: Path | str | None = None,
    format: ExportFormat = ExportFormat.STANDARD,
    include_metadata: bool = True,
    include_confidence: bool = True,
    pretty_print: bool = True,
) -> dict[str, Any]:
    """
    Convenience function to export extraction state to JSON.

    Args:
        state: Extraction state to export.
        output_path: Optional file path to write output.
        format: Export format (minimal/standard/detailed).
        include_metadata: Include processing metadata.
        include_confidence: Include confidence scores.
        pretty_print: Use indented formatting.

    Returns:
        Exported JSON as dictionary.

    Example:
        >>> result = export_to_json(state, "output.json", format=ExportFormat.DETAILED)
        >>> print(result["data"]["patient_name"])
    """
    config = JSONExportConfig(
        format=format,
        include_metadata=include_metadata,
        include_confidence=include_confidence,
        pretty_print=pretty_print,
    )

    exporter = JSONExporter(config)
    return exporter.export(state, output_path)
