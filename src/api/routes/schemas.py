"""
Schema management API routes.

Provides endpoints for listing and managing
extraction schemas.
"""

from typing import Any

from fastapi import APIRouter, HTTPException, Request

from src.api.models import SchemaInfo, SchemaListResponse
from src.config import get_logger
from src.security.path_validator import (
    SecurePathValidator,
    PathTraversalError,
    PathValidationError,
)


logger = get_logger(__name__)
router = APIRouter()

# SECURITY: Initialize path validator for PDF paths
_pdf_validator = SecurePathValidator(
    allowed_extensions=[".pdf"],
    allow_absolute_paths=True,
    resolve_symlinks=True,
)


def _get_schema_info(schema_name: str, schema_def: dict[str, Any]) -> SchemaInfo:
    """Build SchemaInfo from schema definition."""
    fields = schema_def.get("fields", {})
    if isinstance(fields, dict):
        field_count = len(fields)
    elif isinstance(fields, list):
        field_count = len(fields)
    else:
        field_count = 0

    return SchemaInfo(
        name=schema_name,
        description=schema_def.get("description", ""),
        document_type=schema_def.get("document_type", schema_name),
        field_count=field_count,
        version=schema_def.get("version", "1.0.0"),
    )


@router.get(
    "/schemas",
    response_model=SchemaListResponse,
    summary="List schemas",
    description="List all available extraction schemas.",
)
async def list_schemas(
    http_request: Request,
) -> SchemaListResponse:
    """
    List all available extraction schemas.

    Args:
        http_request: HTTP request object.

    Returns:
        List of available schemas.
    """
    request_id = getattr(http_request.state, "request_id", "")

    logger.info(
        "list_schemas_request",
        request_id=request_id,
    )

    try:
        from src.schemas import get_all_schemas

        all_schemas = get_all_schemas()
        schemas = []
        for schema in all_schemas:
            # Handle both DocumentSchema objects and dict schemas
            if hasattr(schema, "name"):
                schemas.append(SchemaInfo(
                    name=schema.name,
                    description=getattr(schema, "description", ""),
                    document_type=getattr(schema, "document_type", schema.name),
                    field_count=len(getattr(schema, "fields", [])),
                    version=getattr(schema, "version", "1.0.0"),
                ))
            elif isinstance(schema, dict):
                schemas.append(_get_schema_info(
                    schema.get("name", "unknown"),
                    schema
                ))

        return SchemaListResponse(
            schemas=schemas,
            count=len(schemas),
        )

    except Exception as e:
        logger.error(
            "list_schemas_error",
            request_id=request_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to list schemas: {str(e)}",
        )


@router.get(
    "/schemas/{schema_name}",
    response_model=dict[str, Any],
    summary="Get schema",
    description="Get a specific extraction schema.",
)
async def get_schema(
    schema_name: str,
    http_request: Request,
) -> dict[str, Any]:
    """
    Get a specific extraction schema.

    Args:
        schema_name: Name of the schema.
        http_request: HTTP request object.

    Returns:
        Schema definition.

    Raises:
        HTTPException: If schema not found.
    """
    request_id = getattr(http_request.state, "request_id", "")

    logger.info(
        "get_schema_request",
        request_id=request_id,
        schema_name=schema_name,
    )

    try:
        from src.schemas import get_schema

        schema = get_schema(schema_name)
        if not schema:
            raise HTTPException(
                status_code=404,
                detail=f"Schema not found: {schema_name}",
            )

        return schema

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "get_schema_error",
            request_id=request_id,
            schema_name=schema_name,
            error=str(e),
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get schema: {str(e)}",
        )


@router.get(
    "/schemas/{schema_name}/fields",
    response_model=list[dict[str, Any]],
    summary="Get schema fields",
    description="Get the fields defined in a schema.",
)
async def get_schema_fields(
    schema_name: str,
    http_request: Request,
) -> list[dict[str, Any]]:
    """
    Get the fields defined in a schema.

    Args:
        schema_name: Name of the schema.
        http_request: HTTP request object.

    Returns:
        List of field definitions.

    Raises:
        HTTPException: If schema not found.
    """
    request_id = getattr(http_request.state, "request_id", "")

    logger.info(
        "get_schema_fields_request",
        request_id=request_id,
        schema_name=schema_name,
    )

    try:
        from src.schemas import get_schema

        schema = get_schema(schema_name)
        if not schema:
            raise HTTPException(
                status_code=404,
                detail=f"Schema not found: {schema_name}",
            )

        fields = schema.get("fields", {})

        if isinstance(fields, dict):
            return [
                {"name": name, **field_def}
                for name, field_def in fields.items()
            ]
        elif isinstance(fields, list):
            return fields
        else:
            return []

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "get_schema_fields_error",
            request_id=request_id,
            schema_name=schema_name,
            error=str(e),
        )
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get schema fields: {str(e)}",
        )


@router.post(
    "/schemas/detect",
    response_model=dict[str, Any],
    summary="Detect document type",
    description="Detect the document type and suggest a schema.",
)
async def detect_schema(
    http_request: Request,
    pdf_path: str,
) -> dict[str, Any]:
    """
    Detect the document type and suggest a schema.

    Args:
        http_request: HTTP request object.
        pdf_path: Path to the PDF file.

    Returns:
        Detection result with suggested schema.

    Raises:
        HTTPException: If file not found or detection fails.
    """
    request_id = getattr(http_request.state, "request_id", "")

    logger.info(
        "detect_schema_request",
        request_id=request_id,
        pdf_path=pdf_path,
    )

    from pathlib import Path

    # SECURITY: Validate path for traversal attacks before any file operations
    try:
        validated_path = _pdf_validator.validate(pdf_path)
    except PathTraversalError as e:
        logger.warning(
            "detect_schema_path_traversal",
            request_id=request_id,
            path=pdf_path[:100],  # Truncate for safe logging
            error=str(e),
        )
        raise HTTPException(
            status_code=400,
            detail="Invalid file path",  # Generic message to prevent info disclosure
        )
    except PathValidationError as e:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file path: {e}",
        )

    if not validated_path.exists():
        raise HTTPException(
            status_code=404,
            detail=f"PDF file not found: {pdf_path}",
        )

    try:
        from src.agents.classifier import DocumentClassifier

        classifier = DocumentClassifier()
        result = classifier.classify(str(validated_path))

        return {
            "pdf_path": str(validated_path),
            "detected_type": result.get("document_type", "unknown"),
            "confidence": result.get("confidence", 0.0),
            "suggested_schema": result.get("suggested_schema"),
            "alternative_schemas": result.get("alternative_schemas", []),
        }

    except Exception as e:
        logger.error(
            "detect_schema_error",
            request_id=request_id,
            pdf_path=str(validated_path),
            error=str(e),
        )
        raise HTTPException(
            status_code=500,
            detail=f"Detection failed: {str(e)}",
        )
