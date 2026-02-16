"""
Schema management API routes.

Provides endpoints for listing and managing
extraction schemas, plus the schema suggestion wizard.
"""

from typing import Any

from fastapi import APIRouter, HTTPException, Request

from src.api.models import (
    SchemaInfo,
    SchemaListResponse,
    SchemaProposalResponse,
    SchemaRefineRequest,
    SchemaSaveRequest,
    SchemaSuggestRequest,
)
from src.config import get_logger
from src.security.path_validator import (
    PathTraversalError,
    PathValidationError,
    SecurePathValidator,
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
    if isinstance(fields, dict) or isinstance(fields, list):
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
                schemas.append(
                    SchemaInfo(
                        name=schema.name,
                        description=getattr(schema, "description", ""),
                        document_type=getattr(schema, "document_type", schema.name),
                        field_count=len(getattr(schema, "fields", [])),
                        version=getattr(schema, "version", "1.0.0"),
                    )
                )
            elif isinstance(schema, dict):
                schemas.append(_get_schema_info(schema.get("name", "unknown"), schema))

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
            detail=f"Failed to list schemas: {e!s}",
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
            detail=f"Failed to get schema: {e!s}",
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
            return [{"name": name, **field_def} for name, field_def in fields.items()]
        if isinstance(fields, list):
            return fields
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
            detail=f"Failed to get schema fields: {e!s}",
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
            detail=f"Detection failed: {e!s}",
        )


# ──────────────────────────────────────────────────────────────────
# Schema Wizard Endpoints (Phase 2C)
# ──────────────────────────────────────────────────────────────────

# Singleton agent — created lazily to avoid import-time VLM connection
_schema_proposal_agent = None


def _get_proposal_agent():
    """Get or create the schema proposal agent singleton."""
    global _schema_proposal_agent
    if _schema_proposal_agent is None:
        from src.agents.schema_proposal import SchemaProposalAgent
        _schema_proposal_agent = SchemaProposalAgent()
    return _schema_proposal_agent


@router.post(
    "/schemas/suggest",
    response_model=SchemaProposalResponse,
    summary="Suggest extraction schema",
    description="Analyze a document image and propose an extraction schema.",
)
async def suggest_schema(
    body: SchemaSuggestRequest,
    http_request: Request,
) -> SchemaProposalResponse:
    """
    Analyze a document and suggest an extraction schema.

    Args:
        body: Request with base64-encoded image and optional context.
        http_request: HTTP request object.

    Returns:
        Schema proposal with suggested fields.
    """
    request_id = getattr(http_request.state, "request_id", "")

    logger.info("suggest_schema_request", request_id=request_id)

    try:
        agent = _get_proposal_agent()
        proposal = agent.suggest(
            image_data=body.image_base64,
            context=body.context,
        )

        data = proposal.to_dict()
        return SchemaProposalResponse(
            proposal_id=data["proposal_id"],
            schema_name=data["schema_name"],
            document_type_description=data["document_type_description"],
            fields=data["fields"],
            field_count=data["field_count"],
            groups=data["groups"],
            cross_field_rules=data["cross_field_rules"],
            confidence=data["confidence"],
            revision=data["revision"],
            status=data["status"],
        )

    except Exception as e:
        logger.error(
            "suggest_schema_error",
            request_id=request_id,
            error=str(e),
        )
        raise HTTPException(
            status_code=500,
            detail=f"Schema suggestion failed: {e!s}",
        )


@router.post(
    "/schemas/proposals/{proposal_id}/refine",
    response_model=SchemaProposalResponse,
    summary="Refine schema proposal",
    description="Apply feedback to refine an existing schema proposal.",
)
async def refine_schema(
    proposal_id: str,
    body: SchemaRefineRequest,
    http_request: Request,
) -> SchemaProposalResponse:
    """
    Refine an existing schema proposal with user feedback.

    Args:
        proposal_id: ID of the proposal to refine.
        body: Request with feedback and optional image.
        http_request: HTTP request object.

    Returns:
        Updated schema proposal.
    """
    request_id = getattr(http_request.state, "request_id", "")

    logger.info(
        "refine_schema_request",
        request_id=request_id,
        proposal_id=proposal_id,
    )

    try:
        agent = _get_proposal_agent()
        proposal = agent.refine(
            proposal_id=proposal_id,
            feedback=body.feedback,
            image_data=body.image_base64,
        )

        data = proposal.to_dict()
        return SchemaProposalResponse(
            proposal_id=data["proposal_id"],
            schema_name=data["schema_name"],
            document_type_description=data["document_type_description"],
            fields=data["fields"],
            field_count=data["field_count"],
            groups=data["groups"],
            cross_field_rules=data["cross_field_rules"],
            confidence=data["confidence"],
            revision=data["revision"],
            status=data["status"],
        )

    except Exception as e:
        error_str = str(e)
        if "not found" in error_str.lower():
            raise HTTPException(status_code=404, detail=error_str)

        logger.error(
            "refine_schema_error",
            request_id=request_id,
            proposal_id=proposal_id,
            error=error_str,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Schema refinement failed: {error_str}",
        )


@router.post(
    "/schemas/proposals/{proposal_id}/save",
    response_model=dict[str, Any],
    summary="Save schema proposal",
    description="Convert a proposal into a registered schema definition.",
)
async def save_schema_proposal(
    proposal_id: str,
    body: SchemaSaveRequest,
    http_request: Request,
) -> dict[str, Any]:
    """
    Save a schema proposal as a reusable schema definition.

    Args:
        proposal_id: ID of the proposal to save.
        body: Request with optional schema name override.
        http_request: HTTP request object.

    Returns:
        Saved schema definition.
    """
    request_id = getattr(http_request.state, "request_id", "")

    logger.info(
        "save_schema_request",
        request_id=request_id,
        proposal_id=proposal_id,
    )

    try:
        agent = _get_proposal_agent()
        schema_def = agent.save(
            proposal_id=proposal_id,
            schema_name=body.schema_name,
        )

        return {
            "status": "saved",
            "proposal_id": proposal_id,
            "schema": schema_def,
        }

    except Exception as e:
        error_str = str(e)
        if "not found" in error_str.lower():
            raise HTTPException(status_code=404, detail=error_str)

        logger.error(
            "save_schema_error",
            request_id=request_id,
            proposal_id=proposal_id,
            error=error_str,
        )
        raise HTTPException(
            status_code=500,
            detail=f"Schema save failed: {error_str}",
        )


@router.get(
    "/schemas/proposals",
    response_model=list[dict[str, Any]],
    summary="List schema proposals",
    description="List all cached schema proposals.",
)
async def list_proposals(
    http_request: Request,
) -> list[dict[str, Any]]:
    """List all cached schema proposals."""
    request_id = getattr(http_request.state, "request_id", "")
    logger.info("list_proposals_request", request_id=request_id)

    agent = _get_proposal_agent()
    return agent.list_proposals()


@router.get(
    "/schemas/proposals/{proposal_id}",
    response_model=dict[str, Any],
    summary="Get schema proposal",
    description="Get a specific schema proposal by ID.",
)
async def get_proposal(
    proposal_id: str,
    http_request: Request,
) -> dict[str, Any]:
    """Get a specific schema proposal."""
    request_id = getattr(http_request.state, "request_id", "")
    logger.info(
        "get_proposal_request",
        request_id=request_id,
        proposal_id=proposal_id,
    )

    agent = _get_proposal_agent()
    proposal = agent.get_proposal(proposal_id)
    if proposal is None:
        raise HTTPException(
            status_code=404,
            detail=f"Proposal not found: {proposal_id}",
        )

    return proposal.to_dict()


@router.delete(
    "/schemas/proposals/{proposal_id}",
    summary="Delete schema proposal",
    description="Delete a cached schema proposal.",
)
async def delete_proposal(
    proposal_id: str,
    http_request: Request,
) -> dict[str, str]:
    """Delete a schema proposal."""
    request_id = getattr(http_request.state, "request_id", "")
    logger.info(
        "delete_proposal_request",
        request_id=request_id,
        proposal_id=proposal_id,
    )

    agent = _get_proposal_agent()
    if not agent.delete_proposal(proposal_id):
        raise HTTPException(
            status_code=404,
            detail=f"Proposal not found: {proposal_id}",
        )

    return {"status": "deleted", "proposal_id": proposal_id}
