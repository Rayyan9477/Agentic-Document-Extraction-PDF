"""
Schema Proposal Agent — interactive schema suggestion wizard.

Allows users to:
1. Upload a document sample and get a proposed extraction schema
2. Refine the proposal iteratively (add/remove/rename fields)
3. Save the finalized schema to the registry for future use

Builds on SchemaGeneratorAgent (pipeline-internal, zero-shot) by adding
a user-facing conversational loop with persistent proposal state.
"""

import json
import secrets
import time
from dataclasses import dataclass, field
from typing import Any

from src.agents.base import AgentError, BaseAgent
from src.agents.utils import RetryConfig, retry_with_backoff
from src.client.lm_client import LMStudioClient
from src.config import get_logger, get_settings
from src.pipeline.state import ExtractionState, update_state
from src.schemas.field_types import FieldDefinition, FieldType


logger = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────
# Data structures
# ──────────────────────────────────────────────────────────────────

FIELD_TYPE_MAP: dict[str, FieldType] = {
    "string": FieldType.STRING,
    "text": FieldType.STRING,
    "integer": FieldType.INTEGER,
    "int": FieldType.INTEGER,
    "float": FieldType.FLOAT,
    "number": FieldType.FLOAT,
    "boolean": FieldType.BOOLEAN,
    "bool": FieldType.BOOLEAN,
    "date": FieldType.DATE,
    "time": FieldType.TIME,
    "datetime": FieldType.DATETIME,
    "currency": FieldType.CURRENCY,
    "money": FieldType.CURRENCY,
    "percentage": FieldType.PERCENTAGE,
    "percent": FieldType.PERCENTAGE,
    "phone": FieldType.PHONE,
    "email": FieldType.EMAIL,
    "address": FieldType.ADDRESS,
    "name": FieldType.NAME,
    "ssn": FieldType.SSN,
    "npi": FieldType.NPI,
    "cpt_code": FieldType.CPT_CODE,
    "cpt": FieldType.CPT_CODE,
    "icd10_code": FieldType.ICD10_CODE,
    "icd10": FieldType.ICD10_CODE,
    "icd-10": FieldType.ICD10_CODE,
    "list": FieldType.LIST,
    "table": FieldType.TABLE,
    "object": FieldType.OBJECT,
    "checkbox": FieldType.CHECKBOX,
    "signature": FieldType.SIGNATURE,
    "zip_code": FieldType.ZIP_CODE,
    "zip": FieldType.ZIP_CODE,
    "state": FieldType.STATE,
}


@dataclass(slots=True)
class ProposedField:
    """A single field in a schema proposal."""

    name: str
    display_name: str
    field_type: str  # string representation for serialization
    description: str = ""
    required: bool = False
    examples: list[str] = field(default_factory=list)
    location_hint: str = ""
    confidence: float = 0.5
    group: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "display_name": self.display_name,
            "field_type": self.field_type,
            "description": self.description,
            "required": self.required,
            "examples": self.examples,
            "location_hint": self.location_hint,
            "confidence": self.confidence,
            "group": self.group,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ProposedField":
        return cls(
            name=data.get("name", ""),
            display_name=data.get("display_name", data.get("name", "")),
            field_type=data.get("field_type", "string"),
            description=data.get("description", ""),
            required=data.get("required", False),
            examples=data.get("examples", []),
            location_hint=data.get("location_hint", ""),
            confidence=data.get("confidence", 0.5),
            group=data.get("group", ""),
        )


@dataclass(slots=True)
class SchemaProposal:
    """A complete schema proposal that can be refined iteratively."""

    proposal_id: str
    schema_name: str
    document_type_description: str
    fields: list[ProposedField]
    groups: list[dict[str, Any]] = field(default_factory=list)
    cross_field_rules: list[dict[str, Any]] = field(default_factory=list)
    vlm_reasoning: str = ""
    confidence: float = 0.5
    revision: int = 0
    status: str = "draft"  # draft | refined | saved

    def to_dict(self) -> dict[str, Any]:
        return {
            "proposal_id": self.proposal_id,
            "schema_name": self.schema_name,
            "document_type_description": self.document_type_description,
            "fields": [f.to_dict() for f in self.fields],
            "field_count": len(self.fields),
            "groups": self.groups,
            "cross_field_rules": self.cross_field_rules,
            "vlm_reasoning": self.vlm_reasoning,
            "confidence": self.confidence,
            "revision": self.revision,
            "status": self.status,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SchemaProposal":
        return cls(
            proposal_id=data.get("proposal_id", f"prop_{secrets.token_hex(6)}"),
            schema_name=data.get("schema_name", ""),
            document_type_description=data.get("document_type_description", ""),
            fields=[ProposedField.from_dict(f) for f in data.get("fields", [])],
            groups=data.get("groups", []),
            cross_field_rules=data.get("cross_field_rules", []),
            vlm_reasoning=data.get("vlm_reasoning", ""),
            confidence=data.get("confidence", 0.5),
            revision=data.get("revision", 0),
            status=data.get("status", "draft"),
        )


# ──────────────────────────────────────────────────────────────────
# Prompts
# ──────────────────────────────────────────────────────────────────

SCHEMA_SUGGEST_SYSTEM_PROMPT = """\
You are a document analysis expert specializing in extraction schema design.

Given a document image, propose an extraction schema that captures all meaningful \
fields. Be thorough but practical — include fields that appear in the document, \
not hypothetical ones.

Respond in JSON:
{
  "document_type_description": "Brief description of the document type",
  "schema_name": "snake_case name for this schema",
  "fields": [
    {
      "name": "field_name_snake_case",
      "display_name": "Human Readable Name",
      "field_type": "string|integer|float|date|currency|boolean|phone|email|address|name|list|table",
      "description": "What this field contains",
      "required": true,
      "examples": ["Example Value 1"],
      "location_hint": "Where this field appears on the document",
      "group": "logical_group_name"
    }
  ],
  "groups": [
    {
      "name": "group_name",
      "display_name": "Group Display Name",
      "description": "What fields in this group represent"
    }
  ],
  "cross_field_rules": [
    {
      "description": "total_charges should equal sum of line_item_charges",
      "source_field": "total_charges",
      "target_field": "line_item_charges",
      "rule_type": "sum_equals"
    }
  ],
  "reasoning": "Why you chose these fields and structure"
}

Guidelines:
- Use snake_case for field names
- Choose the most specific field_type (e.g., "currency" not "string" for money)
- Mark fields as required only if they are critical identifiers
- Group related fields logically (e.g., patient_info, billing, provider)
- Include cross-field rules where relationships exist
"""

SCHEMA_REFINE_SYSTEM_PROMPT = """\
You are a document analysis expert. You are refining an existing schema proposal \
based on user feedback.

You will receive:
1. The current schema proposal
2. User feedback describing changes

Apply the requested changes and return the COMPLETE updated schema in the same \
JSON format. Preserve fields and structure that the user did not mention.

Return the full updated schema JSON with all fields (modified + unchanged).
"""


# ──────────────────────────────────────────────────────────────────
# Agent
# ──────────────────────────────────────────────────────────────────


class SchemaProposalError(AgentError):
    """Error during schema proposal."""


class SchemaProposalAgent(BaseAgent):
    """
    Interactive schema suggestion agent.

    Three operations:
    - suggest(): Analyze a document image and propose a schema
    - refine(): Apply user feedback to modify a proposal
    - save(): Convert a proposal into a registered DocumentSchema
    """

    def __init__(self, client: LMStudioClient | None = None) -> None:
        super().__init__(name="schema_proposal", client=client)
        self._proposals: dict[str, SchemaProposal] = {}

    def process(self, state: ExtractionState) -> ExtractionState:
        """
        Pipeline entry point — suggest schema from page images.

        This is the LangGraph-compatible interface. For interactive use,
        call suggest() / refine() / save() directly.
        """
        page_images = state.get("page_images", [])
        if not page_images:
            return update_state(state, {"schema_proposal": None})

        first_page = page_images[0]
        image_data = first_page.get("data_uri", first_page.get("base64_encoded", ""))

        proposal = self.suggest(image_data)
        return update_state(state, {"schema_proposal": proposal.to_dict()})

    # ── Suggest ───────────────────────────────────────────────────

    def suggest(
        self,
        image_data: str,
        context: str = "",
    ) -> SchemaProposal:
        """
        Analyze a document image and propose an extraction schema.

        Args:
            image_data: Base64-encoded image or data URI.
            context: Optional user context (e.g., "This is a medical invoice").

        Returns:
            SchemaProposal with suggested fields.
        """
        start = time.time()

        prompt_parts = [
            "Analyze this document and propose a complete extraction schema.",
            "Identify every extractable field, its type, and where it appears.",
        ]
        if context:
            prompt_parts.append(f"\nAdditional context from user: {context}")

        settings = get_settings()
        retry_config = RetryConfig(
            max_retries=settings.extraction.max_retries,
            base_delay_ms=500,
            max_delay_ms=settings.agent.max_retry_delay_ms,
        )

        def vlm_call() -> dict[str, Any]:
            return self.send_vision_request_with_json(
                image_data=image_data,
                prompt="\n".join(prompt_parts),
                system_prompt=SCHEMA_SUGGEST_SYSTEM_PROMPT,
                temperature=0.2,
                max_tokens=4096,
            )

        try:
            result = retry_with_backoff(
                func=vlm_call,
                config=retry_config,
                on_retry=lambda attempt, e: self._logger.warning(
                    "suggest_retry", attempt=attempt + 1, error=str(e),
                ),
            )
        except Exception as e:
            raise SchemaProposalError(
                f"Schema suggestion failed: {e}",
                agent_name=self.name,
                recoverable=True,
            ) from e

        proposal = self._parse_suggestion(result)

        elapsed_ms = int((time.time() - start) * 1000)
        self._logger.info(
            "schema_suggested",
            proposal_id=proposal.proposal_id,
            field_count=len(proposal.fields),
            elapsed_ms=elapsed_ms,
        )

        # Cache for later refinement
        self._proposals[proposal.proposal_id] = proposal
        return proposal

    # ── Refine ────────────────────────────────────────────────────

    def refine(
        self,
        proposal_id: str,
        feedback: str,
        image_data: str | None = None,
    ) -> SchemaProposal:
        """
        Refine an existing schema proposal based on user feedback.

        Args:
            proposal_id: ID of proposal to refine.
            feedback: Natural language feedback describing changes.
            image_data: Optional document image for visual context.

        Returns:
            Updated SchemaProposal.

        Raises:
            SchemaProposalError: If proposal not found.
        """
        proposal = self._proposals.get(proposal_id)
        if proposal is None:
            raise SchemaProposalError(
                f"Proposal not found: {proposal_id}",
                agent_name=self.name,
                recoverable=False,
            )

        # Try VLM-assisted refinement first, then fall back to local
        if image_data:
            refined = self._refine_with_vlm(proposal, feedback, image_data)
        else:
            refined = self._refine_locally(proposal, feedback)

        refined.revision = proposal.revision + 1
        refined.status = "refined"
        refined.proposal_id = proposal_id  # keep same ID

        self._proposals[proposal_id] = refined

        self._logger.info(
            "schema_refined",
            proposal_id=proposal_id,
            revision=refined.revision,
            field_count=len(refined.fields),
        )

        return refined

    # ── Save ──────────────────────────────────────────────────────

    def save(
        self,
        proposal_id: str,
        schema_name: str | None = None,
    ) -> dict[str, Any]:
        """
        Convert a proposal into a schema definition ready for registration.

        Args:
            proposal_id: ID of the proposal to save.
            schema_name: Override the schema name.

        Returns:
            Schema definition dict compatible with SchemaBuilder / build_custom_schema.

        Raises:
            SchemaProposalError: If proposal not found.
        """
        proposal = self._proposals.get(proposal_id)
        if proposal is None:
            raise SchemaProposalError(
                f"Proposal not found: {proposal_id}",
                agent_name=self.name,
                recoverable=False,
            )

        name = schema_name or proposal.schema_name
        if not name:
            name = f"custom_{secrets.token_hex(4)}"

        schema_def = self._proposal_to_schema_def(proposal, name)

        proposal.status = "saved"
        self._proposals[proposal_id] = proposal

        self._logger.info(
            "schema_saved",
            proposal_id=proposal_id,
            schema_name=name,
            field_count=len(proposal.fields),
        )

        return schema_def

    # ── Proposal management ───────────────────────────────────────

    def get_proposal(self, proposal_id: str) -> SchemaProposal | None:
        """Get a cached proposal by ID."""
        return self._proposals.get(proposal_id)

    def list_proposals(self) -> list[dict[str, Any]]:
        """List all cached proposals (summary only)."""
        return [
            {
                "proposal_id": p.proposal_id,
                "schema_name": p.schema_name,
                "field_count": len(p.fields),
                "revision": p.revision,
                "status": p.status,
                "confidence": p.confidence,
            }
            for p in self._proposals.values()
        ]

    def delete_proposal(self, proposal_id: str) -> bool:
        """Delete a cached proposal. Returns True if found and deleted."""
        return self._proposals.pop(proposal_id, None) is not None

    # ── Internal helpers ──────────────────────────────────────────

    def _parse_suggestion(self, vlm_response: dict[str, Any]) -> SchemaProposal:
        """Parse VLM suggestion response into a SchemaProposal."""
        proposal_id = f"prop_{secrets.token_hex(6)}"

        raw_fields = vlm_response.get("fields", [])
        fields = []
        for rf in raw_fields:
            name = rf.get("name", "")
            if not name:
                continue
            # Normalize field name to snake_case
            name = self._normalize_field_name(name)
            fields.append(ProposedField(
                name=name,
                display_name=rf.get("display_name", name.replace("_", " ").title()),
                field_type=self._normalize_field_type(rf.get("field_type", "string")),
                description=rf.get("description", ""),
                required=rf.get("required", False),
                examples=rf.get("examples", []),
                location_hint=rf.get("location_hint", ""),
                confidence=rf.get("confidence", 0.7),
                group=rf.get("group", ""),
            ))

        return SchemaProposal(
            proposal_id=proposal_id,
            schema_name=vlm_response.get("schema_name", ""),
            document_type_description=vlm_response.get("document_type_description", ""),
            fields=fields,
            groups=vlm_response.get("groups", []),
            cross_field_rules=vlm_response.get("cross_field_rules", []),
            vlm_reasoning=vlm_response.get("reasoning", ""),
            confidence=vlm_response.get("confidence", 0.7),
            status="draft",
        )

    def _refine_with_vlm(
        self,
        proposal: SchemaProposal,
        feedback: str,
        image_data: str,
    ) -> SchemaProposal:
        """Use VLM to refine proposal with visual context."""
        prompt = (
            f"## Current Schema Proposal\n\n"
            f"```json\n{json.dumps(proposal.to_dict(), indent=2)}\n```\n\n"
            f"## User Feedback\n\n{feedback}\n\n"
            f"Apply the user's requested changes and return the complete "
            f"updated schema in the same JSON format."
        )

        try:
            result = self.send_vision_request_with_json(
                image_data=image_data,
                prompt=prompt,
                system_prompt=SCHEMA_REFINE_SYSTEM_PROMPT,
                temperature=0.15,
                max_tokens=4096,
            )
            return self._parse_suggestion(result)
        except Exception as e:
            self._logger.warning("vlm_refine_failed_using_local", error=str(e))
            return self._refine_locally(proposal, feedback)

    def _refine_locally(
        self,
        proposal: SchemaProposal,
        feedback: str,
    ) -> SchemaProposal:
        """
        Apply structured feedback without VLM.

        Supports simple commands embedded in feedback:
        - "remove field_name" → removes the field
        - "add field_name type description" → adds a field
        - "rename old_name new_name" → renames a field
        - "require field_name" → marks field as required
        - "optional field_name" → marks field as optional

        Falls back to returning the proposal unchanged if no commands match.
        """
        existing = {f.name: f for f in proposal.fields}
        lines = [line.strip() for line in feedback.strip().splitlines() if line.strip()]
        changed = False

        for line in lines:
            lower = line.lower()
            parts = line.split()

            if lower.startswith("remove ") and len(parts) >= 2:
                target = self._normalize_field_name(parts[1])
                if target in existing:
                    del existing[target]
                    changed = True

            elif lower.startswith("add ") and len(parts) >= 3:
                name = self._normalize_field_name(parts[1])
                ftype = self._normalize_field_type(parts[2]) if len(parts) > 2 else "string"
                desc = " ".join(parts[3:]) if len(parts) > 3 else ""
                existing[name] = ProposedField(
                    name=name,
                    display_name=name.replace("_", " ").title(),
                    field_type=ftype,
                    description=desc,
                    required=False,
                )
                changed = True

            elif lower.startswith("rename ") and len(parts) >= 3:
                old = self._normalize_field_name(parts[1])
                new = self._normalize_field_name(parts[2])
                if old in existing:
                    old_field = existing.pop(old)
                    old_field.name = new
                    old_field.display_name = new.replace("_", " ").title()
                    existing[new] = old_field
                    changed = True

            elif lower.startswith("require ") and len(parts) >= 2:
                target = self._normalize_field_name(parts[1])
                if target in existing:
                    existing[target].required = True
                    changed = True

            elif lower.startswith("optional ") and len(parts) >= 2:
                target = self._normalize_field_name(parts[1])
                if target in existing:
                    existing[target].required = False
                    changed = True

        return SchemaProposal(
            proposal_id=proposal.proposal_id,
            schema_name=proposal.schema_name,
            document_type_description=proposal.document_type_description,
            fields=list(existing.values()),
            groups=proposal.groups,
            cross_field_rules=proposal.cross_field_rules,
            vlm_reasoning=proposal.vlm_reasoning,
            confidence=proposal.confidence,
            revision=proposal.revision,
            status=proposal.status,
        )

    def _proposal_to_schema_def(
        self,
        proposal: SchemaProposal,
        name: str,
    ) -> dict[str, Any]:
        """Convert a proposal to a schema definition dict for build_custom_schema."""
        fields = []
        for pf in proposal.fields:
            field_def: dict[str, Any] = {
                "name": pf.name,
                "display_name": pf.display_name,
                "type": self._resolve_field_type(pf.field_type),
                "description": pf.description,
                "required": pf.required,
            }
            if pf.examples:
                field_def["examples"] = pf.examples
            if pf.location_hint:
                field_def["location_hint"] = pf.location_hint
            fields.append(field_def)

        rules = []
        for cr in proposal.cross_field_rules:
            rules.append({
                "source_field": cr.get("source_field", ""),
                "target_field": cr.get("target_field", ""),
                "operator": cr.get("rule_type", "equals"),
                "error_message": cr.get("description", ""),
            })

        return {
            "name": name,
            "description": proposal.document_type_description,
            "display_name": name.replace("_", " ").title(),
            "fields": fields,
            "rules": rules,
        }

    @staticmethod
    def _normalize_field_name(raw: str) -> str:
        """Normalize a raw string to a valid snake_case field name."""
        import re
        # Replace non-alphanumeric with underscore, lowercase
        name = re.sub(r"[^a-zA-Z0-9]", "_", raw.strip()).lower()
        # Collapse multiple underscores
        name = re.sub(r"_+", "_", name).strip("_")
        # Ensure starts with letter
        if name and not name[0].isalpha():
            name = "field_" + name
        return name or "unnamed_field"

    @staticmethod
    def _normalize_field_type(raw: str) -> str:
        """Normalize a raw field type string to a recognized type name."""
        normalized = raw.strip().lower()
        if normalized in FIELD_TYPE_MAP:
            return FIELD_TYPE_MAP[normalized].value
        return "string"

    @staticmethod
    def _resolve_field_type(type_str: str) -> str:
        """Resolve a field type string to its FieldType enum name for SchemaBuilder."""
        normalized = type_str.strip().lower()
        if normalized in FIELD_TYPE_MAP:
            return FIELD_TYPE_MAP[normalized].name
        # Try direct enum lookup
        for ft in FieldType:
            if ft.value == normalized:
                return ft.name
        return "STRING"
