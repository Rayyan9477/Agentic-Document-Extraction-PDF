"""
Unit tests for Phase 2C: Schema Proposal Agent (Schema Wizard).

Tests SchemaProposalAgent suggest/refine/save flow,
ProposedField/SchemaProposal data structures, and local refinement logic.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.agents.schema_proposal import (
    FIELD_TYPE_MAP,
    SCHEMA_SUGGEST_SYSTEM_PROMPT,
    ProposedField,
    SchemaProposal,
    SchemaProposalAgent,
    SchemaProposalError,
)
from src.pipeline.state import create_initial_state, update_state


# ──────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────

MOCK_VLM_SUGGESTION = {
    "document_type_description": "Medical billing invoice",
    "schema_name": "medical_invoice",
    "fields": [
        {
            "name": "patient_name",
            "display_name": "Patient Name",
            "field_type": "name",
            "description": "Full name of patient",
            "required": True,
            "examples": ["John Smith"],
            "location_hint": "Top left",
            "confidence": 0.9,
            "group": "patient_info",
        },
        {
            "name": "date_of_service",
            "display_name": "Date of Service",
            "field_type": "date",
            "description": "Service date",
            "required": True,
            "examples": ["01/15/2024"],
            "location_hint": "Header",
            "confidence": 0.85,
            "group": "billing",
        },
        {
            "name": "total_charges",
            "display_name": "Total Charges",
            "field_type": "currency",
            "description": "Total billed amount",
            "required": False,
            "examples": ["$150.00"],
            "location_hint": "Bottom right",
            "confidence": 0.8,
            "group": "billing",
        },
    ],
    "groups": [
        {"name": "patient_info", "display_name": "Patient Information"},
        {"name": "billing", "display_name": "Billing Details"},
    ],
    "cross_field_rules": [],
    "reasoning": "Medical invoice with patient info and billing sections",
    "confidence": 0.85,
}


def _make_agent() -> SchemaProposalAgent:
    """Create agent with mocked VLM client."""
    mock_client = MagicMock()
    return SchemaProposalAgent(client=mock_client)


def _mock_vlm_response(agent: SchemaProposalAgent, response: dict[str, Any]) -> None:
    """Set up the agent's VLM to return a specific response."""
    mock_vision_response = MagicMock()
    mock_vision_response.has_json = True
    mock_vision_response.parsed_json = response
    mock_vision_response.content = "json response"
    mock_vision_response.latency_ms = 100
    agent._client.send_vision_request.return_value = mock_vision_response


# ──────────────────────────────────────────────────────────────────
# ProposedField Tests
# ──────────────────────────────────────────────────────────────────


class TestProposedField:
    def test_to_dict_roundtrip(self):
        field = ProposedField(
            name="patient_name",
            display_name="Patient Name",
            field_type="name",
            description="Full name",
            required=True,
            examples=["John"],
            location_hint="Top",
            confidence=0.9,
            group="info",
        )
        d = field.to_dict()
        restored = ProposedField.from_dict(d)
        assert restored.name == "patient_name"
        assert restored.display_name == "Patient Name"
        assert restored.field_type == "name"
        assert restored.required is True
        assert restored.examples == ["John"]
        assert restored.confidence == 0.9

    def test_from_dict_defaults(self):
        field = ProposedField.from_dict({"name": "test"})
        assert field.display_name == "test"
        assert field.field_type == "string"
        assert field.required is False
        assert field.confidence == 0.5

    def test_to_dict_all_fields(self):
        field = ProposedField(
            name="x", display_name="X", field_type="string",
        )
        d = field.to_dict()
        assert "name" in d
        assert "display_name" in d
        assert "field_type" in d
        assert "required" in d
        assert "examples" in d
        assert "location_hint" in d
        assert "confidence" in d
        assert "group" in d


# ──────────────────────────────────────────────────────────────────
# SchemaProposal Tests
# ──────────────────────────────────────────────────────────────────


class TestSchemaProposal:
    def test_to_dict_roundtrip(self):
        proposal = SchemaProposal(
            proposal_id="prop_abc123",
            schema_name="test_schema",
            document_type_description="Test doc",
            fields=[
                ProposedField(name="f1", display_name="F1", field_type="string"),
                ProposedField(name="f2", display_name="F2", field_type="date"),
            ],
            confidence=0.8,
            revision=1,
            status="refined",
        )
        d = proposal.to_dict()
        assert d["proposal_id"] == "prop_abc123"
        assert d["field_count"] == 2
        assert d["status"] == "refined"
        assert len(d["fields"]) == 2

        restored = SchemaProposal.from_dict(d)
        assert restored.schema_name == "test_schema"
        assert len(restored.fields) == 2
        assert restored.revision == 1

    def test_field_count_in_to_dict(self):
        proposal = SchemaProposal(
            proposal_id="p1",
            schema_name="s",
            document_type_description="d",
            fields=[ProposedField(name="a", display_name="A", field_type="string")],
        )
        assert proposal.to_dict()["field_count"] == 1

    def test_empty_proposal(self):
        proposal = SchemaProposal(
            proposal_id="p2",
            schema_name="",
            document_type_description="",
            fields=[],
        )
        d = proposal.to_dict()
        assert d["field_count"] == 0
        assert d["status"] == "draft"

    def test_from_dict_generates_id(self):
        proposal = SchemaProposal.from_dict({"schema_name": "test"})
        assert proposal.proposal_id.startswith("prop_")


# ──────────────────────────────────────────────────────────────────
# Field Type Normalization Tests
# ──────────────────────────────────────────────────────────────────


class TestFieldTypeNormalization:
    def test_common_type_aliases(self):
        agent = _make_agent()
        assert agent._normalize_field_type("text") == "string"
        assert agent._normalize_field_type("money") == "currency"
        assert agent._normalize_field_type("bool") == "boolean"
        assert agent._normalize_field_type("int") == "integer"
        assert agent._normalize_field_type("number") == "float"
        assert agent._normalize_field_type("percent") == "percentage"

    def test_direct_type_names(self):
        agent = _make_agent()
        assert agent._normalize_field_type("string") == "string"
        assert agent._normalize_field_type("date") == "date"
        assert agent._normalize_field_type("currency") == "currency"
        assert agent._normalize_field_type("phone") == "phone"

    def test_unknown_type_defaults_to_string(self):
        agent = _make_agent()
        assert agent._normalize_field_type("unknown_type") == "string"
        assert agent._normalize_field_type("") == "string"

    def test_case_insensitive(self):
        agent = _make_agent()
        assert agent._normalize_field_type("Date") == "date"
        assert agent._normalize_field_type("CURRENCY") == "currency"
        assert agent._normalize_field_type("Boolean") == "boolean"

    def test_medical_code_types(self):
        agent = _make_agent()
        assert agent._normalize_field_type("cpt") == "cpt_code"
        assert agent._normalize_field_type("icd10") == "icd10_code"
        assert agent._normalize_field_type("icd-10") == "icd10_code"
        assert agent._normalize_field_type("npi") == "npi"

    def test_field_type_map_completeness(self):
        """All mapped types resolve to valid FieldType values."""
        from src.schemas.field_types import FieldType
        valid_values = {ft.value for ft in FieldType}
        for alias, ft in FIELD_TYPE_MAP.items():
            assert ft.value in valid_values, f"Alias '{alias}' maps to invalid type {ft}"


# ──────────────────────────────────────────────────────────────────
# Field Name Normalization Tests
# ──────────────────────────────────────────────────────────────────


class TestFieldNameNormalization:
    def test_basic_snake_case(self):
        assert SchemaProposalAgent._normalize_field_name("Patient Name") == "patient_name"
        assert SchemaProposalAgent._normalize_field_name("Date of Birth") == "date_of_birth"

    def test_special_characters(self):
        assert SchemaProposalAgent._normalize_field_name("field-name") == "field_name"
        assert SchemaProposalAgent._normalize_field_name("field.name") == "field_name"
        assert SchemaProposalAgent._normalize_field_name("field/name") == "field_name"

    def test_numeric_prefix_gets_prefix(self):
        result = SchemaProposalAgent._normalize_field_name("123field")
        assert result.startswith("field_")

    def test_empty_string(self):
        assert SchemaProposalAgent._normalize_field_name("") == "unnamed_field"

    def test_collapses_multiple_underscores(self):
        result = SchemaProposalAgent._normalize_field_name("a___b")
        assert result == "a_b"

    def test_strips_leading_trailing_underscores(self):
        result = SchemaProposalAgent._normalize_field_name("_test_")
        assert result == "test"

    def test_already_valid(self):
        assert SchemaProposalAgent._normalize_field_name("valid_name") == "valid_name"


# ──────────────────────────────────────────────────────────────────
# Suggest Tests
# ──────────────────────────────────────────────────────────────────


class TestSuggest:
    def test_suggest_returns_proposal(self):
        agent = _make_agent()
        _mock_vlm_response(agent, MOCK_VLM_SUGGESTION)

        proposal = agent.suggest("data:image/png;base64,abc123")

        assert isinstance(proposal, SchemaProposal)
        assert proposal.proposal_id.startswith("prop_")
        assert proposal.schema_name == "medical_invoice"
        assert len(proposal.fields) == 3
        assert proposal.status == "draft"

    def test_suggest_with_context(self):
        agent = _make_agent()
        _mock_vlm_response(agent, MOCK_VLM_SUGGESTION)

        proposal = agent.suggest("base64data", context="This is a medical invoice")

        # Verify VLM was called
        assert agent._client.send_vision_request.called
        assert proposal.schema_name == "medical_invoice"

    def test_suggest_caches_proposal(self):
        agent = _make_agent()
        _mock_vlm_response(agent, MOCK_VLM_SUGGESTION)

        proposal = agent.suggest("base64data")

        cached = agent.get_proposal(proposal.proposal_id)
        assert cached is not None
        assert cached.proposal_id == proposal.proposal_id

    def test_suggest_normalizes_field_names(self):
        agent = _make_agent()
        response = {
            "schema_name": "test",
            "document_type_description": "test",
            "fields": [
                {"name": "Patient Name", "field_type": "string"},
                {"name": "Date-of-Service", "field_type": "date"},
            ],
        }
        _mock_vlm_response(agent, response)

        proposal = agent.suggest("base64data")
        names = [f.name for f in proposal.fields]
        assert "patient_name" in names
        assert "date_of_service" in names

    def test_suggest_normalizes_field_types(self):
        agent = _make_agent()
        response = {
            "schema_name": "test",
            "document_type_description": "test",
            "fields": [
                {"name": "amount", "field_type": "money"},
                {"name": "active", "field_type": "bool"},
            ],
        }
        _mock_vlm_response(agent, response)

        proposal = agent.suggest("base64data")
        types = {f.name: f.field_type for f in proposal.fields}
        assert types["amount"] == "currency"
        assert types["active"] == "boolean"

    def test_suggest_skips_empty_field_names(self):
        agent = _make_agent()
        response = {
            "schema_name": "test",
            "document_type_description": "test",
            "fields": [
                {"name": "", "field_type": "string"},
                {"name": "valid", "field_type": "string"},
            ],
        }
        _mock_vlm_response(agent, response)

        proposal = agent.suggest("base64data")
        assert len(proposal.fields) == 1
        assert proposal.fields[0].name == "valid"

    def test_suggest_vlm_failure_raises_error(self):
        agent = _make_agent()
        agent._client.send_vision_request.side_effect = Exception("VLM unavailable")

        with pytest.raises(SchemaProposalError, match="Schema suggestion failed"):
            agent.suggest("base64data")

    def test_suggest_tracks_vlm_calls(self):
        agent = _make_agent()
        _mock_vlm_response(agent, MOCK_VLM_SUGGESTION)

        agent.suggest("base64data")
        assert agent.vlm_calls >= 1


# ──────────────────────────────────────────────────────────────────
# Refine Tests (Local)
# ──────────────────────────────────────────────────────────────────


class TestRefineLocal:
    def _setup_proposal(self, agent: SchemaProposalAgent) -> str:
        """Create a proposal and return its ID."""
        _mock_vlm_response(agent, MOCK_VLM_SUGGESTION)
        proposal = agent.suggest("base64data")
        return proposal.proposal_id

    def test_refine_remove_field(self):
        agent = _make_agent()
        pid = self._setup_proposal(agent)

        refined = agent.refine(pid, "remove total_charges")

        names = [f.name for f in refined.fields]
        assert "total_charges" not in names
        assert "patient_name" in names
        assert refined.revision == 1
        assert refined.status == "refined"

    def test_refine_add_field(self):
        agent = _make_agent()
        pid = self._setup_proposal(agent)

        refined = agent.refine(pid, "add insurance_id string Insurance ID number")

        names = [f.name for f in refined.fields]
        assert "insurance_id" in names
        new_field = next(f for f in refined.fields if f.name == "insurance_id")
        assert new_field.field_type == "string"

    def test_refine_rename_field(self):
        agent = _make_agent()
        pid = self._setup_proposal(agent)

        refined = agent.refine(pid, "rename patient_name patient_full_name")

        names = [f.name for f in refined.fields]
        assert "patient_full_name" in names
        assert "patient_name" not in names

    def test_refine_require_field(self):
        agent = _make_agent()
        pid = self._setup_proposal(agent)

        # total_charges is not required in mock
        original = agent.get_proposal(pid)
        charges_field = next(f for f in original.fields if f.name == "total_charges")
        assert charges_field.required is False

        refined = agent.refine(pid, "require total_charges")
        charges_field = next(f for f in refined.fields if f.name == "total_charges")
        assert charges_field.required is True

    def test_refine_optional_field(self):
        agent = _make_agent()
        pid = self._setup_proposal(agent)

        # patient_name is required in mock
        refined = agent.refine(pid, "optional patient_name")
        pn_field = next(f for f in refined.fields if f.name == "patient_name")
        assert pn_field.required is False

    def test_refine_multiple_commands(self):
        agent = _make_agent()
        pid = self._setup_proposal(agent)

        feedback = "remove total_charges\nadd copay currency Copay amount"
        refined = agent.refine(pid, feedback)

        names = [f.name for f in refined.fields]
        assert "total_charges" not in names
        assert "copay" in names

    def test_refine_unknown_command_preserves(self):
        agent = _make_agent()
        pid = self._setup_proposal(agent)
        original_count = len(agent.get_proposal(pid).fields)

        refined = agent.refine(pid, "Please make the schema better")

        # No structured commands → nothing changes
        assert len(refined.fields) == original_count

    def test_refine_increments_revision(self):
        agent = _make_agent()
        pid = self._setup_proposal(agent)

        r1 = agent.refine(pid, "remove total_charges")
        assert r1.revision == 1

        r2 = agent.refine(pid, "add new_field string A new field")
        assert r2.revision == 2

    def test_refine_nonexistent_proposal_raises(self):
        agent = _make_agent()

        with pytest.raises(SchemaProposalError, match="not found"):
            agent.refine("nonexistent_id", "remove something")


# ──────────────────────────────────────────────────────────────────
# Refine Tests (VLM-assisted)
# ──────────────────────────────────────────────────────────────────


class TestRefineWithVLM:
    def test_refine_with_image_uses_vlm(self):
        agent = _make_agent()
        _mock_vlm_response(agent, MOCK_VLM_SUGGESTION)
        pid = agent.suggest("base64data").proposal_id

        # Second VLM call for refinement
        refined_response = dict(MOCK_VLM_SUGGESTION)
        refined_response["fields"] = [MOCK_VLM_SUGGESTION["fields"][0]]
        _mock_vlm_response(agent, refined_response)

        refined = agent.refine(pid, "Only keep patient name", image_data="base64data")
        assert len(refined.fields) == 1

    def test_refine_with_image_falls_back_on_vlm_failure(self):
        agent = _make_agent()
        _mock_vlm_response(agent, MOCK_VLM_SUGGESTION)
        pid = agent.suggest("base64data").proposal_id

        # Make VLM fail for refinement
        agent._client.send_vision_request.side_effect = Exception("VLM down")

        # Should fall back to local refinement
        refined = agent.refine(pid, "remove total_charges", image_data="base64data")
        names = [f.name for f in refined.fields]
        assert "total_charges" not in names


# ──────────────────────────────────────────────────────────────────
# Save Tests
# ──────────────────────────────────────────────────────────────────


class TestSave:
    def test_save_returns_schema_def(self):
        agent = _make_agent()
        _mock_vlm_response(agent, MOCK_VLM_SUGGESTION)
        pid = agent.suggest("base64data").proposal_id

        schema_def = agent.save(pid)

        assert schema_def["name"] == "medical_invoice"
        assert "fields" in schema_def
        assert len(schema_def["fields"]) == 3
        assert "description" in schema_def

    def test_save_with_name_override(self):
        agent = _make_agent()
        _mock_vlm_response(agent, MOCK_VLM_SUGGESTION)
        pid = agent.suggest("base64data").proposal_id

        schema_def = agent.save(pid, schema_name="my_custom_schema")

        assert schema_def["name"] == "my_custom_schema"

    def test_save_generates_name_if_empty(self):
        agent = _make_agent()
        response = dict(MOCK_VLM_SUGGESTION)
        response["schema_name"] = ""
        _mock_vlm_response(agent, response)
        pid = agent.suggest("base64data").proposal_id

        schema_def = agent.save(pid)

        assert schema_def["name"].startswith("custom_")

    def test_save_marks_status(self):
        agent = _make_agent()
        _mock_vlm_response(agent, MOCK_VLM_SUGGESTION)
        pid = agent.suggest("base64data").proposal_id

        agent.save(pid)
        proposal = agent.get_proposal(pid)
        assert proposal.status == "saved"

    def test_save_nonexistent_raises(self):
        agent = _make_agent()

        with pytest.raises(SchemaProposalError, match="not found"):
            agent.save("nonexistent_id")

    def test_save_field_types_are_enum_names(self):
        """Saved schema uses FieldType enum names for build_custom_schema compatibility."""
        agent = _make_agent()
        _mock_vlm_response(agent, MOCK_VLM_SUGGESTION)
        pid = agent.suggest("base64data").proposal_id

        schema_def = agent.save(pid)
        for field in schema_def["fields"]:
            # Type should be a valid FieldType enum name (e.g., "NAME", "DATE", "CURRENCY")
            assert field["type"].isupper() or field["type"] == field["type"].upper()

    def test_save_includes_cross_field_rules(self):
        agent = _make_agent()
        response = dict(MOCK_VLM_SUGGESTION)
        response["cross_field_rules"] = [
            {
                "description": "Total equals sum of items",
                "source_field": "total_charges",
                "target_field": "line_items",
                "rule_type": "sum_equals",
            }
        ]
        _mock_vlm_response(agent, response)
        pid = agent.suggest("base64data").proposal_id

        schema_def = agent.save(pid)
        assert len(schema_def["rules"]) == 1
        assert schema_def["rules"][0]["source_field"] == "total_charges"


# ──────────────────────────────────────────────────────────────────
# Proposal Management Tests
# ──────────────────────────────────────────────────────────────────


class TestProposalManagement:
    def test_list_proposals_empty(self):
        agent = _make_agent()
        assert agent.list_proposals() == []

    def test_list_proposals_after_suggest(self):
        agent = _make_agent()
        _mock_vlm_response(agent, MOCK_VLM_SUGGESTION)
        agent.suggest("base64data")

        proposals = agent.list_proposals()
        assert len(proposals) == 1
        assert "proposal_id" in proposals[0]
        assert proposals[0]["field_count"] == 3
        assert proposals[0]["status"] == "draft"

    def test_get_proposal_nonexistent(self):
        agent = _make_agent()
        assert agent.get_proposal("nonexistent") is None

    def test_delete_proposal(self):
        agent = _make_agent()
        _mock_vlm_response(agent, MOCK_VLM_SUGGESTION)
        pid = agent.suggest("base64data").proposal_id

        assert agent.delete_proposal(pid) is True
        assert agent.get_proposal(pid) is None
        assert len(agent.list_proposals()) == 0

    def test_delete_nonexistent(self):
        agent = _make_agent()
        assert agent.delete_proposal("nonexistent") is False


# ──────────────────────────────────────────────────────────────────
# Process (LangGraph Interface) Tests
# ──────────────────────────────────────────────────────────────────


class TestProcess:
    def test_process_with_page_images(self):
        agent = _make_agent()
        _mock_vlm_response(agent, MOCK_VLM_SUGGESTION)

        state = create_initial_state(
            pdf_path="/test.pdf",
            page_images=[{"page_number": 1, "data_uri": "data:image/png;base64,abc"}],
        )

        result = agent.process(state)

        assert result.get("schema_proposal") is not None
        proposal = result["schema_proposal"]
        assert "proposal_id" in proposal
        assert proposal["field_count"] == 3

    def test_process_empty_pages_returns_none(self):
        agent = _make_agent()

        state = create_initial_state(pdf_path="/test.pdf", page_images=[])
        result = agent.process(state)

        assert result.get("schema_proposal") is None

    def test_process_preserves_other_state(self):
        agent = _make_agent()
        _mock_vlm_response(agent, MOCK_VLM_SUGGESTION)

        state = create_initial_state(
            pdf_path="/test.pdf",
            page_images=[{"page_number": 1, "data_uri": "data:image/png;base64,abc"}],
        )

        result = agent.process(state)
        assert result["pdf_path"] == "/test.pdf"
        assert result.get("schema_proposal") is not None


# ──────────────────────────────────────────────────────────────────
# State Field Tests
# ──────────────────────────────────────────────────────────────────


class TestStateFields:
    def test_initial_state_has_schema_proposal(self):
        state = create_initial_state(pdf_path="/test.pdf")
        assert "schema_proposal" in state
        assert state["schema_proposal"] is None

    def test_update_state_preserves_schema_proposal(self):
        state = create_initial_state(pdf_path="/test.pdf")
        updated = update_state(state, {"schema_proposal": {"proposal_id": "p1"}})
        assert updated["schema_proposal"]["proposal_id"] == "p1"


# ──────────────────────────────────────────────────────────────────
# System Prompt Tests
# ──────────────────────────────────────────────────────────────────


class TestPrompts:
    def test_suggest_prompt_contains_json_format(self):
        assert "JSON" in SCHEMA_SUGGEST_SYSTEM_PROMPT or "json" in SCHEMA_SUGGEST_SYSTEM_PROMPT.lower()

    def test_suggest_prompt_mentions_field_types(self):
        prompt = SCHEMA_SUGGEST_SYSTEM_PROMPT.lower()
        assert "string" in prompt
        assert "date" in prompt
        assert "currency" in prompt

    def test_suggest_prompt_mentions_snake_case(self):
        assert "snake_case" in SCHEMA_SUGGEST_SYSTEM_PROMPT


# ──────────────────────────────────────────────────────────────────
# Edge Cases
# ──────────────────────────────────────────────────────────────────


class TestEdgeCases:
    def test_suggest_empty_fields_from_vlm(self):
        agent = _make_agent()
        _mock_vlm_response(agent, {
            "schema_name": "empty",
            "document_type_description": "empty doc",
            "fields": [],
        })

        proposal = agent.suggest("base64data")
        assert len(proposal.fields) == 0

    def test_suggest_missing_optional_response_keys(self):
        agent = _make_agent()
        _mock_vlm_response(agent, {
            "fields": [
                {"name": "test_field", "field_type": "string"},
            ],
        })

        proposal = agent.suggest("base64data")
        assert proposal.schema_name == ""
        assert len(proposal.fields) == 1

    def test_refine_with_nonexistent_field_in_command(self):
        agent = _make_agent()
        _mock_vlm_response(agent, MOCK_VLM_SUGGESTION)
        pid = agent.suggest("base64data").proposal_id

        # Removing a field that doesn't exist should not raise
        refined = agent.refine(pid, "remove nonexistent_field")
        assert len(refined.fields) == 3  # unchanged

    def test_multiple_suggest_creates_separate_proposals(self):
        agent = _make_agent()
        _mock_vlm_response(agent, MOCK_VLM_SUGGESTION)

        p1 = agent.suggest("base64data")
        p2 = agent.suggest("base64data")

        assert p1.proposal_id != p2.proposal_id
        assert len(agent.list_proposals()) == 2

    def test_resolve_field_type_for_save(self):
        # Verify _resolve_field_type returns enum names
        assert SchemaProposalAgent._resolve_field_type("string") == "STRING"
        assert SchemaProposalAgent._resolve_field_type("currency") == "CURRENCY"
        assert SchemaProposalAgent._resolve_field_type("date") == "DATE"
        assert SchemaProposalAgent._resolve_field_type("name") == "NAME"
        assert SchemaProposalAgent._resolve_field_type("unknown") == "STRING"
