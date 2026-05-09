"""
Shared permissive Pydantic envelopes for V3 Phase 1 schema-bound calls.

Several agents (``component_detector``, ``layout_agent``,
``table_detector``, ``schema_proposal``, ``schema_generator``) issue
structured-output VLM calls whose response shapes are normalised
downstream. Forcing each one into a strict Pydantic model would either
duplicate logic or produce schemas that drift from the tolerant
post-processing they already run.

Phase 1's goal is narrower: eliminate the entire malformed-JSON
failure class. A *permissive* envelope (a JSON object with
``additionalProperties: True``) accomplishes that — the decoder must
emit a JSON object, but the keys and value shapes are unconstrained.
Strict per-shape schemas land in Phase 2's dual-VLM EXTRACTOR /
AUDITOR work where the contract matters more than the existing
post-processing tolerance.

Use the most semantically narrow envelope that still keeps existing
post-processing happy. Today that's a single permissive class; if a
caller benefits from a stricter shape later, define a new class
alongside this one rather than tightening this one in place (which
would silently affect every consumer).
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict


class JSONObjectEnvelope(BaseModel):
    """Decoder must emit a JSON object. Keys and values are unconstrained."""

    model_config = ConfigDict(extra="allow")
