"""
V3 Phase 4 — provenance data model.

This module exists for one purpose: make ``(value, where-it-came-from)``
a single, indivisible unit. Until Phase 4, ``merged_extraction`` was a
dict of scalars and the bbox / confidence / extraction-path lived in a
parallel ``field_metadata`` dict that exporters had to cross-reference
by hand. That cross-reference is fragile — fields drop out of one map
but not the other, ordering assumptions creep in, the legacy
``bbox_overlay.py`` and the new FHIR exporter end up reading
slightly-different shapes.

This file fixes that by introducing ``FieldValue[T]``: a wrapper that
carries the value and its provenance together, all the way through
the pipeline to every exporter. ``Provenance`` is the structured
metadata; ``FieldValue`` is the wrapper.

Migration is staged. The orchestrator's reconciler closure
``dual-writes`` both ``merged_extraction`` (legacy scalar dict) and
``merged_extraction_v2`` (``FieldValue`` dict) when
``settings.provenance.enforce_field_value_wrapper=False`` (the
default). Exporters consume whichever path is populated. When the
flag flips to ``True``, only the wrapper path remains.

The migration flag is stored at ``settings.provenance.enforce_field_value_wrapper``
and surfaced via the env-var ``PROVENANCE_ENFORCE_FIELD_VALUE_WRAPPER``.
"""

from __future__ import annotations

from typing import Any, Generic, TypeVar

from pydantic import BaseModel, ConfigDict, Field

from src.pipeline.state import BoundingBoxCoords


T = TypeVar("T")


# ---------------------------------------------------------------------------
# Errors
# ---------------------------------------------------------------------------


class ProvenanceMissingError(ValueError):
    """Raised when a non-null value lacks provenance.

    The pipeline invariant: every leaf value either carries a
    populated ``Provenance`` OR is ``null``. Bare scalars without
    provenance are forbidden once ``enforce_field_value_wrapper`` is
    on. This error surfaces the violation so the offending node can
    be patched rather than silently dropping evidence.

    The check is enforced in :func:`unwrap_value` when
    ``strict=True``. Validators in the test suite assert it cannot
    happen for any registered exporter.
    """


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


class Provenance(BaseModel):
    """Where an extracted value came from.

    Captures everything a downstream consumer needs to verify, audit,
    or replay an extraction:

    * **page** — 1-based page number.
    * **bbox** — normalised rectangle on the page (``BoundingBoxCoords``)
      where the value lives. ``None`` is allowed when the VLM omitted
      coordinates AND no grounding fallback ran. Most exporters treat
      a null bbox as "provenance partial; render without overlay".
    * **source_block_id** — VLM-generated token like ``"blk_p2_047"``.
      Lets multi-pass pipelines correlate two passes' outputs even when
      the bboxes don't perfectly overlap.
    * **extraction_path** — ordered list of stages that touched this
      value: ``["pass1_vlm", "pass2_vlm", "reconciler"]``. Append-only;
      never replaced by a downstream node.
    * **agent_signatures** — agents that touched the value, in order.
      Distinct from ``extraction_path`` because some agents (e.g.
      ``validator``) don't change the extraction stage but DO touch
      the value.
    * **confidence** — calibrated confidence after every layer that
      modified it. Never the raw VLM logprob — exporters should treat
      this as the post-calibration trust score for the value.
    * **vlm_model_id** — concrete model identifier the value came
      from. e.g. ``"qwen3.6-27b-vl@8001"``. Used for audit + Phoenix
      span correlation. Single string in v1; if a Phase 5+ pipeline
      needs multi-model attribution, extend to a list.
    * **mem0_match** — FAISS memory key when memory resolved a
      reconciler tiebreak. Best-effort advisory; FAISS is approximate
      so consumers should not rely on this as a hard reference.
    """

    model_config = ConfigDict(
        # Allow extra keys so future fields don't break old serialised
        # provenance — important for the dual-write transition where
        # the reconciler may be a newer version than the exporter.
        extra="allow",
        # Make Provenance instances hashable so they can be cached.
        frozen=False,
    )

    page: int = Field(
        ge=0,
        description=(
            "1-based page number on the source PDF. ``0`` is the "
            "sentinel for legacy / migrated values where the source "
            "page is unknown."
        ),
    )
    bbox: BoundingBoxCoords | None = Field(
        default=None,
        description=(
            "Normalised bbox on the source page. ``None`` when neither "
            "extractor pass nor the grounding fallback could localise "
            "the value. Exporters render without overlay in that case."
        ),
    )
    source_block_id: str = Field(
        default="",
        description=(
            "VLM-generated block identifier (e.g. ``blk_p2_047``). Empty "
            "when the VLM didn't tag a block. ``grounded_<field>`` when "
            "a post-extraction grounding call resolved the bbox."
        ),
    )
    extraction_path: list[str] = Field(
        default_factory=list,
        description=(
            "Ordered list of pipeline stages that produced/refined this "
            "value. Append-only across nodes; never replaced. Examples: "
            "``['pass1_vlm']``, ``['pass1_vlm', 'pass2_vlm', 'reconciler']``."
        ),
    )
    agent_signatures: list[str] = Field(
        default_factory=list,
        description=(
            "Ordered list of agents that touched this value. Distinct "
            "from extraction_path: validators run without changing the "
            "extraction stage but their signature still belongs here."
        ),
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        default=0.0,
        description=(
            "Calibrated confidence in [0, 1]. Updated by the validator "
            "and the critic combiner. Exporters use this for the "
            "confidence-coloured bbox overlay (green ≥0.85, amber, red)."
        ),
    )
    vlm_model_id: str = Field(
        default="",
        description=(
            "Concrete VLM identifier that produced the value. "
            "e.g. ``qwen3.6-27b-vl@8001``."
        ),
    )
    mem0_match: str | None = Field(
        default=None,
        description=(
            "FAISS memory key when historical match resolved a "
            "reconciler tiebreak. Advisory; FAISS is approximate."
        ),
    )

    def append_stage(
        self,
        stage: str,
        *,
        agent: str | None = None,
    ) -> "Provenance":
        """Return a copy with ``stage`` appended to ``extraction_path``.

        Optionally appends ``agent`` to ``agent_signatures`` in the
        same call. Returns a new instance so callers can use it as an
        immutable update without worrying about shared mutation.
        """
        new_path = [*self.extraction_path, stage]
        new_agents = self.agent_signatures
        if agent is not None and agent not in new_agents:
            new_agents = [*new_agents, agent]
        return self.model_copy(
            update={"extraction_path": new_path, "agent_signatures": new_agents}
        )

    def to_serialisable(self) -> dict[str, Any]:
        """Return a JSON-friendly dict for export consumers.

        Identical to ``model_dump()`` but flattens ``BoundingBoxCoords``
        to its dict form so JSON exporters don't need a Pydantic
        encoder. Keys are exporter-friendly: ``_provenance`` consumers
        can drop this dict in directly.
        """
        result: dict[str, Any] = {
            "page": self.page,
            "source_block_id": self.source_block_id,
            "extraction_path": list(self.extraction_path),
            "agent_signatures": list(self.agent_signatures),
            "confidence": self.confidence,
            "vlm_model_id": self.vlm_model_id,
            "mem0_match": self.mem0_match,
        }
        if self.bbox is not None:
            try:
                result["bbox"] = self.bbox.to_dict()
            except AttributeError:
                # Defensive: if BoundingBoxCoords' shape changes, fall
                # back to model_dump.
                result["bbox"] = self.bbox.model_dump()
        else:
            result["bbox"] = None
        return result


# Sentinel provenance for migrated / legacy values. Used when an old
# checkpoint or a non-V3 code path writes a value without provenance —
# the dual-write shim wraps it in a FieldValue with this sentinel so
# downstream consumers never see a bare scalar.
LEGACY_SENTINEL_PROVENANCE = Provenance(
    page=0,
    bbox=None,
    source_block_id="legacy",
    extraction_path=["legacy"],
    agent_signatures=[],
    confidence=0.0,
    vlm_model_id="",
    mem0_match=None,
)


def empty_provenance(*, stage: str = "extraction_failed") -> Provenance:
    """Sentinel provenance for null fields where extraction failed.

    Produced when an agent couldn't localise a value AND the value is
    ``null``. The invariant holds (``null`` requires no real
    provenance) but exporters still need a placeholder so the
    serialised shape is stable.
    """
    return Provenance(
        page=0,
        bbox=None,
        source_block_id="",
        extraction_path=[stage],
        agent_signatures=[],
        confidence=0.0,
        vlm_model_id="",
        mem0_match=None,
    )


# ---------------------------------------------------------------------------
# FieldValue wrapper
# ---------------------------------------------------------------------------


class FieldValue(BaseModel, Generic[T]):
    """``(value, provenance)`` tuple — the canonical leaf type.

    Generic over the value type so a CMS-1500 NPI is ``FieldValue[str]``
    and a UB-04 line-item charge is ``FieldValue[float]``. The wrapper
    is intentionally thin: no validation, no transformation, no
    coercion — just a container that keeps the value and its
    provenance together through every pipeline node and exporter.

    Why ``BaseModel`` (not a dataclass): exporters want JSON
    round-tripping, and the FHIR exporter wants to recursively flatten
    nested wrappers. Both are easier with Pydantic's
    ``model_dump_json``.

    Why ``extra="allow"``: see ``Provenance``. Forward-compat during
    migration.
    """

    model_config = ConfigDict(extra="allow")

    value: T | None = Field(
        description="The extracted value, or ``None`` when not localisable.",
    )
    provenance: Provenance = Field(
        description="Where the value came from (page, bbox, lineage).",
    )

    def to_serialisable(self) -> dict[str, Any]:
        """Render as ``{"value": ..., "_provenance": {...}}``.

        Used by the JSON exporter directly. The ``_provenance`` key
        starts with an underscore so consumers that expected a bare
        scalar can keep working by dropping the underscore-prefixed
        key — see ``unwrap_value`` for the back-compat path.
        """
        return {
            "value": self.value,
            "_provenance": self.provenance.to_serialisable(),
        }


class PHIFieldValue(FieldValue[T]):
    """``FieldValue`` whose ``value`` is PHI and must round-trip redacted.

    Adds:

    * **encrypted_value** — AES-256-GCM ciphertext of the original
      value. Held server-side; downstream consumers without a KMS key
      see only ``[REDACTED]`` in the ``value`` field.
    * **redacted_value** — the literal ``"[REDACTED]"`` string;
      semantically redundant with ``value`` in non-privileged
      contexts but kept explicit so exporters don't have to guess.

    Provenance survives PHI redaction unchanged: page/bbox/confidence
    are metadata about the *extraction act*, not about the patient,
    and removing them would destroy the audit trail.

    The encryption + decryption flow is owned by ``src/security/encryption.py``;
    this wrapper just defines the shape.
    """

    encrypted_value: bytes | None = Field(
        default=None,
        description=(
            "AES-256-GCM ciphertext of the original value. Server-side "
            "only; serialised to base64 in audit logs and elided from "
            "user-facing exports."
        ),
    )
    redacted_value: str = Field(
        default="[REDACTED]",
        description="Literal redaction marker; semantically equal to ``value`` in non-privileged contexts.",
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def is_field_value(obj: Any) -> bool:
    """True when ``obj`` is a ``FieldValue`` instance OR a dict with the wrapper shape.

    Exporters often work on serialised state (after a LangGraph
    checkpoint round-trip) where a ``FieldValue`` becomes a dict
    ``{"value": ..., "_provenance": {...}}`` or
    ``{"value": ..., "provenance": {...}}``. Both forms count.
    """
    if isinstance(obj, FieldValue):
        return True
    if isinstance(obj, dict):
        return "value" in obj and ("_provenance" in obj or "provenance" in obj)
    return False


def unwrap_value(obj: Any, *, strict: bool = False) -> Any:
    """Return the bare value from a ``FieldValue`` (or pass-through).

    Recognised shapes (in order):

    1. A :class:`FieldValue` instance → returns ``obj.value``.
    2. A serialised wrapper dict with both ``"value"`` and a
       ``_provenance`` / ``provenance`` key → returns ``obj["value"]``.
    3. A legacy envelope dict with a ``"value"`` key but no provenance
       (e.g. WS-8's ``{"value": "x", "confidence": 0.91}``) → returns
       ``obj["value"]``. This catches the long-established convention
       in this codebase where any dict-with-``value`` is an envelope.
    4. Anything else → returned unchanged (bare-scalar / legacy path).

    Args:
        obj: a ``FieldValue``, a wrapper dict, an envelope dict, or a
            bare scalar.
        strict: when ``True``, raises ``ProvenanceMissingError`` if a
            non-null bare scalar arrives (cases 1 and 2 are valid;
            cases 3 and 4 raise). Used by exporters that require
            provenance once the migration flag flips.

    Returns:
        The bare value.
    """
    if isinstance(obj, FieldValue):
        return obj.value
    if isinstance(obj, dict) and "value" in obj:
        has_provenance = "_provenance" in obj or "provenance" in obj
        if has_provenance:
            return obj["value"]
        # Legacy envelope shape — unwrap but, in strict mode, signal
        # the missing provenance.
        if strict:
            raise ProvenanceMissingError(
                f"strict=True but envelope lacks provenance: {sorted(obj.keys())}"
            )
        return obj["value"]
    # Bare scalar / legacy path.
    if strict and obj is not None:
        raise ProvenanceMissingError(
            f"strict=True but received bare scalar without provenance: {obj!r}"
        )
    return obj


def unwrap_provenance(obj: Any) -> Provenance | None:
    """Return the ``Provenance`` for a ``FieldValue`` (or ``None``).

    Mirrors :func:`unwrap_value`. Returns ``None`` for legacy bare
    scalars and for dicts that lack a provenance key — the caller
    decides whether absence is an error (exporters in strict mode
    raise; tolerant exporters render without overlay).
    """
    if isinstance(obj, FieldValue):
        return obj.provenance
    if isinstance(obj, dict):
        prov = obj.get("_provenance") or obj.get("provenance")
        if isinstance(prov, dict):
            try:
                return Provenance.model_validate(prov)
            except Exception:
                return None
        if isinstance(prov, Provenance):
            return prov
    return None


def wrap_value(
    value: Any,
    *,
    provenance: Provenance | None = None,
) -> FieldValue:
    """Wrap a bare value into a ``FieldValue``.

    Used by the reconciler's dual-write closure to build the
    ``merged_extraction_v2`` shape from the same raw fields it writes
    to ``merged_extraction``. When ``provenance`` is omitted we use
    :data:`LEGACY_SENTINEL_PROVENANCE` so the wrapper is always
    well-formed, even for migrated values.
    """
    return FieldValue(
        value=value,
        provenance=provenance if provenance is not None else LEGACY_SENTINEL_PROVENANCE,
    )


__all__ = [
    "FieldValue",
    "LEGACY_SENTINEL_PROVENANCE",
    "PHIFieldValue",
    "Provenance",
    "ProvenanceMissingError",
    "empty_provenance",
    "is_field_value",
    "unwrap_provenance",
    "unwrap_value",
    "wrap_value",
]
