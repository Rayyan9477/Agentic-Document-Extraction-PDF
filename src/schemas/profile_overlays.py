"""
V3 Phase 5 — Profile-scoped schema overlays.

When a profile names a ``schema_overlay_fields`` entry (e.g.
``("healthcare_core",)`` for medical-RCM), the analyzer adds those
field blocks to the resolved document schema *only* when that profile
is active.

This is the rehoming of the ``HEALTHCARE_FIELDS`` block that used to
live in ``generic_fallback.py`` and would invent
``patient_name``-shaped phantom fields on a non-medical doc. The
overlay only fires when ``profile == "medical-rcm"``, so a generic
invoice will never see these fields proposed.

Public surface:

* ``OVERLAY_BUNDLES`` — name → list[FieldDefinition] map.
* ``apply_overlay(schema, profile)`` — return a new ``DocumentSchema``
  with overlay fields appended (de-duplicated by ``name``). Original
  schema is not mutated.
"""

from __future__ import annotations

from src.schemas.base import DocumentSchema
from src.schemas.field_types import FieldDefinition, FieldType
from src.profiles.descriptor import ProfileDescriptor


# ---------------------------------------------------------------------------
# Overlay bundles
# ---------------------------------------------------------------------------


# Healthcare core fields. Previously embedded in ``generic_fallback``;
# now scoped to medical-RCM profile only.
HEALTHCARE_CORE_FIELDS: list[FieldDefinition] = [
    FieldDefinition(
        name="patient_name",
        display_name="Patient Name",
        field_type=FieldType.NAME,
        description="Patient's full name as printed on the document.",
        required=False,
        location_hint="Patient information section",
    ),
    FieldDefinition(
        name="provider_name",
        display_name="Provider Name",
        field_type=FieldType.NAME,
        description="Healthcare provider name (rendering or billing).",
        required=False,
        location_hint="Provider section or signature area",
    ),
    FieldDefinition(
        name="service_date",
        display_name="Date of Service",
        field_type=FieldType.DATE,
        description="Date services were provided.",
        required=False,
        location_hint="Service details section",
    ),
    FieldDefinition(
        name="diagnosis_codes",
        display_name="Diagnosis Codes",
        field_type=FieldType.STRING,  # comma-separated list
        description="ICD-10-CM diagnosis codes (comma-separated).",
        required=False,
        location_hint="Diagnosis section",
    ),
    FieldDefinition(
        name="procedure_codes",
        display_name="Procedure Codes",
        field_type=FieldType.STRING,
        description="CPT/HCPCS procedure codes (comma-separated).",
        required=False,
        location_hint="Services or procedures section",
    ),
]


# Map of overlay-bundle-name → field list. Profile descriptors name
# the bundles they want; this module owns the actual ``FieldDefinition``
# instances.
OVERLAY_BUNDLES: dict[str, list[FieldDefinition]] = {
    "healthcare_core": HEALTHCARE_CORE_FIELDS,
}


# ---------------------------------------------------------------------------
# Public function
# ---------------------------------------------------------------------------


def apply_overlay(
    schema: DocumentSchema,
    profile: ProfileDescriptor,
) -> DocumentSchema:
    """
    Return a new ``DocumentSchema`` with the profile's overlay applied.

    Overlay fields are *appended* — fields that already exist in the
    base schema (matched by ``name``) are NOT duplicated. The base
    schema is the source of truth for field semantics; the overlay
    only fills gaps.

    The original ``schema`` is not mutated.
    """
    if not profile.schema_overlay_fields:
        return schema

    existing_names = {f.name for f in schema.fields}
    additions: list[FieldDefinition] = []
    for bundle_name in profile.schema_overlay_fields:
        bundle = OVERLAY_BUNDLES.get(bundle_name)
        if bundle is None:
            # Unknown overlay name — log and skip rather than fail.
            # New profiles can ship before their overlay bundles, and
            # we don't want a half-built profile to crash extraction.
            continue
        for f in bundle:
            if f.name not in existing_names:
                additions.append(f)
                existing_names.add(f.name)

    if not additions:
        return schema

    return DocumentSchema(
        name=schema.name,
        display_name=schema.display_name,
        document_type=schema.document_type,
        description=schema.description,
        fields=list(schema.fields) + additions,
        cross_field_rules=schema.cross_field_rules,
        required_sections=schema.required_sections,
        version=schema.version,
        classification_hints=schema.classification_hints,
    )
