"""
Profile descriptor data model.

A ``ProfileDescriptor`` is a frozen value object that fully describes a
document profile to the rest of the pipeline. It carries:

* identity (``name``, ``display_name``)
* auto-detection signals (regex patterns that, when matched against
  page text, contribute to the profile's score)
* prompt fragment (medical-RCM-style reminders to inject into the
  extraction prompt for this profile)
* schema overlay name (the schema to *add* on top of the document-type
  schema — used by medical-RCM to bring HEALTHCARE_FIELDS back in,
  scoped only to medical documents)
* validator hints (which validator packs are in advisory vs. blocking
  mode for this profile)
* export emitters that this profile enables (``ccda``, ``x12_275`` for
  medical-RCM)

Profiles are deliberately **declarative**, not imperative — adding a
new profile means adding a registration line, not editing the
analyzer / prompt builder / validator.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Pattern


# ---------------------------------------------------------------------------
# Detection signals
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProfileSignal:
    """
    A single auto-detection signal.

    Each signal carries a pre-compiled regex and a positive score
    contribution. The detection step runs every signal against the
    available text (header text + classification key features) and sums
    contributions per profile.

    Attributes
    ----------
    name:
        Short name for diagnostics (e.g. ``"npi_pattern"``).
    pattern:
        Pre-compiled regex. Should be permissive enough to match noisy
        OCR output but specific enough not to fire on unrelated text.
    score:
        Positive float contributed when the pattern matches at least
        once. Magnitudes are tuned so a single strong signal (header
        match) plus one supporting signal can clear the
        confidence threshold (≥ 0.6) without requiring three or more.
    description:
        One-line human-readable description for logs/UI.
    """

    name: str
    pattern: Pattern[str]
    score: float
    description: str

    def matches(self, text: str) -> bool:
        """Return True iff the regex finds at least one match."""
        return self.pattern.search(text) is not None


# ---------------------------------------------------------------------------
# Profile descriptor
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProfileDescriptor:
    """
    Frozen description of a document profile.

    Attributes
    ----------
    name:
        Stable id used in config, logs, and on-disk paths
        (e.g. ``"medical-rcm"``). Must match
        ``^[a-z0-9][a-z0-9\\-]*$``.
    display_name:
        Human-readable label shown in the UI
        (e.g. ``"Medical / Revenue Cycle"``).
    description:
        One-paragraph description of what this profile covers.
    signals:
        Ordered list of ``ProfileSignal``. Detection sums positive
        scores; ties are broken by registration order (registry
        responsibility).
    prompt_fragment:
        Markdown-formatted text injected into the extraction prompt
        when this profile is active. Pure additive — never replaces
        any base prompt content. Must be self-contained markdown.
    schema_overlay_fields:
        Names of additional ``FieldDefinition`` blocks to attach to
        the resolved schema when this profile is active. The actual
        ``FieldDefinition`` objects live in
        ``src.schemas.profile_overlays``; this descriptor just names
        them.
    validator_packs:
        Validator packs that should run for this profile, with
        per-pack mode (``"blocking"`` | ``"advisory"``). The validator
        layer reads this; unknown packs are ignored.
    enabled_emitters:
        Export emitters that become available for this profile
        (``"ccda"``, ``"x12_275"``, …). Generic profile leaves this
        empty.
    confidence_floor:
        Minimum confidence required for this profile to be selected.
        Defaults to 0.6 — anything below that falls back to generic.
    """

    name: str
    display_name: str
    description: str
    signals: tuple[ProfileSignal, ...] = field(default_factory=tuple)
    prompt_fragment: str = ""
    schema_overlay_fields: tuple[str, ...] = field(default_factory=tuple)
    validator_packs: dict[str, str] = field(default_factory=dict)
    enabled_emitters: tuple[str, ...] = field(default_factory=tuple)
    confidence_floor: float = 0.6

    def to_serialisable(self) -> dict[str, object]:
        """Lossy summary safe for logs / UI / API responses."""
        return {
            "name": self.name,
            "display_name": self.display_name,
            "description": self.description,
            "signals": [
                {"name": s.name, "score": s.score, "description": s.description}
                for s in self.signals
            ],
            "schema_overlay_fields": list(self.schema_overlay_fields),
            "validator_packs": dict(self.validator_packs),
            "enabled_emitters": list(self.enabled_emitters),
            "confidence_floor": self.confidence_floor,
        }


# ---------------------------------------------------------------------------
# Detection result
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ProfileDetectionResult:
    """
    Output of ``detect_profile``.

    Attributes
    ----------
    profile_name:
        The selected profile (``"generic-document"`` if no profile
        cleared the floor).
    confidence:
        Score in [0, 1]. We map raw additive scores onto [0, 1] via
        the deterministic ``min(raw_score, 1.0)`` mapping so
        downstream code can reason about it as a probability without
        needing to know per-signal magnitudes.
    score_by_profile:
        Raw score map for diagnostics.
    matched_signals:
        Per-profile list of matched signal names — useful for the
        audit log / Phoenix span attribute.
    fallback_to_generic:
        True when the highest-scoring profile failed to clear its
        ``confidence_floor`` and we fell back to generic. Audit
        consumers want to know the difference between "we were sure
        it's generic" and "we couldn't decide so we defaulted".
    """

    profile_name: str
    confidence: float
    score_by_profile: dict[str, float]
    matched_signals: dict[str, list[str]]
    fallback_to_generic: bool


# ---------------------------------------------------------------------------
# Helper: compile a regex once
# ---------------------------------------------------------------------------


def compile_signal(
    *,
    name: str,
    pattern: str,
    score: float,
    description: str,
    flags: int = re.IGNORECASE,
) -> ProfileSignal:
    """Convenience wrapper that pre-compiles ``pattern``."""
    return ProfileSignal(
        name=name,
        pattern=re.compile(pattern, flags),
        score=score,
        description=description,
    )
