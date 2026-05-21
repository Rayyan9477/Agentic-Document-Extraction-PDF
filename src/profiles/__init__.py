"""
V3 Phase 5 — Document profile system.

A *profile* is the orthogonal axis to *modality*: where modalities describe
**how** a page looks (printed / handwritten / fax / table / form / visual),
profiles describe **what kind of document it is** for downstream policy
purposes — generic, medical-RCM, finance, legal, …

The profile decision controls:

* Which prompt fragment is appended to the extraction prompt (medical-RCM
  gets CPT/ICD/modifier reminders; generic gets nothing).
* Which validator pack runs in advisory vs. blocking mode (medical-RCM
  enforces NPI Luhn check; generic logs but does not block).
* Which schema overlay is applied on top of the document-type schema
  (medical-RCM overlays HEALTHCARE_FIELDS; generic does not).
* Calibration table partitioning (per-(profile, tenant) tables).
* RCM-specific export shapes (C-CDA / X12N 275) only run when the
  document is medical-RCM.

Profile selection is **conservative**: when auto-detection signals are
weak (confidence < 0.6), we fall back to ``generic-document`` with all
validator packs running advisory-only. We never silently disable a
check, and we never silently invent medical fields on a non-medical
doc.

Public API:

* ``ProfileDescriptor`` — frozen dataclass describing a profile.
* ``ProfileRegistry`` — registry of all profiles, with auto-detection.
* ``get_profile(name)`` — fetch a profile by name (default: ``generic-document``).
* ``detect_profile(...)`` — score-based auto-detection from analyzer output.
"""

from __future__ import annotations

from src.profiles.descriptor import (
    ProfileDescriptor,
    ProfileDetectionResult,
    ProfileSignal,
)
from src.profiles.registry import (
    ProfileRegistry,
    detect_profile,
    get_profile,
)

# Eagerly register the built-in profiles. Submodule imports are
# deliberate — each module's import side-effect registers the profile
# on the singleton ``ProfileRegistry``.
from src.profiles import finance as _finance  # noqa: F401
from src.profiles import generic as _generic  # noqa: F401
from src.profiles import medical_rcm as _medical_rcm  # noqa: F401


__all__ = [
    "ProfileDescriptor",
    "ProfileDetectionResult",
    "ProfileRegistry",
    "ProfileSignal",
    "detect_profile",
    "get_profile",
]
