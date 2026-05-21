"""
Profile registry + auto-detection.

The registry is a thread-safe singleton. Profiles register themselves
at module-import time (see ``src/profiles/__init__.py``).

Auto-detection (``detect_profile``) takes the analyzer's structured
output plus the page-text content and returns a ``ProfileDetectionResult``.
The detection rule is intentionally simple and conservative:

1. For each registered profile, sum the score contribution of every
   signal whose pattern matches the input text.
2. Pick the profile with the highest raw score whose score is at
   least its ``confidence_floor``.
3. If no profile clears its floor → fall back to ``generic-document``
   with ``fallback_to_generic=True``.

We deliberately do NOT use a VLM call for profile detection. Profile
detection runs on text the VLM has already produced (or text extracted
from the PDF text layer), so it adds zero latency and the decision is
purely deterministic.
"""

from __future__ import annotations

import threading
from typing import Iterable

from src.config import get_logger
from src.profiles.descriptor import (
    ProfileDescriptor,
    ProfileDetectionResult,
)


logger = get_logger(__name__)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


GENERIC_PROFILE_NAME = "generic-document"


class ProfileRegistry:
    """
    Thread-safe singleton registry of ``ProfileDescriptor``.

    Implements the registry pattern with a class-level lock — callers
    don't need to coordinate, and tests can ``reset()`` between runs.
    """

    _instance: "ProfileRegistry | None" = None
    _instance_lock: threading.Lock = threading.Lock()

    def __new__(cls) -> "ProfileRegistry":
        with cls._instance_lock:
            if cls._instance is None:
                instance = super().__new__(cls)
                instance._init_state()  # type: ignore[attr-defined]
                cls._instance = instance
            return cls._instance

    def _init_state(self) -> None:
        self._lock = threading.Lock()
        self._profiles: dict[str, ProfileDescriptor] = {}
        self._registration_order: list[str] = []

    def register(self, descriptor: ProfileDescriptor) -> None:
        """
        Register a profile.

        Re-registering the same name **overwrites** the previous
        descriptor — useful in tests, harmless in production where
        each profile module registers exactly once at import.
        """
        with self._lock:
            if descriptor.name in self._profiles:
                logger.debug("profile_re_registered", name=descriptor.name)
                # Preserve registration order: don't re-append.
            else:
                self._registration_order.append(descriptor.name)
            self._profiles[descriptor.name] = descriptor

    def get(self, name: str) -> ProfileDescriptor:
        """
        Return the descriptor for ``name``.

        Falls back to the generic profile if ``name`` is unknown,
        emitting a warning. We never raise here — profile names flow
        from config, and a typo in config should not crash extraction.
        """
        with self._lock:
            descriptor = self._profiles.get(name)
            if descriptor is None:
                logger.warning(
                    "profile_unknown_falling_back_to_generic",
                    requested=name,
                    available=list(self._profiles.keys()),
                )
                generic = self._profiles.get(GENERIC_PROFILE_NAME)
                if generic is None:
                    raise RuntimeError(
                        f"profile '{name}' not registered and "
                        f"'{GENERIC_PROFILE_NAME}' fallback not registered either"
                    )
                return generic
            return descriptor

    def names(self) -> list[str]:
        """Return registered profile names in registration order."""
        with self._lock:
            return list(self._registration_order)

    def all(self) -> list[ProfileDescriptor]:
        """Return all registered descriptors in registration order."""
        with self._lock:
            return [self._profiles[n] for n in self._registration_order]

    def reset(self) -> None:
        """Wipe registry state. Tests only."""
        with self._lock:
            self._profiles.clear()
            self._registration_order.clear()


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------


def get_profile(name: str | None) -> ProfileDescriptor:
    """
    Resolve a profile by name, defaulting to generic.

    ``None`` and the empty string both resolve to generic. This is the
    single canonical entry point — call sites should not construct a
    ``ProfileRegistry`` directly.
    """
    registry = ProfileRegistry()
    if not name:
        return registry.get(GENERIC_PROFILE_NAME)
    return registry.get(name)


def detect_profile(
    *,
    classification_features: Iterable[str] | None = None,
    page_text: str | None = None,
    document_type: str | None = None,
    profile_override: str | None = None,
) -> ProfileDetectionResult:
    """
    Auto-detect the document profile from analyzer output.

    Parameters
    ----------
    classification_features:
        ``key_features_found`` from the analyzer's classification VLM
        call. Treat as a list of phrases — the strongest signal source
        because it's already filtered by the VLM.
    page_text:
        Concatenated text content of the first 1-2 pages
        (``PageImage.text_content``). Optional — when the PDF is
        image-only this will be empty and we lean on
        ``classification_features`` plus ``document_type``.
    document_type:
        The classifier's normalized document type (e.g. ``"CMS-1500"``,
        ``"UB-04"``). When this matches a known medical document type
        it provides a strong implicit signal regardless of text content.
    profile_override:
        User-supplied override (e.g. from the upload UI's profile chip).
        When set and the name is registered, returns it directly with
        confidence 1.0 and no detection scoring. We honour the override
        without question — operators choosing a profile know what they
        want.

    Returns
    -------
    ProfileDetectionResult
        Detection outcome. Always returns a valid profile name, even
        if it's the generic fallback.
    """
    registry = ProfileRegistry()

    # ----- Override path: trust the operator. ---------------------------
    if profile_override:
        descriptor = registry.get(profile_override)
        return ProfileDetectionResult(
            profile_name=descriptor.name,
            confidence=1.0,
            score_by_profile={descriptor.name: 1.0},
            matched_signals={descriptor.name: ["operator_override"]},
            fallback_to_generic=False,
        )

    # ----- Build the search corpus. -------------------------------------
    text_parts: list[str] = []
    if classification_features:
        text_parts.extend(str(f) for f in classification_features)
    if page_text:
        text_parts.append(page_text)
    if document_type:
        text_parts.append(document_type)

    haystack = "\n".join(text_parts)

    # ----- Score each profile. ------------------------------------------
    score_by_profile: dict[str, float] = {}
    matched_signals: dict[str, list[str]] = {}
    for descriptor in registry.all():
        score = 0.0
        matched: list[str] = []
        for signal in descriptor.signals:
            if signal.matches(haystack):
                score += signal.score
                matched.append(signal.name)
        score_by_profile[descriptor.name] = score
        matched_signals[descriptor.name] = matched

    # ----- Pick the winner. ---------------------------------------------
    # Sort by raw score descending; tiebreak by registration order so
    # generic is last.
    ordered = registry.names()
    best_name: str | None = None
    best_score = 0.0
    for name in ordered:
        score = score_by_profile.get(name, 0.0)
        if score > best_score and score >= registry.get(name).confidence_floor:
            best_name = name
            best_score = score

    if best_name is None:
        # No profile cleared its floor → fallback.
        return ProfileDetectionResult(
            profile_name=GENERIC_PROFILE_NAME,
            confidence=score_by_profile.get(GENERIC_PROFILE_NAME, 0.0),
            score_by_profile=score_by_profile,
            matched_signals=matched_signals,
            fallback_to_generic=True,
        )

    return ProfileDetectionResult(
        profile_name=best_name,
        confidence=min(best_score, 1.0),
        score_by_profile=score_by_profile,
        matched_signals=matched_signals,
        fallback_to_generic=False,
    )
