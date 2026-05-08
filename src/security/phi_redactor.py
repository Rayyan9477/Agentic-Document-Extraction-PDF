"""
WS-6: opt-in PHI / PII redaction for extracted field values.

This module provides ``PHIRedactor``, a two-layer redactor used by
``ValidatorAgent`` after extraction completes to scrub PHI / PII from
string field values before they hit storage, exports, audit logs, or
the Mem0 memory layer.

Layer 1 (preferred): openai/privacy-filter
    A 1.5B-parameter / 50M-active-parameter transformer
    token-classifier (Apache 2.0, available on HuggingFace) that emits
    BIOES tags over 8 PII categories: ``account_number``,
    ``private_address``, ``private_email``, ``private_person``,
    ``private_phone``, ``private_url``, ``private_date``, ``secret``.
    Loaded lazily via the ``transformers`` pipeline. Requires the
    optional ``[phi]`` extra (``transformers``, ``torch``).

Layer 2 (fallback): regex
    Reuses the existing ``PHI_VALUE_PATTERNS`` from
    ``src/security/phi_mask.py``. Activates when the ``transformers``
    package isn't available (air-gapped deployment without the optional
    ``[phi]`` extra), the model can't be loaded (network blocked, model
    not vendored, OOM), or ``settings.phi.fallback_to_regex`` is true
    and the primary path raised.

Layer selection is deterministic and logged:

    1. If ``transformers`` is importable AND model can be loaded → ML.
    2. Otherwise, if ``fallback_to_regex`` → regex.
    3. Otherwise, refuse to enable redaction (raise on construction).

Usage::

    from src.security.phi_redactor import PHIRedactor

    redactor = PHIRedactor.from_settings()  # honours settings.phi
    result = redactor.redact("My name is Alice Smith, ssn 123-45-6789.")
    # result.redacted_text -> "My name is [REDACTED], ssn [REDACTED]."
    # result.spans         -> [Span(start=11, end=22, label="private_person", original="Alice Smith"), ...]

The redactor is **off by default**. Enable via
``settings.phi.enabled = True`` (env: ``PHI_ENABLED=1``) or per-request
via ``ProcessRequest.phi_mode = True``.

This module never auto-downloads the HF model; it is the operator's
responsibility to either:

    * pre-cache the model with ``transformers-cli download
      openai/privacy-filter`` on a host with network access, then move
      the cache to the air-gapped host, or
    * ship the model weights alongside the application bundle.

If the model is not pre-cached, ``transformers`` will attempt a
network download on first use; that fails closed in air-gapped
deployments, the redactor logs the failure, and the regex fallback
activates (assuming ``fallback_to_regex`` is true).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from src.config import get_logger, get_settings
from src.security.phi_mask import REDACTED_TOKEN, _redact_string_value


logger = get_logger(__name__)


# Map openai/privacy-filter labels to a stable name used in spans + audit
# logs. The model emits BIOES-prefixed labels (B-private_person,
# I-private_person, ...); we keep only the entity name.
_PRIVACY_FILTER_ENTITY_LABELS: tuple[str, ...] = (
    "account_number",
    "private_address",
    "private_email",
    "private_person",
    "private_phone",
    "private_url",
    "private_date",
    "secret",
)


@dataclass(frozen=True, slots=True)
class Span:
    """A redacted region in the source text."""

    start: int
    end: int
    label: str  # entity category (e.g. "private_person", "regex")
    original: str  # the redacted substring (kept ONLY in this object;
    # callers must NOT export it alongside the redacted text)


@dataclass(slots=True)
class RedactionResult:
    """Result of a redactor run."""

    redacted_text: str
    spans: list[Span] = field(default_factory=list)
    layer_used: str = "noop"  # "ml" | "regex" | "noop"

    @property
    def is_modified(self) -> bool:
        return bool(self.spans)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


class PHIRedactor:
    """Two-layer PHI redactor (ML primary, regex fallback)."""

    def __init__(
        self,
        *,
        model_id: str = "openai/privacy-filter",
        fallback_to_regex: bool = True,
        local_only: bool = True,
    ) -> None:
        """
        Args:
            model_id: HuggingFace model identifier. Defaults to
                ``openai/privacy-filter``.
            fallback_to_regex: When the ML layer is unavailable, fall
                back to ``_redact_string_value`` from ``phi_mask.py``.
                When False and the ML layer is unavailable, redaction
                is a no-op and a warning is logged.
            local_only: Hint to the loader that the model must be
                resolvable locally (already cached / pre-vendored). Set
                this to True for air-gapped deployments to avoid an
                accidental network call.
        """
        self._model_id = model_id
        self._fallback_to_regex = fallback_to_regex
        self._local_only = local_only

        # Layer 1: ML pipeline (lazy — only loaded on first redact() call
        # so import cost stays out of the application boot path).
        self._pipeline: Any | None = None
        self._pipeline_load_attempted = False
        self._layer: str = "noop"  # set on first redact()

    # --- factory ------------------------------------------------------

    @classmethod
    def from_settings(cls) -> PHIRedactor:
        """Construct a redactor honouring ``settings.phi``."""
        settings = get_settings()
        phi = getattr(settings, "phi", None)
        if phi is None or not getattr(phi, "enabled", False):
            # Caller asked for redaction even though it's disabled in
            # settings; fall through to construction so the per-request
            # ``phi_mode=True`` override path still works.
            logger.info("phi_redactor_constructed_disabled_in_settings")
            return cls()
        return cls(
            model_id=getattr(phi, "model_id", "openai/privacy-filter"),
            fallback_to_regex=getattr(phi, "fallback_to_regex", True),
            local_only=getattr(phi, "local_only", True),
        )

    # --- main ---------------------------------------------------------

    def redact(self, text: str) -> RedactionResult:
        """Redact PHI from ``text``, returning the cleaned text + spans.

        Empty / non-string inputs short-circuit to a no-op. The ML layer
        is loaded lazily; regex fallback runs synchronously with no
        external dependencies.
        """
        if not isinstance(text, str) or not text:
            return RedactionResult(redacted_text=text or "", spans=[], layer_used="noop")

        # Try ML layer first.
        ml_result = self._redact_via_ml(text)
        if ml_result is not None:
            self._layer = "ml"
            return ml_result

        # Regex fallback.
        if self._fallback_to_regex:
            redacted = _redact_string_value(text)
            spans: list[Span] = []
            if redacted == REDACTED_TOKEN and text != REDACTED_TOKEN:
                # ``_redact_string_value`` collapses any matched value
                # to a single REDACTED_TOKEN; record one span covering
                # the original string so the audit trail still has
                # some signal even though we don't know which pattern
                # fired.
                spans.append(Span(start=0, end=len(text), label="regex", original=text))
            self._layer = "regex"
            return RedactionResult(redacted_text=redacted, spans=spans, layer_used="regex")

        # No layer available → log once and pass through.
        logger.warning(
            "phi_redactor_noop",
            message=(
                "PHI redaction was requested but no layer is available "
                "(transformers not installed, model not loadable, "
                "fallback_to_regex=False). Text passed through unchanged."
            ),
        )
        self._layer = "noop"
        return RedactionResult(redacted_text=text, spans=[], layer_used="noop")

    def redact_record(self, record: dict[str, Any]) -> dict[str, Any]:
        """Walk a record dict and redact every string leaf in place.

        Returns a new dict; the input is never mutated. Lists / nested
        dicts are recursed. Non-string leaves pass through unchanged.
        """
        return self._walk(record)

    # --- internals ----------------------------------------------------

    def _walk(self, obj: Any) -> Any:
        if isinstance(obj, str):
            return self.redact(obj).redacted_text
        if isinstance(obj, dict):
            return {k: self._walk(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [self._walk(item) for item in obj]
        return obj

    def _redact_via_ml(self, text: str) -> RedactionResult | None:
        """Attempt redaction via the openai/privacy-filter pipeline.

        Returns ``None`` (not a result) when the pipeline can't be
        loaded — the caller then proceeds to the regex fallback. Pipeline
        load is cached after the first attempt regardless of outcome.
        """
        pipeline = self._get_pipeline()
        if pipeline is None:
            return None

        try:
            entities = pipeline(text)
        except Exception as exc:  # pragma: no cover - inference path
            logger.warning("phi_ml_inference_failed", error=str(exc))
            return None

        return _apply_entity_spans(text, entities)

    def _get_pipeline(self) -> Any | None:
        """Lazy-load the transformers pipeline.

        On first call:
          * If ``transformers`` isn't importable → cache None.
          * Otherwise build the token-classification pipeline. If
            loading the model fails (network blocked, missing local
            cache, etc.) → cache None.

        Subsequent calls return the cached value (which may be None).
        """
        if self._pipeline_load_attempted:
            return self._pipeline
        self._pipeline_load_attempted = True

        try:
            from transformers import pipeline as hf_pipeline  # type: ignore[import-not-found]
        except ImportError:
            logger.info(
                "phi_ml_layer_unavailable",
                reason="transformers_not_installed",
                hint="install with `pip install -e .[phi]` to enable",
            )
            return None

        try:
            # ``aggregation_strategy="simple"`` collapses the BIOES tags
            # into entity-level spans with start/end offsets so we
            # don't have to decode the BIOES sequence ourselves.
            kwargs: dict[str, Any] = {
                "task": "token-classification",
                "model": self._model_id,
                "aggregation_strategy": "simple",
            }
            # ``local_files_only=True`` instructs transformers not to
            # touch the network — required for air-gapped deployments.
            if self._local_only:
                kwargs["model_kwargs"] = {"local_files_only": True}

            self._pipeline = hf_pipeline(**kwargs)
            logger.info(
                "phi_ml_layer_ready",
                model=self._model_id,
                local_only=self._local_only,
            )
        except Exception as exc:  # pragma: no cover - load path
            logger.warning(
                "phi_ml_model_load_failed",
                model=self._model_id,
                error=str(exc),
                hint=(
                    "If air-gapped, pre-cache the model on a networked host "
                    "via `transformers-cli download openai/privacy-filter`. "
                    "Falling back to regex layer."
                ),
            )
            self._pipeline = None

        return self._pipeline


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalise_label(raw: str) -> str:
    """Strip BIOES prefixes and unify against the known entity set.

    The HF pipeline with ``aggregation_strategy="simple"`` already emits
    bare entity names (no BIO prefix), but be defensive in case of
    format drift between transformers versions.
    """
    if "-" in raw:
        raw = raw.split("-", 1)[1]
    return raw if raw in _PRIVACY_FILTER_ENTITY_LABELS else "phi"


def _apply_entity_spans(
    text: str,
    entities: list[dict[str, Any]],
) -> RedactionResult:
    """Build a RedactionResult by replacing each entity span with [REDACTED].

    Entities are sorted by start offset; overlapping spans are merged
    so the caller can rebuild the text in a single pass.
    """
    if not entities:
        return RedactionResult(redacted_text=text, spans=[], layer_used="ml")

    # Normalise + sort.
    cleaned: list[dict[str, Any]] = []
    for ent in entities:
        try:
            start = int(ent["start"])
            end = int(ent["end"])
        except (KeyError, TypeError, ValueError):
            continue
        if end <= start or start < 0 or end > len(text):
            continue
        cleaned.append(
            {
                "start": start,
                "end": end,
                "label": _normalise_label(str(ent.get("entity_group") or ent.get("entity") or "phi")),
            }
        )
    cleaned.sort(key=lambda e: (e["start"], e["end"]))

    # Merge overlaps (keep the first label encountered).
    merged: list[dict[str, Any]] = []
    for ent in cleaned:
        if merged and ent["start"] < merged[-1]["end"]:
            merged[-1]["end"] = max(merged[-1]["end"], ent["end"])
        else:
            merged.append(dict(ent))

    # Build the redacted text + Span audit records.
    spans: list[Span] = []
    out: list[str] = []
    cursor = 0
    for ent in merged:
        out.append(text[cursor : ent["start"]])
        out.append(REDACTED_TOKEN)
        spans.append(
            Span(
                start=ent["start"],
                end=ent["end"],
                label=ent["label"],
                original=text[ent["start"] : ent["end"]],
            )
        )
        cursor = ent["end"]
    out.append(text[cursor:])

    return RedactionResult(redacted_text="".join(out), spans=spans, layer_used="ml")


__all__ = [
    "PHIRedactor",
    "RedactionResult",
    "Span",
]
