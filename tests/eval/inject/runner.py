"""V3 Phase 6 — hallucination-injection runner.

Takes a known-good extraction (a "golden" record) and produces a
mutated copy carrying one of six well-defined distortions. The
distortion is what we're testing the Critic / validator stack to
catch.

The runner is **deterministic**: a seeded RNG plus a stable JSON
key-ordering guarantees the same input + injection_type always
yields the same output. That matters because nightly catch-rate
comparisons need to compare apples to apples.

The runner does NOT call any VLM. It works on already-extracted
JSON. Tests of the *full* extract → critic → catch loop live in
``tests/integration/test_critic_routing.py``; the harness here is
the offline driver that builds the golden mutation set those tests
consume.
"""

from __future__ import annotations

import copy
import random
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Injection vocabulary
# ---------------------------------------------------------------------------


class InjectionType(str, Enum):
    """The six canonical injection types.

    Naming matches the EXECUTION_PLAN. Adding a new type is a
    breaking change to the catch-rate baseline — coordinate.
    """

    VALUE_SWAP = "value_swap"
    AMOUNT_FAKE = "amount_fake"
    PHANTOM_FIELD = "phantom_field"
    BBOX_DRIFT = "bbox_drift"
    FIELD_DROP = "field_drop"
    PLACEHOLDER_INJECT = "placeholder_inject"


@dataclass(frozen=True, slots=True)
class InjectionConfig:
    """Tuning knobs for the runner.

    Each injection type has a small handful of well-known knobs;
    rather than scatter them across kwargs we collect them here.
    All have sensible defaults — most callers construct
    ``InjectionConfig()`` and pass it as-is.
    """

    rng_seed: int = 42
    # PHANTOM_FIELD — name of the synthetic field to add.
    phantom_field_name: str = "phantom_provider_id"
    phantom_field_value: str = "PHANTOM-9999"
    # AMOUNT_FAKE — round real currency to this magnitude when faking.
    amount_round_to: float = 100.0
    # BBOX_DRIFT — fraction of the page to drift by.
    bbox_drift_fraction: float = 0.4
    # PLACEHOLDER_INJECT — pool of placeholders we know payers
    # frequently see in test data.
    placeholder_pool: tuple[str, ...] = (
        "John Doe",
        "Jane Doe",
        "ACME Corporation",
        "123-45-6789",
        "1111111111",
        "TEST PATIENT",
    )


@dataclass(slots=True)
class InjectionResult:
    """The outcome of one injection on one record."""

    record_id: str
    injection_type: InjectionType
    field_name: str | None
    original_value: Any
    injected_value: Any
    mutated_extraction: dict[str, Any]
    notes: str = ""


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------


_AMOUNT_PATTERN = re.compile(r"^\$?-?\d[\d,]*(?:\.\d{1,2})?$")


@dataclass(slots=True)
class InjectionRunner:
    """Deterministic injection driver.

    Construct once per evaluation pass; call ``run`` for each
    record. The runner reseeds the RNG at the start of every
    ``run`` so identical inputs always produce identical mutations.
    """

    config: InjectionConfig = field(default_factory=InjectionConfig)

    def run(
        self,
        extraction: dict[str, Any],
        *,
        injection_type: InjectionType,
        record_id: str = "rec_0",
    ) -> InjectionResult:
        """Apply a single injection to ``extraction``.

        ``extraction`` is the merged-extraction dict
        (``{field_name: value}``). The runner returns a fresh,
        mutated copy — the input is never modified.
        """
        rng = random.Random(self.config.rng_seed)
        mutated = copy.deepcopy(extraction)

        if injection_type == InjectionType.VALUE_SWAP:
            return self._inject_value_swap(rng, extraction, mutated, record_id)
        if injection_type == InjectionType.AMOUNT_FAKE:
            return self._inject_amount_fake(rng, extraction, mutated, record_id)
        if injection_type == InjectionType.PHANTOM_FIELD:
            return self._inject_phantom_field(extraction, mutated, record_id)
        if injection_type == InjectionType.BBOX_DRIFT:
            return self._inject_bbox_drift(rng, extraction, mutated, record_id)
        if injection_type == InjectionType.FIELD_DROP:
            return self._inject_field_drop(rng, extraction, mutated, record_id)
        if injection_type == InjectionType.PLACEHOLDER_INJECT:
            return self._inject_placeholder(rng, extraction, mutated, record_id)
        raise ValueError(f"unknown injection type: {injection_type}")

    # ----- Injectors --------------------------------------------------

    def _inject_value_swap(
        self,
        rng: random.Random,
        extraction: dict[str, Any],
        mutated: dict[str, Any],
        record_id: str,
    ) -> InjectionResult:
        """Swap the values of two compatible fields."""
        # Find two distinct fields with non-null primitive values of
        # the same broad type.
        items = [
            (k, v)
            for k, v in extraction.items()
            if v is not None and isinstance(v, (str, int, float, bool))
        ]
        if len(items) < 2:
            return InjectionResult(
                record_id=record_id,
                injection_type=InjectionType.VALUE_SWAP,
                field_name=None,
                original_value=None,
                injected_value=None,
                mutated_extraction=mutated,
                notes="not enough swappable fields — no-op",
            )
        # Sort to keep the choice deterministic across Python dict
        # ordering changes.
        items.sort(key=lambda kv: kv[0])
        a, b = rng.sample(items, k=2)
        mutated[a[0]] = b[1]
        mutated[b[0]] = a[1]
        return InjectionResult(
            record_id=record_id,
            injection_type=InjectionType.VALUE_SWAP,
            field_name=a[0],
            original_value=a[1],
            injected_value=b[1],
            mutated_extraction=mutated,
            notes=f"swapped {a[0]!r} ↔ {b[0]!r}",
        )

    def _inject_amount_fake(
        self,
        rng: random.Random,
        extraction: dict[str, Any],
        mutated: dict[str, Any],
        record_id: str,
    ) -> InjectionResult:
        """Replace a currency-looking value with a "nice" round fake."""
        items = sorted(
            (k, v) for k, v in extraction.items()
            if isinstance(v, str) and _AMOUNT_PATTERN.match(str(v).strip())
        )
        if not items:
            # Try numeric values too.
            items = sorted(
                (k, v) for k, v in extraction.items()
                if isinstance(v, (int, float))
                and ("amount" in k.lower() or "charge" in k.lower())
            )
        if not items:
            return InjectionResult(
                record_id=record_id,
                injection_type=InjectionType.AMOUNT_FAKE,
                field_name=None,
                original_value=None,
                injected_value=None,
                mutated_extraction=mutated,
                notes="no amount-shaped field found — no-op",
            )
        chosen = rng.choice(items)
        # Build a "nice" fake: a multiple of ``amount_round_to`` near
        # 5x the expected order of magnitude — large enough to be
        # noticeably wrong on a real charge but not so far out it
        # blows past validator min/max.
        fake_value = self.config.amount_round_to * 7
        fake_str = (
            f"${fake_value:,.2f}"
            if isinstance(chosen[1], str) and chosen[1].lstrip("-").startswith("$")
            else fake_value
        )
        mutated[chosen[0]] = fake_str
        return InjectionResult(
            record_id=record_id,
            injection_type=InjectionType.AMOUNT_FAKE,
            field_name=chosen[0],
            original_value=chosen[1],
            injected_value=fake_str,
            mutated_extraction=mutated,
            notes=f"faked currency on {chosen[0]!r}",
        )

    def _inject_phantom_field(
        self,
        extraction: dict[str, Any],
        mutated: dict[str, Any],
        record_id: str,
    ) -> InjectionResult:
        """Add a synthetic field that isn't on the source page.

        The Critic / bbox-roundtrip layer should catch this because
        a phantom field has no anchored bbox AND its value is not
        present in the source image text.
        """
        if self.config.phantom_field_name in mutated:
            # Don't accidentally overwrite a real field.
            return InjectionResult(
                record_id=record_id,
                injection_type=InjectionType.PHANTOM_FIELD,
                field_name=None,
                original_value=None,
                injected_value=None,
                mutated_extraction=mutated,
                notes=(
                    f"phantom field name {self.config.phantom_field_name!r} "
                    "already exists — no-op"
                ),
            )
        mutated[self.config.phantom_field_name] = self.config.phantom_field_value
        return InjectionResult(
            record_id=record_id,
            injection_type=InjectionType.PHANTOM_FIELD,
            field_name=self.config.phantom_field_name,
            original_value=None,
            injected_value=self.config.phantom_field_value,
            mutated_extraction=mutated,
            notes="added phantom field",
        )

    def _inject_bbox_drift(
        self,
        rng: random.Random,
        extraction: dict[str, Any],
        mutated: dict[str, Any],
        record_id: str,
    ) -> InjectionResult:
        """Keep the value but corrupt its bbox.

        We expect the input shape to use the FieldValue envelope or a
        ``_bbox`` sibling key. We try a couple of common shapes and
        no-op gracefully if neither is present.
        """
        # Shape 1: FieldValue dict — ``{value, _provenance: {bbox: ...}}``
        # Shape 2: legacy ``{value, bbox: [...]}``
        candidates: list[tuple[str, dict[str, Any]]] = []
        for k, v in extraction.items():
            if isinstance(v, dict):
                if isinstance(v.get("_provenance"), dict) and v["_provenance"].get("bbox"):
                    candidates.append((k, v))
                elif v.get("bbox"):
                    candidates.append((k, v))
        if not candidates:
            return InjectionResult(
                record_id=record_id,
                injection_type=InjectionType.BBOX_DRIFT,
                field_name=None,
                original_value=None,
                injected_value=None,
                mutated_extraction=mutated,
                notes="no bbox-bearing field — no-op",
            )
        candidates.sort(key=lambda kv: kv[0])
        chosen_name, chosen_value = rng.choice(candidates)
        drift = self.config.bbox_drift_fraction

        def _drift_bbox(b: Any) -> Any:
            # Accept either {x, y, width, height} dict or
            # [x1, y1, x2, y2] list.
            if isinstance(b, dict):
                drifted = dict(b)
                drifted["x"] = max(0.0, min(1.0, float(b.get("x", 0)) + drift))
                drifted["y"] = max(0.0, min(1.0, float(b.get("y", 0)) + drift))
                return drifted
            if isinstance(b, (list, tuple)) and len(b) == 4:
                shifted = [float(c) + drift for c in b]
                # Clamp to [0, 1].
                return [max(0.0, min(1.0, c)) for c in shifted]
            return b

        original_bbox = None
        mutated_value = copy.deepcopy(chosen_value)
        if isinstance(mutated_value.get("_provenance"), dict):
            original_bbox = mutated_value["_provenance"].get("bbox")
            mutated_value["_provenance"]["bbox"] = _drift_bbox(original_bbox)
        else:
            original_bbox = mutated_value.get("bbox")
            mutated_value["bbox"] = _drift_bbox(original_bbox)
        mutated[chosen_name] = mutated_value
        return InjectionResult(
            record_id=record_id,
            injection_type=InjectionType.BBOX_DRIFT,
            field_name=chosen_name,
            original_value=original_bbox,
            injected_value=mutated_value.get("_provenance", {}).get("bbox")
            or mutated_value.get("bbox"),
            mutated_extraction=mutated,
            notes=f"drifted bbox on {chosen_name!r} by {drift}",
        )

    def _inject_field_drop(
        self,
        rng: random.Random,
        extraction: dict[str, Any],
        mutated: dict[str, Any],
        record_id: str,
    ) -> InjectionResult:
        """Delete a non-null field outright."""
        droppable = sorted(k for k, v in extraction.items() if v is not None)
        if not droppable:
            return InjectionResult(
                record_id=record_id,
                injection_type=InjectionType.FIELD_DROP,
                field_name=None,
                original_value=None,
                injected_value=None,
                mutated_extraction=mutated,
                notes="no non-null field to drop — no-op",
            )
        target = rng.choice(droppable)
        original = mutated.pop(target, None)
        return InjectionResult(
            record_id=record_id,
            injection_type=InjectionType.FIELD_DROP,
            field_name=target,
            original_value=original,
            injected_value=None,
            mutated_extraction=mutated,
            notes=f"dropped {target!r}",
        )

    def _inject_placeholder(
        self,
        rng: random.Random,
        extraction: dict[str, Any],
        mutated: dict[str, Any],
        record_id: str,
    ) -> InjectionResult:
        """Replace a string value with a textbook placeholder."""
        items = sorted(
            (k, v) for k, v in extraction.items()
            if isinstance(v, str) and v
        )
        if not items:
            return InjectionResult(
                record_id=record_id,
                injection_type=InjectionType.PLACEHOLDER_INJECT,
                field_name=None,
                original_value=None,
                injected_value=None,
                mutated_extraction=mutated,
                notes="no string field — no-op",
            )
        target = rng.choice(items)
        placeholder = rng.choice(self.config.placeholder_pool)
        mutated[target[0]] = placeholder
        return InjectionResult(
            record_id=record_id,
            injection_type=InjectionType.PLACEHOLDER_INJECT,
            field_name=target[0],
            original_value=target[1],
            injected_value=placeholder,
            mutated_extraction=mutated,
            notes=f"placeholder on {target[0]!r}",
        )
