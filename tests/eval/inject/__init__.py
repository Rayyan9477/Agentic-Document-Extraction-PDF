"""Hallucination injection harness.

Mutates a known-good extraction with one of six well-defined
distortions and runs the validator + Critic + reconciler stack
against the corrupted output. The catch-rate per injection type is
the headline metric; we publish it nightly so regressions show up
within 24 hours.

Six injection types (see ``InjectionType`` in ``runner.py``):

* ``value_swap`` — swap two field values across the document.
* ``amount_fake`` — replace a currency amount with a plausible-looking
  fake value (rounded to a "nice" number).
* ``phantom_field`` — invent an entire field that isn't on the page.
* ``bbox_drift`` — keep the value but move its bbox to a different
  page region.
* ``field_drop`` — drop a required field entirely.
* ``placeholder_inject`` — replace a value with a textbook placeholder
  ("John Doe", "ACME Corp", "123-45-6789").

Public API:

* ``InjectionRunner.run(extraction, injections=...)`` →
  ``InjectionReport``.
* ``classify_caught(...)`` translates a Critic / validator output
  into a "caught vs missed" verdict for the catch-rate matrix.
"""

from tests.eval.inject.runner import (
    InjectionRunner,
    InjectionResult,
    InjectionType,
    InjectionConfig,
)
from tests.eval.inject.report import (
    CAUGHT,
    MISSED,
    NOT_APPLICABLE,
    TRACKED_LAYERS,
    InjectionReport,
    classify_caught,
    confusion_matrix,
)

__all__ = [
    "InjectionRunner",
    "InjectionResult",
    "InjectionType",
    "InjectionConfig",
    "InjectionReport",
    "classify_caught",
    "confusion_matrix",
    "CAUGHT",
    "MISSED",
    "NOT_APPLICABLE",
    "TRACKED_LAYERS",
]
