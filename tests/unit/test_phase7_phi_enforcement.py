"""V3 Phase 7 — PHI mode production enforcement tests.

The Settings model_validator must:

* Allow PHI=False in development / testing (legacy).
* Refuse to boot in production with PHI=False unless
  ``PHI_BYPASS_ACK`` is explicitly set to an acknowledged value.
* Allow PHI=True in production unconditionally.
"""

from __future__ import annotations

import os
from typing import Any

import pytest


# Strong-enough secrets for the production validator's other checks
# so this file's tests don't trip on unrelated rules.
_STRONG_SECRET = "Q8wXz7vN3pR4mY9bL2kJ6hG5fD1sA0c#Z!%"
_STRONG_KEY = "P9rQ2nM5bV8cK1jH4gF7dS3aL6tY0iO@!#$"


def _construct_settings(env: dict[str, str]) -> Any:
    """Build a Settings instance from a synthetic env dict.

    Clears the ``get_settings`` lru_cache between calls so each
    test gets a fresh Settings instance.

    V3 Phase 8 — these PHI-focused tests pre-acknowledge the auth
    bypass so they isolate the PHI gate. Tests that exercise the
    auth gate explicitly live in ``test_phase8_auth_enforcement.py``.
    Callers can override AUTH_BYPASS_ACK in their own ``env`` to
    test interaction.
    """
    from src.config import get_settings, settings as settings_module

    # Patch os.environ so Pydantic env-loaders see this fixture.
    original = dict(os.environ)
    try:
        # Pre-ack the auth gate unless the caller is testing it.
        if "AUTH_BYPASS_ACK" not in env and "API_AUTH_ENABLED" not in env:
            os.environ["AUTH_BYPASS_ACK"] = "acknowledged"
        for k, v in env.items():
            os.environ[k] = v
        get_settings.cache_clear()
        # Construct directly so the lru_cache isn't reused.
        return settings_module.Settings()
    finally:
        os.environ.clear()
        os.environ.update(original)
        get_settings.cache_clear()


# ---------------------------------------------------------------------------
# Production: PHI=False without bypass ack → ValueError
# ---------------------------------------------------------------------------


class TestPHIProductionEnforcement:
    def test_dev_with_phi_disabled_is_fine(self) -> None:
        # Dev environment never trips the PHI gate.
        settings = _construct_settings(
            {
                "APP_ENV": "development",
                "PHI_ENABLED": "false",
            }
        )
        assert settings.phi.enabled is False

    def test_production_with_phi_disabled_no_ack_raises(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            _construct_settings(
                {
                    "APP_ENV": "production",
                    "DEBUG": "false",
                    "SECRET_KEY": _STRONG_SECRET,
                    "ENCRYPTION_KEY": _STRONG_KEY,
                    "PHI_ENABLED": "false",
                    # No PHI_BYPASS_ACK.
                }
            )
        msg = str(excinfo.value)
        assert "PHI" in msg
        assert "PHI_BYPASS_ACK" in msg

    def test_production_with_phi_enabled_passes(self) -> None:
        settings = _construct_settings(
            {
                "APP_ENV": "production",
                "DEBUG": "false",
                "SECRET_KEY": _STRONG_SECRET,
                "ENCRYPTION_KEY": _STRONG_KEY,
                "PHI_ENABLED": "true",
            }
        )
        assert settings.phi.enabled is True

    def test_production_with_phi_disabled_and_ack_passes(self) -> None:
        # The explicit bypass is the operator's "I know what I'm
        # doing" escape hatch for non-PHI deployments.
        settings = _construct_settings(
            {
                "APP_ENV": "production",
                "DEBUG": "false",
                "SECRET_KEY": _STRONG_SECRET,
                "ENCRYPTION_KEY": _STRONG_KEY,
                "PHI_ENABLED": "false",
                "PHI_BYPASS_ACK": "acknowledged",
            }
        )
        assert settings.phi.enabled is False

    def test_bypass_ack_accepts_synonyms(self) -> None:
        # We accept "1" / "true" / "yes" / "acknowledged" — operators
        # who type yes shouldn't be tripped by case.
        for ack in ("1", "true", "TRUE", "yes", "Yes", "acknowledged"):
            settings = _construct_settings(
                {
                    "APP_ENV": "production",
                    "DEBUG": "false",
                    "SECRET_KEY": _STRONG_SECRET,
                    "ENCRYPTION_KEY": _STRONG_KEY,
                    "PHI_ENABLED": "false",
                    "PHI_BYPASS_ACK": ack,
                }
            )
            assert settings.phi.enabled is False, f"ack={ack!r} should pass"

    def test_bypass_ack_rejects_off(self) -> None:
        # Random non-truthy strings should NOT count as ack.
        for ack in ("no", "off", "0", "false", "lol"):
            with pytest.raises(ValueError):
                _construct_settings(
                    {
                        "APP_ENV": "production",
                        "DEBUG": "false",
                        "SECRET_KEY": _STRONG_SECRET,
                        "ENCRYPTION_KEY": _STRONG_KEY,
                        "PHI_ENABLED": "false",
                        "PHI_BYPASS_ACK": ack,
                    }
                )
