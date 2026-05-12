"""V3 Phase 8 — auth_enabled fail-closed enforcement tests.

Production refuses to boot with ``api.auth_enabled=False`` unless
``AUTH_BYPASS_ACK`` is set to a truthy ack value.
"""

from __future__ import annotations

import os
from typing import Any

import pytest


_STRONG_SECRET = "Q8wXz7vN3pR4mY9bL2kJ6hG5fD1sA0c#Z!%"
_STRONG_KEY = "P9rQ2nM5bV8cK1jH4gF7dS3aL6tY0iO@!#$"


def _construct_settings(env: dict[str, str]) -> Any:
    """Construct a fresh Settings from synthetic env."""
    from src.config import get_settings, settings as settings_module

    original = dict(os.environ)
    try:
        # Pre-ack the PHI gate unless the caller is testing it.
        if "PHI_BYPASS_ACK" not in env and "PHI_ENABLED" not in env:
            os.environ["PHI_BYPASS_ACK"] = "acknowledged"
        for k, v in env.items():
            os.environ[k] = v
        get_settings.cache_clear()
        return settings_module.Settings()
    finally:
        os.environ.clear()
        os.environ.update(original)
        get_settings.cache_clear()


class TestAuthProductionEnforcement:
    def test_dev_with_auth_disabled_is_fine(self) -> None:
        # Dev environment never trips the auth gate.
        settings = _construct_settings(
            {"APP_ENV": "development", "API_AUTH_ENABLED": "false"}
        )
        assert settings.api.auth_enabled is False

    def test_production_auth_off_no_ack_raises(self) -> None:
        with pytest.raises(ValueError) as excinfo:
            _construct_settings(
                {
                    "APP_ENV": "production",
                    "DEBUG": "false",
                    "SECRET_KEY": _STRONG_SECRET,
                    "ENCRYPTION_KEY": _STRONG_KEY,
                    "API_AUTH_ENABLED": "false",
                }
            )
        msg = str(excinfo.value)
        assert "authentication" in msg.lower()
        assert "AUTH_BYPASS_ACK" in msg

    def test_production_auth_on_passes(self) -> None:
        settings = _construct_settings(
            {
                "APP_ENV": "production",
                "DEBUG": "false",
                "SECRET_KEY": _STRONG_SECRET,
                "ENCRYPTION_KEY": _STRONG_KEY,
                "API_AUTH_ENABLED": "true",
            }
        )
        assert settings.api.auth_enabled is True

    def test_production_auth_off_with_ack_passes(self) -> None:
        settings = _construct_settings(
            {
                "APP_ENV": "production",
                "DEBUG": "false",
                "SECRET_KEY": _STRONG_SECRET,
                "ENCRYPTION_KEY": _STRONG_KEY,
                "API_AUTH_ENABLED": "false",
                "AUTH_BYPASS_ACK": "acknowledged",
            }
        )
        assert settings.api.auth_enabled is False

    def test_ack_synonyms(self) -> None:
        for ack in ("1", "true", "TRUE", "yes", "Yes", "acknowledged"):
            settings = _construct_settings(
                {
                    "APP_ENV": "production",
                    "DEBUG": "false",
                    "SECRET_KEY": _STRONG_SECRET,
                    "ENCRYPTION_KEY": _STRONG_KEY,
                    "API_AUTH_ENABLED": "false",
                    "AUTH_BYPASS_ACK": ack,
                }
            )
            assert settings.api.auth_enabled is False, f"ack={ack!r} should pass"

    def test_ack_off_values_rejected(self) -> None:
        for ack in ("no", "off", "0", "false", "lol"):
            with pytest.raises(ValueError):
                _construct_settings(
                    {
                        "APP_ENV": "production",
                        "DEBUG": "false",
                        "SECRET_KEY": _STRONG_SECRET,
                        "ENCRYPTION_KEY": _STRONG_KEY,
                        "API_AUTH_ENABLED": "false",
                        "AUTH_BYPASS_ACK": ack,
                    }
                )
