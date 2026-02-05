"""
Extraction pipeline configuration loaded from config.json.

Provides a cached, validated loader for the extraction feature flags
(validation, self-correction, consensus) used by the multi-record pipeline.
"""

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from src.config.logging_config import get_logger

logger = get_logger(__name__)

_CONFIG_PATH = Path(__file__).resolve().parents[2] / "config.json"


@lru_cache(maxsize=1)
def _load_raw() -> dict[str, Any]:
    """Load config.json once and cache it for the process lifetime."""
    if _CONFIG_PATH.exists():
        with open(_CONFIG_PATH) as f:
            return json.load(f)
    logger.warning("config_json_not_found", path=str(_CONFIG_PATH))
    return {}


def get_extraction_config() -> dict[str, Any]:
    """Return extraction pipeline configuration from config.json.

    Returns a plain dict with validated defaults for all extraction flags.
    """
    raw = _load_raw()
    return {
        "enable_validation_stage": raw.get("enable_validation_stage", False),
        "enable_self_correction": raw.get("enable_self_correction", False),
        "validation_confidence_threshold": raw.get("validation_confidence_threshold", 0.85),
        "enable_consensus_for_critical_fields": raw.get("enable_consensus_for_critical_fields", False),
        "critical_field_keywords": raw.get("critical_field_keywords", None),
    }


def reload_extraction_config() -> dict[str, Any]:
    """Force-reload config.json (clears cache). Useful after config changes."""
    _load_raw.cache_clear()
    return get_extraction_config()
