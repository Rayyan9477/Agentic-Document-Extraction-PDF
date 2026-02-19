"""
Tests for src/config/extraction_config.py — config.json loader for extraction flags.
"""

import json

import pytest
from unittest.mock import patch, mock_open

from src.config.extraction_config import (
    _load_raw,
    get_extraction_config,
    reload_extraction_config,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def clear_cache():
    """Clear the LRU cache before each test."""
    _load_raw.cache_clear()
    yield
    _load_raw.cache_clear()


# ---------------------------------------------------------------------------
# get_extraction_config defaults
# ---------------------------------------------------------------------------


class TestGetExtractionConfig:

    @patch("src.config.extraction_config._CONFIG_PATH")
    def test_defaults_when_no_config_file(self, mock_path):
        mock_path.exists.return_value = False
        cfg = get_extraction_config()
        assert cfg["enable_validation_stage"] is False
        assert cfg["enable_self_correction"] is False
        assert cfg["validation_confidence_threshold"] == 0.85
        assert cfg["enable_consensus_for_critical_fields"] is False
        assert cfg["critical_field_keywords"] is None
        assert cfg["max_fields_per_extraction_call"] == 10
        assert cfg["enable_schema_decomposition"] is True
        assert cfg["enable_synthetic_few_shot_examples"] is False

    @patch("builtins.open", new_callable=mock_open, read_data=json.dumps({
        "enable_validation_stage": True,
        "enable_self_correction": True,
        "validation_confidence_threshold": 0.90,
    }))
    @patch("src.config.extraction_config._CONFIG_PATH")
    def test_overrides_from_config_file(self, mock_path, _mock_file):
        mock_path.exists.return_value = True
        cfg = get_extraction_config()
        assert cfg["enable_validation_stage"] is True
        assert cfg["enable_self_correction"] is True
        assert cfg["validation_confidence_threshold"] == 0.90
        # Non-overridden keys keep defaults
        assert cfg["enable_schema_decomposition"] is True

    @patch("builtins.open", new_callable=mock_open, read_data=json.dumps({}))
    @patch("src.config.extraction_config._CONFIG_PATH")
    def test_empty_config_file_uses_defaults(self, mock_path, _mock_file):
        mock_path.exists.return_value = True
        cfg = get_extraction_config()
        assert cfg["enable_validation_stage"] is False
        assert cfg["max_fields_per_extraction_call"] == 10


# ---------------------------------------------------------------------------
# reload_extraction_config
# ---------------------------------------------------------------------------


class TestReloadExtractionConfig:

    @patch("src.config.extraction_config._CONFIG_PATH")
    def test_reload_clears_cache(self, mock_path):
        mock_path.exists.return_value = False
        # First call caches
        cfg1 = get_extraction_config()
        # Reload clears and re-fetches
        cfg2 = reload_extraction_config()
        assert cfg1 == cfg2

    @patch("builtins.open", new_callable=mock_open, read_data=json.dumps({
        "enable_validation_stage": True,
    }))
    @patch("src.config.extraction_config._CONFIG_PATH")
    def test_reload_picks_up_new_values(self, mock_path, _mock_file):
        mock_path.exists.return_value = True
        cfg = reload_extraction_config()
        assert cfg["enable_validation_stage"] is True


# ---------------------------------------------------------------------------
# _load_raw caching behavior
# ---------------------------------------------------------------------------


class TestLoadRawCaching:

    @patch("src.config.extraction_config._CONFIG_PATH")
    def test_cached_after_first_call(self, mock_path):
        mock_path.exists.return_value = False
        r1 = _load_raw()
        r2 = _load_raw()
        # exists() should only be called once due to caching
        assert mock_path.exists.call_count == 1

    @patch("src.config.extraction_config._CONFIG_PATH")
    def test_returns_empty_dict_when_missing(self, mock_path):
        mock_path.exists.return_value = False
        assert _load_raw() == {}
