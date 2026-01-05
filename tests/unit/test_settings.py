"""
Unit tests for Settings classes.

Tests cover:
- AgentSettings configuration and defaults
- Settings validation and constraints
- Environment variable loading
- Production settings validation
- Settings caching behavior
"""

import pytest
from pydantic import ValidationError

from src.config.settings import (
    AgentSettings,
    Environment,
    ExtractionSettings,
    LMStudioSettings,
    PDFProcessingSettings,
    SecuritySettings,
    Settings,
    get_settings,
)


class TestAgentSettings:
    """Tests for AgentSettings class."""

    def test_default_values(self) -> None:
        """Test default configuration values."""
        settings = AgentSettings()

        assert settings.cache_max_size == 1000
        assert settings.cache_ttl_seconds == 3600
        assert settings.metrics_buffer_size == 1000
        assert settings.alert_latency_threshold_ms == 5000
        assert settings.max_retry_delay_ms == 5000

    def test_env_prefix_loading(self, monkeypatch) -> None:
        """Test loading from environment variables with AGENT_ prefix."""
        monkeypatch.setenv("AGENT_CACHE_MAX_SIZE", "2000")
        monkeypatch.setenv("AGENT_CACHE_TTL_SECONDS", "7200")
        monkeypatch.setenv("AGENT_ALERT_LATENCY_THRESHOLD_MS", "10000")

        settings = AgentSettings()

        assert settings.cache_max_size == 2000
        assert settings.cache_ttl_seconds == 7200
        assert settings.alert_latency_threshold_ms == 10000

    def test_validation_constraints_cache_max_size(self) -> None:
        """Test cache_max_size validation constraints (ge=100, le=10000)."""
        # Too small
        with pytest.raises(ValidationError) as exc:
            AgentSettings(cache_max_size=50)
        assert "greater_than_equal" in str(exc.value) or "100" in str(exc.value)

        # Too large
        with pytest.raises(ValidationError) as exc:
            AgentSettings(cache_max_size=20000)
        assert "less_than_equal" in str(exc.value) or "10000" in str(exc.value)

        # Valid boundaries
        assert AgentSettings(cache_max_size=100).cache_max_size == 100
        assert AgentSettings(cache_max_size=10000).cache_max_size == 10000

    def test_validation_constraints_cache_ttl(self) -> None:
        """Test cache_ttl_seconds validation constraints (ge=300, le=86400)."""
        # Too small (less than 5 minutes)
        with pytest.raises(ValidationError):
            AgentSettings(cache_ttl_seconds=100)

        # Too large (more than 1 day)
        with pytest.raises(ValidationError):
            AgentSettings(cache_ttl_seconds=100000)

        # Valid boundaries
        assert AgentSettings(cache_ttl_seconds=300).cache_ttl_seconds == 300
        assert AgentSettings(cache_ttl_seconds=86400).cache_ttl_seconds == 86400

    def test_validation_constraints_metrics_buffer(self) -> None:
        """Test metrics_buffer_size validation constraints."""
        with pytest.raises(ValidationError):
            AgentSettings(metrics_buffer_size=50)

        with pytest.raises(ValidationError):
            AgentSettings(metrics_buffer_size=15000)

    def test_validation_constraints_alert_latency(self) -> None:
        """Test alert_latency_threshold_ms validation constraints."""
        with pytest.raises(ValidationError):
            AgentSettings(alert_latency_threshold_ms=500)

        with pytest.raises(ValidationError):
            AgentSettings(alert_latency_threshold_ms=40000)

    def test_validation_constraints_retry_delay(self) -> None:
        """Test max_retry_delay_ms validation constraints."""
        with pytest.raises(ValidationError):
            AgentSettings(max_retry_delay_ms=500)

        with pytest.raises(ValidationError):
            AgentSettings(max_retry_delay_ms=40000)

    def test_serialization(self) -> None:
        """Test settings can be serialized to dict."""
        settings = AgentSettings(cache_max_size=2000)
        data = settings.model_dump()

        assert data["cache_max_size"] == 2000
        assert "cache_ttl_seconds" in data
        assert "metrics_buffer_size" in data


class TestMainSettings:
    """Tests for main Settings class."""

    def test_agent_settings_integration(self) -> None:
        """Test AgentSettings is properly integrated in main Settings."""
        settings = Settings()

        assert hasattr(settings, "agent")
        assert isinstance(settings.agent, AgentSettings)
        assert settings.agent.cache_max_size == 1000

    def test_is_development_property(self) -> None:
        """Test is_development property."""
        settings = Settings(app_env=Environment.DEVELOPMENT)
        assert settings.is_development is True

        # Use staging instead of production (production requires valid secrets)
        settings = Settings(app_env=Environment.STAGING)
        assert settings.is_development is False

    def test_is_production_property(self) -> None:
        """Test is_production property."""
        # Note: Production validation requires proper secrets, so we test with staging
        settings = Settings(app_env=Environment.STAGING)
        assert settings.is_production is False

        settings = Settings(app_env=Environment.DEVELOPMENT)
        assert settings.is_production is False

    def test_is_testing_property(self) -> None:
        """Test is_testing property."""
        settings = Settings(app_env=Environment.TESTING)
        assert settings.is_testing is True


class TestProductionValidation:
    """Tests for production settings validation."""

    def test_weak_secret_detection(self) -> None:
        """Test _is_weak_secret detects common weak patterns."""
        assert Settings._is_weak_secret("change-this-secret") is True
        assert Settings._is_weak_secret("your-secret-key") is True
        assert Settings._is_weak_secret("myPassword123") is True
        assert Settings._is_weak_secret("default_key") is True
        assert Settings._is_weak_secret("example-key") is True
        assert Settings._is_weak_secret("dev-secret") is True

    def test_strong_secret_accepted(self) -> None:
        """Test _is_weak_secret accepts strong random secrets."""
        strong_secret = "Xk9$mN2@pL5#qR8&vT1*wY4^zU7!aB0"
        assert Settings._is_weak_secret(strong_secret) is False

    def test_entropy_check_length(self) -> None:
        """Test _has_sufficient_entropy checks minimum length."""
        short_secret = "Ab1@"  # Only 4 chars
        assert Settings._has_sufficient_entropy(short_secret, min_length=32) is False

        long_secret = "Ab1@" * 10  # 40 chars, has variety
        assert Settings._has_sufficient_entropy(long_secret, min_length=32) is True

    def test_entropy_check_variety(self) -> None:
        """Test _has_sufficient_entropy checks character variety."""
        # Only lowercase (1 type) - should fail
        no_variety = "a" * 40
        assert Settings._has_sufficient_entropy(no_variety, min_length=32) is False

        # Upper, lower, digit (3 types) - should pass
        good_variety = "Abc123defGHI456jkl789"
        # Pad to 32 chars
        good_variety = good_variety + "Abc123defGHI"
        assert Settings._has_sufficient_entropy(good_variety, min_length=32) is True

    def test_production_rejects_weak_secret_key(self) -> None:
        """Test production validation rejects weak SECRET_KEY."""
        with pytest.raises(ValueError) as exc:
            Settings(
                app_env=Environment.PRODUCTION,
                security=SecuritySettings(
                    secret_key="change-this-secret-key-in-production",
                    encryption_key="Strong3ncryption!K3y@2024#Secure",
                ),
            )
        assert "SECRET_KEY" in str(exc.value)

    def test_production_rejects_weak_encryption_key(self) -> None:
        """Test production validation rejects weak ENCRYPTION_KEY."""
        with pytest.raises(ValueError) as exc:
            Settings(
                app_env=Environment.PRODUCTION,
                security=SecuritySettings(
                    secret_key="Strong!S3cret@K3y#2024$Pr0duct!0n",
                    encryption_key="your-encryption-key-32-bytes-long",
                ),
            )
        assert "ENCRYPTION_KEY" in str(exc.value)

    def test_production_rejects_short_secrets(self) -> None:
        """Test production validation rejects short secrets."""
        with pytest.raises(ValueError) as exc:
            Settings(
                app_env=Environment.PRODUCTION,
                security=SecuritySettings(
                    secret_key="Short!1",  # Too short
                    encryption_key="Strong3ncryption!K3y@2024#Secure",
                ),
            )
        assert "32 characters" in str(exc.value) or "SECRET_KEY" in str(exc.value)

    def test_production_rejects_debug_true(self) -> None:
        """Test production validation rejects DEBUG=True."""
        with pytest.raises(ValueError) as exc:
            Settings(
                app_env=Environment.PRODUCTION,
                debug=True,
                security=SecuritySettings(
                    secret_key="Strong!S3cret@K3y#2024$Pr0duct!0n",
                    encryption_key="Strong3ncryption!K3y@2024#Secure",
                ),
            )
        assert "DEBUG" in str(exc.value)

    def test_production_accepts_strong_configuration(self) -> None:
        """Test production validation accepts properly configured settings."""
        # This should not raise any errors
        settings = Settings(
            app_env=Environment.PRODUCTION,
            debug=False,
            security=SecuritySettings(
                secret_key="Strong!S3cret@K3y#2024$Pr0duct!0n",
                encryption_key="Strong3ncryption!K3y@2024#Secure",
            ),
        )
        assert settings.is_production is True


class TestSettingsCaching:
    """Tests for get_settings() caching behavior."""

    def test_get_settings_returns_cached_instance(self) -> None:
        """Test get_settings() returns same instance (LRU cache)."""
        # Clear cache first
        get_settings.cache_clear()

        settings1 = get_settings()
        settings2 = get_settings()

        assert settings1 is settings2

    def test_get_settings_maxsize_one(self) -> None:
        """Test get_settings() uses lru_cache(maxsize=1)."""
        cache_info = get_settings.cache_info()
        assert cache_info.maxsize == 1

    def test_cache_clear_creates_new_instance(self) -> None:
        """Test cache_clear() allows creating new instance."""
        settings1 = get_settings()
        get_settings.cache_clear()
        settings2 = get_settings()

        # After cache clear, should be different object
        # (though content may be same)
        assert settings1 is not settings2


class TestOtherSettingsClasses:
    """Tests for other settings classes to ensure consistency."""

    def test_lm_studio_settings_defaults(self) -> None:
        """Test LMStudioSettings default values."""
        settings = LMStudioSettings()

        assert str(settings.base_url) == "http://localhost:1234/v1"
        assert settings.model == "qwen3-vl"
        assert settings.max_tokens == 4096
        assert settings.temperature == 0.1
        assert settings.timeout == 120
        assert settings.max_retries == 3

    def test_pdf_processing_defaults(self) -> None:
        """Test PDFProcessingSettings default values."""
        settings = PDFProcessingSettings()

        assert settings.dpi == 300
        assert settings.max_pages == 100
        assert settings.max_file_size_mb == 50

    def test_extraction_settings_confidence_thresholds(self) -> None:
        """Test ExtractionSettings confidence threshold defaults."""
        settings = ExtractionSettings()

        assert settings.confidence_auto_accept == 0.85
        assert settings.confidence_retry == 0.50
        assert settings.confidence_human_review == 0.50

        # Ensure auto_accept > retry/human_review
        assert settings.confidence_auto_accept > settings.confidence_retry
        assert settings.confidence_auto_accept > settings.confidence_human_review


class TestAgentSettingsIntegration:
    """Test agent settings provide sensible defaults for agent operations."""

    def test_cache_settings_reasonable(self) -> None:
        """Test cache settings are reasonable for production."""
        settings = Settings()

        # Cache should be large enough for typical workloads
        assert settings.agent.cache_max_size >= 1000

        # TTL should be at least 1 hour for efficiency
        assert settings.agent.cache_ttl_seconds >= 3600

    def test_alert_threshold_reasonable(self) -> None:
        """Test alert latency threshold is reasonable."""
        settings = Settings()

        # 5 seconds is reasonable for VLM operations
        assert settings.agent.alert_latency_threshold_ms == 5000

    def test_retry_delay_cap_reasonable(self) -> None:
        """Test max retry delay prevents excessive waiting."""
        settings = Settings()

        # Should not wait more than 5 seconds between retries
        assert settings.agent.max_retry_delay_ms <= 5000

    def test_metrics_buffer_sufficient(self) -> None:
        """Test metrics buffer size is sufficient."""
        settings = Settings()

        # Should buffer at least 1000 metrics before flush
        assert settings.agent.metrics_buffer_size >= 1000
