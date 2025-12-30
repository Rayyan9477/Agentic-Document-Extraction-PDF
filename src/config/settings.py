"""
Application settings using Pydantic Settings.

Provides type-safe, validated configuration from environment variables
with sensible defaults for the document extraction system.
"""

from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Annotated, Any

from pydantic import (
    AnyHttpUrl,
    DirectoryPath,
    Field,
    SecretStr,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict


class Environment(str, Enum):
    """Application environment enumeration."""

    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"


class LogLevel(str, Enum):
    """Logging level enumeration."""

    DEBUG = "DEBUG"
    INFO = "INFO"
    WARNING = "WARNING"
    ERROR = "ERROR"
    CRITICAL = "CRITICAL"


class LogFormat(str, Enum):
    """Log output format enumeration."""

    JSON = "json"
    CONSOLE = "console"
    TEXT = "text"


class VectorStoreType(str, Enum):
    """Supported vector store types for Mem0."""

    FAISS = "faiss"
    QDRANT = "qdrant"


class ImageOutputFormat(str, Enum):
    """Supported image output formats."""

    PNG = "PNG"
    JPEG = "JPEG"
    TIFF = "TIFF"


class LMStudioSettings(BaseSettings):
    """LM Studio VLM configuration settings."""

    model_config = SettingsConfigDict(
        env_prefix="LM_STUDIO_",
        extra="ignore",
    )

    base_url: AnyHttpUrl = Field(
        default="http://localhost:1234/v1",
        description="LM Studio server base URL",
    )
    model: str = Field(
        default="qwen3-vl",
        description="Model identifier for VLM requests",
    )
    max_tokens: Annotated[int, Field(ge=1, le=32768)] = Field(
        default=4096,
        description="Maximum tokens in VLM response",
    )
    temperature: Annotated[float, Field(ge=0.0, le=2.0)] = Field(
        default=0.1,
        description="Sampling temperature for VLM (lower = more deterministic)",
    )
    timeout: Annotated[int, Field(ge=1, le=600)] = Field(
        default=120,
        description="Request timeout in seconds",
    )
    max_retries: Annotated[int, Field(ge=0, le=10)] = Field(
        default=3,
        description="Maximum retry attempts for failed requests",
    )
    retry_min_wait: Annotated[int, Field(ge=1, le=60)] = Field(
        default=2,
        description="Minimum wait time between retries in seconds",
    )
    retry_max_wait: Annotated[int, Field(ge=1, le=300)] = Field(
        default=30,
        description="Maximum wait time between retries in seconds",
    )

    @property
    def api_url(self) -> str:
        """Get the full API URL for chat completions."""
        return f"{self.base_url}/chat/completions"


class PDFProcessingSettings(BaseSettings):
    """PDF processing configuration settings."""

    model_config = SettingsConfigDict(
        env_prefix="PDF_",
        extra="ignore",
    )

    dpi: Annotated[int, Field(ge=72, le=600)] = Field(
        default=300,
        description="DPI for PDF to image conversion",
    )
    max_pages: Annotated[int, Field(ge=1, le=1000)] = Field(
        default=100,
        description="Maximum pages to process per document",
    )
    max_file_size_mb: Annotated[int, Field(ge=1, le=500)] = Field(
        default=50,
        description="Maximum PDF file size in megabytes",
    )
    temp_dir: Path = Field(
        default=Path("./temp/pdf"),
        description="Temporary directory for PDF processing",
    )
    output_format: ImageOutputFormat = Field(
        default=ImageOutputFormat.PNG,
        description="Output image format",
    )
    enable_enhancement: bool = Field(
        default=True,
        description="Enable image enhancement pipeline",
    )

    @field_validator("temp_dir", mode="before")
    @classmethod
    def create_temp_dir(cls, v: Any) -> Path:
        """Ensure temp directory exists."""
        path = Path(v) if isinstance(v, str) else v
        path.mkdir(parents=True, exist_ok=True)
        return path

    @property
    def max_file_size_bytes(self) -> int:
        """Get maximum file size in bytes."""
        return self.max_file_size_mb * 1024 * 1024


class ImageEnhancementSettings(BaseSettings):
    """Image enhancement configuration settings."""

    model_config = SettingsConfigDict(
        env_prefix="IMAGE_",
        extra="ignore",
    )

    enable_deskew: bool = Field(
        default=True,
        description="Enable automatic deskewing",
    )
    enable_denoise: bool = Field(
        default=True,
        description="Enable noise reduction",
    )
    enable_contrast: bool = Field(
        default=True,
        description="Enable contrast enhancement (CLAHE)",
    )
    clahe_clip_limit: Annotated[float, Field(ge=0.5, le=10.0)] = Field(
        default=2.0,
        description="CLAHE clip limit for contrast enhancement",
    )
    clahe_tile_size: Annotated[int, Field(ge=4, le=32)] = Field(
        default=8,
        description="CLAHE tile grid size",
    )
    denoise_strength: Annotated[int, Field(ge=1, le=30)] = Field(
        default=10,
        description="Denoising filter strength",
    )
    deskew_max_angle: Annotated[float, Field(ge=1.0, le=90.0)] = Field(
        default=45.0,
        description="Maximum angle for deskew detection in degrees",
    )


class Mem0Settings(BaseSettings):
    """Mem0 memory layer configuration settings."""

    model_config = SettingsConfigDict(
        env_prefix="MEM0_",
        extra="ignore",
    )

    enabled: bool = Field(
        default=True,
        description="Enable Mem0 context management",
    )
    vector_store: VectorStoreType = Field(
        default=VectorStoreType.FAISS,
        description="Vector store backend for Mem0",
    )
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence transformer model for embeddings",
    )
    top_k: Annotated[int, Field(ge=1, le=20)] = Field(
        default=5,
        description="Number of relevant memories to retrieve",
    )
    similarity_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=0.7,
        description="Minimum similarity score for memory retrieval",
    )
    data_dir: Path = Field(
        default=Path("./data/memory"),
        description="Directory for persistent memory storage",
    )

    @field_validator("data_dir", mode="before")
    @classmethod
    def create_data_dir(cls, v: Any) -> Path:
        """Ensure data directory exists."""
        path = Path(v) if isinstance(v, str) else v
        path.mkdir(parents=True, exist_ok=True)
        return path


class ExtractionSettings(BaseSettings):
    """Extraction pipeline configuration settings."""

    model_config = SettingsConfigDict(
        env_prefix="EXTRACTION_",
        extra="ignore",
    )

    dual_pass_enabled: bool = Field(
        default=True,
        description="Enable dual-pass extraction for verification",
    )
    max_retries: Annotated[int, Field(ge=0, le=5)] = Field(
        default=2,
        description="Maximum extraction retries for low confidence",
    )
    confidence_auto_accept: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=0.85,
        description="Confidence threshold for automatic acceptance",
    )
    confidence_retry: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=0.50,
        description="Confidence threshold triggering retry",
    )
    confidence_human_review: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=0.50,
        description="Confidence threshold requiring human review",
    )
    batch_size: Annotated[int, Field(ge=1, le=50)] = Field(
        default=5,
        description="Batch size for processing multiple pages",
    )


class ValidationSettings(BaseSettings):
    """Validation configuration settings."""

    model_config = SettingsConfigDict(
        env_prefix="VALIDATION_",
        extra="ignore",
    )

    enable_hallucination_detection: bool = Field(
        default=True,
        description="Enable hallucination pattern detection",
    )
    enable_medical_code_check: bool = Field(
        default=True,
        description="Enable medical code format validation",
    )
    enable_cross_field_rules: bool = Field(
        default=True,
        description="Enable cross-field validation rules",
    )
    strict_mode: bool = Field(
        default=True,
        description="Enable strict validation mode",
    )


class AgentSettings(BaseSettings):
    """Agent optimization and caching settings."""

    model_config = SettingsConfigDict(
        env_prefix="AGENT_",
        extra="ignore",
    )

    cache_max_size: Annotated[int, Field(ge=100, le=10000)] = Field(
        default=1000,
        description="Maximum number of items in agent cache",
    )
    cache_ttl_seconds: Annotated[int, Field(ge=300, le=86400)] = Field(
        default=3600,
        description="Cache TTL in seconds (default 1 hour)",
    )
    metrics_buffer_size: Annotated[int, Field(ge=100, le=10000)] = Field(
        default=1000,
        description="Maximum metrics buffer size before flush",
    )
    alert_latency_threshold_ms: Annotated[int, Field(ge=1000, le=30000)] = Field(
        default=5000,
        description="Latency threshold for alerts in milliseconds",
    )
    max_retry_delay_ms: Annotated[int, Field(ge=1000, le=30000)] = Field(
        default=5000,
        description="Maximum retry delay in milliseconds",
    )


class APISettings(BaseSettings):
    """FastAPI server configuration settings."""

    model_config = SettingsConfigDict(
        env_prefix="API_",
        extra="ignore",
    )

    host: str = Field(
        default="0.0.0.0",
        description="API server host address",
    )
    port: Annotated[int, Field(ge=1, le=65535)] = Field(
        default=8000,
        description="API server port",
    )
    workers: Annotated[int, Field(ge=1, le=32)] = Field(
        default=4,
        description="Number of worker processes",
    )
    reload: bool = Field(
        default=False,
        description="Enable auto-reload for development",
    )
    cors_origins: list[str] = Field(
        default=["http://localhost:8501"],
        description="Allowed CORS origins",
    )


class CelerySettings(BaseSettings):
    """Celery task queue configuration settings."""

    model_config = SettingsConfigDict(
        env_prefix="CELERY_",
        extra="ignore",
    )

    broker_url: str = Field(
        default="redis://localhost:6379/0",
        description="Celery message broker URL",
    )
    result_backend: str = Field(
        default="redis://localhost:6379/1",
        description="Celery result backend URL",
    )
    task_serializer: str = Field(
        default="json",
        description="Task serialization format",
    )
    result_serializer: str = Field(
        default="json",
        description="Result serialization format",
    )
    accept_content: list[str] = Field(
        default=["json"],
        description="Accepted content types",
    )
    task_time_limit: Annotated[int, Field(ge=60, le=3600)] = Field(
        default=600,
        description="Hard task time limit in seconds",
    )
    task_soft_time_limit: Annotated[int, Field(ge=60, le=3600)] = Field(
        default=540,
        description="Soft task time limit in seconds",
    )


class SecuritySettings(BaseSettings):
    """Security configuration settings."""

    model_config = SettingsConfigDict(
        env_prefix="",
        extra="ignore",
    )

    secret_key: SecretStr = Field(
        default=SecretStr("change-this-secret-key-in-production"),
        description="Application secret key",
    )
    encryption_key: SecretStr = Field(
        default=SecretStr("change-this-encryption-key-32b"),
        description="AES-256 encryption key (32 bytes)",
    )
    jwt_algorithm: str = Field(
        default="HS256",
        description="JWT signing algorithm",
    )
    jwt_access_token_expire_minutes: Annotated[int, Field(ge=5, le=1440)] = Field(
        default=30,
        description="Access token expiration time in minutes",
    )
    jwt_refresh_token_expire_days: Annotated[int, Field(ge=1, le=30)] = Field(
        default=7,
        description="Refresh token expiration time in days",
    )


class HIPAASettings(BaseSettings):
    """HIPAA compliance configuration settings."""

    model_config = SettingsConfigDict(
        env_prefix="HIPAA_",
        extra="ignore",
    )

    audit_enabled: bool = Field(
        default=True,
        description="Enable HIPAA audit logging",
    )
    audit_log_path: Path = Field(
        default=Path("./logs/audit"),
        description="Path for audit log files",
    )
    data_retention_days: Annotated[int, Field(ge=1, le=365)] = Field(
        default=90,
        description="Data retention period in days",
    )
    secure_delete_passes: Annotated[int, Field(ge=1, le=7)] = Field(
        default=3,
        description="Number of secure deletion passes",
    )
    phi_masking_enabled: bool = Field(
        default=True,
        description="Enable PHI masking in logs",
    )
    encrypt_at_rest: bool = Field(
        default=True,
        description="Enable encryption for data at rest",
    )

    @field_validator("audit_log_path", mode="before")
    @classmethod
    def create_audit_log_path(cls, v: Any) -> Path:
        """Ensure audit log directory exists."""
        path = Path(v) if isinstance(v, str) else v
        path.mkdir(parents=True, exist_ok=True)
        return path


class ExportSettings(BaseSettings):
    """Export configuration settings."""

    model_config = SettingsConfigDict(
        env_prefix="EXPORT_",
        extra="ignore",
    )

    output_dir: Path = Field(
        default=Path("./output"),
        description="Output directory for exports",
    )
    excel_enabled: bool = Field(
        default=True,
        description="Enable Excel export",
    )
    json_enabled: bool = Field(
        default=True,
        description="Enable JSON export",
    )
    include_metadata: bool = Field(
        default=True,
        description="Include extraction metadata in exports",
    )
    include_confidence_scores: bool = Field(
        default=True,
        description="Include confidence scores in exports",
    )

    @field_validator("output_dir", mode="before")
    @classmethod
    def create_output_dir(cls, v: Any) -> Path:
        """Ensure output directory exists."""
        path = Path(v) if isinstance(v, str) else v
        path.mkdir(parents=True, exist_ok=True)
        return path


class StreamlitSettings(BaseSettings):
    """Streamlit UI configuration settings."""

    model_config = SettingsConfigDict(
        env_prefix="STREAMLIT_",
        extra="ignore",
    )

    server_port: Annotated[int, Field(ge=1, le=65535)] = Field(
        default=8501,
        description="Streamlit server port",
    )
    server_address: str = Field(
        default="0.0.0.0",
        description="Streamlit server address",
    )
    theme_base: str = Field(
        default="light",
        description="Base theme (light/dark)",
    )
    max_upload_size_mb: Annotated[int, Field(ge=1, le=200)] = Field(
        default=50,
        description="Maximum upload file size in MB",
    )


class MonitoringSettings(BaseSettings):
    """Monitoring configuration settings."""

    model_config = SettingsConfigDict(
        env_prefix="",
        extra="ignore",
    )

    prometheus_enabled: bool = Field(
        default=True,
        description="Enable Prometheus metrics",
    )
    prometheus_port: Annotated[int, Field(ge=1, le=65535)] = Field(
        default=9090,
        description="Prometheus metrics port",
    )
    metrics_collection_interval: Annotated[int, Field(ge=5, le=300)] = Field(
        default=15,
        description="Metrics collection interval in seconds",
    )


class DatabaseSettings(BaseSettings):
    """Database configuration settings."""

    model_config = SettingsConfigDict(
        env_prefix="DATABASE_",
        extra="ignore",
    )

    url: str = Field(
        default="sqlite:///./data/extraction.db",
        description="Database connection URL",
    )
    echo: bool = Field(
        default=False,
        description="Enable SQL echo for debugging",
    )


class LoggingSettings(BaseSettings):
    """Logging configuration settings."""

    model_config = SettingsConfigDict(
        env_prefix="LOG_",
        extra="ignore",
    )

    level: LogLevel = Field(
        default=LogLevel.INFO,
        description="Logging level",
    )
    format: LogFormat = Field(
        default=LogFormat.JSON,
        description="Log output format",
    )
    file_path: Path = Field(
        default=Path("./logs/app.log"),
        description="Log file path",
    )
    file_max_size_mb: Annotated[int, Field(ge=1, le=1000)] = Field(
        default=100,
        description="Maximum log file size in MB",
    )
    file_backup_count: Annotated[int, Field(ge=1, le=20)] = Field(
        default=5,
        description="Number of backup log files to keep",
    )
    include_timestamp: bool = Field(
        default=True,
        description="Include timestamp in log entries",
    )
    include_caller: bool = Field(
        default=True,
        description="Include caller information in log entries",
    )

    @field_validator("file_path", mode="before")
    @classmethod
    def create_log_dir(cls, v: Any) -> Path:
        """Ensure log directory exists."""
        path = Path(v) if isinstance(v, str) else v
        path.parent.mkdir(parents=True, exist_ok=True)
        return path


class Settings(BaseSettings):
    """
    Main application settings aggregating all configuration sections.

    Settings are loaded from environment variables with optional .env file support.
    Each section has its own prefix for environment variable naming.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=False,
    )

    # Application metadata
    app_name: str = Field(
        default="doc-extraction-system",
        description="Application name",
    )
    app_version: str = Field(
        default="2.0.0",
        description="Application version",
    )
    app_env: Environment = Field(
        default=Environment.DEVELOPMENT,
        description="Application environment",
    )
    debug: bool = Field(
        default=False,
        description="Enable debug mode",
    )

    # Component settings
    lm_studio: LMStudioSettings = Field(default_factory=LMStudioSettings)
    pdf: PDFProcessingSettings = Field(default_factory=PDFProcessingSettings)
    image: ImageEnhancementSettings = Field(default_factory=ImageEnhancementSettings)
    mem0: Mem0Settings = Field(default_factory=Mem0Settings)
    extraction: ExtractionSettings = Field(default_factory=ExtractionSettings)
    validation: ValidationSettings = Field(default_factory=ValidationSettings)
    agent: AgentSettings = Field(default_factory=AgentSettings)
    api: APISettings = Field(default_factory=APISettings)
    celery: CelerySettings = Field(default_factory=CelerySettings)
    security: SecuritySettings = Field(default_factory=SecuritySettings)
    hipaa: HIPAASettings = Field(default_factory=HIPAASettings)
    export: ExportSettings = Field(default_factory=ExportSettings)
    streamlit: StreamlitSettings = Field(default_factory=StreamlitSettings)
    monitoring: MonitoringSettings = Field(default_factory=MonitoringSettings)
    database: DatabaseSettings = Field(default_factory=DatabaseSettings)
    logging: LoggingSettings = Field(default_factory=LoggingSettings)

    @staticmethod
    def _is_weak_secret(secret: str) -> bool:
        """Check if a secret is weak or uses default patterns."""
        weak_patterns = [
            "change-this",
            "your-secret",
            "your-encryption",
            "changeme",
            "password",
            "secret",
            "default",
            "example",
            "test",
            "dev-",
        ]
        secret_lower = secret.lower()
        return any(pattern in secret_lower for pattern in weak_patterns)

    @staticmethod
    def _has_sufficient_entropy(secret: str, min_length: int = 32) -> bool:
        """Check if secret has sufficient length and character variety."""
        if len(secret) < min_length:
            return False
        # Check for character variety (at least 3 of: upper, lower, digit, special)
        has_upper = any(c.isupper() for c in secret)
        has_lower = any(c.islower() for c in secret)
        has_digit = any(c.isdigit() for c in secret)
        has_special = any(not c.isalnum() for c in secret)
        variety_count = sum([has_upper, has_lower, has_digit, has_special])
        return variety_count >= 3

    @model_validator(mode="after")
    def validate_production_settings(self) -> "Settings":
        """Validate critical settings for production environment."""
        if self.app_env == Environment.PRODUCTION:
            secret_key = self.security.secret_key.get_secret_value()
            encryption_key = self.security.encryption_key.get_secret_value()

            # Validate SECRET_KEY
            if self._is_weak_secret(secret_key):
                raise ValueError(
                    "SECRET_KEY appears to be a default or weak value. "
                    "Use a strong, randomly generated secret in production."
                )
            if not self._has_sufficient_entropy(secret_key, min_length=32):
                raise ValueError(
                    "SECRET_KEY must be at least 32 characters with mixed "
                    "uppercase, lowercase, digits, and special characters."
                )

            # Validate ENCRYPTION_KEY
            if self._is_weak_secret(encryption_key):
                raise ValueError(
                    "ENCRYPTION_KEY appears to be a default or weak value. "
                    "Use a strong, randomly generated key in production."
                )
            if not self._has_sufficient_entropy(encryption_key, min_length=32):
                raise ValueError(
                    "ENCRYPTION_KEY must be at least 32 characters with mixed "
                    "uppercase, lowercase, digits, and special characters."
                )

            # Ensure debug is disabled
            if self.debug:
                raise ValueError("DEBUG must be False in production")

        return self

    @property
    def is_development(self) -> bool:
        """Check if running in development environment."""
        return self.app_env == Environment.DEVELOPMENT

    @property
    def is_production(self) -> bool:
        """Check if running in production environment."""
        return self.app_env == Environment.PRODUCTION

    @property
    def is_testing(self) -> bool:
        """Check if running in testing environment."""
        return self.app_env == Environment.TESTING


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """
    Get cached application settings instance.

    Returns:
        Settings: Application settings singleton.
    """
    return Settings()
