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


class VLMBackendName(str, Enum):
    """Selector for the VLM backend (Phase 0 + Phase K).

    All backends are equally first-class. The choice is operational:

    * ``LM_STUDIO`` — the current shipping default. Single-binary,
      easier ops, ``response_format=json_schema`` constrained decoding.
    * ``VLLM`` — production-grade. XGrammar guided decoding, real
      tensor parallelism, dual-instance heterogeneous extraction.
    * ``GEMMA`` — Gemma 4 26B-A4B-it via LM Studio at a dedicated port.
      Native function-calling for medical-code validators.
      ~17 GB VRAM at Q4_K_M; runs on a 24+ GB GPU.
    """

    LM_STUDIO = "lm_studio"
    VLLM = "vllm"
    GEMMA = "gemma"


class VLMMode(str, Enum):
    """Operational mode of the VLM stack (Phase 0).

    * ``LITE`` — single VLM, no Critic, no dual-pass. Resource-constrained.
    * ``STANDARD`` — dual-VLM extraction; Critic invoked on disagreement.
    * ``HARD`` — dual-VLM + Critic always + bbox round-trip on every
      flagged field.
    """

    LITE = "lite"
    STANDARD = "standard"
    HARD = "hard"


class LMStudioBackendSettings(BaseSettings):
    """Backend-specific settings for LM Studio in the V3 dual-VLM topology.

    Distinct from the legacy :class:`LMStudioSettings` so the existing
    deployment-wide LM Studio configuration is untouched. The factory
    in ``src/client/backends/factory.py`` reads from this block first,
    falling back to ``LMStudioSettings`` when ``primary_url`` /
    ``primary_model`` are blank — preserving zero-config behaviour for
    today's installs.
    """

    model_config = SettingsConfigDict(
        env_prefix="VLM_LM_STUDIO_",
        extra="ignore",
    )

    primary_url: str = Field(
        default="",
        description="Primary LM Studio endpoint. Blank → fall back to LMStudioSettings.base_url.",
    )
    primary_model: str = Field(
        default="",
        description="Primary LM Studio model. Blank → fall back to LMStudioSettings.model.",
    )
    secondary_url: str | None = Field(
        default=None,
        description=(
            "Secondary LM Studio endpoint URL for dual-VLM operation. "
            "Required when dual_mode=dual_instance."
        ),
    )
    secondary_model: str | None = Field(
        default=None,
        description="Secondary LM Studio model identifier.",
    )
    dual_mode: str = Field(
        default="single_only",
        description=(
            "How LM Studio handles the secondary role: "
            "'dual_instance' (real heterogeneous, requires secondary_url+model), "
            "'jit_swap' (single instance, swaps models — slow, dev-only), "
            "'single_only' (collapse secondary to primary; Lite-mode behaviour)."
        ),
    )

    @field_validator("dual_mode")
    @classmethod
    def _validate_dual_mode(cls, v: str) -> str:
        valid = {"dual_instance", "jit_swap", "single_only"}
        if v not in valid:
            raise ValueError(f"dual_mode must be one of {valid}, got {v!r}")
        return v


class VLLMBackendSettings(BaseSettings):
    """vLLM backend configuration (Phase 0)."""

    model_config = SettingsConfigDict(
        env_prefix="VLLM_",
        extra="ignore",
    )

    primary_url: str = Field(
        default="http://localhost:8001/v1",
        description="Primary vLLM OpenAI-compat endpoint.",
    )
    primary_model: str = Field(
        default="Qwen/Qwen3.6-27B-VL-Instruct",
        description="Primary VLM model identifier (HF or local path).",
    )
    secondary_url: str | None = Field(
        default=None,
        description="Secondary vLLM endpoint for dual-VLM extraction.",
    )
    secondary_model: str | None = Field(
        default=None,
        description="Secondary VLM model identifier.",
    )
    guided_decoding_backend: str = Field(
        default="xgrammar",
        description="vLLM guided-decoding backend (xgrammar | outlines).",
    )

    @field_validator("guided_decoding_backend")
    @classmethod
    def _validate_guided_decoding_backend(cls, v: str) -> str:
        valid = {"xgrammar", "outlines"}
        if v not in valid:
            raise ValueError(f"guided_decoding_backend must be one of {valid}, got {v!r}")
        return v


class GemmaBackendSettings(BaseSettings):
    """Gemma 4 backend configuration.

    Targets ``lmstudio-community/gemma-4-26B-A4B-it-GGUF`` served by LM
    Studio at a dedicated port (default 1235, distinct from the
    legacy primary at 1234 so both can coexist). Native function-calling
    is exposed via the OpenAI-compat ``tools`` + ``tool_choice`` body
    fields, which LM Studio 0.3+ forwards verbatim for Gemma 4 GGUFs
    whose chat template advertises tool support.

    All roles (PRIMARY / SECONDARY / CRITIC / LITE) collapse to the same
    endpoint by design — the orchestrator differentiates Pass 1 / Pass 2
    / Critic via prompt frame, not model identity. This keeps the
    operator runbook to one model load.
    """

    model_config = SettingsConfigDict(
        env_prefix="GEMMA_",
        extra="ignore",
    )

    primary_url: str = Field(
        default="http://localhost:1235/v1",
        description=(
            "LM Studio endpoint serving Gemma 4. Defaults to port 1235 "
            "so it can run alongside an existing LM Studio at 1234."
        ),
    )
    primary_model: str = Field(
        default="gemma-4-26b-a4b-it",
        description=(
            "Model identifier as listed by LM Studio. The GGUF community "
            "tag at quant UD-Q4_K_M is the recommended starting point."
        ),
    )
    tool_call_timeout: int = Field(
        default=180,
        ge=10,
        le=600,
        description=(
            "Seconds to wait for a tool-use response. Higher than the "
            "base 120s timeout because tool-forcing adds a decoding "
            "step on top of generation."
        ),
    )
    max_retries: int = Field(
        default=3,
        ge=0,
        le=10,
        description="Retry attempts on transient failures before raising.",
    )
    temperature: float = Field(
        default=0.1,
        ge=0.0,
        le=2.0,
        description="Sampling temperature. Low value for deterministic extraction.",
    )
    register_rcm_tools: bool = Field(
        default=True,
        description=(
            "When True, attach the five medical-code validator tools "
            "(npi_luhn_check, cpt_validate, icd_normalize, sum_reconcile, "
            "validate_date_ordering) to every extraction request so the "
            "model can invoke them mid-extraction. Healthcare mode only."
        ),
    )
    fail_open_on_health: bool = Field(
        default=False,
        description=(
            "When True, ``health()`` returns ``overall_healthy=True`` "
            "even if the underlying endpoint is unreachable. Useful for "
            "demo environments where the LM Studio process boots after "
            "the API. Default False — fail closed."
        ),
    )


class VLMSettings(BaseSettings):
    """Top-level VLM stack settings (Phase 0 + Phase K).

    Selects the backend and mode, and holds the per-backend nested
    settings. Existing ``settings.lm_studio`` continues to drive the
    legacy single-VLM call-sites; the new ``settings.vlm.*`` blocks
    drive the V3 backend abstraction.
    """

    model_config = SettingsConfigDict(
        env_prefix="VLM_",
        extra="ignore",
    )

    backend: VLMBackendName = Field(
        default=VLMBackendName.LM_STUDIO,
        description="Which VLM backend to use (lm_studio | vllm | gemma).",
    )
    mode: VLMMode = Field(
        default=VLMMode.LITE,
        description=(
            "Extraction mode. Lite = single VLM (current default). "
            "Standard = dual-VLM. Hard = dual-VLM + Critic always."
        ),
    )

    lm_studio: LMStudioBackendSettings = Field(default_factory=LMStudioBackendSettings)
    vllm: VLLMBackendSettings = Field(default_factory=VLLMBackendSettings)
    gemma: GemmaBackendSettings = Field(default_factory=GemmaBackendSettings)

    # V3 Phase 7 — process-wide VLM queue-depth gate. ``0`` disables.
    # Production deployments should set this to a value matching the
    # backend's batch-size headroom so a thundering herd of requests
    # cannot drive the GPU into thrash.
    max_concurrent_requests: int = Field(
        default=0,
        ge=0,
        description=(
            "Maximum concurrent VLM requests per Python process. "
            "0 = unbounded (legacy default). Set to N>0 to gate "
            "concurrency at the application level (in addition to "
            "any backend-side limits like vLLM's --max-num-batched-tokens)."
        ),
    )

    @field_validator("backend", mode="before")
    @classmethod
    def _coerce_backend(cls, v: Any) -> Any:
        # Allow case-insensitive env values like ``VLM_BACKEND=LM_Studio``.
        if isinstance(v, str):
            return v.lower()
        return v

    @field_validator("mode", mode="before")
    @classmethod
    def _coerce_mode(cls, v: Any) -> Any:
        if isinstance(v, str):
            return v.lower()
        return v


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


class ExtractionEngine(str, Enum):
    """V3 Phase 2 — extraction-engine selector.

    * ``LEGACY`` — single-VLM dual-pass (current shipping default).
    * ``DUAL_VLM`` — heterogeneous dual-VLM (Qwen primary + Gemma
      secondary), reconciled by ``HeterogeneousReconciler``. Behind a
      flag until shadow-validated against legacy on the eval harness.
    """

    LEGACY = "legacy"
    DUAL_VLM = "dual_vlm"


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

    # === V3 Phase 2 — heterogeneous dual-VLM ===
    engine: ExtractionEngine = Field(
        default=ExtractionEngine.LEGACY,
        description=(
            "Extraction engine. legacy = current single-VLM dual-pass; "
            "dual_vlm = Qwen primary + Gemma secondary + HeterogeneousReconciler. "
            "dual_vlm requires either VLM_BACKEND=vllm with both endpoints up, "
            "or VLM_BACKEND=lm_studio with VLM_LM_STUDIO_DUAL_MODE=dual_instance."
        ),
    )
    # Reconciler tiebreaker thresholds. The defaults track MVP/EXTRACTION.md §4
    # and the eval-harness Day 1 baseline.
    reconciler_bbox_iou_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=0.4,
        description=(
            "Minimum IoU for Pass 1 value to be considered inside Pass 2's "
            "reported bbox region during reconciler tiebreak step 2."
        ),
    )
    reconciler_history_similarity_threshold: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=0.88,
        description=(
            "Minimum FAISS similarity for field-history match to win during "
            "reconciler tiebreak step 5."
        ),
    )
    bbox_roundtrip_low_conf_band: Annotated[
        tuple[float, float], Field(...)
    ] = Field(
        default=(0.5, 0.85),
        description=(
            "Dual-pass similarity band that triggers a bbox round-trip "
            "verification call for the disputed field."
        ),
    )

    # === V3 Phase 3 — Critic agent ===
    critic_enabled: bool = Field(
        default=False,
        description=(
            "Enable the Critic agent (V3 Phase 3). Independent VLM call "
            "after validation that audits whether each extracted value "
            "is actually visible in the page. Adds ~3-5s per document. "
            "Strongest signal in dual-VLM mode (family-rotated against "
            "the consensus of Pass 1 / Pass 2)."
        ),
    )
    critic_min_trust_score: Annotated[float, Field(ge=0.0, le=1.0)] = Field(
        default=0.50,
        description=(
            "Minimum Critic trust_score below which the document routes "
            "to human review even if individual concerns aren't errors."
        ),
    )
    critic_combiner_weights: Annotated[
        tuple[float, float, float], Field(...)
    ] = Field(
        default=(0.50, 0.30, 0.20),
        description=(
            "Weights for the final-confidence combiner: "
            "(dual_pass_agreement, critic_trust, 1 - modality_penalty). "
            "Must sum to 1.0."
        ),
    )

    @field_validator("engine", mode="before")
    @classmethod
    def _coerce_engine(cls, v: Any) -> Any:
        if isinstance(v, str):
            return v.lower()
        return v

    @field_validator("critic_combiner_weights", mode="after")
    @classmethod
    def _validate_combiner_weights(cls, v: tuple[float, float, float]) -> tuple[float, float, float]:
        total = sum(v)
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"critic_combiner_weights must sum to 1.0, got {total} for {v!r}"
            )
        return v


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
        default="0.0.0.0",  # nosec B104 - intentional bind to all interfaces for containerized deployments
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
        default=[
            "http://localhost:8501",
            # Phase K — Next.js dev / production server ports.
            "http://localhost:3000",
            "http://127.0.0.1:3000",
        ],
        description="Allowed CORS origins",
    )
    auth_enabled: bool = Field(
        default=False,
        description=(
            "V3 Phase 8 — install AuthenticationMiddleware. Defaults False "
            "for dev/test convenience. In production environments the "
            "Settings model_validator refuses to boot with this False "
            "unless ``AUTH_BYPASS_ACK`` is explicitly acknowledged."
        ),
    )
    multi_tenant_enabled: bool = Field(
        default=False,
        description=(
            "V3 Phase 8 — when True, install TenantResolverMiddleware "
            "and gate per-tenant FAISS / rate-limit / checkpoint paths. "
            "Single-tenant deployments keep this False (every request "
            "lands on the ``default`` tenant)."
        ),
    )
    default_tenant_id: str = Field(
        default="default",
        description=(
            "Tenant id used when multi-tenancy is disabled OR when an "
            "authenticated request has no tenant_id claim and isn't an "
            "admin override."
        ),
    )
    auth_refresh_query_param_legacy: bool = Field(
        default=False,
        description=(
            "V3 Phase 8.5-A1 — legacy compatibility flag. When True, "
            "POST /auth/refresh continues to accept the refresh token "
            "via the ``?refresh_token=...`` query string in addition to "
            "the JSON body and HttpOnly cookie. Default False: query "
            "strings flow into audit logs and access logs verbatim, so "
            "the new body-only path is fail-closed. Set to True only "
            "for one release while migrating unmigrated API clients."
        ),
    )
    export_receipt_signing_key: str = Field(
        default="",
        description=(
            "Phase K — HMAC-SHA256 shared secret used to sign export "
            "receipts (see src/export/signed_receipt.py). Empty = "
            "receipts are minted but unsigned (still useful for offline "
            "artefact-hash verification). Production deployments should "
            "set this to a 32-byte hex string and rotate alongside the "
            "audit-chain anchor key."
        ),
    )
    export_receipt_signer_key_id: str = Field(
        default="",
        description=(
            "Phase K — operator-chosen identifier for the receipt "
            "signing key (e.g. ``'ops-2026-Q2'``). Stamped into every "
            "receipt so key-rotation events can be traced post-hoc. "
            "Empty = no key id field."
        ),
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
        default="0.0.0.0",  # nosec B104 - intentional bind to all interfaces for containerized deployments
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


class ProvenanceSettings(BaseSettings):
    """V3 Phase 4 — provenance threading configuration.

    The migration from bare-scalar ``merged_extraction`` to
    ``FieldValue``-wrapped ``merged_extraction_v2`` is staged via the
    ``enforce_field_value_wrapper`` flag:

    * ``False`` (default) — the reconciler dual-writes both shapes;
      legacy exporters keep working unchanged; provenance-aware
      exporters opt in by reading ``merged_extraction_v2`` first.
    * ``True`` — the reconciler only writes the wrapper shape; bare
      scalars in ``merged_extraction`` raise ``ProvenanceMissingError``
      when an exporter tries to read them via ``unwrap_value(strict=True)``.

    Flip the flag tenant-by-tenant during rollout. Once every consumer
    is on the wrapper path, set the global default to ``True`` and
    delete the dual-write code in a follow-up cleanup PR.
    """

    model_config = SettingsConfigDict(
        env_prefix="PROVENANCE_",
        extra="ignore",
    )

    enforce_field_value_wrapper: bool = Field(
        default=False,
        description=(
            "When True, the reconciler stops writing the legacy "
            "``merged_extraction`` (scalar dict) shape and writes only "
            "``merged_extraction_v2`` (FieldValue dict). Exporters that "
            "haven't migrated will see empty merged_extraction and "
            "must use ``unwrap_value()`` on the v2 path."
        ),
    )
    fhir_extension_url: str = Field(
        default="urn:veridoc:provenance:1.0",
        description=(
            "FHIR R4 extension URL stamped on every emitted resource. "
            "Use ``urn:chartsend:provenance:1.0`` only inside the "
            "Medical-RCM profile's C-CDA emitter (see PROVENANCE.md §5)."
        ),
    )


class ProfileSettings(BaseSettings):
    """V3 Phase 5 — document profile configuration.

    A *profile* is the orthogonal axis to *modality*: profiles describe
    **what kind of document** it is (generic, medical-RCM, finance, …),
    modalities describe **how it looks** (printed, fax, handwritten, …).

    Profile detection runs as a pure-text post-step in the analyzer
    (no extra VLM call). When ``detection_enabled=True`` and the
    detection score clears the profile's ``confidence_floor``, the
    selected profile drives:

    * Prompt fragment injection (medical-RCM gets CPT/ICD reminders).
    * Schema overlay (medical-RCM re-introduces healthcare fields).
    * Validator pack mode (blocking vs advisory per profile).
    * Available export emitters (C-CDA / X12N 275 only on medical-RCM).

    The setting is on by default. Operators can disable it to force
    every doc to ``generic-document`` while debugging a misdetection.
    """

    model_config = SettingsConfigDict(
        env_prefix="PROFILE_",
        extra="ignore",
    )

    detection_enabled: bool = Field(
        default=True,
        description=(
            "Enable profile auto-detection in the analyzer. When False, "
            "every document falls back to 'generic-document'."
        ),
    )
    default_profile: str = Field(
        default="generic-document",
        description="Profile name used when detection is disabled or fails.",
    )
    apply_overlay: bool = Field(
        default=True,
        description=(
            "When True, the analyzer applies the detected profile's "
            "schema overlay (e.g. medical-RCM re-adds HEALTHCARE_FIELDS) "
            "before extraction. Disable to test base schemas in isolation."
        ),
    )


class CalibrationSettings(BaseSettings):
    """Confidence calibration configuration settings."""

    model_config = SettingsConfigDict(
        env_prefix="CALIBRATION_",
        extra="ignore",
    )

    # WS-2: enabled by default. Calibrator gracefully degrades to a pass-through
    # (returns raw confidence) until at least 3 calibration samples are
    # collected, so flipping this on has no behavioural impact for fresh
    # deployments but produces meaningful calibrated scores once a golden
    # dataset has been processed.
    enabled: bool = Field(
        default=True,
        description="Enable confidence score calibration",
    )
    method: str = Field(
        default="auto",
        description="Calibration method: auto, platt, isotonic, or linear",
    )
    storage_path: Path = Field(
        default=Path("./data/calibration"),
        description="Directory for calibration model persistence",
    )

    @field_validator("storage_path", mode="before")
    @classmethod
    def create_storage_dir(cls, v: Any) -> Path:
        """Ensure storage directory exists."""
        path = Path(v) if isinstance(v, str) else v
        path.mkdir(parents=True, exist_ok=True)
        return path


class ObservabilitySettings(BaseSettings):
    """WS-7: AI observability sinks (Phoenix + PostHog).

    Both sinks are off by default. When enabled, the
    ``ObservabilityDispatcher`` fans out spans / events to whichever
    sinks are both flagged on AND have their SDK installed (via the
    ``[observability]`` extra). Failure to load a sink degrades to no-op
    gracefully — observability never blocks extraction.
    """

    model_config = SettingsConfigDict(
        env_prefix="OBSERVABILITY_",
        extra="ignore",
    )

    phoenix_enabled: bool = Field(
        default=False,
        description="Enable Arize Phoenix (OpenInference) LLM tracing.",
    )
    phoenix_endpoint: str = Field(
        default="http://localhost:6006",
        description="Phoenix collector endpoint. Self-hosted by default.",
    )
    phoenix_project_name: str = Field(
        default="doc-extraction",
        description="Project name shown in the Phoenix UI.",
    )

    posthog_enabled: bool = Field(
        default=False,
        description="Enable PostHog product-analytics event capture.",
    )
    posthog_api_key: str = Field(
        default="",
        description="PostHog project API key. Required when enabled.",
    )
    posthog_host: str = Field(
        default="https://us.posthog.com",
        description="PostHog instance URL (override for self-hosted).",
    )


class PHISettings(BaseSettings):
    """WS-6: opt-in PHI redaction configuration.

    PHI mode is **off by default**. When enabled, every string field in
    extracted records is routed through ``src.security.phi_redactor.PHIRedactor``
    after extraction completes. The redactor uses
    ``openai/privacy-filter`` (HuggingFace, Apache 2.0) for ML-grade
    BIOES-tagged token classification, with a regex fallback for
    air-gapped deployments where the model is not vendored.

    Environment variables (``PHI_*`` prefix) override the defaults below.
    """

    model_config = SettingsConfigDict(
        env_prefix="PHI_",
        extra="ignore",
    )

    enabled: bool = Field(
        default=False,
        description=(
            "Master flag. When False, PHI redaction never runs even if a "
            "request sets phi_mode=True (callers must opt in via settings "
            "AND request)."
        ),
    )
    model_id: str = Field(
        default="openai/privacy-filter",
        description="HuggingFace model identifier for ML-layer redaction.",
    )
    local_only: bool = Field(
        default=True,
        description=(
            "When True, the transformers loader is forced to use only "
            "locally-cached weights (no network call). Required for "
            "HIPAA-style air-gapped deployments."
        ),
    )
    fallback_to_regex: bool = Field(
        default=True,
        description=(
            "When the ML layer cannot be loaded (air-gapped without "
            "pre-vendored weights, missing optional [phi] extra), fall "
            "back to the regex redactor in src/security/phi_mask.py "
            "rather than passing PHI through unchanged."
        ),
    )


class ModelRoutingSettings(BaseSettings):
    """Multi-model routing configuration settings."""

    model_config = SettingsConfigDict(
        env_prefix="MODEL_ROUTING_",
        extra="ignore",
    )

    enabled: bool = Field(
        default=False,
        description="Enable task-based multi-model routing",
    )
    default_model: str = Field(
        default="qwen3-vl",
        description="Default model ID when no task-specific routing applies",
    )
    default_base_url: str = Field(
        default="http://localhost:1234/v1",
        description="Default base URL for model endpoints",
    )
    task_models: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping of task name to model ID (e.g. layout_analysis: florence-2)",
    )


class WebhookSettings(BaseSettings):
    """Webhook subscription and delivery settings."""

    model_config = SettingsConfigDict(
        env_prefix="WEBHOOK_",
        extra="ignore",
    )

    store_path: Path = Field(
        default=Path("./data/webhooks.json"),
        description="JSON file for webhook subscription persistence",
    )
    max_retries: Annotated[int, Field(ge=0, le=10)] = Field(
        default=3,
        description="Maximum delivery retry attempts per subscription",
    )
    timeout_seconds: Annotated[int, Field(ge=1, le=120)] = Field(
        default=30,
        description="HTTP timeout for webhook delivery in seconds",
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
    # VLM backend abstraction (Phase 0) — selects between LM Studio and vLLM,
    # and carries dual-instance settings. ``lm_studio`` (legacy) continues
    # to drive default URL/model when ``vlm.lm_studio.primary_url`` is blank.
    vlm: VLMSettings = Field(default_factory=VLMSettings)
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
    calibration: CalibrationSettings = Field(default_factory=CalibrationSettings)
    provenance: ProvenanceSettings = Field(default_factory=ProvenanceSettings)
    profile: ProfileSettings = Field(default_factory=ProfileSettings)
    model_routing: ModelRoutingSettings = Field(default_factory=ModelRoutingSettings)
    phi: PHISettings = Field(default_factory=PHISettings)
    observability: ObservabilitySettings = Field(default_factory=ObservabilitySettings)
    webhook: WebhookSettings = Field(default_factory=WebhookSettings)

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

            # V3 Phase 8 — Auth fail-closed in production.
            # ``api.auth_enabled`` defaults False for dev convenience.
            # Production refuses to boot with auth off unless the
            # operator has explicitly acknowledged the bypass via
            # ``AUTH_BYPASS_ACK``. Mirrors the PHI bypass pattern so
            # auth-off deployments are deliberate decisions logged in
            # config, not forgotten flags.
            if not self.api.auth_enabled:
                import os as _os

                auth_ack = _os.environ.get("AUTH_BYPASS_ACK", "").strip().lower()
                if auth_ack not in {"1", "true", "yes", "acknowledged"}:
                    raise ValueError(
                        "API authentication is disabled in production "
                        "(API_AUTH_ENABLED=false). Set "
                        "API_AUTH_ENABLED=true OR set "
                        "AUTH_BYPASS_ACK=acknowledged to confirm this "
                        "deployment intentionally ships unauthenticated. "
                        "Refusing to boot."
                    )

            # V3 Phase 7 — PHI mode production enforcement.
            # When environment == production, we refuse to boot with PHI
            # mode disabled unless the operator has explicitly
            # acknowledged the bypass via the ``PHI_BYPASS_ACK`` env
            # var. This protects against the most common production
            # incident: a config oversight that ships PHI in the clear.
            #
            # Rationale: HIPAA exposure from disabled PHI redaction is
            # a high-blast-radius mistake; making the bypass require
            # an explicit ack ensures it's a deliberate decision logged
            # in deployment config, not a forgotten flag.
            if not self.phi.enabled:
                import os as _os

                bypass_ack = _os.environ.get("PHI_BYPASS_ACK", "").strip().lower()
                if bypass_ack not in {"1", "true", "yes", "acknowledged"}:
                    raise ValueError(
                        "PHI redaction is disabled in production "
                        "(PHI_ENABLED=false). Set PHI_ENABLED=true OR "
                        "set PHI_BYPASS_ACK=acknowledged to confirm "
                        "this deployment intentionally ships without "
                        "PHI redaction. Refusing to boot."
                    )

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
