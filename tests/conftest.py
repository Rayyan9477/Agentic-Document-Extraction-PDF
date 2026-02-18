"""
Pytest Configuration and Shared Fixtures for Authentication Tests.

Provides common fixtures and configuration for all test files.
"""

import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


# Add src directory to Python path
repo_root = Path(__file__).parent.parent
sys.path.insert(0, str(repo_root))

from src.security.rbac import RBACManager


# =============================================================================
# Session-scoped Fixtures
# =============================================================================


@pytest.fixture(scope="session")
def test_secret_key() -> str:
    """Secret key for testing."""
    return "test-secret-key-for-pytest-sessions-12345-do-not-use-in-production"


# =============================================================================
# Function-scoped Fixtures
# =============================================================================


@pytest.fixture
def test_data_dir(tmp_path) -> Path:
    """Create temporary directory for test data storage."""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


@pytest.fixture(autouse=True)
def reset_rbac_manager():
    """Automatically reset RBAC manager singleton before each test."""
    RBACManager.reset_instance()
    yield
    RBACManager.reset_instance()


@pytest.fixture
def rbac_manager(test_secret_key, test_data_dir) -> RBACManager:
    """Create fresh RBAC manager for each test with isolated storage."""
    user_storage = str(test_data_dir / "users.json")
    revocation_storage = str(test_data_dir / "revoked_tokens.json")
    return RBACManager.get_instance(
        secret_key=test_secret_key,
        user_storage_path=user_storage,
        revocation_storage_path=revocation_storage,
    )


@pytest.fixture
def test_app(rbac_manager):
    """Create FastAPI test application with auth routes."""
    from fastapi import FastAPI

    from src.api.routes.auth import get_rbac_manager, router

    app = FastAPI(title="Test App")
    app.include_router(router, prefix="/api/v1")

    # Override dependency
    app.dependency_overrides[get_rbac_manager] = lambda: rbac_manager

    return app


@pytest.fixture
def test_client(test_app) -> TestClient:
    """Create FastAPI test client."""
    return TestClient(test_app)


# =============================================================================
# Pytest Configuration
# =============================================================================


def pytest_configure(config):
    """Pytest configuration hook."""
    # Add custom markers
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests")
    config.addinivalue_line("markers", "security: Security tests")
    config.addinivalue_line("markers", "e2e: End-to-end tests")
    config.addinivalue_line("markers", "slow: Slow running tests")


# =============================================================================
# Shared Extraction Fixtures
# =============================================================================


@pytest.fixture
def mock_lm_client():
    """Create a MagicMock LMStudioClient for agent tests."""
    from unittest.mock import MagicMock

    from src.client.lm_client import VisionResponse

    client = MagicMock()
    client.send_vision_request.return_value = VisionResponse(
        content='{"document_type": "medical_claim"}',
        parsed_json={"document_type": "medical_claim"},
        model="qwen3-vl",
        usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
        latency_ms=200,
    )
    client.is_healthy.return_value = True
    return client


@pytest.fixture
def mock_vision_response():
    """Factory fixture returning VisionResponse with configurable content."""
    from src.client.lm_client import VisionResponse

    def _make(
        content: str = '{"result": "ok"}',
        parsed_json: dict | None = None,
        model: str = "qwen3-vl",
        latency_ms: int = 200,
    ) -> VisionResponse:
        if parsed_json is None:
            import json

            try:
                parsed_json = json.loads(content)
            except (json.JSONDecodeError, TypeError):
                parsed_json = None
        return VisionResponse(
            content=content,
            parsed_json=parsed_json,
            model=model,
            usage={
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150,
            },
            latency_ms=latency_ms,
        )

    return _make


@pytest.fixture
def sample_page_image_data() -> str:
    """Base64-encoded 1x1 white PNG for image-based tests."""
    import base64

    # Minimal valid 1x1 white PNG
    png_bytes = (
        b"\x89PNG\r\n\x1a\n\x00\x00\x00\rIHDR\x00\x00\x00\x01"
        b"\x00\x00\x00\x01\x08\x02\x00\x00\x00\x90wS\xde\x00"
        b"\x00\x00\x0cIDATx\x9cc\xf8\x0f\x00\x00\x01\x01\x00"
        b"\x05\x18\xd8N\x00\x00\x00\x00IEND\xaeB`\x82"
    )
    encoded = base64.b64encode(png_bytes).decode("ascii")
    return f"data:image/png;base64,{encoded}"


@pytest.fixture
def make_extraction_state():
    """Factory returning ExtractionState dicts with sensible defaults."""
    from src.pipeline.state import ConfidenceLevel, ExtractionStatus

    def _make(**overrides) -> dict:
        base = {
            "processing_id": "test-proc-id",
            "pdf_path": "/tmp/test.pdf",
            "status": ExtractionStatus.EXTRACTING.value,
            "current_step": "extraction",
            "overall_confidence": 0.85,
            "confidence_level": ConfidenceLevel.HIGH.value,
            "retry_count": 0,
            "errors": [],
            "warnings": [],
            "merged_extraction": {},
            "validation": {},
            "page_images": [],
        }
        base.update(overrides)
        return base

    return _make


def pytest_collection_modifyitems(config, items):
    """Modify test collection."""
    # Add markers based on test location
    for item in items:
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        elif "security" in str(item.fspath):
            item.add_marker(pytest.mark.security)
        elif "e2e" in str(item.fspath):
            item.add_marker(pytest.mark.e2e)
