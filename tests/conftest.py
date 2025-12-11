"""
Pytest Configuration and Shared Fixtures for Authentication Tests.

Provides common fixtures and configuration for all test files.
"""

import pytest
import os
import sys
import tempfile
import shutil
from pathlib import Path
from typing import Generator
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
    from src.api.routes.auth import router, get_rbac_manager

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
