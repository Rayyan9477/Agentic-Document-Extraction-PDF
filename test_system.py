#!/usr/bin/env python3
"""
System Integration Test Script

Tests the complete PDF extraction pipeline including:
1. API module imports
2. Security module
3. Pipeline state management
4. Document processing (mock mode)
"""

import os
import sys
import tempfile
from pathlib import Path

# Ensure we can import from the project
PROJECT_ROOT = Path(__file__).parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

# Set required environment variable for testing
os.environ["JWT_SECRET_KEY"] = "test-secret-key-for-system-testing-must-be-at-least-32-chars-long"


def test_module_imports():
    """Test that all critical modules can be imported."""
    print("\n[1/5] Testing module imports...")

    modules = [
        ("FastAPI App", "src.api.app"),
        ("Auth Routes", "src.api.routes.auth"),
        ("Document Routes", "src.api.routes.documents"),
        ("Health Routes", "src.api.routes.health"),
        ("RBAC Security", "src.security.rbac"),
        ("Encryption", "src.security.encryption"),
        ("Audit Logging", "src.security.audit"),
        ("Pipeline State", "src.pipeline.state"),
        ("LM Client", "src.client.lm_client"),
        ("Config", "src.config"),
    ]

    failed = []
    for name, module in modules:
        try:
            __import__(module)
            print(f"  [OK] {name}")
        except Exception as e:
            print(f"  [FAIL] {name}: {e}")
            failed.append((name, str(e)))

    return len(failed) == 0, failed


def test_security_module():
    """Test security module functionality."""
    print("\n[2/5] Testing security module...")

    with tempfile.TemporaryDirectory() as tmpdir:
        from src.security.rbac import RBACManager, Role, Permission

        # Reset singleton for clean test
        RBACManager.reset_instance()

        # Create manager with isolated storage
        manager = RBACManager(
            secret_key="test-secret-key-for-security-testing-32chars",
            user_storage_path=os.path.join(tmpdir, "users.json"),
            revocation_storage_path=os.path.join(tmpdir, "revoked.json"),
        )

        # Create user
        user = manager.users.create_user(
            username="testadmin",
            email="admin@test.com",
            password="SecureP@ss123!",
            roles=[Role.ADMIN],
        )
        print(f"  [OK] User created: {user.username}")

        # Authenticate
        tokens = manager.authenticate("testadmin", "SecureP@ss123!")
        assert tokens is not None
        print(f"  [OK] Authentication successful")

        # Validate access
        payload = manager.validate_access(tokens.access_token)
        assert payload.username == "testadmin"
        print(f"  [OK] Token validation successful")

        # Check permissions
        payload = manager.validate_access(
            tokens.access_token,
            required_permissions={Permission.DOCUMENT_READ}
        )
        print(f"  [OK] Permission check successful")

        # Token revocation
        manager.tokens.revoke_token(tokens.access_token)
        print(f"  [OK] Token revocation successful")

        # Reset singleton
        RBACManager.reset_instance()

    return True, []


def test_pipeline_state():
    """Test pipeline state management."""
    print("\n[3/5] Testing pipeline state...")

    from src.pipeline.state import (
        create_initial_state,
        update_state,
        add_error,
    )

    # Create initial state
    state = create_initial_state(
        pdf_path="/test/doc.pdf",
    )
    assert "processing_id" in state
    assert "pdf_path" in state
    print(f"  [OK] Initial state created with processing_id: {state.get('processing_id', 'N/A')[:8]}...")

    # Update state
    new_state = update_state(state, {
        "total_pages": 5,
        "status": "processing",
    })
    assert new_state["total_pages"] == 5
    assert new_state["status"] == "processing"
    # Verify original state is unchanged (immutability via deep copy)
    assert state.get("total_pages") != 5 or "total_pages" not in state
    print(f"  [OK] State update preserves immutability")

    # Add error
    error_state = add_error(new_state, "Test error message")
    assert "Test error message" in error_state["errors"]
    print(f"  [OK] Error handling works")

    return True, []


def test_encryption():
    """Test encryption functionality."""
    print("\n[4/5] Testing encryption module...")

    from src.security.encryption import (
        EncryptionService,
        KeyManager,
        EncryptionConfig,
    )

    # Generate key
    key = KeyManager.generate_key()
    assert len(key) == 32
    print(f"  [OK] Key generation successful")

    # Encrypt/decrypt data using EncryptionService with master_key
    service = EncryptionService(master_key=key)

    plaintext = b"Sensitive patient data: John Doe, SSN: 123-45-6789"
    encrypted = service.encrypt(plaintext)
    assert encrypted != plaintext
    print(f"  [OK] Data encryption successful")

    decrypted = service.decrypt(encrypted)
    assert decrypted == plaintext
    print(f"  [OK] Data decryption successful")

    return True, []


def test_api_endpoints():
    """Test API endpoints using TestClient."""
    print("\n[5/5] Testing API endpoints...")

    with tempfile.TemporaryDirectory() as tmpdir:
        # Import after setting up environment
        from fastapi.testclient import TestClient
        from src.api.routes.health import router as health_router
        from fastapi import FastAPI

        # Create minimal test app
        app = FastAPI()
        app.include_router(health_router, prefix="/api/v1")

        client = TestClient(app)

        # Test health endpoint
        response = client.get("/api/v1/health")
        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        print(f"  [OK] Health endpoint: {data.get('status', 'unknown')}")

        # Test ready endpoint
        response = client.get("/api/v1/health/ready")
        assert response.status_code in [200, 503]  # May not be ready without full setup
        print(f"  [OK] Ready endpoint: status {response.status_code}")

        # Test live endpoint
        response = client.get("/api/v1/health/live")
        assert response.status_code == 200
        print(f"  [OK] Live endpoint: status {response.status_code}")

    return True, []


def main():
    """Run all system tests."""
    print("=" * 60)
    print("PDF Document Extraction System - Integration Tests")
    print("=" * 60)

    tests = [
        ("Module Imports", test_module_imports),
        ("Security Module", test_security_module),
        ("Pipeline State", test_pipeline_state),
        ("Encryption", test_encryption),
        ("API Endpoints", test_api_endpoints),
    ]

    results = []
    for name, test_func in tests:
        try:
            success, errors = test_func()
            results.append((name, success, errors))
        except Exception as e:
            import traceback
            print(f"  [ERROR] Test failed with exception: {e}")
            traceback.print_exc()
            results.append((name, False, [str(e)]))

    # Summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)

    passed = sum(1 for _, success, _ in results if success)
    failed = len(results) - passed

    for name, success, errors in results:
        status = "[PASS]" if success else "[FAIL]"
        print(f"  {status} {name}")
        if errors:
            for err in errors[:3]:  # Limit to 3 errors per test
                print(f"         - {err}")

    print()
    print(f"Results: {passed}/{len(results)} tests passed")

    if failed > 0:
        print(f"\n{failed} test(s) failed!")
        return 1
    else:
        print("\nAll tests passed!")
        return 0


if __name__ == "__main__":
    sys.exit(main())
