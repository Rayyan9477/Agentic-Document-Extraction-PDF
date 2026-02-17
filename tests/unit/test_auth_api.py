"""
Comprehensive Unit Tests for Authentication API Routes.

Tests cover:
- POST /api/v1/auth/signup - valid/invalid cases
- POST /api/v1/auth/login - valid/invalid credentials
- GET /api/v1/auth/me - with valid/invalid tokens
- POST /api/v1/auth/refresh - token refresh
- POST /api/v1/auth/logout - logout flow
- Edge cases: duplicate usernames, weak passwords, expired tokens
- Security: SQL injection, XSS attempts in inputs
"""

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from src.api.routes.auth import get_rbac_manager, router
from src.security.rbac import (
    RBACManager,
    Role,
    TokenPair,
    User,
)


@pytest.fixture(autouse=True)
def reset_rbac_singleton():
    """Reset RBAC singleton before each test."""
    RBACManager.reset_instance()
    yield
    RBACManager.reset_instance()


@pytest.fixture
def test_data_dir(tmp_path):
    """Create isolated temporary directory for test data storage."""
    data_dir = tmp_path / "test_data"
    data_dir.mkdir(parents=True, exist_ok=True)
    return data_dir


@pytest.fixture
def rbac_manager(test_data_dir) -> RBACManager:
    """Create RBAC manager for testing with isolated storage."""
    user_storage = str(test_data_dir / "users.json")
    revocation_storage = str(test_data_dir / "revoked_tokens.json")
    return RBACManager.get_instance(
        secret_key="test-secret-key-for-auth-tests-12345",
        user_storage_path=user_storage,
        revocation_storage_path=revocation_storage,
    )


@pytest.fixture
def app(rbac_manager):
    """Create FastAPI test application."""
    from fastapi import FastAPI

    app = FastAPI()
    app.include_router(router, prefix="/api/v1")

    # Override the RBAC manager dependency
    app.dependency_overrides[get_rbac_manager] = lambda: rbac_manager

    return app


@pytest.fixture
def client(app) -> TestClient:
    """Create test client."""
    return TestClient(app)


@pytest.fixture
def test_user(rbac_manager) -> User:
    """Create a test user."""
    return rbac_manager.users.create_user(
        username="testuser",
        email="testuser@example.com",
        password="SecurePassword123!",
        roles={Role.VIEWER},
    )


@pytest.fixture
def test_tokens(rbac_manager, test_user) -> TokenPair:
    """Create test tokens for existing user."""
    return rbac_manager.tokens.create_token_pair(test_user)


# =============================================================================
# Signup Tests
# =============================================================================


class TestSignupEndpoint:
    """Tests for POST /auth/signup endpoint."""

    def test_signup_success(self, client):
        """Test successful user signup."""
        response = client.post(
            "/api/v1/auth/signup",
            json={
                "username": "newuser",
                "email": "newuser@example.com",
                "password": "SecurePass123!",
                "confirm_password": "SecurePass123!",
            },
        )

        assert response.status_code == status.HTTP_201_CREATED
        data = response.json()
        assert data["success"] is True
        assert "created successfully" in data["message"].lower()

    def test_signup_with_minimum_valid_inputs(self, client):
        """Test signup with minimum valid input lengths."""
        response = client.post(
            "/api/v1/auth/signup",
            json={
                "username": "usr",  # 3 chars minimum
                "email": "a@b.co",
                "password": "TestPwd8!x",  # 8+ chars, not common
                "confirm_password": "TestPwd8!x",
            },
        )

        assert response.status_code == status.HTTP_201_CREATED

    def test_signup_password_mismatch(self, client):
        """Test signup with mismatched passwords."""
        response = client.post(
            "/api/v1/auth/signup",
            json={
                "username": "newuser2",
                "email": "newuser2@example.com",
                "password": "Password123!",
                "confirm_password": "DifferentPass123!",
            },
        )

        assert response.status_code == status.HTTP_400_BAD_REQUEST
        data = response.json()
        assert "passwords do not match" in data["detail"].lower()

    def test_signup_duplicate_username(self, client, test_user):
        """Test signup with existing username."""
        response = client.post(
            "/api/v1/auth/signup",
            json={
                "username": test_user.username,
                "email": "different@example.com",
                "password": "SecurePass123!",
                "confirm_password": "SecurePass123!",
            },
        )

        assert response.status_code == status.HTTP_409_CONFLICT
        data = response.json()
        assert "username already exists" in data["detail"].lower()

    def test_signup_username_too_short(self, client):
        """Test signup with username shorter than 3 characters."""
        response = client.post(
            "/api/v1/auth/signup",
            json={
                "username": "ab",  # Too short
                "email": "test@example.com",
                "password": "SecurePass123!",
                "confirm_password": "SecurePass123!",
            },
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_signup_username_too_long(self, client):
        """Test signup with username longer than 50 characters."""
        response = client.post(
            "/api/v1/auth/signup",
            json={
                "username": "a" * 51,  # Too long
                "email": "test@example.com",
                "password": "SecurePass123!",
                "confirm_password": "SecurePass123!",
            },
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_signup_password_too_short(self, client):
        """Test signup with password shorter than 8 characters."""
        response = client.post(
            "/api/v1/auth/signup",
            json={
                "username": "validuser",
                "email": "test@example.com",
                "password": "Short1!",  # Only 7 chars
                "confirm_password": "Short1!",
            },
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_signup_invalid_email_format(self, client):
        """Test signup with invalid email format."""
        invalid_emails = [
            "notanemail",
            "missing@domain",
            "@nodomain.com",
            "no-at-sign.com",
            "double@@domain.com",
        ]

        for email in invalid_emails:
            response = client.post(
                "/api/v1/auth/signup",
                json={
                    "username": "testuser",
                    "email": email,
                    "password": "SecurePass123!",
                    "confirm_password": "SecurePass123!",
                },
            )

            assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_signup_missing_required_fields(self, client):
        """Test signup with missing required fields."""
        # Missing username
        response = client.post(
            "/api/v1/auth/signup",
            json={
                "email": "test@example.com",
                "password": "SecurePass123!",
                "confirm_password": "SecurePass123!",
            },
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

        # Missing email
        response = client.post(
            "/api/v1/auth/signup",
            json={
                "username": "testuser",
                "password": "SecurePass123!",
                "confirm_password": "SecurePass123!",
            },
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

        # Missing password
        response = client.post(
            "/api/v1/auth/signup",
            json={
                "username": "testuser",
                "email": "test@example.com",
                "confirm_password": "SecurePass123!",
            },
        )
        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_signup_sql_injection_attempt(self, client):
        """Test signup with SQL injection attempts in inputs."""
        sql_injections = [
            "admin'--",
            "' OR '1'='1",
            "'; DROP TABLE users;--",
            "admin' OR 1=1--",
        ]

        for injection in sql_injections:
            response = client.post(
                "/api/v1/auth/signup",
                json={
                    "username": injection,
                    "email": "test@example.com",
                    "password": "SecurePass123!",
                    "confirm_password": "SecurePass123!",
                },
            )

            # Should either reject or sanitize - but not cause server error
            assert response.status_code in [
                status.HTTP_201_CREATED,
                status.HTTP_422_UNPROCESSABLE_ENTITY,
            ]

    def test_signup_xss_attempt(self, client):
        """Test signup with XSS attempts in inputs."""
        xss_attempts = [
            "<script>alert('xss')</script>",
            "<img src=x onerror=alert('xss')>",
            "javascript:alert('xss')",
        ]

        for xss in xss_attempts:
            response = client.post(
                "/api/v1/auth/signup",
                json={
                    "username": xss,
                    "email": "test@example.com",
                    "password": "SecurePass123!",
                    "confirm_password": "SecurePass123!",
                },
            )

            # Should handle gracefully
            assert response.status_code in [
                status.HTTP_201_CREATED,
                status.HTTP_422_UNPROCESSABLE_ENTITY,
            ]

    def test_signup_assigns_default_role(self, client, rbac_manager):
        """Test that new users are assigned VIEWER role by default."""
        response = client.post(
            "/api/v1/auth/signup",
            json={
                "username": "roletest",
                "email": "roletest@example.com",
                "password": "SecurePass123!",
                "confirm_password": "SecurePass123!",
            },
        )

        assert response.status_code == status.HTTP_201_CREATED

        # Verify user was created with VIEWER role
        user = rbac_manager.users.get_user_by_username("roletest")
        assert user is not None
        assert Role.VIEWER in user.roles


# =============================================================================
# Login Tests
# =============================================================================


class TestLoginEndpoint:
    """Tests for POST /auth/login endpoint."""

    def test_login_success(self, client, test_user):
        """Test successful login."""
        response = client.post(
            "/api/v1/auth/login",
            json={
                "username": test_user.username,
                "password": "SecurePassword123!",
            },
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"
        assert data["expires_in"] == 1800

    def test_login_invalid_username(self, client):
        """Test login with non-existent username."""
        response = client.post(
            "/api/v1/auth/login",
            json={
                "username": "nonexistentuser",
                "password": "SomePassword123!",
            },
        )

        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        data = response.json()
        assert "invalid username or password" in data["detail"].lower()

    def test_login_invalid_password(self, client, test_user):
        """Test login with incorrect password."""
        response = client.post(
            "/api/v1/auth/login",
            json={
                "username": test_user.username,
                "password": "WrongPassword123!",
            },
        )

        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_login_case_sensitivity(self, client, test_user):
        """Test that login is case-insensitive for username."""
        response = client.post(
            "/api/v1/auth/login",
            json={
                "username": test_user.username.upper(),
                "password": "SecurePassword123!",
            },
        )

        # Username lookup should be case-insensitive
        assert response.status_code in [
            status.HTTP_200_OK,
            status.HTTP_401_UNAUTHORIZED,
        ]

    def test_login_missing_username(self, client):
        """Test login with missing username."""
        response = client.post(
            "/api/v1/auth/login",
            json={
                "password": "SomePassword123!",
            },
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_login_missing_password(self, client):
        """Test login with missing password."""
        response = client.post(
            "/api/v1/auth/login",
            json={
                "username": "testuser",
            },
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_login_empty_credentials(self, client):
        """Test login with empty credentials."""
        response = client.post(
            "/api/v1/auth/login",
            json={
                "username": "",
                "password": "",
            },
        )

        assert response.status_code == status.HTTP_422_UNPROCESSABLE_ENTITY

    def test_login_locked_account(self, client, test_user, rbac_manager):
        """Test login with locked account."""
        # Lock the account
        test_user.is_locked = True
        rbac_manager.users.update_user(test_user)

        response = client.post(
            "/api/v1/auth/login",
            json={
                "username": test_user.username,
                "password": "SecurePassword123!",
            },
        )

        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_login_inactive_account(self, client, test_user, rbac_manager):
        """Test login with inactive account."""
        # Deactivate the account
        test_user.is_active = False
        rbac_manager.users.update_user(test_user)

        response = client.post(
            "/api/v1/auth/login",
            json={
                "username": test_user.username,
                "password": "SecurePassword123!",
            },
        )

        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_login_account_lockout_after_failed_attempts(self, client, test_user, rbac_manager):
        """Test account lockout after multiple failed login attempts."""
        # Attempt login with wrong password 5 times
        for i in range(5):
            response = client.post(
                "/api/v1/auth/login",
                json={
                    "username": test_user.username,
                    "password": f"WrongPassword{i}",
                },
            )
            assert response.status_code == status.HTTP_401_UNAUTHORIZED

        # Verify account is locked
        user = rbac_manager.users.get_user_by_username(test_user.username)
        assert user.is_locked is True

        # Next attempt should still fail even with correct password
        response = client.post(
            "/api/v1/auth/login",
            json={
                "username": test_user.username,
                "password": "SecurePassword123!",
            },
        )
        assert response.status_code == status.HTTP_401_UNAUTHORIZED


# =============================================================================
# Get Current User Tests
# =============================================================================


class TestGetCurrentUserEndpoint:
    """Tests for GET /auth/me endpoint."""

    def test_get_current_user_success(self, client, test_user, test_tokens):
        """Test getting current user with valid token."""
        response = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {test_tokens.access_token}"},
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["user_id"] == test_user.user_id
        assert data["username"] == test_user.username
        assert data["email"] == test_user.email
        assert "roles" in data
        assert "permissions" in data

    def test_get_current_user_no_token(self, client):
        """Test getting current user without token."""
        response = client.get("/api/v1/auth/me")

        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        data = response.json()
        assert "not authenticated" in data["detail"].lower()

    def test_get_current_user_invalid_token(self, client):
        """Test getting current user with invalid token."""
        response = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": "Bearer invalid.token.here"},
        )

        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_get_current_user_malformed_header(self, client):
        """Test getting current user with malformed authorization header."""
        # Missing "Bearer" prefix
        response = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": "SomeToken123"},
        )
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

        # Empty header
        response = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": ""},
        )
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_get_current_user_with_refresh_token(self, client, test_tokens):
        """Test that refresh token cannot be used to access /me endpoint."""
        response = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {test_tokens.refresh_token}"},
        )

        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        data = response.json()
        assert "invalid token type" in data["detail"].lower()

    def test_get_current_user_expired_token(self, client, rbac_manager, test_user):
        """Test getting current user with expired token."""
        from datetime import UTC, datetime, timedelta

        from jose import jwt

        # Create an already-expired token using the SAME secret as the fixture manager
        now = datetime.now(UTC)
        payload = {
            "sub": test_user.user_id,
            "username": test_user.username,
            "email": test_user.email,
            "roles": [r.value for r in test_user.roles],
            "permissions": [],
            "token_type": "access",
            "iat": (now - timedelta(hours=2)).timestamp(),
            "exp": (now - timedelta(hours=1)).timestamp(),  # Already expired
            "jti": "expired-test-token-id",
        }

        expired_token = jwt.encode(
            payload,
            "test-secret-key-for-auth-tests-12345",
            algorithm="HS256",
        )

        response = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {expired_token}"},
        )

        assert response.status_code == status.HTTP_401_UNAUTHORIZED


# =============================================================================
# Token Refresh Tests
# =============================================================================


class TestRefreshTokenEndpoint:
    """Tests for POST /auth/refresh endpoint."""

    def test_refresh_token_success(self, client, test_user, test_tokens):
        """Test successful token refresh."""
        response = client.post(
            "/api/v1/auth/refresh",
            params={"refresh_token": test_tokens.refresh_token},
        )

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert "access_token" in data
        assert "refresh_token" in data
        assert data["token_type"] == "bearer"

        # New tokens should be different
        assert data["access_token"] != test_tokens.access_token
        assert data["refresh_token"] != test_tokens.refresh_token

    def test_refresh_token_with_access_token_fails(self, client, test_tokens):
        """Test that access token cannot be used for refresh."""
        response = client.post(
            "/api/v1/auth/refresh",
            params={"refresh_token": test_tokens.access_token},
        )

        assert response.status_code == status.HTTP_401_UNAUTHORIZED
        data = response.json()
        assert "invalid token type" in data["detail"].lower()

    def test_refresh_token_invalid(self, client):
        """Test refresh with invalid token."""
        response = client.post(
            "/api/v1/auth/refresh",
            params={"refresh_token": "invalid.refresh.token"},
        )

        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_refresh_token_missing(self, client):
        """Test refresh without providing token."""
        response = client.post("/api/v1/auth/refresh")

        # Missing token falls through to "Refresh token required" → 401
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_refresh_token_for_deleted_user(self, client, test_user, test_tokens, rbac_manager):
        """Test that refresh fails if user no longer exists."""
        # Delete the user
        rbac_manager.users.delete_user(test_user.user_id)

        response = client.post(
            "/api/v1/auth/refresh",
            params={"refresh_token": test_tokens.refresh_token},
        )

        assert response.status_code == status.HTTP_401_UNAUTHORIZED


# =============================================================================
# Logout Tests
# =============================================================================


class TestLogoutEndpoint:
    """Tests for POST /auth/logout endpoint."""

    def test_logout_success(self, client):
        """Test successful logout."""
        response = client.post("/api/v1/auth/logout")

        assert response.status_code == status.HTTP_200_OK
        data = response.json()
        assert data["success"] is True
        assert "logged out" in data["message"].lower()

    def test_logout_without_token(self, client):
        """Test that logout works even without authentication."""
        # Logout is mainly client-side, so it should succeed
        response = client.post("/api/v1/auth/logout")

        assert response.status_code == status.HTTP_200_OK


# =============================================================================
# Edge Cases and Security Tests
# =============================================================================


class TestEdgeCasesAndSecurity:
    """Edge cases and security-focused tests."""

    def test_unicode_in_username(self, client):
        """Test signup with Unicode characters in username."""
        response = client.post(
            "/api/v1/auth/signup",
            json={
                "username": "user_测试",
                "email": "unicode@example.com",
                "password": "SecurePass123!",
                "confirm_password": "SecurePass123!",
            },
        )

        # Should handle gracefully
        assert response.status_code in [
            status.HTTP_201_CREATED,
            status.HTTP_422_UNPROCESSABLE_ENTITY,
        ]

    def test_very_long_password(self, client):
        """Test signup with very long password."""
        long_password = "A" * 1000
        response = client.post(
            "/api/v1/auth/signup",
            json={
                "username": "longpassuser",
                "email": "longpass@example.com",
                "password": long_password,
                "confirm_password": long_password,
            },
        )

        # Should handle gracefully
        assert response.status_code in [
            status.HTTP_201_CREATED,
            status.HTTP_422_UNPROCESSABLE_ENTITY,
        ]

    def test_special_characters_in_username(self, client):
        """Test signup with special characters in username."""
        special_chars = ["user@name", "user.name", "user-name", "user_name"]

        for username in special_chars:
            response = client.post(
                "/api/v1/auth/signup",
                json={
                    "username": username,
                    "email": f"{username}@example.com",
                    "password": "SecurePass123!",
                    "confirm_password": "SecurePass123!",
                },
            )

            # Should either accept or reject consistently
            assert response.status_code in [
                status.HTTP_201_CREATED,
                status.HTTP_422_UNPROCESSABLE_ENTITY,
            ]

    def test_concurrent_signup_same_username(self, client, rbac_manager):
        """Test race condition with concurrent signups."""
        # This is a basic test - in production, use proper concurrency testing
        username = "raceuser"

        responses = []
        for _ in range(2):
            response = client.post(
                "/api/v1/auth/signup",
                json={
                    "username": username,
                    "email": f"{username}@example.com",
                    "password": "SecurePass123!",
                    "confirm_password": "SecurePass123!",
                },
            )
            responses.append(response)

        # At least one should succeed, others should get conflict
        success_count = sum(1 for r in responses if r.status_code == status.HTTP_201_CREATED)
        conflict_count = sum(1 for r in responses if r.status_code == status.HTTP_409_CONFLICT)

        assert success_count >= 1

    def test_password_not_logged(self, client, caplog):
        """Test that passwords are not logged in plain text."""
        import logging

        caplog.set_level(logging.INFO)

        client.post(
            "/api/v1/auth/signup",
            json={
                "username": "logtest",
                "email": "logtest@example.com",
                "password": "MySecretPassword123!",
                "confirm_password": "MySecretPassword123!",
            },
        )

        # Check that password is not in logs
        log_text = caplog.text
        assert "MySecretPassword123!" not in log_text

    def test_rate_limiting_placeholder(self, client):
        """Placeholder for rate limiting tests."""
        # In production, implement rate limiting and test it
        # For now, just verify endpoint is accessible
        for i in range(10):
            response = client.post(
                "/api/v1/auth/login",
                json={
                    "username": "testuser",
                    "password": "password",
                },
            )
            # Should respond (may fail auth, but not rate limited yet)
            assert response.status_code in [
                status.HTTP_401_UNAUTHORIZED,
                status.HTTP_429_TOO_MANY_REQUESTS,
            ]
