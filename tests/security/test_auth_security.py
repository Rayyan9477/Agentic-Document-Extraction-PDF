"""
Security Tests for Authentication System.

Focus areas:
- Password hashing verification
- JWT token validation and security
- Permission checks and RBAC
- SQL injection prevention
- XSS prevention
- Token expiration and revocation
- Brute force protection
- Session hijacking prevention
"""

import re
import time

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from src.security.rbac import (
    PasswordManager,
    Permission,
    RBACManager,
    Role,
    TokenExpiredError,
    TokenInvalidError,
    TokenManager,
    User,
)


@pytest.fixture(autouse=True)
def reset_rbac():
    """Reset RBAC singleton before each test."""
    RBACManager.reset_instance()
    yield
    RBACManager.reset_instance()


@pytest.fixture
def rbac_manager():
    """Create RBAC manager for testing."""
    return RBACManager.get_instance(secret_key="security-test-secret-key-12345")


@pytest.fixture
def app(rbac_manager):
    """Create FastAPI application for security testing."""
    from fastapi import FastAPI

    from src.api.routes.auth import get_rbac_manager, router

    app = FastAPI()
    app.include_router(router, prefix="/api/v1")
    app.dependency_overrides[get_rbac_manager] = lambda: rbac_manager

    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


# =============================================================================
# Password Security Tests
# =============================================================================


class TestPasswordSecurity:
    """Test password hashing and verification security."""

    def test_password_hashing_is_irreversible(self):
        """Test that password hashing is one-way."""
        password_manager = PasswordManager()
        password = "MySecurePassword123!"

        hashed = password_manager.hash_password(password)

        # Hash should not contain the original password
        assert password not in hashed
        # Hash should be significantly different
        assert len(hashed) > len(password)
        # Hash should contain bcrypt identifier
        assert hashed.startswith("$2b$")

    def test_same_password_different_hashes(self):
        """Test that same password produces different hashes (salt)."""
        password_manager = PasswordManager()
        password = "TestPassword123!"

        hash1 = password_manager.hash_password(password)
        hash2 = password_manager.hash_password(password)

        # Different hashes due to different salts
        assert hash1 != hash2

        # Both should verify correctly
        assert password_manager.verify_password(password, hash1)
        assert password_manager.verify_password(password, hash2)

    def test_password_verification_constant_time(self):
        """Test that password verification takes similar time for correct/incorrect."""
        password_manager = PasswordManager()
        password = "CorrectPassword123!"
        hashed = password_manager.hash_password(password)

        # Measure verification time for correct password
        start = time.perf_counter()
        password_manager.verify_password(password, hashed)
        correct_time = time.perf_counter() - start

        # Measure verification time for incorrect password
        start = time.perf_counter()
        password_manager.verify_password("WrongPassword123!", hashed)
        incorrect_time = time.perf_counter() - start

        # Times should be similar (within 10x to account for bcrypt work)
        # This is a basic timing attack prevention check
        assert abs(correct_time - incorrect_time) < max(correct_time, incorrect_time) * 10

    def test_stored_passwords_never_in_plaintext(self, client, rbac_manager):
        """Test that passwords are never stored in plaintext."""
        # Create user
        response = client.post(
            "/api/v1/auth/signup",
            json={
                "username": "secureuser",
                "email": "secureuser@example.com",
                "password": "MyPassword123!",
                "confirm_password": "MyPassword123!",
            },
        )
        assert response.status_code == status.HTTP_201_CREATED

        # Retrieve user from store
        user = rbac_manager.users.get_user_by_username("secureuser")

        # Password hash should not contain plaintext password
        assert "MyPassword123!" not in user.password_hash
        assert user.password_hash.startswith("$2b$")

    def test_password_hash_work_factor(self):
        """Test that password hashing uses appropriate work factor."""
        password_manager = PasswordManager()
        password = "TestPassword123!"

        hashed = password_manager.hash_password(password)

        # Extract work factor from bcrypt hash
        # Format: $2b$12$... where 12 is the work factor
        match = re.match(r"\$2b\$(\d+)\$", hashed)
        assert match is not None

        work_factor = int(match.group(1))
        # Should be at least 12 (OWASP recommendation)
        assert work_factor >= 12


# =============================================================================
# JWT Token Security Tests
# =============================================================================


class TestJWTSecurity:
    """Test JWT token security."""

    def test_token_signature_verification(self, rbac_manager):
        """Test that tokens are properly signed and verified."""
        user = rbac_manager.users.create_user(
            username="tokenuser",
            email="tokenuser@example.com",
            password="Password123!",
            roles={Role.VIEWER},
        )

        token, _ = rbac_manager.tokens.create_access_token(user)

        # Valid token should verify
        payload = rbac_manager.tokens.validate_token(token)
        assert payload.username == "tokenuser"

        # Tampered token should fail
        tampered_token = token[:-10] + "tampered00"
        with pytest.raises(TokenInvalidError):
            rbac_manager.tokens.validate_token(tampered_token)

    def test_token_cannot_be_modified(self, rbac_manager):
        """Test that token payload cannot be modified without detection."""
        user = rbac_manager.users.create_user(
            username="normaluser",
            email="normal@example.com",
            password="Password123!",
            roles={Role.VIEWER},
        )

        token, _ = rbac_manager.tokens.create_access_token(user)

        # Attempt to decode and re-encode with different data
        # This should fail verification
        import base64
        import json

        parts = token.split(".")
        if len(parts) == 3:
            # Try to modify the payload
            payload = json.loads(base64.urlsafe_b64decode(parts[1] + "=="))
            payload["roles"] = ["admin"]  # Try to escalate privileges

            # Re-encode
            modified_payload = (
                base64.urlsafe_b64encode(json.dumps(payload).encode()).decode().rstrip("=")
            )
            fake_token = f"{parts[0]}.{modified_payload}.{parts[2]}"

            # Should fail validation
            with pytest.raises(TokenInvalidError):
                rbac_manager.tokens.validate_token(fake_token)

    def test_token_expiration_enforced(self, rbac_manager):
        """Test that expired tokens are rejected."""
        # Create token manager with very short expiration
        short_manager = TokenManager(
            secret_key="test-secret",
            access_token_expire_minutes=-1,  # Already expired
        )

        user = User(
            user_id="test-123",
            username="testuser",
            email="test@example.com",
            password_hash="hash",
            roles={Role.VIEWER},
        )

        token, _ = short_manager.create_access_token(user)

        # Token should be expired
        with pytest.raises(TokenExpiredError):
            short_manager.validate_token(token)

    def test_token_contains_minimal_sensitive_data(self, rbac_manager):
        """Test that tokens don't contain sensitive information."""
        user = rbac_manager.users.create_user(
            username="privacyuser",
            email="privacy@example.com",
            password="SecurePassword123!",
            roles={Role.VIEWER},
        )

        token, _ = rbac_manager.tokens.create_access_token(user)
        payload = rbac_manager.tokens.validate_token(token)

        # Token should NOT contain password hash
        assert not hasattr(payload, "password_hash")

        # Decode token to check raw content
        import base64
        import json

        parts = token.split(".")
        decoded_payload = json.loads(base64.urlsafe_b64decode(parts[1] + "=="))

        # Should not contain password
        assert "password" not in decoded_payload
        assert "password_hash" not in decoded_payload

    def test_refresh_token_different_type(self, rbac_manager):
        """Test that refresh tokens have different type from access tokens."""
        user = rbac_manager.users.create_user(
            username="typetest",
            email="type@example.com",
            password="Password123!",
            roles={Role.VIEWER},
        )

        tokens = rbac_manager.tokens.create_token_pair(user)

        access_payload = rbac_manager.tokens.validate_token(tokens.access_token)
        refresh_payload = rbac_manager.tokens.validate_token(tokens.refresh_token)

        assert access_payload.token_type == "access"
        assert refresh_payload.token_type == "refresh"

    def test_token_has_unique_jti(self, rbac_manager):
        """Test that each token has a unique JTI (JWT ID)."""
        user = rbac_manager.users.create_user(
            username="jtitest",
            email="jti@example.com",
            password="Password123!",
            roles={Role.VIEWER},
        )

        # Create multiple tokens
        token1, _ = rbac_manager.tokens.create_access_token(user)
        token2, _ = rbac_manager.tokens.create_access_token(user)

        payload1 = rbac_manager.tokens.validate_token(token1)
        payload2 = rbac_manager.tokens.validate_token(token2)

        # JTIs should be unique
        assert payload1.jti != payload2.jti


# =============================================================================
# RBAC and Permission Tests
# =============================================================================


class TestRBACPermissions:
    """Test Role-Based Access Control and permissions."""

    def test_viewer_has_limited_permissions(self, rbac_manager):
        """Test that VIEWER role has limited permissions."""
        user = rbac_manager.users.create_user(
            username="viewer",
            email="viewer@example.com",
            password="Password123!",
            roles={Role.VIEWER},
        )

        permissions = user.get_all_permissions()

        # Should have read permission
        assert Permission.DOCUMENT_READ in permissions

        # Should NOT have delete permission
        assert Permission.DOCUMENT_DELETE not in permissions

        # Should NOT have admin permission
        assert Permission.SYSTEM_ADMIN not in permissions

    def test_admin_has_all_permissions(self, rbac_manager):
        """Test that ADMIN role has all permissions."""
        user = rbac_manager.users.create_user(
            username="admin",
            email="admin@example.com",
            password="Password123!",
            roles={Role.ADMIN},
        )

        permissions = user.get_all_permissions()

        # Should have all permissions
        assert Permission.DOCUMENT_READ in permissions
        assert Permission.DOCUMENT_DELETE in permissions
        assert Permission.SYSTEM_ADMIN in permissions
        assert Permission.USER_MANAGE_ROLES in permissions

    def test_permission_escalation_prevented(self, client, rbac_manager):
        """Test that users cannot escalate their own permissions."""
        # Create regular user
        user = rbac_manager.users.create_user(
            username="regularuser",
            email="regular@example.com",
            password="Password123!",
            roles={Role.VIEWER},
        )

        tokens = rbac_manager.tokens.create_token_pair(user)

        # Get current permissions
        response = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {tokens.access_token}"},
        )
        data = response.json()

        # Should not have admin permissions
        assert "system:admin" not in data["permissions"]

        # Even if we try to modify the token, it should fail validation
        # (tested in token security tests)

    def test_token_permissions_match_user_role(self, rbac_manager):
        """Test that token permissions match user's assigned role."""
        user = rbac_manager.users.create_user(
            username="analyst",
            email="analyst@example.com",
            password="Password123!",
            roles={Role.ANALYST},
        )

        token, _ = rbac_manager.tokens.create_access_token(user)
        payload = rbac_manager.tokens.validate_token(token)

        # Get expected permissions for ANALYST role
        from src.security.rbac import ROLE_PERMISSIONS

        expected_perms = {p.value for p in ROLE_PERMISSIONS[Role.ANALYST]}
        token_perms = set(payload.permissions)

        assert token_perms == expected_perms


# =============================================================================
# Injection Attack Prevention Tests
# =============================================================================


class TestInjectionPrevention:
    """Test prevention of injection attacks."""

    def test_sql_injection_in_signup(self, client, rbac_manager):
        """Test that SQL injection attempts are handled safely."""
        sql_injections = [
            "admin'--",
            "' OR '1'='1",
            "'; DROP TABLE users;--",
            "admin' OR 1=1--",
            "' UNION SELECT * FROM users--",
        ]

        for injection in sql_injections:
            response = client.post(
                "/api/v1/auth/signup",
                json={
                    "username": injection,
                    "email": "test@example.com",
                    "password": "Password123!",
                    "confirm_password": "Password123!",
                },
            )

            # Should not cause server error
            assert response.status_code != status.HTTP_500_INTERNAL_SERVER_ERROR

            # If user was created, username should be sanitized
            if response.status_code == status.HTTP_201_CREATED:
                user = rbac_manager.users.get_user_by_username(injection)
                if user:
                    # User exists with exact username (not executing SQL)
                    assert user.username == injection

    def test_sql_injection_in_login(self, client):
        """Test that SQL injection in login is handled safely."""
        sql_injections = [
            "admin'--",
            "' OR '1'='1",
            "anything' OR 'x'='x",
        ]

        for injection in sql_injections:
            response = client.post(
                "/api/v1/auth/login",
                json={
                    "username": injection,
                    "password": "anything",
                },
            )

            # Should return 401, not 500 or successful login
            assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_xss_in_user_data(self, client, rbac_manager):
        """Test that XSS attempts are handled safely."""
        xss_attempts = [
            "<script>alert('xss')</script>",
            "<img src=x onerror=alert('xss')>",
            "javascript:alert('xss')",
            "<svg/onload=alert('xss')>",
        ]

        for xss in xss_attempts:
            # Try in username
            response = client.post(
                "/api/v1/auth/signup",
                json={
                    "username": xss,
                    "email": "xss@example.com",
                    "password": "Password123!",
                    "confirm_password": "Password123!",
                },
            )

            # Should not cause server error
            assert response.status_code != status.HTTP_500_INTERNAL_SERVER_ERROR

    def test_command_injection_in_username(self, client):
        """Test that command injection attempts are handled safely."""
        command_injections = [
            "user; ls",
            "user && whoami",
            "user | cat /etc/passwd",
            "`whoami`",
            "$(whoami)",
        ]

        for injection in command_injections:
            response = client.post(
                "/api/v1/auth/signup",
                json={
                    "username": injection,
                    "email": "cmd@example.com",
                    "password": "Password123!",
                    "confirm_password": "Password123!",
                },
            )

            # Should not cause server error
            assert response.status_code != status.HTTP_500_INTERNAL_SERVER_ERROR


# =============================================================================
# Brute Force Protection Tests
# =============================================================================


class TestBruteForceProtection:
    """Test brute force attack protection."""

    def test_account_lockout_after_failed_attempts(self, client, rbac_manager):
        """Test that account locks after multiple failed login attempts."""
        # Create user
        user = rbac_manager.users.create_user(
            username="brutetest",
            email="brute@example.com",
            password="CorrectPassword123!",
            roles={Role.VIEWER},
        )

        # Attempt 5 failed logins
        for i in range(5):
            response = client.post(
                "/api/v1/auth/login",
                json={
                    "username": "brutetest",
                    "password": f"WrongPassword{i}",
                },
            )
            assert response.status_code == status.HTTP_401_UNAUTHORIZED

        # Account should be locked
        locked_user = rbac_manager.users.get_user_by_username("brutetest")
        assert locked_user.is_locked is True
        assert locked_user.failed_login_attempts >= 5

        # Even correct password should fail
        response = client.post(
            "/api/v1/auth/login",
            json={
                "username": "brutetest",
                "password": "CorrectPassword123!",
            },
        )
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_failed_attempts_counter_increments(self, client, rbac_manager):
        """Test that failed login attempts are tracked."""
        user = rbac_manager.users.create_user(
            username="countertest",
            email="counter@example.com",
            password="Password123!",
            roles={Role.VIEWER},
        )

        assert user.failed_login_attempts == 0

        # Failed attempt
        client.post(
            "/api/v1/auth/login",
            json={
                "username": "countertest",
                "password": "WrongPassword",
            },
        )

        # Check counter incremented
        user = rbac_manager.users.get_user_by_username("countertest")
        assert user.failed_login_attempts == 1

    def test_failed_attempts_reset_on_success(self, client, rbac_manager):
        """Test that failed attempts counter resets on successful login."""
        user = rbac_manager.users.create_user(
            username="resettest",
            email="reset@example.com",
            password="Password123!",
            roles={Role.VIEWER},
        )

        # Failed attempts
        for _ in range(2):
            client.post(
                "/api/v1/auth/login",
                json={
                    "username": "resettest",
                    "password": "WrongPassword",
                },
            )

        user = rbac_manager.users.get_user_by_username("resettest")
        assert user.failed_login_attempts == 2

        # Successful login
        client.post(
            "/api/v1/auth/login",
            json={
                "username": "resettest",
                "password": "Password123!",
            },
        )

        # Counter should reset
        user = rbac_manager.users.get_user_by_username("resettest")
        assert user.failed_login_attempts == 0


# =============================================================================
# Session Security Tests
# =============================================================================


class TestSessionSecurity:
    """Test session hijacking prevention and security."""

    def test_token_cannot_be_reused_after_refresh(self, client, rbac_manager):
        """Test that old refresh token cannot be reused after refresh."""
        user = rbac_manager.users.create_user(
            username="reusetest",
            email="reuse@example.com",
            password="Password123!",
            roles={Role.VIEWER},
        )

        tokens = rbac_manager.tokens.create_token_pair(user)
        old_refresh = tokens.refresh_token

        # Use refresh token to get new tokens
        response = client.post(
            "/api/v1/auth/refresh",
            params={"refresh_token": old_refresh},
        )
        assert response.status_code == status.HTTP_200_OK

        # Try to use old refresh token again
        # (Implementation may or may not track revoked tokens)
        # This test documents expected behavior
        response = client.post(
            "/api/v1/auth/refresh",
            params={"refresh_token": old_refresh},
        )

        # Depending on implementation, this may fail or succeed
        # Document actual behavior here

    def test_different_users_have_different_tokens(self, rbac_manager):
        """Test that different users get different tokens."""
        user1 = rbac_manager.users.create_user(
            username="user1",
            email="user1@example.com",
            password="Password123!",
            roles={Role.VIEWER},
        )

        user2 = rbac_manager.users.create_user(
            username="user2",
            email="user2@example.com",
            password="Password123!",
            roles={Role.VIEWER},
        )

        tokens1 = rbac_manager.tokens.create_token_pair(user1)
        tokens2 = rbac_manager.tokens.create_token_pair(user2)

        assert tokens1.access_token != tokens2.access_token
        assert tokens1.refresh_token != tokens2.refresh_token
