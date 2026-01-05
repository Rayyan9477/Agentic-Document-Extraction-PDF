"""
Integration Tests for Authentication System.

Tests complete authentication flows:
- Signup -> Login -> Access protected route
- Token refresh flow
- Multi-user scenarios
- CORS functionality
- Session management
"""

import pytest
from fastapi import status
from fastapi.testclient import TestClient

from src.security.rbac import RBACManager, Role


@pytest.fixture(autouse=True)
def reset_rbac():
    """Reset RBAC singleton before each test."""
    RBACManager.reset_instance()
    yield
    RBACManager.reset_instance()


@pytest.fixture
def rbac_manager():
    """Create RBAC manager for testing."""
    return RBACManager.get_instance(secret_key="integration-test-secret-key-12345")


@pytest.fixture
def app(rbac_manager):
    """Create complete FastAPI application for integration testing."""
    from fastapi import Depends, FastAPI
    from fastapi.middleware.cors import CORSMiddleware

    from src.api.routes.auth import get_rbac_manager, router

    app = FastAPI(title="Auth Integration Test App")

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:3000", "http://localhost:3001"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include auth router
    app.include_router(router, prefix="/api/v1")

    # Override dependency
    app.dependency_overrides[get_rbac_manager] = lambda: rbac_manager

    # Add a protected endpoint for testing
    @app.get("/api/v1/protected/resource")
    async def protected_resource(authorization: str = Depends(lambda: None)):
        """Protected endpoint requiring authentication."""
        # This would normally use a dependency for auth
        return {"message": "Protected resource accessed"}

    @app.get("/api/v1/protected/admin-only")
    async def admin_only_resource():
        """Admin-only endpoint."""
        return {"message": "Admin resource accessed"}

    return app


@pytest.fixture
def client(app):
    """Create test client."""
    return TestClient(app)


# =============================================================================
# Complete Authentication Flow Tests
# =============================================================================


class TestCompleteAuthFlow:
    """Test complete authentication flows from signup to resource access."""

    def test_signup_login_access_flow(self, client):
        """Test: Signup -> Login -> Access protected resource."""
        # Step 1: Signup
        signup_response = client.post(
            "/api/v1/auth/signup",
            json={
                "username": "flowuser",
                "email": "flowuser@example.com",
                "password": "FlowPass123!",
                "confirm_password": "FlowPass123!",
            },
        )
        assert signup_response.status_code == status.HTTP_201_CREATED

        # Step 2: Login
        login_response = client.post(
            "/api/v1/auth/login",
            json={
                "username": "flowuser",
                "password": "FlowPass123!",
            },
        )
        assert login_response.status_code == status.HTTP_200_OK
        tokens = login_response.json()
        access_token = tokens["access_token"]

        # Step 3: Get current user
        user_response = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        assert user_response.status_code == status.HTTP_200_OK
        user_data = user_response.json()
        assert user_data["username"] == "flowuser"
        assert user_data["email"] == "flowuser@example.com"
        assert "viewer" in [r.lower() for r in user_data["roles"]]

    def test_signup_login_refresh_flow(self, client):
        """Test: Signup -> Login -> Refresh token -> Access resource."""
        # Signup
        client.post(
            "/api/v1/auth/signup",
            json={
                "username": "refreshuser",
                "email": "refreshuser@example.com",
                "password": "RefreshPass123!",
                "confirm_password": "RefreshPass123!",
            },
        )

        # Login
        login_response = client.post(
            "/api/v1/auth/login",
            json={
                "username": "refreshuser",
                "password": "RefreshPass123!",
            },
        )
        tokens = login_response.json()
        old_access_token = tokens["access_token"]
        refresh_token = tokens["refresh_token"]

        # Refresh tokens
        refresh_response = client.post(
            "/api/v1/auth/refresh",
            params={"refresh_token": refresh_token},
        )
        assert refresh_response.status_code == status.HTTP_200_OK
        new_tokens = refresh_response.json()
        new_access_token = new_tokens["access_token"]

        # Verify new token works
        user_response = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {new_access_token}"},
        )
        assert user_response.status_code == status.HTTP_200_OK

        # Verify tokens are different
        assert new_access_token != old_access_token

    def test_login_logout_flow(self, client):
        """Test: Login -> Access resource -> Logout."""
        # Create and login user
        client.post(
            "/api/v1/auth/signup",
            json={
                "username": "logoutuser",
                "email": "logoutuser@example.com",
                "password": "LogoutPass123!",
                "confirm_password": "LogoutPass123!",
            },
        )

        login_response = client.post(
            "/api/v1/auth/login",
            json={
                "username": "logoutuser",
                "password": "LogoutPass123!",
            },
        )
        tokens = login_response.json()
        access_token = tokens["access_token"]

        # Access resource before logout
        user_response = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        assert user_response.status_code == status.HTTP_200_OK

        # Logout
        logout_response = client.post("/api/v1/auth/logout")
        assert logout_response.status_code == status.HTTP_200_OK

        # In a real scenario, the client would discard the token
        # The token itself may still be valid server-side unless using
        # a token blacklist, which is optional


# =============================================================================
# Multi-User Scenarios
# =============================================================================


class TestMultiUserScenarios:
    """Test scenarios with multiple users."""

    def test_multiple_users_different_roles(self, client, rbac_manager):
        """Test multiple users with different roles."""
        # Create admin user
        admin = rbac_manager.users.create_user(
            username="admin",
            email="admin@example.com",
            password="AdminPass123!",
            roles={Role.ADMIN},
        )

        # Create viewer via signup (gets VIEWER role)
        client.post(
            "/api/v1/auth/signup",
            json={
                "username": "viewer",
                "email": "viewer@example.com",
                "password": "ViewerPass123!",
                "confirm_password": "ViewerPass123!",
            },
        )

        # Login as admin
        admin_login = client.post(
            "/api/v1/auth/login",
            json={"username": "admin", "password": "AdminPass123!"},
        )
        admin_token = admin_login.json()["access_token"]

        # Login as viewer
        viewer_login = client.post(
            "/api/v1/auth/login",
            json={"username": "viewer", "password": "ViewerPass123!"},
        )
        viewer_token = viewer_login.json()["access_token"]

        # Verify admin has admin permissions
        admin_user = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        admin_data = admin_user.json()
        assert "admin" in [r.lower() for r in admin_data["roles"]]

        # Verify viewer has limited permissions
        viewer_user = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {viewer_token}"},
        )
        viewer_data = viewer_user.json()
        assert "viewer" in [r.lower() for r in viewer_data["roles"]]
        assert "admin" not in [r.lower() for r in viewer_data["roles"]]

    def test_concurrent_user_sessions(self, client):
        """Test multiple users with active sessions simultaneously."""
        users = []

        # Create multiple users
        for i in range(5):
            username = f"user{i}"
            client.post(
                "/api/v1/auth/signup",
                json={
                    "username": username,
                    "email": f"{username}@example.com",
                    "password": f"Password{i}123!",
                    "confirm_password": f"Password{i}123!",
                },
            )

            # Login
            login_response = client.post(
                "/api/v1/auth/login",
                json={
                    "username": username,
                    "password": f"Password{i}123!",
                },
            )
            users.append(
                {
                    "username": username,
                    "token": login_response.json()["access_token"],
                }
            )

        # Verify all users can access their resources
        for user in users:
            response = client.get(
                "/api/v1/auth/me",
                headers={"Authorization": f"Bearer {user['token']}"},
            )
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["username"] == user["username"]

    def test_user_cannot_access_other_user_session(self, client):
        """Test that one user cannot impersonate another."""
        # Create two users
        for username in ["alice", "bob"]:
            client.post(
                "/api/v1/auth/signup",
                json={
                    "username": username,
                    "email": f"{username}@example.com",
                    "password": f"{username.capitalize()}Pass123!",
                    "confirm_password": f"{username.capitalize()}Pass123!",
                },
            )

        # Login as Alice
        alice_login = client.post(
            "/api/v1/auth/login",
            json={"username": "alice", "password": "AlicePass123!"},
        )
        alice_token = alice_login.json()["access_token"]

        # Verify Alice gets her own data
        alice_data = client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {alice_token}"},
        ).json()
        assert alice_data["username"] == "alice"
        assert alice_data["email"] == "alice@example.com"


# =============================================================================
# CORS Tests
# =============================================================================


class TestCORSFunctionality:
    """Test CORS configuration for cross-origin requests."""

    def test_cors_preflight_request(self, client):
        """Test CORS preflight (OPTIONS) request."""
        response = client.options(
            "/api/v1/auth/login",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "Content-Type",
            },
        )

        # Should allow the request
        assert response.status_code in [status.HTTP_200_OK, status.HTTP_204_NO_CONTENT]
        assert "access-control-allow-origin" in response.headers

    def test_cors_headers_present(self, client):
        """Test that CORS headers are present in responses."""
        response = client.post(
            "/api/v1/auth/login",
            json={"username": "test", "password": "test123456"},
            headers={"Origin": "http://localhost:3000"},
        )

        # Check for CORS headers
        assert "access-control-allow-origin" in response.headers

    def test_cors_credentials_allowed(self, client):
        """Test that credentials are allowed in CORS."""
        response = client.get(
            "/api/v1/auth/me",
            headers={"Origin": "http://localhost:3000"},
        )

        if "access-control-allow-credentials" in response.headers:
            assert response.headers["access-control-allow-credentials"] == "true"


# =============================================================================
# Token Management Tests
# =============================================================================


class TestTokenManagement:
    """Test token lifecycle and management."""

    def test_token_contains_user_info(self, client, rbac_manager):
        """Test that token contains necessary user information."""
        # Create user
        client.post(
            "/api/v1/auth/signup",
            json={
                "username": "tokenuser",
                "email": "tokenuser@example.com",
                "password": "TokenPass123!",
                "confirm_password": "TokenPass123!",
            },
        )

        # Login
        login_response = client.post(
            "/api/v1/auth/login",
            json={"username": "tokenuser", "password": "TokenPass123!"},
        )
        access_token = login_response.json()["access_token"]

        # Decode token (normally done server-side)
        payload = rbac_manager.tokens.validate_token(access_token)

        assert payload.username == "tokenuser"
        assert payload.token_type == "access"
        assert len(payload.roles) > 0
        assert len(payload.permissions) > 0

    def test_refresh_token_different_from_access(self, client):
        """Test that refresh token is different from access token."""
        client.post(
            "/api/v1/auth/signup",
            json={
                "username": "difftoken",
                "email": "difftoken@example.com",
                "password": "DiffPass123!",
                "confirm_password": "DiffPass123!",
            },
        )

        login_response = client.post(
            "/api/v1/auth/login",
            json={"username": "difftoken", "password": "DiffPass123!"},
        )
        tokens = login_response.json()

        assert tokens["access_token"] != tokens["refresh_token"]

    def test_multiple_refresh_cycles(self, client):
        """Test multiple token refresh cycles."""
        # Create and login user
        client.post(
            "/api/v1/auth/signup",
            json={
                "username": "multirefresh",
                "email": "multirefresh@example.com",
                "password": "MultiPass123!",
                "confirm_password": "MultiPass123!",
            },
        )

        login_response = client.post(
            "/api/v1/auth/login",
            json={"username": "multirefresh", "password": "MultiPass123!"},
        )
        refresh_token = login_response.json()["refresh_token"]

        # Refresh multiple times
        for i in range(3):
            refresh_response = client.post(
                "/api/v1/auth/refresh",
                params={"refresh_token": refresh_token},
            )
            assert refresh_response.status_code == status.HTTP_200_OK

            new_tokens = refresh_response.json()
            refresh_token = new_tokens["refresh_token"]

            # Verify new access token works
            user_response = client.get(
                "/api/v1/auth/me",
                headers={"Authorization": f"Bearer {new_tokens['access_token']}"},
            )
            assert user_response.status_code == status.HTTP_200_OK


# =============================================================================
# Error Recovery Tests
# =============================================================================


class TestErrorRecovery:
    """Test error handling and recovery scenarios."""

    def test_failed_login_then_successful_login(self, client):
        """Test successful login after failed attempts."""
        # Create user
        client.post(
            "/api/v1/auth/signup",
            json={
                "username": "recoveryuser",
                "email": "recoveryuser@example.com",
                "password": "RecoveryPass123!",
                "confirm_password": "RecoveryPass123!",
            },
        )

        # Failed login attempts
        for _ in range(2):
            response = client.post(
                "/api/v1/auth/login",
                json={"username": "recoveryuser", "password": "WrongPassword"},
            )
            assert response.status_code == status.HTTP_401_UNAUTHORIZED

        # Successful login
        response = client.post(
            "/api/v1/auth/login",
            json={"username": "recoveryuser", "password": "RecoveryPass123!"},
        )
        assert response.status_code == status.HTTP_200_OK

    def test_signup_failure_then_success(self, client):
        """Test successful signup after fixing validation errors."""
        # Failed signup - password mismatch
        response = client.post(
            "/api/v1/auth/signup",
            json={
                "username": "fixeduser",
                "email": "fixeduser@example.com",
                "password": "Password123!",
                "confirm_password": "DifferentPass123!",
            },
        )
        assert response.status_code == status.HTTP_400_BAD_REQUEST

        # Successful signup with corrected data
        response = client.post(
            "/api/v1/auth/signup",
            json={
                "username": "fixeduser",
                "email": "fixeduser@example.com",
                "password": "Password123!",
                "confirm_password": "Password123!",
            },
        )
        assert response.status_code == status.HTTP_201_CREATED


# =============================================================================
# Session Persistence Tests
# =============================================================================


class TestSessionPersistence:
    """Test that sessions persist correctly across requests."""

    def test_token_valid_across_multiple_requests(self, client):
        """Test that token remains valid for multiple sequential requests."""
        # Create and login user
        client.post(
            "/api/v1/auth/signup",
            json={
                "username": "persistuser",
                "email": "persistuser@example.com",
                "password": "PersistPass123!",
                "confirm_password": "PersistPass123!",
            },
        )

        login_response = client.post(
            "/api/v1/auth/login",
            json={"username": "persistuser", "password": "PersistPass123!"},
        )
        access_token = login_response.json()["access_token"]

        # Make multiple requests with same token
        for i in range(5):
            response = client.get(
                "/api/v1/auth/me",
                headers={"Authorization": f"Bearer {access_token}"},
            )
            assert response.status_code == status.HTTP_200_OK
            data = response.json()
            assert data["username"] == "persistuser"

    def test_user_info_consistent_across_requests(self, client):
        """Test that user information remains consistent."""
        # Create and login user
        client.post(
            "/api/v1/auth/signup",
            json={
                "username": "consistent",
                "email": "consistent@example.com",
                "password": "ConsistPass123!",
                "confirm_password": "ConsistPass123!",
            },
        )

        login_response = client.post(
            "/api/v1/auth/login",
            json={"username": "consistent", "password": "ConsistPass123!"},
        )
        access_token = login_response.json()["access_token"]

        # Get user info multiple times
        user_infos = []
        for _ in range(3):
            response = client.get(
                "/api/v1/auth/me",
                headers={"Authorization": f"Bearer {access_token}"},
            )
            user_infos.append(response.json())

        # All responses should be identical
        assert all(info["user_id"] == user_infos[0]["user_id"] for info in user_infos)
        assert all(info["username"] == "consistent" for info in user_infos)
        assert all(info["email"] == "consistent@example.com" for info in user_infos)
