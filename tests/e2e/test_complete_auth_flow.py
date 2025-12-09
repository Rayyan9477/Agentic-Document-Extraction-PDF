"""
End-to-End Authentication Flow Tests.

Tests the complete authentication system from start to finish,
simulating real user workflows.
"""

import pytest
import time
from datetime import datetime, timedelta
from fastapi import status
from fastapi.testclient import TestClient

from src.security.rbac import RBACManager, Role


@pytest.mark.e2e
class TestCompleteAuthenticationFlow:
    """End-to-end tests for complete authentication flows."""

    def test_new_user_complete_journey(self, test_client, rbac_manager):
        """
        Test: Complete new user journey.

        Flow: Signup -> Login -> Access Protected Resource -> Refresh Token -> Logout
        """
        # Step 1: User signs up
        signup_response = test_client.post(
            "/api/v1/auth/signup",
            json={
                "username": "newuser",
                "email": "newuser@example.com",
                "password": "NewUserPass123!",
                "confirm_password": "NewUserPass123!",
            },
        )
        assert signup_response.status_code == status.HTTP_201_CREATED
        signup_data = signup_response.json()
        assert signup_data["success"] is True

        # Step 2: User logs in
        login_response = test_client.post(
            "/api/v1/auth/login",
            json={
                "username": "newuser",
                "password": "NewUserPass123!",
            },
        )
        assert login_response.status_code == status.HTTP_200_OK
        tokens = login_response.json()
        assert "access_token" in tokens
        assert "refresh_token" in tokens

        access_token = tokens["access_token"]
        refresh_token = tokens["refresh_token"]

        # Step 3: User accesses their profile
        me_response = test_client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        assert me_response.status_code == status.HTTP_200_OK
        user_data = me_response.json()
        assert user_data["username"] == "newuser"
        assert user_data["email"] == "newuser@example.com"
        assert "viewer" in [r.lower() for r in user_data["roles"]]

        # Step 4: User refreshes their token
        refresh_response = test_client.post(
            "/api/v1/auth/refresh",
            params={"refresh_token": refresh_token},
        )
        assert refresh_response.status_code == status.HTTP_200_OK
        new_tokens = refresh_response.json()
        assert new_tokens["access_token"] != access_token

        # Step 5: User accesses profile with new token
        me_response_2 = test_client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {new_tokens['access_token']}"},
        )
        assert me_response_2.status_code == status.HTTP_200_OK

        # Step 6: User logs out
        logout_response = test_client.post("/api/v1/auth/logout")
        assert logout_response.status_code == status.HTTP_200_OK

    def test_user_with_wrong_password_recovery(self, test_client):
        """
        Test: User tries wrong password then succeeds.

        Flow: Signup -> Failed Login (wrong password) -> Successful Login
        """
        # Signup
        test_client.post(
            "/api/v1/auth/signup",
            json={
                "username": "forgetfuluser",
                "email": "forgetful@example.com",
                "password": "CorrectPass123!",
                "confirm_password": "CorrectPass123!",
            },
        )

        # Try wrong password
        wrong_login = test_client.post(
            "/api/v1/auth/login",
            json={
                "username": "forgetfuluser",
                "password": "WrongPassword123!",
            },
        )
        assert wrong_login.status_code == status.HTTP_401_UNAUTHORIZED

        # Try correct password
        correct_login = test_client.post(
            "/api/v1/auth/login",
            json={
                "username": "forgetfuluser",
                "password": "CorrectPass123!",
            },
        )
        assert correct_login.status_code == status.HTTP_200_OK

    def test_multiple_sessions_different_users(self, test_client):
        """
        Test: Multiple users with concurrent sessions.

        Flow: Create 3 users -> All login -> All access their profiles concurrently
        """
        users = [
            {"username": "user1", "email": "user1@example.com", "password": "Pass1123!"},
            {"username": "user2", "email": "user2@example.com", "password": "Pass2123!"},
            {"username": "user3", "email": "user3@example.com", "password": "Pass3123!"},
        ]

        # Signup all users
        for user in users:
            response = test_client.post(
                "/api/v1/auth/signup",
                json={
                    **user,
                    "confirm_password": user["password"],
                },
            )
            assert response.status_code == status.HTTP_201_CREATED

        # Login all users and store tokens
        user_sessions = []
        for user in users:
            login_response = test_client.post(
                "/api/v1/auth/login",
                json={
                    "username": user["username"],
                    "password": user["password"],
                },
            )
            assert login_response.status_code == status.HTTP_200_OK
            tokens = login_response.json()
            user_sessions.append({
                "username": user["username"],
                "email": user["email"],
                "token": tokens["access_token"],
            })

        # All users access their profiles
        for session in user_sessions:
            me_response = test_client.get(
                "/api/v1/auth/me",
                headers={"Authorization": f"Bearer {session['token']}"},
            )
            assert me_response.status_code == status.HTTP_200_OK
            data = me_response.json()
            assert data["username"] == session["username"]
            assert data["email"] == session["email"]

    def test_token_refresh_chain(self, test_client):
        """
        Test: Multiple token refresh cycles.

        Flow: Signup -> Login -> Refresh x3 -> Access with final token
        """
        # Signup and login
        test_client.post(
            "/api/v1/auth/signup",
            json={
                "username": "refresher",
                "email": "refresher@example.com",
                "password": "RefreshPass123!",
                "confirm_password": "RefreshPass123!",
            },
        )

        login_response = test_client.post(
            "/api/v1/auth/login",
            json={
                "username": "refresher",
                "password": "RefreshPass123!",
            },
        )
        refresh_token = login_response.json()["refresh_token"]

        # Refresh 3 times
        for i in range(3):
            refresh_response = test_client.post(
                "/api/v1/auth/refresh",
                params={"refresh_token": refresh_token},
            )
            assert refresh_response.status_code == status.HTTP_200_OK
            tokens = refresh_response.json()
            refresh_token = tokens["refresh_token"]
            access_token = tokens["access_token"]

        # Access with final token
        me_response = test_client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {access_token}"},
        )
        assert me_response.status_code == status.HTTP_200_OK
        assert me_response.json()["username"] == "refresher"

    def test_account_lockout_scenario(self, test_client, rbac_manager):
        """
        Test: Account gets locked after failed attempts.

        Flow: Signup -> 5 Failed Logins -> Account Locked -> Cannot Login
        """
        # Signup
        test_client.post(
            "/api/v1/auth/signup",
            json={
                "username": "locked_user",
                "email": "locked@example.com",
                "password": "CorrectPass123!",
                "confirm_password": "CorrectPass123!",
            },
        )

        # 5 failed login attempts
        for i in range(5):
            response = test_client.post(
                "/api/v1/auth/login",
                json={
                    "username": "locked_user",
                    "password": f"WrongPass{i}",
                },
            )
            assert response.status_code == status.HTTP_401_UNAUTHORIZED

        # Verify account is locked
        user = rbac_manager.users.get_user_by_username("locked_user")
        assert user.is_locked is True

        # Try correct password - should still fail
        response = test_client.post(
            "/api/v1/auth/login",
            json={
                "username": "locked_user",
                "password": "CorrectPass123!",
            },
        )
        assert response.status_code == status.HTTP_401_UNAUTHORIZED

    def test_permission_based_access(self, test_client, rbac_manager):
        """
        Test: Different users have different permissions.

        Flow: Create Viewer and Admin -> Both login -> Verify different permissions
        """
        # Create viewer via signup
        test_client.post(
            "/api/v1/auth/signup",
            json={
                "username": "viewer_user",
                "email": "viewer@example.com",
                "password": "ViewerPass123!",
                "confirm_password": "ViewerPass123!",
            },
        )

        # Create admin directly
        admin = rbac_manager.users.create_user(
            username="admin_user",
            email="admin@example.com",
            password="AdminPass123!",
            roles={Role.ADMIN},
        )

        # Login both users
        viewer_login = test_client.post(
            "/api/v1/auth/login",
            json={"username": "viewer_user", "password": "ViewerPass123!"},
        )
        viewer_token = viewer_login.json()["access_token"]

        admin_login = test_client.post(
            "/api/v1/auth/login",
            json={"username": "admin_user", "password": "AdminPass123!"},
        )
        admin_token = admin_login.json()["access_token"]

        # Get viewer permissions
        viewer_me = test_client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {viewer_token}"},
        )
        viewer_perms = viewer_me.json()["permissions"]

        # Get admin permissions
        admin_me = test_client.get(
            "/api/v1/auth/me",
            headers={"Authorization": f"Bearer {admin_token}"},
        )
        admin_perms = admin_me.json()["permissions"]

        # Admin should have more permissions
        assert len(admin_perms) > len(viewer_perms)
        assert "system:admin" in admin_perms
        assert "system:admin" not in viewer_perms

    def test_signup_validation_error_recovery(self, test_client):
        """
        Test: User fixes validation errors and successfully signs up.

        Flow: Failed Signup (mismatch) -> Fixed Signup -> Login
        """
        # Failed signup - password mismatch
        failed_response = test_client.post(
            "/api/v1/auth/signup",
            json={
                "username": "validator",
                "email": "validator@example.com",
                "password": "Password123!",
                "confirm_password": "DifferentPass123!",
            },
        )
        assert failed_response.status_code == status.HTTP_400_BAD_REQUEST

        # Successful signup with corrected data
        success_response = test_client.post(
            "/api/v1/auth/signup",
            json={
                "username": "validator",
                "email": "validator@example.com",
                "password": "Password123!",
                "confirm_password": "Password123!",
            },
        )
        assert success_response.status_code == status.HTTP_201_CREATED

        # Successful login
        login_response = test_client.post(
            "/api/v1/auth/login",
            json={"username": "validator", "password": "Password123!"},
        )
        assert login_response.status_code == status.HTTP_200_OK


@pytest.mark.e2e
@pytest.mark.slow
class TestExtendedAuthFlows:
    """Extended end-to-end tests for edge cases and complex scenarios."""

    def test_rapid_login_logout_cycles(self, test_client):
        """Test rapid login/logout cycles don't cause issues."""
        # Signup
        test_client.post(
            "/api/v1/auth/signup",
            json={
                "username": "rapid_user",
                "email": "rapid@example.com",
                "password": "RapidPass123!",
                "confirm_password": "RapidPass123!",
            },
        )

        # 5 rapid login/logout cycles
        for _ in range(5):
            # Login
            login_response = test_client.post(
                "/api/v1/auth/login",
                json={"username": "rapid_user", "password": "RapidPass123!"},
            )
            assert login_response.status_code == status.HTTP_200_OK

            # Logout
            logout_response = test_client.post("/api/v1/auth/logout")
            assert logout_response.status_code == status.HTTP_200_OK

    def test_concurrent_refresh_requests(self, test_client):
        """Test that concurrent refresh requests are handled correctly."""
        # Signup and login
        test_client.post(
            "/api/v1/auth/signup",
            json={
                "username": "concurrent",
                "email": "concurrent@example.com",
                "password": "ConcurrentPass123!",
                "confirm_password": "ConcurrentPass123!",
            },
        )

        login_response = test_client.post(
            "/api/v1/auth/login",
            json={"username": "concurrent", "password": "ConcurrentPass123!"},
        )
        refresh_token = login_response.json()["refresh_token"]

        # First refresh
        refresh1 = test_client.post(
            "/api/v1/auth/refresh",
            params={"refresh_token": refresh_token},
        )
        assert refresh1.status_code == status.HTTP_200_OK

        # Second refresh with same token (simulating race condition)
        # Depending on implementation, this may or may not succeed
        refresh2 = test_client.post(
            "/api/v1/auth/refresh",
            params={"refresh_token": refresh_token},
        )
        # Document behavior - either succeeds or fails gracefully
        assert refresh2.status_code in [status.HTTP_200_OK, status.HTTP_401_UNAUTHORIZED]
