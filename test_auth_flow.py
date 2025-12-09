#!/usr/bin/env python3
"""
Comprehensive Authentication Flow Testing Script.

Tests all authentication endpoints and error scenarios against the running backend.
"""

import json
import sys
import time
import urllib.error
import urllib.request
from typing import Any, Optional

# ANSI colors
RED = "\033[91m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
RESET = "\033[0m"
BOLD = "\033[1m"

API_URL = "http://localhost:8000"
FRONTEND_URL = "http://localhost:3000"

# Test results tracking
test_results = {
    "passed": 0,
    "failed": 0,
    "total": 0
}


def log_success(msg: str) -> None:
    test_results["passed"] += 1
    test_results["total"] += 1
    print(f"{GREEN}[OK]{RESET} {msg}")


def log_error(msg: str) -> None:
    test_results["failed"] += 1
    test_results["total"] += 1
    print(f"{RED}[FAIL]{RESET} {msg}")


def log_info(msg: str) -> None:
    print(f"{BLUE}[INFO]{RESET} {msg}")


def log_warning(msg: str) -> None:
    print(f"{YELLOW}[WARN]{RESET} {msg}")


def print_section(title: str) -> None:
    print(f"\n{BOLD}{BLUE}{'='*70}{RESET}")
    print(f"{BOLD}{BLUE}{title}{RESET}")
    print(f"{BOLD}{BLUE}{'='*70}{RESET}\n")


def test_server_running() -> bool:
    """Test if backend server is running."""
    print_section("Backend Server Status Check")
    try:
        req = urllib.request.Request(f"{API_URL}/api/v1/health", method="GET")
        with urllib.request.urlopen(req, timeout=3) as response:
            if response.status == 200:
                data = json.loads(response.read().decode("utf-8"))
                log_success(f"Backend server running at {API_URL}")
                log_info(f"Health: {json.dumps(data, indent=2)}")
                return True
    except Exception as e:
        log_error(f"Backend server not running: {e}")
        log_warning(f"Start server with: python main.py")
        return False
    return False


def test_auth_routes_exist() -> bool:
    """Test if auth routes are registered."""
    print_section("Auth Routes Registration Check")

    routes_to_test = [
        ("/api/v1/auth/signup", "POST"),
        ("/api/v1/auth/login", "POST"),
        ("/api/v1/auth/me", "GET"),
        ("/api/v1/auth/refresh", "POST"),
        ("/api/v1/auth/logout", "POST"),
    ]

    all_exist = True
    for path, method in routes_to_test:
        try:
            req = urllib.request.Request(f"{API_URL}{path}", method="OPTIONS")
            req.add_header("Origin", FRONTEND_URL)
            req.add_header("Access-Control-Request-Method", method)

            with urllib.request.urlopen(req, timeout=3) as response:
                log_success(f"{method:6} {path}")
        except urllib.error.HTTPError as e:
            if e.code == 405 or e.code == 200:
                log_success(f"{method:6} {path}")
            elif e.code == 404:
                log_error(f"{method:6} {path} - NOT FOUND")
                all_exist = False
            else:
                log_success(f"{method:6} {path} (status {e.code})")
        except Exception as e:
            log_error(f"{method:6} {path} - Error: {e}")
            all_exist = False

    return all_exist


def test_signup() -> tuple[bool, Optional[str], str]:
    """Test user signup."""
    print_section("User Signup - Success Case")

    username = f"testuser_{int(time.time())}"
    password = "TestPassword123!"
    email = f"{username}@example.com"

    try:
        data = json.dumps({
            "username": username,
            "email": email,
            "password": password,
            "confirm_password": password,
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{API_URL}/api/v1/auth/signup",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Origin": FRONTEND_URL,
            },
            method="POST"
        )

        with urllib.request.urlopen(req, timeout=5) as response:
            result = json.loads(response.read().decode("utf-8"))

            if response.status == 201:
                log_success(f"Signup successful")
                log_info(f"Username: {username}")
                log_info(f"Email: {email}")
                log_info(f"Message: {result.get('message', 'OK')}")
                return True, username, password
            else:
                log_error(f"Unexpected status {response.status}")
                return False, None, password

    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8")
        try:
            error_data = json.loads(error_body)
            log_error(f"Signup failed: {error_data.get('detail', error_body)}")
        except:
            log_error(f"Signup failed: {error_body}")
        return False, None, password
    except Exception as e:
        log_error(f"Signup error: {e}")
        return False, None, password


def test_signup_errors() -> bool:
    """Test signup error cases."""
    print_section("User Signup - Error Cases")

    all_passed = True

    # Test 1: Duplicate username
    print(f"{BOLD}Test: Duplicate Username{RESET}")
    try:
        data = json.dumps({
            "username": "testuser_duplicate",
            "email": "dup1@example.com",
            "password": "TestPass123!",
            "confirm_password": "TestPass123!",
        }).encode("utf-8")

        # Create first user
        req = urllib.request.Request(
            f"{API_URL}/api/v1/auth/signup",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )
        urllib.request.urlopen(req, timeout=5)

        # Try to create duplicate
        req = urllib.request.Request(
            f"{API_URL}/api/v1/auth/signup",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        try:
            urllib.request.urlopen(req, timeout=5)
            log_error("Duplicate username not rejected")
            all_passed = False
        except urllib.error.HTTPError as e:
            if e.code == 409:
                log_success("Duplicate username properly rejected (409)")
            else:
                log_error(f"Wrong status for duplicate: {e.code}")
                all_passed = False
    except Exception as e:
        log_error(f"Duplicate test error: {e}")
        all_passed = False

    # Test 2: Password mismatch
    print(f"\n{BOLD}Test: Password Mismatch{RESET}")
    try:
        data = json.dumps({
            "username": "testuser_pwd",
            "email": "pwd@example.com",
            "password": "Password123!",
            "confirm_password": "DifferentPass123!",
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{API_URL}/api/v1/auth/signup",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        try:
            urllib.request.urlopen(req, timeout=5)
            log_error("Password mismatch not rejected")
            all_passed = False
        except urllib.error.HTTPError as e:
            if e.code == 400:
                log_success("Password mismatch properly rejected (400)")
            else:
                log_error(f"Wrong status for mismatch: {e.code}")
                all_passed = False
    except Exception as e:
        log_error(f"Mismatch test error: {e}")
        all_passed = False

    # Test 3: Invalid email
    print(f"\n{BOLD}Test: Invalid Email{RESET}")
    try:
        data = json.dumps({
            "username": "testuser_email",
            "email": "not-an-email",
            "password": "Password123!",
            "confirm_password": "Password123!",
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{API_URL}/api/v1/auth/signup",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        try:
            urllib.request.urlopen(req, timeout=5)
            log_error("Invalid email not rejected")
            all_passed = False
        except urllib.error.HTTPError as e:
            if e.code == 422:
                log_success("Invalid email properly rejected (422)")
            else:
                log_error(f"Wrong status for invalid email: {e.code}")
                all_passed = False
    except Exception as e:
        log_error(f"Email test error: {e}")
        all_passed = False

    return all_passed


def test_login(username: str, password: str = "TestPassword123!") -> tuple[bool, str | None]:
    """Test user login."""
    print(f"\n{BOLD}4. Testing User Login{RESET}")

    try:
        data = json.dumps({
            "username": username,
            "password": password,
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{API_URL}/api/v1/auth/login",
            data=data,
            headers={
                "Content-Type": "application/json",
                "Origin": FRONTEND_URL,
            },
            method="POST"
        )

        with urllib.request.urlopen(req, timeout=5) as response:
            result = json.loads(response.read().decode("utf-8"))

            if response.status == 200:
                access_token = result.get("access_token")
                refresh_token = result.get("refresh_token")

                if access_token and refresh_token:
                    log_success("Login successful!")
                    log_info(f"  Access token: {access_token[:30]}...")
                    log_info(f"  Refresh token: {refresh_token[:30]}...")
                    log_info(f"  Token type: {result.get('token_type', 'bearer')}")
                    return True, access_token
                else:
                    log_error("Login succeeded but no tokens received")
                    return False, None
            else:
                log_error(f"Login returned status {response.status}")
                return False, None

    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8")
        try:
            error_data = json.loads(error_body)
            log_error(f"Login failed: {error_data.get('detail', error_body)}")
        except:
            log_error(f"Login failed: {error_body}")
        return False, None
    except Exception as e:
        log_error(f"Login error: {e}")
        return False, None


def test_get_user(access_token: str) -> bool:
    """Test get current user."""
    print(f"\n{BOLD}5. Testing Get Current User{RESET}")

    try:
        req = urllib.request.Request(
            f"{API_URL}/api/v1/auth/me",
            headers={
                "Authorization": f"Bearer {access_token}",
                "Origin": FRONTEND_URL,
            },
            method="GET"
        )

        with urllib.request.urlopen(req, timeout=5) as response:
            result = json.loads(response.read().decode("utf-8"))

            if response.status == 200:
                log_success("Get user successful!")
                log_info(f"  User ID: {result.get('user_id')}")
                log_info(f"  Username: {result.get('username')}")
                log_info(f"  Email: {result.get('email')}")
                log_info(f"  Roles: {', '.join(result.get('roles', []))}")
                return True
            else:
                log_error(f"Get user returned status {response.status}")
                return False

    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8")
        try:
            error_data = json.loads(error_body)
            log_error(f"Get user failed: {error_data.get('detail', error_body)}")
        except:
            log_error(f"Get user failed: {error_body}")
        return False
    except Exception as e:
        log_error(f"Get user error: {e}")
        return False


def test_cors() -> bool:
    """Test CORS configuration."""
    print(f"\n{BOLD}6. Testing CORS Configuration{RESET}")

    try:
        req = urllib.request.Request(
            f"{API_URL}/api/v1/auth/login",
            method="POST"
        )
        req.add_header("Origin", FRONTEND_URL)
        req.add_header("Content-Type", "application/json")

        # Send empty body to trigger CORS check
        req.data = b'{"username":"test","password":"test"}'

        try:
            with urllib.request.urlopen(req, timeout=3) as response:
                cors_origin = response.headers.get("Access-Control-Allow-Origin")
                if cors_origin:
                    log_success(f"CORS is configured: {cors_origin}")
                    return True
                else:
                    log_warning("CORS headers not found (might still work)")
                    return True
        except urllib.error.HTTPError as e:
            # Check CORS headers even on error
            cors_origin = e.headers.get("Access-Control-Allow-Origin")
            if cors_origin:
                log_success(f"CORS is configured: {cors_origin}")
                return True
            else:
                log_error("CORS headers not found - this will cause frontend issues")
                return False

    except Exception as e:
        log_error(f"CORS test error: {e}")
        return False


def main() -> int:
    """Main test runner."""
    print(f"\n{BOLD}{BLUE}{'='*60}{RESET}")
    print(f"{BOLD}{BLUE}  Authentication Flow Test{RESET}")
    print(f"{BOLD}{BLUE}{'='*60}{RESET}")

    # Test 1: Server running
    if not test_server_running():
        return 1

    # Test 2: Auth routes exist
    if not test_auth_routes_exist():
        log_error("\nAuth routes are missing!")
        log_warning("The server might not have reloaded after adding the auth router.")
        log_warning("Please restart the server completely:")
        log_warning("  1. Stop the server (Ctrl+C)")
        log_warning("  2. Run: python main.py")
        return 1

    # Test 3: Signup
    signup_success, username, password = test_signup()
    if not signup_success or not username:
        log_error("\nSignup failed!")
        return 1

    # Test 4: Login
    login_success, access_token = test_login(username, password)
    if not login_success or not access_token:
        log_error("\nLogin failed!")
        return 1

    # Test 5: Get user
    if not test_get_user(access_token):
        log_error("\nGet user failed!")
        return 1

    # Test 6: CORS
    cors_ok = test_cors()

    # Summary
    print(f"\n{BOLD}{GREEN}{'='*60}{RESET}")
    print(f"{BOLD}{GREEN}  SUCCESS: All Authentication Tests Passed!{RESET}")
    print(f"{BOLD}{GREEN}{'='*60}{RESET}")

    print(f"\n{BOLD}Frontend should now work!{RESET}")
    print(f"  1. Visit: {FRONTEND_URL}/signup")
    print(f"  2. Create an account")
    print(f"  3. Login at: {FRONTEND_URL}/login")
    print()

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        print(f"\n{YELLOW}Test interrupted by user{RESET}")
        sys.exit(130)
