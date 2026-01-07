#!/usr/bin/env python3
"""
Quick authentication flow test runner.
Runs actual HTTP tests against http://localhost:8000
"""

import json
import sys
import time
import urllib.error
import urllib.request


# Colors
G = "\033[92m"  # Green
R = "\033[91m"  # Red
Y = "\033[93m"  # Yellow
B = "\033[94m"  # Blue
RESET = "\033[0m"
BOLD = "\033[1m"

BASE = "http://localhost:8000/api/v1"
test_count = {"pass": 0, "fail": 0}


def req(method: str, path: str, data=None, token=None) -> tuple[int, dict]:
    """Make HTTP request."""
    url = f"{BASE}{path}"
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    request = urllib.request.Request(
        url,
        data=json.dumps(data).encode() if data else None,
        headers=headers,
        method=method
    )

    try:
        with urllib.request.urlopen(request, timeout=5) as response:
            return response.status, json.loads(response.read().decode())
    except urllib.error.HTTPError as e:
        try:
            return e.code, json.loads(e.read().decode())
        except:
            return e.code, {"error": str(e)}
    except Exception as e:
        return 0, {"error": str(e)}


def test(name: str, condition: bool, details: str = "") -> None:
    """Record test result."""
    if condition:
        test_count["pass"] += 1
        if details:
            pass
    else:
        test_count["fail"] += 1
        if details:
            pass


def main():

    # Test 1: Server health
    status, data = req("GET", "/health")
    test("Backend server running", status == 200, f"Status: {status}")

    if status != 200:
        return 1

    # Test 2: Signup
    username = f"test_{int(time.time())}"
    password = "SecurePass123!"
    email = f"{username}@test.com"

    status, data = req("POST", "/auth/signup", {
        "username": username,
        "email": email,
        "password": password,
        "confirm_password": password
    })
    test("Signup successful", status == 201, f"User: {username}")

    # Test 3: Duplicate signup
    status, data = req("POST", "/auth/signup", {
        "username": username,
        "email": email,
        "password": password,
        "confirm_password": password
    })
    test("Duplicate rejected (409)", status == 409, f"Detail: {data.get('detail', '')}")

    # Test 4: Password mismatch
    status, data = req("POST", "/auth/signup", {
        "username": f"bad_{int(time.time())}",
        "email": "bad@test.com",
        "password": "Pass1",
        "confirm_password": "Pass2"
    })
    test("Mismatch rejected (400)", status == 400, f"Detail: {data.get('detail', '')}")

    # Test 5: Login success
    status, data = req("POST", "/auth/login", {
        "username": username,
        "password": password
    })
    test("Login successful", status == 200, f"Token: {data.get('access_token', '')[:30]}...")

    if status != 200:
        return 1

    access_token = data.get("access_token")
    refresh_token = data.get("refresh_token")

    # Test 6: Wrong password
    status, data = req("POST", "/auth/login", {
        "username": username,
        "password": "WrongPassword!"
    })
    test("Wrong password rejected (401)", status == 401, f"Detail: {data.get('detail', '')}")

    # Test 7: Nonexistent user
    status, data = req("POST", "/auth/login", {
        "username": "nonexistent_12345",
        "password": "SomePass123!"
    })
    test("Nonexistent user rejected (401)", status == 401, f"Detail: {data.get('detail', '')}")

    # Test 8: Get current user (authenticated)
    status, data = req("GET", "/auth/me", token=access_token)
    test("Get user successful", status == 200, f"Username: {data.get('username')}, Roles: {data.get('roles')}")

    # Test 9: Get user without token
    status, data = req("GET", "/auth/me")
    test("No token rejected (401)", status == 401, f"Detail: {data.get('detail', '')}")

    # Test 10: Get user with invalid token
    status, data = req("GET", "/auth/me", token="invalid_token_123")
    test("Invalid token rejected (401)", status == 401, f"Detail: {data.get('detail', '')}")

    # Test 11: Refresh token
    url = f"{BASE}/auth/refresh?refresh_token={refresh_token}"
    try:
        request = urllib.request.Request(url, method="POST")
        with urllib.request.urlopen(request, timeout=5) as response:
            data = json.loads(response.read().decode())
            test("Token refresh successful", response.status == 200, f"New token: {data.get('access_token', '')[:30]}...")
    except urllib.error.HTTPError as e:
        test("Token refresh successful", False, f"Status {e.code}")

    # Test 12: Invalid refresh token
    url = f"{BASE}/auth/refresh?refresh_token=invalid_refresh_123"
    try:
        request = urllib.request.Request(url, method="POST")
        with urllib.request.urlopen(request, timeout=5) as response:
            test("Invalid refresh rejected (401)", False, "Should have been rejected")
    except urllib.error.HTTPError as e:
        test("Invalid refresh rejected (401)", e.code == 401, f"Status: {e.code}")

    # Test 13: Use access token as refresh (wrong type)
    url = f"{BASE}/auth/refresh?refresh_token={access_token}"
    try:
        request = urllib.request.Request(url, method="POST")
        with urllib.request.urlopen(request, timeout=5) as response:
            test("Access token as refresh rejected (401)", False, "Should have been rejected")
    except urllib.error.HTTPError as e:
        test("Access token as refresh rejected (401)", e.code == 401, f"Status: {e.code}")

    # Test 14: Logout
    status, data = req("POST", "/auth/logout")
    test("Logout successful", status == 200, f"Message: {data.get('message', '')}")

    # Summary
    total = test_count["pass"] + test_count["fail"]
    pass_rate = (test_count["pass"] / total * 100) if total > 0 else 0

    if test_count["fail"] == 0:
        return 0
    return 1


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(130)
