#!/usr/bin/env python3
"""Quick authentication test - single signup and login."""

import json
import time
import urllib.request
import urllib.error

API_URL = "http://localhost:8000"

def test_signup_and_login():
    """Test signup and login with a new user."""
    username = f"quicktest_{int(time.time())}"
    password = "TestPass@Word!42"  # No sequential chars
    email = f"{username}@example.com"

    print(f"\n=== Testing Authentication Flow ===\n")
    print(f"Username: {username}")
    print(f"Email: {email}")

    # Test 1: Signup
    print("\n[1/3] Testing signup...")
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
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        with urllib.request.urlopen(req, timeout=5) as response:
            result = json.loads(response.read().decode("utf-8"))
            print(f"[OK] Signup successful: {result.get('message', 'OK')}")
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8")
        try:
            error_data = json.loads(error_body)
            print(f"[FAIL] Signup failed ({e.code}):")
            print(f"  Error: {json.dumps(error_data, indent=2)}")
        except:
            print(f"[FAIL] Signup failed ({e.code}): {error_body}")
        return False
    except Exception as e:
        print(f"[FAIL] Signup error: {e}")
        return False

    # Test 2: Login
    print("\n[2/3] Testing login...")
    try:
        data = json.dumps({
            "username": username,
            "password": password,
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{API_URL}/api/v1/auth/login",
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST"
        )

        with urllib.request.urlopen(req, timeout=5) as response:
            result = json.loads(response.read().decode("utf-8"))
            access_token = result.get("access_token")
            print(f"[OK] Login successful")
            print(f"  Token: {access_token[:40]}...")
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8")
        print(f"[FAIL] Login failed ({e.code}): {error_body}")
        return False
    except Exception as e:
        print(f"[FAIL] Login error: {e}")
        return False

    # Test 3: Get user info
    print("\n[3/3] Testing get current user...")
    try:
        req = urllib.request.Request(
            f"{API_URL}/api/v1/auth/me",
            headers={"Authorization": f"Bearer {access_token}"},
            method="GET"
        )

        with urllib.request.urlopen(req, timeout=5) as response:
            result = json.loads(response.read().decode("utf-8"))
            print(f"[OK] Get user successful")
            print(f"  User ID: {result.get('user_id')}")
            print(f"  Username: {result.get('username')}")
            print(f"  Roles: {', '.join(result.get('roles', []))}")
    except urllib.error.HTTPError as e:
        error_body = e.read().decode("utf-8")
        print(f"[FAIL] Get user failed ({e.code}): {error_body}")
        return False
    except Exception as e:
        print(f"[FAIL] Get user error: {e}")
        return False

    print("\n=== [SUCCESS] All Tests Passed! ===\n")
    return True

if __name__ == "__main__":
    success = test_signup_and_login()
    exit(0 if success else 1)
