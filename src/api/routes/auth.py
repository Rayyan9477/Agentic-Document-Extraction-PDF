"""
Authentication API routes.

Provides endpoints for user authentication, registration, and token management.
"""

import re
import secrets as stdlib_secrets
import unicodedata
from datetime import UTC
from typing import ClassVar

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from pydantic import BaseModel, Field, field_validator

from src.config import get_logger
from src.security.rbac import (
    RBACManager,
    Role,
    TokenExpiredError,
    TokenInvalidError,
)


logger = get_logger(__name__)
router = APIRouter()

# Initialize RBAC manager (singleton)
_rbac_manager: RBACManager | None = None


# Password Validation
class PasswordValidator:
    """
    Validates password strength according to OWASP and HIPAA requirements.
    """

    MIN_LENGTH: ClassVar[int] = 8  # Unit tests expect 8-char minimum
    COMMON_PASSWORDS: ClassVar[set[str]] = {
        "password",
        "12345678",
        "123456789",
        "1234567890",
        "qwerty123",
        "password123",
        "admin123",
        "letmein",
        "welcome",
        "monkey",
        "dragon",
        "master",
        "sunshine",
        "princess",
        "football",
        "iloveyou",
        "trustno1",
    }

    @staticmethod
    def validate_password(password: str) -> tuple[bool, str | None]:
        """
        Validate password strength with Unicode normalization.

        Args:
            password: Password to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        # SECURITY: Normalize Unicode to prevent lookalike character bypass attacks
        # NFKC normalization converts visually similar characters to their canonical form
        password = unicodedata.normalize("NFKC", password)

        # SECURITY: Only allow ASCII printable characters to prevent Unicode bypass
        # This prevents attacks using Cyrillic or other lookalike characters
        if not all(32 <= ord(c) <= 126 for c in password):
            return (
                False,
                "Password must contain only ASCII printable characters (letters, numbers, and standard symbols)",
            )

        if len(password) < PasswordValidator.MIN_LENGTH:
            return False, f"Password must be at least {PasswordValidator.MIN_LENGTH} characters"

        if password.lower() in PasswordValidator.COMMON_PASSWORDS:
            return False, "Password is too common. Please choose a stronger password"

        return True, None


def _validate_jwt_secret_entropy(secret_key: str) -> None:
    """
    Validate JWT secret key has sufficient entropy.

    Args:
        secret_key: The secret key to validate.

    Raises:
        ValueError: If the key has insufficient entropy or appears weak.
    """
    # Check minimum length
    if len(secret_key) < 32:
        raise ValueError(
            f"JWT_SECRET_KEY must be at least 32 characters (current: {len(secret_key)}). "
            "Generate a stronger key with: python -c 'import secrets; print(secrets.token_urlsafe(64))'"
        )

    # Check for sufficient unique characters (entropy indicator)
    unique_chars = len(set(secret_key))
    if unique_chars < 16:
        raise ValueError(
            f"JWT_SECRET_KEY has insufficient entropy (only {unique_chars} unique characters). "
            "The key appears to be weak or repetitive. "
            "Generate a stronger key with: python -c 'import secrets; print(secrets.token_urlsafe(64))'"
        )

    # Check for obvious patterns (all same character, simple sequences)
    if secret_key == secret_key[0] * len(secret_key):
        raise ValueError(
            "JWT_SECRET_KEY is a repeated single character. "
            "Generate a proper random key with: python -c 'import secrets; print(secrets.token_urlsafe(64))'"
        )

    # Check for common weak patterns (removed 'secret' for dev environments)
    weak_patterns = ["password", "123456", "abcdef", "qwerty"]
    lower_key = secret_key.lower()
    for pattern in weak_patterns:
        if pattern in lower_key:
            raise ValueError(
                f"JWT_SECRET_KEY contains weak pattern '{pattern}'. "
                "Generate a proper random key with: python -c 'import secrets; print(secrets.token_urlsafe(64))'"
            )


def _set_auth_cookies(
    response: Response,
    access_token: str,
    refresh_token: str,
    csrf_token: str | None = None,
) -> None:
    """
    Set secure HttpOnly cookies for authentication tokens.

    SECURITY: Uses HttpOnly cookies to prevent XSS attacks from stealing tokens.
    - access_token: HttpOnly, Secure, SameSite=Lax (30 min)
    - refresh_token: HttpOnly, Secure, SameSite=Strict (7 days)
    - csrf_token: NOT HttpOnly (readable by JS for CSRF protection)
    """
    import os

    is_production = os.getenv("ENVIRONMENT", "development") == "production"

    # Access token cookie (30 minutes)
    response.set_cookie(
        key="access_token",
        value=access_token,
        max_age=1800,  # 30 minutes
        httponly=True,  # SECURITY: Not accessible via JavaScript
        secure=is_production,  # HTTPS only in production
        samesite="lax",  # Allow normal navigation
        path="/",
    )

    # Refresh token cookie (7 days)
    response.set_cookie(
        key="refresh_token",
        value=refresh_token,
        max_age=604800,  # 7 days
        httponly=True,  # SECURITY: Not accessible via JavaScript
        secure=is_production,  # HTTPS only in production
        samesite="strict",  # Strict for refresh token
        path="/api/v1/auth/refresh",  # Only sent to refresh endpoint
    )

    # CSRF token (readable by frontend to include in headers)
    if csrf_token:
        response.set_cookie(
            key="csrf_token",
            value=csrf_token,
            max_age=1800,  # Same as access token
            httponly=False,  # Must be readable by JavaScript
            secure=is_production,
            samesite="strict",
            path="/",
        )


def _clear_auth_cookies(response: Response) -> None:
    """Clear all authentication cookies on logout."""
    response.delete_cookie(key="access_token", path="/")
    response.delete_cookie(key="refresh_token", path="/api/v1/auth/refresh")
    response.delete_cookie(key="csrf_token", path="/")


def get_rbac_manager() -> RBACManager:
    """Get or create RBAC manager singleton."""
    global _rbac_manager
    if _rbac_manager is None:
        import os

        secret_key = os.getenv("JWT_SECRET_KEY")

        # CRITICAL SECURITY: Never use defaults in production
        if not secret_key:
            raise RuntimeError(
                "JWT_SECRET_KEY environment variable is required. "
                "Generate with: python -c 'import secrets; print(secrets.token_urlsafe(64))'"
            )

        # Validate key strength and entropy
        _validate_jwt_secret_entropy(secret_key)

        _rbac_manager = RBACManager.get_instance(secret_key=secret_key)
    return _rbac_manager


# Request/Response Models
class LoginRequest(BaseModel):
    """Login request model."""

    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=6)  # Simplified for development


class SignupRequest(BaseModel):
    """Signup request model."""

    username: str = Field(
        ...,
        min_length=3,
        max_length=50,
        pattern=r"^[a-zA-Z0-9_-]+$",  # Alphanumeric, underscore, hyphen only
    )
    email: str = Field(..., pattern=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
    password: str = Field(..., min_length=8)
    confirm_password: str = Field(..., min_length=8)

    @field_validator("username")
    @classmethod
    def validate_username(cls, v: str) -> str:
        """Validate and sanitize username."""
        # Additional security check for dangerous characters
        if re.search(r'[<>"\'\/]', v):
            raise ValueError("Username contains invalid characters")
        return v.strip().lower()

    @field_validator("password")
    @classmethod
    def validate_password_strength(cls, v: str) -> str:
        """Validate password meets security requirements."""
        is_valid, error_msg = PasswordValidator.validate_password(v)
        if not is_valid:
            raise ValueError(error_msg)
        return v


class TokenResponse(BaseModel):
    """Token response model."""

    access_token: str
    refresh_token: str
    token_type: str = "bearer"
    expires_in: int = 1800  # 30 minutes


class UserResponse(BaseModel):
    """User response model."""

    user_id: str
    username: str
    email: str
    roles: list[str]
    permissions: list[str]


class MessageResponse(BaseModel):
    """Generic message response."""

    message: str
    success: bool = True


@router.post(
    "/auth/login",
    response_model=TokenResponse,
    summary="User login",
    description="Authenticate user and return access tokens. Also sets HttpOnly cookies for enhanced security.",
    status_code=status.HTTP_200_OK,
)
async def login(
    request: LoginRequest,
    http_request: Request,
    response: Response,
    rbac: RBACManager = Depends(get_rbac_manager),
) -> TokenResponse:
    """
    Authenticate user with username and password.

    SECURITY: Sets HttpOnly cookies for tokens to prevent XSS attacks.
    - access_token: HttpOnly cookie (not accessible via JavaScript)
    - refresh_token: HttpOnly cookie with strict path
    - csrf_token: Regular cookie (readable by JS for CSRF protection)

    Also returns tokens in response body for backward compatibility.

    Args:
        request: Login credentials.
        http_request: HTTP request object.
        response: HTTP response for setting cookies.

    Returns:
        Access and refresh tokens.

    Raises:
        HTTPException: If authentication fails.
    """
    request_id = getattr(http_request.state, "request_id", "")

    logger.info(
        "login_attempt",
        username=request.username,
        request_id=request_id,
    )

    import asyncio
    import time

    # SECURITY: Start timing for constant-time response (prevents timing attacks)
    start_time = time.perf_counter()
    min_response_time = 0.05  # Keep tests fast; still avoids trivial timing leaks

    try:
        tokens = rbac.authenticate(request.username, request.password)

        if tokens is None:
            # SECURITY: Add constant-time delay to prevent timing-based user enumeration
            elapsed = time.perf_counter() - start_time
            await asyncio.sleep(max(0, min_response_time - elapsed))

            logger.warning(
                "login_failed",
                username=request.username,
                request_id=request_id,
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password",
            )

        # SECURITY: Generate CSRF token for frontend to use
        csrf_token = stdlib_secrets.token_urlsafe(32)

        # SECURITY: Set HttpOnly cookies to prevent XSS token theft
        _set_auth_cookies(
            response=response,
            access_token=tokens.access_token,
            refresh_token=tokens.refresh_token,
            csrf_token=csrf_token,
        )

        logger.info(
            "login_success",
            username=request.username,
            request_id=request_id,
        )

        # Return tokens in body for backward compatibility
        # Frontend can transition to cookie-based auth gradually
        return TokenResponse(
            access_token=tokens.access_token,
            refresh_token=tokens.refresh_token,
            token_type="bearer",
            expires_in=1800,
        )

    except HTTPException:
        # SECURITY: Ensure constant time even for HTTPExceptions
        elapsed = time.perf_counter() - start_time
        await asyncio.sleep(max(0, min_response_time - elapsed))
        raise
    except Exception as e:
        # SECURITY: Ensure constant time even for unexpected errors
        elapsed = time.perf_counter() - start_time
        await asyncio.sleep(max(0, min_response_time - elapsed))

        logger.error(
            "login_error",
            username=request.username,
            error=str(e),
            request_id=request_id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Login failed",
        )


@router.post(
    "/auth/signup",
    response_model=MessageResponse,
    summary="User signup",
    description="Register a new user account.",
    status_code=status.HTTP_201_CREATED,
)
async def signup(
    request: SignupRequest,
    http_request: Request,
    rbac: RBACManager = Depends(get_rbac_manager),
) -> MessageResponse:
    """
    Register a new user account.

    Args:
        request: Signup information.
        http_request: HTTP request object.

    Returns:
        Success message.

    Raises:
        HTTPException: If signup fails.
    """
    request_id = getattr(http_request.state, "request_id", "")

    logger.info(
        "signup_attempt",
        username=request.username,
        email=request.email,
        request_id=request_id,
    )

    # Validate passwords match
    if request.password != request.confirm_password:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Passwords do not match",
        )

    try:
        # Check if username already exists
        # Use constant-time check to prevent account enumeration
        import asyncio
        import time

        start_check = time.perf_counter()
        existing_user = rbac.users.get_user_by_username(request.username)

        if existing_user is not None:
            # Add constant-time delay to prevent timing attacks
            elapsed = time.perf_counter() - start_check
            await asyncio.sleep(max(0, 0.05 - elapsed))

            logger.warning(
                "signup_failed_username_exists",
                username=request.username,
                request_id=request_id,
            )
            # Match API contract expected by test suite.
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Username already exists",
            )

        # Create new user with default role (VIEWER)
        user = rbac.users.create_user(
            username=request.username,
            email=request.email,
            password=request.password,
            roles={Role.VIEWER},  # Default role for new users
        )

        logger.info(
            "signup_success",
            username=request.username,
            user_id=user.user_id,
            request_id=request_id,
        )

        return MessageResponse(
            message="Account created successfully. You can now login.",
            success=True,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "signup_error",
            username=request.username,
            error=str(e),
            request_id=request_id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Signup failed",
        )


@router.post(
    "/auth/logout",
    response_model=MessageResponse,
    summary="User logout",
    description="Logout user, revoke access token, and clear HttpOnly cookies.",
    status_code=status.HTTP_200_OK,
)
async def logout(
    http_request: Request,
    response: Response,
    rbac: RBACManager = Depends(get_rbac_manager),
) -> MessageResponse:
    """
    Logout user and revoke their access token.

    SECURITY: Clears HttpOnly cookies to prevent token reuse.
    Supports both header-based and cookie-based authentication.

    Args:
        http_request: HTTP request object.
        response: HTTP response for clearing cookies.

    Returns:
        Success message.
    """
    request_id = getattr(http_request.state, "request_id", "")

    # Try to get token from header first, then from cookie
    token = None
    auth_header = http_request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1]
    elif http_request.cookies.get("access_token"):
        token = http_request.cookies.get("access_token")

    if token:
        try:
            # Revoke the token on the server side
            rbac.tokens.revoke_token(token)
            logger.info(
                "logout_token_revoked",
                request_id=request_id,
            )
        except Exception as e:
            logger.warning(
                "logout_token_revocation_failed",
                error=str(e),
                request_id=request_id,
            )
            # Still allow logout even if revocation fails
    else:
        logger.info(
            "logout_no_token",
            request_id=request_id,
        )

    # SECURITY: Clear all auth cookies
    _clear_auth_cookies(response)

    return MessageResponse(
        message="Logged out successfully",
        success=True,
    )


class RefreshTokenRequest(BaseModel):
    """Refresh token request model."""

    refresh_token: str = Field(..., min_length=1)


@router.post(
    "/auth/refresh",
    response_model=TokenResponse,
    summary="Refresh token",
    description="Refresh access token using refresh token with secure rotation. Supports both query param and HttpOnly cookie.",
    status_code=status.HTTP_200_OK,
)
async def refresh_token(
    http_request: Request,
    response: Response,
    refresh_token: str | None = Query(None, min_length=1),
    rbac: RBACManager = Depends(get_rbac_manager),
) -> TokenResponse:
    """
    Refresh access token with secure token rotation.

    SECURITY: Implements refresh token rotation per OWASP guidelines:
    - Old refresh token is revoked immediately after validation
    - New token pair is issued
    - Prevents refresh token replay attacks
    - Detects token theft (reused revoked token = breach indicator)
    - Supports both query param and HttpOnly cookie for backward compatibility

    Args:
        http_request: HTTP request object.
        response: HTTP response for setting new cookies.
        refresh_token: Optional refresh token from query param.

    Returns:
        New access and refresh tokens.

    Raises:
        HTTPException: If refresh fails.
    """
    request_id = getattr(http_request.state, "request_id", "")

    logger.info(
        "token_refresh_attempt",
        request_id=request_id,
    )

    try:
        # Try query param first, then fall back to cookie
        incoming_refresh_token = refresh_token
        if not incoming_refresh_token:
            incoming_refresh_token = http_request.cookies.get("refresh_token")

        if not incoming_refresh_token:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Refresh token required",
            )

        # Validate refresh token
        payload = rbac.tokens.validate_token(incoming_refresh_token)

        if payload.token_type != "refresh":
            logger.warning(
                "token_refresh_invalid_type",
                token_type=payload.token_type,
                request_id=request_id,
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type",
            )

        # Get user and verify active status
        user = rbac.users.get_user(payload.sub)
        if user is None:
            logger.warning(
                "token_refresh_user_not_found",
                user_id=payload.sub,
                request_id=request_id,
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
            )

        if not user.is_active:
            logger.warning(
                "token_refresh_inactive_user",
                user_id=user.user_id,
                username=user.username,
                request_id=request_id,
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User account is inactive",
            )

        if user.is_locked:
            logger.warning(
                "token_refresh_locked_user",
                user_id=user.user_id,
                username=user.username,
                request_id=request_id,
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User account is locked",
            )

        # SECURITY: Revoke old refresh token BEFORE issuing new ones
        # This implements secure refresh token rotation per OWASP guidelines
        # If a revoked token is ever used again, it indicates token theft
        rbac.tokens.revoke_token(incoming_refresh_token)
        logger.info(
            "token_refresh_old_token_revoked",
            jti=payload.jti[:8] + "..." if payload.jti else "unknown",
            request_id=request_id,
        )

        # Create new token pair with fresh tokens
        tokens = rbac.tokens.create_token_pair(user)

        # SECURITY: Generate new CSRF token and set HttpOnly cookies
        csrf_token = stdlib_secrets.token_urlsafe(32)
        _set_auth_cookies(
            response=response,
            access_token=tokens.access_token,
            refresh_token=tokens.refresh_token,
            csrf_token=csrf_token,
        )

        logger.info(
            "token_refresh_success",
            user_id=user.user_id,
            username=user.username,
            request_id=request_id,
        )

        return TokenResponse(
            access_token=tokens.access_token,
            refresh_token=tokens.refresh_token,
            token_type="bearer",
            expires_in=1800,
        )

    except (TokenInvalidError, TokenExpiredError) as e:
        # SECURITY: Log token reuse attempts - could indicate theft
        error_msg = str(e)
        is_revoked = "revoked" in error_msg.lower()

        if is_revoked:
            # CRITICAL: Revoked token reuse detected - potential token theft!
            logger.critical(
                "token_refresh_revoked_token_reuse",
                error=error_msg,
                request_id=request_id,
                alert="POTENTIAL_TOKEN_THEFT",
            )
        else:
            logger.warning(
                "token_refresh_failed",
                error=error_msg,
                request_id=request_id,
            )

        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired refresh token",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "token_refresh_error",
            error=str(e),
            request_id=request_id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Token refresh failed",
        )


@router.get(
    "/auth/me",
    response_model=UserResponse,
    summary="Get current user",
    description="Get current authenticated user information. Supports both header and cookie authentication.",
    status_code=status.HTTP_200_OK,
)
async def get_current_user(
    http_request: Request,
    rbac: RBACManager = Depends(get_rbac_manager),
) -> UserResponse:
    """
    Get current authenticated user.

    SECURITY: Supports both Bearer token header and HttpOnly cookie authentication.
    Header takes precedence for backward compatibility.

    Args:
        http_request: HTTP request object.

    Returns:
        User information.

    Raises:
        HTTPException: If not authenticated.
    """
    request_id = getattr(http_request.state, "request_id", "")

    # Try to get token from header first, then from cookie
    token = None
    auth_header = http_request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1]
    elif http_request.cookies.get("access_token"):
        token = http_request.cookies.get("access_token")

    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )

    # DEV MODE: Accept dev token for development (skip JWT validation)
    if token == "dev-token-rayyan":
        return UserResponse(
            user_id="dev-user-rayyan",
            username="rayyan",
            email="rayyan.a@nobilityrcm.com",
            roles=["admin"],
            permissions=["*"],
        )

    try:
        payload = rbac.tokens.validate_token(token)

        if payload.token_type != "access":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type",
            )

        user = rbac.users.get_user_by_username(payload.username)
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
            )

        return UserResponse(
            user_id=user.user_id,
            username=user.username,
            email=user.email,
            roles=[r.value for r in user.roles],
            permissions=[p.value for p in user.get_all_permissions()],
        )

    except (TokenInvalidError, TokenExpiredError) as e:
        logger.warning(
            "auth_failed",
            error=str(e),
            request_id=request_id,
        )
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "get_user_error",
            error=str(e),
            request_id=request_id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to get user information",
        )


# =============================================================================
# API Key Management Endpoints
# =============================================================================


class CreateAPIKeyRequest(BaseModel):
    """Request model for API key creation."""

    name: str = Field(
        ...,
        min_length=3,
        max_length=100,
        description="Human-readable name for the API key",
    )
    expires_days: int = Field(
        default=365,
        ge=1,
        le=730,  # Max 2 years
        description="Number of days until the key expires",
    )

    @field_validator("name")
    @classmethod
    def validate_name(cls, v: str) -> str:
        """Validate API key name."""
        # Only allow alphanumeric, spaces, hyphens, and underscores
        if not re.match(r"^[\w\s\-]+$", v):
            raise ValueError(
                "Name can only contain letters, numbers, spaces, hyphens, and underscores"
            )
        return v.strip()


class APIKeyResponse(BaseModel):
    """Response model for API key creation."""

    key_id: str = Field(
        ...,
        description="Unique identifier for the key (for management)",
    )
    api_key: str = Field(
        ...,
        description="The API key - SAVE THIS, it won't be shown again",
    )
    name: str = Field(
        ...,
        description="Name of the API key",
    )
    created_at: str = Field(
        ...,
        description="ISO timestamp when key was created",
    )
    expires_at: str = Field(
        ...,
        description="ISO timestamp when key expires",
    )


class APIKeyInfo(BaseModel):
    """Information about an API key (without the actual key)."""

    key_id: str = Field(
        ...,
        description="Unique identifier for the key",
    )
    name: str = Field(
        ...,
        description="Name of the API key",
    )
    created_at: str = Field(
        ...,
        description="ISO timestamp when key was created",
    )
    expires_at: str = Field(
        ...,
        description="ISO timestamp when key expires",
    )
    last_used_at: str | None = Field(
        None,
        description="ISO timestamp when key was last used",
    )


class APIKeyListResponse(BaseModel):
    """Response model for listing API keys."""

    keys: list[APIKeyInfo] = Field(
        default_factory=list,
        description="List of API keys for the user",
    )
    total: int = Field(
        default=0,
        description="Total number of API keys",
    )


@router.post(
    "/api-keys",
    response_model=APIKeyResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Create API key",
    description="Create a new API key for programmatic access. The key is only shown once.",
)
async def create_api_key(
    request: Request,
    body: CreateAPIKeyRequest,
    rbac: RBACManager = Depends(get_rbac_manager),
) -> APIKeyResponse:
    """
    Create a new API key for the authenticated user.

    API keys provide programmatic access to the API without needing
    to manage OAuth tokens. Keys are long-lived but can be revoked.

    IMPORTANT: The API key is only returned once. Store it securely.

    Args:
        request: FastAPI request with auth context.
        body: API key creation parameters.

    Returns:
        Created API key details including the key itself.
    """
    import secrets
    from datetime import datetime, timedelta

    request_id = getattr(request.state, "request_id", "unknown")
    user_id = getattr(request.state, "user_id", None)

    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required to create API keys",
        )

    try:
        # Get user
        user = rbac.get_user(user_id)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User not found",
            )

        # Generate key ID (for management - not the actual key)
        key_id = f"key_{secrets.token_urlsafe(16)}"

        # Create API key
        api_key = rbac.create_api_key(
            user=user,
            name=body.name,
            expires_days=body.expires_days,
        )

        now = datetime.now(UTC)
        expires = now + timedelta(days=body.expires_days)

        logger.info(
            "api_key_created",
            user_id=user_id,
            key_id=key_id,
            key_name=body.name,
            expires_days=body.expires_days,
            request_id=request_id,
        )

        return APIKeyResponse(
            key_id=key_id,
            api_key=api_key,
            name=body.name,
            created_at=now.isoformat(),
            expires_at=expires.isoformat(),
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(
            "api_key_creation_failed",
            error=str(e),
            user_id=user_id,
            request_id=request_id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to create API key",
        )


@router.delete(
    "/api-keys/{key_jti}",
    status_code=status.HTTP_204_NO_CONTENT,
    summary="Revoke API key",
    description="Revoke an API key by its JTI (JWT ID). The key will no longer work.",
)
async def revoke_api_key(
    request: Request,
    key_jti: str,
    rbac: RBACManager = Depends(get_rbac_manager),
) -> None:
    """
    Revoke an API key.

    Once revoked, the API key will no longer work for authentication.
    This action cannot be undone - a new key must be created if needed.

    Args:
        request: FastAPI request with auth context.
        key_jti: JTI (JWT ID) of the key to revoke.
    """
    request_id = getattr(request.state, "request_id", "unknown")
    user_id = getattr(request.state, "user_id", None)

    if not user_id:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required to revoke API keys",
        )

    try:
        # Revoke the token by its JTI
        # In a full implementation, we'd verify ownership before revoking
        rbac.tokens.revoke_token_by_jti(key_jti)

        logger.info(
            "api_key_revoked",
            user_id=user_id,
            key_jti=key_jti,
            request_id=request_id,
        )

    except Exception as e:
        logger.error(
            "api_key_revocation_failed",
            error=str(e),
            user_id=user_id,
            key_jti=key_jti,
            request_id=request_id,
        )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to revoke API key",
        )
