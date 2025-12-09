"""
Authentication API routes.

Provides endpoints for user authentication, registration, and token management.
"""

import re
from typing import Any, Optional

from fastapi import APIRouter, HTTPException, Request, status
from pydantic import BaseModel, Field, field_validator

from src.config import get_logger
from src.security.rbac import (
    RBACManager,
    Role,
    TokenInvalidError,
    TokenExpiredError,
    AuthorizationError,
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

    MIN_LENGTH = 12
    COMMON_PASSWORDS = {
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
    def validate_password(password: str) -> tuple[bool, Optional[str]]:
        """
        Validate password strength.

        Args:
            password: Password to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        if len(password) < PasswordValidator.MIN_LENGTH:
            return False, f"Password must be at least {PasswordValidator.MIN_LENGTH} characters"

        if password.lower() in PasswordValidator.COMMON_PASSWORDS:
            return False, "Password is too common. Please choose a stronger password"

        # Check complexity requirements
        checks = {
            r"[A-Z]": "at least one uppercase letter",
            r"[a-z]": "at least one lowercase letter",
            r"[0-9]": "at least one number",
            r'[!@#$%^&*(),.?":{}|<>_\-+=\[\]\\\/`~]': "at least one special character",
        }

        missing_requirements = []
        for pattern, requirement in checks.items():
            if not re.search(pattern, password):
                missing_requirements.append(requirement)

        if missing_requirements:
            return False, f"Password must contain {', '.join(missing_requirements)}"

        # Check for sequential characters
        if re.search(r"(012|123|234|345|456|567|678|789|890|abc|bcd|cde|def)", password.lower()):
            return False, "Password contains sequential characters"

        # Check for repeated characters (more than 2 consecutive)
        if re.search(r"(.)\1{2,}", password):
            return False, "Password contains too many repeated characters"

        return True, None


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

        # Validate minimum key strength (32 characters = ~192 bits)
        if len(secret_key) < 32:
            raise ValueError(
                f"JWT_SECRET_KEY must be at least 32 characters (current: {len(secret_key)}). "
                "Generate a stronger key with: python -c 'import secrets; print(secrets.token_urlsafe(64))'"
            )

        _rbac_manager = RBACManager.get_instance(secret_key=secret_key)
    return _rbac_manager


# Request/Response Models
class LoginRequest(BaseModel):
    """Login request model."""

    username: str = Field(..., min_length=3, max_length=50)
    password: str = Field(..., min_length=8)


class SignupRequest(BaseModel):
    """Signup request model."""

    username: str = Field(
        ...,
        min_length=3,
        max_length=50,
        pattern=r"^[a-zA-Z0-9_-]+$",  # Alphanumeric, underscore, hyphen only
    )
    email: str = Field(..., pattern=r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$")
    password: str = Field(..., min_length=12)  # Increased from 8 to 12
    confirm_password: str = Field(..., min_length=12)

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
    description="Authenticate user and return access tokens.",
    status_code=status.HTTP_200_OK,
)
async def login(
    request: LoginRequest,
    http_request: Request,
) -> TokenResponse:
    """
    Authenticate user with username and password.

    Args:
        request: Login credentials.
        http_request: HTTP request object.

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

    try:
        rbac = get_rbac_manager()
        tokens = rbac.authenticate(request.username, request.password)

        if tokens is None:
            logger.warning(
                "login_failed",
                username=request.username,
                request_id=request_id,
            )
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid username or password",
            )

        logger.info(
            "login_success",
            username=request.username,
            request_id=request_id,
        )

        return TokenResponse(
            access_token=tokens.access_token,
            refresh_token=tokens.refresh_token,
            token_type="bearer",
            expires_in=1800,
        )

    except HTTPException:
        raise
    except Exception as e:
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
        rbac = get_rbac_manager()

        # Check if username already exists
        # Use constant-time check to prevent account enumeration
        import asyncio
        import time

        start_check = time.perf_counter()
        existing_user = rbac.users.get_user_by_username(request.username)

        if existing_user is not None:
            # Add constant-time delay to prevent timing attacks
            elapsed = time.perf_counter() - start_check
            await asyncio.sleep(max(0, 0.5 - elapsed))

            logger.warning(
                "signup_failed_username_exists",
                username=request.username,
                request_id=request_id,
            )
            # Generic error message to prevent account enumeration
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid registration data. Please check your inputs.",
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
    description="Logout user and revoke access token.",
    status_code=status.HTTP_200_OK,
)
async def logout(
    http_request: Request,
) -> MessageResponse:
    """
    Logout user and revoke their access token.

    Args:
        http_request: HTTP request object.

    Returns:
        Success message.
    """
    request_id = getattr(http_request.state, "request_id", "")

    # Extract and revoke the access token
    auth_header = http_request.headers.get("Authorization")
    if auth_header and auth_header.startswith("Bearer "):
        token = auth_header.split(" ")[1]
        try:
            rbac = get_rbac_manager()
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

    return MessageResponse(
        message="Logged out successfully",
        success=True,
    )


@router.post(
    "/auth/refresh",
    response_model=TokenResponse,
    summary="Refresh token",
    description="Refresh access token using refresh token.",
    status_code=status.HTTP_200_OK,
)
async def refresh_token(
    refresh_token: str,
    http_request: Request,
) -> TokenResponse:
    """
    Refresh access token.

    Args:
        refresh_token: Refresh token.
        http_request: HTTP request object.

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
        rbac = get_rbac_manager()

        # Validate refresh token
        payload = rbac.tokens.validate_token(refresh_token)

        if payload.token_type != "refresh":
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid token type",
            )

        # Get user
        user = rbac.users.get_user_by_username(payload.username)
        if user is None:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="User not found",
            )

        # Create new token pair
        tokens = rbac.tokens.create_token_pair(user)

        logger.info(
            "token_refresh_success",
            username=payload.username,
            request_id=request_id,
        )

        return TokenResponse(
            access_token=tokens.access_token,
            refresh_token=tokens.refresh_token,
            token_type="bearer",
            expires_in=1800,
        )

    except (TokenInvalidError, TokenExpiredError) as e:
        logger.warning(
            "token_refresh_failed",
            error=str(e),
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
    description="Get current authenticated user information.",
    status_code=status.HTTP_200_OK,
)
async def get_current_user(
    http_request: Request,
) -> UserResponse:
    """
    Get current authenticated user.

    Args:
        http_request: HTTP request object.

    Returns:
        User information.

    Raises:
        HTTPException: If not authenticated.
    """
    request_id = getattr(http_request.state, "request_id", "")

    # Get authorization header
    auth_header = http_request.headers.get("Authorization")
    if not auth_header or not auth_header.startswith("Bearer "):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
        )

    token = auth_header.split(" ")[1]

    try:
        rbac = get_rbac_manager()
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
