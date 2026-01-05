"""
API Middleware for Security, Metrics, and Audit Logging.

Provides comprehensive middleware for the FastAPI application including:
- Authentication and authorization
- Request/response metrics collection
- Audit logging for HIPAA compliance
- Rate limiting
- Security headers
"""

from __future__ import annotations

import time
import uuid
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any

import structlog
from fastapi import HTTPException, Request, Response
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from src.monitoring.metrics import MetricsCollector
from src.security.audit import (
    AuditLogger,
    AuditOutcome,
)
from src.security.rbac import (
    Permission,
    RBACManager,
    TokenError,
    TokenExpiredError,
    TokenPayload,
)


logger = structlog.get_logger(__name__)


# ============================================================================
# Secure Client IP Extraction (HIGH-009 Fix)
# ============================================================================

# Trusted proxy IP ranges - only trust X-Forwarded-For from these IPs
# Configure based on your infrastructure (load balancers, reverse proxies)
TRUSTED_PROXY_RANGES = {
    "127.0.0.1",  # localhost
    "10.0.0.0/8",  # Private network
    "172.16.0.0/12",  # Private network
    "192.168.0.0/16",  # Private network
}


def _is_trusted_proxy(client_ip: str) -> bool:
    """
    Check if the request comes from a trusted proxy.

    Args:
        client_ip: Direct client IP from the connection.

    Returns:
        True if the IP is in the trusted proxy list.
    """
    import ipaddress

    if not client_ip or client_ip == "unknown":
        return False

    try:
        ip = ipaddress.ip_address(client_ip)

        for trusted in TRUSTED_PROXY_RANGES:
            if "/" in trusted:
                # Network range
                if ip in ipaddress.ip_network(trusted, strict=False):
                    return True
            # Single IP
            elif ip == ipaddress.ip_address(trusted):
                return True

        return False
    except ValueError:
        # Invalid IP address format
        return False


def _validate_ip_format(ip_str: str) -> bool:
    """
    Validate that a string is a properly formatted IP address.

    Args:
        ip_str: String to validate.

    Returns:
        True if valid IP address format.
    """
    import ipaddress

    try:
        ipaddress.ip_address(ip_str.strip())
        return True
    except ValueError:
        return False


def get_secure_client_ip(request: Request, trust_proxy: bool = True) -> str:
    """
    Securely extract client IP address from request.

    Security measures:
    - Only trusts X-Forwarded-For when behind a known proxy
    - Validates IP address format
    - Logs suspicious header manipulation attempts
    - Falls back to direct connection IP when not behind proxy

    Args:
        request: FastAPI request object.
        trust_proxy: Whether to trust X-Forwarded-For headers at all.

    Returns:
        Client IP address string.
    """
    # Get direct connection IP
    direct_ip = request.client.host if request.client else "unknown"

    if not trust_proxy:
        return direct_ip

    # Only trust X-Forwarded-For if request comes from a trusted proxy
    if not _is_trusted_proxy(direct_ip):
        # Request not from trusted proxy - log if X-Forwarded-For was attempted
        forwarded = request.headers.get("X-Forwarded-For")
        if forwarded:
            logger.warning(
                "untrusted_forwarded_header",
                direct_ip=direct_ip,
                forwarded_for=forwarded[:100],  # Truncate to prevent log injection
                message="X-Forwarded-For header ignored - request not from trusted proxy",
            )
        return direct_ip

    # Request from trusted proxy - extract and validate X-Forwarded-For
    forwarded = request.headers.get("X-Forwarded-For")
    if not forwarded:
        return direct_ip

    # X-Forwarded-For format: "client, proxy1, proxy2, ..."
    # The leftmost (first) IP is the original client
    ips = [ip.strip() for ip in forwarded.split(",")]

    if not ips:
        return direct_ip

    client_ip = ips[0]

    # Validate the IP format to prevent injection attacks
    if not _validate_ip_format(client_ip):
        logger.warning(
            "invalid_forwarded_ip_format",
            invalid_ip=client_ip[:50],  # Truncate
            direct_ip=direct_ip,
            message="X-Forwarded-For contains invalid IP format",
        )
        return direct_ip

    return client_ip


@dataclass(slots=True)
class RateLimitConfig:
    """Rate limiting configuration."""

    requests_per_minute: int = 60
    burst_size: int = 10


@dataclass
class RateLimitState:
    """Rate limiting state for a client."""

    requests: list[float] = field(default_factory=list)
    last_cleanup: float = 0.0


class RateLimiter:
    """
    Token bucket rate limiter.

    Provides per-client rate limiting with configurable limits.
    """

    def __init__(
        self,
        default_rpm: int = 60,
        burst_size: int = 10,
    ) -> None:
        """
        Initialize rate limiter.

        Args:
            default_rpm: Default requests per minute.
            burst_size: Maximum burst size.
        """
        self._default_rpm = default_rpm
        self._burst_size = burst_size
        self._clients: dict[str, RateLimitState] = defaultdict(RateLimitState)
        self._endpoint_limits: dict[str, RateLimitConfig] = {}

    def set_endpoint_limit(
        self,
        endpoint: str,
        rpm: int,
        burst: int | None = None,
    ) -> None:
        """Set rate limit for a specific endpoint."""
        self._endpoint_limits[endpoint] = RateLimitConfig(
            requests_per_minute=rpm,
            burst_size=burst or max(rpm // 6, 1),
        )

    def is_allowed(
        self,
        client_id: str,
        endpoint: str | None = None,
    ) -> tuple[bool, dict[str, int]]:
        """
        Check if request is allowed.

        Args:
            client_id: Client identifier (IP or user ID).
            endpoint: Endpoint path for specific limits.

        Returns:
            Tuple of (allowed, headers_dict).
        """
        now = time.time()
        window_start = now - 60  # 1 minute window

        # Get config
        config = self._endpoint_limits.get(endpoint or "", None)
        rpm = config.requests_per_minute if config else self._default_rpm

        # Get/create client state
        state = self._clients[client_id]

        # Cleanup old requests
        if now - state.last_cleanup > 10:
            state.requests = [t for t in state.requests if t > window_start]
            state.last_cleanup = now

        # Check limit
        remaining = max(0, rpm - len(state.requests))
        headers = {
            "X-RateLimit-Limit": rpm,
            "X-RateLimit-Remaining": remaining,
            "X-RateLimit-Reset": int(window_start + 60),
        }

        if len(state.requests) >= rpm:
            return False, headers

        # Record request
        state.requests.append(now)
        headers["X-RateLimit-Remaining"] = remaining - 1

        return True, headers


class SecurityHeadersMiddleware(BaseHTTPMiddleware):
    """
    Middleware to add security headers to all responses.

    Implements OWASP recommended security headers.
    """

    SECURITY_HEADERS = {
        "X-Content-Type-Options": "nosniff",
        "X-Frame-Options": "DENY",
        "X-XSS-Protection": "1; mode=block",
        "Strict-Transport-Security": "max-age=31536000; includeSubDomains; preload",
        "Cache-Control": "no-store, no-cache, must-revalidate",
        "Pragma": "no-cache",
        "Referrer-Policy": "strict-origin-when-cross-origin",
        "Permissions-Policy": "geolocation=(), microphone=(), camera=()",
        # Content Security Policy - defense in depth against XSS
        "Content-Security-Policy": (
            "default-src 'self'; "
            "script-src 'self'; "
            "style-src 'self' 'unsafe-inline'; "
            "img-src 'self' data: https:; "
            "font-src 'self'; "
            "connect-src 'self'; "
            "frame-ancestors 'none'; "
            "base-uri 'self'; "
            "form-action 'self'; "
            "object-src 'none'"
        ),
    }

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        """Add security headers to response."""
        response = await call_next(request)

        for header, value in self.SECURITY_HEADERS.items():
            response.headers[header] = value

        return response


class MetricsMiddleware(BaseHTTPMiddleware):
    """
    Middleware to collect request metrics for Prometheus.

    Records request duration, count, and size metrics.
    """

    def __init__(self, app: Any) -> None:
        """Initialize metrics middleware."""
        super().__init__(app)
        self._collector = MetricsCollector()

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        """Collect metrics for request."""
        method = request.method
        path = self._normalize_path(request.url.path)
        start_time = time.perf_counter()

        # Track in-progress requests
        self._collector._registry.api_requests_in_progress.labels(
            method=method,
            endpoint=path,
        ).inc()

        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception:
            status_code = 500
            raise
        finally:
            duration = time.perf_counter() - start_time

            # Record metrics
            self._collector.record_api_request(
                method=method,
                endpoint=path,
                status_code=status_code,
                duration=duration,
                request_size=int(request.headers.get("content-length", 0)),
            )

            # Decrement in-progress
            self._collector._registry.api_requests_in_progress.labels(
                method=method,
                endpoint=path,
            ).dec()

        return response

    def _normalize_path(self, path: str) -> str:
        """Normalize path by replacing dynamic segments."""
        # Replace UUIDs
        import re

        path = re.sub(
            r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}",
            "{id}",
            path,
        )
        # Replace numeric IDs
        path = re.sub(r"/\d+(?=/|$)", "/{id}", path)
        return path


class AuditMiddleware(BaseHTTPMiddleware):
    """
    Middleware for HIPAA-compliant audit logging.

    Logs all API requests with appropriate context.
    """

    def __init__(
        self,
        app: Any,
        log_dir: str = "./logs/audit",
        mask_phi: bool = True,
    ) -> None:
        """
        Initialize audit middleware.

        Args:
            app: FastAPI application.
            log_dir: Audit log directory.
            mask_phi: Enable PHI masking.
        """
        super().__init__(app)
        self._audit_logger = AuditLogger(
            log_dir=log_dir,
            mask_phi=mask_phi,
        )

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        """Audit log the request."""
        request_id = getattr(request.state, "request_id", str(uuid.uuid4()))
        user_id = getattr(request.state, "user_id", None)
        client_ip = self._get_client_ip(request)
        start_time = time.perf_counter()

        # Set audit context
        self._audit_logger.set_context(
            request_id=request_id,
            user_id=user_id,
            client_ip=client_ip,
        )

        try:
            response = await call_next(request)
            outcome = AuditOutcome.SUCCESS if response.status_code < 400 else AuditOutcome.FAILURE
        except Exception:
            outcome = AuditOutcome.FAILURE
            raise
        finally:
            duration_ms = (time.perf_counter() - start_time) * 1000

            # Log API request
            self._audit_logger.log_api_request(
                method=request.method,
                path=request.url.path,
                status_code=response.status_code if "response" in locals() else 500,
                duration_ms=duration_ms,
                client_ip=client_ip,
                user_id=user_id,
                query_params=str(request.query_params) if request.query_params else None,
            )

            # Clear context
            self._audit_logger.clear_context()

        return response

    def _get_client_ip(self, request: Request) -> str:
        """
        Get client IP address from request securely.

        Uses secure IP extraction that validates X-Forwarded-For
        only when behind a trusted proxy.
        """
        return get_secure_client_ip(request, trust_proxy=True)


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """
    Middleware for JWT token authentication.

    Validates tokens and populates request context with user info.
    """

    # Paths that don't require authentication
    PUBLIC_PATHS = {
        "/",
        "/docs",
        "/redoc",
        "/openapi.json",
        "/api/v1/health",
        "/api/v1/health/ready",
        "/api/v1/health/live",
        "/metrics",
    }

    def __init__(
        self,
        app: Any,
        rbac_manager: RBACManager | None = None,
        api_key_header: str = "X-API-Key",
        bearer_header: str = "Authorization",
    ) -> None:
        """
        Initialize authentication middleware.

        Args:
            app: FastAPI application.
            rbac_manager: RBAC manager instance.
            api_key_header: Header name for API key.
            bearer_header: Header name for Bearer token.
        """
        super().__init__(app)
        self._rbac = rbac_manager
        self._api_key_header = api_key_header
        self._bearer_header = bearer_header

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        """Authenticate the request."""
        path = request.url.path

        # Skip authentication for public paths
        if self._is_public_path(path):
            return await call_next(request)

        # Skip if no RBAC manager configured
        if self._rbac is None:
            return await call_next(request)

        # Try to authenticate
        try:
            payload = await self._authenticate(request)
            if payload:
                request.state.user_id = payload.sub
                request.state.username = payload.username
                request.state.roles = payload.roles
                request.state.permissions = payload.permissions
        except TokenExpiredError:
            return JSONResponse(
                status_code=401,
                content={
                    "error": "token_expired",
                    "message": "Authentication token has expired",
                    "timestamp": datetime.now(UTC).isoformat(),
                },
            )
        except TokenError as e:
            return JSONResponse(
                status_code=401,
                content={
                    "error": "authentication_failed",
                    "message": str(e),
                    "timestamp": datetime.now(UTC).isoformat(),
                },
            )

        return await call_next(request)

    def _is_public_path(self, path: str) -> bool:
        """Check if path is public."""
        if path in self.PUBLIC_PATHS:
            return True
        # Check prefixes
        public_prefixes = ["/docs", "/redoc", "/openapi"]
        return any(path.startswith(p) for p in public_prefixes)

    async def _authenticate(self, request: Request) -> TokenPayload | None:
        """
        Authenticate request using token or API key.

        Args:
            request: FastAPI request.

        Returns:
            Token payload if authenticated.
        """
        # Try Bearer token
        auth_header = request.headers.get(self._bearer_header)
        if auth_header and auth_header.startswith("Bearer "):
            token = auth_header[7:]

            # DEV MODE: Accept dev token for development (skip JWT validation)
            if token == "dev-token-rayyan":
                now = datetime.now(UTC)
                return TokenPayload(
                    sub="dev-user-rayyan",
                    username="rayyan",
                    roles=["admin"],
                    permissions=["*"],  # All permissions for dev
                    exp=now + timedelta(days=365),
                    iat=now,
                    jti="dev-token-jti",
                    token_type="access",
                )

            return self._rbac.tokens.validate_token(token)

        # Try API key
        api_key = request.headers.get(self._api_key_header)
        if api_key:
            return self._rbac.tokens.validate_token(api_key)

        return None


class RateLimitMiddleware(BaseHTTPMiddleware):
    """
    Middleware for rate limiting.

    Implements per-client and per-endpoint rate limiting.
    """

    def __init__(
        self,
        app: Any,
        default_rpm: int = 60,
        burst_size: int = 10,
    ) -> None:
        """
        Initialize rate limit middleware.

        Args:
            app: FastAPI application.
            default_rpm: Default requests per minute.
            burst_size: Maximum burst size.
        """
        super().__init__(app)
        self._limiter = RateLimiter(default_rpm, burst_size)

        # Rate limits for authentication endpoints (SECURITY: strict limits to prevent brute force)
        self._limiter.set_endpoint_limit(
            "/api/v1/auth/login", 5, 2
        )  # 5 attempts/min (strict for auth)
        self._limiter.set_endpoint_limit(
            "/api/v1/auth/signup", 3, 1
        )  # 3 attempts/min (prevent enumeration)
        self._limiter.set_endpoint_limit("/api/v1/auth/refresh", 30, 5)  # 30/min
        self._limiter.set_endpoint_limit("/api/v1/auth/me", 60, 10)  # 60/min

        # Set endpoint-specific limits for documents
        self._limiter.set_endpoint_limit("/api/v1/documents/process", 10, 2)
        self._limiter.set_endpoint_limit("/api/v1/documents/batch", 5, 1)
        self._limiter.set_endpoint_limit("/api/v1/documents/export", 30, 5)

    async def dispatch(
        self,
        request: Request,
        call_next: RequestResponseEndpoint,
    ) -> Response:
        """Apply rate limiting."""
        # Skip rate limiting for OPTIONS preflight requests (required for CORS)
        if request.method == "OPTIONS":
            return await call_next(request)

        client_id = self._get_client_id(request)
        path = request.url.path

        allowed, headers = self._limiter.is_allowed(client_id, path)

        if not allowed:
            return JSONResponse(
                status_code=429,
                content={
                    "error": "rate_limit_exceeded",
                    "message": "Too many requests",
                    "retry_after": headers.get("X-RateLimit-Reset", 60),
                    "timestamp": datetime.now(UTC).isoformat(),
                },
                headers={k: str(v) for k, v in headers.items()},
            )

        response = await call_next(request)

        # Add rate limit headers to response
        for key, value in headers.items():
            response.headers[key] = str(value)

        return response

    def _get_client_id(self, request: Request) -> str:
        """
        Get client identifier for rate limiting.

        Uses secure IP extraction to prevent rate limit bypass via
        X-Forwarded-For header spoofing.
        """
        # Use user ID if authenticated (most reliable)
        user_id = getattr(request.state, "user_id", None)
        if user_id:
            return f"user:{user_id}"

        # Fall back to secure IP address extraction
        client_ip = get_secure_client_ip(request, trust_proxy=True)
        return f"ip:{client_ip}"


def get_current_user(request: Request) -> dict[str, Any] | None:
    """
    Get current user from request state.

    Args:
        request: FastAPI request.

    Returns:
        User info dict or None.
    """
    user_id = getattr(request.state, "user_id", None)
    if not user_id:
        return None

    return {
        "user_id": user_id,
        "username": getattr(request.state, "username", None),
        "roles": getattr(request.state, "roles", []),
        "permissions": getattr(request.state, "permissions", []),
    }


def require_permission(permission: Permission) -> Callable:
    """
    Dependency for requiring a specific permission.

    Args:
        permission: Required permission.

    Returns:
        FastAPI dependency.
    """

    async def dependency(request: Request) -> None:
        permissions = getattr(request.state, "permissions", [])
        if permission.value not in permissions:
            raise HTTPException(
                status_code=403,
                detail={
                    "error": "forbidden",
                    "message": f"Missing required permission: {permission.value}",
                },
            )

    return dependency
