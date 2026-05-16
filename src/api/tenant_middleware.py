"""V3 Phase 8 — Tenant resolver middleware.

Reads the tenant id from one of (in precedence order):

1. The authenticated user's JWT ``tenant_id`` claim
   (``request.state.user_claims['tenant_id']``).
2. ``X-Tenant-ID`` header *only when* the caller has the
   ``"system:admin"`` permission (admin override for support flows).
3. ``settings.api.default_tenant_id`` (defaults to ``"default"``).

Sets ``request.state.tenant_id`` for every downstream middleware
and route handler. Skips the resolution on public health/docs paths
so static OpenAPI / liveness probes don't pay the cost.

The middleware is gated by ``settings.api.multi_tenant_enabled``
(default ``False``). When off, every request gets ``"default"`` and
the rest of the system sees today's single-tenant behaviour.

Design choices:

* The tenant id is **always set** on ``request.state.tenant_id``
  even when multi-tenancy is disabled. This means downstream
  middleware (rate limiter, orchestrator, audit) can rely on the
  attribute existing without defensive ``getattr`` calls.
* Header-based override is **admin-only**. A non-admin user who
  sends ``X-Tenant-ID: someone-else`` gets their JWT-claim tenant
  back; the header is silently ignored and a structlog warning
  is recorded so support engineers can investigate.
* Skipped routes are exact-match prefixes only. We never let the
  client provide a path that bypasses tenant resolution by
  coincidentally starting with one of these prefixes.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.types import ASGIApp

from src.config import get_logger


logger = get_logger(__name__)


# Routes whose response shape never depends on tenant scope.
DEFAULT_SKIP_PREFIXES: tuple[str, ...] = (
    "/health",
    "/docs",
    "/openapi.json",
    "/redoc",
)


class TenantResolverMiddleware(BaseHTTPMiddleware):
    """Resolve and bind tenant id for every request."""

    def __init__(
        self,
        app: ASGIApp,
        *,
        default_tenant_id: str = "default",
        skip_prefixes: Iterable[str] = DEFAULT_SKIP_PREFIXES,
        admin_permission: str = "system:admin",
        enabled: bool = True,
    ) -> None:
        super().__init__(app)
        self._default = default_tenant_id
        self._skip = tuple(skip_prefixes)
        self._admin_perm = admin_permission
        self._enabled = enabled

    async def dispatch(
        self,
        request: Request,
        call_next: Callable[[Request], Any],
    ) -> Response:
        # Skip public paths — they never carry a tenant decision.
        path = request.url.path
        if any(path.startswith(p) for p in self._skip):
            request.state.tenant_id = self._default
            return await call_next(request)

        # Multi-tenancy disabled → every request is "default".
        if not self._enabled:
            request.state.tenant_id = self._default
            return await call_next(request)

        # 1. JWT claim first.
        claims = getattr(request.state, "user_claims", None) or {}
        claim_tenant = (
            claims.get("tenant_id")
            if isinstance(claims, dict)
            else None
        )
        # 2. Admin-only header override.
        header_tenant = request.headers.get("X-Tenant-ID")
        permissions = getattr(request.state, "permissions", []) or []
        is_admin = self._admin_perm in permissions

        if claim_tenant:
            tenant_id = str(claim_tenant)
        elif header_tenant and is_admin:
            tenant_id = str(header_tenant)
            logger.info(
                "tenant_admin_override",
                tenant_id=tenant_id,
                user_id=getattr(request.state, "user_id", None),
                path=path,
            )
        elif header_tenant and not is_admin:
            # Non-admin tried to set the header — ignore it but log.
            logger.warning(
                "tenant_header_ignored_non_admin",
                requested_tenant=header_tenant,
                user_id=getattr(request.state, "user_id", None),
                path=path,
            )
            tenant_id = self._default
        else:
            tenant_id = self._default

        request.state.tenant_id = tenant_id
        return await call_next(request)
