"""
Role-Based Access Control (RBAC) Module.

Provides comprehensive RBAC implementation with hierarchical roles,
fine-grained permissions, and JWT token management for HIPAA compliance.
"""

from __future__ import annotations

import hashlib
import secrets
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum, auto
from functools import wraps
from typing import Any, Callable, TypeVar

import structlog
from jose import ExpiredSignatureError, JWTError, jwt
from passlib.context import CryptContext

logger = structlog.get_logger(__name__)


class Permission(str, Enum):
    """System permissions for access control."""

    # Document permissions
    DOCUMENT_READ = "document:read"
    DOCUMENT_CREATE = "document:create"
    DOCUMENT_UPDATE = "document:update"
    DOCUMENT_DELETE = "document:delete"
    DOCUMENT_EXPORT = "document:export"
    DOCUMENT_PROCESS = "document:process"

    # PHI permissions
    PHI_VIEW = "phi:view"
    PHI_EXPORT = "phi:export"
    PHI_MODIFY = "phi:modify"
    PHI_DELETE = "phi:delete"

    # User management permissions
    USER_READ = "user:read"
    USER_CREATE = "user:create"
    USER_UPDATE = "user:update"
    USER_DELETE = "user:delete"
    USER_MANAGE_ROLES = "user:manage_roles"

    # System permissions
    SYSTEM_ADMIN = "system:admin"
    SYSTEM_CONFIG = "system:config"
    SYSTEM_AUDIT_READ = "system:audit_read"
    SYSTEM_METRICS = "system:metrics"

    # API permissions
    API_ACCESS = "api:access"
    API_BATCH = "api:batch"
    API_WEBHOOK = "api:webhook"

    # Export permissions
    EXPORT_JSON = "export:json"
    EXPORT_EXCEL = "export:excel"
    EXPORT_MARKDOWN = "export:markdown"
    EXPORT_ALL = "export:all"


class Role(str, Enum):
    """System roles with hierarchical permissions."""

    ADMIN = "admin"
    MANAGER = "manager"
    ANALYST = "analyst"
    PROCESSOR = "processor"
    VIEWER = "viewer"
    API_USER = "api_user"
    AUDITOR = "auditor"


# Role permission mappings
ROLE_PERMISSIONS: dict[Role, set[Permission]] = {
    Role.ADMIN: set(Permission),  # All permissions
    Role.MANAGER: {
        Permission.DOCUMENT_READ,
        Permission.DOCUMENT_CREATE,
        Permission.DOCUMENT_UPDATE,
        Permission.DOCUMENT_DELETE,
        Permission.DOCUMENT_EXPORT,
        Permission.DOCUMENT_PROCESS,
        Permission.PHI_VIEW,
        Permission.PHI_EXPORT,
        Permission.USER_READ,
        Permission.USER_CREATE,
        Permission.USER_UPDATE,
        Permission.SYSTEM_METRICS,
        Permission.API_ACCESS,
        Permission.API_BATCH,
        Permission.EXPORT_JSON,
        Permission.EXPORT_EXCEL,
        Permission.EXPORT_MARKDOWN,
        Permission.EXPORT_ALL,
    },
    Role.ANALYST: {
        Permission.DOCUMENT_READ,
        Permission.DOCUMENT_CREATE,
        Permission.DOCUMENT_UPDATE,
        Permission.DOCUMENT_EXPORT,
        Permission.DOCUMENT_PROCESS,
        Permission.PHI_VIEW,
        Permission.PHI_EXPORT,
        Permission.API_ACCESS,
        Permission.EXPORT_JSON,
        Permission.EXPORT_EXCEL,
        Permission.EXPORT_MARKDOWN,
    },
    Role.PROCESSOR: {
        Permission.DOCUMENT_READ,
        Permission.DOCUMENT_CREATE,
        Permission.DOCUMENT_PROCESS,
        Permission.PHI_VIEW,
        Permission.API_ACCESS,
        Permission.EXPORT_JSON,
    },
    Role.VIEWER: {
        Permission.DOCUMENT_READ,
        Permission.PHI_VIEW,
        Permission.API_ACCESS,
    },
    Role.API_USER: {
        Permission.DOCUMENT_READ,
        Permission.DOCUMENT_CREATE,
        Permission.DOCUMENT_PROCESS,
        Permission.DOCUMENT_EXPORT,
        Permission.API_ACCESS,
        Permission.API_BATCH,
        Permission.EXPORT_JSON,
        Permission.EXPORT_EXCEL,
    },
    Role.AUDITOR: {
        Permission.DOCUMENT_READ,
        Permission.SYSTEM_AUDIT_READ,
        Permission.SYSTEM_METRICS,
        Permission.USER_READ,
    },
}


class AuthenticationError(Exception):
    """Exception raised for authentication failures."""


class AuthorizationError(Exception):
    """Exception raised for authorization failures."""


class TokenError(Exception):
    """Exception raised for token-related errors."""


class TokenExpiredError(TokenError):
    """Exception raised when token has expired."""


class TokenInvalidError(TokenError):
    """Exception raised when token is invalid."""


@dataclass(slots=True)
class User:
    """User model with RBAC attributes."""

    user_id: str
    username: str
    email: str
    password_hash: str
    roles: set[Role] = field(default_factory=set)
    permissions: set[Permission] = field(default_factory=set)
    is_active: bool = True
    is_locked: bool = False
    failed_login_attempts: int = 0
    last_login: datetime | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: dict[str, Any] = field(default_factory=dict)

    def get_all_permissions(self) -> set[Permission]:
        """Get all permissions from roles and direct assignments."""
        all_perms = set(self.permissions)
        for role in self.roles:
            all_perms.update(ROLE_PERMISSIONS.get(role, set()))
        return all_perms

    def has_permission(self, permission: Permission) -> bool:
        """Check if user has a specific permission."""
        return permission in self.get_all_permissions()

    def has_any_permission(self, permissions: set[Permission]) -> bool:
        """Check if user has any of the specified permissions."""
        return bool(permissions & self.get_all_permissions())

    def has_all_permissions(self, permissions: set[Permission]) -> bool:
        """Check if user has all specified permissions."""
        return permissions <= self.get_all_permissions()

    def has_role(self, role: Role) -> bool:
        """Check if user has a specific role."""
        return role in self.roles

    def to_dict(self) -> dict[str, Any]:
        """Convert user to dictionary (excluding sensitive data)."""
        return {
            "user_id": self.user_id,
            "username": self.username,
            "email": self.email,
            "roles": [r.value for r in self.roles],
            "permissions": [p.value for p in self.permissions],
            "is_active": self.is_active,
            "is_locked": self.is_locked,
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
        }


@dataclass(slots=True)
class TokenPayload:
    """JWT token payload."""

    sub: str  # Subject (user_id)
    username: str
    roles: list[str]
    permissions: list[str]
    exp: datetime
    iat: datetime
    jti: str  # JWT ID for revocation
    token_type: str = "access"

    def to_dict(self) -> dict[str, Any]:
        """Convert payload to dictionary for JWT encoding."""
        return {
            "sub": self.sub,
            "username": self.username,
            "roles": self.roles,
            "permissions": self.permissions,
            "exp": int(self.exp.timestamp()),
            "iat": int(self.iat.timestamp()),
            "jti": self.jti,
            "token_type": self.token_type,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TokenPayload:
        """Create payload from dictionary."""
        return cls(
            sub=data["sub"],
            username=data.get("username", ""),
            roles=data.get("roles", []),
            permissions=data.get("permissions", []),
            exp=datetime.fromtimestamp(data["exp"], tz=timezone.utc),
            iat=datetime.fromtimestamp(data["iat"], tz=timezone.utc),
            jti=data.get("jti", ""),
            token_type=data.get("token_type", "access"),
        )


@dataclass(slots=True)
class TokenPair:
    """Access and refresh token pair."""

    access_token: str
    refresh_token: str
    access_expires_at: datetime
    refresh_expires_at: datetime
    token_type: str = "Bearer"


class PasswordManager:
    """
    Secure password hashing and verification.

    Uses bcrypt with proper work factor for HIPAA compliance.
    """

    def __init__(self, schemes: list[str] | None = None) -> None:
        """
        Initialize password manager.

        Args:
            schemes: Password hashing schemes (default: bcrypt).
        """
        schemes = schemes or ["bcrypt"]
        self._context = CryptContext(
            schemes=schemes,
            deprecated="auto",
            bcrypt__rounds=14,  # OWASP 2024 recommendation (increased from 12)
            bcrypt__ident="2b",  # Use modern bcrypt variant
        )

    def hash_password(self, password: str) -> str:
        """
        Hash a password.

        Args:
            password: Plain text password.

        Returns:
            Hashed password.
        """
        return self._context.hash(password)

    def verify_password(self, password: str, password_hash: str) -> bool:
        """
        Verify a password against its hash.

        Args:
            password: Plain text password.
            password_hash: Stored password hash.

        Returns:
            True if password matches.
        """
        try:
            return self._context.verify(password, password_hash)
        except Exception:
            return False

    def needs_rehash(self, password_hash: str) -> bool:
        """
        Check if password hash needs to be updated.

        Args:
            password_hash: Current password hash.

        Returns:
            True if rehash is needed.
        """
        return self._context.needs_update(password_hash)


class TokenManager:
    """
    JWT token management with secure signing and validation.

    Supports access and refresh tokens with persistent revocation capability.
    Revoked tokens are stored in a file and survive server restarts.

    For production with multiple servers, use Redis instead:
        redis_client = redis.Redis(host='localhost', port=6379, db=0)
        # Store with TTL: redis_client.setex(f"revoked:{jti}", ttl_seconds, "1")
        # Check: redis_client.exists(f"revoked:{jti}")
    """

    def __init__(
        self,
        secret_key: str,
        algorithm: str = "HS256",
        access_token_expire_minutes: int = 30,
        refresh_token_expire_days: int = 7,
        revocation_storage_path: str | None = None,
    ) -> None:
        """
        Initialize token manager with persistent revocation storage.

        Args:
            secret_key: Secret key for JWT signing.
            algorithm: JWT algorithm.
            access_token_expire_minutes: Access token lifetime.
            refresh_token_expire_days: Refresh token lifetime.
            revocation_storage_path: Path to file for storing revoked tokens.
        """
        import json
        import os
        from pathlib import Path as FilePath

        self._secret_key = secret_key
        self._algorithm = algorithm
        self._access_expire = timedelta(minutes=access_token_expire_minutes)
        self._refresh_expire = timedelta(days=refresh_token_expire_days)

        # SECURITY: Persistent token revocation with file-based storage
        # NOTE: For multi-server deployments, use Redis with TTL instead
        if revocation_storage_path is None:
            revocation_storage_path = os.path.join(
                os.path.dirname(__file__), "..", "..", "data", "revoked_tokens.json"
            )
        self._revocation_path = FilePath(revocation_storage_path).resolve()
        self._revocation_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing revoked tokens from file
        self._revoked_tokens: dict[str, float] = {}  # jti -> expiration timestamp
        self._load_revoked_tokens()

        # Clean up expired revocations on startup
        self._cleanup_expired_revocations()

    def create_access_token(self, user: User) -> tuple[str, datetime]:
        """
        Create an access token for a user.

        Args:
            user: User to create token for.

        Returns:
            Tuple of (token, expiration).
        """
        now = datetime.now(timezone.utc)
        expires = now + self._access_expire

        payload = TokenPayload(
            sub=user.user_id,
            username=user.username,
            roles=[r.value for r in user.roles],
            permissions=[p.value for p in user.get_all_permissions()],
            exp=expires,
            iat=now,
            jti=secrets.token_urlsafe(32),
            token_type="access",
        )

        token = jwt.encode(
            payload.to_dict(),
            self._secret_key,
            algorithm=self._algorithm,
        )

        return token, expires

    def create_refresh_token(self, user: User) -> tuple[str, datetime]:
        """
        Create a refresh token for a user.

        Args:
            user: User to create token for.

        Returns:
            Tuple of (token, expiration).
        """
        now = datetime.now(timezone.utc)
        expires = now + self._refresh_expire

        payload = TokenPayload(
            sub=user.user_id,
            username=user.username,
            roles=[],
            permissions=[],
            exp=expires,
            iat=now,
            jti=secrets.token_urlsafe(32),
            token_type="refresh",
        )

        token = jwt.encode(
            payload.to_dict(),
            self._secret_key,
            algorithm=self._algorithm,
        )

        return token, expires

    def create_token_pair(self, user: User) -> TokenPair:
        """
        Create an access/refresh token pair.

        Args:
            user: User to create tokens for.

        Returns:
            TokenPair with both tokens.
        """
        access_token, access_expires = self.create_access_token(user)
        refresh_token, refresh_expires = self.create_refresh_token(user)

        return TokenPair(
            access_token=access_token,
            refresh_token=refresh_token,
            access_expires_at=access_expires,
            refresh_expires_at=refresh_expires,
        )

    def validate_token(self, token: str) -> TokenPayload:
        """
        Validate and decode a JWT token.

        Args:
            token: JWT token string.

        Returns:
            TokenPayload if valid.

        Raises:
            TokenExpiredError: If token has expired.
            TokenInvalidError: If token is invalid or revoked.
        """
        try:
            payload = jwt.decode(
                token,
                self._secret_key,
                algorithms=[self._algorithm],
            )

            token_payload = TokenPayload.from_dict(payload)

            # Check if token is revoked (with cleanup of expired revocations)
            if self.is_revoked(token_payload.jti):
                raise TokenInvalidError("Token has been revoked")

            return token_payload

        except ExpiredSignatureError as e:
            raise TokenExpiredError("Token has expired") from e
        except JWTError as e:
            raise TokenInvalidError(f"Invalid token: {e}") from e

    def _load_revoked_tokens(self) -> None:
        """Load revoked tokens from persistent storage."""
        import json

        if not self._revocation_path.exists():
            logger.info("revocation_store_initialized", path=str(self._revocation_path))
            return

        try:
            with open(self._revocation_path, "r") as f:
                data = json.load(f)
            self._revoked_tokens = {k: float(v) for k, v in data.get("revoked", {}).items()}
            logger.info("revoked_tokens_loaded", count=len(self._revoked_tokens))
        except Exception as e:
            logger.error("revoked_tokens_load_error", error=str(e))
            self._revoked_tokens = {}

    def _save_revoked_tokens(self) -> None:
        """Save revoked tokens to persistent storage."""
        import json

        try:
            with open(self._revocation_path, "w") as f:
                json.dump({"revoked": self._revoked_tokens}, f, indent=2)
            logger.debug("revoked_tokens_saved", count=len(self._revoked_tokens))
        except Exception as e:
            logger.error("revoked_tokens_save_error", error=str(e))

    def _cleanup_expired_revocations(self) -> int:
        """
        Remove expired token revocations from storage.

        Returns:
            Number of entries cleaned up.
        """
        now = datetime.now(timezone.utc).timestamp()
        expired = [jti for jti, exp_time in self._revoked_tokens.items() if exp_time < now]

        for jti in expired:
            del self._revoked_tokens[jti]

        if expired:
            self._save_revoked_tokens()
            logger.info("revoked_tokens_cleanup", removed=len(expired))

        return len(expired)

    def revoke_token(self, token: str) -> None:
        """
        Revoke a token with persistent storage.

        Args:
            token: Token to revoke.
        """
        try:
            # Decode without verification to get JTI and expiration
            payload = jwt.decode(
                token,
                self._secret_key,
                algorithms=[self._algorithm],
                options={"verify_exp": False},
            )
            jti = payload.get("jti", "")
            exp = payload.get("exp", 0)

            if jti:
                # Store with expiration timestamp for cleanup
                self._revoked_tokens[jti] = float(exp)
                self._save_revoked_tokens()
                logger.info("token_revoked", jti=jti[:8] + "...")
        except JWTError as e:
            logger.warning("token_revocation_failed", error=str(e))

    def is_revoked(self, jti: str) -> bool:
        """
        Check if a token ID is revoked.

        Also cleans up the entry if it has expired.
        """
        if jti not in self._revoked_tokens:
            return False

        # Check if the revocation entry has expired
        exp_time = self._revoked_tokens[jti]
        if exp_time < datetime.now(timezone.utc).timestamp():
            # Entry expired, remove it
            del self._revoked_tokens[jti]
            self._save_revoked_tokens()
            return False

        return True

    def cleanup_expired_revocations(self) -> int:
        """
        Public method to clean up expired token revocations.

        Returns:
            Number of entries cleaned up.
        """
        return self._cleanup_expired_revocations()


class UserStore:
    """
    Persistent user store with JSON file storage.

    Stores users in a JSON file for persistence across server restarts.
    In production, replace with database-backed implementation.
    """

    def __init__(self, storage_path: str | None = None) -> None:
        """Initialize user store with persistence.

        Args:
            storage_path: Path to JSON file for user storage.
                         Defaults to ./data/users.json
        """
        import json
        import os
        from pathlib import Path

        self._users: dict[str, User] = {}
        self._username_index: dict[str, str] = {}
        self._email_index: dict[str, str] = {}
        self._password_manager = PasswordManager()

        # Set up storage path
        if storage_path is None:
            storage_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "users.json")
        self._storage_path = Path(storage_path).resolve()

        # Ensure directory exists
        self._storage_path.parent.mkdir(parents=True, exist_ok=True)

        # Load existing users from file
        self._load_users()

    def _load_users(self) -> None:
        """Load users from JSON file."""
        import json

        if not self._storage_path.exists():
            logger.info("user_store_initialized", path=str(self._storage_path))
            return

        try:
            with open(self._storage_path, "r") as f:
                data = json.load(f)

            for user_data in data.get("users", []):
                user = User(
                    user_id=user_data["user_id"],
                    username=user_data["username"],
                    email=user_data["email"],
                    password_hash=user_data["password_hash"],
                    roles={Role(r) for r in user_data.get("roles", ["viewer"])},
                    permissions={Permission(p) for p in user_data.get("permissions", [])},
                    is_active=user_data.get("is_active", True),
                    is_locked=user_data.get("is_locked", False),
                    failed_login_attempts=user_data.get("failed_login_attempts", 0),
                    last_login=datetime.fromisoformat(user_data["last_login"]) if user_data.get("last_login") else None,
                    created_at=datetime.fromisoformat(user_data["created_at"]) if user_data.get("created_at") else datetime.now(timezone.utc),
                    updated_at=datetime.fromisoformat(user_data["updated_at"]) if user_data.get("updated_at") else datetime.now(timezone.utc),
                    metadata=user_data.get("metadata", {}),
                )
                self._users[user.user_id] = user
                self._username_index[user.username.lower()] = user.user_id
                self._email_index[user.email.lower()] = user.user_id

            logger.info("users_loaded", count=len(self._users), path=str(self._storage_path))
        except Exception as e:
            logger.error("users_load_error", error=str(e), path=str(self._storage_path))

    def _save_users(self) -> None:
        """Save users to JSON file."""
        import json

        try:
            users_data = []
            for user in self._users.values():
                users_data.append({
                    "user_id": user.user_id,
                    "username": user.username,
                    "email": user.email,
                    "password_hash": user.password_hash,
                    "roles": [r.value for r in user.roles],
                    "permissions": [p.value for p in user.permissions],
                    "is_active": user.is_active,
                    "is_locked": user.is_locked,
                    "failed_login_attempts": user.failed_login_attempts,
                    "last_login": user.last_login.isoformat() if user.last_login else None,
                    "created_at": user.created_at.isoformat() if user.created_at else None,
                    "updated_at": user.updated_at.isoformat() if user.updated_at else None,
                    "metadata": user.metadata,
                })

            with open(self._storage_path, "w") as f:
                json.dump({"users": users_data}, f, indent=2)

            logger.debug("users_saved", count=len(users_data), path=str(self._storage_path))
        except Exception as e:
            logger.error("users_save_error", error=str(e), path=str(self._storage_path))

    def create_user(
        self,
        username: str,
        email: str,
        password: str,
        roles: set[Role] | None = None,
        permissions: set[Permission] | None = None,
    ) -> User:
        """
        Create a new user.

        Args:
            username: User's username.
            email: User's email.
            password: Plain text password.
            roles: User's roles.
            permissions: Direct permissions.

        Returns:
            Created user.

        Raises:
            ValueError: If username or email already exists.
        """
        if username.lower() in self._username_index:
            raise ValueError(f"Username '{username}' already exists")

        if email.lower() in self._email_index:
            raise ValueError(f"Email '{email}' already exists")

        user_id = secrets.token_urlsafe(16)
        user = User(
            user_id=user_id,
            username=username,
            email=email,
            password_hash=self._password_manager.hash_password(password),
            roles=roles or {Role.VIEWER},
            permissions=permissions or set(),
        )

        self._users[user_id] = user
        self._username_index[username.lower()] = user_id
        self._email_index[email.lower()] = user_id

        # Persist to file
        self._save_users()

        logger.info("user_created", user_id=user_id, username=username)

        return user

    def get_user(self, user_id: str) -> User | None:
        """Get user by ID."""
        return self._users.get(user_id)

    def get_user_by_username(self, username: str) -> User | None:
        """Get user by username."""
        user_id = self._username_index.get(username.lower())
        return self._users.get(user_id) if user_id else None

    def get_user_by_email(self, email: str) -> User | None:
        """Get user by email."""
        user_id = self._email_index.get(email.lower())
        return self._users.get(user_id) if user_id else None

    def update_user(self, user: User) -> None:
        """Update user in store."""
        user.updated_at = datetime.now(timezone.utc)
        self._users[user.user_id] = user
        # Persist to file
        self._save_users()

    def delete_user(self, user_id: str) -> bool:
        """Delete user from store."""
        user = self._users.get(user_id)
        if user:
            del self._users[user_id]
            self._username_index.pop(user.username.lower(), None)
            self._email_index.pop(user.email.lower(), None)
            # Persist to file
            self._save_users()
            logger.info("user_deleted", user_id=user_id)
            return True
        return False

    def list_users(
        self,
        limit: int = 100,
        offset: int = 0,
        role_filter: Role | None = None,
    ) -> list[User]:
        """List users with optional filtering."""
        users = list(self._users.values())

        if role_filter:
            users = [u for u in users if role_filter in u.roles]

        return users[offset : offset + limit]

    def authenticate(self, username: str, password: str) -> User | None:
        """
        Authenticate user with username and password.

        Args:
            username: Username.
            password: Plain text password.

        Returns:
            User if authenticated, None otherwise.
        """
        user = self.get_user_by_username(username)

        if not user:
            return None

        if user.is_locked:
            logger.warning("auth_locked_user", username=username)
            return None

        if not user.is_active:
            logger.warning("auth_inactive_user", username=username)
            return None

        if self._password_manager.verify_password(password, user.password_hash):
            # Reset failed attempts on success
            user.failed_login_attempts = 0
            user.last_login = datetime.now(timezone.utc)

            # Check if password needs rehash
            if self._password_manager.needs_rehash(user.password_hash):
                user.password_hash = self._password_manager.hash_password(password)

            self.update_user(user)
            logger.info("auth_success", user_id=user.user_id, username=username)
            return user

        # Increment failed attempts
        user.failed_login_attempts += 1
        if user.failed_login_attempts >= 5:
            user.is_locked = True
            logger.warning(
                "user_locked",
                user_id=user.user_id,
                username=username,
                attempts=user.failed_login_attempts,
            )

        self.update_user(user)
        logger.warning(
            "auth_failed",
            username=username,
            attempts=user.failed_login_attempts,
        )
        return None


class RBACManager:
    """
    Main RBAC management interface.

    Provides unified access to authentication, authorization,
    and token management.
    """

    _instance: RBACManager | None = None

    def __init__(
        self,
        secret_key: str,
        algorithm: str = "HS256",
        access_token_expire_minutes: int = 30,
        refresh_token_expire_days: int = 7,
        user_storage_path: str | None = None,
        revocation_storage_path: str | None = None,
    ) -> None:
        """
        Initialize RBAC manager.

        Args:
            secret_key: Secret key for JWT signing.
            algorithm: JWT algorithm.
            access_token_expire_minutes: Access token lifetime.
            refresh_token_expire_days: Refresh token lifetime.
            user_storage_path: Path to user storage file (for testing isolation).
            revocation_storage_path: Path to revoked tokens file (for testing isolation).
        """
        self._token_manager = TokenManager(
            secret_key=secret_key,
            algorithm=algorithm,
            access_token_expire_minutes=access_token_expire_minutes,
            refresh_token_expire_days=refresh_token_expire_days,
            revocation_storage_path=revocation_storage_path,
        )
        self._user_store = UserStore(storage_path=user_storage_path)
        self._password_manager = PasswordManager()

    @classmethod
    def get_instance(
        cls,
        secret_key: str | None = None,
        **kwargs: Any,
    ) -> RBACManager:
        """Get or create singleton instance."""
        if cls._instance is None:
            if secret_key is None:
                raise ValueError("secret_key is required for first initialization")
            cls._instance = cls(secret_key, **kwargs)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (for testing)."""
        cls._instance = None

    @property
    def users(self) -> UserStore:
        """Get user store."""
        return self._user_store

    @property
    def tokens(self) -> TokenManager:
        """Get token manager."""
        return self._token_manager

    def authenticate(self, username: str, password: str) -> TokenPair | None:
        """
        Authenticate user and return token pair.

        Args:
            username: Username.
            password: Password.

        Returns:
            TokenPair if authenticated, None otherwise.
        """
        user = self._user_store.authenticate(username, password)
        if user:
            return self._token_manager.create_token_pair(user)
        return None

    def validate_access(
        self,
        token: str,
        required_permissions: set[Permission] | None = None,
        required_roles: set[Role] | None = None,
    ) -> TokenPayload:
        """
        Validate access token and check permissions.

        Args:
            token: Access token.
            required_permissions: Required permissions.
            required_roles: Required roles.

        Returns:
            TokenPayload if valid and authorized.

        Raises:
            TokenError: If token is invalid.
            AuthorizationError: If permissions/roles not met.
        """
        payload = self._token_manager.validate_token(token)

        if payload.token_type != "access":
            raise TokenInvalidError("Not an access token")

        user_permissions = {Permission(p) for p in payload.permissions}
        user_roles = {Role(r) for r in payload.roles}

        if required_permissions:
            if not required_permissions <= user_permissions:
                missing = required_permissions - user_permissions
                raise AuthorizationError(
                    f"Missing permissions: {[p.value for p in missing]}"
                )

        if required_roles:
            if not required_roles & user_roles:
                raise AuthorizationError(
                    f"Required roles: {[r.value for r in required_roles]}"
                )

        return payload

    def refresh_access_token(self, refresh_token: str) -> TokenPair | None:
        """
        Refresh access token using refresh token.

        Args:
            refresh_token: Refresh token.

        Returns:
            New TokenPair if valid, None otherwise.
        """
        try:
            payload = self._token_manager.validate_token(refresh_token)

            if payload.token_type != "refresh":
                return None

            user = self._user_store.get_user(payload.sub)
            if not user or not user.is_active:
                return None

            # Revoke old refresh token
            self._token_manager.revoke_token(refresh_token)

            # Create new token pair
            return self._token_manager.create_token_pair(user)

        except TokenError:
            return None

    def logout(self, token: str) -> None:
        """
        Logout by revoking token.

        Args:
            token: Token to revoke.
        """
        self._token_manager.revoke_token(token)

    def create_api_key(self, user: User, name: str, expires_days: int = 365) -> str:
        """
        Create an API key for a user.

        Args:
            user: User to create key for.
            name: Key name/description.
            expires_days: Key validity in days.

        Returns:
            API key string.
        """
        now = datetime.now(timezone.utc)
        expires = now + timedelta(days=expires_days)

        payload = {
            "sub": user.user_id,
            "type": "api_key",
            "name": name,
            "permissions": [p.value for p in user.get_all_permissions()],
            "exp": int(expires.timestamp()),
            "iat": int(now.timestamp()),
            "jti": secrets.token_urlsafe(32),
        }

        return jwt.encode(
            payload,
            self._token_manager._secret_key,
            algorithm=self._token_manager._algorithm,
        )


# Decorators for permission checking
F = TypeVar("F", bound=Callable[..., Any])


def require_permissions(*permissions: Permission) -> Callable[[F], F]:
    """
    Decorator to require specific permissions.

    Args:
        *permissions: Required permissions.

    Returns:
        Decorated function.
    """
    required = set(permissions)

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # Get user from context (implementation specific)
            user = kwargs.get("current_user")
            if not user:
                raise AuthorizationError("No authenticated user")

            if not user.has_all_permissions(required):
                missing = required - user.get_all_permissions()
                raise AuthorizationError(
                    f"Missing permissions: {[p.value for p in missing]}"
                )

            return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


def require_roles(*roles: Role) -> Callable[[F], F]:
    """
    Decorator to require any of the specified roles.

    Args:
        *roles: Acceptable roles (any match is sufficient).

    Returns:
        Decorated function.
    """
    required = set(roles)

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            user = kwargs.get("current_user")
            if not user:
                raise AuthorizationError("No authenticated user")

            if not (required & user.roles):
                raise AuthorizationError(
                    f"Required roles: {[r.value for r in required]}"
                )

            return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


def require_admin(func: F) -> F:
    """Decorator to require admin role."""
    return require_roles(Role.ADMIN)(func)
