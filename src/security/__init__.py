"""
Security Module for HIPAA-Compliant Document Extraction System.

Provides enterprise-grade security features including:
- AES-256 encryption for data at rest and in transit
- HIPAA-compliant audit logging with tamper-evident chains
- Secure data cleanup with multi-pass deletion
- Role-Based Access Control (RBAC) with JWT tokens
"""

from src.security.audit import (
    AuditContext,
    AuditEvent,
    AuditEventType,
    AuditLogger,
    AuditOutcome,
    AuditSeverity,
    PHIMasker,
    audit_log,
)
from src.security.data_cleanup import (
    CleanupError,
    CleanupStats,
    DeletionMethod,
    DeletionResult,
    MemorySecurityManager,
    RetentionManager,
    RetentionPolicy,
    SecureDataCleanup,
    SecureDeletionError,
    SecureOverwriter,
    TempFileManager,
)
from src.security.encryption import (
    AESEncryptor,
    DecryptionError,
    EncryptedData,
    EncryptionAlgorithm,
    EncryptionConfig,
    EncryptionError,
    EncryptionService,
    FileEncryptor,
    IntegrityError,
    KeyDerivationError,
    KeyDerivationFunction,
    KeyManager,
)
from src.security.rbac import (
    AuthenticationError,
    AuthorizationError,
    PasswordManager,
    Permission,
    RBACManager,
    Role,
    ROLE_PERMISSIONS,
    TokenError,
    TokenExpiredError,
    TokenInvalidError,
    TokenManager,
    TokenPair,
    TokenPayload,
    User,
    UserStore,
    require_admin,
    require_permissions,
    require_roles,
)

__all__ = [
    # Encryption
    "AESEncryptor",
    "DecryptionError",
    "EncryptedData",
    "EncryptionAlgorithm",
    "EncryptionConfig",
    "EncryptionError",
    "EncryptionService",
    "FileEncryptor",
    "IntegrityError",
    "KeyDerivationError",
    "KeyDerivationFunction",
    "KeyManager",
    # Audit
    "AuditContext",
    "AuditEvent",
    "AuditEventType",
    "AuditLogger",
    "AuditOutcome",
    "AuditSeverity",
    "PHIMasker",
    "audit_log",
    # Data Cleanup
    "CleanupError",
    "CleanupStats",
    "DeletionMethod",
    "DeletionResult",
    "MemorySecurityManager",
    "RetentionManager",
    "RetentionPolicy",
    "SecureDataCleanup",
    "SecureDeletionError",
    "SecureOverwriter",
    "TempFileManager",
    # RBAC
    "AuthenticationError",
    "AuthorizationError",
    "PasswordManager",
    "Permission",
    "RBACManager",
    "Role",
    "ROLE_PERMISSIONS",
    "TokenError",
    "TokenExpiredError",
    "TokenInvalidError",
    "TokenManager",
    "TokenPair",
    "TokenPayload",
    "User",
    "UserStore",
    "require_admin",
    "require_permissions",
    "require_roles",
]
