"""
Comprehensive Unit Tests for Security Module.

Tests cover:
- AES-256 encryption/decryption
- Key management
- HIPAA-compliant audit logging
- PHI masking
- Secure data deletion
- Role-Based Access Control
- JWT token management
"""

from __future__ import annotations

import asyncio
import hashlib
import os
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Generator
from unittest.mock import MagicMock, patch

import pytest


# =============================================================================
# Encryption Tests
# =============================================================================


class TestEncryptionConfig:
    """Tests for EncryptionConfig."""

    def test_default_config(self) -> None:
        """Test default encryption configuration."""
        from src.security.encryption import EncryptionConfig, EncryptionAlgorithm

        config = EncryptionConfig()
        assert config.algorithm == EncryptionAlgorithm.AES_256_GCM
        assert config.pbkdf2_iterations >= 100_000

    def test_custom_config(self) -> None:
        """Test custom encryption configuration."""
        from src.security.encryption import (
            EncryptionConfig,
            EncryptionAlgorithm,
            KeyDerivationFunction,
        )

        config = EncryptionConfig(
            algorithm=EncryptionAlgorithm.AES_256_CBC,
            pbkdf2_iterations=200_000,
            kdf=KeyDerivationFunction.SCRYPT,
        )
        assert config.algorithm == EncryptionAlgorithm.AES_256_CBC
        assert config.pbkdf2_iterations == 200_000
        assert config.kdf == KeyDerivationFunction.SCRYPT


class TestKeyManager:
    """Tests for KeyManager."""

    def test_generate_key(self) -> None:
        """Test key generation."""
        from src.security.encryption import KeyManager

        key = KeyManager.generate_key()

        assert isinstance(key, bytes)
        assert len(key) == 32  # AES-256

    def test_derive_key_from_password(self) -> None:
        """Test key derivation from password."""
        from src.security.encryption import KeyManager

        key_manager = KeyManager()
        password = b"secure_password_123"  # Must be bytes
        salt = os.urandom(32)

        key = key_manager.derive_key(password, salt)

        assert isinstance(key, bytes)
        assert len(key) == 32

    def test_derive_key_deterministic(self) -> None:
        """Test that key derivation is deterministic."""
        from src.security.encryption import KeyManager

        key_manager = KeyManager()
        password = b"test_password"
        salt = b"fixed_salt_value" + b"\x00" * 16  # 32 bytes

        key1 = key_manager.derive_key(password, salt)
        key2 = key_manager.derive_key(password, salt)

        assert key1 == key2

    def test_derive_key_different_salts(self) -> None:
        """Test that different salts produce different keys."""
        from src.security.encryption import KeyManager

        key_manager = KeyManager()
        password = b"test_password"

        key1 = key_manager.derive_key(password, os.urandom(32))
        key2 = key_manager.derive_key(password, os.urandom(32))

        assert key1 != key2

    def test_set_master_key(self) -> None:
        """Test setting master key."""
        from src.security.encryption import KeyManager

        key_manager = KeyManager()
        master_key = KeyManager.generate_key()

        key_manager.set_master_key(master_key)
        derived_key, salt = key_manager.get_encryption_key()

        assert len(derived_key) == 32
        assert len(salt) == 32


class TestAESEncryptor:
    """Tests for AESEncryptor."""

    @pytest.fixture
    def encryptor(self):
        """Create encryptor with master key set."""
        from src.security.encryption import AESEncryptor, KeyManager

        key_manager = KeyManager()
        key_manager.set_master_key(KeyManager.generate_key())
        return AESEncryptor(key_manager=key_manager)

    def test_encrypt_decrypt_gcm(self, encryptor) -> None:
        """Test encryption and decryption with GCM mode."""
        plaintext = b"Sensitive medical data - Patient: John Doe, SSN: 123-45-6789"
        encrypted = encryptor.encrypt(plaintext)

        assert encrypted.ciphertext != plaintext
        assert encrypted.nonce is not None
        assert encrypted.tag is not None

        decrypted = encryptor.decrypt(encrypted)
        assert decrypted == plaintext

    def test_encrypt_with_aad(self, encryptor) -> None:
        """Test encryption with additional authenticated data."""
        plaintext = b"Patient data"
        aad = b"metadata:patient_id=12345"

        encrypted = encryptor.encrypt(plaintext, aad)
        decrypted = encryptor.decrypt(encrypted, aad)

        assert decrypted == plaintext

    def test_decrypt_wrong_aad_fails(self, encryptor) -> None:
        """Test that decryption fails with wrong AAD."""
        from src.security.encryption import DecryptionError

        plaintext = b"Patient data"
        aad = b"correct_aad"

        encrypted = encryptor.encrypt(plaintext, aad)

        with pytest.raises(DecryptionError):
            encryptor.decrypt(encrypted, b"wrong_aad")

    def test_encrypt_empty_data(self, encryptor) -> None:
        """Test encryption of empty data."""
        plaintext = b""
        encrypted = encryptor.encrypt(plaintext)
        decrypted = encryptor.decrypt(encrypted)

        assert decrypted == plaintext

    def test_encrypt_large_data(self, encryptor) -> None:
        """Test encryption of large data."""
        # 1 MB of data
        plaintext = os.urandom(1024 * 1024)
        encrypted = encryptor.encrypt(plaintext)
        decrypted = encryptor.decrypt(encrypted)

        assert decrypted == plaintext


class TestFileEncryptor:
    """Tests for FileEncryptor."""

    @pytest.fixture
    def temp_dir(self) -> Generator[Path, None, None]:
        """Create temporary directory for tests."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    @pytest.fixture
    def file_encryptor(self):
        """Create file encryptor with master key."""
        from src.security.encryption import FileEncryptor, KeyManager

        key_manager = KeyManager()
        key_manager.set_master_key(KeyManager.generate_key())
        return FileEncryptor(key_manager=key_manager)

    def test_encrypt_decrypt_file(self, temp_dir: Path, file_encryptor) -> None:
        """Test file encryption and decryption."""
        # Create test file
        source_file = temp_dir / "test.txt"
        source_file.write_bytes(b"Test file content with sensitive data")

        # Encrypt
        encrypted_file = temp_dir / "test.txt.enc"
        file_encryptor.encrypt_file(source_file, encrypted_file)

        assert encrypted_file.exists()
        assert encrypted_file.read_bytes() != source_file.read_bytes()

        # Decrypt
        decrypted_file = temp_dir / "test.txt.dec"
        file_encryptor.decrypt_file(encrypted_file, decrypted_file)

        assert decrypted_file.read_bytes() == source_file.read_bytes()


class TestEncryptionService:
    """Tests for EncryptionService."""

    def test_encrypt_decrypt_data(self) -> None:
        """Test data encryption and decryption."""
        from src.security.encryption import EncryptionService

        service = EncryptionService()
        data = b"Medical record data"

        encrypted = service.encrypt(data)
        decrypted = service.decrypt(encrypted)

        assert decrypted == data

    def test_encrypt_with_password(self) -> None:
        """Test password-based encryption."""
        from src.security.encryption import EncryptionService

        service = EncryptionService()
        data = b"Secure data"
        password = "strong_password_123"

        encrypted = service.encrypt_with_password(data, password)
        decrypted = service.decrypt_with_password(encrypted, password)

        assert decrypted == data

    def test_decrypt_wrong_password_fails(self) -> None:
        """Test that wrong password fails decryption."""
        from src.security.encryption import DecryptionError, EncryptionService

        service = EncryptionService()
        data = b"Secure data"

        encrypted = service.encrypt_with_password(data, "correct_password")

        with pytest.raises(DecryptionError):
            service.decrypt_with_password(encrypted, "wrong_password")


# =============================================================================
# Audit Logging Tests
# =============================================================================


class TestPHIMasker:
    """Tests for PHIMasker."""

    def test_mask_ssn(self) -> None:
        """Test SSN masking."""
        from src.security.audit import PHIMasker

        masker = PHIMasker()
        result = masker.mask("SSN: 123-45-6789")

        # SSN should be masked
        assert "123-45-6789" not in result

    def test_mask_email(self) -> None:
        """Test email masking."""
        from src.security.audit import PHIMasker

        masker = PHIMasker()
        masked = masker.mask("Email: patient@hospital.com")

        assert "patient" not in masked

    def test_mask_phone(self) -> None:
        """Test phone number masking."""
        from src.security.audit import PHIMasker

        masker = PHIMasker()

        result = masker.mask("Phone: 555-123-4567")
        assert "555-123-4567" not in result


class TestAuditEvent:
    """Tests for AuditEvent."""

    def test_create_event(self) -> None:
        """Test audit event creation."""
        from src.security.audit import (
            AuditEvent,
            AuditEventType,
            AuditOutcome,
            AuditSeverity,
        )

        event = AuditEvent(
            event_type=AuditEventType.DATA_ACCESS,
            severity=AuditSeverity.INFO,
            outcome=AuditOutcome.SUCCESS,
            actor_id="user123",
            actor_ip="192.168.1.100",
            resource_type="document",
            resource_id="doc-456",
            action="view",
            details={"pages": 10},
        )

        assert event.event_type == AuditEventType.DATA_ACCESS
        assert event.actor_id == "user123"
        assert event.outcome == AuditOutcome.SUCCESS

    def test_event_to_dict(self) -> None:
        """Test event serialization."""
        from src.security.audit import (
            AuditEvent,
            AuditEventType,
            AuditOutcome,
            AuditSeverity,
        )

        event = AuditEvent(
            event_type=AuditEventType.AUTHENTICATION,
            severity=AuditSeverity.WARNING,
            outcome=AuditOutcome.FAILURE,
            actor_id="user456",
            action="login",
        )

        data = event.to_dict()

        assert data["event_type"] == "AUTHENTICATION"
        assert data["severity"] == "WARNING"
        assert data["outcome"] == "FAILURE"
        assert "timestamp" in data


class TestAuditLogger:
    """Tests for AuditLogger."""

    @pytest.fixture
    def temp_log_dir(self) -> Generator[Path, None, None]:
        """Create temporary log directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_log_event(self, temp_log_dir: Path) -> None:
        """Test logging an event."""
        from src.security.audit import (
            AuditEventType,
            AuditLogger,
            AuditOutcome,
            AuditSeverity,
        )

        logger = AuditLogger(log_dir=str(temp_log_dir), mask_phi=True)

        logger.log_event(
            event_type=AuditEventType.DATA_ACCESS,
            severity=AuditSeverity.INFO,
            outcome=AuditOutcome.SUCCESS,
            action="document_view",
            resource_type="document",
            resource_id="doc-123",
        )

    def test_log_api_request(self, temp_log_dir: Path) -> None:
        """Test API request logging."""
        from src.security.audit import AuditLogger

        logger = AuditLogger(log_dir=str(temp_log_dir))

        logger.log_api_request(
            method="POST",
            path="/api/v1/documents/process",
            status_code=200,
            duration_ms=150.5,
            client_ip="10.0.0.1",
            user_id="user123",
        )

    def test_log_data_access(self, temp_log_dir: Path) -> None:
        """Test data access logging."""
        from src.security.audit import AuditLogger

        logger = AuditLogger(log_dir=str(temp_log_dir))

        logger.log_data_access(
            resource_type="patient_record",
            resource_id="patient-456",
            action="read",
            user_id="doctor123",
            details={"fields_accessed": ["name", "dob", "diagnosis"]},
        )

    def test_context_management(self, temp_log_dir: Path) -> None:
        """Test context setting and clearing."""
        from src.security.audit import AuditLogger

        logger = AuditLogger(log_dir=str(temp_log_dir))

        logger.set_context(
            request_id="req-123",
            user_id="user456",
            client_ip="192.168.1.1",
        )

        # Verify context is set
        context = logger.get_context()
        assert context.request_id == "req-123"
        assert context.user_id == "user456"

        # Clear context
        logger.clear_context()
        context = logger.get_context()
        assert context.request_id is None


class TestAuditLogDecorator:
    """Tests for audit_log decorator."""

    @pytest.fixture
    def temp_log_dir(self) -> Generator[Path, None, None]:
        """Create temporary log directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_decorator_logs_function_call(self, temp_log_dir: Path) -> None:
        """Test that decorator logs function calls."""
        from src.security.audit import AuditEventType, AuditLogger, audit_log

        logger = AuditLogger(log_dir=str(temp_log_dir))

        @audit_log(
            logger=logger,
            event_type=AuditEventType.DATA_ACCESS,
            action="process_document",
        )
        def process_document(doc_id: str) -> str:
            return f"Processed {doc_id}"

        result = process_document("doc-123")
        assert result == "Processed doc-123"


# =============================================================================
# Data Cleanup Tests
# =============================================================================


class TestSecureOverwriter:
    """Tests for SecureOverwriter."""

    @pytest.fixture
    def temp_dir(self) -> Generator[Path, None, None]:
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_single_pass_overwrite(self, temp_dir: Path) -> None:
        """Test single-pass secure overwrite."""
        from src.security.data_cleanup import DeletionMethod, SecureOverwriter

        overwriter = SecureOverwriter()
        test_file = temp_dir / "test.txt"
        test_file.write_bytes(b"Sensitive data here")

        original_size = test_file.stat().st_size
        result = overwriter.overwrite_file(test_file, DeletionMethod.SINGLE_PASS)

        assert result.success
        assert result.passes_completed == 1
        assert test_file.stat().st_size == original_size

    def test_dod_3_pass_overwrite(self, temp_dir: Path) -> None:
        """Test DoD 3-pass secure overwrite."""
        from src.security.data_cleanup import DeletionMethod, SecureOverwriter

        overwriter = SecureOverwriter()
        test_file = temp_dir / "test.txt"
        test_file.write_bytes(b"Top secret data")

        result = overwriter.overwrite_file(test_file, DeletionMethod.DOD_3_PASS)

        assert result.success
        assert result.passes_completed == 3

    def test_overwrite_and_delete(self, temp_dir: Path) -> None:
        """Test secure overwrite followed by deletion."""
        from src.security.data_cleanup import SecureOverwriter

        overwriter = SecureOverwriter()
        test_file = temp_dir / "test.txt"
        test_file.write_bytes(b"Delete this securely")

        result = overwriter.secure_delete(test_file)

        assert result.success
        assert not test_file.exists()


class TestSecureDataCleanup:
    """Tests for SecureDataCleanup."""

    @pytest.fixture
    def temp_dir(self) -> Generator[Path, None, None]:
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_cleanup_file(self, temp_dir: Path) -> None:
        """Test file cleanup."""
        from src.security.data_cleanup import SecureDataCleanup

        cleanup = SecureDataCleanup()
        test_file = temp_dir / "sensitive.pdf"
        test_file.write_bytes(b"Patient medical records")

        stats = cleanup.cleanup_file(test_file)

        assert stats.files_deleted == 1
        assert not test_file.exists()

    def test_cleanup_directory(self, temp_dir: Path) -> None:
        """Test directory cleanup."""
        from src.security.data_cleanup import SecureDataCleanup

        cleanup = SecureDataCleanup()

        # Create test files
        (temp_dir / "file1.txt").write_bytes(b"Data 1")
        (temp_dir / "file2.txt").write_bytes(b"Data 2")
        subdir = temp_dir / "subdir"
        subdir.mkdir()
        (subdir / "file3.txt").write_bytes(b"Data 3")

        stats = cleanup.cleanup_directory(temp_dir, recursive=True)

        assert stats.files_deleted >= 3

    def test_cleanup_with_pattern(self, temp_dir: Path) -> None:
        """Test cleanup with file pattern."""
        from src.security.data_cleanup import SecureDataCleanup

        cleanup = SecureDataCleanup()

        # Create mixed files
        (temp_dir / "data.txt").write_bytes(b"Keep this")
        (temp_dir / "temp1.tmp").write_bytes(b"Delete this")
        (temp_dir / "temp2.tmp").write_bytes(b"Delete this too")

        stats = cleanup.cleanup_directory(temp_dir, pattern="*.tmp")

        assert stats.files_deleted == 2
        assert (temp_dir / "data.txt").exists()


class TestRetentionManager:
    """Tests for RetentionManager."""

    @pytest.fixture
    def temp_dir(self) -> Generator[Path, None, None]:
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_create_policy(self) -> None:
        """Test retention policy creation."""
        from src.security.data_cleanup import RetentionManager, RetentionPolicy

        manager = RetentionManager()

        policy = RetentionPolicy(
            name="medical_records",
            retention_days=2555,  # 7 years
            description="HIPAA medical record retention",
        )

        manager.add_policy(policy)
        assert manager.get_policy("medical_records") is not None


class TestTempFileManager:
    """Tests for TempFileManager."""

    def test_create_temp_file(self) -> None:
        """Test temporary file creation."""
        from src.security.data_cleanup import TempFileManager

        manager = TempFileManager()

        temp_path = manager.create_temp_file(suffix=".pdf")

        assert temp_path.exists()
        assert temp_path.suffix == ".pdf"

        # Cleanup
        manager.cleanup()
        assert not temp_path.exists()

    def test_context_manager(self) -> None:
        """Test context manager for temp files."""
        from src.security.data_cleanup import TempFileManager

        manager = TempFileManager()

        with manager.temp_file(suffix=".txt") as temp_path:
            temp_path.write_bytes(b"Temporary data")
            assert temp_path.exists()

        # File should be cleaned up
        assert not temp_path.exists()


class TestMemorySecurityManager:
    """Tests for MemorySecurityManager."""

    def test_secure_clear_bytearray(self) -> None:
        """Test secure clearing of bytearray."""
        from src.security.data_cleanup import MemorySecurityManager

        manager = MemorySecurityManager()

        data = bytearray(b"Sensitive password data")
        manager.secure_clear(data)

        assert all(b == 0 for b in data)

    def test_secure_string_context(self) -> None:
        """Test secure string context manager."""
        from src.security.data_cleanup import MemorySecurityManager

        manager = MemorySecurityManager()

        with manager.secure_string("password123") as secure_data:
            assert b"password123" in secure_data

        # Data should be cleared
        assert all(b == 0 for b in secure_data)


# =============================================================================
# RBAC Tests
# =============================================================================


class TestPermissionAndRole:
    """Tests for Permission and Role enums."""

    def test_permissions_exist(self) -> None:
        """Test that required permissions exist."""
        from src.security.rbac import Permission

        assert Permission.DOCUMENTS_READ
        assert Permission.DOCUMENTS_WRITE
        assert Permission.DOCUMENTS_DELETE
        assert Permission.ADMIN_USERS

    def test_roles_exist(self) -> None:
        """Test that required roles exist."""
        from src.security.rbac import Role

        assert Role.VIEWER
        assert Role.OPERATOR
        assert Role.ADMIN

    def test_role_permissions_mapping(self) -> None:
        """Test role to permissions mapping."""
        from src.security.rbac import Permission, Role, ROLE_PERMISSIONS

        # Viewer should have read access
        assert Permission.DOCUMENTS_READ in ROLE_PERMISSIONS[Role.VIEWER]

        # Admin should have all permissions
        admin_perms = ROLE_PERMISSIONS[Role.ADMIN]
        assert Permission.DOCUMENTS_READ in admin_perms
        assert Permission.DOCUMENTS_WRITE in admin_perms
        assert Permission.ADMIN_USERS in admin_perms


class TestUser:
    """Tests for User model."""

    def test_create_user(self) -> None:
        """Test user creation."""
        from src.security.rbac import Role, User

        user = User(
            user_id="user-123",
            username="johndoe",
            email="john@example.com",
            roles=[Role.OPERATOR],
        )

        assert user.user_id == "user-123"
        assert user.username == "johndoe"
        assert Role.OPERATOR in user.roles
        assert user.is_active

    def test_user_permissions(self) -> None:
        """Test user permission calculation."""
        from src.security.rbac import Permission, Role, User

        user = User(
            user_id="user-123",
            username="operator",
            roles=[Role.OPERATOR],
        )

        perms = user.get_permissions()
        assert Permission.DOCUMENTS_READ in perms
        assert Permission.DOCUMENTS_WRITE in perms


class TestPasswordManager:
    """Tests for PasswordManager."""

    def test_hash_password(self) -> None:
        """Test password hashing."""
        from src.security.rbac import PasswordManager

        manager = PasswordManager()
        password = "SecurePass123!"

        hashed = manager.hash_password(password)

        assert hashed != password
        assert len(hashed) > 20

    def test_verify_password(self) -> None:
        """Test password verification."""
        from src.security.rbac import PasswordManager

        manager = PasswordManager()
        password = "SecurePass123!"

        hashed = manager.hash_password(password)

        assert manager.verify_password(password, hashed)
        assert not manager.verify_password("WrongPassword", hashed)

    def test_validate_password_strength(self) -> None:
        """Test password strength validation."""
        from src.security.rbac import PasswordManager

        manager = PasswordManager()

        # Weak password
        weak_result = manager.validate_password_strength("weak")
        assert not weak_result.is_valid

        # Strong password
        strong_result = manager.validate_password_strength("SecureP@ss123!")
        assert strong_result.is_valid


class TestTokenManager:
    """Tests for TokenManager."""

    def test_create_access_token(self) -> None:
        """Test access token creation."""
        from src.security.rbac import Role, TokenManager, User

        manager = TokenManager(secret_key="test-secret-key-12345")
        user = User(
            user_id="user-123",
            username="testuser",
            roles=[Role.OPERATOR],
        )

        token = manager.create_access_token(user)

        assert token is not None
        assert len(token) > 50

    def test_validate_token(self) -> None:
        """Test token validation."""
        from src.security.rbac import Role, TokenManager, User

        manager = TokenManager(secret_key="test-secret-key-12345")
        user = User(
            user_id="user-123",
            username="testuser",
            roles=[Role.OPERATOR],
        )

        token = manager.create_access_token(user)
        payload = manager.validate_token(token)

        assert payload is not None
        assert payload.sub == "user-123"
        assert payload.username == "testuser"

    def test_expired_token_fails(self) -> None:
        """Test that expired tokens are rejected."""
        from src.security.rbac import Role, TokenExpiredError, TokenManager, User

        manager = TokenManager(
            secret_key="test-secret-key-12345",
            access_token_expire_minutes=-1,  # Already expired
        )
        user = User(
            user_id="user-123",
            username="testuser",
            roles=[Role.VIEWER],
        )

        token = manager.create_access_token(user)

        with pytest.raises(TokenExpiredError):
            manager.validate_token(token)

    def test_invalid_token_fails(self) -> None:
        """Test that invalid tokens are rejected."""
        from src.security.rbac import TokenInvalidError, TokenManager

        manager = TokenManager(secret_key="test-secret-key-12345")

        with pytest.raises(TokenInvalidError):
            manager.validate_token("invalid.token.here")

    def test_create_token_pair(self) -> None:
        """Test creation of access and refresh token pair."""
        from src.security.rbac import Role, TokenManager, User

        manager = TokenManager(secret_key="test-secret-key-12345")
        user = User(
            user_id="user-123",
            username="testuser",
            roles=[Role.VIEWER],
        )

        pair = manager.create_token_pair(user)

        assert pair.access_token is not None
        assert pair.refresh_token is not None
        assert pair.access_token != pair.refresh_token


class TestUserStore:
    """Tests for UserStore."""

    def test_create_user(self) -> None:
        """Test user creation in store."""
        from src.security.rbac import Role, UserStore

        store = UserStore()

        user = store.create_user(
            username="newuser",
            password="SecureP@ss123!",
            email="new@example.com",
            roles=[Role.VIEWER],
        )

        assert user.username == "newuser"
        assert user.user_id is not None

    def test_get_user_by_username(self) -> None:
        """Test user retrieval by username."""
        from src.security.rbac import Role, UserStore

        store = UserStore()
        store.create_user(
            username="testuser",
            password="SecureP@ss123!",
            roles=[Role.VIEWER],
        )

        user = store.get_by_username("testuser")

        assert user is not None
        assert user.username == "testuser"

    def test_authenticate_user(self) -> None:
        """Test user authentication."""
        from src.security.rbac import Role, UserStore

        store = UserStore()
        store.create_user(
            username="authuser",
            password="CorrectPassword123!",
            roles=[Role.VIEWER],
        )

        # Correct credentials
        user = store.authenticate("authuser", "CorrectPassword123!")
        assert user is not None

        # Wrong password
        user = store.authenticate("authuser", "WrongPassword")
        assert user is None


class TestRBACManager:
    """Tests for RBACManager."""

    def test_login(self) -> None:
        """Test user login."""
        from src.security.rbac import RBACManager, Role

        manager = RBACManager(secret_key="test-secret-key-12345")

        # Create user
        manager.users.create_user(
            username="loginuser",
            password="SecureP@ss123!",
            roles=[Role.VIEWER],
        )

        # Login
        tokens = manager.login("loginuser", "SecureP@ss123!")

        assert tokens is not None
        assert tokens.access_token is not None

    def test_check_permission(self) -> None:
        """Test permission checking."""
        from src.security.rbac import Permission, RBACManager, Role

        manager = RBACManager(secret_key="test-secret-key-12345")

        user = manager.users.create_user(
            username="permuser",
            password="SecureP@ss123!",
            roles=[Role.OPERATOR],
        )

        # Should have document read
        assert manager.check_permission(user, Permission.DOCUMENTS_READ)

        # Should not have admin permissions
        assert not manager.check_permission(user, Permission.ADMIN_USERS)


class TestRequirePermissionsDecorator:
    """Tests for permission decorators."""

    def test_require_permissions(self) -> None:
        """Test require_permissions decorator."""
        from src.security.rbac import (
            AuthorizationError,
            Permission,
            RBACManager,
            Role,
            require_permissions,
        )

        manager = RBACManager(secret_key="test-secret-key-12345")

        @require_permissions(Permission.DOCUMENTS_READ, rbac_manager=manager)
        def read_document(user: Any, doc_id: str) -> str:
            return f"Read {doc_id}"

        # Create user with permission
        user = manager.users.create_user(
            username="reader",
            password="SecureP@ss123!",
            roles=[Role.VIEWER],
        )

        # Should succeed
        result = read_document(user, "doc-123")
        assert result == "Read doc-123"

    def test_require_admin(self) -> None:
        """Test require_admin decorator."""
        from src.security.rbac import (
            AuthorizationError,
            RBACManager,
            Role,
            require_admin,
        )

        manager = RBACManager(secret_key="test-secret-key-12345")

        @require_admin(rbac_manager=manager)
        def admin_action(user: Any) -> str:
            return "Admin action completed"

        # Non-admin user
        viewer = manager.users.create_user(
            username="viewer",
            password="SecureP@ss123!",
            roles=[Role.VIEWER],
        )

        with pytest.raises(AuthorizationError):
            admin_action(viewer)

        # Admin user
        admin = manager.users.create_user(
            username="admin",
            password="SecureP@ss123!",
            roles=[Role.ADMIN],
        )

        result = admin_action(admin)
        assert result == "Admin action completed"


# =============================================================================
# Integration Tests
# =============================================================================


class TestSecurityIntegration:
    """Integration tests for security components."""

    @pytest.fixture
    def temp_dir(self) -> Generator[Path, None, None]:
        """Create temporary directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)

    def test_encrypt_audit_cleanup_flow(self, temp_dir: Path) -> None:
        """Test complete security flow: encrypt, audit, cleanup."""
        from src.security.audit import AuditEventType, AuditLogger
        from src.security.data_cleanup import SecureDataCleanup
        from src.security.encryption import EncryptionService, FileEncryptor, KeyManager

        # 1. Create and encrypt sensitive file
        key_manager = KeyManager()
        key_manager.set_master_key(KeyManager.generate_key())
        file_encryptor = FileEncryptor(key_manager=key_manager)

        source_file = temp_dir / "patient_record.pdf"
        source_file.write_bytes(b"Patient: John Doe, SSN: 123-45-6789")

        encrypted_file = temp_dir / "patient_record.pdf.enc"
        file_encryptor.encrypt_file(source_file, encrypted_file)

        # 2. Log the encryption event
        audit_logger = AuditLogger(log_dir=str(temp_dir / "audit"), mask_phi=True)
        audit_logger.log_data_access(
            resource_type="patient_record",
            resource_id="patient-001",
            action="encrypt",
            user_id="system",
        )

        # 3. Securely clean up original file
        cleanup = SecureDataCleanup()
        stats = cleanup.cleanup_file(source_file)

        assert stats.files_deleted == 1
        assert not source_file.exists()
        assert encrypted_file.exists()

        # 4. Decrypt and verify
        decrypted_file = temp_dir / "patient_record.pdf.dec"
        file_encryptor.decrypt_file(encrypted_file, decrypted_file)

        assert b"John Doe" in decrypted_file.read_bytes()

    def test_rbac_with_audit_logging(self, temp_dir: Path) -> None:
        """Test RBAC actions with audit logging."""
        from src.security.audit import AuditLogger
        from src.security.rbac import Permission, RBACManager, Role

        # Set up RBAC
        manager = RBACManager(secret_key="test-secret-key-12345")
        audit_logger = AuditLogger(log_dir=str(temp_dir / "audit"))

        # Create users
        admin = manager.users.create_user(
            username="admin",
            password="AdminP@ss123!",
            roles=[Role.ADMIN],
        )

        viewer = manager.users.create_user(
            username="viewer",
            password="ViewerP@ss123!",
            roles=[Role.VIEWER],
        )

        # Log authentication
        tokens = manager.login("admin", "AdminP@ss123!")
        audit_logger.log_authentication(
            user_id=admin.user_id,
            success=True,
            method="password",
        )

        # Check permissions and log access attempts
        can_read = manager.check_permission(viewer, Permission.DOCUMENTS_READ)
        can_delete = manager.check_permission(viewer, Permission.DOCUMENTS_DELETE)

        assert can_read
        assert not can_delete
