"""
AES-256 Encryption Module for HIPAA-Compliant Data Protection.

Provides enterprise-grade encryption for data at rest and in transit,
with secure key derivation, authenticated encryption (AES-GCM),
and proper key management for PHI protection.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets
import struct
import time
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import BinaryIO, Union

from cryptography.exceptions import InvalidTag
from cryptography.hazmat.backends import default_backend
from cryptography.hazmat.primitives import hashes, padding
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.kdf.scrypt import Scrypt


class EncryptionError(Exception):
    """Base exception for encryption operations."""


class DecryptionError(Exception):
    """Exception raised when decryption fails."""


class KeyDerivationError(Exception):
    """Exception raised when key derivation fails."""


class IntegrityError(Exception):
    """Exception raised when data integrity check fails."""


class EncryptionAlgorithm(str, Enum):
    """Supported encryption algorithms."""

    AES_256_GCM = "aes-256-gcm"
    AES_256_CBC = "aes-256-cbc"


class KeyDerivationFunction(str, Enum):
    """Supported key derivation functions."""

    PBKDF2 = "pbkdf2"
    SCRYPT = "scrypt"


# Security constants
AES_KEY_SIZE = 32  # 256 bits
AES_BLOCK_SIZE = 16  # 128 bits
GCM_NONCE_SIZE = 12  # 96 bits (recommended for GCM)
GCM_TAG_SIZE = 16  # 128 bits
PBKDF2_ITERATIONS = 600_000  # OWASP recommended minimum
SCRYPT_N = 2**14  # CPU/memory cost parameter
SCRYPT_R = 8  # Block size parameter
SCRYPT_P = 1  # Parallelization parameter
SALT_SIZE = 32  # 256 bits
CHUNK_SIZE = 64 * 1024  # 64KB for streaming encryption


@dataclass(slots=True)
class EncryptionConfig:
    """Configuration for encryption operations."""

    algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256_GCM
    kdf: KeyDerivationFunction = KeyDerivationFunction.PBKDF2
    pbkdf2_iterations: int = PBKDF2_ITERATIONS
    scrypt_n: int = SCRYPT_N
    scrypt_r: int = SCRYPT_R
    scrypt_p: int = SCRYPT_P
    chunk_size: int = CHUNK_SIZE
    include_timestamp: bool = True
    version: int = 1


@dataclass(slots=True)
class EncryptedData:
    """Container for encrypted data with metadata."""

    ciphertext: bytes
    nonce: bytes
    salt: bytes
    tag: bytes | None = None
    algorithm: EncryptionAlgorithm = EncryptionAlgorithm.AES_256_GCM
    version: int = 1
    timestamp: int = field(default_factory=lambda: int(time.time()))
    metadata: dict | None = None

    def to_bytes(self) -> bytes:
        """Serialize encrypted data to bytes with header."""
        header = struct.pack(
            ">BBBBII",
            self.version,
            len(self.algorithm.value),
            len(self.salt),
            len(self.nonce),
            len(self.tag) if self.tag else 0,
            self.timestamp,
        )
        algo_bytes = self.algorithm.value.encode("utf-8")
        tag_bytes = self.tag if self.tag else b""

        return (
            header
            + algo_bytes
            + self.salt
            + self.nonce
            + tag_bytes
            + self.ciphertext
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> EncryptedData:
        """Deserialize encrypted data from bytes."""
        if len(data) < 14:
            raise DecryptionError("Invalid encrypted data format: too short")

        header_size = struct.calcsize(">BBBBII")
        header = struct.unpack(">BBBBII", data[:header_size])
        version, algo_len, salt_len, nonce_len, tag_len, timestamp = header

        offset = header_size
        algorithm = EncryptionAlgorithm(data[offset : offset + algo_len].decode("utf-8"))
        offset += algo_len

        salt = data[offset : offset + salt_len]
        offset += salt_len

        nonce = data[offset : offset + nonce_len]
        offset += nonce_len

        tag = data[offset : offset + tag_len] if tag_len > 0 else None
        offset += tag_len

        ciphertext = data[offset:]

        return cls(
            ciphertext=ciphertext,
            nonce=nonce,
            salt=salt,
            tag=tag,
            algorithm=algorithm,
            version=version,
            timestamp=timestamp,
        )

    def to_base64(self) -> str:
        """Encode encrypted data as URL-safe base64 string."""
        return base64.urlsafe_b64encode(self.to_bytes()).decode("utf-8")

    @classmethod
    def from_base64(cls, data: str) -> EncryptedData:
        """Decode encrypted data from URL-safe base64 string."""
        try:
            raw_bytes = base64.urlsafe_b64decode(data.encode("utf-8"))
            return cls.from_bytes(raw_bytes)
        except Exception as e:
            raise DecryptionError(f"Failed to decode base64 data: {e}") from e


class KeyManager:
    """
    Secure key management for encryption operations.

    Handles key derivation, rotation, and secure storage patterns.
    """

    def __init__(
        self,
        master_key: bytes | str | None = None,
        config: EncryptionConfig | None = None,
    ) -> None:
        """
        Initialize key manager.

        Args:
            master_key: Master encryption key (32 bytes for AES-256).
            config: Encryption configuration.
        """
        self._config = config or EncryptionConfig()
        self._master_key: bytes | None = None

        if master_key is not None:
            self.set_master_key(master_key)

    def set_master_key(self, key: bytes | str) -> None:
        """
        Set the master encryption key.

        Args:
            key: Master key as bytes or hex/base64 encoded string.

        Raises:
            KeyDerivationError: If key is invalid.
        """
        if isinstance(key, str):
            # Try hex decoding first
            try:
                key_bytes = bytes.fromhex(key)
            except ValueError:
                # Try base64 decoding
                try:
                    key_bytes = base64.urlsafe_b64decode(key)
                except Exception:
                    # Use as passphrase for key derivation
                    key_bytes = self._derive_key_from_passphrase(key)
        else:
            key_bytes = key

        if len(key_bytes) != AES_KEY_SIZE:
            raise KeyDerivationError(
                f"Master key must be {AES_KEY_SIZE} bytes, got {len(key_bytes)}"
            )

        self._master_key = key_bytes

    def _derive_key_from_passphrase(self, passphrase: str) -> bytes:
        """Derive a key from a passphrase using the configured KDF."""
        salt = secrets.token_bytes(SALT_SIZE)
        return self.derive_key(passphrase.encode("utf-8"), salt)

    def derive_key(self, password: bytes, salt: bytes) -> bytes:
        """
        Derive an encryption key from password and salt.

        Args:
            password: Password or passphrase bytes.
            salt: Cryptographic salt.

        Returns:
            Derived key bytes.
        """
        if self._config.kdf == KeyDerivationFunction.PBKDF2:
            kdf = PBKDF2HMAC(
                algorithm=hashes.SHA256(),
                length=AES_KEY_SIZE,
                salt=salt,
                iterations=self._config.pbkdf2_iterations,
                backend=default_backend(),
            )
            return kdf.derive(password)
        elif self._config.kdf == KeyDerivationFunction.SCRYPT:
            kdf = Scrypt(
                salt=salt,
                length=AES_KEY_SIZE,
                n=self._config.scrypt_n,
                r=self._config.scrypt_r,
                p=self._config.scrypt_p,
                backend=default_backend(),
            )
            return kdf.derive(password)
        else:
            raise KeyDerivationError(f"Unknown KDF: {self._config.kdf}")

    def get_encryption_key(self, salt: bytes | None = None) -> tuple[bytes, bytes]:
        """
        Get an encryption key derived from the master key.

        Args:
            salt: Optional salt (generated if not provided).

        Returns:
            Tuple of (derived_key, salt).
        """
        if self._master_key is None:
            raise KeyDerivationError("Master key not set")

        salt = salt or secrets.token_bytes(SALT_SIZE)
        derived_key = self.derive_key(self._master_key, salt)
        return derived_key, salt

    @staticmethod
    def generate_key() -> bytes:
        """Generate a cryptographically secure random key."""
        return secrets.token_bytes(AES_KEY_SIZE)

    @staticmethod
    def generate_nonce() -> bytes:
        """Generate a cryptographically secure random nonce."""
        return secrets.token_bytes(GCM_NONCE_SIZE)

    def rotate_key(self, new_key: bytes | str) -> bytes:
        """
        Rotate to a new master key.

        Args:
            new_key: New master key.

        Returns:
            Previous master key for re-encryption.
        """
        old_key = self._master_key
        self.set_master_key(new_key)
        return old_key if old_key else b""


class AESEncryptor:
    """
    AES-256 encryption implementation with GCM mode.

    Provides authenticated encryption with associated data (AEAD),
    ensuring both confidentiality and integrity of encrypted data.
    """

    def __init__(
        self,
        key_manager: KeyManager | None = None,
        config: EncryptionConfig | None = None,
    ) -> None:
        """
        Initialize AES encryptor.

        Args:
            key_manager: Key manager instance.
            config: Encryption configuration.
        """
        self._config = config or EncryptionConfig()
        self._key_manager = key_manager or KeyManager(config=self._config)

    def encrypt(
        self,
        plaintext: bytes,
        associated_data: bytes | None = None,
    ) -> EncryptedData:
        """
        Encrypt data using AES-256-GCM.

        Args:
            plaintext: Data to encrypt.
            associated_data: Additional authenticated data (not encrypted).

        Returns:
            EncryptedData containing ciphertext and metadata.

        Raises:
            EncryptionError: If encryption fails.
        """
        try:
            key, salt = self._key_manager.get_encryption_key()
            nonce = self._key_manager.generate_nonce()

            if self._config.algorithm == EncryptionAlgorithm.AES_256_GCM:
                return self._encrypt_gcm(plaintext, key, nonce, salt, associated_data)
            elif self._config.algorithm == EncryptionAlgorithm.AES_256_CBC:
                return self._encrypt_cbc(plaintext, key, nonce, salt)
            else:
                raise EncryptionError(f"Unsupported algorithm: {self._config.algorithm}")
        except Exception as e:
            if isinstance(e, EncryptionError):
                raise
            raise EncryptionError(f"Encryption failed: {e}") from e

    def _encrypt_gcm(
        self,
        plaintext: bytes,
        key: bytes,
        nonce: bytes,
        salt: bytes,
        associated_data: bytes | None = None,
    ) -> EncryptedData:
        """Encrypt using AES-GCM mode."""
        aesgcm = AESGCM(key)
        ciphertext = aesgcm.encrypt(nonce, plaintext, associated_data)

        # GCM appends tag to ciphertext, extract it
        tag = ciphertext[-GCM_TAG_SIZE:]
        ciphertext = ciphertext[:-GCM_TAG_SIZE]

        return EncryptedData(
            ciphertext=ciphertext,
            nonce=nonce,
            salt=salt,
            tag=tag,
            algorithm=EncryptionAlgorithm.AES_256_GCM,
            version=self._config.version,
        )

    def _encrypt_cbc(
        self,
        plaintext: bytes,
        key: bytes,
        iv: bytes,
        salt: bytes,
    ) -> EncryptedData:
        """Encrypt using AES-CBC mode with PKCS7 padding."""
        # Pad plaintext to block size
        padder = padding.PKCS7(AES_BLOCK_SIZE * 8).padder()
        padded_data = padder.update(plaintext) + padder.finalize()

        # Use first 16 bytes of nonce as IV for CBC
        iv = iv[:AES_BLOCK_SIZE] if len(iv) >= AES_BLOCK_SIZE else iv.ljust(AES_BLOCK_SIZE, b"\x00")

        cipher = Cipher(
            algorithms.AES(key),
            modes.CBC(iv),
            backend=default_backend(),
        )
        encryptor = cipher.encryptor()
        ciphertext = encryptor.update(padded_data) + encryptor.finalize()

        # Calculate HMAC for integrity
        hmac_key = hashlib.sha256(key + b"hmac").digest()
        tag = hmac.new(hmac_key, iv + ciphertext, hashlib.sha256).digest()

        return EncryptedData(
            ciphertext=ciphertext,
            nonce=iv,
            salt=salt,
            tag=tag,
            algorithm=EncryptionAlgorithm.AES_256_CBC,
            version=self._config.version,
        )

    def decrypt(
        self,
        encrypted_data: EncryptedData,
        associated_data: bytes | None = None,
    ) -> bytes:
        """
        Decrypt data using AES-256-GCM or CBC.

        Args:
            encrypted_data: Encrypted data container.
            associated_data: Additional authenticated data (must match encryption).

        Returns:
            Decrypted plaintext bytes.

        Raises:
            DecryptionError: If decryption or authentication fails.
        """
        try:
            key, _ = self._key_manager.get_encryption_key(encrypted_data.salt)

            if encrypted_data.algorithm == EncryptionAlgorithm.AES_256_GCM:
                return self._decrypt_gcm(encrypted_data, key, associated_data)
            elif encrypted_data.algorithm == EncryptionAlgorithm.AES_256_CBC:
                return self._decrypt_cbc(encrypted_data, key)
            else:
                raise DecryptionError(f"Unsupported algorithm: {encrypted_data.algorithm}")
        except (InvalidTag, IntegrityError) as e:
            raise DecryptionError(f"Authentication failed: data may be tampered") from e
        except Exception as e:
            if isinstance(e, DecryptionError):
                raise
            raise DecryptionError(f"Decryption failed: {e}") from e

    def _decrypt_gcm(
        self,
        encrypted_data: EncryptedData,
        key: bytes,
        associated_data: bytes | None = None,
    ) -> bytes:
        """Decrypt using AES-GCM mode."""
        aesgcm = AESGCM(key)
        # Reconstruct ciphertext with tag for GCM
        ciphertext_with_tag = encrypted_data.ciphertext + encrypted_data.tag
        return aesgcm.decrypt(encrypted_data.nonce, ciphertext_with_tag, associated_data)

    def _decrypt_cbc(
        self,
        encrypted_data: EncryptedData,
        key: bytes,
    ) -> bytes:
        """Decrypt using AES-CBC mode with PKCS7 unpadding."""
        # Verify HMAC first
        hmac_key = hashlib.sha256(key + b"hmac").digest()
        expected_tag = hmac.new(
            hmac_key,
            encrypted_data.nonce + encrypted_data.ciphertext,
            hashlib.sha256,
        ).digest()

        if not hmac.compare_digest(encrypted_data.tag or b"", expected_tag):
            raise IntegrityError("HMAC verification failed")

        cipher = Cipher(
            algorithms.AES(key),
            modes.CBC(encrypted_data.nonce),
            backend=default_backend(),
        )
        decryptor = cipher.decryptor()
        padded_data = decryptor.update(encrypted_data.ciphertext) + decryptor.finalize()

        # Remove padding
        unpadder = padding.PKCS7(AES_BLOCK_SIZE * 8).unpadder()
        return unpadder.update(padded_data) + unpadder.finalize()


class FileEncryptor:
    """
    File encryption with streaming support for large files.

    Provides memory-efficient encryption of files without loading
    entire contents into memory.
    """

    def __init__(
        self,
        key_manager: KeyManager,
        config: EncryptionConfig | None = None,
    ) -> None:
        """
        Initialize file encryptor.

        Args:
            key_manager: Key manager instance.
            config: Encryption configuration.
        """
        self._config = config or EncryptionConfig()
        self._key_manager = key_manager
        self._encryptor = AESEncryptor(key_manager, config)

    def encrypt_file(
        self,
        input_path: Path | str,
        output_path: Path | str | None = None,
        delete_original: bool = False,
    ) -> Path:
        """
        Encrypt a file using AES-256-GCM.

        Args:
            input_path: Path to file to encrypt.
            output_path: Output path (default: input_path + .enc).
            delete_original: Securely delete original after encryption.

        Returns:
            Path to encrypted file.
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Input file not found: {input_path}")

        output_path = Path(output_path) if output_path else input_path.with_suffix(input_path.suffix + ".enc")

        # Read and encrypt file contents
        with open(input_path, "rb") as f:
            plaintext = f.read()

        # Add file metadata as associated data
        associated_data = f"{input_path.name}:{input_path.stat().st_size}".encode("utf-8")
        encrypted_data = self._encryptor.encrypt(plaintext, associated_data)

        # Write encrypted data
        with open(output_path, "wb") as f:
            f.write(encrypted_data.to_bytes())

        if delete_original:
            self._secure_delete(input_path)

        return output_path

    def decrypt_file(
        self,
        input_path: Path | str,
        output_path: Path | str | None = None,
        delete_encrypted: bool = False,
    ) -> Path:
        """
        Decrypt a file using AES-256-GCM.

        Args:
            input_path: Path to encrypted file.
            output_path: Output path (default: removes .enc suffix).
            delete_encrypted: Delete encrypted file after decryption.

        Returns:
            Path to decrypted file.
        """
        input_path = Path(input_path)
        if not input_path.exists():
            raise FileNotFoundError(f"Encrypted file not found: {input_path}")

        if output_path is None:
            if input_path.suffix == ".enc":
                output_path = input_path.with_suffix("")
            else:
                output_path = input_path.with_suffix(input_path.suffix + ".dec")
        output_path = Path(output_path)

        # Read encrypted data
        with open(input_path, "rb") as f:
            encrypted_bytes = f.read()

        encrypted_data = EncryptedData.from_bytes(encrypted_bytes)

        # Use original filename as associated data if available
        associated_data = f"{output_path.name}:{len(encrypted_data.ciphertext)}".encode("utf-8")

        try:
            plaintext = self._encryptor.decrypt(encrypted_data, associated_data)
        except DecryptionError:
            # Retry without associated data for backward compatibility
            plaintext = self._encryptor.decrypt(encrypted_data, None)

        # Write decrypted data
        with open(output_path, "wb") as f:
            f.write(plaintext)

        if delete_encrypted:
            input_path.unlink()

        return output_path

    def encrypt_stream(
        self,
        input_stream: BinaryIO,
        output_stream: BinaryIO,
    ) -> tuple[bytes, bytes]:
        """
        Encrypt a stream using chunked encryption.

        Args:
            input_stream: Input binary stream.
            output_stream: Output binary stream.

        Returns:
            Tuple of (salt, nonce) for decryption.
        """
        key, salt = self._key_manager.get_encryption_key()
        nonce = self._key_manager.generate_nonce()

        # Write header with salt and nonce
        header = struct.pack(">BB", len(salt), len(nonce)) + salt + nonce
        output_stream.write(header)

        # Encrypt in chunks
        chunk_num = 0
        while True:
            chunk = input_stream.read(self._config.chunk_size)
            if not chunk:
                break

            # Create unique nonce for each chunk
            chunk_nonce = hashlib.sha256(nonce + struct.pack(">Q", chunk_num)).digest()[:GCM_NONCE_SIZE]
            aesgcm = AESGCM(key)
            encrypted_chunk = aesgcm.encrypt(chunk_nonce, chunk, None)

            # Write chunk length and data
            output_stream.write(struct.pack(">I", len(encrypted_chunk)))
            output_stream.write(encrypted_chunk)
            chunk_num += 1

        # Write end marker
        output_stream.write(struct.pack(">I", 0))

        return salt, nonce

    def _secure_delete(self, path: Path, passes: int = 3) -> None:
        """Securely delete a file by overwriting with random data."""
        from src.security.data_cleanup import SecureDataCleanup

        cleanup = SecureDataCleanup(overwrite_passes=passes)
        cleanup.secure_delete_file(path)


class EncryptionService:
    """
    High-level encryption service for the application.

    Provides a unified interface for all encryption operations
    with automatic key management and configuration.
    """

    _instance: EncryptionService | None = None

    def __init__(
        self,
        master_key: bytes | str | None = None,
        config: EncryptionConfig | None = None,
    ) -> None:
        """
        Initialize encryption service.

        Args:
            master_key: Master encryption key.
            config: Encryption configuration.
        """
        self._config = config or EncryptionConfig()
        self._key_manager = KeyManager(master_key, self._config)
        self._encryptor = AESEncryptor(self._key_manager, self._config)
        self._file_encryptor = FileEncryptor(self._key_manager, self._config)

    @classmethod
    def get_instance(
        cls,
        master_key: bytes | str | None = None,
        config: EncryptionConfig | None = None,
    ) -> EncryptionService:
        """Get or create singleton encryption service instance."""
        if cls._instance is None:
            cls._instance = cls(master_key, config)
        return cls._instance

    @classmethod
    def reset_instance(cls) -> None:
        """Reset singleton instance (for testing)."""
        cls._instance = None

    def set_master_key(self, key: bytes | str) -> None:
        """Set the master encryption key."""
        self._key_manager.set_master_key(key)

    def encrypt(
        self,
        data: bytes | str,
        associated_data: bytes | str | None = None,
    ) -> str:
        """
        Encrypt data and return base64-encoded result.

        Args:
            data: Data to encrypt (bytes or string).
            associated_data: Optional associated data.

        Returns:
            Base64-encoded encrypted data.
        """
        if isinstance(data, str):
            data = data.encode("utf-8")
        if isinstance(associated_data, str):
            associated_data = associated_data.encode("utf-8")

        encrypted = self._encryptor.encrypt(data, associated_data)
        return encrypted.to_base64()

    def decrypt(
        self,
        encrypted_data: str,
        associated_data: bytes | str | None = None,
    ) -> bytes:
        """
        Decrypt base64-encoded data.

        Args:
            encrypted_data: Base64-encoded encrypted data.
            associated_data: Optional associated data.

        Returns:
            Decrypted bytes.
        """
        if isinstance(associated_data, str):
            associated_data = associated_data.encode("utf-8")

        encrypted = EncryptedData.from_base64(encrypted_data)
        return self._encryptor.decrypt(encrypted, associated_data)

    def encrypt_file(
        self,
        input_path: Path | str,
        output_path: Path | str | None = None,
        delete_original: bool = False,
    ) -> Path:
        """Encrypt a file."""
        return self._file_encryptor.encrypt_file(input_path, output_path, delete_original)

    def decrypt_file(
        self,
        input_path: Path | str,
        output_path: Path | str | None = None,
        delete_encrypted: bool = False,
    ) -> Path:
        """Decrypt a file."""
        return self._file_encryptor.decrypt_file(input_path, output_path, delete_encrypted)

    def generate_key(self) -> str:
        """Generate a new random encryption key."""
        return base64.urlsafe_b64encode(KeyManager.generate_key()).decode("utf-8")

    def hash_data(self, data: bytes | str, salt: bytes | None = None) -> str:
        """
        Create a secure hash of data using SHA-256.

        Args:
            data: Data to hash.
            salt: Optional salt.

        Returns:
            Hex-encoded hash.
        """
        if isinstance(data, str):
            data = data.encode("utf-8")
        salt = salt or secrets.token_bytes(16)
        return hashlib.sha256(salt + data).hexdigest()

    def verify_integrity(self, data: bytes, expected_hash: str, salt: bytes) -> bool:
        """
        Verify data integrity using hash comparison.

        Args:
            data: Data to verify.
            expected_hash: Expected hash value.
            salt: Salt used for hashing.

        Returns:
            True if integrity verified.
        """
        actual_hash = self.hash_data(data, salt)
        return hmac.compare_digest(actual_hash, expected_hash)
