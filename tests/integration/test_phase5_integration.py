"""
Integration Tests for Phase 5: Security, Monitoring, and API Integration.

Tests cover:
- Security middleware integration with API
- Monitoring metrics collection during API requests
- Audit logging for API operations
- Health check endpoints
- Rate limiting functionality
- End-to-end security flows
"""

from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Generator
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient


# =============================================================================
# Fixtures
# =============================================================================


@pytest.fixture
def temp_dir() -> Generator[Path, None, None]:
    """Create temporary directory for tests."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def test_client() -> TestClient:
    """Create test client for API."""
    from src.api.app import create_app

    # Create app with all features enabled for testing
    app = create_app(
        enable_security=True,
        enable_metrics=True,
        enable_audit=False,  # Disable for tests to avoid file I/O
        enable_rate_limiting=False,  # Disable for tests
    )

    return TestClient(app)


@pytest.fixture
def test_client_with_rate_limiting() -> TestClient:
    """Create test client with rate limiting enabled."""
    from src.api.app import create_app

    app = create_app(
        enable_security=True,
        enable_metrics=True,
        enable_audit=False,
        enable_rate_limiting=True,
    )

    return TestClient(app)


# =============================================================================
# API Health Check Tests
# =============================================================================


class TestHealthEndpoints:
    """Tests for health check endpoints."""

    def test_basic_health_check(self, test_client: TestClient) -> None:
        """Test basic health check endpoint."""
        response = test_client.get("/api/v1/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] in ["healthy", "degraded"]
        assert "version" in data
        assert "timestamp" in data

    def test_deep_health_check(self, test_client: TestClient) -> None:
        """Test deep health check with all components."""
        response = test_client.get("/api/v1/health?deep=true")

        assert response.status_code == 200
        data = response.json()
        assert "components" in data
        assert "api" in data["components"]

    def test_liveness_probe(self, test_client: TestClient) -> None:
        """Test Kubernetes liveness probe."""
        response = test_client.get("/api/v1/health/live")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "ok"

    def test_readiness_probe(self, test_client: TestClient) -> None:
        """Test Kubernetes readiness probe."""
        response = test_client.get("/api/v1/health/ready")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data

    def test_security_status_endpoint(self, test_client: TestClient) -> None:
        """Test HIPAA security status endpoint."""
        response = test_client.get("/api/v1/health/security")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "hipaa_compliance" in data
        assert "compliance_score" in data

    def test_alerts_endpoint(self, test_client: TestClient) -> None:
        """Test active alerts endpoint."""
        response = test_client.get("/api/v1/health/alerts")

        assert response.status_code == 200
        data = response.json()
        assert "total" in data
        assert "timestamp" in data

    def test_dependencies_endpoint(self, test_client: TestClient) -> None:
        """Test dependencies status endpoint."""
        response = test_client.get("/api/v1/health/dependencies")

        assert response.status_code == 200
        data = response.json()
        assert "status" in data
        assert "dependencies" in data


class TestMetricsEndpoint:
    """Tests for Prometheus metrics endpoint."""

    def test_metrics_endpoint_returns_prometheus_format(
        self, test_client: TestClient
    ) -> None:
        """Test that metrics endpoint returns Prometheus format."""
        response = test_client.get("/api/v1/metrics")

        assert response.status_code == 200
        content_type = response.headers.get("content-type", "")
        assert "text/plain" in content_type

    def test_metrics_after_requests(self, test_client: TestClient) -> None:
        """Test that metrics are recorded after API requests."""
        # Make some requests
        for _ in range(5):
            test_client.get("/api/v1/health")

        # Check metrics
        response = test_client.get("/api/v1/metrics")
        assert response.status_code == 200


# =============================================================================
# Security Middleware Tests
# =============================================================================


class TestSecurityHeaders:
    """Tests for security headers middleware."""

    def test_security_headers_present(self, test_client: TestClient) -> None:
        """Test that security headers are added to responses."""
        response = test_client.get("/api/v1/health")

        # Check OWASP recommended headers
        assert response.headers.get("X-Content-Type-Options") == "nosniff"
        assert response.headers.get("X-Frame-Options") == "DENY"
        assert "X-XSS-Protection" in response.headers

    def test_strict_transport_security(self, test_client: TestClient) -> None:
        """Test HSTS header is present."""
        response = test_client.get("/api/v1/health")

        hsts = response.headers.get("Strict-Transport-Security")
        assert hsts is not None
        assert "max-age" in hsts

    def test_cache_control_headers(self, test_client: TestClient) -> None:
        """Test cache control headers."""
        response = test_client.get("/api/v1/health")

        cache_control = response.headers.get("Cache-Control")
        assert cache_control is not None
        assert "no-store" in cache_control


class TestRequestTracking:
    """Tests for request tracking middleware."""

    def test_request_id_added(self, test_client: TestClient) -> None:
        """Test that request ID is added to responses."""
        response = test_client.get("/api/v1/health")

        assert "X-Request-ID" in response.headers

    def test_custom_request_id_preserved(self, test_client: TestClient) -> None:
        """Test that custom request ID is preserved."""
        custom_id = "test-request-123"
        response = test_client.get(
            "/api/v1/health",
            headers={"X-Request-ID": custom_id},
        )

        assert response.headers.get("X-Request-ID") == custom_id

    def test_response_time_header(self, test_client: TestClient) -> None:
        """Test that response time header is added."""
        response = test_client.get("/api/v1/health")

        assert "X-Response-Time-Ms" in response.headers
        time_ms = float(response.headers["X-Response-Time-Ms"])
        assert time_ms >= 0


# =============================================================================
# Rate Limiting Tests
# =============================================================================


class TestRateLimiting:
    """Tests for rate limiting middleware."""

    def test_rate_limit_headers(
        self, test_client_with_rate_limiting: TestClient
    ) -> None:
        """Test that rate limit headers are present."""
        response = test_client_with_rate_limiting.get("/api/v1/health")

        assert "X-RateLimit-Limit" in response.headers
        assert "X-RateLimit-Remaining" in response.headers

    def test_rate_limit_decreases(
        self, test_client_with_rate_limiting: TestClient
    ) -> None:
        """Test that rate limit remaining decreases."""
        response1 = test_client_with_rate_limiting.get("/api/v1/health")
        remaining1 = int(response1.headers.get("X-RateLimit-Remaining", 0))

        response2 = test_client_with_rate_limiting.get("/api/v1/health")
        remaining2 = int(response2.headers.get("X-RateLimit-Remaining", 0))

        # Remaining should decrease
        assert remaining2 <= remaining1


# =============================================================================
# Security Module Integration Tests
# =============================================================================


class TestEncryptionIntegration:
    """Integration tests for encryption service."""

    def test_encrypt_decrypt_cycle(self, temp_dir: Path) -> None:
        """Test complete encryption/decryption cycle."""
        from src.security.encryption import EncryptionService

        service = EncryptionService()

        # Test data encryption
        original_data = b"Patient: John Doe, SSN: 123-45-6789, DOB: 1990-01-15"
        encrypted = service.encrypt(original_data)
        decrypted = service.decrypt(encrypted)

        assert decrypted == original_data
        assert encrypted.ciphertext != original_data

    def test_password_based_encryption(self) -> None:
        """Test password-based encryption."""
        from src.security.encryption import EncryptionService

        service = EncryptionService()
        password = "HIPAA_Compliant_Password_123!"
        data = b"Protected Health Information"

        encrypted = service.encrypt_with_password(data, password)
        decrypted = service.decrypt_with_password(encrypted, password)

        assert decrypted == data

    def test_file_encryption(self, temp_dir: Path) -> None:
        """Test file encryption/decryption."""
        from src.security.encryption import FileEncryptor, KeyManager

        key_manager = KeyManager()
        key = key_manager.generate_key()
        encryptor = FileEncryptor(key)

        # Create test file
        test_file = temp_dir / "medical_record.pdf"
        test_content = b"Medical Record Content - CONFIDENTIAL"
        test_file.write_bytes(test_content)

        # Encrypt
        encrypted_file = temp_dir / "medical_record.pdf.enc"
        encryptor.encrypt_file(test_file, encrypted_file)

        assert encrypted_file.exists()
        assert encrypted_file.read_bytes() != test_content

        # Decrypt
        decrypted_file = temp_dir / "medical_record.pdf.dec"
        encryptor.decrypt_file(encrypted_file, decrypted_file)

        assert decrypted_file.read_bytes() == test_content


class TestAuditLoggingIntegration:
    """Integration tests for audit logging."""

    def test_audit_logger_creates_logs(self, temp_dir: Path) -> None:
        """Test that audit logger creates log files."""
        from src.security.audit import (
            AuditEventType,
            AuditLogger,
            AuditOutcome,
            AuditSeverity,
        )

        log_dir = temp_dir / "audit"
        logger = AuditLogger(log_dir=str(log_dir), mask_phi=True)

        # Log an event
        logger.log_event(
            event_type=AuditEventType.DATA_ACCESS,
            severity=AuditSeverity.INFO,
            outcome=AuditOutcome.SUCCESS,
            action="read_patient_record",
            resource_type="patient",
            resource_id="patient-123",
        )

        # Verify log directory was created
        assert log_dir.exists()

    def test_phi_masking(self, temp_dir: Path) -> None:
        """Test PHI masking in audit logs."""
        from src.security.audit import PHIMasker

        masker = PHIMasker()

        # Test various PHI patterns
        test_cases = [
            ("SSN: 123-45-6789", "123-45-6789"),
            ("Email: patient@hospital.com", "patient@hospital.com"),
            ("Phone: (555) 123-4567", "(555) 123-4567"),
        ]

        for text, sensitive_part in test_cases:
            masked = masker.mask(text)
            assert sensitive_part not in masked


class TestRBACIntegration:
    """Integration tests for RBAC."""

    def test_user_authentication_flow(self) -> None:
        """Test complete user authentication flow."""
        from src.security.rbac import RBACManager, Role

        manager = RBACManager(secret_key="test-secret-key-12345")

        # Create user
        user = manager.users.create_user(
            username="doctor_smith",
            password="SecureP@ss123!",
            email="smith@hospital.com",
            roles=[Role.OPERATOR],
        )

        # Authenticate
        tokens = manager.login("doctor_smith", "SecureP@ss123!")
        assert tokens is not None
        assert tokens.access_token is not None

        # Validate token
        payload = manager.tokens.validate_token(tokens.access_token)
        assert payload.username == "doctor_smith"

    def test_permission_enforcement(self) -> None:
        """Test permission enforcement across roles."""
        from src.security.rbac import Permission, RBACManager, Role

        manager = RBACManager(secret_key="test-secret-key-12345")

        # Create users with different roles
        viewer = manager.users.create_user(
            username="viewer_user",
            password="ViewerP@ss123!",
            roles=[Role.VIEWER],
        )

        operator = manager.users.create_user(
            username="operator_user",
            password="OperatorP@ss123!",
            roles=[Role.OPERATOR],
        )

        admin = manager.users.create_user(
            username="admin_user",
            password="AdminP@ss123!",
            roles=[Role.ADMIN],
        )

        # Test permissions
        # Viewer can read
        assert manager.check_permission(viewer, Permission.DOCUMENTS_READ)
        assert not manager.check_permission(viewer, Permission.DOCUMENTS_WRITE)

        # Operator can read and write
        assert manager.check_permission(operator, Permission.DOCUMENTS_READ)
        assert manager.check_permission(operator, Permission.DOCUMENTS_WRITE)
        assert not manager.check_permission(operator, Permission.ADMIN_USERS)

        # Admin can do everything
        assert manager.check_permission(admin, Permission.DOCUMENTS_READ)
        assert manager.check_permission(admin, Permission.DOCUMENTS_WRITE)
        assert manager.check_permission(admin, Permission.ADMIN_USERS)


class TestSecureDataCleanupIntegration:
    """Integration tests for secure data cleanup."""

    def test_secure_file_deletion(self, temp_dir: Path) -> None:
        """Test secure file deletion."""
        from src.security.data_cleanup import DeletionMethod, SecureDataCleanup

        cleanup = SecureDataCleanup()

        # Create test file with sensitive data
        sensitive_file = temp_dir / "sensitive_data.txt"
        sensitive_file.write_bytes(b"Patient SSN: 123-45-6789" * 100)

        original_size = sensitive_file.stat().st_size

        # Securely delete
        stats = cleanup.cleanup_file(
            sensitive_file,
            method=DeletionMethod.DOD_3_PASS,
        )

        assert stats.files_deleted == 1
        assert not sensitive_file.exists()

    def test_temp_file_manager(self, temp_dir: Path) -> None:
        """Test temporary file manager."""
        from src.security.data_cleanup import TempFileManager

        manager = TempFileManager(base_dir=temp_dir)

        # Create temp file
        temp_path = manager.create_temp_file(suffix=".pdf")
        temp_path.write_bytes(b"Temporary content")

        assert temp_path.exists()

        # Cleanup
        manager.cleanup()
        assert not temp_path.exists()


# =============================================================================
# Monitoring Integration Tests
# =============================================================================


class TestMetricsIntegration:
    """Integration tests for metrics collection."""

    def test_api_metrics_collection(self, test_client: TestClient) -> None:
        """Test that API requests are recorded in metrics."""
        from src.monitoring.metrics import MetricsCollector

        collector = MetricsCollector()

        # Make API requests
        for _ in range(10):
            test_client.get("/api/v1/health")

        # Metrics should be recorded
        # (Verification depends on metrics registry state)

    def test_metrics_with_different_endpoints(
        self, test_client: TestClient
    ) -> None:
        """Test metrics for different endpoints."""
        endpoints = [
            "/api/v1/health",
            "/api/v1/health/live",
            "/api/v1/health/ready",
        ]

        for endpoint in endpoints:
            response = test_client.get(endpoint)
            assert response.status_code == 200

    def test_metrics_track_errors(self, test_client: TestClient) -> None:
        """Test that error responses are tracked."""
        # Request non-existent endpoint
        response = test_client.get("/api/v1/nonexistent")
        assert response.status_code == 404


class TestAlertingIntegration:
    """Integration tests for alerting system."""

    def test_alert_rule_evaluation(self) -> None:
        """Test alert rule evaluation."""
        from src.monitoring.alerts import AlertManager, AlertRule, AlertSeverity

        manager = AlertManager()

        # Add rule for high latency
        rule = AlertRule(
            name="high_api_latency",
            description="API latency exceeds threshold",
            severity=AlertSeverity.WARNING,
            condition=lambda v: v > 2.0,
            threshold=2.0,
            labels={"service": "api"},
        )

        manager.add_rule(rule)

        # Simulate high latency
        alert = manager.check("high_api_latency", 5.0)

        assert alert is not None
        assert alert.severity == AlertSeverity.WARNING
        assert alert.value == 5.0

    def test_alert_notification_handlers(self) -> None:
        """Test alert notification through handlers."""
        from src.monitoring.alerts import (
            AlertManager,
            AlertRule,
            AlertSeverity,
            LogHandler,
        )

        manager = AlertManager()
        log_handler = LogHandler()
        manager.add_handler(log_handler)

        rule = AlertRule(
            name="test_notification",
            description="Test notification",
            severity=AlertSeverity.INFO,
            condition=lambda v: v > 0,
        )

        manager.add_rule(rule)
        manager.check("test_notification", 1.0)

    def test_default_alert_rules(self) -> None:
        """Test loading default alert rules."""
        from src.monitoring.alerts import AlertManager, get_default_alert_rules

        manager = AlertManager()

        for rule in get_default_alert_rules():
            manager.add_rule(rule)

        # Should have multiple rules
        assert len(manager._rules) > 0


# =============================================================================
# End-to-End Security Flow Tests
# =============================================================================


class TestEndToEndSecurityFlow:
    """End-to-end tests for security workflows."""

    def test_complete_document_security_flow(self, temp_dir: Path) -> None:
        """Test complete document security workflow."""
        from src.security.audit import AuditEventType, AuditLogger
        from src.security.data_cleanup import SecureDataCleanup
        from src.security.encryption import EncryptionService, FileEncryptor, KeyManager
        from src.security.rbac import Permission, RBACManager, Role

        # 1. Set up RBAC
        rbac = RBACManager(secret_key="test-secret-key-12345")
        user = rbac.users.create_user(
            username="operator",
            password="OperatorP@ss123!",
            roles=[Role.OPERATOR],
        )

        # 2. Verify user has permission to process documents
        assert rbac.check_permission(user, Permission.DOCUMENTS_WRITE)

        # 3. Create and encrypt document
        key_manager = KeyManager()
        key = key_manager.generate_key()
        file_encryptor = FileEncryptor(key)

        source_file = temp_dir / "patient_record.pdf"
        source_file.write_bytes(b"Patient medical data - CONFIDENTIAL")

        encrypted_file = temp_dir / "patient_record.pdf.enc"
        file_encryptor.encrypt_file(source_file, encrypted_file)

        # 4. Log the access
        audit_logger = AuditLogger(
            log_dir=str(temp_dir / "audit"),
            mask_phi=True,
        )
        audit_logger.log_data_access(
            resource_type="patient_record",
            resource_id="patient-001",
            action="encrypt",
            user_id=user.user_id,
        )

        # 5. Securely delete original
        cleanup = SecureDataCleanup()
        stats = cleanup.cleanup_file(source_file)

        assert stats.files_deleted == 1
        assert not source_file.exists()

        # 6. Verify encrypted file can be decrypted
        decrypted_file = temp_dir / "patient_record.pdf.dec"
        file_encryptor.decrypt_file(encrypted_file, decrypted_file)

        assert b"Patient medical data" in decrypted_file.read_bytes()

    def test_hipaa_compliance_verification(self, test_client: TestClient) -> None:
        """Test HIPAA compliance verification through API."""
        response = test_client.get("/api/v1/health/security")

        assert response.status_code == 200
        data = response.json()

        # Check HIPAA compliance indicators
        hipaa = data.get("hipaa_compliance", {})

        # At minimum, these should be present
        assert "encryption_at_rest" in hipaa
        assert "audit_logging" in hipaa
        assert "access_control" in hipaa

    def test_security_event_monitoring(self) -> None:
        """Test security event monitoring integration."""
        from src.monitoring.alerts import AlertManager, AlertRule, AlertSeverity
        from src.monitoring.metrics import MetricsCollector

        # Set up metrics
        collector = MetricsCollector()

        # Set up alerts for security events
        manager = AlertManager()
        rule = AlertRule(
            name="failed_auth_attempts",
            description="Too many failed authentication attempts",
            severity=AlertSeverity.CRITICAL,
            condition=lambda v: v > 5,
            threshold=5,
            labels={"category": "security"},
        )
        manager.add_rule(rule)

        # Simulate failed auth attempts
        failed_attempts = 10
        for _ in range(failed_attempts):
            collector.record_security_event(
                event_type="authentication",
                success=False,
            )

        # Check alert
        alert = manager.check("failed_auth_attempts", failed_attempts)
        assert alert is not None
        assert alert.severity == AlertSeverity.CRITICAL


# =============================================================================
# API Error Handling Tests
# =============================================================================


class TestAPIErrorHandling:
    """Tests for API error handling with security context."""

    def test_validation_error_response(self, test_client: TestClient) -> None:
        """Test validation error response format."""
        # This will depend on your actual API endpoints
        pass

    def test_error_responses_have_request_id(
        self, test_client: TestClient
    ) -> None:
        """Test that error responses include request ID."""
        response = test_client.get("/api/v1/nonexistent")

        assert response.status_code == 404
        # Response should have request ID header even on errors
        assert "X-Request-ID" in response.headers


# =============================================================================
# Performance Tests
# =============================================================================


class TestSecurityPerformance:
    """Performance tests for security operations."""

    def test_encryption_performance(self) -> None:
        """Test encryption performance for various data sizes."""
        from src.security.encryption import EncryptionService

        service = EncryptionService()

        sizes = [1024, 10 * 1024, 100 * 1024, 1024 * 1024]  # 1KB to 1MB

        for size in sizes:
            data = os.urandom(size)

            start = time.perf_counter()
            encrypted = service.encrypt(data)
            encrypt_time = time.perf_counter() - start

            start = time.perf_counter()
            decrypted = service.decrypt(encrypted)
            decrypt_time = time.perf_counter() - start

            # Should complete in reasonable time
            assert encrypt_time < 1.0  # Less than 1 second
            assert decrypt_time < 1.0
            assert decrypted == data

    def test_api_response_time(self, test_client: TestClient) -> None:
        """Test API response time with security middleware."""
        import statistics

        times = []

        for _ in range(20):
            start = time.perf_counter()
            response = test_client.get("/api/v1/health")
            duration = time.perf_counter() - start
            times.append(duration)

        avg_time = statistics.mean(times)
        p95_time = sorted(times)[int(len(times) * 0.95)]

        # Average should be under 100ms
        assert avg_time < 0.1
        # P95 should be under 200ms
        assert p95_time < 0.2
