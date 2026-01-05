"""
Webhook callback system for async task notifications.

Provides reliable webhook delivery with retry logic, signature
verification, and comprehensive error handling.
"""

from __future__ import annotations

import asyncio
import hashlib
import hmac
import json
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from typing import Any
from urllib.parse import urlparse

import httpx

from src.config import get_logger


logger = get_logger(__name__)


class WebhookEventType(str, Enum):
    """Types of webhook events."""

    # Document processing events
    PROCESSING_STARTED = "processing.started"
    PROCESSING_COMPLETED = "processing.completed"
    PROCESSING_FAILED = "processing.failed"

    # Batch processing events
    BATCH_STARTED = "batch.started"
    BATCH_COMPLETED = "batch.completed"
    BATCH_FAILED = "batch.failed"

    # Validation events
    VALIDATION_REQUIRED = "validation.human_review_required"

    # Export events
    EXPORT_READY = "export.ready"


class WebhookDeliveryStatus(str, Enum):
    """Status of webhook delivery attempts."""

    PENDING = "pending"
    DELIVERED = "delivered"
    FAILED = "failed"
    RETRYING = "retrying"


@dataclass(slots=True)
class WebhookPayload:
    """
    Webhook notification payload.

    Attributes:
        event_type: Type of event that triggered the webhook.
        processing_id: Document processing ID.
        task_id: Celery task ID.
        timestamp: Event timestamp.
        data: Event-specific data.
    """

    event_type: WebhookEventType
    processing_id: str
    task_id: str
    timestamp: str
    data: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "event_type": self.event_type.value,
            "processing_id": self.processing_id,
            "task_id": self.task_id,
            "timestamp": self.timestamp,
            "data": self.data,
        }

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), default=str)


@dataclass(slots=True)
class WebhookDeliveryResult:
    """
    Result of a webhook delivery attempt.

    Attributes:
        status: Delivery status.
        status_code: HTTP status code (if delivered).
        response_body: Response body (truncated).
        attempts: Number of delivery attempts.
        error: Error message (if failed).
        delivered_at: Timestamp of successful delivery.
    """

    status: WebhookDeliveryStatus
    status_code: int | None = None
    response_body: str | None = None
    attempts: int = 0
    error: str | None = None
    delivered_at: str | None = None


class WebhookClient:
    """
    HTTP client for sending webhook notifications.

    Provides reliable webhook delivery with:
    - Automatic retry with exponential backoff
    - Request signing for security
    - Timeout handling
    - URL validation
    - Response logging
    """

    # Allowed URL schemes
    ALLOWED_SCHEMES = {"http", "https"}

    # Blocked hosts (security)
    BLOCKED_HOSTS = {
        "localhost",
        "127.0.0.1",
        "0.0.0.0",
        "::1",
        "169.254.169.254",  # AWS metadata
        "metadata.google.internal",  # GCP metadata
    }

    def __init__(
        self,
        timeout: float = 30.0,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        signing_secret: str | None = None,
        user_agent: str = "PDFExtraction-Webhook/1.0",
    ) -> None:
        """
        Initialize webhook client.

        Args:
            timeout: Request timeout in seconds.
            max_retries: Maximum number of retry attempts.
            retry_delay: Base delay between retries (exponential backoff).
            signing_secret: Secret key for signing webhooks.
            user_agent: User-Agent header value.
        """
        self._timeout = timeout
        self._max_retries = max_retries
        self._retry_delay = retry_delay
        self._signing_secret = signing_secret
        self._user_agent = user_agent
        self._logger = logger

    def validate_url(self, url: str) -> tuple[bool, str | None]:
        """
        Validate webhook URL for security.

        Args:
            url: URL to validate.

        Returns:
            Tuple of (is_valid, error_message).
        """
        try:
            parsed = urlparse(url)

            # Check scheme
            if parsed.scheme not in self.ALLOWED_SCHEMES:
                return False, f"Invalid scheme: {parsed.scheme}. Must be http or https."

            # Check host
            if not parsed.netloc:
                return False, "URL must have a valid host."

            # Extract hostname (without port)
            hostname = parsed.hostname or ""

            # Check blocked hosts
            if hostname.lower() in self.BLOCKED_HOSTS:
                return False, f"Host not allowed: {hostname}"

            # Check for private network ranges (basic check)
            if hostname.startswith("10.") or hostname.startswith("192.168."):
                self._logger.warning(
                    "webhook_private_ip_warning",
                    url=url,
                    hostname=hostname,
                    message="Webhook URL points to private IP range",
                )

            return True, None

        except Exception as e:
            return False, f"Invalid URL: {e}"

    def _generate_signature(self, payload: str, timestamp: str) -> str:
        """
        Generate HMAC signature for webhook payload.

        Args:
            payload: JSON payload string.
            timestamp: Request timestamp.

        Returns:
            Hex-encoded HMAC signature.
        """
        if not self._signing_secret:
            return ""

        # Create message: timestamp.payload
        message = f"{timestamp}.{payload}"

        signature = hmac.new(
            self._signing_secret.encode("utf-8"),
            message.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

        return f"sha256={signature}"

    def _build_headers(
        self,
        payload: str,
        timestamp: str,
    ) -> dict[str, str]:
        """
        Build request headers.

        Args:
            payload: JSON payload.
            timestamp: Request timestamp.

        Returns:
            Headers dictionary.
        """
        headers = {
            "Content-Type": "application/json",
            "User-Agent": self._user_agent,
            "X-Webhook-Timestamp": timestamp,
        }

        # Add signature if secret configured
        if self._signing_secret:
            signature = self._generate_signature(payload, timestamp)
            headers["X-Webhook-Signature"] = signature

        return headers

    def send_sync(
        self,
        url: str,
        payload: WebhookPayload,
    ) -> WebhookDeliveryResult:
        """
        Send webhook synchronously with retry.

        Args:
            url: Webhook URL.
            payload: Webhook payload.

        Returns:
            Delivery result.
        """
        # Validate URL
        is_valid, error = self.validate_url(url)
        if not is_valid:
            self._logger.error(
                "webhook_url_invalid",
                url=url,
                error=error,
            )
            return WebhookDeliveryResult(
                status=WebhookDeliveryStatus.FAILED,
                error=error,
            )

        payload_json = payload.to_json()
        timestamp = str(int(time.time()))
        headers = self._build_headers(payload_json, timestamp)

        last_error: str | None = None
        attempts = 0

        for attempt in range(self._max_retries + 1):
            attempts = attempt + 1

            try:
                with httpx.Client(timeout=self._timeout) as client:
                    response = client.post(
                        url,
                        content=payload_json,
                        headers=headers,
                    )

                # Log response
                self._logger.info(
                    "webhook_sent",
                    url=url,
                    event_type=payload.event_type.value,
                    processing_id=payload.processing_id,
                    status_code=response.status_code,
                    attempt=attempts,
                )

                # Check for success (2xx status codes)
                if 200 <= response.status_code < 300:
                    return WebhookDeliveryResult(
                        status=WebhookDeliveryStatus.DELIVERED,
                        status_code=response.status_code,
                        response_body=response.text[:500] if response.text else None,
                        attempts=attempts,
                        delivered_at=datetime.now(UTC).isoformat(),
                    )

                # Non-2xx but received response
                last_error = f"HTTP {response.status_code}: {response.text[:200]}"

                # Don't retry on client errors (4xx)
                if 400 <= response.status_code < 500:
                    self._logger.warning(
                        "webhook_client_error",
                        url=url,
                        status_code=response.status_code,
                        message="Not retrying client error",
                    )
                    return WebhookDeliveryResult(
                        status=WebhookDeliveryStatus.FAILED,
                        status_code=response.status_code,
                        response_body=response.text[:500] if response.text else None,
                        attempts=attempts,
                        error=last_error,
                    )

            except httpx.TimeoutException as e:
                last_error = f"Timeout: {e}"
                self._logger.warning(
                    "webhook_timeout",
                    url=url,
                    attempt=attempts,
                    timeout=self._timeout,
                )

            except httpx.RequestError as e:
                last_error = f"Request error: {e}"
                self._logger.warning(
                    "webhook_request_error",
                    url=url,
                    attempt=attempts,
                    error=str(e),
                )

            except Exception as e:
                last_error = f"Unexpected error: {e}"
                self._logger.error(
                    "webhook_unexpected_error",
                    url=url,
                    attempt=attempts,
                    error=str(e),
                    error_type=type(e).__name__,
                )

            # Sleep before retry (exponential backoff)
            if attempt < self._max_retries:
                sleep_time = self._retry_delay * (2**attempt)
                time.sleep(sleep_time)

        # All retries exhausted
        self._logger.error(
            "webhook_delivery_failed",
            url=url,
            event_type=payload.event_type.value,
            processing_id=payload.processing_id,
            attempts=attempts,
            error=last_error,
        )

        return WebhookDeliveryResult(
            status=WebhookDeliveryStatus.FAILED,
            attempts=attempts,
            error=last_error,
        )

    async def send_async(
        self,
        url: str,
        payload: WebhookPayload,
    ) -> WebhookDeliveryResult:
        """
        Send webhook asynchronously with retry.

        Args:
            url: Webhook URL.
            payload: Webhook payload.

        Returns:
            Delivery result.
        """
        # Validate URL
        is_valid, error = self.validate_url(url)
        if not is_valid:
            self._logger.error(
                "webhook_url_invalid",
                url=url,
                error=error,
            )
            return WebhookDeliveryResult(
                status=WebhookDeliveryStatus.FAILED,
                error=error,
            )

        payload_json = payload.to_json()
        timestamp = str(int(time.time()))
        headers = self._build_headers(payload_json, timestamp)

        last_error: str | None = None
        attempts = 0

        for attempt in range(self._max_retries + 1):
            attempts = attempt + 1

            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.post(
                        url,
                        content=payload_json,
                        headers=headers,
                    )

                # Log response
                self._logger.info(
                    "webhook_sent_async",
                    url=url,
                    event_type=payload.event_type.value,
                    processing_id=payload.processing_id,
                    status_code=response.status_code,
                    attempt=attempts,
                )

                # Check for success (2xx status codes)
                if 200 <= response.status_code < 300:
                    return WebhookDeliveryResult(
                        status=WebhookDeliveryStatus.DELIVERED,
                        status_code=response.status_code,
                        response_body=response.text[:500] if response.text else None,
                        attempts=attempts,
                        delivered_at=datetime.now(UTC).isoformat(),
                    )

                # Non-2xx but received response
                last_error = f"HTTP {response.status_code}: {response.text[:200]}"

                # Don't retry on client errors (4xx)
                if 400 <= response.status_code < 500:
                    return WebhookDeliveryResult(
                        status=WebhookDeliveryStatus.FAILED,
                        status_code=response.status_code,
                        response_body=response.text[:500] if response.text else None,
                        attempts=attempts,
                        error=last_error,
                    )

            except httpx.TimeoutException as e:
                last_error = f"Timeout: {e}"
                self._logger.warning(
                    "webhook_timeout_async",
                    url=url,
                    attempt=attempts,
                )

            except httpx.RequestError as e:
                last_error = f"Request error: {e}"
                self._logger.warning(
                    "webhook_request_error_async",
                    url=url,
                    attempt=attempts,
                    error=str(e),
                )

            except Exception as e:
                last_error = f"Unexpected error: {e}"
                self._logger.error(
                    "webhook_unexpected_error_async",
                    url=url,
                    attempt=attempts,
                    error=str(e),
                )

            # Sleep before retry (exponential backoff)
            if attempt < self._max_retries:
                sleep_time = self._retry_delay * (2**attempt)
                await asyncio.sleep(sleep_time)

        # All retries exhausted
        self._logger.error(
            "webhook_delivery_failed_async",
            url=url,
            event_type=payload.event_type.value,
            processing_id=payload.processing_id,
            attempts=attempts,
            error=last_error,
        )

        return WebhookDeliveryResult(
            status=WebhookDeliveryStatus.FAILED,
            attempts=attempts,
            error=last_error,
        )


# Module-level client instance (can be configured at startup)
_webhook_client: WebhookClient | None = None


def get_webhook_client() -> WebhookClient:
    """
    Get or create the webhook client singleton.

    Returns:
        WebhookClient instance.
    """
    global _webhook_client
    if _webhook_client is None:
        _webhook_client = WebhookClient()
    return _webhook_client


def configure_webhook_client(
    signing_secret: str | None = None,
    timeout: float = 30.0,
    max_retries: int = 3,
) -> WebhookClient:
    """
    Configure the webhook client singleton.

    Args:
        signing_secret: Secret for signing webhooks.
        timeout: Request timeout.
        max_retries: Maximum retry attempts.

    Returns:
        Configured WebhookClient instance.
    """
    global _webhook_client
    _webhook_client = WebhookClient(
        signing_secret=signing_secret,
        timeout=timeout,
        max_retries=max_retries,
    )
    return _webhook_client


def send_webhook_notification(
    callback_url: str,
    event_type: WebhookEventType,
    processing_id: str,
    task_id: str,
    data: dict[str, Any] | None = None,
) -> WebhookDeliveryResult:
    """
    Send a webhook notification (synchronously).

    Convenience function for sending webhooks from Celery tasks.

    Args:
        callback_url: URL to send webhook to.
        event_type: Type of event.
        processing_id: Document processing ID.
        task_id: Celery task ID.
        data: Additional event data.

    Returns:
        Delivery result.
    """
    payload = WebhookPayload(
        event_type=event_type,
        processing_id=processing_id,
        task_id=task_id,
        timestamp=datetime.now(UTC).isoformat(),
        data=data or {},
    )

    client = get_webhook_client()
    return client.send_sync(callback_url, payload)


async def send_webhook_notification_async(
    callback_url: str,
    event_type: WebhookEventType,
    processing_id: str,
    task_id: str,
    data: dict[str, Any] | None = None,
) -> WebhookDeliveryResult:
    """
    Send a webhook notification (asynchronously).

    Convenience function for sending webhooks from async code.

    Args:
        callback_url: URL to send webhook to.
        event_type: Type of event.
        processing_id: Document processing ID.
        task_id: Celery task ID.
        data: Additional event data.

    Returns:
        Delivery result.
    """
    payload = WebhookPayload(
        event_type=event_type,
        processing_id=processing_id,
        task_id=task_id,
        timestamp=datetime.now(UTC).isoformat(),
        data=data or {},
    )

    client = get_webhook_client()
    return await client.send_async(callback_url, payload)
