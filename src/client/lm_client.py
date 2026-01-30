"""
LM Studio client for Vision Language Model communication.

Provides a robust client for sending vision requests to a local
LM Studio server with retry logic, connection pooling, and
comprehensive error handling.
"""

import asyncio
import base64
import json
import re
import threading
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any

import httpx
from openai import APIConnectionError, APITimeoutError, AsyncOpenAI, OpenAI, RateLimitError
from tenacity import (
    RetryError,
    Retrying,
    before_sleep_log,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from src.config import get_logger, get_settings
from src.preprocessing.pdf_processor import PageImage


logger = get_logger(__name__)


class LMClientError(Exception):
    """Base exception for LM client errors."""


class LMConnectionError(LMClientError):
    """Raised when connection to LM Studio fails."""


class LMTimeoutError(LMClientError):
    """Raised when request times out."""


class LMRateLimitError(LMClientError):
    """Raised when rate limit is exceeded."""


class LMResponseError(LMClientError):
    """Raised when response parsing fails."""


class LMValidationError(LMClientError):
    """Raised when request validation fails."""


class MessageRole(str, Enum):
    """Chat message roles."""

    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True, slots=True)
class VisionRequest:
    """
    Immutable container for a vision API request.

    Attributes:
        image_data: Base64-encoded image data or data URI.
        prompt: Text prompt for the VLM.
        system_prompt: Optional system prompt for context.
        max_tokens: Maximum tokens in response.
        temperature: Sampling temperature.
        json_mode: Whether to request JSON output.
        request_id: Unique identifier for this request.
    """

    image_data: str
    prompt: str
    system_prompt: str | None = None
    max_tokens: int = 4096
    temperature: float = 0.1
    json_mode: bool = True
    request_id: str = field(default_factory=lambda: f"req_{int(time.time() * 1000)}")

    @classmethod
    def from_page_image(
        cls,
        page: PageImage,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.1,
        json_mode: bool = True,
    ) -> "VisionRequest":
        """
        Create a VisionRequest from a PageImage.

        Args:
            page: PageImage to include in request.
            prompt: Text prompt for extraction.
            system_prompt: Optional system context.
            max_tokens: Maximum response tokens.
            temperature: Sampling temperature.
            json_mode: Request JSON output.

        Returns:
            VisionRequest configured with the page image.
        """
        return cls(
            image_data=page.data_uri,
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            json_mode=json_mode,
        )

    @classmethod
    def from_file(
        cls,
        image_path: Path,
        prompt: str,
        system_prompt: str | None = None,
        max_tokens: int = 4096,
        temperature: float = 0.1,
        json_mode: bool = True,
    ) -> "VisionRequest":
        """
        Create a VisionRequest from an image file.

        Args:
            image_path: Path to image file.
            prompt: Text prompt for extraction.
            system_prompt: Optional system context.
            max_tokens: Maximum response tokens.
            temperature: Sampling temperature.
            json_mode: Request JSON output.

        Returns:
            VisionRequest configured with the image.

        Raises:
            LMValidationError: If file cannot be read.
        """
        if not image_path.exists():
            raise LMValidationError(f"Image file not found: {image_path}")

        # Determine MIME type
        suffix = image_path.suffix.lower()
        mime_types = {
            ".png": "image/png",
            ".jpg": "image/jpeg",
            ".jpeg": "image/jpeg",
            ".gif": "image/gif",
            ".webp": "image/webp",
        }
        mime_type = mime_types.get(suffix, "image/png")

        # Read and encode
        with open(image_path, "rb") as f:
            image_bytes = f.read()

        base64_data = base64.b64encode(image_bytes).decode("utf-8")
        data_uri = f"data:{mime_type};base64,{base64_data}"

        return cls(
            image_data=data_uri,
            prompt=prompt,
            system_prompt=system_prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            json_mode=json_mode,
        )


@dataclass(slots=True)
class VisionResponse:
    """
    Container for vision API response.

    Attributes:
        content: Raw text content from the model.
        parsed_json: Parsed JSON if response was valid JSON.
        model: Model identifier used.
        usage: Token usage statistics.
        latency_ms: Request latency in milliseconds.
        request_id: Original request identifier.
        timestamp: Response timestamp.
    """

    content: str
    parsed_json: dict[str, Any] | None = None
    model: str = ""
    usage: dict[str, int] = field(default_factory=dict)
    latency_ms: int = 0
    request_id: str = ""
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))

    @property
    def has_json(self) -> bool:
        """Check if response contains valid JSON."""
        return self.parsed_json is not None

    @property
    def prompt_tokens(self) -> int:
        """Get number of prompt tokens used."""
        return self.usage.get("prompt_tokens", 0)

    @property
    def completion_tokens(self) -> int:
        """Get number of completion tokens used."""
        return self.usage.get("completion_tokens", 0)

    @property
    def total_tokens(self) -> int:
        """Get total tokens used."""
        return self.usage.get("total_tokens", 0)

    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "content": self.content,
            "parsed_json": self.parsed_json,
            "model": self.model,
            "usage": self.usage,
            "latency_ms": self.latency_ms,
            "request_id": self.request_id,
            "timestamp": self.timestamp.isoformat(),
            "has_json": self.has_json,
        }


class LMStudioClient:
    """
    Robust client for LM Studio Vision Language Model.

    Provides reliable communication with a local LM Studio server,
    including retry logic, connection pooling, and JSON extraction.

    Example:
        client = LMStudioClient()

        # Check server health
        if client.is_healthy():
            request = VisionRequest.from_page_image(page, "Extract patient data")
            response = client.send_vision_request(request)
            if response.has_json:
                data = response.parsed_json
    """

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        timeout: int | None = None,
        max_retries: int | None = None,
        retry_min_wait: int | None = None,
        retry_max_wait: int | None = None,
    ) -> None:
        """
        Initialize the LM Studio client.

        Args:
            base_url: LM Studio server URL. Defaults to settings.
            model: Model identifier. Defaults to settings.
            max_tokens: Default max tokens. Defaults to settings.
            temperature: Default temperature. Defaults to settings.
            timeout: Request timeout in seconds. Defaults to settings.
            max_retries: Maximum retry attempts. Defaults to settings.
            retry_min_wait: Minimum retry wait in seconds. Defaults to settings.
            retry_max_wait: Maximum retry wait in seconds. Defaults to settings.
        """
        settings = get_settings()

        self._base_url = base_url or str(settings.lm_studio.base_url)
        self._model = model or settings.lm_studio.model
        self._max_tokens = max_tokens or settings.lm_studio.max_tokens
        self._temperature = temperature or settings.lm_studio.temperature
        self._timeout = timeout or settings.lm_studio.timeout
        self._max_retries = max_retries or settings.lm_studio.max_retries
        self._retry_min_wait = retry_min_wait or settings.lm_studio.retry_min_wait
        self._retry_max_wait = retry_max_wait or settings.lm_studio.retry_max_wait

        # Thread-local storage for OpenAI clients (thread-safety for concurrent requests)
        # Each thread gets its own client to avoid connection pool conflicts
        self._thread_local = threading.local()

        # Lock for thread-safe initialization
        self._client_lock = threading.Lock()

        # Async client for async operations (created lazily)
        self._async_client: AsyncOpenAI | None = None
        self._async_client_lock = asyncio.Lock()

        # HTTP client for health checks
        self._http_client = httpx.Client(
            base_url=self._base_url.rstrip("/v1"),
            timeout=10.0,
        )

        # Track if closed
        self._closed = False

        # Build retry policy using configurable max_retries (not hardcoded in decorator)
        self._retryer = Retrying(
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(
                multiplier=1,
                min=self._retry_min_wait,
                max=self._retry_max_wait,
            ),
            retry=retry_if_exception_type(
                (APIConnectionError, APITimeoutError, RateLimitError)
            ),
            before_sleep=before_sleep_log(logger, "WARNING"),
            reraise=True,
        )

        # JSON extraction patterns
        self._json_patterns = [
            re.compile(r"```json\s*([\s\S]*?)\s*```", re.MULTILINE),
            re.compile(r"```\s*([\s\S]*?)\s*```", re.MULTILINE),
            re.compile(r"\{[\s\S]*\}", re.MULTILINE),
        ]

        logger.info(
            "lm_client_initialized",
            base_url=self._base_url,
            model=self._model,
            timeout=self._timeout,
        )

    def _get_client(self) -> OpenAI:
        """
        Get a thread-local OpenAI client.

        Each thread gets its own client instance to avoid connection pool
        conflicts under concurrent access. Clients are lazily created.

        Returns:
            Thread-local OpenAI client instance.
        """
        if not hasattr(self._thread_local, "client"):
            with self._client_lock:
                # Double-check after acquiring lock
                if not hasattr(self._thread_local, "client"):
                    self._thread_local.client = OpenAI(
                        base_url=self._base_url,
                        api_key="not-needed",  # LM Studio doesn't require API key
                        timeout=float(self._timeout),
                        max_retries=0,  # We handle retries ourselves
                    )
        return self._thread_local.client

    def is_healthy(self) -> bool:
        """
        Check if the LM Studio server is healthy.

        Returns:
            True if server is responding, False otherwise.
        """
        try:
            response = self._http_client.get("/v1/models")
            return response.status_code == 200
        except Exception:
            return False

    def get_models(self) -> list[str]:
        """
        Get list of available models.

        Returns:
            List of model identifiers.

        Raises:
            LMConnectionError: If connection fails.
        """
        try:
            response = self._http_client.get("/v1/models")
            response.raise_for_status()
            data = response.json()
            return [model["id"] for model in data.get("data", [])]
        except httpx.ConnectError as e:
            raise LMConnectionError(f"Failed to connect to LM Studio: {e}") from e
        except Exception as e:
            raise LMClientError(f"Failed to get models: {e}") from e

    def send_vision_request(
        self,
        request: VisionRequest,
    ) -> VisionResponse:
        """
        Send a vision request with retry logic.

        Args:
            request: VisionRequest to send.

        Returns:
            VisionResponse with model output.

        Raises:
            LMConnectionError: If connection fails after retries.
            LMTimeoutError: If request times out after retries.
            LMResponseError: If response parsing fails.
        """
        start_time = time.perf_counter()

        try:
            response = self._send_with_retry(request)
            latency_ms = int((time.perf_counter() - start_time) * 1000)

            # Extract content
            content = response.choices[0].message.content or ""

            # Attempt JSON extraction
            parsed_json = self._extract_json(content)

            # Build response
            vision_response = VisionResponse(
                content=content,
                parsed_json=parsed_json,
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": (
                        response.usage.completion_tokens if response.usage else 0
                    ),
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                },
                latency_ms=latency_ms,
                request_id=request.request_id,
            )

            logger.info(
                "vision_request_complete",
                request_id=request.request_id,
                latency_ms=latency_ms,
                tokens=vision_response.total_tokens,
                has_json=vision_response.has_json,
            )

            return vision_response

        except RetryError as e:
            latency_ms = int((time.perf_counter() - start_time) * 1000)
            last_exception = e.last_attempt.exception()

            if isinstance(last_exception, APIConnectionError):
                raise LMConnectionError(
                    f"Connection failed after {self._max_retries} retries: {last_exception}"
                ) from e
            if isinstance(last_exception, APITimeoutError):
                raise LMTimeoutError(
                    f"Request timed out after {self._max_retries} retries: {last_exception}"
                ) from e
            raise LMClientError(
                f"Request failed after {self._max_retries} retries: {last_exception}"
            ) from e

    def _send_with_retry(self, request: VisionRequest) -> Any:
        """
        Send request with tenacity retry logic.

        Uses configurable exponential backoff for transient failures.
        The retry count is read from settings at runtime, not hardcoded.
        """
        return self._retryer(self._send_single_request, request)

    def _send_single_request(self, request: VisionRequest) -> Any:
        """Execute a single VLM request (called by the retryer)."""
        # Build messages
        messages: list[dict[str, Any]] = []

        # System prompt
        if request.system_prompt:
            messages.append(
                {
                    "role": MessageRole.SYSTEM.value,
                    "content": request.system_prompt,
                }
            )

        # User message with image and text
        user_content: list[dict[str, Any]] = [
            {
                "type": "image_url",
                "image_url": {
                    "url": request.image_data,
                    "detail": "high",
                },
            },
            {
                "type": "text",
                "text": request.prompt,
            },
        ]

        messages.append(
            {
                "role": MessageRole.USER.value,
                "content": user_content,
            }
        )

        # Send request using thread-local client for thread safety
        client = self._get_client()

        # Build API kwargs
        api_kwargs: dict[str, Any] = {
            "model": self._model,
            "messages": messages,
            "max_tokens": request.max_tokens,
            "temperature": request.temperature,
        }

        # NOTE: LM Studio does not support response_format={"type":"json_object"}.
        # It only accepts "json_schema" or "text". Since we already have robust
        # JSON extraction via _extract_json() (handles markdown blocks, embedded
        # JSON, etc.), we rely on prompt-level "respond in JSON" instructions
        # instead of API-level enforcement.

        response = client.chat.completions.create(**api_kwargs)

        return response

    def _extract_json(self, content: str) -> dict[str, Any] | None:
        """
        Extract JSON from response content.

        Handles various formats:
        - Direct JSON
        - Markdown code blocks
        - Embedded JSON in text

        Args:
            content: Raw response content.

        Returns:
            Parsed JSON dict or None if extraction fails.
        """
        if not content:
            return None

        content = content.strip()

        # Try direct JSON parse first
        try:
            if content.startswith("{"):
                return json.loads(content)
        except json.JSONDecodeError:
            pass

        # Try extraction patterns
        for pattern in self._json_patterns:
            matches = pattern.findall(content)
            for match in matches:
                try:
                    candidate = match.strip()
                    if candidate.startswith("{"):
                        return json.loads(candidate)
                except json.JSONDecodeError:
                    continue

        # Final attempt: find JSON-like structure
        try:
            start = content.find("{")
            end = content.rfind("}") + 1
            if start >= 0 and end > start:
                candidate = content[start:end]
                return json.loads(candidate)
        except json.JSONDecodeError:
            pass

        logger.debug(
            "json_extraction_failed",
            content_length=len(content),
            content_preview=content[:200] if len(content) > 200 else content,
        )

        return None

    async def _get_async_client(self) -> AsyncOpenAI:
        """Get or create the async OpenAI client (thread-safe)."""
        if self._async_client is None:
            async with self._async_client_lock:
                # Double-check after acquiring lock
                if self._async_client is None:
                    self._async_client = AsyncOpenAI(
                        base_url=self._base_url,
                        api_key="not-needed",
                        timeout=float(self._timeout),
                        max_retries=0,
                    )
        return self._async_client

    async def send_vision_request_async(
        self,
        request: VisionRequest,
    ) -> VisionResponse:
        """
        Send a vision request asynchronously using native async client.

        Args:
            request: VisionRequest to send.

        Returns:
            VisionResponse with model output.
        """
        start_time = time.perf_counter()

        # Get async client
        client = await self._get_async_client()

        # Build messages
        messages: list[dict[str, Any]] = []

        if request.system_prompt:
            messages.append(
                {
                    "role": MessageRole.SYSTEM.value,
                    "content": request.system_prompt,
                }
            )

        user_content: list[dict[str, Any]] = [
            {
                "type": "image_url",
                "image_url": {
                    "url": request.image_data,
                    "detail": "high",
                },
            },
            {
                "type": "text",
                "text": request.prompt,
            },
        ]

        messages.append(
            {
                "role": MessageRole.USER.value,
                "content": user_content,
            }
        )

        try:
            # Send async request using native async client
            response = await client.chat.completions.create(
                model=self._model,
                messages=messages,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
            )

            latency_ms = int((time.perf_counter() - start_time) * 1000)
            content = response.choices[0].message.content or ""
            parsed_json = self._extract_json(content)

            vision_response = VisionResponse(
                content=content,
                parsed_json=parsed_json,
                model=response.model,
                usage={
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else 0,
                    "completion_tokens": response.usage.completion_tokens if response.usage else 0,
                    "total_tokens": response.usage.total_tokens if response.usage else 0,
                },
                latency_ms=latency_ms,
                request_id=request.request_id,
            )

            logger.info(
                "async_vision_request_complete",
                request_id=request.request_id,
                latency_ms=latency_ms,
                tokens=vision_response.total_tokens,
                has_json=vision_response.has_json,
            )

            return vision_response

        except APIConnectionError as e:
            raise LMConnectionError(f"Async connection failed: {e}") from e
        except APITimeoutError as e:
            raise LMTimeoutError(f"Async request timed out: {e}") from e
        except Exception as e:
            raise LMClientError(f"Async request failed: {e}") from e

    async def send_batch_async(
        self,
        requests: list[VisionRequest],
        max_concurrent: int = 3,
    ) -> list[VisionResponse]:
        """
        Send multiple requests concurrently with rate limiting.

        Args:
            requests: List of VisionRequests.
            max_concurrent: Maximum concurrent requests.

        Returns:
            List of VisionResponses in same order as requests.
        """
        semaphore = asyncio.Semaphore(max_concurrent)

        async def send_limited(req: VisionRequest) -> VisionResponse:
            async with semaphore:
                return await self.send_vision_request_async(req)

        tasks = [send_limited(req) for req in requests]
        return await asyncio.gather(*tasks)

    def close(self) -> None:
        """Close client connections and release resources."""
        if self._closed:
            return

        self._closed = True

        # Close HTTP client
        try:
            self._http_client.close()
        except Exception:
            pass  # Best effort cleanup

        # Close async client if initialized
        if self._async_client is not None:
            try:
                # Try to get a running loop first (Python 3.10+ compliant)
                try:
                    loop = asyncio.get_running_loop()
                    # If we're in an async context, schedule the cleanup
                    loop.create_task(self._async_client.close())
                except RuntimeError:
                    # No running loop - try to create one for cleanup
                    try:
                        # Create new event loop for cleanup (works in sync context)
                        loop = asyncio.new_event_loop()
                        try:
                            loop.run_until_complete(self._async_client.close())
                        finally:
                            loop.close()
                    except Exception:
                        # If all else fails, let GC handle it
                        # The client will eventually be cleaned up
                        pass
            except Exception:
                pass  # Best effort cleanup

        logger.debug("lm_client_closed")

    async def close_async(self) -> None:
        """
        Async close for proper cleanup when called from async context.

        Use this method instead of close() when in an async function
        to ensure proper async resource cleanup.
        """
        if self._closed:
            return

        self._closed = True

        # Close HTTP client (sync)
        try:
            self._http_client.close()
        except Exception:
            pass

        # Close async client properly with await
        if self._async_client is not None:
            try:
                await self._async_client.close()
            except Exception:
                pass

        logger.debug("lm_client_closed_async")

    def __del__(self) -> None:
        """Destructor to ensure resources are released."""
        try:
            if not getattr(self, "_closed", True):
                self.close()
        except Exception:
            pass  # Best effort cleanup during garbage collection

    def __enter__(self) -> "LMStudioClient":
        """Context manager entry."""
        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        """Context manager exit."""
        self.close()
