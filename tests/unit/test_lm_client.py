"""
Unit tests for the LMStudioClient.

Tests cover:
- VisionRequest construction (from_page_image, from_file)
- VisionResponse properties
- LMStudioClient initialization
- Request validation
- JSON extraction from responses
- Error handling (connection, timeout, rate limit)
- Health check
"""

from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest
from openai import APIConnectionError, APITimeoutError
from tenacity import Future, RetryError

from src.client.lm_client import (
    LMClientError,
    LMConnectionError,
    LMResponseError,
    LMStudioClient,
    LMTimeoutError,
    LMValidationError,
    MessageRole,
    VisionRequest,
    VisionResponse,
)


# ---------------------------------------------------------------------------
# TestVisionRequest
# ---------------------------------------------------------------------------


class TestVisionRequest:
    """Tests for VisionRequest dataclass."""

    def test_basic_construction(self) -> None:
        req = VisionRequest(
            image_data="data:image/png;base64,abc123",
            prompt="Extract patient name",
        )
        assert req.image_data == "data:image/png;base64,abc123"
        assert req.prompt == "Extract patient name"
        assert req.max_tokens == 4096
        assert req.temperature == 0.1
        assert req.json_mode is True
        assert req.request_id.startswith("req_")

    def test_custom_params(self) -> None:
        req = VisionRequest(
            image_data="data:image/png;base64,abc",
            prompt="Extract",
            system_prompt="You are a medical extractor",
            max_tokens=2048,
            temperature=0.5,
            json_mode=False,
        )
        assert req.system_prompt == "You are a medical extractor"
        assert req.max_tokens == 2048
        assert req.temperature == 0.5
        assert req.json_mode is False

    def test_from_file_nonexistent(self, tmp_path: Path) -> None:
        with pytest.raises(LMValidationError, match="Image file not found"):
            VisionRequest.from_file(
                tmp_path / "nonexistent.png",
                prompt="Extract",
            )

    def test_from_file_valid(self, tmp_path: Path) -> None:
        img_path = tmp_path / "test.png"
        img_path.write_bytes(b"\x89PNG\r\n\x1a\n" + b"\x00" * 20)

        req = VisionRequest.from_file(img_path, prompt="Extract data")

        assert req.prompt == "Extract data"
        assert req.image_data.startswith("data:image/png;base64,")

    def test_from_file_jpeg(self, tmp_path: Path) -> None:
        img_path = tmp_path / "test.jpg"
        img_path.write_bytes(b"\xff\xd8\xff\xe0" + b"\x00" * 20)

        req = VisionRequest.from_file(img_path, prompt="Extract")

        assert "image/jpeg" in req.image_data

    def test_request_id_unique(self) -> None:
        ids = {
            VisionRequest(image_data="x", prompt="p").request_id
            for _ in range(20)
        }
        # Most should be unique (time-based, ms precision)
        assert len(ids) >= 1

    def test_immutable(self) -> None:
        req = VisionRequest(image_data="x", prompt="p")
        with pytest.raises(AttributeError):
            req.prompt = "new"  # type: ignore[misc]


# ---------------------------------------------------------------------------
# TestVisionResponse
# ---------------------------------------------------------------------------


class TestVisionResponse:
    """Tests for VisionResponse dataclass."""

    def test_basic_response(self) -> None:
        resp = VisionResponse(
            content='{"patient_name": "John"}',
            parsed_json={"patient_name": "John"},
            model="qwen3-vl",
            usage={"prompt_tokens": 100, "completion_tokens": 50, "total_tokens": 150},
            latency_ms=500,
        )
        assert resp.has_json is True
        assert resp.prompt_tokens == 100
        assert resp.completion_tokens == 50
        assert resp.total_tokens == 150

    def test_no_json_response(self) -> None:
        resp = VisionResponse(content="plain text", parsed_json=None)
        assert resp.has_json is False
        assert resp.prompt_tokens == 0

    def test_to_dict(self) -> None:
        resp = VisionResponse(
            content='{"a": 1}',
            parsed_json={"a": 1},
            model="test",
            latency_ms=42,
        )
        d = resp.to_dict()
        assert d["content"] == '{"a": 1}'
        assert d["parsed_json"] == {"a": 1}
        assert d["model"] == "test"
        assert d["latency_ms"] == 42
        assert d["has_json"] is True

    def test_empty_usage(self) -> None:
        resp = VisionResponse(content="x")
        assert resp.prompt_tokens == 0
        assert resp.completion_tokens == 0
        assert resp.total_tokens == 0


# ---------------------------------------------------------------------------
# TestMessageRole
# ---------------------------------------------------------------------------


class TestMessageRole:
    """Tests for MessageRole enum."""

    def test_values(self) -> None:
        assert MessageRole.SYSTEM.value == "system"
        assert MessageRole.USER.value == "user"
        assert MessageRole.ASSISTANT.value == "assistant"


# ---------------------------------------------------------------------------
# TestLMStudioClientInit
# ---------------------------------------------------------------------------


class TestLMStudioClientInit:
    """Tests for LMStudioClient initialization."""

    def test_default_init(self) -> None:
        client = LMStudioClient()
        assert client._model is not None
        assert client._timeout > 0
        assert client._max_retries > 0

    def test_custom_params(self) -> None:
        client = LMStudioClient(
            base_url="http://custom:5555/v1",
            model="custom-model",
            max_tokens=8192,
            temperature=0.5,
            timeout=60,
            max_retries=5,
        )
        assert client._model == "custom-model"
        assert client._max_tokens == 8192
        assert client._timeout == 60
        assert client._max_retries == 5


# ---------------------------------------------------------------------------
# TestLMStudioClientMethods
# ---------------------------------------------------------------------------


class TestLMStudioClientMethods:
    """Tests for LMStudioClient methods."""

    def test_extract_json_valid(self) -> None:
        client = LMStudioClient()
        text = '```json\n{"name": "Alice"}\n```'
        result = client._extract_json(text)
        assert result == {"name": "Alice"}

    def test_extract_json_plain(self) -> None:
        client = LMStudioClient()
        text = '{"name": "Bob"}'
        result = client._extract_json(text)
        assert result == {"name": "Bob"}

    def test_extract_json_invalid(self) -> None:
        client = LMStudioClient()
        result = client._extract_json("not json at all")
        assert result is None

    def test_is_healthy_success(self) -> None:
        client = LMStudioClient()
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        client._http_client = MagicMock()
        client._http_client.get.return_value = mock_resp

        assert client.is_healthy() is True

    def test_is_healthy_failure(self) -> None:
        client = LMStudioClient()
        client._http_client = MagicMock()
        client._http_client.get.side_effect = Exception("Connection refused")

        assert client.is_healthy() is False


# ---------------------------------------------------------------------------
# TestExceptionHierarchy
# ---------------------------------------------------------------------------


class TestExceptionHierarchy:
    """Tests for exception class hierarchy."""

    def test_connection_error_is_client_error(self) -> None:
        assert issubclass(LMConnectionError, LMClientError)

    def test_timeout_error_is_client_error(self) -> None:
        assert issubclass(LMTimeoutError, LMClientError)

    def test_response_error_is_client_error(self) -> None:
        assert issubclass(LMResponseError, LMClientError)

    def test_validation_error_is_client_error(self) -> None:
        assert issubclass(LMValidationError, LMClientError)


# ---------------------------------------------------------------------------
# TestRetryErrorMapping
# ---------------------------------------------------------------------------


def _make_retry_error(inner_exc: BaseException) -> RetryError:
    """
    Build a tenacity ``RetryError`` whose ``last_attempt`` carries ``inner_exc``.

    Used to drive ``send_vision_request``'s ``except RetryError`` branch
    deterministically without spinning the real retry loop.
    """
    future = Future(attempt_number=3)
    future.set_exception(inner_exc)
    return RetryError(future)


def _fake_httpx_request() -> httpx.Request:
    """openai's APIConnectionError/APITimeoutError require an httpx.Request."""
    return httpx.Request("POST", "http://test/v1/chat/completions")


@pytest.fixture
def vision_request() -> VisionRequest:
    """Reusable minimal VisionRequest for retry-mapping tests."""
    return VisionRequest(
        image_data="data:image/png;base64,iVBORw0KG",
        prompt="Extract patient name",
    )


class TestRetryErrorMapping:
    """
    Coverage for src/client/lm_client.py:479-493.

    The wrapper unpacks ``RetryError.last_attempt.exception()`` and maps:

    * ``APIConnectionError`` -> ``LMConnectionError``
    * ``APITimeoutError``    -> ``LMTimeoutError``
    * anything else          -> ``LMClientError``

    Strategy: patch ``LMStudioClient._send_with_retry`` to raise a
    pre-built ``RetryError`` so we exercise the mapping branch directly,
    bypassing the real tenacity loop (kept fast + deterministic).
    """

    def test_retry_error_wrapping_api_connection_error_maps_to_lm_connection_error(
        self, vision_request: VisionRequest
    ) -> None:
        client = LMStudioClient()
        inner = APIConnectionError(request=_fake_httpx_request())
        retry_err = _make_retry_error(inner)

        with patch.object(
            client, "_send_with_retry", side_effect=retry_err
        ):
            with pytest.raises(LMConnectionError) as excinfo:
                client.send_vision_request(vision_request)

        # The mapping must preserve causal context via ``raise ... from e``.
        assert excinfo.value.__cause__ is retry_err
        # And the message must mention the retry count for operator triage.
        assert str(client._max_retries) in str(excinfo.value)

    def test_retry_error_wrapping_api_timeout_error_maps_to_lm_timeout_error(
        self, vision_request: VisionRequest
    ) -> None:
        client = LMStudioClient()
        inner = APITimeoutError(request=_fake_httpx_request())
        retry_err = _make_retry_error(inner)

        with patch.object(
            client, "_send_with_retry", side_effect=retry_err
        ):
            with pytest.raises(LMTimeoutError) as excinfo:
                client.send_vision_request(vision_request)

        assert excinfo.value.__cause__ is retry_err
        # LMTimeoutError must NOT be mistaken for LMConnectionError —
        # both subclass LMClientError, so isinstance() can't distinguish on
        # the base type alone. Pin the exact type.
        assert type(excinfo.value) is LMTimeoutError

    def test_retry_error_wrapping_generic_exception_maps_to_lm_client_error(
        self, vision_request: VisionRequest
    ) -> None:
        client = LMStudioClient()
        inner = RuntimeError("upstream model crashed")
        retry_err = _make_retry_error(inner)

        with patch.object(
            client, "_send_with_retry", side_effect=retry_err
        ):
            with pytest.raises(LMClientError) as excinfo:
                client.send_vision_request(vision_request)

        # Must be the BASE LMClientError, not one of its specialised subclasses.
        assert type(excinfo.value) is LMClientError
        assert excinfo.value.__cause__ is retry_err


# ---------------------------------------------------------------------------
# TestSendVisionRequestSanity
# ---------------------------------------------------------------------------


class TestSendVisionRequestSanity:
    """Round-out coverage: happy path + non-RetryError passthrough."""

    def test_happy_path_returns_vision_response(
        self, vision_request: VisionRequest
    ) -> None:
        """A successful call wraps the OpenAI response into a VisionResponse."""
        client = LMStudioClient()

        # Build a fake OpenAI ChatCompletion response object.
        fake_response = MagicMock()
        fake_response.choices = [MagicMock()]
        fake_response.choices[0].message.content = '{"patient_name": "Alice"}'
        fake_response.model = "qwen3-vl"
        fake_response.usage.prompt_tokens = 120
        fake_response.usage.completion_tokens = 30
        fake_response.usage.total_tokens = 150

        with patch.object(client, "_send_with_retry", return_value=fake_response):
            response = client.send_vision_request(vision_request)

        assert isinstance(response, VisionResponse)
        assert response.has_json is True
        assert response.parsed_json == {"patient_name": "Alice"}
        assert response.model == "qwen3-vl"
        assert response.prompt_tokens == 120
        assert response.completion_tokens == 30
        assert response.total_tokens == 150
        assert response.request_id == vision_request.request_id

    def test_non_retry_error_passes_through_unchanged(
        self, vision_request: VisionRequest
    ) -> None:
        """
        Exceptions that aren't ``RetryError`` must NOT be swallowed or
        remapped by the retry-mapping ``except`` block. A malformed-schema
        ``ValueError`` (raised eagerly before retry, e.g. during request
        construction) should bubble up as-is so the caller sees the real
        cause.
        """
        client = LMStudioClient()

        with patch.object(
            client, "_send_with_retry", side_effect=ValueError("bad schema")
        ):
            with pytest.raises(ValueError, match="bad schema"):
                client.send_vision_request(vision_request)

    def test_retry_error_with_none_last_exception_falls_through_to_generic(
        self, vision_request: VisionRequest
    ) -> None:
        """
        Defensive: if ``last_attempt.exception()`` returns ``None``
        (e.g. a cancelled future), the mapping should fall through to
        the generic ``LMClientError`` branch rather than raise an
        ``AttributeError`` from ``isinstance(None, ...)``.

        ``isinstance(None, X)`` is False for any X, so both specific
        branches are skipped and the generic ``raise LMClientError``
        executes. This guards against future refactors that might
        replace ``isinstance`` checks with attribute access.
        """
        client = LMStudioClient()
        future = Future(attempt_number=3)
        future.set_result(None)  # No exception -> .exception() returns None
        retry_err = RetryError(future)

        with patch.object(client, "_send_with_retry", side_effect=retry_err):
            with pytest.raises(LMClientError) as excinfo:
                client.send_vision_request(vision_request)

        assert type(excinfo.value) is LMClientError
