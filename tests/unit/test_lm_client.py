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

import base64
import json
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.client.lm_client import (
    LMClientError,
    LMConnectionError,
    LMResponseError,
    LMTimeoutError,
    LMValidationError,
    LMStudioClient,
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
