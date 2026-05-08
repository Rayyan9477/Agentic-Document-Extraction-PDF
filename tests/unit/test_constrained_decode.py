"""
Phase 0 tests — constrained decoding wrapper.

The wrapper is a no-op pass-through in Phase 0; we verify:

* Schema is converted to JSON Schema and the generated name surfaces
  in ``DecodingTrace``.
* Backend is called with role + schema kwargs.
* Backend response is returned unchanged when JSON parses.
* When a schema is requested but JSON extraction fails, the wrapper
  raises ``ConstrainedDecodingError``.
* When NO schema is requested, parse failures are passed through
  (legacy contract).
* Async variant has the same semantics.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import BaseModel

from src.client.backends.protocol import BackendCapabilities, VLMRole
from src.client.constrained import (
    ConstrainedDecodingError,
    DecodingTrace,
    constrained_decode,
    constrained_decode_async,
)
from src.client.lm_client import VisionRequest, VisionResponse


# ---------------------------------------------------------------------------
# Fakes
# ---------------------------------------------------------------------------


class _DummySchema(BaseModel):
    """Trivial schema so we exercise the JSON Schema conversion."""

    name: str
    age: int


def _make_request() -> VisionRequest:
    return VisionRequest(
        image_data="data:image/png;base64,AAAA",
        prompt="extract",
    )


def _make_response(parsed_json: dict[str, Any] | None = None) -> VisionResponse:
    return VisionResponse(
        content="" if parsed_json is None else "{}",
        parsed_json=parsed_json,
        model="dummy",
        usage={"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
        latency_ms=12,
        request_id="req_test",
    )


def _make_backend(response: VisionResponse, async_response: VisionResponse | None = None) -> MagicMock:
    """Construct a mocked ``VLMBackend``-shaped object."""
    backend = MagicMock()
    backend.name = "test_backend"
    backend.capabilities.return_value = BackendCapabilities(
        name="test_backend",
        supports_dual_vlm=False,
        supports_constrained_decoding=True,
        supports_logprobs=False,
        supports_multi_image=True,
        supports_tensor_parallelism=False,
    )
    backend.resolve.return_value = ("http://test/v1", "test-model")
    backend.send_vision_request.return_value = response
    backend.send_vision_request_async = AsyncMock(
        return_value=async_response or response
    )
    return backend


# ---------------------------------------------------------------------------
# Sync
# ---------------------------------------------------------------------------


class TestConstrainedDecodeSync:
    def test_passes_role_and_schema_dict_to_backend(self) -> None:
        response = _make_response({"name": "Alice", "age": 30})
        backend = _make_backend(response)

        _, trace = constrained_decode(
            backend,
            _make_request(),
            role=VLMRole.SECONDARY,
            schema=_DummySchema,
        )

        backend.send_vision_request.assert_called_once()
        kwargs = backend.send_vision_request.call_args.kwargs
        assert kwargs["role"] is VLMRole.SECONDARY
        # Schema arrives as a JSON-Schema dict, not the Pydantic class.
        assert isinstance(kwargs["schema"], dict)
        assert "name" in kwargs["schema"].get("properties", {})

        assert isinstance(trace, DecodingTrace)
        assert trace.role is VLMRole.SECONDARY
        assert trace.schema_name == "_DummySchema"
        assert trace.backend_name == "test_backend"
        assert trace.model_id == "test-model"
        assert trace.tokens_in == 7
        assert trace.tokens_out == 3
        # Phase 1: schema enforced when backend reports support AND
        # schema is supplied. ``_make_backend`` reports
        # ``supports_constrained_decoding=True`` so this should be True.
        assert trace.schema_enforced is True

    def test_returns_response_unchanged_when_json_parses(self) -> None:
        response = _make_response({"name": "Bob", "age": 7})
        backend = _make_backend(response)

        result_response, _ = constrained_decode(
            backend,
            _make_request(),
            schema=_DummySchema,
        )
        assert result_response is response
        assert result_response.parsed_json == {"name": "Bob", "age": 7}

    def test_raises_when_schema_requested_but_no_json(self) -> None:
        response = _make_response(parsed_json=None)
        backend = _make_backend(response)

        with pytest.raises(ConstrainedDecodingError, match="non-JSON for schema"):
            constrained_decode(
                backend,
                _make_request(),
                schema=_DummySchema,
            )

    def test_no_schema_no_raise_on_missing_json(self) -> None:
        """Legacy contract: without a schema, parse failure returns ``has_json=False``."""
        response = _make_response(parsed_json=None)
        backend = _make_backend(response)

        result, trace = constrained_decode(
            backend,
            _make_request(),
            # schema=None
        )
        assert result.has_json is False
        assert trace.schema_name is None

    def test_default_role_is_primary(self) -> None:
        response = _make_response({"name": "x", "age": 1})
        backend = _make_backend(response)
        constrained_decode(backend, _make_request(), schema=_DummySchema)
        assert backend.send_vision_request.call_args.kwargs["role"] is VLMRole.PRIMARY


# ---------------------------------------------------------------------------
# Async
# ---------------------------------------------------------------------------


class TestConstrainedDecodeAsync:
    @pytest.mark.asyncio
    async def test_async_happy_path(self) -> None:
        response = _make_response({"name": "y", "age": 9})
        backend = _make_backend(response, async_response=response)

        result, trace = await constrained_decode_async(
            backend,
            _make_request(),
            role=VLMRole.CRITIC,
            schema=_DummySchema,
        )
        assert result.parsed_json == {"name": "y", "age": 9}
        assert trace.role is VLMRole.CRITIC
        assert trace.schema_name == "_DummySchema"

    @pytest.mark.asyncio
    async def test_async_raises_on_missing_json_with_schema(self) -> None:
        response = _make_response(parsed_json=None)
        backend = _make_backend(response, async_response=response)

        with pytest.raises(ConstrainedDecodingError):
            await constrained_decode_async(
                backend,
                _make_request(),
                schema=_DummySchema,
            )

    @pytest.mark.asyncio
    async def test_async_no_schema_returns_unchanged(self) -> None:
        response = _make_response(parsed_json=None)
        backend = _make_backend(response, async_response=response)

        result, trace = await constrained_decode_async(
            backend,
            _make_request(),
        )
        assert result.has_json is False
        assert trace.schema_name is None
