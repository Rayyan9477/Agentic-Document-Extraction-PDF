"""
Phase 1 integration — constrained decoding wired into the agents.

Verifies that every agent that issues structured-output VLM calls now
binds a Pydantic schema at decode time. The contract under test:

* When the schema-aware backend reports
  ``supports_constrained_decoding=True``, the response carries a
  ``schema_enforced=True`` trace.
* When the underlying client returns a malformed-JSON body even
  though the schema was requested, ``ConstrainedDecodingError``
  surfaces as a typed ``AgentError`` — never silently as ``None``.
* The legacy ``send_vision_request_with_json`` path still works for
  any caller that has not yet adopted Pydantic schemas (none in
  ``src/agents/`` today, but the method is preserved for downstream
  consumers).

These tests stub ``get_backend()`` so no live VLM is required.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock, patch

import pytest

from src.agents._constrained_envelopes import JSONObjectEnvelope
from src.agents.base import AgentError, BaseAgent
from src.client.backends.factory import reset_cache
from src.client.backends.protocol import BackendCapabilities, VLMRole
from src.client.constrained import ConstrainedDecodingError, DecodingTrace
from src.client.lm_client import LMStudioClient, VisionResponse


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear_backend_cache():
    reset_cache()
    yield
    reset_cache()


def _stub_backend(
    parsed_json: dict[str, Any] | None = None,
    *,
    supports_constrained: bool = True,
) -> MagicMock:
    """Construct a ``VLMBackend``-shaped mock with controllable behaviour."""
    backend = MagicMock()
    backend.name = "stub"
    backend.capabilities.return_value = BackendCapabilities(
        name="stub",
        supports_dual_vlm=False,
        supports_constrained_decoding=supports_constrained,
        supports_logprobs=False,
        supports_multi_image=True,
        supports_tensor_parallelism=False,
    )
    backend.resolve.return_value = ("http://stub/v1", "stub-model")
    backend.send_vision_request.return_value = VisionResponse(
        content="" if parsed_json is None else "{}",
        parsed_json=parsed_json,
        model="stub-model",
        usage={"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
        latency_ms=3,
        request_id="r",
    )
    return backend


class _StubAgent(BaseAgent):
    """Minimal concrete agent for exercising the schema-bound path."""

    def __init__(self, client: Any) -> None:
        super().__init__(name="stub", client=client)

    def process(self, state):  # type: ignore[no-untyped-def]
        return state


def _make_stub_client(parsed_json: dict[str, Any] | None) -> MagicMock:
    """Build a mock LMStudioClient that returns a controllable VisionResponse."""
    client = MagicMock(spec_set=LMStudioClient)
    client.send_vision_request.return_value = VisionResponse(
        content="" if parsed_json is None else "{}",
        parsed_json=parsed_json,
        model="stub-model",
        usage={"prompt_tokens": 5, "completion_tokens": 5, "total_tokens": 10},
        latency_ms=3,
        request_id="r",
    )
    return client


# ---------------------------------------------------------------------------
# BaseAgent.send_vision_request_with_schema
# ---------------------------------------------------------------------------


class TestBaseAgentSchemaPath:
    def test_happy_path_returns_payload_and_trace(self) -> None:
        client = _make_stub_client({"x": 1})
        agent = _StubAgent(client)

        payload, trace = agent.send_vision_request_with_schema(
            image_data="data:image/png;base64,AAAA",
            prompt="extract",
            schema=JSONObjectEnvelope,
        )

        assert payload == {"x": 1}
        assert trace.schema_enforced is True
        assert trace.schema_name == "JSONObjectEnvelope"
        # The schema is forwarded to ``self._client.send_vision_request``
        # as ``response_format``.
        kwargs = client.send_vision_request.call_args.kwargs
        rf = kwargs["response_format"]
        assert rf["type"] == "json_schema"
        assert rf["json_schema"]["schema"] == JSONObjectEnvelope.model_json_schema()

    def test_non_json_response_raises_agent_error(self) -> None:
        client = _make_stub_client(parsed_json=None)
        agent = _StubAgent(client)
        with pytest.raises(AgentError, match="returned non-JSON"):
            agent.send_vision_request_with_schema(
                image_data="data:image/png;base64,AAAA",
                prompt="extract",
                schema=JSONObjectEnvelope,
            )

    def test_lm_client_error_wraps_to_agent_error(self) -> None:
        from src.client.lm_client import LMClientError

        client = MagicMock(spec_set=LMStudioClient)
        client.send_vision_request.side_effect = LMClientError("boom")
        agent = _StubAgent(client)
        with pytest.raises(AgentError, match="VLM request failed"):
            agent.send_vision_request_with_schema(
                image_data="x",
                prompt="p",
                schema=JSONObjectEnvelope,
            )

    def test_vlm_calls_counter_increments(self) -> None:
        client = _make_stub_client({"x": 1})
        agent = _StubAgent(client)
        n_before = agent.vlm_calls
        agent.send_vision_request_with_schema(
            image_data="x",
            prompt="p",
            schema=JSONObjectEnvelope,
        )
        agent.send_vision_request_with_schema(
            image_data="x",
            prompt="p",
            schema=JSONObjectEnvelope,
        )
        assert agent.vlm_calls - n_before == 2


# ---------------------------------------------------------------------------
# Backend wiring: response_format / extra_body
# ---------------------------------------------------------------------------


class TestBackendWiring:
    def test_lm_studio_passes_response_format_when_schema_supplied(self) -> None:
        from src.client.backends.lm_studio_backend import LMStudioBackend
        from src.client.lm_client import VisionRequest

        with patch("src.client.backends.lm_studio_backend.LMStudioClient") as mock_cls:
            inst = MagicMock()
            inst.send_vision_request.return_value = MagicMock()
            mock_cls.return_value = inst
            b = LMStudioBackend(
                primary_url="http://primary/v1",
                primary_model="primary-model",
            )
            schema = JSONObjectEnvelope.model_json_schema()
            b.send_vision_request(
                VisionRequest(image_data="x", prompt="p"),
                schema=schema,
            )
            kwargs = inst.send_vision_request.call_args.kwargs
            rf = kwargs["response_format"]
            assert rf["type"] == "json_schema"
            assert rf["json_schema"]["schema"] == schema

    def test_vllm_passes_extra_body_when_schema_supplied(self) -> None:
        from src.client.backends.vllm_backend import VLLMBackend
        from src.client.lm_client import VisionRequest

        with patch("src.client.backends.vllm_backend.LMStudioClient") as mock_cls:
            inst = MagicMock()
            inst.send_vision_request.return_value = MagicMock()
            mock_cls.return_value = inst
            b = VLLMBackend(
                primary_url="http://primary/v1",
                primary_model="primary-model",
                guided_decoding_backend="xgrammar",
            )
            schema = JSONObjectEnvelope.model_json_schema()
            b.send_vision_request(
                VisionRequest(image_data="x", prompt="p"),
                schema=schema,
            )
            kwargs = inst.send_vision_request.call_args.kwargs
            extra = kwargs["extra_body"]
            assert extra["guided_json"] == schema
            assert extra["guided_decoding_backend"] == "xgrammar"

    def test_no_schema_means_no_response_format(self) -> None:
        from src.client.backends.lm_studio_backend import LMStudioBackend
        from src.client.lm_client import VisionRequest

        with patch("src.client.backends.lm_studio_backend.LMStudioClient") as mock_cls:
            inst = MagicMock()
            inst.send_vision_request.return_value = MagicMock()
            mock_cls.return_value = inst
            b = LMStudioBackend(
                primary_url="http://primary/v1",
                primary_model="primary-model",
            )
            b.send_vision_request(
                VisionRequest(image_data="x", prompt="p"),
                schema=None,
            )
            kwargs = inst.send_vision_request.call_args.kwargs
            assert kwargs["response_format"] is None


# ---------------------------------------------------------------------------
# LMStudioClient.send_vision_request — kwargs forwarding
# ---------------------------------------------------------------------------


class TestLMStudioClientForwardsKwargs:
    def test_response_format_reaches_openai_call(self) -> None:
        """The actual OpenAI ``chat.completions.create`` should receive the
        ``response_format`` kwarg when the caller supplies one. We patch the
        thread-local OpenAI client to a MagicMock and verify."""
        client = LMStudioClient(base_url="http://x/v1", model="m")
        # Pre-seed a fake thread-local OpenAI client so _get_client returns it.
        fake_oa = MagicMock()
        fake_completion = MagicMock()
        fake_completion.choices = [MagicMock()]
        fake_completion.choices[0].message.content = '{"ok": true}'
        fake_completion.usage = MagicMock(
            prompt_tokens=1, completion_tokens=1, total_tokens=2
        )
        fake_completion.model = "m"
        fake_oa.chat.completions.create.return_value = fake_completion
        client._thread_local.client = fake_oa  # type: ignore[attr-defined]

        from src.client.lm_client import VisionRequest

        rf = {
            "type": "json_schema",
            "json_schema": {"name": "veridoc", "schema": {"type": "object"}},
        }
        eb = {"guided_json": {"type": "object"}, "guided_decoding_backend": "xgrammar"}
        client.send_vision_request(
            VisionRequest(image_data="x", prompt="p"),
            response_format=rf,
            extra_body=eb,
        )
        kwargs = fake_oa.chat.completions.create.call_args.kwargs
        assert kwargs["response_format"] == rf
        assert kwargs["extra_body"] == eb

    def test_no_kwargs_omits_them(self) -> None:
        client = LMStudioClient(base_url="http://x/v1", model="m")
        fake_oa = MagicMock()
        fake_completion = MagicMock()
        fake_completion.choices = [MagicMock()]
        fake_completion.choices[0].message.content = "{}"
        fake_completion.usage = MagicMock(
            prompt_tokens=0, completion_tokens=0, total_tokens=0
        )
        fake_completion.model = "m"
        fake_oa.chat.completions.create.return_value = fake_completion
        client._thread_local.client = fake_oa  # type: ignore[attr-defined]

        from src.client.lm_client import VisionRequest

        client.send_vision_request(VisionRequest(image_data="x", prompt="p"))
        kwargs = fake_oa.chat.completions.create.call_args.kwargs
        assert "response_format" not in kwargs
        assert "extra_body" not in kwargs


# ---------------------------------------------------------------------------
# Forced-malformation guarantee
# ---------------------------------------------------------------------------


class TestForcedMalformationGuard:
    """When constrained decoding is on, malformed responses CANNOT silently
    leak through. They must surface as ConstrainedDecodingError."""

    def test_schema_with_malformed_response_raises(self) -> None:
        backend = _stub_backend(parsed_json=None)
        # Override content so the legacy parse-attempt also fails.
        backend.send_vision_request.return_value = VisionResponse(
            content="not valid json at all",
            parsed_json=None,
            model="m",
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            latency_ms=1,
            request_id="r",
        )

        from src.client.constrained import constrained_decode
        from src.client.lm_client import VisionRequest

        with pytest.raises(ConstrainedDecodingError, match="non-JSON for schema"):
            constrained_decode(
                backend,
                VisionRequest(image_data="x", prompt="p"),
                schema=JSONObjectEnvelope,
            )

    def test_no_schema_returns_no_json_response_quietly(self) -> None:
        """Legacy contract: ``schema=None`` allows ``parsed_json=None``."""
        backend = _stub_backend(parsed_json=None)

        from src.client.constrained import constrained_decode
        from src.client.lm_client import VisionRequest

        response, trace = constrained_decode(
            backend,
            VisionRequest(image_data="x", prompt="p"),
            schema=None,
        )
        assert response.has_json is False
        assert trace.schema_enforced is False
