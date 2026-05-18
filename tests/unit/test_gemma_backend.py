"""Tests for the Phase K Gemma 4 backend adapter.

Mirrors the LMStudioBackend test pattern: patch ``LMStudioClient`` so
the adapter constructs MagicMocks and the tests don't need a running
LM Studio instance. The Gemma 4 model is never actually loaded.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.client.backends.gemma_backend import GemmaBackend
from src.client.backends.protocol import BackendCapabilities, VLMRole
from src.client.lm_client import VisionRequest, VisionResponse


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_client_factory():
    """Patch ``LMStudioClient`` so the adapter constructs MagicMocks."""
    with patch("src.client.backends.gemma_backend.LMStudioClient") as mock_cls:
        instances: list[MagicMock] = []

        def _factory(*args, **kwargs):
            inst = MagicMock()
            inst.send_vision_request.return_value = VisionResponse(
                content="ok",
                parsed_json={"ok": True},
                model=kwargs.get("model", "gemma-4-26b-a4b-it"),
                usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
                latency_ms=5,
                request_id="r",
            )
            inst.is_healthy.return_value = True
            inst._init_kwargs = kwargs
            instances.append(inst)
            return inst

        mock_cls.side_effect = _factory
        yield mock_cls, instances


@pytest.fixture
def gemma_backend(fake_client_factory):
    """Standard backend with the five medical-code tools registered."""
    return GemmaBackend(
        primary_url="http://localhost:1235/v1",
        primary_model="gemma-4-26b-a4b-it",
        register_rcm_tools=True,
    )


@pytest.fixture
def vision_request():
    return VisionRequest(
        prompt="Extract the patient name from this CMS-1500.",
        image_data="data:image/png;base64,iVBORw0KGgo=",
        max_tokens=512,
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestBackendIdentity:
    """The backend reports its own name + plausible capabilities."""

    def test_name_is_gemma(self) -> None:
        # Don't even need an instance — class-level attribute.
        assert GemmaBackend.name == "gemma"

    def test_capabilities_shape(self, gemma_backend) -> None:
        caps = gemma_backend.capabilities()
        assert isinstance(caps, BackendCapabilities)
        assert caps.name == "gemma"
        assert caps.supports_constrained_decoding is True
        assert caps.supports_multi_image is True
        # Phase K — single-instance by design.
        assert caps.supports_dual_vlm is False
        assert caps.supports_tensor_parallelism is False
        # Notes include the role-collapse explanation.
        assert any("share one endpoint" in n for n in caps.notes)

    def test_capabilities_note_when_tools_registered(self, gemma_backend) -> None:
        notes_text = " ".join(gemma_backend.capabilities().notes)
        assert "tools attached" in notes_text


class TestRoleResolution:
    """All roles collapse to the same endpoint by design."""

    def test_primary_resolves(self, gemma_backend) -> None:
        url, model = gemma_backend.resolve(VLMRole.PRIMARY)
        assert url == "http://localhost:1235/v1"
        assert model == "gemma-4-26b-a4b-it"

    def test_secondary_collapses_to_primary(self, gemma_backend) -> None:
        assert gemma_backend.resolve(VLMRole.SECONDARY) == (
            "http://localhost:1235/v1",
            "gemma-4-26b-a4b-it",
        )

    def test_critic_collapses_to_primary(self, gemma_backend) -> None:
        assert gemma_backend.resolve(VLMRole.CRITIC) == (
            "http://localhost:1235/v1",
            "gemma-4-26b-a4b-it",
        )

    def test_lite_resolves(self, gemma_backend) -> None:
        assert gemma_backend.resolve(VLMRole.LITE) == (
            "http://localhost:1235/v1",
            "gemma-4-26b-a4b-it",
        )


class TestLazyClient:
    """No ``LMStudioClient`` is created until the first request."""

    def test_init_creates_no_clients_eagerly(self, fake_client_factory):
        mock_cls, instances = fake_client_factory
        GemmaBackend(
            primary_url="http://localhost:1235/v1",
            primary_model="gemma-4-26b-a4b-it",
        )
        assert mock_cls.call_count == 0
        assert instances == []

    def test_first_request_creates_one_client(
        self, fake_client_factory, gemma_backend, vision_request
    ):
        mock_cls, instances = fake_client_factory
        gemma_backend.send_vision_request(vision_request, role=VLMRole.PRIMARY)
        assert mock_cls.call_count == 1
        assert len(instances) == 1

    def test_close_releases_all_clients(
        self, fake_client_factory, gemma_backend, vision_request
    ):
        _, instances = fake_client_factory
        gemma_backend.send_vision_request(vision_request, role=VLMRole.PRIMARY)
        gemma_backend.send_vision_request(vision_request, role=VLMRole.SECONDARY)
        gemma_backend.close()
        for inst in instances:
            inst.close.assert_called_once()


class TestSchemaBoundDecode:
    """Schema-bound calls use LM Studio's ``response_format=json_schema``."""

    def test_no_schema_no_response_format(
        self, fake_client_factory, gemma_backend, vision_request
    ):
        _, instances = fake_client_factory
        gemma_backend.send_vision_request(vision_request)
        call = instances[0].send_vision_request.call_args
        assert call.kwargs.get("response_format") is None

    def test_with_schema_passes_json_schema(
        self, fake_client_factory, gemma_backend, vision_request
    ):
        _, instances = fake_client_factory
        schema = {
            "type": "object",
            "properties": {"patient_name": {"type": "string"}},
            "required": ["patient_name"],
        }
        gemma_backend.send_vision_request(vision_request, schema=schema)
        rf = instances[0].send_vision_request.call_args.kwargs["response_format"]
        assert rf["type"] == "json_schema"
        assert rf["json_schema"]["name"] == "veridoc"
        assert rf["json_schema"]["schema"] == schema


class TestCallWithTools:
    """``call_with_tools`` forwards the function-calling registry to LM Studio."""

    def test_default_attaches_rcm_tools(
        self, fake_client_factory, gemma_backend, vision_request
    ):
        _, instances = fake_client_factory
        gemma_backend.call_with_tools(vision_request)
        extra_body = instances[0].send_vision_request.call_args.kwargs.get(
            "extra_body"
        )
        assert extra_body is not None
        tool_names = {t["function"]["name"] for t in extra_body["tools"]}
        assert {
            "npi_luhn_check",
            "cpt_validate",
            "icd_normalize",
            "sum_reconcile",
            "validate_date_ordering",
        } <= tool_names
        assert extra_body["tool_choice"] == "auto"

    def test_explicit_tool_choice_forced(
        self, fake_client_factory, gemma_backend, vision_request
    ):
        _, instances = fake_client_factory
        gemma_backend.call_with_tools(
            vision_request,
            tool_choice={"type": "function", "function": {"name": "cpt_validate"}},
        )
        extra_body = instances[0].send_vision_request.call_args.kwargs["extra_body"]
        assert extra_body["tool_choice"]["function"]["name"] == "cpt_validate"

    def test_disabled_tools_no_extra_body(
        self, fake_client_factory, vision_request
    ):
        _, instances = fake_client_factory
        backend = GemmaBackend(
            primary_url="http://localhost:1235/v1",
            primary_model="gemma-4-26b-a4b-it",
            register_rcm_tools=False,
        )
        backend.call_with_tools(vision_request)
        kwargs = instances[0].send_vision_request.call_args.kwargs
        # No tools => normal request, no extra_body injected.
        assert kwargs.get("extra_body") is None


class TestParseToolCalls:
    """The parser handles both structured and fallback regex paths."""

    def test_structured_tool_calls_parsed(self, gemma_backend) -> None:
        response = VisionResponse(
            content="",
            parsed_json=None,
            model="gemma-4-26b-a4b-it",
            usage={
                "prompt_tokens": 5,
                "completion_tokens": 5,
                "total_tokens": 10,
                "tool_calls": [
                    {
                        "id": "call_abc",
                        "function": {
                            "name": "cpt_validate",
                            "arguments": json.dumps({"cpt_code": "99213"}),
                        },
                    }
                ],
            },
            latency_ms=5,
            request_id="r",
        )
        calls = gemma_backend.parse_tool_calls(response)
        assert len(calls) == 1
        assert calls[0]["name"] == "cpt_validate"
        assert calls[0]["arguments"] == {"cpt_code": "99213"}
        assert calls[0]["call_id"] == "call_abc"

    def test_fallback_regex_path(self, gemma_backend) -> None:
        # Gemma 4 chat-template fallback: tool call embedded in content.
        content = (
            "Some narration before.\n"
            '<tool_call>{"name": "icd_normalize", "arguments": {"raw_code": "J069"}}</tool_call>\n'
            "More text."
        )
        response = VisionResponse(
            content=content,
            parsed_json=None,
            model="gemma-4-26b-a4b-it",
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            latency_ms=5,
            request_id="r",
        )
        calls = gemma_backend.parse_tool_calls(response)
        assert len(calls) == 1
        assert calls[0]["name"] == "icd_normalize"
        assert calls[0]["arguments"] == {"raw_code": "J069"}

    def test_no_tool_calls_returns_empty(self, gemma_backend) -> None:
        response = VisionResponse(
            content="just text",
            parsed_json=None,
            model="gemma-4-26b-a4b-it",
            usage={"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
            latency_ms=5,
            request_id="r",
        )
        assert gemma_backend.parse_tool_calls(response) == []

    def test_dispatch_calls_runs_through_registry(self, gemma_backend) -> None:
        calls = [
            {"name": "cpt_validate", "arguments": {"cpt_code": "99213"}, "call_id": "x"}
        ]
        results = gemma_backend.dispatch_tool_calls(calls)
        assert len(results) == 1
        assert results[0]["name"] == "cpt_validate"
        assert results[0]["result"]["valid"] is True
        assert results[0]["call_id"] == "x"


class TestProvenanceStageProjection:
    """Phase K-5c — tool-call results project into Provenance stages."""

    def test_single_call_produces_one_stage(self, gemma_backend) -> None:
        dispatched = [
            {
                "name": "cpt_validate",
                "arguments": {"cpt_code": "99213"},
                "result": {"valid": True, "category": "I"},
                "call_id": "call_abc",
            }
        ]
        stages = gemma_backend.tool_calls_to_provenance_stages(dispatched)
        assert len(stages) == 1
        assert stages[0]["stage"] == "gemma_tool_call"
        assert stages[0]["agent"] == "gemma:cpt_validate"
        assert stages[0]["metadata"]["tool_name"] == "cpt_validate"
        assert stages[0]["metadata"]["result"]["valid"] is True
        assert stages[0]["metadata"]["call_id"] == "call_abc"

    def test_empty_input_returns_empty_list(self, gemma_backend) -> None:
        assert gemma_backend.tool_calls_to_provenance_stages([]) == []

    def test_missing_name_falls_back(self, gemma_backend) -> None:
        stages = gemma_backend.tool_calls_to_provenance_stages([{"arguments": {}}])
        assert stages[0]["agent"] == "gemma:unknown_tool"

    def test_end_to_end_through_parse_dispatch_project(
        self, gemma_backend
    ) -> None:
        """Round-trip: parse → dispatch → project, all under one backend."""
        import json

        response = VisionResponse(
            content="",
            parsed_json=None,
            model="gemma-4-26b-a4b-it",
            usage={
                "prompt_tokens": 1,
                "completion_tokens": 1,
                "total_tokens": 2,
                "tool_calls": [
                    {
                        "function": {
                            "name": "icd_normalize",
                            "arguments": json.dumps({"raw_code": "J069"}),
                        }
                    }
                ],
            },
            latency_ms=5,
            request_id="r",
        )
        parsed = gemma_backend.parse_tool_calls(response)
        dispatched = gemma_backend.dispatch_tool_calls(parsed)
        stages = gemma_backend.tool_calls_to_provenance_stages(dispatched)

        assert len(stages) == 1
        assert stages[0]["stage"] == "gemma_tool_call"
        assert stages[0]["agent"] == "gemma:icd_normalize"
        assert stages[0]["metadata"]["result"]["normalised"] == "J06.9"


class TestHealth:
    """``health()`` probes the configured endpoint and reflects connectivity."""

    def test_healthy_endpoint(self, fake_client_factory, gemma_backend):
        result = gemma_backend.health()
        assert result.overall_healthy is True
        assert result.backend_name == "gemma"
        assert VLMRole.PRIMARY in result.roles
        assert result.roles[VLMRole.PRIMARY]["healthy"] is True

    def test_unhealthy_endpoint_default_fails_closed(self, fake_client_factory):
        _, instances = fake_client_factory
        backend = GemmaBackend(
            primary_url="http://localhost:1235/v1",
            primary_model="gemma-4-26b-a4b-it",
        )
        # Trigger lazy creation of the client, then mark it unhealthy.
        result_first = backend.health()
        instances[0].is_healthy.return_value = False
        # New probe must report unhealthy.
        result = backend.health()
        assert result_first.overall_healthy is True  # sanity
        assert result.overall_healthy is False

    def test_fail_open_for_demo_environments(self, fake_client_factory):
        _, instances = fake_client_factory
        backend = GemmaBackend(
            primary_url="http://localhost:1235/v1",
            primary_model="gemma-4-26b-a4b-it",
            fail_open_on_health=True,
        )
        # Force unhealthy on first probe.
        backend.health()
        instances[0].is_healthy.return_value = False
        result = backend.health()
        assert result.overall_healthy is True  # fail-open
        assert result.roles[VLMRole.PRIMARY]["healthy"] is False


class TestFactoryWiring:
    """``get_backend(settings)`` resolves a GemmaBackend when configured."""

    def test_factory_dispatches_to_gemma(self, fake_client_factory):
        from src.client.backends.factory import get_backend, reset_cache
        from src.config.settings import get_settings

        reset_cache()
        settings = get_settings()
        # Force gemma + ensure defaults are populated (they should be).
        original_backend = settings.vlm.backend
        try:
            settings.vlm.backend = "gemma"
            backend = get_backend(settings)
            assert backend.name == "gemma"
            assert isinstance(backend, GemmaBackend)
        finally:
            settings.vlm.backend = original_backend
            reset_cache()


class TestProtocolConformance:
    """``GemmaBackend`` satisfies the runtime-checkable ``VLMBackend`` Protocol."""

    def test_isinstance_vlm_backend(self, gemma_backend):
        from src.client.backends.protocol import VLMBackend

        assert isinstance(gemma_backend, VLMBackend)
