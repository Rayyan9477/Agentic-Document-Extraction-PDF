"""
Phase 0 tests — LMStudioBackend adapter.

The adapter wraps the existing ``LMStudioClient`` and adds role
resolution. We mock the client to keep tests fast and not require a
running LM Studio.

Coverage:

* Per-role lazy client creation: each role gets its own client bound
  to the role's resolved (URL, model) pair.
* ``send_vision_request`` forwards the resolved model to the underlying
  client's ``model=`` kwarg.
* ``health()`` probes only configured roles and reports their state.
* ``close()`` closes every per-role client.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from src.client.backends.lm_studio_backend import LMStudioBackend, LMStudioDualMode
from src.client.backends.protocol import VLMRole
from src.client.lm_client import VisionRequest, VisionResponse


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def fake_client_factory():
    """Patch ``LMStudioClient`` so the adapter constructs MagicMocks."""
    with patch("src.client.backends.lm_studio_backend.LMStudioClient") as mock_cls:
        instances: list[MagicMock] = []

        def _factory(*args, **kwargs):
            # Use plain MagicMock (no spec_set) so we can stash _init_kwargs
            # for assertions. We control which attributes the test exercises
            # explicitly below.
            inst = MagicMock()
            inst.send_vision_request.return_value = VisionResponse(
                content="ok",
                parsed_json={"ok": True},
                model=kwargs.get("model", "unknown"),
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


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestLMStudioBackendResolution:
    def test_primary_resolves_to_primary(self) -> None:
        b = LMStudioBackend(
            primary_url="http://primary/v1",
            primary_model="primary-model",
        )
        assert b.resolve(VLMRole.PRIMARY) == ("http://primary/v1", "primary-model")
        assert b.resolve(VLMRole.LITE) == ("http://primary/v1", "primary-model")

    def test_secondary_in_single_only_collapses_to_primary(self) -> None:
        b = LMStudioBackend(
            primary_url="http://primary/v1",
            primary_model="primary-model",
            dual_mode=LMStudioDualMode.SINGLE_ONLY,
        )
        assert b.resolve(VLMRole.SECONDARY) == ("http://primary/v1", "primary-model")
        assert b.resolve(VLMRole.CRITIC) == ("http://primary/v1", "primary-model")

    def test_secondary_in_dual_instance_uses_secondary(self) -> None:
        b = LMStudioBackend(
            primary_url="http://primary/v1",
            primary_model="primary-model",
            secondary_url="http://secondary/v1",
            secondary_model="secondary-model",
            dual_mode=LMStudioDualMode.DUAL_INSTANCE,
        )
        assert b.resolve(VLMRole.SECONDARY) == (
            "http://secondary/v1",
            "secondary-model",
        )

    def test_dual_instance_without_secondary_collapses(self) -> None:
        b = LMStudioBackend(
            primary_url="http://primary/v1",
            primary_model="primary-model",
            dual_mode=LMStudioDualMode.DUAL_INSTANCE,
        )
        assert b.resolve(VLMRole.SECONDARY) == ("http://primary/v1", "primary-model")
        assert b.capabilities().supports_dual_vlm is False


class TestLMStudioBackendDispatch:
    def test_send_passes_resolved_model(self, fake_client_factory) -> None:
        _, instances = fake_client_factory
        b = LMStudioBackend(
            primary_url="http://primary/v1",
            primary_model="primary-model",
            secondary_url="http://secondary/v1",
            secondary_model="secondary-model",
            dual_mode=LMStudioDualMode.DUAL_INSTANCE,
        )
        request = VisionRequest(image_data="x", prompt="p")

        b.send_vision_request(request, role=VLMRole.SECONDARY)

        # One LMStudioClient instantiated for the secondary role
        assert len(instances) == 1
        inst = instances[0]
        assert inst._init_kwargs["base_url"] == "http://secondary/v1"
        assert inst._init_kwargs["model"] == "secondary-model"
        # The client itself is called with model="secondary-model" and
        # ``response_format=None`` because no schema was supplied.
        inst.send_vision_request.assert_called_once_with(
            request, model="secondary-model", response_format=None
        )

    def test_schema_translates_to_response_format(self, fake_client_factory) -> None:
        _, instances = fake_client_factory
        b = LMStudioBackend(
            primary_url="http://primary/v1",
            primary_model="primary-model",
        )
        request = VisionRequest(image_data="x", prompt="p")
        schema = {"type": "object", "properties": {"x": {"type": "integer"}}}

        b.send_vision_request(request, role=VLMRole.PRIMARY, schema=schema)

        inst = instances[0]
        kwargs = inst.send_vision_request.call_args.kwargs
        assert kwargs["model"] == "primary-model"
        # The schema is wrapped in LM Studio's ``response_format`` shape.
        rf = kwargs["response_format"]
        assert rf["type"] == "json_schema"
        assert rf["json_schema"]["name"] == "veridoc"
        assert rf["json_schema"]["schema"] == schema

    def test_per_role_clients_are_isolated(self, fake_client_factory) -> None:
        _, instances = fake_client_factory
        b = LMStudioBackend(
            primary_url="http://primary/v1",
            primary_model="primary-model",
            secondary_url="http://secondary/v1",
            secondary_model="secondary-model",
            dual_mode=LMStudioDualMode.DUAL_INSTANCE,
        )
        request = VisionRequest(image_data="x", prompt="p")

        b.send_vision_request(request, role=VLMRole.PRIMARY)
        b.send_vision_request(request, role=VLMRole.SECONDARY)
        b.send_vision_request(request, role=VLMRole.PRIMARY)  # should reuse

        # Two distinct clients (primary, secondary), primary reused on call 3.
        assert len(instances) == 2
        primary_client = instances[0]
        secondary_client = instances[1]
        assert primary_client._init_kwargs["model"] == "primary-model"
        assert secondary_client._init_kwargs["model"] == "secondary-model"
        # primary called twice, secondary called once
        assert primary_client.send_vision_request.call_count == 2
        assert secondary_client.send_vision_request.call_count == 1


class TestLMStudioBackendHealth:
    def test_probes_only_configured_roles(self, fake_client_factory) -> None:
        _, instances = fake_client_factory
        b = LMStudioBackend(
            primary_url="http://primary/v1",
            primary_model="primary-model",
            dual_mode=LMStudioDualMode.SINGLE_ONLY,
        )
        report = b.health()
        assert report.overall_healthy is True
        assert set(report.roles.keys()) == {VLMRole.PRIMARY}
        assert report.roles[VLMRole.PRIMARY]["base_url"] == "http://primary/v1"

    def test_dual_mode_probes_both(self, fake_client_factory) -> None:
        _, _ = fake_client_factory
        b = LMStudioBackend(
            primary_url="http://primary/v1",
            primary_model="primary-model",
            secondary_url="http://secondary/v1",
            secondary_model="secondary-model",
            dual_mode=LMStudioDualMode.DUAL_INSTANCE,
        )
        report = b.health()
        assert set(report.roles.keys()) == {VLMRole.PRIMARY, VLMRole.SECONDARY}
        assert report.overall_healthy is True

    def test_unhealthy_propagates(self, fake_client_factory) -> None:
        mock_cls, _instances = fake_client_factory
        # Override the side_effect for this test: client reports unhealthy
        unhealthy = MagicMock()
        unhealthy.is_healthy.return_value = False
        mock_cls.side_effect = lambda *a, **kw: unhealthy

        b = LMStudioBackend(
            primary_url="http://primary/v1",
            primary_model="primary-model",
        )
        report = b.health()
        assert report.overall_healthy is False
        assert report.roles[VLMRole.PRIMARY]["healthy"] is False


class TestLMStudioBackendClose:
    def test_close_closes_each_per_role_client(self, fake_client_factory) -> None:
        _, instances = fake_client_factory
        b = LMStudioBackend(
            primary_url="http://primary/v1",
            primary_model="primary-model",
            secondary_url="http://secondary/v1",
            secondary_model="secondary-model",
            dual_mode=LMStudioDualMode.DUAL_INSTANCE,
        )
        request = VisionRequest(image_data="x", prompt="p")
        b.send_vision_request(request, role=VLMRole.PRIMARY)
        b.send_vision_request(request, role=VLMRole.SECONDARY)

        b.close()
        for inst in instances:
            inst.close.assert_called_once()


class TestLMStudioBackendCapabilities:
    def test_caps_in_single_only_mode(self) -> None:
        b = LMStudioBackend(
            primary_url="http://primary/v1",
            primary_model="primary-model",
            dual_mode=LMStudioDualMode.SINGLE_ONLY,
        )
        caps = b.capabilities()
        assert caps.name == "lm_studio"
        assert caps.supports_dual_vlm is False
        assert caps.supports_constrained_decoding is True
        # The notes should warn the operator the secondary role is collapsed.
        assert any("dual_mode=single_only" in note for note in caps.notes)

    def test_caps_in_dual_instance_mode(self) -> None:
        b = LMStudioBackend(
            primary_url="http://primary/v1",
            primary_model="primary-model",
            secondary_url="http://secondary/v1",
            secondary_model="secondary-model",
            dual_mode=LMStudioDualMode.DUAL_INSTANCE,
        )
        caps = b.capabilities()
        assert caps.supports_dual_vlm is True
        assert caps.notes == ()
