"""
Unit tests for BaseAgent, AgentError hierarchy, and AgentResult.

Tests cover:
- AgentError and subclass construction
- AgentResult factory methods and to_dict
- BaseAgent initialization, properties, metrics
- BaseAgent.send_vision_request success/failure
- BaseAgent.send_vision_request_with_json
- BaseAgent.extract_field_value
- BaseAgent.merge_field_results and _values_match
- BaseAgent.log_operation_start / log_operation_complete
"""

from datetime import UTC, datetime
from unittest.mock import MagicMock, patch

import pytest

from src.agents.base import (
    AgentError,
    AgentResult,
    AnalysisError,
    BaseAgent,
    ExtractionError,
    OrchestrationError,
    ValidationError,
)
from src.client.lm_client import LMClientError, VisionResponse


# ---------------------------------------------------------------------------
# Concrete subclass for testing the abstract BaseAgent
# ---------------------------------------------------------------------------


class _StubAgent(BaseAgent):
    """Minimal concrete subclass for testing."""

    def __init__(self, client=None, **kwargs):
        super().__init__(name="stub", client=client, **kwargs)
        self._last_state = None

    def process(self, state):
        self._last_state = state
        return state


# ---------------------------------------------------------------------------
# TestAgentError
# ---------------------------------------------------------------------------


class TestAgentError:
    """Tests for AgentError and subclass hierarchy."""

    def test_basic_construction(self) -> None:
        err = AgentError("something failed")
        assert str(err) == "something failed"
        assert err.agent_name == ""
        assert err.recoverable is True
        assert err.details == {}

    def test_full_construction(self) -> None:
        err = AgentError(
            "bad input",
            agent_name="analyzer",
            recoverable=False,
            details={"field": "patient_name"},
        )
        assert err.agent_name == "analyzer"
        assert err.recoverable is False
        assert err.details == {"field": "patient_name"}

    def test_analysis_error_is_agent_error(self) -> None:
        assert issubclass(AnalysisError, AgentError)

    def test_extraction_error_is_agent_error(self) -> None:
        assert issubclass(ExtractionError, AgentError)

    def test_validation_error_is_agent_error(self) -> None:
        assert issubclass(ValidationError, AgentError)

    def test_orchestration_error_is_agent_error(self) -> None:
        assert issubclass(OrchestrationError, AgentError)

    def test_subclass_inherits_attributes(self) -> None:
        err = AnalysisError("no images", agent_name="analyzer", recoverable=False)
        assert err.agent_name == "analyzer"
        assert err.recoverable is False

    def test_catchable_as_agent_error(self) -> None:
        with pytest.raises(AgentError):
            raise ExtractionError("fail")


# ---------------------------------------------------------------------------
# TestAgentResult
# ---------------------------------------------------------------------------


class TestAgentResult:
    """Tests for AgentResult factory methods and to_dict."""

    def test_ok_factory(self) -> None:
        r = AgentResult.ok(
            data={"name": "Alice"},
            agent_name="extractor",
            operation="extract",
            vlm_calls=2,
            processing_time_ms=150,
        )
        assert r.success is True
        assert r.data == {"name": "Alice"}
        assert r.error is None
        assert r.agent_name == "extractor"
        assert r.vlm_calls == 2

    def test_fail_factory(self) -> None:
        r = AgentResult.fail(
            error="timeout",
            agent_name="validator",
            operation="validate",
        )
        assert r.success is False
        assert r.data is None
        assert r.error == "timeout"

    def test_ok_with_metadata(self) -> None:
        r = AgentResult.ok(data=42, extra_info="test")
        assert r.metadata.get("extra_info") == "test"

    def test_fail_with_metadata(self) -> None:
        r = AgentResult.fail(error="err", context="test")
        assert r.metadata.get("context") == "test"

    def test_to_dict(self) -> None:
        r = AgentResult.ok(data="result", agent_name="a", operation="op")
        d = r.to_dict()
        assert d["success"] is True
        assert d["data"] == "result"
        assert d["agent_name"] == "a"
        assert d["operation"] == "op"
        assert isinstance(d["metadata"], dict)

    def test_default_values(self) -> None:
        r = AgentResult(success=True)
        assert r.data is None
        assert r.error is None
        assert r.vlm_calls == 0
        assert r.processing_time_ms == 0


# ---------------------------------------------------------------------------
# TestBaseAgentInit
# ---------------------------------------------------------------------------


class TestBaseAgentInit:
    """Tests for BaseAgent initialization and properties."""

    def test_default_client(self) -> None:
        with patch("src.agents.base.LMStudioClient") as mock_cls:
            mock_cls.return_value = MagicMock()
            agent = _StubAgent()
            assert agent.name == "stub"
            assert agent._client is not None

    def test_custom_client(self) -> None:
        client = MagicMock()
        agent = _StubAgent(client=client)
        assert agent._client is client

    def test_name_property(self) -> None:
        agent = _StubAgent(client=MagicMock())
        assert agent.name == "stub"

    def test_initial_metrics(self) -> None:
        agent = _StubAgent(client=MagicMock())
        assert agent.vlm_calls == 0
        assert agent.total_processing_ms == 0

    def test_model_router_default_none(self) -> None:
        agent = _StubAgent(client=MagicMock())
        assert agent.model_router is None

    def test_model_router_injection(self) -> None:
        router = MagicMock()
        agent = _StubAgent(client=MagicMock(), model_router=router)
        assert agent.model_router is router


# ---------------------------------------------------------------------------
# TestBaseAgentMetrics
# ---------------------------------------------------------------------------


class TestBaseAgentMetrics:
    """Tests for metrics tracking."""

    def test_get_metrics(self) -> None:
        agent = _StubAgent(client=MagicMock())
        m = agent.get_metrics()
        assert m["agent_name"] == "stub"
        assert m["vlm_calls"] == 0
        assert m["total_processing_ms"] == 0

    def test_reset_metrics(self) -> None:
        agent = _StubAgent(client=MagicMock())
        agent._vlm_calls = 5
        agent._total_processing_ms = 1000
        agent.reset_metrics()
        assert agent.vlm_calls == 0
        assert agent.total_processing_ms == 0

    def test_vlm_calls_increment(self) -> None:
        client = MagicMock()
        client.send_vision_request.return_value = VisionResponse(
            content="test", parsed_json=None, latency_ms=50,
        )
        agent = _StubAgent(client=client)
        agent.send_vision_request("img", "prompt")
        assert agent.vlm_calls == 1

    def test_processing_time_accumulates(self) -> None:
        client = MagicMock()
        client.send_vision_request.return_value = VisionResponse(
            content="test", parsed_json=None, latency_ms=100,
        )
        agent = _StubAgent(client=client)
        agent.send_vision_request("img", "prompt")
        agent.send_vision_request("img", "prompt")
        assert agent.total_processing_ms == 200


# ---------------------------------------------------------------------------
# TestSendVisionRequest
# ---------------------------------------------------------------------------


class TestSendVisionRequest:
    """Tests for send_vision_request."""

    def test_success(self) -> None:
        expected_resp = VisionResponse(
            content='{"a":1}', parsed_json={"a": 1}, latency_ms=150,
        )
        client = MagicMock()
        client.send_vision_request.return_value = expected_resp

        agent = _StubAgent(client=client)
        resp = agent.send_vision_request("img_data", "extract fields")

        assert resp is expected_resp
        client.send_vision_request.assert_called_once()

    def test_lm_client_error_raises_agent_error(self) -> None:
        client = MagicMock()
        client.send_vision_request.side_effect = LMClientError("connection refused")

        agent = _StubAgent(client=client)
        with pytest.raises(AgentError, match="VLM request failed"):
            agent.send_vision_request("img", "prompt")

    def test_custom_parameters(self) -> None:
        client = MagicMock()
        client.send_vision_request.return_value = VisionResponse(
            content="ok", latency_ms=10,
        )
        agent = _StubAgent(client=client)
        agent.send_vision_request(
            "img", "prompt", system_prompt="sys", max_tokens=2048, temperature=0.5,
        )
        call_args = client.send_vision_request.call_args
        req = call_args[0][0]
        assert req.prompt == "prompt"
        assert req.system_prompt == "sys"
        assert req.max_tokens == 2048
        assert req.temperature == 0.5


# ---------------------------------------------------------------------------
# TestSendVisionRequestWithJson
# ---------------------------------------------------------------------------


class TestSendVisionRequestWithJson:
    """Tests for send_vision_request_with_json."""

    def test_success_returns_json(self) -> None:
        client = MagicMock()
        client.send_vision_request.return_value = VisionResponse(
            content='{"name": "John"}',
            parsed_json={"name": "John"},
            latency_ms=100,
        )
        agent = _StubAgent(client=client)
        result = agent.send_vision_request_with_json("img", "extract")
        assert result == {"name": "John"}

    def test_no_json_raises_agent_error(self) -> None:
        client = MagicMock()
        client.send_vision_request.return_value = VisionResponse(
            content="plain text no json here",
            parsed_json=None,
            latency_ms=100,
        )
        agent = _StubAgent(client=client)
        with pytest.raises(AgentError, match="Failed to extract JSON"):
            agent.send_vision_request_with_json("img", "extract")


# ---------------------------------------------------------------------------
# TestExtractFieldValue
# ---------------------------------------------------------------------------


class TestExtractFieldValue:
    """Tests for extract_field_value with dot notation."""

    def setup_method(self) -> None:
        self.agent = _StubAgent(client=MagicMock())

    def test_simple_key(self) -> None:
        assert self.agent.extract_field_value({"a": 1}, "a") == 1

    def test_nested_key(self) -> None:
        data = {"fields": {"patient_name": {"value": "Alice"}}}
        assert self.agent.extract_field_value(data, "fields.patient_name.value") == "Alice"

    def test_missing_key_returns_default(self) -> None:
        assert self.agent.extract_field_value({"a": 1}, "b") is None

    def test_missing_nested_returns_default(self) -> None:
        assert self.agent.extract_field_value({"a": {"b": 1}}, "a.c") is None

    def test_custom_default(self) -> None:
        assert self.agent.extract_field_value({}, "x", default="N/A") == "N/A"

    def test_non_dict_intermediate(self) -> None:
        # If an intermediate value is not a dict, should return default
        assert self.agent.extract_field_value({"a": "string"}, "a.b") is None


# ---------------------------------------------------------------------------
# TestMergeFieldResults
# ---------------------------------------------------------------------------


class TestMergeFieldResults:
    """Tests for merge_field_results dual-pass merging."""

    def setup_method(self) -> None:
        self.agent = _StubAgent(client=MagicMock())

    def test_both_agree(self) -> None:
        pass1 = {"name": {"value": "Alice", "confidence": 0.9}}
        pass2 = {"name": {"value": "Alice", "confidence": 0.8}}
        merged = self.agent.merge_field_results(pass1, pass2)

        assert merged["name"]["passes_agree"] is True
        # Agreement boosts confidence
        assert merged["name"]["confidence"] > 0.85

    def test_disagree_uses_higher_confidence(self) -> None:
        pass1 = {"name": {"value": "Alice", "confidence": 0.9}}
        pass2 = {"name": {"value": "Bob", "confidence": 0.5}}
        merged = self.agent.merge_field_results(pass1, pass2)

        assert merged["name"]["passes_agree"] is False
        assert merged["name"]["value"] == "Alice"

    def test_field_only_in_pass1(self) -> None:
        pass1 = {"name": {"value": "Alice", "confidence": 0.8}}
        pass2 = {}
        merged = self.agent.merge_field_results(pass1, pass2)

        assert "name" in merged
        # One None vs one value → disagreement
        assert merged["name"]["passes_agree"] is False

    def test_field_only_in_pass2(self) -> None:
        pass1 = {}
        pass2 = {"name": {"value": "Bob", "confidence": 0.7}}
        merged = self.agent.merge_field_results(pass1, pass2)

        assert "name" in merged

    def test_both_empty(self) -> None:
        merged = self.agent.merge_field_results({}, {})
        assert merged == {}

    def test_plain_values(self) -> None:
        """Non-dict values trigger AttributeError on location — test dict form."""
        pass1 = {"name": {"value": "Alice", "confidence": 0.5}}
        pass2 = {"name": {"value": "Alice", "confidence": 0.5}}
        merged = self.agent.merge_field_results(pass1, pass2)
        assert merged["name"]["passes_agree"] is True


# ---------------------------------------------------------------------------
# TestValuesMatch
# ---------------------------------------------------------------------------


class TestValuesMatch:
    """Tests for _values_match comparison logic."""

    def setup_method(self) -> None:
        self.agent = _StubAgent(client=MagicMock())

    def test_both_none(self) -> None:
        assert self.agent._values_match(None, None) is True

    def test_one_none(self) -> None:
        assert self.agent._values_match("a", None) is False
        assert self.agent._values_match(None, "a") is False

    def test_string_case_insensitive(self) -> None:
        assert self.agent._values_match("Alice Smith", "alice smith") is True

    def test_string_whitespace_normalized(self) -> None:
        assert self.agent._values_match("Alice  Smith", "Alice Smith") is True

    def test_string_different(self) -> None:
        assert self.agent._values_match("Alice", "Bob") is False

    def test_numeric_exact(self) -> None:
        assert self.agent._values_match(100.0, 100.0) is True

    def test_numeric_tolerance(self) -> None:
        # Within 0.1% tolerance
        assert self.agent._values_match(1000.0, 1000.5) is True

    def test_numeric_beyond_tolerance(self) -> None:
        assert self.agent._values_match(100.0, 200.0) is False

    def test_numeric_zero(self) -> None:
        assert self.agent._values_match(0, 0) is True
        assert self.agent._values_match(0, 1) is False

    def test_list_match(self) -> None:
        assert self.agent._values_match([1, 2], [1, 2]) is True

    def test_list_different_length(self) -> None:
        assert self.agent._values_match([1, 2], [1]) is False

    def test_list_different_values(self) -> None:
        assert self.agent._values_match([1, 2], [1, 3]) is False

    def test_direct_equality(self) -> None:
        assert self.agent._values_match(True, True) is True
        assert self.agent._values_match(True, False) is False


# ---------------------------------------------------------------------------
# TestLogOperations
# ---------------------------------------------------------------------------


class TestLogOperations:
    """Tests for log_operation_start and log_operation_complete."""

    def test_log_start_returns_datetime(self) -> None:
        agent = _StubAgent(client=MagicMock())
        result = agent.log_operation_start("test_op")
        assert isinstance(result, datetime)

    def test_log_complete_returns_duration(self) -> None:
        agent = _StubAgent(client=MagicMock())
        start = datetime.now(UTC)
        duration = agent.log_operation_complete("test_op", start, success=True)
        assert isinstance(duration, int)
        assert duration >= 0

    def test_log_complete_failure(self) -> None:
        agent = _StubAgent(client=MagicMock())
        start = datetime.now(UTC)
        duration = agent.log_operation_complete("test_op", start, success=False)
        assert isinstance(duration, int)


# ---------------------------------------------------------------------------
# TestProcess
# ---------------------------------------------------------------------------


class TestProcess:
    """Tests for process() abstract method enforcement."""

    def test_abstract_method_enforcement(self) -> None:
        """Cannot instantiate BaseAgent directly."""
        with pytest.raises(TypeError):
            BaseAgent(name="fail")  # type: ignore

    def test_subclass_process_called(self) -> None:
        agent = _StubAgent(client=MagicMock())
        state = {"status": "test"}
        result = agent.process(state)
        assert result is state
        assert agent._last_state is state
