"""
Phase 2 — bbox round-trip verification unit tests.

Coverage:

* Bbox normalisation (clamping, [x, y, w, h] vs [x1, y1, x2, y2]).
* Bbox padding stays within page bounds.
* Image cropping returns a smaller PNG with a data URI prefix.
* Decision rule: which pass the round-trip ratifies under varying
  similarity values.
* Backend exceptions surface as ``error`` field (no propagation).
* Crop failure surfaces as ``error`` field with ``winning_pass=neither``.
* Quote/markdown stripping on raw VLM responses.

No live VLM, no GPU. We patch ``backend.send_vision_request`` to
return canned ``VisionResponse`` instances and use a tiny PIL image as
the page input.
"""

from __future__ import annotations

import base64
import io
from typing import Any
from unittest.mock import MagicMock

import pytest
from PIL import Image as PILImage

from src.client.backends.protocol import BackendCapabilities, VLMRole
from src.client.lm_client import VisionResponse
from src.validation.bbox_roundtrip import (
    BboxRoundtripResult,
    _expand_bbox,
    _normalise_bbox,
    _string_similarity,
    crop_image_to_bbox,
    perform_bbox_roundtrip,
    value_similarity,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_test_page(width_px: int = 200, height_px: int = 200) -> str:
    """Generate a tiny PNG and return as a data URI."""
    img = PILImage.new("RGB", (width_px, height_px), color=(255, 255, 255))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _stub_backend(content: str = "100") -> MagicMock:
    backend = MagicMock()
    backend.name = "stub"
    backend.capabilities.return_value = BackendCapabilities(
        name="stub",
        supports_dual_vlm=True,
        supports_constrained_decoding=True,
        supports_logprobs=False,
        supports_multi_image=True,
        supports_tensor_parallelism=False,
    )
    backend.resolve.return_value = ("http://stub", "stub-model")
    backend.send_vision_request.return_value = VisionResponse(
        content=content,
        parsed_json=None,
        model="stub-model",
        usage={"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
        latency_ms=10,
        request_id="r",
    )
    return backend


# ---------------------------------------------------------------------------
# Pure helpers
# ---------------------------------------------------------------------------


class TestNormaliseBbox:
    def test_passthrough_valid_bbox(self) -> None:
        assert _normalise_bbox([0.1, 0.2, 0.3, 0.4]) == (0.1, 0.2, 0.3, 0.4)

    def test_clamps_to_unit_range(self) -> None:
        result = _normalise_bbox([-0.1, -0.2, 1.5, 2.0])
        assert result == (0.0, 0.0, 1.0, 1.0)

    def test_xywh_form_converted(self) -> None:
        # If x2 < x1, treat as [x, y, w, h].
        # Input [0.1, 0.2, 0.05, 0.1] would have x2=0.05 < x1=0.1; coerce.
        result = _normalise_bbox([0.1, 0.2, 0.05, 0.1])
        # After coercion: x1=0.1, y1=0.2, x2=0.1+0.05=0.15, y2=0.2+0.1=0.3
        assert result[0] == pytest.approx(0.1)
        assert result[1] == pytest.approx(0.2)
        assert result[2] == pytest.approx(0.15)
        assert result[3] == pytest.approx(0.3)

    def test_invalid_length_raises(self) -> None:
        with pytest.raises(ValueError, match="must have 4 elements"):
            _normalise_bbox([0.1, 0.2, 0.3])

    def test_degenerate_after_clamp_raises(self) -> None:
        with pytest.raises(ValueError, match="degenerate"):
            _normalise_bbox([0.5, 0.5, 0.5, 0.5])  # zero area


class TestExpandBbox:
    def test_pads_outward(self) -> None:
        out = _expand_bbox((0.4, 0.4, 0.6, 0.6), padding_pct=0.5)
        # bbox dim = 0.2; padding 0.5 * 0.2 = 0.1 → expand to (0.3, 0.3, 0.7, 0.7)
        assert out[0] == pytest.approx(0.3)
        assert out[1] == pytest.approx(0.3)
        assert out[2] == pytest.approx(0.7)
        assert out[3] == pytest.approx(0.7)

    def test_clamps_at_edges(self) -> None:
        out = _expand_bbox((0.0, 0.0, 0.2, 0.2), padding_pct=0.5)
        # x1 would be -0.1, clamps to 0.0
        assert out[0] == 0.0
        assert out[1] == 0.0


# ---------------------------------------------------------------------------
# Crop
# ---------------------------------------------------------------------------


class TestCrop:
    def test_returns_data_uri_and_size(self) -> None:
        page = _make_test_page(200, 200)
        cropped_uri, size = crop_image_to_bbox(page, [0.25, 0.25, 0.75, 0.75])
        assert cropped_uri.startswith("data:image/png;base64,")
        # Cropped region is 50% of 200px in each dim = 100, plus 10% padding
        # → ~120px each side; allow some rounding slop.
        assert 90 <= size[0] <= 140
        assert 90 <= size[1] <= 140

    def test_works_with_bare_base64(self) -> None:
        page = _make_test_page(100, 100)
        bare = page.split(",", 1)[1]
        cropped_uri, _ = crop_image_to_bbox(bare, [0.1, 0.1, 0.5, 0.5])
        assert cropped_uri.startswith("data:image/png;base64,")


# ---------------------------------------------------------------------------
# Similarity
# ---------------------------------------------------------------------------


class TestSimilarity:
    def test_identical_strings(self) -> None:
        assert _string_similarity("abc", "abc") == 1.0

    def test_case_insensitive_near_match(self) -> None:
        assert _string_similarity("ABC", "abc") == pytest.approx(0.95)

    def test_completely_different(self) -> None:
        # Allow some overlap tolerance — short strings always match a bit.
        assert _string_similarity("abc", "xyz") < 0.5

    def test_value_similarity_with_none(self) -> None:
        assert value_similarity(None, "x") == 0.0
        assert value_similarity("x", None) == 0.0

    def test_value_similarity_numeric_string_round_trip(self) -> None:
        # Both coerced to "100"
        assert value_similarity(100, "100") == 1.0


# ---------------------------------------------------------------------------
# perform_bbox_roundtrip — decision rule
# ---------------------------------------------------------------------------


class TestRoundtripDecision:
    def test_strong_match_to_pass1(self) -> None:
        page = _make_test_page()
        backend = _stub_backend(content="alice")
        result = perform_bbox_roundtrip(
            backend=backend,
            image_data_uri=page,
            bbox=[0.1, 0.1, 0.4, 0.4],
            pass1_value="alice",
            pass2_value="alyce",
            field_name="name",
        )
        assert result.winning_pass == "pass1"
        assert result.value == "alice"
        assert result.similarity_to_pass1 > result.similarity_to_pass2

    def test_strong_match_to_pass2(self) -> None:
        page = _make_test_page()
        backend = _stub_backend(content="alyce")
        result = perform_bbox_roundtrip(
            backend=backend,
            image_data_uri=page,
            bbox=[0.1, 0.1, 0.4, 0.4],
            pass1_value="bob",
            pass2_value="alyce",
            field_name="name",
        )
        assert result.winning_pass == "pass2"

    def test_neither_when_response_is_unrelated(self) -> None:
        page = _make_test_page()
        backend = _stub_backend(content="zzz")
        result = perform_bbox_roundtrip(
            backend=backend,
            image_data_uri=page,
            bbox=[0.1, 0.1, 0.4, 0.4],
            pass1_value="alice",
            pass2_value="bob",
            field_name="name",
        )
        assert result.winning_pass == "neither"

    def test_null_response(self) -> None:
        page = _make_test_page()
        backend = _stub_backend(content="null")
        result = perform_bbox_roundtrip(
            backend=backend,
            image_data_uri=page,
            bbox=[0.1, 0.1, 0.4, 0.4],
            pass1_value="alice",
            pass2_value="bob",
            field_name="name",
        )
        assert result.value is None
        assert result.winning_pass == "neither"

    def test_strips_quotes(self) -> None:
        page = _make_test_page()
        backend = _stub_backend(content='"alice"')
        result = perform_bbox_roundtrip(
            backend=backend,
            image_data_uri=page,
            bbox=[0.1, 0.1, 0.4, 0.4],
            pass1_value="alice",
            pass2_value="bob",
            field_name="name",
        )
        assert result.value == "alice"

    def test_strips_backticks(self) -> None:
        page = _make_test_page()
        backend = _stub_backend(content="`alice`")
        result = perform_bbox_roundtrip(
            backend=backend,
            image_data_uri=page,
            bbox=[0.1, 0.1, 0.4, 0.4],
            pass1_value="alice",
            pass2_value="bob",
            field_name="name",
        )
        assert result.value == "alice"

    def test_close_call_falls_to_neither(self) -> None:
        # Roundtrip value is similar to BOTH candidates (both are 1-char
        # changes from the response). Decision rule requires |delta| >= 0.1.
        page = _make_test_page()
        backend = _stub_backend(content="abc")
        result = perform_bbox_roundtrip(
            backend=backend,
            image_data_uri=page,
            bbox=[0.1, 0.1, 0.4, 0.4],
            pass1_value="abd",
            pass2_value="abe",
            field_name="x",
        )
        assert result.winning_pass == "neither"


# ---------------------------------------------------------------------------
# Error paths
# ---------------------------------------------------------------------------


class TestRoundtripErrors:
    def test_backend_exception_returns_error_result(self) -> None:
        page = _make_test_page()
        backend = MagicMock()
        backend.name = "broken"
        backend.send_vision_request.side_effect = RuntimeError("vlm timeout")
        result = perform_bbox_roundtrip(
            backend=backend,
            image_data_uri=page,
            bbox=[0.1, 0.1, 0.4, 0.4],
            pass1_value="x",
            pass2_value="y",
            field_name="z",
        )
        assert result.winning_pass == "neither"
        assert result.error is not None
        assert "vlm timeout" in result.error
        assert result.value is None

    def test_invalid_bbox_returns_error_result(self) -> None:
        page = _make_test_page()
        backend = _stub_backend()
        result = perform_bbox_roundtrip(
            backend=backend,
            image_data_uri=page,
            bbox=[0.5, 0.5, 0.5, 0.5],  # degenerate
            pass1_value="x",
            pass2_value="y",
            field_name="z",
        )
        assert result.winning_pass == "neither"
        assert result.error is not None
        assert "degenerate" in result.error or "crop_failed" in result.error

    def test_role_passed_to_backend(self) -> None:
        page = _make_test_page()
        backend = _stub_backend()
        perform_bbox_roundtrip(
            backend=backend,
            image_data_uri=page,
            bbox=[0.1, 0.1, 0.4, 0.4],
            pass1_value="x",
            pass2_value="y",
            field_name="z",
            role=VLMRole.CRITIC,
        )
        kwargs = backend.send_vision_request.call_args.kwargs
        assert kwargs["role"] is VLMRole.CRITIC
