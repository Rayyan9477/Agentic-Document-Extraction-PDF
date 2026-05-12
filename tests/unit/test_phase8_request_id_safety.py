"""V3 Phase 8 — X-Request-ID sanitisation."""

from __future__ import annotations

from src.api.app import _safe_request_id


class TestSafeRequestId:
    def test_normal_value_passes(self) -> None:
        assert _safe_request_id("req-abc-123") == "req-abc-123"

    def test_empty_returns_uuid_hex(self) -> None:
        out = _safe_request_id("")
        assert out  # non-empty
        assert len(out) == 32  # uuid4().hex is 32 chars
        assert all(c in "0123456789abcdef" for c in out)

    def test_none_returns_uuid_hex(self) -> None:
        out = _safe_request_id(None)
        assert len(out) == 32

    def test_path_traversal_replaced(self) -> None:
        out = _safe_request_id("../../../etc/passwd")
        # Forward-slash fails the regex → replaced with uuid hex.
        assert "/" not in out
        assert ".." not in out
        assert len(out) == 32

    def test_backslash_replaced(self) -> None:
        out = _safe_request_id("..\\..\\windows")
        assert "\\" not in out
        assert len(out) == 32

    def test_null_byte_replaced(self) -> None:
        out = _safe_request_id("abc\x00def")
        assert "\x00" not in out

    def test_too_long_replaced(self) -> None:
        out = _safe_request_id("a" * 100)
        assert len(out) == 32

    def test_special_chars_replaced(self) -> None:
        # Spaces, slashes, semicolons, CRLF — all rejected.
        for bad in ("a b c", "a;b", "a\r\nb", "../x", "x/y", "x?q=1"):
            out = _safe_request_id(bad)
            assert len(out) == 32, f"input {bad!r} should be replaced"

    def test_underscore_dash_allowed(self) -> None:
        assert _safe_request_id("a_b-c") == "a_b-c"
