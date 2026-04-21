"""Tests for llmdebug._json adapter module."""

from __future__ import annotations

import pytest

from llmdebug import _json as json

# ---------------------------------------------------------------------------
# dumps - basic round-trips
# ---------------------------------------------------------------------------


class TestDumpsBasic:
    def test_simple_dict(self) -> None:
        result = json.dumps({"a": 1, "b": 2})
        assert '"a"' in result
        assert "1" in result

    def test_simple_list(self) -> None:
        result = json.dumps([1, 2, 3])
        assert "1" in result and "2" in result and "3" in result

    def test_none(self) -> None:
        assert json.dumps(None) == "null"

    def test_bool_true(self) -> None:
        assert json.dumps(True) == "true"

    def test_bool_false(self) -> None:
        assert json.dumps(False) == "false"

    def test_integer(self) -> None:
        assert json.dumps(42) == "42"

    def test_string(self) -> None:
        result = json.dumps("hello")
        assert result == '"hello"'

    def test_nested(self) -> None:
        result = json.dumps({"x": {"y": [1, 2]}})
        data = json.loads(result)
        assert data["x"]["y"] == [1, 2]


# ---------------------------------------------------------------------------
# dumps - indent support
# ---------------------------------------------------------------------------


class TestDumpsIndent:
    def test_indent_2_produces_newlines(self) -> None:
        result = json.dumps({"a": 1}, indent=2)
        assert "\n" in result

    def test_indent_2_roundtrip(self) -> None:
        payload = {"key": "value", "num": 99}
        result = json.dumps(payload, indent=2)
        data = json.loads(result)
        assert data == payload

    def test_indent_non_2_raises(self) -> None:
        with pytest.raises(ValueError, match="indent=2"):
            json.dumps({"a": 1}, indent=4)

    def test_indent_1_raises(self) -> None:
        with pytest.raises(ValueError, match="indent=2"):
            json.dumps([1], indent=1)

    def test_indent_0_raises(self) -> None:
        with pytest.raises(ValueError, match="indent=2"):
            json.dumps({}, indent=0)


# ---------------------------------------------------------------------------
# dumps - separators support
# ---------------------------------------------------------------------------


class TestDumpsSeparators:
    def test_compact_separators_no_spaces(self) -> None:
        result = json.dumps({"a": 1, "b": 2}, separators=(",", ":"))
        assert ", " not in result
        assert ": " not in result

    def test_spaced_separators(self) -> None:
        result = json.dumps({"a": 1}, separators=(", ", ": "))
        assert ": " in result

    def test_none_separators_without_indent_uses_spaced(self) -> None:
        # separators=None with no indent → spaced separators applied
        result = json.dumps({"a": 1}, separators=None)
        assert ": " in result

    def test_unsupported_separators_raises(self) -> None:
        with pytest.raises(ValueError, match="separators"):
            json.dumps({"a": 1}, separators=("; ", " = "))

    def test_unsupported_separators_tuple_raises(self) -> None:
        with pytest.raises(ValueError, match="separators"):
            json.dumps([1, 2], separators=(" ", "->"))


# ---------------------------------------------------------------------------
# dumps - sort_keys
# ---------------------------------------------------------------------------


class TestDumpsSortKeys:
    def test_sort_keys_true(self) -> None:
        result = json.dumps({"z": 1, "a": 2, "m": 3}, sort_keys=True)
        a_pos = result.index('"a"')
        m_pos = result.index('"m"')
        z_pos = result.index('"z"')
        assert a_pos < m_pos < z_pos

    def test_sort_keys_false_default(self) -> None:
        # Just ensure it doesn't raise
        result = json.dumps({"z": 1, "a": 2}, sort_keys=False)
        assert '"z"' in result
        assert '"a"' in result


# ---------------------------------------------------------------------------
# dumps - ensure_ascii
# ---------------------------------------------------------------------------


class TestDumpsEnsureAscii:
    def test_ensure_ascii_true_escapes_non_ascii(self) -> None:
        result = json.dumps({"msg": "caf\u00e9"}, ensure_ascii=True)
        assert "\u00e9" not in result
        assert "\\u" in result

    def test_ensure_ascii_false_preserves_unicode(self) -> None:
        result = json.dumps({"msg": "caf\u00e9"}, ensure_ascii=False)
        assert "\u00e9" in result

    def test_ensure_ascii_surrogate_pair_encoding(self) -> None:
        # U+1F600 GRINNING FACE (supplementary plane)
        result = json.dumps({"emoji": "\U0001f600"}, ensure_ascii=True)
        assert "\\ud83d" in result.lower() or "\\uD83D" in result
        # Verify it decodes cleanly via loads
        data = json.loads(result)
        assert data["emoji"] == "\U0001f600"

    def test_ensure_ascii_bmp_char(self) -> None:
        result = json.dumps({"ch": "\u4e2d"}, ensure_ascii=True)
        assert "\\u4e2d" in result

    def test_ensure_ascii_false_bmp_char_preserved(self) -> None:
        result = json.dumps({"ch": "\u4e2d"}, ensure_ascii=False)
        assert "\u4e2d" in result


# ---------------------------------------------------------------------------
# dumps - default callable
# ---------------------------------------------------------------------------


class TestDumpsDefault:
    def test_default_callable_for_unserializable(self) -> None:
        class _Custom:
            def __init__(self, v: int) -> None:
                self.v = v

        result = json.dumps({"obj": _Custom(7)}, default=lambda o: {"v": o.v})
        data = json.loads(result)
        assert data["obj"]["v"] == 7

    def test_default_str_fallback(self) -> None:
        from datetime import date

        d = date(2026, 1, 1)
        result = json.dumps({"date": d}, default=str)
        data = json.loads(result)
        assert "2026" in data["date"]


# ---------------------------------------------------------------------------
# loads - basic
# ---------------------------------------------------------------------------


class TestLoads:
    def test_loads_string(self) -> None:
        assert json.loads('"hello"') == "hello"

    def test_loads_bytes(self) -> None:
        assert json.loads(b'{"a": 1}') == {"a": 1}

    def test_loads_bytearray(self) -> None:
        assert json.loads(bytearray(b"[1,2,3]")) == [1, 2, 3]

    def test_loads_number(self) -> None:
        assert json.loads("42") == 42

    def test_loads_null(self) -> None:
        assert json.loads("null") is None

    def test_loads_bool(self) -> None:
        assert json.loads("true") is True
        assert json.loads("false") is False

    def test_loads_invalid_raises_json_decode_error(self) -> None:
        with pytest.raises(json.JSONDecodeError):
            json.loads("not json")


# ---------------------------------------------------------------------------
# _apply_separators - edge cases
# ---------------------------------------------------------------------------


class TestApplySeparators:
    def test_string_with_backslash_before_quote(self) -> None:
        # A string like {"k": "a\\\"b"} must not confuse the parser
        payload = {"k": 'a\\"b'}
        result = json.dumps(payload, separators=(",", ":"))
        data = json.loads(result)
        assert data["k"] == 'a\\"b'

    def test_string_containing_comma_and_colon(self) -> None:
        payload = {"msg": "key: value, other"}
        result = json.dumps(payload, separators=(",", ":"))
        data = json.loads(result)
        assert data["msg"] == "key: value, other"

    def test_nested_strings_with_separators(self) -> None:
        payload = {"a": {"b": "c:d,e"}}
        result = json.dumps(payload, separators=(",", ":"))
        data = json.loads(result)
        assert data["a"]["b"] == "c:d,e"


# ---------------------------------------------------------------------------
# JSONDecodeError alias
# ---------------------------------------------------------------------------


def test_json_decode_error_is_stdlib_alias() -> None:
    import json as _stdlib

    assert json.JSONDecodeError is _stdlib.JSONDecodeError


# ---------------------------------------------------------------------------
# Coverage gap tests — lines 36, 69, 76, 85, 115
# ---------------------------------------------------------------------------


class TestDumpsEnsureAsciiCoverage:
    """Additional ensure_ascii coverage for non-ASCII input paths."""

    def test_ensure_ascii_false_non_ascii_preserved(self) -> None:
        """Line 112: ensure_ascii=False returns raw unicode (skip _escape_non_ascii)."""
        result = json.dumps({"key": "données"}, ensure_ascii=False)
        assert "données" in result
        data = json.loads(result)
        assert data["key"] == "données"

    def test_ensure_ascii_true_non_ascii_escaped(self) -> None:
        """Line 76: ensure_ascii=True branch calls _escape_non_ascii."""
        result = json.dumps({"key": "données"}, ensure_ascii=True)
        assert "\\u" in result
        data = json.loads(result)
        assert data["key"] == "données"


class TestDumpsDefaultSeparators:
    """Coverage for separators=None and no indent → spaced separators (line 69)."""

    def test_none_separators_no_indent_applies_spaced(self) -> None:
        """Line 69 / 108: separators is None and indent is None → _apply_separators."""
        result = json.dumps({"a": 1, "b": 2})
        # Default call: separators=None, indent=None → spaced
        assert ": " in result
        assert ", " in result or result.count(",") >= 1

    def test_spaced_separators_explicit(self) -> None:
        """Line 69 / 108: separators == _SPACED_SEPARATORS branch."""
        result = json.dumps({"x": 99}, separators=(", ", ": "))
        assert ": " in result


class TestLoadsBasicCoverage:
    """Basic loads coverage (line 115)."""

    def test_loads_dict_roundtrip(self) -> None:
        """Line 115: loads() function definition."""
        payload = {"alpha": [1, 2], "beta": True}
        serialized = json.dumps(payload)
        assert json.loads(serialized) == payload

    def test_loads_memoryview(self) -> None:
        data = memoryview(b'{"k": 42}')
        assert json.loads(data) == {"k": 42}


class TestApplySeparatorsCustom:
    """Coverage for _apply_separators with custom separator tuples (line 36)."""

    def test_apply_separators_directly(self) -> None:
        """Line 36: _apply_separators function definition."""
        from llmdebug._json import _apply_separators

        text = '{"a":1,"b":2}'
        result = _apply_separators(text, separators=(", ", ": "))
        assert result == '{"a": 1, "b": 2}'

    def test_apply_separators_preserves_string_content(self) -> None:
        """Colons/commas inside strings must not be modified."""
        from llmdebug._json import _apply_separators

        text = '{"msg":"hello, world: foo","x":1}'
        result = _apply_separators(text, separators=(", ", ": "))
        parsed = json.loads(result)
        assert parsed["msg"] == "hello, world: foo"
        assert parsed["x"] == 1

    def test_apply_separators_skips_trailing_space_after_comma(self) -> None:
        """Lines 68-69: whitespace after comma is consumed by the while-loop body."""
        from llmdebug._json import _apply_separators

        # Input already has a space after the comma; compact separators should strip it.
        text = '{"a":1, "b":2}'
        result = _apply_separators(text, separators=(",", ":"))
        assert result == '{"a":1,"b":2}'

    def test_apply_separators_skips_trailing_space_after_colon(self) -> None:
        """Lines 75-76: whitespace after colon is consumed by the while-loop body."""
        from llmdebug._json import _apply_separators

        # Input already has a space after the colon; compact separators should strip it.
        text = '{"a": 1}'
        result = _apply_separators(text, separators=(",", ":"))
        assert result == '{"a":1}'
