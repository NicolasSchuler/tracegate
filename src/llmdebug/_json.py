"""Thin JSON compatibility helpers backed by orjson."""

from __future__ import annotations

import json as _stdlib_json
from collections.abc import Callable
from typing import Any

import orjson

JSONDecodeError = _stdlib_json.JSONDecodeError
JSONDecoder = _stdlib_json.JSONDecoder

_COMPACT_SEPARATORS = (",", ":")
_SPACED_SEPARATORS = (", ", ": ")


def _escape_non_ascii(text: str) -> str:
    """Escape non-ASCII code points using JSON unicode escape sequences."""
    escaped: list[str] = []
    for char in text:
        codepoint = ord(char)
        if codepoint < 128:
            escaped.append(char)
            continue
        if codepoint <= 0xFFFF:
            escaped.append(f"\\u{codepoint:04x}")
            continue
        codepoint -= 0x10000
        high_surrogate = 0xD800 + (codepoint >> 10)
        low_surrogate = 0xDC00 + (codepoint & 0x3FF)
        escaped.append(f"\\u{high_surrogate:04x}\\u{low_surrogate:04x}")
    return "".join(escaped)


def _apply_separators(text: str, *, separators: tuple[str, str]) -> str:
    """Apply separator spacing outside string literals."""
    comma_sep, colon_sep = separators
    out: list[str] = []
    in_string = False
    escaped = False
    idx = 0
    length = len(text)

    while idx < length:
        char = text[idx]

        if in_string:
            out.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            idx += 1
            continue

        if char == '"':
            in_string = True
            out.append(char)
            idx += 1
            continue

        if char == ",":
            out.append(comma_sep)
            idx += 1
            while idx < length and text[idx] in {" ", "\t"}:
                idx += 1
            continue

        if char == ":":
            out.append(colon_sep)
            idx += 1
            while idx < length and text[idx] in {" ", "\t"}:
                idx += 1
            continue

        out.append(char)
        idx += 1

    return "".join(out)


def dumps(
    value: Any,
    *,
    indent: int | None = None,
    ensure_ascii: bool = True,
    default: Callable[[Any], Any] | None = None,
    sort_keys: bool = False,
    separators: tuple[str, str] | None = None,
) -> str:
    """Serialize to a JSON string using orjson with stdlib-like kwargs."""
    option = orjson.OPT_NON_STR_KEYS
    if indent is not None:
        if indent != 2:
            raise ValueError("llmdebug JSON adapter only supports indent=2")
        option |= orjson.OPT_INDENT_2
    if sort_keys:
        option |= orjson.OPT_SORT_KEYS
    if separators not in (None, _COMPACT_SEPARATORS, _SPACED_SEPARATORS):
        raise ValueError(
            "llmdebug JSON adapter only supports separators=None, (',', ':'), or (', ', ': ')"
        )

    rendered = orjson.dumps(value, default=default, option=option).decode("utf-8")
    if separators == _SPACED_SEPARATORS or (separators is None and indent is None):
        rendered = _apply_separators(rendered, separators=_SPACED_SEPARATORS)
    if ensure_ascii:
        return _escape_non_ascii(rendered)
    return rendered


def loads(data: str | bytes | bytearray | memoryview) -> Any:
    """Deserialize JSON from string/bytes payloads."""
    return orjson.loads(data)
