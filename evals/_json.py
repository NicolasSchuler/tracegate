"""JSON helpers for eval tooling, backed by llmdebug's orjson adapter."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from llmdebug import _json as _core_json

JSONDecodeError = _core_json.JSONDecodeError
JSONDecoder = _core_json.JSONDecoder


def dumps(
    value: Any,
    *,
    indent: int | None = None,
    ensure_ascii: bool = True,
    default: Callable[[Any], Any] | None = None,
    sort_keys: bool = False,
    separators: tuple[str, str] | None = None,
) -> str:
    return _core_json.dumps(
        value,
        indent=indent,
        ensure_ascii=ensure_ascii,
        default=default,
        sort_keys=sort_keys,
        separators=separators,
    )


def loads(data: str | bytes | bytearray | memoryview) -> Any:
    return _core_json.loads(data)
