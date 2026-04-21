"""Tests for small coercion helpers."""

from __future__ import annotations

import importlib
from types import ModuleType

import pytest


def _reload_coerce_module() -> ModuleType:
    import llmdebug._coerce as coerce_module

    return importlib.reload(coerce_module)


def test_coerce_int_handles_invalid_and_bounded_values() -> None:
    coerce = _reload_coerce_module()

    assert coerce.coerce_int("7", default=0) == 7
    assert coerce.coerce_int("bad", default=3) == 3
    assert coerce.coerce_int(-1, default=5, minimum=0) == 5
    assert coerce.coerce_int(11, default=5, maximum=10) == 5


def test_read_env_int_handles_missing_invalid_and_stripped_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    coerce = _reload_coerce_module()

    monkeypatch.delenv("LLMDEBUG_SAMPLE_INT", raising=False)
    assert coerce.read_env_int("LLMDEBUG_SAMPLE_INT", 9) == 9

    monkeypatch.setenv("LLMDEBUG_SAMPLE_INT", " 12 ")
    assert coerce.read_env_int("LLMDEBUG_SAMPLE_INT", 9, maximum=20) == 12

    monkeypatch.setenv("LLMDEBUG_SAMPLE_INT", "bad")
    assert coerce.read_env_int("LLMDEBUG_SAMPLE_INT", 9) == 9
