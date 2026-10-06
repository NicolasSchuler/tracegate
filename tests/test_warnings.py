"""Tests for llmdebug._warnings module."""

from __future__ import annotations

import logging
import sys

import pytest

from llmdebug._warnings import LOGGER_NAME, emit_warning
from llmdebug.capture import _warn


def test_emit_warning_writes_to_stderr(capsys: pytest.CaptureFixture[str]) -> None:
    emit_warning("something went wrong")
    captured = capsys.readouterr()
    assert captured.err == "llmdebug: warning: something went wrong\n"
    assert captured.out == ""


def test_emit_warning_format(capsys: pytest.CaptureFixture[str]) -> None:
    emit_warning("test message")
    captured = capsys.readouterr()
    assert captured.err.startswith("llmdebug: warning: ")
    assert "test message" in captured.err


def test_emit_warning_empty_message(capsys: pytest.CaptureFixture[str]) -> None:
    emit_warning("")
    captured = capsys.readouterr()
    assert captured.err == "llmdebug: warning: \n"


def test_emit_warning_special_characters(capsys: pytest.CaptureFixture[str]) -> None:
    emit_warning("path/to/file: unexpected \t tab and newline\nnext line")
    captured = capsys.readouterr()
    assert "path/to/file" in captured.err
    assert captured.err.startswith("llmdebug: warning: ")


def test_emit_warning_returns_none(capsys: pytest.CaptureFixture[str]) -> None:
    result = emit_warning("check return value")
    assert result is None


def test_warnings_go_through_the_diagnostics_logger(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """X-07: one configurable channel; raising its level silences stderr output."""
    diagnostics = logging.getLogger(LOGGER_NAME)
    previous = diagnostics.level
    diagnostics.setLevel(logging.ERROR)
    try:
        emit_warning("hidden")
    finally:
        diagnostics.setLevel(previous)
    assert capsys.readouterr().err == ""


def test_diagnostics_do_not_reach_the_root_logger(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """log_capture records root-logger output into snapshots; llmdebug's own must stay out."""
    seen: list[str] = []

    class _Collect(logging.Handler):
        def emit(self, record: logging.LogRecord) -> None:
            seen.append(record.getMessage())

    root_handler = _Collect()
    logging.getLogger().addHandler(root_handler)
    try:
        emit_warning("not for the root logger")
    finally:
        logging.getLogger().removeHandler(root_handler)
    assert seen == []
    assert capsys.readouterr().err == "llmdebug: warning: not for the root logger\n"


def test_capture_warnings_are_logged_at_debug(capsys: pytest.CaptureFixture[str]) -> None:
    diagnostics = logging.getLogger(LOGGER_NAME)
    previous = diagnostics.level
    buffer: list[str] = []
    diagnostics.setLevel(logging.DEBUG)
    try:
        _warn(buffer, "locals count failed")
    finally:
        diagnostics.setLevel(previous)
    assert buffer == ["locals count failed"]
    assert capsys.readouterr().err == "llmdebug: debug: capture: locals count failed\n"


def test_reimporting_the_module_keeps_one_stderr_handler(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A re-imported _warnings must not print every warning twice."""
    import importlib

    import llmdebug._warnings as warnings_module

    reloaded = importlib.reload(warnings_module)
    reloaded.emit_warning("once")
    assert capsys.readouterr().err == "llmdebug: warning: once\n"
