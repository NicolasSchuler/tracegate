"""Tests for llmdebug._warnings module."""

from __future__ import annotations

import sys

import pytest

from llmdebug._warnings import emit_warning


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
