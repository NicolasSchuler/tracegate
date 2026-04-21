"""Tests for _build_prompt, _load_snapshot_text, and _stringify_field."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evals.swebench.generate import _build_prompt, _load_snapshot_text, _stringify_field
from evals.swebench.types import CONDITION_BASELINE, CONDITION_LLMDEBUG


def _make_row(**overrides: Any) -> dict[str, Any]:
    """Return a standard dataset row, optionally overriding fields."""
    row: dict[str, Any] = {
        "instance_id": "test-123",
        "repo": "demo/repo",
        "base_commit": "abc123",
        "problem_statement": "Fix the bug.",
        "hints_text": "",
        "FAIL_TO_PASS": None,
        "PASS_TO_PASS": None,
    }
    row.update(overrides)
    return row


# -- _build_prompt tests ------------------------------------------------------


def test_build_prompt_baseline_contains_problem_statement() -> None:
    row = _make_row()
    prompt = _build_prompt(row=row, condition=CONDITION_BASELINE, snapshot_dir=None)

    assert "demo/repo" in prompt
    assert "test-123" in prompt
    assert "abc123" in prompt
    assert "Fix the bug." in prompt


def test_build_prompt_baseline_excludes_snapshot(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshots"
    snapshot_dir.mkdir()
    (snapshot_dir / "test-123.txt").write_text("snapshot content", encoding="utf-8")

    row = _make_row()
    prompt = _build_prompt(row=row, condition=CONDITION_BASELINE, snapshot_dir=snapshot_dir)

    assert "llmdebug runtime evidence" not in prompt


def test_build_prompt_llmdebug_includes_snapshot(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshots"
    snapshot_dir.mkdir()
    (snapshot_dir / "test-123.txt").write_text("runtime trace output", encoding="utf-8")

    row = _make_row()
    prompt = _build_prompt(row=row, condition=CONDITION_LLMDEBUG, snapshot_dir=snapshot_dir)

    assert "llmdebug runtime evidence" in prompt
    assert "runtime trace output" in prompt


def test_build_prompt_llmdebug_no_snapshot_dir() -> None:
    row = _make_row()
    prompt = _build_prompt(row=row, condition=CONDITION_LLMDEBUG, snapshot_dir=None)

    assert "llmdebug runtime evidence" in prompt
    assert "No snapshot directory was provided for this run." in prompt


def test_build_prompt_includes_hints_when_present() -> None:
    row = _make_row(hints_text="Look at module X.")
    prompt = _build_prompt(row=row, condition=CONDITION_BASELINE, snapshot_dir=None)

    assert "Hints:" in prompt
    assert "Look at module X." in prompt


def test_build_prompt_omits_hints_when_empty() -> None:
    row = _make_row(hints_text="")
    prompt = _build_prompt(row=row, condition=CONDITION_BASELINE, snapshot_dir=None)

    assert "Hints:" not in prompt


def test_build_prompt_includes_fail_to_pass_tests() -> None:
    row = _make_row(FAIL_TO_PASS=["test_foo.py::test_bar", "test_foo.py::test_baz"])
    prompt = _build_prompt(row=row, condition=CONDITION_BASELINE, snapshot_dir=None)

    assert "Fail-to-pass tests:" in prompt
    assert "test_foo.py::test_bar" in prompt
    assert "test_foo.py::test_baz" in prompt


# -- _load_snapshot_text tests ------------------------------------------------


def test_load_snapshot_text_returns_fallback_when_no_dir() -> None:
    result = _load_snapshot_text(snapshot_dir=None, instance_id="test-123")

    assert result == "No snapshot directory was provided for this run."


def test_load_snapshot_text_reads_txt_file(tmp_path: Path) -> None:
    (tmp_path / "test-123.txt").write_text("  snapshot payload  ", encoding="utf-8")

    result = _load_snapshot_text(snapshot_dir=tmp_path, instance_id="test-123")

    assert result == "snapshot payload"


def test_load_snapshot_text_prefers_txt_over_md(tmp_path: Path) -> None:
    (tmp_path / "test-123.txt").write_text("from txt", encoding="utf-8")
    (tmp_path / "test-123.md").write_text("from md", encoding="utf-8")

    result = _load_snapshot_text(snapshot_dir=tmp_path, instance_id="test-123")

    assert result == "from txt"


def test_load_snapshot_text_falls_back_to_md(tmp_path: Path) -> None:
    (tmp_path / "test-123.md").write_text("markdown content", encoding="utf-8")

    result = _load_snapshot_text(snapshot_dir=tmp_path, instance_id="test-123")

    assert result == "markdown content"


def test_load_snapshot_text_returns_fallback_when_no_file(tmp_path: Path) -> None:
    result = _load_snapshot_text(snapshot_dir=tmp_path, instance_id="test-123")

    assert result == "No instance-specific snapshot artifact was found."


# -- _stringify_field tests ---------------------------------------------------


def test_stringify_field_none() -> None:
    assert _stringify_field(None) == ""


def test_stringify_field_string() -> None:
    assert _stringify_field("  hello  ") == "hello"


def test_stringify_field_list() -> None:
    assert _stringify_field(["a", "b", "c"]) == "a\nb\nc"


def test_stringify_field_dict() -> None:
    result = _stringify_field({"k": "v"})
    parsed = json.loads(result)
    assert parsed == {"k": "v"}
    # Verify indented formatting (not compact single-line)
    assert "\n" in result


def test_stringify_field_number() -> None:
    assert _stringify_field(42) == "42"
