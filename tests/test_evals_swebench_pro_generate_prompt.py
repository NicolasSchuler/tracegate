"""Tests for _build_prompt_pro, load_snapshot_text, and _stringify_field."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from evals.swebench.prompt_policy import PromptInputs, load_snapshot_text
from evals.swebench.types import CONDITION_BASELINE, CONDITION_LLMDEBUG
from evals.swebench_pro.generate import _build_prompt_pro, _stringify_field


def _make_row(**overrides: Any) -> dict[str, Any]:
    """Return a standard Pro dataset row, optionally overriding fields."""
    row: dict[str, Any] = {
        "instance_id": "pro-test-123",
        "repo": "demo/pro-repo",
        "base_commit": "abc123",
        "problem_statement": "Fix the Pro bug.",
        "repo_language": "Python",
        "requirements": "",
        "interface": "",
        "fail_to_pass": None,
        "pass_to_pass": None,
    }
    row.update(overrides)
    return row


# -- _build_prompt_pro tests --------------------------------------------------


def test_build_prompt_pro_baseline_contains_problem_statement() -> None:
    row = _make_row()
    prompt = _build_prompt_pro(
        row=row, condition=CONDITION_BASELINE, prompt_inputs=PromptInputs(snapshot_dir=None)
    )

    assert "demo/pro-repo" in prompt
    assert "pro-test-123" in prompt
    assert "abc123" in prompt
    assert "Fix the Pro bug." in prompt


def test_build_prompt_pro_includes_repo_language() -> None:
    row = _make_row(repo_language="Rust")
    prompt = _build_prompt_pro(
        row=row, condition=CONDITION_BASELINE, prompt_inputs=PromptInputs(snapshot_dir=None)
    )

    assert "Language: Rust" in prompt


def test_build_prompt_pro_omits_language_when_empty() -> None:
    row = _make_row(repo_language="")
    prompt = _build_prompt_pro(
        row=row, condition=CONDITION_BASELINE, prompt_inputs=PromptInputs(snapshot_dir=None)
    )

    assert "Language:" not in prompt


def test_build_prompt_pro_includes_requirements_when_present() -> None:
    row = _make_row(requirements="numpy>=1.0\npandas>=2.0")
    prompt = _build_prompt_pro(
        row=row, condition=CONDITION_BASELINE, prompt_inputs=PromptInputs(snapshot_dir=None)
    )

    assert "Requirements:" in prompt
    assert "numpy>=1.0" in prompt
    assert "pandas>=2.0" in prompt


def test_build_prompt_pro_omits_requirements_when_empty() -> None:
    row = _make_row(requirements="")
    prompt = _build_prompt_pro(
        row=row, condition=CONDITION_BASELINE, prompt_inputs=PromptInputs(snapshot_dir=None)
    )

    assert "Requirements:" not in prompt


def test_build_prompt_pro_includes_interface_when_present() -> None:
    row = _make_row(interface="def solve(x: int) -> int: ...")
    prompt = _build_prompt_pro(
        row=row, condition=CONDITION_BASELINE, prompt_inputs=PromptInputs(snapshot_dir=None)
    )

    assert "Interface:" in prompt
    assert "def solve(x: int) -> int: ..." in prompt


def test_build_prompt_pro_omits_interface_when_empty() -> None:
    row = _make_row(interface="")
    prompt = _build_prompt_pro(
        row=row, condition=CONDITION_BASELINE, prompt_inputs=PromptInputs(snapshot_dir=None)
    )

    assert "Interface:" not in prompt


def test_build_prompt_pro_uses_lowercase_fail_to_pass() -> None:
    """Pro uses lowercase field names (fail_to_pass), not uppercase (FAIL_TO_PASS)."""
    row = _make_row(
        fail_to_pass=["test_foo.py::test_bar", "test_foo.py::test_baz"],
        # Set FAIL_TO_PASS to something different to prove Pro ignores it
        FAIL_TO_PASS=["SHOULD_NOT_APPEAR"],
    )
    prompt = _build_prompt_pro(
        row=row, condition=CONDITION_BASELINE, prompt_inputs=PromptInputs(include_test_ids=True)
    )

    assert "Fail-to-pass tests:" in prompt
    assert "test_foo.py::test_bar" in prompt
    assert "test_foo.py::test_baz" in prompt
    assert "SHOULD_NOT_APPEAR" not in prompt


def test_build_prompt_pro_uses_lowercase_pass_to_pass() -> None:
    """Pro uses lowercase field names (pass_to_pass), not uppercase (PASS_TO_PASS)."""
    row = _make_row(
        pass_to_pass=["test_stable.py::test_ok"],
        PASS_TO_PASS=["SHOULD_NOT_APPEAR_EITHER"],
    )
    prompt = _build_prompt_pro(
        row=row, condition=CONDITION_BASELINE, prompt_inputs=PromptInputs(include_test_ids=True)
    )

    assert "Pass-to-pass tests:" in prompt
    assert "test_stable.py::test_ok" in prompt
    assert "SHOULD_NOT_APPEAR_EITHER" not in prompt


def test_build_prompt_pro_baseline_excludes_snapshot(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshots"
    snapshot_dir.mkdir()
    (snapshot_dir / "pro-test-123.txt").write_text("snapshot content", encoding="utf-8")

    row = _make_row()
    prompt = _build_prompt_pro(
        row=row, condition=CONDITION_BASELINE, prompt_inputs=PromptInputs(snapshot_dir=snapshot_dir)
    )

    assert "llmdebug runtime evidence" not in prompt


def test_build_prompt_pro_llmdebug_includes_snapshot(tmp_path: Path) -> None:
    snapshot_dir = tmp_path / "snapshots"
    snapshot_dir.mkdir()
    (snapshot_dir / "pro-test-123.txt").write_text("runtime trace output", encoding="utf-8")

    row = _make_row()
    prompt = _build_prompt_pro(
        row=row, condition=CONDITION_LLMDEBUG, prompt_inputs=PromptInputs(snapshot_dir=snapshot_dir)
    )

    assert "llmdebug runtime evidence" in prompt
    assert "runtime trace output" in prompt


def test_build_prompt_pro_llmdebug_no_snapshot_dir() -> None:
    row = _make_row()
    prompt = _build_prompt_pro(
        row=row, condition=CONDITION_LLMDEBUG, prompt_inputs=PromptInputs(snapshot_dir=None)
    )

    assert "llmdebug runtime evidence" in prompt
    assert "No instance-specific snapshot artifact was found." in prompt


def test_build_prompt_pro_includes_fail_to_pass_tests() -> None:
    row = _make_row(fail_to_pass=["test_pro.py::test_a", "test_pro.py::test_b"])
    prompt = _build_prompt_pro(
        row=row, condition=CONDITION_BASELINE, prompt_inputs=PromptInputs(include_test_ids=True)
    )

    assert "Fail-to-pass tests:" in prompt
    assert "test_pro.py::test_a" in prompt
    assert "test_pro.py::test_b" in prompt


def test_build_prompt_pro_omits_scorer_test_ids_by_default() -> None:  # DAT-01 / D2
    row = _make_row(
        fail_to_pass=["test_pro.py::test_a"],
        pass_to_pass=["test_pro.py::test_b"],
    )
    prompt = _build_prompt_pro(row=row, condition=CONDITION_BASELINE, prompt_inputs=PromptInputs())

    assert "Fail-to-pass tests:" not in prompt
    assert "Pass-to-pass tests:" not in prompt
    assert "test_pro.py::test_a" not in prompt


def test_build_prompt_pro_omits_fail_to_pass_when_none() -> None:
    row = _make_row(fail_to_pass=None)
    prompt = _build_prompt_pro(
        row=row, condition=CONDITION_BASELINE, prompt_inputs=PromptInputs(snapshot_dir=None)
    )

    assert "Fail-to-pass tests:" not in prompt


def test_build_prompt_pro_all_extra_sections() -> None:
    """Verify requirements, interface, tests, and snapshot all co-exist."""
    row = _make_row(
        requirements="flask>=2.0",
        interface="class Handler: ...",
        fail_to_pass=["test_x.py::test_fail"],
        pass_to_pass=["test_x.py::test_pass"],
    )
    snapshot_dir_path = None  # No snapshot for baseline
    prompt = _build_prompt_pro(
        row=row,
        condition=CONDITION_BASELINE,
        prompt_inputs=PromptInputs(snapshot_dir=snapshot_dir_path, include_test_ids=True),
    )

    assert "Requirements:" in prompt
    assert "flask>=2.0" in prompt
    assert "Interface:" in prompt
    assert "class Handler: ..." in prompt
    assert "Fail-to-pass tests:" in prompt
    assert "Pass-to-pass tests:" in prompt


# -- _load_snapshot_text tests ------------------------------------------------


def test_load_snapshot_text_returns_none_when_no_dir() -> None:
    assert load_snapshot_text(snapshot_dir=None, instance_id="pro-test-123") is None


def test_load_snapshot_text_reads_txt_file(tmp_path: Path) -> None:
    (tmp_path / "pro-test-123.txt").write_text("  snapshot payload  ", encoding="utf-8")

    result = load_snapshot_text(snapshot_dir=tmp_path, instance_id="pro-test-123")

    assert result == "snapshot payload"


def test_load_snapshot_text_prefers_txt_over_md(tmp_path: Path) -> None:
    (tmp_path / "pro-test-123.txt").write_text("from txt", encoding="utf-8")
    (tmp_path / "pro-test-123.md").write_text("from md", encoding="utf-8")

    result = load_snapshot_text(snapshot_dir=tmp_path, instance_id="pro-test-123")

    assert result == "from txt"


def test_load_snapshot_text_falls_back_to_md(tmp_path: Path) -> None:
    (tmp_path / "pro-test-123.md").write_text("markdown content", encoding="utf-8")

    result = load_snapshot_text(snapshot_dir=tmp_path, instance_id="pro-test-123")

    assert result == "markdown content"


def test_load_snapshot_text_falls_back_to_json(tmp_path: Path) -> None:
    (tmp_path / "pro-test-123.json").write_text('{"trace": "data"}', encoding="utf-8")

    result = load_snapshot_text(snapshot_dir=tmp_path, instance_id="pro-test-123")

    assert result == '{"trace": "data"}'


def test_load_snapshot_text_returns_none_when_no_file(tmp_path: Path) -> None:
    assert load_snapshot_text(snapshot_dir=tmp_path, instance_id="pro-test-123") is None


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
