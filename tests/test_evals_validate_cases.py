"""Tests for eval case validation (expected exceptions, file structure, metadata)."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from evals import validate_cases
from evals.eval_types import CaseSpec, RunResult
from evals.validate_cases import (
    check_pytest_available,
    detect_environment_error,
    expected_exception_in_output,
    extract_exception_candidates,
    validate_one_case,
)
from evals.validate_cases import main as validate_main
from llmdebug import _json as json

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_case_dir(tmp_path: Path, case_id: str = "idx_case") -> Path:
    """Create a minimal valid case directory and return its path."""
    case_dir = tmp_path / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "buggy.py").write_text(
        "def bad_index(values):\n    return values[2]\n",
        encoding="utf-8",
    )
    (case_dir / "test_buggy.py").write_text(
        "from buggy import bad_index\n\ndef test_bad_index():\n    assert bad_index([7, 8]) == 8\n",
        encoding="utf-8",
    )
    (case_dir / "case.json").write_text(
        json.dumps(
            {
                "description": "Mini validator fixture.",
                "category": "traceback_controls",
                "difficulty": "easy",
                "snapshot_dependency": "none",
                "expected_exception": "IndexError",
                "tags": ["control", "index"],
                "python_args": ["-m", "pytest", "-q"],
                "timeout_sec": 30,
            }
        ),
        encoding="utf-8",
    )
    (case_dir / "oracle.patch").write_text(
        "--- a/buggy.py\n"
        "+++ b/buggy.py\n"
        "@@ -1,2 +1,2 @@\n"
        "-def bad_index(values):\n"
        "+def bad_index(values):\n"
        "-    return values[2]\n"
        "+    return values[1]\n",
        encoding="utf-8",
    )
    return case_dir


def _make_case_spec(
    tmp_path: Path,
    *,
    case_id: str = "c1",
    category: str = "traceback_controls",
    difficulty: str = "easy",
    snapshot_dependency: str = "none",
    expected_exception: str = "IndexError",
    bug_source: str | None = None,
) -> CaseSpec:
    """Build a CaseSpec pointing at a valid case directory."""
    case_dir = _make_case_dir(tmp_path, case_id)
    return CaseSpec(
        case_id=case_id,
        path=case_dir,
        category=category,
        difficulty=difficulty,
        snapshot_dependency=snapshot_dependency,
        expected_exception=expected_exception,
        tags=["control"],
        python_args=["-m", "pytest", "-q"],
        timeout_sec=30,
        bug_source=bug_source,
    )


# ---------------------------------------------------------------------------
# Original tests (preserved)
# ---------------------------------------------------------------------------


def test_validate_cases_succeeds_for_synthetic_case(tmp_path: Path) -> None:
    """Schema-only validation succeeds for a well-formed synthetic case."""
    _make_case_dir(tmp_path, "idx_case")
    exit_code = validate_main(["--cases-dir", str(tmp_path), "--schema-only"])
    assert exit_code == 0


def test_validate_cases_schema_only_detects_missing_required_file(
    tmp_path: Path,
) -> None:
    """Schema-only validation fails when oracle.patch is missing."""
    case_dir = tmp_path / "bad_case"
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "buggy.py").write_text("def f():\n    return 1\n", encoding="utf-8")
    (case_dir / "test_buggy.py").write_text("def test_f():\n    assert True\n", encoding="utf-8")
    (case_dir / "case.json").write_text(
        json.dumps(
            {
                "description": "Missing oracle patch should fail schema check.",
                "category": "traceback_controls",
                "difficulty": "easy",
                "snapshot_dependency": "none",
                "expected_exception": "AssertionError",
                "tags": [],
            }
        ),
        encoding="utf-8",
    )
    exit_code = validate_main(["--cases-dir", str(tmp_path), "--schema-only"])
    assert exit_code == 1


def test_validate_cases_schema_only_detects_missing_implementation_file(
    tmp_path: Path,
) -> None:
    """Schema-only validation fails when no implementation .py file exists."""
    case_dir = tmp_path / "bad_case_impl"
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "test_buggy.py").write_text("def test_f():\n    assert True\n", encoding="utf-8")
    (case_dir / "case.json").write_text(
        json.dumps(
            {
                "description": "Missing impl file.",
                "category": "traceback_controls",
                "difficulty": "easy",
                "snapshot_dependency": "none",
                "expected_exception": "AssertionError",
                "tags": [],
            }
        ),
        encoding="utf-8",
    )
    (case_dir / "oracle.patch").write_text(
        "--- a/buggy.py\n+++ b/buggy.py\n@@ -1 +1 @@\n-def f():\n+def f():\n",
        encoding="utf-8",
    )
    exit_code = validate_main(["--cases-dir", str(tmp_path), "--schema-only"])
    assert exit_code == 1


def test_expected_exception_matching_handles_qualified_names() -> None:
    """Qualified and simple exception names are both matched."""
    output = (
        "E       asyncio.exceptions.InvalidStateError: Result is not set.\n"
        "FAILED test_buggy.py::test_cancelled_task_is_handled - "
        "asyncio.exceptions.InvalidStateError: Result is not set.\n"
    )
    assert expected_exception_in_output("InvalidStateError", output)
    assert expected_exception_in_output("asyncio.exceptions.InvalidStateError", output)
    assert not expected_exception_in_output("CancelledError", output)


def test_validate_one_case_reports_timeout_cleanly(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """validate_one_case returns a clean timeout message on baseline timeout."""
    case_dir = _make_case_dir(tmp_path, "timeout_case")
    timed_out = RunResult(
        returncode=124,
        duration_sec=0.1,
        stdout="",
        stderr="pytest timed out after 1s",
        timed_out=True,
        timeout_sec=1,
    )
    monkeypatch.setattr(validate_cases, "run_pytest", lambda *_a, **_kw: timed_out)
    ok, message = validate_one_case(
        case_dir,
        expected_exception="AssertionError",
        python_args=["-m", "pytest", "-q"],
        timeout_sec=1,
        output_format="json_compact",
    )
    assert not ok
    assert message == "baseline run timed out after 1s"


# ---------------------------------------------------------------------------
# main() - argument validation and filtering (lines 81-117)
# ---------------------------------------------------------------------------


def test_main_returns_2_on_invalid_execution_args(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """main() returns 2 when build_execution_config_from_args raises ValueError."""
    monkeypatch.setattr(
        validate_cases,
        "build_execution_config_from_args",
        MagicMock(side_effect=ValueError("bad executor")),
    )
    exit_code = validate_main(["--cases-dir", str(tmp_path), "--schema-only"])
    assert exit_code == 2


def test_main_returns_2_on_jobs_less_than_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """main() returns 2 when --jobs is 0 or negative."""
    _make_case_dir(tmp_path)
    exit_code = validate_main(["--cases-dir", str(tmp_path), "--schema-only", "--jobs", "0"])
    assert exit_code == 2


def test_main_returns_2_on_load_cases_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """main() returns 2 when load_cases raises ValueError."""
    monkeypatch.setattr(
        validate_cases,
        "load_cases",
        MagicMock(side_effect=ValueError("corrupt json")),
    )
    exit_code = validate_main(["--cases-dir", "/absolute/nonexistent", "--schema-only"])
    assert exit_code == 2


def test_main_case_filter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--case filter selects only the named case."""
    specs = [
        _make_case_spec(tmp_path / "a", case_id="keep"),
        _make_case_spec(tmp_path / "b", case_id="drop"),
    ]
    monkeypatch.setattr(validate_cases, "load_cases", lambda _d: specs)
    exit_code = validate_main(
        [
            "--cases-dir",
            str(tmp_path),
            "--schema-only",
            "--case",
            "keep",
        ]
    )
    assert exit_code == 0


def test_main_category_filter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--category filter selects only matching categories."""
    specs = [
        _make_case_spec(tmp_path / "a", case_id="c1", category="wanted"),
        _make_case_spec(tmp_path / "b", case_id="c2", category="other"),
    ]
    monkeypatch.setattr(validate_cases, "load_cases", lambda _d: specs)
    exit_code = validate_main(
        [
            "--cases-dir",
            str(tmp_path),
            "--schema-only",
            "--category",
            "wanted",
        ]
    )
    assert exit_code == 0


def test_main_difficulty_filter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--difficulty filter selects only matching difficulty levels."""
    specs = [
        _make_case_spec(tmp_path / "a", case_id="c1", difficulty="hard"),
        _make_case_spec(tmp_path / "b", case_id="c2", difficulty="easy"),
    ]
    monkeypatch.setattr(validate_cases, "load_cases", lambda _d: specs)
    exit_code = validate_main(
        [
            "--cases-dir",
            str(tmp_path),
            "--schema-only",
            "--difficulty",
            "hard",
        ]
    )
    assert exit_code == 0


def test_main_snapshot_dependency_filter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--snapshot-dependency filter selects only matching values."""
    specs = [
        _make_case_spec(tmp_path / "a", case_id="c1", snapshot_dependency="required"),
        _make_case_spec(tmp_path / "b", case_id="c2", snapshot_dependency="none"),
    ]
    monkeypatch.setattr(validate_cases, "load_cases", lambda _d: specs)
    exit_code = validate_main(
        [
            "--cases-dir",
            str(tmp_path),
            "--schema-only",
            "--snapshot-dependency",
            "required",
        ]
    )
    assert exit_code == 0


def test_main_bug_source_filter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--bug-source filter selects cases by bug_source field."""
    specs = [
        _make_case_spec(tmp_path / "a", case_id="c1", bug_source="swebench"),
        _make_case_spec(tmp_path / "b", case_id="c2", bug_source=None),
    ]
    monkeypatch.setattr(validate_cases, "load_cases", lambda _d: specs)
    # bug_source=None defaults to "hand_crafted"
    exit_code = validate_main(
        [
            "--cases-dir",
            str(tmp_path),
            "--schema-only",
            "--bug-source",
            "hand_crafted",
        ]
    )
    assert exit_code == 0


def test_main_max_cases(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """--max-cases limits number of validated cases."""
    specs = [
        _make_case_spec(tmp_path / "a", case_id="c1"),
        _make_case_spec(tmp_path / "b", case_id="c2"),
        _make_case_spec(tmp_path / "c", case_id="c3"),
    ]
    monkeypatch.setattr(validate_cases, "load_cases", lambda _d: specs)
    exit_code = validate_main(
        [
            "--cases-dir",
            str(tmp_path),
            "--schema-only",
            "--max-cases",
            "2",
        ]
    )
    assert exit_code == 0


def test_main_returns_2_no_cases_selected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """main() returns 2 when no cases match the filters."""
    specs = [_make_case_spec(tmp_path / "a", case_id="c1")]
    monkeypatch.setattr(validate_cases, "load_cases", lambda _d: specs)
    exit_code = validate_main(
        [
            "--cases-dir",
            str(tmp_path),
            "--schema-only",
            "--case",
            "nonexistent",
        ]
    )
    assert exit_code == 2


# ---------------------------------------------------------------------------
# main() - environment checks (lines 119-126)
# ---------------------------------------------------------------------------


def test_main_relative_cases_dir_resolved(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A relative --cases-dir is resolved against _REPO_ROOT."""
    specs = [_make_case_spec(tmp_path / "a", case_id="c1")]
    monkeypatch.setattr(validate_cases, "_REPO_ROOT", tmp_path)
    monkeypatch.setattr(validate_cases, "load_cases", lambda _d: specs)
    exit_code = validate_main(["--cases-dir", "relative/path", "--schema-only"])
    # load_cases is called with tmp_path / "relative/path" (absolute)
    assert exit_code == 0


def test_main_env_check_local_pytest_available_proceeds(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """main() proceeds past env check when local pytest is available."""
    specs = [_make_case_spec(tmp_path / "a", case_id="c1")]
    monkeypatch.setattr(validate_cases, "load_cases", lambda _d: specs)
    monkeypatch.setattr(
        validate_cases,
        "check_pytest_available",
        lambda: (True, "ok"),
    )
    monkeypatch.setattr(
        validate_cases,
        "_validate_selected_case",
        lambda **_kw: (True, "ok"),
    )
    # Not schema-only, so env check runs; local executor succeeds
    exit_code = validate_main(["--cases-dir", str(tmp_path), "--executor", "local"])
    assert exit_code == 0


def test_main_env_check_local_pytest_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """main() returns 2 when check_pytest_available fails for local executor."""
    specs = [_make_case_spec(tmp_path / "a", case_id="c1")]
    monkeypatch.setattr(validate_cases, "load_cases", lambda _d: specs)
    monkeypatch.setattr(
        validate_cases,
        "check_pytest_available",
        lambda: (False, "pytest not found"),
    )
    # Not passing --schema-only so env check runs; local executor
    exit_code = validate_main(["--cases-dir", str(tmp_path), "--executor", "local"])
    assert exit_code == 2


def test_main_env_check_container_runtime_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """main() returns 2 when container runtime check fails."""
    specs = [_make_case_spec(tmp_path / "a", case_id="c1")]
    monkeypatch.setattr(validate_cases, "load_cases", lambda _d: specs)
    monkeypatch.setattr(
        validate_cases,
        "check_testcontainers_runtime_available",
        lambda: (False, "docker not found"),
    )
    exit_code = validate_main(["--cases-dir", str(tmp_path), "--executor", "testcontainers"])
    assert exit_code == 2


# ---------------------------------------------------------------------------
# main() - single-threaded exception handling (lines 140-142)
# ---------------------------------------------------------------------------


def test_main_single_threaded_internal_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Internal exception in _validate_selected_case is reported as FAIL."""
    specs = [_make_case_spec(tmp_path / "a", case_id="c1")]
    monkeypatch.setattr(validate_cases, "load_cases", lambda _d: specs)

    def _boom(**_kw: Any) -> tuple[bool, str]:
        raise RuntimeError("boom")

    monkeypatch.setattr(validate_cases, "_validate_selected_case", _boom)
    exit_code = validate_main(["--cases-dir", str(tmp_path), "--schema-only"])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "internal error: RuntimeError: boom" in captured.out


# ---------------------------------------------------------------------------
# main() - multi-threaded path (lines 150-173)
# ---------------------------------------------------------------------------


def test_main_multithreaded_pass(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Multi-threaded (--jobs 2) path runs and reports PASS for valid cases."""
    specs = [
        _make_case_spec(tmp_path / "a", case_id="c1"),
        _make_case_spec(tmp_path / "b", case_id="c2"),
    ]
    monkeypatch.setattr(validate_cases, "load_cases", lambda _d: specs)
    monkeypatch.setattr(
        validate_cases,
        "_validate_selected_case",
        lambda **_kw: (True, "ok"),
    )
    exit_code = validate_main(["--cases-dir", str(tmp_path), "--schema-only", "--jobs", "2"])
    assert exit_code == 0


def test_main_multithreaded_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Multi-threaded path reports FAIL for cases that fail validation."""
    specs = [_make_case_spec(tmp_path / "a", case_id="c1")]
    monkeypatch.setattr(validate_cases, "load_cases", lambda _d: specs)
    monkeypatch.setattr(
        validate_cases,
        "_validate_selected_case",
        lambda **_kw: (False, "schema bad"),
    )
    exit_code = validate_main(["--cases-dir", str(tmp_path), "--schema-only", "--jobs", "2"])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "[FAIL]" in captured.out


def test_main_multithreaded_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Multi-threaded path handles exceptions from futures gracefully."""
    specs = [_make_case_spec(tmp_path / "a", case_id="c1")]
    monkeypatch.setattr(validate_cases, "load_cases", lambda _d: specs)

    def _boom(**_kw: Any) -> tuple[bool, str]:
        raise RuntimeError("thread boom")

    monkeypatch.setattr(validate_cases, "_validate_selected_case", _boom)
    exit_code = validate_main(["--cases-dir", str(tmp_path), "--schema-only", "--jobs", "2"])
    assert exit_code == 1
    captured = capsys.readouterr()
    assert "internal error: RuntimeError: thread boom" in captured.out


# ---------------------------------------------------------------------------
# _validate_selected_case (line 197 - execution path)
# ---------------------------------------------------------------------------


def test_validate_selected_case_delegates_to_validate_one_case(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """_validate_selected_case calls validate_one_case when not schema_only."""
    spec = _make_case_spec(tmp_path / "a", case_id="c1")
    monkeypatch.setattr(
        validate_cases,
        "validate_one_case",
        lambda *_a, **_kw: (True, "baseline fails and oracle patch passes"),
    )
    ok, msg = validate_cases._validate_selected_case(
        case=spec, schema_only=False, output_format="json_compact"
    )
    assert ok
    assert "oracle patch passes" in msg


# ---------------------------------------------------------------------------
# check_pytest_available (lines 228-241)
# ---------------------------------------------------------------------------


def test_check_pytest_available_success(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """check_pytest_available returns True when subprocess succeeds."""
    fake_proc = MagicMock(returncode=0, stdout="pytest 8.0", stderr="")
    monkeypatch.setattr(validate_cases.subprocess, "run", lambda *_a, **_kw: fake_proc)
    ok, msg = check_pytest_available()
    assert ok
    assert msg == "ok"


def test_check_pytest_available_returncode_nonzero(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """check_pytest_available returns False on non-zero return code."""
    fake_proc = MagicMock(returncode=1, stdout="", stderr="No module named pytest")
    monkeypatch.setattr(validate_cases.subprocess, "run", lambda *_a, **_kw: fake_proc)
    ok, msg = check_pytest_available()
    assert not ok
    assert "not available" in msg


def test_check_pytest_available_exception(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """check_pytest_available handles subprocess exceptions gracefully."""

    def _raise(*_a: Any, **_kw: Any) -> None:
        raise OSError("no such file")

    monkeypatch.setattr(validate_cases.subprocess, "run", _raise)
    ok, msg = check_pytest_available()
    assert not ok
    assert "unable to execute pytest preflight" in msg


# ---------------------------------------------------------------------------
# extract_exception_candidates (line 256 - fallback branch)
# ---------------------------------------------------------------------------


def test_extract_exception_candidates_fallback() -> None:
    """Falls back to full text when no E-prefixed lines contain exceptions."""
    output = "FAILED - ValueError: bad value\nsome other line\n"
    candidates = extract_exception_candidates(output)
    assert "ValueError" in candidates


def test_extract_exception_candidates_e_lines() -> None:
    """Extracts from E-prefixed lines first."""
    output = "E   KeyError: 'missing'\nFAILED - ValueError: bad\n"
    candidates = extract_exception_candidates(output)
    assert "KeyError" in candidates
    # ValueError from non-E line should NOT appear (E-lines matched)
    assert "ValueError" not in candidates


# ---------------------------------------------------------------------------
# detect_environment_error (lines 276-284)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "marker",
    [
        "No module named pytest",
        "pytest: command not found",
        "No such file or directory: 'pytest'",
    ],
)
def test_detect_environment_error_finds_markers(marker: str) -> None:
    """Each known environment-error marker is detected."""
    result = detect_environment_error(f"some prefix {marker} some suffix")
    assert result == marker


def test_detect_environment_error_returns_none_for_clean_output() -> None:
    """Returns None when no environment-error markers are present."""
    assert detect_environment_error("all good, tests passed") is None


# ---------------------------------------------------------------------------
# validate_one_case - full execution paths (lines 302, 322-356)
# ---------------------------------------------------------------------------


def test_validate_one_case_missing_oracle_patch(tmp_path: Path) -> None:
    """Returns False when oracle.patch is absent from the case directory."""
    case_dir = tmp_path / "no_patch"
    case_dir.mkdir()
    ok, msg = validate_one_case(
        case_dir,
        expected_exception="IndexError",
        python_args=[],
        timeout_sec=30,
        output_format="json_compact",
    )
    assert not ok
    assert "missing oracle.patch" in msg


def test_validate_one_case_baseline_unexpectedly_passes(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Returns False when the baseline run passes (returncode 0)."""
    case_dir = _make_case_dir(tmp_path, "pass_case")
    passed = RunResult(returncode=0, duration_sec=0.1, stdout="1 passed", stderr="")
    monkeypatch.setattr(validate_cases, "run_pytest", lambda *_a, **_kw: passed)
    ok, msg = validate_one_case(
        case_dir,
        expected_exception="IndexError",
        python_args=[],
        timeout_sec=30,
        output_format="json_compact",
    )
    assert not ok
    assert "baseline unexpectedly passed" in msg


def test_validate_one_case_environment_error_in_baseline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Returns False when baseline output contains an environment error."""
    case_dir = _make_case_dir(tmp_path, "env_err")
    env_err = RunResult(
        returncode=1,
        duration_sec=0.1,
        stdout="",
        stderr="No module named pytest",
    )
    monkeypatch.setattr(validate_cases, "run_pytest", lambda *_a, **_kw: env_err)
    ok, msg = validate_one_case(
        case_dir,
        expected_exception="IndexError",
        python_args=[],
        timeout_sec=30,
        output_format="json_compact",
    )
    assert not ok
    assert "environment/setup error" in msg


def test_validate_one_case_exception_not_found_with_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reports observed candidates when expected exception is absent."""
    case_dir = _make_case_dir(tmp_path, "wrong_exc")
    wrong_exc = RunResult(
        returncode=1,
        duration_sec=0.1,
        stdout="E   KeyError: 'x'\n",
        stderr="",
    )
    monkeypatch.setattr(validate_cases, "run_pytest", lambda *_a, **_kw: wrong_exc)
    ok, msg = validate_one_case(
        case_dir,
        expected_exception="IndexError",
        python_args=[],
        timeout_sec=30,
        output_format="json_compact",
    )
    assert not ok
    assert "not found" in msg
    assert "KeyError" in msg


def test_validate_one_case_exception_not_found_no_candidates(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reports generic message when no exception candidates at all."""
    case_dir = _make_case_dir(tmp_path, "no_exc")
    no_exc = RunResult(
        returncode=1,
        duration_sec=0.1,
        stdout="some failure with no exception name\n",
        stderr="",
    )
    monkeypatch.setattr(validate_cases, "run_pytest", lambda *_a, **_kw: no_exc)
    ok, msg = validate_one_case(
        case_dir,
        expected_exception="IndexError",
        python_args=[],
        timeout_sec=30,
        output_format="json_compact",
    )
    assert not ok
    assert "not found in baseline output" in msg


def test_validate_one_case_oracle_patch_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Returns False when oracle patch cannot be applied."""
    case_dir = _make_case_dir(tmp_path, "bad_patch")
    baseline_fail = RunResult(
        returncode=1,
        duration_sec=0.1,
        stdout="E   IndexError: list index out of range\n",
        stderr="",
    )
    monkeypatch.setattr(validate_cases, "run_pytest", lambda *_a, **_kw: baseline_fail)
    monkeypatch.setattr(validate_cases, "apply_diff", lambda *_a, **_kw: (False, "hunk failed"))
    ok, msg = validate_one_case(
        case_dir,
        expected_exception="IndexError",
        python_args=[],
        timeout_sec=30,
        output_format="json_compact",
    )
    assert not ok
    assert "oracle patch failed" in msg


def test_validate_one_case_patched_run_times_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Returns False when the patched (post-oracle) run times out."""
    case_dir = _make_case_dir(tmp_path, "patch_timeout")
    baseline_fail = RunResult(
        returncode=1,
        duration_sec=0.1,
        stdout="E   IndexError: list index out of range\n",
        stderr="",
    )
    patched_timeout = RunResult(
        returncode=124,
        duration_sec=30.0,
        stdout="",
        stderr="timed out",
        timed_out=True,
        timeout_sec=30,
    )
    call_count = 0

    def _run_pytest(*_a: Any, **_kw: Any) -> RunResult:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return baseline_fail
        return patched_timeout

    monkeypatch.setattr(validate_cases, "run_pytest", _run_pytest)
    monkeypatch.setattr(validate_cases, "apply_diff", lambda *_a, **_kw: (True, "ok"))
    ok, msg = validate_one_case(
        case_dir,
        expected_exception="IndexError",
        python_args=[],
        timeout_sec=30,
        output_format="json_compact",
    )
    assert not ok
    assert "patched run timed out" in msg


def test_validate_one_case_patched_run_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Returns False when the patched run still fails."""
    case_dir = _make_case_dir(tmp_path, "patch_fail")
    baseline_fail = RunResult(
        returncode=1,
        duration_sec=0.1,
        stdout="E   IndexError: list index out of range\n",
        stderr="",
    )
    patched_fail = RunResult(
        returncode=1,
        duration_sec=0.1,
        stdout="still failing",
        stderr="",
    )
    call_count = 0

    def _run_pytest(*_a: Any, **_kw: Any) -> RunResult:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return baseline_fail
        return patched_fail

    monkeypatch.setattr(validate_cases, "run_pytest", _run_pytest)
    monkeypatch.setattr(validate_cases, "apply_diff", lambda *_a, **_kw: (True, "ok"))
    ok, msg = validate_one_case(
        case_dir,
        expected_exception="IndexError",
        python_args=[],
        timeout_sec=30,
        output_format="json_compact",
    )
    assert not ok
    assert "patched run did not pass" in msg


def test_validate_one_case_full_success(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Returns True when baseline fails correctly and oracle patch fixes it."""
    case_dir = _make_case_dir(tmp_path, "success")
    baseline_fail = RunResult(
        returncode=1,
        duration_sec=0.1,
        stdout="E   IndexError: list index out of range\n",
        stderr="",
    )
    patched_pass = RunResult(
        returncode=0,
        duration_sec=0.1,
        stdout="1 passed",
        stderr="",
    )
    call_count = 0

    def _run_pytest(*_a: Any, **_kw: Any) -> RunResult:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return baseline_fail
        return patched_pass

    monkeypatch.setattr(validate_cases, "run_pytest", _run_pytest)
    monkeypatch.setattr(validate_cases, "apply_diff", lambda *_a, **_kw: (True, "ok"))
    ok, msg = validate_one_case(
        case_dir,
        expected_exception="IndexError",
        python_args=[],
        timeout_sec=30,
        output_format="json_compact",
    )
    assert ok
    assert "baseline fails and oracle patch passes" in msg
