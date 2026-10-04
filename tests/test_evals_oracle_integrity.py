"""Tests for the test-surface restore and JUnit verdict helpers (ATT-01, D10)."""

from __future__ import annotations

from pathlib import Path

import pytest

from evals.oracle_integrity import (
    JUNIT_REPORT_NAME,
    PytestCounts,
    consume_junit_counts,
    is_test_surface_path,
    read_junit_counts,
    restore_test_surface,
    verified_success,
    with_junit_report,
)


@pytest.mark.parametrize(
    ("relative", "expected"),
    [
        ("test_buggy.py", True),
        ("pkg/test_more.py", True),
        ("checks_test.py", True),
        ("conftest.py", True),
        ("sub/conftest.py", True),
        ("pytest.ini", True),
        ("pyproject.toml", True),
        ("setup.cfg", True),
        ("tox.ini", True),
        ("buggy.py", False),
        ("testing_utils.py", False),
        ("test_data.json", False),
    ],
)
def test_is_test_surface_path(relative: str, expected: bool) -> None:
    assert is_test_surface_path(Path(relative)) is expected


def _write(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def test_restore_test_surface_reverts_edits_additions_and_deletions(tmp_path: Path) -> None:
    pristine = tmp_path / "case"
    work = tmp_path / "work"
    _write(pristine / "buggy.py", "x = 1\n")
    _write(pristine / "test_buggy.py", "def test_x():\n    assert False\n")
    _write(pristine / "pkg" / "test_pkg.py", "def test_pkg():\n    pass\n")
    _write(work / "buggy.py", "x = 2\n")
    _write(work / "test_buggy.py", "def test_x():\n    assert True\n")
    _write(work / "conftest.py", "def pytest_sessionfinish(session):\n    pass\n")
    _write(work / "__pycache__" / "test_cached.py", "stale\n")

    changed = restore_test_surface(workdir=work, pristine_dir=pristine)

    assert changed == ("conftest.py", "pkg/test_pkg.py", "test_buggy.py")
    assert (work / "test_buggy.py").read_text(encoding="utf-8") == (
        pristine / "test_buggy.py"
    ).read_text(encoding="utf-8")
    assert not (work / "conftest.py").exists()
    assert (work / "pkg" / "test_pkg.py").exists()
    # Source edits are the patch under test and stay in place.
    assert (work / "buggy.py").read_text(encoding="utf-8") == "x = 2\n"
    assert restore_test_surface(workdir=work, pristine_dir=pristine) == ()


_JUNIT = (
    '<?xml version="1.0" encoding="utf-8"?><testsuites>'
    '<testsuite name="pytest" errors="{errors}" failures="{failures}" skipped="{skipped}"'
    ' tests="{tests}" time="0.01"/></testsuites>'
)


def test_read_junit_counts_sums_suites(tmp_path: Path) -> None:
    report = tmp_path / "r.xml"
    report.write_text(
        '<testsuites><testsuite tests="2" failures="1" errors="0" skipped="0"/>'
        '<testsuite tests="3" failures="0" errors="1" skipped="1"/></testsuites>',
        encoding="utf-8",
    )
    assert read_junit_counts(report) == PytestCounts(tests=5, failures=1, errors=1, skipped=1)


@pytest.mark.parametrize(
    "payload",
    [
        "not xml",
        "<testsuites/>",
        '<testsuite tests="x"/>',
        '<!DOCTYPE lolz [<!ENTITY lol "lol">]><testsuite tests="1"/>',
    ],
)
def test_read_junit_counts_rejects_bad_reports(tmp_path: Path, payload: str) -> None:
    report = tmp_path / "r.xml"
    report.write_text(payload, encoding="utf-8")
    assert read_junit_counts(report) is None


def test_read_junit_counts_missing_file(tmp_path: Path) -> None:
    assert read_junit_counts(tmp_path / "absent.xml") is None


def test_consume_junit_counts_deletes_report(tmp_path: Path) -> None:
    (tmp_path / JUNIT_REPORT_NAME).write_text(
        _JUNIT.format(tests=1, failures=0, errors=0, skipped=0), encoding="utf-8"
    )
    assert consume_junit_counts(tmp_path) == PytestCounts(1, 0, 0, 0)
    assert not (tmp_path / JUNIT_REPORT_NAME).exists()
    assert consume_junit_counts(tmp_path) is None


@pytest.mark.parametrize(
    ("returncode", "counts", "expected"),
    [
        (0, PytestCounts(1, 0, 0, 0), True),
        (0, None, False),
        (1, PytestCounts(1, 0, 0, 0), False),
        (0, PytestCounts(1, 1, 0, 0), False),
        (0, PytestCounts(1, 0, 1, 0), False),
        (0, PytestCounts(0, 0, 0, 0), False),
        (0, PytestCounts(2, 0, 0, 2), False),
    ],
)
def test_verified_success(returncode: int, counts: PytestCounts | None, expected: bool) -> None:
    assert verified_success(returncode=returncode, counts=counts) is expected


def test_with_junit_report_appends_flag() -> None:
    assert with_junit_report(["-m", "pytest", "-q"]) == [
        "-m",
        "pytest",
        "-q",
        f"--junitxml={JUNIT_REPORT_NAME}",
    ]


def test_genuine_fix_is_success_via_junit_end_to_end(tmp_path: Path) -> None:
    """A real fix through the local executor is still a success (ATT-01 positive path)."""
    import shutil
    from dataclasses import dataclass

    from evals import run_eval
    from evals.eval_types import Condition
    from evals.patchers.base import PatchResult

    case_dir = tmp_path / "case_src"
    case_dir.mkdir()
    (case_dir / "buggy.py").write_text("def f() -> int:\n    return 1\n", encoding="utf-8")
    (case_dir / "test_buggy.py").write_text(
        "from buggy import f\n\n\ndef test_f() -> None:\n    assert f() == 2\n",
        encoding="utf-8",
    )
    case = run_eval.CaseSpec(
        case_id="oracle_fix",
        path=case_dir,
        category="logic",
        difficulty="easy",
        snapshot_dependency="helpful",
        expected_exception="AssertionError",
        tags=[],
        python_args=["-m", "pytest", "-q"],
        timeout_sec=60,
    )
    fix = (
        "--- a/buggy.py\n+++ b/buggy.py\n@@ -1,2 +1,2 @@\n"
        " def f() -> int:\n-    return 1\n+    return 2\n"
    )

    @dataclass
    class _Patcher:
        name: str = "fix"

        def propose_patch(self, *, prompt: str, context_dir: Path, cwd: Path) -> PatchResult:
            del prompt, context_dir, cwd
            return PatchResult(diff=fix, meta={})

    workdir = tmp_path / "work"
    shutil.copytree(case_dir, workdir)
    previous = run_eval.set_active_execution_config(run_eval.ExecutionConfig(executor="local"))
    try:
        records = run_eval.run_one_condition(
            case=case,
            condition=Condition.BASE_REPAIR,
            workdir=workdir,
            context_dir_base=tmp_path / "ctx",
            patcher=_Patcher(),
            k=1,
            output_format="json_compact",
            run_id="oracle-fix",
            include_files_context=True,
            force_patch_attempt=False,
        )
    finally:
        run_eval.set_active_execution_config(previous)

    record = next(r for r in records if int(r.get("attempt", 0)) == 1)
    assert record["success"] is True
    assert record["note"] == "success"
    assert record["patch_meta"]["test_surface_modified"] is False
    assert record["patch_meta"]["pytest_verdict_source"] == "junit"
    assert record["patch_meta"]["pytest_verdict_counts"] == {
        "tests": 1,
        "failures": 0,
        "errors": 0,
        "skipped": 0,
    }
    assert not (workdir / JUNIT_REPORT_NAME).exists()
