"""Behavioral regressions for the line-and-branch coverage ratchet."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import pytest

from llmdebug import _json as json
from scripts.quality import check_coverage_no_regression as checker


def _coverage_payload(
    *,
    covered_lines: int = 9_000,
    num_statements: int = 10_000,
    covered_branches: int = 8_000,
    num_branches: int = 10_000,
) -> dict[str, Any]:
    return {
        "totals": {
            "covered_lines": covered_lines,
            "num_statements": num_statements,
            "covered_branches": covered_branches,
            "num_branches": num_branches,
        }
    }


def _run_checker(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    coverage_payload: object,
    baseline_payload: object | None = None,
    extra_args: tuple[str, ...] = (),
) -> int:
    coverage_path = tmp_path / "coverage.json"
    baseline_path = tmp_path / "baseline.json"
    coverage_path.write_text(json.dumps(coverage_payload), encoding="utf-8")
    baseline_path.write_text(
        json.dumps(
            baseline_payload
            if baseline_payload is not None
            else {"line_percent": 90.0, "branch_percent": 80.0}
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(checker.__file__),
            "--coverage-json",
            str(coverage_path),
            "--baseline-json",
            str(baseline_path),
            *extra_args,
        ],
    )
    return checker.main()


def test_line_only_regression_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A line regression must fail even when branch coverage is unchanged."""
    exit_code = _run_checker(
        tmp_path,
        monkeypatch,
        coverage_payload=_coverage_payload(covered_lines=8_990),
    )

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "FAIL: line coverage regressed below baseline" in output
    assert "FAIL: branch coverage regressed below baseline" not in output


def test_branch_only_regression_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """A branch regression must fail even when line coverage is unchanged."""
    exit_code = _run_checker(
        tmp_path,
        monkeypatch,
        coverage_payload=_coverage_payload(covered_branches=7_990),
    )

    output = capsys.readouterr().out
    assert exit_code == 1
    assert "FAIL: branch coverage regressed below baseline" in output
    assert "FAIL: line coverage regressed below baseline" not in output


@pytest.mark.parametrize(
    ("covered_lines", "covered_branches"),
    [
        pytest.param(8_995, 7_995, id="exactly-at-tolerance"),
        pytest.param(9_100, 8_100, id="both-metrics-improve"),
    ],
)
def test_tolerance_and_improvements_pass(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    covered_lines: int,
    covered_branches: int,
) -> None:
    """The 0.05 pp allowance and genuine improvements must pass."""
    exit_code = _run_checker(
        tmp_path,
        monkeypatch,
        coverage_payload=_coverage_payload(
            covered_lines=covered_lines,
            covered_branches=covered_branches,
        ),
    )

    output = capsys.readouterr().out
    assert exit_code == 0
    assert "line current=" in output
    assert "branch current=" in output
    assert "coverage no-regression: PASS" in output


@pytest.mark.parametrize(
    ("coverage_payload", "message"),
    [
        pytest.param({}, "missing the 'totals' object", id="missing-totals"),
        pytest.param(
            {"totals": []},
            "field 'totals' must be a JSON object",
            id="malformed-totals",
        ),
        pytest.param(
            _coverage_payload() | {"totals": {"covered_lines": 9_000}},
            "field 'num_statements' must be an integer",
            id="missing-line-total",
        ),
        pytest.param(
            _coverage_payload(covered_lines=10_001),
            "inconsistent line counts",
            id="inconsistent-line-counts",
        ),
        pytest.param(
            _coverage_payload(num_branches=0, covered_branches=0),
            "field 'num_branches' must be greater than zero",
            id="zero-branch-total",
        ),
    ],
)
def test_malformed_or_missing_totals_return_clear_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    coverage_payload: object,
    message: str,
) -> None:
    """Malformed pytest-cov totals must return an actionable input error."""
    exit_code = _run_checker(
        tmp_path,
        monkeypatch,
        coverage_payload=coverage_payload,
    )

    output = capsys.readouterr().out
    assert exit_code == 2
    assert "coverage no-regression: ERROR:" in output
    assert message in output


@pytest.mark.parametrize(
    ("baseline_payload", "message"),
    [
        pytest.param(
            {"branch_percent": 80.0},
            "field 'line_percent' must be a number",
            id="missing-line-baseline",
        ),
        pytest.param(
            {"line_percent": 90.0, "branch_percent": "80"},
            "field 'branch_percent' must be a number",
            id="malformed-branch-baseline",
        ),
        pytest.param(
            {"line_percent": 101.0, "branch_percent": 80.0},
            "field 'line_percent' must be a finite number between 0 and 100",
            id="out-of-range-line-baseline",
        ),
    ],
)
def test_malformed_baselines_return_clear_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    baseline_payload: object,
    message: str,
) -> None:
    """Both committed baseline percentages must be finite numeric percentages."""
    exit_code = _run_checker(
        tmp_path,
        monkeypatch,
        coverage_payload=_coverage_payload(),
        baseline_payload=baseline_payload,
    )

    output = capsys.readouterr().out
    assert exit_code == 2
    assert "coverage no-regression: ERROR:" in output
    assert message in output


@pytest.mark.parametrize("tolerance", ["-0.01", "nan"])
def test_invalid_tolerance_returns_clear_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tolerance: str,
) -> None:
    """A negative or non-finite tolerance must not disable the ratchet."""
    exit_code = _run_checker(
        tmp_path,
        monkeypatch,
        coverage_payload=_coverage_payload(),
        extra_args=("--tolerance", tolerance),
    )

    output = capsys.readouterr().out
    assert exit_code == 2
    assert "tolerance must be a finite, non-negative number" in output
