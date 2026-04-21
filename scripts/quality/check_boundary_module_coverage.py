#!/usr/bin/env python3
"""Enforce focused coverage thresholds for the baseline patch boundary seam."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from llmdebug import _json as json

DEFAULT_COVERAGE_JSON = Path(".pytest_cache/baseline-boundary-coverage.json")


@dataclass(frozen=True)
class ModuleCoverageThreshold:
    module: str
    min_line_percent: float
    min_branch_percent: float


MODULE_THRESHOLDS: tuple[ModuleCoverageThreshold, ...] = (
    ModuleCoverageThreshold(
        "baselines/adapter_base.py", min_line_percent=100.0, min_branch_percent=100.0
    ),
    ModuleCoverageThreshold(
        "evals/patchers/baseline_patcher.py",
        min_line_percent=100.0,
        min_branch_percent=100.0,
    ),
    ModuleCoverageThreshold(
        "evals/attempt_verification.py",
        min_line_percent=100.0,
        min_branch_percent=100.0,
    ),
    ModuleCoverageThreshold(
        "evals/eval_patches.py", min_line_percent=97.0, min_branch_percent=95.0
    ),
)


def _load_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"coverage report not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"coverage report is not valid JSON: {path} ({exc})") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"coverage report must contain a JSON object: {path}")
    return payload


def _module_summary(coverage_payload: dict[str, Any], module: str) -> dict[str, Any]:
    files = coverage_payload.get("files")
    if not isinstance(files, dict):
        raise RuntimeError("coverage report is missing the 'files' object")
    module_payload = files.get(module)
    if not isinstance(module_payload, dict):
        raise RuntimeError(f"coverage report is missing module entry: {module}")
    summary = module_payload.get("summary")
    if not isinstance(summary, dict):
        raise RuntimeError(f"coverage report is missing summary data for module: {module}")
    return summary


def _extract_percent(summary: dict[str, Any], *, key: str, label: str, module: str) -> float:
    value = summary.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    raise RuntimeError(f"coverage summary for {module} is missing numeric '{label}'")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail when focused baseline-boundary modules miss their coverage thresholds."
    )
    parser.add_argument(
        "--coverage-json",
        type=Path,
        default=DEFAULT_COVERAGE_JSON,
        help=(
            "Path to pytest-cov JSON report for the focused boundary suite "
            f"(default: {DEFAULT_COVERAGE_JSON})."
        ),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        coverage_payload = _load_json(args.coverage_json)
    except RuntimeError as exc:
        print(f"boundary coverage: ERROR: {exc}")
        return 2

    failures: list[str] = []
    for threshold in MODULE_THRESHOLDS:
        try:
            summary = _module_summary(coverage_payload, threshold.module)
            line_percent = _extract_percent(
                summary,
                key="percent_statements_covered",
                label="percent_statements_covered",
                module=threshold.module,
            )
            branch_percent = _extract_percent(
                summary,
                key="percent_branches_covered",
                label="percent_branches_covered",
                module=threshold.module,
            )
        except RuntimeError as exc:
            print(f"boundary coverage: ERROR: {exc}")
            return 2

        print(
            "boundary coverage: "
            f"{threshold.module} "
            f"line={line_percent:.2f}% (min {threshold.min_line_percent:.2f}%) "
            f"branch={branch_percent:.2f}% (min {threshold.min_branch_percent:.2f}%)"
        )
        if line_percent < threshold.min_line_percent:
            failures.append(
                f"{threshold.module}: line coverage {line_percent:.2f}% < {threshold.min_line_percent:.2f}%"
            )
        if branch_percent < threshold.min_branch_percent:
            failures.append(
                f"{threshold.module}: branch coverage {branch_percent:.2f}% < {threshold.min_branch_percent:.2f}%"
            )

    if failures:
        print("boundary coverage: FAIL")
        for failure in failures:
            print(f"  - {failure}")
        return 1

    print("boundary coverage: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
