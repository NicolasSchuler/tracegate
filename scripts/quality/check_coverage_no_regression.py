#!/usr/bin/env python3
"""Fail when branch coverage regresses below the repository baseline."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from llmdebug import _json as json

DEFAULT_COVERAGE_JSON = Path("coverage.json")
DEFAULT_BASELINE_JSON = Path("quality/coverage_baseline.json")
FLOAT_TOLERANCE = 0.05


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"{label} not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{label} is not valid JSON: {path} ({exc})") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} must contain a JSON object: {path}")
    return payload


def _extract_branch_percent(coverage_payload: dict[str, Any]) -> float:
    totals_raw = coverage_payload.get("totals")
    if not isinstance(totals_raw, dict):
        raise RuntimeError("coverage.json is missing the 'totals' object")
    totals = totals_raw

    num_branches = totals.get("num_branches")
    covered_branches = totals.get("covered_branches")
    if not isinstance(num_branches, int) or not isinstance(covered_branches, int):
        raise RuntimeError(
            "coverage.json is missing branch metrics. "
            "Run pytest with '--cov-branch --cov-report=json'."
        )
    if num_branches <= 0:
        raise RuntimeError(
            "coverage.json reports zero executable branches. "
            "Run pytest with '--cov-branch'."
        )
    if covered_branches < 0 or covered_branches > num_branches:
        raise RuntimeError(
            "coverage.json has inconsistent branch counts "
            f"(covered={covered_branches}, total={num_branches})."
        )

    return 100.0 * float(covered_branches) / float(num_branches)


def _extract_baseline_percent(baseline_payload: dict[str, Any]) -> float:
    value = baseline_payload.get("branch_percent")
    if isinstance(value, (int, float)):
        return float(value)
    raise RuntimeError(
        "coverage baseline is missing numeric 'branch_percent' in quality/coverage_baseline.json"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail when branch coverage regresses below the configured baseline."
    )
    parser.add_argument(
        "--coverage-json",
        type=Path,
        default=DEFAULT_COVERAGE_JSON,
        help="Path to pytest-cov JSON report (default: coverage.json).",
    )
    parser.add_argument(
        "--baseline-json",
        type=Path,
        default=DEFAULT_BASELINE_JSON,
        help="Path to baseline JSON file (default: quality/coverage_baseline.json).",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=FLOAT_TOLERANCE,
        help="Allowed regression margin in percentage points (default: 0.05).",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    try:
        coverage_payload = _load_json(args.coverage_json, label="coverage report")
        baseline_payload = _load_json(args.baseline_json, label="coverage baseline")
        current_branch_percent = _extract_branch_percent(coverage_payload)
        baseline_branch_percent = _extract_baseline_percent(baseline_payload)
    except RuntimeError as exc:
        print(f"coverage no-regression: ERROR: {exc}")
        return 2

    delta = current_branch_percent - baseline_branch_percent
    print(
        "coverage no-regression: "
        f"branch current={current_branch_percent:.4f}% "
        f"baseline={baseline_branch_percent:.4f}% "
        f"delta={delta:+.4f}pp "
        f"tolerance={args.tolerance:.4f}pp"
    )

    if delta + args.tolerance < 0.0:
        print("coverage no-regression: FAIL: branch coverage regressed below baseline.")
        return 1

    print("coverage no-regression: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
