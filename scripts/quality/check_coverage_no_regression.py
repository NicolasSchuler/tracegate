#!/usr/bin/env python3
"""Fail when line or branch coverage regresses below the repository baseline."""

from __future__ import annotations

import argparse
import math
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


def _extract_count(
    totals: dict[str, Any],
    *,
    key: str,
    metric: str,
    guidance: str,
) -> int:
    value = totals.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise RuntimeError(
            f"coverage report totals field '{key}' must be an integer for {metric} coverage. "
            f"{guidance}"
        )
    return value


def _extract_metric_percent(
    totals: dict[str, Any],
    *,
    total_key: str,
    covered_key: str,
    metric: str,
    guidance: str,
) -> float:
    total = _extract_count(
        totals,
        key=total_key,
        metric=metric,
        guidance=guidance,
    )
    covered = _extract_count(
        totals,
        key=covered_key,
        metric=metric,
        guidance=guidance,
    )
    if total <= 0:
        raise RuntimeError(
            f"coverage report totals field '{total_key}' must be greater than zero. {guidance}"
        )
    if covered < 0 or covered > total:
        raise RuntimeError(
            f"coverage report has inconsistent {metric} counts (covered={covered}, total={total})."
        )
    return covered / total * 100.0


def _extract_coverage_percentages(coverage_payload: dict[str, Any]) -> dict[str, float]:
    totals_raw = coverage_payload.get("totals")
    if totals_raw is None:
        raise RuntimeError("coverage report is missing the 'totals' object")
    if not isinstance(totals_raw, dict):
        raise RuntimeError("coverage report field 'totals' must be a JSON object")

    return {
        "line": _extract_metric_percent(
            totals_raw,
            total_key="num_statements",
            covered_key="covered_lines",
            metric="line",
            guidance="Run pytest with '--cov-report=json'.",
        ),
        "branch": _extract_metric_percent(
            totals_raw,
            total_key="num_branches",
            covered_key="covered_branches",
            metric="branch",
            guidance="Run pytest with '--cov-branch --cov-report=json'.",
        ),
    }


def _extract_baseline_percentages(baseline_payload: dict[str, Any]) -> dict[str, float]:
    percentages: dict[str, float] = {}
    for metric in ("line", "branch"):
        key = f"{metric}_percent"
        value = baseline_payload.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RuntimeError(f"coverage baseline field '{key}' must be a number")
        try:
            percentage = float(value)
        except OverflowError as exc:
            raise RuntimeError(f"coverage baseline field '{key}' must be finite") from exc
        if not math.isfinite(percentage) or not 0.0 <= percentage <= 100.0:
            raise RuntimeError(
                f"coverage baseline field '{key}' must be a finite number between 0 and 100"
            )
        percentages[metric] = percentage
    return percentages


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail when line or branch coverage regresses below the configured baseline."
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
        if not math.isfinite(args.tolerance) or args.tolerance < 0.0:
            raise RuntimeError("tolerance must be a finite, non-negative number")
        coverage_payload = _load_json(args.coverage_json, label="coverage report")
        baseline_payload = _load_json(args.baseline_json, label="coverage baseline")
        current_percentages = _extract_coverage_percentages(coverage_payload)
        baseline_percentages = _extract_baseline_percentages(baseline_payload)
    except RuntimeError as exc:
        print(f"coverage no-regression: ERROR: {exc}")
        return 2

    regressions: list[str] = []
    for metric in ("line", "branch"):
        current_percent = current_percentages[metric]
        baseline_percent = baseline_percentages[metric]
        delta = current_percent - baseline_percent
        print(
            "coverage no-regression: "
            f"{metric} current={current_percent:.4f}% "
            f"baseline={baseline_percent:.4f}% "
            f"delta={delta:+.4f}pp "
            f"tolerance={args.tolerance:.4f}pp"
        )
        if delta + args.tolerance < 0.0:
            regressions.append(metric)

    if regressions:
        for metric in regressions:
            print(f"coverage no-regression: FAIL: {metric} coverage regressed below baseline.")
        return 1

    print("coverage no-regression: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
