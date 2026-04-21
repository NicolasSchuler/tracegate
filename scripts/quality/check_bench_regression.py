#!/usr/bin/env python3
"""Fail when any capture-overhead benchmark regresses beyond a threshold.

Compares pytest-benchmark JSON output against a local baseline and exits
non-zero when any metric regresses by more than the allowed percentage
(default 20 %).

Usage
-----
    # Check for regressions (after `just bench-capture`):
    python scripts/quality/check_bench_regression.py

    # Custom paths / threshold:
    python scripts/quality/check_bench_regression.py \
        --current .benchmarks/capture-overhead-current.json \
        --baseline .benchmarks/capture-overhead-regression-baseline.json \
        --threshold 15

    # Overwrite baseline with current medians:
    python scripts/quality/check_bench_regression.py --update-baseline
"""

from __future__ import annotations

import argparse
import datetime
from pathlib import Path
from typing import Any

from llmdebug import _json as json

DEFAULT_CURRENT = Path(".benchmarks/capture-overhead-current.json")
DEFAULT_BASELINE = Path(".benchmarks/capture-overhead-regression-baseline.json")
DEFAULT_THRESHOLD_PCT = 20.0


# -------------------------------------------------------------------
# Helpers
# -------------------------------------------------------------------


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    """Read and parse a JSON file, raising *RuntimeError* on problems."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"{label} not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{label} is not valid JSON: {path} ({exc})") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} must contain a JSON object: {path}")
    return payload


def _extract_medians(pytest_bench: dict[str, Any]) -> dict[str, float]:
    """Return ``{test_name: median_seconds}`` from pytest-benchmark JSON."""
    rows = pytest_bench.get("benchmarks", [])
    if not isinstance(rows, list):
        return {}
    out: dict[str, float] = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        name = str(row.get("name", "")).strip()
        if not name:
            continue
        stats: Any = row.get("stats", {})
        if not isinstance(stats, dict):
            continue
        try:
            out[name] = float(stats["median"])
        except (KeyError, TypeError, ValueError):
            continue
    return out


def _extract_baseline(
    baseline: dict[str, Any],
) -> tuple[dict[str, float], float]:
    """Return ``({test_name: median}, threshold_pct)`` from the baseline file."""
    benchmarks = baseline.get("benchmarks", {})
    if not isinstance(benchmarks, dict):
        raise RuntimeError("baseline is missing 'benchmarks' mapping")
    threshold = float(baseline.get("threshold_pct", DEFAULT_THRESHOLD_PCT))
    medians: dict[str, float] = {}
    for name, entry in benchmarks.items():
        if isinstance(entry, dict):
            medians[name] = float(entry.get("median", 0.0))
        else:
            medians[name] = float(entry)
    return medians, threshold


# -------------------------------------------------------------------
# Update-baseline mode
# -------------------------------------------------------------------


def _update_baseline(
    current_path: Path,
    baseline_path: Path,
) -> int:
    """Overwrite *baseline_path* with medians from *current_path*."""
    current_data = _load_json(current_path, label="current benchmarks")
    medians = _extract_medians(current_data)
    if not medians:
        print("bench-regression: ERROR: no benchmarks found in current file")
        return 2

    payload: dict[str, Any] = {
        "description": (
            "Capture-overhead regression baselines (median seconds). "
            "Regenerate with: just bench-capture && "
            "python scripts/quality/check_bench_regression.py --update-baseline"
        ),
        "threshold_pct": DEFAULT_THRESHOLD_PCT,
        "benchmarks": {
            name: {"median": median, "unit": "seconds"} for name, median in sorted(medians.items())
        },
        "source": "local",
        "updated_at_utc": datetime.datetime.now(datetime.timezone.utc)
        .replace(microsecond=0)
        .isoformat(),
    }

    baseline_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"bench-regression: baseline updated → {baseline_path}")
    for name, median in sorted(medians.items()):
        print(f"  {name}: {median:.6e} s")
    return 0


# -------------------------------------------------------------------
# Regression check
# -------------------------------------------------------------------


def _check_regression(
    current_path: Path,
    baseline_path: Path,
    threshold_override: float | None,
) -> int:
    """Compare current benchmarks against baseline; return 0/1/2."""
    current_data = _load_json(current_path, label="current benchmarks")
    baseline_data = _load_json(baseline_path, label="baseline")

    current_medians = _extract_medians(current_data)
    baseline_medians, file_threshold = _extract_baseline(baseline_data)
    threshold = threshold_override if threshold_override is not None else file_threshold

    if not current_medians:
        print("bench-regression: ERROR: no benchmarks in current file")
        return 2
    if not baseline_medians:
        print("bench-regression: ERROR: no benchmarks in baseline file")
        return 2

    names = sorted(set(baseline_medians) | set(current_medians))

    print(f"bench-regression: threshold = {threshold:.1f}%")
    print(f"{'Benchmark':58} {'Baseline':>12} {'Current':>12} {'Delta %':>10} {'Status':>8}")
    print("-" * 106)

    regressions: list[str] = []
    for name in names:
        base = baseline_medians.get(name)
        curr = current_medians.get(name)

        if base is None:
            print(f"{name:58} {'(new)':>12} {curr!s:>12} {'n/a':>10} {'SKIP':>8}")
            continue
        if curr is None:
            print(f"{name:58} {base!s:>12} {'(gone)':>12} {'n/a':>10} {'SKIP':>8}")
            continue

        pct = (curr - base) / base * 100.0 if base else 0.0
        pct_text = f"{pct:+.1f}%"
        regressed = pct > threshold

        status = "REGR" if regressed else "OK"
        print(f"{name:58} {base:12.6e} {curr:12.6e} {pct_text:>10} {status:>8}")
        if regressed:
            regressions.append(f"  {name}: {pct:+.1f}% (threshold {threshold:.1f}%)")

    print()

    if regressions:
        print(f"bench-regression: FAIL — {len(regressions)} regression(s):")
        for line in regressions:
            print(line)
        return 1

    print("bench-regression: PASS")
    return 0


# -------------------------------------------------------------------
# CLI
# -------------------------------------------------------------------


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fail when capture-overhead benchmarks regress beyond threshold."
    )
    parser.add_argument(
        "--current",
        type=Path,
        default=DEFAULT_CURRENT,
        help="Path to pytest-benchmark JSON (default: %(default)s).",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=DEFAULT_BASELINE,
        help="Path to baseline JSON (default: %(default)s).",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Max allowed regression %% (overrides baseline file value).",
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        help="Overwrite baseline with current medians instead of checking.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.update_baseline:
            return _update_baseline(args.current, args.baseline)
        return _check_regression(args.current, args.baseline, args.threshold)
    except RuntimeError as exc:
        print(f"bench-regression: ERROR: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
