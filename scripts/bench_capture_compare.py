"""Compare capture-overhead benchmark medians between two pytest-benchmark JSON files.

Reads two pytest-benchmark JSON result files and prints a side-by-side table of
median timings with absolute and percentage deltas, making it easy to spot
regressions or improvements between runs.

Usage
-----
    # After running `just bench-capture` twice (e.g. before and after a change):
    python scripts/bench_capture_compare.py <baseline.json> <current.json>

    # Example:
    python scripts/bench_capture_compare.py \\
        .benchmarks/capture-overhead-baseline.json \\
        .benchmarks/capture-overhead-current.json
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from llmdebug import _json as json


def _bench_medians(path: Path) -> dict[str, float]:
    """Extract benchmark median times from a pytest-benchmark JSON file.

    Args:
        path: Path to a pytest-benchmark JSON result file.

    Returns:
        A mapping of benchmark name to its median execution time in seconds.
        Benchmarks with missing or invalid stats are silently skipped.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    rows = payload.get("benchmarks", [])
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
            out[name] = float(stats.get("median", 0.0))
        except (TypeError, ValueError):
            continue
    return out


def main(argv: list[str] | None = None) -> int:
    """Run the benchmark comparison and print a formatted table to stdout.

    Parses two positional file arguments (baseline and current JSON paths),
    extracts median timings from each, and prints a comparison table showing
    the absolute and percentage delta for every benchmark.

    Args:
        argv: Argument list to parse; defaults to ``sys.argv[1:]`` when *None*.

    Returns:
        Exit code: ``0`` on success, ``2`` when the wrong number of arguments
        is supplied.
    """
    args = argv or sys.argv[1:]
    if len(args) != 2:
        print(
            "usage: bench_capture_compare.py <baseline.json> <current.json>",
            file=sys.stderr,
        )
        return 2

    baseline_path = Path(args[0])
    current_path = Path(args[1])

    baseline = _bench_medians(baseline_path)
    current = _bench_medians(current_path)
    names = sorted(set(baseline) | set(current))

    print("Capture overhead median comparison (seconds)")
    print(f"{'Benchmark':58} {'Baseline':>12} {'Current':>12} {'Delta':>12} {'Delta %':>10}")
    print("-" * 108)

    for name in names:
        base = baseline.get(name)
        curr = current.get(name)
        if base is None or curr is None:
            print(f"{name:58} {base!s:>12} {curr!s:>12} {'n/a':>12} {'n/a':>10}")
            continue
        delta = curr - base
        pct = (delta / base * 100.0) if base else float("inf")
        pct_text = "inf" if pct == float("inf") else f"{pct:+.2f}%"
        print(f"{name:58} {base:12.6f} {curr:12.6f} {delta:+12.6f} {pct_text:>10}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
