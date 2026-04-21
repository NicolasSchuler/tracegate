#!/usr/bin/env python3
"""Report pyright type-completeness and optionally gate on regressions.

Uses ``pyright --verifytypes`` to measure the percentage of exported symbols
that have fully known types.  The score is compared against a committed
baseline; by default the check is **informational** (exit 0 on regression)
unless ``--strict`` is passed.

The script follows the same conventions as ``check_coverage_no_regression.py``.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import subprocess
from pathlib import Path
from typing import Any

PACKAGE = "llmdebug"
DEFAULT_BASELINE_JSON = Path("quality/type_coverage_baseline.json")
FLOAT_TOLERANCE = 0.05  # percentage-points


# ---------------------------------------------------------------------- helpers


def _load_json(path: Path, *, label: str) -> dict[str, Any]:
    try:
        payload: object = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise RuntimeError(f"{label} not found: {path}") from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"{label} is not valid JSON: {path} ({exc})") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"{label} must contain a JSON object: {path}")
    return payload  # type: ignore[return-value]


def _run_pyright(python_version: str | None) -> dict[str, Any]:
    """Run ``pyright --verifytypes`` and return the parsed JSON."""
    cmd: list[str] = ["uv", "run"]
    if python_version:
        cmd += ["--python", python_version]
    cmd += ["pyright", "--verifytypes", PACKAGE, "--ignoreexternal", "--outputjson"]

    result = subprocess.run(cmd, capture_output=True, text=True)

    # pyright exits non-zero when completeness is below its own threshold.
    # We still want the JSON payload.
    if not result.stdout.strip():
        raise RuntimeError(
            f"pyright produced no JSON output (exit {result.returncode}).\n"
            f"stderr: {result.stderr[:500]}"
        )

    try:
        payload: object = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"pyright output is not valid JSON: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError("pyright output must be a JSON object")
    return payload  # type: ignore[return-value]


def _extract_completeness(pyright_payload: dict[str, Any]) -> dict[str, Any]:
    """Pull completeness metrics from the pyright JSON blob."""
    tc = pyright_payload.get("typeCompleteness")
    if not isinstance(tc, dict):
        raise RuntimeError(
            "pyright JSON is missing 'typeCompleteness'. "
            "Ensure pyright was invoked with --verifytypes."
        )

    score = tc.get("completenessScore")
    if not isinstance(score, (int, float)):
        raise RuntimeError("typeCompleteness is missing numeric 'completenessScore'")

    exported = tc.get("exportedSymbolCounts", {})
    return {
        "completeness_percent": round(float(score) * 100, 4),
        "exported_known": exported.get("withKnownType", 0),
        "exported_ambiguous": exported.get("withAmbiguousType", 0),
        "exported_unknown": exported.get("withUnknownType", 0),
    }


def _extract_baseline_percent(baseline_payload: dict[str, Any]) -> float:
    value = baseline_payload.get("completeness_percent")
    if isinstance(value, (int, float)):
        return float(value)
    raise RuntimeError(
        "type-coverage baseline is missing numeric 'completeness_percent' "
        "in quality/type_coverage_baseline.json"
    )


# ---------------------------------------------------------------------- CLI


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Report pyright type-completeness and compare to baseline."
    )
    parser.add_argument(
        "--baseline-json",
        type=Path,
        default=DEFAULT_BASELINE_JSON,
        help="Path to baseline JSON (default: quality/type_coverage_baseline.json).",
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=FLOAT_TOLERANCE,
        help="Allowed regression margin in percentage points (default: 0.05).",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        default=False,
        help="Exit non-zero on regression (default: informational only).",
    )
    parser.add_argument(
        "--update-baseline",
        action="store_true",
        default=False,
        help="Write current completeness to the baseline file and exit.",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------- main


def main() -> int:
    args = parse_args()
    python_version = None
    env_val = __import__("os").environ.get("PYTHON_VERSION")
    if env_val:
        python_version = env_val

    # ── Run pyright ──────────────────────────────────────────────────
    try:
        pyright_data = _run_pyright(python_version)
        metrics = _extract_completeness(pyright_data)
    except RuntimeError as exc:
        print(f"type-coverage: ERROR: {exc}")
        return 2

    current_pct = metrics["completeness_percent"]

    # ── Update-baseline mode ─────────────────────────────────────────
    if args.update_baseline:
        baseline_data = {
            "completeness_percent": current_pct,
            "exported_known": metrics["exported_known"],
            "exported_ambiguous": metrics["exported_ambiguous"],
            "exported_unknown": metrics["exported_unknown"],
            "source": "pyright --verifytypes --ignoreexternal",
            "updated_at_utc": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        }
        args.baseline_json.parent.mkdir(parents=True, exist_ok=True)
        args.baseline_json.write_text(
            json.dumps(baseline_data, indent=2) + "\n", encoding="utf-8"
        )
        print(f"type-coverage: baseline updated → {current_pct:.2f}% in {args.baseline_json}")
        return 0

    # ── Report ───────────────────────────────────────────────────────
    print(
        f"type-coverage: completeness={current_pct:.2f}% "
        f"(known={metrics['exported_known']}, "
        f"ambiguous={metrics['exported_ambiguous']}, "
        f"unknown={metrics['exported_unknown']})"
    )

    # ── Compare to baseline (if it exists) ───────────────────────────
    if not args.baseline_json.exists():
        print(
            f"type-coverage: SKIP: no baseline at {args.baseline_json} "
            "(run with --update-baseline to create one)"
        )
        return 0

    try:
        baseline_payload = _load_json(args.baseline_json, label="type-coverage baseline")
        baseline_pct = _extract_baseline_percent(baseline_payload)
    except RuntimeError as exc:
        print(f"type-coverage: ERROR: {exc}")
        return 2

    delta = current_pct - baseline_pct
    print(
        f"type-coverage: baseline={baseline_pct:.2f}% "
        f"delta={delta:+.2f}pp tolerance={args.tolerance:.2f}pp"
    )

    if delta + args.tolerance < 0.0:
        msg = "type-coverage: REGRESSION: type completeness dropped below baseline."
        if args.strict:
            print(f"{msg} (strict mode → FAIL)")
            return 1
        print(f"{msg} (informational only → WARN)")
        return 0

    print("type-coverage: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
