"""Auto-correct expected_exception in case.json based on actual test output.

Many imported cases predict the wrong exception type (e.g., AssertionError when
the actual error is NameError). This script runs each case's baseline test,
captures the actual exception, updates case.json, and verifies the oracle patch
still fixes it.

Usage::

    PYTHONPATH=. uv run python evals/importers/fixup_expected_exceptions.py [--bug-source debugbench]
"""

from __future__ import annotations

import argparse
import re
import shutil
import tempfile
from pathlib import Path

from evals.case_loading import load_cases
from evals.eval_config import (
    _REPO_ROOT,
    add_execution_arguments,
    build_execution_config_from_args,
    set_active_execution_config,
)
from evals.eval_patches import apply_diff
from evals.eval_types import Condition
from evals.pytest_runner import run_pytest
from evals.validate_cases import (
    expected_exception_in_output,
    extract_exception_candidates,
)

from .. import _json as json

_EXCEPTION_TOKEN_RE = re.compile(r"\b([A-Za-z_][\w\.]*(?:Error|Exception|Exit|Interrupt))\b")


def _select_best_exception(output: str) -> str | None:
    """Select the most relevant exception candidate from pytest output."""
    lines = output.splitlines()

    # Prioritize exceptions found near buggy.py traceback frames.
    buggy_nearby: list[str] = []
    for i, line in enumerate(lines):
        if 'File "' not in line or "buggy.py" not in line:
            continue
        for nearby in lines[i + 1 : i + 8]:
            match = _EXCEPTION_TOKEN_RE.search(nearby)
            if match and (":" in nearby or nearby.lstrip().startswith("E")):
                buggy_nearby.append(match.group(1))
    if buggy_nearby:
        return buggy_nearby[-1]

    candidates = extract_exception_candidates(output)
    return candidates[0] if candidates else None


def _delete_skipped_cases(cases_dir_path: Path, results: dict[str, str]) -> int:
    """Delete cases explicitly marked as skipped by fixup run."""
    deleted = 0
    for case_id, status in results.items():
        if not status.startswith("skip"):
            continue
        case_dir = cases_dir_path / case_id
        if case_dir.exists():
            shutil.rmtree(case_dir, ignore_errors=True)
            deleted += 1
    return deleted


def fixup_cases(
    cases_dir: Path,
    *,
    bug_source: str | None = None,
    timeout_sec: int = 30,
    output_format: str = "json_compact",
) -> dict[str, str]:
    """Auto-correct expected_exception for cases that fail only on exception mismatch.

    Returns dict mapping case_id -> status.
    """
    cases = load_cases(cases_dir)
    results: dict[str, str] = {}

    for spec in cases:
        case_id = spec.case_id
        case_path = spec.path

        # Filter by bug_source if specified.
        if bug_source and spec.bug_source != bug_source:
            continue

        # Read full case.json for modification.
        meta = json.loads((case_path / "case.json").read_text(encoding="utf-8"))
        expected_exc = spec.expected_exception
        python_args = spec.python_args
        case_timeout = timeout_sec if timeout_sec > 0 else spec.timeout_sec

        with tempfile.TemporaryDirectory(prefix=f"llmdebug-fixup-{case_id}-") as tmp:
            workdir = Path(tmp)
            shutil.copytree(
                case_path,
                workdir,
                dirs_exist_ok=True,
                ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
            )

            # Run baseline.
            baseline = run_pytest(
                workdir, python_args, Condition.BASE_REPAIR, case_timeout, output_format
            )

            if baseline.timed_out:
                results[case_id] = "skip: timed out"
                continue

            if baseline.returncode == 0:
                results[case_id] = "skip: baseline passed"
                continue

            combined = (baseline.stdout or "") + "\n" + (baseline.stderr or "")

            # Check if current expected_exception is already correct.
            if expected_exception_in_output(expected_exc, combined):
                # Already correct — verify oracle patch works.
                patch_text = (case_path / "oracle.patch").read_text(encoding="utf-8")
                applied, _ = apply_diff(workdir, patch_text)
                if applied:
                    patched = run_pytest(
                        workdir,
                        python_args,
                        Condition.BASE_REPAIR,
                        case_timeout,
                        output_format,
                    )
                    if patched.returncode == 0:
                        results[case_id] = "ok: already correct"
                        continue
                    results[case_id] = "skip: oracle didn't fix"
                    continue
                results[case_id] = "skip: oracle patch failed"
                continue

            # Find actual exception.
            actual_exc = _select_best_exception(combined)
            if actual_exc is None:
                results[case_id] = "skip: no exception found"
                continue

            # Verify oracle patch fixes the issue with the new exception.
            patch_text = (case_path / "oracle.patch").read_text(encoding="utf-8")
            applied, _ = apply_diff(workdir, patch_text)
            if not applied:
                results[case_id] = "skip: oracle patch failed"
                continue

            patched = run_pytest(
                workdir, python_args, Condition.BASE_REPAIR, case_timeout, output_format
            )
            if patched.returncode != 0:
                results[case_id] = "skip: oracle didn't fix"
                continue

            # Update case.json with corrected expected_exception.
            meta["expected_exception"] = actual_exc
            (case_path / "case.json").write_text(
                json.dumps(meta, indent=2, ensure_ascii=False) + "\n",
                encoding="utf-8",
            )
            results[case_id] = f"fixed: {expected_exc} -> {actual_exc}"
            print(f"  [fixed] {case_id}: {expected_exc} -> {actual_exc}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Auto-correct expected_exception in case.json.")
    parser.add_argument("--cases-dir", default="evals/cases")
    parser.add_argument("--bug-source", default=None)
    parser.add_argument("--timeout-sec", type=int, default=30)
    parser.add_argument(
        "--output-format",
        default="json_compact",
        choices=["json", "json_compact", "toon"],
    )
    parser.add_argument(
        "--delete-skipped",
        action="store_true",
        help="Delete skipped cases after fixup (destructive, disabled by default).",
    )
    add_execution_arguments(parser)
    args = parser.parse_args()
    try:
        execution_config = build_execution_config_from_args(args)
    except ValueError as exc:
        raise SystemExit(f"Invalid arguments: {exc}") from exc

    cases_dir = Path(args.cases_dir)
    if not cases_dir.is_absolute():
        cases_dir = _REPO_ROOT / cases_dir

    previous_execution_config = set_active_execution_config(execution_config)
    try:
        results = fixup_cases(
            cases_dir,
            bug_source=args.bug_source,
            timeout_sec=args.timeout_sec,
            output_format=args.output_format,
        )
    finally:
        set_active_execution_config(previous_execution_config)

    fixed = sum(1 for v in results.values() if v.startswith("fixed"))
    ok = sum(1 for v in results.values() if v.startswith("ok"))
    skipped = sum(1 for v in results.values() if v.startswith("skip"))

    print(f"\nResults: {fixed} fixed, {ok} already correct, {skipped} skipped")

    if args.delete_skipped:
        cases_dir_path = Path(args.cases_dir)
        if not cases_dir_path.is_absolute():
            cases_dir_path = _REPO_ROOT / cases_dir_path
        deleted = _delete_skipped_cases(cases_dir_path, results)
        if deleted:
            print(f"Deleted {deleted} skipped cases.")
    else:
        print("Skipped cases were kept (pass --delete-skipped to remove them).")


if __name__ == "__main__":
    main()
