"""Validate eval cases against their oracle patches.

Checks that every case directory contains the required files, that the
baseline test run fails with the expected exception, and that applying
the oracle patch makes the test pass.  Supports concurrent validation
via ``--jobs`` and a schema-only mode that skips execution.
"""

from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import Future, ThreadPoolExecutor, as_completed
from pathlib import Path

from .run_eval import (
    _REPO_ROOT,
    CaseSpec,
    Condition,
    add_execution_arguments,
    apply_diff,
    build_execution_config_from_args,
    check_testcontainers_runtime_available,
    load_cases,
    run_pytest,
    set_active_execution_config,
)

REQUIRED_CASE_FILES = ("test_buggy.py", "case.json", "oracle.patch")
EXCEPTION_TOKEN_RE = re.compile(r"\b([A-Za-z_][\w\.]*(?:Error|Exception|Exit|Interrupt))\b")


def main(argv: list[str] | None = None) -> int:
    """CLI entry point: validate eval cases with oracle patches."""
    parser = argparse.ArgumentParser(description="Validate eval cases with oracle patches.")
    parser.add_argument("--cases-dir", default="evals/cases", help="Cases directory.")
    parser.add_argument("--case", action="append", default=[], help="Case id(s) to validate.")
    parser.add_argument(
        "--category", action="append", default=[], help="Category filter (repeatable)."
    )
    parser.add_argument(
        "--difficulty", action="append", default=[], help="Difficulty filter (repeatable)."
    )
    parser.add_argument(
        "--snapshot-dependency",
        action="append",
        default=[],
        help="Snapshot dependency filter (repeatable): required|helpful|none.",
    )
    parser.add_argument(
        "--bug-source", action="append", default=[], help="Bug source filter (repeatable)."
    )
    parser.add_argument(
        "--max-cases", type=int, default=0, help="Max number of cases to validate (0 = all)."
    )
    parser.add_argument(
        "--schema-only",
        action="store_true",
        help="Validate case schema/contract only; skip baseline-fail/oracle-pass execution.",
    )
    parser.add_argument(
        "--output-format",
        default="json_compact",
        choices=["json", "json_compact", "toon"],
        help="Output format env for snapshot condition compatibility.",
    )
    parser.add_argument(
        "--jobs",
        type=int,
        default=1,
        help="Number of cases to validate concurrently.",
    )
    add_execution_arguments(parser)
    args = parser.parse_args(argv)
    try:
        execution_config = build_execution_config_from_args(args)
    except ValueError as exc:
        print(f"Invalid arguments: {exc}")
        return 2
    if int(args.jobs) < 1:
        print("Invalid --jobs: must be >= 1")
        return 2

    cases_dir = Path(args.cases_dir)
    if not cases_dir.is_absolute():
        cases_dir = _REPO_ROOT / cases_dir
    try:
        cases = load_cases(cases_dir)
    except ValueError as e:
        print(f"Invalid case metadata: {e}")
        return 2

    if args.case:
        wanted = set(args.case)
        cases = [c for c in cases if c.case_id in wanted]
    if args.category:
        wanted_categories = {x.strip() for x in args.category if x.strip()}
        cases = [c for c in cases if c.category in wanted_categories]
    if args.difficulty:
        wanted_difficulties = {x.strip() for x in args.difficulty if x.strip()}
        cases = [c for c in cases if c.difficulty in wanted_difficulties]
    if args.snapshot_dependency:
        wanted_snapshot_dependencies = {x.strip() for x in args.snapshot_dependency if x.strip()}
        cases = [c for c in cases if c.snapshot_dependency in wanted_snapshot_dependencies]
    if args.bug_source:
        wanted_bug_sources = {x.strip() for x in args.bug_source if x.strip()}
        cases = [c for c in cases if (c.bug_source or "hand_crafted") in wanted_bug_sources]
    if args.max_cases > 0:
        cases = cases[: args.max_cases]

    if not cases:
        print("No cases selected for validation.")
        return 2

    if not args.schema_only:
        if execution_config.executor == "local":
            ok, message = check_pytest_available()
        else:
            ok, message = check_testcontainers_runtime_available()
        if not ok:
            print(f"Environment check failed: {message}")
            return 2

    previous_execution_config = set_active_execution_config(execution_config)
    try:
        failures: list[str] = []
        jobs = int(args.jobs)
        if jobs == 1:
            for case in cases:
                try:
                    ok, message = _validate_selected_case(
                        case=case,
                        schema_only=bool(args.schema_only),
                        output_format=str(args.output_format),
                    )
                except Exception as exc:
                    ok = False
                    message = f"internal error: {type(exc).__name__}: {exc}"
                status = "PASS" if ok else "FAIL"
                print(
                    f"[{status}] {case.case_id} ({case.category}/{case.difficulty}/{case.snapshot_dependency}) - {message}"
                )
                if not ok:
                    failures.append(f"{case.case_id}: {message}")
        else:
            future_map: dict[Future[tuple[bool, str]], CaseSpec] = {}
            with ThreadPoolExecutor(max_workers=jobs) as executor:
                for case in cases:
                    future = executor.submit(
                        _validate_selected_case,
                        case=case,
                        schema_only=bool(args.schema_only),
                        output_format=str(args.output_format),
                    )
                    future_map[future] = case

                for future in as_completed(future_map):
                    case = future_map[future]
                    try:
                        ok, message = future.result()
                    except Exception as exc:
                        ok = False
                        message = f"internal error: {type(exc).__name__}: {exc}"
                    status = "PASS" if ok else "FAIL"
                    print(
                        f"[{status}] {case.case_id} ({case.category}/{case.difficulty}/{case.snapshot_dependency}) - {message}"
                    )
                    if not ok:
                        failures.append(f"{case.case_id}: {message}")

        print(
            f"\nValidated {len(cases)} case(s): {len(cases) - len(failures)} passed, {len(failures)} failed."
        )
        if failures:
            return 1
        return 0
    finally:
        set_active_execution_config(previous_execution_config)


def _validate_selected_case(
    *,
    case: CaseSpec,
    schema_only: bool,
    output_format: str,
) -> tuple[bool, str]:
    """Run schema and optional execution validation for a single case."""
    schema_ok, schema_message = validate_case_contract(case.path)
    if not schema_ok:
        return False, schema_message
    if schema_only:
        return True, "schema and contract valid"
    return validate_one_case(
        case.path,
        case.expected_exception,
        case.python_args,
        case.timeout_sec,
        output_format,
    )


def validate_case_contract(case_path: Path) -> tuple[bool, str]:
    """Check that *case_path* contains all required files.

    Returns (ok, message) where *ok* is ``True`` when the directory
    has ``test_buggy.py``, ``case.json``, ``oracle.patch``, and at
    least one non-test Python implementation file.
    """
    for required_name in REQUIRED_CASE_FILES:
        if not (case_path / required_name).exists():
            return False, f"missing required file: {required_name}"
    implementation_files = [
        p
        for p in case_path.rglob("*.py")
        if p.name != "test_buggy.py" and not p.name.startswith("test_")
    ]
    if not implementation_files:
        return False, "missing implementation python file (non-test)"
    return True, "ok"


def check_pytest_available() -> tuple[bool, str]:
    """Verify that pytest is importable in the current interpreter."""
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "--version"],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,
        )
    except Exception as exc:
        return False, f"unable to execute pytest preflight: {type(exc).__name__}: {exc}"
    if proc.returncode != 0:
        details = (proc.stderr or proc.stdout or "").strip()
        return False, f"pytest is not available in this interpreter: {details or 'unknown error'}"
    return True, "ok"


def extract_exception_candidates(output: str) -> list[str]:
    """Extract exception class names from pytest output.

    Scans lines starting with ``E`` first; falls back to full text
    when no matches are found.
    """
    candidates: list[str] = []
    for line in output.splitlines():
        if not line.startswith("E"):
            continue
        candidates.extend(EXCEPTION_TOKEN_RE.findall(line))
    if not candidates:
        candidates.extend(EXCEPTION_TOKEN_RE.findall(output))
    return candidates


def expected_exception_in_output(expected_exception: str, output: str) -> bool:
    """Return ``True`` if *expected_exception* appears in *output*.

    Matches against both the fully-qualified name and the simple
    (unqualified) class name of each candidate found in the output.
    """
    expected_simple = expected_exception.rsplit(".", 1)[-1]
    for candidate in extract_exception_candidates(output):
        candidate_simple = candidate.rsplit(".", 1)[-1]
        if candidate == expected_exception or candidate_simple == expected_simple:
            return True
    return False


def detect_environment_error(output: str) -> str | None:
    """Return the first environment-setup error marker found, or ``None``."""
    checks = (
        "No module named pytest",
        "pytest: command not found",
        "No such file or directory: 'pytest'",
    )
    for marker in checks:
        if marker in output:
            return marker
    return None


def validate_one_case(
    case_path: Path,
    expected_exception: str,
    python_args: list[str],
    timeout_sec: int,
    output_format: str,
) -> tuple[bool, str]:
    """Run full execution validation for a single case.

    Copies the case to a temp directory, verifies the baseline run fails
    with the expected exception, applies the oracle patch, and checks that
    the patched run passes.
    """
    oracle_patch = case_path / "oracle.patch"
    if not oracle_patch.exists():
        return False, "missing oracle.patch"

    with tempfile.TemporaryDirectory(prefix=f"llmdebug-case-{case_path.name}-") as tmp:
        workdir = Path(tmp)
        shutil.copytree(
            case_path,
            workdir,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
        )

        baseline = run_pytest(
            workdir,
            python_args,
            Condition.BASE_REPAIR,
            timeout_sec,
            output_format,
        )
        if baseline.timed_out:
            return False, f"baseline run timed out after {baseline.timeout_sec}s"
        if baseline.returncode == 0:
            return False, "baseline unexpectedly passed"

        combined = (baseline.stdout or "") + "\n" + (baseline.stderr or "")
        environment_error = detect_environment_error(combined)
        if environment_error is not None:
            return False, f"environment/setup error during baseline run: {environment_error}"

        if not expected_exception_in_output(expected_exception, combined):
            candidates = extract_exception_candidates(combined)
            if candidates:
                return (
                    False,
                    f"expected exception '{expected_exception}' not found; observed candidates: {sorted(set(candidates))}",
                )
            return False, f"expected exception '{expected_exception}' not found in baseline output"

        patch_text = oracle_patch.read_text(encoding="utf-8")
        applied, apply_message = apply_diff(workdir, patch_text)
        if not applied:
            return False, f"oracle patch failed: {apply_message}"

        patched = run_pytest(
            workdir,
            python_args,
            Condition.BASE_REPAIR,
            timeout_sec,
            output_format,
        )
        if patched.timed_out:
            return False, f"patched run timed out after {patched.timeout_sec}s"
        if patched.returncode != 0:
            return False, "patched run did not pass"

    return True, "baseline fails and oracle patch passes"


if __name__ == "__main__":
    raise SystemExit(main())
