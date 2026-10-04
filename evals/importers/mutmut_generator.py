"""mutmut-based mutation testing case generator for llmdebug framework.

Runs mutmut on a target Python module, extracts interesting mutants
(weakly-killed), and converts them into standard case directories.

contamination_risk = "none" — freshly generated from the user's own codebase.
"""

from __future__ import annotations

import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from evals.category_mapping import infer_snapshot_dependency, map_mutmut_operator
from evals.importers.base import (
    generate_oracle_patch,
    validate_generated_case,
    write_case_files,
)

# ---------------------------------------------------------------------------
# Filter criteria
# ---------------------------------------------------------------------------

MAX_FIX_LINES = 15


# ---------------------------------------------------------------------------
# mutmut interaction
# ---------------------------------------------------------------------------


def _run_mutmut(target: str) -> tuple[bool, str]:
    """Run mutmut on the target module. Returns (success, message)."""
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "mutmut", "run", "--paths-to-mutate", target],
            capture_output=True,
            text=True,
            timeout=600,
        )
        if proc.returncode != 0:
            return False, f"mutmut exited with code {proc.returncode}: {proc.stderr[:200]}"
        return True, ""
    except subprocess.TimeoutExpired:
        return False, "mutmut timed out after 600s"
    except FileNotFoundError:
        return False, "mutmut not found. Install with: pip install mutmut"


def _get_mutmut_results() -> tuple[list[dict[str, Any]], str]:
    """Query mutmut results. Returns (mutants, error_message)."""
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "mutmut", "results"],
            capture_output=True,
            text=True,
            timeout=30,
        )
        if proc.returncode != 0:
            return [], f"mutmut results exited with code {proc.returncode}: {proc.stderr[:200]}"
        return _parse_mutmut_results(proc.stdout), ""
    except subprocess.TimeoutExpired:
        return [], "mutmut results timed out after 30s"
    except FileNotFoundError:
        return [], "mutmut not found. Install with: pip install mutmut"


def _parse_mutmut_results(output: str) -> list[dict[str, Any]]:
    """Parse mutmut results output into structured data."""
    mutants: list[dict[str, Any]] = []
    # mutmut results format varies; parse for survived/killed/suspicious mutants.
    for line in output.splitlines():
        line = line.strip()
        if not line:
            continue
        # Look for mutant IDs and status.
        match = re.match(r"(\d+)\s+(.+)", line)
        if match:
            mutants.append(
                {
                    "id": match.group(1),
                    "status": match.group(2).strip(),
                }
            )
    return mutants


def _get_mutant_diff(mutant_id: str) -> str | None:
    """Get the diff for a specific mutant."""
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "mutmut", "show", mutant_id],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if proc.returncode == 0 and proc.stdout.strip():
            return proc.stdout
        return None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None


def _apply_diff_to_source(original: str, diff: str) -> str | None:
    """Apply a unified diff to original source to produce mutated code.

    Returns mutated source, or None if the diff can't be applied.
    """
    import tempfile

    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as src_f:
        src_f.write(original)
        src_path = src_f.name
    with tempfile.NamedTemporaryFile(mode="w", suffix=".diff", delete=False) as diff_f:
        diff_f.write(diff)
        diff_path = diff_f.name

    try:
        proc = subprocess.run(
            ["patch", "-s", "--no-backup-if-mismatch", src_path, diff_path],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if proc.returncode == 0:
            return Path(src_path).read_text(encoding="utf-8")
        return None
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return None
    finally:
        Path(src_path).unlink(missing_ok=True)
        Path(diff_path).unlink(missing_ok=True)


def _infer_operator_type(diff: str) -> str:
    """Infer mutmut operator type from diff content."""
    if not diff:
        return "keyword"

    # Strip diff markers (+/-/@@) to avoid false positives on diff syntax.
    content_lines = [
        line[1:]
        for line in diff.splitlines()
        if line.startswith("+") or line.startswith("-")
        if not line.startswith("+++") and not line.startswith("---")
    ]
    content = "\n".join(content_lines)
    content_lower = content.lower()

    # Check for common mutation patterns.
    if re.search(r"\w\s*[+\-*/%]\s*\w", content):
        return "arithmetic"
    if any(op in content for op in ["==", "!=", "<=", ">="]):
        return "comparison"
    if re.search(r"(?<!=)[<>](?!=)", content):
        return "comparison"
    if any(kw in content_lower for kw in ["true", "false", "not ", " and ", " or "]):
        return "boolean"
    if "return" in content_lower:
        return "changed_return"
    if re.search(r"\w+\(", content) and re.search(r"^-", diff, re.MULTILINE):
        return "removed_call"
    return "keyword"


# ---------------------------------------------------------------------------
# Test generation for mutants
# ---------------------------------------------------------------------------


def _find_test_file(target: str) -> Path | None:
    """Find the test file that exercises the target module."""
    target_path = Path(target)
    if target_path.is_file():
        module_name = target_path.stem
        # Look in common test locations.
        candidates = [
            target_path.parent / f"test_{module_name}.py",
            target_path.parent.parent / "tests" / f"test_{module_name}.py",
            Path("tests") / f"test_{module_name}.py",
        ]
        for c in candidates:
            if c.exists():
                return c
    return None


# ---------------------------------------------------------------------------
# Main import function
# ---------------------------------------------------------------------------


def import_mutmut(
    output_dir: Path,
    *,
    target: str,
    max_cases: int = 30,
    validate: bool = False,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """Generate cases from mutmut mutations on a target module.

    Pipeline:
    1. Run mutmut on target
    2. Query results for interesting mutants
    3. For each mutant: extract diff, create buggy.py, oracle.patch
    4. Validate and write case files

    Args:
        output_dir: Directory to write cases to.
        target: Python module/package to mutate.
        max_cases: Maximum number of cases to generate.
        validate: Whether to schema-validate each case.
        dry_run: If True, preview without writing files.

    Returns list of generated case metadata dicts.
    """
    if dry_run:
        print(f"[dry-run] Would run mutmut on {target} and generate up to {max_cases} cases.")
        return [
            {
                "case_id": f"mut_{Path(target).stem}_preview",
                "metadata": {
                    "bug_source": "mutmut",
                    "contamination_risk": "none",
                    "target": target,
                },
                "fix_lines": 0,
            }
        ]

    # Step 1: Run mutmut.
    print(f"Running mutmut on {target}...")
    ok, err_msg = _run_mutmut(target)
    if not ok:
        print(f"mutmut run failed: {err_msg}")
        return []

    # Step 2: Get results.
    mutants, err_msg = _get_mutmut_results()
    if not mutants:
        print(f"No mutants found.{f' ({err_msg})' if err_msg else ''}")
        return []

    # Step 3: Process killed mutants only (survived mutants pass tests, useless for eval).
    killed = [m for m in mutants if "killed" in m.get("status", "").lower()]
    print(f"Found {len(killed)} killed mutant(s) out of {len(mutants)} total.")

    target_path = Path(target)
    if not target_path.is_file():
        print(f"Target must be a single Python file for case generation: {target}")
        return []

    module_name = target_path.stem
    original_source = target_path.read_text(encoding="utf-8")

    # Find associated test file
    test_file = _find_test_file(target)
    test_code = test_file.read_text(encoding="utf-8") if test_file else ""

    imported: list[dict[str, Any]] = []
    skipped = 0

    for mutant in killed:
        if len(imported) >= max_cases:
            break

        mutant_id = mutant["id"]
        diff = _get_mutant_diff(mutant_id)
        if not diff:
            skipped += 1
            continue

        operator_type = _infer_operator_type(diff)
        category = map_mutmut_operator(operator_type)
        snapshot_dep = infer_snapshot_dependency(category)

        # Apply mutation to get buggy code
        buggy_code = _apply_diff_to_source(original_source, diff)
        if not buggy_code or buggy_code == original_source:
            skipped += 1
            continue

        # Oracle patch: buggy -> fixed (original)
        oracle_patch = generate_oracle_patch(buggy_code, original_source)
        if not oracle_patch.strip():
            skipped += 1
            continue

        if not test_code:
            skipped += 1
            continue

        case_id = f"mut_{module_name}_{mutant_id}"

        metadata = {
            "description": f"mutmut #{mutant_id} on {module_name} ({operator_type})",
            "category": category,
            "difficulty": "easy",
            "snapshot_dependency": snapshot_dep,
            "expected_exception": "AssertionError",
            "tags": ["mutmut", operator_type],
            "python_args": ["-m", "pytest", "-q"],
            "timeout_sec": 30,
            "bug_source": "mutmut",
            "contamination_risk": "none",
            "source_id": f"mutmut_{module_name}_{mutant_id}",
        }

        case_info = {"case_id": case_id, "metadata": metadata, "fix_lines": 0}

        case_dir = output_dir / case_id
        write_case_files(
            case_dir,
            buggy_code=buggy_code,
            test_code=test_code,
            oracle_patch=oracle_patch,
            metadata=metadata,
        )

        # DAT-05: the schema is always checked; --validate also runs the case
        # (the baseline must fail and the oracle must fix it).
        ok, msg = validate_generated_case(case_dir, schema_only=not validate)
        if not ok:
            print(f"  [skip] {case_id}: validation failed: {msg}")
            skipped += 1
            shutil.rmtree(case_dir, ignore_errors=True)
            continue

        imported.append(case_info)

    if skipped:
        print(f"Skipped {skipped} mutant(s).")
    return imported
