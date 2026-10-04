"""ConDefects dataset importer for llmdebug framework.

Source: GitHub ``appmlk/ConDefects``.
1,625 Python cases from AtCoder (post-Oct 2021), specifically designed to
avoid LLM training data leakage.

contamination_risk = "low" — highest value for honest benchmarking since
results are least likely to be inflated by memorization.

Data layout
-----------
Code (GitHub repo)::

    Code/abc221_a/Python/45728797/faultyVersion.py
    Code/abc221_a/Python/45728797/correctVersion.py
    Code/abc221_a/Python/45728797/faultLocation.txt

Tests (Test.zip)::

    Test/abc221/A/in/example_00.txt
    Test/abc221/A/out/example_00.txt

Mapping rule: ``abc221_a`` → ``Test/abc221/A/`` (split on last ``_``,
uppercase the letter). Fallback: ``Test/{contest}/Ex/``.
"""

from __future__ import annotations

import shutil
import zipfile
from collections import defaultdict
from pathlib import Path
from typing import Any

from evals.category_mapping import infer_snapshot_dependency
from evals.importers.base import (
    count_patch_lines,
    generate_oracle_patch,
    has_external_deps,
    is_deterministic,
    ruff_format,
    validate_generated_case,
    write_case_files,
)

# ---------------------------------------------------------------------------
# Filter criteria
# ---------------------------------------------------------------------------

MAX_FIX_LINES = 15
MAX_CODE_LINES = 200
MAX_TEST_INPUT_BYTES = 10_240  # Skip large competitive programming inputs
MAX_TEST_PAIRS = 5


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_difficulty_file(repo_path: Path) -> dict[str, int | None]:
    """Parse ``difficulty.txt`` from repo root.

    Format: one ``task_name score`` pair per line.
    Returns mapping of task_name → numeric score (None for "unknown").
    """
    diff_file = repo_path / "difficulty.txt"
    if not diff_file.exists():
        return {}

    result: dict[str, int | None] = {}
    for line in diff_file.read_text(encoding="utf-8", errors="replace").splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 2:
            task_name = parts[0]
            try:
                result[task_name] = int(parts[1])
            except ValueError:
                result[task_name] = None
    return result


def _difficulty_from_score(score: int | None) -> str | None:
    """Map AtCoder numeric difficulty score to easy/medium/hard.

    Returns None if score is unknown (caller should use fix-size heuristic).
    """
    if score is None:
        return None
    if score <= 400:
        return "easy"
    if score <= 1200:
        return "medium"
    return "hard"


def _code_path_to_test_prefix(task_name: str) -> tuple[str, str]:
    """Map code task name to test directory prefix in Test.zip.

    ``"abc221_a"`` → ``("Test/abc221/A", "Test/abc221/Ex")``

    Uses ``rsplit("_", 1)`` to handle multi-part contest names like
    ``abc221_a`` correctly (not splitting ``abc`` from ``221_a``).
    """
    parts = task_name.rsplit("_", 1)
    if len(parts) == 2:
        contest, letter = parts
        primary = f"Test/{contest}/{letter.upper()}"
        fallback = f"Test/{contest}/Ex"
    else:
        primary = f"Test/{task_name}"
        fallback = f"Test/{task_name}/Ex"
    return primary, fallback


def _build_zip_index(zf: zipfile.ZipFile) -> dict[str, list[str]]:
    """Build index of zip entries grouped by parent directory.

    Single pass over ``zf.namelist()`` (~108K entries).
    Returns e.g. ``{"Test/abc221/A/in": ["example_00.txt", ...], ...}``.
    """
    index: dict[str, list[str]] = defaultdict(list)
    for name in zf.namelist():
        if name.endswith("/"):
            continue  # skip directory entries
        # Split into parent dir and filename
        last_slash = name.rfind("/")
        if last_slash >= 0:
            parent = name[:last_slash]
            filename = name[last_slash + 1 :]
            index[parent].append(filename)
    return dict(index)


def _extract_test_pairs_from_zip(
    zf: zipfile.ZipFile,
    zip_index: dict[str, list[str]],
    test_prefix: str,
) -> list[tuple[str, str]]:
    """Extract input/output test pairs from zip for a given test prefix.

    Matches by filename stem: ``in/test_00.txt`` ↔ ``out/test_00.txt``.
    Prioritizes ``example*`` files first (simpler tests).
    Skips inputs > MAX_TEST_INPUT_BYTES.
    """
    in_dir = f"{test_prefix}/in"
    out_dir = f"{test_prefix}/out"

    in_files = zip_index.get(in_dir, [])
    out_files_set = set(zip_index.get(out_dir, []))

    if not in_files or not out_files_set:
        return []

    # Sort: example files first, then alphabetical
    def sort_key(name: str) -> tuple[int, str]:
        return (0 if name.startswith("example") else 1, name)

    in_files_sorted = sorted(in_files, key=sort_key)

    pairs: list[tuple[str, str]] = []
    for in_name in in_files_sorted:
        if in_name not in out_files_set:
            continue

        in_path = f"{in_dir}/{in_name}"
        out_path = f"{out_dir}/{in_name}"

        # Check input size before reading
        info = zf.getinfo(in_path)
        if info.file_size > MAX_TEST_INPUT_BYTES:
            continue

        try:
            inp = zf.read(in_path).decode("utf-8", errors="replace")
            out = zf.read(out_path).decode("utf-8", errors="replace")
        except (KeyError, OSError):
            continue

        if inp.strip() and out.strip():
            pairs.append((inp, out))

        if len(pairs) >= MAX_TEST_PAIRS:
            break

    return pairs


# ---------------------------------------------------------------------------
# Test wrapping
# ---------------------------------------------------------------------------


def _generate_test(
    test_pairs: list[tuple[str, str]],
) -> str:
    """Generate test_buggy.py that runs buggy.py as a script with stdin/stdout.

    Uses multiple test input/output pairs for stronger validation.
    """
    lines = ["import subprocess", "import sys", "", ""]
    for i, (test_input, expected_output) in enumerate(test_pairs):
        lines.append(f"def test_condefects_{i}():")
        lines.append("    proc = subprocess.run(")
        lines.append('        [sys.executable, "buggy.py"],')
        lines.append(f"        input={test_input!r},")
        lines.append("        capture_output=True,")
        lines.append("        text=True,")
        lines.append("        timeout=10,")
        lines.append("    )")
        lines.append(f"    expected = {expected_output!r}")
        lines.append("    actual = proc.stdout.strip()")
        lines.append("    assert actual == expected.strip(), (")
        lines.append('        f"Expected {expected.strip()!r}, got {actual!r}"')
        lines.append("    )")
        lines.append("")
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _infer_exception(verdict: str) -> str:
    """Infer expected exception from verdict type."""
    verdict_lower = verdict.lower()
    if "runtime" in verdict_lower:
        return "RuntimeError"
    return "AssertionError"


def _classify_bug_from_patch(buggy_code: str, fixed_code: str) -> str:
    """Heuristic: classify bug type from patch content differences."""
    import difflib

    buggy_lines = buggy_code.splitlines()
    fixed_lines = fixed_code.splitlines()
    diff = list(difflib.unified_diff(buggy_lines, fixed_lines, lineterm=""))

    removed = [line[1:] for line in diff if line.startswith("-") and not line.startswith("---")]
    added = [line[1:] for line in diff if line.startswith("+") and not line.startswith("+++")]
    changed = " ".join(removed + added).lower()

    # Check for comparison/boundary changes
    if any(op in changed for op in ["<=", ">=", "< ", "> ", "==", "!="]):
        return "logic_error"
    # Check for off-by-one patterns
    if any(p in changed for p in ["+ 1", "- 1", "+1", "-1", "range("]):
        return "logic_error"
    # Check for type-related changes
    if any(p in changed for p in ["int(", "str(", "float(", "list(", "type("]):
        return "type_coercion"
    # Default
    return "logic_error"


def _estimate_difficulty(fix_lines: int, code_lines: int) -> str:
    """Estimate difficulty from fix size and code size."""
    if fix_lines <= 2 and code_lines <= 30:
        return "easy"
    if fix_lines <= 4 and code_lines <= 80:
        return "medium"
    return "hard"


# ---------------------------------------------------------------------------
# Main import function
# ---------------------------------------------------------------------------


def import_condefects(
    output_dir: Path,
    *,
    repo_path: Path | None = None,
    test_zip: Path | None = None,
    max_cases: int = 0,
    validate: bool = False,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """Import cases from ConDefects repository.

    The ConDefects repo must be cloned locally. Test data comes from a
    separate Test.zip file (22GB, downloaded from OneDrive).

    Args:
        output_dir: Directory to write cases to.
        repo_path: Path to cloned ConDefects repository.
        test_zip: Path to Test.zip file containing test inputs/outputs.
        max_cases: Maximum number of cases to import (0 = unlimited).
        validate: Whether to schema-validate each case.
        dry_run: If True, preview cases without writing files.

    Returns list of imported case metadata dicts.
    """
    if repo_path is None:
        raise ValueError(
            "ConDefects requires a local clone. "
            "Clone with: git clone https://github.com/appmlk/ConDefects.git"
        )

    code_dir = repo_path / "Code"
    if not code_dir.exists():
        raise FileNotFoundError(f"ConDefects Code/ directory not found: {code_dir}")

    if test_zip is None:
        raise ValueError(
            "ConDefects requires Test.zip for test input/output data. "
            "Download from the ConDefects OneDrive link and pass --test-zip."
        )
    if not test_zip.exists():
        raise FileNotFoundError(f"Test.zip not found: {test_zip}")

    # Load difficulty map
    difficulty_map = _parse_difficulty_file(repo_path)

    # Open zip and build index
    with zipfile.ZipFile(test_zip, "r") as zf:
        print("Building zip index...")
        zip_index = _build_zip_index(zf)
        print(f"Zip index: {len(zip_index)} directories")

        imported: list[dict[str, Any]] = []
        skipped = 0
        tasks_processed = 0
        no_python = 0
        no_tests = 0

        task_dirs = sorted(code_dir.iterdir())
        for task_dir in task_dirs:
            if max_cases > 0 and len(imported) >= max_cases:
                break
            if not task_dir.is_dir():
                continue

            task_name = task_dir.name
            tasks_processed += 1

            if tasks_processed % 50 == 0:
                print(
                    f"  Progress: {tasks_processed}/{len(task_dirs)} tasks, "
                    f"{len(imported)} imported, {skipped} skipped"
                )

            # Skip tasks without Python submissions
            python_dir = task_dir / "Python"
            if not python_dir.exists():
                no_python += 1
                continue

            # Map task name to test prefix and find test pairs
            primary_prefix, fallback_prefix = _code_path_to_test_prefix(task_name)
            test_pairs = _extract_test_pairs_from_zip(zf, zip_index, primary_prefix)
            if not test_pairs:
                test_pairs = _extract_test_pairs_from_zip(zf, zip_index, fallback_prefix)
            if not test_pairs:
                no_tests += 1
                continue

            # Get difficulty from AtCoder score
            # The difficulty map may key by the contest base (abc221) not task (abc221_a)
            score = difficulty_map.get(task_name)
            if score is None:
                # Try the contest prefix
                contest_prefix = task_name.rsplit("_", 1)[0] if "_" in task_name else task_name
                score = difficulty_map.get(contest_prefix)
            atcoder_difficulty = _difficulty_from_score(score)

            # Process each submission within this task
            for submission_dir in sorted(python_dir.iterdir()):
                if max_cases > 0 and len(imported) >= max_cases:
                    break
                if not submission_dir.is_dir():
                    continue

                submission_id = submission_dir.name
                buggy_file = submission_dir / "faultyVersion.py"
                fixed_file = submission_dir / "correctVersion.py"

                if not buggy_file.exists() or not fixed_file.exists():
                    skipped += 1
                    continue

                buggy_code = buggy_file.read_text(encoding="utf-8", errors="replace")
                fixed_code = fixed_file.read_text(encoding="utf-8", errors="replace")

                if not buggy_code.strip() or not fixed_code.strip():
                    skipped += 1
                    continue

                # Filter: external deps
                if has_external_deps(buggy_code):
                    skipped += 1
                    continue

                # Filter: determinism
                if not is_deterministic(buggy_code):
                    skipped += 1
                    continue

                # Filter: code size
                if len(buggy_code.splitlines()) > MAX_CODE_LINES:
                    skipped += 1
                    continue

                # Format both before generating patch
                buggy_code = ruff_format(buggy_code)
                fixed_code = ruff_format(fixed_code)

                oracle_patch = generate_oracle_patch(buggy_code, fixed_code)
                if not oracle_patch.strip():
                    skipped += 1
                    continue

                fix_lines = count_patch_lines(oracle_patch)
                if fix_lines > MAX_FIX_LINES:
                    skipped += 1
                    continue

                # Classify and determine difficulty
                category = _classify_bug_from_patch(buggy_code, fixed_code)
                snapshot_dep = infer_snapshot_dependency(category)
                expected_exception = _infer_exception("wrong answer")

                difficulty = atcoder_difficulty or _estimate_difficulty(
                    fix_lines, len(buggy_code.splitlines())
                )

                case_id = f"cd_{task_name}_{submission_id}_{difficulty}"

                # Skip if case already exists (idempotent re-runs)
                case_dir = output_dir / case_id
                if case_dir.exists():
                    skipped += 1
                    continue

                test_code = _generate_test(test_pairs)

                metadata = {
                    "description": f"ConDefects {task_name}/{submission_id}",
                    "category": category,
                    "difficulty": difficulty,
                    "snapshot_dependency": snapshot_dep,
                    "expected_exception": expected_exception,
                    "tags": ["condefects", "atcoder"],
                    "python_args": ["-m", "pytest", "-q"],
                    "timeout_sec": 30,
                    "bug_source": "condefects",
                    "contamination_risk": "low",
                    "expected_fix_lines": fix_lines,
                    "source_id": f"condefects_{task_name}_{submission_id}",
                }

                case_info = {"case_id": case_id, "metadata": metadata, "fix_lines": fix_lines}

                if dry_run:
                    imported.append(case_info)
                    continue

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

        print(
            f"\nDone: {len(imported)} imported, {skipped} skipped, "
            f"{no_python} tasks without Python, {no_tests} tasks without tests"
        )

    return imported
