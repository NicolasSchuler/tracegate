"""PyResBugs dataset importer for llmdebug eval framework.

Source: HuggingFace ``OSS-forge/PyResBugs``.
5,007 residual Python bugs from production code, annotated with ODC fault
types (62 categories) and natural language descriptions.

Code is stored as single functions/methods with escaped newlines.
Unlike DebugBench (LeetCode Solution classes), PyResBugs code is standalone
and comes from real open-source projects.
"""

from __future__ import annotations

import ast
import subprocess
import textwrap
from pathlib import Path
from typing import Any

from evals.category_mapping import infer_snapshot_dependency, map_pyresbugs_type
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
MAX_CODE_LINES = 300


# ---------------------------------------------------------------------------
# Code decoding
# ---------------------------------------------------------------------------


def _unescape_code(s: str) -> str:
    r"""Unescape PyResBugs code strings (literal \\n → newline, \\t → tab).

    The encoding scheme is: original ``\`` → ``\\``, then newline → ``\\n``,
    tab → ``\\t``.  To decode correctly we must protect escaped backslashes
    before converting ``\\n``/``\\t`` (otherwise ``\\\\n`` decodes wrong).
    """
    # Phase 1: protect literal backslashes (\\  → placeholder).
    s = s.replace("\\\\", "\x00")
    # Phase 2: decode escape sequences.
    s = s.replace("\\n", "\n")
    s = s.replace("\\t", "\t")
    s = s.replace("\\'", "'")
    s = s.replace('\\"', '"')
    # Phase 3: restore backslashes.
    s = s.replace("\x00", "\\")
    return s


# ---------------------------------------------------------------------------
# Difficulty inference
# ---------------------------------------------------------------------------


def _infer_difficulty(faulty: str, fixed: str, patch_lines: int) -> str:
    """Infer difficulty from code complexity and patch size."""
    code_lines = len(faulty.splitlines())
    if patch_lines <= 2 and code_lines <= 15:
        return "easy"
    if patch_lines >= 8 or code_lines >= 60:
        return "hard"
    return "medium"


# ---------------------------------------------------------------------------
# Exception inference from fault type
# ---------------------------------------------------------------------------


def _infer_exception(fault_acronym: str, faulty_code: str, test_type: str) -> str:
    """Infer expected exception type from fault acronym and code.

    Args:
        test_type: "import" if test only does import, "call" if it calls function.
    """
    # Check if the code has a syntax error.
    try:
        compile(faulty_code, "<buggy>", "exec")
    except SyntaxError as exc:
        return type(exc).__name__

    # Missing import → ImportError
    if fault_acronym == "MIA":
        return "ImportError"

    # For import-only tests, most bugs inside function bodies won't trigger.
    # The test will pass, and validation will reject the case. But some bugs
    # at module level (decorators, default args, class bodies) DO trigger.
    if test_type == "import":
        # Module-level errors are typically NameError or TypeError
        return "NameError"

    # For call tests (zero-param functions), the actual error depends on the bug.
    if fault_acronym in ("WPFV", "WVIV"):
        return "NameError"
    return "AssertionError"


# ---------------------------------------------------------------------------
# Test generation for standalone functions
# ---------------------------------------------------------------------------


def _find_top_level_functions(code: str) -> list[str]:
    """Find names of top-level function definitions in code."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return []
    return [
        node.name
        for node in ast.iter_child_nodes(tree)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
    ]


def _find_changed_function(faulty: str, fixed: str) -> str | None:
    """Identify which function was changed between faulty and fixed code."""
    faulty_funcs = set(_find_top_level_functions(faulty))
    fixed_funcs = set(_find_top_level_functions(fixed))

    # If only one function exists, that's the changed one.
    all_funcs = faulty_funcs | fixed_funcs
    if len(all_funcs) == 1:
        return all_funcs.pop()

    # Compare function bodies to find the one that differs.
    try:
        faulty_tree = ast.parse(faulty)
        fixed_tree = ast.parse(fixed)
    except SyntaxError:
        return None

    faulty_bodies: dict[str, str] = {}
    for node in ast.iter_child_nodes(faulty_tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            faulty_bodies[node.name] = ast.dump(node)

    for node in ast.iter_child_nodes(fixed_tree):
        if (
            isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            and node.name in faulty_bodies
            and ast.dump(node) != faulty_bodies[node.name]
        ):
            return node.name

    return None


def _get_function_params(code: str, func_name: str) -> list[str] | None:
    """Get parameter names for a function (excluding self/cls)."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == func_name:
            return [arg.arg for arg in node.args.args if arg.arg not in ("self", "cls")]
    return None


def _generate_test(
    faulty_code: str,
    fixed_code: str,
    fault_acronym: str,
) -> str | None:
    """Generate test_buggy.py for a PyResBugs case.

    Strategy:
    1. For import errors: test that the module imports
    2. For functions with no required params: call directly
    3. Otherwise: import-only test (only catches import/syntax errors)
    """
    # For missing imports, just check the module imports.
    if fault_acronym == "MIA":
        return textwrap.dedent("""\
            def test_import():
                import buggy  # noqa: F401
        """)

    # Find the changed function.
    func_name = _find_changed_function(faulty_code, fixed_code)
    if not func_name:
        funcs = _find_top_level_functions(fixed_code)
        func_name = funcs[0] if funcs else None

    if not func_name:
        return textwrap.dedent("""\
            def test_import():
                import buggy  # noqa: F401
        """)

    params = _get_function_params(fixed_code, func_name)
    if params is None:
        params = []

    # Only generate a call test for zero-param functions.
    # For functions with params, we can't infer valid arguments,
    # so an import-only test is the safest option.
    if not params:
        return textwrap.dedent(f"""\
            from buggy import {func_name}


            def test_call():
                {func_name}()
        """)

    # For functions with required parameters, just test import.
    # This catches syntax errors and top-level name errors.
    return textwrap.dedent("""\
        def test_import():
            import buggy  # noqa: F401
    """)


# ---------------------------------------------------------------------------
# Code wrapping
# ---------------------------------------------------------------------------


def _verify_patch(case_dir: Path) -> bool:
    """Verify that oracle.patch applies cleanly to buggy.py in the case dir."""
    try:
        result = subprocess.run(
            ["patch", "--dry-run", "-p1"],
            input=(case_dir / "oracle.patch").read_text(encoding="utf-8"),
            capture_output=True,
            text=True,
            cwd=case_dir,
            timeout=5,
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return False


def _ensure_valid_module(code: str) -> str:
    """Ensure code is a valid Python module.

    PyResBugs code is typically a single function. If it uses names
    that aren't defined in the snippet (e.g., from the original module),
    we can't fix that — the filter steps will catch those cases.
    """
    # Dedent if the whole code is indented (method extracted from a class).
    lines = code.splitlines()
    if lines and all(ln.startswith(("\t", "    ")) or not ln.strip() for ln in lines):
        code = textwrap.dedent(code)
    return code


# ---------------------------------------------------------------------------
# Main import function
# ---------------------------------------------------------------------------


def import_pyresbugs(
    output_dir: Path,
    *,
    max_cases: int = 0,
    validate: bool = True,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """Import cases from PyResBugs dataset.

    Returns list of imported case metadata dicts (or preview dicts if dry_run).
    """
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError(
            "The 'datasets' package is required for PyResBugs import. "
            "Install with: pip install datasets"
        ) from None

    ds = load_dataset("OSS-forge/PyResBugs", split="train")

    imported: list[dict[str, Any]] = []
    skipped = 0

    for idx, row in enumerate(ds):
        if max_cases > 0 and len(imported) >= max_cases:
            break

        if idx > 0 and idx % 500 == 0:
            print(
                f"  [pyresbugs] processed {idx} rows, imported {len(imported)}, skipped {skipped}"
            )

        # Extract and decode fields.
        faulty_raw = str(row.get("Faulty Code", "") or "")
        fixed_raw = str(row.get("Fault Free Code", "") or "")
        fault_acronym = str(row.get("Fault_Acronym", "") or "")

        if not faulty_raw.strip() or not fixed_raw.strip():
            skipped += 1
            continue

        # Unescape the code (PyResBugs stores \\n as literal backslash-n).
        faulty_code = _unescape_code(faulty_raw)
        fixed_code = _unescape_code(fixed_raw)

        # Ensure valid module structure (dedent if needed).
        faulty_code = _ensure_valid_module(faulty_code)
        fixed_code = _ensure_valid_module(fixed_code)

        # Map metadata.
        category = map_pyresbugs_type(fault_acronym)
        case_id = f"prb_{idx:04d}"

        # Skip if case already exists.
        case_dir = output_dir / case_id
        if not dry_run and case_dir.exists():
            skipped += 1
            continue

        # Apply filters.
        if has_external_deps(faulty_code) or has_external_deps(fixed_code):
            skipped += 1
            continue

        if not is_deterministic(faulty_code):
            skipped += 1
            continue

        if len(faulty_code.splitlines()) > MAX_CODE_LINES:
            skipped += 1
            continue

        # Apply ruff formatting before generating patches.
        faulty_formatted = ruff_format(faulty_code)
        fixed_formatted = ruff_format(fixed_code)

        # Generate oracle patch.
        oracle_patch = generate_oracle_patch(faulty_formatted, fixed_formatted)
        if not oracle_patch.strip():
            skipped += 1
            continue

        fix_lines = count_patch_lines(oracle_patch)
        if fix_lines > MAX_FIX_LINES:
            skipped += 1
            continue

        # Generate test.
        test_code = _generate_test(faulty_formatted, fixed_formatted, fault_acronym)
        if test_code is None:
            skipped += 1
            continue

        # Determine test type for exception inference.
        test_type = "call" if "def test_call" in test_code else "import"

        difficulty = _infer_difficulty(faulty_formatted, fixed_formatted, fix_lines)
        snapshot_dep = infer_snapshot_dependency(category)
        expected_exception = _infer_exception(fault_acronym, faulty_formatted, test_type)

        # Append difficulty to case_id for consistency with other importers.
        case_id = f"prb_{idx:04d}_{difficulty}"
        case_dir = output_dir / case_id

        if not dry_run and case_dir.exists():
            skipped += 1
            continue

        nl_desc = str(row.get("High-Level Description", "") or "")
        project = str(row.get("Project", "") or "")

        metadata = {
            # Neutral by design (ATT-02); the fault type and description are analysis-only.
            "description": f"PyResBugs #{idx} in {project}",
            "source_fault_description": nl_desc,
            "category": category,
            "difficulty": difficulty,
            "snapshot_dependency": snapshot_dep,
            "expected_exception": expected_exception,
            "tags": ["pyresbugs", fault_acronym],
            "python_args": ["-m", "pytest", "-q"],
            "timeout_sec": 30,
            "bug_source": "pyresbugs",
            "contamination_risk": "low",
            "expected_fix_lines": fix_lines,
            "source_id": f"pyresbugs_{idx}",
        }

        case_info = {"case_id": case_id, "metadata": metadata, "fix_lines": fix_lines}

        if dry_run:
            imported.append(case_info)
            continue

        write_case_files(
            case_dir,
            buggy_code=faulty_formatted,
            test_code=test_code,
            oracle_patch=oracle_patch,
            metadata=metadata,
        )

        # Verify the patch applies before final validation.
        if not _verify_patch(case_dir):
            skipped += 1
            import shutil

            shutil.rmtree(case_dir, ignore_errors=True)
            continue

        if validate:
            ok, msg = validate_generated_case(case_dir, schema_only=False)
            if not ok:
                print(f"  [skip] {case_id}: validation failed: {msg}")
                skipped += 1
                import shutil

                shutil.rmtree(case_dir, ignore_errors=True)
                continue

        imported.append(case_info)

    return imported
