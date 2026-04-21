"""DebugBench dataset importer for llmdebug eval framework.

Source: HuggingFace ``Rtian/DebugBench`` (Apache-2.0 license).
~1,414 Python cases with GPT-4-implanted bugs in LeetCode-style problems.

Each case has a difficulty level and bug type taxonomy matching our needs.
The dataset uses ``python3`` as the language field. Each row contains a
``Solution`` class with a single method, ``examples`` with Input/Output
pairs, and ``category``/``subtype`` for bug classification.
"""

from __future__ import annotations

import ast
import re
import textwrap
from pathlib import Path
from typing import Any

from evals.category_mapping import infer_snapshot_dependency, map_debugbench_type
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

# LeetCode solutions often assume these imports are available.
_LEETCODE_PREAMBLE = """\
from typing import List, Optional, Tuple, Dict, Set
from collections import defaultdict, Counter, deque
import heapq
import math
import bisect
import itertools
import functools
"""

_VALID_DIFFICULTIES = {"easy", "medium", "hard"}


# ---------------------------------------------------------------------------
# Test generation for LeetCode-style Solution classes
# ---------------------------------------------------------------------------


def _extract_method_name(code: str) -> str | None:
    """Extract the first non-__init__ method name from a Solution class."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return None
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "Solution":
            for item in node.body:
                if (
                    isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                    and item.name != "__init__"
                ):
                    return item.name
    return None


def _parse_examples(examples: list[str]) -> list[tuple[str, str]]:
    """Parse LeetCode-style examples into (input_str, output_str) pairs."""
    pairs: list[tuple[str, str]] = []
    for ex in examples:
        input_match = re.search(r"Input:\s*(.+?)(?:\nOutput:|\n\n|$)", ex, re.DOTALL)
        output_match = re.search(r"Output:\s*(.+?)(?:\nExplanation:|\n\n|$)", ex, re.DOTALL)
        if input_match and output_match:
            pairs.append((input_match.group(1).strip(), output_match.group(1).strip()))
    return pairs


def _parse_input_args(input_str: str) -> list[tuple[str, str]]:
    """Parse 'key1 = val1, key2 = val2' into [(key, val), ...]."""
    args: list[tuple[str, str]] = []
    # Split on top-level commas (not inside brackets)
    parts = re.split(r",\s*(?=[a-zA-Z_]\w*\s*=)", input_str)
    for part in parts:
        match = re.match(r"([a-zA-Z_]\w*)\s*=\s*(.+)", part.strip(), re.DOTALL)
        if match:
            args.append((match.group(1).strip(), match.group(2).strip()))
    return args


def _method_accepts_call_without_args(code: str, method_name: str) -> bool:
    """Whether Solution.method can be called as `sol.method()` safely."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return False

    for node in ast.walk(tree):
        if not isinstance(node, ast.ClassDef) or node.name != "Solution":
            continue
        for item in node.body:
            if (
                not isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef))
                or item.name != method_name
            ):
                continue

            if item.args.posonlyargs:
                all_positional = [*item.args.posonlyargs, *item.args.args]
            else:
                all_positional = list(item.args.args)

            positional = all_positional[1:] if all_positional else []
            required_positional = max(0, len(positional) - len(item.args.defaults))
            required_kwonly = sum(1 for default in item.args.kw_defaults if default is None)
            return required_positional == 0 and required_kwonly == 0
    return False


def _generate_test(fixed_code: str, examples: list[str] | None, category: str) -> str | None:
    """Generate test_buggy.py for a DebugBench LeetCode-style case.

    Returns None if no usable tests can be generated.
    """
    # For syntax errors, just test that the module imports cleanly.
    if category == "syntax error":
        return textwrap.dedent("""\
            def test_syntax():
                import buggy  # noqa: F401
        """)

    method_name = _extract_method_name(fixed_code)
    if not method_name:
        return None

    parsed = _parse_examples(examples or [])
    if parsed:
        # Standard path: generate assertion tests from parsed examples.
        lines = ["from buggy import Solution", "", ""]
        for i, (input_str, expected) in enumerate(parsed):
            args = _parse_input_args(input_str)
            if not args:
                continue
            arg_values = ", ".join(v for _, v in args)
            lines.append(f"def test_case_{i}():")
            lines.append("    sol = Solution()")
            lines.append(f"    result = sol.{method_name}({arg_values})")
            lines.append(f"    assert result == {expected}")
            lines.append("")
            lines.append("")

        test_count = sum(1 for ln in lines if ln.startswith("def test_"))
        if test_count > 0:
            return "\n".join(lines).rstrip() + "\n"

    # Fallback: for reference errors, a smoke test that calls the method
    # is often enough — the bug will raise NameError on invocation.
    # This won't catch method-level bugs that only trigger with correct args,
    # but validation will filter those out.
    if category == "reference error" and _method_accepts_call_without_args(fixed_code, method_name):
        return textwrap.dedent(f"""\
            from buggy import Solution


            def test_smoke():
                sol = Solution()
                # Calling the method triggers runtime reference failures.
                sol.{method_name}()
        """)

    return None


# ---------------------------------------------------------------------------
# Infer exception type from buggy code
# ---------------------------------------------------------------------------


def _infer_exception(bug_type: str, bug_subtype: str, buggy_code: str) -> str:
    """Best-effort exception type inference from bug type and code."""
    bug_lower = bug_type.lower()
    subtype_lower = bug_subtype.lower()
    if "syntax" in bug_lower:
        if "indent" in subtype_lower:
            return "IndentationError"
        if "undefined" in subtype_lower or "misused" in subtype_lower:
            return "NameError"
        return "SyntaxError"
    if "reference" in bug_lower:
        return "NameError"
    if "type" in bug_lower:
        return "TypeError"
    return "AssertionError"


# ---------------------------------------------------------------------------
# Main import function
# ---------------------------------------------------------------------------


def import_debugbench(
    output_dir: Path,
    *,
    max_cases: int = 0,
    validate: bool = True,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """Import cases from DebugBench dataset.

    Returns list of imported case metadata dicts (or preview dicts if dry_run).
    """
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError(
            "The 'datasets' package is required for DebugBench import. "
            "Install with: pip install datasets"
        ) from None

    ds = load_dataset("Rtian/DebugBench", split="test")

    imported: list[dict[str, Any]] = []
    skipped = 0

    for idx, row in enumerate(ds):
        if max_cases > 0 and len(imported) >= max_cases:
            break

        if idx > 0 and idx % 100 == 0:
            print(
                f"  [debugbench] processed {idx} rows, imported {len(imported)}, skipped {skipped}"
            )

        # Extract fields — DebugBench uses "python3" as the language.
        language = str(row.get("language", "")).lower()
        if language != "python3":
            continue

        buggy_code = str(row.get("buggy_code", ""))
        fixed_code = str(row.get("solution", ""))
        bug_type = str(row.get("category", ""))
        bug_subtype = str(row.get("subtype", ""))
        difficulty = str(row.get("level", "medium")).lower()
        examples = row.get("examples")

        if not buggy_code.strip() or not fixed_code.strip():
            skipped += 1
            continue

        # Prepend standard LeetCode imports.
        buggy_code = _LEETCODE_PREAMBLE + buggy_code.lstrip("\n")
        fixed_code = _LEETCODE_PREAMBLE + fixed_code.lstrip("\n")

        # Map metadata early so we can check syntax_error before ruff.
        category = map_debugbench_type(bug_type)
        mapped_difficulty = difficulty if difficulty in _VALID_DIFFICULTIES else "medium"
        case_id = f"db_{idx:04d}_{mapped_difficulty}"

        # Skip if case already exists (don't overwrite).
        case_dir = output_dir / case_id
        if not dry_run and case_dir.exists():
            skipped += 1
            continue

        # Pre-flight check for syntax errors BEFORE ruff formatting
        # (ruff can't format syntactically invalid code).
        expected_exception = _infer_exception(bug_type, bug_subtype, buggy_code)
        if category == "syntax_error":
            try:
                compile(buggy_code, "<buggy>", "exec")
                skipped += 1
                continue
            except SyntaxError as exc:
                expected_exception = type(exc).__name__
            try:
                compile(fixed_code, "<fixed>", "exec")
            except SyntaxError:
                skipped += 1
                continue

        # Apply ruff formatting to both codes BEFORE generating the patch.
        # For syntax errors, only format the fixed code (buggy can't be parsed).
        if category != "syntax_error":
            buggy_code = ruff_format(buggy_code)
        fixed_code = ruff_format(fixed_code)

        # Apply filters.
        if has_external_deps(buggy_code) or has_external_deps(fixed_code):
            skipped += 1
            continue

        if not is_deterministic(buggy_code):
            skipped += 1
            continue

        if len(buggy_code.splitlines()) > MAX_CODE_LINES:
            skipped += 1
            continue

        # Generate oracle patch from (potentially ruff-formatted) code.
        oracle_patch = generate_oracle_patch(buggy_code, fixed_code)
        if not oracle_patch.strip():
            skipped += 1
            continue

        fix_lines = count_patch_lines(oracle_patch)
        if fix_lines > MAX_FIX_LINES:
            skipped += 1
            continue

        snapshot_dep = infer_snapshot_dependency(category)

        test_code = _generate_test(fixed_code, examples, bug_type)
        if test_code is None:
            skipped += 1
            continue

        metadata = {
            "description": f"DebugBench #{idx}: {bug_type}/{bug_subtype} ({difficulty})",
            "category": category,
            "difficulty": mapped_difficulty,
            "snapshot_dependency": snapshot_dep,
            "expected_exception": expected_exception,
            "tags": ["debugbench", bug_type.replace(" ", "_")],
            "python_args": ["-m", "pytest", "-q"],
            "timeout_sec": 30,
            "bug_source": "debugbench",
            "contamination_risk": "low",
            "expected_fix_lines": fix_lines,
            "source_id": f"debugbench_{idx}",
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
