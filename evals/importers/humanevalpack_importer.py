"""HumanEvalPack dataset importer for llmdebug framework.

Source: HuggingFace ``bigcode/humanevalpack`` (Apache-2.0 license).
164 Python bug-fix tasks with buggy_solution, canonical_solution, test,
bug_type, and failure_symptoms.

contamination_risk = "high" (this benchmark is extremely well-known).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from evals.category_mapping import infer_snapshot_dependency, map_humanevalpack_type
from evals.importers.base import (
    count_patch_lines,
    generate_oracle_patch,
    validate_generated_case,
    write_case_files,
)

# ---------------------------------------------------------------------------
# Filter criteria
# ---------------------------------------------------------------------------

MAX_FIX_LINES = 15
MAX_CODE_LINES = 150


# ---------------------------------------------------------------------------
# Test generation
# ---------------------------------------------------------------------------


def _generate_test(entry_point: str, test_code: str, prompt: str) -> str:
    """Generate test_buggy.py from test string and entry point."""
    lines = [
        "from buggy import *",
        "",
        "",
    ]

    # The test code usually defines a check() function.
    if "def check" in test_code:
        lines.append(test_code.rstrip())
        lines.append("")
        lines.append("")
        lines.append("def test_main():")
        lines.append(f"    check({entry_point})")
        lines.append("")
    else:
        lines.append("def test_main():")
        lines.extend(f"    {line}" for line in test_code.splitlines())
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Exception inference
# ---------------------------------------------------------------------------


def _infer_exception(bug_type: str, failure_symptoms: str) -> str:
    """Infer expected exception from bug_type and failure info."""
    symptoms_lower = failure_symptoms.lower() if failure_symptoms else ""
    if "assertion" in symptoms_lower:
        return "AssertionError"
    if "type" in symptoms_lower:
        return "TypeError"
    if "index" in symptoms_lower:
        return "IndexError"
    if "name" in symptoms_lower or "attribute" in symptoms_lower:
        return "NameError"
    return "AssertionError"


# ---------------------------------------------------------------------------
# Main import function
# ---------------------------------------------------------------------------


def import_humanevalpack(
    output_dir: Path,
    *,
    max_cases: int = 20,
    validate: bool = True,
    dry_run: bool = False,
) -> list[dict[str, Any]]:
    """Import cases from HumanEvalPack dataset.

    Returns list of imported case metadata dicts (or preview dicts if dry_run).
    """
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError(
            "The 'datasets' package is required. Install with: pip install datasets"
        ) from None

    ds = load_dataset("bigcode/humanevalpack", "python", split="test")

    imported: list[dict[str, Any]] = []
    skipped = 0

    for row in ds:
        if len(imported) >= max_cases:
            break

        task_id = str(row.get("task_id", ""))
        buggy_code = str(row.get("buggy_solution", ""))
        canonical = str(row.get("canonical_solution", ""))
        prompt = str(row.get("prompt", ""))
        test_code = str(row.get("test", ""))
        bug_type = str(row.get("bug_type", ""))
        entry_point = str(row.get("entry_point", ""))
        failure_symptoms = str(row.get("failure_symptoms", ""))

        if not buggy_code.strip() or not canonical.strip():
            skipped += 1
            continue

        # Combine prompt + solution to form full code.
        full_buggy = prompt + buggy_code
        full_fixed = prompt + canonical

        oracle_patch = generate_oracle_patch(full_buggy, full_fixed)
        if not oracle_patch.strip():
            skipped += 1
            continue

        fix_lines = count_patch_lines(oracle_patch)
        if fix_lines > MAX_FIX_LINES:
            skipped += 1
            continue

        if len(full_buggy.splitlines()) > MAX_CODE_LINES:
            skipped += 1
            continue

        if not test_code.strip():
            skipped += 1
            continue

        # Map metadata.
        category = map_humanevalpack_type(bug_type)
        snapshot_dep = infer_snapshot_dependency(category)
        expected_exception = _infer_exception(bug_type, failure_symptoms)

        safe_task_id = task_id.replace("/", "_").replace(" ", "_").lower()
        safe_bug_type = bug_type.replace(" ", "_").lower()
        case_id = f"hep_{safe_task_id}_{safe_bug_type}"

        generated_test = _generate_test(entry_point, test_code, prompt)

        metadata = {
            "description": f"HumanEvalPack {task_id}: {bug_type}",
            "category": category,
            "difficulty": "medium",
            "snapshot_dependency": snapshot_dep,
            "expected_exception": expected_exception,
            "tags": ["humanevalpack", safe_bug_type],
            "python_args": ["-m", "pytest", "-q"],
            "timeout_sec": 30,
            "bug_source": "humanevalpack",
            "contamination_risk": "high",
            "expected_fix_lines": fix_lines,
            "source_id": task_id,
        }

        case_info = {"case_id": case_id, "metadata": metadata, "fix_lines": fix_lines}

        if dry_run:
            imported.append(case_info)
            continue

        case_dir = output_dir / case_id
        write_case_files(
            case_dir,
            buggy_code=full_buggy,
            test_code=generated_test,
            oracle_patch=oracle_patch,
            metadata=metadata,
        )

        if validate:
            ok, msg = validate_generated_case(case_dir, schema_only=True)
            if not ok:
                print(f"  [skip] {case_id}: validation failed: {msg}")
                skipped += 1
                import shutil

                shutil.rmtree(case_dir, ignore_errors=True)
                continue

        imported.append(case_info)

    return imported
