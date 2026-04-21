"""Repair broken DebugBench oracle patches.

After initial import, ruff reformatted buggy.py (e.g. ``List[int]`` → ``list[int]``,
``k-1`` → ``k - 1``), but oracle patches weren't regenerated.  This script fetches
the original fixed code from HuggingFace, applies ruff formatting to both buggy and
fixed code, and regenerates oracle.patch for each broken case.

Usage::

    PYTHONPATH=. uv run python evals/importers/repair_debugbench_patches.py
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from evals.importers.base import count_patch_lines, generate_oracle_patch, ruff_format

from .. import _json as json

# Same preamble used during original import.
_LEETCODE_PREAMBLE = """\
from typing import List, Optional, Tuple, Dict, Set
from collections import defaultdict, Counter, deque
import heapq
import math
import bisect
import itertools
import functools
"""

CASES_DIR = Path("evals/cases")


def _patch_applies(case_dir: Path) -> bool:
    """Check whether oracle.patch applies cleanly to buggy.py."""
    try:
        result = subprocess.run(
            ["patch", "--dry-run", "-p1"],
            input=(case_dir / "oracle.patch").read_text(encoding="utf-8"),
            capture_output=True,
            text=True,
            cwd=case_dir,
            timeout=10,
        )
    except (subprocess.TimeoutExpired, OSError):
        return False
    return result.returncode == 0


def repair_debugbench_patches(cases_dir: Path | None = None) -> dict[str, str]:
    """Repair broken oracle patches for DebugBench cases.

    Returns a dict mapping case_id -> status ("repaired", "already_ok", "failed: ...").
    """
    try:
        from datasets import load_dataset  # type: ignore[import-untyped]
    except ImportError:
        raise ImportError(
            "The 'datasets' package is required. Install with: pip install datasets"
        ) from None

    cases_dir = cases_dir or CASES_DIR

    # Load the DebugBench dataset once.
    print("Loading DebugBench dataset from HuggingFace...")
    ds = load_dataset("Rtian/DebugBench", split="test")

    # Build index: row_idx -> row for quick lookup.
    row_index: dict[int, dict] = {
        idx: row for idx, row in enumerate(ds) if str(row.get("language", "")).lower() == "python3"
    }

    results: dict[str, str] = {}

    # Find all db_* case directories.
    db_dirs = sorted(cases_dir.glob("db_*"))
    print(f"Found {len(db_dirs)} DebugBench case directories.")

    for case_dir in db_dirs:
        case_id = case_dir.name

        # Check if patch already applies.
        if _patch_applies(case_dir):
            results[case_id] = "already_ok"
            continue

        # Read case.json to get source_id.
        case_json_path = case_dir / "case.json"
        if not case_json_path.exists():
            results[case_id] = "failed: missing case.json"
            continue

        meta = json.loads(case_json_path.read_text(encoding="utf-8"))
        source_id = meta.get("source_id", "")

        # Parse row index from source_id (e.g. "debugbench_2910" -> 2910).
        if not source_id.startswith("debugbench_"):
            results[case_id] = f"failed: unexpected source_id={source_id}"
            continue

        try:
            row_idx = int(source_id.split("_")[1])
        except (IndexError, ValueError):
            results[case_id] = f"failed: can't parse row index from {source_id}"
            continue

        row = row_index.get(row_idx)
        if row is None:
            results[case_id] = f"failed: row {row_idx} not in dataset"
            continue

        # Get original buggy and fixed code, apply preamble.
        buggy_raw = _LEETCODE_PREAMBLE + str(row.get("buggy_code", "")).lstrip("\n")
        fixed_raw = _LEETCODE_PREAMBLE + str(row.get("solution", "")).lstrip("\n")

        # Format both through ruff (matches what ruff did to buggy.py on disk).
        buggy_formatted = ruff_format(buggy_raw)
        fixed_formatted = ruff_format(fixed_raw)

        # Regenerate oracle patch from formatted versions.
        new_patch = generate_oracle_patch(buggy_formatted, fixed_formatted)
        if not new_patch.strip():
            results[case_id] = "failed: empty patch after ruff formatting"
            continue

        # Also update buggy.py to match the ruff-formatted version
        # (it should already match, but ensure consistency).
        meta["expected_fix_lines"] = count_patch_lines(new_patch)
        (case_dir / "buggy.py").write_text(buggy_formatted, encoding="utf-8")
        (case_dir / "oracle.patch").write_text(new_patch, encoding="utf-8")
        case_json_path.write_text(
            json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )

        # Verify the new patch applies.
        if _patch_applies(case_dir):
            results[case_id] = "repaired"
            print(f"  [repaired] {case_id}")
        else:
            results[case_id] = "failed: new patch still doesn't apply"
            print(f"  [FAILED]   {case_id}: patch still broken after regeneration")

    return results


def main() -> None:
    results = repair_debugbench_patches()

    repaired = sum(1 for v in results.values() if v == "repaired")
    ok = sum(1 for v in results.values() if v == "already_ok")
    failed = sum(1 for v in results.values() if v.startswith("failed"))

    print(f"\nResults: {repaired} repaired, {ok} already ok, {failed} failed")
    if failed > 0:
        for case_id, status in sorted(results.items()):
            if status.startswith("failed"):
                print(f"  {case_id}: {status}")


if __name__ == "__main__":
    main()
