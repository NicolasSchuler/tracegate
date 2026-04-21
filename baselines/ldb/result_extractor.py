"""Extract repaired code from LDB output and produce unified diffs.

LDB returns repaired code as a string in its result dict.  This module
diffs it against the original ``buggy.py`` to produce a
``git apply``-compatible unified patch.
"""

from __future__ import annotations

from pathlib import Path

from baselines.adapter_base import _render_unified_diff_text


def make_diff(original_path: Path, repaired_code: str) -> str | None:
    """Produce a unified diff between the original file and repaired code.

    Parameters:
        original_path: Path to the original ``buggy.py``.
        repaired_code: Full repaired source code string.

    Returns:
        A ``git apply``-compatible unified diff string, or *None* if
        the code is identical to the original or repaired_code is empty.
    """
    if not repaired_code or not repaired_code.strip():
        return None

    original = original_path.read_text(encoding="utf-8")
    if original == repaired_code:
        return None

    diff_text = _render_unified_diff_text(
        fromfile="a/buggy.py",
        tofile="b/buggy.py",
        original_text=original,
        updated_text=repaired_code,
    )
    if not diff_text:
        return None

    header = "diff --git a/buggy.py b/buggy.py\n"
    return header + diff_text
