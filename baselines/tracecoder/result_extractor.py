"""Extract repaired code from TraceCoder output and produce unified diffs."""

from __future__ import annotations

from pathlib import Path

from baselines.adapter_base import _render_unified_diff_text


def make_diff(original_path: Path, repaired_code: str) -> str | None:
    """Produce a unified diff between the original file and repaired code."""
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
    return "diff --git a/buggy.py b/buggy.py\n" + diff_text
