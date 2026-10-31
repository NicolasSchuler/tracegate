"""Pure helpers for MCP git-context tool normalization."""

from __future__ import annotations

from typing import Any, cast


def parse_max_commits(max_commits: Any) -> int:
    """Parse and clamp *max_commits* into the supported ``[1, 50]`` range.

    Args:
        max_commits: Requested number of recent commits, as an int or any value
            ``int()`` accepts.

    Returns:
        The parsed count, clamped to the range 1 to 50.

    Raises:
        ValueError: If *max_commits* cannot be converted to an integer.
    """
    try:
        parsed = int(max_commits)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Invalid max_commits value: {max_commits!r}. Expected an integer between 1 and 50."
        ) from exc
    return max(1, min(parsed, 50))


def collect_git_evidence_refs(enhanced: dict[str, Any]) -> list[str]:
    """Extract RCA evidence refs from an enhanced git metadata payload.

    Args:
        enhanced: Enhanced git context with optional ``head`` and ``crash_file``
            blocks, as returned by ``get_enhanced_git_context``.

    Returns:
        Refs for the head commit, branch, crash file, and crash-line blame commit, in
        that order; missing or empty values are skipped.
    """
    evidence_refs: list[str] = []
    head_raw = enhanced.get("head")
    if isinstance(head_raw, dict):
        head = cast(dict[str, Any], head_raw)
        commit = str(head.get("commit") or "").strip()
        branch = str(head.get("branch") or "").strip()
        if commit:
            evidence_refs.append(f"git_head_commit:{commit}")
        if branch:
            evidence_refs.append(f"git_branch:{branch}")

    crash_file_raw = enhanced.get("crash_file")
    if isinstance(crash_file_raw, dict):
        crash_file = cast(dict[str, Any], crash_file_raw)
        file_rel = str(crash_file.get("file_rel") or "").strip()
        if file_rel:
            evidence_refs.append(f"git_crash_file:{file_rel}")
        blame_raw = crash_file.get("blame")
        if isinstance(blame_raw, dict):
            blame = cast(dict[str, Any], blame_raw)
            blame_commit = str(blame.get("commit") or "").strip()
            if blame_commit:
                evidence_refs.append(f"git_blame_commit:{blame_commit}")

    return evidence_refs
