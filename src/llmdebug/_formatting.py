"""Shared text formatting helpers used by CLI and MCP server.

This is an internal module (leading underscore). Functions are public
so they can be imported by cli.py and mcp_server.py.
"""

from __future__ import annotations

from typing import Any, cast


def frame_file_label(frame: Any) -> str:
    """Return the best available file label for a stack frame.

    Args:
        frame: Frame payload; non-dict values are tolerated.

    Returns:
        ``file_rel`` if set, else the absolute ``file``, else ``"?"``.
    """
    if not isinstance(frame, dict):
        return "?"
    frame_d: dict[str, Any] = cast(dict[str, Any], frame)
    file_rel = frame_d.get("file_rel")
    if isinstance(file_rel, str) and file_rel:
        return file_rel
    file_abs = frame_d.get("file")
    if isinstance(file_abs, str) and file_abs:
        return file_abs
    return "?"


def build_source_text(source: dict[str, Any], crash_line: int) -> str:
    """Build a source snippet string with an arrow marker on the crash line.

    Args:
        source: Frame ``source`` payload whose ``snippet`` lists ``lineno``/``code`` entries.
        crash_line: Line number to mark with the arrow.

    Returns:
        Newline-joined ``marker lineno | code`` lines.
    """
    snippet = source.get("snippet", [])
    lines: list[str] = []
    for entry in snippet:
        lineno = entry.get("lineno", 0)
        code = entry.get("code", "")
        marker = " \u2192 " if lineno == crash_line else "   "
        lines.append(f"{marker}{lineno:>4} | {code}")
    return "\n".join(lines)


def format_local_value(key: str, value: Any, meta: dict[str, Any] | None) -> tuple[str, str, str]:
    """Format a single local variable for display.

    Array, DataFrame, and Series summaries render as their shape and dtype; other
    values are stringified and truncated to 120 characters.

    Args:
        key: Local variable name.
        value: Serialized local value.
        meta: Optional ``locals_meta`` entry; its ``type`` supplies the type label.

    Returns:
        A ``(name, type_str, value_str)`` tuple. ``type_str`` is the unqualified type
        name, the array kind (``"array"``, ``"dataframe"``, ``"series"``), or empty.
    """
    type_str = ""
    if meta:
        type_str = meta.get("type", "")
        # Shorten qualified types
        if "." in type_str:
            type_str = type_str.rsplit(".", 1)[-1]

    # Format array-like values specially
    if isinstance(value, dict):
        vd: dict[str, Any] = cast(dict[str, Any], value)
        for array_key in ("__array__", "__dataframe__", "__series__"):
            if vd.get(array_key):
                dtype: Any = vd.get("dtype", "")
                shape: Any = vd.get("shape", cast(list[Any], []))
                shape_str = str(shape)
                extra = f" dtype={dtype}" if dtype else ""
                return key, array_key.strip("_"), f"{shape_str}{extra}"

    value_str = str(cast(Any, value)) if not isinstance(value, str) else value
    if len(value_str) > 120:
        value_str = value_str[:117] + "..."
    return key, type_str, value_str


def format_git_head_section(head: Any) -> list[str]:
    """Format git HEAD info as plain text lines.

    Args:
        head: ``head`` entry of the enhanced git context.

    Returns:
        Section lines, or an empty list when *head* is not a dict.
    """
    if not isinstance(head, dict):
        return []
    head_d: dict[str, Any] = cast(dict[str, Any], head)
    commit = str(head_d.get("commit") or "?")
    branch = str(head_d.get("branch") or "?")
    dirty = "true" if bool(head_d.get("dirty")) else "false"
    detached = "true" if bool(head_d.get("detached")) else "false"
    return [
        "## Git Head",
        f"commit: {commit}",
        f"branch: {branch}",
        f"dirty: {dirty}",
        f"detached: {detached}",
    ]


def format_crash_file_section(crash_file: Any) -> list[str]:
    """Format crash file blame/location as plain text lines.

    Args:
        crash_file: ``crash_file`` entry of the enhanced git context.

    Returns:
        Section lines, or an empty list when *crash_file* is not a dict.
    """
    if not isinstance(crash_file, dict):
        return []
    cf: dict[str, Any] = cast(dict[str, Any], crash_file)
    lines = [
        "",
        "## Crash File",
        f"file_rel: {cf.get('file_rel') or '?'!s}",
    ]
    line: Any = cf.get("line")
    if line is not None:
        lines.append(f"line: {line}")
    in_repo = "true" if bool(cf.get("in_repo")) else "false"
    lines.append(f"in_repo: {in_repo}")
    blame = cf.get("blame")
    if isinstance(blame, dict):
        blame_d: dict[str, Any] = cast(dict[str, Any], blame)
        lines.extend(
            (
                "blame:",
                f"  commit: {blame_d.get('commit', '?')}",
                f"  author: {blame_d.get('author', '?')}",
                f"  author_time: {blame_d.get('author_time', '?')}",
                f"  summary: {blame_d.get('summary', '?')}",
            )
        )
    return lines


def format_recent_commits_section(recent_commits: Any) -> list[str]:
    """Format recent commit metadata as plain text lines.

    Args:
        recent_commits: ``recent_commits`` list of the enhanced git context.

    Returns:
        Section lines, or an empty list when there are no commits.
    """
    if not isinstance(recent_commits, list) or not recent_commits:
        return []
    lines = ["", "## Recent Commits"]
    for idx, entry in enumerate(cast(list[Any], recent_commits), 1):
        if not isinstance(entry, dict):
            continue
        entry_d: dict[str, Any] = cast(dict[str, Any], entry)
        commit = str(entry_d.get("commit") or "?")
        author = str(entry_d.get("author") or "?")
        when = str(entry_d.get("author_time") or "?")
        summary = str(entry_d.get("summary") or "")
        files = int(entry_d.get("files_changed") or 0)
        ins = int(entry_d.get("insertions") or 0)
        dele = int(entry_d.get("deletions") or 0)
        lines.append(f"{idx}. {commit} {when} {author}")
        lines.append(f"   {summary}")
        lines.append(f"   stats: files={files} +{ins} -{dele}")
    return lines


def format_crash_diffstat_section(diffstat: Any) -> list[str]:
    """Format crash file diffstat as plain text lines.

    Args:
        diffstat: ``crash_file_diffstat`` entry of the enhanced git context.

    Returns:
        Section lines, or an empty list when *diffstat* is not a dict.
    """
    if not isinstance(diffstat, dict):
        return []
    ds: dict[str, Any] = cast(dict[str, Any], diffstat)
    touched = "true" if bool(ds.get("touched")) else "false"
    return [
        "",
        "## Crash File Diffstat",
        f"files_changed: {int(ds.get('files_changed') or 0)}",
        f"insertions: {int(ds.get('insertions') or 0)}",
        f"deletions: {int(ds.get('deletions') or 0)}",
        f"touched: {touched}",
    ]


def format_git_warnings_section(warnings: Any) -> list[str]:
    """Format git warning lines.

    Args:
        warnings: ``_git_warnings`` list of the enhanced git context.

    Returns:
        Section lines, or an empty list when there are no warnings.
    """
    if not isinstance(warnings, list) or not warnings:
        return []
    warnings_l: list[Any] = cast(list[Any], warnings)
    lines = ["", "## Warnings"]
    lines.extend(f"- {warning}" for warning in warnings_l)
    return lines


def format_enhanced_git_context_text(context: dict[str, Any]) -> str:
    """Render enhanced git metadata as plain text.

    Args:
        context: Enhanced git context with optional ``head``, ``crash_file``,
            ``recent_commits``, ``crash_file_diffstat``, and ``_git_warnings`` entries.

    Returns:
        The joined sections, or a placeholder message when no section has content.
    """
    lines: list[str] = []
    sections = (
        format_git_head_section(context.get("head")),
        format_crash_file_section(context.get("crash_file")),
        format_recent_commits_section(context.get("recent_commits")),
        format_crash_diffstat_section(context.get("crash_file_diffstat")),
        format_git_warnings_section(context.get("_git_warnings")),
    )
    for section in sections:
        lines.extend(section)

    if not lines:
        return "No enhanced git metadata available."
    return "\n".join(lines)
