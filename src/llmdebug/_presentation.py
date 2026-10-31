"""Shared plain-text presentation helpers for CLI and MCP surfaces."""

from __future__ import annotations

from typing import Any, cast

from ._formatting import build_source_text, format_local_value, frame_file_label


def format_local_value_line(key: str, value: Any, meta: dict[str, Any] | None) -> str:
    """Format a single local variable as one plain-text line.

    Args:
        key: Local variable name.
        value: Serialized local value.
        meta: Optional ``locals_meta`` entry for the local; only its ``type`` is used.

    Returns:
        Indented ``name = value  (type)`` line; the type suffix is omitted when unknown.
    """
    name, type_str, value_str = format_local_value(key, value, meta)
    type_part = f"  ({type_str})" if type_str else ""
    return f"  {name} = {value_str}{type_part}"


def format_frame_text(
    frame: dict[str, Any],
    index: int,
    is_crash: bool,
    *,
    show_locals: bool = True,
    crash_tag: bool = True,
) -> str:
    """Format a stack frame as plain text.

    Args:
        frame: Frame payload to format.
        index: Frame index.
        is_crash: Whether this is the crash frame.
        show_locals: Include local variable section.
        crash_tag: Prefix crash frame title with ``[CRASH]``.

    Returns:
        Multi-line text with a title line, optional source block, and optional
        locals block (pytest ``@py_*`` rewrite locals are skipped).
    """
    file_rel = frame_file_label(frame)
    line = frame.get("line", "?")
    func = frame.get("function", "?")
    prefix = "[CRASH] " if (is_crash and crash_tag) else ""

    parts: list[str] = [f"{prefix}Frame #{index}: {file_rel}:{line} in {func}"]

    source = frame.get("source")
    line_value = frame.get("line", 0)
    try:
        crash_line = int(line_value)
    except (TypeError, ValueError):
        crash_line = 0
    if isinstance(source, dict) and source:
        parts.append("")
        parts.append("Source:")
        parts.append(build_source_text(cast(dict[str, Any], source), crash_line))

    locals_dict = frame.get("locals", {})
    locals_meta = frame.get("locals_meta", {})
    if show_locals and isinstance(locals_dict, dict) and locals_dict:
        locals_typed = cast(dict[str, Any], locals_dict)
        locals_meta_typed = (
            cast(dict[str, Any], locals_meta) if isinstance(locals_meta, dict) else {}
        )
        parts.append("")
        parts.append("Locals:")
        for key, value in locals_typed.items():
            if str(key).startswith("@py_"):
                continue
            parts.append(
                format_local_value_line(
                    str(key),
                    value,
                    locals_meta_typed.get(str(key)),
                )
            )

    return "\n".join(parts)


__all__ = ["format_frame_text", "format_local_value_line"]
