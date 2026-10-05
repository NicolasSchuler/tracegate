"""CLI display and rendering helpers for terminal output.

This is an internal module (leading underscore).  It contains all the Rich
and plain-text rendering logic used by ``cli.py`` Click commands.  The
module depends on core formatting helpers (``_formatting``, ``_presentation``,
``_debug_session``) but never imports from ``cli.py``.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Any, cast

import click

from . import _json as json
from ._debug_session import coerce_crash_frame_index
from ._formatting import (
    build_source_text,
    format_local_value,
    frame_file_label,
)
from ._presentation import format_frame_text

# Rich is optional — degrade gracefully to plain text.
if TYPE_CHECKING:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

HAS_RICH: bool = False
try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    HAS_RICH = True  # pyright: ignore[reportConstantRedefinition]
except ImportError:
    pass


# ---------------------------------------------------------------------------
# TTY / console detection
# ---------------------------------------------------------------------------


def _stdout_is_tty() -> bool:
    """Return True when stdout is attached to an interactive terminal."""
    stream = click.get_text_stream("stdout")
    isatty = getattr(stream, "isatty", None)
    if not callable(isatty):
        return False
    try:
        return bool(isatty())
    except Exception:
        return False


def _get_console() -> Any | None:  # pyright: ignore[reportUnusedFunction] — used by cli.py
    """Return a Rich Console if available, else None."""
    if HAS_RICH and _stdout_is_tty():
        return Console(stderr=False, no_color=(os.getenv("NO_COLOR") is not None))
    return None


# ---------------------------------------------------------------------------
# Text truncation utilities
# ---------------------------------------------------------------------------


def _truncate_text(value: Any, max_len: int) -> str:
    """Truncate text to `max_len` using ASCII ellipsis."""
    text = str(value)
    if max_len <= 0:
        return ""
    if len(text) <= max_len:
        return text
    if max_len <= 3:
        return text[:max_len]
    return text[: max_len - 3] + "..."


def _truncate_tail(value: Any, max_len: int) -> str:  # pyright: ignore[reportUnusedFunction] — used by cli.py
    """Truncate long paths/locations from the left to preserve the tail."""
    text = str(value)
    if max_len <= 0:
        return ""
    if len(text) <= max_len:
        return text
    if max_len <= 3:
        return text[-max_len:]
    return "..." + text[-(max_len - 3) :]


# ---------------------------------------------------------------------------
# Frame iteration
# ---------------------------------------------------------------------------


def _iter_valid_frame_items(snapshot: dict[str, Any]) -> list[tuple[int, dict[str, Any]]]:
    """Return `(index, frame)` pairs for dict-valued frames only."""
    frames = snapshot.get("frames")
    if not isinstance(frames, list):
        return []
    frames_list = cast(list[Any], frames)
    return [
        (i, cast(dict[str, Any], frame))
        for i, frame in enumerate(frames_list)
        if isinstance(frame, dict)
    ]


# ---------------------------------------------------------------------------
# JSON / detail helpers
# ---------------------------------------------------------------------------


def _emit_json_payload(payload: dict[str, Any]) -> None:  # pyright: ignore[reportUnusedFunction] — used by cli.py
    """Write a JSON payload to stdout using stable CLI formatting."""
    click.echo(json.dumps(payload, indent=2, ensure_ascii=False, default=str))


def _resolve_show_detail(detail: str | None, full: bool) -> str:  # pyright: ignore[reportUnusedFunction] — used by cli.py
    """Resolve the effective detail level for `show`."""
    if detail is not None:
        return detail
    return "full" if full else "crash"


# ---------------------------------------------------------------------------
# Rich display helpers
# ---------------------------------------------------------------------------


def _print_exception_panel(snapshot: dict[str, Any], console: Any) -> None:  # pyright: ignore[reportUnusedFunction] — used by cli.py
    """Print exception info as a Rich panel or plain text."""
    exc_raw = snapshot.get("exception")
    exc = cast(dict[str, Any], exc_raw) if isinstance(exc_raw, dict) else {}
    exc_type = exc.get("type", "UnknownError")
    exc_msg = exc.get("message", "")

    if console:
        title_text = Text(f"{exc_type}: {exc_msg}", style="bold red")
        console.print(Panel(title_text, title="Exception", border_style="red"))
    else:
        click.echo(f">>> Exception: {exc_type}: {exc_msg}")


def _print_selected_frame(
    frame: dict[str, Any],
    index: int,
    is_crash: bool,
    console: Any,
) -> None:
    """Render a single frame with the active rich/plain presenter."""
    if console:
        _print_frame_rich(frame, index, is_crash, console)
        return
    _print_frame(frame, index, is_crash)


def _print_remaining_frames_hint(snapshot: dict[str, Any], console: Any) -> None:
    """Show a hint when crash-only mode hides additional frames."""
    all_frames_obj = snapshot.get("frames", [])
    all_frames: list[Any] = (
        cast(list[Any], all_frames_obj) if isinstance(all_frames_obj, list) else cast(list[Any], [])
    )
    if len(all_frames) <= 1:
        return
    remaining = len(all_frames) - 1
    plural = "s" if remaining != 1 else ""
    hint = f"  ({remaining} more frame{plural}; --detail full)"
    if _stdout_is_tty() and console:
        console.print(Text(hint, style="dim"))
    elif _stdout_is_tty():
        click.echo(hint)


def _print_crash_frame_only(
    frame_items: list[tuple[int, dict[str, Any]]],
    crash_idx: int,
    snapshot: dict[str, Any],
    console: Any,
) -> None:
    """Render only the crash frame and an optional hidden-frame hint."""
    if not frame_items:
        return
    crash_pair = next(
        ((i, frame) for i, frame in frame_items if i == crash_idx),
        frame_items[0],
    )
    idx, crash_frame = crash_pair
    _print_selected_frame(crash_frame, idx, True, console)
    _print_remaining_frames_hint(snapshot, console)


def _print_filtered_frames(  # pyright: ignore[reportUnusedFunction] — used by cli.py
    filtered: dict[str, Any],
    snapshot: dict[str, Any],
    detail: str,
    console: Any,
) -> None:
    """Render frames for the requested detail level."""
    frame_items = _iter_valid_frame_items(filtered)
    crash_idx = coerce_crash_frame_index(filtered)
    if detail in ("full", "context"):
        for i, frame in frame_items:
            _print_selected_frame(frame, i, i == crash_idx, console)
        return
    _print_crash_frame_only(frame_items, crash_idx, snapshot, console)


def _print_context_if_requested(filtered: dict[str, Any], detail: str, console: Any) -> None:  # pyright: ignore[reportUnusedFunction] — used by cli.py
    """Render context sections when `show --detail context` is active."""
    if detail != "context":
        return
    if console:
        console.print()
    else:
        click.echo()
    _print_context_sections(filtered, console)


def _print_show_footer(snapshot: dict[str, Any], console: Any) -> None:  # pyright: ignore[reportUnusedFunction] — used by cli.py
    """Render the trailing footer with the existing spacing rules."""
    if console:
        console.print()
    _print_footer(snapshot, console)


def _print_frame(
    frame: dict[str, Any],
    index: int,
    is_crash: bool,
    *,
    show_locals: bool = True,
) -> None:
    """Print a single stack frame as plain text; Rich output is ``_print_frame_rich``."""
    rendered = format_frame_text(frame, index, is_crash, show_locals=show_locals, crash_tag=False)
    lines = rendered.splitlines()
    if not lines:
        return
    prefix = ">>> " if is_crash else "    "
    click.echo(f"{prefix}{lines[0]}")
    for line in lines[1:]:
        click.echo(f"    {line}")


def _print_frame_rich(
    frame: dict[str, Any],
    index: int,
    is_crash: bool,
    console: Any,
    *,
    show_locals: bool = True,
) -> None:
    """Print a single stack frame with Rich rendering (separate source and locals panels for ``frames``)."""
    file_rel = frame_file_label(frame)
    line = frame.get("line", "?")
    func = frame.get("function", "?")
    location = f"{file_rel}:{line} in {func}"

    source_raw = frame.get("source")
    source = cast(dict[str, Any], source_raw) if isinstance(source_raw, dict) else {}
    source_text = build_source_text(source, frame.get("line", 0)) if source else ""

    locals_dict = frame.get("locals", {})
    locals_meta_raw = frame.get("locals_meta")
    locals_meta = cast(dict[str, Any], locals_meta_raw) if isinstance(locals_meta_raw, dict) else {}

    border = "red" if is_crash else "dim"
    title = f"{'[CRASH] ' if is_crash else ''}Frame #{index}: {location}"

    # Source
    if source_text:
        console.print(Panel(Text(source_text), title=title, border_style=border))
    else:
        code = frame.get("code", "")
        console.print(Panel(Text(f"  {code}"), title=title, border_style=border))

    # Locals table
    if show_locals and isinstance(locals_dict, dict) and locals_dict:
        ld = cast(dict[str, Any], locals_dict)
        lm = locals_meta
        table = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
        table.add_column("Variable", style="cyan", no_wrap=True)
        table.add_column("Type", style="dim")
        table.add_column("Value")

        for k, v in ld.items():
            if k.startswith("@py_"):
                continue
            name, type_str, val_str = format_local_value(k, v, lm.get(k))
            # Highlight arrays with shape[0]==0 in red — empty arrays are a
            # frequent root cause of downstream crashes (e.g. mean-of-empty).
            array_keys = ("__array__", "__dataframe__", "__series__")
            if (
                HAS_RICH
                and isinstance(v, dict)
                and any(cast(dict[str, Any], v).get(ak) for ak in array_keys)
            ):
                shape = cast(dict[str, Any], v).get("shape", cast(list[Any], []))
                if shape and cast(list[Any], shape)[0] == 0:
                    val_str = f"[red]{val_str}[/red]"
            table.add_row(name, type_str, val_str)

        console.print(table)


def _print_footer(snapshot: dict[str, Any], console: Any) -> None:
    """Print git info and repro command footer."""
    parts: list[str] = []

    git_raw = snapshot.get("git")
    git = cast(dict[str, Any], git_raw) if isinstance(git_raw, dict) else {}
    if git:
        commit = git.get("commit", "")
        branch = git.get("branch", "")
        dirty = " *dirty*" if git.get("dirty") else ""
        if commit:
            parts.append(f"git: {commit} {branch}{dirty}")

    pt_raw = snapshot.get("pytest")
    pt = cast(dict[str, Any], pt_raw) if isinstance(pt_raw, dict) else {}
    repro = pt.get("repro")
    if repro and isinstance(repro, list):
        cmd_parts: list[str] = []
        for r in cast(list[Any], repro):
            r_str = str(r)
            if r_str.endswith("/python3") or r_str.endswith("/python"):
                cmd_parts.append("python")
            else:
                cmd_parts.append(r_str)
        parts.append(f"repro: {' '.join(cmd_parts)}")

    if not parts:
        return

    footer = " | ".join(parts)
    if console:
        console.print(Text(footer, style="dim"))
    else:
        click.echo(f"--- {footer}")


# ---------------------------------------------------------------------------
# Context section rendering
# ---------------------------------------------------------------------------


def _render_metadata_value(value: Any) -> str:
    """Render context metadata as a compact single-line string."""
    if isinstance(value, list):
        value_list = cast(list[Any], value)
        if all(not isinstance(item, (dict, list)) for item in value_list):
            rendered = " ".join(str(item) for item in value_list)
        else:
            rendered = json.dumps(value_list, ensure_ascii=False, default=str)
    elif isinstance(value, dict):
        rendered = json.dumps(value, ensure_ascii=False, default=str)
    else:
        rendered = str(value)
    return _truncate_text(rendered, 200)


def _context_block_lines(title: str, payload: Any, keys: tuple[str, ...]) -> list[str]:
    """Build a plain-text block for a context mapping."""
    if not isinstance(payload, dict):
        return []
    payload_d = cast(dict[str, Any], payload)
    lines = [f"## {title}"]
    for key in keys:
        value = payload_d.get(key)
        if value in (None, "", [], {}):
            continue
        lines.append(f"{key}: {_render_metadata_value(value)}")
    return lines if len(lines) > 1 else []


def _print_context_sections(snapshot: dict[str, Any], console: Any) -> None:
    """Render the extra context exposed by ``--detail context`` in text mode."""
    blocks: list[list[str]] = []
    blocks.append(
        _context_block_lines(
            "Pytest",
            snapshot.get("pytest"),
            ("nodeid", "outcome", "when", "params", "xdist_worker", "flaky"),
        )
    )
    blocks.append(
        _context_block_lines(
            "Environment",
            snapshot.get("env"),
            ("python", "platform", "cwd", "executable", "argv"),
        )
    )
    blocks.append(
        _context_block_lines(
            "Capture Config",
            snapshot.get("capture_config"),
            (
                "redaction_profile",
                "output_format",
                "frames",
                "source_mode",
                "locals_mode",
                "include_git",
                "include_coverage",
            ),
        )
    )

    capture_warnings = snapshot.get("_capture_warnings")
    if isinstance(capture_warnings, list) and capture_warnings:
        warning_lines = ["## Capture Warnings"]
        warning_lines.extend(f"- {warning}" for warning in cast(list[Any], capture_warnings))
        blocks.append(warning_lines)

    pytest_payload = snapshot.get("pytest")
    if isinstance(pytest_payload, dict):
        plugin_warnings = cast(dict[str, Any], pytest_payload).get("_plugin_warnings")
        if isinstance(plugin_warnings, list) and plugin_warnings:
            warning_lines = ["## Pytest Capture Warnings"]
            warning_lines.extend(f"- {warning}" for warning in cast(list[Any], plugin_warnings))
            blocks.append(warning_lines)

    rendered_blocks = [block for block in blocks if block]
    if not rendered_blocks:
        return

    for idx, block in enumerate(rendered_blocks):
        text = "\n".join(block)
        if console:
            if idx > 0:
                console.print()
            console.print(Text(text))
        else:
            if idx > 0:
                click.echo()
            click.echo(text)


# ---------------------------------------------------------------------------
# Diff rendering
# ---------------------------------------------------------------------------


def _print_diff_rich(result: Any, console: Any) -> None:  # pyright: ignore[reportUnusedFunction] — used by cli.py
    """Render a SnapshotDiffResult using Rich panels and tables."""
    from .snapshot_diff import _truncate  # pyright: ignore[reportPrivateUsage]

    # Header panel
    header = Text()
    header.append("old: ", style="dim")
    header.append(result.old_name, style="bold")
    header.append("  new: ", style="dim")
    header.append(result.new_name, style="bold")
    if not result.same_test:
        header.append("  (different test)", style="yellow")
    console.print(Panel(header, title="Snapshot Diff", border_style="cyan"))

    # Exception
    exc = result.exception
    if exc.changed:
        body = Text()
        body.append("CHANGED: ", style="bold yellow")
        body.append(exc.old_type, style="red")
        body.append(" -> ", style="dim")
        body.append(exc.new_type, style="green")
        if exc.old_message != exc.new_message:
            body.append(f"\n  old: {_truncate(exc.old_message)}", style="red")
            body.append(f"\n  new: {_truncate(exc.new_message)}", style="green")
        if exc.old_category != exc.new_category:
            body.append(f"\n  category: {exc.old_category} -> {exc.new_category}", style="dim")
        console.print(Panel(body, title="Exception", border_style="yellow"))
    else:
        body = Text()
        body.append("UNCHANGED: ", style="dim")
        body.append(f"{exc.old_type}: {_truncate(exc.old_message)}")
        console.print(Panel(body, title="Exception", border_style="dim"))

    # Location
    loc = result.location
    if loc.changed:
        body = Text()
        body.append("MOVED: ", style="bold yellow")
        body.append(loc.old_location, style="red")
        body.append(" -> ", style="dim")
        body.append(loc.new_location, style="green")
        console.print(Panel(body, title="Crash Location", border_style="yellow"))
    else:
        console.print(
            Panel(
                Text(f"UNCHANGED: {loc.old_location}", style="dim"),
                title="Crash Location",
                border_style="dim",
            )
        )

    # Locals table
    ld = result.locals_diff
    if ld.changes:
        table = Table(show_header=True, header_style="bold", box=None, padding=(0, 1))
        table.add_column("Status", no_wrap=True)
        table.add_column("Variable", style="cyan", no_wrap=True)
        table.add_column("Old")
        table.add_column("New")

        for c in ld.changes:
            if c.kind == "modified":
                table.add_row(
                    Text("MODIFIED", style="yellow"),
                    c.name,
                    _truncate(c.old_value),
                    _truncate(c.new_value),
                )
            elif c.kind == "added":
                table.add_row(
                    Text("ADDED", style="green"),
                    c.name,
                    "",
                    _truncate(c.new_value),
                )
            elif c.kind == "removed":
                table.add_row(
                    Text("REMOVED", style="red"),
                    c.name,
                    _truncate(c.old_value),
                    "",
                )
            elif c.kind == "type_changed":
                table.add_row(
                    Text("TYPE", style="magenta"),
                    c.name,
                    f"{_truncate(c.old_value)} ({c.old_type})",
                    f"{_truncate(c.new_value)} ({c.new_type})",
                )
            elif c.kind == "shape_changed":
                new_val: Any = c.new_value
                new_val_list = (
                    cast(list[Any], new_val) if isinstance(new_val, list) else cast(list[Any], [])
                )
                empty = (
                    " !! EMPTY"
                    if (isinstance(new_val, list) and new_val_list and new_val_list[0] == 0)
                    else ""
                )
                table.add_row(
                    Text(f"SHAPE{empty}", style="red" if empty else "yellow"),
                    c.name,
                    str(c.old_value),
                    str(cast(Any, new_val)),
                )

        console.print(Panel(table, title="Locals (crash frame)", border_style="cyan"))

    if ld.unchanged_count > 0:
        console.print(Text(f"  ({ld.unchanged_count} unchanged)", style="dim"))

    # Git
    if result.git_changed:
        old_commit = result.old_git.get("commit", "?")
        new_commit = result.new_git.get("commit", "?")
        body = Text()
        body.append(f"commit: {old_commit}", style="red")
        body.append(" -> ", style="dim")
        body.append(new_commit, style="green")
        console.print(Panel(body, title="Git", border_style="dim"))
