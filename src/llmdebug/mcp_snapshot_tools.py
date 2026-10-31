"""Snapshot-related MCP tool helpers and handler implementations.

Provides:
- Snapshot loading, validation, and evidence extraction helpers
- Diagnose section builders (exception, crash frame, context)
- Handler implementations for llmdebug_list and llmdebug_frame
"""

from __future__ import annotations

import logging
from typing import Any, cast

from . import _json as json
from ._coerce import coerce_int
from ._debug_session import coerce_crash_frame_index
from ._presentation import format_frame_text
from ._snapshot_loader import find_all_snapshots, load_snapshot, load_snapshot_raw
from .mcp_contract import (
    default_evidence_fields,
    render_invalid_response_format_error,
    render_tool_error,
    render_tool_response,
    validate_evidence_schema,
    validate_response_format,
)
from .mcp_rca import (
    advance_rca_with_snapshot_refs,
    extract_evidence_refs,
    render_rca_update_failed,
)
from .mcp_types import EvidenceLocalSummary, StableEvidencePayload
from .rca_types import RCAAttemptRequest, RCASnapshotAdvanceRequest

_logger = logging.getLogger("llmdebug.mcp")

VALID_DETAIL_LEVELS = ("crash", "full", "context")


# ---------------------------------------------------------------------------
# Rendering helpers
# ---------------------------------------------------------------------------


def render_local_preview(value: Any) -> str:
    """Render a local variable value as a short preview string.

    Serialized arrays render as ``<array shape=... dtype=...>``, tagged with the
    recorded array kind. Other values are JSON-encoded (falling back to ``repr``)
    and truncated to 120 characters.

    Args:
        value: Serialized local variable value from a snapshot frame.

    Returns:
        The preview string.
    """
    # Use a separate binding so the isinstance narrowing doesn't affect `value` below.
    _v: Any = value
    if isinstance(_v, dict) and "__array__" in _v:
        value_d = cast(dict[str, Any], _v)
        shape = value_d.get("shape")
        dtype = value_d.get("dtype")
        arr_type = value_d.get("__array__", "array")
        return f"<{arr_type} shape={shape} dtype={dtype}>"
    try:
        rendered = json.dumps(value, ensure_ascii=False, default=str)
    except Exception:
        rendered = repr(value)
    if len(rendered) > 120:
        rendered = rendered[:117] + "..."
    return rendered


def normalize_detail(detail: str) -> str | None:
    """Validate and normalize a detail level string, returning None if invalid.

    Args:
        detail: Requested detail level, in any case and with optional whitespace.

    Returns:
        ``crash``, ``full``, or ``context``; ``None`` for any other value.
    """
    candidate = str(detail or "").strip().lower()
    if candidate in VALID_DETAIL_LEVELS:
        return candidate
    return None


def render_invalid_detail_error(
    *,
    tool: str,
    detail: str,
    response_format: str,
    evidence_schema: str,
    with_rca: bool,
) -> str:
    """Render an ``INVALID_ARGUMENT`` error for an unsupported ``detail`` value.

    Args:
        tool: Name of the MCP tool that received the value.
        detail: The rejected detail value.
        response_format: ``json`` for the error envelope, ``text`` for the bare message.
        evidence_schema: Requested evidence schema for the envelope.
        with_rca: Whether the caller requested RCA, echoed in the metadata.

    Returns:
        The rendered error, listing the allowed detail levels.
    """
    return render_tool_error(
        tool=tool,
        response_format=response_format,
        evidence_schema=evidence_schema,
        with_rca=with_rca,
        code="INVALID_ARGUMENT",
        message=f"Invalid detail {detail!r}. Use 'crash', 'full', or 'context'.",
        details={
            "argument": "detail",
            "value": detail,
            "allowed_values": list(VALID_DETAIL_LEVELS),
        },
    )


def locals_summary(frame: dict[str, Any], *, max_items: int = 10) -> list[EvidenceLocalSummary]:
    """Extract up to ``max_items`` local variables from a frame as typed summary dicts.

    Only the first *max_items* locals are considered; internal entries whose names
    start with ``@`` are then skipped, so fewer summaries may be returned.

    Args:
        frame: Snapshot frame with ``locals`` and optional ``locals_meta``.
        max_items: Maximum number of locals to consider.

    Returns:
        Name, recorded type, and value preview for each kept local; empty when the
        frame has no locals mapping.
    """
    out: list[EvidenceLocalSummary] = []
    locals_obj = frame.get("locals")
    locals_meta = frame.get("locals_meta")
    if not isinstance(locals_obj, dict):
        return out
    locals_typed = cast(dict[str, Any], locals_obj)
    _empty: dict[str, Any] = {}
    meta_dict = cast(dict[str, Any], locals_meta) if isinstance(locals_meta, dict) else _empty
    for name, value in list(locals_typed.items())[:max_items]:
        key = str(name)
        if key.startswith("@"):
            continue
        type_name = ""
        meta = meta_dict.get(key)
        if isinstance(meta, dict):
            type_name = str(cast(dict[str, Any], meta).get("type") or "")
        out.append(
            {
                "name": key,
                "type": type_name,
                "preview": render_local_preview(value),
            }
        )
    return out


# ---------------------------------------------------------------------------
# Evidence extraction
# ---------------------------------------------------------------------------


def core_evidence_from_snapshot(
    snapshot: dict[str, Any] | None,
) -> StableEvidencePayload:
    """Extract a stable evidence payload from a snapshot.

    Args:
        snapshot: Loaded snapshot, or ``None``.

    Returns:
        The exception, crash frame, crash-frame locals, repro command, and evidence
        IDs found in the snapshot, over the default empty fields; only the defaults
        when *snapshot* is not a dict.
    """
    evidence = default_evidence_fields()
    if not isinstance(snapshot, dict):
        return evidence

    exc_raw = snapshot.get("exception")
    if isinstance(exc_raw, dict):
        exc = cast(dict[str, Any], exc_raw)
        evidence["exception"] = {
            "type": str(exc.get("type") or ""),
            "message": str(exc.get("message") or ""),
        }

    frames_raw = snapshot.get("frames")
    frames = cast(list[Any], frames_raw) if isinstance(frames_raw, list) else cast(list[Any], [])
    crash_idx = coerce_crash_frame_index(snapshot) if frames else -1
    if 0 <= crash_idx < len(frames) and isinstance(frames[crash_idx], dict):
        frame = cast(dict[str, Any], frames[crash_idx])
        evidence["crash_frame"] = {
            "index": crash_idx,
            "file_rel": str(frame.get("file_rel", frame.get("file", "")) or ""),
            "line": coerce_int(frame.get("line"), default=0, minimum=0),
            "function": str(frame.get("function") or ""),
            "code": str(frame.get("code") or ""),
        }
        evidence["locals_summary"] = locals_summary(frame)

    pytest_raw = snapshot.get("pytest")
    if isinstance(pytest_raw, dict):
        pytest_obj = cast(dict[str, Any], pytest_raw)
        argv_raw = pytest_obj.get("repro")
        evidence["repro"] = {
            "argv": list(cast(list[Any], argv_raw)) if isinstance(argv_raw, list) else [],
            "nodeid": str(pytest_obj.get("nodeid") or ""),
        }

    evidence["evidence_ids"] = extract_evidence_refs(snapshot)
    return evidence


def snapshot_error_code(message: str) -> str:
    """Map a human error message to a machine-readable error code string.

    Args:
        message: Error message produced by :func:`load_snapshot_safe`.

    Returns:
        ``NO_SNAPSHOTS`` when the message reports that no snapshots were found, else
        ``SNAPSHOT_LOAD_ERROR``.
    """
    if message.startswith("No snapshots found"):
        return "NO_SNAPSHOTS"
    if message.startswith("Error loading snapshot") or message.startswith(
        "Error loading snapshots"
    ):
        return "SNAPSHOT_LOAD_ERROR"
    return "SNAPSHOT_LOAD_ERROR"


def load_snapshot_safe(
    snapshot_path: str | None,
    out_dir: str,
    tool_name: str,
    *,
    raw: bool = False,
) -> dict[str, Any] | str:
    """Load a snapshot, returning an error message string on failure.

    Args:
        snapshot_path: Snapshot file to load; ``None`` loads the latest snapshot.
        out_dir: Snapshot directory.
        tool_name: Calling tool, named in the log for unexpected errors.
        raw: Whether to return the stored payload without DebugSession normalization
            instead of the flat snapshot view.

    Returns:
        The loaded snapshot, or an error message when no snapshot exists or loading
        fails.
    """
    try:
        loaded = (
            load_snapshot_raw(snapshot_path, out_dir)
            if raw
            else load_snapshot(snapshot_path, out_dir)
        )
        return loaded
    except FileNotFoundError:
        return (
            f"No snapshots found in {out_dir}. "
            "Run your tests with llmdebug installed to capture snapshots."
        )
    except (ValueError, ImportError, json.JSONDecodeError, OSError) as e:
        return f"Error loading snapshot: {type(e).__name__}: {e}"
    except Exception as e:
        _logger.error("Unexpected error in %s: %s: %s", tool_name, type(e).__name__, e)
        return f"Error loading snapshot: {type(e).__name__}: {e}"


# ---------------------------------------------------------------------------
# llmdebug_diagnose section builders
# ---------------------------------------------------------------------------


def diagnose_exception_section(snapshot: dict[str, Any]) -> str:
    """Build the ``## Exception`` markdown section for diagnose output.

    Args:
        snapshot: Loaded snapshot.

    Returns:
        The section with the exception type and message.
    """
    _empty: dict[str, Any] = {}
    exc = cast(dict[str, Any], snapshot.get("exception") or _empty)
    exc_type = exc.get("type", "UnknownError")
    exc_msg = exc.get("message", "")
    return f"## Exception\n{exc_type}: {exc_msg}"


def diagnose_crash_frame_section(snapshot: dict[str, Any], detail: str) -> str | None:
    """Build the ``## Crash Frame`` markdown section(s) for diagnose output.

    Args:
        snapshot: Loaded snapshot.
        detail: Normalized detail level; ``full`` and ``context`` add a section per
            other frame, ``crash`` adds a note on how many frames were left out.

    Returns:
        The rendered sections, or ``None`` when the snapshot has no usable crash
        frame.
    """
    frames_raw = snapshot.get("frames")
    frames = cast(list[Any], frames_raw) if isinstance(frames_raw, list) else cast(list[Any], [])
    if not frames:
        return None
    crash_idx = coerce_crash_frame_index(snapshot)
    if not (0 <= crash_idx < len(frames)):
        return None
    crash_frame_raw = frames[crash_idx]
    parts: list[str] = []
    if isinstance(crash_frame_raw, dict):
        crash_frame = cast(dict[str, Any], crash_frame_raw)
        parts.append(
            f"## Crash Frame (#{crash_idx})\n"
            + format_frame_text(crash_frame, crash_idx, True, crash_tag=True)
        )
    if detail in ("full", "context") and len(frames) > 1:
        for i, frame_raw in enumerate(frames):
            if i == crash_idx or not isinstance(frame_raw, dict):
                continue
            frame = cast(dict[str, Any], frame_raw)
            parts.append(f"## Frame #{i}\n" + format_frame_text(frame, i, False, crash_tag=True))
    elif len(frames) > 1:
        parts.append(f"({len(frames) - 1} more frame(s) — use llmdebug_frame to inspect)")
    return "\n\n".join(parts) if parts else None


def diagnose_context_section(snapshot: dict[str, Any], detail: str) -> str | None:
    """Build the ``## Repro`` / ``## Git`` markdown sections for diagnose output.

    Args:
        snapshot: Loaded snapshot.
        detail: Normalized detail level; only ``full`` and ``context`` include these
            sections.

    Returns:
        The rendered sections, or ``None`` when the detail level excludes them or
        the snapshot has neither repro nor git commit data.
    """
    if detail not in ("full", "context"):
        return None
    _empty: dict[str, Any] = {}
    parts: list[str] = []
    pt = cast(dict[str, Any], snapshot.get("pytest") or _empty)
    repro_raw = pt.get("repro")
    if repro_raw and isinstance(repro_raw, list):
        repro_list = cast(list[Any], repro_raw)
        cmd_parts = [
            "python" if str(r).endswith("/python3") or str(r).endswith("/python") else str(r)
            for r in repro_list
        ]
        parts.append(f"## Repro\n{' '.join(cmd_parts)}")
    git = cast(dict[str, Any], snapshot.get("git") or _empty)
    if git:
        commit = git.get("commit", "")
        branch = git.get("branch", "")
        dirty = " *dirty*" if git.get("dirty") else ""
        if commit:
            parts.append(f"## Git\n{commit} {branch}{dirty}")
    return "\n\n".join(parts) if parts else None


# ---------------------------------------------------------------------------
# Handler implementations
# ---------------------------------------------------------------------------


def handle_list(
    out_dir: str = ".llmdebug",
    limit: int = 10,
    response_format: str = "text",
    evidence_schema: str = "summary",
) -> str:
    """List snapshots for the ``llmdebug_list`` MCP tool.

    Args:
        out_dir: Snapshot directory.
        limit: Maximum number of snapshots to list; must be at least 1.
        response_format: ``text`` or ``json``.
        evidence_schema: ``summary`` or ``full`` JSON payload detail.

    Returns:
        The newest snapshots with timestamp, name, exception, location, and path, or
        an error response (a bare message in text mode for scan failures and an empty
        directory).
    """
    format_error = validate_response_format(response_format)
    if format_error is not None:
        return render_invalid_response_format_error(
            tool="llmdebug_list",
            response_format=response_format,
            evidence_schema=evidence_schema,
            with_rca=False,
        )
    schema_error = validate_evidence_schema(evidence_schema)
    if schema_error is not None:
        return render_tool_error(
            tool="llmdebug_list",
            response_format=response_format,
            evidence_schema=evidence_schema,
            with_rca=False,
            code="INVALID_ARGUMENT",
            message=schema_error,
        )
    if limit <= 0:
        # Same rule as llmdebug_rca_history; limit <= 0 used to list everything (RCA-17).
        return render_tool_error(
            tool="llmdebug_list",
            response_format=response_format,
            evidence_schema=evidence_schema,
            with_rca=False,
            code="INVALID_ARGUMENT",
            message=f"limit must be >= 1 (got {limit})",
        )
    try:
        entries = find_all_snapshots(out_dir, limit=limit)
    except OSError as e:
        message = f"Error scanning snapshots: {type(e).__name__}: {e}"
        if response_format == "text":
            return message
        return render_tool_error(
            tool="llmdebug_list",
            response_format=response_format,
            evidence_schema=evidence_schema,
            with_rca=False,
            code="SNAPSHOT_LOAD_ERROR",
            message=message,
        )
    except Exception as e:
        _logger.error("Unexpected error in llmdebug_list: %s: %s", type(e).__name__, e)
        message = f"Error scanning snapshots: {type(e).__name__}: {e}"
        if response_format == "text":
            return message
        return render_tool_error(
            tool="llmdebug_list",
            response_format=response_format,
            evidence_schema=evidence_schema,
            with_rca=False,
            code="SNAPSHOT_LOAD_ERROR",
            message=message,
        )

    if not entries:
        message = f"No snapshots found in {out_dir}."
        if response_format == "text":
            return message
        return render_tool_error(
            tool="llmdebug_list",
            response_format=response_format,
            evidence_schema=evidence_schema,
            with_rca=False,
            code="NO_SNAPSHOTS",
            message=message,
        )

    lines: list[str] = [f"Found {len(entries)} snapshot(s) in {out_dir}:\n"]
    for i, entry in enumerate(entries, 1):
        from datetime import datetime, timezone

        ts = datetime.fromtimestamp(entry["mtime"], tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        loc = f"{entry['file']}:{entry['line']}"
        msg = entry.get("exc_message", "")
        msg_part = f" — {msg}" if msg else ""
        lines.append(f"  {i}. [{ts}] {entry['name']}  {entry['exc_type']}{msg_part}  ({loc})")
        lines.append(f"     path: {entry['path']}")

    body = "\n".join(lines)
    if response_format == "text":
        return body
    return render_tool_response(
        tool="llmdebug_list",
        body=body,
        response_format=response_format,
        evidence_schema=evidence_schema,
        with_rca=False,
        evidence_payload={
            "snapshot_list": {
                "out_dir": out_dir,
                "limit": int(limit),
                "count": len(entries),
                "entries": entries,
            }
        },
    )


def handle_frame(
    index: int = 0,
    out_dir: str = ".llmdebug",
    snapshot_path: str = "",
    gate_mode: str = "",
    exploratory: bool = False,
    response_format: str = "json",
    with_rca: bool = False,
    evidence_schema: str = "summary",
) -> str:
    """Show one stack frame for the ``llmdebug_frame`` MCP tool.

    Args:
        index: Frame index; 0 is the crash site and higher indices are callers.
        out_dir: Snapshot directory.
        snapshot_path: Snapshot file to read; empty uses the latest snapshot.
        gate_mode: RCA gate mode override (``off``, ``soft``, or ``strict``); empty
            uses ``LLMDEBUG_RCA_GATE_MODE`` or ``soft``.
        exploratory: Whether to mark the RCA step as exploratory.
        response_format: ``json`` or ``text``.
        with_rca: Whether to record a ``bind_evidence`` RCA attempt and include the
            RCA block.
        evidence_schema: ``summary`` or ``full`` JSON payload detail.

    Returns:
        The rendered frame with its source and locals, or an error response when an
        argument is invalid, the snapshot cannot be loaded, the index is out of
        range, the frame is malformed, or the RCA update fails.
    """
    format_error = validate_response_format(response_format)
    if format_error is not None:
        return render_invalid_response_format_error(
            tool="llmdebug_frame",
            response_format=response_format,
            evidence_schema=evidence_schema,
            with_rca=with_rca,
        )
    schema_error = validate_evidence_schema(evidence_schema)
    if schema_error is not None:
        return render_tool_error(
            tool="llmdebug_frame",
            response_format=response_format,
            evidence_schema=evidence_schema,
            with_rca=with_rca,
            code="INVALID_ARGUMENT",
            message=schema_error,
        )

    result = load_snapshot_safe(snapshot_path or None, out_dir, "llmdebug_frame")
    if isinstance(result, str):
        return render_tool_error(
            tool="llmdebug_frame",
            response_format=response_format,
            evidence_schema=evidence_schema,
            with_rca=with_rca,
            code=snapshot_error_code(result),
            message=result,
        )
    snapshot = result

    frames_raw = snapshot.get("frames")
    frames = cast(list[Any], frames_raw) if isinstance(frames_raw, list) else cast(list[Any], [])
    if not frames:
        return render_tool_error(
            tool="llmdebug_frame",
            response_format=response_format,
            evidence_schema=evidence_schema,
            with_rca=with_rca,
            code="MALFORMED_SNAPSHOT",
            message="No frames in snapshot.",
        )

    if index < 0 or index >= len(frames):
        return render_tool_error(
            tool="llmdebug_frame",
            response_format=response_format,
            evidence_schema=evidence_schema,
            with_rca=with_rca,
            code="FRAME_OUT_OF_RANGE",
            message=(
                f"Frame index {index} out of range. "
                f"Available frames: 0-{len(frames) - 1} ({len(frames)} total)."
            ),
        )

    frame_raw = frames[index]
    if not isinstance(frame_raw, dict):
        return render_tool_error(
            tool="llmdebug_frame",
            response_format=response_format,
            evidence_schema=evidence_schema,
            with_rca=with_rca,
            code="MALFORMED_SNAPSHOT",
            message=f"Frame index {index} is malformed in snapshot.",
        )
    frame = cast(dict[str, Any], frame_raw)
    crash_idx = coerce_crash_frame_index(snapshot)
    is_crash = index == crash_idx
    core_evidence = core_evidence_from_snapshot(snapshot)
    core_evidence["crash_frame"] = {
        "index": index,
        "file_rel": str(frame.get("file_rel", frame.get("file", "")) or ""),
        "line": coerce_int(frame.get("line"), default=0, minimum=0),
        "function": str(frame.get("function") or ""),
        "code": str(frame.get("code") or ""),
    }
    core_evidence["locals_summary"] = locals_summary(frame)
    evidence_ids = list(core_evidence.get("evidence_ids") or [])
    evidence_ids.append(f"frame:{index}")
    core_evidence["evidence_ids"] = list(dict.fromkeys(evidence_ids))

    body = format_frame_text(frame, index, is_crash, crash_tag=True)
    rca_record = (
        advance_rca_with_snapshot_refs(
            request=RCASnapshotAdvanceRequest(
                out_dir=out_dir,
                snapshot=snapshot,
                attempt=RCAAttemptRequest(
                    session_id="",
                    action="bind_evidence",
                    failure_signature="",
                    gate_mode=gate_mode,
                    exploratory=exploratory,
                ),
            ),
        )
        if with_rca
        else None
    )
    if with_rca and rca_record is None:
        return render_rca_update_failed(
            tool="llmdebug_frame",
            response_format=response_format,
            evidence_schema=evidence_schema,
        )
    return render_tool_response(
        tool="llmdebug_frame",
        body=body,
        response_format=response_format,
        evidence_schema=evidence_schema,
        with_rca=with_rca,
        rca_record=rca_record,
        evidence_payload={
            **core_evidence,
            "snapshot": snapshot,
            "frame_index": index,
            "frame": frame,
        },
    )
