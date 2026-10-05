"""Identity/signature helpers for RCA workflows."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any, cast

from . import _json as json
from ._coerce import coerce_int
from ._debug_session import coerce_crash_frame_index, normalize_snapshot_view
from ._hashing import xxh64_hexdigest_text

# RCA-12: run-specific parts of failure text, replaced before hashing so that a repeated
# failure keeps its signature: UUIDs, memory addresses, absolute paths (tmp_path), long hex
# ids, then any remaining number (counts, timestamps, ports). Order matters.
_VOLATILE_TEXT: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b"
        ),
        "<uuid>",
    ),
    (re.compile(r"0x[0-9a-fA-F]+"), "<addr>"),
    (
        re.compile(r"(?<![\w.])(?:[A-Za-z]:\\|/)(?:[^\s/\\'\"(),:]+[/\\])+[^\s/\\'\"(),:]*"),
        "<path>",
    ),
    (re.compile(r"\b[0-9a-fA-F]{12,}\b"), "<hex>"),
    (re.compile(r"\d+"), "<n>"),
)


def normalize_failure_text(text: str) -> str:
    """Replace run-specific values (addresses, tmp paths, ids, numbers) with placeholders."""
    for pattern, placeholder in _VOLATILE_TEXT:
        text = pattern.sub(placeholder, text)
    return text


def _signature_from_snapshot(view: dict[str, Any], *, normalize: bool) -> str:
    """Build failure signature from a normalized snapshot view."""
    exc = view.get("exception", {})
    exc_type = str(exc.get("type") or "Exception")
    exc_message = str(exc.get("message") or "")
    if normalize:
        exc_message = normalize_failure_text(exc_message)
    frames_raw = view.get("frames")
    frames: list[dict[str, Any]] = (
        cast(list[dict[str, Any]], frames_raw) if isinstance(frames_raw, list) else []
    )
    crash_frame: dict[str, Any] = {}
    if frames:
        crash_idx = coerce_crash_frame_index(view)
        candidate = frames[crash_idx] if 0 <= crash_idx < len(frames) else frames[0]
        if isinstance(candidate, dict):  # pyright: ignore[reportUnnecessaryIsInstance] — defense-in-depth for malformed data
            crash_frame = candidate
    location = {
        "file": str(crash_frame.get("file_rel") or crash_frame.get("file") or "?"),
        "line": coerce_int(crash_frame.get("line"), default=0, minimum=0),
        "function": str(crash_frame.get("function") or "?"),
    }
    nodeid = ""
    pytest_block = view.get("pytest")
    if isinstance(pytest_block, dict):
        nodeid = str(cast(dict[str, Any], pytest_block).get("nodeid") or "")
    payload = {
        "exception_type": exc_type,
        "exception_message": exc_message,
        "location": location,
        "nodeid": nodeid,
    }
    digest = xxh64_hexdigest_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    )[:16]
    return f"{exc_type}:{digest}"


def _signature_from_text(stdout: str, stderr: str, *, normalize: bool) -> str:
    """Build failure signature from raw text output."""
    text = (stderr or "").strip() or (stdout or "").strip() or "unknown_failure"
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    head = "\n".join(lines[-3:]) if lines else text
    if normalize:
        head = normalize_failure_text(head)
    digest = xxh64_hexdigest_text(head)[:16]
    return f"TextFailure:{digest}"


def build_failure_signature(
    snapshot: dict[str, Any] | None,
    *,
    stdout: str = "",
    stderr: str = "",
    normalize: bool = True,
) -> str:
    """Build a stable failure signature from snapshot (preferred) or textual output.

    With ``normalize`` (the default), run-specific values in the exception message or
    output tail are replaced first (:func:`normalize_failure_text`), so a failure that
    repeats with a new address, tmp path or count keeps its signature and stall detection
    can fire. ``normalize=False`` gives the raw signature used before RCA-12.
    """
    if snapshot:
        return _signature_from_snapshot(normalize_snapshot_view(snapshot), normalize=normalize)
    return _signature_from_text(stdout, stderr, normalize=normalize)


def derive_session_id(
    snapshot: dict[str, Any] | None,
    *,
    out_dir: str,
    fallback: str = "global",
) -> str:
    """Derive stable session id from out_dir + pytest nodeid/name when available."""
    resolved = str(Path(out_dir).resolve())
    path_hash = xxh64_hexdigest_text(resolved)[:8]
    scope = fallback
    if snapshot:
        view = normalize_snapshot_view(snapshot)
        pytest_block = view.get("pytest")
        if isinstance(pytest_block, dict):
            pytest_dict = cast(dict[str, Any], pytest_block)
            raw_nodeid = pytest_dict.get("nodeid")
            if isinstance(raw_nodeid, str):
                nodeid = raw_nodeid.strip()
                if nodeid:
                    scope = nodeid
        if scope == fallback and isinstance(view.get("name"), str):
            name = str(view.get("name")).strip()
            if name:
                scope = name
    return f"{path_hash}::{scope}"
