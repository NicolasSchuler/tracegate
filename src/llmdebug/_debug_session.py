"""DebugSession schema helpers.

Provides write/read compatibility between the new versioned DebugSession
envelope and the legacy flat snapshot view consumed by existing tools.
"""

from __future__ import annotations

from typing import Any, cast

DEBUG_SESSION_SCHEMA_VERSION = "2.0"
DEBUG_SESSION_KIND = "llmdebug.debug_session"


def split_extra_context(
    extra: dict[str, Any] | None,
) -> tuple[dict[str, Any], Any | None, Any | None]:
    """Split capture extra payload into context, trace, and eval sections.

    Three-way split: "trace" is the optional trace/diagnostic payload,
    "eval" (or its alias "eval_outcome") is the eval-outcome overlay, and the
    remainder is generic context. When both "eval" and "eval_outcome" are
    given, "eval" wins and "eval_outcome" stays in the context.
    """
    if not extra:
        return {}, None, None
    context = dict(extra)
    trace_payload = context.pop("trace", None)
    eval_key = "eval" if "eval" in context else "eval_outcome"
    eval_payload = context.pop(eval_key, None)
    return context, trace_payload, eval_payload


def build_debug_session(
    *,
    name: str,
    timestamp_utc: str,
    llmdebug_version: str,
    snapshot: dict[str, Any],
    context: dict[str, Any] | None = None,
    trace_payload: Any = None,
    eval_payload: Any = None,
) -> dict[str, Any]:
    """Build a versioned DebugSession envelope.

    Wraps a snapshot + session metadata + optional context/trace/eval
    in a schema-versioned DebugSession payload.
    """
    payload: dict[str, Any] = {
        "schema_version": DEBUG_SESSION_SCHEMA_VERSION,
        "kind": DEBUG_SESSION_KIND,
        "session": {
            "name": name,
            "timestamp_utc": timestamp_utc,
            "llmdebug_version": llmdebug_version,
        },
        "snapshot": snapshot,
    }
    if context:
        payload["context"] = context
    if trace_payload is not None:
        payload["trace"] = trace_payload
    if eval_payload is not None:
        payload["eval"] = eval_payload
    return payload


def is_debug_session_payload(data: dict[str, Any]) -> bool:
    """Return True when *data* looks like a DebugSession envelope."""
    if not isinstance(data, dict):  # pyright: ignore[reportUnnecessaryIsInstance]
        return False
    session = data.get("session")
    snapshot = data.get("snapshot")
    if not isinstance(session, dict) or not isinstance(snapshot, dict):
        return False

    kind = data.get("kind")
    if kind is not None:
        return kind == DEBUG_SESSION_KIND

    # Backward-compatible detection for envelopes without explicit `kind`.
    schema_version = data.get("schema_version")
    return isinstance(schema_version, str) and schema_version.startswith("2")


def normalize_snapshot_view(data: dict[str, Any]) -> dict[str, Any]:
    """Normalize DebugSession payloads to the legacy flat snapshot view.

    Legacy snapshots are returned unchanged.
    """
    if not is_debug_session_payload(data):
        return data

    session = data.get("session")
    snapshot = data.get("snapshot")
    context = data.get("context")

    out: dict[str, Any] = {}

    schema_version = data.get("schema_version")
    if schema_version is not None:
        out["schema_version"] = schema_version

    if isinstance(session, dict):
        session_dict = cast(dict[str, Any], session)
        name = session_dict.get("name")
        if name is not None:
            out["name"] = name
        elif "name" in data:
            out["name"] = data["name"]

        timestamp_utc = session_dict.get("timestamp_utc")
        if timestamp_utc is not None:
            out["timestamp_utc"] = timestamp_utc
        elif "timestamp_utc" in data:
            out["timestamp_utc"] = data["timestamp_utc"]
    else:
        if "name" in data:
            out["name"] = data["name"]
        if "timestamp_utc" in data:
            out["timestamp_utc"] = data["timestamp_utc"]

    if isinstance(snapshot, dict):
        out.update(cast(dict[str, Any], snapshot))

    # Preserve legacy precedence where capture `extra` could override core fields.
    if isinstance(context, dict):
        out.update(cast(dict[str, Any], context))

    if isinstance(session, dict):
        llmdebug_version = cast(dict[str, Any], session).get("llmdebug_version")
        if llmdebug_version is None:
            llmdebug_version = data.get("llmdebug_version")
    else:
        llmdebug_version = data.get("llmdebug_version")
    if llmdebug_version is not None:
        out["llmdebug_version"] = llmdebug_version

    if "trace" in data:
        out["trace"] = data["trace"]
    if "eval" in data:
        out["eval"] = data["eval"]

    reserved = {"schema_version", "kind", "session", "snapshot", "context", "trace", "eval"}
    for key, value in data.items():
        if key not in reserved and key not in out:
            out[key] = value

    return out


def coerce_crash_frame_index(snapshot: dict[str, Any]) -> int:
    """Return a safe crash-frame index for potentially malformed snapshots.

    Falls back to the first dict frame when the stored index is missing,
    invalid, out-of-range, or points to a non-dict frame.
    """
    frames = snapshot.get("frames")
    if not isinstance(frames, list) or not frames:
        return 0
    frames_list = cast(list[Any], frames)

    try:
        idx = int(snapshot.get("crash_frame_index", 0))
    except (TypeError, ValueError):
        idx = 0

    if 0 <= idx < len(frames_list) and isinstance(frames_list[idx], dict):
        return idx

    for i, frame in enumerate(frames_list):
        if isinstance(frame, dict):
            return i
    return 0
