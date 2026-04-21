"""Snapshot diffing — compare two debug snapshots to show what changed.

Pure logic module with zero external dependencies. Used by CLI and MCP server.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from typing import Any, Literal, cast

from ._debug_session import coerce_crash_frame_index, normalize_snapshot_view
from .snapshot_types import ExceptionDict, SnapshotDict

# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

_ARRAY_KEYS = ("__array__", "__dataframe__", "__series__")

# Typed empty sentinels reused across the module to avoid repeated `cast(dict[str,Any], {})`
# and `cast(list[Any], [])` expressions.
_EMPTY_DICT: dict[str, Any] = {}
_EMPTY_LIST: list[Any] = []


def _as_dict(value: Any) -> dict[str, Any]:
    return cast(dict[str, Any], value) if isinstance(value, dict) else _EMPTY_DICT


def _as_list(value: Any) -> list[Any]:
    return cast(list[Any], value) if isinstance(value, list) else _EMPTY_LIST


def _text_or_empty(value: Any) -> str:
    if value is None:
        return ""
    return value if isinstance(value, str) else str(value)


def _frame_file_value(frame: Mapping[str, Any]) -> str:
    file_rel = frame.get("file_rel")
    if isinstance(file_rel, str) and file_rel:
        return file_rel
    file_abs = frame.get("file")
    if isinstance(file_abs, str) and file_abs:
        return file_abs
    return "?"


ChangeKind = Literal["modified", "added", "removed", "type_changed", "shape_changed"]
HypothesisDeltaKind = Literal["unknown", "unchanged", "refined", "replaced", "added", "removed"]


@dataclass(frozen=True)
class ExceptionDiff:
    """Difference summary for exception identity and high-level category."""

    changed: bool
    old_type: str
    new_type: str
    old_message: str
    new_message: str
    old_category: str
    new_category: str


@dataclass(frozen=True)
class LocationDiff:
    """Difference summary for the selected crash location."""

    changed: bool
    old_location: str  # "file.py:42 in func_name"
    new_location: str


@dataclass(frozen=True)
class LocalChange:
    """Single local-variable change detected between two snapshots."""

    kind: ChangeKind
    name: str
    old_value: Any = None
    new_value: Any = None
    old_type: str = ""
    new_type: str = ""


@dataclass(frozen=True)
class LocalsDiff:
    """Grouped local-variable diff plus count of unchanged entries."""

    changes: tuple[LocalChange, ...] = ()
    unchanged_count: int = 0


@dataclass(frozen=True)
class EvidenceDelta:
    """High-level change in evidence surface between two attempts."""

    new_frames: tuple[str, ...] = ()
    dropped_frames: tuple[str, ...] = ()
    message_changed: bool = False


@dataclass(frozen=True)
class ChangeDelta:
    """Touched-file/function summary derived from git metadata."""

    touched_files: tuple[str, ...] = ()
    touched_functions: tuple[str, ...] = ()


@dataclass(frozen=True)
class VerificationDelta:
    """Change in downstream verification status between snapshots."""

    old_status: str = "unknown"
    new_status: str = "unknown"
    status_changed: bool = False


@dataclass(frozen=True)
class SnapshotDiffResult:
    """Structured diff result consumed by CLI and MCP presentation layers."""

    exception: ExceptionDiff
    location: LocationDiff
    locals_diff: LocalsDiff
    git_changed: bool
    old_git: dict[str, Any] = field(default_factory=lambda: cast(dict[str, Any], {}))
    new_git: dict[str, Any] = field(default_factory=lambda: cast(dict[str, Any], {}))
    old_name: str = ""
    new_name: str = ""
    same_test: bool = True
    failure_signature_same: bool = False
    evidence_delta: EvidenceDelta = field(default_factory=EvidenceDelta)
    change_delta: ChangeDelta = field(default_factory=ChangeDelta)
    hypothesis_delta: HypothesisDeltaKind = "unknown"
    verification_delta: VerificationDelta = field(default_factory=VerificationDelta)


# ---------------------------------------------------------------------------
# Diff functions
# ---------------------------------------------------------------------------


def diff_exception(
    old_exc: dict[str, Any] | ExceptionDict,
    new_exc: dict[str, Any] | ExceptionDict,
) -> ExceptionDiff:
    """Compare exception type, message, and error category."""
    old_exc_t = _as_dict(old_exc)
    new_exc_t = _as_dict(new_exc)
    old_type = _text_or_empty(old_exc_t.get("type", ""))
    new_type = _text_or_empty(new_exc_t.get("type", ""))
    old_msg = _text_or_empty(old_exc_t.get("message", ""))
    new_msg = _text_or_empty(new_exc_t.get("message", ""))
    old_cat = _text_or_empty(_as_dict(old_exc_t.get("error_category")).get("category", ""))
    new_cat = _text_or_empty(_as_dict(new_exc_t.get("error_category")).get("category", ""))

    changed = old_type != new_type or old_msg != new_msg or old_cat != new_cat
    return ExceptionDiff(
        changed=changed,
        old_type=old_type,
        new_type=new_type,
        old_message=old_msg,
        new_message=new_msg,
        old_category=old_cat,
        new_category=new_cat,
    )


def _format_location(frame: Mapping[str, Any]) -> str:
    """Build 'file:line in function' from a frame dict."""
    file_rel = _frame_file_value(frame)
    line = frame.get("line", "?")
    func = _text_or_empty(frame.get("function", "?")) or "?"
    return f"{file_rel}:{line} in {func}"


def diff_location(
    old_frames: list[Any],
    new_frames: list[Any],
    old_crash_idx: int,
    new_crash_idx: int,
) -> LocationDiff:
    """Compare crash location between two snapshots."""
    old_loc = (
        _format_location(old_frames[old_crash_idx])
        if old_frames
        and 0 <= old_crash_idx < len(old_frames)
        and isinstance(old_frames[old_crash_idx], dict)
        else "?"
    )
    new_loc = (
        _format_location(new_frames[new_crash_idx])
        if new_frames
        and 0 <= new_crash_idx < len(new_frames)
        and isinstance(new_frames[new_crash_idx], dict)
        else "?"
    )
    return LocationDiff(changed=old_loc != new_loc, old_location=old_loc, new_location=new_loc)


def _is_array_summary(value: Any) -> bool:
    """Check if value is an llmdebug array/dataframe/series summary dict."""
    if not isinstance(value, dict):
        return False
    value_dict = cast(dict[str, Any], value)
    return any(value_dict.get(k) for k in _ARRAY_KEYS)


def _get_shape(value: dict[str, Any]) -> list[Any] | None:
    """Extract shape from an array summary, or None."""
    shape = value.get("shape")
    return list(cast(list[Any], shape)) if isinstance(shape, (list, tuple)) else None


def _meta_type(meta: dict[str, Any], k: str) -> str:
    """Extract the type string from a locals_meta entry."""
    entry = meta.get(k)
    if isinstance(entry, dict):
        return _text_or_empty(cast(dict[str, Any], entry).get("type", ""))
    return ""


def _diff_removed(
    old_keys: set[str],
    new_keys: set[str],
    old_locals: dict[str, Any],
    old_meta: dict[str, Any],
) -> list[LocalChange]:
    return [
        LocalChange(
            kind="removed",
            name=k,
            old_value=old_locals[k],
            new_value=None,
            old_type=_meta_type(old_meta, k),
            new_type="",
        )
        for k in sorted(old_keys - new_keys)
    ]


def _diff_added(
    old_keys: set[str],
    new_keys: set[str],
    new_locals: dict[str, Any],
    new_meta: dict[str, Any],
) -> list[LocalChange]:
    return [
        LocalChange(
            kind="added",
            name=k,
            old_value=None,
            new_value=new_locals[k],
            old_type="",
            new_type=_meta_type(new_meta, k),
        )
        for k in sorted(new_keys - old_keys)
    ]


def _diff_shared_key(
    k: str,
    old_locals: dict[str, Any],
    new_locals: dict[str, Any],
    old_meta: dict[str, Any],
    new_meta: dict[str, Any],
) -> LocalChange | None:
    """Compare a single shared key; return a LocalChange or None if unchanged."""
    old_val = old_locals[k]
    new_val = new_locals[k]
    old_type = _meta_type(old_meta, k)
    new_type = _meta_type(new_meta, k)

    if _is_array_summary(old_val) and _is_array_summary(new_val):
        old_shape = _get_shape(old_val)
        new_shape = _get_shape(new_val)
        if old_shape != new_shape:
            return LocalChange(
                kind="shape_changed",
                name=k,
                old_value=old_shape,
                new_value=new_shape,
                old_type=old_val.get("dtype", ""),
                new_type=new_val.get("dtype", ""),
            )
        if old_val.get("dtype", "") != new_val.get("dtype", "") or old_val != new_val:
            return LocalChange(
                kind="modified",
                name=k,
                old_value=old_val,
                new_value=new_val,
                old_type=old_type,
                new_type=new_type,
            )
        return None

    if old_type and new_type and old_type != new_type:
        return LocalChange(
            kind="type_changed",
            name=k,
            old_value=old_val,
            new_value=new_val,
            old_type=old_type,
            new_type=new_type,
        )
    if old_val != new_val:
        return LocalChange(
            kind="modified",
            name=k,
            old_value=old_val,
            new_value=new_val,
            old_type=old_type,
            new_type=new_type,
        )
    return None


def diff_locals(
    old_frame: Any,
    new_frame: Any,
) -> LocalsDiff:
    """Compare local variables between two crash frames."""
    old_fd = _as_dict(old_frame)
    new_fd = _as_dict(new_frame)
    old_locals = _as_dict(old_fd.get("locals"))
    new_locals = _as_dict(new_fd.get("locals"))
    old_meta = _as_dict(old_fd.get("locals_meta"))
    new_meta = _as_dict(new_fd.get("locals_meta"))

    old_keys = {k for k in old_locals if not k.startswith("@py_")}
    new_keys = {k for k in new_locals if not k.startswith("@py_")}

    changes: list[LocalChange] = []
    changes.extend(_diff_removed(old_keys, new_keys, old_locals, old_meta))
    changes.extend(_diff_added(old_keys, new_keys, new_locals, new_meta))

    unchanged = 0
    for k in sorted(old_keys & new_keys):
        change = _diff_shared_key(k, old_locals, new_locals, old_meta, new_meta)
        if change is not None:
            changes.append(change)
        else:
            unchanged += 1

    return LocalsDiff(changes=tuple(changes), unchanged_count=unchanged)


def _frame_id(frame: Mapping[str, Any]) -> str:
    file_rel = _frame_file_value(frame)
    func = _text_or_empty(frame.get("function", "?")) or "?"
    line = frame.get("line", "?")
    return f"{file_rel}:{line} in {func}"


def _diff_evidence(
    old_view: dict[str, Any],
    new_view: dict[str, Any],
) -> EvidenceDelta:
    old_frames = _as_list(old_view.get("frames"))
    new_frames = _as_list(new_view.get("frames"))
    old_ids = {
        _frame_id(cast(dict[str, Any], frame)) for frame in old_frames if isinstance(frame, dict)
    }
    new_ids = {
        _frame_id(cast(dict[str, Any], frame)) for frame in new_frames if isinstance(frame, dict)
    }
    old_exc = _as_dict(old_view.get("exception"))
    new_exc = _as_dict(new_view.get("exception"))
    old_msg = _text_or_empty(old_exc.get("message", ""))
    new_msg = _text_or_empty(new_exc.get("message", ""))
    return EvidenceDelta(
        new_frames=tuple(sorted(new_ids - old_ids)),
        dropped_frames=tuple(sorted(old_ids - new_ids)),
        message_changed=old_msg != new_msg,
    )


def _coerce_crash_idx(view: dict[str, Any]) -> int:
    frames = _as_list(view.get("frames"))
    if not frames:
        return 0
    return coerce_crash_frame_index(
        {
            "frames": frames,
            "crash_frame_index": view.get("crash_frame_index", 0),
        }
    )


def _frame_file_key(frame: Mapping[str, Any]) -> str:
    return _frame_file_value(frame)


def _diff_touched_dimension(
    old_frames: list[Any],
    new_frames: list[Any],
    get_key: Callable[[Any], str],
    old_cfi: int,
    new_cfi: int,
    *,
    location_changed: bool,
) -> list[str]:
    """Compute the symmetric-difference of a frame dimension, with crash-frame fallback."""
    old_vals = {get_key(f) for f in old_frames if isinstance(f, dict)}
    new_vals = {get_key(f) for f in new_frames if isinstance(f, dict)}
    touched = sorted(old_vals ^ new_vals)
    if not touched and location_changed:
        if 0 <= old_cfi < len(old_frames) and isinstance(old_frames[old_cfi], dict):
            touched.append(get_key(old_frames[old_cfi]))
        if 0 <= new_cfi < len(new_frames) and isinstance(new_frames[new_cfi], dict):
            touched.append(get_key(new_frames[new_cfi]))
        touched = sorted({v for v in touched if v})
    return touched


def _frame_func_key(f: dict[str, Any]) -> str:
    return str(f.get("function", "?"))


def _diff_change_delta(
    old_view: dict[str, Any],
    new_view: dict[str, Any],
    *,
    location_changed: bool,
) -> ChangeDelta:
    old_frames = _as_list(old_view.get("frames"))
    new_frames = _as_list(new_view.get("frames"))
    old_cfi = _coerce_crash_idx(old_view)
    new_cfi = _coerce_crash_idx(new_view)

    touched_files = _diff_touched_dimension(
        old_frames, new_frames, _frame_file_key, old_cfi, new_cfi, location_changed=location_changed
    )
    touched_functions = _diff_touched_dimension(
        old_frames,
        new_frames,
        _frame_func_key,
        old_cfi,
        new_cfi,
        location_changed=location_changed,
    )

    return ChangeDelta(
        touched_files=tuple(touched_files),
        touched_functions=tuple(touched_functions),
    )


def _extract_hypothesis(view: dict[str, Any]) -> str:
    hyp = view.get("hypothesis")
    if isinstance(hyp, str):
        return hyp
    rca_raw = view.get("rca")
    if isinstance(rca_raw, dict):
        rca = cast(dict[str, Any], rca_raw)
        rca_hyp = rca.get("hypothesis")
        if isinstance(rca_hyp, str):
            return rca_hyp
    return ""


def _classify_hypothesis_delta(old_value: str, new_value: str) -> HypothesisDeltaKind:
    old = old_value.strip()
    new = new_value.strip()
    if not old and not new:
        return "unknown"
    if old == new:
        return "unchanged"
    if not old and new:
        return "added"
    if old and not new:
        return "removed"
    if old and new and old in new:
        return "refined"
    return "replaced"


def _extract_verification_status(view: dict[str, Any]) -> str:
    vr_raw = view.get("verification_result")
    if isinstance(vr_raw, dict):
        status = str(cast(dict[str, Any], vr_raw).get("status") or "").strip()
        if status:
            return status
    rca_raw = view.get("rca")
    if isinstance(rca_raw, dict):
        rca = cast(dict[str, Any], rca_raw)
        inner_vr_raw = rca.get("verification_result")
        if isinstance(inner_vr_raw, dict):
            status = str(cast(dict[str, Any], inner_vr_raw).get("status") or "").strip()
            if status:
                return status
    return "unknown"


def _failure_signature(view: dict[str, Any]) -> str:
    exc = _as_dict(view.get("exception"))
    exc_type = _text_or_empty(exc.get("type", ""))
    exc_msg = _text_or_empty(exc.get("message", ""))
    frames = _as_list(view.get("frames"))
    crash_idx = _coerce_crash_idx(view)
    location = "?"
    if frames and 0 <= crash_idx < len(frames) and isinstance(frames[crash_idx], dict):
        location = _frame_id(cast(dict[str, Any], frames[crash_idx]))
    return f"{exc_type}|{exc_msg}|{location}"


def diff_snapshots(old: dict[str, Any], new: dict[str, Any]) -> SnapshotDiffResult:
    """Compare two full snapshots and return a structured diff."""
    old_view = normalize_snapshot_view(old)
    new_view = normalize_snapshot_view(new)
    old_view_t = cast(SnapshotDict, old_view)
    new_view_t = cast(SnapshotDict, new_view)

    _empty_exc = cast(ExceptionDict, _EMPTY_DICT)
    old_exc_raw = old_view_t.get("exception")
    new_exc_raw = new_view_t.get("exception")
    old_exc = old_exc_raw if isinstance(old_exc_raw, dict) else _empty_exc
    new_exc = new_exc_raw if isinstance(new_exc_raw, dict) else _empty_exc
    exc_diff = diff_exception(old_exc, new_exc)

    old_frames = _as_list(old_view_t.get("frames"))
    new_frames = _as_list(new_view_t.get("frames"))
    old_cfi = _coerce_crash_idx(old_view)
    new_cfi = _coerce_crash_idx(new_view)
    loc_diff = diff_location(old_frames, new_frames, old_cfi, new_cfi)

    # Locals diff from crash frames
    old_crash = (
        old_frames[old_cfi] if old_frames and 0 <= old_cfi < len(old_frames) else _EMPTY_DICT
    )
    new_crash = (
        new_frames[new_cfi] if new_frames and 0 <= new_cfi < len(new_frames) else _EMPTY_DICT
    )
    locals_diff = diff_locals(old_crash, new_crash)

    # Git
    old_git = _as_dict(old_view_t.get("git"))
    new_git = _as_dict(new_view_t.get("git"))
    git_changed = old_git.get("commit", "") != new_git.get("commit", "")

    # Test identity
    old_nodeid = _as_dict(old_view_t.get("pytest")).get("nodeid", "")
    new_nodeid = _as_dict(new_view_t.get("pytest")).get("nodeid", "")
    same_test = old_nodeid == new_nodeid

    evidence_delta = _diff_evidence(old_view, new_view)
    change_delta = _diff_change_delta(old_view, new_view, location_changed=loc_diff.changed)
    old_hypothesis = _extract_hypothesis(old_view)
    new_hypothesis = _extract_hypothesis(new_view)
    hypothesis_delta = _classify_hypothesis_delta(old_hypothesis, new_hypothesis)
    old_verification_status = _extract_verification_status(old_view)
    new_verification_status = _extract_verification_status(new_view)
    verification_delta = VerificationDelta(
        old_status=old_verification_status,
        new_status=new_verification_status,
        status_changed=old_verification_status != new_verification_status,
    )
    failure_signature_same = _failure_signature(old_view) == _failure_signature(new_view)

    return SnapshotDiffResult(
        exception=exc_diff,
        location=loc_diff,
        locals_diff=locals_diff,
        git_changed=git_changed,
        old_git=old_git,
        new_git=new_git,
        old_name=old_view_t.get("name", "?"),
        new_name=new_view_t.get("name", "?"),
        same_test=same_test,
        failure_signature_same=failure_signature_same,
        evidence_delta=evidence_delta,
        change_delta=change_delta,
        hypothesis_delta=hypothesis_delta,
        verification_delta=verification_delta,
    )


# ---------------------------------------------------------------------------
# Text formatter
# ---------------------------------------------------------------------------


def _truncate(value: Any, max_len: int = 80) -> str:
    """Stringify and truncate a value for display."""
    s = str(value)
    return s[: max_len - 3] + "..." if len(s) > max_len else s


def structured_diff_summary(result: SnapshotDiffResult) -> dict[str, Any]:
    """Return machine-oriented structured diff summary for prompts/MCP."""
    return {
        "attempt_prev": result.old_name,
        "attempt_curr": result.new_name,
        "failure_signature_same": result.failure_signature_same,
        "evidence_delta": {
            "new_frames": list(result.evidence_delta.new_frames),
            "dropped_frames": list(result.evidence_delta.dropped_frames),
            "message_changed": result.evidence_delta.message_changed,
        },
        "change_delta": {
            "touched_files": list(result.change_delta.touched_files),
            "touched_functions": list(result.change_delta.touched_functions),
        },
        "hypothesis_delta": result.hypothesis_delta,
        "verification_delta": {
            "old_status": result.verification_delta.old_status,
            "new_status": result.verification_delta.new_status,
            "status_changed": result.verification_delta.status_changed,
        },
    }


def _fmt_exception_section(exc: ExceptionDiff) -> str:
    if exc.changed:
        lines = [f"## Exception\nCHANGED: {exc.old_type} -> {exc.new_type}"]
        if exc.old_message != exc.new_message:
            lines.append(f"  old: {_truncate(exc.old_message)}")
            lines.append(f"  new: {_truncate(exc.new_message)}")
        if exc.old_category != exc.new_category:
            lines.append(f"  category: {exc.old_category} -> {exc.new_category}")
        return "\n".join(lines)
    return f"## Exception\nUNCHANGED: {exc.old_type}: {_truncate(exc.old_message)}"


def _fmt_location_section(loc: LocationDiff) -> str:
    if loc.changed:
        return f"## Crash Location\nMOVED: {loc.old_location} -> {loc.new_location}"
    return f"## Crash Location\nUNCHANGED: {loc.old_location}"


def _fmt_local_change_line(c: LocalChange) -> str:
    if c.kind == "modified":
        return f"MODIFIED: {c.name}  {_truncate(c.old_value)} -> {_truncate(c.new_value)}"
    if c.kind == "added":
        return f"ADDED: {c.name} = {_truncate(c.new_value)}"
    if c.kind == "removed":
        return f"REMOVED: {c.name}  (was: {_truncate(c.old_value)})"
    if c.kind == "type_changed":
        return (
            f"TYPE CHANGED: {c.name}  {c.old_type} -> {c.new_type}"
            f"  ({_truncate(c.old_value)} -> {_truncate(c.new_value)})"
        )
    # shape_changed
    empty_flag = ""
    new_val: Any = c.new_value
    if isinstance(new_val, list):
        nv_list = cast(list[Any], new_val)
        if nv_list and nv_list[0] == 0:
            empty_flag = "  !! EMPTY"
    return f"SHAPE CHANGED: {c.name}  {c.old_value} -> {new_val}{empty_flag}"


def _fmt_locals_section(ld: LocalsDiff) -> str | None:
    if not ld.changes and ld.unchanged_count == 0:
        return None
    lines = ["## Locals (crash frame)"]
    lines.extend(_fmt_local_change_line(c) for c in ld.changes)
    if ld.unchanged_count > 0:
        lines.append(f"({ld.unchanged_count} unchanged)")
    return "\n".join(lines)


def _fmt_git_section(result: SnapshotDiffResult) -> str | None:
    if not result.git_changed:
        return None
    old_commit = result.old_git.get("commit", "?")
    new_commit = result.new_git.get("commit", "?")
    return f"## Git\ncommit: {old_commit} -> {new_commit}"


def _fmt_structured_section(result: SnapshotDiffResult) -> str:
    structured = structured_diff_summary(result)
    return (
        "## Structured Delta\n"
        f"failure_signature_same: {structured['failure_signature_same']}\n"
        f"hypothesis_delta: {structured['hypothesis_delta']}\n"
        "verification_status: "
        f"{structured['verification_delta']['old_status']} -> "
        f"{structured['verification_delta']['new_status']}"
    )


def _has_structured_changes(result: SnapshotDiffResult) -> bool:
    if result.evidence_delta.new_frames or result.evidence_delta.dropped_frames:
        return True
    if result.evidence_delta.message_changed:
        return True
    if result.change_delta.touched_files or result.change_delta.touched_functions:
        return True
    if result.hypothesis_delta not in {"unknown", "unchanged"}:
        return True
    if result.verification_delta.status_changed:
        return True
    if not result.failure_signature_same:
        return True
    return not result.same_test


def format_diff_text(result: SnapshotDiffResult) -> str:
    """Render a SnapshotDiffResult as structured text for MCP/CLI fallback."""
    if (
        not result.exception.changed
        and not result.location.changed
        and not result.locals_diff.changes
        and not result.git_changed
        and not _has_structured_changes(result)
    ):
        return "No changes detected. Snapshots are identical."

    sections: list[str] = [
        f"## Snapshot Diff\nold: {result.old_name}\nnew: {result.new_name}",
        _fmt_exception_section(result.exception),
        _fmt_location_section(result.location),
    ]
    optional: list[str | None] = [
        _fmt_locals_section(result.locals_diff),
        _fmt_git_section(result),
    ]
    sections.extend(s for s in optional if s is not None)
    sections.append(_fmt_structured_section(result))
    return "\n\n".join(sections)
