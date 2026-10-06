"""Pure snapshot loading logic, shared by CLI and MCP server.

No click/rich/display dependencies — raises standard exceptions on error.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, cast

from . import _json as json
from ._debug_session import normalize_snapshot_view
from ._formatting import frame_file_label
from ._warnings import emit_warning
from .compact_keys import expand_keys, is_compact_snapshot
from .output import get_latest_snapshot
from .snapshot_index import index_covers_disk, is_confined_to, snapshot_files
from .snapshot_types import SnapshotDict
from .state_db import (
    list_snapshot_records,
    remove_snapshot_records,
    replace_snapshot_records,
    state_db_exists,
)


def _load_snapshot_file_raw(path: Path) -> SnapshotDict:
    """Load a snapshot file as raw expanded dict (no session normalization)."""
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix == ".toon":
        try:
            from .toon_encoder import decode_snapshot_toon

            data = decode_snapshot_toon(path.read_text("utf-8"))
        except ImportError:
            raise
        except Exception as e:
            raise ValueError(f"Failed to decode TOON file {path}: {e}") from e
    else:
        data = json.loads(path.read_text("utf-8"))

    if isinstance(data, dict):
        data_dict = cast(dict[str, Any], data)
        if is_compact_snapshot(data_dict):
            expanded = expand_keys(data_dict)
            if isinstance(expanded, dict):
                cast(dict[str, Any], expanded).pop("__llmdebug_compact__", None)
                return cast(SnapshotDict, expanded)
        return cast(SnapshotDict, data_dict)
    raise ValueError(f"Expected dict in snapshot file {path}, got {type(data).__name__}")


def load_snapshot_from_file(path: Path) -> dict[str, Any]:
    """Load and auto-expand a single snapshot file.

    Raises:
        FileNotFoundError: If path doesn't exist.
        ImportError: If TOON format without toons package.
    """
    return normalize_snapshot_view(cast(dict[str, Any], _load_snapshot_file_raw(path)))


def load_snapshot_from_file_raw(path: Path) -> dict[str, Any]:
    """Load and auto-expand a single snapshot file without normalization."""
    return cast(dict[str, Any], _load_snapshot_file_raw(path))


def _validate_snapshot_path(path: Path, out_dir: str) -> None:
    """Ensure path is within the snapshot directory to prevent path traversal.

    Raises:
        ValueError: If path is outside the allowed directory.
    """
    try:
        resolved = path.resolve()
        allowed = Path(out_dir).resolve()
        resolved.relative_to(allowed)
    except ValueError:
        raise ValueError(f"Path {path} is outside snapshot directory {out_dir}") from None


def load_snapshot(path: str | None, out_dir: str, *, _trust_path: bool = False) -> dict[str, Any]:
    """Load from explicit path or fall back to latest.

    Args:
        path: Explicit path to snapshot file, or None for latest.
        out_dir: Snapshot directory.
        _trust_path: If False (default), validate that path is within out_dir.
            Set True only for trusted internal callers.

    Raises:
        FileNotFoundError: If no snapshots found.
        ValueError: If path is outside the snapshot directory.
    """
    if path is not None:
        p = Path(path)
        if not _trust_path:
            _validate_snapshot_path(p, out_dir)
        return load_snapshot_from_file(p)
    snapshot = get_latest_snapshot(out_dir)
    if snapshot is None:
        raise FileNotFoundError(f"No snapshots found in {out_dir}")
    return snapshot


def load_snapshot_raw(
    path: str | None, out_dir: str, *, _trust_path: bool = False
) -> dict[str, Any]:
    """Load a raw snapshot payload without DebugSession normalization."""
    if path is not None:
        p = Path(path)
        if not _trust_path:
            _validate_snapshot_path(p, out_dir)
        return load_snapshot_from_file_raw(p)
    snapshot = get_latest_snapshot(out_dir, normalize=False)
    if snapshot is None:
        raise FileNotFoundError(f"No snapshots found in {out_dir}")
    return snapshot


def _has_snapshot_files(out_path: Path) -> bool:
    return bool(snapshot_files(out_path))


def _normalize_line_value(value: Any) -> Any:
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.isdigit():
            return int(stripped)
    return value


def _normalize_mtime_value(value: Any) -> float | None:
    try:
        mtime = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(mtime):
        return None
    return mtime


def _latest_pointer_path(out_dir: str) -> Path | None:
    out_path = Path(out_dir)
    latest_json = out_path / "latest.json"
    latest_toon = out_path / "latest.toon"
    candidates: list[tuple[float, Path]] = []
    for path in (latest_json, latest_toon):
        if not path.exists() or not path.is_file():
            continue
        # A pointer that resolves outside the directory is not a snapshot of it (RCA-02).
        if not is_confined_to(path, out_dir):
            continue
        try:
            candidates.append((path.stat().st_mtime, path))
        except OSError:
            candidates.append((0.0, path))
    if not candidates:
        return None
    candidates.sort(key=lambda item: item[0], reverse=True)
    return candidates[0][1]


def _entries_from_index(
    out_dir: str,
    *,
    limit: int | None = None,
    search: str | None = None,
) -> tuple[list[dict[str, Any]], bool, bool]:
    query_limit = None
    if limit is not None and limit > 0:
        # Over-fetch from DB to compensate for stale index entries (deleted files).
        # 4x handles high-churn; +20 floor handles small requested limits.
        query_limit = max(limit * 4, limit + 20)
    try:
        rows = list_snapshot_records(out_dir, limit=query_limit, search=search)
    except Exception as e:
        emit_warning(f"snapshot index read failed: {type(e).__name__}: {e}")
        return [], False, True
    if not rows:
        return [], True, False

    stale_paths: list[str] = []
    entries: list[dict[str, Any]] = []
    for row in rows:
        path_value = str(row.get("path", "")).strip()
        if not path_value:
            continue
        path = Path(path_value)
        if not is_confined_to(path, out_dir) or not path.exists() or not path.is_file():
            stale_paths.append(path_value)
            continue
        mtime = _normalize_mtime_value(row.get("mtime", 0.0))
        if mtime is None:
            stale_paths.append(path_value)
            continue
        entry = {
            "path": path.resolve(),
            "mtime": mtime,
            "name": row.get("name", "?"),
            "exc_type": row.get("exc_type", "?"),
            "exc_message": row.get("exc_message", ""),
            "file": row.get("file", "?"),
            "line": _normalize_line_value(row.get("line", "?")),
        }
        if not _matches_snapshot_search(entry, search):
            continue
        entries.append(entry)
        if limit is not None and limit > 0 and len(entries) >= limit:
            break
    if stale_paths:
        try:
            remove_snapshot_records(out_dir, stale_paths)
        except Exception as e:
            emit_warning(f"failed to prune stale snapshot index rows: {type(e).__name__}: {e}")
    return entries, True, False


def _matches_snapshot_search(entry: dict[str, Any], search: str | None) -> bool:
    query = str(search or "").strip().lower()
    if not query:
        return True
    haystacks = (
        str(entry.get("name") or ""),
        str(entry.get("exc_type") or ""),
        str(entry.get("exc_message") or ""),
        str(entry.get("file") or ""),
    )
    return any(query in text.lower() for text in haystacks)


def _scan_snapshot_entries(out_dir: str) -> tuple[list[dict[str, Any]], int]:
    candidates: list[tuple[float, Path]] = []
    for path in snapshot_files(out_dir):
        try:
            candidates.append((path.stat().st_mtime, path))
        except OSError:
            continue

    candidates.sort(key=lambda item: item[0], reverse=True)
    entries: list[dict[str, Any]] = []
    skipped = 0
    for mtime, path in candidates:
        try:
            data = load_snapshot_from_file(path)
            exc = cast(dict[str, Any], data.get("exception") or {})
            frames = cast(list[dict[str, Any]], data.get("frames") or [{}])
            fr0: dict[str, Any] = frames[0] if frames else {}
            entries.append(
                {
                    "path": path.resolve(),
                    "mtime": mtime,
                    "name": data.get("name", "?"),
                    "exc_type": exc.get("type", "?"),
                    "exc_message": (exc.get("message") or "")[:80],
                    "file": frame_file_label(fr0),
                    "line": fr0.get("line", "?"),
                }
            )
        except Exception:
            skipped += 1
            continue
    return entries, skipped


def _refresh_snapshot_index(out_dir: str, entries: list[dict[str, Any]]) -> None:
    try:
        replace_snapshot_records(
            out_dir,
            [
                {
                    "path": str(entry["path"].resolve()),
                    "mtime": float(entry["mtime"]),
                    "name": entry["name"],
                    "exc_type": entry["exc_type"],
                    "exc_message": entry["exc_message"],
                    "file": entry["file"],
                    "line": str(entry["line"]),
                }
                for entry in entries
            ],
        )
    except Exception as e:
        emit_warning(f"failed to refresh snapshot index: {type(e).__name__}: {e}")


def find_all_snapshots(
    out_dir: str,
    *,
    limit: int | None = None,
    search: str | None = None,
) -> list[dict[str, Any]]:
    """Scan output dir, return snapshot metadata sorted newest-first.

    Args:
        out_dir: Snapshot output directory.
        limit: Optional max entries to return. If None, returns all.
        search: Optional case-insensitive metadata substring filter.

    Returns [] if dir doesn't exist. Silently skips corrupted files.
    """
    out_path = Path(out_dir)
    if not out_path.is_dir():
        return []

    effective_limit = limit if limit is not None and limit > 0 else None

    indexed_entries, index_ok, index_read_failed = _entries_from_index(
        out_dir, limit=effective_limit, search=search
    )
    # A partially populated index (best-effort writes) used to hide the files it missed
    # for good; it is only trusted while it covers every snapshot on disk (RCA-03).
    if (
        indexed_entries
        and (effective_limit is None or len(indexed_entries) >= effective_limit)
        and index_covers_disk(out_dir)
    ):
        return indexed_entries[:effective_limit] if effective_limit is not None else indexed_entries

    if index_ok and not indexed_entries and not _has_snapshot_files(out_path):
        return []

    if index_read_failed:
        emit_warning("snapshot index unavailable; falling back to filesystem scan")

    # Fallback scan for index miss/corruption. Only an existing index is rebuilt: reads
    # never create the state database (RCA-02).
    entries, skipped = _scan_snapshot_entries(out_dir)
    if state_db_exists(out_dir):
        _refresh_snapshot_index(out_dir, entries)

    if skipped > 0:
        emit_warning(f"skipped {skipped} corrupted snapshot(s) in {out_dir}")

    if search:
        entries = [entry for entry in entries if _matches_snapshot_search(entry, search)]

    if effective_limit is None:
        return entries
    if len(entries) <= effective_limit:
        return entries
    return entries[:effective_limit]


def resolve_snapshot_ref_path(
    ref: str | None,
    out_dir: str,
    all_snapshots: list[dict[str, Any]] | None = None,
    *,
    _trust_path: bool = False,
) -> Path:
    """Resolve a snapshot reference to a concrete snapshot file path."""
    if ref is None or ref == "latest":
        latest_pointer = _latest_pointer_path(out_dir)
        if latest_pointer is not None:
            return latest_pointer
        snapshots = (
            all_snapshots if all_snapshots is not None else find_all_snapshots(out_dir, limit=1)
        )
        if snapshots:
            return Path(str(snapshots[0]["path"]))
        raise FileNotFoundError(f"No snapshots found in {out_dir}")

    if ref == "previous":
        snapshots = (
            all_snapshots if all_snapshots is not None else find_all_snapshots(out_dir, limit=2)
        )
        if len(snapshots) < 2:
            raise FileNotFoundError(
                f"Need at least 2 snapshots to use 'previous', "
                f"but only {len(snapshots)} found in {out_dir}"
            )
        return Path(str(snapshots[1]["path"]))

    if ref.startswith("#"):
        try:
            n = int(ref[1:])
        except ValueError:
            raise ValueError(f"Invalid snapshot reference: {ref!r}") from None
        snapshots = (
            all_snapshots if all_snapshots is not None else find_all_snapshots(out_dir, limit=n)
        )
        if n < 1 or n > len(snapshots):
            raise ValueError(
                f"Snapshot #{n} not found. Only {len(snapshots)} available in {out_dir}"
            )
        return Path(str(snapshots[n - 1]["path"]))

    path = Path(ref)
    if not _trust_path:
        _validate_snapshot_path(path, out_dir)
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")
    return path


def resolve_snapshot_ref(
    ref: str | None,
    out_dir: str,
    all_snapshots: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Resolve a snapshot reference to a loaded snapshot dict.

    Handles:
    - None / "latest" → most recent snapshot
    - "previous" → second most recent
    - "#N" → Nth snapshot from list (1-based)
    - file path → load directly

    Raises:
        FileNotFoundError: If the referenced snapshot doesn't exist.
        ValueError: If the reference is out of range.
    """
    path = resolve_snapshot_ref_path(ref, out_dir, all_snapshots)
    return load_snapshot_from_file(path)
