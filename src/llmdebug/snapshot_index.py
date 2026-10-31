"""Snapshot files and their index rows: one definition for writers and readers (RCA-03).

The snapshot index (the ``snapshots`` table in the state database) is a cache of
per-file metadata. Writers add rows on capture, readers use it to list snapshots
without parsing every file, and both need the same notion of which files count.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, cast

from ._debug_session import normalize_snapshot_view
from ._formatting import frame_file_label
from .state_db import list_snapshot_records

SNAPSHOT_SUFFIXES = frozenset({".json", ".toon"})


def is_snapshot_name(name: str) -> bool:
    """Return whether a file name denotes a snapshot file.

    Args:
        name: Bare file name, without directory components.

    Returns:
        ``True`` for ``.json``/``.toon`` names that are neither a ``latest.*`` pointer
        nor an in-progress ``.tmp`` write.
    """
    return (
        Path(name).suffix in SNAPSHOT_SUFFIXES
        and not name.startswith("latest")
        and not name.endswith(".tmp")
    )


def traceback_sidecar(snapshot_path: Path) -> Path:
    """Return the path of the ``.traceback.txt`` file written next to a snapshot.

    Replaces only the final suffix: snapshot names keep dots (``test_mod.py``), and
    ``with_suffix("").with_suffix(...)`` cut the name at the first of them (CAP-05).

    Args:
        snapshot_path: Path of the snapshot file.

    Returns:
        Sibling path named ``<snapshot stem>.traceback.txt``.
    """
    return snapshot_path.with_name(f"{snapshot_path.stem}.traceback.txt")


def is_confined_to(path: Path, out_dir: str | Path) -> bool:
    """Return whether *path* still sits inside *out_dir* after following symlinks (RCA-02).

    MCP callers only get to name a directory, so everything read from inside it -- the
    ``latest.*`` pointer and the files a scan finds -- has to be checked too, or a
    symlink planted in an allowed directory reads an arbitrary file.

    Args:
        path: Candidate file path, possibly a symlink.
        out_dir: Directory the resolved path must stay within.

    Returns:
        ``True`` when the resolved path is inside the resolved *out_dir*; ``False``
        otherwise, including when resolution fails.
    """
    try:
        return path.resolve().is_relative_to(Path(out_dir).resolve())
    except OSError:
        return False


def snapshot_files(out_dir: str | Path) -> list[Path]:
    """List the snapshot files directly inside *out_dir*.

    Uses one directory scan, so no file is opened or parsed. Symlinks are resolved
    (and dropped when they leave *out_dir*); plain entries cannot escape, so the
    capture-path callers pay nothing extra for the check.

    Args:
        out_dir: Snapshot directory to scan (not recursively).

    Returns:
        Paths of the snapshot files in scan order; empty if the directory cannot be
        listed.
    """
    try:
        with os.scandir(out_dir) as entries:
            return [
                Path(entry.path)
                for entry in entries
                if is_snapshot_name(entry.name)
                and entry.is_file()
                and (not entry.is_symlink() or is_confined_to(Path(entry.path), out_dir))
            ]
    except OSError:
        return []


def index_covers_disk(out_dir: str | Path) -> bool:
    """Return whether every snapshot file on disk has an index row.

    Index writes are best-effort, so the index can miss files (for example under
    pytest-xdist contention), and a listing that cannot trust the index has to rescan
    the directory. Equal row and file counts do not establish that trust: a stale row
    for a deleted file balances out a missing row for a new one, and a reader that
    stopped at its limit before reaching the stale row never pruned it, so the newer
    snapshot stayed hidden for good (RCA-03).

    Args:
        out_dir: Snapshot directory whose files and index are compared.

    Returns:
        ``True`` when the directory has no snapshot files or every resolved file path
        appears in the index; ``False`` when a file is missing from the index or the
        index cannot be read.
    """
    disk_paths = {str(path.resolve()) for path in snapshot_files(out_dir)}
    if not disk_paths:
        return True
    try:
        indexed_paths = {str(row.get("path", "")) for row in list_snapshot_records(out_dir)}
    except Exception:
        return False
    return disk_paths <= indexed_paths


def index_entry_from_payload(snapshot_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """Build the index row for a snapshot from its in-memory payload.

    Args:
        snapshot_path: Path of the written snapshot file; supplies the row path and
            modification time.
        payload: Snapshot payload in either the flat or the DebugSession envelope shape.

    Returns:
        Index row with the resolved path, mtime (``0.0`` when the file cannot be
        stat-ed), capture name, exception type, exception message truncated to 80
        characters, and the first frame's file label and line.
    """
    view = normalize_snapshot_view(payload)
    exc = cast(dict[str, Any], view.get("exception") or {})
    frames_raw = view.get("frames")
    frames: list[dict[str, Any]] = (
        cast(list[dict[str, Any]], frames_raw) if isinstance(frames_raw, list) else []
    )
    frame0: dict[str, Any] = frames[0] if frames else {}
    try:
        mtime = snapshot_path.stat().st_mtime
    except OSError:
        mtime = 0.0
    return {
        "path": str(snapshot_path.resolve()),
        "mtime": mtime,
        "name": view.get("name", "?"),
        "exc_type": exc.get("type", "?"),
        "exc_message": (exc.get("message") or "")[:80],
        "file": frame_file_label(frame0),
        "line": frame0.get("line", "?"),
    }
