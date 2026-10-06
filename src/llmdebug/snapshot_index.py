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
from .state_db import count_snapshot_records

SNAPSHOT_SUFFIXES = frozenset({".json", ".toon"})


def is_snapshot_name(name: str) -> bool:
    """Whether a file name is a snapshot (not ``latest.*``, not a ``.tmp`` write)."""
    return (
        Path(name).suffix in SNAPSHOT_SUFFIXES
        and not name.startswith("latest")
        and not name.endswith(".tmp")
    )


def traceback_sidecar(snapshot_path: Path) -> Path:
    """The ``.traceback.txt`` file written next to a snapshot.

    Replaces only the final suffix: snapshot names keep dots (``test_mod.py``), and
    ``with_suffix("").with_suffix(...)`` cut the name at the first of them (CAP-05).
    """
    return snapshot_path.with_name(f"{snapshot_path.stem}.traceback.txt")


def snapshot_files(out_dir: str | Path) -> list[Path]:
    """Snapshot files directly in *out_dir*; empty if it can't be listed.

    Uses one directory scan, so no file is opened or parsed.
    """
    try:
        with os.scandir(out_dir) as entries:
            return [
                Path(entry.path)
                for entry in entries
                if is_snapshot_name(entry.name) and entry.is_file()
            ]
    except OSError:
        return []


def index_matches_disk(out_dir: str | Path) -> bool:
    """Whether the index has as many rows as there are snapshot files on disk.

    Index writes are best-effort, so the index can miss files (for example under
    pytest-xdist contention). A mismatch means the listing must rescan the directory.
    """
    return count_snapshot_records(out_dir) == len(snapshot_files(out_dir))


def index_entry_from_payload(snapshot_path: Path, payload: dict[str, Any]) -> dict[str, Any]:
    """Build the index row for a snapshot from its in-memory payload."""
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
