"""File output and latest.json management."""

from __future__ import annotations

import datetime as dt
import os
import re
import sys
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import timezone
from pathlib import Path
from typing import Any, cast

from filelock import FileLock, Timeout

from . import _json as json
from ._debug_session import normalize_snapshot_view
from ._hashing import xxh64_hexdigest_text
from ._warnings import emit_warning
from .config import SnapshotConfig
from .snapshot_index import index_entry_from_payload, snapshot_files
from .state_db import (
    list_snapshot_records,
    remove_snapshot_records,
    snapshot_index_window,
    upsert_snapshot_record,
)


@contextmanager
def _directory_lock(out_dir: Path, timeout: float) -> Iterator[bool]:
    """Acquire lock for output directory operations.

    Yields True if lock acquired, False if timeout.
    """
    lock = FileLock(out_dir / ".llmdebug.lock", timeout=timeout)
    try:
        lock.acquire()
        yield True
    except Timeout:
        yield False
    finally:
        if lock.is_locked:
            lock.release()


# Keeps "<timestamp>_<name>_<nonce>.traceback.txt" far below the 255-byte filename limit.
_SNAPSHOT_NAME_MAX_CHARS = 120
_UNSAFE_NAME_CHAR = re.compile(r"[^A-Za-z0-9._-]")


def _snapshot_file_name(name: str) -> str:
    """Filesystem-safe form of a snapshot name: ``[A-Za-z0-9._-]``, at most 120 characters.

    A longer name (e.g. a parametrized pytest node ID) is cut and suffixed with a hash of
    the full name, so distinct long names stay distinguishable. The payload keeps the
    original name.
    """
    safe = _UNSAFE_NAME_CHAR.sub("_", name) or "snapshot"
    if len(safe) <= _SNAPSHOT_NAME_MAX_CHARS:
        return safe
    digest = xxh64_hexdigest_text(name)
    return f"{safe[: _SNAPSHOT_NAME_MAX_CHARS - len(digest) - 1]}_{digest}"


def _timestamp() -> str:
    """Generate timestamp string."""
    return dt.datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def _update_latest_symlink(out_dir: Path, target_name: str, ext: str = ".json") -> None:
    """Update latest symlink/copy.

    Args:
        out_dir: Output directory
        target_name: Name of the target file
        ext: File extension for latest file (default: .json)
    """
    latest = out_dir / f"latest{ext}"
    target = out_dir / target_name

    # Clean up stale latest file from other format to avoid confusion
    opposite_ext = ".toon" if ext == ".json" else ".json"
    stale_latest = out_dir / f"latest{opposite_ext}"
    try:
        stale_latest.unlink(missing_ok=True)
    except Exception:  # nosec B110 — best-effort cleanup
        pass  # Best effort cleanup

    if sys.platform == "win32":
        # Windows: atomically replace latest.* after writing a temp file.
        tmp = out_dir / f"latest{ext}.tmp.{os.getpid()}"
        try:
            tmp.write_text(target.read_text(encoding="utf-8"), encoding="utf-8")
            tmp.replace(latest)
        except Exception as e:
            try:
                tmp.unlink(missing_ok=True)
            except Exception:  # nosec B110 — best-effort temp cleanup
                pass
            emit_warning(f"could not update {latest.name}: {type(e).__name__}: {e}")
    else:
        # POSIX: atomic symlink update (PID-unique temp avoids TOCTOU race)
        tmp = out_dir / f".latest{ext}.tmp.{os.getpid()}"
        try:
            tmp.symlink_to(target_name)
            tmp.rename(latest)
        except Exception as e:
            # Clean up partial temp file
            try:
                tmp.unlink(missing_ok=True)
            except Exception:  # nosec B110 — best-effort temp cleanup
                pass
            emit_warning(f"could not update {latest.name}: {type(e).__name__}: {e}")


def _delete_snapshot_artifacts(snapshot: Path) -> bool:
    """Delete a snapshot file and its traceback sidecar."""
    try:
        snapshot.unlink(missing_ok=True)
    except Exception as e:
        emit_warning(f"could not clean up {snapshot.name}: {type(e).__name__}: {e}")
        return False

    traceback_file = snapshot.with_suffix("").with_suffix(".traceback.txt")
    try:
        traceback_file.unlink(missing_ok=True)
    except Exception as e:
        emit_warning(f"could not clean up {snapshot.name}: {type(e).__name__}: {e}")
    return True


def _remove_deleted_snapshot_records(out_dir: Path, deleted_paths: list[str]) -> None:
    """Remove snapshot index rows for files that were deleted."""
    if not deleted_paths:
        return
    try:
        remove_snapshot_records(str(out_dir), deleted_paths)
    except Exception as index_error:
        emit_warning(
            "snapshot index cleanup failed for "
            f"{len(deleted_paths)} file(s): "
            f"{type(index_error).__name__}: {index_error}"
        )


def _cleanup_indexed_snapshots(
    out_dir: Path, indexed: list[dict[str, Any]], max_snapshots: int
) -> bool:
    """Remove overflow snapshots using the persisted index ordering."""
    if len(indexed) <= max_snapshots:
        return False

    deleted_snapshot_paths: list[str] = []
    for record in indexed[max_snapshots:]:
        record_path = str(record["path"])
        if _delete_snapshot_artifacts(Path(record_path)):
            deleted_snapshot_paths.append(record_path)

    _remove_deleted_snapshot_records(out_dir, deleted_snapshot_paths)
    return True


def _cleanup_filesystem_snapshots(out_dir: Path, max_snapshots: int) -> None:
    """Remove overflow snapshots by scanning the output directory."""
    snapshots: list[tuple[float, Path]] = []
    for path in snapshot_files(out_dir):
        try:
            snapshots.append((path.stat().st_mtime, path))
        except OSError:
            continue
    if len(snapshots) <= max_snapshots:
        return

    snapshots.sort(key=lambda item: item[0])
    overflow = len(snapshots) - max_snapshots
    deleted_fs_paths = [
        str(snapshot.resolve())
        for _, snapshot in snapshots[:overflow]
        if _delete_snapshot_artifacts(snapshot)
    ]

    _remove_deleted_snapshot_records(out_dir, deleted_fs_paths)


def _cleanup_old_snapshots(out_dir: Path, max_snapshots: int) -> None:
    """Remove old snapshots beyond the limit.

    While the snapshot index covers every file on disk, it answers "over the limit?" with
    a ``LIMIT max_snapshots + 1`` query and the cleanup uses its ordering. Only a missing,
    incomplete or unreadable index falls back to ``iterdir()`` + ``stat()``; that used to
    happen on every write while under the limit (CAP-15).
    """
    if max_snapshots <= 0:
        return  # Unlimited

    try:
        indexed_count, newest = snapshot_index_window(str(out_dir), limit=max_snapshots + 1)
        index_healthy = indexed_count == len(snapshot_files(out_dir))
    except Exception:
        index_healthy, newest = False, []

    if index_healthy and len(newest) <= max_snapshots:
        return
    if index_healthy:
        try:
            indexed = list_snapshot_records(str(out_dir))
        except Exception:
            indexed = []
        if indexed and _cleanup_indexed_snapshots(out_dir, indexed, max_snapshots):
            return

    _cleanup_filesystem_snapshots(out_dir, max_snapshots)


def write_bundle(payload: dict[str, Any], cfg: SnapshotConfig) -> Path:
    """Write snapshot bundle to disk in configured format."""
    out_dir = Path(cfg.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    name_value = payload.get("name")
    if not isinstance(name_value, str):
        session = payload.get("session")
        if isinstance(session, dict):
            session_name = cast(dict[str, Any], session).get("name")
            name_value = session_name if isinstance(session_name, str) else "snapshot"
        else:
            name_value = "snapshot"
    name = _snapshot_file_name(name_value)
    # Add a short nonce to avoid same-second name collisions.
    base = f"{_timestamp()}_{name}_{uuid.uuid4().hex[:8]}"

    # Determine content and extension based on format
    if cfg.output_format == "json":
        content = json.dumps(payload, indent=2, ensure_ascii=False, default=str)
        ext = ".json"
    elif cfg.output_format == "json_compact":
        from .compact_keys import compact_keys

        compacted = compact_keys(payload)
        if isinstance(compacted, dict):
            # Explicit marker avoids ambiguous compact-format detection on read.
            compacted["__llmdebug_compact__"] = 1
        content = json.dumps(compacted, separators=(",", ":"), ensure_ascii=False, default=str)
        ext = ".json"  # Still valid JSON, just minified with short keys
    elif cfg.output_format == "toon":
        from .toon_encoder import encode_snapshot_toon

        content = encode_snapshot_toon(payload)
        ext = ".toon"
    else:
        raise ValueError(f"Unknown output_format: {cfg.output_format}")

    snapshot_path = out_dir / f"{base}{ext}"

    # Atomic write
    tmp = snapshot_path.with_suffix(".tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.rename(snapshot_path)
    try:
        upsert_snapshot_record(str(out_dir), index_entry_from_payload(snapshot_path, payload))
    except Exception as index_error:
        emit_warning(
            "snapshot index update failed for "
            f"{snapshot_path.name}: {type(index_error).__name__}: {index_error}"
        )

    # Write human-readable traceback (legacy flat payload or DebugSession envelope).
    traceback_text: str | None = None
    if isinstance(payload.get("traceback"), str):
        traceback_text = str(payload["traceback"])
    else:
        snapshot = payload.get("snapshot")
        if isinstance(snapshot, dict):
            tb_raw = cast(dict[str, Any], snapshot).get("traceback")
            if isinstance(tb_raw, str):
                traceback_text = tb_raw
    if traceback_text is not None:
        tb_path = out_dir / f"{base}.traceback.txt"
        tb_path.write_text(traceback_text, encoding="utf-8")

    # Lock for symlink update and cleanup (these can race with concurrent writers)
    with _directory_lock(out_dir, cfg.lock_timeout) as acquired:
        if acquired:
            _update_latest_symlink(out_dir, snapshot_path.name, ext)
            _cleanup_old_snapshots(out_dir, cfg.max_snapshots)
        else:
            emit_warning("lock timeout — latest symlink may be stale")

    return snapshot_path


def get_latest_snapshot(
    out_dir: str = ".llmdebug",
    *,
    detail: str | None = None,
    normalize: bool = True,
) -> dict[str, Any] | None:
    """Read the latest snapshot if it exists, auto-detecting format.

    If both latest.json and latest.toon exist (e.g. stale leftovers from
    a previous run), prefers the newest by mtime. Automatically expands
    compact JSON keys to full keys for compatibility.

    Args:
        out_dir: Output directory containing snapshots
        detail: Optional detail level filter ("crash", "full", "context").
                None means no filtering (returns full snapshot).
        normalize: If True (default), normalize DebugSession payloads into the
                legacy flat snapshot view. If False, return raw expanded payload.

    Returns:
        Snapshot data with full keys, or None if not found
    """
    if detail is not None and not normalize:
        raise ValueError("detail filter requires normalize=True")

    out_path = Path(out_dir)

    latest_json = out_path / "latest.json"
    latest_toon = out_path / "latest.toon"
    candidates: list[tuple[str, Path, float]] = []
    for kind, path in (("json", latest_json), ("toon", latest_toon)):
        if not path.exists():
            continue
        try:
            candidates.append((kind, path, path.stat().st_mtime))
        except Exception:
            candidates.append((kind, path, 0.0))

    if not candidates:
        return None

    candidates.sort(key=lambda t: t[2], reverse=True)

    def _read_json(path: Path) -> dict[str, Any] | None:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            emit_warning(f"{path.name} is corrupted: {type(e).__name__}: {e}")
            return None
        if not isinstance(data, dict):
            emit_warning(f"{path.name} contains {type(data).__name__}, expected dict")
            return None
        from .compact_keys import expand_keys, is_compact_snapshot

        data_dict = cast(dict[str, Any], data)
        if is_compact_snapshot(data_dict):
            try:
                expanded = expand_keys(data_dict)
            except Exception as e:
                emit_warning(f"{path.name} compact key expansion failed: {type(e).__name__}: {e}")
                return None
            if isinstance(expanded, dict):
                cast(dict[str, Any], expanded).pop("__llmdebug_compact__", None)
                return (
                    normalize_snapshot_view(cast(dict[str, Any], expanded))
                    if normalize
                    else cast(dict[str, Any], expanded)
                )
            return None
        return normalize_snapshot_view(data_dict) if normalize else data_dict

    def _read_toon(path: Path) -> dict[str, Any] | None:
        try:
            from .toon_encoder import decode_snapshot_toon

            data = decode_snapshot_toon(path.read_text(encoding="utf-8"))
            if isinstance(data, dict):  # pyright: ignore[reportUnnecessaryIsInstance]
                return normalize_snapshot_view(data) if normalize else data
            return None
        except Exception as e:
            emit_warning(f"{path.name} decode failed: {type(e).__name__}: {e}")
            return None

    for kind, path, _ in candidates:
        snapshot = _read_json(path) if kind == "json" else _read_toon(path)
        if snapshot is not None:
            if detail is not None:
                from .detail_filter import filter_snapshot

                snapshot = filter_snapshot(snapshot, detail)  # type: ignore[arg-type]
            return snapshot

    return None
