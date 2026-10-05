"""Persistence adapters for RCA state (sqlite/jsonl)."""

from __future__ import annotations

import os
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Literal, cast

from filelock import FileLock

from . import _json as json
from ._warnings import emit_warning
from .rca_store import (
    history,
    latest_record,
    store_rca_record,
    store_rca_records_bulk,
)

RCABackend = Literal["sqlite", "jsonl"]

_RCA_FILE = "rca_attempts.jsonl"
_RCA_LOCK_FILE = ".llmdebug.rca.lock"
_RCA_BACKEND_ENV = "LLMDEBUG_RCA_BACKEND"
_DEFAULT_BACKEND: RCABackend = "sqlite"
_DEFAULT_MAX_RECORDS = 5000
_REQUIRED_RCA_FIELDS = (
    "attempt_id",
    "session_id",
    "updated_at",
    "created_at",
    "failure_signature",
    "state",
)


def _store_path(out_dir: str) -> Path:
    return Path(out_dir) / _RCA_FILE


@contextmanager
def _store_lock(path: Path) -> Iterator[None]:
    """Serialize RCA writes and migration operations."""
    lock = FileLock(path.parent / _RCA_LOCK_FILE)
    with lock:
        yield


def _backend() -> RCABackend:
    raw = os.getenv(_RCA_BACKEND_ENV, _DEFAULT_BACKEND).strip().lower()
    if raw in {"sqlite", "jsonl"}:
        return cast(RCABackend, raw)
    return _DEFAULT_BACKEND


def max_rca_records() -> int:
    """Return configured max retained RCA records (0 means unlimited)."""
    raw_value = os.getenv("LLMDEBUG_RCA_MAX_RECORDS", str(_DEFAULT_MAX_RECORDS))
    try:
        value = int(raw_value)
    except ValueError:
        return _DEFAULT_MAX_RECORDS
    return max(value, 0)


def _iter_jsonl_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    records: list[dict[str, Any]] = []
    try:
        raw = path.read_text(encoding="utf-8")
    except OSError:
        return []
    for line in raw.splitlines():
        item = line.strip()
        if not item:
            continue
        try:
            data = json.loads(item)
        except json.JSONDecodeError:
            continue
        if isinstance(data, dict):
            records.append(cast(dict[str, Any], data))
    return records


def _is_valid_rca_record(record: dict[str, Any]) -> bool:
    for key in _REQUIRED_RCA_FIELDS:
        value = record.get(key)
        if value is None or not str(value).strip():
            return False
    return True


def _iter_jsonl_lines_reverse(path: Path, *, block_size: int = 8192) -> Iterator[str]:
    """Yield JSONL lines in reverse order without loading the whole file."""
    with path.open("rb") as handle:
        handle.seek(0, os.SEEK_END)
        position = handle.tell()
        pending = b""

        while position > 0:
            read_size = min(block_size, position)
            position -= read_size
            handle.seek(position)
            chunk = handle.read(read_size)
            pending = chunk + pending
            lines = pending.split(b"\n")
            pending = lines[0]
            for raw_line in reversed(lines[1:]):
                line = raw_line.strip()
                if line:
                    yield line.decode("utf-8", errors="replace")

        if pending.strip():
            yield pending.strip().decode("utf-8", errors="replace")


def _latest_jsonl_record(path: Path, *, session_id: str | None = None) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        for line in _iter_jsonl_lines_reverse(path):
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not isinstance(data, dict):
                continue
            data_dict = cast(dict[str, Any], data)
            if session_id and data_dict.get("session_id") != session_id:
                continue
            return data_dict
    except OSError:
        return None
    return None


def _append_record(path: Path, record: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with _store_lock(path):
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")

        max_records = max_rca_records()
        if max_records <= 0:
            return

        try:
            lines = path.read_text(encoding="utf-8").splitlines()
        except OSError:
            return
        if len(lines) <= max_records:
            return

        trimmed = lines[-max_records:]
        try:
            path.write_text("\n".join(trimmed) + "\n", encoding="utf-8")
        except OSError:
            return


def ensure_sqlite_migration(out_dir: str) -> None:
    """Best-effort one-way migration from JSONL store to sqlite store.

    JSONL records are merged by ``attempt_id`` (rows already in sqlite win), then the
    JSONL file is rotated to ``.bak``, so the migration runs once. It used to stop as soon
    as sqlite had any row, leaving the JSONL history unmerged and the file in place, so
    every later read and write paid for the lock and a count query again (RCA-05).
    """
    jsonl_path = _store_path(out_dir)
    if not jsonl_path.exists():
        return

    with _store_lock(jsonl_path):
        if not jsonl_path.exists():
            return

        valid_records = [
            record for record in _iter_jsonl_records(jsonl_path) if _is_valid_rca_record(record)
        ]
        if valid_records:
            store_rca_records_bulk(
                valid_records,
                out_dir=out_dir,
                max_records=max_rca_records(),
                keep_existing=True,
            )

        backup_path = Path(f"{jsonl_path}.bak")
        try:
            backup_path.unlink(missing_ok=True)
            jsonl_path.replace(backup_path)
        except OSError as exc:
            # The merge is idempotent, so it is retried on the next call.
            emit_warning(f"could not rotate {jsonl_path.name} after migration: {exc}")


def _pending_jsonl_records(out_dir: str, session_id: str | None) -> list[dict[str, Any]]:
    """Valid JSONL rows not migrated to sqlite yet (reads don't migrate: no file writes)."""
    return [
        record
        for record in _iter_jsonl_records(_store_path(out_dir))
        if _is_valid_rca_record(record)
        and (not session_id or record.get("session_id") == session_id)
    ]


def get_rca_history(
    *,
    out_dir: str,
    session_id: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Load RCA history, newest-first."""
    if _backend() == "sqlite":
        records = history(out_dir=out_dir, session_id=session_id, limit=limit)
        pending = _pending_jsonl_records(out_dir, session_id)
        if not pending:
            return records
        stored_ids = {record.get("attempt_id") for record in records}
        merged = records + [row for row in pending if row.get("attempt_id") not in stored_ids]
        merged.sort(key=lambda record: str(record.get("updated_at") or ""), reverse=True)
        return merged[:limit] if limit > 0 else merged

    records = _iter_jsonl_records(_store_path(out_dir))
    if session_id:
        records = [record for record in records if record.get("session_id") == session_id]
    records = list(reversed(records))
    if limit > 0:
        records = records[:limit]
    return records


def get_rca_status(*, out_dir: str, session_id: str | None = None) -> dict[str, Any] | None:
    """Return latest RCA attempt for a session."""
    if _backend() == "sqlite":
        stored = latest_record(out_dir=out_dir, session_id=session_id)
        pending = _pending_jsonl_records(out_dir, session_id)
        if not pending:
            return stored
        newest_pending = max(pending, key=lambda record: str(record.get("updated_at") or ""))
        if stored is None or str(newest_pending["updated_at"]) > str(stored.get("updated_at")):
            return newest_pending
        return stored

    return _latest_jsonl_record(_store_path(out_dir), session_id=session_id)


def persist_rca_record(*, out_dir: str, record: Mapping[str, Any]) -> None:
    """Persist one RCA record using the configured backend (migrating JSONL first)."""
    if _backend() == "sqlite":
        ensure_sqlite_migration(out_dir)
        store_rca_record(record, out_dir=out_dir, max_records=max_rca_records())
        return

    _append_record(_store_path(out_dir), record)


def backend_name() -> RCABackend:
    """Expose backend name for targeted tests/diagnostics."""
    return _backend()
