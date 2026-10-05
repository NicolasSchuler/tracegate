"""SQLite state database helpers for RCA and snapshot metadata."""

from __future__ import annotations  # pragma: no cover

import sqlite3  # pragma: no cover
import time  # pragma: no cover
from collections.abc import Iterator  # pragma: no cover
from contextlib import contextmanager  # pragma: no cover
from pathlib import Path  # pragma: no cover
from threading import Lock  # pragma: no cover
from typing import Any, cast  # pragma: no cover

from . import _json as json  # pragma: no cover
from ._coerce import read_env_int  # pragma: no cover

STATE_DB_FILENAME = "llmdebug_state.sqlite3"  # pragma: no cover
_BUSY_TIMEOUT_MS = 30_000  # pragma: no cover
_WAL_RETRY_ATTEMPTS = 8  # pragma: no cover
_WAL_BASE_SLEEP_SEC = 0.01  # pragma: no cover
_WAL_INIT_LOCK = Lock()  # pragma: no cover
_WAL_INITIALIZED_PATHS: set[str] = set()  # pragma: no cover


def state_db_path(out_dir: str | Path) -> Path:
    """Return the SQLite database path for a snapshot output directory."""
    return Path(out_dir) / STATE_DB_FILENAME


def _enable_wal_mode(
    conn: sqlite3.Connection,
    *,
    attempts: int = _WAL_RETRY_ATTEMPTS,
    base_sleep_sec: float = _WAL_BASE_SLEEP_SEC,
) -> None:
    """Set WAL mode with retries to tolerate brief startup lock contention."""
    total_attempts = max(1, attempts)
    for attempt in range(total_attempts):
        try:
            conn.execute("PRAGMA journal_mode=WAL")
            return
        except sqlite3.OperationalError as exc:
            if "locked" not in str(exc).lower() or attempt == total_attempts - 1:
                raise
            if base_sleep_sec > 0:
                time.sleep(base_sleep_sec * float(attempt + 1))


def _configure_wal_mode_once(conn: sqlite3.Connection, *, db_path: Path) -> None:
    key = str(db_path.resolve())
    with _WAL_INIT_LOCK:
        if key in _WAL_INITIALIZED_PATHS:
            return
        _enable_wal_mode(conn)
        _WAL_INITIALIZED_PATHS.add(key)


def _configure_connection(conn: sqlite3.Connection, *, db_path: Path) -> None:
    busy_timeout = read_env_int(
        "LLMDEBUG_DB_BUSY_TIMEOUT_MS",
        _BUSY_TIMEOUT_MS,
        minimum=0,
    )
    conn.execute(f"PRAGMA busy_timeout={busy_timeout}")
    _configure_wal_mode_once(conn, db_path=db_path)
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")


def _ensure_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS rca_attempts (
            attempt_id TEXT PRIMARY KEY,
            session_id TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            created_at TEXT NOT NULL,
            failure_signature TEXT NOT NULL,
            state TEXT NOT NULL,
            payload_json TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_rca_attempts_session_updated
            ON rca_attempts(session_id, updated_at DESC)
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_rca_attempts_updated
            ON rca_attempts(updated_at DESC)
        """
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS snapshots (
            path TEXT PRIMARY KEY,
            mtime REAL NOT NULL,
            name TEXT NOT NULL,
            exc_type TEXT NOT NULL,
            exc_message TEXT NOT NULL,
            file TEXT NOT NULL,
            line TEXT NOT NULL
        )
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_snapshots_mtime
            ON snapshots(mtime DESC)
        """
    )


@contextmanager
def open_state_db(out_dir: str | Path) -> Iterator[sqlite3.Connection]:
    """Open the state DB connection with schema initialization."""
    path = state_db_path(out_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30.0)
    try:
        _configure_connection(conn, db_path=path)
        _ensure_schema(conn)
        yield conn
        conn.commit()
    finally:
        conn.close()


def count_rca_records(out_dir: str | Path) -> int:
    """Return the number of persisted RCA attempt records for ``out_dir``."""
    with open_state_db(out_dir) as conn:
        row = conn.execute("SELECT COUNT(*) FROM rca_attempts").fetchone()
    return int(row[0]) if row else 0


def _require_text_field(record: dict[str, Any], key: str) -> str:
    raw_value = record.get(key)
    if raw_value is None:
        raise ValueError(f"RCA record missing required field: {key}")
    value = str(raw_value).strip()
    if not value:
        raise ValueError(f"RCA record has empty required field: {key}")
    return value


def _rca_row(record: dict[str, Any]) -> tuple[str, str, str, str, str, str, str]:
    payload = json.dumps(record, ensure_ascii=False, sort_keys=True)
    return (
        _require_text_field(record, "attempt_id"),
        _require_text_field(record, "session_id"),
        _require_text_field(record, "updated_at"),
        _require_text_field(record, "created_at"),
        _require_text_field(record, "failure_signature"),
        _require_text_field(record, "state"),
        payload,
    )


def _prune_rca_records(conn: sqlite3.Connection, *, max_records: int) -> None:
    if max_records <= 0:
        return
    conn.execute(
        """
        DELETE FROM rca_attempts
        WHERE rowid IN (
            SELECT rowid
            FROM rca_attempts
            ORDER BY updated_at DESC, rowid DESC
            LIMIT -1 OFFSET ?
        )
        """,
        (max_records,),
    )


def append_rca_record(out_dir: str | Path, record: dict[str, Any], *, max_records: int) -> None:
    """Insert or replace one RCA record, then prune to the retention limit.

    Records are retained in ``updated_at DESC, rowid DESC`` order. Validation
    errors for missing required fields raise ``ValueError`` before any write.
    """
    row = _rca_row(record)
    with open_state_db(out_dir) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO rca_attempts(
                attempt_id, session_id, updated_at, created_at, failure_signature, state, payload_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            row,
        )
        _prune_rca_records(conn, max_records=max_records)


def append_rca_records_bulk(
    out_dir: str | Path,
    records: list[dict[str, Any]],
    *,
    max_records: int,
    keep_existing: bool = False,
) -> None:
    """Insert multiple RCA records, then prune by recency.

    A record whose ``attempt_id`` is already stored replaces it, or is skipped with
    ``keep_existing=True`` (a merge where the stored row wins).
    """
    if not records:
        return
    rows = [_rca_row(record) for record in records]
    conflict = "IGNORE" if keep_existing else "REPLACE"
    with open_state_db(out_dir) as conn:
        conn.executemany(
            f"""
            INSERT OR {conflict} INTO rca_attempts(
                attempt_id, session_id, updated_at, created_at, failure_signature, state, payload_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        _prune_rca_records(conn, max_records=max_records)


def _decode_payload_rows(rows: list[tuple[str]]) -> list[dict[str, Any]]:
    decoded: list[dict[str, Any]] = []
    for (payload_json,) in rows:
        try:
            payload = json.loads(payload_json)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            decoded.append(cast(dict[str, Any], payload))
    return decoded


def fetch_latest_rca_record(
    out_dir: str | Path, *, session_id: str | None = None
) -> dict[str, Any] | None:
    """Return the newest decodable RCA record for the directory or session.

    Newest is defined by ``updated_at DESC, rowid DESC``. Returns ``None`` when
    no matching row exists or when stored payloads fail JSON decoding.
    """
    sql = "SELECT payload_json FROM rca_attempts"
    params: tuple[Any, ...] = ()
    if session_id:
        sql += " WHERE session_id = ?"
        params = (session_id,)
    sql += " ORDER BY updated_at DESC, rowid DESC LIMIT 1"
    with open_state_db(out_dir) as conn:
        rows = conn.execute(sql, params).fetchall()
    decoded = _decode_payload_rows(rows)
    return decoded[0] if decoded else None


def fetch_rca_history_records(
    out_dir: str | Path,
    *,
    session_id: str | None = None,
    limit: int = 20,
) -> list[dict[str, Any]]:
    """Return RCA history records ordered from newest to oldest.

    ``limit <= 0`` means "all matching records". Rows with malformed JSON are
    skipped so one corrupt entry does not break history inspection.
    """
    sql = "SELECT payload_json FROM rca_attempts"
    params: list[Any] = []
    if session_id:
        sql += " WHERE session_id = ?"
        params.append(session_id)
    sql += " ORDER BY updated_at DESC, rowid DESC"
    if limit > 0:
        sql += " LIMIT ?"
        params.append(limit)
    with open_state_db(out_dir) as conn:
        rows = conn.execute(sql, tuple(params)).fetchall()
    return _decode_payload_rows(rows)


def upsert_snapshot_record(out_dir: str | Path, record: dict[str, Any]) -> None:
    """Insert or replace one snapshot index row for CLI/MCP lookup."""
    with open_state_db(out_dir) as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO snapshots(path, mtime, name, exc_type, exc_message, file, line)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                str(record.get("path", "")),
                float(record.get("mtime", 0.0)),
                str(record.get("name", "?")),
                str(record.get("exc_type", "?")),
                str(record.get("exc_message", "")),
                str(record.get("file", "?")),
                str(record.get("line", "?")),
            ),
        )


def remove_snapshot_records(out_dir: str | Path, paths: list[str]) -> None:
    """Delete snapshot index rows for the provided on-disk paths."""
    if not paths:
        return
    with open_state_db(out_dir) as conn:
        conn.executemany("DELETE FROM snapshots WHERE path = ?", [(path,) for path in paths])


def list_snapshot_records(
    out_dir: str | Path,
    *,
    limit: int | None = None,
    search: str | None = None,
) -> list[dict[str, Any]]:
    """List indexed snapshots ordered by newest mtime first.

    When ``search`` is provided, matching is case-insensitive across the stored
    name, exception type/message, and file fields.
    """
    sql = """
        SELECT path, mtime, name, exc_type, exc_message, file, line
        FROM snapshots
    """
    params: list[Any] = []
    search_text = str(search or "").strip()
    if search_text:
        pattern = f"%{search_text}%"
        sql += """
        WHERE
            name LIKE ? COLLATE NOCASE
            OR exc_type LIKE ? COLLATE NOCASE
            OR exc_message LIKE ? COLLATE NOCASE
            OR file LIKE ? COLLATE NOCASE
        """
        params.extend((pattern, pattern, pattern, pattern))
    sql += " ORDER BY mtime DESC, path DESC"
    if limit is not None and limit > 0:
        sql += " LIMIT ?"
        params.append(limit)
    with open_state_db(out_dir) as conn:
        rows = conn.execute(sql, tuple(params)).fetchall()
    out: list[dict[str, Any]] = []
    for path, mtime, name, exc_type, exc_message, file_name, line in rows:
        out.append(
            {
                "path": str(path),
                "mtime": float(mtime),
                "name": str(name),
                "exc_type": str(exc_type),
                "exc_message": str(exc_message),
                "file": str(file_name),
                "line": str(line),
            }
        )
    return out


def replace_snapshot_records(out_dir: str | Path, records: list[dict[str, Any]]) -> None:
    """Replace the entire snapshot index with the provided record set."""
    with open_state_db(out_dir) as conn:
        conn.execute("DELETE FROM snapshots")
        if records:
            rows = [
                (
                    str(record.get("path", "")),
                    float(record.get("mtime", 0.0)),
                    str(record.get("name", "?")),
                    str(record.get("exc_type", "?")),
                    str(record.get("exc_message", "")),
                    str(record.get("file", "?")),
                    str(record.get("line", "?")),
                )
                for record in records
            ]
            conn.executemany(
                """
                INSERT OR REPLACE INTO snapshots(path, mtime, name, exc_type, exc_message, file, line)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
