"""SQLite state database helpers for RCA and snapshot metadata."""

from __future__ import annotations

import sqlite3
import time
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any, cast

from . import _json as json
from ._coerce import read_env_int
from ._warnings import emit_warning

STATE_DB_FILENAME = "llmdebug_state.sqlite3"
_BUSY_TIMEOUT_MS = 30_000
_WAL_RETRY_ATTEMPTS = 8
_WAL_BASE_SLEEP_SEC = 0.01
_ID_QUERY_CHUNK = 500
_WAL_WARNED_PATHS: set[str] = set()
_WAL_VERIFIED_PATHS: set[str] = set()


def state_db_path(out_dir: str | Path) -> Path:
    """Return the SQLite database path for a snapshot output directory."""
    return Path(out_dir) / STATE_DB_FILENAME


def state_db_exists(out_dir: str | Path) -> bool:
    """Whether the state database exists. Read helpers never create it (RCA-02)."""
    return state_db_path(out_dir).is_file()


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


def _journal_mode(conn: sqlite3.Connection) -> str:
    row = conn.execute("PRAGMA journal_mode").fetchone()
    return str(row[0]).lower() if row else ""


def _file_identity(db_path: Path) -> str:
    """A key that changes when the file at *db_path* is replaced, not just renamed."""
    try:
        stat = db_path.stat()
    except OSError:
        return str(db_path)
    return f"{db_path}:{stat.st_dev}:{stat.st_ino}"


def _ensure_wal_mode(conn: sqlite3.Connection, *, db_path: Path, created: bool) -> None:
    """Switch the database to WAL unless it already is, and check that it worked (RCA-16).

    The journal mode is stored in the database, so it is checked when this process first
    opens the file and whenever the file was just created: the "done" cache used to
    survive the database being deleted and recreated in rollback mode. It is keyed by
    device and inode, so a replacement this process did not create is re-verified too.
    Reading the mode on every open costs about 0.2 ms, three times per capture. The
    result of the switch is checked (filesystems without shared memory stay in rollback
    mode).
    """
    key = _file_identity(db_path)
    if key in _WAL_VERIFIED_PATHS and not created:
        return
    if _journal_mode(conn) != "wal":
        _enable_wal_mode(conn)
        mode = _journal_mode(conn)
        if mode != "wal":
            if str(db_path) not in _WAL_WARNED_PATHS:
                _WAL_WARNED_PATHS.add(str(db_path))
                emit_warning(
                    f"could not enable WAL mode for {db_path} (journal_mode={mode or '?'}); "
                    "concurrent llmdebug processes may block each other"
                )
            return
    _WAL_VERIFIED_PATHS.add(key)


def _configure_connection(
    conn: sqlite3.Connection, *, db_path: Path, created: bool = False
) -> None:
    busy_timeout = read_env_int(
        "LLMDEBUG_DB_BUSY_TIMEOUT_MS",
        _BUSY_TIMEOUT_MS,
        minimum=0,
    )
    conn.execute(f"PRAGMA busy_timeout={busy_timeout}")
    _ensure_wal_mode(conn, db_path=db_path, created=created)
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA temp_store=MEMORY")


def _create_schema_v1(conn: sqlite3.Connection) -> None:
    """Version 1: the tables and indexes as they existed before versioning (RCA-11).

    ``IF NOT EXISTS`` keeps this a no-op on databases created by earlier releases,
    which all have ``user_version`` 0.
    """
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


# Index i upgrades a database from user_version i to i + 1. Append new migrations; never
# edit a released one.
_SCHEMA_MIGRATIONS: tuple[Callable[[sqlite3.Connection], None], ...] = (_create_schema_v1,)
STATE_DB_SCHEMA_VERSION = len(_SCHEMA_MIGRATIONS)


def _schema_version(conn: sqlite3.Connection) -> int:
    row = conn.execute("PRAGMA user_version").fetchone()
    return int(row[0]) if row else 0


def _require_known_schema(conn: sqlite3.Connection) -> None:
    """Refuse a database written by a newer llmdebug (RCA-11). Reads check this too."""
    version = _schema_version(conn)
    if version > STATE_DB_SCHEMA_VERSION:
        raise RuntimeError(
            f"state database schema version {version} is newer than this llmdebug "
            f"supports ({STATE_DB_SCHEMA_VERSION}); upgrade llmdebug"
        )


def _ensure_schema(conn: sqlite3.Connection) -> None:
    """Bring the schema to ``STATE_DB_SCHEMA_VERSION`` via ``PRAGMA user_version`` (RCA-11).

    Migrations run in one ``BEGIN IMMEDIATE`` transaction and the version is re-read under
    that write lock, so concurrent openers apply each step once.

    Raises:
        RuntimeError: If the database was written by a newer llmdebug with a schema this
            version doesn't know.
    """
    version = _schema_version(conn)
    if version == STATE_DB_SCHEMA_VERSION:
        return
    conn.execute("BEGIN IMMEDIATE")
    try:
        _require_known_schema(conn)
        version = _schema_version(conn)
        for migrate in _SCHEMA_MIGRATIONS[version:]:
            migrate(conn)
        conn.execute(f"PRAGMA user_version = {STATE_DB_SCHEMA_VERSION:d}")
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")


@contextmanager
def open_state_db(out_dir: str | Path) -> Iterator[sqlite3.Connection]:
    """Open the state DB connection with schema initialization."""
    path = state_db_path(out_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    created = not path.exists()
    conn = sqlite3.connect(path, timeout=30.0)
    try:
        _configure_connection(conn, db_path=path, created=created)
        _ensure_schema(conn)
        yield conn
        conn.commit()
    finally:
        conn.close()


@contextmanager
def open_state_db_readonly(out_dir: str | Path) -> Iterator[sqlite3.Connection]:
    """Open an existing state DB for reading only.

    Read helpers must not create or change files (RCA-02, decision D11), and
    ``open_state_db`` does both even when the database is already there: the WAL switch
    and ``PRAGMA user_version`` write to the database and its ``-wal``/``-shm`` sidecars,
    which also makes reading a legacy (unversioned) database in a non-writable directory
    fail instead of returning the records it holds.
    """
    # Absolute: `as_uri()` rejects a relative path, and ``out_dir`` is usually relative
    # (the CLI default is ".llmdebug").
    path = state_db_path(out_dir).resolve()
    conn = sqlite3.connect(f"{path.as_uri()}?mode=ro", uri=True, timeout=30.0)
    try:
        busy_timeout = read_env_int("LLMDEBUG_DB_BUSY_TIMEOUT_MS", _BUSY_TIMEOUT_MS, minimum=0)
        conn.execute(f"PRAGMA busy_timeout={busy_timeout}")
        yield conn
    finally:
        conn.close()


def _read_rows(
    out_dir: str | Path,
    statements: tuple[tuple[str, tuple[Any, ...]], ...],
) -> tuple[list[Any], ...]:
    """Run read-only statements on one connection, with empty results for no database.

    A database that exists but predates a table reports the same "nothing stored yet"
    as a missing one; running migrations to create the table would be a write.
    """
    empty = tuple([] for _ in statements)
    if not state_db_exists(out_dir):
        return empty
    try:
        with open_state_db_readonly(out_dir) as conn:
            _require_known_schema(conn)
            return tuple(conn.execute(sql, params).fetchall() for sql, params in statements)
    except sqlite3.OperationalError as exc:
        if "no such table" not in str(exc).lower():
            raise
        return empty


def count_rca_records(out_dir: str | Path) -> int:
    """Return the number of persisted RCA attempt records for ``out_dir``."""
    (rows,) = _read_rows(out_dir, (("SELECT COUNT(*) FROM rca_attempts", ()),))
    return int(rows[0][0]) if rows else 0


def stored_rca_attempt_ids(out_dir: str | Path, attempt_ids: list[str]) -> set[str]:
    """Which of *attempt_ids* already have a stored row.

    The migration lets stored rows win over pending JSONL copies, so merged reads need
    to know about every conflicting id, not just the ones on the page they fetched.
    """
    wanted = [value for value in dict.fromkeys(attempt_ids) if value]
    if not wanted:
        return set()
    found: set[str] = set()
    # Chunked to stay under SQLite's bound-parameter limit.
    for start in range(0, len(wanted), _ID_QUERY_CHUNK):
        chunk = wanted[start : start + _ID_QUERY_CHUNK]
        placeholders = ",".join("?" * len(chunk))
        (rows,) = _read_rows(
            out_dir,
            (
                (
                    f"SELECT attempt_id FROM rca_attempts WHERE attempt_id IN ({placeholders})",
                    tuple(chunk),
                ),
            ),
        )
        found.update(str(row[0]) for row in rows)
    return found


def _require_text_field(record: Mapping[str, Any], key: str) -> str:
    raw_value = record.get(key)
    if raw_value is None:
        raise ValueError(f"RCA record missing required field: {key}")
    value = str(raw_value).strip()
    if not value:
        raise ValueError(f"RCA record has empty required field: {key}")
    return value


def _rca_row(record: Mapping[str, Any]) -> tuple[str, str, str, str, str, str, str]:
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


def append_rca_record(out_dir: str | Path, record: Mapping[str, Any], *, max_records: int) -> None:
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
    (rows,) = _read_rows(out_dir, ((sql, params),))
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
    (rows,) = _read_rows(out_dir, ((sql, tuple(params)),))
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
    if not paths or not state_db_exists(out_dir):
        return
    with open_state_db(out_dir) as conn:
        conn.executemany("DELETE FROM snapshots WHERE path = ?", [(path,) for path in paths])


def count_snapshot_records(out_dir: str | Path) -> int:
    """Return the number of snapshot index rows (0 without a database)."""
    (rows,) = _read_rows(out_dir, (("SELECT COUNT(*) FROM snapshots", ()),))
    return int(rows[0][0]) if rows else 0


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
    (rows,) = _read_rows(out_dir, ((sql, tuple(params)),))
    return _snapshot_rows(rows)


def _snapshot_rows(rows: list[Any]) -> list[dict[str, Any]]:
    return [
        {
            "path": str(path),
            "mtime": float(mtime),
            "name": str(name),
            "exc_type": str(exc_type),
            "exc_message": str(exc_message),
            "file": str(file_name),
            "line": str(line),
        }
        for path, mtime, name, exc_type, exc_message, file_name, line in rows
    ]


def snapshot_index_window(out_dir: str | Path, *, limit: int) -> tuple[int, list[dict[str, Any]]]:
    """Return the number of index rows and the newest *limit* rows, from one connection.

    Retention runs on every capture; one connection instead of two keeps the healthy-index
    check cheaper than the filesystem scan it replaces (CAP-15).
    """
    count_rows, rows = _read_rows(
        out_dir,
        (
            ("SELECT COUNT(*) FROM snapshots", ()),
            (
                "SELECT path, mtime, name, exc_type, exc_message, file, line FROM snapshots "
                "ORDER BY mtime DESC, path DESC LIMIT ?",
                (limit,),
            ),
        ),
    )
    return (int(count_rows[0][0]) if count_rows else 0), _snapshot_rows(rows)


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
