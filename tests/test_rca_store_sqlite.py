"""SQLite backend tests for RCA state persistence."""

from __future__ import annotations

import concurrent.futures
import sqlite3
from contextlib import closing
from pathlib import Path

import pytest

from llmdebug import _json as json
from llmdebug import state_db
from llmdebug.rca_backend import ensure_sqlite_migration
from llmdebug.rca_state import advance_rca_attempt, get_rca_history, get_rca_status
from llmdebug.rca_types import RCAAttemptRequest


def test_sqlite_backend_persists_and_reads(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LLMDEBUG_RCA_BACKEND", "sqlite")
    out_dir = str(tmp_path)

    first = advance_rca_attempt(
        out_dir=out_dir,
        request=RCAAttemptRequest(
            session_id="sess-1",
            action="bind_evidence",
            failure_signature="sig-1",
            evidence_refs=("r1",),
        ),
    )
    second = advance_rca_attempt(
        out_dir=out_dir,
        request=RCAAttemptRequest(
            session_id="sess-1",
            action="hypothesize",
            failure_signature="sig-1",
            evidence_refs=("r1",),
            hypothesis="h1",
            experiment_plan="p1",
        ),
    )

    status = get_rca_status(out_dir=out_dir, session_id="sess-1")
    assert status is not None
    assert status["attempt_id"] == second["attempt_id"]

    history = get_rca_history(out_dir=out_dir, session_id="sess-1", limit=10)
    assert len(history) == 2
    assert history[0]["attempt_id"] == second["attempt_id"]
    assert history[1]["attempt_id"] == first["attempt_id"]

    db_path = Path(out_dir) / "llmdebug_state.sqlite3"
    assert db_path.exists()

    with closing(sqlite3.connect(db_path)) as conn:
        row = conn.execute("SELECT COUNT(*) FROM rca_attempts").fetchone()
    assert row is not None
    assert int(row[0]) == 2


def test_sqlite_backend_migrates_jsonl_and_keeps_backup(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LLMDEBUG_RCA_BACKEND", "sqlite")
    out_dir = str(tmp_path)
    jsonl_path = Path(out_dir) / "rca_attempts.jsonl"

    records = [
        {
            "attempt_id": "a1",
            "session_id": "sess-a",
            "updated_at": "2026-02-16T10:00:00Z",
            "created_at": "2026-02-16T10:00:00Z",
            "failure_signature": "sig-a",
            "state": "captured",
        },
        {
            "attempt_id": "a2",
            "session_id": "sess-a",
            "updated_at": "2026-02-16T10:01:00Z",
            "created_at": "2026-02-16T10:00:00Z",
            "failure_signature": "sig-a",
            "state": "evidence_bound",
        },
    ]
    jsonl_path.write_text(
        "\n".join([json.dumps(records[0]), "{bad-json", json.dumps(records[1])]) + "\n",
        encoding="utf-8",
    )

    ensure_sqlite_migration(out_dir)  # writes migrate; reads only merge in memory (RCA-02)
    history = get_rca_history(out_dir=out_dir, session_id="sess-a", limit=10)
    assert len(history) == 2
    assert history[0]["attempt_id"] == "a2"

    backup_path = Path(f"{jsonl_path}.bak")
    assert backup_path.exists()
    assert not jsonl_path.exists()

    db_path = Path(out_dir) / "llmdebug_state.sqlite3"
    with closing(sqlite3.connect(db_path)) as conn:
        row = conn.execute("SELECT COUNT(*) FROM rca_attempts").fetchone()
    assert row is not None
    assert int(row[0]) == 2


def test_sqlite_backend_concurrent_appends_are_queryable(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LLMDEBUG_RCA_BACKEND", "sqlite")
    out_dir = str(tmp_path)
    writes = 64

    def _append(i: int) -> None:
        advance_rca_attempt(
            out_dir=out_dir,
            request=RCAAttemptRequest(
                session_id="sess-concurrent",
                action="bind_evidence",
                failure_signature=f"sig-{i % 3}",
                evidence_refs=(f"ref-{i}",),
            ),
        )

    with concurrent.futures.ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(_append, range(writes)))

    history = get_rca_history(out_dir=out_dir, session_id="sess-concurrent", limit=200)
    assert len(history) == writes


def test_enable_wal_mode_retries_transient_database_locked(monkeypatch) -> None:
    class _FlakyConn:
        def __init__(self) -> None:
            self.calls = 0

        def execute(self, sql: str) -> None:
            assert sql == "PRAGMA journal_mode=WAL"
            self.calls += 1
            if self.calls < 3:
                raise sqlite3.OperationalError("database is locked")

    sleeps: list[float] = []
    monkeypatch.setattr(state_db.time, "sleep", lambda value: sleeps.append(float(value)))

    conn = _FlakyConn()
    state_db._enable_wal_mode(conn, attempts=5, base_sleep_sec=0.001)

    assert conn.calls == 3
    assert sleeps == [0.001, 0.002]


def test_enable_wal_mode_does_not_retry_non_lock_errors(monkeypatch) -> None:
    class _BadConn:
        def execute(self, sql: str) -> None:
            assert sql == "PRAGMA journal_mode=WAL"
            raise sqlite3.OperationalError('near "wal": syntax error')

    slept = {"value": False}
    monkeypatch.setattr(state_db.time, "sleep", lambda _value: slept.update(value=True))

    with pytest.raises(sqlite3.OperationalError, match="syntax error"):
        state_db._enable_wal_mode(_BadConn(), attempts=3, base_sleep_sec=0.001)

    assert slept["value"] is False


def test_configure_connection_invalid_busy_timeout_falls_back(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class _Conn:
        def __init__(self) -> None:
            self.commands: list[str] = []

        def execute(self, sql: str) -> None:
            self.commands.append(sql)

    monkeypatch.setenv("LLMDEBUG_DB_BUSY_TIMEOUT_MS", "nope")
    monkeypatch.setattr(state_db, "_ensure_wal_mode", lambda _conn, **_kwargs: None)

    conn = _Conn()
    state_db._configure_connection(conn, db_path=tmp_path / "llmdebug_state.sqlite3")

    assert conn.commands[0] == f"PRAGMA busy_timeout={state_db._BUSY_TIMEOUT_MS}"
    assert "PRAGMA synchronous=NORMAL" in conn.commands
    assert "PRAGMA temp_store=MEMORY" in conn.commands


# ---------------------------------------------------------------------------
# Snapshot CRUD tests
# ---------------------------------------------------------------------------


def test_upsert_and_list_snapshot_records(tmp_path: Path) -> None:
    record = {
        "path": "/tmp/snap1.json",
        "mtime": 1_700_000_000.0,
        "name": "test_foo",
        "exc_type": "ValueError",
        "exc_message": "bad value",
        "file": "test_foo.py",
        "line": "42",
    }
    state_db.upsert_snapshot_record(str(tmp_path), record)
    results = state_db.list_snapshot_records(str(tmp_path))
    assert len(results) == 1
    assert results[0]["path"] == "/tmp/snap1.json"
    assert results[0]["exc_type"] == "ValueError"
    assert results[0]["name"] == "test_foo"


def test_list_snapshot_records_empty(tmp_path: Path) -> None:
    results = state_db.list_snapshot_records(str(tmp_path))
    assert results == []


def test_upsert_snapshot_record_updates_existing(tmp_path: Path) -> None:
    record = {
        "path": "/tmp/snap.json",
        "mtime": 1.0,
        "name": "original",
        "exc_type": "TypeError",
        "exc_message": "msg",
        "file": "f.py",
        "line": "1",
    }
    state_db.upsert_snapshot_record(str(tmp_path), record)

    updated = dict(record)
    updated["name"] = "updated"
    updated["mtime"] = 2.0
    state_db.upsert_snapshot_record(str(tmp_path), updated)

    results = state_db.list_snapshot_records(str(tmp_path))
    assert len(results) == 1
    assert results[0]["name"] == "updated"
    assert results[0]["mtime"] == 2.0


def test_remove_snapshot_records(tmp_path: Path) -> None:
    for i in range(3):
        state_db.upsert_snapshot_record(
            str(tmp_path),
            {
                "path": f"/tmp/snap{i}.json",
                "mtime": float(i),
                "name": f"test_{i}",
                "exc_type": "E",
                "exc_message": "",
                "file": "f.py",
                "line": str(i),
            },
        )
    state_db.remove_snapshot_records(str(tmp_path), ["/tmp/snap0.json", "/tmp/snap1.json"])
    results = state_db.list_snapshot_records(str(tmp_path))
    assert len(results) == 1
    assert results[0]["path"] == "/tmp/snap2.json"


def test_remove_snapshot_records_empty_list_is_noop(tmp_path: Path) -> None:
    state_db.upsert_snapshot_record(
        str(tmp_path),
        {
            "path": "/tmp/snap.json",
            "mtime": 1.0,
            "name": "t",
            "exc_type": "E",
            "exc_message": "",
            "file": "f.py",
            "line": "1",
        },
    )
    state_db.remove_snapshot_records(str(tmp_path), [])
    results = state_db.list_snapshot_records(str(tmp_path))
    assert len(results) == 1


def test_replace_snapshot_records(tmp_path: Path) -> None:
    for i in range(3):
        state_db.upsert_snapshot_record(
            str(tmp_path),
            {
                "path": f"/tmp/old{i}.json",
                "mtime": float(i),
                "name": "old",
                "exc_type": "E",
                "exc_message": "",
                "file": "f.py",
                "line": str(i),
            },
        )
    new_records = [
        {
            "path": "/tmp/new.json",
            "mtime": 99.0,
            "name": "new_test",
            "exc_type": "RuntimeError",
            "exc_message": "oops",
            "file": "new.py",
            "line": "7",
        }
    ]
    state_db.replace_snapshot_records(str(tmp_path), new_records)
    results = state_db.list_snapshot_records(str(tmp_path))
    assert len(results) == 1
    assert results[0]["name"] == "new_test"
    assert results[0]["exc_type"] == "RuntimeError"


def test_replace_snapshot_records_with_empty_clears_all(tmp_path: Path) -> None:
    state_db.upsert_snapshot_record(
        str(tmp_path),
        {
            "path": "/tmp/snap.json",
            "mtime": 1.0,
            "name": "t",
            "exc_type": "E",
            "exc_message": "",
            "file": "f.py",
            "line": "1",
        },
    )
    state_db.replace_snapshot_records(str(tmp_path), [])
    results = state_db.list_snapshot_records(str(tmp_path))
    assert results == []


def test_list_snapshot_records_search_by_name(tmp_path: Path) -> None:
    for name in ["test_alpha", "test_beta", "check_gamma"]:
        state_db.upsert_snapshot_record(
            str(tmp_path),
            {
                "path": f"/tmp/{name}.json",
                "mtime": 1.0,
                "name": name,
                "exc_type": "E",
                "exc_message": "",
                "file": "f.py",
                "line": "1",
            },
        )
    results = state_db.list_snapshot_records(str(tmp_path), search="alpha")
    assert len(results) == 1
    assert results[0]["name"] == "test_alpha"


def test_list_snapshot_records_search_by_exc_type(tmp_path: Path) -> None:
    state_db.upsert_snapshot_record(
        str(tmp_path),
        {
            "path": "/tmp/s1.json",
            "mtime": 1.0,
            "name": "t1",
            "exc_type": "ValueError",
            "exc_message": "",
            "file": "f.py",
            "line": "1",
        },
    )
    state_db.upsert_snapshot_record(
        str(tmp_path),
        {
            "path": "/tmp/s2.json",
            "mtime": 2.0,
            "name": "t2",
            "exc_type": "TypeError",
            "exc_message": "",
            "file": "f.py",
            "line": "2",
        },
    )
    results = state_db.list_snapshot_records(str(tmp_path), search="TypeError")
    assert len(results) == 1
    assert results[0]["exc_type"] == "TypeError"


def test_list_snapshot_records_limit(tmp_path: Path) -> None:
    for i in range(5):
        state_db.upsert_snapshot_record(
            str(tmp_path),
            {
                "path": f"/tmp/snap{i}.json",
                "mtime": float(i),
                "name": f"t{i}",
                "exc_type": "E",
                "exc_message": "",
                "file": "f.py",
                "line": str(i),
            },
        )
    results = state_db.list_snapshot_records(str(tmp_path), limit=3)
    assert len(results) == 3


def test_decode_payload_rows_skips_malformed_json(tmp_path: Path) -> None:
    # Insert a record then corrupt the payload_json manually
    import sqlite3 as _sqlite3

    record = {
        "attempt_id": "bad-1",
        "session_id": "s",
        "updated_at": "2026-01-01T00:00:00Z",
        "created_at": "2026-01-01T00:00:00Z",
        "failure_signature": "sig",
        "state": "captured",
    }
    state_db.append_rca_record(str(tmp_path), record, max_records=100)

    db_path = state_db.state_db_path(tmp_path)
    with _sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE rca_attempts SET payload_json = 'NOT_JSON' WHERE attempt_id = 'bad-1'")
        conn.commit()

    result = state_db.fetch_latest_rca_record(str(tmp_path))
    assert result is None


def test_reopening_a_wal_database_does_not_switch_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The journal mode is read on open; an already-WAL database isn't switched again."""
    with state_db.open_state_db(str(tmp_path)):
        pass
    calls: list[object] = []
    monkeypatch.setattr(state_db, "_enable_wal_mode", lambda conn, **_kw: calls.append(conn))

    with state_db.open_state_db(str(tmp_path)):
        pass

    assert calls == []


def test_prune_rca_records_respects_max(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("LLMDEBUG_RCA_BACKEND", "sqlite")
    from llmdebug.rca_state import advance_rca_attempt
    from llmdebug.rca_types import RCAAttemptRequest

    for i in range(10):
        advance_rca_attempt(
            out_dir=str(tmp_path),
            request=RCAAttemptRequest(
                session_id="sess-prune",
                action="bind_evidence",
                failure_signature=f"sig-{i}",
                evidence_refs=(f"r{i}",),
            ),
        )
    # Prune to 3 via a direct append with max_records=3
    record = {
        "attempt_id": "extra",
        "session_id": "sess-prune",
        "updated_at": "2099-01-01T00:00:00Z",
        "created_at": "2099-01-01T00:00:00Z",
        "failure_signature": "sig-extra",
        "state": "captured",
    }
    state_db.append_rca_record(str(tmp_path), record, max_records=3)
    count = state_db.count_rca_records(str(tmp_path))
    assert count == 3


# ---------------------------------------------------------------------------
# Additional branch-coverage tests
# ---------------------------------------------------------------------------


def test_enable_wal_mode_no_sleep_when_base_sleep_zero(monkeypatch: pytest.MonkeyPatch) -> None:
    """base_sleep_sec=0 means no sleep even on locked errors."""

    class _FlakyConn:
        def __init__(self) -> None:
            self.calls = 0

        def execute(self, sql: str) -> None:
            self.calls += 1
            if self.calls < 2:
                raise sqlite3.OperationalError("database is locked")

    sleeps: list[float] = []
    monkeypatch.setattr(state_db.time, "sleep", lambda v: sleeps.append(v))

    conn = _FlakyConn()
    state_db._enable_wal_mode(conn, attempts=3, base_sleep_sec=0.0)
    assert conn.calls == 2
    assert sleeps == []


def test_prune_rca_records_noop_when_max_zero(tmp_path: Path) -> None:
    """_prune_rca_records with max_records=0 is a no-op."""
    record = {
        "attempt_id": "p1",
        "session_id": "s",
        "updated_at": "2026-01-01T00:00:00Z",
        "created_at": "2026-01-01T00:00:00Z",
        "failure_signature": "sig",
        "state": "captured",
    }
    state_db.append_rca_record(str(tmp_path), record, max_records=0)
    # max_records=0 → prune is skipped, record survives
    assert state_db.count_rca_records(str(tmp_path)) == 1


def test_require_text_field_missing_raises() -> None:
    with pytest.raises(ValueError, match="missing required field"):
        state_db._require_text_field({}, "attempt_id")


def test_require_text_field_empty_raises() -> None:
    with pytest.raises(ValueError, match="empty required field"):
        state_db._require_text_field({"attempt_id": "  "}, "attempt_id")


def test_append_rca_records_bulk_noop_for_empty(tmp_path: Path) -> None:
    """append_rca_records_bulk with empty list does nothing."""
    state_db.append_rca_records_bulk(str(tmp_path), [], max_records=100)
    assert state_db.count_rca_records(str(tmp_path)) == 0


def test_append_rca_records_bulk_multiple(tmp_path: Path) -> None:
    records = [
        {
            "attempt_id": f"bulk-{i}",
            "session_id": "s",
            "updated_at": f"2026-01-01T0{i}:00:00Z",
            "created_at": "2026-01-01T00:00:00Z",
            "failure_signature": "sig",
            "state": "captured",
        }
        for i in range(3)
    ]
    state_db.append_rca_records_bulk(str(tmp_path), records, max_records=100)
    assert state_db.count_rca_records(str(tmp_path)) == 3


def test_fetch_latest_rca_record_with_session_id(tmp_path: Path) -> None:
    for i, sid in enumerate(["sess-a", "sess-b", "sess-a"]):
        state_db.append_rca_record(
            str(tmp_path),
            {
                "attempt_id": f"id-{i}",
                "session_id": sid,
                "updated_at": f"2026-01-01T0{i}:00:00Z",
                "created_at": "2026-01-01T00:00:00Z",
                "failure_signature": "sig",
                "state": "captured",
            },
            max_records=100,
        )
    result = state_db.fetch_latest_rca_record(str(tmp_path), session_id="sess-a")
    assert result is not None
    assert result["session_id"] == "sess-a"
    assert result["attempt_id"] == "id-2"


def test_fetch_rca_history_records_with_session_id_and_limit(tmp_path: Path) -> None:
    for i in range(5):
        state_db.append_rca_record(
            str(tmp_path),
            {
                "attempt_id": f"h-{i}",
                "session_id": "sess-hist",
                "updated_at": f"2026-01-01T0{i}:00:00Z",
                "created_at": "2026-01-01T00:00:00Z",
                "failure_signature": "sig",
                "state": "captured",
            },
            max_records=100,
        )
    results = state_db.fetch_rca_history_records(str(tmp_path), session_id="sess-hist", limit=3)
    assert len(results) == 3
    assert all(r["session_id"] == "sess-hist" for r in results)


def test_fetch_rca_history_records_no_limit(tmp_path: Path) -> None:
    """limit=0 means no LIMIT clause — all records returned."""
    for i in range(4):
        state_db.append_rca_record(
            str(tmp_path),
            {
                "attempt_id": f"nl-{i}",
                "session_id": "sess-nl",
                "updated_at": f"2026-01-01T0{i}:00:00Z",
                "created_at": "2026-01-01T00:00:00Z",
                "failure_signature": "sig",
                "state": "captured",
            },
            max_records=100,
        )
    results = state_db.fetch_rca_history_records(str(tmp_path), limit=0)
    assert len(results) == 4


def test_list_snapshot_records_search_by_file(tmp_path: Path) -> None:
    state_db.upsert_snapshot_record(
        str(tmp_path),
        {
            "path": "/tmp/a.json",
            "mtime": 1.0,
            "name": "t",
            "exc_type": "E",
            "exc_message": "",
            "file": "src/mymodule.py",
            "line": "10",
        },
    )
    state_db.upsert_snapshot_record(
        str(tmp_path),
        {
            "path": "/tmp/b.json",
            "mtime": 2.0,
            "name": "t2",
            "exc_type": "E",
            "exc_message": "",
            "file": "tests/test_other.py",
            "line": "5",
        },
    )
    results = state_db.list_snapshot_records(str(tmp_path), search="mymodule")
    assert len(results) == 1
    assert results[0]["file"] == "src/mymodule.py"


def test_list_snapshot_records_no_limit_returns_all(tmp_path: Path) -> None:
    """limit=None returns all records."""
    for i in range(4):
        state_db.upsert_snapshot_record(
            str(tmp_path),
            {
                "path": f"/tmp/snap{i}.json",
                "mtime": float(i),
                "name": f"t{i}",
                "exc_type": "E",
                "exc_message": "",
                "file": "f.py",
                "line": str(i),
            },
        )
    results = state_db.list_snapshot_records(str(tmp_path), limit=None)
    assert len(results) == 4


def test_state_db_path_returns_correct_filename(tmp_path: Path) -> None:
    p = state_db.state_db_path(tmp_path)
    assert p == tmp_path / "llmdebug_state.sqlite3"
