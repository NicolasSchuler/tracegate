"""Unit tests for llmdebug.state_db — SQLite state database helpers."""

from __future__ import annotations

import sqlite3
from contextlib import closing
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from llmdebug import _json as json
from llmdebug import state_db
from llmdebug.state_db import (
    _decode_payload_rows,
    _enable_wal_mode,
    _ensure_schema,
    _prune_rca_records,
    _rca_row,
    _require_text_field,
    append_rca_record,
    append_rca_records_bulk,
    count_rca_records,
    fetch_latest_rca_record,
    fetch_rca_history_records,
    list_snapshot_records,
    open_state_db,
    remove_snapshot_records,
    replace_snapshot_records,
    state_db_path,
    upsert_snapshot_record,
)  # fmt: skip

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_rca_record(**overrides: Any) -> dict[str, Any]:
    """Return a minimal valid RCA record with optional overrides."""
    base: dict[str, Any] = {
        "attempt_id": "a1",
        "session_id": "s1",
        "updated_at": "2024-01-01T00:00:00Z",
        "created_at": "2024-01-01T00:00:00Z",
        "failure_signature": "ValueError in foo",
        "state": "open",
    }
    base.update(overrides)
    return base


def _make_snapshot(**overrides: Any) -> dict[str, Any]:
    """Return a minimal valid snapshot record with optional overrides."""
    base: dict[str, Any] = {
        "path": "/snap/test.json",
        "mtime": 1_700_000_000.0,
        "name": "test_example",
        "exc_type": "ValueError",
        "exc_message": "bad value",
        "file": "example.py",
        "line": "42",
    }
    base.update(overrides)
    return base


@pytest.fixture(autouse=True)
def _clear_wal_warnings() -> None:
    """Each test may see the one-time "could not enable WAL" warning again."""
    state_db._WAL_WARNED_PATHS.clear()


# ---------------------------------------------------------------------------
# state_db_path  (line 24)
# ---------------------------------------------------------------------------


class TestStateDbPath:
    def test_returns_correct_path(self, tmp_path: Path) -> None:
        result = state_db_path(tmp_path)
        assert result == tmp_path / "llmdebug_state.sqlite3"

    def test_accepts_string(self, tmp_path: Path) -> None:
        result = state_db_path(str(tmp_path))
        assert result.name == "llmdebug_state.sqlite3"
        assert result.parent == tmp_path


# ---------------------------------------------------------------------------
# _enable_wal_mode  (line 29)
# ---------------------------------------------------------------------------


class TestEnableWalMode:
    def test_success_on_first_try(self) -> None:
        executed: list[str] = []

        class _Conn:
            def execute(self, sql: str) -> None:
                executed.append(sql)

        _enable_wal_mode(_Conn(), attempts=3, base_sleep_sec=0.0)
        assert executed == ["PRAGMA journal_mode=WAL"]

    def test_retries_on_locked_then_succeeds(self) -> None:
        call_count = 0

        class _FlakyConn:
            def execute(self, sql: str) -> None:
                nonlocal call_count
                call_count += 1
                if call_count < 3:
                    raise sqlite3.OperationalError("database is locked")

        sleeps: list[float] = []
        with patch.object(state_db.time, "sleep", side_effect=lambda v: sleeps.append(v)):
            _enable_wal_mode(_FlakyConn(), attempts=5, base_sleep_sec=0.01)

        assert call_count == 3
        assert len(sleeps) == 2

    def test_raises_when_all_retries_exhausted(self) -> None:
        class _AlwaysLocked:
            def execute(self, sql: str) -> None:
                raise sqlite3.OperationalError("database is locked")

        with (
            patch.object(state_db.time, "sleep", lambda _v: None),
            pytest.raises(sqlite3.OperationalError, match="locked"),
        ):
            _enable_wal_mode(_AlwaysLocked(), attempts=2, base_sleep_sec=0.0)

    def test_non_lock_error_raised_immediately(self) -> None:
        class _SyntaxError:
            def execute(self, sql: str) -> None:
                raise sqlite3.OperationalError("disk I/O error")

        with pytest.raises(sqlite3.OperationalError, match="disk I/O"):
            _enable_wal_mode(_SyntaxError(), attempts=5, base_sleep_sec=0.0)

    def test_no_sleep_when_base_sleep_is_zero(self) -> None:
        call_count = 0

        class _FlakyConn:
            def execute(self, sql: str) -> None:
                nonlocal call_count
                call_count += 1
                if call_count < 2:
                    raise sqlite3.OperationalError("database is locked")

        sleeps: list[float] = []
        with patch.object(state_db.time, "sleep", side_effect=lambda v: sleeps.append(v)):
            _enable_wal_mode(_FlakyConn(), attempts=3, base_sleep_sec=0.0)

        assert call_count == 2
        assert sleeps == []


# ---------------------------------------------------------------------------
# _ensure_wal_mode — tested via open_state_db and TestWalAndSchemaVersion
# _configure_connection  (line 57) — tested via open_state_db
# _ensure_schema  (line 69)
# ---------------------------------------------------------------------------


class TestEnsureSchema:
    def test_creates_tables_and_indexes(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        conn = sqlite3.connect(str(db))
        try:
            _ensure_schema(conn)
            # Verify rca_attempts table exists
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='rca_attempts'"
            ).fetchone()
            assert row is not None

            # Verify snapshots table exists
            row = conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='snapshots'"
            ).fetchone()
            assert row is not None

            # Verify indexes
            indexes = {
                r[0]
                for r in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='index'"
                ).fetchall()
            }
            assert "idx_rca_attempts_session_updated" in indexes
            assert "idx_rca_attempts_updated" in indexes
            assert "idx_snapshots_mtime" in indexes
        finally:
            conn.close()

    def test_idempotent(self, tmp_path: Path) -> None:
        db = tmp_path / "test.db"
        conn = sqlite3.connect(str(db))
        try:
            _ensure_schema(conn)
            _ensure_schema(conn)  # second call should not raise
            tables = conn.execute(
                "SELECT COUNT(*) FROM sqlite_master WHERE type='table'"
            ).fetchone()
            assert tables is not None
            assert tables[0] == 2  # rca_attempts + snapshots
        finally:
            conn.close()


# ---------------------------------------------------------------------------
# open_state_db  (lines 116-117)
# ---------------------------------------------------------------------------


class TestOpenStateDb:
    def test_creates_database_file(self, tmp_path: Path) -> None:
        with open_state_db(tmp_path) as conn:
            assert conn is not None
        assert state_db_path(tmp_path).exists()

    def test_creates_parent_dirs(self, tmp_path: Path) -> None:
        nested = tmp_path / "a" / "b" / "c"
        with open_state_db(nested) as conn:
            assert conn is not None
        assert state_db_path(nested).exists()

    def test_yields_usable_connection(self, tmp_path: Path) -> None:
        with open_state_db(tmp_path) as conn:
            row = conn.execute("SELECT COUNT(*) FROM rca_attempts").fetchone()
            assert row is not None
            assert row[0] == 0

    def test_commits_on_normal_exit(self, tmp_path: Path) -> None:
        rec = _make_rca_record()
        row = _rca_row(rec)
        with open_state_db(tmp_path) as conn:
            conn.execute(
                "INSERT INTO rca_attempts"
                "(attempt_id, session_id, updated_at, created_at,"
                " failure_signature, state, payload_json)"
                " VALUES (?, ?, ?, ?, ?, ?, ?)",
                row,
            )
        # Verify committed by opening a new connection
        with open_state_db(tmp_path) as conn:
            count = conn.execute("SELECT COUNT(*) FROM rca_attempts").fetchone()
            assert count is not None and count[0] == 1


# ---------------------------------------------------------------------------
# count_rca_records  (line 131)
# ---------------------------------------------------------------------------


class TestCountRcaRecords:
    def test_empty_db(self, tmp_path: Path) -> None:
        assert count_rca_records(tmp_path) == 0

    def test_after_inserts(self, tmp_path: Path) -> None:
        for i in range(3):
            append_rca_record(
                tmp_path,
                _make_rca_record(attempt_id=f"a{i}"),
                max_records=100,
            )
        assert count_rca_records(tmp_path) == 3


# ---------------------------------------------------------------------------
# _require_text_field  (line 137)
# ---------------------------------------------------------------------------


class TestRequireTextField:
    def test_valid_value(self) -> None:
        assert _require_text_field({"key": "hello"}, "key") == "hello"

    def test_strips_whitespace(self) -> None:
        assert _require_text_field({"key": "  val  "}, "key") == "val"

    def test_non_string_coerced(self) -> None:
        assert _require_text_field({"key": 42}, "key") == "42"

    def test_missing_key_raises(self) -> None:
        with pytest.raises(ValueError, match="missing required field"):
            _require_text_field({}, "key")

    def test_none_value_raises(self) -> None:
        with pytest.raises(ValueError, match="missing required field"):
            _require_text_field({"key": None}, "key")

    def test_empty_string_raises(self) -> None:
        with pytest.raises(ValueError, match="empty required field"):
            _require_text_field({"key": ""}, "key")

    def test_whitespace_only_raises(self) -> None:
        with pytest.raises(ValueError, match="empty required field"):
            _require_text_field({"key": "   "}, "key")


# ---------------------------------------------------------------------------
# _rca_row  (line 147)
# ---------------------------------------------------------------------------


class TestRcaRow:
    def test_returns_seven_element_tuple(self) -> None:
        rec = _make_rca_record()
        row = _rca_row(rec)
        assert len(row) == 7
        assert row[0] == "a1"  # attempt_id
        assert row[1] == "s1"  # session_id
        assert row[5] == "open"  # state
        # Last element is JSON payload
        payload = json.loads(row[6])
        assert payload["attempt_id"] == "a1"

    def test_raises_on_invalid_record(self) -> None:
        with pytest.raises(ValueError):
            _rca_row({"attempt_id": "a1"})  # missing other fields


# ---------------------------------------------------------------------------
# _prune_rca_records  (line 160)
# ---------------------------------------------------------------------------


class TestPruneRcaRecords:
    def test_prune_keeps_max(self, tmp_path: Path) -> None:
        for i in range(5):
            append_rca_record(
                tmp_path,
                _make_rca_record(
                    attempt_id=f"p{i}",
                    updated_at=f"2024-01-01T0{i}:00:00Z",
                ),
                max_records=1000,
            )
        with open_state_db(tmp_path) as conn:
            _prune_rca_records(conn, max_records=2)
        assert count_rca_records(tmp_path) == 2

    def test_prune_zero_is_noop(self, tmp_path: Path) -> None:
        append_rca_record(tmp_path, _make_rca_record(), max_records=1000)
        with open_state_db(tmp_path) as conn:
            _prune_rca_records(conn, max_records=0)
        assert count_rca_records(tmp_path) == 1

    def test_prune_negative_is_noop(self, tmp_path: Path) -> None:
        append_rca_record(tmp_path, _make_rca_record(), max_records=1000)
        with open_state_db(tmp_path) as conn:
            _prune_rca_records(conn, max_records=-1)
        assert count_rca_records(tmp_path) == 1


# ---------------------------------------------------------------------------
# append_rca_record  (line 177)
# ---------------------------------------------------------------------------


class TestAppendRcaRecord:
    def test_insert_and_count(self, tmp_path: Path) -> None:
        append_rca_record(tmp_path, _make_rca_record(), max_records=100)
        assert count_rca_records(tmp_path) == 1

    def test_upsert_replaces_same_id(self, tmp_path: Path) -> None:
        append_rca_record(
            tmp_path,
            _make_rca_record(state="open"),
            max_records=100,
        )
        append_rca_record(
            tmp_path,
            _make_rca_record(state="resolved"),
            max_records=100,
        )
        assert count_rca_records(tmp_path) == 1
        rec = fetch_latest_rca_record(tmp_path)
        assert rec is not None
        assert rec["state"] == "resolved"

    def test_prunes_on_insert(self, tmp_path: Path) -> None:
        for i in range(5):
            append_rca_record(
                tmp_path,
                _make_rca_record(
                    attempt_id=f"x{i}",
                    updated_at=f"2024-01-01T0{i}:00:00Z",
                ),
                max_records=3,
            )
        assert count_rca_records(tmp_path) == 3


# ---------------------------------------------------------------------------
# append_rca_records_bulk  (line 192)
# ---------------------------------------------------------------------------


class TestAppendRcaRecordsBulk:
    def test_empty_list_is_noop(self, tmp_path: Path) -> None:
        append_rca_records_bulk(tmp_path, [], max_records=100)
        # Should not even create the database
        assert count_rca_records(tmp_path) == 0

    def test_bulk_insert_multiple(self, tmp_path: Path) -> None:
        records = [
            _make_rca_record(
                attempt_id=f"b{i}",
                updated_at=f"2024-01-01T0{i}:00:00Z",
            )
            for i in range(4)
        ]
        append_rca_records_bulk(tmp_path, records, max_records=100)
        assert count_rca_records(tmp_path) == 4

    def test_bulk_insert_with_pruning(self, tmp_path: Path) -> None:
        records = [
            _make_rca_record(
                attempt_id=f"bp{i}",
                updated_at=f"2024-01-01T0{i}:00:00Z",
            )
            for i in range(6)
        ]
        append_rca_records_bulk(tmp_path, records, max_records=3)
        assert count_rca_records(tmp_path) == 3


# ---------------------------------------------------------------------------
# _decode_payload_rows  (line 214)
# ---------------------------------------------------------------------------


class TestDecodePayloadRows:
    def test_valid_json_dicts(self) -> None:
        rows = [
            (json.dumps({"k": "v1"}),),
            (json.dumps({"k": "v2"}),),
        ]
        result = _decode_payload_rows(rows)
        assert len(result) == 2
        assert result[0]["k"] == "v1"
        assert result[1]["k"] == "v2"

    def test_skips_corrupted_json(self) -> None:
        rows = [
            (json.dumps({"ok": True}),),
            ("NOT_VALID_JSON{{{",),
            (json.dumps({"also_ok": True}),),
        ]
        result = _decode_payload_rows(rows)
        assert len(result) == 2
        assert result[0]["ok"] is True
        assert result[1]["also_ok"] is True

    def test_skips_non_dict_json(self) -> None:
        """Valid JSON that is not a dict should be silently dropped."""
        rows = [
            ("[1, 2, 3]",),
            ('"just a string"',),
            ("42",),
            (json.dumps({"real": "record"}),),
        ]
        result = _decode_payload_rows(rows)
        assert len(result) == 1
        assert result[0]["real"] == "record"

    def test_empty_rows(self) -> None:
        assert _decode_payload_rows([]) == []


# ---------------------------------------------------------------------------
# fetch_latest_rca_record  (line 226)
# ---------------------------------------------------------------------------


class TestFetchLatestRcaRecord:
    def test_empty_db_returns_none(self, tmp_path: Path) -> None:
        assert fetch_latest_rca_record(tmp_path) is None

    def test_returns_most_recent(self, tmp_path: Path) -> None:
        for i in range(3):
            append_rca_record(
                tmp_path,
                _make_rca_record(
                    attempt_id=f"r{i}",
                    updated_at=f"2024-01-0{i + 1}T00:00:00Z",
                ),
                max_records=100,
            )
        rec = fetch_latest_rca_record(tmp_path)
        assert rec is not None
        assert rec["attempt_id"] == "r2"

    def test_with_session_id_filter(self, tmp_path: Path) -> None:
        append_rca_record(
            tmp_path,
            _make_rca_record(
                attempt_id="sa1",
                session_id="sess-A",
                updated_at="2024-01-01T00:00:00Z",
            ),
            max_records=100,
        )
        append_rca_record(
            tmp_path,
            _make_rca_record(
                attempt_id="sb1",
                session_id="sess-B",
                updated_at="2024-01-02T00:00:00Z",
            ),
            max_records=100,
        )
        rec = fetch_latest_rca_record(tmp_path, session_id="sess-A")
        assert rec is not None
        assert rec["attempt_id"] == "sa1"

    def test_without_session_id_returns_global_latest(self, tmp_path: Path) -> None:
        append_rca_record(
            tmp_path,
            _make_rca_record(
                attempt_id="ga",
                session_id="s1",
                updated_at="2024-01-01T00:00:00Z",
            ),
            max_records=100,
        )
        append_rca_record(
            tmp_path,
            _make_rca_record(
                attempt_id="gb",
                session_id="s2",
                updated_at="2024-01-02T00:00:00Z",
            ),
            max_records=100,
        )
        rec = fetch_latest_rca_record(tmp_path)
        assert rec is not None
        assert rec["attempt_id"] == "gb"

    def test_roundtrip_preserves_payload(self, tmp_path: Path) -> None:
        original = _make_rca_record(extra_field="extra_value")
        append_rca_record(tmp_path, original, max_records=100)
        rec = fetch_latest_rca_record(tmp_path)
        assert rec is not None
        assert rec["attempt_id"] == "a1"
        assert rec["extra_field"] == "extra_value"


# ---------------------------------------------------------------------------
# fetch_rca_history_records  (line 241)
# ---------------------------------------------------------------------------


class TestFetchRcaHistoryRecords:
    def test_empty_db(self, tmp_path: Path) -> None:
        assert fetch_rca_history_records(tmp_path) == []

    def test_returns_all_with_default_limit(self, tmp_path: Path) -> None:
        for i in range(5):
            append_rca_record(
                tmp_path,
                _make_rca_record(
                    attempt_id=f"h{i}",
                    updated_at=f"2024-01-0{i + 1}T00:00:00Z",
                ),
                max_records=100,
            )
        results = fetch_rca_history_records(tmp_path)
        assert len(results) == 5

    def test_with_session_id_filter(self, tmp_path: Path) -> None:
        pairs = [("sa", "hs-0"), ("sb", "hs-1"), ("sa", "hs-2")]
        for sid, aid in pairs:
            append_rca_record(
                tmp_path,
                _make_rca_record(attempt_id=aid, session_id=sid),
                max_records=100,
            )
        results = fetch_rca_history_records(tmp_path, session_id="sa")
        assert len(results) == 2
        assert all(r["session_id"] == "sa" for r in results)

    def test_respects_limit(self, tmp_path: Path) -> None:
        for i in range(5):
            append_rca_record(
                tmp_path,
                _make_rca_record(
                    attempt_id=f"hl{i}",
                    updated_at=f"2024-01-0{i + 1}T00:00:00Z",
                ),
                max_records=100,
            )
        results = fetch_rca_history_records(tmp_path, limit=2)
        assert len(results) == 2

    def test_limit_zero_returns_all(self, tmp_path: Path) -> None:
        for i in range(3):
            append_rca_record(
                tmp_path,
                _make_rca_record(
                    attempt_id=f"lz{i}",
                    updated_at=f"2024-01-0{i + 1}T00:00:00Z",
                ),
                max_records=100,
            )
        results = fetch_rca_history_records(tmp_path, limit=0)
        assert len(results) == 3

    def test_ordered_by_updated_at_desc(self, tmp_path: Path) -> None:
        for i in range(3):
            append_rca_record(
                tmp_path,
                _make_rca_record(
                    attempt_id=f"ord{i}",
                    updated_at=f"2024-01-0{i + 1}T00:00:00Z",
                ),
                max_records=100,
            )
        results = fetch_rca_history_records(tmp_path)
        ids = [r["attempt_id"] for r in results]
        assert ids == ["ord2", "ord1", "ord0"]


# ---------------------------------------------------------------------------
# upsert_snapshot_record  (line 261)
# ---------------------------------------------------------------------------


class TestUpsertSnapshotRecord:
    def test_insert_new(self, tmp_path: Path) -> None:
        upsert_snapshot_record(tmp_path, _make_snapshot())
        results = list_snapshot_records(tmp_path)
        assert len(results) == 1
        assert results[0]["path"] == "/snap/test.json"

    def test_update_existing(self, tmp_path: Path) -> None:
        upsert_snapshot_record(tmp_path, _make_snapshot(name="orig"))
        upsert_snapshot_record(tmp_path, _make_snapshot(name="updated", mtime=2.0))
        results = list_snapshot_records(tmp_path)
        assert len(results) == 1
        assert results[0]["name"] == "updated"

    def test_default_values(self, tmp_path: Path) -> None:
        """Fields missing from the record use safe defaults."""
        upsert_snapshot_record(tmp_path, {"path": "/p"})
        results = list_snapshot_records(tmp_path)
        assert len(results) == 1
        assert results[0]["mtime"] == 0.0
        assert results[0]["name"] == "?"


# ---------------------------------------------------------------------------
# remove_snapshot_records  (line 280)
# ---------------------------------------------------------------------------


class TestRemoveSnapshotRecords:
    def test_removes_by_path(self, tmp_path: Path) -> None:
        for i in range(3):
            upsert_snapshot_record(tmp_path, _make_snapshot(path=f"/snap/{i}.json"))
        remove_snapshot_records(tmp_path, ["/snap/0.json", "/snap/1.json"])
        results = list_snapshot_records(tmp_path)
        assert len(results) == 1
        assert results[0]["path"] == "/snap/2.json"

    def test_empty_list_is_noop(self, tmp_path: Path) -> None:
        upsert_snapshot_record(tmp_path, _make_snapshot())
        remove_snapshot_records(tmp_path, [])
        assert len(list_snapshot_records(tmp_path)) == 1

    def test_nonexistent_path_no_error(self, tmp_path: Path) -> None:
        upsert_snapshot_record(tmp_path, _make_snapshot())
        remove_snapshot_records(tmp_path, ["/does/not/exist"])
        assert len(list_snapshot_records(tmp_path)) == 1


# ---------------------------------------------------------------------------
# list_snapshot_records  (line 287)
# ---------------------------------------------------------------------------


class TestListSnapshotRecords:
    def test_empty_db(self, tmp_path: Path) -> None:
        assert list_snapshot_records(tmp_path) == []

    def test_search_by_name(self, tmp_path: Path) -> None:
        upsert_snapshot_record(tmp_path, _make_snapshot(path="/a", name="test_alpha"))
        upsert_snapshot_record(tmp_path, _make_snapshot(path="/b", name="test_beta"))
        results = list_snapshot_records(tmp_path, search="alpha")
        assert len(results) == 1
        assert results[0]["name"] == "test_alpha"

    def test_search_by_exc_type(self, tmp_path: Path) -> None:
        upsert_snapshot_record(
            tmp_path,
            _make_snapshot(path="/a", exc_type="ValueError"),
        )
        upsert_snapshot_record(
            tmp_path,
            _make_snapshot(path="/b", exc_type="TypeError"),
        )
        results = list_snapshot_records(tmp_path, search="TypeError")
        assert len(results) == 1
        assert results[0]["exc_type"] == "TypeError"

    def test_search_by_exc_message(self, tmp_path: Path) -> None:
        upsert_snapshot_record(
            tmp_path,
            _make_snapshot(path="/a", exc_message="division by zero"),
        )
        upsert_snapshot_record(
            tmp_path,
            _make_snapshot(path="/b", exc_message="key error"),
        )
        results = list_snapshot_records(tmp_path, search="division")
        assert len(results) == 1

    def test_search_by_file(self, tmp_path: Path) -> None:
        upsert_snapshot_record(
            tmp_path,
            _make_snapshot(path="/a", file="src/core.py"),
        )
        upsert_snapshot_record(
            tmp_path,
            _make_snapshot(path="/b", file="tests/test_x.py"),
        )
        results = list_snapshot_records(tmp_path, search="core")
        assert len(results) == 1
        assert results[0]["file"] == "src/core.py"

    def test_search_case_insensitive(self, tmp_path: Path) -> None:
        upsert_snapshot_record(
            tmp_path,
            _make_snapshot(path="/a", name="MyTestCase"),
        )
        results = list_snapshot_records(tmp_path, search="mytestcase")
        assert len(results) == 1

    def test_limit(self, tmp_path: Path) -> None:
        for i in range(5):
            upsert_snapshot_record(
                tmp_path,
                _make_snapshot(path=f"/s/{i}", mtime=float(i)),
            )
        results = list_snapshot_records(tmp_path, limit=3)
        assert len(results) == 3

    def test_limit_none_returns_all(self, tmp_path: Path) -> None:
        for i in range(4):
            upsert_snapshot_record(
                tmp_path,
                _make_snapshot(path=f"/s/{i}", mtime=float(i)),
            )
        results = list_snapshot_records(tmp_path, limit=None)
        assert len(results) == 4

    def test_ordered_by_mtime_desc(self, tmp_path: Path) -> None:
        for i in range(3):
            upsert_snapshot_record(
                tmp_path,
                _make_snapshot(path=f"/s/{i}", mtime=float(i)),
            )
        results = list_snapshot_records(tmp_path)
        mtimes = [r["mtime"] for r in results]
        assert mtimes == sorted(mtimes, reverse=True)

    def test_search_with_limit_combined(self, tmp_path: Path) -> None:
        for i in range(5):
            upsert_snapshot_record(
                tmp_path,
                _make_snapshot(
                    path=f"/s/{i}",
                    mtime=float(i),
                    name=f"match_{i}",
                ),
            )
        results = list_snapshot_records(tmp_path, search="match", limit=2)
        assert len(results) == 2

    def test_empty_search_string_returns_all(self, tmp_path: Path) -> None:
        for i in range(3):
            upsert_snapshot_record(
                tmp_path,
                _make_snapshot(path=f"/s/{i}", mtime=float(i)),
            )
        results = list_snapshot_records(tmp_path, search="")
        assert len(results) == 3


# ---------------------------------------------------------------------------
# replace_snapshot_records  (line 331)
# ---------------------------------------------------------------------------


class TestReplaceSnapshotRecords:
    def test_replaces_all(self, tmp_path: Path) -> None:
        for i in range(3):
            upsert_snapshot_record(
                tmp_path,
                _make_snapshot(path=f"/old/{i}", mtime=float(i)),
            )
        new = [_make_snapshot(path="/new/0", name="new_snap")]
        replace_snapshot_records(tmp_path, new)
        results = list_snapshot_records(tmp_path)
        assert len(results) == 1
        assert results[0]["name"] == "new_snap"

    def test_empty_list_clears_all(self, tmp_path: Path) -> None:
        upsert_snapshot_record(tmp_path, _make_snapshot())
        replace_snapshot_records(tmp_path, [])
        assert list_snapshot_records(tmp_path) == []

    def test_bulk_replace_multiple(self, tmp_path: Path) -> None:
        new_records = [_make_snapshot(path=f"/bulk/{i}", mtime=float(i)) for i in range(5)]
        replace_snapshot_records(tmp_path, new_records)
        results = list_snapshot_records(tmp_path)
        assert len(results) == 5


# ---------------------------------------------------------------------------
# RCA-16: WAL mode is verified on every open; RCA-11: versioned schema
# ---------------------------------------------------------------------------


class TestWalAndSchemaVersion:
    def test_recreated_database_is_switched_to_wal_again(self, tmp_path: Path) -> None:
        """RCA-16: a per-path "WAL done" cache survived deleting the database."""
        with open_state_db(tmp_path):
            pass
        for db_file in tmp_path.glob("llmdebug_state.sqlite3*"):
            db_file.unlink()

        with open_state_db(tmp_path) as conn:
            assert conn.execute("PRAGMA journal_mode").fetchone()[0] == "wal"

    def test_failed_wal_switch_warns_once(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setattr(state_db, "_enable_wal_mode", lambda _conn, **_kwargs: None)

        for _ in range(2):
            with open_state_db(tmp_path):
                pass

        assert capsys.readouterr().err.count("could not enable WAL mode") == 1

    def test_new_database_records_the_schema_version(self, tmp_path: Path) -> None:
        with open_state_db(tmp_path) as conn:
            version = conn.execute("PRAGMA user_version").fetchone()[0]

        assert version == state_db.STATE_DB_SCHEMA_VERSION == 1

    def test_unversioned_database_from_an_earlier_release_is_upgraded(self, tmp_path: Path) -> None:
        """RCA-11: databases created before versioning have user_version 0 and keep their rows."""
        db = state_db_path(tmp_path)
        conn = sqlite3.connect(db)
        state_db._create_schema_v1(conn)
        conn.execute(
            "INSERT INTO snapshots VALUES (?, ?, ?, ?, ?, ?, ?)",
            ("/s.json", 1.0, "n", "E", "m", "f.py", "1"),
        )
        conn.commit()
        assert conn.execute("PRAGMA user_version").fetchone()[0] == 0
        conn.close()

        rows = list_snapshot_records(tmp_path)

        assert [row["path"] for row in rows] == ["/s.json"]
        with closing(sqlite3.connect(db)) as check:
            assert check.execute("PRAGMA user_version").fetchone()[0] == 1

    def test_newer_schema_is_refused(self, tmp_path: Path) -> None:
        with open_state_db(tmp_path):
            pass
        with closing(sqlite3.connect(state_db_path(tmp_path))) as conn:
            conn.execute("PRAGMA user_version = 99")

        with pytest.raises(RuntimeError, match="schema version 99 is newer"):
            count_rca_records(tmp_path)


def test_count_snapshot_records_without_a_database_creates_nothing(tmp_path: Path) -> None:
    from llmdebug.state_db import count_snapshot_records

    assert count_snapshot_records(tmp_path / "missing") == 0
    assert not (tmp_path / "missing").exists()
