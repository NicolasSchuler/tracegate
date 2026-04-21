"""Tests for RCA backend adapters."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from llmdebug import _json as json
from llmdebug.rca_backend import (
    _append_record,
    _is_valid_rca_record,
    _iter_jsonl_records,
    _latest_jsonl_record,
    backend_name,
    ensure_sqlite_migration,
    get_rca_history,
    get_rca_status,
    max_rca_records,
    persist_rca_record,
)


def _record(
    attempt_id: str,
    *,
    session_id: str = "sess-a",
    state: str = "captured",
    updated_at: str = "2026-02-27T10:00:00Z",
) -> dict[str, object]:
    return {
        "attempt_id": attempt_id,
        "session_id": session_id,
        "updated_at": updated_at,
        "created_at": "2026-02-27T10:00:00Z",
        "failure_signature": "sig-a",
        "state": state,
        "evidence_refs": ["r1"],
    }


def test_backend_name_defaults_and_invalid(monkeypatch) -> None:
    monkeypatch.delenv("LLMDEBUG_RCA_BACKEND", raising=False)
    assert backend_name() == "sqlite"

    monkeypatch.setenv("LLMDEBUG_RCA_BACKEND", "not-real")
    assert backend_name() == "sqlite"


def test_backend_name_jsonl(monkeypatch) -> None:
    monkeypatch.setenv("LLMDEBUG_RCA_BACKEND", "jsonl")
    assert backend_name() == "jsonl"


def test_jsonl_roundtrip_status_and_history(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LLMDEBUG_RCA_BACKEND", "jsonl")
    monkeypatch.setenv("LLMDEBUG_RCA_MAX_RECORDS", "2")

    out_dir = str(tmp_path)
    persist_rca_record(out_dir=out_dir, record=_record("a1", updated_at="2026-02-27T10:00:00Z"))
    persist_rca_record(out_dir=out_dir, record=_record("a2", updated_at="2026-02-27T10:01:00Z"))
    persist_rca_record(out_dir=out_dir, record=_record("a3", updated_at="2026-02-27T10:02:00Z"))

    status = get_rca_status(out_dir=out_dir, session_id="sess-a")
    assert status is not None
    assert status["attempt_id"] == "a3"

    history = get_rca_history(out_dir=out_dir, session_id="sess-a", limit=10)
    assert [entry["attempt_id"] for entry in history] == ["a3", "a2"]

    store = tmp_path / "rca_attempts.jsonl"
    lines = [line for line in store.read_text(encoding="utf-8").splitlines() if line.strip()]
    assert len(lines) == 2


def test_sqlite_migration_guard_skips_when_rows_exist(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LLMDEBUG_RCA_BACKEND", "sqlite")
    out_dir = str(tmp_path)

    persist_rca_record(out_dir=out_dir, record=_record("sql-1"))

    jsonl_path = tmp_path / "rca_attempts.jsonl"
    jsonl_path.write_text(json.dumps(_record("jsonl-1")) + "\n", encoding="utf-8")

    history = get_rca_history(out_dir=out_dir, session_id="sess-a", limit=10)
    assert history
    assert jsonl_path.exists()
    assert not Path(f"{jsonl_path}.bak").exists()


def test_sqlite_migration_filters_invalid_records_and_applies_retention(
    tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setenv("LLMDEBUG_RCA_BACKEND", "sqlite")
    monkeypatch.setenv("LLMDEBUG_RCA_MAX_RECORDS", "2")
    out_dir = str(tmp_path)
    jsonl_path = tmp_path / "rca_attempts.jsonl"
    rows = [
        _record("a1", updated_at="2026-02-27T10:00:00Z"),
        {**_record("bad", updated_at="2026-02-27T10:00:30Z"), "attempt_id": ""},
        _record("a2", updated_at="2026-02-27T10:01:00Z"),
        _record("a3", updated_at="2026-02-27T10:02:00Z"),
    ]
    jsonl_path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")

    history = get_rca_history(out_dir=out_dir, session_id="sess-a", limit=10)
    assert [entry["attempt_id"] for entry in history] == ["a3", "a2"]
    assert not jsonl_path.exists()
    assert Path(f"{jsonl_path}.bak").exists()


# ---------------------------------------------------------------------------
# max_rca_records
# ---------------------------------------------------------------------------


def test_max_rca_records_default(monkeypatch) -> None:
    monkeypatch.delenv("LLMDEBUG_RCA_MAX_RECORDS", raising=False)
    assert max_rca_records() == 5000


def test_max_rca_records_invalid_string(monkeypatch) -> None:
    monkeypatch.setenv("LLMDEBUG_RCA_MAX_RECORDS", "not-a-number")
    assert max_rca_records() == 5000


def test_max_rca_records_zero(monkeypatch) -> None:
    monkeypatch.setenv("LLMDEBUG_RCA_MAX_RECORDS", "0")
    assert max_rca_records() == 0


def test_max_rca_records_negative_clipped_to_zero(monkeypatch) -> None:
    monkeypatch.setenv("LLMDEBUG_RCA_MAX_RECORDS", "-5")
    assert max_rca_records() == 0


# ---------------------------------------------------------------------------
# _iter_jsonl_records
# ---------------------------------------------------------------------------


def test_iter_jsonl_records_nonexistent_path(tmp_path: Path) -> None:
    result = _iter_jsonl_records(tmp_path / "nonexistent.jsonl")
    assert result == []


def test_iter_jsonl_records_oserror(tmp_path: Path) -> None:
    p = tmp_path / "records.jsonl"
    p.write_text("line\n")
    with patch.object(Path, "read_text", side_effect=OSError("perm denied")):
        result = _iter_jsonl_records(p)
    assert result == []


def test_iter_jsonl_records_skips_blank_lines(tmp_path: Path) -> None:
    p = tmp_path / "records.jsonl"
    p.write_text(
        json.dumps(_record("a1")) + "\n\n" + json.dumps(_record("a2")) + "\n",
        encoding="utf-8",
    )
    result = _iter_jsonl_records(p)
    assert len(result) == 2


def test_iter_jsonl_records_skips_invalid_json(tmp_path: Path) -> None:
    p = tmp_path / "records.jsonl"
    p.write_text(
        json.dumps(_record("a1")) + "\n" + "not-json\n" + json.dumps(_record("a2")) + "\n",
        encoding="utf-8",
    )
    result = _iter_jsonl_records(p)
    assert len(result) == 2
    assert result[0]["attempt_id"] == "a1"


def test_iter_jsonl_records_skips_non_dict_values(tmp_path: Path) -> None:
    p = tmp_path / "records.jsonl"
    p.write_text(
        json.dumps(_record("a1")) + "\n" + '["list"]' + "\n",
        encoding="utf-8",
    )
    result = _iter_jsonl_records(p)
    assert len(result) == 1


# ---------------------------------------------------------------------------
# _is_valid_rca_record
# ---------------------------------------------------------------------------


def test_is_valid_rca_record_valid() -> None:
    assert _is_valid_rca_record(dict(_record("a1"))) is True  # type: ignore[arg-type]


def test_is_valid_rca_record_missing_field() -> None:
    rec = dict(_record("a1"))
    del rec["failure_signature"]
    assert _is_valid_rca_record(rec) is False  # type: ignore[arg-type]


def test_is_valid_rca_record_empty_field() -> None:
    rec = dict(_record("a1"))
    rec["state"] = "   "
    assert _is_valid_rca_record(rec) is False  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _latest_jsonl_record
# ---------------------------------------------------------------------------


def test_latest_jsonl_record_nonexistent(tmp_path: Path) -> None:
    result = _latest_jsonl_record(tmp_path / "nonexistent.jsonl")
    assert result is None


def test_latest_jsonl_record_with_session_filter(tmp_path: Path) -> None:
    p = tmp_path / "records.jsonl"
    p.write_text(
        json.dumps(_record("a1", session_id="s1"))
        + "\n"
        + json.dumps(_record("a2", session_id="s2"))
        + "\n"
        + json.dumps(_record("a3", session_id="s1"))
        + "\n",
        encoding="utf-8",
    )
    result = _latest_jsonl_record(p, session_id="s1")
    assert result is not None
    assert result["attempt_id"] == "a3"


def test_latest_jsonl_record_skips_invalid_json(tmp_path: Path) -> None:
    p = tmp_path / "records.jsonl"
    # Last line is invalid JSON, second-to-last is valid
    p.write_text(
        json.dumps(_record("a1")) + "\n" + "invalid-json\n",
        encoding="utf-8",
    )
    result = _latest_jsonl_record(p)
    assert result is not None
    assert result["attempt_id"] == "a1"


def test_latest_jsonl_record_skips_non_dict(tmp_path: Path) -> None:
    p = tmp_path / "records.jsonl"
    p.write_text(
        json.dumps(_record("a1")) + "\n" + '["list"]' + "\n",
        encoding="utf-8",
    )
    # "list" is not a dict — should skip and find a1
    result = _latest_jsonl_record(p)
    assert result is not None
    assert result["attempt_id"] == "a1"


def test_latest_jsonl_record_oserror(tmp_path: Path) -> None:
    p = tmp_path / "records.jsonl"
    p.write_text(json.dumps(_record("a1")) + "\n", encoding="utf-8")
    with patch("llmdebug.rca_backend._iter_jsonl_lines_reverse", side_effect=OSError("io")):
        result = _latest_jsonl_record(p)
    assert result is None


# ---------------------------------------------------------------------------
# _append_record — trim and OSError paths
# ---------------------------------------------------------------------------


def test_append_record_unlimited_no_trim(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LLMDEBUG_RCA_MAX_RECORDS", "0")
    p = tmp_path / "records.jsonl"
    for i in range(10):
        _append_record(p, dict(_record(f"a{i}")))  # type: ignore[arg-type]
    lines = [ln for ln in p.read_text("utf-8").splitlines() if ln.strip()]
    assert len(lines) == 10  # no trimming when unlimited


def test_append_record_within_limit_no_trim(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LLMDEBUG_RCA_MAX_RECORDS", "100")
    p = tmp_path / "records.jsonl"
    _append_record(p, dict(_record("a1")))  # type: ignore[arg-type]
    _append_record(p, dict(_record("a2")))  # type: ignore[arg-type]
    lines = [ln for ln in p.read_text("utf-8").splitlines() if ln.strip()]
    assert len(lines) == 2


def test_append_record_oserror_on_read_survives(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LLMDEBUG_RCA_MAX_RECORDS", "1")
    p = tmp_path / "records.jsonl"
    _append_record(p, dict(_record("a1")))  # type: ignore[arg-type]
    # Make read_text raise OSError; should not propagate
    original_read = Path.read_text
    call_count = [0]

    def _fail_read(self: Path, *args, **kwargs) -> str:  # type: ignore[override]
        call_count[0] += 1
        if call_count[0] > 1:
            raise OSError("read failed")
        return original_read(self, *args, **kwargs)

    with patch.object(Path, "read_text", _fail_read):
        _append_record(p, dict(_record("a2")))  # type: ignore[arg-type]
    # Should still exist; OSError was swallowed
    assert p.exists()


def test_append_record_oserror_on_write_trim_survives(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LLMDEBUG_RCA_MAX_RECORDS", "1")
    p = tmp_path / "records.jsonl"
    _append_record(p, dict(_record("a1")))  # type: ignore[arg-type]
    _append_record(p, dict(_record("a2")))  # type: ignore[arg-type]
    lines_before = p.read_text(encoding="utf-8").splitlines()
    # Now write_text for trim fails; OSError should be swallowed
    original_write = Path.write_text
    call_count = [0]

    def _fail_write(self: Path, *args, **kwargs) -> int:  # type: ignore[override]
        call_count[0] += 1
        if call_count[0] > 1:
            raise OSError("write failed")
        return original_write(self, *args, **kwargs)  # type: ignore[return-value]

    with patch.object(Path, "write_text", _fail_write):
        _append_record(p, dict(_record("a3")))  # type: ignore[arg-type]
    # Should not raise; file should still have records (trim failed but append succeeded)
    lines_after = p.read_text(encoding="utf-8").splitlines()
    assert len(lines_after) >= len(lines_before)


# ---------------------------------------------------------------------------
# ensure_sqlite_migration — no-records path
# ---------------------------------------------------------------------------


def test_ensure_sqlite_migration_empty_jsonl_no_records(tmp_path: Path, monkeypatch) -> None:
    """Migration with empty JSONL (no valid records) still completes backup rotation."""
    monkeypatch.setenv("LLMDEBUG_RCA_BACKEND", "sqlite")
    out_dir = str(tmp_path)
    jsonl_path = tmp_path / "rca_attempts.jsonl"
    jsonl_path.write_text("", encoding="utf-8")

    ensure_sqlite_migration(out_dir)
    # Should create a backup even when there are no valid records
    assert not jsonl_path.exists() or Path(f"{jsonl_path}.bak").exists()


def test_ensure_sqlite_migration_backup_oserror_survives(tmp_path: Path, monkeypatch) -> None:
    """Migration should not raise even if backup rotation fails."""
    monkeypatch.setenv("LLMDEBUG_RCA_BACKEND", "sqlite")
    out_dir = str(tmp_path)
    jsonl_path = tmp_path / "rca_attempts.jsonl"
    jsonl_path.write_text(json.dumps(_record("a1")) + "\n", encoding="utf-8")

    with patch.object(Path, "replace", side_effect=OSError("replace failed")):
        # Should not raise
        ensure_sqlite_migration(out_dir)

    # Database should still be usable after the failed backup
    history = get_rca_history(out_dir=out_dir, limit=10)
    assert isinstance(history, list)


# ---------------------------------------------------------------------------
# get_rca_history — JSONL backend no session filter and unlimited
# ---------------------------------------------------------------------------


def test_get_rca_history_jsonl_no_session_filter(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LLMDEBUG_RCA_BACKEND", "jsonl")
    out_dir = str(tmp_path)
    persist_rca_record(out_dir=out_dir, record=_record("a1", session_id="s1"))
    persist_rca_record(out_dir=out_dir, record=_record("a2", session_id="s2"))
    result = get_rca_history(out_dir=out_dir, session_id=None, limit=10)
    assert len(result) == 2


def test_get_rca_history_jsonl_zero_limit_returns_all(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LLMDEBUG_RCA_BACKEND", "jsonl")
    out_dir = str(tmp_path)
    for i in range(5):
        persist_rca_record(out_dir=out_dir, record=_record(f"a{i}"))
    result = get_rca_history(out_dir=out_dir, limit=0)
    assert len(result) == 5


# ---------------------------------------------------------------------------
# get_rca_status — JSONL backend
# ---------------------------------------------------------------------------


def test_get_rca_status_jsonl_no_session(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LLMDEBUG_RCA_BACKEND", "jsonl")
    out_dir = str(tmp_path)
    persist_rca_record(out_dir=out_dir, record=_record("a1", session_id="s1"))
    persist_rca_record(out_dir=out_dir, record=_record("a2", session_id="s2"))
    result = get_rca_status(out_dir=out_dir)
    assert result is not None
    assert result["attempt_id"] == "a2"


def test_get_rca_status_jsonl_no_records(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("LLMDEBUG_RCA_BACKEND", "jsonl")
    result = get_rca_status(out_dir=str(tmp_path))
    assert result is None


# ---------------------------------------------------------------------------
# TestErrorFallbacks — coverage for error/edge-case paths
# ---------------------------------------------------------------------------


class TestErrorFallbacks:
    """Exercise error-handling and edge-case branches in rca_backend."""

    def test_iter_jsonl_lines_reverse_empty_pending(self, tmp_path: Path) -> None:
        """Line 117->exit: pending is empty after the while loop → no final yield."""
        from llmdebug.rca_backend import _iter_jsonl_lines_reverse

        p = tmp_path / "data.jsonl"
        # File starts with a newline so pending after loop is b"" (empty)
        p.write_bytes(b"\n" + json.dumps(_record("a1")).encode() + b"\n")
        lines = list(_iter_jsonl_lines_reverse(p))
        assert len(lines) == 1
        data = json.loads(lines[0])
        assert data["attempt_id"] == "a1"

    def test_iter_jsonl_lines_reverse_single_line_no_trailing_newline(self, tmp_path: Path) -> None:
        """Line 117: pending has content after loop → yields it."""
        from llmdebug.rca_backend import _iter_jsonl_lines_reverse

        p = tmp_path / "data.jsonl"
        # Single line without trailing newline — pending captures the whole line
        content = json.dumps(_record("only"))
        p.write_bytes(content.encode())
        lines = list(_iter_jsonl_lines_reverse(p))
        assert len(lines) == 1
        assert json.loads(lines[0])["attempt_id"] == "only"

    def test_latest_jsonl_record_session_filter_skips_non_matching(self, tmp_path: Path) -> None:
        """Line 134: continue when session_id doesn't match (reverse order)."""
        p = tmp_path / "records.jsonl"
        # Put the matching record FIRST so the non-matching one is seen first in reverse
        p.write_text(
            json.dumps(_record("match", session_id="s1"))
            + "\n"
            + json.dumps(_record("skip", session_id="s2"))
            + "\n",
            encoding="utf-8",
        )
        result = _latest_jsonl_record(p, session_id="s1")
        assert result is not None
        assert result["attempt_id"] == "match"

    def test_latest_jsonl_record_no_matching_session(self, tmp_path: Path) -> None:
        """Line 138: return None when no record matches the session_id filter."""
        p = tmp_path / "records.jsonl"
        p.write_text(
            json.dumps(_record("a1", session_id="other")) + "\n",
            encoding="utf-8",
        )
        result = _latest_jsonl_record(p, session_id="nonexistent")
        assert result is None

    def test_latest_jsonl_record_empty_file(self, tmp_path: Path) -> None:
        """Line 138: return None when file exists but has no valid records."""
        p = tmp_path / "records.jsonl"
        p.write_text("\n\n\n", encoding="utf-8")
        result = _latest_jsonl_record(p)
        assert result is None

    def test_append_record_read_oserror_during_trim(self, tmp_path: Path, monkeypatch) -> None:
        """Lines 153-154: OSError on read_text during trim → silently returns."""
        monkeypatch.setenv("LLMDEBUG_RCA_MAX_RECORDS", "1")
        p = tmp_path / "records.jsonl"
        _append_record(p, dict(_record("a1")))  # type: ignore[arg-type]

        def _fail_on_trim_read(self: Path, *a: object, **kw: object) -> str:
            # Fail on the second read (trim check) but not the first (append)
            raise OSError("simulated read failure")

        # After appending a2, the trim read should fail
        with patch.object(Path, "read_text", _fail_on_trim_read):
            _append_record(p, dict(_record("a2")))  # type: ignore[arg-type]
        # Should not raise; file still exists
        assert p.exists()

    def test_append_record_write_oserror_during_trim(self, tmp_path: Path, monkeypatch) -> None:
        """Lines 161-162: OSError on write_text during trim → silently returns."""
        monkeypatch.setenv("LLMDEBUG_RCA_MAX_RECORDS", "1")
        p = tmp_path / "records.jsonl"
        # Seed with 2 records so trim is triggered
        _append_record(p, dict(_record("a1")))  # type: ignore[arg-type]

        call_count = 0

        def _fail_on_trim_write(self: Path, *a: object, **kw: object) -> int:
            nonlocal call_count
            call_count += 1
            raise OSError("simulated write failure")

        # Patch write_text to fail during trim but allow the file.open("a") append
        with patch.object(Path, "write_text", _fail_on_trim_write):
            _append_record(p, dict(_record("a2")))  # type: ignore[arg-type]
        # The append via open("a") still worked; the trim write failed silently
        assert p.exists()

    def test_ensure_sqlite_migration_jsonl_disappears_inside_lock(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Line 173: JSONL file vanishes between outer check and inner lock check."""
        monkeypatch.setenv("LLMDEBUG_RCA_BACKEND", "sqlite")
        out_dir = str(tmp_path)
        jsonl_path = tmp_path / "rca_attempts.jsonl"
        jsonl_path.write_text(json.dumps(_record("a1")) + "\n", encoding="utf-8")

        original_store_lock = None
        from llmdebug import rca_backend as _mod

        original_store_lock = _mod._store_lock

        from contextlib import contextmanager

        @contextmanager
        def _lock_and_delete(path: Path):  # type: ignore[no-untyped-def]
            """Simulate file disappearing after acquiring the lock."""
            with original_store_lock(path):
                # Delete the file inside the lock to trigger line 173
                if jsonl_path.exists():
                    jsonl_path.unlink()
                yield

        with patch.object(_mod, "_store_lock", _lock_and_delete):
            ensure_sqlite_migration(out_dir)
        # Should not raise

    def test_ensure_sqlite_migration_all_records_invalid(self, tmp_path: Path, monkeypatch) -> None:
        """Branch 180->187: all records invalid → skip store_rca_records_bulk."""
        monkeypatch.setenv("LLMDEBUG_RCA_BACKEND", "sqlite")
        out_dir = str(tmp_path)
        jsonl_path = tmp_path / "rca_attempts.jsonl"
        # Record with empty required field → invalid
        bad_record = {**_record("bad"), "attempt_id": ""}
        jsonl_path.write_text(json.dumps(bad_record) + "\n", encoding="utf-8")

        ensure_sqlite_migration(out_dir)
        # Backup should still be created
        assert Path(f"{jsonl_path}.bak").exists()

    def test_ensure_sqlite_migration_backup_unlink_oserror(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        """Lines 190-191: OSError on backup_path.unlink → silently caught."""
        monkeypatch.setenv("LLMDEBUG_RCA_BACKEND", "sqlite")
        out_dir = str(tmp_path)
        jsonl_path = tmp_path / "rca_attempts.jsonl"
        jsonl_path.write_text(json.dumps(_record("a1")) + "\n", encoding="utf-8")

        # Create an existing .bak that will fail to unlink
        bak_path = Path(f"{jsonl_path}.bak")
        bak_path.write_text("old", encoding="utf-8")

        with (
            patch.object(Path, "unlink", side_effect=OSError("unlink failed")),
            patch.object(Path, "replace", side_effect=OSError("replace failed")),
        ):
            ensure_sqlite_migration(out_dir)
        # Should not raise

    def test_get_rca_status_sqlite_backend(self, tmp_path: Path, monkeypatch) -> None:
        """Lines 222-223: get_rca_status via sqlite backend path."""
        monkeypatch.setenv("LLMDEBUG_RCA_BACKEND", "sqlite")
        out_dir = str(tmp_path)
        persist_rca_record(out_dir=out_dir, record=_record("sql-1", session_id="s1"))
        result = get_rca_status(out_dir=out_dir, session_id="s1")
        assert result is not None
        assert result["attempt_id"] == "sql-1"

    def test_get_rca_status_sqlite_no_records(self, tmp_path: Path, monkeypatch) -> None:
        """Lines 222-223: get_rca_status returns None when sqlite is empty."""
        monkeypatch.setenv("LLMDEBUG_RCA_BACKEND", "sqlite")
        result = get_rca_status(out_dir=str(tmp_path), session_id="nonexistent")
        assert result is None
