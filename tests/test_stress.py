"""Stress and concurrency tests for llmdebug.

Validates robustness under load: bulk cleanup, concurrent writes,
large serialization, SQLite WAL contention, and rate-limiter correctness.
"""

from __future__ import annotations

import sqlite3
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from re import Pattern
from typing import Any

import pytest

from llmdebug.config import SnapshotConfig
from llmdebug.hooks import _RateLimiter
from llmdebug.output import _cleanup_old_snapshots, write_bundle
from llmdebug.serialize import (
    compile_redactors,
    serialize_locals_bundle,
    to_jsonlike,
)
from llmdebug.state_db import (
    append_rca_record,
    fetch_latest_rca_record,
    list_snapshot_records,
    open_state_db,
    upsert_snapshot_record,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_MINIMAL_PAYLOAD: dict[str, Any] = {
    "name": "stress_test",
    "exception": {"type": "RuntimeError", "message": "boom"},
    "frames": [],
}


def _make_snapshot_file(out_dir: Path, index: int) -> Path:
    """Create a fake snapshot file with deterministic mtime."""
    p = out_dir / f"20250101T{index:06d}Z_snap_{index:08x}.json"
    p.write_text(f'{{"index": {index}}}', encoding="utf-8")
    # Stagger mtimes so cleanup ordering is deterministic.
    mtime = 1_700_000_000.0 + index
    import os

    os.utime(p, (mtime, mtime))
    return p


def _cfg(tmp_path: Path, **overrides: Any) -> SnapshotConfig:
    return SnapshotConfig(out_dir=str(tmp_path), **overrides)


# ---------------------------------------------------------------------------
# (a) Bulk snapshot cleanup — filesystem fallback path
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_cleanup_keeps_max_snapshots_filesystem(tmp_path: Path) -> None:
    """Create 1 200 snapshots, verify cleanup retains exactly max_snapshots."""
    total = 1_200
    max_keep = 50

    for i in range(total):
        _make_snapshot_file(tmp_path, i)

    remaining_before = [
        p for p in tmp_path.iterdir() if p.suffix == ".json" and not p.name.startswith("latest")
    ]
    assert len(remaining_before) == total

    _cleanup_old_snapshots(tmp_path, max_keep)

    remaining = [
        p for p in tmp_path.iterdir() if p.suffix == ".json" and not p.name.startswith("latest")
    ]
    assert len(remaining) == max_keep

    # The kept files should be the *newest* ones (highest index/mtime).
    kept_mtimes = sorted(p.stat().st_mtime for p in remaining)
    expected_start = 1_700_000_000.0 + (total - max_keep)
    assert kept_mtimes[0] >= expected_start


@pytest.mark.slow
def test_cleanup_keeps_max_snapshots_indexed(tmp_path: Path) -> None:
    """Cleanup via DB index path with 1 000+ indexed records."""
    total = 1_100
    max_keep = 30

    paths: list[Path] = []
    for i in range(total):
        p = _make_snapshot_file(tmp_path, i)
        paths.append(p)
        upsert_snapshot_record(
            str(tmp_path),
            {
                "path": str(p.resolve()),
                "mtime": 1_700_000_000.0 + i,
                "name": f"snap_{i}",
                "exc_type": "Err",
                "exc_message": "msg",
                "file": "f.py",
                "line": "1",
            },
        )

    records_before = list_snapshot_records(str(tmp_path))
    assert len(records_before) == total

    _cleanup_old_snapshots(tmp_path, max_keep)

    records_after = list_snapshot_records(str(tmp_path))
    assert len(records_after) == max_keep

    remaining_files = [
        p
        for p in tmp_path.iterdir()
        if p.suffix == ".json" and not p.name.startswith("latest") and not p.name.endswith(".tmp")
    ]
    assert len(remaining_files) == max_keep


# ---------------------------------------------------------------------------
# (b) Concurrent snapshot writes
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_concurrent_write_bundle(tmp_path: Path) -> None:
    """10 threads writing snapshots simultaneously — no corruption."""
    n_threads = 12
    writes_per_thread = 8
    cfg = _cfg(tmp_path, max_snapshots=0)  # unlimited — no cleanup race

    errors: list[Exception] = []

    def _writer(thread_id: int) -> list[Path]:
        thread_paths: list[Path] = []
        for j in range(writes_per_thread):
            payload = {
                "name": f"t{thread_id}_w{j}",
                "exception": {
                    "type": "ValueError",
                    "message": f"thread-{thread_id}-{j}",
                },
                "frames": [],
            }
            try:
                p = write_bundle(payload, cfg)
                thread_paths.append(p)
            except Exception as exc:
                errors.append(exc)
        return thread_paths

    all_paths: list[Path] = []
    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        futures = [pool.submit(_writer, tid) for tid in range(n_threads)]
        for fut in as_completed(futures):
            all_paths.extend(fut.result())

    assert not errors, f"Write errors: {errors}"
    expected = n_threads * writes_per_thread
    assert len(all_paths) == expected

    # Every file should be valid JSON.
    from llmdebug import _json as json

    for p in all_paths:
        assert p.exists(), f"Missing snapshot: {p}"
        data = json.loads(p.read_text(encoding="utf-8"))
        assert isinstance(data, dict)


# ---------------------------------------------------------------------------
# (c) Large locals serialization — bounded output
# ---------------------------------------------------------------------------


def _large_locals_dict(n_keys: int = 200) -> dict[str, Any]:
    """Build a locals dict with many large objects to exceed max_items."""
    d: dict[str, Any] = {}
    # Pad with enough keys to exceed max_items (50)
    for i in range(n_keys):
        d[f"var_{i}"] = f"value_{i}" * 20
    # Replace a few with interesting large objects
    d["big_dict"] = {f"key_{i}": f"value_{i}" * 20 for i in range(500)}
    d["deep_nested"] = _nested_dict(depth=30)
    d["long_string"] = "x" * 100_000
    d["big_list"] = list(range(10_000))
    return d


def _nested_dict(depth: int) -> dict[str, Any]:
    if depth <= 0:
        return {"leaf": True}
    return {"level": depth, "child": _nested_dict(depth - 1)}


def test_large_locals_serialization_bounded(tmp_path: Path) -> None:
    """Serializing very large locals stays bounded and completes."""
    cfg = _cfg(tmp_path, max_str=500, max_items=50)
    redactors: list[Pattern[str]] = compile_redactors(())
    big_locals = _large_locals_dict(500)

    bundle = serialize_locals_bundle(big_locals, cfg, redactors)

    # max_items caps the number of serialized keys
    assert len(bundle.values) <= cfg.max_items
    assert bundle.truncated is True
    assert bundle.truncated_keys > 0

    # The serialized output should be JSON-representable
    from llmdebug import _json as json

    text = json.dumps(bundle.values, default=str)
    # Output should be bounded — not megabytes of deep nesting
    assert len(text) < 500_000, f"Output too large: {len(text)} bytes"


def test_deep_nesting_capped_by_depth() -> None:
    """to_jsonlike truncates deep structures via depth limit."""
    cfg = SnapshotConfig(max_str=200, max_items=50)
    deep = _nested_dict(depth=50)

    result = to_jsonlike(deep, cfg, depth=0)

    # At depth >=2, objects are repr'd, so nesting is flat
    from llmdebug import _json as json

    text = json.dumps(result, default=str)
    assert len(text) < 50_000


@pytest.mark.slow
def test_large_numpy_array_serialization() -> None:
    """Serializing a large numpy array produces bounded summary."""
    np = pytest.importorskip("numpy")

    cfg = SnapshotConfig(max_str=500, max_items=50, include_array_stats=True)
    arr = np.random.randn(10_000, 100)  # 1M elements
    big_locals: dict[str, Any] = {"big_array": arr}
    redactors: list[Pattern[str]] = compile_redactors(())

    bundle = serialize_locals_bundle(big_locals, cfg, redactors)

    serialized = bundle.values.get("big_array")
    assert isinstance(serialized, dict)
    assert "__array__" in serialized  # array summary marker
    assert serialized["shape"] == [10_000, 100]

    from llmdebug import _json as json

    text = json.dumps(bundle.values, default=str)
    # Array summary should be compact, not a dump of 1M floats
    assert len(text) < 10_000


# ---------------------------------------------------------------------------
# (d) SQLite concurrent access — WAL mode contention
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_sqlite_concurrent_writes(tmp_path: Path) -> None:
    """Multiple threads writing RCA records concurrently via WAL mode."""
    n_threads = 10
    records_per_thread = 20
    now = "2025-01-01T00:00:00Z"

    errors: list[Exception] = []

    def _writer(tid: int) -> None:
        for j in range(records_per_thread):
            try:
                append_rca_record(
                    str(tmp_path),
                    {
                        "attempt_id": f"att-{tid}-{j}",
                        "session_id": f"sess-{tid}",
                        "updated_at": now,
                        "created_at": now,
                        "failure_signature": f"sig-{tid}-{j}",
                        "state": "investigating",
                    },
                    max_records=500,
                )
            except Exception as exc:
                errors.append(exc)

    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        futures = [pool.submit(_writer, tid) for tid in range(n_threads)]
        for fut in as_completed(futures):
            fut.result()

    assert not errors, f"DB write errors: {errors}"

    # Verify all records are present
    with open_state_db(str(tmp_path)) as conn:
        row = conn.execute("SELECT COUNT(*) FROM rca_attempts").fetchone()
    assert row is not None
    total = row[0]
    assert total == n_threads * records_per_thread


@pytest.mark.slow
def test_sqlite_concurrent_read_write(tmp_path: Path) -> None:
    """Readers and writers hitting the DB simultaneously."""
    now = "2025-06-01T00:00:00Z"

    # Seed some records.
    for i in range(20):
        append_rca_record(
            str(tmp_path),
            {
                "attempt_id": f"seed-{i}",
                "session_id": "sess-0",
                "updated_at": now,
                "created_at": now,
                "failure_signature": f"sig-{i}",
                "state": "investigating",
            },
            max_records=500,
        )

    read_results: list[dict[str, Any] | None] = []
    write_errors: list[Exception] = []

    def _reader() -> None:
        for _ in range(30):
            rec = fetch_latest_rca_record(str(tmp_path))
            read_results.append(rec)

    def _writer() -> None:
        for j in range(30):
            try:
                append_rca_record(
                    str(tmp_path),
                    {
                        "attempt_id": f"concurrent-{j}",
                        "session_id": "sess-1",
                        "updated_at": now,
                        "created_at": now,
                        "failure_signature": f"csig-{j}",
                        "state": "investigating",
                    },
                    max_records=500,
                )
            except Exception as exc:
                write_errors.append(exc)

    with ThreadPoolExecutor(max_workers=6) as pool:
        # 3 readers + 3 writers
        futs = []
        for _ in range(3):
            futs.append(pool.submit(_reader))
            futs.append(pool.submit(_writer))
        for fut in as_completed(futs):
            fut.result()

    assert not write_errors, f"Concurrent write errors: {write_errors}"
    # Readers should never get a corrupt result
    for rec in read_results:
        assert rec is None or isinstance(rec, dict)


@pytest.mark.slow
def test_sqlite_wal_journal_mode(tmp_path: Path) -> None:
    """Verify WAL mode is actually set on the database."""
    with open_state_db(str(tmp_path)) as conn:
        mode = conn.execute("PRAGMA journal_mode").fetchone()
    assert mode is not None
    assert mode[0] == "wal"


# ---------------------------------------------------------------------------
# (e) Rate limiter under load
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_rate_limiter_concurrent_bursts() -> None:
    """Fire 200 calls from 10 threads — rate limit respected."""
    max_captures = 5
    window = 60.0
    limiter = _RateLimiter(max_captures, window)

    allowed_count = 0
    denied_count = 0
    lock = __import__("threading").Lock()

    def _fire(thread_id: int) -> None:
        nonlocal allowed_count, denied_count
        local_allowed = 0
        local_denied = 0
        for _ in range(20):
            # Same signature from all threads
            if limiter.allow("ValueError", "test.py", 42):
                local_allowed += 1
            else:
                local_denied += 1
        with lock:
            allowed_count += local_allowed
            denied_count += local_denied

    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = [pool.submit(_fire, tid) for tid in range(10)]
        for fut in as_completed(futures):
            fut.result()

    total = allowed_count + denied_count
    assert total == 200
    # At most max_captures should be allowed for one signature
    assert allowed_count <= max_captures
    assert allowed_count >= 1  # At least one should get through


def test_rate_limiter_distinct_signatures() -> None:
    """Different signatures get independent windows."""
    limiter = _RateLimiter(max_captures=2, window_seconds=60.0)

    # Signature A: 2 allowed, 3rd blocked
    assert limiter.allow("ErrorA", "a.py", 1) is True
    assert limiter.allow("ErrorA", "a.py", 1) is True
    assert limiter.allow("ErrorA", "a.py", 1) is False

    # Signature B: independent, should still allow
    assert limiter.allow("ErrorB", "b.py", 2) is True
    assert limiter.allow("ErrorB", "b.py", 2) is True
    assert limiter.allow("ErrorB", "b.py", 2) is False


def test_rate_limiter_window_expiry() -> None:
    """After the window expires, calls are allowed again."""
    limiter = _RateLimiter(max_captures=1, window_seconds=0.05)

    assert limiter.allow("Err", "f.py", 1) is True
    assert limiter.allow("Err", "f.py", 1) is False

    time.sleep(0.08)  # Wait for window to expire

    assert limiter.allow("Err", "f.py", 1) is True


@pytest.mark.slow
def test_rate_limiter_many_signatures_cleanup() -> None:
    """Verify bucket cleanup when signature count exceeds threshold."""
    limiter = _RateLimiter(max_captures=5, window_seconds=0.01)

    # Fill with many unique signatures to trigger cleanup
    for i in range(1_100):
        limiter.allow("Err", f"file_{i}.py", i)

    time.sleep(0.02)  # Let all windows expire

    # After expiry + cleanup trigger, a new call should be allowed
    assert limiter.allow("Err", "new.py", 9999) is True
    # Buckets should have been pruned (internal detail, but verify no crash)
    assert len(limiter._buckets) < 1_100


# ---------------------------------------------------------------------------
# (f) Snapshot index concurrent upserts
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_snapshot_index_concurrent_upserts(tmp_path: Path) -> None:
    """Concurrent upsert_snapshot_record calls don't lose data."""
    n_threads = 8
    records_per_thread = 25

    errors: list[Exception] = []

    def _upserter(tid: int) -> None:
        for j in range(records_per_thread):
            try:
                upsert_snapshot_record(
                    str(tmp_path),
                    {
                        "path": f"/fake/t{tid}_s{j}.json",
                        "mtime": 1_700_000_000.0 + tid * 1000 + j,
                        "name": f"snap_{tid}_{j}",
                        "exc_type": "Err",
                        "exc_message": "msg",
                        "file": "f.py",
                        "line": str(j),
                    },
                )
            except Exception as exc:
                errors.append(exc)

    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        futures = [pool.submit(_upserter, tid) for tid in range(n_threads)]
        for fut in as_completed(futures):
            fut.result()

    assert not errors, f"Upsert errors: {errors}"

    records = list_snapshot_records(str(tmp_path))
    assert len(records) == n_threads * records_per_thread


# ---------------------------------------------------------------------------
# (g) write_bundle + cleanup under concurrency
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_write_bundle_with_cleanup_concurrent(tmp_path: Path) -> None:
    """Concurrent write_bundle calls with cleanup enabled."""
    max_snaps = 20
    n_threads = 6
    writes_per_thread = 10
    cfg = _cfg(tmp_path, max_snapshots=max_snaps)

    errors: list[Exception] = []

    def _writer(tid: int) -> None:
        for j in range(writes_per_thread):
            payload = {
                "name": f"conc_t{tid}_w{j}",
                "exception": {
                    "type": "RuntimeError",
                    "message": f"thread-{tid}-write-{j}",
                },
                "frames": [],
            }
            try:
                write_bundle(payload, cfg)
            except Exception as exc:
                errors.append(exc)

    with ThreadPoolExecutor(max_workers=n_threads) as pool:
        futures = [pool.submit(_writer, tid) for tid in range(n_threads)]
        for fut in as_completed(futures):
            fut.result()

    assert not errors, f"Write errors: {errors}"

    # After concurrent writes + cleanup, snapshot count should be bounded.
    snap_files = [
        p
        for p in tmp_path.iterdir()
        if p.suffix == ".json" and not p.name.startswith("latest") and not p.name.endswith(".tmp")
    ]
    assert len(snap_files) <= max_snaps + n_threads  # small race margin
