"""Additional edge-case coverage for output helpers."""

from __future__ import annotations

import contextlib
import os
import re
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

from llmdebug import _json as json
from llmdebug.config import SnapshotConfig
from llmdebug.output import get_latest_snapshot, write_bundle


def test_timestamp_format() -> None:
    from llmdebug.output import _timestamp

    stamp = _timestamp()
    assert re.fullmatch(r"\d{8}T\d{6}Z", stamp)


def test_directory_lock_timeout_branch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from llmdebug import output

    class _FakeLock:
        def __init__(self, _path: Path, timeout: float) -> None:
            self.timeout = timeout
            self.is_locked = False

        def acquire(self) -> None:
            raise output.Timeout(str(self.timeout))

        def release(self) -> None:
            self.is_locked = False

    monkeypatch.setattr(output, "FileLock", _FakeLock)

    with output._directory_lock(tmp_path, timeout=0.01) as acquired:
        assert acquired is False


def test_write_bundle_warns_when_lock_not_acquired(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from llmdebug import output

    @contextlib.contextmanager
    def _never_lock(_out_dir: Path, _timeout: float) -> Iterator[bool]:
        yield False

    monkeypatch.setattr(output, "_directory_lock", _never_lock)
    cfg = SnapshotConfig(out_dir=str(tmp_path), output_format="json")
    path = write_bundle({"name": "lock-timeout", "frames": []}, cfg)

    assert path.exists()
    assert "lock timeout" in capsys.readouterr().err


@pytest.mark.skipif(os.name == "nt", reason="POSIX-only symlink behavior")
def test_update_latest_symlink_posix_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from llmdebug import output

    target = tmp_path / "snapshot.json"
    target.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(output.sys, "platform", "darwin")
    output._update_latest_symlink(tmp_path, target.name, ".json")

    latest = tmp_path / "latest.json"
    assert latest.exists()
    assert latest.is_symlink()
    assert latest.resolve() == target.resolve()


@pytest.mark.skipif(os.name == "nt", reason="POSIX-only symlink behavior")
def test_update_latest_symlink_posix_failure_cleans_temp(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from llmdebug import output

    target = tmp_path / "snapshot.json"
    target.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(output.sys, "platform", "darwin")
    real_rename = output.Path.rename

    def _failing_rename(self: Path, dst: Path) -> Path:
        if self.name.startswith(".latest.json.tmp."):
            raise OSError("rename failed")
        return real_rename(self, dst)

    monkeypatch.setattr(output.Path, "rename", _failing_rename)
    output._update_latest_symlink(tmp_path, target.name, ".json")

    assert "could not update latest.json" in capsys.readouterr().err
    assert list(tmp_path.glob(".latest.json.tmp.*")) == []


def test_snapshot_index_entry_handles_missing_path_stat() -> None:
    from llmdebug.snapshot_index import index_entry_from_payload as _snapshot_index_entry

    missing = Path("/tmp/llmdebug-this-file-does-not-exist.json")
    payload = {
        "name": "entry-test",
        "exception": {"type": "ValueError", "message": "boom"},
        "frames": [{"file": "app.py", "line": 12}],
    }
    entry = _snapshot_index_entry(missing, payload)
    assert entry["mtime"] == 0.0
    assert entry["name"] == "entry-test"
    assert entry["exc_type"] == "ValueError"
    assert entry["file"] == "app.py"


def test_get_latest_snapshot_falls_back_from_invalid_toon_to_json(
    tmp_path: Path,
) -> None:
    latest_json = tmp_path / "latest.json"
    latest_json.write_text(
        json.dumps({"name": "json-fallback", "exception": {"type": "E"}, "frames": []}),
        encoding="utf-8",
    )
    latest_toon = tmp_path / "latest.toon"
    latest_toon.write_text("not-toon", encoding="utf-8")
    now = time.time()
    os.utime(latest_json, (now - 10, now - 10))
    os.utime(latest_toon, (now + 10, now + 10))

    snapshot = get_latest_snapshot(str(tmp_path))
    assert snapshot is not None
    assert snapshot["name"] == "json-fallback"


def test_get_latest_snapshot_warns_when_latest_json_is_not_dict(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / "latest.json").write_text(json.dumps(["bad", "payload"]), encoding="utf-8")

    assert get_latest_snapshot(str(tmp_path)) is None
    assert "contains list, expected dict" in capsys.readouterr().err


def test_get_latest_snapshot_warns_on_compact_expansion_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import llmdebug.compact_keys as compact_keys

    (tmp_path / "latest.json").write_text(
        json.dumps({"__llmdebug_compact__": 1, "_exc": {"_t": "ValueError"}}),
        encoding="utf-8",
    )

    monkeypatch.setattr(
        compact_keys, "expand_keys", lambda _data: (_ for _ in ()).throw(ValueError("bad"))
    )

    assert get_latest_snapshot(str(tmp_path)) is None
    assert "compact key expansion failed" in capsys.readouterr().err


def test_get_latest_snapshot_warns_on_corrupted_json(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    (tmp_path / "latest.json").write_text("{not-json", encoding="utf-8")

    assert get_latest_snapshot(str(tmp_path)) is None
    assert "is corrupted" in capsys.readouterr().err


def test_cleanup_filesystem_snapshots_tolerates_disappearing_file(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from llmdebug import output

    for i in range(4):
        (tmp_path / f"snapshot_{i}.json").write_text(
            json.dumps({"name": f"snapshot_{i}", "frames": []}),
            encoding="utf-8",
        )

    real_stat = output.Path.stat
    vanished_once = {"value": False}

    def _flaky_stat(self: Path, *args: object, **kwargs: object):
        if self.name == "snapshot_1.json" and not vanished_once["value"]:
            vanished_once["value"] = True
            self.unlink(missing_ok=True)
            raise FileNotFoundError("snapshot disappeared")
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(output.Path, "stat", _flaky_stat)
    output._cleanup_filesystem_snapshots(tmp_path, max_snapshots=2)
    monkeypatch.setattr(output.Path, "stat", real_stat)

    remaining = sorted(path.name for path in tmp_path.glob("snapshot_*.json"))
    assert remaining == ["snapshot_2.json", "snapshot_3.json"]


# ---------------------------------------------------------------------------
# Additional edge-case tests for uncovered lines
# ---------------------------------------------------------------------------


def test_update_latest_symlink_stale_cleanup_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lines 63-64: _update_latest_symlink — stale latest unlink raises."""
    from llmdebug import output

    target = tmp_path / "snapshot.json"
    target.write_text("{}", encoding="utf-8")

    # Create a stale latest.toon so the cleanup path is entered
    stale = tmp_path / "latest.toon"
    stale.write_text("stale", encoding="utf-8")

    real_unlink = Path.unlink

    def _failing_unlink(self: Path, *, missing_ok: bool = False) -> None:
        if self.name == "latest.toon":
            raise PermissionError("cannot remove stale")
        return real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(output.sys, "platform", "darwin")
    monkeypatch.setattr(output.Path, "unlink", _failing_unlink)
    # Should not raise — best-effort cleanup
    output._update_latest_symlink(tmp_path, target.name, ".json")

    # The target snapshot file should still exist after cleanup failure
    assert target.exists()


@pytest.mark.skipif(os.name != "nt", reason="Windows-only temp file cleanup")
def test_update_latest_symlink_windows_temp_cleanup_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Lines 75-76: Windows branch — temp file cleanup exception after write failure."""
    from llmdebug import output

    target = tmp_path / "snapshot.json"
    target.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(output.sys, "platform", "win32")

    real_write_text = Path.write_text

    def _failing_write(self: Path, data: str, *a: Any, **kw: Any) -> None:
        if "latest" in self.name and self.suffix == ".tmp":
            raise OSError("write failed")
        return real_write_text(self, data, *a, **kw)

    real_unlink = Path.unlink

    def _failing_unlink(self: Path, *, missing_ok: bool = False) -> None:
        if "latest" in self.name and ".tmp" in self.name:
            raise OSError("unlink failed too")
        return real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(output.Path, "write_text", _failing_write)
    monkeypatch.setattr(output.Path, "unlink", _failing_unlink)
    output._update_latest_symlink(tmp_path, target.name, ".json")
    assert "could not update latest.json" in capsys.readouterr().err


@pytest.mark.skipif(os.name == "nt", reason="POSIX-only")
def test_update_latest_symlink_posix_temp_cleanup_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Lines 88-89: POSIX — temp file cleanup exception after symlink rename failure."""
    from llmdebug import output

    target = tmp_path / "snapshot.json"
    target.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(output.sys, "platform", "darwin")

    real_rename = Path.rename

    def _failing_rename(self: Path, dst: Path) -> Path:
        if self.name.startswith(".latest.json.tmp."):
            raise OSError("rename failed")
        return real_rename(self, dst)

    real_unlink = Path.unlink

    def _failing_unlink(self: Path, *, missing_ok: bool = False) -> None:
        if self.name.startswith(".latest.json.tmp."):
            raise OSError("unlink also failed")
        return real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(output.Path, "rename", _failing_rename)
    monkeypatch.setattr(output.Path, "unlink", _failing_unlink)
    output._update_latest_symlink(tmp_path, target.name, ".json")
    assert "could not update latest.json" in capsys.readouterr().err


def test_delete_snapshot_artifacts_unlink_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Lines 97-99: _delete_snapshot_artifacts — snapshot unlink raises."""
    from llmdebug import output

    snap = tmp_path / "snap.json"
    snap.write_text("{}", encoding="utf-8")

    real_unlink = Path.unlink

    def _failing_unlink(self: Path, *, missing_ok: bool = False) -> None:
        if self.name == "snap.json":
            raise PermissionError("locked file")
        return real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(output.Path, "unlink", _failing_unlink)
    result = output._delete_snapshot_artifacts(snap)
    assert result is False
    assert "could not clean up snap.json" in capsys.readouterr().err


def test_delete_snapshot_artifacts_traceback_unlink_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Lines 104-105: _delete_snapshot_artifacts — traceback sidecar unlink raises."""
    from llmdebug import output

    snap = tmp_path / "snap.json"
    snap.write_text("{}", encoding="utf-8")
    tb = tmp_path / "snap.traceback.txt"
    tb.write_text("traceback", encoding="utf-8")

    real_unlink = Path.unlink

    def _failing_unlink(self: Path, *, missing_ok: bool = False) -> None:
        if self.name == "snap.traceback.txt":
            raise PermissionError("locked traceback")
        return real_unlink(self, missing_ok=missing_ok)

    monkeypatch.setattr(output.Path, "unlink", _failing_unlink)
    result = output._delete_snapshot_artifacts(snap)
    # Primary file is deleted, returns True
    assert result is True
    assert "could not clean up snap.json" in capsys.readouterr().err


def test_remove_deleted_snapshot_records_empty_list() -> None:
    """Line 112: _remove_deleted_snapshot_records — empty list short-circuit."""
    from llmdebug import output

    # Should return immediately without calling remove_snapshot_records
    result = output._remove_deleted_snapshot_records(Path("/nonexistent"), [])
    # Empty list returns None (early return), not an error
    assert result is None


def test_remove_deleted_snapshot_records_exception(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Lines 115-116: _remove_deleted_snapshot_records — exception handling."""
    from llmdebug import output

    def _failing_remove(_out_dir: str, _paths: list[str]) -> None:
        raise RuntimeError("db error")

    monkeypatch.setattr(output, "remove_snapshot_records", _failing_remove)
    output._remove_deleted_snapshot_records(tmp_path, ["/some/file.json"])
    assert "snapshot index cleanup failed" in capsys.readouterr().err


def test_write_snapshot_unknown_output_format(tmp_path: Path) -> None:
    """Line 251: write_bundle with unknown output_format → ValueError."""
    cfg = SnapshotConfig(out_dir=str(tmp_path), output_format="json")
    # Bypass __post_init__ validation by mutating after construction
    object.__setattr__(cfg, "output_format", "unknown_format")
    with pytest.raises(ValueError, match="Unknown output_format"):
        write_bundle({"name": "test", "frames": []}, cfg)


def test_get_latest_snapshot_stat_exception_fallback(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Lines 327-328: stat exception when gathering latest candidates."""
    from llmdebug import output

    latest_json = tmp_path / "latest.json"
    latest_json.write_text(
        json.dumps({"name": "stat-test", "exception": {"type": "E"}, "frames": []}),
        encoding="utf-8",
    )

    real_stat = Path.stat
    call_count: dict[str, int] = {"latest.json": 0}

    def _failing_stat(self: Path, *args: Any, **kwargs: Any) -> Any:
        if self.name == "latest.json":
            call_count["latest.json"] += 1
            # First call is from path.exists(), let it pass.
            # Second call is the explicit stat() for mtime — make it fail.
            if call_count["latest.json"] >= 2:
                raise OSError("stat failed")
        return real_stat(self, *args, **kwargs)

    monkeypatch.setattr(output.Path, "stat", _failing_stat)
    snapshot = get_latest_snapshot(str(tmp_path))
    assert snapshot is not None
    assert snapshot["name"] == "stat-test"


def test_get_latest_snapshot_toon_decode_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Lines 360, 371-373: TOON decode failure."""
    from llmdebug import toon_encoder

    latest_toon = tmp_path / "latest.toon"
    latest_toon.write_text("dummy", encoding="utf-8")

    def _exploding_decode(_content: str) -> Any:
        raise RuntimeError("decode exploded")

    monkeypatch.setattr(toon_encoder, "decode_snapshot_toon", _exploding_decode)
    snapshot = get_latest_snapshot(str(tmp_path))
    assert snapshot is None
    err = capsys.readouterr().err
    assert "decode failed" in err


def test_get_latest_snapshot_with_detail_filter(
    tmp_path: Path,
) -> None:
    """Lines 379, 381: detail filtering on snapshot read."""
    payload = {
        "name": "detail-test",
        "exception": {"type": "ValueError", "message": "boom"},
        "frames": [{"file": "app.py", "line": 12, "locals": {"x": 1}}],
        "traceback": "Traceback ...",
    }
    (tmp_path / "latest.json").write_text(json.dumps(payload), encoding="utf-8")
    snapshot = get_latest_snapshot(str(tmp_path), detail="crash")
    assert snapshot is not None
    # "crash" detail filters out traceback
    assert "traceback" not in snapshot


def test_get_latest_snapshot_compact_expansion_returns_non_dict(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Line 360: compact expand_keys returns non-dict → returns None."""
    import llmdebug.compact_keys as compact_keys

    (tmp_path / "latest.json").write_text(
        json.dumps({"__llmdebug_compact__": 1, "_exc": {"_t": "ValueError"}}),
        encoding="utf-8",
    )

    # Make expand_keys return a list instead of dict
    monkeypatch.setattr(compact_keys, "expand_keys", lambda _data: ["not", "a", "dict"])
    snapshot = get_latest_snapshot(str(tmp_path))
    assert snapshot is None
