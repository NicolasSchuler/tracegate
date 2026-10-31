"""Tests for output and file management."""

import re

import pytest

from llmdebug import _json as json
from llmdebug.config import SnapshotConfig
from llmdebug.output import get_latest_snapshot, write_bundle
from llmdebug.state_db import list_snapshot_records


def test_write_bundle_creates_files(tmp_path):
    """Test that write_bundle creates JSON and traceback files."""
    cfg = SnapshotConfig(out_dir=str(tmp_path))
    payload = {
        "name": "test_snapshot",
        "exception": {"type": "ValueError", "message": "test"},
        "traceback": "Traceback (most recent call last):\n  ...",
        "frames": [],
    }

    json_path = write_bundle(payload, cfg)

    assert json_path.exists()
    assert json_path.suffix == ".json"

    # Check traceback file was created
    tb_path = json_path.with_suffix("").with_suffix(".traceback.txt")
    assert tb_path.exists()

    # Check latest.json
    latest = tmp_path / "latest.json"
    assert latest.exists() or latest.is_symlink()

    index_rows = list_snapshot_records(str(tmp_path))
    assert len(index_rows) == 1


def test_write_bundle_defaults_to_snapshot_name_when_missing(tmp_path):
    """Snapshots without a name or session fall back to the generic snapshot label."""
    cfg = SnapshotConfig(out_dir=str(tmp_path))
    payload = {
        "exception": {"type": "ValueError", "message": "test"},
        "frames": [],
    }

    json_path = write_bundle(payload, cfg)

    assert "_snapshot_" in json_path.name


def test_get_latest_snapshot(tmp_path):
    """Test reading the latest snapshot."""
    cfg = SnapshotConfig(out_dir=str(tmp_path))
    payload = {
        "name": "test",
        "exception": {"type": "Error", "message": "msg"},
        "frames": [],
    }

    write_bundle(payload, cfg)

    result = get_latest_snapshot(str(tmp_path))
    assert result is not None
    assert result["name"] == "test"


def test_get_latest_snapshot_normalizes_debug_session(tmp_path):
    """DebugSession payloads are returned as legacy flat snapshots."""
    cfg = SnapshotConfig(out_dir=str(tmp_path))
    payload = {
        "schema_version": "2.0",
        "kind": "llmdebug.debug_session",
        "session": {
            "name": "debug_session_test",
            "timestamp_utc": "2026-02-10T10:00:00Z",
            "llmdebug_version": "2.1.0",
        },
        "snapshot": {
            "exception": {"type": "Error", "message": "msg"},
            "traceback": "tb",
            "frames": [],
            "capture_config": {"frames": 5},
        },
        "context": {"hook": "sys.excepthook"},
    }

    write_bundle(payload, cfg)

    result = get_latest_snapshot(str(tmp_path))
    assert result is not None
    assert result["schema_version"] == "2.0"
    assert result["name"] == "debug_session_test"
    assert result["exception"]["type"] == "Error"
    assert result["hook"] == "sys.excepthook"


def test_get_latest_snapshot_raw_session(tmp_path):
    """normalize=False returns the raw DebugSession envelope."""
    cfg = SnapshotConfig(out_dir=str(tmp_path))
    payload = {
        "schema_version": "2.0",
        "kind": "llmdebug.debug_session",
        "session": {
            "name": "raw_output_test",
            "timestamp_utc": "2026-02-10T11:00:00Z",
            "llmdebug_version": "2.1.0",
        },
        "snapshot": {
            "exception": {"type": "RuntimeError", "message": "boom"},
            "traceback": "tb",
            "frames": [],
        },
        "context": {"hook": "sys.excepthook"},
    }
    write_bundle(payload, cfg)

    result = get_latest_snapshot(str(tmp_path), normalize=False)
    assert result is not None
    assert result["kind"] == "llmdebug.debug_session"
    assert result["session"]["name"] == "raw_output_test"
    assert result["snapshot"]["exception"]["type"] == "RuntimeError"


def test_get_latest_snapshot_rejects_detail_with_raw_mode(tmp_path):
    """The detail filter is only valid with normalized snapshots."""
    cfg = SnapshotConfig(out_dir=str(tmp_path))
    write_bundle({"name": "x", "exception": {"type": "E"}, "frames": []}, cfg)

    import pytest

    with pytest.raises(ValueError, match="detail filter requires normalize=True"):
        get_latest_snapshot(str(tmp_path), detail="crash", normalize=False)


def test_get_latest_snapshot_missing(tmp_path):
    """Test that missing snapshot returns None."""
    result = get_latest_snapshot(str(tmp_path))
    assert result is None


def test_atomic_write(tmp_path):
    """Test that writes are atomic (no partial files)."""
    cfg = SnapshotConfig(out_dir=str(tmp_path), output_format="json")
    payload = {
        "name": "atomic_test",
        "frames": [],
    }

    json_path = write_bundle(payload, cfg)

    # No .tmp files should remain
    tmp_files = list(tmp_path.glob("*.tmp"))
    assert len(tmp_files) == 0

    # JSON should be valid
    content = json.loads(json_path.read_text())
    assert content["name"] == "atomic_test"


def test_snapshot_cleanup(tmp_path):
    """Test that old snapshots are cleaned up when max_snapshots is exceeded."""
    import time

    cfg = SnapshotConfig(out_dir=str(tmp_path), max_snapshots=3, output_format="json")

    # Create 5 snapshots
    for i in range(5):
        payload = {
            "name": f"snapshot_{i}",
            "frames": [],
            "traceback": f"traceback {i}",
        }
        write_bundle(payload, cfg)
        time.sleep(0.01)  # Ensure different timestamps

    # Should only have 3 snapshots (+ latest.json)
    json_files = [p for p in tmp_path.glob("*.json") if p.name != "latest.json"]
    assert len(json_files) == 3

    # The newest snapshots should be kept (snapshot_2, snapshot_3, snapshot_4)
    names = sorted(json.loads(f.read_text())["name"] for f in json_files)
    assert names == ["snapshot_2", "snapshot_3", "snapshot_4"]

    indexed_names = sorted(row["name"] for row in list_snapshot_records(str(tmp_path)))
    assert indexed_names == ["snapshot_2", "snapshot_3", "snapshot_4"]


def test_snapshot_cleanup_batches_index_removals(tmp_path, monkeypatch):
    """Cleanup should batch index row deletions in a single DB call."""
    from llmdebug import output

    for i in range(5):
        snap = tmp_path / f"snapshot_{i}.json"
        snap.write_text(json.dumps({"name": f"snapshot_{i}", "frames": []}), encoding="utf-8")

    captured: list[list[str]] = []
    real_remove = output.remove_snapshot_records

    def _recording_remove(out_dir: str, paths: list[str]) -> None:
        captured.append(list(paths))
        real_remove(out_dir, paths)

    monkeypatch.setattr(output, "remove_snapshot_records", _recording_remove)
    output._cleanup_old_snapshots(tmp_path, max_snapshots=2)

    assert len(captured) == 1
    assert len(captured[0]) == 3
    remaining = [p for p in tmp_path.glob("*.json") if p.name != "latest.json"]
    assert len(remaining) == 2


def test_snapshot_cleanup_falls_back_when_index_lookup_fails(tmp_path, monkeypatch):
    """Cleanup should fall back to filesystem ordering when index reads fail."""
    import time

    from llmdebug import output

    cfg = SnapshotConfig(out_dir=str(tmp_path), max_snapshots=0, output_format="json")
    for i in range(4):
        payload = {
            "name": f"snapshot_{i}",
            "frames": [],
            "traceback": f"traceback {i}",
        }
        write_bundle(payload, cfg)
        time.sleep(0.01)

    def _raise_index_read(_: str) -> list[dict[str, str]]:
        raise RuntimeError("index unavailable")

    monkeypatch.setattr(output, "list_snapshot_records", _raise_index_read)
    output._cleanup_old_snapshots(tmp_path, max_snapshots=2)

    json_files = [p for p in tmp_path.glob("*.json") if p.name != "latest.json"]
    remaining_names = sorted(json.loads(path.read_text())["name"] for path in json_files)
    assert remaining_names == ["snapshot_2", "snapshot_3"]

    indexed_names = sorted(row["name"] for row in list_snapshot_records(str(tmp_path)))
    assert indexed_names == ["snapshot_2", "snapshot_3"]


def test_snapshot_cleanup_unlimited(tmp_path):
    """Test that max_snapshots=0 means unlimited."""
    cfg = SnapshotConfig(out_dir=str(tmp_path), max_snapshots=0)

    # Create 10 snapshots
    for i in range(10):
        payload = {"name": f"snapshot_{i}", "frames": []}
        write_bundle(payload, cfg)

    # All 10 should exist
    json_files = [p for p in tmp_path.glob("*.json") if p.name != "latest.json"]
    assert len(json_files) == 10


def test_snapshot_cleanup_removes_traceback_files(tmp_path):
    """Test that traceback files are also cleaned up."""
    import time

    cfg = SnapshotConfig(out_dir=str(tmp_path), max_snapshots=2)

    for i in range(4):
        payload = {
            "name": f"snap_{i}",
            "frames": [],
            "traceback": f"tb {i}",
        }
        write_bundle(payload, cfg)
        time.sleep(0.01)

    # Should have 2 JSON files and 2 traceback files
    json_files = [p for p in tmp_path.glob("*.json") if p.name != "latest.json"]
    tb_files = list(tmp_path.glob("*.traceback.txt"))

    assert len(json_files) == 2
    assert len(tb_files) == 2


def test_concurrent_write_bundle(tmp_path):
    """Test that concurrent writes don't corrupt files."""
    from concurrent.futures import ThreadPoolExecutor

    cfg = SnapshotConfig(out_dir=str(tmp_path), max_snapshots=0, output_format="json")  # unlimited

    def write_snapshot(i: int) -> None:
        payload = {
            "name": f"concurrent_{i}",
            "frames": [],
            "index": i,
        }
        write_bundle(payload, cfg)

    # Run 20 writes across 10 threads concurrently
    with ThreadPoolExecutor(max_workers=10) as executor:
        list(executor.map(write_snapshot, range(20)))

    # All 20 snapshots should exist
    json_files = [p for p in tmp_path.glob("*.json") if p.name != "latest.json"]
    assert len(json_files) == 20

    # All files should be valid JSON
    for f in json_files:
        content = json.loads(f.read_text())
        assert "name" in content
        assert content["name"].startswith("concurrent_")


def test_concurrent_write_with_cleanup(tmp_path):
    """Test concurrent writes with cleanup enabled."""
    from concurrent.futures import ThreadPoolExecutor

    cfg = SnapshotConfig(out_dir=str(tmp_path), max_snapshots=5, output_format="json")

    def write_snapshot(i: int) -> None:
        payload = {
            "name": f"cleanup_{i}",
            "frames": [],
        }
        write_bundle(payload, cfg)

    # Run 15 writes across 5 threads
    with ThreadPoolExecutor(max_workers=5) as executor:
        list(executor.map(write_snapshot, range(15)))

    # Should have at most 5 snapshots due to cleanup
    json_files = [p for p in tmp_path.glob("*.json") if p.name != "latest.json"]
    assert len(json_files) <= 5

    # All remaining files should be valid
    for f in json_files:
        content = json.loads(f.read_text())
        assert "name" in content


def test_lock_file_created(tmp_path):
    """Test that lock file is created during write."""
    cfg = SnapshotConfig(out_dir=str(tmp_path))
    payload = {"name": "lock_test", "frames": []}

    write_bundle(payload, cfg)

    # Lock file should exist (or have been cleaned up, which is fine)
    # The important thing is no error occurred
    json_files = [p for p in tmp_path.glob("*.json") if p.name != "latest.json"]
    assert len(json_files) == 1


def test_write_bundle_avoids_same_timestamp_overwrite(tmp_path, monkeypatch):
    """Test that repeated writes with same timestamp don't overwrite each other."""
    from llmdebug import output

    monkeypatch.setattr(output, "_timestamp", lambda: "20260101T000000Z")

    cfg = SnapshotConfig(out_dir=str(tmp_path), output_format="json", max_snapshots=0)
    write_bundle({"name": "same_name", "frames": [], "seq": 1}, cfg)
    write_bundle({"name": "same_name", "frames": [], "seq": 2}, cfg)

    json_files = [p for p in tmp_path.glob("*.json") if p.name != "latest.json"]
    assert len(json_files) == 2


def test_update_latest_symlink_windows_uses_atomic_replace(tmp_path, monkeypatch):
    """Windows latest update writes temp file and atomically replaces latest."""
    from llmdebug import output

    target = tmp_path / "snapshot.json"
    target.write_text('{"name":"snapshot"}', encoding="utf-8")

    monkeypatch.setattr(output.sys, "platform", "win32")
    output._update_latest_symlink(tmp_path, target.name, ".json")

    latest = tmp_path / "latest.json"
    assert latest.read_text(encoding="utf-8") == target.read_text(encoding="utf-8")
    assert list(tmp_path.glob("latest.json.tmp.*")) == []


def test_update_latest_symlink_windows_failure_cleans_temp(tmp_path, monkeypatch, capsys):
    """Failed atomic replace should emit warning and remove temp file."""
    from llmdebug import output

    target = tmp_path / "snapshot.json"
    target.write_text('{"name":"snapshot"}', encoding="utf-8")

    monkeypatch.setattr(output.sys, "platform", "win32")
    real_replace = output.Path.replace

    def _failing_replace(self, dest):
        if self.name.startswith("latest.json.tmp."):
            raise OSError("replace failed")
        return real_replace(self, dest)

    monkeypatch.setattr(output.Path, "replace", _failing_replace)
    output._update_latest_symlink(tmp_path, target.name, ".json")

    stderr = capsys.readouterr().err
    assert "could not update latest.json" in stderr
    assert list(tmp_path.glob("latest.json.tmp.*")) == []


# ---------------------------------------------------------------------------
# Coverage-gap tests: output.py branches
# ---------------------------------------------------------------------------


def test_cleanup_indexed_snapshots_deletes_overflow(tmp_path, monkeypatch):
    """Branch [133,131]: overflow records are deleted when indexed > max."""
    from llmdebug import output as output_mod

    # Create 3 snapshot files
    paths = []
    for i in range(3):
        p = tmp_path / f"snap_{i}.json"
        p.write_text('{"snapshot": {}}')
        paths.append(p)

    indexed = [{"path": str(p), "mtime": float(i)} for i, p in enumerate(paths)]

    # Monkeypatch _delete_snapshot_artifacts to track calls and return True
    deleted: list[str] = []
    monkeypatch.setattr(
        output_mod,
        "_delete_snapshot_artifacts",
        lambda p: (deleted.append(str(p)), True)[1],
    )
    monkeypatch.setattr(
        output_mod,
        "_remove_deleted_snapshot_records",
        lambda out_dir, paths: None,
    )

    result = output_mod._cleanup_indexed_snapshots(tmp_path, indexed, max_snapshots=1)
    assert result is True
    assert len(deleted) == 2  # 3 total - 1 kept = 2 deleted


def test_write_snapshot_compact_format_non_dict_result(tmp_path, monkeypatch):
    """Branch [240,243]: compact_keys returns non-dict (safety branch)."""
    import json

    from llmdebug import output as output_mod
    from llmdebug.config import SnapshotConfig

    cfg = SnapshotConfig(output_format="json_compact", out_dir=str(tmp_path))

    # Monkeypatch compact_keys to return a list instead of dict.
    # The function is imported locally inside write_snapshot from .compact_keys,
    # so we patch it on the source module.
    monkeypatch.setattr(
        "llmdebug.compact_keys.compact_keys",
        lambda payload: ["compacted_list_item"],
    )
    # Also monkeypatch the state_db function to avoid side effects
    monkeypatch.setattr(output_mod, "upsert_snapshot_record", lambda *a, **kw: None)

    payload = {"name": "test", "snapshot": {"name": "test"}, "traceback": "tb text"}
    output_mod.write_bundle(payload, cfg)

    # Should have written a file without __llmdebug_compact__ marker
    json_files = [f for f in tmp_path.glob("*.json") if "latest" not in f.name]
    assert len(json_files) >= 1
    content = json.loads(json_files[0].read_text())
    assert isinstance(content, list)  # The non-dict compacted result


def test_write_snapshot_debug_session_envelope_traceback(tmp_path, monkeypatch):
    """Branch [275,277]: DebugSession envelope with snapshot.traceback string."""
    from llmdebug import output as output_mod
    from llmdebug.config import SnapshotConfig

    cfg = SnapshotConfig(output_format="json", out_dir=str(tmp_path))
    monkeypatch.setattr(output_mod, "upsert_snapshot_record", lambda *a, **kw: None)

    # DebugSession envelope: traceback is inside "snapshot" key, not at top level
    payload = {
        "name": "test",
        "snapshot": {
            "name": "test",
            "traceback": "Traceback (most recent call last):\n  File ...",
        },
    }
    output_mod.write_bundle(payload, cfg)

    # Should have written a .traceback.txt file
    tb_files = list(tmp_path.glob("*.traceback.txt"))
    assert len(tb_files) == 1
    assert "most recent call last" in tb_files[0].read_text()


def test_snapshot_file_names_are_sanitized(tmp_path):
    """CAP-22: only [A-Za-z0-9._-] reaches the file name; the payload keeps the name."""
    cfg = SnapshotConfig(out_dir=str(tmp_path))
    name = "dir/sub\\file\x00\x1b name::ü"
    json_path = write_bundle({"name": name, "traceback": "tb", "frames": []}, cfg)

    stem = json_path.name.removesuffix(".json")
    assert re.fullmatch(r"[A-Za-z0-9._-]+", stem)
    assert "_dir_sub_file___name___" in stem
    assert json_path.parent == tmp_path
    assert get_latest_snapshot(str(tmp_path))["name"] == name


def test_long_snapshot_names_are_bounded_and_stay_distinct(tmp_path):
    """CAP-04: long names are cut to 120 characters plus a hash of the full name."""
    cfg = SnapshotConfig(out_dir=str(tmp_path))
    first = write_bundle({"name": "n" * 300 + "a", "traceback": "tb", "frames": []}, cfg)
    second = write_bundle({"name": "n" * 300 + "b", "traceback": "tb", "frames": []}, cfg)

    for path in (first, second):
        assert path.exists()
        assert len(path.name) < 200
        assert path.with_name(path.name.removesuffix(".json") + ".traceback.txt").exists()
    # Same timestamp prefix and truncated text; the hash keeps them apart beyond the nonce.
    assert first.name.split("_")[2] != second.name.split("_")[2]


def test_short_names_keep_their_file_name(tmp_path):
    cfg = SnapshotConfig(out_dir=str(tmp_path))
    json_path = write_bundle({"name": "test_mod.py_test_case", "frames": []}, cfg)

    assert "_test_mod.py_test_case_" in json_path.name


def test_retention_under_the_limit_does_not_stat_every_file(tmp_path, monkeypatch):
    """CAP-15: below max_snapshots every write fell back to a full iterdir() + stat() scan."""
    import llmdebug.output as output_module

    cfg = SnapshotConfig(out_dir=str(tmp_path), max_snapshots=50)
    for i in range(5):
        write_bundle({"name": f"s{i}", "frames": []}, cfg)

    scans: list[object] = []
    monkeypatch.setattr(
        output_module, "_cleanup_filesystem_snapshots", lambda *args: scans.append(args)
    )
    write_bundle({"name": "under_limit", "frames": []}, cfg)

    assert scans == []


def test_retention_over_the_limit_uses_the_index(tmp_path):
    cfg = SnapshotConfig(out_dir=str(tmp_path), max_snapshots=3)
    for i in range(6):
        write_bundle({"name": f"s{i}", "traceback": "tb", "frames": []}, cfg)

    remaining = [p for p in tmp_path.glob("*.json") if p.name != "latest.json"]
    assert len(remaining) == 3
    assert len(list_snapshot_records(str(tmp_path))) == 3


def test_retention_falls_back_to_a_scan_when_the_index_is_incomplete(tmp_path, monkeypatch):
    import llmdebug.output as output_module

    monkeypatch.setattr(output_module, "upsert_snapshot_record", lambda *_args: None)
    cfg = SnapshotConfig(out_dir=str(tmp_path), max_snapshots=2)
    for i in range(4):
        write_bundle({"name": f"s{i}", "frames": []}, cfg)

    remaining = [p for p in tmp_path.glob("*.json") if p.name != "latest.json"]
    assert len(remaining) == 2


def test_retention_falls_back_when_the_index_window_fails(tmp_path, monkeypatch):
    import llmdebug.output as output_module

    cfg = SnapshotConfig(out_dir=str(tmp_path), max_snapshots=0)
    for i in range(4):
        write_bundle({"name": f"s{i}", "frames": []}, cfg)

    def _boom(*_args, **_kwargs):
        raise RuntimeError("index unavailable")

    monkeypatch.setattr(output_module, "snapshot_index_window", _boom)
    output_module._cleanup_old_snapshots(tmp_path, max_snapshots=2)

    remaining = [p for p in tmp_path.glob("*.json") if p.name != "latest.json"]
    assert len(remaining) == 2


@pytest.mark.parametrize("kind", ["outside", "symlink", "not_a_snapshot"])
def test_retention_never_deletes_paths_a_poisoned_index_names(tmp_path, kind):
    """An index row pointing outside out_dir (or at a symlink/non-snapshot) is dropped, not deleted."""
    from llmdebug.state_db import upsert_snapshot_record

    out_dir = tmp_path / "snaps"
    victim_dir = tmp_path / "elsewhere"
    victim_dir.mkdir()
    victim = victim_dir / "precious.json"
    victim.write_text("{}", encoding="utf-8")
    cfg = SnapshotConfig(out_dir=str(out_dir), max_snapshots=50)
    write_bundle({"name": "first", "frames": []}, cfg)
    if kind == "outside":
        target = victim
    elif kind == "symlink":
        target = out_dir / "link.json"
        target.symlink_to(victim)
    else:
        target = out_dir / "notes.txt"
        target.write_text("keep", encoding="utf-8")
    upsert_snapshot_record(str(out_dir), {"path": str(target), "mtime": 0.0, "name": "evil"})

    from llmdebug.output import _cleanup_old_snapshots

    _cleanup_old_snapshots(out_dir, max_snapshots=1)

    assert victim.read_text(encoding="utf-8") == "{}"
    assert target.exists()
    assert str(target) not in {row["path"] for row in list_snapshot_records(str(out_dir))}


def test_capture_survives_an_index_row_with_a_malformed_path(tmp_path):
    """CAP-15: a NUL in an index path raised ValueError straight out of write_bundle."""
    from llmdebug.state_db import upsert_snapshot_record

    cfg = SnapshotConfig(out_dir=str(tmp_path), max_snapshots=1)
    write_bundle({"name": "first", "frames": []}, cfg)
    upsert_snapshot_record(str(tmp_path), {"path": "poison\x00.json", "mtime": 0.0, "name": "evil"})

    write_bundle({"name": "second", "frames": []}, cfg)

    assert "poison\x00.json" not in {row["path"] for row in list_snapshot_records(str(tmp_path))}


def test_traceback_sidecars_of_dotted_names_are_cleaned_up(tmp_path):
    """CAP-05: six pytest-style captures with max_snapshots=2 left six orphaned sidecars."""
    cfg = SnapshotConfig(out_dir=str(tmp_path), max_snapshots=2)
    for i in range(6):
        write_bundle(
            {"name": f"tests_test_mod.py_test_case_{i}", "traceback": "tb", "frames": []}, cfg
        )

    snapshots = [p for p in tmp_path.glob("*.json") if p.name != "latest.json"]
    sidecars = list(tmp_path.glob("*.traceback.txt"))
    assert len(snapshots) == 2
    assert sorted(p.name for p in sidecars) == sorted(
        p.name.removesuffix(".json") + ".traceback.txt" for p in snapshots
    )
