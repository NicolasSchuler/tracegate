"""Tests for snapshot loader/index robustness paths."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import llmdebug._snapshot_loader as snapshot_loader
from llmdebug._snapshot_loader import (
    _has_snapshot_files,
    _latest_pointer_path,
    _normalize_line_value,
    _normalize_mtime_value,
    _scan_snapshot_entries,
    find_all_snapshots,
    load_snapshot,
    load_snapshot_raw,
    resolve_snapshot_ref_path,
)
from llmdebug.snapshot_index import is_confined_to


def test_find_all_snapshots_skips_invalid_index_mtime(
    tmp_path: Path,
    monkeypatch,
) -> None:
    good_path = tmp_path / "good.json"
    good_path.write_text("{}", encoding="utf-8")
    bad_path = tmp_path / "bad.json"
    bad_path.write_text("{}", encoding="utf-8")

    # The fake rows stand for a complete index (RCA-03 checks completeness separately).
    monkeypatch.setattr(snapshot_loader, "index_matches_disk", lambda _out_dir: True)
    monkeypatch.setattr(
        snapshot_loader,
        "list_snapshot_records",
        lambda _out_dir, limit=None, search=None: [
            {
                "path": str(good_path),
                "mtime": "1700000000.0",
                "name": "good",
                "exc_type": "ValueError",
                "exc_message": "ok",
                "file": "app.py",
                "line": "12",
            },
            {
                "path": str(bad_path),
                "mtime": "nan",
                "name": "bad",
                "exc_type": "TypeError",
                "exc_message": "bad",
                "file": "bad.py",
                "line": "5",
            },
        ],
    )

    removed: list[str] = []
    monkeypatch.setattr(
        snapshot_loader,
        "remove_snapshot_records",
        lambda _out_dir, paths: removed.extend(paths),
    )

    entries = snapshot_loader.find_all_snapshots(str(tmp_path))
    assert len(entries) == 1
    assert entries[0]["path"] == good_path.resolve()
    assert str(bad_path) in removed


def test_find_all_snapshots_filters_search_matches_from_index(
    tmp_path: Path,
    monkeypatch,
) -> None:
    keyerror_path = tmp_path / "keyerror.json"
    keyerror_path.write_text("{}", encoding="utf-8")
    valueerror_path = tmp_path / "valueerror.json"
    valueerror_path.write_text("{}", encoding="utf-8")

    # The fake rows stand for a complete index (RCA-03 checks completeness separately).
    monkeypatch.setattr(snapshot_loader, "index_matches_disk", lambda _out_dir: True)
    monkeypatch.setattr(
        snapshot_loader,
        "list_snapshot_records",
        lambda _out_dir, limit=None, search=None: (
            [
                {
                    "path": str(keyerror_path),
                    "mtime": "1700000001.0",
                    "name": "key_snap",
                    "exc_type": "KeyError",
                    "exc_message": "missing timeout",
                    "file": "config.py",
                    "line": "12",
                },
                {
                    "path": str(valueerror_path),
                    "mtime": "1700000000.0",
                    "name": "value_snap",
                    "exc_type": "ValueError",
                    "exc_message": "bad int",
                    "file": "app.py",
                    "line": "7",
                },
            ]
            if not search or "key" in str(search).lower()
            else []
        ),
    )

    entries = snapshot_loader.find_all_snapshots(str(tmp_path), search="KeyError")
    assert len(entries) == 1
    assert entries[0]["path"] == keyerror_path.resolve()
    assert entries[0]["exc_type"] == "KeyError"


# ---------------------------------------------------------------------------
# _load_snapshot_file_raw: TOON file paths
# ---------------------------------------------------------------------------


def test_load_snapshot_file_raw_toon_success(tmp_path: Path) -> None:
    """TOON file is decoded via toon_encoder.decode_snapshot_toon."""
    toon_path = tmp_path / "snap.toon"
    toon_path.write_text("dummy-toon-content", encoding="utf-8")

    fake_data = {"schema_version": "1.0", "name": "toon_test"}
    with patch.object(
        snapshot_loader, "_load_snapshot_file_raw", wraps=snapshot_loader._load_snapshot_file_raw
    ):
        # Patch decode_snapshot_toon inside the module
        import llmdebug.toon_encoder as toon_enc

        original_decode = toon_enc.decode_snapshot_toon
        toon_enc.decode_snapshot_toon = lambda _text: fake_data  # type: ignore[assignment]
        try:
            result = snapshot_loader._load_snapshot_file_raw(toon_path)
        finally:
            toon_enc.decode_snapshot_toon = original_decode  # type: ignore[assignment]
    assert result["name"] == "toon_test"


def test_load_snapshot_file_raw_toon_exception_raises_value_error(tmp_path: Path) -> None:
    """TOON decode exception (non-ImportError) becomes ValueError."""
    toon_path = tmp_path / "bad.toon"
    toon_path.write_text("junk", encoding="utf-8")

    import llmdebug.toon_encoder as toon_enc

    original_decode = toon_enc.decode_snapshot_toon

    def _boom(_text: str) -> dict:
        raise RuntimeError("decode failed")

    toon_enc.decode_snapshot_toon = _boom  # type: ignore[assignment]
    try:
        with pytest.raises(ValueError, match="Failed to decode TOON file"):
            snapshot_loader._load_snapshot_file_raw(toon_path)
    finally:
        toon_enc.decode_snapshot_toon = original_decode  # type: ignore[assignment]


def test_load_snapshot_file_raw_non_dict_raises(tmp_path: Path) -> None:
    """JSON file containing a list raises ValueError."""
    path = tmp_path / "list.json"
    path.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ValueError, match="Expected dict"):
        snapshot_loader._load_snapshot_file_raw(path)


def test_load_snapshot_file_raw_compact_expand_non_dict(tmp_path: Path, monkeypatch) -> None:
    """When expand_keys returns a non-dict, falls back to original dict."""
    from llmdebug import _json as json

    snap = {"__llmdebug_compact__": 1, "k": "v"}
    path = tmp_path / "compact.json"
    path.write_text(json.dumps(snap), encoding="utf-8")

    # Force is_compact_snapshot=True and expand_keys to return a list
    monkeypatch.setattr(snapshot_loader, "is_compact_snapshot", lambda _d: True)
    monkeypatch.setattr(snapshot_loader, "expand_keys", lambda _d: ["not", "a", "dict"])

    result = snapshot_loader._load_snapshot_file_raw(path)
    # Falls through to return the original data_dict
    assert result == snap


# ---------------------------------------------------------------------------
# load_snapshot / load_snapshot_raw: explicit path + validation
# ---------------------------------------------------------------------------


def test_load_snapshot_with_explicit_path(tmp_path: Path) -> None:
    """load_snapshot with explicit path validates and loads."""
    from llmdebug import _json as json

    snap = {"schema_version": "1.0", "name": "explicit", "exception": {"type": "E"}, "frames": []}
    p = tmp_path / "snap.json"
    p.write_text(json.dumps(snap), encoding="utf-8")

    result = load_snapshot(str(p), str(tmp_path))
    assert result["name"] == "explicit"


def test_load_snapshot_explicit_path_outside_dir_raises(tmp_path: Path) -> None:
    """load_snapshot with path outside out_dir raises ValueError."""
    other = tmp_path / "other" / "snap.json"
    other.parent.mkdir()
    other.write_text("{}", encoding="utf-8")
    subdir = tmp_path / "snapshots"
    subdir.mkdir()

    with pytest.raises(ValueError, match="outside snapshot directory"):
        load_snapshot(str(other), str(subdir))


def test_load_snapshot_no_snapshots_raises(tmp_path: Path) -> None:
    """load_snapshot with path=None and empty dir raises FileNotFoundError."""
    with (
        patch.object(snapshot_loader, "get_latest_snapshot", return_value=None),
        pytest.raises(FileNotFoundError, match="No snapshots found"),
    ):
        load_snapshot(None, str(tmp_path))


def test_load_snapshot_raw_explicit_path(tmp_path: Path) -> None:
    """load_snapshot_raw with explicit path validates and loads raw."""
    from llmdebug import _json as json

    snap = {"schema_version": "1.0", "name": "raw_explicit"}
    p = tmp_path / "snap.json"
    p.write_text(json.dumps(snap), encoding="utf-8")

    result = load_snapshot_raw(str(p), str(tmp_path))
    assert result["name"] == "raw_explicit"


def test_load_snapshot_raw_explicit_path_outside_dir_raises(tmp_path: Path) -> None:
    """load_snapshot_raw with path outside dir raises ValueError."""
    other_dir = tmp_path / "other"
    other_dir.mkdir()
    snap = other_dir / "snap.json"
    snap.write_text("{}", encoding="utf-8")
    allowed = tmp_path / "allowed"
    allowed.mkdir()

    with pytest.raises(ValueError, match="outside snapshot directory"):
        load_snapshot_raw(str(snap), str(allowed))


def test_load_snapshot_raw_no_snapshots_raises(tmp_path: Path) -> None:
    """load_snapshot_raw with path=None and no latest raises FileNotFoundError."""
    with (
        patch.object(snapshot_loader, "get_latest_snapshot", return_value=None),
        pytest.raises(FileNotFoundError, match="No snapshots found"),
    ):
        load_snapshot_raw(None, str(tmp_path))


# ---------------------------------------------------------------------------
# _has_snapshot_files: OSError path
# ---------------------------------------------------------------------------


def test_has_snapshot_files_oserror_returns_false(tmp_path: Path, monkeypatch) -> None:
    """_has_snapshot_files returns False on OSError from iterdir."""
    fake_path = MagicMock(spec=Path)
    fake_path.iterdir.side_effect = OSError("permission denied")
    result = _has_snapshot_files(fake_path)
    assert result is False


def test_has_snapshot_files_empty_dir(tmp_path: Path) -> None:
    result = _has_snapshot_files(tmp_path)
    assert result is False


def test_has_snapshot_files_with_json(tmp_path: Path) -> None:
    (tmp_path / "snap.json").write_text("{}", encoding="utf-8")
    result = _has_snapshot_files(tmp_path)
    assert result is True


# ---------------------------------------------------------------------------
# _normalize_line_value / _normalize_mtime_value
# ---------------------------------------------------------------------------


def test_normalize_line_value_digit_string() -> None:
    """String of digits is converted to int."""
    assert _normalize_line_value("42") == 42


def test_normalize_line_value_non_digit_string() -> None:
    assert _normalize_line_value("abc") == "abc"


def test_normalize_line_value_int_passthrough() -> None:
    assert _normalize_line_value(10) == 10


def test_normalize_mtime_value_type_error() -> None:
    """None triggers TypeError in float() → returns None."""
    assert _normalize_mtime_value(None) is None


def test_normalize_mtime_value_value_error() -> None:
    """Non-numeric string triggers ValueError in float() → returns None."""
    assert _normalize_mtime_value("not-a-number") is None


def test_normalize_mtime_value_nan() -> None:
    import math

    result = _normalize_mtime_value(float("nan"))
    assert result is None


def test_normalize_mtime_value_inf() -> None:
    result = _normalize_mtime_value(float("inf"))
    assert result is None


def test_normalize_mtime_value_valid() -> None:
    assert _normalize_mtime_value(1700000000.0) == 1700000000.0


# ---------------------------------------------------------------------------
# is_confined_to: OSError path
# ---------------------------------------------------------------------------


def test_is_within_out_dir_oserror_returns_false(tmp_path: Path) -> None:
    """is_confined_to returns False on OSError from resolve()."""
    fake_path = MagicMock(spec=Path)
    fake_path.resolve.side_effect = OSError("cannot resolve")
    result = is_confined_to(fake_path, str(tmp_path))
    assert result is False


def test_is_within_out_dir_outside_returns_false(tmp_path: Path) -> None:
    other = tmp_path.parent
    subdir = tmp_path / "sub"
    subdir.mkdir()
    result = is_confined_to(other, str(subdir))
    assert result is False


def test_is_within_out_dir_inside_returns_true(tmp_path: Path) -> None:
    child = tmp_path / "snap.json"
    child.write_text("{}", encoding="utf-8")
    result = is_confined_to(child, str(tmp_path))
    assert result is True


# ---------------------------------------------------------------------------
# _latest_pointer_path: OSError on stat
# ---------------------------------------------------------------------------


def test_latest_pointer_path_oserror_on_stat(tmp_path: Path, monkeypatch) -> None:
    """When stat() raises OSError, mtime defaults to 0.0 and path is still used."""
    latest = tmp_path / "latest.json"
    latest.write_text("{}", encoding="utf-8")

    call_count = {"n": 0}
    real_stat = Path.stat

    def _patched_stat(self: Path, **kwargs):
        if self.name == "latest.json":
            call_count["n"] += 1
            # First call is from exists()/is_file(), let those pass
            # The explicit .stat() call is the 3rd call (exists + is_file + explicit)
            if call_count["n"] >= 3:
                raise OSError("stat failed")
        return real_stat(self, **kwargs)

    monkeypatch.setattr(Path, "stat", _patched_stat)
    result = _latest_pointer_path(str(tmp_path))
    # Should still return latest.json with mtime=0.0 as default
    assert result is not None
    assert result.name == "latest.json"


def test_latest_pointer_path_no_latest_files(tmp_path: Path) -> None:
    result = _latest_pointer_path(str(tmp_path))
    assert result is None


# ---------------------------------------------------------------------------
# _entries_from_index: exception from list_snapshot_records
# ---------------------------------------------------------------------------


def test_entries_from_index_list_records_exception(tmp_path: Path, monkeypatch) -> None:
    """If list_snapshot_records raises, returns empty list with error flags."""
    monkeypatch.setattr(
        snapshot_loader,
        "list_snapshot_records",
        lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("db error")),
    )
    warnings: list[str] = []
    monkeypatch.setattr(snapshot_loader, "emit_warning", lambda msg: warnings.append(msg))

    entries, index_ok, index_failed = snapshot_loader._entries_from_index(str(tmp_path))
    assert entries == []
    assert index_ok is False
    assert index_failed is True
    assert any("snapshot index read failed" in w for w in warnings)


# ---------------------------------------------------------------------------
# _entries_from_index: stale path pruning exception
# ---------------------------------------------------------------------------


def test_entries_from_index_prune_stale_exception(tmp_path: Path, monkeypatch) -> None:
    """remove_snapshot_records exception emits warning but doesn't crash."""
    good_path = tmp_path / "good.json"
    good_path.write_text("{}", encoding="utf-8")
    missing_path = tmp_path / "missing.json"  # doesn't exist → stale

    monkeypatch.setattr(
        snapshot_loader,
        "list_snapshot_records",
        lambda *_a, **_kw: [
            {
                "path": str(good_path),
                "mtime": "1700000000.0",
                "name": "g",
                "exc_type": "E",
                "exc_message": "",
                "file": "f.py",
                "line": "1",
            },
            {
                "path": str(missing_path),
                "mtime": "1700000001.0",
                "name": "m",
                "exc_type": "E",
                "exc_message": "",
                "file": "f.py",
                "line": "2",
            },
        ],
    )
    monkeypatch.setattr(
        snapshot_loader,
        "remove_snapshot_records",
        lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("prune failed")),
    )
    warnings: list[str] = []
    monkeypatch.setattr(snapshot_loader, "emit_warning", lambda msg: warnings.append(msg))

    entries, _ok, _failed = snapshot_loader._entries_from_index(str(tmp_path))
    # good_path is a real file and within out_dir
    assert len(entries) == 1
    assert any("prune stale" in w for w in warnings)


# ---------------------------------------------------------------------------
# _scan_snapshot_entries: OSError from iterdir
# ---------------------------------------------------------------------------


def test_scan_snapshot_entries_oserror_from_iterdir(tmp_path: Path, monkeypatch) -> None:
    """If iterdir raises OSError, returns empty list."""
    fake_path = MagicMock(spec=Path)
    fake_path.iterdir.side_effect = OSError("no access")

    with patch("llmdebug._snapshot_loader.Path", return_value=fake_path):
        entries, skipped = _scan_snapshot_entries(str(tmp_path))
    assert entries == []
    assert skipped == 0


def test_scan_snapshot_entries_skips_corrupted_file(tmp_path: Path) -> None:
    """Files that fail to load are counted in skipped."""
    bad = tmp_path / "bad.json"
    bad.write_text("NOT VALID JSON{{{{", encoding="utf-8")

    entries, skipped = _scan_snapshot_entries(str(tmp_path))
    assert skipped >= 1
    assert all(e["path"] != bad.resolve() for e in entries)


def test_scan_snapshot_entries_skips_latest_files(tmp_path: Path) -> None:
    """Files named latest.* are skipped."""
    latest = tmp_path / "latest.json"
    latest.write_text("{}", encoding="utf-8")
    entries, _skipped = _scan_snapshot_entries(str(tmp_path))
    assert entries == []


# ---------------------------------------------------------------------------
# find_all_snapshots: replace_snapshot_records exception + skipped warning
# ---------------------------------------------------------------------------


def test_find_all_snapshots_replace_records_exception_emits_warning(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """replace_snapshot_records exception emits warning but still returns entries."""
    from llmdebug import _json as json
    from llmdebug.config import SnapshotConfig
    from llmdebug.output import write_bundle

    snap = {
        "schema_version": "1.0",
        "name": "warn_test",
        "exception": {"type": "ValueError", "message": "test"},
        "frames": [{"file": "f.py", "line": 1, "function": "f", "code": "", "locals": {}}],
    }
    cfg = SnapshotConfig(out_dir=str(tmp_path), output_format="json", max_snapshots=10)
    write_bundle(snap, cfg)

    # Force index miss by returning no rows
    monkeypatch.setattr(
        snapshot_loader,
        "list_snapshot_records",
        lambda *_a, **_kw: [],
    )
    monkeypatch.setattr(
        snapshot_loader,
        "replace_snapshot_records",
        lambda *_a, **_kw: (_ for _ in ()).throw(RuntimeError("db write failed")),
    )
    warnings: list[str] = []
    monkeypatch.setattr(snapshot_loader, "emit_warning", lambda msg: warnings.append(msg))

    entries = find_all_snapshots(str(tmp_path))
    # Should still return entries from filesystem scan
    assert len(entries) >= 1
    assert any("refresh snapshot index" in w for w in warnings)


def test_find_all_snapshots_skipped_warning(tmp_path: Path, monkeypatch) -> None:
    """When scan skips files, emits a warning about corrupted snapshots."""
    from llmdebug import _json as json

    # Create a bad json file
    (tmp_path / "bad.json").write_text("NOT JSON", encoding="utf-8")

    monkeypatch.setattr(
        snapshot_loader,
        "list_snapshot_records",
        lambda *_a, **_kw: [],
    )
    monkeypatch.setattr(snapshot_loader, "replace_snapshot_records", lambda *_a, **_kw: None)
    warnings: list[str] = []
    monkeypatch.setattr(snapshot_loader, "emit_warning", lambda msg: warnings.append(msg))

    find_all_snapshots(str(tmp_path))
    assert any("skipped" in w and "corrupted" in w for w in warnings)


# ---------------------------------------------------------------------------
# resolve_snapshot_ref_path: no snapshots for latest
# ---------------------------------------------------------------------------


def test_resolve_snapshot_ref_path_latest_no_snapshots_raises(tmp_path: Path, monkeypatch) -> None:
    """resolve_snapshot_ref_path raises FileNotFoundError when no snapshots exist."""
    monkeypatch.setattr(snapshot_loader, "_latest_pointer_path", lambda _d: None)
    monkeypatch.setattr(snapshot_loader, "find_all_snapshots", lambda *_a, **_kw: [])

    with pytest.raises(FileNotFoundError, match="No snapshots found"):
        resolve_snapshot_ref_path(None, str(tmp_path))


def test_resolve_snapshot_ref_path_previous_not_enough(tmp_path: Path, monkeypatch) -> None:
    """resolve_snapshot_ref_path 'previous' with only 1 snapshot raises."""
    from llmdebug import _json as json
    from llmdebug.config import SnapshotConfig
    from llmdebug.output import write_bundle

    snap = {
        "schema_version": "1.0",
        "name": "only_one",
        "exception": {"type": "E"},
        "frames": [],
    }
    cfg = SnapshotConfig(out_dir=str(tmp_path), output_format="json", max_snapshots=10)
    write_bundle(snap, cfg)

    with pytest.raises(FileNotFoundError, match="Need at least 2 snapshots"):
        resolve_snapshot_ref_path("previous", str(tmp_path))


def test_resolve_snapshot_ref_path_hash_invalid_ref_raises(
    tmp_path: Path,
) -> None:
    """resolve_snapshot_ref_path '#abc' raises ValueError."""
    with pytest.raises(ValueError, match="Invalid snapshot reference"):
        resolve_snapshot_ref_path("#abc", str(tmp_path))


def test_resolve_snapshot_ref_path_hash_out_of_range_raises(tmp_path: Path, monkeypatch) -> None:
    """resolve_snapshot_ref_path '#5' when only 2 snapshots raises ValueError."""
    monkeypatch.setattr(
        snapshot_loader,
        "find_all_snapshots",
        lambda *_a, **_kw: [{"path": tmp_path / "a.json"}, {"path": tmp_path / "b.json"}],
    )
    with pytest.raises(ValueError, match="not found"):
        resolve_snapshot_ref_path("#5", str(tmp_path))


def test_resolve_snapshot_ref_path_file_not_found_raises(tmp_path: Path) -> None:
    """resolve_snapshot_ref_path with nonexistent file path raises FileNotFoundError."""
    missing = str(tmp_path / "nonexistent.json")
    with pytest.raises((FileNotFoundError, ValueError)):
        resolve_snapshot_ref_path(missing, str(tmp_path))


# ---------------------------------------------------------------------------
# Coverage-gap tests: _snapshot_loader.py branches
# ---------------------------------------------------------------------------


def test_load_snapshot_from_file_toon_import_error(tmp_path: Path, monkeypatch) -> None:
    """Line 32-33: ImportError from toon_encoder import is re-raised, not swallowed."""
    import sys

    toon_file = tmp_path / "test.toon"
    toon_file.write_text("some toon content", encoding="utf-8")

    # Setting sys.modules[name] = None causes Python to raise
    # ModuleNotFoundError (subclass of ImportError) on the next import attempt.
    monkeypatch.setitem(sys.modules, "llmdebug.toon_encoder", None)

    with pytest.raises(ImportError):
        snapshot_loader.load_snapshot_from_file(toon_file)


def test_entries_from_index_empty_path_row_skipped(tmp_path: Path, monkeypatch) -> None:
    """Branch [202,203]: rows whose path_value is blank are silently skipped."""
    monkeypatch.setattr(
        snapshot_loader,
        "list_snapshot_records",
        lambda *_a, **_kw: [
            {
                "path": "",
                "mtime": "1.0",
                "name": "a",
                "exc_type": "E",
                "exc_message": "",
                "file": "f.py",
                "line": "1",
            },
            {
                "path": "   ",
                "mtime": "2.0",
                "name": "b",
                "exc_type": "E",
                "exc_message": "",
                "file": "f.py",
                "line": "2",
            },
        ],
    )
    # Safety: ensure no real DB operations even though stale_paths should be empty
    monkeypatch.setattr(snapshot_loader, "remove_snapshot_records", lambda *_a, **_kw: None)

    entries, index_ok, index_failed = snapshot_loader._entries_from_index(str(tmp_path))
    assert entries == []
    assert index_ok is True
    assert index_failed is False


def test_scan_snapshot_entries_stat_oserror_skips_file(tmp_path: Path) -> None:
    """Lines 260-262: OSError on path.stat() inside the loop is caught; file skipped."""
    fake_file = MagicMock()
    fake_file.is_file.return_value = True
    fake_file.suffix = ".json"
    fake_file.name = "test.json"
    fake_file.stat.side_effect = OSError("permission denied")

    fake_dir = MagicMock(spec=Path)
    fake_dir.iterdir.return_value = iter([fake_file])

    with patch("llmdebug._snapshot_loader.Path", return_value=fake_dir):
        entries, skipped = _scan_snapshot_entries(str(tmp_path))

    # File was silently skipped — no entries, no load-error skip count
    assert entries == []
    assert skipped == 0


def test_resolve_snapshot_ref_latest_fallback_to_snapshots(
    tmp_path: Path,
    monkeypatch,
) -> None:
    """Branch [374,375]: ref='latest', no pointer file, find_all_snapshots non-empty."""
    snap = tmp_path / "snap.json"
    snap.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(snapshot_loader, "_latest_pointer_path", lambda _d: None)
    monkeypatch.setattr(
        snapshot_loader,
        "find_all_snapshots",
        lambda *_a, **_kw: [{"path": snap}],
    )

    result = resolve_snapshot_ref_path("latest", str(tmp_path))
    assert result == snap


def test_load_snapshot_trust_path_skips_validation(tmp_path: Path) -> None:
    """Branch [94,96]: _trust_path=True bypasses _validate_snapshot_path."""
    import json as json_std

    outside = tmp_path / "outside"
    outside.mkdir()
    snap = outside / "snap.json"
    snap.write_text(json_std.dumps({"name": "test"}), encoding="utf-8")

    out_dir = tmp_path / "snapshots"
    out_dir.mkdir()

    # Without _trust_path the path-traversal guard rejects the file
    with pytest.raises(ValueError, match="outside snapshot directory"):
        load_snapshot(path=str(snap), out_dir=str(out_dir))

    # With _trust_path=True, validation is skipped and load succeeds
    result = load_snapshot(path=str(snap), out_dir=str(out_dir), _trust_path=True)
    assert isinstance(result, dict)


def test_load_snapshot_raw_trust_path_skips_validation(tmp_path: Path) -> None:
    """Branch [109,111]: _trust_path=True bypasses validation in raw loader."""
    import json as json_std

    outside = tmp_path / "outside"
    outside.mkdir()
    snap = outside / "snap.json"
    snap.write_text(json_std.dumps({"name": "test_raw"}), encoding="utf-8")

    out_dir = tmp_path / "snapshots"
    out_dir.mkdir()

    # Without _trust_path the path-traversal guard rejects the file
    with pytest.raises(ValueError, match="outside snapshot directory"):
        load_snapshot_raw(path=str(snap), out_dir=str(out_dir))

    # With _trust_path=True, validation is skipped and load succeeds
    result = load_snapshot_raw(path=str(snap), out_dir=str(out_dir), _trust_path=True)
    assert isinstance(result, dict)


def test_partially_populated_index_does_not_hide_snapshots(tmp_path: Path) -> None:
    """RCA-03: 3 files on disk and 1 indexed row returned 1 snapshot, forever."""
    import os
    import time

    from llmdebug.config import SnapshotConfig
    from llmdebug.output import write_bundle
    from llmdebug.state_db import count_snapshot_records, list_snapshot_records

    cfg = SnapshotConfig(out_dir=str(tmp_path), output_format="json")
    paths = []
    for index, name in enumerate(("oldest", "middle", "newest")):
        path = write_bundle({"name": name, "exception": {"type": "E"}, "frames": []}, cfg)
        os.utime(path, (time.time() + index, time.time() + index))
        paths.append(path)
    # Best-effort index writes failed for the two newest files (e.g. xdist contention).
    from llmdebug.state_db import remove_snapshot_records

    remove_snapshot_records(str(tmp_path), [str(p.resolve()) for p in paths[1:]])
    assert len(list_snapshot_records(str(tmp_path))) == 1

    newest = snapshot_loader.find_all_snapshots(str(tmp_path), limit=1)
    everything = snapshot_loader.find_all_snapshots(str(tmp_path))

    assert [entry["name"] for entry in newest] == ["newest"]
    assert len(everything) == 3
    assert count_snapshot_records(str(tmp_path)) == 3  # the rescan repaired the index


# ---------------------------------------------------------------------------
# RCA-02: confining the directory must also confine what is read from inside it
# ---------------------------------------------------------------------------


def _external_snapshot(tmp_path: Path) -> Path:
    outside = tmp_path / "outside"
    outside.mkdir()
    target = outside / "secret.json"
    target.write_text('{"name": "secret", "exception": {"type": "Secret"}}', encoding="utf-8")
    return target


def test_latest_pointer_symlink_out_of_out_dir_is_not_loaded(tmp_path: Path) -> None:
    target = _external_snapshot(tmp_path)
    out_dir = tmp_path / "snaps"
    out_dir.mkdir()
    (out_dir / "latest.json").symlink_to(target)

    assert _latest_pointer_path(str(out_dir)) is None
    with pytest.raises(FileNotFoundError):
        load_snapshot(None, str(out_dir))
    with pytest.raises(FileNotFoundError):
        load_snapshot_raw(None, str(out_dir))
    with pytest.raises(FileNotFoundError):
        resolve_snapshot_ref_path("latest", str(out_dir))


def test_snapshot_symlink_out_of_out_dir_is_not_listed(tmp_path: Path) -> None:
    target = _external_snapshot(tmp_path)
    out_dir = tmp_path / "snaps"
    out_dir.mkdir()
    (out_dir / "escape.json").symlink_to(target)
    inside = out_dir / "real.json"
    inside.write_text('{"name": "real", "exception": {"type": "ValueError"}}', encoding="utf-8")

    names = {str(entry["name"]) for entry in find_all_snapshots(str(out_dir))}
    assert names == {"real"}
