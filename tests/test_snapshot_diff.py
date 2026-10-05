"""Tests for snapshot diffing logic and resolve_snapshot_ref."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import pytest

from llmdebug import _json as json
from llmdebug._snapshot_loader import resolve_snapshot_ref
from llmdebug.snapshot_diff import (
    diff_exception,
    diff_locals,
    diff_location,
    diff_snapshots,
    format_diff_text,
    structured_diff_summary,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def snapshot_a() -> dict[str, Any]:
    """Base snapshot — ValueError at app.py:42 in parse_input."""
    return {
        "schema_version": "1.0",
        "name": "test_parse_a",
        "timestamp_utc": "2026-02-08T10:00:00Z",
        "llmdebug_version": "2.0.0",
        "exception": {
            "type": "ValueError",
            "qualified_type": "builtins.ValueError",
            "message": "invalid literal for int(): 'abc'",
            "error_category": {"category": "value_error", "suggestion": "Check input types"},
        },
        "frames": [
            {
                "file": "/code/app.py",
                "file_rel": "app.py",
                "line": 42,
                "function": "parse_input",
                "locals": {
                    "value": "'abc'",
                    "x": 10,
                    "old_var": "hello",
                    "@py_builtins": "<module>",
                },
                "locals_meta": {
                    "value": {"type": "builtins.str"},
                    "x": {"type": "builtins.int"},
                    "old_var": {"type": "builtins.str"},
                },
            },
        ],
        "crash_frame_index": 0,
        "git": {"commit": "abc1234", "branch": "main", "dirty": False},
        "pytest": {"nodeid": "tests/test_app.py::test_parse"},
    }


@pytest.fixture
def snapshot_b() -> dict[str, Any]:
    """Modified snapshot — TypeError at app.py:38 in transform_data, changed locals."""
    return {
        "schema_version": "1.0",
        "name": "test_parse_b",
        "timestamp_utc": "2026-02-08T10:05:00Z",
        "llmdebug_version": "2.0.0",
        "exception": {
            "type": "TypeError",
            "qualified_type": "builtins.TypeError",
            "message": "unsupported operand type(s) for +: 'int' and 'str'",
            "error_category": {"category": "type_error", "suggestion": "Check operand types"},
        },
        "frames": [
            {
                "file": "/code/app.py",
                "file_rel": "app.py",
                "line": 38,
                "function": "transform_data",
                "locals": {
                    "value": 42,
                    "x": 10,
                    "new_var": [1, 2, 3],
                    "@py_assert0": True,
                },
                "locals_meta": {
                    "value": {"type": "builtins.int"},
                    "x": {"type": "builtins.int"},
                    "new_var": {"type": "builtins.list"},
                },
            },
        ],
        "crash_frame_index": 0,
        "git": {"commit": "def5678", "branch": "main", "dirty": True},
        "pytest": {"nodeid": "tests/test_app.py::test_parse"},
    }


def _write_snapshot(directory: Path, data: dict[str, Any], name: str) -> Path:
    """Write a snapshot JSON file to a directory."""
    path = directory / f"{name}.json"
    path.write_text(json.dumps(data, default=str), encoding="utf-8")
    return path


def _as_debug_session(snapshot: dict[str, Any]) -> dict[str, Any]:
    """Wrap a legacy snapshot as DebugSession for compatibility checks."""
    return {
        "schema_version": "2.0",
        "kind": "llmdebug.debug_session",
        "session": {
            "name": snapshot.get("name", "?"),
            "timestamp_utc": snapshot.get("timestamp_utc", "2026-02-08T10:00:00Z"),
            "llmdebug_version": snapshot.get("llmdebug_version", "2.0.0"),
        },
        "snapshot": {
            "exception": snapshot.get("exception", {}),
            "traceback": snapshot.get("traceback", ""),
            "frames": snapshot.get("frames", []),
            "capture_config": snapshot.get("capture_config", {}),
            "crash_frame_index": snapshot.get("crash_frame_index", 0),
        },
        "context": {
            "git": snapshot.get("git", {}),
            "pytest": snapshot.get("pytest", {}),
        },
    }


@pytest.fixture
def two_snapshot_dir(tmp_path: Path, snapshot_a: dict, snapshot_b: dict) -> Path:
    """Dir with two different snapshots; snapshot_a is older, snapshot_b is newer."""
    _write_snapshot(tmp_path, snapshot_a, "snap_old")
    time.sleep(0.05)
    _write_snapshot(tmp_path, snapshot_b, "snap_new")
    # Also write latest.json pointing to the newer one
    latest_path = tmp_path / "latest.json"
    latest_path.write_text(json.dumps(snapshot_b, default=str), encoding="utf-8")
    return tmp_path


# ---------------------------------------------------------------------------
# TestResolveSnapshotRef
# ---------------------------------------------------------------------------


class TestResolveSnapshotRef:
    def test_latest(self, two_snapshot_dir: Path) -> None:
        result = resolve_snapshot_ref("latest", str(two_snapshot_dir))
        assert result["name"] == "test_parse_b"

    def test_latest_none(self, two_snapshot_dir: Path) -> None:
        result = resolve_snapshot_ref(None, str(two_snapshot_dir))
        assert result["name"] == "test_parse_b"

    def test_latest_prefers_latest_pointer_over_newest_snapshot(self, tmp_path: Path) -> None:
        older = {
            "schema_version": "1.0",
            "name": "older_pointed",
            "exception": {"type": "ValueError", "message": "older"},
            "frames": [{"file_rel": "older.py", "line": 1, "function": "f"}],
            "crash_frame_index": 0,
        }
        newer = {
            "schema_version": "1.0",
            "name": "newer_file",
            "exception": {"type": "TypeError", "message": "newer"},
            "frames": [{"file_rel": "newer.py", "line": 2, "function": "g"}],
            "crash_frame_index": 0,
        }
        _write_snapshot(tmp_path, older, "snap_old")
        time.sleep(0.05)
        _write_snapshot(tmp_path, newer, "snap_new")
        (tmp_path / "latest.json").write_text(json.dumps(older), encoding="utf-8")

        result = resolve_snapshot_ref("latest", str(tmp_path))
        assert result["name"] == "older_pointed"

    def test_previous(self, two_snapshot_dir: Path) -> None:
        result = resolve_snapshot_ref("previous", str(two_snapshot_dir))
        assert result["name"] == "test_parse_a"

    def test_index_ref_1(self, two_snapshot_dir: Path) -> None:
        result = resolve_snapshot_ref("#1", str(two_snapshot_dir))
        # #1 = newest (find_all_snapshots is newest-first)
        assert result["name"] == "test_parse_b"

    def test_index_ref_2(self, two_snapshot_dir: Path) -> None:
        result = resolve_snapshot_ref("#2", str(two_snapshot_dir))
        assert result["name"] == "test_parse_a"

    def test_file_path(self, two_snapshot_dir: Path) -> None:
        path = two_snapshot_dir / "snap_old.json"
        result = resolve_snapshot_ref(str(path), str(two_snapshot_dir))
        assert result["name"] == "test_parse_a"

    def test_previous_with_one_snapshot(self, tmp_path: Path, snapshot_a: dict) -> None:
        _write_snapshot(tmp_path, snapshot_a, "only_one")
        with pytest.raises(FileNotFoundError, match="at least 2 snapshots"):
            resolve_snapshot_ref("previous", str(tmp_path))

    def test_index_out_of_range(self, two_snapshot_dir: Path) -> None:
        with pytest.raises(ValueError, match="Snapshot #99 not found"):
            resolve_snapshot_ref("#99", str(two_snapshot_dir))

    def test_invalid_index_format(self, two_snapshot_dir: Path) -> None:
        with pytest.raises(ValueError, match="Invalid snapshot reference"):
            resolve_snapshot_ref("#abc", str(two_snapshot_dir))

    def test_nonexistent_file_path(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            resolve_snapshot_ref(str(tmp_path / "nope.json"), str(tmp_path))


# ---------------------------------------------------------------------------
# TestDiffException
# ---------------------------------------------------------------------------


class TestDiffException:
    def test_same_exception(self) -> None:
        exc = {
            "type": "ValueError",
            "message": "bad",
            "error_category": {"category": "value_error"},
        }
        result = diff_exception(exc, exc)
        assert not result.changed

    def test_different_type(self) -> None:
        old = {"type": "ValueError", "message": "bad"}
        new = {"type": "TypeError", "message": "bad"}
        result = diff_exception(old, new)
        assert result.changed
        assert result.old_type == "ValueError"
        assert result.new_type == "TypeError"

    def test_different_message(self) -> None:
        old = {"type": "ValueError", "message": "bad"}
        new = {"type": "ValueError", "message": "worse"}
        result = diff_exception(old, new)
        assert result.changed
        assert result.old_message == "bad"
        assert result.new_message == "worse"

    def test_different_category(self) -> None:
        old = {"type": "E", "message": "", "error_category": {"category": "a"}}
        new = {"type": "E", "message": "", "error_category": {"category": "b"}}
        result = diff_exception(old, new)
        assert result.changed
        assert result.old_category == "a"
        assert result.new_category == "b"

    def test_empty_dicts(self) -> None:
        result = diff_exception({}, {})
        assert not result.changed


# ---------------------------------------------------------------------------
# TestDiffLocation
# ---------------------------------------------------------------------------


class TestDiffLocation:
    def test_same_location(self) -> None:
        frame = {"file_rel": "app.py", "line": 42, "function": "f"}
        result = diff_location([frame], [frame], 0, 0)
        assert not result.changed
        assert result.old_location == "app.py:42 in f"

    def test_different_file(self) -> None:
        old = [{"file_rel": "a.py", "line": 1, "function": "f"}]
        new = [{"file_rel": "b.py", "line": 1, "function": "f"}]
        result = diff_location(old, new, 0, 0)
        assert result.changed

    def test_different_line(self) -> None:
        old = [{"file_rel": "a.py", "line": 1, "function": "f"}]
        new = [{"file_rel": "a.py", "line": 2, "function": "f"}]
        result = diff_location(old, new, 0, 0)
        assert result.changed

    def test_different_function(self) -> None:
        old = [{"file_rel": "a.py", "line": 1, "function": "f"}]
        new = [{"file_rel": "a.py", "line": 1, "function": "g"}]
        result = diff_location(old, new, 0, 0)
        assert result.changed

    def test_no_frames(self) -> None:
        result = diff_location([], [], 0, 0)
        assert not result.changed
        assert result.old_location == "?"


# ---------------------------------------------------------------------------
# TestDiffLocals
# ---------------------------------------------------------------------------


class TestDiffLocals:
    def test_identical_locals(self) -> None:
        frame = {"locals": {"x": 1, "y": 2}, "locals_meta": {}}
        result = diff_locals(frame, frame)
        assert len(result.changes) == 0
        assert result.unchanged_count == 2

    def test_modified_value(self) -> None:
        old = {"locals": {"x": 1}, "locals_meta": {"x": {"type": "int"}}}
        new = {"locals": {"x": 2}, "locals_meta": {"x": {"type": "int"}}}
        result = diff_locals(old, new)
        assert len(result.changes) == 1
        assert result.changes[0].kind == "modified"
        assert result.changes[0].name == "x"
        assert result.changes[0].old_value == 1
        assert result.changes[0].new_value == 2

    def test_added_variable(self) -> None:
        old = {"locals": {}, "locals_meta": {}}
        new = {"locals": {"y": 42}, "locals_meta": {"y": {"type": "int"}}}
        result = diff_locals(old, new)
        assert len(result.changes) == 1
        assert result.changes[0].kind == "added"
        assert result.changes[0].name == "y"

    def test_removed_variable(self) -> None:
        old = {"locals": {"z": "gone"}, "locals_meta": {"z": {"type": "str"}}}
        new = {"locals": {}, "locals_meta": {}}
        result = diff_locals(old, new)
        assert len(result.changes) == 1
        assert result.changes[0].kind == "removed"
        assert result.changes[0].name == "z"

    def test_type_changed(self) -> None:
        old = {"locals": {"v": "'abc'"}, "locals_meta": {"v": {"type": "str"}}}
        new = {"locals": {"v": 42}, "locals_meta": {"v": {"type": "int"}}}
        result = diff_locals(old, new)
        assert len(result.changes) == 1
        assert result.changes[0].kind == "type_changed"

    def test_array_shape_changed(self) -> None:
        old_arr = {"__array__": True, "shape": [100, 50], "dtype": "float64"}
        new_arr = {"__array__": True, "shape": [100, 30], "dtype": "float64"}
        old = {"locals": {"matrix": old_arr}, "locals_meta": {}}
        new = {"locals": {"matrix": new_arr}, "locals_meta": {}}
        result = diff_locals(old, new)
        assert len(result.changes) == 1
        assert result.changes[0].kind == "shape_changed"
        assert result.changes[0].old_value == [100, 50]
        assert result.changes[0].new_value == [100, 30]

    def test_array_became_empty(self) -> None:
        old_arr = {"__array__": True, "shape": [100, 50], "dtype": "float64"}
        new_arr = {"__array__": True, "shape": [0, 50], "dtype": "float64"}
        old = {"locals": {"matrix": old_arr}, "locals_meta": {}}
        new = {"locals": {"matrix": new_arr}, "locals_meta": {}}
        result = diff_locals(old, new)
        assert len(result.changes) == 1
        assert result.changes[0].kind == "shape_changed"
        assert result.changes[0].new_value == [0, 50]

    def test_skips_pytest_internals(self) -> None:
        old = {"locals": {"@py_builtins": "x", "real": 1}, "locals_meta": {}}
        new = {"locals": {"@py_assert0": "y", "real": 1}, "locals_meta": {}}
        result = diff_locals(old, new)
        assert len(result.changes) == 0
        assert result.unchanged_count == 1

    def test_empty_frames(self) -> None:
        result = diff_locals({}, {})
        assert len(result.changes) == 0
        assert result.unchanged_count == 0


# ---------------------------------------------------------------------------
# TestDiffSnapshots
# ---------------------------------------------------------------------------


class TestDiffSnapshots:
    def test_identical_snapshots(self, snapshot_a: dict) -> None:
        result = diff_snapshots(snapshot_a, snapshot_a)
        assert not result.exception.changed
        assert not result.location.changed
        assert len(result.locals_diff.changes) == 0
        assert not result.git_changed
        assert result.same_test
        assert result.failure_signature_same

    def test_full_diff(self, snapshot_a: dict, snapshot_b: dict) -> None:
        result = diff_snapshots(snapshot_a, snapshot_b)
        assert result.exception.changed
        assert result.exception.old_type == "ValueError"
        assert result.exception.new_type == "TypeError"
        assert result.location.changed
        assert result.git_changed
        assert result.same_test  # Same nodeid

        # Locals: value modified, old_var removed, new_var added, x unchanged
        kinds = {c.name: c.kind for c in result.locals_diff.changes}
        assert "old_var" in kinds
        assert kinds["old_var"] == "removed"
        assert "new_var" in kinds
        assert kinds["new_var"] == "added"
        # value changed type (str -> int)
        assert kinds["value"] == "type_changed"
        # x stayed the same
        assert result.locals_diff.unchanged_count == 1
        assert not result.failure_signature_same
        assert result.evidence_delta.message_changed
        assert "app.py" in result.change_delta.touched_files

    def test_same_test_detection(self, snapshot_a: dict, snapshot_b: dict) -> None:
        # Same nodeid
        assert diff_snapshots(snapshot_a, snapshot_b).same_test
        # Different nodeid
        modified = {**snapshot_b, "pytest": {"nodeid": "other::test"}}
        assert not diff_snapshots(snapshot_a, modified).same_test

    def test_no_frames(self) -> None:
        old = {"exception": {"type": "E"}, "frames": []}
        new = {"exception": {"type": "E"}, "frames": []}
        result = diff_snapshots(old, new)
        assert not result.location.changed
        assert len(result.locals_diff.changes) == 0

    def test_accepts_debug_session_inputs(
        self, snapshot_a: dict[str, Any], snapshot_b: dict[str, Any]
    ) -> None:
        old = _as_debug_session(snapshot_a)
        new = _as_debug_session(snapshot_b)
        result = diff_snapshots(old, new)
        assert result.exception.changed
        assert result.old_name == "test_parse_a"
        assert result.new_name == "test_parse_b"

    def test_hypothesis_delta_classification(self, snapshot_a: dict, snapshot_b: dict) -> None:
        old = {**snapshot_a, "rca": {"hypothesis": "Input parsing fails."}}
        new = {**snapshot_b, "rca": {"hypothesis": "Input parsing fails. Guard int conversion."}}
        result = diff_snapshots(old, new)
        assert result.hypothesis_delta == "refined"

    def test_structured_summary_shape(self, snapshot_a: dict, snapshot_b: dict) -> None:
        result = diff_snapshots(snapshot_a, snapshot_b)
        structured = structured_diff_summary(result)
        assert "failure_signature_same" in structured
        assert "evidence_delta" in structured
        assert "change_delta" in structured
        assert "verification_delta" in structured

    def test_handles_string_crash_frame_index(self) -> None:
        old = {
            "exception": {"type": "ValueError", "message": "bad"},
            "frames": [
                {
                    "file_rel": "a.py",
                    "line": 1,
                    "function": "f",
                    "locals": {"x": 1},
                    "locals_meta": {"x": {"type": "int"}},
                },
                {
                    "file_rel": "b.py",
                    "line": 2,
                    "function": "g",
                    "locals": {"x": 2},
                    "locals_meta": {"x": {"type": "int"}},
                },
            ],
            "crash_frame_index": "1",
        }
        new = {
            "exception": {"type": "ValueError", "message": "bad"},
            "frames": [
                {
                    "file_rel": "a.py",
                    "line": 1,
                    "function": "f",
                    "locals": {"x": 1},
                    "locals_meta": {"x": {"type": "int"}},
                },
                {
                    "file_rel": "b.py",
                    "line": 2,
                    "function": "g",
                    "locals": {"x": 3},
                    "locals_meta": {"x": {"type": "int"}},
                },
            ],
            "crash_frame_index": "1",
        }
        result = diff_snapshots(old, new)
        assert len(result.locals_diff.changes) == 1
        assert result.locals_diff.changes[0].name == "x"
        assert result.change_delta.touched_files == ()

    def test_handles_invalid_crash_frame_index_without_raising(self) -> None:
        old = {
            "exception": {"type": "ValueError", "message": "bad"},
            "frames": [
                "malformed",
                {
                    "file_rel": "b.py",
                    "line": 2,
                    "function": "g",
                    "locals": {"x": 1},
                    "locals_meta": {"x": {"type": "int"}},
                },
            ],
            "crash_frame_index": "not-an-int",
        }
        new = {
            "exception": {"type": "ValueError", "message": "worse"},
            "frames": [
                "malformed",
                {
                    "file_rel": "c.py",
                    "line": 3,
                    "function": "h",
                    "locals": {"x": 2},
                    "locals_meta": {"x": {"type": "int"}},
                },
            ],
            "crash_frame_index": "still-bad",
        }
        result = diff_snapshots(old, new)
        assert result.location.changed
        assert "b.py" in result.location.old_location
        assert "c.py" in result.location.new_location
        assert "b.py" in result.change_delta.touched_files
        assert "c.py" in result.change_delta.touched_files

    def test_handles_malformed_exception_and_locals_shapes(self) -> None:
        old = {
            "exception": "not-a-dict",
            "frames": [
                {
                    "file_rel": None,
                    "file": "/repo/app.py",
                    "line": 10,
                    "function": None,
                    "locals": ["not", "a", "dict"],
                    "locals_meta": "bad-meta",
                }
            ],
            "crash_frame_index": "bad",
        }
        new = {
            "exception": {"type": "TypeError", "message": "boom", "error_category": "bad"},
            "frames": [
                {
                    "file_rel": None,
                    "file": "/repo/app.py",
                    "line": 10,
                    "function": "run",
                    "locals": {"x": 1},
                    "locals_meta": {"x": {"type": None}},
                }
            ],
            "crash_frame_index": 0,
        }
        result = diff_snapshots(old, new)
        assert result.exception.new_type == "TypeError"
        assert result.location.old_location.startswith("/repo/app.py:")
        assert any(change.name == "x" for change in result.locals_diff.changes)

    def test_file_rel_none_falls_back_to_file_for_identity(self) -> None:
        old = {
            "exception": {"type": "ValueError", "message": "bad"},
            "frames": [
                {
                    "file_rel": None,
                    "file": "/repo/old.py",
                    "line": 1,
                    "function": "f",
                    "locals": {},
                    "locals_meta": {},
                }
            ],
            "crash_frame_index": 0,
        }
        new = {
            "exception": {"type": "ValueError", "message": "bad"},
            "frames": [
                {
                    "file_rel": None,
                    "file": "/repo/new.py",
                    "line": 1,
                    "function": "f",
                    "locals": {},
                    "locals_meta": {},
                }
            ],
            "crash_frame_index": 0,
        }
        result = diff_snapshots(old, new)
        assert result.location.changed
        assert "/repo/old.py" in result.location.old_location
        assert "/repo/new.py" in result.location.new_location
        assert "/repo/old.py" in result.change_delta.touched_files
        assert "/repo/new.py" in result.change_delta.touched_files


# ---------------------------------------------------------------------------
# TestFormatDiffText
# ---------------------------------------------------------------------------


class TestFormatDiffText:
    def test_identical_shows_no_changes(self, snapshot_a: dict) -> None:
        result = diff_snapshots(snapshot_a, snapshot_a)
        text = format_diff_text(result)
        assert text == "No changes detected. Snapshots are identical."

    def test_all_sections_present(self, snapshot_a: dict, snapshot_b: dict) -> None:
        result = diff_snapshots(snapshot_a, snapshot_b)
        text = format_diff_text(result)
        assert "## Snapshot Diff" in text
        assert "## Exception" in text
        assert "CHANGED" in text
        assert "## Crash Location" in text
        assert "MOVED" in text
        assert "## Locals (crash frame)" in text
        assert "## Git" in text
        assert "abc1234" in text
        assert "def5678" in text
        assert "## Structured Delta" in text

    def test_empty_shape_flagged(self) -> None:
        old_arr = {"__array__": True, "shape": [100, 50], "dtype": "float64"}
        new_arr = {"__array__": True, "shape": [0, 50], "dtype": "float64"}
        old = {
            "exception": {"type": "E", "message": ""},
            "frames": [
                {
                    "file_rel": "a.py",
                    "line": 1,
                    "function": "f",
                    "locals": {"m": old_arr},
                    "locals_meta": {},
                }
            ],
            "crash_frame_index": 0,
        }
        new = {
            "exception": {"type": "E", "message": ""},
            "frames": [
                {
                    "file_rel": "a.py",
                    "line": 1,
                    "function": "f",
                    "locals": {"m": new_arr},
                    "locals_meta": {},
                }
            ],
            "crash_frame_index": 0,
        }
        result = diff_snapshots(old, new)
        text = format_diff_text(result)
        assert "SHAPE CHANGED" in text
        assert "!! EMPTY" in text

    def test_unchanged_exception_section(self) -> None:
        snap = {
            "exception": {"type": "ValueError", "message": "bad"},
            "frames": [
                {
                    "file_rel": "a.py",
                    "line": 1,
                    "function": "f",
                    "locals": {"x": 1},
                    "locals_meta": {},
                }
            ],
            "crash_frame_index": 0,
            "git": {"commit": "aaa"},
        }
        snap2 = {
            "exception": {"type": "ValueError", "message": "bad"},
            "frames": [
                {
                    "file_rel": "a.py",
                    "line": 1,
                    "function": "f",
                    "locals": {"x": 2},
                    "locals_meta": {},
                }
            ],
            "crash_frame_index": 0,
            "git": {"commit": "aaa"},
        }
        result = diff_snapshots(snap, snap2)
        text = format_diff_text(result)
        assert "UNCHANGED: ValueError: bad" in text
        assert "MODIFIED: x" in text

    def test_structured_changes_prevent_identical_short_circuit(self) -> None:
        old = {
            "name": "old",
            "exception": {"type": "ValueError", "message": "bad"},
            "frames": [
                {
                    "file_rel": "a.py",
                    "line": 1,
                    "function": "f",
                    "locals": {"x": 1},
                    "locals_meta": {"x": {"type": "int"}},
                }
            ],
            "crash_frame_index": 0,
            "git": {"commit": "abc"},
        }
        new = {
            "name": "new",
            "exception": {"type": "ValueError", "message": "bad"},
            "frames": [
                {
                    "file_rel": "a.py",
                    "line": 1,
                    "function": "f",
                    "locals": {"x": 1},
                    "locals_meta": {"x": {"type": "int"}},
                },
                {
                    "file_rel": "helper.py",
                    "line": 9,
                    "function": "helper",
                    "locals": {},
                    "locals_meta": {},
                },
            ],
            "crash_frame_index": 0,
            "git": {"commit": "abc"},
        }
        result = diff_snapshots(old, new)
        text = format_diff_text(result)
        assert text != "No changes detected. Snapshots are identical."
        assert "## Structured Delta" in text


# ---------------------------------------------------------------------------
# Coverage gap tests — lines 48, 275-276, 284, 441, 457-464, 470-480, 676-684
# ---------------------------------------------------------------------------


class TestFrameFileValueNoFile:
    """Line 48: _frame_file_value returns '?' when frame has no file."""

    def test_empty_frame_returns_question_mark(self) -> None:
        from llmdebug.snapshot_diff import _frame_file_value

        assert _frame_file_value({}) == "?"

    def test_none_file_rel_and_file_returns_question_mark(self) -> None:
        from llmdebug.snapshot_diff import _frame_file_value

        assert _frame_file_value({"file_rel": None, "file": None}) == "?"

    def test_empty_strings_returns_question_mark(self) -> None:
        from llmdebug.snapshot_diff import _frame_file_value

        assert _frame_file_value({"file_rel": "", "file": ""}) == "?"


class TestDiffSharedKeyArrayPaths:
    """Lines 275-276, 284: array dtype change, value change, and matching arrays."""

    def test_array_dtype_change_returns_modified(self) -> None:
        """Line 275: same shape but different dtype → modified."""
        from llmdebug.snapshot_diff import _diff_shared_key

        old_locals = {"arr": {"__array__": True, "shape": [3, 3], "dtype": "float32"}}
        new_locals = {"arr": {"__array__": True, "shape": [3, 3], "dtype": "float64"}}
        result = _diff_shared_key("arr", old_locals, new_locals, {}, {})
        assert result is not None
        assert result.kind == "modified"

    def test_array_same_shape_same_dtype_returns_none(self) -> None:
        """Line 284: both arrays identical → return None."""
        from llmdebug.snapshot_diff import _diff_shared_key

        arr = {"__array__": True, "shape": [10], "dtype": "float64"}
        old_locals = {"arr": arr}
        new_locals = {"arr": dict(arr)}  # Same content
        result = _diff_shared_key("arr", old_locals, new_locals, {}, {})
        assert result is None

    def test_array_different_values_same_shape_dtype(self) -> None:
        """Line 275: same shape/dtype but different extra keys → modified."""
        from llmdebug.snapshot_diff import _diff_shared_key

        old_locals = {
            "arr": {"__array__": True, "shape": [3], "dtype": "float64", "stats": {"mean": 1.0}}
        }
        new_locals = {
            "arr": {"__array__": True, "shape": [3], "dtype": "float64", "stats": {"mean": 2.0}}
        }
        result = _diff_shared_key("arr", old_locals, new_locals, {}, {})
        assert result is not None
        assert result.kind == "modified"


class TestClassifyHypothesisDelta:
    """Lines 451-464: _classify_hypothesis_delta all branches."""

    def test_both_empty_returns_unknown(self) -> None:
        from llmdebug.snapshot_diff import _classify_hypothesis_delta

        assert _classify_hypothesis_delta("", "") == "unknown"

    def test_equal_returns_unchanged(self) -> None:
        from llmdebug.snapshot_diff import _classify_hypothesis_delta

        assert _classify_hypothesis_delta("same", "same") == "unchanged"

    def test_empty_to_new_returns_added(self) -> None:
        from llmdebug.snapshot_diff import _classify_hypothesis_delta

        assert _classify_hypothesis_delta("", "new hypothesis") == "added"

    def test_old_to_empty_returns_removed(self) -> None:
        from llmdebug.snapshot_diff import _classify_hypothesis_delta

        assert _classify_hypothesis_delta("old hypothesis", "") == "removed"

    def test_old_contained_in_new_returns_refined(self) -> None:
        from llmdebug.snapshot_diff import _classify_hypothesis_delta

        assert (
            _classify_hypothesis_delta("parsing fails", "parsing fails on edge cases") == "refined"
        )

    def test_completely_different_returns_replaced(self) -> None:
        from llmdebug.snapshot_diff import _classify_hypothesis_delta

        assert _classify_hypothesis_delta("first theory", "completely different") == "replaced"


class TestExtractHypothesis:
    """Lines 438-448: _extract_hypothesis with rca.hypothesis nested path."""

    def test_top_level_hypothesis_string(self) -> None:
        from llmdebug.snapshot_diff import _extract_hypothesis

        view = {"hypothesis": "Direct hypothesis"}
        assert _extract_hypothesis(view) == "Direct hypothesis"

    def test_rca_nested_hypothesis(self) -> None:
        from llmdebug.snapshot_diff import _extract_hypothesis

        view = {"rca": {"hypothesis": "Nested hypothesis"}}
        assert _extract_hypothesis(view) == "Nested hypothesis"

    def test_no_hypothesis_returns_empty(self) -> None:
        from llmdebug.snapshot_diff import _extract_hypothesis

        assert _extract_hypothesis({}) == ""

    def test_non_string_hypothesis_ignored(self) -> None:
        from llmdebug.snapshot_diff import _extract_hypothesis

        assert _extract_hypothesis({"hypothesis": 42}) == ""


class TestExtractVerificationStatus:
    """Lines 467-481: _extract_verification_status with nested rca.verification_result."""

    def test_top_level_verification_result(self) -> None:
        from llmdebug.snapshot_diff import _extract_verification_status

        view = {"verification_result": {"status": "pass"}}
        assert _extract_verification_status(view) == "pass"

    def test_rca_nested_verification_result(self) -> None:
        from llmdebug.snapshot_diff import _extract_verification_status

        view = {"rca": {"verification_result": {"status": "fail"}}}
        assert _extract_verification_status(view) == "fail"

    def test_no_verification_returns_unknown(self) -> None:
        from llmdebug.snapshot_diff import _extract_verification_status

        assert _extract_verification_status({}) == "unknown"

    def test_empty_status_returns_unknown(self) -> None:
        from llmdebug.snapshot_diff import _extract_verification_status

        assert _extract_verification_status({"verification_result": {"status": ""}}) == "unknown"


class TestHasStructuredChanges:
    """Lines 672-685: _has_structured_changes checks."""

    def test_new_frames_in_evidence_delta(self) -> None:
        from llmdebug.snapshot_diff import (
            EvidenceDelta,
            ExceptionDiff,
            LocalsDiff,
            LocationDiff,
            SnapshotDiffResult,
            _has_structured_changes,
        )

        result = SnapshotDiffResult(
            exception=ExceptionDiff(False, "", "", "", "", "", ""),
            location=LocationDiff(False, "", ""),
            locals_diff=LocalsDiff(),
            git_changed=False,
            evidence_delta=EvidenceDelta(new_frames=("frame1",)),
        )
        assert _has_structured_changes(result) is True

    def test_hypothesis_delta_replaced(self) -> None:
        from llmdebug.snapshot_diff import (
            ExceptionDiff,
            LocalsDiff,
            LocationDiff,
            SnapshotDiffResult,
            _has_structured_changes,
        )

        result = SnapshotDiffResult(
            exception=ExceptionDiff(False, "", "", "", "", "", ""),
            location=LocationDiff(False, "", ""),
            locals_diff=LocalsDiff(),
            git_changed=False,
            hypothesis_delta="replaced",
        )
        assert _has_structured_changes(result) is True

    def test_verification_status_changed(self) -> None:
        from llmdebug.snapshot_diff import (
            ExceptionDiff,
            LocalsDiff,
            LocationDiff,
            SnapshotDiffResult,
            VerificationDelta,
            _has_structured_changes,
        )

        result = SnapshotDiffResult(
            exception=ExceptionDiff(False, "", "", "", "", "", ""),
            location=LocationDiff(False, "", ""),
            locals_diff=LocalsDiff(),
            git_changed=False,
            verification_delta=VerificationDelta("unknown", "pass", True),
        )
        assert _has_structured_changes(result) is True

    def test_failure_signature_not_same(self) -> None:
        from llmdebug.snapshot_diff import (
            ExceptionDiff,
            LocalsDiff,
            LocationDiff,
            SnapshotDiffResult,
            _has_structured_changes,
        )

        result = SnapshotDiffResult(
            exception=ExceptionDiff(False, "", "", "", "", "", ""),
            location=LocationDiff(False, "", ""),
            locals_diff=LocalsDiff(),
            git_changed=False,
            failure_signature_same=False,
        )
        assert _has_structured_changes(result) is True


class TestDiffSnapshotsWithHypothesisAndVerification:
    """Integration tests for hypothesis/verification in diff_snapshots."""

    def test_diff_with_added_hypothesis_via_rca(self) -> None:
        old = {
            "exception": {"type": "ValueError", "message": "bad"},
            "frames": [
                {"file_rel": "a.py", "line": 1, "function": "f", "locals": {}, "locals_meta": {}}
            ],
            "crash_frame_index": 0,
        }
        new = {
            **old,
            "rca": {"hypothesis": "New hypothesis about bug"},
        }
        result = diff_snapshots(old, new)
        assert result.hypothesis_delta == "added"

    def test_diff_with_verification_status_change(self) -> None:
        old = {
            "exception": {"type": "ValueError", "message": "bad"},
            "frames": [
                {"file_rel": "a.py", "line": 1, "function": "f", "locals": {}, "locals_meta": {}}
            ],
            "crash_frame_index": 0,
            "rca": {"verification_result": {"status": "fail"}},
        }
        new = {
            "exception": {"type": "ValueError", "message": "bad"},
            "frames": [
                {"file_rel": "a.py", "line": 1, "function": "f", "locals": {}, "locals_meta": {}}
            ],
            "crash_frame_index": 0,
            "rca": {"verification_result": {"status": "pass"}},
        }
        result = diff_snapshots(old, new)
        assert result.verification_delta.status_changed is True
        assert result.verification_delta.old_status == "fail"
        assert result.verification_delta.new_status == "pass"


# ---------------------------------------------------------------------------
# Coverage-gap tests: snapshot_diff.py branches
# ---------------------------------------------------------------------------


def test_diff_touched_dimension_location_changed_same_frames() -> None:
    """Branches [397,399] and [399,401]: location_changed=True with identical frame sets."""
    from llmdebug.snapshot_diff import _diff_touched_dimension

    def get_key(f):
        return str(f.get("file", "?"))

    # Same frames in old and new → symmetric difference is empty
    old_frames: list[Any] = [{"file": "a.py"}, {"file": "b.py"}]
    new_frames: list[Any] = [{"file": "a.py"}, {"file": "b.py"}]
    result = _diff_touched_dimension(
        old_frames,
        new_frames,
        get_key,
        old_cfi=0,
        new_cfi=1,
        location_changed=True,
    )
    # Fallback: crash frames at index 0 (a.py) and 1 (b.py) should be touched
    assert "a.py" in result
    assert "b.py" in result


def test_extract_verification_status_from_rca_nested() -> None:
    """Branch [479,481]: rca.verification_result has a non-empty status."""
    from llmdebug.snapshot_diff import _extract_verification_status

    view: dict[str, Any] = {
        "rca": {
            "verification_result": {
                "status": "pass",
            },
        },
    }
    assert _extract_verification_status(view) == "pass"


def test_fmt_local_change_line_shape_changed_empty_array() -> None:
    """Branches [635,639] and [637,639]: shape_changed with list new_value starting with 0."""
    from llmdebug.snapshot_diff import LocalChange, _fmt_local_change_line

    change = LocalChange(
        name="data",
        kind="shape_changed",
        old_value=[10, 3],
        new_value=[0, 3],
        old_type="ndarray",
        new_type="ndarray",
    )
    line = _fmt_local_change_line(change)
    assert "SHAPE CHANGED" in line
    assert "!! EMPTY" in line


def test_fmt_local_change_line_shape_changed_list_nonzero() -> None:
    """Branch [635,639]: shape_changed with list new_value NOT starting with 0."""
    from llmdebug.snapshot_diff import LocalChange, _fmt_local_change_line

    change = LocalChange(
        name="arr",
        kind="shape_changed",
        old_value=[5, 3],
        new_value=[3, 3],
        old_type="ndarray",
        new_type="ndarray",
    )
    line = _fmt_local_change_line(change)
    assert "SHAPE CHANGED" in line
    assert "!! EMPTY" not in line


def _make_diff_result(
    *,
    message_changed: bool = False,
    touched_files: tuple[str, ...] = (),
    touched_functions: tuple[str, ...] = (),
) -> Any:
    """Helper to build a minimal SnapshotDiffResult for _has_structured_changes tests."""
    from llmdebug.snapshot_diff import (
        ChangeDelta,
        EvidenceDelta,
        ExceptionDiff,
        LocalsDiff,
        LocationDiff,
        SnapshotDiffResult,
        VerificationDelta,
    )

    return SnapshotDiffResult(
        exception=ExceptionDiff(
            changed=False,
            old_type="ValueError",
            new_type="ValueError",
            old_message="x",
            new_message="x",
            old_category="",
            new_category="",
        ),
        location=LocationDiff(
            changed=False, old_location="a.py:1 in f", new_location="a.py:1 in f"
        ),
        locals_diff=LocalsDiff(),
        git_changed=False,
        same_test=True,
        failure_signature_same=True,
        evidence_delta=EvidenceDelta(
            new_frames=(),
            dropped_frames=(),
            message_changed=message_changed,
        ),
        change_delta=ChangeDelta(
            touched_files=touched_files,
            touched_functions=touched_functions,
        ),
        hypothesis_delta="unchanged",
        verification_delta=VerificationDelta(
            status_changed=False,
            old_status="unknown",
            new_status="unknown",
        ),
    )


def test_has_structured_changes_message_changed() -> None:
    """Branch [675,676]: message_changed=True triggers early return True."""
    from llmdebug.snapshot_diff import _has_structured_changes

    result = _make_diff_result(message_changed=True)
    assert _has_structured_changes(result) is True


def test_has_structured_changes_touched_files() -> None:
    """Branch [677,678]: touched_files non-empty triggers early return True."""
    from llmdebug.snapshot_diff import _has_structured_changes

    result = _make_diff_result(touched_files=("a.py",))
    assert _has_structured_changes(result) is True


def test_has_structured_changes_touched_functions() -> None:
    """Branch [677,678]: touched_functions non-empty triggers early return True."""
    from llmdebug.snapshot_diff import _has_structured_changes

    result = _make_diff_result(touched_functions=("do_stuff",))
    assert _has_structured_changes(result) is True


def test_mutating_a_result_does_not_leak_into_later_diffs() -> None:
    """CAP-14: old_git/new_git were a shared module-level dict when a snapshot had no git."""
    from llmdebug.snapshot_diff import diff_snapshots

    snap = {"exception": {"type": "E", "message": "m"}, "frames": []}
    first = diff_snapshots(snap, snap)
    first.old_git["commit"] = "poisoned"
    first.new_git["branch"] = "poisoned"

    second = diff_snapshots(snap, snap)

    assert second.old_git == {}
    assert second.new_git == {}
    assert second.old_git is not first.old_git
