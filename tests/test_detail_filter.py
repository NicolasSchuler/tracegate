"""Tests for detail_filter — layered snapshot disclosure."""

from __future__ import annotations

import copy
from typing import Any

import pytest

from llmdebug.detail_filter import (
    _LAYER_CRASH_KEYS,
    _LAYER_FULL_KEYS,
    filter_snapshot,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def expanded_snapshot() -> dict[str, Any]:
    """Realistic expanded-key snapshot with multiple frames and all context keys."""
    return {
        "schema_version": "1.0",
        "name": "test_example",
        "timestamp_utc": "2026-02-08T12:00:00Z",
        "llmdebug_version": "2.0.0",
        "exception": {
            "type": "IndexError",
            "qualified_type": "builtins.IndexError",
            "message": "list index out of range",
            "error_category": {"category": "index_error", "suggestion": "Check bounds"},
        },
        "crash_frame_index": 1,
        "frames": [
            {
                "file": "/code/lib/utils.py",
                "file_rel": "lib/utils.py",
                "line": 10,
                "function": "helper",
                "locals": {"items": [1, 2, 3]},
                "locals_meta": {"items": {"type": "builtins.list"}},
            },
            {
                "file": "/code/app.py",
                "file_rel": "app.py",
                "line": 55,
                "function": "process",
                "locals": {"idx": 10, "data": [1, 2]},
                "locals_meta": {
                    "idx": {"type": "builtins.int"},
                    "data": {"type": "builtins.list"},
                },
            },
            {
                "file": "/code/main.py",
                "file_rel": "main.py",
                "line": 8,
                "function": "run",
                "locals": {"config": {"debug": True}},
                "locals_meta": {"config": {"type": "builtins.dict"}},
            },
        ],
        "traceback": (
            "Traceback (most recent call last):\n"
            '  File "/code/app.py", line 55, in process\n'
            "    data[idx]\n"
            "IndexError: list index out of range"
        ),
        "capture_config": {"max_string_len": 200, "max_depth": 4},
        "_capture_warnings": [],
        "env": {
            "python_version": "3.11.5",
            "platform": "Linux-6.1.0-x86_64",
            "cwd": "/code",
        },
        "git": {"commit": "abc1234", "branch": "feat/detail", "dirty": False},
        "pytest": {
            "nodeid": "tests/test_app.py::test_process",
            "outcome": "failed",
        },
        "extra_context": {"gpu_memory": "8GB"},
    }


# ---------------------------------------------------------------------------
# TestCrashLayer
# ---------------------------------------------------------------------------


class TestCrashLayer:
    """Tests for detail='crash' — minimal layer."""

    def test_includes_only_crash_keys(self, expanded_snapshot: dict[str, Any]) -> None:
        result = filter_snapshot(expanded_snapshot, "crash")
        # crash_frame_index is overwritten to 0 when frames are trimmed,
        # so just check all keys are within the allowed set.
        assert set(result.keys()) <= _LAYER_CRASH_KEYS

    def test_has_exactly_one_frame(self, expanded_snapshot: dict[str, Any]) -> None:
        result = filter_snapshot(expanded_snapshot, "crash")
        assert len(result["frames"]) == 1

    def test_selects_crash_frame_by_index(self, expanded_snapshot: dict[str, Any]) -> None:
        result = filter_snapshot(expanded_snapshot, "crash")
        # crash_frame_index was 1, so the selected frame should be frames[1]
        assert result["frames"][0]["function"] == "process"
        assert result["frames"][0]["line"] == 55

    def test_crash_frame_index_reset_to_zero(self, expanded_snapshot: dict[str, Any]) -> None:
        result = filter_snapshot(expanded_snapshot, "crash")
        assert result["crash_frame_index"] == 0

    def test_excludes_traceback(self, expanded_snapshot: dict[str, Any]) -> None:
        result = filter_snapshot(expanded_snapshot, "crash")
        assert "traceback" not in result

    def test_excludes_env(self, expanded_snapshot: dict[str, Any]) -> None:
        result = filter_snapshot(expanded_snapshot, "crash")
        assert "env" not in result

    def test_excludes_git(self, expanded_snapshot: dict[str, Any]) -> None:
        result = filter_snapshot(expanded_snapshot, "crash")
        assert "git" not in result

    def test_excludes_pytest(self, expanded_snapshot: dict[str, Any]) -> None:
        result = filter_snapshot(expanded_snapshot, "crash")
        assert "pytest" not in result

    def test_excludes_extra_context(self, expanded_snapshot: dict[str, Any]) -> None:
        result = filter_snapshot(expanded_snapshot, "crash")
        assert "extra_context" not in result

    def test_excludes_capture_config(self, expanded_snapshot: dict[str, Any]) -> None:
        result = filter_snapshot(expanded_snapshot, "crash")
        assert "capture_config" not in result

    def test_includes_exception(self, expanded_snapshot: dict[str, Any]) -> None:
        result = filter_snapshot(expanded_snapshot, "crash")
        assert result["exception"]["type"] == "IndexError"

    def test_includes_schema_version(self, expanded_snapshot: dict[str, Any]) -> None:
        result = filter_snapshot(expanded_snapshot, "crash")
        assert result["schema_version"] == "1.0"

    def test_includes_name(self, expanded_snapshot: dict[str, Any]) -> None:
        result = filter_snapshot(expanded_snapshot, "crash")
        assert result["name"] == "test_example"

    def test_includes_timestamp(self, expanded_snapshot: dict[str, Any]) -> None:
        result = filter_snapshot(expanded_snapshot, "crash")
        assert result["timestamp_utc"] == "2026-02-08T12:00:00Z"

    def test_includes_llmdebug_version(self, expanded_snapshot: dict[str, Any]) -> None:
        result = filter_snapshot(expanded_snapshot, "crash")
        assert result["llmdebug_version"] == "2.0.0"

    def test_is_default_detail_level(self, expanded_snapshot: dict[str, Any]) -> None:
        default = filter_snapshot(expanded_snapshot)
        explicit = filter_snapshot(expanded_snapshot, "crash")
        assert default == explicit


# ---------------------------------------------------------------------------
# TestFullLayer
# ---------------------------------------------------------------------------


class TestFullLayer:
    """Tests for detail='full' — all frames + traceback, no env/git/pytest."""

    def test_includes_only_full_keys(self, expanded_snapshot: dict[str, Any]) -> None:
        result = filter_snapshot(expanded_snapshot, "full")
        assert set(result.keys()) <= _LAYER_FULL_KEYS

    def test_has_all_frames(self, expanded_snapshot: dict[str, Any]) -> None:
        result = filter_snapshot(expanded_snapshot, "full")
        assert len(result["frames"]) == 3

    def test_preserves_crash_frame_index(self, expanded_snapshot: dict[str, Any]) -> None:
        result = filter_snapshot(expanded_snapshot, "full")
        assert result["crash_frame_index"] == 1

    def test_includes_traceback(self, expanded_snapshot: dict[str, Any]) -> None:
        result = filter_snapshot(expanded_snapshot, "full")
        assert "traceback" in result
        assert "IndexError" in result["traceback"]

    def test_includes_capture_config(self, expanded_snapshot: dict[str, Any]) -> None:
        result = filter_snapshot(expanded_snapshot, "full")
        assert "capture_config" in result

    def test_includes_capture_warnings(self, expanded_snapshot: dict[str, Any]) -> None:
        result = filter_snapshot(expanded_snapshot, "full")
        assert "_capture_warnings" in result

    def test_excludes_env(self, expanded_snapshot: dict[str, Any]) -> None:
        result = filter_snapshot(expanded_snapshot, "full")
        assert "env" not in result

    def test_excludes_git(self, expanded_snapshot: dict[str, Any]) -> None:
        result = filter_snapshot(expanded_snapshot, "full")
        assert "git" not in result

    def test_excludes_pytest(self, expanded_snapshot: dict[str, Any]) -> None:
        result = filter_snapshot(expanded_snapshot, "full")
        assert "pytest" not in result

    def test_excludes_extra_context(self, expanded_snapshot: dict[str, Any]) -> None:
        result = filter_snapshot(expanded_snapshot, "full")
        assert "extra_context" not in result


# ---------------------------------------------------------------------------
# TestContextLayer
# ---------------------------------------------------------------------------


class TestContextLayer:
    """Tests for detail='context' — everything, shallow copy."""

    def test_includes_all_keys(self, expanded_snapshot: dict[str, Any]) -> None:
        result = filter_snapshot(expanded_snapshot, "context")
        assert set(result.keys()) == set(expanded_snapshot.keys())

    def test_includes_env(self, expanded_snapshot: dict[str, Any]) -> None:
        result = filter_snapshot(expanded_snapshot, "context")
        assert result["env"]["python_version"] == "3.11.5"

    def test_includes_git(self, expanded_snapshot: dict[str, Any]) -> None:
        result = filter_snapshot(expanded_snapshot, "context")
        assert result["git"]["commit"] == "abc1234"

    def test_includes_pytest(self, expanded_snapshot: dict[str, Any]) -> None:
        result = filter_snapshot(expanded_snapshot, "context")
        assert result["pytest"]["nodeid"] == "tests/test_app.py::test_process"

    def test_includes_extra_context(self, expanded_snapshot: dict[str, Any]) -> None:
        result = filter_snapshot(expanded_snapshot, "context")
        assert result["extra_context"]["gpu_memory"] == "8GB"

    def test_includes_all_frames(self, expanded_snapshot: dict[str, Any]) -> None:
        result = filter_snapshot(expanded_snapshot, "context")
        assert len(result["frames"]) == 3

    def test_is_shallow_copy(self, expanded_snapshot: dict[str, Any]) -> None:
        result = filter_snapshot(expanded_snapshot, "context")
        # It is a different dict object
        assert result is not expanded_snapshot
        # But inner mutable objects are shared (shallow copy)
        assert result["frames"] is expanded_snapshot["frames"]
        assert result["exception"] is expanded_snapshot["exception"]


# ---------------------------------------------------------------------------
# TestNoMutation
# ---------------------------------------------------------------------------


class TestNoMutation:
    """Filtering must never mutate the original snapshot."""

    def test_crash_does_not_mutate_original(self, expanded_snapshot: dict[str, Any]) -> None:
        original = copy.deepcopy(expanded_snapshot)
        filter_snapshot(expanded_snapshot, "crash")
        assert expanded_snapshot == original

    def test_full_does_not_mutate_original(self, expanded_snapshot: dict[str, Any]) -> None:
        original = copy.deepcopy(expanded_snapshot)
        filter_snapshot(expanded_snapshot, "full")
        assert expanded_snapshot == original

    def test_context_does_not_mutate_original(self, expanded_snapshot: dict[str, Any]) -> None:
        original = copy.deepcopy(expanded_snapshot)
        filter_snapshot(expanded_snapshot, "context")
        assert expanded_snapshot == original

    def test_crash_frame_is_deep_copied(self, expanded_snapshot: dict[str, Any]) -> None:
        result = filter_snapshot(expanded_snapshot, "crash")
        # Mutating the crash result frame should not affect the original
        result["frames"][0]["function"] = "MUTATED"
        assert expanded_snapshot["frames"][1]["function"] == "process"

    def test_original_frames_list_untouched(self, expanded_snapshot: dict[str, Any]) -> None:
        filter_snapshot(expanded_snapshot, "crash")
        assert len(expanded_snapshot["frames"]) == 3


# ---------------------------------------------------------------------------
# TestCompactSnapshotRejection
# ---------------------------------------------------------------------------


class TestCompactSnapshotRejection:
    """Compact-key snapshots must raise ValueError."""

    def test_rejects_compact_marker(self) -> None:
        snap: dict[str, Any] = {"__llmdebug_compact__": True, "schema_version": "1.0"}
        with pytest.raises(ValueError, match="expanded-key"):
            filter_snapshot(snap, "crash")

    def test_rejects_compact_exc_key(self) -> None:
        snap: dict[str, Any] = {"_exc": {"type": "E"}, "schema_version": "1.0"}
        with pytest.raises(ValueError, match="expanded-key"):
            filter_snapshot(snap, "full")

    def test_rejects_compact_fr_key(self) -> None:
        snap: dict[str, Any] = {"_fr": [], "schema_version": "1.0"}
        with pytest.raises(ValueError, match="expanded-key"):
            filter_snapshot(snap, "context")

    def test_rejects_compact_with_multiple_markers(self) -> None:
        snap: dict[str, Any] = {
            "__llmdebug_compact__": True,
            "_exc": {"type": "E"},
            "_fr": [],
        }
        with pytest.raises(ValueError, match="expanded-key"):
            filter_snapshot(snap)


# ---------------------------------------------------------------------------
# TestEmptyFrames
# ---------------------------------------------------------------------------


class TestEmptyFrames:
    """Handle snapshots with an empty frames list."""

    @pytest.fixture
    def no_frames_snapshot(self) -> dict[str, Any]:
        return {
            "schema_version": "1.0",
            "name": "test_no_frames",
            "timestamp_utc": "2026-02-08T12:00:00Z",
            "llmdebug_version": "2.0.0",
            "exception": {
                "type": "RuntimeError",
                "message": "something broke",
            },
            "crash_frame_index": 0,
            "frames": [],
        }

    def test_crash_with_empty_frames(self, no_frames_snapshot: dict[str, Any]) -> None:
        result = filter_snapshot(no_frames_snapshot, "crash")
        # frames key should not be present or should be empty
        # The implementation only sets frames when the list is truthy
        assert result.get("frames", []) == []

    def test_full_with_empty_frames(self, no_frames_snapshot: dict[str, Any]) -> None:
        result = filter_snapshot(no_frames_snapshot, "full")
        assert result["frames"] == []

    def test_context_with_empty_frames(self, no_frames_snapshot: dict[str, Any]) -> None:
        result = filter_snapshot(no_frames_snapshot, "context")
        assert result["frames"] == []


# ---------------------------------------------------------------------------
# TestMissingOptionalKeys
# ---------------------------------------------------------------------------


class TestMissingOptionalKeys:
    """Gracefully handle snapshots missing optional keys."""

    @pytest.fixture
    def minimal_snapshot(self) -> dict[str, Any]:
        """Snapshot with only the bare essentials — no git, env, pytest, etc."""
        return {
            "schema_version": "1.0",
            "name": "test_minimal",
            "timestamp_utc": "2026-02-08T12:00:00Z",
            "llmdebug_version": "2.0.0",
            "exception": {"type": "ValueError", "message": "oops"},
            "frames": [
                {
                    "file": "/code/x.py",
                    "file_rel": "x.py",
                    "line": 1,
                    "function": "f",
                    "locals": {},
                    "locals_meta": {},
                },
            ],
            "crash_frame_index": 0,
        }

    def test_crash_without_optional_keys(self, minimal_snapshot: dict[str, Any]) -> None:
        result = filter_snapshot(minimal_snapshot, "crash")
        assert result["exception"]["type"] == "ValueError"
        assert len(result["frames"]) == 1

    def test_full_without_traceback(self, minimal_snapshot: dict[str, Any]) -> None:
        result = filter_snapshot(minimal_snapshot, "full")
        assert "traceback" not in result
        assert "capture_config" not in result

    def test_context_without_git_env_pytest(self, minimal_snapshot: dict[str, Any]) -> None:
        result = filter_snapshot(minimal_snapshot, "context")
        assert "git" not in result
        assert "env" not in result
        assert "pytest" not in result

    def test_crash_missing_frames_key(self) -> None:
        snap: dict[str, Any] = {
            "schema_version": "1.0",
            "name": "test_no_frames_key",
            "timestamp_utc": "2026-02-08T12:00:00Z",
            "llmdebug_version": "2.0.0",
            "exception": {"type": "ValueError", "message": "oops"},
        }
        result = filter_snapshot(snap, "crash")
        # No frames key at all — should not crash
        assert "frames" not in result or result["frames"] == []

    def test_crash_missing_crash_frame_index(self) -> None:
        snap: dict[str, Any] = {
            "schema_version": "1.0",
            "name": "test_no_idx",
            "timestamp_utc": "2026-02-08T12:00:00Z",
            "llmdebug_version": "2.0.0",
            "exception": {"type": "ValueError", "message": "oops"},
            "frames": [
                {"file_rel": "a.py", "line": 1, "function": "f", "locals": {}},
            ],
        }
        result = filter_snapshot(snap, "crash")
        # Defaults to index 0
        assert len(result["frames"]) == 1
        assert result["crash_frame_index"] == 0


# ---------------------------------------------------------------------------
# TestCrashFrameIndexOutOfRange
# ---------------------------------------------------------------------------


class TestCrashFrameIndexOutOfRange:
    """Out-of-range crash_frame_index falls back to the first frame."""

    def test_index_too_high(self, expanded_snapshot: dict[str, Any]) -> None:
        expanded_snapshot["crash_frame_index"] = 99
        result = filter_snapshot(expanded_snapshot, "crash")
        # Falls back to frames[0]
        assert len(result["frames"]) == 1
        assert result["frames"][0]["function"] == "helper"
        assert result["crash_frame_index"] == 0

    def test_negative_index(self, expanded_snapshot: dict[str, Any]) -> None:
        expanded_snapshot["crash_frame_index"] = -1
        result = filter_snapshot(expanded_snapshot, "crash")
        # Negative is out of the 0 <= idx < len check, falls back to frames[0]
        assert len(result["frames"]) == 1
        assert result["frames"][0]["function"] == "helper"
        assert result["crash_frame_index"] == 0

    def test_index_equal_to_length(self, expanded_snapshot: dict[str, Any]) -> None:
        # Exactly len(frames) is out of range (off-by-one boundary)
        expanded_snapshot["crash_frame_index"] = 3  # len is 3
        result = filter_snapshot(expanded_snapshot, "crash")
        assert result["frames"][0]["function"] == "helper"
        assert result["crash_frame_index"] == 0

    def test_valid_boundary_index(self, expanded_snapshot: dict[str, Any]) -> None:
        # Last valid index (len-1) should work correctly
        expanded_snapshot["crash_frame_index"] = 2
        result = filter_snapshot(expanded_snapshot, "crash")
        assert result["frames"][0]["function"] == "run"
        assert result["crash_frame_index"] == 0

    def test_malformed_index_falls_back_to_first_dict_frame(self) -> None:
        snap: dict[str, Any] = {
            "schema_version": "1.0",
            "name": "test_malformed_idx",
            "timestamp_utc": "2026-02-08T12:00:00Z",
            "llmdebug_version": "2.0.0",
            "exception": {"type": "ValueError", "message": "oops"},
            "crash_frame_index": "bad-index",
            "frames": [
                "bad-frame",
                {"file_rel": "x.py", "line": 1, "function": "f", "locals": {}},
            ],
        }
        result = filter_snapshot(snap, "crash")
        assert result["frames"][0]["function"] == "f"
        assert result["crash_frame_index"] == 0


# ---------------------------------------------------------------------------
# Coverage gap tests — lines 33, 42 (_LAYER_FULL_KEYS and filter_snapshot def)
# ---------------------------------------------------------------------------


class TestFullLayerKeysDefinition:
    """Ensure _LAYER_FULL_KEYS includes the expected superset of crash keys."""

    def test_full_keys_include_traceback_and_capture_config(self) -> None:
        """Line 33: _LAYER_FULL_KEYS definition is exercised."""
        assert "traceback" in _LAYER_FULL_KEYS
        assert "capture_config" in _LAYER_FULL_KEYS
        assert "_capture_warnings" in _LAYER_FULL_KEYS

    def test_full_keys_superset_of_crash_keys(self) -> None:
        assert _LAYER_CRASH_KEYS < _LAYER_FULL_KEYS

    def test_filter_snapshot_full_returns_traceback(self) -> None:
        """Line 42/66-71: filter_snapshot with detail='full' iterates _LAYER_FULL_KEYS."""
        snap: dict[str, Any] = {
            "schema_version": "1.0",
            "name": "full_test",
            "timestamp_utc": "2026-01-01T00:00:00Z",
            "llmdebug_version": "2.0.0",
            "exception": {"type": "RuntimeError", "message": "boom"},
            "crash_frame_index": 0,
            "frames": [
                {"file_rel": "x.py", "line": 1, "function": "f", "locals": {}},
            ],
            "traceback": "Traceback ...",
            "capture_config": {"max_depth": 4},
            "_capture_warnings": ["warn1"],
            # These should NOT appear in full output:
            "env": {"python_version": "3.12"},
            "git": {"commit": "abc"},
        }
        result = filter_snapshot(snap, "full")
        assert result.get("traceback") == "Traceback ..."
        assert result.get("capture_config") == {"max_depth": 4}
        assert result.get("_capture_warnings") == ["warn1"]
        assert "env" not in result
        assert "git" not in result
