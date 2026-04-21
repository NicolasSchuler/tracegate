"""Tests for compact output formats."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from llmdebug import SnapshotConfig, debug_snapshot, get_latest_snapshot
from llmdebug import _json as json
from llmdebug.compact_keys import (
    KEY_MAP,
    REVERSE_KEY_MAP,
    compact_keys,
    expand_keys,
    is_compact_snapshot,
)
from llmdebug.config import OutputFormat
from llmdebug.output import write_bundle


# Check if toons is available
def _has_toons() -> bool:
    try:
        import toons  # pyright: ignore[reportMissingImports]

        return True
    except ImportError:
        return False


class TestCompactKeys:
    """Tests for key compaction/expansion."""

    def test_key_map_is_bijective(self):
        """Verify KEY_MAP has no duplicate values."""
        assert len(KEY_MAP) == len(set(KEY_MAP.values())), "KEY_MAP has duplicate short keys"

    def test_reverse_map_matches(self):
        """Verify REVERSE_KEY_MAP is correct inverse."""
        for full, short in KEY_MAP.items():
            assert REVERSE_KEY_MAP[short] == full

    def test_compact_keys_simple_dict(self):
        """Test key compaction on simple dict."""
        data = {"schema_version": 1, "timestamp_utc": "2024-01-01"}
        result = compact_keys(data)
        assert result == {"_v": 1, "_ts": "2024-01-01"}

    def test_compact_keys_nested_dict(self):
        """Test key compaction on nested dict."""
        data = {
            "exception": {
                "type": "ValueError",
                "message": "test error",
            }
        }
        result = compact_keys(data)
        assert result == {
            "_exc": {
                "_t": "ValueError",
                "_msg": "test error",
            }
        }

    def test_compact_keys_with_list(self):
        """Test key compaction on dict containing list."""
        data = {
            "frames": [
                {"file": "/test.py", "line": 10},
                {"file": "/other.py", "line": 20},
            ]
        }
        result = compact_keys(data)
        assert result == {
            "_fr": [
                {"_f": "/test.py", "_l": 10},
                {"_f": "/other.py", "_l": 20},
            ]
        }

    def test_compact_keys_preserves_unknown_keys(self):
        """Test that unknown keys are preserved as-is."""
        data = {"unknown_key": "value", "type": "test"}
        result = compact_keys(data)
        assert result == {"unknown_key": "value", "_t": "test"}

    def test_expand_keys_reverses_compact(self):
        """Test that expand_keys reverses compact_keys."""
        original = {
            "schema_version": 1,
            "frames": [
                {"file": "/test.py", "locals": {"x": 1}},
            ],
            "exception": {"type": "Error", "message": "msg"},
        }
        compacted = compact_keys(original)
        expanded = expand_keys(compacted)
        assert expanded == original

    def test_is_compact_snapshot_detects_compact(self):
        """Test compact snapshot detection."""
        compact = {"_v": 1, "_fr": [], "_exc": {}, "_ts": "2024"}
        assert is_compact_snapshot(compact) is True

    def test_is_compact_snapshot_detects_full(self):
        """Test full snapshot detection."""
        full = {"schema_version": 1, "frames": [], "exception": {}}
        assert is_compact_snapshot(full) is False

    def test_is_compact_snapshot_detects_marker(self):
        """Test explicit compact marker detection."""
        assert is_compact_snapshot({"__llmdebug_compact__": 1}) is True

    def test_is_compact_snapshot_ignores_short_key_false_positive(self):
        """Single short-like keys in normal JSON should not imply compact mode."""
        assert is_compact_snapshot({"_v": "user_value"}) is False

    def test_compact_keys_preserves_user_data(self):
        """Test that user variable names in locals/args are not transformed."""
        data: dict[str, list[dict[str, object]]] = {
            "frames": [
                {
                    "file": "/test.py",
                    "locals": {"a": 1, "type": "test", "message": "hello"},
                    "args": {"x": 10, "type": "arg_type"},
                }
            ]
        }
        result = compact_keys(data)
        assert isinstance(result, dict)
        # Frame keys should be compacted
        assert "_fr" in result
        frames = result["_fr"]
        assert isinstance(frames, list) and len(frames) > 0
        frame = frames[0]
        assert isinstance(frame, dict)
        assert "_f" in frame
        assert "_loc" in frame
        # But locals content should preserve user variable names
        assert frame["_loc"] == {"a": 1, "type": "test", "message": "hello"}
        assert frame["_args"] == {"x": 10, "type": "arg_type"}

    def test_expand_keys_preserves_user_data(self):
        """Test that expansion preserves user variable names."""
        data: dict[str, list[dict[str, object]]] = {
            "_fr": [
                {
                    "_f": "/test.py",
                    "_loc": {"a": 1, "_t": "should_stay", "type": "user_var"},
                    "_args": {"b": 2, "_msg": "also_stay"},
                }
            ]
        }
        result = expand_keys(data)
        assert isinstance(result, dict)
        # Frame keys should be expanded
        assert "frames" in result
        frames = result["frames"]
        assert isinstance(frames, list) and len(frames) > 0
        frame = frames[0]
        assert isinstance(frame, dict)
        assert "file" in frame
        assert "locals" in frame
        # But locals content should be unchanged (user data)
        assert frame["locals"] == {"a": 1, "_t": "should_stay", "type": "user_var"}
        assert frame["args"] == {"b": 2, "_msg": "also_stay"}

    def test_enhanced_pytest_context_roundtrip(self):
        """Test compact/expand round-trip for all 6 enhanced pytest context types."""
        original = {
            "schema_version": 1,
            "frames": [],
            "exception": {"type": "AssertionError", "message": "test"},
            "pytest": {
                "nodeid": "test_it.py::test_example",
                "outcome": "failed",
                "when": "call",
                "fixtures": {
                    "data": "[1, 2, 3]",
                    "db_conn": "<Connection host='localhost'>",
                },
                "markers": {
                    "slow": {},
                    "skipif": {"args": ["sys.platform == 'win32'"]},
                },
                "timeout_config": {
                    "seconds": 30,
                    "kwargs": {"method": "'signal'"},
                },
                "param_indices": {"x": 1, "y": 0},
                "xdist_worker": "gw0",
                "rerun_count": 3,
                "flaky": True,
                "params": {"x": "20"},
                "repro": ["python", "-m", "pytest", "test_it.py::test_example", "-q"],
            },
        }
        compacted = compact_keys(original)
        expanded = expand_keys(compacted)
        assert expanded == original

    def test_compact_keys_preserves_locals_meta(self):
        """Test that locals_meta keys are treated as user data."""
        data = {
            "frames": [
                {
                    "file": "/test.py",
                    "locals_meta": {
                        # Variable name "type" should not be compacted to "_t"
                        "type": {"type": "builtins.str"},
                        "message": {"type": "builtins.str"},
                    },
                }
            ]
        }
        compacted = compact_keys(data)
        expanded = expand_keys(compacted)
        assert expanded == data


class TestCompactJSONOutput:
    """Tests for compact JSON output format."""

    def test_compact_json_is_valid_json(self, tmp_path: Path):
        """Test that compact JSON output is valid JSON."""
        cfg = SnapshotConfig(out_dir=str(tmp_path), output_format="json_compact")
        payload = {
            "name": "test",
            "schema_version": 1,
            "frames": [{"file": "/test.py", "line": 1}],
            "exception": {"type": "Error"},
        }
        path = write_bundle(payload, cfg)
        content = path.read_text()
        # Should parse as valid JSON
        parsed = json.loads(content)
        assert "_v" in parsed  # Uses short keys with underscore prefix

    def test_compact_json_smaller_than_pretty(self, tmp_path: Path):
        """Test that compact JSON is smaller than pretty JSON."""
        payload = {
            "name": "test",
            "schema_version": 1,
            "timestamp_utc": "2024-01-01T00:00:00Z",
            "frames": [{"file": "/test.py", "line": i, "function": f"func_{i}"} for i in range(5)],
            "exception": {"type": "ValueError", "message": "test error"},
        }

        # Write pretty JSON
        pretty_cfg = SnapshotConfig(out_dir=str(tmp_path / "pretty"), output_format="json")
        pretty_path = write_bundle(payload, pretty_cfg)
        pretty_size = pretty_path.stat().st_size

        # Write compact JSON
        compact_dir = str(tmp_path / "compact")
        compact_cfg = SnapshotConfig(out_dir=compact_dir, output_format="json_compact")
        compact_path = write_bundle(payload, compact_cfg)
        compact_size = compact_path.stat().st_size

        # Compact should be at least 20% smaller
        assert compact_size < pretty_size * 0.8, (
            f"Compact ({compact_size}) should be <80% of pretty ({pretty_size})"
        )

    def test_get_latest_auto_expands_compact(self, tmp_path: Path):
        """Test that get_latest_snapshot auto-expands compact keys."""
        cfg = SnapshotConfig(out_dir=str(tmp_path), output_format="json_compact")
        payload = {
            "name": "test",
            "schema_version": 1,
            "frames": [{"file": "/test.py", "line": 1}],
            "exception": {"type": "Error", "message": "test"},
        }
        write_bundle(payload, cfg)

        # Read back should have full keys
        snapshot = get_latest_snapshot(str(tmp_path))
        assert snapshot is not None
        assert "schema_version" in snapshot
        assert "frames" in snapshot
        assert "exception" in snapshot
        assert snapshot["frames"][0]["file"] == "/test.py"
        assert "__llmdebug_compact__" not in snapshot

    def test_get_latest_does_not_expand_plain_json_with_short_keys(self, tmp_path: Path):
        """Plain JSON with user key names like _v should remain unchanged."""
        cfg = SnapshotConfig(out_dir=str(tmp_path), output_format="json")
        payload = {
            "name": "test",
            "schema_version": 1,
            "frames": [],
            "exception": {},
            "_v": "user_value",
        }
        write_bundle(payload, cfg)

        snapshot = get_latest_snapshot(str(tmp_path))
        assert snapshot is not None
        assert snapshot["schema_version"] == 1
        assert snapshot["_v"] == "user_value"

    def test_output_format_validation(self):
        """Test that invalid output_format raises ValueError."""
        with pytest.raises(ValueError, match="output_format must be"):
            SnapshotConfig(output_format=cast(OutputFormat, "invalid"))


class TestDecoratorWithCompactFormat:
    """Test decorator with compact output format."""

    def test_debug_snapshot_with_compact_format(self, tmp_path: Path):
        """Test @debug_snapshot with json_compact format."""

        @debug_snapshot(out_dir=str(tmp_path), output_format="json_compact")
        def failing_func():
            raise ValueError("test error")

        with pytest.raises(ValueError):
            failing_func()

        # Verify snapshot exists and is compact
        snapshot = get_latest_snapshot(str(tmp_path))
        assert snapshot is not None
        assert snapshot["exception"]["type"] == "ValueError"

    def test_debug_snapshot_default_is_compact(self, tmp_path: Path):
        """Test that default output format is json_compact."""

        @debug_snapshot(out_dir=str(tmp_path))
        def failing_func():
            raise ValueError("test")

        with pytest.raises(ValueError):
            failing_func()

        # Read raw file to verify it's compact
        latest = tmp_path / "latest.json"
        raw = json.loads(latest.read_text())
        assert "_v" in raw or "_exc" in raw  # Uses short keys with underscore prefix


@pytest.mark.skipif(not _has_toons(), reason="toons package not installed")
class TestTOONFormat:
    """Tests for TOON output format (requires toons package)."""

    def test_toon_format_writes_toon_file(self, tmp_path: Path):
        """Test that TOON format writes .toon file."""
        cfg = SnapshotConfig(out_dir=str(tmp_path), output_format="toon")
        payload = {
            "name": "test",
            "schema_version": 1,
            "frames": [{"file": "/test.py", "line": 1}],
        }
        path = write_bundle(payload, cfg)
        assert path.suffix == ".toon"
        assert (tmp_path / "latest.toon").exists()

    def test_toon_format_roundtrip(self, tmp_path: Path):
        """Test TOON encoding/decoding roundtrip."""
        cfg = SnapshotConfig(out_dir=str(tmp_path), output_format="toon")
        payload = {
            "name": "test",
            "schema_version": 1,
            "frames": [{"file": "/test.py", "line": i, "function": f"f{i}"} for i in range(3)],
            "exception": {"type": "Error", "message": "test"},
        }
        write_bundle(payload, cfg)

        snapshot = get_latest_snapshot(str(tmp_path))
        assert snapshot is not None
        assert snapshot["schema_version"] == 1
        assert len(snapshot["frames"]) == 3

    def test_get_latest_prefers_newest_when_both_exist(self, tmp_path: Path):
        """If both latest.json and latest.toon exist, prefer newest by mtime."""
        import os

        # Write a TOON snapshot (this creates latest.toon).
        toon_cfg = SnapshotConfig(out_dir=str(tmp_path), output_format="toon")
        payload = {"name": "toon_wins", "schema_version": 1, "frames": []}
        write_bundle(payload, toon_cfg)

        latest_toon = tmp_path / "latest.toon"
        assert latest_toon.exists()

        # Create a stale latest.json with older mtime.
        (tmp_path / "latest.json").write_text(json.dumps({"name": "stale_json"}))
        old = latest_toon.stat().st_mtime - 10
        os.utime(tmp_path / "latest.json", (old, old))

        snapshot = get_latest_snapshot(str(tmp_path))
        assert snapshot is not None
        assert snapshot["name"] == "toon_wins"

    def test_toon_smaller_than_json(self, tmp_path: Path):
        """Test TOON format is smaller than pretty JSON."""
        payload = {
            "name": "test",
            "schema_version": 1,
            "frames": [
                {"file": f"/path/to/file{i}.py", "line": i, "function": f"function_{i}"}
                for i in range(10)
            ],
            "exception": {"type": "ValueError", "message": "test error message"},
        }

        # Write pretty JSON
        json_cfg = SnapshotConfig(out_dir=str(tmp_path / "json"), output_format="json")
        json_path = write_bundle(payload, json_cfg)
        json_size = json_path.stat().st_size

        # Write TOON
        toon_cfg = SnapshotConfig(out_dir=str(tmp_path / "toon"), output_format="toon")
        toon_path = write_bundle(payload, toon_cfg)
        toon_size = toon_path.stat().st_size

        # TOON should be smaller (allow for small payloads where overhead matters)
        # For larger payloads, savings should be more significant
        assert toon_size <= json_size, f"TOON ({toon_size}) should be <= JSON ({json_size})"


class TestMixedFormatCleanup:
    """Test snapshot cleanup works with mixed formats."""

    def test_cleanup_handles_mixed_formats(self, tmp_path: Path):
        """Test cleanup considers both .json and .toon files."""
        # Create some JSON files manually
        for i in range(3):
            (tmp_path / f"2024010{i}T000000Z_test.json").write_text("{}")

        # Write one more with low max_snapshots
        cfg = SnapshotConfig(
            out_dir=str(tmp_path),
            output_format="json_compact",
            max_snapshots=2,
        )
        payload = {"name": "test", "schema_version": 1}
        write_bundle(payload, cfg)

        # Should only have 2 snapshots left (plus latest.json)
        json_files = [f for f in tmp_path.glob("*.json") if not f.name.startswith("latest")]
        assert len(json_files) == 2
