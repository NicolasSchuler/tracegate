"""Tests for mcp_diff_tools handle_diff handler."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from llmdebug import _json as json
from llmdebug.config import SnapshotConfig
from llmdebug.mcp_diff_tools import handle_diff
from llmdebug.output import write_bundle

# ---------------------------------------------------------------------------
# Helper: write a minimal snapshot
# ---------------------------------------------------------------------------


def _make_snapshot(name: str = "test_snap", exc_type: str = "ValueError") -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "name": name,
        "timestamp_utc": "2026-01-01T00:00:00Z",
        "exception": {"type": exc_type, "message": "oops"},
        "frames": [
            {
                "file": "/code/app.py",
                "file_rel": "app.py",
                "line": 10,
                "function": "run",
                "locals": {},
                "locals_meta": {},
            }
        ],
        "crash_frame_index": 0,
    }


def _write_snapshot(out_dir: str, snap: dict[str, Any] | None = None) -> None:
    cfg = SnapshotConfig(out_dir=out_dir, output_format="json")
    write_bundle(snap or _make_snapshot(), cfg)


# ---------------------------------------------------------------------------
# Validation errors
# ---------------------------------------------------------------------------


def test_handle_diff_invalid_response_format(tmp_path: Path) -> None:
    result = handle_diff(out_dir=str(tmp_path), response_format="xml")
    parsed = json.loads(result)
    assert parsed["metadata"]["status"] == "error"
    assert "response_format" in parsed["error"]["message"]


def test_handle_diff_invalid_evidence_schema(tmp_path: Path) -> None:
    result = handle_diff(out_dir=str(tmp_path), response_format="json", evidence_schema="verbose")
    parsed = json.loads(result)
    assert parsed["metadata"]["status"] == "error"


# ---------------------------------------------------------------------------
# Snapshot resolution errors
# ---------------------------------------------------------------------------


def test_handle_diff_no_snapshots_file_not_found(tmp_path: Path) -> None:
    result = handle_diff(out_dir=str(tmp_path), response_format="json")
    parsed = json.loads(result)
    assert parsed["error"]["code"] in ("NO_SNAPSHOTS", "SNAPSHOT_RESOLUTION_ERROR")


def test_handle_diff_file_not_found_no_snapshots_message(tmp_path: Path) -> None:
    with patch(
        "llmdebug.mcp_diff_tools.resolve_snapshot_ref",
        side_effect=FileNotFoundError("No snapshots found in /x"),
    ):
        result = handle_diff(out_dir=str(tmp_path), response_format="json")
    parsed = json.loads(result)
    assert parsed["error"]["code"] == "NO_SNAPSHOTS"


def test_handle_diff_file_not_found_resolution_error(tmp_path: Path) -> None:
    with patch(
        "llmdebug.mcp_diff_tools.resolve_snapshot_ref",
        side_effect=FileNotFoundError("snapshot not found"),
    ):
        result = handle_diff(out_dir=str(tmp_path), response_format="json")
    parsed = json.loads(result)
    assert parsed["error"]["code"] == "SNAPSHOT_RESOLUTION_ERROR"


def test_handle_diff_value_error(tmp_path: Path) -> None:
    with patch(
        "llmdebug.mcp_diff_tools.resolve_snapshot_ref",
        side_effect=ValueError("out of range"),
    ):
        result = handle_diff(out_dir=str(tmp_path), response_format="json")
    parsed = json.loads(result)
    assert parsed["error"]["code"] == "SNAPSHOT_RESOLUTION_ERROR"


def test_handle_diff_oserror(tmp_path: Path) -> None:
    with patch(
        "llmdebug.mcp_diff_tools.resolve_snapshot_ref",
        side_effect=OSError("permission denied"),
    ):
        result = handle_diff(out_dir=str(tmp_path), response_format="json")
    parsed = json.loads(result)
    assert parsed["error"]["code"] == "SNAPSHOT_LOAD_ERROR"


def test_handle_diff_unexpected_error(tmp_path: Path) -> None:
    with patch(
        "llmdebug.mcp_diff_tools.resolve_snapshot_ref",
        side_effect=RuntimeError("unexpected crash"),
    ):
        result = handle_diff(out_dir=str(tmp_path), response_format="json")
    parsed = json.loads(result)
    assert parsed["error"]["code"] == "SNAPSHOT_LOAD_ERROR"


# ---------------------------------------------------------------------------
# Successful diff
# ---------------------------------------------------------------------------


def _write_two_snapshots(out_dir: str) -> None:
    """Write two snapshots sequentially so we have a previous and latest."""
    import time

    cfg = SnapshotConfig(out_dir=out_dir, output_format="json")
    write_bundle(_make_snapshot("snap1", "ValueError"), cfg)
    time.sleep(0.02)
    write_bundle(_make_snapshot("snap2", "TypeError"), cfg)


def test_handle_diff_success_json(tmp_path: Path) -> None:
    _write_two_snapshots(str(tmp_path))
    result = handle_diff(
        old_ref="previous",
        new_ref="latest",
        out_dir=str(tmp_path),
        response_format="json",
        evidence_schema="summary",
    )
    parsed = json.loads(result)
    assert parsed["metadata"]["status"] == "ok"
    assert "evidence" in parsed


def test_handle_diff_success_text(tmp_path: Path) -> None:
    _write_two_snapshots(str(tmp_path))
    result = handle_diff(
        old_ref="previous",
        new_ref="latest",
        out_dir=str(tmp_path),
        response_format="text",
    )
    assert isinstance(result, str)
    # diff text output should mention exception change or frames
    assert len(result) > 0


def test_handle_diff_success_full_schema(tmp_path: Path) -> None:
    _write_two_snapshots(str(tmp_path))
    result = handle_diff(
        out_dir=str(tmp_path),
        response_format="json",
        evidence_schema="full",
    )
    parsed = json.loads(result)
    assert parsed["metadata"]["status"] == "ok"
    assert "diff" in parsed["evidence"]


# ---------------------------------------------------------------------------
# with_rca=True → rca_record is None path
# ---------------------------------------------------------------------------


def test_handle_diff_with_rca_failure(tmp_path: Path) -> None:
    _write_two_snapshots(str(tmp_path))
    with patch(
        "llmdebug.mcp_diff_tools.advance_rca_with_snapshot_refs",
        return_value=None,
    ):
        result = handle_diff(
            out_dir=str(tmp_path),
            response_format="json",
            with_rca=True,
        )
    parsed = json.loads(result)
    assert parsed["metadata"]["status"] == "error"
