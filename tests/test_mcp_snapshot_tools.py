"""Tests for mcp_snapshot_tools handler implementations and helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from llmdebug import _json as json
from llmdebug.config import SnapshotConfig
from llmdebug.mcp_snapshot_tools import (
    core_evidence_from_snapshot,
    diagnose_context_section,
    diagnose_crash_frame_section,
    diagnose_exception_section,
    handle_frame,
    handle_list,
    load_snapshot_safe,
    locals_summary,
    normalize_detail,
    render_local_preview,
    snapshot_error_code,
)
from llmdebug.output import write_bundle

# ---------------------------------------------------------------------------
# Minimal snapshot fixture
# ---------------------------------------------------------------------------


def _make_snapshot(
    *,
    exc_type: str = "ValueError",
    exc_msg: str = "bad value",
    frames: list[dict[str, Any]] | None = None,
    crash_frame_index: int = 0,
    pytest_data: dict[str, Any] | None = None,
    git_data: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if frames is None:
        frames = [
            {
                "file": "/code/app.py",
                "file_rel": "app.py",
                "line": 42,
                "function": "process",
                "locals": {"x": 1},
                "locals_meta": {"x": {"type": "builtins.int"}},
            }
        ]
    snap: dict[str, Any] = {
        "schema_version": "1.0",
        "name": "test_snap",
        "timestamp_utc": "2026-01-01T00:00:00Z",
        "exception": {"type": exc_type, "message": exc_msg},
        "frames": frames,
        "crash_frame_index": crash_frame_index,
    }
    if pytest_data:
        snap["pytest"] = pytest_data
    if git_data:
        snap["git"] = git_data
    return snap


def _write_snapshot(out_dir: str, snap: dict[str, Any] | None = None) -> None:
    payload = snap or _make_snapshot()
    cfg = SnapshotConfig(out_dir=out_dir, output_format="json")
    write_bundle(payload, cfg)


# ---------------------------------------------------------------------------
# render_local_preview
# ---------------------------------------------------------------------------


def test_render_local_preview_array_dict() -> None:
    value = {"__array__": "ndarray", "shape": "(3, 4)", "dtype": "float32"}
    result = render_local_preview(value)
    assert result == "<ndarray shape=(3, 4) dtype=float32>"


def test_render_local_preview_plain_value() -> None:
    result = render_local_preview(42)
    assert result == "42"


def test_render_local_preview_long_value_truncated() -> None:
    long_str = "x" * 200
    result = render_local_preview(long_str)
    assert len(result) <= 120
    assert result.endswith("...")


def test_render_local_preview_non_serializable_falls_back_to_repr() -> None:
    class _Unserializable:
        def __repr__(self) -> str:
            return "REPR_VALUE"

    result = render_local_preview(_Unserializable())
    assert "REPR_VALUE" in result


def test_render_local_preview_falls_back_to_repr_when_str_raises() -> None:
    class _Boom:
        def __str__(self) -> str:
            raise RuntimeError("boom")

        def __repr__(self) -> str:
            return "REPR_VALUE"

    result = render_local_preview(_Boom())
    assert result == "REPR_VALUE"


# ---------------------------------------------------------------------------
# normalize_detail
# ---------------------------------------------------------------------------


def test_normalize_detail_valid_values() -> None:
    assert normalize_detail("crash") == "crash"
    assert normalize_detail("full") == "full"
    assert normalize_detail("context") == "context"
    assert normalize_detail("  FULL  ") == "full"


def test_normalize_detail_invalid_returns_none() -> None:
    assert normalize_detail("verbose") is None
    assert normalize_detail("") is None


# ---------------------------------------------------------------------------
# locals_summary
# ---------------------------------------------------------------------------


def test_locals_summary_basic() -> None:
    frame = {
        "locals": {"x": 1, "y": "hello"},
        "locals_meta": {"x": {"type": "builtins.int"}, "y": {"type": "builtins.str"}},
    }
    result = locals_summary(frame)
    names = [r["name"] for r in result]
    assert "x" in names
    assert "y" in names
    assert result[0]["type"] in ("builtins.int", "builtins.str")


def test_locals_summary_skips_at_prefix() -> None:
    frame = {
        "locals": {"@hidden": 99, "visible": 1},
        "locals_meta": {},
    }
    result = locals_summary(frame)
    names = [r["name"] for r in result]
    assert "@hidden" not in names
    assert "visible" in names


def test_locals_summary_non_dict_locals_returns_empty() -> None:
    frame = {"locals": None}
    assert locals_summary(frame) == []


def test_locals_summary_respects_max_items() -> None:
    frame = {"locals": {f"v{i}": i for i in range(20)}, "locals_meta": {}}
    result = locals_summary(frame, max_items=3)
    assert len(result) == 3


def test_locals_summary_no_meta() -> None:
    frame = {"locals": {"a": 1}}
    result = locals_summary(frame)
    assert result[0]["type"] == ""


# ---------------------------------------------------------------------------
# snapshot_error_code
# ---------------------------------------------------------------------------


def test_snapshot_error_code_no_snapshots() -> None:
    assert snapshot_error_code("No snapshots found in /foo") == "NO_SNAPSHOTS"


def test_snapshot_error_code_load_error() -> None:
    assert snapshot_error_code("Error loading snapshot: ValueError: oops") == "SNAPSHOT_LOAD_ERROR"
    assert snapshot_error_code("Error loading snapshots: foo") == "SNAPSHOT_LOAD_ERROR"
    assert snapshot_error_code("Something else entirely") == "SNAPSHOT_LOAD_ERROR"


# ---------------------------------------------------------------------------
# core_evidence_from_snapshot
# ---------------------------------------------------------------------------


def test_core_evidence_from_snapshot_basic() -> None:
    snap = _make_snapshot()
    evidence = core_evidence_from_snapshot(snap)
    assert evidence["exception"]["type"] == "ValueError"
    assert evidence["crash_frame"]["function"] == "process"
    assert evidence["crash_frame"]["line"] == 42


def test_core_evidence_from_snapshot_none() -> None:
    evidence = core_evidence_from_snapshot(None)
    assert evidence["exception"]["type"] == ""
    assert evidence["crash_frame"]["index"] == -1


def test_core_evidence_from_snapshot_with_pytest_repro() -> None:
    snap = _make_snapshot(pytest_data={"nodeid": "tests/t.py::test_fn", "repro": ["pytest", "-x"]})
    evidence = core_evidence_from_snapshot(snap)
    assert evidence["repro"]["nodeid"] == "tests/t.py::test_fn"
    assert evidence["repro"]["argv"] == ["pytest", "-x"]


def test_core_evidence_no_frames() -> None:
    snap = _make_snapshot(frames=[])
    evidence = core_evidence_from_snapshot(snap)
    assert evidence["crash_frame"]["index"] == -1


# ---------------------------------------------------------------------------
# load_snapshot_safe
# ---------------------------------------------------------------------------


def test_load_snapshot_safe_file_not_found(tmp_path: Path) -> None:
    result = load_snapshot_safe(None, str(tmp_path), "test_tool")
    assert isinstance(result, str)
    assert "No snapshots found" in result


def test_load_snapshot_safe_success(tmp_path: Path) -> None:
    _write_snapshot(str(tmp_path))
    result = load_snapshot_safe(None, str(tmp_path), "test_tool")
    assert isinstance(result, dict)
    assert result.get("exception", {}).get("type") == "ValueError"


def test_load_snapshot_safe_value_error(tmp_path: Path) -> None:
    bad_file = tmp_path / "bad.json"
    bad_file.write_text("{not valid json}", encoding="utf-8")
    with patch("llmdebug.mcp_snapshot_tools.load_snapshot", side_effect=ValueError("bad json")):
        result = load_snapshot_safe(str(bad_file), str(tmp_path), "test_tool")
    assert isinstance(result, str)
    assert "Error loading snapshot" in result


def test_load_snapshot_safe_unexpected_error(tmp_path: Path) -> None:
    with patch(
        "llmdebug.mcp_snapshot_tools.load_snapshot",
        side_effect=RuntimeError("unexpected"),
    ):
        result = load_snapshot_safe("fake.json", str(tmp_path), "test_tool")
    assert isinstance(result, str)
    assert "Error loading snapshot" in result


def test_load_snapshot_safe_raw_mode(tmp_path: Path) -> None:
    _write_snapshot(str(tmp_path))
    result = load_snapshot_safe(None, str(tmp_path), "test_tool", raw=True)
    assert isinstance(result, dict)


# ---------------------------------------------------------------------------
# diagnose section builders
# ---------------------------------------------------------------------------


def test_diagnose_exception_section_basic() -> None:
    snap = _make_snapshot(exc_type="TypeError", exc_msg="bad type")
    section = diagnose_exception_section(snap)
    assert "TypeError" in section
    assert "bad type" in section


def test_diagnose_exception_section_missing_exception() -> None:
    section = diagnose_exception_section({})
    assert "UnknownError" in section


def test_diagnose_crash_frame_section_crash_detail() -> None:
    snap = _make_snapshot()
    section = diagnose_crash_frame_section(snap, "crash")
    assert section is not None
    assert "Crash Frame" in section
    # "crash" detail with one frame → no extra frames listed
    assert "more frame(s)" not in section


def test_diagnose_crash_frame_section_full_detail_multiple_frames() -> None:
    snap = _make_snapshot(
        frames=[
            {"file": "/a.py", "file_rel": "a.py", "line": 1, "function": "f1", "locals": {}},
            {"file": "/b.py", "file_rel": "b.py", "line": 2, "function": "f2", "locals": {}},
        ],
        crash_frame_index=0,
    )
    section = diagnose_crash_frame_section(snap, "full")
    assert section is not None
    assert "Frame #1" in section


def test_diagnose_crash_frame_section_crash_detail_multiple_frames_hint() -> None:
    snap = _make_snapshot(
        frames=[
            {"file": "/a.py", "file_rel": "a.py", "line": 1, "function": "f1", "locals": {}},
            {"file": "/b.py", "file_rel": "b.py", "line": 2, "function": "f2", "locals": {}},
        ],
        crash_frame_index=0,
    )
    section = diagnose_crash_frame_section(snap, "crash")
    assert section is not None
    assert "more frame(s)" in section


def test_diagnose_crash_frame_section_no_frames() -> None:
    snap = _make_snapshot(frames=[])
    assert diagnose_crash_frame_section(snap, "crash") is None


def test_diagnose_crash_frame_section_out_of_range_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    snap = _make_snapshot()
    monkeypatch.setattr(
        "llmdebug.mcp_snapshot_tools.coerce_crash_frame_index",
        lambda _snapshot: 99,
    )
    assert diagnose_crash_frame_section(snap, "crash") is None


def test_diagnose_context_section_crash_returns_none() -> None:
    snap = _make_snapshot()
    assert diagnose_context_section(snap, "crash") is None


def test_diagnose_context_section_full_with_repro_and_git() -> None:
    snap = _make_snapshot(
        pytest_data={"nodeid": "t.py::t", "repro": ["pytest", "tests/"]},
        git_data={"commit": "abc123", "branch": "main", "dirty": False},
    )
    section = diagnose_context_section(snap, "full")
    assert section is not None
    assert "Repro" in section
    assert "Git" in section
    assert "abc123" in section


def test_diagnose_context_section_dirty_flag() -> None:
    snap = _make_snapshot(git_data={"commit": "def456", "branch": "dev", "dirty": True})
    section = diagnose_context_section(snap, "context")
    assert section is not None
    assert "*dirty*" in section


def test_diagnose_context_section_no_repro_no_git() -> None:
    snap = _make_snapshot()
    section = diagnose_context_section(snap, "full")
    # No pytest/git data → returns None
    assert section is None


# ---------------------------------------------------------------------------
# handle_list
# ---------------------------------------------------------------------------


def test_handle_list_invalid_response_format() -> None:
    result = handle_list(response_format="xml")
    parsed = json.loads(result)
    assert parsed["metadata"]["status"] == "error"
    assert "response_format" in parsed["error"]["message"]


def test_handle_list_invalid_evidence_schema() -> None:
    result = handle_list(response_format="json", evidence_schema="verbose")
    parsed = json.loads(result)
    assert parsed["metadata"]["status"] == "error"


def test_handle_list_no_snapshots_text(tmp_path: Path) -> None:
    result = handle_list(out_dir=str(tmp_path), response_format="text")
    assert "No snapshots found" in result


def test_handle_list_no_snapshots_json(tmp_path: Path) -> None:
    result = handle_list(out_dir=str(tmp_path), response_format="json")
    parsed = json.loads(result)
    assert parsed["metadata"]["status"] == "error"
    assert parsed["error"]["code"] == "NO_SNAPSHOTS"


def test_handle_list_with_snapshots_text(tmp_path: Path) -> None:
    _write_snapshot(str(tmp_path))
    result = handle_list(out_dir=str(tmp_path), response_format="text")
    assert "Found" in result
    assert "ValueError" in result


def test_handle_list_with_snapshots_json(tmp_path: Path) -> None:
    _write_snapshot(str(tmp_path))
    result = handle_list(out_dir=str(tmp_path), response_format="json")
    parsed = json.loads(result)
    assert parsed["metadata"]["status"] == "ok"
    # Evidence summary contains the snapshot count in the text
    summary = parsed["evidence"]["summary"]
    assert "1 snapshot" in summary


def test_handle_list_oserror_text(tmp_path: Path) -> None:
    with patch(
        "llmdebug.mcp_snapshot_tools.find_all_snapshots", side_effect=OSError("perm denied")
    ):
        result = handle_list(out_dir=str(tmp_path), response_format="text")
    assert "Error scanning snapshots" in result


def test_handle_list_oserror_json(tmp_path: Path) -> None:
    with patch(
        "llmdebug.mcp_snapshot_tools.find_all_snapshots", side_effect=OSError("perm denied")
    ):
        result = handle_list(out_dir=str(tmp_path), response_format="json")
    parsed = json.loads(result)
    assert parsed["error"]["code"] == "SNAPSHOT_LOAD_ERROR"


def test_handle_list_unexpected_error_text(tmp_path: Path) -> None:
    with patch(
        "llmdebug.mcp_snapshot_tools.find_all_snapshots",
        side_effect=RuntimeError("unexpected"),
    ):
        result = handle_list(out_dir=str(tmp_path), response_format="text")
    assert "Error scanning snapshots" in result


def test_handle_list_unexpected_error_json(tmp_path: Path) -> None:
    with patch(
        "llmdebug.mcp_snapshot_tools.find_all_snapshots",
        side_effect=RuntimeError("unexpected"),
    ):
        result = handle_list(out_dir=str(tmp_path), response_format="json")
    parsed = json.loads(result)
    assert parsed["error"]["code"] == "SNAPSHOT_LOAD_ERROR"


# ---------------------------------------------------------------------------
# handle_frame
# ---------------------------------------------------------------------------


def test_handle_frame_invalid_response_format(tmp_path: Path) -> None:
    result = handle_frame(response_format="xml", out_dir=str(tmp_path))
    parsed = json.loads(result)
    assert parsed["metadata"]["status"] == "error"


def test_handle_frame_invalid_evidence_schema(tmp_path: Path) -> None:
    result = handle_frame(response_format="json", out_dir=str(tmp_path), evidence_schema="verbose")
    parsed = json.loads(result)
    assert parsed["metadata"]["status"] == "error"


def test_handle_frame_no_snapshot(tmp_path: Path) -> None:
    result = handle_frame(response_format="json", out_dir=str(tmp_path))
    parsed = json.loads(result)
    assert parsed["error"]["code"] == "NO_SNAPSHOTS"


def test_handle_frame_success(tmp_path: Path) -> None:
    _write_snapshot(str(tmp_path))
    result = handle_frame(index=0, out_dir=str(tmp_path), response_format="json")
    parsed = json.loads(result)
    assert parsed["metadata"]["status"] == "ok"
    assert parsed["evidence"]["crash_frame"]["function"] == "process"


def test_handle_frame_index_out_of_range(tmp_path: Path) -> None:
    _write_snapshot(str(tmp_path))
    result = handle_frame(index=99, out_dir=str(tmp_path), response_format="json")
    parsed = json.loads(result)
    assert parsed["error"]["code"] == "FRAME_OUT_OF_RANGE"


def test_handle_frame_no_frames_in_snapshot(tmp_path: Path) -> None:
    _write_snapshot(str(tmp_path), _make_snapshot(frames=[]))
    result = handle_frame(index=0, out_dir=str(tmp_path), response_format="json")
    parsed = json.loads(result)
    assert parsed["error"]["code"] == "MALFORMED_SNAPSHOT"


def test_handle_frame_malformed_frame_entry(tmp_path: Path) -> None:
    snap = _make_snapshot(frames=["not-a-dict"])
    _write_snapshot(str(tmp_path), snap)
    result = handle_frame(index=0, out_dir=str(tmp_path), response_format="json")
    parsed = json.loads(result)
    assert parsed["error"]["code"] == "MALFORMED_SNAPSHOT"


def test_handle_frame_text_format(tmp_path: Path) -> None:
    _write_snapshot(str(tmp_path))
    result = handle_frame(index=0, out_dir=str(tmp_path), response_format="text")
    assert isinstance(result, str)
    assert "process" in result


def test_handle_frame_rca_failure_returns_update_failed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write_snapshot(str(tmp_path))
    monkeypatch.setattr(
        "llmdebug.mcp_snapshot_tools.advance_rca_with_snapshot_refs",
        lambda *_a, **_k: None,
    )
    result = handle_frame(
        index=0,
        out_dir=str(tmp_path),
        response_format="json",
        with_rca=True,
    )
    parsed = json.loads(result)
    assert parsed["error"]["code"] == "RCA_STATE_UPDATE_ERROR"
