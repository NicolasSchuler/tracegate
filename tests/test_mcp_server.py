"""Tests for the llmdebug MCP server tools and snapshot loader."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

import llmdebug.mcp_server as mcp_server
from llmdebug import _json as json
from llmdebug._snapshot_loader import (
    find_all_snapshots,
    load_snapshot,
    load_snapshot_from_file,
    load_snapshot_from_file_raw,
    load_snapshot_raw,
)
from llmdebug.compact_keys import compact_keys
from llmdebug.config import SnapshotConfig
from llmdebug.mcp_server import (
    llmdebug_diagnose,
    llmdebug_diff,
    llmdebug_frame,
    llmdebug_git_context,
    llmdebug_list,
    llmdebug_rca_advance,
    llmdebug_rca_history,
    llmdebug_rca_status,
    llmdebug_show,
)
from llmdebug.output import write_bundle
from llmdebug.state_db import list_snapshot_records, upsert_snapshot_record


def _assert_json_error(
    raw: str,
    *,
    tool: str,
    code: str,
    message_contains: str | None = None,
    with_rca: bool = False,
    evidence_schema: str = "summary",
) -> dict[str, object]:
    data = json.loads(raw)
    assert data["tool"] == tool
    assert data["format_version"] == "2.0"
    assert data["metadata"]["response_format"] == "json"
    assert data["metadata"]["with_rca"] is with_rca
    assert data["metadata"]["evidence_schema"] == evidence_schema
    assert data["metadata"]["status"] == "error"
    assert data["error"]["code"] == code
    if message_contains is not None:
        assert message_contains in data["error"]["message"]
    assert set(data["evidence"].keys()) >= {
        "summary",
        "exception",
        "crash_frame",
        "locals_summary",
        "repro",
        "evidence_ids",
        "sections",
    }
    return data


@pytest.fixture
def sample_snapshot() -> dict:
    """A representative expanded snapshot dict."""
    return {
        "schema_version": "1.0",
        "name": "test_example",
        "timestamp_utc": "2026-02-07T21:00:00Z",
        "llmdebug_version": "2.0.0",
        "exception": {
            "type": "ValueError",
            "qualified_type": "builtins.ValueError",
            "message": "invalid literal for int(): 'abc'",
            "error_category": {"category": "value_error", "suggestion": "Check input types"},
        },
        "traceback": (
            "Traceback (most recent call last):\n"
            '  File "app.py", line 42\n'
            "    return int(value)\n"
            "ValueError: invalid literal"
        ),
        "frames": [
            {
                "file": "/code/app.py",
                "file_rel": "app.py",
                "line": 42,
                "function": "parse_input",
                "code": "    return int(value)",
                "source": {
                    "start": 40,
                    "end": 44,
                    "snippet": [
                        {"lineno": 40, "code": "def parse_input(value):"},
                        {"lineno": 41, "code": '    """Parse user input."""'},
                        {"lineno": 42, "code": "    return int(value)"},
                        {"lineno": 43, "code": ""},
                    ],
                },
                "locals": {"value": "'abc'", "self": "<App>"},
                "locals_meta": {
                    "value": {"type": "builtins.str", "len": 3},
                    "self": {"type": "app.App"},
                },
            },
            {
                "file": "/code/main.py",
                "file_rel": "main.py",
                "line": 10,
                "function": "run",
                "code": "    result = parse_input(data)",
                "source": {
                    "start": 8,
                    "end": 12,
                    "snippet": [
                        {"lineno": 8, "code": "def run():"},
                        {"lineno": 9, "code": "    data = get_input()"},
                        {"lineno": 10, "code": "    result = parse_input(data)"},
                        {"lineno": 11, "code": "    return result"},
                    ],
                },
                "locals": {"data": "'abc'"},
                "locals_meta": {"data": {"type": "builtins.str", "len": 3}},
            },
        ],
        "crash_frame_index": 0,
        "env": {"python": "3.13.0", "platform": "Linux", "cwd": "/code"},
        "git": {"commit": "abc1234", "branch": "main", "dirty": True},
        "pytest": {
            "nodeid": "tests/test_app.py::test_parse",
            "repro": ["python", "-m", "pytest", "tests/test_app.py::test_parse", "-q"],
        },
    }


@pytest.fixture
def snapshot_dir(tmp_path: Path, sample_snapshot: dict) -> Path:
    """Temp dir with 3 sample snapshots + latest.json."""
    cfg = SnapshotConfig(out_dir=str(tmp_path), output_format="json", max_snapshots=50)
    for i in range(3):
        payload = {**sample_snapshot, "name": f"test_{i}"}
        write_bundle(payload, cfg)
        time.sleep(0.05)
    return tmp_path


@pytest.fixture
def snapshot_dir_str(snapshot_dir: Path) -> str:
    return str(snapshot_dir)


# ---------------------------------------------------------------------------
# _snapshot_loader tests
# ---------------------------------------------------------------------------


class TestSnapshotLoader:
    def test_load_latest(self, snapshot_dir: Path) -> None:
        result = load_snapshot(None, str(snapshot_dir))
        assert result["exception"]["type"] == "ValueError"

    def test_load_from_path(self, snapshot_dir: Path) -> None:
        files = sorted(
            (
                p
                for p in snapshot_dir.iterdir()
                if p.suffix == ".json" and not p.name.startswith("latest")
            ),
            key=lambda p: p.stat().st_mtime,
        )
        result = load_snapshot(str(files[0]), str(snapshot_dir))
        assert result["exception"]["type"] == "ValueError"

    def test_load_compact_auto_expands(self, tmp_path: Path) -> None:
        snapshot = {
            "schema_version": "1.0",
            "name": "compact_test",
            "exception": {"type": "E"},
            "frames": [],
        }
        compacted = compact_keys(snapshot)
        assert isinstance(compacted, dict)
        compacted["__llmdebug_compact__"] = 1
        path = tmp_path / "compact.json"
        path.write_text(json.dumps(compacted), encoding="utf-8")

        loaded = load_snapshot_from_file(path)
        assert "schema_version" in loaded
        assert "__llmdebug_compact__" not in loaded

    def test_load_raw_latest(self, tmp_path: Path) -> None:
        cfg = SnapshotConfig(out_dir=str(tmp_path), output_format="json")
        payload = {
            "schema_version": "2.0",
            "kind": "llmdebug.debug_session",
            "session": {
                "name": "raw_latest",
                "timestamp_utc": "2026-02-10T14:00:00Z",
                "llmdebug_version": "2.1.0",
            },
            "snapshot": {
                "exception": {"type": "RuntimeError", "message": "boom"},
                "frames": [],
                "traceback": "tb",
                "capture_config": {"frames": 5},
            },
        }
        write_bundle(payload, cfg)
        loaded = load_snapshot_raw(None, str(tmp_path))
        assert loaded["kind"] == "llmdebug.debug_session"
        assert loaded["session"]["name"] == "raw_latest"

    def test_load_raw_from_path(self, tmp_path: Path) -> None:
        payload = {
            "schema_version": "2.0",
            "kind": "llmdebug.debug_session",
            "session": {
                "name": "raw_path",
                "timestamp_utc": "2026-02-10T14:05:00Z",
                "llmdebug_version": "2.1.0",
            },
            "snapshot": {"exception": {"type": "RuntimeError"}, "frames": []},
        }
        path = tmp_path / "raw_path.json"
        path.write_text(json.dumps(payload), encoding="utf-8")
        loaded = load_snapshot_raw(str(path), str(tmp_path))
        assert loaded["kind"] == "llmdebug.debug_session"
        assert loaded["session"]["name"] == "raw_path"

    def test_load_raw_path_validation(self, tmp_path: Path) -> None:
        out_dir = tmp_path / "allowed"
        out_dir.mkdir()
        outside = tmp_path / "outside.json"
        outside.write_text(json.dumps({"name": "x"}), encoding="utf-8")
        with pytest.raises(ValueError, match="outside snapshot directory"):
            load_snapshot_raw(str(outside), str(out_dir))

    def test_load_snapshot_from_file_raw_expands_compact(self, tmp_path: Path) -> None:
        snapshot = {
            "schema_version": "1.0",
            "name": "compact_raw",
            "exception": {"type": "E"},
            "frames": [],
        }
        compacted = compact_keys(snapshot)
        assert isinstance(compacted, dict)
        compacted["__llmdebug_compact__"] = 1
        path = tmp_path / "compact_raw.json"
        path.write_text(json.dumps(compacted), encoding="utf-8")
        loaded = load_snapshot_from_file_raw(path)
        assert "schema_version" in loaded
        assert "__llmdebug_compact__" not in loaded

    def test_load_debug_session_normalizes(self, tmp_path: Path) -> None:
        session_payload = {
            "schema_version": "2.0",
            "kind": "llmdebug.debug_session",
            "session": {
                "name": "debug_session_test",
                "timestamp_utc": "2026-02-10T12:00:00Z",
                "llmdebug_version": "2.1.0",
            },
            "snapshot": {
                "exception": {"type": "RuntimeError", "message": "boom"},
                "frames": [],
                "traceback": "tb",
                "capture_config": {"frames": 5},
            },
            "context": {"hook": "sys.excepthook"},
        }
        path = tmp_path / "debug_session.json"
        path.write_text(json.dumps(session_payload), encoding="utf-8")

        loaded = load_snapshot_from_file(path)
        assert loaded["schema_version"] == "2.0"
        assert loaded["name"] == "debug_session_test"
        assert loaded["exception"]["type"] == "RuntimeError"
        assert loaded["hook"] == "sys.excepthook"

    def test_file_not_found_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_snapshot_from_file(tmp_path / "nonexistent.json")

    def test_load_no_snapshots_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError, match="No snapshots found"):
            load_snapshot(None, str(tmp_path))

    def test_find_all_empty(self, tmp_path: Path) -> None:
        result = find_all_snapshots(str(tmp_path))
        assert result == []

    def test_find_all_nonexistent_dir(self) -> None:
        result = find_all_snapshots("/nonexistent/path")
        assert result == []

    def test_find_all_returns_sorted(self, snapshot_dir: Path) -> None:
        entries = find_all_snapshots(str(snapshot_dir))
        assert len(entries) == 3
        for i in range(len(entries) - 1):
            assert entries[i]["mtime"] >= entries[i + 1]["mtime"]

    def test_find_all_respects_limit(self, snapshot_dir: Path) -> None:
        entries = find_all_snapshots(str(snapshot_dir), limit=1)
        assert len(entries) == 1

    def test_find_all_removes_stale_index_rows(self, tmp_path: Path, sample_snapshot: dict) -> None:
        cfg = SnapshotConfig(out_dir=str(tmp_path), output_format="json", max_snapshots=50)
        written = write_bundle(sample_snapshot, cfg)
        indexed = list_snapshot_records(str(tmp_path))
        assert indexed

        written.unlink()
        entries = find_all_snapshots(str(tmp_path))
        assert entries == []
        assert list_snapshot_records(str(tmp_path)) == []

    def test_find_all_prunes_index_rows_outside_out_dir(self, tmp_path: Path) -> None:
        outside = tmp_path.parent / "outside_snapshot.json"
        outside.write_text("{}", encoding="utf-8")
        upsert_snapshot_record(
            str(tmp_path),
            {
                "path": str(outside.resolve()),
                "mtime": outside.stat().st_mtime,
                "name": "outside",
                "exc_type": "ValueError",
                "exc_message": "boom",
                "file": "outside.py",
                "line": "1",
            },
        )

        entries = find_all_snapshots(str(tmp_path))
        assert entries == []
        assert list_snapshot_records(str(tmp_path)) == []

    def test_find_all_warns_and_falls_back_when_index_read_fails(
        self,
        tmp_path: Path,
        sample_snapshot: dict,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        cfg = SnapshotConfig(out_dir=str(tmp_path), output_format="json", max_snapshots=50)
        write_bundle(sample_snapshot, cfg)

        def _raise_index_read(*_args: object, **_kwargs: object) -> list[dict[str, object]]:
            raise RuntimeError("sqlite unavailable")

        warnings: list[str] = []
        monkeypatch.setattr("llmdebug._snapshot_loader.list_snapshot_records", _raise_index_read)
        monkeypatch.setattr("llmdebug._snapshot_loader.emit_warning", warnings.append)

        entries = find_all_snapshots(str(tmp_path))
        assert len(entries) == 1
        assert warnings
        joined = "\n".join(warnings)
        assert "snapshot index read failed: RuntimeError" in joined
        assert "snapshot index unavailable; falling back to filesystem scan" in joined


# ---------------------------------------------------------------------------
# llmdebug_diagnose tests
# ---------------------------------------------------------------------------


class TestDiagnose:
    def test_returns_exception_info(self, snapshot_dir_str: str) -> None:
        result = llmdebug_diagnose(out_dir=snapshot_dir_str, response_format="text")
        assert "ValueError" in result
        assert "invalid literal" in result

    def test_includes_crash_frame(self, snapshot_dir_str: str) -> None:
        result = llmdebug_diagnose(out_dir=snapshot_dir_str, response_format="text")
        assert "parse_input" in result
        assert "app.py" in result

    def test_includes_source_snippet(self, snapshot_dir_str: str) -> None:
        result = llmdebug_diagnose(out_dir=snapshot_dir_str, response_format="text")
        assert "return int(value)" in result

    def test_includes_locals(self, snapshot_dir_str: str) -> None:
        result = llmdebug_diagnose(out_dir=snapshot_dir_str, response_format="text")
        assert "value" in result
        assert "'abc'" in result

    def test_includes_repro_command(self, snapshot_dir_str: str) -> None:
        result = llmdebug_diagnose(out_dir=snapshot_dir_str, response_format="text")
        assert "pytest" in result
        assert "test_app.py" in result

    def test_omits_error_category_hints(self, snapshot_dir_str: str) -> None:
        result = llmdebug_diagnose(out_dir=snapshot_dir_str, response_format="text")
        assert "## Hint" not in result
        assert "Category:" not in result
        assert "Suggestion:" not in result

    def test_includes_git(self, snapshot_dir_str: str) -> None:
        result = llmdebug_diagnose(out_dir=snapshot_dir_str, response_format="text")
        assert "abc1234" in result
        assert "main" in result
        assert "*dirty*" in result

    def test_no_snapshots_returns_message(self, tmp_path: Path) -> None:
        result = llmdebug_diagnose(out_dir=str(tmp_path))
        _assert_json_error(
            result,
            tool="llmdebug_diagnose",
            code="NO_SNAPSHOTS",
            message_contains="llmdebug installed",
        )

    def test_invalid_evidence_schema_json_error(self, snapshot_dir_str: str) -> None:
        result = llmdebug_diagnose(
            out_dir=snapshot_dir_str,
            response_format="json",
            with_rca=True,
            evidence_schema="invalid",  # type: ignore[arg-type]
        )
        _assert_json_error(
            result,
            tool="llmdebug_diagnose",
            code="INVALID_ARGUMENT",
            message_contains="Invalid evidence_schema",
            with_rca=True,
            evidence_schema="summary",
        )

    def test_includes_rca_block(self, snapshot_dir_str: str) -> None:
        result = llmdebug_diagnose(
            out_dir=snapshot_dir_str,
            response_format="text",
            with_rca=True,
        )
        assert "## RCA" in result
        assert "failure_signature" in result

    def test_handles_invalid_crash_frame_index(self, tmp_path: Path, sample_snapshot: dict) -> None:
        malformed = {**sample_snapshot, "crash_frame_index": "abc"}
        snapshot_path = tmp_path / "bad.json"
        snapshot_path.write_text(json.dumps(malformed), encoding="utf-8")
        (tmp_path / "latest.json").write_text(json.dumps(malformed), encoding="utf-8")

        result = llmdebug_diagnose(
            out_dir=str(tmp_path),
            response_format="text",
            with_rca=True,
        )
        assert "parse_input" in result
        assert "## RCA" in result

    def test_default_json_response_has_stable_evidence_fields(self, snapshot_dir_str: str) -> None:
        result = llmdebug_diagnose(out_dir=snapshot_dir_str)
        data = json.loads(result)
        evidence = data["evidence"]
        assert data["tool"] == "llmdebug_diagnose"
        assert data["format_version"] == "2.0"
        assert "rca" not in data
        assert set(evidence.keys()) >= {
            "exception",
            "crash_frame",
            "locals_summary",
            "repro",
            "evidence_ids",
            "sections",
        }

    def test_json_response_envelope(self, snapshot_dir_str: str) -> None:
        result = llmdebug_diagnose(
            out_dir=snapshot_dir_str,
            response_format="json",
            with_rca=True,
            evidence_schema="full",
        )
        data = json.loads(result)
        assert data["tool"] == "llmdebug_diagnose"
        assert "evidence" in data
        assert "snapshot" in data["evidence"]
        assert "rca" in data

    def test_with_rca_returns_error_when_rca_update_fails(
        self,
        snapshot_dir_str: str,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        import llmdebug.mcp_server as server

        monkeypatch.setattr(server, "_advance_rca_with_snapshot_refs", lambda **_kwargs: None)
        result = llmdebug_diagnose(
            out_dir=snapshot_dir_str,
            response_format="json",
            with_rca=True,
        )
        _assert_json_error(
            result,
            tool="llmdebug_diagnose",
            code="RCA_STATE_UPDATE_ERROR",
            with_rca=True,
        )

    def test_json_response_without_rca(self, snapshot_dir_str: str) -> None:
        result = llmdebug_diagnose(
            out_dir=snapshot_dir_str,
            response_format="json",
            with_rca=False,
        )
        data = json.loads(result)
        assert "rca" not in data

    def test_json_response_metadata_contract(self, snapshot_dir_str: str) -> None:
        result = llmdebug_diagnose(
            out_dir=snapshot_dir_str,
            response_format="json",
            with_rca=False,
            evidence_schema="summary",
        )
        data = json.loads(result)
        assert data["metadata"] == {
            "response_format": "json",
            "with_rca": False,
            "evidence_schema": "summary",
            "status": "ok",
        }

    def test_invalid_response_format_json_error(self, snapshot_dir_str: str) -> None:
        result = llmdebug_diagnose(
            out_dir=snapshot_dir_str,
            response_format="yaml",  # type: ignore[arg-type]
            with_rca=True,
        )
        data = _assert_json_error(
            result,
            tool="llmdebug_diagnose",
            code="INVALID_ARGUMENT",
            message_contains="Invalid response_format",
            with_rca=True,
        )
        details = data["error"]["details"]
        assert details["argument"] == "response_format"
        assert details["value"] == "yaml"

    def test_invalid_detail_json_error(self, snapshot_dir_str: str) -> None:
        result = llmdebug_diagnose(
            out_dir=snapshot_dir_str,
            detail="verbose",  # type: ignore[arg-type]
            response_format="json",
            with_rca=True,
        )
        data = _assert_json_error(
            result,
            tool="llmdebug_diagnose",
            code="INVALID_ARGUMENT",
            message_contains="Invalid detail",
            with_rca=True,
        )
        details = data["error"]["details"]
        assert details["argument"] == "detail"
        assert details["value"] == "verbose"
        assert details["allowed_values"] == ["crash", "full", "context"]

    def test_invalid_detail_text_error(self, snapshot_dir_str: str) -> None:
        result = llmdebug_diagnose(
            out_dir=snapshot_dir_str,
            detail="verbose",  # type: ignore[arg-type]
            response_format="text",
        )
        assert result.startswith("Invalid detail")

    def test_detail_normalization_accepts_whitespace_and_case(self, snapshot_dir_str: str) -> None:
        result = llmdebug_diagnose(
            out_dir=snapshot_dir_str,
            detail=" Full ",
            response_format="json",
            evidence_schema="full",
        )
        data = json.loads(result)
        assert data["metadata"]["status"] == "ok"
        assert data["evidence"]["detail"] == "full"


# ---------------------------------------------------------------------------
# llmdebug_show tests
# ---------------------------------------------------------------------------


class TestShow:
    def test_returns_valid_json(self, snapshot_dir_str: str) -> None:
        result = llmdebug_show(out_dir=snapshot_dir_str)
        data = json.loads(result)
        assert data["tool"] == "llmdebug_show"
        assert data["evidence"]["exception"]["type"] == "ValueError"
        assert "rca" not in data

    def test_show_specific_path(self, snapshot_dir: Path) -> None:
        files = [
            p
            for p in snapshot_dir.iterdir()
            if p.suffix == ".json" and not p.name.startswith("latest")
        ]
        result = llmdebug_show(
            out_dir=str(snapshot_dir),
            snapshot_path=str(files[0]),
            evidence_schema="full",
        )
        data = json.loads(result)
        assert "snapshot" in data["evidence"]

    def test_raw_session_returns_envelope(self, tmp_path: Path) -> None:
        cfg = SnapshotConfig(out_dir=str(tmp_path), output_format="json")
        payload = {
            "schema_version": "2.0",
            "kind": "llmdebug.debug_session",
            "session": {
                "name": "mcp_raw_test",
                "timestamp_utc": "2026-02-10T15:00:00Z",
                "llmdebug_version": "2.1.0",
            },
            "snapshot": {
                "exception": {"type": "ValueError", "message": "boom"},
                "frames": [],
                "traceback": "tb",
                "capture_config": {"frames": 5},
            },
            "context": {"hook": "sys.excepthook"},
        }
        write_bundle(payload, cfg)

        result = llmdebug_show(
            out_dir=str(tmp_path),
            raw_session=True,
            response_format="text",
        )
        data = json.loads(result)
        assert data["kind"] == "llmdebug.debug_session"
        assert data["session"]["name"] == "mcp_raw_test"
        assert data["snapshot"]["exception"]["type"] == "ValueError"

    def test_not_found(self, tmp_path: Path) -> None:
        result = llmdebug_show(out_dir=str(tmp_path))
        _assert_json_error(result, tool="llmdebug_show", code="NO_SNAPSHOTS")

    def test_show_with_rca_envelope(self, snapshot_dir_str: str) -> None:
        result = llmdebug_show(
            out_dir=snapshot_dir_str,
            with_rca=True,
            evidence_schema="full",
        )
        data = json.loads(result)
        assert "snapshot" in data["evidence"]
        assert "rca" in data

    def test_show_json_mode_full_schema(self, snapshot_dir_str: str) -> None:
        result = llmdebug_show(
            out_dir=snapshot_dir_str,
            response_format="json",
            with_rca=True,
            evidence_schema="full",
        )
        data = json.loads(result)
        assert data["tool"] == "llmdebug_show"
        assert "snapshot" in data["evidence"]
        assert "rca" in data

    def test_invalid_response_format_json_error(self, snapshot_dir_str: str) -> None:
        result = llmdebug_show(
            out_dir=snapshot_dir_str,
            response_format="yaml",  # type: ignore[arg-type]
            with_rca=True,
        )
        _assert_json_error(
            result,
            tool="llmdebug_show",
            code="INVALID_ARGUMENT",
            message_contains="Invalid response_format",
            with_rca=True,
        )

    def test_invalid_evidence_schema_json_error(self, snapshot_dir_str: str) -> None:
        result = llmdebug_show(
            out_dir=snapshot_dir_str,
            response_format="json",
            with_rca=True,
            evidence_schema="invalid",  # type: ignore[arg-type]
        )
        _assert_json_error(
            result,
            tool="llmdebug_show",
            code="INVALID_ARGUMENT",
            message_contains="Invalid evidence_schema",
            with_rca=True,
            evidence_schema="summary",
        )

    def test_invalid_detail_json_error(self, snapshot_dir_str: str) -> None:
        result = llmdebug_show(
            out_dir=snapshot_dir_str,
            detail="verbose",  # type: ignore[arg-type]
            response_format="json",
            with_rca=True,
        )
        data = _assert_json_error(
            result,
            tool="llmdebug_show",
            code="INVALID_ARGUMENT",
            message_contains="Invalid detail",
            with_rca=True,
        )
        details = data["error"]["details"]
        assert details["argument"] == "detail"
        assert details["value"] == "verbose"
        assert details["allowed_values"] == ["crash", "full", "context"]

    def test_invalid_detail_text_error(self, snapshot_dir_str: str) -> None:
        result = llmdebug_show(
            out_dir=snapshot_dir_str,
            detail="verbose",  # type: ignore[arg-type]
            response_format="text",
        )
        assert result.startswith("Invalid detail")

    def test_detail_normalization_accepts_whitespace_and_case(self, snapshot_dir_str: str) -> None:
        result = llmdebug_show(
            out_dir=snapshot_dir_str,
            detail=" Full ",
            response_format="json",
            evidence_schema="full",
        )
        data = json.loads(result)
        assert data["metadata"]["status"] == "ok"
        assert data["evidence"]["detail"] == "full"

    def test_show_handles_non_numeric_crash_line(self, tmp_path: Path) -> None:
        cfg = SnapshotConfig(out_dir=str(tmp_path), output_format="json")
        write_bundle(
            {
                "name": "bad-line-show",
                "exception": {"type": "ValueError", "message": "boom"},
                "frames": [{"file": "/code/app.py", "file_rel": "app.py", "line": "?"}],
                "crash_frame_index": 0,
            },
            cfg,
        )

        result = llmdebug_show(out_dir=str(tmp_path), response_format="json")
        data = json.loads(result)

        assert data["metadata"]["status"] == "ok"
        assert data["evidence"]["crash_frame"]["line"] == 0


def test_invalid_detail_uses_shared_helper_for_diagnose_and_show(
    monkeypatch: pytest.MonkeyPatch, snapshot_dir_str: str
) -> None:
    calls: list[str] = []

    def _fake_invalid_detail_error(
        *,
        tool: str,
        detail: str,
        response_format: str,
        evidence_schema: str,
        with_rca: bool,
    ) -> str:
        calls.append(tool)
        assert detail == "verbose"
        assert response_format == "json"
        assert evidence_schema == "summary"
        assert with_rca is True
        return f"{tool}:invalid"

    monkeypatch.setattr(mcp_server, "_render_invalid_detail_error", _fake_invalid_detail_error)

    diagnose_result = llmdebug_diagnose(
        out_dir=snapshot_dir_str,
        detail="verbose",  # type: ignore[arg-type]
        response_format="json",
        with_rca=True,
    )
    show_result = llmdebug_show(
        out_dir=snapshot_dir_str,
        detail="verbose",  # type: ignore[arg-type]
        response_format="json",
        with_rca=True,
    )

    assert diagnose_result == "llmdebug_diagnose:invalid"
    assert show_result == "llmdebug_show:invalid"
    assert calls == ["llmdebug_diagnose", "llmdebug_show"]


# ---------------------------------------------------------------------------
# llmdebug_list tests
# ---------------------------------------------------------------------------


class TestList:
    def test_returns_entries(self, snapshot_dir_str: str) -> None:
        result = llmdebug_list(out_dir=snapshot_dir_str)
        assert "test_" in result
        assert "ValueError" in result

    def test_empty_dir(self, tmp_path: Path) -> None:
        result = llmdebug_list(out_dir=str(tmp_path))
        assert "No snapshots found" in result

    @pytest.mark.parametrize("limit", [0, -1])
    def test_list_rejects_non_positive_limit(self, snapshot_dir_str: str, limit: int) -> None:
        """RCA-17: limit <= 0 returned every snapshot; llmdebug_rca_history rejects it."""
        data = json.loads(
            llmdebug_list(out_dir=snapshot_dir_str, limit=limit, response_format="json")
        )

        assert data["error"]["code"] == "INVALID_ARGUMENT"
        assert "limit must be >= 1" in data["error"]["message"]

    def test_respects_limit(self, snapshot_dir_str: str) -> None:
        result = llmdebug_list(out_dir=snapshot_dir_str, limit=1)
        assert "1 snapshot(s)" in result

    def test_includes_path(self, snapshot_dir_str: str) -> None:
        result = llmdebug_list(out_dir=snapshot_dir_str)
        assert "path:" in result

    def test_json_mode_full_schema(self, snapshot_dir_str: str) -> None:
        result = llmdebug_list(
            out_dir=snapshot_dir_str,
            response_format="json",
            evidence_schema="full",
        )
        data = json.loads(result)
        assert data["tool"] == "llmdebug_list"
        assert data["metadata"]["status"] == "ok"
        snapshot_list = data["evidence"]["snapshot_list"]
        assert snapshot_list["count"] >= 1
        assert snapshot_list["entries"]

    def test_json_mode_no_snapshots(self, tmp_path: Path) -> None:
        result = llmdebug_list(out_dir=str(tmp_path), response_format="json")
        _assert_json_error(
            result,
            tool="llmdebug_list",
            code="NO_SNAPSHOTS",
            message_contains="No snapshots found",
        )

    def test_invalid_response_format_json_error(self, snapshot_dir_str: str) -> None:
        result = llmdebug_list(
            out_dir=snapshot_dir_str,
            response_format="yaml",  # type: ignore[arg-type]
        )
        _assert_json_error(
            result,
            tool="llmdebug_list",
            code="INVALID_ARGUMENT",
            message_contains="Invalid response_format",
        )


# ---------------------------------------------------------------------------
# llmdebug_frame tests
# ---------------------------------------------------------------------------


class TestFrame:
    def test_returns_crash_frame(self, snapshot_dir_str: str) -> None:
        result = llmdebug_frame(index=0, out_dir=snapshot_dir_str, response_format="text")
        assert "parse_input" in result
        assert "[CRASH]" in result

    def test_returns_caller_frame(self, snapshot_dir_str: str) -> None:
        result = llmdebug_frame(index=1, out_dir=snapshot_dir_str, response_format="text")
        assert "run" in result
        assert "[CRASH]" not in result

    def test_out_of_range(self, snapshot_dir_str: str) -> None:
        result = llmdebug_frame(index=99, out_dir=snapshot_dir_str)
        _assert_json_error(
            result,
            tool="llmdebug_frame",
            code="FRAME_OUT_OF_RANGE",
            message_contains="out of range",
        )

    def test_includes_locals(self, snapshot_dir_str: str) -> None:
        result = llmdebug_frame(index=0, out_dir=snapshot_dir_str, response_format="text")
        assert "value" in result
        assert "'abc'" in result

    def test_includes_source(self, snapshot_dir_str: str) -> None:
        result = llmdebug_frame(index=0, out_dir=snapshot_dir_str, response_format="text")
        assert "return int(value)" in result

    def test_plain_sections_preserved(self, snapshot_dir_str: str) -> None:
        result = llmdebug_frame(index=0, out_dir=snapshot_dir_str, response_format="text")
        assert "Source:" in result
        assert "Locals:" in result

    def test_no_snapshots(self, tmp_path: Path) -> None:
        result = llmdebug_frame(index=0, out_dir=str(tmp_path), with_rca=True)
        _assert_json_error(result, tool="llmdebug_frame", code="NO_SNAPSHOTS", with_rca=True)

    def test_malformed_frame_entry(self, tmp_path: Path, sample_snapshot: dict) -> None:
        malformed = {**sample_snapshot, "frames": [sample_snapshot["frames"][0], "bad-frame"]}
        snapshot_path = tmp_path / "bad.json"
        snapshot_path.write_text(json.dumps(malformed), encoding="utf-8")
        (tmp_path / "latest.json").write_text(json.dumps(malformed), encoding="utf-8")

        result = llmdebug_frame(index=1, out_dir=str(tmp_path))
        _assert_json_error(
            result,
            tool="llmdebug_frame",
            code="MALFORMED_SNAPSHOT",
            message_contains="malformed",
        )

    def test_frame_json_without_rca(self, snapshot_dir_str: str) -> None:
        result = llmdebug_frame(
            index=0,
            out_dir=snapshot_dir_str,
            response_format="json",
            with_rca=False,
            evidence_schema="full",
        )
        data = json.loads(result)
        assert data["tool"] == "llmdebug_frame"
        assert "frame" in data["evidence"]
        assert "rca" not in data

    def test_invalid_response_format_json_error(self, snapshot_dir_str: str) -> None:
        result = llmdebug_frame(
            index=0,
            out_dir=snapshot_dir_str,
            response_format="yaml",  # type: ignore[arg-type]
            with_rca=True,
        )
        _assert_json_error(
            result,
            tool="llmdebug_frame",
            code="INVALID_ARGUMENT",
            message_contains="Invalid response_format",
            with_rca=True,
        )

    def test_frame_handles_non_numeric_line(self, tmp_path: Path) -> None:
        cfg = SnapshotConfig(out_dir=str(tmp_path), output_format="json")
        write_bundle(
            {
                "name": "bad-line-frame",
                "exception": {"type": "ValueError", "message": "boom"},
                "frames": [
                    {
                        "file": "/code/app.py",
                        "file_rel": "app.py",
                        "line": "??",
                        "function": "parse_input",
                        "code": "return int(value)",
                    }
                ],
                "crash_frame_index": 0,
            },
            cfg,
        )

        result = llmdebug_frame(index=0, out_dir=str(tmp_path), response_format="json")
        data = json.loads(result)

        assert data["metadata"]["status"] == "ok"
        assert data["evidence"]["crash_frame"]["line"] == 0


class TestGitContext:
    def test_returns_enhanced_context(self, snapshot_dir_str: str, monkeypatch) -> None:
        import llmdebug.mcp_server as server

        monkeypatch.setattr(
            server,
            "get_enhanced_git_context",
            lambda *_args, **_kwargs: {
                "head": {
                    "commit": "abc1234",
                    "branch": "main",
                    "dirty": False,
                    "detached": False,
                },
                "crash_file": {
                    "file_rel": "app.py",
                    "line": 42,
                    "in_repo": True,
                    "blame": {
                        "commit": "def5678",
                        "author": "Alice",
                        "author_time": "2026-02-10T10:00:00Z",
                        "summary": "Fix parser",
                    },
                },
                "recent_commits": [
                    {
                        "commit": "abc1234",
                        "author": "Alice",
                        "author_time": "2026-02-10T10:00:00+00:00",
                        "summary": "Fix parser",
                        "files_changed": 1,
                        "insertions": 2,
                        "deletions": 1,
                    }
                ],
            },
        )
        result = llmdebug_git_context(
            out_dir=snapshot_dir_str,
            max_commits=3,
            response_format="text",
            with_rca=True,
        )
        assert "## Git Head" in result
        assert "abc1234" in result
        assert "## RCA" in result

    def test_unavailable_returns_message(self, snapshot_dir_str: str, monkeypatch) -> None:
        import llmdebug.mcp_server as server

        monkeypatch.setattr(server, "get_enhanced_git_context", lambda *_args, **_kwargs: None)
        result = llmdebug_git_context(
            out_dir=snapshot_dir_str,
            response_format="text",
            with_rca=True,
        )
        assert "unavailable" in result.lower()
        assert "## RCA" in result

    def test_no_snapshots(self, tmp_path: Path) -> None:
        result = llmdebug_git_context(out_dir=str(tmp_path))
        _assert_json_error(result, tool="llmdebug_git_context", code="NO_SNAPSHOTS")

    def test_invalid_max_commits(self, snapshot_dir_str: str) -> None:
        result = llmdebug_git_context(
            out_dir=snapshot_dir_str,
            max_commits="abc",  # type: ignore[arg-type]
        )
        _assert_json_error(
            result,
            tool="llmdebug_git_context",
            code="INVALID_ARGUMENT",
            message_contains="Invalid max_commits value",
        )

    def test_git_context_failure_stays_in_the_error_envelope(
        self, snapshot_dir_str: str, monkeypatch
    ) -> None:
        """RCA-10: an exception from git used to escape the tool."""
        import llmdebug.mcp_server as server

        def _boom(*_args, **_kwargs):
            raise UnicodeDecodeError("utf-8", b"\xff", 0, 1, "invalid start byte")

        monkeypatch.setattr(server, "get_enhanced_git_context", _boom)
        data = json.loads(llmdebug_git_context(out_dir=snapshot_dir_str, response_format="json"))

        assert data["error"]["code"] == "GIT_CONTEXT_ERROR"
        assert "UnicodeDecodeError" in data["error"]["message"]

    def test_git_context_json_mode(self, snapshot_dir_str: str, monkeypatch) -> None:
        import llmdebug.mcp_server as server

        monkeypatch.setattr(server, "get_enhanced_git_context", lambda *_args, **_kwargs: None)
        result = llmdebug_git_context(
            out_dir=snapshot_dir_str,
            response_format="json",
            with_rca=False,
            evidence_schema="full",
        )
        data = json.loads(result)
        assert data["tool"] == "llmdebug_git_context"
        assert "rca" not in data

    def test_invalid_response_format_json_error(self, snapshot_dir_str: str) -> None:
        result = llmdebug_git_context(
            out_dir=snapshot_dir_str,
            response_format="yaml",  # type: ignore[arg-type]
            with_rca=True,
        )
        _assert_json_error(
            result,
            tool="llmdebug_git_context",
            code="INVALID_ARGUMENT",
            message_contains="Invalid response_format",
            with_rca=True,
        )


# ---------------------------------------------------------------------------
# llmdebug_diff tests
# ---------------------------------------------------------------------------


@pytest.fixture
def diff_snapshot_dir(tmp_path: Path) -> Path:
    """Dir with two *different* snapshots for diff testing."""
    snap_old = {
        "schema_version": "1.0",
        "name": "test_old",
        "timestamp_utc": "2026-02-08T10:00:00Z",
        "llmdebug_version": "2.0.0",
        "exception": {
            "type": "ValueError",
            "message": "invalid literal for int(): 'abc'",
            "error_category": {"category": "value_error"},
        },
        "frames": [
            {
                "file": "/code/app.py",
                "file_rel": "app.py",
                "line": 42,
                "function": "parse_input",
                "locals": {"value": "'abc'"},
                "locals_meta": {"value": {"type": "str"}},
            }
        ],
        "crash_frame_index": 0,
        "git": {"commit": "aaa1111", "branch": "main"},
        "pytest": {"nodeid": "tests/test_app.py::test_parse"},
    }
    snap_new = {
        "schema_version": "1.0",
        "name": "test_new",
        "timestamp_utc": "2026-02-08T10:05:00Z",
        "llmdebug_version": "2.0.0",
        "exception": {
            "type": "TypeError",
            "message": "unsupported operand type(s)",
            "error_category": {"category": "type_error"},
        },
        "frames": [
            {
                "file": "/code/app.py",
                "file_rel": "app.py",
                "line": 38,
                "function": "transform_data",
                "locals": {"value": 42},
                "locals_meta": {"value": {"type": "int"}},
            }
        ],
        "crash_frame_index": 0,
        "git": {"commit": "bbb2222", "branch": "main"},
        "pytest": {"nodeid": "tests/test_app.py::test_parse"},
    }
    old_path = tmp_path / "snap_old.json"
    old_path.write_text(json.dumps(snap_old), encoding="utf-8")
    time.sleep(0.05)
    new_path = tmp_path / "snap_new.json"
    new_path.write_text(json.dumps(snap_new), encoding="utf-8")
    (tmp_path / "latest.json").write_text(json.dumps(snap_new), encoding="utf-8")
    return tmp_path


class TestDiff:
    def test_diff_default(self, diff_snapshot_dir: Path) -> None:
        result = llmdebug_diff(
            out_dir=str(diff_snapshot_dir),
            response_format="text",
            with_rca=True,
        )
        assert "CHANGED" in result
        assert "ValueError" in result
        assert "TypeError" in result
        assert "## Structured Diff" in result
        assert "## RCA" in result

    def test_diff_by_path(self, diff_snapshot_dir: Path) -> None:
        old = str(diff_snapshot_dir / "snap_old.json")
        new = str(diff_snapshot_dir / "snap_new.json")
        result = llmdebug_diff(
            old_ref=old,
            new_ref=new,
            out_dir=str(diff_snapshot_dir),
            response_format="text",
        )
        assert "CHANGED" in result
        assert "MOVED" in result

    def test_diff_not_found(self, tmp_path: Path) -> None:
        result = llmdebug_diff(out_dir=str(tmp_path))
        _assert_json_error(
            result,
            tool="llmdebug_diff",
            code="SNAPSHOT_RESOLUTION_ERROR",
            message_contains="at least 2 snapshots",
        )

    def test_diff_one_snapshot_error(self, tmp_path: Path) -> None:
        snap = {
            "schema_version": "1.0",
            "name": "only",
            "exception": {"type": "E", "message": ""},
            "frames": [],
            "crash_frame_index": 0,
        }
        (tmp_path / "only.json").write_text(json.dumps(snap), encoding="utf-8")
        (tmp_path / "latest.json").write_text(json.dumps(snap), encoding="utf-8")
        result = llmdebug_diff(out_dir=str(tmp_path))
        _assert_json_error(
            result,
            tool="llmdebug_diff",
            code="SNAPSHOT_RESOLUTION_ERROR",
            message_contains="at least 2",
        )

    def test_diff_json_mode_full_schema(self, diff_snapshot_dir: Path) -> None:
        result = llmdebug_diff(
            out_dir=str(diff_snapshot_dir),
            response_format="json",
            with_rca=False,
            evidence_schema="full",
        )
        data = json.loads(result)
        assert data["tool"] == "llmdebug_diff"
        assert "diff" in data["evidence"]
        assert "rca" not in data

    def test_invalid_response_format_json_error(self, diff_snapshot_dir: Path) -> None:
        result = llmdebug_diff(
            out_dir=str(diff_snapshot_dir),
            response_format="yaml",  # type: ignore[arg-type]
            with_rca=True,
        )
        _assert_json_error(
            result,
            tool="llmdebug_diff",
            code="INVALID_ARGUMENT",
            message_contains="Invalid response_format",
            with_rca=True,
        )


class TestRCAStateTools:
    def test_rca_status_and_history(self, snapshot_dir_str: str) -> None:
        _ = llmdebug_diagnose(out_dir=snapshot_dir_str, with_rca=True)
        status_raw = llmdebug_rca_status(out_dir=snapshot_dir_str)
        status = json.loads(status_raw)
        assert "state" in status
        assert "attempt_id" in status

        history_raw = llmdebug_rca_history(out_dir=snapshot_dir_str, limit=5)
        history = json.loads(history_raw)
        assert isinstance(history, list)
        assert len(history) >= 1
        assert "state" in history[0]

    def test_rca_advance_manual(self, snapshot_dir_str: str) -> None:
        output = llmdebug_rca_advance(
            action="hypothesize",
            out_dir=snapshot_dir_str,
            evidence_refs="crash_frame:app.py:42:parse_input",
            hypothesis="Input is not numeric",
            experiment_plan="Guard with str.isdigit() before int conversion",
            gate_mode="soft",
        )
        data = json.loads(output)
        assert data["state"] in {"hypothesized", "stalled"}
        assert data["hypothesis"]

    def test_rca_status_json_mode(self, snapshot_dir_str: str) -> None:
        _ = llmdebug_diagnose(out_dir=snapshot_dir_str, with_rca=True)
        result = llmdebug_rca_status(
            out_dir=snapshot_dir_str,
            response_format="json",
            evidence_schema="full",
        )
        data = json.loads(result)
        assert data["tool"] == "llmdebug_rca_status"
        assert data["metadata"]["status"] == "ok"
        assert "rca_status" in data["evidence"]

    def test_rca_history_json_mode(self, snapshot_dir_str: str) -> None:
        _ = llmdebug_diagnose(out_dir=snapshot_dir_str, with_rca=True)
        result = llmdebug_rca_history(
            out_dir=snapshot_dir_str,
            response_format="json",
            evidence_schema="full",
        )
        data = json.loads(result)
        assert data["tool"] == "llmdebug_rca_history"
        assert data["metadata"]["status"] == "ok"
        assert isinstance(data["evidence"]["rca_history"], list)

    def test_rca_status_without_session_returns_latest_global_attempt(
        self, snapshot_dir_str: str
    ) -> None:
        first = llmdebug_rca_advance(
            action="bind_evidence",
            out_dir=snapshot_dir_str,
            session_id="session-a",
            failure_signature="sig-a",
            response_format="json",
            evidence_schema="full",
        )
        second = llmdebug_rca_advance(
            action="bind_evidence",
            out_dir=snapshot_dir_str,
            session_id="session-b",
            failure_signature="sig-b",
            response_format="json",
            evidence_schema="full",
        )
        first_id = json.loads(first)["evidence"]["rca_record"]["attempt_id"]
        second_id = json.loads(second)["evidence"]["rca_record"]["attempt_id"]
        assert first_id != second_id

        result = llmdebug_rca_status(
            out_dir=snapshot_dir_str,
            response_format="json",
            evidence_schema="full",
        )
        data = json.loads(result)
        assert data["tool"] == "llmdebug_rca_status"
        assert data["evidence"]["rca_status"]["session_id"] == "session-b"

    def test_rca_advance_json_mode(self, snapshot_dir_str: str) -> None:
        result = llmdebug_rca_advance(
            action="hypothesize",
            out_dir=snapshot_dir_str,
            response_format="json",
            evidence_schema="full",
        )
        data = json.loads(result)
        assert data["tool"] == "llmdebug_rca_advance"
        assert data["metadata"]["status"] == "ok"
        assert data["evidence"]["rca_record"]["state"] in {"hypothesized", "stalled"}

    def test_rca_advance_rejects_invalid_action(self, snapshot_dir_str: str) -> None:
        result = llmdebug_rca_advance(
            action="not-a-real-action",
            out_dir=snapshot_dir_str,
            response_format="json",
        )
        _assert_json_error(
            result,
            tool="llmdebug_rca_advance",
            code="INVALID_ARGUMENT",
            message_contains="Invalid RCA action",
        )

    def test_rca_advance_rejects_partial_manual_identity(self, snapshot_dir_str: str) -> None:
        result = llmdebug_rca_advance(
            action="bind_evidence",
            out_dir=snapshot_dir_str,
            session_id="manual-session",
            failure_signature="",
            response_format="json",
        )
        _assert_json_error(
            result,
            tool="llmdebug_rca_advance",
            code="INVALID_ARGUMENT",
            message_contains="provide both session_id and failure_signature",
        )

    def test_rca_advance_json_snapshot_load_error(self, snapshot_dir: Path) -> None:
        missing = snapshot_dir / "missing.json"
        result = llmdebug_rca_advance(
            action="hypothesize",
            out_dir=str(snapshot_dir),
            snapshot_path=str(missing),
            response_format="json",
        )
        _assert_json_error(
            result,
            tool="llmdebug_rca_advance",
            code="SNAPSHOT_LOAD_ERROR",
            message_contains="Error loading snapshot:",
        )

    def test_rca_status_json_no_attempts(self, tmp_path: Path) -> None:
        result = llmdebug_rca_status(out_dir=str(tmp_path), response_format="json")
        _assert_json_error(
            result,
            tool="llmdebug_rca_status",
            code="NO_RCA_ATTEMPTS",
            message_contains="No RCA attempts found",
        )

    def test_rca_history_rejects_non_positive_limit(self, snapshot_dir_str: str) -> None:
        result = llmdebug_rca_history(
            out_dir=snapshot_dir_str,
            limit=0,
            response_format="json",
        )
        _assert_json_error(
            result,
            tool="llmdebug_rca_history",
            code="INVALID_ARGUMENT",
            message_contains="limit must be >= 1",
        )

    def test_rca_invalid_response_format_json_error(self, snapshot_dir_str: str) -> None:
        result = llmdebug_rca_status(
            out_dir=snapshot_dir_str,
            response_format="yaml",  # type: ignore[arg-type]
        )
        _assert_json_error(
            result,
            tool="llmdebug_rca_status",
            code="INVALID_ARGUMENT",
            message_contains="Invalid response_format",
        )


# ---------------------------------------------------------------------------
# Rate-limit paths — inject a limiter that always returns False
# ---------------------------------------------------------------------------


class _ExhaustedLimiter:
    """A fake rate limiter that always denies."""

    rate = 1.0
    burst = 1

    def acquire(self) -> bool:
        return False


def _set_exhausted_limiter(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(mcp_server, "_mcp_limiter", _ExhaustedLimiter())


def test_rate_limit_diagnose(snapshot_dir_str: str, monkeypatch: pytest.MonkeyPatch) -> None:
    _set_exhausted_limiter(monkeypatch)
    result = llmdebug_diagnose(out_dir=snapshot_dir_str)
    data = json.loads(result)
    assert data["error"]["code"] == "RATE_LIMITED"


def test_rate_limit_show(snapshot_dir_str: str, monkeypatch: pytest.MonkeyPatch) -> None:
    _set_exhausted_limiter(monkeypatch)
    result = llmdebug_show(out_dir=snapshot_dir_str)
    data = json.loads(result)
    assert data["error"]["code"] == "RATE_LIMITED"


def test_rate_limit_list(snapshot_dir_str: str, monkeypatch: pytest.MonkeyPatch) -> None:
    _set_exhausted_limiter(monkeypatch)
    result = llmdebug_list(out_dir=snapshot_dir_str, response_format="json")
    data = json.loads(result)
    assert data["error"]["code"] == "RATE_LIMITED"


def test_rate_limit_frame(snapshot_dir_str: str, monkeypatch: pytest.MonkeyPatch) -> None:
    _set_exhausted_limiter(monkeypatch)
    result = llmdebug_frame(out_dir=snapshot_dir_str)
    data = json.loads(result)
    assert data["error"]["code"] == "RATE_LIMITED"


def test_rate_limit_git_context(snapshot_dir_str: str, monkeypatch: pytest.MonkeyPatch) -> None:
    _set_exhausted_limiter(monkeypatch)
    result = llmdebug_git_context(out_dir=snapshot_dir_str)
    data = json.loads(result)
    assert data["error"]["code"] == "RATE_LIMITED"


def test_rate_limit_diff(diff_snapshot_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _set_exhausted_limiter(monkeypatch)
    result = llmdebug_diff(out_dir=str(diff_snapshot_dir))
    data = json.loads(result)
    assert data["error"]["code"] == "RATE_LIMITED"


def test_rate_limit_rca_status(snapshot_dir_str: str, monkeypatch: pytest.MonkeyPatch) -> None:
    _set_exhausted_limiter(monkeypatch)
    result = llmdebug_rca_status(out_dir=snapshot_dir_str, response_format="json")
    data = json.loads(result)
    assert data["error"]["code"] == "RATE_LIMITED"


def test_rate_limit_rca_history(snapshot_dir_str: str, monkeypatch: pytest.MonkeyPatch) -> None:
    _set_exhausted_limiter(monkeypatch)
    result = llmdebug_rca_history(out_dir=snapshot_dir_str, response_format="json")
    data = json.loads(result)
    assert data["error"]["code"] == "RATE_LIMITED"


def test_rate_limit_rca_advance(snapshot_dir_str: str, monkeypatch: pytest.MonkeyPatch) -> None:
    _set_exhausted_limiter(monkeypatch)
    result = llmdebug_rca_advance(
        action="hypothesize", out_dir=snapshot_dir_str, response_format="json"
    )
    data = json.loads(result)
    assert data["error"]["code"] == "RATE_LIMITED"


# ---------------------------------------------------------------------------
# llmdebug_show: filter_snapshot exception, raw_session normalize, text+rca
# ---------------------------------------------------------------------------


def test_show_filter_snapshot_exception_falls_back(
    snapshot_dir_str: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If filter_snapshot raises, show returns unfiltered snapshot."""
    import llmdebug.detail_filter as detail_filter

    monkeypatch.setattr(
        detail_filter,
        "filter_snapshot",
        lambda _snap, _detail: (_ for _ in ()).throw(RuntimeError("filter boom")),
    )

    result = llmdebug_show(out_dir=snapshot_dir_str, response_format="json", detail="context")
    data = json.loads(result)
    # Should still succeed, just unfiltered
    assert data["metadata"]["status"] == "ok"


def test_show_raw_session_normalize_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """If normalize_snapshot_view raises in raw_session path, falls back to raw snapshot."""
    cfg = SnapshotConfig(out_dir=str(tmp_path), output_format="json")
    payload = {
        "schema_version": "2.0",
        "kind": "llmdebug.debug_session",
        "session": {
            "name": "norm_fail_test",
            "timestamp_utc": "2026-02-10T15:00:00Z",
            "llmdebug_version": "2.1.0",
        },
        "snapshot": {
            "exception": {"type": "ValueError", "message": "boom"},
            "frames": [],
            "traceback": "tb",
            "capture_config": {"frames": 5},
        },
    }
    write_bundle(payload, cfg)

    import llmdebug._debug_session as _ds

    monkeypatch.setattr(
        _ds,
        "normalize_snapshot_view",
        lambda _s: (_ for _ in ()).throw(RuntimeError("normalize failed")),
    )

    result = llmdebug_show(
        out_dir=str(tmp_path),
        raw_session=True,
        response_format="json",
    )
    data = json.loads(result)
    assert data["metadata"]["status"] == "ok"


def test_show_text_format_with_rca(snapshot_dir_str: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """llmdebug_show with response_format='text' and with_rca=True returns JSON payload."""
    result = llmdebug_show(
        out_dir=snapshot_dir_str,
        response_format="text",
        with_rca=True,
    )
    data = json.loads(result)
    assert "snapshot" in data
    assert "rca" in data


def test_show_with_rca_returns_error_when_rca_update_fails(
    snapshot_dir_str: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mcp_server, "_advance_rca_with_snapshot_refs", lambda **_kw: None)
    result = llmdebug_show(
        out_dir=snapshot_dir_str,
        response_format="json",
        with_rca=True,
    )
    _assert_json_error(
        result,
        tool="llmdebug_show",
        code="RCA_STATE_UPDATE_ERROR",
        with_rca=True,
    )


# ---------------------------------------------------------------------------
# llmdebug_git_context: schema error + rca_update_failed for both enhanced paths
# ---------------------------------------------------------------------------


def test_git_context_invalid_evidence_schema(
    snapshot_dir_str: str,
) -> None:
    result = llmdebug_git_context(
        out_dir=snapshot_dir_str,
        evidence_schema="bogus",  # type: ignore[arg-type]
    )
    _assert_json_error(
        result,
        tool="llmdebug_git_context",
        code="INVALID_ARGUMENT",
        message_contains="Invalid evidence_schema",
    )


def test_git_context_rca_update_failed_when_enhanced_none(
    snapshot_dir_str: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When enhanced=None and advance_rca returns None, returns RCA_STATE_UPDATE_ERROR."""
    monkeypatch.setattr(mcp_server, "get_enhanced_git_context", lambda *_a, **_kw: None)
    monkeypatch.setattr(mcp_server, "_advance_rca_with_snapshot_refs", lambda **_kw: None)
    result = llmdebug_git_context(
        out_dir=snapshot_dir_str,
        response_format="json",
        with_rca=True,
    )
    _assert_json_error(
        result,
        tool="llmdebug_git_context",
        code="RCA_STATE_UPDATE_ERROR",
        with_rca=True,
    )


def test_git_context_rca_update_failed_when_enhanced_present(
    snapshot_dir_str: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When enhanced context is available and advance_rca returns None, returns error."""
    fake_enhanced = {
        "head": {"commit": "abc", "branch": "main", "dirty": False, "detached": False},
        "crash_file": {"file_rel": "app.py", "line": 1, "in_repo": True},
        "recent_commits": [],
    }
    monkeypatch.setattr(mcp_server, "get_enhanced_git_context", lambda *_a, **_kw: fake_enhanced)
    monkeypatch.setattr(mcp_server, "_advance_rca_with_snapshot_refs", lambda **_kw: None)
    result = llmdebug_git_context(
        out_dir=snapshot_dir_str,
        response_format="json",
        with_rca=True,
    )
    _assert_json_error(
        result,
        tool="llmdebug_git_context",
        code="RCA_STATE_UPDATE_ERROR",
        with_rca=True,
    )


# ---------------------------------------------------------------------------
# mcp_server.main()
# ---------------------------------------------------------------------------


def test_main_configures_logging_and_runs(monkeypatch: pytest.MonkeyPatch) -> None:
    """main() sets up logging and calls _mcp.run(transport='stdio')."""
    run_calls: list[dict] = []
    monkeypatch.setattr(
        mcp_server._mcp, "run", lambda transport: run_calls.append({"transport": transport})
    )
    mcp_server.main()
    assert run_calls == [{"transport": "stdio"}]
