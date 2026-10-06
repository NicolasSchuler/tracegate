"""Tests for the llmdebug CLI."""

from __future__ import annotations

import shlex
import time
from pathlib import Path

import pytest
from click.testing import CliRunner

from llmdebug import _json as json
from llmdebug.cli import main
from llmdebug.compact_keys import compact_keys
from llmdebug.config import SnapshotConfig
from llmdebug.output import write_bundle
from llmdebug.rca_backend import persist_rca_record
from llmdebug.rca_state import derive_session_id


def test_main_help_lists_quickstart() -> None:
    """Top-level help should expose the offline first-run command."""
    runner = CliRunner(mix_stderr=False)

    result = runner.invoke(main, ["--help"])

    assert result.exit_code == 0
    assert "quickstart" in result.output


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

    # Write 3 snapshots with slight time gaps
    for i in range(3):
        payload = {**sample_snapshot, "name": f"test_{i}"}
        write_bundle(payload, cfg)
        time.sleep(0.05)  # Ensure different mtimes

    return tmp_path


# ---------------------------------------------------------------------------
# show command
# ---------------------------------------------------------------------------


class TestShow:
    def test_show_latest(self, snapshot_dir: Path) -> None:
        runner = CliRunner(mix_stderr=False)
        result = runner.invoke(main, ["show", "--dir", str(snapshot_dir)])
        assert result.exit_code == 0
        assert "ValueError" in result.output
        assert "invalid literal" in result.output

    def test_show_specific_path(self, snapshot_dir: Path) -> None:
        # Find a specific snapshot file
        files = sorted(snapshot_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
        files = [f for f in files if not f.name.startswith("latest")]
        assert len(files) >= 1

        runner = CliRunner(mix_stderr=False)
        result = runner.invoke(main, ["show", str(files[0])])
        assert result.exit_code == 0
        assert "ValueError" in result.output

    def test_show_accepts_snapshot_ref(self, snapshot_dir: Path) -> None:
        runner = CliRunner(mix_stderr=False)
        result = runner.invoke(main, ["show", "previous", "--dir", str(snapshot_dir)])
        assert result.exit_code == 0
        assert "ValueError" in result.output

    def test_show_full_flag(self, snapshot_dir: Path) -> None:
        runner = CliRunner(mix_stderr=False)
        result = runner.invoke(main, ["show", "--full", "--dir", str(snapshot_dir)])
        assert result.exit_code == 0
        # Both frames should appear
        assert "parse_input" in result.output
        assert "run" in result.output

    def test_show_json_flag(self, snapshot_dir: Path) -> None:
        runner = CliRunner(mix_stderr=False)
        result = runner.invoke(main, ["show", "--json", "--dir", str(snapshot_dir)])
        assert result.exit_code == 0
        # Should be valid JSON
        data = json.loads(result.output)
        assert data["exception"]["type"] == "ValueError"

    def test_show_context_json_includes_environment(self, snapshot_dir: Path) -> None:
        runner = CliRunner(mix_stderr=False)
        result = runner.invoke(
            main,
            ["show", "--json", "--detail", "context", "--dir", str(snapshot_dir)],
        )
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["env"]["cwd"] == "/code"
        assert data["pytest"]["nodeid"] == "tests/test_app.py::test_parse"

    def test_show_context_text_includes_context_sections(self, snapshot_dir: Path) -> None:
        runner = CliRunner(mix_stderr=False)
        result = runner.invoke(main, ["show", "--detail", "context", "--dir", str(snapshot_dir)])
        assert result.exit_code == 0
        assert "## Pytest" in result.output
        assert "## Environment" in result.output
        assert "cwd: /code" in result.output

    def test_show_tolerates_non_numeric_crash_line(
        self, tmp_path: Path, sample_snapshot: dict
    ) -> None:
        cfg = SnapshotConfig(out_dir=str(tmp_path), output_format="json")
        malformed = {
            **sample_snapshot,
            "frames": [{**sample_snapshot["frames"][0], "line": "?"}],
            "crash_frame_index": 0,
        }
        write_bundle(malformed, cfg)

        runner = CliRunner(mix_stderr=False)
        result = runner.invoke(main, ["show", "--dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "Frame #0" in result.output

    def test_show_raw_session_returns_envelope(self, tmp_path: Path) -> None:
        cfg = SnapshotConfig(out_dir=str(tmp_path), output_format="json")
        payload = {
            "schema_version": "2.0",
            "kind": "llmdebug.debug_session",
            "session": {
                "name": "raw_test",
                "timestamp_utc": "2026-02-10T13:00:00Z",
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

        runner = CliRunner(mix_stderr=False)
        result = runner.invoke(main, ["show", "--raw-session", "--dir", str(tmp_path)])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["kind"] == "llmdebug.debug_session"
        assert data["session"]["name"] == "raw_test"
        assert data["snapshot"]["exception"]["type"] == "ValueError"

    def test_show_raw_session_uses_latest_pointer_when_only_latest_json_exists(
        self, tmp_path: Path
    ) -> None:
        payload = {
            "schema_version": "2.0",
            "kind": "llmdebug.debug_session",
            "session": {
                "name": "raw_only",
                "timestamp_utc": "2026-02-10T13:00:00Z",
                "llmdebug_version": "2.1.0",
            },
            "snapshot": {
                "exception": {"type": "ValueError", "message": "boom"},
                "frames": [],
            },
        }
        (tmp_path / "latest.json").write_text(json.dumps(payload), encoding="utf-8")

        runner = CliRunner(mix_stderr=False)
        result = runner.invoke(main, ["show", "--raw-session", "--dir", str(tmp_path)])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["kind"] == "llmdebug.debug_session"
        assert data["session"]["name"] == "raw_only"

    def test_show_prefers_latest_pointer_over_newest_snapshot_file(self, tmp_path: Path) -> None:
        older = {
            "schema_version": "1.0",
            "name": "older_pointed",
            "exception": {"type": "ValueError", "message": "older"},
            "frames": [
                {"file": "/code/older.py", "file_rel": "older.py", "line": 1, "function": "f"}
            ],
            "crash_frame_index": 0,
        }
        newer = {
            "schema_version": "1.0",
            "name": "newer_file",
            "exception": {"type": "TypeError", "message": "newer"},
            "frames": [
                {"file": "/code/newer.py", "file_rel": "newer.py", "line": 2, "function": "g"}
            ],
            "crash_frame_index": 0,
        }
        old_path = tmp_path / "snap_old.json"
        old_path.write_text(json.dumps(older), encoding="utf-8")
        time.sleep(0.05)
        new_path = tmp_path / "snap_new.json"
        new_path.write_text(json.dumps(newer), encoding="utf-8")
        (tmp_path / "latest.json").write_text(json.dumps(older), encoding="utf-8")

        runner = CliRunner(mix_stderr=False)
        result = runner.invoke(main, ["show", "--json", "--dir", str(tmp_path)])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["name"] == "older_pointed"

    def test_show_no_snapshot(self, tmp_path: Path) -> None:
        runner = CliRunner(mix_stderr=False)
        result = runner.invoke(main, ["show", "--dir", str(tmp_path)])
        assert result.exit_code == 1
        assert "No snapshots found" in result.stderr
        expected = shlex.join(("llmdebug", "quickstart", "--dir", str(tmp_path)))
        assert f"Try: {expected}" in result.stderr

    def test_show_uses_latest_pointer_when_only_latest_json_exists(self, tmp_path: Path) -> None:
        payload = {
            "schema_version": "1.0",
            "name": "latest_only",
            "exception": {"type": "ValueError", "message": "boom"},
            "frames": [
                {
                    "file": "/code/app.py",
                    "file_rel": "app.py",
                    "line": 3,
                    "function": "run",
                }
            ],
            "crash_frame_index": 0,
        }
        (tmp_path / "latest.json").write_text(json.dumps(payload), encoding="utf-8")

        runner = CliRunner(mix_stderr=False)
        result = runner.invoke(main, ["show", "--dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "ValueError" in result.output
        assert "app.py:3" in result.output

    def test_show_default_invocation(self, snapshot_dir: Path) -> None:
        """Bare `llmdebug` (no subcommand) should equal `llmdebug show`."""
        runner = CliRunner(mix_stderr=False)
        result = runner.invoke(main, ["--dir", str(snapshot_dir)])
        assert result.exit_code == 0
        assert "ValueError" in result.output

    def test_show_handles_invalid_crash_frame_index(
        self,
        tmp_path: Path,
        sample_snapshot: dict,
    ) -> None:
        malformed = {**sample_snapshot, "crash_frame_index": "abc"}
        snapshot_path = tmp_path / "bad.json"
        snapshot_path.write_text(json.dumps(malformed), encoding="utf-8")
        (tmp_path / "latest.json").write_text(json.dumps(malformed), encoding="utf-8")

        runner = CliRunner(mix_stderr=False)
        result = runner.invoke(main, ["show", "--dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "parse_input" in result.output

    def test_show_falls_back_to_file_when_file_rel_missing(
        self,
        tmp_path: Path,
        sample_snapshot: dict,
    ) -> None:
        frame0 = dict(sample_snapshot["frames"][0])
        frame0["file_rel"] = None
        malformed = {**sample_snapshot, "frames": [frame0]}
        snapshot_path = tmp_path / "bad.json"
        snapshot_path.write_text(json.dumps(malformed), encoding="utf-8")
        (tmp_path / "latest.json").write_text(json.dumps(malformed), encoding="utf-8")

        runner = CliRunner(mix_stderr=False)
        result = runner.invoke(main, ["show", "--dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "/code/app.py:42" in result.output
        assert "None:42" not in result.output

    def test_show_hides_frame_hint_in_non_tty(self, snapshot_dir: Path) -> None:
        runner = CliRunner(mix_stderr=False)
        result = runner.invoke(main, ["show", "--dir", str(snapshot_dir)])
        assert result.exit_code == 0
        assert "more frame" not in result.output

    def test_show_displays_frame_hint_in_tty(self, snapshot_dir: Path, monkeypatch) -> None:
        monkeypatch.setattr("llmdebug._cli_display.HAS_RICH", False)
        monkeypatch.setattr("llmdebug._cli_display._stdout_is_tty", lambda: True)
        runner = CliRunner(mix_stderr=False)
        result = runner.invoke(main, ["show", "--dir", str(snapshot_dir)])
        assert result.exit_code == 0
        assert "more frame" in result.output


# ---------------------------------------------------------------------------
# list command
# ---------------------------------------------------------------------------


class TestList:
    def test_list_snapshots(self, snapshot_dir: Path) -> None:
        runner = CliRunner(mix_stderr=False)
        result = runner.invoke(main, ["list", "--dir", str(snapshot_dir)])
        assert result.exit_code == 0
        # Should show test_0, test_1, test_2 snapshot names
        assert "test_" in result.output

    def test_list_limit(self, snapshot_dir: Path) -> None:
        runner = CliRunner(mix_stderr=False)
        result = runner.invoke(main, ["list", "--limit", "2", "--dir", str(snapshot_dir)])
        assert result.exit_code == 0
        assert "2 shown" in result.output
        assert "UTC" in result.output

    def test_list_empty_dir(self, tmp_path: Path) -> None:
        empty_dir = tmp_path / "custom snapshot store"
        runner = CliRunner(mix_stderr=False)
        result = runner.invoke(main, ["list", "--dir", str(empty_dir)])
        assert result.exit_code == 0
        assert "No snapshots found" in result.output
        expected = shlex.join(("llmdebug", "quickstart", "--dir", str(empty_dir)))
        assert f"Try: {expected}" in result.output

    def test_list_truncates_long_fields(self, monkeypatch: pytest.MonkeyPatch) -> None:
        now = time.time()

        def _fake_find_all_snapshots(
            _out_dir: str,
            *,
            limit: int = 10,
            search: str | None = None,
        ) -> list[dict[str, object]]:
            del search
            entries = [
                {
                    "mtime": now,
                    "name": "test_" + ("x" * 80),
                    "exc_type": "VeryLongExceptionTypeName" + ("y" * 40),
                    "file": "/a/very/long/path/for/snapshot/location/module_with_long_name.py",
                    "line": 1234,
                }
            ]
            return entries[:limit]

        monkeypatch.setattr("llmdebug.cli._find_all_snapshots_impl", _fake_find_all_snapshots)
        runner = CliRunner(mix_stderr=False)
        result = runner.invoke(main, ["list", "--dir", "/unused"])
        assert result.exit_code == 0
        rows = [line for line in result.output.splitlines() if line.strip().startswith("1.")]
        assert rows
        assert "..." in rows[0]

    def test_list_falls_back_to_file_when_file_rel_missing(
        self,
        tmp_path: Path,
        sample_snapshot: dict,
    ) -> None:
        cfg = SnapshotConfig(out_dir=str(tmp_path), output_format="json")
        frame0 = dict(sample_snapshot["frames"][0])
        frame0["file_rel"] = None
        payload = {**sample_snapshot, "frames": [frame0]}
        write_bundle(payload, cfg)

        runner = CliRunner(mix_stderr=False)
        result = runner.invoke(main, ["list", "--dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "/code/app.py:42" in result.output

    def test_list_empty_dir_shows_guidance_in_tty(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("llmdebug.cli._stdout_is_tty", lambda: True)
        runner = CliRunner(mix_stderr=False)
        result = runner.invoke(main, ["list", "--dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "Run a failing test to create a snapshot" in result.output

    def test_list_search_filters_snapshot_metadata(
        self, tmp_path: Path, sample_snapshot: dict
    ) -> None:
        cfg = SnapshotConfig(out_dir=str(tmp_path), output_format="json", max_snapshots=50)
        write_bundle(
            {
                **sample_snapshot,
                "name": "key_snapshot",
                "exception": {"type": "KeyError", "message": "missing timeout"},
            },
            cfg,
        )
        time.sleep(0.05)
        write_bundle(
            {
                **sample_snapshot,
                "name": "value_snapshot",
                "exception": {"type": "ValueError", "message": "bad int"},
            },
            cfg,
        )

        runner = CliRunner(mix_stderr=False)
        result = runner.invoke(
            main,
            ["list", "--search", "KeyError", "--dir", str(tmp_path)],
        )
        assert result.exit_code == 0
        assert "KeyError" in result.output
        assert "ValueError" not in result.output


# ---------------------------------------------------------------------------
# clean command
# ---------------------------------------------------------------------------


class TestClean:
    def test_clean_keep(self, snapshot_dir: Path) -> None:
        runner = CliRunner(mix_stderr=False)
        result = runner.invoke(main, ["clean", "--keep", "1", "--yes", "--dir", str(snapshot_dir)])
        assert result.exit_code == 0
        assert "Deleted" in result.output
        # Should have 1 snapshot left (+ latest symlink)
        remaining = [
            p
            for p in snapshot_dir.iterdir()
            if p.is_file()
            and p.suffix == ".json"
            and not p.name.startswith("latest")
            and not p.name.endswith(".tmp")
            and ".traceback" not in p.name
        ]
        assert len(remaining) == 1

    def test_clean_all(self, snapshot_dir: Path) -> None:
        runner = CliRunner(mix_stderr=False)
        result = runner.invoke(main, ["clean", "--all", "--yes", "--dir", str(snapshot_dir)])
        assert result.exit_code == 0
        assert "Deleted" in result.output
        # Directory may still exist but should have no snapshot files
        remaining = [
            p for p in snapshot_dir.iterdir() if p.is_file() and p.suffix in {".json", ".toon"}
        ]
        assert len(remaining) == 0

    def test_clean_all_keeps_the_lock_and_drops_index_rows(self, snapshot_dir: Path) -> None:
        """RCA-21: --all deleted a lock another process may hold and left stale index rows."""
        from llmdebug.state_db import count_rca_records, count_snapshot_records

        lock = snapshot_dir / ".llmdebug.lock"
        lock.touch()
        assert count_snapshot_records(str(snapshot_dir)) > 0

        result = CliRunner(mix_stderr=False).invoke(
            main, ["clean", "--all", "--yes", "--dir", str(snapshot_dir)]
        )

        assert result.exit_code == 0
        assert lock.exists()
        assert count_snapshot_records(str(snapshot_dir)) == 0
        assert (snapshot_dir / "llmdebug_state.sqlite3").exists()  # RCA history is kept
        assert count_rca_records(str(snapshot_dir)) == 0

    def test_clean_all_reports_deletions_when_the_index_is_corrupt(
        self, snapshot_dir: Path
    ) -> None:
        """RCA-03: a corrupt database must not swallow a completed deletion."""
        db = snapshot_dir / "llmdebug_state.sqlite3"
        assert db.exists()
        db.write_bytes(b"this is not a sqlite database")

        result = CliRunner(mix_stderr=False).invoke(
            main, ["clean", "--all", "--yes", "--dir", str(snapshot_dir)]
        )

        assert result.exit_code == 0
        assert "Deleted" in result.output
        assert "snapshot index cleanup failed" in result.stderr
        remaining = [
            p for p in snapshot_dir.iterdir() if p.is_file() and p.suffix in {".json", ".toon"}
        ]
        assert remaining == []

    def test_clean_purge_state_deletes_the_database(self, snapshot_dir: Path) -> None:
        (snapshot_dir / "rca_attempts.jsonl").write_text("{}\n", encoding="utf-8")

        result = CliRunner(mix_stderr=False).invoke(
            main, ["clean", "--all", "--purge-state", "--yes", "--dir", str(snapshot_dir)]
        )

        assert result.exit_code == 0
        assert not list(snapshot_dir.glob("llmdebug_state.sqlite3*"))
        assert not (snapshot_dir / "rca_attempts.jsonl").exists()

    def test_clean_survives_a_file_vanishing_mid_run(
        self, snapshot_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """RCA-21: sorting called stat() on files another process had just deleted."""
        import llmdebug.snapshot_index as snapshot_index

        real = snapshot_index.snapshot_files

        def _with_ghost(out_dir: object) -> list[Path]:
            return [*real(out_dir), snapshot_dir / "ghost.json"]  # type: ignore[arg-type]

        monkeypatch.setattr(snapshot_index, "snapshot_files", _with_ghost)
        result = CliRunner(mix_stderr=False).invoke(
            main, ["clean", "--keep", "1", "--yes", "--dir", str(snapshot_dir)]
        )

        assert result.exit_code == 0
        assert "Deleted" in result.output

    def test_clean_confirmation(self, snapshot_dir: Path) -> None:
        runner = CliRunner(mix_stderr=False)
        result = runner.invoke(
            main, ["clean", "--keep", "1", "--dir", str(snapshot_dir)], input="y\n"
        )
        assert result.exit_code == 0
        assert "Deleted" in result.output

    def test_clean_rejects_negative_keep(self, snapshot_dir: Path) -> None:
        runner = CliRunner(mix_stderr=False)
        result = runner.invoke(main, ["clean", "--keep", "-1", "--yes", "--dir", str(snapshot_dir)])
        assert result.exit_code != 0
        assert "Invalid value for '--keep'" in result.stderr

    def test_clean_missing_directory_prints_error_to_stderr(self, tmp_path: Path) -> None:
        missing_dir = tmp_path / "missing"
        runner = CliRunner(mix_stderr=False)
        result = runner.invoke(main, ["clean", "--dir", str(missing_dir)])
        assert result.exit_code == 1  # RCA-15
        assert "Directory not found" in result.stderr


# ---------------------------------------------------------------------------
# export command
# ---------------------------------------------------------------------------


class TestExport:
    def test_export_json_writes_shareable_payload_with_rca(
        self, tmp_path: Path, sample_snapshot: dict
    ) -> None:
        cfg = SnapshotConfig(out_dir=str(tmp_path), output_format="json", max_snapshots=50)
        write_bundle(sample_snapshot, cfg)

        session_id = derive_session_id(sample_snapshot, out_dir=str(tmp_path), fallback="export")
        persist_rca_record(
            out_dir=str(tmp_path),
            record={
                "attempt_id": "a1",
                "session_id": session_id,
                "updated_at": "2026-03-16T10:00:00Z",
                "created_at": "2026-03-16T10:00:00Z",
                "failure_signature": "ValueError:test",
                "state": "hypothesized",
                "evidence_refs": ["exception:ValueError"],
                "hypothesis": "timeout key missing",
            },
        )

        output_path = tmp_path / "artifact.json"
        runner = CliRunner(mix_stderr=False)
        result = runner.invoke(
            main,
            ["export", "--format", "json", "--output", str(output_path), "--dir", str(tmp_path)],
        )
        assert result.exit_code == 0
        assert output_path.exists()
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        assert payload["kind"] == "llmdebug.export"
        assert payload["export_mode"] == "full"
        assert payload["export_redaction_profile"] is None
        assert payload["snapshot"]["exception"]["type"] == "ValueError"
        assert payload["rca"]["status"]["state"] == "hypothesized"
        assert payload["frames"][0]["is_crash"] is True

    def test_export_html_writes_single_file_artifact(
        self, tmp_path: Path, sample_snapshot: dict
    ) -> None:
        cfg = SnapshotConfig(out_dir=str(tmp_path), output_format="json", max_snapshots=50)
        older = {**sample_snapshot, "name": "older_export"}
        newer = {
            **sample_snapshot,
            "name": "newer_export",
            "exception": {"type": "ValueError", "message": "newer message"},
        }
        write_bundle(older, cfg)
        time.sleep(0.05)
        write_bundle(newer, cfg)

        output_path = tmp_path / "artifact.html"
        runner = CliRunner(mix_stderr=False)
        result = runner.invoke(
            main,
            [
                "export",
                "--format",
                "html",
                "--compare",
                "previous",
                "--output",
                str(output_path),
                "--dir",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 0
        html = output_path.read_text(encoding="utf-8")
        assert "<!DOCTYPE html>" in html
        assert "llmdebug export" in html
        assert "Embedded JSON" in html
        assert "Comparison" in html

    def test_export_html_renders_retry_timeline_when_present(
        self, tmp_path: Path, sample_snapshot: dict
    ) -> None:
        cfg = SnapshotConfig(out_dir=str(tmp_path), output_format="json", max_snapshots=50)
        payload = {
            **sample_snapshot,
            "retry_summaries": [
                {
                    "attempt": 1,
                    "hypothesis_tested": "bad input normalization",
                    "evidence_used": "traceback + crash locals",
                    "outcome": "tests_still_failing",
                    "new_constraint": "value remains quoted",
                }
            ],
        }
        write_bundle(payload, cfg)

        output_path = tmp_path / "retry.html"
        runner = CliRunner(mix_stderr=False)
        result = runner.invoke(
            main,
            ["export", "--output", str(output_path), "--dir", str(tmp_path)],
        )
        assert result.exit_code == 0
        html = output_path.read_text(encoding="utf-8")
        assert "Retry Timeline" in html
        assert "bad input normalization" in html

    def test_export_default_output_uses_immutable_latest_snapshot_file(
        self, tmp_path: Path, sample_snapshot: dict
    ) -> None:
        cfg = SnapshotConfig(out_dir=str(tmp_path), output_format="json", max_snapshots=50)
        older = {**sample_snapshot, "name": "older_default_export"}
        newer = {**sample_snapshot, "name": "newer_default_export"}
        write_bundle(older, cfg)
        time.sleep(0.05)
        write_bundle(newer, cfg)
        snapshot_files = sorted(
            (
                path
                for path in tmp_path.glob("*.json")
                if not path.name.startswith("latest") and ".report." not in path.name
            ),
            key=lambda path: path.stat().st_mtime,
        )
        latest_snapshot = snapshot_files[-1]

        runner = CliRunner(mix_stderr=False)
        result = runner.invoke(
            main,
            ["export", "--format", "json", "--dir", str(tmp_path)],
        )

        assert result.exit_code == 0
        output_path = latest_snapshot.with_name(f"{latest_snapshot.stem}.report.json")
        assert output_path.exists()
        assert not (tmp_path / "latest.report.json").exists()
        payload = json.loads(output_path.read_text(encoding="utf-8"))
        assert payload["source"]["snapshot_path"] == str(latest_snapshot)
        assert payload["source"]["snapshot_file"] == latest_snapshot.name

    def test_export_bundle_manifest_tracks_requested_snapshot_refs(
        self, tmp_path: Path, sample_snapshot: dict
    ) -> None:
        cfg = SnapshotConfig(out_dir=str(tmp_path), output_format="json", max_snapshots=50)
        older = {**sample_snapshot, "name": "older_ref"}
        newer = {**sample_snapshot, "name": "newer_ref"}
        write_bundle(older, cfg)
        time.sleep(0.05)
        write_bundle(newer, cfg)

        bundle_dir = tmp_path / "bundle_refs"
        runner = CliRunner(mix_stderr=False)
        result = runner.invoke(
            main,
            [
                "export",
                "previous",
                "--bundle-dir",
                str(bundle_dir),
                "--compare",
                "latest",
                "--dir",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 0
        manifest = json.loads((bundle_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["primary_snapshot"]["ref"] == "previous"
        assert manifest["compare_snapshot"]["ref"] == "latest"

    def test_export_redacted_mode_rejects_the_dev_profile(
        self, tmp_path: Path, sample_snapshot: dict
    ) -> None:
        """CAP-06: dev redacts nothing, so a "redacted" export with it is a usage error."""
        write_bundle(sample_snapshot, SnapshotConfig(out_dir=str(tmp_path), output_format="json"))
        output_path = tmp_path / "out.json"

        result = CliRunner(mix_stderr=False).invoke(
            main,
            [
                "export",
                "--format",
                "json",
                "--mode",
                "redacted",
                "--redaction-profile",
                "dev",
                "--output",
                str(output_path),
                "--dir",
                str(tmp_path),
            ],
        )

        assert result.exit_code == 2
        assert "dev redacts nothing" in result.stderr
        assert not output_path.exists()

    def test_export_bundle_defaults_to_redacted_ci_and_writes_compare_artifacts(
        self, tmp_path: Path, sample_snapshot: dict
    ) -> None:
        cfg = SnapshotConfig(out_dir=str(tmp_path), output_format="json", max_snapshots=50)
        older = {**sample_snapshot, "name": "older_bundle"}
        newer = {
            **sample_snapshot,
            "name": "newer_bundle",
            "exception": {
                "type": "ValueError",
                "message": "invalid email for alice@example.com",
            },
            "traceback": (
                "Traceback (most recent call last):\n"
                "ValueError: invalid email for alice@example.com"
            ),
            "frames": [
                {
                    **sample_snapshot["frames"][0],
                    "locals": {"user_email": "alice@example.com"},
                }
            ],
            "retry_summaries": [
                {
                    "attempt": 1,
                    "hypothesis_tested": "email parsing edge case",
                    "evidence_used": "crash frame locals",
                    "outcome": "tests_still_failing",
                    "new_constraint": "email leaked into value parser",
                }
            ],
        }
        write_bundle(older, cfg)
        time.sleep(0.05)
        write_bundle(newer, cfg)

        bundle_dir = tmp_path / "bundle"
        runner = CliRunner(mix_stderr=False)
        result = runner.invoke(
            main,
            [
                "export",
                "--format",
                "json",
                "--bundle-dir",
                str(bundle_dir),
                "--compare",
                "previous",
                "--dir",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 0
        assert (bundle_dir / "report.json").exists()
        assert (bundle_dir / "compare.json").exists()
        assert (bundle_dir / "manifest.json").exists()
        assert (bundle_dir / "index.html").exists()

        report_text = (bundle_dir / "report.json").read_text(encoding="utf-8")
        compare_text = (bundle_dir / "compare.json").read_text(encoding="utf-8")
        manifest_text = (bundle_dir / "manifest.json").read_text(encoding="utf-8")
        payload = json.loads(report_text)
        compare_payload = json.loads(compare_text)
        manifest = json.loads(manifest_text)
        index_html = (bundle_dir / "index.html").read_text(encoding="utf-8")

        assert payload["export_mode"] == "redacted"
        assert payload["export_redaction_profile"] == "ci"
        assert "[REDACTED]" in payload["snapshot"]["exception"]["message"]
        assert "alice@example.com" in payload["snapshot"]["traceback"]
        assert payload["retry_summaries"][0]["hypothesis_tested"] == "email parsing edge case"
        assert payload["source"]["out_dir"] is None
        assert payload["source"]["snapshot_path"] == payload["source"]["snapshot_file"]
        assert (
            compare_payload["source"]["snapshot_path"] == compare_payload["source"]["snapshot_file"]
        )
        assert compare_payload["comparison"]["ref"] == "latest"
        assert manifest["artifacts"]["primary"] == "report.json"
        assert manifest["artifacts"]["compare"] == "compare.json"
        assert manifest["retry_summary_count"] == 1
        assert manifest["primary_snapshot"]["path"] == payload["source"]["snapshot_file"]
        assert manifest["compare_snapshot"]["path"] == compare_payload["source"]["snapshot_file"]
        assert "Primary artifact" in index_html
        assert "Compare artifact" in index_html
        assert str(tmp_path) not in report_text
        assert str(tmp_path) not in compare_text
        assert str(tmp_path) not in manifest_text

    def test_export_rejects_output_and_bundle_dir_together(
        self, tmp_path: Path, sample_snapshot: dict
    ) -> None:
        cfg = SnapshotConfig(out_dir=str(tmp_path), output_format="json", max_snapshots=50)
        write_bundle(sample_snapshot, cfg)

        runner = CliRunner(mix_stderr=False)
        result = runner.invoke(
            main,
            [
                "export",
                "--output",
                str(tmp_path / "artifact.html"),
                "--bundle-dir",
                str(tmp_path / "bundle"),
                "--dir",
                str(tmp_path),
            ],
        )
        assert result.exit_code != 0
        assert "Use either --output or --bundle-dir" in result.stderr


# ---------------------------------------------------------------------------
# frames command
# ---------------------------------------------------------------------------


class TestFrames:
    def test_frames_all(self, snapshot_dir: Path) -> None:
        runner = CliRunner(mix_stderr=False)
        result = runner.invoke(main, ["frames", "--dir", str(snapshot_dir)])
        assert result.exit_code == 0
        assert "parse_input" in result.output
        assert "run" in result.output

    def test_frames_index(self, snapshot_dir: Path) -> None:
        runner = CliRunner(mix_stderr=False)
        result = runner.invoke(main, ["frames", "--index", "0", "--dir", str(snapshot_dir)])
        assert result.exit_code == 0
        assert "parse_input" in result.output

    def test_frames_accepts_snapshot_ref(self, snapshot_dir: Path) -> None:
        runner = CliRunner(mix_stderr=False)
        result = runner.invoke(
            main,
            ["frames", "previous", "--index", "0", "--dir", str(snapshot_dir)],
        )
        assert result.exit_code == 0
        assert "parse_input" in result.output

    def test_frames_plain_sections_preserved(
        self, snapshot_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr("llmdebug._cli_display.HAS_RICH", False)
        runner = CliRunner(mix_stderr=False)
        result = runner.invoke(main, ["frames", "--index", "0", "--dir", str(snapshot_dir)])
        assert result.exit_code == 0
        assert "Source:" in result.output
        assert "Locals:" in result.output

    def test_frames_rejects_malformed_frame(self, tmp_path: Path, sample_snapshot: dict) -> None:
        malformed = {**sample_snapshot, "frames": [sample_snapshot["frames"][0], "bad-frame"]}
        snapshot_path = tmp_path / "bad.json"
        snapshot_path.write_text(json.dumps(malformed), encoding="utf-8")
        (tmp_path / "latest.json").write_text(json.dumps(malformed), encoding="utf-8")

        runner = CliRunner(mix_stderr=False)
        result = runner.invoke(main, ["frames", "--index", "1", "--dir", str(tmp_path)])
        assert result.exit_code == 1
        assert "malformed" in result.stderr.lower()

    def test_frames_skips_malformed_frame_when_listing(
        self,
        tmp_path: Path,
        sample_snapshot: dict,
    ) -> None:
        malformed = {
            **sample_snapshot,
            "frames": [
                sample_snapshot["frames"][0],
                "bad-frame",
                sample_snapshot["frames"][1],
            ],
        }
        snapshot_path = tmp_path / "bad.json"
        snapshot_path.write_text(json.dumps(malformed), encoding="utf-8")
        (tmp_path / "latest.json").write_text(json.dumps(malformed), encoding="utf-8")

        runner = CliRunner(mix_stderr=False)
        result = runner.invoke(main, ["frames", "--dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "parse_input" in result.output
        assert "run" in result.output


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_load_snapshot_from_file_compact(self, tmp_path: Path) -> None:
        """Compact JSON should be auto-expanded when loaded."""
        from llmdebug.cli import _load_snapshot_from_file

        snapshot = {
            "schema_version": "1.0",
            "name": "test",
            "exception": {"type": "E"},
            "frames": [],
        }
        compacted = compact_keys(snapshot)
        assert isinstance(compacted, dict)
        compacted["__llmdebug_compact__"] = 1
        path = tmp_path / "compact.json"
        path.write_text(json.dumps(compacted), encoding="utf-8")

        loaded = _load_snapshot_from_file(path)
        # Keys should be expanded
        assert "schema_version" in loaded
        assert "name" in loaded
        assert "__llmdebug_compact__" not in loaded

    def test_find_all_snapshots_sorted(self, snapshot_dir: Path) -> None:
        """Snapshots should be sorted newest-first."""
        from llmdebug.cli import _find_all_snapshots

        entries = _find_all_snapshots(str(snapshot_dir))
        assert len(entries) == 3
        # Verify newest first
        for i in range(len(entries) - 1):
            assert entries[i]["mtime"] >= entries[i + 1]["mtime"]

    def test_format_timestamp(self) -> None:
        from llmdebug.cli import _format_timestamp

        assert _format_timestamp("2026-02-07T21:00:00Z") == "2026-02-07 21:00:00"
        assert _format_timestamp(None) == "?"
        assert _format_timestamp("bad") == "bad"

    def test_iter_valid_frame_items_handles_non_list_frames(self) -> None:
        from llmdebug._cli_display import _iter_valid_frame_items

        assert _iter_valid_frame_items({"frames": None}) == []
        assert _iter_valid_frame_items({"frames": "not-a-list"}) == []
        assert _iter_valid_frame_items({"frames": [{"function": "ok"}, "bad"]}) == [
            (0, {"function": "ok"})
        ]

    def test_format_enhanced_git_context_text_sections(self) -> None:
        from llmdebug._formatting import format_enhanced_git_context_text

        context = {
            "head": {
                "commit": "abc1234",
                "branch": "main",
                "dirty": True,
                "detached": False,
            },
            "crash_file": {
                "file_rel": "app.py",
                "line": 42,
                "in_repo": True,
                "blame": {"commit": "def5678", "author": "Alice"},
            },
            "recent_commits": [
                {
                    "commit": "def5678",
                    "author": "Alice",
                    "author_time": "2026-02-10T12:00:00Z",
                    "summary": "Fix crash",
                    "files_changed": 1,
                    "insertions": 3,
                    "deletions": 1,
                },
                "skip-me",
            ],
            "crash_file_diffstat": {
                "files_changed": 1,
                "insertions": 3,
                "deletions": 1,
                "touched": True,
            },
            "_git_warnings": ["git timeout"],
        }

        output = format_enhanced_git_context_text(context)
        assert "## Git Head" in output
        assert "## Crash File" in output
        assert "## Recent Commits" in output
        assert "## Crash File Diffstat" in output
        assert "## Warnings" in output
        assert "git timeout" in output


# ---------------------------------------------------------------------------
# diff command
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
                "locals": {"value": "'abc'", "x": 10},
                "locals_meta": {"value": {"type": "str"}, "x": {"type": "int"}},
            }
        ],
        "crash_frame_index": 0,
        "git": {"commit": "aaa1111", "branch": "main", "dirty": False},
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
                "locals": {"value": 42, "x": 10},
                "locals_meta": {"value": {"type": "int"}, "x": {"type": "int"}},
            }
        ],
        "crash_frame_index": 0,
        "git": {"commit": "bbb2222", "branch": "main", "dirty": True},
        "pytest": {"nodeid": "tests/test_app.py::test_parse"},
    }
    old_path = tmp_path / "snap_old.json"
    old_path.write_text(json.dumps(snap_old), encoding="utf-8")
    time.sleep(0.05)
    new_path = tmp_path / "snap_new.json"
    new_path.write_text(json.dumps(snap_new), encoding="utf-8")
    # latest.json
    (tmp_path / "latest.json").write_text(json.dumps(snap_new), encoding="utf-8")
    return tmp_path


class TestDiff:
    def test_diff_latest_vs_previous(self, diff_snapshot_dir: Path) -> None:
        runner = CliRunner(mix_stderr=False)
        result = runner.invoke(main, ["diff", "--dir", str(diff_snapshot_dir)])
        assert result.exit_code == 0
        assert "ValueError" in result.output or "TypeError" in result.output

    def test_diff_by_path(self, diff_snapshot_dir: Path) -> None:
        old = str(diff_snapshot_dir / "snap_old.json")
        new = str(diff_snapshot_dir / "snap_new.json")
        runner = CliRunner(mix_stderr=False)
        result = runner.invoke(main, ["diff", old, new, "--dir", str(diff_snapshot_dir)])
        assert result.exit_code == 0

    def test_diff_by_path_outside_snapshot_dir(self, diff_snapshot_dir: Path) -> None:
        old = str(diff_snapshot_dir / "snap_old.json")
        new = str(diff_snapshot_dir / "snap_new.json")
        other_dir = diff_snapshot_dir.parent / "other"
        other_dir.mkdir()
        runner = CliRunner(mix_stderr=False)
        result = runner.invoke(main, ["diff", old, new, "--dir", str(other_dir)])
        assert result.exit_code == 0

    def test_diff_json_output(self, diff_snapshot_dir: Path) -> None:
        runner = CliRunner(mix_stderr=False)
        result = runner.invoke(main, ["diff", "--json", "--dir", str(diff_snapshot_dir)])
        assert result.exit_code == 0
        data = json.loads(result.output)
        assert data["exception"]["changed"] is True
        assert data["location"]["changed"] is True

    def test_diff_no_snapshots(self, tmp_path: Path) -> None:
        runner = CliRunner(mix_stderr=False)
        result = runner.invoke(main, ["diff", "--dir", str(tmp_path)])
        assert result.exit_code == 1

    def test_diff_one_snapshot(self, tmp_path: Path) -> None:
        snap = {
            "schema_version": "1.0",
            "name": "only",
            "exception": {"type": "E", "message": ""},
            "frames": [],
            "crash_frame_index": 0,
        }
        (tmp_path / "only.json").write_text(json.dumps(snap), encoding="utf-8")
        (tmp_path / "latest.json").write_text(json.dumps(snap), encoding="utf-8")
        runner = CliRunner(mix_stderr=False)
        result = runner.invoke(main, ["diff", "--dir", str(tmp_path)])
        assert result.exit_code == 1
        assert "at least 2" in result.stderr


# ---------------------------------------------------------------------------
# git-context command
# ---------------------------------------------------------------------------


class TestGitContext:
    def test_git_context_json_output(self, snapshot_dir: Path, monkeypatch) -> None:
        import llmdebug.git_context as git_context_module

        monkeypatch.setattr(
            git_context_module,
            "get_enhanced_git_context",
            lambda *_args, **_kwargs: {
                "head": {"commit": "abc1234", "branch": "main", "dirty": False, "detached": False},
                "crash_file": {"file_rel": "app.py", "line": 42, "in_repo": True},
            },
        )
        runner = CliRunner(mix_stderr=False)
        result = runner.invoke(main, ["git-context", "--json", "--dir", str(snapshot_dir)])
        assert result.exit_code == 0
        payload = json.loads(result.output)
        assert payload["head"]["commit"] == "abc1234"
        assert payload["crash_file"]["file_rel"] == "app.py"

    def test_git_context_failure_exits_1_without_a_traceback(
        self, snapshot_dir: Path, monkeypatch
    ) -> None:
        """RCA-10: the CLI printed a raw traceback when git context raised."""
        import llmdebug.git_context as git_context_module

        def _boom(*_args, **_kwargs):
            raise OSError("git exploded")

        monkeypatch.setattr(git_context_module, "get_enhanced_git_context", _boom)
        result = CliRunner(mix_stderr=False).invoke(
            main, ["git-context", "--dir", str(snapshot_dir)]
        )

        assert result.exit_code == 1
        assert "Enhanced git context failed: OSError: git exploded" in result.stderr
        assert result.exception is None or isinstance(result.exception, SystemExit)

    def test_git_context_text_output(self, snapshot_dir: Path, monkeypatch) -> None:
        import llmdebug.git_context as git_context_module

        monkeypatch.setattr(
            git_context_module,
            "get_enhanced_git_context",
            lambda *_args, **_kwargs: {
                "head": {"commit": "abc1234", "branch": "main", "dirty": True, "detached": False},
                "crash_file": {
                    "file_rel": "app.py",
                    "line": 42,
                    "in_repo": True,
                    "blame": {"commit": "def5678", "author": "Alice"},
                },
            },
        )
        runner = CliRunner(mix_stderr=False)
        result = runner.invoke(main, ["git-context", "--dir", str(snapshot_dir)])
        assert result.exit_code == 0
        assert "## Git Head" in result.output
        assert "abc1234" in result.output
        assert "## Crash File" in result.output

    def test_git_context_unavailable(self, snapshot_dir: Path, monkeypatch) -> None:
        import llmdebug.git_context as git_context_module

        monkeypatch.setattr(
            git_context_module,
            "get_enhanced_git_context",
            lambda *_args, **_kwargs: None,
        )
        runner = CliRunner(mix_stderr=False)
        result = runner.invoke(main, ["git-context", "--dir", str(snapshot_dir)])
        assert result.exit_code == 1  # RCA-15: nothing on stdout for --json consumers
        assert "unavailable" in result.stderr.lower()
        assert result.output == ""


# ---------------------------------------------------------------------------
# Coverage-gap tests
# ---------------------------------------------------------------------------


class TestResolveExportSnapshotPath:
    """Cover _resolve_export_snapshot_path edge cases (lines 165-175, branch [169,171])."""

    def test_oserror_on_parent_resolve(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """OSError during path.parent.resolve() sets path_is_latest_pointer=False."""
        from llmdebug.cli import _resolve_export_snapshot_path

        # Create a snapshot so _latest_snapshot_path_for_export returns something
        cfg = SnapshotConfig(out_dir=str(tmp_path), output_format="json", max_snapshots=50)
        write_bundle(
            {
                "schema_version": "1.0",
                "name": "test_oserror",
                "exception": {"type": "E", "message": "m"},
                "frames": [],
            },
            cfg,
        )

        original_resolve = Path.resolve

        def _oserror_resolve(self: Path, strict: bool = False) -> Path:
            if self.name == tmp_path.name:
                raise OSError("permission denied")
            return original_resolve(self, strict=strict)

        monkeypatch.setattr(Path, "resolve", _oserror_resolve)
        result = _resolve_export_snapshot_path(None, str(tmp_path))
        assert isinstance(result, Path)

    def test_latest_snapshot_path_returns_none(self, tmp_path: Path) -> None:
        """Branch [169,171]: _latest_snapshot_path_for_export returns None."""
        from llmdebug.cli import _resolve_export_snapshot_path

        # Create only latest.json (no immutable snapshot files)
        (tmp_path / "latest.json").write_text(
            json.dumps(
                {"schema_version": "1.0", "name": "x", "exception": {"type": "E"}, "frames": []}
            ),
            encoding="utf-8",
        )
        result = _resolve_export_snapshot_path(None, str(tmp_path))
        assert isinstance(result, Path)
        assert result.name == "latest.json"

    def test_oserror_on_symlink_resolve(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """OSError during is_symlink()/resolve() returns path unchanged."""
        from llmdebug.cli import _resolve_export_snapshot_path

        snap_path = tmp_path / "snap.json"
        snap_path.write_text(
            json.dumps(
                {"schema_version": "1.0", "name": "x", "exception": {"type": "E"}, "frames": []}
            ),
            encoding="utf-8",
        )

        original_is_symlink = Path.is_symlink

        def _oserror_symlink(self: Path) -> bool:
            if self == snap_path:
                raise OSError("broken")
            return original_is_symlink(self)

        monkeypatch.setattr(Path, "is_symlink", _oserror_symlink)
        result = _resolve_export_snapshot_path(str(snap_path), str(tmp_path))
        assert result == snap_path


class TestPrintCrashFrameOnlyEmpty:
    """Cover _print_crash_frame_only with empty frame_items (line 313, branch [312,313])."""

    def test_empty_frame_items_returns_early(self) -> None:
        from llmdebug._cli_display import _print_crash_frame_only

        # Should not raise; just returns immediately
        _print_crash_frame_only([], 0, {}, None)


class TestPrintFrameEmptyText:
    """Cover _print_frame early return when formatted text is empty (line 372, branch [371,372])."""

    def test_empty_frame_produces_no_output(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from llmdebug.cli import _print_frame

        monkeypatch.setattr("llmdebug._cli_display.format_frame_text", lambda *a, **kw: "")
        _print_frame({}, 0, False)


class TestPrintFrameRichPyLocals:
    """Cover @py_ local variable skipping in _print_frame_rich (line 475, branch [474,475])."""

    def test_skips_py_prefix_locals(self) -> None:
        from io import StringIO

        from rich.console import Console

        from llmdebug.cli import _print_frame_rich

        console = Console(file=StringIO(), force_terminal=True, no_color=True)
        frame: dict = {
            "file_rel": "test.py",
            "line": 1,
            "function": "f",
            "source": {
                "start": 1,
                "end": 2,
                "snippet": [{"lineno": 1, "code": "x = 1"}],
            },
            "locals": {"@py_builtins": "<module>", "real_var": "42"},
            "locals_meta": {"real_var": {"type": "int"}},
        }
        _print_frame_rich(frame, 0, True, console)
        output = console.file.getvalue()  # type: ignore[union-attr]
        assert "real_var" in output
        assert "@py_builtins" not in output


class TestPrintFrameRichArrayShape:
    """Cover array shape highlighting branches [480,488], [486,488]."""

    def test_empty_array_highlighted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from io import StringIO

        from rich.console import Console

        from llmdebug.cli import _print_frame_rich

        monkeypatch.setattr("llmdebug._cli_display.HAS_RICH", True)
        console = Console(file=StringIO(), force_terminal=True, no_color=True)
        frame: dict = {
            "file_rel": "test.py",
            "line": 1,
            "function": "f",
            "locals": {
                "arr": {"__array__": True, "shape": [0, 3], "dtype": "float64"},
            },
            "locals_meta": {"arr": {"type": "numpy.ndarray"}},
        }
        _print_frame_rich(frame, 0, True, console)
        output = console.file.getvalue()  # type: ignore[union-attr]
        assert "arr" in output

    def test_nonempty_array_no_red(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from io import StringIO

        from rich.console import Console

        from llmdebug.cli import _print_frame_rich

        monkeypatch.setattr("llmdebug._cli_display.HAS_RICH", True)
        console = Console(file=StringIO(), force_terminal=True, no_color=True)
        frame: dict = {
            "file_rel": "test.py",
            "line": 1,
            "function": "f",
            "locals": {
                "arr": {"__array__": True, "shape": [5, 3], "dtype": "float64"},
            },
            "locals_meta": {"arr": {"type": "numpy.ndarray"}},
        }
        _print_frame_rich(frame, 0, True, console)
        output = console.file.getvalue()  # type: ignore[union-attr]
        assert "arr" in output


class TestPrintContextSectionsEmpty:
    """Cover _print_context_sections empty-blocks early return (line 608, branch [607,608])."""

    def test_no_context_produces_no_output(self) -> None:
        from llmdebug._cli_display import _print_context_sections

        # Snapshot with no pytest, env, capture_config, or warnings
        _print_context_sections({}, None)

    def test_plugin_warnings_rendered(self) -> None:
        """Cover branches [599,606]: pytest payload with _plugin_warnings."""
        from llmdebug._cli_display import _print_context_sections

        snapshot: dict = {
            "pytest": {"_plugin_warnings": ["plugin X slow"]},
        }
        # Should not raise, exercises the plugin_warnings path
        _print_context_sections(snapshot, None)


class TestPrintFooterNoCommit:
    """Cover branch [503,506]: git dict present but no commit key."""

    def test_git_without_commit_skips_git_line(self) -> None:
        from llmdebug._cli_display import _print_footer

        # git dict with branch but no commit -> parts stays empty for git
        snapshot: dict = {
            "git": {"branch": "main", "dirty": False},
        }
        _print_footer(snapshot, None)


class TestCleanFailedDeletion:
    """Cover line 1033, branch [1032,1033]: deletion failure message."""

    def test_failed_deletion_reports_count(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg = SnapshotConfig(out_dir=str(tmp_path), output_format="json", max_snapshots=50)
        for i in range(3):
            write_bundle(
                {
                    "schema_version": "1.0",
                    "name": f"del_{i}",
                    "exception": {"type": "E", "message": "m"},
                    "frames": [],
                },
                cfg,
            )
            time.sleep(0.05)

        original_unlink = Path.unlink

        def _fail_unlink(self: Path, missing_ok: bool = False) -> None:
            if self.suffix == ".json" and not self.name.startswith("latest"):
                raise PermissionError("denied")
            original_unlink(self, missing_ok=missing_ok)

        monkeypatch.setattr(Path, "unlink", _fail_unlink)

        runner = CliRunner(mix_stderr=False)
        result = runner.invoke(main, ["clean", "--keep", "1", "--yes", "--dir", str(tmp_path)])
        assert result.exit_code == 1  # RCA-15: deletions failed
        assert "Failed to delete" in result.output


class TestFramesRichRendering:
    """Cover Rich rendering in frames command (lines 1064-1065, 1074-1075)."""

    def test_frames_rich_single_index(
        self, snapshot_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from io import StringIO

        from rich.console import Console

        buf = StringIO()
        monkeypatch.setattr(
            "llmdebug.cli._get_console",
            lambda: Console(file=buf, force_terminal=True, no_color=True),
        )
        runner = CliRunner(mix_stderr=False)
        result = runner.invoke(main, ["frames", "--index", "0", "--dir", str(snapshot_dir)])
        assert result.exit_code == 0
        rich_output = buf.getvalue()
        assert "parse_input" in rich_output

    def test_frames_rich_all(self, snapshot_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from io import StringIO

        from rich.console import Console

        buf = StringIO()
        monkeypatch.setattr(
            "llmdebug.cli._get_console",
            lambda: Console(file=buf, force_terminal=True, no_color=True),
        )
        runner = CliRunner(mix_stderr=False)
        result = runner.invoke(main, ["frames", "--dir", str(snapshot_dir)])
        assert result.exit_code == 0
        rich_output = buf.getvalue()
        assert "parse_input" in rich_output
        assert "run" in rich_output


class TestExportBundleWithoutCompare:
    """Cover bundle export without --compare (branches [897,900], [902,914], [909,914])."""

    def test_bundle_json_no_compare(self, tmp_path: Path, sample_snapshot: dict) -> None:
        cfg = SnapshotConfig(out_dir=str(tmp_path), output_format="json", max_snapshots=50)
        write_bundle(sample_snapshot, cfg)

        bundle_dir = tmp_path / "bundle_no_cmp"
        runner = CliRunner(mix_stderr=False)
        result = runner.invoke(
            main,
            [
                "export",
                "--format",
                "json",
                "--bundle-dir",
                str(bundle_dir),
                "--dir",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 0
        assert (bundle_dir / "report.json").exists()
        assert (bundle_dir / "manifest.json").exists()
        assert not (bundle_dir / "compare.json").exists()

    def test_bundle_html_no_compare(self, tmp_path: Path, sample_snapshot: dict) -> None:
        cfg = SnapshotConfig(out_dir=str(tmp_path), output_format="json", max_snapshots=50)
        write_bundle(sample_snapshot, cfg)

        bundle_dir = tmp_path / "bundle_html"
        runner = CliRunner(mix_stderr=False)
        result = runner.invoke(
            main,
            [
                "export",
                "--format",
                "html",
                "--bundle-dir",
                str(bundle_dir),
                "--dir",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 0
        assert (bundle_dir / "report.html").exists()
        assert (bundle_dir / "manifest.json").exists()
        assert not (bundle_dir / "compare.html").exists()

    def test_bundle_html_with_compare(self, tmp_path: Path, sample_snapshot: dict) -> None:
        """Cover branch [909,914]: html bundle with compare artifact."""
        cfg = SnapshotConfig(out_dir=str(tmp_path), output_format="json", max_snapshots=50)
        write_bundle({**sample_snapshot, "name": "older_html"}, cfg)
        time.sleep(0.05)
        write_bundle({**sample_snapshot, "name": "newer_html"}, cfg)

        bundle_dir = tmp_path / "bundle_html_cmp"
        runner = CliRunner(mix_stderr=False)
        result = runner.invoke(
            main,
            [
                "export",
                "--format",
                "html",
                "--bundle-dir",
                str(bundle_dir),
                "--compare",
                "previous",
                "--dir",
                str(tmp_path),
            ],
        )
        assert result.exit_code == 0
        assert (bundle_dir / "report.html").exists()
        assert (bundle_dir / "compare.html").exists()
        assert (bundle_dir / "manifest.json").exists()


class TestDiffRichRendering:
    """Cover _print_diff_rich branches [1172,1175] and [1240,1211]."""

    def test_diff_rich_same_message_different_type(self) -> None:
        """Branch [1172,1175]: exception types differ but messages are identical."""
        from io import StringIO

        from rich.console import Console

        from llmdebug.cli import _print_diff_rich
        from llmdebug.snapshot_diff import (
            ExceptionDiff,
            LocalChange,
            LocalsDiff,
            LocationDiff,
            SnapshotDiffResult,
        )

        diff_result = SnapshotDiffResult(
            exception=ExceptionDiff(
                changed=True,
                old_type="ValueError",
                new_type="TypeError",
                old_message="bad input",
                new_message="bad input",  # same message
                old_category="value_error",
                new_category="type_error",
            ),
            location=LocationDiff(
                changed=False, old_location="app.py:10 in f", new_location="app.py:10 in f"
            ),
            locals_diff=LocalsDiff(changes=(), unchanged_count=0),
            git_changed=False,
            old_name="snap_a",
            new_name="snap_b",
            same_test=True,
        )
        console = Console(file=StringIO(), force_terminal=True, no_color=True)
        _print_diff_rich(diff_result, console)
        output = console.file.getvalue()  # type: ignore[union-attr]
        assert "ValueError" in output
        assert "TypeError" in output

    def test_diff_rich_same_category(self) -> None:
        """Branch [1175,1177]: exception types differ but categories are the same."""
        from io import StringIO

        from rich.console import Console

        from llmdebug.cli import _print_diff_rich
        from llmdebug.snapshot_diff import (
            ExceptionDiff,
            LocalsDiff,
            LocationDiff,
            SnapshotDiffResult,
        )

        diff_result = SnapshotDiffResult(
            exception=ExceptionDiff(
                changed=True,
                old_type="ValueError",
                new_type="TypeError",
                old_message="bad",
                new_message="bad",
                old_category="value_error",
                new_category="value_error",  # same category
            ),
            location=LocationDiff(
                changed=False, old_location="a.py:1 in f", new_location="a.py:1 in f"
            ),
            locals_diff=LocalsDiff(changes=(), unchanged_count=0),
            git_changed=False,
            old_name="s1",
            new_name="s2",
            same_test=True,
        )
        console = Console(file=StringIO(), force_terminal=True, no_color=True)
        _print_diff_rich(diff_result, console)
        output = console.file.getvalue()  # type: ignore[union-attr]
        assert "CHANGED" in output
        # Category should not appear since old_category == new_category
        assert "category:" not in output

    def test_diff_rich_shape_changed_locals(self) -> None:
        """Branch [1240,1211]: locals diff with shape_changed kind."""
        from io import StringIO

        from rich.console import Console

        from llmdebug.cli import _print_diff_rich
        from llmdebug.snapshot_diff import (
            ExceptionDiff,
            LocalChange,
            LocalsDiff,
            LocationDiff,
            SnapshotDiffResult,
        )

        diff_result = SnapshotDiffResult(
            exception=ExceptionDiff(
                changed=False,
                old_type="ValueError",
                new_type="ValueError",
                old_message="x",
                new_message="x",
                old_category="value_error",
                new_category="value_error",
            ),
            location=LocationDiff(
                changed=False, old_location="a.py:1 in f", new_location="a.py:1 in f"
            ),
            locals_diff=LocalsDiff(
                changes=(
                    LocalChange(
                        kind="shape_changed", name="arr", old_value=[5, 3], new_value=[0, 3]
                    ),
                    LocalChange(kind="modified", name="x", old_value="1", new_value="2"),
                ),
                unchanged_count=1,
            ),
            git_changed=False,
            old_name="s1",
            new_name="s2",
            same_test=True,
        )
        console = Console(file=StringIO(), force_terminal=True, no_color=True)
        _print_diff_rich(diff_result, console)
        output = console.file.getvalue()  # type: ignore[union-attr]
        assert "SHAPE" in output
        assert "EMPTY" in output
        assert "arr" in output

    def test_diff_rich_different_test(self) -> None:
        """Cover same_test=False in _print_diff_rich header."""
        from io import StringIO

        from rich.console import Console

        from llmdebug.cli import _print_diff_rich
        from llmdebug.snapshot_diff import (
            ExceptionDiff,
            LocalChange,
            LocalsDiff,
            LocationDiff,
            SnapshotDiffResult,
        )

        diff_result = SnapshotDiffResult(
            exception=ExceptionDiff(
                changed=True,
                old_type="ValueError",
                new_type="TypeError",
                old_message="old msg",
                new_message="new msg",
                old_category="value_error",
                new_category="type_error",
            ),
            location=LocationDiff(
                changed=True,
                old_location="a.py:1 in f",
                new_location="b.py:2 in g",
            ),
            locals_diff=LocalsDiff(
                changes=(
                    LocalChange(kind="added", name="y", new_value="3"),
                    LocalChange(kind="removed", name="z", old_value="4"),
                    LocalChange(
                        kind="type_changed",
                        name="v",
                        old_value="1",
                        new_value="1.0",
                        old_type="int",
                        new_type="float",
                    ),
                ),
                unchanged_count=2,
            ),
            git_changed=True,
            old_git={"commit": "aaa"},
            new_git={"commit": "bbb"},
            old_name="snap_old",
            new_name="snap_new",
            same_test=False,
        )
        console = Console(file=StringIO(), force_terminal=True, no_color=True)
        _print_diff_rich(diff_result, console)
        output = console.file.getvalue()  # type: ignore[union-attr]
        assert "different test" in output
        assert "MOVED" in output
        assert "ADDED" in output
        assert "REMOVED" in output
        assert "TYPE" in output


class TestShowWithCaptureWarnings:
    """Cover _print_context_sections with capture warnings."""

    def test_context_with_capture_warnings(self, tmp_path: Path) -> None:
        """Cover capture_warnings path in _print_context_sections."""
        snapshot = {
            "schema_version": "1.0",
            "name": "warn_test",
            "exception": {"type": "ValueError", "message": "boom"},
            "frames": [{"file": "/code/a.py", "file_rel": "a.py", "line": 1, "function": "f"}],
            "crash_frame_index": 0,
            "_capture_warnings": ["Serialization truncated for large object"],
            "env": {"python": "3.13", "platform": "Linux", "cwd": "/code"},
            "pytest": {
                "nodeid": "tests/test_x.py::test_y",
                "_plugin_warnings": ["Coverage collector unavailable"],
            },
        }
        cfg = SnapshotConfig(out_dir=str(tmp_path), output_format="json", max_snapshots=50)
        write_bundle(snapshot, cfg)

        runner = CliRunner(mix_stderr=False)
        result = runner.invoke(main, ["show", "--detail", "context", "--dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "Capture Warnings" in result.output
        assert "Serialization truncated" in result.output
        assert "Pytest Capture Warnings" in result.output
        assert "Coverage collector" in result.output


class TestShowFrameEmptySnapshot:
    """Cover show command with snapshot that has empty frames list."""

    def test_show_snapshot_no_frames(self, tmp_path: Path) -> None:
        snapshot = {
            "schema_version": "1.0",
            "name": "empty_frames",
            "exception": {"type": "RuntimeError", "message": "no frames"},
            "frames": [],
            "crash_frame_index": 0,
        }
        cfg = SnapshotConfig(out_dir=str(tmp_path), output_format="json", max_snapshots=50)
        write_bundle(snapshot, cfg)

        runner = CliRunner(mix_stderr=False)
        result = runner.invoke(main, ["show", "--dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "RuntimeError" in result.output

    def test_frames_command_no_frames(self, tmp_path: Path) -> None:
        snapshot = {
            "schema_version": "1.0",
            "name": "empty_frames",
            "exception": {"type": "RuntimeError", "message": "no frames"},
            "frames": [],
            "crash_frame_index": 0,
        }
        cfg = SnapshotConfig(out_dir=str(tmp_path), output_format="json", max_snapshots=50)
        write_bundle(snapshot, cfg)

        runner = CliRunner(mix_stderr=False)
        result = runner.invoke(main, ["frames", "--dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "No frames" in result.output


class TestCleanNothingToDelete:
    """Cover clean command nothing-to-delete path (lines 1018-1019)."""

    def test_clean_nothing_to_delete(self, tmp_path: Path) -> None:
        cfg = SnapshotConfig(out_dir=str(tmp_path), output_format="json", max_snapshots=50)
        write_bundle(
            {
                "schema_version": "1.0",
                "name": "keep_me",
                "exception": {"type": "E", "message": "m"},
                "frames": [],
            },
            cfg,
        )
        runner = CliRunner(mix_stderr=False)
        result = runner.invoke(main, ["clean", "--keep", "10", "--yes", "--dir", str(tmp_path)])
        assert result.exit_code == 0
        assert "Nothing to delete" in result.output

    def test_clean_aborted_by_user(self, tmp_path: Path) -> None:
        """Cover lines 1027-1028: user declines confirmation."""
        cfg = SnapshotConfig(out_dir=str(tmp_path), output_format="json", max_snapshots=50)
        for i in range(3):
            write_bundle(
                {
                    "schema_version": "1.0",
                    "name": f"snap_{i}",
                    "exception": {"type": "E", "message": "m"},
                    "frames": [],
                },
                cfg,
            )
            time.sleep(0.05)

        runner = CliRunner(mix_stderr=False)
        result = runner.invoke(main, ["clean", "--keep", "1", "--dir", str(tmp_path)], input="n\n")
        assert result.exit_code == 0
        assert "Aborted" in result.output


class TestFramesIndexOutOfRange:
    """Cover frames command index out of range (lines 1057-1058)."""

    def test_frames_index_out_of_range(self, snapshot_dir: Path) -> None:
        runner = CliRunner(mix_stderr=False)
        result = runner.invoke(main, ["frames", "--index", "99", "--dir", str(snapshot_dir)])
        assert result.exit_code == 1
        assert "out of range" in result.stderr


class TestDiffRichViaCommand:
    """Cover diff Rich rendering via command (line 1114)."""

    def test_diff_rich_via_command(
        self, diff_snapshot_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from io import StringIO

        from rich.console import Console

        buf = StringIO()
        monkeypatch.setattr(
            "llmdebug.cli._get_console",
            lambda: Console(file=buf, force_terminal=True, no_color=True),
        )
        runner = CliRunner(mix_stderr=False)
        result = runner.invoke(main, ["diff", "--dir", str(diff_snapshot_dir)])
        assert result.exit_code == 0
        rich_output = buf.getvalue()
        assert "Snapshot Diff" in rich_output


class TestShowRichRendering:
    """Cover Rich rendering paths in show command (lines 379-429)."""

    def test_show_rich_rendering(self, snapshot_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from io import StringIO

        from rich.console import Console

        buf = StringIO()
        monkeypatch.setattr(
            "llmdebug.cli._get_console",
            lambda: Console(file=buf, force_terminal=True, no_color=True),
        )
        runner = CliRunner(mix_stderr=False)
        result = runner.invoke(main, ["show", "--dir", str(snapshot_dir)])
        assert result.exit_code == 0
        rich_output = buf.getvalue()
        assert "Exception" in rich_output
        assert "Frame #0" in rich_output

    def test_show_full_rich_rendering(
        self, snapshot_dir: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from io import StringIO

        from rich.console import Console

        buf = StringIO()
        monkeypatch.setattr(
            "llmdebug.cli._get_console",
            lambda: Console(file=buf, force_terminal=True, no_color=True),
        )
        runner = CliRunner(mix_stderr=False)
        result = runner.invoke(main, ["show", "--full", "--dir", str(snapshot_dir)])
        assert result.exit_code == 0
        rich_output = buf.getvalue()
        assert "parse_input" in rich_output
        assert "run" in rich_output

    def test_show_rich_frame_without_source(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """`show --full` with Rich renders a frame without source (via _print_frame_rich)."""
        from io import StringIO

        from rich.console import Console

        snapshot = {
            "schema_version": "1.0",
            "name": "no_source",
            "exception": {"type": "ValueError", "message": "test"},
            "frames": [
                {
                    "file": "/code/a.py",
                    "file_rel": "a.py",
                    "line": 1,
                    "function": "f",
                    "code": "x = bad()",
                    "locals": {},
                }
            ],
            "crash_frame_index": 0,
        }
        cfg = SnapshotConfig(out_dir=str(tmp_path), output_format="json", max_snapshots=50)
        write_bundle(snapshot, cfg)

        buf = StringIO()
        monkeypatch.setattr(
            "llmdebug.cli._get_console",
            lambda: Console(file=buf, force_terminal=True, no_color=True),
        )
        runner = CliRunner(mix_stderr=False)
        result = runner.invoke(main, ["show", "--dir", str(tmp_path)])
        assert result.exit_code == 0
        rich_output = buf.getvalue()
        assert "x = bad()" in rich_output

    def test_show_rich_context(self, snapshot_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        """Cover Rich path in context sections rendering."""
        from io import StringIO

        from rich.console import Console

        buf = StringIO()
        monkeypatch.setattr(
            "llmdebug.cli._get_console",
            lambda: Console(file=buf, force_terminal=True, no_color=True),
        )
        runner = CliRunner(mix_stderr=False)
        result = runner.invoke(main, ["show", "--detail", "context", "--dir", str(snapshot_dir)])
        assert result.exit_code == 0
        rich_output = buf.getvalue()
        assert "Exception" in rich_output


class TestListRichRendering:
    """Cover Rich rendering in list command (lines 723-756)."""

    def test_list_rich_rendering(self, snapshot_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        from io import StringIO

        from rich.console import Console

        buf = StringIO()
        monkeypatch.setattr(
            "llmdebug.cli._get_console",
            lambda: Console(file=buf, force_terminal=True, no_color=True),
        )
        runner = CliRunner(mix_stderr=False)
        result = runner.invoke(main, ["list", "--dir", str(snapshot_dir)])
        assert result.exit_code == 0
        rich_output = buf.getvalue()
        assert "Snapshots" in rich_output
        assert "test_" in rich_output
