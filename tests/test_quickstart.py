"""Regression tests for the offline CLI quickstart."""

from __future__ import annotations

import shlex
from pathlib import Path

import pytest
from click.testing import CliRunner

from llmdebug import _quickstart
from llmdebug._quickstart import create_quickstart_snapshot
from llmdebug._snapshot_loader import load_snapshot_from_file
from llmdebug.cli import main


def _immutable_json_snapshots(out_dir: Path) -> list[Path]:
    """Return immutable JSON snapshots without the latest pointer."""
    return sorted(path for path in out_dir.glob("*.json") if path.name != "latest.json")


def test_create_quickstart_snapshot_captures_expected_failure(tmp_path: Path) -> None:
    """The helper should write a loadable core snapshot with stable example evidence."""
    result = create_quickstart_snapshot(tmp_path)

    assert result.snapshot_path.exists()
    snapshot = load_snapshot_from_file(result.snapshot_path)
    assert snapshot["name"] == "quickstart_example"
    assert snapshot["exception"]["type"] == "ZeroDivisionError"
    assert snapshot["exception"]["message"] == "division by zero"

    crash_frame = snapshot["frames"][snapshot["crash_frame_index"]]
    assert crash_frame["function"] == "_calculate_average_latency"
    assert crash_frame["locals"] == {
        "total_latency_ms": 120,
        "completed_requests": 0,
    }

    assert "git" not in snapshot
    assert "env" not in snapshot
    assert "context" not in snapshot
    assert snapshot["capture_config"]["include_git"] is False
    assert snapshot["capture_config"]["include_env"] is False


def test_quickstart_repeated_runs_preserve_previous_snapshots(tmp_path: Path) -> None:
    """Repeated quickstarts should append snapshots without removing earlier artifacts."""
    existing = tmp_path / "existing.json"
    existing.write_text("{}", encoding="utf-8")

    first = create_quickstart_snapshot(tmp_path)
    second = create_quickstart_snapshot(tmp_path)

    assert first.snapshot_path != second.snapshot_path
    assert first.snapshot_path.exists()
    assert second.snapshot_path.exists()
    assert existing.read_text(encoding="utf-8") == "{}"
    assert len(_immutable_json_snapshots(tmp_path)) == 3


def test_quickstart_cli_supports_custom_directory_with_spaces(tmp_path: Path) -> None:
    """The command should create, display, and quote a custom snapshot directory."""
    out_dir = tmp_path / "snapshot folder"
    runner = CliRunner(mix_stderr=False)

    result = runner.invoke(main, ["quickstart", "--dir", str(out_dir)])

    assert result.exit_code == 0, result.output
    assert "ZeroDivisionError: division by zero" in result.output
    snapshots = _immutable_json_snapshots(out_dir)
    assert len(snapshots) == 1
    assert f"Snapshot written to: {snapshots[0]}" in result.output
    assert shlex.join(("llmdebug", "show", "--dir", str(out_dir))) in result.output
    assert shlex.join(("llmdebug", "list", "--dir", str(out_dir))) in result.output


def test_quickstart_snapshot_works_with_show_and_list(tmp_path: Path) -> None:
    """A generated example should be immediately usable by normal inspection commands."""
    create_quickstart_snapshot(tmp_path)
    runner = CliRunner(mix_stderr=False)

    show_result = runner.invoke(main, ["show", "--dir", str(tmp_path)])
    list_result = runner.invoke(main, ["list", "--dir", str(tmp_path)])

    assert show_result.exit_code == 0
    assert "ZeroDivisionError" in show_result.output
    assert "completed_requests = 0" in show_result.output
    assert list_result.exit_code == 0
    assert "quickstart_example" in list_result.output


def test_quickstart_reports_an_unexpected_non_failure(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The helper should fail clearly if the synthetic example stops raising."""
    monkeypatch.setattr(_quickstart, "_calculate_average_latency", lambda: 1.0)

    with pytest.raises(RuntimeError, match="did not raise"):
        create_quickstart_snapshot(tmp_path)

    assert _immutable_json_snapshots(tmp_path) == []
