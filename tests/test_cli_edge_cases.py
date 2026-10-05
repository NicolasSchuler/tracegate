"""Additional edge-case tests for CLI helper branches."""

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import click
import pytest
from click.testing import CliRunner

from llmdebug import _cli_display, cli


class _FakeConsole:
    def __init__(self) -> None:
        self.items: list[Any] = []

    def print(self, item: Any = "", *args: Any, **kwargs: Any) -> None:
        self.items.append(item)


def test_stdout_is_tty_handles_noncallable_and_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    class _NoIsatty:
        isatty = True

    class _BadIsatty:
        def isatty(self) -> bool:
            raise RuntimeError("boom")

    monkeypatch.setattr(click, "get_text_stream", lambda _name: _NoIsatty())
    assert cli._stdout_is_tty() is False

    monkeypatch.setattr(click, "get_text_stream", lambda _name: _BadIsatty())
    assert cli._stdout_is_tty() is False


def test_truncation_helpers_cover_boundary_conditions() -> None:
    assert cli._truncate_text("abcdef", 0) == ""
    assert cli._truncate_text("abcdef", 3) == "abc"
    assert cli._truncate_text("abcdef", 5) == "ab..."

    assert cli._truncate_tail("abcdef", 0) == ""
    assert cli._truncate_tail("abcdef", 3) == "def"
    assert cli._truncate_tail("abcdef", 5) == "...ef"


def test_get_console_returns_console_when_rich_and_tty(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_cli_display, "HAS_RICH", True)
    monkeypatch.setattr(_cli_display, "_stdout_is_tty", lambda: True)
    monkeypatch.setitem(os.environ, "NO_COLOR", "1")
    assert cli._get_console() is not None


def test_snapshot_loader_wrappers_translate_to_system_exit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cli,
        "_load_snapshot_impl",
        lambda *_a, **_k: (_ for _ in ()).throw(FileNotFoundError("missing snapshot")),
    )
    with pytest.raises(SystemExit, match="1"):
        cli._load_snapshot(None, ".llmdebug")

    monkeypatch.setattr(
        cli,
        "_load_snapshot_raw_impl",
        lambda *_a, **_k: (_ for _ in ()).throw(FileNotFoundError("missing raw snapshot")),
    )
    with pytest.raises(SystemExit, match="1"):
        cli._load_snapshot_raw(None, ".llmdebug")

    monkeypatch.setattr(
        cli,
        "_load_snapshot_from_file_impl",
        lambda *_a, **_k: (_ for _ in ()).throw(ImportError("toon missing")),
    )
    with pytest.raises(SystemExit, match="1"):
        cli._load_snapshot_from_file(Path("missing.json"))


def test_print_footer_plaintext_includes_git_and_repro(capsys: pytest.CaptureFixture[str]) -> None:
    snapshot = {
        "git": {"commit": "abc1234", "branch": "main", "dirty": True},
        "pytest": {"repro": ["/usr/bin/python3", "-m", "pytest", "tests/test_x.py::test_y", "-q"]},
    }
    _cli_display._print_footer(snapshot, None)
    out = capsys.readouterr().out
    assert "git: abc1234 main *dirty*" in out
    assert "repro: python -m pytest tests/test_x.py::test_y -q" in out


def test_print_exception_panel_handles_malformed_exception(
    capsys: pytest.CaptureFixture[str],
) -> None:
    cli._print_exception_panel({"exception": "not-a-dict"}, None)
    out = capsys.readouterr().out
    assert "UnknownError" in out


def test_snapshot_ref_loaders_and_console_render_helpers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        cli,
        "_resolve_snapshot_ref_path_impl",
        lambda *_a, **_k: (_ for _ in ()).throw(FileNotFoundError("missing snapshot")),
    )
    with pytest.raises(SystemExit, match="1"):
        cli._load_snapshot_ref(None, ".llmdebug")
    with pytest.raises(SystemExit, match="1"):
        cli._load_snapshot_ref_raw(None, ".llmdebug")

    console = _FakeConsole()
    cli._print_exception_panel({"exception": {"type": "ValueError", "message": "boom"}}, console)
    assert console.items

    _cli_display._print_selected_frame(
        {
            "file": "app.py",
            "line": 10,
            "function": "fn",
            "code": "return 1",
        },
        0,
        True,
        console,
    )
    assert console.items

    _cli_display._print_context_if_requested(
        {"pytest": {"nodeid": "tests/test_x.py::test_y"}}, "context", console
    )
    assert any("## Pytest" in str(item) for item in console.items)

    _cli_display._print_show_footer({"git": {"commit": "abc123", "branch": "main"}}, console)
    assert any("git: abc123 main" in str(item) for item in console.items)


def test_print_footer_handles_malformed_git_and_pytest(capsys: pytest.CaptureFixture[str]) -> None:
    _cli_display._print_footer({"git": "bad", "pytest": "bad"}, None)
    out = capsys.readouterr().out
    assert out == ""


def test_render_metadata_and_context_blocks_cover_list_dict_and_empty_paths() -> None:
    assert _cli_display._render_metadata_value(["a", "b", 3]) == "a b 3"
    assert '"x": 1' in _cli_display._render_metadata_value([{"x": 1}])
    assert '"k": "v"' in _cli_display._render_metadata_value({"k": "v"})
    assert _cli_display._render_metadata_value("abcdef") == "abcdef"

    assert _cli_display._context_block_lines("Empty", None, ("x",)) == []
    assert _cli_display._context_block_lines("Empty", {"x": ""}, ("x",)) == []
    assert _cli_display._context_block_lines("Block", {"x": "value", "y": ""}, ("x", "y")) == [
        "## Block",
        "x: value",
    ]


def test_print_context_sections_covers_warning_and_console_paths(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    snapshot = {
        "pytest": {
            "nodeid": "tests/test_x.py::test_y",
            "outcome": "failed",
            "when": "call",
            "params": {"x": 1},
            "xdist_worker": "gw0",
            "flaky": True,
            "_plugin_warnings": ["plugin warning"],
        },
        "env": {
            "python": "3.13",
            "platform": "Darwin",
            "cwd": "/tmp",
            "executable": "/usr/bin/python",
            "argv": ["python", "-m", "pytest"],
        },
        "capture_config": {
            "redaction_profile": "ci",
            "output_format": "json",
            "frames": 5,
            "source_mode": "all",
            "locals_mode": "safe",
            "include_git": True,
            "include_coverage": True,
        },
        "_capture_warnings": ["capture warning"],
    }

    console = _FakeConsole()
    _cli_display._print_context_sections(snapshot, console)
    rendered = "\n".join(str(item) for item in console.items)
    assert "## Pytest" in rendered
    assert "## Environment" in rendered
    assert "## Capture Config" in rendered
    assert "## Capture Warnings" in rendered
    assert "## Pytest Capture Warnings" in rendered

    _cli_display._print_context_sections(snapshot, None)
    stdout = capsys.readouterr().out
    assert "## Pytest" in stdout
    assert "capture warning" in stdout
    assert "plugin warning" in stdout


def test_print_remaining_frames_hint_covers_console_and_plain_text(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    snapshot = {"frames": [{"function": "a"}, {"function": "b"}]}
    monkeypatch.setattr(_cli_display, "_stdout_is_tty", lambda: True)

    console = _FakeConsole()
    _cli_display._print_remaining_frames_hint(snapshot, console)
    assert any("more frame" in str(item) for item in console.items)

    _cli_display._print_remaining_frames_hint(snapshot, None)
    stdout = capsys.readouterr().out
    assert "more frame" in stdout


def test_cli_commands_cover_list_clean_frames_export_and_diff_branches(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    class _BoomDateTime:
        @staticmethod
        def fromtimestamp(*_args: Any, **_kwargs: Any) -> None:
            raise ValueError("boom")

    list_console = _FakeConsole()
    monkeypatch.setattr(cli, "datetime", _BoomDateTime)
    monkeypatch.setattr(cli, "_get_console", lambda: list_console)
    monkeypatch.setattr(
        cli,
        "_find_all_snapshots_impl",
        lambda _out_dir, limit=10, search=None: [
            {
                "mtime": "bad",
                "name": "snap",
                "exc_type": "TypeError",
                "file": "/a.py",
                "line": 1,
            }
        ],
    )
    cli.list_cmd.callback(limit=10, search="", out_dir=str(tmp_path))
    assert list_console.items

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    cli.clean.callback(keep=1, delete_all=False, yes=True, out_dir=str(empty_dir))
    assert "Nothing to delete." in capsys.readouterr().out

    dirty_dir = tmp_path / "dirty"
    dirty_dir.mkdir()
    (dirty_dir / "snap.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(cli.click, "confirm", lambda *_a, **_k: False)
    cli.clean.callback(keep=0, delete_all=False, yes=False, out_dir=str(dirty_dir))
    assert "Aborted." in capsys.readouterr().out

    monkeypatch.setattr(cli, "_load_snapshot_ref", lambda *_a, **_k: {"frames": []})
    cli.frames.callback(snapshot_ref=None, index=None, out_dir=str(tmp_path))
    assert "No frames in snapshot." in capsys.readouterr().out

    monkeypatch.setattr(
        cli, "_load_snapshot_ref", lambda *_a, **_k: {"frames": [{"function": "f"}]}
    )
    with pytest.raises(SystemExit, match="1"):
        cli.frames.callback(snapshot_ref=None, index=5, out_dir=str(tmp_path))
    assert "out of range" in capsys.readouterr().err

    with pytest.raises(SystemExit, match="1"):
        cli.export.callback(
            snapshot_ref=None,
            export_format="json",
            output=None,
            export_mode=None,
            redaction_profile=None,
            bundle_dir=None,
            compare_ref=None,
            out_dir=str(tmp_path / "missing-export"),
        )

    diff_console = _FakeConsole()
    monkeypatch.setattr(cli, "_get_console", lambda: diff_console)
    old_snapshot = {
        "name": "old",
        "exception": {"type": "ValueError", "message": "old"},
        "frames": [{"file": "/code/app.py", "file_rel": "app.py", "line": 1, "function": "fn"}],
        "crash_frame_index": 0,
    }
    new_snapshot = {
        "name": "new",
        "exception": {"type": "TypeError", "message": "new"},
        "frames": [{"file": "/code/app.py", "file_rel": "app.py", "line": 2, "function": "fn"}],
        "crash_frame_index": 0,
    }
    monkeypatch.setattr(
        cli,
        "_load_snapshot_ref",
        lambda ref, _out_dir: old_snapshot if ref in {"old", "previous"} else new_snapshot,
    )
    cli.diff.callback("old", "new", False, str(tmp_path))
    assert diff_console.items


def test_footer_and_export_path_helpers_cover_console_and_resolution_paths(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    console = _FakeConsole()
    snapshot = {
        "git": {"commit": "abc1234", "branch": "main", "dirty": True},
        "pytest": {"repro": ["/usr/bin/python3", "-m", "pytest", "tests/test_x.py::test_y", "-q"]},
    }
    _cli_display._print_footer(snapshot, console)
    rendered = "\n".join(str(item) for item in console.items)
    assert "git: abc1234 main *dirty*" in rendered
    assert "repro: python -m pytest tests/test_x.py::test_y -q" in rendered

    latest_path = tmp_path / "newest.json"
    latest_path.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        cli,
        "_find_all_snapshots_impl",
        lambda _out_dir, limit=1: [{"path": None}],
    )
    assert cli._latest_snapshot_path_for_export(str(tmp_path)) is None
    monkeypatch.setattr(
        cli,
        "_find_all_snapshots_impl",
        lambda _out_dir, limit=1: [{"path": str(latest_path)}],
    )
    assert cli._latest_snapshot_path_for_export(str(tmp_path)) == latest_path
    monkeypatch.setattr(cli, "_find_all_snapshots_impl", lambda _out_dir, limit=1: [])
    assert cli._latest_snapshot_path_for_export(str(tmp_path)) is None

    pointer = tmp_path / "latest.json"
    pointer.write_text("{}", encoding="utf-8")
    resolved = tmp_path / "resolved.json"
    resolved.write_text("{}", encoding="utf-8")
    monkeypatch.setattr(
        cli,
        "_resolve_snapshot_ref_path_impl",
        lambda ref, out_dir, _trust_path=True: pointer,
    )
    monkeypatch.setattr(cli, "_latest_snapshot_path_for_export", lambda _out_dir: resolved)
    assert cli._resolve_export_snapshot_path(None, str(tmp_path)) == resolved
    assert cli._resolve_export_snapshot_path("latest", str(tmp_path)) == resolved

    real = tmp_path / "real.json"
    real.write_text("{}", encoding="utf-8")
    link = tmp_path / "link.json"
    try:
        link.symlink_to(real)
    except OSError:
        pytest.skip("symlinks are unavailable in this environment")

    monkeypatch.setattr(
        cli,
        "_resolve_snapshot_ref_path_impl",
        lambda ref, out_dir, _trust_path=True: link,
    )
    monkeypatch.setattr(cli, "_latest_snapshot_path_for_export", lambda _out_dir: None)
    assert cli._resolve_export_snapshot_path("ref", str(tmp_path)) == real


def test_print_frame_rich_with_and_without_source(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_cli_display, "HAS_RICH", True)
    console = _FakeConsole()

    with_source = {
        "file": "app.py",
        "line": 10,
        "function": "fn",
        "source": {"start": 10, "end": 10, "snippet": [{"lineno": 10, "code": "x = 1"}]},
        "locals": {"arr": {"__array__": "numpy.ndarray", "shape": [0, 2]}},
        "locals_meta": {"arr": {"type": "numpy.ndarray"}},
    }
    cli._print_frame_rich(with_source, 0, True, console)

    no_source = {"file": "app.py", "line": 11, "function": "fn2", "code": "return x", "locals": {}}
    cli._print_frame_rich(no_source, 1, False, console)

    assert len(console.items) >= 2


def test_print_diff_rich_covers_change_kinds(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_cli_display, "HAS_RICH", True)
    console = _FakeConsole()

    changes = [
        SimpleNamespace(kind="modified", name="a", old_value="1", new_value="2"),
        SimpleNamespace(kind="added", name="b", old_value="", new_value="3"),
        SimpleNamespace(kind="removed", name="c", old_value="4", new_value=""),
        SimpleNamespace(
            kind="type_changed",
            name="d",
            old_value="1",
            new_value="'1'",
            old_type="int",
            new_type="str",
        ),
        SimpleNamespace(kind="shape_changed", name="arr", old_value=[1, 2], new_value=[0, 2]),
    ]
    result = SimpleNamespace(
        old_name="old",
        new_name="new",
        same_test=False,
        exception=SimpleNamespace(
            changed=True,
            old_type="ValueError",
            new_type="TypeError",
            old_message="old message",
            new_message="new message",
            old_category="value_error",
            new_category="type_error",
        ),
        location=SimpleNamespace(changed=True, old_location="a.py:1", new_location="b.py:2"),
        locals_diff=SimpleNamespace(changes=changes, unchanged_count=3),
        git_changed=True,
        old_git={"commit": "aaa"},
        new_git={"commit": "bbb"},
    )
    cli._print_diff_rich(result, console)

    unchanged_result = SimpleNamespace(
        old_name="same",
        new_name="same",
        same_test=True,
        exception=SimpleNamespace(
            changed=False,
            old_type="ValueError",
            new_type="ValueError",
            old_message="same message",
            new_message="same message",
            old_category=None,
            new_category=None,
        ),
        location=SimpleNamespace(changed=False, old_location="a.py:1", new_location="a.py:1"),
        locals_diff=SimpleNamespace(changes=[], unchanged_count=0),
        git_changed=False,
        old_git={},
        new_git={},
    )
    cli._print_diff_rich(unchanged_result, console)

    assert len(console.items) >= 5


def test_perform_deletions_counts_success_and_failures(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    ok_file = tmp_path / "ok.json"
    ok_file.write_text("{}", encoding="utf-8")
    missing = tmp_path / "missing.json"

    deleted, failed = cli._perform_deletions([ok_file, missing])
    assert deleted == 1
    assert failed == 1
    assert "could not delete" in capsys.readouterr().err


def test_list_command_rich_branch(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_console = _FakeConsole()
    monkeypatch.setattr(cli, "_get_console", lambda: fake_console)
    monkeypatch.setattr(
        cli,
        "_find_all_snapshots_impl",
        lambda _out_dir, limit=10, search=None: [
            {
                "mtime": 1_700_000_000.0,
                "name": "test-name",
                "exc_type": "ValueError",
                "file": "app.py",
                "line": 42,
            }
        ],
    )

    runner = CliRunner(mix_stderr=False)
    result = runner.invoke(cli.main, ["list", "--dir", ".llmdebug"])
    assert result.exit_code == 0
    assert fake_console.items


def test_list_command_handles_malformed_entry_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cli,
        "_find_all_snapshots_impl",
        lambda _out_dir, limit=10, search=None: [
            {"mtime": "nan", "name": "bad-row"},
            "not-a-dict",
        ],
    )

    runner = CliRunner(mix_stderr=False)
    result = runner.invoke(cli.main, ["list", "--dir", ".llmdebug"])
    assert result.exit_code == 0
    assert "?" in result.output
