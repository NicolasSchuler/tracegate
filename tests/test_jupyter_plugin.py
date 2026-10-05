"""Tests for Jupyter/IPython integration."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from llmdebug.jupyter_plugin import (
    _display_html_or_text,
    _display_markdown_or_text,
    _extract_jupyter_context,
    _format_full_snapshot_html,
    _format_full_snapshot_text,
    _format_local_value,
    _format_snapshot_banner_html,
    _format_snapshot_banner_text,
    _html_crash_frame_parts,
    _is_notebook,
    _text_crash_frame_lines,
    install,
    load_ipython_extension,
    uninstall,
    unload_ipython_extension,
)

from .conftest import read_latest_snapshot

# ---------------------------------------------------------------------------
# Fixtures and helpers
# ---------------------------------------------------------------------------

SAMPLE_SNAPSHOT: dict[str, Any] = {
    "name": "jupyter_cell",
    "exception": {
        "type": "ValueError",
        "message": "invalid literal for int(): 'abc'",
        "error_category": {
            "category": "value_error",
            "suggestion": "Check the input type before conversion.",
        },
    },
    "frames": [
        {
            "file": "/tmp/ipykernel/cell.py",
            "file_rel": "cell.py",
            "line": 3,
            "function": "<module>",
            "code": "x = int('abc')",
            "locals": {
                "data": [1, 2, 3],
                "x": "abc",
            },
        }
    ],
    "jupyter": {
        "execution_count": 5,
        "cell_source": "x = int('abc')",
    },
}


class _FakeTerminalShell:
    """Minimal fake IPython shell for terminal mode."""

    def __init__(self) -> None:
        self.execution_count = 42
        self._original_showtraceback = self.showtraceback
        self._registered_magics: list[Any] = []
        self.history_manager = _FakeHistoryManager()

    def showtraceback(self, *args: Any, **kwargs: Any) -> None:
        pass

    def register_magics(self, cls: Any) -> None:
        self._registered_magics.append(cls)


class _FakeNotebookShell(_FakeTerminalShell):
    """Fake shell whose class name is ZMQInteractiveShell."""

    pass


# Rename the class so type().__name__ returns the right thing
_FakeNotebookShell.__name__ = "ZMQInteractiveShell"
_FakeNotebookShell.__qualname__ = "ZMQInteractiveShell"


class _FakeHistoryManager:
    def get_tail(self, n: int = 1, raw: bool = True) -> list[tuple[int, int, str]]:
        return [(1, 1, "x = 1/0")]


def _make_ipython(*, notebook: bool = False) -> Any:
    """Create a fake IPython shell."""
    if notebook:
        return _FakeNotebookShell()
    return _FakeTerminalShell()


@pytest.fixture(autouse=True)
def _cleanup_jupyter():
    """Ensure jupyter plugin is uninstalled after each test."""
    yield
    uninstall()


@pytest.fixture()
def snap_dir(tmp_path: Path) -> Path:
    """Create a temp snapshot directory."""
    d = tmp_path / ".llmdebug"
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# Installation tests
# ---------------------------------------------------------------------------


class TestInstall:
    def test_failed_magic_registration_restores_the_hook(self):
        """RCA-07: the hook stayed wrapped with _installed=False, so a retry wrapped it twice."""
        ip = _make_ipython()
        original = ip.showtraceback

        def _fail(_cls: Any) -> None:
            raise RuntimeError("magics unavailable")

        ip.register_magics = _fail
        with pytest.raises(RuntimeError, match="magics unavailable"):
            install(ip, out_dir="/tmp/test_llmdebug")

        assert ip.showtraceback == original
        del ip.register_magics  # back to the working class method
        install(ip, out_dir="/tmp/test_llmdebug")
        uninstall()
        assert ip.showtraceback == original

    def test_install_wraps_showtraceback(self):
        ip = _make_ipython()
        original = ip.showtraceback
        install(ip, out_dir="/tmp/test_llmdebug")

        assert ip.showtraceback is not original
        assert len(ip._registered_magics) > 0

    def test_install_idempotent(self):
        ip = _make_ipython()
        install(ip, out_dir="/tmp/test_llmdebug")
        first_wrapper = ip.showtraceback

        install(ip, out_dir="/tmp/test_llmdebug")
        assert ip.showtraceback is first_wrapper

    def test_uninstall_restores_original(self):
        ip = _make_ipython()
        # Bound methods are created fresh each access; compare via __func__
        original_func = _FakeTerminalShell.showtraceback
        install(ip, out_dir="/tmp/test_llmdebug")

        # After install, showtraceback should be a closure (not the class method)
        assert ip.showtraceback is not original_func

        uninstall()
        # After uninstall, the stored bound method is restored as instance attr
        # It should point to the same underlying function
        restored = ip.showtraceback
        assert getattr(restored, "__func__", restored) is original_func

    def test_uninstall_idempotent(self):
        uninstall()
        uninstall()

        # Plugin state should be clean after uninstall
        import llmdebug.jupyter_plugin as jp

        assert jp._installed is False
        assert jp._cfg is None

    def test_install_explicit_ipython(self):
        ip = _make_ipython()
        install(ip, out_dir="/tmp/test_llmdebug")
        assert len(ip._registered_magics) > 0

    def test_install_defaults_to_prod_redaction_profile(self):
        import llmdebug.jupyter_plugin as jupyter_plugin_module

        ip = _make_ipython()
        install(ip, out_dir="/tmp/test_llmdebug")
        assert jupyter_plugin_module._cfg is not None
        assert jupyter_plugin_module._cfg.redaction_profile == "prod"


# ---------------------------------------------------------------------------
# Exception capture tests
# ---------------------------------------------------------------------------


class TestExceptionCapture:
    def test_calls_original_showtraceback_first(self):
        ip = _make_ipython()
        original_called = []

        def tracking_showtraceback(*args: Any, **kwargs: Any) -> None:
            original_called.append(True)

        ip.showtraceback = tracking_showtraceback
        install(ip, out_dir="/tmp/test_llmdebug_capture")

        wrapper = ip.showtraceback

        try:
            raise ValueError("test error")
        except ValueError:
            with (
                patch("llmdebug.jupyter_plugin._capture_safely"),
                patch("llmdebug.jupyter_plugin._display_snapshot_banner"),
            ):
                wrapper()

        assert len(original_called) == 1

    def test_captures_exception_with_jupyter_context(self, snap_dir: Path):
        ip = _make_ipython()
        install(ip, out_dir=str(snap_dir))

        wrapper = ip.showtraceback

        try:
            raise ValueError("test capture")
        except ValueError:
            with patch("llmdebug.jupyter_plugin._display_snapshot_banner"):
                wrapper()

        snapshot = read_latest_snapshot(snap_dir)
        assert snapshot["exception"]["type"] == "ValueError"
        assert "jupyter" in snapshot
        assert snapshot["jupyter"]["execution_count"] == 42

    def test_jupyter_context_redacts_cell_source_by_default(self, snap_dir: Path):
        class SecretHistory:
            def get_tail(self, n: int = 1, raw: bool = True) -> list[tuple[int, int, str]]:
                return [(1, 1, "api_key=secret123")]

        ip = _make_ipython()
        ip.history_manager = SecretHistory()
        install(ip, out_dir=str(snap_dir))

        wrapper = ip.showtraceback

        try:
            raise ValueError("test capture")
        except ValueError:
            with patch("llmdebug.jupyter_plugin._display_snapshot_banner"):
                wrapper()

        snapshot = read_latest_snapshot(snap_dir)
        assert snapshot["jupyter"]["cell_source"] == "[REDACTED]"

    def test_rate_limiting_prevents_flood(self, snap_dir: Path):
        ip = _make_ipython()
        install(
            ip,
            out_dir=str(snap_dir),
            max_captures_per_minute=2,
        )

        wrapper = ip.showtraceback

        for _ in range(5):
            try:
                raise ValueError("repeated error")
            except ValueError:
                with patch("llmdebug.jupyter_plugin._display_snapshot_banner"):
                    wrapper()

        from llmdebug._snapshot_loader import find_all_snapshots

        snaps = find_all_snapshots(str(snap_dir))
        assert len(snaps) <= 3  # 2 allowed + possible timing slack

    def test_capture_failure_does_not_propagate(self):
        ip = _make_ipython()
        original_called = []

        def tracking_showtraceback(*args: Any, **kwargs: Any) -> None:
            original_called.append(True)

        ip.showtraceback = tracking_showtraceback
        install(ip, out_dir="/tmp/test_llmdebug_fail")

        wrapper = ip.showtraceback

        try:
            raise ValueError("test")
        except ValueError:
            with patch(
                "llmdebug.jupyter_plugin._capture_safely",
                side_effect=Exception("capture broke"),
            ):
                # Should NOT raise
                wrapper()

        assert len(original_called) == 1


# ---------------------------------------------------------------------------
# Display tests
# ---------------------------------------------------------------------------


class TestDisplay:
    def test_banner_html_contains_exception(self):
        html = _format_snapshot_banner_html(SAMPLE_SNAPSHOT)
        assert "ValueError" in html
        assert "llmdebug" in html
        assert "cell.py:3" in html

    def test_banner_text_contains_exception(self):
        text = _format_snapshot_banner_text(SAMPLE_SNAPSHOT)
        assert "ValueError" in text
        assert "cell.py:3" in text
        assert "%llmdebug" in text

    def test_full_html_contains_details(self):
        html = _format_full_snapshot_html(SAMPLE_SNAPSHOT)
        assert "ValueError" in html
        assert "invalid literal" in html
        assert "cell.py" in html
        assert "Suggestion:" not in html
        assert "Cell In[5]" in html

    def test_full_text_contains_details(self):
        text = _format_full_snapshot_text(SAMPLE_SNAPSHOT)
        assert "ValueError" in text
        assert "invalid literal" in text
        assert "cell.py" in text
        assert "**Suggestion:**" not in text
        assert "Cell In[5]" in text

    def test_no_display_when_no_snapshot(self, capsys: pytest.CaptureFixture[str]):
        ip = _make_ipython()
        from llmdebug.jupyter_plugin import _display_snapshot_banner

        _display_snapshot_banner(ip, "/nonexistent/dir")
        captured = capsys.readouterr()
        assert "llmdebug" not in captured.out

    def test_html_escapes_special_chars(self):
        snapshot: dict[str, Any] = {
            "exception": {
                "type": "ValueError",
                "message": '<script>alert("xss")</script>',
            },
            "frames": [{"file": "test.py", "line": 1, "function": "f", "code": ""}],
        }
        html = _format_snapshot_banner_html(snapshot)
        assert "<script>" not in html
        assert "&lt;script&gt;" in html

    def test_banner_truncates_long_messages(self):
        snapshot: dict[str, Any] = {
            "exception": {
                "type": "ValueError",
                "message": "x" * 200,
            },
            "frames": [{}],
        }
        html = _format_snapshot_banner_html(snapshot)
        assert "..." in html

    def test_full_html_shows_array_locals(self):
        snapshot: dict[str, Any] = {
            "exception": {"type": "ValueError", "message": "shape mismatch"},
            "frames": [
                {
                    "file": "train.py",
                    "line": 10,
                    "function": "step",
                    "code": "x + y",
                    "locals": {
                        "x": {
                            "__array__": "numpy.ndarray",
                            "shape": [32, 64],
                            "dtype": "float32",
                        },
                    },
                }
            ],
        }
        html = _format_full_snapshot_html(snapshot)
        assert "numpy.ndarray" in html
        assert "shape=[32, 64]" in html

    def test_displays_text_in_terminal(self, snap_dir: Path, capsys: pytest.CaptureFixture[str]):
        ip = _make_ipython(notebook=False)
        from llmdebug.jupyter_plugin import _display_snapshot_banner

        with patch("llmdebug.output.get_latest_snapshot", return_value=SAMPLE_SNAPSHOT):
            _display_snapshot_banner(ip, str(snap_dir))

        captured = capsys.readouterr()
        assert "ValueError" in captured.out
        assert "%llmdebug" in captured.out


# ---------------------------------------------------------------------------
# Magic command tests
# ---------------------------------------------------------------------------


class TestMagicCommands:
    def _get_magic_instance(self, ip: Any, snap_dir: str) -> Any:
        """Install and return an instantiated magic class."""
        install(ip, out_dir=snap_dir)
        assert len(ip._registered_magics) > 0
        magics_cls = ip._registered_magics[0]
        # IPython Magics expects a Configurable shell, pass None to skip validation
        instance = magics_cls(shell=None)
        # Manually set shell for _is_notebook() checks
        instance.shell = ip
        return instance

    def test_magic_show_no_snapshot(self, snap_dir: Path, capsys: pytest.CaptureFixture[str]):
        ip = _make_ipython()
        magic = self._get_magic_instance(ip, str(snap_dir))

        magic.llmdebug("show")
        captured = capsys.readouterr()
        assert "No snapshots found" in captured.out

    def test_magic_show_displays_snapshot(self, snap_dir: Path, capsys: pytest.CaptureFixture[str]):
        ip = _make_ipython()
        magic = self._get_magic_instance(ip, str(snap_dir))

        with patch("llmdebug.output.get_latest_snapshot", return_value=SAMPLE_SNAPSHOT):
            magic.llmdebug("show")

        captured = capsys.readouterr()
        assert "ValueError" in captured.out

    def test_magic_list_shows_recent(self, snap_dir: Path, capsys: pytest.CaptureFixture[str]):
        ip = _make_ipython()
        magic = self._get_magic_instance(ip, str(snap_dir))

        with patch(
            "llmdebug._snapshot_loader.find_all_snapshots",
            return_value=[
                {
                    "path": snap_dir / "snap1.json",
                    "exc_type": "ValueError",
                    "exc_message": "test",
                    "file": "test.py",
                    "line": 1,
                }
            ],
        ):
            magic.llmdebug("list")

        captured = capsys.readouterr()
        assert "ValueError" in captured.out

    def test_magic_list_no_snapshots(self, snap_dir: Path, capsys: pytest.CaptureFixture[str]):
        ip = _make_ipython()
        magic = self._get_magic_instance(ip, str(snap_dir))

        magic.llmdebug("list")
        captured = capsys.readouterr()
        assert "No snapshots found" in captured.out

    def test_magic_list_handles_malformed_entries(
        self,
        snap_dir: Path,
        capsys: pytest.CaptureFixture[str],
    ):
        ip = _make_ipython()
        magic = self._get_magic_instance(ip, str(snap_dir))

        with patch(
            "llmdebug._snapshot_loader.find_all_snapshots",
            return_value=[
                "bad-entry",
                {"exc_type": "ValueError"},
            ],
        ):
            magic.llmdebug("list")

        captured = capsys.readouterr()
        assert "#1" in captured.out
        assert "ValueError" in captured.out

    def test_magic_diff_no_snapshots(self, snap_dir: Path, capsys: pytest.CaptureFixture[str]):
        ip = _make_ipython()
        magic = self._get_magic_instance(ip, str(snap_dir))

        magic.llmdebug("diff")
        captured = capsys.readouterr()
        # Should report not enough snapshots
        assert "Need at least 2" in captured.out or "No snapshots found" in captured.out

    def test_magic_install_uninstall(self, capsys: pytest.CaptureFixture[str]):
        ip = _make_ipython()
        magic = self._get_magic_instance(ip, "/tmp/test_llmdebug_magic")

        magic.llmdebug("uninstall")
        captured = capsys.readouterr()
        assert "uninstalled" in captured.out

    def test_magic_config_shows_settings(self, snap_dir: Path, capsys: pytest.CaptureFixture[str]):
        ip = _make_ipython()
        magic = self._get_magic_instance(ip, str(snap_dir))

        magic.llmdebug("config")
        captured = capsys.readouterr()
        assert "out_dir" in captured.out
        assert "Active configuration" in captured.out

    def test_magic_unknown_shows_help(self, capsys: pytest.CaptureFixture[str]):
        ip = _make_ipython()
        magic = self._get_magic_instance(ip, "/tmp/test_llmdebug_help")

        magic.llmdebug("nonexistent_command")
        captured = capsys.readouterr()
        assert "Usage:" in captured.out
        assert "show" in captured.out
        assert "diff" in captured.out

    def test_magic_default_is_show(self, snap_dir: Path, capsys: pytest.CaptureFixture[str]):
        ip = _make_ipython()
        magic = self._get_magic_instance(ip, str(snap_dir))

        magic.llmdebug("")
        captured = capsys.readouterr()
        assert "No snapshots found" in captured.out


# ---------------------------------------------------------------------------
# Extension API tests
# ---------------------------------------------------------------------------


class TestExtensionAPI:
    def test_load_extension_installs_and_registers_magics(self):
        ip = _make_ipython()
        load_ipython_extension(ip)

        assert len(ip._registered_magics) > 0

    def test_unload_extension_uninstalls(self):
        ip = _make_ipython()
        load_ipython_extension(ip)

        unload_ipython_extension(ip)
        from llmdebug.jupyter_plugin import _installed

        assert not _installed


# ---------------------------------------------------------------------------
# Jupyter context extraction
# ---------------------------------------------------------------------------


class TestJupyterContext:
    def test_extracts_execution_count(self):
        ip = _make_ipython()
        ip.execution_count = 7
        ctx = _extract_jupyter_context(ip)
        assert ctx["execution_count"] == 7

    def test_extracts_cell_source(self):
        ip = _make_ipython()
        ip.history_manager = _FakeHistoryManager()
        ctx = _extract_jupyter_context(ip)
        assert ctx["cell_source"] == "x = 1/0"

    def test_truncates_long_cell_source(self):
        ip = _make_ipython()

        class LongHistory:
            def get_tail(self, n: int = 1, raw: bool = True) -> list[tuple[int, int, str]]:
                return [(1, 1, "x" * 600)]

        ip.history_manager = LongHistory()
        ctx = _extract_jupyter_context(ip)
        assert len(ctx["cell_source"]) == 500
        assert ctx["cell_source"].endswith("...")

    def test_handles_missing_history_manager(self):
        ip = _make_ipython()
        ip.history_manager = None
        ctx = _extract_jupyter_context(ip)
        assert "execution_count" in ctx
        assert "cell_source" not in ctx

    def test_handles_missing_execution_count(self):
        class BareShell:
            pass

        ip = BareShell()
        ctx = _extract_jupyter_context(ip)
        assert "execution_count" not in ctx


# ---------------------------------------------------------------------------
# Edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_handles_keyboard_interrupt(self, snap_dir: Path):
        ip = _make_ipython()
        install(ip, out_dir=str(snap_dir))

        wrapper = ip.showtraceback

        try:
            raise KeyboardInterrupt()
        except KeyboardInterrupt:
            with patch("llmdebug.jupyter_plugin._display_snapshot_banner") as mock_display:
                wrapper()

            mock_display.assert_not_called()

    def test_handles_system_exit(self, snap_dir: Path):
        ip = _make_ipython()
        install(ip, out_dir=str(snap_dir))

        wrapper = ip.showtraceback

        try:
            raise SystemExit(1)
        except SystemExit:
            with patch("llmdebug.jupyter_plugin._display_snapshot_banner") as mock_display:
                wrapper()

            mock_display.assert_not_called()

    def test_is_notebook_detects_zmq(self):
        ip = _make_ipython(notebook=True)
        assert _is_notebook(ip) is True

    def test_is_notebook_detects_terminal(self):
        ip = _make_ipython(notebook=False)
        assert _is_notebook(ip) is False

    def test_empty_snapshot_frames(self):
        snapshot: dict[str, Any] = {
            "exception": {"type": "RuntimeError", "message": "boom"},
            "frames": [],
        }
        html = _format_snapshot_banner_html(snapshot)
        assert "RuntimeError" in html

        text = _format_full_snapshot_text(snapshot)
        assert "RuntimeError" in text

    def test_full_render_handles_malformed_exception_and_frames(self):
        snapshot: dict[str, Any] = {
            "exception": "not-a-dict",
            "frames": [
                "bad-frame",
                {"file": "cell.py", "line": 7, "function": "run", "locals": {}},
            ],
            "crash_frame_index": 0,
        }
        html = _format_full_snapshot_html(snapshot)
        text = _format_full_snapshot_text(snapshot)
        assert "UnknownError" in html
        assert "UnknownError" in text
        assert "cell.py:7" in text


# ---------------------------------------------------------------------------
# __init__.py integration
# ---------------------------------------------------------------------------


class TestInitIntegration:
    def test_load_jupyter_exported(self):
        import llmdebug

        assert hasattr(llmdebug, "load_jupyter")
        assert "load_jupyter" in llmdebug.__all__

    def test_load_ipython_extension_forwarding(self):
        import llmdebug

        ip = _make_ipython()
        llmdebug.load_ipython_extension(ip)
        assert len(ip._registered_magics) > 0
        uninstall()

    def test_unload_ipython_extension_forwarding(self):
        import llmdebug

        ip = _make_ipython()
        llmdebug.load_ipython_extension(ip)
        llmdebug.unload_ipython_extension(ip)

        from llmdebug.jupyter_plugin import _installed

        assert not _installed


# ---------------------------------------------------------------------------
# Additional coverage: display helpers notebook paths
# ---------------------------------------------------------------------------


class TestDisplayHelpersNotebook:
    """Cover notebook (ZMQ) paths that use IPython.display."""

    def test_display_html_or_text_notebook_path(self, capsys: pytest.CaptureFixture[str]) -> None:
        """_display_html_or_text in notebook mode calls HTML/display, not print."""
        ip = _make_ipython(notebook=True)
        displayed: list[Any] = []

        class FakeHTML:
            def __init__(self, content: str) -> None:
                self.content = content

        def fake_display(obj: Any) -> None:
            displayed.append(obj)

        with (
            patch("llmdebug.jupyter_plugin._is_notebook", return_value=True),
            patch("IPython.display.HTML", FakeHTML),
            patch("IPython.display.display", fake_display),
        ):
            _display_html_or_text(
                ip,
                html_fn=lambda: "<b>hello</b>",
                text_fn=lambda: "hello",
            )

        # display was called (not print)
        assert len(displayed) == 1
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_display_html_or_text_notebook_import_error_falls_back(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """_display_html_or_text falls back to print when IPython.display is missing."""
        ip = _make_ipython(notebook=True)

        import builtins

        real_import = builtins.__import__

        def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "IPython.display":
                raise ImportError("no IPython.display")
            return real_import(name, *args, **kwargs)

        with (
            patch("llmdebug.jupyter_plugin._is_notebook", return_value=True),
            patch("builtins.__import__", side_effect=mock_import),
        ):
            _display_html_or_text(
                ip,
                html_fn=lambda: "<b>hello</b>",
                text_fn=lambda: "fallback text",
            )

        captured = capsys.readouterr()
        assert "fallback text" in captured.out

    def test_display_markdown_or_text_notebook_path(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """_display_markdown_or_text in notebook mode calls Markdown/display."""
        ip = _make_ipython(notebook=True)
        displayed: list[Any] = []

        class FakeMarkdown:
            def __init__(self, content: str) -> None:
                self.content = content

        def fake_display(obj: Any) -> None:
            displayed.append(obj)

        with (
            patch("llmdebug.jupyter_plugin._is_notebook", return_value=True),
            patch("IPython.display.Markdown", FakeMarkdown),
            patch("IPython.display.display", fake_display),
        ):
            _display_markdown_or_text(ip, "## Hello")

        assert len(displayed) == 1
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_display_markdown_or_text_notebook_import_error_falls_back(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """_display_markdown_or_text falls back to print when IPython.display is missing."""
        ip = _make_ipython(notebook=True)

        import builtins

        real_import = builtins.__import__

        def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "IPython.display":
                raise ImportError("no IPython.display")
            return real_import(name, *args, **kwargs)

        with (
            patch("llmdebug.jupyter_plugin._is_notebook", return_value=True),
            patch("builtins.__import__", side_effect=mock_import),
        ):
            _display_markdown_or_text(ip, "## markdown text")

        captured = capsys.readouterr()
        assert "## markdown text" in captured.out


# ---------------------------------------------------------------------------
# Additional coverage: crash frame helpers with empty fr0
# ---------------------------------------------------------------------------


class TestCrashFrameHelpersEmptyFr0:
    def test_html_crash_frame_parts_empty_frames(self) -> None:
        """_html_crash_frame_parts returns [] when frames list is empty."""
        result = _html_crash_frame_parts([], 0)
        assert result == []

    def test_html_crash_frame_parts_empty_fr0_from_bad_frames(self) -> None:
        """_html_crash_frame_parts returns [] when _select_frame returns {}."""
        # Pass frames where all items are non-dict → _select_frame returns {}
        result = _html_crash_frame_parts(["bad", 42, None], 0)
        assert result == []

    def test_text_crash_frame_lines_empty_fr0_from_bad_frames(self) -> None:
        """_text_crash_frame_lines returns [] when _select_frame returns {}."""
        result = _text_crash_frame_lines(["bad", 42, None], 0)
        assert result == []


# ---------------------------------------------------------------------------
# Additional coverage: _format_local_value dict paths
# ---------------------------------------------------------------------------


class TestFormatLocalValue:
    def test_truncated_string_value(self) -> None:
        """_format_local_value handles __truncated__ key."""
        val: dict[str, Any] = {"__truncated__": True, "value": "truncated content"}
        result = _format_local_value(val)
        assert result == "truncated content"

    def test_truncated_no_value_key(self) -> None:
        """_format_local_value with __truncated__ but no 'value' key returns dict str."""
        val: dict[str, Any] = {"__truncated__": True}
        result = _format_local_value(val)
        assert "__truncated__" in result

    def test_plain_dict_no_special_keys(self) -> None:
        """_format_local_value with plain dict (no array/truncated keys) returns str."""
        val: dict[str, Any] = {"a": 1, "b": 2}
        result = _format_local_value(val)
        assert isinstance(result, str)
        # Should NOT match array format, just str of the dict truncated to 100
        assert len(result) <= 100


# ---------------------------------------------------------------------------
# Additional coverage: _extract_jupyter_context error branches
# ---------------------------------------------------------------------------


class TestExtractJupyterContextErrorBranches:
    def test_history_manager_get_tail_raises(self) -> None:
        """When get_tail raises, cell_source_error is set."""

        class BadHistory:
            def get_tail(self, n: int = 1, raw: bool = True) -> list[Any]:
                raise RuntimeError("history broken")

        ip = _make_ipython()
        ip.history_manager = BadHistory()
        ctx = _extract_jupyter_context(ip)
        assert "cell_source_error" in ctx
        assert "RuntimeError" in ctx["cell_source_error"]

    def test_history_returns_empty_list(self) -> None:
        """When get_tail returns empty, no cell_source key is added."""

        class EmptyHistory:
            def get_tail(self, n: int = 1, raw: bool = True) -> list[Any]:
                return []

        ip = _make_ipython()
        ip.history_manager = EmptyHistory()
        ctx = _extract_jupyter_context(ip)
        assert "cell_source" not in ctx

    def test_history_cell_source_not_string(self) -> None:
        """When cell_source is not a string, it is not added."""

        class NonStringHistory:
            def get_tail(self, n: int = 1, raw: bool = True) -> list[Any]:
                return [(1, 1, 42)]  # cell source is an int

        ip = _make_ipython()
        ip.history_manager = NonStringHistory()
        ctx = _extract_jupyter_context(ip)
        assert "cell_source" not in ctx


# ---------------------------------------------------------------------------
# Additional coverage: magic install subcommand
# ---------------------------------------------------------------------------


class TestMagicInstallSubcommand:
    def _get_magic_instance(self, ip: Any, snap_dir: str) -> Any:
        install(ip, out_dir=snap_dir)
        magics_cls = ip._registered_magics[0]
        instance = magics_cls(shell=None)
        instance.shell = ip
        return instance

    def test_magic_install_subcommand(
        self, snap_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """%llmdebug install calls install() and prints confirmation."""
        ip = _make_ipython()
        magic = self._get_magic_instance(ip, str(snap_dir))
        # Already installed; calling install again is idempotent
        magic.llmdebug("install")
        captured = capsys.readouterr()
        assert "installed" in captured.out

    def test_magic_config_when_not_installed(
        self, snap_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """_cmd_config when _cfg is None prints 'Not installed' message."""
        import llmdebug.jupyter_plugin as mod

        ip = _make_ipython()
        install(ip, out_dir=str(snap_dir))
        magics_cls = ip._registered_magics[0]
        instance = magics_cls(shell=None)
        instance.shell = ip

        # Force _cfg to None while keeping the magic instance valid
        saved_cfg = mod._cfg
        mod._cfg = None
        try:
            instance.llmdebug("config")
        finally:
            mod._cfg = saved_cfg

        captured = capsys.readouterr()
        assert "Not installed" in captured.out


# ---------------------------------------------------------------------------
# Additional coverage: _cmd_list notebook (HTML) path
# ---------------------------------------------------------------------------


class TestMagicListNotebookPath:
    def _get_magic_instance(self, ip: Any, snap_dir: str) -> Any:
        install(ip, out_dir=snap_dir)
        magics_cls = ip._registered_magics[0]
        instance = magics_cls(shell=None)
        instance.shell = ip
        return instance

    def test_magic_list_notebook_html_path(
        self, snap_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """_cmd_list in notebook mode calls _display_html_or_text with HTML builder."""
        ip = _make_ipython(notebook=True)
        magic = self._get_magic_instance(ip, str(snap_dir))

        entries = [
            {
                "path": snap_dir / "snap1.json",
                "exc_type": "ValueError",
                "exc_message": "bad value",
                "file": "test.py",
                "line": 5,
            }
        ]

        html_rendered: list[str] = []

        class FakeHTML:
            def __init__(self, content: str) -> None:
                html_rendered.append(content)

        def fake_display(obj: Any) -> None:
            pass

        with (
            patch(
                "llmdebug._snapshot_loader.find_all_snapshots",
                return_value=entries,
            ),
            patch("llmdebug.jupyter_plugin._is_notebook", return_value=True),
            patch("IPython.display.HTML", FakeHTML),
            patch("IPython.display.display", fake_display),
        ):
            magic.llmdebug("list")

        # HTML branch was used — HTML was rendered with ValueError
        assert any("ValueError" in h for h in html_rendered)

    def test_magic_list_escapes_the_line_value(self, snap_dir: Path) -> None:
        """RCA-08: index rows from a shared directory are untrusted, line included."""
        ip = _make_ipython(notebook=True)
        magic = self._get_magic_instance(ip, str(snap_dir))
        entries = [
            {"exc_type": "E", "exc_message": "m", "file": "f.py", "line": _HOSTILE_HTML},
        ]
        html_rendered: list[str] = []

        with (
            patch("llmdebug._snapshot_loader.find_all_snapshots", return_value=entries),
            patch("llmdebug.jupyter_plugin._is_notebook", return_value=True),
            patch("IPython.display.HTML", lambda content: html_rendered.append(content)),
            patch("IPython.display.display", lambda _obj: None),
        ):
            magic.llmdebug("list")

        assert _HOSTILE_HTML not in html_rendered[0]
        assert "&lt;img" in html_rendered[0]


_HOSTILE_HTML = "<img src=x onerror=alert(1)>"


class TestHtmlEscaping:
    """RCA-08: snapshot values interpolated into notebook HTML must be escaped."""

    def _hostile_snapshot(self) -> dict[str, Any]:
        snapshot = copy.deepcopy(SAMPLE_SNAPSHOT)
        snapshot["frames"][0]["line"] = _HOSTILE_HTML
        snapshot["jupyter"]["execution_count"] = "<script>alert(2)</script>"
        return snapshot

    def test_banner_escapes_the_crash_line(self) -> None:
        html = _format_snapshot_banner_html(self._hostile_snapshot())

        assert _HOSTILE_HTML not in html
        assert "&lt;img src=x onerror=alert(1)&gt;" in html

    def test_full_view_escapes_line_and_execution_count(self) -> None:
        html = _format_full_snapshot_html(self._hostile_snapshot())

        assert _HOSTILE_HTML not in html
        assert "<script>" not in html
        assert "Cell In[&lt;script&gt;" in html


# ---------------------------------------------------------------------------
# Additional coverage: _cmd_diff paths
# ---------------------------------------------------------------------------


class TestMagicDiffPaths:
    def _get_magic_instance(self, ip: Any, snap_dir: str) -> Any:
        install(ip, out_dir=snap_dir)
        magics_cls = ip._registered_magics[0]
        instance = magics_cls(shell=None)
        instance.shell = ip
        return instance

    def test_magic_diff_renders_text(
        self, snap_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """_cmd_diff with two snapshots formats and displays diff."""
        import json

        ip = _make_ipython()
        magic = self._get_magic_instance(ip, str(snap_dir))

        snap_a: dict[str, Any] = {
            "exception": {"type": "ValueError", "message": "first"},
            "frames": [],
        }
        snap_b: dict[str, Any] = {
            "exception": {"type": "ValueError", "message": "second"},
            "frames": [],
        }
        p_a = snap_dir / "snap_a.json"
        p_b = snap_dir / "snap_b.json"
        p_a.write_text(json.dumps(snap_a))
        p_b.write_text(json.dumps(snap_b))

        entries = [
            {"path": p_b, "exc_type": "ValueError"},
            {"path": p_a, "exc_type": "ValueError"},
        ]

        with (
            patch(
                "llmdebug._snapshot_loader.find_all_snapshots",
                return_value=entries,
            ),
            patch(
                "llmdebug._snapshot_loader.resolve_snapshot_ref",
                side_effect=[snap_b, snap_a],
            ),
        ):
            magic.llmdebug("diff")

        captured = capsys.readouterr()
        # diff was displayed
        assert len(captured.out) > 0

    def test_magic_diff_file_not_found(
        self, snap_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """_cmd_diff when resolve_snapshot_ref raises FileNotFoundError prints error."""
        ip = _make_ipython()
        magic = self._get_magic_instance(ip, str(snap_dir))

        with (
            patch(
                "llmdebug._snapshot_loader.find_all_snapshots",
                return_value=[],
            ),
            patch(
                "llmdebug._snapshot_loader.resolve_snapshot_ref",
                side_effect=FileNotFoundError("no snapshot"),
            ),
        ):
            magic.llmdebug("diff")

        captured = capsys.readouterr()
        assert "no snapshot" in captured.out

    def test_magic_diff_notebook_markdown_path(
        self, snap_dir: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """_cmd_diff in notebook mode uses _display_markdown_or_text → Markdown."""
        import json

        ip = _make_ipython(notebook=True)
        magic = self._get_magic_instance(ip, str(snap_dir))

        snap_a: dict[str, Any] = {
            "exception": {"type": "ValueError", "message": "first"},
            "frames": [],
        }
        snap_b: dict[str, Any] = {
            "exception": {"type": "TypeError", "message": "second"},
            "frames": [],
        }
        p_a = snap_dir / "snap_a.json"
        p_b = snap_dir / "snap_b.json"
        p_a.write_text(json.dumps(snap_a))
        p_b.write_text(json.dumps(snap_b))

        entries = [
            {"path": p_b, "exc_type": "TypeError"},
            {"path": p_a, "exc_type": "ValueError"},
        ]

        displayed: list[Any] = []

        class FakeMarkdown:
            def __init__(self, content: str) -> None:
                displayed.append(content)

        with (
            patch(
                "llmdebug._snapshot_loader.find_all_snapshots",
                return_value=entries,
            ),
            patch(
                "llmdebug._snapshot_loader.resolve_snapshot_ref",
                side_effect=[snap_b, snap_a],
            ),
            patch("llmdebug.jupyter_plugin._is_notebook", return_value=True),
            patch("IPython.display.Markdown", FakeMarkdown),
            patch("IPython.display.display", lambda _: None),
        ):
            magic.llmdebug("diff")

        assert len(displayed) > 0


# ---------------------------------------------------------------------------
# Additional coverage: install() with ipython=None paths
# ---------------------------------------------------------------------------


class TestInstallNoIPython:
    def test_install_raises_when_ipython_not_installed(self) -> None:
        """install(None) raises RuntimeError when IPython is not importable."""
        import builtins

        real_import = builtins.__import__

        def mock_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "IPython":
                raise ImportError("no IPython")
            return real_import(name, *args, **kwargs)

        with (
            patch("builtins.__import__", side_effect=mock_import),
            pytest.raises(RuntimeError, match="IPython is not installed"),
        ):
            install(None)

    def test_install_raises_when_no_active_session(self) -> None:
        """install(None) raises RuntimeError when get_ipython() returns None."""
        import sys

        class FakeIPythonPkg:
            def get_ipython(self) -> None:
                return None

        with (
            patch.dict(sys.modules, {"IPython": FakeIPythonPkg()}),
            pytest.raises(RuntimeError, match="No active IPython session"),
        ):  # type: ignore[arg-type]
            install(None)

    def test_install_none_with_active_session_uses_it(self) -> None:
        """install(None) uses ipython returned by get_ipython() when it is not None."""
        import sys

        ip = _make_ipython()

        class FakeIPythonPkg:
            def get_ipython(self) -> Any:
                return ip

        with patch.dict(sys.modules, {"IPython": FakeIPythonPkg()}):  # type: ignore[arg-type]
            install(None, out_dir="/tmp/test_llmdebug_autodetect")

        assert ip.showtraceback is not _FakeTerminalShell.showtraceback


# ---------------------------------------------------------------------------
# Additional coverage: uninstall when _ipython_ref is None
# ---------------------------------------------------------------------------


class TestUninstallEdgeCases:
    def test_uninstall_when_ipython_ref_none(self) -> None:
        """uninstall() when _installed=True but _ipython_ref=None still resets state."""
        import llmdebug.jupyter_plugin as mod

        # Manually set state to simulate installed without ipython ref
        mod._installed = True
        mod._ipython_ref = None
        mod._original_showtraceback = None
        mod._cfg = None
        mod._limiter = None

        uninstall()  # Should not raise

        assert mod._installed is False
        assert mod._ipython_ref is None
