"""Tests for baselines.subprocess_monitor — monitored subprocess execution."""

from __future__ import annotations

import subprocess
import sys
from typing import Any
from unittest.mock import MagicMock

import pytest

from baselines.subprocess_monitor import (
    MonitoredSubprocessResult,
    run_monitored_subprocess,
    sanitize_progress_excerpt,
)


class TestSanitizeProgressExcerpt:
    """Tests for sanitize_progress_excerpt()."""

    def test_empty_string_returns_empty(self) -> None:
        assert sanitize_progress_excerpt("") == ""

    def test_none_ish_returns_empty(self) -> None:
        assert sanitize_progress_excerpt("   ") == ""

    def test_short_string_returned_as_is(self) -> None:
        assert sanitize_progress_excerpt("hello world") == "hello world"

    def test_long_string_truncated_with_ellipsis(self) -> None:
        long = "a" * 200
        result = sanitize_progress_excerpt(long, max_chars=20)
        assert len(result) <= 20
        assert result.endswith("...")

    def test_whitespace_collapsed(self) -> None:
        assert sanitize_progress_excerpt("  hello   world  ") == "hello world"

    def test_max_chars_one(self) -> None:
        """Verify truncation with very small max_chars."""
        result = sanitize_progress_excerpt("abcdef", max_chars=4)
        assert result.endswith("...")
        assert len(result) <= 4


class TestRunMonitoredSubprocess:
    """Tests for run_monitored_subprocess()."""

    def test_simple_echo(self) -> None:
        """Verify basic subprocess captures stdout."""
        result = run_monitored_subprocess(
            cmd=[sys.executable, "-c", "print('hello')"],
            cwd=None,
            env=None,
            timeout_sec=10,
        )
        assert result.returncode == 0
        assert "hello" in result.stdout
        assert not result.timed_out
        assert result.timeout_sec is None

    def test_nonzero_exit_code(self) -> None:
        """Verify non-zero returncode is captured."""
        result = run_monitored_subprocess(
            cmd=[sys.executable, "-c", "import sys; sys.exit(42)"],
            cwd=None,
            env=None,
            timeout_sec=10,
        )
        assert result.returncode == 42
        assert not result.timed_out

    def test_stderr_captured(self) -> None:
        """Verify stderr output is captured."""
        result = run_monitored_subprocess(
            cmd=[sys.executable, "-c", "import sys; sys.stderr.write('err\\n')"],
            cwd=None,
            env=None,
            timeout_sec=10,
        )
        assert "err" in result.stderr

    def test_timeout_kills_process(self) -> None:
        """Verify process is killed on timeout (lines 170-172)."""
        result = run_monitored_subprocess(
            cmd=[sys.executable, "-c", "import time; time.sleep(60)"],
            cwd=None,
            env=None,
            timeout_sec=1,
        )
        assert result.timed_out is True
        assert result.timeout_sec == 1
        assert result.returncode != 0

    def test_progress_callback_receives_events(self) -> None:
        """Verify progress callback is called with output excerpts."""
        events: list[tuple[str, dict[str, Any] | None]] = []

        def callback(event: str, payload: dict[str, Any] | None) -> None:
            events.append((event, payload))

        result = run_monitored_subprocess(
            cmd=[sys.executable, "-c", "print('progress line')"],
            cwd=None,
            env=None,
            timeout_sec=10,
            progress_callback=callback,
            progress_event="test_progress",
        )
        assert result.returncode == 0
        assert any(e[0] == "test_progress" for e in events)

    def test_no_callback_emit_returns_early(self) -> None:
        """When progress_callback is None, _emit returns early (lines 69-70)."""
        result = run_monitored_subprocess(
            cmd=[sys.executable, "-c", "print('hello')"],
            cwd=None,
            env=None,
            timeout_sec=10,
            progress_callback=None,
        )
        assert result.returncode == 0

    def test_callback_exception_swallowed(self) -> None:
        """Callback that raises should not crash the monitor (lines 76-77)."""

        def bad_callback(event: str, payload: dict[str, Any] | None) -> None:
            raise ValueError("boom")

        result = run_monitored_subprocess(
            cmd=[sys.executable, "-c", "print('hello')"],
            cwd=None,
            env=None,
            timeout_sec=10,
            progress_callback=bad_callback,
        )
        assert result.returncode == 0

    def test_state_supplier_called(self) -> None:
        """Verify state_supplier is polled during execution (lines 94-105)."""
        _states: list[dict[str, Any]] = []
        call_count = 0

        def supplier() -> dict[str, Any]:
            nonlocal call_count
            call_count += 1
            return {"step": call_count}

        events: list[tuple[str, dict[str, Any] | None]] = []

        def callback(event: str, payload: dict[str, Any] | None) -> None:
            events.append((event, payload))

        result = run_monitored_subprocess(
            cmd=[sys.executable, "-c", "import time; time.sleep(0.5); print('done')"],
            cwd=None,
            env=None,
            timeout_sec=10,
            progress_callback=callback,
            state_supplier=supplier,
            poll_interval_sec=0.1,
        )
        assert result.returncode == 0
        assert call_count >= 1

    def test_state_supplier_returns_non_mapping(self) -> None:
        """State supplier returning non-Mapping is silently ignored (lines 98-99)."""

        def bad_supplier() -> Any:
            return "not a mapping"

        events: list[tuple[str, dict[str, Any] | None]] = []

        def callback(event: str, payload: dict[str, Any] | None) -> None:
            events.append((event, payload))

        result = run_monitored_subprocess(
            cmd=[sys.executable, "-c", "import time; time.sleep(0.3); print('ok')"],
            cwd=None,
            env=None,
            timeout_sec=10,
            progress_callback=callback,
            state_supplier=bad_supplier,
            poll_interval_sec=0.1,
        )
        assert result.returncode == 0

    def test_state_supplier_exception_swallowed(self) -> None:
        """State supplier that raises should not crash (lines 96-97)."""

        def error_supplier() -> dict[str, Any]:
            raise RuntimeError("supplier error")

        events: list[tuple[str, dict[str, Any] | None]] = []

        def callback(event: str, payload: dict[str, Any] | None) -> None:
            events.append((event, payload))

        result = run_monitored_subprocess(
            cmd=[sys.executable, "-c", "import time; time.sleep(0.3); print('ok')"],
            cwd=None,
            env=None,
            timeout_sec=10,
            progress_callback=callback,
            state_supplier=error_supplier,
            poll_interval_sec=0.1,
        )
        assert result.returncode == 0

    def test_state_supplier_dedup_skips_unchanged(self) -> None:
        """Duplicate state is not re-emitted (lines 102-103)."""
        emit_count = 0

        def constant_supplier() -> dict[str, Any]:
            return {"constant": True}

        def callback(event: str, payload: dict[str, Any] | None) -> None:
            nonlocal emit_count
            if payload and "constant" in payload:
                emit_count += 1

        result = run_monitored_subprocess(
            cmd=[sys.executable, "-c", "import time; time.sleep(0.5); print('ok')"],
            cwd=None,
            env=None,
            timeout_sec=10,
            progress_callback=callback,
            state_supplier=constant_supplier,
            poll_interval_sec=0.1,
        )
        assert result.returncode == 0
        # First emit + force=True at end = at most 2 for constant state
        assert emit_count <= 3

    def test_progress_base_payload_merged(self) -> None:
        """Verify base_payload is included in callback payloads."""
        events: list[dict[str, Any] | None] = []

        def callback(event: str, payload: dict[str, Any] | None) -> None:
            events.append(payload)

        run_monitored_subprocess(
            cmd=[sys.executable, "-c", "print('data')"],
            cwd=None,
            env=None,
            timeout_sec=10,
            progress_callback=callback,
            progress_base_payload={"run_id": "test123"},
        )
        payloads_with_run_id = [e for e in events if e and e.get("run_id") == "test123"]
        assert len(payloads_with_run_id) > 0

    def test_pipes_unavailable_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Verify RuntimeError when pipes are None (lines 118-119)."""

        class FakePopen:
            """Mock Popen that returns None pipes."""

            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self.stdout = None
                self.stderr = None
                self.pid = 0

            def poll(self) -> int:
                return 0

            def wait(self, timeout: float | None = None) -> int:
                return 0

            def kill(self) -> None:
                pass

        monkeypatch.setattr(
            "baselines.subprocess_monitor.subprocess.Popen",
            FakePopen,
        )
        with pytest.raises(RuntimeError, match="subprocess pipes unavailable"):
            run_monitored_subprocess(
                cmd=["echo", "test"],
                cwd=None,
                env=None,
                timeout_sec=10,
            )

    def test_reader_pipe_close_exception_swallowed(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """pipe.close() raising in _reader is silently swallowed (lines 87-88).

        We wrap the real Popen so its stdout/stderr close() methods raise,
        then verify the subprocess still completes normally.
        """
        real_popen = subprocess.Popen

        class _BadClosePopen(real_popen):  # type: ignore[type-arg]
            def __init__(self, *args: Any, **kwargs: Any) -> None:
                super().__init__(*args, **kwargs)
                # Wrap stdout pipe to raise on close()
                if self.stdout is not None:
                    orig_stdout = self.stdout
                    mock_stdout = MagicMock(wraps=orig_stdout)
                    mock_stdout.readline = orig_stdout.readline
                    mock_stdout.close = MagicMock(
                        side_effect=OSError("stdout close failed"),
                    )
                    self.stdout = mock_stdout
                # Wrap stderr pipe to raise on close()
                if self.stderr is not None:
                    orig_stderr = self.stderr
                    mock_stderr = MagicMock(wraps=orig_stderr)
                    mock_stderr.readline = orig_stderr.readline
                    mock_stderr.close = MagicMock(
                        side_effect=OSError("stderr close failed"),
                    )
                    self.stderr = mock_stderr

        monkeypatch.setattr(
            "baselines.subprocess_monitor.subprocess.Popen",
            _BadClosePopen,
        )
        result = run_monitored_subprocess(
            cmd=[sys.executable, "-c", "print('hi')"],
            cwd=None,
            env=None,
            timeout_sec=10,
            poll_interval_sec=0.05,
        )
        assert result.returncode == 0
        assert "hi" in result.stdout

    def test_returncode_none_after_loop_uses_wait(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """process.poll() → None after loop triggers wait() fallback (line 183).

        Uses a fully-fake Popen with StringIO pipes so we can control the
        exact poll() call sequence:

        * Calls 1-5: return ``None`` (process appears running while readers
          drain their pipes and post sentinel events).
        * Call 6+: return ``0`` — except call 7 which returns ``None`` once.
          Call 6 lets the loop break (streams done, queue empty, poll != None).
          Call 7 is the post-loop ``process.poll()`` at line 181 → ``None`` →
          code falls through to ``process.wait(timeout=1.0)`` at line 183.
        """
        import io

        class _FakePipe:
            """Pipe mock backed by StringIO for instant reading."""

            def __init__(self, text: str) -> None:
                self._sio = io.StringIO(text)

            def readline(self) -> str:
                return self._sio.readline()

            def close(self) -> None:
                pass

        class _PostLoopNullPopen:
            """Fake Popen: poll() returns None once at the post-loop call."""

            def __init__(self, *args: Any, **kwargs: Any) -> None:
                self.pid = 12345
                self.stdout = _FakePipe("done\n")
                self.stderr = _FakePipe("")
                self._call = 0
                self._zero_count = 0

            def poll(self) -> int | None:
                self._call += 1
                if self._call <= 5:
                    return None  # process "still running"
                self._zero_count += 1
                if self._zero_count == 2:  # post-loop call at line 181
                    return None
                return 0

            def wait(self, timeout: float | None = None) -> int:
                return 0

            def kill(self) -> None:
                pass

        monkeypatch.setattr(
            "baselines.subprocess_monitor.subprocess.Popen",
            _PostLoopNullPopen,
        )
        result = run_monitored_subprocess(
            cmd=["ignored"],
            cwd=None,
            env=None,
            timeout_sec=10,
            poll_interval_sec=0.05,
        )
        assert result.returncode == 0
        assert "done" in result.stdout

    def test_duplicate_excerpt_suppressed(self) -> None:
        """Repeated identical output lines are de-duplicated (branch 161→165).

        Subprocess emits the same line many times; the callback should only
        receive one excerpt event for that source/content pair.
        """
        excerpts: list[str] = []

        def callback(event: str, payload: dict[str, Any] | None) -> None:
            if payload and "activity_excerpt" in payload:
                excerpts.append(payload["activity_excerpt"])

        # Emit 5 identical lines — only one should generate an excerpt event
        script = "for _ in range(5): print('same')"
        result = run_monitored_subprocess(
            cmd=[sys.executable, "-c", script],
            cwd=None,
            env=None,
            timeout_sec=10,
            progress_callback=callback,
            poll_interval_sec=0.05,
        )
        assert result.returncode == 0
        same_excerpts = [e for e in excerpts if e == "same"]
        assert len(same_excerpts) == 1


class TestMonitoredSubprocessResult:
    """Smoke tests for the dataclass."""

    def test_defaults(self) -> None:
        r = MonitoredSubprocessResult(returncode=0, duration_sec=1.0, stdout="", stderr="")
        assert r.timed_out is False
        assert r.timeout_sec is None

    def test_timed_out_fields(self) -> None:
        r = MonitoredSubprocessResult(
            returncode=-9,
            duration_sec=5.0,
            stdout="",
            stderr="",
            timed_out=True,
            timeout_sec=5,
        )
        assert r.timed_out is True
        assert r.timeout_sec == 5
