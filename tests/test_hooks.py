"""Tests for production exception hooks."""

from __future__ import annotations

import re
import sys
import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from llmdebug.hooks import (
    PII_PATTERNS,
    HookName,
    _build_production_config,
    _capture_safely,
    _RateLimiter,
    install_hooks,
    uninstall_hooks,
)

from .conftest import read_latest_snapshot

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _cleanup_hooks():
    """Ensure hooks are uninstalled after each test."""
    yield
    uninstall_hooks()


def _make_exc_with_tb() -> tuple[BaseException, Any]:
    """Create a real exception with traceback for testing."""
    try:
        raise ValueError("test crash")
    except ValueError as e:
        return e, e.__traceback__


# ---------------------------------------------------------------------------
# _RateLimiter
# ---------------------------------------------------------------------------


class TestRateLimiter:
    def test_rejects_zero_max_captures(self):
        with pytest.raises(ValueError, match="max_captures must be >= 1"):
            _RateLimiter(max_captures=0, window_seconds=60.0)

    def test_rejects_negative_max_captures(self):
        with pytest.raises(ValueError, match="max_captures must be >= 1"):
            _RateLimiter(max_captures=-1, window_seconds=60.0)

    def test_rejects_zero_window(self):
        with pytest.raises(ValueError, match="window_seconds must be > 0"):
            _RateLimiter(max_captures=10, window_seconds=0)

    def test_rejects_negative_window(self):
        with pytest.raises(ValueError, match="window_seconds must be > 0"):
            _RateLimiter(max_captures=10, window_seconds=-1.0)

    def test_allows_under_limit(self):
        limiter = _RateLimiter(max_captures=3, window_seconds=60.0)
        assert limiter.allow("ValueError", "test.py", 10) is True
        assert limiter.allow("ValueError", "test.py", 10) is True
        assert limiter.allow("ValueError", "test.py", 10) is True

    def test_blocks_over_limit(self):
        limiter = _RateLimiter(max_captures=2, window_seconds=60.0)
        assert limiter.allow("ValueError", "test.py", 10) is True
        assert limiter.allow("ValueError", "test.py", 10) is True
        assert limiter.allow("ValueError", "test.py", 10) is False

    def test_window_expiry(self):
        limiter = _RateLimiter(max_captures=1, window_seconds=0.1)
        assert limiter.allow("ValueError", "test.py", 10) is True
        assert limiter.allow("ValueError", "test.py", 10) is False
        # Wait for window to expire
        time.sleep(0.15)
        assert limiter.allow("ValueError", "test.py", 10) is True

    def test_different_signatures_independent(self):
        limiter = _RateLimiter(max_captures=1, window_seconds=60.0)
        assert limiter.allow("ValueError", "test.py", 10) is True
        assert limiter.allow("ValueError", "test.py", 10) is False
        # Different exception type
        assert limiter.allow("TypeError", "test.py", 10) is True
        # Different file
        assert limiter.allow("ValueError", "other.py", 10) is True
        # Different line
        assert limiter.allow("ValueError", "test.py", 20) is True

    def test_thread_safety(self):
        limiter = _RateLimiter(max_captures=100, window_seconds=60.0)
        results: list[bool] = []
        lock = threading.Lock()

        def hammer():
            for _ in range(50):
                result = limiter.allow("ValueError", "test.py", 10)
                with lock:
                    results.append(result)

        threads = [threading.Thread(target=hammer) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # Exactly 100 should be True, rest False
        assert sum(results) == 100
        assert len(results) == 500

    def test_cleanup_expired_removes_stale_entries(self):
        limiter = _RateLimiter(max_captures=2, window_seconds=0.01)
        assert limiter.allow("ValueError", "test.py", 10) is True
        time.sleep(0.02)
        limiter._cleanup_expired(time.monotonic())
        assert limiter._buckets == {}

    def test_rate_limit_warning_only_emitted_once(self, capsys: pytest.CaptureFixture[str]):
        limiter = _RateLimiter(max_captures=1, window_seconds=60.0)
        assert limiter.allow("ValueError", "test.py", 10) is True
        assert limiter.allow("ValueError", "test.py", 10) is False
        assert limiter.allow("ValueError", "test.py", 10) is False
        err = capsys.readouterr().err
        assert err.count("rate limit reached") == 1


# ---------------------------------------------------------------------------
# _capture_safely
# ---------------------------------------------------------------------------


class TestCaptureSafely:
    def test_captures_exception(self, tmp_path: Path):
        cfg = _build_production_config(str(tmp_path), pii_safe=False)
        limiter = _RateLimiter(10, 60.0)
        exc, tb = _make_exc_with_tb()

        _capture_safely("test", exc, tb, cfg, limiter)

        snapshot = read_latest_snapshot(tmp_path)
        assert snapshot["exception"]["type"] == "ValueError"
        assert snapshot["exception"]["message"] == "test crash"
        assert snapshot["name"] == "test"

    def test_swallows_capture_errors(self):
        """A broken config should not cause _capture_safely to raise."""
        # Use an invalid out_dir that will fail on write
        cfg = _build_production_config("/nonexistent/deeply/nested/path", pii_safe=False)
        limiter = _RateLimiter(10, 60.0)
        exc, tb = _make_exc_with_tb()

        # Should not raise
        _capture_safely("test", exc, tb, cfg, limiter)

        # The error should prevent any snapshot file from being created
        assert not Path("/nonexistent/deeply/nested/path").exists()

    def test_rate_limited(self, tmp_path: Path):
        cfg = _build_production_config(str(tmp_path), pii_safe=False)
        limiter = _RateLimiter(1, 60.0)
        exc, tb = _make_exc_with_tb()

        _capture_safely("test", exc, tb, cfg, limiter)
        # Second call with same signature should be rate-limited
        _capture_safely("test_2", exc, tb, cfg, limiter)

        # Only one snapshot should exist; latest still has the first name
        snapshot = read_latest_snapshot(tmp_path)
        assert snapshot["name"] == "test"

    def test_handles_none_traceback(self, tmp_path: Path):
        """When tb is None, capture_safely should handle gracefully."""
        cfg = _build_production_config(str(tmp_path), pii_safe=False)
        limiter = _RateLimiter(10, 60.0)
        exc = ValueError("no traceback")

        # Should not raise — tb.extract_tb(None) returns empty list
        _capture_safely("test", exc, None, cfg, limiter)

        # A snapshot should still be created despite None traceback
        snapshot = read_latest_snapshot(tmp_path)
        assert snapshot["exception"]["type"] == "ValueError"

    def test_includes_extra(self, tmp_path: Path):
        cfg = _build_production_config(str(tmp_path), pii_safe=False)
        limiter = _RateLimiter(10, 60.0)
        exc, tb = _make_exc_with_tb()

        _capture_safely("test", exc, tb, cfg, limiter, extra={"hook": "sys.excepthook"})

        snapshot = read_latest_snapshot(tmp_path)
        assert snapshot["hook"] == "sys.excepthook"


# ---------------------------------------------------------------------------
# install_hooks / uninstall_hooks
# ---------------------------------------------------------------------------


class TestInstallHooks:
    def test_installs_and_uninstalls(self):
        original_excepthook = sys.excepthook
        original_threading = threading.excepthook
        original_unraisable = sys.unraisablehook

        install_hooks()

        assert sys.excepthook is not original_excepthook
        assert threading.excepthook is not original_threading
        assert sys.unraisablehook is not original_unraisable

        uninstall_hooks()

        assert sys.excepthook is original_excepthook
        assert threading.excepthook is original_threading
        assert sys.unraisablehook is original_unraisable

    def test_idempotent_install(self):
        install_hooks()
        hook_after_first = sys.excepthook

        install_hooks()  # Second call should be a no-op
        assert sys.excepthook is hook_after_first

    def test_idempotent_uninstall(self):
        """Uninstalling when not installed should be a no-op."""
        original_excepthook = sys.excepthook
        original_threading = threading.excepthook
        original_unraisable = sys.unraisablehook

        uninstall_hooks()  # Should not raise

        assert sys.excepthook is original_excepthook
        assert threading.excepthook is original_threading
        assert sys.unraisablehook is original_unraisable

    def test_preserves_original_excepthook(self):
        """Our hook should call the original after capturing."""
        calls: list[tuple[Any, ...]] = []
        original = sys.excepthook

        def tracking_hook(exc_type, exc_value, exc_tb):
            calls.append((exc_type, exc_value, exc_tb))

        sys.excepthook = tracking_hook
        try:
            install_hooks()
            exc, tb = _make_exc_with_tb()
            sys.excepthook(type(exc), exc, tb)

            # Original (tracking) hook should have been called
            assert len(calls) == 1
            assert calls[0][0] is ValueError
        finally:
            sys.excepthook = original

    def test_excepthook_captures_snapshot(self, tmp_path: Path):
        # Use a no-op original to suppress traceback printing
        original = sys.excepthook
        sys.excepthook = lambda *a: None
        try:
            install_hooks(out_dir=str(tmp_path), pii_safe=False)
            exc, tb = _make_exc_with_tb()
            sys.excepthook(type(exc), exc, tb)
        finally:
            # uninstall_hooks will restore the no-op; reset to real original
            uninstall_hooks()
            sys.excepthook = original

        snapshot = read_latest_snapshot(tmp_path)
        assert snapshot["name"] == "unhandled"
        assert snapshot["hook"] == "sys.excepthook"
        assert snapshot["exception"]["type"] == "ValueError"

    def test_threading_hook_captures_snapshot(self, tmp_path: Path):
        # Replace threading.excepthook with a no-op before install so our
        # wrapper's "call original" doesn't trigger pytest's thread exception
        # handler.
        original = threading.excepthook
        threading.excepthook = lambda args: None
        try:
            install_hooks(out_dir=str(tmp_path), pii_safe=False, hooks=("threading",))
            exc, tb = _make_exc_with_tb()

            args = SimpleNamespace(
                exc_type=type(exc),
                exc_value=exc,
                exc_traceback=tb,
                thread=SimpleNamespace(name="worker-1"),
            )
            args = cast(Any, args)
            threading.excepthook(args)
        finally:
            uninstall_hooks()
            threading.excepthook = original

        snapshot = read_latest_snapshot(tmp_path)
        assert snapshot["name"] == "thread_crash"
        assert snapshot["hook"] == "threading.excepthook"
        assert snapshot["thread"] == "worker-1"

    def test_unraisable_hook_captures_snapshot(self, tmp_path: Path):
        # Replace sys.unraisablehook with a no-op before install so our
        # wrapper's "call original" doesn't trigger pytest's handler.
        original = sys.unraisablehook
        sys.unraisablehook = lambda u: None
        try:
            install_hooks(out_dir=str(tmp_path), pii_safe=False, hooks=("unraisable",))
            exc, tb = _make_exc_with_tb()

            unraisable = SimpleNamespace(
                exc_type=type(exc),
                exc_value=exc,
                exc_traceback=tb,
                err_msg=None,
                object=None,
            )
            unraisable = cast(Any, unraisable)
            sys.unraisablehook(unraisable)
        finally:
            uninstall_hooks()
            sys.unraisablehook = original

        snapshot = read_latest_snapshot(tmp_path)
        assert snapshot["name"] == "unraisable"
        assert snapshot["hook"] == "sys.unraisablehook"

    @staticmethod
    def _call_unraisable_hook(tmp_path: Path, exc: BaseException, obj: object) -> list[Any]:
        forwarded: list[Any] = []
        original = sys.unraisablehook
        sys.unraisablehook = forwarded.append
        try:
            install_hooks(out_dir=str(tmp_path), pii_safe=False, hooks=("unraisable",))
            try:
                raise exc
            except BaseException as raised:
                tb = raised.__traceback__
            unraisable = SimpleNamespace(
                exc_type=type(exc), exc_value=exc, exc_traceback=tb, err_msg=None, object=obj
            )
            sys.unraisablehook(cast(Any, unraisable))
        finally:
            uninstall_hooks()
            sys.unraisablehook = original
        return forwarded

    @pytest.mark.parametrize("exc", [KeyboardInterrupt(), SystemExit(3)])
    def test_unraisable_hook_skips_interrupts(self, tmp_path: Path, exc: BaseException):
        """CAP-18: KeyboardInterrupt and SystemExit are never captured, as documented."""
        forwarded = self._call_unraisable_hook(tmp_path, exc, None)

        assert len(forwarded) == 1  # the original hook still runs
        assert not list(tmp_path.glob("*.json"))

    def test_unraisable_hook_records_falsy_objects(self, tmp_path: Path):
        """CAP-18: an empty container is still the object, not "no object"."""
        self._call_unraisable_hook(tmp_path, ValueError("in __del__"), [])

        assert read_latest_snapshot(tmp_path)["object"] == "[]"

    def test_asyncio_handler_captures_snapshot(self, tmp_path: Path):
        import asyncio

        async def run():
            install_hooks(
                out_dir=str(tmp_path),
                pii_safe=False,
                hooks=("asyncio",),
            )

            loop = asyncio.get_running_loop()
            exc, tb = _make_exc_with_tb()
            exc.__traceback__ = tb
            loop.call_exception_handler({"message": "test failure", "exception": exc})

        asyncio.run(run())

        snapshot = read_latest_snapshot(tmp_path)
        assert snapshot["name"] == "asyncio"
        assert snapshot["hook"] == "asyncio"
        assert snapshot["message"] == "test failure"

    def test_pii_safe_applies_patterns(self):
        cfg = _build_production_config(".llmdebug", pii_safe=True)
        assert len(cfg.redact) == len(PII_PATTERNS)
        for pattern in cfg.redact:
            assert isinstance(pattern, str)

    def test_pii_safe_disabled(self):
        cfg = _build_production_config(".llmdebug", pii_safe=False)
        assert cfg.redact == ()

    def test_redaction_profile_override_applies(self):
        cfg = _build_production_config(
            ".llmdebug",
            pii_safe=False,
            redaction_profile="prod",
        )
        assert cfg.redaction_profile == "prod"
        assert cfg.redact_traceback is True
        assert cfg.redact_exception_strings is True
        assert len(cfg.redact) > 0

    def test_pii_safe_defaults_to_prod_profile(self):
        cfg = _build_production_config(".llmdebug", pii_safe=True)
        assert cfg.redaction_profile == "prod"
        assert cfg.redact_traceback is True
        assert cfg.redact_exception_strings is True

    def test_config_overrides(self):
        cfg = _build_production_config(".llmdebug", pii_safe=False, max_str=200, max_items=10)
        assert cfg.max_str == 200
        assert cfg.max_items == 10
        # Production defaults should still apply
        assert cfg.source_mode == "crash_only"
        assert cfg.frames == 10

    def test_selective_hooks(self):
        """Installing only 'excepthook' should leave others untouched."""
        original_threading = threading.excepthook
        original_unraisable = sys.unraisablehook

        install_hooks(hooks=("excepthook",))

        assert sys.excepthook is not sys.__excepthook__
        assert threading.excepthook is original_threading
        assert sys.unraisablehook is original_unraisable

    def test_invalid_hook_name_raises(self):
        """Passing an invalid hook name should raise ValueError."""
        with pytest.raises(ValueError, match="Invalid hook name"):
            install_hooks(hooks=cast(tuple[HookName, ...], ("excepthook", "bogus")))

    def test_pii_safe_with_redact_override_raises(self):
        """Cannot override redact when pii_safe=True."""
        with pytest.raises(ValueError, match="Cannot override 'redact'"):
            install_hooks(pii_safe=True, redact=(r"custom",))

    def test_invalid_redaction_pattern_raises(self):
        """Bad regex in redact should raise ValueError at install time."""
        with pytest.raises(ValueError, match="Invalid redaction pattern"):
            install_hooks(pii_safe=False, redact=(r"[unclosed",))

    def test_threading_hook_with_no_thread(self, tmp_path: Path):
        """Thread=None in excepthook args should use '?' as thread name."""
        original = threading.excepthook
        threading.excepthook = lambda args: None
        try:
            install_hooks(out_dir=str(tmp_path), pii_safe=False, hooks=("threading",))
            exc, tb = _make_exc_with_tb()

            args = SimpleNamespace(
                exc_type=type(exc),
                exc_value=exc,
                exc_traceback=tb,
                thread=None,  # No thread info
            )
            args = cast(Any, args)
            threading.excepthook(args)
        finally:
            uninstall_hooks()
            threading.excepthook = original

        snapshot = read_latest_snapshot(tmp_path)
        assert snapshot["name"] == "thread_crash"
        assert snapshot["thread"] == "?"

    def test_keyboard_interrupt_not_captured(self, tmp_path: Path):
        """KeyboardInterrupt should pass through without capture."""
        # Install with a no-op original so our hook doesn't print
        original = sys.excepthook
        sys.excepthook = lambda *a: None
        try:
            install_hooks(out_dir=str(tmp_path), pii_safe=False, hooks=("excepthook",))

            try:
                raise KeyboardInterrupt()
            except KeyboardInterrupt as exc:
                tb = exc.__traceback__
                sys.excepthook(KeyboardInterrupt, exc, tb)
        finally:
            uninstall_hooks()
            sys.excepthook = original

        # No snapshot should be written for KeyboardInterrupt
        latest = tmp_path / "latest.json"
        assert not latest.exists()

    def test_asyncio_hook_warns_when_no_running_loop(self):
        with pytest.warns(RuntimeWarning, match="asyncio hook skipped"):
            install_hooks(hooks=("asyncio",))

    def test_uninstall_handles_closed_asyncio_loop(self, monkeypatch: pytest.MonkeyPatch):
        from llmdebug import hooks as hooks_module

        class _Loop:
            def set_exception_handler(self, _handler: Any) -> None:
                raise RuntimeError("loop closed")

        monkeypatch.setattr(hooks_module, "_installed", True)
        hooks_module._originals["asyncio"] = (_Loop(), None)

        uninstall_hooks()
        assert hooks_module._installed is False


# ---------------------------------------------------------------------------
# PII patterns
# ---------------------------------------------------------------------------


class TestPIIPatterns:
    def _matches(self, text: str) -> bool:
        return any(re.search(p, text) for p in PII_PATTERNS)

    def test_matches_password(self):
        assert self._matches("password=secret123")
        assert self._matches("PASSWORD: hunter2")

    def test_matches_api_key(self):
        assert self._matches("api_key=abcdef123456")
        assert self._matches("secret-key=xyz")
        assert self._matches("auth_token=tok123")

    def test_matches_bearer_token(self):
        assert self._matches("Bearer eyJhbGciOiJIUzI1NiJ9")

    def test_matches_email(self):
        assert self._matches("user@example.com")
        assert self._matches("contact user.name@domain.org for help")

    def test_matches_long_api_keys(self):
        assert self._matches("sk-abcdefghijklmnopqrstuvwxyz")
        assert self._matches("pk_live_abcdefghijklmnopqrst")
        assert self._matches("token-ABCDEFGHIJKLMNOPQRSTUVWXYZ")

    def test_email_regex_no_pipe_false_positive(self):
        """Pipe '|' should not match in email domains (old bug: [A-Z|a-z])."""
        assert not self._matches("foo|bar")
        assert not self._matches("some|value|here")

    def test_no_false_positives(self):
        assert not self._matches("hello world")
        assert not self._matches("x = 42")
        assert not self._matches("ValueError: invalid input")
        assert not self._matches("import os")
