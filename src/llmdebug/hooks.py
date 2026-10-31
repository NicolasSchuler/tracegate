"""Production exception hooks for automatic crash capture.

Installs hooks into Python's exception handling machinery so that unhandled
exceptions, thread crashes, unraisable exceptions, and asyncio failures
automatically generate debug snapshots.

Usage:
    import llmdebug
    llmdebug.install_hooks(out_dir=".llmdebug")

    # ... application code ...
    # Any unhandled exception will produce a snapshot.

    llmdebug.uninstall_hooks()

Pattern: follows log_capture.py — module-level state, threading.Lock, idempotent
install/uninstall.
"""

from __future__ import annotations

import re
import sys
import threading
import time
import traceback as tb_mod
import warnings
from collections.abc import Callable
from typing import Any, Literal

from .capture import capture_exception
from .config import SnapshotConfig
from .redaction_policy import (
    PII_PATTERNS,
    RedactionProfile,
    normalize_redaction_profile,
    resolve_redaction_policy,
)

# Validate built-in patterns at import time (fail fast, not at first crash)
for _p in PII_PATTERNS:
    re.compile(_p)

# ---------------------------------------------------------------------------
# Types
# ---------------------------------------------------------------------------

HookName = Literal["excepthook", "threading", "unraisable", "asyncio"]
_VALID_HOOK_NAMES: frozenset[str] = frozenset(HookName.__args__)  # type: ignore[attr-defined]


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------


_CLEANUP_THRESHOLD = 1000


class _RateLimiter:
    """Thread-safe sliding-window rate limiter keyed by error signature.

    A signature is ``(exception type, file, line)``. Once a signature reaches
    *max_captures* within *window_seconds*, further captures are refused. The first
    refusal writes a notice to stderr; the notice re-arms once the signature's
    window has fully drained.

    Args:
        max_captures: Captures allowed per signature within one window.
        window_seconds: Length of the sliding window in seconds.

    Raises:
        ValueError: If *max_captures* is below 1 or *window_seconds* is not positive.
    """

    def __init__(self, max_captures: int, window_seconds: float) -> None:
        if max_captures < 1:
            raise ValueError(f"max_captures must be >= 1, got {max_captures}")
        if window_seconds <= 0:
            raise ValueError(f"window_seconds must be > 0, got {window_seconds}")
        self._max = max_captures
        self._window = window_seconds
        self._lock = threading.Lock()
        self._buckets: dict[tuple[str, str, int], list[float]] = {}
        self._suppressed: set[tuple[str, str, int]] = set()

    def _cleanup_expired(self, now: float) -> None:
        """Remove buckets whose timestamps have all expired.

        The caller must hold ``_lock``.
        """
        expired = [k for k, ts in self._buckets.items() if all(now - t >= self._window for t in ts)]
        for k in expired:
            del self._buckets[k]
            self._suppressed.discard(k)

    def allow(self, exc_type: str, file: str, line: int) -> bool:
        """Record a capture for this error signature if it is under the rate limit.

        Args:
            exc_type: Exception type name.
            file: File of the innermost traceback frame.
            line: Line number of the innermost traceback frame.

        Returns:
            True if the capture is allowed (and counted against the window); False if
            the signature has reached its limit.
        """
        key = (exc_type, file, line)
        now = time.monotonic()
        with self._lock:
            if len(self._buckets) > _CLEANUP_THRESHOLD:
                self._cleanup_expired(now)
            timestamps = self._buckets.get(key, [])
            # Prune expired entries
            timestamps = [t for t in timestamps if now - t < self._window]
            if not timestamps:
                # Window fully expired — clean up stale state
                self._buckets.pop(key, None)
                self._suppressed.discard(key)
            if len(timestamps) >= self._max:
                self._buckets[key] = timestamps
                if key not in self._suppressed:
                    self._suppressed.add(key)
                    sys.stderr.write(
                        f"llmdebug: rate limit reached for {exc_type} "
                        f"at {file}:{line} ({self._max}/{self._window}s) "
                        f"— further captures suppressed\n"
                    )
                return False
            timestamps.append(now)
            self._buckets[key] = timestamps
            return True


# ---------------------------------------------------------------------------
# Safe capture helper
# ---------------------------------------------------------------------------


def _capture_safely(
    name: str,
    exc: BaseException,
    tb: Any,
    cfg: SnapshotConfig,
    limiter: _RateLimiter,
    *,
    extra: dict[str, Any] | None = None,
) -> None:
    """Capture *exc* into a snapshot, swallowing any error; never raises.

    The innermost traceback frame provides the rate-limit signature, so captures
    beyond the limit are skipped. Capture failures are always written to stderr.

    Args:
        name: Snapshot name.
        exc: The exception to capture.
        tb: The exception's traceback.
        cfg: Snapshot configuration.
        limiter: Rate limiter shared by the installing hook or middleware.
        extra: Additional context stored with the snapshot, such as hook metadata.
    """
    try:
        # Extract signature from last traceback frame for rate limiting
        frames = tb_mod.extract_tb(tb)
        if frames:
            last = frames[-1]
            if not limiter.allow(type(exc).__name__, last.filename, last.lineno or 0):
                return
        capture_exception(name, exc, tb, cfg, extra=extra)
    except Exception as capture_exc:
        # Always log — a broken safety net is worse than a noisy one
        sys.stderr.write(
            f"llmdebug: capture failed for {name}: {type(capture_exc).__name__}: {capture_exc}\n"
        )


# ---------------------------------------------------------------------------
# Production-safe config builder
# ---------------------------------------------------------------------------


def _build_production_config(
    out_dir: str,
    pii_safe: bool,
    *,
    redaction_profile: RedactionProfile | str | None = None,
    include_git: bool = False,
    include_array_stats: bool = False,
    lock_timeout: float = 1.0,
    debug: bool = False,
    **config_overrides: Any,
) -> SnapshotConfig:
    """Build a SnapshotConfig with production-safe defaults.

    Keeps at most 10 frames and source for the crash frame only. When *pii_safe* is
    set and no profile is given, the ``"prod"`` redaction profile is used.

    Args:
        out_dir: Snapshot output directory.
        pii_safe: Default to the ``"prod"`` profile and forbid a ``redact`` override.
        redaction_profile: Named redaction profile; may instead be passed in
            *config_overrides*, but not both.
        include_git: Include git context.
        include_array_stats: Include array statistics in local summaries.
        lock_timeout: File-lock timeout in seconds when writing snapshots.
        debug: Report capture warnings through the diagnostics logger.
        **config_overrides: Additional :class:`SnapshotConfig` fields; the
            ``redact*`` fields override the profile's settings.

    Returns:
        The resolved snapshot configuration.

    Raises:
        ValueError: If the redaction profile is given twice, or ``redact`` is
            overridden while *pii_safe* is True.
    """
    profile_override = config_overrides.pop("redaction_profile", None)
    if redaction_profile is not None and profile_override is not None:
        raise ValueError(
            "Specify redaction_profile either as a named argument or in config_overrides, not both."
        )
    profile_source = redaction_profile if redaction_profile is not None else profile_override
    if pii_safe and profile_source is None:
        # Production capture surfaces should default to strict redaction.
        profile_source = "prod"
    if pii_safe and "redact" in config_overrides:
        raise ValueError(
            "Cannot override 'redact' when pii_safe=True. "
            "Set pii_safe=False to provide custom redaction patterns."
        )
    profile = normalize_redaction_profile(profile_source)
    explicit_redact = config_overrides.pop("redact", None)
    explicit_redact_keys = config_overrides.pop("redact_keys", None)
    explicit_redact_traceback = config_overrides.pop("redact_traceback", None)
    explicit_redact_exception_strings = config_overrides.pop(
        "redact_exception_strings",
        None,
    )
    policy = resolve_redaction_policy(
        redaction_profile=profile,
        redact=tuple(explicit_redact) if explicit_redact is not None else None,
        redact_keys=explicit_redact_keys,
        redact_traceback=explicit_redact_traceback,
        redact_exception_strings=explicit_redact_exception_strings,
        legacy_redact=PII_PATTERNS if pii_safe else (),
        legacy_redact_keys=False,
        legacy_redact_traceback=False,
        legacy_redact_exception_strings=False,
    )
    return SnapshotConfig(
        out_dir=out_dir,
        redaction_profile=policy.profile,
        include_git=include_git,
        include_array_stats=include_array_stats,
        lock_timeout=lock_timeout,
        debug=debug,
        redact=policy.patterns,
        redact_keys=policy.redact_keys,
        redact_traceback=policy.redact_traceback,
        redact_exception_strings=policy.redact_exception_strings,
        source_mode="crash_only",
        frames=10,
        **config_overrides,
    )


# ---------------------------------------------------------------------------
# Hook wrapper factories
# ---------------------------------------------------------------------------


def _is_intentional_exit(exc: BaseException) -> bool:
    """Return whether *exc* is an intentional exit, without probing the instance.

    ``isinstance`` reads ``exc.__class__``, so an object whose ``__getattribute__``
    raises took the whole hook down before it could delegate to the original handler
    and report the failure. ``type()`` cannot be intercepted.
    """
    return issubclass(type(exc), (KeyboardInterrupt, SystemExit))


def _make_excepthook(
    original: Callable[..., Any],
    cfg: SnapshotConfig,
    limiter: _RateLimiter,
) -> Callable[..., Any]:
    """Create a ``sys.excepthook`` replacement that captures, then delegates."""

    def hook(
        exc_type: type[BaseException],
        exc_value: BaseException,
        exc_tb: Any,
    ) -> None:
        """Capture an unhandled exception, then call the original excepthook."""
        # Don't capture intentional exits
        if not _is_intentional_exit(exc_value):
            _capture_safely(
                "unhandled",
                exc_value,
                exc_tb,
                cfg,
                limiter,
                extra={"hook": "sys.excepthook"},
            )
        original(exc_type, exc_value, exc_tb)

    return hook


def _make_threading_hook(
    original: Callable[..., Any] | None,
    cfg: SnapshotConfig,
    limiter: _RateLimiter,
) -> Callable[..., Any]:
    """Create a ``threading.excepthook`` replacement that captures, then delegates."""

    def hook(args: Any) -> None:
        """Capture a thread crash, then call the original hook if there was one."""
        exc_value = args.exc_value
        exc_tb = args.exc_traceback
        if exc_value is not None and exc_tb is not None and not _is_intentional_exit(exc_value):
            thread_name = args.thread.name if args.thread else "?"
            _capture_safely(
                "thread_crash",
                exc_value,
                exc_tb,
                cfg,
                limiter,
                extra={"hook": "threading.excepthook", "thread": thread_name},
            )
        if original is not None:
            original(args)

    return hook


def _make_unraisable_hook(
    original: Callable[..., Any],
    cfg: SnapshotConfig,
    limiter: _RateLimiter,
) -> Callable[..., Any]:
    """Create a ``sys.unraisablehook`` replacement that captures, then delegates."""

    def hook(unraisable: Any) -> None:
        """Capture an unraisable exception, then call the original hook."""
        exc = unraisable.exc_value
        tb = unraisable.exc_traceback
        if exc is not None and tb is not None and not _is_intentional_exit(exc):
            try:
                obj = unraisable.object
                obj_repr = repr(obj) if obj is not None else None
            except Exception:
                obj_repr = f"<repr failed: {type(unraisable.object).__name__}>"
            _capture_safely(
                "unraisable",
                exc,
                tb,
                cfg,
                limiter,
                extra={"hook": "sys.unraisablehook", "object": obj_repr},
            )
        original(unraisable)

    return hook


def _make_asyncio_handler(
    original: Callable[..., Any] | None,
    cfg: SnapshotConfig,
    limiter: _RateLimiter,
) -> Callable[..., Any]:
    """Create an asyncio exception handler that captures, then delegates."""

    def handler(loop: Any, context: dict[str, Any]) -> None:
        """Capture the context's exception, then call the original or default handler."""
        exc = context.get("exception")
        if exc is not None:
            tb = exc.__traceback__
            if tb is not None and not _is_intentional_exit(exc):
                _capture_safely(
                    "asyncio",
                    exc,
                    tb,
                    cfg,
                    limiter,
                    extra={
                        "hook": "asyncio",
                        "message": context.get("message", ""),
                    },
                )
        if original is not None:
            original(loop, context)
        else:
            loop.default_exception_handler(context)

    return handler


# ---------------------------------------------------------------------------
# Module-level state (follows log_capture.py pattern)
# ---------------------------------------------------------------------------

_installed = False
_lock = threading.Lock()
_originals: dict[str, Any] = {}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def install_hooks(
    *,
    out_dir: str = ".llmdebug",
    max_captures_per_minute: int = 10,
    pii_safe: bool = True,
    redaction_profile: RedactionProfile | str | None = None,
    include_git: bool = False,
    include_array_stats: bool = False,
    lock_timeout: float = 1.0,
    debug: bool = False,
    hooks: tuple[HookName, ...] = ("excepthook", "threading", "unraisable"),
    **config_overrides: Any,
) -> None:
    """Install exception hooks for automatic production crash capture.

    Patches up to four Python exception-handling entry points so that
    unhandled exceptions automatically produce debug snapshots:

    - **sys.excepthook** — catches unhandled exceptions on the main thread.
    - **threading.excepthook** — catches crashes in spawned threads.
    - **sys.unraisablehook** — catches exceptions that cannot propagate
      (e.g. ``__del__`` failures, GC-related errors).
    - **asyncio exception handler** — catches unhandled exceptions in
      asyncio tasks and callbacks (requires a running event loop at
      install time).

    Each hook delegates to the original handler after capturing, so
    existing behaviour (e.g. printing the traceback) is preserved.
    ``KeyboardInterrupt`` and ``SystemExit`` are never captured.

    Idempotent — calling multiple times is a no-op after the first
    successful install.  Thread-safe.

    Args:
        out_dir: Snapshot output directory.
        max_captures_per_minute: Rate limit per unique error signature
            (type + file + line).  A sliding 60-second window is used.
        pii_safe: When ``True`` (the default), use the ``"prod"``
            redaction profile (built-in PII patterns plus traceback and
            exception-string redaction) unless *redaction_profile* names
            another, and reject a custom ``redact`` override.
        redaction_profile: Named redaction profile — ``"dev"``,
            ``"ci"``, or ``"prod"``.  Overrides the profile that
            *pii_safe* would select.
        include_git: Include git context in snapshots (disabled by
            default because it shells out to ``git``).
        include_array_stats: Include NumPy/Tensor array statistics in
            local-variable summaries.
        lock_timeout: File-lock timeout in seconds when writing snapshot
            files (keep short for production to avoid blocking).
        debug: When ``True``, report capture warnings through the
            ``llmdebug.diagnostics`` logger (stderr by default). Capture
            failures are written to stderr regardless.
        hooks: Which hooks to install.  Valid names: ``"excepthook"``,
            ``"threading"``, ``"unraisable"``, ``"asyncio"``.
            Default: all except ``"asyncio"`` (asyncio requires a
            running event loop at install time; a ``RuntimeWarning`` is
            emitted if requested without one).
        **config_overrides: Additional :class:`SnapshotConfig` fields
            forwarded to the underlying config builder.

    Raises:
        ValueError: If *hooks* contains an invalid name, if *redact*
            is passed while *pii_safe* is ``True``, if *redaction_profile*
            is not a known profile, if a custom redaction pattern is not a
            valid regular expression, or if *max_captures_per_minute* is
            below 1.

    Example::

        import llmdebug

        llmdebug.install_hooks(out_dir=".llmdebug")
        # Application code runs normally.
        # Any unhandled exception now writes a snapshot to .llmdebug/

        llmdebug.uninstall_hooks()  # Restore original hooks.

    See Also:
        :func:`uninstall_hooks`: Restore all original hooks.
    """
    global _installed

    with _lock:
        if _installed:
            return

        # Validate hook names early
        invalid = set[str](hooks) - _VALID_HOOK_NAMES
        if invalid:
            raise ValueError(
                f"Invalid hook name(s): {invalid}. Valid options: {sorted(_VALID_HOOK_NAMES)}"
            )

        cfg = _build_production_config(
            out_dir,
            pii_safe,
            redaction_profile=redaction_profile,
            include_git=include_git,
            include_array_stats=include_array_stats,
            lock_timeout=lock_timeout,
            debug=debug,
            **config_overrides,
        )

        # Validate redaction patterns at install time (fail fast)
        for pattern in cfg.redact or ():
            if isinstance(pattern, str):
                try:
                    re.compile(pattern)
                except re.error as e:
                    raise ValueError(f"Invalid redaction pattern {pattern!r}: {e}") from e

        limiter = _RateLimiter(max_captures_per_minute, 60.0)

        if "excepthook" in hooks:
            _originals["excepthook"] = sys.excepthook
            sys.excepthook = _make_excepthook(sys.excepthook, cfg, limiter)

        if "threading" in hooks:
            _originals["threading"] = threading.excepthook
            threading.excepthook = _make_threading_hook(threading.excepthook, cfg, limiter)

        if "unraisable" in hooks:
            _originals["unraisable"] = sys.unraisablehook
            sys.unraisablehook = _make_unraisable_hook(sys.unraisablehook, cfg, limiter)

        if "asyncio" in hooks:
            try:
                import asyncio

                loop = asyncio.get_running_loop()
            except RuntimeError:
                # No running event loop — always warn since user explicitly requested it
                warnings.warn(
                    "llmdebug: asyncio hook skipped — no running event loop",
                    RuntimeWarning,
                    stacklevel=2,
                )
            else:
                original_handler = loop.get_exception_handler()
                _originals["asyncio"] = (loop, original_handler)
                loop.set_exception_handler(_make_asyncio_handler(original_handler, cfg, limiter))

        _installed = True


def uninstall_hooks() -> None:
    """Remove all installed hooks, restoring original Python handlers.

    Reverses every hook installed by :func:`install_hooks`, restoring
    ``sys.excepthook``, ``threading.excepthook``, ``sys.unraisablehook``,
    and (if applicable) the asyncio exception handler to their pre-install
    values.

    Idempotent — safe to call multiple times or before any install.
    Thread-safe.

    See Also:
        :func:`install_hooks`: Install the exception hooks.
    """
    global _installed

    with _lock:
        if not _installed:
            return

        if "excepthook" in _originals:
            sys.excepthook = _originals["excepthook"]

        if "threading" in _originals:
            threading.excepthook = _originals["threading"]

        if "unraisable" in _originals:
            sys.unraisablehook = _originals["unraisable"]

        if "asyncio" in _originals:
            loop, original_handler = _originals["asyncio"]
            try:
                loop.set_exception_handler(original_handler)
            except RuntimeError:
                pass  # Loop closed or shutting down — safe to skip restore

        _originals.clear()
        _installed = False
