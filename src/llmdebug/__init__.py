"""Structured debug snapshots for LLM-assisted debugging."""

from __future__ import annotations

import contextlib
import functools
import sys
import warnings
from collections.abc import Callable, Iterable
from dataclasses import fields
from re import Pattern
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .middleware import LLMDebugASGIMiddleware as LLMDebugASGIMiddleware
    from .middleware import LLMDebugWSGIMiddleware as LLMDebugWSGIMiddleware

from .capture import capture_exception
from .config import DetailLevel, GitDirtyMode, LocalsMode, OutputFormat, SnapshotConfig, SourceMode
from .detail_filter import filter_snapshot
from .hooks import PII_PATTERNS, install_hooks, uninstall_hooks
from .log_capture import install_log_handler as enable_log_capture
from .output import get_latest_snapshot
from .redaction_policy import (
    RedactionProfile,
    normalize_redaction_profile,
    resolve_redaction_policy,
)

__version__ = "3.1.0"
_unredacted_default_warning_emitted = False
_UNREDACTED_DEFAULT_WARNING = (
    "llmdebug: no redaction settings were provided for snapshot capture. "
    "For safer defaults, pass redaction_profile='ci' (dev/CI) or "
    "redaction_profile='prod' (production)."
)
_LEGACY_CAPTURE_KWARGS_DEPRECATION = (
    "llmdebug: passing capture settings as flat keyword arguments to {api}() is deprecated. "
    "Pass config=SnapshotConfig(...) instead."
)
_legacy_capture_kwargs_warning_emitted: set[str] = set()
_SNAPSHOT_CONFIG_FIELD_NAMES = frozenset(field.name for field in fields(SnapshotConfig))
__all__ = [
    "PII_PATTERNS",
    "DetailLevel",
    "GitDirtyMode",
    "LLMDebugASGIMiddleware",
    "LLMDebugWSGIMiddleware",
    "LocalsMode",
    "OutputFormat",
    "RedactionProfile",
    "SnapshotConfig",
    "SourceMode",
    "debug_snapshot",
    "enable_log_capture",
    "filter_snapshot",
    "get_latest_snapshot",
    "install_hooks",
    "load_ipython_extension",
    "load_jupyter",
    "resolve_redaction_policy",
    "snapshot_section",
    "uninstall_hooks",
    "unload_ipython_extension",
]


def _maybe_warn_about_unredacted_defaults(
    *,
    redaction_profile: RedactionProfile | str | None,
    redact: Iterable[str | Pattern[str]] | None,
    redact_keys: bool | None,
    redact_traceback: bool | None,
    redact_exception_strings: bool | None,
) -> None:
    global _unredacted_default_warning_emitted
    if _unredacted_default_warning_emitted:
        return
    has_no_redaction_inputs = (
        redaction_profile is None
        and redact is None
        and redact_keys is None
        and redact_traceback is None
        and redact_exception_strings is None
    )
    if not has_no_redaction_inputs:
        return
    warnings.warn(_UNREDACTED_DEFAULT_WARNING, UserWarning, stacklevel=3)
    _unredacted_default_warning_emitted = True


def _warn_once_about_legacy_capture_kwargs(*, api: str, stacklevel: int) -> None:
    if api in _legacy_capture_kwargs_warning_emitted:
        return
    warnings.warn(
        _LEGACY_CAPTURE_KWARGS_DEPRECATION.format(api=api),
        DeprecationWarning,
        stacklevel=stacklevel,
    )
    _legacy_capture_kwargs_warning_emitted.add(api)


def _resolve_snapshot_config(
    *,
    api: str,
    config: object | None,
    legacy_kwargs: dict[str, Any],
    stacklevel: int,
) -> SnapshotConfig:
    """Resolve capture config from config-first or legacy kwargs."""
    if config is not None and legacy_kwargs:
        legacy_keys = ", ".join(sorted(legacy_kwargs))
        raise TypeError(
            f"{api}() received both config=... and legacy capture kwargs: {legacy_keys}"
        )
    if config is not None:
        if not isinstance(config, SnapshotConfig):
            raise TypeError(f"{api}() config must be a SnapshotConfig instance")
        _maybe_warn_about_unredacted_defaults(
            redaction_profile=config.redaction_profile,
            redact=config.redact,
            redact_keys=config.redact_keys,
            redact_traceback=config.redact_traceback,
            redact_exception_strings=config.redact_exception_strings,
        )
        return config

    unknown_keys = sorted(set(legacy_kwargs) - _SNAPSHOT_CONFIG_FIELD_NAMES)
    if unknown_keys:
        if len(unknown_keys) == 1:
            raise TypeError(f"{api}() got an unexpected keyword argument {unknown_keys[0]!r}")
        joined = ", ".join(repr(key) for key in unknown_keys)
        raise TypeError(f"{api}() got unexpected keyword arguments: {joined}")

    if legacy_kwargs:
        _warn_once_about_legacy_capture_kwargs(api=api, stacklevel=stacklevel)

    redaction_profile = legacy_kwargs.get("redaction_profile")
    redact = legacy_kwargs.get("redact")
    redact_keys = legacy_kwargs.get("redact_keys")
    redact_traceback = legacy_kwargs.get("redact_traceback")
    redact_exception_strings = legacy_kwargs.get("redact_exception_strings")
    _maybe_warn_about_unredacted_defaults(
        redaction_profile=redaction_profile,
        redact=redact,
        redact_keys=redact_keys,
        redact_traceback=redact_traceback,
        redact_exception_strings=redact_exception_strings,
    )
    profile_value = normalize_redaction_profile(redaction_profile)
    policy = resolve_redaction_policy(
        redaction_profile=profile_value,
        redact=tuple(redact) if redact is not None else None,
        redact_keys=redact_keys,
        redact_traceback=redact_traceback,
        redact_exception_strings=redact_exception_strings,
        legacy_redact=(),
        legacy_redact_keys=False,
        legacy_redact_traceback=False,
        legacy_redact_exception_strings=False,
    )

    normalized_kwargs = {
        key: value
        for key, value in legacy_kwargs.items()
        if key
        not in {
            "redaction_profile",
            "redact",
            "redact_keys",
            "redact_traceback",
            "redact_exception_strings",
            "include_modules",
        }
    }
    include_modules = legacy_kwargs.get("include_modules")
    if include_modules is not None:
        normalized_kwargs["include_modules"] = tuple(include_modules)

    return SnapshotConfig(
        **normalized_kwargs,
        redaction_profile=policy.profile,
        redact=policy.patterns,
        redact_keys=policy.redact_keys,
        redact_traceback=policy.redact_traceback,
        redact_exception_strings=policy.redact_exception_strings,
    )


def _try_capture(name: str, exc: Exception, cfg: SnapshotConfig) -> None:
    """Attempt to capture an exception snapshot, swallowing capture errors."""
    _, _, tb = sys.exc_info()
    try:
        capture_exception(name, exc, tb, cfg)
    except Exception as capture_exc:
        # Always log — a broken safety net is worse than a noisy one
        sys.stderr.write(
            f"llmdebug: capture failed for {name}: {type(capture_exc).__name__}: {capture_exc}\n"
        )


def debug_snapshot(
    *,
    name: str | None = None,
    config: SnapshotConfig | None = None,
    **legacy_kwargs: Any,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorator that captures debug snapshots on exception.

    Usage:
        cfg = SnapshotConfig(out_dir=".llmdebug", redaction_profile="ci")
        @debug_snapshot(config=cfg)
        def main():
            ...

    Args:
        name: Snapshot name. Defaults to the decorated function name.
        config: Preferred capture settings object.
        legacy_kwargs: Deprecated flat capture kwargs matching
            :class:`SnapshotConfig` fields. Use ``config=SnapshotConfig(...)``
            instead.

    Note:
        include_coverage: Coverage toggle for pytest-plugin enrichment when pytest-cov is active.
            Decorator/context-manager captures do not currently attach coverage data.
    """
    cfg = _resolve_snapshot_config(
        api="debug_snapshot",
        config=config,
        legacy_kwargs=legacy_kwargs,
        stacklevel=3,
    )

    def decorator(fn: Callable[..., Any]) -> Callable[..., Any]:
        snap_name = name or fn.__name__

        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                _try_capture(snap_name, e, cfg)
                raise

        return wrapper

    return decorator


@contextlib.contextmanager
def snapshot_section(
    section_name: str,
    *,
    config: SnapshotConfig | None = None,
    **legacy_kwargs: Any,
):
    """Context manager that captures debug snapshots on exception.

    Usage:
        cfg = SnapshotConfig(out_dir=".llmdebug", redaction_profile="prod")
        with snapshot_section("data_loading", config=cfg):
            ...

    Note:
        ``legacy_kwargs`` temporarily accepts flat :class:`SnapshotConfig`
        fields for backward compatibility. Prefer ``config=SnapshotConfig(...)``.
        ``include_coverage`` is reserved for pytest-plugin enrichment when
        pytest-cov is active. Context-manager captures do not currently attach
        coverage data.
    """
    cfg = _resolve_snapshot_config(
        api="snapshot_section",
        config=config,
        legacy_kwargs=legacy_kwargs,
        stacklevel=3,
    )
    try:
        yield
    except Exception as e:
        _try_capture(section_name, e, cfg)
        raise


def load_jupyter(**kwargs: Any) -> None:
    """Install llmdebug into the current IPython/Jupyter session.

    Convenience wrapper around :func:`jupyter_plugin.install`.
    Requires the ``jupyter`` extra: ``pip install 'llmdebug[jupyter]'``
    """
    try:
        from .jupyter_plugin import install
    except ImportError:
        raise RuntimeError(
            "IPython is not installed. Install with: pip install 'llmdebug[jupyter]'"
        ) from None
    install(**kwargs)


def load_ipython_extension(ipython: Any) -> None:
    """IPython extension entry point for ``%load_ext llmdebug``."""
    from .jupyter_plugin import load_ipython_extension as _load

    _load(ipython)


def unload_ipython_extension(ipython: Any) -> None:
    """IPython extension entry point for ``%unload_ext llmdebug``."""
    from .jupyter_plugin import unload_ipython_extension as _unload

    _unload(ipython)


def __getattr__(name: str) -> Any:
    if name in ("LLMDebugASGIMiddleware", "LLMDebugWSGIMiddleware"):
        from .middleware import LLMDebugASGIMiddleware, LLMDebugWSGIMiddleware

        globals()["LLMDebugASGIMiddleware"] = LLMDebugASGIMiddleware
        globals()["LLMDebugWSGIMiddleware"] = LLMDebugWSGIMiddleware
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
