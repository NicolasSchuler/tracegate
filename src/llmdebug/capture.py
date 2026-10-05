"""Core exception capture logic."""

from __future__ import annotations

import datetime as dt
import inspect as _inspect_module
import os
import platform
import sys
import traceback
import types
from datetime import timezone
from pathlib import Path
from re import Pattern
from typing import Any, cast

from ._debug_session import (
    DEBUG_SESSION_SCHEMA_VERSION,
    build_debug_session,
    split_extra_context,
)
from .config import SnapshotConfig, resolve_snapshot_config
from .output import write_bundle
from .serialize import (
    compile_redactors,
    locals_metadata,
    redact_jsonlike,
    redact_text,
    serialize_locals_bundle,
    truncate_str,
)

SCHEMA_VERSION = DEBUG_SESSION_SCHEMA_VERSION


def _warn(warnings: list[str], message: str) -> None:
    """Append a capture warning to the current capture-local buffer."""
    warnings.append(message)


def _safe_str(value: object) -> str:
    """``str(value)`` that never raises, like the stdlib ``traceback`` module."""
    try:
        return str(value)
    except Exception as e:
        return f"<str() failed: {type(e).__name__}>"


try:  # Python 3.11+
    BaseExceptionGroup  # type: ignore[name-defined]  # noqa: B018
except NameError:  # pragma: no cover - Python 3.10 fallback
    BaseExceptionGroup = None  # type: ignore[assignment]


def _serialize_config(cfg: SnapshotConfig) -> dict[str, Any]:
    """Build a JSON-serialisable dict representation of the active capture config."""
    import dataclasses

    result = {
        key: value for key, value in dataclasses.asdict(cfg).items() if not key.startswith("_")
    }
    # Pattern objects aren't JSON-serializable — convert to pattern strings
    result["redact"] = [p if isinstance(p, str) else p.pattern for p in (cfg.redact or ())]
    # Convert tuple to list for JSON
    if cfg.include_modules:
        result["include_modules"] = list(cfg.include_modules)
    return result


def _summarize_exception(
    exc: BaseException,
    cfg: SnapshotConfig,
    *,
    warnings: list[str],
    redactors: list[Pattern[str]],
    redact_strings: bool,
    depth: int = 0,
    max_depth: int = 2,
    seen: set[int] | None = None,
) -> dict[str, Any]:
    def _clean(value: str) -> str:
        text = truncate_str(value, cfg.max_str)
        if redact_strings and redactors:
            return redact_text(text, redactors)
        return text

    if seen is None:
        seen = set()
    if id(exc) in seen:
        exc_type = type(exc)
        return {
            "type": exc_type.__name__,
            "qualified_type": f"{exc_type.__module__}.{exc_type.__name__}",
            "message": "...[CYCLE]",
        }
    seen.add(id(exc))

    exc_type = type(exc)
    summary: dict[str, Any] = {
        "type": exc_type.__name__,
        "qualified_type": f"{exc_type.__module__}.{exc_type.__name__}",
        "message": _clean(_safe_str(exc)),
    }
    try:
        if exc.args:
            summary["args"] = [_clean(_safe_str(a)) for a in list(exc.args)[: cfg.max_items]]
    except Exception as e:
        summary["args"] = [f"<serialization failed: {type(e).__name__}>"]
    try:
        notes = getattr(exc, "__notes__", None)
        if notes:
            summary["notes"] = [_clean(_safe_str(n)) for n in list(notes)[: cfg.max_items]]
    except Exception as e:
        summary["notes"] = [f"<serialization failed: {type(e).__name__}>"]

    if depth >= max_depth:
        return summary

    if BaseExceptionGroup is not None and isinstance(exc, BaseExceptionGroup):
        exceptions = cast(
            list[BaseException],
            exc.exceptions,  # pyright: ignore[reportAttributeAccessIssue, reportUnknownMemberType]
        )
        limit = min(len(exceptions), cfg.max_items)
        summary["is_exception_group"] = True
        summary["exceptions"] = [
            _summarize_exception(
                e,
                cfg,
                warnings=warnings,
                redactors=redactors,
                redact_strings=redact_strings,
                depth=depth + 1,
                max_depth=max_depth,
                seen=seen,
            )
            for e in exceptions[:limit]
        ]
        if len(exceptions) > limit:
            summary["exceptions_truncated"] = True

    cause = getattr(exc, "__cause__", None)
    if cause is not None:
        summary["cause"] = _summarize_exception(
            cause,
            cfg,
            warnings=warnings,
            redactors=redactors,
            redact_strings=redact_strings,
            depth=depth + 1,
            max_depth=max_depth,
            seen=seen,
        )

    context = getattr(exc, "__context__", None)
    if context is not None:
        summary["context"] = _summarize_exception(
            context,
            cfg,
            warnings=warnings,
            redactors=redactors,
            redact_strings=redact_strings,
            depth=depth + 1,
            max_depth=max_depth,
            seen=seen,
        )

    summary["suppress_context"] = bool(getattr(exc, "__suppress_context__", False))

    # Error categorization (only at top level)
    if depth == 0 and cfg.categorize_errors:
        try:
            from .error_categories import categorize_exception

            category = categorize_exception(exc)
            if category:
                summary["error_category"] = category
        except Exception as e:
            _warn(warnings, f"error categorization failed: {type(e).__name__}: {e}")

    return summary


def get_env_info(cfg: SnapshotConfig) -> dict[str, Any]:
    """Collect environment information."""
    argv = []
    try:
        argv = [truncate_str(str(a), cfg.max_str) for a in sys.argv]
    except Exception:
        argv = []
    return {
        "python": sys.version.replace("\n", " "),
        "executable": sys.executable,
        "platform": platform.platform(),
        "cwd": os.getcwd(),
        "argv": argv,
    }


def get_source_snippet(filename: str, lineno: int, ctx: int) -> dict[str, Any]:
    """Extract source code around a line."""
    try:
        if not filename or not os.path.exists(filename):
            return {}
        start_line = max(lineno - ctx, 1)
        end_line = max(lineno + ctx, start_line)
        snippet: list[dict[str, Any]] = []
        last_line = 0
        has_more_lines = False

        # Stream file content and capture only the requested window.
        with open(filename, encoding="utf-8", errors="replace") as f:
            for line_no, line in enumerate(f, start=1):
                last_line = line_no
                if line_no < start_line:
                    continue
                if line_no > end_line:
                    has_more_lines = True
                    break
                snippet.append({"lineno": line_no, "code": line.rstrip("\n")})

        if last_line == 0:
            return {}

        actual_end = end_line if has_more_lines else min(end_line, last_line)
        return {"start": start_line, "end": actual_end, "snippet": snippet}
    except Exception:
        return {}


def extract_function_args(frame: types.FrameType) -> tuple[list[str], dict[str, Any]] | None:
    """Extract function argument names and values from a frame.

    Returns:
        Tuple of (arg_names, arg_values) where arg_names is a list of argument
        names in order, and arg_values is a dict mapping arg names to values.
        Returns None on error.

    Note: varnames layout is
    [positional-only][positional-or-keyword][keyword-only][*args][**kwargs][locals]
    """
    try:
        code = frame.f_code
        posonly_count = getattr(code, "co_posonlyargcount", 0)
        arg_count = code.co_argcount
        kwonly_count = code.co_kwonlyargcount
        varnames = code.co_varnames
        flags = code.co_flags

        arg_names: list[str] = []
        arg_values: dict[str, Any] = {}
        # CPython includes positional-only names in co_argcount. Keep a fallback
        # for runtimes that may report positional-only and positional-or-keyword
        # counts separately.
        positional_count = arg_count
        if posonly_count > arg_count:
            positional_count = posonly_count + arg_count

        # Positional-only + positional/keyword args
        for name in varnames[:positional_count]:
            arg_names.append(name)
            if name in frame.f_locals:
                arg_values[name] = frame.f_locals[name]

        # Track position after positional args
        pos = positional_count

        # Keyword-only args (come after positional args in varnames)
        for name in varnames[pos : pos + kwonly_count]:
            arg_names.append(name)
            if name in frame.f_locals:
                arg_values[name] = frame.f_locals[name]
        pos += kwonly_count

        # Check for *args (CO_VARARGS = 0x04)
        if flags & _inspect_module.CO_VARARGS:
            args_name = varnames[pos]
            arg_names.append(f"*{args_name}")
            if args_name in frame.f_locals:
                arg_values[args_name] = frame.f_locals[args_name]
            pos += 1

        # Check for **kwargs (CO_VARKEYWORDS = 0x08)
        if flags & _inspect_module.CO_VARKEYWORDS:
            kwargs_name = varnames[pos]
            arg_names.append(f"**{kwargs_name}")
            if kwargs_name in frame.f_locals:
                arg_values[kwargs_name] = frame.f_locals[kwargs_name]

        return arg_names, arg_values
    except Exception:
        return None


def _extract_tb_pairs(tb: types.TracebackType, cfg: SnapshotConfig) -> list[tuple[Any, Any]]:
    """Walk the traceback and pair tb objects with extracted frame info, applying module filter."""
    extracted = traceback.extract_tb(tb)[-cfg.frames :]
    tb_list: list[Any] = []
    cur = tb
    while cur is not None:
        tb_list.append(cur)
        cur = cur.tb_next
    tb_list = tb_list[-cfg.frames :]
    paired = list(zip(tb_list, extracted, strict=True))

    if cfg.include_modules is not None:
        filtered: list[tuple[Any, Any]] = []
        for i, (tb_item, ex_item) in enumerate(paired):
            frame = tb_item.tb_frame
            module = frame.f_globals.get("__name__", "")
            is_crash_frame = i == len(paired) - 1
            matches_filter = module.startswith(cfg.include_modules)
            if is_crash_frame or matches_filter:
                filtered.append((tb_item, ex_item))
        paired = filtered
    return paired


def _add_locals_safe(
    frame_dict: dict[str, Any],
    frame: types.FrameType,
    cfg: SnapshotConfig,
    effective_redactors: list[Pattern[str]],
) -> None:
    """Add safe-mode locals (with metadata and optional args) to frame_dict."""
    locals_bundle = serialize_locals_bundle(frame.f_locals, cfg, effective_redactors)
    frame_dict["locals"] = locals_bundle.values
    frame_dict["locals_meta"] = locals_bundle.metadata
    if locals_bundle.truncated:
        frame_dict["locals_truncated"] = True
        frame_dict["locals_truncated_keys"] = locals_bundle.truncated_keys
    if cfg.include_args:
        arg_info = extract_function_args(frame)
        if arg_info:
            arg_names, arg_values = arg_info
            args_bundle = serialize_locals_bundle(arg_values, cfg, effective_redactors)
            frame_dict["args"] = args_bundle.values
            frame_dict["arg_names"] = arg_names


def _add_locals_meta(
    frame_dict: dict[str, Any],
    frame: types.FrameType,
    cfg: SnapshotConfig,
    warnings: list[str],
) -> None:
    """Add meta-mode locals to frame_dict."""
    frame_dict["locals_meta"] = locals_metadata(frame.f_locals, cfg)
    try:
        total_locals = len(frame.f_locals)
    except Exception as e:
        _warn(warnings, f"locals count failed: {type(e).__name__}: {e}")
        total_locals = None
    if total_locals is not None and total_locals > cfg.max_items:
        frame_dict["locals_truncated"] = True
        frame_dict["locals_truncated_keys"] = total_locals - cfg.max_items
    if cfg.include_args:
        arg_info = extract_function_args(frame)
        if arg_info:
            arg_names, arg_values = arg_info
            frame_dict["args_meta"] = locals_metadata(arg_values, cfg)
            frame_dict["arg_names"] = arg_names


def _serialize_frame(
    tb_item: Any,
    ex_item: Any,
    idx: int,
    total: int,
    cfg: SnapshotConfig,
    warnings: list[str],
    effective_redactors: list[Pattern[str]],
) -> dict[str, Any]:
    """Serialize a single traceback frame to a dict."""
    frame = tb_item.tb_frame
    lineno = ex_item.lineno or 0
    file_rel = None
    try:
        cwd = os.getcwd()
        file_rel = os.path.relpath(ex_item.filename, cwd)
        if file_rel.startswith(".."):
            file_rel = None
    except Exception as e:
        _warn(warnings, f"relative path failed: {type(e).__name__}: {e}")

    frame_dict: dict[str, Any] = {
        "file": ex_item.filename,
        "file_rel": file_rel,
        "line": lineno,
        "function": ex_item.name,
        "module": frame.f_globals.get("__name__"),
        "code": ex_item.line,
    }

    if cfg.include_async_context:
        try:
            from .async_context import is_coroutine_frame

            if is_coroutine_frame(frame):
                frame_dict["is_coroutine"] = True
        except Exception as e:
            _warn(warnings, f"coroutine frame detection failed: {type(e).__name__}: {e}")

    if cfg.source_mode == "all" or (cfg.source_mode == "crash_only" and idx == total - 1):
        frame_dict["source"] = get_source_snippet(ex_item.filename, lineno, cfg.source_context)

    if cfg.locals_mode == "safe":
        _add_locals_safe(frame_dict, frame, cfg, effective_redactors)
    elif cfg.locals_mode == "meta":
        _add_locals_meta(frame_dict, frame, cfg, warnings)

    return frame_dict


def collect_frames(
    tb: types.TracebackType,
    cfg: SnapshotConfig,
    *,
    warnings: list[str],
    redactors: list[Pattern[str]] | None = None,
) -> list[dict[str, Any]]:
    """Collect stack frames from traceback."""
    effective_redactors = (
        redactors
        if redactors is not None
        else (compile_redactors(cfg.redact or ()) if cfg.locals_mode == "safe" else [])
    )
    paired = _extract_tb_pairs(tb, cfg)
    total = len(paired)
    frames_out = [
        _serialize_frame(tb_item, ex_item, i, total, cfg, warnings, effective_redactors)
        for i, (tb_item, ex_item) in enumerate(paired)
    ]
    return list(reversed(frames_out))


def _try_git_context(cfg: SnapshotConfig, warnings: list[str]) -> Any:
    try:
        from .git_context import get_git_context

        return get_git_context(dirty_mode=cfg.git_dirty_mode)
    except Exception as e:
        _warn(warnings, f"git context failed: {type(e).__name__}: {e}")
        return None


def _try_async_context(warnings: list[str]) -> Any:
    try:
        from .async_context import get_async_context

        return get_async_context()
    except Exception as e:
        _warn(warnings, f"async context failed: {type(e).__name__}: {e}")
        return None


def _try_log_context(cfg: SnapshotConfig, warnings: list[str]) -> Any:
    try:
        from .log_capture import get_recent_logs

        return get_recent_logs(cfg.log_max_records)
    except Exception as e:
        _warn(warnings, f"log capture failed: {type(e).__name__}: {e}")
        return None


def _redact_context(
    context_payload: dict[str, Any],
    redactors: list[Pattern[str]],
    cfg: SnapshotConfig,
    warnings: list[str],
) -> dict[str, Any]:
    if not context_payload or not redactors:
        return context_payload
    try:
        cleaned = redact_jsonlike(context_payload, redactors, redact_keys=bool(cfg.redact_keys))
        if isinstance(cleaned, dict):
            return cleaned
        _warn(warnings, "context redaction returned non-dict payload; preserving original context")
    except Exception as e:
        _warn(warnings, f"context redaction failed: {type(e).__name__}: {e}")
    return context_payload


def _collect_context(
    cfg: SnapshotConfig,
    warnings: list[str],
    redactors: list[Pattern[str]],
    extra: dict[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any] | None, dict[str, Any] | None]:
    """Assemble context payload (git, async, logs, extra). Returns (context, trace, eval)."""
    context_payload: dict[str, Any] = {}
    if cfg.include_env:
        context_payload["env"] = get_env_info(cfg)
    if cfg.include_git:
        git_ctx = _try_git_context(cfg, warnings)
        if git_ctx:
            context_payload["git"] = git_ctx
    if cfg.include_async_context:
        async_ctx = _try_async_context(warnings)
        if async_ctx:
            context_payload["async"] = async_ctx
    if cfg.capture_logs:
        logs = _try_log_context(cfg, warnings)
        if logs:
            context_payload["recent_logs"] = logs

    extra_context, trace_payload, eval_payload = split_extra_context(extra)
    if extra_context:
        context_payload.update(extra_context)
    context_payload = _redact_context(context_payload, redactors, cfg, warnings)
    return context_payload, trace_payload, eval_payload


def capture_exception(
    name: str,
    exc: BaseException,
    tb: types.TracebackType | None,
    cfg: SnapshotConfig,
    *,
    extra: dict[str, Any] | None = None,
) -> Path:
    """Capture exception details and write snapshot.

    Args:
        name: Identifier for this snapshot (e.g. ``"unhandled"``), used in
            the output filename and the debug-session envelope.
        exc: The exception instance to capture.
        tb: Traceback attached to *exc*, or ``None`` if unavailable.  When
            ``None`` no stack frames are collected.
        cfg: Snapshot configuration controlling redaction, frame depth,
            locals serialization, and output format.
        extra: Optional dict of additional context merged into the snapshot
            envelope.  Keys ``"_trace"`` and ``"_eval"`` are reserved and
            split into their own top-level payloads.
    """
    cfg = resolve_snapshot_config(cfg)
    warnings: list[str] = []
    redactors = compile_redactors(cfg.redact or ())
    frames = (
        collect_frames(tb, cfg, warnings=warnings, redactors=redactors) if tb is not None else []
    )
    timestamp_utc = dt.datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    traceback_text = "".join(traceback.format_exception(type(exc), exc, tb))
    if cfg.redact_traceback and redactors:
        traceback_text = redact_text(traceback_text, redactors)

    snapshot_payload: dict[str, Any] = {
        "exception": _summarize_exception(
            exc,
            cfg,
            warnings=warnings,
            redactors=redactors,
            redact_strings=bool(cfg.redact_exception_strings),
            max_depth=cfg.max_exception_depth,
        ),
        "traceback": traceback_text,
        "frames": frames,
        "capture_config": _serialize_config(cfg),
    }
    if frames:
        snapshot_payload["crash_frame_index"] = 0

    context_payload, trace_payload, eval_payload = _collect_context(cfg, warnings, redactors, extra)

    try:
        from ._version import __version__ as llmdebug_version
    except Exception as e:
        llmdebug_version = "unknown"
        _warn(warnings, f"version detection failed: {type(e).__name__}: {e}")

    if warnings:
        snapshot_payload["_capture_warnings"] = list(warnings)
        if cfg.debug:
            for w in warnings:
                sys.stderr.write(f"llmdebug: {w}\n")

    payload = build_debug_session(
        name=name,
        timestamp_utc=timestamp_utc,
        llmdebug_version=llmdebug_version,
        snapshot=snapshot_payload,
        context=context_payload or None,
        trace_payload=trace_payload,
        eval_payload=eval_payload,
    )
    return write_bundle(payload, cfg)
