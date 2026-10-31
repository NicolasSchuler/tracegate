"""Jupyter/IPython integration for automatic crash capture.

Captures debug snapshots on cell errors with rich HTML display in notebooks.
Follows the hooks.py pattern: module-level state, idempotent install/uninstall,
rate limiting, PII redaction.

Usage:
    # In a notebook cell:
    %load_ext llmdebug

    # Or programmatically:
    import llmdebug
    llmdebug.load_jupyter()

    # After any cell error, a compact banner is shown.
    # Use magic commands for deeper analysis:
    %llmdebug              # Show latest snapshot
    %llmdebug diff         # Compare latest vs previous
    %llmdebug list         # List recent snapshots
"""

from __future__ import annotations

import html as _html_mod
import sys
import threading
from collections.abc import Callable
from typing import Any, cast

from .config import SnapshotConfig
from .hooks import (
    _build_production_config,  # pyright: ignore[reportPrivateUsage]
    _capture_safely,  # pyright: ignore[reportPrivateUsage]
    _RateLimiter,  # pyright: ignore[reportPrivateUsage]
)
from .redaction_policy import RedactionProfile

# ---------------------------------------------------------------------------
# Module-level state (same pattern as hooks.py)
# ---------------------------------------------------------------------------

_installed = False
_install_lock = threading.Lock()
_original_showtraceback: Callable[..., Any] | None = None
_cfg: SnapshotConfig | None = None
_limiter: _RateLimiter | None = None
_ipython_ref: Any = None  # Weak reference isn't needed — IPython is a singleton

# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------

_BANNER_CSS = (
    "border-left:4px solid #e74c3c; background:#fdf2f2; padding:8px 12px;"
    "margin:8px 0; font-family:monospace; font-size:13px; border-radius:0 4px 4px 0"
)

_FULL_CSS = (
    "border:1px solid #ddd; background:#fafafa; padding:12px;"
    "margin:8px 0; font-family:monospace; font-size:13px; border-radius:4px"
)


def _as_dict(value: Any) -> dict[str, Any]:
    """Return *value* if it is a dict, else an empty dict."""
    return cast(dict[str, Any], value) if isinstance(value, dict) else {}


def _as_list(value: Any) -> list[Any]:
    """Return *value* if it is a list, else an empty list."""
    return cast(list[Any], value) if isinstance(value, list) else []


def _select_frame(frames: list[Any], crash_idx: int) -> dict[str, Any]:
    """Return the crash frame, else the first dict frame, else an empty dict."""
    if 0 <= crash_idx < len(frames) and isinstance(frames[crash_idx], dict):
        return cast(dict[str, Any], frames[crash_idx])
    for frame in frames:
        if isinstance(frame, dict):
            return cast(dict[str, Any], frame)
    return {}


def _is_notebook(ipython: Any) -> bool:
    """Return whether *ipython* is a Jupyter notebook kernel rather than terminal IPython."""
    return type(ipython).__name__ == "ZMQInteractiveShell"


def _get_out_dir() -> str:
    """Return the configured output directory, defaulting to '.llmdebug'."""
    return _cfg.out_dir if _cfg else ".llmdebug"


def _display_html_or_text(
    ipython: Any,
    *,
    html_fn: Callable[[], str],
    text_fn: Callable[[], str],
) -> None:
    """Display HTML in notebooks and plain text in terminals.

    Falls back to text when ``IPython.display`` cannot be imported.
    """
    if _is_notebook(ipython):
        try:
            from IPython.display import HTML, display  # pyright: ignore[reportUnknownVariableType]

            display(HTML(html_fn()))  # pyright: ignore[reportUnknownVariableType]
            return
        except ImportError:
            pass
    print(text_fn())


def _display_markdown_or_text(ipython: Any, text: str) -> None:
    """Display Markdown in notebooks and plain text in terminals.

    Falls back to text when ``IPython.display`` cannot be imported.
    """
    if _is_notebook(ipython):
        try:
            from IPython.display import (  # pyright: ignore[reportUnknownVariableType]
                Markdown,
                display,  # pyright: ignore[reportUnknownVariableType]
            )

            display(Markdown(text))  # pyright: ignore[reportUnknownVariableType]
            return
        except ImportError:
            pass
    print(text)


def _exc_summary(snapshot: dict[str, Any]) -> tuple[str, str, str, int | str]:
    """Return ``(exc_type, exc_message, crash_file, crash_line)`` for *snapshot*."""
    exc = _as_dict(snapshot.get("exception"))
    exc_type = str(exc.get("type") or "UnknownError")
    exc_msg = str(exc.get("message") or "")
    frames = _as_list(snapshot.get("frames"))
    crash_idx = snapshot.get("crash_frame_index", 0)
    cfi = crash_idx if isinstance(crash_idx, int) else 0
    fr0 = _select_frame(frames, cfi)
    crash_file = str(fr0.get("file_rel") or fr0.get("file") or "?")
    crash_line_raw = fr0.get("line", "?")
    crash_line: int | str = crash_line_raw if isinstance(crash_line_raw, (int, str)) else "?"
    return exc_type, exc_msg, crash_file, crash_line


def _format_snapshot_banner_html(snapshot: dict[str, Any]) -> str:
    """Render the compact HTML banner (~4 lines) shown after a cell error."""
    exc_type, exc_msg, crash_file, crash_line = _exc_summary(snapshot)
    # Truncate long messages
    if len(exc_msg) > 120:
        exc_msg = exc_msg[:117] + "..."
    return (
        f'<div style="{_BANNER_CSS}">'
        f"<b>llmdebug</b> captured <b>{_esc(exc_type)}</b>"
        f" at {_esc(str(crash_file))}:{_esc(str(crash_line))}<br>"
        f"<span style='color:#666'>{_esc(exc_msg)}</span><br>"
        f"<span style='color:#888; font-size:11px'>"
        f"Use <code>%llmdebug</code> to view full snapshot</span>"
        f"</div>"
    )


def _format_snapshot_banner_text(snapshot: dict[str, Any]) -> str:
    """Render the plain-text banner shown after a cell error in terminal IPython."""
    exc_type, exc_msg, crash_file, crash_line = _exc_summary(snapshot)
    if len(exc_msg) > 100:
        exc_msg = exc_msg[:97] + "..."
    return (
        f"\n[llmdebug] captured {exc_type} at {crash_file}:{crash_line}\n"
        f"  {exc_msg}\n"
        f"  Use %llmdebug to view details\n"
    )


def _parse_snapshot_display_data(
    snapshot: dict[str, Any],
) -> tuple[dict[str, Any], list[Any], int]:
    """Return the exception dict, frame list, and crash-frame index of *snapshot*."""
    exc = _as_dict(snapshot.get("exception"))
    frames = _as_list(snapshot.get("frames"))
    crash_idx_raw = snapshot.get("crash_frame_index", 0)
    crash_idx = crash_idx_raw if isinstance(crash_idx_raw, int) else 0
    return exc, frames, crash_idx


def _html_crash_frame_parts(frames: list[Any], crash_idx: int) -> list[str]:
    """Return HTML parts for the crash frame and its first 20 locals."""
    if not frames:
        return []
    fr0 = _select_frame(frames, crash_idx)
    if not fr0:
        return []
    file_name = str(fr0.get("file_rel") or fr0.get("file") or "?")
    line = fr0.get("line", "?")
    func = str(fr0.get("function") or "?")
    code = str(fr0.get("code") or "")
    parts: list[str] = [
        "<hr style='border:none;border-top:1px solid #eee;margin:8px 0'>",
        f"<p style='margin:4px 0'><b>Crash frame:</b> "
        f"{_esc(file_name)}:{_esc(str(line))} in <code>{_esc(func)}</code></p>",
    ]
    if code:
        parts.append(
            f"<pre style='background:#f5f5f5;padding:8px;border-radius:3px;"
            f"overflow-x:auto;margin:4px 0'>{_esc(code)}</pre>"
        )
    locals_raw = fr0.get("locals")
    if isinstance(locals_raw, dict) and locals_raw:
        locals_data = cast(dict[str, Any], locals_raw)
        parts.append("<p style='margin:8px 0 4px 0'><b>Locals:</b></p>")
        parts.append("<table style='border-collapse:collapse;width:100%;font-size:12px'>")
        for var_name, var_val in list(locals_data.items())[:20]:
            val_str = _format_local_value(var_val)
            parts.append(
                f"<tr><td style='padding:2px 8px 2px 0;color:#7f8c8d;"
                f"vertical-align:top;white-space:nowrap'>{_esc(str(var_name))}</td>"
                f"<td style='padding:2px 0'><code>{_esc(val_str)}</code></td></tr>"
            )
        parts.append("</table>")
    return parts


def _html_jupyter_part(snapshot: dict[str, Any]) -> list[str]:
    """Return HTML parts for the Jupyter execution-count section."""
    _empty: dict[str, Any] = {}
    jupyter_ctx_raw = snapshot.get("jupyter")
    if not jupyter_ctx_raw:
        return []
    jupyter_ctx = (
        cast(dict[str, Any], jupyter_ctx_raw) if isinstance(jupyter_ctx_raw, dict) else _empty
    )
    ec = jupyter_ctx.get("execution_count", "?")
    return [
        "<hr style='border:none;border-top:1px solid #eee;margin:8px 0'>",
        f"<p style='margin:4px 0;color:#888;font-size:11px'>Cell In[{_esc(str(ec))}]</p>",
    ]


def _format_full_snapshot_html(snapshot: dict[str, Any]) -> str:
    """Render the full snapshot as HTML for ``%llmdebug show`` in notebooks."""
    exc, frames, crash_idx = _parse_snapshot_display_data(snapshot)
    parts = [
        f'<div style="{_FULL_CSS}">',
        f"<h3 style='margin:0 0 8px 0; color:#e74c3c'>{_esc(str(exc.get('type') or 'UnknownError'))}</h3>",
        f"<p style='margin:0 0 8px 0'>{_esc(str(exc.get('message') or ''))}</p>",
        *_html_crash_frame_parts(frames, crash_idx),
        *_html_jupyter_part(snapshot),
        "</div>",
    ]
    return "\n".join(parts)


def _text_crash_frame_lines(frames: list[Any], crash_idx: int) -> list[str]:
    """Return text lines for the crash frame and its first 20 locals."""
    if not frames:
        return []
    fr0 = _select_frame(frames, crash_idx)
    if not fr0:
        return []
    file_name = str(fr0.get("file_rel") or fr0.get("file") or "?")
    line = fr0.get("line", "?")
    func = str(fr0.get("function") or "?")
    code = str(fr0.get("code") or "")
    lines: list[str] = [f"**Crash frame:** {file_name}:{line} in `{func}`"]
    if code:
        lines.append(f"```python\n{code}\n```")
    locals_raw = fr0.get("locals")
    if isinstance(locals_raw, dict) and locals_raw:
        locals_data = cast(dict[str, Any], locals_raw)
        lines.append("\n**Locals:**")
        for var_name, var_val in list(locals_data.items())[:20]:
            val_str = _format_local_value(var_val)
            lines.append(f"  {var_name} = {val_str}")
    return lines


def _text_jupyter_line(snapshot: dict[str, Any]) -> list[str]:
    """Return text lines for the Jupyter execution-count section."""
    _empty: dict[str, Any] = {}
    jupyter_ctx_raw = snapshot.get("jupyter")
    if not jupyter_ctx_raw:
        return []
    jupyter_ctx = (
        cast(dict[str, Any], jupyter_ctx_raw) if isinstance(jupyter_ctx_raw, dict) else _empty
    )
    ec = jupyter_ctx.get("execution_count", "?")
    return [f"\n_Cell In[{ec}]_"]


def _format_full_snapshot_text(snapshot: dict[str, Any]) -> str:
    """Render the full snapshot as Markdown text for terminal IPython."""
    exc, frames, crash_idx = _parse_snapshot_display_data(snapshot)
    lines: list[str] = [
        f"## {exc.get('type') or 'UnknownError'!s}",
        f"{exc.get('message') or ''!s}",
        "",
        *_text_crash_frame_lines(frames, crash_idx),
        *_text_jupyter_line(snapshot),
    ]
    return "\n".join(lines)


def _format_local_value(val: Any) -> str:
    """Format a local variable value for display, truncating plain values to 100 chars."""
    if isinstance(val, dict):
        val_d = cast(dict[str, Any], val)
        # Array-like objects
        for key in ("__array__", "__dataframe__", "__series__"):
            if key in val_d:
                shape = val_d.get("shape", "?")
                dtype = val_d.get("dtype", "?")
                return f"{val_d[key]} shape={shape} dtype={dtype}"
        # Truncated string
        if "__truncated__" in val_d:
            return str(val_d.get("value", val_d))
    return str(cast(Any, val))[:100]


def _esc(text: str) -> str:
    """Escape HTML entities (safe for both text content and attribute values)."""
    return _html_mod.escape(text)


# ---------------------------------------------------------------------------
# Core: showtraceback wrapper
# ---------------------------------------------------------------------------


def _make_wrapped_showtraceback(
    original: Callable[..., Any],
    cfg: SnapshotConfig,
    limiter: _RateLimiter,
    ipython: Any,
) -> Callable[..., Any]:
    """Create a wrapped showtraceback that captures snapshots after errors."""

    def wrapped_showtraceback(*args: Any, **kwargs: Any) -> None:
        """Show the traceback via the original, then capture and display a banner."""
        # Always call the original first — user sees normal traceback
        original(*args, **kwargs)

        # Extract the current exception info
        _exc_type, exc_value, exc_tb = sys.exc_info()
        if exc_value is None or exc_tb is None:
            return

        # Don't capture intentional exits
        if isinstance(exc_value, (KeyboardInterrupt, SystemExit)):
            return

        # Capture + display must never break IPython
        try:
            extra: dict[str, Any] = {"jupyter": _extract_jupyter_context(ipython)}
            _capture_safely("jupyter_cell", exc_value, exc_tb, cfg, limiter, extra=extra)
            _display_snapshot_banner(ipython, cfg.out_dir)
        except Exception as capture_exc:
            # A broken safety net is worse than a noisy one (hooks.py principle)
            sys.stderr.write(
                f"llmdebug: jupyter capture/display failed: "
                f"{type(capture_exc).__name__}: {capture_exc}\n"
            )

    return wrapped_showtraceback


def _extract_jupyter_context(ipython: Any) -> dict[str, Any]:
    """Extract Jupyter-specific context from the IPython shell.

    Records the execution count and the latest cell source, truncated to 500
    characters, or ``cell_source_error`` if the history lookup fails.
    """
    ctx: dict[str, Any] = {}

    try:
        ctx["execution_count"] = ipython.execution_count
    except AttributeError:
        pass

    try:
        # Get the most recent cell source
        hm = ipython.history_manager
        if hm is not None:
            history = list(hm.get_tail(1, raw=True))
            if history:
                _, _, cell_source = history[0]
                # Truncate long cells
                if isinstance(cell_source, str):
                    if len(cell_source) > 500:
                        cell_source = cell_source[:497] + "..."
                    ctx["cell_source"] = cell_source
    except Exception as e:
        ctx["cell_source_error"] = f"{type(e).__name__}: {e}"

    return ctx


def _display_snapshot_banner(ipython: Any, out_dir: str) -> None:
    """Display the banner for the latest snapshot in *out_dir*, if there is one."""
    from .output import get_latest_snapshot

    snapshot = get_latest_snapshot(out_dir)
    if snapshot is None:
        return

    _display_html_or_text(
        ipython,
        html_fn=lambda: _format_snapshot_banner_html(snapshot),
        text_fn=lambda: _format_snapshot_banner_text(snapshot),
    )


# ---------------------------------------------------------------------------
# Magic commands
# ---------------------------------------------------------------------------


def _register_magics(ipython: Any) -> None:
    """Register the ``%llmdebug`` magic command on *ipython*.

    IPython is imported only when this is called.
    """
    from IPython.core.magic import (  # pyright: ignore[reportUnknownVariableType]
        Magics,
        line_magic,  # pyright: ignore[reportUnknownVariableType]
        magics_class,  # pyright: ignore[reportUnknownVariableType]
    )

    @magics_class
    class LLMDebugMagics(Magics):  # type: ignore[misc]
        """IPython magics class that provides the ``%llmdebug`` line magic."""

        @property
        def _shell(self) -> Any:
            """Return the shell typed as ``Any``, since IPython stubs vary across versions."""
            return self.shell  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]

        @line_magic
        def llmdebug(self, line: str) -> None:  # type: ignore[override]
            """Run a ``%llmdebug`` subcommand; usage: ``%llmdebug [subcommand]``.

            Subcommands are ``show`` (the default), ``list``, ``diff``, ``install``,
            ``uninstall``, and ``config``; anything else prints the help text.

            Args:
                line: The text after ``%llmdebug``; its first word selects the
                    subcommand.
            """
            parts = line.strip().split(None, 1)
            subcmd = parts[0] if parts else "show"

            if subcmd == "show":
                self._cmd_show()
            elif subcmd == "list":
                self._cmd_list()
            elif subcmd == "diff":
                self._cmd_diff()
            elif subcmd == "install":
                install(ipython)
                print("[llmdebug] installed")
            elif subcmd == "uninstall":
                uninstall()
                print("[llmdebug] uninstalled")
            elif subcmd == "config":
                self._cmd_config()
            else:
                self._cmd_help()

        def _cmd_show(self) -> None:
            from .output import get_latest_snapshot

            snapshot = get_latest_snapshot(_get_out_dir())
            if snapshot is None:
                print("[llmdebug] No snapshots found.")
                return

            _display_html_or_text(
                self._shell,
                html_fn=lambda: _format_full_snapshot_html(snapshot),
                text_fn=lambda: _format_full_snapshot_text(snapshot),
            )

        def _cmd_list(self) -> None:
            from ._snapshot_loader import find_all_snapshots

            entries = find_all_snapshots(_get_out_dir(), limit=20)
            if not entries:
                print("[llmdebug] No snapshots found.")
                return

            def _html() -> str:
                rows: list[str] = []
                for i, entry in enumerate(entries[:20], 1):
                    e = _as_dict(entry)
                    exc_type = str(e.get("exc_type", "?"))
                    exc_message = str(e.get("exc_message", ""))
                    file_name = str(e.get("file", "?"))
                    line = e.get("line", "?")
                    rows.append(
                        f"<tr><td>#{i}</td>"
                        f"<td>{_esc(exc_type)}</td>"
                        f"<td>{_esc(exc_message)}</td>"
                        f"<td>{_esc(file_name)}:{_esc(str(line))}</td></tr>"
                    )
                return (
                    "<table style='border-collapse:collapse;font-family:monospace;"
                    "font-size:12px'>"
                    "<tr style='border-bottom:1px solid #ddd'>"
                    "<th style='padding:4px 8px'>#</th>"
                    "<th style='padding:4px 8px'>Type</th>"
                    "<th style='padding:4px 8px'>Message</th>"
                    "<th style='padding:4px 8px'>Location</th></tr>" + "\n".join(rows) + "</table>"
                )

            def _text() -> str:
                lines: list[str] = []
                for i, entry in enumerate(entries[:20], 1):
                    e = _as_dict(entry)
                    exc_type = str(e.get("exc_type", "?"))
                    exc_message = str(e.get("exc_message", ""))
                    file_name = str(e.get("file", "?"))
                    line = e.get("line", "?")
                    lines.append(f"  #{i}  {exc_type}: {exc_message}  ({file_name}:{line})")
                return "\n".join(lines)

            _display_html_or_text(
                self._shell,
                html_fn=_html,
                text_fn=_text,
            )

        def _cmd_diff(self) -> None:
            from ._snapshot_loader import find_all_snapshots, resolve_snapshot_ref
            from .snapshot_diff import diff_snapshots, format_diff_text

            out_dir = _get_out_dir()
            all_snaps = find_all_snapshots(out_dir)

            try:
                new = resolve_snapshot_ref("latest", out_dir, all_snaps)
                old = resolve_snapshot_ref("previous", out_dir, all_snaps)
            except FileNotFoundError as e:
                print(f"[llmdebug] {e}")
                return

            result = diff_snapshots(old, new)
            text = format_diff_text(result)
            _display_markdown_or_text(self._shell, text)

        def _cmd_config(self) -> None:
            if _cfg is None:
                print("[llmdebug] Not installed. Run %llmdebug install first.")
                return
            import dataclasses

            print("[llmdebug] Active configuration:")
            for f in dataclasses.fields(_cfg):
                val = getattr(_cfg, f.name)
                print(f"  {f.name} = {val!r}")

        def _cmd_help(self) -> None:
            print(
                "Usage: %llmdebug [command]\n"
                "\n"
                "Commands:\n"
                "  show         Show latest snapshot (default)\n"
                "  list         List recent snapshots\n"
                "  diff         Compare latest vs previous snapshot\n"
                "  install      Install capture hooks\n"
                "  uninstall    Remove capture hooks\n"
                "  config       Show active configuration"
            )

    ipython.register_magics(LLMDebugMagics)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def install(
    ipython: Any = None,
    *,
    out_dir: str = ".llmdebug",
    max_captures_per_minute: int = 10,
    pii_safe: bool = True,
    redaction_profile: RedactionProfile | str | None = None,
    include_git: bool = False,
    include_array_stats: bool = True,
    lock_timeout: float = 1.0,
    debug: bool = False,
    **config_overrides: Any,
) -> None:
    """Install llmdebug into the current IPython session.

    Idempotent — safe to call multiple times.

    Args:
        ipython: IPython shell to hook; detected with ``get_ipython()`` if None.
        out_dir: Snapshot output directory.
        max_captures_per_minute: Captures allowed per unique error signature
            (exception type, file, line) in a sliding 60-second window.
        pii_safe: Default to the ``"prod"`` redaction profile (built-in PII patterns
            plus traceback and exception-string redaction) and reject a custom
            ``redact`` override.
        redaction_profile: Named redaction profile (``"dev"``, ``"ci"``, or
            ``"prod"``) that overrides the one *pii_safe* selects.
        include_git: Include git context (disabled by default).
        include_array_stats: Include array shape/stats (enabled for data science).
        lock_timeout: File lock timeout in seconds.
        debug: Report capture warnings through the ``llmdebug.diagnostics`` logger
            (stderr by default); capture failures are written to stderr regardless.
        **config_overrides: Additional SnapshotConfig fields.

    Raises:
        RuntimeError: If *ipython* is None and IPython is not installed or no
            IPython session is active.
        ValueError: If the redaction settings are invalid or conflict (such as
            ``redact`` passed while *pii_safe* is True) or *max_captures_per_minute*
            is below 1.
    """
    global _installed, _original_showtraceback, _cfg, _limiter, _ipython_ref

    with _install_lock:
        if _installed:
            return

        if ipython is None:
            try:
                from IPython import get_ipython  # type: ignore[attr-defined]

                ipython = cast(Any, get_ipython())
            except ImportError:
                raise RuntimeError(
                    "IPython is not installed. Install with: pip install 'llmdebug[jupyter]'"
                ) from None
            if ipython is None:
                raise RuntimeError(
                    "No active IPython session found. "
                    "Run this inside IPython or a Jupyter notebook."
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

        limiter = _RateLimiter(max_captures_per_minute, 60.0)

        # Save original and wrap showtraceback
        original: Callable[..., Any] = cast(Callable[..., Any], ipython.showtraceback)
        ipython.showtraceback = _make_wrapped_showtraceback(original, cfg, limiter, ipython)
        try:
            _register_magics(ipython)
        except BaseException:
            # RCA-07: a wrapped hook with _installed=False survived uninstall() and was
            # wrapped again by the next install(), printing and capturing twice.
            ipython.showtraceback = original
            raise

        _original_showtraceback = original
        _cfg = cfg
        _limiter = limiter
        _ipython_ref = ipython
        _installed = True


def uninstall() -> None:
    """Remove llmdebug from the current IPython session.

    Restores the original ``showtraceback``; the ``%llmdebug`` magic stays
    registered. Idempotent — safe to call multiple times.
    """
    global _installed, _original_showtraceback, _cfg, _limiter, _ipython_ref

    with _install_lock:
        if not _installed:
            return

        # Restore original showtraceback
        if _ipython_ref is not None and _original_showtraceback is not None:
            _ipython_ref.showtraceback = _original_showtraceback

        _original_showtraceback = None
        _cfg = None
        _limiter = None
        _ipython_ref = None
        _installed = False


# ---------------------------------------------------------------------------
# IPython extension API
# ---------------------------------------------------------------------------


def load_ipython_extension(ipython: Any) -> None:
    """Install llmdebug when IPython runs ``%load_ext llmdebug``.

    Args:
        ipython: The IPython shell loading the extension.
    """
    install(ipython)


def unload_ipython_extension(ipython: Any) -> None:
    """Uninstall llmdebug when IPython runs ``%unload_ext llmdebug``.

    Args:
        ipython: The IPython shell unloading the extension; unused, because
            :func:`uninstall` restores the shell recorded at install time.
    """
    uninstall()
