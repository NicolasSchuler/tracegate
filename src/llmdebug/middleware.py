"""WSGI and ASGI middleware for automatic crash capture.

Zero framework dependencies — works with any WSGI or ASGI application.

Usage (Flask):
    app.wsgi_app = LLMDebugWSGIMiddleware(app.wsgi_app)

Usage (FastAPI):
    app.add_middleware(LLMDebugASGIMiddleware)

Usage (Django WSGI):
    application = LLMDebugWSGIMiddleware(application)
"""

from __future__ import annotations

import re
from collections.abc import Callable
from typing import Any

from .hooks import (
    PII_PATTERNS,
    _build_production_config,  # pyright: ignore[reportPrivateUsage]
    _capture_safely,  # pyright: ignore[reportPrivateUsage]
    _RateLimiter,  # pyright: ignore[reportPrivateUsage]
)
from .redaction_policy import RedactionProfile

_PII_COMPILED = [re.compile(p) for p in PII_PATTERNS]


def _redact_query_string(qs: str) -> str:
    """Apply PII patterns to a query string."""
    for pattern in _PII_COMPILED:
        qs = pattern.sub("[REDACTED]", qs)
    return qs


# ---------------------------------------------------------------------------
# Context extraction helpers (PII-safe — no headers or body)
# ---------------------------------------------------------------------------


def _extract_wsgi_context(environ: dict[str, Any]) -> dict[str, Any]:
    """Extract PII-safe request context from a WSGI environ."""
    path = environ.get("PATH_INFO", "?")
    path_text = path if isinstance(path, str) else str(path)
    return {
        "hook": "wsgi",
        "request": {
            "method": environ.get("REQUEST_METHOD", "?"),
            "path": _redact_query_string(path_text),
            "query": _redact_query_string(environ.get("QUERY_STRING", "")),
            "content_type": environ.get("CONTENT_TYPE", ""),
            "server": environ.get("SERVER_NAME", ""),
        },
    }


def _extract_asgi_context(scope: dict[str, Any]) -> dict[str, Any]:
    """Extract PII-safe request context from an ASGI scope."""
    path = scope.get("path", "?")
    if isinstance(path, bytes):
        path = path.decode("latin-1", errors="replace")
    if not isinstance(path, str):
        path = str(path)
    query = scope.get("query_string", b"")
    if isinstance(query, bytes):
        query = query.decode("latin-1", errors="replace")
    return {
        "hook": "asgi",
        "request": {
            "type": scope.get("type", "?"),
            "method": scope.get("method", "?"),
            "path": _redact_query_string(path),
            "query": _redact_query_string(query),
            "server": (
                f"{scope['server'][0]}:{scope['server'][1]}"
                if scope.get("server") and len(scope["server"]) >= 2
                else ""
            ),
        },
    }


# ---------------------------------------------------------------------------
# WSGI Middleware
# ---------------------------------------------------------------------------


class LLMDebugWSGIMiddleware:
    """WSGI middleware that captures crash snapshots on unhandled exceptions.

    Transparent — catches exceptions, writes a snapshot, then re-raises
    so the WSGI server can return an appropriate error response.

    Args:
        app: The WSGI application to wrap.
        out_dir: Snapshot output directory.
        pii_safe: Enable built-in PII redaction patterns.
        redaction_profile: Built-in redaction profile override.
        max_captures_per_minute: Rate limit per unique error signature.
        **config_overrides: Additional :class:`SnapshotConfig` fields.
    """

    def __init__(
        self,
        app: Callable[..., Any],
        *,
        out_dir: str = ".llmdebug",
        pii_safe: bool = True,
        redaction_profile: RedactionProfile | str | None = None,
        max_captures_per_minute: int = 10,
        **config_overrides: Any,
    ) -> None:
        self.app = app
        self._cfg = _build_production_config(
            out_dir,
            pii_safe,
            redaction_profile=redaction_profile,
            **config_overrides,
        )
        self._limiter = _RateLimiter(max_captures_per_minute, 60.0)

    def _capture_wsgi_exception(self, exc: Exception, environ: dict[str, Any]) -> None:
        tb = exc.__traceback__
        if tb is None:
            return
        extra = _extract_wsgi_context(environ)
        _capture_safely("wsgi_error", exc, tb, self._cfg, self._limiter, extra=extra)

    def __call__(self, environ: dict[str, Any], start_response: Callable[..., Any]) -> Any:
        try:
            iterable = self.app(environ, start_response)
        except Exception as exc:
            self._capture_wsgi_exception(exc, environ)
            raise

        def _iter_with_capture() -> Any:
            try:
                yield from iterable
            except Exception as exc:
                self._capture_wsgi_exception(exc, environ)
                raise
            finally:
                close = getattr(iterable, "close", None)
                if callable(close):
                    close()

        return _iter_with_capture()


# ---------------------------------------------------------------------------
# ASGI Middleware
# ---------------------------------------------------------------------------


class LLMDebugASGIMiddleware:
    """ASGI middleware that captures crash snapshots on unhandled exceptions.

    Transparent — catches exceptions, writes a snapshot, then re-raises.
    Only intercepts ``http`` and ``websocket`` scopes; ``lifespan`` events
    pass through untouched.

    Example (FastAPI)::

        from fastapi import FastAPI
        from llmdebug.middleware import LLMDebugASGIMiddleware

        app = FastAPI()
        app.add_middleware(LLMDebugASGIMiddleware, pii_safe=True)

    Example (Starlette)::

        from starlette.applications import Starlette
        from llmdebug.middleware import LLMDebugASGIMiddleware

        app = Starlette(...)
        app = LLMDebugASGIMiddleware(app)

    Args:
        app: The ASGI application to wrap.
        out_dir: Snapshot output directory.
        pii_safe: Enable built-in PII redaction patterns.
        redaction_profile: Built-in redaction profile override.
        max_captures_per_minute: Rate limit per unique error signature.
        **config_overrides: Additional :class:`SnapshotConfig` fields.
    """

    def __init__(
        self,
        app: Callable[..., Any],
        *,
        out_dir: str = ".llmdebug",
        pii_safe: bool = True,
        redaction_profile: RedactionProfile | str | None = None,
        max_captures_per_minute: int = 10,
        **config_overrides: Any,
    ) -> None:
        self.app = app
        self._cfg = _build_production_config(
            out_dir,
            pii_safe,
            redaction_profile=redaction_profile,
            **config_overrides,
        )
        self._limiter = _RateLimiter(max_captures_per_minute, 60.0)

    async def __call__(
        self,
        scope: dict[str, Any],
        receive: Callable[..., Any],
        send: Callable[..., Any],
    ) -> None:
        if scope.get("type") not in ("http", "websocket"):
            await self.app(scope, receive, send)
            return
        try:
            await self.app(scope, receive, send)
        except Exception as exc:
            tb = exc.__traceback__
            if tb is not None:
                extra = _extract_asgi_context(scope)
                _capture_safely("asgi_error", exc, tb, self._cfg, self._limiter, extra=extra)
            raise
