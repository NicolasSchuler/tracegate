"""Tests for WSGI and ASGI middleware."""

from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Any

import pytest

from llmdebug.middleware import (
    LLMDebugASGIMiddleware,
    LLMDebugWSGIMiddleware,
    _extract_asgi_context,
    _extract_wsgi_context,
    _redact_query_string,
)

from .conftest import read_latest_snapshot

# ---------------------------------------------------------------------------
# WSGI helpers
# ---------------------------------------------------------------------------


def _make_ok_wsgi_app():
    def app(environ, start_response):
        start_response("200 OK", [("Content-Type", "text/plain")])
        return [b"OK"]

    return app


def _make_failing_wsgi_app():
    def app(environ, start_response):
        raise ValueError("wsgi test crash")

    return app


def _make_iter_failing_wsgi_app():
    def app(environ, start_response):
        _ = environ
        start_response("200 OK", [("Content-Type", "text/plain")])

        def _gen():
            yield b"OK"
            raise RuntimeError("wsgi iter crash")

        return _gen()

    return app


def _make_wsgi_environ(**overrides: Any) -> dict[str, Any]:
    base = {
        "REQUEST_METHOD": "GET",
        "PATH_INFO": "/test",
        "QUERY_STRING": "q=1",
        "CONTENT_TYPE": "text/html",
        "SERVER_NAME": "localhost",
    }
    base.update(overrides)
    return base


def _dummy_start_response(status, headers):
    pass


# ---------------------------------------------------------------------------
# ASGI helpers
# ---------------------------------------------------------------------------


async def _ok_asgi_app(scope, receive, send):
    await send(
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [[b"content-type", b"text/plain"]],
        }
    )
    await send({"type": "http.response.body", "body": b"OK"})


async def _failing_asgi_app(scope, receive, send):
    raise ValueError("asgi test crash")


def _make_asgi_scope(**overrides: Any) -> dict[str, Any]:
    base = {
        "type": "http",
        "method": "POST",
        "path": "/api/data",
        "query_string": b"page=2",
        "server": ("127.0.0.1", 8000),
    }
    base.update(overrides)
    return base


async def _noop_receive():
    return {}


async def _noop_send(message):
    pass


# ---------------------------------------------------------------------------
# WSGI Middleware Tests
# ---------------------------------------------------------------------------


class TestWSGIMiddleware:
    def test_passes_through_normal_request(self):
        app = _make_ok_wsgi_app()
        mw = LLMDebugWSGIMiddleware(app)
        environ = _make_wsgi_environ()
        result = mw(environ, _dummy_start_response)
        assert list(result) == [b"OK"]

    def test_captures_on_exception(self, tmp_path: Path):
        app = _make_failing_wsgi_app()
        mw = LLMDebugWSGIMiddleware(app, out_dir=str(tmp_path), pii_safe=False)
        environ = _make_wsgi_environ()

        with pytest.raises(ValueError, match="wsgi test crash"):
            mw(environ, _dummy_start_response)

        snapshot = read_latest_snapshot(tmp_path)
        assert snapshot["name"] == "wsgi_error"
        assert snapshot["exception"]["type"] == "ValueError"
        assert snapshot["hook"] == "wsgi"

    def test_reraises_exception(self):
        app = _make_failing_wsgi_app()
        mw = LLMDebugWSGIMiddleware(app)

        with pytest.raises(ValueError, match="wsgi test crash"):
            mw(_make_wsgi_environ(), _dummy_start_response)

    def test_captures_on_wsgi_iteration_exception(self, tmp_path: Path):
        app = _make_iter_failing_wsgi_app()
        mw = LLMDebugWSGIMiddleware(app, out_dir=str(tmp_path), pii_safe=False)

        result = mw(_make_wsgi_environ(), _dummy_start_response)
        iterator = iter(result)
        assert next(iterator) == b"OK"

        with pytest.raises(RuntimeError, match="wsgi iter crash"):
            next(iterator)

        snapshot = read_latest_snapshot(tmp_path)
        assert snapshot["name"] == "wsgi_error"
        assert snapshot["exception"]["type"] == "RuntimeError"
        assert snapshot["hook"] == "wsgi"

    def test_includes_request_context(self, tmp_path: Path):
        app = _make_failing_wsgi_app()
        mw = LLMDebugWSGIMiddleware(app, out_dir=str(tmp_path), pii_safe=False)
        environ = _make_wsgi_environ(
            REQUEST_METHOD="POST", PATH_INFO="/submit", QUERY_STRING="id=42"
        )

        with pytest.raises(ValueError):
            mw(environ, _dummy_start_response)

        snapshot = read_latest_snapshot(tmp_path)
        req = snapshot["request"]
        assert req["method"] == "POST"
        assert req["path"] == "/submit"
        assert req["query"] == "id=42"

    def test_rate_limited(self, tmp_path: Path):
        app = _make_failing_wsgi_app()
        mw = LLMDebugWSGIMiddleware(
            app,
            out_dir=str(tmp_path),
            pii_safe=False,
            max_captures_per_minute=1,
        )
        environ = _make_wsgi_environ()

        # First request — captured
        with pytest.raises(ValueError):
            mw(environ, _dummy_start_response)
        snapshot1 = read_latest_snapshot(tmp_path)
        assert snapshot1["name"] == "wsgi_error"

        # Record timestamp of first snapshot
        ts1 = snapshot1["timestamp_utc"]

        # Second request — rate-limited, so snapshot should NOT be updated
        with pytest.raises(ValueError):
            mw(environ, _dummy_start_response)
        snapshot2 = read_latest_snapshot(tmp_path)
        assert snapshot2["timestamp_utc"] == ts1

    def test_pii_safe_default(self):
        app = _make_ok_wsgi_app()
        mw = LLMDebugWSGIMiddleware(app)
        # PII patterns should be in the config's redact tuple
        assert len(mw._cfg.redact) > 0


# ---------------------------------------------------------------------------
# ASGI Middleware Tests
# ---------------------------------------------------------------------------


class TestASGIMiddleware:
    def test_passes_through_normal_request(self):
        mw = LLMDebugASGIMiddleware(_ok_asgi_app)
        scope = _make_asgi_scope()
        sent_messages: list[dict[str, Any]] = []

        async def _tracking_send(message: dict[str, Any]) -> None:
            sent_messages.append(message)

        asyncio.run(mw(scope, _noop_receive, _tracking_send))

        # Verify the response was properly forwarded
        assert any(m.get("type") == "http.response.start" for m in sent_messages)
        assert any(m.get("type") == "http.response.body" for m in sent_messages)

    def test_captures_on_exception(self, tmp_path: Path):
        mw = LLMDebugASGIMiddleware(_failing_asgi_app, out_dir=str(tmp_path), pii_safe=False)
        scope = _make_asgi_scope()

        with pytest.raises(ValueError, match="asgi test crash"):
            asyncio.run(mw(scope, _noop_receive, _noop_send))

        snapshot = read_latest_snapshot(tmp_path)
        assert snapshot["name"] == "asgi_error"
        assert snapshot["exception"]["type"] == "ValueError"
        assert snapshot["hook"] == "asgi"

    def test_reraises_exception(self):
        mw = LLMDebugASGIMiddleware(_failing_asgi_app)
        scope = _make_asgi_scope()

        with pytest.raises(ValueError, match="asgi test crash"):
            asyncio.run(mw(scope, _noop_receive, _noop_send))

    def test_includes_request_context(self, tmp_path: Path):
        mw = LLMDebugASGIMiddleware(_failing_asgi_app, out_dir=str(tmp_path), pii_safe=False)
        scope = _make_asgi_scope(method="PUT", path="/update", query_string=b"v=3")

        with pytest.raises(ValueError):
            asyncio.run(mw(scope, _noop_receive, _noop_send))

        snapshot = read_latest_snapshot(tmp_path)
        req = snapshot["request"]
        assert req["method"] == "PUT"
        assert req["path"] == "/update"
        assert req["query"] == "v=3"

    def test_captures_websocket_exception(self, tmp_path: Path):
        """Websocket scope errors should also be captured."""
        mw = LLMDebugASGIMiddleware(_failing_asgi_app, out_dir=str(tmp_path), pii_safe=False)
        scope = _make_asgi_scope(type="websocket", method="GET", path="/ws")

        with pytest.raises(ValueError, match="asgi test crash"):
            asyncio.run(mw(scope, _noop_receive, _noop_send))

        snapshot = read_latest_snapshot(tmp_path)
        assert snapshot["name"] == "asgi_error"
        assert snapshot["request"]["type"] == "websocket"

    def test_missing_scope_type_passes_through(self):
        """A scope without 'type' should pass through (not KeyError)."""
        calls: list[dict[str, Any]] = []

        async def tracking_app(scope, receive, send):
            calls.append(scope)

        mw = LLMDebugASGIMiddleware(tracking_app)
        scope: dict[str, Any] = {"path": "/no-type"}

        asyncio.run(mw(scope, _noop_receive, _noop_send))
        # App should be called — missing type treated as non-http
        assert len(calls) == 1
        assert calls[0]["path"] == "/no-type"

    def test_skips_non_http_scope(self):
        """Lifespan events should pass through without interception."""
        calls: list[str] = []

        async def lifespan_app(scope, receive, send):
            calls.append(scope["type"])

        mw = LLMDebugASGIMiddleware(lifespan_app)
        scope = {"type": "lifespan"}

        asyncio.run(mw(scope, _noop_receive, _noop_send))
        assert calls == ["lifespan"]


# ---------------------------------------------------------------------------
# Context extraction
# ---------------------------------------------------------------------------


class TestExtractContext:
    def test_redact_query_string_applies_pii_patterns(self):
        redacted = _redact_query_string("email=alice@example.com&token=abc123")
        assert "[REDACTED]" in redacted
        assert "alice@example.com" not in redacted

    def test_wsgi_context_extraction(self):
        environ = _make_wsgi_environ(
            REQUEST_METHOD="DELETE",
            PATH_INFO="/resource/42",
            QUERY_STRING="force=true",
            CONTENT_TYPE="application/json",
            SERVER_NAME="api.example.com",
        )
        ctx = _extract_wsgi_context(environ)
        assert ctx["hook"] == "wsgi"
        assert ctx["request"]["method"] == "DELETE"
        assert ctx["request"]["path"] == "/resource/42"
        assert ctx["request"]["query"] == "force=true"
        assert ctx["request"]["content_type"] == "application/json"
        assert ctx["request"]["server"] == "api.example.com"

    def test_asgi_context_extraction(self):
        scope = _make_asgi_scope(
            type="websocket",
            method="GET",
            path="/ws",
            query_string=b"token=abc",
            server=("ws.example.com", 443),
        )
        ctx = _extract_asgi_context(scope)
        assert ctx["hook"] == "asgi"
        assert ctx["request"]["type"] == "websocket"
        assert ctx["request"]["method"] == "GET"
        assert ctx["request"]["path"] == "/ws"
        assert ctx["request"]["query"] == "token=abc"
        assert ctx["request"]["server"] == "ws.example.com:443"

    def test_wsgi_context_redacts_sensitive_path_and_query(self):
        environ = _make_wsgi_environ(
            PATH_INFO="/users/alice@example.com",
            QUERY_STRING="auth_token=topsecret",
        )
        ctx = _extract_wsgi_context(environ)
        path = ctx["request"]["path"]
        query = ctx["request"]["query"]
        assert "[REDACTED]" in path
        assert "alice@example.com" not in path
        assert "[REDACTED]" in query
        assert "auth_token=topsecret" not in query

    def test_asgi_context_redacts_sensitive_path_and_query(self):
        scope = _make_asgi_scope(
            path="/download/api_key=abcdef",
            query_string=b"auth_token=topsecret",
        )
        ctx = _extract_asgi_context(scope)
        path = ctx["request"]["path"]
        query = ctx["request"]["query"]
        assert "[REDACTED]" in path
        assert "api_key=abcdef" not in path
        assert "[REDACTED]" in query
        assert "auth_token=topsecret" not in query

    def test_non_sensitive_paths_remain_unchanged(self):
        wsgi_ctx = _extract_wsgi_context(_make_wsgi_environ(PATH_INFO="/health"))
        assert wsgi_ctx["request"]["path"] == "/health"

        asgi_ctx = _extract_asgi_context(_make_asgi_scope(path="/health"))
        assert asgi_ctx["request"]["path"] == "/health"

    def test_missing_keys_handled(self):
        """Empty dicts should produce safe defaults, not KeyErrors."""
        wsgi_ctx = _extract_wsgi_context({})
        assert wsgi_ctx["request"]["method"] == "?"
        assert wsgi_ctx["request"]["path"] == "?"

        asgi_ctx = _extract_asgi_context({})
        assert asgi_ctx["request"]["path"] == "?"
        assert asgi_ctx["request"]["server"] == ""

    def test_asgi_context_coerces_non_string_and_bytes_path(self):
        bytes_ctx = _extract_asgi_context(
            _make_asgi_scope(path=b"/bytes/\xff", query_string=b"q=1", server=("host", 8080))
        )
        assert bytes_ctx["request"]["path"].startswith("/bytes/")
        assert bytes_ctx["request"]["query"] == "q=1"
        assert bytes_ctx["request"]["server"] == "host:8080"

        coerced_ctx = _extract_asgi_context(
            {"path": 123, "query_string": "x=1", "server": ("only",)}
        )
        assert coerced_ctx["request"]["path"] == "123"
        assert coerced_ctx["request"]["query"] == "x=1"
        assert coerced_ctx["request"]["server"] == ""


# ---------------------------------------------------------------------------
# Coverage-gap tests: middleware.py branches
# ---------------------------------------------------------------------------


def test_wsgi_middleware_exception_no_traceback(monkeypatch):
    """Branch [127,128]: exc.__traceback__ is None → early return."""
    from llmdebug.middleware import LLMDebugWSGIMiddleware

    captured_calls: list[int] = []
    monkeypatch.setattr(
        "llmdebug.middleware._capture_safely",
        lambda *args, **kwargs: captured_calls.append(1),
    )

    def failing_app(environ, start_response):
        raise RuntimeError("no tb")

    mw = LLMDebugWSGIMiddleware(failing_app)

    # Create an exception with no traceback
    exc = RuntimeError("no traceback")
    exc.__traceback__ = None

    # Call the internal capture method directly
    mw._capture_wsgi_exception(exc, {"REQUEST_METHOD": "GET", "PATH_INFO": "/"})
    # _capture_safely should NOT have been called since tb is None
    assert captured_calls == []
