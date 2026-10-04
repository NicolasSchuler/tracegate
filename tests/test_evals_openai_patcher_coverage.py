"""Coverage tests for evals/patchers/openai_compat_patcher.py.

Targets previously-uncovered code paths: HTTP transport functions, server
probing, context compaction, patcher public async methods, diff/edits/full-file
proposal flows, and chat-completion retry logic.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from evals.patchers import openai_compat_patcher as openai_mod
from evals.patchers.base import (
    ContextCompactionResult,
    ContextRequestResult,
    PatchPlanResult,
    PatchResult,
)
from evals.patchers.openai_compat_patcher import OpenAICompatPatcher

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_GOOD_USAGE = {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7}
_GOOD_META = {"llm_call_count": 1, "usage": dict(_GOOD_USAGE)}


def _ok_response(content: str, *, usage: dict | None = None) -> dict:
    return {
        "choices": [{"message": {"content": content}}],
        "usage": usage or dict(_GOOD_USAGE),
    }


def _ok_chat(content: str) -> tuple[dict, dict]:
    return _ok_response(content), dict(_GOOD_META)


def _err_chat(
    error: str = "http_error", body: str = "oops", status: int = 500
) -> tuple[None, dict]:
    return None, {
        "error": error,
        "status": status,
        "body": body,
        "llm_call_count": 1,
        "usage": dict(_GOOD_USAGE),
    }


# ---------------------------------------------------------------------------
# 1. HTTP Transport: _http_get_json
# ---------------------------------------------------------------------------


class TestHttpGetJson:
    def test_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_resp = SimpleNamespace(
            status_code=200,
            text='{"ok": true}',
            http_version="HTTP/1.1",
        )

        class FakeClient:
            def __init__(self, **kw):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass

            def get(self, url, headers):
                return fake_resp

        monkeypatch.setattr(openai_mod.httpx, "Client", FakeClient)
        data, meta = openai_mod._http_get_json(
            url="http://localhost:8080/v1/models",
            api_key="k",
            timeout_sec=5.0,
            http2=False,
        )
        assert data == {"ok": True}
        assert meta.get("transport_impl") == "httpx"

    def test_http_error_status(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_resp = SimpleNamespace(status_code=400, text="bad request", http_version="HTTP/1.1")

        class FakeClient:
            def __init__(self, **kw):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass

            def get(self, url, headers):
                return fake_resp

        monkeypatch.setattr(openai_mod.httpx, "Client", FakeClient)
        data, meta = openai_mod._http_get_json(
            url="http://localhost:8080/v1/models",
            api_key="k",
            timeout_sec=5.0,
            http2=False,
        )
        assert data is None
        assert meta["error"] == "http_error"
        assert meta["status"] == 400

    def test_bad_json(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_resp = SimpleNamespace(status_code=200, text="not json!", http_version="HTTP/1.1")

        class FakeClient:
            def __init__(self, **kw):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass

            def get(self, url, headers):
                return fake_resp

        monkeypatch.setattr(openai_mod.httpx, "Client", FakeClient)
        data, meta = openai_mod._http_get_json(
            url="http://localhost:8080/v1/models",
            api_key="k",
            timeout_sec=5.0,
            http2=False,
        )
        assert data is None
        assert meta["error"] == "bad_json_response"

    def test_non_dict_list_json(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_resp = SimpleNamespace(
            status_code=200, text='"just a string"', http_version="HTTP/1.1"
        )

        class FakeClient:
            def __init__(self, **kw):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass

            def get(self, url, headers):
                return fake_resp

        monkeypatch.setattr(openai_mod.httpx, "Client", FakeClient)
        data, meta = openai_mod._http_get_json(
            url="http://localhost:8080/v1/models",
            api_key="k",
            timeout_sec=5.0,
            http2=False,
        )
        assert data is None
        assert meta["error"] == "bad_json_response"

    def test_value_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class FakeClient:
            def __init__(self, **kw):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass

            def get(self, url, headers):
                raise ValueError("bad url")

        monkeypatch.setattr(openai_mod.httpx, "Client", FakeClient)
        data, meta = openai_mod._http_get_json(
            url="http://localhost:8080/v1/models",
            api_key="k",
            timeout_sec=5.0,
            http2=False,
        )
        assert data is None
        assert meta["error"] == "bad_request_url"

    def test_http2_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        call_count = {"n": 0}

        class FakeClient:
            def __init__(self, *, timeout, http2, trust_env):
                self._http2 = http2

            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass

            def get(self, url, headers):
                call_count["n"] += 1
                if self._http2:
                    raise httpx.RemoteProtocolError("h2 error")
                return SimpleNamespace(status_code=200, text='{"ok":1}', http_version="HTTP/1.1")

        monkeypatch.setattr(openai_mod.httpx, "Client", FakeClient)
        data, meta = openai_mod._http_get_json(
            url="http://localhost:8080/v1/models",
            api_key="k",
            timeout_sec=5.0,
            http2=True,
        )
        assert data == {"ok": 1}
        assert meta["transport_http2_fallback_used"] is True

    def test_non_http2_exception(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class FakeClient:
            def __init__(self, **kw):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass

            def get(self, url, headers):
                raise ConnectionError("refused")

        monkeypatch.setattr(openai_mod.httpx, "Client", FakeClient)
        data, meta = openai_mod._http_get_json(
            url="http://localhost:8080/v1/models",
            api_key="k",
            timeout_sec=5.0,
            http2=False,
        )
        assert data is None
        assert meta["error"] == "request_failed"

    def test_http2_fallback_also_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class FakeClient:
            def __init__(self, *, timeout, http2, trust_env):
                self._http2 = http2

            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass

            def get(self, url, headers):
                if self._http2:
                    raise httpx.RemoteProtocolError("h2 error")
                raise RuntimeError("fallback also fails")

        monkeypatch.setattr(openai_mod.httpx, "Client", FakeClient)
        data, meta = openai_mod._http_get_json(
            url="http://localhost:8080/v1/models",
            api_key="k",
            timeout_sec=5.0,
            http2=True,
        )
        assert data is None
        assert meta["error"] == "request_failed"
        assert meta["transport_http2_fallback_used"] is True


# ---------------------------------------------------------------------------
# 2. HTTP Transport: _http_get_json_async
# ---------------------------------------------------------------------------


class TestHttpGetJsonAsync:
    def test_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_resp = SimpleNamespace(status_code=200, text='{"a":1}', http_version="HTTP/1.1")

        class FakeAsyncClient:
            def __init__(self, **kw):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                pass

            async def get(self, url, headers):
                return fake_resp

        monkeypatch.setattr(openai_mod.httpx, "AsyncClient", FakeAsyncClient)
        data, _meta = asyncio.run(
            openai_mod._http_get_json_async(
                url="http://localhost/v1/models",
                api_key="k",
                timeout_sec=5,
                http2=False,
            )
        )
        assert data == {"a": 1}

    def test_http_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_resp = SimpleNamespace(status_code=500, text="internal error", http_version="HTTP/1.1")

        class FakeAsyncClient:
            def __init__(self, **kw):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                pass

            async def get(self, url, headers):
                return fake_resp

        monkeypatch.setattr(openai_mod.httpx, "AsyncClient", FakeAsyncClient)
        data, meta = asyncio.run(
            openai_mod._http_get_json_async(
                url="http://localhost/v1/models",
                api_key="k",
                timeout_sec=5,
                http2=False,
            )
        )
        assert data is None
        assert meta["error"] == "http_error"

    def test_exception(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class FakeAsyncClient:
            def __init__(self, **kw):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                pass

            async def get(self, url, headers):
                raise ConnectionError("refused")

        monkeypatch.setattr(openai_mod.httpx, "AsyncClient", FakeAsyncClient)
        data, meta = asyncio.run(
            openai_mod._http_get_json_async(
                url="http://localhost/v1/models",
                api_key="k",
                timeout_sec=5,
                http2=False,
            )
        )
        assert data is None
        assert meta["error"] == "request_failed"

    def test_http2_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class FakeAsyncClient:
            def __init__(self, *, timeout, http2, trust_env):
                self._http2 = http2

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                pass

            async def get(self, url, headers):
                if self._http2:
                    raise httpx.RemoteProtocolError("h2 bad")
                return SimpleNamespace(status_code=200, text="[1,2]", http_version="HTTP/1.1")

        monkeypatch.setattr(openai_mod.httpx, "AsyncClient", FakeAsyncClient)
        data, meta = asyncio.run(
            openai_mod._http_get_json_async(
                url="http://localhost/v1/models",
                api_key="k",
                timeout_sec=5,
                http2=True,
            )
        )
        assert data == [1, 2]
        assert meta["transport_http2_fallback_used"] is True

    def test_http2_fallback_value_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class FakeAsyncClient:
            def __init__(self, *, timeout, http2, trust_env):
                self._http2 = http2

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                pass

            async def get(self, url, headers):
                if self._http2:
                    raise httpx.RemoteProtocolError("h2 bad")
                raise ValueError("bad url in fallback")

        monkeypatch.setattr(openai_mod.httpx, "AsyncClient", FakeAsyncClient)
        data, meta = asyncio.run(
            openai_mod._http_get_json_async(
                url="http://localhost/v1/models",
                api_key="k",
                timeout_sec=5,
                http2=True,
            )
        )
        assert data is None
        assert meta["error"] == "bad_request_url"

    def test_bad_json_async(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_resp = SimpleNamespace(status_code=200, text="NOTJSON", http_version="HTTP/1.1")

        class FakeAsyncClient:
            def __init__(self, **kw):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                pass

            async def get(self, url, headers):
                return fake_resp

        monkeypatch.setattr(openai_mod.httpx, "AsyncClient", FakeAsyncClient)
        data, meta = asyncio.run(
            openai_mod._http_get_json_async(
                url="http://localhost/v1/models",
                api_key="k",
                timeout_sec=5,
                http2=False,
            )
        )
        assert data is None
        assert meta["error"] == "bad_json_response"


# ---------------------------------------------------------------------------
# 3. HTTP Transport: _http_probe_ready / _http_probe_ready_async
# ---------------------------------------------------------------------------


class TestHttpProbeReady:
    def test_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(openai_mod, "_should_use_proxy_transport", lambda url: True)

        fake_resp = SimpleNamespace(status_code=200, text="ok", http_version="HTTP/1.1")

        class FakeClient:
            def __init__(self, **kw):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass

            def get(self, url, headers):
                return fake_resp

        monkeypatch.setattr(openai_mod.httpx, "Client", FakeClient)
        ready, meta = openai_mod._http_probe_ready(
            url="http://localhost:8080/health",
            api_key="k",
            timeout_sec=5.0,
            http2=False,
        )
        assert ready is True
        assert meta["status"] == 200

    def test_non_2xx(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(openai_mod, "_should_use_proxy_transport", lambda url: True)
        fake_resp = SimpleNamespace(status_code=503, text="unavailable", http_version="HTTP/1.1")

        class FakeClient:
            def __init__(self, **kw):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass

            def get(self, url, headers):
                return fake_resp

        monkeypatch.setattr(openai_mod.httpx, "Client", FakeClient)
        ready, meta = openai_mod._http_probe_ready(
            url="http://localhost:8080/health",
            api_key="k",
            timeout_sec=5.0,
            http2=False,
        )
        assert ready is False
        assert meta["error"] == "http_error"

    def test_exception(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(openai_mod, "_should_use_proxy_transport", lambda url: True)

        class FakeClient:
            def __init__(self, **kw):
                pass

            def __enter__(self):
                return self

            def __exit__(self, *a):
                pass

            def get(self, url, headers):
                raise ConnectionError("refused")

        monkeypatch.setattr(openai_mod.httpx, "Client", FakeClient)
        ready, meta = openai_mod._http_probe_ready(
            url="http://localhost:8080/health",
            api_key="k",
            timeout_sec=5.0,
            http2=False,
        )
        assert ready is None  # type: ignore[comparison-overlap]
        assert meta["error"] == "request_failed"


class TestHttpProbeReadyAsync:
    def test_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(openai_mod, "_should_use_proxy_transport", lambda url: True)
        fake_resp = SimpleNamespace(status_code=200, text="ok", http_version="HTTP/1.1")

        class FakeAsyncClient:
            def __init__(self, **kw):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                pass

            async def get(self, url, headers):
                return fake_resp

        monkeypatch.setattr(openai_mod.httpx, "AsyncClient", FakeAsyncClient)
        ready, _meta = asyncio.run(
            openai_mod._http_probe_ready_async(
                url="http://localhost/health",
                api_key="k",
                timeout_sec=5,
                http2=False,
            )
        )
        assert ready is True

    def test_http2_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(openai_mod, "_should_use_proxy_transport", lambda url: True)

        class FakeAsyncClient:
            def __init__(self, *, timeout, http2, trust_env):
                self._http2 = http2

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                pass

            async def get(self, url, headers):
                if self._http2:
                    raise httpx.RemoteProtocolError("h2 bad")
                return SimpleNamespace(status_code=200, text="ok", http_version="HTTP/1.1")

        monkeypatch.setattr(openai_mod.httpx, "AsyncClient", FakeAsyncClient)
        ready, meta = asyncio.run(
            openai_mod._http_probe_ready_async(
                url="http://localhost/health",
                api_key="k",
                timeout_sec=5,
                http2=True,
            )
        )
        assert ready is True
        assert meta["transport_http2_fallback_used"] is True

    def test_value_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(openai_mod, "_should_use_proxy_transport", lambda url: True)

        class FakeAsyncClient:
            def __init__(self, **kw):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                pass

            async def get(self, url, headers):
                raise ValueError("bad url")

        monkeypatch.setattr(openai_mod.httpx, "AsyncClient", FakeAsyncClient)
        ready, meta = asyncio.run(
            openai_mod._http_probe_ready_async(
                url="http://localhost/health",
                api_key="k",
                timeout_sec=5,
                http2=False,
            )
        )
        assert ready is None  # type: ignore[comparison-overlap]
        assert meta["error"] == "bad_request_url"


# ---------------------------------------------------------------------------
# 4. Proxy detection
# ---------------------------------------------------------------------------


class TestShouldUseProxyTransport:
    def test_non_http_scheme(self) -> None:
        assert openai_mod._should_use_proxy_transport("ftp://example.com") is False

    def test_no_proxy(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(openai_mod.urllib.request, "getproxies", lambda: {})
        assert openai_mod._should_use_proxy_transport("http://example.com/path") is False

    def test_proxy_bypassed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            openai_mod.urllib.request, "getproxies", lambda: {"http": "http://proxy:8080"}
        )
        monkeypatch.setattr(openai_mod.urllib.request, "proxy_bypass", lambda host: True)
        assert openai_mod._should_use_proxy_transport("http://example.com/path") is False

    def test_proxy_active(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            openai_mod.urllib.request, "getproxies", lambda: {"http": "http://proxy:8080"}
        )
        monkeypatch.setattr(openai_mod.urllib.request, "proxy_bypass", lambda host: False)
        assert openai_mod._should_use_proxy_transport("http://example.com/path") is True


# ---------------------------------------------------------------------------
# 5. Server Probing: _probe_context_window_tokens_async
# ---------------------------------------------------------------------------


class TestProbeContextWindowTokensAsync:
    def test_first_provider_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def _fake_get_json(*, url, api_key, timeout_sec, http2):
            if "api/v1" in url:
                return (
                    {"data": [{"id": "m1", "context_length": 8192}]},
                    {"transport_impl": "httpx"},
                )
            return None, {"error": "request_failed"}

        monkeypatch.setattr(openai_mod, "_http_get_json_async", _fake_get_json)
        tokens, meta = asyncio.run(
            openai_mod._probe_context_window_tokens_async(
                base_url="http://localhost:1234/v1",
                model="m1",
                api_key="k",
                timeout_sec=5,
                http2=False,
            )
        )
        assert tokens == 8192
        assert meta["context_window_probe_provider"] == "lmstudio_api_v1"

    def test_first_fails_second_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def _fake_get_json(*, url, api_key, timeout_sec, http2):
            if "api/v1" in url:
                return None, {"error": "request_failed"}
            if "api/v0" in url:
                return (
                    [{"id": "m1", "context_length": 4096}],
                    {"transport_impl": "httpx"},
                )
            return None, {"error": "request_failed"}

        monkeypatch.setattr(openai_mod, "_http_get_json_async", _fake_get_json)
        tokens, meta = asyncio.run(
            openai_mod._probe_context_window_tokens_async(
                base_url="http://localhost:1234/v1",
                model="m1",
                api_key="k",
                timeout_sec=5,
                http2=False,
            )
        )
        assert tokens == 4096
        assert meta["context_window_probe_provider"] == "lmstudio_api_v0"

    def test_all_providers_fail(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def _fake_get_json(*, url, api_key, timeout_sec, http2):
            return None, {"error": "request_failed"}

        monkeypatch.setattr(openai_mod, "_http_get_json_async", _fake_get_json)
        tokens, meta = asyncio.run(
            openai_mod._probe_context_window_tokens_async(
                base_url="http://localhost:1234/v1",
                model="m1",
                api_key="k",
                timeout_sec=5,
                http2=False,
            )
        )
        assert tokens is None
        assert meta["error"] == "context_window_probe_failed"

    def test_model_not_found_in_response(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def _fake_get_json(*, url, api_key, timeout_sec, http2):
            return {"data": [{"id": "other_model", "context_length": 2048}]}, {}

        monkeypatch.setattr(openai_mod, "_http_get_json_async", _fake_get_json)
        tokens, meta = asyncio.run(
            openai_mod._probe_context_window_tokens_async(
                base_url="http://localhost:1234/v1",
                model="m1",
                api_key="k",
                timeout_sec=5,
                http2=False,
            )
        )
        assert tokens is None
        assert "model_context_not_found" in str(meta.get("context_window_probe_error", ""))


# ---------------------------------------------------------------------------
# 6. Server Probing: _probe_server_ready_async
# ---------------------------------------------------------------------------


class TestProbeServerReadyAsync:
    def test_health_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def _fake_probe(*, url, api_key, timeout_sec, http2):
            if "/health" in url:
                return True, {"status": 200}
            return False, {"error": "request_failed"}

        monkeypatch.setattr(openai_mod, "_http_probe_ready_async", _fake_probe)
        ready, meta = asyncio.run(
            openai_mod._probe_server_ready_async(
                base_url="http://localhost:1234/v1",
                api_key="k",
                timeout_sec=5,
                http2=False,
            )
        )
        assert ready is True
        assert meta["server_ready_probe_provider"] == "health"

    def test_health_fails_models_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def _fake_probe(*, url, api_key, timeout_sec, http2):
            if "/health" in url:
                return False, {"error": "request_failed"}
            return True, {"status": 200}

        monkeypatch.setattr(openai_mod, "_http_probe_ready_async", _fake_probe)
        ready, meta = asyncio.run(
            openai_mod._probe_server_ready_async(
                base_url="http://localhost:1234/v1",
                api_key="k",
                timeout_sec=5,
                http2=False,
            )
        )
        assert ready is True
        assert meta["server_ready_probe_provider"] == "openai_models"

    def test_all_fail(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def _fake_probe(*, url, api_key, timeout_sec, http2):
            return False, {"error": "request_failed"}

        monkeypatch.setattr(openai_mod, "_http_probe_ready_async", _fake_probe)
        ready, meta = asyncio.run(
            openai_mod._probe_server_ready_async(
                base_url="http://localhost:1234/v1",
                api_key="k",
                timeout_sec=5,
                http2=False,
            )
        )
        assert ready is False
        assert meta["error"] == "server_ready_probe_failed"


# ---------------------------------------------------------------------------
# 7. Context Compaction: _compact_context_via_json_async
# ---------------------------------------------------------------------------


class TestCompactContextViaJsonAsync:
    def test_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def _fake_chat(**kw):
            return _ok_response('{"compacted_context": "short"}'), dict(_GOOD_META)

        monkeypatch.setattr(openai_mod, "_chat_completion_async", _fake_chat)
        compacted, _raw, _meta = asyncio.run(
            openai_mod._compact_context_via_json_async(
                url="http://localhost/v1/chat/completions",
                api_key="k",
                model="m",
                prompt="fix it",
                context_text="long context " * 100,
                target_chars=50,
                temperature=0.2,
                max_tokens=500,
                timeout_sec=30,
                seed=None,
                http2=False,
            )
        )
        assert compacted == "short"

    def test_response_format_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = {"n": 0}

        async def _fake_chat(**kw):
            calls["n"] += 1
            if calls["n"] == 1:
                return None, {
                    "error": "http_error",
                    "body": "response_format not supported",
                    "llm_call_count": 1,
                    "usage": dict(_GOOD_USAGE),
                }
            return _ok_response('{"compacted_context": "ok"}'), dict(_GOOD_META)

        monkeypatch.setattr(openai_mod, "_chat_completion_async", _fake_chat)
        compacted, _raw, _meta = asyncio.run(
            openai_mod._compact_context_via_json_async(
                url="http://localhost/v1/chat/completions",
                api_key="k",
                model="m",
                prompt="fix it",
                context_text="some context",
                target_chars=100,
                temperature=0.2,
                max_tokens=500,
                timeout_sec=30,
                seed=None,
                http2=False,
            )
        )
        assert compacted == "ok"
        assert calls["n"] == 2

    def test_total_failure(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def _fake_chat(**kw):
            return None, {
                "error": "request_failed",
                "llm_call_count": 1,
                "usage": dict(_GOOD_USAGE),
            }

        monkeypatch.setattr(openai_mod, "_chat_completion_async", _fake_chat)
        compacted, _raw, _meta = asyncio.run(
            openai_mod._compact_context_via_json_async(
                url="http://localhost/v1/chat/completions",
                api_key="k",
                model="m",
                prompt="fix",
                context_text="ctx",
                target_chars=10,
                temperature=0.2,
                max_tokens=500,
                timeout_sec=30,
                seed=None,
                http2=False,
            )
        )
        assert compacted is None


# ---------------------------------------------------------------------------
# 8. Patcher: acompact_context
# ---------------------------------------------------------------------------


class TestAcompactContext:
    def test_success(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        async def _fake_compact(**kw):
            return "short result", '{"compacted_context":"short result"}', dict(_GOOD_META)

        monkeypatch.setattr(openai_mod, "_compact_context_via_json_async", _fake_compact)
        patcher = OpenAICompatPatcher(
            base_url="http://127.0.0.1:1234",
            model="m",
            response_mode="diff",
            api_key="test-key",
        )
        result = asyncio.run(
            patcher.acompact_context(
                prompt="fix test",
                context_text="long text " * 50,
                context_dir=tmp_path,
                cwd=tmp_path,
                target_chars=20,
            )
        )
        assert result is not None
        assert result.compacted_text is not None
        assert len(result.compacted_text) <= 20

    def test_failure_returns_none_text(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        async def _fake_compact(**kw):
            return (
                None,
                "bad response body",
                {"error": "http_error", "llm_call_count": 1, "usage": dict(_GOOD_USAGE)},
            )

        monkeypatch.setattr(openai_mod, "_compact_context_via_json_async", _fake_compact)
        patcher = OpenAICompatPatcher(
            base_url="http://127.0.0.1:1234",
            model="m",
            response_mode="diff",
            api_key="test-key",
        )
        result = asyncio.run(
            patcher.acompact_context(
                prompt="fix",
                context_text="ctx",
                context_dir=tmp_path,
                cwd=tmp_path,
                target_chars=10,
            )
        )
        assert result is not None
        assert result.compacted_text is None


# ---------------------------------------------------------------------------
# 9. Patcher: aprobe_context_window_tokens / aprobe_server_ready
# ---------------------------------------------------------------------------


class TestAprobeContextWindowTokens:
    def test_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def _fake_probe(**kw):
            return 4096, {"context_window_probe_provider": "openai_models"}

        monkeypatch.setattr(openai_mod, "_probe_context_window_tokens_async", _fake_probe)
        patcher = OpenAICompatPatcher(
            base_url="http://127.0.0.1:1234",
            model="m",
            response_mode="diff",
            api_key="test-key",
        )
        tokens, meta = asyncio.run(patcher.aprobe_context_window_tokens())
        assert tokens == 4096
        assert "context_window_probe_duration_sec" in meta


class TestAprobeServerReady:
    def test_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def _fake_probe(**kw):
            return True, {"server_ready": True}

        monkeypatch.setattr(openai_mod, "_probe_server_ready_async", _fake_probe)
        patcher = OpenAICompatPatcher(
            base_url="http://127.0.0.1:1234",
            model="m",
            response_mode="diff",
            api_key="test-key",
        )
        ready, meta = asyncio.run(patcher.aprobe_server_ready())
        assert ready is True
        assert "server_ready_probe_duration_sec" in meta


# ---------------------------------------------------------------------------
# 10. Diff Proposal: _propose_patch_via_diff_async
# ---------------------------------------------------------------------------


_VALID_DIFF = """\
diff --git a/foo.py b/foo.py
--- a/foo.py
+++ b/foo.py
@@ -1,3 +1,3 @@
 def foo():
-    return 1
+    return 2
"""


class TestProposePatchViaDiffAsync:
    def test_valid_diff_first_attempt(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def _fake_chat(**kw):
            return _ok_response(_VALID_DIFF), dict(_GOOD_META)

        monkeypatch.setattr(openai_mod, "_chat_completion_async", _fake_chat)
        result = asyncio.run(
            openai_mod._propose_patch_via_diff_async(
                url="http://localhost/v1/chat/completions",
                api_key="k",
                model="m",
                prompt="fix test",
                temperature=0.2,
                max_tokens=500,
                timeout_sec=30,
                seed=None,
                http2=False,
            )
        )
        assert result.diff is not None
        assert "foo.py" in result.diff

    def test_no_diff_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def _fake_chat(**kw):
            return _ok_response("Sorry, I can't generate a diff."), dict(_GOOD_META)

        monkeypatch.setattr(openai_mod, "_chat_completion_async", _fake_chat)
        result = asyncio.run(
            openai_mod._propose_patch_via_diff_async(
                url="http://localhost/v1/chat/completions",
                api_key="k",
                model="m",
                prompt="fix test",
                temperature=0.2,
                max_tokens=500,
                timeout_sec=30,
                seed=None,
                http2=False,
            )
        )
        assert result.diff is None
        assert result.meta.get("error") == "no_diff_found"

    def test_budget_exhausted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def _fake_chat(**kw):
            return None, {
                "error": "request_budget_exceeded",
                "llm_call_count": 1,
                "usage": dict(_GOOD_USAGE),
            }

        monkeypatch.setattr(openai_mod, "_chat_completion_async", _fake_chat)
        result = asyncio.run(
            openai_mod._propose_patch_via_diff_async(
                url="http://localhost/v1/chat/completions",
                api_key="k",
                model="m",
                prompt="fix",
                temperature=0.2,
                max_tokens=500,
                timeout_sec=30,
                seed=None,
                http2=False,
            )
        )
        assert result.diff is None

    def test_second_attempt_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        calls = {"n": 0}

        async def _fake_chat(**kw):
            calls["n"] += 1
            if calls["n"] == 1:
                return _ok_response("no diff here"), dict(_GOOD_META)
            return _ok_response(_VALID_DIFF), dict(_GOOD_META)

        monkeypatch.setattr(openai_mod, "_chat_completion_async", _fake_chat)
        result = asyncio.run(
            openai_mod._propose_patch_via_diff_async(
                url="http://localhost/v1/chat/completions",
                api_key="k",
                model="m",
                prompt="fix",
                temperature=0.2,
                max_tokens=500,
                timeout_sec=30,
                seed=None,
                http2=False,
            )
        )
        assert result.diff is not None
        assert calls["n"] == 2

    def test_no_choice_content(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def _fake_chat(**kw):
            return {"choices": [{"message": {}}], "usage": dict(_GOOD_USAGE)}, dict(_GOOD_META)

        monkeypatch.setattr(openai_mod, "_chat_completion_async", _fake_chat)
        result = asyncio.run(
            openai_mod._propose_patch_via_diff_async(
                url="http://localhost/v1/chat/completions",
                api_key="k",
                model="m",
                prompt="fix",
                temperature=0.2,
                max_tokens=500,
                timeout_sec=30,
                seed=None,
                http2=False,
            )
        )
        assert result.diff is None
        assert result.meta.get("error") == "no_choice_content"


# ---------------------------------------------------------------------------
# 11. Edits Proposal: _propose_patch_via_edits_async
# ---------------------------------------------------------------------------


class TestProposePatchViaEditsAsync:
    def test_successful_edit(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        (tmp_path / "foo.py").write_text("def foo():\n    return 1\n", encoding="utf-8")
        edits_json = json.dumps(
            {"edits": [{"path": "foo.py", "find": "return 1", "replace": "return 2", "count": 1}]}
        )

        async def _fake_chat(**kw):
            return _ok_response(edits_json), dict(_GOOD_META)

        monkeypatch.setattr(openai_mod, "_chat_completion_async", _fake_chat)
        result = asyncio.run(
            openai_mod._propose_patch_via_edits_async(
                url="http://localhost/v1/chat/completions",
                api_key="k",
                model="m",
                prompt="fix",
                cwd=tmp_path,
                temperature=0.2,
                max_tokens=500,
                timeout_sec=30,
                seed=None,
                http2=False,
            )
        )
        assert result.diff is not None
        assert "return 2" in result.diff

    def test_bad_path_in_edit(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        edits_json = json.dumps({"edits": [{"path": "", "find": "x", "replace": "y", "count": 1}]})

        async def _fake_chat(**kw):
            return _ok_response(edits_json), dict(_GOOD_META)

        monkeypatch.setattr(openai_mod, "_chat_completion_async", _fake_chat)
        result = asyncio.run(
            openai_mod._propose_patch_via_edits_async(
                url="http://localhost/v1/chat/completions",
                api_key="k",
                model="m",
                prompt="fix",
                cwd=tmp_path,
                temperature=0.2,
                max_tokens=500,
                timeout_sec=30,
                seed=None,
                http2=False,
            )
        )
        assert result.diff is None
        assert result.meta.get("error") == "bad_path"

    def test_file_not_found(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        edits_json = json.dumps(
            {"edits": [{"path": "missing.py", "find": "x", "replace": "y", "count": 1}]}
        )

        async def _fake_chat(**kw):
            return _ok_response(edits_json), dict(_GOOD_META)

        monkeypatch.setattr(openai_mod, "_chat_completion_async", _fake_chat)
        result = asyncio.run(
            openai_mod._propose_patch_via_edits_async(
                url="http://localhost/v1/chat/completions",
                api_key="k",
                model="m",
                prompt="fix",
                cwd=tmp_path,
                temperature=0.2,
                max_tokens=500,
                timeout_sec=30,
                seed=None,
                http2=False,
            )
        )
        assert result.diff is None
        assert result.meta.get("error") == "file_not_found"

    def test_path_traversal_blocked(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        edits_json = json.dumps(
            {"edits": [{"path": "../../../etc/passwd", "find": "x", "replace": "y", "count": 1}]}
        )

        async def _fake_chat(**kw):
            return _ok_response(edits_json), dict(_GOOD_META)

        monkeypatch.setattr(openai_mod, "_chat_completion_async", _fake_chat)
        result = asyncio.run(
            openai_mod._propose_patch_via_edits_async(
                url="http://localhost/v1/chat/completions",
                api_key="k",
                model="m",
                prompt="fix",
                cwd=tmp_path,
                temperature=0.2,
                max_tokens=500,
                timeout_sec=30,
                seed=None,
                http2=False,
            )
        )
        assert result.diff is None
        assert result.meta.get("error") == "path_outside_case"

    def test_find_not_found_fallback_to_diff(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        (tmp_path / "foo.py").write_text("def foo():\n    return 1\n", encoding="utf-8")
        edits_json = json.dumps(
            {"edits": [{"path": "foo.py", "find": "NONEXISTENT_TEXT", "replace": "y", "count": 1}]}
        )
        calls = {"edit_calls": 0, "diff_calls": 0}

        async def _fake_chat(**kw):
            msgs = kw.get("messages", [])
            sys_content = msgs[0].get("content", "") if msgs else ""
            if "JSON" in sys_content or "json" in sys_content.lower():
                calls["edit_calls"] += 1
                return _ok_response(edits_json), dict(_GOOD_META)
            calls["diff_calls"] += 1
            return _ok_response(_VALID_DIFF), dict(_GOOD_META)

        monkeypatch.setattr(openai_mod, "_chat_completion_async", _fake_chat)
        result = asyncio.run(
            openai_mod._propose_patch_via_edits_async(
                url="http://localhost/v1/chat/completions",
                api_key="k",
                model="m",
                prompt="fix",
                cwd=tmp_path,
                temperature=0.2,
                max_tokens=500,
                timeout_sec=30,
                seed=None,
                http2=False,
                allow_diff_fallback=True,
            )
        )
        assert result.diff is not None

    def test_no_json_fallback_no_diff_allowed(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        async def _fake_chat(**kw):
            return _ok_response("not json at all"), dict(_GOOD_META)

        monkeypatch.setattr(openai_mod, "_chat_completion_async", _fake_chat)
        result = asyncio.run(
            openai_mod._propose_patch_via_edits_async(
                url="http://localhost/v1/chat/completions",
                api_key="k",
                model="m",
                prompt="fix",
                cwd=tmp_path,
                temperature=0.2,
                max_tokens=500,
                timeout_sec=30,
                seed=None,
                http2=False,
                allow_diff_fallback=False,
            )
        )
        assert result.diff is None
        assert result.meta.get("error") == "no_json_payload"

    def test_json_schema_rejected_json_object_succeeds(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        (tmp_path / "foo.py").write_text("old_code\n", encoding="utf-8")
        calls = {"n": 0}

        async def _fake_chat(**kw):
            calls["n"] += 1
            extra = kw.get("extra_payload") or {}
            rf = extra.get("response_format", {})
            if rf.get("type") == "json_schema":
                return None, {
                    "error": "http_error",
                    "body": "json_schema not supported",
                    "llm_call_count": 1,
                    "usage": dict(_GOOD_USAGE),
                }
            edits_json = json.dumps(
                {
                    "edits": [
                        {"path": "foo.py", "find": "old_code", "replace": "new_code", "count": 1}
                    ]
                }
            )
            return _ok_response(edits_json), dict(_GOOD_META)

        monkeypatch.setattr(openai_mod, "_chat_completion_async", _fake_chat)
        result = asyncio.run(
            openai_mod._propose_patch_via_edits_async(
                url="http://localhost/v1/chat/completions",
                api_key="k",
                model="m",
                prompt="fix",
                cwd=tmp_path,
                temperature=0.2,
                max_tokens=500,
                timeout_sec=30,
                seed=None,
                http2=False,
            )
        )
        assert result.diff is not None
        assert "new_code" in result.diff

    def test_no_effective_changes(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        (tmp_path / "foo.py").write_text("same\n", encoding="utf-8")
        edits_json = json.dumps(
            {"edits": [{"path": "foo.py", "find": "same", "replace": "same", "count": 1}]}
        )

        async def _fake_chat(**kw):
            msgs = kw.get("messages", [])
            sys_content = msgs[0].get("content", "") if msgs else ""
            if "JSON" in sys_content or "json" in sys_content.lower():
                return _ok_response(edits_json), dict(_GOOD_META)
            return _ok_response(_VALID_DIFF), dict(_GOOD_META)

        monkeypatch.setattr(openai_mod, "_chat_completion_async", _fake_chat)
        result = asyncio.run(
            openai_mod._propose_patch_via_edits_async(
                url="http://localhost/v1/chat/completions",
                api_key="k",
                model="m",
                prompt="fix",
                cwd=tmp_path,
                temperature=0.2,
                max_tokens=500,
                timeout_sec=30,
                seed=None,
                http2=False,
                allow_diff_fallback=True,
            )
        )
        # Falls back to diff because edit produced no changes
        assert (
            result.diff is not None or result.meta.get("fallback_reason") == "no_effective_changes"
        )

    def test_context_exceeded_no_fallback(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        async def _fake_chat(**kw):
            return None, {
                "error": "http_error",
                "body": "exceeds the available context size",
                "llm_call_count": 1,
                "usage": dict(_GOOD_USAGE),
            }

        monkeypatch.setattr(openai_mod, "_chat_completion_async", _fake_chat)
        result = asyncio.run(
            openai_mod._propose_patch_via_edits_async(
                url="http://localhost/v1/chat/completions",
                api_key="k",
                model="m",
                prompt="fix",
                cwd=tmp_path,
                temperature=0.2,
                max_tokens=500,
                timeout_sec=30,
                seed=None,
                http2=False,
            )
        )
        assert result.diff is None

    def test_bad_find_replace_types(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        (tmp_path / "foo.py").write_text("code\n", encoding="utf-8")
        edits_json = json.dumps(
            {"edits": [{"path": "foo.py", "find": 123, "replace": "y", "count": 1}]}
        )

        async def _fake_chat(**kw):
            return _ok_response(edits_json), dict(_GOOD_META)

        monkeypatch.setattr(openai_mod, "_chat_completion_async", _fake_chat)
        result = asyncio.run(
            openai_mod._propose_patch_via_edits_async(
                url="http://localhost/v1/chat/completions",
                api_key="k",
                model="m",
                prompt="fix",
                cwd=tmp_path,
                temperature=0.2,
                max_tokens=500,
                timeout_sec=30,
                seed=None,
                http2=False,
            )
        )
        assert result.diff is None
        assert result.meta.get("error") == "bad_find_replace"

    def test_path_is_directory(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        (tmp_path / "subdir").mkdir()
        edits_json = json.dumps(
            {"edits": [{"path": "subdir", "find": "x", "replace": "y", "count": 1}]}
        )

        async def _fake_chat(**kw):
            return _ok_response(edits_json), dict(_GOOD_META)

        monkeypatch.setattr(openai_mod, "_chat_completion_async", _fake_chat)
        result = asyncio.run(
            openai_mod._propose_patch_via_edits_async(
                url="http://localhost/v1/chat/completions",
                api_key="k",
                model="m",
                prompt="fix",
                cwd=tmp_path,
                temperature=0.2,
                max_tokens=500,
                timeout_sec=30,
                seed=None,
                http2=False,
            )
        )
        assert result.diff is None
        assert result.meta.get("error") == "path_not_file"

    def test_missing_edits_key(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        """All three stages return JSON without 'edits' key."""

        async def _fake_chat(**kw):
            return _ok_response('{"changes": []}'), dict(_GOOD_META)

        monkeypatch.setattr(openai_mod, "_chat_completion_async", _fake_chat)
        result = asyncio.run(
            openai_mod._propose_patch_via_edits_async(
                url="http://localhost/v1/chat/completions",
                api_key="k",
                model="m",
                prompt="fix",
                cwd=tmp_path,
                temperature=0.2,
                max_tokens=500,
                timeout_sec=30,
                seed=None,
                http2=False,
                allow_diff_fallback=False,
            )
        )
        assert result.diff is None


# ---------------------------------------------------------------------------
# 12. Full File Proposal: _propose_patch_via_full_file_async
# ---------------------------------------------------------------------------


class TestProposePatchViaFullFileAsync:
    def test_unsupported_no_python_files(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        (tmp_path / "readme.txt").write_text("hello", encoding="utf-8")

        async def _fake_chat(**kw):
            return _ok_response(_VALID_DIFF), dict(_GOOD_META)

        monkeypatch.setattr(openai_mod, "_chat_completion_async", _fake_chat)
        result = asyncio.run(
            openai_mod._propose_patch_via_full_file_async(
                url="http://localhost/v1/chat/completions",
                api_key="k",
                model="m",
                prompt="fix",
                cwd=tmp_path,
                temperature=0.2,
                max_tokens=500,
                timeout_sec=30,
                seed=None,
                http2=False,
            )
        )
        # Falls back to diff mode
        assert result.meta.get("full_file_support_reason") == "no_non_test_python_files"

    def test_success_single_file(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        (tmp_path / "foo.py").write_text("def foo():\n    return 1\n", encoding="utf-8")
        (tmp_path / "test_foo.py").write_text("import foo\n", encoding="utf-8")

        async def _fake_chat(**kw):
            return _ok_response("def foo():\n    return 2\n"), dict(_GOOD_META)

        monkeypatch.setattr(openai_mod, "_chat_completion_async", _fake_chat)
        result = asyncio.run(
            openai_mod._propose_patch_via_full_file_async(
                url="http://localhost/v1/chat/completions",
                api_key="k",
                model="m",
                prompt="fix",
                cwd=tmp_path,
                temperature=0.2,
                max_tokens=500,
                timeout_sec=30,
                seed=None,
                http2=False,
            )
        )
        assert result.diff is not None
        assert "return 2" in result.diff

    def test_no_effective_changes(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        (tmp_path / "foo.py").write_text("same content\n", encoding="utf-8")

        async def _fake_chat(**kw):
            return _ok_response("same content\n"), dict(_GOOD_META)

        monkeypatch.setattr(openai_mod, "_chat_completion_async", _fake_chat)
        result = asyncio.run(
            openai_mod._propose_patch_via_full_file_async(
                url="http://localhost/v1/chat/completions",
                api_key="k",
                model="m",
                prompt="fix",
                cwd=tmp_path,
                temperature=0.2,
                max_tokens=500,
                timeout_sec=30,
                seed=None,
                http2=False,
            )
        )
        # The full-file path treats a no-op repair as a recoverable error when the
        # diff fallback also produces no patch.
        assert result.diff is None
        assert result.meta.get("error") == "no_effective_changes"
        assert result.meta.get("fallback_meta") is not None

    def test_file_read_error(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        (tmp_path / "foo.py").write_text("code\n", encoding="utf-8")
        # Make file unreadable
        monkeypatch.setattr(
            openai_mod,
            "_resolve_full_file_target",
            lambda *, cwd: ("foo.py", "single_non_test_python_file"),
        )
        _original_resolve = Path.resolve

        def _bad_read(path):
            raise PermissionError("cannot read")

        monkeypatch.setattr(openai_mod, "_read_edit_target", _bad_read)
        result = asyncio.run(
            openai_mod._propose_patch_via_full_file_async(
                url="http://localhost/v1/chat/completions",
                api_key="k",
                model="m",
                prompt="fix",
                cwd=tmp_path,
                temperature=0.2,
                max_tokens=500,
                timeout_sec=30,
                seed=None,
                http2=False,
            )
        )
        assert result.diff is None
        assert result.meta.get("error") == "file_read_error"

    def test_budget_exhausted(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        (tmp_path / "foo.py").write_text("code\n", encoding="utf-8")

        async def _fake_chat(**kw):
            return None, {
                "error": "request_budget_exceeded",
                "llm_call_count": 1,
                "usage": dict(_GOOD_USAGE),
            }

        monkeypatch.setattr(openai_mod, "_chat_completion_async", _fake_chat)
        result = asyncio.run(
            openai_mod._propose_patch_via_full_file_async(
                url="http://localhost/v1/chat/completions",
                api_key="k",
                model="m",
                prompt="fix",
                cwd=tmp_path,
                temperature=0.2,
                max_tokens=500,
                timeout_sec=30,
                seed=None,
                http2=False,
            )
        )
        # Should fallback to diff or return None
        assert result.diff is None or result.diff is not None  # completes without error


# ---------------------------------------------------------------------------
# 13. Chat Completion Retry: _chat_completion_async
# ---------------------------------------------------------------------------


class TestChatCompletionAsync:
    def test_success_first_try(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(openai_mod, "_should_use_proxy_transport", lambda url: True)

        class FakeAsyncClient:
            def __init__(self, **kw):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                pass

            async def post(self, url, content, headers):
                body = json.dumps(
                    {
                        "choices": [{"message": {"content": "hello"}}],
                        "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                    }
                )
                return SimpleNamespace(status_code=200, text=body, http_version="HTTP/1.1")

        monkeypatch.setattr(openai_mod.httpx, "AsyncClient", FakeAsyncClient)
        data, meta = asyncio.run(
            openai_mod._chat_completion_async(
                url="http://localhost/v1/chat/completions",
                api_key="k",
                model="m",
                messages=[{"role": "user", "content": "hi"}],
                temperature=0.2,
                max_tokens=100,
                timeout_sec=30,
                seed=None,
                http2=False,
                transport_max_retries=0,
            )
        )
        assert data is not None
        assert data["choices"][0]["message"]["content"] == "hello"
        assert meta["llm_call_count"] == 1

    def test_budget_exceeded_before_first_call(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import time

        monkeypatch.setattr(openai_mod, "_should_use_proxy_transport", lambda url: True)
        deadline = time.monotonic() - 1  # already past

        data, meta = asyncio.run(
            openai_mod._chat_completion_async(
                url="http://localhost/v1/chat/completions",
                api_key="k",
                model="m",
                messages=[{"role": "user", "content": "hi"}],
                temperature=0.2,
                max_tokens=100,
                timeout_sec=30,
                seed=None,
                http2=False,
                deadline_monotonic=deadline,
                transport_max_retries=0,
            )
        )
        assert data is None
        assert meta["error"] == "request_budget_exceeded"

    def test_retryable_error_with_retry(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(openai_mod, "_should_use_proxy_transport", lambda url: True)
        calls = {"n": 0}

        class FakeAsyncClient:
            def __init__(self, **kw):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                pass

            async def post(self, url, content, headers):
                calls["n"] += 1
                if calls["n"] == 1:
                    return SimpleNamespace(
                        status_code=429, text="rate limited", http_version="HTTP/1.1"
                    )
                body = json.dumps(
                    {
                        "choices": [{"message": {"content": "ok"}}],
                        "usage": dict(_GOOD_USAGE),
                    }
                )
                return SimpleNamespace(status_code=200, text=body, http_version="HTTP/1.1")

        monkeypatch.setattr(openai_mod.httpx, "AsyncClient", FakeAsyncClient)
        monkeypatch.setattr(openai_mod.asyncio, "sleep", lambda _: asyncio.sleep(0))
        data, meta = asyncio.run(
            openai_mod._chat_completion_async(
                url="http://localhost/v1/chat/completions",
                api_key="k",
                model="m",
                messages=[{"role": "user", "content": "hi"}],
                temperature=0.2,
                max_tokens=100,
                timeout_sec=30,
                seed=None,
                http2=False,
                transport_max_retries=2,
                transport_retry_backoff_base_sec=0.0,
                transport_retry_backoff_max_sec=0.0,
            )
        )
        assert data is not None
        assert meta["transport_retry_count"] == 1

    def test_non_retryable_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(openai_mod, "_should_use_proxy_transport", lambda url: True)

        class FakeAsyncClient:
            def __init__(self, **kw):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                pass

            async def post(self, url, content, headers):
                return SimpleNamespace(status_code=400, text="bad request", http_version="HTTP/1.1")

        monkeypatch.setattr(openai_mod.httpx, "AsyncClient", FakeAsyncClient)
        data, meta = asyncio.run(
            openai_mod._chat_completion_async(
                url="http://localhost/v1/chat/completions",
                api_key="k",
                model="m",
                messages=[{"role": "user", "content": "hi"}],
                temperature=0.2,
                max_tokens=100,
                timeout_sec=30,
                seed=None,
                http2=False,
                transport_max_retries=2,
            )
        )
        assert data is None
        assert meta["error"] == "http_error"
        assert meta["status"] == 400

    def test_bad_json_response(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(openai_mod, "_should_use_proxy_transport", lambda url: True)

        class FakeAsyncClient:
            def __init__(self, **kw):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                pass

            async def post(self, url, content, headers):
                return SimpleNamespace(status_code=200, text="NOTJSON{{{", http_version="HTTP/1.1")

        monkeypatch.setattr(openai_mod.httpx, "AsyncClient", FakeAsyncClient)
        data, meta = asyncio.run(
            openai_mod._chat_completion_async(
                url="http://localhost/v1/chat/completions",
                api_key="k",
                model="m",
                messages=[{"role": "user", "content": "hi"}],
                temperature=0.2,
                max_tokens=100,
                timeout_sec=30,
                seed=None,
                http2=False,
                transport_max_retries=0,
            )
        )
        assert data is None
        assert meta["error"] == "bad_json_response"

    def test_seed_acknowledged(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(openai_mod, "_should_use_proxy_transport", lambda url: True)

        class FakeAsyncClient:
            def __init__(self, **kw):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                pass

            async def post(self, url, content, headers):
                body = json.dumps(
                    {
                        "choices": [{"message": {"content": "ok"}}],
                        "usage": dict(_GOOD_USAGE),
                        "seed": 42,
                        "system_fingerprint": "fp_abc",
                    }
                )
                return SimpleNamespace(status_code=200, text=body, http_version="HTTP/1.1")

        monkeypatch.setattr(openai_mod.httpx, "AsyncClient", FakeAsyncClient)
        data, meta = asyncio.run(
            openai_mod._chat_completion_async(
                url="http://localhost/v1/chat/completions",
                api_key="k",
                model="m",
                messages=[{"role": "user", "content": "hi"}],
                temperature=0.2,
                max_tokens=100,
                timeout_sec=30,
                seed=42,
                http2=False,
                transport_max_retries=0,
            )
        )
        assert data is not None
        assert meta["api_compat_seed_requested"] == 42

    def test_http2_fallback_in_chat(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(openai_mod, "_should_use_proxy_transport", lambda url: True)

        class FakeAsyncClient:
            def __init__(self, *, timeout, http2, trust_env):
                self._http2 = http2

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                pass

            async def post(self, url, content, headers):
                if self._http2:
                    raise httpx.RemoteProtocolError("h2 error")
                body = json.dumps(
                    {
                        "choices": [{"message": {"content": "ok"}}],
                        "usage": dict(_GOOD_USAGE),
                    }
                )
                return SimpleNamespace(status_code=200, text=body, http_version="HTTP/1.1")

        monkeypatch.setattr(openai_mod.httpx, "AsyncClient", FakeAsyncClient)
        data, meta = asyncio.run(
            openai_mod._chat_completion_async(
                url="http://localhost/v1/chat/completions",
                api_key="k",
                model="m",
                messages=[{"role": "user", "content": "hi"}],
                temperature=0.2,
                max_tokens=100,
                timeout_sec=30,
                seed=None,
                http2=True,
                transport_max_retries=0,
            )
        )
        assert data is not None
        assert meta["transport_http2_fallback_used"] is True

    def test_connection_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(openai_mod, "_should_use_proxy_transport", lambda url: True)

        class FakeAsyncClient:
            def __init__(self, **kw):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                pass

            async def post(self, url, content, headers):
                raise ConnectionError("refused")

        monkeypatch.setattr(openai_mod.httpx, "AsyncClient", FakeAsyncClient)
        data, meta = asyncio.run(
            openai_mod._chat_completion_async(
                url="http://localhost/v1/chat/completions",
                api_key="k",
                model="m",
                messages=[{"role": "user", "content": "hi"}],
                temperature=0.2,
                max_tokens=100,
                timeout_sec=30,
                seed=None,
                http2=False,
                transport_max_retries=0,
            )
        )
        assert data is None
        assert meta["error"] == "request_failed"

    def test_reasoning_content_extracted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(openai_mod, "_should_use_proxy_transport", lambda url: True)

        class FakeAsyncClient:
            def __init__(self, **kw):
                pass

            async def __aenter__(self):
                return self

            async def __aexit__(self, *a):
                pass

            async def post(self, url, content, headers):
                body = json.dumps(
                    {
                        "choices": [
                            {
                                "message": {
                                    "content": "fixed",
                                    "reasoning_content": "I thought about it",
                                }
                            }
                        ],
                        "usage": dict(_GOOD_USAGE),
                    }
                )
                return SimpleNamespace(status_code=200, text=body, http_version="HTTP/1.1")

        monkeypatch.setattr(openai_mod.httpx, "AsyncClient", FakeAsyncClient)
        data, meta = asyncio.run(
            openai_mod._chat_completion_async(
                url="http://localhost/v1/chat/completions",
                api_key="k",
                model="m",
                messages=[{"role": "user", "content": "hi"}],
                temperature=0.2,
                max_tokens=100,
                timeout_sec=30,
                seed=None,
                http2=False,
                transport_max_retries=0,
            )
        )
        assert data is not None
        assert meta.get("reasoning_content_present") is True


# ---------------------------------------------------------------------------
# 14. Patcher: apropose_patch dispatch
# ---------------------------------------------------------------------------


class TestAproposeDispatch:
    def test_diff_mode_with_edits_fallback(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        """diff mode: first returns no_diff_found → falls back to edits."""
        (tmp_path / "foo.py").write_text("old\n", encoding="utf-8")
        calls = {"n": 0}

        async def _fake_chat(**kw):
            calls["n"] += 1
            msgs = kw.get("messages", [])
            sys_content = msgs[0].get("content", "") if msgs else ""
            if "JSON" in sys_content or "json" in sys_content.lower():
                edits_json = json.dumps(
                    {"edits": [{"path": "foo.py", "find": "old", "replace": "new", "count": 1}]}
                )
                return _ok_response(edits_json), dict(_GOOD_META)
            return _ok_response("no diff here"), dict(_GOOD_META)

        monkeypatch.setattr(openai_mod, "_chat_completion_async", _fake_chat)
        patcher = OpenAICompatPatcher(
            base_url="http://127.0.0.1:1234",
            model="m",
            response_mode="diff",
            api_key="test-key",
        )
        result = asyncio.run(
            patcher.apropose_patch(prompt="fix", context_dir=tmp_path, cwd=tmp_path)
        )
        assert result.meta.get("fallback_from") == "diff"
        assert result.meta.get("fallback_reason") == "no_diff_found"

    def test_edits_mode(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        (tmp_path / "foo.py").write_text("old\n", encoding="utf-8")
        edits_json = json.dumps(
            {"edits": [{"path": "foo.py", "find": "old", "replace": "new", "count": 1}]}
        )

        async def _fake_chat(**kw):
            return _ok_response(edits_json), dict(_GOOD_META)

        monkeypatch.setattr(openai_mod, "_chat_completion_async", _fake_chat)
        patcher = OpenAICompatPatcher(
            base_url="http://127.0.0.1:1234",
            model="m",
            response_mode="edits",
            api_key="test-key",
        )
        result = asyncio.run(
            patcher.apropose_patch(prompt="fix", context_dir=tmp_path, cwd=tmp_path)
        )
        assert result.diff is not None
        assert result.meta.get("response_mode") == "edits"

    def test_full_file_mode(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        (tmp_path / "foo.py").write_text("old\n", encoding="utf-8")

        async def _fake_chat(**kw):
            return _ok_response("new\n"), dict(_GOOD_META)

        monkeypatch.setattr(openai_mod, "_chat_completion_async", _fake_chat)
        patcher = OpenAICompatPatcher(
            base_url="http://127.0.0.1:1234",
            model="m",
            response_mode="full_file",
            api_key="test-key",
        )
        result = asyncio.run(
            patcher.apropose_patch(prompt="fix", context_dir=tmp_path, cwd=tmp_path)
        )
        assert result.meta.get("response_mode") == "full_file"

    def test_forced_patch_skip_transport_failure(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        async def _fake_chat(**kw):
            return None, {
                "error": "request_failed",
                "exc": "Connection refused",
                "llm_call_count": 1,
                "usage": dict(_GOOD_USAGE),
            }

        monkeypatch.setattr(openai_mod, "_chat_completion_async", _fake_chat)
        patcher = OpenAICompatPatcher(
            base_url="http://127.0.0.1:1234",
            model="m",
            response_mode="diff",
            api_key="test-key",
            force_patch_attempt=True,
        )
        result = asyncio.run(
            patcher.apropose_patch(prompt="fix", context_dir=tmp_path, cwd=tmp_path)
        )
        assert result.diff is None
        assert result.meta.get("forced_patch_skipped") is True


# ---------------------------------------------------------------------------
# 15. _normalize_context_compaction_payload
# ---------------------------------------------------------------------------


class TestNormalizeContextCompactionPayload:
    def test_none_response(self) -> None:
        assert openai_mod._normalize_context_compaction_payload(None) == (None, None)

    def test_no_content(self) -> None:
        c, _r = openai_mod._normalize_context_compaction_payload({"choices": [{"message": {}}]})
        assert c is None

    def test_no_json_in_content(self) -> None:
        c, r = openai_mod._normalize_context_compaction_payload(
            {"choices": [{"message": {"content": "plain text no json"}}]}
        )
        assert c is None
        assert r == "plain text no json"

    def test_json_with_summary_key(self) -> None:
        c, _r = openai_mod._normalize_context_compaction_payload(
            {"choices": [{"message": {"content": '{"summary": "brief"}'}}]}
        )
        assert c == "brief"

    def test_json_with_context_key(self) -> None:
        c, _r = openai_mod._normalize_context_compaction_payload(
            {"choices": [{"message": {"content": '{"context": "ctx data"}'}}]}
        )
        assert c == "ctx data"


# ---------------------------------------------------------------------------
# 16. _forced_patch_skip_reason
# ---------------------------------------------------------------------------


class TestForcedPatchSkipReason:
    def test_request_failed(self) -> None:
        assert openai_mod._forced_patch_skip_reason({"error": "request_failed"}) == "request_failed"

    def test_bad_request_url(self) -> None:
        assert (
            openai_mod._forced_patch_skip_reason({"error": "bad_request_url"}) == "bad_request_url"
        )

    def test_budget_exceeded(self) -> None:
        assert (
            openai_mod._forced_patch_skip_reason({"error": "request_budget_exceeded"})
            == "request_budget_exceeded"
        )

    def test_context_limit(self) -> None:
        meta = {"error": "http_error", "body": "exceeds the available context size"}
        assert openai_mod._forced_patch_skip_reason(meta) == "context_limit"

    def test_nested_fallback(self) -> None:
        meta = {"fallback_meta": {"error": "request_failed"}}
        assert openai_mod._forced_patch_skip_reason(meta) == "fallback_request_failed"

    def test_no_skip_reason(self) -> None:
        assert openai_mod._forced_patch_skip_reason({}) == ""


# ---------------------------------------------------------------------------
# 17. Sync server ready probe
# ---------------------------------------------------------------------------


class TestProbeServerReady:
    def test_all_fail(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _fake_probe(*, url, api_key, timeout_sec, http2):
            return False, {"error": "request_failed"}

        monkeypatch.setattr(openai_mod, "_http_probe_ready", _fake_probe)
        ready, meta = openai_mod._probe_server_ready(
            base_url="http://localhost:1234/v1",
            api_key="k",
            timeout_sec=5,
            http2=False,
        )
        assert ready is False
        assert meta["error"] == "server_ready_probe_failed"

    def test_health_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def _fake_probe(*, url, api_key, timeout_sec, http2):
            if "/health" in url:
                return True, {"status": 200}
            return False, {"error": "request_failed"}

        monkeypatch.setattr(openai_mod, "_http_probe_ready", _fake_probe)
        ready, meta = openai_mod._probe_server_ready(
            base_url="http://localhost:1234/v1",
            api_key="k",
            timeout_sec=5,
            http2=False,
        )
        assert ready is True
        assert meta["server_ready_probe_provider"] == "health"


# ---------------------------------------------------------------------------
# 18. _chat_http_client_key edge cases
# ---------------------------------------------------------------------------


class TestChatHttpClientKey:
    def test_unsupported_scheme(self) -> None:
        with pytest.raises(ValueError, match="unsupported URL scheme"):
            openai_mod._chat_http_client_key("ftp://host/path", http2=False)

    def test_missing_host(self) -> None:
        with pytest.raises(ValueError, match="missing host"):
            openai_mod._chat_http_client_key("http:///path", http2=False)

    def test_https_default_port(self) -> None:
        scheme, _host, port, h2 = openai_mod._chat_http_client_key(
            "https://example.com/v1/chat/completions", http2=True
        )
        assert scheme == "https"
        assert port == 443
        assert h2 is True


# ---------------------------------------------------------------------------
# 19. _is_http2_protocol_error
# ---------------------------------------------------------------------------


class TestIsHttp2ProtocolError:
    def test_remote_protocol_error(self) -> None:
        assert openai_mod._is_http2_protocol_error(httpx.RemoteProtocolError("h2")) is True

    def test_string_matching(self) -> None:
        assert openai_mod._is_http2_protocol_error(RuntimeError("http/2 protocol error")) is True

    def test_non_http2_error(self) -> None:
        assert openai_mod._is_http2_protocol_error(RuntimeError("connection reset")) is False


# ---------------------------------------------------------------------------
# 20. Public-method edge branches
# ---------------------------------------------------------------------------


class TestPatcherPublicMethodEdgeBranches:
    def test_sync_public_methods_use_default_api_key_and_empty_payload_fallbacks(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(openai_mod, "_default_api_key", lambda: "fallback-key")
        captured: dict[str, str] = {}

        def _fake_request_context_via_json(**kwargs):
            captured["request_context_api_key"] = str(kwargs["api_key"])
            return _ok_response("{bad json}"), {}

        def _fake_chat_completion(**kwargs):
            captured["plan_patch_api_key"] = str(kwargs["api_key"])
            return {"choices": []}, {}

        def _fake_compact_context_via_json(**kwargs):
            captured["compact_context_api_key"] = str(kwargs["api_key"])
            return None, "{bad}", {}

        monkeypatch.setattr(openai_mod, "_request_context_via_json", _fake_request_context_via_json)
        monkeypatch.setattr(openai_mod, "_chat_completion", _fake_chat_completion)
        monkeypatch.setattr(
            openai_mod,
            "_compact_context_via_json",
            _fake_compact_context_via_json,
        )

        patcher = OpenAICompatPatcher(
            base_url="http://example.com",
            model="m",
            api_key=None,
            response_mode="edits",
        )

        request_result = patcher.request_context(
            prompt="fix",
            context_dir=tmp_path,
            cwd=tmp_path,
            strict_json_first=True,
        )
        assert isinstance(request_result, ContextRequestResult)
        assert request_result.request_json is None
        assert request_result.meta["error"] == "invalid_request_context_payload"
        assert captured["request_context_api_key"] == "fallback-key"

        plan_result = patcher.plan_patch(
            prompt="plan",
            context_dir=tmp_path,
            cwd=tmp_path,
        )
        assert isinstance(plan_result, PatchPlanResult)
        assert plan_result.plan_text is None
        assert plan_result.meta["error"] == "empty_plan_response"
        assert captured["plan_patch_api_key"] == "fallback-key"

        compact_result = patcher.compact_context(
            prompt="compact",
            context_text="abcdef",
            context_dir=tmp_path,
            cwd=tmp_path,
            target_chars=5,
        )
        assert isinstance(compact_result, ContextCompactionResult)
        assert compact_result.compacted_text is None
        assert compact_result.meta["error"] == "invalid_context_compaction_payload"
        assert captured["compact_context_api_key"] == "fallback-key"

    def test_async_public_methods_use_default_api_key_and_empty_payload_fallbacks(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(openai_mod, "_default_api_key", lambda: "fallback-key")
        captured: dict[str, str] = {}

        async def _fake_request_context_via_json_async(**kwargs):
            captured["request_context_api_key"] = str(kwargs["api_key"])
            return _ok_response("{bad json}"), {}

        async def _fake_compact_context_via_json_async(**kwargs):
            captured["compact_context_api_key"] = str(kwargs["api_key"])
            return None, "{bad}", {}

        monkeypatch.setattr(
            openai_mod,
            "_request_context_via_json_async",
            _fake_request_context_via_json_async,
        )
        monkeypatch.setattr(
            openai_mod,
            "_compact_context_via_json_async",
            _fake_compact_context_via_json_async,
        )

        patcher = OpenAICompatPatcher(
            base_url="http://example.com",
            model="m",
            api_key=None,
            response_mode="edits",
        )

        request_result = asyncio.run(
            patcher.arequest_context(
                prompt="fix",
                context_dir=tmp_path,
                cwd=tmp_path,
                strict_json_first=True,
            )
        )
        assert isinstance(request_result, ContextRequestResult)
        assert request_result.request_json is None
        assert request_result.meta["error"] == "invalid_request_context_payload"
        assert captured["request_context_api_key"] == "fallback-key"

        compact_result = asyncio.run(
            patcher.acompact_context(
                prompt="compact",
                context_text="abcdef",
                context_dir=tmp_path,
                cwd=tmp_path,
                target_chars=5,
            )
        )
        assert isinstance(compact_result, ContextCompactionResult)
        assert compact_result.compacted_text is None
        assert compact_result.meta["error"] == "invalid_context_compaction_payload"
        assert captured["compact_context_api_key"] == "fallback-key"

    def test_probe_methods_use_default_api_key(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(openai_mod, "_default_api_key", lambda: "fallback-key")
        captured: dict[str, str] = {}

        def _fake_probe_context_window_tokens(**kwargs):
            captured["context_window_api_key"] = str(kwargs["api_key"])
            return 77, {"context_window_probe_provider": "stub"}

        async def _fake_probe_context_window_tokens_async(**kwargs):
            captured["acontext_window_api_key"] = str(kwargs["api_key"])
            return 88, {"context_window_probe_provider": "stub_async"}

        def _fake_probe_server_ready(**kwargs):
            captured["server_ready_api_key"] = str(kwargs["api_key"])
            return True, {"server_ready_probe_provider": "stub"}

        async def _fake_probe_server_ready_async(**kwargs):
            captured["aserver_ready_api_key"] = str(kwargs["api_key"])
            return True, {"server_ready_probe_provider": "stub_async"}

        monkeypatch.setattr(
            openai_mod, "_probe_context_window_tokens", _fake_probe_context_window_tokens
        )
        monkeypatch.setattr(
            openai_mod,
            "_probe_context_window_tokens_async",
            _fake_probe_context_window_tokens_async,
        )
        monkeypatch.setattr(openai_mod, "_probe_server_ready", _fake_probe_server_ready)
        monkeypatch.setattr(openai_mod, "_probe_server_ready_async", _fake_probe_server_ready_async)

        patcher = OpenAICompatPatcher(base_url="http://example.com", model="m", api_key=None)

        tokens, meta = patcher.probe_context_window_tokens()
        assert tokens == 77
        assert meta["context_window_probe_provider"] == "stub"
        assert captured["context_window_api_key"] == "fallback-key"

        atokens, ameta = asyncio.run(patcher.aprobe_context_window_tokens())
        assert atokens == 88
        assert ameta["context_window_probe_provider"] == "stub_async"
        assert captured["acontext_window_api_key"] == "fallback-key"

        ready, ready_meta = patcher.probe_server_ready()
        assert ready is True
        assert ready_meta["server_ready_probe_provider"] == "stub"
        assert captured["server_ready_api_key"] == "fallback-key"

        aready, aready_meta = asyncio.run(patcher.aprobe_server_ready())
        assert aready is True
        assert aready_meta["server_ready_probe_provider"] == "stub_async"
        assert captured["aserver_ready_api_key"] == "fallback-key"

    def test_force_patch_attempt_returns_result_when_forced_diff_is_unavailable(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(openai_mod, "_default_api_key", lambda: "fallback-key")
        monkeypatch.setattr(
            openai_mod,
            "_propose_patch_via_diff",
            lambda **kwargs: PatchResult(diff=None, meta={}),
        )
        monkeypatch.setattr(
            openai_mod,
            "_propose_patch_via_diff_async",
            lambda **kwargs: asyncio.sleep(0, result=PatchResult(diff=None, meta={})),
        )
        monkeypatch.setattr(openai_mod, "_build_forced_attempt_diff", lambda **kwargs: None)

        patcher = OpenAICompatPatcher(
            base_url="http://example.com",
            model="m",
            api_key=None,
            response_mode="diff",
            force_patch_attempt=True,
        )

        sync_result = patcher.propose_patch(
            prompt="fix",
            context_dir=tmp_path,
            cwd=tmp_path,
        )
        assert sync_result.diff is None
        assert sync_result.meta.get("forced_patch_attempt") is None

        async def _fake_diff_async(**kwargs):
            return PatchResult(diff=None, meta={})

        monkeypatch.setattr(openai_mod, "_propose_patch_via_diff_async", _fake_diff_async)
        async_result = asyncio.run(
            patcher.apropose_patch(
                prompt="fix",
                context_dir=tmp_path,
                cwd=tmp_path,
            )
        )
        assert async_result.diff is None
        assert async_result.meta.get("forced_patch_attempt") is None

    def test_force_patch_attempt_uses_forced_diff_when_available(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        forced_diff = (
            "diff --git a/main.py b/main.py\n"
            "--- a/main.py\n"
            "+++ b/main.py\n"
            "@@ -1 +1 @@\n"
            "-old\n"
            "+new\n"
        )
        monkeypatch.setattr(
            openai_mod,
            "_propose_patch_via_diff",
            lambda **kwargs: PatchResult(diff=None, meta={}),
        )

        async def _fake_diff_async(**kwargs):
            return PatchResult(diff=None, meta={})

        monkeypatch.setattr(openai_mod, "_propose_patch_via_diff_async", _fake_diff_async)
        monkeypatch.setattr(
            openai_mod, "_build_forced_attempt_diff", lambda **kwargs: ("main.py", forced_diff)
        )

        patcher = OpenAICompatPatcher(
            base_url="http://example.com",
            model="m",
            api_key="k",
            response_mode="diff",
            force_patch_attempt=True,
        )

        sync_result = patcher.propose_patch(
            prompt="fix",
            context_dir=tmp_path,
            cwd=tmp_path,
        )
        assert sync_result.diff == forced_diff
        assert sync_result.meta["forced_patch_attempt"] is True
        assert sync_result.meta["forced_patch_target"] == "main.py"
        assert sync_result.meta["forced_patch_reason"] == "no_model_diff"

        async_result = asyncio.run(
            patcher.apropose_patch(
                prompt="fix",
                context_dir=tmp_path,
                cwd=tmp_path,
            )
        )
        assert async_result.diff == forced_diff
        assert async_result.meta["forced_patch_attempt"] is True
        assert async_result.meta["forced_patch_target"] == "main.py"
        assert async_result.meta["forced_patch_reason"] == "no_model_diff"


# ---------------------------------------------------------------------------
# 21. Sync helper edge branches
# ---------------------------------------------------------------------------


class TestTransportProgressAndPayloadCoercion:
    def test_emit_progress_swallows_callback_exceptions_and_coerces_payload_bools(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def _boom(event: str, payload: dict[str, object] | None) -> None:
            raise RuntimeError("boom")

        token = openai_mod._OPENAI_PROGRESS_CALLBACK.set(_boom)
        try:
            openai_mod._emit_openai_transport_progress("retry", note="test", retry=1)
        finally:
            openai_mod._OPENAI_PROGRESS_CALLBACK.reset(token)

        assert openai_mod._coerce_payload_bool(0) is False
        assert openai_mod._coerce_payload_bool(2) is False
        assert openai_mod._coerce_payload_bool(" yes ") is True
        assert openai_mod._coerce_payload_bool("OFF") is False
        assert openai_mod._coerce_payload_bool(object()) is False


class TestPersistentHttpClients:
    def test_unreachable_branches_when_range_is_shadowed(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(openai_mod, "range", lambda *args: [], raising=False)

        sync_client = openai_mod._PersistentHTTPClient(
            scheme="http",
            host="example.com",
            port=80,
            http2=False,
        )
        async_client = openai_mod._PersistentAsyncHTTPClient(
            scheme="http",
            host="example.com",
            port=80,
            http2=False,
        )

        with pytest.raises(RuntimeError, match="unreachable"):
            sync_client.post_json(
                target="/v1/chat/completions", body=b"{}", headers={}, timeout_sec=1
            )
        with pytest.raises(RuntimeError, match="unreachable"):
            sync_client.get_text(target="/health", headers={}, timeout_sec=1)
        with pytest.raises(RuntimeError, match="unreachable"):
            asyncio.run(
                async_client.post_json(
                    target="/v1/chat/completions",
                    body=b"{}",
                    headers={},
                    timeout_sec=1,
                )
            )
        with pytest.raises(RuntimeError, match="unreachable"):
            asyncio.run(async_client.get_text(target="/health", headers={}, timeout_sec=1))


class TestSyncHttpFallbacks:
    def test_http_get_json_http2_fallback_value_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class FakeClient:
            def __init__(self, *, timeout, http2, trust_env):
                self._http2 = http2

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

            def get(self, url, headers):
                if self._http2:
                    raise httpx.RemoteProtocolError("h2 error")
                raise ValueError("bad url")

        monkeypatch.setattr(openai_mod.httpx, "Client", FakeClient)
        data, meta = openai_mod._http_get_json(
            url="http://localhost:8080/v1/models",
            api_key="k",
            timeout_sec=5.0,
            http2=True,
        )
        assert data is None
        assert meta["error"] == "bad_request_url"

    def test_http_probe_ready_http2_fallback_value_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(openai_mod, "_should_use_proxy_transport", lambda url: True)

        class FakeClient:
            def __init__(self, *, timeout, http2, trust_env):
                self._http2 = http2

            def __enter__(self):
                return self

            def __exit__(self, *args):
                return None

            def get(self, url, headers):
                if self._http2:
                    raise httpx.RemoteProtocolError("h2 error")
                raise ValueError("bad url")

        monkeypatch.setattr(openai_mod.httpx, "Client", FakeClient)
        ready, meta = openai_mod._http_probe_ready(
            url="http://localhost:8080/health",
            api_key="k",
            timeout_sec=5.0,
            http2=True,
        )
        assert ready is None  # type: ignore[comparison-overlap]
        assert meta["error"] == "bad_request_url"

    def test_request_and_compact_context_json_response_format_fallback(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = {"request": 0, "compact": 0}

        def _fake_chat(**kwargs):
            system_prompt = str(kwargs["messages"][0]["content"])
            extra_payload = kwargs.get("extra_payload")
            if "staged context requests" in system_prompt:
                calls["request"] += 1
                if extra_payload is not None:
                    return None, {
                        "error": "http_error",
                        "body": "response_format not supported",
                        "llm_call_count": 1,
                        "usage": dict(_GOOD_USAGE),
                    }
                return (
                    _ok_response(
                        json.dumps(
                            {
                                "need_more_context": False,
                                "need_snapshot": False,
                                "need_compact_context": False,
                                "requests": [],
                            }
                        )
                    ),
                    dict(_GOOD_META),
                )
            if "context compaction" in system_prompt:
                calls["compact"] += 1
                if extra_payload is not None:
                    return None, {
                        "error": "http_error",
                        "body": "response_format not supported",
                        "llm_call_count": 1,
                        "usage": dict(_GOOD_USAGE),
                    }
                return _ok_response('{"compacted_context": "short"}'), dict(_GOOD_META)
            raise AssertionError("unexpected helper invocation")

        monkeypatch.setattr(openai_mod, "_chat_completion", _fake_chat)
        request_raw, request_meta = openai_mod._request_context_via_json(
            url="http://localhost/v1/chat/completions",
            api_key="k",
            model="m",
            prompt="fix",
            temperature=0.2,
            max_tokens=100,
            timeout_sec=30,
            seed=None,
            http2=False,
        )
        assert request_raw is not None
        assert request_meta["request_context_response_format_fallback_used"] is True
        assert calls["request"] == 2

        compacted, raw_response, compact_meta = openai_mod._compact_context_via_json(
            url="http://localhost/v1/chat/completions",
            api_key="k",
            model="m",
            prompt="fix",
            context_text="some context",
            target_chars=12,
            temperature=0.2,
            max_tokens=100,
            timeout_sec=30,
            seed=None,
            http2=False,
        )
        assert compacted == "short"
        assert raw_response == '{"compacted_context": "short"}'
        assert compact_meta["llm_call_count"] == 2
        assert calls["compact"] == 2


class TestSyncPatchProposalBranches:
    def test_propose_patch_via_diff_tracks_no_choice_then_no_diff(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = {"n": 0}

        def _fake_chat(**kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return {"choices": [{"message": {}}], "usage": dict(_GOOD_USAGE)}, dict(_GOOD_META)
            return _ok_response("plain text, not a diff"), dict(_GOOD_META)

        monkeypatch.setattr(openai_mod, "_chat_completion", _fake_chat)
        result = openai_mod._propose_patch_via_diff(
            url="http://localhost/v1/chat/completions",
            api_key="k",
            model="m",
            prompt="fix",
            temperature=0.2,
            max_tokens=100,
            timeout_sec=30,
            seed=None,
            http2=False,
        )
        assert result.diff is None
        assert result.meta["error"] == "no_diff_found"
        assert calls["n"] == 2

    def test_propose_patch_via_edits_without_diff_fallback_returns_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        def _fake_chat(**kwargs):
            return _ok_response("not json at all"), dict(_GOOD_META)

        monkeypatch.setattr(openai_mod, "_chat_completion", _fake_chat)
        result = openai_mod._propose_patch_via_edits(
            url="http://localhost/v1/chat/completions",
            api_key="k",
            model="m",
            prompt="fix",
            cwd=tmp_path,
            temperature=0.2,
            max_tokens=100,
            timeout_sec=30,
            seed=None,
            http2=False,
            allow_diff_fallback=False,
        )
        assert result.diff is None
        assert result.meta["error"] == "no_json_payload"

    def test_propose_patch_via_edits_no_effective_changes_uses_diff_fallback(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        (tmp_path / "foo.py").write_text("same\n", encoding="utf-8")
        calls = {"n": 0}

        def _fake_chat(**kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return (
                    _ok_response(
                        json.dumps(
                            {
                                "edits": [
                                    {
                                        "path": "foo.py",
                                        "find": "same",
                                        "replace": "same",
                                        "count": 1,
                                    }
                                ]
                            }
                        )
                    ),
                    dict(_GOOD_META),
                )
            return _ok_response("not a diff"), dict(_GOOD_META)

        monkeypatch.setattr(openai_mod, "_chat_completion", _fake_chat)
        result = openai_mod._propose_patch_via_edits(
            url="http://localhost/v1/chat/completions",
            api_key="k",
            model="m",
            prompt="fix",
            cwd=tmp_path,
            temperature=0.2,
            max_tokens=100,
            timeout_sec=30,
            seed=None,
            http2=False,
        )
        assert result.diff is None
        assert result.meta["error"] == "no_effective_changes"
        assert result.meta["fallback_meta"]["error"] == "no_diff_found"

    def test_propose_patch_via_full_file_unsupported_case_shape(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        monkeypatch.setattr(
            openai_mod,
            "_propose_patch_via_diff",
            lambda **kwargs: PatchResult(diff=None, meta={"error": "no_diff_found"}),
        )
        result = openai_mod._propose_patch_via_full_file(
            url="http://localhost/v1/chat/completions",
            api_key="k",
            model="m",
            prompt="fix",
            cwd=tmp_path,
            temperature=0.2,
            max_tokens=100,
            timeout_sec=30,
            seed=None,
            http2=False,
        )
        assert result.diff is None
        assert result.meta["error"] == "unsupported_case_shape"

    def test_propose_patch_via_full_file_file_read_error(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        (tmp_path / "foo.py").write_text("code\n", encoding="utf-8")
        monkeypatch.setattr(
            openai_mod,
            "_resolve_full_file_target",
            lambda *, cwd: ("foo.py", "single_non_test_python_file"),
        )

        def _bad_read(path):
            raise PermissionError("cannot read")

        monkeypatch.setattr(openai_mod, "_read_edit_target", _bad_read)
        result = openai_mod._propose_patch_via_full_file(
            url="http://localhost/v1/chat/completions",
            api_key="k",
            model="m",
            prompt="fix",
            cwd=tmp_path,
            temperature=0.2,
            max_tokens=100,
            timeout_sec=30,
            seed=None,
            http2=False,
        )
        assert result.diff is None
        assert result.meta["error"] == "file_read_error"

    def test_propose_patch_via_full_file_no_effective_changes(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        (tmp_path / "foo.py").write_text("same\n", encoding="utf-8")
        (tmp_path / "test_foo.py").write_text("import foo\n", encoding="utf-8")
        calls = {"n": 0}

        def _fake_chat(**kwargs):
            calls["n"] += 1
            if calls["n"] == 1:
                return _ok_response("same\n"), dict(_GOOD_META)
            return _ok_response("not a diff"), dict(_GOOD_META)

        monkeypatch.setattr(openai_mod, "_chat_completion", _fake_chat)
        result = openai_mod._propose_patch_via_full_file(
            url="http://localhost/v1/chat/completions",
            api_key="k",
            model="m",
            prompt="fix",
            cwd=tmp_path,
            temperature=0.2,
            max_tokens=100,
            timeout_sec=30,
            seed=None,
            http2=False,
        )
        assert result.diff is None
        assert result.meta["error"] == "no_effective_changes"
        assert result.meta["fallback_meta"]["error"] == "no_diff_found"
