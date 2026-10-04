"""Coverage gap tests for evals/patchers/openai_compat_patcher.py.

Targets the largest untested code paths — primarily async variants, HTTP
transport edge cases, and branch conditions in helpers — to drive coverage
from ~63 % toward 80 %+.
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from evals.patchers import openai_compat_patcher as mod
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
_GOOD_META: dict[str, Any] = {"llm_call_count": 1, "usage": dict(_GOOD_USAGE)}


def _ok_response(content: str, *, usage: dict[str, Any] | None = None) -> dict[str, Any]:
    return {
        "choices": [{"message": {"content": content}, "finish_reason": "stop"}],
        "usage": usage or dict(_GOOD_USAGE),
    }


def _ok_chat(content: str) -> tuple[dict[str, Any], dict[str, Any]]:
    return _ok_response(content), dict(_GOOD_META)


def _err_chat(
    error: str = "http_error",
    body: str = "oops",
    status: int = 500,
) -> tuple[None, dict[str, Any]]:
    return None, {
        "error": error,
        "status": status,
        "body": body,
        "llm_call_count": 1,
        "usage": dict(_GOOD_USAGE),
    }


def _make_patcher(**overrides: Any) -> OpenAICompatPatcher:
    defaults: dict[str, Any] = {
        "base_url": "http://localhost:1234",
        "model": "test-model",
        "api_key": "test-key",
        "timeout_sec": 5.0,
        "http2": False,
        "transport_max_retries": 0,
        "max_propose_sec": 0,
    }
    defaults.update(overrides)
    return OpenAICompatPatcher(**defaults)


# ---------------------------------------------------------------------------
# 1. _PersistentHTTPClient — close exception, post retry-then-raise, get retry
# ---------------------------------------------------------------------------


class TestPersistentHTTPClient:
    def test_ensure_client_reuses_existing_client_for_same_timeout(self) -> None:
        client = mod._PersistentHTTPClient(scheme="http", host="localhost", port=1234, http2=False)
        existing = MagicMock()
        client._client = existing
        client._timeout_sec = 5.0

        with patch.object(client, "_new_client", side_effect=AssertionError("should not rebuild")):
            result = client._ensure_client(timeout_sec=5.0)

        assert result is existing

    def test_close_exception_swallowed(self) -> None:
        client = mod._PersistentHTTPClient(scheme="http", host="localhost", port=1234, http2=False)
        mock_httpx = MagicMock()
        mock_httpx.close.side_effect = RuntimeError("boom")
        client._client = mock_httpx
        client._timeout_sec = 5.0
        client.close()
        assert client._client is None

    def test_post_json_retry_then_raise(self) -> None:
        client = mod._PersistentHTTPClient(scheme="http", host="localhost", port=1234, http2=False)
        first_client = MagicMock()
        first_client.post.side_effect = httpx.ConnectError("fail-once")
        second_client = MagicMock()
        second_client.post.side_effect = httpx.ConnectError("fail-twice")

        with (
            patch.object(
                client, "_new_client", side_effect=[first_client, second_client]
            ) as factory,
            pytest.raises(httpx.ConnectError, match="fail-twice"),
        ):
            client.post_json(
                target="/v1/chat/completions",
                body=b"{}",
                headers={},
                timeout_sec=5.0,
            )

        assert factory.call_count == 2
        first_client.close.assert_called_once()
        second_client.close.assert_called_once()
        assert client._client is None

    def test_get_text_retry_then_raise(self) -> None:
        client = mod._PersistentHTTPClient(scheme="http", host="localhost", port=1234, http2=False)
        first_client = MagicMock()
        first_client.get.side_effect = httpx.ConnectError("fail-once")
        second_client = MagicMock()
        second_client.get.side_effect = httpx.ConnectError("fail-twice")

        with (
            patch.object(
                client, "_new_client", side_effect=[first_client, second_client]
            ) as factory,
            pytest.raises(httpx.ConnectError, match="fail-twice"),
        ):
            client.get_text(
                target="/v1/models",
                headers={},
                timeout_sec=5.0,
            )

        assert factory.call_count == 2
        first_client.close.assert_called_once()
        second_client.close.assert_called_once()
        assert client._client is None


# ---------------------------------------------------------------------------
# 2. _PersistentAsyncHTTPClient — close, post retry, get retry
# ---------------------------------------------------------------------------


class TestPersistentAsyncHTTPClient:
    @pytest.mark.asyncio
    async def test_ensure_client_reuses_existing_client_for_same_timeout(self) -> None:
        client = mod._PersistentAsyncHTTPClient(
            scheme="http", host="localhost", port=1234, http2=False
        )
        existing = AsyncMock()
        client._client = existing
        client._timeout_sec = 5.0

        with patch.object(client, "_new_client", side_effect=AssertionError("should not rebuild")):
            result = await client._ensure_client(timeout_sec=5.0)

        assert result is existing

    @pytest.mark.asyncio
    async def test_close_exception_swallowed(self) -> None:
        client = mod._PersistentAsyncHTTPClient(
            scheme="http", host="localhost", port=1234, http2=False
        )
        mock_httpx = AsyncMock()
        mock_httpx.aclose.side_effect = RuntimeError("boom")
        client._client = mock_httpx
        client._timeout_sec = 5.0
        await client.close()
        assert client._client is None

    @pytest.mark.asyncio
    async def test_post_json_retry_then_raise(self) -> None:
        client = mod._PersistentAsyncHTTPClient(
            scheme="http", host="localhost", port=1234, http2=False
        )
        first_client = AsyncMock()
        first_client.post.side_effect = httpx.ConnectError("fail-once")
        second_client = AsyncMock()
        second_client.post.side_effect = httpx.ConnectError("fail-twice")

        with (
            patch.object(
                client, "_new_client", side_effect=[first_client, second_client]
            ) as factory,
            pytest.raises(httpx.ConnectError, match="fail-twice"),
        ):
            await client.post_json(
                target="/v1/chat/completions",
                body=b"{}",
                headers={},
                timeout_sec=5.0,
            )

        assert factory.call_count == 2
        first_client.aclose.assert_awaited_once()
        second_client.aclose.assert_awaited_once()
        assert client._client is None

    @pytest.mark.asyncio
    async def test_get_text_retry_then_raise(self) -> None:
        client = mod._PersistentAsyncHTTPClient(
            scheme="http", host="localhost", port=1234, http2=False
        )
        first_client = AsyncMock()
        first_client.get.side_effect = httpx.ConnectError("fail-once")
        second_client = AsyncMock()
        second_client.get.side_effect = httpx.ConnectError("fail-twice")

        with (
            patch.object(
                client, "_new_client", side_effect=[first_client, second_client]
            ) as factory,
            pytest.raises(httpx.ConnectError, match="fail-twice"),
        ):
            await client.get_text(
                target="/v1/models",
                headers={},
                timeout_sec=5.0,
            )

        assert factory.call_count == 2
        first_client.aclose.assert_awaited_once()
        second_client.aclose.assert_awaited_once()
        assert client._client is None


# ---------------------------------------------------------------------------
# 3. Helper edge cases
# ---------------------------------------------------------------------------


class TestCoercePayloadBool:
    def test_int_truthy(self) -> None:
        assert mod._coerce_payload_bool(1) is True

    def test_int_zero(self) -> None:
        assert mod._coerce_payload_bool(0) is False

    def test_float_truthy(self) -> None:
        assert mod._coerce_payload_bool(1.0) is True

    def test_str_other_fails_closed(self) -> None:
        assert mod._coerce_payload_bool("something") is False

    def test_str_empty(self) -> None:
        assert mod._coerce_payload_bool("") is False

    def test_custom_object_fails_closed(self) -> None:
        assert mod._coerce_payload_bool(object()) is False


class TestChatHttpClientKey:
    def test_unsupported_scheme(self) -> None:
        with pytest.raises(ValueError, match="unsupported URL scheme"):
            mod._chat_http_client_key("ftp://localhost/v1", http2=False)

    def test_missing_host(self) -> None:
        with pytest.raises(ValueError, match="missing host"):
            mod._chat_http_client_key("http:///v1", http2=False)


class TestBaseUrlFromChatUrl:
    def test_non_chat_suffix(self) -> None:
        assert (
            mod._base_url_from_chat_url("http://localhost/v1/models")
            == "http://localhost/v1/models"
        )


class TestFirstNonemptyEnv:
    def test_no_match(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("NONEXISTENT_1", raising=False)
        monkeypatch.delenv("NONEXISTENT_2", raising=False)
        assert mod._first_nonempty_env("NONEXISTENT_1", "NONEXISTENT_2") == ""


class TestShouldUseProxy:
    def test_proxy_active(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(mod.urllib.request, "getproxies", lambda: {"http": "http://proxy:8080"})
        monkeypatch.setattr(mod.urllib.request, "proxy_bypass", lambda h: False)
        assert mod._should_use_proxy_transport("http://localhost:1234/v1") is True

    def test_proxy_bypassed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(mod.urllib.request, "getproxies", lambda: {"http": "http://proxy:8080"})
        monkeypatch.setattr(mod.urllib.request, "proxy_bypass", lambda h: True)
        assert mod._should_use_proxy_transport("http://localhost:1234/v1") is False


# ---------------------------------------------------------------------------
# 4. _iter_file_diffs edge cases: git headers
# ---------------------------------------------------------------------------


class TestExtractGitDiffBlocks:
    def test_diff_header_only_no_files(self) -> None:
        lines = ["diff --git a/foo.py b/foo.py", "some random text"]
        assert mod._iter_file_diffs(lines) == []

    def test_diff_header_only_no_plus(self) -> None:
        lines = [
            "diff --git a/foo.py b/foo.py",
            "--- a/foo.py",
            "some random text",
        ]
        assert mod._iter_file_diffs(lines) == []


# ---------------------------------------------------------------------------
# 5. _iter_file_diffs edge cases: plain unified headers
# ---------------------------------------------------------------------------


class TestNormalizeUnifiedBlocksEdges:
    def test_no_plus_plus_plus_after_minus(self) -> None:
        lines = ["--- a/foo.py", "some other line"]
        assert mod._iter_file_diffs(lines) == []

    def test_null_path_skipped(self) -> None:
        lines = ["--- a/", "+++ b/"]
        assert mod._iter_file_diffs(lines) == []

    def test_break_on_next_header(self) -> None:
        lines = [
            "--- a/foo.py",
            "+++ b/foo.py",
            "@@ -1,1 +1,1 @@",
            "-old",
            "+new",
            "--- a/bar.py",
            "+++ b/bar.py",
            "@@ -1,1 +1,1 @@",
            "-x",
            "+y",
        ]
        assert [path for path, _ in mod._iter_file_diffs(lines)] == ["foo.py", "bar.py"]


class TestCleanPathToken:
    def test_empty_string(self) -> None:
        assert mod._clean_path_token("") is None

    def test_whitespace_only(self) -> None:
        assert mod._clean_path_token("   ") is None

    def test_bare_prefix(self) -> None:
        assert mod._clean_path_token("a/") is None


# ---------------------------------------------------------------------------
# 6. _extract_unified_diff — empty input
# ---------------------------------------------------------------------------


class TestExtractUnifiedDiffEdges:
    def test_empty_text(self) -> None:
        assert mod._extract_unified_diff("   \n\n  ") is None


# ---------------------------------------------------------------------------
# 7. _emit_openai_transport_progress — callback exception
# ---------------------------------------------------------------------------


class TestEmitProgressCallbackException:
    def test_callback_raises_is_swallowed(self) -> None:
        def bad_callback(event: str, payload: Any) -> None:
            raise ValueError("boom")

        token = mod._OPENAI_PROGRESS_CALLBACK.set(bad_callback)
        try:
            # Should not raise
            mod._emit_openai_transport_progress("test_event", note="hi")
        finally:
            mod._OPENAI_PROGRESS_CALLBACK.reset(token)


# ---------------------------------------------------------------------------
# 8. _http_get_json — HTTP/2 fallback paths (sync)
# ---------------------------------------------------------------------------


class TestHttpGetJsonHttp2Fallback:
    def test_http2_protocol_error_falls_back_to_http1(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        call_count = 0

        class FakeClient:
            def __init__(self, **kw: Any) -> None:
                self.http2 = kw.get("http2", False)

            def __enter__(self) -> FakeClient:
                return self

            def __exit__(self, *a: Any) -> None:
                pass

            def get(self, url: str, headers: Any) -> SimpleNamespace:
                nonlocal call_count
                call_count += 1
                if self.http2:
                    raise httpx.RemoteProtocolError("http/2 protocol error")
                return SimpleNamespace(
                    status_code=200,
                    text='{"ok": true}',
                    http_version="HTTP/1.1",
                )

        monkeypatch.setattr(mod.httpx, "Client", FakeClient)
        data, meta = mod._http_get_json(
            url="http://localhost:8080/v1/models",
            api_key="k",
            timeout_sec=5.0,
            http2=True,
        )
        assert data == {"ok": True}
        assert meta.get("transport_http2_fallback_used") is True
        assert call_count == 2

    def test_http2_fallback_inner_value_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class FakeClient:
            def __init__(self, **kw: Any) -> None:
                self.http2 = kw.get("http2", False)

            def __enter__(self) -> FakeClient:
                return self

            def __exit__(self, *a: Any) -> None:
                pass

            def get(self, url: str, headers: Any) -> SimpleNamespace:
                if self.http2:
                    raise httpx.RemoteProtocolError("http/2 protocol error")
                raise ValueError("bad url")

        monkeypatch.setattr(mod.httpx, "Client", FakeClient)
        data, meta = mod._http_get_json(
            url="http://localhost:8080/v1/models",
            api_key="k",
            timeout_sec=5.0,
            http2=True,
        )
        assert data is None
        assert meta["error"] == "bad_request_url"

    def test_http2_fallback_inner_exception(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class FakeClient:
            def __init__(self, **kw: Any) -> None:
                self.http2 = kw.get("http2", False)

            def __enter__(self) -> FakeClient:
                return self

            def __exit__(self, *a: Any) -> None:
                pass

            def get(self, url: str, headers: Any) -> SimpleNamespace:
                if self.http2:
                    raise httpx.RemoteProtocolError("http/2 protocol error")
                raise OSError("connection failed")

        monkeypatch.setattr(mod.httpx, "Client", FakeClient)
        data, meta = mod._http_get_json(
            url="http://localhost:8080/v1/models",
            api_key="k",
            timeout_sec=5.0,
            http2=True,
        )
        assert data is None
        assert meta["error"] == "request_failed"
        assert meta["transport_http2_fallback_used"] is True

    def test_success_then_bad_json(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class FakeClient:
            def __init__(self, **kw: Any) -> None:
                pass

            def __enter__(self) -> FakeClient:
                return self

            def __exit__(self, *a: Any) -> None:
                pass

            def get(self, url: str, headers: Any) -> SimpleNamespace:
                return SimpleNamespace(
                    status_code=200, text="not json at all", http_version="HTTP/1.1"
                )

        monkeypatch.setattr(mod.httpx, "Client", FakeClient)
        data, meta = mod._http_get_json(
            url="http://localhost:8080/v1/models",
            api_key="k",
            timeout_sec=5.0,
            http2=False,
        )
        assert data is None
        assert meta["error"] == "bad_json_response"

    def test_non_dict_list_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class FakeClient:
            def __init__(self, **kw: Any) -> None:
                pass

            def __enter__(self) -> FakeClient:
                return self

            def __exit__(self, *a: Any) -> None:
                pass

            def get(self, url: str, headers: Any) -> SimpleNamespace:
                return SimpleNamespace(
                    status_code=200, text='"just a string"', http_version="HTTP/1.1"
                )

        monkeypatch.setattr(mod.httpx, "Client", FakeClient)
        data, meta = mod._http_get_json(
            url="http://localhost:8080/v1/models",
            api_key="k",
            timeout_sec=5.0,
            http2=False,
        )
        assert data is None
        assert meta["error"] == "bad_json_response"

    def test_http_error_status(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class FakeClient:
            def __init__(self, **kw: Any) -> None:
                pass

            def __enter__(self) -> FakeClient:
                return self

            def __exit__(self, *a: Any) -> None:
                pass

            def get(self, url: str, headers: Any) -> SimpleNamespace:
                return SimpleNamespace(
                    status_code=500, text="internal error", http_version="HTTP/1.1"
                )

        monkeypatch.setattr(mod.httpx, "Client", FakeClient)
        data, meta = mod._http_get_json(
            url="http://localhost:8080/v1/models",
            api_key="k",
            timeout_sec=5.0,
            http2=False,
        )
        assert data is None
        assert meta["error"] == "http_error"
        assert meta["status"] == 500


# ---------------------------------------------------------------------------
# 9. _http_get_json_async
# ---------------------------------------------------------------------------


class TestHttpGetJsonAsync:
    @pytest.mark.asyncio
    async def test_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        fake_resp = SimpleNamespace(
            status_code=200, text='{"result": true}', http_version="HTTP/1.1"
        )

        class FakeAsyncClient:
            def __init__(self, **kw: Any) -> None:
                pass

            async def __aenter__(self) -> FakeAsyncClient:
                return self

            async def __aexit__(self, *a: Any) -> None:
                pass

            async def get(self, url: str, headers: Any) -> SimpleNamespace:
                return fake_resp

        monkeypatch.setattr(mod.httpx, "AsyncClient", FakeAsyncClient)
        data, meta = await mod._http_get_json_async(
            url="http://localhost:8080/v1/models",
            api_key="k",
            timeout_sec=5.0,
            http2=False,
        )
        assert data == {"result": True}
        assert meta["transport_impl"] == "httpx"

    @pytest.mark.asyncio
    async def test_http_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class FakeAsyncClient:
            def __init__(self, **kw: Any) -> None:
                pass

            async def __aenter__(self) -> FakeAsyncClient:
                return self

            async def __aexit__(self, *a: Any) -> None:
                pass

            async def get(self, url: str, headers: Any) -> SimpleNamespace:
                return SimpleNamespace(
                    status_code=500, text="server error", http_version="HTTP/1.1"
                )

        monkeypatch.setattr(mod.httpx, "AsyncClient", FakeAsyncClient)
        data, meta = await mod._http_get_json_async(
            url="http://localhost:8080/v1/models",
            api_key="k",
            timeout_sec=5.0,
            http2=False,
        )
        assert data is None
        assert meta["error"] == "http_error"

    @pytest.mark.asyncio
    async def test_bad_json(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class FakeAsyncClient:
            def __init__(self, **kw: Any) -> None:
                pass

            async def __aenter__(self) -> FakeAsyncClient:
                return self

            async def __aexit__(self, *a: Any) -> None:
                pass

            async def get(self, url: str, headers: Any) -> SimpleNamespace:
                return SimpleNamespace(status_code=200, text="not-json", http_version="HTTP/1.1")

        monkeypatch.setattr(mod.httpx, "AsyncClient", FakeAsyncClient)
        data, meta = await mod._http_get_json_async(
            url="http://localhost:8080/v1/models",
            api_key="k",
            timeout_sec=5.0,
            http2=False,
        )
        assert data is None
        assert meta["error"] == "bad_json_response"

    @pytest.mark.asyncio
    async def test_non_dict_list_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class FakeAsyncClient:
            def __init__(self, **kw: Any) -> None:
                pass

            async def __aenter__(self) -> FakeAsyncClient:
                return self

            async def __aexit__(self, *a: Any) -> None:
                pass

            async def get(self, url: str, headers: Any) -> SimpleNamespace:
                return SimpleNamespace(status_code=200, text='"hello"', http_version="HTTP/1.1")

        monkeypatch.setattr(mod.httpx, "AsyncClient", FakeAsyncClient)
        data, meta = await mod._http_get_json_async(
            url="http://localhost:8080/v1/models",
            api_key="k",
            timeout_sec=5.0,
            http2=False,
        )
        assert data is None
        assert meta["error"] == "bad_json_response"

    @pytest.mark.asyncio
    async def test_http2_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class FakeAsyncClient:
            def __init__(self, **kw: Any) -> None:
                self.http2 = kw.get("http2", False)

            async def __aenter__(self) -> FakeAsyncClient:
                return self

            async def __aexit__(self, *a: Any) -> None:
                pass

            async def get(self, url: str, headers: Any) -> SimpleNamespace:
                if self.http2:
                    raise httpx.RemoteProtocolError("http/2 protocol error")
                return SimpleNamespace(
                    status_code=200, text='{"ok": true}', http_version="HTTP/1.1"
                )

        monkeypatch.setattr(mod.httpx, "AsyncClient", FakeAsyncClient)
        data, meta = await mod._http_get_json_async(
            url="http://localhost:8080/v1/models",
            api_key="k",
            timeout_sec=5.0,
            http2=True,
        )
        assert data == {"ok": True}
        assert meta["transport_http2_fallback_used"] is True

    @pytest.mark.asyncio
    async def test_http2_fallback_value_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class FakeAsyncClient:
            def __init__(self, **kw: Any) -> None:
                self.http2 = kw.get("http2", False)

            async def __aenter__(self) -> FakeAsyncClient:
                return self

            async def __aexit__(self, *a: Any) -> None:
                pass

            async def get(self, url: str, headers: Any) -> SimpleNamespace:
                if self.http2:
                    raise httpx.RemoteProtocolError("http/2 protocol error")
                raise ValueError("bad")

        monkeypatch.setattr(mod.httpx, "AsyncClient", FakeAsyncClient)
        data, meta = await mod._http_get_json_async(
            url="http://localhost:8080/v1/models",
            api_key="k",
            timeout_sec=5.0,
            http2=True,
        )
        assert data is None
        assert meta["error"] == "bad_request_url"

    @pytest.mark.asyncio
    async def test_http2_fallback_generic_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class FakeAsyncClient:
            def __init__(self, **kw: Any) -> None:
                self.http2 = kw.get("http2", False)

            async def __aenter__(self) -> FakeAsyncClient:
                return self

            async def __aexit__(self, *a: Any) -> None:
                pass

            async def get(self, url: str, headers: Any) -> SimpleNamespace:
                if self.http2:
                    raise httpx.RemoteProtocolError("http/2 protocol error")
                raise OSError("fail")

        monkeypatch.setattr(mod.httpx, "AsyncClient", FakeAsyncClient)
        data, meta = await mod._http_get_json_async(
            url="http://localhost:8080/v1/models",
            api_key="k",
            timeout_sec=5.0,
            http2=True,
        )
        assert data is None
        assert meta["error"] == "request_failed"
        assert meta["transport_http2_fallback_used"] is True

    @pytest.mark.asyncio
    async def test_non_http2_exception(self, monkeypatch: pytest.MonkeyPatch) -> None:
        class FakeAsyncClient:
            def __init__(self, **kw: Any) -> None:
                pass

            async def __aenter__(self) -> FakeAsyncClient:
                return self

            async def __aexit__(self, *a: Any) -> None:
                pass

            async def get(self, url: str, headers: Any) -> SimpleNamespace:
                raise OSError("connection refused")

        monkeypatch.setattr(mod.httpx, "AsyncClient", FakeAsyncClient)
        data, meta = await mod._http_get_json_async(
            url="http://localhost:8080/v1/models",
            api_key="k",
            timeout_sec=5.0,
            http2=False,
        )
        assert data is None
        assert meta["error"] == "request_failed"


# ---------------------------------------------------------------------------
# 10. _http_probe_ready / _http_probe_ready_async
# ---------------------------------------------------------------------------


class TestHttpProbeReady:
    def test_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(mod, "_should_use_proxy_transport", lambda url: False)
        fake_client = MagicMock()
        fake_client.get_text.return_value = (200, "ok")
        fake_client.last_http_version = "HTTP/1.1"
        monkeypatch.setattr(mod, "_get_thread_http_client", lambda **kw: fake_client)
        ready, meta = mod._http_probe_ready(
            url="http://localhost:1234/health",
            api_key="k",
            timeout_sec=5.0,
            http2=False,
        )
        assert ready is True
        assert meta["status"] == 200

    def test_non_2xx(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(mod, "_should_use_proxy_transport", lambda url: False)
        fake_client = MagicMock()
        fake_client.get_text.return_value = (500, "error")
        fake_client.last_http_version = "HTTP/1.1"
        monkeypatch.setattr(mod, "_get_thread_http_client", lambda **kw: fake_client)
        ready, meta = mod._http_probe_ready(
            url="http://localhost:1234/health",
            api_key="k",
            timeout_sec=5.0,
            http2=False,
        )
        assert ready is False
        assert meta["error"] == "http_error"

    def test_value_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(mod, "_should_use_proxy_transport", lambda url: False)
        fake_client = MagicMock()
        fake_client.get_text.side_effect = ValueError("bad")
        monkeypatch.setattr(mod, "_get_thread_http_client", lambda **kw: fake_client)
        ready, meta = mod._http_probe_ready(
            url="http://localhost:1234/health",
            api_key="k",
            timeout_sec=5.0,
            http2=False,
        )
        assert ready is None
        assert meta["error"] == "bad_request_url"

    def test_generic_exception(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(mod, "_should_use_proxy_transport", lambda url: False)
        fake_client = MagicMock()
        fake_client.get_text.side_effect = OSError("fail")
        monkeypatch.setattr(mod, "_get_thread_http_client", lambda **kw: fake_client)
        ready, meta = mod._http_probe_ready(
            url="http://localhost:1234/health",
            api_key="k",
            timeout_sec=5.0,
            http2=False,
        )
        assert ready is None
        assert meta["error"] == "request_failed"

    def test_http2_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(mod, "_should_use_proxy_transport", lambda url: False)
        call_count = 0

        def fake_get_client(**kw: Any) -> MagicMock:
            nonlocal call_count
            client = MagicMock()
            if kw.get("http2", False):
                client.get_text.side_effect = httpx.RemoteProtocolError("http/2 protocol error")
            else:
                client.get_text.return_value = (200, "ok")
                client.last_http_version = "HTTP/1.1"
            call_count += 1
            return client

        monkeypatch.setattr(mod, "_get_thread_http_client", fake_get_client)
        ready, meta = mod._http_probe_ready(
            url="http://localhost:1234/health",
            api_key="k",
            timeout_sec=5.0,
            http2=True,
        )
        assert ready is True
        assert meta.get("transport_http2_fallback_used") is True

    def test_http2_fallback_inner_value_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(mod, "_should_use_proxy_transport", lambda url: False)

        def fake_get_client(**kw: Any) -> MagicMock:
            client = MagicMock()
            if kw.get("http2", False):
                client.get_text.side_effect = httpx.RemoteProtocolError("http/2 protocol error")
            else:
                client.get_text.side_effect = ValueError("bad url")
            return client

        monkeypatch.setattr(mod, "_get_thread_http_client", fake_get_client)
        ready, meta = mod._http_probe_ready(
            url="http://localhost:1234/health",
            api_key="k",
            timeout_sec=5.0,
            http2=True,
        )
        assert ready is None
        assert meta["error"] == "bad_request_url"

    def test_http2_fallback_inner_generic_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(mod, "_should_use_proxy_transport", lambda url: False)

        def fake_get_client(**kw: Any) -> MagicMock:
            client = MagicMock()
            if kw.get("http2", False):
                client.get_text.side_effect = httpx.RemoteProtocolError("http/2 protocol error")
            else:
                client.get_text.side_effect = OSError("fail")
            return client

        monkeypatch.setattr(mod, "_get_thread_http_client", fake_get_client)
        ready, meta = mod._http_probe_ready(
            url="http://localhost:1234/health",
            api_key="k",
            timeout_sec=5.0,
            http2=True,
        )
        assert ready is None
        assert meta["error"] == "request_failed"
        assert meta["transport_http2_fallback_used"] is True

    def test_proxy_transport_path(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(mod, "_should_use_proxy_transport", lambda url: True)

        class FakeClient:
            def __init__(self, **kw: Any) -> None:
                pass

            def __enter__(self) -> FakeClient:
                return self

            def __exit__(self, *a: Any) -> None:
                pass

            def get(self, url: str, headers: Any) -> SimpleNamespace:
                return SimpleNamespace(status_code=200, text="ok", http_version="HTTP/1.1")

        monkeypatch.setattr(mod.httpx, "Client", FakeClient)
        ready, _meta = mod._http_probe_ready(
            url="http://localhost:1234/health",
            api_key="k",
            timeout_sec=5.0,
            http2=False,
        )
        assert ready is True


class TestHttpProbeReadyAsync:
    @pytest.mark.asyncio
    async def test_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(mod, "_should_use_proxy_transport", lambda url: False)
        fake_client = AsyncMock()
        fake_client.get_text = AsyncMock(return_value=(200, "ok"))
        fake_client.last_http_version = "HTTP/1.1"
        monkeypatch.setattr(mod, "_get_async_http_client", lambda **kw: fake_client)
        ready, _meta = await mod._http_probe_ready_async(
            url="http://localhost:1234/health",
            api_key="k",
            timeout_sec=5.0,
            http2=False,
        )
        assert ready is True

    @pytest.mark.asyncio
    async def test_success_with_real_persistent_async_client(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(mod, "_should_use_proxy_transport", lambda url: False)
        transport = AsyncMock()
        transport.get.return_value = SimpleNamespace(
            status_code=200,
            text="ok",
            http_version="HTTP/1.1",
        )
        client = mod._PersistentAsyncHTTPClient(
            scheme="http",
            host="localhost",
            port=1234,
            http2=False,
        )
        monkeypatch.setattr(client, "_new_client", lambda *, timeout_sec: transport)
        monkeypatch.setattr(mod, "_get_async_http_client", lambda **kw: client)

        ready, meta = await mod._http_probe_ready_async(
            url="http://localhost:1234/health",
            api_key="k",
            timeout_sec=5.0,
            http2=False,
        )

        assert ready is True
        assert meta["transport_http_version"] == "HTTP/1.1"
        transport.get.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_non_2xx(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(mod, "_should_use_proxy_transport", lambda url: False)
        fake_client = AsyncMock()
        fake_client.get_text = AsyncMock(return_value=(500, "err"))
        fake_client.last_http_version = "HTTP/1.1"
        monkeypatch.setattr(mod, "_get_async_http_client", lambda **kw: fake_client)
        ready, _meta = await mod._http_probe_ready_async(
            url="http://localhost:1234/health",
            api_key="k",
            timeout_sec=5.0,
            http2=False,
        )
        assert ready is False

    @pytest.mark.asyncio
    async def test_value_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(mod, "_should_use_proxy_transport", lambda url: False)
        fake_client = AsyncMock()
        fake_client.get_text = AsyncMock(side_effect=ValueError("bad"))
        monkeypatch.setattr(mod, "_get_async_http_client", lambda **kw: fake_client)
        _ready, meta = await mod._http_probe_ready_async(
            url="http://localhost:1234/health",
            api_key="k",
            timeout_sec=5.0,
            http2=False,
        )
        assert meta["error"] == "bad_request_url"

    @pytest.mark.asyncio
    async def test_generic_exception(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(mod, "_should_use_proxy_transport", lambda url: False)
        fake_client = AsyncMock()
        fake_client.get_text = AsyncMock(side_effect=OSError("fail"))
        monkeypatch.setattr(mod, "_get_async_http_client", lambda **kw: fake_client)
        _ready, meta = await mod._http_probe_ready_async(
            url="http://localhost:1234/health",
            api_key="k",
            timeout_sec=5.0,
            http2=False,
        )
        assert meta["error"] == "request_failed"

    @pytest.mark.asyncio
    async def test_http2_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(mod, "_should_use_proxy_transport", lambda url: False)

        def fake_get_client(**kw: Any) -> AsyncMock:
            client = AsyncMock()
            if kw.get("http2", False):
                client.get_text = AsyncMock(
                    side_effect=httpx.RemoteProtocolError("http/2 protocol error")
                )
            else:
                client.get_text = AsyncMock(return_value=(200, "ok"))
                client.last_http_version = "HTTP/1.1"
            return client

        monkeypatch.setattr(mod, "_get_async_http_client", fake_get_client)
        ready, _meta = await mod._http_probe_ready_async(
            url="http://localhost:1234/health",
            api_key="k",
            timeout_sec=5.0,
            http2=True,
        )
        assert ready is True

    @pytest.mark.asyncio
    async def test_http2_fallback_inner_value_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(mod, "_should_use_proxy_transport", lambda url: False)

        def fake_get_client(**kw: Any) -> AsyncMock:
            client = AsyncMock()
            if kw.get("http2", False):
                client.get_text = AsyncMock(
                    side_effect=httpx.RemoteProtocolError("http/2 protocol error")
                )
            else:
                client.get_text = AsyncMock(side_effect=ValueError("bad"))
            return client

        monkeypatch.setattr(mod, "_get_async_http_client", fake_get_client)
        _ready, meta = await mod._http_probe_ready_async(
            url="http://localhost:1234/health",
            api_key="k",
            timeout_sec=5.0,
            http2=True,
        )
        assert meta["error"] == "bad_request_url"

    @pytest.mark.asyncio
    async def test_http2_fallback_inner_generic_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(mod, "_should_use_proxy_transport", lambda url: False)

        def fake_get_client(**kw: Any) -> AsyncMock:
            client = AsyncMock()
            if kw.get("http2", False):
                client.get_text = AsyncMock(
                    side_effect=httpx.RemoteProtocolError("http/2 protocol error")
                )
            else:
                client.get_text = AsyncMock(side_effect=OSError("fail"))
            return client

        monkeypatch.setattr(mod, "_get_async_http_client", fake_get_client)
        _ready, meta = await mod._http_probe_ready_async(
            url="http://localhost:1234/health",
            api_key="k",
            timeout_sec=5.0,
            http2=True,
        )
        assert meta["error"] == "request_failed"
        assert meta["transport_http2_fallback_used"] is True


# ---------------------------------------------------------------------------
# 11. _probe_context_window_tokens_async
# ---------------------------------------------------------------------------


class TestProbeContextWindowTokensAsync:
    @pytest.mark.asyncio
    async def test_first_probe_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def fake_get_json_async(**kw: Any) -> tuple[Any, dict[str, Any]]:
            if "api/v1" in kw["url"]:
                return {"data": [{"id": "m1", "context_length": 4096}]}, {"transport_impl": "httpx"}
            return None, {"error": "request_failed"}

        monkeypatch.setattr(mod, "_http_get_json_async", fake_get_json_async)
        tokens, _meta = await mod._probe_context_window_tokens_async(
            base_url="http://localhost:1234/v1",
            model="m1",
            api_key="k",
            timeout_sec=5.0,
            http2=False,
        )
        assert tokens == 4096

    @pytest.mark.asyncio
    async def test_all_fail(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def fake_get_json_async(**kw: Any) -> tuple[None, dict[str, Any]]:
            return None, {"error": "request_failed"}

        monkeypatch.setattr(mod, "_http_get_json_async", fake_get_json_async)
        tokens, meta = await mod._probe_context_window_tokens_async(
            base_url="http://localhost:1234/v1",
            model="m1",
            api_key="k",
            timeout_sec=5.0,
            http2=False,
        )
        assert tokens is None
        assert meta["error"] == "context_window_probe_failed"

    @pytest.mark.asyncio
    async def test_model_not_in_response(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def fake_get_json_async(**kw: Any) -> tuple[Any, dict[str, Any]]:
            return {"data": [{"id": "other-model", "context_length": 4096}]}, {
                "transport_impl": "httpx"
            }

        monkeypatch.setattr(mod, "_http_get_json_async", fake_get_json_async)
        tokens, _meta = await mod._probe_context_window_tokens_async(
            base_url="http://localhost:1234/v1",
            model="m1",
            api_key="k",
            timeout_sec=5.0,
            http2=False,
        )
        assert tokens is None

    @pytest.mark.asyncio
    async def test_duplicate_openai_models_probe_url_is_skipped(
        self,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        calls: list[str] = []

        async def fake_get_json_async(**kw: Any) -> tuple[None, dict[str, Any]]:
            calls.append(str(kw["url"]))
            return None, {"error": "request_failed"}

        monkeypatch.setattr(mod, "_http_get_json_async", fake_get_json_async)
        tokens, meta = await mod._probe_context_window_tokens_async(
            base_url="http://localhost:1234/api/v1",
            model="m1",
            api_key="k",
            timeout_sec=5.0,
            http2=False,
        )
        assert tokens is None
        assert calls == [
            "http://localhost:1234/api/v1/models",
            "http://localhost:1234/api/v0/models",
        ]
        assert meta["error"] == "context_window_probe_failed"


class TestExtractModelEntries:
    def test_non_list_payload_returns_empty(self) -> None:
        assert mod._extract_model_entries({"data": "not-a-list"}) == []


# ---------------------------------------------------------------------------
# 12. _probe_server_ready_async
# ---------------------------------------------------------------------------


class TestProbeServerReadyAsync:
    @pytest.mark.asyncio
    async def test_health_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def fake_probe(**kw: Any) -> tuple[bool, dict[str, Any]]:
            if "/health" in kw["url"]:
                return True, {"status": 200, "transport_impl": "httpx"}
            return False, {"error": "request_failed"}

        monkeypatch.setattr(mod, "_http_probe_ready_async", fake_probe)
        ready, _meta = await mod._probe_server_ready_async(
            base_url="http://localhost:1234/v1",
            api_key="k",
            timeout_sec=5.0,
            http2=False,
        )
        assert ready is True

    @pytest.mark.asyncio
    async def test_all_fail(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def fake_probe(**kw: Any) -> tuple[bool, dict[str, Any]]:
            return False, {"error": "request_failed"}

        monkeypatch.setattr(mod, "_http_probe_ready_async", fake_probe)
        ready, meta = await mod._probe_server_ready_async(
            base_url="http://localhost:1234/v1",
            api_key="k",
            timeout_sec=5.0,
            http2=False,
        )
        assert ready is False
        assert meta["error"] == "server_ready_probe_failed"


# ---------------------------------------------------------------------------
# 13. _chat_completion_async
# ---------------------------------------------------------------------------


class TestChatCompletionAsync:
    @pytest.mark.asyncio
    async def test_basic_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(mod, "_should_use_proxy_transport", lambda url: False)
        fake_client = AsyncMock()
        fake_client.post_json = AsyncMock(
            return_value=(
                200,
                json.dumps(
                    _ok_response(
                        "hello",
                        usage={"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
                    )
                ),
            )
        )
        fake_client.last_http_version = "HTTP/1.1"
        monkeypatch.setattr(mod, "_get_async_http_client", lambda **kw: fake_client)
        data, _meta = await mod._chat_completion_async(
            url="http://localhost:1234/v1/chat/completions",
            api_key="k",
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.0,
            max_tokens=100,
            timeout_sec=5.0,
            seed=None,
            http2=False,
        )
        assert data is not None
        assert data["choices"][0]["message"]["content"] == "hello"

    @pytest.mark.asyncio
    async def test_with_seed(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(mod, "_should_use_proxy_transport", lambda url: False)
        resp = _ok_response("hi")
        resp["seed"] = 42
        fake_client = AsyncMock()
        fake_client.post_json = AsyncMock(return_value=(200, json.dumps(resp)))
        fake_client.last_http_version = "HTTP/1.1"
        monkeypatch.setattr(mod, "_get_async_http_client", lambda **kw: fake_client)
        _data, meta = await mod._chat_completion_async(
            url="http://localhost:1234/v1/chat/completions",
            api_key="k",
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.0,
            max_tokens=100,
            timeout_sec=5.0,
            seed=42,
            http2=False,
        )
        assert meta.get("api_compat_seed_acknowledged") is True
        assert meta.get("api_compat_seed_response") == 42

    @pytest.mark.asyncio
    async def test_with_extra_payload(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(mod, "_should_use_proxy_transport", lambda url: False)
        fake_client = AsyncMock()
        fake_client.post_json = AsyncMock(return_value=(200, json.dumps(_ok_response("hi"))))
        fake_client.last_http_version = "HTTP/1.1"
        monkeypatch.setattr(mod, "_get_async_http_client", lambda **kw: fake_client)
        data, _meta = await mod._chat_completion_async(
            url="http://localhost:1234/v1/chat/completions",
            api_key="k",
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.0,
            max_tokens=100,
            timeout_sec=5.0,
            seed=None,
            http2=False,
            extra_payload={"response_format": {"type": "json_object"}},
        )
        assert data is not None

    @pytest.mark.asyncio
    async def test_bad_json_response(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(mod, "_should_use_proxy_transport", lambda url: False)
        fake_client = AsyncMock()
        fake_client.post_json = AsyncMock(return_value=(200, "NOT JSON"))
        fake_client.last_http_version = "HTTP/1.1"
        monkeypatch.setattr(mod, "_get_async_http_client", lambda **kw: fake_client)
        data, meta = await mod._chat_completion_async(
            url="http://localhost:1234/v1/chat/completions",
            api_key="k",
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.0,
            max_tokens=100,
            timeout_sec=5.0,
            seed=None,
            http2=False,
        )
        assert data is None
        assert meta["error"] == "bad_json_response"

    @pytest.mark.asyncio
    async def test_http_error_non_retryable(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(mod, "_should_use_proxy_transport", lambda url: False)
        fake_client = AsyncMock()
        fake_client.post_json = AsyncMock(return_value=(400, "bad request"))
        fake_client.last_http_version = "HTTP/1.1"
        monkeypatch.setattr(mod, "_get_async_http_client", lambda **kw: fake_client)
        data, meta = await mod._chat_completion_async(
            url="http://localhost:1234/v1/chat/completions",
            api_key="k",
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.0,
            max_tokens=100,
            timeout_sec=5.0,
            seed=None,
            http2=False,
            transport_max_retries=0,
        )
        assert data is None
        assert meta["error"] == "http_error"

    @pytest.mark.asyncio
    async def test_budget_exhausted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import time

        monkeypatch.setattr(mod, "_should_use_proxy_transport", lambda url: False)
        data, meta = await mod._chat_completion_async(
            url="http://localhost:1234/v1/chat/completions",
            api_key="k",
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.0,
            max_tokens=100,
            timeout_sec=5.0,
            seed=None,
            http2=False,
            deadline_monotonic=time.monotonic() - 10,  # already expired
        )
        assert data is None
        assert meta["error"] == "request_budget_exceeded"

    @pytest.mark.asyncio
    async def test_retries_then_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(mod, "_should_use_proxy_transport", lambda url: False)
        monkeypatch.setattr(mod.asyncio, "sleep", AsyncMock())
        call_count = 0

        def make_client(**kw: Any) -> AsyncMock:
            nonlocal call_count
            client = AsyncMock()

            async def post(*a: Any, **k: Any) -> tuple[int, str]:
                nonlocal call_count
                call_count += 1
                if call_count == 1:
                    return 429, "rate limited"
                return 200, json.dumps(_ok_response("ok"))

            client.post_json = post
            client.last_http_version = "HTTP/1.1"
            return client

        monkeypatch.setattr(mod, "_get_async_http_client", make_client)
        data, _meta = await mod._chat_completion_async(
            url="http://localhost:1234/v1/chat/completions",
            api_key="k",
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.0,
            max_tokens=100,
            timeout_sec=5.0,
            seed=None,
            http2=False,
            transport_max_retries=2,
            transport_retry_backoff_base_sec=0.0,
            transport_retry_backoff_max_sec=0.0,
        )
        assert data is not None
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_reasoning_content_extracted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(mod, "_should_use_proxy_transport", lambda url: False)
        resp = {
            "choices": [
                {
                    "message": {
                        "content": "answer",
                        "reasoning_content": "thinking...",
                        "reasoning": "detailed thought",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": {
                **_GOOD_USAGE,
                "output_tokens_details": {"reasoning_tokens": 4},
            },
        }
        fake_client = AsyncMock()
        fake_client.post_json = AsyncMock(return_value=(200, json.dumps(resp)))
        fake_client.last_http_version = "HTTP/1.1"
        monkeypatch.setattr(mod, "_get_async_http_client", lambda **kw: fake_client)
        _data, meta = await mod._chat_completion_async(
            url="http://localhost:1234/v1/chat/completions",
            api_key="k",
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.0,
            max_tokens=100,
            timeout_sec=5.0,
            seed=None,
            http2=False,
        )
        assert meta.get("reasoning_content_present") is True
        assert meta.get("reasoning_present") is True
        assert meta.get("usage", {}).get("reasoning_tokens") == 4

    @pytest.mark.asyncio
    async def test_backoff_max_adjusted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When backoff_max < backoff_base, it gets adjusted upward."""
        monkeypatch.setattr(mod, "_should_use_proxy_transport", lambda url: False)
        fake_client = AsyncMock()
        fake_client.post_json = AsyncMock(return_value=(200, json.dumps(_ok_response("ok"))))
        fake_client.last_http_version = "HTTP/1.1"
        monkeypatch.setattr(mod, "_get_async_http_client", lambda **kw: fake_client)
        data, _meta = await mod._chat_completion_async(
            url="http://localhost:1234/v1/chat/completions",
            api_key="k",
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.0,
            max_tokens=100,
            timeout_sec=5.0,
            seed=None,
            http2=False,
            transport_retry_backoff_base_sec=5.0,
            transport_retry_backoff_max_sec=1.0,  # less than base
        )
        assert data is not None


# ---------------------------------------------------------------------------
# 14. _propose_patch_via_diff_async
# ---------------------------------------------------------------------------


class TestProposePatchViaDiffAsync:
    @pytest.mark.asyncio
    async def test_valid_diff(self, monkeypatch: pytest.MonkeyPatch) -> None:
        diff_text = (
            "diff --git a/foo.py b/foo.py\n"
            "--- a/foo.py\n"
            "+++ b/foo.py\n"
            "@@ -1,1 +1,1 @@\n"
            "-old\n"
            "+new\n"
        )

        async def fake_chat_async(**kw: Any) -> tuple[dict[str, Any], dict[str, Any]]:
            return _ok_response(diff_text), dict(_GOOD_META)

        monkeypatch.setattr(mod, "_chat_completion_async", fake_chat_async)
        result = await mod._propose_patch_via_diff_async(
            url="http://localhost:1234/v1/chat/completions",
            api_key="k",
            model="m",
            prompt="fix it",
            temperature=0.0,
            max_tokens=100,
            timeout_sec=5.0,
            seed=None,
            http2=False,
        )
        assert result.diff is not None
        assert "foo.py" in result.diff

    @pytest.mark.asyncio
    async def test_no_diff_found(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def fake_chat_async(**kw: Any) -> tuple[dict[str, Any], dict[str, Any]]:
            return _ok_response("I don't know how to fix this."), dict(_GOOD_META)

        monkeypatch.setattr(mod, "_chat_completion_async", fake_chat_async)
        result = await mod._propose_patch_via_diff_async(
            url="http://localhost:1234/v1/chat/completions",
            api_key="k",
            model="m",
            prompt="fix it",
            temperature=0.0,
            max_tokens=100,
            timeout_sec=5.0,
            seed=None,
            http2=False,
        )
        assert result.diff is None

    @pytest.mark.asyncio
    async def test_no_choice_content(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def fake_chat_async(**kw: Any) -> tuple[dict[str, Any], dict[str, Any]]:
            return {"choices": [{"message": {}}]}, dict(_GOOD_META)

        monkeypatch.setattr(mod, "_chat_completion_async", fake_chat_async)
        result = await mod._propose_patch_via_diff_async(
            url="http://localhost:1234/v1/chat/completions",
            api_key="k",
            model="m",
            prompt="fix it",
            temperature=0.0,
            max_tokens=100,
            timeout_sec=5.0,
            seed=None,
            http2=False,
        )
        assert result.diff is None
        assert result.meta is not None
        assert result.meta.get("error") == "no_choice_content"

    @pytest.mark.asyncio
    async def test_budget_exhausted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def fake_chat_async(**kw: Any) -> tuple[None, dict[str, Any]]:
            return None, {"error": "request_budget_exceeded", "llm_call_count": 1}

        monkeypatch.setattr(mod, "_chat_completion_async", fake_chat_async)
        result = await mod._propose_patch_via_diff_async(
            url="http://localhost:1234/v1/chat/completions",
            api_key="k",
            model="m",
            prompt="fix it",
            temperature=0.0,
            max_tokens=100,
            timeout_sec=5.0,
            seed=None,
            http2=False,
        )
        assert result.diff is None

    @pytest.mark.asyncio
    async def test_all_attempts_fail_returns_unknown(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def fake_chat_async(**kw: Any) -> tuple[None, dict[str, Any]]:
            return None, {"error": "request_failed", "llm_call_count": 1}

        monkeypatch.setattr(mod, "_chat_completion_async", fake_chat_async)
        result = await mod._propose_patch_via_diff_async(
            url="http://localhost:1234/v1/chat/completions",
            api_key="k",
            model="m",
            prompt="fix it",
            temperature=0.0,
            max_tokens=100,
            timeout_sec=5.0,
            seed=None,
            http2=False,
        )
        assert result.diff is None


# ---------------------------------------------------------------------------
# 15. _propose_patch_via_edits_async
# ---------------------------------------------------------------------------


class TestProposePatchViaEditsAsync:
    @pytest.mark.asyncio
    async def test_successful_edit(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        src = tmp_path / "foo.py"
        src.write_text("old line\n", encoding="utf-8")
        edits_json = json.dumps(
            {"edits": [{"path": "foo.py", "find": "old line", "replace": "new line"}]}
        )

        async def fake_chat_async(**kw: Any) -> tuple[dict[str, Any], dict[str, Any]]:
            return _ok_response(edits_json), dict(_GOOD_META)

        monkeypatch.setattr(mod, "_chat_completion_async", fake_chat_async)
        result = await mod._propose_patch_via_edits_async(
            url="http://localhost:1234/v1/chat/completions",
            api_key="k",
            model="m",
            prompt="fix it",
            cwd=tmp_path,
            temperature=0.0,
            max_tokens=100,
            timeout_sec=5.0,
            seed=None,
            http2=False,
        )
        assert result.diff is not None
        assert "foo.py" in result.diff

    @pytest.mark.asyncio
    async def test_no_json_fallback_to_diff(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        call_count = 0

        async def fake_chat_async(**kw: Any) -> tuple[dict[str, Any] | None, dict[str, Any]]:
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                # Edits stages fail: no JSON
                return _ok_response("not json at all"), dict(_GOOD_META)
            # Diff fallback
            diff = (
                "diff --git a/foo.py b/foo.py\n"
                "--- a/foo.py\n+++ b/foo.py\n"
                "@@ -1,1 +1,1 @@\n-old\n+new\n"
            )
            return _ok_response(diff), dict(_GOOD_META)

        monkeypatch.setattr(mod, "_chat_completion_async", fake_chat_async)
        result = await mod._propose_patch_via_edits_async(
            url="http://localhost:1234/v1/chat/completions",
            api_key="k",
            model="m",
            prompt="fix it",
            cwd=tmp_path,
            temperature=0.0,
            max_tokens=100,
            timeout_sec=5.0,
            seed=None,
            http2=False,
        )
        assert result.diff is not None
        assert result.meta is not None
        assert result.meta.get("fallback_from") == "edits"

    @pytest.mark.asyncio
    async def test_no_json_no_diff_fallback_allowed(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        async def fake_chat_async(**kw: Any) -> tuple[dict[str, Any], dict[str, Any]]:
            return _ok_response("no json here"), dict(_GOOD_META)

        monkeypatch.setattr(mod, "_chat_completion_async", fake_chat_async)
        result = await mod._propose_patch_via_edits_async(
            url="http://localhost:1234/v1/chat/completions",
            api_key="k",
            model="m",
            prompt="fix it",
            cwd=tmp_path,
            temperature=0.0,
            max_tokens=100,
            timeout_sec=5.0,
            seed=None,
            http2=False,
            allow_diff_fallback=False,
        )
        assert result.diff is None

    @pytest.mark.asyncio
    async def test_context_exceeded_no_fallback(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        async def fake_chat_async(**kw: Any) -> tuple[None, dict[str, Any]]:
            return None, {
                "error": "http_error",
                "status": 400,
                "body": "prompt is too long",
                "llm_call_count": 1,
            }

        monkeypatch.setattr(mod, "_chat_completion_async", fake_chat_async)
        result = await mod._propose_patch_via_edits_async(
            url="http://localhost:1234/v1/chat/completions",
            api_key="k",
            model="m",
            prompt="fix it",
            cwd=tmp_path,
            temperature=0.0,
            max_tokens=100,
            timeout_sec=5.0,
            seed=None,
            http2=False,
        )
        assert result.diff is None

    @pytest.mark.asyncio
    async def test_find_not_found_fallback(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        src = tmp_path / "foo.py"
        src.write_text("original content\n", encoding="utf-8")
        edits_json = json.dumps(
            {"edits": [{"path": "foo.py", "find": "nonexistent", "replace": "new"}]}
        )
        call_count = 0

        async def fake_chat_async(**kw: Any) -> tuple[dict[str, Any], dict[str, Any]]:
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                return _ok_response(edits_json), dict(_GOOD_META)
            diff = (
                "diff --git a/foo.py b/foo.py\n"
                "--- a/foo.py\n+++ b/foo.py\n"
                "@@ -1,1 +1,1 @@\n-original content\n+fixed content\n"
            )
            return _ok_response(diff), dict(_GOOD_META)

        monkeypatch.setattr(mod, "_chat_completion_async", fake_chat_async)
        result = await mod._propose_patch_via_edits_async(
            url="http://localhost:1234/v1/chat/completions",
            api_key="k",
            model="m",
            prompt="fix it",
            cwd=tmp_path,
            temperature=0.0,
            max_tokens=100,
            timeout_sec=5.0,
            seed=None,
            http2=False,
        )
        assert result.diff is not None

    @pytest.mark.asyncio
    async def test_file_not_found(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        edits_json = json.dumps({"edits": [{"path": "missing.py", "find": "x", "replace": "y"}]})

        async def fake_chat_async(**kw: Any) -> tuple[dict[str, Any], dict[str, Any]]:
            return _ok_response(edits_json), dict(_GOOD_META)

        monkeypatch.setattr(mod, "_chat_completion_async", fake_chat_async)
        result = await mod._propose_patch_via_edits_async(
            url="http://localhost:1234/v1/chat/completions",
            api_key="k",
            model="m",
            prompt="fix it",
            cwd=tmp_path,
            temperature=0.0,
            max_tokens=100,
            timeout_sec=5.0,
            seed=None,
            http2=False,
        )
        assert result.diff is None
        assert result.meta is not None
        assert result.meta.get("error") == "file_not_found"

    @pytest.mark.asyncio
    async def test_bad_path(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        edits_json = json.dumps({"edits": [{"path": "", "find": "x", "replace": "y"}]})

        async def fake_chat_async(**kw: Any) -> tuple[dict[str, Any], dict[str, Any]]:
            return _ok_response(edits_json), dict(_GOOD_META)

        monkeypatch.setattr(mod, "_chat_completion_async", fake_chat_async)
        result = await mod._propose_patch_via_edits_async(
            url="http://localhost:1234/v1/chat/completions",
            api_key="k",
            model="m",
            prompt="fix it",
            cwd=tmp_path,
            temperature=0.0,
            max_tokens=100,
            timeout_sec=5.0,
            seed=None,
            http2=False,
        )
        assert result.diff is None
        assert result.meta.get("error") == "bad_path"

    @pytest.mark.asyncio
    async def test_bad_find_replace(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        src = tmp_path / "foo.py"
        src.write_text("content\n", encoding="utf-8")
        edits_json = json.dumps({"edits": [{"path": "foo.py", "find": 123, "replace": "y"}]})

        async def fake_chat_async(**kw: Any) -> tuple[dict[str, Any], dict[str, Any]]:
            return _ok_response(edits_json), dict(_GOOD_META)

        monkeypatch.setattr(mod, "_chat_completion_async", fake_chat_async)
        result = await mod._propose_patch_via_edits_async(
            url="http://localhost:1234/v1/chat/completions",
            api_key="k",
            model="m",
            prompt="fix it",
            cwd=tmp_path,
            temperature=0.0,
            max_tokens=100,
            timeout_sec=5.0,
            seed=None,
            http2=False,
        )
        assert result.meta.get("error") == "bad_find_replace"

    @pytest.mark.asyncio
    async def test_no_effective_changes(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        src = tmp_path / "foo.py"
        src.write_text("same\n", encoding="utf-8")
        edits_json = json.dumps({"edits": [{"path": "foo.py", "find": "same", "replace": "same"}]})
        call_count = 0

        async def fake_chat_async(**kw: Any) -> tuple[dict[str, Any] | None, dict[str, Any]]:
            nonlocal call_count
            call_count += 1
            if call_count <= 1:
                return _ok_response(edits_json), dict(_GOOD_META)
            return None, {"error": "request_failed", "llm_call_count": 1}

        monkeypatch.setattr(mod, "_chat_completion_async", fake_chat_async)
        result = await mod._propose_patch_via_edits_async(
            url="http://localhost:1234/v1/chat/completions",
            api_key="k",
            model="m",
            prompt="fix it",
            cwd=tmp_path,
            temperature=0.0,
            max_tokens=100,
            timeout_sec=5.0,
            seed=None,
            http2=False,
        )
        assert result.diff is None

    @pytest.mark.asyncio
    async def test_missing_edits_key_fallback(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        call_count = 0

        async def fake_chat_async(**kw: Any) -> tuple[dict[str, Any] | None, dict[str, Any]]:
            nonlocal call_count
            call_count += 1
            if call_count <= 3:
                return _ok_response(json.dumps({"something": "else"})), dict(_GOOD_META)
            return None, {"error": "request_failed", "llm_call_count": 1}

        monkeypatch.setattr(mod, "_chat_completion_async", fake_chat_async)
        result = await mod._propose_patch_via_edits_async(
            url="http://localhost:1234/v1/chat/completions",
            api_key="k",
            model="m",
            prompt="fix it",
            cwd=tmp_path,
            temperature=0.0,
            max_tokens=100,
            timeout_sec=5.0,
            seed=None,
            http2=False,
        )
        assert result.diff is None

    @pytest.mark.asyncio
    async def test_reasoning_content_tracked(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        src = tmp_path / "foo.py"
        src.write_text("old\n", encoding="utf-8")
        resp = {
            "choices": [
                {
                    "message": {
                        "content": json.dumps(
                            {"edits": [{"path": "foo.py", "find": "old", "replace": "new"}]}
                        ),
                        "reasoning_content": "thinking about it",
                        "reasoning": "deep thought",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": _GOOD_USAGE,
        }

        async def fake_chat_async(**kw: Any) -> tuple[dict[str, Any], dict[str, Any]]:
            return resp, dict(_GOOD_META)

        monkeypatch.setattr(mod, "_chat_completion_async", fake_chat_async)
        result = await mod._propose_patch_via_edits_async(
            url="http://localhost:1234/v1/chat/completions",
            api_key="k",
            model="m",
            prompt="fix it",
            cwd=tmp_path,
            temperature=0.0,
            max_tokens=100,
            timeout_sec=5.0,
            seed=None,
            http2=False,
        )
        assert result.diff is not None
        assert result.meta.get("reasoning_content_present") is True


# ---------------------------------------------------------------------------
# 16. _propose_patch_via_full_file_async
# ---------------------------------------------------------------------------


class TestProposePatchViaFullFileAsync:
    @pytest.mark.asyncio
    async def test_success_single_file(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        src = tmp_path / "buggy.py"
        src.write_text("def f():\n    return 1\n", encoding="utf-8")

        async def fake_chat_async(**kw: Any) -> tuple[dict[str, Any], dict[str, Any]]:
            return _ok_response("def f():\n    return 2\n"), dict(_GOOD_META)

        monkeypatch.setattr(mod, "_chat_completion_async", fake_chat_async)
        result = await mod._propose_patch_via_full_file_async(
            url="http://localhost:1234/v1/chat/completions",
            api_key="k",
            model="m",
            prompt="fix it",
            cwd=tmp_path,
            temperature=0.0,
            max_tokens=100,
            timeout_sec=5.0,
            seed=None,
            http2=False,
        )
        assert result.diff is not None
        assert "buggy.py" in result.diff

    @pytest.mark.asyncio
    async def test_unsupported_shape_fallback(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        # Multiple py files → unsupported
        (tmp_path / "a.py").write_text("a\n", encoding="utf-8")
        (tmp_path / "b.py").write_text("b\n", encoding="utf-8")
        diff = "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1,1 +1,1 @@\n-a\n+c\n"

        async def fake_chat_async(**kw: Any) -> tuple[dict[str, Any], dict[str, Any]]:
            return _ok_response(diff), dict(_GOOD_META)

        monkeypatch.setattr(mod, "_chat_completion_async", fake_chat_async)
        result = await mod._propose_patch_via_full_file_async(
            url="http://localhost:1234/v1/chat/completions",
            api_key="k",
            model="m",
            prompt="fix it",
            cwd=tmp_path,
            temperature=0.0,
            max_tokens=100,
            timeout_sec=5.0,
            seed=None,
            http2=False,
        )
        # Falls back to diff mode
        assert result.meta is not None
        assert result.meta.get("fallback_from") == "full_file"


# ---------------------------------------------------------------------------
# 17. OpenAICompatPatcher async public methods
# ---------------------------------------------------------------------------


class TestPatcherAsyncProposePatch:
    @pytest.mark.asyncio
    async def test_edits_mode(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        src = tmp_path / "foo.py"
        src.write_text("old\n", encoding="utf-8")
        json.dumps({"edits": [{"path": "foo.py", "find": "old", "replace": "new"}]})

        async def fake_edits(**kw: Any) -> PatchResult:
            return PatchResult(diff="diff --git ...", meta=dict(_GOOD_META))

        monkeypatch.setattr(mod, "_propose_patch_via_edits_async", fake_edits)
        patcher = _make_patcher(response_mode="edits")
        result = await patcher.apropose_patch(prompt="fix it", context_dir=tmp_path, cwd=tmp_path)
        assert result.diff is not None

    @pytest.mark.asyncio
    async def test_full_file_mode(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        async def fake_full(**kw: Any) -> PatchResult:
            return PatchResult(diff="diff --git ...", meta=dict(_GOOD_META))

        monkeypatch.setattr(mod, "_propose_patch_via_full_file_async", fake_full)
        patcher = _make_patcher(response_mode="full_file")
        result = await patcher.apropose_patch(prompt="fix it", context_dir=tmp_path, cwd=tmp_path)
        assert result.diff is not None

    @pytest.mark.asyncio
    async def test_diff_mode_fallback_to_edits(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        call_count = 0

        async def fake_diff(**kw: Any) -> PatchResult:
            return PatchResult(diff=None, meta={"error": "no_diff_found", "llm_call_count": 1})

        async def fake_edits(**kw: Any) -> PatchResult:
            nonlocal call_count
            call_count += 1
            return PatchResult(diff="diff --git a/x b/x\n...", meta=dict(_GOOD_META))

        monkeypatch.setattr(mod, "_propose_patch_via_diff_async", fake_diff)
        monkeypatch.setattr(mod, "_propose_patch_via_edits_async", fake_edits)
        patcher = _make_patcher(response_mode="diff")
        result = await patcher.apropose_patch(prompt="fix it", context_dir=tmp_path, cwd=tmp_path)
        assert result.diff is not None
        assert result.meta.get("fallback_from") == "diff"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_diff_mode_no_fallback_needed(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        async def fake_diff(**kw: Any) -> PatchResult:
            return PatchResult(diff="diff --git a/x b/x\n...", meta=dict(_GOOD_META))

        monkeypatch.setattr(mod, "_propose_patch_via_diff_async", fake_diff)
        patcher = _make_patcher(response_mode="diff")
        result = await patcher.apropose_patch(prompt="fix it", context_dir=tmp_path, cwd=tmp_path)
        assert result.diff is not None

    @pytest.mark.asyncio
    async def test_force_patch_attempt_success(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        src = tmp_path / "buggy.py"
        src.write_text("def f():\n    pass\n", encoding="utf-8")

        async def fake_edits(**kw: Any) -> PatchResult:
            return PatchResult(diff=None, meta={"error": "no_json_payload", "llm_call_count": 1})

        monkeypatch.setattr(mod, "_propose_patch_via_edits_async", fake_edits)
        patcher = _make_patcher(response_mode="edits", force_patch_attempt=True)
        result = await patcher.apropose_patch(
            prompt="Available files\nbuggy.py\n\nfix it",
            context_dir=tmp_path,
            cwd=tmp_path,
        )
        assert result.meta.get("forced_patch_attempt") is True

    @pytest.mark.asyncio
    async def test_force_patch_skipped_on_transport_failure(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        async def fake_edits(**kw: Any) -> PatchResult:
            return PatchResult(diff=None, meta={"error": "request_failed", "llm_call_count": 1})

        monkeypatch.setattr(mod, "_propose_patch_via_edits_async", fake_edits)
        patcher = _make_patcher(response_mode="edits", force_patch_attempt=True)
        result = await patcher.apropose_patch(prompt="fix it", context_dir=tmp_path, cwd=tmp_path)
        assert result.meta.get("forced_patch_skipped") is True


class TestPatcherAsyncProbe:
    @pytest.mark.asyncio
    async def test_aprobe_context_window_tokens(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def fake_probe(**kw: Any) -> tuple[int | None, dict[str, Any]]:
            return 8192, {"context_window_source": "test"}

        monkeypatch.setattr(mod, "_probe_context_window_tokens_async", fake_probe)
        patcher = _make_patcher()
        tokens, _meta = await patcher.aprobe_context_window_tokens()
        assert tokens == 8192

    @pytest.mark.asyncio
    async def test_aprobe_server_ready(self, monkeypatch: pytest.MonkeyPatch) -> None:
        async def fake_probe(**kw: Any) -> tuple[bool, dict[str, Any]]:
            return True, {"server_ready": True}

        monkeypatch.setattr(mod, "_probe_server_ready_async", fake_probe)
        patcher = _make_patcher()
        ready, _meta = await patcher.aprobe_server_ready()
        assert ready is True


class TestPatcherAsyncCompactContext:
    @pytest.mark.asyncio
    async def test_acompact_context_success(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        async def fake_compact(**kw: Any) -> tuple[str, str, dict[str, Any]]:
            return "compacted text", "raw", dict(_GOOD_META)

        monkeypatch.setattr(mod, "_compact_context_via_json_async", fake_compact)
        patcher = _make_patcher()
        result = await patcher.acompact_context(
            prompt="fix it",
            context_text="long context here",
            context_dir=tmp_path,
            cwd=tmp_path,
            target_chars=100,
        )
        assert isinstance(result, ContextCompactionResult)
        assert result.compacted_text == "compacted text"

    @pytest.mark.asyncio
    async def test_acompact_context_failure(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
    ) -> None:
        async def fake_compact(**kw: Any) -> tuple[None, str, dict[str, Any]]:
            return None, "raw response", dict(_GOOD_META)

        monkeypatch.setattr(mod, "_compact_context_via_json_async", fake_compact)
        patcher = _make_patcher()
        result = await patcher.acompact_context(
            prompt="fix it",
            context_text="long context here",
            context_dir=tmp_path,
            cwd=tmp_path,
            target_chars=100,
        )
        assert isinstance(result, ContextCompactionResult)
        assert result.compacted_text is None
        assert result.meta.get("error") == "invalid_context_compaction_payload"


# ---------------------------------------------------------------------------
# 18. _compact_context_via_json_async
# ---------------------------------------------------------------------------


class TestCompactContextViaJsonAsync:
    @pytest.mark.asyncio
    async def test_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        response = _ok_response(json.dumps({"compacted_context": "short version"}))

        async def fake_chat(**kw: Any) -> tuple[dict[str, Any], dict[str, Any]]:
            return response, dict(_GOOD_META)

        monkeypatch.setattr(mod, "_chat_completion_async", fake_chat)
        text, _raw, _meta = await mod._compact_context_via_json_async(
            url="http://localhost:1234/v1/chat/completions",
            api_key="k",
            model="m",
            prompt="fix",
            context_text="long text",
            target_chars=100,
            temperature=0.0,
            max_tokens=100,
            timeout_sec=5.0,
            seed=None,
            http2=False,
        )
        assert text == "short version"

    @pytest.mark.asyncio
    async def test_response_format_fallback(self, monkeypatch: pytest.MonkeyPatch) -> None:
        call_count = 0

        async def fake_chat(**kw: Any) -> tuple[dict[str, Any] | None, dict[str, Any]]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return None, {
                    "error": "http_error",
                    "body": "response_format is not supported",
                    "llm_call_count": 1,
                }
            return _ok_response(json.dumps({"compacted_context": "ok"})), dict(_GOOD_META)

        monkeypatch.setattr(mod, "_chat_completion_async", fake_chat)
        text, _raw, _meta = await mod._compact_context_via_json_async(
            url="http://localhost:1234/v1/chat/completions",
            api_key="k",
            model="m",
            prompt="fix",
            context_text="long text",
            target_chars=100,
            temperature=0.0,
            max_tokens=100,
            timeout_sec=5.0,
            seed=None,
            http2=False,
        )
        assert text == "ok"
        assert call_count == 2


# ---------------------------------------------------------------------------
# 19. _request_context_via_json_async
# ---------------------------------------------------------------------------


class TestRequestContextViaJsonAsync:
    @pytest.mark.asyncio
    async def test_success(self, monkeypatch: pytest.MonkeyPatch) -> None:
        payload = {
            "requests": [{"type": "file_content", "path": "foo.py"}],
        }
        response = _ok_response(json.dumps(payload))

        async def fake_chat(**kw: Any) -> tuple[dict[str, Any], dict[str, Any]]:
            return response, dict(_GOOD_META)

        monkeypatch.setattr(mod, "_chat_completion_async", fake_chat)
        raw, _meta = await mod._request_context_via_json_async(
            url="http://localhost:1234/v1/chat/completions",
            api_key="k",
            model="m",
            prompt="diagnose",
            temperature=0.0,
            max_tokens=100,
            timeout_sec=5.0,
            seed=None,
            http2=False,
        )
        assert raw is not None


# ---------------------------------------------------------------------------
# 20. _normalize_context_compaction_payload edge cases
# ---------------------------------------------------------------------------


class TestNormalizeContextCompaction:
    def test_none_response(self) -> None:
        text, raw = mod._normalize_context_compaction_payload(None)
        assert text is None
        assert raw is None

    def test_no_content(self) -> None:
        resp: dict[str, Any] = {"choices": [{"message": {}}]}
        text, _raw = mod._normalize_context_compaction_payload(resp)
        assert text is None

    def test_summary_fallback(self) -> None:
        resp = _ok_response(json.dumps({"summary": "brief"}))
        text, _raw = mod._normalize_context_compaction_payload(resp)
        assert text == "brief"

    def test_no_json(self) -> None:
        resp = _ok_response("not json")
        text, _raw = mod._normalize_context_compaction_payload(resp)
        assert text is None


# ---------------------------------------------------------------------------
# 21. _normalize_context_request_payload edge cases
# ---------------------------------------------------------------------------


class TestNormalizeContextRequest:
    def test_none_response(self) -> None:
        payload, raw = mod._normalize_context_request_payload(None)
        assert payload is None
        assert raw is None

    def test_no_content(self) -> None:
        resp: dict[str, Any] = {"choices": [{"message": {}}]}
        payload, _raw = mod._normalize_context_request_payload(resp)
        assert payload is None


# ---------------------------------------------------------------------------
# 22. _request_context_outcome / _request_context_failure_kind
# ---------------------------------------------------------------------------


class TestRequestContextOutcome:
    def test_valid_payload(self) -> None:
        result = mod._request_context_outcome(
            meta={},
            request_json={"requests": []},
            raw_response="test",
        )
        assert result == "valid_payload"

    def test_transport_failed(self) -> None:
        result = mod._request_context_outcome(
            meta={"error": "request_failed"},
            request_json=None,
            raw_response=None,
        )
        assert result == "transport_failed"

    def test_request_budget_exceeded(self) -> None:
        result = mod._request_context_outcome(
            meta={"error": "request_budget_exceeded"},
            request_json=None,
            raw_response=None,
        )
        assert result == "request_budget_exceeded"

    def test_invalid_payload(self) -> None:
        result = mod._request_context_outcome(
            meta={},
            request_json=None,
            raw_response="some bad json",
        )
        assert result == "invalid_payload"


# ---------------------------------------------------------------------------
# 23. _forced_patch_skip_reason with nested fallback
# ---------------------------------------------------------------------------


class TestForcedPatchSkipReason:
    def test_nested_fallback_meta(self) -> None:
        meta = {
            "error": "no_diff_found",
            "fallback_meta": {"error": "request_failed"},
        }
        result = mod._forced_patch_skip_reason(meta)
        assert result == "fallback_request_failed"

    def test_context_limit(self) -> None:
        meta = {
            "error": "http_error",
            "body": "prompt is too long",
        }
        result = mod._forced_patch_skip_reason(meta)
        assert result == "context_limit"


# ---------------------------------------------------------------------------
# 24. plan_patch — empty response
# ---------------------------------------------------------------------------


class TestPlanPatch:
    def test_empty_plan_sets_error(self, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
        def fake_chat(**kw: Any) -> tuple[dict[str, Any], dict[str, Any]]:
            return _ok_response("   "), dict(_GOOD_META)

        monkeypatch.setattr(mod, "_chat_completion", fake_chat)
        patcher = _make_patcher()
        result = patcher.plan_patch(prompt="fix", context_dir=tmp_path, cwd=tmp_path)
        assert result is not None
        assert result.plan_text is None
        assert result.meta.get("error") == "empty_plan_response"

    def test_plan_patch_respects_explicit_plan_cap(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        observed: dict[str, Any] = {}

        def fake_chat(**kw: Any) -> tuple[dict[str, Any], dict[str, Any]]:
            observed.update(kw)
            return _ok_response("Diagnosis\nMinimal edit plan"), dict(_GOOD_META)

        monkeypatch.setattr(mod, "_chat_completion", fake_chat)
        patcher = _make_patcher(max_tokens=4096, plan_patch_max_tokens=2048)

        result = patcher.plan_patch(prompt="fix", context_dir=tmp_path, cwd=tmp_path)

        assert result is not None
        assert observed["max_tokens"] == 2048
        assert result.meta["plan_patch_max_tokens"] == 2048
        assert result.meta["plan_patch_max_tokens_requested"] == 2048


# ---------------------------------------------------------------------------
# 25. _entry_matches_model — loaded_instances path
# ---------------------------------------------------------------------------


class TestEntryMatchesModelLoadedInstances:
    def test_loaded_instance_non_dict_skipped(self) -> None:
        entry = {"id": "other", "loaded_instances": ["not_a_dict"]}
        assert mod._entry_matches_model(entry, model="m1") is False

    def test_loaded_instance_id_match(self) -> None:
        entry = {
            "id": "other",
            "loaded_instances": [{"id": "m1"}],
        }
        assert mod._entry_matches_model(entry, model="m1") is True


# ---------------------------------------------------------------------------
# 26. _extract_lmstudio_v1_context_window — loaded instance paths
# ---------------------------------------------------------------------------


class TestLmstudioV1ContextWindow:
    def test_loaded_instance_non_dict_skipped(self) -> None:
        payload = {"data": [{"id": "m1", "loaded_instances": ["bad"]}]}
        tokens, _meta = mod._extract_lmstudio_v1_context_window(payload=payload, model="m1")
        assert tokens is None

    def test_loaded_instance_no_context_falls_to_entry(self) -> None:
        payload = {
            "data": [
                {
                    "id": "m1",
                    "loaded_instances": [{"id": "m1"}],
                    "max_context_length": 2048,
                }
            ]
        }
        tokens, _meta = mod._extract_lmstudio_v1_context_window(payload=payload, model="m1")
        assert tokens == 2048

    def test_loaded_instance_config_context_length(self) -> None:
        payload = {
            "data": [
                {
                    "id": "m1",
                    "loaded_instances": [{"id": "m1", "config": {"context_length": 4096}}],
                }
            ]
        }
        tokens, _meta = mod._extract_lmstudio_v1_context_window(payload=payload, model="m1")
        assert tokens == 4096


# ---------------------------------------------------------------------------
# 27. _extract_openai_models_context_window
# ---------------------------------------------------------------------------


class TestExtractOpenaiModelsContextWindow:
    def test_finds_max_input_tokens(self) -> None:
        payload = {"data": [{"id": "m1", "max_input_tokens": 16384}]}
        tokens, _meta = mod._extract_openai_models_context_window(payload=payload, model="m1")
        assert tokens == 16384

    def test_no_match(self) -> None:
        payload = {"data": [{"id": "other", "context_length": 4096}]}
        tokens, _meta = mod._extract_openai_models_context_window(payload=payload, model="m1")
        assert tokens is None


# ---------------------------------------------------------------------------
# 28. _propose_patch_via_diff — sync no_choice_content path
# ---------------------------------------------------------------------------


class TestProposePatchViaDiffSyncEdge:
    def test_no_choice_content(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def fake_chat(**kw: Any) -> tuple[dict[str, Any], dict[str, Any]]:
            return {"choices": [{"message": {}}]}, dict(_GOOD_META)

        monkeypatch.setattr(mod, "_chat_completion", fake_chat)
        result = mod._propose_patch_via_diff(
            url="http://localhost:1234/v1/chat/completions",
            api_key="k",
            model="m",
            prompt="fix",
            temperature=0.0,
            max_tokens=100,
            timeout_sec=5.0,
            seed=None,
            http2=False,
        )
        assert result.diff is None
        assert result.meta.get("error") == "no_choice_content"

    def test_second_attempt_succeeds(self, monkeypatch: pytest.MonkeyPatch) -> None:
        call_count = 0

        def fake_chat(**kw: Any) -> tuple[dict[str, Any], dict[str, Any]]:
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                return _ok_response("garbage"), dict(_GOOD_META)
            diff = (
                "diff --git a/foo.py b/foo.py\n"
                "--- a/foo.py\n+++ b/foo.py\n"
                "@@ -1,1 +1,1 @@\n-old\n+new\n"
            )
            return _ok_response(diff), dict(_GOOD_META)

        monkeypatch.setattr(mod, "_chat_completion", fake_chat)
        result = mod._propose_patch_via_diff(
            url="http://localhost:1234/v1/chat/completions",
            api_key="k",
            model="m",
            prompt="fix",
            temperature=0.0,
            max_tokens=100,
            timeout_sec=5.0,
            seed=None,
            http2=False,
        )
        assert result.diff is not None
        assert call_count == 2


# ---------------------------------------------------------------------------
# 29. _chat_completion sync — extra_payload + backoff_max_sec fix
# ---------------------------------------------------------------------------


class TestChatCompletionSyncEdge:
    def test_extra_payload_applied(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(mod, "_should_use_proxy_transport", lambda url: False)
        fake_client = MagicMock()
        fake_client.post_json.return_value = (200, json.dumps(_ok_response("ok")))
        fake_client.last_http_version = "HTTP/1.1"
        monkeypatch.setattr(mod, "_get_thread_http_client", lambda **kw: fake_client)
        data, _meta = mod._chat_completion(
            url="http://localhost:1234/v1/chat/completions",
            api_key="k",
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.0,
            max_tokens=100,
            timeout_sec=5.0,
            seed=None,
            http2=False,
            extra_payload={"response_format": {"type": "json_object"}},
        )
        assert data is not None

    def test_backoff_max_adjusted(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(mod, "_should_use_proxy_transport", lambda url: False)
        fake_client = MagicMock()
        fake_client.post_json.return_value = (200, json.dumps(_ok_response("ok")))
        fake_client.last_http_version = "HTTP/1.1"
        monkeypatch.setattr(mod, "_get_thread_http_client", lambda **kw: fake_client)
        data, _meta = mod._chat_completion(
            url="http://localhost:1234/v1/chat/completions",
            api_key="k",
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.0,
            max_tokens=100,
            timeout_sec=5.0,
            seed=None,
            http2=False,
            transport_retry_backoff_base_sec=5.0,
            transport_retry_backoff_max_sec=1.0,
        )
        assert data is not None

    def test_bad_json_response(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(mod, "_should_use_proxy_transport", lambda url: False)
        fake_client = MagicMock()
        fake_client.post_json.return_value = (200, "NOT JSON")
        fake_client.last_http_version = "HTTP/1.1"
        monkeypatch.setattr(mod, "_get_thread_http_client", lambda **kw: fake_client)
        data, meta = mod._chat_completion(
            url="http://localhost:1234/v1/chat/completions",
            api_key="k",
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.0,
            max_tokens=100,
            timeout_sec=5.0,
            seed=None,
            http2=False,
        )
        assert data is None
        assert meta["error"] == "bad_json_response"

    def test_seed_and_reasoning(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(mod, "_should_use_proxy_transport", lambda url: False)
        resp = {
            "choices": [
                {
                    "message": {
                        "content": "ok",
                        "reasoning_content": "think",
                        "reasoning": "reason",
                    },
                    "finish_reason": "stop",
                }
            ],
            "usage": _GOOD_USAGE,
            "seed": 42,
        }
        fake_client = MagicMock()
        fake_client.post_json.return_value = (200, json.dumps(resp))
        fake_client.last_http_version = "HTTP/1.1"
        monkeypatch.setattr(mod, "_get_thread_http_client", lambda **kw: fake_client)
        _data, meta = mod._chat_completion(
            url="http://localhost:1234/v1/chat/completions",
            api_key="k",
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.0,
            max_tokens=100,
            timeout_sec=5.0,
            seed=42,
            http2=False,
        )
        assert meta.get("api_compat_seed_acknowledged") is True
        assert meta.get("reasoning_content_present") is True
        assert meta.get("reasoning_present") is True

    def test_budget_with_retry_attempts(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import time

        monkeypatch.setattr(mod, "_should_use_proxy_transport", lambda url: False)
        data, meta = mod._chat_completion(
            url="http://localhost:1234/v1/chat/completions",
            api_key="k",
            model="m",
            messages=[{"role": "user", "content": "hi"}],
            temperature=0.0,
            max_tokens=100,
            timeout_sec=5.0,
            seed=None,
            http2=False,
            deadline_monotonic=time.monotonic() - 10,
        )
        assert data is None
        assert meta["error"] == "request_budget_exceeded"


# ---------------------------------------------------------------------------
# 30. _safe_trunc edge
# ---------------------------------------------------------------------------


class TestSafeTrunc:
    def test_non_serializable(self) -> None:
        result = mod._safe_trunc(object())
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# 31. generate_text — success, raw=None, no content
# ---------------------------------------------------------------------------


class TestGenerateText:
    """generate_text happy path and failure paths."""

    def _make_patcher(self) -> Any:
        return _make_patcher()

    def test_success_returns_text(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """generate_text returns TextGenerationResult with text set."""
        patcher = self._make_patcher()
        from evals.patchers.openai_compat_patcher import TextGenerationResult

        monkeypatch.setattr(
            mod,
            "_chat_completion",
            lambda **kw: (_ok_response("hello from model"), dict(_GOOD_META)),
        )
        result = patcher.generate_text(
            system_prompt="You are helpful.",
            prompt="Say hello.",
            context_dir=Path("."),
            cwd=Path("."),
        )
        assert isinstance(result, TextGenerationResult)
        assert result.text == "hello from model"
        assert result.meta is not None
        assert result.meta.get("text_generation_operation") == "generate_text"

    def test_raw_none_returns_text_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When _chat_completion returns (None, meta), generate_text returns text=None."""
        patcher = self._make_patcher()
        from evals.patchers.openai_compat_patcher import TextGenerationResult

        monkeypatch.setattr(
            mod,
            "_chat_completion",
            lambda **kw: (None, {"error": "http_error", "llm_call_count": 1, "usage": {}}),
        )
        result = patcher.generate_text(
            system_prompt="sys",
            prompt="user",
            context_dir=Path("."),
            cwd=Path("."),
        )
        assert isinstance(result, TextGenerationResult)
        assert result.text is None

    def test_no_choice_content_returns_text_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When response has no choices, generate_text returns text=None."""
        patcher = self._make_patcher()
        from evals.patchers.openai_compat_patcher import TextGenerationResult

        monkeypatch.setattr(
            mod,
            "_chat_completion",
            lambda **kw: ({"choices": [], "usage": {}}, dict(_GOOD_META)),
        )
        result = patcher.generate_text(
            system_prompt="sys",
            prompt="user",
            context_dir=Path("."),
            cwd=Path("."),
        )
        assert isinstance(result, TextGenerationResult)
        assert result.text is None
        assert result.meta is not None
        assert result.meta.get("error") == "no_choice_content"

    def test_custom_temperature_and_max_tokens(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """generate_text passes temperature/max_tokens overrides to _chat_completion."""
        patcher = self._make_patcher()
        captured: dict[str, Any] = {}

        def _fake_completion(**kw: Any) -> Any:
            captured.update(kw)
            return (_ok_response("ok"), dict(_GOOD_META))

        monkeypatch.setattr(mod, "_chat_completion", _fake_completion)
        patcher.generate_text(
            system_prompt="sys",
            prompt="user",
            context_dir=Path("."),
            cwd=Path("."),
            temperature=0.1,
            max_tokens=512,
        )
        assert captured.get("temperature") == pytest.approx(0.1)
        assert captured.get("max_tokens") == 512


# ---------------------------------------------------------------------------
# 32. generate_json — success, raw=None, fallback retry, no json payload
# ---------------------------------------------------------------------------


class TestGenerateJson:
    """generate_json happy path and failure paths."""

    def _make_patcher(self) -> Any:
        return _make_patcher()

    def test_success_returns_payload(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """generate_json parses JSON from the response content."""
        patcher = self._make_patcher()
        from evals.patchers.openai_compat_patcher import JsonGenerationResult

        monkeypatch.setattr(
            mod,
            "_chat_completion",
            lambda **kw: (_ok_response('{"key": "value"}'), dict(_GOOD_META)),
        )
        result = patcher.generate_json(
            system_prompt="sys",
            prompt="user",
            context_dir=Path("."),
            cwd=Path("."),
            schema_name="test_schema",
            schema={"type": "object"},
        )
        assert isinstance(result, JsonGenerationResult)
        assert result.payload == {"key": "value"}
        assert result.meta is not None
        assert result.meta.get("json_generation_operation") == "generate_json"

    def test_raw_none_returns_payload_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When _chat_completion returns (None, meta), generate_json returns payload=None."""
        patcher = self._make_patcher()
        from evals.patchers.openai_compat_patcher import JsonGenerationResult

        monkeypatch.setattr(
            mod,
            "_chat_completion",
            lambda **kw: (None, {"error": "timeout", "llm_call_count": 1, "usage": {}}),
        )
        result = patcher.generate_json(
            system_prompt="sys",
            prompt="user",
            context_dir=Path("."),
            cwd=Path("."),
            schema_name="test_schema",
            schema={"type": "object"},
        )
        assert isinstance(result, JsonGenerationResult)
        assert result.payload is None

    def test_http_error_with_response_format_retries(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When first call gets http_error with 'response_format' in body, retries without schema."""
        patcher = self._make_patcher()
        from evals.patchers.openai_compat_patcher import JsonGenerationResult

        call_count = {"n": 0}

        def _fake_completion(**kw: Any) -> Any:
            call_count["n"] += 1
            if call_count["n"] == 1:
                # First call: fail with response_format error
                return (
                    None,
                    {
                        "error": "http_error",
                        "body": "response_format not supported",
                        "llm_call_count": 1,
                        "usage": {},
                    },
                )
            # Retry: succeed with plain JSON response
            return (_ok_response('{"result": 42}'), dict(_GOOD_META))

        monkeypatch.setattr(mod, "_chat_completion", _fake_completion)
        result = patcher.generate_json(
            system_prompt="sys",
            prompt="user",
            context_dir=Path("."),
            cwd=Path("."),
            schema_name="test_schema",
            schema={"type": "object"},
        )
        assert isinstance(result, JsonGenerationResult)
        assert call_count["n"] == 2
        assert result.payload == {"result": 42}

    def test_non_json_content_returns_payload_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """When content cannot be parsed as JSON, payload is None."""
        patcher = self._make_patcher()
        from evals.patchers.openai_compat_patcher import JsonGenerationResult

        monkeypatch.setattr(
            mod,
            "_chat_completion",
            lambda **kw: (_ok_response("not valid json at all"), dict(_GOOD_META)),
        )
        result = patcher.generate_json(
            system_prompt="sys",
            prompt="user",
            context_dir=Path("."),
            cwd=Path("."),
            schema_name="test_schema",
            schema={"type": "object"},
        )
        assert isinstance(result, JsonGenerationResult)
        assert result.payload is None
        assert result.meta is not None
        assert result.meta.get("error") == "no_json_payload"


# ---------------------------------------------------------------------------
# 33. _optional_nonnegative_int — non-coercible value returns None
# ---------------------------------------------------------------------------


class TestOptionalNonnegativeInt:
    def test_string_returns_none(self) -> None:
        assert mod._optional_nonnegative_int("abc") is None

    def test_none_returns_none(self) -> None:
        assert mod._optional_nonnegative_int(None) is None

    def test_valid_int_returns_value(self) -> None:
        assert mod._optional_nonnegative_int(5) == 5

    def test_negative_clamped_to_zero(self) -> None:
        assert mod._optional_nonnegative_int(-3) == 0
