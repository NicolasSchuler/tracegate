"""Dense unit coverage for OpenAI-compatible patcher helper branches."""

from __future__ import annotations

import asyncio
import time
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from evals.patchers import openai_compat_patcher as openai_mod


def test_openai_client_wrappers_cover_cache_retry_and_timeout_paths(
    monkeypatch, tmp_path: Path
) -> None:
    monkeypatch.setenv("Z_API", "env-z-api")
    assert openai_mod._normalize_base_url("http://example.com") == "http://example.com/v1"
    assert openai_mod._normalize_base_url("http://example.com/v2/") == "http://example.com/v2"
    assert openai_mod._origin_from_base_url("https://api.example.com/v1/chat") == (
        "https://api.example.com"
    )
    assert openai_mod._origin_from_base_url("localhost:8000/") == "localhost:8000"
    assert openai_mod._base_url_from_chat_url("http://host/v1/chat/completions") == "http://host/v1"
    assert openai_mod._first_nonempty_env("MISSING", "Z_API") == "env-z-api"
    assert openai_mod._default_api_key() == "env-z-api"
    assert openai_mod._chat_http_client_key(
        "https://example.com:8443/v1/chat/completions", http2=False
    ) == ("https", "example.com", 8443, False)
    assert openai_mod._chat_http_target("https://example.com/v1/chat/completions?foo=1") == (
        "/v1/chat/completions?foo=1"
    )
    assert openai_mod._is_http2_protocol_error(httpx.RemoteProtocolError("HTTP/2 stream error"))
    assert not openai_mod._is_http2_protocol_error(RuntimeError("plain error"))

    monkeypatch.setattr(
        openai_mod.urllib.request,
        "getproxies",
        lambda: {"http": "http://proxy.example:3128"},
    )
    monkeypatch.setattr(openai_mod.urllib.request, "proxy_bypass", lambda host: False)
    assert openai_mod._should_use_proxy_transport("http://example.com")
    monkeypatch.setattr(openai_mod.urllib.request, "proxy_bypass", lambda host: True)
    assert not openai_mod._should_use_proxy_transport("http://example.com")
    assert not openai_mod._should_use_proxy_transport("file:///tmp/x")

    class FakeSyncClient:
        created: list[FakeSyncClient] = []

        def __init__(self, *, timeout: float, http2: bool, limits, trust_env: bool) -> None:
            self.timeout = timeout
            self.http2 = http2
            self.limits = limits
            self.trust_env = trust_env
            self.closed = False
            self.post_calls = 0
            self.get_calls = 0
            self.id = len(self.created)
            self.created.append(self)

        def post(self, url, content, headers):
            self.post_calls += 1
            if self.id == 0 and self.post_calls == 1:
                raise ConnectionResetError("retry me")
            return SimpleNamespace(status_code=201, text=f"post:{self.id}", http_version="HTTP/2")

        def get(self, url, headers):
            self.get_calls += 1
            if self.id == 2 and self.get_calls == 1:
                raise ConnectionResetError("retry me")
            return SimpleNamespace(status_code=200, text=f"get:{self.id}", http_version="HTTP/1.1")

        def close(self) -> None:
            self.closed = True

    class FakeAsyncClient:
        created: list[FakeAsyncClient] = []

        def __init__(self, *, timeout: float, http2: bool, limits, trust_env: bool) -> None:
            self.timeout = timeout
            self.http2 = http2
            self.limits = limits
            self.trust_env = trust_env
            self.closed = False
            self.post_calls = 0
            self.get_calls = 0
            self.id = len(self.created)
            self.created.append(self)

        async def post(self, url, content, headers):
            self.post_calls += 1
            if self.id == 0 and self.post_calls == 1:
                raise ConnectionResetError("retry me")
            return SimpleNamespace(status_code=202, text=f"post:{self.id}", http_version="HTTP/2")

        async def get(self, url, headers):
            self.get_calls += 1
            if self.id == 2 and self.get_calls == 1:
                raise ConnectionResetError("retry me")
            return SimpleNamespace(status_code=203, text=f"get:{self.id}", http_version="HTTP/1.1")

        async def aclose(self) -> None:
            self.closed = True

    FakeSyncClient.created.clear()
    FakeAsyncClient.created.clear()
    monkeypatch.setattr(openai_mod.httpx, "Client", FakeSyncClient)
    monkeypatch.setattr(openai_mod.httpx, "AsyncClient", FakeAsyncClient)
    openai_mod._THREAD_HTTP_CLIENTS.clients = {}
    openai_mod._ASYNC_HTTP_CLIENTS.clear()

    sync_client = openai_mod._PersistentHTTPClient(
        scheme="http",
        host="example.com",
        port=80,
        http2=True,
    )
    assert sync_client.origin == "http://example.com:80"
    status, text = sync_client.post_json(
        target="/v1/chat/completions",
        body=b"{}",
        headers={"x-test": "1"},
        timeout_sec=1.5,
    )
    assert (status, text) == (201, "post:1")
    assert sync_client.last_http_version == "HTTP/2"
    assert len(FakeSyncClient.created) == 2
    assert FakeSyncClient.created[0].closed is True

    status, text = sync_client.get_text(
        target="/v1/models",
        headers={"x-test": "2"},
        timeout_sec=2.5,
    )
    assert (status, text) == (200, "get:3")
    assert sync_client.last_http_version == "HTTP/1.1"
    assert len(FakeSyncClient.created) == 4
    sync_client.close()

    assert openai_mod._get_thread_http_client(
        url="http://example.com/v1/chat/completions", http2=True
    ) is openai_mod._get_thread_http_client(
        url="http://example.com/v1/chat/completions", http2=True
    )

    async def _exercise_async() -> None:
        async_client = openai_mod._PersistentAsyncHTTPClient(
            scheme="http",
            host="example.com",
            port=80,
            http2=True,
        )
        assert async_client.origin == "http://example.com:80"
        status, text = await async_client.post_json(
            target="/v1/chat/completions",
            body=b"{}",
            headers={"x-test": "1"},
            timeout_sec=1.5,
        )
        assert (status, text) == (202, "post:1")
        assert async_client.last_http_version == "HTTP/2"
        assert len(FakeAsyncClient.created) == 2

        status, text = await async_client.get_text(
            target="/v1/models",
            headers={"x-test": "2"},
            timeout_sec=2.5,
        )
        assert (status, text) == (203, "get:3")
        assert async_client.last_http_version == "HTTP/1.1"
        assert len(FakeAsyncClient.created) == 4
        await async_client.close()

        cached_a = openai_mod._get_async_http_client(
            url="http://example.com/v1/chat/completions",
            http2=True,
        )
        cached_b = openai_mod._get_async_http_client(
            url="http://example.com/v1/chat/completions",
            http2=True,
        )
        assert cached_a is cached_b

    asyncio.run(_exercise_async())


def test_openai_payload_and_diff_helpers_cover_parser_branches() -> None:
    assert openai_mod._nonnegative_int(-3) == 0
    assert openai_mod._nonnegative_int("12") == 12
    assert openai_mod._nonnegative_int("bad", default=4) == 4

    assert openai_mod._coerce_payload_bool(True) is True
    assert openai_mod._coerce_payload_bool(False) is False
    assert openai_mod._coerce_payload_bool(None) is False
    assert openai_mod._coerce_payload_bool(" yes ") is True
    assert openai_mod._coerce_payload_bool(" off ") is False
    assert openai_mod._coerce_payload_bool("maybe") is False

    assert openai_mod._usage_counts_from_payload(
        {"prompt_tokens": 5, "completion_tokens": 7, "total_tokens": 15}
    ) == (5, 7, 15)
    assert openai_mod._usage_counts_from_payload({"input_tokens": 3, "output_tokens": 4}) == (
        3,
        4,
        7,
    )
    assert openai_mod._reasoning_tokens_from_payload(
        {"output_tokens_details": {"reasoning_tokens": 9}}
    ) == (9, True)
    assert openai_mod._usage_counts_from_payload(None) == (0, 0, 0)
    assert openai_mod._usage_payload_from_counts(
        prompt_tokens=-1,
        completion_tokens=2,
        total_tokens=-5,
        reasoning_tokens=4,
    ) == {
        "prompt_tokens": 0,
        "completion_tokens": 2,
        "total_tokens": 0,
        "reasoning_tokens": 4,
    }

    assert openai_mod._llm_call_count_from_meta(None, default=3) == 3
    assert openai_mod._llm_call_count_from_meta({"llm_call_count": 4}, default=3) == 4
    totals = openai_mod._accumulate_llm_totals(
        totals=(1, 2, 3, 4, 1, True, True),
        meta={
            "llm_call_count": 5,
            "usage": {
                "prompt_tokens": 6,
                "completion_tokens": 7,
                "output_tokens_details": {"reasoning_tokens": 2},
            },
        },
        default_calls=1,
    )
    assert totals == (6, 8, 10, 17, 3, True, True)
    assert openai_mod._meta_with_llm_totals(
        meta={"seed": 1},
        totals=totals,
    ) == {
        "seed": 1,
        "llm_call_count": 6,
        "usage": {
            "prompt_tokens": 8,
            "completion_tokens": 10,
            "total_tokens": 17,
            "reasoning_tokens": 3,
        },
    }

    system_prompt = openai_mod._request_context_system_prompt(strict_json=False)
    assert "Return only JSON" in system_prompt
    assert "Allowed request types" in system_prompt
    strict_prompt = openai_mod._request_context_system_prompt(strict_json=True)
    assert "JSON only" in strict_prompt
    assert "need_more_context=false" in strict_prompt
    assert "Diagnosis" in openai_mod._plan_patch_system_prompt()

    diff_text = (
        "Before explanation\n"
        "```diff\n"
        "diff --git a/foo.py b/foo.py\n"
        "--- a/foo.py\n"
        "+++ b/foo.py\n"
        "@@ -1 +1 @@\n"
        "-old\n"
        "+new\n"
        "```\n"
        "After explanation"
    )
    extracted = openai_mod._extract_unified_diff(diff_text)
    assert extracted is not None
    assert "diff --git a/foo.py b/foo.py" in extracted
    assert openai_mod._strip_think("<think>ignored</think>\nvalue") == "value"
    assert openai_mod._clean_path_token('"a/foo.py"') == "foo.py"
    assert openai_mod._clean_path_token("a/foo.py\t2024-01-01") == "foo.py"
    assert openai_mod._clean_path_token("b/bar.py ---") == "bar.py"
    assert openai_mod._clean_path_token("/dev/null") == "/dev/null"
    assert openai_mod._clean_path_token("a/") is None

    [(path, normalized)] = openai_mod._iter_file_diffs(
        [
            "--- a/foo.py",
            "+++ b/bar.py",
            "@@ -1 +1 @@",
            "-old",
            "+new",
            "narration starts here",
        ]
    )
    assert path == "foo.py"
    assert "diff --git a/foo.py b/foo.py" in normalized
    assert "narration" not in normalized
    [(_, git_block)] = openai_mod._iter_file_diffs(
        [
            "intro",
            "diff --git a/foo.py b/foo.py",
            "--- a/foo.py",
            "+++ b/foo.py",
            "@@ -1 +1 @@",
            "-old",
            "+new",
            "extra prose",
        ]
    )
    assert "extra prose" not in git_block

    assert openai_mod._get_first_choice_content(
        {"choices": [{"message": {"content": "text"}}]}
    ) == ("text")
    assert openai_mod._get_first_choice_content({"choices": [{"text": "legacy"}]}) == "legacy"
    assert openai_mod._get_first_choice_finish_reason({"choices": [{"finish_reason": "stop"}]}) == (
        "stop"
    )
    assert (
        openai_mod._get_first_choice_reasoning_content(
            {"choices": [{"message": {"reasoning_content": "chain"}}]}
        )
        == "chain"
    )
    assert openai_mod._coerce_content_to_text(["a", "b"]) == "ab"
    assert (
        openai_mod._coerce_content_to_text(
            [{"type": "reasoning", "text": "skip"}, {"type": "weird", "content": {"value": "x"}}]
        )
        == "x"
    )
    assert openai_mod._extract_content_part_text({"content": {"text": "nested"}}) == "nested"

    deadline = time.monotonic() + 1.0
    assert 0.0 < openai_mod._remaining_request_budget_sec(deadline) <= 1.0
    assert openai_mod._chat_completion_retry_backoff_sec(-1, base_sec=0.5, max_sec=2.0) == 0.5
    assert openai_mod._chat_completion_retry_backoff_sec(4, base_sec=0.5, max_sec=2.0) == 2.0
    assert openai_mod._is_retryable_chat_completion_http_status(429)
    assert openai_mod._is_retryable_chat_completion_http_status(503)
    assert not openai_mod._is_retryable_chat_completion_http_status(404)
    assert openai_mod._is_retryable_chat_completion_error({"error": "request_failed"})
    assert openai_mod._is_retryable_chat_completion_error({"error": "http_error", "status": 429})
    assert openai_mod._is_backend_unavailable_error({"error": "http_error", "status": 503})
    assert openai_mod._is_backend_unavailable_error(
        {"error": "request_failed", "exc": "Connection refused"}
    )
    assert (
        openai_mod._request_context_failure_kind(
            meta={"error": "request_budget_exceeded"}, request_json=None, raw_response=None
        )
        == "request_budget_exceeded"
    )
    assert (
        openai_mod._request_context_failure_kind(
            meta={"error": "request_failed", "exc": "Connection refused"},
            request_json=None,
            raw_response=None,
        )
        == "backend_unavailable"
    )
    assert (
        openai_mod._request_context_outcome(
            meta={"error": "request_failed", "exc": "Connection refused"},
            request_json=None,
            raw_response=None,
        )
        == "unavailable_backend"
    )
    assert openai_mod._is_request_budget_exceeded({"error": "request_budget_exceeded"})
    assert openai_mod._forced_patch_skip_reason({"error": "request_failed"}) == "request_failed"
    assert (
        openai_mod._forced_patch_skip_reason({"error": "http_error", "body": "prompt is too long"})
        == "context_limit"
    )
    assert openai_mod._is_context_exceeded_http_error(
        {"error": "http_error", "body": "maximum context length exceeded"}
    )


def test_openai_edit_and_full_file_helpers_cover_recovery_and_targeting(
    tmp_path: Path,
) -> None:
    assert (
        openai_mod._normalize_edits_payload(
            {"edits": [{"path": "foo.py", "find": "a", "replace": "b"}]}
        )[1]
        == "edits_list"
    )
    assert (
        openai_mod._normalize_edits_payload(
            {"edits": {"path": "foo.py", "find": "a", "replace": "b"}}
        )[1]
        == "edits_object"
    )
    assert (
        openai_mod._normalize_edits_payload({"path": "foo.py", "old": "a", "new": "b"})[1]
        == "single_edit_root"
    )
    assert (
        openai_mod._normalize_edits_payload(
            {"edit": {"path": "foo.py", "find": "a", "replace": "b"}}
        )[1]
        == "edit_object"
    )
    assert (
        openai_mod._normalize_edits_payload(
            {"changes": [{"path": "foo.py", "find": "a", "replace": "b"}]}
        )[1]
        == "changes_list"
    )
    assert openai_mod._normalize_edits_payload({"path": "foo.py"}) == (None, None)
    assert openai_mod._looks_like_edit_entry({"path": "foo.py", "find": "a", "replace": "b"})

    after, reason = openai_mod._apply_find_replace_with_recovery(
        before="abc def",
        find="abc",
        replace="xyz",
        count=1,
    )
    assert after == "xyz def"
    assert reason == "exact"
    after, reason = openai_mod._apply_find_replace_with_recovery(
        before="line1\r\nline2\r\n",
        find="line1\nline2\n",
        replace="A\nB\n",
        count=1,
    )
    assert after == "A\nB\n"
    assert reason == "normalized_newlines"
    after, reason = openai_mod._apply_find_replace_with_recovery(
        before="foo  bar\n",
        find="foo bar\n",
        replace="baz\n",
        count=1,
    )
    assert after == "baz\n"
    assert reason == "normalized_whitespace"
    fuzzy = openai_mod._apply_fuzzy_block_replacement(
        before_nl="alpha beta gamma\ndelta epsilon zeta\n",
        find_nl="alpha beta gamma\ndelta epsilon zeta\n",
        replace_nl="replacement\n",
    )
    assert fuzzy == "replacement\n"
    assert (
        openai_mod._apply_fuzzy_block_replacement(
            before_nl="short\n",
            find_nl="short\n",
            replace_nl="replacement\n",
        )
        is None
    )

    assert openai_mod._normalize_newlines("a\r\nb\rc") == "a\nb\nc"
    pattern = openai_mod._compile_whitespace_normalized_pattern("a \tb\n")
    assert pattern.search("a\t\tb\n")
    assert openai_mod._safe_trunc({"a": "b"}, limit=8)
    assert openai_mod._extract_response_seed({"seed": "7"}) == 7
    assert openai_mod._extract_json_object('```json\n{"a": 1}\n```') == {"a": 1}
    assert openai_mod._extract_json_object('prefix {"a": 1} suffix') == {"a": 1}
    assert openai_mod._extract_json_object("no json") is None
    block, fenced = openai_mod._extract_first_fenced_code_block("```py\nprint(1)\n```")
    assert fenced is True and "print(1)" in block
    assert openai_mod._looks_like_raw_unified_diff(
        "diff --git a/foo.py b/foo.py\n--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-a\n+b\n"
    )

    full, meta = openai_mod._normalize_full_file_response(
        {
            "choices": [{"message": {"content": "```python\nprint('ok')\n```"}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1},
        }
    )
    assert full == "print('ok')\n"
    assert meta["full_file_code_fence_stripped"] is True
    assert meta["full_file_output_chars"] == len("print('ok')\n")
    assert openai_mod._normalize_full_file_response({"choices": []})[0] is None
    assert (
        openai_mod._normalize_full_file_response(
            {
                "choices": [
                    {
                        "message": {
                            "content": (
                                "diff --git a/foo.py b/foo.py\n"
                                "--- a/foo.py\n"
                                "+++ b/foo.py\n"
                                "@@ -1 +1 @@\n"
                                "-old\n"
                                "+new\n"
                            )
                        }
                    }
                ],
            }
        )[1]["error"]
        == "returned_diff_instead_of_file"
    )

    assert openai_mod._build_diff_from_changes([("foo.py", "a\n", "b\n")]).startswith(
        "diff --git a/foo.py b/foo.py"
    )
    assert (
        openai_mod._fix_hunk_counts(
            "diff --git a/foo.py b/foo.py\n--- a/foo.py\n+++ b/foo.py\n@@ -1,9 +1,9 @@\n-a\n+b\n"
        )
        .splitlines()[3]
        .startswith("@@ -1,1 +1,1 @@")
    )
    assert openai_mod._extract_file_index_paths(
        "Available files:\nfoo.py\ntest_foo.py\n\nPytest output:\nfailed\n"
    ) == ["foo.py", "test_foo.py"]

    (tmp_path / "foo.py").write_text("print('a')\n", encoding="utf-8")
    (tmp_path / "test_foo.py").write_text("print('test')\n", encoding="utf-8")
    (tmp_path / "subdir").mkdir()
    (tmp_path / "subdir" / "bar.py").write_text("print('b')\n", encoding="utf-8")
    target, reason = openai_mod._resolve_full_file_target(cwd=tmp_path)
    assert target is None and reason == "multiple_non_test_python_files"

    single = tmp_path / "single"
    single.mkdir()
    (single / "main.py").write_text("print('ok')\n", encoding="utf-8")
    (single / "test_main.py").write_text("print('test')\n", encoding="utf-8")
    assert openai_mod._resolve_full_file_target(cwd=single) == (
        "main.py",
        "single_non_test_python_file",
    )

    prompt = (
        "Available files (use exact relative paths):\n"
        "test_foo.py\n"
        "main.py\n\n"
        "Pytest output (stdout):\nfailed\n"
    )
    forced = openai_mod._build_forced_attempt_diff(prompt=prompt, cwd=single)
    assert forced is not None
    target_rel, diff = forced
    assert target_rel == "main.py"
    assert "llmdebug-force-attempt" in diff


def test_openai_helper_edge_branches_cover_missing_env_and_parser_fallbacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("Z_API", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLMDEBUG_EVAL_OPENAI_API_KEY", raising=False)

    assert openai_mod._base_url_from_chat_url("http://host/v1/models") == "http://host/v1/models"
    assert openai_mod._first_nonempty_env("MISSING_A", "MISSING_B") == ""
    assert openai_mod._default_api_key() == "local"
    assert openai_mod._clean_path_token(None) is None
    assert openai_mod._clean_path_token("   ") is None
    assert openai_mod._iter_file_diffs(["--- a/foo.py", "narration starts here"]) == []
    assert openai_mod._iter_file_diffs(["--- a/", "+++ b/"]) == []
    assert (
        openai_mod._iter_file_diffs(
            ["diff --git a/foo.py b/foo.py", "index 123..456 100644", "explanation starts"]
        )
        == []
    )

    monkeypatch.setattr(openai_mod._THREAD_HTTP_CLIENTS, "clients", [], raising=False)
    client = openai_mod._get_thread_http_client(
        url="http://example.com/v1/chat/completions",
        http2=False,
    )
    assert isinstance(openai_mod._THREAD_HTTP_CLIENTS.clients, dict)
    assert client.origin == "http://example.com:80"


def test_openai_persistent_client_edge_branches_cover_retry_exhaustion_and_close_swallow(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class AlwaysRetrySyncClient:
        def __init__(self, **kwargs):
            self.closed = False

        def post(self, url, content, headers):
            raise ConnectionResetError("retry me")

        def get(self, url, headers):
            raise ConnectionResetError("retry me")

        def close(self) -> None:
            self.closed = True

    class ExplodingCloseSyncClient:
        def close(self) -> None:
            raise RuntimeError("boom")

    class AlwaysRetryAsyncClient:
        def __init__(self, **kwargs):
            self.closed = False

        async def post(self, url, content, headers):
            raise ConnectionResetError("retry me")

        async def get(self, url, headers):
            raise ConnectionResetError("retry me")

        async def aclose(self) -> None:
            self.closed = True

    class ExplodingCloseAsyncClient:
        async def aclose(self) -> None:
            raise RuntimeError("boom")

    monkeypatch.setattr(openai_mod.httpx, "Client", AlwaysRetrySyncClient)
    monkeypatch.setattr(openai_mod.httpx, "AsyncClient", AlwaysRetryAsyncClient)

    sync_client = openai_mod._PersistentHTTPClient(
        scheme="http",
        host="example.com",
        port=80,
        http2=True,
    )
    with pytest.raises(ConnectionResetError):
        sync_client.post_json(
            target="/v1/chat/completions",
            body=b"{}",
            headers={"x-test": "1"},
            timeout_sec=1.5,
        )
    with pytest.raises(ConnectionResetError):
        sync_client.get_text(
            target="/v1/models",
            headers={"x-test": "2"},
            timeout_sec=1.5,
        )
    sync_client._client = ExplodingCloseSyncClient()
    sync_client.close()

    monkeypatch.setattr(openai_mod._THREAD_HTTP_CLIENTS, "clients", [], raising=False)
    threaded_client = openai_mod._get_thread_http_client(
        url="http://example.com/v1/chat/completions",
        http2=False,
    )
    assert isinstance(threaded_client, openai_mod._PersistentHTTPClient)

    async def _exercise_async() -> None:
        async_client = openai_mod._PersistentAsyncHTTPClient(
            scheme="http",
            host="example.com",
            port=80,
            http2=True,
        )
        with pytest.raises(ConnectionResetError):
            await async_client.post_json(
                target="/v1/chat/completions",
                body=b"{}",
                headers={"x-test": "1"},
                timeout_sec=1.5,
            )
        with pytest.raises(ConnectionResetError):
            await async_client.get_text(
                target="/v1/models",
                headers={"x-test": "2"},
                timeout_sec=1.5,
            )
        async_client._client = ExplodingCloseAsyncClient()
        await async_client.close()

    asyncio.run(_exercise_async())
