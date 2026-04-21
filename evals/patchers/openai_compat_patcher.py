"""LLM-based patch generation using an OpenAI-compatible chat-completions API.

Implements the :class:`~evals.patchers.base.Patcher` protocol by routing every
patch-generation request to any server that speaks the OpenAI
``/v1/chat/completions`` API — vLLM, LM Studio, llama.cpp, OpenAI, and
compatible local or remote endpoints.

Three response modes are supported:

* ``"edits"`` *(default)* — the model returns a structured JSON payload
  ``{"edits": [{"path": ..., "find": ..., "replace": ...}]}`` that is applied
  as a series of find-replace operations directly against the source files.  The
  patcher derives a unified diff from the resulting file changes.  Three
  ``response_format`` variants are attempted in order (``json_schema`` →
  ``json_object`` → plain text) and, if none succeeds, the patcher falls back
  to ``"diff"`` mode.
* ``"diff"`` — the model returns a raw unified diff which the patcher extracts
  and normalises.  When no parseable diff is found the patcher falls back to
  ``"edits"`` mode.
* ``"full_file"`` — when the case has exactly one non-test Python source file,
  the model returns the full repaired contents of that file and the patcher
  derives a unified diff locally. Unsupported case shapes and unusable
  responses fall back to ``"diff"`` mode.

Optional multi-stage context-gathering is also supported:

* :meth:`~OpenAICompatPatcher.request_context` /
  :meth:`~OpenAICompatPatcher.arequest_context` — Stage A: ask the model which
  additional files, snapshots, or other artefacts it needs before generating a
  patch, and return the request as a structured :data:`ContextRequestPayload`.
* :meth:`~OpenAICompatPatcher.compact_context` /
  :meth:`~OpenAICompatPatcher.acompact_context` — compress a large accumulated
  context block down to a caller-supplied character budget.
* :meth:`~OpenAICompatPatcher.plan_patch` — ask the model for a concise
  natural-language repair plan that can be prepended to subsequent prompts to
  steer it away from repeated mistakes.

Both synchronous and ``async`` variants are provided for every public operation
so that the patcher can be embedded in both threaded and async eval runners
without blocking the event loop.

Example::

    patcher = OpenAICompatPatcher(
        base_url="http://127.0.0.1:1234",
        model="my-model",
        response_mode="edits",
    )
    result = patcher.propose_patch(prompt=..., context_dir=..., cwd=...)
    if result.diff:
        ...
"""

from __future__ import annotations

import asyncio
import difflib
import functools
import os
import re
import threading
import time
import urllib.parse
import urllib.request
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import httpx

from baselines.openai_compat import (
    BaselineOpenAICompatConfig,
    OpenAICompatRequestMeta,
    OpenAICompatResponseError,
    apply_compat_request_profile,
    normalize_chat_completion_response,
    normalize_compat_profile,
    require_final_content,
    response_error_meta,
)

from .. import _json as json
from ..eval_patches import _is_test_file_path
from ..stage_a_contract import (
    STAGE_A_REQUEST_SCHEMA_COMPACT,
    coerce_stage_a_bool,
    normalize_stage_a_request_payload,
    stage_a_allowed_request_types_text,
)
from .base import (
    ContextCompactionResult,
    ContextRequestResult,
    Patcher,
    PatchPlanResult,
    PatchResult,
)
from .payload_types import ContextRequestPayload, PatcherMeta, UsagePayload

OPENAI_API_KEY_ENV_FALLBACKS = ("Z_API", "OPENAI_API_KEY", "LLMDEBUG_EVAL_OPENAI_API_KEY")
_THREAD_HTTP_CLIENTS = threading.local()
_ASYNC_HTTP_CLIENTS: dict[tuple[int, str, str, int, bool], _PersistentAsyncHTTPClient] = {}
_ASYNC_HTTP_CLIENTS_LOCK = threading.Lock()
_RETRYABLE_HTTP_ERRORS = (
    httpx.ConnectError,
    httpx.ConnectTimeout,
    httpx.ReadTimeout,
    httpx.WriteTimeout,
    httpx.ReadError,
    httpx.WriteError,
    httpx.PoolTimeout,
    httpx.RemoteProtocolError,
    BrokenPipeError,
    ConnectionAbortedError,
    ConnectionResetError,
    TimeoutError,
    OSError,
)
_HTTP2_PROTOCOL_ERRORS = (httpx.RemoteProtocolError,)
_DEFAULT_CHAT_COMPLETION_TRANSPORT_MAX_RETRIES = 6
_CHAT_COMPLETION_RETRYABLE_HTTP_STATUSES = {408, 409, 425, 429}
_DEFAULT_CHAT_COMPLETION_RETRY_BACKOFF_BASE_SEC = 1.0
_DEFAULT_CHAT_COMPLETION_RETRY_BACKOFF_MAX_SEC = 8.0
_REQUEST_CONTEXT_MAX_TRANSPORT_REISSUES = 1
_REQUEST_CONTEXT_MAX_STRICT_JSON_RETRIES = 1
_REQUEST_CONTEXT_STRICT_JSON_MAX_TOKENS = 768
_PLAN_PATCH_MAX_TOKENS = 768
_BACKEND_UNAVAILABLE_HTTP_STATUSES = {502, 503, 504}
_DEFAULT_MAX_PROPOSE_SEC = 300.0
_DEFAULT_HTTPX_MAX_CONNECTIONS = 100
_DEFAULT_HTTPX_MAX_KEEPALIVE_CONNECTIONS = 20
_OPENAI_COMPAT_REASONING_ONLY_ERROR = "openai_compat_reasoning_only"
_DEFAULT_HTTPX_KEEPALIVE_EXPIRY_SEC = 30.0
_OPENAI_PROGRESS_CALLBACK: ContextVar[Callable[[str, dict[str, Any] | None], None] | None] = (
    ContextVar("llmdebug_openai_progress_callback", default=None)
)
_OPENAI_PROGRESS_BASE_PAYLOAD: ContextVar[dict[str, Any] | None] = ContextVar(
    "llmdebug_openai_progress_base_payload",
    default=None,
)


def _emit_openai_transport_progress(
    event: str,
    *,
    note: str = "",
    **extra: Any,
) -> None:
    callback = _OPENAI_PROGRESS_CALLBACK.get()
    if callback is None:
        return
    payload: dict[str, Any] = {
        "owner": "openai",
        "source": "transport",
        "waiting_for": "openai_server",
    }
    base_payload = _OPENAI_PROGRESS_BASE_PAYLOAD.get()
    if isinstance(base_payload, dict):
        payload.update(base_payload)
    if note:
        payload["note"] = note
    payload.update(extra)
    try:
        callback(event, payload)
    except Exception:
        return


def _with_openai_transport_progress(*, external_operation: str, phase_label: str):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(self, *args: Any, **kwargs: Any):
            callback_token = _OPENAI_PROGRESS_CALLBACK.set(self._current_progress_callback())
            payload: dict[str, Any] = {
                "owner": "openai",
                "source": "transport",
                "waiting_for": "openai_server",
                "external_operation": external_operation,
                "phase_label": phase_label,
            }
            response_mode = getattr(self, "response_mode", None)
            if isinstance(response_mode, str) and response_mode:
                payload["response_mode"] = response_mode
            payload_token = _OPENAI_PROGRESS_BASE_PAYLOAD.set(payload)
            try:
                return func(self, *args, **kwargs)
            finally:
                _OPENAI_PROGRESS_BASE_PAYLOAD.reset(payload_token)
                _OPENAI_PROGRESS_CALLBACK.reset(callback_token)

        return wrapper

    return decorator


def _with_openai_transport_progress_async(*, external_operation: str, phase_label: str):
    def decorator(func):
        @functools.wraps(func)
        async def wrapper(self, *args: Any, **kwargs: Any):
            callback_token = _OPENAI_PROGRESS_CALLBACK.set(self._current_progress_callback())
            payload: dict[str, Any] = {
                "owner": "openai",
                "source": "transport",
                "waiting_for": "openai_server",
                "external_operation": external_operation,
                "phase_label": phase_label,
            }
            response_mode = getattr(self, "response_mode", None)
            if isinstance(response_mode, str) and response_mode:
                payload["response_mode"] = response_mode
            payload_token = _OPENAI_PROGRESS_BASE_PAYLOAD.set(payload)
            try:
                return await func(self, *args, **kwargs)
            finally:
                _OPENAI_PROGRESS_BASE_PAYLOAD.reset(payload_token)
                _OPENAI_PROGRESS_CALLBACK.reset(callback_token)

        return wrapper

    return decorator


@dataclass
class _PersistentHTTPClient:
    scheme: str
    host: str
    port: int
    http2: bool
    _client: httpx.Client | None = None
    _timeout_sec: float | None = None
    last_http_version: str = ""

    @property
    def origin(self) -> str:
        return f"{self.scheme}://{self.host}:{self.port}"

    def _new_client(self, *, timeout_sec: float) -> httpx.Client:
        limits = httpx.Limits(
            max_connections=_DEFAULT_HTTPX_MAX_CONNECTIONS,
            max_keepalive_connections=_DEFAULT_HTTPX_MAX_KEEPALIVE_CONNECTIONS,
            keepalive_expiry=_DEFAULT_HTTPX_KEEPALIVE_EXPIRY_SEC,
        )
        return httpx.Client(timeout=timeout_sec, http2=self.http2, limits=limits, trust_env=True)

    def _ensure_client(self, *, timeout_sec: float) -> httpx.Client:
        if self._client is None or self._timeout_sec != timeout_sec:
            self.close()
            self._client = self._new_client(timeout_sec=timeout_sec)
            self._timeout_sec = timeout_sec
        return self._client

    def close(self) -> None:
        if self._client is not None:
            try:
                self._client.close()
            except Exception:
                pass
        self._client = None
        self._timeout_sec = None

    def post_json(
        self,
        *,
        target: str,
        body: bytes,
        headers: dict[str, str],
        timeout_sec: float,
    ) -> tuple[int, str]:
        url = f"{self.origin}{target}"
        for attempt_idx in range(2):
            client = self._ensure_client(timeout_sec=timeout_sec)
            try:
                response = client.post(url, content=body, headers=headers)
                self.last_http_version = str(getattr(response, "http_version", "") or "")
                return int(response.status_code), response.text
            except _RETRYABLE_HTTP_ERRORS:
                self.close()
                if attempt_idx == 0:
                    continue
                raise
        raise RuntimeError("unreachable")

    def get_text(
        self,
        *,
        target: str,
        headers: dict[str, str],
        timeout_sec: float,
    ) -> tuple[int, str]:
        url = f"{self.origin}{target}"
        for attempt_idx in range(2):
            client = self._ensure_client(timeout_sec=timeout_sec)
            try:
                response = client.get(url, headers=headers)
                self.last_http_version = str(getattr(response, "http_version", "") or "")
                return int(response.status_code), response.text
            except _RETRYABLE_HTTP_ERRORS:
                self.close()
                if attempt_idx == 0:
                    continue
                raise
        raise RuntimeError("unreachable")


@dataclass
class _PersistentAsyncHTTPClient:
    scheme: str
    host: str
    port: int
    http2: bool
    _client: httpx.AsyncClient | None = None
    _timeout_sec: float | None = None
    last_http_version: str = ""

    @property
    def origin(self) -> str:
        return f"{self.scheme}://{self.host}:{self.port}"

    def _new_client(self, *, timeout_sec: float) -> httpx.AsyncClient:
        limits = httpx.Limits(
            max_connections=_DEFAULT_HTTPX_MAX_CONNECTIONS,
            max_keepalive_connections=_DEFAULT_HTTPX_MAX_KEEPALIVE_CONNECTIONS,
            keepalive_expiry=_DEFAULT_HTTPX_KEEPALIVE_EXPIRY_SEC,
        )
        return httpx.AsyncClient(
            timeout=timeout_sec, http2=self.http2, limits=limits, trust_env=True
        )

    async def _ensure_client(self, *, timeout_sec: float) -> httpx.AsyncClient:
        if self._client is None or self._timeout_sec != timeout_sec:
            await self.close()
            self._client = self._new_client(timeout_sec=timeout_sec)
            self._timeout_sec = timeout_sec
        return self._client

    async def close(self) -> None:
        if self._client is not None:
            try:
                await self._client.aclose()
            except Exception:
                pass
        self._client = None
        self._timeout_sec = None

    async def post_json(
        self,
        *,
        target: str,
        body: bytes,
        headers: dict[str, str],
        timeout_sec: float,
    ) -> tuple[int, str]:
        url = f"{self.origin}{target}"
        for attempt_idx in range(2):
            client = await self._ensure_client(timeout_sec=timeout_sec)
            try:
                response = await client.post(url, content=body, headers=headers)
                self.last_http_version = str(getattr(response, "http_version", "") or "")
                return int(response.status_code), response.text
            except _RETRYABLE_HTTP_ERRORS:
                await self.close()
                if attempt_idx == 0:
                    continue
                raise
        raise RuntimeError("unreachable")

    async def get_text(
        self,
        *,
        target: str,
        headers: dict[str, str],
        timeout_sec: float,
    ) -> tuple[int, str]:
        url = f"{self.origin}{target}"
        for attempt_idx in range(2):
            client = await self._ensure_client(timeout_sec=timeout_sec)
            try:
                response = await client.get(url, headers=headers)
                self.last_http_version = str(getattr(response, "http_version", "") or "")
                return int(response.status_code), response.text
            except _RETRYABLE_HTTP_ERRORS:
                await self.close()
                if attempt_idx == 0:
                    continue
                raise
        raise RuntimeError("unreachable")


@dataclass(frozen=True)
class TextGenerationResult:
    text: str | None
    raw_response: str | None = None
    meta: PatcherMeta | None = None


@dataclass(frozen=True)
class JsonGenerationResult:
    payload: dict[str, Any] | None
    raw_response: str | None = None
    meta: PatcherMeta | None = None


PATCH_EDITS_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["edits"],
    "additionalProperties": False,
    "properties": {
        "edits": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["path", "find", "replace"],
                "additionalProperties": False,
                "properties": {
                    "path": {"type": "string", "minLength": 1},
                    "find": {"type": "string"},
                    "replace": {"type": "string"},
                    "count": {"type": "integer", "minimum": 0},
                },
            },
        }
    },
}


def _normalize_base_url(base_url: str) -> str:
    base = base_url.rstrip("/")
    # Already has a versioned path (e.g. /v1, /v4) — don't append /v1.
    if base.endswith(("/v1", "/v2", "/v3", "/v4")):
        return base
    return base + "/v1"


def _origin_from_base_url(base_url: str) -> str:
    parsed = urllib.parse.urlsplit(base_url)
    if not parsed.scheme or not parsed.netloc:
        return base_url.rstrip("/")
    return f"{parsed.scheme}://{parsed.netloc}"


def _base_url_from_chat_url(url: str) -> str:
    suffix = "/chat/completions"
    if url.endswith(suffix):
        return url[: -len(suffix)]
    return url


def _first_nonempty_env(*keys: str) -> str:
    for key in keys:
        value = os.getenv(str(key), "")
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _default_api_key() -> str:
    return _first_nonempty_env(*OPENAI_API_KEY_ENV_FALLBACKS) or "local"


def _chat_http_client_key(url: str, *, http2: bool) -> tuple[str, str, int, bool]:
    parsed = urllib.parse.urlsplit(url)
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        raise ValueError(f"unsupported URL scheme for chat completions: {parsed.scheme!r}")
    host = parsed.hostname
    if not host:
        raise ValueError(f"missing host in chat completions URL: {url!r}")
    port = parsed.port or (443 if scheme == "https" else 80)
    return scheme, host, port, bool(http2)


def _chat_http_target(url: str) -> str:
    parsed = urllib.parse.urlsplit(url)
    path = parsed.path or "/"
    if parsed.query:
        return f"{path}?{parsed.query}"
    return path


def _should_use_proxy_transport(url: str) -> bool:
    parsed = urllib.parse.urlsplit(url)
    scheme = parsed.scheme.lower()
    if scheme not in {"http", "https"}:
        return False
    proxy_url = urllib.request.getproxies().get(scheme)
    if not proxy_url:
        return False
    host = parsed.hostname or ""
    return not urllib.request.proxy_bypass(host)


def _get_thread_http_client(*, url: str, http2: bool) -> _PersistentHTTPClient:
    cache = getattr(_THREAD_HTTP_CLIENTS, "clients", None)
    if not isinstance(cache, dict):
        cache = {}
        _THREAD_HTTP_CLIENTS.clients = cache
    key = _chat_http_client_key(url, http2=http2)
    client = cache.get(key)
    if not isinstance(client, _PersistentHTTPClient):
        client = _PersistentHTTPClient(scheme=key[0], host=key[1], port=key[2], http2=key[3])
        cache[key] = client
    return client


def _get_async_http_client(*, url: str, http2: bool) -> _PersistentAsyncHTTPClient:
    loop_id = id(asyncio.get_running_loop())
    key_tuple = _chat_http_client_key(url, http2=http2)
    cache_key = (loop_id, key_tuple[0], key_tuple[1], key_tuple[2], key_tuple[3])
    with _ASYNC_HTTP_CLIENTS_LOCK:
        client = _ASYNC_HTTP_CLIENTS.get(cache_key)
        if not isinstance(client, _PersistentAsyncHTTPClient):
            client = _PersistentAsyncHTTPClient(
                scheme=key_tuple[0],
                host=key_tuple[1],
                port=key_tuple[2],
                http2=key_tuple[3],
            )
            _ASYNC_HTTP_CLIENTS[cache_key] = client
    return client


def _is_http2_protocol_error(exc: Exception) -> bool:
    if isinstance(exc, _HTTP2_PROTOCOL_ERRORS):
        return True
    text = str(exc).lower()
    return "http/2" in text and any(marker in text for marker in ("protocol", "stream", "h2"))


def _nonnegative_int(value: Any, *, default: int = 0) -> int:
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        return int(default)
    return max(0, coerced)


def _optional_nonnegative_int(value: Any) -> int | None:
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        return None
    return max(0, coerced)


def _coerce_payload_bool(value: Any) -> bool:
    return coerce_stage_a_bool(value)


def _usage_counts_from_payload(usage: Any) -> tuple[int, int, int]:
    if not isinstance(usage, dict):
        return 0, 0, 0
    prompt_tokens = _nonnegative_int(usage.get("prompt_tokens"), default=0)
    completion_tokens = _nonnegative_int(usage.get("completion_tokens"), default=0)
    if prompt_tokens == 0 and completion_tokens == 0:
        prompt_tokens = _nonnegative_int(usage.get("input_tokens"), default=0)
        completion_tokens = _nonnegative_int(usage.get("output_tokens"), default=0)
    total_tokens = _nonnegative_int(
        usage.get("total_tokens"),
        default=prompt_tokens + completion_tokens,
    )
    return prompt_tokens, completion_tokens, total_tokens


def _reasoning_tokens_from_payload(usage: Any) -> tuple[int, bool]:
    if not isinstance(usage, dict):
        return 0, False
    candidates = [
        usage,
        usage.get("output_tokens_details"),
        usage.get("completion_tokens_details"),
    ]
    for candidate in candidates:
        if not isinstance(candidate, dict) or "reasoning_tokens" not in candidate:
            continue
        parsed = _optional_nonnegative_int(candidate.get("reasoning_tokens"))
        if parsed is not None:
            return parsed, True
    return 0, False


def _usage_payload_from_counts(
    *,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    reasoning_tokens: int | None = None,
) -> dict[str, int]:
    out = {
        "prompt_tokens": _nonnegative_int(prompt_tokens, default=0),
        "completion_tokens": _nonnegative_int(completion_tokens, default=0),
        "total_tokens": _nonnegative_int(
            total_tokens,
            default=_nonnegative_int(prompt_tokens, default=0)
            + _nonnegative_int(completion_tokens, default=0),
        ),
    }
    if reasoning_tokens is not None:
        out["reasoning_tokens"] = _nonnegative_int(reasoning_tokens, default=0)
    return out


def _llm_call_count_from_meta(meta: PatcherMeta | None, *, default: int = 0) -> int:
    if not isinstance(meta, dict):
        return _nonnegative_int(default, default=0)
    value = meta.get("llm_call_count")
    if isinstance(value, int):
        return _nonnegative_int(value, default=0)
    return _nonnegative_int(default, default=0)


def _accumulate_llm_totals(
    *,
    totals: tuple[int, int, int, int, int, bool, bool],
    meta: PatcherMeta | None,
    default_calls: int = 0,
) -> tuple[int, int, int, int, int, bool, bool]:
    (
        calls,
        prompt_tokens,
        completion_tokens,
        total_tokens,
        reasoning_tokens,
        reasoning_applicable,
        reasoning_coverage,
    ) = totals
    meta_calls = _llm_call_count_from_meta(meta, default=default_calls)
    meta_prompt, meta_completion, meta_total = _usage_counts_from_payload(
        meta.get("usage") if isinstance(meta, dict) else None
    )
    meta_reasoning, meta_reasoning_available = _reasoning_tokens_from_payload(
        meta.get("usage") if isinstance(meta, dict) else None
    )
    meta_reasoning_applicable = bool(
        meta_calls > 0 or meta_prompt > 0 or meta_completion > 0 or meta_total > 0
    )
    return (
        calls + meta_calls,
        prompt_tokens + meta_prompt,
        completion_tokens + meta_completion,
        total_tokens + meta_total,
        reasoning_tokens + (meta_reasoning if meta_reasoning_available else 0),
        reasoning_applicable or meta_reasoning_applicable,
        reasoning_coverage and (meta_reasoning_available or not meta_reasoning_applicable),
    )


def _meta_with_llm_totals(
    *,
    meta: PatcherMeta,
    totals: tuple[int, int, int, int, int, bool, bool],
) -> PatcherMeta:
    (
        calls,
        prompt_tokens,
        completion_tokens,
        total_tokens,
        reasoning_tokens,
        reasoning_applicable,
        reasoning_coverage,
    ) = totals
    out = cast(PatcherMeta, dict(meta))
    out["llm_call_count"] = _nonnegative_int(calls, default=0)
    out["usage"] = cast(
        UsagePayload,
        _usage_payload_from_counts(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
            reasoning_tokens=(
                reasoning_tokens if reasoning_applicable and reasoning_coverage else None
            ),
        ),
    )
    return out


def _request_context_system_prompt(*, strict_json: bool) -> str:
    base = (
        "Return only JSON for staged context requests. "
        "Use this schema exactly: "
        + STAGE_A_REQUEST_SCHEMA_COMPACT
        + " "
        + f"Allowed request types: {stage_a_allowed_request_types_text()}. "
        "Set need_compact_context=true only when compacted context is needed instead of raw bulk context. "
        "Use need_more_context=false with empty requests when evidence is sufficient. "
        "No prose and no markdown."
    )
    if not strict_json:
        return base
    return (
        base + " JSON only. Do not explain. Do not wrap the object in markdown. "
        "If unsure, return a valid object with need_more_context=false and requests=[]."
    )


def _plan_patch_system_prompt() -> str:
    return (
        "You are a software repair planner. "
        "Return only a concise plain-text repair plan with these headings exactly: "
        "Diagnosis, Why previous attempt failed, New constraint, Minimal edit plan, "
        "Verification plan. "
        "Do not return code, a diff, markdown fences, or any extra commentary."
    )


def _extract_unified_diff(text: str) -> str | None:
    s = _strip_think(text)
    s = s.strip("\n")
    if not s.strip():
        return None

    # Prefer fenced blocks if present.
    fence = re.search(r"```(?:diff|patch)?\n(.*?)\n```", s, re.DOTALL | re.IGNORECASE)
    if fence:
        candidate = fence.group(1).strip("\n")
        extracted = _extract_unified_diff(candidate)
        if extracted:
            return extracted

    lines = s.splitlines()

    # If it already looks like a git diff, keep from the first header.
    git = _extract_git_diff_blocks(lines)
    if git is not None:
        return git

    # Otherwise try to normalize "unified diff" blocks into git-apply format.
    normalized = _normalize_unified_blocks(lines)
    return normalized


def _strip_think(text: str) -> str:
    # Some reasoning models emit <think>...</think>; strip it.
    return re.sub(r"<think>.*?</think>\n?", "", text, flags=re.DOTALL | re.IGNORECASE)


def _normalize_unified_blocks(lines: list[str]) -> str | None:
    out_lines: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        if not line.startswith("--- "):
            i += 1
            continue
        if i + 1 >= n or not lines[i + 1].startswith("+++ "):
            i += 1
            continue

        old_raw = lines[i][4:].strip()
        new_raw = lines[i + 1][4:].strip()
        old_path = _clean_path_token(old_raw)
        new_path = _clean_path_token(new_raw)
        if old_path is None or new_path is None:
            i += 2
            continue

        # Avoid accidental renames in model output (common mistake).
        if old_path != "/dev/null" and new_path != old_path:
            new_path = old_path

        out_lines.append(f"diff --git a/{old_path} b/{new_path}")
        out_lines.append(f"--- a/{old_path}")
        out_lines.append(f"+++ b/{new_path}")

        # Copy hunks until next file header, but stop on non-diff junk.
        i += 2
        seen_hunk = False
        while i < n:
            nxt = lines[i]
            if nxt.startswith("--- ") and (i + 1) < n and lines[i + 1].startswith("+++ "):
                break
            if nxt.startswith("@@"):
                seen_hunk = True
                out_lines.append(nxt)
                i += 1
                continue
            if seen_hunk and (
                nxt.startswith(" ")
                or nxt.startswith("+")
                or nxt.startswith("-")
                or nxt.startswith("\\")
            ):
                out_lines.append(nxt)
                i += 1
                continue
            # End of diff content (model started explaining, etc).
            break
        # Ensure a blank line between file diffs for readability.
        if out_lines and out_lines[-1] != "":
            out_lines.append("")

    if not out_lines:
        return None

    text = "\n".join(out_lines).rstrip("\n") + "\n"
    if "diff --git " not in text:
        return None
    return text


def _clean_path_token(raw: str) -> str | None:
    if not raw:
        return None
    token = raw.strip()
    if not token:
        return None
    if "\t" in token:
        token = token.split("\t", 1)[0].rstrip()
    if len(token) >= 2 and token[0] == token[-1] == '"':
        token = token[1:-1]
    token = re.sub(r"\s+(?:---|\+\+\+)$", "", token).rstrip()

    # Common weird outputs:
    # - "--- buggy.py ---" (extra trailing token)
    # - "--- a/foo.py\t2026-..." (timestamps)
    if token in {"a/", "b/"}:
        return None
    token = token.removeprefix("a/")
    token = token.removeprefix("b/")
    if not token:
        return None
    if token == "/dev/null":
        return token
    return token


def _extract_git_diff_blocks(lines: list[str]) -> str | None:
    starts = [i for i, line in enumerate(lines) if line.startswith("diff --git ")]
    if not starts:
        return None

    captured: list[str] = []
    i = starts[0]
    n = len(lines)

    # Capture only "diff-like" lines; stop when model starts narrating.
    while i < n:
        line = lines[i]

        if line.startswith("diff --git "):
            captured.append(line)
            i += 1
            continue

        allowed = (
            line.startswith("index ")
            or line.startswith("new file mode ")
            or line.startswith("deleted file mode ")
            or line.startswith("old mode ")
            or line.startswith("new mode ")
            or line.startswith("similarity index ")
            or line.startswith("rename from ")
            or line.startswith("rename to ")
            or line.startswith("copy from ")
            or line.startswith("copy to ")
            or line.startswith("--- a/")
            or line == "--- /dev/null"
            or line.startswith("+++ b/")
            or line == "+++ /dev/null"
            or line.startswith("@@")
            or line.startswith(" ")
            or line.startswith("+")
            or line.startswith("-")
            or line.startswith("\\")
            or line == ""
        )
        if not allowed:
            break

        captured.append(line)
        i += 1

    text = "\n".join(captured).strip("\n") + "\n"
    if "diff --git " not in text:
        return None
    if "--- a/" not in text and "--- /dev/null" not in text:
        return None
    if "+++ b/" not in text and "+++ /dev/null" not in text:
        return None
    return text


@dataclass(frozen=True)
class OpenAICompatPatcher(Patcher):
    """OpenAI API-compatible chat-completions patcher.

    Routes patch-generation and context-gathering calls to any server that
    implements the OpenAI ``/v1/chat/completions`` REST API.  Tested against
    vLLM, LM Studio, llama.cpp, and the upstream OpenAI service.

    The class is an immutable :func:`~dataclasses.dataclass`; all fields are
    set at construction time and serve as the patcher's configuration.

    Attributes:
        base_url: Base URL of the OpenAI-compatible API server, e.g.
            ``"http://127.0.0.1:1234"``.  A ``/v1`` suffix is appended
            automatically when the URL does not already end with a versioned
            path segment.
        model: Model identifier to pass in every ``chat/completions`` request,
            e.g. ``"mistral-7b-instruct"`` or ``"gpt-4o"``.
        api_key: Bearer token sent in the ``Authorization`` header.  When
            ``None`` the value is resolved from the environment variables
            ``Z_API``, ``OPENAI_API_KEY``, or
            ``LLMDEBUG_EVAL_OPENAI_API_KEY`` (first non-empty wins), falling
            back to the string ``"local"``.
        timeout_sec: Per-request HTTP timeout in seconds.  Defaults to
            ``120.0``.
        http2: Whether to attempt an HTTP/2 connection.  Falls back to HTTP/1.1
            automatically on protocol negotiation errors.  Defaults to ``True``.
        transport_max_retries: Maximum number of additional transport-layer
            retry attempts on transient errors (connection resets, timeouts,
            retryable HTTP status codes such as 429/503).  Defaults to
            ``_DEFAULT_CHAT_COMPLETION_TRANSPORT_MAX_RETRIES`` (6).
        transport_retry_backoff_base_sec: Base delay in seconds for exponential
            back-off between transport retries.  Defaults to ``1.0``.
        transport_retry_backoff_max_sec: Upper cap on the exponential back-off
            delay.  Defaults to ``8.0``.
        max_propose_sec: Wall-clock deadline (seconds) for the entire
            :meth:`propose_patch` / :meth:`apropose_patch` call, measured from
            the moment the method is entered.  A value of ``0`` disables the
            deadline.  Defaults to ``300.0``.
        temperature: Sampling temperature forwarded to the model.  Use ``0.0``
            for deterministic (greedy) decoding.  Defaults to ``0.0``.
        max_tokens: Maximum number of tokens to generate per request.  Defaults
            to ``2048``.
        plan_patch_max_tokens: Native ``cot_repair`` planner token cap used by
            :meth:`plan_patch`. Defaults to ``768``.
        compat_profile: OpenAI-compatible request-shaping profile to apply
            before each request. ``"qwen_openai_raw"`` sends
            ``chat_template_kwargs.enable_thinking=false``;
            ``"qwen_dashscope"`` sends top-level ``enable_thinking=false``;
            ``"lmstudio_model_yaml"`` and ``"off"`` do not inject request-side
            controls. ``"auto"`` preserves existing default behaviour.
        allow_reasoning_fallback: When ``True``, reasoning-only responses are
            normalized into downstream content instead of surfacing a
            compatibility error. Defaults to ``False``.
        enable_thinking: Whether to request server-side thinking/reasoning mode
            when the chosen compatibility profile supports it. Enabling this
            also auto-enables reasoning fallback. Defaults to ``False``.
        response_mode: Controls the patch-generation strategy.  ``"edits"``
            (default) requests a structured JSON edits payload and builds the
            diff locally; ``"diff"`` requests a raw unified diff from the model.
        force_patch_attempt: When ``True`` and the model produces no usable
            diff, a deterministic minimal patch (appending a marker comment to
            the first non-test ``.py`` file) is constructed as a last resort.
            Defaults to ``False``.
        seed: Optional integer seed forwarded to the model for reproducibility.
            Not all servers honour this field.  Defaults to ``None``.
        name: Symbolic name for this patcher instance, used in run logs.
            Defaults to ``"openai"``.
    """

    base_url: str
    model: str
    api_key: str | None = None
    timeout_sec: float = 120.0
    http2: bool = True
    transport_max_retries: int = _DEFAULT_CHAT_COMPLETION_TRANSPORT_MAX_RETRIES
    transport_retry_backoff_base_sec: float = _DEFAULT_CHAT_COMPLETION_RETRY_BACKOFF_BASE_SEC
    transport_retry_backoff_max_sec: float = _DEFAULT_CHAT_COMPLETION_RETRY_BACKOFF_MAX_SEC
    max_propose_sec: float = _DEFAULT_MAX_PROPOSE_SEC
    temperature: float = 0.0
    max_tokens: int = 2048
    plan_patch_max_tokens: int = _PLAN_PATCH_MAX_TOKENS
    compat_profile: str = "auto"
    allow_reasoning_fallback: bool = False
    enable_thinking: bool = False
    response_mode: str = "edits"  # "edits" | "diff" | "full_file"
    force_patch_attempt: bool = False
    seed: int | None = None
    name: str = "openai"

    def supports_request_context(self) -> bool:
        """OpenAI-compatible patchers implement Stage-A helper requests."""
        return True

    def supports_context_compaction(self) -> bool:
        """OpenAI-compatible patchers implement delegated context compaction."""
        return True

    def supports_strict_json_first(self) -> bool:
        """OpenAI-compatible patchers support strict-JSON-first helper retries."""
        return True

    @_with_openai_transport_progress(
        external_operation="request_context",
        phase_label="openai: request context",
    )
    def request_context(
        self,
        *,
        prompt: str,
        context_dir: Path,
        cwd: Path,
        strict_json_first: bool = False,
    ) -> ContextRequestResult | None:
        """Ask the model which additional context it needs before patching.

        Stage A of the context-gathering pipeline.  Sends *prompt* to the
        ``chat/completions`` endpoint with a system instruction that constrains
        the response to the :data:`~evals.stage_a_contract.STAGE_A_REQUEST_SCHEMA_COMPACT`
        JSON schema.  The parsed payload declares which files, runtime
        snapshots, or other artefacts should be fetched and injected into the
        prompt before the model attempts to generate a patch.

        The method retries automatically on transient transport failures.  When
        the model returns a response that cannot be parsed as a valid
        :data:`ContextRequestPayload`, one additional attempt is made using a
        shorter ``strict_json`` system prompt and a reduced token budget.

        Args:
            prompt: The assembled debugging/repair prompt for the current
                repair attempt.
            context_dir: Directory containing pre-cached context artefacts.
                Not used by this implementation; reserved for the protocol.
            cwd: Working directory of the case under repair.  Not used by this
                implementation; reserved for the protocol.
            strict_json_first: When ``True`` the very first attempt already
                uses the strict-JSON system prompt and a reduced token budget
                (``_REQUEST_CONTEXT_STRICT_JSON_MAX_TOKENS``).  Useful when
                the model is known to be reliable at structured output.

        Returns:
            A :class:`~evals.patchers.base.ContextRequestResult` containing:

            - ``request_json``: the parsed :data:`ContextRequestPayload` on
              success, or ``None`` on failure.
            - ``raw_response``: the raw model output string.
            - ``meta``: telemetry dict with timing, retry counts, token usage,
              outcome classification (``request_context_outcome``), and
              per-attempt details (``request_context_attempts``).
        """
        del cwd
        base_url = _normalize_base_url(self.base_url)
        url = f"{base_url}/chat/completions"
        api_key = self.api_key
        if api_key is None:
            api_key = _default_api_key()

        start = time.perf_counter()
        totals = (0, 0, 0, 0, 0, False, True)
        helper_attempts: list[dict[str, Any]] = []
        transport_reissues_used = 0
        strict_json_retries_used = 0
        strict_json = bool(strict_json_first)
        raw_response: str | None = None
        request_json: ContextRequestPayload | None = None
        meta: PatcherMeta = {}

        while True:
            raw, attempt_meta = _request_context_via_json(
                url=url,
                api_key=api_key,
                model=self.model,
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=(
                    max(1, min(int(self.max_tokens), _REQUEST_CONTEXT_STRICT_JSON_MAX_TOKENS))
                    if strict_json
                    else self.max_tokens
                ),
                compat_profile=self.compat_profile,
                allow_reasoning_fallback=self.allow_reasoning_fallback,
                enable_thinking=self.enable_thinking,
                timeout_sec=self.timeout_sec,
                seed=self.seed,
                http2=self.http2,
                strict_json=strict_json,
                transport_max_retries=self.transport_max_retries,
                transport_retry_backoff_base_sec=self.transport_retry_backoff_base_sec,
                transport_retry_backoff_max_sec=self.transport_retry_backoff_max_sec,
            )
            totals = _accumulate_llm_totals(totals=totals, meta=attempt_meta, default_calls=0)
            meta = cast(PatcherMeta, dict(attempt_meta or {}))
            request_json, raw_response = _normalize_context_request_payload(raw)
            failure_kind = _request_context_failure_kind(
                meta=meta,
                request_json=request_json,
                raw_response=raw_response,
            )
            outcome = _request_context_outcome(
                meta=meta,
                request_json=request_json,
                raw_response=raw_response,
            )
            helper_attempts.append(
                {
                    "attempt": len(helper_attempts) + 1,
                    "strict_json": bool(strict_json),
                    "ok": request_json is not None,
                    "outcome": outcome,
                    "failure_kind": failure_kind,
                    "error": str(meta.get("error") or ""),
                    "status": meta.get("status"),
                    "llm_call_count": _llm_call_count_from_meta(meta, default=0),
                    "transport_retry_count": _nonnegative_int(
                        meta.get("transport_retry_count"), default=0
                    ),
                    "response_format_fallback_used": bool(
                        meta.get("request_context_response_format_fallback_used")
                    ),
                }
            )
            if request_json is not None:
                break
            if _is_openai_compat_reasoning_only(meta):
                break
            if (
                failure_kind in {"transport", "backend_unavailable"}
                and transport_reissues_used < _REQUEST_CONTEXT_MAX_TRANSPORT_REISSUES
            ):
                transport_reissues_used += 1
                strict_json = False
                continue
            if (
                failure_kind == "invalid_payload"
                and strict_json_retries_used < _REQUEST_CONTEXT_MAX_STRICT_JSON_RETRIES
                and not strict_json
            ):
                strict_json_retries_used += 1
                strict_json = True
                continue
            break
        duration_sec = time.perf_counter() - start

        result_meta: PatcherMeta = dict(meta or {})
        result_meta.setdefault("base_url", base_url)
        result_meta.setdefault("model", self.model)
        result_meta.setdefault("temperature", self.temperature)
        result_meta.setdefault("max_tokens", self.max_tokens)
        result_meta.setdefault(
            "compat_profile_requested", normalize_compat_profile(self.compat_profile)
        )
        result_meta.setdefault("response_mode", self.response_mode)
        result_meta.setdefault("api_compat_seed_requested", self.seed)
        result_meta.setdefault(
            "api_compat_seed_acknowledged", False if self.seed is not None else None
        )
        result_meta.setdefault("api_compat_seed_response", None)
        result_meta["request_context_duration_sec"] = duration_sec
        result_meta["request_context_attempts"] = helper_attempts
        result_meta["request_context_retry_count"] = max(0, len(helper_attempts) - 1)
        result_meta["request_context_transport_retry_count"] = transport_reissues_used
        result_meta["request_context_strict_retry_count"] = strict_json_retries_used
        result_meta["request_context_strict_json_first"] = bool(strict_json_first)
        failure_kind = _request_context_failure_kind(
            meta=result_meta,
            request_json=request_json,
            raw_response=raw_response,
        )
        outcome = _request_context_outcome(
            meta=result_meta,
            request_json=request_json,
            raw_response=raw_response,
        )
        result_meta["request_context_outcome"] = outcome
        result_meta["request_context_failure_kind"] = failure_kind
        result_meta["request_context_backend_unavailable"] = failure_kind == "backend_unavailable"
        result_meta["request_context_effective_success"] = request_json is not None
        result_meta = _meta_with_llm_totals(meta=result_meta, totals=totals)
        if request_json is None and raw_response is not None and "error" not in result_meta:
            result_meta["error"] = "invalid_request_context_payload"
        return ContextRequestResult(
            request_json=request_json,
            raw_response=raw_response,
            meta=result_meta,
        )

    @_with_openai_transport_progress(
        external_operation="plan_patch",
        phase_label="openai: plan patch",
    )
    def plan_patch(self, *, prompt: str, context_dir: Path, cwd: Path) -> PatchPlanResult | None:
        """Ask the model for a concise natural-language repair plan.

        Issues a single ``chat/completions`` request using a specialised system
        prompt that instructs the model to produce a plain-text plan structured
        around five headings: *Diagnosis*, *Why previous attempt failed*,
        *New constraint*, *Minimal edit plan*, and *Verification plan*.

        The returned plan text is intended to be prepended to the next
        :meth:`propose_patch` prompt so that the model is guided away from
        repeating the same mistake.  The token budget is capped at
        :attr:`plan_patch_max_tokens` regardless of :attr:`max_tokens`.

        Args:
            prompt: The repair/diagnostic prompt containing failure information
                from the previous attempt.
            context_dir: Directory containing pre-cached context artefacts.
                Not used by this implementation; reserved for the protocol.
            cwd: Working directory of the case under repair.  Not used by this
                implementation; reserved for the protocol.

        Returns:
            A :class:`~evals.patchers.base.PatchPlanResult` containing:

            - ``plan_text``: the stripped plan string on success, or ``None``
              when the model returns an empty response.
            - ``raw_response``: the raw model output string.
            - ``meta``: telemetry dict with timing, token usage, and the
              effective token cap used (``plan_patch_max_tokens``).
        """
        del cwd
        base_url = _normalize_base_url(self.base_url)
        url = f"{base_url}/chat/completions"
        api_key = self.api_key
        if api_key is None:
            api_key = _default_api_key()

        deadline_monotonic: float | None = None
        if self.max_propose_sec > 0:
            deadline_monotonic = time.monotonic() + self.max_propose_sec

        plan_max_tokens = max(1, min(int(self.max_tokens), int(self.plan_patch_max_tokens)))
        start = time.perf_counter()
        data, meta = _chat_completion(
            url=url,
            api_key=api_key,
            model=self.model,
            messages=[
                {"role": "system", "content": _plan_patch_system_prompt()},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=plan_max_tokens,
            compat_profile=self.compat_profile,
            allow_reasoning_fallback=self.allow_reasoning_fallback,
            enable_thinking=self.enable_thinking,
            timeout_sec=self.timeout_sec,
            seed=self.seed,
            http2=self.http2,
            deadline_monotonic=deadline_monotonic,
            transport_max_retries=self.transport_max_retries,
            transport_retry_backoff_base_sec=self.transport_retry_backoff_base_sec,
            transport_retry_backoff_max_sec=self.transport_retry_backoff_max_sec,
        )
        duration_sec = time.perf_counter() - start
        result_meta: PatcherMeta = dict(meta or {})
        result_meta.setdefault("base_url", base_url)
        result_meta.setdefault("model", self.model)
        result_meta.setdefault("temperature", self.temperature)
        result_meta.setdefault("timeout_sec", self.timeout_sec)
        result_meta.setdefault("transport_max_retries", self.transport_max_retries)
        result_meta.setdefault(
            "transport_retry_backoff_base_sec", self.transport_retry_backoff_base_sec
        )
        result_meta.setdefault(
            "transport_retry_backoff_max_sec", self.transport_retry_backoff_max_sec
        )
        result_meta.setdefault("max_propose_sec", self.max_propose_sec)
        result_meta.setdefault(
            "compat_profile_requested", normalize_compat_profile(self.compat_profile)
        )
        result_meta.setdefault("plan_patch_max_tokens_requested", int(self.plan_patch_max_tokens))
        result_meta.setdefault("api_compat_seed_requested", self.seed)
        result_meta.setdefault(
            "api_compat_seed_acknowledged", False if self.seed is not None else None
        )
        result_meta.setdefault("api_compat_seed_response", None)
        result_meta["plan_patch_duration_sec"] = duration_sec
        result_meta["plan_patch_max_tokens"] = plan_max_tokens

        raw_response = _get_first_choice_content(data) if isinstance(data, dict) else None
        plan_text = (
            raw_response.strip() if isinstance(raw_response, str) and raw_response.strip() else None
        )
        if plan_text is None and "error" not in result_meta:
            result_meta["error"] = "empty_plan_response"
        if plan_text is not None:
            result_meta["plan_patch_output_chars"] = len(plan_text)
        return PatchPlanResult(plan_text=plan_text, raw_response=raw_response, meta=result_meta)

    def generate_text(
        self,
        *,
        system_prompt: str,
        prompt: str,
        context_dir: Path,
        cwd: Path,
        temperature: float | None = None,
        max_tokens: int | None = None,
        operation_name: str = "generate_text",
    ) -> TextGenerationResult:
        del context_dir
        del cwd
        base_url = _normalize_base_url(self.base_url)
        url = f"{base_url}/chat/completions"
        api_key = self.api_key
        if api_key is None:
            api_key = _default_api_key()

        selected_temperature = self.temperature if temperature is None else float(temperature)
        selected_max_tokens = self.max_tokens if max_tokens is None else max(1, int(max_tokens))
        start = time.perf_counter()
        raw, meta = _chat_completion(
            url=url,
            api_key=api_key,
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=selected_temperature,
            max_tokens=selected_max_tokens,
            compat_profile=self.compat_profile,
            allow_reasoning_fallback=self.allow_reasoning_fallback,
            enable_thinking=self.enable_thinking,
            timeout_sec=self.timeout_sec,
            seed=self.seed,
            http2=self.http2,
            transport_max_retries=self.transport_max_retries,
            transport_retry_backoff_base_sec=self.transport_retry_backoff_base_sec,
            transport_retry_backoff_max_sec=self.transport_retry_backoff_max_sec,
        )
        duration_sec = time.perf_counter() - start
        result_meta: PatcherMeta = dict(meta or {})
        result_meta.setdefault("base_url", base_url)
        result_meta.setdefault("model", self.model)
        result_meta.setdefault("temperature", selected_temperature)
        result_meta.setdefault("timeout_sec", self.timeout_sec)
        result_meta["text_generation_operation"] = operation_name
        result_meta["text_generation_duration_sec"] = duration_sec
        result_meta["text_generation_max_tokens"] = selected_max_tokens
        if raw is None:
            return TextGenerationResult(text=None, raw_response=None, meta=result_meta)
        content = _get_first_choice_content(raw)
        if content is None:
            result_meta.setdefault("error", "no_choice_content")
            return TextGenerationResult(text=None, raw_response=_safe_trunc(raw), meta=result_meta)
        return TextGenerationResult(text=content, raw_response=content, meta=result_meta)

    def generate_json(
        self,
        *,
        system_prompt: str,
        prompt: str,
        context_dir: Path,
        cwd: Path,
        schema_name: str,
        schema: dict[str, Any],
        temperature: float | None = None,
        max_tokens: int | None = None,
        operation_name: str = "generate_json",
    ) -> JsonGenerationResult:
        del context_dir
        del cwd
        base_url = _normalize_base_url(self.base_url)
        url = f"{base_url}/chat/completions"
        api_key = self.api_key
        if api_key is None:
            api_key = _default_api_key()

        selected_temperature = self.temperature if temperature is None else float(temperature)
        selected_max_tokens = self.max_tokens if max_tokens is None else max(1, int(max_tokens))
        start = time.perf_counter()
        totals = (0, 0, 0, 0, 0, False, True)
        raw, meta = _chat_completion(
            url=url,
            api_key=api_key,
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=selected_temperature,
            max_tokens=selected_max_tokens,
            compat_profile=self.compat_profile,
            allow_reasoning_fallback=self.allow_reasoning_fallback,
            enable_thinking=self.enable_thinking,
            timeout_sec=self.timeout_sec,
            seed=self.seed,
            http2=self.http2,
            extra_payload={
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": schema_name, "schema": schema},
                }
            },
            transport_max_retries=self.transport_max_retries,
            transport_retry_backoff_base_sec=self.transport_retry_backoff_base_sec,
            transport_retry_backoff_max_sec=self.transport_retry_backoff_max_sec,
        )
        totals = _accumulate_llm_totals(totals=totals, meta=meta, default_calls=1)
        if raw is None:
            err = str(meta.get("error", ""))
            body = str(meta.get("body", ""))
            if err == "http_error" and "response_format" in body:
                raw, meta = _chat_completion(
                    url=url,
                    api_key=api_key,
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=selected_temperature,
                    max_tokens=selected_max_tokens,
                    compat_profile=self.compat_profile,
                    allow_reasoning_fallback=self.allow_reasoning_fallback,
                    enable_thinking=self.enable_thinking,
                    timeout_sec=self.timeout_sec,
                    seed=self.seed,
                    http2=self.http2,
                    extra_payload={"response_format": {"type": "json_object"}},
                    transport_max_retries=self.transport_max_retries,
                    transport_retry_backoff_base_sec=self.transport_retry_backoff_base_sec,
                    transport_retry_backoff_max_sec=self.transport_retry_backoff_max_sec,
                )
                totals = _accumulate_llm_totals(totals=totals, meta=meta, default_calls=1)
        duration_sec = time.perf_counter() - start
        result_meta: PatcherMeta = _meta_with_llm_totals(
            meta=cast(PatcherMeta, dict(meta or {})),
            totals=totals,
        )
        result_meta.setdefault("base_url", base_url)
        result_meta.setdefault("model", self.model)
        result_meta.setdefault("temperature", selected_temperature)
        result_meta.setdefault("timeout_sec", self.timeout_sec)
        result_meta["json_generation_operation"] = operation_name
        result_meta["json_generation_duration_sec"] = duration_sec
        result_meta["json_generation_max_tokens"] = selected_max_tokens
        if raw is None:
            return JsonGenerationResult(payload=None, raw_response=None, meta=result_meta)
        content = _get_first_choice_content(raw)
        if content is None:
            result_meta.setdefault("error", "no_choice_content")
            return JsonGenerationResult(
                payload=None, raw_response=_safe_trunc(raw), meta=result_meta
            )
        payload = _extract_json_object(content)
        if payload is None:
            result_meta.setdefault("error", "no_json_payload")
            return JsonGenerationResult(payload=None, raw_response=content[:2000], meta=result_meta)
        return JsonGenerationResult(payload=payload, raw_response=content, meta=result_meta)

    @_with_openai_transport_progress_async(
        external_operation="request_context",
        phase_label="openai: request context",
    )
    async def arequest_context(
        self,
        *,
        prompt: str,
        context_dir: Path,
        cwd: Path,
        strict_json_first: bool = False,
    ) -> ContextRequestResult | None:
        """Async variant of :meth:`request_context`.

        Ask the model which additional context it needs before patching,
        without blocking the running event loop.  Behaviour and return value
        are identical to :meth:`request_context`; see that method for full
        parameter and return-value documentation.

        Args:
            prompt: The assembled debugging/repair prompt for the current
                repair attempt.
            context_dir: Directory containing pre-cached context artefacts.
                Not used by this implementation; reserved for the protocol.
            cwd: Working directory of the case under repair.  Not used by this
                implementation; reserved for the protocol.
            strict_json_first: When ``True`` the very first attempt already
                uses the strict-JSON system prompt and a reduced token budget.

        Returns:
            A :class:`~evals.patchers.base.ContextRequestResult` — see
            :meth:`request_context` for field descriptions.
        """
        del context_dir
        del cwd
        base_url = _normalize_base_url(self.base_url)
        url = f"{base_url}/chat/completions"
        api_key = self.api_key
        if api_key is None:
            api_key = _default_api_key()

        start = time.perf_counter()
        totals = (0, 0, 0, 0, 0, False, True)
        helper_attempts: list[dict[str, Any]] = []
        transport_reissues_used = 0
        strict_json_retries_used = 0
        strict_json = bool(strict_json_first)
        raw_response: str | None = None
        request_json: ContextRequestPayload | None = None
        meta: PatcherMeta = {}

        while True:
            raw, attempt_meta = await _request_context_via_json_async(
                url=url,
                api_key=api_key,
                model=self.model,
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=(
                    max(1, min(int(self.max_tokens), _REQUEST_CONTEXT_STRICT_JSON_MAX_TOKENS))
                    if strict_json
                    else self.max_tokens
                ),
                compat_profile=self.compat_profile,
                allow_reasoning_fallback=self.allow_reasoning_fallback,
                enable_thinking=self.enable_thinking,
                timeout_sec=self.timeout_sec,
                seed=self.seed,
                http2=self.http2,
                strict_json=strict_json,
                transport_max_retries=self.transport_max_retries,
                transport_retry_backoff_base_sec=self.transport_retry_backoff_base_sec,
                transport_retry_backoff_max_sec=self.transport_retry_backoff_max_sec,
            )
            totals = _accumulate_llm_totals(totals=totals, meta=attempt_meta, default_calls=0)
            meta = cast(PatcherMeta, dict(attempt_meta or {}))
            request_json, raw_response = _normalize_context_request_payload(raw)
            failure_kind = _request_context_failure_kind(
                meta=meta,
                request_json=request_json,
                raw_response=raw_response,
            )
            outcome = _request_context_outcome(
                meta=meta,
                request_json=request_json,
                raw_response=raw_response,
            )
            helper_attempts.append(
                {
                    "attempt": len(helper_attempts) + 1,
                    "strict_json": bool(strict_json),
                    "ok": request_json is not None,
                    "outcome": outcome,
                    "failure_kind": failure_kind,
                    "error": str(meta.get("error") or ""),
                    "status": meta.get("status"),
                    "llm_call_count": _llm_call_count_from_meta(meta, default=0),
                    "transport_retry_count": _nonnegative_int(
                        meta.get("transport_retry_count"), default=0
                    ),
                    "response_format_fallback_used": bool(
                        meta.get("request_context_response_format_fallback_used")
                    ),
                }
            )
            if request_json is not None:
                break
            if _is_openai_compat_reasoning_only(meta):
                break
            if (
                failure_kind in {"transport", "backend_unavailable"}
                and transport_reissues_used < _REQUEST_CONTEXT_MAX_TRANSPORT_REISSUES
            ):
                transport_reissues_used += 1
                strict_json = False
                continue
            if (
                failure_kind == "invalid_payload"
                and strict_json_retries_used < _REQUEST_CONTEXT_MAX_STRICT_JSON_RETRIES
                and not strict_json
            ):
                strict_json_retries_used += 1
                strict_json = True
                continue
            break
        duration_sec = time.perf_counter() - start

        result_meta: PatcherMeta = dict(meta or {})
        result_meta.setdefault("base_url", base_url)
        result_meta.setdefault("model", self.model)
        result_meta.setdefault("temperature", self.temperature)
        result_meta.setdefault("max_tokens", self.max_tokens)
        result_meta.setdefault(
            "compat_profile_requested", normalize_compat_profile(self.compat_profile)
        )
        result_meta.setdefault("response_mode", self.response_mode)
        result_meta.setdefault("api_compat_seed_requested", self.seed)
        result_meta.setdefault(
            "api_compat_seed_acknowledged", False if self.seed is not None else None
        )
        result_meta.setdefault("api_compat_seed_response", None)
        result_meta["request_context_duration_sec"] = duration_sec
        result_meta["request_context_attempts"] = helper_attempts
        result_meta["request_context_retry_count"] = max(0, len(helper_attempts) - 1)
        result_meta["request_context_transport_retry_count"] = transport_reissues_used
        result_meta["request_context_strict_retry_count"] = strict_json_retries_used
        result_meta["request_context_strict_json_first"] = bool(strict_json_first)
        failure_kind = _request_context_failure_kind(
            meta=result_meta,
            request_json=request_json,
            raw_response=raw_response,
        )
        outcome = _request_context_outcome(
            meta=result_meta,
            request_json=request_json,
            raw_response=raw_response,
        )
        result_meta["request_context_outcome"] = outcome
        result_meta["request_context_failure_kind"] = failure_kind
        result_meta["request_context_backend_unavailable"] = failure_kind == "backend_unavailable"
        result_meta["request_context_effective_success"] = request_json is not None
        result_meta = _meta_with_llm_totals(meta=result_meta, totals=totals)
        if request_json is None and raw_response is not None and "error" not in result_meta:
            result_meta["error"] = "invalid_request_context_payload"
        return ContextRequestResult(
            request_json=request_json,
            raw_response=raw_response,
            meta=result_meta,
        )

    @_with_openai_transport_progress(
        external_operation="compact_context",
        phase_label="openai: compact context",
    )
    def compact_context(
        self,
        *,
        prompt: str,
        context_text: str,
        context_dir: Path,
        cwd: Path,
        target_chars: int,
    ) -> ContextCompactionResult | None:
        """Ask the model to compress a context block to a target character budget.

        Sends *context_text* together with *prompt* to the ``chat/completions``
        endpoint.  The system instruction constrains the response to a JSON
        payload with a ``compacted_context`` field whose value must not exceed
        *target_chars* characters.  Critical diagnostic information (tracebacks,
        file paths, assertion messages, relevant state) is to be preserved;
        low-signal content is dropped.

        After the model responds the compacted text is hard-truncated to
        *target_chars* as a safety measure regardless of what the model reports.

        Falls back gracefully when the server does not support
        ``response_format: json_object`` by re-issuing the request without that
        constraint.

        Args:
            prompt: The repair prompt that provides task context for the
                compaction (used to anchor what is considered "critical").
            context_text: The full context string to be compressed.
            context_dir: Directory containing pre-cached context artefacts.
                Not used by this implementation; reserved for the protocol.
            cwd: Working directory of the case under repair.  Not used by this
                implementation; reserved for the protocol.
            target_chars: Maximum character length of the returned compacted
                text.  Must be a positive integer.

        Returns:
            A :class:`~evals.patchers.base.ContextCompactionResult` containing:

            - ``compacted_text``: the compressed and hard-truncated text on
              success, or ``None`` on failure.
            - ``raw_response``: the raw model output string.
            - ``meta``: telemetry dict with timing, input/output character
              counts, truncation flag, and token usage.
        """
        del cwd
        base_url = _normalize_base_url(self.base_url)
        url = f"{base_url}/chat/completions"
        api_key = self.api_key
        if api_key is None:
            api_key = _default_api_key()

        target_chars_safe = max(1, int(target_chars))
        start = time.perf_counter()
        compacted_text, raw_response, meta = _compact_context_via_json(
            url=url,
            api_key=api_key,
            model=self.model,
            prompt=prompt,
            context_text=context_text,
            target_chars=target_chars_safe,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            compat_profile=self.compat_profile,
            allow_reasoning_fallback=self.allow_reasoning_fallback,
            enable_thinking=self.enable_thinking,
            timeout_sec=self.timeout_sec,
            seed=self.seed,
            http2=self.http2,
            transport_max_retries=self.transport_max_retries,
            transport_retry_backoff_base_sec=self.transport_retry_backoff_base_sec,
            transport_retry_backoff_max_sec=self.transport_retry_backoff_max_sec,
        )
        duration_sec = time.perf_counter() - start

        result_meta: PatcherMeta = dict(meta or {})
        result_meta.setdefault("base_url", base_url)
        result_meta.setdefault("model", self.model)
        result_meta.setdefault("temperature", self.temperature)
        result_meta.setdefault("max_tokens", self.max_tokens)
        result_meta.setdefault(
            "compat_profile_requested", normalize_compat_profile(self.compat_profile)
        )
        result_meta.setdefault("response_mode", self.response_mode)
        result_meta.setdefault("api_compat_seed_requested", self.seed)
        result_meta.setdefault(
            "api_compat_seed_acknowledged", False if self.seed is not None else None
        )
        result_meta.setdefault("api_compat_seed_response", None)
        result_meta["compact_context_duration_sec"] = duration_sec
        result_meta["compact_context_target_chars"] = target_chars_safe
        result_meta["compact_context_input_chars"] = len(context_text or "")

        if compacted_text is None:
            if raw_response is not None and "error" not in result_meta:
                result_meta["error"] = "invalid_context_compaction_payload"
            return ContextCompactionResult(
                compacted_text=None,
                raw_response=raw_response,
                meta=result_meta,
            )

        trimmed = compacted_text[:target_chars_safe]
        result_meta["compact_context_output_chars"] = len(compacted_text)
        result_meta["compact_context_output_truncated"] = len(trimmed) < len(compacted_text)
        return ContextCompactionResult(
            compacted_text=trimmed,
            raw_response=raw_response,
            meta=result_meta,
        )

    @_with_openai_transport_progress_async(
        external_operation="compact_context",
        phase_label="openai: compact context",
    )
    async def acompact_context(
        self,
        *,
        prompt: str,
        context_text: str,
        context_dir: Path,
        cwd: Path,
        target_chars: int,
    ) -> ContextCompactionResult | None:
        """Async variant of :meth:`compact_context`.

        Ask the model to compress a context block to a target character budget
        without blocking the running event loop.  Behaviour and return value
        are identical to :meth:`compact_context`; see that method for full
        parameter and return-value documentation.

        Args:
            prompt: The repair prompt providing task context for compaction.
            context_text: The full context string to be compressed.
            context_dir: Directory containing pre-cached context artefacts.
                Not used by this implementation; reserved for the protocol.
            cwd: Working directory of the case under repair.  Not used by this
                implementation; reserved for the protocol.
            target_chars: Maximum character length of the returned compacted
                text.

        Returns:
            A :class:`~evals.patchers.base.ContextCompactionResult` — see
            :meth:`compact_context` for field descriptions.
        """
        del cwd
        base_url = _normalize_base_url(self.base_url)
        url = f"{base_url}/chat/completions"
        api_key = self.api_key
        if api_key is None:
            api_key = _default_api_key()

        target_chars_safe = max(1, int(target_chars))
        start = time.perf_counter()
        compacted_text, raw_response, meta = await _compact_context_via_json_async(
            url=url,
            api_key=api_key,
            model=self.model,
            prompt=prompt,
            context_text=context_text,
            target_chars=target_chars_safe,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            compat_profile=self.compat_profile,
            allow_reasoning_fallback=self.allow_reasoning_fallback,
            enable_thinking=self.enable_thinking,
            timeout_sec=self.timeout_sec,
            seed=self.seed,
            http2=self.http2,
            transport_max_retries=self.transport_max_retries,
            transport_retry_backoff_base_sec=self.transport_retry_backoff_base_sec,
            transport_retry_backoff_max_sec=self.transport_retry_backoff_max_sec,
        )
        duration_sec = time.perf_counter() - start

        result_meta: PatcherMeta = dict(meta or {})
        result_meta.setdefault("base_url", base_url)
        result_meta.setdefault("model", self.model)
        result_meta.setdefault("temperature", self.temperature)
        result_meta.setdefault("max_tokens", self.max_tokens)
        result_meta.setdefault(
            "compat_profile_requested", normalize_compat_profile(self.compat_profile)
        )
        result_meta.setdefault("response_mode", self.response_mode)
        result_meta.setdefault("api_compat_seed_requested", self.seed)
        result_meta.setdefault(
            "api_compat_seed_acknowledged", False if self.seed is not None else None
        )
        result_meta.setdefault("api_compat_seed_response", None)
        result_meta["compact_context_duration_sec"] = duration_sec
        result_meta["compact_context_target_chars"] = target_chars_safe
        result_meta["compact_context_input_chars"] = len(context_text or "")

        if compacted_text is None:
            if raw_response is not None and "error" not in result_meta:
                result_meta["error"] = "invalid_context_compaction_payload"
            return ContextCompactionResult(
                compacted_text=None,
                raw_response=raw_response,
                meta=result_meta,
            )

        trimmed = compacted_text[:target_chars_safe]
        result_meta["compact_context_output_chars"] = len(compacted_text)
        result_meta["compact_context_output_truncated"] = len(trimmed) < len(compacted_text)
        return ContextCompactionResult(
            compacted_text=trimmed,
            raw_response=raw_response,
            meta=result_meta,
        )

    @_with_openai_transport_progress(
        external_operation="propose_patch",
        phase_label="openai: propose patch",
    )
    def propose_patch(
        self,
        *,
        prompt: str,
        context_dir: Path,
        cwd: Path,
        system_prompt_override: str | None = None,
    ) -> PatchResult:
        """Generate a patch for the failing test case at *cwd*.

        Dispatches to one of three sub-strategies based on :attr:`response_mode`:

        * ``"edits"`` — calls :func:`_propose_patch_via_edits`, which asks the
          model for a JSON ``{"edits": [...]}`` payload and attempts to obtain a
          valid response across three ``response_format`` escalation stages:
          ``json_schema`` → ``json_object`` → plain text.  Once a valid payload
          is received the find-replace edits are applied to the source files
          under *cwd* and a unified diff is generated locally.  Falls back to
          the ``"diff"`` strategy automatically when no edits can be applied
          (e.g. ``find`` string not found).
        * ``"diff"`` — calls :func:`_propose_patch_via_diff`, which asks the
          model to emit a raw unified diff and makes up to two attempts with
          progressively stronger formatting instructions.  Falls back to the
          ``"edits"`` strategy when no parseable diff is found in the output.
        * ``"full_file"`` — calls :func:`_propose_patch_via_full_file`, which
          asks the model for the complete repaired contents of a single target
          source file and derives a unified diff locally. Unsupported case
          shapes and unusable responses fall back to the ``"diff"`` strategy.

        When :attr:`force_patch_attempt` is ``True`` and both strategies fail
        to produce a usable diff, a deterministic minimal patch (a marker
        comment appended to the first non-test ``.py`` file listed in the
        prompt's file index) is constructed as a last resort — unless the
        failure is attributed to a transport error, context-budget exhaustion,
        or context-length overflow.

        Args:
            prompt: The assembled repair prompt including file contents, pytest
                output, and any gathered context.
            context_dir: Directory containing pre-cached context artefacts.
                Not used by this implementation; reserved for the protocol.
            cwd: Working directory of the case under repair.  Used by the
                ``"edits"`` strategy to resolve relative file paths and to
                enforce that no edit escapes the case directory.

        Returns:
            A :class:`~evals.patchers.base.PatchResult` containing:

            - ``diff``: a unified diff string on success, or ``None`` when no
              patch could be generated.
            - ``meta``: telemetry dict including the response mode, token
              usage, transport retry details, structured-output attempt log,
              fallback chain metadata, and forced-attempt flags if applicable.
        """
        base_url = _normalize_base_url(self.base_url)
        url = f"{base_url}/chat/completions"

        api_key = self.api_key
        if api_key is None:
            api_key = _default_api_key()

        deadline_monotonic: float | None = None
        if self.max_propose_sec > 0:
            deadline_monotonic = time.monotonic() + self.max_propose_sec

        if self.response_mode == "edits":
            result = _propose_patch_via_edits(
                url=url,
                api_key=api_key,
                model=self.model,
                prompt=prompt,
                cwd=cwd,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                compat_profile=self.compat_profile,
                allow_reasoning_fallback=self.allow_reasoning_fallback,
                enable_thinking=self.enable_thinking,
                timeout_sec=self.timeout_sec,
                seed=self.seed,
                http2=self.http2,
                deadline_monotonic=deadline_monotonic,
                system_prompt_override=system_prompt_override,
                transport_max_retries=self.transport_max_retries,
                transport_retry_backoff_base_sec=self.transport_retry_backoff_base_sec,
                transport_retry_backoff_max_sec=self.transport_retry_backoff_max_sec,
            )
        elif self.response_mode == "full_file":
            result = _propose_patch_via_full_file(
                url=url,
                api_key=api_key,
                model=self.model,
                prompt=prompt,
                cwd=cwd,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                compat_profile=self.compat_profile,
                allow_reasoning_fallback=self.allow_reasoning_fallback,
                enable_thinking=self.enable_thinking,
                timeout_sec=self.timeout_sec,
                seed=self.seed,
                http2=self.http2,
                deadline_monotonic=deadline_monotonic,
                system_prompt_override=system_prompt_override,
                transport_max_retries=self.transport_max_retries,
                transport_retry_backoff_base_sec=self.transport_retry_backoff_base_sec,
                transport_retry_backoff_max_sec=self.transport_retry_backoff_max_sec,
            )
        else:
            diff_result = _propose_patch_via_diff(
                url=url,
                api_key=api_key,
                model=self.model,
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                compat_profile=self.compat_profile,
                allow_reasoning_fallback=self.allow_reasoning_fallback,
                enable_thinking=self.enable_thinking,
                timeout_sec=self.timeout_sec,
                seed=self.seed,
                http2=self.http2,
                deadline_monotonic=deadline_monotonic,
                system_prompt_override=system_prompt_override,
                transport_max_retries=self.transport_max_retries,
                transport_retry_backoff_base_sec=self.transport_retry_backoff_base_sec,
                transport_retry_backoff_max_sec=self.transport_retry_backoff_max_sec,
            )
            if (
                diff_result.diff is None
                and str((diff_result.meta or {}).get("error", "")) == "no_diff_found"
            ):
                edits_fallback = _propose_patch_via_edits(
                    url=url,
                    api_key=api_key,
                    model=self.model,
                    prompt=prompt,
                    cwd=cwd,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    compat_profile=self.compat_profile,
                    allow_reasoning_fallback=self.allow_reasoning_fallback,
                    enable_thinking=self.enable_thinking,
                    timeout_sec=self.timeout_sec,
                    seed=self.seed,
                    http2=self.http2,
                    deadline_monotonic=deadline_monotonic,
                    allow_diff_fallback=False,
                    system_prompt_override=system_prompt_override,
                    transport_max_retries=self.transport_max_retries,
                    transport_retry_backoff_base_sec=self.transport_retry_backoff_base_sec,
                    transport_retry_backoff_max_sec=self.transport_retry_backoff_max_sec,
                )
                totals = (0, 0, 0, 0, 0, False, True)
                diff_meta = dict(diff_result.meta or {})
                edits_meta = dict(edits_fallback.meta or {})
                totals = _accumulate_llm_totals(totals=totals, meta=diff_meta, default_calls=0)
                totals = _accumulate_llm_totals(totals=totals, meta=edits_meta, default_calls=0)
                merged_meta = dict(edits_meta)
                merged_meta["fallback_from"] = "diff"
                merged_meta["fallback_reason"] = "no_diff_found"
                merged_meta["fallback_meta"] = diff_meta
                if merged_meta.get("error") is None:
                    merged_meta.pop("error", None)
                result = PatchResult(
                    diff=edits_fallback.diff,
                    meta=_meta_with_llm_totals(meta=merged_meta, totals=totals),
                )
            else:
                result = diff_result

        result_meta = dict(result.meta or {})
        result_meta.setdefault("base_url", base_url)
        result_meta.setdefault("model", self.model)
        result_meta.setdefault("temperature", self.temperature)
        result_meta.setdefault("max_tokens", self.max_tokens)
        result_meta.setdefault(
            "compat_profile_requested", normalize_compat_profile(self.compat_profile)
        )
        result_meta.setdefault("timeout_sec", self.timeout_sec)
        result_meta.setdefault("transport_max_retries", self.transport_max_retries)
        result_meta.setdefault(
            "transport_retry_backoff_base_sec", self.transport_retry_backoff_base_sec
        )
        result_meta.setdefault(
            "transport_retry_backoff_max_sec", self.transport_retry_backoff_max_sec
        )
        result_meta.setdefault("max_propose_sec", self.max_propose_sec)
        result_meta.setdefault("response_mode", self.response_mode)
        result_meta.setdefault("api_compat_seed_requested", self.seed)
        result_meta.setdefault(
            "api_compat_seed_acknowledged", False if self.seed is not None else None
        )
        result_meta.setdefault("api_compat_seed_response", None)
        result = PatchResult(diff=result.diff, meta=result_meta)

        if result.diff is not None or not self.force_patch_attempt:
            return result

        skip_reason = _forced_patch_skip_reason(cast(PatcherMeta, result_meta))
        if skip_reason:
            meta = dict(result.meta or {})
            meta["forced_patch_skipped"] = True
            meta["forced_patch_skip_reason"] = _forced_patch_skip_category(skip_reason)
            meta["forced_patch_skip_detail"] = skip_reason
            return PatchResult(diff=None, meta=meta)

        forced = _build_forced_attempt_diff(prompt=prompt, cwd=cwd)
        if forced is None:
            return result

        meta = dict(result.meta or {})
        meta["forced_patch_attempt"] = True
        meta["forced_patch_target"] = forced[0]
        meta["forced_patch_reason"] = "no_model_diff"
        return PatchResult(diff=forced[1], meta=meta)

    @_with_openai_transport_progress_async(
        external_operation="propose_patch",
        phase_label="openai: propose patch",
    )
    async def apropose_patch(self, *, prompt: str, context_dir: Path, cwd: Path) -> PatchResult:
        """Async variant of :meth:`propose_patch`.

        Generate a patch for the failing test case at *cwd* without blocking
        the running event loop.  Behaviour, fallback logic, and return value
        are identical to :meth:`propose_patch`; see that method for full
        parameter and return-value documentation.

        Args:
            prompt: The assembled repair prompt including file contents, pytest
                output, and any gathered context.
            context_dir: Directory containing pre-cached context artefacts.
                Not used by this implementation; reserved for the protocol.
            cwd: Working directory of the case under repair.  Used to resolve
                and validate relative file paths in the ``"edits"`` strategy.

        Returns:
            A :class:`~evals.patchers.base.PatchResult` — see
            :meth:`propose_patch` for field descriptions.
        """
        base_url = _normalize_base_url(self.base_url)
        url = f"{base_url}/chat/completions"

        api_key = self.api_key
        if api_key is None:
            api_key = _default_api_key()

        deadline_monotonic: float | None = None
        if self.max_propose_sec > 0:
            deadline_monotonic = time.monotonic() + self.max_propose_sec

        if self.response_mode == "edits":
            result = await _propose_patch_via_edits_async(
                url=url,
                api_key=api_key,
                model=self.model,
                prompt=prompt,
                cwd=cwd,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                compat_profile=self.compat_profile,
                allow_reasoning_fallback=self.allow_reasoning_fallback,
                enable_thinking=self.enable_thinking,
                timeout_sec=self.timeout_sec,
                seed=self.seed,
                http2=self.http2,
                deadline_monotonic=deadline_monotonic,
                transport_max_retries=self.transport_max_retries,
                transport_retry_backoff_base_sec=self.transport_retry_backoff_base_sec,
                transport_retry_backoff_max_sec=self.transport_retry_backoff_max_sec,
            )
        elif self.response_mode == "full_file":
            result = await _propose_patch_via_full_file_async(
                url=url,
                api_key=api_key,
                model=self.model,
                prompt=prompt,
                cwd=cwd,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                compat_profile=self.compat_profile,
                allow_reasoning_fallback=self.allow_reasoning_fallback,
                enable_thinking=self.enable_thinking,
                timeout_sec=self.timeout_sec,
                seed=self.seed,
                http2=self.http2,
                deadline_monotonic=deadline_monotonic,
                transport_max_retries=self.transport_max_retries,
                transport_retry_backoff_base_sec=self.transport_retry_backoff_base_sec,
                transport_retry_backoff_max_sec=self.transport_retry_backoff_max_sec,
            )
        else:
            diff_result = await _propose_patch_via_diff_async(
                url=url,
                api_key=api_key,
                model=self.model,
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                compat_profile=self.compat_profile,
                allow_reasoning_fallback=self.allow_reasoning_fallback,
                enable_thinking=self.enable_thinking,
                timeout_sec=self.timeout_sec,
                seed=self.seed,
                http2=self.http2,
                deadline_monotonic=deadline_monotonic,
                transport_max_retries=self.transport_max_retries,
                transport_retry_backoff_base_sec=self.transport_retry_backoff_base_sec,
                transport_retry_backoff_max_sec=self.transport_retry_backoff_max_sec,
            )
            if (
                diff_result.diff is None
                and str((diff_result.meta or {}).get("error", "")) == "no_diff_found"
            ):
                edits_fallback = await _propose_patch_via_edits_async(
                    url=url,
                    api_key=api_key,
                    model=self.model,
                    prompt=prompt,
                    cwd=cwd,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    compat_profile=self.compat_profile,
                    allow_reasoning_fallback=self.allow_reasoning_fallback,
                    enable_thinking=self.enable_thinking,
                    timeout_sec=self.timeout_sec,
                    seed=self.seed,
                    http2=self.http2,
                    deadline_monotonic=deadline_monotonic,
                    allow_diff_fallback=False,
                    transport_max_retries=self.transport_max_retries,
                    transport_retry_backoff_base_sec=self.transport_retry_backoff_base_sec,
                    transport_retry_backoff_max_sec=self.transport_retry_backoff_max_sec,
                )
                totals = (0, 0, 0, 0, 0, False, True)
                diff_meta = dict(diff_result.meta or {})
                edits_meta = dict(edits_fallback.meta or {})
                totals = _accumulate_llm_totals(totals=totals, meta=diff_meta, default_calls=0)
                totals = _accumulate_llm_totals(totals=totals, meta=edits_meta, default_calls=0)
                merged_meta = dict(edits_meta)
                merged_meta["fallback_from"] = "diff"
                merged_meta["fallback_reason"] = "no_diff_found"
                merged_meta["fallback_meta"] = diff_meta
                if merged_meta.get("error") is None:
                    merged_meta.pop("error", None)
                result = PatchResult(
                    diff=edits_fallback.diff,
                    meta=_meta_with_llm_totals(meta=merged_meta, totals=totals),
                )
            else:
                result = diff_result

        result_meta = dict(result.meta or {})
        result_meta.setdefault("base_url", base_url)
        result_meta.setdefault("model", self.model)
        result_meta.setdefault("temperature", self.temperature)
        result_meta.setdefault("max_tokens", self.max_tokens)
        result_meta.setdefault(
            "compat_profile_requested", normalize_compat_profile(self.compat_profile)
        )
        result_meta.setdefault("timeout_sec", self.timeout_sec)
        result_meta.setdefault("transport_max_retries", self.transport_max_retries)
        result_meta.setdefault(
            "transport_retry_backoff_base_sec", self.transport_retry_backoff_base_sec
        )
        result_meta.setdefault(
            "transport_retry_backoff_max_sec", self.transport_retry_backoff_max_sec
        )
        result_meta.setdefault("max_propose_sec", self.max_propose_sec)
        result_meta.setdefault("response_mode", self.response_mode)
        result_meta.setdefault("api_compat_seed_requested", self.seed)
        result_meta.setdefault(
            "api_compat_seed_acknowledged", False if self.seed is not None else None
        )
        result_meta.setdefault("api_compat_seed_response", None)
        result = PatchResult(diff=result.diff, meta=result_meta)

        if result.diff is not None or not self.force_patch_attempt:
            return result

        skip_reason = _forced_patch_skip_reason(cast(PatcherMeta, result_meta))
        if skip_reason:
            meta = dict(result.meta or {})
            meta["forced_patch_skipped"] = True
            meta["forced_patch_skip_reason"] = _forced_patch_skip_category(skip_reason)
            meta["forced_patch_skip_detail"] = skip_reason
            return PatchResult(diff=None, meta=meta)

        forced = _build_forced_attempt_diff(prompt=prompt, cwd=cwd)
        if forced is None:
            return result

        meta = dict(result.meta or {})
        meta["forced_patch_attempt"] = True
        meta["forced_patch_target"] = forced[0]
        meta["forced_patch_reason"] = "no_model_diff"
        return PatchResult(diff=forced[1], meta=meta)

    def probe_context_window_tokens(self) -> tuple[int | None, PatcherMeta]:
        """Probe the server for the model's context-window size in tokens.

        Queries up to three provider-specific model listing endpoints in order
        of preference, stopping at the first that returns a value for
        :attr:`model`:

        1. **LM Studio API v1** — ``{origin}/api/v1/models``
        2. **LM Studio API v0** — ``{origin}/api/v0/models``
        3. **OpenAI-compatible** — ``{base_url}/models``

        Each response is parsed with a provider-specific extractor that looks
        for ``context_length``, ``max_context_length``, or
        ``max_input_tokens`` fields on the matching model entry.  For LM Studio
        v1, per-instance ``context_length`` overrides the model-level field
        when the model is currently loaded.

        Returns:
            A 2-tuple ``(tokens, meta)`` where:

            - ``tokens`` is the context-window size in tokens as a positive
              ``int``, or ``None`` if no probe succeeded.
            - ``meta`` is a telemetry dict with keys such as
              ``context_window_probe_provider``, ``context_window_source``,
              ``context_window_probe_url``, ``context_window_probe_attempts``
              (per-probe error details), and ``context_window_probe_duration_sec``.
        """
        base_url = _normalize_base_url(self.base_url)
        api_key = self.api_key
        if api_key is None:
            api_key = _default_api_key()
        started = time.perf_counter()
        tokens, meta = _probe_context_window_tokens(
            base_url=base_url,
            model=self.model,
            api_key=api_key,
            timeout_sec=self.timeout_sec,
            http2=self.http2,
        )
        result_meta: PatcherMeta = dict(meta or {})
        result_meta.setdefault("base_url", base_url)
        result_meta.setdefault("model", self.model)
        result_meta["context_window_probe_duration_sec"] = time.perf_counter() - started
        return tokens, result_meta

    async def aprobe_context_window_tokens(self) -> tuple[int | None, PatcherMeta]:
        """Async variant of :meth:`probe_context_window_tokens`.

        Probe the server for the model's context-window size in tokens without
        blocking the running event loop.  Behaviour and return value are
        identical to :meth:`probe_context_window_tokens`; see that method for
        full documentation of the probe order and return fields.

        Returns:
            A 2-tuple ``(tokens, meta)`` — see
            :meth:`probe_context_window_tokens` for field descriptions.
        """
        base_url = _normalize_base_url(self.base_url)
        api_key = self.api_key
        if api_key is None:
            api_key = _default_api_key()
        started = time.perf_counter()
        tokens, meta = await _probe_context_window_tokens_async(
            base_url=base_url,
            model=self.model,
            api_key=api_key,
            timeout_sec=self.timeout_sec,
            http2=self.http2,
        )
        result_meta: PatcherMeta = dict(meta or {})
        result_meta.setdefault("base_url", base_url)
        result_meta.setdefault("model", self.model)
        result_meta["context_window_probe_duration_sec"] = time.perf_counter() - started
        return tokens, result_meta

    def probe_server_ready(self) -> tuple[bool, PatcherMeta]:
        """Probe whether the backend is ready to accept fresh OpenAI calls.

        Probe order:

        1. ``GET <origin>/health``
        2. ``GET <base_url>/models``

        Any ``2xx`` response counts as healthy regardless of body contents.
        """
        base_url = _normalize_base_url(self.base_url)
        api_key = self.api_key
        if api_key is None:
            api_key = _default_api_key()
        started = time.perf_counter()
        ready, meta = _probe_server_ready(
            base_url=base_url,
            api_key=api_key,
            timeout_sec=self.timeout_sec,
            http2=self.http2,
        )
        result_meta: PatcherMeta = dict(meta or {})
        result_meta.setdefault("base_url", base_url)
        result_meta.setdefault("model", self.model)
        result_meta["server_ready_probe_duration_sec"] = time.perf_counter() - started
        return ready, result_meta

    async def aprobe_server_ready(self) -> tuple[bool, PatcherMeta]:
        """Async variant of :meth:`probe_server_ready`."""
        base_url = _normalize_base_url(self.base_url)
        api_key = self.api_key
        if api_key is None:
            api_key = _default_api_key()
        started = time.perf_counter()
        ready, meta = await _probe_server_ready_async(
            base_url=base_url,
            api_key=api_key,
            timeout_sec=self.timeout_sec,
            http2=self.http2,
        )
        result_meta: PatcherMeta = dict(meta or {})
        result_meta.setdefault("base_url", base_url)
        result_meta.setdefault("model", self.model)
        result_meta["server_ready_probe_duration_sec"] = time.perf_counter() - started
        return ready, result_meta


def _http_get_json(
    *,
    url: str,
    api_key: str,
    timeout_sec: float,
    http2: bool,
) -> tuple[dict[str, Any] | list[Any] | None, PatcherMeta]:
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    fallback_used = False
    http2_effective = bool(http2)
    try:
        with httpx.Client(timeout=timeout_sec, http2=http2_effective, trust_env=True) as client:
            response = client.get(url, headers=headers)
            status = int(response.status_code)
            raw_text = response.text
            http_version = str(getattr(response, "http_version", "") or "")
    except ValueError as exc:
        return None, {"error": "bad_request_url", "exc": f"{type(exc).__name__}: {exc}"}
    except Exception as exc:
        if http2_effective and _is_http2_protocol_error(exc):
            fallback_used = True
            http2_effective = False
            try:
                with httpx.Client(timeout=timeout_sec, http2=False, trust_env=True) as client:
                    response = client.get(url, headers=headers)
                    status = int(response.status_code)
                    raw_text = response.text
                    http_version = str(getattr(response, "http_version", "") or "")
            except ValueError as inner_exc:
                return None, {
                    "error": "bad_request_url",
                    "exc": f"{type(inner_exc).__name__}: {inner_exc}",
                }
            except Exception as inner_exc:
                return None, {
                    "error": "request_failed",
                    "exc": f"{type(inner_exc).__name__}: {inner_exc}",
                    "transport_http2_requested": bool(http2),
                    "transport_http2_effective": False,
                    "transport_http2_fallback_used": True,
                }
        else:
            return None, {"error": "request_failed", "exc": f"{type(exc).__name__}: {exc}"}

    response_meta: PatcherMeta = {
        "transport_impl": "httpx",
        "transport_http2_requested": bool(http2),
        "transport_http2_effective": bool(http2_effective),
        "transport_http2_fallback_used": bool(fallback_used),
        "transport_http_version": http_version,
    }
    if status >= 400:
        return None, {
            "error": "http_error",
            "status": status,
            "body": raw_text[:2000],
            **response_meta,
        }
    try:
        data = json.loads(raw_text)
    except Exception:
        return None, {"error": "bad_json_response", "body": raw_text[:2000], **response_meta}
    if isinstance(data, (dict, list)):
        return data, response_meta
    return None, {"error": "bad_json_response", "body": raw_text[:2000], **response_meta}


def _http_probe_ready(
    *,
    url: str,
    api_key: str,
    timeout_sec: float,
    http2: bool,
) -> tuple[bool, PatcherMeta]:
    headers = {
        "Accept": "application/json, text/plain;q=0.9, */*;q=0.1",
        "Authorization": f"Bearer {api_key}",
    }
    fallback_used = False
    http2_effective = bool(http2)

    def _send_request(*, use_http2: bool) -> tuple[int, str, str]:
        if _should_use_proxy_transport(url):
            with httpx.Client(timeout=timeout_sec, http2=use_http2, trust_env=True) as client:
                response = client.get(url, headers=headers)
                return (
                    int(response.status_code),
                    response.text,
                    str(getattr(response, "http_version", "") or ""),
                )
        client = _get_thread_http_client(url=url, http2=use_http2)
        status, raw = client.get_text(
            target=_chat_http_target(url),
            headers=headers,
            timeout_sec=timeout_sec,
        )
        return status, raw, str(getattr(client, "last_http_version", "") or "")

    try:
        status, raw_text, http_version = _send_request(use_http2=http2_effective)
    except ValueError as exc:
        return None, {"error": "bad_request_url", "exc": f"{type(exc).__name__}: {exc}"}
    except Exception as exc:
        if http2_effective and _is_http2_protocol_error(exc):
            fallback_used = True
            http2_effective = False
            try:
                status, raw_text, http_version = _send_request(use_http2=False)
            except ValueError as inner_exc:
                return None, {
                    "error": "bad_request_url",
                    "exc": f"{type(inner_exc).__name__}: {inner_exc}",
                }
            except Exception as inner_exc:
                return None, {
                    "error": "request_failed",
                    "exc": f"{type(inner_exc).__name__}: {inner_exc}",
                    "transport_http2_requested": bool(http2),
                    "transport_http2_effective": False,
                    "transport_http2_fallback_used": True,
                }
        else:
            return None, {"error": "request_failed", "exc": f"{type(exc).__name__}: {exc}"}

    response_meta: PatcherMeta = {
        "transport_impl": "httpx",
        "transport_http2_requested": bool(http2),
        "transport_http2_effective": bool(http2_effective),
        "transport_http2_fallback_used": bool(fallback_used),
        "transport_http_version": http_version,
        "status": int(status),
    }
    if 200 <= status < 300:
        return True, response_meta
    return False, {
        "error": "http_error",
        "status": int(status),
        "body": raw_text[:2000],
        **response_meta,
    }


async def _http_get_json_async(
    *,
    url: str,
    api_key: str,
    timeout_sec: float,
    http2: bool,
) -> tuple[dict[str, Any] | list[Any] | None, PatcherMeta]:
    headers = {
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    fallback_used = False
    http2_effective = bool(http2)
    try:
        async with httpx.AsyncClient(
            timeout=timeout_sec, http2=http2_effective, trust_env=True
        ) as client:
            response = await client.get(url, headers=headers)
            status = int(response.status_code)
            raw_text = response.text
            http_version = str(getattr(response, "http_version", "") or "")
    except Exception as exc:
        if http2_effective and _is_http2_protocol_error(exc):
            fallback_used = True
            http2_effective = False
            try:
                async with httpx.AsyncClient(
                    timeout=timeout_sec, http2=False, trust_env=True
                ) as client:
                    response = await client.get(url, headers=headers)
                    status = int(response.status_code)
                    raw_text = response.text
                    http_version = str(getattr(response, "http_version", "") or "")
            except ValueError as inner_exc:
                return None, {
                    "error": "bad_request_url",
                    "exc": f"{type(inner_exc).__name__}: {inner_exc}",
                }
            except Exception as inner_exc:
                return None, {
                    "error": "request_failed",
                    "exc": f"{type(inner_exc).__name__}: {inner_exc}",
                    "transport_http2_requested": bool(http2),
                    "transport_http2_effective": False,
                    "transport_http2_fallback_used": True,
                }
        else:
            return None, {"error": "request_failed", "exc": f"{type(exc).__name__}: {exc}"}

    response_meta: PatcherMeta = {
        "transport_impl": "httpx",
        "transport_http2_requested": bool(http2),
        "transport_http2_effective": bool(http2_effective),
        "transport_http2_fallback_used": bool(fallback_used),
        "transport_http_version": http_version,
    }
    if status >= 400:
        return None, {
            "error": "http_error",
            "status": status,
            "body": raw_text[:2000],
            **response_meta,
        }
    try:
        data = json.loads(raw_text)
    except Exception:
        return None, {"error": "bad_json_response", "body": raw_text[:2000], **response_meta}
    if isinstance(data, (dict, list)):
        return data, response_meta
    return None, {"error": "bad_json_response", "body": raw_text[:2000], **response_meta}


async def _http_probe_ready_async(
    *,
    url: str,
    api_key: str,
    timeout_sec: float,
    http2: bool,
) -> tuple[bool, PatcherMeta]:
    headers = {
        "Accept": "application/json, text/plain;q=0.9, */*;q=0.1",
        "Authorization": f"Bearer {api_key}",
    }
    fallback_used = False
    http2_effective = bool(http2)

    async def _send_request(*, use_http2: bool) -> tuple[int, str, str]:
        if _should_use_proxy_transport(url):
            async with httpx.AsyncClient(
                timeout=timeout_sec,
                http2=use_http2,
                trust_env=True,
            ) as client:
                response = await client.get(url, headers=headers)
                return (
                    int(response.status_code),
                    response.text,
                    str(getattr(response, "http_version", "") or ""),
                )
        client = _get_async_http_client(url=url, http2=use_http2)
        status, raw = await client.get_text(
            target=_chat_http_target(url),
            headers=headers,
            timeout_sec=timeout_sec,
        )
        return status, raw, str(getattr(client, "last_http_version", "") or "")

    try:
        status, raw_text, http_version = await _send_request(use_http2=http2_effective)
    except ValueError as exc:
        return None, {"error": "bad_request_url", "exc": f"{type(exc).__name__}: {exc}"}
    except Exception as exc:
        if http2_effective and _is_http2_protocol_error(exc):
            fallback_used = True
            http2_effective = False
            try:
                status, raw_text, http_version = await _send_request(use_http2=False)
            except ValueError as inner_exc:
                return None, {
                    "error": "bad_request_url",
                    "exc": f"{type(inner_exc).__name__}: {inner_exc}",
                }
            except Exception as inner_exc:
                return None, {
                    "error": "request_failed",
                    "exc": f"{type(inner_exc).__name__}: {inner_exc}",
                    "transport_http2_requested": bool(http2),
                    "transport_http2_effective": False,
                    "transport_http2_fallback_used": True,
                }
        else:
            return None, {"error": "request_failed", "exc": f"{type(exc).__name__}: {exc}"}

    response_meta: PatcherMeta = {
        "transport_impl": "httpx",
        "transport_http2_requested": bool(http2),
        "transport_http2_effective": bool(http2_effective),
        "transport_http2_fallback_used": bool(fallback_used),
        "transport_http_version": http_version,
        "status": int(status),
    }
    if 200 <= status < 300:
        return True, response_meta
    return False, {
        "error": "http_error",
        "status": int(status),
        "body": raw_text[:2000],
        **response_meta,
    }


def _coerce_positive_int(value: Any) -> int | None:
    try:
        out = int(value)
    except (TypeError, ValueError):
        return None
    return out if out > 0 else None


def _extract_model_entries(payload: dict[str, Any] | list[Any]) -> list[dict[str, Any]]:
    entries_raw: Any = payload
    if isinstance(payload, dict):
        for key in ("models", "data"):
            value = payload.get(key)
            if isinstance(value, list):
                entries_raw = value
                break
        else:
            entries_raw = []
    if not isinstance(entries_raw, list):
        return []
    return [item for item in entries_raw if isinstance(item, dict)]


def _entry_matches_model(entry: dict[str, Any], *, model: str) -> bool:
    target = model.strip()
    if not target:
        return False
    for key in ("id", "key", "model", "name"):
        value = entry.get(key)
        if isinstance(value, str) and value.strip() == target:
            return True
    loaded_instances = entry.get("loaded_instances")
    if isinstance(loaded_instances, list):
        for instance in loaded_instances:
            if not isinstance(instance, dict):
                continue
            value = instance.get("id")
            if isinstance(value, str) and value.strip() == target:
                return True
    return False


def _extract_lmstudio_instance_context_length(instance: dict[str, Any]) -> int | None:
    direct = _coerce_positive_int(instance.get("context_length"))
    if direct is not None:
        return direct
    config = instance.get("config")
    if isinstance(config, dict):
        return _coerce_positive_int(config.get("context_length"))
    return None


def _extract_lmstudio_v1_context_window(
    *,
    payload: dict[str, Any] | list[Any],
    model: str,
) -> tuple[int | None, dict[str, Any]]:
    for entry in _extract_model_entries(payload):
        if not _entry_matches_model(entry, model=model):
            continue
        loaded_instances = entry.get("loaded_instances")
        if isinstance(loaded_instances, list):
            for instance in loaded_instances:
                if not isinstance(instance, dict):
                    continue
                context_length = _extract_lmstudio_instance_context_length(instance)
                if context_length is not None:
                    return context_length, {
                        "context_window_source": "loaded_instance.context_length",
                        "context_window_probe_instance_id": str(instance.get("id") or ""),
                    }
        for field in ("max_context_length", "context_length", "max_input_tokens"):
            context_length = _coerce_positive_int(entry.get(field))
            if context_length is not None:
                return context_length, {"context_window_source": field}
    return None, {}


def _extract_lmstudio_v0_context_window(
    *,
    payload: dict[str, Any] | list[Any],
    model: str,
) -> tuple[int | None, dict[str, Any]]:
    for entry in _extract_model_entries(payload):
        if not _entry_matches_model(entry, model=model):
            continue
        for field in ("context_length", "max_context_length", "max_input_tokens"):
            context_length = _coerce_positive_int(entry.get(field))
            if context_length is not None:
                return context_length, {"context_window_source": field}
    return None, {}


def _extract_openai_models_context_window(
    *,
    payload: dict[str, Any] | list[Any],
    model: str,
) -> tuple[int | None, dict[str, Any]]:
    for entry in _extract_model_entries(payload):
        if not _entry_matches_model(entry, model=model):
            continue
        for field in ("context_length", "max_context_length", "max_input_tokens"):
            context_length = _coerce_positive_int(entry.get(field))
            if context_length is not None:
                return context_length, {"context_window_source": field}
    return None, {}


def _probe_context_window_tokens(
    *,
    base_url: str,
    model: str,
    api_key: str,
    timeout_sec: float,
    http2: bool,
) -> tuple[int | None, PatcherMeta]:
    origin = _origin_from_base_url(base_url).rstrip("/")
    normalized_base = base_url.rstrip("/")
    probes: list[tuple[str, str, Any]] = [
        ("lmstudio_api_v1", f"{origin}/api/v1/models", _extract_lmstudio_v1_context_window),
        ("lmstudio_api_v0", f"{origin}/api/v0/models", _extract_lmstudio_v0_context_window),
        ("openai_models", f"{normalized_base}/models", _extract_openai_models_context_window),
    ]
    seen_urls: set[str] = set()
    attempts: list[dict[str, Any]] = []
    for provider, probe_url, parser in probes:
        if probe_url in seen_urls:
            continue
        seen_urls.add(probe_url)
        payload, meta = _http_get_json(
            url=probe_url,
            api_key=api_key,
            timeout_sec=timeout_sec,
            http2=http2,
        )
        if payload is None:
            attempts.append(
                {
                    "provider": provider,
                    "url": probe_url,
                    "error": str(meta.get("error") or "request_failed"),
                    "status": meta.get("status"),
                }
            )
            continue
        context_tokens, parsed_meta = parser(payload=payload, model=model)
        if context_tokens is None:
            attempts.append(
                {
                    "provider": provider,
                    "url": probe_url,
                    "error": "model_context_not_found",
                }
            )
            continue
        result_meta: PatcherMeta = {
            "context_window_probe_provider": provider,
            "context_window_probe_url": probe_url,
            "context_window_source": str(parsed_meta.get("context_window_source") or "auto_probe"),
            "context_window_probe_attempts": attempts,
        }
        result_meta.update(parsed_meta)
        return context_tokens, result_meta

    if attempts:
        summary = ", ".join(f"{item['provider']}:{item['error']}" for item in attempts[:3])
    else:
        summary = "no_probe_attempts"
    return None, {
        "error": "context_window_probe_failed",
        "context_window_probe_provider": "",
        "context_window_source": "auto_probe_failed",
        "context_window_probe_error": summary,
        "context_window_probe_attempts": attempts,
    }


async def _probe_context_window_tokens_async(
    *,
    base_url: str,
    model: str,
    api_key: str,
    timeout_sec: float,
    http2: bool,
) -> tuple[int | None, PatcherMeta]:
    origin = _origin_from_base_url(base_url).rstrip("/")
    normalized_base = base_url.rstrip("/")
    probes: list[tuple[str, str, Any]] = [
        ("lmstudio_api_v1", f"{origin}/api/v1/models", _extract_lmstudio_v1_context_window),
        ("lmstudio_api_v0", f"{origin}/api/v0/models", _extract_lmstudio_v0_context_window),
        ("openai_models", f"{normalized_base}/models", _extract_openai_models_context_window),
    ]
    seen_urls: set[str] = set()
    attempts: list[dict[str, Any]] = []
    for provider, probe_url, parser in probes:
        if probe_url in seen_urls:
            continue
        seen_urls.add(probe_url)
        payload, meta = await _http_get_json_async(
            url=probe_url,
            api_key=api_key,
            timeout_sec=timeout_sec,
            http2=http2,
        )
        if payload is None:
            attempts.append(
                {
                    "provider": provider,
                    "url": probe_url,
                    "error": str(meta.get("error") or "request_failed"),
                    "status": meta.get("status"),
                }
            )
            continue
        context_tokens, parsed_meta = parser(payload=payload, model=model)
        if context_tokens is None:
            attempts.append(
                {
                    "provider": provider,
                    "url": probe_url,
                    "error": "model_context_not_found",
                }
            )
            continue
        result_meta: PatcherMeta = {
            "context_window_probe_provider": provider,
            "context_window_probe_url": probe_url,
            "context_window_source": str(parsed_meta.get("context_window_source") or "auto_probe"),
            "context_window_probe_attempts": attempts,
        }
        result_meta.update(parsed_meta)
        return context_tokens, result_meta

    if attempts:
        summary = ", ".join(f"{item['provider']}:{item['error']}" for item in attempts[:3])
    else:
        summary = "no_probe_attempts"
    return None, {
        "error": "context_window_probe_failed",
        "context_window_probe_provider": "",
        "context_window_source": "auto_probe_failed",
        "context_window_probe_error": summary,
        "context_window_probe_attempts": attempts,
    }


def _probe_server_ready(
    *,
    base_url: str,
    api_key: str,
    timeout_sec: float,
    http2: bool,
) -> tuple[bool, PatcherMeta]:
    origin = _origin_from_base_url(base_url).rstrip("/")
    normalized_base = base_url.rstrip("/")
    probes = [
        ("health", f"{origin}/health"),
        ("openai_models", f"{normalized_base}/models"),
    ]
    seen_urls: set[str] = set()
    attempts: list[dict[str, Any]] = []
    for provider, probe_url in probes:
        if probe_url in seen_urls:
            continue
        seen_urls.add(probe_url)
        ready, meta = _http_probe_ready(
            url=probe_url,
            api_key=api_key,
            timeout_sec=timeout_sec,
            http2=http2,
        )
        if ready:
            result_meta: PatcherMeta = {
                "server_ready_probe_provider": provider,
                "server_ready_probe_url": probe_url,
                "server_ready_probe_attempts": attempts,
                "server_ready": True,
            }
            result_meta.update(meta)
            return True, result_meta
        attempts.append(
            {
                "provider": provider,
                "url": probe_url,
                "error": str(meta.get("error") or "request_failed"),
                "status": meta.get("status"),
            }
        )
    summary = ", ".join(f"{item['provider']}:{item['error']}" for item in attempts[:3])
    return False, {
        "error": "server_ready_probe_failed",
        "server_ready": False,
        "server_ready_probe_provider": "",
        "server_ready_probe_error": summary or "no_probe_attempts",
        "server_ready_probe_attempts": attempts,
    }


async def _probe_server_ready_async(
    *,
    base_url: str,
    api_key: str,
    timeout_sec: float,
    http2: bool,
) -> tuple[bool, PatcherMeta]:
    origin = _origin_from_base_url(base_url).rstrip("/")
    normalized_base = base_url.rstrip("/")
    probes = [
        ("health", f"{origin}/health"),
        ("openai_models", f"{normalized_base}/models"),
    ]
    seen_urls: set[str] = set()
    attempts: list[dict[str, Any]] = []
    for provider, probe_url in probes:
        if probe_url in seen_urls:
            continue
        seen_urls.add(probe_url)
        ready, meta = await _http_probe_ready_async(
            url=probe_url,
            api_key=api_key,
            timeout_sec=timeout_sec,
            http2=http2,
        )
        if ready:
            result_meta: PatcherMeta = {
                "server_ready_probe_provider": provider,
                "server_ready_probe_url": probe_url,
                "server_ready_probe_attempts": attempts,
                "server_ready": True,
            }
            result_meta.update(meta)
            return True, result_meta
        attempts.append(
            {
                "provider": provider,
                "url": probe_url,
                "error": str(meta.get("error") or "request_failed"),
                "status": meta.get("status"),
            }
        )
    summary = ", ".join(f"{item['provider']}:{item['error']}" for item in attempts[:3])
    return False, {
        "error": "server_ready_probe_failed",
        "server_ready": False,
        "server_ready_probe_provider": "",
        "server_ready_probe_error": summary or "no_probe_attempts",
        "server_ready_probe_attempts": attempts,
    }


def _request_context_via_json(
    *,
    url: str,
    api_key: str,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    compat_profile: str = "auto",
    allow_reasoning_fallback: bool = False,
    enable_thinking: bool = False,
    timeout_sec: float,
    seed: int | None,
    http2: bool,
    strict_json: bool = False,
    transport_max_retries: int = _DEFAULT_CHAT_COMPLETION_TRANSPORT_MAX_RETRIES,
    transport_retry_backoff_base_sec: float = _DEFAULT_CHAT_COMPLETION_RETRY_BACKOFF_BASE_SEC,
    transport_retry_backoff_max_sec: float = _DEFAULT_CHAT_COMPLETION_RETRY_BACKOFF_MAX_SEC,
) -> tuple[dict[str, Any] | None, PatcherMeta]:
    messages = [
        {
            "role": "system",
            "content": _request_context_system_prompt(strict_json=strict_json),
        },
        {"role": "user", "content": prompt},
    ]

    totals = (0, 0, 0, 0, 0, False, True)
    response_format_fallback_used = False
    raw, meta = _chat_completion(
        url=url,
        api_key=api_key,
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        compat_profile=compat_profile,
        allow_reasoning_fallback=allow_reasoning_fallback,
        enable_thinking=enable_thinking,
        timeout_sec=timeout_sec,
        seed=seed,
        http2=http2,
        extra_payload={"response_format": {"type": "json_object"}},
        transport_max_retries=transport_max_retries,
        transport_retry_backoff_base_sec=transport_retry_backoff_base_sec,
        transport_retry_backoff_max_sec=transport_retry_backoff_max_sec,
    )
    totals = _accumulate_llm_totals(totals=totals, meta=meta, default_calls=1)
    if raw is None:
        err = str(meta.get("error", ""))
        body = str(meta.get("body", ""))
        if err == "http_error" and "response_format" in body:
            response_format_fallback_used = True
            raw, meta = _chat_completion(
                url=url,
                api_key=api_key,
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                compat_profile=compat_profile,
                allow_reasoning_fallback=allow_reasoning_fallback,
                enable_thinking=enable_thinking,
                timeout_sec=timeout_sec,
                seed=seed,
                http2=http2,
                transport_max_retries=transport_max_retries,
                transport_retry_backoff_base_sec=transport_retry_backoff_base_sec,
                transport_retry_backoff_max_sec=transport_retry_backoff_max_sec,
            )
            totals = _accumulate_llm_totals(totals=totals, meta=meta, default_calls=1)
    meta = _meta_with_llm_totals(meta=cast(PatcherMeta, dict(meta)), totals=totals)
    meta["request_context_response_format_fallback_used"] = response_format_fallback_used
    meta["request_context_strict_json"] = bool(strict_json)
    return raw, meta


async def _request_context_via_json_async(
    *,
    url: str,
    api_key: str,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    compat_profile: str = "auto",
    allow_reasoning_fallback: bool = False,
    enable_thinking: bool = False,
    timeout_sec: float,
    seed: int | None,
    http2: bool,
    strict_json: bool = False,
    transport_max_retries: int = _DEFAULT_CHAT_COMPLETION_TRANSPORT_MAX_RETRIES,
    transport_retry_backoff_base_sec: float = _DEFAULT_CHAT_COMPLETION_RETRY_BACKOFF_BASE_SEC,
    transport_retry_backoff_max_sec: float = _DEFAULT_CHAT_COMPLETION_RETRY_BACKOFF_MAX_SEC,
) -> tuple[dict[str, Any] | None, PatcherMeta]:
    messages = [
        {
            "role": "system",
            "content": _request_context_system_prompt(strict_json=strict_json),
        },
        {"role": "user", "content": prompt},
    ]

    totals = (0, 0, 0, 0, 0, False, True)
    response_format_fallback_used = False
    raw, meta = await _chat_completion_async(
        url=url,
        api_key=api_key,
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        compat_profile=compat_profile,
        allow_reasoning_fallback=allow_reasoning_fallback,
        enable_thinking=enable_thinking,
        timeout_sec=timeout_sec,
        seed=seed,
        http2=http2,
        extra_payload={"response_format": {"type": "json_object"}},
        transport_max_retries=transport_max_retries,
        transport_retry_backoff_base_sec=transport_retry_backoff_base_sec,
        transport_retry_backoff_max_sec=transport_retry_backoff_max_sec,
    )
    totals = _accumulate_llm_totals(totals=totals, meta=meta, default_calls=1)
    if raw is None:
        err = str(meta.get("error", ""))
        body = str(meta.get("body", ""))
        if err == "http_error" and "response_format" in body:
            response_format_fallback_used = True
            raw, meta = await _chat_completion_async(
                url=url,
                api_key=api_key,
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                compat_profile=compat_profile,
                allow_reasoning_fallback=allow_reasoning_fallback,
                enable_thinking=enable_thinking,
                timeout_sec=timeout_sec,
                seed=seed,
                http2=http2,
                transport_max_retries=transport_max_retries,
                transport_retry_backoff_base_sec=transport_retry_backoff_base_sec,
                transport_retry_backoff_max_sec=transport_retry_backoff_max_sec,
            )
            totals = _accumulate_llm_totals(totals=totals, meta=meta, default_calls=1)
    meta = _meta_with_llm_totals(meta=cast(PatcherMeta, dict(meta)), totals=totals)
    meta["request_context_response_format_fallback_used"] = response_format_fallback_used
    meta["request_context_strict_json"] = bool(strict_json)
    return raw, meta


def _compact_context_via_json(
    *,
    url: str,
    api_key: str,
    model: str,
    prompt: str,
    context_text: str,
    target_chars: int,
    temperature: float,
    max_tokens: int,
    compat_profile: str = "auto",
    allow_reasoning_fallback: bool = False,
    enable_thinking: bool = False,
    timeout_sec: float,
    seed: int | None,
    http2: bool,
    transport_max_retries: int = _DEFAULT_CHAT_COMPLETION_TRANSPORT_MAX_RETRIES,
    transport_retry_backoff_base_sec: float = _DEFAULT_CHAT_COMPLETION_RETRY_BACKOFF_BASE_SEC,
    transport_retry_backoff_max_sec: float = _DEFAULT_CHAT_COMPLETION_RETRY_BACKOFF_MAX_SEC,
) -> tuple[str | None, str | None, PatcherMeta]:
    target_chars_safe = max(1, int(target_chars))
    messages = [
        {
            "role": "system",
            "content": (
                "Return only JSON for context compaction. "
                "Use this schema exactly: "
                '{"compacted_context":"...",'
                '"reduction_rationale":"..."} '
                f"The compacted_context must be <= {target_chars_safe} characters. "
                "Preserve critical traceback facts, file paths, assertions, and state clues. "
                "No prose and no markdown."
            ),
        },
        {
            "role": "user",
            "content": (
                "Task:\n"
                + prompt.rstrip()
                + "\n\n"
                + f"Target characters: {target_chars_safe}\n"
                + "Context to compact:\n"
                + context_text.rstrip()
                + "\n"
            ),
        },
    ]

    totals = (0, 0, 0, 0, 0, False, True)
    raw, meta = _chat_completion(
        url=url,
        api_key=api_key,
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        compat_profile=compat_profile,
        allow_reasoning_fallback=allow_reasoning_fallback,
        enable_thinking=enable_thinking,
        timeout_sec=timeout_sec,
        seed=seed,
        http2=http2,
        extra_payload={"response_format": {"type": "json_object"}},
        transport_max_retries=transport_max_retries,
        transport_retry_backoff_base_sec=transport_retry_backoff_base_sec,
        transport_retry_backoff_max_sec=transport_retry_backoff_max_sec,
    )
    totals = _accumulate_llm_totals(totals=totals, meta=meta, default_calls=1)
    if raw is None:
        err = str(meta.get("error", ""))
        body = str(meta.get("body", ""))
        if err == "http_error" and "response_format" in body:
            raw, meta = _chat_completion(
                url=url,
                api_key=api_key,
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                compat_profile=compat_profile,
                allow_reasoning_fallback=allow_reasoning_fallback,
                enable_thinking=enable_thinking,
                timeout_sec=timeout_sec,
                seed=seed,
                http2=http2,
                transport_max_retries=transport_max_retries,
                transport_retry_backoff_base_sec=transport_retry_backoff_base_sec,
                transport_retry_backoff_max_sec=transport_retry_backoff_max_sec,
            )
            totals = _accumulate_llm_totals(totals=totals, meta=meta, default_calls=1)

    meta = _meta_with_llm_totals(meta=cast(PatcherMeta, dict(meta)), totals=totals)
    compacted_context, raw_response = _normalize_context_compaction_payload(raw)
    return compacted_context, raw_response, meta


async def _compact_context_via_json_async(
    *,
    url: str,
    api_key: str,
    model: str,
    prompt: str,
    context_text: str,
    target_chars: int,
    temperature: float,
    max_tokens: int,
    compat_profile: str = "auto",
    allow_reasoning_fallback: bool = False,
    enable_thinking: bool = False,
    timeout_sec: float,
    seed: int | None,
    http2: bool,
    transport_max_retries: int = _DEFAULT_CHAT_COMPLETION_TRANSPORT_MAX_RETRIES,
    transport_retry_backoff_base_sec: float = _DEFAULT_CHAT_COMPLETION_RETRY_BACKOFF_BASE_SEC,
    transport_retry_backoff_max_sec: float = _DEFAULT_CHAT_COMPLETION_RETRY_BACKOFF_MAX_SEC,
) -> tuple[str | None, str | None, PatcherMeta]:
    target_chars_safe = max(1, int(target_chars))
    messages = [
        {
            "role": "system",
            "content": (
                "Return only JSON for context compaction. "
                "Use this schema exactly: "
                '{"compacted_context":"...",'
                '"reduction_rationale":"..."} '
                f"The compacted_context must be <= {target_chars_safe} characters. "
                "Preserve critical traceback facts, file paths, assertions, and state clues. "
                "No prose and no markdown."
            ),
        },
        {
            "role": "user",
            "content": (
                "Task:\n"
                + prompt.rstrip()
                + "\n\n"
                + f"Target characters: {target_chars_safe}\n"
                + "Context to compact:\n"
                + context_text.rstrip()
                + "\n"
            ),
        },
    ]

    totals = (0, 0, 0, 0, 0, False, True)
    raw, meta = await _chat_completion_async(
        url=url,
        api_key=api_key,
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        compat_profile=compat_profile,
        allow_reasoning_fallback=allow_reasoning_fallback,
        enable_thinking=enable_thinking,
        timeout_sec=timeout_sec,
        seed=seed,
        http2=http2,
        extra_payload={"response_format": {"type": "json_object"}},
        transport_max_retries=transport_max_retries,
        transport_retry_backoff_base_sec=transport_retry_backoff_base_sec,
        transport_retry_backoff_max_sec=transport_retry_backoff_max_sec,
    )
    totals = _accumulate_llm_totals(totals=totals, meta=meta, default_calls=1)
    if raw is None:
        err = str(meta.get("error", ""))
        body = str(meta.get("body", ""))
        if err == "http_error" and "response_format" in body:
            raw, meta = await _chat_completion_async(
                url=url,
                api_key=api_key,
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                compat_profile=compat_profile,
                allow_reasoning_fallback=allow_reasoning_fallback,
                enable_thinking=enable_thinking,
                timeout_sec=timeout_sec,
                seed=seed,
                http2=http2,
                transport_max_retries=transport_max_retries,
                transport_retry_backoff_base_sec=transport_retry_backoff_base_sec,
                transport_retry_backoff_max_sec=transport_retry_backoff_max_sec,
            )
            totals = _accumulate_llm_totals(totals=totals, meta=meta, default_calls=1)

    meta = _meta_with_llm_totals(meta=cast(PatcherMeta, dict(meta)), totals=totals)
    compacted_context, raw_response = _normalize_context_compaction_payload(raw)
    return compacted_context, raw_response, meta


def _normalize_context_compaction_payload(
    response: dict[str, Any] | None,
) -> tuple[str | None, str | None]:
    if response is None:
        return None, None

    content = _get_first_choice_content(response)
    if content is None:
        return None, _safe_trunc(response)
    payload = _extract_json_object(content)
    if payload is None:
        return None, content

    candidates = (
        payload.get("compacted_context"),
        payload.get("summary"),
        payload.get("context"),
        payload.get("text"),
    )
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip(), content
    return None, content


def _normalize_context_request_payload(
    response: dict[str, Any] | None,
) -> tuple[ContextRequestPayload | None, str | None]:
    if response is None:
        return None, None

    content = _get_first_choice_content(response)
    if content is None:
        return None, _safe_trunc(response)
    raw_payload = _extract_json_object(content)
    if raw_payload is None:
        return None, content
    normalized = normalize_stage_a_request_payload(raw_payload)
    if normalized is None:
        return None, content
    return normalized, content


def _propose_patch_via_diff(
    *,
    url: str,
    api_key: str,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    compat_profile: str = "auto",
    allow_reasoning_fallback: bool = False,
    enable_thinking: bool = False,
    timeout_sec: float,
    seed: int | None,
    http2: bool,
    deadline_monotonic: float | None = None,
    system_prompt_override: str | None = None,
    transport_max_retries: int = _DEFAULT_CHAT_COMPLETION_TRANSPORT_MAX_RETRIES,
    transport_retry_backoff_base_sec: float = _DEFAULT_CHAT_COMPLETION_RETRY_BACKOFF_BASE_SEC,
    transport_retry_backoff_max_sec: float = _DEFAULT_CHAT_COMPLETION_RETRY_BACKOFF_MAX_SEC,
) -> PatchResult:
    base_messages = [
        {
            "role": "system",
            "content": system_prompt_override
            or (
                "You fix failing tests by returning a single unified diff. "
                "Return ONLY the diff; no explanations. Do not use Markdown fences."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    # Try twice: second attempt strengthens formatting instructions without assistant prefill.
    attempts: list[list[dict[str, str]]] = [
        base_messages,
        [
            {
                "role": "system",
                "content": system_prompt_override
                or (
                    "You fix failing tests by returning a single unified diff. "
                    "Return ONLY the diff; no explanations. Do not use Markdown fences. "
                    "Start your response with 'diff --git a/'."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    ]

    last_meta: PatcherMeta | None = None
    totals = (0, 0, 0, 0, 0, False, True)
    for idx, messages in enumerate(attempts, start=1):
        raw, meta = _chat_completion(
            url=url,
            api_key=api_key,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            compat_profile=compat_profile,
            allow_reasoning_fallback=allow_reasoning_fallback,
            enable_thinking=enable_thinking,
            timeout_sec=timeout_sec,
            seed=seed,
            http2=http2,
            deadline_monotonic=deadline_monotonic,
            transport_max_retries=transport_max_retries,
            transport_retry_backoff_base_sec=transport_retry_backoff_base_sec,
            transport_retry_backoff_max_sec=transport_retry_backoff_max_sec,
        )
        totals = _accumulate_llm_totals(totals=totals, meta=meta, default_calls=1)
        last_meta = _meta_with_llm_totals(meta={"attempt": idx, **meta}, totals=totals)
        if raw is None:
            if _is_openai_compat_reasoning_only(meta):
                break
            if _is_request_budget_exceeded(meta):
                break
            continue

        content = _get_first_choice_content(raw)
        if content is None:
            last_meta["error"] = "no_choice_content"
            last_meta["response"] = _safe_trunc(raw)
            continue

        diff = _extract_unified_diff(content)
        if diff is None:
            last_meta["error"] = "no_diff_found"
            last_meta["raw_response"] = content[:2000]
            continue

        fixed = _fix_hunk_counts(diff)
        return PatchResult(diff=fixed, meta=last_meta)

    if last_meta is None:
        last_meta = _meta_with_llm_totals(meta={"error": "unknown"}, totals=totals)
    return PatchResult(diff=None, meta=last_meta)


def _propose_patch_via_edits(
    *,
    url: str,
    api_key: str,
    model: str,
    prompt: str,
    cwd: Path,
    temperature: float,
    max_tokens: int,
    compat_profile: str = "auto",
    allow_reasoning_fallback: bool = False,
    enable_thinking: bool = False,
    timeout_sec: float,
    seed: int | None,
    http2: bool,
    deadline_monotonic: float | None = None,
    allow_diff_fallback: bool = True,
    system_prompt_override: str | None = None,
    transport_max_retries: int = _DEFAULT_CHAT_COMPLETION_TRANSPORT_MAX_RETRIES,
    transport_retry_backoff_base_sec: float = _DEFAULT_CHAT_COMPLETION_RETRY_BACKOFF_BASE_SEC,
    transport_retry_backoff_max_sec: float = _DEFAULT_CHAT_COMPLETION_RETRY_BACKOFF_MAX_SEC,
) -> PatchResult:
    messages = [
        {
            "role": "system",
            "content": system_prompt_override
            or (
                "Return only JSON with this schema: "
                '{"edits":[{"path":"relative/path.py","find":"exact old text","replace":"new text","count":1}]}. '
                "No prose. No markdown. "
                "Use only exact existing paths from the prompt's available file list. "
                "Never use placeholder paths like relative/path.py."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    totals = (0, 0, 0, 0, 0, False, True)

    def _with_totals(meta: PatcherMeta) -> PatcherMeta:
        return _meta_with_llm_totals(meta=meta, totals=totals)

    payload: dict[str, Any] | None = None
    meta: PatcherMeta = {}
    stages = [
        (
            "json_schema",
            {
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "patch_edits",
                        "schema": PATCH_EDITS_JSON_SCHEMA,
                    },
                }
            },
        ),
        ("json_object", {"response_format": {"type": "json_object"}}),
        ("plain", None),
    ]
    stage_attempts: list[PatcherMeta] = []
    fallback_reason: str | None = None
    fallback_raw_response: str | None = None
    fallback_payload: dict[str, Any] | None = None

    def _fallback_to_diff(
        *,
        reason: str,
        extra_error_meta: PatcherMeta | None = None,
    ) -> PatchResult:
        nonlocal totals
        if not allow_diff_fallback:
            error_meta: PatcherMeta = {"error": reason, **meta}
            if extra_error_meta:
                error_meta.update(extra_error_meta)
            return PatchResult(diff=None, meta=_with_totals(error_meta))

        diff_fallback = _propose_patch_via_diff(
            url=url,
            api_key=api_key,
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            compat_profile=compat_profile,
            allow_reasoning_fallback=allow_reasoning_fallback,
            enable_thinking=enable_thinking,
            timeout_sec=timeout_sec,
            seed=seed,
            http2=http2,
            deadline_monotonic=deadline_monotonic,
            system_prompt_override=system_prompt_override,
            transport_max_retries=transport_max_retries,
            transport_retry_backoff_base_sec=transport_retry_backoff_base_sec,
            transport_retry_backoff_max_sec=transport_retry_backoff_max_sec,
        )
        if diff_fallback.diff is not None:
            fallback_meta = dict(diff_fallback.meta or {})
            totals = _accumulate_llm_totals(totals=totals, meta=fallback_meta, default_calls=0)
            fallback_meta["fallback_from"] = "edits"
            fallback_meta["fallback_reason"] = reason
            fallback_meta["edits_response_mode"] = str(meta.get("structured_output_mode") or "")
            fallback_meta["edits_response_attempts"] = stage_attempts
            if extra_error_meta:
                fallback_meta["fallback_trigger"] = dict(extra_error_meta)
            return PatchResult(diff=diff_fallback.diff, meta=_with_totals(fallback_meta))

        fallback_meta = dict(diff_fallback.meta or {})
        totals = _accumulate_llm_totals(totals=totals, meta=fallback_meta, default_calls=0)
        error_meta: PatcherMeta = {
            "error": reason,
            "fallback_meta": fallback_meta,
            **meta,
        }
        if extra_error_meta:
            error_meta.update(extra_error_meta)
        return PatchResult(diff=None, meta=_with_totals(error_meta))

    for stage_name, extra_payload in stages:
        raw, stage_meta = _chat_completion(
            url=url,
            api_key=api_key,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            compat_profile=compat_profile,
            allow_reasoning_fallback=allow_reasoning_fallback,
            enable_thinking=enable_thinking,
            timeout_sec=timeout_sec,
            seed=seed,
            http2=http2,
            extra_payload=extra_payload,
            deadline_monotonic=deadline_monotonic,
            transport_max_retries=transport_max_retries,
            transport_retry_backoff_base_sec=transport_retry_backoff_base_sec,
            transport_retry_backoff_max_sec=transport_retry_backoff_max_sec,
        )
        totals = _accumulate_llm_totals(totals=totals, meta=stage_meta, default_calls=1)
        meta = dict(stage_meta)
        meta["structured_output_mode"] = stage_name
        attempt: PatcherMeta = {
            "mode": stage_name,
            "ok": raw is not None,
            "error": str(meta.get("error", "")),
            "status": meta.get("status"),
        }
        if raw is None:
            stage_attempts.append(attempt)
            # Don't spam retries when prompt already overflows available context.
            if _is_context_exceeded_http_error(meta) or _is_request_budget_exceeded(meta):
                attempt["terminal"] = True
                break
            if _is_openai_compat_reasoning_only(meta):
                attempt["terminal"] = True
                break
            continue

        raw_reasoning = _get_first_choice_reasoning_content(raw)
        if raw_reasoning is not None:
            meta["reasoning_content_present"] = bool(raw_reasoning.strip())
            meta["reasoning_content_chars"] = len(raw_reasoning)
        raw_reasoning_text = _get_first_choice_reasoning(raw)
        if raw_reasoning_text is not None:
            meta["reasoning_present"] = bool(raw_reasoning_text.strip())
            meta["reasoning_chars"] = len(raw_reasoning_text)

        content = _get_first_choice_content(raw)
        if content is None:
            attempt["parse_error"] = "no_choice_content"
            stage_attempts.append(attempt)
            fallback_reason = "no_json_payload"
            fallback_raw_response = _safe_trunc(raw)
            continue

        payload_candidate = _extract_json_object(content)
        if payload_candidate is None:
            attempt["parse_error"] = "no_json_payload"
            stage_attempts.append(attempt)
            fallback_reason = "no_json_payload"
            fallback_raw_response = content[:2000]
            continue

        edits, edits_shape = _normalize_edits_payload(payload_candidate)
        if edits is None:
            attempt["parse_error"] = "missing_edits"
            stage_attempts.append(attempt)
            fallback_reason = "missing_edits"
            fallback_payload = payload_candidate
            continue

        payload = dict(payload_candidate)
        payload["edits"] = edits
        if edits_shape is not None:
            attempt["edits_shape"] = edits_shape
            meta["edits_payload_shape"] = edits_shape
        stage_attempts.append(attempt)
        break

    if stage_attempts:
        meta["structured_output_attempts"] = stage_attempts

    if payload is None:
        # If prompt is already over context budget, a diff fallback call will also fail.
        if _is_context_exceeded_http_error(meta):
            return PatchResult(diff=None, meta=_with_totals(dict(meta)))
        if _is_openai_compat_reasoning_only(meta):
            return PatchResult(diff=None, meta=_with_totals(dict(meta)))
        if fallback_reason is None:
            return PatchResult(diff=None, meta=_with_totals(dict(meta)))

        error_meta: PatcherMeta = {}
        if fallback_reason == "missing_edits":
            error_meta["payload"] = _safe_trunc(fallback_payload) if fallback_payload else "{}"
        else:
            error_meta["raw_response"] = fallback_raw_response or ""
        return _fallback_to_diff(reason=fallback_reason, extra_error_meta=error_meta)

    edits = payload.get("edits")
    if not isinstance(edits, list) or not edits:
        # Defensive guard: schema/validation above should prevent this.
        return PatchResult(diff=None, meta=_with_totals(dict(meta)))

    changed: list[tuple[str, str, str]] = []
    find_recovery_modes: list[str] = []
    for entry in edits:
        if not isinstance(entry, dict):
            return PatchResult(
                diff=None,
                meta=_with_totals({"error": "bad_edit_entry", "entry": _safe_trunc(entry), **meta}),
            )

        path = entry.get("path")
        find = entry.get("find", entry.get("old"))
        replace = entry.get("replace", entry.get("new"))
        count_raw = entry.get("count", 1)
        count = count_raw if isinstance(count_raw, int) else 1

        if not isinstance(path, str) or not path.strip():
            return PatchResult(
                diff=None,
                meta=_with_totals({"error": "bad_path", "entry": _safe_trunc(entry), **meta}),
            )
        if not isinstance(find, str) or not isinstance(replace, str):
            return PatchResult(
                diff=None,
                meta=_with_totals(
                    {"error": "bad_find_replace", "entry": _safe_trunc(entry), **meta}
                ),
            )
        if count < 0:
            count = 0

        file_path = (cwd / path).resolve()
        try:
            file_path.relative_to(cwd.resolve())
        except Exception:
            return PatchResult(
                diff=None,
                meta=_with_totals({"error": "path_outside_case", "path": path, **meta}),
            )
        if not file_path.exists():
            return PatchResult(
                diff=None,
                meta=_with_totals({"error": "file_not_found", "path": path, **meta}),
            )
        if not file_path.is_file():
            return PatchResult(
                diff=None,
                meta=_with_totals({"error": "path_not_file", "path": path, **meta}),
            )

        try:
            before = file_path.read_text(encoding="utf-8")
        except Exception as exc:
            return PatchResult(
                diff=None,
                meta=_with_totals(
                    {
                        "error": "file_read_error",
                        "path": path,
                        "exc": f"{type(exc).__name__}: {exc}",
                        **meta,
                    }
                ),
            )
        after, match_mode = _apply_find_replace_with_recovery(
            before=before,
            find=find,
            replace=replace,
            count=count,
        )
        if after is None:
            return _fallback_to_diff(
                reason="find_not_found",
                extra_error_meta={"path": path, "find_preview": find[:200]},
            )
        normalized_mode = match_mode.removesuffix("_noop")
        if normalized_mode != "exact":
            find_recovery_modes.append(normalized_mode)
        if after == before:
            continue
        changed.append((path, before, after))

    if not changed:
        return _fallback_to_diff(reason="no_effective_changes")

    if find_recovery_modes:
        unique_modes = sorted(set(find_recovery_modes))
        meta["find_recovery_applied"] = True
        meta["find_recovery_mode"] = unique_modes[0] if len(unique_modes) == 1 else "mixed"
        meta["find_recovery_modes"] = unique_modes
    else:
        meta["find_recovery_applied"] = False

    diff = _build_diff_from_changes(changed)
    return PatchResult(diff=diff, meta=_with_totals(dict(meta)))


def _propose_patch_via_full_file(
    *,
    url: str,
    api_key: str,
    model: str,
    prompt: str,
    cwd: Path,
    temperature: float,
    max_tokens: int,
    compat_profile: str = "auto",
    allow_reasoning_fallback: bool = False,
    enable_thinking: bool = False,
    timeout_sec: float,
    seed: int | None,
    http2: bool,
    deadline_monotonic: float | None = None,
    system_prompt_override: str | None = None,
    transport_max_retries: int = _DEFAULT_CHAT_COMPLETION_TRANSPORT_MAX_RETRIES,
    transport_retry_backoff_base_sec: float = _DEFAULT_CHAT_COMPLETION_RETRY_BACKOFF_BASE_SEC,
    transport_retry_backoff_max_sec: float = _DEFAULT_CHAT_COMPLETION_RETRY_BACKOFF_MAX_SEC,
) -> PatchResult:
    totals = (0, 0, 0, 0, 0, False, True)
    target_rel, support_reason = _resolve_full_file_target(cwd=cwd)
    full_file_meta: PatcherMeta = {
        "full_file_target_path": target_rel or "",
        "full_file_supported": target_rel is not None,
        "full_file_support_reason": support_reason,
        "full_file_output_chars": 0,
        "full_file_code_fence_stripped": False,
    }

    def _with_totals(meta: PatcherMeta) -> PatcherMeta:
        return _meta_with_llm_totals(meta=meta, totals=totals)

    def _fallback_to_diff(
        *,
        reason: str,
        extra_error_meta: PatcherMeta | None = None,
    ) -> PatchResult:
        nonlocal totals
        diff_fallback = _propose_patch_via_diff(
            url=url,
            api_key=api_key,
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            compat_profile=compat_profile,
            allow_reasoning_fallback=allow_reasoning_fallback,
            enable_thinking=enable_thinking,
            timeout_sec=timeout_sec,
            seed=seed,
            http2=http2,
            deadline_monotonic=deadline_monotonic,
            system_prompt_override=system_prompt_override,
            transport_max_retries=transport_max_retries,
            transport_retry_backoff_base_sec=transport_retry_backoff_base_sec,
            transport_retry_backoff_max_sec=transport_retry_backoff_max_sec,
        )
        fallback_meta = dict(diff_fallback.meta or {})
        totals = _accumulate_llm_totals(totals=totals, meta=fallback_meta, default_calls=0)
        if diff_fallback.diff is not None:
            fallback_meta.update(full_file_meta)
            fallback_meta["fallback_from"] = "full_file"
            fallback_meta["fallback_reason"] = reason
            if extra_error_meta:
                fallback_meta["fallback_trigger"] = dict(extra_error_meta)
            return PatchResult(diff=diff_fallback.diff, meta=_with_totals(fallback_meta))

        error_meta: PatcherMeta = {
            **full_file_meta,
            "error": reason,
            "fallback_meta": fallback_meta,
        }
        if extra_error_meta:
            error_meta.update(extra_error_meta)
        return PatchResult(diff=None, meta=_with_totals(error_meta))

    if target_rel is None:
        return _fallback_to_diff(reason="unsupported_case_shape")

    target_path = (cwd / target_rel).resolve()
    try:
        target_path.relative_to(cwd.resolve())
    except Exception:
        return PatchResult(
            diff=None,
            meta=_with_totals(
                {
                    **full_file_meta,
                    "error": "path_outside_case",
                    "path": target_rel,
                }
            ),
        )
    try:
        before = target_path.read_text(encoding="utf-8")
    except Exception as exc:
        return PatchResult(
            diff=None,
            meta=_with_totals(
                {
                    **full_file_meta,
                    "error": "file_read_error",
                    "path": target_rel,
                    "exc": f"{type(exc).__name__}: {exc}",
                }
            ),
        )

    attempts: list[list[dict[str, str]]] = [
        [
            {
                "role": "system",
                "content": system_prompt_override
                or (
                    f"You fix failing tests by returning the complete repaired contents of "
                    f"{target_rel}. Return ONLY the full contents of {target_rel}. "
                    "No prose. No markdown fences. Do not return a diff."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        [
            {
                "role": "system",
                "content": system_prompt_override
                or (
                    f"Return ONLY the complete repaired contents of {target_rel}. "
                    "Start directly with the file contents. "
                    "No prose. No markdown fences. Do not return a diff."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    ]

    last_error_meta: PatcherMeta | None = None
    for idx, messages in enumerate(attempts, start=1):
        raw, meta = _chat_completion(
            url=url,
            api_key=api_key,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            compat_profile=compat_profile,
            allow_reasoning_fallback=allow_reasoning_fallback,
            enable_thinking=enable_thinking,
            timeout_sec=timeout_sec,
            seed=seed,
            http2=http2,
            deadline_monotonic=deadline_monotonic,
            transport_max_retries=transport_max_retries,
            transport_retry_backoff_base_sec=transport_retry_backoff_base_sec,
            transport_retry_backoff_max_sec=transport_retry_backoff_max_sec,
        )
        totals = _accumulate_llm_totals(totals=totals, meta=meta, default_calls=1)
        attempt_meta: PatcherMeta = {
            **full_file_meta,
            "attempt": idx,
            **meta,
        }
        if raw is None:
            last_error_meta = attempt_meta
            if _is_openai_compat_reasoning_only(meta):
                break
            if _is_request_budget_exceeded(meta):
                break
            continue

        repaired, response_meta = _normalize_full_file_response(raw)
        attempt_meta.update(response_meta)
        full_file_meta["full_file_code_fence_stripped"] = bool(
            attempt_meta.get("full_file_code_fence_stripped")
        )
        full_file_meta["full_file_output_chars"] = _nonnegative_int(
            attempt_meta.get("full_file_output_chars"),
            default=0,
        )
        if repaired is None:
            last_error_meta = attempt_meta
            if str(attempt_meta.get("error") or "") in {
                "truncated_full_file_response",
                "returned_diff_instead_of_file",
            }:
                break
            continue
        if repaired == before:
            return _fallback_to_diff(reason="no_effective_changes")

        diff = _build_diff_from_changes([(target_rel, before, repaired)])
        if diff.strip():
            return PatchResult(diff=diff, meta=_with_totals(attempt_meta))
        return _fallback_to_diff(reason="no_effective_changes")

    fallback_reason = (
        str(last_error_meta.get("error") or "").strip() if last_error_meta is not None else ""
    )
    if not fallback_reason:
        fallback_reason = "empty_full_file_response"
    fallback_trigger = None
    if last_error_meta is not None and any(
        key in last_error_meta for key in ("raw_response", "attempt", "status", "finish_reason")
    ):
        fallback_trigger = {
            key: last_error_meta[key]
            for key in ("attempt", "status", "finish_reason", "error", "raw_response")
            if key in last_error_meta
        }
    return _fallback_to_diff(reason=fallback_reason, extra_error_meta=fallback_trigger)


async def _propose_patch_via_diff_async(
    *,
    url: str,
    api_key: str,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    compat_profile: str = "auto",
    allow_reasoning_fallback: bool = False,
    enable_thinking: bool = False,
    timeout_sec: float,
    seed: int | None,
    http2: bool,
    deadline_monotonic: float | None = None,
    transport_max_retries: int = _DEFAULT_CHAT_COMPLETION_TRANSPORT_MAX_RETRIES,
    transport_retry_backoff_base_sec: float = _DEFAULT_CHAT_COMPLETION_RETRY_BACKOFF_BASE_SEC,
    transport_retry_backoff_max_sec: float = _DEFAULT_CHAT_COMPLETION_RETRY_BACKOFF_MAX_SEC,
) -> PatchResult:
    base_messages = [
        {
            "role": "system",
            "content": (
                "You fix failing tests by returning a single unified diff. "
                "Return ONLY the diff; no explanations. Do not use Markdown fences."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    attempts: list[list[dict[str, str]]] = [
        base_messages,
        [
            {
                "role": "system",
                "content": (
                    "You fix failing tests by returning a single unified diff. "
                    "Return ONLY the diff; no explanations. Do not use Markdown fences. "
                    "Start your response with 'diff --git a/'."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    ]

    last_meta: PatcherMeta | None = None
    totals = (0, 0, 0, 0, 0, False, True)
    for idx, messages in enumerate(attempts, start=1):
        raw, meta = await _chat_completion_async(
            url=url,
            api_key=api_key,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            compat_profile=compat_profile,
            allow_reasoning_fallback=allow_reasoning_fallback,
            enable_thinking=enable_thinking,
            timeout_sec=timeout_sec,
            seed=seed,
            http2=http2,
            deadline_monotonic=deadline_monotonic,
            transport_max_retries=transport_max_retries,
            transport_retry_backoff_base_sec=transport_retry_backoff_base_sec,
            transport_retry_backoff_max_sec=transport_retry_backoff_max_sec,
        )
        totals = _accumulate_llm_totals(totals=totals, meta=meta, default_calls=1)
        last_meta = _meta_with_llm_totals(meta={"attempt": idx, **meta}, totals=totals)
        if raw is None:
            if _is_openai_compat_reasoning_only(meta):
                break
            if _is_request_budget_exceeded(meta):
                break
            continue

        content = _get_first_choice_content(raw)
        if content is None:
            last_meta["error"] = "no_choice_content"
            last_meta["response"] = _safe_trunc(raw)
            continue

        diff = _extract_unified_diff(content)
        if diff is None:
            last_meta["error"] = "no_diff_found"
            last_meta["raw_response"] = content[:2000]
            continue

        fixed = _fix_hunk_counts(diff)
        return PatchResult(diff=fixed, meta=last_meta)

    if last_meta is None:
        last_meta = _meta_with_llm_totals(meta={"error": "unknown"}, totals=totals)
    return PatchResult(diff=None, meta=last_meta)


async def _propose_patch_via_edits_async(
    *,
    url: str,
    api_key: str,
    model: str,
    prompt: str,
    cwd: Path,
    temperature: float,
    max_tokens: int,
    compat_profile: str = "auto",
    allow_reasoning_fallback: bool = False,
    enable_thinking: bool = False,
    timeout_sec: float,
    seed: int | None,
    http2: bool,
    deadline_monotonic: float | None = None,
    allow_diff_fallback: bool = True,
    transport_max_retries: int = _DEFAULT_CHAT_COMPLETION_TRANSPORT_MAX_RETRIES,
    transport_retry_backoff_base_sec: float = _DEFAULT_CHAT_COMPLETION_RETRY_BACKOFF_BASE_SEC,
    transport_retry_backoff_max_sec: float = _DEFAULT_CHAT_COMPLETION_RETRY_BACKOFF_MAX_SEC,
) -> PatchResult:
    messages = [
        {
            "role": "system",
            "content": (
                "Return only JSON with this schema: "
                '{"edits":[{"path":"relative/path.py","find":"exact old text","replace":"new text","count":1}]}. '
                "No prose. No markdown. "
                "Use only exact existing paths from the prompt's available file list. "
                "Never use placeholder paths like relative/path.py."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    totals = (0, 0, 0, 0, 0, False, True)

    def _with_totals(meta: PatcherMeta) -> PatcherMeta:
        return _meta_with_llm_totals(meta=meta, totals=totals)

    payload: dict[str, Any] | None = None
    meta: PatcherMeta = {}
    stages = [
        (
            "json_schema",
            {
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "patch_edits",
                        "schema": PATCH_EDITS_JSON_SCHEMA,
                    },
                }
            },
        ),
        ("json_object", {"response_format": {"type": "json_object"}}),
        ("plain", None),
    ]
    stage_attempts: list[PatcherMeta] = []
    fallback_reason: str | None = None
    fallback_raw_response: str | None = None
    fallback_payload: dict[str, Any] | None = None

    async def _fallback_to_diff(
        *,
        reason: str,
        extra_error_meta: PatcherMeta | None = None,
    ) -> PatchResult:
        nonlocal totals
        if not allow_diff_fallback:
            error_meta: PatcherMeta = {"error": reason, **meta}
            if extra_error_meta:
                error_meta.update(extra_error_meta)
            return PatchResult(diff=None, meta=_with_totals(error_meta))

        diff_fallback = await _propose_patch_via_diff_async(
            url=url,
            api_key=api_key,
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            compat_profile=compat_profile,
            allow_reasoning_fallback=allow_reasoning_fallback,
            enable_thinking=enable_thinking,
            timeout_sec=timeout_sec,
            seed=seed,
            http2=http2,
            deadline_monotonic=deadline_monotonic,
            transport_max_retries=transport_max_retries,
            transport_retry_backoff_base_sec=transport_retry_backoff_base_sec,
            transport_retry_backoff_max_sec=transport_retry_backoff_max_sec,
        )
        if diff_fallback.diff is not None:
            fallback_meta = dict(diff_fallback.meta or {})
            totals = _accumulate_llm_totals(totals=totals, meta=fallback_meta, default_calls=0)
            fallback_meta["fallback_from"] = "edits"
            fallback_meta["fallback_reason"] = reason
            fallback_meta["edits_response_mode"] = str(meta.get("structured_output_mode") or "")
            fallback_meta["edits_response_attempts"] = stage_attempts
            if extra_error_meta:
                fallback_meta["fallback_trigger"] = dict(extra_error_meta)
            return PatchResult(diff=diff_fallback.diff, meta=_with_totals(fallback_meta))

        fallback_meta = dict(diff_fallback.meta or {})
        totals = _accumulate_llm_totals(totals=totals, meta=fallback_meta, default_calls=0)
        error_meta: PatcherMeta = {
            "error": reason,
            "fallback_meta": fallback_meta,
            **meta,
        }
        if extra_error_meta:
            error_meta.update(extra_error_meta)
        return PatchResult(diff=None, meta=_with_totals(error_meta))

    for stage_name, extra_payload in stages:
        raw, stage_meta = await _chat_completion_async(
            url=url,
            api_key=api_key,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            compat_profile=compat_profile,
            allow_reasoning_fallback=allow_reasoning_fallback,
            enable_thinking=enable_thinking,
            timeout_sec=timeout_sec,
            seed=seed,
            http2=http2,
            extra_payload=extra_payload,
            deadline_monotonic=deadline_monotonic,
            transport_max_retries=transport_max_retries,
            transport_retry_backoff_base_sec=transport_retry_backoff_base_sec,
            transport_retry_backoff_max_sec=transport_retry_backoff_max_sec,
        )
        totals = _accumulate_llm_totals(totals=totals, meta=stage_meta, default_calls=1)
        meta = dict(stage_meta)
        meta["structured_output_mode"] = stage_name
        attempt: PatcherMeta = {
            "mode": stage_name,
            "ok": raw is not None,
            "error": str(meta.get("error", "")),
            "status": meta.get("status"),
        }
        if raw is None:
            stage_attempts.append(attempt)
            if _is_context_exceeded_http_error(meta) or _is_request_budget_exceeded(meta):
                attempt["terminal"] = True
                break
            if _is_openai_compat_reasoning_only(meta):
                attempt["terminal"] = True
                break
            continue

        raw_reasoning = _get_first_choice_reasoning_content(raw)
        if raw_reasoning is not None:
            meta["reasoning_content_present"] = bool(raw_reasoning.strip())
            meta["reasoning_content_chars"] = len(raw_reasoning)
        raw_reasoning_text = _get_first_choice_reasoning(raw)
        if raw_reasoning_text is not None:
            meta["reasoning_present"] = bool(raw_reasoning_text.strip())
            meta["reasoning_chars"] = len(raw_reasoning_text)

        content = _get_first_choice_content(raw)
        if content is None:
            attempt["parse_error"] = "no_choice_content"
            stage_attempts.append(attempt)
            fallback_reason = "no_json_payload"
            fallback_raw_response = _safe_trunc(raw)
            continue

        payload_candidate = _extract_json_object(content)
        if payload_candidate is None:
            attempt["parse_error"] = "no_json_payload"
            stage_attempts.append(attempt)
            fallback_reason = "no_json_payload"
            fallback_raw_response = content[:2000]
            continue

        edits, edits_shape = _normalize_edits_payload(payload_candidate)
        if edits is None:
            attempt["parse_error"] = "missing_edits"
            stage_attempts.append(attempt)
            fallback_reason = "missing_edits"
            fallback_payload = payload_candidate
            continue

        payload = dict(payload_candidate)
        payload["edits"] = edits
        if edits_shape is not None:
            attempt["edits_shape"] = edits_shape
            meta["edits_payload_shape"] = edits_shape
        stage_attempts.append(attempt)
        break

    if stage_attempts:
        meta["structured_output_attempts"] = stage_attempts

    if payload is None:
        if _is_context_exceeded_http_error(meta):
            return PatchResult(diff=None, meta=_with_totals(dict(meta)))
        if _is_openai_compat_reasoning_only(meta):
            return PatchResult(diff=None, meta=_with_totals(dict(meta)))
        if fallback_reason is None:
            return PatchResult(diff=None, meta=_with_totals(dict(meta)))

        error_meta: PatcherMeta = {}
        if fallback_reason == "missing_edits":
            error_meta["payload"] = _safe_trunc(fallback_payload) if fallback_payload else "{}"
        else:
            error_meta["raw_response"] = fallback_raw_response or ""
        return await _fallback_to_diff(reason=fallback_reason, extra_error_meta=error_meta)

    edits = payload.get("edits")
    if not isinstance(edits, list) or not edits:
        return PatchResult(diff=None, meta=_with_totals(dict(meta)))

    changed: list[tuple[str, str, str]] = []
    find_recovery_modes: list[str] = []
    for entry in edits:
        if not isinstance(entry, dict):
            return PatchResult(
                diff=None,
                meta=_with_totals({"error": "bad_edit_entry", "entry": _safe_trunc(entry), **meta}),
            )

        path = entry.get("path")
        find = entry.get("find", entry.get("old"))
        replace = entry.get("replace", entry.get("new"))
        count_raw = entry.get("count", 1)
        count = count_raw if isinstance(count_raw, int) else 1

        if not isinstance(path, str) or not path.strip():
            return PatchResult(
                diff=None,
                meta=_with_totals({"error": "bad_path", "entry": _safe_trunc(entry), **meta}),
            )
        if not isinstance(find, str) or not isinstance(replace, str):
            return PatchResult(
                diff=None,
                meta=_with_totals(
                    {"error": "bad_find_replace", "entry": _safe_trunc(entry), **meta}
                ),
            )
        if count < 0:
            count = 0

        file_path = (cwd / path).resolve()
        try:
            file_path.relative_to(cwd.resolve())
        except Exception:
            return PatchResult(
                diff=None,
                meta=_with_totals({"error": "path_outside_case", "path": path, **meta}),
            )
        if not file_path.exists():
            return PatchResult(
                diff=None,
                meta=_with_totals({"error": "file_not_found", "path": path, **meta}),
            )
        if not file_path.is_file():
            return PatchResult(
                diff=None,
                meta=_with_totals({"error": "path_not_file", "path": path, **meta}),
            )

        try:
            before = file_path.read_text(encoding="utf-8")
        except Exception as exc:
            return PatchResult(
                diff=None,
                meta=_with_totals(
                    {
                        "error": "file_read_error",
                        "path": path,
                        "exc": f"{type(exc).__name__}: {exc}",
                        **meta,
                    }
                ),
            )
        after, match_mode = _apply_find_replace_with_recovery(
            before=before,
            find=find,
            replace=replace,
            count=count,
        )
        if after is None:
            return await _fallback_to_diff(
                reason="find_not_found",
                extra_error_meta={"path": path, "find_preview": find[:200]},
            )
        normalized_mode = match_mode.removesuffix("_noop")
        if normalized_mode != "exact":
            find_recovery_modes.append(normalized_mode)
        if after == before:
            continue
        changed.append((path, before, after))

    if not changed:
        return await _fallback_to_diff(reason="no_effective_changes")

    if find_recovery_modes:
        unique_modes = sorted(set(find_recovery_modes))
        meta["find_recovery_applied"] = True
        meta["find_recovery_mode"] = unique_modes[0] if len(unique_modes) == 1 else "mixed"
        meta["find_recovery_modes"] = unique_modes
    else:
        meta["find_recovery_applied"] = False

    diff = _build_diff_from_changes(changed)
    return PatchResult(diff=diff, meta=_with_totals(dict(meta)))


async def _propose_patch_via_full_file_async(
    *,
    url: str,
    api_key: str,
    model: str,
    prompt: str,
    cwd: Path,
    temperature: float,
    max_tokens: int,
    compat_profile: str = "auto",
    allow_reasoning_fallback: bool = False,
    enable_thinking: bool = False,
    timeout_sec: float,
    seed: int | None,
    http2: bool,
    deadline_monotonic: float | None = None,
    transport_max_retries: int = _DEFAULT_CHAT_COMPLETION_TRANSPORT_MAX_RETRIES,
    transport_retry_backoff_base_sec: float = _DEFAULT_CHAT_COMPLETION_RETRY_BACKOFF_BASE_SEC,
    transport_retry_backoff_max_sec: float = _DEFAULT_CHAT_COMPLETION_RETRY_BACKOFF_MAX_SEC,
) -> PatchResult:
    totals = (0, 0, 0, 0, 0, False, True)
    target_rel, support_reason = _resolve_full_file_target(cwd=cwd)
    full_file_meta: PatcherMeta = {
        "full_file_target_path": target_rel or "",
        "full_file_supported": target_rel is not None,
        "full_file_support_reason": support_reason,
        "full_file_output_chars": 0,
        "full_file_code_fence_stripped": False,
    }

    def _with_totals(meta: PatcherMeta) -> PatcherMeta:
        return _meta_with_llm_totals(meta=meta, totals=totals)

    async def _fallback_to_diff(
        *,
        reason: str,
        extra_error_meta: PatcherMeta | None = None,
    ) -> PatchResult:
        nonlocal totals
        diff_fallback = await _propose_patch_via_diff_async(
            url=url,
            api_key=api_key,
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            compat_profile=compat_profile,
            allow_reasoning_fallback=allow_reasoning_fallback,
            enable_thinking=enable_thinking,
            timeout_sec=timeout_sec,
            seed=seed,
            http2=http2,
            deadline_monotonic=deadline_monotonic,
            transport_max_retries=transport_max_retries,
            transport_retry_backoff_base_sec=transport_retry_backoff_base_sec,
            transport_retry_backoff_max_sec=transport_retry_backoff_max_sec,
        )
        fallback_meta = dict(diff_fallback.meta or {})
        totals = _accumulate_llm_totals(totals=totals, meta=fallback_meta, default_calls=0)
        if diff_fallback.diff is not None:
            fallback_meta.update(full_file_meta)
            fallback_meta["fallback_from"] = "full_file"
            fallback_meta["fallback_reason"] = reason
            if extra_error_meta:
                fallback_meta["fallback_trigger"] = dict(extra_error_meta)
            return PatchResult(diff=diff_fallback.diff, meta=_with_totals(fallback_meta))

        error_meta: PatcherMeta = {
            **full_file_meta,
            "error": reason,
            "fallback_meta": fallback_meta,
        }
        if extra_error_meta:
            error_meta.update(extra_error_meta)
        return PatchResult(diff=None, meta=_with_totals(error_meta))

    if target_rel is None:
        return await _fallback_to_diff(reason="unsupported_case_shape")

    target_path = (cwd / target_rel).resolve()
    try:
        target_path.relative_to(cwd.resolve())
    except Exception:
        return PatchResult(
            diff=None,
            meta=_with_totals(
                {
                    **full_file_meta,
                    "error": "path_outside_case",
                    "path": target_rel,
                }
            ),
        )
    try:
        before = target_path.read_text(encoding="utf-8")
    except Exception as exc:
        return PatchResult(
            diff=None,
            meta=_with_totals(
                {
                    **full_file_meta,
                    "error": "file_read_error",
                    "path": target_rel,
                    "exc": f"{type(exc).__name__}: {exc}",
                }
            ),
        )

    attempts: list[list[dict[str, str]]] = [
        [
            {
                "role": "system",
                "content": (
                    f"You fix failing tests by returning the complete repaired contents of "
                    f"{target_rel}. Return ONLY the full contents of {target_rel}. "
                    "No prose. No markdown fences. Do not return a diff."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        [
            {
                "role": "system",
                "content": (
                    f"Return ONLY the complete repaired contents of {target_rel}. "
                    "Start directly with the file contents. "
                    "No prose. No markdown fences. Do not return a diff."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    ]

    last_error_meta: PatcherMeta | None = None
    for idx, messages in enumerate(attempts, start=1):
        raw, meta = await _chat_completion_async(
            url=url,
            api_key=api_key,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            compat_profile=compat_profile,
            allow_reasoning_fallback=allow_reasoning_fallback,
            enable_thinking=enable_thinking,
            timeout_sec=timeout_sec,
            seed=seed,
            http2=http2,
            deadline_monotonic=deadline_monotonic,
            transport_max_retries=transport_max_retries,
            transport_retry_backoff_base_sec=transport_retry_backoff_base_sec,
            transport_retry_backoff_max_sec=transport_retry_backoff_max_sec,
        )
        totals = _accumulate_llm_totals(totals=totals, meta=meta, default_calls=1)
        attempt_meta: PatcherMeta = {
            **full_file_meta,
            "attempt": idx,
            **meta,
        }
        if raw is None:
            last_error_meta = attempt_meta
            if _is_openai_compat_reasoning_only(meta):
                break
            if _is_request_budget_exceeded(meta):
                break
            continue

        repaired, response_meta = _normalize_full_file_response(raw)
        attempt_meta.update(response_meta)
        full_file_meta["full_file_code_fence_stripped"] = bool(
            attempt_meta.get("full_file_code_fence_stripped")
        )
        full_file_meta["full_file_output_chars"] = _nonnegative_int(
            attempt_meta.get("full_file_output_chars"),
            default=0,
        )
        if repaired is None:
            last_error_meta = attempt_meta
            if str(attempt_meta.get("error") or "") in {
                "truncated_full_file_response",
                "returned_diff_instead_of_file",
            }:
                break
            continue
        if repaired == before:
            return await _fallback_to_diff(reason="no_effective_changes")

        diff = _build_diff_from_changes([(target_rel, before, repaired)])
        if diff.strip():
            return PatchResult(diff=diff, meta=_with_totals(attempt_meta))
        return await _fallback_to_diff(reason="no_effective_changes")

    fallback_reason = (
        str(last_error_meta.get("error") or "").strip() if last_error_meta is not None else ""
    )
    if not fallback_reason:
        fallback_reason = "empty_full_file_response"
    fallback_trigger = None
    if last_error_meta is not None and any(
        key in last_error_meta for key in ("raw_response", "attempt", "status", "finish_reason")
    ):
        fallback_trigger = {
            key: last_error_meta[key]
            for key in ("attempt", "status", "finish_reason", "error", "raw_response")
            if key in last_error_meta
        }
    return await _fallback_to_diff(reason=fallback_reason, extra_error_meta=fallback_trigger)


def _eval_openai_compat_config(
    *,
    compat_profile: str,
    allow_reasoning_fallback: bool = False,
    enable_thinking: bool = False,
) -> BaselineOpenAICompatConfig:
    if enable_thinking and not allow_reasoning_fallback:
        allow_reasoning_fallback = True
    return BaselineOpenAICompatConfig(
        profile=normalize_compat_profile(compat_profile),
        allow_reasoning_fallback=allow_reasoning_fallback,
        enable_thinking=enable_thinking,
    )


def _merge_extra_body_into_payload(payload: dict[str, Any]) -> dict[str, Any]:
    normalized = dict(payload)
    extra_body = normalized.pop("extra_body", None)
    if not isinstance(extra_body, dict):
        return normalized
    for key, value in extra_body.items():
        normalized_key = str(key)
        existing = normalized.get(normalized_key)
        if isinstance(existing, dict) and isinstance(value, dict):
            merged = dict(existing)
            merged.update(value)
            normalized[normalized_key] = merged
            continue
        normalized[normalized_key] = value
    return normalized


def _apply_eval_openai_compat_request_profile(
    *,
    payload: dict[str, Any],
    compat_profile: str,
    allow_reasoning_fallback: bool = False,
    enable_thinking: bool = False,
    base_url: str,
) -> tuple[dict[str, Any], OpenAICompatRequestMeta, BaselineOpenAICompatConfig]:
    compat_config = _eval_openai_compat_config(
        compat_profile=compat_profile,
        allow_reasoning_fallback=allow_reasoning_fallback,
        enable_thinking=enable_thinking,
    )
    shaped_payload, request_meta = apply_compat_request_profile(
        payload,
        compat_config=compat_config,
        base_url=base_url,
    )
    return _merge_extra_body_into_payload(shaped_payload), request_meta, compat_config


def _apply_reasoning_fallback_to_response_payload(
    response: dict[str, Any],
    *,
    normalized_text: str,
) -> None:
    choices = response.get("choices")
    if not isinstance(choices, list) or not choices:
        return
    choice0 = choices[0]
    if not isinstance(choice0, dict):
        return
    message = choice0.get("message")
    if isinstance(message, dict):
        if not _coerce_content_to_text(message.get("content")):
            message["content"] = normalized_text
        return
    text = choice0.get("text")
    if not isinstance(text, str) or not text.strip():
        choice0["text"] = normalized_text


def _chat_completion(
    *,
    url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    compat_profile: str = "auto",
    allow_reasoning_fallback: bool = False,
    enable_thinking: bool = False,
    timeout_sec: float,
    seed: int | None,
    http2: bool,
    extra_payload: dict[str, Any] | None = None,
    deadline_monotonic: float | None = None,
    transport_max_retries: int = _DEFAULT_CHAT_COMPLETION_TRANSPORT_MAX_RETRIES,
    transport_retry_backoff_base_sec: float = _DEFAULT_CHAT_COMPLETION_RETRY_BACKOFF_BASE_SEC,
    transport_retry_backoff_max_sec: float = _DEFAULT_CHAT_COMPLETION_RETRY_BACKOFF_MAX_SEC,
) -> tuple[dict[str, Any] | None, PatcherMeta]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if seed is not None:
        payload["seed"] = seed
    if extra_payload:
        payload.update(extra_payload)
    compat_base_url = _base_url_from_chat_url(url)
    payload, compat_request_meta, compat_config = _apply_eval_openai_compat_request_profile(
        payload=payload,
        compat_profile=compat_profile,
        allow_reasoning_fallback=allow_reasoning_fallback,
        enable_thinking=enable_thinking,
        base_url=compat_base_url,
    )

    max_retries = max(0, int(transport_max_retries))
    retry_backoff_base_sec = max(0.0, float(transport_retry_backoff_base_sec))
    retry_backoff_max_sec = max(0.0, float(transport_retry_backoff_max_sec))
    if retry_backoff_max_sec < retry_backoff_base_sec:
        retry_backoff_max_sec = retry_backoff_base_sec
    retry_attempts: list[dict[str, Any]] = []
    call_count = 0

    base_meta: PatcherMeta = {
        "model": model,
        "base_url": compat_base_url,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout_sec": timeout_sec,
        "effective_timeout_sec": timeout_sec,
        "remaining_budget_sec": _remaining_request_budget_sec(deadline_monotonic),
        "llm_call_count": 0,
        "usage": _usage_payload_from_counts(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        **compat_request_meta.to_dict(),
        "transport_retry_limit": max_retries,
        "transport_retry_limit_effective": max_retries,
        "transport_retry_backoff_base_sec": retry_backoff_base_sec,
        "transport_retry_backoff_max_sec": retry_backoff_max_sec,
        "api_compat_seed_requested": seed,
        "api_compat_seed_acknowledged": False if seed is not None else None,
        "api_compat_seed_response": None,
    }

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")

    def _send_request(*, timeout: float, use_http2: bool) -> tuple[int, str, str]:
        if _should_use_proxy_transport(url):
            with httpx.Client(timeout=timeout, http2=use_http2, trust_env=True) as client:
                response = client.post(url, content=payload_bytes, headers=headers)
                http_version = str(getattr(response, "http_version", "") or "")
                return int(response.status_code), response.text, http_version
        client = _get_thread_http_client(url=url, http2=use_http2)
        status, raw = client.post_json(
            target=_chat_http_target(url),
            body=payload_bytes,
            headers=headers,
            timeout_sec=timeout,
        )
        http_version = str(getattr(client, "last_http_version", "") or "")
        return status, raw, http_version

    for attempt_index in range(max_retries + 1):
        remaining_budget_sec = _remaining_request_budget_sec(deadline_monotonic)
        effective_timeout_sec = timeout_sec
        if remaining_budget_sec is not None:
            effective_timeout_sec = min(timeout_sec, remaining_budget_sec)

        if effective_timeout_sec <= 0:
            _emit_openai_transport_progress(
                "openai_transport",
                note="request budget exhausted",
                transport_state="request_budget_exhausted",
                transport_attempt=call_count,
                transport_retry_count=len(retry_attempts),
            )
            budget_meta: PatcherMeta = {
                **base_meta,
                "error": "request_budget_exceeded",
                "llm_call_count": call_count,
                "effective_timeout_sec": effective_timeout_sec,
                "remaining_budget_sec": remaining_budget_sec,
                "transport_retry_count": len(retry_attempts),
            }
            if retry_attempts:
                budget_meta["transport_retry_attempts"] = retry_attempts
            return None, budget_meta

        call_count += 1
        transport_http2_effective = bool(http2)
        transport_http2_fallback_used = False
        _emit_openai_transport_progress(
            "openai_transport",
            note=f"request {call_count} started",
            transport_state="request_started",
            transport_attempt=call_count,
            transport_retry_count=len(retry_attempts),
            effective_timeout_sec=effective_timeout_sec,
            remaining_budget_sec=remaining_budget_sec,
            transport_http2_effective=transport_http2_effective,
        )
        attempt_meta: PatcherMeta = {
            **base_meta,
            "llm_call_count": call_count,
            "effective_timeout_sec": effective_timeout_sec,
            "remaining_budget_sec": remaining_budget_sec,
            "transport_impl": "httpx",
            "transport_http2_requested": bool(http2),
            "transport_http2_effective": bool(transport_http2_effective),
            "transport_http2_fallback_used": bool(transport_http2_fallback_used),
        }
        raw_text = ""
        status = 0
        http_version = ""
        error_meta: PatcherMeta | None = None
        try:
            status, raw_text, http_version = _send_request(
                timeout=effective_timeout_sec,
                use_http2=transport_http2_effective,
            )
        except Exception as exc:
            if transport_http2_effective and _is_http2_protocol_error(exc):
                transport_http2_effective = False
                transport_http2_fallback_used = True
                _emit_openai_transport_progress(
                    "openai_transport",
                    note="HTTP/2 protocol error; retrying over HTTP/1.1",
                    transport_state="http2_fallback",
                    transport_attempt=call_count,
                    transport_retry_count=len(retry_attempts),
                )
                attempt_meta["transport_http2_effective"] = False
                attempt_meta["transport_http2_fallback_used"] = True
                try:
                    status, raw_text, http_version = _send_request(
                        timeout=effective_timeout_sec,
                        use_http2=False,
                    )
                except ValueError as inner_exc:
                    error_meta = {
                        **attempt_meta,
                        "error": "bad_request_url",
                        "exc": f"{type(inner_exc).__name__}: {inner_exc}",
                    }
                except Exception as inner_exc:
                    error_meta = {
                        **attempt_meta,
                        "error": "request_failed",
                        "exc": f"{type(inner_exc).__name__}: {inner_exc}",
                    }
            elif isinstance(exc, ValueError):
                error_meta = {
                    **attempt_meta,
                    "error": "bad_request_url",
                    "exc": f"{type(exc).__name__}: {exc}",
                }
            else:
                error_meta = {
                    **attempt_meta,
                    "error": "request_failed",
                    "exc": f"{type(exc).__name__}: {exc}",
                }
        if http_version:
            attempt_meta["transport_http_version"] = http_version
        if error_meta is None and status >= 400:
            error_meta = {
                **attempt_meta,
                "error": "http_error",
                "status": int(status),
                "body": raw_text[:2000],
            }

        if error_meta is not None:
            should_retry = attempt_index < max_retries and _is_retryable_chat_completion_error(
                error_meta
            )
            if should_retry:
                backoff_sec = _chat_completion_retry_backoff_sec(
                    attempt_index,
                    base_sec=retry_backoff_base_sec,
                    max_sec=retry_backoff_max_sec,
                )
                remaining_retry_budget_sec = _remaining_request_budget_sec(deadline_monotonic)
                if remaining_retry_budget_sec is not None:
                    backoff_sec = min(backoff_sec, max(0.0, remaining_retry_budget_sec - 1e-3))
                retry_meta: dict[str, Any] = {
                    "attempt": call_count,
                    "error": str(error_meta.get("error", "")),
                    "backoff_sec": backoff_sec,
                }
                status = error_meta.get("status")
                if isinstance(status, int):
                    retry_meta["status"] = status
                retry_attempts.append(retry_meta)
                retry_count = len(retry_attempts)
                status_suffix = f" status={status}" if isinstance(status, int) else ""
                _emit_openai_transport_progress(
                    "openai_transport",
                    note=(
                        f"retry {retry_count}/{max_retries} scheduled for "
                        f"{error_meta.get('error', 'request_failed')}{status_suffix}"
                    ),
                    transport_state="retry_scheduled",
                    transport_attempt=call_count,
                    next_transport_attempt=call_count + 1,
                    transport_retry_count=retry_count,
                    error=str(error_meta.get("error", "")),
                    status=status if isinstance(status, int) else None,
                    backoff_sec=backoff_sec,
                )
                if backoff_sec > 0:
                    _emit_openai_transport_progress(
                        "openai_transport",
                        note=f"backoff {backoff_sec:.1f}s before retry {call_count + 1}",
                        transport_state="backoff_sleep",
                        transport_attempt=call_count,
                        next_transport_attempt=call_count + 1,
                        transport_retry_count=retry_count,
                        backoff_sec=backoff_sec,
                    )
                    time.sleep(backoff_sec)
                continue

            final_error_meta: PatcherMeta = {
                **error_meta,
                "transport_retry_count": len(retry_attempts),
            }
            if retry_attempts:
                final_error_meta["transport_retry_attempts"] = retry_attempts
            return None, final_error_meta

        try:
            data = json.loads(raw_text)
        except Exception:
            bad_json_meta: PatcherMeta = {
                **attempt_meta,
                "error": "bad_json_response",
                "body": raw_text[:2000],
                "transport_retry_count": len(retry_attempts),
            }
            if retry_attempts:
                bad_json_meta["transport_retry_attempts"] = retry_attempts
            return None, bad_json_meta

        meta: PatcherMeta = {
            **attempt_meta,
            "transport_retry_count": len(retry_attempts),
        }
        if retry_attempts:
            meta["transport_retry_attempts"] = retry_attempts
        usage = data.get("usage")
        if isinstance(usage, dict):
            prompt_tokens, completion_tokens, total_tokens = _usage_counts_from_payload(usage)
            reasoning_tokens, reasoning_available = _reasoning_tokens_from_payload(usage)
            meta["usage"] = _usage_payload_from_counts(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                reasoning_tokens=reasoning_tokens if reasoning_available else None,
            )
        if seed is not None:
            response_seed = _extract_response_seed(data)
            meta["api_compat_seed_response"] = response_seed
            meta["api_compat_seed_acknowledged"] = response_seed == seed
        normalized = normalize_chat_completion_response(
            data,
            compat_config=compat_config,
            base_url=compat_base_url,
            request_meta=compat_request_meta,
        )
        meta.update(normalized.meta())
        try:
            require_final_content(normalized, base_url=compat_base_url)
        except OpenAICompatResponseError as exc:
            compat_error_meta: PatcherMeta = {
                **meta,
                **response_error_meta(exc, error=_OPENAI_COMPAT_REASONING_ONLY_ERROR),
            }
            if retry_attempts:
                compat_error_meta["transport_retry_attempts"] = retry_attempts
            return None, compat_error_meta
        if normalized.used_reasoning_fallback:
            _apply_reasoning_fallback_to_response_payload(data, normalized_text=normalized.text)
        reasoning = _get_first_choice_reasoning_content(data)
        if reasoning is not None:
            meta["reasoning_content_present"] = bool(reasoning.strip())
            meta["reasoning_content_chars"] = len(reasoning)
        reasoning_text = _get_first_choice_reasoning(data)
        if reasoning_text is not None:
            meta["reasoning_present"] = bool(reasoning_text.strip())
            meta["reasoning_chars"] = len(reasoning_text)
        return data if isinstance(data, dict) else None, meta

    return None, {
        **base_meta,
        "error": "request_failed",
        "exc": "RuntimeError: transport retry loop exhausted without terminal result",
        "llm_call_count": call_count,
        "transport_retry_count": len(retry_attempts),
        "transport_impl": "httpx",
        "transport_http2_requested": bool(http2),
        "transport_http2_effective": bool(http2),
        "transport_http2_fallback_used": False,
    }


async def _chat_completion_async(
    *,
    url: str,
    api_key: str,
    model: str,
    messages: list[dict[str, str]],
    temperature: float,
    max_tokens: int,
    compat_profile: str = "auto",
    allow_reasoning_fallback: bool = False,
    enable_thinking: bool = False,
    timeout_sec: float,
    seed: int | None,
    http2: bool,
    extra_payload: dict[str, Any] | None = None,
    deadline_monotonic: float | None = None,
    transport_max_retries: int = _DEFAULT_CHAT_COMPLETION_TRANSPORT_MAX_RETRIES,
    transport_retry_backoff_base_sec: float = _DEFAULT_CHAT_COMPLETION_RETRY_BACKOFF_BASE_SEC,
    transport_retry_backoff_max_sec: float = _DEFAULT_CHAT_COMPLETION_RETRY_BACKOFF_MAX_SEC,
) -> tuple[dict[str, Any] | None, PatcherMeta]:
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if seed is not None:
        payload["seed"] = seed
    if extra_payload:
        payload.update(extra_payload)
    compat_base_url = _base_url_from_chat_url(url)
    payload, compat_request_meta, compat_config = _apply_eval_openai_compat_request_profile(
        payload=payload,
        compat_profile=compat_profile,
        allow_reasoning_fallback=allow_reasoning_fallback,
        enable_thinking=enable_thinking,
        base_url=compat_base_url,
    )

    max_retries = max(0, int(transport_max_retries))
    retry_backoff_base_sec = max(0.0, float(transport_retry_backoff_base_sec))
    retry_backoff_max_sec = max(0.0, float(transport_retry_backoff_max_sec))
    if retry_backoff_max_sec < retry_backoff_base_sec:
        retry_backoff_max_sec = retry_backoff_base_sec
    retry_attempts: list[dict[str, Any]] = []
    call_count = 0

    base_meta: PatcherMeta = {
        "model": model,
        "base_url": compat_base_url,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "timeout_sec": timeout_sec,
        "effective_timeout_sec": timeout_sec,
        "remaining_budget_sec": _remaining_request_budget_sec(deadline_monotonic),
        "llm_call_count": 0,
        "usage": _usage_payload_from_counts(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        **compat_request_meta.to_dict(),
        "transport_retry_limit": max_retries,
        "transport_retry_limit_effective": max_retries,
        "transport_retry_backoff_base_sec": retry_backoff_base_sec,
        "transport_retry_backoff_max_sec": retry_backoff_max_sec,
        "api_compat_seed_requested": seed,
        "api_compat_seed_acknowledged": False if seed is not None else None,
        "api_compat_seed_response": None,
    }

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")

    async def _send_request(*, timeout: float, use_http2: bool) -> tuple[int, str, str]:
        if _should_use_proxy_transport(url):
            async with httpx.AsyncClient(
                timeout=timeout, http2=use_http2, trust_env=True
            ) as client:
                response = await client.post(url, content=payload_bytes, headers=headers)
                http_version = str(getattr(response, "http_version", "") or "")
                return int(response.status_code), response.text, http_version
        client = _get_async_http_client(url=url, http2=use_http2)
        status, raw = await client.post_json(
            target=_chat_http_target(url),
            body=payload_bytes,
            headers=headers,
            timeout_sec=timeout,
        )
        http_version = str(getattr(client, "last_http_version", "") or "")
        return status, raw, http_version

    for attempt_index in range(max_retries + 1):
        remaining_budget_sec = _remaining_request_budget_sec(deadline_monotonic)
        effective_timeout_sec = timeout_sec
        if remaining_budget_sec is not None:
            effective_timeout_sec = min(timeout_sec, remaining_budget_sec)

        if effective_timeout_sec <= 0:
            _emit_openai_transport_progress(
                "openai_transport",
                note="request budget exhausted",
                transport_state="request_budget_exhausted",
                transport_attempt=call_count,
                transport_retry_count=len(retry_attempts),
            )
            budget_meta: PatcherMeta = {
                **base_meta,
                "error": "request_budget_exceeded",
                "llm_call_count": call_count,
                "effective_timeout_sec": effective_timeout_sec,
                "remaining_budget_sec": remaining_budget_sec,
                "transport_retry_count": len(retry_attempts),
            }
            if retry_attempts:
                budget_meta["transport_retry_attempts"] = retry_attempts
            return None, budget_meta

        call_count += 1
        transport_http2_effective = bool(http2)
        transport_http2_fallback_used = False
        _emit_openai_transport_progress(
            "openai_transport",
            note=f"request {call_count} started",
            transport_state="request_started",
            transport_attempt=call_count,
            transport_retry_count=len(retry_attempts),
            effective_timeout_sec=effective_timeout_sec,
            remaining_budget_sec=remaining_budget_sec,
            transport_http2_effective=transport_http2_effective,
        )
        attempt_meta: PatcherMeta = {
            **base_meta,
            "llm_call_count": call_count,
            "effective_timeout_sec": effective_timeout_sec,
            "remaining_budget_sec": remaining_budget_sec,
            "transport_impl": "httpx",
            "transport_http2_requested": bool(http2),
            "transport_http2_effective": bool(transport_http2_effective),
            "transport_http2_fallback_used": bool(transport_http2_fallback_used),
        }
        raw_text = ""
        status = 0
        http_version = ""
        error_meta: PatcherMeta | None = None
        try:
            status, raw_text, http_version = await _send_request(
                timeout=effective_timeout_sec,
                use_http2=transport_http2_effective,
            )
        except Exception as exc:
            if transport_http2_effective and _is_http2_protocol_error(exc):
                transport_http2_effective = False
                transport_http2_fallback_used = True
                _emit_openai_transport_progress(
                    "openai_transport",
                    note="HTTP/2 protocol error; retrying over HTTP/1.1",
                    transport_state="http2_fallback",
                    transport_attempt=call_count,
                    transport_retry_count=len(retry_attempts),
                )
                attempt_meta["transport_http2_effective"] = False
                attempt_meta["transport_http2_fallback_used"] = True
                try:
                    status, raw_text, http_version = await _send_request(
                        timeout=effective_timeout_sec,
                        use_http2=False,
                    )
                except ValueError as inner_exc:
                    error_meta = {
                        **attempt_meta,
                        "error": "bad_request_url",
                        "exc": f"{type(inner_exc).__name__}: {inner_exc}",
                    }
                except Exception as inner_exc:
                    error_meta = {
                        **attempt_meta,
                        "error": "request_failed",
                        "exc": f"{type(inner_exc).__name__}: {inner_exc}",
                    }
            elif isinstance(exc, ValueError):
                error_meta = {
                    **attempt_meta,
                    "error": "bad_request_url",
                    "exc": f"{type(exc).__name__}: {exc}",
                }
            else:
                error_meta = {
                    **attempt_meta,
                    "error": "request_failed",
                    "exc": f"{type(exc).__name__}: {exc}",
                }
        if http_version:
            attempt_meta["transport_http_version"] = http_version
        if error_meta is None and status >= 400:
            error_meta = {
                **attempt_meta,
                "error": "http_error",
                "status": int(status),
                "body": raw_text[:2000],
            }

        if error_meta is not None:
            should_retry = attempt_index < max_retries and _is_retryable_chat_completion_error(
                error_meta
            )
            if should_retry:
                backoff_sec = _chat_completion_retry_backoff_sec(
                    attempt_index,
                    base_sec=retry_backoff_base_sec,
                    max_sec=retry_backoff_max_sec,
                )
                remaining_retry_budget_sec = _remaining_request_budget_sec(deadline_monotonic)
                if remaining_retry_budget_sec is not None:
                    backoff_sec = min(backoff_sec, max(0.0, remaining_retry_budget_sec - 1e-3))
                retry_meta: dict[str, Any] = {
                    "attempt": call_count,
                    "error": str(error_meta.get("error", "")),
                    "backoff_sec": backoff_sec,
                }
                status_code = error_meta.get("status")
                if isinstance(status_code, int):
                    retry_meta["status"] = status_code
                retry_attempts.append(retry_meta)
                retry_count = len(retry_attempts)
                status_suffix = f" status={status_code}" if isinstance(status_code, int) else ""
                _emit_openai_transport_progress(
                    "openai_transport",
                    note=(
                        f"retry {retry_count}/{max_retries} scheduled for "
                        f"{error_meta.get('error', 'request_failed')}{status_suffix}"
                    ),
                    transport_state="retry_scheduled",
                    transport_attempt=call_count,
                    next_transport_attempt=call_count + 1,
                    transport_retry_count=retry_count,
                    error=str(error_meta.get("error", "")),
                    status=status_code if isinstance(status_code, int) else None,
                    backoff_sec=backoff_sec,
                )
                if backoff_sec > 0:
                    _emit_openai_transport_progress(
                        "openai_transport",
                        note=f"backoff {backoff_sec:.1f}s before retry {call_count + 1}",
                        transport_state="backoff_sleep",
                        transport_attempt=call_count,
                        next_transport_attempt=call_count + 1,
                        transport_retry_count=retry_count,
                        backoff_sec=backoff_sec,
                    )
                    await asyncio.sleep(backoff_sec)
                continue

            final_error_meta: PatcherMeta = {
                **error_meta,
                "transport_retry_count": len(retry_attempts),
            }
            if retry_attempts:
                final_error_meta["transport_retry_attempts"] = retry_attempts
            return None, final_error_meta

        try:
            data = json.loads(raw_text)
        except Exception:
            bad_json_meta: PatcherMeta = {
                **attempt_meta,
                "error": "bad_json_response",
                "body": raw_text[:2000],
                "transport_retry_count": len(retry_attempts),
            }
            if retry_attempts:
                bad_json_meta["transport_retry_attempts"] = retry_attempts
            return None, bad_json_meta

        meta: PatcherMeta = {
            **attempt_meta,
            "transport_retry_count": len(retry_attempts),
        }
        if retry_attempts:
            meta["transport_retry_attempts"] = retry_attempts
        usage = data.get("usage")
        if isinstance(usage, dict):
            prompt_tokens, completion_tokens, total_tokens = _usage_counts_from_payload(usage)
            reasoning_tokens, reasoning_available = _reasoning_tokens_from_payload(usage)
            meta["usage"] = _usage_payload_from_counts(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                reasoning_tokens=reasoning_tokens if reasoning_available else None,
            )
        if seed is not None:
            response_seed = _extract_response_seed(data)
            meta["api_compat_seed_response"] = response_seed
            meta["api_compat_seed_acknowledged"] = response_seed == seed
        normalized = normalize_chat_completion_response(
            data,
            compat_config=compat_config,
            base_url=compat_base_url,
            request_meta=compat_request_meta,
        )
        meta.update(normalized.meta())
        try:
            require_final_content(normalized, base_url=compat_base_url)
        except OpenAICompatResponseError as exc:
            compat_error_meta: PatcherMeta = {
                **meta,
                **response_error_meta(exc, error=_OPENAI_COMPAT_REASONING_ONLY_ERROR),
            }
            if retry_attempts:
                compat_error_meta["transport_retry_attempts"] = retry_attempts
            return None, compat_error_meta
        if normalized.used_reasoning_fallback:
            _apply_reasoning_fallback_to_response_payload(data, normalized_text=normalized.text)
        reasoning = _get_first_choice_reasoning_content(data)
        if reasoning is not None:
            meta["reasoning_content_present"] = bool(reasoning.strip())
            meta["reasoning_content_chars"] = len(reasoning)
        reasoning_text = _get_first_choice_reasoning(data)
        if reasoning_text is not None:
            meta["reasoning_present"] = bool(reasoning_text.strip())
            meta["reasoning_chars"] = len(reasoning_text)
        return data if isinstance(data, dict) else None, meta

    return None, {
        **base_meta,
        "error": "request_failed",
        "exc": "RuntimeError: transport retry loop exhausted without terminal result",
        "llm_call_count": call_count,
        "transport_retry_count": len(retry_attempts),
        "transport_impl": "httpx",
        "transport_http2_requested": bool(http2),
        "transport_http2_effective": bool(http2),
        "transport_http2_fallback_used": False,
    }


def _get_first_choice_content(resp: dict[str, Any]) -> str | None:
    choices = resp.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    c0 = choices[0]
    if not isinstance(c0, dict):
        return None
    msg = c0.get("message")
    if isinstance(msg, dict):
        return _coerce_content_to_text(msg.get("content"))
    # Some servers return `text` (completions-like).
    text = c0.get("text")
    return text if isinstance(text, str) else None


def _get_first_choice_finish_reason(resp: dict[str, Any]) -> str:
    choices = resp.get("choices")
    if not isinstance(choices, list) or not choices:
        return ""
    c0 = choices[0]
    if not isinstance(c0, dict):
        return ""
    finish_reason = c0.get("finish_reason")
    return finish_reason if isinstance(finish_reason, str) else ""


def _get_first_choice_reasoning_content(resp: dict[str, Any]) -> str | None:
    choices = resp.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    c0 = choices[0]
    if not isinstance(c0, dict):
        return None
    msg = c0.get("message")
    if not isinstance(msg, dict):
        return None
    reasoning_content = msg.get("reasoning_content")
    return reasoning_content if isinstance(reasoning_content, str) else None


def _get_first_choice_reasoning(resp: dict[str, Any]) -> str | None:
    choices = resp.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    c0 = choices[0]
    if not isinstance(c0, dict):
        return None
    msg = c0.get("message")
    if not isinstance(msg, dict):
        return None
    reasoning = msg.get("reasoning")
    return reasoning if isinstance(reasoning, str) else None


def _coerce_content_to_text(content: Any) -> str | None:
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return None
    direct_chunks: list[str] = []
    fallback_chunks: list[str] = []
    for part in content:
        if isinstance(part, str):
            fallback_chunks.append(part)
            continue
        if not isinstance(part, dict):
            continue

        text = _extract_content_part_text(part)
        if not text:
            continue

        part_type_raw = part.get("type")
        part_type = str(part_type_raw).strip().lower() if isinstance(part_type_raw, str) else ""
        if part_type in {
            "reasoning",
            "reasoning_text",
            "thinking",
            "analysis",
            "thought",
            "thought_text",
            "summary",
            "summary_text",
        }:
            continue

        if not part_type or part_type in {"text", "output_text", "input_text", "message"}:
            direct_chunks.append(text)
        else:
            # Unknown non-reasoning type: keep as fallback for compatibility.
            fallback_chunks.append(text)

    if direct_chunks:
        return "".join(direct_chunks)
    if fallback_chunks:
        return "".join(fallback_chunks)
    return None


def _extract_content_part_text(part: dict[str, Any]) -> str | None:
    for key in ("text", "content", "value"):
        value = part.get(key)
        if isinstance(value, str):
            return value
        if isinstance(value, dict):
            nested = value.get("value")
            if isinstance(nested, str):
                return nested
            nested = value.get("text")
            if isinstance(nested, str):
                return nested
    return None


def _remaining_request_budget_sec(deadline_monotonic: float | None) -> float | None:
    if deadline_monotonic is None:
        return None
    return max(0.0, deadline_monotonic - time.monotonic())


def _chat_completion_retry_backoff_sec(
    retry_index: int,
    *,
    base_sec: float = _DEFAULT_CHAT_COMPLETION_RETRY_BACKOFF_BASE_SEC,
    max_sec: float = _DEFAULT_CHAT_COMPLETION_RETRY_BACKOFF_MAX_SEC,
) -> float:
    safe_index = max(0, int(retry_index))
    base = max(0.0, float(base_sec))
    max_cap = max(base, float(max_sec))
    backoff = base * (2**safe_index)
    return min(max_cap, max(0.0, backoff))


def _is_retryable_chat_completion_http_status(status: int) -> bool:
    normalized = _nonnegative_int(status, default=0)
    if normalized >= 500:
        return True
    return normalized in _CHAT_COMPLETION_RETRYABLE_HTTP_STATUSES


def _is_retryable_chat_completion_error(meta: PatcherMeta) -> bool:
    error = str(meta.get("error", "")).strip()
    if error == "request_failed":
        return True
    if error != "http_error":
        return False
    status = _nonnegative_int(meta.get("status"), default=0)
    return _is_retryable_chat_completion_http_status(status)


def _is_backend_unavailable_error(meta: PatcherMeta) -> bool:
    error = str(meta.get("error", "")).strip()
    if error == "http_error":
        status = _nonnegative_int(meta.get("status"), default=0)
        return status in _BACKEND_UNAVAILABLE_HTTP_STATUSES
    if error != "request_failed":
        return False
    exc = str(meta.get("exc", "")).lower()
    markers = (
        "connectionrefusederror",
        "connection refused",
        "server disconnected",
        "temporarily unavailable",
        "service unavailable",
    )
    return any(marker in exc for marker in markers)


def _request_context_failure_kind(
    *,
    meta: PatcherMeta,
    request_json: ContextRequestPayload | None,
    raw_response: str | None,
) -> str:
    if request_json is not None:
        return ""
    error = str(meta.get("error", "")).strip()
    if error == "request_budget_exceeded":
        return "request_budget_exceeded"
    if error == _OPENAI_COMPAT_REASONING_ONLY_ERROR:
        return "compat_reasoning_only"
    if _is_backend_unavailable_error(meta):
        return "backend_unavailable"
    if _is_retryable_chat_completion_error(meta):
        return "transport"
    if raw_response is not None:
        return "invalid_payload"
    if error == "invalid_request_context_payload":
        return "invalid_payload"
    return ""


def _request_context_outcome(
    *,
    meta: PatcherMeta,
    request_json: ContextRequestPayload | None,
    raw_response: str | None,
) -> str:
    if request_json is not None:
        return "valid_payload"
    failure_kind = _request_context_failure_kind(
        meta=meta,
        request_json=request_json,
        raw_response=raw_response,
    )
    return {
        "transport": "transport_failed",
        "backend_unavailable": "unavailable_backend",
        "request_budget_exceeded": "request_budget_exceeded",
        "compat_reasoning_only": "compat_reasoning_only",
        "invalid_payload": "invalid_payload",
    }.get(failure_kind, "")


def _is_request_budget_exceeded(meta: PatcherMeta) -> bool:
    return str(meta.get("error", "")) == "request_budget_exceeded"


def _is_openai_compat_reasoning_only(meta: PatcherMeta | None) -> bool:
    if not isinstance(meta, dict):
        return False
    return str(meta.get("error", "")).strip() == _OPENAI_COMPAT_REASONING_ONLY_ERROR


def _forced_patch_skip_reason(meta: PatcherMeta) -> str:
    error = str(meta.get("error", "")).strip()
    if error in {"request_failed", "bad_request_url"}:
        return error
    if _is_request_budget_exceeded(meta):
        return "request_budget_exceeded"
    if error == _OPENAI_COMPAT_REASONING_ONLY_ERROR:
        return error
    if _is_context_exceeded_http_error(meta):
        return "context_limit"

    fallback_meta = meta.get("fallback_meta")
    if isinstance(fallback_meta, dict):
        nested = _forced_patch_skip_reason(cast(PatcherMeta, fallback_meta))
        if nested:
            return f"fallback_{nested}"
    return ""


def _forced_patch_skip_category(skip_reason: str) -> str:
    if skip_reason.endswith(_OPENAI_COMPAT_REASONING_ONLY_ERROR):
        return "compatibility_failure"
    return "transport_or_context_failure"


def _is_context_exceeded_http_error(meta: PatcherMeta) -> bool:
    error = str(meta.get("error", ""))
    if error != "http_error":
        return False
    body = str(meta.get("body", "")).lower()
    return any(
        marker in body
        for marker in (
            "exceeds the available context size",
            "context size",
            "greater than the context length",
            "tokens to keep from the initial prompt is greater than the context length",
            "maximum context length",
            "prompt is too long",
        )
    )


def _normalize_edits_payload(
    payload: dict[str, Any],
) -> tuple[list[dict[str, Any]] | None, str | None]:
    edits = payload.get("edits")
    if isinstance(edits, list) and edits:
        return edits, "edits_list"
    if isinstance(edits, dict) and _looks_like_edit_entry(edits):
        return [edits], "edits_object"

    if _looks_like_edit_entry(payload):
        return [payload], "single_edit_root"

    edit = payload.get("edit")
    if isinstance(edit, dict) and _looks_like_edit_entry(edit):
        return [edit], "edit_object"

    changes = payload.get("changes")
    if isinstance(changes, list) and changes:
        return changes, "changes_list"

    return None, None


def _looks_like_edit_entry(entry: dict[str, Any]) -> bool:
    has_path = "path" in entry
    has_find = "find" in entry or "old" in entry
    has_replace = "replace" in entry or "new" in entry
    return has_path and has_find and has_replace


def _apply_find_replace_with_recovery(
    *,
    before: str,
    find: str,
    replace: str,
    count: int,
) -> tuple[str | None, str]:
    replace_count = count or -1

    if find in before:
        after = before.replace(find, replace, replace_count)
        if after != before:
            return after, "exact"
        return before, "exact_noop"

    before_nl = _normalize_newlines(before)
    find_nl = _normalize_newlines(find)
    replace_nl = _normalize_newlines(replace)

    if not find_nl:
        return None, "none"

    if before_nl.count(find_nl) == 1:
        after_nl = before_nl.replace(find_nl, replace_nl, replace_count)
        if after_nl != before_nl:
            return after_nl, "normalized_newlines"
        return before, "normalized_newlines_noop"

    pattern = _compile_whitespace_normalized_pattern(find_nl)
    matches = list(pattern.finditer(before_nl))
    if len(matches) == 1:
        match = matches[0]
        after_nl = before_nl[: match.start()] + replace_nl + before_nl[match.end() :]
        if after_nl != before_nl:
            return after_nl, "normalized_whitespace"
        return before, "normalized_whitespace_noop"

    # Fuzzy recovery is intentionally restricted to single targeted replacements.
    if count == 1:
        after_fuzzy = _apply_fuzzy_block_replacement(
            before_nl=before_nl,
            find_nl=find_nl,
            replace_nl=replace_nl,
        )
        if after_fuzzy is not None:
            if after_fuzzy != before_nl:
                return after_fuzzy, "fuzzy_block"
            return before, "fuzzy_block_noop"

    return None, "none"


def _normalize_newlines(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _compile_whitespace_normalized_pattern(text: str) -> re.Pattern[str]:
    pieces: list[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch in {" ", "\t"}:
            while i < len(text) and text[i] in {" ", "\t"}:
                i += 1
            pieces.append(r"[ \t]+")
            continue
        if ch == "\n":
            # Ignore trailing horizontal whitespace before line breaks.
            pieces.append(r"[ \t]*\n")
            i += 1
            continue
        pieces.append(re.escape(ch))
        i += 1
    return re.compile("".join(pieces))


def _apply_fuzzy_block_replacement(
    *,
    before_nl: str,
    find_nl: str,
    replace_nl: str,
) -> str | None:
    find_block = find_nl.strip()
    if not find_block:
        return None
    # Keep fuzzy replacement conservative to avoid accidental rewrites.
    if len(find_block) < 16:
        return None

    line_chunks = before_nl.splitlines(keepends=True)
    if not line_chunks:
        return None
    if len(line_chunks) > 6000:
        return None

    target_lines = len(find_block.splitlines())
    if target_lines <= 0:
        return None

    candidate_window_lengths = {target_lines}
    if target_lines > 1:
        candidate_window_lengths.add(target_lines - 1)
        candidate_window_lengths.add(target_lines + 1)
    candidate_window_lengths = {
        size for size in candidate_window_lengths if 1 <= size <= len(line_chunks)
    }
    if not candidate_window_lengths:
        return None

    cumulative_lengths = [0]
    for chunk in line_chunks:
        cumulative_lengths.append(cumulative_lengths[-1] + len(chunk))

    scored_windows: list[tuple[float, int, int]] = []
    for window_len in sorted(candidate_window_lengths):
        max_start = len(line_chunks) - window_len
        for start in range(max_start + 1):
            raw_window = "".join(line_chunks[start : start + window_len]).strip()
            if not raw_window:
                continue
            ratio = difflib.SequenceMatcher(None, raw_window, find_block).ratio()
            scored_windows.append((ratio, start, window_len))

    if not scored_windows:
        return None

    scored_windows.sort(key=lambda item: item[0], reverse=True)
    best_ratio, best_start, best_len = scored_windows[0]
    second_ratio = scored_windows[1][0] if len(scored_windows) > 1 else 0.0

    min_ratio = 0.96 if target_lines == 1 else 0.92
    if best_ratio < min_ratio:
        return None
    if (best_ratio - second_ratio) < 0.03:
        return None

    start_offset = cumulative_lengths[best_start]
    end_offset = cumulative_lengths[best_start + best_len]
    return before_nl[:start_offset] + replace_nl + before_nl[end_offset:]


def _safe_trunc(x: object, limit: int = 2000) -> str:
    try:
        s = json.dumps(x, ensure_ascii=False, default=str)
    except Exception:
        s = repr(x)
    return s[:limit]


def _extract_response_seed(resp: dict[str, Any]) -> int | None:
    candidates: list[Any] = [
        resp.get("seed"),
        resp.get("response_seed"),
    ]
    metadata = resp.get("metadata")
    if isinstance(metadata, dict):
        candidates.append(metadata.get("seed"))
        candidates.append(metadata.get("response_seed"))
    for candidate in candidates:
        if isinstance(candidate, int):
            return candidate
        if isinstance(candidate, str):
            try:
                return int(candidate)
            except ValueError:
                continue
    return None


def _extract_json_object(text: str) -> dict[str, Any] | None:
    s = _strip_think(text).strip()
    if not s:
        return None

    fence = re.search(r"```(?:json)?\n(.*?)\n```", s, flags=re.DOTALL | re.IGNORECASE)
    if fence:
        s = fence.group(1).strip()

    try:
        data = json.loads(s)
        return data if isinstance(data, dict) else None
    except Exception:
        pass

    decoder = json.JSONDecoder()
    for i, ch in enumerate(s):
        if ch != "{":
            continue
        try:
            obj, _end = decoder.raw_decode(s[i:])
            return obj if isinstance(obj, dict) else None
        except Exception:
            continue
    return None


def _extract_first_fenced_code_block(text: str) -> tuple[str, bool]:
    without_think = _strip_think(text)
    if not without_think.strip():
        return "", False
    fence = re.search(r"```(?:[^\n`]*)\n(.*?)```", without_think, flags=re.DOTALL)
    if fence:
        return fence.group(1), True
    return without_think, False


def _looks_like_raw_unified_diff(text: str) -> bool:
    return _extract_unified_diff(text) is not None


def _normalize_full_file_response(raw: dict[str, Any]) -> tuple[str | None, PatcherMeta]:
    content = _get_first_choice_content(raw)
    if content is None:
        return None, {"error": "empty_full_file_response", "raw_response": _safe_trunc(raw)}

    repaired, code_fence_stripped = _extract_first_fenced_code_block(content)
    meta: PatcherMeta = {
        "full_file_code_fence_stripped": code_fence_stripped,
        "full_file_output_chars": len(repaired),
    }
    finish_reason = _get_first_choice_finish_reason(raw)
    if finish_reason:
        meta["finish_reason"] = finish_reason
    if finish_reason == "length":
        meta["error"] = "truncated_full_file_response"
        meta["raw_response"] = content[:2000]
        return None, meta
    if not repaired.strip():
        meta["error"] = "empty_full_file_response"
        meta["raw_response"] = content[:2000]
        return None, meta
    if _looks_like_raw_unified_diff(repaired):
        meta["error"] = "returned_diff_instead_of_file"
        meta["raw_response"] = repaired[:2000]
        return None, meta
    return repaired, meta


def _resolve_full_file_target(*, cwd: Path) -> tuple[str | None, str]:
    exclude_dirs = {".llmdebug", "__pycache__", ".venv", ".git"}
    candidates: list[str] = []
    for dirpath, dirnames, filenames in os.walk(cwd):
        dirnames[:] = sorted(name for name in dirnames if name not in exclude_dirs)
        dirpath_path = Path(dirpath)
        for filename in sorted(filenames):
            if not filename.endswith(".py"):
                continue
            path = dirpath_path / filename
            if not path.is_file():
                continue
            rel = str(path.relative_to(cwd))
            if _is_test_file_path(rel):
                continue
            candidates.append(rel)
    candidates.sort()
    if len(candidates) == 1:
        return candidates[0], "single_non_test_python_file"
    if not candidates:
        return None, "no_non_test_python_files"
    return None, "multiple_non_test_python_files"


def _build_diff_from_changes(changes: list[tuple[str, str, str]]) -> str:
    parts: list[str] = []
    for path, before, after in changes:
        udiff = difflib.unified_diff(
            before.splitlines(),
            after.splitlines(),
            fromfile=f"a/{path}",
            tofile=f"b/{path}",
            lineterm="",
        )
        file_lines = list(udiff)
        if not file_lines:
            continue
        parts.append(f"diff --git a/{path} b/{path}")
        parts.extend(file_lines)
        if parts and parts[-1] != "":
            parts.append("")
    return "\n".join(parts).rstrip("\n") + "\n"


def _build_forced_attempt_diff(*, prompt: str, cwd: Path) -> tuple[str, str] | None:
    """Build a deterministic minimal patch when model output is empty/invalid."""
    candidates = _extract_file_index_paths(prompt)
    if not candidates:
        return None

    target_rel: str | None = None
    for rel in candidates:
        if not rel.endswith(".py"):
            continue
        lower = rel.lower()
        name = Path(rel).name.lower()
        if name.startswith("test_") or "/test" in lower or "\\test" in lower:
            continue
        target_rel = rel
        break
    if target_rel is None:
        for rel in candidates:
            if rel.endswith(".py"):
                target_rel = rel
                break
    if target_rel is None:
        return None

    target_path = (cwd / target_rel).resolve()
    try:
        target_path.relative_to(cwd.resolve())
    except Exception:
        return None
    if not target_path.exists():
        return None

    before = target_path.read_text(encoding="utf-8")
    marker = "# llmdebug-force-attempt"
    if marker in before:
        marker = "# llmdebug-force-attempt-2"
    sep = "" if before.endswith("\n") else "\n"
    after = before + f"{sep}{marker}\n"
    if after == before:
        return None

    diff = _build_diff_from_changes([(target_rel, before, after)])
    return target_rel, diff


def _extract_file_index_paths(prompt: str) -> list[str]:
    lines = prompt.splitlines()
    start = None
    for i, line in enumerate(lines):
        if line.strip().startswith("Available files"):
            start = i + 1
            break
    if start is None:
        return []

    out: list[str] = []
    for line in lines[start:]:
        s = line.strip()
        if not s:
            if out:
                break
            continue
        if s.endswith(":"):
            break
        if s.startswith("Pytest output") or s.startswith("Project files:"):
            break
        out.append(s)
    return out


_HUNK_RE = re.compile(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@(.*)$")


def _fix_hunk_counts(diff: str) -> str:
    """Make unified diff hunks syntactically consistent.

    LLMs often emit correct hunk bodies but incorrect `@@ -l,s +l,s @@` counts.
    Git treats those diffs as corrupt. We recompute the counts from the hunk body
    and rewrite the header counts, keeping the start line numbers as-is.
    """
    lines = diff.splitlines()
    out: list[str] = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i]
        m = _HUNK_RE.match(line)
        if not m:
            out.append(line)
            i += 1
            continue

        old_start = max(1, int(m.group(1)))
        new_start = max(1, int(m.group(3)))
        suffix = m.group(5) or ""

        # Collect hunk body.
        body: list[str] = []
        i += 1
        while i < n:
            nxt = lines[i]
            if nxt.startswith("diff --git ") or nxt.startswith("@@ "):
                break
            if nxt.startswith("--- a/") or nxt.startswith("+++ b/"):
                break
            body.append(nxt)
            i += 1

        old_count = 0
        new_count = 0
        for b in body:
            if not b:
                continue
            c = b[0]
            if c == " ":
                old_count += 1
                new_count += 1
            elif c == "-":
                old_count += 1
            elif c == "+":
                new_count += 1
            else:
                # "\ No newline..." or other metadata does not affect counts.
                pass

        out.append(f"@@ -{old_start},{old_count} +{new_start},{new_count} @@{suffix}")
        out.extend(body)

    return "\n".join(out).rstrip("\n") + "\n"
