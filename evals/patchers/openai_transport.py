"""HTTP transport for OpenAI-compatible backends: pooled clients, retries, compat profiles and chat completions."""

from __future__ import annotations

import asyncio
import functools
import math
import random
import threading
import time
import urllib.parse
import urllib.request
from collections.abc import Callable
from contextvars import ContextVar
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any

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
from evals import _json as json
from evals.eval_patch_constants import TRUNCATED_RESPONSE_ERROR
from evals.patchers.openai_response import (
    _coerce_content_to_text,
    _extract_response_seed,
    _get_first_choice_reasoning,
    _get_first_choice_reasoning_content,
    _is_truncated_response,
    _nonnegative_int,
    _reasoning_tokens_from_payload,
    _usage_counts_from_payload,
    _usage_payload_from_counts,
)
from evals.patchers.payload_types import PatcherMeta, copy_patcher_meta

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


_BACKEND_UNAVAILABLE_HTTP_STATUSES = {502, 503, 504}


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
    last_retry_after: str = ""

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
                self.last_retry_after = _retry_after_header(response)
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
    last_retry_after: str = ""

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
                self.last_retry_after = _retry_after_header(response)
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

    base_meta = copy_patcher_meta(
        {
            "model": model,
            "base_url": compat_base_url,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout_sec": timeout_sec,
            "effective_timeout_sec": timeout_sec,
            "remaining_budget_sec": _remaining_request_budget_sec(deadline_monotonic),
            "llm_call_count": 0,
            "usage": _usage_payload_from_counts(
                prompt_tokens=0, completion_tokens=0, total_tokens=0
            ),
            **compat_request_meta.to_dict(),
            "transport_retry_limit": max_retries,
            "transport_retry_limit_effective": max_retries,
            "transport_retry_backoff_base_sec": retry_backoff_base_sec,
            "transport_retry_backoff_max_sec": retry_backoff_max_sec,
            "api_compat_seed_requested": seed,
            "api_compat_seed_acknowledged": False if seed is not None else None,
            "api_compat_seed_response": None,
        }
    )

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")

    def _send_request(*, timeout: float, use_http2: bool) -> tuple[int, str, str, str]:
        if _should_use_proxy_transport(url):
            with httpx.Client(timeout=timeout, http2=use_http2, trust_env=True) as client:
                response = client.post(url, content=payload_bytes, headers=headers)
                http_version = str(getattr(response, "http_version", "") or "")
                return (
                    int(response.status_code),
                    response.text,
                    http_version,
                    _retry_after_header(response),
                )
        client = _get_thread_http_client(url=url, http2=use_http2)
        status, raw = client.post_json(
            target=_chat_http_target(url),
            body=payload_bytes,
            headers=headers,
            timeout_sec=timeout,
        )
        http_version = str(getattr(client, "last_http_version", "") or "")
        return status, raw, http_version, str(getattr(client, "last_retry_after", "") or "")

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
        retry_after = ""
        error_meta: PatcherMeta | None = None
        try:
            status, raw_text, http_version, retry_after = _send_request(
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
                    status, raw_text, http_version, retry_after = _send_request(
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
                **_retry_after_meta(retry_after),
            }
        data: Any = None
        if error_meta is None:
            try:
                data = json.loads(raw_text)
            except Exception:
                # A garbled body is a transport fault; retry it like one (LLM-19).
                error_meta = {
                    **attempt_meta,
                    "error": "bad_json_response",
                    "body": raw_text[:2000],
                }

        if error_meta is not None:
            should_retry = attempt_index < max_retries and _is_retryable_chat_completion_error(
                error_meta
            )
            backoff_sec = 0.0
            retry_delay_exceeds_deadline = False
            if should_retry:
                backoff_sec = _jittered_retry_delay_sec(
                    _chat_completion_retry_backoff_sec(
                        attempt_index,
                        base_sec=retry_backoff_base_sec,
                        max_sec=retry_backoff_max_sec,
                    ),
                    retry_after_sec=error_meta.get("retry_after_sec"),
                )
                remaining_retry_budget_sec = _remaining_request_budget_sec(deadline_monotonic)
                # Waiting less than the server asked for only buys another rejection:
                # if the required delay doesn't fit the deadline, stop here (LLM-22).
                retry_delay_exceeds_deadline = (
                    remaining_retry_budget_sec is not None
                    and backoff_sec >= remaining_retry_budget_sec
                )
                should_retry = not retry_delay_exceeds_deadline
            if should_retry:
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

            final_error_meta = copy_patcher_meta(error_meta)
            final_error_meta["transport_retry_count"] = len(retry_attempts)
            if retry_attempts:
                final_error_meta["transport_retry_attempts"] = retry_attempts
            if retry_delay_exceeds_deadline:
                # The deadline, not the server, ends the request here, so it keeps the
                # budget classification the trimmed-sleep-then-retry path produced
                # before, with the refused call's own error alongside it (LLM-22).
                final_error_meta["transport_retry_stop_reason"] = "retry_delay_exceeds_deadline"
                final_error_meta["transport_retry_delay_sec"] = backoff_sec
                final_error_meta["transport_error"] = str(error_meta.get("error") or "")
                final_error_meta["error"] = "request_budget_exceeded"
            return None, final_error_meta

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
        meta.update(copy_patcher_meta(normalized.meta()))
        try:
            require_final_content(normalized, base_url=compat_base_url)
        except OpenAICompatResponseError as exc:
            # A length-limited answer without usable final content is a truncation, not
            # a compatibility problem: it must not be retried or relabelled (LLM-02).
            rejection = (
                TRUNCATED_RESPONSE_ERROR
                if _is_truncated_response(data if isinstance(data, dict) else {})
                else _OPENAI_COMPAT_REASONING_ONLY_ERROR
            )
            compat_error_meta = copy_patcher_meta(
                {
                    **meta,
                    **response_error_meta(exc, error=rejection),
                }
            )
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

    base_meta = copy_patcher_meta(
        {
            "model": model,
            "base_url": compat_base_url,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "timeout_sec": timeout_sec,
            "effective_timeout_sec": timeout_sec,
            "remaining_budget_sec": _remaining_request_budget_sec(deadline_monotonic),
            "llm_call_count": 0,
            "usage": _usage_payload_from_counts(
                prompt_tokens=0, completion_tokens=0, total_tokens=0
            ),
            **compat_request_meta.to_dict(),
            "transport_retry_limit": max_retries,
            "transport_retry_limit_effective": max_retries,
            "transport_retry_backoff_base_sec": retry_backoff_base_sec,
            "transport_retry_backoff_max_sec": retry_backoff_max_sec,
            "api_compat_seed_requested": seed,
            "api_compat_seed_acknowledged": False if seed is not None else None,
            "api_compat_seed_response": None,
        }
    )

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    payload_bytes = json.dumps(payload, separators=(",", ":")).encode("utf-8")

    async def _send_request(*, timeout: float, use_http2: bool) -> tuple[int, str, str, str]:
        if _should_use_proxy_transport(url):
            async with httpx.AsyncClient(
                timeout=timeout, http2=use_http2, trust_env=True
            ) as client:
                response = await client.post(url, content=payload_bytes, headers=headers)
                http_version = str(getattr(response, "http_version", "") or "")
                return (
                    int(response.status_code),
                    response.text,
                    http_version,
                    _retry_after_header(response),
                )
        client = _get_async_http_client(url=url, http2=use_http2)
        status, raw = await client.post_json(
            target=_chat_http_target(url),
            body=payload_bytes,
            headers=headers,
            timeout_sec=timeout,
        )
        http_version = str(getattr(client, "last_http_version", "") or "")
        return status, raw, http_version, str(getattr(client, "last_retry_after", "") or "")

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
        retry_after = ""
        error_meta: PatcherMeta | None = None
        try:
            status, raw_text, http_version, retry_after = await _send_request(
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
                    status, raw_text, http_version, retry_after = await _send_request(
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
                **_retry_after_meta(retry_after),
            }
        data: Any = None
        if error_meta is None:
            try:
                data = json.loads(raw_text)
            except Exception:
                # A garbled body is a transport fault; retry it like one (LLM-19).
                error_meta = {
                    **attempt_meta,
                    "error": "bad_json_response",
                    "body": raw_text[:2000],
                }

        if error_meta is not None:
            should_retry = attempt_index < max_retries and _is_retryable_chat_completion_error(
                error_meta
            )
            backoff_sec = 0.0
            retry_delay_exceeds_deadline = False
            if should_retry:
                backoff_sec = _jittered_retry_delay_sec(
                    _chat_completion_retry_backoff_sec(
                        attempt_index,
                        base_sec=retry_backoff_base_sec,
                        max_sec=retry_backoff_max_sec,
                    ),
                    retry_after_sec=error_meta.get("retry_after_sec"),
                )
                remaining_retry_budget_sec = _remaining_request_budget_sec(deadline_monotonic)
                # Waiting less than the server asked for only buys another rejection:
                # if the required delay doesn't fit the deadline, stop here (LLM-22).
                retry_delay_exceeds_deadline = (
                    remaining_retry_budget_sec is not None
                    and backoff_sec >= remaining_retry_budget_sec
                )
                should_retry = not retry_delay_exceeds_deadline
            if should_retry:
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
                    await asyncio.sleep(backoff_sec)
                continue

            final_error_meta = copy_patcher_meta(error_meta)
            final_error_meta["transport_retry_count"] = len(retry_attempts)
            if retry_attempts:
                final_error_meta["transport_retry_attempts"] = retry_attempts
            if retry_delay_exceeds_deadline:
                # The deadline, not the server, ends the request here, so it keeps the
                # budget classification the trimmed-sleep-then-retry path produced
                # before, with the refused call's own error alongside it (LLM-22).
                final_error_meta["transport_retry_stop_reason"] = "retry_delay_exceeds_deadline"
                final_error_meta["transport_retry_delay_sec"] = backoff_sec
                final_error_meta["transport_error"] = str(error_meta.get("error") or "")
                final_error_meta["error"] = "request_budget_exceeded"
            return None, final_error_meta

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
        meta.update(copy_patcher_meta(normalized.meta()))
        try:
            require_final_content(normalized, base_url=compat_base_url)
        except OpenAICompatResponseError as exc:
            # A length-limited answer without usable final content is a truncation, not
            # a compatibility problem: it must not be retried or relabelled (LLM-02).
            rejection = (
                TRUNCATED_RESPONSE_ERROR
                if _is_truncated_response(data if isinstance(data, dict) else {})
                else _OPENAI_COMPAT_REASONING_ONLY_ERROR
            )
            compat_error_meta = copy_patcher_meta(
                {
                    **meta,
                    **response_error_meta(exc, error=rejection),
                }
            )
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


def _remaining_request_budget_sec(deadline_monotonic: float | None) -> float | None:
    if deadline_monotonic is None:
        return None
    return max(0.0, deadline_monotonic - time.monotonic())


def _retry_after_header(response: Any) -> str:
    headers = getattr(response, "headers", None)
    try:
        return str(headers.get("retry-after") or "") if headers is not None else ""
    except Exception:
        return ""


def _retry_after_meta(raw: str) -> PatcherMeta:
    """``retry_after_sec`` from a ``Retry-After`` header (delta-seconds or an HTTP-date)."""
    text = raw.strip()
    if not text:
        return {}
    try:
        seconds = float(text)
    except ValueError:
        try:
            when = parsedate_to_datetime(text)
        except (TypeError, ValueError):
            return {}
        if when.tzinfo is None:
            when = when.replace(tzinfo=timezone.utc)
        seconds = (when - datetime.now(timezone.utc)).total_seconds()
    if not math.isfinite(seconds):
        return {}
    return {"retry_after_sec": max(0.0, seconds)}


_RETRY_AFTER_MAX_SEC = 120.0
"""Sanity ceiling on an honoured ``Retry-After``, for runs without a deadline (LLM-22)."""


def _jittered_retry_delay_sec(
    backoff_sec: float,
    *,
    retry_after_sec: object,
    rng: random.Random | None = None,
) -> float:
    """The delay before a retry: jittered backoff, but at least the server's Retry-After.

    Equal jitter (half the backoff plus a random share of the other half) keeps parallel
    workers from retrying in lockstep after a shared outage. The exponential backoff is
    capped by its own ``max_sec``; the server's retry floor is not, since clipping a
    ``Retry-After: 60`` to an eight-second cap only buys another rejection (LLM-22).
    The caller compares the result with the remaining deadline and stops instead of
    retrying early; ``_RETRY_AFTER_MAX_SEC`` bounds a deadline-less run.
    """
    base = max(0.0, float(backoff_sec))
    delay = base / 2 + (rng or random).uniform(0.0, base / 2)
    if isinstance(retry_after_sec, (int, float)) and retry_after_sec > 0:
        delay = max(delay, min(float(retry_after_sec), _RETRY_AFTER_MAX_SEC))
    return delay


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
    if error in {"request_failed", "bad_json_response"}:
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
