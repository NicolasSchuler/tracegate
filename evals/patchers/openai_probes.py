"""Probe an OpenAI-compatible server: readiness and the served context window."""

from __future__ import annotations

from typing import Any

import httpx

from evals import _json as json
from evals.patchers.openai_transport import (
    _chat_http_target,
    _get_async_http_client,
    _get_thread_http_client,
    _is_http2_protocol_error,
    _origin_from_base_url,
    _should_use_proxy_transport,
)
from evals.patchers.payload_types import PatcherMeta


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
    except ValueError as exc:
        return None, {"error": "bad_request_url", "exc": f"{type(exc).__name__}: {exc}"}
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
