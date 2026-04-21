"""Shared OpenAI-compatible request/response compatibility helpers.

This module is intentionally stdlib-only so adapter-owned env shims and
patched upstream baseline repos can import it via ``PYTHONPATH`` without
depending on the rest of the llmdebug package runtime.
"""

from __future__ import annotations

import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

COMPAT_PROFILE_CHOICES: tuple[str, ...] = (
    "auto",
    "qwen_openai_raw",
    "qwen_dashscope",
    "lmstudio_model_yaml",
    "off",
)

_REASONING_PART_TYPES = {
    "analysis",
    "reasoning",
    "reasoning_text",
    "summary",
    "summary_text",
    "thinking",
    "thought",
    "thought_text",
}
_DIRECT_TEXT_PART_TYPES = {"", "input_text", "message", "output_text", "text"}
_THINK_RE = re.compile(r"<think>.*?</think>\n?", flags=re.DOTALL | re.IGNORECASE)
_THINK_CAPTURE_RE = re.compile(r"<think>(.*?)</think>", flags=re.DOTALL | re.IGNORECASE)
_REQUEST_PREFILL_THINKING_CONFLICT_RE = re.compile(
    r"assistant response prefill is incompatible with enable_thinking"
    r"|prefill.*enable_thinking"
    r"|enable_thinking.*prefill",
    flags=re.DOTALL | re.IGNORECASE,
)


@dataclass(frozen=True)
class BaselineOpenAICompatConfig:
    profile: str = "auto"
    allow_reasoning_fallback: bool = False
    enable_thinking: bool = False


@dataclass(frozen=True)
class OpenAICompatRequestMeta:
    compat_profile_requested: str
    compat_profile_applied: str
    direct_output_requested: bool
    allow_reasoning_fallback: bool
    enable_thinking: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "compat_profile_requested": self.compat_profile_requested,
            "compat_profile_applied": self.compat_profile_applied,
            "direct_output_requested": self.direct_output_requested,
            "allow_reasoning_fallback": self.allow_reasoning_fallback,
            "enable_thinking": self.enable_thinking,
        }


@dataclass(frozen=True)
class OpenAICompatRetryPlan:
    request_kwargs: dict[str, Any]
    request_meta: OpenAICompatRequestMeta
    reason: str


@dataclass(frozen=True)
class OpenAICompatResponse:
    text: str
    content_text: str
    reasoning: str
    reasoning_content: str
    content_present: bool
    reasoning_present: bool
    empty_content_with_reasoning: bool
    used_reasoning_fallback: bool
    finish_reason: str
    request_meta: OpenAICompatRequestMeta

    def meta(self) -> dict[str, Any]:
        return {
            **self.request_meta.to_dict(),
            "content_present": self.content_present,
            "reasoning_present": self.reasoning_present,
            "empty_content_with_reasoning": self.empty_content_with_reasoning,
            "used_reasoning_fallback": self.used_reasoning_fallback,
            "reasoning_chars": len(self.reasoning),
            "reasoning_content_chars": len(self.reasoning_content),
            "final_text_chars": len(self.text),
            "finish_reason": self.finish_reason,
        }


class OpenAICompatResponseError(RuntimeError):
    """Raised when a server returns reasoning but no final answer content."""

    def __init__(self, message: str, *, meta: dict[str, Any]):
        super().__init__(message)
        self.meta = dict(meta)


def compat_config_from_extra(extra: Mapping[str, Any]) -> BaselineOpenAICompatConfig:
    profile = normalize_compat_profile(extra.get("baseline_openai_compat_profile"))
    enable_thinking = bool(extra.get("baseline_openai_enable_thinking", False))
    allow_reasoning_fallback = bool(extra.get("baseline_openai_allow_reasoning_fallback", False))
    # When thinking is explicitly enabled, auto-enable reasoning fallback so
    # that thinking-model responses that contain only reasoning are usable.
    if enable_thinking and not allow_reasoning_fallback:
        allow_reasoning_fallback = True
    return BaselineOpenAICompatConfig(
        profile=profile,
        allow_reasoning_fallback=allow_reasoning_fallback,
        enable_thinking=enable_thinking,
    )


def normalize_compat_profile(value: Any) -> str:
    profile = str(value or "auto").strip().lower() or "auto"
    if profile not in COMPAT_PROFILE_CHOICES:
        return "auto"
    return profile


def apply_compat_request_profile(
    request_kwargs: Mapping[str, Any],
    *,
    compat_config: BaselineOpenAICompatConfig,
    base_url: str = "",
) -> tuple[dict[str, Any], OpenAICompatRequestMeta]:
    normalized = dict(request_kwargs)
    requested = normalize_compat_profile(compat_config.profile)
    applied = _resolve_compat_profile(requested, base_url=base_url)
    direct_output_requested = (
        applied in {"qwen_openai_raw", "qwen_dashscope"} and not compat_config.enable_thinking
    )

    if applied == "qwen_openai_raw":
        extra_body = _copy_mapping_like(normalized.get("extra_body"))
        chat_template_kwargs = _copy_mapping_like(extra_body.get("chat_template_kwargs"))
        chat_template_kwargs["enable_thinking"] = bool(compat_config.enable_thinking)
        extra_body["chat_template_kwargs"] = chat_template_kwargs
        normalized["extra_body"] = extra_body
    elif applied == "qwen_dashscope":
        normalized["enable_thinking"] = bool(compat_config.enable_thinking)

    return normalized, OpenAICompatRequestMeta(
        compat_profile_requested=requested,
        compat_profile_applied=applied,
        direct_output_requested=direct_output_requested,
        allow_reasoning_fallback=compat_config.allow_reasoning_fallback,
        enable_thinking=compat_config.enable_thinking,
    )


def normalize_chat_completion_response(
    response: Any,
    *,
    compat_config: BaselineOpenAICompatConfig,
    base_url: str = "",
    request_meta: OpenAICompatRequestMeta | None = None,
) -> OpenAICompatResponse:
    request_info = (
        request_meta
        or apply_compat_request_profile(
            {},
            compat_config=compat_config,
            base_url=base_url,
        )[1]
    )
    choices = _list_like(_get_value(response, "choices"))
    choice0 = choices[0] if choices else None
    message = _get_value(choice0, "message")

    content_text_raw = _coerce_content_to_text(_get_value(message, "content"))
    if not content_text_raw:
        alt_text = _get_value(choice0, "text")
        content_text_raw = alt_text if isinstance(alt_text, str) else ""
    content_text = _strip_think(content_text_raw).strip()
    think_reasoning = _extract_think_reasoning(content_text_raw)

    reasoning = _coerce_text(_get_value(message, "reasoning"))
    reasoning_content = _coerce_text(_get_value(message, "reasoning_content"))
    merged_reasoning = _merge_reasoning(reasoning, reasoning_content, think_reasoning)
    content_present = bool(content_text)
    reasoning_present = bool(merged_reasoning)
    empty_content_with_reasoning = bool(not content_present and reasoning_present)
    used_reasoning_fallback = bool(
        empty_content_with_reasoning and compat_config.allow_reasoning_fallback
    )
    text = merged_reasoning if used_reasoning_fallback else content_text

    finish_reason_raw = _get_value(choice0, "finish_reason")
    finish_reason = finish_reason_raw if isinstance(finish_reason_raw, str) else ""

    return OpenAICompatResponse(
        text=text,
        content_text=content_text,
        reasoning=reasoning,
        reasoning_content=reasoning_content,
        content_present=content_present,
        reasoning_present=reasoning_present,
        empty_content_with_reasoning=empty_content_with_reasoning,
        used_reasoning_fallback=used_reasoning_fallback,
        finish_reason=finish_reason,
        request_meta=request_info,
    )


def require_final_content(
    normalized: OpenAICompatResponse,
    *,
    base_url: str = "",
) -> str:
    if normalized.empty_content_with_reasoning and not normalized.used_reasoning_fallback:
        raise OpenAICompatResponseError(
            compatibility_error_message(normalized, base_url=base_url),
            meta=normalized.meta(),
        )
    return normalized.text


def compatibility_error_message(
    normalized: OpenAICompatResponse,
    *,
    base_url: str = "",
) -> str:
    profile = normalized.request_meta.compat_profile_applied
    details = [
        "OpenAI-compatible response contained reasoning but no final answer in message.content."
    ]
    if normalized.request_meta.direct_output_requested:
        details.append(
            "Direct-output mode was requested, but the server still omitted final content."
        )
    elif profile == "qwen_openai_raw":
        details.append(
            "For Qwen3/Qwen3.5 on vLLM or SGLang, set "
            "`baseline_openai_compat_profile=qwen_openai_raw` so requests send "
            "`chat_template_kwargs.enable_thinking=false`."
        )
    elif profile == "qwen_dashscope":
        details.append(
            "For DashScope-compatible Qwen endpoints, set "
            "`baseline_openai_compat_profile=qwen_dashscope` so requests send "
            "`enable_thinking=false`."
        )
    if profile == "lmstudio_model_yaml" or _is_probably_local_template_server(base_url):
        details.append(
            "For LM Studio / llama.cpp / model-template-driven local servers, disable "
            "thinking in the model or server template, or use a non-thinking variant."
        )
    return " ".join(details)


def response_error_meta(
    exc: OpenAICompatResponseError,
    *,
    error: str = "openai_compat_reasoning_only",
) -> dict[str, Any]:
    return {"error": error, **dict(exc.meta), "compat_message": str(exc)}


def request_validation_compat_error_from_exception(
    exc: BaseException,
    *,
    request_meta: OpenAICompatRequestMeta,
    base_url: str = "",
) -> OpenAICompatResponseError | None:
    """Classify known request-validation failures as compat errors.

    Args:
        exc: Original exception raised by the OpenAI-compatible client.
        request_meta: Metadata for the request that triggered the error.
        base_url: Endpoint URL used for the request.

    Returns:
        An ``OpenAICompatResponseError`` when the exception matches a known
        request-level compatibility failure, otherwise ``None``.
    """

    raw_error = _exception_text(exc)
    if not _REQUEST_PREFILL_THINKING_CONFLICT_RE.search(raw_error):
        return None

    return OpenAICompatResponseError(
        request_validation_compatibility_error_message(
            request_meta=request_meta,
            base_url=base_url,
            raw_error=raw_error,
        ),
        meta={
            **request_meta.to_dict(),
            "request_validation_error": True,
            "request_validation_failure_kind": "assistant_prefill_enable_thinking_conflict",
            "exception_type": type(exc).__name__,
            "raw_error": raw_error,
        },
    )


def build_request_validation_retry_plan(
    base_request_kwargs: Mapping[str, Any],
    *,
    compat_config: BaselineOpenAICompatConfig,
    request_meta: OpenAICompatRequestMeta,
    exc: BaseException,
    base_url: str = "",
) -> OpenAICompatRetryPlan | None:
    """Build a bounded retry plan for known OpenAI-compatible request mismatches."""

    compat_exc = request_validation_compat_error_from_exception(
        exc,
        request_meta=request_meta,
        base_url=base_url,
    )
    if compat_exc is None:
        return None

    retry_profile, reason = _request_validation_retry_profile(
        request_meta=request_meta,
        base_url=base_url,
    )
    if retry_profile is None:
        return None

    retry_config = BaselineOpenAICompatConfig(
        profile=retry_profile,
        allow_reasoning_fallback=compat_config.allow_reasoning_fallback,
        enable_thinking=False,
    )
    retry_kwargs, retry_meta = apply_compat_request_profile(
        base_request_kwargs,
        compat_config=retry_config,
        base_url=base_url,
    )
    if retry_kwargs == dict(base_request_kwargs) and retry_meta == request_meta:
        return None
    return OpenAICompatRetryPlan(
        request_kwargs=retry_kwargs,
        request_meta=retry_meta,
        reason=reason,
    )


def request_validation_compatibility_error_message(
    *,
    request_meta: OpenAICompatRequestMeta,
    base_url: str = "",
    raw_error: str = "",
) -> str:
    """Build actionable guidance for known request-validation compat failures."""

    profile = request_meta.compat_profile_applied
    details = [
        "OpenAI-compatible request was rejected before generation because assistant "
        "response prefill is incompatible with server-side thinking (`enable_thinking`)."
    ]
    if profile == "qwen_openai_raw":
        details.append(
            "Verify the endpoint accepts `chat_template_kwargs.enable_thinking=false`, "
            "or disable server-side thinking on the server."
        )
    elif profile == "qwen_dashscope":
        details.append(
            "For DashScope-compatible Qwen endpoints, set "
            "`baseline_openai_compat_profile=qwen_dashscope` so requests send "
            "`enable_thinking=false`."
        )
    else:
        details.append(
            "For Qwen3/Qwen3.5 on vLLM or SGLang-style OpenAI-compatible endpoints, set "
            "`baseline_openai_compat_profile=qwen_openai_raw` so requests send "
            "`chat_template_kwargs.enable_thinking=false`."
        )
    if profile == "lmstudio_model_yaml" or _is_probably_local_template_server(base_url):
        details.append(
            "For LM Studio / llama.cpp / model-template-driven local servers, disable "
            "thinking in the server or model template."
        )
    if raw_error:
        details.append(f"Server error: {raw_error}")
    return " ".join(details)


def _resolve_compat_profile(profile: str, *, base_url: str) -> str:
    if profile != "auto":
        return profile
    host = (urlparse(base_url).hostname or "").strip().lower()
    if host.endswith("dashscope.aliyuncs.com"):
        return "qwen_dashscope"
    return "auto"


def _request_validation_retry_profile(
    *,
    request_meta: OpenAICompatRequestMeta,
    base_url: str,
) -> tuple[str | None, str]:
    applied = request_meta.compat_profile_applied
    if applied == "qwen_openai_raw":
        if request_meta.enable_thinking:
            return "qwen_openai_raw", "retry_disable_thinking_qwen_openai_raw"
        return None, ""
    if applied == "qwen_dashscope":
        if request_meta.enable_thinking:
            return "qwen_dashscope", "retry_disable_thinking_qwen_dashscope"
        return None, ""
    if applied == "lmstudio_model_yaml":
        return None, ""
    if applied in {"auto", "off"}:
        inferred = _resolve_compat_profile("auto", base_url=base_url)
        profile = "qwen_dashscope" if inferred == "qwen_dashscope" else "qwen_openai_raw"
        return profile, f"retry_switch_profile_{profile}"
    return None, ""


def _is_probably_local_template_server(base_url: str) -> bool:
    host = (urlparse(base_url).hostname or "").strip().lower()
    return (
        host in {"127.0.0.1", "localhost"} or host.startswith("192.168.") or host.startswith("10.")
    )


def _copy_mapping_like(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return {str(key): val for key, val in value.items()}
    return {}


def _get_value(source: Any, key: str) -> Any:
    if isinstance(source, dict):
        return source.get(key)
    return getattr(source, key, None)


def _list_like(value: Any) -> list[Any]:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    return []


def _coerce_text(value: Any) -> str:
    return value.strip() if isinstance(value, str) else ""


def _exception_text(exc: BaseException) -> str:
    parts: list[str] = []
    for candidate in (
        str(exc),
        getattr(exc, "message", None),
        getattr(exc, "body", None),
        getattr(exc, "response", None),
    ):
        if isinstance(candidate, str):
            text = candidate.strip()
            if text and text not in parts:
                parts.append(text)
        elif isinstance(candidate, Mapping):
            text = str(dict(candidate)).strip()
            if text and text not in parts:
                parts.append(text)
    return " | ".join(parts)


def _merge_reasoning(reasoning: str, reasoning_content: str, think_reasoning: str = "") -> str:
    parts: list[str] = []
    for candidate in (reasoning_content, reasoning, think_reasoning):
        text = candidate.strip()
        if text and text not in parts:
            parts.append(text)
    return "\n".join(parts)


def _coerce_content_to_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    parts = _list_like(content)
    if not parts:
        return ""
    direct_chunks: list[str] = []
    fallback_chunks: list[str] = []
    for part in parts:
        if isinstance(part, str):
            fallback_chunks.append(part)
            continue
        text = _extract_part_text(part)
        if not text:
            continue
        part_type_raw = _get_value(part, "type")
        part_type = part_type_raw.strip().lower() if isinstance(part_type_raw, str) else ""
        if part_type in _REASONING_PART_TYPES:
            continue
        if part_type in _DIRECT_TEXT_PART_TYPES:
            direct_chunks.append(text)
        else:
            fallback_chunks.append(text)
    if direct_chunks:
        return "".join(direct_chunks)
    if fallback_chunks:
        return "".join(fallback_chunks)
    return ""


def _extract_part_text(part: Any) -> str:
    for key in ("text", "content", "output_text", "input_text"):
        value = _get_value(part, key)
        if isinstance(value, str) and value:
            return value
    return ""


def _strip_think(text: str) -> str:
    return _THINK_RE.sub("", text)


def _extract_think_reasoning(text: str) -> str:
    chunks: list[str] = []
    for match in _THINK_CAPTURE_RE.finditer(text):
        chunk = match.group(1).strip()
        if chunk:
            chunks.append(chunk)
    return "\n".join(chunks)
