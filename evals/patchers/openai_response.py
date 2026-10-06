"""Read OpenAI-compatible responses: content, reasoning, finish reasons, usage and embedded JSON."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any, cast

from evals import _json as json
from evals.patchers.openai_diff_text import _strip_think
from evals.patchers.payload_types import PatcherMeta, UsagePayload
from evals.stage_a_contract import coerce_stage_a_bool


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


def _is_truncated_response(resp: dict[str, Any]) -> bool:
    """Whether the completion stopped at ``max_tokens`` (LLM-02).

    A cut-off diff or JSON payload can still parse, and ``_fix_hunk_counts`` would
    turn a truncated hunk into a valid patch that drops lines, so every parse path
    treats truncation as a terminal ``truncated_response`` instead.
    """
    return _get_first_choice_finish_reason(resp) == "length"


def _error_meta(meta: Mapping[str, Any], reason: str, **extra: Any) -> PatcherMeta:
    """*meta* with ``error=reason``; an earlier call error is kept as ``transport_error``.

    Spreading *meta* after the reason let a stale transport error from an earlier
    stage replace the specific parse failure (LLM-14).
    """
    out: PatcherMeta = {**meta, **extra}
    stale = str(out.get("error") or "")
    if stale and stale != reason:
        out["transport_error"] = stale
    out["error"] = reason
    return out


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
