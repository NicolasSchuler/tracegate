"""Token, character and LLM-call accounting for attempt records."""

from __future__ import annotations

from typing import Any

from evals.coerce import nonnegative_int as _nonnegative_int
from evals.record_keys import KEY_PROMPT_CHARS_TOTAL


def _optional_nonnegative_int(value: Any) -> int | None:
    try:
        coerced = int(value)
    except (TypeError, ValueError):
        return None
    return max(0, coerced)


def _usage_token_counts_from_usage_payload(usage: Any) -> tuple[int, int, int]:
    if not isinstance(usage, dict):
        return 0, 0, 0

    prompt_tokens = _nonnegative_int(usage.get("prompt_tokens"), default=0)
    completion_tokens = _nonnegative_int(usage.get("completion_tokens"), default=0)
    # Anthropic-compatible payloads often use input/output token fields.
    if prompt_tokens == 0 and completion_tokens == 0:
        prompt_tokens = _nonnegative_int(usage.get("input_tokens"), default=0)
        completion_tokens = _nonnegative_int(usage.get("output_tokens"), default=0)

    total_tokens_raw = usage.get("total_tokens")
    if total_tokens_raw is None:
        total_tokens = prompt_tokens + completion_tokens
    else:
        total_tokens = _nonnegative_int(total_tokens_raw, default=prompt_tokens + completion_tokens)
    return prompt_tokens, completion_tokens, total_tokens


def _reasoning_token_counts_from_usage_payload(usage: Any) -> tuple[int, bool]:
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


def _token_usage_blocked_from_meta(meta: dict[str, Any] | None) -> bool:
    if not isinstance(meta, dict):
        return False
    explicit = meta.get("token_usage_available")
    if isinstance(explicit, bool):
        return not explicit
    usage_available = meta.get("usage_available")
    if isinstance(usage_available, bool):
        return not usage_available
    baseline_metrics = meta.get("baseline_metrics")
    if not isinstance(baseline_metrics, dict):
        return False
    baseline_usage_available = baseline_metrics.get("usage_available")
    return isinstance(baseline_usage_available, bool) and not baseline_usage_available


def _usage_token_counts_from_patch_meta(patch_meta: dict[str, Any] | None) -> tuple[int, int, int]:
    if not isinstance(patch_meta, dict):
        return 0, 0, 0
    if _token_usage_blocked_from_meta(patch_meta):
        return 0, 0, 0

    usage_prompt, usage_completion, usage_total = _usage_token_counts_from_usage_payload(
        patch_meta.get("usage")
    )
    baseline_metrics = _baseline_metrics_from_patch_meta(patch_meta)
    reported_prompt = 0
    reported_completion = 0
    reported_total = 0
    baseline_usage_available = False
    if isinstance(baseline_metrics, dict):
        baseline_usage_available = bool(baseline_metrics.get("usage_available"))
        reported_prompt = _nonnegative_int(
            baseline_metrics.get("reported_tokens_prompt"),
            default=0,
        )
        reported_completion = _nonnegative_int(
            baseline_metrics.get("reported_tokens_completion"),
            default=0,
        )
        reported_total = _nonnegative_int(
            baseline_metrics.get("reported_tokens_total"),
            default=reported_prompt + reported_completion,
        )
    if (
        baseline_usage_available
        and usage_prompt == 0
        and usage_completion == 0
        and usage_total == 0
        and (reported_prompt > 0 or reported_completion > 0 or reported_total > 0)
    ):
        usage_prompt, usage_completion, usage_total = (
            reported_prompt,
            reported_completion,
            reported_total,
        )
    prompt_tokens = _nonnegative_int(patch_meta.get("tokens_prompt"), default=usage_prompt)
    completion_tokens = _nonnegative_int(
        patch_meta.get("tokens_completion"),
        default=usage_completion,
    )
    total_tokens = _nonnegative_int(
        patch_meta.get("tokens_total"),
        default=usage_total if usage_total > 0 else (prompt_tokens + completion_tokens),
    )
    return prompt_tokens, completion_tokens, total_tokens


def _reasoning_token_counts_from_patch_meta(
    patch_meta: dict[str, Any] | None,
) -> tuple[int, bool]:
    if not isinstance(patch_meta, dict):
        return 0, False
    if _token_usage_blocked_from_meta(patch_meta):
        return 0, False

    explicit = patch_meta.get("reasoning_token_usage_available")
    usage_reasoning, usage_available = _reasoning_token_counts_from_usage_payload(
        patch_meta.get("usage")
    )
    baseline_reasoning = 0
    baseline_available = False
    baseline_metrics = _baseline_metrics_from_patch_meta(patch_meta)
    if isinstance(baseline_metrics, dict) and bool(baseline_metrics.get("usage_available")):
        reported = _optional_nonnegative_int(baseline_metrics.get("reported_tokens_reasoning"))
        if reported is not None:
            baseline_reasoning = reported
            baseline_available = True

    top_level_raw = patch_meta.get("tokens_reasoning")
    top_level_present = "tokens_reasoning" in patch_meta
    top_level_reasoning = _nonnegative_int(
        top_level_raw,
        default=usage_reasoning if usage_available else baseline_reasoning,
    )
    if isinstance(explicit, bool):
        return (top_level_reasoning if explicit else 0), explicit
    if top_level_present:
        return top_level_reasoning, True
    if usage_available:
        return usage_reasoning, True
    if baseline_available:
        return baseline_reasoning, True
    return 0, False


def _reasoning_usage_segment_from_meta(
    meta: dict[str, Any] | None,
    *,
    default_llm_calls: int = 0,
) -> tuple[int, bool, bool]:
    if not isinstance(meta, dict):
        if _nonnegative_int(default_llm_calls, default=0) > 0:
            return 0, True, False
        return 0, False, False
    llm_calls = _llm_call_count_from_meta(meta, default=default_llm_calls)
    usage_available = _token_usage_available_from_meta(meta)
    prompt_tokens, completion_tokens, total_tokens = _usage_token_counts_from_patch_meta(meta)
    applicable = bool(
        llm_calls > 0
        or usage_available
        or prompt_tokens > 0
        or completion_tokens > 0
        or total_tokens > 0
    )
    if not applicable:
        return 0, False, False
    reasoning_tokens, reasoning_available = _reasoning_token_counts_from_patch_meta(meta)
    return reasoning_tokens, True, reasoning_available


def _combine_reasoning_usage_segments(
    *segments: tuple[dict[str, Any] | None, int],
) -> tuple[int, bool]:
    total_reasoning_tokens = 0
    any_applicable = False
    full_coverage = True
    for meta, default_llm_calls in segments:
        reasoning_tokens, applicable, available = _reasoning_usage_segment_from_meta(
            meta,
            default_llm_calls=default_llm_calls,
        )
        if not applicable:
            continue
        any_applicable = True
        if available:
            total_reasoning_tokens += reasoning_tokens
        else:
            full_coverage = False
    return total_reasoning_tokens, bool(any_applicable and full_coverage)


def _completion_chars_from_response(
    *,
    diff: str | None = None,
    patch_meta: dict[str, Any] | None = None,
) -> int:
    if not isinstance(patch_meta, dict):
        return max(0, len(diff)) if isinstance(diff, str) else 0

    for key in (
        "completion_chars_total",
        "response_output_chars_total",
        "response_output_chars",
        "chars_output",
        "patch_diff_chars",
    ):
        value = patch_meta.get(key)
        if isinstance(value, int):
            return _nonnegative_int(value, default=0)

    if isinstance(diff, str):
        return max(0, len(diff))

    raw_response = patch_meta.get("raw_response")
    if isinstance(raw_response, str):
        return max(0, len(raw_response))
    response = patch_meta.get("response")
    if isinstance(response, str):
        return max(0, len(response))
    return 0


def _prompt_chars_from_patch_meta(patch_meta: dict[str, Any] | None) -> int:
    if not isinstance(patch_meta, dict):
        return 0
    for key in (
        KEY_PROMPT_CHARS_TOTAL,
        "response_input_chars_total",
        "response_input_chars",
        "chars_input",
    ):
        value = patch_meta.get(key)
        if isinstance(value, int):
            return _nonnegative_int(value, default=0)
    return 0


def _patch_attempt_char_deltas(
    *,
    cycle_prompt: str,
    diff: str | None,
    patch_meta: dict[str, Any] | None,
    is_baseline_patcher: bool,
) -> tuple[int, int]:
    if is_baseline_patcher:
        return (
            _prompt_chars_from_patch_meta(patch_meta),
            _completion_chars_from_response(diff=None, patch_meta=patch_meta),
        )
    return len(cycle_prompt), _completion_chars_from_response(diff=diff, patch_meta=patch_meta)


def _baseline_metrics_from_patch_meta(patch_meta: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(patch_meta, dict):
        return None
    baseline_metrics = patch_meta.get("baseline_metrics")
    return baseline_metrics if isinstance(baseline_metrics, dict) else None


def _baseline_metric_float(
    patch_meta: dict[str, Any] | None,
    key: str,
) -> float | None:
    baseline_metrics = _baseline_metrics_from_patch_meta(patch_meta)
    if baseline_metrics is None:
        return None
    value = baseline_metrics.get(key)
    if not isinstance(value, (int, float)):
        return None
    return max(0.0, float(value))


def _llm_call_count_from_meta(meta: dict[str, Any] | None, *, default: int = 0) -> int:
    if not isinstance(meta, dict):
        return max(0, int(default))
    value = meta.get("llm_call_count")
    if isinstance(value, int) and value >= 0:
        return value
    return max(0, int(default))


def _usage_available_flag_from_meta(meta: dict[str, Any] | None) -> bool | None:
    if not isinstance(meta, dict):
        return None
    usage_available = meta.get("usage_available")
    if isinstance(usage_available, bool):
        return usage_available
    baseline_metrics = _baseline_metrics_from_patch_meta(meta)
    if baseline_metrics is None:
        return None
    baseline_usage_available = baseline_metrics.get("usage_available")
    return baseline_usage_available if isinstance(baseline_usage_available, bool) else None


def _token_usage_available_from_meta(meta: dict[str, Any] | None) -> bool:
    if not isinstance(meta, dict):
        return False
    explicit = meta.get("token_usage_available")
    prompt_tokens, completion_tokens, total_tokens = _usage_token_counts_from_patch_meta(meta)
    has_counts = bool(prompt_tokens > 0 or completion_tokens > 0 or total_tokens > 0)
    baseline_metrics = _baseline_metrics_from_patch_meta(meta)
    reported_has_counts = False
    if isinstance(baseline_metrics, dict):
        reported_prompt = _nonnegative_int(
            baseline_metrics.get("reported_tokens_prompt"), default=0
        )
        reported_completion = _nonnegative_int(
            baseline_metrics.get("reported_tokens_completion"),
            default=0,
        )
        reported_total = _nonnegative_int(
            baseline_metrics.get("reported_tokens_total"),
            default=reported_prompt + reported_completion,
        )
        reported_has_counts = bool(
            reported_prompt > 0 or reported_completion > 0 or reported_total > 0
        )
    if isinstance(explicit, bool):
        if not explicit:
            return False
        return has_counts or reported_has_counts
    usage_available = _usage_available_flag_from_meta(meta)
    if isinstance(usage_available, bool):
        if not usage_available:
            return False
        return has_counts or reported_has_counts
    usage = meta.get("usage")
    if not isinstance(usage, dict):
        return False
    prompt_tokens, completion_tokens, total_tokens = _usage_token_counts_from_usage_payload(usage)
    return bool(prompt_tokens > 0 or completion_tokens > 0 or total_tokens > 0)


def _timeout_error_from_meta(meta: dict[str, Any] | None) -> bool:
    if not isinstance(meta, dict):
        return False
    error = meta.get("error")
    if isinstance(error, str) and error.strip().lower() == "timeout":
        return True
    baseline_meta = meta.get("baseline_meta")
    return _timeout_error_from_meta(baseline_meta) if isinstance(baseline_meta, dict) else False


def _record_token_usage_state(
    *,
    attempt_meta: dict[str, Any] | None,
    patch_meta: dict[str, Any] | None,
    prompt_tokens: int,
    total_tokens: int,
) -> str:
    if prompt_tokens > 0 or total_tokens > 0:
        return "available"
    llm_call_count = max(
        _llm_call_count_from_meta(attempt_meta), _llm_call_count_from_meta(patch_meta)
    )
    timeout_without_usage = _timeout_error_from_meta(attempt_meta) or _timeout_error_from_meta(
        patch_meta
    )
    for meta in (attempt_meta, patch_meta):
        if _usage_available_flag_from_meta(meta) is True:
            return "available"
    for meta in (attempt_meta, patch_meta):
        if _usage_available_flag_from_meta(meta) is False:
            return (
                "unavailable" if llm_call_count > 0 or timeout_without_usage else "not_applicable"
            )
    if llm_call_count > 0 or timeout_without_usage:
        return "unavailable"
    return "not_applicable"


def _format_progress_token_usage(
    *,
    prompt_tokens: int,
    completion_tokens: int,
    total_tokens: int,
    available_records: int,
    unavailable_records: int,
) -> str:
    if available_records <= 0:
        return "tokens=unavailable" if unavailable_records > 0 else "tokens=n/a"
    rendered = f"tokens=prompt:{prompt_tokens},completion:{completion_tokens},total:{total_tokens}"
    if unavailable_records > 0:
        rendered += ",coverage=partial"
    return rendered


def _aggregate_attempt_llm_usage(
    *,
    stage_a_meta: dict[str, Any] | None,
    patch_meta: dict[str, Any] | None,
    stage_a_default_llm_calls: int,
    patch_default_llm_calls: int,
) -> tuple[int, int, int, int]:
    """Sum LLM call counts and token usage across Stage-A and patch steps."""
    stage_calls = _llm_call_count_from_meta(stage_a_meta, default=stage_a_default_llm_calls)
    patch_calls = _llm_call_count_from_meta(patch_meta, default=patch_default_llm_calls)
    stage_prompt, stage_completion, stage_total = _usage_token_counts_from_patch_meta(stage_a_meta)
    patch_prompt, patch_completion, patch_total = _usage_token_counts_from_patch_meta(patch_meta)
    return (
        stage_calls + patch_calls,
        stage_prompt + patch_prompt,
        stage_completion + patch_completion,
        stage_total + patch_total,
    )


def _apply_usage_totals_to_patch_meta(
    *,
    patch_meta: dict[str, Any],
    llm_call_count: int,
    tokens_prompt: int,
    tokens_completion: int,
    tokens_total: int,
    tokens_reasoning: int | None = None,
    reasoning_token_usage_available: bool | None = None,
) -> None:
    normalized_calls = _nonnegative_int(llm_call_count, default=0)
    token_usage_blocked = _token_usage_blocked_from_meta(patch_meta)
    if token_usage_blocked:
        normalized_prompt = 0
        normalized_completion = 0
        normalized_total = 0
        normalized_reasoning = 0
        normalized_reasoning_available = False
    else:
        normalized_prompt = _nonnegative_int(tokens_prompt, default=0)
        normalized_completion = _nonnegative_int(tokens_completion, default=0)
        normalized_total = _nonnegative_int(
            tokens_total,
            default=normalized_prompt + normalized_completion,
        )
        if reasoning_token_usage_available is None:
            normalized_reasoning, normalized_reasoning_available = (
                _reasoning_token_counts_from_patch_meta(patch_meta)
            )
        else:
            normalized_reasoning = _nonnegative_int(tokens_reasoning, default=0)
            normalized_reasoning_available = bool(reasoning_token_usage_available)
        if not normalized_reasoning_available:
            normalized_reasoning = 0
    patch_meta["llm_call_count"] = normalized_calls
    patch_meta["tokens_prompt"] = normalized_prompt
    patch_meta["tokens_completion"] = normalized_completion
    patch_meta["tokens_total"] = normalized_total
    patch_meta["tokens_reasoning"] = normalized_reasoning
    patch_meta["reasoning_token_usage_available"] = normalized_reasoning_available
    if token_usage_blocked:
        patch_meta.pop("usage", None)
        return
    patch_meta["usage"] = _usage_payload_from_counts(
        prompt_tokens=normalized_prompt,
        completion_tokens=normalized_completion,
        total_tokens=normalized_total,
        reasoning_tokens=normalized_reasoning if normalized_reasoning_available else None,
    )


def _sync_attempt_usage_metadata(
    *,
    attempt_metadata: dict[str, Any],
    patch_meta: dict[str, Any],
    llm_call_count: int,
) -> None:
    token_usage_available = _token_usage_available_from_meta(patch_meta)
    if token_usage_available:
        prompt_tokens, completion_tokens, total_tokens = _usage_token_counts_from_patch_meta(
            patch_meta
        )
        reasoning_tokens, reasoning_token_usage_available = _reasoning_token_counts_from_patch_meta(
            patch_meta
        )
    else:
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        reasoning_tokens = 0
        reasoning_token_usage_available = False
    normalized_calls = _nonnegative_int(llm_call_count, default=0)
    attempt_metadata["llm_call_count"] = normalized_calls
    attempt_metadata["token_usage_available"] = token_usage_available
    attempt_metadata["tokens_prompt"] = prompt_tokens
    attempt_metadata["tokens_completion"] = completion_tokens
    attempt_metadata["tokens_total"] = total_tokens
    attempt_metadata["tokens_reasoning"] = reasoning_tokens
    attempt_metadata["reasoning_token_usage_available"] = reasoning_token_usage_available
    attempt_metadata["tokens_input"] = prompt_tokens
    attempt_metadata["tokens_output"] = completion_tokens
    patch_meta["llm_call_count"] = normalized_calls
    patch_meta["token_usage_available"] = token_usage_available
    patch_meta["tokens_prompt"] = prompt_tokens
    patch_meta["tokens_completion"] = completion_tokens
    patch_meta["tokens_total"] = total_tokens
    patch_meta["tokens_reasoning"] = reasoning_tokens
    patch_meta["reasoning_token_usage_available"] = reasoning_token_usage_available
    patch_meta["tokens_input"] = prompt_tokens
    patch_meta["tokens_output"] = completion_tokens


def _sync_attempt_prompt_char_metadata(
    *,
    attempt_metadata: dict[str, Any],
    patch_meta: dict[str, Any],
    prompt_chars_total: int,
    completion_chars_total: int = 0,
) -> None:
    normalized_prompt = _nonnegative_int(prompt_chars_total, default=0)
    normalized_completion = _nonnegative_int(completion_chars_total, default=0)
    normalized_total = normalized_prompt + normalized_completion
    attempt_metadata[KEY_PROMPT_CHARS_TOTAL] = normalized_prompt
    attempt_metadata["completion_chars_total"] = normalized_completion
    attempt_metadata["chars_input"] = normalized_prompt
    attempt_metadata["chars_output"] = normalized_completion
    attempt_metadata["chars_total"] = normalized_total
    patch_meta[KEY_PROMPT_CHARS_TOTAL] = normalized_prompt
    patch_meta["completion_chars_total"] = normalized_completion
    patch_meta["chars_input"] = normalized_prompt
    patch_meta["chars_output"] = normalized_completion
    patch_meta["chars_total"] = normalized_total


def _prompt_chars_for_llm_calls(prompt: str | None, *, llm_call_count: int) -> int:
    if not isinstance(prompt, str) or not prompt:
        return 0
    return len(prompt) * _nonnegative_int(llm_call_count, default=0)


def _clip_patcher_meta_text(value: Any, *, max_chars: int) -> str:
    text = str(value or "").strip()
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


def _summarize_context_compaction_meta(meta: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in (
        "llm_call_count",
        "compact_context_duration_sec",
        "compact_context_target_chars",
        "compact_context_input_chars",
        "compact_context_output_chars",
        "compact_context_output_truncated",
        "api_compat_seed_requested",
        "api_compat_seed_acknowledged",
        "api_compat_seed_response",
    ):
        if key in meta:
            out[key] = meta[key]
    status_value = meta.get("status")
    if isinstance(status_value, int):
        out["status"] = status_value
    usage = meta.get("usage")
    if isinstance(usage, dict):
        prompt_tokens, completion_tokens, total_tokens = _usage_token_counts_from_usage_payload(
            usage
        )
        reasoning_tokens, reasoning_available = _reasoning_token_counts_from_usage_payload(usage)
        out["usage"] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }
        if reasoning_available:
            out["usage"]["reasoning_tokens"] = reasoning_tokens
    for key in ("error", "exc", "reason", "fallback_reason"):
        text = _clip_patcher_meta_text(meta.get(key), max_chars=240)
        if text:
            out[key] = text
    body_text = _clip_patcher_meta_text(meta.get("body"), max_chars=320)
    if body_text:
        out["body"] = body_text
    return out


def _summarize_recovery_usage_meta(meta: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key in (
        "llm_call_count",
        "api_compat_seed_requested",
        "api_compat_seed_acknowledged",
        "api_compat_seed_response",
    ):
        if key in meta:
            out[key] = meta[key]
    status_value = meta.get("status")
    if isinstance(status_value, int):
        out["status"] = status_value
    if any(
        key in meta
        for key in (
            "usage",
            "tokens_prompt",
            "tokens_completion",
            "tokens_total",
            "tokens_reasoning",
        )
    ):
        prompt_tokens, completion_tokens, total_tokens = _usage_token_counts_from_patch_meta(meta)
        reasoning_tokens, reasoning_available = _reasoning_token_counts_from_patch_meta(meta)
        out["usage"] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": total_tokens,
        }
        if reasoning_available:
            out["usage"]["reasoning_tokens"] = reasoning_tokens
    for key in ("error", "exc", "reason", "fallback_reason"):
        text = _clip_patcher_meta_text(meta.get(key), max_chars=240)
        if text:
            out[key] = text
    body_text = _clip_patcher_meta_text(meta.get("body"), max_chars=320)
    if body_text:
        out["body"] = body_text
    return out
