"""Optional token-counting helpers for eval-only telemetry.

These helpers intentionally degrade to "estimate unavailable" when
``tiktoken`` is not installed or the tokenizer cannot be resolved.
"""

from __future__ import annotations

from typing import Any

_FALLBACK_ENCODINGS = ("o200k_base", "cl100k_base")


def estimate_text_token_count(text: str, *, model: str = "") -> dict[str, Any]:
    """Estimate token count for *text* using ``tiktoken`` when available."""
    normalized_text = str(text or "")
    normalized_model = str(model or "").strip()
    result: dict[str, Any] = {
        "available": False,
        "token_count": None,
        "model": normalized_model,
        "tokenizer": "",
        "resolution_mode": "",
        "error": "",
    }
    try:
        import tiktoken  # pyright: ignore[reportMissingImports]
    except ImportError:
        result["error"] = "tiktoken_unavailable"
        return result

    encoding: Any = None
    resolution_mode = ""
    resolution_error = ""
    if normalized_model:
        try:
            encoding = tiktoken.encoding_for_model(normalized_model)
            resolution_mode = "model"
        except Exception as exc:  # pragma: no cover - exact exception type is library-defined
            resolution_error = f"encoding_for_model_failed:{type(exc).__name__}"

    if encoding is None:
        for encoding_name in _FALLBACK_ENCODINGS:
            try:
                encoding = tiktoken.get_encoding(encoding_name)
                resolution_mode = f"fallback:{encoding_name}"
                break
            except Exception:  # pragma: no cover - exact exception type is library-defined
                continue

    if encoding is None:
        result["error"] = resolution_error or "tokenizer_resolution_failed"
        return result

    try:
        token_count = len(encoding.encode(normalized_text))
    except Exception as exc:  # pragma: no cover - exact exception type is library-defined
        result["tokenizer"] = str(getattr(encoding, "name", "") or "")
        result["resolution_mode"] = resolution_mode
        result["error"] = f"token_count_failed:{type(exc).__name__}"
        return result

    result["available"] = True
    result["token_count"] = int(token_count)
    result["tokenizer"] = str(getattr(encoding, "name", "") or "")
    result["resolution_mode"] = resolution_mode
    if resolution_error and resolution_mode.startswith("fallback:"):
        result["error"] = resolution_error
    return result


def merge_llm_call_usage(base: Any, extra: Any) -> dict[str, Any]:
    """Sum ``llm_call_count`` and the ``usage`` token counts of two LLM-call metas.

    Used where one logical step makes several calls (TEP equilibrium re-plans, ATT-09;
    transport-recovery replays, ATT-15) so none of them drops out of the totals.
    """
    merged = dict(base) if isinstance(base, dict) else {}
    if not isinstance(extra, dict):
        return merged
    base_calls = _merged_llm_call_count(merged)
    extra_calls = _merged_llm_call_count(extra)
    merged["llm_call_count"] = base_calls + extra_calls
    base_usage_raw = merged.get("usage")
    extra_usage_raw = extra.get("usage")
    base_usage = base_usage_raw if isinstance(base_usage_raw, dict) else {}
    extra_usage = extra_usage_raw if isinstance(extra_usage_raw, dict) else {}
    if base_usage or extra_usage:
        usage: dict[str, int] = {
            key: _usage_value(base_usage, key) + _usage_value(extra_usage, key)
            for key in ("prompt_tokens", "completion_tokens", "total_tokens")
        }
        reasoning_tokens, reasoning_covered = _merged_reasoning_tokens(
            (base_calls, base_usage), (extra_calls, extra_usage)
        )
        if reasoning_covered:
            usage["reasoning_tokens"] = reasoning_tokens
        merged["usage"] = usage
        if "tokens_reasoning" in merged or "reasoning_token_usage_available" in merged:
            # Keep the canonical top-level fields consistent with the merged payload.
            merged["tokens_reasoning"] = reasoning_tokens if reasoning_covered else 0
            merged["reasoning_token_usage_available"] = reasoning_covered
    return merged


def _merged_llm_call_count(meta: dict[str, Any]) -> int:
    """Served-call count of one meta, defaulting to one only when it is missing.

    A patcher that was rejected before it sent a request reports ``llm_call_count=0``;
    treating that as one would invent a call that nobody paid for.
    """
    if "llm_call_count" not in meta:
        return 1
    try:
        return max(0, int(meta["llm_call_count"]))
    except (TypeError, ValueError):
        return 1


def _merged_reasoning_tokens(
    *segments: tuple[int, dict[str, Any]],
) -> tuple[int, bool]:
    """Sum reasoning tokens, reporting coverage only when every served call has them.

    A segment that served no call and reported no usage is not applicable and cannot
    break coverage; this mirrors the patcher-side aggregation.
    """
    total = 0
    covered = True
    for calls, usage in segments:
        if not usage and calls <= 0:
            continue
        reasoning_tokens, available = _reasoning_tokens_from_usage(usage)
        if available:
            total += reasoning_tokens
        else:
            covered = False
    return total, covered


def _reasoning_tokens_from_usage(usage: dict[str, Any]) -> tuple[int, bool]:
    candidates = (
        usage,
        usage.get("output_tokens_details"),
        usage.get("completion_tokens_details"),
    )
    for candidate in candidates:
        if not isinstance(candidate, dict) or "reasoning_tokens" not in candidate:
            continue
        try:
            parsed = int(candidate["reasoning_tokens"])
        except (TypeError, ValueError):
            continue
        return max(0, parsed), True
    return 0, False


def _usage_value(usage: dict[str, Any], key: str) -> int:
    if key == "total_tokens" and not usage.get(key):
        return int(usage.get("prompt_tokens") or 0) + int(usage.get("completion_tokens") or 0)
    return int(usage.get(key) or 0)
