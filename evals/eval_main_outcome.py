"""Per-unit progress metrics and run-level usage totals of an eval run.

Moved out of ``run_main_flow``, where the metrics were an 11-tuple (X-01).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from evals.attempt_usage import (
    _completion_chars_from_response,
    _nonnegative_int,
    _record_token_usage_state,
    _usage_token_counts_from_patch_meta,
)
from evals.record_keys import KEY_PROMPT_CHARS_TOTAL
from evals.transport_failures import _filter_attempt_records


@dataclass(frozen=True)
class UnitProgressMetrics:
    """Attempt, time and usage totals of one finished unit (one case, one condition)."""

    attempts_used: int
    max_attempts: int
    total_llm_sec: float
    total_test_sec: float
    total_prompt_chars: int
    total_completion_chars: int
    total_prompt_tokens: int
    total_completion_tokens: int
    total_tokens: int
    token_usage_available_records: int
    token_usage_unavailable_records: int


@dataclass
class RunUsageTotals:
    """Character and token totals over every unit of a run."""

    prompt_chars: int = 0
    completion_chars: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    usage_available_records: int = 0
    usage_unavailable_records: int = 0

    def add(self, metrics: UnitProgressMetrics) -> None:
        self.prompt_chars += metrics.total_prompt_chars
        self.completion_chars += metrics.total_completion_chars
        self.prompt_tokens += metrics.total_prompt_tokens
        self.completion_tokens += metrics.total_completion_tokens
        self.total_tokens += metrics.total_tokens
        self.usage_available_records += metrics.token_usage_available_records
        self.usage_unavailable_records += metrics.token_usage_unavailable_records


def extract_unit_progress_metrics(case_records: list[dict[str, Any]]) -> UnitProgressMetrics:
    """Return attempt/time/usage totals for a case.

    For test time: count the initial ``before_duration_sec`` once, then
    only ``after_duration_sec`` for each attempt (the next attempt's
    "before" is the previous attempt's "after", so this avoids
    double-counting).
    """
    scored_records = _filter_attempt_records(case_records)
    attempts_used = len(scored_records)
    max_attempts = 0
    if scored_records:
        meta = scored_records[0].get("attempt_metadata")
        if isinstance(meta, dict):
            max_attempts = int(meta.get("max_attempts", 0) or 0)

    total_llm_sec = 0.0
    total_test_sec = (
        float(scored_records[0].get("before_duration_sec", 0) or 0) if scored_records else 0.0
    )
    total_prompt_chars = 0
    total_completion_chars = 0
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_tokens = 0
    token_usage_available_records = 0
    token_usage_unavailable_records = 0
    for r in scored_records:
        pm = r.get("patch_meta")
        attempt_meta = r.get("attempt_metadata")
        if isinstance(pm, dict):
            total_llm_sec += float(pm.get("propose_duration_sec", 0) or 0)
        total_test_sec += float(r.get("after_duration_sec", 0) or 0)
        prompt_chars = (
            _nonnegative_int(
                attempt_meta.get(KEY_PROMPT_CHARS_TOTAL),
                default=_nonnegative_int(pm.get(KEY_PROMPT_CHARS_TOTAL), default=0)
                if isinstance(pm, dict)
                else 0,
            )
            if isinstance(attempt_meta, dict)
            else _nonnegative_int(pm.get(KEY_PROMPT_CHARS_TOTAL), default=0)
            if isinstance(pm, dict)
            else 0
        )
        completion_chars = (
            _nonnegative_int(
                attempt_meta.get("completion_chars_total"),
                default=_nonnegative_int(pm.get("completion_chars_total"), default=0)
                if isinstance(pm, dict)
                else _completion_chars_from_response(
                    patch_meta=pm if isinstance(pm, dict) else None
                ),
            )
            if isinstance(attempt_meta, dict)
            else _nonnegative_int(pm.get("completion_chars_total"), default=0)
            if isinstance(pm, dict)
            else 0
        )
        if isinstance(attempt_meta, dict):
            prompt_tokens = _nonnegative_int(
                attempt_meta.get("tokens_prompt"),
                default=_usage_token_counts_from_patch_meta(pm)[0] if isinstance(pm, dict) else 0,
            )
            completion_tokens = _nonnegative_int(
                attempt_meta.get("tokens_completion"),
                default=_usage_token_counts_from_patch_meta(pm)[1] if isinstance(pm, dict) else 0,
            )
            tokens_total = _nonnegative_int(
                attempt_meta.get("tokens_total"),
                default=_usage_token_counts_from_patch_meta(pm)[2] if isinstance(pm, dict) else 0,
            )
        else:
            prompt_tokens, completion_tokens, tokens_total = _usage_token_counts_from_patch_meta(pm)
        total_prompt_chars += prompt_chars
        total_completion_chars += completion_chars
        total_prompt_tokens += prompt_tokens
        total_completion_tokens += completion_tokens
        total_tokens += tokens_total
        token_usage_state = _record_token_usage_state(
            attempt_meta=attempt_meta if isinstance(attempt_meta, dict) else None,
            patch_meta=pm if isinstance(pm, dict) else None,
            prompt_tokens=prompt_tokens,
            total_tokens=tokens_total,
        )
        if token_usage_state == "available":
            token_usage_available_records += 1
        elif token_usage_state == "unavailable":
            token_usage_unavailable_records += 1

    return UnitProgressMetrics(
        attempts_used=attempts_used,
        max_attempts=max_attempts,
        total_llm_sec=total_llm_sec,
        total_test_sec=total_test_sec,
        total_prompt_chars=total_prompt_chars,
        total_completion_chars=total_completion_chars,
        total_prompt_tokens=total_prompt_tokens,
        total_completion_tokens=total_completion_tokens,
        total_tokens=total_tokens,
        token_usage_available_records=token_usage_available_records,
        token_usage_unavailable_records=token_usage_unavailable_records,
    )
