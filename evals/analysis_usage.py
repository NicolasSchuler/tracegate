"""Per-row usage measures: attempts, tokens, runtime, timing breakdowns and baseline metrics."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any, cast

from evals.analysis_numeric import _as_int, _mean
from evals.analysis_records import CURRENT_PATCH_DIFF_HASH_FIELD, LEGACY_PATCH_DIFF_HASH_FIELD
from evals.case_outcome import scored_attempts
from evals.eval_types import (
    CONDITION_FAMILIES,
    CONDITION_FAMILY_EXTERNAL_BASELINE,
    condition_family,
)

EXTERNAL_BASELINE_CONDITIONS = frozenset(
    name
    for name, family in CONDITION_FAMILIES.items()
    if family == CONDITION_FAMILY_EXTERNAL_BASELINE
)


def _looks_like_test_path(path: str) -> bool:
    """Heuristic check whether a file path looks like a test file."""
    normalized = path.replace("\\", "/").strip()
    if not normalized:
        return False
    parts = [part.lower() for part in normalized.split("/") if part]
    if not parts:
        return False
    file_name = parts[-1]
    if any(part in {"tests", "test"} for part in parts):
        return True
    if file_name == "conftest.py":
        return True
    if file_name.startswith("test_"):
        return True
    return bool(file_name.endswith("_test.py"))


def _patch_meta_flag_count(records: list[dict[str, Any]], key: str, value: Any) -> int:
    """Count attempts whose ``patch_meta[key]`` equals *value*."""
    return sum(
        1
        for record in records
        if isinstance(record.get("patch_meta"), dict) and record["patch_meta"].get(key) == value
    )


def _test_touch_attempt_metrics(records: list[dict[str, Any]]) -> tuple[int, int]:
    """Count patch attempts and how many touch test files.

    Returns:
        Tuple of (patch_attempts, test_touch_attempts).
    """
    patch_attempts = 0
    test_touch_attempts = 0
    for record in records:
        patch_meta = record.get("patch_meta")
        if not isinstance(patch_meta, dict):
            continue
        paths_raw = patch_meta.get("patch_file_paths")
        paths = [str(item) for item in paths_raw] if isinstance(paths_raw, list) else []
        file_count_raw = patch_meta.get("patch_file_count")
        file_count = int(file_count_raw) if isinstance(file_count_raw, (int, float)) else len(paths)
        patch_digest = str(
            patch_meta.get(CURRENT_PATCH_DIFF_HASH_FIELD)
            or patch_meta.get(LEGACY_PATCH_DIFF_HASH_FIELD)
            or ""
        ).strip()
        has_patch = file_count > 0 or bool(patch_digest)
        if not has_patch:
            continue
        patch_attempts += 1
        touches_tests_raw = patch_meta.get("patch_touches_tests")
        if isinstance(touches_tests_raw, bool):
            touches_tests = touches_tests_raw
        else:
            touches_tests = any(_looks_like_test_path(path) for path in paths)
        if touches_tests:
            test_touch_attempts += 1
    return patch_attempts, test_touch_attempts


def _mean_prompt_chars_by_condition(rows: list[dict[str, Any]]) -> dict[str, float | None]:
    """Group mean stage-B prompt character counts by condition."""
    grouped: dict[str, list[float]] = defaultdict(list)
    for row in rows:
        condition = str(row.get("condition") or "")
        value = row.get("stage_b_prompt_chars_mean")
        if not condition or not isinstance(value, (int, float)):
            continue
        grouped[condition].append(float(value))
    return {condition: _mean(values) for condition, values in sorted(grouped.items())}


def _attempts_to_success(records: list[dict[str, Any]]) -> int | None:
    """Return the first successful attempt number, or None."""
    for record in records:
        if bool(record.get("success")):
            return _as_int(record.get("attempt"))
    return None


def _dedupe_records_by_attempt(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """One scored row per attempt number, by the shared rule in ``evals.case_outcome``."""
    return [cast(dict[str, Any], record) for record in scored_attempts(records)]


def _attempts_executed(records: list[dict[str, Any]]) -> int:
    """Return the number of attempt records."""
    return len(records)


def _attempts_budget_from_records(records: list[dict[str, Any]]) -> int:
    """Determine the maximum attempt budget from records.

    Prefers ``max_attempts`` in attempt_metadata; falls back to the
    highest observed attempt number.
    """
    max_budget = 0
    for record in records:
        attempt_meta = record.get("attempt_metadata")
        if not isinstance(attempt_meta, dict):
            continue
        value = attempt_meta.get("max_attempts")
        if isinstance(value, int) and value > max_budget:
            max_budget = value
    if max_budget > 0:
        return max_budget
    return max((_as_int(record.get("attempt")) for record in records), default=0)


def _llm_calls_total(records: list[dict[str, Any]]) -> int:
    """Sum LLM call counts across all attempt records."""
    total = 0
    for record in records:
        attempt_meta = record.get("attempt_metadata")
        if not isinstance(attempt_meta, dict):
            continue
        value = attempt_meta.get("llm_call_count")
        if isinstance(value, int) and value > 0:
            total += value
    return total


@dataclass(frozen=True)
class TokenUsage:
    """Token counts summed over the attempts that reported them (STA-01, STA-09)."""

    prompt: float | None
    completion: float | None
    total: float | None
    covered_attempts: int


def _token_usage(records: list[dict[str, Any]]) -> TokenUsage:
    """Sum prompt, completion and total tokens over the attempts that reported them.

    An attempt with ``token_usage_available is False`` is not covered. When no attempt
    is covered the totals are ``None``; they used to be 0, so a baseline that timed out
    without usage looked cheapest and token coverage read 100% (STA-01). Falls back to
    legacy ``patch_meta.usage.total_tokens`` for records without structured counts.
    """
    prompt_total = 0.0
    completion_total = 0.0
    total_total = 0.0
    covered = 0
    found_attempt_meta = False
    for record in records:
        attempt_meta = record.get("attempt_metadata")
        if not isinstance(attempt_meta, dict):
            continue
        if attempt_meta.get("token_usage_available") is False:
            found_attempt_meta = True
            continue
        prompt_tokens = attempt_meta.get("tokens_prompt")
        completion_tokens = attempt_meta.get("tokens_completion")
        total_tokens = attempt_meta.get("tokens_total")
        if not isinstance(prompt_tokens, (int, float)):
            continue
        if not isinstance(completion_tokens, (int, float)):
            continue
        if not isinstance(total_tokens, (int, float)):
            continue
        found_attempt_meta = True
        covered += 1
        prompt_total += float(prompt_tokens)
        completion_total += float(completion_tokens)
        total_total += float(total_tokens)
    if covered:
        return TokenUsage(prompt_total, completion_total, total_total, covered)
    if found_attempt_meta:
        return TokenUsage(None, None, None, 0)

    # Backward-compatible fallback for older records.
    fallback_total = 0.0
    for record in records:
        patch_meta = record.get("patch_meta")
        if not isinstance(patch_meta, dict):
            continue
        usage = patch_meta.get("usage")
        if not isinstance(usage, dict):
            continue
        value = usage.get("total_tokens")
        if isinstance(value, (int, float)):
            covered += 1
            fallback_total += float(value)
    if not covered:
        return TokenUsage(None, None, None, 0)
    return TokenUsage(None, None, fallback_total, covered)


def _token_usage_breakdown(
    records: list[dict[str, Any]],
) -> tuple[float | None, float | None, float | None]:
    """Return (prompt, completion, total) from :func:`_token_usage`."""
    usage = _token_usage(records)
    return usage.prompt, usage.completion, usage.total


def _legacy_patch_meta_has_usage(patch_meta: dict[str, Any] | None) -> bool:
    if not isinstance(patch_meta, dict):
        return False
    usage = patch_meta.get("usage")
    if not isinstance(usage, dict):
        return False
    for key in (
        "prompt_tokens",
        "completion_tokens",
        "total_tokens",
        "input_tokens",
        "output_tokens",
    ):
        if isinstance(usage.get(key), (int, float)):
            return True
    return False


def _legacy_patch_meta_reasoning_tokens(
    patch_meta: dict[str, Any] | None,
) -> tuple[float | None, bool]:
    if not isinstance(patch_meta, dict):
        return None, False
    usage = patch_meta.get("usage")
    if not isinstance(usage, dict):
        return None, False
    candidates = [
        usage,
        usage.get("output_tokens_details"),
        usage.get("completion_tokens_details"),
    ]
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue
        value = candidate.get("reasoning_tokens")
        if isinstance(value, (int, float)):
            return float(value), True
    return None, False


def _reasoning_token_usage_breakdown(
    records: list[dict[str, Any]],
) -> tuple[float | None, bool, float | None]:
    """Return case-level reasoning tokens with covered-only semantics."""
    total_reasoning = 0.0
    applicable_attempts = 0
    found_attempt_meta = False
    for record in records:
        attempt_meta = record.get("attempt_metadata")
        patch_meta = record.get("patch_meta")
        if isinstance(attempt_meta, dict):
            found_attempt_meta = True
            llm_calls = attempt_meta.get("llm_call_count")
            token_usage_available = attempt_meta.get("token_usage_available")
            applicable = bool(
                (isinstance(llm_calls, int) and llm_calls > 0)
                or token_usage_available is True
                or _legacy_patch_meta_has_usage(
                    patch_meta if isinstance(patch_meta, dict) else None
                )
            )
            if not applicable:
                continue
            applicable_attempts += 1
            if attempt_meta.get("reasoning_token_usage_available") is True:
                value = attempt_meta.get("tokens_reasoning")
                if isinstance(value, (int, float)):
                    total_reasoning += float(value)
                    continue
            legacy_value, legacy_available = _legacy_patch_meta_reasoning_tokens(
                patch_meta if isinstance(patch_meta, dict) else None
            )
            if legacy_available and legacy_value is not None:
                total_reasoning += legacy_value
                continue
            return None, False, None
        patch_meta_dict = patch_meta if isinstance(patch_meta, dict) else None
        if not _legacy_patch_meta_has_usage(patch_meta_dict):
            continue
        applicable_attempts += 1
        legacy_value, legacy_available = _legacy_patch_meta_reasoning_tokens(patch_meta_dict)
        if legacy_available and legacy_value is not None:
            total_reasoning += legacy_value
            continue
        return None, False, None

    if applicable_attempts > 0:
        return total_reasoning, True, total_reasoning / float(applicable_attempts)
    if found_attempt_meta:
        return None, False, None

    fallback_total = 0.0
    fallback_found = False
    for record in records:
        patch_meta = record.get("patch_meta")
        legacy_value, legacy_available = _legacy_patch_meta_reasoning_tokens(
            patch_meta if isinstance(patch_meta, dict) else None
        )
        if not legacy_available or legacy_value is None:
            continue
        fallback_found = True
        fallback_total += legacy_value
    if not fallback_found:
        return None, False, None
    return fallback_total, True, fallback_total


def _retry_history_attempt_metrics(records: list[dict[str, Any]]) -> dict[str, int]:
    """Aggregate retry-history character, turn, and truncation metrics."""
    attempts = 0
    chars_sum = 0
    turns_sum = 0
    budget_applied_attempts = 0
    truncated_attempts = 0
    for record in records:
        attempt_meta = record.get("attempt_metadata")
        if not isinstance(attempt_meta, dict):
            continue
        attempts += 1
        chars_value = attempt_meta.get("retry_history_chars")
        if isinstance(chars_value, (int, float)):
            chars_sum += int(chars_value)
        turns_value = attempt_meta.get("retry_history_turns_included")
        if isinstance(turns_value, (int, float)):
            turns_sum += int(turns_value)
        if bool(attempt_meta.get("retry_history_budget_applied")):
            budget_applied_attempts += 1
        if bool(attempt_meta.get("retry_history_truncated")):
            truncated_attempts += 1
    return {
        "attempts": attempts,
        "chars_sum": chars_sum,
        "turns_sum": turns_sum,
        "budget_applied_attempts": budget_applied_attempts,
        "truncated_attempts": truncated_attempts,
    }


def _runtime_used(records: list[dict[str, Any]]) -> float | None:
    """Compute total wall-clock runtime from records.

    Tries ``attempt_wall_clock_sec`` first, then
    ``cumulative_runtime_sec``, then falls back to summing legacy
    ``before_duration_sec`` / ``after_duration_sec`` fields.
    """
    if not records:
        return None
    attempt_runtime_values = [
        float(record["attempt_wall_clock_sec"])
        for record in records
        if isinstance(record.get("attempt_wall_clock_sec"), (int, float))
    ]
    if len(attempt_runtime_values) == len(records):
        return sum(attempt_runtime_values)
    cumulative_values = [
        float(record["cumulative_runtime_sec"])
        for record in records
        if isinstance(record.get("cumulative_runtime_sec"), (int, float))
    ]
    if cumulative_values:
        return max(cumulative_values)
    total = 0.0
    found = False
    first_before = records[0].get("before_duration_sec")
    if isinstance(first_before, (int, float)):
        total += float(first_before)
        found = True
    for record in records:
        after = record.get("after_duration_sec")
        if isinstance(after, (int, float)):
            total += float(after)
            found = True
    return total if found else None


def _baseline_metrics_from_record(record: dict[str, Any]) -> dict[str, Any]:
    patch_meta = record.get("patch_meta")
    if not isinstance(patch_meta, dict):
        return {}
    baseline_metrics = patch_meta.get("baseline_metrics")
    return baseline_metrics if isinstance(baseline_metrics, dict) else {}


def _baseline_metric_last_bool(records: list[dict[str, Any]], key: str) -> bool | None:
    for record in reversed(records):
        value = _baseline_metrics_from_record(record).get(key)
        if isinstance(value, bool):
            return value
    return None


def _baseline_metric_last_str(records: list[dict[str, Any]], key: str) -> str | None:
    for record in reversed(records):
        value = _baseline_metrics_from_record(record).get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _baseline_metric_sum(records: list[dict[str, Any]], key: str) -> float | None:
    total = 0.0
    found = False
    for record in records:
        value = _baseline_metrics_from_record(record).get(key)
        if isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            total += float(value)
            found = True
    return total if found else None


def _baseline_metric_last_int(records: list[dict[str, Any]], key: str) -> int | None:
    for record in reversed(records):
        value = _baseline_metrics_from_record(record).get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _is_external_baseline_condition(condition: str) -> bool:
    return condition_family(condition) == CONDITION_FAMILY_EXTERNAL_BASELINE


_TIMING_BREAKDOWN_KEYS = (
    "stage_a_llm_sec",
    "stage_a_context_fulfillment_sec",
    "stage_a_total_sec",
    "snapshot_read_sec",
    "adaptive_controller_prepare_sec",
    "adaptive_controller_summarizer_sec",
    "adaptive_controller_sidecar_build_sec",
    "adaptive_controller_sidecar_write_sec",
    "baseline_case_conversion_sec",
    "baseline_sidecar_load_sec",
    "context_compaction_llm_sec",
    "context_compaction_total_sec",
    "prompt_assembly_sec",
    "stage_b_llm_sec",
    "baseline_internal_total_sec",
    "baseline_eval_sec",
    "baseline_orchestration_sec",
    "patch_apply_sec",
    "syntax_preflight_sec",
    "test_before_sec",
    "test_after_sec",
    "recovery_llm_sec",
    "recovery_total_sec",
    "llmdebug_total_sec",
    "llm_total_sec",
    "overhead_sec",
)


_TOKEN_BREAKDOWN_KEYS = (
    "stage_a_tokens_prompt",
    "stage_a_tokens_completion",
    "stage_a_tokens_total",
    "stage_b_tokens_prompt",
    "stage_b_tokens_completion",
    "stage_b_tokens_total",
    "compaction_tokens_prompt",
    "compaction_tokens_completion",
    "compaction_tokens_total",
    "recovery_tokens_prompt",
    "recovery_tokens_completion",
    "recovery_tokens_total",
    "tool_response_chars",
    "snapshot_chars",
)


def _timing_breakdown_mean(records: list[dict[str, Any]]) -> dict[str, float | None]:
    """Compute mean of each timing breakdown field across attempt records."""
    accumulators: dict[str, list[float]] = {key: [] for key in _TIMING_BREAKDOWN_KEYS}
    for record in records:
        attempt_meta = record.get("attempt_metadata")
        if not isinstance(attempt_meta, dict):
            continue
        timing = attempt_meta.get("timing_breakdown")
        if not isinstance(timing, dict):
            continue
        for key in _TIMING_BREAKDOWN_KEYS:
            value = timing.get(key)
            if isinstance(value, (int, float)):
                accumulators[key].append(float(value))
    return {key: _mean(values) for key, values in accumulators.items()}


def _token_breakdown_mean(records: list[dict[str, Any]]) -> dict[str, float | None]:
    """Compute mean of each token breakdown field across attempt records."""
    accumulators: dict[str, list[float]] = {key: [] for key in _TOKEN_BREAKDOWN_KEYS}
    for record in records:
        attempt_meta = record.get("attempt_metadata")
        if not isinstance(attempt_meta, dict):
            continue
        tokens = attempt_meta.get("token_breakdown")
        if not isinstance(tokens, dict):
            continue
        for key in _TOKEN_BREAKDOWN_KEYS:
            value = tokens.get(key)
            if isinstance(value, (int, float)):
                accumulators[key].append(float(value))
    return {key: _mean(values) for key, values in accumulators.items()}


def _timing_breakdown_sum(records: list[dict[str, Any]]) -> dict[str, float]:
    """Sum each timing breakdown field across attempt records."""
    totals: dict[str, float] = dict.fromkeys(_TIMING_BREAKDOWN_KEYS, 0.0)
    for record in records:
        attempt_meta = record.get("attempt_metadata")
        if not isinstance(attempt_meta, dict):
            continue
        timing = attempt_meta.get("timing_breakdown")
        if not isinstance(timing, dict):
            continue
        for key in _TIMING_BREAKDOWN_KEYS:
            value = timing.get(key)
            if isinstance(value, (int, float)):
                totals[key] += float(value)
    return totals


def _token_breakdown_sum(records: list[dict[str, Any]]) -> dict[str, int]:
    """Sum each token breakdown field across attempt records."""
    totals: dict[str, int] = dict.fromkeys(_TOKEN_BREAKDOWN_KEYS, 0)
    for record in records:
        attempt_meta = record.get("attempt_metadata")
        if not isinstance(attempt_meta, dict):
            continue
        tokens = attempt_meta.get("token_breakdown")
        if not isinstance(tokens, dict):
            continue
        for key in _TOKEN_BREAKDOWN_KEYS:
            value = tokens.get(key)
            if isinstance(value, (int, float)):
                totals[key] += int(value)
    return totals


def _condition_timing_breakdown_mean(
    rows: list[dict[str, Any]],
) -> dict[str, float | None]:
    """Mean of timing breakdown sums across case-condition rows."""
    accumulators: dict[str, list[float]] = {key: [] for key in _TIMING_BREAKDOWN_KEYS}
    for row in rows:
        timing = row.get("timing_breakdown_sum")
        if not isinstance(timing, dict):
            continue
        for key in _TIMING_BREAKDOWN_KEYS:
            value = timing.get(key)
            if isinstance(value, (int, float)) and value > 0:
                accumulators[key].append(float(value))
    return {key: _mean(values) for key, values in accumulators.items()}


def _condition_token_breakdown_mean(
    rows: list[dict[str, Any]],
) -> dict[str, float | None]:
    """Mean of token breakdown sums across case-condition rows."""
    accumulators: dict[str, list[float]] = {key: [] for key in _TOKEN_BREAKDOWN_KEYS}
    for row in rows:
        tokens = row.get("token_breakdown_sum")
        if not isinstance(tokens, dict):
            continue
        for key in _TOKEN_BREAKDOWN_KEYS:
            value = tokens.get(key)
            if isinstance(value, (int, float)) and value > 0:
                accumulators[key].append(float(value))
    return {key: _mean(values) for key, values in accumulators.items()}


def _baseline_patch_meta_last_str(records: list[dict[str, Any]], key: str) -> str | None:
    for record in reversed(records):
        patch_meta = record.get("patch_meta")
        if not isinstance(patch_meta, dict):
            continue
        value = patch_meta.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _baseline_patch_meta_last_int(records: list[dict[str, Any]], key: str) -> int | None:
    for record in reversed(records):
        patch_meta = record.get("patch_meta")
        if not isinstance(patch_meta, dict):
            continue
        value = patch_meta.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            return value
    return None


def _baseline_patch_meta_last_str_list(records: list[dict[str, Any]], key: str) -> list[str]:
    for record in reversed(records):
        patch_meta = record.get("patch_meta")
        if not isinstance(patch_meta, dict):
            continue
        value = patch_meta.get(key)
        if not isinstance(value, list):
            continue
        return [str(item) for item in value if str(item).strip()]
    return []


def _baseline_meta_from_record(record: dict[str, Any]) -> dict[str, Any]:
    patch_meta = record.get("patch_meta")
    if not isinstance(patch_meta, dict):
        return {}
    baseline_meta = patch_meta.get("baseline_meta")
    return baseline_meta if isinstance(baseline_meta, dict) else {}


def _baseline_meta_last_str(records: list[dict[str, Any]], key: str) -> str | None:
    for record in reversed(records):
        value = _baseline_meta_from_record(record).get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _baseline_attempt_summaries(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    for record in reversed(records):
        attempts = _baseline_meta_from_record(record).get("attempts")
        if not isinstance(attempts, list):
            continue
        return [cast(dict[str, Any], attempt) for attempt in attempts if isinstance(attempt, dict)]
    return []


def _baseline_attempt_rollup(records: list[dict[str, Any]]) -> dict[str, int | None]:
    attempts = _baseline_attempt_summaries(records)
    attempt_count = len(attempts)
    if attempt_count <= 0:
        return {
            "attempt_count": 0,
            "iteration_budget_exhausted_attempt_count": None,
            "no_write_buggy_py_attempt_count": None,
            "no_write_buggy_py_attempt_covered": 0,
            "no_test_attempt_count": None,
            "no_test_attempt_covered": 0,
            "subcall_count_total": None,
            "subcall_count_covered": 0,
            "test_run_count_total": None,
            "test_run_count_covered": 0,
        }
    iteration_budget_exhausted_attempt_count = 0
    no_write_buggy_py_attempt_count = 0
    no_write_buggy_py_attempt_covered = 0
    no_test_attempt_count = 0
    subcall_count_total = 0
    subcall_count_covered = 0
    test_run_count_total = 0
    for attempt in attempts:
        if str(attempt.get("stop_reason") or "").strip() == "iteration_budget_exhausted":
            iteration_budget_exhausted_attempt_count += 1
        no_write = attempt.get("no_write_buggy_py")
        if isinstance(no_write, bool):
            no_write_buggy_py_attempt_covered += 1
            if no_write:
                no_write_buggy_py_attempt_count += 1
        else:
            write_calls = attempt.get("write_buggy_py_calls")
            if isinstance(write_calls, int) and not isinstance(write_calls, bool):
                no_write_buggy_py_attempt_covered += 1
                if write_calls <= 0:
                    no_write_buggy_py_attempt_count += 1
        subcall_count = attempt.get("subcall_count")
        if isinstance(subcall_count, int) and not isinstance(subcall_count, bool):
            subcall_count_total += max(0, subcall_count)
            subcall_count_covered += 1
        test_run_count = attempt.get("test_run_count")
        if isinstance(test_run_count, int) and not isinstance(test_run_count, bool):
            normalized_test_run_count = max(0, test_run_count)
            test_run_count_total += normalized_test_run_count
            if normalized_test_run_count <= 0:
                no_test_attempt_count += 1
            continue
        if isinstance(attempt.get("test_returncode"), int):
            test_run_count_total += 1
        else:
            no_test_attempt_count += 1
    return {
        "attempt_count": attempt_count,
        "iteration_budget_exhausted_attempt_count": iteration_budget_exhausted_attempt_count,
        "no_write_buggy_py_attempt_count": (
            no_write_buggy_py_attempt_count if no_write_buggy_py_attempt_covered > 0 else None
        ),
        "no_write_buggy_py_attempt_covered": no_write_buggy_py_attempt_covered,
        "no_test_attempt_count": no_test_attempt_count,
        "no_test_attempt_covered": attempt_count,
        "subcall_count_total": subcall_count_total if subcall_count_covered > 0 else None,
        "subcall_count_covered": subcall_count_covered,
        "test_run_count_total": test_run_count_total,
        "test_run_count_covered": attempt_count,
    }
