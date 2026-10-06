"""Roll the adaptive-condition telemetry of a bucket of rows up into summary metrics."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from evals.analysis_numeric import _as_float_or_none, _mean, _quantile, _stddev
from evals.analysis_ope import (
    ADAPTIVE_BUDGET_TARGET_POLICIES,
    ADAPTIVE_ONLINE_TARGET_POLICIES,
    _adaptive_ope_summary_from_records,
)
from evals.analysis_ope_support import (
    ADAPTIVE_BUDGET_ACTIONS,
    ADAPTIVE_ONLINE_ACTIONS,
    OpeSummarySettings,
)
from evals.analysis_rows import _records_for_rows
from evals.eval_types import CONDITION_FAMILY_ADAPTIVE, condition_family
from evals.stats import DEFAULT_BOOTSTRAP_SEED


def _has_adaptive_telemetry(row: dict[str, Any]) -> bool:
    """An adaptive-family condition, or a run of another condition with an adaptive mode."""
    if condition_family(row.get("condition")) == CONDITION_FAMILY_ADAPTIVE:
        return True
    return str(row.get("adaptive_mode") or "none") != "none"


def _adaptive_rollup_metrics(
    rows: list[dict[str, Any]],
    *,
    records_by_case_condition: dict[tuple[str, str, str], list[dict[str, Any]]] | None,
    adaptive_ope_min_action_support: int,
    adaptive_ope_propensity_floor: float,
    include_condition_only_counts: bool,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Aggregate adaptive helper, online-policy, and budget-policy metrics.

    Rolls up per-row attempt-level counters into bucket-wide totals,
    rates, and off-policy evaluation summaries.
    """
    total_attempts_executed = sum(int(row.get("attempts_executed") or 0) for row in rows)
    helper_invoked_cases = sum(1 for row in rows if bool(row.get("helper_invoked_case")))
    helper_materialized_cases = sum(1 for row in rows if bool(row.get("helper_materialized_case")))
    helper_invocation_attempts_total = sum(
        int(row.get("helper_invocation_attempts") or 0) for row in rows
    )
    helper_materialized_attempts_total = sum(
        int(row.get("helper_materialized_attempts") or 0) for row in rows
    )
    helper_low_yield_attempts_total = sum(
        int(row.get("helper_low_yield_attempts") or 0) for row in rows
    )
    helper_transport_failure_attempts_total = sum(
        int(row.get("helper_transport_failure_attempts") or 0) for row in rows
    )
    helper_backend_unavailable_attempts_total = sum(
        int(row.get("helper_backend_unavailable_attempts") or 0) for row in rows
    )
    helper_compat_reasoning_only_attempts_total = sum(
        int(row.get("helper_compat_reasoning_only_attempts") or 0) for row in rows
    )
    helper_invalid_payload_attempts_total = sum(
        int(row.get("helper_invalid_payload_attempts") or 0) for row in rows
    )
    helper_request_budget_exceeded_attempts_total = sum(
        int(row.get("helper_request_budget_exceeded_attempts") or 0) for row in rows
    )
    helper_retry_attempts_total = sum(int(row.get("helper_retry_attempts") or 0) for row in rows)
    helper_transport_retry_attempts_total = sum(
        int(row.get("helper_transport_retry_attempts") or 0) for row in rows
    )
    helper_effective_attempts_total = sum(
        int(row.get("helper_effective_attempts") or 0) for row in rows
    )
    helper_success_with_invocation_cases = sum(
        1 for row in rows if bool(row.get("helper_success_with_invocation"))
    )
    helper_rescued_cases = sum(1 for row in rows if bool(row.get("helper_rescued")))
    helper_calls_used_values = [
        float(row.get("helper_calls_used_final"))
        for row in rows
        if isinstance(row.get("helper_calls_used_final"), (int, float))
    ]
    helper_decision_attempts_total = sum(
        int(row.get("helper_decision_attempts") or 0) for row in rows
    )
    helper_need_more_true_attempts_total = sum(
        int(row.get("helper_need_more_true_attempts") or 0) for row in rows
    )
    helper_need_more_false_attempts_total = sum(
        int(row.get("helper_need_more_false_attempts") or 0) for row in rows
    )
    helper_need_more_labeled_attempts = (
        helper_need_more_true_attempts_total + helper_need_more_false_attempts_total
    )
    helper_budget_exhausted_attempts_total = sum(
        int(row.get("helper_budget_exhausted_attempts") or 0) for row in rows
    )
    helper_saturation_guard_attempts_total = sum(
        int(row.get("helper_saturation_guard_attempts") or 0) for row in rows
    )
    adaptive_budget_topup_attempts_total = sum(
        int(row.get("adaptive_budget_topup_attempts") or 0) for row in rows
    )
    adaptive_budget_topup_cases = sum(
        1 for row in rows if bool(row.get("adaptive_budget_topup_case"))
    )
    adaptive_online_decisions_total = sum(
        int(row.get("adaptive_online_decisions") or 0) for row in rows
    )
    adaptive_online_selected_action_counts_total: dict[str, int] = defaultdict(int)
    adaptive_online_applied_action_counts_total: dict[str, int] = defaultdict(int)
    adaptive_online_behavior_action_counts_total: dict[str, int] = defaultdict(int)
    for row in rows:
        selected_counts = row.get("adaptive_online_selected_action_counts")
        if isinstance(selected_counts, dict):
            for action in ADAPTIVE_ONLINE_ACTIONS:
                adaptive_online_selected_action_counts_total[action] += int(
                    selected_counts.get(action) or 0
                )
        applied_counts = row.get("adaptive_online_applied_action_counts")
        if isinstance(applied_counts, dict):
            for action in ADAPTIVE_ONLINE_ACTIONS:
                adaptive_online_applied_action_counts_total[action] += int(
                    applied_counts.get(action) or 0
                )
        behavior_counts = row.get("adaptive_online_behavior_action_counts")
        if isinstance(behavior_counts, dict):
            for action in ADAPTIVE_ONLINE_ACTIONS:
                adaptive_online_behavior_action_counts_total[action] += int(
                    behavior_counts.get(action) or 0
                )
    adaptive_online_selected_invoke_total = sum(
        int(row.get("adaptive_online_selected_invoke_attempts") or 0) for row in rows
    )
    adaptive_online_applied_invoke_total = sum(
        int(row.get("adaptive_online_applied_invoke_attempts") or 0) for row in rows
    )
    adaptive_online_override_total = sum(
        int(row.get("adaptive_online_override_attempts") or 0) for row in rows
    )
    adaptive_online_updates_total = sum(
        int(row.get("adaptive_online_updates") or 0) for row in rows
    )
    adaptive_online_reward_sum_total = sum(
        float(row.get("adaptive_online_reward_sum") or 0.0) for row in rows
    )
    adaptive_online_reward_count_total = sum(
        int(row.get("adaptive_online_reward_count") or 0) for row in rows
    )
    adaptive_online_reward_values_total: list[float] = []
    for row in rows:
        reward_values = row.get("adaptive_online_reward_values")
        if not isinstance(reward_values, list):
            continue
        for value in reward_values:
            parsed = _as_float_or_none(value)
            if parsed is not None:
                adaptive_online_reward_values_total.append(parsed)
    adaptive_online_propensity_sum_total = sum(
        float(row.get("adaptive_online_propensity_sum") or 0.0) for row in rows
    )
    adaptive_online_propensity_count_total = sum(
        int(row.get("adaptive_online_propensity_count") or 0) for row in rows
    )
    adaptive_online_local_tokens_total_sum_total = sum(
        float(row.get("adaptive_online_local_tokens_total_sum") or 0.0) for row in rows
    )
    adaptive_online_local_tokens_total_count_total = sum(
        int(row.get("adaptive_online_local_tokens_total_count") or 0) for row in rows
    )
    adaptive_online_local_wall_sec_sum_total = sum(
        float(row.get("adaptive_online_local_wall_sec_sum") or 0.0) for row in rows
    )
    adaptive_online_local_wall_sec_count_total = sum(
        int(row.get("adaptive_online_local_wall_sec_count") or 0) for row in rows
    )
    adaptive_online_local_llm_call_count_sum_total = sum(
        float(row.get("adaptive_online_local_llm_call_count_sum") or 0.0) for row in rows
    )
    adaptive_online_local_llm_call_count_count_total = sum(
        int(row.get("adaptive_online_local_llm_call_count_count") or 0) for row in rows
    )
    adaptive_online_reward_progress_score_sum_total = sum(
        float(row.get("adaptive_online_reward_progress_score_sum") or 0.0) for row in rows
    )
    adaptive_online_reward_progress_score_count_total = sum(
        int(row.get("adaptive_online_reward_progress_score_count") or 0) for row in rows
    )
    adaptive_online_progress_reason_counts_total: dict[str, int] = defaultdict(int)
    for row in rows:
        reason_counts = row.get("adaptive_online_progress_reason_counts")
        if not isinstance(reason_counts, dict):
            continue
        for reason, value in reason_counts.items():
            if isinstance(reason, str) and isinstance(value, (int, float)):
                adaptive_online_progress_reason_counts_total[reason] += int(value)
    adaptive_online_delayed_rescue_bonus_sum_total = sum(
        float(row.get("adaptive_online_delayed_rescue_bonus_sum") or 0.0) for row in rows
    )
    adaptive_online_delayed_rescue_bonus_count_total = sum(
        int(row.get("adaptive_online_delayed_rescue_bonus_count") or 0) for row in rows
    )
    adaptive_online_credited_rescues_plus_1_total = sum(
        int(row.get("adaptive_online_credited_rescues_plus_1") or 0) for row in rows
    )
    adaptive_online_credited_rescues_plus_2_total = sum(
        int(row.get("adaptive_online_credited_rescues_plus_2") or 0) for row in rows
    )
    adaptive_budget_decisions_total = sum(
        int(row.get("adaptive_budget_decisions") or 0) for row in rows
    )
    adaptive_budget_selected_increase_total = sum(
        int(row.get("adaptive_budget_selected_increase_attempts") or 0) for row in rows
    )
    adaptive_budget_applied_increase_total = sum(
        int(row.get("adaptive_budget_applied_increase_attempts") or 0) for row in rows
    )
    adaptive_budget_selected_time_total = sum(
        int(row.get("adaptive_budget_selected_time_attempts") or 0) for row in rows
    )
    adaptive_budget_selected_tokens_total = sum(
        int(row.get("adaptive_budget_selected_tokens_attempts") or 0) for row in rows
    )
    adaptive_budget_selected_context_total = sum(
        int(row.get("adaptive_budget_selected_context_attempts") or 0) for row in rows
    )
    adaptive_budget_applied_time_total = sum(
        int(row.get("adaptive_budget_applied_time_attempts") or 0) for row in rows
    )
    adaptive_budget_applied_tokens_total = sum(
        int(row.get("adaptive_budget_applied_tokens_attempts") or 0) for row in rows
    )
    adaptive_budget_applied_context_total = sum(
        int(row.get("adaptive_budget_applied_context_attempts") or 0) for row in rows
    )
    adaptive_budget_override_total = sum(
        int(row.get("adaptive_budget_override_attempts") or 0) for row in rows
    )
    adaptive_budget_updates_total = sum(
        int(row.get("adaptive_budget_updates") or 0) for row in rows
    )
    adaptive_budget_reward_sum_total = sum(
        float(row.get("adaptive_budget_reward_sum") or 0.0) for row in rows
    )
    adaptive_budget_reward_count_total = sum(
        int(row.get("adaptive_budget_reward_count") or 0) for row in rows
    )
    adaptive_budget_reward_values_total: list[float] = []
    for row in rows:
        reward_values = row.get("adaptive_budget_reward_values")
        if not isinstance(reward_values, list):
            continue
        for value in reward_values:
            parsed = _as_float_or_none(value)
            if parsed is not None:
                adaptive_budget_reward_values_total.append(parsed)
    adaptive_budget_propensity_sum_total = sum(
        float(row.get("adaptive_budget_propensity_sum") or 0.0) for row in rows
    )
    adaptive_budget_propensity_count_total = sum(
        int(row.get("adaptive_budget_propensity_count") or 0) for row in rows
    )
    adaptive_budget_family_propensity_sum_total = sum(
        float(row.get("adaptive_budget_family_propensity_sum") or 0.0) for row in rows
    )
    adaptive_budget_family_propensity_count_total = sum(
        int(row.get("adaptive_budget_family_propensity_count") or 0) for row in rows
    )
    adaptive_budget_request_budget_exceeded_total = sum(
        int(row.get("adaptive_budget_request_budget_exceeded_attempts") or 0) for row in rows
    )
    adaptive_budget_applied_multiplier_sum_total = sum(
        float(row.get("adaptive_budget_applied_multiplier_sum") or 0.0) for row in rows
    )
    adaptive_budget_applied_multiplier_count_total = sum(
        int(row.get("adaptive_budget_applied_multiplier_count") or 0) for row in rows
    )
    adaptive_budget_local_tokens_total_sum_total = sum(
        float(row.get("adaptive_budget_local_tokens_total_sum") or 0.0) for row in rows
    )
    adaptive_budget_local_tokens_total_count_total = sum(
        int(row.get("adaptive_budget_local_tokens_total_count") or 0) for row in rows
    )
    adaptive_budget_local_wall_sec_sum_total = sum(
        float(row.get("adaptive_budget_local_wall_sec_sum") or 0.0) for row in rows
    )
    adaptive_budget_local_wall_sec_count_total = sum(
        int(row.get("adaptive_budget_local_wall_sec_count") or 0) for row in rows
    )
    adaptive_budget_local_llm_call_count_sum_total = sum(
        float(row.get("adaptive_budget_local_llm_call_count_sum") or 0.0) for row in rows
    )
    adaptive_budget_local_llm_call_count_count_total = sum(
        int(row.get("adaptive_budget_local_llm_call_count_count") or 0) for row in rows
    )
    adaptive_budget_reward_progress_score_sum_total = sum(
        float(row.get("adaptive_budget_reward_progress_score_sum") or 0.0) for row in rows
    )
    adaptive_budget_reward_progress_score_count_total = sum(
        int(row.get("adaptive_budget_reward_progress_score_count") or 0) for row in rows
    )
    adaptive_budget_propose_budget_change_total = sum(
        int(row.get("adaptive_budget_propose_budget_change_attempts") or 0) for row in rows
    )
    adaptive_budget_max_tokens_change_total = sum(
        int(row.get("adaptive_budget_max_tokens_change_attempts") or 0) for row in rows
    )
    adaptive_budget_helper_allowance_change_total = sum(
        int(row.get("adaptive_budget_helper_allowance_change_attempts") or 0) for row in rows
    )
    adaptive_budget_stage_a_requests_change_total = sum(
        int(row.get("adaptive_budget_stage_a_requests_change_attempts") or 0) for row in rows
    )
    adaptive_budget_stage_a_chars_change_total = sum(
        int(row.get("adaptive_budget_stage_a_chars_change_attempts") or 0) for row in rows
    )
    adaptive_budget_retry_history_budget_change_total = sum(
        int(row.get("adaptive_budget_retry_history_budget_change_attempts") or 0) for row in rows
    )
    adaptive_budget_compaction_allowance_change_total = sum(
        int(row.get("adaptive_budget_compaction_allowance_change_attempts") or 0) for row in rows
    )
    adaptive_budget_response_mode_change_total = sum(
        int(row.get("adaptive_budget_response_mode_change_attempts") or 0) for row in rows
    )
    adaptive_budget_partial_override_total = sum(
        int(row.get("adaptive_budget_partial_override_attempts") or 0) for row in rows
    )
    adaptive_budget_effective_helper_max_calls_sum_total = sum(
        float(row.get("adaptive_budget_effective_helper_max_calls_sum") or 0.0) for row in rows
    )
    adaptive_budget_effective_helper_max_calls_count_total = sum(
        int(row.get("adaptive_budget_effective_helper_max_calls_count") or 0) for row in rows
    )
    adaptive_budget_effective_stage_a_max_requests_sum_total = sum(
        float(row.get("adaptive_budget_effective_stage_a_max_requests_sum") or 0.0) for row in rows
    )
    adaptive_budget_effective_stage_a_max_requests_count_total = sum(
        int(row.get("adaptive_budget_effective_stage_a_max_requests_count") or 0) for row in rows
    )
    adaptive_budget_effective_stage_a_max_chars_sum_total = sum(
        float(row.get("adaptive_budget_effective_stage_a_max_chars_sum") or 0.0) for row in rows
    )
    adaptive_budget_effective_stage_a_max_chars_count_total = sum(
        int(row.get("adaptive_budget_effective_stage_a_max_chars_count") or 0) for row in rows
    )
    adaptive_budget_effective_retry_history_max_chars_sum_total = sum(
        float(row.get("adaptive_budget_effective_retry_history_max_chars_sum") or 0.0)
        for row in rows
    )
    adaptive_budget_effective_retry_history_max_chars_count_total = sum(
        int(row.get("adaptive_budget_effective_retry_history_max_chars_count") or 0) for row in rows
    )
    adaptive_budget_effective_context_compaction_max_calls_sum_total = sum(
        float(row.get("adaptive_budget_effective_context_compaction_max_calls_sum") or 0.0)
        for row in rows
    )
    adaptive_budget_effective_context_compaction_max_calls_count_total = sum(
        int(row.get("adaptive_budget_effective_context_compaction_max_calls_count") or 0)
        for row in rows
    )
    adaptive_budget_effective_openai_max_tokens_sum_total = sum(
        float(row.get("adaptive_budget_effective_openai_max_tokens_sum") or 0.0) for row in rows
    )
    adaptive_budget_effective_openai_max_tokens_count_total = sum(
        int(row.get("adaptive_budget_effective_openai_max_tokens_count") or 0) for row in rows
    )
    syntax_preflight_enabled_attempts_total = sum(
        int(row.get("syntax_preflight_enabled_attempts") or 0) for row in rows
    )
    syntax_preflight_cycles_sum_total = sum(
        int(row.get("syntax_preflight_cycles_sum") or 0) for row in rows
    )
    syntax_preflight_failures_sum_total = sum(
        int(row.get("syntax_preflight_failures_sum") or 0) for row in rows
    )
    syntax_preflight_failed_attempts_total = sum(
        int(row.get("syntax_preflight_failed_attempts") or 0) for row in rows
    )
    patch_verifier_attempts_total = sum(
        int(row.get("patch_verifier_attempts") or 0) for row in rows
    )
    patch_verifier_source_scope_matches_total = sum(
        int(row.get("patch_verifier_source_scope_matches") or 0) for row in rows
    )
    patch_verifier_recovery_eligible_attempts_total = sum(
        int(row.get("patch_verifier_recovery_eligible_attempts") or 0) for row in rows
    )
    patch_verifier_recovery_invoked_attempts_total = sum(
        int(row.get("patch_verifier_recovery_invoked_attempts") or 0) for row in rows
    )
    patch_verifier_recovery_passed_attempts_total = sum(
        int(row.get("patch_verifier_recovery_passed_attempts") or 0) for row in rows
    )
    patch_verifier_recovery_result_counts_total: dict[str, int] = defaultdict(int)
    for row in rows:
        result_counts = row.get("patch_verifier_recovery_result_counts")
        if not isinstance(result_counts, dict):
            continue
        for result, value in result_counts.items():
            if isinstance(result, str) and isinstance(value, (int, float)):
                patch_verifier_recovery_result_counts_total[result] += int(value)
    patch_verifier_recovery_terminal_note_counts_total: dict[str, int] = defaultdict(int)
    for row in rows:
        note_counts = row.get("patch_verifier_recovery_terminal_note_counts")
        if not isinstance(note_counts, dict):
            continue
        for note, value in note_counts.items():
            if isinstance(note, str) and isinstance(value, (int, float)):
                patch_verifier_recovery_terminal_note_counts_total[note] += int(value)

    scoped_records = _records_for_rows(rows, records_by_case_condition=records_by_case_condition)
    ope_settings = OpeSummarySettings(
        min_action_support=adaptive_ope_min_action_support,
        propensity_floor=adaptive_ope_propensity_floor,
        bootstrap_seed=int(bootstrap_seed),
    )
    adaptive_online_ope = _adaptive_ope_summary_from_records(
        records=scoped_records,
        prefix="adaptive_online",
        action_space=ADAPTIVE_ONLINE_ACTIONS,
        target_policies=ADAPTIVE_ONLINE_TARGET_POLICIES,
        settings=ope_settings,
    )
    adaptive_budget_ope = _adaptive_ope_summary_from_records(
        records=scoped_records,
        prefix="adaptive_budget",
        action_space=ADAPTIVE_BUDGET_ACTIONS,
        target_policies=ADAPTIVE_BUDGET_TARGET_POLICIES,
        settings=ope_settings,
    )
    adaptive_online_reward_std = _stddev(adaptive_online_reward_values_total)
    adaptive_online_reward_min = min(adaptive_online_reward_values_total, default=None)
    adaptive_online_reward_p50 = _quantile(adaptive_online_reward_values_total, 0.5)
    adaptive_online_reward_p90 = _quantile(adaptive_online_reward_values_total, 0.9)
    adaptive_online_reward_max = max(adaptive_online_reward_values_total, default=None)
    adaptive_budget_reward_std = _stddev(adaptive_budget_reward_values_total)
    adaptive_budget_reward_min = min(adaptive_budget_reward_values_total, default=None)
    adaptive_budget_reward_p50 = _quantile(adaptive_budget_reward_values_total, 0.5)
    adaptive_budget_reward_p90 = _quantile(adaptive_budget_reward_values_total, 0.9)
    adaptive_budget_reward_max = max(adaptive_budget_reward_values_total, default=None)

    out = {
        "helper_invoked_cases": helper_invoked_cases,
        "helper_invocation_case_rate": ((helper_invoked_cases / len(rows)) if rows else None),
        "helper_invocation_attempts": helper_invocation_attempts_total,
        "helper_invocation_attempt_rate": (
            (helper_invocation_attempts_total / total_attempts_executed)
            if total_attempts_executed > 0
            else None
        ),
        "helper_materialized_cases": helper_materialized_cases,
        "helper_materialized_case_rate": (
            (helper_materialized_cases / len(rows)) if rows else None
        ),
        "helper_materialized_attempts": helper_materialized_attempts_total,
        "helper_materialized_attempt_rate": (
            (helper_materialized_attempts_total / total_attempts_executed)
            if total_attempts_executed > 0
            else None
        ),
        "helper_low_yield_attempts": helper_low_yield_attempts_total,
        "helper_low_yield_attempt_rate": (
            (helper_low_yield_attempts_total / total_attempts_executed)
            if total_attempts_executed > 0
            else None
        ),
        "helper_transport_failure_attempts": helper_transport_failure_attempts_total,
        "helper_transport_failure_attempt_rate": (
            (helper_transport_failure_attempts_total / total_attempts_executed)
            if total_attempts_executed > 0
            else None
        ),
        "helper_backend_unavailable_attempts": helper_backend_unavailable_attempts_total,
        "helper_backend_unavailable_attempt_rate": (
            (helper_backend_unavailable_attempts_total / total_attempts_executed)
            if total_attempts_executed > 0
            else None
        ),
        "helper_compat_reasoning_only_attempts": helper_compat_reasoning_only_attempts_total,
        "helper_compat_reasoning_only_attempt_rate": (
            (helper_compat_reasoning_only_attempts_total / total_attempts_executed)
            if total_attempts_executed > 0
            else None
        ),
        "helper_invalid_payload_attempts": helper_invalid_payload_attempts_total,
        "helper_invalid_payload_attempt_rate": (
            (helper_invalid_payload_attempts_total / total_attempts_executed)
            if total_attempts_executed > 0
            else None
        ),
        "helper_request_budget_exceeded_attempts": helper_request_budget_exceeded_attempts_total,
        "helper_request_budget_exceeded_attempt_rate": (
            (helper_request_budget_exceeded_attempts_total / total_attempts_executed)
            if total_attempts_executed > 0
            else None
        ),
        "helper_retry_attempts": helper_retry_attempts_total,
        "helper_transport_retry_attempts": helper_transport_retry_attempts_total,
        "helper_mean_retries_per_invocation": (
            (helper_retry_attempts_total / helper_invocation_attempts_total)
            if helper_invocation_attempts_total > 0
            else None
        ),
        "helper_mean_transport_retries_per_invocation": (
            (helper_transport_retry_attempts_total / helper_invocation_attempts_total)
            if helper_invocation_attempts_total > 0
            else None
        ),
        "helper_effective_attempts": helper_effective_attempts_total,
        "helper_effective_success_rate": (
            (helper_materialized_attempts_total / helper_effective_attempts_total)
            if helper_effective_attempts_total > 0
            else None
        ),
        "helper_success_when_invoked": (
            (helper_success_with_invocation_cases / helper_invoked_cases)
            if helper_invoked_cases > 0
            else None
        ),
        "helper_rescue_rate": (
            (helper_rescued_cases / helper_invoked_cases) if helper_invoked_cases > 0 else None
        ),
        "helper_decision_attempts": helper_decision_attempts_total,
        "helper_need_more_true_rate": (
            (helper_need_more_true_attempts_total / helper_need_more_labeled_attempts)
            if helper_need_more_labeled_attempts > 0
            else None
        ),
        "helper_budget_exhausted_rate": (
            (helper_budget_exhausted_attempts_total / total_attempts_executed)
            if total_attempts_executed > 0
            else None
        ),
        "helper_saturation_guard_rate": (
            (helper_saturation_guard_attempts_total / total_attempts_executed)
            if total_attempts_executed > 0
            else None
        ),
        "adaptive_budget_topup_attempts": adaptive_budget_topup_attempts_total,
        "adaptive_budget_topup_attempt_rate": (
            (adaptive_budget_topup_attempts_total / total_attempts_executed)
            if total_attempts_executed > 0
            else None
        ),
        "adaptive_budget_topup_cases": adaptive_budget_topup_cases,
        "adaptive_budget_topup_case_rate": (
            (adaptive_budget_topup_cases / len(rows)) if rows else None
        ),
        "adaptive_online_decisions": adaptive_online_decisions_total,
        "adaptive_online_selected_action_counts": {
            action: int(adaptive_online_selected_action_counts_total.get(action, 0))
            for action in ADAPTIVE_ONLINE_ACTIONS
        },
        "adaptive_online_selected_action_rates": {
            action: (
                adaptive_online_selected_action_counts_total.get(action, 0)
                / adaptive_online_decisions_total
            )
            if adaptive_online_decisions_total > 0
            else None
            for action in ADAPTIVE_ONLINE_ACTIONS
        },
        "adaptive_online_applied_action_counts": {
            action: int(adaptive_online_applied_action_counts_total.get(action, 0))
            for action in ADAPTIVE_ONLINE_ACTIONS
        },
        "adaptive_online_applied_action_rates": {
            action: (
                adaptive_online_applied_action_counts_total.get(action, 0)
                / adaptive_online_decisions_total
            )
            if adaptive_online_decisions_total > 0
            else None
            for action in ADAPTIVE_ONLINE_ACTIONS
        },
        "adaptive_online_behavior_action_counts": {
            action: int(adaptive_online_behavior_action_counts_total.get(action, 0))
            for action in ADAPTIVE_ONLINE_ACTIONS
        },
        "adaptive_online_behavior_action_rates": {
            action: (
                adaptive_online_behavior_action_counts_total.get(action, 0)
                / adaptive_online_decisions_total
            )
            if adaptive_online_decisions_total > 0
            else None
            for action in ADAPTIVE_ONLINE_ACTIONS
        },
        "adaptive_online_selected_invoke_attempts": adaptive_online_selected_invoke_total,
        "adaptive_online_selected_invoke_rate": (
            (adaptive_online_selected_invoke_total / adaptive_online_decisions_total)
            if adaptive_online_decisions_total > 0
            else None
        ),
        "adaptive_online_applied_invoke_attempts": adaptive_online_applied_invoke_total,
        "adaptive_online_applied_invoke_rate": (
            (adaptive_online_applied_invoke_total / adaptive_online_decisions_total)
            if adaptive_online_decisions_total > 0
            else None
        ),
        "adaptive_online_override_attempts": adaptive_online_override_total,
        "adaptive_online_override_rate": (
            (adaptive_online_override_total / adaptive_online_decisions_total)
            if adaptive_online_decisions_total > 0
            else None
        ),
        "adaptive_online_updates": adaptive_online_updates_total,
        "adaptive_online_update_rate": (
            (adaptive_online_updates_total / adaptive_online_decisions_total)
            if adaptive_online_decisions_total > 0
            else None
        ),
        "adaptive_online_mean_reward": (
            (adaptive_online_reward_sum_total / adaptive_online_reward_count_total)
            if adaptive_online_reward_count_total > 0
            else None
        ),
        "adaptive_online_reward_std": adaptive_online_reward_std,
        "adaptive_online_reward_min": adaptive_online_reward_min,
        "adaptive_online_reward_p50": adaptive_online_reward_p50,
        "adaptive_online_reward_p90": adaptive_online_reward_p90,
        "adaptive_online_reward_max": adaptive_online_reward_max,
        "adaptive_online_mean_propensity": (
            (adaptive_online_propensity_sum_total / adaptive_online_propensity_count_total)
            if adaptive_online_propensity_count_total > 0
            else None
        ),
        "adaptive_online_mean_local_tokens_total": (
            (
                adaptive_online_local_tokens_total_sum_total
                / adaptive_online_local_tokens_total_count_total
            )
            if adaptive_online_local_tokens_total_count_total > 0
            else None
        ),
        "adaptive_online_mean_local_wall_sec": (
            (adaptive_online_local_wall_sec_sum_total / adaptive_online_local_wall_sec_count_total)
            if adaptive_online_local_wall_sec_count_total > 0
            else None
        ),
        "adaptive_online_mean_local_llm_call_count": (
            (
                adaptive_online_local_llm_call_count_sum_total
                / adaptive_online_local_llm_call_count_count_total
            )
            if adaptive_online_local_llm_call_count_count_total > 0
            else None
        ),
        "adaptive_online_mean_progress_score": (
            (
                adaptive_online_reward_progress_score_sum_total
                / adaptive_online_reward_progress_score_count_total
            )
            if adaptive_online_reward_progress_score_count_total > 0
            else None
        ),
        "adaptive_online_progress_reason_counts": dict(
            sorted(adaptive_online_progress_reason_counts_total.items())
        ),
        "adaptive_online_delayed_rescue_bonus_sum": adaptive_online_delayed_rescue_bonus_sum_total,
        "adaptive_online_delayed_rescue_bonus_count": (
            adaptive_online_delayed_rescue_bonus_count_total
        ),
        "adaptive_online_delayed_rescue_bonus_mean": (
            (
                adaptive_online_delayed_rescue_bonus_sum_total
                / adaptive_online_delayed_rescue_bonus_count_total
            )
            if adaptive_online_delayed_rescue_bonus_count_total > 0
            else None
        ),
        "adaptive_online_credited_rescues_plus_1": adaptive_online_credited_rescues_plus_1_total,
        "adaptive_online_credited_rescues_plus_2": adaptive_online_credited_rescues_plus_2_total,
        "adaptive_budget_decisions": adaptive_budget_decisions_total,
        "adaptive_budget_selected_increase_attempts": adaptive_budget_selected_increase_total,
        "adaptive_budget_selected_increase_rate": (
            (adaptive_budget_selected_increase_total / adaptive_budget_decisions_total)
            if adaptive_budget_decisions_total > 0
            else None
        ),
        "adaptive_budget_applied_increase_attempts": adaptive_budget_applied_increase_total,
        "adaptive_budget_applied_increase_rate": (
            (adaptive_budget_applied_increase_total / adaptive_budget_decisions_total)
            if adaptive_budget_decisions_total > 0
            else None
        ),
        "adaptive_budget_selected_time_attempts": adaptive_budget_selected_time_total,
        "adaptive_budget_selected_time_rate": (
            (adaptive_budget_selected_time_total / adaptive_budget_decisions_total)
            if adaptive_budget_decisions_total > 0
            else None
        ),
        "adaptive_budget_selected_tokens_attempts": adaptive_budget_selected_tokens_total,
        "adaptive_budget_selected_tokens_rate": (
            (adaptive_budget_selected_tokens_total / adaptive_budget_decisions_total)
            if adaptive_budget_decisions_total > 0
            else None
        ),
        "adaptive_budget_selected_context_attempts": adaptive_budget_selected_context_total,
        "adaptive_budget_selected_context_rate": (
            (adaptive_budget_selected_context_total / adaptive_budget_decisions_total)
            if adaptive_budget_decisions_total > 0
            else None
        ),
        "adaptive_budget_applied_time_attempts": adaptive_budget_applied_time_total,
        "adaptive_budget_applied_time_rate": (
            (adaptive_budget_applied_time_total / adaptive_budget_decisions_total)
            if adaptive_budget_decisions_total > 0
            else None
        ),
        "adaptive_budget_applied_tokens_attempts": adaptive_budget_applied_tokens_total,
        "adaptive_budget_applied_tokens_rate": (
            (adaptive_budget_applied_tokens_total / adaptive_budget_decisions_total)
            if adaptive_budget_decisions_total > 0
            else None
        ),
        "adaptive_budget_applied_context_attempts": adaptive_budget_applied_context_total,
        "adaptive_budget_applied_context_rate": (
            (adaptive_budget_applied_context_total / adaptive_budget_decisions_total)
            if adaptive_budget_decisions_total > 0
            else None
        ),
        "adaptive_budget_override_attempts": adaptive_budget_override_total,
        "adaptive_budget_override_rate": (
            (adaptive_budget_override_total / adaptive_budget_decisions_total)
            if adaptive_budget_decisions_total > 0
            else None
        ),
        "adaptive_budget_propose_budget_change_attempts": adaptive_budget_propose_budget_change_total,
        "adaptive_budget_propose_budget_change_rate": (
            (adaptive_budget_propose_budget_change_total / adaptive_budget_decisions_total)
            if adaptive_budget_decisions_total > 0
            else None
        ),
        "adaptive_budget_max_tokens_change_attempts": adaptive_budget_max_tokens_change_total,
        "adaptive_budget_max_tokens_change_rate": (
            (adaptive_budget_max_tokens_change_total / adaptive_budget_decisions_total)
            if adaptive_budget_decisions_total > 0
            else None
        ),
        "adaptive_budget_helper_allowance_change_attempts": (
            adaptive_budget_helper_allowance_change_total
        ),
        "adaptive_budget_helper_allowance_change_rate": (
            (adaptive_budget_helper_allowance_change_total / adaptive_budget_decisions_total)
            if adaptive_budget_decisions_total > 0
            else None
        ),
        "adaptive_budget_stage_a_requests_change_attempts": (
            adaptive_budget_stage_a_requests_change_total
        ),
        "adaptive_budget_stage_a_requests_change_rate": (
            (adaptive_budget_stage_a_requests_change_total / adaptive_budget_decisions_total)
            if adaptive_budget_decisions_total > 0
            else None
        ),
        "adaptive_budget_stage_a_chars_change_attempts": adaptive_budget_stage_a_chars_change_total,
        "adaptive_budget_stage_a_chars_change_rate": (
            (adaptive_budget_stage_a_chars_change_total / adaptive_budget_decisions_total)
            if adaptive_budget_decisions_total > 0
            else None
        ),
        "adaptive_budget_retry_history_budget_change_attempts": (
            adaptive_budget_retry_history_budget_change_total
        ),
        "adaptive_budget_retry_history_budget_change_rate": (
            (adaptive_budget_retry_history_budget_change_total / adaptive_budget_decisions_total)
            if adaptive_budget_decisions_total > 0
            else None
        ),
        "adaptive_budget_compaction_allowance_change_attempts": (
            adaptive_budget_compaction_allowance_change_total
        ),
        "adaptive_budget_compaction_allowance_change_rate": (
            (adaptive_budget_compaction_allowance_change_total / adaptive_budget_decisions_total)
            if adaptive_budget_decisions_total > 0
            else None
        ),
        "adaptive_budget_response_mode_change_attempts": adaptive_budget_response_mode_change_total,
        "adaptive_budget_response_mode_change_rate": (
            (adaptive_budget_response_mode_change_total / adaptive_budget_decisions_total)
            if adaptive_budget_decisions_total > 0
            else None
        ),
        "adaptive_budget_partial_override_attempts": adaptive_budget_partial_override_total,
        "adaptive_budget_partial_override_rate": (
            (adaptive_budget_partial_override_total / adaptive_budget_decisions_total)
            if adaptive_budget_decisions_total > 0
            else None
        ),
        "adaptive_budget_updates": adaptive_budget_updates_total,
        "adaptive_budget_update_rate": (
            (adaptive_budget_updates_total / adaptive_budget_decisions_total)
            if adaptive_budget_decisions_total > 0
            else None
        ),
        "adaptive_budget_mean_reward": (
            (adaptive_budget_reward_sum_total / adaptive_budget_reward_count_total)
            if adaptive_budget_reward_count_total > 0
            else None
        ),
        "adaptive_budget_reward_std": adaptive_budget_reward_std,
        "adaptive_budget_reward_min": adaptive_budget_reward_min,
        "adaptive_budget_reward_p50": adaptive_budget_reward_p50,
        "adaptive_budget_reward_p90": adaptive_budget_reward_p90,
        "adaptive_budget_reward_max": adaptive_budget_reward_max,
        "adaptive_budget_mean_propensity": (
            (adaptive_budget_propensity_sum_total / adaptive_budget_propensity_count_total)
            if adaptive_budget_propensity_count_total > 0
            else None
        ),
        "adaptive_budget_mean_family_propensity": (
            (
                adaptive_budget_family_propensity_sum_total
                / adaptive_budget_family_propensity_count_total
            )
            if adaptive_budget_family_propensity_count_total > 0
            else None
        ),
        "adaptive_budget_mean_local_tokens_total": (
            (
                adaptive_budget_local_tokens_total_sum_total
                / adaptive_budget_local_tokens_total_count_total
            )
            if adaptive_budget_local_tokens_total_count_total > 0
            else None
        ),
        "adaptive_budget_mean_local_wall_sec": (
            (adaptive_budget_local_wall_sec_sum_total / adaptive_budget_local_wall_sec_count_total)
            if adaptive_budget_local_wall_sec_count_total > 0
            else None
        ),
        "adaptive_budget_mean_local_llm_call_count": (
            (
                adaptive_budget_local_llm_call_count_sum_total
                / adaptive_budget_local_llm_call_count_count_total
            )
            if adaptive_budget_local_llm_call_count_count_total > 0
            else None
        ),
        "adaptive_budget_mean_progress_score": (
            (
                adaptive_budget_reward_progress_score_sum_total
                / adaptive_budget_reward_progress_score_count_total
            )
            if adaptive_budget_reward_progress_score_count_total > 0
            else None
        ),
        "adaptive_budget_request_budget_exceeded_attempts": adaptive_budget_request_budget_exceeded_total,
        "adaptive_budget_request_budget_exceeded_rate": (
            (adaptive_budget_request_budget_exceeded_total / total_attempts_executed)
            if total_attempts_executed > 0
            else None
        ),
        "adaptive_budget_mean_applied_multiplier": (
            (
                adaptive_budget_applied_multiplier_sum_total
                / adaptive_budget_applied_multiplier_count_total
            )
            if adaptive_budget_applied_multiplier_count_total > 0
            else None
        ),
        "adaptive_budget_mean_effective_helper_max_calls": (
            (
                adaptive_budget_effective_helper_max_calls_sum_total
                / adaptive_budget_effective_helper_max_calls_count_total
            )
            if adaptive_budget_effective_helper_max_calls_count_total > 0
            else None
        ),
        "adaptive_budget_mean_effective_stage_a_max_requests": (
            (
                adaptive_budget_effective_stage_a_max_requests_sum_total
                / adaptive_budget_effective_stage_a_max_requests_count_total
            )
            if adaptive_budget_effective_stage_a_max_requests_count_total > 0
            else None
        ),
        "adaptive_budget_mean_effective_stage_a_max_chars": (
            (
                adaptive_budget_effective_stage_a_max_chars_sum_total
                / adaptive_budget_effective_stage_a_max_chars_count_total
            )
            if adaptive_budget_effective_stage_a_max_chars_count_total > 0
            else None
        ),
        "adaptive_budget_mean_effective_retry_history_max_chars": (
            (
                adaptive_budget_effective_retry_history_max_chars_sum_total
                / adaptive_budget_effective_retry_history_max_chars_count_total
            )
            if adaptive_budget_effective_retry_history_max_chars_count_total > 0
            else None
        ),
        "adaptive_budget_mean_effective_context_compaction_max_calls": (
            (
                adaptive_budget_effective_context_compaction_max_calls_sum_total
                / adaptive_budget_effective_context_compaction_max_calls_count_total
            )
            if adaptive_budget_effective_context_compaction_max_calls_count_total > 0
            else None
        ),
        "adaptive_budget_mean_effective_openai_max_tokens": (
            (
                adaptive_budget_effective_openai_max_tokens_sum_total
                / adaptive_budget_effective_openai_max_tokens_count_total
            )
            if adaptive_budget_effective_openai_max_tokens_count_total > 0
            else None
        ),
        "adaptive_online_ope": adaptive_online_ope,
        "adaptive_budget_ope": adaptive_budget_ope,
        "mean_syntax_preflight_cycles": (
            (syntax_preflight_cycles_sum_total / syntax_preflight_enabled_attempts_total)
            if syntax_preflight_enabled_attempts_total > 0
            else None
        ),
        "syntax_preflight_failures": syntax_preflight_failures_sum_total,
        "syntax_preflight_failed_attempts": syntax_preflight_failed_attempts_total,
        "syntax_preflight_failed_attempt_rate": (
            (syntax_preflight_failed_attempts_total / total_attempts_executed)
            if total_attempts_executed > 0
            else None
        ),
        "patch_verifier_source_scope_matches": patch_verifier_source_scope_matches_total,
        "patch_verifier_source_scope_match_rate": (
            (patch_verifier_source_scope_matches_total / patch_verifier_attempts_total)
            if patch_verifier_attempts_total > 0
            else None
        ),
        "patch_verifier_recovery_eligible_attempts": (
            patch_verifier_recovery_eligible_attempts_total
        ),
        "patch_verifier_recovery_invoked_attempts": patch_verifier_recovery_invoked_attempts_total,
        "patch_verifier_recovery_passed_attempts": patch_verifier_recovery_passed_attempts_total,
        "patch_verifier_recovery_pass_rate": (
            (
                patch_verifier_recovery_passed_attempts_total
                / patch_verifier_recovery_invoked_attempts_total
            )
            if patch_verifier_recovery_invoked_attempts_total > 0
            else None
        ),
        "patch_verifier_recovery_result_counts": dict(
            sorted(patch_verifier_recovery_result_counts_total.items())
        ),
        "patch_verifier_recovery_terminal_note_counts": dict(
            sorted(patch_verifier_recovery_terminal_note_counts_total.items())
        ),
    }
    if include_condition_only_counts:
        out.update(
            {
                "helper_success_with_invocation_cases": helper_success_with_invocation_cases,
                "helper_rescued_cases": helper_rescued_cases,
                "mean_helper_calls_used_final": _mean(helper_calls_used_values),
                "helper_materialized_attempts": helper_materialized_attempts_total,
                "helper_low_yield_attempts": helper_low_yield_attempts_total,
                "helper_need_more_true_attempts": helper_need_more_true_attempts_total,
                "helper_need_more_false_attempts": helper_need_more_false_attempts_total,
                "helper_budget_exhausted_attempts": helper_budget_exhausted_attempts_total,
                "helper_saturation_guard_attempts": helper_saturation_guard_attempts_total,
            }
        )
    return out
