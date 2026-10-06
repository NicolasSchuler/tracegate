"""Derive one row per (run, case, condition) from attempt records, and the uplift comparisons over rows."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from evals.analysis_attempt_metrics import (
    _adaptive_budget_policy_attempt_metrics,
    _adaptive_budget_topup_attempt_metrics,
    _adaptive_helper_attempt_metrics,
    _adaptive_online_attempt_metrics,
    _context_protocol_from_records,
    _patch_verifier_source_scope_metrics,
    _retry_gate_counts,
    _stage_a_request_counts,
    _stage_b_prompt_chars_mean,
    _syntax_preflight_attempt_metrics,
)
from evals.analysis_numeric import _ratio
from evals.analysis_usage import (
    _attempts_budget_from_records,
    _attempts_executed,
    _attempts_to_success,
    _baseline_attempt_rollup,
    _baseline_meta_last_str,
    _baseline_metric_last_bool,
    _baseline_metric_last_int,
    _baseline_metric_last_str,
    _baseline_metric_sum,
    _baseline_patch_meta_last_int,
    _baseline_patch_meta_last_str,
    _baseline_patch_meta_last_str_list,
    _dedupe_records_by_attempt,
    _is_external_baseline_condition,
    _llm_calls_total,
    _patch_meta_flag_count,
    _reasoning_token_usage_breakdown,
    _retry_history_attempt_metrics,
    _runtime_used,
    _test_touch_attempt_metrics,
    _timing_breakdown_sum,
    _token_breakdown_sum,
    _token_usage,
)
from evals.case_outcome import case_outcome
from evals.stats import _solved_within


def _derive_case_condition_rows(
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    """Derive one summary row per (run_id, case_id, condition) group.

    Each row aggregates attempts into success, token usage, runtime,
    adaptive helper, online-policy, budget-policy, and patch-verifier
    metrics used by downstream reporting and statistical analysis.
    """
    rows: list[dict[str, Any]] = []
    for (run_id, case_id, condition), records in sorted(grouped.items()):
        ordered = _dedupe_records_by_attempt(records)
        outcome = case_outcome(ordered)  # STA-19: the one definition of "case succeeded"
        success = outcome is not None and outcome.success
        attempts_executed = _attempts_executed(ordered)
        attempts_budget = _attempts_budget_from_records(ordered)
        attempts_to_success = _attempts_to_success(ordered)
        success_attempt_ratio = (
            (float(attempts_to_success) / float(attempts_budget))
            if attempts_to_success is not None and attempts_budget > 0
            else None
        )
        token_usage = _token_usage(ordered)
        token_prompt, token_completion, token_total = (
            token_usage.prompt,
            token_usage.completion,
            token_usage.total,
        )
        token_reasoning, reasoning_token_usage_available, reasoning_token_usage_per_attempt_mean = (
            _reasoning_token_usage_breakdown(ordered)
        )
        tokens = token_total
        # STA-09: divide by the attempts that reported tokens, not all executed attempts.
        token_usage_per_attempt_mean = (
            (token_total / float(token_usage.covered_attempts))
            if token_total is not None and token_usage.covered_attempts > 0
            else None
        )
        runtime = _runtime_used(ordered)
        llm_calls_total = _llm_calls_total(ordered)
        retry_history_stats = _retry_history_attempt_metrics(ordered)
        exemplar = ordered[0]
        context_protocol = _context_protocol_from_records(ordered)
        stage_b_prompt_chars = _stage_b_prompt_chars_mean(ordered)
        stage_a_attempts, stage_a_successes = _stage_a_request_counts(ordered)
        retry_gate_attempts, retry_gate_blocks = _retry_gate_counts(ordered)
        patch_attempts, test_touch_attempts = _test_touch_attempt_metrics(ordered)
        relaxed_apply_attempts = _patch_meta_flag_count(
            ordered, "patch_apply_mode", "relaxed_fallback"
        )
        test_surface_modified_attempts = _patch_meta_flag_count(
            ordered, "test_surface_modified", True
        )
        adaptive_metrics = _adaptive_helper_attempt_metrics(ordered)
        adaptive_online_metrics = _adaptive_online_attempt_metrics(ordered)
        adaptive_budget_policy_metrics = _adaptive_budget_policy_attempt_metrics(ordered)
        adaptive_topup_metrics = _adaptive_budget_topup_attempt_metrics(ordered)
        syntax_preflight_metrics = _syntax_preflight_attempt_metrics(ordered)
        patch_verifier_scope_metrics = _patch_verifier_source_scope_metrics(ordered)
        baseline_contract_version = _baseline_patch_meta_last_str(ordered, "contract_version")
        baseline_fidelity_level = _baseline_patch_meta_last_str(ordered, "fidelity_level")
        baseline_repair_scope_policy = _baseline_patch_meta_last_str(ordered, "repair_scope_policy")
        baseline_prompt_contract_kind = _baseline_patch_meta_last_str(
            ordered, "prompt_contract_kind"
        )
        baseline_hybrid_guidance_mode = _baseline_patch_meta_last_str(
            ordered, "baseline_hybrid_guidance_mode"
        )
        baseline_hybrid_guidance_chars = _baseline_patch_meta_last_int(
            ordered, "baseline_hybrid_guidance_chars"
        )
        baseline_hybrid_guidance_sections = _baseline_patch_meta_last_str_list(
            ordered, "baseline_hybrid_guidance_sections"
        )
        baseline_output_contract = _baseline_patch_meta_last_str(ordered, "output_contract")
        baseline_upstream_commit = _baseline_patch_meta_last_str(ordered, "upstream_commit")
        is_external_baseline = _is_external_baseline_condition(condition)
        # BAS-01 / D3: a case the baseline can't represent is not a baseline failure.
        baseline_supported = not any(
            isinstance(item.get("patch_meta"), dict)
            and item["patch_meta"].get("baseline_supported") is False
            for item in ordered
        )
        baseline_attempt_rollup = (
            _baseline_attempt_rollup(ordered) if is_external_baseline else {"attempt_count": 0}
        )
        baseline_teacher_duration_sec_total = (
            _baseline_metric_sum(ordered, "teacher_duration_sec_total")
            if is_external_baseline
            else None
        )
        baseline_learner_duration_sec_total = (
            _baseline_metric_sum(ordered, "learner_duration_sec_total")
            if is_external_baseline
            else None
        )
        baseline_postcheck_duration_sec_total = (
            _baseline_metric_sum(ordered, "postcheck_duration_sec_total")
            if is_external_baseline
            else None
        )
        baseline_learner_empty_response_count = (
            _baseline_metric_sum(ordered, "learner_empty_response_count")
            if is_external_baseline
            else None
        )
        baseline_attempts_with_postcheck_count = (
            _baseline_metric_sum(ordered, "attempts_with_postcheck_count")
            if is_external_baseline
            else None
        )
        baseline_direct_gen_passed = (
            _baseline_metric_last_bool(ordered, "direct_gen_passed")
            if is_external_baseline
            else None
        )
        baseline_debug_added_value = (
            _baseline_metric_last_bool(ordered, "debug_added_value")
            if is_external_baseline
            else None
        )
        baseline_method_instrumentation = (
            _baseline_metric_last_bool(ordered, "method_instrumentation")
            if is_external_baseline
            else None
        )
        baseline_method_two_step_repair = (
            _baseline_metric_last_bool(ordered, "method_two_step_repair")
            if is_external_baseline
            else None
        )
        baseline_stopped_reason = (
            _baseline_metric_last_str(ordered, "stopped_reason") if is_external_baseline else None
        )
        baseline_is_passing = (
            _baseline_metric_last_bool(ordered, "is_passing") if is_external_baseline else None
        )
        baseline_test_feedback_steps = (
            _baseline_metric_sum(ordered, "test_feedback_steps") if is_external_baseline else None
        )
        baseline_test_feedback_nonempty_steps = (
            _baseline_metric_sum(ordered, "test_feedback_nonempty_steps")
            if is_external_baseline
            else None
        )
        baseline_test_feedback_improvement_steps = (
            _baseline_metric_sum(ordered, "test_feedback_improvement_steps")
            if is_external_baseline
            else None
        )
        baseline_test_feedback_regression_steps = (
            _baseline_metric_sum(ordered, "test_feedback_regression_steps")
            if is_external_baseline
            else None
        )
        baseline_test_feedback_stagnation_steps = (
            _baseline_metric_sum(ordered, "test_feedback_stagnation_steps")
            if is_external_baseline
            else None
        )
        baseline_test_feedback_final_fail_count = (
            _baseline_metric_last_int(ordered, "test_feedback_final_fail_count")
            if is_external_baseline
            else None
        )
        baseline_terminal_timeout = (
            (_baseline_meta_last_str(ordered, "error") == "timeout") and not success
            if is_external_baseline
            else None
        )
        helper_invocation_attempts = int(adaptive_metrics["helper_invocation_attempts"])
        helper_materialized_attempts = int(adaptive_metrics["helper_materialized_attempts"])
        helper_low_yield_attempts = int(adaptive_metrics["helper_low_yield_attempts"])
        helper_transport_failure_attempts = int(
            adaptive_metrics["helper_transport_failure_attempts"]
        )
        helper_backend_unavailable_attempts = int(
            adaptive_metrics["helper_backend_unavailable_attempts"]
        )
        helper_compat_reasoning_only_attempts = int(
            adaptive_metrics["helper_compat_reasoning_only_attempts"]
        )
        helper_invalid_payload_attempts = int(adaptive_metrics["helper_invalid_payload_attempts"])
        helper_request_budget_exceeded_attempts = int(
            adaptive_metrics["helper_request_budget_exceeded_attempts"]
        )
        helper_retry_attempts = int(adaptive_metrics["helper_retry_attempts"])
        helper_transport_retry_attempts = int(adaptive_metrics["helper_transport_retry_attempts"])
        helper_effective_attempts = max(
            0,
            helper_invocation_attempts
            - helper_transport_failure_attempts
            - helper_backend_unavailable_attempts
            - helper_compat_reasoning_only_attempts,
        )
        helper_invoked_case = helper_invocation_attempts > 0
        helper_materialized_case = helper_materialized_attempts > 0
        helper_success_with_invocation = bool(success and helper_invoked_case)
        first_helper_attempt = adaptive_metrics["first_helper_attempt"]
        helper_rescued = bool(
            helper_success_with_invocation
            and isinstance(first_helper_attempt, int)
            and first_helper_attempt > 1
        )
        rows.append(
            {
                "run_id": run_id,
                "case_id": case_id,
                "condition": condition,
                "category": str(exemplar.get("category") or "unknown"),
                "difficulty": str(exemplar.get("difficulty") or "unknown"),
                "snapshot_dependency": str(exemplar.get("snapshot_dependency") or "unknown"),
                "bug_source": exemplar.get("bug_source"),
                "contamination_risk": exemplar.get("contamination_risk"),
                "success": success,
                "attempts_executed": attempts_executed,
                "attempts_budget": attempts_budget,
                "attempts_to_success": attempts_to_success,
                "success_attempt_ratio": success_attempt_ratio,
                "token_usage": tokens,
                "token_usage_prompt": token_prompt,
                "token_usage_completion": token_completion,
                "token_usage_total": token_total,
                "token_usage_reasoning": token_reasoning,
                "reasoning_token_usage_available": reasoning_token_usage_available,
                "token_usage_per_attempt_mean": token_usage_per_attempt_mean,
                "token_covered_attempts": token_usage.covered_attempts,
                "token_covered_fraction": _ratio(token_usage.covered_attempts, attempts_executed),
                "reasoning_token_usage_per_attempt_mean": (reasoning_token_usage_per_attempt_mean),
                "runtime_sec": runtime,
                "llm_calls_total": llm_calls_total,
                "timing_breakdown_sum": _timing_breakdown_sum(ordered),
                "token_breakdown_sum": _token_breakdown_sum(ordered),
                "context_protocol": context_protocol,
                "stage_b_prompt_chars_mean": stage_b_prompt_chars,
                "stage_a_request_attempts": stage_a_attempts,
                "stage_a_request_successes": stage_a_successes,
                "retry_gate_attempts": retry_gate_attempts,
                "retry_gate_blocks": retry_gate_blocks,
                "patch_attempts": patch_attempts,
                "baseline_supported": baseline_supported,
                "baseline_compatibility_reason": (
                    None
                    if baseline_supported
                    else _baseline_patch_meta_last_str(ordered, "compatibility_reason")
                ),
                "test_touch_attempts": test_touch_attempts,
                "test_touch_attempt_rate": (
                    (test_touch_attempts / patch_attempts) if patch_attempts > 0 else None
                ),
                "relaxed_apply_attempts": relaxed_apply_attempts,
                "test_surface_modified_attempts": test_surface_modified_attempts,
                "baseline_teacher_duration_sec_total": baseline_teacher_duration_sec_total,
                "baseline_learner_duration_sec_total": baseline_learner_duration_sec_total,
                "baseline_postcheck_duration_sec_total": baseline_postcheck_duration_sec_total,
                "baseline_contract_version": baseline_contract_version,
                "baseline_fidelity_level": baseline_fidelity_level,
                "baseline_repair_scope_policy": baseline_repair_scope_policy,
                "baseline_prompt_contract_kind": baseline_prompt_contract_kind,
                "baseline_hybrid_guidance_mode": baseline_hybrid_guidance_mode,
                "baseline_hybrid_guidance_chars": baseline_hybrid_guidance_chars,
                "baseline_hybrid_guidance_sections": baseline_hybrid_guidance_sections,
                "baseline_output_contract": baseline_output_contract,
                "baseline_upstream_commit": baseline_upstream_commit,
                "baseline_learner_empty_response_count": baseline_learner_empty_response_count,
                "baseline_attempts_with_postcheck_count": baseline_attempts_with_postcheck_count,
                "baseline_direct_gen_passed": baseline_direct_gen_passed,
                "baseline_debug_added_value": baseline_debug_added_value,
                "baseline_method_instrumentation": baseline_method_instrumentation,
                "baseline_method_two_step_repair": baseline_method_two_step_repair,
                "baseline_stopped_reason": baseline_stopped_reason,
                "baseline_is_passing": baseline_is_passing,
                "baseline_test_feedback_steps": baseline_test_feedback_steps,
                "baseline_test_feedback_nonempty_steps": baseline_test_feedback_nonempty_steps,
                "baseline_test_feedback_improvement_steps": (
                    baseline_test_feedback_improvement_steps
                ),
                "baseline_test_feedback_regression_steps": baseline_test_feedback_regression_steps,
                "baseline_test_feedback_stagnation_steps": baseline_test_feedback_stagnation_steps,
                "baseline_test_feedback_final_fail_count": (
                    baseline_test_feedback_final_fail_count
                ),
                "baseline_terminal_timeout": baseline_terminal_timeout,
                "baseline_attempt_summary_count": baseline_attempt_rollup.get("attempt_count"),
                "baseline_iteration_budget_exhausted_attempt_count": baseline_attempt_rollup.get(
                    "iteration_budget_exhausted_attempt_count"
                ),
                "baseline_no_write_buggy_py_attempt_count": baseline_attempt_rollup.get(
                    "no_write_buggy_py_attempt_count"
                ),
                "baseline_no_write_buggy_py_attempt_covered": baseline_attempt_rollup.get(
                    "no_write_buggy_py_attempt_covered"
                ),
                "baseline_no_test_attempt_count": baseline_attempt_rollup.get(
                    "no_test_attempt_count"
                ),
                "baseline_no_test_attempt_covered": baseline_attempt_rollup.get(
                    "no_test_attempt_covered"
                ),
                "baseline_subcall_count_total": baseline_attempt_rollup.get("subcall_count_total"),
                "baseline_subcall_count_covered": baseline_attempt_rollup.get(
                    "subcall_count_covered"
                ),
                "baseline_test_run_count_total": baseline_attempt_rollup.get(
                    "test_run_count_total"
                ),
                "baseline_test_run_count_covered": baseline_attempt_rollup.get(
                    "test_run_count_covered"
                ),
                "adaptive_mode": adaptive_metrics["adaptive_mode"],
                "helper_invocation_attempts": helper_invocation_attempts,
                "helper_invocation_attempt_rate": (
                    (helper_invocation_attempts / attempts_executed)
                    if attempts_executed > 0
                    else None
                ),
                "helper_invoked_case": helper_invoked_case,
                "helper_materialized_attempts": helper_materialized_attempts,
                "helper_materialized_attempt_rate": (
                    (helper_materialized_attempts / attempts_executed)
                    if attempts_executed > 0
                    else None
                ),
                "helper_materialized_case": helper_materialized_case,
                "helper_low_yield_attempts": helper_low_yield_attempts,
                "helper_low_yield_attempt_rate": (
                    (helper_low_yield_attempts / attempts_executed)
                    if attempts_executed > 0
                    else None
                ),
                "helper_transport_failure_attempts": helper_transport_failure_attempts,
                "helper_transport_failure_attempt_rate": (
                    (helper_transport_failure_attempts / attempts_executed)
                    if attempts_executed > 0
                    else None
                ),
                "helper_backend_unavailable_attempts": helper_backend_unavailable_attempts,
                "helper_backend_unavailable_attempt_rate": (
                    (helper_backend_unavailable_attempts / attempts_executed)
                    if attempts_executed > 0
                    else None
                ),
                "helper_compat_reasoning_only_attempts": helper_compat_reasoning_only_attempts,
                "helper_compat_reasoning_only_attempt_rate": (
                    (helper_compat_reasoning_only_attempts / attempts_executed)
                    if attempts_executed > 0
                    else None
                ),
                "helper_invalid_payload_attempts": helper_invalid_payload_attempts,
                "helper_invalid_payload_attempt_rate": (
                    (helper_invalid_payload_attempts / attempts_executed)
                    if attempts_executed > 0
                    else None
                ),
                "helper_request_budget_exceeded_attempts": (
                    helper_request_budget_exceeded_attempts
                ),
                "helper_request_budget_exceeded_attempt_rate": (
                    (helper_request_budget_exceeded_attempts / attempts_executed)
                    if attempts_executed > 0
                    else None
                ),
                "helper_retry_attempts": helper_retry_attempts,
                "helper_transport_retry_attempts": helper_transport_retry_attempts,
                "helper_effective_attempts": helper_effective_attempts,
                "helper_effective_success_rate": (
                    (helper_materialized_attempts / helper_effective_attempts)
                    if helper_effective_attempts > 0
                    else None
                ),
                "helper_success_with_invocation": helper_success_with_invocation,
                "helper_rescued": helper_rescued,
                "helper_first_attempt": first_helper_attempt,
                "helper_calls_used_final": adaptive_metrics["helper_calls_used_final"],
                "helper_decision_attempts": adaptive_metrics["helper_decision_attempts"],
                "helper_need_more_true_attempts": adaptive_metrics[
                    "helper_need_more_true_attempts"
                ],
                "helper_need_more_false_attempts": adaptive_metrics[
                    "helper_need_more_false_attempts"
                ],
                "helper_budget_exhausted_attempts": adaptive_metrics[
                    "helper_budget_exhausted_attempts"
                ],
                "helper_saturation_guard_attempts": adaptive_metrics[
                    "helper_saturation_guard_attempts"
                ],
                "adaptive_budget_topup_attempts": adaptive_topup_metrics["topup_attempts"],
                "adaptive_budget_topup_case": bool(adaptive_topup_metrics["topup_attempts"] > 0),
                "adaptive_online_mode": adaptive_online_metrics["adaptive_online_mode"],
                "adaptive_online_policy_algo": adaptive_online_metrics[
                    "adaptive_online_policy_algo"
                ],
                "adaptive_online_decisions": adaptive_online_metrics["decisions"],
                "adaptive_online_selected_action_counts": adaptive_online_metrics[
                    "selected_action_counts"
                ],
                "adaptive_online_applied_action_counts": adaptive_online_metrics[
                    "applied_action_counts"
                ],
                "adaptive_online_behavior_action_counts": adaptive_online_metrics[
                    "behavior_action_counts"
                ],
                "adaptive_online_selected_invoke_attempts": adaptive_online_metrics[
                    "selected_invoke_attempts"
                ],
                "adaptive_online_applied_invoke_attempts": adaptive_online_metrics[
                    "applied_invoke_attempts"
                ],
                "adaptive_online_override_attempts": adaptive_online_metrics["override_attempts"],
                "adaptive_online_updates": adaptive_online_metrics["updates"],
                "adaptive_online_reward_sum": adaptive_online_metrics["reward_sum"],
                "adaptive_online_reward_count": adaptive_online_metrics["reward_count"],
                "adaptive_online_reward_values": adaptive_online_metrics["reward_values"],
                "adaptive_online_propensity_sum": adaptive_online_metrics["propensity_sum"],
                "adaptive_online_propensity_count": adaptive_online_metrics["propensity_count"],
                "adaptive_online_local_tokens_total_sum": adaptive_online_metrics[
                    "local_tokens_total_sum"
                ],
                "adaptive_online_local_tokens_total_count": adaptive_online_metrics[
                    "local_tokens_total_count"
                ],
                "adaptive_online_local_wall_sec_sum": adaptive_online_metrics["local_wall_sec_sum"],
                "adaptive_online_local_wall_sec_count": adaptive_online_metrics[
                    "local_wall_sec_count"
                ],
                "adaptive_online_local_llm_call_count_sum": adaptive_online_metrics[
                    "local_llm_call_count_sum"
                ],
                "adaptive_online_local_llm_call_count_count": adaptive_online_metrics[
                    "local_llm_call_count_count"
                ],
                "adaptive_online_reward_progress_score_sum": adaptive_online_metrics[
                    "reward_progress_score_sum"
                ],
                "adaptive_online_reward_progress_score_count": adaptive_online_metrics[
                    "reward_progress_score_count"
                ],
                "adaptive_online_progress_reason_counts": adaptive_online_metrics[
                    "progress_reason_counts"
                ],
                "adaptive_online_delayed_rescue_bonus_sum": adaptive_online_metrics[
                    "delayed_rescue_bonus_sum"
                ],
                "adaptive_online_delayed_rescue_bonus_count": adaptive_online_metrics[
                    "delayed_rescue_bonus_count"
                ],
                "adaptive_online_credited_rescues_plus_1": adaptive_online_metrics[
                    "credited_rescues_plus_1"
                ],
                "adaptive_online_credited_rescues_plus_2": adaptive_online_metrics[
                    "credited_rescues_plus_2"
                ],
                "adaptive_budget_policy_mode": adaptive_budget_policy_metrics[
                    "adaptive_budget_policy_mode"
                ],
                "adaptive_budget_policy_algo": adaptive_budget_policy_metrics[
                    "adaptive_budget_policy_algo"
                ],
                "adaptive_budget_policy_architecture": adaptive_budget_policy_metrics[
                    "adaptive_budget_policy_architecture"
                ],
                "adaptive_budget_decisions": adaptive_budget_policy_metrics["decisions"],
                "adaptive_budget_selected_increase_attempts": adaptive_budget_policy_metrics[
                    "selected_increase_attempts"
                ],
                "adaptive_budget_applied_increase_attempts": adaptive_budget_policy_metrics[
                    "applied_increase_attempts"
                ],
                "adaptive_budget_selected_time_attempts": adaptive_budget_policy_metrics[
                    "selected_time_attempts"
                ],
                "adaptive_budget_selected_tokens_attempts": adaptive_budget_policy_metrics[
                    "selected_tokens_attempts"
                ],
                "adaptive_budget_selected_context_attempts": adaptive_budget_policy_metrics[
                    "selected_context_attempts"
                ],
                "adaptive_budget_applied_time_attempts": adaptive_budget_policy_metrics[
                    "applied_time_attempts"
                ],
                "adaptive_budget_applied_tokens_attempts": adaptive_budget_policy_metrics[
                    "applied_tokens_attempts"
                ],
                "adaptive_budget_applied_context_attempts": adaptive_budget_policy_metrics[
                    "applied_context_attempts"
                ],
                "adaptive_budget_override_attempts": adaptive_budget_policy_metrics[
                    "override_attempts"
                ],
                "adaptive_budget_updates": adaptive_budget_policy_metrics["updates"],
                "adaptive_budget_reward_sum": adaptive_budget_policy_metrics["reward_sum"],
                "adaptive_budget_reward_count": adaptive_budget_policy_metrics["reward_count"],
                "adaptive_budget_reward_values": adaptive_budget_policy_metrics["reward_values"],
                "adaptive_budget_propensity_sum": adaptive_budget_policy_metrics["propensity_sum"],
                "adaptive_budget_propensity_count": adaptive_budget_policy_metrics[
                    "propensity_count"
                ],
                "adaptive_budget_family_propensity_sum": adaptive_budget_policy_metrics[
                    "family_propensity_sum"
                ],
                "adaptive_budget_family_propensity_count": adaptive_budget_policy_metrics[
                    "family_propensity_count"
                ],
                "adaptive_budget_request_budget_exceeded_attempts": adaptive_budget_policy_metrics[
                    "request_budget_exceeded_attempts"
                ],
                "adaptive_budget_applied_multiplier_sum": adaptive_budget_policy_metrics[
                    "applied_multiplier_sum"
                ],
                "adaptive_budget_applied_multiplier_count": adaptive_budget_policy_metrics[
                    "applied_multiplier_count"
                ],
                "adaptive_budget_local_tokens_total_sum": adaptive_budget_policy_metrics[
                    "local_tokens_total_sum"
                ],
                "adaptive_budget_local_tokens_total_count": adaptive_budget_policy_metrics[
                    "local_tokens_total_count"
                ],
                "adaptive_budget_local_wall_sec_sum": adaptive_budget_policy_metrics[
                    "local_wall_sec_sum"
                ],
                "adaptive_budget_local_wall_sec_count": adaptive_budget_policy_metrics[
                    "local_wall_sec_count"
                ],
                "adaptive_budget_local_llm_call_count_sum": adaptive_budget_policy_metrics[
                    "local_llm_call_count_sum"
                ],
                "adaptive_budget_local_llm_call_count_count": adaptive_budget_policy_metrics[
                    "local_llm_call_count_count"
                ],
                "adaptive_budget_reward_progress_score_sum": adaptive_budget_policy_metrics[
                    "reward_progress_score_sum"
                ],
                "adaptive_budget_reward_progress_score_count": adaptive_budget_policy_metrics[
                    "reward_progress_score_count"
                ],
                "adaptive_budget_propose_budget_change_attempts": adaptive_budget_policy_metrics[
                    "propose_budget_change_attempts"
                ],
                "adaptive_budget_max_tokens_change_attempts": adaptive_budget_policy_metrics[
                    "max_tokens_change_attempts"
                ],
                "adaptive_budget_helper_allowance_change_attempts": adaptive_budget_policy_metrics[
                    "helper_allowance_change_attempts"
                ],
                "adaptive_budget_stage_a_requests_change_attempts": (
                    adaptive_budget_policy_metrics["stage_a_requests_change_attempts"]
                ),
                "adaptive_budget_stage_a_chars_change_attempts": adaptive_budget_policy_metrics[
                    "stage_a_chars_change_attempts"
                ],
                "adaptive_budget_retry_history_budget_change_attempts": (
                    adaptive_budget_policy_metrics["retry_history_budget_change_attempts"]
                ),
                "adaptive_budget_compaction_allowance_change_attempts": (
                    adaptive_budget_policy_metrics["compaction_allowance_change_attempts"]
                ),
                "adaptive_budget_response_mode_change_attempts": adaptive_budget_policy_metrics[
                    "response_mode_change_attempts"
                ],
                "adaptive_budget_partial_override_attempts": adaptive_budget_policy_metrics[
                    "partial_override_attempts"
                ],
                "adaptive_budget_effective_helper_max_calls_sum": adaptive_budget_policy_metrics[
                    "effective_helper_max_calls_sum"
                ],
                "adaptive_budget_effective_helper_max_calls_count": adaptive_budget_policy_metrics[
                    "effective_helper_max_calls_count"
                ],
                "adaptive_budget_effective_stage_a_max_requests_sum": (
                    adaptive_budget_policy_metrics["effective_stage_a_max_requests_sum"]
                ),
                "adaptive_budget_effective_stage_a_max_requests_count": (
                    adaptive_budget_policy_metrics["effective_stage_a_max_requests_count"]
                ),
                "adaptive_budget_effective_stage_a_max_chars_sum": adaptive_budget_policy_metrics[
                    "effective_stage_a_max_chars_sum"
                ],
                "adaptive_budget_effective_stage_a_max_chars_count": (
                    adaptive_budget_policy_metrics["effective_stage_a_max_chars_count"]
                ),
                "adaptive_budget_effective_retry_history_max_chars_sum": (
                    adaptive_budget_policy_metrics["effective_retry_history_max_chars_sum"]
                ),
                "adaptive_budget_effective_retry_history_max_chars_count": (
                    adaptive_budget_policy_metrics["effective_retry_history_max_chars_count"]
                ),
                "adaptive_budget_effective_context_compaction_max_calls_sum": (
                    adaptive_budget_policy_metrics["effective_context_compaction_max_calls_sum"]
                ),
                "adaptive_budget_effective_context_compaction_max_calls_count": (
                    adaptive_budget_policy_metrics["effective_context_compaction_max_calls_count"]
                ),
                "adaptive_budget_effective_openai_max_tokens_sum": (
                    adaptive_budget_policy_metrics["effective_openai_max_tokens_sum"]
                ),
                "adaptive_budget_effective_openai_max_tokens_count": (
                    adaptive_budget_policy_metrics["effective_openai_max_tokens_count"]
                ),
                "syntax_preflight_enabled_attempts": syntax_preflight_metrics["enabled_attempts"],
                "syntax_preflight_cycles_sum": syntax_preflight_metrics["cycles_sum"],
                "syntax_preflight_failures_sum": syntax_preflight_metrics["failures_sum"],
                "syntax_preflight_failed_attempts": syntax_preflight_metrics["failed_attempts"],
                "patch_verifier_attempts": patch_verifier_scope_metrics["verifier_attempts"],
                "patch_verifier_source_scope_matches": patch_verifier_scope_metrics[
                    "source_scope_matches"
                ],
                "patch_verifier_recovery_eligible_attempts": patch_verifier_scope_metrics[
                    "recovery_eligible_attempts"
                ],
                "patch_verifier_recovery_invoked_attempts": patch_verifier_scope_metrics[
                    "recovery_invoked_attempts"
                ],
                "patch_verifier_recovery_passed_attempts": patch_verifier_scope_metrics[
                    "recovery_passed_attempts"
                ],
                "patch_verifier_recovery_result_counts": patch_verifier_scope_metrics[
                    "recovery_result_counts"
                ],
                "patch_verifier_recovery_terminal_note_counts": patch_verifier_scope_metrics[
                    "recovery_terminal_note_counts"
                ],
                "retry_history_attempts": retry_history_stats["attempts"],
                "retry_history_chars_sum": retry_history_stats["chars_sum"],
                "retry_history_turns_sum": retry_history_stats["turns_sum"],
                "retry_history_budget_applied_attempts": retry_history_stats[
                    "budget_applied_attempts"
                ],
                "retry_history_truncated_attempts": retry_history_stats["truncated_attempts"],
            }
        )
    return rows


def _records_by_case_condition(
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]],
) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    """Index deduped records by (run_id, case_id, condition) key."""
    out: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for (run_id, case_id, condition), records in grouped.items():
        out[(str(run_id), str(case_id), str(condition))] = _dedupe_records_by_attempt(records)
    return out


def _records_for_rows(
    rows: list[dict[str, Any]],
    *,
    records_by_case_condition: dict[tuple[str, str, str], list[dict[str, Any]]] | None,
) -> list[dict[str, Any]]:
    """Collect raw records corresponding to the given derived rows."""
    if not records_by_case_condition:
        return []
    out: list[dict[str, Any]] = []
    for row in rows:
        key = (
            str(row.get("run_id") or ""),
            str(row.get("case_id") or ""),
            str(row.get("condition") or ""),
        )
        out.extend(records_by_case_condition.get(key, []))
    return out


def _success_within_attempts(rows: list[dict[str, Any]], k: int | None) -> float | None:
    """Share of rows solved within the first *k* attempts (STA-13, D12)."""
    if k is None or not rows:
        return None
    return sum(1 for row in rows if _solved_within(row, k)) / len(rows)


def _case_success_rates(
    rows: list[dict[str, Any]], *, within_attempts: int | None = None
) -> dict[str, float]:
    """Success rate per case_id over its runs (1.0/0.0 for a single run).

    With *within_attempts*, a run counts as solved only when it was solved within that
    many attempts, so conditions with different budgets are compared at the smaller one
    (STA-13, D12).
    """
    outcomes: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        outcomes[str(row.get("case_id") or "")].append(_solved_within(row, within_attempts))
    return {case_id: sum(values) / len(values) for case_id, values in outcomes.items()}


def _matched_uplift(
    case_rates: dict[str, dict[str, float]], a_condition: str, b_condition: str
) -> float | None:
    """Success uplift (B - A) over the cases both conditions ran (STA-03).

    Raw per-condition rates over different case sets used to be subtracted: A on 10
    cases at 30% and B on the 5 easiest at 60% printed +30pp where the matched
    uplift is 0.
    """
    a_rates = case_rates.get(a_condition)
    b_rates = case_rates.get(b_condition)
    if not a_rates or not b_rates:
        return None
    matched = sorted(set(a_rates) & set(b_rates))
    if not matched:
        return None
    return sum(b_rates[case_id] - a_rates[case_id] for case_id in matched) / len(matched)


def _pairwise_uplifts(
    case_rates: dict[str, dict[str, float]],
    *,
    own_budget_rates: dict[str, dict[str, float]] | None = None,
    common_attempt_k: int | None = None,
) -> list[dict[str, Any]]:
    """Pairwise success-rate uplifts over each pair's matched cases (STA-03).

    *case_rates* carries the compared outcomes (at ``common_attempt_k`` when known);
    *own_budget_rates* adds each condition's own-budget uplift as a descriptive (STA-13).
    """
    own_budget_rates = case_rates if own_budget_rates is None else own_budget_rates
    conditions = sorted(condition for condition, rates in case_rates.items() if rates)
    out: list[dict[str, Any]] = []
    for idx, condition_a in enumerate(conditions):
        for condition_b in conditions[idx + 1 :]:
            a_cases = set(case_rates[condition_a])
            b_cases = set(case_rates[condition_b])
            out.append(
                {
                    "condition_a": condition_a,
                    "condition_b": condition_b,
                    "uplift_success_rate": _matched_uplift(case_rates, condition_a, condition_b),
                    "uplift_success_rate_own_budget": _matched_uplift(
                        own_budget_rates, condition_a, condition_b
                    ),
                    "attempt_budget_basis": (
                        "common_k" if common_attempt_k is not None else "own_budget"
                    ),
                    "common_attempt_k": common_attempt_k,
                    "n_matched_cases": len(a_cases & b_cases),
                    "case_sets_match": a_cases == b_cases,
                }
            )
    return out
