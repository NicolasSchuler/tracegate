"""Summarize a bucket of case-condition rows."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from evals.analysis_adaptive_rollup import _adaptive_rollup_metrics, _has_adaptive_telemetry
from evals.analysis_numeric import _increment_string_count, _mean, _median, _ratio
from evals.analysis_ope_support import (
    ADAPTIVE_OPE_MIN_ACTION_SUPPORT,
    ADAPTIVE_OPE_PROPENSITY_FLOOR_DEFAULT,
)
from evals.analysis_rows import (
    _case_success_rates,
    _matched_uplift,
    _pairwise_uplifts,
    _success_within_attempts,
)
from evals.analysis_usage import (
    _condition_timing_breakdown_mean,
    _condition_token_breakdown_mean,
    _is_external_baseline_condition,
    _mean_prompt_chars_by_condition,
)
from evals.stats import DEFAULT_BOOTSTRAP_SEED, comparison_eligible


def _summarize_bucket(
    rows: list[dict[str, Any]],
    *,
    records_by_case_condition: dict[tuple[str, str, str], list[dict[str, Any]]] | None = None,
    adaptive_ope_min_action_support: int = ADAPTIVE_OPE_MIN_ACTION_SUPPORT,
    adaptive_ope_propensity_floor: float = ADAPTIVE_OPE_PROPENSITY_FLOOR_DEFAULT,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Compute per-condition success rates and aggregate metrics for a bucket.

    Produces a dict keyed by condition name, each containing success
    rates, token usage, runtime, helper, online-policy, budget-policy,
    OPE summaries, and pairwise uplift comparisons.
    """
    min_action_support = max(1, int(adaptive_ope_min_action_support))
    by_condition: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_condition[str(row["condition"])].append(row)
    # STA-03: every condition is compared on the cases both ran; the bucket's expected
    # case set is every case any condition ran.
    cases_expected = {str(row.get("case_id") or "") for row in rows}
    # STA-13 / D12: conditions with different attempt budgets are compared at the
    # smallest budget among them, as success within the first K attempts.
    attempt_budgets = [
        int(row.get("attempts_budget") or 0) for row in rows if int(row.get("attempts_budget") or 0)
    ]
    common_attempt_k = min(attempt_budgets) if attempt_budgets else None
    case_rates = {
        condition: _case_success_rates([row for row in condition_rows if comparison_eligible(row)])
        for condition, condition_rows in by_condition.items()
    }

    out: dict[str, Any] = {}
    for condition, all_condition_rows in sorted(by_condition.items()):
        # BAS-01 / D3: baselines are scored on the cases they can represent; the excluded
        # count is reported so the comparison's N stays explicit.
        condition_rows = [row for row in all_condition_rows if comparison_eligible(row)]
        baseline_unsupported_cases = len(all_condition_rows) - len(condition_rows)
        successes = sum(1 for row in condition_rows if bool(row["success"]))
        cases = len(condition_rows)
        success_rate = (successes / cases) if cases else 0.0
        condition_case_ids = {str(row.get("case_id") or "") for row in all_condition_rows}
        successful_attempts = [
            int(row["attempts_to_success"])
            for row in condition_rows
            if row["attempts_to_success"] is not None and bool(row["success"])
        ]
        token_values = [
            float(row["token_usage"]) for row in condition_rows if row["token_usage"] is not None
        ]
        token_prompt_values = [
            float(row.get("token_usage_prompt"))
            for row in condition_rows
            if row.get("token_usage_prompt") is not None
        ]
        token_completion_values = [
            float(row.get("token_usage_completion"))
            for row in condition_rows
            if row.get("token_usage_completion") is not None
        ]
        token_total_values = [
            float(row.get("token_usage_total"))
            for row in condition_rows
            if row.get("token_usage_total") is not None
        ]
        token_reasoning_values = [
            float(row.get("token_usage_reasoning"))
            for row in condition_rows
            if bool(row.get("reasoning_token_usage_available"))
            and row.get("token_usage_reasoning") is not None
        ]
        token_per_attempt_values = [
            float(row.get("token_usage_per_attempt_mean"))
            for row in condition_rows
            if row.get("token_usage_per_attempt_mean") is not None
        ]
        reasoning_token_per_attempt_values = [
            float(row.get("reasoning_token_usage_per_attempt_mean"))
            for row in condition_rows
            if bool(row.get("reasoning_token_usage_available"))
            and row.get("reasoning_token_usage_per_attempt_mean") is not None
        ]
        runtime_values = [
            float(row["runtime_sec"]) for row in condition_rows if row["runtime_sec"] is not None
        ]
        attempts_budget_values = [
            float(row["attempts_budget"])
            for row in condition_rows
            if isinstance(row.get("attempts_budget"), (int, float))
        ]
        failed_attempt_values = [
            float(row["attempts_executed"])
            for row in condition_rows
            if (not bool(row.get("success")))
            and isinstance(row.get("attempts_executed"), (int, float))
        ]
        successful_attempt_values = [
            float(row["attempts_to_success"])
            for row in condition_rows
            if bool(row.get("success")) and isinstance(row.get("attempts_to_success"), (int, float))
        ]
        success_attempt_ratio_values = [
            float(row.get("success_attempt_ratio"))
            for row in condition_rows
            if row.get("success_attempt_ratio") is not None
        ]
        llm_call_values = [
            float(row["llm_calls_total"])
            for row in condition_rows
            if isinstance(row.get("llm_calls_total"), (int, float))
        ]
        is_external_baseline = _is_external_baseline_condition(condition)
        baseline_teacher_duration_values = (
            [
                float(row["baseline_teacher_duration_sec_total"])
                for row in condition_rows
                if isinstance(row.get("baseline_teacher_duration_sec_total"), (int, float))
            ]
            if is_external_baseline
            else []
        )
        baseline_learner_duration_values = (
            [
                float(row["baseline_learner_duration_sec_total"])
                for row in condition_rows
                if isinstance(row.get("baseline_learner_duration_sec_total"), (int, float))
            ]
            if is_external_baseline
            else []
        )
        baseline_postcheck_duration_values = (
            [
                float(row["baseline_postcheck_duration_sec_total"])
                for row in condition_rows
                if isinstance(row.get("baseline_postcheck_duration_sec_total"), (int, float))
            ]
            if is_external_baseline
            else []
        )
        baseline_learner_empty_response_counts = (
            [
                float(row["baseline_learner_empty_response_count"])
                for row in condition_rows
                if isinstance(row.get("baseline_learner_empty_response_count"), (int, float))
            ]
            if is_external_baseline
            else []
        )
        baseline_attempts_with_postcheck_counts = (
            [
                float(row["baseline_attempts_with_postcheck_count"])
                for row in condition_rows
                if isinstance(row.get("baseline_attempts_with_postcheck_count"), (int, float))
            ]
            if is_external_baseline
            else []
        )
        baseline_direct_gen_passed_values = (
            [
                1.0 if bool(row.get("baseline_direct_gen_passed")) else 0.0
                for row in condition_rows
                if isinstance(row.get("baseline_direct_gen_passed"), bool)
            ]
            if is_external_baseline
            else []
        )
        baseline_debug_added_value_values = (
            [
                1.0 if bool(row.get("baseline_debug_added_value")) else 0.0
                for row in condition_rows
                if isinstance(row.get("baseline_debug_added_value"), bool)
            ]
            if is_external_baseline
            else []
        )
        baseline_method_instrumentation_values = (
            [
                1.0 if bool(row.get("baseline_method_instrumentation")) else 0.0
                for row in condition_rows
                if isinstance(row.get("baseline_method_instrumentation"), bool)
            ]
            if is_external_baseline
            else []
        )
        baseline_method_two_step_values = (
            [
                1.0 if bool(row.get("baseline_method_two_step_repair")) else 0.0
                for row in condition_rows
                if isinstance(row.get("baseline_method_two_step_repair"), bool)
            ]
            if is_external_baseline
            else []
        )
        baseline_is_passing_values = (
            [
                1.0 if bool(row.get("baseline_is_passing")) else 0.0
                for row in condition_rows
                if isinstance(row.get("baseline_is_passing"), bool)
            ]
            if is_external_baseline
            else []
        )
        baseline_feedback_improvement_values = (
            [
                float(row["baseline_test_feedback_improvement_steps"])
                for row in condition_rows
                if isinstance(row.get("baseline_test_feedback_improvement_steps"), (int, float))
            ]
            if is_external_baseline
            else []
        )
        baseline_feedback_regression_values = (
            [
                float(row["baseline_test_feedback_regression_steps"])
                for row in condition_rows
                if isinstance(row.get("baseline_test_feedback_regression_steps"), (int, float))
            ]
            if is_external_baseline
            else []
        )
        baseline_feedback_final_fail_values = (
            [
                float(row["baseline_test_feedback_final_fail_count"])
                for row in condition_rows
                if isinstance(row.get("baseline_test_feedback_final_fail_count"), (int, float))
            ]
            if is_external_baseline
            else []
        )
        baseline_terminal_timeout_values = (
            [
                1.0 if bool(row.get("baseline_terminal_timeout")) else 0.0
                for row in condition_rows
                if isinstance(row.get("baseline_terminal_timeout"), bool)
            ]
            if is_external_baseline
            else []
        )
        baseline_attempt_summary_count_total = (
            sum(int(row.get("baseline_attempt_summary_count") or 0) for row in condition_rows)
            if is_external_baseline
            else 0
        )
        baseline_iteration_budget_exhausted_attempts_total = (
            sum(
                int(row.get("baseline_iteration_budget_exhausted_attempt_count") or 0)
                for row in condition_rows
            )
            if is_external_baseline
            else 0
        )
        baseline_no_write_buggy_py_attempts_total = (
            sum(
                int(row.get("baseline_no_write_buggy_py_attempt_count") or 0)
                for row in condition_rows
            )
            if is_external_baseline
            else 0
        )
        baseline_no_write_buggy_py_attempts_covered = (
            sum(
                int(row.get("baseline_no_write_buggy_py_attempt_covered") or 0)
                for row in condition_rows
            )
            if is_external_baseline
            else 0
        )
        baseline_no_test_attempts_total = (
            sum(int(row.get("baseline_no_test_attempt_count") or 0) for row in condition_rows)
            if is_external_baseline
            else 0
        )
        baseline_no_test_attempts_covered = (
            sum(int(row.get("baseline_no_test_attempt_covered") or 0) for row in condition_rows)
            if is_external_baseline
            else 0
        )
        baseline_subcall_count_total = (
            sum(int(row.get("baseline_subcall_count_total") or 0) for row in condition_rows)
            if is_external_baseline
            else 0
        )
        baseline_subcall_count_attempts_covered = (
            sum(int(row.get("baseline_subcall_count_covered") or 0) for row in condition_rows)
            if is_external_baseline
            else 0
        )
        baseline_test_run_count_total = (
            sum(int(row.get("baseline_test_run_count_total") or 0) for row in condition_rows)
            if is_external_baseline
            else 0
        )
        baseline_test_run_count_attempts_covered = (
            sum(int(row.get("baseline_test_run_count_covered") or 0) for row in condition_rows)
            if is_external_baseline
            else 0
        )
        stopped_reason_counts: dict[str, int] = {}
        fidelity_counts: dict[str, int] = {}
        repair_scope_counts: dict[str, int] = {}
        prompt_contract_counts: dict[str, int] = {}
        hybrid_guidance_mode_counts: dict[str, int] = {}
        hybrid_guidance_section_counts: dict[str, int] = {}
        output_contract_counts: dict[str, int] = {}
        upstream_commit_counts: dict[str, int] = {}
        contract_version_counts: dict[str, int] = {}
        hybrid_guidance_chars_values: list[float] = []
        if is_external_baseline:
            for row in condition_rows:
                value = row.get("baseline_stopped_reason")
                if isinstance(value, str) and value:
                    stopped_reason_counts[value] = int(stopped_reason_counts.get(value, 0) + 1)
                _increment_string_count(fidelity_counts, row.get("baseline_fidelity_level"))
                _increment_string_count(
                    repair_scope_counts, row.get("baseline_repair_scope_policy")
                )
                _increment_string_count(
                    prompt_contract_counts, row.get("baseline_prompt_contract_kind")
                )
                _increment_string_count(
                    hybrid_guidance_mode_counts, row.get("baseline_hybrid_guidance_mode")
                )
                guidance_chars = row.get("baseline_hybrid_guidance_chars")
                if isinstance(guidance_chars, (int, float)):
                    hybrid_guidance_chars_values.append(float(guidance_chars))
                sections_value = row.get("baseline_hybrid_guidance_sections")
                if isinstance(sections_value, list):
                    for section_name in sections_value:
                        _increment_string_count(hybrid_guidance_section_counts, section_name)
                _increment_string_count(output_contract_counts, row.get("baseline_output_contract"))
                _increment_string_count(upstream_commit_counts, row.get("baseline_upstream_commit"))
                _increment_string_count(
                    contract_version_counts, row.get("baseline_contract_version")
                )
        baseline_direct_gen_pass_rate_covered_cases = len(baseline_direct_gen_passed_values)
        baseline_debug_added_value_rate_covered_cases = len(baseline_debug_added_value_values)
        baseline_method_instrumentation_rate_covered_cases = len(
            baseline_method_instrumentation_values
        )
        baseline_method_two_step_repair_rate_covered_cases = len(baseline_method_two_step_values)
        baseline_is_passing_rate_covered_cases = len(baseline_is_passing_values)
        baseline_terminal_timeout_rate_covered_cases = len(baseline_terminal_timeout_values)
        baseline_feedback_improved_case_rate_covered_cases = len(
            baseline_feedback_improvement_values
        )
        baseline_feedback_regressed_case_rate_covered_cases = len(
            baseline_feedback_regression_values
        )
        total_patch_attempts = sum(int(row.get("patch_attempts") or 0) for row in condition_rows)
        total_test_touch_attempts = sum(
            int(row.get("test_touch_attempts") or 0) for row in condition_rows
        )
        solve_distribution: dict[str, int] = {}
        for row in condition_rows:
            if not bool(row.get("success")):
                continue
            value = row.get("attempts_to_success")
            if value is None:
                continue
            key = str(int(value))
            solve_distribution[key] = int(solve_distribution.get(key, 0) + 1)
        out[condition] = {
            "cases": cases,
            "successes": successes,
            "success_rate": success_rate,
            "common_attempt_k": common_attempt_k,
            "success_within_common_k": _success_within_attempts(condition_rows, common_attempt_k),
            "cases_expected": len(cases_expected),
            "cases_missing": len(cases_expected - condition_case_ids),
            "baseline_unsupported_cases": baseline_unsupported_cases,
            "median_attempts_to_success": _median(successful_attempts),
            "mean_attempts_to_success": _mean(successful_attempt_values),
            "mean_attempts_when_failed": _mean(failed_attempt_values),
            "mean_attempts_budget": _mean(attempts_budget_values),
            "mean_attempt_ratio_when_successful": _mean(success_attempt_ratio_values),
            "solve_on_attempt_distribution": solve_distribution,
            "mean_token_usage": _mean(token_values),
            "mean_tokens_prompt": _mean(token_prompt_values),
            "mean_tokens_completion": _mean(token_completion_values),
            "mean_tokens_total": _mean(token_total_values),
            "mean_tokens_reasoning": _mean(token_reasoning_values),
            "mean_tokens_per_attempt": _mean(token_per_attempt_values),
            "mean_reasoning_tokens_per_attempt": _mean(reasoning_token_per_attempt_values),
            "mean_llm_calls_per_case": _mean(llm_call_values),
            "token_covered_cases": sum(
                1 for row in condition_rows if row["token_usage"] is not None
            ),
            "token_covered_attempts": sum(
                int(row.get("token_covered_attempts") or 0) for row in condition_rows
            ),
            "token_covered_fraction": _ratio(
                sum(int(row.get("token_covered_attempts") or 0) for row in condition_rows),
                sum(int(row.get("attempts_executed") or 0) for row in condition_rows),
            ),
            "reasoning_token_covered_cases": sum(
                1 for row in condition_rows if bool(row.get("reasoning_token_usage_available"))
            ),
            "reasoning_token_coverage_rate": _ratio(
                sum(
                    1 for row in condition_rows if bool(row.get("reasoning_token_usage_available"))
                ),
                cases,
            ),
            "mean_runtime_sec": _mean(runtime_values),
            "mean_timing_breakdown": _condition_timing_breakdown_mean(condition_rows),
            "mean_token_breakdown": _condition_token_breakdown_mean(condition_rows),
            "baseline_teacher_duration_sec_total_mean": _mean(baseline_teacher_duration_values),
            "baseline_learner_duration_sec_total_mean": _mean(baseline_learner_duration_values),
            "baseline_postcheck_duration_sec_total_mean": _mean(baseline_postcheck_duration_values),
            "baseline_learner_empty_response_count_mean": _mean(
                baseline_learner_empty_response_counts
            ),
            "baseline_attempts_with_postcheck_count_mean": _mean(
                baseline_attempts_with_postcheck_counts
            ),
            "baseline_direct_gen_pass_rate": _mean(baseline_direct_gen_passed_values),
            "baseline_direct_gen_pass_rate_covered_cases": baseline_direct_gen_pass_rate_covered_cases,
            "baseline_direct_gen_pass_rate_coverage_rate": _ratio(
                baseline_direct_gen_pass_rate_covered_cases, cases
            ),
            "baseline_debug_added_value_rate": _mean(baseline_debug_added_value_values),
            "baseline_debug_added_value_rate_covered_cases": (
                baseline_debug_added_value_rate_covered_cases
            ),
            "baseline_debug_added_value_rate_coverage_rate": _ratio(
                baseline_debug_added_value_rate_covered_cases, cases
            ),
            "baseline_method_instrumentation_rate": _mean(baseline_method_instrumentation_values),
            "baseline_method_instrumentation_rate_covered_cases": (
                baseline_method_instrumentation_rate_covered_cases
            ),
            "baseline_method_instrumentation_rate_coverage_rate": _ratio(
                baseline_method_instrumentation_rate_covered_cases, cases
            ),
            "baseline_method_two_step_repair_rate": _mean(baseline_method_two_step_values),
            "baseline_method_two_step_repair_rate_covered_cases": (
                baseline_method_two_step_repair_rate_covered_cases
            ),
            "baseline_method_two_step_repair_rate_coverage_rate": _ratio(
                baseline_method_two_step_repair_rate_covered_cases, cases
            ),
            "baseline_is_passing_rate": _mean(baseline_is_passing_values),
            "baseline_is_passing_rate_covered_cases": baseline_is_passing_rate_covered_cases,
            "baseline_is_passing_rate_coverage_rate": _ratio(
                baseline_is_passing_rate_covered_cases, cases
            ),
            "baseline_terminal_timeout_rate": _mean(baseline_terminal_timeout_values),
            "baseline_terminal_timeout_rate_covered_cases": (
                baseline_terminal_timeout_rate_covered_cases
            ),
            "baseline_terminal_timeout_rate_coverage_rate": _ratio(
                baseline_terminal_timeout_rate_covered_cases, cases
            ),
            "baseline_feedback_improvement_steps_mean": _mean(baseline_feedback_improvement_values),
            "baseline_feedback_regression_steps_mean": _mean(baseline_feedback_regression_values),
            "baseline_feedback_improved_case_rate": _mean(
                [1.0 if value > 0.0 else 0.0 for value in baseline_feedback_improvement_values]
            ),
            "baseline_feedback_improved_case_rate_covered_cases": (
                baseline_feedback_improved_case_rate_covered_cases
            ),
            "baseline_feedback_improved_case_rate_coverage_rate": _ratio(
                baseline_feedback_improved_case_rate_covered_cases, cases
            ),
            "baseline_feedback_regressed_case_rate": _mean(
                [1.0 if value > 0.0 else 0.0 for value in baseline_feedback_regression_values]
            ),
            "baseline_feedback_regressed_case_rate_covered_cases": (
                baseline_feedback_regressed_case_rate_covered_cases
            ),
            "baseline_feedback_regressed_case_rate_coverage_rate": _ratio(
                baseline_feedback_regressed_case_rate_covered_cases, cases
            ),
            "baseline_feedback_final_fail_count_mean": _mean(baseline_feedback_final_fail_values),
            "baseline_attempt_summary_count_total": baseline_attempt_summary_count_total,
            "baseline_iteration_budget_exhausted_attempts_total": (
                baseline_iteration_budget_exhausted_attempts_total
            ),
            "baseline_iteration_budget_exhausted_attempt_rate": _ratio(
                baseline_iteration_budget_exhausted_attempts_total,
                baseline_attempt_summary_count_total,
            ),
            "baseline_no_write_buggy_py_attempts_total": baseline_no_write_buggy_py_attempts_total,
            "baseline_no_write_buggy_py_attempts_covered": (
                baseline_no_write_buggy_py_attempts_covered
            ),
            "baseline_no_write_buggy_py_attempt_rate": _ratio(
                baseline_no_write_buggy_py_attempts_total,
                baseline_no_write_buggy_py_attempts_covered,
            ),
            "baseline_no_test_attempts_total": baseline_no_test_attempts_total,
            "baseline_no_test_attempts_covered": baseline_no_test_attempts_covered,
            "baseline_no_test_attempt_rate": _ratio(
                baseline_no_test_attempts_total,
                baseline_no_test_attempts_covered,
            ),
            "baseline_subcall_count_total": baseline_subcall_count_total,
            "baseline_subcall_count_attempts_covered": baseline_subcall_count_attempts_covered,
            "baseline_subcall_count_per_attempt": (
                baseline_subcall_count_total / baseline_subcall_count_attempts_covered
                if baseline_subcall_count_attempts_covered > 0
                else None
            ),
            "baseline_test_run_count_total": baseline_test_run_count_total,
            "baseline_test_run_count_attempts_covered": (baseline_test_run_count_attempts_covered),
            "baseline_test_run_count_per_attempt": (
                baseline_test_run_count_total / baseline_test_run_count_attempts_covered
                if baseline_test_run_count_attempts_covered > 0
                else None
            ),
            "baseline_stopped_reason_counts": stopped_reason_counts,
            "baseline_fidelity_level_counts": fidelity_counts,
            "baseline_repair_scope_policy_counts": repair_scope_counts,
            "baseline_prompt_contract_kind_counts": prompt_contract_counts,
            "baseline_hybrid_guidance_mode_counts": hybrid_guidance_mode_counts,
            "baseline_hybrid_guidance_section_counts": hybrid_guidance_section_counts,
            "baseline_hybrid_guidance_chars_mean": _mean(hybrid_guidance_chars_values),
            "baseline_output_contract_counts": output_contract_counts,
            "baseline_upstream_commit_counts": upstream_commit_counts,
            "baseline_contract_version_counts": contract_version_counts,
            "patch_attempts": total_patch_attempts,
            "test_touch_attempts": total_test_touch_attempts,
            "test_touch_attempt_rate": (
                (total_test_touch_attempts / total_patch_attempts)
                if total_patch_attempts > 0
                else None
            ),
            "relaxed_apply_attempts": sum(
                int(row.get("relaxed_apply_attempts") or 0) for row in condition_rows
            ),
            "test_surface_modified_attempts": sum(
                int(row.get("test_surface_modified_attempts") or 0) for row in condition_rows
            ),
            **_adaptive_rollup_metrics(
                condition_rows,
                records_by_case_condition=records_by_case_condition,
                adaptive_ope_min_action_support=min_action_support,
                adaptive_ope_propensity_floor=adaptive_ope_propensity_floor,
                include_condition_only_counts=True,
                bootstrap_seed=bootstrap_seed,
            ),
        }
    out["condition_matrix"] = [
        {"condition": condition, **dict(stats)}
        for condition, stats in sorted(
            (
                (condition, stats)
                for condition, stats in out.items()
                if isinstance(stats, dict) and "cases" in stats and "success_rate" in stats
            ),
            key=lambda item: item[0],
        )
    ]
    out["pairwise_uplifts"] = _pairwise_uplifts(case_rates)

    out["uplift_success_rate"] = _matched_uplift(case_rates, "base_repair", "layered_adaptive_llm")
    stage_b_prompt_values = [
        float(row["stage_b_prompt_chars_mean"])
        for row in rows
        if row.get("stage_b_prompt_chars_mean") is not None
    ]
    out["mean_prompt_chars_stage_b"] = _mean(stage_b_prompt_values)
    out["mean_prompt_chars_stage_b_by_condition"] = _mean_prompt_chars_by_condition(rows)
    retry_history_attempts = sum(int(row.get("retry_history_attempts") or 0) for row in rows)
    retry_history_chars_sum = sum(float(row.get("retry_history_chars_sum") or 0.0) for row in rows)
    retry_history_turns_sum = sum(float(row.get("retry_history_turns_sum") or 0.0) for row in rows)
    retry_history_budget_applied_attempts = sum(
        int(row.get("retry_history_budget_applied_attempts") or 0) for row in rows
    )
    retry_history_truncated_attempts = sum(
        int(row.get("retry_history_truncated_attempts") or 0) for row in rows
    )
    out["mean_retry_history_chars"] = (
        (retry_history_chars_sum / retry_history_attempts) if retry_history_attempts > 0 else None
    )
    out["mean_retry_history_turns"] = (
        (retry_history_turns_sum / retry_history_attempts) if retry_history_attempts > 0 else None
    )
    out["retry_history_budget_applied_rate"] = (
        (retry_history_budget_applied_attempts / retry_history_attempts)
        if retry_history_attempts > 0
        else None
    )
    out["retry_history_truncation_rate"] = (
        (retry_history_truncated_attempts / retry_history_attempts)
        if retry_history_attempts > 0
        else None
    )

    stage_a_attempts = sum(int(row.get("stage_a_request_attempts") or 0) for row in rows)
    stage_a_successes = sum(int(row.get("stage_a_request_successes") or 0) for row in rows)
    out["stage_a_request_success_rate"] = (
        (stage_a_successes / stage_a_attempts) if stage_a_attempts > 0 else None
    )
    retry_gate_attempts = sum(int(row.get("retry_gate_attempts") or 0) for row in rows)
    retry_gate_blocks = sum(int(row.get("retry_gate_blocks") or 0) for row in rows)
    out["retry_gate_block_rate"] = (
        (retry_gate_blocks / retry_gate_attempts) if retry_gate_attempts > 0 else None
    )
    total_patch_attempts = sum(int(row.get("patch_attempts") or 0) for row in rows)
    total_test_touch_attempts = sum(int(row.get("test_touch_attempts") or 0) for row in rows)
    out["patch_attempts"] = total_patch_attempts
    out["test_touch_attempts"] = total_test_touch_attempts
    out["relaxed_apply_attempts"] = sum(int(row.get("relaxed_apply_attempts") or 0) for row in rows)
    out["test_surface_modified_attempts"] = sum(
        int(row.get("test_surface_modified_attempts") or 0) for row in rows
    )
    out["test_touch_attempt_rate"] = (
        (total_test_touch_attempts / total_patch_attempts) if total_patch_attempts > 0 else None
    )
    # STA-16: bucket-level adaptive rates count only rows with adaptive telemetry; the
    # non-adaptive conditions used to dilute them.
    adaptive_rows = [row for row in rows if _has_adaptive_telemetry(row)]
    out.update(
        _adaptive_rollup_metrics(
            adaptive_rows,
            records_by_case_condition=records_by_case_condition,
            adaptive_ope_min_action_support=min_action_support,
            adaptive_ope_propensity_floor=adaptive_ope_propensity_floor,
            include_condition_only_counts=False,
            bootstrap_seed=bootstrap_seed,
        )
    )
    out["adaptive_rollup_conditions"] = sorted({str(row["condition"]) for row in adaptive_rows})
    out["adaptive_rollup_cases"] = len(adaptive_rows)

    staged_rows = [
        row
        for row in rows
        if row.get("condition") == "layered_adaptive_llm"
        and row.get("context_protocol") == "staged"
    ]
    monolithic_rows = [
        row
        for row in rows
        if row.get("condition") == "layered_adaptive_llm"
        and row.get("context_protocol") == "monolithic"
    ]
    if staged_rows and monolithic_rows:
        staged_rate = sum(1 for row in staged_rows if bool(row.get("success"))) / len(staged_rows)
        monolithic_rate = sum(1 for row in monolithic_rows if bool(row.get("success"))) / len(
            monolithic_rows
        )
        out["uplift_staged_vs_monolithic"] = staged_rate - monolithic_rate
    else:
        out["uplift_staged_vs_monolithic"] = None
    out["uplift_layered_gated_vs_base"] = _matched_uplift(
        case_rates, "base_repair", "layered_adaptive_gated"
    )
    out["uplift_layered_llm_vs_base"] = _matched_uplift(
        case_rates, "base_repair", "layered_adaptive_llm"
    )
    return out
