"""Plain-text output of the analysis: summaries, buckets, OPE targets and input errors."""

from __future__ import annotations

from typing import Any

from evals.analysis_numeric import _fmt_num, _fmt_num_with_ci, _fmt_percent, _fmt_signed_percent
from evals.analysis_ope import (
    ADAPTIVE_BUDGET_TARGET_POLICY_ORDER,
    ADAPTIVE_ONLINE_TARGET_POLICY_ORDER,
)


def _print_summary(summary: dict[str, Any]) -> None:
    """Print the human-readable evaluation summary to stdout."""
    print("Overall:")
    _print_bucket(summary["overall"])

    print("\nBy category:")
    by_category = summary["by_category"]
    for category, bucket in by_category.items():
        print(f"- {category}:")
        _print_bucket(bucket, prefix="  ")

    print("\nBy difficulty:")
    by_difficulty = summary["by_difficulty"]
    for difficulty, bucket in by_difficulty.items():
        print(f"- {difficulty}:")
        _print_bucket(bucket, prefix="  ")

    print("\nBy snapshot dependency:")
    by_snapshot_dependency = summary["by_snapshot_dependency"]
    for dependency, bucket in by_snapshot_dependency.items():
        print(f"- {dependency}:")
        _print_bucket(bucket, prefix="  ")

    if "by_bug_source" in summary:
        print("\nBy bug source:")
        for source, bucket in summary["by_bug_source"].items():
            print(f"- {source}:")
            _print_bucket(bucket, prefix="  ")

    if "by_contamination_risk" in summary:
        print("\nBy contamination risk:")
        for risk, bucket in summary["by_contamination_risk"].items():
            print(f"- {risk}:")
            _print_bucket(bucket, prefix="  ")

    _print_input_errors(summary.get("input_errors"))


def _print_bucket(bucket: dict[str, Any], prefix: str = "") -> None:
    """Print a single summary bucket with conditions, uplifts, and controls."""
    condition_items = [
        (condition, stats)
        for condition, stats in sorted(bucket.items())
        if isinstance(stats, dict) and "cases" in stats and "success_rate" in stats
    ]
    for condition, stats in condition_items:
        cases = int(stats.get("cases", 0))
        successes = int(stats.get("successes", 0))
        rate = float(stats.get("success_rate", 0.0))
        med_attempts = stats.get("median_attempts_to_success")
        mean_tokens = stats.get("mean_tokens_total", stats.get("mean_token_usage"))
        mean_prompt_tokens = stats.get("mean_tokens_prompt")
        mean_completion_tokens = stats.get("mean_tokens_completion")
        mean_reasoning_tokens = stats.get("mean_tokens_reasoning")
        mean_tokens_per_attempt = stats.get("mean_tokens_per_attempt")
        mean_reasoning_tokens_per_attempt = stats.get("mean_reasoning_tokens_per_attempt")
        mean_llm_calls = stats.get("mean_llm_calls_per_case")
        reasoning_token_covered_cases = int(stats.get("reasoning_token_covered_cases", 0) or 0)
        reasoning_token_coverage_rate = stats.get("reasoning_token_coverage_rate")
        mean_attempt_budget = stats.get("mean_attempts_budget")
        mean_attempt_ratio = stats.get("mean_attempt_ratio_when_successful")
        solve_dist = stats.get("solve_on_attempt_distribution")
        solve_dist_rendered = ""
        if isinstance(solve_dist, dict) and solve_dist:

            def _solve_key(entry: tuple[str, Any]) -> int:
                try:
                    return int(entry[0])
                except (TypeError, ValueError):
                    return 0

            solve_dist_rendered = ", ".join(
                f"{k}:{int(v)}" for k, v in sorted(solve_dist.items(), key=_solve_key)
            )
        mean_runtime = stats.get("mean_runtime_sec")
        patch_attempts = int(stats.get("patch_attempts", 0) or 0)
        test_touch_attempts = int(stats.get("test_touch_attempts", 0) or 0)
        test_touch_attempt_rate = stats.get("test_touch_attempt_rate")
        helper_invoked_cases = int(stats.get("helper_invoked_cases", 0) or 0)
        helper_case_rate = stats.get("helper_invocation_case_rate")
        helper_attempt_rate = stats.get("helper_invocation_attempt_rate")
        helper_materialized_cases = int(stats.get("helper_materialized_cases", 0) or 0)
        helper_materialized_case_rate = stats.get("helper_materialized_case_rate")
        helper_materialized_attempt_rate = stats.get("helper_materialized_attempt_rate")
        helper_low_yield_attempt_rate = stats.get("helper_low_yield_attempt_rate")
        helper_transport_failure_attempt_rate = stats.get("helper_transport_failure_attempt_rate")
        helper_backend_unavailable_attempt_rate = stats.get(
            "helper_backend_unavailable_attempt_rate"
        )
        helper_invalid_payload_attempt_rate = stats.get("helper_invalid_payload_attempt_rate")
        helper_request_budget_exceeded_attempt_rate = stats.get(
            "helper_request_budget_exceeded_attempt_rate"
        )
        helper_effective_success_rate = stats.get("helper_effective_success_rate")
        helper_mean_retries_per_invocation = stats.get("helper_mean_retries_per_invocation")
        helper_mean_transport_retries_per_invocation = stats.get(
            "helper_mean_transport_retries_per_invocation"
        )
        helper_success_when_invoked = stats.get("helper_success_when_invoked")
        helper_rescue_rate = stats.get("helper_rescue_rate")
        mean_helper_calls = stats.get("mean_helper_calls_used_final")
        helper_need_more_true_rate = stats.get("helper_need_more_true_rate")
        helper_budget_exhausted_rate = stats.get("helper_budget_exhausted_rate")
        helper_saturation_guard_rate = stats.get("helper_saturation_guard_rate")
        topup_case_rate = stats.get("adaptive_budget_topup_case_rate")
        topup_attempt_rate = stats.get("adaptive_budget_topup_attempt_rate")
        mean_syntax_cycles = stats.get("mean_syntax_preflight_cycles")
        syntax_fail_attempt_rate = stats.get("syntax_preflight_failed_attempt_rate")
        source_scope_match_rate = stats.get("patch_verifier_source_scope_match_rate")
        adaptive_budget_decisions = int(stats.get("adaptive_budget_decisions", 0) or 0)
        adaptive_budget_selected_rate = stats.get("adaptive_budget_selected_increase_rate")
        adaptive_budget_applied_rate = stats.get("adaptive_budget_applied_increase_rate")
        adaptive_budget_update_rate = stats.get("adaptive_budget_update_rate")
        adaptive_budget_mean_reward = stats.get("adaptive_budget_mean_reward")
        adaptive_budget_mean_multiplier = stats.get("adaptive_budget_mean_applied_multiplier")
        adaptive_budget_budget_exceeded_rate = stats.get(
            "adaptive_budget_request_budget_exceeded_rate"
        )
        adaptive_online_ope = (
            stats.get("adaptive_online_ope")
            if isinstance(stats.get("adaptive_online_ope"), dict)
            else None
        )
        adaptive_budget_ope = (
            stats.get("adaptive_budget_ope")
            if isinstance(stats.get("adaptive_budget_ope"), dict)
            else None
        )
        print(
            f"{prefix}- {condition}: {successes}/{cases} ({rate:.1%}), "
            f"within common k={stats.get('common_attempt_k')} attempts="
            f"{_fmt_percent(stats.get('success_within_common_k'))}, "
            f"missing cases={int(stats.get('cases_missing') or 0)}, "
            f"median attempts={_fmt_num(med_attempts)}, "
            f"mean budget={_fmt_num(mean_attempt_budget)}, "
            f"mean success attempt ratio={_fmt_num(mean_attempt_ratio)}, "
            f"mean tokens(total/prompt/completion)={_fmt_num(mean_tokens)}/"
            f"{_fmt_num(mean_prompt_tokens)}/{_fmt_num(mean_completion_tokens)}, "
            f"mean tokens/attempt={_fmt_num(mean_tokens_per_attempt)}, "
            f"token coverage={_fmt_percent(stats.get('token_covered_fraction'))} of attempts, "
            f"mean llm calls={_fmt_num(mean_llm_calls)}, "
            f"mean runtime={_fmt_num(mean_runtime)}s, "
            f"test-touch patches={test_touch_attempts}/{patch_attempts} "
            f"({_fmt_percent(test_touch_attempt_rate)})"
        )
        if reasoning_token_covered_cases > 0:
            print(
                f"{prefix}  reasoning tokens: mean/case={_fmt_num(mean_reasoning_tokens)}, "
                f"mean/covered-attempt={_fmt_num(mean_reasoning_tokens_per_attempt)}, "
                f"coverage={reasoning_token_covered_cases}/{cases} "
                f"({_fmt_percent(reasoning_token_coverage_rate)})"
            )
        if solve_dist_rendered:
            print(f"{prefix}  solve_on_attempt_distribution: {solve_dist_rendered}")
        if helper_invoked_cases > 0 or isinstance(helper_case_rate, (int, float)):
            print(
                f"{prefix}  helper: invoked_cases={helper_invoked_cases}/{cases} "
                f"({_fmt_percent(helper_case_rate)}), "
                f"invocation_attempt_rate={_fmt_percent(helper_attempt_rate)}, "
                f"materialized_cases={helper_materialized_cases}/{cases} "
                f"({_fmt_percent(helper_materialized_case_rate)}), "
                f"materialized_attempt_rate={_fmt_percent(helper_materialized_attempt_rate)}, "
                f"low_yield_attempt_rate={_fmt_percent(helper_low_yield_attempt_rate)}, "
                f"transport_fail_attempt_rate={_fmt_percent(helper_transport_failure_attempt_rate)}, "
                f"backend_unavailable_attempt_rate={_fmt_percent(helper_backend_unavailable_attempt_rate)}, "
                f"invalid_payload_attempt_rate={_fmt_percent(helper_invalid_payload_attempt_rate)}, "
                f"request_budget_exceeded_attempt_rate={_fmt_percent(helper_request_budget_exceeded_attempt_rate)}, "
                f"effective_success_rate={_fmt_percent(helper_effective_success_rate)}, "
                f"success_when_invoked={_fmt_percent(helper_success_when_invoked)}, "
                f"rescue_rate={_fmt_percent(helper_rescue_rate)}, "
                f"mean_calls_used={_fmt_num(mean_helper_calls)}, "
                f"mean_retries_per_invocation={_fmt_num(helper_mean_retries_per_invocation)}, "
                f"mean_transport_retries_per_invocation="
                f"{_fmt_num(helper_mean_transport_retries_per_invocation)}, "
                f"need_more_context_true_rate={_fmt_percent(helper_need_more_true_rate)}, "
                f"budget_exhausted_rate={_fmt_percent(helper_budget_exhausted_rate)}, "
                f"saturation_guard_rate={_fmt_percent(helper_saturation_guard_rate)}"
            )
        if adaptive_budget_decisions > 0:
            print(
                f"{prefix}  budget_policy: decisions={adaptive_budget_decisions}, "
                f"selected_increase_rate={_fmt_percent(adaptive_budget_selected_rate)}, "
                f"applied_increase_rate={_fmt_percent(adaptive_budget_applied_rate)}, "
                f"update_rate={_fmt_percent(adaptive_budget_update_rate)}, "
                f"mean_reward={_fmt_num(adaptive_budget_mean_reward)}, "
                f"mean_multiplier={_fmt_num(adaptive_budget_mean_multiplier)}, "
                f"request_budget_exceeded_rate={_fmt_percent(adaptive_budget_budget_exceeded_rate)}"
            )
        if adaptive_online_ope is not None:
            print(
                f"{prefix}  online_ope: "
                + _format_ope_targets(
                    adaptive_online_ope,
                    ordered_targets=ADAPTIVE_ONLINE_TARGET_POLICY_ORDER,
                )
            )
        if adaptive_budget_ope is not None:
            print(
                f"{prefix}  budget_ope: "
                + _format_ope_targets(
                    adaptive_budget_ope,
                    ordered_targets=ADAPTIVE_BUDGET_TARGET_POLICY_ORDER,
                )
            )
        print(
            f"{prefix}  controls: topup_case_rate={_fmt_percent(topup_case_rate)}, "
            f"topup_attempt_rate={_fmt_percent(topup_attempt_rate)}, "
            f"mean_syntax_cycles={_fmt_num(mean_syntax_cycles)}, "
            f"syntax_failed_attempt_rate={_fmt_percent(syntax_fail_attempt_rate)}, "
            f"source_scope_match_rate={_fmt_percent(source_scope_match_rate)}"
        )
    pairwise_uplifts = bucket.get("pairwise_uplifts")
    if isinstance(pairwise_uplifts, list) and pairwise_uplifts:
        print(f"{prefix}- pairwise_uplifts ({_attempt_budget_note(bucket)}):")
        for entry in pairwise_uplifts:
            if not isinstance(entry, dict):
                continue
            condition_a = str(entry.get("condition_a") or "")
            condition_b = str(entry.get("condition_b") or "")
            uplift = entry.get("uplift_success_rate")
            if condition_a and condition_b:
                print(
                    f"{prefix}  {condition_a} -> {condition_b}: {_fmt_signed_percent(uplift)} "
                    f"(n={entry.get('n_matched_cases', 'n/a')} matched cases"
                    f"{'' if entry.get('case_sets_match', True) else ', case sets differ'})"
                )
    uplift = bucket.get("uplift_success_rate")
    print(
        f"{prefix}- uplift_success_rate: {_fmt_signed_percent(uplift)} "
        f"({_attempt_budget_note(bucket)})"
    )
    print(
        f"{prefix}- mean_prompt_chars_stage_b: {_fmt_num(bucket.get('mean_prompt_chars_stage_b'))}"
    )
    print(f"{prefix}- mean_retry_history_chars: {_fmt_num(bucket.get('mean_retry_history_chars'))}")
    print(f"{prefix}- mean_retry_history_turns: {_fmt_num(bucket.get('mean_retry_history_turns'))}")
    print(
        f"{prefix}- retry_history_budget_applied_rate: "
        f"{_fmt_percent(bucket.get('retry_history_budget_applied_rate'))}"
    )
    print(
        f"{prefix}- retry_history_truncation_rate: "
        f"{_fmt_percent(bucket.get('retry_history_truncation_rate'))}"
    )
    by_condition = bucket.get("mean_prompt_chars_stage_b_by_condition")
    if isinstance(by_condition, dict) and by_condition:
        rendered = ", ".join(
            f"{condition}={_fmt_num(value)}" for condition, value in sorted(by_condition.items())
        )
        print(f"{prefix}- mean_prompt_chars_stage_b_by_condition: {rendered}")
    print(
        f"{prefix}- stage_a_request_success_rate: "
        f"{_fmt_percent(bucket.get('stage_a_request_success_rate'))}"
    )
    print(f"{prefix}- retry_gate_block_rate: {_fmt_percent(bucket.get('retry_gate_block_rate'))}")
    print(
        f"{prefix}- uplift_staged_vs_monolithic: "
        f"{_fmt_signed_percent(bucket.get('uplift_staged_vs_monolithic'))}"
    )
    print(
        f"{prefix}- uplift_layered_gated_vs_base: "
        f"{_fmt_signed_percent(bucket.get('uplift_layered_gated_vs_base'))}"
    )
    print(
        f"{prefix}- uplift_layered_llm_vs_base: "
        f"{_fmt_signed_percent(bucket.get('uplift_layered_llm_vs_base'))}"
    )
    print(
        f"{prefix}- test_touch_attempts: {int(bucket.get('test_touch_attempts') or 0)}/"
        f"{int(bucket.get('patch_attempts') or 0)} "
        f"({_fmt_percent(bucket.get('test_touch_attempt_rate'))})"
    )


def _attempt_budget_note(bucket: dict[str, Any]) -> str:
    """How the compared outcomes were scored (STA-13, D12)."""
    if str(bucket.get("uplift_attempt_budget_basis") or "") != "common_k":
        return "own attempt budgets"
    return f"within common k={bucket.get('common_attempt_k')} attempts"


def _print_input_errors(input_errors: Any) -> None:
    """Print input diagnostic counts if any errors were encountered."""
    if not isinstance(input_errors, dict):
        return
    files_unreadable = int(input_errors.get("files_unreadable", 0))
    json_lines_invalid = int(input_errors.get("json_lines_invalid", 0))
    records_skipped = int(input_errors.get("records_skipped", 0))
    records_schema_invalid = int(input_errors.get("records_schema_invalid", 0))
    duplicate_attempt_rows_seen = int(input_errors.get("duplicate_attempt_rows_seen", 0))
    duplicate_attempt_rows_resolved = int(input_errors.get("duplicate_attempt_rows_resolved", 0))
    duplicate_conflicting = int(input_errors.get("duplicate_attempt_rows_conflicting", 0))
    if (
        files_unreadable == 0
        and json_lines_invalid == 0
        and records_skipped == 0
        and records_schema_invalid == 0
        and duplicate_attempt_rows_seen == 0
    ):
        return
    print("\nInput diagnostics:")
    if files_unreadable or json_lines_invalid or records_skipped or records_schema_invalid:
        print(
            "- errors: "
            f"files_unreadable={files_unreadable}, "
            f"json_lines_invalid={json_lines_invalid}, "
            f"records_skipped={records_skipped}, "
            f"records_schema_invalid={records_schema_invalid}"
        )
    if duplicate_attempt_rows_seen > 0:
        print(
            "- dedup: "
            f"duplicate_attempt_rows_seen={duplicate_attempt_rows_seen}, "
            f"duplicate_attempt_rows_resolved={duplicate_attempt_rows_resolved}, "
            f"duplicate_attempt_rows_conflicting={duplicate_conflicting}"
        )
        if duplicate_conflicting:
            # DAT-24: equal-priority rows that disagree; the later one won.
            print(
                f"- warning: {duplicate_conflicting} duplicate attempt row(s) disagree about "
                "success with no recorded order; rerun with --strict-input to reject them"
            )


def _ope_target_order(ope: dict[str, Any], *, ordered_targets: tuple[str, ...]) -> list[str]:
    """Target names to render, preferring the summary's own ``target_order``.

    Falling back to *ordered_targets* keeps summaries written before ``target_order``
    existed renderable; trailing targets missing from the order (decision-model targets in
    a hand-built summary, for example) are appended in insertion order.
    """
    raw_order = ope.get("target_order")
    order = (
        [str(name) for name in raw_order]
        if isinstance(raw_order, list) and raw_order
        else list(ordered_targets)
    )
    targets_raw = ope.get("targets")
    if isinstance(targets_raw, dict):
        order.extend(str(name) for name in targets_raw if str(name) not in order)
    return order


def _format_ope_targets(ope: dict[str, Any], *, ordered_targets: tuple[str, ...]) -> str:
    """Format OPE target estimates into a one-line summary string."""
    targets_raw = ope.get("targets")
    targets = targets_raw if isinstance(targets_raw, dict) else {}
    parts: list[str] = []
    for target_name in _ope_target_order(ope, ordered_targets=ordered_targets):
        target = targets.get(target_name)
        if not isinstance(target, dict):
            continue
        ci_raw = target.get("ope_ci")
        ci: dict[str, Any] = ci_raw if isinstance(ci_raw, dict) else {}
        ips = _fmt_num_with_ci(target.get("ips"), ci.get("ips"))
        snips = _fmt_num_with_ci(target.get("snips"), ci.get("snips"))
        dr = _fmt_num_with_ci(target.get("dr"), ci.get("dr"))
        ess = _fmt_num(target.get("effective_sample_size"))
        matched = int(target.get("matched_samples") or 0)
        total = int(target.get("total_samples") or 0)
        diagnostics: list[str] = []
        support_warning = str(target.get("support_warning") or "")
        if support_warning:
            diagnostics.append(f"support={support_warning}")
        agreement_warning = str(target.get("estimator_agreement_warning") or "")
        if agreement_warning:
            diagnostics.append(f"agreement={agreement_warning}")
        diagnostics_fragment = f" [{', '.join(diagnostics)}]" if diagnostics else ""
        parts.append(
            f"{target_name}: ips={ips}, snips={snips}, dr={dr}, ess={ess}, "
            f"support={matched}/{total}{diagnostics_fragment}"
        )
    notes = _ope_sample_notes(ope)
    rendered = "; ".join(parts) if parts else "n/a"
    return f"{notes}{rendered}" if notes else rendered


def _ope_sample_notes(ope: dict[str, Any]) -> str:
    """Samples left out of or corrected in the estimates (STA-12, POL-03)."""
    notes: list[str] = []
    dropped = int(ope.get("propensity_missing_dropped") or 0)
    if dropped:
        notes.append(f"{dropped} sample(s) without a propensity dropped")
    corrected = int(ope.get("shadow_corrected_samples") or 0)
    if corrected:
        notes.append(f"{corrected} shadow-mode propensity(ies) corrected to 1.0")
    return f"[{'; '.join(notes)}] " if notes else ""
