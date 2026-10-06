"""Concise single-run evaluation summary with strict supported-schema validation.

Loads a single JSONL results file, validates records, and prints a
human-readable summary covering condition effectiveness, pairwise uplifts,
reliability diagnostics, adaptive signals, and guardrail statistics.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from evals.analysis_api import (
    derive_case_condition_rows,
    fmt_num_with_ci,
    is_external_baseline_condition,
    load_and_group_records,
    ope_target_order,
    print_input_errors,
    records_by_case_condition,
    summarize_bucket,
)
from evals.eval_types import CONDITION_FAMILY_ADAPTIVE, condition_family


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for single-run evaluation summary.

    Parses arguments, loads the specified JSONL results file, validates
    records against the supported eval-record schemas, and prints a concise
    summary to stdout.

    Args:
        argv: Command-line arguments.  Defaults to ``sys.argv[1:]``
            when ``None``.

    Returns:
        Exit code: 0 on success, 1 for missing or empty data,
        2 for validation errors.
    """
    parser = argparse.ArgumentParser(
        description="Print a concise eval summary for one run_id with strict supported-schema validation."
    )
    parser.add_argument("run_id", help="Run id (expects <results-dir>/<run_id>.jsonl).")
    parser.add_argument(
        "--results-dir",
        default="evals/results",
        help="Directory containing eval result JSONL files (default: evals/results).",
    )
    args = parser.parse_args(argv)

    results_path = Path(str(args.results_dir)) / f"{args.run_id}.jsonl"
    if not results_path.is_file():
        print(f"Input error: results file not found: {results_path}")
        return 1

    try:
        grouped, input_errors = load_and_group_records([str(results_path)], strict_input=True)
    except ValueError as exc:
        print(f"Input error: {exc}")
        return 2

    if not grouped:
        print_input_errors(input_errors)
        print("No records found.")
        return 1

    rows = derive_case_condition_rows(grouped)
    if not rows:
        print("No valid case-condition records found.")
        return 1

    overall = summarize_bucket(rows, records_by_case_condition=records_by_case_condition(grouped))
    _print_core_summary(
        run_id_requested=str(args.run_id),
        results_path=results_path,
        rows=rows,
        overall=overall,
        input_errors=input_errors,
    )
    return 0


def _print_core_summary(
    *,
    run_id_requested: str,
    results_path: Path,
    rows: list[dict[str, Any]],
    overall: dict[str, Any],
    input_errors: dict[str, int],
) -> None:
    run_ids = sorted({str(row.get("run_id") or "") for row in rows if row.get("run_id")})
    condition_matrix = _condition_matrix(overall)
    conditions = [str(item.get("condition") or "") for item in condition_matrix]
    adaptive_condition, adaptive_stats = _select_adaptive_stats(condition_matrix)

    print("Run Summary")
    print(f"- requested run_id: {run_id_requested}")
    print(f"- results file: {results_path}")
    print(f"- run IDs present: {', '.join(run_ids) if run_ids else 'n/a'}")
    print(f"- conditions: {', '.join(conditions) if conditions else 'none'}")
    print(f"- case-condition rows: {len(rows)}")

    print("\nCondition Effectiveness & Efficiency")
    if not condition_matrix:
        print("- none")
    for stats in condition_matrix:
        _print_condition_row(stats)

    print("\nPairwise Uplift Matrix")
    pairwise = _pairwise_uplifts(overall)
    if not pairwise:
        print("- none")
    else:
        for entry in pairwise:
            condition_a = str(entry.get("condition_a") or "?")
            condition_b = str(entry.get("condition_b") or "?")
            uplift = _fmt_signed_percent(entry.get("uplift_success_rate"))
            matched = entry.get("n_matched_cases", "n/a")
            differ = "" if entry.get("case_sets_match", True) else ", case sets differ"
            print(f"- {condition_a} -> {condition_b}: {uplift} (n={matched} matched cases{differ})")

    print("\nReliability & Input Quality")
    patch_attempts = int(overall.get("patch_attempts") or 0)
    test_touch_attempts = int(overall.get("test_touch_attempts") or 0)
    print(
        f"- test-touch patches: {test_touch_attempts}/{patch_attempts} "
        f"({_fmt_percent(overall.get('test_touch_attempt_rate'))})"
    )
    print(
        f"- stage-a request success rate: {_fmt_percent(overall.get('stage_a_request_success_rate'))}"
    )
    print(f"- retry gate block rate: {_fmt_percent(overall.get('retry_gate_block_rate'))}")
    print(
        f"- retry history truncation rate: "
        f"{_fmt_percent(overall.get('retry_history_truncation_rate'))}"
    )
    print(
        "- input diagnostics: "
        f"files_unreadable={int(input_errors.get('files_unreadable', 0))}, "
        f"json_lines_invalid={int(input_errors.get('json_lines_invalid', 0))}, "
        f"records_skipped={int(input_errors.get('records_skipped', 0))}, "
        f"infra_event_rows_seen={int(input_errors.get('infra_event_rows_seen', 0))}, "
        f"infra_event_rows_skipped={int(input_errors.get('infra_event_rows_skipped', 0))}, "
        f"duplicate_attempt_rows_seen={int(input_errors.get('duplicate_attempt_rows_seen', 0))}, "
        f"duplicate_attempt_rows_resolved={int(input_errors.get('duplicate_attempt_rows_resolved', 0))}"
    )

    if _has_adaptive_condition(conditions) and adaptive_stats is not None:
        print(f"\nAdaptive Signals ({adaptive_condition})")
        print(
            "- helper: "
            f"invocation_case_rate={_fmt_percent(adaptive_stats.get('helper_invocation_case_rate'))}, "
            "invocation_attempt_rate="
            f"{_fmt_percent(adaptive_stats.get('helper_invocation_attempt_rate'))}, "
            f"success_when_invoked={_fmt_percent(adaptive_stats.get('helper_success_when_invoked'))}, "
            f"rescue_rate={_fmt_percent(adaptive_stats.get('helper_rescue_rate'))}"
        )
        print(
            "- online policy: "
            f"decisions={int(adaptive_stats.get('adaptive_online_decisions') or 0)}, "
            "selected_invoke_rate="
            f"{_fmt_percent(adaptive_stats.get('adaptive_online_selected_invoke_rate'))}, "
            "applied_invoke_rate="
            f"{_fmt_percent(adaptive_stats.get('adaptive_online_applied_invoke_rate'))}, "
            f"update_rate={_fmt_percent(adaptive_stats.get('adaptive_online_update_rate'))}, "
            f"mean_reward={_fmt_num(adaptive_stats.get('adaptive_online_mean_reward'))}, "
            f"mean_propensity={_fmt_num(adaptive_stats.get('adaptive_online_mean_propensity'))}"
            f"{_adaptive_reward_diag_fragment(adaptive_stats, prefix='adaptive_online')}"
        )
        print(
            "- budget policy: "
            f"decisions={int(adaptive_stats.get('adaptive_budget_decisions') or 0)}, "
            "selected_increase_rate="
            f"{_fmt_percent(adaptive_stats.get('adaptive_budget_selected_increase_rate'))}, "
            "applied_increase_rate="
            f"{_fmt_percent(adaptive_stats.get('adaptive_budget_applied_increase_rate'))}, "
            f"update_rate={_fmt_percent(adaptive_stats.get('adaptive_budget_update_rate'))}, "
            f"mean_reward={_fmt_num(adaptive_stats.get('adaptive_budget_mean_reward'))}, "
            "mean_multiplier="
            f"{_fmt_num(adaptive_stats.get('adaptive_budget_mean_applied_multiplier'))}"
            f"{_adaptive_reward_diag_fragment(adaptive_stats, prefix='adaptive_budget')}"
        )
        online_ope = adaptive_stats.get("adaptive_online_ope")
        if isinstance(online_ope, dict):
            print(
                "- online OPE: "
                + _format_ope_targets(
                    online_ope,
                    ordered_targets=(
                        "always_skip",
                        "always_frame_only",
                        "always_file_only",
                        "always_invoke_helper",
                    ),
                )
            )
        budget_ope = adaptive_stats.get("adaptive_budget_ope")
        if isinstance(budget_ope, dict):
            print(
                "- budget OPE: "
                + _format_ope_targets(
                    budget_ope,
                    ordered_targets=("always_keep_budget", "always_increase_budget"),
                )
            )

    print("\nControls & Guardrails")
    controls_source = adaptive_stats if adaptive_stats is not None else overall
    print(
        "- budget top-up: "
        f"case_rate={_fmt_percent(controls_source.get('adaptive_budget_topup_case_rate'))}, "
        f"attempt_rate={_fmt_percent(controls_source.get('adaptive_budget_topup_attempt_rate'))}"
    )
    print(
        "- syntax preflight: "
        f"failed_attempt_rate={_fmt_percent(controls_source.get('syntax_preflight_failed_attempt_rate'))}, "
        f"failures={int(controls_source.get('syntax_preflight_failures') or 0)}"
    )
    print(
        "- patch verifier source-scope match rate: "
        f"{_fmt_percent(controls_source.get('patch_verifier_source_scope_match_rate'))}"
    )
    print(
        "- helper guardrails: "
        f"budget_exhausted_rate={_fmt_percent(controls_source.get('helper_budget_exhausted_rate'))}, "
        f"saturation_guard_rate={_fmt_percent(controls_source.get('helper_saturation_guard_rate'))}"
    )


def _print_condition_row(stats: dict[str, Any]) -> None:
    condition = str(stats.get("condition") or "unknown")
    cases = int(stats.get("cases") or 0)
    successes = int(stats.get("successes") or 0)
    row = (
        f"- {condition}: {successes}/{cases} ({_fmt_percent(stats.get('success_rate'))}), "
        f"median_attempts={_fmt_num(stats.get('median_attempts_to_success'))}, "
        f"mean_tokens_total={_fmt_num(stats.get('mean_tokens_total'))} "
        f"(token coverage {_fmt_percent(stats.get('token_covered_fraction'))} of attempts), "
        f"mean_llm_calls={_fmt_num(stats.get('mean_llm_calls_per_case'))}, "
        f"mean_runtime_sec={_fmt_num(stats.get('mean_runtime_sec'))}"
    )
    cases_missing = int(stats.get("cases_missing") or 0)
    if cases_missing > 0:  # STA-03
        row += f", missing {cases_missing}/{int(stats.get('cases_expected') or 0)} cases"
    reasoning_covered_cases = int(stats.get("reasoning_token_covered_cases") or 0)
    if reasoning_covered_cases > 0:
        row += (
            f", mean_tokens_reasoning={_fmt_num(stats.get('mean_tokens_reasoning'))}, "
            f"reasoning_coverage={reasoning_covered_cases}/{cases}"
        )
    print(row)
    if not is_external_baseline_condition(condition):
        return

    flow_bits: list[str] = []
    _append_covered_rate(
        flow_bits, stats, cases, "baseline_direct_gen_pass_rate", "direct_gen_pass_rate"
    )
    _append_covered_rate(
        flow_bits, stats, cases, "baseline_debug_added_value_rate", "debug_added_value_rate"
    )
    _append_covered_rate(
        flow_bits,
        stats,
        cases,
        "baseline_method_instrumentation_rate",
        "instrumentation_rate",
    )
    _append_covered_rate(
        flow_bits, stats, cases, "baseline_method_two_step_repair_rate", "two_step_rate"
    )
    stopped_reason_counts = stats.get("baseline_stopped_reason_counts")
    if isinstance(stopped_reason_counts, dict) and stopped_reason_counts:
        rendered = ", ".join(
            f"{name}:{count}"
            for name, count in sorted(
                stopped_reason_counts.items(),
                key=lambda item: (-int(item[1]), str(item[0])),
            )[:3]
        )
        flow_bits.append(f"stopped_reasons={rendered}")
    if flow_bits:
        print(f"  baseline flow: {'; '.join(flow_bits)}")
    contract_bits: list[str] = []
    _append_top_count_fragment(contract_bits, stats, "baseline_fidelity_level_counts", "fidelity")
    _append_top_count_fragment(
        contract_bits, stats, "baseline_repair_scope_policy_counts", "repair_scope"
    )
    _append_top_count_fragment(
        contract_bits, stats, "baseline_prompt_contract_kind_counts", "prompt_contract"
    )
    _append_top_count_fragment(
        contract_bits, stats, "baseline_hybrid_guidance_mode_counts", "hybrid_guidance"
    )
    if contract_bits:
        print(f"  baseline contract: {'; '.join(contract_bits)}")
    timing_bits: list[str] = []
    teacher_sec = stats.get("baseline_teacher_duration_sec_total_mean")
    if isinstance(teacher_sec, (int, float)):
        timing_bits.append(f"teacher_sec={_fmt_num(teacher_sec)}")
    learner_sec = stats.get("baseline_learner_duration_sec_total_mean")
    if isinstance(learner_sec, (int, float)):
        timing_bits.append(f"learner_sec={_fmt_num(learner_sec)}")
    postcheck_sec = stats.get("baseline_postcheck_duration_sec_total_mean")
    if isinstance(postcheck_sec, (int, float)):
        timing_bits.append(f"postcheck_sec={_fmt_num(postcheck_sec)}")
    empty_count = stats.get("baseline_learner_empty_response_count_mean")
    if isinstance(empty_count, (int, float)):
        timing_bits.append(f"empty_learner_mean={_fmt_num(empty_count)}")
    attempts_success = stats.get("mean_attempts_to_success")
    if isinstance(attempts_success, (int, float)):
        timing_bits.append(f"mean_attempts_success={_fmt_num(attempts_success)}")
    attempts_failed = stats.get("mean_attempts_when_failed")
    if isinstance(attempts_failed, (int, float)):
        timing_bits.append(f"mean_attempts_failed={_fmt_num(attempts_failed)}")
    guidance_chars_mean = stats.get("baseline_hybrid_guidance_chars_mean")
    if isinstance(guidance_chars_mean, (int, float)):
        timing_bits.append(f"guidance_chars_mean={_fmt_num(guidance_chars_mean)}")
    if timing_bits:
        print(f"  baseline timings: {'; '.join(timing_bits)}")

    feedback_bits: list[str] = []
    _append_covered_rate(feedback_bits, stats, cases, "baseline_is_passing_rate", "is_passing_rate")
    _append_covered_rate(
        feedback_bits,
        stats,
        cases,
        "baseline_feedback_improved_case_rate",
        "feedback_improved_case_rate",
    )
    _append_covered_rate(
        feedback_bits,
        stats,
        cases,
        "baseline_feedback_regressed_case_rate",
        "feedback_regressed_case_rate",
    )
    final_fail_mean = stats.get("baseline_feedback_final_fail_count_mean")
    if isinstance(final_fail_mean, (int, float)):
        feedback_bits.append(f"feedback_final_fail_mean={_fmt_num(final_fail_mean)}")
    if feedback_bits:
        print(f"  baseline feedback: {'; '.join(feedback_bits)}")


def _append_top_count_fragment(
    parts: list[str],
    stats: dict[str, Any],
    field: str,
    label: str,
) -> None:
    counts = stats.get(field)
    if not isinstance(counts, dict) or not counts:
        return
    name, count = max(
        ((str(key), int(value)) for key, value in counts.items() if isinstance(value, int)),
        key=lambda item: (item[1], item[0]),
        default=("", 0),
    )
    if not name:
        return
    parts.append(f"{label}={name}:{count}")


def _adaptive_reward_diag_fragment(stats: dict[str, Any], *, prefix: str) -> str:
    reward_p50 = stats.get(f"{prefix}_reward_p50")
    reward_p90 = stats.get(f"{prefix}_reward_p90")
    reward_std = stats.get(f"{prefix}_reward_std")
    if reward_p50 is None and reward_p90 is None and reward_std is None:
        return ""
    return (
        f", reward_p50={_fmt_num(reward_p50)}, "
        f"reward_p90={_fmt_num(reward_p90)}, "
        f"reward_std={_fmt_num(reward_std)}"
    )


def _append_covered_rate(
    parts: list[str],
    stats: dict[str, Any],
    cases: int,
    field: str,
    label: str,
) -> None:
    value = stats.get(field)
    covered_cases = stats.get(f"{field}_covered_cases")
    if not isinstance(covered_cases, int) or covered_cases <= 0:
        return
    if not isinstance(value, (int, float)):
        return
    parts.append(f"{label}={_fmt_percent(value)} (coverage {covered_cases}/{cases})")


def _condition_matrix(overall: dict[str, Any]) -> list[dict[str, Any]]:
    raw = overall.get("condition_matrix")
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict)]


def _pairwise_uplifts(overall: dict[str, Any]) -> list[dict[str, Any]]:
    raw = overall.get("pairwise_uplifts")
    if not isinstance(raw, list):
        return []
    return [item for item in raw if isinstance(item, dict)]


def _has_adaptive_condition(conditions: list[str]) -> bool:
    return any(condition_family(condition) == CONDITION_FAMILY_ADAPTIVE for condition in conditions)


def _select_adaptive_stats(
    condition_matrix: list[dict[str, Any]],
) -> tuple[str | None, dict[str, Any] | None]:
    stats_by_condition: dict[str, dict[str, Any]] = {}
    for item in condition_matrix:
        condition = str(item.get("condition") or "")
        if condition:
            stats_by_condition[condition] = item
    # Registered adaptive conditions only; any name containing "adaptive" used to match.
    for preferred in ("layered_adaptive_llm", "layered_adaptive_gated"):
        if preferred in stats_by_condition:
            return preferred, stats_by_condition[preferred]
    return None, None


def _fmt_percent(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    return f"{100.0 * float(value):.1f}%"


def _fmt_signed_percent(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    return f"{100.0 * float(value):+.1f}%"


def _fmt_num(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    value_float = float(value)
    value_int = int(value_float)
    if abs(value_float - value_int) < 1e-9:
        return str(value_int)
    return f"{value_float:.2f}"


def _format_ope_targets(ope: dict[str, Any], *, ordered_targets: tuple[str, ...]) -> str:
    targets_raw = ope.get("targets")
    targets = targets_raw if isinstance(targets_raw, dict) else {}
    parts: list[str] = []
    for target_name in ope_target_order(ope, ordered_targets=ordered_targets):
        target = targets.get(target_name)
        if not isinstance(target, dict):
            continue
        diagnostics: list[str] = []
        support_warning = str(target.get("support_warning") or "")
        if support_warning:
            diagnostics.append(f"support={support_warning}")
        agreement_warning = str(target.get("estimator_agreement_warning") or "")
        if agreement_warning:
            diagnostics.append(f"agreement={agreement_warning}")
        diagnostics_fragment = f" [{', '.join(diagnostics)}]" if diagnostics else ""
        ci_raw = target.get("ope_ci")
        ci: dict[str, Any] = ci_raw if isinstance(ci_raw, dict) else {}
        parts.append(
            f"{target_name}: ips={fmt_num_with_ci(target.get('ips'), ci.get('ips'))}, "
            f"snips={fmt_num_with_ci(target.get('snips'), ci.get('snips'))}, "
            f"dr={fmt_num_with_ci(target.get('dr'), ci.get('dr'))}, "
            f"ess={_fmt_num(target.get('effective_sample_size'))}"
            f"{diagnostics_fragment}"
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


if __name__ == "__main__":
    raise SystemExit(main())
