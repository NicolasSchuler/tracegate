"""Analyze and summarize llmdebug eval JSONL results.

Loads evaluation records, validates schema, derives per-case-condition
summary rows, and aggregates metrics across categories, difficulties,
and snapshot dependencies.  Supports adaptive helper, online-policy,
and budget-policy roll-ups with off-policy evaluation (OPE).  Optional
statistical significance testing via McNemar's test, Cohen's h, and
bootstrap confidence intervals.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from pathlib import Path
from typing import Any

from evals.analysis_adaptive_rollup import (  # noqa: F401
    _adaptive_rollup_metrics,
    _has_adaptive_telemetry,
)
from evals.analysis_attempt_metrics import (  # noqa: F401
    _adaptive_budget_policy_attempt_metrics,
    _adaptive_budget_topup_attempt_metrics,
    _adaptive_helper_attempt_metrics,
    _adaptive_helper_outcome,
    _adaptive_online_attempt_metrics,
    _context_protocol_from_records,
    _helper_retry_count,
    _helper_transport_retry_count,
    _patch_verifier_source_scope_metrics,
    _retry_gate_counts,
    _stage_a_request_counts,
    _stage_b_prompt_chars_mean,
    _syntax_preflight_attempt_metrics,
)
from evals.analysis_numeric import (  # noqa: F401
    _as_float_or_none,
    _as_int,
    _fmt_num,
    _fmt_num_with_ci,
    _fmt_percent,
    _fmt_signed_percent,
    _increment_string_count,
    _mean,
    _median,
    _quantile,
    _ratio,
    _stddev,
)
from evals.analysis_ope import (  # noqa: F401
    ADAPTIVE_BUDGET_TARGET_POLICIES,
    ADAPTIVE_BUDGET_TARGET_POLICY_ITEMS,
    ADAPTIVE_BUDGET_TARGET_POLICY_ORDER,
    ADAPTIVE_ONLINE_TARGET_POLICIES,
    ADAPTIVE_ONLINE_TARGET_POLICY_ITEMS,
    ADAPTIVE_ONLINE_TARGET_POLICY_ORDER,
    ADAPTIVE_OPE_SHADOW_POLICY_MODE,
    _adaptive_ope_samples,
    _adaptive_ope_summary,
    _adaptive_ope_summary_from_records,
    _crossfit_alpha_summary,
    _ope_sample_policy_context,
)
from evals.analysis_ope_estimators import (  # noqa: F401
    _RIDGECV_MIN_SAMPLES,
    ADAPTIVE_OPE_CROSSFIT_FOLDS,
    ADAPTIVE_OPE_RIDGE_ALPHA,
    ADAPTIVE_OPE_RIDGE_ALPHAS,
    OPE_BOOTSTRAP_RESAMPLES,
    CrossFitModels,
    FittedRewardModels,
    _crossfit_reward_models,
    _deterministic_target_ope,
    _doubly_robust_estimate,
    _doubly_robust_terms,
    _fit_action_reward_models,
    _fit_ridge_action_models,
    _ope_case_bootstrap,
    _ope_feature_names,
    _predict_action_reward,
    _stratified_doubly_robust_estimate,
    _stratified_doubly_robust_terms,
    logger,
)
from evals.analysis_ope_stochastic import (  # noqa: F401
    DECISION_MODEL_ADVICE_STATUS_OK,
    DECISION_MODEL_ATOMIC_THRESHOLDS,
    DECISION_MODEL_TARGET_ATOMIC,
    DECISION_MODEL_TARGET_GREEDY,
    DECISION_MODEL_TARGET_KINDS,
    DECISION_MODEL_TARGET_STOCHASTIC,
    StochasticTargetOpeSettings,
    _decision_model_advice_ok,
    _decision_model_target_distribution,
    _decision_model_targets,
    _derived_atomic_action,
    _mean_target_action_probabilities,
    _one_hot_distribution,
    _ope_baseline_action,
    _ordered_actions,
    _sample_available_actions,
    _stochastic_doubly_robust_estimate,
    _stochastic_doubly_robust_terms,
    _stochastic_dr_support_reason,
    _stochastic_target_ope,
    _stratified_stochastic_doubly_robust_estimate,
    _stratified_stochastic_doubly_robust_terms,
    _target_importance_weight,
)
from evals.analysis_ope_support import (  # noqa: F401  # noqa: E402, F401
    _ONLINE_ACTION_RANK,
    ADAPTIVE_BUDGET_ACTIONS,
    ADAPTIVE_BUDGET_FAMILIES,
    ADAPTIVE_BUDGET_REWARD_SCOPE,
    ADAPTIVE_BUDGET_REWARD_SCOPE_LEGACY,
    ADAPTIVE_BUDGET_REWARD_SCOPES,
    ADAPTIVE_ONLINE_ACTIONS,
    ADAPTIVE_ONLINE_REWARD_SCOPE,
    ADAPTIVE_ONLINE_REWARD_SCOPE_LEGACY,
    ADAPTIVE_ONLINE_REWARD_SCOPES,
    ADAPTIVE_OPE_ESTIMATOR_AGREEMENT_ABS_TOL,
    ADAPTIVE_OPE_ESTIMATOR_AGREEMENT_REL_TOL,
    ADAPTIVE_OPE_MIN_ACTION_SUPPORT,
    ADAPTIVE_OPE_PROPENSITY_FLOOR_DEFAULT,
    ADAPTIVE_OPE_RAW_PROPENSITY_FLOOR,
    _action_counts_for_samples,
    _action_rates_for_counts,
    _coerce_feature_values,
    _estimator_agreement_summary,
    _feature_scaling_profile_bucket,
    _finite_estimator_values,
    _metadata_with_fallback,
    _samples_by_feature_scaling_profile,
    _sanitize_propensity,
    _shannon_entropy_from_counts,
    _support_warning_for_target,
)
from evals.analysis_output import (  # noqa: F401
    _format_ope_targets,
    _ope_sample_notes,
    _ope_target_order,
    _print_bucket,
    _print_input_errors,
    _print_summary,
)
from evals.analysis_paired_stats import (  # noqa: F401
    PAIRWISE_CORRECTION_FAMILY,
    _apply_pairwise_holm,
    _print_stats,
    _run_statistical_analysis,
    _run_statistical_analysis_pair,
)
from evals.analysis_records import (  # noqa: F401
    _XXH3_HASH_FIELDS_SINCE_SCHEMA,
    CURRENT_EVIDENCE_HASH_FIELD,
    CURRENT_PATCH_DIFF_HASH_FIELD,
    CURRENT_PROMPT_HASH_FIELD,
    LEGACY_EVIDENCE_HASH_FIELD,
    LEGACY_PATCH_DIFF_HASH_FIELD,
    LEGACY_PROMPT_HASH_FIELD,
    REQUIRED_ATTEMPT_METADATA_FIELDS,
    REQUIRED_MODEL_METADATA_FIELDS,
    REQUIRED_OPENAI_MODEL_METADATA_FIELDS,
    REQUIRED_RUN_METADATA_FIELDS,
    REQUIRED_TOP_LEVEL_FIELDS,
    SUPPORTED_EVAL_RECORD_SCHEMA_VERSIONS,
    VALID_RECORD_TYPES,
    _evidence_hash_field_for_schema,
    _load_and_group_records,
    _prompt_hash_field_for_schema,
    _validate_eval_record_schema,
    _validate_optional_string_field,
)
from evals.analysis_rows import (  # noqa: F401
    _case_success_rates,
    _derive_case_condition_rows,
    _matched_uplift,
    _pairwise_uplifts,
    _records_by_case_condition,
    _records_for_rows,
    _success_within_attempts,
)
from evals.analysis_summary import (
    _summarize_bucket,
)
from evals.analysis_usage import (  # noqa: F401  # noqa: E402, F401
    _TIMING_BREAKDOWN_KEYS,
    _TOKEN_BREAKDOWN_KEYS,
    EXTERNAL_BASELINE_CONDITIONS,
    TokenUsage,
    _attempts_budget_from_records,
    _attempts_executed,
    _attempts_to_success,
    _baseline_attempt_rollup,
    _baseline_attempt_summaries,
    _baseline_meta_from_record,
    _baseline_meta_last_str,
    _baseline_metric_last_bool,
    _baseline_metric_last_int,
    _baseline_metric_last_str,
    _baseline_metric_sum,
    _baseline_metrics_from_record,
    _baseline_patch_meta_last_int,
    _baseline_patch_meta_last_str,
    _baseline_patch_meta_last_str_list,
    _condition_timing_breakdown_mean,
    _condition_token_breakdown_mean,
    _dedupe_records_by_attempt,
    _is_external_baseline_condition,
    _legacy_patch_meta_has_usage,
    _legacy_patch_meta_reasoning_tokens,
    _llm_calls_total,
    _looks_like_test_path,
    _mean_prompt_chars_by_condition,
    _patch_meta_flag_count,
    _reasoning_token_usage_breakdown,
    _retry_history_attempt_metrics,
    _runtime_used,
    _test_touch_attempt_metrics,
    _timing_breakdown_mean,
    _timing_breakdown_sum,
    _token_breakdown_mean,
    _token_breakdown_sum,
    _token_usage,
    _token_usage_breakdown,
)
from evals.coerce import expand_paths as _expand_paths  # noqa: F401 (re-exported)
from evals.eval_types import EVAL_RECORD_SCHEMA_VERSION  # noqa: F401 (re-exported)

from . import _json as json
from .stats import BOOTSTRAP_METHOD, DEFAULT_BOOTSTRAP_SEED


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for analyzing llmdebug eval JSONL results.

    Loads JSONL files, validates records, computes per-condition success
    rates and metrics, and prints a human-readable summary.  Optionally
    runs statistical significance tests and writes a JSON summary file.

    Args:
        argv: Command-line arguments.  Uses sys.argv when None.

    Returns:
        Exit code: 0 on success, 1 when no data found, 2 on input error.
    """
    parser = argparse.ArgumentParser(description="Analyze llmdebug eval JSONL results.")
    parser.add_argument("paths", nargs="+", help="One or more JSONL result files (globs ok).")
    parser.add_argument(
        "--out",
        default="",
        help="Optional path to write machine-readable summary JSON.",
    )
    parser.add_argument(
        "--strict-input",
        action="store_true",
        help=(
            "Fail on the first unreadable file, malformed JSONL line or schema-invalid "
            "record; without it, such records are skipped and counted."
        ),
    )
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Run statistical significance tests (McNemar, Cohen's h, bootstrap CI).",
    )
    parser.add_argument(
        "--aggregation",
        choices=["per_run", "majority_vote", "any_success"],
        default="per_run",
        help=(
            "How repeated runs of a case form one paired observation: per_run (one run "
            "per case; refuses several), majority_vote (ties are left out) or any_success."
        ),
    )
    parser.add_argument(
        "--bootstrap-seed",
        type=int,
        default=DEFAULT_BOOTSTRAP_SEED,
        help=(
            "Seed for every bootstrap interval (default: "
            f"{DEFAULT_BOOTSTRAP_SEED}); recorded in analysis_config."
        ),
    )
    parser.add_argument(
        "--stats-pair-a",
        default="base_repair",
        help="Condition name for A side of statistical pairing (default: base_repair).",
    )
    parser.add_argument(
        "--stats-pair-b",
        default="layered_adaptive_llm",
        help="Condition name for B side of statistical pairing (default: layered_adaptive_llm).",
    )
    parser.add_argument(
        "--adaptive-ope-min-action-support",
        type=int,
        default=ADAPTIVE_OPE_MIN_ACTION_SUPPORT,
        help=(
            "Minimum behavior-policy samples required per action before DR is enabled (default: 2)."
        ),
    )
    parser.add_argument(
        "--adaptive-ope-propensity-floor",
        type=float,
        default=ADAPTIVE_OPE_PROPENSITY_FLOOR_DEFAULT,
        help=(
            "Analysis-only lower bound for adaptive OPE propensities in IPS/SNIPS/DR "
            f"(default: {ADAPTIVE_OPE_PROPENSITY_FLOOR_DEFAULT:g})."
        ),
    )
    args = parser.parse_args(argv)
    adaptive_ope_min_action_support = int(args.adaptive_ope_min_action_support)
    if adaptive_ope_min_action_support < 1:
        print("Input error: --adaptive-ope-min-action-support must be >= 1")
        return 2
    adaptive_ope_propensity_floor = float(args.adaptive_ope_propensity_floor)
    if not (0.0 < adaptive_ope_propensity_floor <= 1.0):
        print("Input error: --adaptive-ope-propensity-floor must be in (0, 1]")
        return 2

    try:
        grouped, input_errors = _load_and_group_records(args.paths, strict_input=args.strict_input)
    except ValueError as exc:
        print(f"Input error: {exc}")
        return 2
    if not grouped:
        _print_input_errors(input_errors)
        print("No records found.")
        return 1

    rows = _derive_case_condition_rows(grouped)
    if not rows:
        print("No valid case-condition records found.")
        return 1
    records_by_case_condition = _records_by_case_condition(grouped)

    by_category: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_difficulty: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_snapshot_dependency: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_bug_source: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_contamination_risk: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_category[row["category"]].append(row)
        by_difficulty[row["difficulty"]].append(row)
        by_snapshot_dependency[row["snapshot_dependency"]].append(row)
        by_bug_source[row.get("bug_source") or "hand_crafted"].append(row)
        by_contamination_risk[row.get("contamination_risk") or "unknown"].append(row)

    summary = {
        "overall": _summarize_bucket(
            rows,
            records_by_case_condition=records_by_case_condition,
            adaptive_ope_min_action_support=adaptive_ope_min_action_support,
            adaptive_ope_propensity_floor=adaptive_ope_propensity_floor,
        ),
        "by_category": {
            key: _summarize_bucket(
                bucket_rows,
                records_by_case_condition=records_by_case_condition,
                adaptive_ope_min_action_support=adaptive_ope_min_action_support,
                adaptive_ope_propensity_floor=adaptive_ope_propensity_floor,
            )
            for key, bucket_rows in sorted(by_category.items())
        },
        "by_difficulty": {
            key: _summarize_bucket(
                bucket_rows,
                records_by_case_condition=records_by_case_condition,
                adaptive_ope_min_action_support=adaptive_ope_min_action_support,
                adaptive_ope_propensity_floor=adaptive_ope_propensity_floor,
            )
            for key, bucket_rows in sorted(by_difficulty.items())
        },
        "by_snapshot_dependency": {
            key: _summarize_bucket(
                bucket_rows,
                records_by_case_condition=records_by_case_condition,
                adaptive_ope_min_action_support=adaptive_ope_min_action_support,
                adaptive_ope_propensity_floor=adaptive_ope_propensity_floor,
            )
            for key, bucket_rows in sorted(by_snapshot_dependency.items())
        },
        "by_bug_source": {
            key: _summarize_bucket(
                bucket_rows,
                records_by_case_condition=records_by_case_condition,
                adaptive_ope_min_action_support=adaptive_ope_min_action_support,
                adaptive_ope_propensity_floor=adaptive_ope_propensity_floor,
            )
            for key, bucket_rows in sorted(by_bug_source.items())
        },
        "by_contamination_risk": {
            key: _summarize_bucket(
                bucket_rows,
                records_by_case_condition=records_by_case_condition,
                adaptive_ope_min_action_support=adaptive_ope_min_action_support,
                adaptive_ope_propensity_floor=adaptive_ope_propensity_floor,
            )
            for key, bucket_rows in sorted(by_contamination_risk.items())
        },
        "analysis_config": {
            "adaptive_ope_min_action_support": adaptive_ope_min_action_support,
            "adaptive_ope_propensity_floor": adaptive_ope_propensity_floor,
            "bootstrap_seed": int(args.bootstrap_seed),
            "bootstrap_method": BOOTSTRAP_METHOD,
        },
        "input_errors": input_errors,
    }

    _print_summary(summary)

    if args.stats:
        stats_result = _run_statistical_analysis(
            rows,
            aggregation=args.aggregation,
            pair_a=str(args.stats_pair_a),
            pair_b=str(args.stats_pair_b),
            seed=int(args.bootstrap_seed),
        )
        summary["statistics"] = stats_result
        _print_stats(stats_result)

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nWrote summary JSON: {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
