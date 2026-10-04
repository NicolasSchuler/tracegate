"""Generate a paper-ready plain-text metrics report for llmdebug eval runs.

Accepts one or more eval JSONL files (via glob), compares a baseline condition A
against a treatment condition B, and prints effectiveness, efficiency, failure
diagnostics, and overhead sections to stdout.  Statistical comparisons use
bootstrap confidence intervals and paired significance tests (McNemar / Wilcoxon).
"""

from __future__ import annotations

import argparse
import glob
import math
import re
from pathlib import Path
from typing import Any, cast

import polars as pl
from scipy import stats
from sklearn.metrics import cohen_kappa_score

from evals.analyze_results import (
    _derive_case_condition_rows,
    _is_external_baseline_condition,
    _load_and_group_records,
    _records_by_case_condition,
    _summarize_bucket,
)
from evals.stats import (
    analyze_paired_results,
    bootstrap_mean_ci,
    exact_binomial_p_value,
    format_p_value,
    paired_numeric_deltas,
)


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for the paper-ready eval metrics report.

    Loads eval records, computes per-condition and pairwise statistics, then
    prints a structured plain-text report to stdout.

    Args:
        argv: Command-line arguments.  Defaults to ``sys.argv[1:]`` when ``None``.

    CLI flags:
        --jsonl (required): Path or glob to one or more eval JSONL files.
        --eval-log: Optional path to eval stdout log with system/energy feedback.
        --a: Condition name for the baseline A side (default: ``base_repair``).
        --b: Condition name for the treatment B side
            (default: ``layered_adaptive_llm``).
        --primary-slice: Comma-separated ``snapshot_dependency`` values that
            define the primary endpoint subset
            (default: ``required,helpful``).
        --bootstrap: Number of bootstrap resamples for confidence intervals
            (default: 10 000).
        --seed: Random seed for bootstrap resampling (default: 42).
        --print-top-failures: Number of top failure reasons to print per
            condition (default: 10).
        --strict-input: Flag; fail on the first unreadable file or malformed
            JSONL record.

    Returns:
        Exit code: 0 on success, 1 for missing or empty data,
        2 for input validation errors.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Generate a paper-ready plain-text report for one llmdebug eval run "
            "(effectiveness + secondary efficiency + diagnostics)."
        )
    )
    parser.add_argument(
        "--jsonl",
        required=True,
        help="Path (or glob) to eval JSONL file(s).",
    )
    parser.add_argument(
        "--eval-log",
        default="",
        help="Optional path to eval stdout log containing run-level system/energy feedback.",
    )
    parser.add_argument(
        "--a",
        default="base_repair",
        help="Condition name for baseline A side.",
    )
    parser.add_argument(
        "--b",
        default="layered_adaptive_llm",
        help="Condition name for treatment B side.",
    )
    parser.add_argument(
        "--primary-slice",
        default="required,helpful",
        help="Comma-separated snapshot_dependency values for primary endpoint.",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=10000,
        help="Bootstrap resamples for confidence intervals.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for bootstrap.",
    )
    parser.add_argument(
        "--print-top-failures",
        type=int,
        default=10,
        help="Number of top failure reasons to print per condition.",
    )
    parser.add_argument(
        "--fixed-budget-token-thresholds",
        default="10000,25000,50000",
        help="Comma-separated total-token thresholds for fixed-budget success reporting.",
    )
    parser.add_argument(
        "--strict-input",
        action="store_true",
        help="Fail on first unreadable input file or malformed JSONL record.",
    )
    args = parser.parse_args(argv)
    try:
        fixed_budget_thresholds = _parse_fixed_budget_thresholds(args.fixed_budget_token_thresholds)
    except ValueError as exc:
        print(f"Input error: {exc}")
        return 2

    expanded = _expand_paths([args.jsonl])
    if not expanded:
        print(f"No input matched --jsonl={args.jsonl!r}")
        return 1

    try:
        grouped, input_errors = _load_and_group_records(expanded, strict_input=args.strict_input)
    except ValueError as exc:
        print(f"Input error: {exc}")
        return 2

    if not grouped:
        print("No records found.")
        _print_input_errors(input_errors)
        return 1

    rows = _derive_case_condition_rows(grouped)
    if not rows:
        print("No valid case-condition rows found.")
        return 1

    raw_records = _flatten_grouped_records(grouped)
    primary_slice = _parse_primary_slice(args.primary_slice)
    run_context = _build_run_context(
        rows=rows, raw_records=raw_records, eval_log_path=args.eval_log
    )
    condition_summary = _summarize_bucket(
        rows, records_by_case_condition=_records_by_case_condition(grouped)
    )
    primary_rows = _filter_rows_by_snapshot_dependency(rows, primary_slice)
    required_rows = _filter_rows_by_snapshot_dependency(rows, {"required"})
    control_rows = _filter_rows_by_snapshot_dependency(rows, {"none"})

    overall_effect = _compute_effectiveness_summary(
        rows=rows,
        condition_a=args.a,
        condition_b=args.b,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )
    primary_effect = _compute_effectiveness_summary(
        rows=primary_rows,
        condition_a=args.a,
        condition_b=args.b,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )
    required_effect = _compute_effectiveness_summary(
        rows=required_rows,
        condition_a=args.a,
        condition_b=args.b,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )
    control_effect = _compute_effectiveness_summary(
        rows=control_rows,
        condition_a=args.a,
        condition_b=args.b,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )
    category_stats = _compute_category_stats(
        rows=primary_rows,
        condition_a=args.a,
        condition_b=args.b,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )
    efficiency = _compute_efficiency_summaries(
        rows=primary_rows,
        condition_a=args.a,
        condition_b=args.b,
        n_bootstrap=max(200, args.bootstrap // 2),
        seed=args.seed,
    )
    dual_objective = _compute_dual_objective_summary(
        rows=primary_rows,
        condition_a=args.a,
        condition_b=args.b,
        n_bootstrap=max(200, args.bootstrap // 2),
        seed=args.seed,
        fixed_budget_thresholds=fixed_budget_thresholds,
    )
    token_estimate_diagnostics = _compute_token_estimate_diagnostics(
        raw_records=raw_records,
        condition_a=args.a,
        condition_b=args.b,
    )
    failures = _compute_failure_diagnostics(
        raw_records=raw_records,
        condition_a=args.a,
        condition_b=args.b,
        top_n=max(1, args.print_top_failures),
    )
    overhead = _compute_overhead_analysis(
        rows=rows,
        condition_a=args.a,
        condition_b=args.b,
    )

    _print_report(
        run_context=run_context,
        condition_a=args.a,
        condition_b=args.b,
        primary_slice=primary_slice,
        overall_effect=overall_effect,
        primary_effect=primary_effect,
        required_effect=required_effect,
        control_effect=control_effect,
        category_stats=category_stats,
        efficiency=efficiency,
        dual_objective=dual_objective,
        token_estimate_diagnostics=token_estimate_diagnostics,
        condition_summary=condition_summary,
        failures=failures,
        overhead=overhead,
        input_errors=input_errors,
    )
    return 0


def _expand_paths(items: list[str]) -> list[str]:
    out: list[str] = []
    for item in items:
        matches = sorted(glob.glob(item))
        out.extend(matches if matches else [item])
    return out


def _flatten_grouped_records(
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]],
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for key in sorted(grouped):
        out.extend(grouped[key])
    return out


def _parse_primary_slice(raw: str) -> set[str]:
    values = {part.strip() for part in raw.split(",") if part.strip()}
    return values or {"required", "helpful"}


def _parse_fixed_budget_thresholds(raw: str) -> list[int]:
    thresholds: list[int] = []
    for part in str(raw or "").split(","):
        token = part.strip()
        if not token:
            continue
        try:
            value = int(token)
        except ValueError as exc:
            msg = f"--fixed-budget-token-thresholds contains non-integer value {token!r}"
            raise ValueError(msg) from exc
        if value <= 0:
            msg = "--fixed-budget-token-thresholds values must be positive"
            raise ValueError(msg)
        thresholds.append(value)
    unique_sorted = sorted(set(thresholds))
    if not unique_sorted:
        msg = "--fixed-budget-token-thresholds must provide at least one positive integer"
        raise ValueError(msg)
    return unique_sorted


def _filter_rows_by_snapshot_dependency(
    rows: list[dict[str, Any]],
    allowed: set[str],
) -> list[dict[str, Any]]:
    return [row for row in rows if str(row.get("snapshot_dependency") or "") in allowed]


def _compute_effectiveness_summary(
    *,
    rows: list[dict[str, Any]],
    condition_a: str,
    condition_b: str,
    n_bootstrap: int,
    seed: int | None,
) -> dict[str, Any]:
    result = analyze_paired_results(
        rows,
        n_bootstrap=max(0, n_bootstrap),
        seed=seed,
        condition_a=condition_a,
        condition_b=condition_b,
    )
    if "error" in result:
        return {"available": False, "error": result["error"]}

    a_rate = float(result["a_success_rate"])
    b_rate = float(result["b_success_rate"])
    uplift = b_rate - a_rate
    relative = (uplift / a_rate) if a_rate > 0 else None
    agreement = _paired_binary_agreement(
        rows=rows,
        condition_a=condition_a,
        condition_b=condition_b,
    )
    return {
        "available": True,
        "n_pairs": int(result["n_pairs"]),
        "a_rate": a_rate,
        "b_rate": b_rate,
        "uplift": uplift,
        "uplift_pp": uplift * 100.0,
        "relative_change": relative,
        "mcnemar": result.get("overall_mcnemar", {}),
        "cohens_h": result.get("overall_cohens_h", {}),
        "agreement": agreement,
        "bootstrap_ci": result.get("overall_bootstrap_ci", {}),
        "power": result.get("overall_power", {}),
        "raw": result,
    }


def _paired_binary_agreement(
    *,
    rows: list[dict[str, Any]],
    condition_a: str,
    condition_b: str,
) -> dict[str, Any]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        run_id = str(row.get("run_id") or "")
        case_id = str(row.get("case_id") or "")
        condition = str(row.get("condition") or "")
        if not run_id or not case_id or condition not in {condition_a, condition_b}:
            continue
        normalized.append(
            {
                "run_id": run_id,
                "case_id": case_id,
                "condition": condition,
                "success": int(bool(row.get("success"))),
            }
        )

    if not normalized:
        return {"available": False, "error": "No matched A/B pairs found."}

    frame = pl.DataFrame(normalized)
    paired = (
        frame.group_by(["run_id", "case_id"])
        .agg(
            pl.col("success").filter(pl.col("condition") == condition_a).first().alias("a_success"),
            pl.col("success").filter(pl.col("condition") == condition_b).first().alias("b_success"),
        )
        .drop_nulls(["a_success", "b_success"])
    )
    if paired.is_empty():
        return {"available": False, "error": "No matched A/B pairs found."}

    a_success = paired["a_success"].cast(pl.Int64).to_list()
    b_success = paired["b_success"].cast(pl.Int64).to_list()
    kappa = float(cohen_kappa_score(a_success, b_success))
    return {
        "available": True,
        "kappa": kappa,
        "magnitude": _cohens_kappa_magnitude(kappa),
    }


def _cohens_kappa_magnitude(kappa: float) -> str:
    abs_kappa = abs(kappa)
    if abs_kappa < 0.2:
        return "slight"
    if abs_kappa < 0.4:
        return "fair"
    if abs_kappa < 0.6:
        return "moderate"
    if abs_kappa < 0.8:
        return "substantial"
    return "almost_perfect"


def _compute_category_stats(
    *,
    rows: list[dict[str, Any]],
    condition_a: str,
    condition_b: str,
    n_bootstrap: int,
    seed: int | None,
) -> dict[str, Any]:
    result = analyze_paired_results(
        rows,
        n_bootstrap=max(0, n_bootstrap),
        seed=seed,
        condition_a=condition_a,
        condition_b=condition_b,
    )
    if "error" in result:
        return {"available": False, "error": result["error"], "rows": []}

    per_category = result.get("per_category", {})
    holm = result.get("holm_bonferroni", {})
    rows_out: list[dict[str, Any]] = []
    if isinstance(per_category, dict):
        for category, value in sorted(per_category.items()):
            if not isinstance(value, dict):
                continue
            a_rate = value.get("a_success_rate")
            b_rate = value.get("b_success_rate")
            uplift = (
                float(b_rate) - float(a_rate)
                if isinstance(a_rate, (int, float)) and isinstance(b_rate, (int, float))
                else None
            )
            mcn = value.get("mcnemar", {}) if isinstance(value.get("mcnemar"), dict) else {}
            hb = holm.get(category, {}) if isinstance(holm, dict) else {}
            rows_out.append(
                {
                    "category": category,
                    "n_pairs": int(value.get("n_pairs", 0)),
                    "uplift": uplift,
                    "mcnemar_p": _as_float(mcn.get("p_value")),
                    "mcnemar_significant": bool(mcn.get("significant", False)),
                    "holm_adjusted_p": _as_float(hb.get("adjusted_p")),
                    "holm_rejected": bool(hb.get("rejected", False)),
                    "cohens_h": _as_float((value.get("cohens_h") or {}).get("h")),
                }
            )
    rows_out.sort(
        key=lambda item: (
            _null_sort_key(item.get("holm_adjusted_p")),
            -abs(_as_float(item.get("uplift")) or 0.0),
        )
    )
    return {"available": True, "rows": rows_out}


def _paired_success_cost_frame(
    *,
    rows: list[dict[str, Any]],
    condition_a: str,
    condition_b: str,
) -> pl.DataFrame:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        run_id = str(row.get("run_id") or "")
        case_id = str(row.get("case_id") or "")
        condition = str(row.get("condition") or "")
        if not run_id or not case_id or condition not in {condition_a, condition_b}:
            continue
        token_value = row.get("token_usage_total")
        token_total = (
            float(token_value)
            if isinstance(token_value, (int, float)) and math.isfinite(float(token_value))
            else None
        )
        normalized.append(
            {
                "run_id": run_id,
                "case_id": case_id,
                "condition": condition,
                "success": int(bool(row.get("success"))),
                "token_usage_total": token_total,
            }
        )

    if not normalized:
        return pl.DataFrame(
            {
                "run_id": [],
                "case_id": [],
                "a_success": [],
                "b_success": [],
                "a_tokens": [],
                "b_tokens": [],
            }
        )

    return (
        pl.DataFrame(normalized)
        .group_by(["run_id", "case_id"])
        .agg(
            pl.col("success").filter(pl.col("condition") == condition_a).first().alias("a_success"),
            pl.col("success").filter(pl.col("condition") == condition_b).first().alias("b_success"),
            pl.col("token_usage_total")
            .filter(pl.col("condition") == condition_a)
            .first()
            .alias("a_tokens"),
            pl.col("token_usage_total")
            .filter(pl.col("condition") == condition_b)
            .first()
            .alias("b_tokens"),
        )
        .drop_nulls(["a_success", "b_success"])
        .sort(["run_id", "case_id"])
    )


def _compute_dual_objective_summary(
    *,
    rows: list[dict[str, Any]],
    condition_a: str,
    condition_b: str,
    n_bootstrap: int,
    seed: int | None,
    fixed_budget_thresholds: list[int],
) -> dict[str, Any]:
    effect = _compute_effectiveness_summary(
        rows=rows,
        condition_a=condition_a,
        condition_b=condition_b,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    paired = _paired_success_cost_frame(
        rows=rows,
        condition_a=condition_a,
        condition_b=condition_b,
    )
    if paired.is_empty():
        return {
            "available": False,
            "error": "No matched A/B pairs found.",
            "effect": effect,
            "fixed_budget_thresholds": fixed_budget_thresholds,
        }

    matched_pairs = paired.height
    covered = paired.drop_nulls(["a_tokens", "b_tokens"])
    covered_pairs = covered.height
    result: dict[str, Any] = {
        "available": True,
        "effect": effect,
        "matched_pairs": matched_pairs,
        "token_covered_pairs": covered_pairs,
        "token_coverage_rate": (covered_pairs / float(matched_pairs))
        if matched_pairs > 0
        else None,
        "fixed_budget_thresholds": fixed_budget_thresholds,
    }
    if covered.is_empty():
        result["cost_per_success"] = {
            "available": False,
            "error": "No matched token-covered pairs.",
        }
        result["paired_cost_delta"] = {
            "available": False,
            "error": "No matched token-covered pairs.",
        }
        result["fixed_budget_success"] = []
        return result

    a_tokens = covered["a_tokens"].cast(pl.Float64).to_list()
    b_tokens = covered["b_tokens"].cast(pl.Float64).to_list()
    a_success = covered["a_success"].cast(pl.Int64).to_list()
    b_success = covered["b_success"].cast(pl.Int64).to_list()
    token_pairs = list(zip(a_tokens, b_tokens, strict=True))
    token_deltas = paired_numeric_deltas([(a_value, b_value) for a_value, b_value in token_pairs])
    a_successes = sum(a_success)
    b_successes = sum(b_success)
    result["cost_per_success"] = {
        "available": True,
        "coverage_pairs": covered_pairs,
        "a_successes": a_successes,
        "b_successes": b_successes,
        "a_total_tokens": sum(a_tokens),
        "b_total_tokens": sum(b_tokens),
        "a_value": (sum(a_tokens) / a_successes) if a_successes > 0 else None,
        "b_value": (sum(b_tokens) / b_successes) if b_successes > 0 else None,
    }
    mean_a = _mean(a_tokens)
    mean_b = _mean(b_tokens)
    result["paired_cost_delta"] = {
        "available": True,
        "coverage_pairs": covered_pairs,
        "mean_a": mean_a,
        "mean_b": mean_b,
        "mean_delta": _mean(token_deltas),
        "pct_change_vs_a": (
            ((mean_b - mean_a) / mean_a)
            if mean_a not in (None, 0.0) and mean_b is not None
            else None
        ),
        "bootstrap_ci_mean_delta": bootstrap_mean_ci(
            token_deltas,
            n_bootstrap=max(0, n_bootstrap),
            confidence=0.95,
            seed=seed,
        ),
    }
    fixed_budget_rows: list[dict[str, Any]] = []
    for threshold in fixed_budget_thresholds:
        a_under = sum(
            1
            for success, token_total in zip(a_success, a_tokens, strict=True)
            if success > 0 and token_total <= float(threshold)
        )
        b_under = sum(
            1
            for success, token_total in zip(b_success, b_tokens, strict=True)
            if success > 0 and token_total <= float(threshold)
        )
        coverage = covered_pairs if covered_pairs > 0 else None
        fixed_budget_rows.append(
            {
                "token_threshold": int(threshold),
                "coverage_pairs": covered_pairs,
                "a_successes_within_budget": a_under,
                "b_successes_within_budget": b_under,
                "a_rate": (a_under / float(coverage)) if coverage else None,
                "b_rate": (b_under / float(coverage)) if coverage else None,
                "delta": ((b_under - a_under) / float(coverage)) if coverage else None,
            }
        )
    result["fixed_budget_success"] = fixed_budget_rows
    return result


def _null_sort_key(value: Any) -> tuple[int, float]:
    if isinstance(value, (int, float)):
        return (0, float(value))
    return (1, math.inf)


def _compute_efficiency_summaries(
    *,
    rows: list[dict[str, Any]],
    condition_a: str,
    condition_b: str,
    n_bootstrap: int,
    seed: int | None,
) -> dict[str, dict[str, Any]]:
    metrics = {
        "token_usage_total": "Tokens total (lower is better)",
        "runtime_sec": "Runtime sec (lower is better)",
        "llm_calls_total": "LLM calls total (lower is better)",
    }
    out: dict[str, dict[str, Any]] = {}
    for metric, label in metrics.items():
        deltas, a_values, b_values = _paired_metric_deltas(
            rows=rows,
            condition_a=condition_a,
            condition_b=condition_b,
            metric=metric,
        )
        out[metric] = _summarize_paired_deltas(
            label=label,
            deltas=deltas,
            a_values=a_values,
            b_values=b_values,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
    return out


def _paired_metric_deltas(
    *,
    rows: list[dict[str, Any]],
    condition_a: str,
    condition_b: str,
    metric: str,
) -> tuple[list[float], list[float], list[float]]:
    normalized: list[dict[str, Any]] = []
    for row in rows:
        run_id = str(row.get("run_id") or "")
        case_id = str(row.get("case_id") or "")
        condition = str(row.get("condition") or "")
        if not run_id or not case_id or condition not in {condition_a, condition_b}:
            continue
        metric_value = row.get(metric)
        if not isinstance(metric_value, (int, float)):
            continue
        normalized.append(
            {
                "run_id": run_id,
                "case_id": case_id,
                "condition": condition,
                metric: float(metric_value),
            }
        )

    if not normalized:
        return [], [], []

    frame = pl.DataFrame(normalized)
    paired = (
        frame.group_by(["run_id", "case_id"])
        .agg(
            pl.col(metric).filter(pl.col("condition") == condition_a).first().alias("a_value"),
            pl.col(metric).filter(pl.col("condition") == condition_b).first().alias("b_value"),
        )
        .drop_nulls(["a_value", "b_value"])
        .sort(["run_id", "case_id"])
    )
    if paired.is_empty():
        return [], [], []

    a_values = paired["a_value"].cast(pl.Float64).to_list()
    b_values = paired["b_value"].cast(pl.Float64).to_list()
    deltas = [b_value - a_value for a_value, b_value in zip(a_values, b_values, strict=True)]
    return deltas, a_values, b_values


def _summarize_paired_deltas(
    *,
    label: str,
    deltas: list[float],
    a_values: list[float],
    b_values: list[float],
    n_bootstrap: int,
    seed: int | None,
) -> dict[str, Any]:
    if not deltas:
        return {"available": False, "label": label, "error": "No matched numeric pairs."}

    mean_delta = _mean(deltas)
    median_delta = _median(deltas)
    mean_a = _mean(a_values)
    mean_b = _mean(b_values)
    pct_change = (
        ((mean_b - mean_a) / mean_a) if mean_a not in (None, 0) and mean_b is not None else None
    )
    sign = _paired_sign_test(deltas)
    wilcoxon = _paired_wilcoxon_test(deltas)
    ci = bootstrap_mean_ci(deltas, n_bootstrap=max(0, n_bootstrap), confidence=0.95, seed=seed)
    return {
        "available": True,
        "label": label,
        "n_pairs": len(deltas),
        "mean_a": mean_a,
        "mean_b": mean_b,
        "mean_delta": mean_delta,
        "median_delta": median_delta,
        "pct_change_vs_a": pct_change,
        "sign_test": sign,
        "wilcoxon": wilcoxon,
        "bootstrap_ci_mean_delta": ci,
    }


def _paired_sign_test(deltas: list[float]) -> dict[str, Any]:
    positives = sum(1 for value in deltas if value > 0)
    negatives = sum(1 for value in deltas if value < 0)
    nonzero = positives + negatives
    if nonzero == 0:
        return {
            "n_nonzero": 0,
            "n_positive": 0,
            "n_negative": 0,
            "p_value": 1.0,
        }
    p_value = exact_binomial_p_value(positives, nonzero)
    return {
        "n_nonzero": nonzero,
        "n_positive": positives,
        "n_negative": negatives,
        "p_value": p_value,
    }


def _paired_wilcoxon_test(deltas: list[float]) -> dict[str, Any]:
    nonzero_deltas = [value for value in deltas if value != 0]
    n_nonzero = len(nonzero_deltas)
    if n_nonzero == 0:
        return {
            "n_nonzero": 0,
            "statistic": 0.0,
            "p_value": 1.0,
            "method": "wilcoxon_signed_rank",
        }
    try:
        result = cast(
            Any,
            stats.wilcoxon(
                nonzero_deltas,
                zero_method="wilcox",
                alternative="two-sided",
                method="auto",
            ),
        )
        return {
            "n_nonzero": n_nonzero,
            "statistic": float(result.statistic),
            "p_value": float(result.pvalue),
            "method": "wilcoxon_signed_rank",
        }
    except ValueError:
        return {
            "n_nonzero": n_nonzero,
            "statistic": None,
            "p_value": None,
            "method": "wilcoxon_signed_rank",
        }


def _compute_failure_diagnostics(
    *,
    raw_records: list[dict[str, Any]],
    condition_a: str,
    condition_b: str,
    top_n: int,
) -> dict[str, Any]:
    """Summarize attempt-level incidents and case-level overflow outcomes.

    Metric semantics:
    - ``http_error_attempts`` counts attempts where ``patch_meta.error == "http_error"``.
      This is an attempt-incident counter (retries included), not a terminal
      failure count.
    - ``context_overflow_attempts`` is the subset of attempts whose HTTP error
      body indicates context-size overflow.
    - ``patch_attempts`` counts attempts that proposed at least one file change.
    - ``test_touch_attempts`` counts patch attempts where proposed edits touched
      test-like paths (for example ``tests/``, ``test_*.py``, ``*_test.py``).
    - ``context_overflow_case_*`` fields are case-level aggregates over
      (run_id, case_id, condition), reporting overflow incidence and whether any
      attempt for the case eventually succeeded.
    """
    conditions = {condition_a, condition_b}
    flattened: list[dict[str, Any]] = []
    terminal_rows: dict[tuple[str, str, str], dict[str, Any]] = {}

    for record in raw_records:
        condition = str(record.get("condition") or "")
        if condition not in conditions:
            continue
        run_id = str(record.get("run_id") or "")
        case_id = str(record.get("case_id") or "")
        key = (run_id, case_id, condition)
        attempt = int(record.get("attempt") or 0)
        previous = terminal_rows.get(key)
        if previous is None or attempt >= int(previous.get("attempt") or 0):
            terminal_rows[key] = record

        patch_meta = record.get("patch_meta")
        error = ""
        body = ""
        patch_touches_tests = False
        patch_file_count = 0
        if isinstance(patch_meta, dict):
            error = str(patch_meta.get("error") or "")
            body = str(patch_meta.get("body") or "")
            patch_touches_tests = bool(patch_meta.get("patch_touches_tests"))
            patch_file_count_raw = patch_meta.get("patch_file_count")
            if isinstance(patch_file_count_raw, (int, float)):
                patch_file_count = max(0, int(patch_file_count_raw))
            elif isinstance(patch_meta.get("patch_file_paths"), list):
                patch_file_count = len(patch_meta.get("patch_file_paths") or [])

        overflow = "exceeds the available context size" in body
        success = bool(record.get("success"))
        flattened.append(
            {
                "run_id": run_id,
                "case_id": case_id,
                "condition": condition,
                "success": success,
                "is_http_error": error == "http_error",
                "is_overflow": overflow,
                "patch_touches_tests": patch_touches_tests,
                "has_patch": patch_file_count > 0,
            }
        )

    attempt_stats_by_condition: dict[str, dict[str, Any]] = {}
    case_stats_by_condition: dict[str, dict[str, Any]] = {}
    if flattened:
        frame = pl.DataFrame(flattened)
        attempt_stats = frame.group_by("condition").agg(
            pl.len().alias("attempts"),
            pl.col("is_http_error").cast(pl.Int64).sum().alias("http_error_attempts"),
            pl.col("is_overflow").cast(pl.Int64).sum().alias("context_overflow_attempts"),
            pl.col("has_patch").cast(pl.Int64).sum().alias("patch_attempts"),
            pl.col("patch_touches_tests").cast(pl.Int64).sum().alias("test_touch_attempts"),
        )
        attempt_stats_by_condition = {row["condition"]: row for row in attempt_stats.to_dicts()}

        case_stats = (
            frame.group_by(["run_id", "case_id", "condition"])
            .agg(
                pl.col("is_overflow").any().alias("case_overflow"),
                pl.col("success").any().alias("case_success"),
            )
            .group_by("condition")
            .agg(
                pl.len().alias("cases"),
                pl.col("case_overflow").cast(pl.Int64).sum().alias("context_overflow_cases"),
                (pl.col("case_overflow") & pl.col("case_success"))
                .cast(pl.Int64)
                .sum()
                .alias("context_overflow_case_successes"),
                (pl.col("case_overflow") & ~pl.col("case_success"))
                .cast(pl.Int64)
                .sum()
                .alias("context_overflow_case_failures"),
            )
        )
        case_stats_by_condition = {row["condition"]: row for row in case_stats.to_dicts()}

    terminal_reason_counts: dict[str, dict[str, int]] = {condition: {} for condition in conditions}
    terminal_usage_stats: dict[str, dict[str, int]] = {
        condition: {
            "usage_applicable_cases": 0,
            "usage_not_applicable_cases": 0,
            "usage_reported_cases": 0,
            "usage_missing_cases": 0,
            "timeout_missing_usage_cases": 0,
            "terminal_timeout_failures": 0,
            "terminal_incompatible_failures": 0,
            "terminal_forced_noop_failures": 0,
        }
        for condition in conditions
    }
    for record in terminal_rows.values():
        condition = str(record.get("condition") or "")
        if condition not in conditions:
            continue
        success = bool(record.get("success"))
        if not success:
            reason = _terminal_failure_reason(record)
            terminal_reason_counts[condition][reason] = (
                int(terminal_reason_counts[condition].get(reason, 0)) + 1
            )
        if _is_external_baseline_condition(condition):
            usage_stats = terminal_usage_stats[condition]
            usage_state = _baseline_terminal_usage_state(record)
            if usage_state == "available":
                usage_stats["usage_applicable_cases"] += 1
                usage_stats["usage_reported_cases"] += 1
            elif usage_state == "unavailable":
                usage_stats["usage_applicable_cases"] += 1
                usage_stats["usage_missing_cases"] += 1
                if _baseline_error(record) == "timeout":
                    usage_stats["timeout_missing_usage_cases"] += 1
            else:
                usage_stats["usage_not_applicable_cases"] += 1
            if not success and _baseline_error(record) == "timeout":
                usage_stats["terminal_timeout_failures"] += 1
            if not success and _compatibility_reason(record):
                usage_stats["terminal_incompatible_failures"] += 1
            if not success and str(record.get("note") or "") == "forced_noop_patch":
                usage_stats["terminal_forced_noop_failures"] += 1

    per_condition: dict[str, dict[str, Any]] = {}
    for condition in sorted(conditions):
        attempt_stats = attempt_stats_by_condition.get(condition, {})
        case_stats = case_stats_by_condition.get(condition, {})
        usage_stats = terminal_usage_stats.get(condition, {})

        attempts = int(attempt_stats.get("attempts", 0) or 0)
        http_error_attempts = int(attempt_stats.get("http_error_attempts", 0) or 0)
        overflow_attempts = int(attempt_stats.get("context_overflow_attempts", 0) or 0)
        patch_attempts = int(attempt_stats.get("patch_attempts", 0) or 0)
        test_touch_attempts = int(attempt_stats.get("test_touch_attempts", 0) or 0)
        cases = int(case_stats.get("cases", 0) or 0)
        overflow_cases = int(case_stats.get("context_overflow_cases", 0) or 0)
        overflow_case_successes = int(case_stats.get("context_overflow_case_successes", 0) or 0)
        overflow_case_failures = int(case_stats.get("context_overflow_case_failures", 0) or 0)
        top_reasons = sorted(
            terminal_reason_counts.get(condition, {}).items(),
            key=lambda item: (-int(item[1]), str(item[0])),
        )[:top_n]

        per_condition[condition] = {
            "attempts": attempts,
            "http_error_attempts": http_error_attempts,
            "http_error_attempt_rate": _ratio(http_error_attempts, attempts),
            "context_overflow_attempts": overflow_attempts,
            "context_overflow_attempt_rate": _ratio(overflow_attempts, attempts),
            "patch_attempts": patch_attempts,
            "test_touch_attempts": test_touch_attempts,
            "test_touch_attempt_rate": _ratio(test_touch_attempts, patch_attempts),
            "cases": cases,
            "context_overflow_cases": overflow_cases,
            "context_overflow_case_rate": _ratio(overflow_cases, cases),
            "context_overflow_case_successes": overflow_case_successes,
            "context_overflow_case_failures": overflow_case_failures,
            "top_reasons": top_reasons,
            "usage_applicable_cases": int(usage_stats.get("usage_applicable_cases", 0) or 0),
            "usage_not_applicable_cases": int(
                usage_stats.get("usage_not_applicable_cases", 0) or 0
            ),
            "usage_reported_cases": int(usage_stats.get("usage_reported_cases", 0) or 0),
            "usage_reported_case_rate": _ratio(
                int(usage_stats.get("usage_reported_cases", 0) or 0),
                int(usage_stats.get("usage_applicable_cases", 0) or 0),
            ),
            "usage_missing_cases": int(usage_stats.get("usage_missing_cases", 0) or 0),
            "usage_missing_case_rate": _ratio(
                int(usage_stats.get("usage_missing_cases", 0) or 0),
                int(usage_stats.get("usage_applicable_cases", 0) or 0),
            ),
            "timeout_missing_usage_cases": int(
                usage_stats.get("timeout_missing_usage_cases", 0) or 0
            ),
            "timeout_missing_usage_case_rate": _ratio(
                int(usage_stats.get("timeout_missing_usage_cases", 0) or 0),
                int(usage_stats.get("usage_applicable_cases", 0) or 0),
            ),
            "terminal_timeout_failures": int(usage_stats.get("terminal_timeout_failures", 0) or 0),
            "terminal_timeout_failure_rate": _ratio(
                int(usage_stats.get("terminal_timeout_failures", 0) or 0),
                cases,
            ),
            "terminal_incompatible_failures": int(
                usage_stats.get("terminal_incompatible_failures", 0) or 0
            ),
            "terminal_forced_noop_failures": int(
                usage_stats.get("terminal_forced_noop_failures", 0) or 0
            ),
        }
    return {"per_condition": per_condition}


def _pearson_correlation(xs: list[float], ys: list[float]) -> float | None:
    if len(xs) != len(ys) or len(xs) < 2:
        return None
    mean_x = sum(xs) / float(len(xs))
    mean_y = sum(ys) / float(len(ys))
    centered_x = [value - mean_x for value in xs]
    centered_y = [value - mean_y for value in ys]
    denom_x = math.sqrt(sum(value * value for value in centered_x))
    denom_y = math.sqrt(sum(value * value for value in centered_y))
    if denom_x <= 0.0 or denom_y <= 0.0:
        return None
    numer = sum(x * y for x, y in zip(centered_x, centered_y, strict=True))
    return numer / (denom_x * denom_y)


def _token_estimate_from_record(record: dict[str, Any]) -> float | None:
    attempt_meta = record.get("attempt_metadata")
    patch_meta = record.get("patch_meta")
    attempt_meta_dict = attempt_meta if isinstance(attempt_meta, dict) else {}
    patch_meta_dict = patch_meta if isinstance(patch_meta, dict) else {}
    available = attempt_meta_dict.get("prompt_token_estimate_available")
    if available is None:
        available = patch_meta_dict.get("prompt_token_estimate_available")
    if available is False:
        return None
    for source in (attempt_meta_dict, patch_meta_dict):
        value = source.get("prompt_tokens_estimate_pre_call")
        if isinstance(value, (int, float)) and math.isfinite(float(value)):
            return float(value)
    return None


def _provider_prompt_tokens_from_record(record: dict[str, Any]) -> float | None:
    attempt_meta = record.get("attempt_metadata")
    patch_meta = record.get("patch_meta")
    attempt_meta_dict = attempt_meta if isinstance(attempt_meta, dict) else {}
    patch_meta_dict = patch_meta if isinstance(patch_meta, dict) else {}
    usage_available = attempt_meta_dict.get("token_usage_available")
    if usage_available is None:
        usage_available = patch_meta_dict.get("token_usage_available")
    if usage_available is False:
        return None
    value = attempt_meta_dict.get("tokens_prompt")
    if isinstance(value, (int, float)) and math.isfinite(float(value)):
        return float(value)
    return None


def _compute_token_estimate_diagnostics(
    *,
    raw_records: list[dict[str, Any]],
    condition_a: str,
    condition_b: str,
) -> dict[str, Any]:
    per_condition: dict[str, dict[str, Any]] = {}
    for condition in (condition_a, condition_b):
        provider_values: list[float] = []
        estimate_values: list[float] = []
        total_attempts = 0
        provider_covered_attempts = 0
        estimate_covered_attempts = 0
        for record in raw_records:
            if str(record.get("condition") or "") != condition:
                continue
            total_attempts += 1
            provider_value = _provider_prompt_tokens_from_record(record)
            estimate_value = _token_estimate_from_record(record)
            if provider_value is not None:
                provider_covered_attempts += 1
            if estimate_value is not None:
                estimate_covered_attempts += 1
            if provider_value is None or estimate_value is None:
                continue
            provider_values.append(provider_value)
            estimate_values.append(estimate_value)

        paired_attempts = len(provider_values)
        correlation = _pearson_correlation(estimate_values, provider_values)
        per_condition[condition] = {
            "total_attempts": total_attempts,
            "provider_covered_attempts": provider_covered_attempts,
            "estimate_covered_attempts": estimate_covered_attempts,
            "paired_attempts": paired_attempts,
            "paired_attempt_rate": (
                paired_attempts / float(total_attempts) if total_attempts > 0 else None
            ),
            "correlation": correlation,
            "mean_abs_error": (
                sum(
                    abs(estimate - provider)
                    for estimate, provider in zip(estimate_values, provider_values, strict=True)
                )
                / float(paired_attempts)
                if paired_attempts > 0
                else None
            ),
            "available": correlation is not None,
            "error": (
                ""
                if correlation is not None
                else "insufficient_paired_coverage"
                if paired_attempts < 2
                else "zero_variance"
            ),
        }
    return {"per_condition": per_condition}


def _failure_reason(*, error: str, status: str, note: str, overflow: bool) -> str:
    if overflow:
        return "http_error:context_exceeded"
    if error and status:
        return f"{error}:{status}"
    if error:
        return error
    if note:
        return f"note:{note}"
    return "unknown"


def _baseline_meta_from_record(record: dict[str, Any]) -> dict[str, Any]:
    patch_meta = record.get("patch_meta")
    if not isinstance(patch_meta, dict):
        return {}
    baseline_meta = patch_meta.get("baseline_meta")
    return baseline_meta if isinstance(baseline_meta, dict) else {}


def _baseline_metrics_from_record(record: dict[str, Any]) -> dict[str, Any]:
    patch_meta = record.get("patch_meta")
    if not isinstance(patch_meta, dict):
        return {}
    baseline_metrics = patch_meta.get("baseline_metrics")
    return baseline_metrics if isinstance(baseline_metrics, dict) else {}


def _baseline_error(record: dict[str, Any]) -> str:
    return str(_baseline_meta_from_record(record).get("error") or "")


def _compatibility_reason(record: dict[str, Any]) -> str:
    patch_meta = record.get("patch_meta")
    if not isinstance(patch_meta, dict):
        return ""
    return str(patch_meta.get("compatibility_reason") or "")


def _baseline_usage_available(record: dict[str, Any]) -> bool:
    return bool(_baseline_metrics_from_record(record).get("usage_available"))


def _baseline_llm_call_count(record: dict[str, Any]) -> int:
    baseline_metrics = _baseline_metrics_from_record(record)
    value = baseline_metrics.get("llm_call_count")
    if isinstance(value, int) and value >= 0:
        return value

    patch_meta = record.get("patch_meta")
    if isinstance(patch_meta, dict):
        patch_value = patch_meta.get("llm_call_count")
        if isinstance(patch_value, int) and patch_value >= 0:
            return patch_value

    attempt_metadata = record.get("attempt_metadata")
    if isinstance(attempt_metadata, dict):
        attempt_value = attempt_metadata.get("llm_call_count")
        if isinstance(attempt_value, int) and attempt_value >= 0:
            return attempt_value

    return 0


def _baseline_terminal_usage_state(record: dict[str, Any]) -> str:
    llm_call_count = _baseline_llm_call_count(record)
    if _baseline_usage_available(record):
        return "available"
    if llm_call_count > 0:
        return "unavailable"
    return "not_applicable"


def _terminal_failure_reason(record: dict[str, Any]) -> str:
    patch_meta = record.get("patch_meta")
    note = str(record.get("note") or "")
    error = ""
    status = ""
    body = ""
    if isinstance(patch_meta, dict):
        error = str(patch_meta.get("error") or "")
        status_val = patch_meta.get("status")
        status = str(status_val) if status_val is not None else ""
        body = str(patch_meta.get("body") or "")
    overflow = "exceeds the available context size" in body
    if overflow:
        return "http_error:context_exceeded"

    compatibility_reason = _compatibility_reason(record)
    if compatibility_reason:
        return _annotate_runner_note(
            f"case_incompatible:{compatibility_reason}",
            note=note,
        )

    baseline_error = _baseline_error(record)
    if baseline_error:
        return _annotate_runner_note(
            f"baseline_error:{baseline_error}",
            note=note,
        )

    if error and status:
        return f"{error}:{status}"
    if error:
        return error
    if note == "forced_noop_patch":
        return "runner_forced_noop_after_no_diff"
    if note:
        return f"note:{note}"
    return "unknown"


def _annotate_runner_note(reason: str, *, note: str) -> str:
    note_clean = str(note or "").strip()
    if not note_clean or note_clean in {"success", "tests_still_failing"}:
        return reason
    return f"{reason} [runner_note={note_clean}]"


def _ratio(numer: int, denom: int) -> float | None:
    if denom <= 0:
        return None
    return float(numer) / float(denom)


_TIMING_SUMMARY_KEYS = (
    "stage_a_llm_sec",
    "stage_a_context_fulfillment_sec",
    "snapshot_read_sec",
    "adaptive_controller_prepare_sec",
    "adaptive_controller_summarizer_sec",
    "adaptive_controller_sidecar_build_sec",
    "adaptive_controller_sidecar_write_sec",
    "baseline_case_conversion_sec",
    "baseline_sidecar_load_sec",
    "context_compaction_llm_sec",
    "prompt_assembly_sec",
    "stage_b_llm_sec",
    "test_before_sec",
    "test_after_sec",
    "recovery_llm_sec",
    "llmdebug_total_sec",
    "llm_total_sec",
    "overhead_sec",
)

_TOKEN_SUMMARY_KEYS = (
    "stage_a_tokens_total",
    "stage_b_tokens_total",
    "compaction_tokens_total",
    "recovery_tokens_total",
    "tool_response_chars",
    "snapshot_chars",
)


def _compute_overhead_analysis(
    *,
    rows: list[dict[str, Any]],
    condition_a: str,
    condition_b: str,
) -> dict[str, Any]:
    """Compute per-condition timing and token breakdown summaries."""
    per_condition: dict[str, dict[str, Any]] = {}
    for condition in (condition_a, condition_b):
        cond_rows = [r for r in rows if r.get("condition") == condition]
        if not cond_rows:
            continue

        timing_accum: dict[str, list[float]] = {k: [] for k in _TIMING_SUMMARY_KEYS}
        token_accum: dict[str, list[float]] = {k: [] for k in _TOKEN_SUMMARY_KEYS}

        for row in cond_rows:
            timing = row.get("timing_breakdown_sum")
            if isinstance(timing, dict):
                for key in _TIMING_SUMMARY_KEYS:
                    val = timing.get(key)
                    if isinstance(val, (int, float)):
                        timing_accum[key].append(float(val))
            tokens = row.get("token_breakdown_sum")
            if isinstance(tokens, dict):
                for key in _TOKEN_SUMMARY_KEYS:
                    val = tokens.get(key)
                    if isinstance(val, (int, float)):
                        token_accum[key].append(float(val))

        timing_mean = {k: _mean(v) for k, v in timing_accum.items()}
        token_mean = {k: _mean(v) for k, v in token_accum.items()}

        llm_total = timing_mean.get("llm_total_sec")
        overhead = timing_mean.get("overhead_sec")
        llmdebug_total = timing_mean.get("llmdebug_total_sec")
        runtime_values = [
            float(r.get("runtime_sec") or 0.0) for r in cond_rows if r.get("runtime_sec")
        ]
        mean_runtime = _mean(runtime_values)
        llm_pct = (llm_total / mean_runtime * 100.0) if llm_total and mean_runtime else None
        overhead_pct = (overhead / mean_runtime * 100.0) if overhead and mean_runtime else None
        llmdebug_pct = (
            (llmdebug_total / mean_runtime * 100.0) if llmdebug_total and mean_runtime else None
        )

        per_condition[condition] = {
            "cases": len(cond_rows),
            "timing_mean": timing_mean,
            "token_mean": token_mean,
            "llm_pct_of_runtime": llm_pct,
            "overhead_pct_of_runtime": overhead_pct,
            "llmdebug_pct_of_runtime": llmdebug_pct,
        }

    return {"per_condition": per_condition}


def _build_run_context(
    *,
    rows: list[dict[str, Any]],
    raw_records: list[dict[str, Any]],
    eval_log_path: str,
) -> dict[str, Any]:
    run_ids = sorted({str(row.get("run_id") or "") for row in rows if row.get("run_id")})
    patchers = sorted(
        {
            str((record.get("run_metadata") or {}).get("patcher_name") or "")
            for record in raw_records
            if isinstance(record.get("run_metadata"), dict)
        }
    )
    model_names = sorted(
        {
            str((record.get("model_metadata") or {}).get("model") or "")
            for record in raw_records
            if isinstance(record.get("model_metadata"), dict)
        }
    )
    out: dict[str, Any] = {
        "run_ids": [value for value in run_ids if value],
        "n_case_condition_rows": len(rows),
        "patchers": [value for value in patchers if value],
        "models": [value for value in model_names if value],
    }
    if eval_log_path:
        log_path = Path(eval_log_path)
        if log_path.exists():
            out["eval_log"] = _parse_eval_log(log_path.read_text(encoding="utf-8"))
            out["eval_log_path"] = str(log_path)
    return out


def _parse_eval_log(text: str) -> dict[str, Any]:
    job_feedback = _parse_job_feedback_block(text)
    progress_elapsed = None
    match = re.search(
        r"Progress:\s*finished\s+executed=\d+,\s*skipped=\d+,\s*total=\d+,\s*elapsed=([^\s]+)",
        text,
    )
    if match:
        progress_elapsed = match.group(1).strip()
    if progress_elapsed:
        job_feedback["runner_elapsed"] = progress_elapsed
    return job_feedback


def _parse_job_feedback_block(text: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    marker = "============================= JOB FEEDBACK ============================="
    idx = text.rfind(marker)
    if idx < 0:
        return out
    block = text[idx:].splitlines()
    for line in block:
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        k = key.strip().lower()
        v = value.strip()
        if not k:
            continue
        out[k] = v

    energy_line = out.get("energy consumed")
    if isinstance(energy_line, str):
        match = re.search(r"([0-9.]+)\s+Joule\s*/\s*([0-9.]+)\s+Watthours", energy_line)
        if match:
            out["energy_joule"] = float(match.group(1))
            out["energy_watthours"] = float(match.group(2))
    avg_power = out.get("average node power draw")
    if isinstance(avg_power, str):
        match = re.search(r"([0-9.]+)\s+Watt", avg_power)
        if match:
            out["average_node_power_watt"] = float(match.group(1))
    return out


def _print_report(
    *,
    run_context: dict[str, Any],
    condition_a: str,
    condition_b: str,
    primary_slice: set[str],
    overall_effect: dict[str, Any],
    primary_effect: dict[str, Any],
    required_effect: dict[str, Any],
    control_effect: dict[str, Any],
    category_stats: dict[str, Any],
    efficiency: dict[str, dict[str, Any]],
    dual_objective: dict[str, Any] | None = None,
    token_estimate_diagnostics: dict[str, Any] | None = None,
    condition_summary: dict[str, Any] | None = None,
    failures: dict[str, Any] | None = None,
    overhead: dict[str, Any] | None = None,
    input_errors: dict[str, int] | None = None,
) -> None:
    dual_objective = dual_objective if isinstance(dual_objective, dict) else {}
    token_estimate_diagnostics = (
        token_estimate_diagnostics if isinstance(token_estimate_diagnostics, dict) else {}
    )
    condition_summary = condition_summary if isinstance(condition_summary, dict) else {}
    failures = failures if isinstance(failures, dict) else {}
    overhead = overhead if isinstance(overhead, dict) else {}
    input_errors = input_errors if isinstance(input_errors, dict) else {}
    print("Run Metadata")
    print(f"- Run IDs: {', '.join(run_context.get('run_ids', [])) or 'n/a'}")
    print(f"- Patcher(s): {', '.join(run_context.get('patchers', [])) or 'n/a'}")
    print(f"- Model(s): {', '.join(run_context.get('models', [])) or 'n/a'}")
    print(f"- Case-condition rows: {run_context.get('n_case_condition_rows', 0)}")
    _print_input_errors(input_errors)

    print("\nPrimary Endpoint")
    print(f"- Primary snapshot slice: {', '.join(sorted(primary_slice))}")
    _print_effect_line("Primary", condition_a, condition_b, primary_effect)

    print("\nSecondary Slices")
    _print_effect_line("All cases", condition_a, condition_b, overall_effect)
    _print_effect_line("Required only", condition_a, condition_b, required_effect)
    _print_effect_line(
        "Controls (snapshot_dependency=none)", condition_a, condition_b, control_effect
    )

    print("\nSecondary Efficiency (paired deltas, B - A)")
    for metric in ("token_usage_total", "runtime_sec", "llm_calls_total"):
        summary = efficiency.get(metric, {})
        _print_efficiency_line(summary)

    print("\nDual Objective")
    _print_dual_objective(dual_objective, condition_a, condition_b)

    print("\nBaseline Diagnostics")
    baseline_lines = _baseline_diagnostic_lines(condition_summary=condition_summary)
    if not baseline_lines:
        print("- none")
    else:
        for line in baseline_lines:
            print(line)

    print("\nFailure Diagnostics")
    per_condition = failures.get("per_condition", {})
    for condition in sorted(per_condition):
        stats = per_condition[condition]
        print(
            f"- {condition}: context-overflow attempts="
            f"{stats.get('context_overflow_attempts', 0)}/{stats.get('attempts', 0)} "
            f"({_fmt_pct(stats.get('context_overflow_attempt_rate'))}), "
            f"context-overflow cases={stats.get('context_overflow_cases', 0)}/"
            f"{stats.get('cases', 0)} ({_fmt_pct(stats.get('context_overflow_case_rate'))}), "
            f"http_error attempts={stats.get('http_error_attempts', 0)}, "
            f"terminal timeouts={stats.get('terminal_timeout_failures', 0)}/"
            f"{stats.get('cases', 0)} ({_fmt_pct(stats.get('terminal_timeout_failure_rate'))}), "
            f"test-touch patch attempts={stats.get('test_touch_attempts', 0)}/"
            f"{stats.get('patch_attempts', 0)} "
            f"({_fmt_pct(stats.get('test_touch_attempt_rate'))})"
        )
        usage_applicable_cases = int(stats.get("usage_applicable_cases", 0) or 0)
        usage_not_applicable_cases = int(stats.get("usage_not_applicable_cases", 0) or 0)
        usage_reported_cases = int(stats.get("usage_reported_cases", 0) or 0)
        usage_missing_cases = int(stats.get("usage_missing_cases", 0) or 0)
        timeout_missing_usage_cases = int(stats.get("timeout_missing_usage_cases", 0) or 0)
        if usage_reported_cases or usage_missing_cases or usage_not_applicable_cases:
            print(
                "  token telemetry coverage (terminal cases): "
                f"reported={usage_reported_cases}/{usage_applicable_cases} "
                f"({_fmt_pct(stats.get('usage_reported_case_rate'))}), "
                f"missing={usage_missing_cases}/{usage_applicable_cases} "
                f"({_fmt_pct(stats.get('usage_missing_case_rate'))}), "
                f"timeout-backed missing={timeout_missing_usage_cases}/{usage_applicable_cases} "
                f"({_fmt_pct(stats.get('timeout_missing_usage_case_rate'))}), "
                f"not-applicable={usage_not_applicable_cases}/{stats.get('cases', 0)}"
            )
        token_estimate_stats = (
            token_estimate_diagnostics.get("per_condition", {})
            if isinstance(token_estimate_diagnostics, dict)
            else {}
        )
        estimate_row = (
            token_estimate_stats.get(condition) if isinstance(token_estimate_stats, dict) else None
        )
        if isinstance(estimate_row, dict) and int(estimate_row.get("total_attempts", 0) or 0) > 0:
            print(
                "  prompt-token estimate coverage (attempts): "
                f"provider={estimate_row.get('provider_covered_attempts', 0)}/"
                f"{estimate_row.get('total_attempts', 0)}, "
                f"estimate={estimate_row.get('estimate_covered_attempts', 0)}/"
                f"{estimate_row.get('total_attempts', 0)}, "
                f"paired={estimate_row.get('paired_attempts', 0)}/"
                f"{estimate_row.get('total_attempts', 0)} "
                f"({_fmt_pct(estimate_row.get('paired_attempt_rate'))})"
            )
            if estimate_row.get("available"):
                print(
                    "  prompt-token estimate correlation: "
                    f"r={_fmt_float(estimate_row.get('correlation'))}, "
                    f"mean_abs_error={_fmt_float(estimate_row.get('mean_abs_error'))}"
                )
            else:
                print(
                    "  prompt-token estimate correlation: "
                    f"unavailable ({estimate_row.get('error', 'unknown')})"
                )
        top = stats.get("top_reasons", [])
        if isinstance(top, list) and top:
            rendered = ", ".join(f"{name}:{count}" for name, count in top)
            print(f"  top terminal failure reasons: {rendered}")

    print("\nCategory Highlights (primary slice)")
    if not category_stats.get("available"):
        print(f"- unavailable: {category_stats.get('error', 'unknown')}")
    else:
        rows = category_stats.get("rows", [])
        if not rows:
            print("- no matched category pairs")
        else:
            for row in rows[:12]:
                uplift = _fmt_signed_pp(row.get("uplift"))
                p_val = _fmt_p(row.get("mcnemar_p"))
                adj = _fmt_p(row.get("holm_adjusted_p"))
                sig = "yes" if row.get("holm_rejected") else "no"
                print(
                    f"- {row.get('category')}: uplift={uplift}, n={row.get('n_pairs', 0)}, "
                    f"McNemar p={p_val}, Holm adj p={adj}, Holm sig={sig}"
                )

    _print_overhead_analysis(overhead, condition_a, condition_b)

    print("\nRun-Level System Context")
    eval_log = run_context.get("eval_log", {})
    if isinstance(eval_log, dict) and eval_log:
        _print_if_present(eval_log, "job wall-clock time", "Job wall-clock")
        _print_if_present(eval_log, "runner_elapsed", "Runner elapsed")
        _print_if_present(eval_log, "cpu utilized", "CPU utilized")
        _print_if_present(eval_log, "cpu efficiency", "CPU efficiency")
        _print_if_present(eval_log, "memory utilized", "Memory utilized")
        _print_if_present(eval_log, "memory efficiency", "Memory efficiency")
        if "energy_joule" in eval_log and "energy_watthours" in eval_log:
            print(
                "- Energy consumed (run-level): "
                f"{_fmt_float(eval_log.get('energy_joule'))} J / "
                f"{_fmt_float(eval_log.get('energy_watthours'))} Wh"
            )
        _print_if_present(eval_log, "average node power draw", "Average node power draw")
    else:
        print("- unavailable (no --eval-log or parseable job feedback block)")

    print("\nOne-Line Conclusion")
    print(_conclusion_line(condition_a=condition_a, condition_b=condition_b, effect=primary_effect))


def _baseline_diagnostic_lines(
    *,
    condition_summary: dict[str, Any],
) -> list[str]:
    lines: list[str] = []
    for condition in sorted(condition_summary):
        if not _is_external_baseline_condition(condition):
            continue
        stats = condition_summary.get(condition)
        if not isinstance(stats, dict):
            continue
        cases = int(stats.get("cases") or 0)
        parts: list[str] = []
        _append_covered_pct(
            parts, stats, cases, "baseline_direct_gen_pass_rate", "direct_gen_pass_rate"
        )
        _append_covered_pct(
            parts,
            stats,
            cases,
            "baseline_debug_added_value_rate",
            "debug_added_value_rate",
        )
        _append_covered_pct(
            parts,
            stats,
            cases,
            "baseline_method_instrumentation_rate",
            "instrumentation_rate",
        )
        _append_covered_pct(
            parts, stats, cases, "baseline_method_two_step_repair_rate", "two_step_rate"
        )
        teacher_sec = stats.get("baseline_teacher_duration_sec_total_mean")
        if isinstance(teacher_sec, (int, float)):
            parts.append(f"teacher_sec={_fmt_float(teacher_sec)}")
        learner_sec = stats.get("baseline_learner_duration_sec_total_mean")
        if isinstance(learner_sec, (int, float)):
            parts.append(f"learner_sec={_fmt_float(learner_sec)}")
        postcheck_sec = stats.get("baseline_postcheck_duration_sec_total_mean")
        if isinstance(postcheck_sec, (int, float)):
            parts.append(f"postcheck_sec={_fmt_float(postcheck_sec)}")
        empty_count = stats.get("baseline_learner_empty_response_count_mean")
        if isinstance(empty_count, (int, float)):
            parts.append(f"empty_learner_mean={_fmt_float(empty_count)}")
        attempts_success = stats.get("mean_attempts_to_success")
        if isinstance(attempts_success, (int, float)):
            parts.append(f"mean_attempts_success={_fmt_float(attempts_success)}")
        attempts_failed = stats.get("mean_attempts_when_failed")
        if isinstance(attempts_failed, (int, float)):
            parts.append(f"mean_attempts_failed={_fmt_float(attempts_failed)}")
        _append_covered_pct(parts, stats, cases, "baseline_is_passing_rate", "is_passing_rate")
        _append_covered_pct(
            parts,
            stats,
            cases,
            "baseline_terminal_timeout_rate",
            "terminal_timeout_rate",
        )
        _append_attempt_ratio(
            parts,
            stats,
            numerator_field="baseline_iteration_budget_exhausted_attempts_total",
            denominator_field="baseline_attempt_summary_count_total",
            label="iteration_budget_exhausted_attempts",
        )
        _append_attempt_ratio(
            parts,
            stats,
            numerator_field="baseline_no_write_buggy_py_attempts_total",
            denominator_field="baseline_no_write_buggy_py_attempts_covered",
            label="no_write_attempts",
        )
        _append_attempt_ratio(
            parts,
            stats,
            numerator_field="baseline_no_test_attempts_total",
            denominator_field="baseline_no_test_attempts_covered",
            label="no_test_attempts",
        )
        subcall_total = stats.get("baseline_subcall_count_total")
        subcall_covered = stats.get("baseline_subcall_count_attempts_covered")
        if (
            isinstance(subcall_total, int)
            and isinstance(subcall_covered, int)
            and subcall_covered > 0
        ):
            parts.append(
                f"subcalls={subcall_total}/{subcall_covered} "
                f"({_fmt_float(stats.get('baseline_subcall_count_per_attempt'))}/attempt)"
            )
        test_run_total = stats.get("baseline_test_run_count_total")
        test_run_covered = stats.get("baseline_test_run_count_attempts_covered")
        if (
            isinstance(test_run_total, int)
            and isinstance(test_run_covered, int)
            and test_run_covered > 0
        ):
            parts.append(
                f"test_runs={test_run_total}/{test_run_covered} "
                f"({_fmt_float(stats.get('baseline_test_run_count_per_attempt'))}/attempt)"
            )
        _append_covered_pct(
            parts,
            stats,
            cases,
            "baseline_feedback_improved_case_rate",
            "feedback_improved_case_rate",
        )
        _append_covered_pct(
            parts,
            stats,
            cases,
            "baseline_feedback_regressed_case_rate",
            "feedback_regressed_case_rate",
        )
        final_fail_mean = stats.get("baseline_feedback_final_fail_count_mean")
        if isinstance(final_fail_mean, (int, float)):
            parts.append(f"feedback_final_fail_mean={_fmt_float(final_fail_mean)}")
        stopped_reason_counts = stats.get("baseline_stopped_reason_counts")
        if isinstance(stopped_reason_counts, dict) and stopped_reason_counts:
            rendered = ", ".join(
                f"{name}:{count}"
                for name, count in sorted(
                    stopped_reason_counts.items(),
                    key=lambda item: (-int(item[1]), str(item[0])),
                )[:3]
            )
            parts.append(f"stopped_reasons={rendered}")
        fidelity_counts = stats.get("baseline_fidelity_level_counts")
        if isinstance(fidelity_counts, dict) and fidelity_counts:
            rendered = ", ".join(
                f"{name}:{count}"
                for name, count in sorted(
                    fidelity_counts.items(),
                    key=lambda item: (-int(item[1]), str(item[0])),
                )[:2]
            )
            parts.append(f"fidelity={rendered}")
        repair_scope_counts = stats.get("baseline_repair_scope_policy_counts")
        if isinstance(repair_scope_counts, dict) and repair_scope_counts:
            rendered = ", ".join(
                f"{name}:{count}"
                for name, count in sorted(
                    repair_scope_counts.items(),
                    key=lambda item: (-int(item[1]), str(item[0])),
                )[:2]
            )
            parts.append(f"repair_scope={rendered}")
        prompt_contract_counts = stats.get("baseline_prompt_contract_kind_counts")
        if isinstance(prompt_contract_counts, dict) and prompt_contract_counts:
            rendered = ", ".join(
                f"{name}:{count}"
                for name, count in sorted(
                    prompt_contract_counts.items(),
                    key=lambda item: (-int(item[1]), str(item[0])),
                )[:2]
            )
            parts.append(f"prompt_contract={rendered}")
        hybrid_guidance_counts = stats.get("baseline_hybrid_guidance_mode_counts")
        if isinstance(hybrid_guidance_counts, dict) and hybrid_guidance_counts:
            rendered = ", ".join(
                f"{name}:{count}"
                for name, count in sorted(
                    hybrid_guidance_counts.items(),
                    key=lambda item: (-int(item[1]), str(item[0])),
                )[:2]
            )
            parts.append(f"hybrid_guidance={rendered}")
        if parts:
            lines.append(f"- {condition}: " + ", ".join(parts))
    return lines


def _append_covered_pct(
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
    parts.append(f"{label}={_fmt_pct(value)} (coverage {covered_cases}/{cases})")


def _append_attempt_ratio(
    parts: list[str],
    stats: dict[str, Any],
    numerator_field: str,
    denominator_field: str,
    label: str,
) -> None:
    numerator = stats.get(numerator_field)
    denominator = stats.get(denominator_field)
    if not isinstance(numerator, int) or not isinstance(denominator, int) or denominator <= 0:
        return
    parts.append(f"{label}={numerator}/{denominator} ({_fmt_pct(_ratio(numerator, denominator))})")


def _print_if_present(data: dict[str, Any], key: str, label: str) -> None:
    value = data.get(key)
    if value not in (None, ""):
        print(f"- {label}: {value}")


_TIMING_DISPLAY_LABELS: dict[str, str] = {
    "stage_a_llm_sec": "Stage-A LLM",
    "stage_a_context_fulfillment_sec": "Stage-A context fulfillment",
    "snapshot_read_sec": "Snapshot read (llmdebug)",
    "adaptive_controller_prepare_sec": "Adaptive controller prepare",
    "adaptive_controller_summarizer_sec": "Adaptive controller summarizer",
    "adaptive_controller_sidecar_build_sec": "Adaptive controller sidecar build",
    "adaptive_controller_sidecar_write_sec": "Adaptive controller sidecar write",
    "baseline_case_conversion_sec": "Baseline case conversion",
    "baseline_sidecar_load_sec": "Baseline sidecar load",
    "context_compaction_llm_sec": "Context compaction LLM",
    "prompt_assembly_sec": "Prompt assembly",
    "stage_b_llm_sec": "Stage-B LLM (patch gen)",
    "baseline_internal_total_sec": "Baseline internal total",
    "baseline_eval_sec": "Baseline internal eval",
    "baseline_orchestration_sec": "Baseline orchestration",
    "test_before_sec": "Test before (pytest)",
    "test_after_sec": "Test after (pytest)",
    "recovery_llm_sec": "Recovery LLM",
    "llmdebug_total_sec": "llmdebug total",
    "llm_total_sec": "LLM total (all phases)",
    "overhead_sec": "Framework overhead",
}

_TOKEN_DISPLAY_LABELS: dict[str, str] = {
    "stage_a_tokens_total": "Stage-A tokens",
    "stage_b_tokens_total": "Stage-B tokens",
    "compaction_tokens_total": "Compaction tokens",
    "recovery_tokens_total": "Recovery tokens",
    "tool_response_chars": "Tool response chars",
    "snapshot_chars": "Snapshot chars",
}


def _print_overhead_analysis(overhead: dict[str, Any], condition_a: str, condition_b: str) -> None:
    """Print the Framework Overhead Analysis section."""
    per_condition = overhead.get("per_condition", {})
    if not per_condition:
        print("\nFramework Overhead Analysis")
        print("- unavailable (no timing breakdown data)")
        return

    print("\nFramework Overhead Analysis")
    for condition in (condition_a, condition_b):
        stats = per_condition.get(condition)
        if not stats:
            continue
        print(f"\n  [{condition}] (n={stats.get('cases', 0)} cases)")

        llm_pct = stats.get("llm_pct_of_runtime")
        overhead_pct = stats.get("overhead_pct_of_runtime")
        llmdebug_pct = stats.get("llmdebug_pct_of_runtime")
        if llm_pct is not None:
            print(f"  LLM % of runtime: {_fmt_float(llm_pct)}%")
        if llmdebug_pct is not None:
            print(f"  llmdebug % of runtime: {_fmt_float(llmdebug_pct)}%")
        if overhead_pct is not None:
            print(f"  Framework overhead % of runtime: {_fmt_float(overhead_pct)}%")

        timing_mean = stats.get("timing_mean", {})
        if any(v is not None for v in timing_mean.values()):
            print("  Timing breakdown (mean sec per case):")
            for key, label in _TIMING_DISPLAY_LABELS.items():
                val = timing_mean.get(key)
                if val is not None:
                    print(f"    {label}: {_fmt_float(val)}")

        token_mean = stats.get("token_mean", {})
        if any(v is not None for v in token_mean.values()):
            print("  Token breakdown (mean per case):")
            for key, label in _TOKEN_DISPLAY_LABELS.items():
                val = token_mean.get(key)
                if val is not None:
                    print(f"    {label}: {_fmt_float(val)}")


def _conclusion_line(*, condition_a: str, condition_b: str, effect: dict[str, Any]) -> str:
    if not effect.get("available"):
        return (
            f"{condition_b} vs {condition_a}: primary slice unavailable "
            f"({effect.get('error', 'no matched pairs')})."
        )
    uplift_pp = effect.get("uplift_pp")
    if not isinstance(uplift_pp, (int, float)):
        return f"{condition_b} vs {condition_a}: primary slice did not yield numeric uplift."
    direction = "better" if uplift_pp > 0 else "worse" if uplift_pp < 0 else "equal"
    return (
        f"On snapshot-relevant cases, {condition_b} is {abs(uplift_pp):.2f} pp {direction} than "
        f"{condition_a} (McNemar p={_fmt_p((effect.get('mcnemar') or {}).get('p_value'))})."
    )


def _print_effect_line(
    label: str,
    condition_a: str,
    condition_b: str,
    effect: dict[str, Any],
) -> None:
    if not effect.get("available"):
        print(f"- {label}: unavailable ({effect.get('error', 'no matched pairs')})")
        return

    a_rate = effect.get("a_rate")
    b_rate = effect.get("b_rate")
    uplift_pp = effect.get("uplift_pp")
    direction = "better" if isinstance(uplift_pp, (int, float)) and uplift_pp > 0 else "worse"
    if isinstance(uplift_pp, (int, float)) and abs(uplift_pp) < 1e-12:
        direction = "equal"
    if not isinstance(uplift_pp, (int, float)):
        print(f"- {label}: unavailable (non-numeric uplift)")
        return
    mcn = effect.get("mcnemar", {})
    bci = effect.get("bootstrap_ci", {})
    ch = effect.get("cohens_h", {})
    agreement = effect.get("agreement", {})
    print(
        f"- {label}: {condition_b} is {abs(float(uplift_pp)):.2f} pp {direction} than {condition_a} "
        f"({_fmt_pct(a_rate)} -> {_fmt_pct(b_rate)}), n={effect.get('n_pairs', 0)}, "
        f"McNemar p={_fmt_p(mcn.get('p_value'))} "
        f"(n_01={mcn.get('n_01', 0)}, n_10={mcn.get('n_10', 0)}), "
        f"95% CI[{_fmt_pct_signed(bci.get('ci_lower'))}, {_fmt_pct_signed(bci.get('ci_upper'))}], "
        f"Cohen h={_fmt_float(ch.get('h'))}, "
        f"Cohen kappa={_fmt_float((agreement or {}).get('kappa'))}"
    )


def _print_efficiency_line(summary: dict[str, Any]) -> None:
    if not summary.get("available"):
        print(
            f"- {summary.get('label', 'metric')}: unavailable ({summary.get('error', 'unknown')})"
        )
        return
    sign = summary.get("sign_test", {})
    wilcoxon = summary.get("wilcoxon", {})
    ci = summary.get("bootstrap_ci_mean_delta", {})
    print(
        f"- {summary.get('label')}: mean delta={_fmt_float(summary.get('mean_delta'))}, "
        f"median delta={_fmt_float(summary.get('median_delta'))}, "
        f"mean A={_fmt_float(summary.get('mean_a'))}, "
        f"mean B={_fmt_float(summary.get('mean_b'))}, "
        f"pct vs A={_fmt_pct_signed(summary.get('pct_change_vs_a'))}, "
        f"sign-test p={_fmt_p(sign.get('p_value'))} "
        f"(+={sign.get('n_positive', 0)}, -={sign.get('n_negative', 0)}), "
        f"Wilcoxon p={_fmt_p(wilcoxon.get('p_value'))}, "
        f"bootstrap 95% CI mean delta="
        f"[{_fmt_float(ci.get('ci_lower'))}, {_fmt_float(ci.get('ci_upper'))}]"
    )


def _print_dual_objective(
    dual_objective: dict[str, Any],
    condition_a: str,
    condition_b: str,
) -> None:
    if not dual_objective.get("available"):
        print(f"- unavailable ({dual_objective.get('error', 'no matched pairs')})")
        return

    effect = dual_objective.get("effect", {})
    if isinstance(effect, dict) and effect.get("available"):
        bootstrap_ci = effect.get("bootstrap_ci", {})
        print(
            f"- Success delta: {_fmt_signed_pp(effect.get('uplift'))} "
            f"({_fmt_pct(effect.get('a_rate'))} -> {_fmt_pct(effect.get('b_rate'))}), "
            f"95% CI[{_fmt_pct_signed(bootstrap_ci.get('ci_lower'))}, "
            f"{_fmt_pct_signed(bootstrap_ci.get('ci_upper'))}]"
        )

    print(
        f"- Token coverage for paired cost metrics: "
        f"{dual_objective.get('token_covered_pairs', 0)}/{dual_objective.get('matched_pairs', 0)} "
        f"({_fmt_pct(dual_objective.get('token_coverage_rate'))})"
    )

    cost_per_success = dual_objective.get("cost_per_success", {})
    if isinstance(cost_per_success, dict) and cost_per_success.get("available"):
        print(
            f"- Cost per success (tokens): "
            f"{condition_a}={_fmt_float(cost_per_success.get('a_value'))} "
            f"(successes={cost_per_success.get('a_successes', 0)}), "
            f"{condition_b}={_fmt_float(cost_per_success.get('b_value'))} "
            f"(successes={cost_per_success.get('b_successes', 0)})"
        )
    else:
        print(
            f"- Cost per success (tokens): unavailable "
            f"({cost_per_success.get('error', 'unknown') if isinstance(cost_per_success, dict) else 'unknown'})"
        )

    paired_cost_delta = dual_objective.get("paired_cost_delta", {})
    if isinstance(paired_cost_delta, dict) and paired_cost_delta.get("available"):
        ci = paired_cost_delta.get("bootstrap_ci_mean_delta", {})
        print(
            f"- Paired token delta (B - A): mean={_fmt_float(paired_cost_delta.get('mean_delta'))}, "
            f"mean A={_fmt_float(paired_cost_delta.get('mean_a'))}, "
            f"mean B={_fmt_float(paired_cost_delta.get('mean_b'))}, "
            f"pct vs A={_fmt_pct_signed(paired_cost_delta.get('pct_change_vs_a'))}, "
            f"95% CI[{_fmt_float(ci.get('ci_lower'))}, {_fmt_float(ci.get('ci_upper'))}]"
        )
    else:
        print(
            f"- Paired token delta (B - A): unavailable "
            f"({paired_cost_delta.get('error', 'unknown') if isinstance(paired_cost_delta, dict) else 'unknown'})"
        )

    fixed_budget_success = dual_objective.get("fixed_budget_success", [])
    if not isinstance(fixed_budget_success, list) or not fixed_budget_success:
        print("- Fixed-budget success: unavailable")
        return
    for entry in fixed_budget_success:
        if not isinstance(entry, dict):
            continue
        threshold = int(entry.get("token_threshold") or 0)
        print(
            f"- Fixed-budget success <= {threshold} tokens: "
            f"{condition_a}={_fmt_pct(entry.get('a_rate'))}, "
            f"{condition_b}={_fmt_pct(entry.get('b_rate'))}, "
            f"delta={_fmt_signed_pp(entry.get('delta'))}, "
            f"coverage={entry.get('coverage_pairs', 0)} pairs"
        )


def _print_input_errors(input_errors: dict[str, int]) -> None:
    files_unreadable = int(input_errors.get("files_unreadable", 0))
    json_lines_invalid = int(input_errors.get("json_lines_invalid", 0))
    records_skipped = int(input_errors.get("records_skipped", 0))
    if files_unreadable == 0 and json_lines_invalid == 0 and records_skipped == 0:
        return
    print(
        "- Input diagnostics: "
        f"files_unreadable={files_unreadable}, "
        f"json_lines_invalid={json_lines_invalid}, "
        f"records_skipped={records_skipped}"
    )


def _as_float(value: Any) -> float | None:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def _median(values: list[float]) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    n = len(ordered)
    mid = n // 2
    if n % 2 == 1:
        return ordered[mid]
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def _fmt_p(value: Any) -> str:
    return format_p_value(float(value)) if isinstance(value, (int, float)) else "n/a"


def _fmt_float(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    return f"{float(value):.4f}"


def _fmt_pct(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    return f"{100.0 * float(value):.2f}%"


def _fmt_pct_signed(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    return f"{100.0 * float(value):+.2f}%"


def _fmt_signed_pp(value: Any) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    return f"{100.0 * float(value):+.2f} pp"


if __name__ == "__main__":
    raise SystemExit(main())
