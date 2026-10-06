"""Two-pass workflow summary: baseline run followed by rescue on failures.

Loads baseline and rescue JSONL result files, identifies cases that failed
in the baseline, checks which were rescued, and prints a summary covering
baseline performance, rescue effectiveness, net outcome, and coverage
diagnostics.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evals.analyze_results import _derive_case_condition_rows, _load_and_group_records


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for two-pass workflow evaluation summary.

    Parses arguments, loads baseline and rescue JSONL result files,
    validates that each contains exactly one run ID, computes rescue
    statistics on baseline failures, and prints a summary to stdout.

    Args:
        argv: Command-line arguments.  Defaults to ``sys.argv[1:]``
            when ``None``.

    Returns:
        Exit code: 0 on success, 1 for missing or empty data,
        2 for validation errors.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Summarize a two-pass workflow: baseline run, then rescue run on baseline failures."
        )
    )
    parser.add_argument("baseline_run_id", help="Baseline run id.")
    parser.add_argument("rescue_run_id", help="Rescue run id.")
    parser.add_argument(
        "--results-dir",
        default="evals/results",
        help="Directory containing eval result JSONL files (default: evals/results).",
    )
    parser.add_argument(
        "--baseline-condition",
        default="base_repair",
        help="Condition to evaluate in baseline run (default: base_repair).",
    )
    parser.add_argument(
        "--rescue-condition",
        default="layered_adaptive_llm",
        help="Condition to evaluate in rescue run (default: layered_adaptive_llm).",
    )
    args = parser.parse_args(argv)

    baseline_path = Path(str(args.results_dir)) / f"{args.baseline_run_id}.jsonl"
    rescue_path = Path(str(args.results_dir)) / f"{args.rescue_run_id}.jsonl"
    for path in (baseline_path, rescue_path):
        if not path.is_file():
            print(f"Input error: results file not found: {path}")
            return 1

    try:
        baseline_rows = _load_rows(path=baseline_path)
        rescue_rows = _load_rows(path=rescue_path)
        _validate_single_run_id(
            rows=baseline_rows,
            expected_run_id=str(args.baseline_run_id),
            path=baseline_path,
            label="baseline",
        )
        _validate_single_run_id(
            rows=rescue_rows,
            expected_run_id=str(args.rescue_run_id),
            path=rescue_path,
            label="rescue",
        )
    except ValueError as exc:
        print(f"Input error: {exc}")
        return 2

    baseline_case_map = _case_success_map(
        rows=baseline_rows, condition=str(args.baseline_condition).strip()
    )
    if not baseline_case_map:
        print(
            "Input error: no baseline rows found for "
            f"condition={str(args.baseline_condition)!r} in {baseline_path}"
        )
        return 1

    rescue_case_map = _case_success_map(
        rows=rescue_rows, condition=str(args.rescue_condition).strip()
    )
    if not rescue_case_map:
        print(
            "Input error: no rescue rows found for "
            f"condition={str(args.rescue_condition)!r} in {rescue_path}"
        )
        return 1

    _print_two_pass_summary(
        baseline=PassSummary(
            run_id=str(args.baseline_run_id),
            condition=str(args.baseline_condition),
            path=baseline_path,
            case_success=baseline_case_map,
            case_costs=_case_costs(rows=baseline_rows, condition=str(args.baseline_condition)),
        ),
        rescue=PassSummary(
            run_id=str(args.rescue_run_id),
            condition=str(args.rescue_condition),
            path=rescue_path,
            case_success=rescue_case_map,
            case_costs=_case_costs(rows=rescue_rows, condition=str(args.rescue_condition)),
        ),
    )
    return 0


@dataclass(frozen=True)
class CaseCost:
    tokens: float | None
    runtime_sec: float | None


@dataclass(frozen=True)
class PassSummary:
    """One pass of the two-pass workflow: its run, condition, case outcomes and costs."""

    run_id: str
    condition: str
    path: Path
    case_success: dict[str, bool]
    case_costs: dict[str, CaseCost]


def _case_costs(*, rows: list[dict[str, Any]], condition: str) -> dict[str, CaseCost]:
    out: dict[str, CaseCost] = {}
    for row in rows:
        if str(row.get("condition") or "").strip() != condition.strip():
            continue
        case_id = str(row.get("case_id") or "")
        tokens = row.get("token_usage_total")
        runtime = row.get("runtime_sec")
        out[case_id] = CaseCost(
            tokens=float(tokens) if isinstance(tokens, (int, float)) else None,
            runtime_sec=float(runtime) if isinstance(runtime, (int, float)) else None,
        )
    return out


def _cost_line(costs: dict[str, CaseCost], case_ids: list[str]) -> str:
    """Total tokens and runtime over *case_ids*, with how many cases reported each."""
    tokens = [costs[c].tokens for c in case_ids if c in costs and costs[c].tokens is not None]
    runtimes = [
        costs[c].runtime_sec for c in case_ids if c in costs and costs[c].runtime_sec is not None
    ]
    return (
        f"tokens={sum(t for t in tokens if t is not None):.0f} "
        f"(reported by {len(tokens)}/{len(case_ids)} cases), "
        f"runtime={sum(r for r in runtimes if r is not None):.1f}s "
        f"(reported by {len(runtimes)}/{len(case_ids)} cases)"
    )


def _load_rows(*, path: Path) -> list[dict[str, Any]]:
    grouped, _input_errors = _load_and_group_records([str(path)], strict_input=True)
    return _derive_case_condition_rows(grouped)


def _validate_single_run_id(
    *,
    rows: list[dict[str, Any]],
    expected_run_id: str,
    path: Path,
    label: str,
) -> None:
    found_ids = sorted({str(row.get("run_id") or "") for row in rows if row.get("run_id")})
    if found_ids != [expected_run_id]:
        raise ValueError(
            f"{path}: {label} file must contain only run_id={expected_run_id!r}, found={found_ids}"
        )


def _case_success_map(*, rows: list[dict[str, Any]], condition: str) -> dict[str, bool]:
    out: dict[str, bool] = {}
    for row in rows:
        if str(row.get("condition") or "") != condition:
            continue
        case_id = str(row.get("case_id") or "")
        if not case_id:
            continue
        row_success = bool(row.get("success"))
        out[case_id] = bool(out.get(case_id, False) or row_success)
    return out


def _print_two_pass_summary(*, baseline: PassSummary, rescue: PassSummary) -> None:
    """Print the two-pass outcome.

    STA-22: the rescue pass runs only on the first pass's failures, so the final rate is
    the cumulative result of up to twice the compute and can never show a regression. It
    is labelled as such, next to each pass's own cost, not as an uplift vs the baseline.
    """
    baseline_case_map = baseline.case_success
    rescue_case_map = rescue.case_success
    baseline_total_cases = len(baseline_case_map)
    baseline_successes = sum(1 for success in baseline_case_map.values() if success)
    baseline_failed_ids = sorted(
        case_id for case_id, success in baseline_case_map.items() if not success
    )
    baseline_failed = len(baseline_failed_ids)
    baseline_success_rate = _ratio(baseline_successes, baseline_total_cases)

    failed_set = set(baseline_failed_ids)
    rescue_attempted_failed_ids = sorted(
        case_id for case_id in failed_set if case_id in rescue_case_map
    )
    rescue_attempted_failed = len(rescue_attempted_failed_ids)
    rescued_ids = sorted(
        case_id for case_id in rescue_attempted_failed_ids if bool(rescue_case_map.get(case_id))
    )
    rescued_successes = len(rescued_ids)
    remaining_failed = baseline_failed - rescued_successes
    rescue_missing_failed = baseline_failed - rescue_attempted_failed

    rescue_only_cases = sorted(
        case_id for case_id in rescue_case_map if case_id not in baseline_case_map
    )

    rescue_coverage_rate = _ratio(rescue_attempted_failed, baseline_failed)
    rescue_success_rate_on_attempted = _ratio(rescued_successes, rescue_attempted_failed)
    rescue_success_rate_on_failed_pool = _ratio(rescued_successes, baseline_failed)

    final_successes = baseline_successes + rescued_successes
    final_success_rate = _ratio(final_successes, baseline_total_cases)
    absolute_uplift = (
        final_success_rate - baseline_success_rate
        if baseline_success_rate is not None and final_success_rate is not None
        else None
    )

    print("Two-Pass Workflow Summary")
    print(
        f"- baseline: run_id={baseline.run_id} condition={baseline.condition} file={baseline.path}"
    )
    print(f"- rescue: run_id={rescue.run_id} condition={rescue.condition} file={rescue.path}")

    print("\nBaseline")
    print(
        f"- success: {baseline_successes}/{baseline_total_cases} "
        f"({_fmt_percent(baseline_success_rate)})"
    )
    print(f"- failed cases: {baseline_failed}")

    print("\nRescue On Baseline Failures")
    print(
        f"- attempted failed cases: {rescue_attempted_failed}/{baseline_failed} "
        f"({_fmt_percent(rescue_coverage_rate)})"
    )
    print(
        "- rescued successes: "
        f"{rescued_successes}/{rescue_attempted_failed} on attempted "
        f"({_fmt_percent(rescue_success_rate_on_attempted)}), "
        f"{rescued_successes}/{baseline_failed} on failed pool "
        f"({_fmt_percent(rescue_success_rate_on_failed_pool)})"
    )
    print(f"- remaining failed after rescue: {remaining_failed}")

    print("\nCumulative Two-Pass Outcome (not a treatment effect)")
    print(
        "- the rescue pass ran only on first-pass failures: this is the result of extra "
        "compute and cannot show a regression"
    )
    print(
        f"- final success after both passes: {final_successes}/{baseline_total_cases} "
        f"({_fmt_percent(final_success_rate)})"
    )
    print(f"- cumulative gain over the first pass: {_fmt_signed_percent(absolute_uplift)}")
    print(
        f"- first-pass cost (all cases): {_cost_line(baseline.case_costs, sorted(baseline_case_map))}"
    )
    print(
        "- rescue-pass cost (attempted failed cases): "
        f"{_cost_line(rescue.case_costs, rescue_attempted_failed_ids)}"
    )

    print("\nCoverage Diagnostics")
    print(f"- baseline failed not attempted in rescue: {rescue_missing_failed}")
    print(f"- rescue-only cases not in baseline: {len(rescue_only_cases)}")


def _ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def _fmt_percent(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * value:.1f}%"


def _fmt_signed_percent(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{100.0 * value:+.1f}%"


if __name__ == "__main__":
    raise SystemExit(main())
