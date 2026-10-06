"""Cross-baseline comparison tool for ISSRE paper experiments.

Loads JSONL results from multiple eval runs (llmdebug conditions +
external baselines) and generates head-to-head statistical comparisons
using the existing ``evals.stats`` primitives.

Usage::

    python -m baselines.compare_results \\
        --jsonl 'evals/results/*.jsonl' \\
        --conditions base_repair ldb intervenor tracecoder \\
        --reference base_repair \\
        --primary layered_adaptive_gated \\
        --bootstrap 10000 --seed 42 \\
        --out comparison.json

The ``--reference`` condition is the baseline (A-side) for all pairwise
comparisons.  The ``--primary`` condition is the main llmdebug treatment.
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path
from typing import Any

from evals.analyze_results import _derive_case_condition_rows, _load_and_group_records
from evals.stats import analyze_paired_results, format_p_value, holm_bonferroni

# ------------------------------------------------------------------
# Data loading
# ------------------------------------------------------------------


def _expand_paths(patterns: list[str]) -> list[str]:
    """Expand shell globs into a list of file paths."""
    result: list[str] = []
    for pattern in patterns:
        expanded = sorted(glob.glob(pattern, recursive=True))
        result.extend(expanded if expanded else [pattern])
    return result


def _load_results(
    jsonl_paths: list[str],
    *,
    strict: bool = False,
) -> list[dict[str, Any]]:
    """Load and derive case-condition rows from JSONL result files.

    Returns a list of dicts, one per (run_id, case_id, condition) triple,
    with derived metrics (success, attempts, tokens, etc.).
    """
    grouped, errors = _load_and_group_records(jsonl_paths, strict_input=strict)
    unreadable = int(errors.get("files_unreadable") or 0)
    if unreadable:
        # BAS-09: the counter is an int; len() on it raised TypeError instead of warning.
        print(f"Warning: {unreadable} unreadable file(s)", file=sys.stderr)
    rows = _derive_case_condition_rows(grouped)
    return rows


# ------------------------------------------------------------------
# Comparison logic
# ------------------------------------------------------------------


def _compute_pairwise(
    rows: list[dict[str, Any]],
    condition_a: str,
    condition_b: str,
    *,
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Compute a full pairwise comparison between two conditions, case as the unit.

    BAS-06: success used to be keyed by case only, so with repeated runs the last run
    in file-glob order overwrote the others and the paired and per-category N
    disagreed. This uses the shared ``analyze_paired_results``: one pair per case,
    repeated runs aggregated by majority vote, the run count reported.
    """
    result = analyze_paired_results(
        rows,
        n_bootstrap=n_bootstrap,
        seed=seed,
        condition_a=condition_a,
        condition_b=condition_b,
    )
    if "error" in result:
        return {
            "condition_a": condition_a,
            "condition_b": condition_b,
            "n_cases": 0,
            "error": "no_common_cases",
        }
    mcnemar = result["overall_mcnemar"]
    a_rate = float(result["a_success_rate"])
    b_rate = float(result["b_success_rate"])
    return {
        "condition_a": condition_a,
        "condition_b": condition_b,
        "n_cases": result["n_pairs"],
        "runs_per_case_max": result["runs_per_case_max"],
        "tied_cases": result["tied_cases"],
        "a_rate": round(a_rate, 4),
        "b_rate": round(b_rate, 4),
        "uplift_pp": round(b_rate - a_rate, 4),
        "relative_change": round((b_rate - a_rate) / a_rate, 4) if a_rate > 0 else None,
        "a_only_wins": mcnemar["n_10"],
        "b_only_wins": mcnemar["n_01"],
        "mcnemar": mcnemar,
        "cohens_h": result["overall_cohens_h"],
        "bootstrap_ci": result["overall_bootstrap_ci"],
        "per_category": {
            category: {
                "a_cases": stats["n_pairs"],
                "a_rate": round(float(stats["a_success_rate"]), 4),
                "b_cases": stats["n_pairs"],
                "b_rate": round(float(stats["b_success_rate"]), 4),
            }
            for category, stats in result["per_category"].items()
        },
    }


def compare_all(
    rows: list[dict[str, Any]],
    conditions: list[str],
    reference: str,
    *,
    n_bootstrap: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Run all pairwise comparisons against the reference condition.

    Returns a dict with:
    - ``conditions``: list of condition names found
    - ``reference``: reference condition name
    - ``pairwise``: list of pairwise comparison dicts
    - ``per_category``: per-category breakdowns for each pair
    - ``holm_bonferroni``: multiple comparison correction across all pairs
    """
    # Discover available conditions.
    available = sorted({row.get("condition", "") for row in rows if row.get("condition")})
    targets = [c for c in conditions if c in available and c != reference]

    if reference not in available:
        return {
            "error": f"Reference condition {reference!r} not in results. Available: {available}",
        }

    pairwise: list[dict[str, Any]] = []
    per_category: dict[str, dict[str, dict[str, Any]]] = {}
    p_values: dict[str, float] = {}
    baseline_contracts = _collect_baseline_contracts(rows)

    for target in targets:
        comparison = _compute_pairwise(rows, reference, target, n_bootstrap=n_bootstrap, seed=seed)
        per_category[f"{reference}_vs_{target}"] = comparison.pop("per_category", {})
        pairwise.append(comparison)

        mcnemar_result = comparison.get("mcnemar", {})
        p = mcnemar_result.get("p_value")
        if p is not None:
            p_values[f"{reference}_vs_{target}"] = p

    # Apply Holm-Bonferroni correction.
    corrected = holm_bonferroni(p_values) if p_values else {}

    return {
        "conditions_available": available,
        "conditions_compared": [reference, *targets],
        "reference": reference,
        "n_total_rows": len(rows),
        "pairwise": pairwise,
        "per_category": per_category,
        "holm_bonferroni": corrected,
        "baseline_contracts": baseline_contracts,
    }


def _collect_baseline_contracts(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    contracts: dict[str, dict[str, Any]] = {}
    for row in rows:
        condition = str(row.get("condition") or "")
        fidelity = row.get("baseline_fidelity_level")
        repair_scope = row.get("baseline_repair_scope_policy")
        prompt_contract = row.get("baseline_prompt_contract_kind")
        output_contract = row.get("baseline_output_contract")
        upstream_commit = row.get("baseline_upstream_commit")
        contract_version = row.get("baseline_contract_version")
        if not any(
            isinstance(value, str) and value.strip()
            for value in (
                fidelity,
                repair_scope,
                prompt_contract,
                output_contract,
                upstream_commit,
                contract_version,
            )
        ):
            continue
        contracts.setdefault(condition, {})
        if isinstance(fidelity, str) and fidelity.strip():
            contracts[condition]["fidelity_level"] = fidelity.strip()
        if isinstance(repair_scope, str) and repair_scope.strip():
            contracts[condition]["repair_scope_policy"] = repair_scope.strip()
        if isinstance(prompt_contract, str) and prompt_contract.strip():
            contracts[condition]["prompt_contract_kind"] = prompt_contract.strip()
        if isinstance(output_contract, str) and output_contract.strip():
            contracts[condition]["output_contract"] = output_contract.strip()
        if isinstance(upstream_commit, str) and upstream_commit.strip():
            contracts[condition]["upstream_commit"] = upstream_commit.strip()
        if isinstance(contract_version, str) and contract_version.strip():
            contracts[condition]["contract_version"] = contract_version.strip()
    return contracts


# ------------------------------------------------------------------
# Text formatting
# ------------------------------------------------------------------


def _format_report(result: dict[str, Any]) -> str:
    """Format comparison results as human-readable text."""
    lines: list[str] = []
    lines.append("=" * 72)
    lines.append("CROSS-BASELINE COMPARISON REPORT")
    lines.append("=" * 72)

    ref = result.get("reference", "?")
    avail = result.get("conditions_available", [])
    compared = result.get("conditions_compared", [])
    lines.append(f"\nReference condition: {ref}")
    lines.append(f"Conditions available: {', '.join(avail)}")
    lines.append(f"Conditions compared:  {', '.join(compared)}")
    lines.append(f"Total result rows:    {result.get('n_total_rows', 0)}")
    baseline_contracts = result.get("baseline_contracts", {})
    if isinstance(baseline_contracts, dict) and baseline_contracts:
        lines.append("\nBaseline Contracts:")
        for condition, info in sorted(baseline_contracts.items()):
            if not isinstance(info, dict):
                continue
            fidelity = str(info.get("fidelity_level") or "n/a")
            repair_scope = str(info.get("repair_scope_policy") or "n/a")
            prompt_contract = str(info.get("prompt_contract_kind") or "n/a")
            lines.append(
                f"- {condition}: fidelity={fidelity}, repair_scope={repair_scope}, prompt_contract={prompt_contract}"
            )

    # Pairwise comparisons.
    pairwise = result.get("pairwise", [])
    for pw in pairwise:
        lines.append("")
        lines.append("-" * 72)
        ca = pw.get("condition_a", "?")
        cb = pw.get("condition_b", "?")
        lines.append(f"  {ca} vs. {cb}")
        lines.append("-" * 72)

        n = pw.get("n_cases", 0)
        ar = pw.get("a_rate", 0)
        br = pw.get("b_rate", 0)
        uplift = pw.get("uplift_pp", 0)
        lines.append(f"  Paired cases:  {n}")
        lines.append(f"  {ca} rate: {ar:.1%}   |   {cb} rate: {br:.1%}")
        lines.append(f"  Uplift (B-A):  {uplift:+.1%}")

        mcn = pw.get("mcnemar", {})
        if mcn:
            p = mcn.get("p_value", 1.0)
            sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "ns"
            lines.append(
                f"  McNemar p={format_p_value(p)} ({sig})  "
                f"[A-only={pw.get('a_only_wins', 0)}, B-only={pw.get('b_only_wins', 0)}]"
            )

        eff = pw.get("cohens_h", {})
        if eff:
            lines.append(f"  Cohen's h={eff.get('h', 0):.3f} ({eff.get('magnitude', '?')})")

        bci = pw.get("bootstrap_ci", {})
        if bci:
            lo = bci.get("ci_lower", 0)
            hi = bci.get("ci_upper", 0)
            lines.append(f"  Bootstrap 95% CI for uplift: [{lo:.1%}, {hi:.1%}]")

    # Per-category breakdown for each pair.
    per_cat = result.get("per_category", {})
    if per_cat:
        lines.append("")
        lines.append("=" * 72)
        lines.append("PER-CATEGORY BREAKDOWN")
        lines.append("=" * 72)

        for pair_key, cats in per_cat.items():
            lines.append(f"\n  {pair_key}:")
            lines.append(f"  {'Category':<30s} {'A rate':>8s} {'B rate':>8s} {'Δ':>8s}")
            lines.append(f"  {'-' * 30} {'-' * 8} {'-' * 8} {'-' * 8}")
            for cat, stats_dict in sorted(cats.items()):
                ar = stats_dict.get("a_rate", 0)
                br = stats_dict.get("b_rate", 0)
                delta = br - ar
                lines.append(f"  {cat:<30s} {ar:>7.1%} {br:>7.1%} {delta:>+7.1%}")

    # Holm-Bonferroni correction.
    hb = result.get("holm_bonferroni", {})
    if hb:
        lines.append("")
        lines.append("-" * 72)
        lines.append("Holm-Bonferroni correction (all pairwise tests):")
        for key, info in sorted(hb.items()):
            adj_p = info.get("adjusted_p", 1.0)
            rejected = info.get("rejected", False)
            sig_str = "REJECT" if rejected else "fail to reject"
            lines.append(f"  {key}: adj_p={format_p_value(adj_p)} → {sig_str}")

    lines.append("")
    return "\n".join(lines)


def _format_latex_table(result: dict[str, Any]) -> str:
    """Generate a LaTeX table for the pairwise comparisons."""
    lines: list[str] = []
    lines.append("% Auto-generated by baselines/compare_results.py")
    lines.append(r"\begin{table}[ht]")
    lines.append(r"\centering")
    lines.append(r"\caption{Cross-baseline comparison of repair success rates.}")
    lines.append(r"\label{tab:baseline-comparison}")
    lines.append(r"\begin{tabular}{lrrrrl}")
    lines.append(r"\toprule")
    lines.append(r"Condition & Cases & Rate & Uplift & $p$ & Sig. \\")
    lines.append(r"\midrule")

    ref = result.get("reference", "?")
    pairwise = result.get("pairwise", [])

    for pw in pairwise:
        cb = pw.get("condition_b", "?")
        n = pw.get("n_cases", 0)
        br = pw.get("b_rate", 0)
        uplift = pw.get("uplift_pp", 0)
        p = pw.get("mcnemar", {}).get("p_value", 1.0)
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else ""

        label = cb.replace("_", r"\_")
        p_text = format_p_value(p).replace("<", "$<$")
        lines.append(f"{label} & {n} & {br:.1%} & {uplift:+.1%} & {p_text} & {sig} \\\\")

    lines.append(r"\bottomrule")
    lines.append(r"\end{tabular}")
    lines.append(
        r"\par\smallskip\footnotesize Reference: "
        + ref.replace("_", r"\_")
        + r". Significance: McNemar's test ($*p<.05$, $**p<.01$, $***p<.001$)."
    )
    lines.append(r"\end{table}")
    return "\n".join(lines)


# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    """CLI entry point for cross-baseline comparison."""
    parser = argparse.ArgumentParser(
        description="Cross-baseline comparison for ISSRE paper experiments.",
    )
    parser.add_argument(
        "--jsonl",
        required=True,
        nargs="+",
        help="Path(s) or glob(s) to eval JSONL result files.",
    )
    parser.add_argument(
        "--conditions",
        nargs="+",
        default=[],
        help="Conditions to compare (if empty, all discovered conditions are used).",
    )
    parser.add_argument(
        "--reference",
        default="base_repair",
        help="Reference (A-side) condition for all pairwise comparisons.",
    )
    parser.add_argument(
        "--primary",
        default="layered_adaptive_gated",
        help="Primary llmdebug condition (highlighted in output).",
    )
    parser.add_argument(
        "--bootstrap",
        type=int,
        default=10_000,
        help="Number of bootstrap resamples.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for bootstrap.",
    )
    parser.add_argument(
        "--out",
        default="",
        help="Output path for JSON results (default: stdout only).",
    )
    parser.add_argument(
        "--latex",
        default="",
        help="Output path for LaTeX table.",
    )
    parser.add_argument(
        "--strict-input",
        action="store_true",
        help="Fail on first unreadable input file.",
    )
    args = parser.parse_args(argv)

    # Expand globs.
    paths = _expand_paths(args.jsonl)
    if not paths:
        print(f"No files matched: {args.jsonl}", file=sys.stderr)
        return 1

    print(f"Loading {len(paths)} result file(s)...", file=sys.stderr)
    rows = _load_results(paths, strict=args.strict_input)

    if not rows:
        print("No result rows loaded.", file=sys.stderr)
        return 1

    conditions = args.conditions or sorted(
        {r.get("condition", "") for r in rows if r.get("condition")}
    )

    result = compare_all(
        rows,
        conditions,
        args.reference,
        n_bootstrap=args.bootstrap,
        seed=args.seed,
    )

    if "error" in result:
        print(f"Error: {result['error']}", file=sys.stderr)
        return 1

    # Print human-readable report.
    print(_format_report(result))

    # Optionally save JSON.
    if args.out:
        out_path = Path(args.out)
        out_path.write_text(json.dumps(result, indent=2, default=str), encoding="utf-8")
        print(f"JSON saved to {out_path}", file=sys.stderr)

    # Optionally save LaTeX.
    if args.latex:
        latex_path = Path(args.latex)
        latex_path.write_text(_format_latex_table(result), encoding="utf-8")
        print(f"LaTeX table saved to {latex_path}", file=sys.stderr)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
