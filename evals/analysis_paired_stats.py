"""Paired significance tests between conditions, with Holm correction across pairs."""

from __future__ import annotations

from typing import Any

from evals.analysis_numeric import _fmt_num, _fmt_signed_percent
from evals.stats import DEFAULT_BOOTSTRAP_SEED, format_p_value


def _run_statistical_analysis(
    rows: list[dict[str, Any]],
    *,
    aggregation: str = "per_run",
    pair_a: str = "base_repair",
    pair_b: str = "layered_adaptive_llm",
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Run statistical analysis on paired rows and emit pairwise matrix entries.

    Every bootstrap uses *seed*, so identical inputs give identical intervals (STA-07).
    """
    primary = _run_statistical_analysis_pair(
        rows,
        aggregation=aggregation,
        pair_a=pair_a,
        pair_b=pair_b,
        seed=seed,
    )
    conditions = sorted({str(row.get("condition") or "") for row in rows if row.get("condition")})
    pairwise: list[dict[str, Any]] = []
    for idx, condition_a in enumerate(conditions):
        for condition_b in conditions[idx + 1 :]:
            result = _run_statistical_analysis_pair(
                rows,
                aggregation=aggregation,
                pair_a=condition_a,
                pair_b=condition_b,
                seed=seed,
            )
            entry: dict[str, Any] = {
                "condition_a": condition_a,
                "condition_b": condition_b,
                "aggregation": result.get("aggregation", aggregation),
            }
            if "error" in result:
                entry["error"] = result["error"]
            else:
                a_rate = result.get("a_success_rate")
                b_rate = result.get("b_success_rate")
                mcn = result.get("overall_mcnemar", {})
                entry.update(
                    {
                        "n_pairs": int(result.get("n_pairs", 0)),
                        "a_success_rate": a_rate,
                        "b_success_rate": b_rate,
                        "uplift_success_rate": (
                            float(b_rate - a_rate)
                            if isinstance(a_rate, (int, float)) and isinstance(b_rate, (int, float))
                            else None
                        ),
                        "mcnemar_p_value": (
                            float(mcn.get("p_value"))
                            if isinstance(mcn, dict)
                            and isinstance(mcn.get("p_value"), (int, float))
                            else None
                        ),
                        "mcnemar_significant": (
                            bool(mcn.get("significant")) if isinstance(mcn, dict) else None
                        ),
                    }
                )
            pairwise.append(entry)
    _apply_pairwise_holm(pairwise)
    primary["pairwise"] = pairwise
    primary["pairwise_correction"] = PAIRWISE_CORRECTION_FAMILY
    return primary


PAIRWISE_CORRECTION_FAMILY = "holm_across_all_condition_pairs"


def _apply_pairwise_holm(pairwise: list[dict[str, Any]]) -> None:
    """Holm-adjust the McNemar p-values across every condition pair (STA-06).

    Nine conditions give 36 tests; at an uncorrected 0.05 about 1.8 are significant by
    chance. ``mcnemar_significant`` keeps the raw test; ``holm_rejected`` is the
    family-wise decision.
    """
    from evals.stats import holm_bonferroni

    p_values = {
        f"{entry['condition_a']}|{entry['condition_b']}": float(entry["mcnemar_p_value"])
        for entry in pairwise
        if isinstance(entry.get("mcnemar_p_value"), (int, float))
    }
    adjusted = holm_bonferroni(p_values)
    for entry in pairwise:
        holm = adjusted.get(f"{entry['condition_a']}|{entry['condition_b']}")
        entry["mcnemar_p_adjusted"] = holm["adjusted_p"] if holm else None
        entry["holm_rejected"] = bool(holm["rejected"]) if holm else None


def _run_statistical_analysis_pair(
    rows: list[dict[str, Any]],
    *,
    aggregation: str,
    pair_a: str,
    pair_b: str,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Run statistical analysis for one selected condition pair, case as the unit.

    ``per_run`` refuses inputs with several runs of a case instead of pooling them as
    independent pairs (STA-02); ``majority_vote`` and ``any_success`` aggregate a
    case's runs first.
    """
    from evals.stats import analyze_paired_results

    result = analyze_paired_results(
        rows,
        condition_a=pair_a,
        condition_b=pair_b,
        seed=seed,
        run_aggregation=aggregation,
    )
    result["aggregation"] = aggregation
    return result


def _print_stats(stats: dict[str, Any]) -> None:
    """Print statistical analysis results."""
    if "error" in stats:
        print(f"\nStatistics: {stats['error']}")
        return

    print(f"\nStatistical analysis (aggregation={stats.get('aggregation', 'per_run')}):")
    print(
        f"  Pairing: A={stats.get('condition_a', 'base_repair')} "
        f"vs B={stats.get('condition_b', 'layered_adaptive_llm')}"
    )
    print(f"  Matched pairs: {stats.get('n_pairs', 0)}")
    print(f"  A success rate: {stats.get('a_success_rate', 0):.1%}")
    print(f"  B success rate: {stats.get('b_success_rate', 0):.1%}")

    mcn = stats.get("overall_mcnemar", {})
    print(
        f"  McNemar's test: n_01={mcn.get('n_01', 0)}, n_10={mcn.get('n_10', 0)}, "
        f"p={format_p_value(mcn.get('p_value'))} ({mcn.get('method', '?')})"
        f"{' *' if mcn.get('significant') else ''}"
    )

    ch = stats.get("overall_cohens_h", {})
    print(f"  Cohen's h: {ch.get('h', 0):.3f} ({ch.get('magnitude', '?')})")

    bci = stats.get("overall_bootstrap_ci", {})
    print(
        f"  Bootstrap CI ({bci.get('confidence', 0.95):.0%}): "
        f"[{bci.get('ci_lower', 0):.3f}, {bci.get('ci_upper', 0):.3f}]"
    )

    pwr = stats.get("overall_power", {})
    print(f"  Power: {pwr.get('approximate_power', 0):.2f} — {pwr.get('recommendation', '')}")

    per_cat = stats.get("per_category", {})
    if per_cat:
        hb = stats.get("holm_bonferroni", {})
        print("  Per-category:")
        for cat, cat_stats in sorted(per_cat.items()):
            cat_mcn = cat_stats.get("mcnemar", {})
            cat_ch = cat_stats.get("cohens_h", {})
            hb_entry = hb.get(cat, {})
            sig_marker = " *" if hb_entry.get("rejected") else ""
            adj_p = hb_entry.get("adjusted_p")
            adj_str = f", adj_p={format_p_value(adj_p)}" if adj_p is not None else ""
            print(
                f"    {cat}: n={cat_stats.get('n_pairs', 0)}, "
                f"p={format_p_value(cat_mcn.get('p_value'))}{adj_str}, "
                f"h={cat_ch.get('h', 0):.3f}{sig_marker}"
            )

    pairwise = stats.get("pairwise")
    if isinstance(pairwise, list) and pairwise:
        print("  Pairwise matrix (Holm across all condition pairs):")
        for entry in pairwise:
            if not isinstance(entry, dict):
                continue
            condition_a = str(entry.get("condition_a") or "")
            condition_b = str(entry.get("condition_b") or "")
            if not condition_a or not condition_b:
                continue
            if "error" in entry:
                print(f"    {condition_a} -> {condition_b}: error={entry['error']}")
                continue
            print(
                f"    {condition_a} -> {condition_b}: "
                f"uplift={_fmt_signed_percent(entry.get('uplift_success_rate'))}, "
                f"n={_fmt_num(entry.get('n_pairs'))}, "
                f"p={format_p_value(entry.get('mcnemar_p_value'))}, "
                f"Holm adj p={format_p_value(entry.get('mcnemar_p_adjusted'))}, "
                f"sig (Holm)={'yes' if entry.get('holm_rejected') else 'no'}"
            )
