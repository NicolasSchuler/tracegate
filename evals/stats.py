"""Statistical analysis for paired binary A/B evaluation results.

Provides McNemar's test (exact binomial), Cohen's h effect size, bootstrap
confidence intervals, Holm-Bonferroni correction, post-hoc power analysis, and
majority-vote aggregation for multi-run eval data.

Conventions shared by every eval report (2026-10-04 review, Phase 2.1):

- McNemar is always the exact binomial test (``scipy.stats.binomtest``); a result
  is significant when ``p <= alpha`` (STA-23).
- Every bootstrap is the percentile method over a seeded ``random.Random``;
  ``DEFAULT_BOOTSTRAP_SEED`` is the default seed (STA-05, STA-07).
- p-values are printed with :func:`format_p_value` (STA-15).
"""

from __future__ import annotations

import math
import random
from typing import Any

from scipy.stats import binomtest

DEFAULT_ALPHA = 0.05
DEFAULT_BOOTSTRAP_SEED = 42
BOOTSTRAP_METHOD = "percentile"

# ---------------------------------------------------------------------------
# Type aliases
# ---------------------------------------------------------------------------

BinaryPair = tuple[bool, bool]  # (condition_A_success, condition_B_success)
NumericPair = tuple[float, float]  # (condition_A_value, condition_B_value)


# ---------------------------------------------------------------------------
# McNemar's test
# ---------------------------------------------------------------------------


def mcnemar_test(pairs: list[BinaryPair], *, alpha: float = DEFAULT_ALPHA) -> dict[str, Any]:
    """Exact McNemar test for paired binary outcomes.

    Counts discordant pairs:
      n_01 = A fails, B succeeds (favors B)
      n_10 = A succeeds, B fails (favors A)

    The p-value is the exact two-sided binomial test of ``n_01`` out of the discordant
    pairs under p=0.5 at any sample size; the chi-square approximation above 25
    discordant pairs and the hardcoded alpha are gone (STA-23).

    Returns dict with: n_01, n_10, n_concordant, statistic, p_value, method, alpha,
    significant.
    """
    n_01 = 0  # A=0, B=1
    n_10 = 0  # A=1, B=0
    n_11 = 0
    n_00 = 0
    for a, b in pairs:
        if a and b:
            n_11 += 1
        elif not a and not b:
            n_00 += 1
        elif not a and b:
            n_01 += 1
        else:
            n_10 += 1

    n_discordant = n_01 + n_10

    if n_discordant == 0:
        return {
            "n_01": n_01,
            "n_10": n_10,
            "n_concordant": n_00 + n_11,
            "n_discordant": 0,
            "statistic": 0.0,
            "p_value": 1.0,
            "method": "no_discordant_pairs",
            "alpha": alpha,
            "significant": False,
        }

    p_value = exact_binomial_p_value(n_01, n_discordant)
    return {
        "n_01": n_01,
        "n_10": n_10,
        "n_concordant": n_00 + n_11,
        "n_discordant": n_discordant,
        "statistic": float(n_01),
        "p_value": p_value,
        "method": "exact_binomial",
        "alpha": alpha,
        "significant": p_value <= alpha,
    }


def exact_binomial_p_value(k: int, n: int) -> float:
    """Two-sided exact binomial p-value under p=0.5 (McNemar, paired sign test)."""
    return float(binomtest(k, n, 0.5, alternative="two-sided").pvalue)


def format_p_value(p_value: float | None) -> str:
    """Format a p-value without hiding the difference between 0.004 and 0.0004 (STA-15).

    Three significant digits, ``<0.0001`` below that, ``n/a`` when missing.
    """
    if p_value is None or not math.isfinite(p_value):
        return "n/a"
    if p_value < 0.0001:
        return "<0.0001"
    return f"{p_value:.3g}"


# ---------------------------------------------------------------------------
# Cohen's h effect size for proportions
# ---------------------------------------------------------------------------


def cohens_h(p1: float, p2: float) -> dict[str, Any]:
    """Cohen's h effect size: 2*(arcsin(sqrt(p2)) - arcsin(sqrt(p1))).

    Positive h means p2 > p1 (B is better than A).
    Thresholds: |h| >= 0.2 small, >= 0.5 medium, >= 0.8 large.
    """
    p1 = max(0.0, min(1.0, p1))
    p2 = max(0.0, min(1.0, p2))
    h = 2.0 * (math.asin(math.sqrt(p2)) - math.asin(math.sqrt(p1)))
    abs_h = abs(h)
    if abs_h >= 0.8:
        magnitude = "large"
    elif abs_h >= 0.5:
        magnitude = "medium"
    elif abs_h >= 0.2:
        magnitude = "small"
    else:
        magnitude = "negligible"
    return {"h": h, "magnitude": magnitude}


# ---------------------------------------------------------------------------
# Bootstrap confidence interval for uplift
# ---------------------------------------------------------------------------


def bootstrap_ci(
    pairs: list[BinaryPair],
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int | None = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Paired bootstrap CI for success-rate uplift (B - A).

    Resamples (case_id, A, B) tuples together to preserve pairing.
    Returns dict with: uplift, ci_lower, ci_upper, confidence, n_bootstrap.
    """
    if not pairs:
        return {
            "uplift": 0.0,
            "ci_lower": 0.0,
            "ci_upper": 0.0,
            "confidence": confidence,
            "n_bootstrap": 0,
            "method": BOOTSTRAP_METHOD,
            "seed": seed,
        }

    rng = random.Random(seed)
    n = len(pairs)

    # Observed uplift.
    a_rate = sum(1 for a, _ in pairs if a) / n
    b_rate = sum(1 for _, b in pairs if b) / n
    observed_uplift = b_rate - a_rate

    # Bootstrap.
    deltas: list[float] = []
    for _ in range(n_bootstrap):
        sample = [pairs[rng.randint(0, n - 1)] for _ in range(n)]
        sa = sum(1 for a, _ in sample if a) / n
        sb = sum(1 for _, b in sample if b) / n
        deltas.append(sb - sa)

    ci_lower, ci_upper = _percentile_interval(deltas, confidence)
    return {
        "uplift": observed_uplift,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "confidence": confidence,
        "n_bootstrap": n_bootstrap,
        "method": BOOTSTRAP_METHOD,
        "seed": seed,
    }


def bootstrap_mean_ci(
    values: list[float],
    *,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
    seed: int | None = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Bootstrap CI for the arithmetic mean of numeric values.

    Degenerate inputs return a point interval instead of raising.
    """
    mean_value = (sum(values) / len(values)) if values else None
    if not values or n_bootstrap <= 0 or len(values) < 2:
        return {
            "mean": mean_value,
            "ci_lower": mean_value,
            "ci_upper": mean_value,
            "confidence": confidence,
            "n_bootstrap": max(0, int(n_bootstrap)),
            "method": BOOTSTRAP_METHOD,
            "seed": seed,
        }

    rng = random.Random(seed)
    n = len(values)
    bootstrap_means: list[float] = []
    for _ in range(int(n_bootstrap)):
        sample = [values[rng.randint(0, n - 1)] for _ in range(n)]
        bootstrap_means.append(sum(sample) / float(n))

    ci_lower, ci_upper = _percentile_interval(bootstrap_means, confidence)
    return {
        "mean": mean_value,
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "confidence": confidence,
        "n_bootstrap": int(n_bootstrap),
        "method": BOOTSTRAP_METHOD,
        "seed": seed,
    }


def _percentile_interval(resampled: list[float], confidence: float) -> tuple[float, float]:
    """Percentile bootstrap interval; the only interval method in eval reports (STA-05)."""
    ordered = sorted(resampled)
    n_bootstrap = len(ordered)
    alpha = 1.0 - confidence
    lo_idx = max(0, math.floor(alpha / 2 * n_bootstrap))
    hi_idx = min(n_bootstrap - 1, math.floor((1 - alpha / 2) * n_bootstrap))
    return ordered[lo_idx], ordered[hi_idx]


def paired_numeric_deltas(pairs: list[NumericPair]) -> list[float]:
    """Return paired numeric deltas as ``B - A``."""
    return [b_value - a_value for a_value, b_value in pairs]


# ---------------------------------------------------------------------------
# Holm-Bonferroni correction
# ---------------------------------------------------------------------------


def holm_bonferroni(
    p_values: dict[str, float], alpha: float = DEFAULT_ALPHA
) -> dict[str, dict[str, Any]]:
    """Holm-Bonferroni step-down correction for multiple comparisons.

    The adjusted p-value is the running maximum of ``p * (m - rank + 1)``, capped at 1,
    and a hypothesis is rejected when its adjusted p is ``<= alpha``. That is the
    step-down procedure, including the boundary case ``p == alpha / (m - k + 1)``,
    which the old strict ``<`` didn't reject (STA-24).

    Returns dict mapping each key to {p_value, adjusted_p, rejected, rank, alpha}.
    """
    if not p_values:
        return {}

    m = len(p_values)
    sorted_items = sorted(p_values.items(), key=lambda kv: (kv[1], kv[0]))

    results: dict[str, dict[str, Any]] = {}
    running_max = 0.0
    for rank_0, (key, p) in enumerate(sorted_items):
        running_max = max(running_max, min(1.0, p * (m - rank_0)))
        results[key] = {
            "p_value": p,
            "adjusted_p": running_max,
            "rejected": running_max <= alpha,
            "rank": rank_0 + 1,
            "alpha": alpha,
        }
    return results


# ---------------------------------------------------------------------------
# Post-hoc power report
# ---------------------------------------------------------------------------


def power_report(n_cases: int, n_discordant: int, uplift: float) -> dict[str, Any]:
    """Rough post-hoc power assessment.

    Uses the normal approximation for McNemar's test power:
    power ~ Phi(sqrt(n_discordant) * |uplift| / sqrt(1 - uplift^2) - z_alpha/2)

    This is approximate — true power depends on the marginal proportions.
    """
    if n_discordant == 0 or abs(uplift) < 1e-10:
        return {
            "n_cases": n_cases,
            "n_discordant": n_discordant,
            "uplift": uplift,
            "approximate_power": 0.0,
            "sufficient": False,
            "recommendation": "Need more discordant pairs or larger effect size.",
        }

    z_alpha_half = 1.96  # alpha = 0.05, two-sided
    denom = math.sqrt(1 - uplift**2) if abs(uplift) < 1.0 else 0.01
    z_power = math.sqrt(n_discordant) * abs(uplift) / denom - z_alpha_half
    power = _normal_cdf(z_power)

    if power >= 0.8:
        recommendation = "Sufficient power (>=0.80) to detect this effect size."
        sufficient = True
    elif power >= 0.5:
        recommendation = f"Moderate power ({power:.2f}). Consider adding more cases."
        sufficient = False
    else:
        recommendation = f"Low power ({power:.2f}). Need substantially more discordant pairs."
        sufficient = False

    return {
        "n_cases": n_cases,
        "n_discordant": n_discordant,
        "uplift": uplift,
        "approximate_power": round(power, 4),
        "sufficient": sufficient,
        "recommendation": recommendation,
    }


# ---------------------------------------------------------------------------
# Majority-vote aggregation for multi-run (k>1)
# ---------------------------------------------------------------------------


def majority_vote_aggregation(
    runs: dict[str, list[BinaryPair]],
) -> list[BinaryPair]:
    """Aggregate k>1 runs per case_id into majority-vote binary pairs.

    ``runs`` maps case_id -> list of (A_success, B_success) across k runs.
    Returns one (A_majority, B_majority) pair per case_id.
    """
    result: list[BinaryPair] = []
    for case_id in sorted(runs):
        case_runs = runs[case_id]
        if not case_runs:
            continue
        a_votes = sum(1 for a, _ in case_runs if a)
        b_votes = sum(1 for _, b in case_runs if b)
        threshold = len(case_runs) / 2
        result.append((a_votes > threshold, b_votes > threshold))
    return result


def any_success_aggregation(
    runs: dict[str, list[BinaryPair]],
) -> list[BinaryPair]:
    """Aggregate k>1 runs: succeed if *any* run succeeded."""
    result: list[BinaryPair] = []
    for case_id in sorted(runs):
        case_runs = runs[case_id]
        if not case_runs:
            continue
        a_any = any(a for a, _ in case_runs)
        b_any = any(b for _, b in case_runs)
        result.append((a_any, b_any))
    return result


# ---------------------------------------------------------------------------
# Run full statistical analysis on paired rows
# ---------------------------------------------------------------------------


def analyze_paired_results(
    rows: list[dict[str, Any]],
    *,
    n_bootstrap: int = 10000,
    seed: int | None = DEFAULT_BOOTSTRAP_SEED,
    condition_a: str = "base_repair",
    condition_b: str = "layered_adaptive_llm",
) -> dict[str, Any]:
    """Run full statistical analysis on derived case-condition rows.

    Expects rows with keys: case_id, condition, category, success, run_id.
    Returns a complete statistics summary.
    """
    pairs, categories = _extract_pairs(
        rows,
        condition_a=condition_a,
        condition_b=condition_b,
    )

    if not pairs:
        return {"error": "No matched A/B pairs found."}

    a_rate = sum(1 for a, _ in pairs if a) / len(pairs)
    b_rate = sum(1 for _, b in pairs if b) / len(pairs)

    overall_mcnemar = mcnemar_test(pairs)
    result: dict[str, Any] = {
        "n_pairs": len(pairs),
        "condition_a": condition_a,
        "condition_b": condition_b,
        "a_success_rate": a_rate,
        "b_success_rate": b_rate,
        "overall_mcnemar": overall_mcnemar,
        "overall_cohens_h": cohens_h(a_rate, b_rate),
        "overall_bootstrap_ci": bootstrap_ci(pairs, n_bootstrap=n_bootstrap, seed=seed),
        "overall_power": power_report(len(pairs), overall_mcnemar["n_discordant"], b_rate - a_rate),
    }

    # Per-category analysis with Holm-Bonferroni.
    if categories:
        cat_results: dict[str, Any] = {}
        cat_p_values: dict[str, float] = {}
        for cat, cat_pairs in sorted(categories.items()):
            if not cat_pairs:
                continue
            cat_a_rate = sum(1 for a, _ in cat_pairs if a) / len(cat_pairs)
            cat_b_rate = sum(1 for _, b in cat_pairs if b) / len(cat_pairs)
            cat_mcnemar = mcnemar_test(cat_pairs)
            cat_results[cat] = {
                "n_pairs": len(cat_pairs),
                "a_success_rate": cat_a_rate,
                "b_success_rate": cat_b_rate,
                "mcnemar": cat_mcnemar,
                "cohens_h": cohens_h(cat_a_rate, cat_b_rate),
                "bootstrap_ci": bootstrap_ci(cat_pairs, n_bootstrap=n_bootstrap, seed=seed),
            }
            cat_p_values[cat] = cat_mcnemar["p_value"]

        result["per_category"] = cat_results
        if cat_p_values:
            result["holm_bonferroni"] = holm_bonferroni(cat_p_values)

    return result


def analyze_condition_matrix(
    rows: list[dict[str, Any]],
    *,
    n_bootstrap: int = 10000,
    seed: int | None = None,
) -> list[dict[str, Any]]:
    """Run pairwise A/B analysis across all discovered conditions."""
    conditions = sorted({str(row.get("condition", "")) for row in rows if row.get("condition")})
    out: list[dict[str, Any]] = []
    for idx, condition_a in enumerate(conditions):
        for condition_b in conditions[idx + 1 :]:
            result = analyze_paired_results(
                rows,
                n_bootstrap=n_bootstrap,
                seed=seed,
                condition_a=condition_a,
                condition_b=condition_b,
            )
            out.append(result)
    return out


def _extract_pairs(
    rows: list[dict[str, Any]],
    *,
    condition_a: str,
    condition_b: str,
) -> tuple[list[BinaryPair], dict[str, list[BinaryPair]]]:
    """Extract matched (A, B) pairs from rows, grouped by (run_id, case_id).

    Returns (overall_pairs, category_pairs_dict).
    """
    by_key: dict[tuple[str, str], dict[str, Any]] = {}
    for row in rows:
        if row.get("baseline_supported") is False:
            continue  # BAS-01 / D3: not a comparable outcome
        run_id = str(row.get("run_id", ""))
        case_id = str(row.get("case_id", ""))
        condition = str(row.get("condition", ""))
        key = (run_id, case_id)
        if key not in by_key:
            by_key[key] = {}
        by_key[key][condition] = row

    pairs: list[BinaryPair] = []
    categories: dict[str, list[BinaryPair]] = {}
    for key in sorted(by_key):
        entries = by_key[key]
        a_row = entries.get(condition_a)
        b_row = entries.get(condition_b)
        if a_row is None or b_row is None:
            continue
        pair = (bool(a_row.get("success")), bool(b_row.get("success")))
        pairs.append(pair)
        cat = str(a_row.get("category", "unknown"))
        categories.setdefault(cat, []).append(pair)

    return pairs, categories


# ---------------------------------------------------------------------------
# Internal math helpers
# ---------------------------------------------------------------------------


def _normal_cdf(z: float) -> float:
    """Standard normal CDF using math.erfc."""
    return 0.5 * math.erfc(-z / math.sqrt(2))
