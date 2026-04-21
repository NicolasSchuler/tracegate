"""Tests for evals.stats — statistical analysis module."""

from __future__ import annotations

import math

from evals.stats import (
    BinaryPair,
    NumericPair,
    analyze_condition_matrix,
    analyze_paired_results,
    any_success_aggregation,
    bootstrap_ci,
    bootstrap_mean_ci,
    cohens_h,
    holm_bonferroni,
    majority_vote_aggregation,
    mcnemar_test,
    paired_numeric_deltas,
    power_report,
)

# ---------------------------------------------------------------------------
# McNemar's test
# ---------------------------------------------------------------------------


class TestMcNemarTest:
    def test_classic_example(self) -> None:
        """n_01=10, n_10=3 -> p ~0.052 (exact binomial, not significant at 0.05)."""
        pairs: list[BinaryPair] = (
            [(False, True)] * 10  # A fails, B succeeds
            + [(True, False)] * 3  # A succeeds, B fails
            + [(True, True)] * 5  # concordant
            + [(False, False)] * 2  # concordant
        )
        result = mcnemar_test(pairs)
        assert result["n_01"] == 10
        assert result["n_10"] == 3
        assert result["n_discordant"] == 13
        assert result["n_concordant"] == 7
        assert result["method"] == "exact_binomial"
        # Exact binomial p-value for 10 out of 13 with p=0.5
        assert 0.04 < result["p_value"] < 0.10
        # At alpha=0.05, this is borderline
        assert isinstance(result["significant"], bool)

    def test_large_sample_chi_squared(self) -> None:
        """>=25 discordant pairs uses chi-squared with continuity correction."""
        pairs: list[BinaryPair] = [(False, True)] * 20 + [(True, False)] * 5
        result = mcnemar_test(pairs)
        assert result["n_discordant"] == 25
        assert result["method"] == "chi_squared_cc"
        # With 20 vs 5, should be significant
        assert result["p_value"] < 0.01
        assert result["significant"] is True

    def test_all_concordant(self) -> None:
        """All concordant pairs -> p=1.0, no discordant."""
        pairs: list[BinaryPair] = [(True, True)] * 10 + [(False, False)] * 5
        result = mcnemar_test(pairs)
        assert result["n_discordant"] == 0
        assert result["p_value"] == 1.0
        assert result["method"] == "no_discordant_pairs"
        assert result["significant"] is False

    def test_single_discordant_pair(self) -> None:
        """Single discordant pair -> exact binomial on 1 trial."""
        pairs: list[BinaryPair] = [(False, True)]
        result = mcnemar_test(pairs)
        assert result["n_01"] == 1
        assert result["n_10"] == 0
        assert result["n_discordant"] == 1
        assert result["p_value"] == 1.0  # 1 out of 1, two-sided

    def test_empty_pairs(self) -> None:
        """Empty input -> no discordant pairs."""
        result = mcnemar_test([])
        assert result["n_discordant"] == 0
        assert result["p_value"] == 1.0

    def test_symmetric_discordant(self) -> None:
        """Equal discordant both ways -> not significant."""
        pairs: list[BinaryPair] = [(False, True)] * 10 + [(True, False)] * 10
        result = mcnemar_test(pairs)
        assert result["n_01"] == 10
        assert result["n_10"] == 10
        assert result["p_value"] > 0.5


# ---------------------------------------------------------------------------
# Cohen's h
# ---------------------------------------------------------------------------


class TestCohensH:
    def test_standard_example(self) -> None:
        """(0.5, 0.7) -> h ~0.407 (small-medium)."""
        result = cohens_h(0.5, 0.7)
        assert 0.38 < result["h"] < 0.44
        assert result["magnitude"] == "small"

    def test_zero_to_one(self) -> None:
        """(0.0, 1.0) -> h = pi (large)."""
        result = cohens_h(0.0, 1.0)
        assert abs(result["h"] - math.pi) < 0.01
        assert result["magnitude"] == "large"

    def test_equal_proportions(self) -> None:
        """Same proportions -> h = 0 (negligible)."""
        result = cohens_h(0.5, 0.5)
        assert abs(result["h"]) < 0.001
        assert result["magnitude"] == "negligible"

    def test_negative_direction(self) -> None:
        """p1 > p2 -> negative h."""
        result = cohens_h(0.8, 0.3)
        assert result["h"] < -0.5

    def test_clamped_inputs(self) -> None:
        """Values outside [0,1] are clamped."""
        result = cohens_h(-0.1, 1.5)
        # Should be equivalent to (0.0, 1.0)
        assert abs(result["h"] - math.pi) < 0.01


# ---------------------------------------------------------------------------
# Bootstrap CI
# ---------------------------------------------------------------------------


class TestBootstrapCI:
    def test_deterministic_with_seed(self) -> None:
        """Same seed produces same CI."""
        pairs: list[BinaryPair] = [(False, True)] * 15 + [(True, True)] * 5
        r1 = bootstrap_ci(pairs, seed=42, n_bootstrap=1000)
        r2 = bootstrap_ci(pairs, seed=42, n_bootstrap=1000)
        assert r1["ci_lower"] == r2["ci_lower"]
        assert r1["ci_upper"] == r2["ci_upper"]

    def test_ci_contains_observed(self) -> None:
        """CI should generally contain the observed uplift."""
        pairs: list[BinaryPair] = (
            [(False, True)] * 10 + [(True, False)] * 5 + [(True, True)] * 5 + [(False, False)] * 10
        )
        result = bootstrap_ci(pairs, seed=123, n_bootstrap=5000)
        # Observed uplift: (10+5)/30 B-rate - (5+5)/30 A-rate
        assert result["ci_lower"] <= result["uplift"] <= result["ci_upper"]

    def test_empty_pairs(self) -> None:
        """Empty input -> zero uplift, zero CI."""
        result = bootstrap_ci([])
        assert result["uplift"] == 0.0
        assert result["ci_lower"] == 0.0
        assert result["ci_upper"] == 0.0
        assert result["n_bootstrap"] == 0

    def test_positive_uplift(self) -> None:
        """Strong B advantage should give positive CI."""
        pairs: list[BinaryPair] = [(False, True)] * 20 + [(False, False)] * 5
        result = bootstrap_ci(pairs, seed=42, n_bootstrap=5000)
        assert result["uplift"] > 0
        assert result["ci_lower"] > 0  # Entirely positive


class TestBootstrapMeanCI:
    def test_degenerate_inputs_return_point_interval(self) -> None:
        result = bootstrap_mean_ci([3.0], n_bootstrap=0, seed=1)
        assert result["mean"] == 3.0
        assert result["ci_lower"] == 3.0
        assert result["ci_upper"] == 3.0

    def test_seeded_bootstrap_is_reproducible(self) -> None:
        values = [1.0, 2.0, 4.0, 8.0]
        left = bootstrap_mean_ci(values, n_bootstrap=500, seed=7)
        right = bootstrap_mean_ci(values, n_bootstrap=500, seed=7)
        assert left["ci_lower"] == right["ci_lower"]
        assert left["ci_upper"] == right["ci_upper"]


class TestPairedNumericDeltas:
    def test_returns_b_minus_a(self) -> None:
        pairs: list[NumericPair] = [(10.0, 8.0), (4.0, 7.5)]
        assert paired_numeric_deltas(pairs) == [-2.0, 3.5]


# ---------------------------------------------------------------------------
# Holm-Bonferroni
# ---------------------------------------------------------------------------


class TestHolmBonferroni:
    def test_basic_correction(self) -> None:
        """Three p-values with different significance levels."""
        p_values = {"a": 0.01, "b": 0.04, "c": 0.06}
        result = holm_bonferroni(p_values, alpha=0.05)

        # a: p=0.01, compare with 0.05/3 = 0.0167 -> rejected
        assert result["a"]["rejected"] is True
        assert result["a"]["rank"] == 1

        # b: p=0.04, compare with 0.05/2 = 0.025 -> NOT rejected
        assert result["b"]["rejected"] is False
        assert result["b"]["rank"] == 2

        # c: p=0.06, not rejected since b wasn't
        assert result["c"]["rejected"] is False
        assert result["c"]["rank"] == 3

    def test_all_significant(self) -> None:
        """Very small p-values -> all rejected."""
        p_values = {"x": 0.001, "y": 0.002, "z": 0.003}
        result = holm_bonferroni(p_values, alpha=0.05)
        assert all(v["rejected"] for v in result.values())

    def test_none_significant(self) -> None:
        """Large p-values -> none rejected."""
        p_values = {"a": 0.5, "b": 0.6}
        result = holm_bonferroni(p_values, alpha=0.05)
        assert not any(v["rejected"] for v in result.values())

    def test_empty_input(self) -> None:
        """Empty dict -> empty result."""
        assert holm_bonferroni({}) == {}

    def test_single_p_value(self) -> None:
        """Single test -> just compare directly against alpha."""
        result = holm_bonferroni({"only": 0.03}, alpha=0.05)
        assert result["only"]["rejected"] is True

    def test_adjusted_p_values_monotonic(self) -> None:
        """Adjusted p-values should be non-decreasing when sorted by original p."""
        p_values = {"a": 0.01, "b": 0.02, "c": 0.03, "d": 0.04}
        result = holm_bonferroni(p_values)
        sorted_by_rank = sorted(result.values(), key=lambda v: v["rank"])
        adjusted = [v["adjusted_p"] for v in sorted_by_rank]
        for i in range(len(adjusted) - 1):
            assert adjusted[i] <= adjusted[i + 1]


# ---------------------------------------------------------------------------
# Power report
# ---------------------------------------------------------------------------


class TestPowerReport:
    def test_sufficient_power(self) -> None:
        """Large n_discordant with strong uplift -> sufficient."""
        result = power_report(200, 80, 0.4)
        assert result["sufficient"] is True
        assert result["approximate_power"] >= 0.8

    def test_zero_discordant(self) -> None:
        """Zero discordant -> zero power."""
        result = power_report(50, 0, 0.2)
        assert result["approximate_power"] == 0.0
        assert result["sufficient"] is False

    def test_zero_uplift(self) -> None:
        """Zero uplift -> zero power."""
        result = power_report(50, 20, 0.0)
        assert result["approximate_power"] == 0.0
        assert result["sufficient"] is False

    def test_moderate_power(self) -> None:
        """Moderate power: 0.5 <= power < 0.8."""
        result = power_report(100, 20, 0.5)
        assert not result["sufficient"]
        assert "Moderate power" in result["recommendation"]

    def test_low_power(self) -> None:
        """Very few discordant pairs -> low power."""
        result = power_report(50, 5, 0.1)
        assert not result["sufficient"]
        assert "Low power" in result["recommendation"]


# ---------------------------------------------------------------------------
# Majority vote / any-success aggregation
# ---------------------------------------------------------------------------


class TestAggregation:
    def test_majority_vote_basic(self) -> None:
        """Majority vote: 2/3 runs succeed -> True."""
        runs: dict[str, list[BinaryPair]] = {
            "case1": [(True, False), (True, True), (False, True)],
            "case2": [(False, False), (False, False), (False, True)],
        }
        result = majority_vote_aggregation(runs)
        assert len(result) == 2
        # case1: A=2/3 -> True, B=2/3 -> True
        assert result[0] == (True, True)
        # case2: A=0/3 -> False, B=1/3 -> False
        assert result[1] == (False, False)

    def test_any_success_basic(self) -> None:
        """Any-success: any True -> True."""
        runs: dict[str, list[BinaryPair]] = {
            "case1": [(False, False), (True, False)],
            "case2": [(False, False), (False, False)],
        }
        result = any_success_aggregation(runs)
        assert len(result) == 2
        assert result[0] == (True, False)
        assert result[1] == (False, False)

    def test_empty_runs(self) -> None:
        """Empty case runs are skipped."""
        runs: dict[str, list[BinaryPair]] = {"case1": [], "case2": [(True, True)]}
        result = majority_vote_aggregation(runs)
        assert len(result) == 1


class TestAnySuccessAggregationEmptyRuns:
    def test_empty_case_runs_skipped(self) -> None:
        """Empty case runs are skipped in any_success_aggregation (line 297)."""
        runs: dict[str, list[BinaryPair]] = {
            "case1": [],
            "case2": [(True, False), (False, True)],
        }
        result = any_success_aggregation(runs)
        assert len(result) == 1
        assert result[0] == (True, True)

    def test_all_empty_case_runs(self) -> None:
        """All empty case runs produce no output."""
        runs: dict[str, list[BinaryPair]] = {"a": [], "b": []}
        result = any_success_aggregation(runs)
        assert result == []


# ---------------------------------------------------------------------------
# Full analysis pipeline
# ---------------------------------------------------------------------------


class TestAnalyzePairedResults:
    def test_full_pipeline(self) -> None:
        """End-to-end: analyze_paired_results with matched rows."""
        rows = [
            {
                "run_id": "r1",
                "case_id": "c1",
                "condition": "base_repair",
                "category": "hidden_state",
                "success": False,
            },
            {
                "run_id": "r1",
                "case_id": "c1",
                "condition": "cot_repair",
                "category": "hidden_state",
                "success": True,
            },
            {
                "run_id": "r1",
                "case_id": "c2",
                "condition": "base_repair",
                "category": "parse_value",
                "success": True,
            },
            {
                "run_id": "r1",
                "case_id": "c2",
                "condition": "cot_repair",
                "category": "parse_value",
                "success": True,
            },
        ]
        result = analyze_paired_results(rows, seed=42, condition_b="cot_repair")
        assert result["n_pairs"] == 2
        assert result["a_success_rate"] == 0.5
        assert result["b_success_rate"] == 1.0
        assert "overall_mcnemar" in result
        assert "overall_cohens_h" in result
        assert "per_category" in result
        assert "hidden_state" in result["per_category"]

    def test_full_pipeline_with_custom_pair_conditions(self) -> None:
        rows = [
            {
                "run_id": "r1",
                "case_id": "c1",
                "condition": "base_repair",
                "category": "hidden_state",
                "success": False,
            },
            {
                "run_id": "r1",
                "case_id": "c1",
                "condition": "layered_adaptive_gated",
                "category": "hidden_state",
                "success": True,
            },
            {
                "run_id": "r1",
                "case_id": "c2",
                "condition": "base_repair",
                "category": "parse_value",
                "success": True,
            },
            {
                "run_id": "r1",
                "case_id": "c2",
                "condition": "layered_adaptive_gated",
                "category": "parse_value",
                "success": True,
            },
        ]
        result = analyze_paired_results(
            rows,
            seed=42,
            condition_a="base_repair",
            condition_b="layered_adaptive_gated",
        )
        assert result["n_pairs"] == 2
        assert result["condition_a"] == "base_repair"
        assert result["condition_b"] == "layered_adaptive_gated"
        assert result["a_success_rate"] == 0.5
        assert result["b_success_rate"] == 1.0

    def test_no_matched_pairs(self) -> None:
        """Unmatched conditions -> error."""
        rows = [
            {
                "run_id": "r1",
                "case_id": "c1",
                "condition": "base_repair",
                "category": "hidden_state",
                "success": False,
            },
        ]
        result = analyze_paired_results(rows)
        assert "error" in result

    def test_empty_rows(self) -> None:
        """Empty input -> error."""
        result = analyze_paired_results([])
        assert "error" in result


class TestAnalyzeConditionMatrix:
    def test_matrix_pairs_all_conditions(self) -> None:
        rows = [
            {
                "run_id": "r1",
                "case_id": "c1",
                "condition": "base_repair",
                "category": "hidden_state",
                "success": False,
            },
            {
                "run_id": "r1",
                "case_id": "c1",
                "condition": "cot_repair",
                "category": "hidden_state",
                "success": True,
            },
            {
                "run_id": "r1",
                "case_id": "c1",
                "condition": "layered_adaptive_gated",
                "category": "hidden_state",
                "success": True,
            },
            {
                "run_id": "r1",
                "case_id": "c2",
                "condition": "base_repair",
                "category": "parse_value",
                "success": False,
            },
            {
                "run_id": "r1",
                "case_id": "c2",
                "condition": "cot_repair",
                "category": "parse_value",
                "success": False,
            },
            {
                "run_id": "r1",
                "case_id": "c2",
                "condition": "layered_adaptive_gated",
                "category": "parse_value",
                "success": True,
            },
        ]
        matrix = analyze_condition_matrix(rows, seed=7, n_bootstrap=100)
        assert len(matrix) == 3
        pairs = {(item.get("condition_a"), item.get("condition_b")) for item in matrix}
        assert ("base_repair", "cot_repair") in pairs
        assert ("base_repair", "layered_adaptive_gated") in pairs
        assert ("cot_repair", "layered_adaptive_gated") in pairs


# ---------------------------------------------------------------------------
# Cohen's h negligible magnitude (line 108)
# ---------------------------------------------------------------------------


class TestCohensHNegligible:
    def test_nearly_identical_rates(self) -> None:
        """Very close proportions -> negligible h."""
        result = cohens_h(0.501, 0.502)
        assert result["magnitude"] == "negligible"
        assert abs(result["h"]) < 0.2

    def test_medium_effect_size(self) -> None:
        """Moderate difference -> medium h (0.5 <= |h| < 0.8)."""
        result = cohens_h(0.2, 0.5)
        assert result["magnitude"] == "medium"
        assert 0.5 <= abs(result["h"]) < 0.8


# ---------------------------------------------------------------------------
# Chi-squared Wilson-Hilferty approximation (line 297 / 491-493)
# ---------------------------------------------------------------------------


class TestChi2SfFallback:
    def test_chi2_sf_negative_x_returns_one(self) -> None:
        """x < 0 should return 1.0 (line 478)."""
        from evals.stats import _chi2_sf

        assert _chi2_sf(-1.0, df=1) == 1.0

    def test_chi2_sf_no_scipy_df1(self, monkeypatch) -> None:
        """Without scipy, df=1 uses Z-based approximation (lines 487-489)."""
        import builtins

        import evals.stats as stats_mod

        real_import = builtins.__import__

        def fail_scipy(name, *a, **kw):
            if "scipy" in name:
                raise ImportError("no scipy")
            return real_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", fail_scipy)
        p = stats_mod._chi2_sf(3.84, df=1)
        assert 0.0 < p < 0.1

    def test_chi2_sf_no_scipy_general_df(self, monkeypatch) -> None:
        """Without scipy, df>1 uses Wilson-Hilferty (lines 492-493)."""
        import builtins

        import evals.stats as stats_mod

        real_import = builtins.__import__

        def fail_scipy(name, *a, **kw):
            if "scipy" in name:
                raise ImportError("no scipy")
            return real_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", fail_scipy)
        p = stats_mod._chi2_sf(7.81, df=3)
        assert 0.0 < p < 0.1


# ---------------------------------------------------------------------------
# Manual binomial fallback (lines 447-462)
# ---------------------------------------------------------------------------


class TestExactBinomialFallback:
    def test_manual_fallback_when_scipy_unavailable(self, monkeypatch) -> None:
        """When scipy is not available, use the manual binomial sum."""
        import builtins

        import evals.stats as stats_mod

        real_import = builtins.__import__

        def fail_scipy(name, *a, **kw):
            if "scipy" in name:
                raise ImportError("no scipy")
            return real_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", fail_scipy)
        # 3 out of 10 under H0:p=0.5 -> two-sided p
        p = stats_mod._exact_binomial_two_sided(3, 10)
        assert 0.0 < p <= 1.0
        # Compare with known value: P(X<=3) + P(X>=7) for Binom(10,0.5)
        # = 2*P(X<=3) by symmetry = 2*(C(10,0)*0.5^10 + ... + C(10,3)*0.5^10)
        assert abs(p - 0.34375) < 0.01

    def test_manual_fallback_symmetric_case(self, monkeypatch) -> None:
        """k == n-k (symmetric) should return 1.0."""
        import builtins

        import evals.stats as stats_mod

        real_import = builtins.__import__

        def fail_scipy(name, *a, **kw):
            if "scipy" in name:
                raise ImportError("no scipy")
            return real_import(name, *a, **kw)

        monkeypatch.setattr(builtins, "__import__", fail_scipy)
        p = stats_mod._exact_binomial_two_sided(5, 10)
        assert p == 1.0

    def test_binom_pmf_out_of_range(self) -> None:
        """_binom_pmf returns 0 for k < 0 or k > n."""
        from evals.stats import _binom_pmf

        assert _binom_pmf(-1, 10, 0.5) == 0.0
        assert _binom_pmf(11, 10, 0.5) == 0.0


# ---------------------------------------------------------------------------
# Per-category analysis with Holm-Bonferroni
# ---------------------------------------------------------------------------


class TestAnalyzeWithCategories:
    def test_per_category_holm_bonferroni(self) -> None:
        """analyze_paired_results with multiple categories triggers HB correction."""
        rows = []
        # Category A: strong B advantage (10 cases)
        for i in range(10):
            rows.append(
                {
                    "run_id": "r1",
                    "case_id": f"cA_{i}",
                    "condition": "base_repair",
                    "category": "cat_A",
                    "success": False,
                }
            )
            rows.append(
                {
                    "run_id": "r1",
                    "case_id": f"cA_{i}",
                    "condition": "cot_repair",
                    "category": "cat_A",
                    "success": True,
                }
            )
        # Category B: no advantage (10 cases, all same)
        for i in range(10):
            rows.append(
                {
                    "run_id": "r1",
                    "case_id": f"cB_{i}",
                    "condition": "base_repair",
                    "category": "cat_B",
                    "success": True,
                }
            )
            rows.append(
                {
                    "run_id": "r1",
                    "case_id": f"cB_{i}",
                    "condition": "cot_repair",
                    "category": "cat_B",
                    "success": True,
                }
            )
        result = analyze_paired_results(
            rows,
            seed=42,
            n_bootstrap=100,
            condition_b="cot_repair",
        )
        assert "per_category" in result
        assert "cat_A" in result["per_category"]
        assert "cat_B" in result["per_category"]
        assert "holm_bonferroni" in result

    def test_empty_category_pairs_skipped(self) -> None:
        """Categories with no paired data are skipped."""
        rows = [
            {
                "run_id": "r1",
                "case_id": "c1",
                "condition": "base_repair",
                "category": "only_a",
                "success": False,
            },
            {
                "run_id": "r1",
                "case_id": "c1",
                "condition": "cot_repair",
                "category": "only_a",
                "success": True,
            },
        ]
        result = analyze_paired_results(rows, seed=42, n_bootstrap=100, condition_b="cot_repair")
        assert result["n_pairs"] == 1

    def test_empty_cat_pairs_continue(self, monkeypatch) -> None:
        """When _extract_pairs returns a category with an empty list, it is skipped (line 353)."""
        import evals.stats as stats_mod

        pairs = [(True, False), (False, True)]
        categories = {"real_cat": [(True, False)], "empty_cat": []}

        monkeypatch.setattr(stats_mod, "_extract_pairs", lambda *a, **kw: (pairs, categories))
        result = analyze_paired_results([{"dummy": True}], seed=42, n_bootstrap=50, condition_b="b")
        assert "per_category" in result
        assert "real_cat" in result["per_category"]
        assert "empty_cat" not in result["per_category"]

    def test_no_categories_skips_per_category(self, monkeypatch) -> None:
        """When _extract_pairs returns empty categories dict, per_category is omitted (branch [348,371])."""
        import evals.stats as stats_mod

        pairs = [(True, False), (False, True)]
        monkeypatch.setattr(stats_mod, "_extract_pairs", lambda *a, **kw: (pairs, {}))
        result = analyze_paired_results([{"dummy": True}], seed=42, n_bootstrap=50, condition_b="b")
        assert "per_category" not in result
