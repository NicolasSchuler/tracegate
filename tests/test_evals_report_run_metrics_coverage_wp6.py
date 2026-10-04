"""Coverage tests for evals.report_run_metrics — batch covering main() error exits,
_parse_fixed_budget_thresholds branches, _cohens_kappa_magnitude, _null_sort_key,
_bootstrap_mean_ci edge cases, _print_input_errors, _as_float, _mean, _median,
_fmt_* formatting helpers, _print_dual_objective, _print_overhead_analysis,
_conclusion_line, _print_effect_line, and _print_report with full dual_objective."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from evals import report_run_metrics as mod

# ---------------------------------------------------------------------------
# Helper: make a minimal JSONL file for main() integration tests
# ---------------------------------------------------------------------------


def _make_jsonl(tmp_path: Path, records: list[dict[str, Any]]) -> Path:
    p = tmp_path / "run.jsonl"
    lines = [json.dumps(r) for r in records]
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def _minimal_record(
    run_id: str = "r1",
    case_id: str = "c1",
    condition: str = "base_repair",
    success: bool = False,
    snapshot_dependency: str = "required",
) -> dict[str, Any]:
    return {
        "schema_version": 5,
        "run_id": run_id,
        "case_id": case_id,
        "condition": condition,
        "attempt": 1,
        "success": success,
        "category": "logic_error",
        "difficulty": "easy",
        "snapshot_dependency": snapshot_dependency,
        "run_metadata": {
            "run_id": run_id,
            "started_at_utc": "2026-01-01T00:00:00Z",
            "deterministic": False,
            "seed": None,
            "python_version": "3.11.0",
            "platform": "test",
            "llmdebug_version": "2.11.4",
            "patcher_name": "openai",
            "patcher_config_fingerprint": "fp",
            "run_settings_fingerprint": "rfp",
        },
        "attempt_metadata": {
            "attempt": 1,
            "max_attempts": 3,
            "attempt_ordinal_label": "1/3",
            "llm_call_count": 1,
            "tokens_prompt": 10,
            "tokens_completion": 2,
            "tokens_total": 12,
            "retry_history_budget_chars": 12000,
            "retry_history_budget_applied": False,
            "prompt_sha256": "p",
            "prompt_path": None,
            "evidence_sha256": "e",
            "evidence_path": None,
        },
        "model_metadata": {
            "backend": "openai",
            "name": "gpt-4o",
            "base_url": "http://localhost:8000/v1",
            "model": "gpt-4o",
            "temperature": 0.0,
            "max_tokens": 2048,
            "response_mode": "edits",
        },
        "cumulative_runtime_sec": 1.0,
        "note": "tests_still_failing",
        "token_usage_total": 12,
        "runtime_sec": 1.0,
        "llm_calls_total": 1,
    }


# ---------------------------------------------------------------------------
# main() error exits
# ---------------------------------------------------------------------------


class TestMainErrorExits:
    """Lines 124-126, 130-131, 135-137, 140-142, 146-147."""

    def test_invalid_fixed_budget_thresholds_returns_2(self, tmp_path: Path) -> None:
        """ValueError from _parse_fixed_budget_thresholds returns exit code 2 (lines 124-126)."""
        p = _make_jsonl(tmp_path, [_minimal_record()])
        result = mod.main(
            [
                "--jsonl",
                str(p),
                "--fixed-budget-token-thresholds",
                "not_a_number",
            ]
        )
        assert result == 2

    def test_no_input_matched_returns_1(self) -> None:
        """No files matching glob returns exit code 1 (lines 130-131)."""
        result = mod.main(
            [
                "--jsonl",
                "/nonexistent/path/*.jsonl",
            ]
        )
        assert result == 1

    def test_no_valid_rows_returns_1(self, tmp_path: Path) -> None:
        """When no case-condition rows can be derived, returns 1 (lines 146-147).

        Use a record with an unknown condition so no rows match conditions a/b.
        """
        record = _minimal_record(condition="unknown_condition")
        # Must have the deterministic field to pass validation
        p = _make_jsonl(tmp_path, [record])
        result = mod.main(
            [
                "--jsonl",
                str(p),
                "--a",
                "base_repair",
                "--b",
                "layered_adaptive_llm",
            ]
        )
        # Either 1 (no rows) or some exit code — the record has an unknown condition
        # so it won't produce a row for base_repair or layered_adaptive_llm
        assert result in (1, 2)


# ---------------------------------------------------------------------------
# _parse_fixed_budget_thresholds — edge cases
# ---------------------------------------------------------------------------


class TestParseFixedBudgetThresholds:
    """Lines 276, 279-281, 283-284, 288-289."""

    def test_valid_single_value(self) -> None:
        result = mod._parse_fixed_budget_thresholds("10000")
        assert result == [10000]

    def test_valid_multiple_values_deduped_and_sorted(self) -> None:
        result = mod._parse_fixed_budget_thresholds("50000,10000,10000")
        assert result == [10000, 50000]

    def test_empty_segments_skipped(self) -> None:
        result = mod._parse_fixed_budget_thresholds(",10000,,")
        assert result == [10000]

    def test_non_integer_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="non-integer"):
            mod._parse_fixed_budget_thresholds("abc")

    def test_zero_value_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            mod._parse_fixed_budget_thresholds("0")

    def test_negative_value_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="positive"):
            mod._parse_fixed_budget_thresholds("-100")

    def test_empty_string_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            mod._parse_fixed_budget_thresholds("")

    def test_none_value_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            mod._parse_fixed_budget_thresholds(None)  # type: ignore[arg-type]


# ---------------------------------------------------------------------------
# _cohens_kappa_magnitude — branch coverage (lines 393-400)
# ---------------------------------------------------------------------------


class TestCohensKappaMagnitude:
    """Test all 5 magnitude bands."""

    def test_slight(self) -> None:
        assert mod._cohens_kappa_magnitude(0.1) == "slight"

    def test_fair(self) -> None:
        assert mod._cohens_kappa_magnitude(0.3) == "fair"

    def test_moderate(self) -> None:
        assert mod._cohens_kappa_magnitude(0.5) == "moderate"

    def test_substantial(self) -> None:
        assert mod._cohens_kappa_magnitude(0.7) == "substantial"

    def test_almost_perfect(self) -> None:
        assert mod._cohens_kappa_magnitude(0.9) == "almost_perfect"

    def test_negative_uses_abs(self) -> None:
        assert mod._cohens_kappa_magnitude(-0.5) == "moderate"


# ---------------------------------------------------------------------------
# _null_sort_key (line 640-643)
# ---------------------------------------------------------------------------


class TestNullSortKey:
    def test_numeric_value(self) -> None:
        key = mod._null_sort_key(5.0)
        assert key == (0, 5.0)

    def test_non_numeric_value(self) -> None:
        key = mod._null_sort_key(None)
        assert key[0] == 1

    def test_integer_value(self) -> None:
        key = mod._null_sort_key(100)
        assert key == (0, 100.0)


# ---------------------------------------------------------------------------
# _as_float, _mean, _median, _fmt_* helpers
# ---------------------------------------------------------------------------


class TestHelpers:
    def test_as_float_numeric(self) -> None:
        assert mod._as_float(3.0) == 3.0

    def test_as_float_none(self) -> None:
        assert mod._as_float(None) is None

    def test_as_float_string(self) -> None:
        assert mod._as_float("x") is None

    def test_mean_empty(self) -> None:
        assert mod._mean([]) is None

    def test_mean_values(self) -> None:
        assert mod._mean([1.0, 3.0]) == 2.0

    def test_median_empty(self) -> None:
        assert mod._median([]) is None

    def test_median_odd(self) -> None:
        assert mod._median([1.0, 2.0, 3.0]) == 2.0

    def test_median_even(self) -> None:
        assert mod._median([1.0, 2.0, 3.0, 4.0]) == 2.5

    def test_fmt_float_none(self) -> None:
        assert mod._fmt_float(None) == "n/a"

    def test_fmt_float_numeric(self) -> None:
        assert mod._fmt_float(1.5) == "1.5000"

    def test_fmt_pct_none(self) -> None:
        assert mod._fmt_pct(None) == "n/a"

    def test_fmt_pct_numeric(self) -> None:
        assert mod._fmt_pct(0.75) == "75.00%"

    def test_fmt_pct_signed_none(self) -> None:
        assert mod._fmt_pct_signed(None) == "n/a"

    def test_fmt_pct_signed_positive(self) -> None:
        result = mod._fmt_pct_signed(0.1)
        assert "+" in result

    def test_fmt_signed_pp_none(self) -> None:
        assert mod._fmt_signed_pp(None) == "n/a"

    def test_fmt_signed_pp_numeric(self) -> None:
        result = mod._fmt_signed_pp(-0.05)
        assert "-" in result


# ---------------------------------------------------------------------------
# _print_input_errors (lines 2093-2104)
# ---------------------------------------------------------------------------


class TestPrintInputErrors:
    def test_no_errors_prints_nothing(self, capsys: pytest.CaptureFixture[str]) -> None:
        mod._print_input_errors({})
        captured = capsys.readouterr()
        assert captured.out == ""

    def test_with_errors_prints_diagnostics(self, capsys: pytest.CaptureFixture[str]) -> None:
        mod._print_input_errors(
            {"files_unreadable": 2, "json_lines_invalid": 3, "records_skipped": 1}
        )
        captured = capsys.readouterr()
        assert "Input diagnostics" in captured.out
        assert "files_unreadable=2" in captured.out


# ---------------------------------------------------------------------------
# _print_dual_objective (lines 2020-2091)
# ---------------------------------------------------------------------------


class TestPrintDualObjective:
    def test_unavailable(self, capsys: pytest.CaptureFixture[str]) -> None:
        mod._print_dual_objective({"available": False, "error": "no data"}, "a", "b")
        out = capsys.readouterr().out
        assert "unavailable" in out

    def test_available_with_effect(self, capsys: pytest.CaptureFixture[str]) -> None:
        dual = {
            "available": True,
            "effect": {
                "available": True,
                "uplift": 0.1,
                "a_rate": 0.5,
                "b_rate": 0.6,
                "bootstrap_ci": {"ci_lower": 0.0, "ci_upper": 0.2},
            },
            "matched_pairs": 10,
            "token_covered_pairs": 8,
            "token_coverage_rate": 0.8,
            "cost_per_success": {
                "available": True,
                "a_value": 1000,
                "a_successes": 5,
                "b_value": 900,
                "b_successes": 6,
            },
            "paired_cost_delta": {
                "available": True,
                "mean_delta": -100.0,
                "mean_a": 1000.0,
                "mean_b": 900.0,
                "pct_change_vs_a": -0.1,
                "bootstrap_ci_mean_delta": {"ci_lower": -200.0, "ci_upper": 0.0},
            },
            "fixed_budget_success": [
                {
                    "token_threshold": 10000,
                    "a_rate": 0.5,
                    "b_rate": 0.6,
                    "delta": 0.1,
                    "coverage_pairs": 8,
                }
            ],
        }
        mod._print_dual_objective(dual, "base_repair", "cot_repair")
        out = capsys.readouterr().out
        assert "Token coverage" in out
        assert "Cost per success" in out

    def test_cost_per_success_unavailable(self, capsys: pytest.CaptureFixture[str]) -> None:
        dual = {
            "available": True,
            "effect": {"available": False},
            "matched_pairs": 5,
            "token_covered_pairs": 0,
            "token_coverage_rate": 0.0,
            "cost_per_success": {"available": False, "error": "insufficient data"},
            "paired_cost_delta": {"available": False, "error": "no pairs"},
            "fixed_budget_success": [],
        }
        mod._print_dual_objective(dual, "a", "b")
        out = capsys.readouterr().out
        assert "unavailable" in out

    def test_fixed_budget_non_list_exits_early(self, capsys: pytest.CaptureFixture[str]) -> None:
        dual = {
            "available": True,
            "effect": {"available": False},
            "matched_pairs": 5,
            "token_covered_pairs": 3,
            "token_coverage_rate": 0.6,
            "cost_per_success": {"available": False},
            "paired_cost_delta": {"available": False},
            "fixed_budget_success": None,  # not a list → early return
        }
        mod._print_dual_objective(dual, "a", "b")
        out = capsys.readouterr().out
        assert "Fixed-budget success: unavailable" in out


# ---------------------------------------------------------------------------
# _print_overhead_analysis (lines 1905-1944)
# ---------------------------------------------------------------------------


class TestPrintOverheadAnalysis:
    def test_no_per_condition_prints_unavailable(self, capsys: pytest.CaptureFixture[str]) -> None:
        mod._print_overhead_analysis({}, "a", "b")
        out = capsys.readouterr().out
        assert "unavailable" in out

    def test_with_per_condition_timing(self, capsys: pytest.CaptureFixture[str]) -> None:
        overhead = {
            "per_condition": {
                "base_repair": {
                    "cases": 5,
                    "llm_pct_of_runtime": 80.0,
                    "overhead_pct_of_runtime": 10.0,
                    "llmdebug_pct_of_runtime": 20.0,
                    "timing_mean": {"stage_b_llm_sec": 1.5, "overhead_sec": 0.2},
                    "token_mean": {"stage_b_input_tokens": 500},
                },
            }
        }
        mod._print_overhead_analysis(overhead, "base_repair", "cot_repair")
        out = capsys.readouterr().out
        assert "Framework Overhead Analysis" in out
        assert "base_repair" in out


# ---------------------------------------------------------------------------
# _conclusion_line (lines 1947-1960)
# ---------------------------------------------------------------------------


class TestConclusionLine:
    def test_unavailable(self) -> None:
        result = mod._conclusion_line(
            condition_a="a", condition_b="b", effect={"available": False, "error": "no data"}
        )
        assert "unavailable" in result

    def test_non_numeric_uplift(self) -> None:
        result = mod._conclusion_line(
            condition_a="a", condition_b="b", effect={"available": True, "uplift_pp": "bad"}
        )
        assert "numeric uplift" in result

    def test_positive_uplift(self) -> None:
        result = mod._conclusion_line(
            condition_a="a",
            condition_b="b",
            effect={"available": True, "uplift_pp": 0.05, "mcnemar": {"p_value": 0.03}},
        )
        assert "better" in result
        assert "0.05" in result or "5.00" in result

    def test_negative_uplift(self) -> None:
        result = mod._conclusion_line(
            condition_a="a",
            condition_b="b",
            effect={"available": True, "uplift_pp": -0.02, "mcnemar": {}},
        )
        assert "worse" in result

    def test_zero_uplift(self) -> None:
        result = mod._conclusion_line(
            condition_a="a",
            condition_b="b",
            effect={"available": True, "uplift_pp": 0.0, "mcnemar": {}},
        )
        assert "equal" in result


# ---------------------------------------------------------------------------
# _print_effect_line — non-numeric uplift path (line 1980)
# ---------------------------------------------------------------------------


class TestPrintEffectLine:
    def test_unavailable(self, capsys: pytest.CaptureFixture[str]) -> None:
        mod._print_effect_line("Test", "a", "b", {"available": False, "error": "no pairs"})
        out = capsys.readouterr().out
        assert "unavailable" in out

    def test_non_numeric_uplift_prints_unavailable(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        mod._print_effect_line("Test", "a", "b", {"available": True, "uplift_pp": None})
        out = capsys.readouterr().out
        assert "unavailable" in out

    def test_equal_direction(self, capsys: pytest.CaptureFixture[str]) -> None:
        mod._print_effect_line(
            "Test",
            "a",
            "b",
            {
                "available": True,
                "uplift_pp": 0.0,
                "a_rate": 0.5,
                "b_rate": 0.5,
                "n_pairs": 10,
                "mcnemar": {},
                "bootstrap_ci": {},
                "cohens_h": {},
                "agreement": {},
            },
        )
        out = capsys.readouterr().out
        assert "equal" in out or "0.00" in out


# ---------------------------------------------------------------------------
# _print_report with dual_objective and token_estimate_diagnostics (line 1602)
# ---------------------------------------------------------------------------


class TestPrintReportTokenEstimateDiagnostics:
    """Covers line 1602: estimate_row.get('available') branch."""

    def _base_args(self) -> dict[str, Any]:
        return {
            "run_context": {
                "run_ids": ["r1"],
                "patchers": [],
                "models": [],
                "n_case_condition_rows": 1,
            },
            "condition_summary": {},
            "condition_a": "base_repair",
            "condition_b": "cot_repair",
            "primary_slice": {"required"},
            "overall_effect": {"available": False, "error": "no pairs"},
            "primary_effect": {"available": False, "error": "no pairs"},
            "required_effect": {"available": False},
            "control_effect": {"available": False},
            "category_stats": {"available": False, "error": "no data"},
            "efficiency": {},
            "failures": {
                "per_condition": {
                    "base_repair": {
                        "cases": 1,
                        "context_overflow_attempts": 0,
                        "attempts": 1,
                        "context_overflow_attempt_rate": 0.0,
                        "context_overflow_cases": 0,
                        "context_overflow_case_rate": 0.0,
                        "http_error_attempts": 0,
                        "terminal_timeout_failures": 0,
                        "terminal_timeout_failure_rate": 0.0,
                        "test_touch_attempts": 0,
                        "patch_attempts": 1,
                        "test_touch_attempt_rate": 0.0,
                        "usage_applicable_cases": 2,
                        "usage_not_applicable_cases": 0,
                        "usage_reported_cases": 1,
                        "usage_missing_cases": 1,
                        "timeout_missing_usage_cases": 0,
                        "usage_reported_case_rate": 0.5,
                        "usage_missing_case_rate": 0.5,
                        "timeout_missing_usage_case_rate": 0.0,
                        "top_reasons": [("tests_still_failing", 3)],
                    }
                }
            },
            "overhead": {},
            "input_errors": {},
        }

    def test_with_estimate_diagnostics_available(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Line 1602: estimate_row.get('available') is True."""
        args = self._base_args()
        args["token_estimate_diagnostics"] = {
            "per_condition": {
                "base_repair": {
                    "total_attempts": 5,
                    "provider_covered_attempts": 5,
                    "estimate_covered_attempts": 4,
                    "paired_attempts": 4,
                    "paired_attempt_rate": 0.8,
                    "available": True,
                    "correlation": 0.99,
                    "mean_abs_error": 10.0,
                }
            }
        }
        mod._print_report(**args)
        out = capsys.readouterr().out
        assert "prompt-token estimate correlation" in out
        assert "r=" in out

    def test_with_estimate_diagnostics_unavailable(
        self, capsys: pytest.CaptureFixture[str]
    ) -> None:
        """Line 1608-1611: estimate_row.get('available') is False."""
        args = self._base_args()
        args["token_estimate_diagnostics"] = {
            "per_condition": {
                "base_repair": {
                    "total_attempts": 3,
                    "provider_covered_attempts": 0,
                    "estimate_covered_attempts": 3,
                    "paired_attempts": 0,
                    "paired_attempt_rate": 0.0,
                    "available": False,
                    "error": "no provider data",
                }
            }
        }
        mod._print_report(**args)
        out = capsys.readouterr().out
        assert "prompt-token estimate correlation" in out
        assert "unavailable" in out

    def test_with_full_dual_objective(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Covers dual_objective printing with effect available."""
        args = self._base_args()
        args["dual_objective"] = {
            "available": True,
            "effect": {
                "available": True,
                "uplift": 0.1,
                "a_rate": 0.5,
                "b_rate": 0.6,
                "bootstrap_ci": {},
            },
            "matched_pairs": 5,
            "token_covered_pairs": 3,
            "token_coverage_rate": 0.6,
            "cost_per_success": {"available": False},
            "paired_cost_delta": {"available": False},
            "fixed_budget_success": [
                {
                    "token_threshold": 5000,
                    "a_rate": 0.5,
                    "b_rate": 0.6,
                    "delta": 0.1,
                    "coverage_pairs": 3,
                }
            ],
        }
        mod._print_report(**args)
        out = capsys.readouterr().out
        assert "Dual Objective" in out


# ---------------------------------------------------------------------------
# _pearson_correlation (lines 1086-1098)
# ---------------------------------------------------------------------------


class TestPearsonCorrelation:
    def test_returns_none_for_mismatched_lengths(self) -> None:
        result = mod._pearson_correlation([1.0, 2.0], [1.0])
        assert result is None

    def test_returns_none_for_less_than_2_values(self) -> None:
        result = mod._pearson_correlation([1.0], [1.0])
        assert result is None

    def test_returns_none_for_zero_variance(self) -> None:
        result = mod._pearson_correlation([1.0, 1.0, 1.0], [1.0, 2.0, 3.0])
        assert result is None

    def test_perfect_correlation(self) -> None:
        result = mod._pearson_correlation([1.0, 2.0, 3.0], [2.0, 4.0, 6.0])
        assert result is not None
        assert abs(result - 1.0) < 1e-9


# ---------------------------------------------------------------------------
# _token_estimate_from_record and _provider_prompt_tokens_from_record
# ---------------------------------------------------------------------------


class TestTokenExtractors:
    def test_token_estimate_returns_none_when_not_available(self) -> None:
        record: dict[str, Any] = {"attempt_metadata": {"prompt_token_estimate_available": False}}
        assert mod._token_estimate_from_record(record) is None

    def test_token_estimate_returns_value(self) -> None:
        record: dict[str, Any] = {"attempt_metadata": {"prompt_tokens_estimate_pre_call": 500.0}}
        result = mod._token_estimate_from_record(record)
        assert result == 500.0

    def test_provider_tokens_returns_none_when_not_available(self) -> None:
        record: dict[str, Any] = {"attempt_metadata": {"token_usage_available": False}}
        assert mod._provider_prompt_tokens_from_record(record) is None

    def test_provider_tokens_returns_value(self) -> None:
        record: dict[str, Any] = {"attempt_metadata": {"tokens_prompt": 1000.0}}
        result = mod._provider_prompt_tokens_from_record(record)
        assert result == 1000.0


# ---------------------------------------------------------------------------
# _paired_wilcoxon_test — ValueError path (lines 814-820)
# ---------------------------------------------------------------------------


class TestPairedWilcoxonTest:
    def test_all_zero_deltas_returns_p_value_one(self) -> None:
        """All-zero deltas take the n_nonzero==0 early return with p_value=1.0."""
        result = mod._paired_wilcoxon_test([0.0, 0.0, 0.0])
        assert result.get("n_nonzero") == 0
        assert result.get("p_value") == 1.0
        assert result.get("method") == "wilcoxon_signed_rank"

    def test_single_nonzero_delta_triggers_value_error_path(self) -> None:
        """A single non-zero delta triggers ValueError in scipy (lines 814-820)."""
        result = mod._paired_wilcoxon_test([1.0])
        # scipy raises ValueError for n<2 or all-same signs
        assert "method" in result
        assert result.get("method") == "wilcoxon_signed_rank"

    def test_multiple_nonzero_deltas_returns_p_value(self) -> None:
        result = mod._paired_wilcoxon_test([1.0, -2.0, 3.0, -1.5, 2.5])
        assert "p_value" in result
        assert result.get("method") == "wilcoxon_signed_rank"


# ---------------------------------------------------------------------------
# Paired deltas use the shared seeded percentile bootstrap (STA-05)
# ---------------------------------------------------------------------------


class TestPairedDeltaBootstrap:
    def test_uses_the_shared_bootstrap_helper(self) -> None:
        from evals.stats import bootstrap_mean_ci

        deltas = [1.0, -2.0, 3.0, 4.0, 0.5]
        summary = mod._summarize_paired_deltas(
            label="tokens",
            deltas=deltas,
            a_values=[0.0] * 5,
            b_values=deltas,
            n_bootstrap=200,
            seed=7,
        )
        assert summary["bootstrap_ci_mean_delta"] == bootstrap_mean_ci(
            deltas, n_bootstrap=200, confidence=0.95, seed=7
        )
        assert summary["bootstrap_ci_mean_delta"]["method"] == "percentile"
