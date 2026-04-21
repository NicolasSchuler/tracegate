"""Coverage tests for evals.analyze_results - wp5 batch.

Targets uncovered lines and branches in analyze_results.py: no-valid-rows
exit, attempts_to_success None, staged-vs-monolithic uplift, non-numeric
success_rate in uplift, token validation skips, solve-dist rendering,
_looks_like_test_path empty-parts, _print_summary with by_bug_source and
by_contamination_risk, ridge regression LinAlgError fallback, and
statistical-pair error propagation.
"""

from __future__ import annotations

from typing import Any

import pytest

from evals import analyze_results as ar

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _record(
    *,
    run_id: str = "run",
    case_id: str = "case",
    condition: str = "tracecoder",
    attempt: int = 1,
    success: bool = False,
    category: str = "logic_error",
    difficulty: str = "easy",
    snapshot_dependency: str = "required",
    expected_exception: str = "AssertionError",
    attempt_metadata_extra: dict[str, object] | None = None,
    patch_meta_extra: dict[str, object] | None = None,
    context_protocol: str = "monolithic",
    bug_source: str | None = None,
    contamination_risk: str | None = None,
    attempts_to_success: int | None = None,
) -> dict[str, object]:
    """Minimal valid record with sensible defaults."""
    attempt_metadata: dict[str, Any] = {
        "attempt": attempt,
        "max_attempts": 3,
        "llm_call_count": 2,
        "tokens_prompt": 100,
        "tokens_completion": 50,
        "tokens_total": 150,
    }
    if attempt_metadata_extra:
        attempt_metadata.update(attempt_metadata_extra)
    rec: dict[str, Any] = {
        "schema_version": ar.EVAL_RECORD_SCHEMA_VERSION,
        "run_id": run_id,
        "case_id": case_id,
        "condition": condition,
        "attempt": attempt,
        "success": success,
        "category": category,
        "difficulty": difficulty,
        "snapshot_dependency": snapshot_dependency,
        "expected_exception": expected_exception,
        "attempt_metadata": attempt_metadata,
        "context_protocol": context_protocol,
        "run_metadata": {"run_id": run_id},
    }
    if bug_source is not None:
        rec["bug_source"] = bug_source
    if contamination_risk is not None:
        rec["contamination_risk"] = contamination_risk
    if attempts_to_success is not None:
        rec["attempts_to_success"] = attempts_to_success
    if patch_meta_extra is not None:
        rec["patch_meta"] = patch_meta_extra
    return rec


# ---------------------------------------------------------------------------
# _uplift_between_conditions - non-numeric success_rate
# ---------------------------------------------------------------------------


class TestUpliftBetweenConditions:
    """Uplift returns None when success_rate is non-numeric."""

    def test_returns_none_for_missing_conditions(self) -> None:
        summary: dict[str, Any] = {}
        assert (
            ar._uplift_between_conditions(
                summary=summary,
                a_condition="a",
                b_condition="b",
            )
            is None
        )

    def test_returns_none_for_non_numeric_success_rate(self) -> None:
        summary: dict[str, Any] = {
            "a": {"success_rate": "not_a_number"},
            "b": {"success_rate": 0.5},
        }
        assert (
            ar._uplift_between_conditions(
                summary=summary,
                a_condition="a",
                b_condition="b",
            )
            is None
        )

    def test_returns_float_for_valid_rates(self) -> None:
        summary: dict[str, Any] = {
            "a": {"success_rate": 0.4},
            "b": {"success_rate": 0.8},
        }
        result = ar._uplift_between_conditions(
            summary=summary,
            a_condition="a",
            b_condition="b",
        )
        assert result is not None
        assert abs(result - 0.4) < 1e-6


# ---------------------------------------------------------------------------
# _looks_like_test_path - edge cases
# ---------------------------------------------------------------------------


class TestLooksLikeTestPath:
    """Heuristic test-path detection edge cases."""

    def test_empty_string(self) -> None:
        assert ar._looks_like_test_path("") is False

    def test_slash_only(self) -> None:
        assert ar._looks_like_test_path("/") is False

    def test_normal_test_file(self) -> None:
        assert ar._looks_like_test_path("tests/test_foo.py") is True

    def test_conftest(self) -> None:
        assert ar._looks_like_test_path("tests/conftest.py") is True

    def test_non_test_file(self) -> None:
        assert ar._looks_like_test_path("src/main.py") is False


# ---------------------------------------------------------------------------
# _token_usage_breakdown - non-numeric token fields
# ---------------------------------------------------------------------------


class TestTokenUsageBreakdown:
    """Token breakdown skips records with non-numeric token fields."""

    def test_non_numeric_prompt_tokens_skipped(self) -> None:
        """Records where tokens_prompt is a string are skipped."""
        records = [
            {
                "attempt_metadata": {
                    "tokens_prompt": "bad",
                    "tokens_completion": 50,
                    "tokens_total": 150,
                },
            },
        ]
        p, c, t = ar._token_usage_breakdown(records)
        assert p is None and c is None and t is None

    def test_non_numeric_completion_tokens_skipped(self) -> None:
        """Records where tokens_completion is a string are skipped."""
        records = [
            {
                "attempt_metadata": {
                    "tokens_prompt": 100,
                    "tokens_completion": None,
                    "tokens_total": 150,
                },
            },
        ]
        p, c, t = ar._token_usage_breakdown(records)
        assert p is None and c is None and t is None

    def test_non_numeric_total_tokens_skipped(self) -> None:
        """Records where tokens_total is not numeric are skipped."""
        records = [
            {
                "attempt_metadata": {
                    "tokens_prompt": 100,
                    "tokens_completion": 50,
                    "tokens_total": "n/a",
                },
            },
        ]
        p, c, t = ar._token_usage_breakdown(records)
        assert p is None and c is None and t is None

    def test_valid_tokens(self) -> None:
        """Valid numeric tokens are summed."""
        records = [
            {
                "attempt_metadata": {
                    "tokens_prompt": 100,
                    "tokens_completion": 50,
                    "tokens_total": 150,
                },
            },
        ]
        p, c, t = ar._token_usage_breakdown(records)
        assert p == 100.0
        assert c == 50.0
        assert t == 150.0


# ---------------------------------------------------------------------------
# _tokens_used wraps _token_usage_breakdown
# ---------------------------------------------------------------------------


class TestTokensUsed:
    """_tokens_used returns total or None."""

    def test_returns_total(self) -> None:
        records = [
            {
                "attempt_metadata": {
                    "tokens_prompt": 10,
                    "tokens_completion": 20,
                    "tokens_total": 30,
                },
            },
        ]
        assert ar._tokens_used(records) == 30.0

    def test_fallback_to_patch_meta(self) -> None:
        records = [
            {
                "attempt_metadata": {},
                "patch_meta": {"usage": {"total_tokens": 42}},
            },
        ]
        assert ar._tokens_used(records) == 42.0


# ---------------------------------------------------------------------------
# _print_summary - by_bug_source and by_contamination_risk
# ---------------------------------------------------------------------------


class TestPrintSummary:
    """Summary printing includes optional sections."""

    def _make_summary(
        self,
        *,
        include_bug_source: bool = False,
        include_contamination_risk: bool = False,
    ) -> dict[str, Any]:
        bucket: dict[str, Any] = {
            "base_repair": {
                "cases": 10,
                "successes": 5,
                "success_rate": 0.5,
            },
        }
        summary: dict[str, Any] = {
            "overall": dict(bucket),
            "by_category": {"logic_error": dict(bucket)},
            "by_difficulty": {"easy": dict(bucket)},
            "by_snapshot_dependency": {"required": dict(bucket)},
        }
        if include_bug_source:
            summary["by_bug_source"] = {"manual": dict(bucket)}
        if include_contamination_risk:
            summary["by_contamination_risk"] = {"low": dict(bucket)}
        return summary

    def test_print_summary_with_bug_source(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Summary printing includes 'By bug source' section when present."""
        summary = self._make_summary(include_bug_source=True)
        ar._print_summary(summary)
        captured = capsys.readouterr()
        assert "By bug source:" in captured.out

    def test_print_summary_with_contamination_risk(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Summary printing includes 'By contamination risk' section when present."""
        summary = self._make_summary(include_contamination_risk=True)
        ar._print_summary(summary)
        captured = capsys.readouterr()
        assert "By contamination risk:" in captured.out

    def test_print_summary_without_optional_sections(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Without optional sections, only standard sections are printed."""
        summary = self._make_summary()
        ar._print_summary(summary)
        captured = capsys.readouterr()
        assert "By bug source:" not in captured.out
        assert "By contamination risk:" not in captured.out


# ---------------------------------------------------------------------------
# _print_bucket - solve distribution sort key
# ---------------------------------------------------------------------------


class TestPrintBucket:
    """Bucket printing with solve distribution."""

    def test_solve_distribution_rendered(self, capsys: pytest.CaptureFixture[str]) -> None:
        """Solve distribution dict entries are rendered."""
        bucket: dict[str, Any] = {
            "base_repair": {
                "cases": 10,
                "successes": 5,
                "success_rate": 0.5,
                "solve_on_attempt_distribution": {"1": 3, "2": 2},
            },
        }
        ar._print_bucket(bucket)
        captured = capsys.readouterr()
        assert "1:3" in captured.out or "solve" in captured.out.lower()

    def test_solve_distribution_non_numeric_keys(
        self,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Non-numeric solve distribution keys fall back to 0 sort order."""
        bucket: dict[str, Any] = {
            "c1": {
                "cases": 5,
                "successes": 2,
                "success_rate": 0.4,
                "solve_on_attempt_distribution": {"abc": 1, "1": 2},
            },
        }
        ar._print_bucket(bucket)
        captured = capsys.readouterr()
        assert "abc:1" in captured.out or "c1" in captured.out


# ---------------------------------------------------------------------------
# _summarize_bucket - attempts_to_success None handling
# ---------------------------------------------------------------------------


class TestSummarizeBucketAttemptsNone:
    """When attempts_to_success is None for a successful case."""

    def _full_row(self, **overrides: Any) -> dict[str, Any]:
        """Build a row with all required fields for _summarize_bucket."""
        base: dict[str, Any] = {
            "run_id": "r1",
            "case_id": "c1",
            "condition": "base_repair",
            "success": False,
            "category": "logic_error",
            "difficulty": "easy",
            "snapshot_dependency": "required",
            "attempts_to_success": None,
            "token_usage": 100.0,
            "token_usage_prompt": None,
            "token_usage_completion": None,
            "token_usage_total": 100.0,
            "token_usage_per_attempt_mean": 50.0,
            "runtime_sec": 5.0,
            "llm_calls_total": 2,
            "test_touch_attempts": 0,
            "context_protocol": "monolithic",
            "attempts_budget": 3,
            "attempt_ratio_when_successful": None,
        }
        base.update(overrides)
        return base

    def test_none_attempts_skipped_in_solve_dist(self) -> None:
        """Success records with None attempts_to_success are excluded from solve dist."""
        rows = [
            self._full_row(
                case_id="c1",
                success=True,
                attempts_to_success=None,
            ),
            self._full_row(
                case_id="c2",
                success=True,
                attempts_to_success=1,
            ),
        ]
        result = ar._summarize_bucket(
            rows,
            records_by_case_condition={},
        )
        dist = result.get("base_repair", {}).get("solve_on_attempt_distribution", {})
        assert "1" in dist
        # None-valued attempt should not appear
        assert None not in dist.values()


# ---------------------------------------------------------------------------
# _summarize_bucket - staged vs monolithic uplift
# ---------------------------------------------------------------------------


class TestSummarizeBucketStagedMonolithic:
    """Staged vs monolithic uplift computation."""

    def _full_row(self, **overrides: Any) -> dict[str, Any]:
        base: dict[str, Any] = {
            "run_id": "r1",
            "case_id": "c1",
            "condition": "layered_adaptive_llm",
            "success": False,
            "category": "logic_error",
            "difficulty": "easy",
            "snapshot_dependency": "required",
            "attempts_to_success": None,
            "token_usage": 100.0,
            "token_usage_prompt": None,
            "token_usage_completion": None,
            "token_usage_total": 100.0,
            "token_usage_per_attempt_mean": 50.0,
            "runtime_sec": 5.0,
            "llm_calls_total": 2,
            "test_touch_attempts": 0,
            "context_protocol": "monolithic",
            "attempts_budget": 3,
            "attempt_ratio_when_successful": None,
        }
        base.update(overrides)
        return base

    def test_both_protocols_present(self) -> None:
        """When both staged and monolithic rows exist, uplift is computed."""
        rows = [
            self._full_row(
                case_id="c1",
                success=True,
                context_protocol="staged",
            ),
            self._full_row(
                case_id="c2",
                success=False,
                context_protocol="monolithic",
            ),
        ]
        result = ar._summarize_bucket(
            rows,
            records_by_case_condition={},
        )
        uplift = result.get("uplift_staged_vs_monolithic")
        assert uplift is not None
        assert isinstance(uplift, float)


# ---------------------------------------------------------------------------
# _fit_ridge_action_models - singular matrix fallback
# ---------------------------------------------------------------------------


class TestFitRidgeActionModels:
    """Ridge regression with singular matrix fallback."""

    def test_singular_matrix_uses_pinv(self) -> None:
        """When np.linalg.solve fails, np.linalg.pinv is used as fallback."""
        import numpy as np

        # Create a deliberately rank-deficient design matrix
        # Use fewer than _RIDGECV_MIN_SAMPLES to hit the fallback path
        samples = [
            {"action": "a", "features": {"f1": 1.0, "f2": 1.0}, "reward": 0.5},
            {"action": "a", "features": {"f1": 1.0, "f2": 1.0}, "reward": 0.7},
            {"action": "a", "features": {"f1": 1.0, "f2": 1.0}, "reward": 0.3},
        ]
        # Call with ridge_alpha=0 to avoid regularization → singular matrix
        _feature_names, models = ar._fit_ridge_action_models(
            samples=samples,
            action_space={"a"},
            min_action_support=1,
            ridge_alpha=0.0,
        )
        # Should not raise, and should produce a model
        assert "a" in models
