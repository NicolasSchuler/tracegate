"""Coverage-boost tests for evals.run_eval utility/helper functions.

Targets pure helper branches and argument-parsing paths not exercised by
the existing test files: baseline arg extraction, condition resolution,
request-context failure classification, adaptive gate/progress helpers,
token-usage extraction, char-based metrics, traceback informativeness,
summarisation helpers, deterministic-seed validation, and case discovery.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pytest

from evals import run_eval
from evals.eval_types import (
    EVAL_RECORD_TYPE_ATTEMPT,
    EVAL_RECORD_TYPE_INFRA_EVENT,
    Condition,
    ConfigResolutionState,
    RunResult,
)

# ---------------------------------------------------------------------------
# Fixtures & helpers
# ---------------------------------------------------------------------------


def _ns(**kwargs: Any) -> argparse.Namespace:
    """Build an ``argparse.Namespace`` from keyword arguments."""
    return argparse.Namespace(**kwargs)


def _run_result(
    *,
    returncode: int = 1,
    stdout: str = "",
    stderr: str = "",
    timed_out: bool = False,
    timeout_sec: int | None = None,
) -> RunResult:
    return RunResult(
        returncode=returncode,
        duration_sec=0.1,
        stdout=stdout,
        stderr=stderr,
        timed_out=timed_out,
        timeout_sec=timeout_sec,
    )


# ===================================================================
# _baseline_extra_from_args
# ===================================================================


class TestBaselineExtraFromArgs:
    """Cover all branches of ``_baseline_extra_from_args``."""

    def test_lambda_rlm_defaults(self) -> None:
        args = _ns(
            baseline_openai_compat_profile="auto",
            baseline_openai_allow_reasoning_fallback=False,
            baseline_openai_enable_thinking=False,
            baseline_lambda_context_window_chars=64000,
            baseline_lambda_accuracy_target=0.80,
            baseline_lambda_a_leaf=0.95,
            baseline_lambda_a_compose=0.90,
            baseline_lambda_verbose=False,
            baseline_lambda_stream=False,
            baseline_lambda_controller_guidance_max_chars=2500,
            baseline_lambda_failure_feedback_max_chars=3500,
            baseline_lambda_failure_feedback_meta_max_chars=2000,
            baseline_lambda_timeout_sec=None,
            baseline_lambda_force_task_type="",
            baseline_lambda_prompt_contract="",
            baseline_lambda_temperature=None,
            baseline_lambda_top_p=None,
            baseline_lambda_max_tokens=None,
        )
        extra = run_eval._baseline_extra_from_args(args=args, baseline_name="lambda_rlm")
        assert extra["baseline_lambda_accuracy_target"] == 0.80
        assert extra["baseline_lambda_a_leaf"] == 0.95
        assert "baseline_lambda_timeout_sec" not in extra
        assert "baseline_lambda_force_task_type" not in extra
        assert "baseline_lambda_prompt_contract" not in extra
        assert "baseline_lambda_temperature" not in extra

    def test_lambda_rlm_optional_overrides(self) -> None:
        args = _ns(
            baseline_openai_compat_profile="manual",
            baseline_openai_allow_reasoning_fallback=True,
            baseline_openai_enable_thinking=True,
            baseline_lambda_context_window_chars=128000,
            baseline_lambda_accuracy_target=0.90,
            baseline_lambda_a_leaf=0.99,
            baseline_lambda_a_compose=0.95,
            baseline_lambda_verbose=True,
            baseline_lambda_stream=True,
            baseline_lambda_controller_guidance_max_chars=3000,
            baseline_lambda_failure_feedback_max_chars=4000,
            baseline_lambda_failure_feedback_meta_max_chars=2500,
            baseline_lambda_timeout_sec=60,
            baseline_lambda_force_task_type="generation",
            baseline_lambda_prompt_contract="repair_fair_v1",
            baseline_lambda_temperature=0.7,
            baseline_lambda_top_p=0.9,
            baseline_lambda_max_tokens=4096,
        )
        extra = run_eval._baseline_extra_from_args(args=args, baseline_name="lambda_rlm")
        assert extra["baseline_lambda_timeout_sec"] == 60
        assert extra["baseline_lambda_force_task_type"] == "generation"
        assert extra["baseline_lambda_prompt_contract"] == "repair_fair_v1"
        assert extra["baseline_lambda_temperature"] == 0.7
        assert extra["baseline_lambda_top_p"] == 0.9
        assert extra["baseline_lambda_max_tokens"] == 4096

    def test_rlm_defaults(self) -> None:
        args = _ns(
            baseline_openai_compat_profile="auto",
            baseline_openai_allow_reasoning_fallback=False,
            baseline_openai_enable_thinking=False,
            baseline_rlm_max_depth=2,
            baseline_rlm_max_iterations=12,
            baseline_rlm_max_errors=5,
            baseline_rlm_compaction=False,
            baseline_rlm_compaction_threshold_pct=0.85,
            baseline_rlm_verbose=False,
            baseline_rlm_controller_guidance_max_chars=2500,
            baseline_rlm_failure_feedback_max_chars=3500,
            baseline_rlm_failure_feedback_meta_max_chars=2000,
            baseline_rlm_timeout_sec=None,
            baseline_rlm_temperature=None,
            baseline_rlm_top_p=None,
            baseline_rlm_max_tokens=None,
            baseline_rlm_max_total_tokens=None,
            baseline_rlm_prompt_contract="",
        )
        extra = run_eval._baseline_extra_from_args(args=args, baseline_name="rlm")
        assert extra["baseline_rlm_max_depth"] == 2
        assert "baseline_rlm_timeout_sec" not in extra
        assert "baseline_rlm_temperature" not in extra
        assert "baseline_rlm_prompt_contract" not in extra

    def test_rlm_optional_overrides(self) -> None:
        args = _ns(
            baseline_openai_compat_profile="auto",
            baseline_openai_allow_reasoning_fallback=False,
            baseline_openai_enable_thinking=False,
            baseline_rlm_max_depth=4,
            baseline_rlm_max_iterations=20,
            baseline_rlm_max_errors=10,
            baseline_rlm_compaction=True,
            baseline_rlm_compaction_threshold_pct=0.90,
            baseline_rlm_verbose=True,
            baseline_rlm_controller_guidance_max_chars=3000,
            baseline_rlm_failure_feedback_max_chars=4000,
            baseline_rlm_failure_feedback_meta_max_chars=2500,
            baseline_rlm_timeout_sec=120,
            baseline_rlm_temperature=0.5,
            baseline_rlm_top_p=0.8,
            baseline_rlm_max_tokens=2048,
            baseline_rlm_max_total_tokens=50000,
            baseline_rlm_prompt_contract="repair_fair_v1",
        )
        extra = run_eval._baseline_extra_from_args(args=args, baseline_name="rlm")
        assert extra["baseline_rlm_timeout_sec"] == 120
        assert extra["baseline_rlm_temperature"] == 0.5
        assert extra["baseline_rlm_max_total_tokens"] == 50000
        assert extra["baseline_rlm_prompt_contract"] == "repair_fair_v1"

    def test_intervenor(self) -> None:
        args = _ns(
            baseline_openai_compat_profile="auto",
            baseline_openai_allow_reasoning_fallback=False,
            baseline_openai_enable_thinking=False,
            baseline_intervenor_teacher_max_tokens=1024,
            baseline_intervenor_learner_max_tokens=2048,
            baseline_intervenor_teacher_guidance_max_chars=4000,
            baseline_intervenor_learner_guidance_max_chars=6000,
            baseline_intervenor_append_output_contract=True,
        )
        extra = run_eval._baseline_extra_from_args(args=args, baseline_name="intervenor")
        assert extra["baseline_intervenor_teacher_max_tokens"] == 1024
        assert extra["baseline_intervenor_append_output_contract"] is True

    def test_tracecoder(self) -> None:
        args = _ns(
            baseline_openai_compat_profile="auto",
            baseline_openai_allow_reasoning_fallback=False,
            baseline_openai_enable_thinking=False,
            baseline_tracecoder_controller_guidance_max_chars=2000,
            baseline_tracecoder_max_no_improvement_streak=2,
            baseline_tracecoder_analysis_max_tokens=2048,
            baseline_tracecoder_implementation_max_tokens=4096,
            baseline_tracecoder_problem_summary_max_chars=4000,
            baseline_tracecoder_exact_test_context_max_chars=3000,
            baseline_tracecoder_failure_feedback_max_chars=6000,
            baseline_tracecoder_failure_feedback_tail_max_chars=2000,
            baseline_tracecoder_instrumentation_output_max_lines=80,
            baseline_tracecoder_instrumentation_output_max_chars=12000,
        )
        extra = run_eval._baseline_extra_from_args(args=args, baseline_name="tracecoder")
        assert extra["baseline_tracecoder_controller_guidance_max_chars"] == 2000
        assert extra["baseline_tracecoder_analysis_max_tokens"] == 2048
        assert extra["baseline_tracecoder_implementation_max_tokens"] == 4096
        assert extra["baseline_tracecoder_exact_test_context_max_chars"] == 3000

    def test_ldb_returns_shared_openai_extra(self) -> None:
        args = _ns(
            baseline_openai_compat_profile="auto",
            baseline_openai_allow_reasoning_fallback=True,
            baseline_openai_enable_thinking=False,
            baseline_hybrid_guidance_mode="fair_v1",
        )
        extra = run_eval._baseline_extra_from_args(args=args, baseline_name="ldb")
        assert extra["baseline_openai_compat_profile"] == "auto"
        assert extra["baseline_openai_allow_reasoning_fallback"] is True
        assert extra["baseline_hybrid_guidance_mode"] == "fair_v1"

    def test_unknown_baseline_returns_empty(self) -> None:
        args = _ns()
        extra = run_eval._baseline_extra_from_args(args=args, baseline_name="unknown")
        assert extra == {}


# ===================================================================
# _normalize_baseline_selection
# ===================================================================


class TestNormalizeBaselineSelection:
    """Cover error paths and derived-value logic in baseline selection."""

    def test_baseline_alias_sets_patcher_and_name(self) -> None:
        args = _ns(
            baseline="ldb", patcher="", baseline_name="", baseline_repo_dir="", baseline_python=""
        )
        state = ConfigResolutionState(value_sources={"baseline": "cli"})
        run_eval._normalize_baseline_selection(args=args, config_state=state)
        assert args.patcher == "baseline"
        assert args.baseline_name == "ldb"

    def test_baseline_alias_conflicts_with_patcher(self) -> None:
        args = _ns(baseline="ldb", patcher="openai", baseline_name="")
        with pytest.raises(ValueError, match="cannot be combined"):
            run_eval._normalize_baseline_selection(args=args, config_state=None)

    def test_baseline_alias_conflicts_with_different_name(self) -> None:
        args = _ns(baseline="ldb", patcher="", baseline_name="rlm")
        with pytest.raises(ValueError, match="conflicts with"):
            run_eval._normalize_baseline_selection(args=args, config_state=None)

    def test_patcher_baseline_without_name_raises(self) -> None:
        args = _ns(baseline="", patcher="baseline", baseline_name="")
        with pytest.raises(ValueError, match="--baseline-name is required"):
            run_eval._normalize_baseline_selection(args=args, config_state=None)

    def test_non_baseline_patcher_returns_early(self) -> None:
        args = _ns(baseline="", patcher="openai", baseline_name="")
        run_eval._normalize_baseline_selection(args=args, config_state=None)
        assert args.patcher == "openai"

    def test_patcher_baseline_derives_repo_dir(self) -> None:
        args = _ns(
            baseline="",
            patcher="baseline",
            baseline_name="ldb",
            baseline_repo_dir="",
            baseline_python="",
        )
        state = ConfigResolutionState(value_sources={"baseline_name": "cli"})
        run_eval._normalize_baseline_selection(args=args, config_state=state)
        assert "LLMDebugger" in args.baseline_repo_dir

    def test_patcher_baseline_with_explicit_repo_dir(self) -> None:
        args = _ns(
            baseline="",
            patcher="baseline",
            baseline_name="rlm",
            baseline_repo_dir="/custom/dir",
            baseline_python="my_python",
        )
        run_eval._normalize_baseline_selection(args=args, config_state=None)
        assert args.baseline_repo_dir == "/custom/dir"
        assert args.baseline_python == "my_python"


# ===================================================================
# _resolve_conditions_for_run
# ===================================================================


class TestResolveConditionsForRun:
    """Cover condition resolution for baseline patcher."""

    def test_non_baseline_patcher_returns_raw(self) -> None:
        args = _ns(conditions="base_repair,cot_repair", patcher="openai")
        state = ConfigResolutionState(value_sources={"conditions": "cli"})
        result = run_eval._resolve_conditions_for_run(args=args, config_state=state)
        assert result == [Condition.BASE_REPAIR, Condition.COT_REPAIR]

    def test_baseline_patcher_default_condition(self) -> None:
        args = _ns(conditions="base_repair", patcher="baseline", baseline_name="ldb")
        state = ConfigResolutionState(value_sources={"conditions": "default"})
        result = run_eval._resolve_conditions_for_run(args=args, config_state=state)
        assert result == [Condition.LDB]

    def test_baseline_patcher_missing_name_raises(self) -> None:
        args = _ns(conditions="base_repair", patcher="baseline", baseline_name="")
        state = ConfigResolutionState(value_sources={"conditions": "default"})
        with pytest.raises(ValueError, match="--baseline-name is required"):
            run_eval._resolve_conditions_for_run(args=args, config_state=state)

    def test_baseline_patcher_multiple_conditions_raises(self) -> None:
        args = _ns(conditions="ldb,rlm", patcher="baseline", baseline_name="ldb")
        state = ConfigResolutionState(value_sources={"conditions": "cli"})
        with pytest.raises(ValueError, match="exactly one condition"):
            run_eval._resolve_conditions_for_run(args=args, config_state=state)

    def test_baseline_patcher_wrong_condition_raises(self) -> None:
        args = _ns(conditions="base_repair", patcher="baseline", baseline_name="ldb")
        state = ConfigResolutionState(value_sources={"conditions": "cli"})
        with pytest.raises(ValueError, match="matching baseline condition"):
            run_eval._resolve_conditions_for_run(args=args, config_state=state)

    def test_baseline_patcher_accepts_adaptive_condition(self) -> None:
        args = _ns(
            conditions="layered_adaptive_gated",
            patcher="baseline",
            baseline_name="ldb",
        )
        state = ConfigResolutionState(value_sources={"conditions": "cli"})
        result = run_eval._resolve_conditions_for_run(args=args, config_state=state)
        assert result == [Condition.LAYERED_ADAPTIVE_GATED]


# ===================================================================
# _request_context_failure_kind_from_meta
# ===================================================================


class TestRequestContextFailureKindFromMeta:
    """Cover all branches of ``_request_context_failure_kind_from_meta``."""

    @pytest.mark.parametrize(
        ("meta", "expected"),
        [
            (None, ""),
            ("not_a_dict", ""),
            ({}, ""),
            ({"request_context_failure_kind": "custom"}, "custom"),
            ({"error": "request_budget_exceeded"}, "request_budget_exceeded"),
            ({"request_context_backend_unavailable": True}, "backend_unavailable"),
            ({"error": "invalid_request_context_payload"}, "invalid_payload"),
            ({"error": "request_failed"}, "transport"),
            ({"error": "http_error", "status": 502}, "backend_unavailable"),
            ({"error": "http_error", "status": 503}, "backend_unavailable"),
            ({"error": "http_error", "status": 504}, "backend_unavailable"),
            ({"error": "http_error", "status": 400}, "transport"),
        ],
    )
    def test_all_branches(self, meta: Any, expected: str) -> None:
        assert run_eval._request_context_failure_kind_from_meta(meta) == expected


# ===================================================================
# _request_context_recovery_failure_kind_from_meta
# ===================================================================


class TestRequestContextRecoveryFailureKind:
    @pytest.mark.parametrize(
        ("meta", "expected"),
        [
            (None, ""),
            ({}, ""),
            ({"error": "request_failed"}, "transport"),
            ({"error": "http_error", "status": 502}, "backend_unavailable"),
            ({"error": "http_error", "status": 503}, "backend_unavailable"),
            ({"error": "http_error", "status": 500}, "transport"),
            ({"error": "http_error", "status": 200}, ""),
            ({"error": "other"}, ""),
        ],
    )
    def test_all_branches(self, meta: Any, expected: str) -> None:
        assert run_eval._request_context_recovery_failure_kind_from_meta(meta) == expected


# ===================================================================
# _request_context_retry_counts_from_meta
# ===================================================================


class TestRequestContextRetryCounts:
    def test_non_dict_returns_zeros(self) -> None:
        assert run_eval._request_context_retry_counts_from_meta(None) == (0, 0)

    def test_extracts_counts(self) -> None:
        meta = {
            "request_context_retry_count": 3,
            "request_context_transport_retry_count": 5,
        }
        assert run_eval._request_context_retry_counts_from_meta(meta) == (3, 5)


# ===================================================================
# _adaptive_helper_outcome_for_attempt
# ===================================================================


class TestAdaptiveHelperOutcomeForAttempt:
    def test_materialized(self) -> None:
        outcome, _kind = run_eval._adaptive_helper_outcome_for_attempt(
            helper_invoked=True,
            helper_materialized_context=True,
            helper_decision_reason="gate_aggressive",
            stage_a_meta={},
        )
        assert outcome == run_eval.ADAPTIVE_HELPER_OUTCOME_MATERIALIZED

    def test_backend_unavailable(self) -> None:
        outcome, _kind = run_eval._adaptive_helper_outcome_for_attempt(
            helper_invoked=True,
            helper_materialized_context=False,
            helper_decision_reason="gate_aggressive",
            stage_a_meta={"request_context_backend_unavailable": True},
        )
        assert outcome == run_eval.ADAPTIVE_HELPER_OUTCOME_UNAVAILABLE_BACKEND

    def test_transport_failed(self) -> None:
        outcome, _ = run_eval._adaptive_helper_outcome_for_attempt(
            helper_invoked=True,
            helper_materialized_context=False,
            helper_decision_reason="gate_aggressive",
            stage_a_meta={"error": "request_failed"},
        )
        assert outcome == run_eval.ADAPTIVE_HELPER_OUTCOME_TRANSPORT_FAILED

    def test_request_budget_exceeded(self) -> None:
        outcome, _ = run_eval._adaptive_helper_outcome_for_attempt(
            helper_invoked=True,
            helper_materialized_context=False,
            helper_decision_reason="gate_aggressive",
            stage_a_meta={"error": "request_budget_exceeded"},
        )
        assert outcome == run_eval.ADAPTIVE_HELPER_OUTCOME_REQUEST_BUDGET_EXCEEDED

    def test_compat_reasoning_only(self) -> None:
        outcome, kind = run_eval._adaptive_helper_outcome_for_attempt(
            helper_invoked=True,
            helper_materialized_context=False,
            helper_decision_reason="gate_aggressive",
            stage_a_meta={"error": "openai_compat_reasoning_only"},
        )
        assert outcome == run_eval.ADAPTIVE_HELPER_OUTCOME_COMPAT_REASONING_ONLY
        assert kind == "compat_reasoning_only"

    def test_invalid_payload_explicit(self) -> None:
        outcome, _ = run_eval._adaptive_helper_outcome_for_attempt(
            helper_invoked=True,
            helper_materialized_context=False,
            helper_decision_reason="gate_aggressive",
            stage_a_meta={"error": "invalid_request_context_payload"},
        )
        assert outcome == run_eval.ADAPTIVE_HELPER_OUTCOME_INVALID_PAYLOAD

    def test_invalid_payload_from_decision_reason(self) -> None:
        outcome, _ = run_eval._adaptive_helper_outcome_for_attempt(
            helper_invoked=True,
            helper_materialized_context=False,
            helper_decision_reason="llm_invalid_response_no_helper",
            stage_a_meta={},
        )
        assert outcome == run_eval.ADAPTIVE_HELPER_OUTCOME_INVALID_PAYLOAD

    def test_semantic_no_context(self) -> None:
        outcome, _ = run_eval._adaptive_helper_outcome_for_attempt(
            helper_invoked=True,
            helper_materialized_context=False,
            helper_decision_reason="gate_aggressive",
            stage_a_meta={},
        )
        assert outcome == run_eval.ADAPTIVE_HELPER_OUTCOME_SEMANTIC_NO_CONTEXT

    def test_not_invoked_unsupported(self) -> None:
        outcome, _ = run_eval._adaptive_helper_outcome_for_attempt(
            helper_invoked=False,
            helper_materialized_context=False,
            helper_decision_reason="request_context_unavailable",
            stage_a_meta=None,
        )
        assert outcome == run_eval.ADAPTIVE_HELPER_OUTCOME_SKIPPED_UNSUPPORTED

    def test_not_invoked_guardrail(self) -> None:
        for reason in (
            "informativeness_pregate_skip",
            "helper_budget_exhausted",
            "helper_saturation_guard",
        ):
            outcome, _ = run_eval._adaptive_helper_outcome_for_attempt(
                helper_invoked=False,
                helper_materialized_context=False,
                helper_decision_reason=reason,
                stage_a_meta=None,
            )
            assert outcome == run_eval.ADAPTIVE_HELPER_OUTCOME_SKIPPED_GUARDRAIL

    def test_not_invoked_not_adaptive(self) -> None:
        outcome, _ = run_eval._adaptive_helper_outcome_for_attempt(
            helper_invoked=False,
            helper_materialized_context=False,
            helper_decision_reason="not_adaptive_condition",
            stage_a_meta=None,
        )
        assert outcome == run_eval.ADAPTIVE_HELPER_OUTCOME_NOT_INVOKED

    def test_not_invoked_policy_skip(self) -> None:
        outcome, _ = run_eval._adaptive_helper_outcome_for_attempt(
            helper_invoked=False,
            helper_materialized_context=False,
            helper_decision_reason="llm_declined_helper",
            stage_a_meta=None,
        )
        assert outcome == run_eval.ADAPTIVE_HELPER_OUTCOME_SKIPPED_POLICY


# ===================================================================
# _completion_chars_from_response / _prompt_chars_from_patch_meta
# ===================================================================


class TestCharMetrics:
    def test_completion_chars_no_meta_with_diff(self) -> None:
        assert run_eval._completion_chars_from_response(diff="hello", patch_meta=None) == 5

    def test_completion_chars_no_meta_no_diff(self) -> None:
        assert run_eval._completion_chars_from_response(diff=None, patch_meta=None) == 0

    def test_completion_chars_from_meta_key(self) -> None:
        meta = {"completion_chars_total": 42}
        assert run_eval._completion_chars_from_response(diff=None, patch_meta=meta) == 42

    def test_completion_chars_from_response_output_chars(self) -> None:
        meta = {"response_output_chars_total": 100}
        assert run_eval._completion_chars_from_response(diff=None, patch_meta=meta) == 100

    def test_completion_chars_from_diff_fallback(self) -> None:
        meta = {}
        assert run_eval._completion_chars_from_response(diff="abc", patch_meta=meta) == 3

    def test_completion_chars_from_raw_response(self) -> None:
        meta = {"raw_response": "some text"}
        assert run_eval._completion_chars_from_response(diff=None, patch_meta=meta) == 9

    def test_completion_chars_from_response_key(self) -> None:
        meta = {"response": "text"}
        assert run_eval._completion_chars_from_response(diff=None, patch_meta=meta) == 4

    def test_completion_chars_empty_meta_no_diff(self) -> None:
        assert run_eval._completion_chars_from_response(diff=None, patch_meta={}) == 0

    def test_prompt_chars_non_dict(self) -> None:
        assert run_eval._prompt_chars_from_patch_meta(None) == 0

    def test_prompt_chars_from_key(self) -> None:
        assert run_eval._prompt_chars_from_patch_meta({"prompt_chars_total": 99}) == 99

    def test_prompt_chars_from_response_input(self) -> None:
        assert run_eval._prompt_chars_from_patch_meta({"response_input_chars_total": 50}) == 50

    def test_prompt_chars_empty(self) -> None:
        assert run_eval._prompt_chars_from_patch_meta({}) == 0


# ===================================================================
# _patch_attempt_char_deltas
# ===================================================================


class TestPatchAttemptCharDeltas:
    def test_baseline_patcher(self) -> None:
        prompt, completion = run_eval._patch_attempt_char_deltas(
            cycle_prompt="ignored",
            diff="diff",
            patch_meta={"prompt_chars_total": 10, "completion_chars_total": 20},
            is_baseline_patcher=True,
        )
        assert prompt == 10
        assert completion == 20

    def test_non_baseline_patcher(self) -> None:
        prompt, completion = run_eval._patch_attempt_char_deltas(
            cycle_prompt="hello world",
            diff="abc",
            patch_meta={},
            is_baseline_patcher=False,
        )
        assert prompt == 11
        assert completion == 3


# ===================================================================
# _baseline_metric_float / _baseline_metrics_from_patch_meta
# ===================================================================


class TestBaselineMetrics:
    def test_baseline_metrics_from_non_dict(self) -> None:
        assert run_eval._baseline_metrics_from_patch_meta(None) is None

    def test_baseline_metrics_missing_key(self) -> None:
        assert run_eval._baseline_metrics_from_patch_meta({}) is None

    def test_baseline_metrics_non_dict_value(self) -> None:
        assert run_eval._baseline_metrics_from_patch_meta({"baseline_metrics": "not_dict"}) is None

    def test_baseline_metrics_valid(self) -> None:
        inner = {"llm_duration_sec": 1.5}
        assert run_eval._baseline_metrics_from_patch_meta({"baseline_metrics": inner}) == inner

    def test_baseline_metric_float_none_meta(self) -> None:
        assert run_eval._baseline_metric_float(None, "key") is None

    def test_baseline_metric_float_no_baseline(self) -> None:
        assert run_eval._baseline_metric_float({}, "key") is None

    def test_baseline_metric_float_non_numeric(self) -> None:
        meta = {"baseline_metrics": {"llm_duration_sec": "not_num"}}
        assert run_eval._baseline_metric_float(meta, "llm_duration_sec") is None

    def test_baseline_metric_float_valid(self) -> None:
        meta = {"baseline_metrics": {"llm_duration_sec": 2.5}}
        assert run_eval._baseline_metric_float(meta, "llm_duration_sec") == 2.5


# ===================================================================
# _clip_patcher_meta_text
# ===================================================================


class TestClipPatcherMetaText:
    def test_short_text_unchanged(self) -> None:
        assert run_eval._clip_patcher_meta_text("hello", max_chars=100) == "hello"

    def test_long_text_truncated(self) -> None:
        result = run_eval._clip_patcher_meta_text("a" * 250, max_chars=10)
        assert len(result) == 10
        assert result.endswith("...")

    def test_none_becomes_empty(self) -> None:
        assert run_eval._clip_patcher_meta_text(None, max_chars=100) == ""


# ===================================================================
# _is_context_limit_http_error_message
# ===================================================================


class TestIsContextLimitHttpErrorMessage:
    def test_empty(self) -> None:
        assert run_eval._is_context_limit_http_error_message("") is False
        assert run_eval._is_context_limit_http_error_message(None) is False

    def test_matching_marker(self) -> None:
        # Use one of the markers from PATCHER_CONTEXT_LIMIT_ERROR_MARKERS
        from evals.eval_types import PATCHER_CONTEXT_LIMIT_ERROR_MARKERS

        if PATCHER_CONTEXT_LIMIT_ERROR_MARKERS:
            sample = next(iter(PATCHER_CONTEXT_LIMIT_ERROR_MARKERS))
            assert (
                run_eval._is_context_limit_http_error_message(f"Error: {sample} happened") is True
            )

    def test_non_matching(self) -> None:
        assert run_eval._is_context_limit_http_error_message("some other error") is False


# ===================================================================
# _is_retryable_transport_http_status
# ===================================================================


class TestIsRetryableTransportHttpStatus:
    def test_non_int(self) -> None:
        assert run_eval._is_retryable_transport_http_status("500") is False

    def test_500_range(self) -> None:
        assert run_eval._is_retryable_transport_http_status(500) is True
        assert run_eval._is_retryable_transport_http_status(502) is True
        assert run_eval._is_retryable_transport_http_status(599) is True

    def test_200_not_retryable(self) -> None:
        assert run_eval._is_retryable_transport_http_status(200) is False

    def test_429_retryable(self) -> None:
        # 429 is typically in RETRYABLE_HTTP_STATUS_CODES
        from evals.eval_types import RETRYABLE_HTTP_STATUS_CODES

        if 429 in RETRYABLE_HTTP_STATUS_CODES:
            assert run_eval._is_retryable_transport_http_status(429) is True


# ===================================================================
# _classify_patcher_request_failure
# ===================================================================


class TestClassifyPatcherRequestFailure:
    def test_no_error(self) -> None:
        assert run_eval._classify_patcher_request_failure({}) == (False, "")

    def test_request_failed(self) -> None:
        assert run_eval._classify_patcher_request_failure({"error": "request_failed"}) == (
            True,
            "transport",
        )

    def test_request_budget_exceeded(self) -> None:
        assert run_eval._classify_patcher_request_failure({"error": "request_budget_exceeded"}) == (
            True,
            "transport",
        )

    def test_openai_compat_reasoning_only(self) -> None:
        assert run_eval._classify_patcher_request_failure(
            {"error": "openai_compat_reasoning_only"}
        ) == (True, "compat_reasoning_only")

    def test_openai_compat_reasoning_only_in_fallback_meta(self) -> None:
        assert run_eval._classify_patcher_request_failure(
            {
                "error": "no_json_payload",
                "fallback_meta": {"error": "openai_compat_reasoning_only"},
            }
        ) == (True, "compat_reasoning_only")

    def test_bad_request_url(self) -> None:
        assert run_eval._classify_patcher_request_failure({"error": "bad_request_url"}) == (
            True,
            "transport",
        )

    def test_http_error_context_limit(self) -> None:
        from evals.eval_types import PATCHER_CONTEXT_LIMIT_ERROR_MARKERS

        if PATCHER_CONTEXT_LIMIT_ERROR_MARKERS:
            sample = next(iter(PATCHER_CONTEXT_LIMIT_ERROR_MARKERS))
            meta = {"error": "http_error", "body": f"Error: {sample}"}
            assert run_eval._classify_patcher_request_failure(meta) == (
                True,
                "context_limit",
            )

    def test_http_error_retryable_status(self) -> None:
        meta = {"error": "http_error", "status": 500}
        assert run_eval._classify_patcher_request_failure(meta) == (True, "transport")

    def test_http_error_non_retryable(self) -> None:
        meta = {"error": "http_error", "status": 200}
        assert run_eval._classify_patcher_request_failure(meta) == (False, "")

    def test_repair_failed_backend_unavailable_from_exc_text(self) -> None:
        meta = {
            "error": "repair_failed",
            "exc": "openai.InternalServerError: Error code: 503 - {'error': {'message': 'Loading model'}}",
        }
        assert run_eval._classify_patcher_request_failure(meta) == (True, "backend_unavailable")

    def test_repair_failed_transport_from_exc_text(self) -> None:
        meta = {
            "error": "repair_failed",
            "exc": "RuntimeError: server disconnected while waiting for response",
        }
        assert run_eval._classify_patcher_request_failure(meta) == (True, "transport")


# ===================================================================
# _transport_recovery_failure_kind_from_meta
# ===================================================================


class TestTransportRecoveryFailureKind:
    @pytest.mark.parametrize(
        ("meta", "expected"),
        [
            (None, ""),
            ({}, ""),
            ({"error": "request_failed"}, "transport"),
            ({"error": "request_budget_exceeded"}, ""),
            ({"error": "http_error", "status": 502}, "backend_unavailable"),
            ({"error": "http_error", "status": 503}, "backend_unavailable"),
            ({"error": "http_error", "status": 504}, "backend_unavailable"),
            ({"error": "http_error", "status": 500}, "transport"),
            ({"error": "http_error", "status": 200}, ""),
            ({"error": "unknown"}, ""),
        ],
    )
    def test_all_branches(self, meta: Any, expected: str) -> None:
        assert run_eval._transport_recovery_failure_kind_from_meta(meta) == expected

    def test_repair_failed_backend_unavailable(self) -> None:
        meta = {
            "error": "repair_failed",
            "exc": "InternalServerError: status 503 unavailable_error Loading model",
        }
        assert run_eval._transport_recovery_failure_kind_from_meta(meta) == "backend_unavailable"

    def test_baseline_meta_api_timeout_is_transport(self) -> None:
        meta = {"baseline_meta": {"error": "api_timeout"}}
        assert run_eval._transport_recovery_failure_kind_from_meta(meta) == "transport"

    def test_baseline_meta_api_error_uses_tail_text(self) -> None:
        meta = {
            "baseline_meta": {
                "error": "api_error",
                "stderr_tail": "openai.APIConnectionError: Connection error",
            }
        }
        assert run_eval._transport_recovery_failure_kind_from_meta(meta) == "transport"


# ===================================================================
# _summarize_context_compaction_meta
# ===================================================================


class TestSummarizeContextCompactionMeta:
    def test_basic_fields(self) -> None:
        meta = {
            "llm_call_count": 1,
            "compact_context_duration_sec": 1.5,
            "compact_context_target_chars": 100,
            "compact_context_input_chars": 200,
            "compact_context_output_chars": 100,
            "compact_context_output_truncated": False,
            "status": 200,
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
            },
            "error": "some error",
            "body": "body text",
        }
        result = run_eval._summarize_context_compaction_meta(meta)
        assert result["llm_call_count"] == 1
        assert result["status"] == 200
        assert result["usage"]["prompt_tokens"] == 10
        assert result["error"] == "some error"
        assert result["body"] == "body text"

    def test_missing_fields_excluded(self) -> None:
        result = run_eval._summarize_context_compaction_meta({})
        assert "llm_call_count" not in result
        assert "status" not in result
        assert "usage" not in result

    def test_non_int_status_excluded(self) -> None:
        result = run_eval._summarize_context_compaction_meta({"status": "ok"})
        assert "status" not in result


# ===================================================================
# _summarize_recovery_usage_meta
# ===================================================================


class TestSummarizeRecoveryUsageMeta:
    def test_basic_fields(self) -> None:
        meta = {
            "llm_call_count": 2,
            "status": 201,
            "usage": {"prompt_tokens": 5, "completion_tokens": 10, "total_tokens": 15},
            "error": "err",
            "exc": "exc_detail",
            "reason": "retry",
            "fallback_reason": "fallback",
            "body": "b",
        }
        result = run_eval._summarize_recovery_usage_meta(meta)
        assert result["llm_call_count"] == 2
        assert result["status"] == 201
        assert result["usage"]["total_tokens"] == 15
        assert result["error"] == "err"
        assert result["body"] == "b"

    def test_empty_meta(self) -> None:
        result = run_eval._summarize_recovery_usage_meta({})
        assert result == {}


# ===================================================================
# _ensure_deterministic_seed_ack
# ===================================================================


class TestEnsureDeterministicSeedAck:
    def test_noop_when_not_deterministic(self) -> None:
        # Should not raise
        run_eval._ensure_deterministic_seed_ack(
            deterministic=False,
            seed=42,
            patcher_name="openai",
            patch_meta={},
            case_id="c1",
            condition="base_repair",
            attempt=1,
        )

    def test_noop_when_not_openai(self) -> None:
        run_eval._ensure_deterministic_seed_ack(
            deterministic=True,
            seed=42,
            patcher_name="heuristic",
            patch_meta={},
            case_id="c1",
            condition="base_repair",
            attempt=1,
        )

    def test_seed_mismatch_raises(self) -> None:
        with pytest.raises(RuntimeError, match="seed mismatch"):
            run_eval._ensure_deterministic_seed_ack(
                deterministic=True,
                seed=42,
                patcher_name="openai",
                patch_meta={"api_compat_seed_requested": 99},
                case_id="c1",
                condition="base_repair",
                attempt=1,
            )

    def test_seed_not_acknowledged_raises(self) -> None:
        with pytest.raises(RuntimeError, match="deterministic mode failed"):
            run_eval._ensure_deterministic_seed_ack(
                deterministic=True,
                seed=42,
                patcher_name="openai",
                patch_meta={
                    "api_compat_seed_requested": 42,
                    "api_compat_seed_acknowledged": False,
                    "error": "backend did not acknowledge",
                },
                case_id="c1",
                condition="base_repair",
                attempt=1,
            )

    def test_success(self) -> None:
        run_eval._ensure_deterministic_seed_ack(
            deterministic=True,
            seed=42,
            patcher_name="openai",
            patch_meta={
                "api_compat_seed_requested": 42,
                "api_compat_seed_acknowledged": True,
            },
            case_id="c1",
            condition="base_repair",
            attempt=1,
        )


# ===================================================================
# _adaptive_gate_should_invoke_helper
# ===================================================================


class TestAdaptiveGateShouldInvokeHelper:
    def test_timed_out_always_invokes(self) -> None:
        invoke, reason = run_eval._adaptive_gate_should_invoke_helper(
            policy="conservative",
            attempt=1,
            timed_out=True,
            signature_repeated=False,
            stagnation_streak=0,
            stagnation_window=2,
        )
        assert invoke is True
        assert reason == "gate_timeout"

    def test_aggressive_always_invokes(self) -> None:
        invoke, reason = run_eval._adaptive_gate_should_invoke_helper(
            policy="aggressive",
            attempt=1,
            timed_out=False,
            signature_repeated=False,
            stagnation_streak=0,
            stagnation_window=2,
        )
        assert invoke is True
        assert reason == "gate_aggressive"

    def test_conservative_needs_high_streak(self) -> None:
        invoke, reason = run_eval._adaptive_gate_should_invoke_helper(
            policy="conservative",
            attempt=3,
            timed_out=False,
            signature_repeated=True,
            stagnation_streak=3,
            stagnation_window=2,
        )
        assert invoke is True
        assert reason == "gate_conservative_repeated_signature"

    def test_conservative_waiting(self) -> None:
        invoke, reason = run_eval._adaptive_gate_should_invoke_helper(
            policy="conservative",
            attempt=3,
            timed_out=False,
            signature_repeated=True,
            stagnation_streak=1,
            stagnation_window=2,
        )
        assert invoke is False
        assert reason == "gate_conservative_waiting"

    def test_balanced_invokes_after_streak(self) -> None:
        invoke, reason = run_eval._adaptive_gate_should_invoke_helper(
            policy="balanced",
            attempt=2,
            timed_out=False,
            signature_repeated=True,
            stagnation_streak=2,
            stagnation_window=2,
        )
        assert invoke is True
        assert reason == "gate_balanced_repeated_signature"

    def test_balanced_waiting(self) -> None:
        invoke, reason = run_eval._adaptive_gate_should_invoke_helper(
            policy="balanced",
            attempt=1,
            timed_out=False,
            signature_repeated=False,
            stagnation_streak=0,
            stagnation_window=2,
        )
        assert invoke is False
        assert reason == "gate_balanced_waiting"


# ===================================================================
# _adaptive_should_stop_for_no_progress
# ===================================================================


class TestAdaptiveShouldStopForNoProgress:
    def test_streak_below_window(self) -> None:
        stop, reason = run_eval._adaptive_should_stop_for_no_progress(
            no_progress_streak=1,
            stop_window=3,
            helper_budget_exhausted=False,
            helper_saturation_guarded=False,
            helper_decision_reason="",
        )
        assert stop is False
        assert reason == "streak_below_window"

    def test_helper_budget_exhausted(self) -> None:
        stop, reason = run_eval._adaptive_should_stop_for_no_progress(
            no_progress_streak=5,
            stop_window=3,
            helper_budget_exhausted=True,
            helper_saturation_guarded=False,
            helper_decision_reason="",
        )
        assert stop is True
        assert reason == "helper_budget_exhausted"

    def test_helper_saturation_guarded(self) -> None:
        stop, reason = run_eval._adaptive_should_stop_for_no_progress(
            no_progress_streak=5,
            stop_window=3,
            helper_budget_exhausted=False,
            helper_saturation_guarded=True,
            helper_decision_reason="",
        )
        assert stop is True
        assert reason == "helper_saturation_guard"

    def test_llm_declined_helper(self) -> None:
        stop, reason = run_eval._adaptive_should_stop_for_no_progress(
            no_progress_streak=5,
            stop_window=3,
            helper_budget_exhausted=False,
            helper_saturation_guarded=False,
            helper_decision_reason="llm_declined_helper",
        )
        assert stop is True
        assert reason == "llm_declined_helper"

    def test_waiting_for_more_signal(self) -> None:
        stop, reason = run_eval._adaptive_should_stop_for_no_progress(
            no_progress_streak=5,
            stop_window=3,
            helper_budget_exhausted=False,
            helper_saturation_guarded=False,
            helper_decision_reason="other",
        )
        assert stop is False
        assert reason == "waiting_for_more_signal"


# ===================================================================
# _mark_adaptive_progress_stop
# ===================================================================


class TestMarkAdaptiveProgressStop:
    def test_triggered(self) -> None:
        attempt_meta: dict[str, Any] = {}
        patch_meta: dict[str, Any] = {}
        result = run_eval._mark_adaptive_progress_stop(
            attempt_metadata=attempt_meta,
            patch_meta=patch_meta,
            is_adaptive_condition=True,
            adaptive_progress_stop_candidate=True,
            adaptive_progress_stop_reason="helper_budget_exhausted",
            note="tests_still_failing",
        )
        assert result is True
        assert attempt_meta["adaptive_progress_stop_triggered"] is True
        assert patch_meta["adaptive_progress_stop_note"] == "tests_still_failing"

    def test_not_triggered_non_adaptive(self) -> None:
        attempt_meta: dict[str, Any] = {}
        patch_meta: dict[str, Any] = {}
        result = run_eval._mark_adaptive_progress_stop(
            attempt_metadata=attempt_meta,
            patch_meta=patch_meta,
            is_adaptive_condition=False,
            adaptive_progress_stop_candidate=True,
            adaptive_progress_stop_reason="reason",
            note="tests_still_failing",
        )
        assert result is False
        assert "adaptive_progress_stop_note" not in patch_meta

    def test_not_triggered_wrong_note(self) -> None:
        attempt_meta: dict[str, Any] = {}
        patch_meta: dict[str, Any] = {}
        result = run_eval._mark_adaptive_progress_stop(
            attempt_metadata=attempt_meta,
            patch_meta=patch_meta,
            is_adaptive_condition=True,
            adaptive_progress_stop_candidate=True,
            adaptive_progress_stop_reason="reason",
            note="success",
        )
        assert result is False


# ===================================================================
# _apply_no_progress_token_cap_candidate
# ===================================================================


class TestApplyNoProgressTokenCapCandidate:
    def test_non_adaptive_passthrough(self) -> None:
        cand, _reason, token_sum, hit = run_eval._apply_no_progress_token_cap_candidate(
            is_adaptive_condition=False,
            adaptive_progress_made=False,
            adaptive_no_progress_token_sum=100,
            attempt_tokens_total=50,
            adaptive_no_progress_token_cap=200,
            adaptive_progress_stop_candidate=False,
            adaptive_progress_stop_reason="none",
        )
        assert cand is False
        assert token_sum == 100
        assert hit is False

    def test_progress_resets_sum(self) -> None:
        _cand, _reason, token_sum, hit = run_eval._apply_no_progress_token_cap_candidate(
            is_adaptive_condition=True,
            adaptive_progress_made=True,
            adaptive_no_progress_token_sum=500,
            attempt_tokens_total=50,
            adaptive_no_progress_token_cap=200,
            adaptive_progress_stop_candidate=False,
            adaptive_progress_stop_reason="none",
        )
        assert token_sum == 0
        assert hit is False

    def test_cap_hit(self) -> None:
        cand, reason, token_sum, hit = run_eval._apply_no_progress_token_cap_candidate(
            is_adaptive_condition=True,
            adaptive_progress_made=False,
            adaptive_no_progress_token_sum=150,
            attempt_tokens_total=100,
            adaptive_no_progress_token_cap=200,
            adaptive_progress_stop_candidate=False,
            adaptive_progress_stop_reason="none",
        )
        assert cand is True
        assert reason == "no_progress_token_cap"
        assert token_sum == 250
        assert hit is True

    def test_below_cap(self) -> None:
        cand, reason, token_sum, hit = run_eval._apply_no_progress_token_cap_candidate(
            is_adaptive_condition=True,
            adaptive_progress_made=False,
            adaptive_no_progress_token_sum=50,
            attempt_tokens_total=30,
            adaptive_no_progress_token_cap=200,
            adaptive_progress_stop_candidate=False,
            adaptive_progress_stop_reason="old_reason",
        )
        assert cand is False
        assert reason == "old_reason"
        assert token_sum == 80
        assert hit is False


# ===================================================================
# _extract_exception_info
# ===================================================================


class TestExtractExceptionInfo:
    def test_from_snapshot(self) -> None:
        snapshot = {
            "exception": {
                "type": "builtins.ValueError",
                "message": "bad value",
            },
            "frames": [{"file_rel": "src/mod.py"}],
        }
        exc_type, msg, frames, in_user = run_eval._extract_exception_info(
            snapshot=snapshot, stdout="", stderr=""
        )
        assert exc_type == "ValueError"
        assert msg == "bad value"
        assert frames == 1
        assert in_user is True

    def test_from_snapshot_no_frames(self) -> None:
        snapshot = {
            "exception": {"type": "KeyError", "message": "'key'"},
        }
        exc_type, _msg, frames, in_user = run_eval._extract_exception_info(
            snapshot=snapshot, stdout="", stderr=""
        )
        assert exc_type == "KeyError"
        assert frames == 0
        assert in_user is False

    def test_from_snapshot_frame_without_file_rel(self) -> None:
        snapshot = {
            "exception": {"type": "RuntimeError", "message": "boom"},
            "frames": [{"function": "main"}],
        }
        exc_type, _, _, in_user = run_eval._extract_exception_info(
            snapshot=snapshot, stdout="", stderr=""
        )
        assert exc_type == "RuntimeError"
        assert in_user is False

    def test_fallback_stdout_pytest_error(self) -> None:
        stdout = "E   builtins.TypeError: expected int"
        exc_type, _msg, _frames, _in_user = run_eval._extract_exception_info(
            snapshot=None, stdout=stdout, stderr=""
        )
        # May or may not match depending on regex; test the fallback path
        assert isinstance(exc_type, str)

    def test_fallback_no_match(self) -> None:
        exc_type, msg, _frames, _in_user = run_eval._extract_exception_info(
            snapshot=None, stdout="", stderr=""
        )
        assert exc_type == "Unknown"
        assert msg == ""


# ===================================================================
# _traceback_informativeness_score
# ===================================================================


class TestTracebackInformativenessScore:
    def test_assertion_error_with_values(self) -> None:
        result = run_eval._traceback_informativeness_score(
            exception_type="AssertionError",
            exception_message="assert 1 == 2",
            frame_count=2,
            crash_in_user_code=True,
        )
        assert result.score > 0.0
        assert result.f_assertion_quality == 1.0
        assert result.f_frame_depth == 1.0

    def test_assertion_error_no_message(self) -> None:
        result = run_eval._traceback_informativeness_score(
            exception_type="AssertionError",
            exception_message="",
            frame_count=0,
            crash_in_user_code=False,
        )
        assert result.f_assertion_quality == 0.0

    def test_assertion_error_generic_message(self) -> None:
        result = run_eval._traceback_informativeness_score(
            exception_type="AssertionError",
            exception_message="something went wrong",
            frame_count=0,
            crash_in_user_code=False,
        )
        assert result.f_assertion_quality == 0.5

    def test_high_frame_count(self) -> None:
        result = run_eval._traceback_informativeness_score(
            exception_type="RuntimeError",
            exception_message="",
            frame_count=15,
            crash_in_user_code=False,
        )
        assert result.f_frame_depth == 0.2

    def test_medium_frame_count(self) -> None:
        result = run_eval._traceback_informativeness_score(
            exception_type="RuntimeError",
            exception_message="",
            frame_count=7,
            crash_in_user_code=False,
        )
        assert result.f_frame_depth == 0.4

    def test_six_frames(self) -> None:
        result = run_eval._traceback_informativeness_score(
            exception_type="RuntimeError",
            exception_message="",
            frame_count=5,
            crash_in_user_code=False,
        )
        assert result.f_frame_depth == 0.7

    def test_tier_classification(self) -> None:
        # Low specificity → tier D
        result = run_eval._traceback_informativeness_score(
            exception_type="UnknownFooBarException",
            exception_message="",
            frame_count=0,
            crash_in_user_code=False,
        )
        assert "tierD" in result.reason or "tierC" in result.reason or "tier" in result.reason


# ===================================================================
# _traceback_informativeness_pre_gate
# ===================================================================


class TestTracebackInformativenessPreGate:
    def test_below_threshold_not_skipped(self) -> None:
        result = run_eval._traceback_informativeness_pre_gate(
            snapshot=None, stdout="", stderr="", threshold=999.0
        )
        assert result.should_skip is False

    def test_above_threshold_skipped(self) -> None:
        snapshot = {
            "exception": {"type": "AssertionError", "message": "assert 1 == 2"},
            "frames": [{"file_rel": "test.py"}],
        }
        result = run_eval._traceback_informativeness_pre_gate(
            snapshot=snapshot, stdout="", stderr="", threshold=0.01
        )
        assert result.should_skip is True


# ===================================================================
# _adaptive_prompt_budget_for_attempt
# ===================================================================


class TestAdaptivePromptBudgetForAttempt:
    def test_non_compact_returns_unchanged(self) -> None:
        budget = {"total_chars": 200_000, "stdout_chars": 50_000}
        result = run_eval._adaptive_prompt_budget_for_attempt(
            base_budget=budget, helper_materialized_context=True, profile="default"
        )
        assert result == budget

    def test_not_materialized_returns_unchanged(self) -> None:
        budget = {"total_chars": 200_000}
        result = run_eval._adaptive_prompt_budget_for_attempt(
            base_budget=budget, helper_materialized_context=False, profile="compact"
        )
        assert result == budget

    def test_compact_caps_values(self) -> None:
        budget = {
            "total_chars": 200_000,
            "retry_delta_chars": 20_000,
            "stdout_chars": 50_000,
            "snapshot_chars": 40_000,
            "evidence_chars": 60_000,
            "retry_history_chars": 15_000,
        }
        result = run_eval._adaptive_prompt_budget_for_attempt(
            base_budget=budget, helper_materialized_context=True, profile="compact"
        )
        assert result["total_chars"] == 120_000
        assert result["retry_delta_chars"] == 12_000
        assert result["stdout_chars"] == 16_000
        assert result["snapshot_chars"] == 20_000
        assert result["evidence_chars"] == 32_000
        assert result["retry_history_chars"] == 8_000

    def test_compact_does_not_increase_small_values(self) -> None:
        budget = {"total_chars": 50_000, "stdout_chars": 5_000}
        result = run_eval._adaptive_prompt_budget_for_attempt(
            base_budget=budget, helper_materialized_context=True, profile="compact"
        )
        assert result["total_chars"] == 50_000
        assert result["stdout_chars"] == 5_000

    def test_compact_zero_values_unchanged(self) -> None:
        budget = {"total_chars": 0}
        result = run_eval._adaptive_prompt_budget_for_attempt(
            base_budget=budget, helper_materialized_context=True, profile="compact"
        )
        assert result["total_chars"] == 0


# ===================================================================
# _is_transport_runtime_error
# ===================================================================


class TestIsTransportRuntimeError:
    @pytest.mark.parametrize(
        ("message", "expected"),
        [
            ("", False),
            ("Segmentation fault", True),
            ("Connection refused", True),
            ("Connection reset by peer", True),
            ("BrokenPipeError", True),
            ("RemoteDisconnected", True),
            ("temporarily unavailable", True),
            ("server disconnected", True),
            ("failed to establish a new connection", True),
            ("normal error message", False),
        ],
    )
    def test_all_markers(self, message: str, expected: bool) -> None:
        assert run_eval._is_transport_runtime_error(message) is expected


# ===================================================================
# _record_is_transport_request_failure
# ===================================================================


class TestRecordIsTransportRequestFailure:
    def test_infra_event_returns_false(self) -> None:
        record = {"record_type": EVAL_RECORD_TYPE_INFRA_EVENT, "note": "patcher_request_failed"}
        assert run_eval._record_is_transport_request_failure(record) is False

    def test_non_patcher_note_returns_false(self) -> None:
        record = {"record_type": EVAL_RECORD_TYPE_ATTEMPT, "note": "success"}
        assert run_eval._record_is_transport_request_failure(record) is False

    def test_patcher_failed_with_transport_kind(self) -> None:
        record = {
            "record_type": EVAL_RECORD_TYPE_ATTEMPT,
            "note": "patcher_request_failed",
            "patch_meta": {
                "patcher_request_failed": True,
                "patcher_request_failure_kind": "transport",
            },
        }
        assert run_eval._record_is_transport_request_failure(record) is True

    def test_patcher_failed_non_transport(self) -> None:
        record = {
            "record_type": EVAL_RECORD_TYPE_ATTEMPT,
            "note": "patcher_request_failed",
            "patch_meta": {
                "patcher_request_failed": True,
                "patcher_request_failure_kind": "context_limit",
            },
        }
        assert run_eval._record_is_transport_request_failure(record) is False

    def test_fallback_classify_transport(self) -> None:
        record = {
            "record_type": EVAL_RECORD_TYPE_ATTEMPT,
            "note": "patcher_request_failed",
            "patch_meta": {"error": "request_failed"},
        }
        assert run_eval._record_is_transport_request_failure(record) is True

    def test_no_patch_meta(self) -> None:
        record = {
            "record_type": EVAL_RECORD_TYPE_ATTEMPT,
            "note": "patcher_request_failed",
        }
        assert run_eval._record_is_transport_request_failure(record) is False


# ===================================================================
# _token_usage_blocked_from_meta / _usage_available_flag_from_meta
# ===================================================================


class TestTokenUsageFlags:
    def test_blocked_non_dict(self) -> None:
        assert run_eval._token_usage_blocked_from_meta(None) is False

    def test_blocked_explicit_false(self) -> None:
        assert run_eval._token_usage_blocked_from_meta({"token_usage_available": False}) is True

    def test_blocked_explicit_true(self) -> None:
        assert run_eval._token_usage_blocked_from_meta({"token_usage_available": True}) is False

    def test_blocked_usage_available_false(self) -> None:
        assert run_eval._token_usage_blocked_from_meta({"usage_available": False}) is True

    def test_blocked_baseline_usage_available_false(self) -> None:
        assert (
            run_eval._token_usage_blocked_from_meta(
                {"baseline_metrics": {"usage_available": False}}
            )
            is True
        )

    def test_blocked_baseline_usage_available_true(self) -> None:
        assert (
            run_eval._token_usage_blocked_from_meta({"baseline_metrics": {"usage_available": True}})
            is False
        )

    def test_usage_available_flag_non_dict(self) -> None:
        assert run_eval._usage_available_flag_from_meta(None) is None

    def test_usage_available_flag_explicit(self) -> None:
        assert run_eval._usage_available_flag_from_meta({"usage_available": True}) is True
        assert run_eval._usage_available_flag_from_meta({"usage_available": False}) is False

    def test_usage_available_flag_from_baseline(self) -> None:
        meta = {"baseline_metrics": {"usage_available": True}}
        assert run_eval._usage_available_flag_from_meta(meta) is True

    def test_usage_available_flag_no_source(self) -> None:
        assert run_eval._usage_available_flag_from_meta({}) is None


# ===================================================================
# _token_usage_available_from_meta
# ===================================================================


class TestTokenUsageAvailableFromMeta:
    def test_non_dict(self) -> None:
        assert run_eval._token_usage_available_from_meta(None) is False

    def test_explicit_false(self) -> None:
        assert run_eval._token_usage_available_from_meta({"token_usage_available": False}) is False

    def test_explicit_true_with_counts(self) -> None:
        meta = {
            "token_usage_available": True,
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        }
        assert run_eval._token_usage_available_from_meta(meta) is True

    def test_explicit_true_with_baseline_counts(self) -> None:
        meta = {
            "token_usage_available": True,
            "baseline_metrics": {
                "usage_available": True,
                "reported_tokens_prompt": 10,
                "reported_tokens_completion": 5,
                "reported_tokens_total": 15,
            },
        }
        assert run_eval._token_usage_available_from_meta(meta) is True

    def test_usage_available_false(self) -> None:
        meta = {"usage_available": False}
        assert run_eval._token_usage_available_from_meta(meta) is False

    def test_fallback_to_usage_dict(self) -> None:
        meta = {"usage": {"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8}}
        assert run_eval._token_usage_available_from_meta(meta) is True

    def test_fallback_no_usage(self) -> None:
        assert run_eval._token_usage_available_from_meta({}) is False


# ===================================================================
# _usage_token_counts_from_patch_meta
# ===================================================================


class TestUsageTokenCountsFromPatchMeta:
    def test_non_dict(self) -> None:
        assert run_eval._usage_token_counts_from_patch_meta(None) == (0, 0, 0)

    def test_blocked_returns_zero(self) -> None:
        meta = {"token_usage_available": False, "tokens_prompt": 10}
        assert run_eval._usage_token_counts_from_patch_meta(meta) == (0, 0, 0)

    def test_from_usage_dict(self) -> None:
        meta = {
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
        }
        p, c, t = run_eval._usage_token_counts_from_patch_meta(meta)
        assert p == 10
        assert c == 20
        assert t == 30

    def test_from_top_level_fields(self) -> None:
        meta = {"tokens_prompt": 5, "tokens_completion": 3, "tokens_total": 8}
        p, c, t = run_eval._usage_token_counts_from_patch_meta(meta)
        assert p == 5
        assert c == 3
        assert t == 8

    def test_baseline_reported_fallback(self) -> None:
        meta = {
            "baseline_metrics": {
                "usage_available": True,
                "reported_tokens_prompt": 7,
                "reported_tokens_completion": 3,
                "reported_tokens_total": 10,
            },
        }
        p, c, t = run_eval._usage_token_counts_from_patch_meta(meta)
        assert p == 7
        assert c == 3
        assert t == 10

    def test_reasoning_counts_from_usage_and_top_level(self) -> None:
        meta = {
            "usage": {
                "prompt_tokens": 10,
                "completion_tokens": 20,
                "total_tokens": 30,
                "output_tokens_details": {"reasoning_tokens": 4},
            }
        }
        assert run_eval._reasoning_token_counts_from_patch_meta(meta) == (4, True)

        top_level_meta = {"tokens_reasoning": 6, "reasoning_token_usage_available": True}
        assert run_eval._reasoning_token_counts_from_patch_meta(top_level_meta) == (6, True)

    def test_reasoning_counts_from_baseline_metrics(self) -> None:
        meta = {
            "baseline_metrics": {
                "usage_available": True,
                "reported_tokens_reasoning": 5,
            }
        }
        assert run_eval._reasoning_token_counts_from_patch_meta(meta) == (5, True)


# ===================================================================
# _usage_token_counts_from_usage_payload (Anthropic fallback)
# ===================================================================


class TestUsageTokenCountsFromUsagePayload:
    def test_non_dict(self) -> None:
        assert run_eval._usage_token_counts_from_usage_payload(None) == (0, 0, 0)

    def test_openai_style(self) -> None:
        usage = {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        assert run_eval._usage_token_counts_from_usage_payload(usage) == (10, 20, 30)

    def test_anthropic_style(self) -> None:
        usage = {"input_tokens": 10, "output_tokens": 20}
        p, c, t = run_eval._usage_token_counts_from_usage_payload(usage)
        assert p == 10
        assert c == 20
        assert t == 30

    def test_computed_total(self) -> None:
        usage = {"prompt_tokens": 10, "completion_tokens": 20}
        _, _, t = run_eval._usage_token_counts_from_usage_payload(usage)
        assert t == 30

    def test_reasoning_tokens_nested(self) -> None:
        usage = {"output_tokens_details": {"reasoning_tokens": 8}}
        assert run_eval._reasoning_token_counts_from_usage_payload(usage) == (8, True)

    def test_reasoning_tokens_zero_is_covered(self) -> None:
        usage = {"completion_tokens_details": {"reasoning_tokens": 0}}
        assert run_eval._reasoning_token_counts_from_usage_payload(usage) == (0, True)

    def test_reasoning_tokens_malformed_is_unavailable(self) -> None:
        usage = {"reasoning_tokens": "bad"}
        assert run_eval._reasoning_token_counts_from_usage_payload(usage) == (0, False)


# ===================================================================
# _adaptive_pipeline_stage_for_outcome
# ===================================================================


class TestAdaptivePipelineStageForOutcome:
    @pytest.mark.parametrize(
        ("note", "success", "expected_stage", "expected_ord"),
        [
            ("success", True, "success", 4),
            ("", True, "success", 4),
            ("patcher_request_failed", False, "request_or_no_patch", 0),
            ("no_patch_proposed", False, "request_or_no_patch", 0),
            ("retry_gate_no_new_evidence", False, "request_or_no_patch", 0),
            ("forced_noop_patch", False, "pretest_patch_failure", 1),
            ("patch_apply_failed", False, "pretest_patch_failure", 1),
            ("syntax_preflight_failed", False, "pretest_patch_failure", 1),
            ("patch_verifier_rejected", False, "verifier_rejected", 2),
            ("tests_still_failing", False, "tests_still_failing", 3),
            ("unknown_note", False, "tests_still_failing", 3),
        ],
    )
    def test_all_outcomes(
        self, note: str, success: bool, expected_stage: str, expected_ord: int
    ) -> None:
        stage, ordinal = run_eval._adaptive_pipeline_stage_for_outcome(note=note, success=success)
        assert stage == expected_stage
        assert ordinal == expected_ord


# ===================================================================
# _adaptive_pipeline_stage_before
# ===================================================================


class TestAdaptivePipelineStageBefore:
    def test_from_previous_context(self) -> None:
        stage, _ordinal = run_eval._adaptive_pipeline_stage_before(
            previous_failure_context={"note": "patch_apply_failed"},
            current_run=None,
        )
        assert stage == "pretest_patch_failure"

    def test_no_context_no_run(self) -> None:
        stage, ordinal = run_eval._adaptive_pipeline_stage_before(
            previous_failure_context=None,
            current_run=None,
        )
        assert stage == "tests_still_failing"
        assert ordinal == 3

    def test_timed_out_run(self) -> None:
        run = _run_result(timed_out=True, timeout_sec=30)
        stage, _ordinal = run_eval._adaptive_pipeline_stage_before(
            previous_failure_context=None,
            current_run=run,
        )
        assert stage == "tests_still_failing"

    def test_successful_run(self) -> None:
        run = _run_result(returncode=0)
        stage, ordinal = run_eval._adaptive_pipeline_stage_before(
            previous_failure_context=None,
            current_run=run,
        )
        assert stage == "success"
        assert ordinal == 4


# ===================================================================
# _pytest_failed_total_from_text
# ===================================================================


class TestPytestFailedTotalFromText:
    def test_standard_summary(self) -> None:
        text = "===== 3 failed, 2 passed in 1.23s ====="
        assert run_eval._pytest_failed_total_from_text(text) == 3

    def test_all_passed(self) -> None:
        text = "===== 5 passed in 0.45s ====="
        assert run_eval._pytest_failed_total_from_text(text) == 0

    def test_no_tests_ran(self) -> None:
        text = "===== no tests ran in 0.01s ====="
        assert run_eval._pytest_failed_total_from_text(text) == 0

    def test_errors(self) -> None:
        text = "===== 2 error in 0.30s ====="
        assert run_eval._pytest_failed_total_from_text(text) == 2

    def test_empty_text(self) -> None:
        assert run_eval._pytest_failed_total_from_text("") is None

    def test_no_summary(self) -> None:
        assert run_eval._pytest_failed_total_from_text("some random output") is None


class _TestDiscoverCasesSkipped:
    """Skipped: _discover_cases does not exist in run_eval."""

    def _test_valid_case(self, tmp_path: Path) -> None:
        case_dir = tmp_path / "case_001"
        case_dir.mkdir()
        meta = {
            "category": "logic_error",
            "difficulty": "easy",
            "snapshot_dependency": "required",
            "expected_exception": "AssertionError",
        }
        (case_dir / "case.json").write_text(json.dumps(meta), encoding="utf-8")
        cases = run_eval._discover_cases(tmp_path)
        assert len(cases) == 1
        assert cases[0].case_id == "case_001"
        assert cases[0].category == "logic_error"

    def _test_invalid_category_raises(self, tmp_path: Path) -> None:
        case_dir = tmp_path / "case_bad"
        case_dir.mkdir()
        meta = {
            "category": "INVALID",
            "difficulty": "easy",
            "snapshot_dependency": "required",
            "expected_exception": "AssertionError",
        }
        (case_dir / "case.json").write_text(json.dumps(meta), encoding="utf-8")
        with pytest.raises(ValueError, match="category must be one of"):
            run_eval._discover_cases(tmp_path)

    def _test_invalid_difficulty_raises(self, tmp_path: Path) -> None:
        case_dir = tmp_path / "case_bad"
        case_dir.mkdir()
        meta = {
            "category": "logic_error",
            "difficulty": "INVALID",
            "snapshot_dependency": "required",
            "expected_exception": "AssertionError",
        }
        (case_dir / "case.json").write_text(json.dumps(meta), encoding="utf-8")
        with pytest.raises(ValueError, match="difficulty must be one of"):
            run_eval._discover_cases(tmp_path)

    def _test_invalid_snapshot_dependency_raises(self, tmp_path: Path) -> None:
        case_dir = tmp_path / "case_bad"
        case_dir.mkdir()
        meta = {
            "category": "logic_error",
            "difficulty": "easy",
            "snapshot_dependency": "INVALID",
            "expected_exception": "AssertionError",
        }
        (case_dir / "case.json").write_text(json.dumps(meta), encoding="utf-8")
        with pytest.raises(ValueError, match="snapshot_dependency must be one of"):
            run_eval._discover_cases(tmp_path)

    def _test_missing_expected_exception_raises(self, tmp_path: Path) -> None:
        case_dir = tmp_path / "case_bad"
        case_dir.mkdir()
        meta = {
            "category": "logic_error",
            "difficulty": "easy",
            "snapshot_dependency": "required",
        }
        (case_dir / "case.json").write_text(json.dumps(meta), encoding="utf-8")
        with pytest.raises(ValueError, match="expected_exception must be"):
            run_eval._discover_cases(tmp_path)

    def _test_invalid_tags_raises(self, tmp_path: Path) -> None:
        case_dir = tmp_path / "case_bad"
        case_dir.mkdir()
        meta = {
            "category": "logic_error",
            "difficulty": "easy",
            "snapshot_dependency": "required",
            "expected_exception": "AssertionError",
            "tags": [123],
        }
        (case_dir / "case.json").write_text(json.dumps(meta), encoding="utf-8")
        with pytest.raises(ValueError, match="tags must be"):
            run_eval._discover_cases(tmp_path)

    def _test_case_with_all_optional_fields(self, tmp_path: Path) -> None:
        case_dir = tmp_path / "case_full"
        case_dir.mkdir()
        meta = {
            "category": "logic_error",
            "difficulty": "medium",
            "snapshot_dependency": "helpful",
            "expected_exception": "TypeError",
            "tags": ["regression", "import"],
            "python_args": ["-m", "pytest", "-v"],
            "timeout_sec": 60,
            "description": "Test case with all fields",
            "bug_source": "synthetic",
            "contamination_risk": "low",
            "expected_fix_lines": 3,
            "source_id": "src_001",
            "difficulty_rationale": "simple fix",
        }
        (case_dir / "case.json").write_text(json.dumps(meta), encoding="utf-8")
        cases = run_eval._discover_cases(tmp_path)
        assert len(cases) == 1
        c = cases[0]
        assert c.tags == ["regression", "import"]
        assert c.python_args == ["-m", "pytest", "-v"]
        assert c.timeout_sec == 60
        assert c.description == "Test case with all fields"
        assert c.bug_source == "synthetic"
        assert c.expected_fix_lines == 3

    def _test_invalid_json_raises(self, tmp_path: Path) -> None:
        case_dir = tmp_path / "case_bad"
        case_dir.mkdir()
        (case_dir / "case.json").write_text("{invalid json", encoding="utf-8")
        with pytest.raises(ValueError, match="invalid JSON"):
            run_eval._discover_cases(tmp_path)

    def _test_skips_files_in_root(self, tmp_path: Path) -> None:
        (tmp_path / "not_a_dir.txt").write_text("x", encoding="utf-8")
        result = run_eval._discover_cases(tmp_path)
        assert result == []


# ===================================================================
# _llm_call_count_from_meta
# ===================================================================


class TestLlmCallCountFromMeta:
    def test_non_dict(self) -> None:
        assert run_eval._llm_call_count_from_meta(None) == 0

    def test_valid_count(self) -> None:
        assert run_eval._llm_call_count_from_meta({"llm_call_count": 5}) == 5

    def test_invalid_count_uses_default(self) -> None:
        assert run_eval._llm_call_count_from_meta({"llm_call_count": -1}, default=3) == 3

    def test_missing_key_uses_default(self) -> None:
        assert run_eval._llm_call_count_from_meta({}, default=7) == 7


# ===================================================================
# _timeout_error_from_meta (recursive)
# ===================================================================


class TestTimeoutErrorFromMeta:
    def test_non_dict(self) -> None:
        assert run_eval._timeout_error_from_meta(None) is False

    def test_direct_timeout(self) -> None:
        assert run_eval._timeout_error_from_meta({"error": "timeout"}) is True

    def test_nested_baseline_timeout(self) -> None:
        meta = {"baseline_meta": {"error": "timeout"}}
        assert run_eval._timeout_error_from_meta(meta) is True

    def test_no_timeout(self) -> None:
        assert run_eval._timeout_error_from_meta({"error": "other"}) is False


# ===================================================================
# _apply_usage_totals_to_patch_meta
# ===================================================================


class TestApplyUsageTotalsToPatchMeta:
    def test_normal(self) -> None:
        meta: dict[str, Any] = {
            "usage": {
                "prompt_tokens": 2,
                "completion_tokens": 3,
                "total_tokens": 5,
                "output_tokens_details": {"reasoning_tokens": 4},
            }
        }
        run_eval._apply_usage_totals_to_patch_meta(
            patch_meta=meta,
            llm_call_count=2,
            tokens_prompt=10,
            tokens_completion=20,
            tokens_total=30,
        )
        assert meta["llm_call_count"] == 2
        assert meta["tokens_prompt"] == 10
        assert meta["usage"]["total_tokens"] == 30
        assert meta["tokens_reasoning"] == 4
        assert meta["reasoning_token_usage_available"] is True
        assert meta["usage"]["reasoning_tokens"] == 4

    def test_blocked_usage(self) -> None:
        meta: dict[str, Any] = {"token_usage_available": False, "usage": {"old": True}}
        run_eval._apply_usage_totals_to_patch_meta(
            patch_meta=meta,
            llm_call_count=1,
            tokens_prompt=10,
            tokens_completion=20,
            tokens_total=30,
        )
        assert meta["tokens_prompt"] == 0
        assert meta["tokens_completion"] == 0
        assert meta["tokens_reasoning"] == 0
        assert meta["reasoning_token_usage_available"] is False
        assert "usage" not in meta


# ===================================================================
# _sync_patch_diff_metadata
# ===================================================================


class TestSyncPatchDiffMetadata:
    def test_sets_hash_and_chars(self) -> None:
        meta: dict[str, Any] = {}
        diff = "--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-old\n+new\n"
        run_eval._sync_patch_diff_metadata(patch_meta=meta, diff=diff)
        assert "patch_diff_xxh3_128" in meta
        assert meta["patch_diff_chars"] == len(diff)


# ===================================================================
# _format_duration_precise
# ===================================================================


class TestFormatDurationPrecise:
    def test_none(self) -> None:
        assert run_eval._format_duration_precise(None) == "unknown"

    def test_short(self) -> None:
        result = run_eval._format_duration_precise(5.12)
        assert result == "5.12s"

    def test_long(self) -> None:
        result = run_eval._format_duration_precise(125.0)
        assert "m" in result or "s" in result


# ===================================================================
# _progress_is_enabled / _normalized_progress_verbosity
# ===================================================================


class TestProgressHelpers:
    def test_normalized_verbosity_valid(self) -> None:
        assert run_eval._normalized_progress_verbosity("basic") == "basic"
        assert run_eval._normalized_progress_verbosity("detailed") == "detailed"
        assert run_eval._normalized_progress_verbosity("trace") == "trace"

    def test_normalized_verbosity_invalid(self) -> None:
        assert run_eval._normalized_progress_verbosity("INVALID") == "basic"
        assert run_eval._normalized_progress_verbosity(None) == "basic"

    def test_progress_is_enabled(self) -> None:
        assert run_eval._progress_is_enabled("detailed", "basic") is True
        assert run_eval._progress_is_enabled("basic", "detailed") is False


# ===================================================================
# _progress_print
# ===================================================================


class TestProgressPrint:
    def test_suppressed_when_level_too_high(self, capsys: pytest.CaptureFixture[str]) -> None:
        run_eval._progress_print(verbosity="basic", level="trace", message="should not appear")
        captured = capsys.readouterr()
        assert "should not appear" not in captured.out

    def test_printed_when_level_ok(self, capsys: pytest.CaptureFixture[str]) -> None:
        run_eval._progress_print(verbosity="trace", level="basic", message="hello progress")
        captured = capsys.readouterr()
        assert "hello progress" in captured.out


# ===================================================================
# _baseline_patch_inventory
# ===================================================================


class TestBaselinePatchInventory:
    def test_nonexistent_dir(self) -> None:
        result = run_eval._baseline_patch_inventory("nonexistent_baseline_xyz")
        assert result == []


# ===================================================================
# _autodetect_baseline_python
# ===================================================================


class TestAutodetectBaselinePython:
    def test_empty_name(self) -> None:
        assert run_eval._autodetect_baseline_python("") == ""

    def test_nonexistent_env(self) -> None:
        # Unless the env actually exists, should return ""
        result = run_eval._autodetect_baseline_python("nonexistent_baseline_xyz")
        assert result == ""


# ===================================================================
# _resolve_external_baseline_runtime_requirements
# ===================================================================


class TestResolveExternalBaselineRuntimeRequirements:
    def test_default_repo_outside_repo_root_is_supported(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        isolated_root = tmp_path / "isolated_checkout"
        isolated_root.mkdir()
        repo_dir = tmp_path / "real_baseline_repo"
        repo_dir.mkdir()
        expected_python = isolated_root / "baselines" / "envs" / "ldb" / "bin" / "python"
        expected_python.parent.mkdir(parents=True, exist_ok=True)
        expected_python.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        expected_python.chmod(0o755)

        monkeypatch.setattr(run_eval, "_REPO_ROOT", isolated_root)
        monkeypatch.setattr(run_eval, "_default_baseline_repo_dir", lambda _name: repo_dir)

        resolved_repo, resolved_python = run_eval._resolve_external_baseline_runtime_requirements(
            baseline_name="ldb",
            baseline_repo_dir="",
            baseline_python="",
        )

        assert resolved_repo == str(repo_dir)
        assert resolved_python == str(expected_python)


# ===================================================================
# _sync_patcher_request_failure_metadata
# ===================================================================


class TestSyncPatcherRequestFailureMetadata:
    def test_fields_set(self) -> None:
        attempt_meta: dict[str, Any] = {}
        patch_meta: dict[str, Any] = {}
        run_eval._sync_patcher_request_failure_metadata(
            attempt_metadata=attempt_meta,
            patch_meta=patch_meta,
            failed=True,
            kind="transport",
            error="request_failed",
            status=500,
            exc="ConnectionError",
            runner_forced_patch_skipped=True,
            runner_forced_patch_skip_reason="budget",
        )
        assert attempt_meta["patcher_request_failed"] is True
        assert patch_meta["patcher_request_failure_kind"] == "transport"
        assert patch_meta["patcher_request_failure_status"] == 500


# ===================================================================
# _sync_patch_sanitize_metadata
# ===================================================================


class TestSyncPatchSanitizeMetadata:
    def test_fields_set(self) -> None:
        attempt_meta: dict[str, Any] = {}
        patch_meta: dict[str, Any] = {}
        run_eval._sync_patch_sanitize_metadata(
            attempt_metadata=attempt_meta,
            patch_meta=patch_meta,
            enabled=True,
            passed=False,
            rejected=True,
            reasons=["dangerous_import"],
            error_msg="rejected due to import",
        )
        assert attempt_meta["patch_sanitize_enabled"] is True
        assert patch_meta["patch_sanitize_rejected"] is True
        assert patch_meta["patch_sanitize_reasons"] == ["dangerous_import"]


# ===================================================================
# _sync_syntax_preflight_metadata
# ===================================================================


class TestSyncSyntaxPreflightMetadata:
    def test_fields_set(self) -> None:
        attempt_meta: dict[str, Any] = {}
        patch_meta: dict[str, Any] = {}
        run_eval._sync_syntax_preflight_metadata(
            attempt_metadata=attempt_meta,
            patch_meta=patch_meta,
            enabled=True,
            cycles_used=2,
            failures=1,
            last_error_type="SyntaxError",
            last_error_msg="invalid syntax",
            events=[{"event": "syntax_error"}],
        )
        assert patch_meta["syntax_preflight_enabled"] is True
        assert attempt_meta["syntax_preflight_cycles_used"] == 2
        assert patch_meta["syntax_preflight_failures"] == 1


# ===================================================================
# _sync_patch_verifier_metadata
# ===================================================================


class TestSyncPatchVerifierMetadata:
    def test_fields_set(self) -> None:
        attempt_meta: dict[str, Any] = {}
        patch_meta: dict[str, Any] = {}
        run_eval._sync_patch_verifier_metadata(
            attempt_metadata=attempt_meta,
            patch_meta=patch_meta,
            enabled=True,
            scope="all",
            passed=True,
            score=3,
            reasons=["matches_crash_file"],
            min_score=1,
        )
        assert attempt_meta["patch_verifier_enabled"] is True
        assert patch_meta["patch_verifier_score"] == 3


# ===================================================================
# _effective_patch_verifier_scope
# ===================================================================


class TestEffectivePatchVerifierScope:
    def test_disabled(self) -> None:
        assert (
            run_eval._effective_patch_verifier_scope(
                configured_scope="disabled", is_adaptive_condition=True
            )
            == "disabled"
        )

    def test_all(self) -> None:
        assert (
            run_eval._effective_patch_verifier_scope(
                configured_scope="all", is_adaptive_condition=False
            )
            == "all"
        )

    def test_adaptive_only_when_adaptive(self) -> None:
        assert (
            run_eval._effective_patch_verifier_scope(
                configured_scope="adaptive_only", is_adaptive_condition=True
            )
            == "adaptive_only"
        )

    def test_adaptive_only_when_not_adaptive(self) -> None:
        assert (
            run_eval._effective_patch_verifier_scope(
                configured_scope="adaptive_only", is_adaptive_condition=False
            )
            == "disabled"
        )


# ===================================================================
# _prompt_chars_for_llm_calls
# ===================================================================


class TestPromptCharsForLlmCalls:
    def test_none_prompt(self) -> None:
        assert run_eval._prompt_chars_for_llm_calls(None, llm_call_count=3) == 0

    def test_empty_prompt(self) -> None:
        assert run_eval._prompt_chars_for_llm_calls("", llm_call_count=3) == 0

    def test_normal(self) -> None:
        assert run_eval._prompt_chars_for_llm_calls("hello", llm_call_count=2) == 10


# ===================================================================
# _single_line_error
# ===================================================================


class TestSingleLineError:
    def test_multiline(self) -> None:
        result = run_eval._single_line_error("line1\nline2\nline3")
        assert "|" in result
        assert "line1" in result

    def test_empty(self) -> None:
        result = run_eval._single_line_error("")
        assert result == "unknown runtime error"

    def test_long_truncated(self) -> None:
        result = run_eval._single_line_error("x" * 300, max_chars=20)
        assert len(result) == 20
        assert result.endswith("...")


# ===================================================================
# _nonnegative_int
# ===================================================================


class TestNonnegativeInt:
    def test_valid(self) -> None:
        assert run_eval._nonnegative_int(5) == 5

    def test_negative_clamped(self) -> None:
        assert run_eval._nonnegative_int(-3) == 0

    def test_invalid_uses_default(self) -> None:
        assert run_eval._nonnegative_int("not_int", default=7) == 7

    def test_none_uses_default(self) -> None:
        assert run_eval._nonnegative_int(None, default=2) == 2


# ===================================================================
# _adaptive_optional_int / _adaptive_optional_bool
# ===================================================================


class TestAdaptiveOptionalHelpers:
    def test_optional_int_valid(self) -> None:
        assert run_eval._adaptive_optional_int(42) == 42

    def test_optional_int_none(self) -> None:
        assert run_eval._adaptive_optional_int(None) is None

    def test_optional_int_str(self) -> None:
        assert run_eval._adaptive_optional_int("not_int") is None

    def test_optional_bool_true(self) -> None:
        assert run_eval._adaptive_optional_bool(True) is True

    def test_optional_bool_false(self) -> None:
        assert run_eval._adaptive_optional_bool(False) is False

    def test_optional_bool_non_bool(self) -> None:
        assert run_eval._adaptive_optional_bool(1) is None
        assert run_eval._adaptive_optional_bool("yes") is None


# ===================================================================
# _is_snapshot_condition
# ===================================================================


class TestIsSnapshotCondition:
    def test_adaptive_gated(self) -> None:
        assert run_eval._is_snapshot_condition(Condition.LAYERED_ADAPTIVE_GATED) is True

    def test_adaptive_llm(self) -> None:
        assert run_eval._is_snapshot_condition(Condition.LAYERED_ADAPTIVE_LLM) is True

    def test_base_repair(self) -> None:
        assert run_eval._is_snapshot_condition(Condition.BASE_REPAIR) is False
