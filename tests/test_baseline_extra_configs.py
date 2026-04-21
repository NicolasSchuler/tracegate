"""Unit tests for baselines.extra_configs typed extra-config validation."""

from __future__ import annotations

import logging

import pytest

from baselines.extra_configs import (
    BaselineExtraConfigError,
    IntervenorExtraConfig,
    LambdaRLMExtraConfig,
    LDBExtraConfig,
    RLMExtraConfig,
    TraceCoderExtraConfig,
)


class TestLDBExtraConfig:
    def test_empty_extra_succeeds(self) -> None:
        result = LDBExtraConfig.validate({})
        assert isinstance(result, LDBExtraConfig)
        assert result.baseline_openai_compat_profile == "auto"
        assert result.baseline_openai_allow_reasoning_fallback is True
        assert result.baseline_openai_enable_thinking is True
        assert result.baseline_hybrid_guidance_mode == "tuned_v1"

    def test_unknown_keys_warn(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            LDBExtraConfig.validate({"foo": "bar"}, strict=False)
        assert "ignoring unrecognised" in caplog.text
        assert "foo" in caplog.text

    def test_unknown_keys_raise_in_strict_mode(self) -> None:
        with pytest.raises(BaselineExtraConfigError, match=r"unknown config\.extra keys"):
            LDBExtraConfig.validate({"foo": "bar"}, strict=True)


class TestIntervenorExtraConfig:
    def test_empty_extra_succeeds(self) -> None:
        result = IntervenorExtraConfig.validate({})
        assert isinstance(result, IntervenorExtraConfig)
        assert result.baseline_openai_compat_profile == "auto"
        assert result.baseline_openai_allow_reasoning_fallback is True
        assert result.baseline_openai_enable_thinking is True
        assert result.baseline_intervenor_teacher_max_tokens == 1024
        assert result.baseline_intervenor_learner_max_tokens == 2048
        assert result.baseline_intervenor_teacher_guidance_max_chars == 4000
        assert result.baseline_intervenor_learner_guidance_max_chars == 6000
        assert result.baseline_intervenor_append_output_contract is True

    def test_known_intervenor_knobs_are_validated(self) -> None:
        result = IntervenorExtraConfig.validate(
            {
                "baseline_intervenor_teacher_max_tokens": 333,
                "baseline_intervenor_learner_max_tokens": 777,
                "baseline_intervenor_teacher_guidance_max_chars": 1200,
                "baseline_intervenor_learner_guidance_max_chars": 2500,
                "baseline_intervenor_append_output_contract": False,
            }
        )
        assert result.baseline_intervenor_teacher_max_tokens == 333
        assert result.baseline_intervenor_learner_max_tokens == 777
        assert result.baseline_intervenor_teacher_guidance_max_chars == 1200
        assert result.baseline_intervenor_learner_guidance_max_chars == 2500
        assert result.baseline_intervenor_append_output_contract is False

    def test_unknown_keys_warn(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            IntervenorExtraConfig.validate({"unexpected": 1}, strict=False)
        assert "ignoring unrecognised" in caplog.text


class TestTraceCoderExtraConfig:
    def test_valid_key(self) -> None:
        result = TraceCoderExtraConfig.validate(
            {
                "tracecoder_fake_responses_path": "/tmp/responses.json",
                "baseline_openai_compat_profile": "qwen_openai_raw",
                "baseline_openai_allow_reasoning_fallback": True,
                "baseline_tracecoder_controller_guidance_max_chars": 3210,
                "baseline_tracecoder_max_no_improvement_streak": 4,
            }
        )
        assert result.tracecoder_fake_responses_path == "/tmp/responses.json"
        assert result.baseline_openai_compat_profile == "qwen_openai_raw"
        assert result.baseline_openai_allow_reasoning_fallback is True
        assert result.baseline_tracecoder_controller_guidance_max_chars == 3210
        assert result.baseline_tracecoder_max_no_improvement_streak == 4
        assert result.baseline_tracecoder_analysis_max_tokens == 4096
        assert result.baseline_tracecoder_implementation_max_tokens == 4096

    def test_empty_extra_succeeds(self) -> None:
        result = TraceCoderExtraConfig.validate({})
        assert result.tracecoder_fake_responses_path == ""
        assert result.baseline_tracecoder_controller_guidance_max_chars == 2000
        assert result.baseline_tracecoder_max_no_improvement_streak == 2
        assert result.baseline_tracecoder_analysis_max_tokens == 4096
        assert result.baseline_tracecoder_implementation_max_tokens == 4096
        assert result.baseline_tracecoder_problem_summary_max_chars == 4000
        assert result.baseline_tracecoder_exact_test_context_max_chars == 3000
        assert result.baseline_tracecoder_failure_feedback_max_chars == 6000
        assert result.baseline_tracecoder_failure_feedback_tail_max_chars == 2000
        assert result.baseline_tracecoder_instrumentation_output_max_lines == 80
        assert result.baseline_tracecoder_instrumentation_output_max_chars == 12000

    def test_tracecoder_compaction_and_stage_knobs_are_validated(self) -> None:
        result = TraceCoderExtraConfig.validate(
            {
                "baseline_tracecoder_analysis_max_tokens": 2048,
                "baseline_tracecoder_implementation_max_tokens": 4096,
                "baseline_tracecoder_problem_summary_max_chars": 3500,
                "baseline_tracecoder_exact_test_context_max_chars": 2800,
                "baseline_tracecoder_failure_feedback_max_chars": 5000,
                "baseline_tracecoder_failure_feedback_tail_max_chars": 1500,
                "baseline_tracecoder_instrumentation_output_max_lines": 60,
                "baseline_tracecoder_instrumentation_output_max_chars": 9000,
            }
        )
        assert result.baseline_tracecoder_analysis_max_tokens == 2048
        assert result.baseline_tracecoder_implementation_max_tokens == 4096
        assert result.baseline_tracecoder_problem_summary_max_chars == 3500
        assert result.baseline_tracecoder_exact_test_context_max_chars == 2800
        assert result.baseline_tracecoder_failure_feedback_max_chars == 5000
        assert result.baseline_tracecoder_failure_feedback_tail_max_chars == 1500
        assert result.baseline_tracecoder_instrumentation_output_max_lines == 60
        assert result.baseline_tracecoder_instrumentation_output_max_chars == 9000

    def test_unknown_key_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            TraceCoderExtraConfig.validate({"tracecoder_wrong_key": "x"}, strict=False)
        assert "tracecoder_wrong_key" in caplog.text


class TestLambdaRLMExtraConfig:
    def test_all_known_keys(self) -> None:
        extra = {
            "baseline_lambda_context_window_chars": 8000,
            "baseline_lambda_accuracy_target": 0.9,
            "baseline_lambda_a_leaf": 0.5,
            "baseline_lambda_a_compose": 0.3,
            "baseline_lambda_verbose": True,
            "baseline_lambda_temperature": 0.7,
            "baseline_lambda_top_p": 0.95,
            "baseline_lambda_max_tokens": 1024,
            "baseline_lambda_stream": False,
            "baseline_lambda_timeout_sec": 60.0,
            "baseline_lambda_controller_guidance_max_chars": 2400,
            "baseline_lambda_failure_feedback_max_chars": 3200,
            "baseline_lambda_failure_feedback_meta_max_chars": 1800,
            "baseline_lambda_force_task_type": "qa",
            "baseline_lambda_prompt_contract": "repair_fair_v1",
        }
        result = LambdaRLMExtraConfig.validate(extra)
        assert result.baseline_lambda_context_window_chars == 8000
        assert result.baseline_lambda_accuracy_target == 0.9
        assert result.baseline_lambda_verbose is True
        assert result.baseline_lambda_temperature == 0.7
        assert result.baseline_lambda_controller_guidance_max_chars == 2400
        assert result.baseline_lambda_failure_feedback_max_chars == 3200
        assert result.baseline_lambda_failure_feedback_meta_max_chars == 1800
        assert result.baseline_lambda_force_task_type == "qa"
        assert result.baseline_lambda_prompt_contract == "repair_fair_v1"
        assert result.baseline_openai_compat_profile == "auto"

    def test_unknown_key_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            LambdaRLMExtraConfig.validate({"baseline_lambda_bad": 1}, strict=False)
        assert "baseline_lambda_bad" in caplog.text

    def test_none_optional_float(self) -> None:
        result = LambdaRLMExtraConfig.validate({})
        assert result.baseline_lambda_temperature is None
        assert result.baseline_lambda_timeout_sec is None
        assert result.baseline_lambda_controller_guidance_max_chars == 2500
        assert result.baseline_lambda_failure_feedback_max_chars == 3500
        assert result.baseline_lambda_failure_feedback_meta_max_chars == 2000
        assert result.baseline_lambda_force_task_type == ""
        assert result.baseline_lambda_prompt_contract == "canonical"

    def test_invalid_prompt_contract_raises(self) -> None:
        with pytest.raises(BaselineExtraConfigError, match="baseline_lambda_prompt_contract"):
            LambdaRLMExtraConfig.validate({"baseline_lambda_prompt_contract": "fast_mode"})


class TestRLMExtraConfig:
    def test_all_known_keys(self) -> None:
        extra = {
            "baseline_rlm_max_depth": 3,
            "baseline_rlm_max_iterations": 10,
            "baseline_rlm_max_errors": 5,
            "baseline_rlm_compaction": True,
            "baseline_rlm_compaction_threshold_pct": 0.75,
            "baseline_rlm_verbose": False,
            "baseline_rlm_max_total_tokens": 50000,
            "baseline_rlm_temperature": 0.8,
            "baseline_rlm_top_p": 0.9,
            "baseline_rlm_max_tokens": 2048,
            "baseline_rlm_timeout_sec": 120.0,
            "baseline_rlm_controller_guidance_max_chars": 2100,
            "baseline_rlm_failure_feedback_max_chars": 3100,
            "baseline_rlm_failure_feedback_meta_max_chars": 1700,
            "baseline_rlm_prompt_contract": "repair_fair_v1",
        }
        result = RLMExtraConfig.validate(extra)
        assert result.baseline_rlm_max_depth == 3
        assert result.baseline_rlm_compaction is True
        assert result.baseline_rlm_temperature == 0.8
        assert result.baseline_rlm_controller_guidance_max_chars == 2100
        assert result.baseline_rlm_failure_feedback_max_chars == 3100
        assert result.baseline_rlm_failure_feedback_meta_max_chars == 1700
        assert result.baseline_rlm_prompt_contract == "repair_fair_v1"
        assert result.baseline_openai_allow_reasoning_fallback is True

    def test_enable_thinking_propagates(self) -> None:
        result = RLMExtraConfig.validate({"baseline_openai_enable_thinking": True})
        assert result.baseline_openai_enable_thinking is True

    def test_unknown_key_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        with caplog.at_level(logging.WARNING):
            RLMExtraConfig.validate({"baseline_rlm_unknown": 99}, strict=False)
        assert "baseline_rlm_unknown" in caplog.text

    def test_empty_defaults(self) -> None:
        result = RLMExtraConfig.validate({})
        assert result.baseline_rlm_max_depth == 0
        assert result.baseline_rlm_temperature is None
        assert result.baseline_rlm_controller_guidance_max_chars == 2500
        assert result.baseline_rlm_failure_feedback_max_chars == 3500
        assert result.baseline_rlm_failure_feedback_meta_max_chars == 2000
        assert result.baseline_rlm_prompt_contract == "canonical"

    def test_invalid_prompt_contract_raises(self) -> None:
        with pytest.raises(BaselineExtraConfigError, match="baseline_rlm_prompt_contract"):
            RLMExtraConfig.validate({"baseline_rlm_prompt_contract": "fast_mode"})


class TestEnableThinkingPropagation:
    """Verify baseline_openai_enable_thinking is accepted and stored by all configs."""

    @pytest.mark.parametrize(
        "config_cls",
        [
            LDBExtraConfig,
            IntervenorExtraConfig,
            TraceCoderExtraConfig,
            LambdaRLMExtraConfig,
            RLMExtraConfig,
        ],
        ids=["ldb", "intervenor", "tracecoder", "lambda_rlm", "rlm"],
    )
    def test_enable_thinking_true_accepted(self, config_cls: type) -> None:  # type: ignore[type-arg]
        result = config_cls.validate({"baseline_openai_enable_thinking": True})
        assert result.baseline_openai_enable_thinking is True

    @pytest.mark.parametrize(
        "config_cls",
        [
            LDBExtraConfig,
            IntervenorExtraConfig,
            TraceCoderExtraConfig,
            LambdaRLMExtraConfig,
            RLMExtraConfig,
        ],
        ids=["ldb", "intervenor", "tracecoder", "lambda_rlm", "rlm"],
    )
    def test_enable_thinking_true_by_default(self, config_cls: type) -> None:  # type: ignore[type-arg]
        result = config_cls.validate({})
        assert result.baseline_openai_enable_thinking is True


class TestHybridGuidanceModeValidation:
    @pytest.mark.parametrize(
        "mode",
        ["fair_v1", "fixed_v1"],
    )
    @pytest.mark.parametrize(
        "config_cls",
        [
            LDBExtraConfig,
            IntervenorExtraConfig,
            TraceCoderExtraConfig,
            LambdaRLMExtraConfig,
            RLMExtraConfig,
        ],
        ids=["ldb", "intervenor", "tracecoder", "lambda_rlm", "rlm"],
    )
    def test_supported_modes_are_accepted(self, config_cls: type, mode: str) -> None:  # type: ignore[type-arg]
        result = config_cls.validate({"baseline_hybrid_guidance_mode": mode})
        assert result.baseline_hybrid_guidance_mode == mode

    @pytest.mark.parametrize(
        "config_cls",
        [
            LDBExtraConfig,
            IntervenorExtraConfig,
            TraceCoderExtraConfig,
            LambdaRLMExtraConfig,
            RLMExtraConfig,
        ],
        ids=["ldb", "intervenor", "tracecoder", "lambda_rlm", "rlm"],
    )
    def test_invalid_mode_raises(self, config_cls: type) -> None:  # type: ignore[type-arg]
        with pytest.raises(BaselineExtraConfigError, match="baseline_hybrid_guidance_mode"):
            config_cls.validate({"baseline_hybrid_guidance_mode": "fast_mode"})
