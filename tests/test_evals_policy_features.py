"""Tests for eval_policy_features — feature extraction, reward, and helpers."""

from __future__ import annotations

import math

import numpy as np
import pytest

from evals.adaptive_budget_online_policy import (
    ACTION_BUDGET_FAMILY_CONTEXT,
    ACTION_BUDGET_FAMILY_TIME,
    ACTION_BUDGET_FAMILY_TOKENS,
    BUDGET_POLICY_ARCH_HIERARCHICAL_V1,
    BUDGET_POLICY_ARCH_TIER_BINARY,
)
from evals.adaptive_online_policy import ONLINE_POLICY_ACTIONS
from evals.eval_policy_features import (
    _adaptive_budget_control_changes,
    _adaptive_budget_control_profile_to_dict,
    _adaptive_budget_controls_changed,
    _adaptive_budget_discrete_widening_delta,
    _adaptive_budget_feature_vector,
    _adaptive_budget_level_multiplier,
    _adaptive_budget_level_step_delta,
    _adaptive_budget_local_reward_components,
    _adaptive_budget_next_tier_index,
    _adaptive_budget_recorded_multiplier,
    _adaptive_budget_union_change_flags,
    _adaptive_online_feature_vector,
    _adaptive_online_local_reward_components,
    _adaptive_policy_local_accounting,
    _apply_adaptive_budget_profile_to_patcher,
    _build_adaptive_budget_control_profile,
    _build_adaptive_budget_family_profile,
    _nonnegative_int,
)
from evals.eval_types import (
    ADAPTIVE_AUX_REWARD_COMPACTION_EFFICIENCY_WEIGHT_DEFAULT,
    ADAPTIVE_AUX_REWARD_EVIDENCE_NOVELTY_WEIGHT_DEFAULT,
    ADAPTIVE_AUX_REWARD_INFORMATIVENESS_ALIGNMENT_WEIGHT_DEFAULT,
    ADAPTIVE_AUX_REWARD_PYTEST_REDUCTION_WEIGHT_DEFAULT,
    ADAPTIVE_BUDGET_CONTROL_CHANGE_KEYS,
    ADAPTIVE_LOCAL_REWARD_HELPER_MATERIALIZATION_BONUS_SCALE,
    ADAPTIVE_LOCAL_REWARD_PROGRESS_HINT_SCALE,
    ADAPTIVE_LOCAL_REWARD_STAGNATION_PENALTY_SCALE,
    ADAPTIVE_LOCAL_REWARD_SUCCESS_HINT,
    AdaptiveBudgetControlProfile,
    PreGateResult,
)
from evals.policy_feature_scaling import (
    INTERACTION_FEATURE_FACTORS,
    is_count_like_feature,
    scale_feature_mapping,
)

# ── Helpers ──────────────────────────────────────────────────────────────


def _default_pregate() -> PreGateResult:
    return PreGateResult(
        should_skip=False,
        score=0.5,
        exception_type="AssertionError",
        reason="",
        f_type_specificity=0.5,
        f_message_diagnostic=0.5,
        f_crash_in_user_code=0.5,
        f_assertion_quality=0.5,
        f_frame_depth=0.5,
    )


def _online_feature_kwargs(**overrides):
    defaults = {
        "attempt": 1,
        "max_attempts": 5,
        "timed_out": False,
        "failure_signature_repeat": False,
        "stagnation_streak": 0,
        "helper_calls_used": 0,
        "helper_calls_without_context": 0,
        "adaptive_no_progress_streak": 0,
        "helper_budget_exhausted": False,
        "helper_saturation_guarded": False,
        "supports_context_request": True,
        "snapshot_available": False,
        "pregate_result": _default_pregate(),
        "effective_helper_budget_limit": 10,
        "adaptive_budget_unlimited": False,
        "adaptive_budget_topup_calls_used": 0,
        "adaptive_budget_topup_max_calls": 0,
        "last_helper_need_more_context": None,
        "last_helper_decision_reason": "",
        "last_helper_action_executed": "",
        "context_compaction_enabled": False,
        "supports_context_compaction": False,
        "context_compaction_calls_used": 0,
        "context_compaction_max_calls": 0,
        "last_context_compaction_requested": False,
        "last_context_compaction_invoked": False,
        "last_context_compaction_success": False,
        "last_context_compaction_ratio": 0.0,
        "last_context_compaction_trigger": "",
        "case_difficulty": "medium",
        "case_snapshot_dependency": "none",
        "case_bug_source": None,
    }
    defaults.update(overrides)
    return defaults


def _budget_feature_kwargs(**overrides):
    defaults = {
        "attempt": 1,
        "max_attempts": 5,
        "timed_out": False,
        "failure_signature_repeat": False,
        "stagnation_streak": 0,
        "helper_calls_used": 0,
        "helper_calls_without_context": 0,
        "adaptive_no_progress_streak": 0,
        "adaptive_no_progress_token_sum": 0,
        "helper_budget_exhausted": False,
        "helper_saturation_guarded": False,
        "supports_context_request": True,
        "snapshot_available": False,
        "pregate_result": _default_pregate(),
        "context_compaction_enabled": False,
        "supports_context_compaction": False,
        "context_compaction_calls_used": 0,
        "context_compaction_max_calls": 0,
        "last_context_compaction_requested": False,
        "last_context_compaction_invoked": False,
        "last_context_compaction_success": False,
        "last_context_compaction_ratio": 0.0,
        "last_context_compaction_trigger": "",
        "current_tier_index": 0,
        "max_tier_index": 2,
        "ups_used": 0,
        "max_ups_per_case": 3,
        "base_propose_budget_sec": 120.0,
        "next_tier_changes": {},
        "next_tier_multiplier_delta": 0.5,
        "last_helper_need_more_context": None,
        "last_helper_decision_reason": "",
    }
    defaults.update(overrides)
    return defaults


# ── _nonnegative_int ─────────────────────────────────────────────────────


class TestNonnegativeInt:
    def test_positive(self) -> None:
        assert _nonnegative_int(5) == 5

    def test_zero(self) -> None:
        assert _nonnegative_int(0) == 0

    def test_negative_clamps(self) -> None:
        assert _nonnegative_int(-3) == 0

    def test_float_truncates(self) -> None:
        assert _nonnegative_int(3.9) == 3

    def test_none_returns_default(self) -> None:
        assert _nonnegative_int(None, default=7) == 7

    def test_string_returns_default(self) -> None:
        assert _nonnegative_int("abc", default=4) == 4

    def test_negative_default_passes_through(self) -> None:
        # The shared evals.coerce helper only clamps coerced values, not the
        # fallback default, which is returned verbatim.
        assert _nonnegative_int(None, default=-2) == -2

    def test_numeric_string(self) -> None:
        assert _nonnegative_int("10") == 10


# ── _adaptive_online_feature_vector ──────────────────────────────────────


class TestOnlineFeatureVector:
    def test_returns_dict_of_floats(self) -> None:
        features = _adaptive_online_feature_vector(**_online_feature_kwargs())
        assert isinstance(features, dict)
        for key, val in features.items():
            assert isinstance(val, float), f"Non-float value for {key}: {val!r}"

    def test_bias_is_one(self) -> None:
        features = _adaptive_online_feature_vector(**_online_feature_kwargs())
        assert features["bias"] == 1.0

    def test_attempt_frac_boundary(self) -> None:
        features = _adaptive_online_feature_vector(
            **_online_feature_kwargs(attempt=5, max_attempts=5)
        )
        assert features["attempt_frac"] == pytest.approx(1.0)
        assert "attempts_remaining_frac" not in features  # always 1 - attempt_frac (POL-06)

    def test_attempt_frac_first_attempt(self) -> None:
        features = _adaptive_online_feature_vector(
            **_online_feature_kwargs(attempt=1, max_attempts=4)
        )
        assert features["attempt_frac"] == pytest.approx(0.25)

    def test_timed_out_flag(self) -> None:
        feat_on = _adaptive_online_feature_vector(**_online_feature_kwargs(timed_out=True))
        feat_off = _adaptive_online_feature_vector(**_online_feature_kwargs(timed_out=False))
        assert feat_on["timed_out"] == 1.0
        assert feat_off["timed_out"] == 0.0

    def test_helper_budget_remaining_frac_unlimited(self) -> None:
        features = _adaptive_online_feature_vector(
            **_online_feature_kwargs(adaptive_budget_unlimited=True)
        )
        assert features["helper_budget_remaining_frac"] == 1.0

    def test_helper_budget_remaining_frac_zero_limit(self) -> None:
        features = _adaptive_online_feature_vector(
            **_online_feature_kwargs(effective_helper_budget_limit=0)
        )
        assert features["helper_budget_remaining_frac"] == 0.0

    def test_helper_budget_remaining_frac_half(self) -> None:
        features = _adaptive_online_feature_vector(
            **_online_feature_kwargs(
                effective_helper_budget_limit=10,
                helper_calls_used=5,
            )
        )
        assert features["helper_budget_remaining_frac"] == pytest.approx(0.5)

    def test_topup_used_frac_zero_max(self) -> None:
        features = _adaptive_online_feature_vector(
            **_online_feature_kwargs(
                adaptive_budget_topup_max_calls=0,
                adaptive_budget_topup_calls_used=0,
            )
        )
        assert features["helper_topup_used_frac"] == 0.0
        assert features["helper_topup_remaining_frac"] == 0.0

    def test_topup_used_frac_normal(self) -> None:
        features = _adaptive_online_feature_vector(
            **_online_feature_kwargs(
                adaptive_budget_topup_max_calls=4,
                adaptive_budget_topup_calls_used=1,
            )
        )
        assert features["helper_topup_used_frac"] == pytest.approx(0.25)
        assert features["helper_topup_remaining_frac"] == pytest.approx(0.75)

    def test_difficulty_ordinals(self) -> None:
        for diff, expected in [("easy", 0.0), ("medium", 0.5), ("hard", 1.0)]:
            features = _adaptive_online_feature_vector(
                **_online_feature_kwargs(case_difficulty=diff)
            )
            assert features["case_difficulty_ord"] == expected, diff

    def test_difficulty_unknown_defaults_medium(self) -> None:
        features = _adaptive_online_feature_vector(
            **_online_feature_kwargs(case_difficulty="unknown")
        )
        assert features["case_difficulty_ord"] == 0.5

    def test_snapshot_dependency_ordinals(self) -> None:
        for dep, expected in [("none", 0.0), ("helpful", 0.5), ("required", 1.0)]:
            features = _adaptive_online_feature_vector(
                **_online_feature_kwargs(case_snapshot_dependency=dep)
            )
            assert features["case_snapshot_dependency_ord"] == expected

    def test_pregate_exception_flags(self) -> None:
        pregate_assert = _default_pregate()._replace(exception_type="AssertionError")
        features = _adaptive_online_feature_vector(
            **_online_feature_kwargs(pregate_result=pregate_assert)
        )
        assert features["pregate_exception_is_assertion"] == 1.0
        assert features["pregate_exception_is_timeout"] == 0.0
        assert features["pregate_exception_is_syntax"] == 0.0

    def test_pregate_timeout_flag(self) -> None:
        pregate = _default_pregate()._replace(exception_type="TimeoutError")
        features = _adaptive_online_feature_vector(**_online_feature_kwargs(pregate_result=pregate))
        assert features["pregate_exception_is_timeout"] == 1.0

    def test_pregate_syntax_flag(self) -> None:
        for exc_type in ("SyntaxError", "IndentationError", "TabError"):
            pregate = _default_pregate()._replace(exception_type=exc_type)
            features = _adaptive_online_feature_vector(
                **_online_feature_kwargs(pregate_result=pregate)
            )
            assert features["pregate_exception_is_syntax"] == 1.0, exc_type

    def test_last_helper_action_one_hot(self) -> None:
        # skip_helper is the reference level since POL-06 (no feature of its own); this
        # test used to expect all ten levels.
        assert "last_helper_action_skip_helper" not in _adaptive_online_feature_vector(
            **_online_feature_kwargs()
        )
        for action in ONLINE_POLICY_ACTIONS:
            features = _adaptive_online_feature_vector(
                **_online_feature_kwargs(last_helper_action_executed=action)
            )
            for a in ONLINE_POLICY_ACTIONS:
                if a == "skip_helper":
                    continue
                expected = 1.0 if a == action else 0.0
                assert features[f"last_helper_action_{a}"] == expected

    def test_bias_is_not_a_linear_combination_of_varying_features(self) -> None:
        """POL-06: two features were exact linear combinations of others.

        ``bias`` equalled the sum of the ten one-hot levels, and ``attempts_remaining_frac``
        was ``1 - attempt_frac``.
        """
        rows = [
            _adaptive_online_feature_vector(
                **_online_feature_kwargs(
                    last_helper_action_executed=action, attempt=attempt, max_attempts=4
                )
            )
            for action in ONLINE_POLICY_ACTIONS
            for attempt in (1, 2, 3, 4)
        ]
        varying = [
            name for name in rows[0] if name != "bias" and len({row[name] for row in rows}) > 1
        ]
        columns = ["bias", *varying]
        design = np.array([[row[name] for name in columns] for row in rows])
        assert np.linalg.matrix_rank(design) == len(columns)

    def test_interaction_profile_adds_curated_online_features(self) -> None:
        features = _adaptive_online_feature_vector(
            **_online_feature_kwargs(
                attempt=3,
                max_attempts=4,
                stagnation_streak=2,
                helper_calls_used=4,
                helper_calls_without_context=3,
                snapshot_available=True,
                feature_scaling_profile="v1_interactions",
            )
        )
        # POL-19: products of v1-scaled factors, under names that aren't count-like. They
        # used to multiply raw values (1.5, 0.5, 3.0) as attempt_frac_x_stagnation_streak,
        # pregate_score_x_snapshot_available and helper_calls_used_x_helper_low_yield_ratio.
        stagnation_scaled = math.log1p(2) / math.log1p(10)
        helper_calls_scaled = math.log1p(4) / math.log1p(8)
        assert features["ix_attempt_frac_x_stagnation"] == pytest.approx(0.75 * stagnation_scaled)
        assert features["ix_pregate_score_x_snapshot_available"] == pytest.approx(0.5)
        assert features["ix_helper_calls_used_x_low_yield_ratio"] == pytest.approx(
            helper_calls_scaled * 0.75
        )
        assert "attempt_frac_x_stagnation_streak" not in features
        scaled = scale_feature_mapping(features, profile="v1_interactions")
        for name in INTERACTION_FEATURE_FACTORS:
            assert not is_count_like_feature(name)
            assert scaled[name] == features[name]

    def test_bug_source_imported(self) -> None:
        features_hc = _adaptive_online_feature_vector(
            **_online_feature_kwargs(case_bug_source="hand_crafted")
        )
        features_imp = _adaptive_online_feature_vector(
            **_online_feature_kwargs(case_bug_source="imported")
        )
        assert features_hc["case_bug_source_imported"] == 0.0
        assert features_imp["case_bug_source_imported"] == 1.0

    def test_bug_source_none_defaults_hand_crafted(self) -> None:
        features = _adaptive_online_feature_vector(**_online_feature_kwargs(case_bug_source=None))
        assert features["case_bug_source_imported"] == 0.0

    def test_low_yield_ratio_zero_calls(self) -> None:
        features = _adaptive_online_feature_vector(
            **_online_feature_kwargs(helper_calls_used=0, helper_calls_without_context=0)
        )
        assert features["helper_low_yield_ratio"] == 0.0

    def test_low_yield_ratio_all_without_context(self) -> None:
        features = _adaptive_online_feature_vector(
            **_online_feature_kwargs(helper_calls_used=4, helper_calls_without_context=4)
        )
        assert features["helper_low_yield_ratio"] == pytest.approx(1.0)

    def test_context_compaction_features(self) -> None:
        features = _adaptive_online_feature_vector(
            **_online_feature_kwargs(
                context_compaction_enabled=True,
                supports_context_compaction=True,
                context_compaction_calls_used=1,
                context_compaction_max_calls=4,
                last_context_compaction_requested=True,
                last_context_compaction_invoked=True,
                last_context_compaction_success=True,
                last_context_compaction_ratio=0.6,
                last_context_compaction_trigger="stage_a_requested",
            )
        )
        assert features["context_compaction_enabled"] == 1.0
        assert features["supports_context_compaction"] == 1.0
        assert features["context_compaction_calls_used_frac"] == pytest.approx(0.25)
        assert features["context_compaction_calls_remaining_frac"] == pytest.approx(0.75)
        assert features["last_context_compaction_trigger_stage_a"] == 1.0
        assert features["last_context_compaction_trigger_overflow"] == 0.0

    def test_last_helper_decision_reason_flags(self) -> None:
        feat = _adaptive_online_feature_vector(
            **_online_feature_kwargs(last_helper_decision_reason="llm_invalid_response_no_helper")
        )
        assert feat["last_helper_invalid_response"] == 1.0
        assert feat["last_helper_declined"] == 0.0

        feat2 = _adaptive_online_feature_vector(
            **_online_feature_kwargs(last_helper_decision_reason="llm_declined_helper")
        )
        assert feat2["last_helper_declined"] == 1.0
        assert feat2["last_helper_invalid_response"] == 0.0

    def test_context_compaction_zero_max(self) -> None:
        features = _adaptive_online_feature_vector(
            **_online_feature_kwargs(
                context_compaction_calls_used=0,
                context_compaction_max_calls=0,
            )
        )
        assert features["context_compaction_calls_used_frac"] == 0.0
        assert features["context_compaction_calls_remaining_frac"] == 0.0


# ── _adaptive_budget_feature_vector ──────────────────────────────────────


class TestBudgetFeatureVector:
    def test_returns_dict_of_floats(self) -> None:
        features = _adaptive_budget_feature_vector(**_budget_feature_kwargs())
        assert isinstance(features, dict)
        for key, val in features.items():
            assert isinstance(val, float), f"Non-float value for {key}: {val!r}"

    def test_bias_is_one(self) -> None:
        features = _adaptive_budget_feature_vector(**_budget_feature_kwargs())
        assert features["bias"] == 1.0

    def test_attempt_progress_is_one_feature(self) -> None:
        """POL-06: ``attempts_remaining_frac`` was always ``1 - attempt_frac``."""
        rows = [
            _adaptive_budget_feature_vector(
                **_budget_feature_kwargs(attempt=attempt, max_attempts=4)
            )
            for attempt in (1, 2, 3, 4)
        ]
        assert "attempts_remaining_frac" not in rows[0]
        assert [row["attempt_frac"] for row in rows] == [0.25, 0.5, 0.75, 1.0]

    def test_tier_frac_at_max(self) -> None:
        features = _adaptive_budget_feature_vector(
            **_budget_feature_kwargs(current_tier_index=2, max_tier_index=2)
        )
        assert features["current_budget_tier_frac"] == pytest.approx(1.0)
        assert features["can_increase_budget_tier"] == 0.0

    def test_tier_frac_at_zero(self) -> None:
        features = _adaptive_budget_feature_vector(
            **_budget_feature_kwargs(current_tier_index=0, max_tier_index=2)
        )
        assert features["current_budget_tier_frac"] == pytest.approx(0.0)
        assert features["can_increase_budget_tier"] == 1.0

    def test_ups_used_frac(self) -> None:
        features = _adaptive_budget_feature_vector(
            **_budget_feature_kwargs(ups_used=1, max_ups_per_case=3)
        )
        assert features["budget_ups_used_frac"] == pytest.approx(1.0 / 3.0)
        assert features["budget_ups_remaining_frac"] == pytest.approx(2.0 / 3.0)

    def test_ups_zero_max(self) -> None:
        features = _adaptive_budget_feature_vector(
            **_budget_feature_kwargs(ups_used=0, max_ups_per_case=0)
        )
        assert features["budget_ups_used_frac"] == 0.0
        assert features["budget_ups_remaining_frac"] == 0.0

    def test_no_progress_token_sum_k(self) -> None:
        features = _adaptive_budget_feature_vector(
            **_budget_feature_kwargs(adaptive_no_progress_token_sum=5000)
        )
        assert features["adaptive_no_progress_token_sum_k"] == pytest.approx(5.0)

    def test_hierarchical_v1_flag(self) -> None:
        features = _adaptive_budget_feature_vector(
            **_budget_feature_kwargs(budget_architecture=BUDGET_POLICY_ARCH_HIERARCHICAL_V1)
        )
        assert features["budget_arch_hierarchical_v1"] == 1.0

    def test_tier_binary_flag(self) -> None:
        features = _adaptive_budget_feature_vector(
            **_budget_feature_kwargs(budget_architecture=BUDGET_POLICY_ARCH_TIER_BINARY)
        )
        assert features["budget_arch_hierarchical_v1"] == 0.0

    def test_family_level_fracs_with_explicit_levels(self) -> None:
        features = _adaptive_budget_feature_vector(
            **_budget_feature_kwargs(
                current_tier_index=0,
                max_tier_index=4,
                current_time_level=2,
                current_tokens_level=1,
                current_context_level=3,
            )
        )
        assert features["current_budget_time_level_frac"] == pytest.approx(0.5)
        assert features["current_budget_tokens_level_frac"] == pytest.approx(0.25)
        assert features["current_budget_context_level_frac"] == pytest.approx(0.75)

    def test_family_level_fracs_fallback_to_tier_index(self) -> None:
        features = _adaptive_budget_feature_vector(
            **_budget_feature_kwargs(
                current_tier_index=1,
                max_tier_index=2,
            )
        )
        assert features["current_budget_time_level_frac"] == pytest.approx(0.5)

    def test_next_tier_changes_flags(self) -> None:
        changes = {"propose_budget": True, "max_tokens": False, "helper_allowance": True}
        features = _adaptive_budget_feature_vector(
            **_budget_feature_kwargs(next_tier_changes=changes)
        )
        assert features["budget_can_raise_propose_budget"] == 1.0
        assert features["budget_can_raise_max_tokens"] == 0.0
        assert features["budget_can_raise_helper_allowance"] == 1.0

    def test_max_tier_index_zero(self) -> None:
        features = _adaptive_budget_feature_vector(
            **_budget_feature_kwargs(
                current_tier_index=0,
                max_tier_index=0,
            )
        )
        # When max_tier_index=0, tier_frac uses special branch
        assert features["current_budget_tier_frac"] == 0.0
        assert features["can_increase_budget_tier"] == 0.0


# ── _adaptive_online_local_reward_components ─────────────────────────────


class TestOnlineRewardComponents:
    def test_success_hint_when_success(self) -> None:
        _reward, success_hint, *_ = _adaptive_online_local_reward_components(
            success=True,
            helper_materialized_context=False,
            local_tokens_total=0,
            local_wall_sec=0.0,
            adaptive_progress_score=0.0,
            adaptive_no_progress_streak=0,
            lambda_tokens=0.0,
            mu_latency=0.0,
            token_scale=1.0,
            latency_scale=1.0,
            eta_progress=0.0,
            kappa_stagnation=0.0,
        )
        assert success_hint == pytest.approx(ADAPTIVE_LOCAL_REWARD_SUCCESS_HINT)

    def test_no_success_hint_when_failure(self) -> None:
        _, success_hint, *_ = _adaptive_online_local_reward_components(
            success=False,
            helper_materialized_context=False,
            local_tokens_total=0,
            local_wall_sec=0.0,
            adaptive_progress_score=0.0,
            adaptive_no_progress_streak=0,
            lambda_tokens=0.0,
            mu_latency=0.0,
            token_scale=1.0,
            latency_scale=1.0,
            eta_progress=0.0,
            kappa_stagnation=0.0,
        )
        assert success_hint == 0.0

    def test_progress_hint_proportional(self) -> None:
        _, _, progress_hint, *_ = _adaptive_online_local_reward_components(
            success=False,
            helper_materialized_context=False,
            local_tokens_total=0,
            local_wall_sec=0.0,
            adaptive_progress_score=0.8,
            adaptive_no_progress_streak=0,
            lambda_tokens=0.0,
            mu_latency=0.0,
            token_scale=1.0,
            latency_scale=1.0,
            eta_progress=1.0,
            kappa_stagnation=0.0,
        )
        expected = ADAPTIVE_LOCAL_REWARD_PROGRESS_HINT_SCALE * 1.0 * 0.8
        assert progress_hint == pytest.approx(expected)

    def test_token_penalty(self) -> None:
        _, _, _, token_penalty, *_ = _adaptive_online_local_reward_components(
            success=False,
            helper_materialized_context=False,
            local_tokens_total=500,
            local_wall_sec=0.0,
            adaptive_progress_score=0.0,
            adaptive_no_progress_streak=0,
            lambda_tokens=0.1,
            mu_latency=0.0,
            token_scale=1000.0,
            latency_scale=1.0,
            eta_progress=0.0,
            kappa_stagnation=0.0,
        )
        expected = 0.1 * (500.0 / 1000.0)
        assert token_penalty == pytest.approx(expected)

    def test_latency_penalty(self) -> None:
        _, _, _, _, latency_penalty, *_ = _adaptive_online_local_reward_components(
            success=False,
            helper_materialized_context=False,
            local_tokens_total=0,
            local_wall_sec=30.0,
            adaptive_progress_score=0.0,
            adaptive_no_progress_streak=0,
            lambda_tokens=0.0,
            mu_latency=0.2,
            token_scale=1.0,
            latency_scale=60.0,
            eta_progress=0.0,
            kappa_stagnation=0.0,
        )
        expected = 0.2 * (30.0 / 60.0)
        assert latency_penalty == pytest.approx(expected)

    def test_stagnation_penalty_caps_at_one(self) -> None:
        _, _, _, _, _, stagnation_penalty, *_ = _adaptive_online_local_reward_components(
            success=False,
            helper_materialized_context=False,
            local_tokens_total=0,
            local_wall_sec=0.0,
            adaptive_progress_score=0.0,
            adaptive_no_progress_streak=100,
            lambda_tokens=0.0,
            mu_latency=0.0,
            token_scale=1.0,
            latency_scale=1.0,
            eta_progress=0.0,
            kappa_stagnation=1.0,
        )
        expected = ADAPTIVE_LOCAL_REWARD_STAGNATION_PENALTY_SCALE * 1.0 * 1.0
        assert stagnation_penalty == pytest.approx(expected)

    def test_helper_materialization_bonus(self) -> None:
        _, _, _, _, _, _, bonus, *_ = _adaptive_online_local_reward_components(
            success=False,
            helper_materialized_context=True,
            local_tokens_total=0,
            local_wall_sec=0.0,
            adaptive_progress_score=0.0,
            adaptive_no_progress_streak=0,
            lambda_tokens=0.0,
            mu_latency=0.0,
            token_scale=1.0,
            latency_scale=1.0,
            eta_progress=1.0,
            kappa_stagnation=0.0,
        )
        expected = ADAPTIVE_LOCAL_REWARD_HELPER_MATERIALIZATION_BONUS_SCALE * 1.0
        assert bonus == pytest.approx(expected)

    def test_helper_materialization_bonus_zero_when_not_materialized(self) -> None:
        _, _, _, _, _, _, bonus, *_ = _adaptive_online_local_reward_components(
            success=False,
            helper_materialized_context=False,
            local_tokens_total=0,
            local_wall_sec=0.0,
            adaptive_progress_score=0.0,
            adaptive_no_progress_streak=0,
            lambda_tokens=0.0,
            mu_latency=0.0,
            token_scale=1.0,
            latency_scale=1.0,
            eta_progress=1.0,
            kappa_stagnation=0.0,
        )
        assert bonus == 0.0

    def test_reward_sums_correctly(self) -> None:
        (
            reward,
            success_hint,
            progress_hint,
            token_penalty,
            latency_penalty,
            stagnation_penalty,
            mat_bonus,
            aux_total,
            _aux_pytest,
            _aux_novelty,
            _aux_compaction,
            _aux_informativeness,
        ) = _adaptive_online_local_reward_components(
            success=True,
            helper_materialized_context=True,
            local_tokens_total=100,
            local_wall_sec=10.0,
            adaptive_progress_score=0.5,
            adaptive_no_progress_streak=1,
            lambda_tokens=0.05,
            mu_latency=0.05,
            token_scale=1000.0,
            latency_scale=60.0,
            eta_progress=0.5,
            kappa_stagnation=0.5,
        )
        expected = (
            success_hint
            + progress_hint
            + mat_bonus
            + aux_total
            - token_penalty
            - latency_penalty
            - stagnation_penalty
        )
        assert reward == pytest.approx(expected)


class TestAuxiliaryRewardComponents:
    """Auxiliary reward signals for online policy."""

    # Base kwargs to reduce duplication
    _BASE: dict[str, object] = {
        "success": False,
        "helper_materialized_context": False,
        "local_tokens_total": 0,
        "local_wall_sec": 0.0,
        "adaptive_progress_score": 0.0,
        "adaptive_no_progress_streak": 0,
        "lambda_tokens": 0.0,
        "mu_latency": 0.0,
        "token_scale": 1.0,
        "latency_scale": 1.0,
        "eta_progress": 0.0,
        "kappa_stagnation": 0.0,
    }

    def test_aux_disabled_by_default_no_aux_contribution(self) -> None:
        (
            reward,
            *_,
            aux_total,
            aux_pytest,
            aux_novelty,
            aux_compaction,
            aux_informativeness,
        ) = _adaptive_online_local_reward_components(
            **self._BASE,  # type: ignore[arg-type]
            enable_auxiliary_rewards=False,
            aux_pytest_fail_reduction=0.8,
            aux_evidence_novelty_ratio=0.5,
            aux_compaction_ratio=0.6,
            aux_informativeness_alignment=1.0,
        )
        assert aux_total == 0.0
        # Clamped signals are still computed even when disabled.
        assert aux_pytest == pytest.approx(0.8)
        assert aux_novelty == pytest.approx(0.5)
        assert aux_compaction == pytest.approx(0.6)
        assert aux_informativeness == pytest.approx(1.0)
        # Reward should equal zero (all base penalties/bonuses are zero).
        assert reward == pytest.approx(0.0)

    def test_aux_enabled_adds_weighted_sum(self) -> None:
        (
            reward,
            _success_hint,
            _progress_hint,
            _token_penalty,
            _latency_penalty,
            _stagnation_penalty,
            _mat_bonus,
            aux_total,
            aux_pytest,
            aux_novelty,
            aux_compaction,
            aux_informativeness,
        ) = _adaptive_online_local_reward_components(
            **self._BASE,  # type: ignore[arg-type]
            enable_auxiliary_rewards=True,
            aux_pytest_fail_reduction=0.8,
            aux_evidence_novelty_ratio=0.5,
            aux_compaction_ratio=0.6,
            aux_informativeness_alignment=1.0,
        )
        expected_aux = (
            ADAPTIVE_AUX_REWARD_PYTEST_REDUCTION_WEIGHT_DEFAULT * 0.8
            + ADAPTIVE_AUX_REWARD_EVIDENCE_NOVELTY_WEIGHT_DEFAULT * 0.5
            + ADAPTIVE_AUX_REWARD_COMPACTION_EFFICIENCY_WEIGHT_DEFAULT * 0.6
            + ADAPTIVE_AUX_REWARD_INFORMATIVENESS_ALIGNMENT_WEIGHT_DEFAULT * 1.0
        )
        assert aux_total == pytest.approx(expected_aux)
        assert aux_total == pytest.approx(0.3)
        assert aux_pytest == pytest.approx(0.8)
        assert aux_novelty == pytest.approx(0.5)
        assert aux_compaction == pytest.approx(0.6)
        assert aux_informativeness == pytest.approx(1.0)
        # reward == aux_total when all base components are zero.
        assert reward == pytest.approx(aux_total)

    def test_aux_signals_clamped_to_unit_interval(self) -> None:
        (
            *_,
            _aux_total,
            aux_pytest,
            aux_novelty,
            aux_compaction,
            aux_informativeness,
        ) = _adaptive_online_local_reward_components(
            **self._BASE,  # type: ignore[arg-type]
            enable_auxiliary_rewards=True,
            aux_pytest_fail_reduction=-0.5,
            aux_evidence_novelty_ratio=1.5,
            aux_compaction_ratio=-0.1,
            aux_informativeness_alignment=2.0,
        )
        assert aux_pytest == pytest.approx(0.0)
        assert aux_novelty == pytest.approx(1.0)
        assert aux_compaction == pytest.approx(0.0)
        assert aux_informativeness == pytest.approx(1.0)

    def test_aux_zero_signals_produce_zero_aux_total(self) -> None:
        *_, aux_total, _, _, _, _ = _adaptive_online_local_reward_components(
            **self._BASE,  # type: ignore[arg-type]
            enable_auxiliary_rewards=True,
            aux_pytest_fail_reduction=0.0,
            aux_evidence_novelty_ratio=0.0,
            aux_compaction_ratio=0.0,
            aux_informativeness_alignment=0.0,
        )
        assert aux_total == 0.0

    def test_aux_custom_weights(self) -> None:
        *_, aux_total, aux_pytest, aux_novelty, aux_compaction, aux_informativeness = (
            _adaptive_online_local_reward_components(
                **self._BASE,  # type: ignore[arg-type]
                enable_auxiliary_rewards=True,
                aux_pytest_fail_reduction=1.0,
                aux_evidence_novelty_ratio=1.0,
                aux_compaction_ratio=1.0,
                aux_informativeness_alignment=1.0,
                aux_pytest_reduction_weight=0.25,
                aux_evidence_novelty_weight=0.20,
                aux_compaction_efficiency_weight=0.10,
                aux_informativeness_alignment_weight=0.15,
            )
        )
        expected_aux = 0.25 + 0.20 + 0.10 + 0.15
        assert aux_total == pytest.approx(expected_aux)
        assert aux_pytest == pytest.approx(1.0)
        assert aux_novelty == pytest.approx(1.0)
        assert aux_compaction == pytest.approx(1.0)
        assert aux_informativeness == pytest.approx(1.0)

    def test_aux_backward_compat_identical_reward(self) -> None:
        """With aux disabled, reward is identical to a call without aux params."""
        reward_with_aux_params, *_ = _adaptive_online_local_reward_components(
            **self._BASE,  # type: ignore[arg-type]
            enable_auxiliary_rewards=False,
            aux_pytest_fail_reduction=0.9,
            aux_evidence_novelty_ratio=0.7,
            aux_compaction_ratio=0.4,
            aux_informativeness_alignment=0.6,
        )
        reward_without_aux_params, *_ = _adaptive_online_local_reward_components(
            **self._BASE,  # type: ignore[arg-type]
        )
        assert reward_with_aux_params == pytest.approx(reward_without_aux_params)


# ── _adaptive_budget_local_reward_components ─────────────────────────────


class TestBudgetRewardComponents:
    def test_budget_exceeded_penalty(self) -> None:
        _reward, _, _, _, _, _, budget_penalty, *_ = _adaptive_budget_local_reward_components(
            success=False,
            local_tokens_total=0,
            local_wall_sec=0.0,
            adaptive_progress_score=0.0,
            adaptive_no_progress_streak=0,
            lambda_tokens=0.0,
            mu_latency=0.0,
            token_scale=1.0,
            latency_scale=1.0,
            eta_progress=0.0,
            kappa_stagnation=0.0,
            rho_budget_exceeded=0.5,
            request_budget_exceeded=True,
        )
        assert budget_penalty == pytest.approx(0.5)

    def test_no_budget_exceeded_penalty(self) -> None:
        _, _, _, _, _, _, budget_penalty, *_ = _adaptive_budget_local_reward_components(
            success=False,
            local_tokens_total=0,
            local_wall_sec=0.0,
            adaptive_progress_score=0.0,
            adaptive_no_progress_streak=0,
            lambda_tokens=0.0,
            mu_latency=0.0,
            token_scale=1.0,
            latency_scale=1.0,
            eta_progress=0.0,
            kappa_stagnation=0.0,
            rho_budget_exceeded=0.5,
            request_budget_exceeded=False,
        )
        assert budget_penalty == 0.0


class TestBudgetAuxiliaryRewardComponents:
    """Auxiliary reward signals for budget policy."""

    _BASE: dict[str, object] = {
        "success": False,
        "local_tokens_total": 0,
        "local_wall_sec": 0.0,
        "adaptive_progress_score": 0.0,
        "adaptive_no_progress_streak": 0,
        "lambda_tokens": 0.0,
        "mu_latency": 0.0,
        "token_scale": 1.0,
        "latency_scale": 1.0,
        "eta_progress": 0.0,
        "kappa_stagnation": 0.0,
        "rho_budget_exceeded": 0.0,
        "request_budget_exceeded": False,
    }

    def test_budget_aux_disabled_no_contribution(self) -> None:
        (
            reward,
            *_,
            aux_total,
            aux_pytest,
            aux_novelty,
        ) = _adaptive_budget_local_reward_components(
            **self._BASE,  # type: ignore[arg-type]
            enable_auxiliary_rewards=False,
            aux_pytest_fail_reduction=0.8,
            aux_evidence_novelty_ratio=0.5,
        )
        assert aux_total == 0.0
        # Clamped signals are still computed even when disabled.
        assert aux_pytest == pytest.approx(0.8)
        assert aux_novelty == pytest.approx(0.5)
        # Reward should equal zero (all base penalties/bonuses are zero).
        assert reward == pytest.approx(0.0)

    def test_budget_aux_enabled_adds_weighted_sum(self) -> None:
        (
            reward,
            _success_hint,
            _progress_hint,
            _token_penalty,
            _latency_penalty,
            _stagnation_penalty,
            _budget_exceeded_penalty,
            aux_total,
            aux_pytest,
            aux_novelty,
        ) = _adaptive_budget_local_reward_components(
            **self._BASE,  # type: ignore[arg-type]
            enable_auxiliary_rewards=True,
            aux_pytest_fail_reduction=0.8,
            aux_evidence_novelty_ratio=0.5,
        )
        expected_aux = (
            ADAPTIVE_AUX_REWARD_PYTEST_REDUCTION_WEIGHT_DEFAULT * 0.8
            + ADAPTIVE_AUX_REWARD_EVIDENCE_NOVELTY_WEIGHT_DEFAULT * 0.5
        )
        assert aux_total == pytest.approx(expected_aux)
        assert aux_total == pytest.approx(0.17)
        assert aux_pytest == pytest.approx(0.8)
        assert aux_novelty == pytest.approx(0.5)
        # reward == aux_total when all base components are zero.
        assert reward == pytest.approx(aux_total)

    def test_budget_aux_signals_clamped(self) -> None:
        *_, _aux_total, aux_pytest, aux_novelty = _adaptive_budget_local_reward_components(
            **self._BASE,  # type: ignore[arg-type]
            enable_auxiliary_rewards=True,
            aux_pytest_fail_reduction=-0.3,
            aux_evidence_novelty_ratio=1.7,
        )
        assert aux_pytest == pytest.approx(0.0)
        assert aux_novelty == pytest.approx(1.0)

    def test_budget_aux_backward_compat(self) -> None:
        """With aux disabled, reward is identical to a call without aux params."""
        reward_with_aux_params, *_ = _adaptive_budget_local_reward_components(
            **self._BASE,  # type: ignore[arg-type]
            enable_auxiliary_rewards=False,
            aux_pytest_fail_reduction=0.9,
            aux_evidence_novelty_ratio=0.7,
        )
        reward_without_aux_params, *_ = _adaptive_budget_local_reward_components(
            **self._BASE,  # type: ignore[arg-type]
        )
        assert reward_with_aux_params == pytest.approx(reward_without_aux_params)


# ── _adaptive_policy_local_accounting ────────────────────────────────────


class TestLocalAccounting:
    def test_defaults(self) -> None:
        acc = _adaptive_policy_local_accounting()
        assert acc.llm_call_count == 0
        assert acc.tokens_total == 0
        assert acc.wall_sec == 0.0

    def test_negative_values_clamp(self) -> None:
        acc = _adaptive_policy_local_accounting(
            llm_call_count=-1,
            tokens_prompt=-5,
            tokens_completion=-3,
            wall_sec=-1.0,
        )
        assert acc.llm_call_count == 0
        assert acc.tokens_prompt == 0
        assert acc.tokens_completion == 0
        assert acc.wall_sec == 0.0

    def test_total_default_sum(self) -> None:
        acc = _adaptive_policy_local_accounting(
            tokens_prompt=100,
            tokens_completion=50,
            tokens_total=None,
        )
        assert acc.tokens_total == 150

    def test_total_explicit(self) -> None:
        acc = _adaptive_policy_local_accounting(
            tokens_prompt=100,
            tokens_completion=50,
            tokens_total=200,
        )
        assert acc.tokens_total == 200


# ── _adaptive_budget_control_profile_to_dict ─────────────────────────────


class TestControlProfileToDict:
    def test_round_trip_keys(self) -> None:
        profile = AdaptiveBudgetControlProfile(
            propose_budget_multiplier=1.5,
            propose_budget_sec=180.0,
            openai_max_tokens=4096,
            helper_max_calls=5,
            stage_a_max_requests=3,
            stage_a_max_chars=10000,
            retry_history_max_chars=5000,
            context_compaction_max_calls=2,
            response_mode="edits",
        )
        d = _adaptive_budget_control_profile_to_dict(profile)
        assert d["propose_budget_multiplier"] == 1.5
        assert d["openai_max_tokens"] == 4096
        assert d["response_mode"] == "edits"


# ── _adaptive_budget_control_changes ─────────────────────────────────────


class TestControlChanges:
    def test_no_changes_all_false(self) -> None:
        profile = AdaptiveBudgetControlProfile()
        changes = _adaptive_budget_control_changes(current=profile, target=profile)
        for v in changes.values():
            assert v is False

    def test_changes_detected(self) -> None:
        current = AdaptiveBudgetControlProfile(propose_budget_sec=100.0)
        target = AdaptiveBudgetControlProfile(propose_budget_sec=200.0)
        changes = _adaptive_budget_control_changes(current=current, target=target)
        assert changes["propose_budget"] is True


class TestControlsChanged:
    def test_all_false_returns_false(self) -> None:
        flags = dict.fromkeys(ADAPTIVE_BUDGET_CONTROL_CHANGE_KEYS, False)
        assert _adaptive_budget_controls_changed(flags) is False

    def test_one_true_returns_true(self) -> None:
        flags = dict.fromkeys(ADAPTIVE_BUDGET_CONTROL_CHANGE_KEYS, False)
        flags["propose_budget"] = True
        assert _adaptive_budget_controls_changed(flags) is True


# ── _build_adaptive_budget_control_profile ───────────────────────────────


class TestBuildControlProfile:
    def test_basic_multiplier(self) -> None:
        profile = _build_adaptive_budget_control_profile(
            tier_index=1,
            base_tier_index=0,
            tier_multipliers=[1.0, 1.5, 2.0],
            base_propose_budget_sec=100.0,
            base_openai_max_tokens=4096,
            base_helper_max_calls=5,
            base_stage_a_max_requests=2,
            base_stage_a_max_chars=10000,
            base_retry_history_max_chars=5000,
            base_context_compaction_max_calls=2,
            context_compaction_enabled=True,
            base_response_mode="edits",
            is_openai_patcher=True,
        )
        assert profile.propose_budget_sec == pytest.approx(150.0)
        assert profile.propose_budget_multiplier == pytest.approx(1.5)
        assert profile.helper_max_calls == 6  # base 5 + delta 1
        assert profile.stage_a_max_requests == 3  # base 2 + delta 1

    def test_empty_tier_multipliers_defaults_to_one(self) -> None:
        profile = _build_adaptive_budget_control_profile(
            tier_index=0,
            base_tier_index=0,
            tier_multipliers=[],
            base_propose_budget_sec=100.0,
            base_openai_max_tokens=0,
            base_helper_max_calls=0,
            base_stage_a_max_requests=1,
            base_stage_a_max_chars=100,
            base_retry_history_max_chars=0,
            base_context_compaction_max_calls=0,
            context_compaction_enabled=False,
            base_response_mode="",
            is_openai_patcher=False,
        )
        assert profile.propose_budget_multiplier == pytest.approx(1.0)

    def test_response_mode_switch_at_max_tier(self) -> None:
        profile = _build_adaptive_budget_control_profile(
            tier_index=2,
            base_tier_index=0,
            tier_multipliers=[1.0, 1.5, 2.0],
            base_propose_budget_sec=100.0,
            base_openai_max_tokens=4096,
            base_helper_max_calls=5,
            base_stage_a_max_requests=2,
            base_stage_a_max_chars=10000,
            base_retry_history_max_chars=5000,
            base_context_compaction_max_calls=2,
            context_compaction_enabled=True,
            base_response_mode="edits",
            is_openai_patcher=True,
        )
        assert profile.response_mode == "diff"

    def test_no_response_mode_switch_non_openai(self) -> None:
        profile = _build_adaptive_budget_control_profile(
            tier_index=2,
            base_tier_index=0,
            tier_multipliers=[1.0, 1.5, 2.0],
            base_propose_budget_sec=100.0,
            base_openai_max_tokens=0,
            base_helper_max_calls=5,
            base_stage_a_max_requests=2,
            base_stage_a_max_chars=10000,
            base_retry_history_max_chars=5000,
            base_context_compaction_max_calls=2,
            context_compaction_enabled=False,
            base_response_mode="edits",
            is_openai_patcher=False,
        )
        assert profile.response_mode == "edits"

    def test_zero_helper_calls_stays_zero(self) -> None:
        profile = _build_adaptive_budget_control_profile(
            tier_index=2,
            base_tier_index=0,
            tier_multipliers=[1.0, 1.5, 2.0],
            base_propose_budget_sec=100.0,
            base_openai_max_tokens=0,
            base_helper_max_calls=0,
            base_stage_a_max_requests=1,
            base_stage_a_max_chars=100,
            base_retry_history_max_chars=0,
            base_context_compaction_max_calls=0,
            context_compaction_enabled=False,
            base_response_mode="",
            is_openai_patcher=False,
        )
        assert profile.helper_max_calls == 0

    def test_openai_max_tokens_zero_for_non_openai(self) -> None:
        profile = _build_adaptive_budget_control_profile(
            tier_index=1,
            base_tier_index=0,
            tier_multipliers=[1.0, 2.0],
            base_propose_budget_sec=100.0,
            base_openai_max_tokens=4096,
            base_helper_max_calls=5,
            base_stage_a_max_requests=2,
            base_stage_a_max_chars=10000,
            base_retry_history_max_chars=5000,
            base_context_compaction_max_calls=2,
            context_compaction_enabled=False,
            base_response_mode="",
            is_openai_patcher=False,
        )
        assert profile.openai_max_tokens == 0

    def test_dense_ladder_widens_discrete_limits_by_one_at_first_coarse_jump(self) -> None:
        """POL-10: any step above base widens by at least 1 (this used to defer to 0)."""
        profile = _build_adaptive_budget_control_profile(
            tier_index=2,
            base_tier_index=0,
            tier_multipliers=[1.0, 1.1, 1.2, 1.35, 1.5, 1.75, 2.0],
            base_propose_budget_sec=100.0,
            base_openai_max_tokens=4096,
            base_helper_max_calls=5,
            base_stage_a_max_requests=2,
            base_stage_a_max_chars=10000,
            base_retry_history_max_chars=5000,
            base_context_compaction_max_calls=1,
            context_compaction_enabled=True,
            base_response_mode="edits",
            is_openai_patcher=True,
        )
        assert profile.propose_budget_multiplier == pytest.approx(1.2)
        assert profile.helper_max_calls == 6
        assert profile.stage_a_max_requests == 3
        assert profile.context_compaction_max_calls == 2

    def test_dense_ladder_caps_discrete_widening_at_plus_two(self) -> None:
        profile = _build_adaptive_budget_control_profile(
            tier_index=6,
            base_tier_index=0,
            tier_multipliers=[1.0, 1.1, 1.2, 1.35, 1.5, 1.75, 2.0],
            base_propose_budget_sec=100.0,
            base_openai_max_tokens=4096,
            base_helper_max_calls=5,
            base_stage_a_max_requests=2,
            base_stage_a_max_chars=10000,
            base_retry_history_max_chars=5000,
            base_context_compaction_max_calls=1,
            context_compaction_enabled=True,
            base_response_mode="edits",
            is_openai_patcher=True,
        )
        assert profile.helper_max_calls == 7
        assert profile.stage_a_max_requests == 4
        assert profile.context_compaction_max_calls == 3

    def test_two_rung_ladder_only_widens_discrete_limits_by_one(self) -> None:
        profile = _build_adaptive_budget_control_profile(
            tier_index=1,
            base_tier_index=0,
            tier_multipliers=[1.0, 1.5],
            base_propose_budget_sec=100.0,
            base_openai_max_tokens=4096,
            base_helper_max_calls=5,
            base_stage_a_max_requests=2,
            base_stage_a_max_chars=10000,
            base_retry_history_max_chars=5000,
            base_context_compaction_max_calls=1,
            context_compaction_enabled=True,
            base_response_mode="edits",
            is_openai_patcher=True,
        )
        assert profile.propose_budget_multiplier == pytest.approx(1.5)
        assert profile.helper_max_calls == 6
        assert profile.stage_a_max_requests == 3
        assert profile.context_compaction_max_calls == 2


# ── _build_adaptive_budget_family_profile ────────────────────────────────


class TestBuildFamilyProfile:
    def test_family_profile_independent_levels(self) -> None:
        profile = _build_adaptive_budget_family_profile(
            time_level=2,
            tokens_level=0,
            context_level=1,
            base_tier_index=0,
            tier_multipliers=[1.0, 1.5, 2.0],
            base_propose_budget_sec=100.0,
            base_openai_max_tokens=4096,
            base_helper_max_calls=5,
            base_stage_a_max_requests=2,
            base_stage_a_max_chars=10000,
            base_retry_history_max_chars=5000,
            base_context_compaction_max_calls=2,
            context_compaction_enabled=False,
            base_response_mode="",
            is_openai_patcher=True,
        )
        # time_level=2 → multiplier 2.0
        assert profile.propose_budget_sec == pytest.approx(200.0)
        # tokens_level=0 → multiplier 1.0
        assert profile.openai_max_tokens == 4096
        # context_level=1 → delta=1
        assert profile.helper_max_calls == 6

    def test_dense_context_family_widens_discrete_limits_from_the_first_step(self) -> None:
        """POL-10: the context family used to keep its discrete limits flat until later levels.

        So with max_ups_per_case=1 it changed nothing.
        """
        profile = _build_adaptive_budget_family_profile(
            time_level=0,
            tokens_level=0,
            context_level=2,
            base_tier_index=0,
            tier_multipliers=[1.0, 1.1, 1.2, 1.35, 1.5, 1.75, 2.0],
            base_propose_budget_sec=100.0,
            base_openai_max_tokens=4096,
            base_helper_max_calls=5,
            base_stage_a_max_requests=2,
            base_stage_a_max_chars=10000,
            base_retry_history_max_chars=5000,
            base_context_compaction_max_calls=1,
            context_compaction_enabled=True,
            base_response_mode="",
            is_openai_patcher=True,
        )
        assert profile.helper_max_calls == 6
        assert profile.stage_a_max_requests == 3
        assert profile.stage_a_max_chars == 12000
        assert profile.context_compaction_max_calls == 2

    def test_default_ladder_context_family_first_step_changes_discrete_limits(self) -> None:
        """POL-10 regression: default ladder, one step up (max_ups_per_case=1)."""
        profile = _build_adaptive_budget_family_profile(
            time_level=0,
            tokens_level=0,
            context_level=1,
            base_tier_index=0,
            tier_multipliers=[1.0, 1.1, 1.2, 1.35, 1.5],
            base_propose_budget_sec=100.0,
            base_openai_max_tokens=4096,
            base_helper_max_calls=5,
            base_stage_a_max_requests=2,
            base_stage_a_max_chars=10000,
            base_retry_history_max_chars=5000,
            base_context_compaction_max_calls=1,
            context_compaction_enabled=True,
            base_response_mode="",
            is_openai_patcher=True,
        )
        assert profile.helper_max_calls == 6
        assert profile.stage_a_max_requests == 3
        assert profile.context_compaction_max_calls == 2

    def test_two_rung_context_family_only_widens_discrete_limits_by_one(self) -> None:
        profile = _build_adaptive_budget_family_profile(
            time_level=0,
            tokens_level=0,
            context_level=1,
            base_tier_index=0,
            tier_multipliers=[1.0, 2.0],
            base_propose_budget_sec=100.0,
            base_openai_max_tokens=4096,
            base_helper_max_calls=5,
            base_stage_a_max_requests=2,
            base_stage_a_max_chars=10000,
            base_retry_history_max_chars=5000,
            base_context_compaction_max_calls=1,
            context_compaction_enabled=True,
            base_response_mode="",
            is_openai_patcher=True,
        )
        assert profile.helper_max_calls == 6
        assert profile.stage_a_max_requests == 3
        assert profile.stage_a_max_chars == 20000
        assert profile.context_compaction_max_calls == 2


# ── _adaptive_budget_union_change_flags ──────────────────────────────────


class TestUnionChangeFlags:
    def test_empty_maps(self) -> None:
        merged = _adaptive_budget_union_change_flags({})
        for v in merged.values():
            assert v is False

    def test_union_merges_flags(self) -> None:
        merged = _adaptive_budget_union_change_flags(
            {
                "time": {"propose_budget": True, "max_tokens": False},
                "context": {"propose_budget": False, "max_tokens": True},
            }
        )
        assert merged["propose_budget"] is True
        assert merged["max_tokens"] is True


class TestBudgetTierHelpers:
    def test_next_tier_index_uses_coarse_dense_targets(self) -> None:
        multipliers = [1.0, 1.1, 1.2, 1.35, 1.5, 1.75, 2.0]
        assert (
            _adaptive_budget_next_tier_index(current_tier_index=0, tier_multipliers=multipliers)
            == 2
        )
        assert (
            _adaptive_budget_next_tier_index(current_tier_index=2, tier_multipliers=multipliers)
            == 4
        )
        assert (
            _adaptive_budget_next_tier_index(current_tier_index=4, tier_multipliers=multipliers)
            == 6
        )

    def test_next_tier_index_falls_back_to_next_larger_tier_when_target_missing(self) -> None:
        multipliers = [1.0, 1.1, 1.35, 1.75]
        assert (
            _adaptive_budget_next_tier_index(current_tier_index=0, tier_multipliers=multipliers)
            == 2
        )
        assert (
            _adaptive_budget_next_tier_index(current_tier_index=2, tier_multipliers=multipliers)
            == 3
        )

    def test_discrete_widening_delta_caps_at_plus_two(self) -> None:
        assert _adaptive_budget_discrete_widening_delta(level=1, base_level=0, max_level=1) == 1
        # POL-10: at least 1 above base (level 2 of 6 used to give 0).
        assert _adaptive_budget_discrete_widening_delta(level=2, base_level=0, max_level=6) == 1
        assert _adaptive_budget_discrete_widening_delta(level=1, base_level=0, max_level=4) == 1
        assert _adaptive_budget_discrete_widening_delta(level=0, base_level=0, max_level=4) == 0
        assert _adaptive_budget_discrete_widening_delta(level=4, base_level=0, max_level=6) == 1
        assert _adaptive_budget_discrete_widening_delta(level=6, base_level=0, max_level=6) == 2


# ── _adaptive_budget_level_step_delta / _level_multiplier ────────────────


class TestLevelStepDelta:
    def test_step_delta_normal(self) -> None:
        delta = _adaptive_budget_level_step_delta(
            current_level=0,
            max_tier_index=2,
            tier_multipliers=[1.0, 1.5, 2.0],
        )
        assert delta == pytest.approx(0.5)

    def test_step_delta_at_max(self) -> None:
        delta = _adaptive_budget_level_step_delta(
            current_level=2,
            max_tier_index=2,
            tier_multipliers=[1.0, 1.5, 2.0],
        )
        assert delta == pytest.approx(0.0)

    def test_step_delta_empty_multipliers(self) -> None:
        delta = _adaptive_budget_level_step_delta(
            current_level=0,
            max_tier_index=0,
            tier_multipliers=[],
        )
        assert delta == 0.0


class TestLevelMultiplier:
    def test_normal_lookup(self) -> None:
        m = _adaptive_budget_level_multiplier(
            level=1,
            tier_multipliers=[1.0, 1.5, 2.0],
            fallback=1.0,
        )
        assert m == pytest.approx(1.5)

    def test_empty_multipliers_returns_fallback(self) -> None:
        m = _adaptive_budget_level_multiplier(
            level=0,
            tier_multipliers=[],
            fallback=3.0,
        )
        assert m == pytest.approx(3.0)

    def test_level_clamped_to_bounds(self) -> None:
        m = _adaptive_budget_level_multiplier(
            level=99,
            tier_multipliers=[1.0, 2.0],
            fallback=1.0,
        )
        assert m == pytest.approx(2.0)


# ── _adaptive_budget_recorded_multiplier ─────────────────────────────────


class TestRecordedMultiplier:
    def test_known_family(self) -> None:
        m = _adaptive_budget_recorded_multiplier(
            family=ACTION_BUDGET_FAMILY_TIME,
            family_levels={
                ACTION_BUDGET_FAMILY_TIME: 1,
                ACTION_BUDGET_FAMILY_TOKENS: 0,
                ACTION_BUDGET_FAMILY_CONTEXT: 0,
            },
            tier_multipliers=[1.0, 2.0, 3.0],
            fallback=1.0,
        )
        assert m == pytest.approx(2.0)

    def test_unknown_family_returns_max(self) -> None:
        m = _adaptive_budget_recorded_multiplier(
            family="unknown",
            family_levels={
                ACTION_BUDGET_FAMILY_TIME: 2,
                ACTION_BUDGET_FAMILY_TOKENS: 0,
                ACTION_BUDGET_FAMILY_CONTEXT: 1,
            },
            tier_multipliers=[1.0, 2.0, 3.0],
            fallback=1.0,
        )
        assert m == pytest.approx(3.0)


# ── _apply_adaptive_budget_profile_to_patcher ────────────────────────────


class TestApplyProfileToPatcher:
    def test_non_openai_returns_identity(self) -> None:
        profile = AdaptiveBudgetControlProfile()
        patcher, _ret_profile, errors = _apply_adaptive_budget_profile_to_patcher(
            patcher="dummy",
            applied_profile=profile,
            base_propose_budget_sec=100.0,
            base_openai_max_tokens=4096,
            base_response_mode="edits",
            is_openai_patcher=False,
        )
        assert patcher == "dummy"
        assert errors == {}

    def test_replace_failure_reverts_profile(self) -> None:
        """When patcher replace() fails, errors dict is populated."""
        profile = AdaptiveBudgetControlProfile(
            propose_budget_sec=200.0,
            openai_max_tokens=8192,
            response_mode="diff",
        )
        # A plain string doesn't support dataclasses.replace
        patcher, ret_profile, errors = _apply_adaptive_budget_profile_to_patcher(
            patcher="not_a_dataclass",
            applied_profile=profile,
            base_propose_budget_sec=100.0,
            base_openai_max_tokens=4096,
            base_response_mode="edits",
            is_openai_patcher=True,
        )
        assert patcher == "not_a_dataclass"
        assert "max_propose_sec" in errors
        assert ret_profile.propose_budget_sec == pytest.approx(100.0)
        assert ret_profile.openai_max_tokens == 4096
        assert ret_profile.response_mode == "edits"
