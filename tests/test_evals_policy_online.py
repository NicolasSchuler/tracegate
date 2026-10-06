"""Tests for eval_policy_online — coercion helpers, reward, finalization."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from evals.adaptive_online_policy import (
    ACTION_INVOKE_HELPER,
    ACTION_SKIP_HELPER,
    ONLINE_POLICY_ACTIONS,
    PolicyDecision,
)
from evals.eval_policy_online import (
    _adaptive_online_behavior_fields,
    _adaptive_online_delayed_rescue_bonus,
    _adaptive_online_reward_breakdown,
    _adaptive_online_should_defer_credit,
    _adaptive_online_update_suppression_reason,
    _adaptive_optional_int,
    _finalize_adaptive_online_attempt,
    _nonnegative_int,
    _record_adaptive_online_policy_outcome,
    _sync_record_metadata_copies,
)
from evals.eval_types import (
    ADAPTIVE_DELAYED_RESCUE_HORIZON_ATTEMPTS,
    ADAPTIVE_HELPER_OUTCOME_COMPAT_REASONING_ONLY,
    ADAPTIVE_HELPER_OUTCOME_INVALID_PAYLOAD,
    ADAPTIVE_HELPER_OUTCOME_MATERIALIZED,
    ADAPTIVE_HELPER_OUTCOME_NOT_INVOKED,
    ADAPTIVE_HELPER_OUTCOME_TRANSPORT_FAILED,
    ADAPTIVE_HELPER_OUTCOME_UNAVAILABLE_BACKEND,
    ADAPTIVE_LOCAL_REWARD_DELAYED_RESCUE_BONUS_BASE,
    ADAPTIVE_LOCAL_REWARD_DELAYED_RESCUE_BONUS_MAX,
    ADAPTIVE_LOCAL_REWARD_DELAYED_RESCUE_CONTEXT_MULTIPLIER,
    ADAPTIVE_LOCAL_REWARD_DELAYED_RESCUE_HORIZON_TWO_FACTOR,
    ADAPTIVE_LOCAL_REWARD_SUCCESS_HINT,
    ADAPTIVE_ONLINE_REWARD_SCOPE,
    EMPTY_ADAPTIVE_POLICY_LOCAL_ACCOUNTING,
    AdaptiveOnlineRewardBreakdown,
    AdaptivePolicyLocalAccounting,
)

# ── _nonnegative_int ─────────────────────────────────────────────────────


class TestNonnegativeInt:
    def test_positive_int(self) -> None:
        assert _nonnegative_int(5) == 5

    def test_zero(self) -> None:
        assert _nonnegative_int(0) == 0

    def test_negative_clamps(self) -> None:
        assert _nonnegative_int(-3) == 0

    def test_none_uses_default(self) -> None:
        assert _nonnegative_int(None, default=7) == 7

    def test_bad_string_uses_default(self) -> None:
        assert _nonnegative_int("abc", default=4) == 4

    def test_negative_default_clamps(self) -> None:
        assert _nonnegative_int(None, default=-5) == 0

    def test_numeric_string(self) -> None:
        assert _nonnegative_int("10") == 10

    def test_float_truncates(self) -> None:
        assert _nonnegative_int(3.9) == 3


# ── _adaptive_optional_int ───────────────────────────────────────────────


class TestAdaptiveOptionalInt:
    def test_none_returns_none(self) -> None:
        assert _adaptive_optional_int(None) is None

    def test_int_returns_int(self) -> None:
        assert _adaptive_optional_int(5) == 5

    def test_negative_int_passes_through(self) -> None:
        assert _adaptive_optional_int(-3) == -3

    def test_float_truncates(self) -> None:
        assert _adaptive_optional_int(3.9) == 3

    def test_bad_string_returns_none(self) -> None:
        assert _adaptive_optional_int("abc") is None

    def test_numeric_string(self) -> None:
        assert _adaptive_optional_int("10") == 10

    def test_empty_list_returns_none(self) -> None:
        # list cannot be int()
        assert _adaptive_optional_int([]) is None


# ── _adaptive_online_delayed_rescue_bonus ────────────────────────────────


class TestDelayedRescueBonus:
    def test_horizon_zero_returns_zero(self) -> None:
        bonus = _adaptive_online_delayed_rescue_bonus(
            origin_attempt=3,
            resolved_attempt=3,
            failure_signature_repeat=False,
            case_difficulty="easy",
            case_snapshot_dependency="none",
        )
        assert bonus == 0.0

    def test_horizon_negative_returns_zero(self) -> None:
        bonus = _adaptive_online_delayed_rescue_bonus(
            origin_attempt=5,
            resolved_attempt=3,
            failure_signature_repeat=False,
            case_difficulty="easy",
            case_snapshot_dependency="none",
        )
        assert bonus == 0.0

    def test_horizon_exceeds_max_returns_zero(self) -> None:
        bonus = _adaptive_online_delayed_rescue_bonus(
            origin_attempt=1,
            resolved_attempt=1 + ADAPTIVE_DELAYED_RESCUE_HORIZON_ATTEMPTS + 1,
            failure_signature_repeat=False,
            case_difficulty="easy",
            case_snapshot_dependency="none",
        )
        assert bonus == 0.0

    def test_horizon_one_base_bonus(self) -> None:
        bonus = _adaptive_online_delayed_rescue_bonus(
            origin_attempt=1,
            resolved_attempt=2,
            failure_signature_repeat=False,
            case_difficulty="easy",
            case_snapshot_dependency="none",
        )
        assert bonus == pytest.approx(ADAPTIVE_LOCAL_REWARD_DELAYED_RESCUE_BONUS_BASE)

    def test_horizon_two_applies_factor(self) -> None:
        bonus = _adaptive_online_delayed_rescue_bonus(
            origin_attempt=1,
            resolved_attempt=3,
            failure_signature_repeat=False,
            case_difficulty="easy",
            case_snapshot_dependency="none",
        )
        expected = (
            ADAPTIVE_LOCAL_REWARD_DELAYED_RESCUE_BONUS_BASE
            * ADAPTIVE_LOCAL_REWARD_DELAYED_RESCUE_HORIZON_TWO_FACTOR
        )
        assert bonus == pytest.approx(expected)

    def test_context_multiplier_for_signature_repeat(self) -> None:
        bonus = _adaptive_online_delayed_rescue_bonus(
            origin_attempt=1,
            resolved_attempt=2,
            failure_signature_repeat=True,
            case_difficulty="easy",
            case_snapshot_dependency="none",
        )
        expected = (
            ADAPTIVE_LOCAL_REWARD_DELAYED_RESCUE_BONUS_BASE
            * ADAPTIVE_LOCAL_REWARD_DELAYED_RESCUE_CONTEXT_MULTIPLIER
        )
        assert bonus == pytest.approx(min(expected, ADAPTIVE_LOCAL_REWARD_DELAYED_RESCUE_BONUS_MAX))

    def test_context_multiplier_for_hard_difficulty(self) -> None:
        # case_difficulty is retained for backward compat but no longer triggers multiplier
        bonus = _adaptive_online_delayed_rescue_bonus(
            origin_attempt=1,
            resolved_attempt=2,
            failure_signature_repeat=False,
            case_difficulty="hard",
            case_snapshot_dependency="none",
        )
        expected = ADAPTIVE_LOCAL_REWARD_DELAYED_RESCUE_BONUS_BASE
        assert bonus == pytest.approx(expected)

    def test_context_multiplier_for_snapshot_dependency_helpful(self) -> None:
        # case_snapshot_dependency is retained for backward compat but no longer triggers multiplier
        bonus = _adaptive_online_delayed_rescue_bonus(
            origin_attempt=1,
            resolved_attempt=2,
            failure_signature_repeat=False,
            case_difficulty="easy",
            case_snapshot_dependency="helpful",
        )
        expected = ADAPTIVE_LOCAL_REWARD_DELAYED_RESCUE_BONUS_BASE
        assert bonus == pytest.approx(expected)

    def test_bonus_capped_at_max(self) -> None:
        bonus = _adaptive_online_delayed_rescue_bonus(
            origin_attempt=1,
            resolved_attempt=2,
            failure_signature_repeat=True,
            case_difficulty="hard",
            case_snapshot_dependency="required",
        )
        assert bonus <= ADAPTIVE_LOCAL_REWARD_DELAYED_RESCUE_BONUS_MAX


# ── _adaptive_online_behavior_fields ─────────────────────────────────────


def _make_policy_decision(**overrides: Any) -> PolicyDecision:
    """Build a minimal PolicyDecision for testing."""
    defaults = {
        "decision_id": "test-decision",
        "mode": "on",
        "algorithm": "linucb",
        "decision_index": 0,
        "created_at_utc": "2024-01-01T00:00:00Z",
        "selected_action": ACTION_INVOKE_HELPER,
        "applied_action": ACTION_INVOKE_HELPER,
        "default_action": ACTION_INVOKE_HELPER,
        "propensity": 0.8,
        "decision_strategy": "ucb",
        "scores": dict.fromkeys(ONLINE_POLICY_ACTIONS, 0.0),
        "feature_values": {"bias": 1.0},
        "available_actions": ONLINE_POLICY_ACTIONS,
        "unavailable_action_reasons": {},
        "feature_scaling_caps": {},
        "feature_scaling_history_sizes": {},
        "feature_scaling_fallback_to_v1": False,
        "configured_alpha": 0.0,
        "configured_epsilon": 0.0,
        "configured_decay": 1.0,
        "configured_action_score_biases": dict.fromkeys(ONLINE_POLICY_ACTIONS, 0.0),
        "effective_epsilon": 0.0,
        "warmup_active": False,
    }
    defaults.update(overrides)
    return PolicyDecision(**defaults)


class TestBehaviorFields:
    def test_none_decision(self) -> None:
        action, propensity = _adaptive_online_behavior_fields(
            decision=None,
            behavior_action="invoke_helper",
        )
        assert action == "none"
        assert propensity == 1.0

    def test_matching_action(self) -> None:
        decision = _make_policy_decision(
            selected_action=ACTION_INVOKE_HELPER,
            propensity=0.6,
        )
        action, propensity = _adaptive_online_behavior_fields(
            decision=decision,
            behavior_action=ACTION_INVOKE_HELPER,
        )
        assert action == ACTION_INVOKE_HELPER
        assert propensity == pytest.approx(0.6)

    def test_non_matching_action(self) -> None:
        decision = _make_policy_decision(
            selected_action=ACTION_SKIP_HELPER,
            propensity=0.6,
        )
        action, propensity = _adaptive_online_behavior_fields(
            decision=decision,
            behavior_action=ACTION_INVOKE_HELPER,
        )
        assert action == ACTION_INVOKE_HELPER
        assert propensity == 1.0

    def test_invalid_behavior_action(self) -> None:
        decision = _make_policy_decision()
        action, propensity = _adaptive_online_behavior_fields(
            decision=decision,
            behavior_action="nonsense_action",
        )
        assert action == "none"
        assert propensity == 1.0

    def test_shadow_mode_behavior_is_deterministic(self) -> None:
        """Shadow mode executes the baseline, so the coincident selection has probability 1."""
        decision = _make_policy_decision(
            mode="shadow",
            selected_action=ACTION_INVOKE_HELPER,
            applied_action=ACTION_INVOKE_HELPER,
            propensity=0.1,
        )
        action, propensity = _adaptive_online_behavior_fields(
            decision=decision,
            behavior_action=ACTION_INVOKE_HELPER,
        )
        assert action == ACTION_INVOKE_HELPER
        assert propensity == 1.0


# ── _adaptive_online_update_suppression_reason ───────────────────────────


class TestUpdateSuppressionReason:
    def test_transport_failed(self) -> None:
        reason = _adaptive_online_update_suppression_reason(
            ADAPTIVE_HELPER_OUTCOME_TRANSPORT_FAILED
        )
        assert reason == "helper_transport_no_update"

    def test_unavailable_backend(self) -> None:
        reason = _adaptive_online_update_suppression_reason(
            ADAPTIVE_HELPER_OUTCOME_UNAVAILABLE_BACKEND
        )
        assert reason == "helper_backend_unavailable_no_update"

    def test_invalid_payload(self) -> None:
        reason = _adaptive_online_update_suppression_reason(ADAPTIVE_HELPER_OUTCOME_INVALID_PAYLOAD)
        assert reason == "helper_invalid_payload_no_update"

    def test_compat_reasoning_only(self) -> None:
        reason = _adaptive_online_update_suppression_reason(
            ADAPTIVE_HELPER_OUTCOME_COMPAT_REASONING_ONLY
        )
        assert reason == "helper_compat_reasoning_only_no_update"

    def test_no_suppression_for_materialized(self) -> None:
        reason = _adaptive_online_update_suppression_reason(ADAPTIVE_HELPER_OUTCOME_MATERIALIZED)
        assert reason == ""

    def test_no_suppression_for_not_invoked(self) -> None:
        reason = _adaptive_online_update_suppression_reason(ADAPTIVE_HELPER_OUTCOME_NOT_INVOKED)
        assert reason == ""


# ── _sync_record_metadata_copies ─────────────────────────────────────────


class TestSyncRecordMetadataCopies:
    def test_none_record_is_noop(self) -> None:
        _sync_record_metadata_copies(
            record=None,
            attempt_metadata={"key": "val"},
            patch_meta={"other": "val"},
        )

    def test_dict_record_gets_copies(self) -> None:
        record: dict[str, Any] = {"patch_meta": {}}
        attempt = {"a": 1}
        patch = {"b": 2}
        _sync_record_metadata_copies(
            record=record,
            attempt_metadata=attempt,
            patch_meta=patch,
        )
        assert record["attempt_metadata"] == {"a": 1}
        assert record["patch_meta"] == {"b": 2}
        # Verify deep copy (mutation doesn't propagate)
        attempt["a"] = 999
        assert record["attempt_metadata"]["a"] == 1

    def test_dict_record_without_patch_meta_key(self) -> None:
        record: dict[str, Any] = {}
        _sync_record_metadata_copies(
            record=record,
            attempt_metadata={"x": 1},
            patch_meta={"y": 2},
        )
        assert record["attempt_metadata"] == {"x": 1}
        # patch_meta is not set if key wasn't already present
        assert "patch_meta" not in record


# ── _adaptive_online_reward_breakdown ────────────────────────────────────


class TestOnlineRewardBreakdown:
    def test_basic_fields(self) -> None:
        bd = _adaptive_online_reward_breakdown(
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
        assert isinstance(bd, AdaptiveOnlineRewardBreakdown)
        assert bd.reward_scope == ADAPTIVE_ONLINE_REWARD_SCOPE
        assert bd.reward_success_hint == pytest.approx(ADAPTIVE_LOCAL_REWARD_SUCCESS_HINT)

    def test_delayed_rescue_bonus_added(self) -> None:
        bd = _adaptive_online_reward_breakdown(
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
            delayed_rescue_bonus=0.3,
            resolved_attempt=5,
            horizon_attempts=2,
            resolution_reason="signature_match",
        )
        assert bd.reward_delayed_rescue_bonus == pytest.approx(0.3)
        assert bd.reward_credit_resolved_attempt == 5
        assert bd.reward_credit_horizon_attempts == 2
        assert bd.reward_credit_resolution_reason == "signature_match"
        # Reward should include the bonus
        assert bd.reward == pytest.approx(bd.reward - 0.3 + 0.3)


# ── _record_adaptive_online_policy_outcome ───────────────────────────────


class TestRecordPolicyOutcome:
    def test_none_decision_returns_policy_off(self) -> None:
        updated, event_logged, update_reason, _override = _record_adaptive_online_policy_outcome(
            adaptive_online_policy_controller=None,
            decision=None,
            run_id="r",
            case_id="c",
            condition_value="adaptive",
            attempt=1,
            note="test",
            success=False,
            reward_breakdown=AdaptiveOnlineRewardBreakdown(
                reward_scope="s",
                reward=0.0,
                reward_success_hint=0.0,
                reward_progress_hint=0.0,
                reward_progress_score=0.0,
                reward_token_penalty=0.0,
                reward_latency_penalty=0.0,
                reward_stagnation_penalty=0.0,
                reward_helper_materialization_bonus=0.0,
                reward_delayed_rescue_bonus=0.0,
                reward_credit_resolved_attempt=0,
                reward_credit_horizon_attempts=0,
                reward_credit_resolution_reason="",
            ),
            local_accounting=EMPTY_ADAPTIVE_POLICY_LOCAL_ACCOUNTING,
            override_reason="",
            behavior_action="none",
            behavior_propensity=1.0,
            allow_model_update=True,
        )
        assert updated is False
        assert event_logged is False
        assert update_reason == "policy_off"

    def test_none_controller_returns_unavailable(self) -> None:
        decision = _make_policy_decision()
        updated, event_logged, update_reason, _override = _record_adaptive_online_policy_outcome(
            adaptive_online_policy_controller=None,
            decision=decision,
            run_id="r",
            case_id="c",
            condition_value="adaptive",
            attempt=1,
            note="test",
            success=False,
            reward_breakdown=AdaptiveOnlineRewardBreakdown(
                reward_scope="s",
                reward=0.0,
                reward_success_hint=0.0,
                reward_progress_hint=0.0,
                reward_progress_score=0.0,
                reward_token_penalty=0.0,
                reward_latency_penalty=0.0,
                reward_stagnation_penalty=0.0,
                reward_helper_materialization_bonus=0.0,
                reward_delayed_rescue_bonus=0.0,
                reward_credit_resolved_attempt=0,
                reward_credit_horizon_attempts=0,
                reward_credit_resolution_reason="",
            ),
            local_accounting=EMPTY_ADAPTIVE_POLICY_LOCAL_ACCOUNTING,
            override_reason="",
            behavior_action="none",
            behavior_propensity=1.0,
            allow_model_update=True,
        )
        assert updated is False
        assert event_logged is False
        assert update_reason == "policy_controller_unavailable"


# ── _adaptive_online_should_defer_credit ─────────────────────────────────


class TestShouldDeferCredit:
    def test_defer_when_all_conditions_met(self) -> None:
        decision = _make_policy_decision()

        @dataclass
        class _FakeController:
            config: Any = None

        result = _adaptive_online_should_defer_credit(
            is_adaptive_llm_discretion=True,
            decision=decision,
            adaptive_online_policy_controller=_FakeController(),
            behavior_action=ACTION_INVOKE_HELPER,
            helper_outcome=ADAPTIVE_HELPER_OUTCOME_MATERIALIZED,
            suppression_reason="",
            success=False,
        )
        assert result is True

    def test_no_defer_when_success(self) -> None:
        decision = _make_policy_decision()

        @dataclass
        class _FakeController:
            config: Any = None

        result = _adaptive_online_should_defer_credit(
            is_adaptive_llm_discretion=True,
            decision=decision,
            adaptive_online_policy_controller=_FakeController(),
            behavior_action=ACTION_INVOKE_HELPER,
            helper_outcome=ADAPTIVE_HELPER_OUTCOME_MATERIALIZED,
            suppression_reason="",
            success=True,
        )
        assert result is False

    def test_no_defer_when_not_adaptive(self) -> None:
        result = _adaptive_online_should_defer_credit(
            is_adaptive_llm_discretion=False,
            decision=_make_policy_decision(),
            adaptive_online_policy_controller=object(),
            behavior_action=ACTION_INVOKE_HELPER,
            helper_outcome=ADAPTIVE_HELPER_OUTCOME_MATERIALIZED,
            suppression_reason="",
            success=False,
        )
        assert result is False

    def test_no_defer_when_skip_helper(self) -> None:
        result = _adaptive_online_should_defer_credit(
            is_adaptive_llm_discretion=True,
            decision=_make_policy_decision(),
            adaptive_online_policy_controller=object(),
            behavior_action=ACTION_SKIP_HELPER,
            helper_outcome=ADAPTIVE_HELPER_OUTCOME_MATERIALIZED,
            suppression_reason="",
            success=False,
        )
        assert result is False


# ── _finalize_adaptive_online_attempt update-reason branches ─────────────


def _finalize_kwargs(**overrides: Any) -> dict[str, Any]:
    """Minimal kwargs for _finalize_adaptive_online_attempt."""
    defaults = {
        "note": "test",
        "success": False,
        "attempt_wall_sec": 1.0,
        "is_adaptive_llm_discretion": True,
        "online_policy_decision": None,
        "adaptive_online_policy_controller": None,
        "run_id": "r",
        "case_id": "c",
        "condition_value": "adaptive",
        "attempt": 1,
        "attempt_metadata": {},
        "patch_meta": {},
        "online_policy_mode_for_attempt": "on",
        "online_policy_algo_for_attempt": "linucb",
        "online_policy_eligible": True,
        "online_policy_override_reason": "",
        "online_policy_feature_scaling_profile": "v1",
        "online_policy_features_raw": {},
        "online_policy_features_scaled": {},
        "helper_decision_reason": "",
        "normalized_adaptive_online_policy_mode": "on",
        "case_difficulty": "medium",
        "case_snapshot_dependency": "none",
        "adaptive_progress_made": False,
        "adaptive_progress_score": 0.0,
        "adaptive_no_progress_streak_current": 0,
        "reward_lambda_tokens": 0.0,
        "reward_mu_latency": 0.0,
        "reward_token_scale": 1.0,
        "reward_latency_scale": 1.0,
        "reward_eta_progress": 0.0,
        "reward_kappa_stagnation": 0.0,
    }
    defaults.update(overrides)
    return defaults


class TestFinalizeOnlineAttemptUpdateReasons:
    """Test the update_reason dispatch branches (lines 683-693)."""

    def test_not_eligible_reason(self) -> None:
        """Not eligible + no decision → reason includes helper_decision_reason."""
        meta: dict[str, Any] = {}
        patch: dict[str, Any] = {}
        _finalize_adaptive_online_attempt(
            **_finalize_kwargs(
                online_policy_decision=None,
                adaptive_online_policy_controller=None,
                online_policy_eligible=False,
                helper_decision_reason="budget_exhausted",
                attempt_metadata=meta,
                patch_meta=patch,
            )
        )
        assert meta["adaptive_online_update_reason"] == "not_eligible:budget_exhausted"

    def test_policy_off_reason(self) -> None:
        meta: dict[str, Any] = {}
        patch: dict[str, Any] = {}
        _finalize_adaptive_online_attempt(
            **_finalize_kwargs(
                online_policy_decision=None,
                adaptive_online_policy_controller=None,
                online_policy_eligible=True,
                normalized_adaptive_online_policy_mode="off",
                attempt_metadata=meta,
                patch_meta=patch,
            )
        )
        assert meta["adaptive_online_update_reason"] == "policy_off"

    def test_no_decision_emitted_reason(self) -> None:
        meta: dict[str, Any] = {}
        patch: dict[str, Any] = {}
        _finalize_adaptive_online_attempt(
            **_finalize_kwargs(
                online_policy_decision=None,
                adaptive_online_policy_controller=None,
                online_policy_eligible=True,
                normalized_adaptive_online_policy_mode="on",
                attempt_metadata=meta,
                patch_meta=patch,
            )
        )
        assert meta["adaptive_online_update_reason"] == "no_decision_emitted"

    def test_controller_unavailable_reason(self) -> None:
        """decision present but controller is None → policy_controller_unavailable."""
        decision = _make_policy_decision()
        meta: dict[str, Any] = {}
        patch: dict[str, Any] = {}
        _finalize_adaptive_online_attempt(
            **_finalize_kwargs(
                online_policy_decision=decision,
                adaptive_online_policy_controller=None,
                attempt_metadata=meta,
                patch_meta=patch,
            )
        )
        assert meta["adaptive_online_update_reason"] == "policy_controller_unavailable"

    def test_non_llm_discretion_condition(self) -> None:
        meta: dict[str, Any] = {}
        patch: dict[str, Any] = {}
        _finalize_adaptive_online_attempt(
            **_finalize_kwargs(
                is_adaptive_llm_discretion=False,
                attempt_metadata=meta,
                patch_meta=patch,
            )
        )
        assert meta["adaptive_online_update_reason"] == "non_llm_discretion_condition"


# ── _record_adaptive_online_policy_outcome with controller ───────────────


def _dummy_reward_breakdown(**overrides: Any) -> AdaptiveOnlineRewardBreakdown:
    defaults = {
        "reward_scope": ADAPTIVE_ONLINE_REWARD_SCOPE,
        "reward": 0.0,
        "reward_success_hint": 0.0,
        "reward_progress_hint": 0.0,
        "reward_progress_score": 0.0,
        "reward_token_penalty": 0.0,
        "reward_latency_penalty": 0.0,
        "reward_stagnation_penalty": 0.0,
        "reward_helper_materialization_bonus": 0.0,
        "reward_delayed_rescue_bonus": 0.0,
        "reward_credit_resolved_attempt": 0,
        "reward_credit_horizon_attempts": 0,
        "reward_credit_resolution_reason": "",
    }
    defaults.update(overrides)
    return AdaptiveOnlineRewardBreakdown(**defaults)


class TestRecordPolicyOutcomeWithController:
    """Lines 324-364: controller.record_outcome success + error paths."""

    def test_successful_record_outcome(self) -> None:
        captured: dict[str, Any] = {}

        @dataclass
        class _FakeEvent:
            model_updated: bool = True
            update_reason: str = "model_updated"
            override_reason: str = ""

        class _FakeController:
            @dataclass
            class config:
                decay: float = 1.0

            def record_outcome(self, **kwargs: Any) -> _FakeEvent:
                captured.update(kwargs)
                return _FakeEvent()

        decision = _make_policy_decision()
        updated, event_logged, update_reason, _override_reason = (
            _record_adaptive_online_policy_outcome(
                adaptive_online_policy_controller=_FakeController(),
                decision=decision,
                run_id="r",
                case_id="c",
                condition_value="adaptive",
                attempt=1,
                note="test",
                success=True,
                reward_breakdown=_dummy_reward_breakdown(),
                local_accounting=EMPTY_ADAPTIVE_POLICY_LOCAL_ACCOUNTING,
                override_reason="",
                behavior_action="invoke_helper",
                behavior_propensity=0.8,
                allow_model_update=True,
                reward_success_hint_scale=1.3,
                reward_progress_hint_scale=1.4,
                reward_stagnation_penalty_scale=1.5,
                reward_helper_materialization_bonus_scale=0.6,
            )
        )
        assert updated is True
        assert event_logged is True
        assert update_reason == "model_updated"
        assert captured["reward_success_hint_scale"] == pytest.approx(1.3)
        assert captured["reward_progress_hint_scale"] == pytest.approx(1.4)
        assert captured["reward_stagnation_penalty_scale"] == pytest.approx(1.5)
        assert captured["reward_helper_materialization_bonus_scale"] == pytest.approx(0.6)

    def test_record_outcome_fills_override_reason(self) -> None:
        @dataclass
        class _FakeEvent:
            model_updated: bool = False
            update_reason: str = "warmup"
            override_reason: str = "policy_warmup"

        class _FakeController:
            @dataclass
            class config:
                decay: float = 1.0

            def record_outcome(self, **kwargs: Any) -> _FakeEvent:
                return _FakeEvent()

        decision = _make_policy_decision()
        _updated, _event_logged, update_reason, override_reason = (
            _record_adaptive_online_policy_outcome(
                adaptive_online_policy_controller=_FakeController(),
                decision=decision,
                run_id="r",
                case_id="c",
                condition_value="adaptive",
                attempt=1,
                note="test",
                success=False,
                reward_breakdown=_dummy_reward_breakdown(),
                local_accounting=EMPTY_ADAPTIVE_POLICY_LOCAL_ACCOUNTING,
                override_reason="",
                behavior_action="none",
                behavior_propensity=1.0,
                allow_model_update=True,
            )
        )
        assert override_reason == "policy_warmup"
        assert update_reason == "warmup"

    def test_record_outcome_exception_sets_error_reason(self) -> None:
        class _FakeController:
            @dataclass
            class config:
                decay: float = 1.0

            def record_outcome(self, **kwargs: Any) -> None:
                raise RuntimeError("test error")

        decision = _make_policy_decision()
        updated, event_logged, update_reason, _ = _record_adaptive_online_policy_outcome(
            adaptive_online_policy_controller=_FakeController(),
            decision=decision,
            run_id="r",
            case_id="c",
            condition_value="adaptive",
            attempt=1,
            note="test",
            success=False,
            reward_breakdown=_dummy_reward_breakdown(),
            local_accounting=EMPTY_ADAPTIVE_POLICY_LOCAL_ACCOUNTING,
            override_reason="",
            behavior_action="none",
            behavior_propensity=1.0,
            allow_model_update=True,
        )
        assert updated is False
        assert event_logged is False
        assert update_reason.startswith("event_logging_error:RuntimeError")

    def test_record_outcome_preserves_existing_override_reason(self) -> None:
        @dataclass
        class _FakeEvent:
            model_updated: bool = False
            update_reason: str = "ok"
            override_reason: str = "should_be_ignored"

        class _FakeController:
            @dataclass
            class config:
                decay: float = 1.0

            def record_outcome(self, **kwargs: Any) -> _FakeEvent:
                return _FakeEvent()

        decision = _make_policy_decision()
        _, _, _, override_reason = _record_adaptive_online_policy_outcome(
            adaptive_online_policy_controller=_FakeController(),
            decision=decision,
            run_id="r",
            case_id="c",
            condition_value="adaptive",
            attempt=1,
            note="test",
            success=False,
            reward_breakdown=_dummy_reward_breakdown(),
            local_accounting=EMPTY_ADAPTIVE_POLICY_LOCAL_ACCOUNTING,
            override_reason="already_set",
            behavior_action="none",
            behavior_propensity=1.0,
            allow_model_update=True,
        )
        assert override_reason == "already_set"


# ── _finalize_adaptive_online_attempt with full controller ───────────────


class TestFinalizeOnlineAttemptWithController:
    """Cover lines 564-682: the full finalize path with a controller stub."""

    def _make_fake_controller(self) -> Any:
        @dataclass
        class _FakeEvent:
            model_updated: bool = False
            update_reason: str = "updated"
            override_reason: str = ""

        class _FakeController:
            @dataclass
            class config:
                decay: float = 1.0

            def record_outcome(self, **kwargs: Any) -> _FakeEvent:
                return _FakeEvent()

        return _FakeController()

    def test_non_deferred_happy_path(self) -> None:
        """decision + controller present, no defer → records outcome."""
        decision = _make_policy_decision()
        controller = self._make_fake_controller()
        meta: dict[str, Any] = {}
        patch: dict[str, Any] = {}
        pending = _finalize_adaptive_online_attempt(
            **_finalize_kwargs(
                is_adaptive_llm_discretion=True,
                online_policy_decision=decision,
                adaptive_online_policy_controller=controller,
                attempt_metadata=meta,
                patch_meta=patch,
                helper_decision_reason="",
                behavior_action=ACTION_INVOKE_HELPER,
                context_materialized=False,
                helper_outcome=ADAPTIVE_HELPER_OUTCOME_NOT_INVOKED,
            )
        )
        assert pending is None
        assert meta["adaptive_online_updated"] is False
        assert meta["adaptive_online_event_logged"] is True

    def test_deferred_credit_path(self) -> None:
        """All defer conditions met → returns PendingAdaptiveOnlineCredit."""
        decision = _make_policy_decision(
            selected_action=ACTION_INVOKE_HELPER,
            applied_action=ACTION_INVOKE_HELPER,
        )
        controller = self._make_fake_controller()
        meta: dict[str, Any] = {}
        patch: dict[str, Any] = {}
        pending = _finalize_adaptive_online_attempt(
            **_finalize_kwargs(
                is_adaptive_llm_discretion=True,
                online_policy_decision=decision,
                adaptive_online_policy_controller=controller,
                attempt_metadata=meta,
                patch_meta=patch,
                success=False,
                behavior_action=ACTION_INVOKE_HELPER,
                context_materialized=True,
                helper_outcome=ADAPTIVE_HELPER_OUTCOME_MATERIALIZED,
            )
        )
        assert pending is not None
        assert meta["adaptive_online_update_reason"] == "pending_delayed_credit"

    def test_transport_failed_suppression(self) -> None:
        """Transport failure → suppressed update reason override."""
        decision = _make_policy_decision()
        controller = self._make_fake_controller()
        meta: dict[str, Any] = {}
        patch: dict[str, Any] = {}
        _finalize_adaptive_online_attempt(
            **_finalize_kwargs(
                is_adaptive_llm_discretion=True,
                online_policy_decision=decision,
                adaptive_online_policy_controller=controller,
                attempt_metadata=meta,
                patch_meta=patch,
                behavior_action=ACTION_INVOKE_HELPER,
                helper_outcome=ADAPTIVE_HELPER_OUTCOME_TRANSPORT_FAILED,
            )
        )
        assert meta["adaptive_online_event_logged"] is True
