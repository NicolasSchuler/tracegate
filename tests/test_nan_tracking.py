"""Tests for NaN replacement tracking in adaptive policy controllers."""

from __future__ import annotations

import math

from evals._policy_utils import safe_float_tracked
from evals.adaptive_budget_online_policy import (
    ACTION_KEEP_BUDGET,
    AdaptiveBudgetPolicyController,
    BudgetPolicyConfig,
)
from evals.adaptive_online_policy import (
    ACTION_INVOKE_HELPER,
    AdaptiveOnlinePolicyController,
    OnlinePolicyConfig,
)
from llmdebug import _json as json

# ---------------------------------------------------------------------------
# safe_float_tracked unit tests
# ---------------------------------------------------------------------------


def test_safe_float_tracked_normal_value() -> None:
    result, replaced = safe_float_tracked(3.14, default=0.0)
    assert result == 3.14
    assert replaced is False


def test_safe_float_tracked_nan() -> None:
    result, replaced = safe_float_tracked(float("nan"), default=0.0)
    assert result == 0.0
    assert replaced is True


def test_safe_float_tracked_inf() -> None:
    result, replaced = safe_float_tracked(float("inf"), default=-1.0)
    assert result == -1.0
    assert replaced is True


def test_safe_float_tracked_neg_inf() -> None:
    result, replaced = safe_float_tracked(float("-inf"), default=0.0)
    assert result == 0.0
    assert replaced is True


def test_safe_float_tracked_unconvertible() -> None:
    result, replaced = safe_float_tracked("not_a_number", default=5.0)
    assert result == 5.0
    assert replaced is True


def test_safe_float_tracked_none() -> None:
    result, replaced = safe_float_tracked(None, default=0.0)
    assert result == 0.0
    assert replaced is True


# ---------------------------------------------------------------------------
# Helpers for building minimal controllers
# ---------------------------------------------------------------------------


def _online_controller(tmp_path, *, mode="on"):
    events_path = tmp_path / "events.jsonl"
    config = OnlinePolicyConfig.normalize(
        mode=mode,
        algorithm="linucb",
        alpha=0.0,
        epsilon=0.0,
        l2=1.0,
        warmup_decisions=0,
        save_every=999,
        state_path=tmp_path / "state.json",
        events_path=events_path,
        random_seed=42,
        decay=1.0,
    )
    return AdaptiveOnlinePolicyController(config), events_path


def _budget_controller(tmp_path, *, mode="on"):
    events_path = tmp_path / "budget_events.jsonl"
    config = BudgetPolicyConfig.normalize(
        mode=mode,
        algorithm="linucb",
        alpha=0.0,
        epsilon=0.0,
        l2=1.0,
        warmup_decisions=0,
        save_every=999,
        state_path=tmp_path / "budget_state.json",
        events_path=events_path,
        random_seed=42,
        decay=1.0,
    )
    return AdaptiveBudgetPolicyController(config), events_path


def _record_online_outcome(controller, decision, *, run_id="run1"):
    return controller.record_outcome(
        decision=decision,
        run_id=run_id,
        case_id="case_a",
        condition="test_cond",
        attempt=1,
        note="test",
        reward_scope="test",
        reward=0.5,
        reward_success_hint=0.0,
        reward_progress_hint=0.0,
        reward_progress_score=0.0,
        reward_token_penalty=0.0,
        reward_latency_penalty=0.0,
        reward_stagnation_penalty=0.0,
        reward_helper_materialization_bonus=0.0,
        success=False,
        local_llm_call_count=1,
        local_tokens_prompt=10,
        local_tokens_completion=5,
        local_tokens_total=15,
        local_wall_sec=0.1,
        override_reason="",
    )


def _record_budget_outcome(controller, decision, *, run_id="run1"):
    return controller.record_outcome(
        decision=decision,
        run_id=run_id,
        case_id="case_a",
        condition="test_cond",
        attempt=1,
        note="test",
        reward_scope="test",
        reward=0.5,
        reward_success_hint=0.0,
        reward_progress_hint=0.0,
        reward_progress_score=0.0,
        reward_token_penalty=0.0,
        reward_latency_penalty=0.0,
        reward_stagnation_penalty=0.0,
        reward_budget_exceeded_penalty=0.0,
        request_budget_exceeded=False,
        success=False,
        local_llm_call_count=1,
        local_tokens_prompt=10,
        local_tokens_completion=5,
        local_tokens_total=15,
        local_wall_sec=0.1,
        selected_multiplier=1.0,
        applied_multiplier=1.0,
        default_multiplier=1.0,
        default_controls={},
        selected_controls={},
        applied_controls={},
        selected_controls_changed={},
        applied_controls_changed={},
        partial_override=False,
        partial_override_errors={},
        tier_index_before=0,
        tier_index_after=0,
        override_reason="",
    )


# ---------------------------------------------------------------------------
# Online policy NaN tracking tests
# ---------------------------------------------------------------------------


class TestOnlinePolicyNanTracking:
    """NaN replacement tracking for AdaptiveOnlinePolicyController."""

    def test_no_nan_no_counts(self, tmp_path) -> None:
        ctrl, _ = _online_controller(tmp_path)
        ctrl.decide(
            feature_values={"bias": 1.0},
            default_action=ACTION_INVOKE_HELPER,
        )
        assert ctrl.get_nan_stats() == {}

    def test_nan_replacement_counted(self, tmp_path) -> None:
        ctrl, _ = _online_controller(tmp_path)
        ctrl.decide(
            feature_values={"bias": float("nan")},
            default_action=ACTION_INVOKE_HELPER,
        )
        stats = ctrl.get_nan_stats()
        assert "bias" in stats
        assert stats["bias"] >= 1

    def test_inf_replacement_counted(self, tmp_path) -> None:
        ctrl, _ = _online_controller(tmp_path)
        ctrl.decide(
            feature_values={"bias": float("inf")},
            default_action=ACTION_INVOKE_HELPER,
        )
        stats = ctrl.get_nan_stats()
        assert "bias" in stats
        assert stats["bias"] >= 1

    def test_per_feature_tracking(self, tmp_path) -> None:
        ctrl, _ = _online_controller(tmp_path)
        ctrl.decide(
            feature_values={"feat_a": float("nan"), "feat_b": 1.0},
            default_action=ACTION_INVOKE_HELPER,
        )
        stats = ctrl.get_nan_stats()
        assert "feat_a" in stats
        assert stats["feat_a"] >= 1
        # feat_b was valid; it should not appear in nan stats
        assert stats.get("feat_b", 0) == 0

    def test_counts_accumulate_across_decisions(self, tmp_path) -> None:
        ctrl, _ = _online_controller(tmp_path)
        ctrl.decide(
            feature_values={"bias": float("nan")},
            default_action=ACTION_INVOKE_HELPER,
        )
        first_count = ctrl.get_nan_stats().get("bias", 0)
        ctrl.decide(
            feature_values={"bias": float("nan")},
            default_action=ACTION_INVOKE_HELPER,
        )
        second_count = ctrl.get_nan_stats().get("bias", 0)
        assert second_count > first_count

    def test_stats_in_event_log(self, tmp_path) -> None:
        ctrl, events_path = _online_controller(tmp_path)
        decision = ctrl.decide(
            feature_values={"bias": float("nan")},
            default_action=ACTION_INVOKE_HELPER,
        )
        event = _record_online_outcome(ctrl, decision)
        assert "nan_replacement_counts" in event.to_json()
        assert event.nan_replacement_counts.get("bias", 0) >= 1

        # Also check the serialised JSONL line
        lines = events_path.read_text(encoding="utf-8").strip().splitlines()
        payload = json.loads(lines[-1])
        assert "nan_replacement_counts" in payload
        assert payload["nan_replacement_counts"].get("bias", 0) >= 1

    def test_event_log_empty_when_no_nan(self, tmp_path) -> None:
        ctrl, _events_path = _online_controller(tmp_path)
        decision = ctrl.decide(
            feature_values={"bias": 1.0},
            default_action=ACTION_INVOKE_HELPER,
        )
        event = _record_online_outcome(ctrl, decision)
        assert event.nan_replacement_counts == {}

    def test_get_nan_stats_returns_copy(self, tmp_path) -> None:
        ctrl, _ = _online_controller(tmp_path)
        ctrl.decide(
            feature_values={"bias": float("nan")},
            default_action=ACTION_INVOKE_HELPER,
        )
        stats_a = ctrl.get_nan_stats()
        stats_b = ctrl.get_nan_stats()
        assert stats_a == stats_b
        # Mutating the returned dict must not affect internals
        stats_a["bias"] = 9999
        assert ctrl.get_nan_stats()["bias"] != 9999


# ---------------------------------------------------------------------------
# Budget policy NaN tracking tests
# ---------------------------------------------------------------------------


class TestBudgetPolicyNanTracking:
    """NaN replacement tracking for AdaptiveBudgetPolicyController."""

    def test_no_nan_no_counts(self, tmp_path) -> None:
        ctrl, _ = _budget_controller(tmp_path)
        ctrl.decide(
            feature_values={"bias": 1.0},
            default_action=ACTION_KEEP_BUDGET,
        )
        assert ctrl.get_nan_stats() == {}

    def test_nan_replacement_counted(self, tmp_path) -> None:
        ctrl, _ = _budget_controller(tmp_path)
        ctrl.decide(
            feature_values={"bias": float("nan")},
            default_action=ACTION_KEEP_BUDGET,
        )
        stats = ctrl.get_nan_stats()
        assert "bias" in stats
        assert stats["bias"] >= 1

    def test_per_feature_tracking(self, tmp_path) -> None:
        ctrl, _ = _budget_controller(tmp_path)
        ctrl.decide(
            feature_values={"feat_x": float("nan"), "feat_y": 2.0},
            default_action=ACTION_KEEP_BUDGET,
        )
        stats = ctrl.get_nan_stats()
        assert "feat_x" in stats
        assert stats["feat_x"] >= 1
        assert stats.get("feat_y", 0) == 0

    def test_counts_accumulate_across_decisions(self, tmp_path) -> None:
        ctrl, _ = _budget_controller(tmp_path)
        ctrl.decide(
            feature_values={"bias": float("nan")},
            default_action=ACTION_KEEP_BUDGET,
        )
        first_count = ctrl.get_nan_stats().get("bias", 0)
        ctrl.decide(
            feature_values={"bias": float("nan")},
            default_action=ACTION_KEEP_BUDGET,
        )
        second_count = ctrl.get_nan_stats().get("bias", 0)
        assert second_count > first_count

    def test_stats_in_event_log(self, tmp_path) -> None:
        ctrl, events_path = _budget_controller(tmp_path)
        decision = ctrl.decide(
            feature_values={"bias": float("nan")},
            default_action=ACTION_KEEP_BUDGET,
        )
        event = _record_budget_outcome(ctrl, decision)
        assert "nan_replacement_counts" in event.to_json()
        assert event.nan_replacement_counts.get("bias", 0) >= 1

        lines = events_path.read_text(encoding="utf-8").strip().splitlines()
        payload = json.loads(lines[-1])
        assert "nan_replacement_counts" in payload
        assert payload["nan_replacement_counts"].get("bias", 0) >= 1

    def test_event_log_empty_when_no_nan(self, tmp_path) -> None:
        ctrl, _events_path = _budget_controller(tmp_path)
        decision = ctrl.decide(
            feature_values={"bias": 1.0},
            default_action=ACTION_KEEP_BUDGET,
        )
        event = _record_budget_outcome(ctrl, decision)
        assert event.nan_replacement_counts == {}

    def test_get_nan_stats_returns_copy(self, tmp_path) -> None:
        ctrl, _ = _budget_controller(tmp_path)
        ctrl.decide(
            feature_values={"bias": float("nan")},
            default_action=ACTION_KEEP_BUDGET,
        )
        stats_a = ctrl.get_nan_stats()
        stats_b = ctrl.get_nan_stats()
        assert stats_a == stats_b
        stats_a["bias"] = 9999
        assert ctrl.get_nan_stats()["bias"] != 9999
