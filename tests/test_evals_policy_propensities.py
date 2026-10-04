"""POL-03 / Phase 3 acceptance: logged propensities in shadow mode, on mode and warm-up.

The behavior propensity is the probability that the logging policy executed the logged
action. Shadow mode executes a deterministic baseline (probability 1); on mode samples
uniformly during warm-up (1/K) and epsilon-greedy afterwards.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from evals import linucb_core as core
from evals.adaptive_budget_online_policy import (
    ACTION_KEEP_BUDGET,
    BUDGET_POLICY_ACTIONS,
    AdaptiveBudgetPolicyController,
    BudgetPolicyConfig,
)
from evals.adaptive_online_policy import (
    ACTION_INVOKE_HELPER,
    ONLINE_POLICY_ACTIONS,
    AdaptiveOnlinePolicyController,
    OnlinePolicyConfig,
)

_COMMON: dict[str, Any] = {
    "run_id": "r",
    "condition": "layered_adaptive_llm",
    "attempt": 1,
    "note": "n",
    "reward": 0.0,
    "reward_success_hint": 0.0,
    "reward_progress_hint": 0.0,
    "reward_progress_score": 0.0,
    "reward_token_penalty": 0.0,
    "reward_latency_penalty": 0.0,
    "reward_stagnation_penalty": 0.0,
    "success": False,
    "local_llm_call_count": 0,
    "local_tokens_prompt": 0,
    "local_tokens_completion": 0,
    "local_tokens_total": 0,
    "local_wall_sec": 0.0,
    "override_reason": "",
}
_ONLINE_OUTCOME: dict[str, Any] = {
    **_COMMON,
    "reward_scope": "helper_credit_v2",
    "reward_helper_materialization_bonus": 0.0,
}
_BUDGET_OUTCOME: dict[str, Any] = {
    **_COMMON,
    "reward_scope": "budget_local_v1",
    "reward_budget_exceeded_penalty": 0.0,
    "request_budget_exceeded": False,
    "selected_multiplier": 1.0,
    "applied_multiplier": 1.0,
    "default_multiplier": 1.0,
    "default_controls": {},
    "selected_controls": {},
    "applied_controls": {},
    "selected_controls_changed": {},
    "applied_controls_changed": {},
    "partial_override": False,
    "partial_override_errors": {},
    "tier_index_before": 0,
    "tier_index_after": 0,
}


def _online(tmp_path: Path, *, mode: str, warmup: int, epsilon: float):
    return AdaptiveOnlinePolicyController(
        OnlinePolicyConfig.normalize(
            mode=mode,
            alpha=0.5,
            epsilon=epsilon,
            warmup_decisions=warmup,
            events_path=tmp_path / f"online_{mode}.jsonl",
            random_seed=17,
            decay=1.0,
        )
    )


def _budget(tmp_path: Path, *, mode: str, warmup: int, epsilon: float):
    return AdaptiveBudgetPolicyController(
        BudgetPolicyConfig.normalize(
            mode=mode,
            alpha=0.5,
            epsilon=epsilon,
            warmup_decisions=warmup,
            events_path=tmp_path / f"budget_{mode}.jsonl",
            random_seed=17,
            decay=1.0,
        )
    )


def test_behavior_propensity_rule() -> None:
    assert core.behavior_propensity(mode="on", selected="a", behavior="a", propensity=0.3) == 0.3
    assert core.behavior_propensity(mode="on", selected="a", behavior="b", propensity=0.3) == 1.0
    assert (
        core.behavior_propensity(mode="shadow", selected="a", behavior="a", propensity=0.1) == 1.0
    )
    assert core.behavior_propensity(mode="off", selected="a", behavior="a", propensity=0.1) == 1.0


def test_helper_on_mode_logs_the_sampling_probability(tmp_path: Path) -> None:
    controller = _online(tmp_path, mode="on", warmup=5, epsilon=0.2)
    k = len(ONLINE_POLICY_ACTIONS)
    for index in range(30):
        decision = controller.decide(
            feature_values={"bias": 1.0, "x": float(index % 3)},
            default_action=ACTION_INVOKE_HELPER,
        )
        event = controller.record_outcome(decision=decision, case_id=f"c{index}", **_ONLINE_OUTCOME)
        if index < 5:
            assert decision.warmup_active is True
            assert decision.propensity == pytest.approx(1.0 / k)
        else:
            tied = core.greedy_ties(decision.scores, decision.available_actions)
            assert decision.propensity == pytest.approx(
                core.epsilon_greedy_propensity(
                    decision.selected_action, tied, epsilon=0.2, n_available=k
                )
            )
        assert event.behavior_action == decision.selected_action
        assert event.behavior_propensity == pytest.approx(decision.propensity)
    controller.close()


def test_helper_shadow_mode_logs_probability_one(tmp_path: Path) -> None:
    controller = _online(tmp_path, mode="shadow", warmup=10, epsilon=0.1)
    agreeing = 0
    for index in range(40):
        decision = controller.decide(
            feature_values={"bias": 1.0}, default_action=ACTION_INVOKE_HELPER
        )
        event = controller.record_outcome(decision=decision, case_id=f"c{index}", **_ONLINE_OUTCOME)
        assert decision.applied_action == ACTION_INVOKE_HELPER
        assert event.behavior_propensity == 1.0
        agreeing += decision.selected_action == ACTION_INVOKE_HELPER
    controller.close()
    assert agreeing > 0  # covers the case that used to log 1/K during warm-up


def test_budget_on_mode_logs_the_sampling_probability(tmp_path: Path) -> None:
    controller = _budget(tmp_path, mode="on", warmup=4, epsilon=0.3)
    k = len(BUDGET_POLICY_ACTIONS)
    for index in range(20):
        decision = controller.decide(
            feature_values={"bias": 1.0, "x": float(index % 2)},
            default_action=ACTION_KEEP_BUDGET,
        )
        event = controller.record_outcome(decision=decision, case_id=f"c{index}", **_BUDGET_OUTCOME)
        if index < 4:
            assert decision.propensity == pytest.approx(1.0 / k)
        assert event.behavior_action == decision.selected_action
        assert event.behavior_propensity == pytest.approx(decision.propensity)
    controller.close()


def test_budget_shadow_mode_logs_probability_one(tmp_path: Path) -> None:
    """The budget controller's events used the selection's propensity whenever it matched
    the applied baseline (POL-03); the attempt metadata path was already fixed."""
    controller = _budget(tmp_path, mode="shadow", warmup=6, epsilon=0.1)
    agreeing = 0
    for index in range(30):
        decision = controller.decide(
            feature_values={"bias": 1.0}, default_action=ACTION_KEEP_BUDGET
        )
        event = controller.record_outcome(decision=decision, case_id=f"c{index}", **_BUDGET_OUTCOME)
        assert decision.applied_action == ACTION_KEEP_BUDGET
        assert event.behavior_propensity == 1.0
        agreeing += decision.selected_action == ACTION_KEEP_BUDGET
    controller.close()
    assert agreeing > 0
