"""POL-24: greedy ties are broken at random, and selected and applied counts are kept apart."""

from __future__ import annotations

import random
from collections import Counter
from pathlib import Path

import pytest

from evals import _json as json
from evals import linucb_core as core
from evals.adaptive_budget_online_policy import (
    ACTION_INCREASE_BUDGET,
    ACTION_KEEP_BUDGET,
    AdaptiveBudgetPolicyController,
    BudgetPolicyConfig,
)
from evals.adaptive_budget_online_policy import (
    STATE_SCHEMA_VERSION as BUDGET_STATE_SCHEMA_VERSION,
)
from evals.adaptive_online_policy import (
    ACTION_FILE_ONLY,
    ACTION_INVOKE_HELPER,
    ACTION_SKIP_HELPER,
    ONLINE_POLICY_ACTIONS,
    STATE_SCHEMA_VERSION,
    AdaptiveOnlinePolicyController,
    OnlinePolicyConfig,
)
from evals.bootstrap_policy_state import main as bootstrap_main

_OUTCOME_KWARGS: dict[str, object] = {
    "run_id": "r",
    "condition": "layered_adaptive_llm",
    "attempt": 1,
    "note": "n",
    "reward_scope": "helper_credit_v2",
    "reward": 0.0,
    "reward_success_hint": 0.0,
    "reward_progress_hint": 0.0,
    "reward_progress_score": 0.0,
    "reward_token_penalty": 0.0,
    "reward_latency_penalty": 0.0,
    "reward_stagnation_penalty": 0.0,
    "reward_helper_materialization_bonus": 0.0,
    "success": False,
    "local_llm_call_count": 0,
    "local_tokens_prompt": 0,
    "local_tokens_completion": 0,
    "local_tokens_total": 0,
    "local_wall_sec": 0.0,
    "override_reason": "",
}


def _online(tmp_path: Path, *, mode: str = "on", epsilon: float = 0.0, **extra: object):
    config = OnlinePolicyConfig.normalize(
        mode=mode,
        algorithm="linucb",
        alpha=0.0,
        epsilon=epsilon,
        warmup_decisions=0,
        state_path=tmp_path / "state.json",
        events_path=tmp_path / "events.jsonl",
        random_seed=11,
        decay=1.0,
        action_score_biases={},
        **extra,
    )
    return AdaptiveOnlinePolicyController(config)


def test_greedy_ties_uses_a_relative_tolerance_and_keeps_available_order() -> None:
    scores = {"a": 1.0, "b": 1.0 + 1e-14, "c": 0.5, "d": 1.0}
    assert core.greedy_ties(scores, ("d", "c", "b", "a")) == ["d", "b", "a"]
    assert core.greedy_ties({"a": 1e6, "b": 1e6 + 1e-7}, ("a", "b")) == ["a", "b"]
    assert core.greedy_ties({"a": 1.0, "b": 1.0 + 1e-9}, ("a", "b")) == ["b"]


def test_pick_greedy_draws_uniformly_and_only_on_ties() -> None:
    rng = random.Random(0)
    state_before = rng.getstate()
    assert core.pick_greedy(["only"], rng) == "only"
    assert rng.getstate() == state_before
    picks = Counter(core.pick_greedy(["a", "b", "c"], rng) for _ in range(3000))
    assert set(picks) == {"a", "b", "c"}
    assert all(800 < count < 1200 for count in picks.values())


@pytest.mark.parametrize("epsilon", [0.0, 0.1, 0.5])
@pytest.mark.parametrize("tied", [("a",), ("a", "b"), ("a", "b", "c")])
def test_epsilon_greedy_propensities_sum_to_one(epsilon: float, tied: tuple[str, ...]) -> None:
    available = ("a", "b", "c", "d")
    total = sum(
        core.epsilon_greedy_propensity(action, tied, epsilon=epsilon, n_available=len(available))
        for action in available
    )
    assert total == pytest.approx(1.0)


def test_online_cold_start_tie_is_random_and_propensity_matches(tmp_path: Path) -> None:
    controller = _online(tmp_path)
    available = [ACTION_SKIP_HELPER, ACTION_INVOKE_HELPER, ACTION_FILE_ONLY]
    decisions = [
        controller.decide(
            feature_values={"bias": 1.0},
            default_action=ACTION_INVOKE_HELPER,
            available_actions=available,
        )
        for _ in range(60)
    ]
    controller.close()

    # Before POL-24 every one of these went to invoke_helper with propensity 1.0.
    assert {decision.selected_action for decision in decisions} == set(available)
    assert all(decision.propensity == pytest.approx(1.0 / 3.0) for decision in decisions)
    assert all(decision.decision_strategy == "linucb_greedy" for decision in decisions)


def test_online_shadow_mode_counts_selected_and_applied_separately(tmp_path: Path) -> None:
    controller = _online(tmp_path, mode="shadow")
    decisions = [
        controller.decide(feature_values={"bias": 1.0}, default_action=ACTION_INVOKE_HELPER)
        for _ in range(30)
    ]
    state = controller.state
    controller.close()

    selected = Counter(decision.selected_action for decision in decisions)
    assert all(decision.applied_action == ACTION_INVOKE_HELPER for decision in decisions)
    assert {k: v for k, v in state.action_counts.items() if v} == dict(selected)
    assert state.applied_action_counts[ACTION_INVOKE_HELPER] == 30
    assert sum(state.applied_action_counts.values()) == 30

    saved = json.loads((tmp_path / "state.json").read_text(encoding="utf-8"))
    assert saved["schema_version"] == STATE_SCHEMA_VERSION == 4
    assert saved["applied_action_counts"][ACTION_INVOKE_HELPER] == 30


def test_online_loads_schema_3_state_with_zero_applied_counts(tmp_path: Path) -> None:
    (tmp_path / "state.json").write_text(
        json.dumps(
            {
                "schema_version": 3,
                "algorithm": "linucb",
                "feature_names": ["bias"],
                "decisions": 2,
                "updates": 0,
                "action_counts": {ACTION_SKIP_HELPER: 2},
                "A": {action: [[1.0]] for action in ONLINE_POLICY_ACTIONS},
                "b": {action: [0.0] for action in ONLINE_POLICY_ACTIONS},
            }
        ),
        encoding="utf-8",
    )
    controller = _online(tmp_path)
    state = controller.state
    controller.close()

    assert controller.load_error is None
    assert state.action_counts[ACTION_SKIP_HELPER] == 2
    assert set(state.applied_action_counts.values()) == {0}


def test_budget_gate_tie_is_random_and_shadow_counts_applied(tmp_path: Path) -> None:
    config = BudgetPolicyConfig.normalize(
        mode="shadow",
        algorithm="linucb",
        alpha=0.0,
        epsilon=0.0,
        warmup_decisions=0,
        state_path=tmp_path / "budget_state.json",
        events_path=tmp_path / "budget_events.jsonl",
        random_seed=5,
        decay=1.0,
    )
    controller = AdaptiveBudgetPolicyController(config)
    decisions = [
        controller.decide(feature_values={"bias": 1.0}, default_action=ACTION_KEEP_BUDGET)
        for _ in range(40)
    ]
    state = controller.state
    controller.close()

    # The tie used to go to the default action every time.
    assert {decision.selected_action for decision in decisions} == {
        ACTION_KEEP_BUDGET,
        ACTION_INCREASE_BUDGET,
    }
    assert all(decision.propensity == pytest.approx(0.5) for decision in decisions)
    assert state.applied_action_counts == {ACTION_KEEP_BUDGET: 40, ACTION_INCREASE_BUDGET: 0}
    assert state.action_counts[ACTION_INCREASE_BUDGET] > 0

    saved = json.loads((tmp_path / "budget_state.json").read_text(encoding="utf-8"))
    assert saved["schema_version"] == BUDGET_STATE_SCHEMA_VERSION == 4
    assert saved["gate_state"]["applied_action_counts"][ACTION_KEEP_BUDGET] == 40


def test_bootstrap_replay_restores_applied_counts(tmp_path: Path) -> None:
    controller = _online(tmp_path, mode="shadow")
    for index in range(12):
        decision = controller.decide(
            feature_values={"bias": 1.0}, default_action=ACTION_INVOKE_HELPER
        )
        controller.record_outcome(decision=decision, case_id=f"c{index}", **_OUTCOME_KWARGS)
    controller.close()

    replay_path = tmp_path / "replay.json"
    assert (
        bootstrap_main(
            ["--events", str(tmp_path / "events.jsonl"), "--output-state", str(replay_path)]
        )
        == 0
    )
    replay = json.loads(replay_path.read_text(encoding="utf-8"))
    assert replay["applied_action_counts"][ACTION_INVOKE_HELPER] == 12
    assert sum(replay["applied_action_counts"].values()) == 12
