"""POL-21: a loaded state must cover the run's features, and arm resets are counted."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from evals import _json as json
from evals import linucb_core as core
from evals.adaptive_budget_online_policy import (
    ACTION_BUDGET_FAMILY_TIME,
    ACTION_INCREASE_BUDGET,
    ACTION_KEEP_BUDGET,
    BUDGET_POLICY_ARCH_HIERARCHICAL_V1,
    AdaptiveBudgetPolicyController,
    BudgetPolicyConfig,
)
from evals.adaptive_online_policy import (
    ACTION_INVOKE_HELPER,
    AdaptiveOnlinePolicyController,
    OnlinePolicyConfig,
)
from evals.bootstrap_policy_state import ReplayState, _ensure_feature_state

_OUTCOME_KWARGS: dict[str, object] = {
    "run_id": "r",
    "case_id": "c",
    "condition": "layered_adaptive_llm",
    "attempt": 1,
    "note": "n",
    "reward_scope": "helper_credit_v2",
    "reward": 1.0,
    "reward_success_hint": 0.0,
    "reward_progress_hint": 0.0,
    "reward_progress_score": 0.0,
    "reward_token_penalty": 0.0,
    "reward_latency_penalty": 0.0,
    "reward_stagnation_penalty": 0.0,
    "reward_helper_materialization_bonus": 0.0,
    "success": True,
    "local_llm_call_count": 0,
    "local_tokens_prompt": 0,
    "local_tokens_completion": 0,
    "local_tokens_total": 0,
    "local_wall_sec": 0.0,
    "override_reason": "",
}


def _online(tmp_path: Path) -> AdaptiveOnlinePolicyController:
    return AdaptiveOnlinePolicyController(
        OnlinePolicyConfig.normalize(
            mode="on",
            warmup_decisions=0,
            save_every=1,
            state_path=tmp_path / "state.json",
            events_path=tmp_path / "events.jsonl",
            random_seed=0,
            decay=1.0,
        )
    )


def _train_online(tmp_path: Path, features: dict[str, float]) -> None:
    controller = _online(tmp_path)
    for _ in range(3):
        decision = controller.decide(feature_values=features, default_action=ACTION_INVOKE_HELPER)
        controller.record_outcome(decision=decision, **_OUTCOME_KWARGS)
    controller.close()


def test_core_reports_added_features_and_reset_arms() -> None:
    state = core.LinUCBState(action_names=("a", "b"))
    first = core.ensure_feature_state(state, ["bias", "x"], l2=1.0)
    assert first == core.FeatureStateChange(added=("bias", "x"))

    state.A["b"] = np.eye(5)  # wrong shape: the arm can't be kept
    change = core.ensure_feature_state(state, ["bias", "x", "y"], l2=1.0)
    assert change.added == ("y",)
    assert change.reset_arms == ("b",)
    np.testing.assert_array_equal(state.A["b"], np.eye(3))

    assert core.missing_features(state, ["bias"]) == ["x", "y"]
    assert core.missing_features(core.LinUCBState(), ["bias"]) == []
    message = core.feature_set_mismatch_error([f"f{i}" for i in range(7)])
    assert message.startswith("FeatureSetMismatch:")
    assert "f4" in message
    assert "(+2 more)" in message


def test_online_loaded_state_missing_a_feature_is_a_load_failure(tmp_path: Path) -> None:
    _train_online(tmp_path, {"bias": 1.0, "retired_feature": 1.0})

    controller = _online(tmp_path)
    assert controller.load_error is None
    decision = controller.decide(feature_values={"bias": 1.0}, default_action=ACTION_INVOKE_HELPER)
    event = controller.record_outcome(decision=decision, **_OUTCOME_KWARGS)
    state = controller.state
    controller.close()

    # The old controller zero-filled retired_feature and kept the learned arms.
    assert controller.load_error is not None
    assert "retired_feature" in controller.load_error
    assert state.feature_names == ["bias"]
    assert state.decisions == 1
    assert state.updates == 1
    assert event.state_resets == {"feature_set_mismatch": 1}
    logged = json.loads((tmp_path / "events.jsonl").read_text(encoding="utf-8").splitlines()[-1])
    assert logged["state_resets"] == {"feature_set_mismatch": 1}


def test_online_loaded_state_may_gain_features(tmp_path: Path) -> None:
    _train_online(tmp_path, {"bias": 1.0})

    controller = _online(tmp_path)
    controller.decide(
        feature_values={"bias": 1.0, "new_feature": 0.5}, default_action=ACTION_INVOKE_HELPER
    )
    state = controller.state
    controller.close()

    assert controller.load_error is None
    assert controller.get_state_resets() == {}
    assert state.feature_names == ["bias", "new_feature"]
    assert state.updates == 3


def test_online_counts_arm_resets(tmp_path: Path) -> None:
    controller = _online(tmp_path)
    controller.decide(feature_values={"bias": 1.0}, default_action=ACTION_INVOKE_HELPER)
    controller._state.A[ACTION_INVOKE_HELPER] = np.eye(4)
    controller.decide(feature_values={"bias": 1.0}, default_action=ACTION_INVOKE_HELPER)
    resets = controller.get_state_resets()
    controller.close()

    assert resets == {f"arm:{ACTION_INVOKE_HELPER}": 1}


def test_budget_loaded_family_state_missing_a_feature_is_a_load_failure(tmp_path: Path) -> None:
    def controller() -> AdaptiveBudgetPolicyController:
        return AdaptiveBudgetPolicyController(
            BudgetPolicyConfig.normalize(
                mode="on",
                architecture=BUDGET_POLICY_ARCH_HIERARCHICAL_V1,
                epsilon=0.0,
                warmup_decisions=0,
                save_every=1,
                state_path=tmp_path / "budget_state.json",
                events_path=tmp_path / "budget_events.jsonl",
                random_seed=1,
                decay=1.0,
            )
        )

    first = controller()
    for _ in range(4):
        first.decide(
            feature_values={"bias": 1.0},
            family_feature_values={"bias": 1.0, "retired_family_feature": 1.0},
            default_action=ACTION_INCREASE_BUDGET,
            available_families=(ACTION_BUDGET_FAMILY_TIME,),
        )
    first._save_state()
    first.close()
    assert "retired_family_feature" in first.family_state.feature_names

    second = controller()
    assert second.load_error is None
    second.decide(
        feature_values={"bias": 1.0},
        family_feature_values={"bias": 1.0},
        default_action=ACTION_KEEP_BUDGET,
        available_families=(ACTION_BUDGET_FAMILY_TIME,),
    )
    gate_state = second.state
    second.close()

    assert second.load_error is not None
    assert "family:retired_family_feature" in second.load_error
    assert second.get_state_resets() == {"feature_set_mismatch": 1}
    assert gate_state.decisions == 1


def test_bootstrap_replay_flags_feature_set_changes() -> None:
    state = ReplayState(action_names=("a",))
    warnings: list[str] = []
    _ensure_feature_state(
        state=state, cleaned_features={"bias": 1.0, "x": 1.0}, l2=1.0, warnings=warnings
    )
    assert warnings == []
    _ensure_feature_state(state=state, cleaned_features={"bias": 1.0}, l2=1.0, warnings=warnings)
    assert warnings == ["feature_set_changed_across_events"]
