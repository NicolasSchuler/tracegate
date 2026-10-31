"""Phase 3 acceptance (2026-10-04 review): the default bandit learns, and seeds are reproducible.

Propensities in shadow mode, on mode and warm-up are covered by
``test_evals_policy_propensities.py``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from evals import _json as json
from evals import eval_types as et
from evals.adaptive_budget_online_policy import (
    ACTION_KEEP_BUDGET,
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
_ONLINE: dict[str, Any] = {
    **_COMMON,
    "reward_scope": "helper_credit_v2",
    "reward_helper_materialization_bonus": 0.0,
}
_BUDGET: dict[str, Any] = {
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


def _default_online_config(**overrides: Any) -> OnlinePolicyConfig:
    """The helper policy as a run configures it from the CLI defaults."""
    settings: dict[str, Any] = {
        "mode": "on",
        "alpha": et.DEFAULT_ADAPTIVE_ONLINE_ALPHA,
        "epsilon": et.DEFAULT_ADAPTIVE_ONLINE_EPSILON,
        "l2": et.DEFAULT_ADAPTIVE_POLICY_L2,
        "warmup_decisions": et.DEFAULT_ADAPTIVE_ONLINE_WARMUP_DECISIONS,
        "save_every": et.DEFAULT_ADAPTIVE_POLICY_SAVE_EVERY,
        "decay": et.DEFAULT_ADAPTIVE_ONLINE_DECAY,
    }
    return OnlinePolicyConfig.normalize(**{**settings, **overrides})


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_default_helper_policy_learns_linear_rewards(seed: int) -> None:
    """The default helper policy learns noisy linear rewards across 10 arms.

    With 53 features (52 + bias), 10 arms and linear rewards with noise, after 500 decisions
    the default policy picks the context's best arm far more often than uniform (10%).

    Before the review the decay of 0.95 on every arm kept about 20 observations of evidence
    and the score biases vetoed two arms, so the policy barely learned (POL-01, POL-07).
    """
    rng = np.random.default_rng(seed)
    dims = 52
    names = [f"f{index:02d}" for index in range(dims)]
    theta = {action: rng.normal(0.0, 0.15, size=dims) for action in ONLINE_POLICY_ACTIONS}
    controller = AdaptiveOnlinePolicyController(_default_online_config(random_seed=seed))
    hits: list[bool] = []
    for step in range(800):
        x = rng.random(dims)
        features = {"bias": 1.0, **{name: float(v) for name, v in zip(names, x, strict=True)}}
        means = {action: float(theta[action] @ x) for action in ONLINE_POLICY_ACTIONS}
        best = max(means, key=lambda action: means[action])
        decision = controller.decide(
            feature_values=features, default_action=ACTION_INVOKE_HELPER, rng_key=f"k{step}"
        )
        reward = means[decision.applied_action] + float(rng.normal(0.0, 0.1))
        controller.record_outcome(decision=decision, case_id=f"c{step}", reward=reward, **_ONLINE)
        hits.append(decision.selected_action == best)
    controller.close()

    best_arm_rate = float(np.mean(hits[-300:]))
    uniform_rate = 1.0 / len(ONLINE_POLICY_ACTIONS)
    assert best_arm_rate > 4 * uniform_rate  # observed 0.58-0.69 for these seeds


def _strip_volatile(line: str) -> dict[str, Any]:
    event = json.loads(line)
    event.pop("decision_id", None)
    event.pop("created_at_utc", None)
    return event


def _online_stream(tmp_path: Path, name: str) -> list[dict[str, Any]]:
    events_path = tmp_path / f"{name}.jsonl"
    controller = AdaptiveOnlinePolicyController(
        _default_online_config(random_seed=7, warmup_decisions=5, events_path=events_path)
    )
    rng = np.random.default_rng(0)
    for step in range(40):
        decision = controller.decide(
            feature_values={"bias": 1.0, "x": float(rng.random())},
            default_action=ACTION_INVOKE_HELPER,
            rng_key=f"case-{step}:attempt-1",
        )
        controller.record_outcome(
            decision=decision, case_id=f"c{step}", reward=float(rng.normal()), **_ONLINE
        )
    controller.close()
    return [_strip_volatile(line) for line in events_path.read_text(encoding="utf-8").splitlines()]


def _budget_stream(tmp_path: Path, name: str) -> list[dict[str, Any]]:
    events_path = tmp_path / f"{name}.jsonl"
    controller = AdaptiveBudgetPolicyController(
        BudgetPolicyConfig.normalize(
            mode="on",
            alpha=et.DEFAULT_ADAPTIVE_BUDGET_ONLINE_ALPHA,
            epsilon=et.DEFAULT_ADAPTIVE_BUDGET_ONLINE_EPSILON,
            warmup_decisions=5,
            decay=et.DEFAULT_ADAPTIVE_BUDGET_ONLINE_DECAY,
            events_path=events_path,
            random_seed=7,
        )
    )
    rng = np.random.default_rng(0)
    for step in range(40):
        decision = controller.decide(
            feature_values={"bias": 1.0, "x": float(rng.random())},
            default_action=ACTION_KEEP_BUDGET,
            rng_key=f"case-{step}:attempt-1",
        )
        controller.record_outcome(
            decision=decision, case_id=f"c{step}", reward=float(rng.normal()), **_BUDGET
        )
    controller.close()
    return [_strip_volatile(line) for line in events_path.read_text(encoding="utf-8").splitlines()]


@pytest.mark.parametrize("stream", [_online_stream, _budget_stream])
def test_identical_seeds_produce_identical_event_streams(tmp_path: Path, stream: Any) -> None:
    """Sequential decisions (``--jobs 1``) with the same seed log the same events.

    Only the random decision id and the timestamp differ.
    """
    first = stream(tmp_path, "first")
    second = stream(tmp_path, "second")
    assert len(first) == 40
    assert first == second
    assert len({event["selected_action"] for event in first}) > 1
