"""Tests for decision-model roles inside the adaptive online policy controller."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

from evals.adaptive_online_policy import (
    EVENT_SCHEMA_VERSION,
    AdaptiveOnlinePolicyController,
    OnlinePolicyConfig,
    PolicyDecision,
    zero_online_action_score_biases,
)
from evals.decision_model.config import DecisionModelPolicyConfig
from evals.decision_model.policy import DM_FEATURE_NAMES, DecisionModelAdvice, DecisionModelBinding
from evals.decision_model.schema import ParsedAnswers

ACTIONS = ("skip_helper", "invoke_helper", "frame_only", "file_only")
FEATURES = {"bias": 1.0, "attempt_frac": 0.5}


def _advice(top: str = "file_only", status: str = "ok") -> DecisionModelAdvice:
    probs = {action: (0.7 if action == top else 0.1) for action in ACTIONS}
    return DecisionModelAdvice(
        status=status,
        profile="evidence_v1",
        answers=ParsedAnswers(
            action_probs=probs,
            traceback_sufficient=0.2,
            missing_information={"source_definition": 0.9, "none": 0.1},
            fix_confidence=0.5,
        ),
    )


class _ClosingAdvisor:
    def __init__(self) -> None:
        self.closed = False

    def advise(self, request: Any) -> DecisionModelAdvice:
        del request
        return _advice()

    def close(self) -> None:
        self.closed = True


def _controller(
    tmp_path: Path,
    *,
    role: str | None = None,
    mode: str = "on",
    warmup: int = 0,
    epsilon: float = 0.0,
    algorithm: str = "linucb",
    **policy: Any,
) -> AdaptiveOnlinePolicyController:
    config = OnlinePolicyConfig.normalize(
        mode=mode,
        algorithm=algorithm,
        alpha=0.0,
        epsilon=epsilon,
        warmup_decisions=warmup,
        decay=1.0,
        state_path=tmp_path / f"state_{role}_{mode}.json",
        events_path=tmp_path / f"events_{role}_{mode}.jsonl",
        random_seed=7,
        action_score_biases=zero_online_action_score_biases(),
    )
    if role is None:
        return AdaptiveOnlinePolicyController(config)
    binding = DecisionModelBinding(
        config=DecisionModelPolicyConfig(role=role, **policy), advisor=_ClosingAdvisor()
    )
    return AdaptiveOnlinePolicyController(config, decision_model=binding)


def _decide(
    controller: AdaptiveOnlinePolicyController, advice: DecisionModelAdvice | None = None
) -> PolicyDecision:
    return controller.decide(
        feature_values=FEATURES,
        default_action="invoke_helper",
        available_actions=ACTIONS,
        advice=advice,
    )


def _record(controller: AdaptiveOnlinePolicyController, decision: PolicyDecision) -> Any:
    return controller.record_outcome(
        decision=decision,
        run_id="run",
        case_id="case",
        condition="layered_adaptive_llm",
        attempt=1,
        note="tests_still_failing",
        reward_scope="helper_local_v1",
        reward=0.1,
        reward_success_hint=0.0,
        reward_progress_hint=0.0,
        reward_progress_score=0.0,
        reward_token_penalty=0.0,
        reward_latency_penalty=0.0,
        reward_stagnation_penalty=0.0,
        reward_helper_materialization_bonus=0.0,
        success=False,
        local_llm_call_count=0,
        local_tokens_prompt=0,
        local_tokens_completion=0,
        local_tokens_total=0,
        local_wall_sec=0.0,
        override_reason="",
    )


def test_without_binding_decisions_are_unchanged(tmp_path: Path) -> None:
    controller = _controller(tmp_path)
    decision = _decide(controller, advice=_advice())

    assert controller.decision_model_role == "off"
    assert decision.decision_model == {}
    assert decision.decision_strategy == "linucb_greedy"
    assert not any(name in decision.feature_values for name in DM_FEATURE_NAMES)


def test_thompson_greedy_strategy_is_labelled_as_thompson(tmp_path: Path) -> None:
    decision = _decide(_controller(tmp_path, algorithm="thompson"))
    assert decision.decision_strategy == "thompson_greedy"


def test_shadow_role_logs_advice_without_changing_selection(tmp_path: Path) -> None:
    plain = _decide(_controller(tmp_path / "plain", warmup=5, epsilon=0.3))
    shadow = _decide(
        _controller(tmp_path / "shadow", role="shadow", warmup=5, epsilon=0.3), _advice()
    )

    assert shadow.selected_action == plain.selected_action
    assert shadow.propensity == plain.propensity
    assert shadow.decision_model["role"] == "shadow"
    assert shadow.decision_model["status"] == "ok"
    assert shadow.decision_model["top_action"] == "file_only"


def test_policy_role_selects_from_the_decision_model(tmp_path: Path) -> None:
    controller = _controller(tmp_path, role="policy", warmup=50)
    decision = _decide(controller, _advice("frame_only"))

    assert decision.selected_action == "frame_only"
    assert decision.applied_action == "frame_only"
    assert decision.decision_strategy == "decision_model_greedy"
    assert decision.propensity == 1.0
    assert decision.warmup_active is False
    assert decision.decision_model["effect"]["distribution"]["frame_only"] == 1.0


def test_policy_role_falls_back_to_the_baseline_action(tmp_path: Path) -> None:
    controller = _controller(tmp_path, role="policy")
    decision = _decide(controller, _advice(status="timeout"))

    assert decision.selected_action == "invoke_helper"
    assert decision.decision_strategy == "decision_model_unavailable"
    assert decision.propensity == 1.0
    assert decision.decision_model["status"] == "timeout"


def test_features_role_adds_compact_dm_features(tmp_path: Path) -> None:
    controller = _controller(tmp_path, role="features")

    with_advice = _decide(controller, _advice())
    without = _decide(controller, None)

    assert with_advice.feature_values["dm_available"] == 1.0
    assert with_advice.feature_values["dm_need_source_definition"] == 0.9
    assert without.feature_values["dm_available"] == 0.0
    assert set(DM_FEATURE_NAMES) <= set(controller.state.feature_names)


def test_prior_role_guides_warmup_with_valid_propensities(tmp_path: Path) -> None:
    controller = _controller(tmp_path, role="prior", warmup=10, warmup_mix=0.2)
    decision = _decide(controller, _advice("file_only"))

    distribution = decision.decision_model["effect"]["warmup_distribution"]
    assert decision.decision_strategy == "decision_model_guided_warmup"
    assert decision.warmup_active is True
    assert math.isclose(sum(distribution.values()), 1.0)
    assert math.isclose(decision.propensity, distribution[decision.selected_action])
    assert math.isclose(distribution["file_only"], 0.8 * 0.7 + 0.2 / 4)


def test_prior_role_breaks_ties_after_warmup_but_logs_raw_scores(tmp_path: Path) -> None:
    controller = _controller(tmp_path, role="prior", prior_beta=0.5)
    decision = _decide(controller, _advice("frame_only"))

    assert decision.selected_action == "frame_only"
    assert decision.decision_strategy == "linucb_greedy"
    assert len({decision.scores[action] for action in ACTIONS}) == 1  # untouched bandit scores
    effect = decision.decision_model["effect"]
    assert math.isclose(effect["prior_beta"], 0.5)
    assert effect["prior_adjustment"]["frame_only"] > 0.0


def test_events_carry_the_decision_model_block_since_schema_v5(tmp_path: Path) -> None:
    controller = _controller(tmp_path, role="shadow")
    event = _record(controller, _decide(controller, _advice()))
    payload = event.to_json()

    # Schema 5 added the block; later versions keep it (6: POL-01, 7: POL-21).
    assert EVENT_SCHEMA_VERSION >= 5
    assert payload["schema_version"] == EVENT_SCHEMA_VERSION
    assert payload["decision_model"]["role"] == "shadow"
    logged = json.loads(
        (tmp_path / "events_shadow_on.jsonl").read_text(encoding="utf-8").splitlines()[0]
    )
    assert logged["decision_model"]["top_action"] == "file_only"


def test_shadow_mode_behavior_propensity_is_one(tmp_path: Path) -> None:
    controller = _controller(tmp_path, mode="shadow", warmup=10)
    decision = _decide(controller)
    while decision.selected_action != decision.applied_action:
        decision = _decide(controller)

    event = _record(controller, decision)

    assert decision.mode == "shadow"
    assert decision.propensity < 1.0
    assert event.behavior_propensity == 1.0


def test_on_mode_behavior_propensity_keeps_the_policy_probability(tmp_path: Path) -> None:
    controller = _controller(tmp_path, warmup=10)
    decision = _decide(controller)
    event = _record(controller, decision)

    assert math.isclose(event.behavior_propensity, decision.propensity)
    assert math.isclose(decision.propensity, 1.0 / len(ACTIONS))


def test_close_releases_the_advisor(tmp_path: Path) -> None:
    controller = _controller(tmp_path, role="shadow")
    assert controller.decision_model is not None
    advisor = controller.decision_model.advisor
    assert isinstance(advisor, _ClosingAdvisor)

    controller.close()

    assert advisor.closed is True
