"""Off-policy evaluation of the decision-model helper targets.

Covers the per-sample (stochastic) estimator against analytically known values, the
read-time shadow-propensity correction, and the baseline fallback used when a decision
model produced no usable advice.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from evals import analyze_results as ar
from evals import eval_summary as es

_ONLINE_ACTION_SPACE = set(ar.ADAPTIVE_ONLINE_ACTIONS)
_TWO_ACTIONS = {"skip_helper", "invoke_helper"}


def _sample(
    *,
    action: str,
    propensity: float,
    reward: float,
    features: dict[str, float] | None = None,
    dm_action_probs: dict[str, float] | None = None,
    dm_status: str = "",
    dm_atomic_action: str = "",
    available_actions: list[str] | None = None,
    dm_traceback_sufficient: float | None = None,
    dm_missing_information: dict[str, float] | None = None,
) -> dict[str, Any]:
    return {
        "action": action,
        "propensity": propensity,
        "reward": reward,
        "features": dict(features or {}),
        "feature_scaling_profile": "v1",
        "propensity_shadow_corrected": False,
        "available_actions": list(available_actions or []),
        "dm_status": dm_status,
        "dm_action_probs": dict(dm_action_probs or {}),
        "dm_atomic_action": dm_atomic_action,
        "dm_traceback_sufficient": dm_traceback_sufficient,
        "dm_missing_information": dict(dm_missing_information or {}),
    }


def _settings(
    *,
    action_space: set[str] | None = None,
    min_action_support: int = 2,
) -> ar.StochasticTargetOpeSettings:
    return ar.StochasticTargetOpeSettings(
        action_space=set(action_space or _TWO_ACTIONS),
        min_action_support=min_action_support,
        ridge_alpha=ar.ADAPTIVE_OPE_RIDGE_ALPHA,
        propensity_floor=1e-3,
    )


def _record(
    *,
    case_id: str,
    attempt: int,
    attempt_metadata_extra: dict[str, Any],
) -> dict[str, Any]:
    metadata = {
        "attempt": attempt,
        "adaptive_online_reward_scope": ar.ADAPTIVE_ONLINE_REWARD_SCOPE,
    }
    metadata.update(attempt_metadata_extra)
    return {
        "schema_version": ar.EVAL_RECORD_SCHEMA_VERSION,
        "run_id": "run-dm-ope",
        "case_id": case_id,
        "condition": "layered_adaptive_llm",
        "attempt": attempt,
        "success": False,
        "attempt_metadata": metadata,
    }


def test_stochastic_target_ips_snips_ess_match_closed_form() -> None:
    samples = [
        _sample(action="skip_helper", propensity=0.5, reward=1.0),
        _sample(action="invoke_helper", propensity=0.25, reward=0.0),
        _sample(action="skip_helper", propensity=0.5, reward=0.0),
        _sample(action="invoke_helper", propensity=0.25, reward=1.0),
    ]
    distribution = {"skip_helper": 0.75, "invoke_helper": 0.25}
    target = ar._stochastic_target_ope(
        samples=samples,
        target_distributions=[dict(distribution) for _ in samples],
        settings=_settings(),
    )

    # weights: 0.75/0.5, 0.25/0.25, 0.75/0.5, 0.25/0.25 = 1.5, 1.0, 1.5, 1.0
    assert target["target_kind"] == "stochastic"
    assert target["ips"] == pytest.approx(2.5 / 4.0)
    assert target["snips"] == pytest.approx(2.5 / 5.0)
    assert target["effective_sample_size"] == pytest.approx(25.0 / 6.5)
    assert target["matched_samples"] == 4
    assert target["total_samples"] == 4
    assert target["support_rate"] == pytest.approx(1.0)
    assert target["support_warning"] == ""
    assert target["mean_action_probabilities"] == {
        "skip_helper": pytest.approx(0.75),
        "invoke_helper": pytest.approx(0.25),
    }


def test_stochastic_one_hot_target_matches_deterministic_target() -> None:
    samples = [
        _sample(action="skip_helper", propensity=0.5, reward=1.0, features={"x": 0.0}),
        _sample(action="invoke_helper", propensity=0.25, reward=0.0, features={"x": 1.0}),
        _sample(action="skip_helper", propensity=0.4, reward=0.5, features={"x": 1.0}),
        _sample(action="invoke_helper", propensity=0.5, reward=1.0, features={"x": 0.0}),
    ]
    one_hot = [{"skip_helper": 1.0, "invoke_helper": 0.0} for _ in samples]
    stochastic = ar._stochastic_target_ope(
        samples=samples,
        target_distributions=one_hot,
        settings=_settings(),
    )
    deterministic = ar._deterministic_target_ope(
        samples=samples,
        target_action="skip_helper",
        action_space=set(_TWO_ACTIONS),
        action_counts=ar._action_counts_for_samples(samples=samples, action_space=_TWO_ACTIONS),
        min_action_support=2,
        ridge_alpha=ar.ADAPTIVE_OPE_RIDGE_ALPHA,
        propensity_floor=1e-3,
    )

    for key in ("ips", "snips", "effective_sample_size", "matched_samples", "total_samples"):
        assert stochastic[key] == pytest.approx(deterministic[key])


def test_stochastic_doubly_robust_uses_fitted_action_models() -> None:
    # Reward is a deterministic function of the action, so the per-action ridge models
    # predict it exactly and the DR correction term vanishes.
    # Three copies so every cross-fitting fold (STA-11) has both action models.
    samples = [
        _sample(action="skip_helper", propensity=0.5, reward=1.0, features={"x": 0.0}),
        _sample(action="skip_helper", propensity=0.5, reward=1.0, features={"x": 1.0}),
        _sample(action="invoke_helper", propensity=0.5, reward=0.0, features={"x": 0.0}),
        _sample(action="invoke_helper", propensity=0.5, reward=0.0, features={"x": 1.0}),
    ] * 3
    target = ar._stochastic_target_ope(
        samples=samples,
        target_distributions=[{"skip_helper": 0.75, "invoke_helper": 0.25} for _ in samples],
        settings=_settings(),
    )

    assert target["dr_supported"] is True
    assert target["dr_reason"] == ""
    assert target["dr"] == pytest.approx(0.75, abs=1e-3)


def test_stochastic_doubly_robust_unsupported_without_action_model() -> None:
    samples = [
        _sample(action="skip_helper", propensity=0.5, reward=1.0),
        _sample(action="skip_helper", propensity=0.5, reward=1.0),
    ]
    target = ar._stochastic_target_ope(
        samples=samples,
        target_distributions=[{"skip_helper": 0.5, "file_only": 0.5} for _ in samples],
        settings=_settings(action_space=_ONLINE_ACTION_SPACE),
    )

    assert target["dr"] is None
    assert target["dr_supported"] is False
    assert target["dr_reason"] == "insufficient_action_support"
    # IPS/SNIPS stay available: half the target mass is on an action the logs did take.
    assert target["ips"] == pytest.approx(1.0)
    assert target["snips"] == pytest.approx(1.0)


def test_stochastic_dr_support_reason_distinguishes_missing_models() -> None:
    beta = np.zeros((2,), dtype=float)
    distributions = [{"skip_helper": 0.5, "invoke_helper": 0.5}]

    assert (
        ar._stochastic_dr_support_reason(
            target_distributions=distributions,
            models={"skip_helper": beta, "invoke_helper": beta},
            action_counts={"skip_helper": 4, "invoke_helper": 4},
            min_action_support=2,
        )
        == ""
    )
    # Enough logged samples but no fitted model is a different failure than thin support.
    assert (
        ar._stochastic_dr_support_reason(
            target_distributions=distributions,
            models={"skip_helper": beta},
            action_counts={"skip_helper": 4, "invoke_helper": 4},
            min_action_support=2,
        )
        == "missing_action_model"
    )
    assert (
        ar._stochastic_dr_support_reason(
            target_distributions=distributions,
            models={"skip_helper": beta},
            action_counts={"skip_helper": 4, "invoke_helper": 1},
            min_action_support=2,
        )
        == "insufficient_action_support"
    )


def test_stochastic_target_reports_no_samples() -> None:
    target = ar._stochastic_target_ope(
        samples=[],
        target_distributions=[],
        settings=_settings(),
    )

    assert target["ips"] is None
    assert target["snips"] is None
    assert target["dr"] is None
    assert target["dr_reason"] == "no_samples"
    assert target["support_warning"] == "no_samples"
    assert target["mean_action_probabilities"] == {}


def test_decision_model_target_distributions_follow_advice() -> None:
    sample = _sample(
        action="skip_helper",
        propensity=0.5,
        reward=1.0,
        dm_status="ok",
        dm_action_probs={"skip_helper": 0.2, "invoke_helper": 0.6, "file_only": 0.2},
        dm_atomic_action="frame_only",
        available_actions=["skip_helper", "invoke_helper", "frame_only"],
    )

    greedy = ar._decision_model_target_distribution(
        sample, kind="greedy", action_space=_ONLINE_ACTION_SPACE
    )
    stochastic = ar._decision_model_target_distribution(
        sample, kind="stochastic", action_space=_ONLINE_ACTION_SPACE
    )
    atomic = ar._decision_model_target_distribution(
        sample, kind="atomic", action_space=_ONLINE_ACTION_SPACE
    )

    # ``file_only`` is infeasible for this sample, so its mass is dropped and the rest
    # renormalized over the logged feasible set.
    assert greedy == {"skip_helper": 0.0, "invoke_helper": 1.0, "frame_only": 0.0}
    assert stochastic == {
        "skip_helper": pytest.approx(0.25),
        "invoke_helper": pytest.approx(0.75),
        "frame_only": pytest.approx(0.0),
    }
    assert atomic == {"skip_helper": 0.0, "invoke_helper": 0.0, "frame_only": 1.0}


def test_decision_model_greedy_breaks_ties_in_controller_action_order() -> None:
    sample = _sample(
        action="skip_helper",
        propensity=0.5,
        reward=1.0,
        dm_status="ok",
        dm_action_probs={"file_only": 0.5, "invoke_helper": 0.5},
        available_actions=["file_only", "invoke_helper"],
    )

    greedy = ar._decision_model_target_distribution(
        sample, kind="greedy", action_space=_ONLINE_ACTION_SPACE
    )

    # ``invoke_helper`` precedes ``file_only`` in ONLINE_POLICY_ACTIONS.
    assert greedy == {"invoke_helper": 1.0, "file_only": 0.0}


@pytest.mark.parametrize(
    ("available_actions", "expected_baseline"),
    [
        # invoke_helper first, then skip_helper, then the first feasible action.
        (["skip_helper", "invoke_helper", "file_only"], "invoke_helper"),
        (["skip_helper", "file_only"], "skip_helper"),
        (["search_only", "file_only"], "file_only"),
    ],
)
def test_decision_model_target_falls_back_to_baseline_action(
    available_actions: list[str], expected_baseline: str
) -> None:
    sample = _sample(
        action="skip_helper",
        propensity=0.5,
        reward=1.0,
        dm_status="timeout",
        dm_action_probs={"skip_helper": 0.9},
        dm_atomic_action="frame_only",
        available_actions=available_actions,
    )

    for kind in ("greedy", "stochastic", "atomic"):
        distribution = ar._decision_model_target_distribution(
            sample, kind=kind, action_space=_ONLINE_ACTION_SPACE
        )
        assert distribution[expected_baseline] == 1.0, kind
        assert sum(distribution.values()) == pytest.approx(1.0)
        assert set(distribution) == set(available_actions)


@pytest.mark.parametrize(
    ("sufficient", "missing", "expected"),
    [
        (0.8, {"source_definition": 0.9}, "skip_helper"),
        (0.3, {"source_definition": 0.6, "none": 0.2}, "file_only"),
        (0.3, {"runtime_values": 0.5, "call_path": 0.1}, "frame_only"),
        (0.3, {"source_definition": 0.45, "none": 0.4}, "invoke_helper"),
        (0.3, {}, "invoke_helper"),
        (None, {"project_usage": 0.7, "none": 0.1}, "search_only"),
    ],
)
def test_decision_model_atomic_target_derives_mapping_from_aux_answers(
    sufficient: float | None, missing: dict[str, float], expected: str
) -> None:
    """Profiles that log aux answers but no mapped action still get a real atomic target."""
    sample = _sample(
        action="skip_helper",
        propensity=0.5,
        reward=1.0,
        dm_status="ok",
        dm_action_probs={"skip_helper": 0.9, "invoke_helper": 0.1},
        dm_traceback_sufficient=sufficient,
        dm_missing_information=missing,
    )

    atomic = ar._decision_model_target_distribution(
        sample, kind="atomic", action_space=_ONLINE_ACTION_SPACE
    )

    assert atomic[expected] == 1.0
    assert sum(atomic.values()) == pytest.approx(1.0)


def test_decision_model_atomic_target_prefers_logged_mapping() -> None:
    sample = _sample(
        action="skip_helper",
        propensity=0.5,
        reward=1.0,
        dm_status="ok",
        dm_atomic_action="frame_only",
        dm_traceback_sufficient=0.9,
    )

    atomic = ar._decision_model_target_distribution(
        sample, kind="atomic", action_space=_ONLINE_ACTION_SPACE
    )

    assert atomic["frame_only"] == 1.0


def test_decision_model_target_defaults_to_full_action_space() -> None:
    """Samples logged without a feasible-action list fall back to the full action space."""
    sample = _sample(
        action="skip_helper",
        propensity=0.5,
        reward=1.0,
        dm_status="ok",
        dm_action_probs={"skip_helper": 0.25, "invoke_helper": 0.75},
    )

    stochastic = ar._decision_model_target_distribution(
        sample, kind="stochastic", action_space=_ONLINE_ACTION_SPACE
    )

    assert set(stochastic) == _ONLINE_ACTION_SPACE
    assert stochastic["invoke_helper"] == pytest.approx(0.75)
    assert stochastic["frame_only"] == pytest.approx(0.0)

    # Samples predating the available-action telemetry have no key at all.
    without_key = {key: value for key, value in sample.items() if key != "available_actions"}
    assert (
        ar._decision_model_target_distribution(
            without_key, kind="stochastic", action_space=_ONLINE_ACTION_SPACE
        )
        == stochastic
    )
    # An empty action space leaves nothing to score.
    assert ar._decision_model_target_distribution(sample, kind="greedy", action_space=set()) == {}


def test_decision_model_target_falls_back_when_advice_has_no_feasible_mass() -> None:
    sample = _sample(
        action="skip_helper",
        propensity=0.5,
        reward=1.0,
        dm_status="ok",
        dm_action_probs={"search_only": 1.0},
        available_actions=["skip_helper", "invoke_helper"],
    )

    stochastic = ar._decision_model_target_distribution(
        sample, kind="stochastic", action_space=_ONLINE_ACTION_SPACE
    )
    atomic = ar._decision_model_target_distribution(
        sample, kind="atomic", action_space=_ONLINE_ACTION_SPACE
    )

    assert stochastic == {"skip_helper": 0.0, "invoke_helper": 1.0}
    assert atomic == {"skip_helper": 0.0, "invoke_helper": 1.0}


def test_summary_omits_decision_model_targets_without_advice() -> None:
    samples = [
        _sample(action="skip_helper", propensity=0.5, reward=1.0),
        _sample(action="invoke_helper", propensity=0.5, reward=0.0),
    ]
    summary = ar._adaptive_ope_summary(
        samples=samples,
        action_space=set(_TWO_ACTIONS),
        target_policies=dict(ar.ADAPTIVE_ONLINE_TARGET_POLICIES),
        min_action_support=1,
        propensity_floor=0.005,
    )

    assert summary["target_order"] == list(ar.ADAPTIVE_ONLINE_TARGET_POLICY_ORDER)
    assert summary["decision_model_sample_count"] == 0
    assert summary["shadow_corrected_samples"] == 0
    assert not any(name.startswith("decision_model_") for name in summary["targets"])


def test_summary_adds_decision_model_targets_with_advice() -> None:
    samples = [
        _sample(
            action="skip_helper",
            propensity=0.5,
            reward=1.0,
            features={"x": 0.0},
            dm_status="ok",
            dm_action_probs={"skip_helper": 0.8, "invoke_helper": 0.2},
            dm_atomic_action="skip_helper",
            available_actions=["skip_helper", "invoke_helper"],
        ),
        _sample(
            action="invoke_helper",
            propensity=0.5,
            reward=0.0,
            features={"x": 1.0},
            dm_status="ok",
            dm_action_probs={"skip_helper": 0.8, "invoke_helper": 0.2},
            dm_atomic_action="skip_helper",
            available_actions=["skip_helper", "invoke_helper"],
        ),
    ]
    summary = ar._adaptive_ope_summary(
        samples=samples,
        action_space=set(_TWO_ACTIONS),
        target_policies=dict(ar.ADAPTIVE_ONLINE_TARGET_POLICIES),
        min_action_support=1,
        propensity_floor=0.005,
    )

    assert summary["decision_model_sample_count"] == 2
    assert summary["target_order"] == [
        *ar.ADAPTIVE_ONLINE_TARGET_POLICY_ORDER,
        ar.DECISION_MODEL_TARGET_GREEDY,
        ar.DECISION_MODEL_TARGET_STOCHASTIC,
        ar.DECISION_MODEL_TARGET_ATOMIC,
    ]
    greedy = summary["targets"][ar.DECISION_MODEL_TARGET_GREEDY]
    stochastic = summary["targets"][ar.DECISION_MODEL_TARGET_STOCHASTIC]
    # Greedy always picks skip_helper: only the first sample contributes, w = 1/0.5.
    assert greedy["ips"] == pytest.approx(1.0)
    assert greedy["matched_samples"] == 1
    # Stochastic: (0.8/0.5)*1.0 + (0.2/0.5)*0.0, averaged over 2 samples.
    assert stochastic["ips"] == pytest.approx(0.8)
    assert stochastic["matched_samples"] == 2
    assert summary["targets"][ar.DECISION_MODEL_TARGET_ATOMIC]["ips"] == pytest.approx(1.0)


def test_adaptive_ope_samples_correct_shadow_mode_propensity() -> None:
    samples = ar._adaptive_ope_samples(
        records=[
            _record(
                case_id="case-shadow-stale",
                attempt=1,
                attempt_metadata_extra={
                    "adaptive_online_decision_id": "d1",
                    "adaptive_online_policy_mode": "shadow",
                    "adaptive_online_reward": 1.0,
                    "adaptive_online_behavior_action": "skip_helper",
                    "adaptive_online_behavior_propensity": 0.25,
                },
            ),
            _record(
                case_id="case-shadow-fixed",
                attempt=1,
                attempt_metadata_extra={
                    "adaptive_online_decision_id": "d2",
                    "adaptive_online_policy_mode": "shadow",
                    "adaptive_online_reward": 1.0,
                    "adaptive_online_behavior_action": "skip_helper",
                    "adaptive_online_behavior_propensity": 1.0,
                },
            ),
            _record(
                case_id="case-on",
                attempt=1,
                attempt_metadata_extra={
                    "adaptive_online_decision_id": "d3",
                    "adaptive_online_policy_mode": "on",
                    "adaptive_online_reward": 1.0,
                    "adaptive_online_behavior_action": "invoke_helper",
                    "adaptive_online_behavior_propensity": 0.25,
                },
            ),
        ],
        prefix="adaptive_online",
        valid_actions=set(_ONLINE_ACTION_SPACE),
    )

    assert [sample["propensity"] for sample in samples] == [1.0, 1.0, 0.25]
    assert [sample["propensity_shadow_corrected"] for sample in samples] == [True, False, False]

    summary = ar._adaptive_ope_summary(
        samples=samples,
        action_space=set(_ONLINE_ACTION_SPACE),
        target_policies=dict(ar.ADAPTIVE_ONLINE_TARGET_POLICIES),
        min_action_support=1,
        propensity_floor=0.005,
    )
    assert summary["shadow_corrected_samples"] == 1
    # Without the fix the stale shadow row would carry weight 4 instead of 1.
    assert summary["targets"]["always_skip"]["ips"] == pytest.approx(2.0 / 3.0)


def test_adaptive_ope_samples_carry_available_actions_and_advice() -> None:
    samples = ar._adaptive_ope_samples(
        records=[
            _record(
                case_id="case-advice",
                attempt=1,
                attempt_metadata_extra={
                    "adaptive_online_decision_id": "d1",
                    "adaptive_online_policy_mode": "on",
                    "adaptive_online_reward": 0.5,
                    "adaptive_online_behavior_action": "skip_helper",
                    "adaptive_online_behavior_propensity": 0.5,
                    "adaptive_online_available_actions": [
                        "skip_helper",
                        "invoke_helper",
                        "not_an_action",
                    ],
                    "adaptive_online_dm_status": "ok",
                    "adaptive_online_dm_action_probs": {
                        "skip_helper": 0.3,
                        "invoke_helper": 0.7,
                    },
                    "adaptive_online_dm_atomic_action": "invoke_helper",
                    "adaptive_online_dm_traceback_sufficient": 0.25,
                    "adaptive_online_dm_missing_information": {
                        "none": 0.1,
                        "runtime_values": "0.9",
                        "bad": "x",
                    },
                },
            )
        ],
        prefix="adaptive_online",
        valid_actions=set(_ONLINE_ACTION_SPACE),
    )

    assert len(samples) == 1
    sample = samples[0]
    assert sample["available_actions"] == ["skip_helper", "invoke_helper"]
    assert sample["dm_status"] == "ok"
    assert sample["dm_action_probs"] == {"skip_helper": 0.3, "invoke_helper": 0.7}
    assert sample["dm_atomic_action"] == "invoke_helper"
    assert sample["dm_traceback_sufficient"] == pytest.approx(0.25)
    assert sample["dm_missing_information"] == {"none": 0.1, "runtime_values": 0.9}


def test_adaptive_ope_samples_without_decision_model_keys() -> None:
    samples = ar._adaptive_ope_samples(
        records=[
            _record(
                case_id="case-legacy",
                attempt=1,
                attempt_metadata_extra={
                    "adaptive_online_decision_id": "d1",
                    "adaptive_online_reward": 0.5,
                    "adaptive_online_behavior_action": "skip_helper",
                    "adaptive_online_propensity": 0.5,
                },
            )
        ],
        prefix="adaptive_online",
        valid_actions=set(_ONLINE_ACTION_SPACE),
    )

    assert samples[0]["available_actions"] == []
    assert samples[0]["dm_status"] == ""
    assert samples[0]["dm_action_probs"] == {}
    assert samples[0]["dm_traceback_sufficient"] is None
    assert samples[0]["dm_missing_information"] == {}
    assert samples[0]["propensity"] == pytest.approx(0.5)


def test_summary_from_records_derives_decision_model_targets() -> None:
    advice = {
        "adaptive_online_dm_status": "ok",
        "adaptive_online_dm_action_probs": {"skip_helper": 0.9, "invoke_helper": 0.1},
        "adaptive_online_dm_atomic_action": "skip_helper",
        "adaptive_online_available_actions": ["skip_helper", "invoke_helper"],
    }
    skip_row = _record(
        case_id="case-1",
        attempt=1,
        attempt_metadata_extra={
            "adaptive_online_decision_id": "d1",
            "adaptive_online_policy_mode": "on",
            "adaptive_online_reward": 1.0,
            "adaptive_online_behavior_action": "skip_helper",
            "adaptive_online_behavior_propensity": 0.5,
            **advice,
        },
    )
    # The second row carries its policy metadata in patch_meta only, exercising the fallback.
    invoke_row = _record(
        case_id="case-2",
        attempt=1,
        attempt_metadata_extra={},
    )
    invoke_row["attempt_metadata"] = {"attempt": 1}
    invoke_row["patch_meta"] = {
        "adaptive_online_decision_id": "d2",
        "adaptive_online_reward_scope": ar.ADAPTIVE_ONLINE_REWARD_SCOPE,
        "adaptive_online_policy_mode": "on",
        "adaptive_online_reward": 0.0,
        "adaptive_online_behavior_action": "invoke_helper",
        "adaptive_online_behavior_propensity": 0.5,
        **advice,
    }

    summary = ar._adaptive_ope_summary_from_records(
        records=[skip_row, invoke_row],
        prefix="adaptive_online",
        action_space=set(_ONLINE_ACTION_SPACE),
        target_policies=dict(ar.ADAPTIVE_ONLINE_TARGET_POLICIES),
        min_action_support=1,
        propensity_floor=0.005,
    )

    assert summary["sample_count"] == 2
    assert summary["decision_model_sample_count"] == 2
    greedy = summary["targets"][ar.DECISION_MODEL_TARGET_GREEDY]
    stochastic = summary["targets"][ar.DECISION_MODEL_TARGET_STOCHASTIC]
    assert greedy["ips"] == pytest.approx(1.0)
    assert stochastic["ips"] == pytest.approx(0.9)
    assert stochastic["mean_action_probabilities"]["skip_helper"] == pytest.approx(0.9)


def test_ope_target_order_prefers_summary_order() -> None:
    ope = {
        "target_order": ["always_skip", ar.DECISION_MODEL_TARGET_GREEDY],
        "targets": {
            "always_skip": {},
            ar.DECISION_MODEL_TARGET_GREEDY: {},
            ar.DECISION_MODEL_TARGET_ATOMIC: {},
        },
    }

    assert ar._ope_target_order(ope, ordered_targets=("always_skip",)) == [
        "always_skip",
        ar.DECISION_MODEL_TARGET_GREEDY,
        ar.DECISION_MODEL_TARGET_ATOMIC,
    ]
    # Summaries written before target_order existed keep the caller's ordering.
    legacy = {"targets": {"always_invoke_helper": {}, "always_skip": {}}}
    assert ar._ope_target_order(
        legacy, ordered_targets=("always_skip", "always_invoke_helper")
    ) == ["always_skip", "always_invoke_helper"]


def test_renderers_include_decision_model_targets() -> None:
    summary = ar._adaptive_ope_summary(
        samples=[
            _sample(
                action="skip_helper",
                propensity=0.5,
                reward=1.0,
                dm_status="ok",
                dm_action_probs={"skip_helper": 0.6, "invoke_helper": 0.4},
                available_actions=["skip_helper", "invoke_helper"],
            ),
            _sample(
                action="invoke_helper",
                propensity=0.5,
                reward=0.0,
                dm_status="ok",
                dm_action_probs={"skip_helper": 0.6, "invoke_helper": 0.4},
                available_actions=["skip_helper", "invoke_helper"],
            ),
        ],
        action_space=set(_TWO_ACTIONS),
        target_policies=dict(ar.ADAPTIVE_ONLINE_TARGET_POLICIES),
        min_action_support=1,
        propensity_floor=0.005,
    )

    analyzer_line = ar._format_ope_targets(
        summary, ordered_targets=ar.ADAPTIVE_ONLINE_TARGET_POLICY_ORDER
    )
    summary_line = es._format_ope_targets(
        summary, ordered_targets=ar.ADAPTIVE_ONLINE_TARGET_POLICY_ORDER
    )

    for line in (analyzer_line, summary_line):
        assert "always_skip: ips=" in line
        assert f"{ar.DECISION_MODEL_TARGET_GREEDY}: ips=" in line
        assert f"{ar.DECISION_MODEL_TARGET_STOCHASTIC}: ips=" in line
        assert f"{ar.DECISION_MODEL_TARGET_ATOMIC}: ips=" in line
    assert analyzer_line.index("always_skip") < analyzer_line.index(ar.DECISION_MODEL_TARGET_GREEDY)


def test_renderers_unchanged_without_decision_model_targets() -> None:
    summary = ar._adaptive_ope_summary(
        samples=[
            _sample(action="skip_helper", propensity=0.5, reward=1.0),
            _sample(action="invoke_helper", propensity=0.5, reward=0.0),
        ],
        action_space=set(_TWO_ACTIONS),
        target_policies=dict(ar.ADAPTIVE_ONLINE_TARGET_POLICIES),
        min_action_support=1,
        propensity_floor=0.005,
    )

    line = ar._format_ope_targets(summary, ordered_targets=ar.ADAPTIVE_ONLINE_TARGET_POLICY_ORDER)

    assert "decision_model_" not in line
    assert [part.split(":")[0] for part in line.split("; ")] == list(
        ar.ADAPTIVE_ONLINE_TARGET_POLICY_ORDER
    )
