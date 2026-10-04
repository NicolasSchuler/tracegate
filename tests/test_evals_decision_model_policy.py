"""Tests for decision-model role math (pure functions)."""

from __future__ import annotations

import math
import random

import pytest

from evals.decision_model import policy, schema
from evals.decision_model.config import DecisionModelPolicyConfig

ACTIONS = ("skip_helper", "invoke_helper", "frame_only", "file_only")


def _advice(**overrides: object) -> policy.DecisionModelAdvice:
    answers = schema.ParsedAnswers(
        action_probs={
            "skip_helper": 0.1,
            "invoke_helper": 0.2,
            "frame_only": 0.6,
            "file_only": 0.1,
        },
        traceback_sufficient=0.2,
        missing_information={
            "none": 0.05,
            "runtime_values": 0.5,
            "call_path": 0.2,
            "source_definition": 0.15,
            "project_usage": 0.05,
            "several_or_unclear": 0.05,
        },
        fix_confidence=1.0 / 3.0,
    )
    values: dict[str, object] = {"status": "ok", "profile": "evidence_v1", "answers": answers}
    values.update(overrides)
    return policy.DecisionModelAdvice(**values)  # type: ignore[arg-type]


def test_feature_values_are_compact_and_zero_without_advice() -> None:
    assert policy.advice_feature_values(None) == dict.fromkeys(policy.DM_FEATURE_NAMES, 0.0)
    assert policy.advice_feature_values(_advice(status="timeout"))["dm_available"] == 0.0

    values = policy.advice_feature_values(_advice())

    assert set(values) == set(policy.DM_FEATURE_NAMES)
    assert len(values) == 6
    assert values["dm_available"] == 1.0
    assert math.isclose(values["dm_need_runtime_values"], 0.7)
    assert math.isclose(values["dm_fix_confidence"], 1.0 / 3.0)
    assert all(0.0 <= value <= 1.0 for value in values.values())
    assert not any(name.endswith("_streak") for name in values)  # v1 scaling passes them through


def _distribution_sums_to_one(distribution: dict[str, float]) -> bool:
    return math.isclose(sum(distribution.values()), 1.0)


def test_policy_role_greedy_mixes_epsilon_and_reports_full_propensity() -> None:
    selection = policy.policy_role_selection(
        _advice(),
        available_actions=ACTIONS,
        config=DecisionModelPolicyConfig(role="policy", temperature=0.0),
        epsilon=0.2,
        rng=random.Random(0),
    )

    assert selection is not None
    distribution = selection.effect["distribution"]
    assert _distribution_sums_to_one(distribution)
    assert math.isclose(distribution["frame_only"], 0.8 + 0.05)
    assert math.isclose(selection.propensity, distribution[selection.action])
    assert selection.warmup_active is False
    expected_label = (
        "decision_model_greedy" if selection.action == "frame_only" else "decision_model_epsilon"
    )
    assert selection.strategy == expected_label


def test_policy_role_tempered_sampling_and_atomic_profile() -> None:
    sampled = policy.policy_role_selection(
        _advice(),
        available_actions=ACTIONS,
        config=DecisionModelPolicyConfig(role="policy", temperature=1.0),
        epsilon=0.0,
        rng=random.Random(1),
    )
    atomic = policy.policy_role_selection(
        _advice(answers=schema.ParsedAnswers(), atomic_action="file_only"),
        available_actions=ACTIONS,
        config=DecisionModelPolicyConfig(role="policy"),
        epsilon=0.0,
        rng=random.Random(2),
    )

    assert sampled is not None and sampled.strategy == "decision_model_sample"
    assert math.isclose(sampled.effect["distribution"]["frame_only"], 0.6)
    assert atomic is not None
    assert (atomic.action, atomic.strategy, atomic.propensity) == (
        "file_only",
        "decision_model_atomic",
        1.0,
    )


def test_policy_role_returns_none_for_unusable_advice() -> None:
    config = DecisionModelPolicyConfig(role="policy")
    for advice in (None, _advice(status="timeout"), _advice(answers=schema.ParsedAnswers())):
        assert (
            policy.policy_role_selection(
                advice, available_actions=ACTIONS, config=config, epsilon=0.1, rng=random.Random()
            )
            is None
        )


def test_guided_warmup_keeps_uniform_support() -> None:
    selection = policy.guided_warmup_selection(
        _advice(),
        available_actions=ACTIONS,
        config=DecisionModelPolicyConfig(role="prior", warmup_mix=0.4),
        rng=random.Random(3),
    )

    assert selection is not None
    distribution = selection.effect["warmup_distribution"]
    assert _distribution_sums_to_one(distribution)
    assert min(distribution.values()) >= 0.4 / len(ACTIONS) - 1e-12
    assert selection.strategy == "decision_model_guided_warmup"
    assert selection.warmup_active is True


def test_prior_adjustment_is_centered_and_anneals() -> None:
    config = DecisionModelPolicyConfig(role="prior", prior_beta=0.5, prior_anneal_updates=10)
    early = policy.prior_score_adjustment(
        _advice(), available_actions=ACTIONS, config=config, updates=0
    )
    late = policy.prior_score_adjustment(
        _advice(), available_actions=ACTIONS, config=config, updates=10
    )

    assert early is not None and late is not None
    assert math.isclose(early["prior_beta"], 0.5)
    assert math.isclose(late["prior_beta"], 0.25)
    assert math.isclose(sum(early["prior_adjustment"].values()), 0.0, abs_tol=1e-12)
    assert max(early["prior_adjustment"], key=early["prior_adjustment"].get) == "frame_only"
    assert (
        policy.prior_score_adjustment(None, available_actions=ACTIONS, config=config, updates=0)
        is None
    )


@pytest.mark.parametrize(
    ("answers", "available", "expected"),
    [
        (
            schema.ParsedAnswers(traceback_sufficient=0.9),
            ACTIONS,
            ("skip_helper", "traceback_sufficient"),
        ),
        (
            schema.ParsedAnswers(
                traceback_sufficient=0.1,
                missing_information={"runtime_values": 0.8, "none": 0.2},
            ),
            ACTIONS,
            ("frame_only", "missing_runtime_values"),
        ),
        (
            schema.ParsedAnswers(
                traceback_sufficient=0.1,
                missing_information={"project_usage": 0.9, "none": 0.1},
            ),
            ("skip_helper", "invoke_helper"),
            ("invoke_helper", "missing_project_usage_fallback_invoke_helper"),
        ),
        (
            schema.ParsedAnswers(
                traceback_sufficient=0.1,
                missing_information={"source_definition": 0.45, "runtime_values": 0.4},
            ),
            ACTIONS,
            ("invoke_helper", "missing_information_uncertain"),
        ),
        (
            schema.ParsedAnswers(),
            ("frame_only",),
            ("frame_only", "missing_information_unavailable_fallback_first_available"),
        ),
    ],
)
def test_atomic_mapping(
    answers: schema.ParsedAnswers,
    available: tuple[str, ...],
    expected: tuple[str, str],
) -> None:
    assert (
        policy.atomic_action_from_answers(
            answers, available_actions=available, thresholds=schema.AtomicThresholds()
        )
        == expected
    )


def test_atomic_mapping_targets_are_helper_actions() -> None:
    assert set(policy._MISSING_TO_ACTION.values()) <= set(schema.HELPER_ACTIONS)


def test_advice_record_and_metadata_fields() -> None:
    assert policy.advice_record(_advice(), role="off") == {}
    assert policy.advice_record(None, role="policy")["status"] == "missing"

    record = policy.advice_record(_advice(latency_ms=12.5), role="shadow", effect={"k": 1})
    fields = policy.decision_model_metadata_fields(record)

    assert record["top_action"] == "frame_only"
    assert math.isclose(record["margin"], 0.4)
    assert fields["adaptive_online_dm_role"] == "shadow"
    assert fields["adaptive_online_dm_status"] == "ok"
    assert fields["adaptive_online_dm_latency_ms"] == 12.5
    assert fields["adaptive_online_dm_effect"] == {"k": 1}

    defaults = policy.decision_model_metadata_fields(None)
    assert defaults["adaptive_online_dm_role"] == "off"
    assert defaults["adaptive_online_dm_action_probs"] == {}
    assert defaults["adaptive_online_dm_traceback_sufficient"] is None
    defaults["adaptive_online_dm_action_probs"]["x"] = 1.0
    assert policy.decision_model_metadata_fields(None)["adaptive_online_dm_action_probs"] == {}


def test_distribution_summary() -> None:
    assert policy.distribution_summary({}) == ("", 0.0, 0.0, 0.0)
    top, prob, margin, entropy = policy.distribution_summary({"a": 0.5, "b": 0.5})
    assert (top, prob, margin) == ("a", 0.5, 0.0)
    assert math.isclose(entropy, math.log(2.0))
