"""Tests for the adaptive budget online policy controller and state round-trips."""

from __future__ import annotations

import math

import pytest

from evals.adaptive_budget_online_policy import (
    ACTION_BUDGET_FAMILY_CONTEXT,
    ACTION_BUDGET_FAMILY_TIME,
    ACTION_BUDGET_FAMILY_TOKENS,
    ACTION_INCREASE_BUDGET,
    ACTION_KEEP_BUDGET,
    BUDGET_POLICY_ARCH_BWK_V1,
    BUDGET_POLICY_ARCH_HIERARCHICAL_V1,
    BUDGET_POLICY_ARCH_TIER_BINARY,
    STATE_SCHEMA_VERSION,
    AdaptiveBudgetPolicyController,
    BudgetPolicyConfig,
)
from evals.policy_feature_scaling import (
    _V1_COUNT_CAPS,
    DEFAULT_POLICY_FEATURE_SCALING_MIN_SAMPLES,
)
from llmdebug import _json as json


def _linucb_state_payload(
    *,
    action_names: tuple[str, ...],
    feature_names: list[str],
    decisions: int = 0,
    updates: int = 0,
    l2: float = 1.0,
    action_counts: dict[str, int] | None = None,
    A: dict[str, list[list[float]]] | None = None,
    b: dict[str, list[float]] | None = None,
) -> dict[str, object]:
    dim = len(feature_names)
    diagonal = [[l2 if row == col else 0.0 for col in range(dim)] for row in range(dim)]
    return {
        "feature_names": list(feature_names),
        "decisions": decisions,
        "updates": updates,
        "action_counts": {
            action: int((action_counts or {}).get(action, 0)) for action in action_names
        },
        "A": {action: (A or {}).get(action, diagonal) for action in action_names},
        "b": {action: (b or {}).get(action, [0.0] * dim) for action in action_names},
    }


def test_budget_policy_on_updates_and_state_roundtrip(tmp_path) -> None:
    state_path = tmp_path / "budget_policy_state.json"
    events_path = tmp_path / "budget_policy_events.jsonl"
    config = BudgetPolicyConfig.normalize(
        mode="on",
        algorithm="linucb",
        alpha=0.2,
        epsilon=0.0,
        l2=1.0,
        warmup_decisions=0,
        save_every=1,
        state_path=state_path,
        events_path=events_path,
        random_seed=7,
        decay=1.0,
    )
    controller = AdaptiveBudgetPolicyController(config)
    decision = controller.decide(feature_values={"bias": 1.0}, default_action=ACTION_KEEP_BUDGET)
    event = controller.record_outcome(
        decision=decision,
        run_id="run_budget_online",
        case_id="case_a",
        condition="layered_adaptive_llm",
        attempt=1,
        note="tests_still_failing",
        reward_scope="budget_credit_v2",
        reward=0.6,
        reward_success_hint=0.0,
        reward_progress_hint=0.0,
        reward_progress_score=0.5,
        reward_token_penalty=0.2,
        reward_latency_penalty=0.1,
        reward_stagnation_penalty=0.0,
        reward_budget_exceeded_penalty=0.0,
        request_budget_exceeded=False,
        success=False,
        local_llm_call_count=2,
        local_tokens_prompt=200,
        local_tokens_completion=121,
        local_tokens_total=321,
        local_wall_sec=2.4,
        selected_multiplier=1.5,
        applied_multiplier=1.5,
        default_multiplier=1.0,
        default_controls={"propose_budget_multiplier": 1.0},
        selected_controls={"propose_budget_multiplier": 1.5},
        applied_controls={"propose_budget_multiplier": 1.5},
        selected_controls_changed={"propose_budget": True},
        applied_controls_changed={"propose_budget": True},
        partial_override=False,
        partial_override_errors={},
        tier_index_before=0,
        tier_index_after=1,
        override_reason="",
        reward_success_hint_scale=1.2,
        reward_progress_hint_scale=1.25,
        reward_stagnation_penalty_scale=1.3,
    )
    state = controller.state
    controller.close()
    payload = json.loads(events_path.read_text(encoding="utf-8").splitlines()[0])

    assert event.model_updated is True
    assert event.update_reason == "on_policy_update"
    assert event.reward_progress_score == 0.5
    assert event.reward_success_hint_scale == pytest.approx(1.2)
    assert event.reward_progress_hint_scale == pytest.approx(1.25)
    assert event.reward_stagnation_penalty_scale == pytest.approx(1.3)
    assert payload["reward_success_hint_scale"] == pytest.approx(1.2)
    assert payload["reward_progress_hint_scale"] == pytest.approx(1.25)
    assert payload["reward_stagnation_penalty_scale"] == pytest.approx(1.3)
    assert event.applied_controls_changed["propose_budget"] is True
    assert state.decisions == 1
    assert state.updates == 1
    assert state.action_counts[ACTION_KEEP_BUDGET] == 1
    assert events_path.exists()
    assert len([line for line in events_path.read_text(encoding="utf-8").splitlines() if line]) == 1

    reloaded = AdaptiveBudgetPolicyController(config)
    reloaded_state = reloaded.state
    reloaded.close()
    assert reloaded.load_error is None
    assert reloaded_state.decisions == 1
    assert reloaded_state.updates == 1


def test_budget_policy_shadow_logs_but_does_not_update(tmp_path) -> None:
    state_path = tmp_path / "budget_policy_state_shadow.json"
    events_path = tmp_path / "budget_policy_events_shadow.jsonl"
    config = BudgetPolicyConfig.normalize(
        mode="shadow",
        algorithm="linucb",
        alpha=0.0,
        epsilon=0.0,
        l2=1.0,
        warmup_decisions=0,
        save_every=1,
        state_path=state_path,
        events_path=events_path,
        random_seed=0,
        decay=1.0,
    )
    controller = AdaptiveBudgetPolicyController(config)
    decision = controller.decide(feature_values={"bias": 1.0}, default_action=ACTION_KEEP_BUDGET)
    event = controller.record_outcome(
        decision=decision,
        run_id="run_budget_shadow",
        case_id="case_b",
        condition="layered_adaptive_llm",
        attempt=1,
        note="no_patch_proposed",
        reward_scope="budget_credit_v2",
        reward=-0.2,
        reward_success_hint=0.0,
        reward_progress_hint=0.0,
        reward_progress_score=0.0,
        reward_token_penalty=0.1,
        reward_latency_penalty=0.1,
        reward_stagnation_penalty=0.0,
        reward_budget_exceeded_penalty=0.5,
        request_budget_exceeded=True,
        success=False,
        local_llm_call_count=1,
        local_tokens_prompt=8,
        local_tokens_completion=2,
        local_tokens_total=10,
        local_wall_sec=0.3,
        selected_multiplier=1.5,
        applied_multiplier=1.0,
        default_multiplier=1.0,
        default_controls={"propose_budget_multiplier": 1.0},
        selected_controls={"propose_budget_multiplier": 1.5},
        applied_controls={"propose_budget_multiplier": 1.0},
        selected_controls_changed={"propose_budget": True},
        applied_controls_changed={"propose_budget": False},
        partial_override=False,
        partial_override_errors={},
        tier_index_before=0,
        tier_index_after=0,
        override_reason="shadow_mode_baseline",
    )
    state = controller.state
    controller.close()

    assert decision.applied_action == ACTION_KEEP_BUDGET
    assert event.model_updated is False
    assert event.update_reason == "shadow_no_update"
    assert state.decisions == 1
    assert state.updates == 0
    assert len([line for line in events_path.read_text(encoding="utf-8").splitlines() if line]) == 1


def test_budget_policy_roundtrip_without_decisions(tmp_path) -> None:
    state_path = tmp_path / "budget_policy_state_empty.json"
    config = BudgetPolicyConfig.normalize(
        mode="on",
        algorithm="linucb",
        state_path=state_path,
        events_path=tmp_path / "budget_policy_events_empty.jsonl",
        decay=1.0,
    )
    controller = AdaptiveBudgetPolicyController(config)
    controller.close()

    assert state_path.exists()
    reloaded = AdaptiveBudgetPolicyController(config)
    state = reloaded.state
    reloaded.close()

    assert reloaded.load_error is None
    assert state.decisions == 0
    assert state.updates == 0
    assert state.feature_names == []


def test_budget_policy_reports_load_error_for_invalid_state(tmp_path) -> None:
    state_path = tmp_path / "budget_invalid_state.json"
    state_path.write_text(
        json.dumps({"schema_version": STATE_SCHEMA_VERSION, "algorithm": "linucb"}),
        encoding="utf-8",
    )
    config = BudgetPolicyConfig.normalize(
        mode="on",
        algorithm="linucb",
        state_path=state_path,
        events_path=tmp_path / "events.jsonl",
        decay=1.0,
    )
    controller = AdaptiveBudgetPolicyController(config)
    state = controller.state
    controller.close()

    assert controller.load_error is not None
    assert state.feature_names == []
    assert state.decisions == 0


def test_budget_policy_rejects_schema_v1_state(tmp_path) -> None:
    state_path = tmp_path / "budget_schema_v1_state.json"
    state_path.write_text(
        json.dumps({"schema_version": 1, "algorithm": "linucb"}),
        encoding="utf-8",
    )
    config = BudgetPolicyConfig.normalize(
        mode="on",
        algorithm="linucb",
        state_path=state_path,
        events_path=tmp_path / "events.jsonl",
        decay=1.0,
    )
    controller = AdaptiveBudgetPolicyController(config)
    state = controller.state
    controller.close()

    assert controller.load_error is not None
    assert "unsupported schema version" in str(controller.load_error)
    assert state.feature_names == []
    assert state.decisions == 0


def test_budget_policy_expands_loaded_state_for_new_feature_dimensions(tmp_path) -> None:
    state_path = tmp_path / "budget_policy_state_expand.json"
    events_path = tmp_path / "budget_policy_events_expand.jsonl"
    state_path.write_text(
        json.dumps(
            {
                "schema_version": STATE_SCHEMA_VERSION,
                "mode": "on",
                "algorithm": "linucb",
                "architecture": BUDGET_POLICY_ARCH_TIER_BINARY,
                "alpha": 0.2,
                "epsilon": 0.0,
                "l2": 1.5,
                "warmup_decisions": 0,
                "save_every": 1,
                "gate_state": _linucb_state_payload(
                    action_names=(ACTION_KEEP_BUDGET, ACTION_INCREASE_BUDGET),
                    feature_names=["bias"],
                    decisions=3,
                    updates=2,
                    l2=1.5,
                    action_counts={
                        ACTION_KEEP_BUDGET: 1,
                        ACTION_INCREASE_BUDGET: 2,
                    },
                    A={
                        ACTION_KEEP_BUDGET: [[3.0]],
                        ACTION_INCREASE_BUDGET: [[4.0]],
                    },
                    b={
                        ACTION_KEEP_BUDGET: [0.7],
                        ACTION_INCREASE_BUDGET: [0.9],
                    },
                ),
                "family_state": _linucb_state_payload(
                    action_names=(
                        ACTION_BUDGET_FAMILY_TIME,
                        ACTION_BUDGET_FAMILY_TOKENS,
                        ACTION_BUDGET_FAMILY_CONTEXT,
                    ),
                    feature_names=[],
                ),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    config = BudgetPolicyConfig.normalize(
        mode="on",
        algorithm="linucb",
        alpha=0.2,
        epsilon=0.0,
        l2=1.5,
        warmup_decisions=0,
        save_every=1,
        state_path=state_path,
        events_path=events_path,
        random_seed=0,
        decay=1.0,
    )
    controller = AdaptiveBudgetPolicyController(config)
    controller.decide(
        feature_values={"bias": 1.0, "attempt_frac": 0.5},
        default_action=ACTION_KEEP_BUDGET,
    )
    state = controller.state
    controller.close()

    assert state.feature_names == ["attempt_frac", "bias"]
    for action in (ACTION_KEEP_BUDGET, ACTION_INCREASE_BUDGET):
        assert state.A[action].shape == (2, 2)
        assert state.b[action].shape == (2,)

    bias_idx = state.feature_names.index("bias")
    new_feature_idx = state.feature_names.index("attempt_frac")
    assert state.A[ACTION_KEEP_BUDGET][bias_idx, bias_idx] == 3.0
    assert state.A[ACTION_INCREASE_BUDGET][bias_idx, bias_idx] == 4.0
    assert state.b[ACTION_KEEP_BUDGET][bias_idx] == 0.7
    assert state.b[ACTION_INCREASE_BUDGET][bias_idx] == 0.9
    assert state.A[ACTION_KEEP_BUDGET][new_feature_idx, new_feature_idx] == 1.5
    assert state.A[ACTION_INCREASE_BUDGET][new_feature_idx, new_feature_idx] == 1.5


def test_budget_policy_decay_discounts_stale_evidence_before_update(tmp_path) -> None:
    state_path = tmp_path / "budget_policy_state_decay.json"
    events_path = tmp_path / "budget_policy_events_decay.jsonl"
    state_path.write_text(
        json.dumps(
            {
                "schema_version": STATE_SCHEMA_VERSION,
                "mode": "on",
                "algorithm": "linucb",
                "architecture": BUDGET_POLICY_ARCH_TIER_BINARY,
                "alpha": 0.0,
                "epsilon": 0.0,
                "l2": 1.0,
                "warmup_decisions": 0,
                "save_every": 1,
                "gate_state": _linucb_state_payload(
                    action_names=(ACTION_KEEP_BUDGET, ACTION_INCREASE_BUDGET),
                    feature_names=["bias"],
                    l2=1.0,
                    A={
                        ACTION_KEEP_BUDGET: [[1.0]],
                        ACTION_INCREASE_BUDGET: [[3.0]],
                    },
                    b={
                        ACTION_KEEP_BUDGET: [0.0],
                        ACTION_INCREASE_BUDGET: [2.0],
                    },
                ),
                "family_state": _linucb_state_payload(
                    action_names=(
                        ACTION_BUDGET_FAMILY_TIME,
                        ACTION_BUDGET_FAMILY_TOKENS,
                        ACTION_BUDGET_FAMILY_CONTEXT,
                    ),
                    feature_names=[],
                ),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    config = BudgetPolicyConfig.normalize(
        mode="on",
        algorithm="linucb",
        alpha=0.0,
        epsilon=0.0,
        decay=0.5,
        warmup_decisions=0,
        state_path=state_path,
        events_path=events_path,
    )
    controller = AdaptiveBudgetPolicyController(config)
    decision = controller.decide(feature_values={"bias": 1.0}, default_action=ACTION_KEEP_BUDGET)
    controller.record_outcome(
        decision=decision,
        run_id="run_budget_decay",
        case_id="case_budget_decay",
        condition="layered_adaptive_llm",
        attempt=1,
        note="tests_still_failing",
        reward_scope="budget_credit_v2",
        reward=1.0,
        reward_success_hint=0.0,
        reward_progress_hint=0.0,
        reward_progress_score=0.0,
        reward_token_penalty=0.0,
        reward_latency_penalty=0.0,
        reward_stagnation_penalty=0.0,
        reward_budget_exceeded_penalty=0.0,
        request_budget_exceeded=False,
        success=False,
        local_llm_call_count=0,
        local_tokens_prompt=0,
        local_tokens_completion=0,
        local_tokens_total=0,
        local_wall_sec=0.0,
        selected_multiplier=1.5,
        applied_multiplier=1.5,
        default_multiplier=1.0,
        default_controls={"propose_budget_multiplier": 1.0},
        selected_controls={"propose_budget_multiplier": 1.5},
        applied_controls={"propose_budget_multiplier": 1.5},
        selected_controls_changed={"propose_budget": True},
        applied_controls_changed={"propose_budget": True},
        partial_override=False,
        partial_override_errors={},
        tier_index_before=0,
        tier_index_after=1,
        override_reason="",
    )
    state = controller.state
    controller.close()

    assert state.A[ACTION_INCREASE_BUDGET][0, 0] == pytest.approx(3.0)
    assert state.b[ACTION_INCREASE_BUDGET][0] == pytest.approx(2.0)


def test_budget_policy_percentile_profile_persists_history_and_caps(tmp_path) -> None:
    state_path = tmp_path / "budget_policy_state_percentile.json"
    config = BudgetPolicyConfig.normalize(
        mode="on",
        algorithm="linucb",
        alpha=0.0,
        epsilon=0.0,
        warmup_decisions=0,
        feature_scaling_profile="percentile_v1",
        state_path=state_path,
        events_path=tmp_path / "budget_policy_events_percentile.jsonl",
        random_seed=0,
        decay=1.0,
    )
    controller = AdaptiveBudgetPolicyController(config)
    for idx in range(DEFAULT_POLICY_FEATURE_SCALING_MIN_SAMPLES - 1):
        controller.decide(
            feature_values={"helper_calls_used": float(idx)},
            default_action=ACTION_KEEP_BUDGET,
        )
    decision_before_threshold = controller.decide(
        feature_values={"helper_calls_used": float(DEFAULT_POLICY_FEATURE_SCALING_MIN_SAMPLES - 1)},
        default_action=ACTION_KEEP_BUDGET,
    )
    decision_after_threshold = controller.decide(
        feature_values={"helper_calls_used": float(DEFAULT_POLICY_FEATURE_SCALING_MIN_SAMPLES)},
        default_action=ACTION_KEEP_BUDGET,
    )
    state = controller.state
    controller.close()

    assert decision_before_threshold.feature_scaling_fallback_to_v1 is True
    assert (
        decision_before_threshold.feature_scaling_history_sizes["helper_calls_used"]
        == DEFAULT_POLICY_FEATURE_SCALING_MIN_SAMPLES - 1
    )
    assert (
        decision_before_threshold.feature_scaling_caps["helper_calls_used"]
        == _V1_COUNT_CAPS["helper_calls_used"]
    )
    assert decision_after_threshold.feature_scaling_fallback_to_v1 is False
    assert (
        decision_after_threshold.feature_scaling_history_sizes["helper_calls_used"]
        == DEFAULT_POLICY_FEATURE_SCALING_MIN_SAMPLES
    )
    assert (
        decision_after_threshold.feature_scaling_caps["helper_calls_used"]
        > _V1_COUNT_CAPS["helper_calls_used"]
    )
    assert (
        len(state.feature_scaling_history["helper_calls_used"])
        == DEFAULT_POLICY_FEATURE_SCALING_MIN_SAMPLES + 1
    )

    reloaded = AdaptiveBudgetPolicyController(config)
    reloaded_state = reloaded.state
    reloaded.close()

    assert reloaded.load_error is None
    assert (
        len(reloaded_state.feature_scaling_history["helper_calls_used"])
        == DEFAULT_POLICY_FEATURE_SCALING_MIN_SAMPLES + 1
    )


def test_budget_policy_hierarchical_family_percentile_profile_uses_prior_history(tmp_path) -> None:
    config = BudgetPolicyConfig.normalize(
        mode="on",
        algorithm="linucb",
        architecture=BUDGET_POLICY_ARCH_HIERARCHICAL_V1,
        alpha=0.0,
        epsilon=0.0,
        warmup_decisions=0,
        feature_scaling_profile="percentile_v1",
        state_path=tmp_path / "budget_policy_hierarchical_percentile_state.json",
        events_path=tmp_path / "budget_policy_hierarchical_percentile_events.jsonl",
        random_seed=0,
        decay=1.0,
    )
    controller = AdaptiveBudgetPolicyController(config)
    for idx in range(DEFAULT_POLICY_FEATURE_SCALING_MIN_SAMPLES - 1):
        controller.decide(
            feature_values={"helper_calls_used": float(idx)},
            default_action=ACTION_INCREASE_BUDGET,
            available_families=(ACTION_BUDGET_FAMILY_TIME,),
        )
    decision_before_threshold = controller.decide(
        feature_values={"helper_calls_used": float(DEFAULT_POLICY_FEATURE_SCALING_MIN_SAMPLES - 1)},
        default_action=ACTION_INCREASE_BUDGET,
        available_families=(ACTION_BUDGET_FAMILY_TIME,),
    )
    decision_after_threshold = controller.decide(
        feature_values={"helper_calls_used": float(DEFAULT_POLICY_FEATURE_SCALING_MIN_SAMPLES)},
        default_action=ACTION_INCREASE_BUDGET,
        available_families=(ACTION_BUDGET_FAMILY_TIME,),
    )
    gate_state = controller.state
    family_state = controller.family_state
    controller.close()

    assert decision_before_threshold.selected_action == ACTION_INCREASE_BUDGET
    assert decision_before_threshold.selected_family == ACTION_BUDGET_FAMILY_TIME
    assert decision_before_threshold.feature_scaling_fallback_to_v1 is True
    assert decision_before_threshold.family_feature_scaling_fallback_to_v1 is True
    assert (
        decision_before_threshold.feature_scaling_history_sizes["helper_calls_used"]
        == DEFAULT_POLICY_FEATURE_SCALING_MIN_SAMPLES - 1
    )
    assert (
        decision_before_threshold.family_feature_scaling_history_sizes["helper_calls_used"]
        == DEFAULT_POLICY_FEATURE_SCALING_MIN_SAMPLES - 1
    )
    assert decision_after_threshold.feature_scaling_fallback_to_v1 is False
    assert decision_after_threshold.family_feature_scaling_fallback_to_v1 is False
    assert (
        decision_after_threshold.feature_scaling_history_sizes["helper_calls_used"]
        == DEFAULT_POLICY_FEATURE_SCALING_MIN_SAMPLES
    )
    assert (
        decision_after_threshold.family_feature_scaling_history_sizes["helper_calls_used"]
        == DEFAULT_POLICY_FEATURE_SCALING_MIN_SAMPLES
    )
    assert (
        decision_after_threshold.family_feature_scaling_caps["helper_calls_used"]
        > _V1_COUNT_CAPS["helper_calls_used"]
    )
    assert (
        len(gate_state.feature_scaling_history["helper_calls_used"])
        == DEFAULT_POLICY_FEATURE_SCALING_MIN_SAMPLES + 1
    )
    assert (
        len(family_state.feature_scaling_history["helper_calls_used"])
        == DEFAULT_POLICY_FEATURE_SCALING_MIN_SAMPLES + 1
    )


def test_budget_policy_hierarchical_updates_gate_and_family(tmp_path) -> None:
    config = BudgetPolicyConfig.normalize(
        mode="on",
        algorithm="linucb",
        architecture=BUDGET_POLICY_ARCH_HIERARCHICAL_V1,
        alpha=0.0,
        epsilon=0.0,
        warmup_decisions=0,
        save_every=1,
        state_path=tmp_path / "budget_policy_hierarchical_state.json",
        events_path=tmp_path / "budget_policy_hierarchical_events.jsonl",
        random_seed=0,
        decay=1.0,
    )
    controller = AdaptiveBudgetPolicyController(config)
    decision = controller.decide(
        feature_values={"bias": 1.0},
        default_action=ACTION_INCREASE_BUDGET,
        available_families=(ACTION_BUDGET_FAMILY_TIME,),
        unavailable_family_reasons={ACTION_BUDGET_FAMILY_CONTEXT: "at_cap"},
    )
    event = controller.record_outcome(
        decision=decision,
        run_id="run_budget_hierarchical",
        case_id="case_budget_hierarchical",
        condition="layered_adaptive_llm",
        attempt=1,
        note="tests_still_failing",
        reward_scope="budget_credit_v2",
        reward=0.4,
        reward_success_hint=0.0,
        reward_progress_hint=0.0,
        reward_progress_score=0.25,
        reward_token_penalty=0.1,
        reward_latency_penalty=0.1,
        reward_stagnation_penalty=0.0,
        reward_budget_exceeded_penalty=0.0,
        request_budget_exceeded=False,
        success=False,
        local_llm_call_count=1,
        local_tokens_prompt=20,
        local_tokens_completion=10,
        local_tokens_total=30,
        local_wall_sec=0.5,
        selected_multiplier=1.5,
        applied_multiplier=1.5,
        default_multiplier=1.0,
        default_controls={"propose_budget_multiplier": 1.0, "openai_max_tokens": 400},
        selected_controls={"propose_budget_multiplier": 1.5, "openai_max_tokens": 400},
        applied_controls={"propose_budget_multiplier": 1.5, "openai_max_tokens": 400},
        selected_controls_changed={"propose_budget": True, "max_tokens": False},
        applied_controls_changed={"propose_budget": True, "max_tokens": False},
        partial_override=False,
        partial_override_errors={},
        tier_index_before=0,
        tier_index_after=1,
        override_reason="",
    )
    gate_state = controller.state
    family_state = controller.family_state
    controller.close()

    assert decision.selected_action == ACTION_INCREASE_BUDGET
    assert decision.selected_family == ACTION_BUDGET_FAMILY_TIME
    assert event.architecture == BUDGET_POLICY_ARCH_HIERARCHICAL_V1
    assert event.gate_model_updated is True
    assert event.family_model_updated is True
    assert gate_state.updates == 1
    assert family_state.updates == 1
    assert family_state.action_counts[ACTION_BUDGET_FAMILY_TIME] == 1


def test_budget_policy_rejects_legacy_state_when_hierarchical_requested(tmp_path) -> None:
    state_path = tmp_path / "budget_policy_legacy_tier_state.json"
    state_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "mode": "on",
                "algorithm": "linucb",
                "feature_names": ["bias"],
                "decisions": 1,
                "updates": 1,
                "action_counts": {
                    ACTION_KEEP_BUDGET: 0,
                    ACTION_INCREASE_BUDGET: 1,
                },
                "A": {
                    ACTION_KEEP_BUDGET: [[1.0]],
                    ACTION_INCREASE_BUDGET: [[2.0]],
                },
                "b": {
                    ACTION_KEEP_BUDGET: [0.0],
                    ACTION_INCREASE_BUDGET: [1.0],
                },
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    config = BudgetPolicyConfig.normalize(
        mode="on",
        algorithm="linucb",
        architecture=BUDGET_POLICY_ARCH_HIERARCHICAL_V1,
        state_path=state_path,
        events_path=tmp_path / "budget_policy_legacy_tier_events.jsonl",
        decay=1.0,
    )
    controller = AdaptiveBudgetPolicyController(config)
    controller.close()

    assert controller.load_error is not None
    assert "incompatible" in str(controller.load_error)


def test_budget_policy_normalize_defaults_warmup_to_20() -> None:
    config = BudgetPolicyConfig.normalize(mode="on", algorithm="linucb")
    assert config.warmup_decisions == 20


def test_budget_policy_warmup_uses_uniform_gate_and_effective_epsilon(tmp_path) -> None:
    config = BudgetPolicyConfig.normalize(
        mode="on",
        algorithm="linucb",
        alpha=0.6,
        epsilon=0.03,
        decay=0.9,
        state_path=tmp_path / "budget_policy_state_warmup.json",
        events_path=tmp_path / "budget_policy_events_warmup.jsonl",
        random_seed=0,
    )
    controller = AdaptiveBudgetPolicyController(config)
    decision = controller.decide(
        feature_values={"bias": 1.0},
        default_action=ACTION_KEEP_BUDGET,
    )
    event = controller.record_outcome(
        decision=decision,
        run_id="run_budget_warmup",
        case_id="case_budget_warmup",
        condition="layered_adaptive_llm",
        attempt=1,
        note="tests_still_failing",
        reward_scope="budget_credit_v2",
        reward=0.1,
        reward_success_hint=0.0,
        reward_progress_hint=0.0,
        reward_progress_score=0.0,
        reward_token_penalty=0.0,
        reward_latency_penalty=0.0,
        reward_stagnation_penalty=0.0,
        reward_budget_exceeded_penalty=0.0,
        request_budget_exceeded=False,
        success=False,
        local_llm_call_count=0,
        local_tokens_prompt=0,
        local_tokens_completion=0,
        local_tokens_total=0,
        local_wall_sec=0.0,
        selected_multiplier=1.0,
        applied_multiplier=1.0,
        default_multiplier=1.0,
        default_controls={"propose_budget_multiplier": 1.0},
        selected_controls={"propose_budget_multiplier": 1.0},
        applied_controls={"propose_budget_multiplier": 1.0},
        selected_controls_changed={"propose_budget": False},
        applied_controls_changed={"propose_budget": False},
        partial_override=False,
        partial_override_errors={},
        tier_index_before=0,
        tier_index_after=0,
        override_reason="",
    )
    controller.close()

    assert decision.decision_strategy == "warmup_uniform"
    assert decision.propensity == pytest.approx(0.5)
    assert decision.configured_alpha == pytest.approx(0.6)
    assert decision.configured_epsilon == pytest.approx(0.03)
    assert decision.configured_decay == pytest.approx(0.9)
    assert decision.effective_epsilon == pytest.approx(1.0)
    assert decision.warmup_active is True
    assert event.configured_alpha == pytest.approx(0.6)
    assert event.configured_epsilon == pytest.approx(0.03)
    assert event.configured_decay == pytest.approx(0.9)
    assert event.effective_epsilon == pytest.approx(1.0)
    assert event.warmup_active is True


def test_budget_policy_linucb_scores_match_known_ucb_values(tmp_path) -> None:
    state_path = tmp_path / "budget_policy_state_known_scores.json"
    state_path.write_text(
        json.dumps(
            {
                "schema_version": STATE_SCHEMA_VERSION,
                "mode": "on",
                "algorithm": "linucb",
                "architecture": BUDGET_POLICY_ARCH_TIER_BINARY,
                "alpha": 0.5,
                "epsilon": 0.0,
                "l2": 1.0,
                "warmup_decisions": 0,
                "save_every": 1,
                "gate_state": _linucb_state_payload(
                    action_names=(ACTION_KEEP_BUDGET, ACTION_INCREASE_BUDGET),
                    feature_names=["bias"],
                    A={
                        ACTION_KEEP_BUDGET: [[1.0]],
                        ACTION_INCREASE_BUDGET: [[2.0]],
                    },
                    b={
                        ACTION_KEEP_BUDGET: [0.2],
                        ACTION_INCREASE_BUDGET: [1.5],
                    },
                ),
                "family_state": _linucb_state_payload(
                    action_names=(
                        ACTION_BUDGET_FAMILY_TIME,
                        ACTION_BUDGET_FAMILY_TOKENS,
                        ACTION_BUDGET_FAMILY_CONTEXT,
                    ),
                    feature_names=[],
                ),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    config = BudgetPolicyConfig.normalize(
        mode="on",
        algorithm="linucb",
        alpha=0.5,
        epsilon=0.0,
        warmup_decisions=0,
        state_path=state_path,
        events_path=tmp_path / "budget_policy_events_known_scores.jsonl",
        decay=1.0,
    )
    controller = AdaptiveBudgetPolicyController(config)
    x = controller._feature_vector(state=controller.state, cleaned_features={"bias": 1.0})
    scores = controller._linucb_scores(state=controller.state, x=x)
    controller.close()

    assert scores[ACTION_INCREASE_BUDGET] == pytest.approx(0.75 + (0.5 / math.sqrt(2.0)))
    assert scores[ACTION_KEEP_BUDGET] == pytest.approx(0.2 + 0.5)
    assert scores[ACTION_INCREASE_BUDGET] > scores[ACTION_KEEP_BUDGET]


def test_budget_policy_linucb_scores_regularize_singular_covariance(tmp_path) -> None:
    state_path = tmp_path / "budget_policy_state_singular.json"
    state_path.write_text(
        json.dumps(
            {
                "schema_version": STATE_SCHEMA_VERSION,
                "mode": "on",
                "algorithm": "linucb",
                "architecture": BUDGET_POLICY_ARCH_TIER_BINARY,
                "alpha": 0.3,
                "epsilon": 0.0,
                "l2": 1.0,
                "warmup_decisions": 0,
                "save_every": 1,
                "gate_state": _linucb_state_payload(
                    action_names=(ACTION_KEEP_BUDGET, ACTION_INCREASE_BUDGET),
                    feature_names=["bias"],
                    A={
                        ACTION_KEEP_BUDGET: [[0.0]],
                        ACTION_INCREASE_BUDGET: [[0.0]],
                    },
                    b={
                        ACTION_KEEP_BUDGET: [0.0],
                        ACTION_INCREASE_BUDGET: [1.0],
                    },
                ),
                "family_state": _linucb_state_payload(
                    action_names=(
                        ACTION_BUDGET_FAMILY_TIME,
                        ACTION_BUDGET_FAMILY_TOKENS,
                        ACTION_BUDGET_FAMILY_CONTEXT,
                    ),
                    feature_names=[],
                ),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    config = BudgetPolicyConfig.normalize(
        mode="on",
        algorithm="linucb",
        alpha=0.3,
        epsilon=0.0,
        l2=1.0,
        warmup_decisions=0,
        state_path=state_path,
        events_path=tmp_path / "budget_policy_events_singular.jsonl",
        decay=1.0,
    )
    controller = AdaptiveBudgetPolicyController(config)
    state = controller._state
    x = controller._feature_vector(state=state, cleaned_features={"bias": 1.0})
    scores = controller._linucb_scores(state=state, x=x)

    assert all(math.isfinite(score) for score in scores.values())
    assert state.A[ACTION_KEEP_BUDGET][0, 0] == pytest.approx(1.0)
    assert state.A[ACTION_INCREASE_BUDGET][0, 0] == pytest.approx(1.0)
    controller.close()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_OUTCOME_DEFAULTS: dict[str, object] = {
    "run_id": "run_cov",
    "case_id": "case_cov",
    "condition": "layered_adaptive_llm",
    "attempt": 1,
    "note": "coverage",
    "reward_scope": "budget_credit_v2",
    "reward": 0.5,
    "reward_success_hint": 0.0,
    "reward_progress_hint": 0.0,
    "reward_progress_score": 0.0,
    "reward_token_penalty": 0.0,
    "reward_latency_penalty": 0.0,
    "reward_stagnation_penalty": 0.0,
    "reward_budget_exceeded_penalty": 0.0,
    "request_budget_exceeded": False,
    "success": False,
    "local_llm_call_count": 1,
    "local_tokens_prompt": 10,
    "local_tokens_completion": 5,
    "local_tokens_total": 15,
    "local_wall_sec": 0.1,
    "selected_multiplier": 1.0,
    "applied_multiplier": 1.0,
    "default_multiplier": 1.0,
    "default_controls": {"propose_budget_multiplier": 1.0},
    "selected_controls": {"propose_budget_multiplier": 1.0},
    "applied_controls": {"propose_budget_multiplier": 1.0},
    "selected_controls_changed": {"propose_budget": False},
    "applied_controls_changed": {"propose_budget": False},
    "partial_override": False,
    "partial_override_errors": {},
    "tier_index_before": 0,
    "tier_index_after": 0,
    "override_reason": "",
}


def _record(ctrl, decision, **overrides):
    """Record an outcome with sensible defaults; *overrides* patch kwargs."""
    kw = {**_OUTCOME_DEFAULTS, "decision": decision, **overrides}
    return ctrl.record_outcome(**kw)


def test_budget_policy_init_state_path_loads_from_separate_source(tmp_path) -> None:
    init_state_path = tmp_path / "init_budget_state.json"
    state_path = tmp_path / "new_budget_state.json"
    init_state_path.write_text(
        json.dumps(
            {
                "schema_version": STATE_SCHEMA_VERSION,
                "mode": "on",
                "algorithm": "linucb",
                "architecture": BUDGET_POLICY_ARCH_TIER_BINARY,
                "feature_scaling_profile": "v1",
                "alpha": 0.0,
                "epsilon": 0.0,
                "l2": 1.0,
                "warmup_decisions": 0,
                "save_every": 1,
                "decay": 1.0,
                "gate_state": _linucb_state_payload(
                    action_names=(ACTION_KEEP_BUDGET, ACTION_INCREASE_BUDGET),
                    feature_names=["bias"],
                    decisions=4,
                    updates=2,
                ),
                "family_state": _linucb_state_payload(
                    action_names=(
                        ACTION_BUDGET_FAMILY_TIME,
                        ACTION_BUDGET_FAMILY_TOKENS,
                        ACTION_BUDGET_FAMILY_CONTEXT,
                    ),
                    feature_names=[],
                ),
            }
        ),
        encoding="utf-8",
    )
    cfg = BudgetPolicyConfig.normalize(
        mode="on",
        init_state_path=init_state_path,
        state_path=state_path,
        events_path=tmp_path / "budget_init_events.jsonl",
        decay=1.0,
    )

    ctrl = AdaptiveBudgetPolicyController(cfg)
    assert ctrl.load_error is None
    assert ctrl.state.decisions == 4

    ctrl.close()
    assert state_path.exists()
    saved = json.loads(state_path.read_text(encoding="utf-8"))
    assert saved["gate_state"]["decisions"] == 4


def test_budget_policy_bwk_tracks_remaining_budget_and_persists_state(tmp_path) -> None:
    state_path = tmp_path / "budget_bwk_state.json"
    events_path = tmp_path / "budget_bwk_events.jsonl"
    cfg = BudgetPolicyConfig.normalize(
        mode="on",
        architecture=BUDGET_POLICY_ARCH_BWK_V1,
        warmup_decisions=0,
        save_every=1,
        bwk_run_token_budget=40,
        state_path=state_path,
        events_path=events_path,
        random_seed=0,
        decay=1.0,
    )
    ctrl = AdaptiveBudgetPolicyController(cfg)
    decision = ctrl.decide(feature_values={"bias": 1.0}, default_action=ACTION_KEEP_BUDGET)
    event = _record(ctrl, decision, local_tokens_total=15)
    ctrl.close()

    assert decision.bwk_run_token_budget == 40
    assert decision.bwk_remaining_token_budget == 40
    assert event.bwk_remaining_token_budget_before == 40
    assert event.bwk_remaining_token_budget_after == 25
    assert event.bwk_tokens_charged == 15
    assert event.bwk_cumulative_tokens_charged == 15
    assert event.bwk_budget_exhausted is False

    reloaded = AdaptiveBudgetPolicyController(cfg)
    assert reloaded.load_error is None
    assert reloaded.bwk_status["remaining_token_budget"] == 25
    assert reloaded.bwk_status["cumulative_tokens_charged"] == 15
    reloaded.close()


def test_budget_policy_bwk_init_state_path_resets_run_budget_counters(tmp_path) -> None:
    init_state_path = tmp_path / "init_budget_bwk_state.json"
    state_path = tmp_path / "new_budget_bwk_state.json"
    init_state_path.write_text(
        json.dumps(
            {
                "schema_version": STATE_SCHEMA_VERSION,
                "mode": "on",
                "algorithm": "linucb",
                "architecture": BUDGET_POLICY_ARCH_BWK_V1,
                "feature_scaling_profile": "v1",
                "alpha": 0.0,
                "epsilon": 0.0,
                "l2": 1.0,
                "warmup_decisions": 0,
                "save_every": 1,
                "decay": 1.0,
                "gate_state": _linucb_state_payload(
                    action_names=(ACTION_KEEP_BUDGET, ACTION_INCREASE_BUDGET),
                    feature_names=["bias"],
                    decisions=4,
                    updates=2,
                ),
                "family_state": _linucb_state_payload(
                    action_names=(
                        ACTION_BUDGET_FAMILY_TIME,
                        ACTION_BUDGET_FAMILY_TOKENS,
                        ACTION_BUDGET_FAMILY_CONTEXT,
                    ),
                    feature_names=[],
                ),
                "bwk_run_token_budget": 100,
                "bwk_remaining_token_budget": 0,
                "bwk_cumulative_tokens_charged": 100,
                "bwk_dual_variable": 1.0,
            }
        ),
        encoding="utf-8",
    )
    cfg = BudgetPolicyConfig.normalize(
        mode="on",
        architecture=BUDGET_POLICY_ARCH_BWK_V1,
        warmup_decisions=0,
        init_state_path=init_state_path,
        state_path=state_path,
        events_path=tmp_path / "budget_bwk_init_events.jsonl",
        bwk_run_token_budget=40,
        random_seed=0,
        decay=1.0,
    )

    ctrl = AdaptiveBudgetPolicyController(cfg)

    assert ctrl.load_error is None
    assert ctrl.state.decisions == 4

    decision = ctrl.decide(feature_values={"bias": 1.0}, default_action=ACTION_KEEP_BUDGET)
    assert ctrl.bwk_status["run_token_budget"] == 40
    assert ctrl.bwk_status["remaining_token_budget"] == 40
    assert ctrl.bwk_status["cumulative_tokens_charged"] == 0
    assert ctrl.bwk_status["dual_variable"] == pytest.approx(0.0)
    assert decision.bwk_budget_exhausted is False
    assert decision.bwk_remaining_token_budget == 40

    ctrl.close()
    saved = json.loads(state_path.read_text(encoding="utf-8"))
    assert saved["bwk_run_token_budget"] == 40
    assert saved["bwk_remaining_token_budget"] == 40
    assert saved["bwk_cumulative_tokens_charged"] == 0
    assert saved["bwk_dual_variable"] == pytest.approx(0.0)


def test_budget_policy_bwk_state_path_resume_preserves_run_budget_counters(tmp_path) -> None:
    state_path = tmp_path / "budget_bwk_resume_state.json"
    events_path = tmp_path / "budget_bwk_resume_events.jsonl"
    cfg = BudgetPolicyConfig.normalize(
        mode="on",
        architecture=BUDGET_POLICY_ARCH_BWK_V1,
        warmup_decisions=0,
        save_every=1,
        bwk_run_token_budget=40,
        state_path=state_path,
        events_path=events_path,
        random_seed=0,
        decay=1.0,
    )
    ctrl = AdaptiveBudgetPolicyController(cfg)
    decision = ctrl.decide(feature_values={"bias": 1.0}, default_action=ACTION_KEEP_BUDGET)
    event = _record(ctrl, decision, local_tokens_total=50)
    ctrl.close()

    resumed = AdaptiveBudgetPolicyController(cfg)
    resumed_decision = resumed.decide(
        feature_values={"bias": 1.0},
        default_action=ACTION_KEEP_BUDGET,
    )

    assert resumed.load_error is None
    assert event.bwk_remaining_token_budget_after == 0
    assert event.bwk_cumulative_tokens_charged == 50
    assert event.bwk_dual_variable_after == pytest.approx(0.25)
    assert resumed.bwk_status["run_token_budget"] == 40
    assert resumed.bwk_status["remaining_token_budget"] == 0
    assert resumed.bwk_status["cumulative_tokens_charged"] == 50
    assert resumed.bwk_status["dual_variable"] == pytest.approx(0.25)
    assert resumed_decision.bwk_budget_exhausted is True
    resumed.close()


def test_budget_policy_bwk_clamps_increase_when_budget_exhausted(tmp_path) -> None:
    cfg = BudgetPolicyConfig.normalize(
        mode="on",
        architecture=BUDGET_POLICY_ARCH_BWK_V1,
        alpha=0.0,
        epsilon=0.0,
        warmup_decisions=0,
        bwk_run_token_budget=10,
        state_path=tmp_path / "budget_bwk_exhausted_state.json",
        events_path=tmp_path / "budget_bwk_exhausted_events.jsonl",
        random_seed=0,
        decay=1.0,
    )
    ctrl = AdaptiveBudgetPolicyController(cfg)
    first_decision = ctrl.decide(feature_values={"bias": 1.0}, default_action=ACTION_KEEP_BUDGET)
    _record(ctrl, first_decision, local_tokens_total=12)

    second_decision = ctrl.decide(
        feature_values={"bias": 1.0},
        default_action=ACTION_KEEP_BUDGET,
    )
    ctrl.close()

    assert second_decision.bwk_budget_exhausted is True
    assert second_decision.selected_action == ACTION_KEEP_BUDGET
    assert second_decision.applied_action == ACTION_KEEP_BUDGET


# ---------------------------------------------------------------------------
# BudgetPolicyConfig.normalize — invalid mode / algorithm / architecture
# (lines 112, 115, 118)
# ---------------------------------------------------------------------------


def test_normalize_invalid_mode_falls_back_to_off() -> None:
    cfg = BudgetPolicyConfig.normalize(mode="bogus")
    assert cfg.mode == "off"


def test_normalize_invalid_algorithm_falls_back_to_linucb() -> None:
    cfg = BudgetPolicyConfig.normalize(algorithm="random_forest")
    assert cfg.algorithm == "linucb"


def test_normalize_invalid_architecture_falls_back_to_default() -> None:
    cfg = BudgetPolicyConfig.normalize(architecture="transformer_v99")
    assert cfg.architecture == BUDGET_POLICY_ARCH_TIER_BINARY


# ---------------------------------------------------------------------------
# BudgetPolicyConfig.normalize — invalid feature_scaling_profile (line 127)
# ---------------------------------------------------------------------------


def test_normalize_invalid_feature_scaling_profile_falls_back() -> None:
    cfg = BudgetPolicyConfig.normalize(feature_scaling_profile="nonexistent")
    assert cfg.feature_scaling_profile == "v1"


# ---------------------------------------------------------------------------
# BudgetPolicyConfig.normalize — non-numeric random_seed (lines 132-133)
# ---------------------------------------------------------------------------


def test_normalize_non_numeric_random_seed_becomes_none() -> None:
    cfg = BudgetPolicyConfig.normalize(random_seed="not_a_number")
    assert cfg.random_seed is None


# ---------------------------------------------------------------------------
# LinUCBState.__post_init__ — empty action_names (line 380)
# ---------------------------------------------------------------------------


def test_linucb_state_empty_action_names_normalised_to_default() -> None:
    from evals.adaptive_budget_online_policy import LinUCBState

    state = LinUCBState(action_names=())
    assert state.action_names == ("default",)
    assert "default" in state.action_counts


# ---------------------------------------------------------------------------
# AdaptiveBudgetPolicyController.architecture property (line 421)
# ---------------------------------------------------------------------------


def test_controller_architecture_property(tmp_path) -> None:
    cfg = BudgetPolicyConfig.normalize(
        mode="on",
        architecture=BUDGET_POLICY_ARCH_HIERARCHICAL_V1,
        state_path=tmp_path / "s.json",
        events_path=tmp_path / "e.jsonl",
        decay=1.0,
    )
    ctrl = AdaptiveBudgetPolicyController(cfg)
    assert ctrl.architecture == BUDGET_POLICY_ARCH_HIERARCHICAL_V1
    ctrl.close()


# ---------------------------------------------------------------------------
# _record_feature_scaling_history — history exceeds limit (line 476)
# ---------------------------------------------------------------------------


def test_feature_scaling_history_truncated_at_limit(tmp_path) -> None:
    from evals.policy_feature_scaling import (
        DEFAULT_POLICY_FEATURE_SCALING_HISTORY_LIMIT,
    )

    cfg = BudgetPolicyConfig.normalize(
        mode="on",
        feature_scaling_profile="percentile_v1",
        warmup_decisions=0,
        state_path=tmp_path / "s.json",
        events_path=tmp_path / "e.jsonl",
        random_seed=0,
        decay=1.0,
    )
    ctrl = AdaptiveBudgetPolicyController(cfg)
    for i in range(DEFAULT_POLICY_FEATURE_SCALING_HISTORY_LIMIT + 10):
        ctrl.decide(
            feature_values={"x": float(i)},
            default_action=ACTION_KEEP_BUDGET,
        )
    state = ctrl.state
    ctrl.close()
    assert len(state.feature_scaling_history["x"]) <= DEFAULT_POLICY_FEATURE_SCALING_HISTORY_LIMIT


# ---------------------------------------------------------------------------
# record_outcome — mode="off" → "off_no_update" (lines 793-794)
# ---------------------------------------------------------------------------


def test_record_outcome_off_mode_returns_off_no_update(tmp_path) -> None:
    cfg = BudgetPolicyConfig.normalize(
        mode="off",
        state_path=tmp_path / "s.json",
        events_path=tmp_path / "e.jsonl",
        decay=1.0,
    )
    ctrl = AdaptiveBudgetPolicyController(cfg)
    decision = ctrl.decide(
        feature_values={"bias": 1.0},
        default_action=ACTION_KEEP_BUDGET,
    )
    event = _record(ctrl, decision)
    ctrl.close()
    assert event.update_reason == "off_no_update"
    assert event.model_updated is False


# ---------------------------------------------------------------------------
# record_outcome — gate updated but incomplete family →
#   "gate_updated_family_skipped" (line 786) and
#   "update_skipped_incomplete_state" (line 790)
# ---------------------------------------------------------------------------


def test_record_outcome_gate_updated_family_skipped(tmp_path) -> None:
    """Hierarchical: gate updates but family has no A/b → family returns False."""
    state_path = tmp_path / "s.json"
    state_path.write_text(
        json.dumps(
            {
                "schema_version": STATE_SCHEMA_VERSION,
                "algorithm": "linucb",
                "architecture": BUDGET_POLICY_ARCH_HIERARCHICAL_V1,
                "feature_scaling_profile": "v1",
                "gate_state": _linucb_state_payload(
                    action_names=(ACTION_KEEP_BUDGET, ACTION_INCREASE_BUDGET),
                    feature_names=["bias"],
                    l2=1.0,
                ),
                "family_state": _linucb_state_payload(
                    action_names=(
                        ACTION_BUDGET_FAMILY_TIME,
                        ACTION_BUDGET_FAMILY_TOKENS,
                        ACTION_BUDGET_FAMILY_CONTEXT,
                    ),
                    feature_names=[],
                ),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    cfg = BudgetPolicyConfig.normalize(
        mode="on",
        architecture=BUDGET_POLICY_ARCH_HIERARCHICAL_V1,
        epsilon=0.0,
        warmup_decisions=0,
        state_path=state_path,
        events_path=tmp_path / "e.jsonl",
        random_seed=42,
        decay=1.0,
    )
    ctrl = AdaptiveBudgetPolicyController(cfg)
    # Force the gate to select increase_budget and assign a family
    decision = ctrl.decide(
        feature_values={"bias": 1.0},
        default_action=ACTION_INCREASE_BUDGET,
        available_families=(ACTION_BUDGET_FAMILY_TIME,),
    )
    assert decision.applied_action == ACTION_INCREASE_BUDGET
    assert decision.applied_family == ACTION_BUDGET_FAMILY_TIME

    # Sabotage family state so _update_model returns False
    ctrl._family_state.A = {}
    ctrl._family_state.b = {}

    event = _record(ctrl, decision)
    ctrl.close()
    assert event.gate_model_updated is True
    assert event.family_model_updated is False
    assert event.update_reason == "gate_updated_family_skipped"


def test_record_outcome_update_skipped_incomplete_state(tmp_path) -> None:
    """Gate _update_model returns False → update_skipped_incomplete_state."""
    cfg = BudgetPolicyConfig.normalize(
        mode="on",
        epsilon=0.0,
        warmup_decisions=0,
        state_path=tmp_path / "s.json",
        events_path=tmp_path / "e.jsonl",
        random_seed=0,
        decay=1.0,
    )
    ctrl = AdaptiveBudgetPolicyController(cfg)
    decision = ctrl.decide(
        feature_values={"bias": 1.0},
        default_action=ACTION_KEEP_BUDGET,
    )
    # Sabotage gate state so _update_model fails
    ctrl._state.A = {}
    ctrl._state.b = {}

    event = _record(ctrl, decision)
    ctrl.close()
    assert event.gate_model_updated is False
    assert event.update_reason == "update_skipped_incomplete_state"


# ---------------------------------------------------------------------------
# _ensure_feature_state — algorithm != "linucb" → return (line 927)
# ---------------------------------------------------------------------------


def test_ensure_feature_state_skips_non_linucb() -> None:
    from evals.adaptive_budget_online_policy import LinUCBState

    cfg = BudgetPolicyConfig.normalize(algorithm="linucb")
    ctrl = AdaptiveBudgetPolicyController(cfg)
    # Patch algorithm to something else after construction
    object.__setattr__(ctrl.config, "algorithm", "random")
    state = LinUCBState(action_names=(ACTION_KEEP_BUDGET,))
    ctrl._ensure_feature_state(state=state, cleaned_features={"f": 1.0})
    # State should remain untouched — no feature_names set
    assert state.feature_names == []


# ---------------------------------------------------------------------------
# _ensure_feature_state — empty features → ["bias"] (lines 932-933)
# ---------------------------------------------------------------------------


def test_ensure_feature_state_empty_features_default_bias(tmp_path) -> None:
    from evals.adaptive_budget_online_policy import LinUCBState

    cfg = BudgetPolicyConfig.normalize(
        mode="on",
        state_path=tmp_path / "s.json",
        events_path=tmp_path / "e.jsonl",
        decay=1.0,
    )
    ctrl = AdaptiveBudgetPolicyController(cfg)
    state = LinUCBState(action_names=(ACTION_KEEP_BUDGET,))
    ctrl._ensure_feature_state(state=state, cleaned_features={})
    assert state.feature_names == ["bias"]
    assert state.A[ACTION_KEEP_BUDGET].shape == (1, 1)
    assert state.b[ACTION_KEEP_BUDGET].shape == (1,)
    ctrl.close()


# ---------------------------------------------------------------------------
# _ensure_feature_state — matrix shape mismatch → reinitialize (lines 977,979)
# ---------------------------------------------------------------------------


def test_ensure_feature_state_reinitialises_on_shape_mismatch(tmp_path) -> None:
    import numpy as np

    from evals.adaptive_budget_online_policy import LinUCBState

    cfg = BudgetPolicyConfig.normalize(
        mode="on",
        l2=2.0,
        state_path=tmp_path / "s.json",
        events_path=tmp_path / "e.jsonl",
        decay=1.0,
    )
    ctrl = AdaptiveBudgetPolicyController(cfg)
    state = LinUCBState(
        action_names=(ACTION_KEEP_BUDGET,),
        feature_names=["bias"],
        A={ACTION_KEEP_BUDGET: np.eye(3, dtype=float)},  # wrong shape
        b={ACTION_KEEP_BUDGET: np.zeros(3, dtype=float)},  # wrong shape
    )
    ctrl._ensure_feature_state(
        state=state,
        cleaned_features={"bias": 1.0},
    )
    assert state.A[ACTION_KEEP_BUDGET].shape == (1, 1)
    assert state.A[ACTION_KEEP_BUDGET][0, 0] == pytest.approx(2.0)
    assert state.b[ACTION_KEEP_BUDGET].shape == (1,)
    ctrl.close()


# ---------------------------------------------------------------------------
# _linucb_scores — algorithm != "linucb" → zeros (line 1006)
# ---------------------------------------------------------------------------


def test_linucb_scores_non_linucb_returns_zeros() -> None:
    import numpy as np

    cfg = BudgetPolicyConfig.normalize(algorithm="linucb")
    ctrl = AdaptiveBudgetPolicyController(cfg)
    object.__setattr__(ctrl.config, "algorithm", "random")
    state = ctrl._state
    x = np.array([1.0])
    scores = ctrl._linucb_scores(state=state, x=x)
    assert all(v == 0.0 for v in scores.values())


# ---------------------------------------------------------------------------
# _linucb_scores — missing A/b for action → 0.0 (lines 1011-1012)
# ---------------------------------------------------------------------------


def test_linucb_scores_missing_matrices_returns_zero(tmp_path) -> None:
    import numpy as np

    cfg = BudgetPolicyConfig.normalize(
        mode="on",
        state_path=tmp_path / "s.json",
        events_path=tmp_path / "e.jsonl",
        decay=1.0,
    )
    ctrl = AdaptiveBudgetPolicyController(cfg)
    # Initialise feature state so feature_names are set
    ctrl._ensure_feature_state(
        state=ctrl._state,
        cleaned_features={"bias": 1.0},
    )
    # Remove matrices for one action
    ctrl._state.A.pop(ACTION_KEEP_BUDGET, None)
    ctrl._state.b.pop(ACTION_KEEP_BUDGET, None)
    x = np.array([1.0])
    scores = ctrl._linucb_scores(state=ctrl._state, x=x)
    assert scores[ACTION_KEEP_BUDGET] == 0.0
    ctrl.close()


# ---------------------------------------------------------------------------
# _update_model — not linucb (line 1034)
# ---------------------------------------------------------------------------


def test_update_model_not_linucb_returns_false() -> None:
    cfg = BudgetPolicyConfig.normalize(algorithm="linucb", decay=1.0)
    ctrl = AdaptiveBudgetPolicyController(cfg)
    object.__setattr__(ctrl.config, "algorithm", "random")
    result = ctrl._update_model(
        state=ctrl._state,
        action=ACTION_KEEP_BUDGET,
        reward=1.0,
        feature_values={"bias": 1.0},
    )
    assert result is False


# ---------------------------------------------------------------------------
# _update_model — invalid action / empty features (line 1036)
# ---------------------------------------------------------------------------


def test_update_model_invalid_action_returns_false(tmp_path) -> None:
    cfg = BudgetPolicyConfig.normalize(
        mode="on",
        state_path=tmp_path / "s.json",
        events_path=tmp_path / "e.jsonl",
        decay=1.0,
    )
    ctrl = AdaptiveBudgetPolicyController(cfg)
    result = ctrl._update_model(
        state=ctrl._state,
        action="nonexistent_action",
        reward=1.0,
        feature_values={"bias": 1.0},
    )
    assert result is False
    ctrl.close()


def test_update_model_empty_feature_names_returns_false(tmp_path) -> None:
    cfg = BudgetPolicyConfig.normalize(
        mode="on",
        state_path=tmp_path / "s.json",
        events_path=tmp_path / "e.jsonl",
        decay=1.0,
    )
    ctrl = AdaptiveBudgetPolicyController(cfg)
    # feature_names is empty by default
    assert ctrl._state.feature_names == []
    result = ctrl._update_model(
        state=ctrl._state,
        action=ACTION_KEEP_BUDGET,
        reward=1.0,
        feature_values={"bias": 1.0},
    )
    assert result is False
    ctrl.close()


# ---------------------------------------------------------------------------
# _update_model — missing A/b for the action (line 1042)
# ---------------------------------------------------------------------------


def test_update_model_missing_matrices_returns_false(tmp_path) -> None:
    cfg = BudgetPolicyConfig.normalize(
        mode="on",
        state_path=tmp_path / "s.json",
        events_path=tmp_path / "e.jsonl",
        decay=1.0,
    )
    ctrl = AdaptiveBudgetPolicyController(cfg)
    ctrl._ensure_feature_state(
        state=ctrl._state,
        cleaned_features={"bias": 1.0},
    )
    ctrl._state.A.pop(ACTION_KEEP_BUDGET, None)
    ctrl._state.b.pop(ACTION_KEEP_BUDGET, None)
    result = ctrl._update_model(
        state=ctrl._state,
        action=ACTION_KEEP_BUDGET,
        reward=1.0,
        feature_values={"bias": 1.0},
    )
    assert result is False
    ctrl.close()


# ---------------------------------------------------------------------------
# _apply_recency_decay — A/b None or wrong shape → continue (line 1061)
# ---------------------------------------------------------------------------


def test_apply_recency_decay_skips_bad_shapes(tmp_path) -> None:
    import numpy as np

    from evals.adaptive_budget_online_policy import LinUCBState

    cfg = BudgetPolicyConfig.normalize(
        mode="on",
        decay=0.9,
        state_path=tmp_path / "s.json",
        events_path=tmp_path / "e.jsonl",
    )
    ctrl = AdaptiveBudgetPolicyController(cfg)
    state = LinUCBState(
        action_names=(ACTION_KEEP_BUDGET, ACTION_INCREASE_BUDGET),
        feature_names=["bias"],
        A={
            ACTION_KEEP_BUDGET: None,  # type: ignore[dict-item]
            ACTION_INCREASE_BUDGET: np.eye(5, dtype=float),  # wrong shape
        },
        b={
            ACTION_KEEP_BUDGET: None,  # type: ignore[dict-item]
            ACTION_INCREASE_BUDGET: np.zeros(5, dtype=float),  # wrong shape
        },
    )
    # Should not raise — both actions are skipped
    ctrl._apply_recency_decay(state=state, action=ACTION_KEEP_BUDGET)
    ctrl._apply_recency_decay(state=state, action=ACTION_INCREASE_BUDGET)
    # Matrices remain unchanged
    assert state.A[ACTION_KEEP_BUDGET] is None
    assert state.A[ACTION_INCREASE_BUDGET].shape == (5, 5)
    ctrl.close()


# ---------------------------------------------------------------------------
# _save_state — state_path=None → return (line 1192)
# ---------------------------------------------------------------------------


def test_save_state_noop_when_no_state_path() -> None:
    cfg = BudgetPolicyConfig.normalize(mode="on", state_path=None)
    ctrl = AdaptiveBudgetPolicyController(cfg)
    # Should not raise
    ctrl._save_state()
    ctrl.close()

    # Verify no state file was created (state_path is None)
    assert cfg.state_path is None


# ---------------------------------------------------------------------------
# _load_state_if_present — non-dict JSON (line 1207)
# ---------------------------------------------------------------------------


def test_load_state_non_dict_json_sets_load_error(tmp_path) -> None:
    state_path = tmp_path / "s.json"
    state_path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    cfg = BudgetPolicyConfig.normalize(
        mode="on",
        state_path=state_path,
        events_path=tmp_path / "e.jsonl",
        decay=1.0,
    )
    ctrl = AdaptiveBudgetPolicyController(cfg)
    ctrl.close()
    assert ctrl.load_error is not None
    assert "not an object" in str(ctrl.load_error)


# ---------------------------------------------------------------------------
# _load_state_if_present — algorithm mismatch (line 1211)
# ---------------------------------------------------------------------------


def test_load_state_algorithm_mismatch_sets_load_error(tmp_path) -> None:
    state_path = tmp_path / "s.json"
    state_path.write_text(
        json.dumps(
            {
                "schema_version": STATE_SCHEMA_VERSION,
                "algorithm": "some_other_algo",
            }
        ),
        encoding="utf-8",
    )
    cfg = BudgetPolicyConfig.normalize(
        mode="on",
        state_path=state_path,
        events_path=tmp_path / "e.jsonl",
        decay=1.0,
    )
    ctrl = AdaptiveBudgetPolicyController(cfg)
    ctrl.close()
    assert ctrl.load_error is not None
    assert "algorithm mismatch" in str(ctrl.load_error)


# ---------------------------------------------------------------------------
# _load_state_if_present — feature_scaling_profile mismatch (line 1219)
# ---------------------------------------------------------------------------


def test_load_state_feature_scaling_profile_mismatch(tmp_path) -> None:
    state_path = tmp_path / "s.json"
    state_path.write_text(
        json.dumps(
            {
                "schema_version": STATE_SCHEMA_VERSION,
                "algorithm": "linucb",
                "feature_scaling_profile": "percentile_v1",
            }
        ),
        encoding="utf-8",
    )
    cfg = BudgetPolicyConfig.normalize(
        mode="on",
        feature_scaling_profile="v1",
        state_path=state_path,
        events_path=tmp_path / "e.jsonl",
        decay=1.0,
    )
    ctrl = AdaptiveBudgetPolicyController(cfg)
    ctrl.close()
    assert ctrl.load_error is not None
    assert "feature_scaling_profile mismatch" in str(ctrl.load_error)


# ---------------------------------------------------------------------------
# _load_state_if_present — legacy schema_version=2 loading (lines 1223-1229)
# ---------------------------------------------------------------------------


def test_load_state_legacy_schema_v2_tier_binary(tmp_path) -> None:
    state_path = tmp_path / "s.json"
    state_path.write_text(
        json.dumps(
            {
                "schema_version": 2,
                "algorithm": "linucb",
                "feature_scaling_profile": "v1",
                "feature_names": ["bias"],
                "decisions": 5,
                "updates": 3,
                "action_counts": {
                    ACTION_KEEP_BUDGET: 2,
                    ACTION_INCREASE_BUDGET: 3,
                },
                "A": {
                    ACTION_KEEP_BUDGET: [[2.0]],
                    ACTION_INCREASE_BUDGET: [[3.0]],
                },
                "b": {
                    ACTION_KEEP_BUDGET: [0.5],
                    ACTION_INCREASE_BUDGET: [1.5],
                },
            }
        ),
        encoding="utf-8",
    )
    cfg = BudgetPolicyConfig.normalize(
        mode="on",
        architecture=BUDGET_POLICY_ARCH_TIER_BINARY,
        state_path=state_path,
        events_path=tmp_path / "e.jsonl",
        decay=1.0,
    )
    ctrl = AdaptiveBudgetPolicyController(cfg)
    ctrl.close()
    assert ctrl.load_error is None
    gate = ctrl.state
    assert gate.decisions == 5
    assert gate.updates == 3
    assert gate.feature_names == ["bias"]
    # Family state should be freshly initialised (empty)
    family = ctrl.family_state
    assert family.feature_names == []
    assert family.decisions == 0


# ---------------------------------------------------------------------------
# _load_state_if_present — architecture mismatch (line 1234)
# ---------------------------------------------------------------------------


def test_load_state_architecture_mismatch(tmp_path) -> None:
    state_path = tmp_path / "s.json"
    state_path.write_text(
        json.dumps(
            {
                "schema_version": STATE_SCHEMA_VERSION,
                "algorithm": "linucb",
                "feature_scaling_profile": "v1",
                "architecture": BUDGET_POLICY_ARCH_TIER_BINARY,
                "gate_state": _linucb_state_payload(
                    action_names=(ACTION_KEEP_BUDGET, ACTION_INCREASE_BUDGET),
                    feature_names=["bias"],
                ),
                "family_state": _linucb_state_payload(
                    action_names=(
                        ACTION_BUDGET_FAMILY_TIME,
                        ACTION_BUDGET_FAMILY_TOKENS,
                        ACTION_BUDGET_FAMILY_CONTEXT,
                    ),
                    feature_names=[],
                ),
            }
        ),
        encoding="utf-8",
    )
    cfg = BudgetPolicyConfig.normalize(
        mode="on",
        architecture=BUDGET_POLICY_ARCH_HIERARCHICAL_V1,
        state_path=state_path,
        events_path=tmp_path / "e.jsonl",
        decay=1.0,
    )
    ctrl = AdaptiveBudgetPolicyController(cfg)
    ctrl.close()
    assert ctrl.load_error is not None
    assert "architecture mismatch" in str(ctrl.load_error)


# ---------------------------------------------------------------------------
# _load_state_if_present — family_raw not a dict → empty family (line 1240)
# ---------------------------------------------------------------------------


def test_load_state_family_raw_not_dict_initialises_empty(tmp_path) -> None:
    state_path = tmp_path / "s.json"
    state_path.write_text(
        json.dumps(
            {
                "schema_version": STATE_SCHEMA_VERSION,
                "algorithm": "linucb",
                "feature_scaling_profile": "v1",
                "architecture": BUDGET_POLICY_ARCH_TIER_BINARY,
                "gate_state": _linucb_state_payload(
                    action_names=(ACTION_KEEP_BUDGET, ACTION_INCREASE_BUDGET),
                    feature_names=["bias"],
                ),
                "family_state": "not_a_dict",
            }
        ),
        encoding="utf-8",
    )
    cfg = BudgetPolicyConfig.normalize(
        mode="on",
        architecture=BUDGET_POLICY_ARCH_TIER_BINARY,
        state_path=state_path,
        events_path=tmp_path / "e.jsonl",
        decay=1.0,
    )
    ctrl = AdaptiveBudgetPolicyController(cfg)
    ctrl.close()
    assert ctrl.load_error is None
    gate = ctrl.state
    assert gate.feature_names == ["bias"]
    family = ctrl.family_state
    assert family.feature_names == []
    assert family.decisions == 0


# ---------------------------------------------------------------------------
# _append_event — events_path=None → return (line 1259)
# ---------------------------------------------------------------------------


def test_append_event_noop_when_no_events_path(tmp_path) -> None:
    cfg = BudgetPolicyConfig.normalize(
        mode="on",
        state_path=tmp_path / "s.json",
        events_path=None,
        decay=1.0,
    )
    ctrl = AdaptiveBudgetPolicyController(cfg)
    decision = ctrl.decide(
        feature_values={"bias": 1.0},
        default_action=ACTION_KEEP_BUDGET,
    )
    # Should not raise even though events_path is None
    event = _record(ctrl, decision)
    assert event.update_reason == "on_policy_update"
    ctrl.close()


# ---------------------------------------------------------------------------
# Thompson Sampling — budget policy
# ---------------------------------------------------------------------------


def test_budget_thompson_algorithm_accepted() -> None:
    cfg = BudgetPolicyConfig.normalize(algorithm="thompson")
    assert cfg.algorithm == "thompson"


def test_budget_thompson_scores_returns_all_actions(tmp_path) -> None:
    cfg = BudgetPolicyConfig.normalize(
        mode="on",
        algorithm="thompson",
        warmup_decisions=0,
        epsilon=0.0,
        state_path=tmp_path / "bts_state.json",
        events_path=tmp_path / "bts_events.jsonl",
        random_seed=42,
        decay=1.0,
    )
    ctrl = AdaptiveBudgetPolicyController(cfg)
    decision = ctrl.decide(
        feature_values={"bias": 1.0},
        default_action=ACTION_KEEP_BUDGET,
    )
    assert decision.algorithm == "thompson"
    for score in decision.scores.values():
        assert math.isfinite(score)
    ctrl.close()


def test_budget_thompson_update_model_succeeds(tmp_path) -> None:
    cfg = BudgetPolicyConfig.normalize(
        mode="on",
        algorithm="thompson",
        warmup_decisions=0,
        epsilon=0.0,
        state_path=tmp_path / "bts_upd.json",
        events_path=tmp_path / "bts_upd.jsonl",
        random_seed=7,
        decay=1.0,
    )
    ctrl = AdaptiveBudgetPolicyController(cfg)
    decision = ctrl.decide(
        feature_values={"bias": 1.0},
        default_action=ACTION_KEEP_BUDGET,
    )
    event = _record(ctrl, decision)
    assert event.model_updated is True
    ctrl.close()


def test_budget_thompson_end_to_end_cycle(tmp_path) -> None:
    cfg = BudgetPolicyConfig.normalize(
        mode="on",
        algorithm="thompson",
        warmup_decisions=0,
        epsilon=0.0,
        state_path=tmp_path / "bts_e2e.json",
        events_path=tmp_path / "bts_e2e.jsonl",
        random_seed=12,
        decay=1.0,
    )
    ctrl = AdaptiveBudgetPolicyController(cfg)
    for _ in range(5):
        decision = ctrl.decide(
            feature_values={"bias": 1.0},
            default_action=ACTION_KEEP_BUDGET,
        )
        assert decision.algorithm == "thompson"
        _record(ctrl, decision)
    ctrl.close()


def test_budget_thompson_state_roundtrip(tmp_path) -> None:
    state_path = tmp_path / "bts_rt.json"
    events_path = tmp_path / "bts_rt.jsonl"

    cfg1 = BudgetPolicyConfig.normalize(
        mode="on",
        algorithm="thompson",
        warmup_decisions=0,
        epsilon=0.0,
        state_path=state_path,
        events_path=events_path,
        random_seed=42,
        decay=1.0,
    )
    ctrl1 = AdaptiveBudgetPolicyController(cfg1)
    for _ in range(3):
        d = ctrl1.decide(feature_values={"bias": 1.0}, default_action=ACTION_KEEP_BUDGET)
        _record(ctrl1, d)
    ctrl1.close()

    cfg2 = BudgetPolicyConfig.normalize(
        mode="on",
        algorithm="thompson",
        warmup_decisions=0,
        epsilon=0.0,
        state_path=state_path,
        events_path=events_path,
        random_seed=42,
        decay=1.0,
    )
    ctrl2 = AdaptiveBudgetPolicyController(cfg2)
    assert ctrl2.state.decisions == 3
    assert ctrl2.state.updates == 3
    assert sum(ctrl2.state.action_counts.values()) == 3
    d2 = ctrl2.decide(feature_values={"bias": 1.0}, default_action=ACTION_KEEP_BUDGET)
    assert d2.algorithm == "thompson"
    for score in d2.scores.values():
        assert math.isfinite(score)
    ctrl2.close()
