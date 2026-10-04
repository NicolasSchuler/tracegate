"""Tests for bootstrapping adaptive controller state from historical events."""

from __future__ import annotations

import dataclasses
from pathlib import Path

import numpy as np
import pytest

from evals.adaptive_budget_online_policy import (
    ACTION_BUDGET_FAMILY_TOKENS,
    ACTION_INCREASE_BUDGET,
    ACTION_KEEP_BUDGET,
    AdaptiveBudgetPolicyController,
    BudgetPolicyConfig,
)
from evals.adaptive_online_policy import (
    ACTION_INVOKE_HELPER,
    ACTION_SKIP_HELPER,
    AdaptiveOnlinePolicyController,
    OnlinePolicyConfig,
)
from evals.bootstrap_policy_state import (
    LoadedPolicyEvent,
    ReplayState,
    _detect_policy_kind,
    _load_policy_events,
)
from evals.bootstrap_policy_state import (
    main as bootstrap_policy_state_main,
)
from llmdebug import _json as json


def _load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def _assert_replay_state_matches(
    expected: dict[str, object],
    observed: dict[str, object],
    *,
    keys: tuple[str, ...] = (
        "feature_names",
        "decisions",
        "updates",
        "action_counts",
        "feature_scaling_history",
        "feature_scaling_caps",
    ),
) -> None:
    for key in keys:
        assert observed[key] == expected[key]
    expected_A = expected["A"]
    observed_A = observed["A"]
    assert isinstance(expected_A, dict)
    assert isinstance(observed_A, dict)
    assert set(observed_A) == set(expected_A)
    for action, expected_matrix in expected_A.items():
        assert np.allclose(np.array(observed_A[action]), np.array(expected_matrix))
    expected_b = expected["b"]
    observed_b = observed["b"]
    assert isinstance(expected_b, dict)
    assert isinstance(observed_b, dict)
    assert set(observed_b) == set(expected_b)
    for action, expected_vector in expected_b.items():
        assert np.allclose(np.array(observed_b[action]), np.array(expected_vector))


def test_bootstrap_online_state_replays_saved_controller_state(tmp_path: Path) -> None:
    events_path = tmp_path / "online_events.jsonl"
    saved_state_path = tmp_path / "online_state_saved.json"
    bootstrapped_state_path = tmp_path / "online_state_bootstrapped.json"
    controller = AdaptiveOnlinePolicyController(
        OnlinePolicyConfig.normalize(
            mode="on",
            decay=0.98,
            state_path=saved_state_path,
            events_path=events_path,
            random_seed=0,
        )
    )
    decision_one = controller.decide(
        feature_values={"bias": 1.0, "attempt_frac": 0.25},
        default_action=ACTION_SKIP_HELPER,
        available_actions=(ACTION_SKIP_HELPER,),
    )
    controller.record_outcome(
        decision=decision_one,
        run_id="bootstrap-online",
        case_id="case-1",
        condition="layered_adaptive_llm",
        attempt=1,
        note="first",
        reward_scope="helper_local_v1",
        reward=0.25,
        reward_success_hint=0.0,
        reward_progress_hint=0.1,
        reward_progress_score=0.1,
        reward_token_penalty=0.02,
        reward_latency_penalty=0.01,
        reward_stagnation_penalty=0.0,
        reward_helper_materialization_bonus=0.0,
        success=False,
        local_llm_call_count=1,
        local_tokens_prompt=20,
        local_tokens_completion=10,
        local_tokens_total=30,
        local_wall_sec=1.0,
        override_reason="",
        behavior_action=ACTION_SKIP_HELPER,
    )
    decision_two = controller.decide(
        feature_values={"bias": 1.0, "attempt_frac": 0.5, "stagnation_streak": 1.0},
        default_action=ACTION_INVOKE_HELPER,
        available_actions=(ACTION_INVOKE_HELPER,),
    )
    controller.record_outcome(
        decision=decision_two,
        run_id="bootstrap-online",
        case_id="case-2",
        condition="layered_adaptive_llm",
        attempt=1,
        note="second",
        reward_scope="helper_local_v1",
        reward=0.8,
        reward_success_hint=1.0,
        reward_progress_hint=0.2,
        reward_progress_score=0.2,
        reward_token_penalty=0.01,
        reward_latency_penalty=0.01,
        reward_stagnation_penalty=0.0,
        reward_helper_materialization_bonus=0.05,
        success=True,
        local_llm_call_count=1,
        local_tokens_prompt=15,
        local_tokens_completion=10,
        local_tokens_total=25,
        local_wall_sec=0.8,
        override_reason="",
        behavior_action=ACTION_INVOKE_HELPER,
    )
    controller.close()

    exit_code = bootstrap_policy_state_main(
        [
            "--policy-kind",
            "online",
            "--events",
            str(events_path),
            "--output-state",
            str(bootstrapped_state_path),
        ]
    )

    assert exit_code == 0
    saved = _load_json(saved_state_path)
    bootstrapped = _load_json(bootstrapped_state_path)
    assert bootstrapped["schema_version"] == saved["schema_version"]
    assert bootstrapped["algorithm"] == saved["algorithm"]
    assert bootstrapped["feature_scaling_profile"] == saved["feature_scaling_profile"]
    assert bootstrapped["decay"] == saved["decay"]
    _assert_replay_state_matches(saved, bootstrapped)
    assert bootstrapped["bootstrap_kind"] == "online"
    assert bootstrapped["bootstrap_warning_codes"] == []
    reloaded_controller = AdaptiveOnlinePolicyController(
        OnlinePolicyConfig.normalize(
            mode="on",
            init_state_path=bootstrapped_state_path,
            state_path=tmp_path / "online_state_reloaded.json",
            decay=1.0,
        )
    )
    assert reloaded_controller.load_error is None


def test_bootstrap_online_thompson_state_replays_saved_controller_state(tmp_path: Path) -> None:
    events_path = tmp_path / "online_thompson_events.jsonl"
    saved_state_path = tmp_path / "online_thompson_state_saved.json"
    bootstrapped_state_path = tmp_path / "online_thompson_state_bootstrapped.json"
    controller = AdaptiveOnlinePolicyController(
        OnlinePolicyConfig.normalize(
            mode="on",
            algorithm="thompson",
            decay=0.98,
            state_path=saved_state_path,
            events_path=events_path,
            random_seed=0,
        )
    )
    decision_one = controller.decide(
        feature_values={"bias": 1.0, "attempt_frac": 0.25},
        default_action=ACTION_SKIP_HELPER,
        available_actions=(ACTION_SKIP_HELPER,),
    )
    controller.record_outcome(
        decision=decision_one,
        run_id="bootstrap-online-thompson",
        case_id="case-1",
        condition="layered_adaptive_llm",
        attempt=1,
        note="first",
        reward_scope="helper_local_v1",
        reward=0.25,
        reward_success_hint=0.0,
        reward_progress_hint=0.1,
        reward_progress_score=0.1,
        reward_token_penalty=0.02,
        reward_latency_penalty=0.01,
        reward_stagnation_penalty=0.0,
        reward_helper_materialization_bonus=0.0,
        success=False,
        local_llm_call_count=1,
        local_tokens_prompt=20,
        local_tokens_completion=10,
        local_tokens_total=30,
        local_wall_sec=1.0,
        override_reason="",
        behavior_action=ACTION_SKIP_HELPER,
    )
    decision_two = controller.decide(
        feature_values={"bias": 1.0, "attempt_frac": 0.5, "stagnation_streak": 1.0},
        default_action=ACTION_INVOKE_HELPER,
        available_actions=(ACTION_INVOKE_HELPER,),
    )
    controller.record_outcome(
        decision=decision_two,
        run_id="bootstrap-online-thompson",
        case_id="case-2",
        condition="layered_adaptive_llm",
        attempt=1,
        note="second",
        reward_scope="helper_local_v1",
        reward=0.8,
        reward_success_hint=1.0,
        reward_progress_hint=0.2,
        reward_progress_score=0.2,
        reward_token_penalty=0.01,
        reward_latency_penalty=0.01,
        reward_stagnation_penalty=0.0,
        reward_helper_materialization_bonus=0.05,
        success=True,
        local_llm_call_count=1,
        local_tokens_prompt=15,
        local_tokens_completion=10,
        local_tokens_total=25,
        local_wall_sec=0.8,
        override_reason="",
        behavior_action=ACTION_INVOKE_HELPER,
    )
    controller.close()

    exit_code = bootstrap_policy_state_main(
        [
            "--policy-kind",
            "online",
            "--events",
            str(events_path),
            "--output-state",
            str(bootstrapped_state_path),
        ]
    )

    assert exit_code == 0
    saved = _load_json(saved_state_path)
    bootstrapped = _load_json(bootstrapped_state_path)
    assert bootstrapped["algorithm"] == "thompson"
    _assert_replay_state_matches(saved, bootstrapped)
    reloaded_controller = AdaptiveOnlinePolicyController(
        OnlinePolicyConfig.normalize(
            mode="on",
            algorithm="thompson",
            init_state_path=bootstrapped_state_path,
            state_path=tmp_path / "online_thompson_state_reloaded.json",
            decay=1.0,
        )
    )
    assert reloaded_controller.load_error is None


def test_bootstrap_budget_hierarchical_state_replays_saved_controller_state(
    tmp_path: Path,
) -> None:
    events_path = tmp_path / "budget_events.jsonl"
    saved_state_path = tmp_path / "budget_state_saved.json"
    bootstrapped_state_path = tmp_path / "budget_state_bootstrapped.json"
    controller = AdaptiveBudgetPolicyController(
        BudgetPolicyConfig.normalize(
            mode="on",
            architecture="hierarchical_v1",
            decay=0.98,
            state_path=saved_state_path,
            events_path=events_path,
            random_seed=0,
        )
    )
    decision_one = controller.decide(
        feature_values={"bias": 1.0, "attempt_frac": 0.25},
        default_action=ACTION_KEEP_BUDGET,
    )
    controller.record_outcome(
        decision=decision_one,
        run_id="bootstrap-budget",
        case_id="case-1",
        condition="layered_adaptive_llm",
        attempt=1,
        note="gate-only",
        reward_scope="budget_local_v1",
        reward=-0.1,
        reward_success_hint=0.0,
        reward_progress_hint=0.1,
        reward_progress_score=0.1,
        reward_token_penalty=0.02,
        reward_latency_penalty=0.01,
        reward_stagnation_penalty=0.0,
        reward_budget_exceeded_penalty=0.0,
        request_budget_exceeded=False,
        success=False,
        local_llm_call_count=1,
        local_tokens_prompt=12,
        local_tokens_completion=8,
        local_tokens_total=20,
        local_wall_sec=0.6,
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
    decision_two = controller.decide(
        feature_values={"bias": 1.0, "attempt_frac": 0.5},
        default_action=ACTION_INCREASE_BUDGET,
        family_feature_values={"bias": 1.0, "budget_can_raise_tokens": 1.0},
        available_families=(ACTION_BUDGET_FAMILY_TOKENS,),
    )
    controller.record_outcome(
        decision=decision_two,
        run_id="bootstrap-budget",
        case_id="case-2",
        condition="layered_adaptive_llm",
        attempt=1,
        note="gate-and-family",
        reward_scope="budget_local_v1",
        reward=0.4,
        reward_success_hint=1.0,
        reward_progress_hint=0.25,
        reward_progress_score=0.25,
        reward_token_penalty=0.03,
        reward_latency_penalty=0.01,
        reward_stagnation_penalty=0.0,
        reward_budget_exceeded_penalty=0.0,
        request_budget_exceeded=False,
        success=True,
        local_llm_call_count=1,
        local_tokens_prompt=14,
        local_tokens_completion=9,
        local_tokens_total=23,
        local_wall_sec=0.7,
        selected_multiplier=1.5,
        applied_multiplier=1.5,
        default_multiplier=1.0,
        default_controls={},
        selected_controls={},
        applied_controls={},
        selected_controls_changed={},
        applied_controls_changed={},
        partial_override=False,
        partial_override_errors={},
        tier_index_before=0,
        tier_index_after=1,
        override_reason="",
    )
    controller.close()

    exit_code = bootstrap_policy_state_main(
        [
            "--policy-kind",
            "budget",
            "--events",
            str(events_path),
            "--output-state",
            str(bootstrapped_state_path),
        ]
    )

    assert exit_code == 0
    saved = _load_json(saved_state_path)
    bootstrapped = _load_json(bootstrapped_state_path)
    assert bootstrapped["schema_version"] == saved["schema_version"]
    assert bootstrapped["architecture"] == saved["architecture"]
    assert bootstrapped["feature_scaling_profile"] == saved["feature_scaling_profile"]
    _assert_replay_state_matches(saved["gate_state"], bootstrapped["gate_state"])
    _assert_replay_state_matches(saved["family_state"], bootstrapped["family_state"])
    assert bootstrapped["bootstrap_kind"] == "budget"
    reloaded_controller = AdaptiveBudgetPolicyController(
        BudgetPolicyConfig.normalize(
            mode="on",
            architecture="hierarchical_v1",
            init_state_path=bootstrapped_state_path,
            state_path=tmp_path / "budget_state_reloaded.json",
            decay=1.0,
        )
    )
    assert reloaded_controller.load_error is None


def test_bootstrap_budget_thompson_state_replays_saved_controller_state(
    tmp_path: Path,
) -> None:
    events_path = tmp_path / "budget_thompson_events.jsonl"
    saved_state_path = tmp_path / "budget_thompson_state_saved.json"
    bootstrapped_state_path = tmp_path / "budget_thompson_state_bootstrapped.json"
    controller = AdaptiveBudgetPolicyController(
        BudgetPolicyConfig.normalize(
            mode="on",
            algorithm="thompson",
            architecture="hierarchical_v1",
            decay=0.98,
            state_path=saved_state_path,
            events_path=events_path,
            random_seed=0,
        )
    )
    decision_one = controller.decide(
        feature_values={"bias": 1.0, "attempt_frac": 0.25},
        default_action=ACTION_KEEP_BUDGET,
    )
    controller.record_outcome(
        decision=decision_one,
        run_id="bootstrap-budget-thompson",
        case_id="case-1",
        condition="layered_adaptive_llm",
        attempt=1,
        note="gate-only",
        reward_scope="budget_local_v1",
        reward=-0.1,
        reward_success_hint=0.0,
        reward_progress_hint=0.1,
        reward_progress_score=0.1,
        reward_token_penalty=0.02,
        reward_latency_penalty=0.01,
        reward_stagnation_penalty=0.0,
        reward_budget_exceeded_penalty=0.0,
        request_budget_exceeded=False,
        success=False,
        local_llm_call_count=1,
        local_tokens_prompt=12,
        local_tokens_completion=8,
        local_tokens_total=20,
        local_wall_sec=0.6,
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
    decision_two = controller.decide(
        feature_values={"bias": 1.0, "attempt_frac": 0.5},
        default_action=ACTION_INCREASE_BUDGET,
        family_feature_values={"bias": 1.0, "budget_can_raise_tokens": 1.0},
        available_families=(ACTION_BUDGET_FAMILY_TOKENS,),
    )
    controller.record_outcome(
        decision=decision_two,
        run_id="bootstrap-budget-thompson",
        case_id="case-2",
        condition="layered_adaptive_llm",
        attempt=1,
        note="gate-and-family",
        reward_scope="budget_local_v1",
        reward=0.4,
        reward_success_hint=1.0,
        reward_progress_hint=0.25,
        reward_progress_score=0.25,
        reward_token_penalty=0.03,
        reward_latency_penalty=0.01,
        reward_stagnation_penalty=0.0,
        reward_budget_exceeded_penalty=0.0,
        request_budget_exceeded=False,
        success=True,
        local_llm_call_count=1,
        local_tokens_prompt=14,
        local_tokens_completion=9,
        local_tokens_total=23,
        local_wall_sec=0.7,
        selected_multiplier=1.5,
        applied_multiplier=1.5,
        default_multiplier=1.0,
        default_controls={},
        selected_controls={},
        applied_controls={},
        selected_controls_changed={},
        applied_controls_changed={},
        partial_override=False,
        partial_override_errors={},
        tier_index_before=0,
        tier_index_after=1,
        override_reason="",
    )
    controller.close()

    exit_code = bootstrap_policy_state_main(
        [
            "--policy-kind",
            "budget",
            "--events",
            str(events_path),
            "--output-state",
            str(bootstrapped_state_path),
        ]
    )

    assert exit_code == 0
    saved = _load_json(saved_state_path)
    bootstrapped = _load_json(bootstrapped_state_path)
    assert bootstrapped["algorithm"] == "thompson"
    _assert_replay_state_matches(saved["gate_state"], bootstrapped["gate_state"])
    _assert_replay_state_matches(saved["family_state"], bootstrapped["family_state"])
    reloaded_controller = AdaptiveBudgetPolicyController(
        BudgetPolicyConfig.normalize(
            mode="on",
            algorithm="thompson",
            architecture="hierarchical_v1",
            init_state_path=bootstrapped_state_path,
            state_path=tmp_path / "budget_thompson_state_reloaded.json",
            decay=1.0,
        )
    )
    assert reloaded_controller.load_error is None


def test_bootstrap_budget_bwk_replays_budget_telemetry(tmp_path: Path) -> None:
    events_path = tmp_path / "bwk_events.jsonl"
    saved_state_path = tmp_path / "bwk_state_saved.json"
    bootstrapped_state_path = tmp_path / "bwk_state_bootstrapped.json"
    controller = AdaptiveBudgetPolicyController(
        dataclasses.replace(
            BudgetPolicyConfig.normalize(
                mode="on",
                architecture="bwk_v1",
                decay=0.98,
                bwk_run_token_budget=100,
                state_path=saved_state_path,
                events_path=events_path,
                random_seed=0,
            ),
            # Pacing target 100 / 5 = 20 tokens per decision (POL-04).
            bwk_horizon_decisions=5,
        )
    )
    decision_one = controller.decide(
        feature_values={"bias": 1.0, "attempt_frac": 0.25},
        default_action=ACTION_KEEP_BUDGET,
    )
    controller.record_outcome(
        decision=decision_one,
        run_id="bootstrap-bwk",
        case_id="case-1",
        condition="layered_adaptive_llm",
        attempt=1,
        note="first",
        reward_scope="budget_local_v1",
        reward=0.0,
        reward_success_hint=0.0,
        reward_progress_hint=0.0,
        reward_progress_score=0.0,
        reward_token_penalty=0.01,
        reward_latency_penalty=0.01,
        reward_stagnation_penalty=0.0,
        reward_budget_exceeded_penalty=0.0,
        request_budget_exceeded=False,
        success=False,
        local_llm_call_count=1,
        local_tokens_prompt=20,
        local_tokens_completion=10,
        local_tokens_total=30,
        local_wall_sec=0.5,
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
    decision_two = controller.decide(
        feature_values={"bias": 1.0, "attempt_frac": 0.75},
        default_action=ACTION_KEEP_BUDGET,
    )
    controller.record_outcome(
        decision=decision_two,
        run_id="bootstrap-bwk",
        case_id="case-2",
        condition="layered_adaptive_llm",
        attempt=1,
        note="second",
        reward_scope="budget_local_v1",
        reward=0.0,
        reward_success_hint=0.0,
        reward_progress_hint=0.0,
        reward_progress_score=0.0,
        reward_token_penalty=0.01,
        reward_latency_penalty=0.01,
        reward_stagnation_penalty=0.0,
        reward_budget_exceeded_penalty=0.0,
        request_budget_exceeded=False,
        success=False,
        local_llm_call_count=1,
        local_tokens_prompt=40,
        local_tokens_completion=40,
        local_tokens_total=80,
        local_wall_sec=0.8,
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
    controller.close()

    exit_code = bootstrap_policy_state_main(
        [
            "--policy-kind",
            "budget",
            "--events",
            str(events_path),
            "--output-state",
            str(bootstrapped_state_path),
        ]
    )

    assert exit_code == 0
    saved = _load_json(saved_state_path)
    bootstrapped = _load_json(bootstrapped_state_path)
    assert bootstrapped["architecture"] == "bwk_v1"
    assert bootstrapped["bwk_run_token_budget"] == saved["bwk_run_token_budget"]
    assert bootstrapped["bwk_remaining_token_budget"] == saved["bwk_remaining_token_budget"]
    assert bootstrapped["bwk_cumulative_tokens_charged"] == saved["bwk_cumulative_tokens_charged"]
    assert float(saved["bwk_dual_variable"]) > 0.0
    assert np.isclose(
        float(bootstrapped["bwk_dual_variable"]),
        float(saved["bwk_dual_variable"]),
    )
    assert bootstrapped["bwk_cost_counts"] == saved["bwk_cost_counts"]
    assert bootstrapped["bwk_cost_sums"] == pytest.approx(saved["bwk_cost_sums"])
    assert "bwk_dual_variable_mismatch" not in bootstrapped["bootstrap_warning_codes"]


def test_bootstrap_online_preserves_feature_expansion_without_model_update(
    tmp_path: Path,
) -> None:
    events_path = tmp_path / "shadow_events.jsonl"
    output_path = tmp_path / "shadow_state.json"
    events = [
        {
            "run_id": "shadow-replay",
            "case_id": "case-shadow",
            "condition": "layered_adaptive_llm",
            "attempt": 1,
            "decision_id": "shadow-1",
            "mode": "shadow",
            "algorithm": "linucb",
            "decision_index": 1,
            "selected_action": "invoke_helper",
            "applied_action": "skip_helper",
            "behavior_action": "skip_helper",
            "feature_scaling_profile": "v1",
            "feature_values": {"bias": 1.0, "new_signal": 1.0},
            "feature_scaling_caps": {},
            "reward": 0.0,
            "configured_alpha": 0.8,
            "configured_epsilon": 0.05,
            "configured_decay": 1.0,
            "warmup_active": False,
            "model_updated": False,
        },
        {
            "run_id": "shadow-replay",
            "case_id": "case-updated",
            "condition": "layered_adaptive_llm",
            "attempt": 1,
            "decision_id": "shadow-2",
            "mode": "on",
            "algorithm": "linucb",
            "decision_index": 2,
            "selected_action": "skip_helper",
            "applied_action": "skip_helper",
            "behavior_action": "skip_helper",
            "feature_scaling_profile": "v1",
            "feature_values": {"bias": 1.0},
            "feature_scaling_caps": {},
            "reward": 0.5,
            "configured_alpha": 0.8,
            "configured_epsilon": 0.05,
            "configured_decay": 1.0,
            "warmup_active": False,
            "model_updated": True,
        },
    ]
    events_path.write_text(
        "\n".join(json.dumps(event, ensure_ascii=False) for event in events) + "\n",
        encoding="utf-8",
    )

    exit_code = bootstrap_policy_state_main(
        [
            "--policy-kind",
            "online",
            "--events",
            str(events_path),
            "--output-state",
            str(output_path),
        ]
    )

    assert exit_code == 0
    payload = _load_json(output_path)
    assert payload["decisions"] == 2
    assert payload["updates"] == 1
    assert "new_signal" in payload["feature_names"]
    assert np.array(payload["A"]["skip_helper"]).shape == (2, 2)


def test_bootstrap_policy_state_rejects_mixed_policy_kinds(tmp_path: Path) -> None:
    events_path = tmp_path / "mixed_events.jsonl"
    output_path = tmp_path / "mixed_state.json"
    events_path.write_text(
        "\n".join(
            [
                json.dumps(
                    {
                        "run_id": "mixed",
                        "case_id": "online",
                        "condition": "layered_adaptive_llm",
                        "attempt": 1,
                        "decision_id": "o1",
                        "algorithm": "linucb",
                        "selected_action": "skip_helper",
                        "behavior_action": "skip_helper",
                    }
                ),
                json.dumps(
                    {
                        "run_id": "mixed",
                        "case_id": "budget",
                        "condition": "layered_adaptive_llm",
                        "attempt": 1,
                        "decision_id": "b1",
                        "algorithm": "linucb",
                        "architecture": "tier_binary",
                        "selected_action": "keep_budget",
                        "behavior_action": "keep_budget",
                    }
                ),
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    exit_code = bootstrap_policy_state_main(
        [
            "--events",
            str(events_path),
            "--output-state",
            str(output_path),
        ]
    )

    assert exit_code == 2
    assert not output_path.exists()


def test_bootstrap_policy_state_rejects_mixed_decay_logs(tmp_path: Path) -> None:
    events_path = tmp_path / "mixed_decay_events.jsonl"
    output_path = tmp_path / "mixed_decay_state.json"
    events = [
        {
            "run_id": "mixed-decay",
            "case_id": "case-1",
            "condition": "layered_adaptive_llm",
            "attempt": 1,
            "decision_id": "d1",
            "mode": "on",
            "algorithm": "linucb",
            "selected_action": "skip_helper",
            "behavior_action": "skip_helper",
            "feature_scaling_profile": "v1",
            "feature_values": {"bias": 1.0},
            "reward": 0.1,
            "configured_alpha": 0.8,
            "configured_epsilon": 0.05,
            "configured_decay": 1.0,
            "model_updated": True,
        },
        {
            "run_id": "mixed-decay",
            "case_id": "case-2",
            "condition": "layered_adaptive_llm",
            "attempt": 1,
            "decision_id": "d2",
            "mode": "on",
            "algorithm": "linucb",
            "selected_action": "skip_helper",
            "behavior_action": "skip_helper",
            "feature_scaling_profile": "v1",
            "feature_values": {"bias": 1.0},
            "reward": 0.2,
            "configured_alpha": 0.8,
            "configured_epsilon": 0.05,
            "configured_decay": 0.98,
            "model_updated": True,
        },
    ]
    events_path.write_text(
        "\n".join(json.dumps(event, ensure_ascii=False) for event in events) + "\n",
        encoding="utf-8",
    )

    exit_code = bootstrap_policy_state_main(
        [
            "--policy-kind",
            "online",
            "--events",
            str(events_path),
            "--output-state",
            str(output_path),
        ]
    )

    assert exit_code == 2
    assert not output_path.exists()


# ---------------------------------------------------------------------------
# New coverage tests - error paths, warnings, and strict-input mode
# ---------------------------------------------------------------------------


def _write_jsonl(path: Path, events: list) -> None:  # type: ignore[type-arg]
    """Write a list of dicts as JSONL."""
    path.write_text(
        "\n".join(json.dumps(event, ensure_ascii=False) for event in events) + "\n",
        encoding="utf-8",
    )


def _minimal_online_event(
    *,
    run_id: str = "run-1",
    case_id: str = "case-1",
    condition: str = "layered_adaptive_llm",
    attempt: int = 1,
    decision_id: str = "d1",
    algorithm: str = "linucb",
    selected_action: str = "skip_helper",
    behavior_action: str = "skip_helper",
    model_updated: bool = True,
    reward: float = 0.5,
    **extra,
) -> dict:  # type: ignore[type-arg]
    return {
        "run_id": run_id,
        "case_id": case_id,
        "condition": condition,
        "attempt": attempt,
        "decision_id": decision_id,
        "mode": "on",
        "algorithm": algorithm,
        "selected_action": selected_action,
        "behavior_action": behavior_action,
        "feature_scaling_profile": "v1",
        "feature_values": {"bias": 1.0},
        "feature_scaling_caps": {},
        "reward": reward,
        "configured_alpha": 0.8,
        "configured_epsilon": 0.05,
        "configured_decay": 1.0,
        "warmup_active": False,
        "model_updated": model_updated,
        **extra,
    }


def _minimal_budget_event(
    *,
    run_id: str = "run-b",
    case_id: str = "case-b",
    condition: str = "layered_adaptive_llm",
    attempt: int = 1,
    decision_id: str = "db1",
    algorithm: str = "linucb",
    architecture: str = "tier_binary",
    selected_action: str = "keep_budget",
    behavior_action: str = "keep_budget",
    **extra,
) -> dict:  # type: ignore[type-arg]
    return {
        "run_id": run_id,
        "case_id": case_id,
        "condition": condition,
        "attempt": attempt,
        "decision_id": decision_id,
        "mode": "on",
        "algorithm": algorithm,
        "architecture": architecture,
        "selected_action": selected_action,
        "behavior_action": behavior_action,
        "feature_scaling_profile": "v1",
        "feature_values": {"bias": 1.0},
        "feature_scaling_caps": {},
        "reward": 0.3,
        "configured_alpha": 0.6,
        "configured_epsilon": 0.03,
        "configured_decay": 1.0,
        "warmup_active": False,
        "model_updated": False,
        **extra,
    }


# --- main() error/warning paths ---


def test_main_returns_1_when_no_events_glob_matches(tmp_path: Path) -> None:
    """No file matches --events glob → exit code 1, no output."""
    output_path = tmp_path / "out.json"
    exit_code = bootstrap_policy_state_main(
        [
            "--policy-kind",
            "online",
            "--events",
            str(tmp_path / "nonexistent_*.jsonl"),
            "--output-state",
            str(output_path),
        ]
    )
    assert exit_code == 1
    assert not output_path.exists()


def test_main_returns_1_when_no_valid_events_found(tmp_path: Path) -> None:
    """JSONL with only blank lines → no valid events → exit code 1."""
    events_path = tmp_path / "empty.jsonl"
    events_path.write_text("\n\n\n", encoding="utf-8")
    output_path = tmp_path / "out.json"

    exit_code = bootstrap_policy_state_main(
        [
            "--policy-kind",
            "online",
            "--events",
            str(events_path),
            "--output-state",
            str(output_path),
        ]
    )

    assert exit_code == 1
    assert not output_path.exists()


def test_main_prints_warnings_and_returns_0(tmp_path: Path) -> None:
    """Events with feature_scaling_history_sizes → warning in output, still exit 0."""
    events_path = tmp_path / "history_events.jsonl"
    event = _minimal_online_event()
    event["feature_scaling_history_sizes"] = {"bias": 5}
    _write_jsonl(events_path, [event])
    output_path = tmp_path / "out.json"

    exit_code = bootstrap_policy_state_main(
        [
            "--policy-kind",
            "online",
            "--events",
            str(events_path),
            "--output-state",
            str(output_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert "feature_scaling_history_not_reconstructable" in payload["bootstrap_warning_codes"]


def test_main_strict_input_fails_on_unreadable_file(tmp_path: Path) -> None:
    """--strict-input with non-existent file → exit code 2."""
    output_path = tmp_path / "out.json"
    missing = tmp_path / "does_not_exist.jsonl"
    exit_code = bootstrap_policy_state_main(
        [
            "--policy-kind",
            "online",
            "--strict-input",
            "--events",
            str(missing),
            "--output-state",
            str(output_path),
        ]
    )
    assert exit_code == 2
    assert not output_path.exists()


def test_main_strict_input_fails_on_malformed_json(tmp_path: Path) -> None:
    """--strict-input with file containing malformed JSON → exit code 2."""
    events_path = tmp_path / "bad.jsonl"
    events_path.write_text("not valid json{{{\n", encoding="utf-8")
    output_path = tmp_path / "out.json"

    exit_code = bootstrap_policy_state_main(
        [
            "--policy-kind",
            "online",
            "--strict-input",
            "--events",
            str(events_path),
            "--output-state",
            str(output_path),
        ]
    )

    assert exit_code == 2
    assert not output_path.exists()


def test_main_strict_input_fails_on_non_dict_record(tmp_path: Path) -> None:
    """--strict-input with a JSON-array root → exit code 2."""
    events_path = tmp_path / "array.jsonl"
    events_path.write_text('["not", "a", "dict"]\n', encoding="utf-8")
    output_path = tmp_path / "out.json"

    exit_code = bootstrap_policy_state_main(
        [
            "--policy-kind",
            "online",
            "--strict-input",
            "--events",
            str(events_path),
            "--output-state",
            str(output_path),
        ]
    )

    assert exit_code == 2
    assert not output_path.exists()


def test_main_strict_input_fails_on_missing_required_fields(tmp_path: Path) -> None:
    """--strict-input with valid JSON but missing run_id/case_id → exit code 2."""
    events_path = tmp_path / "partial.jsonl"
    events_path.write_text('{"algorithm": "linucb"}\n', encoding="utf-8")
    output_path = tmp_path / "out.json"

    exit_code = bootstrap_policy_state_main(
        [
            "--policy-kind",
            "online",
            "--strict-input",
            "--events",
            str(events_path),
            "--output-state",
            str(output_path),
        ]
    )

    assert exit_code == 2
    assert not output_path.exists()


# --- unsupported algorithm / architecture ---


def test_bootstrap_online_rejects_unsupported_algorithm(tmp_path: Path) -> None:
    """Online event with unknown algorithm → ValueError → exit code 2."""
    events_path = tmp_path / "events.jsonl"
    _write_jsonl(events_path, [_minimal_online_event(algorithm="unknown_algo")])
    output_path = tmp_path / "out.json"

    exit_code = bootstrap_policy_state_main(
        [
            "--policy-kind",
            "online",
            "--events",
            str(events_path),
            "--output-state",
            str(output_path),
        ]
    )

    assert exit_code == 2
    assert not output_path.exists()


def test_bootstrap_budget_rejects_unsupported_algorithm(tmp_path: Path) -> None:
    """Budget event with unknown algorithm → ValueError → exit code 2."""
    events_path = tmp_path / "events.jsonl"
    _write_jsonl(events_path, [_minimal_budget_event(algorithm="unknown_algo")])
    output_path = tmp_path / "out.json"

    exit_code = bootstrap_policy_state_main(
        [
            "--policy-kind",
            "budget",
            "--events",
            str(events_path),
            "--output-state",
            str(output_path),
        ]
    )

    assert exit_code == 2
    assert not output_path.exists()


def test_bootstrap_budget_rejects_unsupported_architecture(tmp_path: Path) -> None:
    """Budget event with unsupported architecture → ValueError → exit code 2."""
    events_path = tmp_path / "events.jsonl"
    _write_jsonl(events_path, [_minimal_budget_event(architecture="unknown_arch")])
    output_path = tmp_path / "out.json"

    exit_code = bootstrap_policy_state_main(
        [
            "--policy-kind",
            "budget",
            "--events",
            str(events_path),
            "--output-state",
            str(output_path),
        ]
    )

    assert exit_code == 2
    assert not output_path.exists()


# --- online warning codes ---


def test_bootstrap_online_warns_on_invalid_selected_action(tmp_path: Path) -> None:
    """selected_action not in ONLINE_POLICY_ACTIONS → warning code emitted."""
    events_path = tmp_path / "events.jsonl"
    _write_jsonl(
        events_path,
        [_minimal_online_event(selected_action="totally_invalid_action", model_updated=False)],
    )
    output_path = tmp_path / "out.json"

    exit_code = bootstrap_policy_state_main(
        [
            "--policy-kind",
            "online",
            "--events",
            str(events_path),
            "--output-state",
            str(output_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert "invalid_selected_action_skipped" in payload["bootstrap_warning_codes"]


def test_bootstrap_online_warns_on_invalid_behavior_action(tmp_path: Path) -> None:
    """model_updated=True with invalid behavior_action → warning code emitted."""
    events_path = tmp_path / "events.jsonl"
    _write_jsonl(
        events_path,
        [
            _minimal_online_event(
                behavior_action="totally_invalid_action",
                model_updated=True,
            )
        ],
    )
    output_path = tmp_path / "out.json"

    exit_code = bootstrap_policy_state_main(
        [
            "--policy-kind",
            "online",
            "--events",
            str(events_path),
            "--output-state",
            str(output_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert "invalid_behavior_action_skipped" in payload["bootstrap_warning_codes"]


# --- budget warning codes ---


def test_bootstrap_budget_warns_invalid_selected_action(tmp_path: Path) -> None:
    """Budget gate selected_action not in BUDGET_POLICY_ACTIONS → warning."""
    events_path = tmp_path / "events.jsonl"
    _write_jsonl(
        events_path,
        [_minimal_budget_event(selected_action="totally_invalid")],
    )
    output_path = tmp_path / "out.json"

    exit_code = bootstrap_policy_state_main(
        [
            "--policy-kind",
            "budget",
            "--events",
            str(events_path),
            "--output-state",
            str(output_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert "invalid_budget_selected_action_skipped" in payload["bootstrap_warning_codes"]


def test_bootstrap_budget_warns_invalid_selected_family(tmp_path: Path) -> None:
    """Budget family decision with invalid selected_family → warning."""
    events_path = tmp_path / "events.jsonl"
    event = _minimal_budget_event()
    event["selected_family"] = "totally_invalid_family"
    _write_jsonl(events_path, [event])
    output_path = tmp_path / "out.json"

    exit_code = bootstrap_policy_state_main(
        [
            "--policy-kind",
            "budget",
            "--events",
            str(events_path),
            "--output-state",
            str(output_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert "invalid_budget_selected_family_skipped" in payload["bootstrap_warning_codes"]


def test_bootstrap_budget_warns_invalid_behavior_action(tmp_path: Path) -> None:
    """gate_model_updated=True with invalid behavior_action → warning."""
    events_path = tmp_path / "events.jsonl"
    event = _minimal_budget_event(behavior_action="totally_invalid")
    event["gate_model_updated"] = True
    _write_jsonl(events_path, [event])
    output_path = tmp_path / "out.json"

    exit_code = bootstrap_policy_state_main(
        [
            "--policy-kind",
            "budget",
            "--events",
            str(events_path),
            "--output-state",
            str(output_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert "invalid_budget_behavior_action_skipped" in payload["bootstrap_warning_codes"]


def test_bootstrap_budget_warns_invalid_behavior_family(tmp_path: Path) -> None:
    """family_model_updated=True with invalid behavior_family → warning."""
    events_path = tmp_path / "events.jsonl"
    event = _minimal_budget_event()
    event["family_model_updated"] = True
    event["behavior_family"] = "totally_invalid_family"
    _write_jsonl(events_path, [event])
    output_path = tmp_path / "out.json"

    exit_code = bootstrap_policy_state_main(
        [
            "--policy-kind",
            "budget",
            "--events",
            str(events_path),
            "--output-state",
            str(output_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert "invalid_budget_behavior_family_skipped" in payload["bootstrap_warning_codes"]


def test_bootstrap_bwk_warns_on_remaining_budget_mismatch(tmp_path: Path) -> None:
    """BWK event with tampered bwk_remaining_token_budget_after → mismatch warning."""
    events_path = tmp_path / "bwk_events.jsonl"
    event = _minimal_budget_event(architecture="bwk_v1")
    event["bwk_run_token_budget"] = 1000
    event["bwk_tokens_charged"] = 100
    # Report remaining that doesn't match what replay computes (1000 - 100 = 900)
    event["bwk_remaining_token_budget_after"] = 500  # wrong → mismatch
    _write_jsonl(events_path, [event])
    output_path = tmp_path / "out.json"

    exit_code = bootstrap_policy_state_main(
        [
            "--policy-kind",
            "budget",
            "--events",
            str(events_path),
            "--output-state",
            str(output_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert "bwk_remaining_token_budget_mismatch" in payload["bootstrap_warning_codes"]


# --- ReplayState edge cases ---


def test_replay_state_post_init_normalises_empty_action_names() -> None:
    """ReplayState with empty action_names → normalises to ('default',)."""
    state = ReplayState(action_names=())
    assert state.action_names == ("default",)


# --- _load_policy_events edge cases ---


def test_load_policy_events_skips_invalid_json_in_lenient_mode(tmp_path: Path) -> None:
    """Malformed JSON lines in lenient mode → counted, not raised."""
    events_path = tmp_path / "bad.jsonl"
    good = _minimal_online_event()
    events_path.write_text(
        json.dumps(good, ensure_ascii=False) + "\n" + "not json{{{\n",
        encoding="utf-8",
    )

    events, errors = _load_policy_events([str(events_path)], strict_input=False)

    assert errors["json_lines_invalid"] == 1
    assert len(events) == 1


def test_load_policy_events_skips_non_dict_records_in_lenient_mode(tmp_path: Path) -> None:
    """JSON-array root record in lenient mode → counted in records_skipped."""
    events_path = tmp_path / "array.jsonl"
    events_path.write_text('["not", "a", "dict"]\n', encoding="utf-8")

    events, errors = _load_policy_events([str(events_path)], strict_input=False)

    assert errors["records_skipped"] == 1
    assert events == []


def test_load_policy_events_skips_records_missing_required_keys(tmp_path: Path) -> None:
    """Valid JSON dict without run_id/case_id/condition/attempt → records_skipped."""
    events_path = tmp_path / "partial.jsonl"
    events_path.write_text('{"algorithm": "linucb"}\n', encoding="utf-8")

    events, errors = _load_policy_events([str(events_path)], strict_input=False)

    assert errors["records_skipped"] == 1
    assert events == []


def test_load_policy_events_deduplicates_events(tmp_path: Path) -> None:
    """Two events with the same dedup key → duplicate_events_seen == 1, first wins (POL-18;
    this used to keep the last one)."""
    events_path = tmp_path / "dupe.jsonl"
    e1 = _minimal_online_event(reward=0.1)
    e2 = _minimal_online_event(reward=0.9)  # same key (run/case/condition/attempt/decision_id)
    _write_jsonl(events_path, [e1, e2])

    events, errors = _load_policy_events([str(events_path)], strict_input=False)

    assert errors["duplicate_events_seen"] == 1
    assert len(events) == 1
    assert float(events[0].payload["reward"]) == pytest.approx(0.1)


# --- _detect_policy_kind mismatch ---


def test_detect_policy_kind_mismatch_raises() -> None:
    """Requesting 'online' but event logs look like 'budget' → ValueError."""
    budget_event = LoadedPolicyEvent(
        payload=_minimal_budget_event(),
        src="test",
        line_no=1,
    )
    with pytest.raises(ValueError, match="requested --policy-kind"):
        _detect_policy_kind([budget_event], requested_kind="online")


# ---------------------------------------------------------------------------
# Additional coverage for private helpers (lines 80-81, 326-328, 439,
# 541-543, 558-564, 598, 607-614, 630-634, 647-649, 667-669, 684-688,
# 699, 723-741, 780-782, 794, 817, 828-832, 865-867)
# ---------------------------------------------------------------------------

from evals.bootstrap_policy_state import (  # noqa: E402
    _apply_recency_decay,
    _apply_reward_update,
    _budget_family_decision_logged,
    _ensure_feature_state,
    _event_key,
    _infer_uniform_float,
    _infer_uniform_positive_int,
    _infer_uniform_string,
    _last_float,
    _last_string,
    _resolved_warmup_decisions,
    _safe_float,
    _update_feature_scaling_caps,
)

# --- ReplayState.__post_init__ normalises extra keys in action_counts (lines 80-81) ---


def test_replay_state_post_init_drops_unknown_action_count_keys() -> None:
    """action_counts keys not in action_names are dropped on normalisation."""
    state = ReplayState(
        action_names=("a", "b"),
        action_counts={"a": 3, "c": 99},  # "c" not in names
    )
    assert "c" not in state.action_counts
    assert state.action_counts["a"] == 3
    assert state.action_counts["b"] == 0


# --- _event_key fallback for decision_id (lines 541, 543) ---


def test_event_key_uses_created_at_utc_when_decision_id_missing() -> None:
    """decision_id absent → falls back to created_at_utc (line 541)."""
    payload = {
        "run_id": "r1",
        "case_id": "c1",
        "condition": "cond",
        "attempt": 1,
        "created_at_utc": "2024-01-01T00:00:00Z",
    }
    key = _event_key(payload)
    assert key is not None
    assert key[-1] == "2024-01-01T00:00:00Z"


def test_event_key_uses_unknown_when_both_ids_missing() -> None:
    """decision_id and created_at_utc both absent → 'unknown' (line 543)."""
    payload = {
        "run_id": "r1",
        "case_id": "c1",
        "condition": "cond",
        "attempt": 1,
    }
    key = _event_key(payload)
    assert key is not None
    assert key[-1] == "unknown"


# --- _detect_policy_kind: empty events and mixed kinds (lines 558, 563-564) ---


def test_detect_policy_kind_raises_on_empty_events_auto() -> None:
    """Auto-detection with empty list raises ValueError (line 558)."""
    with pytest.raises(ValueError, match="unable to infer policy kind"):
        _detect_policy_kind([], requested_kind="auto")


def test_detect_policy_kind_raises_on_mixed_kinds() -> None:
    """Mixed online + budget events in auto mode raises ValueError (lines 563-564)."""
    online_event = LoadedPolicyEvent(
        payload=_minimal_online_event(),
        src="test",
        line_no=1,
    )
    budget_event = LoadedPolicyEvent(
        payload=_minimal_budget_event(),
        src="test",
        line_no=2,
    )
    with pytest.raises(ValueError, match="mixed"):
        _detect_policy_kind([online_event, budget_event], requested_kind="auto")


# --- _safe_float non-finite path (line 598) ---


def test_safe_float_returns_default_for_inf() -> None:
    """float('inf') is not finite → returns default (line 598)."""
    result = _safe_float(float("inf"), default=0.0)
    assert result == 0.0


def test_safe_float_returns_default_for_nan() -> None:
    """float('nan') is not finite → returns default."""
    result = _safe_float(float("nan"), default=5.0)
    assert result == 5.0


# --- _last_string / _last_float return-default paths (lines 607, 614) ---


def test_last_string_returns_default_when_field_missing() -> None:
    """No events have the field → returns default (line 607)."""
    result = _last_string([], field="mode", default="off")
    assert result == "off"


def test_last_float_returns_default_when_field_missing() -> None:
    """No events have the field → returns default (line 614)."""
    result = _last_float([], field="configured_alpha", default=0.9)
    assert result == pytest.approx(0.9)


# --- _infer_uniform_string: required=True raise and mixed values (lines 630-634) ---


def test_infer_uniform_string_raises_when_required_and_missing() -> None:
    """required=True with no matching field → ValueError (line 630-631)."""
    with pytest.raises(ValueError, match="missing required event field"):
        _infer_uniform_string([], field="algorithm", default="", required=True)


def test_infer_uniform_string_raises_on_mixed_values() -> None:
    """Multiple different values for the field → ValueError (lines 633-634)."""
    events = [
        LoadedPolicyEvent(payload={"algorithm": "linucb"}, src="t", line_no=1),
        LoadedPolicyEvent(payload={"algorithm": "thompson"}, src="t", line_no=2),
    ]
    with pytest.raises(ValueError, match="mixed algorithm values"):
        _infer_uniform_string(events, field="algorithm", default="", required=False)


def test_infer_uniform_string_returns_default_when_not_required_and_missing() -> None:
    """required=False with no field → returns default (line 632)."""
    result = _infer_uniform_string([], field="algorithm", default="linucb", required=False)
    assert result == "linucb"


# --- _infer_uniform_float: required raise and mixed (lines 667-671) ---


def test_infer_uniform_float_raises_when_required_and_missing() -> None:
    """required=True with no events → ValueError (lines 667-668)."""
    with pytest.raises(ValueError, match="missing required event field"):
        _infer_uniform_float(
            [], field="configured_alpha", default=0.6, required=True, description="alpha"
        )


def test_infer_uniform_float_raises_on_mixed_values() -> None:
    """Mixed float values for the field → ValueError (lines 670-671)."""
    events = [
        LoadedPolicyEvent(payload={"configured_alpha": 0.5}, src="t", line_no=1),
        LoadedPolicyEvent(payload={"configured_alpha": 0.9}, src="t", line_no=2),
    ]
    with pytest.raises(ValueError, match="mixed alpha values"):
        _infer_uniform_float(
            events, field="configured_alpha", default=0.6, required=False, description="alpha"
        )


# --- _infer_uniform_positive_int: empty and mixed (lines 684, 688) ---


def test_infer_uniform_positive_int_raises_when_no_positive_values() -> None:
    """No positive values → ValueError (line 684)."""
    events = [
        LoadedPolicyEvent(payload={"bwk_run_token_budget": 0}, src="t", line_no=1),
    ]
    with pytest.raises(ValueError, match="missing positive"):
        _infer_uniform_positive_int(events, field="bwk_run_token_budget", description="budget")


def test_infer_uniform_positive_int_raises_on_mixed_values() -> None:
    """Two different positive values → ValueError (line 688)."""
    events = [
        LoadedPolicyEvent(payload={"bwk_run_token_budget": 100}, src="t", line_no=1),
        LoadedPolicyEvent(payload={"bwk_run_token_budget": 200}, src="t", line_no=2),
    ]
    with pytest.raises(ValueError, match="mixed bwk_run_token_budget values"):
        _infer_uniform_positive_int(
            events, field="bwk_run_token_budget", description="bwk_run_token_budget"
        )


# --- _resolved_warmup_decisions: no warmup-true entries (line 699) ---


def test_resolved_warmup_decisions_returns_default_when_no_warmup_true() -> None:
    """No events with warmup_active=True → returns default (line 699)."""
    events = [
        LoadedPolicyEvent(
            payload={"warmup_active": False, "decision_index": 5}, src="t", line_no=1
        ),
    ]
    result = _resolved_warmup_decisions(events=events, requested=-1, default=20)
    assert result == 20


# --- _ensure_feature_state: dim<=0 → ["bias"] (lines 723, 740-741) ---


def test_ensure_feature_state_uses_bias_when_features_empty() -> None:
    """Empty cleaned_features → dim=0 → sets feature_names=['bias'] (lines 740-741)."""
    import numpy as np

    state = ReplayState(action_names=("a",))
    _ensure_feature_state(state=state, cleaned_features={}, l2=1.0)
    assert state.feature_names == ["bias"]
    assert "a" in state.A
    assert state.A["a"].shape == (1, 1)
    assert np.isclose(state.A["a"][0, 0], 1.0)  # l2 * eye


# --- _ensure_feature_state: matrix wrong shape → reinit (lines 780, 782) ---


def test_ensure_feature_state_reinits_wrong_shape_matrices() -> None:
    """Existing matrix with wrong shape → re-initialised to eye (lines 780-782)."""
    import numpy as np

    state = ReplayState(action_names=("a",))
    state.feature_names = ["x"]
    # Put a matrix with wrong shape (2x2 instead of 1x1)
    state.A = {"a": np.eye(2, dtype=float)}
    state.b = {"a": np.zeros((2,), dtype=float)}

    _ensure_feature_state(state=state, cleaned_features={"x": 1.0}, l2=2.0)

    # Should have been reinitialised
    assert state.A["a"].shape == (1, 1)
    assert state.b["a"].shape == (1,)


# --- _apply_recency_decay: missing matrix/vector → skip (line 794) ---


def test_apply_recency_decay_skips_action_with_missing_matrix() -> None:
    """Action without A/b matrices is skipped gracefully (line 794)."""
    state = ReplayState(action_names=("a",))
    state.feature_names = ["x"]
    # Leave A and b empty (None for action "a")
    _apply_recency_decay(state=state, action="a", decay=0.9, l2=1.0)
    # Should not raise; A and b remain empty
    assert state.A.get("a") is None


# --- _apply_reward_update: matrix/vector None → return (line 817) ---


def test_apply_reward_update_returns_early_when_no_matrix() -> None:
    """action not in A/b → returns without updating (line 817)."""
    state = ReplayState(action_names=("a",))
    state.feature_names = ["x"]
    # Don't set A/b for "a"
    _apply_reward_update(state=state, action="a", reward=1.0, feature_values={"x": 0.5})
    # updates counter should NOT be incremented (returned early)
    assert state.updates == 0


# --- _update_feature_scaling_caps: finite check (lines 828-832) ---


def test_update_feature_scaling_caps_skips_non_finite_values() -> None:
    """Non-finite cap values (inf, nan) are not stored (lines 828-832)."""
    import math

    state = ReplayState(action_names=("a",))
    _update_feature_scaling_caps(
        state=state, raw_caps={"x": float("inf"), "y": 2.0, "z": float("nan")}
    )
    assert "x" not in state.feature_scaling_caps
    assert "z" not in state.feature_scaling_caps
    assert math.isclose(state.feature_scaling_caps["y"], 2.0)


def test_update_feature_scaling_caps_skips_empty_key() -> None:
    """Empty-string keys are skipped (line 828)."""
    state = ReplayState(action_names=("a",))
    _update_feature_scaling_caps(state=state, raw_caps={"": 1.0, "valid": 3.0})
    assert "" not in state.feature_scaling_caps
    assert "valid" in state.feature_scaling_caps


# --- _budget_family_decision_logged: applied_family and strategy branches (lines 865-867) ---


def test_budget_family_decision_logged_true_for_applied_family() -> None:
    """applied_family non-empty → returns True (line 865)."""
    assert _budget_family_decision_logged({"applied_family": "tier1"}) is True


def test_budget_family_decision_logged_true_for_non_applicable_strategy() -> None:
    """family_decision_strategy not in {'', 'not_applicable'} → True (line 866-867)."""
    assert _budget_family_decision_logged({"family_decision_strategy": "linucb"}) is True


def test_budget_family_decision_logged_false_for_empty_payload() -> None:
    """All fields empty/absent → False."""
    assert _budget_family_decision_logged({}) is False


# --- main() no events (lines 144-145) ---


def test_main_returns_1_when_glob_yields_no_files(tmp_path: Path) -> None:
    """_expand_paths returns paths that don't exist → _load_policy_events gets empty list → exit 1 (lines 144-145)."""
    # Pass a literal path that doesn't exist (no glob expansion)
    output_path = tmp_path / "out.json"
    # Write an empty file so expand_paths returns it, but no records inside
    events_path = tmp_path / "empty2.jsonl"
    events_path.write_text("", encoding="utf-8")

    exit_code = bootstrap_policy_state_main(
        [
            "--policy-kind",
            "online",
            "--events",
            str(events_path),
            "--output-state",
            str(output_path),
        ]
    )
    assert exit_code == 1  # "No valid policy events found."


# --- case-label feature mask on replay ---


def _bootstrap_online_with_case_labels(tmp_path: Path, *, extra_argv: list[str]) -> dict:  # type: ignore[type-arg]
    events_path = tmp_path / "labelled_events.jsonl"
    _write_jsonl(
        events_path,
        [
            _minimal_online_event(
                decision_id="d1",
                behavior_action=ACTION_INVOKE_HELPER,
                selected_action=ACTION_INVOKE_HELPER,
                feature_values={
                    "bias": 1.0,
                    "attempt_frac": 0.5,
                    "case_difficulty_ord": 2.0,
                    "case_snapshot_dependency_ord": 1.0,
                    "case_bug_source_imported": 1.0,
                },
            )
        ],
    )
    output_path = tmp_path / f"out_{len(extra_argv)}.json"
    exit_code = bootstrap_policy_state_main(
        [
            "--policy-kind",
            "online",
            "--events",
            str(events_path),
            "--output-state",
            str(output_path),
            *extra_argv,
        ]
    )
    assert exit_code == 0
    return json.loads(output_path.read_text(encoding="utf-8"))


def test_bootstrap_online_drops_case_label_features_by_default(tmp_path: Path) -> None:
    """Replay masks the oracle case-label features, matching the runtime default."""
    payload = _bootstrap_online_with_case_labels(tmp_path, extra_argv=[])

    assert payload["feature_names"] == ["attempt_frac", "bias"]
    assert payload["bootstrap_case_label_features_kept"] is False
    assert "case_label_features_dropped" in payload["bootstrap_warning_codes"]
    for matrix in payload["A"].values():
        assert np.array(matrix).shape == (2, 2)


def test_bootstrap_online_keeps_case_label_features_on_request(tmp_path: Path) -> None:
    """--keep-case-label-features reproduces historical runs that learned the labels."""
    payload = _bootstrap_online_with_case_labels(
        tmp_path, extra_argv=["--keep-case-label-features"]
    )

    assert payload["feature_names"] == [
        "attempt_frac",
        "bias",
        "case_bug_source_imported",
        "case_difficulty_ord",
        "case_snapshot_dependency_ord",
    ]
    assert payload["bootstrap_case_label_features_kept"] is True
    assert payload["bootstrap_warning_codes"] == []


def test_bootstrap_online_without_case_labels_emits_no_mask_warning(tmp_path: Path) -> None:
    """Logs that never carried case labels replay without the mask warning."""
    events_path = tmp_path / "unlabelled_events.jsonl"
    _write_jsonl(events_path, [_minimal_online_event(behavior_action=ACTION_SKIP_HELPER)])
    output_path = tmp_path / "out_unlabelled.json"

    exit_code = bootstrap_policy_state_main(
        [
            "--policy-kind",
            "online",
            "--events",
            str(events_path),
            "--output-state",
            str(output_path),
        ]
    )

    assert exit_code == 0
    payload = json.loads(output_path.read_text(encoding="utf-8"))
    assert payload["feature_names"] == ["bias"]
    assert payload["bootstrap_warning_codes"] == []


# --- POL-14 / POL-15: percentile refusal, exclusions and target overlap ---


def _bootstrap(tmp_path: Path, events: list, *extra: str) -> tuple[int, Path]:  # type: ignore[type-arg]
    events_path = tmp_path / "events.jsonl"
    output = tmp_path / "state.json"
    _write_jsonl(events_path, events)
    code = bootstrap_policy_state_main(
        ["--events", str(events_path), "--output-state", str(output), *extra]
    )
    return code, output


def test_bootstrap_refuses_percentile_profile(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    code, output = _bootstrap(
        tmp_path, [_minimal_online_event(feature_scaling_profile="percentile_v1")]
    )
    assert code == 2
    assert not output.exists()
    assert "refusing to bootstrap percentile_v1" in capsys.readouterr().out


def test_bootstrap_excludes_cases_and_runs(tmp_path: Path) -> None:
    ids_file = tmp_path / "exclude.txt"
    ids_file.write_text("case-2\ncase-3\n", encoding="utf-8")
    events = [
        _minimal_online_event(case_id="case-1", decision_id="d1"),
        _minimal_online_event(case_id="case-2", decision_id="d2"),
        _minimal_online_event(case_id="case-3", decision_id="d3"),
        _minimal_online_event(run_id="run-old", case_id="case-4", decision_id="d4"),
    ]
    code, output = _bootstrap(
        tmp_path,
        events,
        "--exclude-case-ids",
        f"@{ids_file}",
        "--exclude-run-ids",
        "run-old",
    )
    payload = _load_json(output)
    assert code == 0
    assert payload["bootstrap_events_replayed"] == 1
    assert payload["bootstrap_excluded_events"] == 3
    assert payload["bootstrap_target_case_overlap"] == 0


def test_bootstrap_warns_when_replayed_cases_overlap_targets(tmp_path: Path, capsys) -> None:  # type: ignore[no-untyped-def]
    events = [
        _minimal_online_event(case_id="case-1", decision_id="d1"),
        _minimal_online_event(case_id="case-2", decision_id="d2"),
    ]
    code, output = _bootstrap(tmp_path, events, "--target-case-ids", "case-2,case-9")
    payload = _load_json(output)
    assert code == 0
    assert payload["bootstrap_target_case_overlap"] == 1
    assert "target_case_overlap" in payload["bootstrap_warning_codes"]
    assert "WARNING: 1 replayed case(s) are also target cases" in capsys.readouterr().out
