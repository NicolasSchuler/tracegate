"""Regression tests for adaptive policy metadata defaults and sync helpers."""

from __future__ import annotations

import argparse

import pytest

from evals.adaptive_budget_online_policy import (
    ACTION_KEEP_BUDGET,
    BUDGET_POLICY_ARCH_BWK_V1,
    AdaptiveBudgetPolicyController,
    BudgetPolicyConfig,
)
from evals.adaptive_online_policy import (
    ACTION_INVOKE_HELPER,
    ACTION_SKIP_HELPER,
    AdaptiveOnlinePolicyController,
    OnlinePolicyConfig,
)
from evals.eval_policy_budget import AdaptiveBudgetMetadataUpdate
from evals.eval_policy_online import AdaptiveOnlineMetadataUpdate
from evals.eval_recording import _build_patcher_config, _empty_attempt_metadata
from evals.run_eval import _sync_adaptive_budget_policy_metadata, _sync_adaptive_online_metadata


def _recording_args_namespace() -> argparse.Namespace:
    return argparse.Namespace(
        patcher="openai_compat",
        k=1,
        output_format="jsonl",
        force_patch_attempt=False,
        deterministic=False,
        seed=None,
        no_files_context=False,
        max_context_chars=40000,
        context_protocol="staged",
        context_stage_a_max_requests=1,
        context_stage_a_max_chars=4000,
        retry_prompt_mode="full",
        retry_context_mode="history",
        retry_history_max_chars=12000,
        prompt_budget_total_chars=32000,
        prompt_budget_retry_delta_chars=4000,
        prompt_budget_stdout_chars=8000,
        prompt_budget_snapshot_chars=12000,
        prompt_budget_evidence_chars=12000,
        prompt_budget_retry_history_chars=6000,
        context_compaction_mode="off",
        context_compaction_scope="stage_b",
        context_compaction_max_input_chars=24000,
        context_compaction_target_chars=12000,
        context_compaction_max_calls=0,
    )


def test_build_patcher_config_defaults_policy_warmups() -> None:
    config = _build_patcher_config(
        args=_recording_args_namespace(),
        patcher_name="openai_compat",
    )
    assert config["adaptive_online_warmup_decisions"] == 50
    assert config["adaptive_budget_online_warmup_decisions"] == 20


def test_empty_attempt_metadata_includes_adaptive_tuning_defaults() -> None:
    meta = _empty_attempt_metadata(attempt=1)

    assert meta["adaptive_online_configured_alpha"] == 0.0
    assert meta["adaptive_online_configured_epsilon"] == 0.0
    assert meta["adaptive_online_configured_decay"] == 1.0
    assert meta["adaptive_online_effective_epsilon"] == 0.0
    assert meta["adaptive_online_warmup_active"] is False
    assert meta["adaptive_online_action_score_biases"] == {}
    assert meta["adaptive_budget_configured_alpha"] == 0.0
    assert meta["adaptive_budget_configured_epsilon"] == 0.0
    assert meta["adaptive_budget_configured_decay"] == 1.0
    assert meta["adaptive_budget_effective_epsilon"] == 0.0
    assert meta["adaptive_budget_warmup_active"] is False
    assert meta["adaptive_online_reward_success_hint_scale"] == pytest.approx(1.0)
    assert meta["adaptive_online_reward_progress_hint_scale"] == pytest.approx(1.0)
    assert meta["adaptive_online_reward_stagnation_penalty_scale"] == pytest.approx(1.0)
    assert meta["adaptive_online_reward_helper_materialization_bonus_scale"] == pytest.approx(0.5)
    assert meta["adaptive_budget_reward_success_hint_scale"] == pytest.approx(1.0)
    assert meta["adaptive_budget_reward_progress_hint_scale"] == pytest.approx(1.0)
    assert meta["adaptive_budget_reward_stagnation_penalty_scale"] == pytest.approx(1.0)


def test_sync_adaptive_online_metadata_copies_tuning_fields(tmp_path) -> None:
    config = OnlinePolicyConfig.normalize(
        mode="on",
        algorithm="linucb",
        alpha=0.7,
        epsilon=0.05,
        decay=0.85,
        warmup_decisions=0,
        state_path=tmp_path / "policy_state_online_metadata.json",
        events_path=tmp_path / "policy_events_online_metadata.jsonl",
        random_seed=0,
    )
    controller = AdaptiveOnlinePolicyController(config)
    decision = controller.decide(
        feature_values={"bias": 1.0},
        default_action=ACTION_INVOKE_HELPER,
        available_actions=(ACTION_INVOKE_HELPER, ACTION_SKIP_HELPER),
    )
    attempt_metadata: dict[str, object] = {}
    patch_meta: dict[str, object] = {}

    _sync_adaptive_online_metadata(
        update=AdaptiveOnlineMetadataUpdate(
            attempt_metadata=attempt_metadata,
            patch_meta=patch_meta,
            mode="on",
            algo="linucb",
            decision=decision,
            eligible=True,
            override_reason="",
            feature_scaling_profile="v1",
            feature_values_raw={"bias": 1.0},
            feature_values_scaled=decision.feature_values,
            reward_lambda_tokens=0.0,
            reward_mu_latency=0.0,
            reward_token_scale=1000.0,
            reward_latency_scale=10.0,
            reward_eta_progress=0.15,
            reward_kappa_stagnation=0.1,
            reward_success_hint_scale=1.3,
            reward_progress_hint_scale=1.4,
            reward_stagnation_penalty_scale=1.5,
            reward_helper_materialization_bonus_scale=0.7,
            decay=config.decay,
        )
    )
    controller.close()

    assert attempt_metadata["adaptive_online_configured_alpha"] == pytest.approx(0.7)
    assert attempt_metadata["adaptive_online_configured_epsilon"] == pytest.approx(0.05)
    assert attempt_metadata["adaptive_online_configured_decay"] == pytest.approx(0.85)
    assert attempt_metadata["adaptive_online_effective_epsilon"] == pytest.approx(0.05)
    assert attempt_metadata["adaptive_online_warmup_active"] is False
    assert patch_meta["adaptive_online_configured_alpha"] == pytest.approx(0.7)
    assert patch_meta["adaptive_online_configured_epsilon"] == pytest.approx(0.05)
    assert patch_meta["adaptive_online_configured_decay"] == pytest.approx(0.85)
    assert patch_meta["adaptive_online_effective_epsilon"] == pytest.approx(0.05)
    assert patch_meta["adaptive_online_warmup_active"] is False
    # The controller default is -0.10/-0.05 since POL-07 (it was -0.50/-0.25).
    assert attempt_metadata["adaptive_online_action_score_biases"]["search_only"] == pytest.approx(
        -0.10
    )
    assert patch_meta["adaptive_online_action_score_biases"]["frame_only"] == pytest.approx(-0.05)
    assert attempt_metadata["adaptive_online_reward_success_hint_scale"] == pytest.approx(1.3)
    assert attempt_metadata["adaptive_online_reward_progress_hint_scale"] == pytest.approx(1.4)
    assert attempt_metadata["adaptive_online_reward_stagnation_penalty_scale"] == pytest.approx(1.5)
    assert attempt_metadata[
        "adaptive_online_reward_helper_materialization_bonus_scale"
    ] == pytest.approx(0.7)
    assert patch_meta["adaptive_online_reward_success_hint_scale"] == pytest.approx(1.3)
    assert patch_meta["adaptive_online_reward_progress_hint_scale"] == pytest.approx(1.4)
    assert patch_meta["adaptive_online_reward_stagnation_penalty_scale"] == pytest.approx(1.5)
    assert patch_meta["adaptive_online_reward_helper_materialization_bonus_scale"] == pytest.approx(
        0.7
    )


def test_sync_adaptive_budget_metadata_copies_tuning_fields(tmp_path) -> None:
    config = BudgetPolicyConfig.normalize(
        mode="on",
        algorithm="linucb",
        alpha=0.6,
        epsilon=0.03,
        decay=0.9,
        state_path=tmp_path / "policy_state_budget_metadata.json",
        events_path=tmp_path / "policy_events_budget_metadata.jsonl",
        random_seed=0,
    )
    controller = AdaptiveBudgetPolicyController(config)
    decision = controller.decide(
        feature_values={"bias": 1.0},
        default_action=ACTION_KEEP_BUDGET,
    )
    attempt_metadata: dict[str, object] = {}
    patch_meta: dict[str, object] = {}

    _sync_adaptive_budget_policy_metadata(
        update=AdaptiveBudgetMetadataUpdate(
            attempt_metadata=attempt_metadata,
            patch_meta=patch_meta,
            mode="on",
            algo="linucb",
            architecture=config.architecture,
            decision=decision,
            eligible=True,
            override_reason="",
            feature_scaling_profile="v1",
            feature_values_raw={"bias": 1.0},
            feature_values_scaled=decision.feature_values,
            tiers=[1.0, 1.5],
            tier_index_before=0,
            tier_index_after=0,
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
            max_ups_per_case=1,
            ups_used=0,
            reward_lambda_tokens=0.0,
            reward_mu_latency=0.0,
            reward_token_scale=1000.0,
            reward_latency_scale=10.0,
            reward_rho_budget_exceeded=0.5,
            reward_eta_progress=0.15,
            reward_kappa_stagnation=0.1,
            reward_success_hint_scale=1.2,
            reward_progress_hint_scale=1.25,
            reward_stagnation_penalty_scale=1.3,
            decay=config.decay,
        )
    )
    controller.close()

    assert attempt_metadata["adaptive_budget_configured_alpha"] == pytest.approx(0.6)
    assert attempt_metadata["adaptive_budget_configured_epsilon"] == pytest.approx(0.03)
    assert attempt_metadata["adaptive_budget_configured_decay"] == pytest.approx(0.9)
    assert attempt_metadata["adaptive_budget_effective_epsilon"] == pytest.approx(1.0)
    assert attempt_metadata["adaptive_budget_warmup_active"] is True
    assert patch_meta["adaptive_budget_configured_alpha"] == pytest.approx(0.6)
    assert patch_meta["adaptive_budget_configured_epsilon"] == pytest.approx(0.03)
    assert patch_meta["adaptive_budget_configured_decay"] == pytest.approx(0.9)
    assert patch_meta["adaptive_budget_effective_epsilon"] == pytest.approx(1.0)
    assert patch_meta["adaptive_budget_warmup_active"] is True
    assert attempt_metadata["adaptive_budget_reward_success_hint_scale"] == pytest.approx(1.2)
    assert attempt_metadata["adaptive_budget_reward_progress_hint_scale"] == pytest.approx(1.25)
    assert attempt_metadata["adaptive_budget_reward_stagnation_penalty_scale"] == pytest.approx(1.3)
    assert patch_meta["adaptive_budget_reward_success_hint_scale"] == pytest.approx(1.2)
    assert patch_meta["adaptive_budget_reward_progress_hint_scale"] == pytest.approx(1.25)
    assert patch_meta["adaptive_budget_reward_stagnation_penalty_scale"] == pytest.approx(1.3)


def test_sync_adaptive_budget_metadata_copies_bwk_fields(tmp_path) -> None:
    config = BudgetPolicyConfig.normalize(
        mode="on",
        architecture=BUDGET_POLICY_ARCH_BWK_V1,
        bwk_run_token_budget=100,
        state_path=tmp_path / "budget_bwk_metadata_state.json",
        events_path=tmp_path / "budget_bwk_metadata_events.jsonl",
        decay=1.0,
    )
    controller = AdaptiveBudgetPolicyController(config)
    decision = controller.decide(
        feature_values={"bias": 1.0},
        default_action=ACTION_KEEP_BUDGET,
    )
    attempt_metadata: dict[str, object] = {}
    patch_meta: dict[str, object] = {}

    _sync_adaptive_budget_policy_metadata(
        update=AdaptiveBudgetMetadataUpdate(
            attempt_metadata=attempt_metadata,
            patch_meta=patch_meta,
            mode="on",
            algo="linucb",
            architecture=config.architecture,
            decision=decision,
            eligible=True,
            override_reason="",
            feature_scaling_profile="v1",
            feature_values_raw={"bias": 1.0},
            feature_values_scaled=decision.feature_values,
            tiers=[1.0],
            tier_index_before=0,
            tier_index_after=0,
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
            max_ups_per_case=1,
            ups_used=0,
            reward_lambda_tokens=0.0,
            reward_mu_latency=0.0,
            reward_token_scale=1000.0,
            reward_latency_scale=10.0,
            reward_rho_budget_exceeded=0.5,
            reward_eta_progress=0.15,
            reward_kappa_stagnation=0.1,
            bwk_run_token_budget=100,
            bwk_remaining_token_budget_before=100,
            bwk_remaining_token_budget_after=60,
            bwk_tokens_charged=40,
            bwk_cumulative_tokens_charged=40,
            bwk_dual_variable_before=0.0,
            bwk_dual_variable_after=0.2,
            bwk_budget_exhausted=False,
        )
    )
    controller.close()

    assert attempt_metadata["adaptive_budget_bwk_run_token_budget"] == 100
    assert attempt_metadata["adaptive_budget_bwk_remaining_token_budget_after"] == 60
    assert patch_meta["adaptive_budget_bwk_tokens_charged"] == 40
    assert patch_meta["adaptive_budget_bwk_dual_variable_after"] == pytest.approx(0.2)
