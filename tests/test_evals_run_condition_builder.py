"""Tests for evals/run_condition_builder.py coverage gaps.

Covers:
- Line 46: unknown adaptive_tep_mode normalised to "off"
- Line 47-48: legacy_tep_requested upgrades "off" → "legacy"
- The full build_run_condition_inputs happy path
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

from evals.run_condition_builder import build_run_condition_inputs


def _minimal_args(**overrides: Any) -> SimpleNamespace:
    """Build a minimal SimpleNamespace that satisfies build_run_condition_inputs."""
    defaults: dict[str, Any] = {
        # PromptAssemblyConfig
        "native_baseline_mode": "",
        "context_protocol": "monolithic",
        "context_stage_a_max_requests": 6,
        "context_stage_a_max_chars": 120000,
        "retry_prompt_mode": "auto",
        "retry_context_mode": "history",
        "retry_history_max_chars": 48000,
        "prompt_retry_reflection": False,
        "prompt_adaptive_stage_b_cot": False,
        "prompt_repair_method_scaffold": False,
        "prompt_retry_plan_then_patch": False,
        "pytest_output_tail_lines": 4000,
        "snapshot_artifact_max_string_chars": 16000,
        "prompt_budget_total_chars": 180000,
        "prompt_budget_retry_delta_chars": 16000,
        "prompt_budget_stdout_chars": 32000,
        "prompt_budget_snapshot_chars": 32000,
        "prompt_budget_evidence_chars": 64000,
        "prompt_budget_retry_history_chars": 12000,
        "context_compaction_mode": "auto_overflow",
        "context_compaction_scope": "snapshot,pytest,files",
        "context_compaction_max_input_chars": 120000,
        "context_compaction_target_chars": 18000,
        "context_compaction_max_calls": 1,
        # AdaptiveRuntimeConfig
        "adaptive_gate_policy": "balanced",
        "adaptive_max_helper_calls": 1,
        "adaptive_budget_topup_max_calls": 1,
        "adaptive_stagnation_window": 1,
        "adaptive_saturation_threshold": 0.75,
        "adaptive_no_progress_token_cap": 150000,
        "adaptive_helper_budget_profile": "default",
        "adaptive_openai_helper_response_mode": "inherit",
        "adaptive_controller_lambda_rlm": False,
        "adaptive_controller_lambda_rlm_repo_dir": "",
        "adaptive_controller_lambda_rlm_python": "",
        "adaptive_controller_lambda_rlm_context_window_chars": 100000,
        "adaptive_controller_lambda_rlm_timeout_sec": 120,
        "adaptive_informativeness_pregate": False,
        "adaptive_informativeness_pregate_threshold": 0.7,
        "adaptive_online_policy_mode": "off",
        "adaptive_online_policy_algo": "linucb",
        "adaptive_online_feature_scaling_profile": "v1",
        "adaptive_online_action_score_biases": "search_only=-1.00,frame_only=-0.50",
        "adaptive_budget_policy_mode": "off",
        "adaptive_budget_policy_algo": "linucb",
        "adaptive_budget_policy_architecture": "tier_binary",
        "adaptive_budget_feature_scaling_profile": "v1",
        "adaptive_budget_policy_tiers": "1.0,1.1,1.2,1.35,1.5",
        "adaptive_budget_policy_max_ups_per_case": 1,
        "adaptive_reward_lambda_tokens": 0.08,
        "adaptive_reward_mu_latency": 0.05,
        "adaptive_reward_token_scale": 1000.0,
        "adaptive_reward_latency_scale": 10.0,
        "adaptive_reward_eta_progress": 0.15,
        "adaptive_reward_kappa_stagnation": 0.1,
        "adaptive_reward_success_hint": 1.0,
        "adaptive_reward_progress_scale": 1.0,
        "adaptive_reward_stagnation_scale": 1.0,
        "adaptive_reward_materialization_scale": 0.5,
        "adaptive_budget_reward_rho_budget_exceeded": 0.5,
        "adaptive_toon_encoding": False,
        "adaptive_tep": False,
        "adaptive_tep_equilibrium": False,
        "adaptive_tep_mode": "off",
        # VerificationConfig
        "syntax_preflight_enabled": False,
        "syntax_preflight_max_cycles": 1,
        "syntax_preflight_fallthrough": False,
        "patch_sanitize_enabled": False,
        "patch_verifier_scope": "none",
        "patch_verifier_min_score": 0,
        "patch_verifier_infer_test_source_scope": False,
        # TransportConfig
        "openai_response_mode_autoswitch": "off",
        "openai_transport_recovery": False,
        # RecordingConfig
        "record_prompt": "off",
        "record_evidence": "off",
        "deterministic": False,
        "seed": None,
        # RunConditionInputs.from_components
        "k": 1,
        "output_format": "jsonl",
        "no_files_context": False,
        "force_patch_attempt": False,
        "max_context_chars": 120000,
        "case_fail_fast_on_request_budget_exceeded": False,
    }
    defaults.update(overrides)
    return SimpleNamespace(**defaults)


class TestBuildRunConditionInputsTepModeNormalisation:
    """build_run_condition_inputs normalises unknown tep modes to 'off'."""

    def test_unknown_tep_mode_normalised_to_off(self) -> None:
        """Line 46: unknown adaptive_tep_mode → falls through to 'off'."""
        args = _minimal_args(adaptive_tep_mode="unknown_mode_xyz")
        result = build_run_condition_inputs(
            args=args,
            run_id="test-run",
            run_metadata={},
            patcher=None,
            online_policy_controller=None,
            budget_policy_controller=None,
            openai_context_window_tokens_effective=128000,
            openai_context_window_source="default",
            openai_context_window_probe_provider="",
            openai_context_window_probe_error="",
            transport_recovery_gate=None,
            record_sink=None,
        )
        assert result.adaptive_runtime.adaptive_tep_mode == "off"

    def test_legacy_tep_flag_upgrades_off_to_legacy(self) -> None:
        """Lines 47-48: adaptive_tep=True with mode='off' → mode becomes 'legacy'."""
        args = _minimal_args(adaptive_tep=True, adaptive_tep_mode="off")
        result = build_run_condition_inputs(
            args=args,
            run_id="test-run",
            run_metadata={},
            patcher=None,
            online_policy_controller=None,
            budget_policy_controller=None,
            openai_context_window_tokens_effective=128000,
            openai_context_window_source="default",
            openai_context_window_probe_provider="",
            openai_context_window_probe_error="",
            transport_recovery_gate=None,
            record_sink=None,
        )
        assert result.adaptive_runtime.adaptive_tep_mode == "legacy"

    def test_explicit_full_mode_preserved(self) -> None:
        """Valid 'full' tep mode is preserved unchanged."""
        args = _minimal_args(adaptive_tep_mode="full")
        result = build_run_condition_inputs(
            args=args,
            run_id="test-run",
            run_metadata={},
            patcher=None,
            online_policy_controller=None,
            budget_policy_controller=None,
            openai_context_window_tokens_effective=128000,
            openai_context_window_source="default",
            openai_context_window_probe_provider="",
            openai_context_window_probe_error="",
            transport_recovery_gate=None,
            record_sink=None,
        )
        assert result.adaptive_runtime.adaptive_tep_mode == "full"
