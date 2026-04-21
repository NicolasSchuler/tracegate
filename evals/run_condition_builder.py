"""Helpers for assembling typed run-condition inputs."""

from __future__ import annotations

from typing import Any

from .eval_types import (
    ADAPTIVE_AUX_REWARD_COMPACTION_EFFICIENCY_WEIGHT_DEFAULT,
    ADAPTIVE_AUX_REWARD_EVIDENCE_NOVELTY_WEIGHT_DEFAULT,
    ADAPTIVE_AUX_REWARD_INFORMATIVENESS_ALIGNMENT_WEIGHT_DEFAULT,
    ADAPTIVE_AUX_REWARD_PYTEST_REDUCTION_WEIGHT_DEFAULT,
    DEFAULT_ADAPTIVE_ONLINE_ACTION_SCORE_BIASES,
    AdaptiveRuntimeConfig,
    PromptAssemblyConfig,
    RecordingConfig,
    RunConditionInputs,
    TransportConfig,
    VerificationConfig,
)


def _arg_value(args: Any, name: str, default: Any) -> Any:
    return getattr(args, name, default)


def build_run_condition_inputs(
    *,
    args: Any,
    run_id: str,
    run_metadata: dict[str, Any],
    patcher: Any | None,
    online_policy_controller: Any | None,
    budget_policy_controller: Any | None,
    openai_context_window_tokens_effective: int,
    openai_context_window_source: str,
    openai_context_window_probe_provider: str,
    openai_context_window_probe_error: str,
    transport_recovery_gate: Any | None,
    record_sink: Any | None,
) -> RunConditionInputs:
    """Build ``RunConditionInputs`` from CLI args plus resolved runtime state."""
    requested_tep_mode = str(getattr(args, "adaptive_tep_mode", "off") or "off").strip().lower()
    legacy_tep_requested = bool(getattr(args, "adaptive_tep", False)) or bool(
        getattr(args, "adaptive_tep_equilibrium", False)
    )
    if requested_tep_mode not in {"off", "legacy", "full"}:
        requested_tep_mode = "off"
    if requested_tep_mode == "off" and legacy_tep_requested:
        requested_tep_mode = "legacy"

    prompt_assembly = PromptAssemblyConfig(
        native_baseline_mode=str(getattr(args, "native_baseline_mode", "") or ""),
        context_protocol=str(args.context_protocol),
        context_stage_a_max_requests=int(args.context_stage_a_max_requests),
        context_stage_a_max_chars=int(args.context_stage_a_max_chars),
        retry_prompt_mode=str(args.retry_prompt_mode),
        retry_context_mode=str(args.retry_context_mode),
        retry_history_max_chars=int(args.retry_history_max_chars),
        prompt_retry_reflection=bool(args.prompt_retry_reflection),
        prompt_adaptive_stage_b_cot=bool(args.prompt_adaptive_stage_b_cot),
        prompt_repair_method_scaffold=bool(args.prompt_repair_method_scaffold),
        prompt_retry_plan_then_patch=bool(args.prompt_retry_plan_then_patch),
        pytest_output_tail_lines=int(args.pytest_output_tail_lines),
        snapshot_artifact_max_string_chars=int(args.snapshot_artifact_max_string_chars),
        prompt_budget_total_chars=int(args.prompt_budget_total_chars),
        prompt_budget_retry_delta_chars=int(args.prompt_budget_retry_delta_chars),
        prompt_budget_stdout_chars=int(args.prompt_budget_stdout_chars),
        prompt_budget_snapshot_chars=int(args.prompt_budget_snapshot_chars),
        prompt_budget_evidence_chars=int(args.prompt_budget_evidence_chars),
        prompt_budget_retry_history_chars=int(args.prompt_budget_retry_history_chars),
        context_compaction_mode=str(args.context_compaction_mode),
        context_compaction_scope=str(args.context_compaction_scope),
        context_compaction_max_input_chars=int(args.context_compaction_max_input_chars),
        context_compaction_target_chars=int(args.context_compaction_target_chars),
        context_compaction_max_calls=int(args.context_compaction_max_calls),
        prompt_order_mode=str(_arg_value(args, "prompt_order_mode", "current")),
        adaptive_reveal_mode=str(_arg_value(args, "adaptive_reveal_mode", "full")),
        snapshot_frame_pruning_mode=str(_arg_value(args, "snapshot_frame_pruning_mode", "off")),
    )
    adaptive_runtime = AdaptiveRuntimeConfig(
        adaptive_gate_policy=str(args.adaptive_gate_policy),
        adaptive_max_helper_calls=int(args.adaptive_max_helper_calls),
        adaptive_budget_topup_max_calls=int(args.adaptive_budget_topup_max_calls),
        adaptive_stagnation_window=int(args.adaptive_stagnation_window),
        adaptive_saturation_threshold=float(args.adaptive_saturation_threshold),
        adaptive_no_progress_token_cap=int(args.adaptive_no_progress_token_cap),
        adaptive_helper_budget_profile=str(args.adaptive_helper_budget_profile),
        adaptive_openai_helper_response_mode=str(args.adaptive_openai_helper_response_mode),
        adaptive_controller_lambda_rlm=bool(args.adaptive_controller_lambda_rlm),
        adaptive_controller_lambda_rlm_repo_dir=str(args.adaptive_controller_lambda_rlm_repo_dir),
        adaptive_controller_lambda_rlm_python=str(args.adaptive_controller_lambda_rlm_python),
        adaptive_controller_lambda_rlm_context_window_chars=int(
            args.adaptive_controller_lambda_rlm_context_window_chars
        ),
        adaptive_controller_lambda_rlm_timeout_sec=int(
            args.adaptive_controller_lambda_rlm_timeout_sec
        ),
        adaptive_informativeness_pregate=bool(args.adaptive_informativeness_pregate),
        adaptive_informativeness_pregate_threshold=float(
            args.adaptive_informativeness_pregate_threshold
        ),
        adaptive_online_policy_controller=online_policy_controller,
        adaptive_online_policy_mode=str(args.adaptive_online_policy_mode),
        adaptive_online_policy_algo=str(args.adaptive_online_policy_algo),
        adaptive_online_feature_scaling_profile=str(args.adaptive_online_feature_scaling_profile),
        adaptive_online_action_score_biases=str(
            _arg_value(
                args,
                "adaptive_online_action_score_biases",
                DEFAULT_ADAPTIVE_ONLINE_ACTION_SCORE_BIASES,
            )
        ),
        adaptive_budget_policy_controller=budget_policy_controller,
        adaptive_budget_policy_mode=str(args.adaptive_budget_policy_mode),
        adaptive_budget_policy_algo=str(args.adaptive_budget_policy_algo),
        adaptive_budget_policy_architecture=str(args.adaptive_budget_policy_architecture),
        adaptive_budget_feature_scaling_profile=str(args.adaptive_budget_feature_scaling_profile),
        adaptive_budget_policy_tiers=str(args.adaptive_budget_policy_tiers),
        adaptive_budget_policy_max_ups_per_case=int(args.adaptive_budget_policy_max_ups_per_case),
        adaptive_budget_bwk_run_token_budget=int(
            _arg_value(args, "adaptive_budget_bwk_run_token_budget", 0)
        ),
        adaptive_budget_reward_rho_budget_exceeded=float(
            args.adaptive_budget_reward_rho_budget_exceeded
        ),
        adaptive_reward_lambda_tokens=float(args.adaptive_reward_lambda_tokens),
        adaptive_reward_mu_latency=float(args.adaptive_reward_mu_latency),
        adaptive_reward_token_scale=float(args.adaptive_reward_token_scale),
        adaptive_reward_latency_scale=float(args.adaptive_reward_latency_scale),
        adaptive_reward_eta_progress=float(args.adaptive_reward_eta_progress),
        adaptive_reward_kappa_stagnation=float(args.adaptive_reward_kappa_stagnation),
        adaptive_reward_success_hint=float(args.adaptive_reward_success_hint),
        adaptive_reward_progress_scale=float(args.adaptive_reward_progress_scale),
        adaptive_reward_stagnation_scale=float(args.adaptive_reward_stagnation_scale),
        adaptive_reward_materialization_scale=float(args.adaptive_reward_materialization_scale),
        adaptive_toon_encoding=bool(args.adaptive_toon_encoding),
        enable_auxiliary_rewards=bool(getattr(args, "adaptive_aux_reward", True)),
        adaptive_aux_pytest_reduction_weight=float(
            getattr(
                args,
                "adaptive_aux_pytest_reduction_weight",
                ADAPTIVE_AUX_REWARD_PYTEST_REDUCTION_WEIGHT_DEFAULT,
            )
        ),
        adaptive_aux_evidence_novelty_weight=float(
            getattr(
                args,
                "adaptive_aux_evidence_novelty_weight",
                ADAPTIVE_AUX_REWARD_EVIDENCE_NOVELTY_WEIGHT_DEFAULT,
            )
        ),
        adaptive_aux_compaction_efficiency_weight=float(
            getattr(
                args,
                "adaptive_aux_compaction_efficiency_weight",
                ADAPTIVE_AUX_REWARD_COMPACTION_EFFICIENCY_WEIGHT_DEFAULT,
            )
        ),
        adaptive_aux_informativeness_alignment_weight=float(
            getattr(
                args,
                "adaptive_aux_informativeness_alignment_weight",
                ADAPTIVE_AUX_REWARD_INFORMATIVENESS_ALIGNMENT_WEIGHT_DEFAULT,
            )
        ),
        adaptive_tep_mode=requested_tep_mode,
        adaptive_tep_state_path=str(getattr(args, "adaptive_tep_state_path", "")),
        adaptive_tep_init_state_path=str(getattr(args, "adaptive_tep_init_state_path", "")),
        adaptive_tep_beta_init=float(getattr(args, "adaptive_tep_beta_init", 1.0)),
        adaptive_tep_beta_decay=float(getattr(args, "adaptive_tep_beta_decay", 0.9)),
        adaptive_tep_equilibrium_max_iters=int(
            getattr(args, "adaptive_tep_full_equilibrium_max_iters", 20)
        ),
        adaptive_tep_equilibrium_window=int(
            getattr(args, "adaptive_tep_full_equilibrium_window", 3)
        ),
        adaptive_tep_equilibrium_variance_threshold=float(
            getattr(args, "adaptive_tep_full_equilibrium_variance_threshold", 0.5)
        ),
        adaptive_tep_validation_buffer_size=int(
            getattr(args, "adaptive_tep_validation_buffer_size", 4)
        ),
        enable_tep=bool(getattr(args, "adaptive_tep", False))
        or bool(getattr(args, "adaptive_tep_equilibrium", False)),
        tep_nudge_escalation=True,
        tep_equilibrium_check=bool(getattr(args, "adaptive_tep_equilibrium", False)),
        tep_equilibrium_max_iterations=int(
            getattr(args, "adaptive_tep_equilibrium_max_iterations", 2)
        ),
        tep_equilibrium_similarity_threshold=float(
            getattr(args, "adaptive_tep_equilibrium_similarity_threshold", 0.85)
        ),
    )
    verification = VerificationConfig(
        syntax_preflight_enabled=bool(args.syntax_preflight_enabled),
        syntax_preflight_max_cycles=int(args.syntax_preflight_max_cycles),
        syntax_preflight_fallthrough=bool(args.syntax_preflight_fallthrough),
        patch_sanitize_enabled=bool(args.patch_sanitize_enabled),
        patch_verifier_scope=str(args.patch_verifier_scope),
        patch_verifier_min_score=int(args.patch_verifier_min_score),
        patch_verifier_infer_test_source_scope=bool(args.patch_verifier_infer_test_source_scope),
    )
    transport = TransportConfig(
        openai_context_window_tokens=int(openai_context_window_tokens_effective),
        openai_context_window_source=openai_context_window_source,
        openai_context_window_probe_provider=openai_context_window_probe_provider,
        openai_context_window_probe_error=openai_context_window_probe_error,
        openai_response_mode_autoswitch=str(args.openai_response_mode_autoswitch),
        transport_recovery_gate=transport_recovery_gate,
        openai_transport_recovery=bool(args.openai_transport_recovery),
    )
    recording = RecordingConfig(
        run_metadata=run_metadata,
        record_prompt=str(args.record_prompt),
        record_evidence=str(args.record_evidence),
        deterministic=bool(args.deterministic),
        seed=args.seed if args.deterministic else None,
        record_sink=record_sink,
    )
    return RunConditionInputs.from_components(
        k=int(args.k),
        output_format=str(args.output_format),
        run_id=run_id,
        include_files_context=not bool(args.no_files_context),
        force_patch_attempt=bool(args.force_patch_attempt),
        patcher=patcher,
        max_context_chars=int(args.max_context_chars),
        case_fail_fast_on_request_budget_exceeded=bool(
            args.case_fail_fast_on_request_budget_exceeded
        ),
        prompt_assembly=prompt_assembly,
        adaptive_runtime=adaptive_runtime,
        verification=verification,
        transport=transport,
        recording=recording,
    )
