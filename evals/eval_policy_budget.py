"""Adaptive budget-policy metadata and finalization helpers."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from . import linucb_core as core
from .adaptive_budget_online_policy import (
    ACTION_INCREASE_BUDGET,
    BUDGET_POLICY_ACTIONS,
    BUDGET_POLICY_ARCH_TIER_BINARY,
    AdaptiveBudgetPolicyController,
    BudgetPolicyDecision,
)
from .eval_policy_features import _adaptive_budget_local_reward_components
from .eval_policy_online import _finalize_adaptive_online_attempt
from .eval_types import (
    ADAPTIVE_AUX_REWARD_EVIDENCE_NOVELTY_WEIGHT_DEFAULT,
    ADAPTIVE_AUX_REWARD_PYTEST_REDUCTION_WEIGHT_DEFAULT,
    ADAPTIVE_BUDGET_REWARD_SCOPE,
    ADAPTIVE_LOCAL_REWARD_PROGRESS_HINT_SCALE,
    ADAPTIVE_LOCAL_REWARD_STAGNATION_PENALTY_SCALE,
    ADAPTIVE_LOCAL_REWARD_SUCCESS_HINT,
    EMPTY_ADAPTIVE_POLICY_LOCAL_ACCOUNTING,
    AdaptivePolicyLocalAccounting,
    PendingAdaptiveOnlineCredit,
)


def _nonnegative_int(value: Any, *, default: int = 0) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return max(0, int(default))


def _adaptive_budget_behavior_fields(
    *,
    decision: BudgetPolicyDecision | None,
) -> tuple[str, float]:
    if decision is None:
        return "none", 1.0
    selected_action = str(decision.selected_action)
    applied_action = str(decision.applied_action)
    behavior_action = applied_action if applied_action in BUDGET_POLICY_ACTIONS else "none"
    # Shadow mode applies the deterministic baseline, so only "on" decisions carry propensities
    # (POL-03).
    behavior_propensity = core.behavior_propensity(
        mode=decision.mode,
        selected=selected_action,
        behavior=behavior_action,
        propensity=decision.propensity,
    )
    sampled = str(decision.mode) == "on" and selected_action == behavior_action
    if (
        sampled
        and str(decision.applied_family)
        and str(decision.selected_family) == str(decision.applied_family)
    ):
        behavior_propensity = min(
            1.0,
            max(1e-9, behavior_propensity * float(decision.family_propensity)),
        )
    return behavior_action, behavior_propensity


@dataclass(frozen=True)
class AdaptiveBudgetMetadataUpdate:
    """Typed payload for adaptive budget metadata synchronization."""

    attempt_metadata: dict[str, Any]
    patch_meta: dict[str, Any]
    mode: str
    algo: str
    architecture: str
    decision: BudgetPolicyDecision | None
    eligible: bool
    override_reason: str
    feature_scaling_profile: str
    feature_values_raw: dict[str, float]
    feature_values_scaled: dict[str, float]
    tiers: list[float]
    tier_index_before: int
    tier_index_after: int
    selected_multiplier: float
    applied_multiplier: float
    default_multiplier: float
    default_controls: Mapping[str, Any]
    selected_controls: Mapping[str, Any]
    applied_controls: Mapping[str, Any]
    selected_controls_changed: Mapping[str, bool]
    applied_controls_changed: Mapping[str, bool]
    partial_override: bool
    partial_override_errors: Mapping[str, str]
    max_ups_per_case: int
    ups_used: int
    reward_lambda_tokens: float
    reward_mu_latency: float
    reward_token_scale: float
    reward_latency_scale: float
    reward_rho_budget_exceeded: float
    reward_eta_progress: float
    reward_kappa_stagnation: float
    reward_success_hint_scale: float = ADAPTIVE_LOCAL_REWARD_SUCCESS_HINT
    reward_progress_hint_scale: float = ADAPTIVE_LOCAL_REWARD_PROGRESS_HINT_SCALE
    reward_stagnation_penalty_scale: float = ADAPTIVE_LOCAL_REWARD_STAGNATION_PENALTY_SCALE
    reward_scope: str = ADAPTIVE_BUDGET_REWARD_SCOPE
    reward: float | None = None
    reward_success_hint: float = 0.0
    reward_progress_hint: float = 0.0
    reward_progress_score: float = 0.0
    reward_token_penalty: float = 0.0
    reward_latency_penalty: float = 0.0
    reward_stagnation_penalty: float = 0.0
    reward_budget_exceeded_penalty: float = 0.0
    request_budget_exceeded: bool = False
    local_accounting: AdaptivePolicyLocalAccounting = EMPTY_ADAPTIVE_POLICY_LOCAL_ACCOUNTING
    updated: bool = False
    update_reason: str = ""
    event_logged: bool = False
    decay: float = 1.0
    bwk_run_token_budget: int = 0
    bwk_remaining_token_budget_before: int = 0
    bwk_remaining_token_budget_after: int = 0
    bwk_tokens_charged: int = 0
    bwk_cumulative_tokens_charged: int = 0
    bwk_dual_variable_before: float = 0.0
    bwk_dual_variable_after: float = 0.0
    bwk_budget_exhausted: bool = False


def _sync_adaptive_budget_policy_metadata(
    *,
    update: AdaptiveBudgetMetadataUpdate | None = None,
    **legacy_kwargs: Any,
) -> None:
    if update is None:
        update = AdaptiveBudgetMetadataUpdate(**legacy_kwargs)
    decision = update.decision
    selected_action = decision.selected_action if decision is not None else "none"
    applied_action = decision.applied_action if decision is not None else "none"
    decision_id = decision.decision_id if decision is not None else ""
    decision_strategy = decision.decision_strategy if decision is not None else ""
    propensity = float(decision.propensity) if decision is not None else 1.0
    feature_scaling_caps = dict(decision.feature_scaling_caps) if decision is not None else {}
    feature_scaling_history_sizes = (
        dict(decision.feature_scaling_history_sizes) if decision is not None else {}
    )
    feature_scaling_fallback_to_v1 = (
        bool(decision.feature_scaling_fallback_to_v1) if decision is not None else False
    )
    selected_family = decision.selected_family if decision is not None else ""
    applied_family = decision.applied_family if decision is not None else ""
    available_families = list(decision.available_families) if decision is not None else []
    unavailable_family_reasons = (
        dict(decision.unavailable_family_reasons) if decision is not None else {}
    )
    family_propensity = float(decision.family_propensity) if decision is not None else 1.0
    family_decision_strategy = (
        str(decision.family_decision_strategy) if decision is not None else ""
    )
    family_feature_scaling_caps = (
        dict(decision.family_feature_scaling_caps) if decision is not None else {}
    )
    family_feature_scaling_history_sizes = (
        dict(decision.family_feature_scaling_history_sizes) if decision is not None else {}
    )
    family_feature_scaling_fallback_to_v1 = (
        bool(decision.family_feature_scaling_fallback_to_v1) if decision is not None else False
    )
    configured_alpha = float(decision.configured_alpha) if decision is not None else 0.0
    configured_epsilon = float(decision.configured_epsilon) if decision is not None else 0.0
    configured_decay = (
        float(decision.configured_decay) if decision is not None else float(update.decay)
    )
    effective_epsilon = float(decision.effective_epsilon) if decision is not None else 0.0
    warmup_active = bool(decision.warmup_active) if decision is not None else False
    behavior_action, behavior_propensity = _adaptive_budget_behavior_fields(decision=decision)
    fields = {
        "adaptive_budget_policy_mode": str(update.mode),
        "adaptive_budget_policy_algo": str(update.algo),
        "adaptive_budget_policy_architecture": str(update.architecture),
        "adaptive_budget_decision_id": str(decision_id),
        "adaptive_budget_decision_strategy": str(decision_strategy),
        "adaptive_budget_action_selected": str(selected_action),
        "adaptive_budget_action_applied": str(applied_action),
        "adaptive_budget_family_selected": str(selected_family),
        "adaptive_budget_family_applied": str(applied_family),
        "adaptive_budget_available_families": list(available_families),
        "adaptive_budget_unavailable_family_reasons": dict(unavailable_family_reasons),
        "adaptive_budget_family_propensity": float(family_propensity),
        "adaptive_budget_family_decision_strategy": str(family_decision_strategy),
        "adaptive_budget_propensity": float(propensity),
        "adaptive_budget_behavior_action": behavior_action,
        "adaptive_budget_behavior_propensity": float(behavior_propensity),
        "adaptive_budget_override_reason": str(update.override_reason),
        "adaptive_budget_feature_scaling_profile": str(update.feature_scaling_profile),
        "adaptive_budget_feature_scaling_caps": feature_scaling_caps,
        "adaptive_budget_feature_scaling_history_sizes": feature_scaling_history_sizes,
        "adaptive_budget_feature_scaling_fallback_to_v1": bool(feature_scaling_fallback_to_v1),
        "adaptive_budget_family_feature_scaling_caps": family_feature_scaling_caps,
        "adaptive_budget_family_feature_scaling_history_sizes": family_feature_scaling_history_sizes,
        "adaptive_budget_family_feature_scaling_fallback_to_v1": bool(
            family_feature_scaling_fallback_to_v1
        ),
        "adaptive_budget_feature_vector_raw": dict(update.feature_values_raw),
        "adaptive_budget_feature_vector": dict(update.feature_values_scaled),
        "adaptive_budget_configured_alpha": float(configured_alpha),
        "adaptive_budget_configured_epsilon": float(configured_epsilon),
        "adaptive_budget_configured_decay": float(configured_decay),
        "adaptive_budget_effective_epsilon": float(effective_epsilon),
        "adaptive_budget_warmup_active": bool(warmup_active),
        "adaptive_budget_decay": float(update.decay),
        "adaptive_budget_eligible": bool(update.eligible),
        "adaptive_budget_tiers": [float(value) for value in update.tiers],
        "adaptive_budget_tier_index_before": int(update.tier_index_before),
        "adaptive_budget_tier_index_after": int(update.tier_index_after),
        "adaptive_budget_selected_multiplier": float(update.selected_multiplier),
        "adaptive_budget_applied_multiplier": float(update.applied_multiplier),
        "adaptive_budget_default_multiplier": float(update.default_multiplier),
        "adaptive_budget_default_control_values": dict(update.default_controls),
        "adaptive_budget_selected_control_values": dict(update.selected_controls),
        "adaptive_budget_applied_control_values": dict(update.applied_controls),
        "adaptive_budget_selected_controls_changed": {
            str(key): bool(value) for key, value in update.selected_controls_changed.items()
        },
        "adaptive_budget_applied_controls_changed": {
            str(key): bool(value) for key, value in update.applied_controls_changed.items()
        },
        "adaptive_budget_partial_override": bool(update.partial_override),
        "adaptive_budget_partial_override_errors": {
            str(key): str(value) for key, value in update.partial_override_errors.items()
        },
        "adaptive_budget_effective_helper_max_calls": int(
            update.applied_controls.get("helper_max_calls", 0) or 0
        ),
        "adaptive_budget_effective_stage_a_max_requests": int(
            update.applied_controls.get("stage_a_max_requests", 1) or 1
        ),
        "adaptive_budget_effective_stage_a_max_chars": int(
            update.applied_controls.get("stage_a_max_chars", 1) or 1
        ),
        "adaptive_budget_effective_retry_history_max_chars": int(
            update.applied_controls.get("retry_history_max_chars", 0) or 0
        ),
        "adaptive_budget_effective_context_compaction_max_calls": int(
            update.applied_controls.get("context_compaction_max_calls", 0) or 0
        ),
        "adaptive_budget_effective_openai_max_tokens": int(
            update.applied_controls.get("openai_max_tokens", 0) or 0
        ),
        "adaptive_budget_effective_response_mode": str(
            update.applied_controls.get("response_mode") or ""
        ),
        "adaptive_budget_max_ups_per_case": int(update.max_ups_per_case),
        "adaptive_budget_ups_used": int(update.ups_used),
        "adaptive_budget_reward_lambda_tokens": float(update.reward_lambda_tokens),
        "adaptive_budget_reward_mu_latency": float(update.reward_mu_latency),
        "adaptive_budget_reward_token_scale": float(update.reward_token_scale),
        "adaptive_budget_reward_latency_scale": float(update.reward_latency_scale),
        "adaptive_budget_reward_rho_budget_exceeded": float(update.reward_rho_budget_exceeded),
        "adaptive_budget_reward_eta_progress": float(update.reward_eta_progress),
        "adaptive_budget_reward_kappa_stagnation": float(update.reward_kappa_stagnation),
        "adaptive_budget_reward_success_hint_scale": float(update.reward_success_hint_scale),
        "adaptive_budget_reward_progress_hint_scale": float(update.reward_progress_hint_scale),
        "adaptive_budget_reward_stagnation_penalty_scale": float(
            update.reward_stagnation_penalty_scale
        ),
        "adaptive_budget_reward_scope": str(update.reward_scope),
        "adaptive_budget_reward": update.reward,
        "adaptive_budget_reward_success_hint": float(update.reward_success_hint),
        "adaptive_budget_reward_progress_hint": float(update.reward_progress_hint),
        "adaptive_budget_reward_progress_score": float(update.reward_progress_score),
        "adaptive_budget_reward_token_penalty": float(update.reward_token_penalty),
        "adaptive_budget_reward_latency_penalty": float(update.reward_latency_penalty),
        "adaptive_budget_reward_stagnation_penalty": float(update.reward_stagnation_penalty),
        "adaptive_budget_reward_budget_exceeded_penalty": float(
            update.reward_budget_exceeded_penalty
        ),
        "adaptive_budget_request_budget_exceeded": bool(update.request_budget_exceeded),
        "adaptive_budget_local_llm_call_count": int(update.local_accounting.llm_call_count),
        "adaptive_budget_local_tokens_prompt": int(update.local_accounting.tokens_prompt),
        "adaptive_budget_local_tokens_completion": int(update.local_accounting.tokens_completion),
        "adaptive_budget_local_tokens_total": int(update.local_accounting.tokens_total),
        "adaptive_budget_local_wall_sec": float(update.local_accounting.wall_sec),
        "adaptive_budget_updated": bool(update.updated),
        "adaptive_budget_update_reason": str(update.update_reason),
        "adaptive_budget_event_logged": bool(update.event_logged),
        "adaptive_budget_bwk_run_token_budget": int(update.bwk_run_token_budget),
        "adaptive_budget_bwk_remaining_token_budget_before": int(
            update.bwk_remaining_token_budget_before
        ),
        "adaptive_budget_bwk_remaining_token_budget_after": int(
            update.bwk_remaining_token_budget_after
        ),
        "adaptive_budget_bwk_tokens_charged": int(update.bwk_tokens_charged),
        "adaptive_budget_bwk_cumulative_tokens_charged": int(update.bwk_cumulative_tokens_charged),
        "adaptive_budget_bwk_dual_variable_before": float(update.bwk_dual_variable_before),
        "adaptive_budget_bwk_dual_variable_after": float(update.bwk_dual_variable_after),
        "adaptive_budget_bwk_budget_exhausted": bool(update.bwk_budget_exhausted),
    }
    update.attempt_metadata.update(fields)
    update.patch_meta.update(fields)


def _escalation_cost_share(*, applied_multiplier: float, default_multiplier: float) -> float:
    """Share of an escalated attempt's Stage-B cost that the escalation added (POL-08).

    The proposal is paid under either arm, so ``keep_budget`` costs nothing extra and
    ``increase_budget`` is charged ``1 - default / applied`` of the attempt's tokens and
    time, treating cost as proportional to the budget multiplier. It used to be charged the
    whole attempt: about 0.40 of a 1.0 success reward at ``lambda_tokens=0.08`` and 5k
    tokens.
    """
    base = max(0.0, float(default_multiplier))
    applied = float(applied_multiplier)
    if base <= 0.0 or applied <= base:
        return 0.0
    return 1.0 - base / applied


def _escalation_accounting(
    accounting: AdaptivePolicyLocalAccounting, *, share: float
) -> AdaptivePolicyLocalAccounting:
    """*accounting* scaled to the escalation's share of the cost."""
    return AdaptivePolicyLocalAccounting(
        llm_call_count=accounting.llm_call_count,
        tokens_prompt=round(accounting.tokens_prompt * share),
        tokens_completion=round(accounting.tokens_completion * share),
        tokens_total=round(accounting.tokens_total * share),
        wall_sec=float(accounting.wall_sec) * share,
    )


def _finalize_adaptive_budget_policy_attempt(
    *,
    note: str,
    success: bool,
    attempt_wall_sec: float,
    is_adaptive_condition: bool,
    budget_policy_decision: BudgetPolicyDecision | None,
    adaptive_budget_policy_controller: AdaptiveBudgetPolicyController | None,
    run_id: str,
    case_id: str,
    condition_value: str,
    attempt: int,
    attempt_metadata: dict[str, Any],
    patch_meta: dict[str, Any],
    budget_policy_mode_for_attempt: str,
    budget_policy_algo_for_attempt: str,
    budget_policy_eligible: bool,
    budget_policy_override_reason: str,
    budget_policy_feature_scaling_profile: str,
    budget_policy_features_raw: dict[str, float],
    budget_policy_features_scaled: dict[str, float],
    normalized_adaptive_budget_policy_mode: str,
    adaptive_progress_made: bool,
    adaptive_progress_score: float,
    adaptive_no_progress_streak_current: int,
    reward_lambda_tokens: float,
    reward_mu_latency: float,
    reward_token_scale: float,
    reward_latency_scale: float,
    reward_eta_progress: float,
    reward_kappa_stagnation: float,
    reward_rho_budget_exceeded: float,
    reward_success_hint_scale: float = ADAPTIVE_LOCAL_REWARD_SUCCESS_HINT,
    reward_progress_hint_scale: float = ADAPTIVE_LOCAL_REWARD_PROGRESS_HINT_SCALE,
    reward_stagnation_penalty_scale: float = ADAPTIVE_LOCAL_REWARD_STAGNATION_PENALTY_SCALE,
    local_accounting: AdaptivePolicyLocalAccounting = EMPTY_ADAPTIVE_POLICY_LOCAL_ACCOUNTING,
    tiers: list[float] | None = None,
    tier_index_before: int = 0,
    tier_index_after: int = 0,
    selected_multiplier: float = 1.0,
    applied_multiplier: float = 1.0,
    default_multiplier: float = 1.0,
    default_controls: Mapping[str, Any] | None = None,
    selected_controls: Mapping[str, Any] | None = None,
    applied_controls: Mapping[str, Any] | None = None,
    selected_controls_changed: Mapping[str, bool] | None = None,
    applied_controls_changed: Mapping[str, bool] | None = None,
    partial_override: bool = False,
    partial_override_errors: Mapping[str, str] | None = None,
    max_ups_per_case: int = 0,
    ups_used: int = 0,
    override_reason: str = "",
    enable_auxiliary_rewards: bool = False,
    aux_pytest_fail_reduction: float = 0.0,
    aux_evidence_novelty_ratio: float = 0.0,
    aux_pytest_reduction_weight: float = ADAPTIVE_AUX_REWARD_PYTEST_REDUCTION_WEIGHT_DEFAULT,
    aux_evidence_novelty_weight: float = ADAPTIVE_AUX_REWARD_EVIDENCE_NOVELTY_WEIGHT_DEFAULT,
) -> None:
    default_controls = default_controls or {}
    selected_controls = selected_controls or {}
    applied_controls = applied_controls or {}
    selected_controls_changed = selected_controls_changed or {}
    applied_controls_changed = applied_controls_changed or {}
    partial_override_errors = partial_override_errors or {}
    tiers = tiers or [1.0]
    effective_override_reason = (
        budget_policy_override_reason
        if budget_policy_override_reason
        else str(override_reason or "")
    )
    failure_error = str(
        attempt_metadata.get(
            "patcher_request_failure_error",
            patch_meta.get("patcher_request_failure_error", patch_meta.get("error")),
        )
        or ""
    ).strip()
    request_budget_exceeded = failure_error == "request_budget_exceeded"
    behavior_action, _ = _adaptive_budget_behavior_fields(decision=budget_policy_decision)
    charged_local_accounting = (
        _escalation_accounting(
            local_accounting,
            share=_escalation_cost_share(
                applied_multiplier=applied_multiplier, default_multiplier=default_multiplier
            ),
        )
        if behavior_action == ACTION_INCREASE_BUDGET
        else AdaptivePolicyLocalAccounting()
    )
    (
        reward,
        reward_success_hint,
        reward_progress_hint,
        reward_token_penalty,
        reward_latency_penalty,
        reward_stagnation_penalty,
        reward_budget_exceeded_penalty,
        _budget_aux_total,
        _budget_aux_pytest,
        _budget_aux_novelty,
    ) = _adaptive_budget_local_reward_components(
        success=success,
        local_tokens_total=charged_local_accounting.tokens_total,
        local_wall_sec=charged_local_accounting.wall_sec,
        adaptive_progress_score=adaptive_progress_score,
        adaptive_no_progress_streak=adaptive_no_progress_streak_current,
        lambda_tokens=reward_lambda_tokens,
        mu_latency=reward_mu_latency,
        token_scale=reward_token_scale,
        latency_scale=reward_latency_scale,
        eta_progress=reward_eta_progress,
        kappa_stagnation=reward_kappa_stagnation,
        rho_budget_exceeded=reward_rho_budget_exceeded,
        request_budget_exceeded=request_budget_exceeded,
        success_hint_scale=reward_success_hint_scale,
        progress_hint_scale=reward_progress_hint_scale,
        stagnation_penalty_scale=reward_stagnation_penalty_scale,
        enable_auxiliary_rewards=enable_auxiliary_rewards,
        aux_pytest_fail_reduction=aux_pytest_fail_reduction,
        aux_evidence_novelty_ratio=aux_evidence_novelty_ratio,
        aux_pytest_reduction_weight=aux_pytest_reduction_weight,
        aux_evidence_novelty_weight=aux_evidence_novelty_weight,
    )
    updated = False
    event_logged = False
    update_reason = "policy_off"
    policy_event = None
    if (
        is_adaptive_condition
        and budget_policy_decision is not None
        and adaptive_budget_policy_controller is not None
    ):
        try:
            policy_event = adaptive_budget_policy_controller.record_outcome(
                decision=budget_policy_decision,
                run_id=run_id,
                case_id=case_id,
                condition=condition_value,
                attempt=attempt,
                note=note,
                reward_scope=ADAPTIVE_BUDGET_REWARD_SCOPE,
                reward_success_hint_scale=reward_success_hint_scale,
                reward_progress_hint_scale=reward_progress_hint_scale,
                reward_stagnation_penalty_scale=reward_stagnation_penalty_scale,
                reward=reward,
                reward_success_hint=reward_success_hint,
                reward_progress_hint=reward_progress_hint,
                reward_progress_score=max(0.0, min(1.0, float(adaptive_progress_score))),
                reward_token_penalty=reward_token_penalty,
                reward_latency_penalty=reward_latency_penalty,
                reward_stagnation_penalty=reward_stagnation_penalty,
                reward_budget_exceeded_penalty=reward_budget_exceeded_penalty,
                request_budget_exceeded=request_budget_exceeded,
                success=success,
                local_llm_call_count=local_accounting.llm_call_count,
                local_tokens_prompt=local_accounting.tokens_prompt,
                local_tokens_completion=local_accounting.tokens_completion,
                local_tokens_total=local_accounting.tokens_total,
                local_wall_sec=local_accounting.wall_sec,
                selected_multiplier=selected_multiplier,
                applied_multiplier=applied_multiplier,
                default_multiplier=default_multiplier,
                default_controls=default_controls,
                selected_controls=selected_controls,
                applied_controls=applied_controls,
                selected_controls_changed=selected_controls_changed,
                applied_controls_changed=applied_controls_changed,
                partial_override=partial_override,
                partial_override_errors=partial_override_errors,
                tier_index_before=tier_index_before,
                tier_index_after=tier_index_after,
                override_reason=effective_override_reason,
            )
            updated = bool(policy_event.model_updated)
            event_logged = True
            update_reason = str(policy_event.update_reason)
            if not effective_override_reason:
                effective_override_reason = str(policy_event.override_reason)
        except Exception as exc:
            update_reason = f"event_logging_error:{type(exc).__name__}"
    elif is_adaptive_condition and budget_policy_decision is None:
        if not budget_policy_eligible:
            update_reason = "not_eligible"
        elif normalized_adaptive_budget_policy_mode == "off":
            update_reason = "policy_off"
        else:
            update_reason = "no_decision_emitted"
    elif is_adaptive_condition:
        update_reason = "policy_controller_unavailable"
    else:
        update_reason = "non_adaptive_condition"

    _sync_adaptive_budget_policy_metadata(
        update=AdaptiveBudgetMetadataUpdate(
            attempt_metadata=attempt_metadata,
            patch_meta=patch_meta,
            mode=budget_policy_mode_for_attempt,
            algo=budget_policy_algo_for_attempt,
            architecture=(
                str(getattr(budget_policy_decision, "architecture", ""))
                if budget_policy_decision is not None
                else str(
                    getattr(
                        adaptive_budget_policy_controller.config,
                        "architecture",
                        BUDGET_POLICY_ARCH_TIER_BINARY,
                    )
                )
                if adaptive_budget_policy_controller is not None
                else BUDGET_POLICY_ARCH_TIER_BINARY
            ),
            decision=budget_policy_decision,
            eligible=budget_policy_eligible,
            override_reason=effective_override_reason,
            feature_scaling_profile=budget_policy_feature_scaling_profile,
            feature_values_raw=budget_policy_features_raw,
            feature_values_scaled=budget_policy_features_scaled,
            tiers=tiers,
            tier_index_before=tier_index_before,
            tier_index_after=tier_index_after,
            selected_multiplier=selected_multiplier,
            applied_multiplier=applied_multiplier,
            default_multiplier=default_multiplier,
            default_controls=default_controls,
            selected_controls=selected_controls,
            applied_controls=applied_controls,
            selected_controls_changed=selected_controls_changed,
            applied_controls_changed=applied_controls_changed,
            partial_override=partial_override,
            partial_override_errors=partial_override_errors,
            max_ups_per_case=max_ups_per_case,
            ups_used=ups_used,
            reward_lambda_tokens=reward_lambda_tokens,
            reward_mu_latency=reward_mu_latency,
            reward_token_scale=reward_token_scale,
            reward_latency_scale=reward_latency_scale,
            reward_rho_budget_exceeded=reward_rho_budget_exceeded,
            reward_eta_progress=reward_eta_progress,
            reward_kappa_stagnation=reward_kappa_stagnation,
            reward_success_hint_scale=reward_success_hint_scale,
            reward_progress_hint_scale=reward_progress_hint_scale,
            reward_stagnation_penalty_scale=reward_stagnation_penalty_scale,
            reward_scope=ADAPTIVE_BUDGET_REWARD_SCOPE,
            reward=reward,
            reward_success_hint=reward_success_hint,
            reward_progress_hint=reward_progress_hint,
            reward_progress_score=max(0.0, min(1.0, float(adaptive_progress_score))),
            reward_token_penalty=reward_token_penalty,
            reward_latency_penalty=reward_latency_penalty,
            reward_stagnation_penalty=reward_stagnation_penalty,
            reward_budget_exceeded_penalty=reward_budget_exceeded_penalty,
            request_budget_exceeded=request_budget_exceeded,
            local_accounting=local_accounting,
            updated=updated,
            update_reason=update_reason,
            event_logged=event_logged,
            decay=(
                float(getattr(adaptive_budget_policy_controller.config, "decay", 1.0))
                if adaptive_budget_policy_controller is not None
                else 1.0
            ),
            bwk_run_token_budget=(
                int(getattr(policy_event, "bwk_run_token_budget", 0) or 0)
                if policy_event is not None
                else int(getattr(budget_policy_decision, "bwk_run_token_budget", 0) or 0)
            ),
            bwk_remaining_token_budget_before=(
                int(getattr(policy_event, "bwk_remaining_token_budget_before", 0) or 0)
                if policy_event is not None
                else int(getattr(budget_policy_decision, "bwk_remaining_token_budget", 0) or 0)
            ),
            bwk_remaining_token_budget_after=int(
                getattr(policy_event, "bwk_remaining_token_budget_after", 0) or 0
            ),
            bwk_tokens_charged=int(getattr(policy_event, "bwk_tokens_charged", 0) or 0),
            bwk_cumulative_tokens_charged=int(
                getattr(policy_event, "bwk_cumulative_tokens_charged", 0) or 0
            ),
            bwk_dual_variable_before=(
                float(getattr(policy_event, "bwk_dual_variable_before", 0.0) or 0.0)
                if policy_event is not None
                else float(getattr(budget_policy_decision, "bwk_dual_variable", 0.0) or 0.0)
            ),
            bwk_dual_variable_after=float(
                getattr(policy_event, "bwk_dual_variable_after", 0.0) or 0.0
            ),
            bwk_budget_exhausted=bool(getattr(policy_event, "bwk_budget_exhausted", False)),
        )
    )


def _sync_finalize_progress_kwargs_from_attempt_metadata(
    *,
    online_finalize_kwargs: dict[str, Any],
    budget_finalize_kwargs: dict[str, Any],
) -> None:
    streak_default = _nonnegative_int(
        online_finalize_kwargs.get("adaptive_no_progress_streak_current"),
        default=0,
    )
    progress_default = bool(online_finalize_kwargs.get("adaptive_progress_made"))
    progress_score_default = max(
        0.0,
        float(online_finalize_kwargs.get("adaptive_progress_score") or 0.0),
    )
    attempt_metadata = online_finalize_kwargs.get("attempt_metadata")
    if isinstance(attempt_metadata, dict):
        progress_default = bool(attempt_metadata.get("adaptive_progress_made", progress_default))
        try:
            progress_score_default = max(
                0.0,
                float(attempt_metadata.get("adaptive_progress_score", progress_score_default)),
            )
        except (TypeError, ValueError):
            pass
        streak_default = _nonnegative_int(
            attempt_metadata.get("adaptive_no_progress_streak"),
            default=streak_default,
        )
    online_finalize_kwargs["adaptive_progress_made"] = progress_default
    online_finalize_kwargs["adaptive_progress_score"] = progress_score_default
    online_finalize_kwargs["adaptive_no_progress_streak_current"] = streak_default
    budget_finalize_kwargs["adaptive_progress_made"] = progress_default
    budget_finalize_kwargs["adaptive_progress_score"] = progress_score_default
    budget_finalize_kwargs["adaptive_no_progress_streak_current"] = streak_default


def _finalize_adaptive_policy_attempt(
    *,
    note: str,
    success: bool,
    attempt_wall_sec: float,
    online_finalize_kwargs: dict[str, Any],
    budget_finalize_kwargs: dict[str, Any],
) -> PendingAdaptiveOnlineCredit | None:
    _sync_finalize_progress_kwargs_from_attempt_metadata(
        online_finalize_kwargs=online_finalize_kwargs,
        budget_finalize_kwargs=budget_finalize_kwargs,
    )
    pending_credit = _finalize_adaptive_online_attempt(
        note=note,
        success=success,
        attempt_wall_sec=attempt_wall_sec,
        **online_finalize_kwargs,
    )
    _finalize_adaptive_budget_policy_attempt(
        note=note,
        success=success,
        attempt_wall_sec=attempt_wall_sec,
        **budget_finalize_kwargs,
    )
    return pending_credit
