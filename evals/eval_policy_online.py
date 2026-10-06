"""Adaptive online-policy metadata and finalization helpers."""

from __future__ import annotations

from collections.abc import Collection
from dataclasses import dataclass, field
from typing import Any

from . import linucb_core as core
from .adaptive_online_policy import (
    ACTION_SKIP_HELPER,
    ONLINE_POLICY_ACTIONS,
    AdaptiveOnlinePolicyController,
    PolicyDecision,
)
from .decision_model.policy import decision_model_metadata_fields
from .eval_policy_features import _adaptive_online_local_reward_components
from .eval_types import (
    ADAPTIVE_AUX_REWARD_COMPACTION_EFFICIENCY_WEIGHT_DEFAULT,
    ADAPTIVE_AUX_REWARD_EVIDENCE_NOVELTY_WEIGHT_DEFAULT,
    ADAPTIVE_AUX_REWARD_INFORMATIVENESS_ALIGNMENT_WEIGHT_DEFAULT,
    ADAPTIVE_AUX_REWARD_PYTEST_REDUCTION_WEIGHT_DEFAULT,
    ADAPTIVE_DELAYED_RESCUE_HORIZON_ATTEMPTS,
    ADAPTIVE_HELPER_OUTCOME_COMPAT_REASONING_ONLY,
    ADAPTIVE_HELPER_OUTCOME_INVALID_PAYLOAD,
    ADAPTIVE_HELPER_OUTCOME_MATERIALIZED,
    ADAPTIVE_HELPER_OUTCOME_NOT_INVOKED,
    ADAPTIVE_HELPER_OUTCOME_TRANSPORT_FAILED,
    ADAPTIVE_HELPER_OUTCOME_UNAVAILABLE_BACKEND,
    ADAPTIVE_LOCAL_REWARD_DELAYED_RESCUE_BONUS_BASE,
    ADAPTIVE_LOCAL_REWARD_DELAYED_RESCUE_BONUS_MAX,
    ADAPTIVE_LOCAL_REWARD_DELAYED_RESCUE_CONTEXT_MULTIPLIER,
    ADAPTIVE_LOCAL_REWARD_DELAYED_RESCUE_HORIZON_TWO_FACTOR,
    ADAPTIVE_LOCAL_REWARD_HELPER_MATERIALIZATION_BONUS_SCALE,
    ADAPTIVE_LOCAL_REWARD_PROGRESS_HINT_SCALE,
    ADAPTIVE_LOCAL_REWARD_STAGNATION_PENALTY_SCALE,
    ADAPTIVE_LOCAL_REWARD_SUCCESS_HINT,
    ADAPTIVE_ONLINE_REWARD_SCOPE,
    EMPTY_ADAPTIVE_POLICY_LOCAL_ACCOUNTING,
    HELPER_CONTEXT_MATERIALIZING_ACTIONS,
    AdaptiveOnlineRewardBreakdown,
    AdaptivePolicyLocalAccounting,
    AdaptiveProgressState,
    PendingAdaptiveOnlineCredit,
)


def _nonnegative_int(value: Any, *, default: int = 0) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return max(0, int(default))


def _adaptive_optional_int(value: Any) -> int | None:
    try:
        return int(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def _helper_evidence_reached_prompt(
    *, helper_evidence_ids: Collection[str], prompt_evidence_ids: Collection[str]
) -> bool:
    """Whether evidence the helper added was sent again in a later prompt (POL-11)."""
    added = {str(item) for item in helper_evidence_ids if str(item)}
    return bool(added & {str(item) for item in prompt_evidence_ids})


def _adaptive_online_delayed_rescue_bonus(
    *,
    origin_attempt: int,
    resolved_attempt: int,
    failure_signature_repeat: bool,
    helper_evidence_ids: Collection[str],
    prompt_evidence_ids: Collection[str],
) -> float:
    """Bonus for a helper call followed by a success within the horizon.

    Paid only when evidence the helper added at *origin_attempt* is in the successful
    attempt's prompt (POL-11); timing alone used to earn up to 0.75. The unused
    ``case_difficulty`` and ``case_snapshot_dependency`` inputs are gone.
    """
    horizon_attempts = max(0, int(resolved_attempt) - int(origin_attempt))
    if horizon_attempts <= 0 or horizon_attempts > ADAPTIVE_DELAYED_RESCUE_HORIZON_ATTEMPTS:
        return 0.0
    if not _helper_evidence_reached_prompt(
        helper_evidence_ids=helper_evidence_ids, prompt_evidence_ids=prompt_evidence_ids
    ):
        return 0.0
    bonus = ADAPTIVE_LOCAL_REWARD_DELAYED_RESCUE_BONUS_BASE
    if horizon_attempts == 2:
        bonus *= ADAPTIVE_LOCAL_REWARD_DELAYED_RESCUE_HORIZON_TWO_FACTOR
    if failure_signature_repeat:
        bonus *= ADAPTIVE_LOCAL_REWARD_DELAYED_RESCUE_CONTEXT_MULTIPLIER
    return min(ADAPTIVE_LOCAL_REWARD_DELAYED_RESCUE_BONUS_MAX, max(0.0, float(bonus)))


def _adaptive_online_behavior_fields(
    *,
    decision: PolicyDecision | None,
    behavior_action: str,
) -> tuple[str, float]:
    if decision is None:
        return "none", 1.0
    behavior_action_norm = (
        str(behavior_action).strip()
        if str(behavior_action).strip() in ONLINE_POLICY_ACTIONS
        else "none"
    )
    if behavior_action_norm == "none":
        return "none", 1.0
    # In shadow mode the executed baseline action is deterministic (probability 1); only an
    # applied ("on") selection carries the policy's propensity (POL-03).
    behavior_propensity = core.behavior_propensity(
        mode=decision.mode,
        selected=decision.selected_action,
        behavior=behavior_action_norm,
        propensity=decision.propensity,
    )
    return behavior_action_norm, behavior_propensity


def _adaptive_online_update_suppression_reason(helper_outcome: str) -> str:
    if helper_outcome == ADAPTIVE_HELPER_OUTCOME_TRANSPORT_FAILED:
        return "helper_transport_no_update"
    if helper_outcome == ADAPTIVE_HELPER_OUTCOME_UNAVAILABLE_BACKEND:
        return "helper_backend_unavailable_no_update"
    if helper_outcome == ADAPTIVE_HELPER_OUTCOME_COMPAT_REASONING_ONLY:
        return "helper_compat_reasoning_only_no_update"
    if helper_outcome == ADAPTIVE_HELPER_OUTCOME_INVALID_PAYLOAD:
        return "helper_invalid_payload_no_update"
    return ""


@dataclass(frozen=True)
class AdaptiveOnlineMetadataUpdate:
    """Typed payload for adaptive online metadata synchronization."""

    attempt_metadata: dict[str, Any]
    patch_meta: dict[str, Any]
    mode: str
    algo: str
    decision: PolicyDecision | None
    eligible: bool
    override_reason: str
    feature_scaling_profile: str
    feature_values_raw: dict[str, float]
    feature_values_scaled: dict[str, float]
    reward_lambda_tokens: float
    reward_mu_latency: float
    reward_token_scale: float
    reward_latency_scale: float
    reward_eta_progress: float
    reward_kappa_stagnation: float
    reward_success_hint_scale: float = ADAPTIVE_LOCAL_REWARD_SUCCESS_HINT
    reward_progress_hint_scale: float = ADAPTIVE_LOCAL_REWARD_PROGRESS_HINT_SCALE
    reward_stagnation_penalty_scale: float = ADAPTIVE_LOCAL_REWARD_STAGNATION_PENALTY_SCALE
    reward_helper_materialization_bonus_scale: float = (
        ADAPTIVE_LOCAL_REWARD_HELPER_MATERIALIZATION_BONUS_SCALE
    )
    reward_scope: str = ADAPTIVE_ONLINE_REWARD_SCOPE
    reward: float | None = None
    reward_success_hint: float = 0.0
    reward_progress_hint: float = 0.0
    reward_progress_score: float = 0.0
    reward_token_penalty: float = 0.0
    reward_latency_penalty: float = 0.0
    reward_stagnation_penalty: float = 0.0
    reward_helper_materialization_bonus: float = 0.0
    reward_delayed_rescue_bonus: float = 0.0
    reward_credit_resolved_attempt: int = 0
    reward_credit_horizon_attempts: int = 0
    reward_credit_resolution_reason: str = ""
    # Auxiliary reward breakdown (additive, gated by enable flag).
    reward_aux_total: float = 0.0
    reward_aux_pytest_fail_reduction: float = 0.0
    reward_aux_evidence_novelty_ratio: float = 0.0
    reward_aux_compaction_ratio: float = 0.0
    reward_aux_informativeness_alignment: float = 0.0
    local_accounting: AdaptivePolicyLocalAccounting = EMPTY_ADAPTIVE_POLICY_LOCAL_ACCOUNTING
    updated: bool = False
    update_reason: str = ""
    event_logged: bool = False
    behavior_action: str = "none"
    decay: float = 1.0
    configured_action_score_biases: dict[str, float] = field(default_factory=dict)


def _sync_record_metadata_copies(
    *,
    record: dict[str, Any] | None,
    attempt_metadata: dict[str, Any],
    patch_meta: dict[str, Any],
) -> None:
    if not isinstance(record, dict):
        return
    record["attempt_metadata"] = dict(attempt_metadata)
    if "patch_meta" in record:
        record["patch_meta"] = dict(patch_meta)


def _adaptive_online_reward_breakdown(
    *,
    success: bool,
    helper_materialized_context: bool,
    local_tokens_total: int,
    local_wall_sec: float,
    adaptive_progress_score: float,
    adaptive_no_progress_streak: int,
    lambda_tokens: float,
    mu_latency: float,
    token_scale: float,
    latency_scale: float,
    eta_progress: float,
    kappa_stagnation: float,
    success_hint_scale: float = ADAPTIVE_LOCAL_REWARD_SUCCESS_HINT,
    progress_hint_scale: float = ADAPTIVE_LOCAL_REWARD_PROGRESS_HINT_SCALE,
    stagnation_penalty_scale: float = ADAPTIVE_LOCAL_REWARD_STAGNATION_PENALTY_SCALE,
    helper_materialization_bonus_scale: float = ADAPTIVE_LOCAL_REWARD_HELPER_MATERIALIZATION_BONUS_SCALE,
    delayed_rescue_bonus: float = 0.0,
    resolved_attempt: int = 0,
    horizon_attempts: int = 0,
    resolution_reason: str = "",
    reward_scope: str = ADAPTIVE_ONLINE_REWARD_SCOPE,
    enable_auxiliary_rewards: bool = False,
    aux_pytest_fail_reduction: float = 0.0,
    aux_evidence_novelty_ratio: float = 0.0,
    aux_compaction_ratio: float = 0.0,
    aux_informativeness_alignment: float = 0.0,
    aux_pytest_reduction_weight: float = ADAPTIVE_AUX_REWARD_PYTEST_REDUCTION_WEIGHT_DEFAULT,
    aux_evidence_novelty_weight: float = ADAPTIVE_AUX_REWARD_EVIDENCE_NOVELTY_WEIGHT_DEFAULT,
    aux_compaction_efficiency_weight: float = ADAPTIVE_AUX_REWARD_COMPACTION_EFFICIENCY_WEIGHT_DEFAULT,
    aux_informativeness_alignment_weight: float = ADAPTIVE_AUX_REWARD_INFORMATIVENESS_ALIGNMENT_WEIGHT_DEFAULT,
) -> AdaptiveOnlineRewardBreakdown:
    (
        reward,
        reward_success_hint,
        reward_progress_hint,
        reward_token_penalty,
        reward_latency_penalty,
        reward_stagnation_penalty,
        reward_helper_materialization_bonus,
        reward_aux_total,
        reward_aux_pytest_fail_reduction,
        reward_aux_evidence_novelty_ratio,
        reward_aux_compaction_ratio,
        reward_aux_informativeness_alignment,
    ) = _adaptive_online_local_reward_components(
        success=success,
        helper_materialized_context=helper_materialized_context,
        local_tokens_total=local_tokens_total,
        local_wall_sec=local_wall_sec,
        adaptive_progress_score=adaptive_progress_score,
        adaptive_no_progress_streak=adaptive_no_progress_streak,
        lambda_tokens=lambda_tokens,
        mu_latency=mu_latency,
        token_scale=token_scale,
        latency_scale=latency_scale,
        eta_progress=eta_progress,
        kappa_stagnation=kappa_stagnation,
        success_hint_scale=success_hint_scale,
        progress_hint_scale=progress_hint_scale,
        stagnation_penalty_scale=stagnation_penalty_scale,
        helper_materialization_bonus_scale=helper_materialization_bonus_scale,
        enable_auxiliary_rewards=enable_auxiliary_rewards,
        aux_pytest_fail_reduction=aux_pytest_fail_reduction,
        aux_evidence_novelty_ratio=aux_evidence_novelty_ratio,
        aux_compaction_ratio=aux_compaction_ratio,
        aux_informativeness_alignment=aux_informativeness_alignment,
        aux_pytest_reduction_weight=aux_pytest_reduction_weight,
        aux_evidence_novelty_weight=aux_evidence_novelty_weight,
        aux_compaction_efficiency_weight=aux_compaction_efficiency_weight,
        aux_informativeness_alignment_weight=aux_informativeness_alignment_weight,
    )
    final_reward = reward + max(0.0, float(delayed_rescue_bonus))
    return AdaptiveOnlineRewardBreakdown(
        reward_scope=str(reward_scope),
        reward=float(final_reward),
        reward_success_hint=float(reward_success_hint),
        reward_progress_hint=float(reward_progress_hint),
        reward_progress_score=max(0.0, min(1.0, float(adaptive_progress_score))),
        reward_token_penalty=float(reward_token_penalty),
        reward_latency_penalty=float(reward_latency_penalty),
        reward_stagnation_penalty=float(reward_stagnation_penalty),
        reward_helper_materialization_bonus=float(reward_helper_materialization_bonus),
        reward_delayed_rescue_bonus=max(0.0, float(delayed_rescue_bonus)),
        reward_credit_resolved_attempt=max(0, int(resolved_attempt)),
        reward_credit_horizon_attempts=max(0, int(horizon_attempts)),
        reward_credit_resolution_reason=str(resolution_reason or ""),
        reward_aux_total=float(reward_aux_total),
        reward_aux_pytest_fail_reduction=float(reward_aux_pytest_fail_reduction),
        reward_aux_evidence_novelty_ratio=float(reward_aux_evidence_novelty_ratio),
        reward_aux_compaction_ratio=float(reward_aux_compaction_ratio),
        reward_aux_informativeness_alignment=float(reward_aux_informativeness_alignment),
    )


def _sync_adaptive_online_metadata(
    *,
    update: AdaptiveOnlineMetadataUpdate | None = None,
    **legacy_kwargs: Any,
) -> None:
    if update is None:
        update = AdaptiveOnlineMetadataUpdate(**legacy_kwargs)
    decision = update.decision
    selected_action = decision.selected_action if decision is not None else "none"
    applied_action = decision.applied_action if decision is not None else "none"
    decision_id = decision.decision_id if decision is not None else ""
    decision_strategy = decision.decision_strategy if decision is not None else ""
    propensity = float(decision.propensity) if decision is not None else 1.0
    available_actions = list(decision.available_actions) if decision is not None else []
    unavailable_action_reasons = (
        dict(decision.unavailable_action_reasons) if decision is not None else {}
    )
    feature_scaling_caps = dict(decision.feature_scaling_caps) if decision is not None else {}
    feature_scaling_history_sizes = (
        dict(decision.feature_scaling_history_sizes) if decision is not None else {}
    )
    feature_scaling_fallback_to_v1 = (
        bool(decision.feature_scaling_fallback_to_v1) if decision is not None else False
    )
    configured_alpha = float(decision.configured_alpha) if decision is not None else 0.0
    configured_epsilon = float(decision.configured_epsilon) if decision is not None else 0.0
    configured_decay = (
        float(decision.configured_decay) if decision is not None else float(update.decay)
    )
    effective_epsilon = float(decision.effective_epsilon) if decision is not None else 0.0
    warmup_active = bool(decision.warmup_active) if decision is not None else False
    behavior_action, behavior_propensity = _adaptive_online_behavior_fields(
        decision=decision,
        behavior_action=update.behavior_action,
    )
    configured_action_score_biases = (
        {
            str(action): float(value)
            for action, value in decision.configured_action_score_biases.items()
        }
        if decision is not None
        else {
            str(action): float(value)
            for action, value in update.configured_action_score_biases.items()
        }
    )
    fields = {
        "adaptive_online_policy_mode": str(update.mode),
        "adaptive_online_policy_algo": str(update.algo),
        "adaptive_online_decision_id": str(decision_id),
        "adaptive_online_decision_strategy": str(decision_strategy),
        "adaptive_online_action_selected": str(selected_action),
        "adaptive_online_action_applied": str(applied_action),
        "adaptive_online_propensity": float(propensity),
        "adaptive_online_behavior_action": behavior_action,
        "adaptive_online_behavior_propensity": float(behavior_propensity),
        "adaptive_online_override_reason": str(update.override_reason),
        "adaptive_online_feature_scaling_profile": str(update.feature_scaling_profile),
        "adaptive_online_feature_scaling_caps": feature_scaling_caps,
        "adaptive_online_feature_scaling_history_sizes": feature_scaling_history_sizes,
        "adaptive_online_feature_scaling_fallback_to_v1": bool(feature_scaling_fallback_to_v1),
        "adaptive_online_feature_vector_raw": dict(update.feature_values_raw),
        "adaptive_online_feature_vector": dict(update.feature_values_scaled),
        "adaptive_online_available_actions": available_actions,
        "adaptive_online_unavailable_action_reasons": unavailable_action_reasons,
        "adaptive_online_action_score_biases": configured_action_score_biases,
        "adaptive_online_configured_alpha": float(configured_alpha),
        "adaptive_online_configured_epsilon": float(configured_epsilon),
        "adaptive_online_configured_decay": float(configured_decay),
        "adaptive_online_effective_epsilon": float(effective_epsilon),
        "adaptive_online_warmup_active": bool(warmup_active),
        "adaptive_online_decay": float(update.decay),
        "adaptive_online_eligible": bool(update.eligible),
        "adaptive_online_reward_lambda_tokens": float(update.reward_lambda_tokens),
        "adaptive_online_reward_mu_latency": float(update.reward_mu_latency),
        "adaptive_online_reward_token_scale": float(update.reward_token_scale),
        "adaptive_online_reward_latency_scale": float(update.reward_latency_scale),
        "adaptive_online_reward_eta_progress": float(update.reward_eta_progress),
        "adaptive_online_reward_kappa_stagnation": float(update.reward_kappa_stagnation),
        "adaptive_online_reward_success_hint_scale": float(update.reward_success_hint_scale),
        "adaptive_online_reward_progress_hint_scale": float(update.reward_progress_hint_scale),
        "adaptive_online_reward_stagnation_penalty_scale": float(
            update.reward_stagnation_penalty_scale
        ),
        "adaptive_online_reward_helper_materialization_bonus_scale": float(
            update.reward_helper_materialization_bonus_scale
        ),
        "adaptive_online_reward_scope": str(update.reward_scope),
        "adaptive_online_reward": update.reward,
        "adaptive_online_reward_success_hint": float(update.reward_success_hint),
        "adaptive_online_reward_progress_hint": float(update.reward_progress_hint),
        "adaptive_online_reward_progress_score": float(update.reward_progress_score),
        "adaptive_online_reward_token_penalty": float(update.reward_token_penalty),
        "adaptive_online_reward_latency_penalty": float(update.reward_latency_penalty),
        "adaptive_online_reward_stagnation_penalty": float(update.reward_stagnation_penalty),
        "adaptive_online_reward_helper_materialization_bonus": float(
            update.reward_helper_materialization_bonus
        ),
        "adaptive_online_reward_delayed_rescue_bonus": float(update.reward_delayed_rescue_bonus),
        "adaptive_online_reward_credit_resolved_attempt": int(
            update.reward_credit_resolved_attempt
        ),
        "adaptive_online_reward_credit_horizon_attempts": int(
            update.reward_credit_horizon_attempts
        ),
        "adaptive_online_reward_credit_resolution_reason": str(
            update.reward_credit_resolution_reason
        ),
        "adaptive_online_reward_aux_total": float(update.reward_aux_total),
        "adaptive_online_reward_aux_pytest_fail_reduction": float(
            update.reward_aux_pytest_fail_reduction
        ),
        "adaptive_online_reward_aux_evidence_novelty_ratio": float(
            update.reward_aux_evidence_novelty_ratio
        ),
        "adaptive_online_reward_aux_compaction_ratio": float(update.reward_aux_compaction_ratio),
        "adaptive_online_reward_aux_informativeness_alignment": float(
            update.reward_aux_informativeness_alignment
        ),
        "adaptive_online_local_llm_call_count": int(update.local_accounting.llm_call_count),
        "adaptive_online_local_tokens_prompt": int(update.local_accounting.tokens_prompt),
        "adaptive_online_local_tokens_completion": int(update.local_accounting.tokens_completion),
        "adaptive_online_local_tokens_total": int(update.local_accounting.tokens_total),
        "adaptive_online_local_wall_sec": float(update.local_accounting.wall_sec),
        "adaptive_online_updated": bool(update.updated),
        "adaptive_online_update_reason": str(update.update_reason),
        "adaptive_online_event_logged": bool(update.event_logged),
        **decision_model_metadata_fields(decision.decision_model if decision is not None else None),
    }
    update.attempt_metadata.update(fields)
    update.patch_meta.update(fields)


def _record_adaptive_online_policy_outcome(
    *,
    adaptive_online_policy_controller: AdaptiveOnlinePolicyController | None,
    decision: PolicyDecision | None,
    run_id: str,
    case_id: str,
    condition_value: str,
    attempt: int,
    note: str,
    success: bool,
    reward_breakdown: AdaptiveOnlineRewardBreakdown,
    local_accounting: AdaptivePolicyLocalAccounting,
    override_reason: str,
    behavior_action: str,
    behavior_propensity: float,
    allow_model_update: bool,
    reward_success_hint_scale: float = ADAPTIVE_LOCAL_REWARD_SUCCESS_HINT,
    reward_progress_hint_scale: float = ADAPTIVE_LOCAL_REWARD_PROGRESS_HINT_SCALE,
    reward_stagnation_penalty_scale: float = ADAPTIVE_LOCAL_REWARD_STAGNATION_PENALTY_SCALE,
    reward_helper_materialization_bonus_scale: float = (
        ADAPTIVE_LOCAL_REWARD_HELPER_MATERIALIZATION_BONUS_SCALE
    ),
    update_reason_override: str = "",
) -> tuple[bool, bool, str, str]:
    effective_override_reason = str(override_reason or "")
    updated = False
    event_logged = False
    update_reason = "policy_off"
    if decision is None:
        return updated, event_logged, update_reason, effective_override_reason
    if adaptive_online_policy_controller is None:
        return False, False, "policy_controller_unavailable", effective_override_reason
    try:
        policy_event = adaptive_online_policy_controller.record_outcome(
            decision=decision,
            run_id=run_id,
            case_id=case_id,
            condition=condition_value,
            attempt=attempt,
            note=note,
            reward_scope=reward_breakdown.reward_scope,
            reward_success_hint_scale=reward_success_hint_scale,
            reward_progress_hint_scale=reward_progress_hint_scale,
            reward_stagnation_penalty_scale=reward_stagnation_penalty_scale,
            reward_helper_materialization_bonus_scale=reward_helper_materialization_bonus_scale,
            reward=reward_breakdown.reward,
            reward_success_hint=reward_breakdown.reward_success_hint,
            reward_progress_hint=reward_breakdown.reward_progress_hint,
            reward_progress_score=reward_breakdown.reward_progress_score,
            reward_token_penalty=reward_breakdown.reward_token_penalty,
            reward_latency_penalty=reward_breakdown.reward_latency_penalty,
            reward_stagnation_penalty=reward_breakdown.reward_stagnation_penalty,
            reward_helper_materialization_bonus=reward_breakdown.reward_helper_materialization_bonus,
            reward_delayed_rescue_bonus=reward_breakdown.reward_delayed_rescue_bonus,
            reward_credit_resolved_attempt=reward_breakdown.reward_credit_resolved_attempt,
            reward_credit_horizon_attempts=reward_breakdown.reward_credit_horizon_attempts,
            reward_credit_resolution_reason=reward_breakdown.reward_credit_resolution_reason,
            success=success,
            local_llm_call_count=local_accounting.llm_call_count,
            local_tokens_prompt=local_accounting.tokens_prompt,
            local_tokens_completion=local_accounting.tokens_completion,
            local_tokens_total=local_accounting.tokens_total,
            local_wall_sec=local_accounting.wall_sec,
            override_reason=effective_override_reason,
            behavior_action=behavior_action,
            behavior_propensity=behavior_propensity,
            allow_model_update=allow_model_update,
            update_reason_override=update_reason_override or None,
        )
        updated = bool(policy_event.model_updated)
        event_logged = True
        update_reason = str(policy_event.update_reason)
        if not effective_override_reason:
            effective_override_reason = str(policy_event.override_reason)
    except Exception as exc:
        update_reason = f"event_logging_error:{type(exc).__name__}"
    return updated, event_logged, update_reason, effective_override_reason


def _adaptive_online_should_defer_credit(
    *,
    is_adaptive_llm_discretion: bool,
    decision: PolicyDecision | None,
    adaptive_online_policy_controller: AdaptiveOnlinePolicyController | None,
    behavior_action: str,
    helper_outcome: str,
    suppression_reason: str,
    success: bool,
) -> bool:
    return bool(
        is_adaptive_llm_discretion
        and decision is not None
        and adaptive_online_policy_controller is not None
        and behavior_action in HELPER_CONTEXT_MATERIALIZING_ACTIONS
        and helper_outcome == ADAPTIVE_HELPER_OUTCOME_MATERIALIZED
        and not suppression_reason
        and not success
    )


def _resolve_pending_adaptive_online_credit(
    *,
    pending: PendingAdaptiveOnlineCredit,
    adaptive_online_policy_controller: AdaptiveOnlinePolicyController | None,
    delayed_rescue_bonus: float,
    resolved_attempt: int,
    resolution_reason: str,
) -> None:
    behavior_action, behavior_propensity = _adaptive_online_behavior_fields(
        decision=pending.decision,
        behavior_action=pending.behavior_action,
    )
    reward_breakdown = _adaptive_online_reward_breakdown(
        success=pending.success,
        helper_materialized_context=(
            behavior_action != ACTION_SKIP_HELPER and bool(pending.context_materialized)
        ),
        local_tokens_total=pending.local_accounting.tokens_total,
        local_wall_sec=pending.local_accounting.wall_sec,
        adaptive_progress_score=pending.progress_state.score,
        adaptive_no_progress_streak=pending.adaptive_no_progress_streak_current,
        lambda_tokens=pending.reward_lambda_tokens,
        mu_latency=pending.reward_mu_latency,
        token_scale=pending.reward_token_scale,
        latency_scale=pending.reward_latency_scale,
        eta_progress=pending.reward_eta_progress,
        kappa_stagnation=pending.reward_kappa_stagnation,
        success_hint_scale=pending.reward_success_hint_scale,
        progress_hint_scale=pending.reward_progress_hint_scale,
        stagnation_penalty_scale=pending.reward_stagnation_penalty_scale,
        helper_materialization_bonus_scale=pending.reward_helper_materialization_bonus_scale,
        delayed_rescue_bonus=delayed_rescue_bonus,
        resolved_attempt=resolved_attempt,
        horizon_attempts=max(0, int(resolved_attempt) - int(pending.attempt))
        if resolved_attempt > 0
        else 0,
        resolution_reason=resolution_reason,
        enable_auxiliary_rewards=pending.enable_auxiliary_rewards,
        aux_pytest_fail_reduction=pending.aux_pytest_fail_reduction,
        aux_evidence_novelty_ratio=pending.aux_evidence_novelty_ratio,
        aux_compaction_ratio=pending.aux_compaction_ratio,
        aux_informativeness_alignment=pending.aux_informativeness_alignment,
        aux_pytest_reduction_weight=pending.aux_pytest_reduction_weight,
        aux_evidence_novelty_weight=pending.aux_evidence_novelty_weight,
        aux_compaction_efficiency_weight=pending.aux_compaction_efficiency_weight,
        aux_informativeness_alignment_weight=pending.aux_informativeness_alignment_weight,
    )
    updated, event_logged, update_reason, effective_override_reason = (
        _record_adaptive_online_policy_outcome(
            adaptive_online_policy_controller=adaptive_online_policy_controller,
            decision=pending.decision,
            run_id=pending.run_id,
            case_id=pending.case_id,
            condition_value=pending.condition_value,
            attempt=pending.attempt,
            note=pending.note,
            success=pending.success,
            reward_breakdown=reward_breakdown,
            local_accounting=pending.local_accounting,
            override_reason=pending.online_policy_override_reason or pending.override_reason,
            behavior_action=behavior_action,
            behavior_propensity=behavior_propensity,
            allow_model_update=True,
        )
    )
    _sync_adaptive_online_metadata(
        update=AdaptiveOnlineMetadataUpdate(
            attempt_metadata=pending.attempt_metadata,
            patch_meta=pending.patch_meta,
            mode=pending.online_policy_mode_for_attempt,
            algo=pending.online_policy_algo_for_attempt,
            decision=pending.decision,
            eligible=pending.online_policy_eligible,
            override_reason=effective_override_reason,
            feature_scaling_profile=pending.online_policy_feature_scaling_profile,
            feature_values_raw=pending.online_policy_features_raw,
            feature_values_scaled=pending.online_policy_features_scaled,
            reward_lambda_tokens=pending.reward_lambda_tokens,
            reward_mu_latency=pending.reward_mu_latency,
            reward_token_scale=pending.reward_token_scale,
            reward_latency_scale=pending.reward_latency_scale,
            reward_eta_progress=pending.reward_eta_progress,
            reward_kappa_stagnation=pending.reward_kappa_stagnation,
            reward_success_hint_scale=pending.reward_success_hint_scale,
            reward_progress_hint_scale=pending.reward_progress_hint_scale,
            reward_stagnation_penalty_scale=pending.reward_stagnation_penalty_scale,
            reward_helper_materialization_bonus_scale=pending.reward_helper_materialization_bonus_scale,
            reward_scope=reward_breakdown.reward_scope,
            reward=reward_breakdown.reward,
            reward_success_hint=reward_breakdown.reward_success_hint,
            reward_progress_hint=reward_breakdown.reward_progress_hint,
            reward_progress_score=reward_breakdown.reward_progress_score,
            reward_token_penalty=reward_breakdown.reward_token_penalty,
            reward_latency_penalty=reward_breakdown.reward_latency_penalty,
            reward_stagnation_penalty=reward_breakdown.reward_stagnation_penalty,
            reward_helper_materialization_bonus=(
                reward_breakdown.reward_helper_materialization_bonus
            ),
            reward_delayed_rescue_bonus=reward_breakdown.reward_delayed_rescue_bonus,
            reward_credit_resolved_attempt=reward_breakdown.reward_credit_resolved_attempt,
            reward_credit_horizon_attempts=reward_breakdown.reward_credit_horizon_attempts,
            reward_credit_resolution_reason=reward_breakdown.reward_credit_resolution_reason,
            reward_aux_total=reward_breakdown.reward_aux_total,
            reward_aux_pytest_fail_reduction=reward_breakdown.reward_aux_pytest_fail_reduction,
            reward_aux_evidence_novelty_ratio=reward_breakdown.reward_aux_evidence_novelty_ratio,
            reward_aux_compaction_ratio=reward_breakdown.reward_aux_compaction_ratio,
            reward_aux_informativeness_alignment=(
                reward_breakdown.reward_aux_informativeness_alignment
            ),
            local_accounting=pending.local_accounting,
            updated=updated,
            update_reason=update_reason,
            event_logged=event_logged,
            behavior_action=behavior_action,
            decay=(
                float(getattr(adaptive_online_policy_controller.config, "decay", 1.0))
                if adaptive_online_policy_controller is not None
                else 1.0
            ),
        )
    )
    _sync_record_metadata_copies(
        record=pending.record_ref,
        attempt_metadata=pending.attempt_metadata,
        patch_meta=pending.patch_meta,
    )


def _finalize_adaptive_online_attempt(
    *,
    note: str,
    success: bool,
    attempt_wall_sec: float,
    is_adaptive_llm_discretion: bool,
    online_policy_decision: PolicyDecision | None,
    adaptive_online_policy_controller: AdaptiveOnlinePolicyController | None,
    run_id: str,
    case_id: str,
    condition_value: str,
    attempt: int,
    attempt_metadata: dict[str, Any],
    patch_meta: dict[str, Any],
    online_policy_mode_for_attempt: str,
    online_policy_algo_for_attempt: str,
    online_policy_eligible: bool,
    online_policy_override_reason: str,
    online_policy_feature_scaling_profile: str,
    online_policy_features_raw: dict[str, float],
    online_policy_features_scaled: dict[str, float],
    helper_decision_reason: str,
    normalized_adaptive_online_policy_mode: str,
    adaptive_progress_made: bool,
    adaptive_progress_score: float,
    adaptive_no_progress_streak_current: int,
    reward_lambda_tokens: float,
    reward_mu_latency: float,
    reward_token_scale: float,
    reward_latency_scale: float,
    reward_eta_progress: float,
    reward_kappa_stagnation: float,
    reward_success_hint_scale: float = ADAPTIVE_LOCAL_REWARD_SUCCESS_HINT,
    reward_progress_hint_scale: float = ADAPTIVE_LOCAL_REWARD_PROGRESS_HINT_SCALE,
    reward_stagnation_penalty_scale: float = ADAPTIVE_LOCAL_REWARD_STAGNATION_PENALTY_SCALE,
    reward_helper_materialization_bonus_scale: float = ADAPTIVE_LOCAL_REWARD_HELPER_MATERIALIZATION_BONUS_SCALE,
    local_accounting: AdaptivePolicyLocalAccounting = EMPTY_ADAPTIVE_POLICY_LOCAL_ACCOUNTING,
    override_reason: str = "",
    behavior_action: str = "none",
    context_materialized: bool = False,
    helper_outcome: str = ADAPTIVE_HELPER_OUTCOME_NOT_INVOKED,
    enable_auxiliary_rewards: bool = False,
    aux_pytest_fail_reduction: float = 0.0,
    aux_evidence_novelty_ratio: float = 0.0,
    aux_compaction_ratio: float = 0.0,
    aux_informativeness_alignment: float = 0.0,
    aux_pytest_reduction_weight: float = ADAPTIVE_AUX_REWARD_PYTEST_REDUCTION_WEIGHT_DEFAULT,
    aux_evidence_novelty_weight: float = ADAPTIVE_AUX_REWARD_EVIDENCE_NOVELTY_WEIGHT_DEFAULT,
    aux_compaction_efficiency_weight: float = ADAPTIVE_AUX_REWARD_COMPACTION_EFFICIENCY_WEIGHT_DEFAULT,
    aux_informativeness_alignment_weight: float = ADAPTIVE_AUX_REWARD_INFORMATIVENESS_ALIGNMENT_WEIGHT_DEFAULT,
) -> PendingAdaptiveOnlineCredit | None:
    effective_override_reason = (
        online_policy_override_reason
        if online_policy_override_reason
        else str(override_reason or "")
    )
    behavior_action, behavior_propensity = _adaptive_online_behavior_fields(
        decision=online_policy_decision,
        behavior_action=behavior_action,
    )
    charged_local_accounting = (
        local_accounting
        if behavior_action != ACTION_SKIP_HELPER
        else AdaptivePolicyLocalAccounting()
    )
    reward_breakdown = _adaptive_online_reward_breakdown(
        success=success,
        helper_materialized_context=(
            behavior_action != ACTION_SKIP_HELPER and bool(context_materialized)
        ),
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
        success_hint_scale=reward_success_hint_scale,
        progress_hint_scale=reward_progress_hint_scale,
        stagnation_penalty_scale=reward_stagnation_penalty_scale,
        helper_materialization_bonus_scale=reward_helper_materialization_bonus_scale,
        enable_auxiliary_rewards=enable_auxiliary_rewards,
        aux_pytest_fail_reduction=aux_pytest_fail_reduction,
        aux_evidence_novelty_ratio=aux_evidence_novelty_ratio,
        aux_compaction_ratio=aux_compaction_ratio,
        aux_informativeness_alignment=aux_informativeness_alignment,
        aux_pytest_reduction_weight=aux_pytest_reduction_weight,
        aux_evidence_novelty_weight=aux_evidence_novelty_weight,
        aux_compaction_efficiency_weight=aux_compaction_efficiency_weight,
        aux_informativeness_alignment_weight=aux_informativeness_alignment_weight,
    )
    updated = False
    event_logged = False
    update_reason = "policy_off"
    suppression_reason = _adaptive_online_update_suppression_reason(helper_outcome)
    allow_model_update = not bool(suppression_reason)
    pending_credit: PendingAdaptiveOnlineCredit | None = None
    if (
        is_adaptive_llm_discretion
        and online_policy_decision is not None
        and adaptive_online_policy_controller is not None
    ):
        if _adaptive_online_should_defer_credit(
            is_adaptive_llm_discretion=is_adaptive_llm_discretion,
            decision=online_policy_decision,
            adaptive_online_policy_controller=adaptive_online_policy_controller,
            behavior_action=behavior_action,
            helper_outcome=helper_outcome,
            suppression_reason=suppression_reason,
            success=success,
        ):
            update_reason = "pending_delayed_credit"
            pending_credit = PendingAdaptiveOnlineCredit(
                decision=online_policy_decision,
                run_id=run_id,
                case_id=case_id,
                condition_value=condition_value,
                attempt=attempt,
                note=note,
                success=success,
                attempt_metadata=attempt_metadata,
                patch_meta=patch_meta,
                record_ref=None,
                online_policy_mode_for_attempt=online_policy_mode_for_attempt,
                online_policy_algo_for_attempt=online_policy_algo_for_attempt,
                online_policy_eligible=online_policy_eligible,
                online_policy_override_reason=effective_override_reason,
                online_policy_feature_scaling_profile=online_policy_feature_scaling_profile,
                online_policy_features_raw=dict(online_policy_features_raw),
                online_policy_features_scaled=dict(online_policy_features_scaled),
                helper_decision_reason=helper_decision_reason,
                normalized_adaptive_online_policy_mode=normalized_adaptive_online_policy_mode,
                adaptive_no_progress_streak_current=adaptive_no_progress_streak_current,
                reward_lambda_tokens=reward_lambda_tokens,
                reward_mu_latency=reward_mu_latency,
                reward_token_scale=reward_token_scale,
                reward_latency_scale=reward_latency_scale,
                reward_eta_progress=reward_eta_progress,
                reward_kappa_stagnation=reward_kappa_stagnation,
                reward_success_hint_scale=reward_success_hint_scale,
                reward_progress_hint_scale=reward_progress_hint_scale,
                reward_stagnation_penalty_scale=reward_stagnation_penalty_scale,
                reward_helper_materialization_bonus_scale=reward_helper_materialization_bonus_scale,
                local_accounting=local_accounting,
                override_reason=override_reason,
                behavior_action=behavior_action,
                context_materialized=context_materialized,
                helper_outcome=helper_outcome,
                progress_state=AdaptiveProgressState(
                    made=bool(adaptive_progress_made),
                    reason=str(attempt_metadata.get("adaptive_progress_reason") or ""),
                    score=max(0.0, float(adaptive_progress_score)),
                    component_signature=max(
                        0.0,
                        float(attempt_metadata.get("adaptive_progress_component_signature") or 0.0),
                    ),
                    component_new_evidence=max(
                        0.0,
                        float(
                            attempt_metadata.get("adaptive_progress_component_new_evidence") or 0.0
                        ),
                    ),
                    component_pytest_fail_reduction=max(
                        0.0,
                        float(
                            attempt_metadata.get(
                                "adaptive_progress_component_pytest_fail_reduction"
                            )
                            or 0.0
                        ),
                    ),
                    component_stage_advance=max(
                        0.0,
                        float(
                            attempt_metadata.get("adaptive_progress_component_stage_advance") or 0.0
                        ),
                    ),
                    component_verifier_score=max(
                        0.0,
                        float(
                            attempt_metadata.get("adaptive_progress_component_verifier_score")
                            or 0.0
                        ),
                    ),
                    component_scope_alignment=max(
                        0.0,
                        float(
                            attempt_metadata.get("adaptive_progress_component_scope_alignment")
                            or 0.0
                        ),
                    ),
                    pytest_before_failed_total=_adaptive_optional_int(
                        attempt_metadata.get("adaptive_pytest_before_failed_total")
                    ),
                    pytest_after_failed_total=_adaptive_optional_int(
                        attempt_metadata.get("adaptive_pytest_after_failed_total")
                    ),
                    stage_before=str(attempt_metadata.get("adaptive_pipeline_stage_before") or ""),
                    stage_after=str(attempt_metadata.get("adaptive_pipeline_stage_after") or ""),
                    stage_before_ord=_nonnegative_int(
                        attempt_metadata.get("adaptive_pipeline_stage_before_ord"),
                        default=0,
                    ),
                    stage_after_ord=_nonnegative_int(
                        attempt_metadata.get("adaptive_pipeline_stage_after_ord"),
                        default=0,
                    ),
                ),
                failure_signature_repeat=bool(attempt_metadata.get("adaptive_signature_repeat")),
                enable_auxiliary_rewards=enable_auxiliary_rewards,
                aux_pytest_fail_reduction=aux_pytest_fail_reduction,
                aux_evidence_novelty_ratio=aux_evidence_novelty_ratio,
                aux_compaction_ratio=aux_compaction_ratio,
                aux_informativeness_alignment=aux_informativeness_alignment,
                aux_pytest_reduction_weight=aux_pytest_reduction_weight,
                aux_evidence_novelty_weight=aux_evidence_novelty_weight,
                aux_compaction_efficiency_weight=aux_compaction_efficiency_weight,
                aux_informativeness_alignment_weight=aux_informativeness_alignment_weight,
            )
        else:
            updated, event_logged, update_reason, effective_override_reason = (
                _record_adaptive_online_policy_outcome(
                    adaptive_online_policy_controller=adaptive_online_policy_controller,
                    decision=online_policy_decision,
                    run_id=run_id,
                    case_id=case_id,
                    condition_value=condition_value,
                    attempt=attempt,
                    note=note,
                    success=success,
                    reward_breakdown=reward_breakdown,
                    local_accounting=local_accounting,
                    override_reason=effective_override_reason,
                    behavior_action=behavior_action,
                    behavior_propensity=behavior_propensity,
                    allow_model_update=allow_model_update,
                    reward_success_hint_scale=reward_success_hint_scale,
                    reward_progress_hint_scale=reward_progress_hint_scale,
                    reward_stagnation_penalty_scale=reward_stagnation_penalty_scale,
                    reward_helper_materialization_bonus_scale=(
                        reward_helper_materialization_bonus_scale
                    ),
                    update_reason_override=suppression_reason,
                )
            )
    elif is_adaptive_llm_discretion and online_policy_decision is None:
        if not online_policy_eligible:
            update_reason = f"not_eligible:{helper_decision_reason}"
        elif normalized_adaptive_online_policy_mode == "off":
            update_reason = "policy_off"
        else:
            update_reason = "no_decision_emitted"
    elif is_adaptive_llm_discretion:
        update_reason = "policy_controller_unavailable"
    else:
        update_reason = "non_llm_discretion_condition"

    _sync_adaptive_online_metadata(
        update=AdaptiveOnlineMetadataUpdate(
            attempt_metadata=attempt_metadata,
            patch_meta=patch_meta,
            mode=online_policy_mode_for_attempt,
            algo=online_policy_algo_for_attempt,
            decision=online_policy_decision,
            eligible=online_policy_eligible,
            override_reason=effective_override_reason,
            feature_scaling_profile=online_policy_feature_scaling_profile,
            feature_values_raw=online_policy_features_raw,
            feature_values_scaled=online_policy_features_scaled,
            reward_lambda_tokens=reward_lambda_tokens,
            reward_mu_latency=reward_mu_latency,
            reward_token_scale=reward_token_scale,
            reward_latency_scale=reward_latency_scale,
            reward_eta_progress=reward_eta_progress,
            reward_kappa_stagnation=reward_kappa_stagnation,
            reward_success_hint_scale=reward_success_hint_scale,
            reward_progress_hint_scale=reward_progress_hint_scale,
            reward_stagnation_penalty_scale=reward_stagnation_penalty_scale,
            reward_helper_materialization_bonus_scale=reward_helper_materialization_bonus_scale,
            reward_scope=ADAPTIVE_ONLINE_REWARD_SCOPE,
            reward=reward_breakdown.reward,
            reward_success_hint=reward_breakdown.reward_success_hint,
            reward_progress_hint=reward_breakdown.reward_progress_hint,
            reward_progress_score=reward_breakdown.reward_progress_score,
            reward_token_penalty=reward_breakdown.reward_token_penalty,
            reward_latency_penalty=reward_breakdown.reward_latency_penalty,
            reward_stagnation_penalty=reward_breakdown.reward_stagnation_penalty,
            reward_helper_materialization_bonus=(
                reward_breakdown.reward_helper_materialization_bonus
            ),
            reward_delayed_rescue_bonus=reward_breakdown.reward_delayed_rescue_bonus,
            reward_credit_resolved_attempt=reward_breakdown.reward_credit_resolved_attempt,
            reward_credit_horizon_attempts=reward_breakdown.reward_credit_horizon_attempts,
            reward_credit_resolution_reason=reward_breakdown.reward_credit_resolution_reason,
            reward_aux_total=reward_breakdown.reward_aux_total,
            reward_aux_pytest_fail_reduction=reward_breakdown.reward_aux_pytest_fail_reduction,
            reward_aux_evidence_novelty_ratio=reward_breakdown.reward_aux_evidence_novelty_ratio,
            reward_aux_compaction_ratio=reward_breakdown.reward_aux_compaction_ratio,
            reward_aux_informativeness_alignment=(
                reward_breakdown.reward_aux_informativeness_alignment
            ),
            local_accounting=local_accounting,
            updated=updated,
            update_reason=update_reason,
            event_logged=event_logged,
            behavior_action=behavior_action,
            decay=(
                float(getattr(adaptive_online_policy_controller.config, "decay", 1.0))
                if adaptive_online_policy_controller is not None
                else 1.0
            ),
        )
    )
    return pending_credit
