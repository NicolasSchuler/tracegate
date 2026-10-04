"""Adaptive policy feature and reward helpers for the eval harness."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from .adaptive_budget_online_policy import (
    ACTION_BUDGET_FAMILY_CONTEXT,
    ACTION_BUDGET_FAMILY_TIME,
    ACTION_BUDGET_FAMILY_TOKENS,
    BUDGET_POLICY_ARCH_HIERARCHICAL_V1,
    BUDGET_POLICY_ARCH_TIER_BINARY,
    BUDGET_POLICY_FAMILIES,
)
from .adaptive_online_policy import (
    ACTION_FRAME_LOCALS_LITE,
    ACTION_SKIP_HELPER,
    ACTION_SNAPSHOT_FULL_LITE,
    ACTION_SNAPSHOT_LITE,
    ONLINE_POLICY_ACTIONS,
)
from .eval_types import (
    ADAPTIVE_AUX_REWARD_COMPACTION_EFFICIENCY_WEIGHT_DEFAULT,
    ADAPTIVE_AUX_REWARD_EVIDENCE_NOVELTY_WEIGHT_DEFAULT,
    ADAPTIVE_AUX_REWARD_INFORMATIVENESS_ALIGNMENT_WEIGHT_DEFAULT,
    ADAPTIVE_AUX_REWARD_PYTEST_REDUCTION_WEIGHT_DEFAULT,
    ADAPTIVE_BUDGET_CONTROL_CHANGE_KEYS,
    ADAPTIVE_LOCAL_REWARD_HELPER_MATERIALIZATION_BONUS_SCALE,
    ADAPTIVE_LOCAL_REWARD_PROGRESS_HINT_SCALE,
    ADAPTIVE_LOCAL_REWARD_STAGNATION_PENALTY_SCALE,
    ADAPTIVE_LOCAL_REWARD_SUCCESS_HINT,
    AdaptiveBudgetControlProfile,
    AdaptivePolicyLocalAccounting,
    PreGateResult,
)
from .policy_feature_scaling import (
    INTERACTION_POLICY_FEATURE_SCALING_PROFILE,
    interaction_features,
    normalize_feature_scaling_profile,
)


def _nonnegative_int(value: Any, *, default: int = 0) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return max(0, int(default))


# Dataset labels from case.json (not observable at deployment time); masked by default.
CASE_LABEL_FEATURE_NAMES = (
    "case_difficulty_ord",
    "case_snapshot_dependency_ord",
    "case_bug_source_imported",
)
LITE_SNAPSHOT_ACTIONS = frozenset(
    {ACTION_SNAPSHOT_LITE, ACTION_FRAME_LOCALS_LITE, ACTION_SNAPSHOT_FULL_LITE}
)


def drop_case_label_features(features: Mapping[str, float]) -> dict[str, float]:
    """Return *features* without the oracle case-label features."""
    return {name: value for name, value in features.items() if name not in CASE_LABEL_FEATURE_NAMES}


def lite_action_replaces_snapshot(
    *,
    action: str,
    reveal_mode: str,
    lite_context: str,
    lite_snapshot_mode: str,
) -> bool:
    """Whether a lite action's section replaces the full snapshot in the Stage-B prompt.

    Under ``full`` reveal the full snapshot section is always present, so a lite action only
    changes the evidence when its compact section replaces that section (``replace`` mode).
    """
    return (
        action in LITE_SNAPSHOT_ACTIONS
        and bool(str(lite_context or "").strip())
        and str(reveal_mode) != "traceback_first"
        and str(lite_snapshot_mode) == "replace"
    )


def _adaptive_online_feature_vector(
    *,
    attempt: int,
    max_attempts: int,
    timed_out: bool,
    failure_signature_repeat: bool,
    stagnation_streak: int,
    helper_calls_used: int,
    helper_calls_without_context: int,
    adaptive_no_progress_streak: int,
    helper_budget_exhausted: bool,
    helper_saturation_guarded: bool,
    supports_context_request: bool,
    snapshot_available: bool,
    pregate_result: PreGateResult,
    effective_helper_budget_limit: int,
    adaptive_budget_unlimited: bool,
    adaptive_budget_topup_calls_used: int,
    adaptive_budget_topup_max_calls: int,
    last_helper_need_more_context: bool | None,
    last_helper_decision_reason: str,
    last_helper_action_executed: str,
    context_compaction_enabled: bool,
    supports_context_compaction: bool,
    context_compaction_calls_used: int,
    context_compaction_max_calls: int,
    last_context_compaction_requested: bool,
    last_context_compaction_invoked: bool,
    last_context_compaction_success: bool,
    last_context_compaction_ratio: float,
    last_context_compaction_trigger: str,
    case_difficulty: str,
    case_snapshot_dependency: str,
    case_bug_source: str | None,
    feature_scaling_profile: str = "v1",
) -> dict[str, float]:
    difficulty_ordinal = {
        "easy": 0.0,
        "medium": 0.5,
        "hard": 1.0,
    }.get(str(case_difficulty).strip().lower(), 0.5)
    snapshot_dependency_ordinal = {
        "none": 0.0,
        "helpful": 0.5,
        "required": 1.0,
    }.get(str(case_snapshot_dependency).strip().lower(), 0.5)
    bug_source_normalized = str(case_bug_source or "hand_crafted").strip().lower()
    pregate_exception_type_normalized = str(pregate_result.exception_type).strip().lower()
    max_attempts_safe = max(1, int(max_attempts))
    attempt_safe = max(1, int(attempt))
    helper_calls_used_safe = max(0, int(helper_calls_used))
    helper_budget_limit_safe = max(0, int(effective_helper_budget_limit))
    helper_budget_remaining = max(0, helper_budget_limit_safe - helper_calls_used_safe)
    helper_budget_remaining_frac = (
        1.0
        if adaptive_budget_unlimited
        else (
            helper_budget_remaining / float(helper_budget_limit_safe)
            if helper_budget_limit_safe > 0
            else 0.0
        )
    )
    topup_max_safe = max(0, int(adaptive_budget_topup_max_calls))
    topup_used_safe = max(0, int(adaptive_budget_topup_calls_used))
    helper_topup_used_frac = (
        min(1.0, topup_used_safe / float(topup_max_safe)) if topup_max_safe > 0 else 0.0
    )
    helper_topup_remaining_frac = (
        max(0.0, min(1.0, (topup_max_safe - topup_used_safe) / float(topup_max_safe)))
        if topup_max_safe > 0
        else 0.0
    )
    last_decision_reason = str(last_helper_decision_reason or "").strip()
    last_helper_action_norm = str(last_helper_action_executed or "").strip()
    context_compaction_calls_used_safe = max(0, int(context_compaction_calls_used))
    context_compaction_max_calls_safe = max(0, int(context_compaction_max_calls))
    context_compaction_calls_used_frac = (
        min(1.0, context_compaction_calls_used_safe / float(context_compaction_max_calls_safe))
        if context_compaction_max_calls_safe > 0
        else 0.0
    )
    context_compaction_calls_remaining_frac = (
        max(
            0.0,
            min(
                1.0,
                (context_compaction_max_calls_safe - context_compaction_calls_used_safe)
                / float(context_compaction_max_calls_safe),
            ),
        )
        if context_compaction_max_calls_safe > 0
        else 0.0
    )
    last_compaction_trigger = str(last_context_compaction_trigger or "")
    low_yield_ratio = (
        max(0, int(helper_calls_without_context)) / float(helper_calls_used_safe)
        if helper_calls_used_safe > 0
        else 0.0
    )
    features = {
        "bias": 1.0,
        "attempt_frac": float(attempt_safe / float(max_attempts_safe)),
        "timed_out": 1.0 if timed_out else 0.0,
        "failure_signature_repeat": 1.0 if failure_signature_repeat else 0.0,
        "stagnation_streak": float(max(0, int(stagnation_streak))),
        "helper_calls_used": float(helper_calls_used_safe),
        "helper_low_yield_ratio": float(low_yield_ratio),
        "helper_budget_remaining_frac": float(helper_budget_remaining_frac),
        "helper_topup_used_frac": float(helper_topup_used_frac),
        "helper_topup_remaining_frac": float(helper_topup_remaining_frac),
        "adaptive_no_progress_streak": float(max(0, int(adaptive_no_progress_streak))),
        "helper_budget_exhausted": 1.0 if helper_budget_exhausted else 0.0,
        "helper_saturation_guarded": 1.0 if helper_saturation_guarded else 0.0,
        "supports_context_request": 1.0 if supports_context_request else 0.0,
        "context_compaction_enabled": 1.0 if context_compaction_enabled else 0.0,
        "supports_context_compaction": 1.0 if supports_context_compaction else 0.0,
        "context_compaction_calls_used_frac": float(context_compaction_calls_used_frac),
        "context_compaction_calls_remaining_frac": float(context_compaction_calls_remaining_frac),
        "last_context_compaction_requested": 1.0 if last_context_compaction_requested else 0.0,
        "last_context_compaction_invoked": 1.0 if last_context_compaction_invoked else 0.0,
        "last_context_compaction_success": 1.0 if last_context_compaction_success else 0.0,
        "last_context_compaction_ratio": float(max(0.0, min(1.0, last_context_compaction_ratio))),
        "last_context_compaction_trigger_stage_a": 1.0
        if last_compaction_trigger == "stage_a_requested"
        else 0.0,
        "last_context_compaction_trigger_overflow": 1.0
        if last_compaction_trigger == "prompt_overflow"
        else 0.0,
        "last_context_compaction_trigger_always": 1.0
        if last_compaction_trigger == "mode_always"
        else 0.0,
        "snapshot_available": 1.0 if snapshot_available else 0.0,
        "last_helper_need_more_context": 1.0 if last_helper_need_more_context is True else 0.0,
        "last_helper_invalid_response": 1.0
        if last_decision_reason == "llm_invalid_response_no_helper"
        else 0.0,
        "last_helper_declined": 1.0 if last_decision_reason == "llm_declined_helper" else 0.0,
        "case_difficulty_ord": float(difficulty_ordinal),
        "case_snapshot_dependency_ord": float(snapshot_dependency_ordinal),
        "case_bug_source_imported": 0.0 if bug_source_normalized == "hand_crafted" else 1.0,
        "pregate_score": float(max(0.0, min(1.0, pregate_result.score))),
        "pregate_fired": 1.0 if pregate_result.should_skip else 0.0,
        "pregate_type_specificity": float(max(0.0, min(1.0, pregate_result.f_type_specificity))),
        "pregate_message_diagnostic": float(
            max(0.0, min(1.0, pregate_result.f_message_diagnostic))
        ),
        "pregate_crash_in_user_code": float(
            max(0.0, min(1.0, pregate_result.f_crash_in_user_code))
        ),
        "pregate_assertion_quality": float(max(0.0, min(1.0, pregate_result.f_assertion_quality))),
        "pregate_frame_depth": float(max(0.0, min(1.0, pregate_result.f_frame_depth))),
        "pregate_exception_is_assertion": 1.0
        if pregate_exception_type_normalized == "assertionerror"
        else 0.0,
        "pregate_exception_is_timeout": 1.0
        if pregate_exception_type_normalized == "timeouterror"
        else 0.0,
        "pregate_exception_is_syntax": 1.0
        if pregate_exception_type_normalized in {"syntaxerror", "indentationerror", "taberror"}
        else 0.0,
    }
    # skip_helper (also the value before any helper call) is the reference level: with all
    # ten levels the one-hot features summed to ``bias`` (POL-06).
    for action in ONLINE_POLICY_ACTIONS:
        if action != ACTION_SKIP_HELPER:
            features[f"last_helper_action_{action}"] = (
                1.0 if last_helper_action_norm == action else 0.0
            )
    if (
        normalize_feature_scaling_profile(feature_scaling_profile)
        == INTERACTION_POLICY_FEATURE_SCALING_PROFILE
    ):
        features.update(interaction_features(features))
    return features


def _adaptive_budget_control_profile_to_dict(
    profile: AdaptiveBudgetControlProfile,
) -> dict[str, Any]:
    return {
        "propose_budget_multiplier": float(profile.propose_budget_multiplier),
        "propose_budget_sec": float(profile.propose_budget_sec),
        "openai_max_tokens": int(profile.openai_max_tokens),
        "helper_max_calls": int(profile.helper_max_calls),
        "stage_a_max_requests": int(profile.stage_a_max_requests),
        "stage_a_max_chars": int(profile.stage_a_max_chars),
        "retry_history_max_chars": int(profile.retry_history_max_chars),
        "context_compaction_max_calls": int(profile.context_compaction_max_calls),
        "response_mode": str(profile.response_mode or ""),
    }


def _adaptive_budget_control_changes(
    *,
    current: AdaptiveBudgetControlProfile,
    target: AdaptiveBudgetControlProfile,
) -> dict[str, bool]:
    return {
        "propose_budget": (
            float(target.propose_budget_sec) != float(current.propose_budget_sec)
            or float(target.propose_budget_multiplier) != float(current.propose_budget_multiplier)
        ),
        "max_tokens": int(target.openai_max_tokens) != int(current.openai_max_tokens),
        "helper_allowance": int(target.helper_max_calls) != int(current.helper_max_calls),
        "stage_a_requests": int(target.stage_a_max_requests) != int(current.stage_a_max_requests),
        "stage_a_chars": int(target.stage_a_max_chars) != int(current.stage_a_max_chars),
        "retry_history_budget": int(target.retry_history_max_chars)
        != int(current.retry_history_max_chars),
        "compaction_allowance": int(target.context_compaction_max_calls)
        != int(current.context_compaction_max_calls),
        "response_mode": str(target.response_mode or "") != str(current.response_mode or ""),
    }


def _adaptive_budget_controls_changed(change_flags: Mapping[str, bool]) -> bool:
    return any(bool(change_flags.get(key)) for key in ADAPTIVE_BUDGET_CONTROL_CHANGE_KEYS)


def _adaptive_budget_discrete_widening_delta(
    *,
    level: int,
    base_level: int,
    max_level: int,
) -> int:
    normalized_max_level = max(0, int(max_level))
    normalized_base_level = min(max(0, int(base_level)), normalized_max_level)
    normalized_level = min(max(0, int(level)), normalized_max_level)
    if normalized_level <= normalized_base_level:
        return 0
    span = max(1, normalized_max_level - normalized_base_level)
    if span == 1:
        return 1
    scaled_position = (2.0 * float(normalized_level - normalized_base_level)) / float(span)
    # Any step above base widens by at least 1 (POL-10): on the default five-rung ladder with
    # max_ups_per_case=1 the first step used to floor to 0, so the context family changed
    # nothing.
    return min(2, max(1, math.floor(scaled_position)))


def _adaptive_budget_next_tier_index(
    *,
    current_tier_index: int,
    tier_multipliers: Sequence[float],
) -> int:
    multipliers = [float(value) for value in tier_multipliers] or [1.0]
    max_index = max(0, len(multipliers) - 1)
    normalized_current_index = min(max(0, int(current_tier_index)), max_index)
    if normalized_current_index >= max_index:
        return normalized_current_index
    current_multiplier = float(multipliers[normalized_current_index])
    if current_multiplier < 1.2:
        target_multiplier = 1.2
    elif current_multiplier < 1.5:
        target_multiplier = 1.5
    else:
        return max_index
    for candidate_index in range(normalized_current_index + 1, max_index + 1):
        if float(multipliers[candidate_index]) >= (target_multiplier - 1e-9):
            return candidate_index
    return min(max_index, normalized_current_index + 1)


def _build_adaptive_budget_control_profile(
    *,
    tier_index: int,
    base_tier_index: int,
    tier_multipliers: Sequence[float],
    base_propose_budget_sec: float,
    base_openai_max_tokens: int,
    base_helper_max_calls: int,
    base_stage_a_max_requests: int,
    base_stage_a_max_chars: int,
    base_retry_history_max_chars: int,
    base_context_compaction_max_calls: int,
    context_compaction_enabled: bool,
    base_response_mode: str,
    is_openai_patcher: bool,
) -> AdaptiveBudgetControlProfile:
    multipliers = [float(value) for value in tier_multipliers] or [1.0]
    max_tier_index = max(0, len(multipliers) - 1)
    normalized_tier_index = min(max(0, int(tier_index)), max_tier_index)
    normalized_base_tier_index = min(max(0, int(base_tier_index)), max_tier_index)
    multiplier = max(0.0, float(multipliers[normalized_tier_index]))
    widening_delta = _adaptive_budget_discrete_widening_delta(
        level=normalized_tier_index,
        base_level=normalized_base_tier_index,
        max_level=max_tier_index,
    )
    normalized_base_response_mode = str(base_response_mode or "")
    response_mode = normalized_base_response_mode
    if (
        is_openai_patcher
        and normalized_tier_index > normalized_base_tier_index
        and normalized_tier_index == max_tier_index
        and normalized_base_response_mode == "edits"
    ):
        response_mode = "diff"
    return AdaptiveBudgetControlProfile(
        propose_budget_multiplier=multiplier,
        propose_budget_sec=max(0.0, float(base_propose_budget_sec) * multiplier),
        openai_max_tokens=(
            max(1, round(max(0, int(base_openai_max_tokens)) * multiplier))
            if is_openai_patcher and int(base_openai_max_tokens) > 0
            else 0
        ),
        helper_max_calls=(
            0
            if int(base_helper_max_calls) <= 0
            else max(0, int(base_helper_max_calls) + widening_delta)
        ),
        stage_a_max_requests=max(1, int(base_stage_a_max_requests) + widening_delta),
        stage_a_max_chars=max(1, round(max(1, int(base_stage_a_max_chars)) * multiplier)),
        retry_history_max_chars=max(
            0,
            round(max(0, int(base_retry_history_max_chars)) * multiplier),
        ),
        context_compaction_max_calls=(
            max(0, int(base_context_compaction_max_calls) + widening_delta)
            if context_compaction_enabled
            else max(0, int(base_context_compaction_max_calls))
        ),
        response_mode=response_mode,
    )


def _build_adaptive_budget_family_profile(
    *,
    time_level: int,
    tokens_level: int,
    context_level: int,
    base_tier_index: int,
    tier_multipliers: Sequence[float],
    base_propose_budget_sec: float,
    base_openai_max_tokens: int,
    base_helper_max_calls: int,
    base_stage_a_max_requests: int,
    base_stage_a_max_chars: int,
    base_retry_history_max_chars: int,
    base_context_compaction_max_calls: int,
    context_compaction_enabled: bool,
    base_response_mode: str,
    is_openai_patcher: bool,
) -> AdaptiveBudgetControlProfile:
    multipliers = [float(value) for value in tier_multipliers] or [1.0]
    max_tier_index = max(0, len(multipliers) - 1)
    normalized_base_tier_index = min(max(0, int(base_tier_index)), max_tier_index)
    normalized_time_level = min(max(0, int(time_level)), max_tier_index)
    normalized_tokens_level = min(max(0, int(tokens_level)), max_tier_index)
    normalized_context_level = min(max(0, int(context_level)), max_tier_index)
    time_multiplier = max(0.0, float(multipliers[normalized_time_level]))
    tokens_multiplier = max(0.0, float(multipliers[normalized_tokens_level]))
    context_multiplier = max(0.0, float(multipliers[normalized_context_level]))
    context_widening_delta = _adaptive_budget_discrete_widening_delta(
        level=normalized_context_level,
        base_level=normalized_base_tier_index,
        max_level=max_tier_index,
    )
    return AdaptiveBudgetControlProfile(
        propose_budget_multiplier=time_multiplier,
        propose_budget_sec=max(0.0, float(base_propose_budget_sec) * time_multiplier),
        openai_max_tokens=(
            max(1, round(max(0, int(base_openai_max_tokens)) * tokens_multiplier))
            if is_openai_patcher and int(base_openai_max_tokens) > 0
            else 0
        ),
        helper_max_calls=(
            0
            if int(base_helper_max_calls) <= 0
            else max(0, int(base_helper_max_calls) + context_widening_delta)
        ),
        stage_a_max_requests=max(1, int(base_stage_a_max_requests) + context_widening_delta),
        stage_a_max_chars=max(1, round(max(1, int(base_stage_a_max_chars)) * context_multiplier)),
        retry_history_max_chars=max(
            0,
            round(max(0, int(base_retry_history_max_chars)) * context_multiplier),
        ),
        context_compaction_max_calls=(
            max(0, int(base_context_compaction_max_calls) + context_widening_delta)
            if context_compaction_enabled
            else max(0, int(base_context_compaction_max_calls))
        ),
        response_mode=str(base_response_mode or ""),
    )


def _adaptive_budget_profile_for_family_step(
    *,
    family: str,
    time_level: int,
    tokens_level: int,
    context_level: int,
    max_tier_index: int,
    base_tier_index: int,
    tier_multipliers: Sequence[float],
    base_propose_budget_sec: float,
    base_openai_max_tokens: int,
    base_helper_max_calls: int,
    base_stage_a_max_requests: int,
    base_stage_a_max_chars: int,
    base_retry_history_max_chars: int,
    base_context_compaction_max_calls: int,
    context_compaction_enabled: bool,
    base_response_mode: str,
    is_openai_patcher: bool,
) -> tuple[dict[str, int], AdaptiveBudgetControlProfile]:
    next_levels = {
        ACTION_BUDGET_FAMILY_TIME: max(0, int(time_level)),
        ACTION_BUDGET_FAMILY_TOKENS: max(0, int(tokens_level)),
        ACTION_BUDGET_FAMILY_CONTEXT: max(0, int(context_level)),
    }
    if family in next_levels:
        next_levels[family] = min(max_tier_index, next_levels[family] + 1)
    profile = _build_adaptive_budget_family_profile(
        time_level=next_levels[ACTION_BUDGET_FAMILY_TIME],
        tokens_level=next_levels[ACTION_BUDGET_FAMILY_TOKENS],
        context_level=next_levels[ACTION_BUDGET_FAMILY_CONTEXT],
        base_tier_index=base_tier_index,
        tier_multipliers=tier_multipliers,
        base_propose_budget_sec=base_propose_budget_sec,
        base_openai_max_tokens=base_openai_max_tokens,
        base_helper_max_calls=base_helper_max_calls,
        base_stage_a_max_requests=base_stage_a_max_requests,
        base_stage_a_max_chars=base_stage_a_max_chars,
        base_retry_history_max_chars=base_retry_history_max_chars,
        base_context_compaction_max_calls=base_context_compaction_max_calls,
        context_compaction_enabled=context_compaction_enabled,
        base_response_mode=base_response_mode,
        is_openai_patcher=is_openai_patcher,
    )
    return next_levels, profile


def _adaptive_budget_family_availability(
    *,
    current_profile: AdaptiveBudgetControlProfile,
    time_level: int,
    tokens_level: int,
    context_level: int,
    max_tier_index: int,
    base_tier_index: int,
    tier_multipliers: Sequence[float],
    base_propose_budget_sec: float,
    base_openai_max_tokens: int,
    base_helper_max_calls: int,
    base_stage_a_max_requests: int,
    base_stage_a_max_chars: int,
    base_retry_history_max_chars: int,
    base_context_compaction_max_calls: int,
    context_compaction_enabled: bool,
    base_response_mode: str,
    is_openai_patcher: bool,
) -> tuple[
    tuple[str, ...],
    dict[str, str],
    dict[str, AdaptiveBudgetControlProfile],
    dict[str, dict[str, bool]],
    dict[str, dict[str, int]],
]:
    available: list[str] = []
    reasons: dict[str, str] = {}
    profiles: dict[str, AdaptiveBudgetControlProfile] = {}
    changes: dict[str, dict[str, bool]] = {}
    levels_after: dict[str, dict[str, int]] = {}
    current_levels = {
        ACTION_BUDGET_FAMILY_TIME: max(0, int(time_level)),
        ACTION_BUDGET_FAMILY_TOKENS: max(0, int(tokens_level)),
        ACTION_BUDGET_FAMILY_CONTEXT: max(0, int(context_level)),
    }
    for family in BUDGET_POLICY_FAMILIES:
        if current_levels[family] >= max_tier_index:
            reasons[family] = "family_level_ceiling"
            continue
        next_levels, target_profile = _adaptive_budget_profile_for_family_step(
            family=family,
            time_level=time_level,
            tokens_level=tokens_level,
            context_level=context_level,
            max_tier_index=max_tier_index,
            base_tier_index=base_tier_index,
            tier_multipliers=tier_multipliers,
            base_propose_budget_sec=base_propose_budget_sec,
            base_openai_max_tokens=base_openai_max_tokens,
            base_helper_max_calls=base_helper_max_calls,
            base_stage_a_max_requests=base_stage_a_max_requests,
            base_stage_a_max_chars=base_stage_a_max_chars,
            base_retry_history_max_chars=base_retry_history_max_chars,
            base_context_compaction_max_calls=base_context_compaction_max_calls,
            context_compaction_enabled=context_compaction_enabled,
            base_response_mode=base_response_mode,
            is_openai_patcher=is_openai_patcher,
        )
        change_flags = _adaptive_budget_control_changes(
            current=current_profile,
            target=target_profile,
        )
        if _adaptive_budget_controls_changed(change_flags):
            available.append(family)
            profiles[family] = target_profile
            changes[family] = change_flags
            levels_after[family] = next_levels
            continue
        if family == ACTION_BUDGET_FAMILY_TOKENS and (
            not is_openai_patcher or int(base_openai_max_tokens) <= 0
        ):
            reasons[family] = "tokens_unsupported"
        else:
            reasons[family] = "family_no_control_change"
    return tuple(available), reasons, profiles, changes, levels_after


def _adaptive_budget_union_change_flags(
    change_maps: Mapping[str, Mapping[str, bool]],
) -> dict[str, bool]:
    merged = dict.fromkeys(ADAPTIVE_BUDGET_CONTROL_CHANGE_KEYS, False)
    for change_flags in change_maps.values():
        for key in ADAPTIVE_BUDGET_CONTROL_CHANGE_KEYS:
            merged[key] = bool(merged[key] or bool(change_flags.get(key)))
    return merged


def _adaptive_budget_level_step_delta(
    *,
    current_level: int,
    max_tier_index: int,
    tier_multipliers: Sequence[float],
) -> float:
    multipliers = [float(value) for value in tier_multipliers] or [1.0]
    if not multipliers:
        return 0.0
    max_index = min(max_tier_index, len(multipliers) - 1)
    current_index = min(max(0, int(current_level)), max_index)
    next_index = min(max_index, current_index + 1)
    return max(0.0, float(multipliers[next_index]) - float(multipliers[current_index]))


def _adaptive_budget_level_multiplier(
    *,
    level: int,
    tier_multipliers: Sequence[float],
    fallback: float,
) -> float:
    multipliers = [float(value) for value in tier_multipliers]
    if not multipliers:
        return max(0.0, float(fallback))
    max_index = len(multipliers) - 1
    normalized_level = min(max(0, int(level)), max_index)
    return max(0.0, float(multipliers[normalized_level]))


def _adaptive_budget_recorded_multiplier(
    *,
    family: str,
    family_levels: Mapping[str, int],
    tier_multipliers: Sequence[float],
    fallback: float,
) -> float:
    normalized_family = str(family or "")
    if normalized_family in BUDGET_POLICY_FAMILIES:
        return _adaptive_budget_level_multiplier(
            level=int(family_levels.get(normalized_family, 0)),
            tier_multipliers=tier_multipliers,
            fallback=fallback,
        )
    active_multipliers = [
        _adaptive_budget_level_multiplier(
            level=int(family_levels.get(family_name, 0)),
            tier_multipliers=tier_multipliers,
            fallback=fallback,
        )
        for family_name in BUDGET_POLICY_FAMILIES
    ]
    return max(active_multipliers, default=max(0.0, float(fallback)))


def _apply_adaptive_budget_profile_to_patcher(
    *,
    patcher: Any,
    applied_profile: AdaptiveBudgetControlProfile,
    base_propose_budget_sec: float,
    base_openai_max_tokens: int,
    base_response_mode: str,
    is_openai_patcher: bool,
) -> tuple[Any, AdaptiveBudgetControlProfile, dict[str, str]]:
    if not is_openai_patcher:
        return patcher, applied_profile, {}

    override_kwargs = {
        "max_propose_sec": float(applied_profile.propose_budget_sec),
        "max_tokens": int(applied_profile.openai_max_tokens),
        "response_mode": str(applied_profile.response_mode or ""),
    }
    try:
        return replace(patcher, **override_kwargs), applied_profile, {}
    except Exception as exc:
        error_text = f"{type(exc).__name__}: {exc}"
        reverted_profile = replace(
            applied_profile,
            propose_budget_sec=max(0.0, float(base_propose_budget_sec)),
            openai_max_tokens=max(0, int(base_openai_max_tokens)),
            response_mode=str(base_response_mode or ""),
        )
        return (
            patcher,
            reverted_profile,
            {
                "max_propose_sec": error_text,
                "max_tokens": error_text,
                "response_mode": error_text,
            },
        )


def _adaptive_policy_local_accounting(
    *,
    llm_call_count: int = 0,
    tokens_prompt: int = 0,
    tokens_completion: int = 0,
    tokens_total: int | None = None,
    wall_sec: float = 0.0,
) -> AdaptivePolicyLocalAccounting:
    normalized_llm_calls = _nonnegative_int(llm_call_count, default=0)
    normalized_tokens_prompt = _nonnegative_int(tokens_prompt, default=0)
    normalized_tokens_completion = _nonnegative_int(tokens_completion, default=0)
    normalized_tokens_total = _nonnegative_int(
        tokens_total,
        default=normalized_tokens_prompt + normalized_tokens_completion,
    )
    return AdaptivePolicyLocalAccounting(
        llm_call_count=normalized_llm_calls,
        tokens_prompt=normalized_tokens_prompt,
        tokens_completion=normalized_tokens_completion,
        tokens_total=normalized_tokens_total,
        wall_sec=max(0.0, float(wall_sec)),
    )


def _adaptive_online_local_reward_components(
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
    enable_auxiliary_rewards: bool = False,
    aux_pytest_fail_reduction: float = 0.0,
    aux_evidence_novelty_ratio: float = 0.0,
    aux_compaction_ratio: float = 0.0,
    aux_informativeness_alignment: float = 0.0,
    aux_pytest_reduction_weight: float = ADAPTIVE_AUX_REWARD_PYTEST_REDUCTION_WEIGHT_DEFAULT,
    aux_evidence_novelty_weight: float = ADAPTIVE_AUX_REWARD_EVIDENCE_NOVELTY_WEIGHT_DEFAULT,
    aux_compaction_efficiency_weight: float = ADAPTIVE_AUX_REWARD_COMPACTION_EFFICIENCY_WEIGHT_DEFAULT,
    aux_informativeness_alignment_weight: float = ADAPTIVE_AUX_REWARD_INFORMATIVENESS_ALIGNMENT_WEIGHT_DEFAULT,
) -> tuple[float, float, float, float, float, float, float, float, float, float, float, float]:
    success_hint = max(0.0, float(success_hint_scale)) if bool(success) else 0.0
    progress_score = max(0.0, min(1.0, float(adaptive_progress_score)))
    progress_hint = (
        max(0.0, float(progress_hint_scale)) * max(0.0, float(eta_progress)) * progress_score
    )
    helper_materialization_bonus = (
        max(0.0, float(helper_materialization_bonus_scale)) * max(0.0, float(eta_progress))
        if helper_materialized_context
        else 0.0
    )
    token_penalty = max(0.0, float(lambda_tokens)) * (
        max(0.0, float(local_tokens_total)) / max(1.0, float(token_scale))
    )
    latency_penalty = max(0.0, float(mu_latency)) * (
        max(0.0, float(local_wall_sec)) / max(1.0, float(latency_scale))
    )
    stagnation_penalty = (
        max(0.0, float(stagnation_penalty_scale))
        * max(0.0, float(kappa_stagnation))
        * min(1.0, max(0.0, float(adaptive_no_progress_streak)) / 3.0)
    )
    # Auxiliary reward signals (additive, gated by enable flag).
    clamped_aux_pytest = max(0.0, min(1.0, float(aux_pytest_fail_reduction)))
    clamped_aux_novelty = max(0.0, min(1.0, float(aux_evidence_novelty_ratio)))
    clamped_aux_compaction = max(0.0, min(1.0, float(aux_compaction_ratio)))
    clamped_aux_informativeness = max(0.0, min(1.0, float(aux_informativeness_alignment)))
    aux_total = 0.0
    if enable_auxiliary_rewards:
        aux_total = (
            aux_pytest_reduction_weight * clamped_aux_pytest
            + aux_evidence_novelty_weight * clamped_aux_novelty
            + aux_compaction_efficiency_weight * clamped_aux_compaction
            + aux_informativeness_alignment_weight * clamped_aux_informativeness
        )
    reward = (
        success_hint
        + progress_hint
        + helper_materialization_bonus
        + aux_total
        - token_penalty
        - latency_penalty
        - stagnation_penalty
    )
    return (
        reward,
        success_hint,
        progress_hint,
        token_penalty,
        latency_penalty,
        stagnation_penalty,
        helper_materialization_bonus,
        aux_total,
        clamped_aux_pytest,
        clamped_aux_novelty,
        clamped_aux_compaction,
        clamped_aux_informativeness,
    )


def _adaptive_budget_feature_vector(
    *,
    attempt: int,
    max_attempts: int,
    timed_out: bool,
    failure_signature_repeat: bool,
    stagnation_streak: int,
    helper_calls_used: int,
    helper_calls_without_context: int,
    adaptive_no_progress_streak: int,
    adaptive_no_progress_token_sum: int,
    helper_budget_exhausted: bool,
    helper_saturation_guarded: bool,
    supports_context_request: bool,
    snapshot_available: bool,
    pregate_result: PreGateResult,
    context_compaction_enabled: bool,
    supports_context_compaction: bool,
    context_compaction_calls_used: int,
    context_compaction_max_calls: int,
    last_context_compaction_requested: bool,
    last_context_compaction_invoked: bool,
    last_context_compaction_success: bool,
    last_context_compaction_ratio: float,
    last_context_compaction_trigger: str,
    current_tier_index: int,
    max_tier_index: int,
    ups_used: int,
    max_ups_per_case: int,
    base_propose_budget_sec: float,
    next_tier_changes: Mapping[str, bool],
    next_tier_multiplier_delta: float,
    last_helper_need_more_context: bool | None,
    last_helper_decision_reason: str,
    budget_architecture: str = BUDGET_POLICY_ARCH_TIER_BINARY,
    current_time_level: int | None = None,
    current_tokens_level: int | None = None,
    current_context_level: int | None = None,
    can_raise_time: bool = False,
    can_raise_tokens: bool = False,
    can_raise_context: bool = False,
    time_delta_magnitude: float = 0.0,
    tokens_delta_magnitude: float = 0.0,
    context_delta_magnitude: float = 0.0,
    toon_encoding_active: bool = False,
) -> dict[str, float]:
    max_attempts_safe = max(1, int(max_attempts))
    attempt_safe = max(1, int(attempt))
    helper_calls_used_safe = max(0, int(helper_calls_used))
    low_yield_ratio = (
        max(0, int(helper_calls_without_context)) / float(helper_calls_used_safe)
        if helper_calls_used_safe > 0
        else 0.0
    )
    context_compaction_calls_used_safe = max(0, int(context_compaction_calls_used))
    context_compaction_max_calls_safe = max(0, int(context_compaction_max_calls))
    context_compaction_calls_used_frac = (
        min(1.0, context_compaction_calls_used_safe / float(context_compaction_max_calls_safe))
        if context_compaction_max_calls_safe > 0
        else 0.0
    )
    context_compaction_calls_remaining_frac = (
        max(
            0.0,
            min(
                1.0,
                (context_compaction_max_calls_safe - context_compaction_calls_used_safe)
                / float(context_compaction_max_calls_safe),
            ),
        )
        if context_compaction_max_calls_safe > 0
        else 0.0
    )
    max_tier_safe = max(0, int(max_tier_index))
    tier_index_safe = min(max(0, int(current_tier_index)), max_tier_safe)
    tier_frac = (
        tier_index_safe / float(max_tier_safe) if max_tier_safe > 0 else float(tier_index_safe > 0)
    )
    max_ups_safe = max(0, int(max_ups_per_case))
    ups_used_safe = max(0, int(ups_used))
    ups_used_frac = min(1.0, ups_used_safe / float(max_ups_safe)) if max_ups_safe > 0 else 0.0
    ups_remaining_frac = (
        max(0.0, min(1.0, (max_ups_safe - ups_used_safe) / float(max_ups_safe)))
        if max_ups_safe > 0
        else 0.0
    )
    last_decision_reason = str(last_helper_decision_reason or "").strip()
    pregate_exception_type_normalized = str(pregate_result.exception_type).strip().lower()
    last_compaction_trigger = str(last_context_compaction_trigger or "")
    return {
        "bias": 1.0,
        "attempt_frac": float(attempt_safe / float(max_attempts_safe)),
        "timed_out": 1.0 if timed_out else 0.0,
        "failure_signature_repeat": 1.0 if failure_signature_repeat else 0.0,
        "stagnation_streak": float(max(0, int(stagnation_streak))),
        "helper_calls_used": float(helper_calls_used_safe),
        "helper_low_yield_ratio": float(low_yield_ratio),
        "adaptive_no_progress_streak": float(max(0, int(adaptive_no_progress_streak))),
        "adaptive_no_progress_token_sum_k": float(
            max(0, int(adaptive_no_progress_token_sum)) / 1000.0
        ),
        "helper_budget_exhausted": 1.0 if helper_budget_exhausted else 0.0,
        "helper_saturation_guarded": 1.0 if helper_saturation_guarded else 0.0,
        "supports_context_request": 1.0 if supports_context_request else 0.0,
        "snapshot_available": 1.0 if snapshot_available else 0.0,
        "pregate_score": float(max(0.0, min(1.0, pregate_result.score))),
        "pregate_fired": 1.0 if pregate_result.should_skip else 0.0,
        "pregate_exception_is_assertion": 1.0
        if pregate_exception_type_normalized == "assertionerror"
        else 0.0,
        "pregate_exception_is_timeout": 1.0
        if pregate_exception_type_normalized == "timeouterror"
        else 0.0,
        "pregate_exception_is_syntax": 1.0
        if pregate_exception_type_normalized in {"syntaxerror", "indentationerror", "taberror"}
        else 0.0,
        "context_compaction_enabled": 1.0 if context_compaction_enabled else 0.0,
        "supports_context_compaction": 1.0 if supports_context_compaction else 0.0,
        "context_compaction_calls_used_frac": float(context_compaction_calls_used_frac),
        "context_compaction_calls_remaining_frac": float(context_compaction_calls_remaining_frac),
        "last_context_compaction_requested": 1.0 if last_context_compaction_requested else 0.0,
        "last_context_compaction_invoked": 1.0 if last_context_compaction_invoked else 0.0,
        "last_context_compaction_success": 1.0 if last_context_compaction_success else 0.0,
        "last_context_compaction_ratio": float(max(0.0, min(1.0, last_context_compaction_ratio))),
        "last_context_compaction_trigger_stage_a": 1.0
        if last_compaction_trigger == "stage_a_requested"
        else 0.0,
        "last_context_compaction_trigger_overflow": 1.0
        if last_compaction_trigger == "prompt_overflow"
        else 0.0,
        "last_context_compaction_trigger_always": 1.0
        if last_compaction_trigger == "mode_always"
        else 0.0,
        "current_budget_tier_frac": float(max(0.0, min(1.0, tier_frac))),
        "can_increase_budget_tier": 1.0 if tier_index_safe < max_tier_safe else 0.0,
        "budget_arch_hierarchical_v1": 1.0
        if budget_architecture == BUDGET_POLICY_ARCH_HIERARCHICAL_V1
        else 0.0,
        "current_budget_time_level_frac": float(
            max(
                0.0,
                min(
                    1.0,
                    max(
                        0,
                        int(
                            current_time_level
                            if current_time_level is not None
                            else tier_index_safe
                        ),
                    )
                    / float(max_tier_safe),
                ),
            )
        )
        if max_tier_safe > 0
        else float(
            max(0, int(current_time_level if current_time_level is not None else tier_index_safe))
            > 0
        ),
        "current_budget_tokens_level_frac": float(
            max(
                0.0,
                min(
                    1.0,
                    max(
                        0,
                        int(
                            current_tokens_level
                            if current_tokens_level is not None
                            else tier_index_safe
                        ),
                    )
                    / float(max_tier_safe),
                ),
            )
        )
        if max_tier_safe > 0
        else float(
            max(
                0,
                int(current_tokens_level if current_tokens_level is not None else tier_index_safe),
            )
            > 0
        ),
        "current_budget_context_level_frac": float(
            max(
                0.0,
                min(
                    1.0,
                    max(
                        0,
                        int(
                            current_context_level
                            if current_context_level is not None
                            else tier_index_safe
                        ),
                    )
                    / float(max_tier_safe),
                ),
            )
        )
        if max_tier_safe > 0
        else float(
            max(
                0,
                int(
                    current_context_level if current_context_level is not None else tier_index_safe
                ),
            )
            > 0
        ),
        "budget_can_raise_time": 1.0 if can_raise_time else 0.0,
        "budget_can_raise_tokens": 1.0 if can_raise_tokens else 0.0,
        "budget_can_raise_context": 1.0 if can_raise_context else 0.0,
        "budget_time_delta_magnitude": float(max(0.0, float(time_delta_magnitude))),
        "budget_tokens_delta_magnitude": float(max(0.0, float(tokens_delta_magnitude))),
        "budget_context_delta_magnitude": float(max(0.0, float(context_delta_magnitude))),
        "budget_ups_used_frac": float(max(0.0, min(1.0, ups_used_frac))),
        "budget_ups_remaining_frac": float(max(0.0, min(1.0, ups_remaining_frac))),
        "base_propose_budget_sec": float(max(0.0, float(base_propose_budget_sec))),
        "budget_can_raise_propose_budget": 1.0
        if bool(next_tier_changes.get("propose_budget"))
        else 0.0,
        "budget_can_raise_max_tokens": 1.0 if bool(next_tier_changes.get("max_tokens")) else 0.0,
        "budget_can_raise_helper_allowance": 1.0
        if bool(next_tier_changes.get("helper_allowance"))
        else 0.0,
        "budget_can_raise_stage_a_requests": 1.0
        if bool(next_tier_changes.get("stage_a_requests"))
        else 0.0,
        "budget_can_raise_stage_a_chars": 1.0
        if bool(next_tier_changes.get("stage_a_chars"))
        else 0.0,
        "budget_can_raise_retry_history_budget": 1.0
        if bool(next_tier_changes.get("retry_history_budget"))
        else 0.0,
        "budget_can_raise_compaction_allowance": 1.0
        if bool(next_tier_changes.get("compaction_allowance"))
        else 0.0,
        "budget_can_switch_response_mode": 1.0
        if bool(next_tier_changes.get("response_mode"))
        else 0.0,
        "budget_next_tier_multiplier_delta": float(max(0.0, float(next_tier_multiplier_delta))),
        "last_helper_need_more_context": 1.0 if last_helper_need_more_context is True else 0.0,
        "last_helper_invalid_response": 1.0
        if last_decision_reason == "llm_invalid_response_no_helper"
        else 0.0,
        "last_helper_declined": 1.0 if last_decision_reason == "llm_declined_helper" else 0.0,
        "toon_encoding_active": 1.0 if toon_encoding_active else 0.0,
    }


def _adaptive_budget_local_reward_components(
    *,
    success: bool,
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
    rho_budget_exceeded: float,
    request_budget_exceeded: bool,
    success_hint_scale: float = ADAPTIVE_LOCAL_REWARD_SUCCESS_HINT,
    progress_hint_scale: float = ADAPTIVE_LOCAL_REWARD_PROGRESS_HINT_SCALE,
    stagnation_penalty_scale: float = ADAPTIVE_LOCAL_REWARD_STAGNATION_PENALTY_SCALE,
    enable_auxiliary_rewards: bool = False,
    aux_pytest_fail_reduction: float = 0.0,
    aux_evidence_novelty_ratio: float = 0.0,
    aux_pytest_reduction_weight: float = ADAPTIVE_AUX_REWARD_PYTEST_REDUCTION_WEIGHT_DEFAULT,
    aux_evidence_novelty_weight: float = ADAPTIVE_AUX_REWARD_EVIDENCE_NOVELTY_WEIGHT_DEFAULT,
) -> tuple[float, float, float, float, float, float, float, float, float, float]:
    success_hint = max(0.0, float(success_hint_scale)) if bool(success) else 0.0
    progress_score = max(0.0, min(1.0, float(adaptive_progress_score)))
    progress_hint = (
        max(0.0, float(progress_hint_scale)) * max(0.0, float(eta_progress)) * progress_score
    )
    token_penalty = max(0.0, float(lambda_tokens)) * (
        max(0.0, float(local_tokens_total)) / max(1.0, float(token_scale))
    )
    latency_penalty = max(0.0, float(mu_latency)) * (
        max(0.0, float(local_wall_sec)) / max(1.0, float(latency_scale))
    )
    stagnation_penalty = (
        max(0.0, float(stagnation_penalty_scale))
        * max(0.0, float(kappa_stagnation))
        * min(1.0, max(0.0, float(adaptive_no_progress_streak)) / 3.0)
    )
    budget_exceeded_penalty = (
        max(0.0, float(rho_budget_exceeded)) if request_budget_exceeded else 0.0
    )
    # Auxiliary reward signals (budget subset: pytest + evidence novelty only).
    clamped_aux_pytest = max(0.0, min(1.0, float(aux_pytest_fail_reduction)))
    clamped_aux_novelty = max(0.0, min(1.0, float(aux_evidence_novelty_ratio)))
    aux_total = 0.0
    if enable_auxiliary_rewards:
        aux_total = (
            aux_pytest_reduction_weight * clamped_aux_pytest
            + aux_evidence_novelty_weight * clamped_aux_novelty
        )
    reward = (
        success_hint
        + progress_hint
        + aux_total
        - token_penalty
        - latency_penalty
        - stagnation_penalty
        - budget_exceeded_penalty
    )
    return (
        reward,
        success_hint,
        progress_hint,
        token_penalty,
        latency_penalty,
        stagnation_penalty,
        budget_exceeded_penalty,
        aux_total,
        clamped_aux_pytest,
        clamped_aux_novelty,
    )
