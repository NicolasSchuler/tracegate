"""Decision and event records of the adaptive helper-invocation policy.

Split out of ``evals.adaptive_online_policy`` (which re-exports them) to keep the controller
module small. ``PolicyEvent.to_json`` defines the policy-event JSONL schema.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

# v5 adds the nested ``decision_model`` block (role, answers, applied effect).
EVENT_SCHEMA_VERSION = 5


@dataclass(frozen=True)
class PolicyDecision:
    """Snapshot of a single helper-invocation decision by the policy."""

    decision_id: str
    mode: str
    algorithm: str
    decision_index: int
    created_at_utc: str
    selected_action: str
    applied_action: str
    default_action: str
    propensity: float
    decision_strategy: str
    scores: dict[str, float]
    feature_values: dict[str, float]
    available_actions: tuple[str, ...]
    unavailable_action_reasons: dict[str, str]
    configured_action_score_biases: dict[str, float]
    feature_scaling_caps: dict[str, float]
    feature_scaling_history_sizes: dict[str, int]
    feature_scaling_fallback_to_v1: bool
    configured_alpha: float
    configured_epsilon: float
    configured_decay: float
    effective_epsilon: float
    warmup_active: bool
    decision_model: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyEvent:
    """Logged outcome record pairing a decision with its observed reward."""

    run_id: str
    case_id: str
    condition: str
    attempt: int
    note: str
    decision_id: str
    mode: str
    algorithm: str
    decision_index: int
    selected_action: str
    applied_action: str
    behavior_action: str
    default_action: str
    propensity: float
    behavior_propensity: float
    decision_strategy: str
    feature_scaling_profile: str
    feature_scaling_caps: dict[str, float]
    feature_scaling_history_sizes: dict[str, int]
    feature_scaling_fallback_to_v1: bool
    configured_alpha: float
    configured_epsilon: float
    configured_decay: float
    effective_epsilon: float
    warmup_active: bool
    scores: dict[str, float]
    feature_values: dict[str, float]
    available_actions: tuple[str, ...]
    unavailable_action_reasons: dict[str, str]
    configured_action_score_biases: dict[str, float]
    reward_scope: str
    reward_success_hint_scale: float
    reward_progress_hint_scale: float
    reward_stagnation_penalty_scale: float
    reward_helper_materialization_bonus_scale: float
    reward: float
    reward_success_hint: float
    reward_progress_hint: float
    reward_progress_score: float
    reward_token_penalty: float
    reward_latency_penalty: float
    reward_stagnation_penalty: float
    reward_helper_materialization_bonus: float
    reward_delayed_rescue_bonus: float
    reward_credit_resolved_attempt: int
    reward_credit_horizon_attempts: int
    reward_credit_resolution_reason: str
    success: bool
    local_llm_call_count: int
    local_tokens_prompt: int
    local_tokens_completion: int
    local_tokens_total: int
    local_wall_sec: float
    override_reason: str
    model_updated: bool
    update_reason: str
    nan_replacement_counts: dict[str, int]
    created_at_utc: str
    decision_model: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> dict[str, Any]:
        """Serialize the event to a JSON-compatible dict."""
        return {
            "schema_version": EVENT_SCHEMA_VERSION,
            "run_id": self.run_id,
            "case_id": self.case_id,
            "condition": self.condition,
            "attempt": self.attempt,
            "note": self.note,
            "decision_id": self.decision_id,
            "mode": self.mode,
            "algorithm": self.algorithm,
            "decision_index": self.decision_index,
            "selected_action": self.selected_action,
            "applied_action": self.applied_action,
            "behavior_action": self.behavior_action,
            "default_action": self.default_action,
            "propensity": self.propensity,
            "behavior_propensity": self.behavior_propensity,
            "decision_strategy": self.decision_strategy,
            "feature_scaling_profile": self.feature_scaling_profile,
            "feature_scaling_caps": {k: float(v) for k, v in self.feature_scaling_caps.items()},
            "feature_scaling_history_sizes": {
                k: int(v) for k, v in self.feature_scaling_history_sizes.items()
            },
            "feature_scaling_fallback_to_v1": bool(self.feature_scaling_fallback_to_v1),
            "configured_alpha": self.configured_alpha,
            "configured_epsilon": self.configured_epsilon,
            "configured_decay": self.configured_decay,
            "effective_epsilon": self.effective_epsilon,
            "warmup_active": bool(self.warmup_active),
            "scores": dict(self.scores),
            "feature_values": dict(self.feature_values),
            "available_actions": list(self.available_actions),
            "unavailable_action_reasons": dict(self.unavailable_action_reasons),
            "configured_action_score_biases": dict(self.configured_action_score_biases),
            "reward_scope": self.reward_scope,
            "reward_success_hint_scale": self.reward_success_hint_scale,
            "reward_progress_hint_scale": self.reward_progress_hint_scale,
            "reward_stagnation_penalty_scale": self.reward_stagnation_penalty_scale,
            "reward_helper_materialization_bonus_scale": (
                self.reward_helper_materialization_bonus_scale
            ),
            "reward": self.reward,
            "reward_success_hint": self.reward_success_hint,
            "reward_progress_hint": self.reward_progress_hint,
            "reward_progress_score": self.reward_progress_score,
            "reward_token_penalty": self.reward_token_penalty,
            "reward_latency_penalty": self.reward_latency_penalty,
            "reward_stagnation_penalty": self.reward_stagnation_penalty,
            "reward_helper_materialization_bonus": self.reward_helper_materialization_bonus,
            "reward_delayed_rescue_bonus": self.reward_delayed_rescue_bonus,
            "reward_credit_resolved_attempt": self.reward_credit_resolved_attempt,
            "reward_credit_horizon_attempts": self.reward_credit_horizon_attempts,
            "reward_credit_resolution_reason": self.reward_credit_resolution_reason,
            "success": self.success,
            "local_llm_call_count": self.local_llm_call_count,
            "local_tokens_prompt": self.local_tokens_prompt,
            "local_tokens_completion": self.local_tokens_completion,
            "local_tokens_total": self.local_tokens_total,
            "local_wall_sec": self.local_wall_sec,
            "override_reason": self.override_reason,
            "model_updated": self.model_updated,
            "update_reason": self.update_reason,
            "nan_replacement_counts": dict(self.nan_replacement_counts),
            "decision_model": dict(self.decision_model),
            "created_at_utc": self.created_at_utc,
        }
