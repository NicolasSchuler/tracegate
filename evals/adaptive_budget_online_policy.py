"""Adaptive online budget-escalation policy using LinUCB.

Implements a contextual-bandit controller that decides per-attempt whether
to keep or increase the request budget.  Supports ``off`` / ``shadow`` /
``on`` modes with epsilon-greedy exploration and warm-up uniform sampling.
"""

from __future__ import annotations

import random
import threading
import uuid
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from . import _json as json
from . import linucb_core as core
from ._policy_utils import (
    coerce_float as _coerce_float,
)
from ._policy_utils import (
    coerce_nonnegative_int as _coerce_nonnegative_int,
)
from ._policy_utils import (
    policy_effective_epsilon as _policy_effective_epsilon,
)
from ._policy_utils import (
    safe_float_tracked as _safe_float_tracked,
)
from ._policy_utils import (
    utc_now as _utc_now,
)
from .policy_feature_scaling import (
    DEFAULT_POLICY_FEATURE_SCALING_HISTORY_LIMIT,
    DEFAULT_POLICY_FEATURE_SCALING_MIN_SAMPLES,
    DEFAULT_POLICY_FEATURE_SCALING_PROFILE,
    POLICY_FEATURE_SCALING_PROFILES,
    PercentileScaler,
    is_percentile_feature_scaling_profile,
    normalize_feature_scaling_profile,
    scale_feature_mapping,
)
from .seed_derivation import derive_seed

ACTION_KEEP_BUDGET = "keep_budget"
ACTION_INCREASE_BUDGET = "increase_budget"
BUDGET_POLICY_ACTIONS = (ACTION_KEEP_BUDGET, ACTION_INCREASE_BUDGET)
ACTION_BUDGET_FAMILY_TIME = "time"
ACTION_BUDGET_FAMILY_TOKENS = "tokens"
ACTION_BUDGET_FAMILY_CONTEXT = "context"
BUDGET_POLICY_FAMILIES = (
    ACTION_BUDGET_FAMILY_TIME,
    ACTION_BUDGET_FAMILY_TOKENS,
    ACTION_BUDGET_FAMILY_CONTEXT,
)
BUDGET_POLICY_ARCH_TIER_BINARY = "tier_binary"
BUDGET_POLICY_ARCH_HIERARCHICAL_V1 = "hierarchical_v1"
BUDGET_POLICY_ARCH_BWK_V1 = "bwk_v1"
BUDGET_POLICY_ARCHITECTURES = {
    BUDGET_POLICY_ARCH_TIER_BINARY,
    BUDGET_POLICY_ARCH_HIERARCHICAL_V1,
    BUDGET_POLICY_ARCH_BWK_V1,
}
BUDGET_POLICY_MODES = {"off", "shadow", "on"}
BUDGET_POLICY_ALGOS = {"linucb", "thompson"}
STATE_SCHEMA_VERSION = 3
DEFAULT_ADAPTIVE_BUDGET_ONLINE_DECAY = 1.0
"""Decay is opt-in (decision D6, POL-01); below 1 it only discounts the updated arm."""
EVENT_SCHEMA_VERSION = 5


@dataclass(frozen=True)
class BudgetPolicyConfig:
    """Immutable configuration for the adaptive budget-escalation policy."""

    mode: str = "off"
    algorithm: str = "linucb"
    architecture: str = BUDGET_POLICY_ARCH_TIER_BINARY
    alpha: float = 0.6
    epsilon: float = 0.08
    l2: float = 1.0
    warmup_decisions: int = 20
    save_every: int = 50
    decay: float = DEFAULT_ADAPTIVE_BUDGET_ONLINE_DECAY
    feature_scaling_profile: str = DEFAULT_POLICY_FEATURE_SCALING_PROFILE
    init_state_path: Path | None = None
    state_path: Path | None = None
    events_path: Path | None = None
    bwk_run_token_budget: int | None = None
    random_seed: int | None = None

    @classmethod
    def normalize(
        cls,
        *,
        mode: str = "off",
        algorithm: str = "linucb",
        architecture: str = BUDGET_POLICY_ARCH_TIER_BINARY,
        alpha: float = 0.6,
        epsilon: float = 0.08,
        l2: float = 1.0,
        warmup_decisions: int = 20,
        save_every: int = 50,
        decay: float = DEFAULT_ADAPTIVE_BUDGET_ONLINE_DECAY,
        feature_scaling_profile: str = DEFAULT_POLICY_FEATURE_SCALING_PROFILE,
        init_state_path: Path | None = None,
        state_path: Path | None = None,
        events_path: Path | None = None,
        bwk_run_token_budget: int | None = None,
        random_seed: int | None = None,
    ) -> BudgetPolicyConfig:
        """Validate and clamp raw parameters into a normalised config."""
        mode_norm = str(mode).strip().lower()
        if mode_norm not in BUDGET_POLICY_MODES:
            mode_norm = "off"
        algo_norm = str(algorithm).strip().lower()
        if algo_norm not in BUDGET_POLICY_ALGOS:
            algo_norm = "linucb"
        architecture_norm = str(architecture).strip().lower()
        if architecture_norm not in BUDGET_POLICY_ARCHITECTURES:
            architecture_norm = BUDGET_POLICY_ARCH_TIER_BINARY
        alpha_norm = max(0.0, _coerce_float(alpha, default=0.6))
        epsilon_norm = min(1.0, max(0.0, _coerce_float(epsilon, default=0.08)))
        l2_norm = max(1e-9, _coerce_float(l2, default=1.0))
        warmup_norm = _coerce_nonnegative_int(warmup_decisions, default=20)
        save_every_norm = max(1, _coerce_nonnegative_int(save_every, default=50))
        decay_norm = min(
            1.0, max(1e-9, _coerce_float(decay, default=DEFAULT_ADAPTIVE_BUDGET_ONLINE_DECAY))
        )
        feature_scaling_profile_norm = normalize_feature_scaling_profile(feature_scaling_profile)
        if feature_scaling_profile_norm not in POLICY_FEATURE_SCALING_PROFILES:
            feature_scaling_profile_norm = DEFAULT_POLICY_FEATURE_SCALING_PROFILE
        seed_norm: int | None = None
        if random_seed is not None:
            try:
                seed_norm = int(random_seed)
            except (TypeError, ValueError):
                seed_norm = None
        run_token_budget_norm: int | None = None
        if bwk_run_token_budget is not None:
            try:
                candidate = int(bwk_run_token_budget)
            except (TypeError, ValueError):
                candidate = 0
            run_token_budget_norm = max(0, candidate)
        return cls(
            mode=mode_norm,
            algorithm=algo_norm,
            architecture=architecture_norm,
            alpha=alpha_norm,
            epsilon=epsilon_norm,
            l2=l2_norm,
            warmup_decisions=warmup_norm,
            save_every=save_every_norm,
            decay=decay_norm,
            feature_scaling_profile=feature_scaling_profile_norm,
            init_state_path=Path(init_state_path) if init_state_path is not None else None,
            state_path=Path(state_path) if state_path is not None else None,
            events_path=Path(events_path) if events_path is not None else None,
            bwk_run_token_budget=run_token_budget_norm,
            random_seed=seed_norm,
        )


@dataclass(frozen=True)
class BudgetPolicyDecision:
    """Snapshot of a single budget-escalation decision by the policy."""

    decision_id: str
    mode: str
    algorithm: str
    architecture: str
    decision_index: int
    created_at_utc: str
    selected_action: str
    applied_action: str
    default_action: str
    propensity: float
    decision_strategy: str
    scores: dict[str, float]
    feature_values: dict[str, float]
    feature_scaling_caps: dict[str, float]
    feature_scaling_history_sizes: dict[str, int]
    feature_scaling_fallback_to_v1: bool
    selected_family: str
    applied_family: str
    available_families: tuple[str, ...]
    unavailable_family_reasons: dict[str, str]
    family_propensity: float
    family_decision_strategy: str
    family_scores: dict[str, float]
    family_feature_values: dict[str, float]
    family_feature_scaling_caps: dict[str, float]
    family_feature_scaling_history_sizes: dict[str, int]
    family_feature_scaling_fallback_to_v1: bool
    configured_alpha: float
    configured_epsilon: float
    configured_decay: float
    effective_epsilon: float
    warmup_active: bool
    bwk_run_token_budget: int = 0
    bwk_remaining_token_budget: int = 0
    bwk_dual_variable: float = 0.0
    bwk_budget_exhausted: bool = False


@dataclass(frozen=True)
class BudgetPolicyEvent:
    """Logged outcome record pairing a budget decision with its reward."""

    run_id: str
    case_id: str
    condition: str
    attempt: int
    note: str
    decision_id: str
    mode: str
    algorithm: str
    architecture: str
    decision_index: int
    selected_action: str
    applied_action: str
    behavior_action: str
    default_action: str
    selected_family: str
    applied_family: str
    behavior_family: str
    available_families: tuple[str, ...]
    unavailable_family_reasons: dict[str, str]
    family_propensity: float
    family_decision_strategy: str
    selected_multiplier: float
    applied_multiplier: float
    default_multiplier: float
    default_controls: dict[str, Any]
    selected_controls: dict[str, Any]
    applied_controls: dict[str, Any]
    selected_controls_changed: dict[str, bool]
    applied_controls_changed: dict[str, bool]
    partial_override: bool
    partial_override_errors: dict[str, str]
    tier_index_before: int
    tier_index_after: int
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
    family_feature_scaling_caps: dict[str, float]
    family_feature_scaling_history_sizes: dict[str, int]
    family_feature_scaling_fallback_to_v1: bool
    family_scores: dict[str, float]
    family_feature_values: dict[str, float]
    reward_scope: str
    reward_success_hint_scale: float
    reward_progress_hint_scale: float
    reward_stagnation_penalty_scale: float
    reward: float
    reward_success_hint: float
    reward_progress_hint: float
    reward_progress_score: float
    reward_token_penalty: float
    reward_latency_penalty: float
    reward_stagnation_penalty: float
    reward_budget_exceeded_penalty: float
    request_budget_exceeded: bool
    success: bool
    local_llm_call_count: int
    local_tokens_prompt: int
    local_tokens_completion: int
    local_tokens_total: int
    local_wall_sec: float
    override_reason: str
    gate_model_updated: bool
    family_model_updated: bool
    model_updated: bool
    update_reason: str
    nan_replacement_counts: dict[str, int]
    created_at_utc: str
    bwk_run_token_budget: int = 0
    bwk_remaining_token_budget_before: int = 0
    bwk_remaining_token_budget_after: int = 0
    bwk_tokens_charged: int = 0
    bwk_cumulative_tokens_charged: int = 0
    bwk_dual_variable_before: float = 0.0
    bwk_dual_variable_after: float = 0.0
    bwk_budget_exhausted: bool = False
    effective_evidence: dict[str, dict[str, float]] = field(default_factory=dict)

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
            "architecture": self.architecture,
            "decision_index": self.decision_index,
            "selected_action": self.selected_action,
            "applied_action": self.applied_action,
            "behavior_action": self.behavior_action,
            "default_action": self.default_action,
            "selected_family": self.selected_family,
            "applied_family": self.applied_family,
            "behavior_family": self.behavior_family,
            "available_families": list(self.available_families),
            "unavailable_family_reasons": dict(self.unavailable_family_reasons),
            "family_propensity": self.family_propensity,
            "family_decision_strategy": self.family_decision_strategy,
            "selected_multiplier": self.selected_multiplier,
            "applied_multiplier": self.applied_multiplier,
            "default_multiplier": self.default_multiplier,
            "default_controls": dict(self.default_controls),
            "selected_controls": dict(self.selected_controls),
            "applied_controls": dict(self.applied_controls),
            "selected_controls_changed": {
                key: bool(value) for key, value in self.selected_controls_changed.items()
            },
            "applied_controls_changed": {
                key: bool(value) for key, value in self.applied_controls_changed.items()
            },
            "partial_override": bool(self.partial_override),
            "partial_override_errors": dict(self.partial_override_errors),
            "tier_index_before": self.tier_index_before,
            "tier_index_after": self.tier_index_after,
            "propensity": self.propensity,
            "behavior_propensity": self.behavior_propensity,
            "decision_strategy": self.decision_strategy,
            "feature_scaling_profile": self.feature_scaling_profile,
            "feature_scaling_caps": {k: float(v) for k, v in self.feature_scaling_caps.items()},
            "feature_scaling_history_sizes": {
                key: int(value) for key, value in self.feature_scaling_history_sizes.items()
            },
            "feature_scaling_fallback_to_v1": bool(self.feature_scaling_fallback_to_v1),
            "configured_alpha": self.configured_alpha,
            "configured_epsilon": self.configured_epsilon,
            "configured_decay": self.configured_decay,
            "effective_epsilon": self.effective_epsilon,
            "warmup_active": bool(self.warmup_active),
            "scores": dict(self.scores),
            "feature_values": dict(self.feature_values),
            "family_feature_scaling_caps": {
                k: float(v) for k, v in self.family_feature_scaling_caps.items()
            },
            "family_feature_scaling_history_sizes": {
                key: int(value) for key, value in self.family_feature_scaling_history_sizes.items()
            },
            "family_feature_scaling_fallback_to_v1": bool(
                self.family_feature_scaling_fallback_to_v1
            ),
            "family_scores": dict(self.family_scores),
            "family_feature_values": dict(self.family_feature_values),
            "reward_scope": self.reward_scope,
            "reward_success_hint_scale": self.reward_success_hint_scale,
            "reward_progress_hint_scale": self.reward_progress_hint_scale,
            "reward_stagnation_penalty_scale": self.reward_stagnation_penalty_scale,
            "reward": self.reward,
            "reward_success_hint": self.reward_success_hint,
            "reward_progress_hint": self.reward_progress_hint,
            "reward_progress_score": self.reward_progress_score,
            "reward_token_penalty": self.reward_token_penalty,
            "reward_latency_penalty": self.reward_latency_penalty,
            "reward_stagnation_penalty": self.reward_stagnation_penalty,
            "reward_budget_exceeded_penalty": self.reward_budget_exceeded_penalty,
            "request_budget_exceeded": self.request_budget_exceeded,
            "success": self.success,
            "local_llm_call_count": self.local_llm_call_count,
            "local_tokens_prompt": self.local_tokens_prompt,
            "local_tokens_completion": self.local_tokens_completion,
            "local_tokens_total": self.local_tokens_total,
            "local_wall_sec": self.local_wall_sec,
            "override_reason": self.override_reason,
            "gate_model_updated": bool(self.gate_model_updated),
            "family_model_updated": bool(self.family_model_updated),
            "model_updated": self.model_updated,
            "update_reason": self.update_reason,
            "nan_replacement_counts": dict(self.nan_replacement_counts),
            "created_at_utc": self.created_at_utc,
            "bwk_run_token_budget": self.bwk_run_token_budget,
            "bwk_remaining_token_budget_before": self.bwk_remaining_token_budget_before,
            "bwk_remaining_token_budget_after": self.bwk_remaining_token_budget_after,
            "bwk_tokens_charged": self.bwk_tokens_charged,
            "bwk_cumulative_tokens_charged": self.bwk_cumulative_tokens_charged,
            "bwk_dual_variable_before": self.bwk_dual_variable_before,
            "bwk_dual_variable_after": self.bwk_dual_variable_after,
            "bwk_budget_exhausted": bool(self.bwk_budget_exhausted),
            "effective_evidence": {
                model: {action: float(value) for action, value in evidence.items()}
                for model, evidence in self.effective_evidence.items()
            },
        }


@dataclass
class LinUCBState(core.LinUCBState):
    """The shared LinUCB core state, defaulting to the budget actions."""

    action_names: tuple[str, ...] = BUDGET_POLICY_ACTIONS


class AdaptiveBudgetPolicyController:
    """Thread-safe LinUCB contextual-bandit controller for budget escalation.

    Manages exploration/exploitation via epsilon-greedy with UCB scores,
    persists model state to disk, and logs decision-outcome events as JSONL.
    """

    def __init__(self, config: BudgetPolicyConfig) -> None:
        self.config = config
        self._lock = threading.RLock()
        self._rng = random.Random(config.random_seed)
        self._state = LinUCBState(action_names=BUDGET_POLICY_ACTIONS)
        self._family_state = LinUCBState(action_names=BUDGET_POLICY_FAMILIES)
        self._save_counter = 0
        self._nan_counts: dict[str, int] = {}
        self._bwk_run_token_budget = max(0, int(config.bwk_run_token_budget or 0))
        self._bwk_remaining_token_budget = int(self._bwk_run_token_budget)
        self._bwk_cumulative_tokens_charged = 0
        self._bwk_dual_variable = 0.0
        self.load_error: str | None = None
        self._load_state_if_present()

    @property
    def mode(self) -> str:
        """Current policy mode (``off``, ``shadow``, or ``on``)."""
        return self.config.mode

    @property
    def algorithm(self) -> str:
        """Configured algorithm identifier."""
        return self.config.algorithm

    @property
    def architecture(self) -> str:
        """Configured budget-policy architecture identifier."""
        return self.config.architecture

    @property
    def state(self) -> LinUCBState:
        """Return a deep copy of the gate-controller LinUCB state."""
        with self._lock:
            return self._copy_state(self._state)

    @property
    def family_state(self) -> LinUCBState:
        """Return a deep copy of the family-controller LinUCB state."""
        with self._lock:
            return self._copy_state(self._family_state)

    @property
    def bwk_status(self) -> dict[str, Any]:
        """Return current BwK token-budget telemetry."""
        with self._lock:
            return {
                "run_token_budget": int(self._bwk_run_token_budget),
                "remaining_token_budget": int(self._bwk_remaining_token_budget),
                "cumulative_tokens_charged": int(self._bwk_cumulative_tokens_charged),
                "dual_variable": float(self._bwk_dual_variable),
                "budget_exhausted": bool(self._bwk_remaining_token_budget <= 0),
            }

    def _copy_state(self, state: LinUCBState) -> LinUCBState:
        return LinUCBState(
            action_names=tuple(state.action_names),
            feature_names=list(state.feature_names),
            decisions=int(state.decisions),
            updates=int(state.updates),
            action_counts={k: int(v) for k, v in state.action_counts.items()},
            A={k: v.copy() for k, v in state.A.items()},
            b={k: v.copy() for k, v in state.b.items()},
            feature_scaling_history={
                k: [float(item) for item in values]
                for k, values in state.feature_scaling_history.items()
            },
            feature_scaling_caps={k: float(v) for k, v in state.feature_scaling_caps.items()},
        )

    def get_nan_stats(self) -> dict[str, int]:
        """Return a snapshot of per-feature NaN replacement counts."""
        with self._lock:
            return dict(self._nan_counts)

    def _tracked_safe_float(self, value: Any, *, feature_name: str, default: float = 0.0) -> float:
        """Convert *value* via :func:`safe_float`, tracking NaN replacements.

        The counter is updated under the (re-entrant) lock: ``decide()`` cleans features
        before taking it, and concurrent increments were lost (POL-16).
        """
        result, replaced = _safe_float_tracked(value, default=default)
        if replaced:
            with self._lock:
                self._nan_counts[feature_name] = self._nan_counts.get(feature_name, 0) + 1
        return result

    def _record_feature_scaling_history(
        self,
        *,
        state: LinUCBState,
        cleaned_features: Mapping[str, float],
    ) -> None:
        """Append raw feature observations used by percentile scaling."""
        if not is_percentile_feature_scaling_profile(self.config.feature_scaling_profile):
            return
        for key, value in cleaned_features.items():
            history = list(state.feature_scaling_history.get(str(key), []))
            history.append(float(value))
            if len(history) > DEFAULT_POLICY_FEATURE_SCALING_HISTORY_LIMIT:
                history = history[-DEFAULT_POLICY_FEATURE_SCALING_HISTORY_LIMIT:]
            state.feature_scaling_history[str(key)] = history

    def _scale_features(
        self,
        *,
        state: LinUCBState,
        cleaned_features: Mapping[str, float],
    ) -> tuple[dict[str, float], dict[str, float], dict[str, int], bool]:
        """Scale features and expose percentile-cap telemetry when enabled."""
        if not is_percentile_feature_scaling_profile(self.config.feature_scaling_profile):
            return (
                scale_feature_mapping(
                    cleaned_features,
                    profile=self.config.feature_scaling_profile,
                ),
                {},
                {},
                False,
            )
        history = {
            str(name): [float(item) for item in values]
            for name, values in state.feature_scaling_history.items()
        }
        scaler = PercentileScaler(min_samples=DEFAULT_POLICY_FEATURE_SCALING_MIN_SAMPLES)
        scaler.fit(history)
        caps = scaler.caps
        history_sizes = {str(name): len(values) for name, values in history.items()}
        state.feature_scaling_caps = {str(name): float(value) for name, value in caps.items()}
        fallback_to_v1 = any(
            history_sizes.get(str(name), 0) < DEFAULT_POLICY_FEATURE_SCALING_MIN_SAMPLES
            for name in cleaned_features
        )
        return (
            scaler.scale(cleaned_features, profile="v1"),
            {str(name): float(value) for name, value in caps.items()},
            history_sizes,
            bool(fallback_to_v1),
        )

    def _decide_with_state(
        self,
        *,
        state: LinUCBState,
        cleaned_features: Mapping[str, float],
        default_action: str,
        available_actions: tuple[str, ...],
    ) -> tuple[
        str,
        float,
        str,
        dict[str, float],
        dict[str, float],
        dict[str, float],
        dict[str, int],
        bool,
        int,
    ]:
        (
            scaled_features,
            feature_scaling_caps,
            feature_scaling_history_sizes,
            feature_scaling_fallback_to_v1,
        ) = self._scale_features(state=state, cleaned_features=cleaned_features)
        self._record_feature_scaling_history(state=state, cleaned_features=cleaned_features)
        self._ensure_feature_state(state=state, cleaned_features=scaled_features)
        decision_index = state.decisions + 1
        default_norm = (
            default_action if default_action in state.action_names else state.action_names[0]
        )
        available = tuple(action for action in available_actions if action in state.action_names)
        if not available:
            available = (default_norm,)
        if self.config.mode == "off":
            selected = default_norm
            propensity = 1.0
            strategy = "policy_off"
            scores = dict.fromkeys(state.action_names, 0.0)
        else:
            x = self._feature_vector(state=state, cleaned_features=scaled_features)
            if self.config.algorithm == "thompson":
                scores = self._thompson_scores(state=state, x=x)
            else:
                scores = self._linucb_scores(state=state, x=x)
            greedy = max(
                available,
                key=lambda action: (scores.get(action, 0.0), action == default_norm),
            )
            if decision_index <= self.config.warmup_decisions:
                selected = self._rng.choice(list(available))
                strategy = "warmup_uniform"
            else:
                if self._rng.random() < self.config.epsilon:
                    selected = self._rng.choice(list(available))
                    strategy = "epsilon_uniform"
                else:
                    selected = greedy
                    strategy = "linucb_greedy"
            k_actions = float(len(available))
            base_prob = self.config.epsilon / k_actions
            if decision_index <= self.config.warmup_decisions:
                propensity = 1.0 / k_actions
            else:
                propensity = base_prob + (
                    (1.0 - self.config.epsilon) if selected == greedy else 0.0
                )
            propensity = min(1.0, max(1e-9, propensity))
        state.decisions = decision_index
        state.action_counts[selected] = int(state.action_counts.get(selected, 0)) + 1
        return (
            selected,
            float(propensity),
            strategy,
            {key: float(value) for key, value in scores.items()},
            dict(scaled_features),
            {str(key): float(value) for key, value in feature_scaling_caps.items()},
            {str(key): int(value) for key, value in feature_scaling_history_sizes.items()},
            bool(feature_scaling_fallback_to_v1),
            decision_index,
        )

    @contextmanager
    def _decision_rng(self, rng_key: str | None) -> Iterator[None]:
        """Use an RNG keyed by *rng_key* for one decision (POL-05).

        Keyed draws do not depend on how concurrent units interleave; without a key or a
        seed the controller's shared stream is used.
        """
        if rng_key is None or self.config.random_seed is None:
            yield
            return
        shared = self._rng
        self._rng = random.Random(derive_seed(self.config.random_seed, rng_key))
        try:
            yield
        finally:
            self._rng = shared

    def decide(
        self,
        *,
        feature_values: Mapping[str, float],
        default_action: str = ACTION_KEEP_BUDGET,
        family_feature_values: Mapping[str, float] | None = None,
        available_families: tuple[str, ...] | None = None,
        unavailable_family_reasons: Mapping[str, str] | None = None,
        rng_key: str | None = None,
    ) -> BudgetPolicyDecision:
        """Select a budget action given the current context features."""
        cleaned_features = {
            str(k): self._tracked_safe_float(v, feature_name=str(k), default=0.0)
            for k, v in dict(feature_values).items()
        }
        family_source = feature_values if family_feature_values is None else family_feature_values
        cleaned_family_features = {
            str(k): self._tracked_safe_float(v, feature_name=f"family:{k}", default=0.0)
            for k, v in dict(family_source).items()
        }
        available_family_actions = tuple(
            action for action in (available_families or ()) if action in BUDGET_POLICY_FAMILIES
        )
        unavailable_family_reasons_norm = {
            str(key): str(value)
            for key, value in dict(unavailable_family_reasons or {}).items()
            if str(key) in BUDGET_POLICY_FAMILIES and str(value)
        }
        with self._lock, self._decision_rng(rng_key):
            default_norm = (
                default_action if default_action in BUDGET_POLICY_ACTIONS else ACTION_KEEP_BUDGET
            )
            gate_decision_index = self._state.decisions + 1
            effective_epsilon, warmup_active = _policy_effective_epsilon(
                mode=self.config.mode,
                decision_index=gate_decision_index,
                warmup_decisions=self.config.warmup_decisions,
                epsilon=self.config.epsilon,
            )
            bwk_remaining_token_budget = int(self._bwk_remaining_token_budget)
            bwk_budget_exhausted = bool(
                self.config.architecture == BUDGET_POLICY_ARCH_BWK_V1
                and bwk_remaining_token_budget <= 0
            )
            gate_available_actions: tuple[str, ...] = BUDGET_POLICY_ACTIONS
            if (
                self.config.architecture == BUDGET_POLICY_ARCH_HIERARCHICAL_V1
                and not available_family_actions
            ) or bwk_budget_exhausted:
                gate_available_actions = (ACTION_KEEP_BUDGET,)
            (
                selected,
                propensity,
                strategy,
                scores,
                scaled_features,
                feature_scaling_caps,
                feature_scaling_history_sizes,
                feature_scaling_fallback_to_v1,
                decision_index,
            ) = self._decide_with_state(
                state=self._state,
                cleaned_features=cleaned_features,
                default_action=default_norm,
                available_actions=gate_available_actions,
            )
            mode = self.config.mode
            applied = selected if mode == "on" else default_norm
            selected_family = ""
            applied_family = ""
            family_propensity = 1.0
            family_strategy = "not_applicable"
            family_scores = dict.fromkeys(BUDGET_POLICY_FAMILIES, 0.0)
            family_scaled_features: dict[str, float] = {}
            family_feature_scaling_caps: dict[str, float] = {}
            family_feature_scaling_history_sizes: dict[str, int] = {}
            family_feature_scaling_fallback_to_v1 = False
            if (
                self.config.architecture == BUDGET_POLICY_ARCH_HIERARCHICAL_V1
                and mode != "off"
                and selected == ACTION_INCREASE_BUDGET
                and available_family_actions
            ):
                (
                    selected_family,
                    family_propensity,
                    family_strategy,
                    family_scores,
                    family_scaled_features,
                    family_feature_scaling_caps,
                    family_feature_scaling_history_sizes,
                    family_feature_scaling_fallback_to_v1,
                    _,
                ) = self._decide_with_state(
                    state=self._family_state,
                    cleaned_features=cleaned_family_features,
                    default_action=available_family_actions[0],
                    available_actions=available_family_actions,
                )
                if mode == "on" and applied == ACTION_INCREASE_BUDGET:
                    applied_family = selected_family
            return BudgetPolicyDecision(
                decision_id=uuid.uuid4().hex,
                mode=mode,
                algorithm=self.config.algorithm,
                architecture=self.config.architecture,
                decision_index=decision_index,
                created_at_utc=_utc_now(),
                selected_action=selected,
                applied_action=applied,
                default_action=default_norm,
                propensity=float(propensity),
                decision_strategy=strategy,
                scores=scores,
                feature_values=scaled_features,
                feature_scaling_caps=feature_scaling_caps,
                feature_scaling_history_sizes=feature_scaling_history_sizes,
                feature_scaling_fallback_to_v1=feature_scaling_fallback_to_v1,
                selected_family=str(selected_family),
                applied_family=str(applied_family),
                available_families=available_family_actions,
                unavailable_family_reasons=unavailable_family_reasons_norm,
                family_propensity=float(family_propensity),
                family_decision_strategy=str(family_strategy),
                family_scores={key: float(value) for key, value in family_scores.items()},
                family_feature_values={
                    key: float(value) for key, value in family_scaled_features.items()
                },
                family_feature_scaling_caps=family_feature_scaling_caps,
                family_feature_scaling_history_sizes=family_feature_scaling_history_sizes,
                family_feature_scaling_fallback_to_v1=bool(family_feature_scaling_fallback_to_v1),
                configured_alpha=float(self.config.alpha),
                configured_epsilon=float(self.config.epsilon),
                configured_decay=float(self.config.decay),
                effective_epsilon=float(effective_epsilon),
                warmup_active=bool(warmup_active),
                bwk_run_token_budget=int(self._bwk_run_token_budget),
                bwk_remaining_token_budget=bwk_remaining_token_budget,
                bwk_dual_variable=float(self._bwk_dual_variable),
                bwk_budget_exhausted=bwk_budget_exhausted,
            )

    def record_outcome(
        self,
        *,
        decision: BudgetPolicyDecision,
        run_id: str,
        case_id: str,
        condition: str,
        attempt: int,
        note: str,
        reward_scope: str,
        reward: float,
        reward_success_hint: float,
        reward_progress_hint: float,
        reward_progress_score: float,
        reward_token_penalty: float,
        reward_latency_penalty: float,
        reward_stagnation_penalty: float,
        reward_budget_exceeded_penalty: float,
        request_budget_exceeded: bool,
        success: bool,
        local_llm_call_count: int,
        local_tokens_prompt: int,
        local_tokens_completion: int,
        local_tokens_total: int,
        local_wall_sec: float,
        selected_multiplier: float,
        applied_multiplier: float,
        default_multiplier: float,
        default_controls: Mapping[str, Any],
        selected_controls: Mapping[str, Any],
        applied_controls: Mapping[str, Any],
        selected_controls_changed: Mapping[str, bool],
        applied_controls_changed: Mapping[str, bool],
        partial_override: bool,
        partial_override_errors: Mapping[str, str],
        tier_index_before: int,
        tier_index_after: int,
        override_reason: str,
        reward_success_hint_scale: float = 1.0,
        reward_progress_hint_scale: float = 1.0,
        reward_stagnation_penalty_scale: float = 1.0,
    ) -> BudgetPolicyEvent:
        """Record an observed reward for a prior budget decision."""
        with self._lock:
            bwk_run_token_budget = int(self._bwk_run_token_budget)
            bwk_remaining_token_budget_before = int(self._bwk_remaining_token_budget)
            bwk_dual_variable_before = float(self._bwk_dual_variable)
            bwk_tokens_charged = 0
            bwk_remaining_token_budget_after = bwk_remaining_token_budget_before
            bwk_dual_variable_after = bwk_dual_variable_before
            bwk_cumulative_tokens_charged = int(self._bwk_cumulative_tokens_charged)
            if self.config.architecture == BUDGET_POLICY_ARCH_BWK_V1:
                bwk_tokens_charged = max(0, int(local_tokens_total))
                bwk_cumulative_tokens_charged += bwk_tokens_charged
                self._bwk_cumulative_tokens_charged = bwk_cumulative_tokens_charged
                bwk_remaining_token_budget_after = max(
                    0,
                    bwk_remaining_token_budget_before - bwk_tokens_charged,
                )
                self._bwk_remaining_token_budget = bwk_remaining_token_budget_after
                budget_denom = max(1, bwk_run_token_budget or bwk_tokens_charged)
                overspend_tokens = max(0, bwk_tokens_charged - bwk_remaining_token_budget_before)
                bwk_dual_variable_after = max(
                    0.0,
                    float(self._bwk_dual_variable) * float(self.config.decay)
                    + (float(overspend_tokens) / float(budget_denom)),
                )
                self._bwk_dual_variable = bwk_dual_variable_after

            gate_model_updated = False
            family_model_updated = False
            update_reason = "policy_not_enabled"
            if self.config.mode == "on" and decision.applied_action in BUDGET_POLICY_ACTIONS:
                gate_model_updated = self._update_model(
                    state=self._state,
                    action=decision.applied_action,
                    reward=reward,
                    feature_values=decision.feature_values,
                )
                if (
                    self.config.architecture == BUDGET_POLICY_ARCH_HIERARCHICAL_V1
                    and decision.applied_action == ACTION_INCREASE_BUDGET
                    and decision.applied_family in BUDGET_POLICY_FAMILIES
                ):
                    family_features = decision.family_feature_values or decision.feature_values
                    family_model_updated = self._update_model(
                        state=self._family_state,
                        action=decision.applied_family,
                        reward=reward,
                        feature_values=family_features,
                    )
                if gate_model_updated:
                    if (
                        self.config.architecture == BUDGET_POLICY_ARCH_HIERARCHICAL_V1
                        and decision.applied_action == ACTION_INCREASE_BUDGET
                        and decision.applied_family in BUDGET_POLICY_FAMILIES
                        and not family_model_updated
                    ):
                        update_reason = "gate_updated_family_skipped"
                    else:
                        update_reason = "on_policy_update"
                else:
                    update_reason = "update_skipped_incomplete_state"
            elif self.config.mode == "shadow":
                update_reason = "shadow_no_update"
            elif self.config.mode == "off":
                update_reason = "off_no_update"

            selected_action = str(decision.selected_action)
            applied_action = str(decision.applied_action)
            behavior_action = applied_action
            behavior_family = str(decision.applied_family)
            behavior_propensity = 1.0
            if selected_action == applied_action:
                behavior_propensity = min(1.0, max(1e-9, float(decision.propensity)))
                if decision.selected_family and decision.selected_family == behavior_family:
                    behavior_propensity = min(
                        1.0,
                        max(1e-9, behavior_propensity * float(decision.family_propensity)),
                    )
            model_updated = bool(gate_model_updated or family_model_updated)
            event = BudgetPolicyEvent(
                run_id=str(run_id),
                case_id=str(case_id),
                condition=str(condition),
                attempt=max(0, int(attempt)),
                note=str(note),
                decision_id=str(decision.decision_id),
                mode=str(decision.mode),
                algorithm=str(decision.algorithm),
                architecture=str(decision.architecture),
                decision_index=max(0, int(decision.decision_index)),
                selected_action=selected_action,
                applied_action=applied_action,
                behavior_action=behavior_action,
                default_action=str(decision.default_action),
                selected_family=str(decision.selected_family),
                applied_family=str(decision.applied_family),
                behavior_family=behavior_family,
                available_families=tuple(str(item) for item in decision.available_families),
                unavailable_family_reasons={
                    str(k): str(v) for k, v in decision.unavailable_family_reasons.items()
                },
                family_propensity=min(1.0, max(1e-9, float(decision.family_propensity))),
                family_decision_strategy=str(decision.family_decision_strategy),
                selected_multiplier=float(selected_multiplier),
                applied_multiplier=float(applied_multiplier),
                default_multiplier=float(default_multiplier),
                default_controls={str(k): v for k, v in default_controls.items()},
                selected_controls={str(k): v for k, v in selected_controls.items()},
                applied_controls={str(k): v for k, v in applied_controls.items()},
                selected_controls_changed={
                    str(k): bool(v) for k, v in selected_controls_changed.items()
                },
                applied_controls_changed={
                    str(k): bool(v) for k, v in applied_controls_changed.items()
                },
                partial_override=bool(partial_override),
                partial_override_errors={
                    str(k): str(v) for k, v in partial_override_errors.items()
                },
                tier_index_before=max(0, int(tier_index_before)),
                tier_index_after=max(0, int(tier_index_after)),
                propensity=min(1.0, max(1e-9, float(decision.propensity))),
                behavior_propensity=behavior_propensity,
                decision_strategy=str(decision.decision_strategy),
                feature_scaling_profile=str(self.config.feature_scaling_profile),
                feature_scaling_caps={
                    k: float(v) for k, v in decision.feature_scaling_caps.items()
                },
                feature_scaling_history_sizes={
                    k: int(v) for k, v in decision.feature_scaling_history_sizes.items()
                },
                feature_scaling_fallback_to_v1=bool(decision.feature_scaling_fallback_to_v1),
                configured_alpha=float(decision.configured_alpha),
                configured_epsilon=float(decision.configured_epsilon),
                configured_decay=float(decision.configured_decay),
                effective_epsilon=float(decision.effective_epsilon),
                warmup_active=bool(decision.warmup_active),
                scores={k: float(v) for k, v in decision.scores.items()},
                feature_values={k: float(v) for k, v in decision.feature_values.items()},
                family_feature_scaling_caps={
                    k: float(v) for k, v in decision.family_feature_scaling_caps.items()
                },
                family_feature_scaling_history_sizes={
                    k: int(v) for k, v in decision.family_feature_scaling_history_sizes.items()
                },
                family_feature_scaling_fallback_to_v1=bool(
                    decision.family_feature_scaling_fallback_to_v1
                ),
                family_scores={k: float(v) for k, v in decision.family_scores.items()},
                family_feature_values={
                    k: float(v) for k, v in decision.family_feature_values.items()
                },
                reward_scope=str(reward_scope),
                reward_success_hint_scale=float(reward_success_hint_scale),
                reward_progress_hint_scale=float(reward_progress_hint_scale),
                reward_stagnation_penalty_scale=float(reward_stagnation_penalty_scale),
                reward=float(reward),
                reward_success_hint=float(reward_success_hint),
                reward_progress_hint=float(reward_progress_hint),
                reward_progress_score=float(reward_progress_score),
                reward_token_penalty=float(reward_token_penalty),
                reward_latency_penalty=float(reward_latency_penalty),
                reward_stagnation_penalty=float(reward_stagnation_penalty),
                reward_budget_exceeded_penalty=float(reward_budget_exceeded_penalty),
                request_budget_exceeded=bool(request_budget_exceeded),
                success=bool(success),
                local_llm_call_count=max(0, int(local_llm_call_count)),
                local_tokens_prompt=max(0, int(local_tokens_prompt)),
                local_tokens_completion=max(0, int(local_tokens_completion)),
                local_tokens_total=max(0, int(local_tokens_total)),
                local_wall_sec=max(0.0, float(local_wall_sec)),
                override_reason=str(override_reason),
                gate_model_updated=bool(gate_model_updated),
                family_model_updated=bool(family_model_updated),
                model_updated=model_updated,
                update_reason=update_reason,
                nan_replacement_counts=dict(self._nan_counts),
                created_at_utc=_utc_now(),
                bwk_run_token_budget=bwk_run_token_budget,
                bwk_remaining_token_budget_before=bwk_remaining_token_budget_before,
                bwk_remaining_token_budget_after=bwk_remaining_token_budget_after,
                bwk_tokens_charged=bwk_tokens_charged,
                bwk_cumulative_tokens_charged=bwk_cumulative_tokens_charged,
                bwk_dual_variable_before=bwk_dual_variable_before,
                bwk_dual_variable_after=bwk_dual_variable_after,
                bwk_budget_exhausted=bool(bwk_remaining_token_budget_after <= 0),
                effective_evidence={
                    "gate": core.effective_evidence(self._state, l2=float(self.config.l2)),
                    "family": core.effective_evidence(self._family_state, l2=float(self.config.l2)),
                },
            )
            self._append_event(event=event)
            if model_updated or self.config.architecture == BUDGET_POLICY_ARCH_BWK_V1:
                self._save_counter += 1
                if self._save_counter >= max(1, int(self.config.save_every)):
                    self._save_counter = 0
                    self._save_state()
            return event

    def close(self) -> None:
        """Flush the current model state to disk."""
        with self._lock:
            self._save_state()

    def _ensure_feature_state(
        self,
        *,
        state: LinUCBState,
        cleaned_features: Mapping[str, float],
    ) -> None:
        """Initialise or expand A/b matrices when new features appear."""
        if self.config.algorithm not in BUDGET_POLICY_ALGOS:
            return
        core.ensure_feature_state(state, cleaned_features, l2=float(self.config.l2))

    def _feature_vector(
        self,
        *,
        state: LinUCBState,
        cleaned_features: Mapping[str, float],
    ) -> np.ndarray:
        """Build a numpy feature vector aligned with ``feature_names``."""
        return core.feature_vector(
            state,
            cleaned_features,
            to_float=lambda value, name: self._tracked_safe_float(
                value, feature_name=name, default=0.0
            ),
        )

    def _linucb_scores(self, *, state: LinUCBState, x: np.ndarray) -> dict[str, float]:
        """Compute UCB scores (exploit + alpha * explore) for every action."""
        if self.config.algorithm != "linucb":
            return dict.fromkeys(state.action_names, 0.0)
        return core.linucb_scores(
            state, x, l2=float(self.config.l2), alpha=float(self.config.alpha)
        )

    def _thompson_scores(self, *, state: LinUCBState, x: np.ndarray) -> dict[str, float]:
        """Sample θ ~ N(A⁻¹b, A⁻¹) and return θ·x for each action (Thompson Sampling)."""
        if self.config.algorithm != "thompson":
            return dict.fromkeys(state.action_names, 0.0)
        return core.thompson_scores(state, x, seed=self.config.random_seed)

    def _update_model(
        self,
        *,
        state: LinUCBState,
        action: str,
        reward: float,
        feature_values: Mapping[str, float],
    ) -> bool:
        """Apply a rank-1 LinUCB update: A += x x^T, b += reward * x."""
        if self.config.algorithm not in BUDGET_POLICY_ALGOS:
            return False
        if action not in state.action_names or not state.feature_names:
            return False
        x = self._feature_vector(state=state, cleaned_features=feature_values)
        self._apply_recency_decay(state=state, action=action)
        return core.rank_one_update(state, action=action, reward=reward, x=x)

    def _apply_recency_decay(self, *, state: LinUCBState, action: str) -> None:
        """Discount the updated arm's prior evidence when decay < 1 is configured."""
        core.apply_recency_decay(
            state, action=action, decay=float(self.config.decay), l2=float(self.config.l2)
        )

    def _serialize_state(self, state: LinUCBState) -> dict[str, Any]:
        return {
            "action_names": list(state.action_names),
            "feature_names": list(state.feature_names),
            "decisions": int(state.decisions),
            "updates": int(state.updates),
            "action_counts": {
                action: int(state.action_counts.get(action, 0)) for action in state.action_names
            },
            "A": {
                action: state.A.get(action, np.empty((0, 0), dtype=float)).tolist()
                for action in state.action_names
            },
            "b": {
                action: state.b.get(action, np.empty((0,), dtype=float)).tolist()
                for action in state.action_names
            },
            "feature_scaling_history": {
                name: [float(item) for item in values]
                for name, values in state.feature_scaling_history.items()
            },
            "feature_scaling_caps": {
                name: float(value) for name, value in state.feature_scaling_caps.items()
            },
        }

    def _deserialize_state(
        self,
        *,
        raw_state: Mapping[str, Any],
        action_names: tuple[str, ...],
    ) -> LinUCBState:
        feature_names_raw = raw_state.get("feature_names")
        if not isinstance(feature_names_raw, list) or not all(
            isinstance(item, str) for item in feature_names_raw
        ):
            raise ValueError("missing or invalid feature_names")
        feature_names = [str(item) for item in feature_names_raw]
        dim = len(feature_names)
        decisions = _coerce_nonnegative_int(raw_state.get("decisions"), default=0)
        updates = _coerce_nonnegative_int(raw_state.get("updates"), default=0)
        feature_scaling_history_raw = raw_state.get("feature_scaling_history")
        feature_scaling_history: dict[str, list[float]] = {}
        if isinstance(feature_scaling_history_raw, dict):
            for key, values in feature_scaling_history_raw.items():
                if not isinstance(values, list):
                    continue
                feature_scaling_history[str(key)] = [
                    float(item)
                    for item in values
                    if isinstance(item, (int, float)) and np.isfinite(float(item))
                ][-DEFAULT_POLICY_FEATURE_SCALING_HISTORY_LIMIT:]
        feature_scaling_caps_raw = raw_state.get("feature_scaling_caps")
        feature_scaling_caps: dict[str, float] = {}
        if isinstance(feature_scaling_caps_raw, dict):
            for key, value in feature_scaling_caps_raw.items():
                if isinstance(value, (int, float)) and np.isfinite(float(value)):
                    feature_scaling_caps[str(key)] = float(value)
        action_counts_raw = raw_state.get("action_counts")
        action_counts = dict.fromkeys(action_names, 0)
        if isinstance(action_counts_raw, dict):
            for action in action_names:
                action_counts[action] = _coerce_nonnegative_int(
                    action_counts_raw.get(action), default=0
                )
        if dim <= 0:
            return LinUCBState(
                action_names=action_names,
                feature_names=[],
                decisions=decisions,
                updates=updates,
                action_counts=action_counts,
                A={},
                b={},
                feature_scaling_history=feature_scaling_history,
                feature_scaling_caps=feature_scaling_caps,
            )
        A_raw = raw_state.get("A")
        b_raw = raw_state.get("b")
        if not isinstance(A_raw, dict) or not isinstance(b_raw, dict):
            raise ValueError("missing action matrices")
        A_state: dict[str, np.ndarray] = {}
        b_state: dict[str, np.ndarray] = {}
        for action in action_names:
            A_matrix = np.array(A_raw.get(action), dtype=float)
            b_vector = np.array(b_raw.get(action), dtype=float)
            if A_matrix.shape != (dim, dim):
                raise ValueError(f"invalid A shape for action {action}")
            if b_vector.shape != (dim,):
                raise ValueError(f"invalid b shape for action {action}")
            A_state[action] = A_matrix
            b_state[action] = b_vector
        return LinUCBState(
            action_names=action_names,
            feature_names=feature_names,
            decisions=decisions,
            updates=updates,
            action_counts=action_counts,
            A=A_state,
            b=b_state,
            feature_scaling_history=feature_scaling_history,
            feature_scaling_caps=feature_scaling_caps,
        )

    def _state_payload(self) -> dict[str, Any]:
        """Build a JSON-serialisable snapshot of the full model state."""
        return {
            "schema_version": STATE_SCHEMA_VERSION,
            "mode": self.config.mode,
            "algorithm": self.config.algorithm,
            "architecture": self.config.architecture,
            "alpha": self.config.alpha,
            "epsilon": self.config.epsilon,
            "l2": self.config.l2,
            "warmup_decisions": self.config.warmup_decisions,
            "save_every": self.config.save_every,
            "decay": self.config.decay,
            "feature_scaling_profile": self.config.feature_scaling_profile,
            "gate_state": self._serialize_state(self._state),
            "family_state": self._serialize_state(self._family_state),
            "bwk_run_token_budget": int(self._bwk_run_token_budget),
            "bwk_remaining_token_budget": int(self._bwk_remaining_token_budget),
            "bwk_cumulative_tokens_charged": int(self._bwk_cumulative_tokens_charged),
            "bwk_dual_variable": float(self._bwk_dual_variable),
            "saved_at_utc": _utc_now(),
        }

    def _save_state(self) -> None:
        """Atomically write model state to disk via a temp-file rename."""
        path = self.config.state_path
        if path is None:
            return
        payload = self._state_payload()
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        tmp_path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        tmp_path.replace(path)

    def _load_state_if_present(self) -> None:
        """Load persisted LinUCB state from disk, if it exists."""
        path = self.config.init_state_path or self.config.state_path
        loading_from_init_state = (
            self.config.init_state_path is not None and path == self.config.init_state_path
        )
        if path is None or not path.exists():
            return
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("state payload is not an object")
            schema_version = int(raw.get("schema_version", 0))
            algorithm = str(raw.get("algorithm") or "")
            if algorithm and algorithm != self.config.algorithm:
                raise ValueError("algorithm mismatch")
            state_scaling_profile = normalize_feature_scaling_profile(
                str(raw.get("feature_scaling_profile") or DEFAULT_POLICY_FEATURE_SCALING_PROFILE)
            )
            config_scaling_profile = normalize_feature_scaling_profile(
                self.config.feature_scaling_profile
            )
            if state_scaling_profile != config_scaling_profile:
                raise ValueError("feature_scaling_profile mismatch")
            if schema_version == 2:
                if self.config.architecture != BUDGET_POLICY_ARCH_TIER_BINARY:
                    raise ValueError("legacy tier_binary state incompatible with hierarchical_v1")
                self._state = self._deserialize_state(
                    raw_state=raw,
                    action_names=BUDGET_POLICY_ACTIONS,
                )
                self._family_state = LinUCBState(action_names=BUDGET_POLICY_FAMILIES)
                self.load_error = None
                return
            if schema_version != STATE_SCHEMA_VERSION:
                raise ValueError("unsupported schema version")
            architecture = str(raw.get("architecture") or BUDGET_POLICY_ARCH_TIER_BINARY)
            if architecture != self.config.architecture:
                raise ValueError("architecture mismatch")
            gate_raw = raw.get("gate_state")
            family_raw = raw.get("family_state")
            if not isinstance(gate_raw, dict):
                raise ValueError("missing gate_state")
            if not isinstance(family_raw, dict):
                family_raw = {"feature_names": [], "decisions": 0, "updates": 0}
            self._state = self._deserialize_state(
                raw_state=gate_raw,
                action_names=BUDGET_POLICY_ACTIONS,
            )
            self._family_state = self._deserialize_state(
                raw_state=family_raw,
                action_names=BUDGET_POLICY_FAMILIES,
            )
            if loading_from_init_state and self.config.architecture == BUDGET_POLICY_ARCH_BWK_V1:
                # Warm-starts should reuse learned controller state but begin a fresh run budget.
                self._bwk_run_token_budget = max(0, int(self.config.bwk_run_token_budget or 0))
                self._bwk_remaining_token_budget = int(self._bwk_run_token_budget)
                self._bwk_cumulative_tokens_charged = 0
                self._bwk_dual_variable = 0.0
            else:
                self._bwk_run_token_budget = max(
                    0,
                    _coerce_nonnegative_int(
                        raw.get("bwk_run_token_budget"),
                        default=int(self.config.bwk_run_token_budget or 0),
                    ),
                )
                self._bwk_remaining_token_budget = max(
                    0,
                    _coerce_nonnegative_int(
                        raw.get("bwk_remaining_token_budget"),
                        default=self._bwk_run_token_budget,
                    ),
                )
                self._bwk_cumulative_tokens_charged = max(
                    0,
                    _coerce_nonnegative_int(
                        raw.get("bwk_cumulative_tokens_charged"),
                        default=max(
                            0,
                            self._bwk_run_token_budget - self._bwk_remaining_token_budget,
                        ),
                    ),
                )
                self._bwk_dual_variable = max(
                    0.0,
                    _coerce_float(raw.get("bwk_dual_variable"), default=0.0),
                )
            self.load_error = None
        except Exception as exc:
            self._state = LinUCBState(action_names=BUDGET_POLICY_ACTIONS)
            self._family_state = LinUCBState(action_names=BUDGET_POLICY_FAMILIES)
            self._bwk_run_token_budget = max(0, int(self.config.bwk_run_token_budget or 0))
            self._bwk_remaining_token_budget = int(self._bwk_run_token_budget)
            self._bwk_cumulative_tokens_charged = 0
            self._bwk_dual_variable = 0.0
            self.load_error = f"{type(exc).__name__}: {exc}"

    def _append_event(self, *, event: BudgetPolicyEvent) -> None:
        """Append one JSONL event line to the events log file."""
        path = self.config.events_path
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event.to_json(), ensure_ascii=False))
            handle.write("\n")
