"""Adaptive online helper-invocation policy using LinUCB.

Implements a contextual-bandit controller that decides per-attempt whether
to invoke, skip, or route the helper call through a specific strategy
(file-only, frame-only, search-only, snapshot-lite, compact-then-patch,
or retry-strict-JSON).  Supports ``off`` / ``shadow`` / ``on`` modes with
epsilon-greedy exploration and warm-up uniform sampling.
"""

from __future__ import annotations

import random
import threading
import uuid
from collections.abc import Collection, Iterator, Mapping
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
from .adaptive_policy_records import EVENT_SCHEMA_VERSION as EVENT_SCHEMA_VERSION
from .adaptive_policy_records import PolicyDecision as PolicyDecision
from .adaptive_policy_records import PolicyEvent as PolicyEvent
from .decision_model.policy import (
    ROLE_FEATURES,
    ROLE_OFF,
    ROLE_POLICY,
    ROLE_PRIOR,
    DecisionModelAdvice,
    DecisionModelBinding,
    RoleSelection,
    advice_feature_values,
    advice_record,
    guided_warmup_selection,
    policy_role_selection,
    prior_score_adjustment,
)
from .policy_feature_scaling import (
    DEFAULT_POLICY_FEATURE_SCALING_HISTORY_LIMIT,
    DEFAULT_POLICY_FEATURE_SCALING_MIN_SAMPLES,
    DEFAULT_POLICY_FEATURE_SCALING_PROFILE,
    POLICY_FEATURE_SCALING_PROFILES,
    is_percentile_feature_scaling_profile,
    normalize_feature_scaling_profile,
    percentile_scale,
    prune_percentile_state,
    scale_feature_mapping,
)
from .seed_derivation import derive_seed

ACTION_SKIP_HELPER = "skip_helper"
ACTION_INVOKE_HELPER = "invoke_helper"
ACTION_FRAME_ONLY = "frame_only"
ACTION_FILE_ONLY = "file_only"
ACTION_SEARCH_ONLY = "search_only"
ACTION_SNAPSHOT_LITE = "snapshot_lite"
ACTION_FRAME_LOCALS_LITE = "frame_locals_lite"
ACTION_SNAPSHOT_FULL_LITE = "snapshot_full_lite"
ACTION_COMPACT_THEN_PATCH = "compact_then_patch"
ACTION_RETRY_HELPER_STRICT_JSON = "retry_helper_strict_json"
ONLINE_POLICY_ACTIONS = (
    ACTION_SKIP_HELPER,
    ACTION_INVOKE_HELPER,
    ACTION_FRAME_ONLY,
    ACTION_FILE_ONLY,
    ACTION_SEARCH_ONLY,
    ACTION_SNAPSHOT_LITE,
    ACTION_FRAME_LOCALS_LITE,
    ACTION_SNAPSHOT_FULL_LITE,
    ACTION_COMPACT_THEN_PATCH,
    ACTION_RETRY_HELPER_STRICT_JSON,
)
ONLINE_POLICY_MODES = {"off", "shadow", "on"}
ONLINE_POLICY_ALGOS = {"linucb", "thompson"}
ONLINE_ACTION_SCORE_BIAS_LIMIT = 0.1
"""Largest allowed |score bias|: a small prior, at most 0.1 of the success reward of 1.0
(decision D6, POL-07). Larger biases acted as a veto on the arm."""
DEFAULT_ONLINE_ACTION_SCORE_BIAS_OVERRIDES = {
    ACTION_SEARCH_ONLY: -0.10,
    ACTION_FRAME_ONLY: -0.05,
}
"""The one source of the default score biases (POL-07); the CLI default is derived from it."""
STATE_SCHEMA_VERSION = 4
# Schema 4 adds ``applied_action_counts`` (POL-24); schema 3 states still load.
SUPPORTED_STATE_SCHEMA_VERSIONS = (3, 4)
DEFAULT_ADAPTIVE_ONLINE_DECAY = 1.0
"""Decay is opt-in (decision D6, POL-01); below 1 it only discounts the updated arm."""


def default_online_action_score_biases() -> dict[str, float]:
    """Return the default additive score priors for helper actions."""
    return {
        action: float(DEFAULT_ONLINE_ACTION_SCORE_BIAS_OVERRIDES.get(action, 0.0))
        for action in ONLINE_POLICY_ACTIONS
    }


def format_online_action_score_biases(biases: Mapping[str, float]) -> str:
    """``action=bias`` pairs for the non-zero biases, in action order (the CLI format)."""
    return ",".join(
        f"{action}={float(biases[action]):.2f}"
        for action in ONLINE_POLICY_ACTIONS
        if float(biases.get(action, 0.0)) != 0.0
    )


DEFAULT_ADAPTIVE_ONLINE_ACTION_SCORE_BIASES = format_online_action_score_biases(
    DEFAULT_ONLINE_ACTION_SCORE_BIAS_OVERRIDES
)


def zero_online_action_score_biases() -> dict[str, float]:
    """Return a zero-bias score prior for every helper action."""
    return dict.fromkeys(ONLINE_POLICY_ACTIONS, 0.0)


def normalize_online_action_score_biases(
    action_score_biases: Mapping[str, Any] | None,
) -> dict[str, float]:
    """Normalize score-bias mappings onto the full helper action set."""
    if action_score_biases is None:
        return default_online_action_score_biases()
    normalized = zero_online_action_score_biases()
    for action in ONLINE_POLICY_ACTIONS:
        normalized[action] = _coerce_float(
            action_score_biases.get(action, 0.0),
            default=0.0,
        )
    return normalized


@dataclass(frozen=True)
class OnlinePolicyConfig:
    """Immutable configuration for the adaptive helper-invocation policy."""

    mode: str = "off"
    algorithm: str = "linucb"
    alpha: float = 0.8
    epsilon: float = 0.15
    l2: float = 1.0
    warmup_decisions: int = 50
    save_every: int = 50
    decay: float = DEFAULT_ADAPTIVE_ONLINE_DECAY
    feature_scaling_profile: str = DEFAULT_POLICY_FEATURE_SCALING_PROFILE
    init_state_path: Path | None = None
    state_path: Path | None = None
    events_path: Path | None = None
    random_seed: int | None = None
    action_score_biases: dict[str, float] = field(
        default_factory=default_online_action_score_biases
    )

    @classmethod
    def normalize(
        cls,
        *,
        mode: str = "off",
        algorithm: str = "linucb",
        alpha: float = 0.8,
        epsilon: float = 0.15,
        l2: float = 1.0,
        warmup_decisions: int = 50,
        save_every: int = 50,
        decay: float = DEFAULT_ADAPTIVE_ONLINE_DECAY,
        feature_scaling_profile: str = DEFAULT_POLICY_FEATURE_SCALING_PROFILE,
        init_state_path: Path | None = None,
        state_path: Path | None = None,
        events_path: Path | None = None,
        random_seed: int | None = None,
        action_score_biases: Mapping[str, Any] | None = None,
    ) -> OnlinePolicyConfig:
        """Validate and clamp raw parameters into a normalised config."""
        mode_norm = str(mode).strip().lower()
        if mode_norm not in ONLINE_POLICY_MODES:
            mode_norm = "off"
        algo_norm = str(algorithm).strip().lower()
        if algo_norm not in ONLINE_POLICY_ALGOS:
            algo_norm = "linucb"
        alpha_norm = max(0.0, _coerce_float(alpha, default=0.8))
        epsilon_norm = min(1.0, max(0.0, _coerce_float(epsilon, default=0.15)))
        l2_norm = max(1e-9, _coerce_float(l2, default=1.0))
        warmup_norm = _coerce_nonnegative_int(warmup_decisions, default=50)
        save_every_norm = max(1, _coerce_nonnegative_int(save_every, default=50))
        decay_norm = min(
            1.0, max(1e-9, _coerce_float(decay, default=DEFAULT_ADAPTIVE_ONLINE_DECAY))
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
        return cls(
            mode=mode_norm,
            algorithm=algo_norm,
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
            random_seed=seed_norm,
            action_score_biases=normalize_online_action_score_biases(action_score_biases),
        )


@dataclass
class LinUCBState(core.LinUCBState):
    """The shared LinUCB core state over the helper actions."""

    action_names: tuple[str, ...] = ONLINE_POLICY_ACTIONS


class AdaptiveOnlinePolicyController:
    """Thread-safe LinUCB contextual-bandit controller for helper invocation.

    Manages exploration/exploitation via epsilon-greedy with UCB scores,
    persists model state to disk, and logs decision-outcome events as JSONL.
    """

    def __init__(
        self,
        config: OnlinePolicyConfig,
        decision_model: DecisionModelBinding | None = None,
    ) -> None:
        self.config = config
        self.decision_model = decision_model
        self._lock = threading.RLock()
        self._rng = random.Random(config.random_seed)
        self._state = LinUCBState()
        self._save_counter = 0
        self._nan_counts: dict[str, int] = {}
        # POL-21: cumulative model resets (``arm:<action>``, ``feature_set_mismatch``).
        self._state_resets: dict[str, int] = {}
        self._check_loaded_features = False
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
    def decision_model_role(self) -> str:
        """Role of the attached decision model (``off`` when none is attached)."""
        return self.decision_model.role if self.decision_model is not None else ROLE_OFF

    @property
    def state(self) -> LinUCBState:
        """Return a deep copy of the internal LinUCB state."""
        with self._lock:
            return LinUCBState(
                feature_names=list(self._state.feature_names),
                decisions=int(self._state.decisions),
                updates=int(self._state.updates),
                action_counts={k: int(v) for k, v in self._state.action_counts.items()},
                applied_action_counts={
                    k: int(v) for k, v in self._state.applied_action_counts.items()
                },
                A={k: v.copy() for k, v in self._state.A.items()},
                b={k: v.copy() for k, v in self._state.b.items()},
                feature_scaling_history={
                    k: [float(item) for item in values]
                    for k, values in self._state.feature_scaling_history.items()
                },
                feature_scaling_caps={
                    k: float(v) for k, v in self._state.feature_scaling_caps.items()
                },
            )

    def get_nan_stats(self) -> dict[str, int]:
        """Return a snapshot of per-feature NaN replacement counts."""
        with self._lock:
            return dict(self._nan_counts)

    def get_state_resets(self) -> dict[str, int]:
        """Return a snapshot of the model reset counts (POL-21)."""
        with self._lock:
            return dict(self._state_resets)

    def _count_state_reset(self, key: str) -> None:
        self._state_resets[key] = self._state_resets.get(key, 0) + 1

    def _discard_mismatched_loaded_state(self, feature_names: Collection[str]) -> None:
        """Treat a loaded state that lacks this run's features as a load failure (POL-21).

        The features a model was trained on must all still be produced; zero-filling a
        removed feature would silently reinterpret its learned weights.
        """
        if not self._check_loaded_features:
            return
        self._check_loaded_features = False
        missing = core.missing_features(self._state, feature_names)
        if missing:
            self.load_error = core.feature_set_mismatch_error(missing)
            self._state = LinUCBState()
            self._count_state_reset("feature_set_mismatch")

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

    def _scale_features(
        self, *, cleaned_features: Mapping[str, float]
    ) -> tuple[dict[str, float], dict[str, float], dict[str, int], bool]:
        """Scale features; under ``percentile_v1`` also collect history and freeze caps.

        Each count-like feature's cap is fitted once from its first observations and then
        kept (POL-09); see ``policy_feature_scaling.percentile_scale``.
        """
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
        scaled = percentile_scale(
            cleaned_features,
            history=self._state.feature_scaling_history,
            caps=self._state.feature_scaling_caps,
            min_samples=DEFAULT_POLICY_FEATURE_SCALING_MIN_SAMPLES,
        )
        return (
            scaled.values,
            dict(scaled.caps),
            dict(scaled.history_sizes),
            scaled.fallback_to_v1,
        )

    def _normalize_available_actions(
        self,
        *,
        default_action: str,
        available_actions: Collection[str] | None,
    ) -> tuple[str, ...]:
        """Normalize and order the feasible action set for a decision."""
        if available_actions is None:
            return tuple(ONLINE_POLICY_ACTIONS)
        available_set = {
            str(action) for action in available_actions if str(action) in ONLINE_POLICY_ACTIONS
        }
        ordered = tuple(action for action in ONLINE_POLICY_ACTIONS if action in available_set)
        if ordered:
            return ordered
        if default_action in ONLINE_POLICY_ACTIONS:
            return (default_action,)
        return (ACTION_INVOKE_HELPER,)

    def _baseline_action(
        self,
        *,
        default_action: str,
        available_actions: tuple[str, ...],
    ) -> str:
        """Choose the feasible baseline action used in off/shadow modes."""
        if default_action in available_actions:
            return default_action
        if ACTION_SKIP_HELPER in available_actions:
            return ACTION_SKIP_HELPER
        return str(available_actions[0])

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
        default_action: str = ACTION_INVOKE_HELPER,
        available_actions: Collection[str] | None = None,
        unavailable_action_reasons: Mapping[str, str] | None = None,
        advice: DecisionModelAdvice | None = None,
        rng_key: str | None = None,
    ) -> PolicyDecision:
        """Select an action given the current context features.

        During warm-up or with probability *epsilon* the action is sampled
        uniformly; otherwise the LinUCB greedy arm is chosen.  In ``shadow``
        mode the selected action is logged but the *default_action* is
        applied.  With an attached decision model, *advice* is only logged
        (``shadow``), selects the action (``policy``), adds diagnostic
        features (``features``), or guides warm-up and adds a fading score
        prior (``prior``); see ``evals.decision_model.policy``.
        """
        role = self.decision_model_role
        cleaned_features = {
            str(k): self._tracked_safe_float(v, feature_name=str(k), default=0.0)
            for k, v in dict(feature_values).items()
        }
        if role == ROLE_FEATURES:
            cleaned_features.update(advice_feature_values(advice))
        with self._lock, self._decision_rng(rng_key):
            self._discard_mismatched_loaded_state(cleaned_features)
            (
                scaled_features,
                feature_scaling_caps,
                feature_scaling_history_sizes,
                feature_scaling_fallback_to_v1,
            ) = self._scale_features(cleaned_features=cleaned_features)
            self._ensure_feature_state(cleaned_features=scaled_features)
            decision_index = self._state.decisions + 1
            mode = self.config.mode
            default_norm = (
                default_action if default_action in ONLINE_POLICY_ACTIONS else ACTION_INVOKE_HELPER
            )
            effective_epsilon, warmup_active = _policy_effective_epsilon(
                mode=mode,
                decision_index=decision_index,
                warmup_decisions=self.config.warmup_decisions,
                epsilon=self.config.epsilon,
            )
            available_actions_norm = self._normalize_available_actions(
                default_action=default_norm,
                available_actions=available_actions,
            )
            unavailable_action_reasons_norm = {
                str(action): str(reason)
                for action, reason in dict(unavailable_action_reasons or {}).items()
                if str(action) in ONLINE_POLICY_ACTIONS and str(reason).strip()
            }
            baseline_action = self._baseline_action(
                default_action=default_norm,
                available_actions=available_actions_norm,
            )
            decision_model_effect: dict[str, Any] = {}
            if mode == "off":
                selected = baseline_action
                applied = baseline_action
                propensity = 1.0
                strategy = "policy_off"
                scores = dict.fromkeys(ONLINE_POLICY_ACTIONS, 0.0)
            else:
                x = self._feature_vector(cleaned_features=scaled_features)
                if self.config.algorithm == "thompson":
                    scores = self._apply_action_score_biases(self._thompson_scores(x=x))
                else:
                    scores = self._apply_action_score_biases(self._linucb_scores(x=x))
                ranking = self._ranking_scores(
                    scores, advice, available_actions_norm, decision_model_effect
                )
                tied = core.greedy_ties(ranking, available_actions_norm)
                role_selection = self._role_selection(
                    advice, available_actions_norm, decision_index, baseline_action
                )
                if role_selection is not None:
                    selected = role_selection.action
                    strategy = role_selection.strategy
                    propensity = role_selection.propensity
                    warmup_active = role_selection.warmup_active
                    effective_epsilon = role_selection.effective_epsilon
                    decision_model_effect.update(role_selection.effect)
                else:
                    if decision_index <= self.config.warmup_decisions:
                        selected = self._rng.choice(list(available_actions_norm))
                        strategy = "warmup_uniform"
                    elif self._rng.random() < self.config.epsilon:
                        selected = self._rng.choice(list(available_actions_norm))
                        strategy = "epsilon_uniform"
                    else:
                        selected = core.pick_greedy(tied, self._rng)
                        strategy = f"{self.config.algorithm}_greedy"
                    if decision_index <= self.config.warmup_decisions:
                        propensity = 1.0 / float(len(available_actions_norm))
                    else:
                        propensity = core.epsilon_greedy_propensity(
                            selected,
                            tied,
                            epsilon=self.config.epsilon,
                            n_available=len(available_actions_norm),
                        )
                propensity = min(1.0, max(1e-9, propensity))
                applied = selected if mode == "on" else baseline_action

            self._state.decisions = decision_index
            self._state.action_counts[selected] = (
                int(self._state.action_counts.get(selected, 0)) + 1
            )
            self._state.applied_action_counts[applied] = (
                int(self._state.applied_action_counts.get(applied, 0)) + 1
            )
            return PolicyDecision(
                decision_id=uuid.uuid4().hex,
                mode=mode,
                algorithm=self.config.algorithm,
                decision_index=decision_index,
                created_at_utc=_utc_now(),
                selected_action=selected,
                applied_action=applied,
                default_action=default_norm,
                propensity=float(propensity),
                decision_strategy=strategy,
                scores={key: float(value) for key, value in scores.items()},
                feature_values=scaled_features,
                available_actions=available_actions_norm,
                unavailable_action_reasons=unavailable_action_reasons_norm,
                configured_action_score_biases=dict(self.config.action_score_biases),
                feature_scaling_caps=feature_scaling_caps,
                feature_scaling_history_sizes=feature_scaling_history_sizes,
                feature_scaling_fallback_to_v1=feature_scaling_fallback_to_v1,
                configured_alpha=float(self.config.alpha),
                configured_epsilon=float(self.config.epsilon),
                configured_decay=float(self.config.decay),
                effective_epsilon=float(effective_epsilon),
                warmup_active=bool(warmup_active),
                decision_model=advice_record(advice, role=role, effect=decision_model_effect),
            )

    def _ranking_scores(
        self,
        scores: Mapping[str, float],
        advice: DecisionModelAdvice | None,
        available_actions: tuple[str, ...],
        effect: dict[str, Any],
    ) -> dict[str, float]:
        """Scores for greedy ranking; the ``prior`` role adds its fading log-prior here.

        The logged ``scores`` stay the bandit's own; the adjustment is logged in *effect*.
        """
        ranking = {action: float(value) for action, value in scores.items()}
        binding = self.decision_model
        if binding is None or binding.role != ROLE_PRIOR:
            return ranking
        adjustment = prior_score_adjustment(
            advice,
            available_actions=available_actions,
            config=binding.config,
            updates=int(self._state.updates),
        )
        if adjustment is None:
            return ranking
        effect.update(adjustment)
        for action, delta in adjustment["prior_adjustment"].items():
            ranking[action] = ranking.get(action, 0.0) + float(delta)
        return ranking

    def _role_selection(
        self,
        advice: DecisionModelAdvice | None,
        available_actions: tuple[str, ...],
        decision_index: int,
        baseline_action: str,
    ) -> RoleSelection | None:
        """Selection made by the decision-model role, or ``None`` for the bandit path."""
        binding = self.decision_model
        if binding is None:
            return None
        if binding.role == ROLE_POLICY:
            selection = policy_role_selection(
                advice,
                available_actions=available_actions,
                config=binding.config,
                epsilon=float(self.config.epsilon),
                rng=self._rng,
            )
            if selection is not None:
                return selection
            return RoleSelection(
                action=baseline_action,
                propensity=1.0,
                strategy="decision_model_unavailable",
                warmup_active=False,
                effective_epsilon=0.0,
                effect={},
            )
        if binding.role == ROLE_PRIOR and decision_index <= self.config.warmup_decisions:
            return guided_warmup_selection(
                advice,
                available_actions=available_actions,
                config=binding.config,
                rng=self._rng,
            )
        return None

    def record_outcome(
        self,
        *,
        decision: PolicyDecision,
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
        reward_helper_materialization_bonus: float,
        success: bool,
        local_llm_call_count: int,
        local_tokens_prompt: int,
        local_tokens_completion: int,
        local_tokens_total: int,
        local_wall_sec: float,
        override_reason: str,
        reward_success_hint_scale: float = 1.0,
        reward_progress_hint_scale: float = 1.0,
        reward_stagnation_penalty_scale: float = 1.0,
        reward_helper_materialization_bonus_scale: float = 0.5,
        reward_delayed_rescue_bonus: float = 0.0,
        reward_credit_resolved_attempt: int = 0,
        reward_credit_horizon_attempts: int = 0,
        reward_credit_resolution_reason: str = "",
        behavior_action: str | None = None,
        behavior_propensity: float | None = None,
        allow_model_update: bool = True,
        update_reason_override: str | None = None,
    ) -> PolicyEvent:
        """Record an observed reward for a prior decision.

        Updates the LinUCB model when mode is ``on``, appends a JSONL
        event record, and triggers periodic state persistence.
        """
        with self._lock:
            selected_action = str(decision.selected_action)
            applied_action = str(decision.applied_action)
            behavior_action_norm = str(behavior_action or applied_action)
            if behavior_action_norm not in ONLINE_POLICY_ACTIONS:
                behavior_action_norm = (
                    applied_action if applied_action in ONLINE_POLICY_ACTIONS else "none"
                )
            if behavior_propensity is None:
                # Only an applied ("on") selection was sampled by the policy; in shadow mode the
                # executed baseline action is deterministic, so its probability is 1.
                behavior_propensity_norm = core.behavior_propensity(
                    mode=decision.mode,
                    selected=selected_action,
                    behavior=behavior_action_norm,
                    propensity=decision.propensity,
                )
            else:
                behavior_propensity_norm = min(1.0, max(1e-9, float(behavior_propensity)))
            model_updated = False
            update_reason = "policy_not_enabled"
            if not allow_model_update:
                update_reason = str(update_reason_override or "update_suppressed")
            elif self.config.mode == "on" and behavior_action_norm in ONLINE_POLICY_ACTIONS:
                model_updated = self._update_model(
                    action=behavior_action_norm,
                    reward=reward,
                    feature_values=decision.feature_values,
                )
                update_reason = (
                    "on_policy_update" if model_updated else "update_skipped_incomplete_state"
                )
            elif self.config.mode == "shadow":
                update_reason = "shadow_no_update"
            elif self.config.mode == "off":
                update_reason = "off_no_update"

            event = PolicyEvent(
                run_id=str(run_id),
                case_id=str(case_id),
                condition=str(condition),
                attempt=max(0, int(attempt)),
                note=str(note),
                decision_id=str(decision.decision_id),
                mode=str(decision.mode),
                algorithm=str(decision.algorithm),
                decision_index=max(0, int(decision.decision_index)),
                selected_action=selected_action,
                applied_action=applied_action,
                behavior_action=behavior_action_norm,
                default_action=str(decision.default_action),
                propensity=min(1.0, max(1e-9, float(decision.propensity))),
                behavior_propensity=behavior_propensity_norm,
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
                available_actions=tuple(str(action) for action in decision.available_actions),
                unavailable_action_reasons={
                    str(k): str(v) for k, v in decision.unavailable_action_reasons.items()
                },
                configured_action_score_biases={
                    str(k): float(v) for k, v in decision.configured_action_score_biases.items()
                },
                reward_scope=str(reward_scope),
                reward_success_hint_scale=float(reward_success_hint_scale),
                reward_progress_hint_scale=float(reward_progress_hint_scale),
                reward_stagnation_penalty_scale=float(reward_stagnation_penalty_scale),
                reward_helper_materialization_bonus_scale=float(
                    reward_helper_materialization_bonus_scale
                ),
                reward=float(reward),
                reward_success_hint=float(reward_success_hint),
                reward_progress_hint=float(reward_progress_hint),
                reward_progress_score=float(reward_progress_score),
                reward_token_penalty=float(reward_token_penalty),
                reward_latency_penalty=float(reward_latency_penalty),
                reward_stagnation_penalty=float(reward_stagnation_penalty),
                reward_helper_materialization_bonus=float(reward_helper_materialization_bonus),
                reward_delayed_rescue_bonus=float(reward_delayed_rescue_bonus),
                reward_credit_resolved_attempt=max(0, int(reward_credit_resolved_attempt)),
                reward_credit_horizon_attempts=max(0, int(reward_credit_horizon_attempts)),
                reward_credit_resolution_reason=str(reward_credit_resolution_reason),
                success=bool(success),
                local_llm_call_count=max(0, int(local_llm_call_count)),
                local_tokens_prompt=max(0, int(local_tokens_prompt)),
                local_tokens_completion=max(0, int(local_tokens_completion)),
                local_tokens_total=max(0, int(local_tokens_total)),
                local_wall_sec=max(0.0, float(local_wall_sec)),
                override_reason=str(override_reason),
                model_updated=bool(model_updated),
                effective_evidence=core.effective_evidence(self._state, l2=float(self.config.l2)),
                update_reason=update_reason,
                nan_replacement_counts=dict(self._nan_counts),
                state_resets=dict(self._state_resets),
                created_at_utc=_utc_now(),
                decision_model=dict(getattr(decision, "decision_model", None) or {}),
            )
            self._append_event(event=event)
            if model_updated:
                self._save_counter += 1
                if self._save_counter >= max(1, int(self.config.save_every)):
                    self._save_counter = 0
                    self._save_state()
            return event

    def close(self) -> None:
        """Flush the current model state to disk and release the decision-model client."""
        with self._lock:
            self._save_state()
        advisor = self.decision_model.advisor if self.decision_model is not None else None
        close_advisor = getattr(advisor, "close", None)
        if callable(close_advisor):
            close_advisor()

    def _ensure_feature_state(self, *, cleaned_features: Mapping[str, float]) -> None:
        """Initialise or expand A/b matrices when new features appear."""
        if self.config.algorithm not in ONLINE_POLICY_ALGOS:
            return
        change = core.ensure_feature_state(self._state, cleaned_features, l2=float(self.config.l2))
        for action in change.reset_arms:
            self._count_state_reset(f"arm:{action}")

    def _feature_vector(self, *, cleaned_features: Mapping[str, float]) -> np.ndarray:
        """Build a numpy feature vector aligned with ``feature_names``."""
        return core.feature_vector(
            self._state,
            cleaned_features,
            to_float=lambda value, name: self._tracked_safe_float(
                value, feature_name=name, default=0.0
            ),
        )

    def _linucb_scores(self, *, x: np.ndarray) -> dict[str, float]:
        """Compute UCB scores (exploit + alpha * explore) for every action."""
        if self.config.algorithm != "linucb":
            return dict.fromkeys(ONLINE_POLICY_ACTIONS, 0.0)
        return core.linucb_scores(
            self._state, x, l2=float(self.config.l2), alpha=float(self.config.alpha)
        )

    def _thompson_scores(self, *, x: np.ndarray) -> dict[str, float]:
        """Sample θ ~ N(A⁻¹b, A⁻¹) and return θ·x for each action (Thompson Sampling)."""
        if self.config.algorithm != "thompson":
            return dict.fromkeys(ONLINE_POLICY_ACTIONS, 0.0)
        return core.thompson_scores(self._state, x, seed=self.config.random_seed)

    def _apply_action_score_biases(self, scores: Mapping[str, float]) -> dict[str, float]:
        """Apply additive score priors without changing feasibility masks."""
        return {
            action: float(scores.get(action, 0.0))
            + float(self.config.action_score_biases.get(action, 0.0))
            for action in ONLINE_POLICY_ACTIONS
        }

    def _update_model(
        self,
        *,
        action: str,
        reward: float,
        feature_values: Mapping[str, float],
    ) -> bool:
        """Apply a rank-1 LinUCB update: A += x x^T, b += reward * x."""
        if self.config.algorithm not in ONLINE_POLICY_ALGOS:
            return False
        if action not in ONLINE_POLICY_ACTIONS:
            return False
        if not self._state.feature_names:
            return False
        x = self._feature_vector(cleaned_features=feature_values)
        self._apply_recency_decay(action=action)
        return core.rank_one_update(self._state, action=action, reward=reward, x=x)

    def _apply_recency_decay(self, *, action: str) -> None:
        """Discount the updated arm's prior evidence when decay < 1 is configured."""
        core.apply_recency_decay(
            self._state, action=action, decay=float(self.config.decay), l2=float(self.config.l2)
        )

    def _state_payload(self) -> dict[str, Any]:
        """Build a JSON-serialisable snapshot of the full model state."""
        return {
            "schema_version": STATE_SCHEMA_VERSION,
            "mode": self.config.mode,
            "algorithm": self.config.algorithm,
            "alpha": self.config.alpha,
            "epsilon": self.config.epsilon,
            "l2": self.config.l2,
            "warmup_decisions": self.config.warmup_decisions,
            "save_every": self.config.save_every,
            "decay": self.config.decay,
            "feature_scaling_profile": self.config.feature_scaling_profile,
            "action_score_biases": {
                action: float(self.config.action_score_biases.get(action, 0.0))
                for action in ONLINE_POLICY_ACTIONS
            },
            "feature_names": list(self._state.feature_names),
            "decisions": int(self._state.decisions),
            "updates": int(self._state.updates),
            "action_counts": {
                action: int(self._state.action_counts.get(action, 0))
                for action in ONLINE_POLICY_ACTIONS
            },
            "applied_action_counts": {
                action: int(self._state.applied_action_counts.get(action, 0))
                for action in ONLINE_POLICY_ACTIONS
            },
            "A": {
                action: self._state.A.get(action, np.empty((0, 0), dtype=float)).tolist()
                for action in ONLINE_POLICY_ACTIONS
            },
            "b": {
                action: self._state.b.get(action, np.empty((0,), dtype=float)).tolist()
                for action in ONLINE_POLICY_ACTIONS
            },
            "feature_scaling_history": {
                name: [float(item) for item in values]
                for name, values in self._state.feature_scaling_history.items()
            },
            "feature_scaling_caps": {
                name: float(value) for name, value in self._state.feature_scaling_caps.items()
            },
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
        """Load persisted controller state from disk, if it exists."""
        path = self.config.init_state_path or self.config.state_path
        if path is None or not path.exists():
            return
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                raise ValueError("state payload is not an object")
            if int(raw.get("schema_version", 0)) not in SUPPORTED_STATE_SCHEMA_VERSIONS:
                raise ValueError("unsupported schema version")
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
            feature_names_raw = raw.get("feature_names")
            if not isinstance(feature_names_raw, list) or not all(
                isinstance(item, str) for item in feature_names_raw
            ):
                raise ValueError("missing or invalid feature_names")
            feature_names = [str(item) for item in feature_names_raw]
            dim = len(feature_names)
            action_counts = core.parse_action_counts(
                raw.get("action_counts"), ONLINE_POLICY_ACTIONS
            )
            # Schema 3 states predate applied counts (POL-24); they start at zero.
            applied_action_counts = core.parse_action_counts(
                raw.get("applied_action_counts"), ONLINE_POLICY_ACTIONS
            )
            decisions = _coerce_nonnegative_int(raw.get("decisions"), default=0)
            updates = _coerce_nonnegative_int(raw.get("updates"), default=0)
            feature_scaling_history_raw = raw.get("feature_scaling_history")
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
            feature_scaling_caps_raw = raw.get("feature_scaling_caps")
            feature_scaling_caps: dict[str, float] = {}
            if isinstance(feature_scaling_caps_raw, dict):
                for key, value in feature_scaling_caps_raw.items():
                    if isinstance(value, (int, float)) and np.isfinite(float(value)):
                        feature_scaling_caps[str(key)] = float(value)
            feature_scaling_history, feature_scaling_caps = prune_percentile_state(
                feature_scaling_history,
                feature_scaling_caps,
                min_samples=DEFAULT_POLICY_FEATURE_SCALING_MIN_SAMPLES,
            )
            if dim <= 0:
                self._state = LinUCBState(
                    feature_names=[],
                    decisions=decisions,
                    updates=updates,
                    action_counts=action_counts,
                    applied_action_counts=applied_action_counts,
                    A={},
                    b={},
                    feature_scaling_history=feature_scaling_history,
                    feature_scaling_caps=feature_scaling_caps,
                )
                self.load_error = None
                return
            A_raw = raw.get("A")
            b_raw = raw.get("b")
            if not isinstance(A_raw, dict) or not isinstance(b_raw, dict):
                raise ValueError("missing matrices")
            A: dict[str, np.ndarray] = {}
            b: dict[str, np.ndarray] = {}
            for action in ONLINE_POLICY_ACTIONS:
                matrix_raw = A_raw.get(action)
                vector_raw = b_raw.get(action)
                matrix = np.array(matrix_raw, dtype=float)
                vector = np.array(vector_raw, dtype=float)
                if matrix.shape != (dim, dim):
                    raise ValueError(f"invalid matrix shape for {action}: {matrix.shape}")
                if vector.shape != (dim,):
                    raise ValueError(f"invalid vector shape for {action}: {vector.shape}")
                A[action] = matrix
                b[action] = vector
            self._state = LinUCBState(
                feature_names=feature_names,
                decisions=decisions,
                updates=updates,
                action_counts=action_counts,
                applied_action_counts=applied_action_counts,
                A=A,
                b=b,
                feature_scaling_history=feature_scaling_history,
                feature_scaling_caps=feature_scaling_caps,
            )
            self.load_error = None
            self._check_loaded_features = True
        except Exception as exc:  # pragma: no cover - defensive load hardening
            self.load_error = f"{type(exc).__name__}: {exc}"
            self._state = LinUCBState()

    def _append_event(self, *, event: PolicyEvent) -> None:
        """Append one JSONL event line to the events log file."""
        path = self.config.events_path
        if path is None:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event.to_json(), ensure_ascii=False) + "\n")
