"""Shared LinUCB core for the adaptive policy controllers (2026-10-04 review, Phase 3.1).

The helper controller (``adaptive_online_policy``), the budget gate and family controllers
(``adaptive_budget_online_policy``) and the bootstrap replay (``bootstrap_policy_state``)
each carried their own copy of the same state handling and linear algebra: feature-set
expansion, the feature vector, LinUCB and Thompson scores, the rank-1 update and recency
decay. They now share these functions, so each later fix lands once (X-04).

The greedy helpers break ties uniformly at random and give the matching
epsilon-greedy propensity (POL-24). Each arm's inverse matrix is cached and kept current with
Sherman-Morrison rank-1 updates (POL-20).
"""

from __future__ import annotations

import random
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from ._policy_utils import coerce_nonnegative_int, linucb_exploit_explore

# Relative tolerance under which two greedy scores count as tied (POL-24).
GREEDY_TIE_TOLERANCE = 1e-12
# Sherman-Morrison updates of a cached inverse before it is recomputed exactly (POL-20).
INVERSE_REFRESH_UPDATES = 64


def _normalized_counts(actions: tuple[str, ...], raw: Mapping[str, Any]) -> dict[str, int]:
    counts = dict.fromkeys(actions, 0)
    for key, value in dict(raw).items():
        if key in counts:
            try:
                counts[key] = int(value)
            except (TypeError, ValueError):
                counts[key] = 0
    return counts


@dataclass
class LinUCBState:
    """Per-action LinUCB model: ``A`` matrices, ``b`` vectors and bookkeeping."""

    action_names: tuple[str, ...] = ("default",)
    feature_names: list[str] = field(default_factory=list)
    decisions: int = 0
    updates: int = 0
    action_counts: dict[str, int] = field(default_factory=dict)
    applied_action_counts: dict[str, int] = field(default_factory=dict)
    A: dict[str, np.ndarray] = field(default_factory=dict)
    b: dict[str, np.ndarray] = field(default_factory=dict)
    feature_scaling_history: dict[str, list[float]] = field(default_factory=dict)
    feature_scaling_caps: dict[str, float] = field(default_factory=dict)
    inverse_cache: dict[str, tuple[np.ndarray, np.ndarray, int]] = field(
        default_factory=dict, repr=False, compare=False
    )
    """Per arm: (the ``A`` it inverts, ``A^-1``, Sherman-Morrison steps since an exact
    inverse). Not persisted. An entry is valid only while ``A`` is that same object, so
    decay, feature expansion, arm resets and loads (which all build new matrices)
    invalidate it (POL-20)."""

    def __post_init__(self) -> None:
        normalized_actions = tuple(str(action) for action in self.action_names) or ("default",)
        self.action_names = normalized_actions
        # ``action_counts`` counts the selected action, ``applied_action_counts`` the one
        # that ran; they differ in shadow mode (POL-24).
        self.action_counts = _normalized_counts(normalized_actions, self.action_counts)
        self.applied_action_counts = _normalized_counts(
            normalized_actions, self.applied_action_counts
        )


def parse_action_counts(raw: object, actions: Iterable[str]) -> dict[str, int]:
    """Per-action counts from a persisted state payload (missing or bad entries are 0)."""
    source: Mapping[str, Any] = raw if isinstance(raw, Mapping) else {}
    return {
        str(action): coerce_nonnegative_int(source.get(str(action)), default=0)
        for action in actions
    }


def greedy_ties(scores: Mapping[str, float], available: Sequence[str]) -> list[str]:
    """Available actions whose score ties the best one, in *available* order (POL-24)."""
    values = {action: float(scores.get(action, 0.0)) for action in available}
    best = max(values.values())
    tolerance = GREEDY_TIE_TOLERANCE * max(1.0, abs(best))
    return [action for action in available if values[action] >= best - tolerance]


def pick_greedy(tied: Sequence[str], rng: random.Random) -> str:
    """The greedy action: the only tied action, or a uniform draw among the ties.

    The draw happens only when there is a tie, so untied decisions consume the same
    random numbers as before.
    """
    return tied[0] if len(tied) == 1 else rng.choice(list(tied))


def behavior_propensity(*, mode: str, selected: str, behavior: str, propensity: float) -> float:
    """Probability that the logging policy executed *behavior* (POL-03).

    Only an applied (``on``) selection was sampled by the policy. In ``shadow`` and ``off``
    mode, or when an override replaced the selection, the executed action was
    deterministic, so its probability is 1. Using the selection's propensity there gave
    shadow events weights up to 10x during warm-up.
    """
    if str(mode) == "on" and str(selected) == str(behavior):
        return min(1.0, max(1e-9, float(propensity)))
    return 1.0


def epsilon_greedy_propensity(
    selected: str, tied: Sequence[str], *, epsilon: float, n_available: int
) -> float:
    """P(selected) under epsilon-greedy with uniform tie-breaking among *tied*."""
    propensity = float(epsilon) / float(n_available)
    if selected in tied:
        propensity += (1.0 - float(epsilon)) / float(len(tied))
    return propensity


@dataclass(frozen=True)
class FeatureStateChange:
    """What ``ensure_feature_state`` changed: features added, arms reset to the prior."""

    added: tuple[str, ...] = ()
    reset_arms: tuple[str, ...] = ()


def missing_features(state: LinUCBState, feature_names: Iterable[str]) -> list[str]:
    """Features the model was trained on that *feature_names* lacks (POL-21)."""
    if not state.feature_names:
        return []
    present = {str(name) for name in feature_names}
    return sorted(name for name in state.feature_names if name not in present)


def feature_set_mismatch_error(missing: Sequence[str]) -> str:
    """The ``load_error`` for a loaded state whose features this run doesn't produce."""
    shown = ", ".join(missing[:5])
    more = f" (+{len(missing) - 5} more)" if len(missing) > 5 else ""
    return (
        "FeatureSetMismatch: the loaded state was trained on features this run doesn't "
        f"produce: {shown}{more}"
    )


def ensure_feature_state(
    state: LinUCBState, feature_names: Iterable[str], *, l2: float
) -> FeatureStateChange:
    """Initialise the model on first use, or expand it when new features appear.

    Existing rows and columns are carried over; new features start from the ``l2``
    prior. Arms whose matrices have the wrong shape are reset to the prior and reported in
    ``reset_arms`` (POL-21). Features the model knows but *feature_names* lacks read as 0;
    callers check a loaded state with ``missing_features`` first.
    """
    seen = sorted(str(name) for name in feature_names)
    if not state.feature_names:
        state.feature_names = seen or ["bias"]
        dim = len(state.feature_names)
        state.A = {action: np.eye(dim, dtype=float) * float(l2) for action in state.action_names}
        state.b = {action: np.zeros((dim,), dtype=float) for action in state.action_names}
        return FeatureStateChange(added=tuple(state.feature_names))
    known = set(state.feature_names)
    added = tuple(name for name in seen if name not in known)
    reset_arms: list[str] = []
    expanded = sorted(set(state.feature_names) | set(seen))
    if expanded != state.feature_names:
        previous_index = {name: idx for idx, name in enumerate(state.feature_names)}
        previous_dim = len(state.feature_names)
        new_dim = len(expanded)
        state.feature_names = list(expanded)
        for action in state.action_names:
            previous_A = state.A.get(action)
            previous_b = state.b.get(action)
            next_A = np.eye(new_dim, dtype=float) * float(l2)
            next_b = np.zeros((new_dim,), dtype=float)
            if not (
                isinstance(previous_A, np.ndarray)
                and isinstance(previous_b, np.ndarray)
                and previous_A.shape == (previous_dim, previous_dim)
                and previous_b.shape == (previous_dim,)
            ):
                reset_arms.append(action)
            else:
                for new_row, feature_row in enumerate(expanded):
                    old_row = previous_index.get(feature_row)
                    if old_row is None:
                        continue
                    next_b[new_row] = previous_b[old_row]
                    for new_col, feature_col in enumerate(expanded):
                        old_col = previous_index.get(feature_col)
                        if old_col is None:
                            continue
                        next_A[new_row, new_col] = previous_A[old_row, old_col]
            state.A[action] = next_A
            state.b[action] = next_b
    dim = len(state.feature_names)
    for action in state.action_names:
        A = state.A.get(action)
        b = state.b.get(action)
        if A is None or A.shape != (dim, dim) or b is None or b.shape != (dim,):
            state.A[action] = np.eye(dim, dtype=float) * float(l2)
            state.b[action] = np.zeros((dim,), dtype=float)
            if action not in reset_arms:
                reset_arms.append(action)
    return FeatureStateChange(added=added, reset_arms=tuple(reset_arms))


def feature_vector(
    state: LinUCBState,
    values: Mapping[str, float],
    *,
    to_float: Callable[[object, str], float],
) -> np.ndarray:
    """Feature values in ``state.feature_names`` order (missing features are 0)."""
    names = state.feature_names or ["bias"]
    return np.array([to_float(values.get(name, 0.0), name) for name in names], dtype=float)


def arm_inverse(state: LinUCBState, action: str) -> np.ndarray | None:
    """``A^-1`` for *action* from the cache, computed when missing or stale (POL-20).

    ``None`` when the arm has no matrix or it is singular; nothing is cached then.
    """
    A = state.A.get(action)
    if A is None:
        return None
    cached = state.inverse_cache.get(action)
    if cached is not None and cached[0] is A:
        return cached[1]
    try:
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError:
        state.inverse_cache.pop(action, None)
        return None
    state.inverse_cache[action] = (A, A_inv, 0)
    return A_inv


def linucb_scores(
    state: LinUCBState, x: np.ndarray, *, l2: float, alpha: float
) -> dict[str, float]:
    """UCB score (exploit + alpha * explore) per action, from the cached inverses.

    Scoring never changes the model (POL-22): when ``A`` is singular, the re-regularized
    matrix is used for this score only; it used to be written back, adding ``l2 * I``
    to the arm on every such call.
    """
    scores: dict[str, float] = {}
    for action in state.action_names:
        A = state.A.get(action)
        b = state.b.get(action)
        if A is None or b is None:
            scores[action] = 0.0
            continue
        A_inv = arm_inverse(state, action)
        if A_inv is None:
            exploit, explore, _regularized = linucb_exploit_explore(A, b, x, l2=float(l2))
        else:
            exploit = float((A_inv @ b) @ x)
            explore = float(np.sqrt(max(0.0, float(x @ (A_inv @ x)))))
        scores[action] = exploit + (float(alpha) * explore)
    return scores


def thompson_scores(state: LinUCBState, x: np.ndarray, *, seed: int | None) -> dict[str, float]:
    """Sample theta ~ N(A^-1 b, A^-1) per action and return theta . x."""
    rng = np.random.default_rng(seed=(seed + state.decisions) if seed is not None else None)
    scores: dict[str, float] = {}
    for action in state.action_names:
        A = state.A.get(action)
        b = state.b.get(action)
        if A is None or b is None:
            scores[action] = 0.0
            continue
        A_inv = arm_inverse(state, action)
        if A_inv is None:
            scores[action] = 0.0
            continue
        try:
            scores[action] = float(rng.multivariate_normal(A_inv @ b, A_inv) @ x)
        except np.linalg.LinAlgError:
            scores[action] = 0.0
    return scores


def apply_recency_decay(state: LinUCBState, *, action: str, decay: float, l2: float) -> None:
    """Shrink *action*'s arm toward the ``l2`` prior by *decay* before its update.

    POL-01: decaying every arm on every update left about 20 observations of evidence in
    total, two per arm against 53 features, so the bandit barely learned. Decay is now
    opt-in (the default is 1.0, decision D6) and only touches the arm that gets new
    evidence, for non-stationary settings.
    """
    if float(decay) >= 1.0 or not state.feature_names:
        return
    dim = len(state.feature_names)
    A = state.A.get(action)
    b = state.b.get(action)
    if A is None or b is None or A.shape != (dim, dim) or b.shape != (dim,):
        return
    prior = np.eye(dim, dtype=float) * float(l2)
    state.A[action] = (float(decay) * A) + ((1.0 - float(decay)) * prior)
    state.b[action] = float(decay) * b


def effective_evidence(state: LinUCBState, *, l2: float) -> dict[str, float]:
    """Evidence each arm holds beyond its prior: ``trace(A_k) - d * l2`` (POL-01 telemetry)."""
    dim = len(state.feature_names)
    return {
        action: float(np.trace(A)) - dim * float(l2)
        for action in state.action_names
        if (A := state.A.get(action)) is not None and A.shape == (dim, dim)
    }


def rank_one_update(state: LinUCBState, *, action: str, reward: float, x: np.ndarray) -> bool:
    """``A += x x^T`` and ``b += reward * x`` for *action*; ``False`` when it can't apply.

    A cached inverse follows by Sherman-Morrison,
    ``A^-1 - (A^-1 x)(A^-1 x)^T / (1 + x^T A^-1 x)``, and is recomputed exactly every
    ``INVERSE_REFRESH_UPDATES`` steps to bound rounding drift (POL-20).
    """
    A = state.A.get(action)
    b = state.b.get(action)
    if A is None or b is None:
        return False
    cached = state.inverse_cache.get(action)
    A += np.outer(x, x)
    b += float(reward) * x
    state.A[action] = A
    state.b[action] = b
    state.updates = int(state.updates) + 1
    if cached is not None and cached[0] is A and cached[2] + 1 < INVERSE_REFRESH_UPDATES:
        A_inv_x = cached[1] @ x
        denominator = 1.0 + float(x @ A_inv_x)
        state.inverse_cache[action] = (
            A,
            cached[1] - np.outer(A_inv_x, A_inv_x) / denominator,
            cached[2] + 1,
        )
    else:
        state.inverse_cache.pop(action, None)
    return True
