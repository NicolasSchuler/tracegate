"""Shared LinUCB core for the adaptive policy controllers (2026-10-04 review, Phase 3.1).

The helper controller (``adaptive_online_policy``), the budget gate and family controllers
(``adaptive_budget_online_policy``) and the bootstrap replay (``bootstrap_policy_state``)
each carried their own copy of the same state handling and linear algebra: feature-set
expansion, the feature vector, LinUCB and Thompson scores, the rank-1 update and recency
decay. They now share these functions, so each later fix lands once (X-04).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field

import numpy as np

from ._policy_utils import linucb_exploit_explore


@dataclass
class LinUCBState:
    """Per-action LinUCB model: ``A`` matrices, ``b`` vectors and bookkeeping."""

    action_names: tuple[str, ...] = ("default",)
    feature_names: list[str] = field(default_factory=list)
    decisions: int = 0
    updates: int = 0
    action_counts: dict[str, int] = field(default_factory=dict)
    A: dict[str, np.ndarray] = field(default_factory=dict)
    b: dict[str, np.ndarray] = field(default_factory=dict)
    feature_scaling_history: dict[str, list[float]] = field(default_factory=dict)
    feature_scaling_caps: dict[str, float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized_actions = tuple(str(action) for action in self.action_names) or ("default",)
        self.action_names = normalized_actions
        counts = dict.fromkeys(normalized_actions, 0)
        for key, value in dict(self.action_counts).items():
            if key in counts:
                try:
                    counts[key] = int(value)
                except (TypeError, ValueError):
                    counts[key] = 0
        self.action_counts = counts


def ensure_feature_state(state: LinUCBState, feature_names: Iterable[str], *, l2: float) -> None:
    """Initialise the model on first use, or expand it when new features appear.

    Existing rows and columns are carried over; new features start from the ``l2``
    prior. Arms whose matrices have the wrong shape are reset to the prior.
    """
    seen = sorted(str(name) for name in feature_names)
    if not state.feature_names:
        state.feature_names = seen or ["bias"]
        dim = len(state.feature_names)
        state.A = {action: np.eye(dim, dtype=float) * float(l2) for action in state.action_names}
        state.b = {action: np.zeros((dim,), dtype=float) for action in state.action_names}
        return
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
            if (
                isinstance(previous_A, np.ndarray)
                and isinstance(previous_b, np.ndarray)
                and previous_A.shape == (previous_dim, previous_dim)
                and previous_b.shape == (previous_dim,)
            ):
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
        if A is None or A.shape != (dim, dim):
            state.A[action] = np.eye(dim, dtype=float) * float(l2)
        if b is None or b.shape != (dim,):
            state.b[action] = np.zeros((dim,), dtype=float)


def feature_vector(
    state: LinUCBState,
    values: Mapping[str, float],
    *,
    to_float: Callable[[object, str], float],
) -> np.ndarray:
    """Feature values in ``state.feature_names`` order (missing features are 0)."""
    names = state.feature_names or ["bias"]
    return np.array([to_float(values.get(name, 0.0), name) for name in names], dtype=float)


def linucb_scores(
    state: LinUCBState, x: np.ndarray, *, l2: float, alpha: float
) -> dict[str, float]:
    """UCB score (exploit + alpha * explore) per action.

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
        exploit, explore, _regularized = linucb_exploit_explore(A, b, x, l2=float(l2))
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
        try:
            A_inv = np.linalg.solve(A, np.eye(A.shape[0]))
            theta_hat = A_inv @ b
            scores[action] = float(rng.multivariate_normal(theta_hat, A_inv) @ x)
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
    """``A += x x^T`` and ``b += reward * x`` for *action*; ``False`` when it can't apply."""
    A = state.A.get(action)
    b = state.b.get(action)
    if A is None or b is None:
        return False
    A += np.outer(x, x)
    b += float(reward) * x
    state.A[action] = A
    state.b[action] = b
    state.updates = int(state.updates) + 1
    return True
