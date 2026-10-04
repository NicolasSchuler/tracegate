"""Phase 3.1: the shared LinUCB core matches the implementations it replaced.

The reference functions below are frozen copies of the pre-extraction code (identical
in ``adaptive_online_policy``, ``adaptive_budget_online_policy`` and
``bootstrap_policy_state``). ``tests/golden/policy_online_parity.golden`` was captured from a
seeded online-controller run, plus a bootstrap replay of its events, before the
extraction.
"""

from __future__ import annotations

import tempfile
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest

from evals import _json as json
from evals import linucb_core as core

GOLDEN = Path(__file__).parent / "golden" / "policy_online_parity.golden"


# --- frozen pre-extraction reference -------------------------------------------------


class _RefState:
    def __init__(self, actions: tuple[str, ...]) -> None:
        self.action_names = actions
        self.feature_names: list[str] = []
        self.A: dict[str, np.ndarray] = {}
        self.b: dict[str, np.ndarray] = {}
        self.updates = 0


def _ref_ensure(state: _RefState, cleaned: Mapping[str, float], l2: float) -> None:
    if not state.feature_names:
        state.feature_names = sorted(str(k) for k in cleaned)
        dim = len(state.feature_names)
        if dim <= 0:
            state.feature_names = ["bias"]
            dim = 1
        state.A = {a: np.eye(dim) * float(l2) for a in state.action_names}
        state.b = {a: np.zeros((dim,)) for a in state.action_names}
        return
    expanded = sorted(set(state.feature_names) | {str(k) for k in cleaned})
    if expanded != state.feature_names:
        prev = {n: i for i, n in enumerate(state.feature_names)}
        prev_dim = len(state.feature_names)
        new_dim = len(expanded)
        state.feature_names = list(expanded)
        for action in state.action_names:
            pa, pb = state.A.get(action), state.b.get(action)
            na, nb = np.eye(new_dim) * float(l2), np.zeros((new_dim,))
            if pa is not None and pb is not None and pa.shape == (prev_dim, prev_dim):
                for r, fr in enumerate(expanded):
                    orow = prev.get(fr)
                    if orow is None:
                        continue
                    nb[r] = pb[orow]
                    for c, fc in enumerate(expanded):
                        ocol = prev.get(fc)
                        if ocol is None:
                            continue
                        na[r, c] = pa[orow, ocol]
            state.A[action], state.b[action] = na, nb


def _ref_decay(state: _RefState, decay: float, l2: float) -> None:
    if decay >= 1.0 or not state.feature_names:
        return
    dim = len(state.feature_names)
    prior = np.eye(dim) * float(l2)
    for action in state.action_names:
        state.A[action] = decay * state.A[action] + (1.0 - decay) * prior
        state.b[action] = decay * state.b[action]


def _ref_update(state: _RefState, action: str, reward: float, x: np.ndarray) -> None:
    state.A[action] = state.A[action] + np.outer(x, x)
    state.b[action] = state.b[action] + reward * x
    state.updates += 1


def _ref_scores(state: _RefState, x: np.ndarray, alpha: float) -> dict[str, float]:
    out = {}
    for action in state.action_names:
        sol = np.linalg.solve(state.A[action], np.column_stack((state.b[action], x)))
        out[action] = float(sol[:, 0] @ x) + alpha * float(np.sqrt(max(0.0, x @ sol[:, 1])))
    return out


# --- parity ---------------------------------------------------------------------------


@pytest.mark.parametrize("decay", [1.0, 0.95])
def test_core_matches_frozen_reference_over_random_sequences(decay: float) -> None:
    actions = ("a", "b", "c")
    rng = np.random.default_rng(11)
    ref = _RefState(actions)
    state = core.LinUCBState(action_names=actions)
    names = ["bias", "f1"]
    for step in range(60):
        if step in (20, 40):
            names.append(f"new{step}")
        values = {name: float(rng.normal()) for name in names}
        _ref_ensure(ref, values, l2=1.0)
        core.ensure_feature_state(state, values, l2=1.0)
        assert state.feature_names == ref.feature_names
        x = np.array([values.get(n, 0.0) for n in ref.feature_names])
        assert core.feature_vector(state, values, to_float=lambda v, _n: float(v)).tolist() == (
            x.tolist()
        )
        np.testing.assert_allclose(
            list(core.linucb_scores(state, x, l2=1.0, alpha=0.8).values()),
            list(_ref_scores(ref, x, 0.8).values()),
        )
        action = actions[step % 3]
        reward = float(rng.normal())
        _ref_decay(ref, decay, 1.0)
        core.apply_recency_decay(state, decay=decay, l2=1.0)
        _ref_update(ref, action, reward, x)
        assert core.rank_one_update(state, action=action, reward=reward, x=x)
    for action in actions:
        np.testing.assert_allclose(state.A[action], ref.A[action])
        np.testing.assert_allclose(state.b[action], ref.b[action])
    assert state.updates == ref.updates


def test_online_controller_and_replay_match_the_pre_extraction_golden() -> None:
    from tests._policy_parity_scenario import run_online_scenario

    golden = json.loads(GOLDEN.read_text(encoding="utf-8"))
    with tempfile.TemporaryDirectory() as tmp:
        observed = run_online_scenario(Path(tmp))
    assert [step["selected"] for step in observed["trace"]] == [
        step["selected"] for step in golden["trace"]
    ]
    for got, want in zip(observed["trace"], golden["trace"], strict=True):
        assert got["scores"] == pytest.approx(want["scores"], abs=1e-8)
    assert observed["feature_names"] == golden["feature_names"]
    for key in ("A", "b"):
        for action, values in golden[key].items():
            np.testing.assert_allclose(observed[key][action], values, atol=1e-8)
        for action, values in golden["replay"][key].items():
            np.testing.assert_allclose(observed["replay"][key][action], values, atol=1e-8)
