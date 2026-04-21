"""Shared utility helpers for adaptive policy modules.

Small, pure functions used by both :mod:`adaptive_online_policy` and
:mod:`adaptive_budget_online_policy`.  Kept in a single place to avoid
duplicating logic across policy controllers.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import numpy as np


def utc_now() -> str:
    """Return the current UTC time as a compact ISO-8601 string."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def coerce_float(value: Any, *, default: float) -> float:
    """Convert *value* to float, returning *default* on failure."""
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def coerce_nonnegative_int(value: Any, *, default: int) -> int:
    """Convert *value* to a non-negative int, returning *default* on failure."""
    try:
        out = int(value)
    except (TypeError, ValueError):
        return int(default)
    return max(0, out)


def safe_float(value: Any, *, default: float = 0.0) -> float:
    """Convert *value* to float, replacing NaN/Inf with *default*."""
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if np.isnan(out) or np.isinf(out):
        return float(default)
    return out


def safe_float_tracked(value: Any, *, default: float = 0.0) -> tuple[float, bool]:
    """Like :func:`safe_float` but also report whether a replacement occurred.

    Returns a ``(result, replaced)`` tuple where *replaced* is ``True``
    when the original value was NaN, Inf, or otherwise unconvertible.
    """
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default), True
    if np.isnan(out) or np.isinf(out):
        return float(default), True
    return out, False


def linucb_exploit_explore(
    A: np.ndarray,
    b: np.ndarray,
    x: np.ndarray,
    *,
    l2: float,
) -> tuple[float, float, np.ndarray | None]:
    """Compute LinUCB exploit/explore terms with a single linear solve.

    Returns ``(exploit, explore, updated_A)``. ``updated_A`` is ``None`` when the
    original matrix succeeded, or the re-regularized matrix when the fallback path
    had to retry after a linear-algebra error.
    """
    try:
        solution = np.linalg.solve(A, np.column_stack((b, x)))
        updated_A: np.ndarray | None = None
    except np.linalg.LinAlgError:
        regularized_A = A + (np.eye(A.shape[0], dtype=float) * float(l2))
        solution = np.linalg.solve(regularized_A, np.column_stack((b, x)))
        updated_A = regularized_A
    theta = solution[:, 0]
    variance_vector = solution[:, 1]
    exploit = float(theta @ x)
    explore = float(np.sqrt(max(0.0, float(x @ variance_vector))))
    return exploit, explore, updated_A


def policy_effective_epsilon(
    *,
    mode: str,
    decision_index: int,
    warmup_decisions: int,
    epsilon: float,
) -> tuple[float, bool]:
    """Return the effective epsilon and whether warmup is active for a decision."""
    if str(mode) == "off":
        return 0.0, False
    warmup_active = int(decision_index) <= max(0, int(warmup_decisions))
    if warmup_active:
        return 1.0, True
    return min(1.0, max(0.0, float(epsilon))), False
