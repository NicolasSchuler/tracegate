"""Tests for evals._policy_utils - coercion helpers and policy epsilon logic."""

from __future__ import annotations

import math

import pytest

from evals._policy_utils import (
    coerce_float,
    coerce_nonnegative_int,
    policy_effective_epsilon,
    safe_float,
)

# ---------------------------------------------------------------------------
# coerce_float - lines 25-26 (TypeError/ValueError fallback)
# ---------------------------------------------------------------------------


class TestCoerceFloat:
    """Verify coerce_float returns *default* for unconvertible values."""

    @pytest.mark.parametrize(
        ("value", "default", "expected"),
        [
            (None, 1.5, 1.5),
            ("abc", 0.0, 0.0),
            (object(), -1.0, -1.0),
        ],
        ids=["none", "non-numeric-str", "object"],
    )
    def test_unconvertible_returns_default(
        self, value: object, default: float, expected: float
    ) -> None:
        assert coerce_float(value, default=default) == expected

    def test_valid_float_passthrough(self) -> None:
        assert coerce_float("3.14", default=0.0) == pytest.approx(3.14)

    def test_int_coerces_to_float(self) -> None:
        assert coerce_float(7, default=0.0) == 7.0


# ---------------------------------------------------------------------------
# coerce_nonnegative_int - lines 33-34 (TypeError/ValueError fallback)
# ---------------------------------------------------------------------------


class TestCoerceNonnegativeInt:
    """Verify coerce_nonnegative_int returns *default* for unconvertible values."""

    @pytest.mark.parametrize(
        ("value", "default", "expected"),
        [
            (None, 5, 5),
            ("xyz", 0, 0),
            (object(), 3, 3),
        ],
        ids=["none", "non-numeric-str", "object"],
    )
    def test_unconvertible_returns_default(
        self, value: object, default: int, expected: int
    ) -> None:
        assert coerce_nonnegative_int(value, default=default) == expected

    def test_negative_clamped_to_zero(self) -> None:
        assert coerce_nonnegative_int(-10, default=0) == 0

    def test_positive_passes_through(self) -> None:
        assert coerce_nonnegative_int(42, default=0) == 42


# ---------------------------------------------------------------------------
# safe_float - lines 40-46, branches [44,45] and [44,46]
# ---------------------------------------------------------------------------


class TestSafeFloat:
    """Verify safe_float handles NaN, Inf, unconvertible, and normal values."""

    @pytest.mark.parametrize(
        ("value", "default", "expected"),
        [
            (float("nan"), 0.0, 0.0),
            (float("inf"), -1.0, -1.0),
            (float("-inf"), 5.0, 5.0),
        ],
        ids=["nan", "pos-inf", "neg-inf"],
    )
    def test_nan_and_inf_return_default(
        self, value: float, default: float, expected: float
    ) -> None:
        """Branch [44,45]: np.isnan/isinf is True → return default."""
        result = safe_float(value, default=default)
        assert result == expected
        assert not math.isnan(result)
        assert not math.isinf(result)

    def test_valid_float_returned(self) -> None:
        """Branch [44,46]: np.isnan/isinf is False → return out."""
        assert safe_float(2.718, default=0.0) == pytest.approx(2.718)

    def test_valid_string_float_returned(self) -> None:
        """Branch [44,46] via string that converts cleanly."""
        assert safe_float("1.23", default=0.0) == pytest.approx(1.23)

    @pytest.mark.parametrize(
        "value",
        [None, "not_a_number", object()],
        ids=["none", "non-numeric-str", "object"],
    )
    def test_unconvertible_returns_default(self, value: object) -> None:
        """Lines 42-43: TypeError/ValueError → return default."""
        assert safe_float(value, default=99.0) == 99.0


# ---------------------------------------------------------------------------
# policy_effective_epsilon - lines 99-100, branch [99,100]
# ---------------------------------------------------------------------------


class TestPolicyEffectiveEpsilon:
    """Verify policy_effective_epsilon for mode='off' and other modes."""

    def test_mode_off_returns_zero_epsilon(self) -> None:
        """Lines 99-100: mode='off' short-circuits to (0.0, False)."""
        eps, warmup = policy_effective_epsilon(
            mode="off",
            decision_index=0,
            warmup_decisions=10,
            epsilon=0.5,
        )
        assert eps == 0.0
        assert warmup is False

    def test_warmup_active_returns_full_epsilon(self) -> None:
        eps, warmup = policy_effective_epsilon(
            mode="explore",
            decision_index=1,
            warmup_decisions=5,
            epsilon=0.1,
        )
        assert eps == 1.0
        assert warmup is True

    def test_post_warmup_returns_clamped_epsilon(self) -> None:
        eps, warmup = policy_effective_epsilon(
            mode="explore",
            decision_index=100,
            warmup_decisions=5,
            epsilon=0.3,
        )
        assert eps == pytest.approx(0.3)
        assert warmup is False
