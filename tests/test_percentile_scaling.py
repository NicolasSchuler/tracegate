"""Tests for percentile-based cap computation and PercentileScaler."""

from __future__ import annotations

import math

import pytest

from evals.policy_feature_scaling import (
    _V1_COUNT_CAPS,
    _V1_DEFAULT_STREAK_CAP,
    PercentileScaler,
    compute_percentile_caps,
    scale_feature_mapping,
)

# ---------------------------------------------------------------------------
# compute_percentile_caps - normal data
# ---------------------------------------------------------------------------


class TestComputePercentileCapsNormal:
    """Percentile cap computation with well-formed data."""

    def test_95th_percentile_on_uniform_range(self) -> None:
        values = list(range(1, 101))  # 1..100 → 100 samples
        caps = compute_percentile_caps(
            {"stagnation_streak": values},
            percentile=95,
            min_samples=30,
        )
        # 95th pctile of 1..100 ≈ 95.05
        assert caps["stagnation_streak"] == pytest.approx(95.05, abs=0.1)

    def test_50th_percentile_is_median(self) -> None:
        values = list(range(1, 101))
        caps = compute_percentile_caps(
            {"helper_calls_used": values},
            percentile=50,
            min_samples=30,
        )
        expected_median = 50.5
        assert caps["helper_calls_used"] == pytest.approx(expected_median, abs=0.1)

    def test_100th_percentile_is_max(self) -> None:
        values = [3.0, 7.0, 1.0, 9.0, 5.0] * 10  # 50 samples
        caps = compute_percentile_caps(
            {"stagnation_streak": values},
            percentile=100,
            min_samples=5,
        )
        assert caps["stagnation_streak"] == pytest.approx(9.0)

    def test_0th_percentile_is_min(self) -> None:
        values = [3.0, 7.0, 1.0, 9.0, 5.0] * 10
        caps = compute_percentile_caps(
            {"stagnation_streak": values},
            percentile=0,
            min_samples=5,
        )
        assert caps["stagnation_streak"] == pytest.approx(1.0)

    def test_multiple_features(self) -> None:
        data = {
            "stagnation_streak": list(range(50)),
            "helper_calls_used": [float(i) for i in range(50)],
        }
        caps = compute_percentile_caps(data, percentile=90, min_samples=10)
        assert "stagnation_streak" in caps
        assert "helper_calls_used" in caps
        assert caps["stagnation_streak"] > 1.0
        assert caps["helper_calls_used"] > 1.0

    def test_cap_is_at_least_one(self) -> None:
        """Even if all values are tiny, the cap is clamped to >= 1.0."""
        values = [0.001] * 50
        caps = compute_percentile_caps(
            {"stagnation_streak": values},
            percentile=95,
            min_samples=5,
        )
        assert caps["stagnation_streak"] >= 1.0

    def test_nan_and_inf_are_filtered(self) -> None:
        """NaN/Inf values are stripped before computing the percentile."""
        clean = list(range(1, 51))
        dirty = [*clean, float("nan"), float("inf"), float("-inf")]
        caps = compute_percentile_caps(
            {"stagnation_streak": dirty},
            percentile=95,
            min_samples=30,
        )
        caps_clean = compute_percentile_caps(
            {"stagnation_streak": clean},
            percentile=95,
            min_samples=30,
        )
        assert caps["stagnation_streak"] == pytest.approx(caps_clean["stagnation_streak"])


# ---------------------------------------------------------------------------
# compute_percentile_caps - edge cases
# ---------------------------------------------------------------------------


class TestComputePercentileCapsEdge:
    """Edge cases: empty, single value, all-same, insufficient data."""

    def test_empty_data_falls_back(self) -> None:
        caps = compute_percentile_caps(
            {"stagnation_streak": []},
            percentile=95,
            min_samples=30,
        )
        assert caps["stagnation_streak"] == _V1_COUNT_CAPS["stagnation_streak"]

    def test_single_value_falls_back(self) -> None:
        caps = compute_percentile_caps(
            {"helper_calls_used": [5.0]},
            percentile=95,
            min_samples=30,
        )
        assert caps["helper_calls_used"] == _V1_COUNT_CAPS["helper_calls_used"]

    def test_all_same_values(self) -> None:
        """When every observation is identical the cap is that value (or 1)."""
        values = [7.0] * 50
        caps = compute_percentile_caps(
            {"stagnation_streak": values},
            percentile=95,
            min_samples=5,
        )
        assert caps["stagnation_streak"] == pytest.approx(7.0)

    def test_all_same_zero_caps_to_one(self) -> None:
        values = [0.0] * 50
        caps = compute_percentile_caps(
            {"stagnation_streak": values},
            percentile=95,
            min_samples=5,
        )
        assert caps["stagnation_streak"] == 1.0

    def test_insufficient_data_uses_v1_cap(self) -> None:
        caps = compute_percentile_caps(
            {"helper_calls_used": [1.0] * 29},
            percentile=95,
            min_samples=30,
        )
        assert caps["helper_calls_used"] == _V1_COUNT_CAPS["helper_calls_used"]

    def test_exactly_min_samples_computes(self) -> None:
        values = list(range(30))
        caps = compute_percentile_caps(
            {"stagnation_streak": values},
            percentile=95,
            min_samples=30,
        )
        # Should NOT fall back — computed value expected > 1.
        assert caps["stagnation_streak"] > 1.0
        assert caps["stagnation_streak"] != _V1_COUNT_CAPS["stagnation_streak"]

    def test_unknown_feature_fallback(self) -> None:
        """Unknown streak feature falls back to default streak cap."""
        caps = compute_percentile_caps(
            {"custom_streak": [1.0] * 5},
            percentile=95,
            min_samples=30,
        )
        assert caps["custom_streak"] == _V1_DEFAULT_STREAK_CAP

    def test_unknown_non_streak_feature_fallback(self) -> None:
        """Non-streak, non-v1 feature falls back to 1.0."""
        caps = compute_percentile_caps(
            {"some_other_feature": [1.0] * 5},
            percentile=95,
            min_samples=30,
        )
        assert caps["some_other_feature"] == 1.0

    def test_empty_mapping_returns_empty(self) -> None:
        caps = compute_percentile_caps({}, percentile=95, min_samples=30)
        assert caps == {}


# ---------------------------------------------------------------------------
# compute_percentile_caps - validation
# ---------------------------------------------------------------------------


class TestComputePercentileCapsValidation:
    """Input validation for compute_percentile_caps."""

    def test_percentile_below_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="percentile"):
            compute_percentile_caps({"x": [1.0] * 50}, percentile=-1)

    def test_percentile_above_100_raises(self) -> None:
        with pytest.raises(ValueError, match="percentile"):
            compute_percentile_caps({"x": [1.0] * 50}, percentile=101)

    def test_min_samples_zero_raises(self) -> None:
        with pytest.raises(ValueError, match="min_samples"):
            compute_percentile_caps({"x": [1.0] * 50}, min_samples=0)


# ---------------------------------------------------------------------------
# PercentileScaler
# ---------------------------------------------------------------------------


class TestPercentileScaler:
    """PercentileScaler class behaviour."""

    def test_unfitted_scaler_uses_hardcoded_caps(self) -> None:
        """Before fit(), scaling matches default scale_feature_mapping."""
        scaler = PercentileScaler()
        raw = {"stagnation_streak": 5.0, "attempt_frac": 0.5}
        expected = scale_feature_mapping(raw, profile="v1")
        result = scaler.scale(raw, profile="v1")
        assert result == expected

    def test_fit_returns_self(self) -> None:
        scaler = PercentileScaler()
        ret = scaler.fit({"stagnation_streak": list(range(50))})
        assert ret is scaler

    def test_fitted_caps_are_accessible(self) -> None:
        scaler = PercentileScaler(percentile=90, min_samples=5)
        scaler.fit({"stagnation_streak": list(range(50))})
        caps = scaler.caps
        assert "stagnation_streak" in caps
        assert caps["stagnation_streak"] > 1.0

    def test_caps_property_returns_copy(self) -> None:
        scaler = PercentileScaler(min_samples=5)
        scaler.fit({"stagnation_streak": list(range(50))})
        caps1 = scaler.caps
        caps2 = scaler.caps
        assert caps1 is not caps2
        assert caps1 == caps2

    def test_fitted_scaler_changes_scaling(self) -> None:
        """After fit with a custom distribution, caps differ from v1."""
        # Build data where the 95th percentile is ~5 (not the v1 cap of 10).
        values = list(range(50))  # 0..49 → p95 ≈ 46.55
        scaler = PercentileScaler(percentile=95, min_samples=5)
        scaler.fit({"stagnation_streak": values})

        raw = {"stagnation_streak": 10.0}
        default_scaled = scale_feature_mapping(raw, profile="v1")
        percentile_scaled = scaler.scale(raw, profile="v1")

        # With a much larger cap, the same input produces a smaller value.
        assert percentile_scaled["stagnation_streak"] != default_scaled["stagnation_streak"]

    def test_none_profile_passthrough(self) -> None:
        scaler = PercentileScaler(min_samples=5)
        scaler.fit({"stagnation_streak": list(range(50))})
        raw = {"stagnation_streak": 42.0}
        result = scaler.scale(raw, profile="none")
        assert result["stagnation_streak"] == pytest.approx(42.0)


# ---------------------------------------------------------------------------
# Integration with scale_feature_mapping via cap_overrides
# ---------------------------------------------------------------------------


class TestScaleFeatureMappingCapOverrides:
    """Verify cap_overrides param integrates correctly."""

    def test_override_changes_count_feature_scaling(self) -> None:
        raw = {"helper_calls_used": 4.0}
        default = scale_feature_mapping(raw, profile="v1")
        override = scale_feature_mapping(
            raw, profile="v1", cap_overrides={"helper_calls_used": 20.0}
        )
        # Higher cap → smaller normalised value for the same input.
        assert override["helper_calls_used"] < default["helper_calls_used"]

    def test_override_only_affects_specified_feature(self) -> None:
        raw = {
            "helper_calls_used": 4.0,
            "stagnation_streak": 5.0,
        }
        result = scale_feature_mapping(
            raw,
            profile="v1",
            cap_overrides={"helper_calls_used": 50.0},
        )
        expected_streak = scale_feature_mapping({"stagnation_streak": 5.0}, profile="v1")
        assert result["stagnation_streak"] == pytest.approx(expected_streak["stagnation_streak"])

    def test_override_does_not_affect_none_profile(self) -> None:
        raw = {"helper_calls_used": 4.0}
        result = scale_feature_mapping(
            raw,
            profile="none",
            cap_overrides={"helper_calls_used": 50.0},
        )
        assert result["helper_calls_used"] == pytest.approx(4.0)

    def test_override_respects_min_cap_of_one(self) -> None:
        """Even if override provides a very small cap, max(1.0, ...) holds."""
        raw = {"helper_calls_used": 4.0}
        result = scale_feature_mapping(raw, profile="v1", cap_overrides={"helper_calls_used": 0.5})
        expected = math.log1p(4.0) / math.log1p(1.0)
        clipped = min(expected, 4.0)
        assert result["helper_calls_used"] == pytest.approx(clipped)

    def test_override_with_none_is_noop(self) -> None:
        raw = {"stagnation_streak": 5.0}
        with_none = scale_feature_mapping(raw, profile="v1", cap_overrides=None)
        without = scale_feature_mapping(raw, profile="v1")
        assert with_none == without


# ---------------------------------------------------------------------------
# Existing behaviour preserved (regression)
# ---------------------------------------------------------------------------


class TestExistingBehaviourPreserved:
    """Ensure original scale_feature_mapping behaviour is unbroken."""

    def test_v1_keeps_bounded_features(self) -> None:
        scaled = scale_feature_mapping({"attempt_frac": 0.75, "bias": 1.0}, profile="v1")
        assert scaled["attempt_frac"] == pytest.approx(0.75)
        assert scaled["bias"] == pytest.approx(1.0)

    def test_v1_log_scaling_at_cap(self) -> None:
        scaled = scale_feature_mapping({"stagnation_streak": 10.0}, profile="v1")
        assert scaled["stagnation_streak"] == pytest.approx(1.0)

    def test_v1_clips_large_unbounded(self) -> None:
        scaled = scale_feature_mapping({"custom_signal": 99.0}, profile="v1")
        assert scaled["custom_signal"] == 4.0

    def test_none_passthrough(self) -> None:
        scaled = scale_feature_mapping({"custom_signal": 99.0}, profile="none")
        assert scaled["custom_signal"] == pytest.approx(99.0)
