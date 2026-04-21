"""Tests for policy feature scaling normalization and profile validation."""

from __future__ import annotations

import math

import pytest

from evals.policy_feature_scaling import (
    normalize_feature_scaling_profile,
    scale_feature_mapping,
)


def test_normalize_feature_scaling_profile_defaults_to_v1() -> None:
    assert normalize_feature_scaling_profile("v1") == "v1"
    assert normalize_feature_scaling_profile("none") == "none"
    assert normalize_feature_scaling_profile("percentile_v1") == "percentile_v1"
    assert normalize_feature_scaling_profile("invalid") == "v1"
    assert normalize_feature_scaling_profile(None) == "v1"


def test_scale_feature_mapping_v1_keeps_bounded_features() -> None:
    scaled = scale_feature_mapping(
        {
            "attempt_frac": 0.75,
            "helper_budget_remaining_frac": 0.1,
            "bias": 1.0,
        },
        profile="v1",
    )
    assert scaled["attempt_frac"] == pytest.approx(0.75)
    assert scaled["helper_budget_remaining_frac"] == pytest.approx(0.1)
    assert scaled["bias"] == pytest.approx(1.0)


def test_scale_feature_mapping_v1_applies_log_scaling_to_count_like_features() -> None:
    scaled = scale_feature_mapping(
        {
            "stagnation_streak": 10.0,
            "adaptive_no_progress_token_sum_k": 200.0,
            "base_propose_budget_sec": 300.0,
        },
        profile="v1",
    )
    assert scaled["stagnation_streak"] == pytest.approx(1.0)
    assert scaled["adaptive_no_progress_token_sum_k"] == pytest.approx(1.0)
    assert scaled["base_propose_budget_sec"] == pytest.approx(1.0)

    half = scale_feature_mapping({"helper_calls_used": 4.0}, profile="v1")
    expected_half = math.log1p(4.0) / math.log1p(8.0)
    assert half["helper_calls_used"] == pytest.approx(expected_half)


def test_scale_feature_mapping_v1_count_like_features_are_monotonic_near_one() -> None:
    scaled_1 = scale_feature_mapping({"stagnation_streak": 1.0}, profile="v1")["stagnation_streak"]
    scaled_2 = scale_feature_mapping({"stagnation_streak": 2.0}, profile="v1")["stagnation_streak"]
    assert 0.0 <= scaled_1 < scaled_2 < 1.0


def test_scale_feature_mapping_v1_clips_large_unbounded_features() -> None:
    scaled = scale_feature_mapping({"custom_signal": 99.0, "negative": -99.0}, profile="v1")
    assert scaled["custom_signal"] == 4.0
    assert scaled["negative"] == -4.0


def test_scale_feature_mapping_none_profile_is_passthrough() -> None:
    scaled = scale_feature_mapping({"custom_signal": 99.0, "bias": 1.0}, profile="none")
    assert scaled["custom_signal"] == pytest.approx(99.0)
    assert scaled["bias"] == pytest.approx(1.0)


def test_scale_feature_mapping_percentile_v1_without_caps_matches_v1() -> None:
    raw = {"helper_calls_used": 4.0, "attempt_frac": 0.5}
    assert scale_feature_mapping(raw, profile="percentile_v1") == scale_feature_mapping(
        raw,
        profile="v1",
    )
