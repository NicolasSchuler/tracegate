"""Shared feature scaling helpers for adaptive LinUCB policies.

Provides a pluggable scaling-profile system that normalises raw context
features (counts, streaks, budget caps) into bounded ranges suitable for
LinUCB arm estimation.  The ``v1`` profile applies log1p normalisation
for count-like features and clips all values to a fixed bound.

Optionally, :class:`PercentileScaler` computes caps from historical
feature distributions at a configurable percentile, falling back to
hardcoded v1 caps when fewer than ``min_samples`` observations exist.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field

PERCENTILE_POLICY_FEATURE_SCALING_PROFILE = "percentile_v1"
INTERACTION_POLICY_FEATURE_SCALING_PROFILE = "v1_interactions"
POLICY_FEATURE_SCALING_PROFILES = {
    "none",
    "v1",
    INTERACTION_POLICY_FEATURE_SCALING_PROFILE,
    PERCENTILE_POLICY_FEATURE_SCALING_PROFILE,
}
DEFAULT_POLICY_FEATURE_SCALING_PROFILE = "v1"

# Fixed caps for count-like features in v1 profile.
_V1_COUNT_CAPS: dict[str, float] = {
    "stagnation_streak": 10.0,
    "adaptive_no_progress_streak": 10.0,
    "helper_calls_used": 8.0,
    "adaptive_no_progress_token_sum_k": 200.0,
    "base_propose_budget_sec": 300.0,
}
_V1_DEFAULT_STREAK_CAP = 10.0
_V1_FINAL_CLIP = 4.0

# Percentile cap defaults.
DEFAULT_POLICY_FEATURE_SCALING_PERCENTILE = 95.0
DEFAULT_POLICY_FEATURE_SCALING_MIN_SAMPLES = 30
DEFAULT_POLICY_FEATURE_SCALING_HISTORY_LIMIT = 256


def normalize_feature_scaling_profile(profile: str | None) -> str:
    """Return a validated scaling-profile name, defaulting to ``v1``."""
    candidate = str(profile or "").strip().lower()
    if candidate in POLICY_FEATURE_SCALING_PROFILES:
        return candidate
    return DEFAULT_POLICY_FEATURE_SCALING_PROFILE


def is_percentile_feature_scaling_profile(profile: str | None) -> bool:
    """Return whether *profile* uses runtime percentile-fitted caps."""
    return normalize_feature_scaling_profile(profile) == PERCENTILE_POLICY_FEATURE_SCALING_PROFILE


def _safe_float(value: float) -> float:
    """Convert *value* to float, replacing NaN/Inf with 0.0."""
    out = float(value)
    if math.isnan(out) or math.isinf(out):
        return 0.0
    return out


def _is_count_like_feature(name: str) -> bool:
    """Check whether *name* represents a count-like feature needing log scaling."""
    if name.endswith("_streak"):
        return True
    return name in _V1_COUNT_CAPS


def _count_cap_for(
    name: str,
    overrides: Mapping[str, float] | None = None,
) -> float:
    """Return the normalisation cap for a count-like feature.

    When *overrides* is provided the cap is looked up there first,
    falling back to the hardcoded v1 caps.
    """
    if overrides and name in overrides:
        return overrides[name]
    if name in _V1_COUNT_CAPS:
        return _V1_COUNT_CAPS[name]
    if name.endswith("_streak"):
        return _V1_DEFAULT_STREAK_CAP
    return 1.0


def _clip(value: float, *, clip_abs: float) -> float:
    """Clamp *value* to the range [-clip_abs, +clip_abs]."""
    if value > clip_abs:
        return clip_abs
    if value < -clip_abs:
        return -clip_abs
    return value


# ---------------------------------------------------------------------------
# Percentile-based cap computation
# ---------------------------------------------------------------------------


def compute_percentile_caps(
    historical_data: Mapping[str, Sequence[float]],
    *,
    percentile: float = DEFAULT_POLICY_FEATURE_SCALING_PERCENTILE,
    min_samples: int = DEFAULT_POLICY_FEATURE_SCALING_MIN_SAMPLES,
) -> dict[str, float]:
    """Compute per-feature caps at the given *percentile*.

    Args:
        historical_data: Mapping from feature name to a sequence of
            observed float values.
        percentile: The percentile (0-100) at which to set each cap.
            Defaults to 95.
        min_samples: Minimum number of valid (finite) observations
            required to compute a percentile cap.  Features with fewer
            samples fall back to hardcoded v1 caps.

    Returns:
        A dict mapping feature names to their computed (or fallback)
        cap values.  Caps are always ≥ 1.0.
    """
    if not 0.0 <= percentile <= 100.0:
        raise ValueError(f"percentile must be between 0 and 100, got {percentile}")
    if min_samples < 1:
        raise ValueError(f"min_samples must be >= 1, got {min_samples}")

    caps: dict[str, float] = {}
    for name, raw_values in historical_data.items():
        # Filter to finite values only.
        clean: list[float] = [v for v in (float(x) for x in raw_values) if math.isfinite(v)]

        if len(clean) < min_samples:
            # Not enough data — use hardcoded fallback.
            caps[name] = _count_cap_for(name)
            continue

        sorted_vals = sorted(clean)
        idx = (percentile / 100.0) * (len(sorted_vals) - 1)
        lower = math.floor(idx)
        upper = min(lower + 1, len(sorted_vals) - 1)
        frac = idx - lower
        pval = sorted_vals[lower] + frac * (sorted_vals[upper] - sorted_vals[lower])
        caps[name] = max(1.0, pval)

    return caps


@dataclass
class PercentileScaler:
    """Computes and caches percentile-based caps from historical data.

    Usage::

        scaler = PercentileScaler(percentile=95)
        scaler.fit({"stagnation_streak": [1, 2, 3, ...], ...})
        scaled = scaler.scale(raw_features, profile="v1")

    When :meth:`fit` has not been called, or a feature had insufficient
    data, the scaler falls back to the hardcoded v1 caps — making it a
    safe drop-in enhancement.
    """

    percentile: float = DEFAULT_POLICY_FEATURE_SCALING_PERCENTILE
    min_samples: int = DEFAULT_POLICY_FEATURE_SCALING_MIN_SAMPLES
    _caps: dict[str, float] = field(default_factory=dict, repr=False)

    def fit(
        self,
        historical_data: Mapping[str, Sequence[float]],
    ) -> PercentileScaler:
        """Compute caps from *historical_data* and store them.

        Returns *self* for method chaining.
        """
        self._caps = compute_percentile_caps(
            historical_data,
            percentile=self.percentile,
            min_samples=self.min_samples,
        )
        return self

    @property
    def caps(self) -> dict[str, float]:
        """Return a copy of the currently fitted caps (empty before fit)."""
        return dict(self._caps)

    def scale(
        self,
        feature_values: Mapping[str, float],
        *,
        profile: str | None = DEFAULT_POLICY_FEATURE_SCALING_PROFILE,
    ) -> dict[str, float]:
        """Scale *feature_values* using percentile caps where available.

        Delegates to :func:`scale_feature_mapping` with the fitted cap
        overrides.
        """
        return scale_feature_mapping(
            feature_values,
            profile=profile,
            cap_overrides=self._caps or None,
        )


# ---------------------------------------------------------------------------
# Core scaling entry-point
# ---------------------------------------------------------------------------


def scale_feature_mapping(
    feature_values: Mapping[str, float],
    *,
    profile: str | None = DEFAULT_POLICY_FEATURE_SCALING_PROFILE,
    cap_overrides: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Scale a raw feature mapping according to the chosen *profile*.

    With profile ``v1`` (and ``v1_interactions``), count-like features are
    log1p-normalised and all values are clipped.  Profile ``none`` passes
    values through unchanged.

    Args:
        feature_values: Raw feature name→value mapping.
        profile: Scaling profile name (``"v1"`` or ``"none"``).
        cap_overrides: Optional per-feature cap overrides.  When
            provided, these take priority over the hardcoded v1 caps
            for count-like features.  Typically produced by
            :func:`compute_percentile_caps` or :class:`PercentileScaler`.
    """
    normalized_profile = normalize_feature_scaling_profile(profile)
    out: dict[str, float] = {}
    for key, raw_value in dict(feature_values).items():
        name = str(key)
        value = _safe_float(float(raw_value))

        if normalized_profile == "none":
            out[name] = value
            continue

        if _is_count_like_feature(name):
            cap = max(1.0, _count_cap_for(name, overrides=cap_overrides))
            normalized = math.log1p(max(0.0, value)) / math.log1p(cap)
            out[name] = _clip(normalized, clip_abs=_V1_FINAL_CLIP)
            continue

        # Keep bounded probabilities/features unchanged.
        if 0.0 <= value <= 1.0:
            out[name] = value
            continue

        out[name] = _clip(value, clip_abs=_V1_FINAL_CLIP)
    return out


__all__ = [
    "DEFAULT_POLICY_FEATURE_SCALING_HISTORY_LIMIT",
    "DEFAULT_POLICY_FEATURE_SCALING_MIN_SAMPLES",
    "DEFAULT_POLICY_FEATURE_SCALING_PERCENTILE",
    "DEFAULT_POLICY_FEATURE_SCALING_PROFILE",
    "INTERACTION_POLICY_FEATURE_SCALING_PROFILE",
    "PERCENTILE_POLICY_FEATURE_SCALING_PROFILE",
    "POLICY_FEATURE_SCALING_PROFILES",
    "PercentileScaler",
    "compute_percentile_caps",
    "is_percentile_feature_scaling_profile",
    "normalize_feature_scaling_profile",
    "scale_feature_mapping",
]
