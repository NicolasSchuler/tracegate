"""Shared feature scaling helpers for adaptive LinUCB policies.

Provides a pluggable scaling-profile system that normalises raw context
features (counts, streaks, budget caps) into bounded ranges suitable for
LinUCB arm estimation.  The ``v1`` profile applies log1p normalisation
for count-like features and clips all values to a fixed bound.

Optionally, :class:`PercentileScaler` computes caps from historical
feature distributions at a configurable percentile, falling back to
hardcoded v1 caps when fewer than ``min_samples`` observations exist.
The controllers use :func:`percentile_scale`, which fits each count-like
feature's cap once and then freezes it (POL-09), so a raw value keeps its
scaled value for the rest of the run.
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


def is_count_like_feature(name: str) -> bool:
    """Whether *name* is log-scaled, the only kind of feature a cap applies to."""
    return _is_count_like_feature(str(name))


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
        if pval <= 0.0:
            # No positive observations: a cap of 1.0 would amplify every later value
            # (POL-09), so keep the hardcoded cap.
            caps[name] = _count_cap_for(name)
            continue
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


@dataclass(frozen=True)
class PercentileScaling:
    """Result of :func:`percentile_scale` for one decision."""

    values: dict[str, float]
    caps: dict[str, float]
    """Cap used for each count-like feature in this decision: frozen, or the v1 fallback.
    Features that aren't count-like have no cap (one was logged but never applied)."""
    history_sizes: dict[str, int]
    """Observations collected per count-like feature (stops growing once frozen)."""
    fallback_to_v1: bool
    """Some count-like feature in this decision has no frozen cap yet."""


def prune_percentile_state(
    history: Mapping[str, Sequence[float]],
    caps: Mapping[str, float],
    *,
    min_samples: int = DEFAULT_POLICY_FEATURE_SCALING_MIN_SAMPLES,
) -> tuple[dict[str, list[float]], dict[str, float]]:
    """Keep count-like history, and the caps that count as frozen.

    States saved before POL-09 stored a refitted cap for every feature on every decision,
    including fallback caps (fewer than *min_samples* observations) and caps that were
    never applied; those are dropped. A cap with no history at all is a frozen override
    (a bootstrapped state) and is kept.
    """
    kept_history = {
        str(name): [float(value) for value in values]
        for name, values in history.items()
        if is_count_like_feature(str(name))
    }
    kept_caps: dict[str, float] = {}
    for key, cap in caps.items():
        name = str(key)
        observed = len(kept_history.get(name, ()))
        if is_count_like_feature(name) and (observed == 0 or observed >= min_samples):
            kept_caps[name] = float(cap)
    return kept_history, kept_caps


def percentile_scale(
    feature_values: Mapping[str, float],
    *,
    history: dict[str, list[float]],
    caps: dict[str, float],
    min_samples: int = DEFAULT_POLICY_FEATURE_SCALING_MIN_SAMPLES,
    percentile: float = DEFAULT_POLICY_FEATURE_SCALING_PERCENTILE,
) -> PercentileScaling:
    """Scale with frozen percentile caps, then record this decision's observations.

    *history* and *caps* are the model's state and are updated in place: each count-like
    feature collects observations until it has *min_samples*, then its cap is fitted once
    and frozen (POL-09). Until then the feature uses its v1 cap. The current observation
    is recorded after scaling, so a decision never scales with a cap fitted on itself.
    """
    applied = {
        str(name): float(caps[str(name)]) if str(name) in caps else _count_cap_for(str(name))
        for name in feature_values
        if is_count_like_feature(str(name))
    }
    values = scale_feature_mapping(feature_values, profile="v1", cap_overrides=applied)
    fallback_to_v1 = False
    history_sizes: dict[str, int] = {}
    for key, raw_value in feature_values.items():
        name = str(key)
        if not is_count_like_feature(name):
            continue
        if name in caps:
            history_sizes[name] = len(history.get(name, ()))
            continue
        fallback_to_v1 = True
        samples = history.setdefault(name, [])
        history_sizes[name] = len(samples)
        value = float(raw_value)
        if math.isfinite(value):
            samples.append(value)
        if len(samples) >= min_samples:
            caps[name] = compute_percentile_caps(
                {name: samples}, percentile=percentile, min_samples=min_samples
            )[name]
    return PercentileScaling(
        values=values,
        caps=applied,
        history_sizes=history_sizes,
        fallback_to_v1=fallback_to_v1,
    )


# ---------------------------------------------------------------------------
# Interaction features (``v1_interactions``)
# ---------------------------------------------------------------------------

INTERACTION_FEATURE_FACTORS: dict[str, tuple[str, str]] = {
    "ix_attempt_frac_x_stagnation": ("attempt_frac", "stagnation_streak"),
    "ix_pregate_score_x_snapshot_available": ("pregate_score", "snapshot_available"),
    "ix_helper_calls_used_x_low_yield_ratio": ("helper_calls_used", "helper_low_yield_ratio"),
}
"""Interaction name -> factors. The names don't match the count-like heuristic (POL-19)."""


def interaction_features(feature_values: Mapping[str, float]) -> dict[str, float]:
    """Products of v1-scaled factors, so every interaction is on the same bounded scale.

    The interactions used to multiply raw values (a count times a fraction), and
    ``attempt_frac_x_stagnation_streak`` was then log-scaled again because of its suffix
    (POL-19).
    """
    factors = {name for pair in INTERACTION_FEATURE_FACTORS.values() for name in pair}
    scaled = scale_feature_mapping(
        {name: float(feature_values.get(name, 0.0)) for name in factors}, profile="v1"
    )
    return {
        name: float(scaled[left] * scaled[right])
        for name, (left, right) in INTERACTION_FEATURE_FACTORS.items()
    }


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
    "INTERACTION_FEATURE_FACTORS",
    "INTERACTION_POLICY_FEATURE_SCALING_PROFILE",
    "PERCENTILE_POLICY_FEATURE_SCALING_PROFILE",
    "POLICY_FEATURE_SCALING_PROFILES",
    "PercentileScaler",
    "PercentileScaling",
    "compute_percentile_caps",
    "interaction_features",
    "is_count_like_feature",
    "is_percentile_feature_scaling_profile",
    "normalize_feature_scaling_profile",
    "percentile_scale",
    "prune_percentile_state",
    "scale_feature_mapping",
]
