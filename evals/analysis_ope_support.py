"""Shared pieces of the off-policy analysis: propensities, action counts, support and agreement checks."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from evals.adaptive_online_policy import ONLINE_POLICY_ACTIONS
from evals.analysis_numeric import _as_float_or_none
from evals.stats import DEFAULT_BOOTSTRAP_SEED

# Conservative default remains unchanged; now exposed as an analyzer CLI flag for reproducibility.
ADAPTIVE_OPE_MIN_ACTION_SUPPORT = 2


ADAPTIVE_OPE_RAW_PROPENSITY_FLOOR = 1e-9


ADAPTIVE_OPE_PROPENSITY_FLOOR_DEFAULT = 0.005


ADAPTIVE_OPE_ESTIMATOR_AGREEMENT_ABS_TOL = 0.05


ADAPTIVE_OPE_ESTIMATOR_AGREEMENT_REL_TOL = 0.2


# At or above this propensity a sample's behavior policy is deterministic (shadow mode, or a
# policy role that always applies its own choice), so only the action it took has support there.
ADAPTIVE_OPE_DETERMINISTIC_PROPENSITY = 1.0 - 1e-9


# Mean target probability mass allowed on actions without behavior support before a target's
# estimates stop being identifiable.
ADAPTIVE_OPE_UNSUPPORTED_MASS_TOLERANCE = 1e-6


SUPPORT_WARNING_UNSUPPORTED_MASS = "unsupported_target_mass"


# Ties in the greedy target and the baseline fallback resolve in controller action order.
_ONLINE_ACTION_RANK = {action: index for index, action in enumerate(ONLINE_POLICY_ACTIONS)}


@dataclass(frozen=True)
class OpeSummarySettings:
    """Analysis-only estimator knobs of one OPE summary.

    Grouped so the summary entry points stay within the signature budget while carrying
    the bootstrap seed the CLI asked for (STA-05, STA-07).
    """

    min_action_support: int = ADAPTIVE_OPE_MIN_ACTION_SUPPORT
    propensity_floor: float = ADAPTIVE_OPE_PROPENSITY_FLOOR_DEFAULT
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED


DEFAULT_OPE_SUMMARY_SETTINGS = OpeSummarySettings()


@dataclass(frozen=True)
class TargetOpeSettings:
    """Estimator settings shared by the deterministic and per-sample OPE targets."""

    action_space: set[str]
    min_action_support: int
    ridge_alpha: float
    propensity_floor: float
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED


def _sample_case_keys(samples: list[dict[str, Any]]) -> list[str]:
    """One clustering key per sample: the case it came from (STA-02).

    Every run of a case belongs to the same cluster — repeating a run does not add an
    independent observation, so neither the bootstrap nor cross-fitting may split it.
    Samples with no recorded case fall back to a key of their own.
    """
    return [
        str(sample.get("case_key") or "") or f"\x00sample{idx}"
        for idx, sample in enumerate(samples)
    ]


def _metadata_with_fallback(
    attempt_meta_dict: dict[str, Any],
    patch_meta_dict: dict[str, Any],
    *,
    key: str,
) -> Any:
    """Look up *key* in attempt_metadata, falling back to patch_meta."""
    if key in attempt_meta_dict:
        return attempt_meta_dict.get(key)
    return patch_meta_dict.get(key)


def _sanitize_propensity(
    value: Any,
    *,
    floor: float = ADAPTIVE_OPE_RAW_PROPENSITY_FLOOR,
) -> float:
    """Clamp propensity to ``[floor, 1.0]``, defaulting to 1.0."""
    parsed = _as_float_or_none(value)
    if parsed is None:
        return 1.0
    return min(1.0, max(float(floor), parsed))


def _coerce_feature_values(raw: Any) -> dict[str, float]:
    """Coerce a raw dict to ``{str: float}``, dropping unparseable entries."""
    if not isinstance(raw, dict):
        return {}
    out: dict[str, float] = {}
    for key, value in raw.items():
        parsed = _as_float_or_none(value)
        if parsed is None:
            continue
        out[str(key)] = float(parsed)
    return out


def _feature_scaling_profile_bucket(raw: Any) -> str:
    """Normalize a feature-scaling profile to a bucket label."""
    profile = str(raw or "").strip()
    return profile if profile else "unknown"


def _action_counts_for_samples(
    *,
    samples: list[dict[str, Any]],
    action_space: set[str],
) -> dict[str, int]:
    """Count how many OPE samples chose each action."""
    action_counts = dict.fromkeys(sorted(action_space), 0)
    for sample in samples:
        action = str(sample.get("action") or "")
        if action in action_counts:
            action_counts[action] += 1
    return action_counts


def _samples_by_feature_scaling_profile(
    samples: list[dict[str, Any]],
) -> dict[str, list[dict[str, Any]]]:
    """Partition OPE samples by their feature-scaling profile."""
    buckets: dict[str, list[dict[str, Any]]] = {}
    for sample in samples:
        profile = _feature_scaling_profile_bucket(sample.get("feature_scaling_profile"))
        buckets.setdefault(profile, []).append(sample)
    return {profile: buckets[profile] for profile in sorted(buckets)}


def _action_rates_for_counts(
    *,
    action_counts: dict[str, int],
    total_samples: int,
) -> dict[str, float | None]:
    """Convert action counts into rates with explicit empty-sample handling."""
    if total_samples <= 0:
        return dict.fromkeys(action_counts, None)
    return {action: (count / float(total_samples)) for action, count in action_counts.items()}


def _shannon_entropy_from_counts(action_counts: dict[str, int]) -> float | None:
    """Return action entropy in nats, or ``None`` when no samples are present."""
    total = sum(max(0, int(count)) for count in action_counts.values())
    if total <= 0:
        return None
    entropy = 0.0
    for count in action_counts.values():
        if count <= 0:
            continue
        probability = count / float(total)
        entropy -= probability * math.log(probability)
    return entropy


@dataclass(frozen=True)
class TargetMassSupport:
    """How much of a per-sample target's probability mass the behavior policy identifies.

    ``matched_samples`` (and therefore ``support_rate``) only says that the action a sample
    actually took carries *some* target probability. It says nothing about the rest of the
    target mass: with a deterministic behavior policy, mass on any other action has no
    behavior support at all, so IPS/SNIPS/DR estimate a truncated policy rather than the
    target policy. These fields preserve that distinction.
    """

    supported_mass: float | None
    unsupported_mass: float | None
    unsupported_samples: int
    deterministic_behavior_samples: int

    @property
    def identifiable(self) -> bool:
        """Whether the estimates can be read as estimates of the full target policy."""
        if self.unsupported_mass is None:
            return False
        return float(self.unsupported_mass) <= ADAPTIVE_OPE_UNSUPPORTED_MASS_TOLERANCE


def _target_mass_support(
    *,
    samples: list[dict[str, Any]],
    target_distributions: list[dict[str, float]],
    action_space: set[str],
) -> TargetMassSupport:
    """Behavior support of the target mass, per sample and averaged.

    An action is supported for a sample when the sample took it, or when the sample's behavior
    was stochastic (propensity < 1) and the action appears as a logged action somewhere in the
    data. Everything else receives target mass that no observation can inform.
    """
    total = len(samples)
    if total <= 0:
        return TargetMassSupport(None, None, 0, 0)
    observed = _action_counts_for_samples(samples=samples, action_space=action_space)
    supported_sum = 0.0
    unsupported_sum = 0.0
    unsupported_samples = 0
    deterministic_samples = 0
    for sample, distribution in zip(samples, target_distributions, strict=True):
        propensity = _sanitize_propensity(
            sample.get("propensity"),
            floor=ADAPTIVE_OPE_RAW_PROPENSITY_FLOOR,
        )
        deterministic = propensity >= ADAPTIVE_OPE_DETERMINISTIC_PROPENSITY
        deterministic_samples += int(deterministic)
        executed = str(sample.get("action") or "")
        supported = 0.0
        unsupported = 0.0
        for action, raw_probability in distribution.items():
            probability = _as_float_or_none(raw_probability) or 0.0
            if probability <= 0.0:
                continue
            action_name = str(action)
            has_support = action_name == executed or (
                not deterministic and int(observed.get(action_name, 0)) > 0
            )
            if has_support:
                supported += probability
            else:
                unsupported += probability
        supported_sum += supported
        unsupported_sum += unsupported
        if unsupported > ADAPTIVE_OPE_UNSUPPORTED_MASS_TOLERANCE:
            unsupported_samples += 1
    return TargetMassSupport(
        supported_mass=supported_sum / float(total),
        unsupported_mass=unsupported_sum / float(total),
        unsupported_samples=unsupported_samples,
        deterministic_behavior_samples=deterministic_samples,
    )


def _support_warning_for_target(
    *,
    matched_samples: int,
    total_samples: int,
    min_action_support: int,
) -> str:
    """Summarize behavior-policy support risk for a deterministic target."""
    if total_samples <= 0:
        return "no_samples"
    if matched_samples <= 0:
        return "no_behavior_support"
    if matched_samples < max(2, int(min_action_support)):
        return "weak_behavior_support"
    return ""


def _finite_estimator_values(target: dict[str, Any]) -> list[tuple[str, float]]:
    """Collect finite estimator outputs from an OPE target summary."""
    values: list[tuple[str, float]] = []
    for name in ("ips", "snips", "dr"):
        raw_value = target.get(name)
        if isinstance(raw_value, (int, float)) and math.isfinite(float(raw_value)):
            values.append((name, float(raw_value)))
    return values


def _estimator_agreement_summary(target: dict[str, Any]) -> dict[str, Any]:
    """Flag estimator disagreement without over-interpreting sparse estimates."""
    values = _finite_estimator_values(target)
    estimator_names = [name for name, _value in values]
    if len(values) < 2:
        return {
            "estimators_available": estimator_names,
            "estimator_agreement": "insufficient_estimators",
            "estimator_agreement_warning": "insufficient_estimators",
        }

    numeric_values = [value for _name, value in values]
    span = max(numeric_values) - min(numeric_values)
    scale = max(max(abs(value) for value in numeric_values), 1.0)
    signs = {1 if value > 0.0 else -1 if value < 0.0 else 0 for value in numeric_values}
    agrees = len(signs) <= 1 and span <= max(
        ADAPTIVE_OPE_ESTIMATOR_AGREEMENT_ABS_TOL, scale * ADAPTIVE_OPE_ESTIMATOR_AGREEMENT_REL_TOL
    )
    return {
        "estimators_available": estimator_names,
        "estimator_agreement": "agree" if agrees else "mixed",
        "estimator_agreement_warning": "" if agrees else "mixed_estimator_signal",
    }


ADAPTIVE_ONLINE_REWARD_SCOPE_LEGACY = "helper_local_v1"


ADAPTIVE_BUDGET_REWARD_SCOPE_LEGACY = "budget_local_v1"


ADAPTIVE_ONLINE_REWARD_SCOPE = "helper_credit_v2"


ADAPTIVE_BUDGET_REWARD_SCOPE = "budget_credit_v2"


ADAPTIVE_ONLINE_REWARD_SCOPES = frozenset(
    {ADAPTIVE_ONLINE_REWARD_SCOPE_LEGACY, ADAPTIVE_ONLINE_REWARD_SCOPE}
)


ADAPTIVE_BUDGET_REWARD_SCOPES = frozenset(
    {ADAPTIVE_BUDGET_REWARD_SCOPE_LEGACY, ADAPTIVE_BUDGET_REWARD_SCOPE}
)


ADAPTIVE_ONLINE_ACTIONS = set(ONLINE_POLICY_ACTIONS)


ADAPTIVE_BUDGET_ACTIONS = {"keep_budget", "increase_budget"}


ADAPTIVE_BUDGET_FAMILIES = frozenset({"time", "tokens", "context"})
