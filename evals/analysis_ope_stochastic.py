"""Off-policy estimates for stochastic targets, including the decision model's advice."""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from evals.adaptive_online_policy import ACTION_INVOKE_HELPER, ACTION_SKIP_HELPER
from evals.analysis_numeric import _as_float_or_none, _mean
from evals.analysis_ope_estimators import (
    _crossfit_reward_models,
    _fit_action_reward_models,
    _ope_case_bootstrap,
    _predict_action_reward,
)
from evals.analysis_ope_support import (
    _ONLINE_ACTION_RANK,
    _action_counts_for_samples,
    _coerce_feature_values,
    _estimator_agreement_summary,
    _feature_scaling_profile_bucket,
    _sanitize_propensity,
    _support_warning_for_target,
)
from evals.decision_model.policy import atomic_action_from_answers
from evals.decision_model.schema import AtomicThresholds, ParsedAnswers, get_profile

DECISION_MODEL_ADVICE_STATUS_OK = "ok"


DECISION_MODEL_TARGET_GREEDY = "decision_model_greedy"


DECISION_MODEL_TARGET_STOCHASTIC = "decision_model_stochastic"


DECISION_MODEL_TARGET_ATOMIC = "decision_model_atomic"


# Thresholds of the atomic profile, applied offline to aux answers logged by other profiles.
DECISION_MODEL_ATOMIC_THRESHOLDS = get_profile("evidence_atomic_v1").atomic or AtomicThresholds()


# Target name -> derivation kind for the per-sample decision-model policies.  These are only
# emitted when at least one sample carries usable advice, so runs without a decision model keep
# their historical OPE output.
DECISION_MODEL_TARGET_KINDS: tuple[tuple[str, str], ...] = (
    (DECISION_MODEL_TARGET_GREEDY, "greedy"),
    (DECISION_MODEL_TARGET_STOCHASTIC, "stochastic"),
    (DECISION_MODEL_TARGET_ATOMIC, "atomic"),
)


@dataclass(frozen=True)
class StochasticTargetOpeSettings:
    """Estimator settings shared by the per-sample (stochastic) OPE targets."""

    action_space: set[str]
    min_action_support: int
    ridge_alpha: float
    propensity_floor: float


def _decision_model_advice_ok(sample: Mapping[str, Any]) -> bool:
    """Whether a sample carries usable decision-model advice."""
    return str(sample.get("dm_status") or "") == DECISION_MODEL_ADVICE_STATUS_OK


def _ordered_actions(actions: Iterable[str]) -> list[str]:
    """Order actions by the controller's action order, with unknown actions last."""
    unknown_rank = len(_ONLINE_ACTION_RANK)
    return sorted(
        {str(action) for action in actions},
        key=lambda action: (_ONLINE_ACTION_RANK.get(action, unknown_rank), action),
    )


def _sample_available_actions(
    sample: Mapping[str, Any],
    *,
    action_space: set[str],
) -> list[str]:
    """Feasible actions logged for a sample, falling back to the full action space."""
    raw = sample.get("available_actions")
    if isinstance(raw, (list, tuple)):
        logged = [str(action) for action in raw if str(action) in action_space]
        if logged:
            return _ordered_actions(logged)
    return _ordered_actions(action_space)


def _ope_baseline_action(available_actions: Sequence[str]) -> str:
    """Baseline action used when advice is missing, mirroring the controller's fallback."""
    for action in (ACTION_INVOKE_HELPER, ACTION_SKIP_HELPER):
        if action in available_actions:
            return action
    return str(available_actions[0]) if available_actions else ""


def _one_hot_distribution(action: str, available_actions: Sequence[str]) -> dict[str, float]:
    """Target distribution placing all mass on *action*."""
    return {name: (1.0 if name == action else 0.0) for name in available_actions}


def _derived_atomic_action(sample: Mapping[str, Any], available: Sequence[str]) -> str:
    """Atomic mapping applied to logged aux answers; empty when none were logged.

    Only ``evidence_atomic_v1`` maps in code at decision time; other profiles with aux questions
    log the answers but no mapped action. Clef answers a profile's questions jointly, so aux
    answers logged next to the main question approximate what the atomic profile would answer.
    """
    sufficient = _as_float_or_none(sample.get("dm_traceback_sufficient"))
    raw_missing = sample.get("dm_missing_information")
    missing = _coerce_feature_values(raw_missing)
    if sufficient is None and not missing:
        return ""
    action, _reason = atomic_action_from_answers(
        ParsedAnswers(traceback_sufficient=sufficient, missing_information=missing),
        available_actions=available,
        thresholds=DECISION_MODEL_ATOMIC_THRESHOLDS,
    )
    return action


def _decision_model_target_distribution(
    sample: Mapping[str, Any],
    *,
    kind: str,
    action_space: set[str],
) -> dict[str, float]:
    """Per-sample target distribution for one decision-model target *kind*.

    ``greedy`` is the arg-max over the advice probabilities restricted to the feasible
    actions, ``stochastic`` renormalizes them, and ``atomic`` follows the atomic profile's
    mapped action (derived from the logged aux answers when another profile was used).
    Missing or failed advice falls back to the baseline action.
    """
    available = _sample_available_actions(sample, action_space=action_space)
    if not available:
        return {}
    baseline = _one_hot_distribution(_ope_baseline_action(available), available)
    if not _decision_model_advice_ok(sample):
        return baseline
    if kind == "atomic":
        atomic_action = str(sample.get("dm_atomic_action") or "").strip()
        if not atomic_action:
            atomic_action = _derived_atomic_action(sample, available)
        if atomic_action in available:
            return _one_hot_distribution(atomic_action, available)
        return baseline
    raw_probs = sample.get("dm_action_probs")
    probs = raw_probs if isinstance(raw_probs, dict) else {}
    masses = {
        action: max(0.0, _as_float_or_none(probs.get(action, 0.0)) or 0.0) for action in available
    }
    total = sum(masses.values())
    if total <= 0.0:
        return baseline
    if kind == "greedy":
        best = available[0]
        for action in available[1:]:
            if masses[action] > masses[best]:
                best = action
        return _one_hot_distribution(best, available)
    return {action: (mass / total) for action, mass in masses.items()}


def _decision_model_targets(
    *,
    samples: list[dict[str, Any]],
    settings: StochasticTargetOpeSettings,
) -> dict[str, Any]:
    """Decision-model OPE targets; empty when no sample carries usable advice."""
    if not any(_decision_model_advice_ok(sample) for sample in samples):
        return {}
    targets: dict[str, Any] = {}
    for target_name, kind in DECISION_MODEL_TARGET_KINDS:
        target_distributions = [
            _decision_model_target_distribution(
                sample,
                kind=kind,
                action_space=settings.action_space,
            )
            for sample in samples
        ]
        targets[target_name] = _stochastic_target_ope(
            samples=samples,
            target_distributions=target_distributions,
            settings=settings,
        )
    return targets


def _target_importance_weight(
    sample: Mapping[str, Any],
    distribution: Mapping[str, float],
    *,
    propensity_floor: float,
) -> float:
    """Importance weight ``pi(a_i | x_i) / mu_i`` for one sample."""
    target_probability = _as_float_or_none(distribution.get(str(sample.get("action") or ""), 0.0))
    if target_probability is None or target_probability <= 0.0:
        return 0.0
    behavior_propensity = _sanitize_propensity(sample.get("propensity"), floor=propensity_floor)
    return float(target_probability) / behavior_propensity


def _mean_target_action_probabilities(
    target_distributions: list[dict[str, float]],
) -> dict[str, float]:
    """Mean target probability per action across samples (zero where infeasible)."""
    total = len(target_distributions)
    if total <= 0:
        return {}
    sums: dict[str, float] = {}
    for distribution in target_distributions:
        for action, probability in distribution.items():
            sums[action] = sums.get(action, 0.0) + float(probability)
    return {action: (sums[action] / float(total)) for action in _ordered_actions(sums)}


def _stochastic_target_ope(
    *,
    samples: list[dict[str, Any]],
    target_distributions: list[dict[str, float]],
    settings: StochasticTargetOpeSettings,
) -> dict[str, Any]:
    """IPS / SNIPS / ESS / DR for a target that is a distribution per sample."""
    total = len(samples)
    matched = 0
    weighted_reward_sum = 0.0
    weight_sum = 0.0
    weight_sq_sum = 0.0
    for sample, distribution in zip(samples, target_distributions, strict=True):
        weight = _target_importance_weight(
            sample,
            distribution,
            propensity_floor=settings.propensity_floor,
        )
        if weight <= 0.0:
            continue
        matched += 1
        weighted_reward_sum += weight * float(sample.get("reward") or 0.0)
        weight_sum += weight
        weight_sq_sum += weight * weight
    ips = (weighted_reward_sum / float(total)) if (total > 0 and matched > 0) else None
    snips = (weighted_reward_sum / weight_sum) if weight_sum > 0.0 else None
    ess = ((weight_sum * weight_sum) / weight_sq_sum) if weight_sq_sum > 0.0 else None
    dr_terms, dr_reason = _stratified_stochastic_doubly_robust_terms(
        samples=samples,
        target_distributions=target_distributions,
        settings=settings,
    )
    dr = (sum(dr_terms) / float(len(dr_terms))) if dr_terms else None
    weights = [
        _target_importance_weight(sample, distribution, propensity_floor=settings.propensity_floor)
        for sample, distribution in zip(samples, target_distributions, strict=True)
    ]
    result: dict[str, Any] = {
        "target_kind": "stochastic",
        "ips": ips,
        "snips": snips,
        "dr": dr,
        "effective_sample_size": ess,
        "matched_samples": matched,
        "total_samples": total,
        "support_rate": ((matched / float(total)) if total > 0 else None),
        "support_warning": _support_warning_for_target(
            matched_samples=matched,
            total_samples=total,
            min_action_support=settings.min_action_support,
        ),
        "dr_supported": dr is not None,
        "dr_reason": dr_reason if dr is None else "",
        "mean_action_probabilities": _mean_target_action_probabilities(target_distributions),
        "ope_ci": _ope_case_bootstrap(samples, weights=weights, dr_terms=dr_terms),
    }
    result.update(_estimator_agreement_summary(result))
    return result


def _stratified_stochastic_doubly_robust_estimate(
    *,
    samples: list[dict[str, Any]],
    target_distributions: list[dict[str, float]],
    settings: StochasticTargetOpeSettings,
) -> tuple[float | None, str]:
    """Mean of :func:`_stratified_stochastic_doubly_robust_terms`, or ``None`` and reason."""
    terms, reason = _stratified_stochastic_doubly_robust_terms(
        samples=samples, target_distributions=target_distributions, settings=settings
    )
    if terms is None:
        return None, reason
    return sum(terms) / float(len(terms)), ""


def _stratified_stochastic_doubly_robust_terms(
    *,
    samples: list[dict[str, Any]],
    target_distributions: list[dict[str, float]],
    settings: StochasticTargetOpeSettings,
) -> tuple[list[float] | None, str]:
    """Stratified DR for a per-sample target, reusing the deterministic reward models.

    Like the deterministic estimator this stratifies by feature-scaling profile and
    weight-averages the stratum estimates.  DR is unsupported as soon as one action that
    receives target mass in a stratum has no reward model there.
    """
    total = len(samples)
    if total <= 0:
        return None, "no_samples"
    buckets: dict[str, list[tuple[dict[str, Any], dict[str, float]]]] = {}
    for sample, distribution in zip(samples, target_distributions, strict=True):
        profile = _feature_scaling_profile_bucket(sample.get("feature_scaling_profile"))
        buckets.setdefault(profile, []).append((sample, distribution))

    term_by_sample: dict[int, float] = {}
    for profile in sorted(buckets):
        bucket_samples = [sample for sample, _distribution in buckets[profile]]
        bucket_distributions = [distribution for _sample, distribution in buckets[profile]]
        models = _fit_action_reward_models(
            samples=bucket_samples,
            action_space=settings.action_space,
            min_action_support=settings.min_action_support,
            ridge_alpha=settings.ridge_alpha,
        ).models
        reason = _stochastic_dr_support_reason(
            target_distributions=bucket_distributions,
            models=models,
            action_counts=_action_counts_for_samples(
                samples=bucket_samples,
                action_space=settings.action_space,
            ),
            min_action_support=settings.min_action_support,
        )
        if reason:
            return None, reason
        crossfit = _crossfit_reward_models(
            samples=bucket_samples,
            action_space=settings.action_space,
            min_action_support=settings.min_action_support,
            ridge_alpha=settings.ridge_alpha,
        )
        for sample, distribution, sample_models in zip(
            bucket_samples, bucket_distributions, crossfit.per_sample, strict=True
        ):
            needed = {action for action, prob in distribution.items() if float(prob) > 0.0}
            if _target_importance_weight(
                sample, distribution, propensity_floor=settings.propensity_floor
            ):
                needed.add(str(sample.get("action") or ""))
            if not needed <= set(sample_models):
                return None, "insufficient_crossfit_support"
        bucket_terms = _stochastic_doubly_robust_terms(
            samples=bucket_samples,
            target_distributions=bucket_distributions,
            feature_names=crossfit.feature_names,
            models=crossfit.per_sample,
            propensity_floor=settings.propensity_floor,
        )
        for sample, term in zip(bucket_samples, bucket_terms, strict=True):
            term_by_sample[id(sample)] = term
    return [term_by_sample[id(sample)] for sample in samples], ""


def _stochastic_dr_support_reason(
    *,
    target_distributions: list[dict[str, float]],
    models: dict[str, np.ndarray],
    action_counts: dict[str, int],
    min_action_support: int,
) -> str:
    """Reason why DR is unsupported for one stratum, or ``""`` when it is supported."""
    weighted_actions = {
        action
        for distribution in target_distributions
        for action, probability in distribution.items()
        if float(probability) > 0.0
    }
    for action in _ordered_actions(weighted_actions):
        if action in models:
            continue
        if int(action_counts.get(action, 0)) < int(min_action_support):
            return "insufficient_action_support"
        return "missing_action_model"
    return ""


def _stochastic_doubly_robust_estimate(
    *,
    samples: list[dict[str, Any]],
    target_distributions: list[dict[str, float]],
    feature_names: list[str],
    models: dict[str, np.ndarray] | list[dict[str, np.ndarray]],
    propensity_floor: float,
) -> float:
    """Single-stratum DR estimate for a per-sample target distribution.

    *models* is one shared model set or one out-of-fold set per sample (STA-11).
    """
    return float(
        _mean(
            _stochastic_doubly_robust_terms(
                samples=samples,
                target_distributions=target_distributions,
                feature_names=feature_names,
                models=models,
                propensity_floor=propensity_floor,
            )
        )
        or 0.0
    )


def _stochastic_doubly_robust_terms(
    *,
    samples: list[dict[str, Any]],
    target_distributions: list[dict[str, float]],
    feature_names: list[str],
    models: dict[str, np.ndarray] | list[dict[str, np.ndarray]],
    propensity_floor: float,
) -> list[float]:
    """Per-sample DR terms for a per-sample target distribution."""
    model_sets = models if isinstance(models, list) else [models] * len(samples)
    terms: list[float] = []
    for sample, distribution, models in zip(samples, target_distributions, model_sets, strict=True):
        features_raw = sample.get("features")
        features = features_raw if isinstance(features_raw, dict) else {}
        direct = 0.0
        for action, probability in distribution.items():
            if float(probability) <= 0.0:
                continue
            direct += float(probability) * _predict_action_reward(
                features=features,
                feature_names=feature_names,
                beta=models[action],
            )
        weight = _target_importance_weight(sample, distribution, propensity_floor=propensity_floor)
        correction = 0.0
        if weight > 0.0:
            behavior_hat = _predict_action_reward(
                features=features,
                feature_names=feature_names,
                beta=models[str(sample.get("action") or "")],
            )
            correction = weight * (float(sample.get("reward") or 0.0) - behavior_hat)
        terms.append(direct + correction)
    return terms
