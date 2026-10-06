"""Off-policy estimators: reward models, cross-fitting, doubly-robust estimates and the case bootstrap."""

from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.linear_model import RidgeCV

from evals.analysis_numeric import _mean
from evals.analysis_ope_support import (
    TargetOpeSettings,
    _action_counts_for_samples,
    _estimator_agreement_summary,
    _sample_case_keys,
    _samples_by_feature_scaling_profile,
    _sanitize_propensity,
    _support_warning_for_target,
)
from evals.stats import BOOTSTRAP_METHOD, DEFAULT_BOOTSTRAP_SEED, percentile_interval

logger = logging.getLogger(__name__)


ADAPTIVE_OPE_RIDGE_ALPHA = 1e-3


ADAPTIVE_OPE_RIDGE_ALPHAS = (0.001, 0.01, 0.1, 1.0, 10.0, 100.0)


# Minimum per-action sample count required for cross-validated alpha selection.
# Below this threshold we fall back to ADAPTIVE_OPE_RIDGE_ALPHA.
_RIDGECV_MIN_SAMPLES = 10


# STA-11: DR reward models are cross-fitted over this many folds per stratum.
ADAPTIVE_OPE_CROSSFIT_FOLDS = 5


@dataclass(frozen=True)
class FittedRewardModels:
    """Per-action reward models and the ridge alpha each one used."""

    feature_names: list[str]
    models: dict[str, np.ndarray]
    alphas: dict[str, float]


def _ope_feature_names(samples: list[dict[str, Any]]) -> list[str]:
    return sorted(
        {
            str(name)
            for sample in samples
            for name in (
                sample.get("features", {}).keys()
                if isinstance(sample.get("features"), dict)
                else []
            )
        }
    )


def _fit_action_reward_models(
    *,
    samples: list[dict[str, Any]],
    action_space: set[str],
    min_action_support: int,
    ridge_alpha: float,
    feature_names: list[str] | None = None,
) -> FittedRewardModels:
    """Fit per-action ridge regression models predicting reward.

    When a per-action subset has at least ``_RIDGECV_MIN_SAMPLES`` rows,
    ``RidgeCV`` with 5-fold cross-validation selects the best alpha from
    ``ADAPTIVE_OPE_RIDGE_ALPHAS``.  Otherwise the hardcoded *ridge_alpha*
    fallback is used. The alpha used per action is returned (STA-11).

    Actions with fewer samples than *min_action_support* are skipped.
    """
    if feature_names is None:
        feature_names = _ope_feature_names(samples)
    dim = len(feature_names) + 1  # +1 for bias term
    models: dict[str, np.ndarray] = {}
    alphas: dict[str, float] = {}
    for action in sorted(action_space):
        action_samples = [sample for sample in samples if str(sample.get("action") or "") == action]
        if len(action_samples) < int(min_action_support):
            continue
        X = np.ones((len(action_samples), dim), dtype=float)
        y = np.zeros((len(action_samples),), dtype=float)
        for row_idx, sample in enumerate(action_samples):
            y[row_idx] = float(sample.get("reward") or 0.0)
            features = sample.get("features")
            feature_dict = features if isinstance(features, dict) else {}
            for col_idx, name in enumerate(feature_names, start=1):
                X[row_idx, col_idx] = float(feature_dict.get(name, 0.0) or 0.0)

        if len(action_samples) >= _RIDGECV_MIN_SAMPLES:
            # Enough data: cross-validate to pick the best alpha.
            cv_folds = min(5, len(action_samples))
            model = RidgeCV(
                alphas=ADAPTIVE_OPE_RIDGE_ALPHAS,
                cv=cv_folds,
                fit_intercept=False,  # bias column already in X
            )
            model.fit(X, y)
            beta = np.asarray(model.coef_, dtype=float)
            alphas[action] = float(model.alpha_)
            logger.info(
                "RidgeCV selected alpha=%.4g for action=%r (%d samples, %d-fold)",
                model.alpha_,
                action,
                len(action_samples),
                cv_folds,
            )
        else:
            # Fallback: closed-form ridge with the hardcoded alpha.
            logger.debug(
                "Fallback to hardcoded alpha=%.4g for action=%r (%d samples < %d)",
                ridge_alpha,
                action,
                len(action_samples),
                _RIDGECV_MIN_SAMPLES,
            )
            gram = X.T @ X
            if ridge_alpha > 0.0:
                gram = gram + (np.eye(dim, dtype=float) * float(ridge_alpha))
            rhs = X.T @ y
            try:
                beta = np.linalg.solve(gram, rhs)
            except np.linalg.LinAlgError:
                beta = np.linalg.pinv(gram) @ rhs
            alphas[action] = float(ridge_alpha)

        models[action] = beta
    return FittedRewardModels(feature_names=feature_names, models=models, alphas=alphas)


def _fit_ridge_action_models(
    *,
    samples: list[dict[str, Any]],
    action_space: set[str],
    min_action_support: int,
    ridge_alpha: float,
) -> tuple[list[str], dict[str, np.ndarray]]:
    """Return ``(feature_names, models)`` from :func:`_fit_action_reward_models`."""
    fitted = _fit_action_reward_models(
        samples=samples,
        action_space=action_space,
        min_action_support=min_action_support,
        ridge_alpha=ridge_alpha,
    )
    return fitted.feature_names, fitted.models


@dataclass(frozen=True)
class CrossFitModels:
    """Out-of-fold reward models for every sample of one stratum (STA-11)."""

    feature_names: list[str]
    per_sample: list[dict[str, np.ndarray]]
    alphas: dict[str, list[float]]


def _crossfit_reward_models(
    *,
    samples: list[dict[str, Any]],
    action_space: set[str],
    min_action_support: int,
    ridge_alpha: float,
) -> CrossFitModels:
    """K-fold cross-fitting: each sample's predictions come from the other folds' models.

    The DR reward model used to be fitted on the same samples whose residuals form the
    correction term. Folds are assigned by *case* (sorted case order modulo K), never by
    row position, so every attempt and every repeated run of a case is held out together
    and nothing of a case remains in its own training set (STA-11, STA-02). With fewer
    than two cases there is nothing to cross-fit and every sample gets an empty model
    set, which makes DR unsupported.
    """
    feature_names = _ope_feature_names(samples)
    case_keys = _sample_case_keys(samples)
    distinct_cases = sorted(set(case_keys))
    folds = min(ADAPTIVE_OPE_CROSSFIT_FOLDS, len(distinct_cases))
    if folds < 2:
        return CrossFitModels(feature_names, [{} for _ in samples], {})
    fold_of_case = {case: index % folds for index, case in enumerate(distinct_cases)}
    sample_folds = [fold_of_case[case] for case in case_keys]
    fold_models: list[dict[str, np.ndarray]] = []
    alphas: dict[str, list[float]] = defaultdict(list)
    for fold in range(folds):
        fitted = _fit_action_reward_models(
            samples=[
                sample
                for sample, sample_fold in zip(samples, sample_folds, strict=True)
                if sample_fold != fold
            ],
            action_space=action_space,
            min_action_support=min_action_support,
            ridge_alpha=ridge_alpha,
            feature_names=feature_names,
        )
        fold_models.append(fitted.models)
        for action, alpha in fitted.alphas.items():
            alphas[action].append(alpha)
    return CrossFitModels(
        feature_names=feature_names,
        per_sample=[fold_models[sample_fold] for sample_fold in sample_folds],
        alphas=dict(alphas),
    )


def _predict_action_reward(
    *,
    features: dict[str, float],
    feature_names: list[str],
    beta: np.ndarray,
) -> float:
    """Predict reward for a single sample using a fitted ridge model."""
    vector = np.ones((len(feature_names) + 1,), dtype=float)
    for idx, name in enumerate(feature_names, start=1):
        vector[idx] = float(features.get(name, 0.0) or 0.0)
    return float(vector @ beta)


def _deterministic_target_ope(
    *,
    samples: list[dict[str, Any]],
    target_action: str,
    action_counts: dict[str, int],
    settings: TargetOpeSettings,
) -> dict[str, Any]:
    """Compute IPS, SNIPS, and DR estimates for a deterministic target.

    The target policy always selects *target_action*.  Returns a dict
    with IPS, SNIPS, DR, effective sample size, and support info.
    """
    action_space = settings.action_space
    min_action_support = settings.min_action_support
    ridge_alpha = settings.ridge_alpha
    propensity_floor = settings.propensity_floor
    total = len(samples)
    matched = 0
    weighted_reward_sum = 0.0
    weight_sum = 0.0
    weight_sq_sum = 0.0
    for sample in samples:
        action = str(sample.get("action") or "")
        if action != target_action:
            continue
        propensity = _sanitize_propensity(sample.get("propensity"), floor=propensity_floor)
        reward = float(sample.get("reward") or 0.0)
        weight = 1.0 / propensity
        matched += 1
        weighted_reward_sum += weight * reward
        weight_sum += weight
        weight_sq_sum += weight * weight
    ips = (weighted_reward_sum / float(total)) if (total > 0 and matched > 0) else None
    snips = (weighted_reward_sum / weight_sum) if weight_sum > 0.0 else None
    ess = ((weight_sum * weight_sum) / weight_sq_sum) if weight_sq_sum > 0.0 else None

    dr_terms, dr_reason = _stratified_doubly_robust_terms(
        samples=samples,
        target_action=target_action,
        action_space=action_space,
        min_action_support=min_action_support,
        ridge_alpha=ridge_alpha,
        propensity_floor=propensity_floor,
    )
    dr = (sum(dr_terms) / float(len(dr_terms))) if dr_terms else None
    weights = [
        (
            1.0 / _sanitize_propensity(sample.get("propensity"), floor=propensity_floor)
            if str(sample.get("action") or "") == target_action
            else 0.0
        )
        for sample in samples
    ]

    support_warning = _support_warning_for_target(
        matched_samples=matched,
        total_samples=total,
        min_action_support=min_action_support,
    )
    result = {
        "target_action": target_action,
        "ips": ips,
        "snips": snips,
        "dr": dr,
        "effective_sample_size": ess,
        "matched_samples": matched,
        "total_samples": total,
        "support_rate": ((matched / float(total)) if total > 0 else None),
        "support_warning": support_warning,
        "dr_supported": dr is not None,
        "dr_reason": dr_reason if dr is None else "",
        "ope_ci": _ope_case_bootstrap(
            samples, weights=weights, dr_terms=dr_terms, seed=settings.bootstrap_seed
        ),
    }
    result.update(_estimator_agreement_summary(result))
    return result


OPE_BOOTSTRAP_RESAMPLES = 1000


def _ope_case_bootstrap(
    samples: list[dict[str, Any]],
    *,
    weights: list[float],
    dr_terms: list[float] | None,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """95% percentile bootstrap over cases for IPS, SNIPS and DR (Phase 2.3).

    Off-policy estimates carried no uncertainty. One cluster per *case*, holding all of
    its attempts and all of its repeated runs together, so repeating a run cannot narrow
    the interval (STA-02); the cross-fitted reward models are held fixed, so DR resamples
    its per-sample terms. Seeded with the analysis bootstrap seed (STA-05, STA-07).
    """
    by_case: dict[str, list[int]] = defaultdict(list)
    for idx, case_key in enumerate(_sample_case_keys(samples)):
        by_case[case_key].append(idx)
    cases = [by_case[key] for key in sorted(by_case)]
    if len(cases) < 2:
        return {"available": False, "n_cases": len(cases)}
    rewards = np.array([float(sample.get("reward") or 0.0) for sample in samples])
    weight_arr = np.asarray(weights, dtype=float)
    dr_arr = np.asarray(dr_terms if dr_terms is not None else [0.0] * len(samples), dtype=float)
    totals = np.array(
        [
            [
                float((weight_arr[idx] * rewards[idx]).sum()),
                float(weight_arr[idx].sum()),
                float(dr_arr[idx].sum()),
                float(len(idx)),
            ]
            for idx in cases
        ]
    )
    rng = np.random.default_rng(int(seed))
    draws = totals[rng.integers(0, len(cases), size=(OPE_BOOTSTRAP_RESAMPLES, len(cases)))].sum(
        axis=1
    )
    weighted, weight, dr_sum, n = draws[:, 0], draws[:, 1], draws[:, 2], draws[:, 3]
    ips_draws = (weighted / n).tolist()
    snips_draws = (weighted[weight > 0] / weight[weight > 0]).tolist()
    dr_draws = (dr_sum / n).tolist()
    # No sample carries importance weight: the point estimators report ips/snips as null with
    # no_behavior_support, so the resampled zeros must not become a certain-looking [0, 0].
    importance_weighted = bool((weight_arr > 0.0).any())

    def interval(draws: list[float]) -> list[float] | None:
        return list(percentile_interval(draws, 0.95)) if draws else None

    ips_interval = interval(ips_draws) if importance_weighted else None
    snips_interval = interval(snips_draws) if importance_weighted else None
    dr_interval = interval(dr_draws) if dr_terms is not None else None
    return {
        "available": any(
            estimate is not None for estimate in (ips_interval, snips_interval, dr_interval)
        ),
        "unit": "case",
        "n_cases": len(cases),
        "n_bootstrap": OPE_BOOTSTRAP_RESAMPLES,
        "method": BOOTSTRAP_METHOD,
        "seed": int(seed),
        "ips": ips_interval,
        "snips": snips_interval,
        "dr": dr_interval,
    }


def _stratified_doubly_robust_estimate(
    *,
    samples: list[dict[str, Any]],
    target_action: str,
    action_space: set[str],
    min_action_support: int,
    ridge_alpha: float,
    propensity_floor: float,
) -> tuple[float | None, str]:
    """Stratified doubly-robust OPE across feature-scaling profiles.

    Fits per-action ridge models within each feature-scaling stratum (cross-fitted),
    and averages the per-sample DR terms, which weight-averages the stratum estimates by
    sample count.

    Returns:
        Tuple of (DR estimate or None, reason string if unsupported).
    """
    terms, reason = _stratified_doubly_robust_terms(
        samples=samples,
        target_action=target_action,
        action_space=action_space,
        min_action_support=min_action_support,
        ridge_alpha=ridge_alpha,
        propensity_floor=propensity_floor,
    )
    if terms is None:
        return None, reason
    return sum(terms) / float(len(terms)), ""


def _stratified_doubly_robust_terms(
    *,
    samples: list[dict[str, Any]],
    target_action: str,
    action_space: set[str],
    min_action_support: int,
    ridge_alpha: float,
    propensity_floor: float,
) -> tuple[list[float] | None, str]:
    """Per-sample DR terms aligned with *samples*, or ``None`` and the reason."""
    total = len(samples)
    if total <= 0:
        return None, "no_samples"
    if target_action not in action_space:
        return None, "target_action_out_of_space"

    term_by_sample: dict[int, float] = {}
    for bucket_samples in _samples_by_feature_scaling_profile(samples).values():
        action_counts = _action_counts_for_samples(
            samples=bucket_samples,
            action_space=action_space,
        )
        fitted = _fit_action_reward_models(
            samples=bucket_samples,
            action_space=action_space,
            min_action_support=min_action_support,
            ridge_alpha=ridge_alpha,
        )
        models = fitted.models
        observed_actions = {
            action for action in action_space if int(action_counts.get(action, 0)) > 0
        }
        supported_actions = {
            action
            for action in observed_actions
            if int(action_counts.get(action, 0)) >= min_action_support
        }
        # DR remains stricter than IPS/SNIPS: each feature-scaling bucket needs enough
        # support to fit the target model and at least one alternate behavior model before
        # we aggregate.
        if int(action_counts.get(target_action, 0)) < min_action_support:
            return None, "insufficient_action_support"
        if not any(action != target_action for action in supported_actions):
            return None, "insufficient_action_support"
        if any(
            int(action_counts.get(action, 0)) >= min_action_support and action not in models
            for action in observed_actions
        ):
            return None, "missing_action_model"
        if target_action not in models:
            return None, "missing_target_model"
        crossfit = _crossfit_reward_models(
            samples=bucket_samples,
            action_space=action_space,
            min_action_support=min_action_support,
            ridge_alpha=ridge_alpha,
        )
        if any(target_action not in sample_models for sample_models in crossfit.per_sample):
            return None, "insufficient_crossfit_support"
        bucket_terms = _doubly_robust_terms(
            samples=bucket_samples,
            target_action=target_action,
            feature_names=crossfit.feature_names,
            models=crossfit.per_sample,
            propensity_floor=propensity_floor,
        )
        for sample, term in zip(bucket_samples, bucket_terms, strict=True):
            term_by_sample[id(sample)] = term
    return [term_by_sample[id(sample)] for sample in samples], ""


def _doubly_robust_estimate(
    *,
    samples: list[dict[str, Any]],
    target_action: str,
    feature_names: list[str],
    models: dict[str, np.ndarray] | list[dict[str, np.ndarray]],
    propensity_floor: float,
) -> float:
    """Compute a single-stratum doubly-robust reward estimate.

    Combines the fitted regression prediction with an IPS correction
    term for samples matching the target action. *models* is one shared model set or,
    when cross-fitted, one out-of-fold set per sample (STA-11).
    """
    return float(
        _mean(
            _doubly_robust_terms(
                samples=samples,
                target_action=target_action,
                feature_names=feature_names,
                models=models,
                propensity_floor=propensity_floor,
            )
        )
        or 0.0
    )


def _doubly_robust_terms(
    *,
    samples: list[dict[str, Any]],
    target_action: str,
    feature_names: list[str],
    models: dict[str, np.ndarray] | list[dict[str, np.ndarray]],
    propensity_floor: float,
) -> list[float]:
    """Per-sample DR terms: target prediction plus the IPS-weighted residual."""
    model_sets = models if isinstance(models, list) else [models] * len(samples)
    terms: list[float] = []
    for sample, models in zip(samples, model_sets, strict=True):
        action = str(sample.get("action") or "")
        propensity = _sanitize_propensity(sample.get("propensity"), floor=propensity_floor)
        reward = float(sample.get("reward") or 0.0)
        features_raw = sample.get("features")
        features = features_raw if isinstance(features_raw, dict) else {}
        target_hat = _predict_action_reward(
            features=features,
            feature_names=feature_names,
            beta=models[target_action],
        )
        # Only the target action gets a correction, and for it the behaviour model's
        # prediction is target_hat (STA-26: the other actions' predictions were unused).
        correction = (reward - target_hat) / propensity if action == target_action else 0.0
        terms.append(target_hat + correction)
    return terms
