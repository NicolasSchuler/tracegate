"""Off-policy evaluation of the adaptive policies: decision samples and per-target summaries."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from evals.analysis_numeric import _as_float_or_none, _as_int, _mean
from evals.analysis_ope_estimators import (
    ADAPTIVE_OPE_CROSSFIT_FOLDS,
    ADAPTIVE_OPE_RIDGE_ALPHA,
    _crossfit_reward_models,
    _deterministic_target_ope,
)
from evals.analysis_ope_stochastic import (
    StochasticTargetOpeSettings,
    _decision_model_advice_ok,
    _decision_model_targets,
)
from evals.analysis_ope_support import (
    ADAPTIVE_BUDGET_REWARD_SCOPES,
    ADAPTIVE_ONLINE_REWARD_SCOPES,
    ADAPTIVE_OPE_RAW_PROPENSITY_FLOOR,
    _action_counts_for_samples,
    _action_rates_for_counts,
    _coerce_feature_values,
    _feature_scaling_profile_bucket,
    _metadata_with_fallback,
    _samples_by_feature_scaling_profile,
    _sanitize_propensity,
    _shannon_entropy_from_counts,
)

ADAPTIVE_ONLINE_TARGET_POLICY_ITEMS = (
    ("always_skip", "skip_helper"),
    ("always_frame_only", "frame_only"),
    ("always_file_only", "file_only"),
    ("always_invoke_helper", "invoke_helper"),
)


ADAPTIVE_ONLINE_TARGET_POLICIES = dict(ADAPTIVE_ONLINE_TARGET_POLICY_ITEMS)


ADAPTIVE_ONLINE_TARGET_POLICY_ORDER = tuple(
    target_name for target_name, _target_action in ADAPTIVE_ONLINE_TARGET_POLICY_ITEMS
)


ADAPTIVE_BUDGET_TARGET_POLICY_ITEMS = (
    ("always_keep_budget", "keep_budget"),
    ("always_increase_budget", "increase_budget"),
)


ADAPTIVE_BUDGET_TARGET_POLICIES = dict(ADAPTIVE_BUDGET_TARGET_POLICY_ITEMS)


ADAPTIVE_BUDGET_TARGET_POLICY_ORDER = tuple(
    target_name for target_name, _target_action in ADAPTIVE_BUDGET_TARGET_POLICY_ITEMS
)


ADAPTIVE_OPE_SHADOW_POLICY_MODE = "shadow"


def _adaptive_ope_samples(
    *,
    records: list[dict[str, Any]],
    prefix: str,
    valid_actions: set[str],
) -> list[dict[str, Any]]:
    """Extract off-policy evaluation samples from raw attempt records.

    Each sample contains (action, propensity, reward, features,
    feature_scaling_profile) for use in IPS, SNIPS, and DR estimators, plus the feasible
    action set and any decision-model advice needed by the per-sample targets.

    Shadow-mode rows are corrected at read time: the executed action is the deterministic
    baseline there, so its behavior propensity is 1.0 regardless of what the controller
    logged (older logs recorded the bandit's propensity, which inflates IPS weights).
    """
    samples: list[dict[str, Any]] = []
    for record in records:
        if _as_int(record.get("attempt")) <= 0:
            continue
        attempt_meta = record.get("attempt_metadata")
        patch_meta = record.get("patch_meta")
        attempt_meta_dict = attempt_meta if isinstance(attempt_meta, dict) else {}
        patch_meta_dict = patch_meta if isinstance(patch_meta, dict) else {}

        decision_id = _metadata_with_fallback(
            attempt_meta_dict,
            patch_meta_dict,
            key=f"{prefix}_decision_id",
        )
        if not isinstance(decision_id, str) or not decision_id.strip():
            continue

        reward_scope = _metadata_with_fallback(
            attempt_meta_dict,
            patch_meta_dict,
            key=f"{prefix}_reward_scope",
        )
        expected_reward_scopes = (
            ADAPTIVE_ONLINE_REWARD_SCOPES
            if prefix == "adaptive_online"
            else ADAPTIVE_BUDGET_REWARD_SCOPES
        )
        if reward_scope not in expected_reward_scopes:
            continue

        reward_raw = _metadata_with_fallback(
            attempt_meta_dict,
            patch_meta_dict,
            key=f"{prefix}_reward",
        )
        reward = _as_float_or_none(reward_raw)
        if reward is None:
            continue

        behavior_action_raw = _metadata_with_fallback(
            attempt_meta_dict,
            patch_meta_dict,
            key=f"{prefix}_behavior_action",
        )
        if behavior_action_raw is None:
            behavior_action_raw = _metadata_with_fallback(
                attempt_meta_dict,
                patch_meta_dict,
                key=f"{prefix}_action_applied",
            )
        behavior_action = str(behavior_action_raw or "").strip()
        if behavior_action not in valid_actions:
            continue

        behavior_propensity_raw = _metadata_with_fallback(
            attempt_meta_dict,
            patch_meta_dict,
            key=f"{prefix}_behavior_propensity",
        )
        if behavior_propensity_raw is None:
            behavior_propensity_raw = _metadata_with_fallback(
                attempt_meta_dict,
                patch_meta_dict,
                key=f"{prefix}_propensity",
            )
        # STA-12: a missing propensity used to become 1.0; such samples are now marked
        # and dropped from IPS/SNIPS/DR by the summary, which counts them.
        propensity_missing = _as_float_or_none(behavior_propensity_raw) is None
        logged_propensity = _sanitize_propensity(
            behavior_propensity_raw,
            floor=ADAPTIVE_OPE_RAW_PROPENSITY_FLOOR,
        )
        policy_mode = str(
            _metadata_with_fallback(
                attempt_meta_dict,
                patch_meta_dict,
                key=f"{prefix}_policy_mode",
            )
            or ""
        ).strip()
        shadow_mode = policy_mode == ADAPTIVE_OPE_SHADOW_POLICY_MODE
        propensity = 1.0 if shadow_mode else logged_propensity
        propensity_missing = propensity_missing and not shadow_mode

        features_raw = _metadata_with_fallback(
            attempt_meta_dict,
            patch_meta_dict,
            key=f"{prefix}_feature_vector",
        )
        feature_values = _coerce_feature_values(features_raw)
        feature_scaling_profile = _feature_scaling_profile_bucket(
            _metadata_with_fallback(
                attempt_meta_dict,
                patch_meta_dict,
                key=f"{prefix}_feature_scaling_profile",
            )
        )

        samples.append(
            {
                "action": behavior_action,
                "propensity": propensity,
                "reward": reward,
                "features": feature_values,
                "feature_scaling_profile": feature_scaling_profile,
                "propensity_shadow_corrected": shadow_mode and logged_propensity < 1.0,
                "propensity_missing": propensity_missing,
                "case_key": f"{record.get('run_id') or ''}\x1f{record.get('case_id') or ''}",
                **_ope_sample_policy_context(
                    attempt_meta_dict,
                    patch_meta_dict,
                    prefix=prefix,
                    valid_actions=valid_actions,
                ),
            }
        )
    return samples


def _ope_sample_policy_context(
    attempt_meta_dict: dict[str, Any],
    patch_meta_dict: dict[str, Any],
    *,
    prefix: str,
    valid_actions: set[str],
) -> dict[str, Any]:
    """Feasible action set and decision-model advice for one OPE sample.

    Rows written before the decision model existed (and every budget-policy row) simply
    yield an empty advice block, which keeps the decision-model targets out of the summary.
    """
    available_raw = _metadata_with_fallback(
        attempt_meta_dict,
        patch_meta_dict,
        key=f"{prefix}_available_actions",
    )
    available_actions: list[str] = []
    if isinstance(available_raw, (list, tuple)):
        for item in available_raw:
            action = str(item).strip()
            if action in valid_actions and action not in available_actions:
                available_actions.append(action)
    dm_probs_raw = _metadata_with_fallback(
        attempt_meta_dict,
        patch_meta_dict,
        key=f"{prefix}_dm_action_probs",
    )
    return {
        "available_actions": available_actions,
        "dm_status": str(
            _metadata_with_fallback(attempt_meta_dict, patch_meta_dict, key=f"{prefix}_dm_status")
            or ""
        ).strip(),
        "dm_action_probs": _coerce_feature_values(dm_probs_raw),
        "dm_atomic_action": str(
            _metadata_with_fallback(
                attempt_meta_dict,
                patch_meta_dict,
                key=f"{prefix}_dm_atomic_action",
            )
            or ""
        ).strip(),
        "dm_traceback_sufficient": _as_float_or_none(
            _metadata_with_fallback(
                attempt_meta_dict,
                patch_meta_dict,
                key=f"{prefix}_dm_traceback_sufficient",
            )
        ),
        "dm_missing_information": _coerce_feature_values(
            _metadata_with_fallback(
                attempt_meta_dict,
                patch_meta_dict,
                key=f"{prefix}_dm_missing_information",
            )
        ),
    }


def _adaptive_ope_summary_from_records(
    *,
    records: list[dict[str, Any]],
    prefix: str,
    action_space: set[str],
    target_policies: dict[str, str],
    min_action_support: int,
    propensity_floor: float,
) -> dict[str, Any]:
    """Build an OPE summary directly from raw attempt records."""
    samples = _adaptive_ope_samples(
        records=records,
        prefix=prefix,
        valid_actions=action_space,
    )
    return _adaptive_ope_summary(
        samples=samples,
        action_space=action_space,
        target_policies=target_policies,
        min_action_support=min_action_support,
        propensity_floor=propensity_floor,
    )


def _adaptive_ope_summary(
    *,
    samples: list[dict[str, Any]],
    action_space: set[str],
    target_policies: dict[str, str],
    min_action_support: int,
    propensity_floor: float,
) -> dict[str, Any]:
    """Build an off-policy evaluation summary from pre-extracted samples.

    Returns sample counts, action counts, and per-target-policy IPS /
    SNIPS / DR estimates. Samples without a logged propensity are dropped and counted
    in ``propensity_missing_dropped`` (STA-12).
    """
    propensity_missing_dropped = sum(1 for sample in samples if sample.get("propensity_missing"))
    samples = [sample for sample in samples if not sample.get("propensity_missing")]
    action_counts = _action_counts_for_samples(samples=samples, action_space=action_space)
    support_threshold = max(1, int(min_action_support))
    trimmed_propensity_floor = _sanitize_propensity(
        propensity_floor,
        floor=ADAPTIVE_OPE_RAW_PROPENSITY_FLOOR,
    )
    raw_propensities = [
        _sanitize_propensity(sample.get("propensity"), floor=ADAPTIVE_OPE_RAW_PROPENSITY_FLOOR)
        for sample in samples
    ]
    used_propensities = [
        _sanitize_propensity(propensity, floor=trimmed_propensity_floor)
        for propensity in raw_propensities
    ]
    feature_scaling_profiles = list(_samples_by_feature_scaling_profile(samples))
    action_rates = _action_rates_for_counts(
        action_counts=action_counts,
        total_samples=len(samples),
    )
    targets: dict[str, Any] = {}
    for target_name, target_action in target_policies.items():
        targets[target_name] = _deterministic_target_ope(
            samples=samples,
            target_action=target_action,
            action_space=action_space,
            action_counts=action_counts,
            min_action_support=support_threshold,
            ridge_alpha=ADAPTIVE_OPE_RIDGE_ALPHA,
            propensity_floor=trimmed_propensity_floor,
        )
    targets.update(
        _decision_model_targets(
            samples=samples,
            settings=StochasticTargetOpeSettings(
                action_space=action_space,
                min_action_support=support_threshold,
                ridge_alpha=ADAPTIVE_OPE_RIDGE_ALPHA,
                propensity_floor=trimmed_propensity_floor,
            ),
        )
    )
    return {
        "sample_count": len(samples),
        "action_counts": action_counts,
        "action_rates": action_rates,
        "action_entropy": _shannon_entropy_from_counts(action_counts),
        "min_action_support": support_threshold,
        "propensity_floor": trimmed_propensity_floor,
        "min_propensity_raw": min(raw_propensities, default=None),
        "min_propensity_used": min(used_propensities, default=None),
        "trimmed_sample_count": sum(
            1
            for raw_propensity, used_propensity in zip(
                raw_propensities, used_propensities, strict=True
            )
            if used_propensity > raw_propensity
        ),
        "max_importance_weight": (
            max((1.0 / propensity) for propensity in used_propensities)
            if used_propensities
            else None
        ),
        "ridge_alpha_fallback": ADAPTIVE_OPE_RIDGE_ALPHA,
        "ridge_alphas_selected": _crossfit_alpha_summary(
            samples=samples, action_space=action_space, min_action_support=support_threshold
        ),
        "dr_crossfit_folds": ADAPTIVE_OPE_CROSSFIT_FOLDS,
        "feature_scaling_profiles": feature_scaling_profiles,
        "mean_reward": _mean([float(sample.get("reward") or 0.0) for sample in samples]),
        "propensity_missing_dropped": propensity_missing_dropped,
        "shadow_corrected_samples": sum(
            1 for sample in samples if bool(sample.get("propensity_shadow_corrected"))
        ),
        "decision_model_sample_count": sum(
            1 for sample in samples if _decision_model_advice_ok(sample)
        ),
        "target_order": list(targets),
        "targets": targets,
    }


def _crossfit_alpha_summary(
    *, samples: list[dict[str, Any]], action_space: set[str], min_action_support: int
) -> dict[str, list[float]]:
    """Distinct ridge alphas the cross-fitted DR models used, per action (STA-11).

    The summary used to report the configured fallback alpha although ``RidgeCV`` picks
    one per action, stratum and fold.
    """
    selected: dict[str, set[float]] = defaultdict(set)
    for bucket_samples in _samples_by_feature_scaling_profile(samples).values():
        crossfit = _crossfit_reward_models(
            samples=bucket_samples,
            action_space=action_space,
            min_action_support=min_action_support,
            ridge_alpha=ADAPTIVE_OPE_RIDGE_ALPHA,
        )
        for action, alphas in crossfit.alphas.items():
            selected[action].update(alphas)
    return {action: sorted(values) for action, values in sorted(selected.items())}
