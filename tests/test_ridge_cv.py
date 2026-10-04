"""Tests for RidgeCV alpha selection in analyze_results OPE helpers."""

from __future__ import annotations

import numpy as np
import pytest

from evals.analyze_results import (
    _RIDGECV_MIN_SAMPLES,
    ADAPTIVE_OPE_RIDGE_ALPHA,
    ADAPTIVE_OPE_RIDGE_ALPHAS,
    _fit_ridge_action_models,
    _predict_action_reward,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_samples(
    *,
    n: int,
    action: str,
    feature_names: list[str],
    rng: np.random.Generator,
    noise_scale: float = 0.1,
    true_weights: np.ndarray | None = None,
) -> list[dict]:
    """Generate synthetic OPE samples for a single action.

    reward = bias + X @ true_weights + noise
    """
    dim = len(feature_names)
    if true_weights is None:
        true_weights = rng.standard_normal(dim)
    bias = 1.0
    samples: list[dict] = []
    for _ in range(n):
        x = rng.standard_normal(dim)
        reward = bias + float(x @ true_weights) + rng.normal(0, noise_scale)
        features = {name: float(x[i]) for i, name in enumerate(feature_names)}
        samples.append(
            {
                "action": action,
                "reward": reward,
                "propensity": 0.5,
                "features": features,
            }
        )
    return samples


def _fit_with_hardcoded_alpha(
    samples: list[dict],
    action_space: set[str],
    min_action_support: int = 2,
) -> tuple[list[str], dict[str, np.ndarray]]:
    """Reproduce the old closed-form ridge with hardcoded alpha (for regression tests)."""
    feature_names = sorted(
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
    dim = len(feature_names) + 1
    models: dict[str, np.ndarray] = {}
    for action in sorted(action_space):
        action_samples = [s for s in samples if str(s.get("action") or "") == action]
        if len(action_samples) < min_action_support:
            continue
        X = np.ones((len(action_samples), dim), dtype=float)
        y = np.zeros(len(action_samples), dtype=float)
        for row_idx, sample in enumerate(action_samples):
            y[row_idx] = float(sample.get("reward") or 0.0)
            features = sample.get("features")
            feature_dict = features if isinstance(features, dict) else {}
            for col_idx, name in enumerate(feature_names, start=1):
                X[row_idx, col_idx] = float(feature_dict.get(name, 0.0) or 0.0)
        gram = X.T @ X + np.eye(dim) * ADAPTIVE_OPE_RIDGE_ALPHA
        rhs = X.T @ y
        try:
            beta = np.linalg.solve(gram, rhs)
        except np.linalg.LinAlgError:
            beta = np.linalg.pinv(gram) @ rhs
        models[action] = beta
    return feature_names, models


# ---------------------------------------------------------------------------
# Tests: RidgeCV selects a reasonable alpha on synthetic data
# ---------------------------------------------------------------------------


class TestRidgeCVSelectsReasonableAlpha:
    """RidgeCV should pick an alpha from the grid when given enough data."""

    def test_selected_alpha_in_grid(self) -> None:
        """The fit path should run without error and produce a model."""
        rng = np.random.default_rng(42)
        feature_names = ["f1", "f2", "f3"]
        n = 50  # well above _RIDGECV_MIN_SAMPLES
        samples = _make_samples(
            n=n,
            action="act_a",
            feature_names=feature_names,
            rng=rng,
        )
        _, models = _fit_ridge_action_models(
            samples=samples,
            action_space={"act_a"},
            min_action_support=2,
            ridge_alpha=ADAPTIVE_OPE_RIDGE_ALPHA,
        )
        assert "act_a" in models
        beta = models["act_a"]
        assert beta.shape == (len(feature_names) + 1,)

    def test_prediction_quality_with_cv(self) -> None:
        """CV-fitted model should predict well on clean linear data."""
        rng = np.random.default_rng(99)
        feature_names = ["x1", "x2"]
        true_weights = np.array([3.0, -2.0])
        samples = _make_samples(
            n=60,
            action="act_a",
            feature_names=feature_names,
            rng=rng,
            true_weights=true_weights,
            noise_scale=0.05,
        )
        _feature_names_out, models = _fit_ridge_action_models(
            samples=samples,
            action_space={"act_a"},
            min_action_support=2,
            ridge_alpha=ADAPTIVE_OPE_RIDGE_ALPHA,
        )
        beta = models["act_a"]
        # The bias is at index 0, features at 1..
        # With low noise the recovered weights should be close to truth.
        assert abs(beta[0] - 1.0) < 0.3, f"bias should be ~1.0, got {beta[0]}"
        assert abs(beta[1] - 3.0) < 0.5, f"w1 should be ~3.0, got {beta[1]}"
        assert abs(beta[2] - (-2.0)) < 0.5, f"w2 should be ~-2.0, got {beta[2]}"

    def test_multi_action_cv(self) -> None:
        """Multiple actions each get their own CV-selected model."""
        rng = np.random.default_rng(7)
        feature_names = ["f1"]
        samples_a = _make_samples(
            n=20,
            action="a",
            feature_names=feature_names,
            rng=rng,
        )
        samples_b = _make_samples(
            n=20,
            action="b",
            feature_names=feature_names,
            rng=rng,
        )
        all_samples = samples_a + samples_b
        _, models = _fit_ridge_action_models(
            samples=all_samples,
            action_space={"a", "b"},
            min_action_support=2,
            ridge_alpha=ADAPTIVE_OPE_RIDGE_ALPHA,
        )
        assert "a" in models
        assert "b" in models


# ---------------------------------------------------------------------------
# Tests: Fallback to hardcoded alpha with small datasets
# ---------------------------------------------------------------------------


class TestFallbackToHardcodedAlpha:
    """With fewer than _RIDGECV_MIN_SAMPLES the closed-form fallback is used."""

    def test_small_dataset_uses_fallback(self) -> None:
        """With n < _RIDGECV_MIN_SAMPLES, model should still be fitted."""
        rng = np.random.default_rng(0)
        feature_names = ["f1"]
        n = _RIDGECV_MIN_SAMPLES - 1  # just below threshold
        assert n < _RIDGECV_MIN_SAMPLES
        samples = _make_samples(
            n=n,
            action="act_a",
            feature_names=feature_names,
            rng=rng,
        )
        _, models = _fit_ridge_action_models(
            samples=samples,
            action_space={"act_a"},
            min_action_support=2,
            ridge_alpha=ADAPTIVE_OPE_RIDGE_ALPHA,
        )
        assert "act_a" in models
        assert models["act_a"].shape == (len(feature_names) + 1,)

    def test_fallback_matches_original_closed_form(self) -> None:
        """Small-data fallback should produce identical results to the original code."""
        rng = np.random.default_rng(12)
        feature_names = ["f1", "f2"]
        n = _RIDGECV_MIN_SAMPLES - 1
        samples = _make_samples(
            n=n,
            action="act_a",
            feature_names=feature_names,
            rng=rng,
        )
        _, models_new = _fit_ridge_action_models(
            samples=samples,
            action_space={"act_a"},
            min_action_support=2,
            ridge_alpha=ADAPTIVE_OPE_RIDGE_ALPHA,
        )
        _, models_old = _fit_with_hardcoded_alpha(
            samples=samples,
            action_space={"act_a"},
            min_action_support=2,
        )
        np.testing.assert_allclose(
            models_new["act_a"],
            models_old["act_a"],
            rtol=1e-10,
            err_msg="Fallback path should be identical to the original closed-form",
        )

    def test_boundary_exactly_at_threshold(self) -> None:
        """At exactly _RIDGECV_MIN_SAMPLES, RidgeCV should be used (no error)."""
        rng = np.random.default_rng(55)
        feature_names = ["f1"]
        n = _RIDGECV_MIN_SAMPLES  # exactly at threshold
        samples = _make_samples(
            n=n,
            action="act_a",
            feature_names=feature_names,
            rng=rng,
        )
        _, models = _fit_ridge_action_models(
            samples=samples,
            action_space={"act_a"},
            min_action_support=2,
            ridge_alpha=ADAPTIVE_OPE_RIDGE_ALPHA,
        )
        assert "act_a" in models

    def test_min_action_support_skips_action(self) -> None:
        """Actions below min_action_support are omitted from models."""
        rng = np.random.default_rng(1)
        feature_names = ["f1"]
        samples = _make_samples(
            n=1,
            action="rare",
            feature_names=feature_names,
            rng=rng,
        )
        _, models = _fit_ridge_action_models(
            samples=samples,
            action_space={"rare"},
            min_action_support=5,
            ridge_alpha=ADAPTIVE_OPE_RIDGE_ALPHA,
        )
        assert "rare" not in models


# ---------------------------------------------------------------------------
# Tests: Regression - CV results close to original on moderate data
# ---------------------------------------------------------------------------


class TestRegressionNumericalCloseness:
    """CV-selected alpha should yield predictions close to the hardcoded model."""

    def test_predictions_close_on_clean_data(self) -> None:
        """On well-conditioned data the CV and original predictions should agree."""
        rng = np.random.default_rng(77)
        feature_names = ["f1", "f2", "f3"]
        true_w = np.array([2.0, -1.0, 0.5])
        n = 40
        samples = _make_samples(
            n=n,
            action="act_a",
            feature_names=feature_names,
            rng=rng,
            true_weights=true_w,
            noise_scale=0.01,
        )
        fnames_cv, models_cv = _fit_ridge_action_models(
            samples=samples,
            action_space={"act_a"},
            min_action_support=2,
            ridge_alpha=ADAPTIVE_OPE_RIDGE_ALPHA,
        )
        fnames_old, models_old = _fit_with_hardcoded_alpha(
            samples=samples,
            action_space={"act_a"},
            min_action_support=2,
        )
        assert fnames_cv == fnames_old
        # Generate held-out points and compare predictions.
        test_features = {"f1": 1.0, "f2": -0.5, "f3": 0.3}
        pred_cv = _predict_action_reward(
            features=test_features,
            feature_names=fnames_cv,
            beta=models_cv["act_a"],
        )
        pred_old = _predict_action_reward(
            features=test_features,
            feature_names=fnames_old,
            beta=models_old["act_a"],
        )
        # Both should be close to the true value: 1.0 + 2*1 + (-1)*(-0.5) + 0.5*0.3 = 3.65
        assert abs(pred_cv - 3.65) < 0.5, f"CV pred {pred_cv} too far from 3.65"
        assert abs(pred_old - 3.65) < 0.5, f"Old pred {pred_old} too far from 3.65"
        # And close to each other (both are good fits on clean data).
        assert abs(pred_cv - pred_old) < 0.3, (
            f"CV pred={pred_cv:.4f} vs old pred={pred_old:.4f} diverge too much"
        )

    def test_constants_unchanged(self) -> None:
        """Verify the public constants haven't drifted."""
        assert ADAPTIVE_OPE_RIDGE_ALPHA == 1e-3
        assert _RIDGECV_MIN_SAMPLES == 10
        assert isinstance(ADAPTIVE_OPE_RIDGE_ALPHAS, tuple)
        assert len(ADAPTIVE_OPE_RIDGE_ALPHAS) == 6
        assert 0.001 in ADAPTIVE_OPE_RIDGE_ALPHAS
        assert 100.0 in ADAPTIVE_OPE_RIDGE_ALPHAS


def _ope_sample(action: str, x: float, reward: float) -> dict[str, object]:
    return {
        "action": action,
        "propensity": 0.5,
        "reward": reward,
        "features": {"x": x},
        "feature_scaling_profile": "v1",
    }


def test_sta11_crossfit_models_never_see_their_own_fold() -> None:
    """STA-11: the DR reward model was fitted on the samples it then corrected."""
    from evals.analyze_results import _crossfit_reward_models, _fit_action_reward_models

    samples = [_ope_sample("a", float(i), float(i)) for i in range(10)]
    samples[3]["reward"] = 1000.0  # an outlier the in-sample fit would absorb
    crossfit = _crossfit_reward_models(
        samples=samples, action_space={"a"}, min_action_support=1, ridge_alpha=0.1
    )
    held_out = _fit_action_reward_models(
        samples=[s for idx, s in enumerate(samples) if idx % 5 != 3 % 5],
        action_space={"a"},
        min_action_support=1,
        ridge_alpha=0.1,
        feature_names=["x"],
    )
    assert np.allclose(crossfit.per_sample[3]["a"], held_out.models["a"])
    in_sample = _fit_action_reward_models(
        samples=samples, action_space={"a"}, min_action_support=1, ridge_alpha=0.1
    )
    assert not np.allclose(crossfit.per_sample[3]["a"], in_sample.models["a"])


def test_sta11_summary_reports_the_alphas_actually_used() -> None:
    from evals.analyze_results import ADAPTIVE_OPE_RIDGE_ALPHAS, _adaptive_ope_summary

    samples = [
        _ope_sample(action, float(i), float(i) * (2.0 if action == "a" else 1.0))
        for i in range(30)
        for action in ("a", "b")
    ]
    summary = _adaptive_ope_summary(
        samples=samples,
        action_space={"a", "b"},
        target_policies={"always_a": "a"},
        min_action_support=2,
        propensity_floor=0.01,
    )
    assert summary["dr_crossfit_folds"] == 5
    assert set(summary["ridge_alphas_selected"]) == {"a", "b"}
    for alphas in summary["ridge_alphas_selected"].values():
        assert set(alphas) <= set(ADAPTIVE_OPE_RIDGE_ALPHAS)  # RidgeCV choices, 24+ samples
    assert summary["targets"]["always_a"]["dr_supported"] is True
