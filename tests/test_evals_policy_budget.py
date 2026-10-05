"""Tests for eval_policy_budget — budget behavior, finalization, sync helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pytest

from evals.adaptive_budget_online_policy import (
    ACTION_INCREASE_BUDGET,
    ACTION_KEEP_BUDGET,
    BUDGET_POLICY_ACTIONS,
    BUDGET_POLICY_ARCH_TIER_BINARY,
    BudgetPolicyDecision,
)
from evals.eval_policy_budget import (
    AdaptiveBudgetMetadataUpdate,
    _adaptive_budget_behavior_fields,
    _escalation_accounting,
    _escalation_cost_share,
    _finalize_adaptive_budget_policy_attempt,
    _nonnegative_int,
    _sync_adaptive_budget_policy_metadata,
    _sync_finalize_progress_kwargs_from_attempt_metadata,
)
from evals.eval_types import (
    ADAPTIVE_BUDGET_REWARD_SCOPE,
    EMPTY_ADAPTIVE_POLICY_LOCAL_ACCOUNTING,
    AdaptivePolicyLocalAccounting,
)

# ── _nonnegative_int ─────────────────────────────────────────────────────


class TestNonnegativeInt:
    """Lines 27-29: coercion with TypeError / ValueError fallback."""

    def test_positive_int(self) -> None:
        assert _nonnegative_int(5) == 5

    def test_zero(self) -> None:
        assert _nonnegative_int(0) == 0

    def test_negative_clamps(self) -> None:
        assert _nonnegative_int(-3) == 0

    def test_none_uses_default(self) -> None:
        assert _nonnegative_int(None, default=7) == 7

    def test_bad_string_uses_default(self) -> None:
        assert _nonnegative_int("abc", default=4) == 4

    def test_negative_default_passes_through(self) -> None:
        # The shared evals.coerce helper only clamps coerced values, not the
        # fallback default, which is returned verbatim.
        assert _nonnegative_int(None, default=-5) == -5

    def test_numeric_string(self) -> None:
        assert _nonnegative_int("10") == 10

    def test_float_truncates(self) -> None:
        assert _nonnegative_int(3.9) == 3


# ── Helpers ──────────────────────────────────────────────────────────────


def _make_budget_decision(**overrides: Any) -> BudgetPolicyDecision:
    """Build a minimal BudgetPolicyDecision for testing."""
    defaults: dict[str, Any] = {
        "decision_id": "bd-1",
        "mode": "on",
        "algorithm": "linucb",
        "architecture": BUDGET_POLICY_ARCH_TIER_BINARY,
        "decision_index": 0,
        "created_at_utc": "2024-01-01T00:00:00Z",
        "selected_action": ACTION_KEEP_BUDGET,
        "applied_action": ACTION_KEEP_BUDGET,
        "default_action": ACTION_KEEP_BUDGET,
        "propensity": 0.7,
        "decision_strategy": "ucb",
        "scores": dict.fromkeys(BUDGET_POLICY_ACTIONS, 0.0),
        "feature_values": {"bias": 1.0},
        "feature_scaling_caps": {},
        "feature_scaling_history_sizes": {},
        "feature_scaling_fallback_to_v1": False,
        "selected_family": "",
        "applied_family": "",
        "available_families": (),
        "unavailable_family_reasons": {},
        "family_propensity": 1.0,
        "family_decision_strategy": "",
        "family_scores": {},
        "family_feature_values": {},
        "family_feature_scaling_caps": {},
        "family_feature_scaling_history_sizes": {},
        "family_feature_scaling_fallback_to_v1": False,
        "configured_alpha": 0.0,
        "configured_epsilon": 0.0,
        "configured_decay": 1.0,
        "effective_epsilon": 0.0,
        "warmup_active": False,
    }
    defaults.update(overrides)
    return BudgetPolicyDecision(**defaults)


# ── _adaptive_budget_behavior_fields ─────────────────────────────────────


class TestBudgetBehaviorFields:
    def test_none_decision(self) -> None:
        action, propensity = _adaptive_budget_behavior_fields(decision=None)
        assert action == "none"
        assert propensity == 1.0

    def test_applied_action_in_policy_actions(self) -> None:
        decision = _make_budget_decision(
            selected_action=ACTION_KEEP_BUDGET,
            applied_action=ACTION_KEEP_BUDGET,
            propensity=0.7,
        )
        action, propensity = _adaptive_budget_behavior_fields(decision=decision)
        assert action == ACTION_KEEP_BUDGET
        assert propensity == pytest.approx(0.7)

    def test_shadow_mode_behavior_is_deterministic(self) -> None:
        decision = _make_budget_decision(
            mode="shadow",
            selected_action=ACTION_KEEP_BUDGET,
            applied_action=ACTION_KEEP_BUDGET,
            propensity=0.2,
        )
        action, propensity = _adaptive_budget_behavior_fields(decision=decision)
        assert action == ACTION_KEEP_BUDGET
        assert propensity == 1.0

    def test_applied_action_not_in_actions(self) -> None:
        decision = _make_budget_decision(
            selected_action="keep_budget",
            applied_action="nonsense",
            propensity=0.5,
        )
        action, propensity = _adaptive_budget_behavior_fields(decision=decision)
        assert action == "none"
        assert propensity == 1.0

    def test_family_propensity_multiplied(self) -> None:
        decision = _make_budget_decision(
            selected_action=ACTION_INCREASE_BUDGET,
            applied_action=ACTION_INCREASE_BUDGET,
            selected_family="time",
            applied_family="time",
            propensity=0.5,
            family_propensity=0.6,
        )
        action, propensity = _adaptive_budget_behavior_fields(decision=decision)
        assert action == ACTION_INCREASE_BUDGET
        assert propensity == pytest.approx(0.5 * 0.6)

    def test_family_propensity_not_multiplied_when_different(self) -> None:
        decision = _make_budget_decision(
            selected_action=ACTION_INCREASE_BUDGET,
            applied_action=ACTION_INCREASE_BUDGET,
            selected_family="time",
            applied_family="tokens",
            propensity=0.5,
            family_propensity=0.6,
        )
        action, propensity = _adaptive_budget_behavior_fields(decision=decision)
        assert action == ACTION_INCREASE_BUDGET
        assert propensity == pytest.approx(0.5)

    def test_family_propensity_empty_family(self) -> None:
        decision = _make_budget_decision(
            selected_action=ACTION_INCREASE_BUDGET,
            applied_action=ACTION_INCREASE_BUDGET,
            selected_family="",
            applied_family="",
            propensity=0.8,
            family_propensity=0.9,
        )
        action, propensity = _adaptive_budget_behavior_fields(decision=decision)
        assert action == ACTION_INCREASE_BUDGET
        # applied_family is "" → str("") is falsy → no family multiply
        assert propensity == pytest.approx(0.8)


# ── _sync_adaptive_budget_policy_metadata ────────────────────────────────


class TestSyncBudgetMetadata:
    def test_metadata_populated_with_none_decision(self) -> None:
        attempt_metadata: dict[str, Any] = {}
        patch_meta: dict[str, Any] = {}
        _sync_adaptive_budget_policy_metadata(
            update=AdaptiveBudgetMetadataUpdate(
                attempt_metadata=attempt_metadata,
                patch_meta=patch_meta,
                mode="on",
                algo="linucb",
                architecture=BUDGET_POLICY_ARCH_TIER_BINARY,
                decision=None,
                eligible=True,
                override_reason="",
                feature_scaling_profile="v1",
                feature_values_raw={},
                feature_values_scaled={},
                tiers=[1.0, 1.5],
                tier_index_before=0,
                tier_index_after=0,
                selected_multiplier=1.0,
                applied_multiplier=1.0,
                default_multiplier=1.0,
                default_controls={},
                selected_controls={},
                applied_controls={},
                selected_controls_changed={},
                applied_controls_changed={},
                partial_override=False,
                partial_override_errors={},
                max_ups_per_case=3,
                ups_used=0,
                reward_lambda_tokens=0.0,
                reward_mu_latency=0.0,
                reward_token_scale=1.0,
                reward_latency_scale=1.0,
                reward_rho_budget_exceeded=0.0,
                reward_eta_progress=0.0,
                reward_kappa_stagnation=0.0,
            )
        )
        assert attempt_metadata["adaptive_budget_policy_mode"] == "on"
        assert attempt_metadata["adaptive_budget_action_selected"] == "none"
        assert attempt_metadata["adaptive_budget_action_applied"] == "none"
        assert patch_meta["adaptive_budget_policy_mode"] == "on"

    def test_metadata_populated_with_decision(self) -> None:
        decision = _make_budget_decision(
            selected_action=ACTION_INCREASE_BUDGET,
            applied_action=ACTION_INCREASE_BUDGET,
        )
        attempt_metadata: dict[str, Any] = {}
        patch_meta: dict[str, Any] = {}
        _sync_adaptive_budget_policy_metadata(
            update=AdaptiveBudgetMetadataUpdate(
                attempt_metadata=attempt_metadata,
                patch_meta=patch_meta,
                mode="on",
                algo="linucb",
                architecture=BUDGET_POLICY_ARCH_TIER_BINARY,
                decision=decision,
                eligible=True,
                override_reason="",
                feature_scaling_profile="v1",
                feature_values_raw={"bias": 1.0},
                feature_values_scaled={"bias": 1.0},
                tiers=[1.0, 1.5],
                tier_index_before=0,
                tier_index_after=1,
                selected_multiplier=1.5,
                applied_multiplier=1.5,
                default_multiplier=1.0,
                default_controls={"helper_max_calls": 5},
                selected_controls={"helper_max_calls": 6},
                applied_controls={"helper_max_calls": 6},
                selected_controls_changed={"helper_allowance": True},
                applied_controls_changed={"helper_allowance": True},
                partial_override=False,
                partial_override_errors={},
                max_ups_per_case=3,
                ups_used=1,
                reward_lambda_tokens=0.05,
                reward_mu_latency=0.05,
                reward_token_scale=1000.0,
                reward_latency_scale=60.0,
                reward_rho_budget_exceeded=0.3,
                reward_eta_progress=0.5,
                reward_kappa_stagnation=0.5,
            )
        )
        assert attempt_metadata["adaptive_budget_action_selected"] == ACTION_INCREASE_BUDGET
        assert attempt_metadata["adaptive_budget_tier_index_after"] == 1


# ── _finalize_adaptive_budget_policy_attempt — update_reason branches ────


def _finalize_budget_kwargs(**overrides: Any) -> dict[str, Any]:
    """Minimal kwargs for _finalize_adaptive_budget_policy_attempt."""
    defaults: dict[str, Any] = {
        "note": "test",
        "success": False,
        "attempt_wall_sec": 1.0,
        "is_adaptive_condition": True,
        "budget_policy_decision": None,
        "adaptive_budget_policy_controller": None,
        "run_id": "r",
        "case_id": "c",
        "condition_value": "adaptive",
        "attempt": 1,
        "attempt_metadata": {},
        "patch_meta": {},
        "budget_policy_mode_for_attempt": "on",
        "budget_policy_algo_for_attempt": "linucb",
        "budget_policy_eligible": True,
        "budget_policy_override_reason": "",
        "budget_policy_feature_scaling_profile": "v1",
        "budget_policy_features_raw": {},
        "budget_policy_features_scaled": {},
        "normalized_adaptive_budget_policy_mode": "on",
        "adaptive_progress_made": False,
        "adaptive_progress_score": 0.0,
        "adaptive_no_progress_streak_current": 0,
        "reward_lambda_tokens": 0.0,
        "reward_mu_latency": 0.0,
        "reward_token_scale": 1.0,
        "reward_latency_scale": 1.0,
        "reward_eta_progress": 0.0,
        "reward_kappa_stagnation": 0.0,
        "reward_rho_budget_exceeded": 0.0,
    }
    defaults.update(overrides)
    return defaults


class TestFinalizeBudgetAttemptUpdateReasons:
    """Cover lines 398-408: update_reason dispatch branches."""

    def test_not_eligible_reason(self) -> None:
        meta: dict[str, Any] = {}
        patch: dict[str, Any] = {}
        _finalize_adaptive_budget_policy_attempt(
            **_finalize_budget_kwargs(
                budget_policy_decision=None,
                budget_policy_eligible=False,
                attempt_metadata=meta,
                patch_meta=patch,
            )
        )
        assert meta["adaptive_budget_update_reason"] == "not_eligible"

    def test_policy_off_reason(self) -> None:
        meta: dict[str, Any] = {}
        patch: dict[str, Any] = {}
        _finalize_adaptive_budget_policy_attempt(
            **_finalize_budget_kwargs(
                budget_policy_decision=None,
                budget_policy_eligible=True,
                normalized_adaptive_budget_policy_mode="off",
                attempt_metadata=meta,
                patch_meta=patch,
            )
        )
        assert meta["adaptive_budget_update_reason"] == "policy_off"

    def test_no_decision_emitted_reason(self) -> None:
        meta: dict[str, Any] = {}
        patch: dict[str, Any] = {}
        _finalize_adaptive_budget_policy_attempt(
            **_finalize_budget_kwargs(
                budget_policy_decision=None,
                budget_policy_eligible=True,
                normalized_adaptive_budget_policy_mode="on",
                attempt_metadata=meta,
                patch_meta=patch,
            )
        )
        assert meta["adaptive_budget_update_reason"] == "no_decision_emitted"

    def test_controller_unavailable_reason(self) -> None:
        """decision present but controller is None → policy_controller_unavailable."""
        decision = _make_budget_decision()
        meta: dict[str, Any] = {}
        patch: dict[str, Any] = {}
        _finalize_adaptive_budget_policy_attempt(
            **_finalize_budget_kwargs(
                budget_policy_decision=decision,
                adaptive_budget_policy_controller=None,
                attempt_metadata=meta,
                patch_meta=patch,
            )
        )
        assert meta["adaptive_budget_update_reason"] == "policy_controller_unavailable"

    def test_non_adaptive_condition(self) -> None:
        meta: dict[str, Any] = {}
        patch: dict[str, Any] = {}
        _finalize_adaptive_budget_policy_attempt(
            **_finalize_budget_kwargs(
                is_adaptive_condition=False,
                attempt_metadata=meta,
                patch_meta=patch,
            )
        )
        assert meta["adaptive_budget_update_reason"] == "non_adaptive_condition"

    def test_request_budget_exceeded_penalty(self) -> None:
        meta: dict[str, Any] = {"patcher_request_failure_error": "request_budget_exceeded"}
        patch: dict[str, Any] = {}
        _finalize_adaptive_budget_policy_attempt(
            **_finalize_budget_kwargs(
                is_adaptive_condition=False,
                attempt_metadata=meta,
                patch_meta=patch,
                reward_rho_budget_exceeded=0.5,
            )
        )
        assert meta["adaptive_budget_request_budget_exceeded"] is True

    def test_event_logging_error_on_controller_exception(self) -> None:
        """Lines 398-399: controller.record_outcome raises → error reason."""
        decision = _make_budget_decision()

        @dataclass
        class _FakeConfig:
            decay: float = 1.0
            architecture: str = BUDGET_POLICY_ARCH_TIER_BINARY

        class _FakeController:
            config = _FakeConfig()

            def record_outcome(self, **kwargs: Any) -> None:
                raise RuntimeError("boom")

        meta: dict[str, Any] = {}
        patch: dict[str, Any] = {}
        _finalize_adaptive_budget_policy_attempt(
            **_finalize_budget_kwargs(
                budget_policy_decision=decision,
                adaptive_budget_policy_controller=_FakeController(),
                attempt_metadata=meta,
                patch_meta=patch,
            )
        )
        assert meta["adaptive_budget_update_reason"].startswith("event_logging_error:")


# ── POL-08: escalation is charged only its incremental cost ─────────────


class _CapturingController:
    @dataclass
    class _Config:
        decay: float = 1.0
        architecture: str = BUDGET_POLICY_ARCH_TIER_BINARY

    config = _Config()

    def __init__(self) -> None:
        self.calls: list[dict[str, Any]] = []

    def record_outcome(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        raise RuntimeError("captured")


@pytest.mark.parametrize(
    ("action", "applied_multiplier", "expected_penalty"),
    [
        # 5k tokens at lambda 0.08 / scale 1k used to cost the escalation 0.40.
        (ACTION_INCREASE_BUDGET, 1.25, 0.08 * 5.0 * 0.2),
        (ACTION_INCREASE_BUDGET, 1.0, 0.0),
        (ACTION_KEEP_BUDGET, 1.0, 0.0),
    ],
)
def test_budget_reward_charges_only_the_escalation_share(
    action: str, applied_multiplier: float, expected_penalty: float
) -> None:
    controller = _CapturingController()
    _finalize_adaptive_budget_policy_attempt(
        **_finalize_budget_kwargs(
            budget_policy_decision=_make_budget_decision(
                selected_action=action, applied_action=action
            ),
            adaptive_budget_policy_controller=controller,
            reward_lambda_tokens=0.08,
            reward_token_scale=1000.0,
            local_accounting=AdaptivePolicyLocalAccounting(
                llm_call_count=1, tokens_prompt=4000, tokens_completion=1000, tokens_total=5000
            ),
            applied_multiplier=applied_multiplier,
            default_multiplier=1.0,
        )
    )
    (call,) = controller.calls
    assert call["reward_token_penalty"] == pytest.approx(expected_penalty)
    # The event still logs the attempt's full cost.
    assert call["local_tokens_total"] == 5000


def test_escalation_cost_share_and_accounting() -> None:
    assert _escalation_cost_share(applied_multiplier=1.5, default_multiplier=1.0) == (
        pytest.approx(1.0 / 3.0)
    )
    assert _escalation_cost_share(applied_multiplier=1.0, default_multiplier=1.0) == 0.0
    assert _escalation_cost_share(applied_multiplier=0.8, default_multiplier=1.0) == 0.0
    assert _escalation_cost_share(applied_multiplier=1.5, default_multiplier=0.0) == 0.0
    scaled = _escalation_accounting(
        AdaptivePolicyLocalAccounting(
            llm_call_count=2,
            tokens_prompt=300,
            tokens_completion=100,
            tokens_total=400,
            wall_sec=8.0,
        ),
        share=0.25,
    )
    assert scaled == AdaptivePolicyLocalAccounting(
        llm_call_count=2, tokens_prompt=75, tokens_completion=25, tokens_total=100, wall_sec=2.0
    )


# ── _sync_finalize_progress_kwargs_from_attempt_metadata ─────────────────


class TestSyncFinalizeProgressKwargs:
    """Cover lines 480-513."""

    def test_basic_sync(self) -> None:
        online_kw: dict[str, Any] = {
            "adaptive_no_progress_streak_current": 2,
            "adaptive_progress_made": True,
            "adaptive_progress_score": 0.7,
            "attempt_metadata": {},
        }
        budget_kw: dict[str, Any] = {}
        _sync_finalize_progress_kwargs_from_attempt_metadata(
            online_finalize_kwargs=online_kw,
            budget_finalize_kwargs=budget_kw,
        )
        assert online_kw["adaptive_progress_made"] is True
        assert online_kw["adaptive_progress_score"] == pytest.approx(0.7)
        assert online_kw["adaptive_no_progress_streak_current"] == 2
        assert budget_kw["adaptive_progress_made"] is True
        assert budget_kw["adaptive_progress_score"] == pytest.approx(0.7)

    def test_attempt_metadata_overrides(self) -> None:
        online_kw: dict[str, Any] = {
            "adaptive_no_progress_streak_current": 0,
            "adaptive_progress_made": False,
            "adaptive_progress_score": 0.0,
            "attempt_metadata": {
                "adaptive_progress_made": True,
                "adaptive_progress_score": 0.9,
                "adaptive_no_progress_streak": 3,
            },
        }
        budget_kw: dict[str, Any] = {}
        _sync_finalize_progress_kwargs_from_attempt_metadata(
            online_finalize_kwargs=online_kw,
            budget_finalize_kwargs=budget_kw,
        )
        assert online_kw["adaptive_progress_made"] is True
        assert online_kw["adaptive_progress_score"] == pytest.approx(0.9)
        assert online_kw["adaptive_no_progress_streak_current"] == 3

    def test_attempt_metadata_invalid_score_ignored(self) -> None:
        """Lines 502-503: TypeError / ValueError in score parsing is swallowed."""
        online_kw: dict[str, Any] = {
            "adaptive_no_progress_streak_current": 0,
            "adaptive_progress_made": False,
            "adaptive_progress_score": 0.5,
            "attempt_metadata": {
                "adaptive_progress_score": "not_a_number",
            },
        }
        budget_kw: dict[str, Any] = {}
        _sync_finalize_progress_kwargs_from_attempt_metadata(
            online_finalize_kwargs=online_kw,
            budget_finalize_kwargs=budget_kw,
        )
        # Falls back to original value
        assert online_kw["adaptive_progress_score"] == pytest.approx(0.5)

    def test_no_attempt_metadata(self) -> None:
        online_kw: dict[str, Any] = {
            "adaptive_no_progress_streak_current": 1,
            "adaptive_progress_made": True,
            "adaptive_progress_score": 0.3,
        }
        budget_kw: dict[str, Any] = {}
        _sync_finalize_progress_kwargs_from_attempt_metadata(
            online_finalize_kwargs=online_kw,
            budget_finalize_kwargs=budget_kw,
        )
        assert online_kw["adaptive_progress_made"] is True
        assert budget_kw["adaptive_progress_score"] == pytest.approx(0.3)

    def test_charged_accounting_keep_budget_zeroes(self) -> None:
        """When behavior_action != increase_budget, local_accounting is zeroed."""
        meta: dict[str, Any] = {}
        patch: dict[str, Any] = {}
        _finalize_adaptive_budget_policy_attempt(
            **_finalize_budget_kwargs(
                is_adaptive_condition=False,
                attempt_metadata=meta,
                patch_meta=patch,
                local_accounting=AdaptivePolicyLocalAccounting(
                    llm_call_count=5,
                    tokens_prompt=100,
                    tokens_completion=50,
                    tokens_total=150,
                    wall_sec=10.0,
                ),
            )
        )
        # reward is computed with zeroed accounting (since action != increase_budget)
        # We just verify the call doesn't error and metadata is populated
        assert "adaptive_budget_reward" in meta
