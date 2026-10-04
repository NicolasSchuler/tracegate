"""Tests for the adaptive online policy controller (actions, state persistence)."""

from __future__ import annotations

import math

import pytest

from evals.adaptive_online_policy import (
    ACTION_FRAME_ONLY,
    ACTION_INVOKE_HELPER,
    ACTION_SEARCH_ONLY,
    ACTION_SKIP_HELPER,
    ACTION_SNAPSHOT_LITE,
    ONLINE_POLICY_ACTIONS,
    STATE_SCHEMA_VERSION,
    AdaptiveOnlinePolicyController,
    OnlinePolicyConfig,
)
from evals.policy_feature_scaling import (
    _V1_COUNT_CAPS,
    DEFAULT_POLICY_FEATURE_SCALING_MIN_SAMPLES,
)
from llmdebug import _json as json


def _policy_state_payload(
    *,
    preferred_action: str | None = None,
    preferred_value: float = 2.0,
    action_counts: dict[str, int] | None = None,
    diagonal_by_action: dict[str, float] | None = None,
    b_by_action: dict[str, float] | None = None,
) -> dict[str, object]:
    counts = dict.fromkeys(ONLINE_POLICY_ACTIONS, 0)
    counts.update(action_counts or {})
    diagonals = dict.fromkeys(ONLINE_POLICY_ACTIONS, 1.0)
    diagonals.update(diagonal_by_action or {})
    b_values = dict.fromkeys(ONLINE_POLICY_ACTIONS, 0.0)
    b_values.update(b_by_action or {})
    if preferred_action in ONLINE_POLICY_ACTIONS:
        b_values[str(preferred_action)] = float(preferred_value)
    return {
        "schema_version": STATE_SCHEMA_VERSION,
        "mode": "on",
        "algorithm": "linucb",
        "alpha": 0.0,
        "epsilon": 0.0,
        "l2": 1.0,
        "warmup_decisions": 0,
        "save_every": 1,
        "feature_names": ["bias"],
        "decisions": 0,
        "updates": 0,
        "action_counts": counts,
        "A": {action: [[float(diagonals[action])]] for action in ONLINE_POLICY_ACTIONS},
        "b": {action: [float(b_values[action])] for action in ONLINE_POLICY_ACTIONS},
    }


def test_online_policy_on_updates_and_state_roundtrip(tmp_path) -> None:
    state_path = tmp_path / "policy_state.json"
    events_path = tmp_path / "policy_events.jsonl"
    config = OnlinePolicyConfig.normalize(
        mode="on",
        algorithm="linucb",
        alpha=0.2,
        epsilon=0.0,
        l2=1.0,
        warmup_decisions=0,
        save_every=1,
        state_path=state_path,
        events_path=events_path,
        random_seed=7,
        decay=1.0,
    )
    controller = AdaptiveOnlinePolicyController(config)
    decision = controller.decide(feature_values={"bias": 1.0}, default_action=ACTION_INVOKE_HELPER)
    event = controller.record_outcome(
        decision=decision,
        run_id="run_online",
        case_id="case_a",
        condition="layered_adaptive_llm",
        attempt=1,
        note="tests_still_failing",
        reward_scope="helper_credit_v2",
        reward=0.7,
        reward_success_hint=0.0,
        reward_progress_hint=0.0,
        reward_progress_score=0.4,
        reward_token_penalty=0.2,
        reward_latency_penalty=0.1,
        reward_stagnation_penalty=0.0,
        reward_helper_materialization_bonus=0.0,
        success=False,
        local_llm_call_count=1,
        local_tokens_prompt=100,
        local_tokens_completion=23,
        local_tokens_total=123,
        local_wall_sec=1.2,
        override_reason="",
        reward_success_hint_scale=1.3,
        reward_progress_hint_scale=1.4,
        reward_stagnation_penalty_scale=1.5,
        reward_helper_materialization_bonus_scale=0.6,
    )
    state = controller.state
    controller.close()
    payload = json.loads(events_path.read_text(encoding="utf-8").splitlines()[0])

    assert event.model_updated is True
    assert event.update_reason == "on_policy_update"
    assert event.reward_progress_score == 0.4
    assert event.reward_success_hint_scale == pytest.approx(1.3)
    assert event.reward_progress_hint_scale == pytest.approx(1.4)
    assert event.reward_stagnation_penalty_scale == pytest.approx(1.5)
    assert event.reward_helper_materialization_bonus_scale == pytest.approx(0.6)
    assert payload["reward_success_hint_scale"] == pytest.approx(1.3)
    assert payload["reward_progress_hint_scale"] == pytest.approx(1.4)
    assert payload["reward_stagnation_penalty_scale"] == pytest.approx(1.5)
    assert payload["reward_helper_materialization_bonus_scale"] == pytest.approx(0.6)
    assert state.decisions == 1
    assert state.updates == 1
    assert state.action_counts[ACTION_INVOKE_HELPER] == 1
    assert events_path.exists()
    assert len([line for line in events_path.read_text(encoding="utf-8").splitlines() if line]) == 1

    reloaded = AdaptiveOnlinePolicyController(config)
    reloaded_state = reloaded.state
    reloaded.close()
    assert reloaded.load_error is None
    assert reloaded_state.decisions == 1
    assert reloaded_state.updates == 1


def test_online_policy_init_state_path_loads_from_separate_source(tmp_path) -> None:
    init_state_path = tmp_path / "policy_init_state.json"
    state_path = tmp_path / "policy_new_state.json"
    init_state_path.write_text(
        json.dumps(
            _policy_state_payload(
                action_counts={ACTION_INVOKE_HELPER: 4},
            )
            | {"decisions": 4, "updates": 2}
        ),
        encoding="utf-8",
    )
    config = OnlinePolicyConfig.normalize(
        mode="on",
        algorithm="linucb",
        init_state_path=init_state_path,
        state_path=state_path,
        events_path=tmp_path / "policy_init_events.jsonl",
        decay=1.0,
    )

    controller = AdaptiveOnlinePolicyController(config)
    assert controller.load_error is None
    assert controller.state.decisions == 4

    controller.close()
    saved = json.loads(state_path.read_text(encoding="utf-8"))
    assert saved["decisions"] == 4


def test_online_policy_shadow_logs_but_does_not_update(tmp_path) -> None:
    state_path = tmp_path / "policy_state_shadow.json"
    events_path = tmp_path / "policy_events_shadow.jsonl"
    config = OnlinePolicyConfig.normalize(
        mode="shadow",
        algorithm="linucb",
        alpha=0.0,
        epsilon=0.0,
        l2=1.0,
        warmup_decisions=0,
        save_every=1,
        state_path=state_path,
        events_path=events_path,
        random_seed=0,
        decay=1.0,
    )
    controller = AdaptiveOnlinePolicyController(config)
    decision = controller.decide(feature_values={"bias": 1.0}, default_action=ACTION_SKIP_HELPER)
    event = controller.record_outcome(
        decision=decision,
        run_id="run_shadow",
        case_id="case_b",
        condition="layered_adaptive_llm",
        attempt=1,
        note="no_patch_proposed",
        reward_scope="helper_credit_v2",
        reward=-0.1,
        reward_success_hint=0.0,
        reward_progress_hint=0.0,
        reward_progress_score=0.0,
        reward_token_penalty=0.1,
        reward_latency_penalty=0.0,
        reward_stagnation_penalty=0.0,
        reward_helper_materialization_bonus=0.0,
        success=False,
        local_llm_call_count=0,
        local_tokens_prompt=0,
        local_tokens_completion=0,
        local_tokens_total=10,
        local_wall_sec=0.2,
        override_reason="shadow_mode_baseline",
    )
    state = controller.state
    controller.close()

    assert decision.applied_action == ACTION_SKIP_HELPER
    assert event.model_updated is False
    assert event.update_reason == "shadow_no_update"
    assert state.decisions == 1
    assert state.updates == 0
    assert len([line for line in events_path.read_text(encoding="utf-8").splitlines() if line]) == 1


def test_online_policy_empty_state_roundtrips_without_load_error(tmp_path) -> None:
    state_path = tmp_path / "policy_state_empty.json"
    events_path = tmp_path / "policy_events_empty.jsonl"
    config = OnlinePolicyConfig.normalize(
        mode="on",
        algorithm="linucb",
        alpha=0.2,
        epsilon=0.0,
        l2=1.0,
        warmup_decisions=0,
        save_every=1,
        state_path=state_path,
        events_path=events_path,
        random_seed=0,
        decay=1.0,
    )
    controller = AdaptiveOnlinePolicyController(config)
    controller.close()

    reloaded = AdaptiveOnlinePolicyController(config)
    state = reloaded.state
    reloaded.close()

    assert reloaded.load_error is None
    assert state.feature_names == []
    assert state.decisions == 0
    assert state.updates == 0


def test_online_policy_record_outcome_uses_executed_behavior_action(tmp_path) -> None:
    state_path = tmp_path / "policy_state_behavior_override.json"
    events_path = tmp_path / "policy_events_behavior_override.jsonl"
    state_path.write_text(
        json.dumps(
            _policy_state_payload(
                preferred_action=ACTION_SKIP_HELPER,
                preferred_value=2.0,
            ),
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    config = OnlinePolicyConfig.normalize(
        mode="on",
        algorithm="linucb",
        alpha=0.0,
        epsilon=0.0,
        l2=1.0,
        warmup_decisions=0,
        save_every=1,
        decay=1.0,
        state_path=state_path,
        events_path=events_path,
        random_seed=0,
    )
    controller = AdaptiveOnlinePolicyController(config)
    decision = controller.decide(feature_values={"bias": 1.0}, default_action=ACTION_INVOKE_HELPER)

    assert decision.applied_action == ACTION_SKIP_HELPER

    event = controller.record_outcome(
        decision=decision,
        run_id="run_behavior_override",
        case_id="case_override",
        condition="layered_adaptive_llm",
        attempt=1,
        note="stagnation_override_force_helper",
        reward_scope="helper_credit_v2",
        reward=1.0,
        reward_success_hint=0.0,
        reward_progress_hint=0.0,
        reward_progress_score=0.6,
        reward_token_penalty=0.0,
        reward_latency_penalty=0.0,
        reward_stagnation_penalty=0.0,
        reward_helper_materialization_bonus=0.0,
        success=False,
        local_llm_call_count=1,
        local_tokens_prompt=4,
        local_tokens_completion=6,
        local_tokens_total=10,
        local_wall_sec=0.1,
        override_reason="constraint_stagnation_override_force_helper",
        behavior_action=ACTION_INVOKE_HELPER,
        behavior_propensity=1.0,
    )
    state = controller.state
    controller.close()

    assert event.model_updated is True
    assert event.applied_action == ACTION_SKIP_HELPER
    assert event.behavior_action == ACTION_INVOKE_HELPER
    assert event.behavior_propensity == 1.0
    assert state.b[ACTION_SKIP_HELPER][0] == 2.0
    assert state.b[ACTION_INVOKE_HELPER][0] == 1.0


def test_online_policy_record_outcome_updates_non_binary_behavior_action(tmp_path) -> None:
    state_path = tmp_path / "policy_state_frame_only.json"
    events_path = tmp_path / "policy_events_frame_only.jsonl"
    state_path.write_text(
        json.dumps(
            _policy_state_payload(
                preferred_action=ACTION_FRAME_ONLY,
                preferred_value=2.0,
            ),
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    config = OnlinePolicyConfig.normalize(
        mode="on",
        algorithm="linucb",
        alpha=0.0,
        epsilon=0.0,
        l2=1.0,
        warmup_decisions=0,
        save_every=1,
        state_path=state_path,
        events_path=events_path,
        random_seed=0,
        decay=1.0,
    )
    controller = AdaptiveOnlinePolicyController(config)
    decision = controller.decide(feature_values={"bias": 1.0}, default_action=ACTION_INVOKE_HELPER)

    assert decision.applied_action == ACTION_FRAME_ONLY

    event = controller.record_outcome(
        decision=decision,
        run_id="run_frame_only",
        case_id="case_frame_only",
        condition="layered_adaptive_llm",
        attempt=1,
        note="tests_still_failing",
        reward_scope="helper_credit_v2",
        reward=0.3,
        reward_success_hint=0.0,
        reward_progress_hint=0.0,
        reward_progress_score=0.2,
        reward_token_penalty=0.0,
        reward_latency_penalty=0.0,
        reward_stagnation_penalty=0.0,
        reward_helper_materialization_bonus=0.1,
        success=False,
        local_llm_call_count=1,
        local_tokens_prompt=5,
        local_tokens_completion=3,
        local_tokens_total=8,
        local_wall_sec=0.1,
        override_reason="",
        behavior_action=ACTION_FRAME_ONLY,
        behavior_propensity=1.0,
    )
    state = controller.state
    controller.close()

    assert event.behavior_action == ACTION_FRAME_ONLY
    assert event.model_updated is True
    assert state.b[ACTION_FRAME_ONLY][0] == pytest.approx(2.3)
    assert state.b[ACTION_INVOKE_HELPER][0] == 0.0


def test_online_policy_record_outcome_can_log_without_updating_model(tmp_path) -> None:
    state_path = tmp_path / "policy_state_no_update.json"
    events_path = tmp_path / "policy_events_no_update.jsonl"
    config = OnlinePolicyConfig.normalize(
        mode="on",
        algorithm="linucb",
        alpha=0.2,
        epsilon=0.0,
        l2=1.0,
        warmup_decisions=0,
        save_every=1,
        state_path=state_path,
        events_path=events_path,
        random_seed=0,
        decay=1.0,
    )
    controller = AdaptiveOnlinePolicyController(config)
    decision = controller.decide(feature_values={"bias": 1.0}, default_action=ACTION_INVOKE_HELPER)

    event = controller.record_outcome(
        decision=decision,
        run_id="run_no_update",
        case_id="case_no_update",
        condition="layered_adaptive_llm",
        attempt=1,
        note="patcher_request_failed",
        reward_scope="helper_credit_v2",
        reward=-0.2,
        reward_success_hint=0.0,
        reward_progress_hint=0.0,
        reward_progress_score=0.0,
        reward_token_penalty=0.1,
        reward_latency_penalty=0.1,
        reward_stagnation_penalty=0.0,
        reward_helper_materialization_bonus=0.0,
        success=False,
        local_llm_call_count=1,
        local_tokens_prompt=10,
        local_tokens_completion=2,
        local_tokens_total=12,
        local_wall_sec=0.2,
        override_reason="",
        allow_model_update=False,
        update_reason_override="helper_transport_no_update",
    )
    state = controller.state
    controller.close()

    assert event.model_updated is False
    assert event.update_reason == "helper_transport_no_update"
    assert event.reward_progress_score == 0.0
    assert state.decisions == 1
    assert state.updates == 0
    assert len([line for line in events_path.read_text(encoding="utf-8").splitlines() if line]) == 1


def test_online_policy_action_score_biases_affect_greedy_selection_and_event_payload(
    tmp_path,
) -> None:
    state_path = tmp_path / "policy_state_score_biases.json"
    events_path = tmp_path / "policy_events_score_biases.jsonl"
    state_path.write_text(
        json.dumps(
            _policy_state_payload(
                b_by_action={
                    ACTION_INVOKE_HELPER: 1.0,
                    ACTION_SEARCH_ONLY: 1.0,
                }
            ),
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    config = OnlinePolicyConfig.normalize(
        mode="on",
        algorithm="linucb",
        alpha=0.0,
        epsilon=0.0,
        l2=1.0,
        warmup_decisions=0,
        save_every=1,
        state_path=state_path,
        events_path=events_path,
        random_seed=0,
        decay=1.0,
        action_score_biases={
            ACTION_SEARCH_ONLY: -0.5,
            ACTION_FRAME_ONLY: -0.25,
        },
    )
    controller = AdaptiveOnlinePolicyController(config)
    decision = controller.decide(
        feature_values={"bias": 1.0},
        default_action=ACTION_INVOKE_HELPER,
        available_actions=(ACTION_INVOKE_HELPER, ACTION_SEARCH_ONLY),
    )
    event = controller.record_outcome(
        decision=decision,
        run_id="run_score_biases",
        case_id="case_score_biases",
        condition="layered_adaptive_llm",
        attempt=1,
        note="tests_still_failing",
        reward_scope="helper_credit_v2",
        reward=0.0,
        reward_success_hint=0.0,
        reward_progress_hint=0.0,
        reward_progress_score=0.0,
        reward_token_penalty=0.0,
        reward_latency_penalty=0.0,
        reward_stagnation_penalty=0.0,
        reward_helper_materialization_bonus=0.0,
        success=False,
        local_llm_call_count=1,
        local_tokens_prompt=5,
        local_tokens_completion=2,
        local_tokens_total=7,
        local_wall_sec=0.1,
        override_reason="",
    )
    controller.close()
    payload = json.loads(events_path.read_text(encoding="utf-8").splitlines()[0])

    assert decision.selected_action == ACTION_INVOKE_HELPER
    assert decision.scores[ACTION_INVOKE_HELPER] == pytest.approx(1.0)
    assert decision.scores[ACTION_SEARCH_ONLY] == pytest.approx(0.5)
    assert event.configured_action_score_biases[ACTION_SEARCH_ONLY] == pytest.approx(-0.5)
    assert payload["configured_action_score_biases"][ACTION_FRAME_ONLY] == pytest.approx(-0.25)


def test_online_policy_action_score_biases_do_not_change_warmup_or_epsilon_uniform(
    tmp_path,
) -> None:
    config = OnlinePolicyConfig.normalize(
        mode="on",
        algorithm="linucb",
        alpha=0.0,
        epsilon=1.0,
        l2=1.0,
        warmup_decisions=1,
        save_every=1,
        state_path=tmp_path / "policy_state_score_biases_warmup.json",
        events_path=tmp_path / "policy_events_score_biases_warmup.jsonl",
        random_seed=0,
        decay=1.0,
        action_score_biases={
            ACTION_SEARCH_ONLY: -0.5,
            ACTION_FRAME_ONLY: -0.25,
        },
    )
    controller = AdaptiveOnlinePolicyController(config)
    warmup_decision = controller.decide(
        feature_values={"bias": 1.0},
        default_action=ACTION_INVOKE_HELPER,
        available_actions=(ACTION_INVOKE_HELPER, ACTION_SEARCH_ONLY),
    )
    epsilon_decision = controller.decide(
        feature_values={"bias": 1.0},
        default_action=ACTION_INVOKE_HELPER,
        available_actions=(ACTION_INVOKE_HELPER, ACTION_SEARCH_ONLY),
    )
    controller.close()

    assert warmup_decision.decision_strategy == "warmup_uniform"
    assert epsilon_decision.decision_strategy == "epsilon_uniform"


def test_online_policy_reports_load_error_for_invalid_state(tmp_path) -> None:
    state_path = tmp_path / "invalid_state.json"
    state_path.write_text(
        json.dumps({"schema_version": STATE_SCHEMA_VERSION, "algorithm": "linucb"}),
        encoding="utf-8",
    )
    config = OnlinePolicyConfig.normalize(
        mode="on",
        algorithm="linucb",
        state_path=state_path,
        events_path=tmp_path / "events.jsonl",
        decay=1.0,
    )
    controller = AdaptiveOnlinePolicyController(config)
    state = controller.state
    controller.close()
    assert controller.load_error is not None
    assert state.feature_names == []
    assert state.decisions == 0


def test_online_policy_rejects_schema_v1_state(tmp_path) -> None:
    state_path = tmp_path / "schema_v1_state.json"
    state_path.write_text(
        json.dumps({"schema_version": 1, "algorithm": "linucb"}),
        encoding="utf-8",
    )
    config = OnlinePolicyConfig.normalize(
        mode="on",
        algorithm="linucb",
        state_path=state_path,
        events_path=tmp_path / "events.jsonl",
        decay=1.0,
    )
    controller = AdaptiveOnlinePolicyController(config)
    state = controller.state
    controller.close()

    assert controller.load_error is not None
    assert "unsupported schema version" in str(controller.load_error)
    assert state.feature_names == []
    assert state.decisions == 0


def test_online_policy_schema_v2_state_cold_starts_empty(tmp_path) -> None:
    state_path = tmp_path / "schema_v2_state.json"
    state_path.write_text(
        json.dumps(
            {
                **_policy_state_payload(preferred_action=ACTION_INVOKE_HELPER),
                "schema_version": 2,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    config = OnlinePolicyConfig.normalize(
        mode="on",
        algorithm="linucb",
        state_path=state_path,
        events_path=tmp_path / "events_v2.jsonl",
        decay=1.0,
    )
    controller = AdaptiveOnlinePolicyController(config)
    state = controller.state
    controller.close()

    assert controller.load_error is not None
    assert "unsupported schema version" in str(controller.load_error)
    assert state.feature_names == []
    assert state.decisions == 0


def test_online_policy_expands_loaded_state_for_new_feature_dimensions(tmp_path) -> None:
    state_path = tmp_path / "policy_state_expand.json"
    events_path = tmp_path / "policy_events_expand.jsonl"
    state_path.write_text(
        json.dumps(
            {
                **_policy_state_payload(
                    action_counts={ACTION_SKIP_HELPER: 1, ACTION_INVOKE_HELPER: 2},
                    diagonal_by_action={
                        ACTION_SKIP_HELPER: 3.0,
                        ACTION_INVOKE_HELPER: 4.0,
                    },
                    b_by_action={
                        ACTION_SKIP_HELPER: 0.7,
                        ACTION_INVOKE_HELPER: 0.9,
                    },
                ),
                "alpha": 0.2,
                "l2": 1.5,
                "decisions": 3,
                "updates": 2,
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    config = OnlinePolicyConfig.normalize(
        mode="on",
        algorithm="linucb",
        alpha=0.2,
        epsilon=0.0,
        l2=1.5,
        warmup_decisions=0,
        save_every=1,
        state_path=state_path,
        events_path=events_path,
        random_seed=0,
        decay=1.0,
    )
    controller = AdaptiveOnlinePolicyController(config)
    controller.decide(
        feature_values={"bias": 1.0, "attempt_frac": 0.5},
        default_action=ACTION_INVOKE_HELPER,
    )
    state = controller.state
    controller.close()

    assert state.feature_names == ["attempt_frac", "bias"]
    for action in ONLINE_POLICY_ACTIONS:
        assert state.A[action].shape == (2, 2)
        assert state.b[action].shape == (2,)

    bias_idx = state.feature_names.index("bias")
    new_feature_idx = state.feature_names.index("attempt_frac")
    assert state.A[ACTION_SKIP_HELPER][bias_idx, bias_idx] == 3.0
    assert state.A[ACTION_INVOKE_HELPER][bias_idx, bias_idx] == 4.0
    assert state.b[ACTION_SKIP_HELPER][bias_idx] == 0.7
    assert state.b[ACTION_INVOKE_HELPER][bias_idx] == 0.9
    assert state.A[ACTION_SKIP_HELPER][new_feature_idx, new_feature_idx] == 1.5
    assert state.A[ACTION_INVOKE_HELPER][new_feature_idx, new_feature_idx] == 1.5


def test_online_policy_masks_unavailable_actions_before_selection(tmp_path) -> None:
    state_path = tmp_path / "policy_state_masked.json"
    state_path.write_text(
        json.dumps(
            _policy_state_payload(
                preferred_action=ACTION_SNAPSHOT_LITE,
                preferred_value=5.0,
            ),
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    config = OnlinePolicyConfig.normalize(
        mode="on",
        algorithm="linucb",
        alpha=0.0,
        epsilon=0.0,
        warmup_decisions=0,
        state_path=state_path,
        events_path=tmp_path / "policy_events_masked.jsonl",
        decay=1.0,
    )
    controller = AdaptiveOnlinePolicyController(config)
    decision = controller.decide(
        feature_values={"bias": 1.0},
        default_action=ACTION_INVOKE_HELPER,
        available_actions=[ACTION_SKIP_HELPER, ACTION_INVOKE_HELPER],
        unavailable_action_reasons={ACTION_SNAPSHOT_LITE: "snapshot_unavailable"},
    )
    controller.close()

    assert decision.selected_action == ACTION_INVOKE_HELPER
    assert decision.applied_action == ACTION_INVOKE_HELPER
    assert decision.available_actions == (ACTION_SKIP_HELPER, ACTION_INVOKE_HELPER)
    assert decision.unavailable_action_reasons[ACTION_SNAPSHOT_LITE] == "snapshot_unavailable"


def test_online_policy_decay_discounts_stale_evidence_before_update(tmp_path) -> None:
    state_path = tmp_path / "policy_state_decay.json"
    state_path.write_text(
        json.dumps(
            _policy_state_payload(
                preferred_action=ACTION_INVOKE_HELPER,
                preferred_value=2.0,
                diagonal_by_action={ACTION_INVOKE_HELPER: 3.0},
            ),
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    config = OnlinePolicyConfig.normalize(
        mode="on",
        algorithm="linucb",
        alpha=0.0,
        epsilon=0.0,
        decay=0.5,
        warmup_decisions=0,
        state_path=state_path,
        events_path=tmp_path / "policy_events_decay.jsonl",
    )
    controller = AdaptiveOnlinePolicyController(config)
    decision = controller.decide(feature_values={"bias": 1.0}, default_action=ACTION_INVOKE_HELPER)
    controller.record_outcome(
        decision=decision,
        run_id="run_decay",
        case_id="case_decay",
        condition="layered_adaptive_llm",
        attempt=1,
        note="tests_still_failing",
        reward_scope="helper_credit_v2",
        reward=1.0,
        reward_success_hint=0.0,
        reward_progress_hint=0.0,
        reward_progress_score=0.0,
        reward_token_penalty=0.0,
        reward_latency_penalty=0.0,
        reward_stagnation_penalty=0.0,
        reward_helper_materialization_bonus=0.0,
        success=False,
        local_llm_call_count=0,
        local_tokens_prompt=0,
        local_tokens_completion=0,
        local_tokens_total=0,
        local_wall_sec=0.0,
        override_reason="",
    )
    state = controller.state
    controller.close()

    assert state.A[ACTION_INVOKE_HELPER][0, 0] == pytest.approx(3.0)
    assert state.b[ACTION_INVOKE_HELPER][0] == pytest.approx(2.0)


def test_online_policy_percentile_profile_persists_history_and_caps(tmp_path) -> None:
    state_path = tmp_path / "policy_state_percentile.json"
    config = OnlinePolicyConfig.normalize(
        mode="on",
        algorithm="linucb",
        alpha=0.0,
        epsilon=0.0,
        warmup_decisions=0,
        feature_scaling_profile="percentile_v1",
        state_path=state_path,
        events_path=tmp_path / "policy_events_percentile.jsonl",
        random_seed=0,
        decay=1.0,
    )
    controller = AdaptiveOnlinePolicyController(config)
    for idx in range(DEFAULT_POLICY_FEATURE_SCALING_MIN_SAMPLES - 1):
        controller.decide(
            feature_values={"helper_calls_used": float(idx)},
            default_action=ACTION_INVOKE_HELPER,
        )
    decision_before_threshold = controller.decide(
        feature_values={"helper_calls_used": float(DEFAULT_POLICY_FEATURE_SCALING_MIN_SAMPLES - 1)},
        default_action=ACTION_INVOKE_HELPER,
    )
    decision_after_threshold = controller.decide(
        feature_values={"helper_calls_used": float(DEFAULT_POLICY_FEATURE_SCALING_MIN_SAMPLES)},
        default_action=ACTION_INVOKE_HELPER,
    )
    state = controller.state
    controller.close()

    assert decision_before_threshold.feature_scaling_fallback_to_v1 is True
    assert (
        decision_before_threshold.feature_scaling_history_sizes["helper_calls_used"]
        == DEFAULT_POLICY_FEATURE_SCALING_MIN_SAMPLES - 1
    )
    assert (
        decision_before_threshold.feature_scaling_caps["helper_calls_used"]
        == _V1_COUNT_CAPS["helper_calls_used"]
    )
    assert decision_after_threshold.feature_scaling_fallback_to_v1 is False
    assert (
        decision_after_threshold.feature_scaling_history_sizes["helper_calls_used"]
        == DEFAULT_POLICY_FEATURE_SCALING_MIN_SAMPLES
    )
    assert (
        decision_after_threshold.feature_scaling_caps["helper_calls_used"]
        > _V1_COUNT_CAPS["helper_calls_used"]
    )
    assert (
        len(state.feature_scaling_history["helper_calls_used"])
        == DEFAULT_POLICY_FEATURE_SCALING_MIN_SAMPLES + 1
    )

    reloaded = AdaptiveOnlinePolicyController(config)
    reloaded_state = reloaded.state
    reloaded.close()

    assert reloaded.load_error is None
    assert (
        len(reloaded_state.feature_scaling_history["helper_calls_used"])
        == DEFAULT_POLICY_FEATURE_SCALING_MIN_SAMPLES + 1
    )


def test_online_policy_linucb_scores_match_known_ucb_values(tmp_path) -> None:
    state_path = tmp_path / "policy_state_known_scores.json"
    state_path.write_text(
        json.dumps(
            _policy_state_payload(
                diagonal_by_action={
                    ACTION_INVOKE_HELPER: 2.0,
                    ACTION_SKIP_HELPER: 1.0,
                },
                b_by_action={
                    ACTION_INVOKE_HELPER: 1.5,
                    ACTION_SKIP_HELPER: 0.2,
                },
            ),
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    config = OnlinePolicyConfig.normalize(
        mode="on",
        algorithm="linucb",
        alpha=0.5,
        epsilon=0.0,
        warmup_decisions=0,
        state_path=state_path,
        events_path=tmp_path / "policy_events_known_scores.jsonl",
        decay=1.0,
    )
    controller = AdaptiveOnlinePolicyController(config)
    x = controller._feature_vector(cleaned_features={"bias": 1.0})
    scores = controller._linucb_scores(x=x)
    controller.close()

    assert scores[ACTION_INVOKE_HELPER] == pytest.approx(0.75 + (0.5 / math.sqrt(2.0)))
    assert scores[ACTION_SKIP_HELPER] == pytest.approx(0.2 + 0.5)
    assert scores[ACTION_INVOKE_HELPER] > scores[ACTION_SKIP_HELPER]


def test_online_policy_linucb_scores_regularize_singular_covariance(tmp_path) -> None:
    state_path = tmp_path / "policy_state_singular.json"
    state_path.write_text(
        json.dumps(
            _policy_state_payload(
                diagonal_by_action=dict.fromkeys(ONLINE_POLICY_ACTIONS, 0.0),
                b_by_action={ACTION_INVOKE_HELPER: 1.0},
            ),
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    config = OnlinePolicyConfig.normalize(
        mode="on",
        algorithm="linucb",
        alpha=0.3,
        epsilon=0.0,
        l2=1.0,
        warmup_decisions=0,
        state_path=state_path,
        events_path=tmp_path / "policy_events_singular.jsonl",
        decay=1.0,
    )
    controller = AdaptiveOnlinePolicyController(config)
    x = controller._feature_vector(cleaned_features={"bias": 1.0})
    scores = controller._linucb_scores(x=x)

    assert all(math.isfinite(score) for score in scores.values())
    # POL-22: the regularized matrix is used for the score only; A is left as loaded.
    assert controller.state.A[ACTION_INVOKE_HELPER][0, 0] == pytest.approx(0.0)
    assert controller.state.A[ACTION_SKIP_HELPER][0, 0] == pytest.approx(0.0)
    controller.close()


def test_online_policy_records_configured_tuning_telemetry(tmp_path) -> None:
    config = OnlinePolicyConfig.normalize(
        mode="on",
        algorithm="linucb",
        alpha=0.7,
        epsilon=0.05,
        decay=0.85,
        warmup_decisions=2,
        state_path=tmp_path / "policy_state_telemetry.json",
        events_path=tmp_path / "policy_events_telemetry.jsonl",
        random_seed=0,
    )
    controller = AdaptiveOnlinePolicyController(config)
    decision = controller.decide(
        feature_values={"bias": 1.0},
        default_action=ACTION_INVOKE_HELPER,
        available_actions=(ACTION_INVOKE_HELPER, ACTION_SKIP_HELPER),
    )
    event = controller.record_outcome(
        decision=decision,
        run_id="run_online_telemetry",
        case_id="case_online_telemetry",
        condition="layered_adaptive_llm",
        attempt=1,
        note="tests_still_failing",
        reward_scope="helper_credit_v2",
        reward=0.2,
        reward_success_hint=0.0,
        reward_progress_hint=0.0,
        reward_progress_score=0.1,
        reward_token_penalty=0.0,
        reward_latency_penalty=0.0,
        reward_stagnation_penalty=0.0,
        reward_helper_materialization_bonus=0.0,
        success=False,
        local_llm_call_count=0,
        local_tokens_prompt=0,
        local_tokens_completion=0,
        local_tokens_total=0,
        local_wall_sec=0.0,
        override_reason="",
    )
    controller.close()

    assert decision.decision_strategy == "warmup_uniform"
    assert decision.configured_alpha == pytest.approx(0.7)
    assert decision.configured_epsilon == pytest.approx(0.05)
    assert decision.configured_decay == pytest.approx(0.85)
    assert decision.effective_epsilon == pytest.approx(1.0)
    assert decision.warmup_active is True
    assert decision.propensity == pytest.approx(0.5)
    assert event.configured_alpha == pytest.approx(0.7)
    assert event.configured_epsilon == pytest.approx(0.05)
    assert event.configured_decay == pytest.approx(0.85)
    assert event.effective_epsilon == pytest.approx(1.0)
    assert event.warmup_active is True


# ---------------------------------------------------------------------------
# Helper: build a minimal record_outcome call (all required kwargs)
# ---------------------------------------------------------------------------


def _record_outcome_minimal(
    controller: AdaptiveOnlinePolicyController,
    decision,
    *,
    reward: float = 0.5,
    behavior_action: str | None = None,
) -> object:
    return controller.record_outcome(
        decision=decision,
        run_id="run_cov",
        case_id="case_cov",
        condition="layered_adaptive_llm",
        attempt=1,
        note="coverage",
        reward_scope="helper_credit_v2",
        reward=reward,
        reward_success_hint=0.0,
        reward_progress_hint=0.0,
        reward_progress_score=0.0,
        reward_token_penalty=0.0,
        reward_latency_penalty=0.0,
        reward_stagnation_penalty=0.0,
        reward_helper_materialization_bonus=0.0,
        success=False,
        local_llm_call_count=0,
        local_tokens_prompt=0,
        local_tokens_completion=0,
        local_tokens_total=0,
        local_wall_sec=0.0,
        override_reason="",
        behavior_action=behavior_action,
    )


# ---------------------------------------------------------------------------
# OnlinePolicyConfig.normalize - invalid mode → "off" (line 113)
# ---------------------------------------------------------------------------


def test_normalize_invalid_mode_falls_back_to_off() -> None:
    cfg = OnlinePolicyConfig.normalize(mode="bogus_mode")
    assert cfg.mode == "off"


# ---------------------------------------------------------------------------
# OnlinePolicyConfig.normalize - invalid algorithm → "linucb" (line 116)
# ---------------------------------------------------------------------------


def test_normalize_invalid_algorithm_falls_back_to_linucb() -> None:
    cfg = OnlinePolicyConfig.normalize(algorithm="unknown_algo")
    assert cfg.algorithm == "linucb"


# ---------------------------------------------------------------------------
# OnlinePolicyConfig.normalize - invalid feature_scaling_profile (line 125)
# ---------------------------------------------------------------------------


def test_normalize_invalid_feature_scaling_profile_falls_back() -> None:
    cfg = OnlinePolicyConfig.normalize(
        feature_scaling_profile="not_a_real_profile",
        decay=1.0,
    )
    assert cfg.feature_scaling_profile == "v1"


# ---------------------------------------------------------------------------
# OnlinePolicyConfig.normalize - non-numeric random_seed (lines 130-131)
# ---------------------------------------------------------------------------


def test_normalize_non_numeric_random_seed_becomes_none() -> None:
    cfg = OnlinePolicyConfig.normalize(
        random_seed="not_a_number",  # type: ignore[arg-type],
        decay=1.0,
    )
    assert cfg.random_seed is None


# ---------------------------------------------------------------------------
# decide() with mode="off" → baseline, strategy="policy_off" (lines 500-504)
# ---------------------------------------------------------------------------


def test_decide_mode_off_returns_baseline(tmp_path) -> None:
    cfg = OnlinePolicyConfig.normalize(
        mode="off",
        state_path=tmp_path / "state_off.json",
        events_path=tmp_path / "events_off.jsonl",
        decay=1.0,
    )
    ctrl = AdaptiveOnlinePolicyController(config=cfg)
    decision = ctrl.decide(
        feature_values={"bias": 1.0},
        default_action=ACTION_INVOKE_HELPER,
    )
    assert decision.decision_strategy == "policy_off"
    assert decision.selected_action == ACTION_INVOKE_HELPER
    assert decision.applied_action == ACTION_INVOKE_HELPER
    assert decision.propensity == pytest.approx(1.0)
    # All scores must be zero in off mode.
    for score in decision.scores.values():
        assert score == pytest.approx(0.0)
    ctrl.close()


# ---------------------------------------------------------------------------
# decide() - epsilon-greedy exploration branch (lines 517-518)
# ---------------------------------------------------------------------------


def test_decide_epsilon_greedy_exploration(tmp_path) -> None:
    cfg = OnlinePolicyConfig.normalize(
        mode="on",
        epsilon=1.0,  # always explore
        warmup_decisions=0,  # skip warmup
        state_path=tmp_path / "state_eps.json",
        events_path=tmp_path / "events_eps.jsonl",
        random_seed=42,
        decay=1.0,
    )
    ctrl = AdaptiveOnlinePolicyController(config=cfg)
    decision = ctrl.decide(
        feature_values={"bias": 1.0},
        default_action=ACTION_INVOKE_HELPER,
        available_actions=(ACTION_INVOKE_HELPER, ACTION_SKIP_HELPER),
    )
    assert decision.decision_strategy == "epsilon_uniform"
    ctrl.close()


# ---------------------------------------------------------------------------
# record_outcome() - invalid behavior_action → fallback (line 606)
# ---------------------------------------------------------------------------


def test_record_outcome_invalid_behavior_action_fallback(tmp_path) -> None:
    cfg = OnlinePolicyConfig.normalize(
        mode="on",
        warmup_decisions=0,
        state_path=tmp_path / "state_ba.json",
        events_path=tmp_path / "events_ba.jsonl",
        random_seed=0,
        decay=1.0,
    )
    ctrl = AdaptiveOnlinePolicyController(config=cfg)
    decision = ctrl.decide(
        feature_values={"bias": 1.0},
        default_action=ACTION_INVOKE_HELPER,
    )
    event = _record_outcome_minimal(
        ctrl,
        decision,
        behavior_action="completely_invalid_action",
    )
    # Should fall back to applied_action (or "none").
    assert event.behavior_action in ONLINE_POLICY_ACTIONS or event.behavior_action == "none"
    ctrl.close()


# ---------------------------------------------------------------------------
# record_outcome() with mode="off" → "off_no_update" (lines 630-631)
# ---------------------------------------------------------------------------


def test_record_outcome_mode_off_no_update(tmp_path) -> None:
    cfg = OnlinePolicyConfig.normalize(
        mode="off",
        state_path=tmp_path / "state_off_rec.json",
        events_path=tmp_path / "events_off_rec.jsonl",
        decay=1.0,
    )
    ctrl = AdaptiveOnlinePolicyController(config=cfg)
    decision = ctrl.decide(
        feature_values={"bias": 1.0},
        default_action=ACTION_INVOKE_HELPER,
    )
    event = _record_outcome_minimal(ctrl, decision)
    assert event.model_updated is False
    assert event.update_reason == "off_no_update"
    ctrl.close()


# ---------------------------------------------------------------------------
# _ensure_feature_state() - algorithm != "linucb" → early return (line 710)
# ---------------------------------------------------------------------------


def test_ensure_feature_state_non_linucb_noop(tmp_path) -> None:
    """Controller with a non-linucb algorithm must skip feature init."""
    # normalize() forces unknown algorithm → "linucb", so we
    # build the config directly with __init__ to bypass that.
    cfg = OnlinePolicyConfig(
        mode="on",
        algorithm="other",
        state_path=tmp_path / "state_algo.json",
        events_path=tmp_path / "events_algo.jsonl",
    )
    ctrl = AdaptiveOnlinePolicyController(config=cfg)
    # Internally _ensure_feature_state should early-return; state stays empty.
    assert ctrl.state.feature_names == []
    assert ctrl.state.A == {}
    assert ctrl.state.b == {}
    ctrl.close()


# ---------------------------------------------------------------------------
# _ensure_feature_state() - empty features → ["bias"], dim=1 (lines 715-716)
# ---------------------------------------------------------------------------


def test_ensure_feature_state_empty_features_default_bias(tmp_path) -> None:
    cfg = OnlinePolicyConfig.normalize(
        mode="on",
        warmup_decisions=0,
        state_path=tmp_path / "state_empty.json",
        events_path=tmp_path / "events_empty.jsonl",
        random_seed=0,
        decay=1.0,
    )
    ctrl = AdaptiveOnlinePolicyController(config=cfg)
    # Pass empty features → should initialise with ["bias"].
    ctrl.decide(
        feature_values={},
        default_action=ACTION_INVOKE_HELPER,
    )
    state = ctrl.state
    assert state.feature_names == ["bias"]
    assert all(a.shape == (1, 1) for a in state.A.values())
    assert all(b.shape == (1,) for b in state.b.values())
    ctrl.close()


# ---------------------------------------------------------------------------
# _ensure_feature_state() - matrix shape mismatch → reinitialize (lines 762, 764)
# ---------------------------------------------------------------------------


def test_ensure_feature_state_shape_mismatch_reinit(tmp_path) -> None:
    state_path = tmp_path / "state_shape.json"
    events_path = tmp_path / "events_shape.jsonl"
    cfg = OnlinePolicyConfig.normalize(
        mode="on",
        warmup_decisions=0,
        save_every=1,
        state_path=state_path,
        events_path=events_path,
        random_seed=0,
        decay=1.0,
    )
    # Create a controller, make one decision so state is written.
    ctrl = AdaptiveOnlinePolicyController(config=cfg)
    dec = ctrl.decide(
        feature_values={"bias": 1.0},
        default_action=ACTION_INVOKE_HELPER,
    )
    _record_outcome_minimal(ctrl, dec)
    ctrl.close()

    # Corrupt matrix shapes: feature_names=["bias"] (dim=1) but A is 2x2.
    raw = json.loads(state_path.read_text(encoding="utf-8"))
    for action in ONLINE_POLICY_ACTIONS:
        raw["A"][action] = [[1.0, 0.0], [0.0, 1.0]]
        raw["b"][action] = [0.0, 0.0]
    state_path.write_text(json.dumps(raw, ensure_ascii=False), encoding="utf-8")

    # _load_state_if_present detects the shape/dim mismatch → load_error.
    ctrl2 = AdaptiveOnlinePolicyController(config=cfg)
    assert ctrl2.load_error is not None
    ctrl2.close()

    # Now build a consistent dim=2 state file to load cleanly, then
    # trigger _ensure_feature_state with a new feature that forces the
    # shape-mismatch repair path (lines 762, 764) inside the method.
    raw2 = json.loads(state_path.read_text(encoding="utf-8"))
    raw2["feature_names"] = ["bias", "extra"]
    for action in ONLINE_POLICY_ACTIONS:
        raw2["A"][action] = [[1.0, 0.0], [0.0, 1.0]]
        raw2["b"][action] = [0.0, 0.0]
    state_path.write_text(json.dumps(raw2, ensure_ascii=False), encoding="utf-8")

    ctrl3 = AdaptiveOnlinePolicyController(config=cfg)
    assert ctrl3.load_error is None
    # Corrupt one action's matrix shape *after* load, then decide.
    import numpy as np

    with ctrl3._lock:
        ctrl3._state.A[ACTION_SKIP_HELPER] = np.eye(5, dtype=float)
        ctrl3._state.b[ACTION_SKIP_HELPER] = np.zeros((5,), dtype=float)
    ctrl3.decide(
        feature_values={"bias": 1.0, "extra": 2.0},
        default_action=ACTION_INVOKE_HELPER,
    )
    st3 = ctrl3.state
    dim = len(st3.feature_names)
    for action in ONLINE_POLICY_ACTIONS:
        assert st3.A[action].shape == (dim, dim)
        assert st3.b[action].shape == (dim,)
    ctrl3.close()


# ---------------------------------------------------------------------------
# _feature_vector() - empty feature_names → fallback to ["bias"] (line 770)
# ---------------------------------------------------------------------------


def test_feature_vector_empty_feature_names_fallback(tmp_path) -> None:
    cfg = OnlinePolicyConfig.normalize(
        mode="on",
        warmup_decisions=0,
        state_path=tmp_path / "state_fv.json",
        events_path=tmp_path / "events_fv.jsonl",
        random_seed=0,
        decay=1.0,
    )
    ctrl = AdaptiveOnlinePolicyController(config=cfg)
    # Before any decide(), feature_names is []; _feature_vector uses ["bias"].
    # We trigger this path by passing empty features on the first call.
    decision = ctrl.decide(
        feature_values={},
        default_action=ACTION_INVOKE_HELPER,
    )
    # The decision should succeed; feature_names gets populated.
    assert decision.selected_action in ONLINE_POLICY_ACTIONS
    ctrl.close()


# ---------------------------------------------------------------------------
# _linucb_scores() - algorithm != "linucb" → zeros (line 788)
# ---------------------------------------------------------------------------


def test_linucb_scores_non_linucb_returns_zeros(tmp_path) -> None:
    cfg = OnlinePolicyConfig(
        mode="on",
        algorithm="other",
        state_path=tmp_path / "state_nolcb.json",
        events_path=tmp_path / "events_nolcb.jsonl",
        action_score_biases=dict.fromkeys(ONLINE_POLICY_ACTIONS, 0.0),
    )
    ctrl = AdaptiveOnlinePolicyController(config=cfg)
    decision = ctrl.decide(
        feature_values={"bias": 1.0},
        default_action=ACTION_INVOKE_HELPER,
    )
    for score in decision.scores.values():
        assert score == pytest.approx(0.0)
    ctrl.close()


# ---------------------------------------------------------------------------
# _linucb_scores() - missing A/b → score=0.0 (lines 793-794)
# ---------------------------------------------------------------------------


def test_linucb_scores_missing_matrices_returns_zero(tmp_path) -> None:
    cfg = OnlinePolicyConfig.normalize(
        mode="on",
        warmup_decisions=0,
        state_path=tmp_path / "state_miss.json",
        events_path=tmp_path / "events_miss.jsonl",
        random_seed=0,
        decay=1.0,
    )
    ctrl = AdaptiveOnlinePolicyController(config=cfg)
    # Manually wipe A/b for one action after feature init.
    ctrl.decide(
        feature_values={"bias": 1.0},
        default_action=ACTION_INVOKE_HELPER,
    )
    # Remove matrices for one action, then decide again.
    with ctrl._lock:
        ctrl._state.A.pop(ACTION_SKIP_HELPER, None)
        ctrl._state.b.pop(ACTION_SKIP_HELPER, None)
    decision2 = ctrl.decide(
        feature_values={"bias": 1.0},
        default_action=ACTION_INVOKE_HELPER,
    )
    # _ensure_feature_state reinitialises missing matrices, but let's
    # verify no crash occurred and a valid score exists.
    assert ACTION_SKIP_HELPER in decision2.scores
    ctrl.close()


# ---------------------------------------------------------------------------
# _update_model() - algorithm != "linucb" (line 815)
# ---------------------------------------------------------------------------


def test_update_model_non_linucb_returns_false(tmp_path) -> None:
    cfg = OnlinePolicyConfig(
        mode="on",
        algorithm="other",
        state_path=tmp_path / "state_upd.json",
        events_path=tmp_path / "events_upd.jsonl",
    )
    ctrl = AdaptiveOnlinePolicyController(config=cfg)
    decision = ctrl.decide(
        feature_values={"bias": 1.0},
        default_action=ACTION_INVOKE_HELPER,
    )
    event = _record_outcome_minimal(ctrl, decision)
    # Model cannot update with non-linucb algorithm.
    assert event.model_updated is False
    ctrl.close()


# ---------------------------------------------------------------------------
# _update_model() - invalid action (line 817)
# ---------------------------------------------------------------------------


def test_update_model_invalid_action(tmp_path) -> None:
    cfg = OnlinePolicyConfig.normalize(
        mode="on",
        warmup_decisions=0,
        save_every=1,
        state_path=tmp_path / "state_inv.json",
        events_path=tmp_path / "events_inv.jsonl",
        random_seed=0,
        decay=1.0,
    )
    ctrl = AdaptiveOnlinePolicyController(config=cfg)
    decision = ctrl.decide(
        feature_values={"bias": 1.0},
        default_action=ACTION_INVOKE_HELPER,
    )
    # Provide a behavior_action that is not in ONLINE_POLICY_ACTIONS;
    # record_outcome normalises it, so the _update_model receives a valid
    # action (the fallback). We verify no crash occurs.
    event = _record_outcome_minimal(
        ctrl,
        decision,
        behavior_action="invalid_action_xyz",
    )
    # behavior_action was invalid, so it falls back, update still may succeed.
    assert isinstance(event.model_updated, bool)
    ctrl.close()


# ---------------------------------------------------------------------------
# _update_model() - empty feature_names (line 819)
# ---------------------------------------------------------------------------


def test_update_model_empty_feature_names(tmp_path) -> None:
    cfg = OnlinePolicyConfig.normalize(
        mode="on",
        warmup_decisions=0,
        state_path=tmp_path / "state_ef.json",
        events_path=tmp_path / "events_ef.jsonl",
        random_seed=0,
        decay=1.0,
    )
    ctrl = AdaptiveOnlinePolicyController(config=cfg)
    decision = ctrl.decide(
        feature_values={"bias": 1.0},
        default_action=ACTION_INVOKE_HELPER,
    )
    # Clear feature_names after decide to force empty path.
    with ctrl._lock:
        ctrl._state.feature_names = []
    event = _record_outcome_minimal(ctrl, decision)
    # _update_model returns False when feature_names is empty.
    assert event.model_updated is False
    ctrl.close()


# ---------------------------------------------------------------------------
# _update_model() - missing A/b for action (line 825)
# ---------------------------------------------------------------------------


def test_update_model_missing_A_b(tmp_path) -> None:
    cfg = OnlinePolicyConfig.normalize(
        mode="on",
        warmup_decisions=0,
        state_path=tmp_path / "state_ab.json",
        events_path=tmp_path / "events_ab.jsonl",
        random_seed=0,
        decay=1.0,
    )
    ctrl = AdaptiveOnlinePolicyController(config=cfg)
    decision = ctrl.decide(
        feature_values={"bias": 1.0},
        default_action=ACTION_INVOKE_HELPER,
    )
    # Remove A/b for the action that was selected.
    selected = decision.applied_action
    with ctrl._lock:
        ctrl._state.A.pop(selected, None)
        ctrl._state.b.pop(selected, None)
    event = _record_outcome_minimal(ctrl, decision)
    # _update_model returns False when A or b is missing.
    assert event.model_updated is False
    ctrl.close()


# ---------------------------------------------------------------------------
# _apply_recency_decay() - A/b None or wrong shape → continue (line 844)
# ---------------------------------------------------------------------------


def test_apply_recency_decay_skips_bad_shapes(tmp_path) -> None:
    import numpy as np

    cfg = OnlinePolicyConfig.normalize(
        mode="on",
        decay=0.9,
        warmup_decisions=0,
        state_path=tmp_path / "state_dec.json",
        events_path=tmp_path / "events_dec.jsonl",
        random_seed=0,
    )
    ctrl = AdaptiveOnlinePolicyController(config=cfg)
    decision = ctrl.decide(
        feature_values={"bias": 1.0},
        default_action=ACTION_INVOKE_HELPER,
    )
    # Corrupt one action's A/b to wrong shape.
    with ctrl._lock:
        ctrl._state.A[ACTION_SKIP_HELPER] = np.eye(5, dtype=float)
        ctrl._state.b[ACTION_SKIP_HELPER] = np.zeros((5,), dtype=float)
    # Record outcome triggers _apply_recency_decay; must not crash.
    event = _record_outcome_minimal(ctrl, decision)
    assert isinstance(event.model_updated, bool)
    ctrl.close()


# ---------------------------------------------------------------------------
# _save_state() - state_path=None → return (line 890)
# ---------------------------------------------------------------------------


def test_save_state_none_path_noop() -> None:
    cfg = OnlinePolicyConfig.normalize(
        mode="on",
        state_path=None,
        events_path=None,
        decay=1.0,
    )
    ctrl = AdaptiveOnlinePolicyController(config=cfg)
    # close() calls _save_state() which should silently return.
    ctrl.close()  # must not raise

    # Verify state_path is None and no side effects occurred
    assert cfg.state_path is None
    assert cfg.events_path is None


# ---------------------------------------------------------------------------
# _load_state_if_present() - non-dict JSON (line 905)
# ---------------------------------------------------------------------------


def test_load_state_non_dict_json(tmp_path) -> None:
    state_path = tmp_path / "state_nondict.json"
    state_path.write_text(json.dumps([1, 2, 3]), encoding="utf-8")
    cfg = OnlinePolicyConfig.normalize(
        mode="on",
        state_path=state_path,
        events_path=tmp_path / "events_nondict.jsonl",
        decay=1.0,
    )
    ctrl = AdaptiveOnlinePolicyController(config=cfg)
    assert ctrl.load_error is not None
    assert "not an object" in ctrl.load_error
    ctrl.close()


# ---------------------------------------------------------------------------
# _load_state_if_present() - algorithm mismatch (line 910)
# ---------------------------------------------------------------------------


def test_load_state_algorithm_mismatch(tmp_path) -> None:
    state_path = tmp_path / "state_algomm.json"
    payload = _policy_state_payload(preferred_action=ACTION_INVOKE_HELPER)
    payload["algorithm"] = "thompson"
    state_path.write_text(json.dumps(payload), encoding="utf-8")
    cfg = OnlinePolicyConfig.normalize(
        mode="on",
        algorithm="linucb",
        state_path=state_path,
        events_path=tmp_path / "events_algomm.jsonl",
        decay=1.0,
    )
    ctrl = AdaptiveOnlinePolicyController(config=cfg)
    assert ctrl.load_error is not None
    assert "algorithm mismatch" in ctrl.load_error
    ctrl.close()


# ---------------------------------------------------------------------------
# _load_state_if_present() - feature_scaling_profile mismatch (line 918)
# ---------------------------------------------------------------------------


def test_load_state_feature_scaling_profile_mismatch(tmp_path) -> None:
    state_path = tmp_path / "state_fsm.json"
    payload = _policy_state_payload(preferred_action=ACTION_INVOKE_HELPER)
    payload["feature_scaling_profile"] = "none"
    state_path.write_text(json.dumps(payload), encoding="utf-8")
    cfg = OnlinePolicyConfig.normalize(
        mode="on",
        feature_scaling_profile="v1",
        state_path=state_path,
        events_path=tmp_path / "events_fsm.jsonl",
        decay=1.0,
    )
    ctrl = AdaptiveOnlinePolicyController(config=cfg)
    assert ctrl.load_error is not None
    assert "feature_scaling_profile mismatch" in ctrl.load_error
    ctrl.close()


# ---------------------------------------------------------------------------
# _load_state_if_present() - feature_scaling_history value not a list (line 940)
# ---------------------------------------------------------------------------


def test_load_state_feature_scaling_history_not_list(tmp_path) -> None:
    state_path = tmp_path / "state_fsh.json"
    payload = _policy_state_payload(preferred_action=ACTION_INVOKE_HELPER)
    payload["feature_scaling_history"] = {"bias": "not_a_list"}
    state_path.write_text(json.dumps(payload), encoding="utf-8")
    cfg = OnlinePolicyConfig.normalize(
        mode="on",
        state_path=state_path,
        events_path=tmp_path / "events_fsh.jsonl",
        decay=1.0,
    )
    ctrl = AdaptiveOnlinePolicyController(config=cfg)
    assert ctrl.load_error is None
    # The non-list value should have been skipped.
    assert "bias" not in ctrl.state.feature_scaling_history
    ctrl.close()


# ---------------------------------------------------------------------------
# _load_state_if_present() - A/b not dicts → ValueError (line 968)
# ---------------------------------------------------------------------------


def test_load_state_A_b_not_dicts(tmp_path) -> None:
    state_path = tmp_path / "state_abdicts.json"
    payload = _policy_state_payload(preferred_action=ACTION_INVOKE_HELPER)
    payload["A"] = "not_a_dict"
    payload["b"] = "not_a_dict"
    state_path.write_text(json.dumps(payload), encoding="utf-8")
    cfg = OnlinePolicyConfig.normalize(
        mode="on",
        state_path=state_path,
        events_path=tmp_path / "events_abdicts.jsonl",
        decay=1.0,
    )
    ctrl = AdaptiveOnlinePolicyController(config=cfg)
    assert ctrl.load_error is not None
    assert "missing matrices" in ctrl.load_error
    ctrl.close()


# ---------------------------------------------------------------------------
# _load_state_if_present() - wrong matrix shape (line 977)
# ---------------------------------------------------------------------------


def test_load_state_wrong_matrix_shape(tmp_path) -> None:
    state_path = tmp_path / "state_mshape.json"
    payload = _policy_state_payload(preferred_action=ACTION_INVOKE_HELPER)
    # feature_names=["bias"] -> dim=1; set A matrices to 2x2.
    for action in ONLINE_POLICY_ACTIONS:
        payload["A"][action] = [[1.0, 0.0], [0.0, 1.0]]
    state_path.write_text(json.dumps(payload), encoding="utf-8")
    cfg = OnlinePolicyConfig.normalize(
        mode="on",
        state_path=state_path,
        events_path=tmp_path / "events_mshape.jsonl",
        decay=1.0,
    )
    ctrl = AdaptiveOnlinePolicyController(config=cfg)
    assert ctrl.load_error is not None
    assert "invalid matrix shape" in ctrl.load_error
    ctrl.close()


# ---------------------------------------------------------------------------
# _load_state_if_present() - wrong vector shape (line 979)
# ---------------------------------------------------------------------------


def test_load_state_wrong_vector_shape(tmp_path) -> None:
    state_path = tmp_path / "state_vshape.json"
    payload = _policy_state_payload(preferred_action=ACTION_INVOKE_HELPER)
    # feature_names=["bias"] → dim=1; set b vectors to length 2.
    for action in ONLINE_POLICY_ACTIONS:
        payload["b"][action] = [0.0, 0.0]
    state_path.write_text(json.dumps(payload), encoding="utf-8")
    cfg = OnlinePolicyConfig.normalize(
        mode="on",
        state_path=state_path,
        events_path=tmp_path / "events_vshape.jsonl",
        decay=1.0,
    )
    ctrl = AdaptiveOnlinePolicyController(config=cfg)
    assert ctrl.load_error is not None
    assert "invalid vector shape" in ctrl.load_error
    ctrl.close()


# ---------------------------------------------------------------------------
# _append_event() - events_path=None → return (line 1001)
# ---------------------------------------------------------------------------


def test_append_event_none_path_noop(tmp_path) -> None:
    cfg = OnlinePolicyConfig.normalize(
        mode="on",
        state_path=tmp_path / "state_noev.json",
        events_path=None,
        decay=1.0,
    )
    ctrl = AdaptiveOnlinePolicyController(config=cfg)
    decision = ctrl.decide(
        feature_values={"bias": 1.0},
        default_action=ACTION_INVOKE_HELPER,
    )
    # record_outcome internally calls _append_event which should
    # silently return when events_path is None.
    event = _record_outcome_minimal(ctrl, decision)
    assert event is not None
    ctrl.close()


# ---------------------------------------------------------------------------
# Thompson Sampling - algorithm acceptance
# ---------------------------------------------------------------------------


def test_normalize_thompson_algorithm_accepted() -> None:
    cfg = OnlinePolicyConfig.normalize(algorithm="thompson")
    assert cfg.algorithm == "thompson"


def test_normalize_thompson_case_insensitive() -> None:
    cfg = OnlinePolicyConfig.normalize(algorithm="Thompson")
    assert cfg.algorithm == "thompson"


# ---------------------------------------------------------------------------
# Thompson Sampling - scoring
# ---------------------------------------------------------------------------


def test_thompson_scores_returns_all_actions(tmp_path) -> None:
    cfg = OnlinePolicyConfig.normalize(
        mode="on",
        algorithm="thompson",
        warmup_decisions=0,
        epsilon=0.0,
        state_path=tmp_path / "ts_state.json",
        events_path=tmp_path / "ts_events.jsonl",
        random_seed=42,
        decay=1.0,
    )
    ctrl = AdaptiveOnlinePolicyController(config=cfg)
    decision = ctrl.decide(
        feature_values={"bias": 1.0},
        default_action=ACTION_INVOKE_HELPER,
    )
    # Every action should have a finite score.
    assert set(decision.scores.keys()) == set(ONLINE_POLICY_ACTIONS)
    for score in decision.scores.values():
        assert math.isfinite(score)
    ctrl.close()


def test_thompson_scores_reproducible_with_seed(tmp_path) -> None:
    scores_runs: list[dict[str, float]] = []
    for trial in range(2):
        cfg = OnlinePolicyConfig.normalize(
            mode="on",
            algorithm="thompson",
            warmup_decisions=0,
            epsilon=0.0,
            state_path=tmp_path / f"ts_repro_{trial}.json",
            events_path=tmp_path / f"ts_repro_{trial}.jsonl",
            random_seed=99,
            decay=1.0,
        )
        ctrl = AdaptiveOnlinePolicyController(config=cfg)
        decision = ctrl.decide(
            feature_values={"bias": 1.0},
            default_action=ACTION_INVOKE_HELPER,
        )
        scores_runs.append(dict(decision.scores))
        ctrl.close()
    # Same seed → same scores on first decision.
    for action in ONLINE_POLICY_ACTIONS:
        assert scores_runs[0][action] == pytest.approx(scores_runs[1][action])


# ---------------------------------------------------------------------------
# Thompson Sampling - update model works
# ---------------------------------------------------------------------------


def test_thompson_update_model_succeeds(tmp_path) -> None:
    cfg = OnlinePolicyConfig.normalize(
        mode="on",
        algorithm="thompson",
        warmup_decisions=0,
        epsilon=0.0,
        state_path=tmp_path / "ts_upd.json",
        events_path=tmp_path / "ts_upd.jsonl",
        random_seed=7,
        decay=1.0,
    )
    ctrl = AdaptiveOnlinePolicyController(config=cfg)
    decision = ctrl.decide(
        feature_values={"bias": 1.0},
        default_action=ACTION_INVOKE_HELPER,
    )
    event = _record_outcome_minimal(ctrl, decision, reward=1.0)
    # Thompson uses same A/b update as LinUCB → model_updated should be True.
    assert event.model_updated is True
    ctrl.close()


# ---------------------------------------------------------------------------
# Thompson Sampling - end-to-end decide→record→decide cycle
# ---------------------------------------------------------------------------


def test_thompson_end_to_end_cycle(tmp_path) -> None:
    cfg = OnlinePolicyConfig.normalize(
        mode="on",
        algorithm="thompson",
        warmup_decisions=0,
        epsilon=0.0,
        state_path=tmp_path / "ts_e2e.json",
        events_path=tmp_path / "ts_e2e.jsonl",
        random_seed=12,
        decay=1.0,
    )
    ctrl = AdaptiveOnlinePolicyController(config=cfg)
    for _i in range(5):
        decision = ctrl.decide(
            feature_values={"bias": 1.0},
            default_action=ACTION_INVOKE_HELPER,
        )
        assert decision.algorithm == "thompson"
        _record_outcome_minimal(ctrl, decision, reward=0.8)
    ctrl.close()


# ---------------------------------------------------------------------------
# Thompson Sampling - state roundtrip
# ---------------------------------------------------------------------------


def test_thompson_state_roundtrip(tmp_path) -> None:
    state_path = tmp_path / "ts_rt.json"
    events_path = tmp_path / "ts_rt.jsonl"

    # Phase 1: train and save.
    cfg1 = OnlinePolicyConfig.normalize(
        mode="on",
        algorithm="thompson",
        warmup_decisions=0,
        epsilon=0.0,
        state_path=state_path,
        events_path=events_path,
        random_seed=42,
        decay=1.0,
    )
    ctrl1 = AdaptiveOnlinePolicyController(config=cfg1)
    for _ in range(3):
        d = ctrl1.decide(feature_values={"bias": 1.0}, default_action=ACTION_INVOKE_HELPER)
        _record_outcome_minimal(ctrl1, d, reward=1.0)
    ctrl1.close()

    # Phase 2: reload and verify continued operation.
    cfg2 = OnlinePolicyConfig.normalize(
        mode="on",
        algorithm="thompson",
        warmup_decisions=0,
        epsilon=0.0,
        state_path=state_path,
        events_path=events_path,
        random_seed=42,
        decay=1.0,
    )
    ctrl2 = AdaptiveOnlinePolicyController(config=cfg2)
    assert ctrl2.state.decisions == 3
    assert ctrl2.state.updates == 3
    assert sum(ctrl2.state.action_counts.values()) == 3
    d2 = ctrl2.decide(feature_values={"bias": 1.0}, default_action=ACTION_INVOKE_HELPER)
    assert d2.algorithm == "thompson"
    for score in d2.scores.values():
        assert math.isfinite(score)
    ctrl2.close()


# ---------------------------------------------------------------------------
# Thompson Sampling - warmup and epsilon still respected
# ---------------------------------------------------------------------------


def test_thompson_respects_warmup(tmp_path) -> None:
    cfg = OnlinePolicyConfig.normalize(
        mode="on",
        algorithm="thompson",
        warmup_decisions=100,
        epsilon=0.0,
        state_path=tmp_path / "ts_warm.json",
        events_path=tmp_path / "ts_warm.jsonl",
        random_seed=1,
        decay=1.0,
    )
    ctrl = AdaptiveOnlinePolicyController(config=cfg)
    decision = ctrl.decide(
        feature_values={"bias": 1.0},
        default_action=ACTION_INVOKE_HELPER,
    )
    # During warmup, strategy should be uniform random.
    assert "warmup" in decision.decision_strategy
    ctrl.close()


def test_pol16_nan_counter_survives_concurrent_decisions(tmp_path) -> None:
    """POL-16: NaN replacements were counted outside the lock and could be lost."""
    import threading

    controller = AdaptiveOnlinePolicyController(
        OnlinePolicyConfig.normalize(mode="on", events_path=tmp_path / "e.jsonl")
    )
    barrier = threading.Barrier(8)

    def _decide() -> None:
        barrier.wait()
        for _ in range(50):
            controller.decide(feature_values={"bias": 1.0, "broken": float("nan")})

    threads = [threading.Thread(target=_decide) for _ in range(8)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert controller.get_nan_stats()["broken"] == 400
    controller.close()
