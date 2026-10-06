"""Dense helper coverage for ``evals.analyze_results``."""

from __future__ import annotations

import math
from pathlib import Path

import pytest

from evals import analyze_results as ar
from evals.analysis_ope_support import OpeSummarySettings


def _record(
    *,
    run_id: str = "run",
    case_id: str = "case",
    condition: str = "tracecoder",
    attempt: int = 1,
    success: bool = False,
    category: str = "logic_error",
    difficulty: str = "easy",
    snapshot_dependency: str = "required",
    context_protocol: str = "monolithic",
    attempt_wall_clock_sec: float = 1.0,
    cumulative_runtime_sec: float = 1.0,
    note: str = "tests_still_failing",
    attempt_metadata_extra: dict[str, object] | None = None,
    patch_meta_extra: dict[str, object] | None = None,
) -> dict[str, object]:
    attempt_metadata = {
        "attempt": attempt,
        "max_attempts": 3,
        "attempt_ordinal_label": f"{attempt}/3",
        "llm_call_count": 2,
        "tokens_prompt": 4,
        "tokens_completion": 5,
        "tokens_total": 9,
        "retry_history_budget_chars": 120,
        "retry_history_budget_applied": False,
        "prompt_xxh3_128": None,
        "prompt_path": None,
        "evidence_xxh3_128": None,
        "evidence_path": None,
        "token_usage_available": True,
        "adaptive_mode": "none",
        "adaptive_helper_invoked": False,
        "adaptive_helper_materialized_context": False,
        "adaptive_helper_decision_reason": "not_adaptive_condition",
        "adaptive_helper_failure_kind": "",
        "adaptive_helper_retry_count": 0,
        "adaptive_helper_transport_retry_count": 0,
        "adaptive_helper_calls_used_after_attempt": 0,
        "adaptive_helper_need_more_context": None,
        "adaptive_helper_budget_exhausted": False,
        "adaptive_helper_saturation_guarded": False,
        "adaptive_online_policy_mode": "off",
        "adaptive_online_policy_algo": "linucb",
        "adaptive_budget_policy_mode": "off",
        "adaptive_budget_policy_algo": "linucb",
        "adaptive_budget_policy_architecture": "tier_binary",
        "stage_a_request_success": None,
        "stage_b_prompt_chars": None,
        "retry_history_chars": 0,
        "retry_history_turns_included": 0,
        "retry_history_truncated": False,
        "timing_breakdown": {},
        "token_breakdown": {},
        "syntax_preflight_enabled": False,
        "syntax_preflight_cycles_used": 0,
        "syntax_preflight_failures": 0,
        "patch_verifier_recovery_eligible": False,
        "patch_verifier_recovery_invoked": False,
        "patch_verifier_recovery_result": "not_eligible",
    }
    if attempt_metadata_extra:
        attempt_metadata.update(attempt_metadata_extra)

    record = {
        "schema_version": ar.EVAL_RECORD_SCHEMA_VERSION,
        "run_id": run_id,
        "case_id": case_id,
        "condition": condition,
        "attempt": attempt,
        "success": success,
        "category": category,
        "difficulty": difficulty,
        "snapshot_dependency": snapshot_dependency,
        "run_metadata": {
            "run_id": run_id,
            "started_at_utc": "2026-02-10T00:00:00Z",
            "deterministic": False,
            "seed": None,
            "python_version": "3.12.0",
            "platform": "test",
            "llmdebug_version": "2.0.0",
            "patcher_name": "command",
            "patcher_config_fingerprint": "abc123",
            "run_settings_fingerprint": "runfp123",
        },
        "attempt_metadata": attempt_metadata,
        "model_metadata": {"backend": "command", "name": "command"},
        "cumulative_runtime_sec": cumulative_runtime_sec,
        "attempt_wall_clock_sec": attempt_wall_clock_sec,
        "context_protocol": context_protocol,
        "note": note,
    }
    if patch_meta_extra is not None:
        record["patch_meta"] = patch_meta_extra
    return record


def test_analyze_scalar_and_schema_helpers() -> None:
    record = _record(
        attempt_metadata_extra={
            "stage_a_request_success": True,
            "stage_b_prompt_chars": 123,
        }
    )
    ar._validate_eval_record_schema(record, src="synthetic.jsonl", line_no=1)

    missing_run_id = dict(record)
    missing_run_id.pop("run_id")
    with pytest.raises(ValueError, match="missing required field 'run_id'"):
        ar._validate_eval_record_schema(missing_run_id, src="synthetic.jsonl", line_no=2)

    bad_version = dict(record)
    bad_version["schema_version"] = 999
    with pytest.raises(ValueError, match="unsupported schema_version"):
        ar._validate_eval_record_schema(bad_version, src="synthetic.jsonl", line_no=3)

    with pytest.raises(ValueError, match="must be string\\|null"):
        ar._validate_optional_string_field(
            123, path="attempt_metadata.prompt_path", src="s", line_no=4
        )

    assert ar._metadata_with_fallback({"foo": "attempt"}, {"foo": "patch"}, key="foo") == "attempt"
    assert ar._metadata_with_fallback({}, {"foo": "patch"}, key="foo") == "patch"
    assert ar._as_float_or_none("1.25") == pytest.approx(1.25)
    assert ar._as_float_or_none(float("inf")) is None
    assert ar._sanitize_propensity(None) == 1.0
    assert ar._sanitize_propensity(0.0, floor=0.2) == 0.2
    assert ar._coerce_feature_values({"a": "2", "b": float("inf"), "c": 3}) == {
        "a": 2.0,
        "c": 3.0,
    }
    assert ar._feature_scaling_profile_bucket("  ") == "unknown"
    assert ar._feature_scaling_profile_bucket("v1") == "v1"
    assert ar._looks_like_test_path("tests/foo.py")
    assert ar._looks_like_test_path("src/foo_test.py")
    assert ar._looks_like_test_path("src/conftest.py")
    assert not ar._looks_like_test_path("src/foo.py")
    assert ar._is_external_baseline_condition("ldb")
    assert not ar._is_external_baseline_condition("base_repair")


def test_analyze_count_and_breakdown_helpers() -> None:
    core_records = [
        _record(
            run_id="run-counts",
            case_id="case-1",
            attempt=1,
            success=False,
            context_protocol="staged",
            cumulative_runtime_sec=1.0,
            attempt_wall_clock_sec=1.0,
            note="syntax_preflight_failed",
            attempt_metadata_extra={
                "max_attempts": 4,
                "llm_call_count": 2,
                "tokens_prompt": 5,
                "tokens_completion": 7,
                "tokens_total": 12,
                "stage_a_request_success": True,
                "stage_b_prompt_chars": 100,
                "retry_history_chars": 10,
                "retry_history_turns_included": 1,
                "retry_history_budget_applied": True,
                "adaptive_mode": "gated",
                "adaptive_helper_invoked": True,
                "adaptive_helper_materialized_context": False,
                "adaptive_helper_decision_reason": "llm_invalid_response_no_helper",
                "adaptive_helper_failure_kind": "transport",
                "adaptive_helper_retry_count": 2,
                "adaptive_helper_transport_retry_count": 1,
                "adaptive_helper_calls_used_after_attempt": 1,
                "adaptive_helper_need_more_context": True,
                "adaptive_helper_budget_exhausted": True,
                "adaptive_online_policy_mode": "gated",
                "adaptive_online_decision_id": "online-1",
                "adaptive_online_action_selected": "invoke_helper",
                "adaptive_online_action_applied": "skip_helper",
                "adaptive_online_behavior_action": "invoke_helper",
                "adaptive_online_updated": True,
                "adaptive_online_reward_scope": "helper_credit_v2",
                "adaptive_online_reward": 0.5,
                "adaptive_online_reward_progress_score": 0.25,
                "adaptive_progress_reason": "new_evidence_added",
                "adaptive_online_propensity": 0.8,
                "adaptive_online_local_tokens_total": 11,
                "adaptive_online_local_wall_sec": 1.5,
                "adaptive_online_local_llm_call_count": 2,
                "adaptive_online_reward_delayed_rescue_bonus": 0.5,
                "adaptive_online_reward_credit_horizon_attempts": 1,
                "adaptive_online_feature_vector": {"x": 1.0},
                "adaptive_online_feature_scaling_profile": "v1",
                "adaptive_budget_policy_mode": "on",
                "adaptive_budget_policy_algo": "linucb",
                "adaptive_budget_policy_architecture": "tier_binary",
                "adaptive_budget_decision_id": "budget-1",
                "adaptive_budget_action_selected": "increase_budget",
                "adaptive_budget_action_applied": "keep_budget",
                "adaptive_budget_family_selected": "time",
                "adaptive_budget_family_applied": "tokens",
                "adaptive_budget_reward_scope": "budget_credit_v2",
                "adaptive_budget_reward": 0.3,
                "adaptive_budget_reward_progress_score": 0.2,
                "adaptive_budget_propensity": 0.7,
                "adaptive_budget_family_propensity": 0.6,
                "adaptive_budget_request_budget_exceeded": True,
                "adaptive_budget_applied_multiplier": 1.5,
                "adaptive_budget_default_multiplier": 1.0,
                "adaptive_budget_applied_controls_changed": {
                    "propose_budget": True,
                    "max_tokens": True,
                    "helper_allowance": False,
                    "stage_a_requests": True,
                    "stage_a_chars": False,
                    "retry_history_budget": True,
                    "compaction_allowance": False,
                    "response_mode": True,
                },
                "adaptive_budget_partial_override": True,
                "adaptive_budget_local_tokens_total": 13,
                "adaptive_budget_local_wall_sec": 1.7,
                "adaptive_budget_local_llm_call_count": 3,
                "adaptive_budget_effective_helper_max_calls": 2,
                "adaptive_budget_effective_stage_a_max_requests": 4,
                "adaptive_budget_effective_stage_a_max_chars": 500,
                "adaptive_budget_effective_retry_history_max_chars": 200,
                "adaptive_budget_effective_context_compaction_max_calls": 1,
                "adaptive_budget_effective_openai_max_tokens": 400,
                "syntax_preflight_enabled": True,
                "syntax_preflight_cycles_used": 2,
                "syntax_preflight_failures": 1,
                "patch_verifier_recovery_eligible": True,
                "patch_verifier_recovery_invoked": True,
                "patch_verifier_recovery_result": "passed_verifier",
                "adaptive_budget_topup_granted": True,
                "baseline_metrics": {
                    "usage_available": True,
                    "llm_call_count": 2,
                    "teacher_duration_sec_total": 1.5,
                    "stopped_reason": "first",
                    "debug_added_value": True,
                    "test_feedback_final_fail_count": 3,
                },
                "timing_breakdown": {
                    "stage_a_llm_sec": 1.0,
                    "stage_b_llm_sec": 2.0,
                    "llm_total_sec": 3.0,
                },
                "token_breakdown": {
                    "stage_a_tokens_prompt": 10,
                    "stage_b_tokens_total": 20,
                    "snapshot_chars": 5,
                },
            },
            patch_meta_extra={
                "retry_gate_evaluated": True,
                "retry_gate_blocked": False,
                "patch_verifier_enabled": True,
                "patch_verifier_source_scope_matched": True,
                "patch_verifier_passed": True,
                "baseline_metrics": {
                    "usage_available": True,
                    "llm_call_count": 2,
                    "teacher_duration_sec_total": 1.5,
                    "stopped_reason": "first",
                    "debug_added_value": True,
                    "test_feedback_final_fail_count": 3,
                },
            },
        ),
        _record(
            run_id="run-counts",
            case_id="case-2",
            attempt=2,
            success=True,
            cumulative_runtime_sec=2.0,
            attempt_wall_clock_sec=2.0,
            note="syntax_preflight_failed",
            attempt_metadata_extra={
                "max_attempts": 4,
                "llm_call_count": 1,
                "tokens_prompt": 4,
                "tokens_completion": 6,
                "tokens_total": 10,
                "stage_a_request_success": False,
                "stage_b_prompt_chars": 200,
                "retry_history_chars": 20,
                "retry_history_turns_included": 2,
                "retry_history_budget_applied": False,
                "retry_history_truncated": True,
                "adaptive_mode": "llm_discretion",
                "adaptive_helper_invoked": True,
                "adaptive_helper_materialized_context": True,
                "adaptive_helper_decision_reason": "llm_requested_helper",
                "adaptive_helper_retry_count": 1,
                "adaptive_helper_calls_used_after_attempt": 2,
                "adaptive_helper_need_more_context": False,
                "adaptive_online_policy_mode": "llm_discretion",
                "adaptive_online_decision_id": "online-2",
                "adaptive_online_action_selected": "skip_helper",
                "adaptive_online_action_applied": "invoke_helper",
                "adaptive_online_behavior_action": "skip_helper",
                "adaptive_online_updated": False,
                "adaptive_online_reward_scope": "helper_credit_v2",
                "adaptive_online_reward": 0.4,
                "adaptive_online_reward_progress_score": 0.3,
                "adaptive_progress_reason": "pipeline_stage_advance",
                "adaptive_online_propensity": 0.6,
                "adaptive_online_local_tokens_total": 12,
                "adaptive_online_local_wall_sec": 1.2,
                "adaptive_online_local_llm_call_count": 1,
                "adaptive_online_reward_delayed_rescue_bonus": 0.25,
                "adaptive_online_reward_credit_horizon_attempts": 2,
                "adaptive_online_feature_vector": {"x": 2.0},
                "adaptive_online_feature_scaling_profile": "v2",
                "adaptive_budget_policy_mode": "on",
                "adaptive_budget_policy_algo": "linucb",
                "adaptive_budget_policy_architecture": "tier_binary",
                "adaptive_budget_decision_id": "budget-2",
                "adaptive_budget_action_selected": "increase_budget",
                "adaptive_budget_action_applied": "increase_budget",
                "adaptive_budget_family_selected": "tokens",
                "adaptive_budget_family_applied": "context",
                "adaptive_budget_reward_scope": "budget_credit_v2",
                "adaptive_budget_reward": 0.4,
                "adaptive_budget_reward_progress_score": 0.3,
                "adaptive_budget_propensity": 0.5,
                "adaptive_budget_family_propensity": 0.4,
                "adaptive_budget_request_budget_exceeded": False,
                "adaptive_budget_applied_multiplier": 2.0,
                "adaptive_budget_default_multiplier": 1.0,
                "adaptive_budget_applied_control_values": {
                    "helper_max_calls": 4,
                    "stage_a_max_requests": 5,
                    "stage_a_max_chars": 600,
                    "retry_history_max_chars": 250,
                    "context_compaction_max_calls": 2,
                    "openai_max_tokens": 500,
                },
                "adaptive_budget_applied_controls_changed": {
                    "propose_budget": True,
                    "max_tokens": False,
                    "helper_allowance": True,
                    "stage_a_requests": False,
                    "stage_a_chars": True,
                    "retry_history_budget": False,
                    "compaction_allowance": True,
                    "response_mode": False,
                },
                "adaptive_budget_partial_override": False,
                "adaptive_budget_local_tokens_total": 12,
                "adaptive_budget_local_wall_sec": 1.2,
                "adaptive_budget_local_llm_call_count": 1,
                "syntax_preflight_enabled": False,
                "syntax_preflight_cycles_used": 1,
                "syntax_preflight_failures": 0,
                "patch_verifier_recovery_eligible": True,
                "patch_verifier_recovery_invoked": True,
                "patch_verifier_recovery_result": "still_rejected",
                "baseline_metrics": {
                    "usage_available": False,
                    "llm_call_count": 1,
                    "teacher_duration_sec_total": 2.5,
                    "stopped_reason": "second",
                    "debug_added_value": False,
                    "test_feedback_final_fail_count": 4,
                },
                "timing_breakdown": {
                    "stage_a_llm_sec": 0.5,
                    "stage_b_llm_sec": 1.0,
                    "llm_total_sec": 2.0,
                },
                "token_breakdown": {
                    "stage_a_tokens_prompt": 8,
                    "stage_b_tokens_total": 15,
                    "snapshot_chars": 3,
                },
            },
            patch_meta_extra={
                "retry_gate_evaluated": True,
                "retry_gate_blocked": True,
                "patch_verifier_enabled": True,
                "patch_verifier_source_scope_matched": False,
                "patch_verifier_passed": False,
                "baseline_metrics": {
                    "usage_available": False,
                    "llm_call_count": 1,
                    "teacher_duration_sec_total": 2.5,
                    "stopped_reason": "second",
                    "debug_added_value": False,
                    "test_feedback_final_fail_count": 4,
                },
            },
        ),
        _record(
            run_id="run-counts",
            case_id="case-3",
            attempt=3,
            success=False,
            cumulative_runtime_sec=3.0,
            attempt_wall_clock_sec=3.0,
            attempt_metadata_extra={
                "max_attempts": 4,
                "llm_call_count": 3,
                "tokens_prompt": 6,
                "tokens_completion": 5,
                "tokens_total": 11,
                "stage_a_request_success": True,
                "stage_b_prompt_chars": 150,
                "retry_history_chars": 5,
                "retry_history_turns_included": 0,
                "adaptive_mode": "gated",
                "adaptive_helper_invoked": True,
                "adaptive_helper_materialized_context": False,
                "adaptive_helper_decision_reason": "llm_requested_helper",
                "adaptive_helper_failure_kind": "backend_unavailable",
                "adaptive_helper_retry_count": 3,
                "adaptive_helper_transport_retry_count": 2,
                "adaptive_helper_calls_used_after_attempt": 4,
                "adaptive_helper_need_more_context": False,
                "adaptive_helper_saturation_guarded": True,
                "adaptive_online_policy_mode": "gated",
                "adaptive_online_decision_id": "online-3",
                "adaptive_online_action_selected": "none",
                "adaptive_online_action_applied": "none",
                "adaptive_online_behavior_action": "none",
                "adaptive_online_updated": True,
                "adaptive_online_reward_scope": "helper_credit_v2",
                "adaptive_online_reward": 0.6,
                "adaptive_online_reward_progress_score": 0.1,
                "adaptive_progress_reason": "pytest_failure_count_reduction",
                "adaptive_online_propensity": 0.5,
                "adaptive_online_local_tokens_total": 10,
                "adaptive_online_local_wall_sec": 1.0,
                "adaptive_online_local_llm_call_count": 3,
                "adaptive_online_reward_credit_horizon_attempts": 0,
                "adaptive_online_feature_vector": {"x": 3.0},
                "adaptive_online_feature_scaling_profile": "v1",
                "adaptive_budget_policy_mode": "on",
                "adaptive_budget_policy_algo": "linucb",
                "adaptive_budget_policy_architecture": "tier_binary",
                "adaptive_budget_decision_id": "budget-3",
                "adaptive_budget_action_selected": "keep_budget",
                "adaptive_budget_action_applied": "keep_budget",
                "adaptive_budget_family_selected": "context",
                "adaptive_budget_family_applied": "time",
                "adaptive_budget_reward_scope": "budget_credit_v2",
                "adaptive_budget_reward": 0.2,
                "adaptive_budget_reward_progress_score": 0.1,
                "adaptive_budget_propensity": 0.9,
                "adaptive_budget_family_propensity": 0.8,
                "adaptive_budget_request_budget_exceeded": True,
                "adaptive_budget_applied_multiplier": 1.0,
                "adaptive_budget_default_multiplier": 1.0,
                "adaptive_budget_applied_control_values": {
                    "helper_max_calls": 5,
                    "stage_a_max_requests": 6,
                    "stage_a_max_chars": 700,
                    "retry_history_max_chars": 300,
                    "context_compaction_max_calls": 3,
                    "openai_max_tokens": 600,
                },
                "adaptive_budget_partial_override": False,
                "adaptive_budget_local_tokens_total": 11,
                "adaptive_budget_local_wall_sec": 1.0,
                "adaptive_budget_local_llm_call_count": 2,
                "syntax_preflight_enabled": True,
                "syntax_preflight_cycles_used": 0,
                "syntax_preflight_failures": 0,
                "patch_verifier_recovery_eligible": False,
                "patch_verifier_recovery_invoked": False,
                "patch_verifier_recovery_result": "not_eligible",
                "timing_breakdown": {
                    "stage_a_llm_sec": 0.5,
                    "stage_b_llm_sec": 0.5,
                    "llm_total_sec": 1.0,
                },
                "token_breakdown": {
                    "stage_a_tokens_prompt": 7,
                    "stage_b_tokens_total": 9,
                    "snapshot_chars": 2,
                },
            },
            patch_meta_extra={
                "retry_gate_blocked": True,
                "patch_verifier_enabled": False,
            },
        ),
    ]

    special_records = [
        _record(
            run_id="run-counts",
            case_id="case-invalid",
            attempt=1,
            success=False,
            attempt_metadata_extra={
                "adaptive_mode": "gated",
                "adaptive_helper_invoked": True,
                "adaptive_helper_materialized_context": False,
                "adaptive_helper_decision_reason": "llm_invalid_response_no_helper",
                "adaptive_helper_failure_kind": "invalid_payload",
                "adaptive_helper_retry_count": 1,
                "adaptive_helper_transport_retry_count": 1,
                "adaptive_helper_calls_used_after_attempt": 4,
                "adaptive_helper_need_more_context": False,
            },
        ),
        _record(
            run_id="run-counts",
            case_id="case-budget",
            attempt=2,
            success=False,
            attempt_metadata_extra={
                "adaptive_mode": "gated",
                "adaptive_helper_invoked": True,
                "adaptive_helper_materialized_context": False,
                "adaptive_helper_decision_reason": "llm_requested_helper",
                "adaptive_helper_failure_kind": "request_budget_exceeded",
                "adaptive_helper_retry_count": 0,
                "adaptive_helper_transport_retry_count": 0,
                "adaptive_helper_calls_used_after_attempt": 4,
                "adaptive_helper_need_more_context": True,
                "adaptive_helper_budget_exhausted": True,
                "adaptive_helper_saturation_guarded": True,
            },
        ),
        _record(
            run_id="run-counts",
            case_id="case-semantic",
            attempt=3,
            success=False,
            attempt_metadata_extra={
                "adaptive_mode": "gated",
                "adaptive_helper_invoked": True,
                "adaptive_helper_materialized_context": False,
                "adaptive_helper_decision_reason": "llm_requested_helper",
                "adaptive_helper_failure_kind": "",
                "adaptive_helper_retry_count": 0,
                "adaptive_helper_transport_retry_count": 0,
                "adaptive_helper_calls_used_after_attempt": 4,
                "adaptive_helper_need_more_context": False,
            },
        ),
    ]

    assert ar._context_protocol_from_records(core_records) == "staged"
    assert ar._stage_a_request_counts(core_records) == (3, 2)
    assert ar._retry_gate_counts(core_records) == (3, 2)
    assert ar._adaptive_budget_topup_attempt_metrics(core_records) == {"topup_attempts": 1}
    syntax = ar._syntax_preflight_attempt_metrics(core_records)
    assert syntax["enabled_attempts"] == 2
    assert syntax["failed_attempts"] == 2
    verifier = ar._patch_verifier_source_scope_metrics(core_records)
    assert verifier["verifier_attempts"] == 2
    assert verifier["source_scope_matches"] == 1
    assert verifier["recovery_passed_attempts"] == 1
    assert verifier["recovery_result_counts"] == {
        "passed_verifier": 1,
        "still_rejected": 1,
    }

    helper = ar._adaptive_helper_attempt_metrics(core_records + special_records)
    assert helper["adaptive_mode"] == "gated"
    assert helper["helper_invocation_attempts"] == 6
    assert helper["helper_materialized_attempts"] == 1
    assert helper["helper_low_yield_attempts"] == 1
    assert helper["helper_transport_failure_attempts"] == 1
    assert helper["helper_backend_unavailable_attempts"] == 1
    assert helper["helper_invalid_payload_attempts"] == 1
    assert helper["helper_request_budget_exceeded_attempts"] == 1
    assert helper["helper_need_more_true_attempts"] == 2
    assert helper["helper_need_more_false_attempts"] == 4
    assert helper["helper_budget_exhausted_attempts"] == 2
    assert helper["helper_saturation_guard_attempts"] == 2
    assert helper["helper_decision_attempts"] == 6
    assert helper["helper_calls_used_final"] == 4

    online = ar._adaptive_online_attempt_metrics(core_records)
    assert online["decisions"] == 3
    assert online["override_attempts"] == 2
    assert online["selected_action_counts"]["invoke_helper"] == 1
    assert online["selected_action_counts"]["skip_helper"] == 1
    assert online["behavior_action_counts"]["invoke_helper"] == 1
    assert online["credited_rescues_plus_1"] == 1
    assert online["credited_rescues_plus_2"] == 1

    budget = ar._adaptive_budget_policy_attempt_metrics(core_records)
    assert budget["decisions"] == 3
    assert budget["selected_increase_attempts"] == 2
    assert budget["applied_increase_attempts"] == 1
    assert budget["selected_time_attempts"] == 1
    assert budget["selected_tokens_attempts"] == 1
    assert budget["selected_context_attempts"] == 1
    assert budget["request_budget_exceeded_attempts"] == 2
    assert budget["propose_budget_change_attempts"] == 2
    assert budget["effective_helper_max_calls_count"] == 3
    assert budget["effective_openai_max_tokens_count"] == 3

    timing_sum = ar._timing_breakdown_sum(core_records)
    token_sum = ar._token_breakdown_sum(core_records)
    assert timing_sum["llm_total_sec"] == pytest.approx(6.0)
    assert token_sum["stage_a_tokens_prompt"] == 25
    assert ar._timing_breakdown_mean(core_records)["llm_total_sec"] == pytest.approx(2.0)
    assert ar._token_breakdown_mean(core_records)["stage_a_tokens_prompt"] == pytest.approx(
        8.333333
    )

    assert ar._baseline_metrics_from_record(core_records[0])["usage_available"] is True
    assert ar._baseline_metric_last_bool(core_records, "debug_added_value") is False
    assert ar._baseline_metric_last_str(core_records, "stopped_reason") == "second"
    assert ar._baseline_metric_sum(core_records, "teacher_duration_sec_total") == pytest.approx(4.0)
    assert ar._baseline_metric_last_int(core_records, "test_feedback_final_fail_count") == 4

    assert ar._token_usage_breakdown(core_records) == (15.0, 18.0, 33.0)
    assert ar._attempts_budget_from_records(core_records) == 4
    assert ar._llm_calls_total(core_records) == 6
    assert ar._retry_history_attempt_metrics(core_records)["attempts"] == 3
    assert (
        ar._runtime_used([{"cumulative_runtime_sec": 1.0}, {"cumulative_runtime_sec": 3.5}]) == 3.5
    )
    assert (
        ar._runtime_used(
            [{"before_duration_sec": 1.0}, {"after_duration_sec": 2.0}, {"after_duration_sec": 3.0}]
        )
        == 6.0
    )

    rows = [
        {"condition": "base_repair", "stage_b_prompt_chars_mean": 100.0},
        {"condition": "base_repair", "stage_b_prompt_chars_mean": 200.0},
        {"condition": "cot_repair", "stage_b_prompt_chars_mean": 300.0},
        {"timing_breakdown_sum": timing_sum, "token_breakdown_sum": token_sum},
    ]
    assert ar._mean_prompt_chars_by_condition(rows)["base_repair"] == pytest.approx(150.0)
    assert ar._condition_timing_breakdown_mean(rows)["llm_total_sec"] == pytest.approx(6.0)
    assert ar._condition_token_breakdown_mean(rows)["stage_a_tokens_prompt"] == pytest.approx(25.0)


def test_analyze_ope_and_statistics_helpers(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    samples = ar._adaptive_ope_samples(
        records=[
            _record(
                run_id="run-ope",
                case_id="case-1",
                attempt=1,
                success=False,
                attempt_metadata_extra={
                    "adaptive_mode": "gated",
                    "adaptive_online_decision_id": "online-1",
                    "adaptive_online_reward_scope": "helper_credit_v2",
                    "adaptive_online_reward": 0.5,
                    "adaptive_online_behavior_action": "invoke_helper",
                    "adaptive_online_propensity": 0.8,
                    "adaptive_online_feature_vector": {"x": 1.0},
                    "adaptive_online_feature_scaling_profile": "v1",
                },
            ),
            _record(
                run_id="run-ope",
                case_id="case-2",
                attempt=2,
                success=False,
                attempt_metadata_extra={
                    "adaptive_mode": "gated",
                    "adaptive_online_decision_id": "online-2",
                    "adaptive_online_reward_scope": "helper_credit_v2",
                    "adaptive_online_reward": 0.4,
                    "adaptive_online_behavior_action": "skip_helper",
                    "adaptive_online_propensity": 0.6,
                    "adaptive_online_feature_vector": {"x": 2.0},
                    "adaptive_online_feature_scaling_profile": "v1",
                },
            ),
            _record(
                run_id="run-ope",
                case_id="case-3",
                attempt=3,
                success=True,
                attempt_metadata_extra={
                    "adaptive_mode": "gated",
                    "adaptive_online_decision_id": "online-3",
                    "adaptive_online_reward_scope": "helper_credit_v2",
                    "adaptive_online_reward": 0.6,
                    "adaptive_online_behavior_action": "none",
                    "adaptive_online_propensity": 0.5,
                    "adaptive_online_feature_vector": {"x": 3.0},
                    "adaptive_online_feature_scaling_profile": "v1",
                },
            ),
        ],
        prefix="adaptive_online",
        valid_actions={"invoke_helper", "skip_helper", "none"},
    )
    assert len(samples) == 3

    ope_summary = ar._adaptive_ope_summary_from_records(
        records=[
            _record(
                run_id="run-ope",
                case_id="case-1",
                attempt=1,
                success=False,
                attempt_metadata_extra={
                    "adaptive_mode": "gated",
                    "adaptive_online_decision_id": "online-1",
                    "adaptive_online_reward_scope": "helper_credit_v2",
                    "adaptive_online_reward": 0.5,
                    "adaptive_online_behavior_action": "invoke_helper",
                    "adaptive_online_propensity": 0.8,
                    "adaptive_online_feature_vector": {"x": 1.0},
                    "adaptive_online_feature_scaling_profile": "v1",
                },
            ),
            _record(
                run_id="run-ope",
                case_id="case-2",
                attempt=2,
                success=False,
                attempt_metadata_extra={
                    "adaptive_mode": "gated",
                    "adaptive_online_decision_id": "online-2",
                    "adaptive_online_reward_scope": "helper_credit_v2",
                    "adaptive_online_reward": 0.4,
                    "adaptive_online_behavior_action": "skip_helper",
                    "adaptive_online_propensity": 0.6,
                    "adaptive_online_feature_vector": {"x": 2.0},
                    "adaptive_online_feature_scaling_profile": "v2",
                },
            ),
            _record(
                run_id="run-ope",
                case_id="case-3",
                attempt=3,
                success=True,
                attempt_metadata_extra={
                    "adaptive_mode": "gated",
                    "adaptive_online_decision_id": "online-3",
                    "adaptive_online_reward_scope": "helper_credit_v2",
                    "adaptive_online_reward": 0.6,
                    "adaptive_online_behavior_action": "none",
                    "adaptive_online_propensity": 0.5,
                    "adaptive_online_feature_vector": {"x": 3.0},
                    "adaptive_online_feature_scaling_profile": "v1",
                },
            ),
        ],
        prefix="adaptive_online",
        action_space={"invoke_helper", "skip_helper", "none"},
        target_policies={
            "always_skip": "skip_helper",
            "always_invoke_helper": "invoke_helper",
        },
        settings=OpeSummarySettings(
            min_action_support=1,
            propensity_floor=0.2,
        ),
    )
    assert ope_summary["sample_count"] == 3
    assert ope_summary["action_counts"]["invoke_helper"] == 1
    assert ope_summary["action_rates"]["invoke_helper"] == pytest.approx(1.0 / 3.0)
    assert ope_summary["action_entropy"] is not None
    assert ope_summary["feature_scaling_profiles"] == ["v1", "v2"]
    assert ope_summary["target_order"] == ["always_skip", "always_invoke_helper"]
    assert ope_summary["targets"]["always_skip"]["support_warning"] == "weak_behavior_support"
    assert ope_summary["targets"]["always_skip"]["estimator_agreement"] == "agree"

    feature_names, models = ar._fit_ridge_action_models(
        samples=samples,
        action_space={"invoke_helper", "skip_helper", "none"},
        min_action_support=1,
        ridge_alpha=0.1,
    )
    assert feature_names == ["x"]
    assert set(models) == {"invoke_helper", "skip_helper", "none"}
    assert math.isfinite(
        ar._predict_action_reward(
            features={"x": 1.5},
            feature_names=feature_names,
            beta=models["invoke_helper"],
        )
    )

    # codex 2-3#2: folds are assigned by case, so replicas must stand for distinct cases.
    distinct_case_samples = [
        {**sample, "case_key": f"{sample['case_key']}-copy{copy_idx}"}
        for copy_idx in range(3)
        for sample in samples
    ]
    dr, reason = ar._stratified_doubly_robust_estimate(
        samples=distinct_case_samples,  # enough cases per stratum for cross-fitting (STA-11)
        target_action="invoke_helper",
        action_space={"invoke_helper", "skip_helper", "none"},
        min_action_support=1,
        ridge_alpha=0.1,
        propensity_floor=0.2,
    )
    assert reason == ""
    assert dr is not None
    assert ar._stratified_doubly_robust_estimate(
        samples=samples,
        target_action="invoke_helper",
        action_space={"invoke_helper", "skip_helper", "none"},
        min_action_support=1,
        ridge_alpha=0.1,
        propensity_floor=0.2,
    ) == (None, "insufficient_crossfit_support")
    assert ar._stratified_doubly_robust_estimate(
        samples=[],
        target_action="invoke_helper",
        action_space={"invoke_helper"},
        min_action_support=1,
        ridge_alpha=0.1,
        propensity_floor=0.2,
    ) == (None, "no_samples")

    unsupported = ar._deterministic_target_ope(
        samples=samples,
        target_action="missing",
        action_counts={"invoke_helper": 1, "skip_helper": 1, "none": 1},
        settings=ar.TargetOpeSettings(
            action_space={"invoke_helper", "skip_helper", "none"},
            min_action_support=1,
            ridge_alpha=0.1,
            propensity_floor=0.2,
        ),
    )
    assert unsupported["dr_supported"] is False
    assert unsupported["dr_reason"] == "target_action_out_of_space"
    assert unsupported["support_warning"] == "no_behavior_support"
    assert unsupported["estimator_agreement"] == "insufficient_estimators"

    expanded = ar._expand_paths([str(tmp_path / "*.jsonl"), str(tmp_path / "missing.jsonl")])
    (tmp_path / "result.jsonl").write_text("{}", encoding="utf-8")
    assert str(tmp_path / "missing.jsonl") in expanded
    assert str(tmp_path / "result.jsonl") in ar._expand_paths([str(tmp_path / "*.jsonl")])

    assert ar._fmt_num(None) == "n/a"
    assert ar._fmt_num(1.0) == "1"
    assert ar._fmt_num(1.25) == "1.25"
    assert ar._fmt_signed_percent(0.125) == "+12.5%"
    assert ar._fmt_percent(0.125) == "12.5%"
    assert ar._mean([1.0, 3.0]) == pytest.approx(2.0)
    assert ar._stddev([1.0, 3.0]) == pytest.approx(1.0)
    assert ar._quantile([1.0, 2.0, 3.0], 0.5) == pytest.approx(2.0)
    assert ar._ratio(3, 4) == pytest.approx(0.75)
    assert ar._median([1, 2, 3, 4]) == pytest.approx(2.5)

    summary = {
        "overall": {
            "base_repair": {
                "cases": 2,
                "successes": 1,
                "success_rate": 0.5,
                "median_attempts_to_success": 1,
                "mean_tokens_total": 3.0,
                "mean_tokens_prompt": 1.0,
                "mean_tokens_completion": 2.0,
                "mean_tokens_per_attempt": 3.0,
                "mean_llm_calls_per_case": 2.0,
                "mean_attempts_budget": 2.0,
                "mean_attempt_ratio_when_successful": 0.5,
                "mean_runtime_sec": 1.5,
                "patch_attempts": 1,
                "test_touch_attempts": 1,
                "test_touch_attempt_rate": 1.0,
                "helper_invoked_cases": 1,
                "helper_invocation_case_rate": 0.5,
                "helper_invocation_attempt_rate": 0.5,
                "helper_materialized_cases": 1,
                "helper_materialized_case_rate": 0.5,
                "helper_materialized_attempt_rate": 0.5,
                "helper_low_yield_attempt_rate": 0.0,
                "helper_transport_failure_attempt_rate": 0.0,
                "helper_backend_unavailable_attempt_rate": 0.0,
                "helper_compat_reasoning_only_attempt_rate": 0.0,
                "helper_invalid_payload_attempt_rate": 0.0,
                "helper_request_budget_exceeded_attempt_rate": 0.0,
                "helper_effective_success_rate": 1.0,
                "helper_success_when_invoked": 1.0,
                "helper_rescue_rate": 0.5,
                "helper_mean_retries_per_invocation": 1.0,
                "helper_mean_transport_retries_per_invocation": 0.5,
                "helper_need_more_true_rate": 0.5,
                "helper_budget_exhausted_rate": 0.0,
                "helper_saturation_guard_rate": 0.0,
                "adaptive_budget_decisions": 1,
                "adaptive_budget_selected_increase_rate": 0.5,
                "adaptive_budget_applied_increase_rate": 0.5,
                "adaptive_budget_update_rate": 0.5,
                "adaptive_budget_mean_reward": 0.2,
                "adaptive_budget_mean_applied_multiplier": 1.5,
                "adaptive_budget_request_budget_exceeded_rate": 0.5,
                "adaptive_online_ope": {
                    "targets": {
                        "always_invoke_helper": {
                            "ips": 1.0,
                            "snips": 1.1,
                            "dr": 1.2,
                            "effective_sample_size": 2.0,
                            "matched_samples": 1,
                            "total_samples": 2,
                        }
                    }
                },
                "adaptive_budget_ope": {
                    "targets": {
                        "always_increase_budget": {
                            "ips": 1.0,
                            "snips": 1.0,
                            "dr": 1.0,
                            "effective_sample_size": 1.0,
                            "matched_samples": 1,
                            "total_samples": 1,
                        },
                        "always_keep_budget": {
                            "ips": 0.5,
                            "snips": 0.5,
                            "dr": 0.5,
                            "effective_sample_size": 1.0,
                            "matched_samples": 1,
                            "total_samples": 1,
                        },
                    }
                },
                "solve_on_attempt_distribution": {"1": 1, "2": 1},
                "pairwise_uplifts": [
                    {
                        "condition_a": "base_repair",
                        "condition_b": "cot_repair",
                        "n_pairs": 2,
                        "uplift_success_rate": 0.5,
                        "mcnemar_p_value": 0.04,
                        "mcnemar_significant": True,
                    }
                ],
                "uplift_base_vs_layered": 0.1,
                "uplift_staged_vs_monolithic": 0.2,
                "uplift_layered_gated_vs_base": 0.3,
                "uplift_layered_llm_vs_base": 0.4,
            },
            "pairwise_uplifts": [
                {
                    "condition_a": "base_repair",
                    "condition_b": "cot_repair",
                    "n_pairs": 2,
                    "uplift_success_rate": 0.5,
                    "mcnemar_p_value": 0.04,
                    "mcnemar_significant": True,
                }
            ],
            "uplift_base_vs_layered": 0.1,
            "uplift_staged_vs_monolithic": 0.2,
            "uplift_layered_gated_vs_base": 0.3,
            "uplift_layered_llm_vs_base": 0.4,
            "test_touch_attempts": 1,
            "patch_attempts": 1,
        },
        "by_category": {"logic_error": {}},
        "by_difficulty": {"easy": {}},
        "by_snapshot_dependency": {"required": {}},
        "by_bug_source": {"hand_crafted": {}},
        "by_contamination_risk": {"low": {}},
        "input_errors": {
            "files_unreadable": 1,
            "json_lines_invalid": 0,
            "records_skipped": 0,
            "duplicate_attempt_rows_seen": 1,
            "duplicate_attempt_rows_resolved": 1,
        },
    }
    ar._print_summary(summary)
    printed = capsys.readouterr().out
    assert "Overall:" in printed
    assert "By category:" in printed
    assert "pairwise_uplifts:" in printed
    assert "Input diagnostics:" in printed

    stats = {
        "aggregation": "per_run",
        "condition_a": "base_repair",
        "condition_b": "cot_repair",
        "n_pairs": 2,
        "a_success_rate": 0.25,
        "b_success_rate": 0.75,
        "overall_mcnemar": {
            "n_01": 1,
            "n_10": 2,
            "p_value": 0.04,
            "method": "exact",
            "significant": True,
        },
        "overall_cohens_h": {"h": 0.5, "magnitude": "medium"},
        "overall_bootstrap_ci": {"confidence": 0.95, "ci_lower": 0.1, "ci_upper": 0.6},
    }
    ar._print_stats(stats)
    ar._print_stats({"error": "no pairs"})
    printed = capsys.readouterr().out
    assert "Statistical analysis (aggregation=per_run):" in printed
    assert "Statistics: no pairs" in printed


def test_analyze_main_rejects_invalid_ope_arguments(capsys: pytest.CaptureFixture[str]) -> None:
    assert ar.main(["--adaptive-ope-min-action-support", "0", "missing.jsonl"]) == 2
    assert "must be >= 1" in capsys.readouterr().out
    assert ar.main(["--adaptive-ope-propensity-floor", "1.5", "missing.jsonl"]) == 2
    assert "must be in (0, 1]" in capsys.readouterr().out


def test_reasoning_token_usage_breakdown_requires_full_case_coverage() -> None:
    covered_records = [
        _record(
            attempt=1,
            attempt_metadata_extra={
                "llm_call_count": 1,
                "reasoning_token_usage_available": True,
                "tokens_reasoning": 3,
            },
        ),
        _record(
            attempt=2,
            attempt_metadata_extra={
                "llm_call_count": 1,
                "reasoning_token_usage_available": True,
                "tokens_reasoning": 5,
            },
        ),
    ]
    assert ar._reasoning_token_usage_breakdown(covered_records) == (8.0, True, 4.0)

    partial_records = [
        _record(
            attempt=1,
            attempt_metadata_extra={
                "llm_call_count": 1,
                "reasoning_token_usage_available": True,
                "tokens_reasoning": 3,
            },
        ),
        _record(
            attempt=2,
            attempt_metadata_extra={
                "llm_call_count": 1,
                "reasoning_token_usage_available": False,
            },
        ),
    ]
    assert ar._reasoning_token_usage_breakdown(partial_records) == (None, False, None)


def _tokens(total: int) -> dict[str, object]:
    return {
        "tokens_prompt": total // 2,
        "tokens_completion": total - total // 2,
        "tokens_total": total,
    }


def test_sta09_partial_token_coverage_divides_by_covered_attempts() -> None:
    """STA-09: covered attempts were summed but divided by all executed attempts."""
    records = [
        _record(attempt=1, attempt_metadata_extra=_tokens(100)),
        _record(attempt=2, attempt_metadata_extra={"token_usage_available": False}),
    ]
    [row] = ar._derive_case_condition_rows({("run", "case", "tracecoder"): records})

    assert row["token_usage"] == 100.0
    assert row["token_usage_per_attempt_mean"] == 100.0  # was 50.0
    assert row["token_covered_attempts"] == 1
    assert row["token_covered_fraction"] == 0.5


def test_sta01_uncovered_case_has_no_tokens_and_is_excluded_from_means() -> None:
    covered = [_record(attempt=1, success=True, attempt_metadata_extra=_tokens(10))]
    missing = [
        _record(
            attempt=1,
            success=True,
            case_id="c2",
            attempt_metadata_extra={"token_usage_available": False},
        )
    ]
    rows = ar._derive_case_condition_rows(
        {("run", "case", "tracecoder"): covered, ("run", "c2", "tracecoder"): missing}
    )
    assert [row["token_usage"] for row in rows] == [None, 10.0]  # rows are sorted by case

    stats = ar._summarize_bucket(rows)["tracecoder"]
    assert stats["mean_token_usage"] == 10.0  # the uncovered case used to count as 0
    assert stats["token_covered_cases"] == 1
    assert stats["token_covered_fraction"] == 0.5


def _ope_record(case_id: str, action: str, reward: float, propensity: float | None) -> dict:
    extra: dict[str, object] = {
        "adaptive_mode": "gated",
        "adaptive_online_decision_id": f"d-{case_id}",
        "adaptive_online_reward_scope": "helper_credit_v2",
        "adaptive_online_reward": reward,
        "adaptive_online_behavior_action": action,
        "adaptive_online_feature_vector": {"x": 1.0},
        "adaptive_online_feature_scaling_profile": "v1",
    }
    if propensity is not None:
        extra["adaptive_online_propensity"] = propensity
    return _record(case_id=case_id, attempt=1, attempt_metadata_extra=extra)


def test_sta12_missing_propensity_is_dropped_and_counted() -> None:
    """STA-12: a missing propensity silently became 1.0 in IPS, SNIPS and DR."""
    records = [
        _ope_record("c1", "invoke_helper", 1.0, 0.5),
        _ope_record("c2", "invoke_helper", 0.0, None),
        _ope_record("c3", "skip_helper", 0.2, 0.5),
    ]
    samples = ar._adaptive_ope_samples(
        records=records, prefix="adaptive_online", valid_actions={"invoke_helper", "skip_helper"}
    )
    assert [sample["propensity_missing"] for sample in samples] == [False, True, False]

    summary = ar._adaptive_ope_summary(
        samples=samples,
        action_space={"invoke_helper", "skip_helper"},
        target_policies={"always_invoke": "invoke_helper"},
        settings=OpeSummarySettings(
            min_action_support=1,
            propensity_floor=0.01,
        ),
    )
    assert summary["sample_count"] == 2
    assert summary["propensity_missing_dropped"] == 1
    assert summary["action_counts"]["invoke_helper"] == 1
