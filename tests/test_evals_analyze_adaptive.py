"""Tests for adaptive-policy metrics derivation in analyze_results."""

from __future__ import annotations

import pytest

from evals.analyze_results import (
    ADAPTIVE_OPE_PROPENSITY_FLOOR_DEFAULT,
    _adaptive_budget_policy_attempt_metrics,
    _derive_case_condition_rows,
    _load_and_group_records,
    _records_by_case_condition,
    _summarize_bucket,
)
from llmdebug import _json as json


def _record(
    *,
    run_id: str,
    case_id: str,
    attempt: int,
    success: bool,
    note: str = "tests_still_failing",
    helper_invoked: bool,
    helper_reason: str,
    helper_materialized_context: bool | None = None,
    helper_outcome: str | None = None,
    helper_failure_kind: str = "",
    helper_retry_count: int = 0,
    helper_transport_retry_count: int = 0,
    online_reward_scope: str = "helper_local_v1",
    budget_reward_scope: str = "budget_local_v1",
    online_progress_score: float = 0.0,
    online_progress_reason: str = "no_progress_v2",
    online_delayed_rescue_bonus: float = 0.0,
    online_credit_horizon_attempts: int = 0,
    patch_verifier_enabled: bool = False,
    patch_verifier_passed: bool | None = None,
    patch_verifier_source_scope_matched: bool = False,
    patch_verifier_recovery_eligible: bool = False,
    patch_verifier_recovery_invoked: bool = False,
    patch_verifier_recovery_result: str = "not_eligible",
) -> dict:
    if helper_materialized_context is None:
        helper_materialized_context = helper_invoked
    if helper_outcome is None:
        if helper_invoked and helper_materialized_context:
            helper_outcome = "materialized"
        elif helper_invoked and helper_reason == "llm_invalid_response_no_helper":
            helper_outcome = "invalid_payload"
        elif helper_invoked:
            helper_outcome = "semantic_no_context"
        elif helper_reason == "request_context_unavailable":
            helper_outcome = "skipped_unsupported"
        else:
            helper_outcome = "skipped_policy"
    helper_calls_after = 1 if helper_invoked else 0
    return {
        "schema_version": 5,
        "run_id": run_id,
        "case_id": case_id,
        "condition": "layered_adaptive_gated",
        "attempt": attempt,
        "success": success,
        "note": note,
        "category": "traceback_controls",
        "difficulty": "easy",
        "snapshot_dependency": "helpful",
        "run_metadata": {
            "run_id": run_id,
            "started_at_utc": "2026-02-10T00:00:00Z",
            "deterministic": False,
            "seed": None,
            "python_version": "3.12.0",
            "platform": "test",
            "llmdebug_version": "2.1.0",
            "patcher_name": "command",
            "patcher_config_fingerprint": "abc123",
            "run_settings_fingerprint": "runfp123",
        },
        "attempt_metadata": {
            "attempt": attempt,
            "max_attempts": 2,
            "attempt_ordinal_label": f"{attempt}/2",
            "llm_call_count": 1,
            "tokens_prompt": 0,
            "tokens_completion": 0,
            "tokens_total": 0,
            "retry_history_budget_chars": 12000,
            "retry_history_budget_applied": False,
            "prompt_sha256": "p",
            "prompt_path": None,
            "evidence_sha256": "e",
            "evidence_path": None,
            "adaptive_mode": "gated",
            "adaptive_policy": "balanced",
            "adaptive_helper_invoked": helper_invoked,
            "adaptive_helper_materialized_context": helper_materialized_context,
            "adaptive_helper_outcome": helper_outcome,
            "adaptive_helper_failure_kind": helper_failure_kind,
            "adaptive_helper_retry_count": helper_retry_count,
            "adaptive_helper_transport_retry_count": helper_transport_retry_count,
            "adaptive_helper_decision_reason": helper_reason,
            "adaptive_helper_need_more_context": True if helper_invoked else None,
            "adaptive_helper_calls_used_before_attempt": 0 if attempt == 1 else 1,
            "adaptive_helper_calls_used_after_attempt": helper_calls_after,
            "adaptive_helper_budget_max_calls": 0,
            "adaptive_helper_budget_unlimited": True,
            "adaptive_helper_budget_exhausted": False,
            "adaptive_helper_saturation_guarded": False,
            "adaptive_online_reward_scope": online_reward_scope,
            "adaptive_online_reward_progress_score": online_progress_score,
            "adaptive_online_reward_delayed_rescue_bonus": online_delayed_rescue_bonus,
            "adaptive_online_reward_credit_horizon_attempts": online_credit_horizon_attempts,
            "adaptive_progress_reason": online_progress_reason,
            "adaptive_online_local_llm_call_count": 0,
            "adaptive_online_local_tokens_total": 0,
            "adaptive_online_local_wall_sec": 0.0,
            "adaptive_budget_reward_scope": budget_reward_scope,
            "adaptive_budget_reward_progress_score": online_progress_score,
            "adaptive_budget_local_llm_call_count": 0,
            "adaptive_budget_local_tokens_total": 0,
            "adaptive_budget_local_wall_sec": 0.0,
            "patch_verifier_recovery_eligible": patch_verifier_recovery_eligible,
            "patch_verifier_recovery_invoked": patch_verifier_recovery_invoked,
            "patch_verifier_recovery_result": patch_verifier_recovery_result,
        },
        "model_metadata": {"backend": "command", "name": "command"},
        "patch_meta": {
            "adaptive_mode": "gated",
            "adaptive_helper_invoked": helper_invoked,
            "adaptive_helper_materialized_context": helper_materialized_context,
            "adaptive_helper_outcome": helper_outcome,
            "adaptive_helper_failure_kind": helper_failure_kind,
            "adaptive_helper_retry_count": helper_retry_count,
            "adaptive_helper_transport_retry_count": helper_transport_retry_count,
            "adaptive_helper_decision_reason": helper_reason,
            "adaptive_helper_need_more_context": True if helper_invoked else None,
            "adaptive_helper_calls_used_after_attempt": helper_calls_after,
            "adaptive_online_reward_scope": online_reward_scope,
            "adaptive_online_reward_progress_score": online_progress_score,
            "adaptive_online_reward_delayed_rescue_bonus": online_delayed_rescue_bonus,
            "adaptive_online_reward_credit_horizon_attempts": online_credit_horizon_attempts,
            "adaptive_progress_reason": online_progress_reason,
            "adaptive_budget_reward_scope": budget_reward_scope,
            "adaptive_budget_reward_progress_score": online_progress_score,
            "patch_verifier_enabled": patch_verifier_enabled,
            "patch_verifier_passed": patch_verifier_passed,
            "patch_verifier_source_scope_matched": patch_verifier_source_scope_matched,
            "patch_verifier_recovery_eligible": patch_verifier_recovery_eligible,
            "patch_verifier_recovery_invoked": patch_verifier_recovery_invoked,
            "patch_verifier_recovery_result": patch_verifier_recovery_result,
        },
        "cumulative_runtime_sec": float(attempt),
    }


def test_analyze_adaptive_helper_metrics(tmp_path) -> None:
    records = [
        _record(
            run_id="run_adapt",
            case_id="case_a",
            attempt=1,
            success=False,
            helper_invoked=False,
            helper_reason="gate_balanced_waiting",
        ),
        _record(
            run_id="run_adapt",
            case_id="case_a",
            attempt=2,
            success=True,
            helper_invoked=True,
            helper_reason="gate_balanced_repeated_signature",
        ),
        _record(
            run_id="run_adapt",
            case_id="case_b",
            attempt=1,
            success=False,
            helper_invoked=False,
            helper_reason="gate_balanced_waiting",
        ),
        _record(
            run_id="run_adapt",
            case_id="case_b",
            attempt=2,
            success=False,
            helper_invoked=True,
            helper_reason="gate_balanced_repeated_signature",
            helper_materialized_context=False,
        ),
    ]
    jsonl = tmp_path / "adaptive_results.jsonl"
    jsonl.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    grouped, input_errors = _load_and_group_records([str(jsonl)])
    assert input_errors == {
        "files_unreadable": 0,
        "json_lines_invalid": 0,
        "records_skipped": 0,
        "records_schema_invalid": 0,
        "infra_event_rows_seen": 0,
        "infra_event_rows_skipped": 0,
        "duplicate_attempt_rows_seen": 0,
        "duplicate_attempt_rows_resolved": 0,
    }

    rows = _derive_case_condition_rows(grouped)
    assert len(rows) == 2
    row_map = {row["case_id"]: row for row in rows}
    assert row_map["case_a"]["helper_invoked_case"] is True
    assert row_map["case_a"]["helper_materialized_case"] is True
    assert row_map["case_a"]["helper_rescued"] is True
    assert row_map["case_b"]["helper_invoked_case"] is True
    assert row_map["case_b"]["helper_materialized_case"] is False
    assert row_map["case_b"]["helper_low_yield_attempts"] == 1
    assert row_map["case_b"]["helper_rescued"] is False

    overall = _summarize_bucket(rows)
    adaptive = overall["layered_adaptive_gated"]
    assert adaptive["helper_invoked_cases"] == 2
    assert adaptive["helper_invocation_case_rate"] == pytest.approx(1.0)
    assert adaptive["helper_invocation_attempt_rate"] == pytest.approx(0.5)
    assert adaptive["helper_materialized_cases"] == 1
    assert adaptive["helper_materialized_case_rate"] == pytest.approx(0.5)
    assert adaptive["helper_materialized_attempt_rate"] == pytest.approx(0.25)
    assert adaptive["helper_low_yield_attempt_rate"] == pytest.approx(0.25)
    assert adaptive["helper_success_when_invoked"] == pytest.approx(0.5)
    assert adaptive["helper_rescue_rate"] == pytest.approx(0.5)
    assert adaptive["helper_need_more_true_rate"] == pytest.approx(1.0)


def test_analyze_adaptive_helper_outcome_telemetry_rolls_up(tmp_path) -> None:
    records = [
        _record(
            run_id="run_helper_outcomes",
            case_id="case_transport",
            attempt=1,
            success=False,
            helper_invoked=True,
            helper_reason="llm_invalid_response_no_helper",
            helper_materialized_context=False,
            helper_outcome="transport_failed",
            helper_failure_kind="transport",
            helper_retry_count=1,
            helper_transport_retry_count=1,
        ),
        _record(
            run_id="run_helper_outcomes",
            case_id="case_materialized",
            attempt=1,
            success=True,
            helper_invoked=True,
            helper_reason="llm_requested_helper",
            helper_materialized_context=True,
            helper_outcome="materialized",
            helper_retry_count=2,
            helper_transport_retry_count=1,
        ),
        _record(
            run_id="run_helper_outcomes",
            case_id="case_invalid",
            attempt=1,
            success=False,
            helper_invoked=True,
            helper_reason="llm_invalid_response_no_helper",
            helper_materialized_context=False,
            helper_outcome="invalid_payload",
            helper_failure_kind="invalid_payload",
            helper_retry_count=1,
            helper_transport_retry_count=0,
        ),
        _record(
            run_id="run_helper_outcomes",
            case_id="case_backend",
            attempt=1,
            success=False,
            helper_invoked=True,
            helper_reason="llm_requested_helper",
            helper_materialized_context=False,
            helper_outcome="unavailable_backend",
            helper_failure_kind="backend_unavailable",
            helper_retry_count=1,
            helper_transport_retry_count=1,
        ),
        _record(
            run_id="run_helper_outcomes",
            case_id="case_request_budget",
            attempt=1,
            success=False,
            helper_invoked=True,
            helper_reason="llm_requested_helper",
            helper_materialized_context=False,
            helper_outcome="request_budget_exceeded",
            helper_failure_kind="request_budget_exceeded",
            helper_retry_count=0,
            helper_transport_retry_count=0,
        ),
    ]
    jsonl = tmp_path / "adaptive_helper_outcomes.jsonl"
    jsonl.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    grouped, _ = _load_and_group_records([str(jsonl)])
    rows = _derive_case_condition_rows(grouped)
    assert len(rows) == 5
    row_map = {row["case_id"]: row for row in rows}
    assert row_map["case_transport"]["helper_transport_failure_attempts"] == 1
    assert row_map["case_transport"]["helper_low_yield_attempts"] == 0
    assert row_map["case_transport"]["helper_effective_attempts"] == 0
    assert row_map["case_materialized"]["helper_effective_success_rate"] == pytest.approx(1.0)
    assert row_map["case_invalid"]["helper_invalid_payload_attempts"] == 1
    assert row_map["case_invalid"]["helper_low_yield_attempts"] == 0
    assert row_map["case_backend"]["helper_backend_unavailable_attempts"] == 1
    assert row_map["case_backend"]["helper_effective_attempts"] == 0
    assert row_map["case_request_budget"]["helper_request_budget_exceeded_attempts"] == 1
    assert row_map["case_request_budget"]["helper_low_yield_attempts"] == 0

    summary = _summarize_bucket(
        rows,
        records_by_case_condition=_records_by_case_condition(grouped),
    )
    adaptive = summary["layered_adaptive_gated"]
    assert adaptive["helper_low_yield_attempt_rate"] == pytest.approx(0.0)
    assert adaptive["helper_transport_failure_attempt_rate"] == pytest.approx(0.2)
    assert adaptive["helper_backend_unavailable_attempt_rate"] == pytest.approx(0.2)
    assert adaptive["helper_invalid_payload_attempt_rate"] == pytest.approx(0.2)
    assert adaptive["helper_request_budget_exceeded_attempt_rate"] == pytest.approx(0.2)
    assert adaptive["helper_effective_success_rate"] == pytest.approx(1.0 / 3.0)
    assert adaptive["helper_mean_retries_per_invocation"] == pytest.approx(1.0)
    assert adaptive["helper_mean_transport_retries_per_invocation"] == pytest.approx(0.6)


def test_analyze_adaptive_helper_compat_reasoning_only_rolls_up(tmp_path) -> None:
    records = [
        _record(
            run_id="run_helper_compat_reasoning_only",
            case_id="case_compat",
            attempt=1,
            success=False,
            helper_invoked=True,
            helper_reason="llm_requested_helper",
            helper_materialized_context=False,
            helper_outcome="compat_reasoning_only",
            helper_failure_kind="compat_reasoning_only",
            helper_retry_count=0,
            helper_transport_retry_count=0,
        )
    ]
    jsonl = tmp_path / "adaptive_helper_compat_reasoning_only.jsonl"
    jsonl.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    grouped, _ = _load_and_group_records([str(jsonl)])
    rows = _derive_case_condition_rows(grouped)
    assert len(rows) == 1
    row = rows[0]
    assert row["helper_compat_reasoning_only_attempts"] == 1
    assert row["helper_compat_reasoning_only_attempt_rate"] == pytest.approx(1.0)
    assert row["helper_low_yield_attempts"] == 0
    assert row["helper_effective_attempts"] == 0

    summary = _summarize_bucket(
        rows,
        records_by_case_condition=_records_by_case_condition(grouped),
    )
    adaptive = summary["layered_adaptive_gated"]
    assert adaptive["helper_compat_reasoning_only_attempts"] == 1
    assert adaptive["helper_compat_reasoning_only_attempt_rate"] == pytest.approx(1.0)
    assert adaptive["helper_low_yield_attempt_rate"] == pytest.approx(0.0)


def test_analyze_patch_verifier_recovery_rolls_up(tmp_path) -> None:
    records = [
        _record(
            run_id="run_verifier_recovery",
            case_id="case_pass",
            attempt=1,
            success=False,
            note="patch_apply_failed",
            helper_invoked=False,
            helper_reason="online_policy_skip_helper",
            patch_verifier_enabled=True,
            patch_verifier_passed=True,
            patch_verifier_source_scope_matched=True,
            patch_verifier_recovery_eligible=True,
            patch_verifier_recovery_invoked=True,
            patch_verifier_recovery_result="passed_verifier",
        ),
        _record(
            run_id="run_verifier_recovery",
            case_id="case_reject",
            attempt=1,
            success=False,
            note="patch_verifier_rejected",
            helper_invoked=False,
            helper_reason="online_policy_skip_helper",
            patch_verifier_enabled=True,
            patch_verifier_passed=False,
            patch_verifier_source_scope_matched=False,
            patch_verifier_recovery_eligible=True,
            patch_verifier_recovery_invoked=True,
            patch_verifier_recovery_result="still_rejected",
        ),
        _record(
            run_id="run_verifier_recovery",
            case_id="case_not_eligible",
            attempt=1,
            success=False,
            note="patch_verifier_rejected",
            helper_invoked=False,
            helper_reason="online_policy_skip_helper",
            patch_verifier_enabled=True,
            patch_verifier_passed=False,
            patch_verifier_source_scope_matched=False,
            patch_verifier_recovery_eligible=False,
            patch_verifier_recovery_invoked=False,
            patch_verifier_recovery_result="not_eligible",
        ),
    ]
    jsonl = tmp_path / "adaptive_verifier_recovery.jsonl"
    jsonl.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    grouped, _ = _load_and_group_records([str(jsonl)])
    rows = _derive_case_condition_rows(grouped)
    summary = _summarize_bucket(rows)
    adaptive = summary["layered_adaptive_gated"]

    assert adaptive["patch_verifier_source_scope_matches"] == 1
    assert adaptive["patch_verifier_recovery_eligible_attempts"] == 2
    assert adaptive["patch_verifier_recovery_invoked_attempts"] == 2
    assert adaptive["patch_verifier_recovery_passed_attempts"] == 1
    assert adaptive["patch_verifier_recovery_pass_rate"] == pytest.approx(0.5)
    assert adaptive["patch_verifier_recovery_result_counts"] == {
        "not_eligible": 1,
        "passed_verifier": 1,
        "still_rejected": 1,
    }
    assert adaptive["patch_verifier_recovery_terminal_note_counts"] == {
        "patch_apply_failed": 1,
        "patch_verifier_rejected": 1,
    }


def test_analyze_adaptive_v2_reward_scopes_and_progress_roll_up(tmp_path) -> None:
    records = [
        _record(
            run_id="run_reward_v2",
            case_id="case_credit",
            attempt=1,
            success=False,
            helper_invoked=True,
            helper_reason="llm_requested_helper",
            helper_materialized_context=True,
            helper_outcome="materialized",
            online_reward_scope="helper_credit_v2",
            budget_reward_scope="budget_credit_v2",
            online_progress_score=0.5,
            online_progress_reason="pytest_failure_count_reduction",
            online_delayed_rescue_bonus=0.625,
            online_credit_horizon_attempts=1,
        ),
        _record(
            run_id="run_reward_v2",
            case_id="case_local",
            attempt=1,
            success=False,
            helper_invoked=False,
            helper_reason="gate_balanced_waiting",
            online_reward_scope="helper_credit_v2",
            budget_reward_scope="budget_credit_v2",
            online_progress_score=0.25,
            online_progress_reason="new_evidence_added",
        ),
        _record(
            run_id="run_reward_v2",
            case_id="case_credit_h2",
            attempt=1,
            success=False,
            helper_invoked=True,
            helper_reason="llm_requested_helper",
            helper_materialized_context=True,
            helper_outcome="materialized",
            online_reward_scope="helper_credit_v2",
            budget_reward_scope="budget_credit_v2",
            online_progress_score=0.75,
            online_progress_reason="pipeline_stage_advance",
            online_delayed_rescue_bonus=0.3125,
            online_credit_horizon_attempts=2,
        ),
    ]
    records[0]["attempt_metadata"]["adaptive_online_decision_id"] = "online-credit"
    records[0]["patch_meta"]["adaptive_online_decision_id"] = "online-credit"
    records[0]["attempt_metadata"]["adaptive_budget_decision_id"] = "budget-credit"
    records[0]["patch_meta"]["adaptive_budget_decision_id"] = "budget-credit"
    records[1]["attempt_metadata"]["adaptive_online_decision_id"] = "online-local"
    records[1]["patch_meta"]["adaptive_online_decision_id"] = "online-local"
    records[1]["attempt_metadata"]["adaptive_budget_decision_id"] = "budget-local"
    records[1]["patch_meta"]["adaptive_budget_decision_id"] = "budget-local"
    records[2]["attempt_metadata"]["adaptive_online_decision_id"] = "online-credit-h2"
    records[2]["patch_meta"]["adaptive_online_decision_id"] = "online-credit-h2"
    records[2]["attempt_metadata"]["adaptive_budget_decision_id"] = "budget-credit-h2"
    records[2]["patch_meta"]["adaptive_budget_decision_id"] = "budget-credit-h2"
    jsonl = tmp_path / "adaptive_reward_v2.jsonl"
    jsonl.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    grouped, _ = _load_and_group_records([str(jsonl)])
    rows = _derive_case_condition_rows(grouped)
    summary = _summarize_bucket(
        rows,
        records_by_case_condition=_records_by_case_condition(grouped),
    )

    adaptive = summary["layered_adaptive_gated"]
    assert adaptive["adaptive_online_mean_progress_score"] == pytest.approx(0.5)
    assert adaptive["adaptive_budget_mean_progress_score"] == pytest.approx(0.5)
    assert adaptive["adaptive_online_delayed_rescue_bonus_count"] == 2
    assert adaptive["adaptive_online_delayed_rescue_bonus_mean"] == pytest.approx(0.46875)
    assert adaptive["adaptive_online_credited_rescues_plus_1"] == 1
    assert adaptive["adaptive_online_credited_rescues_plus_2"] == 1
    assert adaptive["adaptive_online_progress_reason_counts"] == {
        "new_evidence_added": 1,
        "pipeline_stage_advance": 1,
        "pytest_failure_count_reduction": 1,
    }


def test_analyze_adaptive_helper_failure_kind_fallback_preserves_low_yield_semantics(
    tmp_path,
) -> None:
    semantic = _record(
        run_id="run_helper_fallback",
        case_id="case_semantic",
        attempt=1,
        success=False,
        helper_invoked=True,
        helper_reason="gate_balanced_repeated_signature",
        helper_materialized_context=False,
    )
    transport = _record(
        run_id="run_helper_fallback",
        case_id="case_transport",
        attempt=1,
        success=False,
        helper_invoked=True,
        helper_reason="llm_requested_helper",
        helper_materialized_context=False,
        helper_failure_kind="transport",
        helper_retry_count=1,
        helper_transport_retry_count=1,
    )
    invalid = _record(
        run_id="run_helper_fallback",
        case_id="case_invalid",
        attempt=1,
        success=False,
        helper_invoked=True,
        helper_reason="llm_requested_helper",
        helper_materialized_context=False,
        helper_failure_kind="invalid_payload",
        helper_retry_count=1,
    )
    backend = _record(
        run_id="run_helper_fallback",
        case_id="case_backend",
        attempt=1,
        success=False,
        helper_invoked=True,
        helper_reason="llm_requested_helper",
        helper_materialized_context=False,
        helper_failure_kind="backend_unavailable",
        helper_retry_count=1,
        helper_transport_retry_count=1,
    )
    budget = _record(
        run_id="run_helper_fallback",
        case_id="case_request_budget",
        attempt=1,
        success=False,
        helper_invoked=True,
        helper_reason="llm_requested_helper",
        helper_materialized_context=False,
        helper_failure_kind="request_budget_exceeded",
    )
    compat = _record(
        run_id="run_helper_fallback",
        case_id="case_compat",
        attempt=1,
        success=False,
        helper_invoked=True,
        helper_reason="llm_requested_helper",
        helper_materialized_context=False,
        helper_failure_kind="compat_reasoning_only",
    )
    records = [semantic, transport, invalid, backend, budget, compat]
    for record in records:
        record["attempt_metadata"].pop("adaptive_helper_outcome", None)
        record["patch_meta"].pop("adaptive_helper_outcome", None)
    transport["attempt_metadata"].pop("adaptive_helper_failure_kind", None)
    backend["patch_meta"].pop("adaptive_helper_failure_kind", None)

    jsonl = tmp_path / "adaptive_helper_failure_kind_fallback.jsonl"
    jsonl.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    grouped, _ = _load_and_group_records([str(jsonl)])
    rows = _derive_case_condition_rows(grouped)
    assert len(rows) == 6
    row_map = {row["case_id"]: row for row in rows}
    assert row_map["case_semantic"]["helper_low_yield_attempts"] == 1
    assert row_map["case_transport"]["helper_transport_failure_attempts"] == 1
    assert row_map["case_transport"]["helper_low_yield_attempts"] == 0
    assert row_map["case_invalid"]["helper_invalid_payload_attempts"] == 1
    assert row_map["case_invalid"]["helper_low_yield_attempts"] == 0
    assert row_map["case_backend"]["helper_backend_unavailable_attempts"] == 1
    assert row_map["case_backend"]["helper_low_yield_attempts"] == 0
    assert row_map["case_request_budget"]["helper_request_budget_exceeded_attempts"] == 1
    assert row_map["case_request_budget"]["helper_low_yield_attempts"] == 0
    assert row_map["case_compat"]["helper_compat_reasoning_only_attempts"] == 1
    assert row_map["case_compat"]["helper_low_yield_attempts"] == 0

    summary = _summarize_bucket(
        rows,
        records_by_case_condition=_records_by_case_condition(grouped),
    )
    adaptive = summary["layered_adaptive_gated"]
    assert adaptive["helper_low_yield_attempt_rate"] == pytest.approx(1.0 / 6.0)
    assert adaptive["helper_transport_failure_attempt_rate"] == pytest.approx(1.0 / 6.0)
    assert adaptive["helper_backend_unavailable_attempt_rate"] == pytest.approx(1.0 / 6.0)
    assert adaptive["helper_invalid_payload_attempt_rate"] == pytest.approx(1.0 / 6.0)
    assert adaptive["helper_request_budget_exceeded_attempt_rate"] == pytest.approx(1.0 / 6.0)
    assert adaptive["helper_compat_reasoning_only_attempt_rate"] == pytest.approx(1.0 / 6.0)
    assert adaptive["helper_effective_attempts"] == 3
    assert adaptive["helper_effective_success_rate"] == pytest.approx(0.0)


def test_analyze_adaptive_decision_attempts_ignore_attempt_zero(tmp_path) -> None:
    run_id = "run_adapt_zero"
    records = [
        {
            "schema_version": 5,
            "run_id": run_id,
            "case_id": "case_zero",
            "condition": "layered_adaptive_gated",
            "attempt": 0,
            "success": True,
            "category": "traceback_controls",
            "difficulty": "easy",
            "snapshot_dependency": "helpful",
            "run_metadata": {
                "run_id": run_id,
                "started_at_utc": "2026-02-10T00:00:00Z",
                "deterministic": False,
                "seed": None,
                "python_version": "3.12.0",
                "platform": "test",
                "llmdebug_version": "2.1.0",
                "patcher_name": "command",
                "patcher_config_fingerprint": "abc123",
                "run_settings_fingerprint": "runfp123",
            },
            "attempt_metadata": {
                "attempt": 0,
                "max_attempts": 2,
                "attempt_ordinal_label": "0/2",
                "llm_call_count": 0,
                "tokens_prompt": 0,
                "tokens_completion": 0,
                "tokens_total": 0,
                "retry_history_budget_chars": 12000,
                "retry_history_budget_applied": False,
                "prompt_sha256": "p",
                "prompt_path": None,
                "evidence_sha256": "e",
                "evidence_path": None,
                "adaptive_mode": "gated",
                "adaptive_policy": "balanced",
                "adaptive_helper_invoked": False,
                "adaptive_helper_decision_reason": "",
                "adaptive_helper_need_more_context": None,
                "adaptive_helper_calls_used_before_attempt": 0,
                "adaptive_helper_calls_used_after_attempt": 0,
                "adaptive_helper_budget_max_calls": 0,
                "adaptive_helper_budget_unlimited": True,
                "adaptive_helper_budget_exhausted": False,
                "adaptive_helper_saturation_guarded": False,
            },
            "model_metadata": {"backend": "command", "name": "command"},
            "patch_meta": {"adaptive_mode": "gated"},
            "cumulative_runtime_sec": 0.01,
        }
    ]
    jsonl = tmp_path / "adaptive_zero.jsonl"
    jsonl.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    grouped, _ = _load_and_group_records([str(jsonl)])
    rows = _derive_case_condition_rows(grouped)
    assert len(rows) == 1
    assert rows[0]["helper_decision_attempts"] == 0

    overall = _summarize_bucket(rows)
    adaptive = overall["layered_adaptive_gated"]
    assert adaptive["helper_decision_attempts"] == 0


def test_analyze_adaptive_online_policy_metrics_roll_up(tmp_path) -> None:
    records = [
        _record(
            run_id="run_online_metrics",
            case_id="case_online_a",
            attempt=1,
            success=False,
            helper_invoked=True,
            helper_reason="llm_requested_helper",
        ),
        _record(
            run_id="run_online_metrics",
            case_id="case_online_a",
            attempt=2,
            success=True,
            helper_invoked=False,
            helper_reason="online_policy_skip_helper",
        ),
        _record(
            run_id="run_online_metrics",
            case_id="case_online_b",
            attempt=1,
            success=False,
            helper_invoked=False,
            helper_reason="request_context_unavailable",
        ),
    ]
    for record in records:
        record["condition"] = "layered_adaptive_llm"
    records[0]["attempt_metadata"].update(
        {
            "adaptive_online_policy_mode": "on",
            "adaptive_online_policy_algo": "linucb",
            "adaptive_online_decision_id": "d1",
            "adaptive_online_action_selected": "invoke_helper",
            "adaptive_online_action_applied": "invoke_helper",
            "adaptive_online_propensity": 0.7,
            "adaptive_online_reward_scope": "helper_local_v1",
            "adaptive_online_reward": -0.2,
            "adaptive_online_local_llm_call_count": 1,
            "adaptive_online_local_tokens_total": 40,
            "adaptive_online_local_wall_sec": 0.4,
            "adaptive_online_updated": True,
        }
    )
    records[0]["patch_meta"].update(records[0]["attempt_metadata"])
    records[1]["attempt_metadata"].update(
        {
            "adaptive_online_policy_mode": "on",
            "adaptive_online_policy_algo": "linucb",
            "adaptive_online_decision_id": "d2",
            "adaptive_online_action_selected": "skip_helper",
            "adaptive_online_action_applied": "skip_helper",
            "adaptive_online_propensity": 0.8,
            "adaptive_online_reward_scope": "helper_local_v1",
            "adaptive_online_reward": 0.9,
            "adaptive_online_local_llm_call_count": 0,
            "adaptive_online_local_tokens_total": 0,
            "adaptive_online_local_wall_sec": 0.0,
            "adaptive_online_updated": True,
        }
    )
    records[1]["patch_meta"].update(records[1]["attempt_metadata"])
    # No decision id on this attempt: reward/propensity should not affect online means.
    records[2]["attempt_metadata"].update(
        {
            "adaptive_online_policy_mode": "on",
            "adaptive_online_policy_algo": "linucb",
            "adaptive_online_decision_id": "",
            "adaptive_online_action_selected": "none",
            "adaptive_online_action_applied": "none",
            "adaptive_online_propensity": 0.1,
            "adaptive_online_reward_scope": "helper_local_v1",
            "adaptive_online_reward": 1.0,
            "adaptive_online_local_llm_call_count": 2,
            "adaptive_online_local_tokens_total": 999,
            "adaptive_online_local_wall_sec": 9.9,
            "adaptive_online_updated": False,
        }
    )
    records[2]["patch_meta"].update(records[2]["attempt_metadata"])

    jsonl = tmp_path / "adaptive_online_metrics.jsonl"
    jsonl.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    grouped, _ = _load_and_group_records([str(jsonl)])
    rows = _derive_case_condition_rows(grouped)
    assert len(rows) == 2
    row_map = {row["case_id"]: row for row in rows}
    assert row_map["case_online_a"]["adaptive_online_decisions"] == 2
    assert row_map["case_online_a"]["adaptive_online_selected_invoke_attempts"] == 1
    assert row_map["case_online_a"]["adaptive_online_applied_invoke_attempts"] == 1
    assert row_map["case_online_a"]["adaptive_online_updates"] == 2
    assert row_map["case_online_a"]["adaptive_online_reward_sum"] == pytest.approx(0.7)

    summary = _summarize_bucket(
        rows,
        records_by_case_condition=_records_by_case_condition(grouped),
    )
    condition_summary = summary["layered_adaptive_llm"]
    assert condition_summary["adaptive_online_decisions"] == 2
    assert condition_summary["adaptive_online_selected_invoke_rate"] == pytest.approx(0.5)
    assert condition_summary["adaptive_online_applied_invoke_rate"] == pytest.approx(0.5)
    assert condition_summary["adaptive_online_update_rate"] == pytest.approx(1.0)
    assert condition_summary["adaptive_online_mean_reward"] == pytest.approx(0.35)
    assert condition_summary["adaptive_online_mean_propensity"] == pytest.approx(0.75)
    assert condition_summary["adaptive_online_mean_local_tokens_total"] == pytest.approx(20.0)
    assert condition_summary["adaptive_online_mean_local_wall_sec"] == pytest.approx(0.2)
    assert condition_summary["adaptive_online_mean_local_llm_call_count"] == pytest.approx(0.5)

    assert summary["adaptive_online_decisions"] == 2
    assert summary["adaptive_online_selected_invoke_rate"] == pytest.approx(0.5)
    assert summary["adaptive_online_applied_invoke_rate"] == pytest.approx(0.5)
    assert summary["adaptive_online_update_rate"] == pytest.approx(1.0)
    assert summary["adaptive_online_mean_reward"] == pytest.approx(0.35)
    assert summary["adaptive_online_mean_propensity"] == pytest.approx(0.75)
    assert summary["adaptive_online_mean_local_tokens_total"] == pytest.approx(20.0)
    assert summary["adaptive_online_mean_local_wall_sec"] == pytest.approx(0.2)


def test_analyze_adaptive_budget_policy_metrics_roll_up(tmp_path) -> None:
    records = [
        _record(
            run_id="run_budget_metrics",
            case_id="case_budget_a",
            attempt=1,
            success=False,
            helper_invoked=False,
            helper_reason="request_context_unavailable",
        ),
        _record(
            run_id="run_budget_metrics",
            case_id="case_budget_a",
            attempt=2,
            success=True,
            helper_invoked=False,
            helper_reason="request_context_unavailable",
        ),
        _record(
            run_id="run_budget_metrics",
            case_id="case_budget_b",
            attempt=1,
            success=False,
            helper_invoked=False,
            helper_reason="request_context_unavailable",
        ),
    ]
    for record in records:
        record["condition"] = "layered_adaptive_llm"
    records[0]["attempt_metadata"].update(
        {
            "adaptive_budget_policy_mode": "on",
            "adaptive_budget_policy_algo": "linucb",
            "adaptive_budget_policy_architecture": "hierarchical_v1",
            "adaptive_budget_decision_id": "b1",
            "adaptive_budget_action_selected": "increase_budget",
            "adaptive_budget_action_applied": "increase_budget",
            "adaptive_budget_family_selected": "tokens",
            "adaptive_budget_family_applied": "tokens",
            "adaptive_budget_propensity": 0.7,
            "adaptive_budget_family_propensity": 0.5,
            "adaptive_budget_reward_scope": "budget_local_v1",
            "adaptive_budget_reward": -0.4,
            "adaptive_budget_updated": True,
            "adaptive_budget_request_budget_exceeded": True,
            "adaptive_budget_applied_multiplier": 1.5,
            "adaptive_budget_default_multiplier": 1.0,
            "adaptive_budget_applied_controls_changed": {
                "propose_budget": True,
                "max_tokens": True,
                "helper_allowance": True,
                "stage_a_requests": True,
                "stage_a_chars": True,
                "retry_history_budget": True,
                "compaction_allowance": True,
                "response_mode": False,
            },
            "adaptive_budget_partial_override": True,
            "adaptive_budget_effective_helper_max_calls": 2,
            "adaptive_budget_effective_stage_a_max_requests": 7,
            "adaptive_budget_effective_stage_a_max_chars": 180000,
            "adaptive_budget_effective_retry_history_max_chars": 72000,
            "adaptive_budget_effective_context_compaction_max_calls": 2,
            "adaptive_budget_effective_openai_max_tokens": 600,
            "adaptive_budget_local_llm_call_count": 2,
            "adaptive_budget_local_tokens_total": 120,
            "adaptive_budget_local_wall_sec": 1.2,
        }
    )
    records[0]["patch_meta"].update(records[0]["attempt_metadata"])
    records[1]["attempt_metadata"].update(
        {
            "adaptive_budget_policy_mode": "on",
            "adaptive_budget_policy_algo": "linucb",
            "adaptive_budget_policy_architecture": "hierarchical_v1",
            "adaptive_budget_decision_id": "b2",
            "adaptive_budget_action_selected": "keep_budget",
            "adaptive_budget_action_applied": "keep_budget",
            "adaptive_budget_propensity": 0.8,
            "adaptive_budget_family_propensity": 1.0,
            "adaptive_budget_reward_scope": "budget_local_v1",
            "adaptive_budget_reward": 0.9,
            "adaptive_budget_updated": True,
            "adaptive_budget_request_budget_exceeded": False,
            "adaptive_budget_applied_multiplier": 1.0,
            "adaptive_budget_default_multiplier": 1.0,
            "adaptive_budget_applied_controls_changed": {
                "propose_budget": False,
                "max_tokens": False,
                "helper_allowance": False,
                "stage_a_requests": False,
                "stage_a_chars": False,
                "retry_history_budget": False,
                "compaction_allowance": False,
                "response_mode": False,
            },
            "adaptive_budget_partial_override": False,
            "adaptive_budget_effective_helper_max_calls": 1,
            "adaptive_budget_effective_stage_a_max_requests": 6,
            "adaptive_budget_effective_stage_a_max_chars": 120000,
            "adaptive_budget_effective_retry_history_max_chars": 48000,
            "adaptive_budget_effective_context_compaction_max_calls": 1,
            "adaptive_budget_effective_openai_max_tokens": 400,
            "adaptive_budget_local_llm_call_count": 0,
            "adaptive_budget_local_tokens_total": 40,
            "adaptive_budget_local_wall_sec": 0.4,
        }
    )
    records[1]["patch_meta"].update(records[1]["attempt_metadata"])
    records[2]["attempt_metadata"].update(
        {
            "adaptive_budget_policy_mode": "on",
            "adaptive_budget_policy_algo": "linucb",
            "adaptive_budget_policy_architecture": "hierarchical_v1",
            "adaptive_budget_decision_id": "",
            "adaptive_budget_action_selected": "none",
            "adaptive_budget_action_applied": "none",
            "adaptive_budget_propensity": 0.2,
            "adaptive_budget_reward_scope": "budget_local_v1",
            "adaptive_budget_reward": 1.0,
            "adaptive_budget_updated": False,
            "adaptive_budget_request_budget_exceeded": False,
            "adaptive_budget_applied_multiplier": 1.0,
            "adaptive_budget_local_llm_call_count": 5,
            "adaptive_budget_local_tokens_total": 999,
            "adaptive_budget_local_wall_sec": 9.9,
        }
    )
    records[2]["patch_meta"].update(records[2]["attempt_metadata"])

    jsonl = tmp_path / "adaptive_budget_metrics.jsonl"
    jsonl.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    grouped, _ = _load_and_group_records([str(jsonl)])
    rows = _derive_case_condition_rows(grouped)
    assert len(rows) == 2
    row_map = {row["case_id"]: row for row in rows}
    assert row_map["case_budget_a"]["adaptive_budget_decisions"] == 2
    assert row_map["case_budget_a"]["adaptive_budget_selected_increase_attempts"] == 1
    assert row_map["case_budget_a"]["adaptive_budget_applied_increase_attempts"] == 1
    assert row_map["case_budget_a"]["adaptive_budget_selected_tokens_attempts"] == 1
    assert row_map["case_budget_a"]["adaptive_budget_applied_tokens_attempts"] == 1
    assert row_map["case_budget_a"]["adaptive_budget_updates"] == 2
    assert row_map["case_budget_a"]["adaptive_budget_reward_sum"] == pytest.approx(0.5)
    assert row_map["case_budget_a"]["adaptive_budget_request_budget_exceeded_attempts"] == 1
    assert row_map["case_budget_a"]["adaptive_budget_max_tokens_change_attempts"] == 1
    assert row_map["case_budget_a"]["adaptive_budget_partial_override_attempts"] == 1

    summary = _summarize_bucket(
        rows,
        records_by_case_condition=_records_by_case_condition(grouped),
    )
    condition_summary = summary["layered_adaptive_llm"]
    assert condition_summary["adaptive_budget_decisions"] == 2
    assert condition_summary["adaptive_budget_selected_increase_rate"] == pytest.approx(0.5)
    assert condition_summary["adaptive_budget_applied_increase_rate"] == pytest.approx(0.5)
    assert condition_summary["adaptive_budget_selected_tokens_rate"] == pytest.approx(0.5)
    assert condition_summary["adaptive_budget_applied_tokens_rate"] == pytest.approx(0.5)
    assert condition_summary["adaptive_budget_update_rate"] == pytest.approx(1.0)
    assert condition_summary["adaptive_budget_mean_reward"] == pytest.approx(0.25)
    assert condition_summary["adaptive_budget_mean_propensity"] == pytest.approx(0.75)
    assert condition_summary["adaptive_budget_mean_family_propensity"] == pytest.approx(0.5)
    assert condition_summary["adaptive_budget_mean_applied_multiplier"] == pytest.approx(1.25)
    assert condition_summary["adaptive_budget_mean_local_tokens_total"] == pytest.approx(80.0)
    assert condition_summary["adaptive_budget_mean_local_wall_sec"] == pytest.approx(0.8)
    assert condition_summary["adaptive_budget_mean_local_llm_call_count"] == pytest.approx(1.0)
    assert condition_summary["adaptive_budget_max_tokens_change_rate"] == pytest.approx(0.5)
    assert condition_summary["adaptive_budget_partial_override_rate"] == pytest.approx(0.5)
    assert condition_summary[
        "adaptive_budget_mean_effective_stage_a_max_requests"
    ] == pytest.approx(6.5)
    assert condition_summary["adaptive_budget_mean_effective_openai_max_tokens"] == pytest.approx(
        500.0
    )
    assert condition_summary["adaptive_budget_request_budget_exceeded_rate"] == pytest.approx(
        1.0 / 3.0
    )

    assert summary["adaptive_budget_decisions"] == 2
    assert summary["adaptive_budget_selected_increase_rate"] == pytest.approx(0.5)
    assert summary["adaptive_budget_applied_increase_rate"] == pytest.approx(0.5)
    assert summary["adaptive_budget_selected_tokens_rate"] == pytest.approx(0.5)
    assert summary["adaptive_budget_update_rate"] == pytest.approx(1.0)
    assert summary["adaptive_budget_mean_reward"] == pytest.approx(0.25)
    assert summary["adaptive_budget_mean_propensity"] == pytest.approx(0.75)
    assert summary["adaptive_budget_mean_family_propensity"] == pytest.approx(0.5)
    assert summary["adaptive_budget_mean_local_tokens_total"] == pytest.approx(80.0)
    assert summary["adaptive_budget_mean_local_wall_sec"] == pytest.approx(0.8)
    assert summary["adaptive_budget_propose_budget_change_rate"] == pytest.approx(0.5)
    assert summary["adaptive_budget_mean_effective_helper_max_calls"] == pytest.approx(1.5)


def test_analyze_budget_exceeded_counts_only_decision_attempts() -> None:
    records = [
        {
            "attempt_metadata": {
                "adaptive_budget_policy_mode": "on",
                "adaptive_budget_policy_algo": "linucb",
                "adaptive_budget_policy_architecture": "hierarchical_v1",
                "adaptive_budget_decision_id": "",
                "adaptive_budget_request_budget_exceeded": True,
            },
            "patch_meta": {},
        },
        {
            "attempt_metadata": {
                "adaptive_budget_policy_mode": "on",
                "adaptive_budget_policy_algo": "linucb",
                "adaptive_budget_policy_architecture": "hierarchical_v1",
                "adaptive_budget_decision_id": "b1",
                "adaptive_budget_request_budget_exceeded": True,
            },
            "patch_meta": {},
        },
    ]
    metrics = _adaptive_budget_policy_attempt_metrics(records)
    assert metrics["decisions"] == 1
    assert metrics["request_budget_exceeded_attempts"] == 1


def test_analyze_budget_control_metrics_fall_back_to_legacy_multiplier_only() -> None:
    records = [
        {
            "attempt_metadata": {
                "adaptive_budget_policy_mode": "on",
                "adaptive_budget_policy_algo": "linucb",
                "adaptive_budget_policy_architecture": "hierarchical_v1",
                "adaptive_budget_decision_id": "b1",
                "adaptive_budget_action_selected": "increase_budget",
                "adaptive_budget_action_applied": "increase_budget",
                "adaptive_budget_family_selected": "time",
                "adaptive_budget_family_applied": "time",
                "adaptive_budget_applied_multiplier": 1.5,
                "adaptive_budget_default_multiplier": 1.0,
            },
            "patch_meta": {},
        }
    ]

    metrics = _adaptive_budget_policy_attempt_metrics(records)
    assert metrics["decisions"] == 1
    assert metrics["propose_budget_change_attempts"] == 1
    assert metrics["selected_time_attempts"] == 1
    assert metrics["applied_time_attempts"] == 1
    assert metrics["max_tokens_change_attempts"] == 0
    assert metrics["partial_override_attempts"] == 0


def test_analyze_budget_family_propensity_ignores_non_family_decisions(tmp_path) -> None:
    records = [
        _record(
            run_id="run_budget_family_propensity",
            case_id="case_budget_family_propensity",
            attempt=1,
            success=False,
            helper_invoked=False,
            helper_reason="request_context_unavailable",
        )
    ]
    records[0]["condition"] = "layered_adaptive_llm"
    records[0]["attempt_metadata"].update(
        {
            "adaptive_budget_policy_mode": "on",
            "adaptive_budget_policy_algo": "linucb",
            "adaptive_budget_policy_architecture": "hierarchical_v1",
            "adaptive_budget_decision_id": "b1",
            "adaptive_budget_action_selected": "keep_budget",
            "adaptive_budget_action_applied": "keep_budget",
            "adaptive_budget_propensity": 0.8,
            "adaptive_budget_family_propensity": 1.0,
            "adaptive_budget_reward_scope": "budget_local_v1",
            "adaptive_budget_reward": 0.1,
        }
    )
    records[0]["patch_meta"].update(records[0]["attempt_metadata"])

    jsonl = tmp_path / "adaptive_budget_no_family_propensity.jsonl"
    jsonl.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    grouped, _ = _load_and_group_records([str(jsonl)])
    rows = _derive_case_condition_rows(grouped)
    summary = _summarize_bucket(
        rows,
        records_by_case_condition=_records_by_case_condition(grouped),
    )

    assert summary["adaptive_budget_mean_family_propensity"] is None
    assert summary["layered_adaptive_llm"]["adaptive_budget_mean_family_propensity"] is None


def test_analyze_adaptive_rolls_up_control_metrics(tmp_path) -> None:
    records = [
        _record(
            run_id="run_controls",
            case_id="case_controls",
            attempt=1,
            success=False,
            helper_invoked=False,
            helper_reason="helper_budget_exhausted",
        ),
        _record(
            run_id="run_controls",
            case_id="case_controls",
            attempt=2,
            success=False,
            helper_invoked=False,
            helper_reason="helper_budget_exhausted",
        ),
    ]
    for record in records:
        record["condition"] = "layered_adaptive_llm"
    records[0]["note"] = "no_patch_proposed"
    records[1]["note"] = "syntax_preflight_failed"
    records[0]["attempt_metadata"].update(
        {
            "adaptive_budget_topup_granted": True,
            "syntax_preflight_enabled": True,
            "syntax_preflight_cycles_used": 2,
            "syntax_preflight_failures": 1,
        }
    )
    records[0]["patch_meta"].update(
        {
            "patch_verifier_enabled": True,
            "patch_verifier_source_scope_matched": True,
            **records[0]["attempt_metadata"],
        }
    )
    records[1]["attempt_metadata"].update(
        {
            "adaptive_budget_topup_granted": False,
            "syntax_preflight_enabled": True,
            "syntax_preflight_cycles_used": 2,
            "syntax_preflight_failures": 2,
        }
    )
    records[1]["patch_meta"].update(
        {
            "patch_verifier_enabled": True,
            "patch_verifier_source_scope_matched": False,
            **records[1]["attempt_metadata"],
        }
    )

    jsonl = tmp_path / "adaptive_controls_metrics.jsonl"
    jsonl.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    grouped, _ = _load_and_group_records([str(jsonl)])
    rows = _derive_case_condition_rows(grouped)
    assert len(rows) == 1
    row = rows[0]
    assert row["adaptive_budget_topup_attempts"] == 1
    assert row["adaptive_budget_topup_case"] is True
    assert row["syntax_preflight_cycles_sum"] == 4
    assert row["syntax_preflight_failures_sum"] == 3
    assert row["syntax_preflight_failed_attempts"] == 1
    assert row["patch_verifier_source_scope_matches"] == 1

    summary = _summarize_bucket(rows)
    condition_summary = summary["layered_adaptive_llm"]
    assert condition_summary["adaptive_budget_topup_case_rate"] == pytest.approx(1.0)
    assert condition_summary["adaptive_budget_topup_attempt_rate"] == pytest.approx(0.5)
    assert condition_summary["mean_syntax_preflight_cycles"] == pytest.approx(2.0)
    assert condition_summary["syntax_preflight_failures"] == 3
    assert condition_summary["syntax_preflight_failed_attempt_rate"] == pytest.approx(0.5)
    assert condition_summary["patch_verifier_source_scope_match_rate"] == pytest.approx(0.5)


def test_analyze_adaptive_off_policy_estimators_known_values(tmp_path) -> None:
    online_payloads = [
        ("invoke_helper", 0.5, 1.0),
        ("invoke_helper", 0.25, 0.5),
        ("skip_helper", 0.8, 0.2),
        ("skip_helper", 0.4, -0.1),
    ]
    budget_payloads = [
        ("increase_budget", 0.5, 0.4),
        ("increase_budget", 0.5, 0.2),
        ("keep_budget", 0.5, -0.2),
        ("keep_budget", 0.5, 0.0),
    ]
    records = []
    for attempt, (online_action, online_p, online_r) in enumerate(online_payloads, start=1):
        budget_action, budget_p, budget_r = budget_payloads[attempt - 1]
        record = _record(
            run_id="run_ope",
            case_id="case_ope",
            attempt=attempt,
            success=False,
            helper_invoked=online_action == "invoke_helper",
            helper_reason="llm_requested_helper",
        )
        record["condition"] = "layered_adaptive_llm"
        decision_payload = {
            "adaptive_online_policy_mode": "on",
            "adaptive_online_policy_algo": "linucb",
            "adaptive_online_decision_id": f"o{attempt}",
            "adaptive_online_action_selected": online_action,
            "adaptive_online_action_applied": online_action,
            "adaptive_online_behavior_action": online_action,
            "adaptive_online_propensity": online_p,
            "adaptive_online_behavior_propensity": online_p,
            "adaptive_online_reward_scope": "helper_local_v1",
            "adaptive_online_reward": online_r,
            "adaptive_online_feature_vector": {
                "bias": 1.0,
                "attempt_frac": float(attempt) / 4.0,
            },
            "adaptive_budget_policy_mode": "on",
            "adaptive_budget_policy_algo": "linucb",
            "adaptive_budget_decision_id": f"b{attempt}",
            "adaptive_budget_action_selected": budget_action,
            "adaptive_budget_action_applied": budget_action,
            "adaptive_budget_behavior_action": budget_action,
            "adaptive_budget_propensity": budget_p,
            "adaptive_budget_behavior_propensity": budget_p,
            "adaptive_budget_reward_scope": "budget_local_v1",
            "adaptive_budget_reward": budget_r,
            "adaptive_budget_feature_vector": {
                "bias": 1.0,
                "attempt_frac": float(attempt) / 4.0,
            },
        }
        record["attempt_metadata"].update(decision_payload)
        record["patch_meta"].update(decision_payload)
        records.append(record)

    jsonl = tmp_path / "adaptive_ope_metrics.jsonl"
    jsonl.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    grouped, _ = _load_and_group_records([str(jsonl)])
    rows = _derive_case_condition_rows(grouped)
    summary = _summarize_bucket(
        rows,
        records_by_case_condition=_records_by_case_condition(grouped),
    )
    condition_summary = summary["layered_adaptive_llm"]

    online_ope = condition_summary["adaptive_online_ope"]["targets"]
    invoke = online_ope["always_invoke_helper"]
    skip = online_ope["always_skip"]
    assert invoke["ips"] == pytest.approx(1.0)
    assert invoke["snips"] == pytest.approx(2.0 / 3.0)
    assert invoke["effective_sample_size"] == pytest.approx(1.8)
    assert isinstance(invoke["dr"], float)
    assert skip["ips"] == pytest.approx(0.0)
    assert skip["snips"] == pytest.approx(0.0)
    assert isinstance(skip["dr"], float)

    budget_ope = condition_summary["adaptive_budget_ope"]["targets"]
    increase = budget_ope["always_increase_budget"]
    keep = budget_ope["always_keep_budget"]
    assert increase["ips"] == pytest.approx(0.3)
    assert increase["snips"] == pytest.approx(0.3)
    assert keep["ips"] == pytest.approx(-0.1)
    assert keep["snips"] == pytest.approx(-0.1)
    assert isinstance(increase["dr"], float)
    assert isinstance(keep["dr"], float)


def test_analyze_adaptive_off_policy_dr_requires_min_support(tmp_path) -> None:
    records = []
    for attempt, reward in enumerate((0.5, 0.2), start=1):
        record = _record(
            run_id="run_ope_low_support",
            case_id="case_ope_low_support",
            attempt=attempt,
            success=False,
            helper_invoked=True,
            helper_reason="llm_requested_helper",
        )
        record["condition"] = "layered_adaptive_llm"
        payload = {
            "adaptive_online_policy_mode": "on",
            "adaptive_online_policy_algo": "linucb",
            "adaptive_online_decision_id": f"o{attempt}",
            "adaptive_online_action_selected": "invoke_helper",
            "adaptive_online_action_applied": "invoke_helper",
            "adaptive_online_behavior_action": "invoke_helper",
            "adaptive_online_propensity": 0.5,
            "adaptive_online_behavior_propensity": 0.5,
            "adaptive_online_reward_scope": "helper_local_v1",
            "adaptive_online_reward": reward,
            "adaptive_online_feature_vector": {"bias": 1.0},
        }
        record["attempt_metadata"].update(payload)
        record["patch_meta"].update(payload)
        records.append(record)

    jsonl = tmp_path / "adaptive_ope_low_support.jsonl"
    jsonl.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    grouped, _ = _load_and_group_records([str(jsonl)])
    rows = _derive_case_condition_rows(grouped)
    summary = _summarize_bucket(
        rows,
        records_by_case_condition=_records_by_case_condition(grouped),
    )
    condition_summary = summary["layered_adaptive_llm"]
    invoke = condition_summary["adaptive_online_ope"]["targets"]["always_invoke_helper"]

    assert invoke["dr"] is None
    assert invoke["dr_supported"] is False
    assert invoke["dr_reason"] == "insufficient_action_support"


def test_analyze_adaptive_off_policy_dr_support_threshold_override(tmp_path) -> None:
    records = []
    payloads = [
        ("invoke_helper", 0.5, 0.4),
        ("invoke_helper", 0.5, 0.2),
        ("skip_helper", 0.8, 0.1),
    ]
    for attempt, (action, propensity, reward) in enumerate(payloads, start=1):
        record = _record(
            run_id="run_ope_threshold",
            case_id="case_ope_threshold",
            attempt=attempt,
            success=False,
            helper_invoked=action == "invoke_helper",
            helper_reason="llm_requested_helper",
        )
        record["condition"] = "layered_adaptive_llm"
        payload = {
            "adaptive_online_policy_mode": "on",
            "adaptive_online_policy_algo": "linucb",
            "adaptive_online_decision_id": f"o{attempt}",
            "adaptive_online_action_selected": action,
            "adaptive_online_action_applied": action,
            "adaptive_online_behavior_action": action,
            "adaptive_online_propensity": propensity,
            "adaptive_online_behavior_propensity": propensity,
            "adaptive_online_reward_scope": "helper_local_v1",
            "adaptive_online_reward": reward,
            "adaptive_online_feature_vector": {"bias": 1.0},
        }
        record["attempt_metadata"].update(payload)
        record["patch_meta"].update(payload)
        records.append(record)

    jsonl = tmp_path / "adaptive_ope_threshold.jsonl"
    jsonl.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    grouped, _ = _load_and_group_records([str(jsonl)])
    rows = _derive_case_condition_rows(grouped)
    records_by_key = _records_by_case_condition(grouped)

    default_summary = _summarize_bucket(
        rows,
        records_by_case_condition=records_by_key,
    )
    invoke_default = default_summary["layered_adaptive_llm"]["adaptive_online_ope"]["targets"][
        "always_invoke_helper"
    ]
    assert invoke_default["dr"] is None
    assert invoke_default["dr_reason"] == "insufficient_action_support"

    relaxed_summary = _summarize_bucket(
        rows,
        records_by_case_condition=records_by_key,
        adaptive_ope_min_action_support=1,
    )
    invoke_relaxed = relaxed_summary["layered_adaptive_llm"]["adaptive_online_ope"]["targets"][
        "always_invoke_helper"
    ]
    assert isinstance(invoke_relaxed["dr"], float)
    assert invoke_relaxed["dr_supported"] is True
    assert relaxed_summary["layered_adaptive_llm"]["adaptive_online_ope"]["min_action_support"] == 1


def test_analyze_adaptive_off_policy_zero_support_target_ips_snips_none(tmp_path) -> None:
    records = []
    for attempt, reward in enumerate((0.5, 0.2), start=1):
        record = _record(
            run_id="run_ope_zero_support",
            case_id="case_ope_zero_support",
            attempt=attempt,
            success=False,
            helper_invoked=True,
            helper_reason="llm_requested_helper",
        )
        record["condition"] = "layered_adaptive_llm"
        payload = {
            "adaptive_online_policy_mode": "on",
            "adaptive_online_policy_algo": "linucb",
            "adaptive_online_decision_id": f"o{attempt}",
            "adaptive_online_action_selected": "invoke_helper",
            "adaptive_online_action_applied": "invoke_helper",
            "adaptive_online_behavior_action": "invoke_helper",
            "adaptive_online_propensity": 0.5,
            "adaptive_online_behavior_propensity": 0.5,
            "adaptive_online_reward_scope": "helper_local_v1",
            "adaptive_online_reward": reward,
            "adaptive_online_feature_vector": {"bias": 1.0},
        }
        record["attempt_metadata"].update(payload)
        record["patch_meta"].update(payload)
        records.append(record)

    jsonl = tmp_path / "adaptive_ope_zero_support.jsonl"
    jsonl.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    grouped, _ = _load_and_group_records([str(jsonl)])
    rows = _derive_case_condition_rows(grouped)
    summary = _summarize_bucket(
        rows,
        records_by_case_condition=_records_by_case_condition(grouped),
    )
    skip_target = summary["layered_adaptive_llm"]["adaptive_online_ope"]["targets"]["always_skip"]
    assert skip_target["matched_samples"] == 0
    assert skip_target["ips"] is None
    assert skip_target["snips"] is None


def test_analyze_adaptive_off_policy_dr_stratifies_by_feature_scaling_profile(tmp_path) -> None:
    records = []
    payloads = [
        ("v1", "invoke_helper", 0.5, 0.4, {"bias": 1.0, "attempt_frac": 0.2}),
        ("v1", "skip_helper", 0.6, 0.1, {"bias": 1.0, "attempt_frac": 0.8}),
        ("none", "invoke_helper", 0.7, 0.6, {"bias": 10.0, "attempt_frac": 2.0}),
        ("none", "skip_helper", 0.8, 0.2, {"bias": 10.0, "attempt_frac": 8.0}),
    ]
    for attempt, (profile, action, propensity, reward, features) in enumerate(payloads, start=1):
        record = _record(
            run_id="run_ope_mixed_profiles",
            case_id="case_ope_mixed_profiles",
            attempt=attempt,
            success=False,
            helper_invoked=action == "invoke_helper",
            helper_reason="llm_requested_helper",
        )
        record["condition"] = "layered_adaptive_llm"
        payload = {
            "adaptive_online_policy_mode": "on",
            "adaptive_online_policy_algo": "linucb",
            "adaptive_online_decision_id": f"o{attempt}",
            "adaptive_online_action_selected": action,
            "adaptive_online_action_applied": action,
            "adaptive_online_behavior_action": action,
            "adaptive_online_propensity": propensity,
            "adaptive_online_behavior_propensity": propensity,
            "adaptive_online_reward_scope": "helper_local_v1",
            "adaptive_online_reward": reward,
            "adaptive_online_feature_scaling_profile": profile,
            "adaptive_online_feature_vector": features,
        }
        record["attempt_metadata"].update(payload)
        record["patch_meta"].update(payload)
        records.append(record)

    jsonl = tmp_path / "adaptive_ope_mixed_profiles.jsonl"
    jsonl.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    grouped, _ = _load_and_group_records([str(jsonl)])
    rows = _derive_case_condition_rows(grouped)
    summary = _summarize_bucket(
        rows,
        records_by_case_condition=_records_by_case_condition(grouped),
        adaptive_ope_min_action_support=1,
    )
    online_ope = summary["layered_adaptive_llm"]["adaptive_online_ope"]
    invoke = online_ope["targets"]["always_invoke_helper"]

    assert online_ope["feature_scaling_profiles"] == ["none", "v1"]
    assert invoke["dr_supported"] is True
    assert isinstance(invoke["dr"], float)


def test_analyze_adaptive_online_action_counts_include_expanded_actions(tmp_path) -> None:
    records = []
    payloads = [
        ("invoke_helper", 0.5),
        ("file_only", 0.3),
        ("snapshot_lite", 0.2),
    ]
    for attempt, (action, reward) in enumerate(payloads, start=1):
        record = _record(
            run_id="run_action_counts",
            case_id=f"case_action_{attempt}",
            attempt=attempt,
            success=False,
            helper_invoked=action in {"invoke_helper", "file_only"},
            helper_reason="llm_requested_helper"
            if action != "snapshot_lite"
            else "policy_action_snapshot_lite",
            helper_materialized_context=action != "snapshot_lite",
            helper_outcome="materialized" if action == "snapshot_lite" else None,
        )
        record["condition"] = "layered_adaptive_llm"
        payload = {
            "adaptive_online_policy_mode": "on",
            "adaptive_online_policy_algo": "linucb",
            "adaptive_online_decision_id": f"o{attempt}",
            "adaptive_online_action_selected": action,
            "adaptive_online_action_applied": action,
            "adaptive_online_behavior_action": action,
            "adaptive_online_propensity": 0.5,
            "adaptive_online_behavior_propensity": 0.5,
            "adaptive_online_reward_scope": "helper_credit_v2",
            "adaptive_online_reward": reward,
            "adaptive_online_feature_vector": {"bias": 1.0},
            "adaptive_helper_action_selected": action,
            "adaptive_helper_action_executed": action,
            "adaptive_snapshot_lite_included": action == "snapshot_lite",
        }
        record["attempt_metadata"].update(payload)
        record["patch_meta"].update(payload)
        records.append(record)

    jsonl = tmp_path / "adaptive_action_counts.jsonl"
    jsonl.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    grouped, _ = _load_and_group_records([str(jsonl)])
    rows = _derive_case_condition_rows(grouped)
    summary = _summarize_bucket(
        rows,
        records_by_case_condition=_records_by_case_condition(grouped),
        adaptive_ope_min_action_support=1,
    )["layered_adaptive_llm"]

    selected_counts = summary["adaptive_online_selected_action_counts"]
    behavior_counts = summary["adaptive_online_behavior_action_counts"]

    assert selected_counts["invoke_helper"] == 1
    assert selected_counts["file_only"] == 1
    assert selected_counts["snapshot_lite"] == 1
    assert behavior_counts["snapshot_lite"] == 1
    assert summary["adaptive_online_ope"]["target_order"] == [
        "always_skip",
        "always_frame_only",
        "always_file_only",
        "always_invoke_helper",
    ]
    assert "always_skip" in summary["adaptive_online_ope"]["targets"]
    assert "always_file_only" in summary["adaptive_online_ope"]["targets"]
    assert "always_snapshot_lite" not in summary["adaptive_online_ope"]["targets"]


def test_analyze_adaptive_single_condition_rollup_matches_overall(tmp_path) -> None:
    records = [
        _record(
            run_id="run_rollup_equiv",
            case_id="case_equiv_a",
            attempt=1,
            success=False,
            helper_invoked=True,
            helper_reason="llm_requested_helper",
        ),
        _record(
            run_id="run_rollup_equiv",
            case_id="case_equiv_a",
            attempt=2,
            success=True,
            helper_invoked=False,
            helper_reason="online_policy_skip_helper",
        ),
    ]
    for index, record in enumerate(records, start=1):
        record["condition"] = "layered_adaptive_llm"
        payload = {
            "adaptive_online_policy_mode": "on",
            "adaptive_online_policy_algo": "linucb",
            "adaptive_online_decision_id": f"o{index}",
            "adaptive_online_action_selected": "invoke_helper" if index == 1 else "skip_helper",
            "adaptive_online_action_applied": "invoke_helper" if index == 1 else "skip_helper",
            "adaptive_online_behavior_action": "invoke_helper" if index == 1 else "skip_helper",
            "adaptive_online_propensity": 0.7 if index == 1 else 0.8,
            "adaptive_online_behavior_propensity": 0.7 if index == 1 else 0.8,
            "adaptive_online_reward_scope": "helper_local_v1",
            "adaptive_online_reward": -0.1 if index == 1 else 0.9,
            "adaptive_online_feature_vector": {"bias": 1.0},
        }
        record["attempt_metadata"].update(payload)
        record["patch_meta"].update(payload)

    jsonl = tmp_path / "adaptive_rollup_equivalence.jsonl"
    jsonl.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    grouped, _ = _load_and_group_records([str(jsonl)])
    rows = _derive_case_condition_rows(grouped)
    summary = _summarize_bucket(
        rows,
        records_by_case_condition=_records_by_case_condition(grouped),
    )
    condition_summary = summary["layered_adaptive_llm"]
    for key in (
        "adaptive_online_decisions",
        "adaptive_online_selected_invoke_attempts",
        "adaptive_online_selected_invoke_rate",
        "adaptive_online_mean_reward",
        "adaptive_online_mean_propensity",
        "adaptive_budget_decisions",
        "adaptive_budget_mean_reward",
        "helper_invocation_case_rate",
        "helper_invocation_attempt_rate",
    ):
        assert summary[key] == condition_summary[key]


def test_analyze_adaptive_ope_propensity_floor_trims_only_pathological_values(tmp_path) -> None:
    records = []
    payloads = [
        (0.00625, 0.3),
        (1e-9, 0.6),
    ]
    for attempt, (propensity, reward) in enumerate(payloads, start=1):
        record = _record(
            run_id="run_ope_propensity_floor",
            case_id="case_ope_propensity_floor",
            attempt=attempt,
            success=False,
            helper_invoked=True,
            helper_reason="llm_requested_helper",
        )
        record["condition"] = "layered_adaptive_llm"
        payload = {
            "adaptive_online_policy_mode": "on",
            "adaptive_online_policy_algo": "linucb",
            "adaptive_online_decision_id": f"o{attempt}",
            "adaptive_online_action_selected": "invoke_helper",
            "adaptive_online_action_applied": "invoke_helper",
            "adaptive_online_behavior_action": "invoke_helper",
            "adaptive_online_propensity": propensity,
            "adaptive_online_behavior_propensity": propensity,
            "adaptive_online_reward_scope": "helper_credit_v2",
            "adaptive_online_reward": reward,
            "adaptive_online_feature_vector": {"bias": 1.0},
        }
        record["attempt_metadata"].update(payload)
        record["patch_meta"].update(payload)
        records.append(record)

    jsonl = tmp_path / "adaptive_ope_propensity_floor.jsonl"
    jsonl.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    grouped, _ = _load_and_group_records([str(jsonl)])
    rows = _derive_case_condition_rows(grouped)
    summary = _summarize_bucket(
        rows,
        records_by_case_condition=_records_by_case_condition(grouped),
    )["layered_adaptive_llm"]
    online_ope = summary["adaptive_online_ope"]
    invoke = online_ope["targets"]["always_invoke_helper"]

    assert online_ope["propensity_floor"] == pytest.approx(ADAPTIVE_OPE_PROPENSITY_FLOOR_DEFAULT)
    assert online_ope["min_propensity_raw"] == pytest.approx(1e-9)
    assert online_ope["min_propensity_used"] == pytest.approx(ADAPTIVE_OPE_PROPENSITY_FLOOR_DEFAULT)
    assert online_ope["trimmed_sample_count"] == 1
    assert online_ope["max_importance_weight"] == pytest.approx(
        1.0 / ADAPTIVE_OPE_PROPENSITY_FLOOR_DEFAULT
    )
    assert invoke["ips"] == pytest.approx((160.0 * 0.3 + 200.0 * 0.6) / 2.0)
    assert invoke["snips"] == pytest.approx((160.0 * 0.3 + 200.0 * 0.6) / 360.0)


def test_analyze_adaptive_reward_distribution_diagnostics_roll_up(tmp_path) -> None:
    records = []
    payloads = [
        (0.0, 0.0),
        (0.5, 1.0),
        (1.0, 2.0),
    ]
    for attempt, (online_reward, budget_reward) in enumerate(payloads, start=1):
        record = _record(
            run_id="run_reward_distribution",
            case_id="case_reward_distribution",
            attempt=attempt,
            success=attempt == len(payloads),
            helper_invoked=attempt < len(payloads),
            helper_reason="llm_requested_helper",
        )
        record["condition"] = "layered_adaptive_llm"
        online_payload = {
            "adaptive_online_policy_mode": "on",
            "adaptive_online_policy_algo": "linucb",
            "adaptive_online_decision_id": f"o{attempt}",
            "adaptive_online_action_selected": "invoke_helper",
            "adaptive_online_action_applied": "invoke_helper",
            "adaptive_online_behavior_action": "invoke_helper",
            "adaptive_online_propensity": 0.5,
            "adaptive_online_behavior_propensity": 0.5,
            "adaptive_online_reward_scope": "helper_credit_v2",
            "adaptive_online_reward": online_reward,
            "adaptive_online_feature_vector": {"bias": 1.0},
        }
        budget_payload = {
            "adaptive_budget_policy_mode": "on",
            "adaptive_budget_policy_algo": "linucb",
            "adaptive_budget_policy_architecture": "tier_binary",
            "adaptive_budget_decision_id": f"b{attempt}",
            "adaptive_budget_action_selected": "keep_budget",
            "adaptive_budget_action_applied": "keep_budget",
            "adaptive_budget_behavior_action": "keep_budget",
            "adaptive_budget_propensity": 0.5,
            "adaptive_budget_behavior_propensity": 0.5,
            "adaptive_budget_reward_scope": "budget_credit_v2",
            "adaptive_budget_reward": budget_reward,
            "adaptive_budget_feature_vector": {"bias": 1.0},
        }
        record["attempt_metadata"].update(online_payload)
        record["attempt_metadata"].update(budget_payload)
        record["patch_meta"].update(online_payload)
        record["patch_meta"].update(budget_payload)
        records.append(record)

    jsonl = tmp_path / "adaptive_reward_distribution.jsonl"
    jsonl.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    grouped, _ = _load_and_group_records([str(jsonl)])
    rows = _derive_case_condition_rows(grouped)
    summary = _summarize_bucket(
        rows,
        records_by_case_condition=_records_by_case_condition(grouped),
    )["layered_adaptive_llm"]

    assert summary["adaptive_online_reward_min"] == pytest.approx(0.0)
    assert summary["adaptive_online_reward_p50"] == pytest.approx(0.5)
    assert summary["adaptive_online_reward_p90"] == pytest.approx(0.9)
    assert summary["adaptive_online_reward_max"] == pytest.approx(1.0)
    assert summary["adaptive_online_reward_std"] == pytest.approx(0.408248290463863)
    assert summary["adaptive_budget_reward_min"] == pytest.approx(0.0)
    assert summary["adaptive_budget_reward_p50"] == pytest.approx(1.0)
    assert summary["adaptive_budget_reward_p90"] == pytest.approx(1.8)
    assert summary["adaptive_budget_reward_max"] == pytest.approx(2.0)
    assert summary["adaptive_budget_reward_std"] == pytest.approx(0.816496580927726)
