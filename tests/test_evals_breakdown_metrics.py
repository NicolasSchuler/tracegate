"""Tests for timing_breakdown and token_breakdown eval metrics."""

from __future__ import annotations

from typing import Any

from evals.analyze_results import (
    _timing_breakdown_mean,
    _timing_breakdown_sum,
    _token_breakdown_mean,
    _token_breakdown_sum,
    _token_usage_breakdown,
)
from evals.eval_recording import _empty_attempt_metadata
from evals.report_run_metrics import _compute_overhead_analysis
from evals.run_eval import _apply_usage_totals_to_patch_meta, _finalize_breakdowns

# ── Helpers ──────────────────────────────────────────────────────────────


def _make_record(
    *,
    timing: dict[str, float] | None = None,
    tokens: dict[str, int] | None = None,
    patch_meta: dict[str, Any] | None = None,
    attempt_wall_clock_sec: float = 10.0,
    before_duration_sec: float = 2.0,
    after_duration_sec: float = 1.5,
    tokens_prompt: int = 100,
    tokens_completion: int = 50,
    tokens_total: int = 150,
) -> dict[str, Any]:
    """Build a minimal eval record for testing breakdowns."""
    meta = _empty_attempt_metadata(attempt=1)
    meta["tokens_prompt"] = tokens_prompt
    meta["tokens_completion"] = tokens_completion
    meta["tokens_total"] = tokens_total
    if timing is not None:
        meta["timing_breakdown"].update(timing)
    if tokens is not None:
        meta["token_breakdown"].update(tokens)
    return {
        "attempt_metadata": meta,
        "patch_meta": patch_meta or {},
        "attempt_wall_clock_sec": attempt_wall_clock_sec,
        "before_duration_sec": before_duration_sec,
        "after_duration_sec": after_duration_sec,
    }


# ── _empty_attempt_metadata defaults ────────────────────────────────────


def test_empty_attempt_metadata_has_timing_breakdown() -> None:
    meta = _empty_attempt_metadata(attempt=1)
    timing = meta["timing_breakdown"]
    assert isinstance(timing, dict)
    assert timing["stage_a_llm_sec"] == 0.0
    assert timing["stage_b_llm_sec"] == 0.0
    assert timing["adaptive_controller_prepare_sec"] == 0.0
    assert timing["adaptive_controller_sidecar_write_sec"] == 0.0
    assert timing["baseline_case_conversion_sec"] == 0.0
    assert timing["baseline_sidecar_load_sec"] == 0.0
    assert timing["baseline_internal_total_sec"] == 0.0
    assert timing["baseline_eval_sec"] == 0.0
    assert timing["baseline_orchestration_sec"] == 0.0
    assert timing["llm_total_sec"] == 0.0
    assert timing["overhead_sec"] == 0.0
    assert timing["llmdebug_total_sec"] == 0.0
    assert timing["tep_learning_llm_sec"] == 0.0
    assert len(timing) == 27


def test_empty_attempt_metadata_has_token_breakdown() -> None:
    meta = _empty_attempt_metadata(attempt=1)
    tokens = meta["token_breakdown"]
    assert isinstance(tokens, dict)
    assert tokens["stage_a_tokens_total"] == 0
    assert tokens["stage_b_tokens_total"] == 0
    assert tokens["snapshot_chars"] == 0
    assert len(tokens) == 14


# ── _finalize_breakdowns ────────────────────────────────────────────────


def test_finalize_breakdowns_extracts_stage_b_llm_sec() -> None:
    record = _make_record(patch_meta={"propose_duration_sec": 3.5})
    _finalize_breakdowns(record)
    timing = record["attempt_metadata"]["timing_breakdown"]
    assert timing["stage_b_llm_sec"] == 3.5


def test_finalize_breakdowns_uses_baseline_llm_time_when_present() -> None:
    record = _make_record(
        patch_meta={
            "propose_duration_sec": 8.0,
            "baseline_metrics": {
                "llm_duration_sec": 2.5,
                "eval_duration_sec": 1.5,
                "internal_total_duration_sec": 5.0,
                "orchestration_duration_sec": 1.0,
                "usage_available": False,
            },
        },
        attempt_wall_clock_sec=10.0,
        before_duration_sec=2.0,
        after_duration_sec=1.5,
    )
    _finalize_breakdowns(record)
    timing = record["attempt_metadata"]["timing_breakdown"]
    assert timing["stage_b_llm_sec"] == 2.5
    assert timing["baseline_eval_sec"] == 1.5
    assert timing["baseline_internal_total_sec"] == 5.0
    assert timing["baseline_orchestration_sec"] == 1.0
    assert timing["llm_total_sec"] == 2.5
    assert timing["overhead_sec"] == 1.5


def test_finalize_breakdowns_extracts_stage_a_from_meta() -> None:
    record = _make_record(
        patch_meta={
            "stage_a_meta": {
                "request_context_duration_sec": 1.2,
                "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
            },
        },
    )
    _finalize_breakdowns(record)
    timing = record["attempt_metadata"]["timing_breakdown"]
    tokens = record["attempt_metadata"]["token_breakdown"]
    assert timing["stage_a_llm_sec"] == 1.2
    assert tokens["stage_a_tokens_prompt"] == 20
    assert tokens["stage_a_tokens_completion"] == 10
    assert tokens["stage_a_tokens_total"] == 30


def test_finalize_breakdowns_extracts_adaptive_bridge_timings() -> None:
    record = _make_record(
        patch_meta={
            "adaptive_controller_prepare_sec": 0.25,
            "adaptive_controller_summarizer_sec": 0.1,
            "adaptive_controller_sidecar_build_sec": 0.2,
            "adaptive_controller_sidecar_write_sec": 0.3,
            "baseline_case_conversion_sec": 0.4,
            "baseline_sidecar_load_sec": 0.5,
        }
    )
    _finalize_breakdowns(record)
    timing = record["attempt_metadata"]["timing_breakdown"]
    assert timing["adaptive_controller_prepare_sec"] == 0.25
    assert timing["adaptive_controller_summarizer_sec"] == 0.1
    assert timing["adaptive_controller_sidecar_build_sec"] == 0.2
    assert timing["adaptive_controller_sidecar_write_sec"] == 0.3
    assert timing["baseline_case_conversion_sec"] == 0.4
    assert timing["baseline_sidecar_load_sec"] == 0.5


def test_finalize_breakdowns_computes_llm_total() -> None:
    record = _make_record(
        patch_meta={
            "propose_duration_sec": 2.0,
            "context_compaction_meta": {"compact_context_duration_sec": 0.5},
            "stage_a_meta": {"request_context_duration_sec": 1.0},
        },
    )
    _finalize_breakdowns(record)
    timing = record["attempt_metadata"]["timing_breakdown"]
    expected = 1.0 + 2.0 + 0.5  # stage_a + stage_b + compaction
    assert timing["llm_total_sec"] == expected


def test_finalize_breakdowns_reads_nested_compaction_duration() -> None:
    record = _make_record(
        patch_meta={"context_compaction_meta": {"compact_context_duration_sec": 0.75}}
    )
    _finalize_breakdowns(record)
    timing = record["attempt_metadata"]["timing_breakdown"]
    assert timing["context_compaction_llm_sec"] == 0.75


def test_finalize_breakdowns_computes_overhead() -> None:
    record = _make_record(
        patch_meta={"propose_duration_sec": 2.0},
        attempt_wall_clock_sec=10.0,
        before_duration_sec=2.0,
        after_duration_sec=1.5,
    )
    _finalize_breakdowns(record)
    timing = record["attempt_metadata"]["timing_breakdown"]
    # overhead = 10.0 - 2.0 - 1.5 - 2.0(llm_total) = 4.5
    assert timing["overhead_sec"] == 4.5


def test_finalize_breakdowns_computes_llmdebug_total() -> None:
    record = _make_record(timing={"snapshot_read_sec": 0.3, "stage_a_context_fulfillment_sec": 0.7})
    _finalize_breakdowns(record)
    timing = record["attempt_metadata"]["timing_breakdown"]
    assert timing["llmdebug_total_sec"] == 1.0


def test_finalize_breakdowns_stage_b_tokens_by_subtraction() -> None:
    record = _make_record(
        patch_meta={
            "stage_a_meta": {
                "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
            },
        },
        tokens_prompt=100,
        tokens_completion=50,
        tokens_total=150,
    )
    _finalize_breakdowns(record)
    tokens = record["attempt_metadata"]["token_breakdown"]
    assert tokens["stage_b_tokens_prompt"] == 80  # 100 - 20
    assert tokens["stage_b_tokens_completion"] == 40  # 50 - 10
    assert tokens["stage_b_tokens_total"] == 120  # 150 - 30


def test_finalize_breakdowns_tool_response_chars() -> None:
    record = _make_record()
    record["attempt_metadata"]["stage_a_chars_returned"] = 5000
    _finalize_breakdowns(record)
    tokens = record["attempt_metadata"]["token_breakdown"]
    assert tokens["tool_response_chars"] == 5000


def test_finalize_breakdowns_noop_without_attempt_metadata() -> None:
    record: dict[str, Any] = {"patch_meta": {}}
    _finalize_breakdowns(record)
    assert "attempt_metadata" not in record or record.get("attempt_metadata") is None


def test_finalize_breakdowns_noop_without_timing_dict() -> None:
    record: dict[str, Any] = {
        "attempt_metadata": {"some_field": 1},
        "patch_meta": {},
    }
    _finalize_breakdowns(record)
    assert "timing_breakdown" not in record["attempt_metadata"]


def test_finalize_breakdowns_recovery_durations() -> None:
    record = _make_record(
        patch_meta={
            "context_recovery_duration_sec": 0.4,
            "no_diff_recovery_duration_sec": 0.3,
            "patch_verifier_recovery_duration_sec": 0.2,
        },
    )
    _finalize_breakdowns(record)
    timing = record["attempt_metadata"]["timing_breakdown"]
    assert abs(timing["recovery_llm_sec"] - 0.9) < 1e-9
    assert abs(timing["recovery_total_sec"] - 0.9) < 1e-9


def test_finalize_breakdowns_llm_total_does_not_double_count_recovery() -> None:
    record = _make_record(
        patch_meta={
            "stage_a_meta": {"request_context_duration_sec": 1.0},
            "propose_duration_sec": 2.4,
            "context_recovery_duration_sec": 0.4,
            "no_diff_recovery_duration_sec": 0.3,
            "patch_verifier_recovery_duration_sec": 0.2,
        }
    )
    _finalize_breakdowns(record)
    timing = record["attempt_metadata"]["timing_breakdown"]
    assert abs(timing["recovery_llm_sec"] - 0.9) < 1e-9
    assert timing["llm_total_sec"] == 3.4


def test_finalize_breakdowns_test_durations_from_record() -> None:
    record = _make_record(before_duration_sec=5.0, after_duration_sec=3.0)
    _finalize_breakdowns(record)
    timing = record["attempt_metadata"]["timing_breakdown"]
    assert timing["test_before_sec"] == 5.0
    assert timing["test_after_sec"] == 3.0


def test_finalize_breakdowns_splits_recovery_tokens_before_stage_b() -> None:
    record = _make_record(
        patch_meta={
            "stage_a_meta": {
                "usage": {"prompt_tokens": 20, "completion_tokens": 10, "total_tokens": 30},
            },
            "context_compaction_invoked": True,
            "context_compaction_meta": {
                "usage": {"prompt_tokens": 5, "completion_tokens": 2, "total_tokens": 7},
            },
            "context_recovery_meta": {
                "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
            },
            "no_diff_recovery_meta": {
                "usage": {"prompt_tokens": 4, "completion_tokens": 1, "total_tokens": 5},
            },
            "patch_verifier_recovery_raw_meta": {
                "usage": {"prompt_tokens": 2, "completion_tokens": 1, "total_tokens": 3},
            },
        },
        tokens_prompt=100,
        tokens_completion=50,
        tokens_total=150,
    )
    _finalize_breakdowns(record)
    tokens = record["attempt_metadata"]["token_breakdown"]
    assert tokens["compaction_tokens_total"] == 7
    assert tokens["recovery_tokens_prompt"] == 13
    assert tokens["recovery_tokens_completion"] == 5
    assert tokens["recovery_tokens_total"] == 18
    assert tokens["stage_b_tokens_prompt"] == 62
    assert tokens["stage_b_tokens_completion"] == 33
    assert tokens["stage_b_tokens_total"] == 95


def test_apply_usage_totals_does_not_invent_usage_for_baseline_without_provider_counts() -> None:
    patch_meta = {"baseline_metrics": {"usage_available": False}}
    _apply_usage_totals_to_patch_meta(
        patch_meta=patch_meta,
        llm_call_count=2,
        tokens_prompt=0,
        tokens_completion=0,
        tokens_total=0,
    )
    assert patch_meta["llm_call_count"] == 2
    assert "usage" not in patch_meta


def test_apply_usage_totals_does_not_overwrite_partial_baseline_usage_with_exact_usage() -> None:
    patch_meta = {
        "baseline_metrics": {
            "usage_available": False,
            "usage_reported_call_count": 2,
            "usage_missing_call_count": 1,
            "reported_tokens_prompt": 20,
            "reported_tokens_completion": 6,
            "reported_tokens_total": 26,
        }
    }
    _apply_usage_totals_to_patch_meta(
        patch_meta=patch_meta,
        llm_call_count=3,
        tokens_prompt=0,
        tokens_completion=0,
        tokens_total=0,
    )
    assert patch_meta["llm_call_count"] == 3
    assert patch_meta["tokens_total"] == 0
    assert "usage" not in patch_meta
    assert patch_meta["baseline_metrics"]["reported_tokens_total"] == 26


def test_apply_usage_totals_removes_stale_partial_baseline_usage_payload() -> None:
    patch_meta = {
        "usage": {
            "prompt_tokens": 20,
            "completion_tokens": 6,
            "total_tokens": 26,
        },
        "baseline_metrics": {
            "usage_available": False,
            "usage_reported_call_count": 2,
            "usage_missing_call_count": 1,
            "reported_tokens_prompt": 20,
            "reported_tokens_completion": 6,
            "reported_tokens_total": 26,
        },
    }
    _apply_usage_totals_to_patch_meta(
        patch_meta=patch_meta,
        llm_call_count=3,
        tokens_prompt=20,
        tokens_completion=6,
        tokens_total=26,
    )
    assert patch_meta["llm_call_count"] == 3
    assert patch_meta["tokens_prompt"] == 0
    assert patch_meta["tokens_completion"] == 0
    assert patch_meta["tokens_total"] == 0
    assert "usage" not in patch_meta
    assert patch_meta["baseline_metrics"]["reported_tokens_total"] == 26


def test_apply_usage_totals_populates_exact_usage_when_available() -> None:
    patch_meta = {"baseline_metrics": {"usage_available": True}}
    _apply_usage_totals_to_patch_meta(
        patch_meta=patch_meta,
        llm_call_count=2,
        tokens_prompt=20,
        tokens_completion=6,
        tokens_total=26,
    )
    assert patch_meta["llm_call_count"] == 2
    assert patch_meta["usage"] == {
        "prompt_tokens": 20,
        "completion_tokens": 6,
        "total_tokens": 26,
    }


def test_token_usage_breakdown_treats_explicitly_unavailable_usage_as_authoritative() -> None:
    prompt_total, completion_total, total_total = _token_usage_breakdown(
        [
            {
                "attempt_metadata": {
                    "token_usage_available": False,
                    "tokens_prompt": 0,
                    "tokens_completion": 0,
                    "tokens_total": 0,
                },
                "patch_meta": {
                    "usage": {
                        "prompt_tokens": 20,
                        "completion_tokens": 6,
                        "total_tokens": 26,
                    }
                },
            }
        ]
    )

    # The explicit flag still wins over legacy patch_meta usage, but unmeasured
    # attempts are None rather than 0 tokens (STA-01).
    assert (prompt_total, completion_total, total_total) == (None, None, None)


# ── Aggregation functions (analyze_results) ─────────────────────────────


def _records_with_breakdowns(n: int = 3) -> list[dict[str, Any]]:
    """Create n records with populated breakdowns."""
    records = []
    for i in range(n):
        record = _make_record(
            patch_meta={
                "propose_duration_sec": 2.0 + i,
                "stage_a_meta": {
                    "request_context_duration_sec": 1.0,
                    "usage": {
                        "prompt_tokens": 20,
                        "completion_tokens": 10,
                        "total_tokens": 30,
                    },
                },
            },
            tokens_prompt=100 + i * 10,
            tokens_completion=50 + i * 5,
            tokens_total=150 + i * 15,
        )
        record["attempt_metadata"]["timing_breakdown"]["snapshot_read_sec"] = 0.1 * (i + 1)
        _finalize_breakdowns(record)
        records.append(record)
    return records


def test_timing_breakdown_mean_computes_correctly() -> None:
    records = _records_with_breakdowns(3)
    result = _timing_breakdown_mean(records)
    # stage_b_llm_sec values: 2.0, 3.0, 4.0 → mean 3.0
    assert result["stage_b_llm_sec"] == 3.0
    # snapshot_read_sec values: 0.1, 0.2, 0.3 → mean 0.2
    assert abs(result["snapshot_read_sec"] - 0.2) < 1e-9


def test_timing_breakdown_mean_empty_records() -> None:
    result = _timing_breakdown_mean([])
    assert all(v is None for v in result.values())


def test_timing_breakdown_mean_no_timing_data() -> None:
    records = [{"attempt_metadata": {"some_field": 1}}]
    result = _timing_breakdown_mean(records)
    assert all(v is None for v in result.values())


def test_token_breakdown_mean_computes_correctly() -> None:
    records = _records_with_breakdowns(3)
    result = _token_breakdown_mean(records)
    # stage_a_tokens_total is always 30 → mean 30.0
    assert result["stage_a_tokens_total"] == 30.0


def test_token_breakdown_mean_empty_records() -> None:
    result = _token_breakdown_mean([])
    assert all(v is None for v in result.values())


def test_timing_breakdown_sum_computes_correctly() -> None:
    records = _records_with_breakdowns(3)
    result = _timing_breakdown_sum(records)
    # stage_b_llm_sec values: 2.0, 3.0, 4.0 → sum 9.0
    assert result["stage_b_llm_sec"] == 9.0


def test_token_breakdown_sum_computes_correctly() -> None:
    records = _records_with_breakdowns(3)
    result = _token_breakdown_sum(records)
    # stage_a_tokens_total is always 30 → sum 90
    assert result["stage_a_tokens_total"] == 90


# ── Derived field consistency ────────────────────────────────────────────


def test_llm_total_equals_sum_of_components() -> None:
    record = _make_record(
        patch_meta={
            "propose_duration_sec": 2.5,
            "context_compaction_meta": {"compact_context_duration_sec": 0.8},
            "context_recovery_duration_sec": 0.3,
            "stage_a_meta": {"request_context_duration_sec": 1.1},
        },
    )
    _finalize_breakdowns(record)
    timing = record["attempt_metadata"]["timing_breakdown"]
    expected = (
        timing["stage_a_llm_sec"] + timing["stage_b_llm_sec"] + timing["context_compaction_llm_sec"]
    )
    assert abs(timing["llm_total_sec"] - expected) < 1e-9


def test_overhead_is_non_negative() -> None:
    # Even with huge LLM time, overhead should be clamped to 0
    record = _make_record(
        patch_meta={"propose_duration_sec": 100.0},
        attempt_wall_clock_sec=5.0,
        before_duration_sec=0.0,
        after_duration_sec=0.0,
    )
    _finalize_breakdowns(record)
    timing = record["attempt_metadata"]["timing_breakdown"]
    assert timing["overhead_sec"] >= 0.0


def test_no_patch_needed_record_does_not_crash() -> None:
    """Records with no timing_breakdown (e.g., no_patch_needed exits) are safe."""
    record: dict[str, Any] = {
        "attempt_metadata": {"some_field": True},
        "patch_meta": {},
    }
    _finalize_breakdowns(record)

    # Record should still have its original keys (function returned early)
    assert "attempt_metadata" in record
    assert record["attempt_metadata"]["some_field"] is True


# ── Report overhead analysis ─────────────────────────────────────────────


def _make_case_row(
    *,
    condition: str,
    runtime_sec: float = 10.0,
    timing_breakdown_sum: dict[str, float] | None = None,
    token_breakdown_sum: dict[str, int] | None = None,
) -> dict[str, Any]:
    return {
        "condition": condition,
        "runtime_sec": runtime_sec,
        "timing_breakdown_sum": timing_breakdown_sum or {},
        "token_breakdown_sum": token_breakdown_sum or {},
    }


def test_compute_overhead_analysis_per_condition() -> None:
    rows = [
        _make_case_row(
            condition="base_repair",
            runtime_sec=10.0,
            timing_breakdown_sum={"llm_total_sec": 5.0, "overhead_sec": 2.0},
            token_breakdown_sum={"stage_b_tokens_total": 500},
        ),
        _make_case_row(
            condition="cot_repair",
            runtime_sec=12.0,
            timing_breakdown_sum={
                "llm_total_sec": 6.0,
                "overhead_sec": 2.5,
                "llmdebug_total_sec": 1.5,
            },
            token_breakdown_sum={"stage_b_tokens_total": 600, "snapshot_chars": 3000},
        ),
    ]
    result = _compute_overhead_analysis(
        rows=rows, condition_a="base_repair", condition_b="cot_repair"
    )
    per_cond = result["per_condition"]
    assert "base_repair" in per_cond
    assert "cot_repair" in per_cond
    assert per_cond["base_repair"]["cases"] == 1
    assert per_cond["cot_repair"]["llm_pct_of_runtime"] == 50.0  # 6/12*100


def test_compute_overhead_analysis_empty_rows() -> None:
    result = _compute_overhead_analysis(
        rows=[], condition_a="base_repair", condition_b="cot_repair"
    )
    assert result["per_condition"] == {}


def test_compute_overhead_analysis_missing_condition() -> None:
    rows = [_make_case_row(condition="base_repair", runtime_sec=10.0)]
    result = _compute_overhead_analysis(
        rows=rows, condition_a="base_repair", condition_b="cot_repair"
    )
    assert "cot_repair" not in result["per_condition"]
    assert "base_repair" in result["per_condition"]


def test_att13_compaction_without_its_own_meta_reports_no_compaction_tokens() -> None:
    record = _make_record(
        patch_meta={
            "context_compaction_invoked": True,
            "usage": {"prompt_tokens": 90, "completion_tokens": 40, "total_tokens": 130},
        },
        tokens_prompt=90,
        tokens_completion=40,
        tokens_total=130,
    )
    _finalize_breakdowns(record)
    tokens = record["attempt_metadata"]["token_breakdown"]
    # The propose call's usage is not compaction usage.
    assert tokens["compaction_tokens_total"] == 0
