"""Tests for run-metrics reporting (effectiveness, efficiency, failure diagnostics)."""

from __future__ import annotations

import pytest

from evals.report_run_metrics import (
    _compute_dual_objective_summary,
    _compute_effectiveness_summary,
    _compute_efficiency_summaries,
    _compute_failure_diagnostics,
    _compute_token_estimate_diagnostics,
    _filter_rows_by_snapshot_dependency,
    _paired_sign_test,
    _parse_eval_log,
    _parse_fixed_budget_thresholds,
    _parse_job_feedback_block,
    main,
)
from llmdebug import _json as json


def _record(
    *,
    run_id: str,
    case_id: str,
    condition: str,
    snapshot_dependency: str,
    category: str = "logic_error",
    difficulty: str = "easy",
    attempt: int = 1,
    success: bool = False,
    tokens_prompt: int = 10,
    tokens_completion: int = 2,
    tokens_total: int = 12,
    llm_call_count: int = 1,
    cumulative_runtime_sec: float = 1.0,
    patch_meta: dict[str, object] | None = None,
) -> dict[str, object]:
    record = {
        "schema_version": 5,
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
            "started_at_utc": "2026-02-18T00:00:00Z",
            "deterministic": False,
            "seed": None,
            "python_version": "3.11.0",
            "platform": "test",
            "llmdebug_version": "2.11.4",
            "patcher_name": "openai",
            "patcher_config_fingerprint": "fp",
            "run_settings_fingerprint": "runfp",
        },
        "attempt_metadata": {
            "attempt": attempt,
            "max_attempts": 3,
            "attempt_ordinal_label": f"{attempt}/3",
            "llm_call_count": llm_call_count,
            "tokens_prompt": tokens_prompt,
            "tokens_completion": tokens_completion,
            "tokens_total": tokens_total,
            "retry_history_budget_chars": 12000,
            "retry_history_budget_applied": False,
            "prompt_sha256": "p",
            "prompt_path": None,
            "evidence_sha256": "e",
            "evidence_path": None,
        },
        "model_metadata": {
            "backend": "openai",
            "name": "openai",
            "base_url": "http://127.0.0.1:8000/v1",
            "model": "dummy",
            "temperature": 0.0,
            "max_tokens": 2048,
            "response_mode": "edits",
            "api_compat_seed_requested": None,
            "api_compat_seed_acknowledged": False,
            "api_compat_seed_response": None,
        },
        "cumulative_runtime_sec": cumulative_runtime_sec,
        "note": "tests_still_failing" if not success else "success",
    }
    if patch_meta is not None:
        record["patch_meta"] = patch_meta
    return record


def test_parse_job_feedback_block_extracts_energy_and_power() -> None:
    text = """
============================= JOB FEEDBACK =============================
Job Wall-clock time: 03:43:16
Energy Consumed: 8982304 Joule / 2495.08444444444 Watthours
Average node power draw: 670.521349656614 Watt
"""
    parsed = _parse_job_feedback_block(text)
    assert parsed["job wall-clock time"] == "03:43:16"
    assert parsed["energy_joule"] == 8982304.0
    assert parsed["energy_watthours"] == 2495.08444444444
    assert parsed["average_node_power_watt"] == 670.521349656614


def test_parse_eval_log_keeps_elapsed_without_prompt_suffix() -> None:
    parsed = _parse_eval_log(
        "Progress: finished executed=4, skipped=0, total=4, elapsed=10m00s "
        "prompt=1234ch tokens=prompt:100,total:140\n"
        "Outcome summary: success=1 failed=2 runtime_errors=1 skipped=0\n"
        "Top failure reasons: request budget exceeded=2, runtime error=1\n"
    )
    assert parsed["runner_elapsed"] == "10m00s"


def test_parse_fixed_budget_thresholds_deduplicates_and_sorts() -> None:
    assert _parse_fixed_budget_thresholds("25000,10000,25000") == [10000, 25000]


def test_compute_dual_objective_summary_handles_zero_success_and_fixed_budget() -> None:
    rows = [
        {
            "run_id": "r1",
            "case_id": "c1",
            "condition": "base_repair",
            "success": False,
            "token_usage_total": 12000.0,
        },
        {
            "run_id": "r1",
            "case_id": "c1",
            "condition": "cot_repair",
            "success": True,
            "token_usage_total": 9000.0,
        },
        {
            "run_id": "r1",
            "case_id": "c2",
            "condition": "base_repair",
            "success": False,
            "token_usage_total": 11000.0,
        },
        {
            "run_id": "r1",
            "case_id": "c2",
            "condition": "cot_repair",
            "success": False,
            "token_usage_total": 13000.0,
        },
    ]
    summary = _compute_dual_objective_summary(
        rows=rows,
        condition_a="base_repair",
        condition_b="cot_repair",
        n_bootstrap=200,
        seed=1,
        fixed_budget_thresholds=[10000, 15000],
    )
    assert summary["available"] is True
    assert summary["cost_per_success"]["a_value"] is None
    assert summary["cost_per_success"]["b_value"] == pytest.approx(22000.0 / 1.0)
    assert summary["paired_cost_delta"]["mean_delta"] == pytest.approx(-500.0)
    assert summary["fixed_budget_success"][0]["a_rate"] == pytest.approx(0.0)
    assert summary["fixed_budget_success"][0]["b_rate"] == pytest.approx(0.5)
    assert summary["fixed_budget_success"][1]["b_rate"] == pytest.approx(0.5)


def test_compute_dual_objective_summary_reports_missing_token_coverage() -> None:
    rows = [
        {
            "run_id": "r1",
            "case_id": "c1",
            "condition": "base_repair",
            "success": False,
            "token_usage_total": None,
        },
        {
            "run_id": "r1",
            "case_id": "c1",
            "condition": "cot_repair",
            "success": True,
            "token_usage_total": 5000.0,
        },
    ]
    summary = _compute_dual_objective_summary(
        rows=rows,
        condition_a="base_repair",
        condition_b="cot_repair",
        n_bootstrap=50,
        seed=1,
        fixed_budget_thresholds=[10000],
    )
    assert summary["available"] is True
    assert summary["token_covered_pairs"] == 0
    assert summary["cost_per_success"]["available"] is False
    assert summary["paired_cost_delta"]["available"] is False
    assert summary["fixed_budget_success"] == []


def _cost_row(run_id: str, case_id: str, condition: str, success: bool, tokens: float | None):
    return {
        "run_id": run_id,
        "case_id": case_id,
        "condition": condition,
        "success": success,
        "token_usage_total": tokens,
    }


def test_fixed_budget_success_applies_the_threshold_to_each_run(tmp_path) -> None:
    """codex 2-2#1: the budget was applied to the mean cost of a case's runs."""
    rows = [
        # One expensive success and one cheap failure: no run met a 6,000-token budget.
        _cost_row("r1", "c1", "base_repair", True, 10_000.0),
        _cost_row("r2", "c1", "base_repair", False, 1_000.0),
        _cost_row("r1", "c1", "cot_repair", False, 5_000.0),
        _cost_row("r2", "c1", "cot_repair", False, 5_000.0),
    ]
    summary = _compute_dual_objective_summary(
        rows=rows,
        condition_a="base_repair",
        condition_b="cot_repair",
        n_bootstrap=50,
        seed=1,
        fixed_budget_thresholds=[6_000, 10_000],
    )

    assert summary["matched_pairs"] == 1
    assert summary["matched_runs"] == 2
    assert summary["token_covered_runs"] == 2
    tight, loose = summary["fixed_budget_success"]
    assert tight["token_threshold"] == 6_000
    assert tight["a_rate"] == pytest.approx(0.0)  # was 0.5 on the 5,500-token mean
    assert loose["a_rate"] == pytest.approx(0.5)  # the 10,000-token run qualifies


def test_token_cost_statistics_need_both_sides_in_the_same_run() -> None:
    """codex 2-2#3: independent null-skipping means manufactured a covered pair."""
    rows = [
        _cost_row("r1", "c1", "base_repair", True, 4_000.0),
        _cost_row("r1", "c1", "cot_repair", True, None),
        _cost_row("r2", "c1", "base_repair", False, None),
        _cost_row("r2", "c1", "cot_repair", False, 8_000.0),
    ]
    summary = _compute_dual_objective_summary(
        rows=rows,
        condition_a="base_repair",
        condition_b="cot_repair",
        n_bootstrap=50,
        seed=1,
        fixed_budget_thresholds=[10_000],
    )

    assert summary["matched_pairs"] == 1
    assert summary["matched_runs"] == 2
    assert summary["token_covered_pairs"] == 0
    assert summary["token_covered_runs"] == 0
    assert summary["token_coverage_rate"] == pytest.approx(0.0)
    assert summary["cost_per_success"]["available"] is False
    assert summary["paired_cost_delta"]["available"] is False
    assert summary["fixed_budget_success"] == []


def test_cost_per_success_counts_only_token_covered_runs() -> None:
    """codex 2-2#3: success fractions included runs the token means excluded."""
    rows = [
        _cost_row("r1", "c1", "base_repair", False, 2_000.0),
        _cost_row("r1", "c1", "cot_repair", False, 2_000.0),
        # The successful run reported no tokens for A, so it is not a cost observation.
        _cost_row("r2", "c1", "base_repair", True, None),
        _cost_row("r2", "c1", "cot_repair", True, 2_000.0),
    ]
    summary = _compute_dual_objective_summary(
        rows=rows,
        condition_a="base_repair",
        condition_b="cot_repair",
        n_bootstrap=50,
        seed=1,
        fixed_budget_thresholds=[10_000],
    )

    cost = summary["cost_per_success"]
    assert summary["token_covered_runs"] == 1
    assert cost["a_successes"] == pytest.approx(0.0)
    assert cost["a_value"] is None  # no success among the token-covered runs
    assert summary["fixed_budget_success"][0]["a_rate"] == pytest.approx(0.0)


def test_compute_token_estimate_diagnostics_handles_partial_coverage() -> None:
    left = _record(
        run_id="run-est",
        case_id="case-1",
        condition="base_repair",
        snapshot_dependency="required",
        tokens_prompt=10,
    )
    left["attempt_metadata"]["prompt_token_estimate_available"] = True
    left["attempt_metadata"]["prompt_tokens_estimate_pre_call"] = 11
    right = _record(
        run_id="run-est",
        case_id="case-2",
        condition="base_repair",
        snapshot_dependency="required",
        tokens_prompt=20,
    )
    right["attempt_metadata"]["prompt_token_estimate_available"] = True
    right["attempt_metadata"]["prompt_tokens_estimate_pre_call"] = 19
    missing_provider = _record(
        run_id="run-est",
        case_id="case-3",
        condition="base_repair",
        snapshot_dependency="required",
        tokens_prompt=0,
    )
    missing_provider["attempt_metadata"]["token_usage_available"] = False
    missing_provider["attempt_metadata"]["prompt_token_estimate_available"] = True
    missing_provider["attempt_metadata"]["prompt_tokens_estimate_pre_call"] = 17
    diagnostics = _compute_token_estimate_diagnostics(
        raw_records=[left, right, missing_provider],
        condition_a="base_repair",
        condition_b="cot_repair",
    )
    row = diagnostics["per_condition"]["base_repair"]
    assert row["provider_covered_attempts"] == 2
    assert row["estimate_covered_attempts"] == 3
    assert row["paired_attempts"] == 2
    assert row["available"] is True
    assert row["correlation"] == pytest.approx(1.0)


def test_failure_diagnostics_detects_context_overflow_rates() -> None:
    records = [
        {
            "run_id": "r1",
            "case_id": "c1",
            "condition": "base_repair",
            "success": False,
            "note": "tests_still_failing",
            "patch_meta": {
                "error": "http_error",
                "status": 400,
                "body": "request exceeds the available context size",
            },
        },
        {
            "run_id": "r1",
            "case_id": "c1",
            "condition": "base_repair",
            "success": False,
            "note": "tests_still_failing",
            "patch_meta": {"error": "no_json_payload"},
        },
        {
            "run_id": "r1",
            "case_id": "c2",
            "condition": "cot_repair",
            "success": False,
            "note": "tests_still_failing",
            "patch_meta": {
                "error": "http_error",
                "status": 400,
                "body": "request exceeds the available context size",
            },
        },
    ]
    out = _compute_failure_diagnostics(
        raw_records=records,
        condition_a="base_repair",
        condition_b="cot_repair",
        top_n=5,
    )
    tb = out["per_condition"]["base_repair"]
    ws = out["per_condition"]["cot_repair"]
    assert tb["attempts"] == 2
    assert tb["context_overflow_attempts"] == 1
    assert tb["context_overflow_cases"] == 1
    assert tb["http_error_attempts"] == 1
    assert tb["patch_attempts"] == 0
    assert tb["test_touch_attempts"] == 0
    assert ws["attempts"] == 1
    assert ws["context_overflow_attempt_rate"] == 1.0
    assert ws["patch_attempts"] == 0
    assert ws["test_touch_attempts"] == 0


def test_failure_diagnostics_tracks_test_touch_patch_attempt_rates() -> None:
    records = [
        {
            "run_id": "r1",
            "case_id": "c1",
            "condition": "base_repair",
            "success": False,
            "note": "tests_still_failing",
            "patch_meta": {"patch_file_count": 2, "patch_touches_tests": True},
        },
        {
            "run_id": "r1",
            "case_id": "c1",
            "condition": "base_repair",
            "success": False,
            "note": "tests_still_failing",
            "patch_meta": {"patch_file_count": 1, "patch_touches_tests": False},
        },
        {
            "run_id": "r1",
            "case_id": "c2",
            "condition": "cot_repair",
            "success": False,
            "note": "tests_still_failing",
            "patch_meta": {"patch_file_count": 1, "patch_touches_tests": True},
        },
    ]
    out = _compute_failure_diagnostics(
        raw_records=records,
        condition_a="base_repair",
        condition_b="cot_repair",
        top_n=5,
    )
    tb = out["per_condition"]["base_repair"]
    ws = out["per_condition"]["cot_repair"]
    assert tb["patch_attempts"] == 2
    assert tb["test_touch_attempts"] == 1
    assert tb["test_touch_attempt_rate"] == 0.5
    assert ws["patch_attempts"] == 1
    assert ws["test_touch_attempts"] == 1
    assert ws["test_touch_attempt_rate"] == 1.0


def test_failure_diagnostics_prefers_terminal_baseline_causes_and_usage_coverage() -> None:
    trace_timeout = _record(
        run_id="run",
        case_id="trace-timeout",
        condition="tracecoder",
        snapshot_dependency="required",
        attempt=1,
        success=False,
        tokens_total=0,
        llm_call_count=2,
        patch_meta={
            "fidelity_level": "upstream_patched",
            "repair_scope_policy": "single_file_only",
            "prompt_contract_kind": "patched_tracecoder_dataset_prompt",
            "baseline_hybrid_guidance_mode": "fair_v1",
            "baseline_metrics": {
                "usage_available": False,
                "llm_call_count": 2,
            },
            "baseline_meta": {
                "error": "timeout",
            },
        },
    )
    trace_timeout["note"] = "no_patch_proposed"
    trace_success = _record(
        run_id="run",
        case_id="trace-success",
        condition="tracecoder",
        snapshot_dependency="required",
        attempt=1,
        success=True,
        tokens_total=40,
        llm_call_count=3,
        patch_meta={
            "baseline_metrics": {
                "usage_available": True,
                "llm_call_count": 3,
            },
        },
    )
    intervenor_noop = _record(
        run_id="run",
        case_id="intervenor-noop",
        condition="intervenor",
        snapshot_dependency="required",
        attempt=1,
        success=False,
        tokens_total=0,
        llm_call_count=0,
        patch_meta={
            "baseline_metrics": {
                "usage_available": False,
                "llm_call_count": 0,
            },
            "baseline_meta": {
                "error": "learner_empty_response",
            },
        },
    )
    intervenor_noop["note"] = "forced_noop_patch"
    records = [trace_timeout, trace_success, intervenor_noop]
    out = _compute_failure_diagnostics(
        raw_records=records,
        condition_a="tracecoder",
        condition_b="intervenor",
        top_n=5,
    )
    trace = out["per_condition"]["tracecoder"]
    intervenor = out["per_condition"]["intervenor"]
    assert trace["top_reasons"] == [("baseline_error:timeout [runner_note=no_patch_proposed]", 1)]
    assert trace["usage_reported_cases"] == 1
    assert trace["usage_missing_cases"] == 1
    assert trace["timeout_missing_usage_cases"] == 1
    assert intervenor["top_reasons"] == [
        ("baseline_error:learner_empty_response [runner_note=forced_noop_patch]", 1)
    ]
    assert intervenor["usage_applicable_cases"] == 0
    assert intervenor["usage_not_applicable_cases"] == 1
    assert intervenor["usage_reported_cases"] == 0
    assert intervenor["usage_missing_cases"] == 0


def test_failure_diagnostics_treats_zero_call_timeouts_as_not_applicable() -> None:
    trace_timeout = _record(
        run_id="run",
        case_id="trace-timeout-no-call",
        condition="tracecoder",
        snapshot_dependency="required",
        attempt=1,
        success=False,
        tokens_total=0,
        llm_call_count=0,
        patch_meta={
            "baseline_metrics": {
                "usage_available": False,
                "llm_call_count": 0,
            },
            "baseline_meta": {
                "error": "timeout",
            },
        },
    )
    trace_timeout["note"] = "no_patch_proposed"
    trace_success = _record(
        run_id="run",
        case_id="trace-success",
        condition="tracecoder",
        snapshot_dependency="required",
        attempt=1,
        success=True,
        tokens_total=40,
        llm_call_count=3,
        patch_meta={
            "baseline_metrics": {
                "usage_available": True,
                "llm_call_count": 3,
            },
        },
    )

    out = _compute_failure_diagnostics(
        raw_records=[trace_timeout, trace_success],
        condition_a="tracecoder",
        condition_b="ldb",
        top_n=5,
    )

    trace = out["per_condition"]["tracecoder"]
    assert trace["usage_applicable_cases"] == 1
    assert trace["usage_reported_cases"] == 1
    assert trace["usage_missing_cases"] == 0
    assert trace["timeout_missing_usage_cases"] == 0
    assert trace["usage_not_applicable_cases"] == 1


def test_effectiveness_summary_on_primary_slice() -> None:
    rows = [
        {
            "run_id": "r1",
            "case_id": "c_req",
            "condition": "base_repair",
            "category": "hidden_state",
            "snapshot_dependency": "required",
            "success": False,
        },
        {
            "run_id": "r1",
            "case_id": "c_req",
            "condition": "cot_repair",
            "category": "hidden_state",
            "snapshot_dependency": "required",
            "success": True,
        },
        {
            "run_id": "r1",
            "case_id": "c_help",
            "condition": "base_repair",
            "category": "logic_error",
            "snapshot_dependency": "helpful",
            "success": False,
        },
        {
            "run_id": "r1",
            "case_id": "c_help",
            "condition": "cot_repair",
            "category": "logic_error",
            "snapshot_dependency": "helpful",
            "success": True,
        },
        {
            "run_id": "r1",
            "case_id": "c_none",
            "condition": "base_repair",
            "category": "traceback_controls",
            "snapshot_dependency": "none",
            "success": True,
        },
        {
            "run_id": "r1",
            "case_id": "c_none",
            "condition": "cot_repair",
            "category": "traceback_controls",
            "snapshot_dependency": "none",
            "success": False,
        },
    ]
    primary_rows = _filter_rows_by_snapshot_dependency(rows, {"required", "helpful"})
    result = _compute_effectiveness_summary(
        rows=primary_rows,
        condition_a="base_repair",
        condition_b="cot_repair",
        n_bootstrap=200,
        seed=123,
    )
    assert result["available"] is True
    assert result["n_pairs"] == 2
    assert result["a_rate"] == 0.0
    assert result["b_rate"] == 1.0
    assert result["uplift_pp"] == 100.0
    assert result["agreement"]["available"] is True
    assert -1.0 <= result["agreement"]["kappa"] <= 1.0


def test_efficiency_summary_has_sign_test_and_bootstrap() -> None:
    rows = [
        {
            "run_id": "r1",
            "case_id": "c1",
            "condition": "base_repair",
            "token_usage_total": 100.0,
            "runtime_sec": 10.0,
            "llm_calls_total": 3.0,
        },
        {
            "run_id": "r1",
            "case_id": "c1",
            "condition": "cot_repair",
            "token_usage_total": 90.0,
            "runtime_sec": 9.0,
            "llm_calls_total": 2.0,
        },
        {
            "run_id": "r1",
            "case_id": "c2",
            "condition": "base_repair",
            "token_usage_total": 50.0,
            "runtime_sec": 5.0,
            "llm_calls_total": 1.0,
        },
        {
            "run_id": "r1",
            "case_id": "c2",
            "condition": "cot_repair",
            "token_usage_total": 70.0,
            "runtime_sec": 6.0,
            "llm_calls_total": 2.0,
        },
    ]
    result = _compute_efficiency_summaries(
        rows=rows,
        condition_a="base_repair",
        condition_b="cot_repair",
        n_bootstrap=200,
        seed=11,
    )
    tokens = result["token_usage_total"]
    assert tokens["available"] is True
    assert tokens["n_pairs"] == 2
    assert "sign_test" in tokens
    assert "wilcoxon" in tokens
    assert "bootstrap_ci_mean_delta" in tokens


def test_paired_sign_test_handles_large_nonzero_count() -> None:
    deltas = [1.0] * 1000 + [-1.0] * 600
    out = _paired_sign_test(deltas)
    assert out["n_nonzero"] == 1600
    assert out["n_positive"] == 1000
    assert out["n_negative"] == 600
    assert 0.0 <= out["p_value"] <= 1.0


def test_main_smoke_prints_expected_sections(tmp_path, capsys) -> None:
    jsonl = tmp_path / "mini.jsonl"
    records = [
        _record(
            run_id="run-smoke",
            case_id="case-required",
            condition="base_repair",
            snapshot_dependency="required",
            success=False,
            tokens_total=100,
            cumulative_runtime_sec=10.0,
            llm_call_count=3,
        ),
        _record(
            run_id="run-smoke",
            case_id="case-required",
            condition="cot_repair",
            snapshot_dependency="required",
            success=True,
            tokens_total=120,
            cumulative_runtime_sec=11.0,
            llm_call_count=2,
        ),
        _record(
            run_id="run-smoke",
            case_id="case-control",
            condition="base_repair",
            snapshot_dependency="none",
            category="traceback_controls",
            success=True,
            tokens_total=80,
            cumulative_runtime_sec=8.0,
            llm_call_count=1,
        ),
        _record(
            run_id="run-smoke",
            case_id="case-control",
            condition="cot_repair",
            snapshot_dependency="none",
            category="traceback_controls",
            success=False,
            tokens_total=90,
            cumulative_runtime_sec=9.0,
            llm_call_count=2,
        ),
    ]
    jsonl.write_text(
        "\n".join(json.dumps(record, ensure_ascii=True) for record in records) + "\n",
        encoding="utf-8",
    )

    eval_log = tmp_path / "eval.out"
    eval_log.write_text(
        "\n".join(
            [
                "Progress: finished executed=4, skipped=0, total=4, elapsed=10m00s "
                "prompt=1234ch tokens=prompt:100,total:140",
                "============================= JOB FEEDBACK =============================",
                "Job Wall-clock time: 00:10:02",
                "CPU Utilized: 00:12:00",
                "Memory Utilized: 2.00 GB",
                "Energy Consumed: 1000 Joule / 0.2777777777 Watthours",
                "Average node power draw: 100 Watt",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    rc = main(
        [
            "--jsonl",
            str(jsonl),
            "--eval-log",
            str(eval_log),
            "--bootstrap",
            "200",
            "--seed",
            "1",
            "--print-top-failures",
            "3",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "Primary Endpoint" in out
    assert "Secondary Efficiency" in out
    assert "Dual Objective" in out
    assert "Failure Diagnostics" in out
    assert "Run-Level System Context" in out
    assert "Energy consumed (run-level)" in out
    assert "One-Line Conclusion" in out


def test_main_prints_baseline_diagnostics_when_present(tmp_path, capsys) -> None:
    jsonl = tmp_path / "baseline.jsonl"
    records = [
        _record(
            run_id="run-baseline",
            case_id="pair-a",
            condition="base_repair",
            snapshot_dependency="required",
            success=False,
        ),
        _record(
            run_id="run-baseline",
            case_id="pair-a",
            condition="layered_adaptive_llm",
            snapshot_dependency="required",
            success=True,
        ),
        _record(
            run_id="run-baseline",
            case_id="tracecoder-case",
            condition="tracecoder",
            snapshot_dependency="required",
            success=True,
            patch_meta={
                "baseline_metrics": {
                    "direct_gen_passed": False,
                    "debug_added_value": True,
                    "method_instrumentation": True,
                    "method_two_step_repair": True,
                    "stopped_reason": "Debug cycle completed",
                }
            },
        ),
        _record(
            run_id="run-baseline",
            case_id="intervenor-case",
            condition="intervenor",
            snapshot_dependency="required",
            success=False,
            patch_meta={
                "baseline_metrics": {
                    "teacher_duration_sec_total": 1.1,
                    "learner_duration_sec_total": 0.7,
                    "postcheck_duration_sec_total": 0.3,
                    "learner_empty_response_count": 1,
                }
            },
        ),
        _record(
            run_id="run-baseline",
            case_id="ldb-case",
            condition="ldb",
            snapshot_dependency="helpful",
            success=True,
            patch_meta={
                "baseline_metrics": {
                    "is_passing": True,
                    "test_feedback_improvement_steps": 1,
                    "test_feedback_regression_steps": 0,
                    "test_feedback_final_fail_count": 0,
                }
            },
        ),
        _record(
            run_id="run-baseline",
            case_id="ldb-case-missing",
            condition="ldb",
            snapshot_dependency="helpful",
            success=False,
        ),
        _record(
            run_id="run-baseline",
            case_id="non-baseline-case",
            condition="base_repair",
            snapshot_dependency="required",
            success=True,
            patch_meta={"baseline_metrics": {"debug_added_value": True}},
        ),
    ]
    jsonl.write_text(
        "\n".join(json.dumps(record, ensure_ascii=True) for record in records) + "\n",
        encoding="utf-8",
    )

    rc = main(["--jsonl", str(jsonl), "--bootstrap", "200", "--seed", "1"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "Baseline Diagnostics" in out
    assert "- tracecoder:" in out
    assert "debug_added_value_rate=100.00% (coverage 1/1)" in out
    assert "teacher_sec=1.1000" in out
    assert "is_passing_rate=100.00% (coverage 1/2)" in out


def test_main_failure_diagnostics_prints_explicit_baseline_causes_and_usage_coverage(
    tmp_path, capsys
) -> None:
    jsonl = tmp_path / "baseline_failures.jsonl"
    trace_timeout = _record(
        run_id="run-baseline",
        case_id="trace-timeout",
        condition="tracecoder",
        snapshot_dependency="required",
        success=False,
        llm_call_count=2,
        patch_meta={
            "fidelity_level": "upstream_patched",
            "repair_scope_policy": "single_file_only",
            "prompt_contract_kind": "patched_tracecoder_dataset_prompt",
            "baseline_hybrid_guidance_mode": "fair_v1",
            "baseline_metrics": {
                "usage_available": False,
                "llm_call_count": 2,
            },
            "baseline_meta": {
                "error": "timeout",
            },
        },
    )
    trace_timeout["note"] = "no_patch_proposed"
    trace_success = _record(
        run_id="run-baseline",
        case_id="trace-success",
        condition="tracecoder",
        snapshot_dependency="required",
        success=True,
        tokens_total=33,
        llm_call_count=3,
        patch_meta={
            "fidelity_level": "upstream_patched",
            "repair_scope_policy": "single_file_only",
            "prompt_contract_kind": "patched_tracecoder_dataset_prompt",
            "baseline_hybrid_guidance_mode": "fair_v1",
            "baseline_metrics": {
                "usage_available": True,
                "llm_call_count": 3,
            },
        },
    )
    ldb_incompat = _record(
        run_id="run-baseline",
        case_id="ldb-incompat",
        condition="ldb",
        snapshot_dependency="required",
        success=False,
        patch_meta={
            "error": "case_incompatible",
            "compatibility_reason": "subprocess_task_unsupported",
            "fidelity_level": "upstream_patched",
            "repair_scope_policy": "single_file_only",
            "prompt_contract_kind": "upstream_native_prompt_contract",
            "baseline_metrics": {
                "usage_available": False,
                "llm_call_count": 0,
            },
        },
    )
    ldb_incompat["note"] = "no_patch_proposed"
    records = [trace_timeout, trace_success, ldb_incompat]
    diagnostics = _compute_failure_diagnostics(
        raw_records=records,
        condition_a="tracecoder",
        condition_b="ldb",
        top_n=5,
    )
    assert diagnostics["per_condition"]["tracecoder"]["usage_applicable_cases"] == 2
    assert diagnostics["per_condition"]["tracecoder"]["usage_reported_cases"] == 1
    assert diagnostics["per_condition"]["tracecoder"]["usage_missing_cases"] == 1
    assert diagnostics["per_condition"]["tracecoder"]["timeout_missing_usage_cases"] == 1
    assert diagnostics["per_condition"]["tracecoder"]["usage_not_applicable_cases"] == 0
    assert diagnostics["per_condition"]["ldb"]["usage_applicable_cases"] == 0
    assert diagnostics["per_condition"]["ldb"]["usage_missing_cases"] == 0
    assert diagnostics["per_condition"]["ldb"]["usage_not_applicable_cases"] == 1
    jsonl.write_text(
        "\n".join(json.dumps(record, ensure_ascii=True) for record in records) + "\n",
        encoding="utf-8",
    )

    rc = main(
        [
            "--jsonl",
            str(jsonl),
            "--a",
            "tracecoder",
            "--b",
            "ldb",
            "--bootstrap",
            "200",
            "--seed",
            "1",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "token telemetry coverage (terminal cases)" in out
    assert "reported=1/2 (50.00%), missing=1/2 (50.00%), timeout-backed missing=1/2 (50.00%)" in out
    assert (
        "reported=0/0 (n/a), missing=0/0 (n/a), timeout-backed missing=0/0 (n/a), not-applicable=1/1"
        in out
    )
    assert "baseline_error:timeout [runner_note=no_patch_proposed]:1" in out
    assert "case_incompatible:subprocess_task_unsupported [runner_note=no_patch_proposed]:1" in out
    assert "fidelity=upstream_patched:2" in out
    assert "repair_scope=single_file_only:2" in out
    assert "hybrid_guidance=fair_v1:2" in out


def test_main_baseline_diagnostics_prints_repair_fair_churn_metrics(tmp_path, capsys) -> None:
    jsonl = tmp_path / "rlm_repair_fair.jsonl"
    rlm_timeout = _record(
        run_id="run-rlm",
        case_id="rlm-timeout",
        condition="rlm",
        snapshot_dependency="required",
        success=False,
        llm_call_count=2,
        patch_meta={
            "fidelity_level": "generic_api_adaptation",
            "repair_scope_policy": "single_file_only",
            "prompt_contract_kind": (
                "adapter_defined_tool_augmented_prompt_with_repair_fallback__repair_fair_v1"
            ),
            "baseline_metrics": {
                "usage_available": False,
                "llm_call_count": 2,
                "stopped_reason": "iteration_budget_exhausted",
            },
            "baseline_meta": {
                "error": "timeout",
                "attempts": [
                    {
                        "attempt": 1,
                        "stop_reason": "iteration_budget_exhausted",
                        "no_write_buggy_py": True,
                        "test_run_count": 0,
                        "subcall_count": 0,
                    },
                    {
                        "attempt": 2,
                        "stop_reason": "iteration_budget_exhausted",
                        "no_write_buggy_py": False,
                        "test_run_count": 1,
                        "subcall_count": 2,
                    },
                ],
            },
        },
    )
    jsonl.write_text(json.dumps(rlm_timeout, ensure_ascii=True) + "\n", encoding="utf-8")

    rc = main(
        [
            "--jsonl",
            str(jsonl),
            "--a",
            "rlm",
            "--b",
            "rlm",
            "--bootstrap",
            "200",
            "--seed",
            "1",
        ]
    )
    assert rc == 0
    out = capsys.readouterr().out
    assert "terminal_timeout_rate=100.00% (coverage 1/1)" in out
    assert "iteration_budget_exhausted_attempts=2/2 (100.00%)" in out
    assert "no_write_attempts=1/2 (50.00%)" in out
    assert "no_test_attempts=1/2 (50.00%)" in out
    assert "subcalls=2/2 (1.0000/attempt)" in out
    assert "test_runs=1/2 (0.5000/attempt)" in out


def test_sta21_report_uses_and_prints_the_requested_ope_settings(
    tmp_path, capsys, monkeypatch
) -> None:
    """STA-21: the report always used the default OPE settings."""
    import evals.report_run_metrics as rrm

    jsonl = tmp_path / "ope.jsonl"
    records = [
        _record(
            run_id="r",
            case_id="c1",
            condition="base_repair",
            success=False,
            snapshot_dependency="required",
        ),
        _record(
            run_id="r",
            case_id="c1",
            condition="layered_adaptive_llm",
            success=True,
            snapshot_dependency="required",
        ),
    ]
    jsonl.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    seen: dict[str, object] = {}
    original = rrm.summarize_bucket

    def _spy(rows, **kwargs):
        seen.update(kwargs)
        return original(rows, **kwargs)

    monkeypatch.setattr(rrm, "summarize_bucket", _spy)

    rc = main(
        [
            "--jsonl",
            str(jsonl),
            "--bootstrap",
            "50",
            "--adaptive-ope-min-action-support",
            "5",
            "--adaptive-ope-propensity-floor",
            "0.1",
        ]
    )

    assert rc == 0
    assert seen["adaptive_ope_min_action_support"] == 5
    assert seen["adaptive_ope_propensity_floor"] == 0.1
    assert "Adaptive OPE settings: min_action_support=5, propensity_floor=0.1" in (
        capsys.readouterr().out
    )
    assert main(["--jsonl", str(jsonl), "--adaptive-ope-propensity-floor", "0"]) == 2
