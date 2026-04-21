"""Tests for the statistical analysis matrix in analyze_results."""

from __future__ import annotations

import pytest

from evals.analyze_results import _run_statistical_analysis, _summarize_bucket


def _rows() -> list[dict[str, object]]:
    return [
        {
            "run_id": "r1",
            "case_id": "c1",
            "condition": "base_repair",
            "category": "hidden_state",
            "difficulty": "easy",
            "snapshot_dependency": "required",
            "success": False,
            "attempts_to_success": None,
            "token_usage": 10.0,
            "runtime_sec": 1.0,
            "stage_b_prompt_chars_mean": 1000.0,
            "stage_a_request_attempts": 0,
            "stage_a_request_successes": 0,
            "retry_gate_attempts": 0,
            "retry_gate_blocks": 0,
            "context_protocol": "monolithic",
        },
        {
            "run_id": "r1",
            "case_id": "c1",
            "condition": "cot_repair",
            "category": "hidden_state",
            "difficulty": "easy",
            "snapshot_dependency": "required",
            "success": True,
            "attempts_to_success": 1,
            "token_usage": 11.0,
            "runtime_sec": 0.9,
            "stage_b_prompt_chars_mean": 900.0,
            "stage_a_request_attempts": 1,
            "stage_a_request_successes": 1,
            "retry_gate_attempts": 0,
            "retry_gate_blocks": 0,
            "context_protocol": "monolithic",
        },
        {
            "run_id": "r1",
            "case_id": "c1",
            "condition": "layered_adaptive_gated",
            "category": "hidden_state",
            "difficulty": "easy",
            "snapshot_dependency": "required",
            "success": True,
            "attempts_to_success": 1,
            "token_usage": 9.0,
            "runtime_sec": 0.8,
            "stage_b_prompt_chars_mean": 700.0,
            "stage_a_request_attempts": 1,
            "stage_a_request_successes": 1,
            "retry_gate_attempts": 1,
            "retry_gate_blocks": 0,
            "context_protocol": "staged",
        },
        {
            "run_id": "r1",
            "case_id": "c2",
            "condition": "base_repair",
            "category": "parse_value",
            "difficulty": "medium",
            "snapshot_dependency": "helpful",
            "success": False,
            "attempts_to_success": None,
            "token_usage": 12.0,
            "runtime_sec": 1.2,
            "stage_b_prompt_chars_mean": 1200.0,
            "stage_a_request_attempts": 0,
            "stage_a_request_successes": 0,
            "retry_gate_attempts": 0,
            "retry_gate_blocks": 0,
            "context_protocol": "monolithic",
        },
        {
            "run_id": "r1",
            "case_id": "c2",
            "condition": "cot_repair",
            "category": "parse_value",
            "difficulty": "medium",
            "snapshot_dependency": "helpful",
            "success": False,
            "attempts_to_success": None,
            "token_usage": 13.0,
            "runtime_sec": 1.1,
            "stage_b_prompt_chars_mean": 950.0,
            "stage_a_request_attempts": 1,
            "stage_a_request_successes": 1,
            "retry_gate_attempts": 0,
            "retry_gate_blocks": 0,
            "context_protocol": "monolithic",
        },
        {
            "run_id": "r1",
            "case_id": "c2",
            "condition": "layered_adaptive_gated",
            "category": "parse_value",
            "difficulty": "medium",
            "snapshot_dependency": "helpful",
            "success": True,
            "attempts_to_success": 1,
            "token_usage": 10.0,
            "runtime_sec": 1.0,
            "stage_b_prompt_chars_mean": 710.0,
            "stage_a_request_attempts": 1,
            "stage_a_request_successes": 1,
            "retry_gate_attempts": 1,
            "retry_gate_blocks": 0,
            "context_protocol": "staged",
        },
    ]


def test_summarize_bucket_exposes_pairwise_uplift_matrix() -> None:
    summary = _summarize_bucket(_rows())
    pairwise = summary["pairwise_uplifts"]
    assert isinstance(pairwise, list)
    assert len(pairwise) == 3

    by_pair = {(entry["condition_a"], entry["condition_b"]): entry for entry in pairwise}
    assert by_pair[("base_repair", "cot_repair")]["uplift_success_rate"] == pytest.approx(0.5)
    assert by_pair[("base_repair", "layered_adaptive_gated")][
        "uplift_success_rate"
    ] == pytest.approx(1.0)
    assert by_pair[("cot_repair", "layered_adaptive_gated")][
        "uplift_success_rate"
    ] == pytest.approx(0.5)


def test_statistical_analysis_includes_pairwise_matrix_entries() -> None:
    stats = _run_statistical_analysis(
        _rows(),
        aggregation="per_run",
        pair_a="base_repair",
        pair_b="cot_repair",
    )
    assert "pairwise" in stats
    pairwise = stats["pairwise"]
    assert isinstance(pairwise, list)
    assert len(pairwise) == 3
    pair_map = {(entry["condition_a"], entry["condition_b"]): entry for entry in pairwise}
    assert pair_map[("base_repair", "cot_repair")]["n_pairs"] == 2
    assert pair_map[("base_repair", "layered_adaptive_gated")]["n_pairs"] == 2
    assert pair_map[("cot_repair", "layered_adaptive_gated")]["n_pairs"] == 2
