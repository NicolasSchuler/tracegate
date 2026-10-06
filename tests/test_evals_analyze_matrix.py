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


def _paired_rows(n: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for idx in range(n):
        for condition, success in (("base_repair", idx % 3 == 0), ("cot_repair", idx % 2 == 0)):
            rows.append(
                {
                    "run_id": "r1",
                    "case_id": f"c{idx}",
                    "condition": condition,
                    "category": "logic_error",
                    "success": success,
                }
            )
    return rows


def test_sta07_statistics_are_reproducible_and_seeded() -> None:
    """STA-07: the --stats bootstrap was unseeded, so reruns gave different CIs."""
    rows = _paired_rows(40)
    first = _run_statistical_analysis(rows, pair_a="base_repair", pair_b="cot_repair")
    second = _run_statistical_analysis(rows, pair_a="base_repair", pair_b="cot_repair")
    assert first["overall_bootstrap_ci"] == second["overall_bootstrap_ci"]
    assert first["overall_bootstrap_ci"]["seed"] == 42
    other = _run_statistical_analysis(rows, pair_a="base_repair", pair_b="cot_repair", seed=1)
    assert other["overall_bootstrap_ci"]["seed"] == 1


def test_sta07_bootstrap_seed_is_recorded_in_analysis_config(tmp_path, capsys) -> None:
    from evals import _json as json
    from evals.analyze_results import main
    from tests.test_evals_analyze_grouping import _record

    results = tmp_path / "results.jsonl"
    records = [
        _record(
            run_id="r1",
            case_id=str(row["case_id"]),
            condition=str(row["condition"]),
            attempt=1,
            success=bool(row["success"]),
            category="logic_error",
            difficulty="easy",
            snapshot_dependency="helpful",
        )
        for row in _paired_rows(6)
    ]
    results.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    out = tmp_path / "summary.json"

    # codex 2-1#2: --stats, so the recorded seed is the seed the bootstrap really used.
    argv = [
        str(results),
        "--bootstrap-seed",
        "7",
        "--stats",
        "--aggregation",
        "majority_vote",
        "--stats-pair-a",
        "base_repair",
        "--stats-pair-b",
        "cot_repair",
        "--out",
        str(out),
    ]
    assert main(argv) == 0

    summary = json.loads(out.read_text(encoding="utf-8"))
    config = summary["analysis_config"]
    assert config["bootstrap_seed"] == 7
    assert config["bootstrap_method"] == "percentile"
    statistics = summary["statistics"]
    assert statistics["overall_bootstrap_ci"]["seed"] == 7
    assert statistics["overall_bootstrap_ci"]["method"] == "percentile"
    assert {entry["bootstrap_ci"]["seed"] for entry in statistics["per_category"].values()} == {7}
    capsys.readouterr()


def test_sta06_pairwise_tests_are_holm_adjusted_as_one_family() -> None:
    """STA-06: 36 uncorrected McNemar tests for nine conditions."""
    from evals.stats import holm_bonferroni

    rows: list[dict[str, object]] = []
    for idx in range(30):
        for condition, success in (
            ("base_repair", idx % 3 == 0),
            ("cot_repair", idx % 2 == 0),
            ("layered_adaptive_llm", idx < 25),
        ):
            rows.append(
                {
                    "run_id": "r1",
                    "case_id": f"c{idx}",
                    "condition": condition,
                    "category": "logic_error",
                    "success": success,
                }
            )
    result = _run_statistical_analysis(rows)
    pairwise = result["pairwise"]
    assert result["pairwise_correction"] == "holm_across_all_condition_pairs"
    expected = holm_bonferroni(
        {f"{e['condition_a']}|{e['condition_b']}": e["mcnemar_p_value"] for e in pairwise}
    )
    for entry in pairwise:
        holm = expected[f"{entry['condition_a']}|{entry['condition_b']}"]
        assert entry["mcnemar_p_adjusted"] == holm["adjusted_p"]
        assert entry["holm_rejected"] == holm["rejected"]
        assert entry["mcnemar_p_adjusted"] >= entry["mcnemar_p_value"]
