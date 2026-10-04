"""Regression tests for cross-baseline comparison reporting."""

from __future__ import annotations

from baselines.compare_results import _compute_pairwise, compare_all


def test_compute_pairwise_accepts_overlapping_case_rows() -> None:
    rows = [
        {"case_id": "case_a", "condition": "base_repair", "success": True, "category": "logic"},
        {"case_id": "case_a", "condition": "ldb", "success": False, "category": "logic"},
        {"case_id": "case_b", "condition": "base_repair", "success": False, "category": "logic"},
        {"case_id": "case_b", "condition": "ldb", "success": True, "category": "logic"},
    ]

    result = _compute_pairwise(rows, "base_repair", "ldb", n_bootstrap=32, seed=7)

    assert result["n_cases"] == 2
    assert result["a_only_wins"] == 1
    assert result["b_only_wins"] == 1
    assert result["mcnemar"]["n_discordant"] == 2
    assert result["bootstrap_ci"]["n_bootstrap"] == 32


def test_compare_all_handles_baseline_condition_labels() -> None:
    rows = [
        {"case_id": "case_a", "condition": "base_repair", "success": True, "category": "logic"},
        {
            "case_id": "case_a",
            "condition": "ldb",
            "success": False,
            "category": "logic",
            "baseline_fidelity_level": "upstream_patched",
            "baseline_repair_scope_policy": "single_file_only",
            "baseline_prompt_contract_kind": "upstream_native_prompt_contract",
        },
        {"case_id": "case_b", "condition": "base_repair", "success": False, "category": "logic"},
        {
            "case_id": "case_b",
            "condition": "ldb",
            "success": True,
            "category": "logic",
            "baseline_fidelity_level": "upstream_patched",
            "baseline_repair_scope_policy": "single_file_only",
            "baseline_prompt_contract_kind": "upstream_native_prompt_contract",
        },
        {
            "case_id": "case_a",
            "condition": "intervenor",
            "success": True,
            "category": "logic",
            "baseline_fidelity_level": "prompt_asset_port",
            "baseline_repair_scope_policy": "single_file_only",
            "baseline_prompt_contract_kind": "upstream_prompt_assets_plus_adapter_instructions",
        },
        {
            "case_id": "case_b",
            "condition": "intervenor",
            "success": True,
            "category": "logic",
            "baseline_fidelity_level": "prompt_asset_port",
            "baseline_repair_scope_policy": "single_file_only",
            "baseline_prompt_contract_kind": "upstream_prompt_assets_plus_adapter_instructions",
        },
    ]

    result = compare_all(
        rows,
        ["base_repair", "ldb", "intervenor"],
        "base_repair",
        n_bootstrap=16,
        seed=11,
    )

    assert result["conditions_available"] == [
        "base_repair",
        "intervenor",
        "ldb",
    ]
    assert len(result["pairwise"]) == 2
    compared = {(item["condition_a"], item["condition_b"]) for item in result["pairwise"]}
    assert compared == {
        ("base_repair", "ldb"),
        ("base_repair", "intervenor"),
    }
    assert result["baseline_contracts"]["ldb"]["fidelity_level"] == "upstream_patched"
    assert result["baseline_contracts"]["intervenor"]["fidelity_level"] == "prompt_asset_port"


def test_bas06_repeated_runs_are_aggregated_not_overwritten() -> None:
    """BAS-06: the last run in glob order overwrote the others."""
    rows = [
        {"run_id": run_id, "case_id": case_id, "condition": cond, "success": ok, "category": cat}
        for run_id, outcomes in (
            ("r1", {"case_a": (False, True), "case_b": (True, True)}),
            ("r2", {"case_a": (False, True), "case_b": (True, False)}),
            ("r3", {"case_a": (False, False), "case_b": (True, False)}),
        )
        for case_id, (a_ok, b_ok) in outcomes.items()
        for cond, ok, cat in (("base_repair", a_ok, "logic"), ("ldb", b_ok, "logic"))
    ]

    result = compare_all(rows, ["base_repair", "ldb"], "base_repair", n_bootstrap=16, seed=1)
    [pair] = result["pairwise"]
    assert pair["n_cases"] == 2
    assert pair["runs_per_case_max"] == 3
    # Majority over 3 runs: case_a B wins (2/3), case_b A wins (B 1/3).
    assert (pair["a_only_wins"], pair["b_only_wins"]) == (1, 1)
    assert result["per_category"]["base_repair_vs_ldb"]["logic"]["a_cases"] == pair["n_cases"]


def test_bas09_unreadable_file_warns_instead_of_crashing(tmp_path, capsys) -> None:
    from baselines.compare_results import _load_results

    assert _load_results([str(tmp_path)]) == []  # a directory can't be opened as JSONL
    assert "1 unreadable file(s)" in capsys.readouterr().err
