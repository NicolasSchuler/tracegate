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
