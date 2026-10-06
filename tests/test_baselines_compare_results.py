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


def _row(run_id: str, case_id: str, condition: str, success: bool, **extra: object) -> dict:
    return {
        "run_id": run_id,
        "case_id": case_id,
        "condition": condition,
        "success": success,
        "category": "logic",
        **extra,
    }


def test_separately_named_runs_still_pair_on_shared_cases() -> None:
    """codex 2-2#2: a native run and a --baseline run have different run IDs."""
    rows = [
        _row("native-run", "case_a", "base_repair", True),
        _row("native-run", "case_b", "base_repair", False),
        _row("ldb-run", "case_a", "ldb", False),
        _row("ldb-run", "case_b", "ldb", True),
    ]

    result = _compute_pairwise(rows, "base_repair", "ldb", n_bootstrap=32, seed=7)

    assert "error" not in result
    assert result["n_cases"] == 2
    assert result["runs_per_case_max"] == 1
    assert result["unpaired_runs"] == 0
    assert (result["a_only_wins"], result["b_only_wins"]) == (1, 1)


def test_replicates_of_separate_runs_pair_by_index() -> None:
    """Repeated native and baseline invocations pair replicate i with replicate i."""
    rows = [
        _row("native-1", "case_a", "base_repair", True),
        _row("native-2", "case_a", "base_repair", True),
        _row("native-3", "case_a", "base_repair", False),
        _row("ldb-1", "case_a", "ldb", False),
        _row("ldb-2", "case_a", "ldb", False),
        # A ran case_a three times, B twice: the third A run has no counterpart.
    ]

    result = _compute_pairwise(rows, "base_repair", "ldb", n_bootstrap=32, seed=7)

    assert result["n_cases"] == 1
    assert result["runs_per_case_max"] == 2
    assert result["unpaired_runs"] == 1
    assert (result["a_only_wins"], result["b_only_wins"]) == (1, 0)


def test_bas01_unsupported_baseline_rows_are_not_losing_pairs() -> None:
    """codex 1e#5: the comparison helper counted baseline_supported=False as a failure."""
    rows = [
        _row("native-run", "case_a", "base_repair", True),
        _row("ldb-run", "case_a", "ldb", False, baseline_supported=False),
    ]

    result = _compute_pairwise(rows, "base_repair", "ldb", n_bootstrap=16, seed=3)

    assert result["error"] == "no_common_cases"
    assert result["n_cases"] == 0


def test_load_results_warns_about_dropped_and_mixed_version_records(tmp_path, capsys) -> None:
    """codex 2-2#5 / 1e#13: compare_results ignored every counter but unreadable files."""
    from baselines.compare_results import _load_results
    from evals import _json as json
    from tests.test_evals_report_run_metrics import _schema_record, _v7_record

    jsonl = tmp_path / "mixed.jsonl"
    jsonl.write_text(
        "".join(
            json.dumps(record) + "\n"
            for record in (
                _schema_record("c1", "base_repair", True),
                _v7_record("c1", "ldb", False),
                _schema_record("c2", "base_repair", True, attempt="nope"),
            )
        )
        + "{not json}\n",
        encoding="utf-8",
    )

    rows = _load_results([str(jsonl)])
    err = capsys.readouterr().err

    assert len(rows) == 2
    assert "records_schema_invalid=1" in err
    assert "json_lines_invalid=1" in err
    assert "several schema versions (5, 7)" in err
