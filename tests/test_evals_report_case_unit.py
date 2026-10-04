"""STA-02 in report_run_metrics: repeated runs of a case are one observation."""

from __future__ import annotations

from typing import Any

import pytest

from evals import report_run_metrics as report

_A, _B = "base_repair", "layered_adaptive_llm"


def _rows(run_ids: tuple[str, ...]) -> list[dict[str, Any]]:
    outcomes = [(False, True, 100.0, 80.0)] * 6 + [(True, False, 50.0, 90.0)] * 2
    outcomes += [(True, True, 40.0, 40.0)] * 12
    rows: list[dict[str, Any]] = []
    for run_id in run_ids:
        for idx, (a_ok, b_ok, a_tok, b_tok) in enumerate(outcomes):
            for condition, ok, tokens in ((_A, a_ok, a_tok), (_B, b_ok, b_tok)):
                rows.append(
                    {
                        "run_id": run_id,
                        "case_id": f"c{idx}",
                        "condition": condition,
                        "category": "logic",
                        "success": ok,
                        "token_usage_total": tokens,
                        "runtime_sec": tokens / 10.0,
                    }
                )
    return rows


def test_tripled_runs_give_the_same_paired_results_as_one_run() -> None:
    single, tripled = _rows(("r1",)), _rows(("r1", "r2", "r3"))

    for rows in (single, tripled):
        assert (
            report._paired_success_cost_frame(rows=rows, condition_a=_A, condition_b=_B).height
            == 20
        )
        deltas, _a, _b = report._paired_metric_deltas(
            rows=rows, condition_a=_A, condition_b=_B, metric="runtime_sec"
        )
        assert len(deltas) == 20

    def effect(rows: list[dict[str, Any]]) -> dict[str, Any]:
        return report._compute_effectiveness_summary(
            rows=rows, condition_a=_A, condition_b=_B, n_bootstrap=200, seed=1
        )

    assert effect(tripled)["mcnemar"]["p_value"] == pytest.approx(
        effect(single)["mcnemar"]["p_value"]
    )
    assert effect(tripled)["bootstrap_ci"] == effect(single)["bootstrap_ci"]
    kappa_single = report._paired_binary_agreement(rows=single, condition_a=_A, condition_b=_B)
    kappa_tripled = report._paired_binary_agreement(rows=tripled, condition_a=_A, condition_b=_B)
    assert kappa_tripled == kappa_single


def test_cost_frame_uses_success_fraction_over_runs() -> None:
    rows = _rows(("r1",))[:2] + [
        {**row, "run_id": "r2", "success": not row["success"]} for row in _rows(("r1",))[:2]
    ]
    frame = report._paired_success_cost_frame(rows=rows, condition_a=_A, condition_b=_B)
    assert frame["a_success"].to_list() == [0.5]
    assert frame["b_success"].to_list() == [0.5]


def test_sta10_undefined_kappa_is_reported_as_undefined() -> None:
    """Both conditions constant: kappa is NaN, which used to read "almost_perfect"."""
    import warnings

    from evals import _json as json

    rows = [
        {"run_id": "r1", "case_id": f"c{idx}", "condition": cond, "success": True}
        for idx in range(3)
        for cond in (_A, _B)
    ]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        agreement = report._paired_binary_agreement(rows=rows, condition_a=_A, condition_b=_B)
    assert agreement == {"available": True, "kappa": None, "magnitude": "undefined"}
    assert "NaN" not in json.dumps(agreement)


def test_sta20_secondary_slices_are_exploratory_and_holm_adjusted() -> None:
    from evals.stats import holm_bonferroni

    primary = {"available": True, "mcnemar": {"p_value": 0.01}}
    exploratory = {
        "all_cases": {"available": True, "mcnemar": {"p_value": 0.01}},
        "required_only": {"available": True, "mcnemar": {"p_value": 0.03}},
        "controls": {"available": False, "error": "no pairs"},
    }
    report._label_slice_family(primary_effect=primary, exploratory=exploratory)

    assert primary["role"] == "primary"
    assert "holm_adjusted_p" not in primary
    expected = holm_bonferroni({"all_cases": 0.01, "required_only": 0.03})
    for name in ("all_cases", "required_only"):
        assert exploratory[name]["role"] == "exploratory"
        assert exploratory[name]["holm_adjusted_p"] == expected[name]["adjusted_p"]
    assert exploratory["controls"]["holm_adjusted_p"] is None
