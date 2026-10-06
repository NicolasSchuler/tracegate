"""STA-19 / DAT-24: one case-outcome definition and one condition registry."""

from __future__ import annotations

from typing import Any

import pytest

from evals import select_cases
from evals.analyze_results import _derive_case_condition_rows
from evals.case_outcome import case_outcome, case_outcomes, scored_attempts
from evals.eval_types import (
    CONDITION_FAMILIES,
    Condition,
    condition_family,
    normalize_condition_name,
)


def _row(attempt: int, success: bool, **extra: Any) -> dict[str, Any]:
    return {
        "run_id": "r1",
        "case_id": "c1",
        "condition": "base_repair",
        "attempt": attempt,
        "success": success,
        **extra,
    }


_RUNTIME_ERROR = {"note": "unit_runtime_error", "patch_meta": {"infra_resume_eligible": True}}


def test_success_is_any_scored_attempt_and_terminal_is_the_highest() -> None:
    outcome = case_outcome([_row(2, False), _row(1, True), _row(3, False)])
    assert outcome is not None
    assert outcome.success is True
    assert outcome.success_attempt == 1
    assert outcome.terminal_attempt == 3


def test_infra_events_are_not_attempts() -> None:
    assert case_outcome([_row(1, False, record_type="infra_event")]) is None
    outcome = case_outcome([_row(1, False, record_type="infra_event"), _row(1, True)])
    assert outcome is not None
    assert outcome.success is True


def test_runtime_error_row_is_a_failed_attempt_that_resume_retries() -> None:
    outcome = case_outcome([_row(1, False, **_RUNTIME_ERROR)])
    assert outcome is not None
    assert outcome.success is False
    assert outcome.resume_eligible is True


@pytest.mark.parametrize("real_first", [True, False])
def test_duplicate_attempt_prefers_the_real_outcome(real_first: bool) -> None:
    """DAT-24: the choice no longer depends on JSONL order."""
    real = _row(1, True)
    error = _row(1, False, **_RUNTIME_ERROR)
    rows = [real, error] if real_first else [error, real]
    assert scored_attempts(rows) == [real]


@pytest.mark.parametrize("success_first", [True, False])
def test_conflicting_equal_priority_duplicates_resolve_the_same_in_both_readers(
    tmp_path: Any, success_first: bool
) -> None:
    """DAT-24: both readers share one resolver and report the conflict (codex 2-1#1)."""
    from evals import _json as json
    from evals.analysis_records import _load_and_group_records
    from tests.test_evals_analyze_grouping import _record

    def record(success: bool) -> dict[str, Any]:
        return _record(
            run_id="r1",
            case_id="c1",
            condition="base_repair",
            attempt=1,
            success=success,
            category="logic_error",
            difficulty="easy",
            snapshot_dependency="helpful",
        )

    ordered = [record(True), record(False)] if success_first else [record(False), record(True)]
    results = tmp_path / "results.jsonl"
    results.write_text("".join(json.dumps(r) + "\n" for r in ordered), encoding="utf-8")

    grouped, input_errors = _load_and_group_records([str(results)])
    assert input_errors["duplicate_attempt_rows_conflicting"] == 1
    loaded = grouped[("r1", "c1", "base_repair")]
    # The one surviving row is the same one the case-outcome reader keeps.
    assert [row["success"] for row in loaded] == [bool(ordered[-1]["success"])]
    assert [row["success"] for row in scored_attempts(ordered)] == [bool(ordered[-1]["success"])]

    with pytest.raises(ValueError, match="conflicting duplicate"):
        _load_and_group_records([str(results)], strict_input=True)


def test_case_outcomes_groups_by_run_case_and_stripped_condition() -> None:
    rows = [
        _row(1, True),
        {**_row(1, False), "run_id": "r2"},
        {**_row(1, False), "condition": " base_repair "},
        {**_row(1, True), "case_id": ""},
    ]
    outcomes = case_outcomes(rows)
    assert set(outcomes) == {("r1", "c1", "base_repair"), ("r2", "c1", "base_repair")}
    # " base_repair " is the same condition, so its row is a later duplicate of attempt 1.
    assert outcomes[("r1", "c1", "base_repair")].attempts == (rows[2],)
    assert set(case_outcomes(rows, condition="cot_repair")) == set()


def test_select_cases_and_analyze_results_agree_on_every_case() -> None:
    """STA-19: a case could succeed in the summary and appear in <run>.failed.txt."""
    rows = [
        _row(1, True),
        _row(3, False),  # a later failed row after a success
        {**_row(1, False, **_RUNTIME_ERROR), "case_id": "c2"},
        {**_row(1, False), "case_id": "c3"},
        {**_row(2, True), "case_id": "c3"},
    ]
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for row in rows:
        grouped.setdefault((row["run_id"], row["case_id"], row["condition"]), []).append(row)
    analyzed = {row["case_id"]: row["success"] for row in _derive_case_condition_rows(grouped)}
    terminal = select_cases._terminal_by_case(rows=rows, condition="base_repair")
    selected = {case_id: row["success"] for case_id, row in terminal.items()}
    assert analyzed == selected == {"c1": True, "c2": False, "c3": True}


def test_condition_registry_covers_every_condition() -> None:
    assert set(CONDITION_FAMILIES) == {condition.value for condition in Condition}
    assert condition_family("layered_adaptive_llm") == "adaptive"
    assert condition_family(" ldb ") == "external_baseline"
    assert condition_family("base_repair") == "native"
    assert condition_family("custom_adaptive_v2") == "unknown"
    assert normalize_condition_name(None) == ""
