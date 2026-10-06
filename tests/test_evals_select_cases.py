"""Tests for eval case selection and failure-based filtering from run logs."""

from __future__ import annotations

from pathlib import Path

import pytest

from evals import select_cases
from llmdebug import _json as json


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")


def test_select_cases_failed_mode_with_duplicate_attempt_last_write_wins(tmp_path, capsys) -> None:
    input_path = tmp_path / "run.jsonl"
    output_path = tmp_path / "failed.txt"
    _write_jsonl(
        input_path,
        [
            # case_a attempt 1 originally failing, then duplicate attempt row marks success.
            {"case_id": "case_a", "condition": "base_repair", "attempt": 1, "success": False},
            {"case_id": "case_a", "condition": "base_repair", "attempt": 1, "success": True},
            {"case_id": "case_b", "condition": "base_repair", "attempt": 1, "success": False},
            {"case_id": "case_c", "condition": "base_repair", "attempt": 2, "success": True},
            {"case_id": "case_d", "condition": "cot_repair", "attempt": 3, "success": False},
        ],
    )

    exit_code = select_cases.main(
        [
            "--results-jsonl",
            str(input_path),
            "--condition",
            "base_repair",
            "--mode",
            "failed",
            "--out",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert output_path.read_text(encoding="utf-8") == "case_b\n"
    assert "Selected 1 case(s)" in capsys.readouterr().out


def test_select_cases_attempt_ge_mode(tmp_path) -> None:
    input_path = tmp_path / "run.jsonl"
    output_path = tmp_path / "attempt_ge.txt"
    _write_jsonl(
        input_path,
        [
            {"case_id": "case_a", "condition": "base_repair", "attempt": 1, "success": True},
            {"case_id": "case_b", "condition": "base_repair", "attempt": 2, "success": True},
            {"case_id": "case_c", "condition": "base_repair", "attempt": 3, "success": False},
        ],
    )

    exit_code = select_cases.main(
        [
            "--results-jsonl",
            str(input_path),
            "--condition",
            "base_repair",
            "--mode",
            "attempt-ge",
            "--attempt-threshold",
            "2",
            "--out",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert output_path.read_text(encoding="utf-8") == "case_b\ncase_c\n"


def test_select_cases_failed_or_attempt_ge_mode(tmp_path) -> None:
    input_path = tmp_path / "run.jsonl"
    output_path = tmp_path / "union.txt"
    _write_jsonl(
        input_path,
        [
            {"case_id": "case_a", "condition": "base_repair", "attempt": 1, "success": False},
            {"case_id": "case_b", "condition": "base_repair", "attempt": 2, "success": True},
            {"case_id": "case_c", "condition": "base_repair", "attempt": 1, "success": True},
        ],
    )

    exit_code = select_cases.main(
        [
            "--results-jsonl",
            str(input_path),
            "--condition",
            "base_repair",
            "--mode",
            "failed-or-attempt-ge",
            "--attempt-threshold",
            "2",
            "--out",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert output_path.read_text(encoding="utf-8") == "case_a\ncase_b\n"


def test_select_cases_fails_when_condition_missing(tmp_path, capsys) -> None:
    input_path = tmp_path / "run.jsonl"
    output_path = tmp_path / "missing.txt"
    _write_jsonl(
        input_path,
        [
            {"case_id": "case_a", "condition": "cot_repair", "attempt": 1, "success": False},
        ],
    )

    exit_code = select_cases.main(
        [
            "--results-jsonl",
            str(input_path),
            "--condition",
            "base_repair",
            "--out",
            str(output_path),
        ]
    )

    assert exit_code == 2
    assert "No rows found for condition='base_repair'" in capsys.readouterr().out


def test_select_cases_failed_mode_includes_request_budget_exceeded(tmp_path, capsys) -> None:
    input_path = tmp_path / "run.jsonl"
    output_path = tmp_path / "failed.txt"
    _write_jsonl(
        input_path,
        [
            {
                "case_id": "case_budget_failed",
                "condition": "base_repair",
                "attempt": 2,
                "success": False,
                "note": "patcher_request_failed",
                "patch_meta": {"patcher_request_failure_error": "request_budget_exceeded"},
            },
            {
                "case_id": "case_tests_failed",
                "condition": "base_repair",
                "attempt": 1,
                "success": False,
                "note": "tests_still_failing",
            },
            {
                "case_id": "case_success",
                "condition": "base_repair",
                "attempt": 1,
                "success": True,
                "note": "success",
            },
        ],
    )

    exit_code = select_cases.main(
        [
            "--results-jsonl",
            str(input_path),
            "--condition",
            "base_repair",
            "--mode",
            "failed",
            "--out",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert output_path.read_text(encoding="utf-8") == "case_budget_failed\ncase_tests_failed\n"
    captured = capsys.readouterr().out
    assert "Terminal request_budget_exceeded failures: 1 (selected=1)." in captured


def test_select_cases_failed_mode_respects_max_attempt_filter(tmp_path) -> None:
    input_path = tmp_path / "run.jsonl"
    output_path = tmp_path / "failed.txt"
    _write_jsonl(
        input_path,
        [
            {
                "case_id": "case_failed_attempt_2",
                "condition": "base_repair",
                "attempt": 2,
                "success": False,
            },
            {
                "case_id": "case_failed_attempt_3",
                "condition": "base_repair",
                "attempt": 3,
                "success": False,
            },
        ],
    )

    exit_code = select_cases.main(
        [
            "--results-jsonl",
            str(input_path),
            "--condition",
            "base_repair",
            "--mode",
            "failed",
            "--max-attempt",
            "2",
            "--out",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert output_path.read_text(encoding="utf-8") == "case_failed_attempt_2\n"


def test_select_cases_ignores_infra_event_rows(tmp_path) -> None:
    input_path = tmp_path / "run.jsonl"
    output_path = tmp_path / "failed.txt"
    _write_jsonl(
        input_path,
        [
            {
                "record_type": "infra_event",
                "case_id": "case_a",
                "condition": "base_repair",
                "attempt": 1,
                "success": False,
                "note": "infra_transport_failure",
            },
            {
                "case_id": "case_a",
                "condition": "base_repair",
                "attempt": 1,
                "success": True,
                "note": "success",
            },
        ],
    )

    exit_code = select_cases.main(
        [
            "--results-jsonl",
            str(input_path),
            "--condition",
            "base_repair",
            "--mode",
            "failed",
            "--out",
            str(output_path),
        ]
    )

    assert exit_code == 0
    assert output_path.read_text(encoding="utf-8") == ""


# ---------------------------------------------------------------------------
# _load_jsonl edge cases
# ---------------------------------------------------------------------------


def test_load_jsonl_skips_empty_lines(tmp_path) -> None:
    """JSONL files with interspersed blank lines should be parsed correctly."""
    path = tmp_path / "data.jsonl"
    path.write_text(
        '\n{"a":1}\n\n\n{"b":2}\n\n',
        encoding="utf-8",
    )
    rows = select_cases._load_jsonl(path)
    assert rows == [{"a": 1}, {"b": 2}]


def test_load_jsonl_raises_on_non_object_rows(tmp_path) -> None:
    """Non-object JSON values (arrays, strings, numbers) must raise ValueError."""
    for payload in ["[1,2,3]", '"hello"', "42"]:
        path = tmp_path / "bad.jsonl"
        path.write_text(payload + "\n", encoding="utf-8")
        with pytest.raises(ValueError, match="expected JSON object row"):
            select_cases._load_jsonl(path)


def test_load_jsonl_raises_on_malformed_json(tmp_path) -> None:
    """Malformed JSON triggers a ValueError with line number."""
    path = tmp_path / "bad.jsonl"
    path.write_text('{"ok":1}\n{bad json\n', encoding="utf-8")
    with pytest.raises(ValueError, match=r":2:"):
        select_cases._load_jsonl(path)


# ---------------------------------------------------------------------------
# _terminal_by_case — infra_event skip & empty case_id
# ---------------------------------------------------------------------------


def test_terminal_by_case_skips_infra_event_rows() -> None:
    rows = [
        {
            "record_type": "infra_event",
            "case_id": "c1",
            "condition": "base_repair",
            "attempt": 1,
            "success": False,
        },
        {
            "case_id": "c1",
            "condition": "base_repair",
            "attempt": 1,
            "success": True,
        },
    ]
    terminal = select_cases._terminal_by_case(rows=rows, condition="base_repair")
    assert terminal["c1"]["success"] is True


def test_terminal_by_case_skips_empty_case_id() -> None:
    rows = [
        {"case_id": "", "condition": "base_repair", "attempt": 1, "success": False},
        {"case_id": "  ", "condition": "base_repair", "attempt": 1, "success": False},
    ]
    terminal = select_cases._terminal_by_case(rows=rows, condition="base_repair")
    assert terminal == {}


# ---------------------------------------------------------------------------
# _coerce_attempt edge cases
# ---------------------------------------------------------------------------


def test_coerce_attempt_non_numeric_returns_zero() -> None:
    assert select_cases._coerce_attempt("abc") == 0
    assert select_cases._coerce_attempt(None) == 0


# ---------------------------------------------------------------------------
# Budget-exceeded detection via attempt_metadata
# ---------------------------------------------------------------------------


def test_budget_exceeded_via_attempt_metadata() -> None:
    """Budget failure detection falls through to attempt_metadata path."""
    row = {
        "success": False,
        "note": "patcher_request_failed",
        "attempt_metadata": {"error": "request_budget_exceeded"},
    }
    assert select_cases._is_request_budget_exceeded_failure(row) is True


def test_budget_exceeded_not_triggered_on_success() -> None:
    row = {
        "success": True,
        "note": "patcher_request_failed",
        "patch_meta": {"error": "request_budget_exceeded"},
    }
    assert select_cases._is_request_budget_exceeded_failure(row) is False


def test_budget_exceeded_not_triggered_on_other_note() -> None:
    row = {
        "success": False,
        "note": "tests_still_failing",
        "patch_meta": {"error": "request_budget_exceeded"},
    }
    assert select_cases._is_request_budget_exceeded_failure(row) is False


# ---------------------------------------------------------------------------
# CLI main() function
# ---------------------------------------------------------------------------


def test_main_file_not_found(tmp_path, capsys) -> None:
    exit_code = select_cases.main(
        [
            "--results-jsonl",
            str(tmp_path / "nonexistent.jsonl"),
            "--out",
            str(tmp_path / "out.txt"),
        ]
    )
    assert exit_code == 2
    assert "not found" in capsys.readouterr().out.lower()


def test_main_malformed_jsonl(tmp_path, capsys) -> None:
    bad = tmp_path / "bad.jsonl"
    bad.write_text("{bad\n", encoding="utf-8")
    exit_code = select_cases.main(["--results-jsonl", str(bad), "--out", str(tmp_path / "out.txt")])
    assert exit_code == 2
    assert "invalid JSON" in capsys.readouterr().out


def test_main_empty_condition(tmp_path, capsys) -> None:
    """An empty --condition string should return exit 2."""
    inp = tmp_path / "run.jsonl"
    _write_jsonl(inp, [{"case_id": "c1", "condition": "x", "attempt": 1, "success": False}])
    exit_code = select_cases.main(
        [
            "--results-jsonl",
            str(inp),
            "--condition",
            " ",
            "--out",
            str(tmp_path / "out.txt"),
        ]
    )
    assert exit_code == 2


# ---------------------------------------------------------------------------
# Branch coverage: _terminal_by_case — later row with lower attempt kept
# ---------------------------------------------------------------------------


def test_terminal_by_case_keeps_higher_attempt_when_lower_follows() -> None:
    """Branch [69→55]: a row with attempt < prev_attempt does NOT overwrite."""
    rows = [
        {"case_id": "c1", "condition": "base_repair", "attempt": 3, "success": False},
        {"case_id": "c1", "condition": "base_repair", "attempt": 1, "success": True},
    ]
    terminal = select_cases._terminal_by_case(rows=rows, condition="base_repair")
    # The attempt-3 row is the terminal row because 1 < 3, but the case succeeded at
    # attempt 1. STA-19: this used to say failed while analyze_results said solved.
    assert terminal["c1"]["attempt"] == 3
    assert terminal["c1"]["success"] is True


# ---------------------------------------------------------------------------
# Branch coverage: _select_case_ids — failed-or-attempt-ge, no match
# ---------------------------------------------------------------------------


def test_select_case_ids_failed_or_attempt_ge_excludes_success_below_threshold() -> None:
    """Branch [104→106]: mode=failed-or-attempt-ge, case is success AND below threshold."""
    terminal = {
        "c1": {"success": True, "attempt": 1},
    }
    result = select_cases._select_case_ids(
        terminal_rows=terminal,
        mode="failed-or-attempt-ge",
        attempt_threshold=2,
        max_attempt=0,
    )
    assert result == []


def test_select_case_ids_unknown_mode_selects_nothing() -> None:
    """Branch [104→106]: unknown mode bypasses all if/elif branches, include stays False."""
    terminal = {
        "c1": {"success": False, "attempt": 5},
    }
    result = select_cases._select_case_ids(
        terminal_rows=terminal,
        mode="nonexistent-mode",
        attempt_threshold=1,
        max_attempt=0,
    )
    assert result == []


# ---------------------------------------------------------------------------
# Branch coverage: _is_request_budget_exceeded_failure — patch_meta non-budget
# ---------------------------------------------------------------------------


def test_budget_exceeded_false_when_patch_meta_has_different_error() -> None:
    """Branch [124→127]: patch_meta error != request_budget_exceeded falls through."""
    row = {
        "success": False,
        "note": "patcher_request_failed",
        "patch_meta": {"error": "rate_limit_exceeded"},
    }
    assert select_cases._is_request_budget_exceeded_failure(row) is False


def test_budget_exceeded_false_when_attempt_metadata_has_different_error() -> None:
    """Branch [132→135]: attempt_metadata error != request_budget_exceeded returns False."""
    row = {
        "success": False,
        "note": "patcher_request_failed",
        "attempt_metadata": {"error": "connection_timeout"},
    }
    assert select_cases._is_request_budget_exceeded_failure(row) is False


def test_sta19_runtime_error_terminal_row_is_a_failure_not_a_missing_case() -> None:
    """A unit_runtime_error row stays in the denominator (ATT-08) instead of vanishing."""
    rows = [
        {
            "case_id": "c1",
            "condition": "base_repair",
            "attempt": 1,
            "success": False,
            "note": "unit_runtime_error",
            "patch_meta": {"infra_resume_eligible": True},
        }
    ]
    terminal = select_cases._terminal_by_case(rows=rows, condition="base_repair")
    assert terminal["c1"]["success"] is False


def test_dat24_duplicate_attempt_prefers_the_real_outcome_over_a_runtime_error() -> None:
    """DAT-24: a resumed unit's rows used to be chosen by file order alone."""
    runtime_error = {
        "case_id": "c1",
        "condition": "base_repair",
        "attempt": 1,
        "success": False,
        "patch_meta": {"infra_resume_eligible": True},
    }
    real = {"case_id": "c1", "condition": "base_repair", "attempt": 1, "success": True}
    for rows in ([runtime_error, real], [real, runtime_error]):
        terminal = select_cases._terminal_by_case(rows=rows, condition="base_repair")
        assert terminal["c1"]["success"] is True


def test_sta19_condition_names_are_stripped() -> None:
    rows = [{"case_id": "c1", "condition": " base_repair ", "attempt": 1, "success": False}]
    assert set(select_cases._terminal_by_case(rows=rows, condition="base_repair")) == {"c1"}
