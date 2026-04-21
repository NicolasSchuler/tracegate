"""Additional coverage tests for ``evals.report_run_metrics``."""

from __future__ import annotations

import json
import math
import textwrap
from pathlib import Path
from typing import Any
from unittest.mock import patch

import pytest

from evals.report_run_metrics import (
    _annotate_runner_note,
    _append_covered_pct,
    _as_float,
    _baseline_diagnostic_lines,
    _baseline_llm_call_count,
    _baseline_terminal_usage_state,
    _cohens_kappa_magnitude,
    _compute_category_stats,
    _conclusion_line,
    _failure_reason,
    _fmt_float,
    _fmt_pct,
    _fmt_pct_signed,
    _fmt_signed_pp,
    _mean,
    _median,
    _null_sort_key,
    _parse_eval_log,
    _parse_job_feedback_block,
    _print_effect_line,
    _print_input_errors,
    _print_overhead_analysis,
    _ratio,
    _terminal_failure_reason,
    main,
)

# ── Shared helpers ───────────────────────────────────────────────────────


def _record(
    *,
    run_id: str = "run",
    case_id: str = "case",
    condition: str = "base_repair",
    snapshot_dependency: str = "required",
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
    note: str = "tests_still_failing",
) -> dict[str, object]:
    record: dict[str, object] = {
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
        "note": note,
    }
    if patch_meta is not None:
        record["patch_meta"] = patch_meta
    return record


def _write_jsonl(path: Path, records: list[dict[str, object]]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.writelines(json.dumps(rec) + "\n" for rec in records)


# ── main() early-exit paths (lines 119-137) ─────────────────────────────


class TestMainExitPaths:
    def test_no_match_returns_1(self, tmp_path: Path) -> None:
        result = main(["--jsonl", str(tmp_path / "nonexistent*.jsonl")])
        assert result == 1

    def test_strict_input_error_returns_2(self, tmp_path: Path) -> None:
        p = tmp_path / "bad.jsonl"
        p.write_text("not json\n", encoding="utf-8")
        result = main(["--jsonl", str(p), "--strict-input"])
        assert result == 2

    def test_empty_records_returns_1(self, tmp_path: Path) -> None:
        p = tmp_path / "empty.jsonl"
        p.write_text("\n", encoding="utf-8")
        result = main(["--jsonl", str(p)])
        assert result == 1

    def test_no_valid_rows_returns_1(self, tmp_path: Path) -> None:
        rec = _record()
        rec["case_id"] = ""
        p = tmp_path / "data.jsonl"
        _write_jsonl(p, [rec])
        result = main(["--jsonl", str(p)])
        assert result == 1


# ── _cohens_kappa_magnitude (lines 348-354) ─────────────────────────────


class TestCohensKappaMagnitude:
    def test_slight(self) -> None:
        assert _cohens_kappa_magnitude(0.1) == "slight"

    def test_fair(self) -> None:
        assert _cohens_kappa_magnitude(0.3) == "fair"

    def test_moderate(self) -> None:
        assert _cohens_kappa_magnitude(0.5) == "moderate"

    def test_substantial(self) -> None:
        assert _cohens_kappa_magnitude(0.7) == "substantial"

    def test_almost_perfect(self) -> None:
        assert _cohens_kappa_magnitude(0.9) == "almost_perfect"

    def test_negative_slight(self) -> None:
        assert _cohens_kappa_magnitude(-0.1) == "slight"


# ── _failure_reason (lines 854-863) ─────────────────────────────────────


class TestFailureReason:
    def test_overflow(self) -> None:
        assert (
            _failure_reason(error="", status="", note="", overflow=True)
            == "http_error:context_exceeded"
        )

    def test_error_and_status(self) -> None:
        assert (
            _failure_reason(error="http_error", status="429", note="", overflow=False)
            == "http_error:429"
        )

    def test_error_only(self) -> None:
        assert _failure_reason(error="timeout", status="", note="", overflow=False) == "timeout"

    def test_note_only(self) -> None:
        assert (
            _failure_reason(error="", status="", note="something", overflow=False)
            == "note:something"
        )

    def test_unknown(self) -> None:
        assert _failure_reason(error="", status="", note="", overflow=False) == "unknown"


# ── _baseline_llm_call_count fallbacks (lines 903-915) ──────────────────


class TestBaselineLlmCallCount:
    def test_from_baseline_metrics(self) -> None:
        record: dict[str, Any] = {"patch_meta": {"baseline_metrics": {"llm_call_count": 3}}}
        assert _baseline_llm_call_count(record) == 3

    def test_fallback_to_patch_meta(self) -> None:
        record: dict[str, Any] = {
            "patch_meta": {
                "baseline_metrics": {},
                "llm_call_count": 5,
            }
        }
        assert _baseline_llm_call_count(record) == 5

    def test_fallback_to_attempt_metadata(self) -> None:
        record: dict[str, Any] = {
            "patch_meta": {"baseline_metrics": {}},
            "attempt_metadata": {"llm_call_count": 7},
        }
        assert _baseline_llm_call_count(record) == 7

    def test_no_data_returns_zero(self) -> None:
        record: dict[str, Any] = {"patch_meta": {"baseline_metrics": {}}}
        assert _baseline_llm_call_count(record) == 0


# ── _terminal_failure_reason (lines 956-964) ────────────────────────────


class TestTerminalFailureReason:
    def test_overflow(self) -> None:
        record: dict[str, Any] = {
            "patch_meta": {"body": "exceeds the available context size"},
            "note": "",
        }
        assert _terminal_failure_reason(record) == "http_error:context_exceeded"

    def test_compatibility_reason(self) -> None:
        record: dict[str, Any] = {
            "patch_meta": {"compatibility_reason": "timeout"},
            "note": "forced_noop_patch",
        }
        reason = _terminal_failure_reason(record)
        assert "case_incompatible:timeout" in reason

    def test_baseline_error(self) -> None:
        record: dict[str, Any] = {
            "patch_meta": {"baseline_meta": {"error": "crash"}},
            "note": "",
        }
        reason = _terminal_failure_reason(record)
        assert "baseline_error:crash" in reason

    def test_error_status(self) -> None:
        record: dict[str, Any] = {
            "patch_meta": {"error": "http_error", "status": "500"},
            "note": "",
        }
        assert _terminal_failure_reason(record) == "http_error:500"

    def test_error_only(self) -> None:
        record: dict[str, Any] = {
            "patch_meta": {"error": "timeout"},
            "note": "",
        }
        assert _terminal_failure_reason(record) == "timeout"

    def test_forced_noop(self) -> None:
        record: dict[str, Any] = {
            "note": "forced_noop_patch",
        }
        assert _terminal_failure_reason(record) == "runner_forced_noop_after_no_diff"

    def test_note_fallback(self) -> None:
        record: dict[str, Any] = {"note": "custom_note"}
        assert _terminal_failure_reason(record) == "note:custom_note"

    def test_unknown(self) -> None:
        record: dict[str, Any] = {"note": ""}
        assert _terminal_failure_reason(record) == "unknown"


# ── _annotate_runner_note (lines 969-970) ────────────────────────────────


class TestAnnotateRunnerNote:
    def test_empty_note(self) -> None:
        assert _annotate_runner_note("reason", note="") == "reason"

    def test_success_note(self) -> None:
        assert _annotate_runner_note("reason", note="success") == "reason"

    def test_tests_still_failing_note(self) -> None:
        assert _annotate_runner_note("reason", note="tests_still_failing") == "reason"

    def test_custom_note(self) -> None:
        assert _annotate_runner_note("reason", note="custom") == "reason [runner_note=custom]"


# ── _null_sort_key (lines 413-415) ──────────────────────────────────────


class TestNullSortKey:
    def test_numeric(self) -> None:
        assert _null_sort_key(3.14) == (0, 3.14)

    def test_none(self) -> None:
        assert _null_sort_key(None) == (1, math.inf)

    def test_string(self) -> None:
        assert _null_sort_key("abc") == (1, math.inf)


# ── _parse_eval_log (lines 1098-1126) ───────────────────────────────────


class TestParseEvalLog:
    def test_with_runner_elapsed(self) -> None:
        text = textwrap.dedent("""\
        Progress: finished executed=100, skipped=0, total=100, elapsed=01:23:45
        ============================= JOB FEEDBACK =============================
        Job Wall-clock time: 02:00:00
        """)
        result = _parse_eval_log(text)
        assert result["runner_elapsed"] == "01:23:45"
        assert result["job wall-clock time"] == "02:00:00"

    def test_without_runner_elapsed(self) -> None:
        text = "no progress line here"
        result = _parse_eval_log(text)
        assert "runner_elapsed" not in result

    def test_empty_key_skipped(self) -> None:
        text = textwrap.dedent("""\
        ============================= JOB FEEDBACK =============================
        : value_for_empty_key
        key: value
        """)
        result = _parse_job_feedback_block(text)
        assert "" not in result
        assert result["key"] == "value"


# ── _print_input_errors (lines 1553-1555) ───────────────────────────────


class TestReportPrintInputErrors:
    def test_no_errors(self, capsys: pytest.CaptureFixture[str]) -> None:
        _print_input_errors(
            {
                "files_unreadable": 0,
                "json_lines_invalid": 0,
                "records_skipped": 0,
            }
        )
        assert capsys.readouterr().out == ""

    def test_with_errors(self, capsys: pytest.CaptureFixture[str]) -> None:
        _print_input_errors(
            {
                "files_unreadable": 2,
                "json_lines_invalid": 1,
                "records_skipped": 3,
            }
        )
        out = capsys.readouterr().out
        assert "Input diagnostics" in out
        assert "files_unreadable=2" in out


# ── Formatting helpers (lines 1566-1606) ────────────────────────────────


class TestReportFormatHelpers:
    def test_as_float_int(self) -> None:
        assert _as_float(3) == 3.0

    def test_as_float_none(self) -> None:
        assert _as_float(None) is None

    def test_as_float_string(self) -> None:
        assert _as_float("abc") is None

    def test_mean_empty(self) -> None:
        assert _mean([]) is None

    def test_mean_values(self) -> None:
        assert _mean([2.0, 4.0]) == 3.0

    def test_median_empty(self) -> None:
        assert _median([]) is None

    def test_median_odd(self) -> None:
        assert _median([1.0, 3.0, 5.0]) == 3.0

    def test_median_even(self) -> None:
        assert _median([1.0, 3.0]) == 2.0

    def test_fmt_float_non_numeric(self) -> None:
        assert _fmt_float("abc") == "n/a"

    def test_fmt_float_numeric(self) -> None:
        assert _fmt_float(3.14) == "3.1400"

    def test_fmt_pct_non_numeric(self) -> None:
        assert _fmt_pct("abc") == "n/a"

    def test_fmt_pct_numeric(self) -> None:
        assert _fmt_pct(0.5) == "50.00%"

    def test_fmt_pct_signed_non_numeric(self) -> None:
        assert _fmt_pct_signed("abc") == "n/a"

    def test_fmt_pct_signed_positive(self) -> None:
        assert _fmt_pct_signed(0.1) == "+10.00%"

    def test_fmt_signed_pp_non_numeric(self) -> None:
        assert _fmt_signed_pp("abc") == "n/a"

    def test_fmt_signed_pp_numeric(self) -> None:
        assert _fmt_signed_pp(0.1) == "+10.00 pp"


# ── _ratio (line 974-976) ───────────────────────────────────────────────


class TestReportRatio:
    def test_zero_denom(self) -> None:
        assert _ratio(5, 0) is None

    def test_normal(self) -> None:
        assert _ratio(3, 6) == 0.5


# ── _conclusion_line (lines 1483-1484) ──────────────────────────────────


class TestConclusionLine:
    def test_unavailable(self) -> None:
        result = _conclusion_line(
            condition_a="A",
            condition_b="B",
            effect={"available": False, "error": "no pairs"},
        )
        assert "unavailable" in result

    def test_non_numeric_uplift(self) -> None:
        result = _conclusion_line(
            condition_a="A",
            condition_b="B",
            effect={"available": True, "uplift_pp": None},
        )
        assert "did not yield" in result

    def test_worse(self) -> None:
        result = _conclusion_line(
            condition_a="A",
            condition_b="B",
            effect={"available": True, "uplift_pp": -5.0, "mcnemar": {"p_value": 0.01}},
        )
        assert "worse" in result

    def test_equal(self) -> None:
        result = _conclusion_line(
            condition_a="A",
            condition_b="B",
            effect={"available": True, "uplift_pp": 0.0, "mcnemar": {"p_value": 1.0}},
        )
        assert "equal" in result


# ── _print_effect_line (lines 1507-1510) ────────────────────────────────


class TestPrintEffectLine:
    def test_unavailable(self, capsys: pytest.CaptureFixture[str]) -> None:
        _print_effect_line("Label", "A", "B", {"available": False})
        out = capsys.readouterr().out
        assert "unavailable" in out

    def test_non_numeric_uplift(self, capsys: pytest.CaptureFixture[str]) -> None:
        _print_effect_line("Label", "A", "B", {"available": True, "uplift_pp": "bad"})
        out = capsys.readouterr().out
        assert "non-numeric uplift" in out

    def test_equal_uplift(self, capsys: pytest.CaptureFixture[str]) -> None:
        _print_effect_line(
            "Label",
            "A",
            "B",
            {
                "available": True,
                "a_rate": 0.5,
                "b_rate": 0.5,
                "uplift_pp": 0.0,
                "mcnemar": {},
                "bootstrap_ci": {},
                "cohens_h": {},
                "agreement": {},
            },
        )
        out = capsys.readouterr().out
        assert "equal" in out


# ── _print_overhead_analysis (lines 1437-1472) ──────────────────────────


class TestPrintOverheadAnalysis:
    def test_empty_per_condition(self, capsys: pytest.CaptureFixture[str]) -> None:
        _print_overhead_analysis({}, "A", "B")
        out = capsys.readouterr().out
        assert "unavailable" in out

    def test_with_timing_and_tokens(self, capsys: pytest.CaptureFixture[str]) -> None:
        overhead = {
            "per_condition": {
                "A": {
                    "cases": 5,
                    "llm_pct_of_runtime": 60.0,
                    "overhead_pct_of_runtime": 10.0,
                    "llmdebug_pct_of_runtime": 5.0,
                    "timing_mean": {"stage_b_llm_sec": 2.0},
                    "token_mean": {"stage_b_tokens_total": 500},
                },
            }
        }
        _print_overhead_analysis(overhead, "A", "B")
        out = capsys.readouterr().out
        assert "[A]" in out
        assert "LLM % of runtime" in out
        assert "llmdebug % of runtime" in out
        assert "Timing breakdown" in out
        assert "Token breakdown" in out

    def test_no_timing_no_tokens(self, capsys: pytest.CaptureFixture[str]) -> None:
        overhead = {
            "per_condition": {
                "A": {
                    "cases": 5,
                    "llm_pct_of_runtime": None,
                    "overhead_pct_of_runtime": None,
                    "llmdebug_pct_of_runtime": None,
                    "timing_mean": {},
                    "token_mean": {},
                },
            }
        }
        _print_overhead_analysis(overhead, "A", "B")
        out = capsys.readouterr().out
        assert "[A]" in out


# ── _append_covered_pct (lines 1395-1396) ───────────────────────────────


class TestAppendCoveredPct:
    def test_missing_covered_cases(self) -> None:
        parts: list[str] = []
        _append_covered_pct(parts, {}, 10, "field", "label")
        assert parts == []

    def test_zero_covered_cases(self) -> None:
        parts: list[str] = []
        _append_covered_pct(parts, {"field_covered_cases": 0}, 10, "field", "label")
        assert parts == []

    def test_non_numeric_value(self) -> None:
        parts: list[str] = []
        _append_covered_pct(
            parts,
            {"field": "bad", "field_covered_cases": 5},
            10,
            "field",
            "label",
        )
        assert parts == []

    def test_valid(self) -> None:
        parts: list[str] = []
        _append_covered_pct(
            parts,
            {"field": 0.8, "field_covered_cases": 5},
            10,
            "field",
            "label",
        )
        assert len(parts) == 1
        assert "label=" in parts[0]
        assert "coverage 5/10" in parts[0]


# ── _baseline_diagnostic_lines (lines 1279-1280) ────────────────────────


class TestBaselineDiagnosticLines:
    def test_non_dict_stats_skipped(self) -> None:
        lines = _baseline_diagnostic_lines(
            condition_summary={"ldb": "not_a_dict"},
        )
        assert lines == []

    def test_non_external_condition_skipped(self) -> None:
        lines = _baseline_diagnostic_lines(
            condition_summary={"base_repair": {"cases": 5}},
        )
        assert lines == []


# ── _baseline_terminal_usage_state (lines 920-924) ──────────────────────


class TestBaselineTerminalUsageState:
    def test_available(self) -> None:
        record: dict[str, Any] = {
            "patch_meta": {"baseline_metrics": {"usage_available": True}},
            "attempt_metadata": {"llm_call_count": 1},
        }
        assert _baseline_terminal_usage_state(record) == "available"

    def test_unavailable(self) -> None:
        record: dict[str, Any] = {
            "patch_meta": {"baseline_metrics": {"llm_call_count": 1}},
            "attempt_metadata": {"llm_call_count": 1},
        }
        assert _baseline_terminal_usage_state(record) == "unavailable"

    def test_not_applicable(self) -> None:
        record: dict[str, Any] = {
            "patch_meta": {"baseline_metrics": {}},
        }
        assert _baseline_terminal_usage_state(record) == "not_applicable"
