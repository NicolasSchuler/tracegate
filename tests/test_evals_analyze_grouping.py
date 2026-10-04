"""Tests for result grouping, summarization, and gate-count logic in analyze_results."""

from __future__ import annotations

import pytest

from evals.analyze_results import (
    _derive_case_condition_rows,
    _load_and_group_records,
    _retry_gate_counts,
    _runtime_used,
    _summarize_bucket,
    _test_touch_attempt_metrics,
)
from evals.analyze_results import (
    main as analyze_main,
)
from llmdebug import _json as json


def _record(
    *,
    run_id: str,
    case_id: str,
    condition: str,
    attempt: int,
    success: bool,
    category: str,
    difficulty: str,
    snapshot_dependency: str,
    **overrides,
):
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
            "started_at_utc": "2026-02-10T00:00:00Z",
            "deterministic": False,
            "seed": None,
            "python_version": "3.12.0",
            "platform": "test",
            "llmdebug_version": "2.1.0",
            "patcher_name": "openai",
            "patcher_config_fingerprint": "abc123",
            "run_settings_fingerprint": "runfp123",
        },
        "attempt_metadata": {
            "attempt": attempt,
            "max_attempts": 3,
            "attempt_ordinal_label": f"{attempt}/3",
            "llm_call_count": 1,
            "tokens_prompt": 0,
            "tokens_completion": 0,
            "tokens_total": 0,
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
            "base_url": "http://127.0.0.1:1234/v1",
            "model": "dummy",
            "temperature": 0.0,
            "max_tokens": 2048,
            "response_mode": "edits",
            "api_compat_seed_requested": None,
            "api_compat_seed_acknowledged": False,
            "api_compat_seed_response": None,
        },
    }
    record.update(overrides)
    return record


def test_analyze_grouping_and_uplift(tmp_path) -> None:
    records = [
        _record(
            run_id="run_a",
            case_id="case_one",
            condition="base_repair",
            category="hidden_state",
            difficulty="easy",
            snapshot_dependency="required",
            attempt=1,
            success=False,
            before_duration_sec=1.0,
            after_duration_sec=0.5,
            cumulative_runtime_sec=1.6,
            patch_meta={"usage": {"total_tokens": 10}},
        ),
        _record(
            run_id="run_a",
            case_id="case_one",
            condition="base_repair",
            category="hidden_state",
            difficulty="easy",
            snapshot_dependency="required",
            attempt=2,
            success=True,
            before_duration_sec=0.5,
            after_duration_sec=0.2,
            cumulative_runtime_sec=2.1,
            patch_meta={"usage": {"total_tokens": 20}},
        ),
        _record(
            run_id="run_a",
            case_id="case_one",
            condition="cot_repair",
            category="hidden_state",
            difficulty="easy",
            snapshot_dependency="required",
            attempt=1,
            success=True,
            before_duration_sec=0.9,
            after_duration_sec=0.2,
            cumulative_runtime_sec=1.4,
            patch_meta={"usage": {"total_tokens": 15}},
        ),
        _record(
            run_id="run_a",
            case_id="case_one",
            condition="layered_adaptive_gated",
            category="hidden_state",
            difficulty="easy",
            snapshot_dependency="required",
            attempt=1,
            success=True,
            before_duration_sec=0.8,
            after_duration_sec=0.2,
            cumulative_runtime_sec=1.3,
            patch_meta={"usage": {"total_tokens": 14}},
        ),
        _record(
            run_id="run_a",
            case_id="case_two",
            condition="base_repair",
            category="parse_value",
            difficulty="medium",
            snapshot_dependency="helpful",
            attempt=1,
            success=False,
            before_duration_sec=1.1,
            cumulative_runtime_sec=1.3,
        ),
        _record(
            run_id="run_a",
            case_id="case_two",
            condition="cot_repair",
            category="parse_value",
            difficulty="medium",
            snapshot_dependency="helpful",
            attempt=1,
            success=False,
            before_duration_sec=1.0,
            cumulative_runtime_sec=1.5,
            patch_meta={"usage": {"total_tokens": 5}},
        ),
        _record(
            run_id="run_b",
            case_id="case_one",
            condition="base_repair",
            category="hidden_state",
            difficulty="easy",
            snapshot_dependency="required",
            attempt=1,
            success=False,
            before_duration_sec=1.3,
            cumulative_runtime_sec=1.8,
        ),
        _record(
            run_id="run_b",
            case_id="case_one",
            condition="cot_repair",
            category="hidden_state",
            difficulty="easy",
            snapshot_dependency="required",
            attempt=1,
            success=True,
            before_duration_sec=1.2,
            after_duration_sec=0.1,
            cumulative_runtime_sec=1.6,
            patch_meta={"usage": {"total_tokens": 12}},
        ),
    ]
    jsonl = tmp_path / "results.jsonl"
    jsonl.write_text("\n".join(json.dumps(record) for record in records) + "\n", encoding="utf-8")

    grouped, input_errors = _load_and_group_records([str(jsonl)])
    assert len(grouped) == 7
    assert input_errors == {
        "files_unreadable": 0,
        "json_lines_invalid": 0,
        "records_skipped": 0,
        "records_schema_invalid": 0,
        "infra_event_rows_seen": 0,
        "infra_event_rows_skipped": 0,
        "duplicate_attempt_rows_seen": 0,
        "duplicate_attempt_rows_resolved": 0,
    }

    rows = _derive_case_condition_rows(grouped)
    assert len(rows) == 7
    row_index = {(row["run_id"], row["case_id"], row["condition"]): row for row in rows}
    assert row_index[("run_a", "case_one", "base_repair")]["runtime_sec"] == pytest.approx(2.1)

    overall = _summarize_bucket(rows)
    assert overall["base_repair"]["cases"] == 3
    assert overall["base_repair"]["successes"] == 1
    assert overall["base_repair"]["success_rate"] == pytest.approx(1 / 3)
    assert overall["base_repair"]["median_attempts_to_success"] == pytest.approx(2.0)
    assert overall["base_repair"]["mean_attempts_budget"] == pytest.approx(3.0)
    assert overall["base_repair"]["solve_on_attempt_distribution"] == {"2": 1}
    assert overall["cot_repair"]["cases"] == 3
    assert overall["cot_repair"]["successes"] == 2
    assert overall["cot_repair"]["success_rate"] == pytest.approx(2 / 3)
    assert overall["uplift_success_rate"] is None  # no layered_adaptive_llm data
    # STA-03: matched on case_one, the only case gated ran. base_repair solved it in one
    # of its two runs (0.5), gated in its one run (1.0). The unmatched rate difference
    # (1.0 - 1/3) used to be reported.
    assert overall["uplift_layered_gated_vs_base"] == pytest.approx(0.5)
    assert overall["layered_adaptive_gated"]["cases_missing"] == 1
    assert "mean_attempts_budget" in overall["base_repair"]
    assert "mean_attempt_ratio_when_successful" in overall["base_repair"]
    assert "solve_on_attempt_distribution" in overall["base_repair"]
    assert "mean_tokens_prompt" in overall["base_repair"]
    assert "mean_tokens_completion" in overall["base_repair"]
    assert "mean_tokens_total" in overall["base_repair"]
    assert "mean_tokens_reasoning" in overall["base_repair"]
    assert "mean_tokens_per_attempt" in overall["base_repair"]
    assert "mean_reasoning_tokens_per_attempt" in overall["base_repair"]
    assert "mean_llm_calls_per_case" in overall["base_repair"]
    assert overall["layered_adaptive_gated"]["cases"] == 1
    assert overall["layered_adaptive_gated"]["success_rate"] == pytest.approx(1.0)
    assert "mean_prompt_chars_stage_b" in overall
    assert "mean_retry_history_chars" in overall
    assert "mean_retry_history_turns" in overall
    assert "retry_history_budget_applied_rate" in overall
    assert "retry_history_truncation_rate" in overall
    assert "stage_a_request_success_rate" in overall
    assert "uplift_staged_vs_monolithic" in overall
    assert "retry_gate_block_rate" in overall
    assert "pairwise_uplifts" in overall
    assert isinstance(overall["pairwise_uplifts"], list)
    assert overall["uplift_layered_gated_vs_base"] == pytest.approx(0.5)
    assert overall["uplift_layered_llm_vs_base"] is None  # no layered_adaptive_llm data

    by_category = {"hidden_state": [], "parse_value": []}
    for row in rows:
        by_category[row["category"]].append(row)
    hidden_state_summary = _summarize_bucket(by_category["hidden_state"])
    assert hidden_state_summary["cot_repair"]["success_rate"] == pytest.approx(1.0)
    assert hidden_state_summary["base_repair"]["success_rate"] == pytest.approx(0.5)
    assert hidden_state_summary["uplift_success_rate"] is None  # no layered_adaptive_llm

    by_snapshot_dependency = {"required": [], "helpful": []}
    for row in rows:
        by_snapshot_dependency[row["snapshot_dependency"]].append(row)
    required_summary = _summarize_bucket(by_snapshot_dependency["required"])
    assert required_summary["base_repair"]["cases"] == 2
    assert required_summary["cot_repair"]["cases"] == 2
    assert required_summary["uplift_success_rate"] is None  # no layered_adaptive_llm


def test_analyze_grouping_accepts_current_v6_xxhash_fields(tmp_path) -> None:
    record = _record(
        run_id="run_v6",
        case_id="case_v6",
        condition="base_repair",
        category="hidden_state",
        difficulty="easy",
        snapshot_dependency="required",
        attempt=1,
        success=True,
    )
    record["schema_version"] = 6
    record["attempt_metadata"].pop("prompt_sha256")
    record["attempt_metadata"].pop("evidence_sha256")
    record["attempt_metadata"]["prompt_xxh3_128"] = "p"
    record["attempt_metadata"]["evidence_xxh3_128"] = "e"

    jsonl = tmp_path / "results_v6.jsonl"
    jsonl.write_text(json.dumps(record) + "\n", encoding="utf-8")

    grouped, input_errors = _load_and_group_records([str(jsonl)])
    assert len(grouped) == 1
    assert input_errors["records_skipped"] == 0


def test_patch_attempt_metrics_accept_current_xxhash_patch_digest() -> None:
    patch_attempts, test_touch_attempts = _test_touch_attempt_metrics(
        [{"patch_meta": {"patch_diff_xxh3_128": "abc123"}}]
    )

    assert patch_attempts == 1
    assert test_touch_attempts == 0


def test_analyze_grouping_tolerates_malformed_jsonl(tmp_path, capsys) -> None:
    valid_a = _record(
        run_id="run",
        case_id="case",
        condition="base_repair",
        category="hidden_state",
        difficulty="easy",
        snapshot_dependency="required",
        attempt=1,
        success=False,
    )
    valid_b = _record(
        run_id="run",
        case_id="case",
        condition="cot_repair",
        category="hidden_state",
        difficulty="easy",
        snapshot_dependency="required",
        attempt=1,
        success=True,
    )
    jsonl = tmp_path / "results_mixed.jsonl"
    jsonl.write_text(
        "\n".join(
            [
                json.dumps(valid_a),
                "this-is-not-json",
                json.dumps(valid_b),
                '{"run_id":"run","condition":"cot_repair","attempt":2}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    grouped, input_errors = _load_and_group_records([str(jsonl)])
    assert len(grouped) == 2
    assert input_errors["json_lines_invalid"] == 1
    assert input_errors["records_skipped"] == 1

    exit_code = analyze_main([str(jsonl)])
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Input diagnostics:" in output
    assert "json_lines_invalid=1" in output
    assert "records_skipped=1" in output


def test_analyze_grouping_dedups_duplicate_attempt_rows_last_write_wins(tmp_path, capsys) -> None:
    original = _record(
        run_id="run_dup",
        case_id="case_dup",
        condition="base_repair",
        category="hidden_state",
        difficulty="easy",
        snapshot_dependency="required",
        attempt=1,
        success=False,
        cumulative_runtime_sec=1.0,
    )
    original["attempt_metadata"]["llm_call_count"] = 1
    original["attempt_metadata"]["tokens_total"] = 10
    original["attempt_metadata"]["tokens_prompt"] = 9
    original["attempt_metadata"]["tokens_completion"] = 1

    rewritten = _record(
        run_id="run_dup",
        case_id="case_dup",
        condition="base_repair",
        category="hidden_state",
        difficulty="easy",
        snapshot_dependency="required",
        attempt=1,
        success=True,
        cumulative_runtime_sec=2.0,
    )
    rewritten["attempt_metadata"]["llm_call_count"] = 3
    rewritten["attempt_metadata"]["tokens_total"] = 30
    rewritten["attempt_metadata"]["tokens_prompt"] = 25
    rewritten["attempt_metadata"]["tokens_completion"] = 5

    jsonl = tmp_path / "results_duplicates.jsonl"
    jsonl.write_text(
        "\n".join([json.dumps(original), json.dumps(rewritten)]) + "\n",
        encoding="utf-8",
    )

    grouped, input_errors = _load_and_group_records([str(jsonl)])
    assert input_errors["duplicate_attempt_rows_seen"] == 1
    assert input_errors["duplicate_attempt_rows_resolved"] == 1

    rows = _derive_case_condition_rows(grouped)
    assert len(rows) == 1
    row = rows[0]
    assert row["success"] is True
    assert row["attempts_executed"] == 1
    assert row["llm_calls_total"] == 3
    assert row["token_usage_total"] == pytest.approx(30)
    assert row["runtime_sec"] == pytest.approx(2.0)

    exit_code = analyze_main([str(jsonl)])
    assert exit_code == 0
    output = capsys.readouterr().out
    assert "duplicate_attempt_rows_seen=1" in output
    assert "duplicate_attempt_rows_resolved=1" in output


def test_retry_gate_counts_use_evaluated_flag() -> None:
    attempts, blocks = _retry_gate_counts(
        [
            {"patch_meta": {"retry_gate_evaluated": True, "retry_gate_blocked": False}},
            {"patch_meta": {"retry_gate_evaluated": True, "retry_gate_blocked": True}},
            {"patch_meta": {"retry_gate_blocked": True}},
            {"patch_meta": {"retry_gate_blocked": None}},
            {"patch_meta": {}},
            {},
        ]
    )
    assert attempts == 3
    assert blocks == 2


def test_runtime_used_prefers_attempt_wall_clock_when_all_attempts_have_it() -> None:
    records = [
        _record(
            run_id="run_a",
            case_id="case_a",
            condition="base_repair",
            category="hidden_state",
            difficulty="easy",
            snapshot_dependency="required",
            attempt=1,
            success=False,
            cumulative_runtime_sec=50.0,  # intentionally inconsistent
            attempt_wall_clock_sec=1.2,
        ),
        _record(
            run_id="run_a",
            case_id="case_a",
            condition="base_repair",
            category="hidden_state",
            difficulty="easy",
            snapshot_dependency="required",
            attempt=2,
            success=True,
            cumulative_runtime_sec=99.0,  # intentionally inconsistent
            attempt_wall_clock_sec=0.8,
        ),
    ]
    assert _runtime_used(records) == pytest.approx(2.0)


def test_runtime_used_falls_back_to_cumulative_when_attempt_wall_clock_is_partial() -> None:
    records = [
        _record(
            run_id="run_a",
            case_id="case_a",
            condition="base_repair",
            category="hidden_state",
            difficulty="easy",
            snapshot_dependency="required",
            attempt=1,
            success=False,
            cumulative_runtime_sec=1.5,
            attempt_wall_clock_sec=1.0,
        ),
        _record(
            run_id="run_a",
            case_id="case_a",
            condition="base_repair",
            category="hidden_state",
            difficulty="easy",
            snapshot_dependency="required",
            attempt=2,
            success=True,
            cumulative_runtime_sec=2.2,
        ),
    ]
    assert _runtime_used(records) == pytest.approx(2.2)


def test_analyze_grouping_tracks_test_touch_patch_rates() -> None:
    grouped = {
        ("run_x", "case_x", "base_repair"): [
            _record(
                run_id="run_x",
                case_id="case_x",
                condition="base_repair",
                category="logic_error",
                difficulty="easy",
                snapshot_dependency="required",
                attempt=1,
                success=False,
                patch_meta={"patch_file_count": 1, "patch_touches_tests": False},
            ),
            _record(
                run_id="run_x",
                case_id="case_x",
                condition="base_repair",
                category="logic_error",
                difficulty="easy",
                snapshot_dependency="required",
                attempt=2,
                success=True,
                patch_meta={"patch_file_count": 2, "patch_touches_tests": True},
            ),
        ]
    }
    rows = _derive_case_condition_rows(grouped)
    assert len(rows) == 1
    row = rows[0]
    assert row["patch_attempts"] == 2
    assert row["test_touch_attempts"] == 1
    assert row["test_touch_attempt_rate"] == pytest.approx(0.5)

    summary = _summarize_bucket(rows)
    tb = summary["base_repair"]
    assert tb["patch_attempts"] == 2
    assert tb["test_touch_attempts"] == 1
    assert tb["test_touch_attempt_rate"] == pytest.approx(0.5)
    assert summary["test_touch_attempt_rate"] == pytest.approx(0.5)


def test_analyze_grouping_strict_input_rejects_invalid_attempt_wall_clock_type(
    tmp_path,
) -> None:
    invalid = _record(
        run_id="run_bad",
        case_id="case_bad",
        condition="base_repair",
        category="hidden_state",
        difficulty="easy",
        snapshot_dependency="required",
        attempt=1,
        success=False,
        attempt_wall_clock_sec="slow",
    )
    jsonl = tmp_path / "results_bad_attempt_wall_clock.jsonl"
    jsonl.write_text(json.dumps(invalid) + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match="attempt_wall_clock_sec"):
        _load_and_group_records([str(jsonl)], strict_input=True)


def test_analyze_grouping_strict_input_fails_on_malformed_jsonl(tmp_path, capsys) -> None:
    valid = _record(
        run_id="run",
        case_id="case",
        condition="base_repair",
        category="hidden_state",
        difficulty="easy",
        snapshot_dependency="required",
        attempt=1,
        success=False,
    )
    jsonl = tmp_path / "results_bad.jsonl"
    jsonl.write_text(
        json.dumps(valid) + "\nnot-json\n",
        encoding="utf-8",
    )

    exit_code = analyze_main(["--strict-input", str(jsonl)])
    assert exit_code == 2
    output = capsys.readouterr().out
    assert "Input error:" in output


def test_summarize_bucket_uses_executed_attempts_for_failed_cases() -> None:
    summary = _summarize_bucket(
        [
            {
                "condition": "intervenor",
                "success": False,
                "attempts_executed": 1,
                "attempts_budget": 5,
                "attempts_to_success": None,
                "token_usage": None,
                "runtime_sec": None,
                "llm_calls_total": None,
                "patch_attempts": 0,
                "test_touch_attempts": 0,
            }
        ]
    )

    assert summary["intervenor"]["mean_attempts_when_failed"] == pytest.approx(1.0)


def test_derive_case_condition_rows_uses_terminal_final_fail_count() -> None:
    grouped = {
        ("run_a", "tracecoder_case", "tracecoder"): [
            _record(
                run_id="run_a",
                case_id="tracecoder_case",
                condition="tracecoder",
                category="hidden_state",
                difficulty="easy",
                snapshot_dependency="required",
                attempt=1,
                success=False,
                patch_meta={
                    "baseline_metrics": {
                        "test_feedback_final_fail_count": 3,
                    }
                },
            ),
            _record(
                run_id="run_a",
                case_id="tracecoder_case",
                condition="tracecoder",
                category="hidden_state",
                difficulty="easy",
                snapshot_dependency="required",
                attempt=2,
                success=True,
                patch_meta={
                    "baseline_metrics": {
                        "test_feedback_final_fail_count": 0,
                    }
                },
            ),
        ]
    }

    rows = _derive_case_condition_rows(grouped)

    assert len(rows) == 1
    assert rows[0]["baseline_test_feedback_final_fail_count"] == 0


def test_analyze_grouping_rejects_missing_schema_fields(tmp_path, capsys) -> None:
    jsonl = tmp_path / "results_missing_schema.jsonl"
    jsonl.write_text(
        '{"run_id":"run","case_id":"case","condition":"base_repair","attempt":1,"success":false}\n',
        encoding="utf-8",
    )

    # STA-25: without --strict-input the invalid record is skipped and counted.
    exit_code = analyze_main([str(jsonl)])
    assert exit_code == 1
    output = capsys.readouterr().out
    assert "No records found." in output

    exit_code = analyze_main([str(jsonl), "--strict-input"])
    assert exit_code == 2
    assert "missing required field" in capsys.readouterr().out


def test_load_and_group_records_skips_infra_event_rows(tmp_path) -> None:
    jsonl = tmp_path / "results_with_infra.jsonl"
    infra_record = _record(
        run_id="run_a",
        case_id="case_one",
        condition="base_repair",
        category="hidden_state",
        difficulty="easy",
        snapshot_dependency="required",
        attempt=1,
        success=False,
        record_type="infra_event",
        note="infra_transport_failure",
    )
    attempt_record = _record(
        run_id="run_a",
        case_id="case_one",
        condition="base_repair",
        category="hidden_state",
        difficulty="easy",
        snapshot_dependency="required",
        attempt=1,
        success=True,
    )
    jsonl.write_text(
        "\n".join(json.dumps(record) for record in [infra_record, attempt_record]) + "\n",
        encoding="utf-8",
    )

    grouped, input_errors = _load_and_group_records([str(jsonl)])

    assert len(grouped) == 1
    assert len(grouped[("run_a", "case_one", "base_repair")]) == 1
    assert input_errors["infra_event_rows_seen"] == 1
    assert input_errors["infra_event_rows_skipped"] == 1
    assert input_errors["duplicate_attempt_rows_seen"] == 0


def test_sta25_schema_invalid_record_is_skipped_without_strict_input(tmp_path) -> None:
    valid = _record(
        run_id="run_a",
        case_id="case_one",
        condition="base_repair",
        category="hidden_state",
        difficulty="easy",
        snapshot_dependency="required",
        attempt=1,
        success=True,
    )
    invalid = {"run_id": "run_a", "case_id": "case_two", "condition": "base_repair", "attempt": 1}
    jsonl = tmp_path / "mixed.jsonl"
    jsonl.write_text(json.dumps(valid) + "\n" + json.dumps(invalid) + "\n", encoding="utf-8")

    grouped, input_errors = _load_and_group_records([str(jsonl)])
    assert list(grouped) == [("run_a", "case_one", "base_repair")]
    assert input_errors["records_schema_invalid"] == 1
    with pytest.raises(ValueError, match="missing required field"):
        _load_and_group_records([str(jsonl)], strict_input=True)
