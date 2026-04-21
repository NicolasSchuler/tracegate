"""Tests for evals.budget_risk_report."""

from __future__ import annotations

from pathlib import Path

import pytest

from evals import budget_risk_report as report


def test_summarize_rows_reports_native_cap_risk_signals() -> None:
    rows = [
        {
            "case_id": "case-a",
            "condition": "cot_repair",
            "success": False,
            "prompt_budget_overflow_chars": 0,
            "context_compaction_auto_overflow_triggered": True,
            "patch_meta": {
                "finish_reason": "length",
                "plan_patch_meta": {"finish_reason": "length"},
            },
        },
        {
            "case_id": "case-a",
            "condition": "cot_repair",
            "success": False,
            "prompt_budget_overflow_chars": 0,
            "context_compaction_auto_overflow_triggered": False,
            "patch_meta": {
                "finish_reason": "length",
                "plan_patch_meta": {"finish_reason": "length"},
            },
        },
        {
            "case_id": "case-b",
            "condition": "cot_repair",
            "success": False,
            "prompt_budget_overflow_chars": 5,
            "context_compaction_auto_overflow_triggered": False,
            "patch_meta": {
                "finish_reason": "stop",
                "plan_patch_meta": {"finish_reason": "stop"},
            },
        },
    ]

    summary = report.summarize_rows(rows, source="synthetic.jsonl")

    assert summary["attempt_finish_reason_distribution"] == {"length": 2, "stop": 1}
    assert summary["planner_finish_reason_distribution"] == {"length": 2, "stop": 1}
    assert summary["prompt_budget_overflow_rows"] == 1
    assert summary["context_compaction_auto_overflow_triggered_rows"] == 1
    assert summary["lower_bound_cases_harmed_by_final_caps"] == 1
    assert summary["lower_bound_cases_harmed_by_planner_caps"] == 1


def test_summarize_rows_ignores_baseline_rows_for_native_unknown_buckets() -> None:
    rows = [
        {
            "case_id": "native-case",
            "condition": "base_repair",
            "success": False,
            "patch_meta": {
                "finish_reason": "stop",
            },
        },
        {
            "case_id": "lambda-case",
            "condition": "lambda_rlm",
            "success": False,
            "patch_meta": {
                "baseline_meta": {
                    "attempts": [
                        {
                            "attempt": 1,
                            "llm_call_count": 1,
                            "stop_reason": "completed",
                        }
                    ]
                }
            },
        },
    ]

    summary = report.summarize_rows(rows, source="mixed.jsonl")

    assert summary["attempt_finish_reason_distribution"] == {"stop": 1}
    assert summary["attempt_finish_reason_unknown_rows"] == 0
    assert summary["planner_finish_reason_unknown_rows"] == 0
    assert not any(
        "top-level finish_reason" in item for item in summary["unknown_due_to_missing_telemetry"]
    )
    assert not any(
        "planner finish_reason" in item for item in summary["unknown_due_to_missing_telemetry"]
    )


def test_summarize_rows_does_not_count_planner_unknown_for_base_repair_without_planner_stage() -> (
    None
):
    rows = [
        {
            "case_id": "base-case",
            "condition": "base_repair",
            "success": False,
            "patch_meta": {
                "finish_reason": "stop",
            },
        }
    ]

    summary = report.summarize_rows(rows, source="base-only.jsonl")

    assert summary["planner_finish_reason_distribution"] == {}
    assert summary["planner_finish_reason_unknown_rows"] == 0


def test_summarize_rows_counts_planner_unknown_when_retry_plan_then_patch_is_expected() -> None:
    rows = [
        {
            "case_id": "base-retry-plan",
            "condition": "base_repair",
            "success": False,
            "patch_meta": {
                "finish_reason": "stop",
                "prompt_retry_plan_then_patch": True,
            },
        }
    ]

    summary = report.summarize_rows(rows, source="retry-plan.jsonl")

    assert summary["planner_finish_reason_distribution"] == {}
    assert summary["planner_finish_reason_unknown_rows"] == 1
    assert any(
        "planner finish_reason" in item for item in summary["unknown_due_to_missing_telemetry"]
    )


def test_summarize_rows_reports_unknown_bucket_for_missing_intervenor_telemetry() -> None:
    rows = [
        {
            "case_id": "case-i",
            "condition": "intervenor",
            "success": False,
            "patch_meta": {
                "baseline_meta": {
                    "attempts": [
                        {"attempt": 1, "teacher_finish_reason": "stop"},
                        {"attempt": 2},
                    ]
                }
            },
        }
    ]

    summary = report.summarize_rows(rows, source="intervenor.jsonl")

    assert summary["teacher_finish_reason_distribution"] == {"stop": 1}
    assert summary["teacher_finish_reason_unknown_calls"] == 1
    assert summary["learner_finish_reason_unknown_calls"] == 2
    assert any(
        "teacher calls are missing finish_reason telemetry" in item
        for item in summary["unknown_due_to_missing_telemetry"]
    )


def test_summarize_rows_does_not_count_stage_unknowns_for_non_intervenor_baselines() -> None:
    rows = [
        {
            "case_id": "rlm-case",
            "condition": "rlm",
            "success": False,
            "patch_meta": {
                "baseline_meta": {
                    "attempts": [
                        {
                            "attempt": 1,
                            "llm_call_count": 1,
                            "stop_reason": "final_answer",
                        }
                    ]
                }
            },
        }
    ]

    summary = report.summarize_rows(rows, source="rlm.jsonl")

    assert summary["teacher_finish_reason_unknown_calls"] == 0
    assert summary["learner_finish_reason_unknown_calls"] == 0
    assert not any(
        "teacher calls are missing finish_reason telemetry" in item
        for item in summary["unknown_due_to_missing_telemetry"]
    )
    assert not any(
        "learner calls are missing finish_reason telemetry" in item
        for item in summary["unknown_due_to_missing_telemetry"]
    )


def test_summarize_rows_reports_lambda_and_rlm_baseline_telemetry() -> None:
    rows = [
        {
            "case_id": "lambda-case",
            "condition": "lambda_rlm",
            "success": False,
            "patch_meta": {
                "baseline_meta": {
                    "attempts": [
                        {
                            "attempt": 1,
                            "llm_call_count": 3,
                            "stop_reason": "completed",
                            "call_finish_reason_counts": {"length": 1, "stop": 2},
                            "call_length_finish_count": 1,
                            "compat_recovery_count": 1,
                            "planned_decomposition": True,
                            "executed_decomposition": True,
                            "task_detection_call_count": 1,
                            "repl_llm_query_call_count": 2,
                        }
                    ]
                }
            },
        },
        {
            "case_id": "rlm-case",
            "condition": "rlm",
            "success": False,
            "patch_meta": {
                "baseline_meta": {
                    "attempts": [
                        {
                            "attempt": 1,
                            "llm_call_count": 2,
                            "stop_reason": "iteration_budget_exhausted",
                            "guard_name": "iteration_budget_exhausted",
                            "stop_due_to_guard": True,
                            "candidate_status": "workspace_unchanged",
                            "outcome_classification": "failure_search_exhausted",
                            "call_finish_reason_counts": {"length": 1, "stop": 1},
                            "call_length_finish_count": 1,
                            "compat_recovery_count": 0,
                        },
                        {
                            "attempt": 2,
                            "llm_call_count": 1,
                            "stop_reason": "final_answer",
                            "candidate_status": "passed",
                            "candidate_source": "final_answer_salvage",
                            "outcome_classification": "success_final_answer_salvage",
                            "call_finish_reason_counts": {"stop": 1},
                            "call_length_finish_count": 0,
                            "compat_recovery_count": 1,
                        },
                    ]
                }
            },
        },
    ]

    summary = report.summarize_rows(rows, source="baseline.jsonl")

    assert summary["baseline_attempt_stop_reason_distribution"] == {
        "completed": 1,
        "final_answer": 1,
        "iteration_budget_exhausted": 1,
    }
    assert summary["baseline_call_finish_reason_distribution"] == {"length": 2, "stop": 4}
    assert summary["baseline_call_length_finish_total"] == 2
    assert summary["baseline_compat_recovery_total"] == 2
    assert summary["baseline_guard_stop_distribution"] == {"iteration_budget_exhausted": 1}
    assert summary["lambda_planned_decomposition_attempts"] == 1
    assert summary["lambda_executed_decomposition_attempts"] == 1
    assert summary["lambda_task_detection_call_total"] == 1
    assert summary["lambda_repl_llm_query_call_total"] == 2
    assert summary["lambda_decomposition_unknown_attempts"] == 0
    assert summary["rlm_outcome_distribution"] == {
        "failure_search_exhausted": 1,
        "success_final_answer_salvage": 1,
    }
    assert summary["rlm_outcome_unknown_attempts"] == 0


def test_summarize_rows_separates_harmed_case_counts_by_condition() -> None:
    rows = [
        {
            "case_id": "shared-case",
            "condition": "base_repair",
            "success": True,
            "patch_meta": {
                "finish_reason": "stop",
            },
        },
        {
            "case_id": "shared-case",
            "condition": "cot_repair",
            "success": False,
            "patch_meta": {
                "finish_reason": "length",
                "plan_patch_meta": {"finish_reason": "length"},
            },
        },
    ]

    summary = report.summarize_rows(rows, source="mixed.jsonl")

    assert summary["lower_bound_cases_harmed_by_final_caps"] == 1
    assert summary["lower_bound_cases_harmed_by_planner_caps"] == 1


def test_summarize_rows_reports_unknown_bucket_for_old_baseline_artifacts() -> None:
    rows = [
        {
            "case_id": "lambda-old",
            "condition": "lambda_rlm",
            "success": False,
            "patch_meta": {
                "baseline_meta": {
                    "attempts": [
                        {
                            "attempt": 1,
                            "llm_call_count": 2,
                        }
                    ]
                }
            },
        },
        {
            "case_id": "rlm-old",
            "condition": "rlm",
            "success": False,
            "patch_meta": {
                "baseline_meta": {
                    "attempts": [
                        {
                            "attempt": 1,
                            "llm_call_count": 1,
                            "stop_reason": "final_answer",
                        }
                    ]
                }
            },
        },
    ]

    summary = report.summarize_rows(rows, source="old-baseline.jsonl")

    assert summary["baseline_call_finish_reason_unknown_attempts"] == 2
    assert summary["lambda_decomposition_unknown_attempts"] == 1
    assert summary["rlm_outcome_unknown_attempts"] == 1
    assert any(
        "baseline attempts are missing per-call finish_reason telemetry" in item
        for item in summary["unknown_due_to_missing_telemetry"]
    )
    assert any(
        "lambda-RLM attempts are insufficient to assess decomposition execution" in item
        for item in summary["unknown_due_to_missing_telemetry"]
    )
    assert any(
        "RLM attempts are missing terminal outcome classification" in item
        for item in summary["unknown_due_to_missing_telemetry"]
    )


def test_main_emits_json_output_for_fixture(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    path = tmp_path / "results.jsonl"
    path.write_text(
        "\n".join(
            [
                '{"case_id":"case-a","condition":"base_repair","success":false,"patch_meta":{"finish_reason":"length"}}',
                '{"case_id":"case-a","condition":"base_repair","success":false,"patch_meta":{"finish_reason":"length"}}',
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    exit_code = report.main(["--format", "json", str(path)])

    captured = capsys.readouterr()
    assert exit_code == 0
    assert '"lower_bound_cases_harmed_by_final_caps": 1' in captured.out


# ---------------------------------------------------------------------------
# New tests to cover previously missing lines
# ---------------------------------------------------------------------------


class TestLoadRowsEdgeCases:
    """Cover lines 37, 40: empty lines and non-dict JSON in _load_rows."""

    def test_empty_lines_are_skipped(self, tmp_path: Path) -> None:
        path = tmp_path / "data.jsonl"
        path.write_text('{"case_id":"a"}\n\n\n{"case_id":"b"}\n', encoding="utf-8")
        rows = report._load_rows(path)  # type: ignore[attr-defined]
        assert len(rows) == 2

    def test_non_dict_row_raises(self, tmp_path: Path) -> None:
        path = tmp_path / "data.jsonl"
        path.write_text("[1, 2, 3]\n", encoding="utf-8")
        with pytest.raises(ValueError, match="expected JSON object rows"):
            report._load_rows(path)  # type: ignore[attr-defined]


class TestExpectsAttemptFinishReasonWithFinishReasonInPatchMeta:
    """Cover line 86: _expects_attempt_finish_reason when finish_reason key is in patch_meta."""

    def test_finish_reason_key_present_even_without_condition(self) -> None:
        row = {
            "case_id": "x",
            "condition": "unknown_condition",
            "patch_meta": {"finish_reason": ""},
        }
        # The row has 'finish_reason' in patch_meta - that means it expects it
        result = report._expects_attempt_finish_reason(row)  # type: ignore[attr-defined]
        assert result is True

    def test_no_finish_reason_in_patch_meta_and_unknown_condition(self) -> None:
        row = {
            "case_id": "x",
            "condition": "lambda_rlm",
            "patch_meta": {},
        }
        result = report._expects_attempt_finish_reason(row)  # type: ignore[attr-defined]
        assert result is False


class TestBaselineCallFinishReasonCallRecordsFallback:
    """Cover lines 163-172: call_records fallback in _baseline_call_finish_reason_stats."""

    def test_call_records_fallback_counts_finish_reasons(self) -> None:
        rows = [
            {
                "case_id": "x",
                "condition": "lambda_rlm",
                "patch_meta": {
                    "baseline_meta": {
                        "attempts": [
                            {
                                "llm_call_count": 2,
                                # No call_finish_reason_counts - fallback to call_records
                                "call_records": [
                                    {"finish_reason": "stop"},
                                    {"finish_reason": "length"},
                                    {"other_key": "no_finish_reason"},  # skipped
                                    {"finish_reason": ""},  # empty - skipped
                                ],
                            }
                        ]
                    }
                },
            }
        ]
        summary = report.summarize_rows(rows, source="test")
        # stop + length from call_records, not counted as missing
        assert summary["baseline_call_finish_reason_distribution"] == {"length": 1, "stop": 1}
        assert summary["baseline_call_finish_reason_unknown_attempts"] == 0

    def test_call_records_fallback_empty_records_counts_as_missing(self) -> None:
        rows = [
            {
                "case_id": "x",
                "condition": "lambda_rlm",
                "patch_meta": {
                    "baseline_meta": {
                        "attempts": [
                            {
                                "llm_call_count": 1,
                                # No call_finish_reason_counts, no call_records
                            }
                        ]
                    }
                },
            }
        ]
        summary = report.summarize_rows(rows, source="test")
        assert summary["baseline_call_finish_reason_unknown_attempts"] == 1

    def test_call_records_all_missing_finish_reason_counts_as_missing(self) -> None:
        rows = [
            {
                "case_id": "x",
                "condition": "lambda_rlm",
                "patch_meta": {
                    "baseline_meta": {
                        "attempts": [
                            {
                                "llm_call_count": 1,
                                "call_records": [
                                    {"other": "no_finish_reason_key"},
                                ],
                            }
                        ]
                    }
                },
            }
        ]
        summary = report.summarize_rows(rows, source="test")
        assert summary["baseline_call_finish_reason_unknown_attempts"] == 1


class TestLambdaAttemptStatsPlanMetadataFallback:
    """Cover lines 194-195: plan_metadata dict fallback in _lambda_attempt_stats."""

    def test_plan_metadata_depth_positive_counts_as_planned(self) -> None:
        rows = [
            {
                "case_id": "x",
                "condition": "lambda_rlm",
                "patch_meta": {
                    "baseline_meta": {
                        "attempts": [
                            {
                                "llm_call_count": 1,
                                # Uses plan_metadata fallback (no 'planned_decomposition' key)
                                "plan_metadata": {"depth": 2},
                                "executed_decomposition": True,
                                "task_detection_call_count": 1,
                                "repl_llm_query_call_count": 0,
                            }
                        ]
                    }
                },
            }
        ]
        summary = report.summarize_rows(rows, source="test")
        assert summary["lambda_planned_decomposition_attempts"] == 1
        assert summary["lambda_executed_decomposition_attempts"] == 1

    def test_plan_metadata_depth_zero_does_not_count(self) -> None:
        rows = [
            {
                "case_id": "x",
                "condition": "lambda_rlm",
                "patch_meta": {
                    "baseline_meta": {
                        "attempts": [
                            {
                                "llm_call_count": 1,
                                "plan_metadata": {"depth": 0},
                                "executed_decomposition": False,
                                "task_detection_call_count": 1,
                                "repl_llm_query_call_count": 0,
                            }
                        ]
                    }
                },
            }
        ]
        summary = report.summarize_rows(rows, source="test")
        assert summary["lambda_planned_decomposition_attempts"] == 0


class TestDeriveRlmOutcomeBranches:
    """Cover lines 230-240: various _derive_rlm_outcome branches."""

    def test_success_workspace_write(self) -> None:
        attempt = {"candidate_status": "passed", "candidate_source": "workspace_write"}
        result = report._derive_rlm_outcome(attempt)  # type: ignore[attr-defined]
        assert result == "success_workspace_write"

    def test_success_final_answer_salvage(self) -> None:
        attempt = {"candidate_status": "passed", "candidate_source": "final_answer_salvage"}
        result = report._derive_rlm_outcome(attempt)  # type: ignore[attr-defined]
        assert result == "success_final_answer_salvage"

    def test_failure_invalid_workspace_write(self) -> None:
        attempt = {"candidate_status": "invalid_workspace_write"}
        result = report._derive_rlm_outcome(attempt)  # type: ignore[attr-defined]
        assert result == "failure_invalid_workspace_write"

    def test_failure_syntax_error(self) -> None:
        attempt = {"candidate_status": "syntax_error"}
        result = report._derive_rlm_outcome(attempt)  # type: ignore[attr-defined]
        assert result == "failure_syntax_error"

    def test_failure_search_exhausted(self) -> None:
        attempt = {
            "candidate_status": "workspace_unchanged",
            "stop_reason": "iteration_budget_exhausted",
        }
        result = report._derive_rlm_outcome(attempt)  # type: ignore[attr-defined]
        assert result == "failure_search_exhausted"

    def test_failure_workspace_unchanged(self) -> None:
        attempt = {"candidate_status": "workspace_unchanged", "stop_reason": "other"}
        result = report._derive_rlm_outcome(attempt)  # type: ignore[attr-defined]
        assert result == "failure_workspace_unchanged"

    def test_empty_returns_empty_string(self) -> None:
        attempt: dict = {}
        result = report._derive_rlm_outcome(attempt)  # type: ignore[attr-defined]
        assert result == ""

    def test_explicit_outcome_classification_takes_priority(self) -> None:
        attempt = {"outcome_classification": "custom_outcome", "candidate_status": "passed"}
        result = report._derive_rlm_outcome(attempt)  # type: ignore[attr-defined]
        assert result == "custom_outcome"


class TestRlmOutcomeStatsUnknownBranch:
    """Cover lines 254->250: unknown RLM attempt with candidate/stop present but not classified."""

    def test_rlm_outcome_unknown_with_candidate_status_but_no_mapping(self) -> None:
        rows = [
            {
                "case_id": "x",
                "condition": "rlm",
                "patch_meta": {
                    "baseline_meta": {
                        "attempts": [
                            {
                                # No outcome_classification, no matching candidate_status
                                "candidate_status": "completely_unknown_status",
                                "stop_reason": "some_reason",
                            }
                        ]
                    }
                },
            }
        ]
        summary = report.summarize_rows(rows, source="test")
        assert summary["rlm_outcome_unknown_attempts"] == 1


class TestBaselineGuardStopReasonsBranches:
    """Cover lines 271, 285: guard_name vs stop_reason in _baseline_guard_stop_reasons."""

    def test_guard_name_takes_priority(self) -> None:
        rows = [
            {
                "case_id": "x",
                "condition": "rlm",
                "patch_meta": {
                    "baseline_meta": {
                        "attempts": [
                            {
                                "guard_name": "custom_guard",
                                "stop_reason": "iteration_budget_exhausted",
                            }
                        ]
                    }
                },
            }
        ]
        summary = report.summarize_rows(rows, source="test")
        # guard_name takes priority over stop_reason
        assert summary["baseline_guard_stop_distribution"] == {"custom_guard": 1}

    def test_stop_reason_used_when_it_is_guard_stop(self) -> None:
        rows = [
            {
                "case_id": "x",
                "condition": "rlm",
                "patch_meta": {
                    "baseline_meta": {
                        "attempts": [
                            {
                                # No guard_name - fall back to stop_reason in GUARD set
                                "stop_reason": "timeout_exceeded",
                            }
                        ]
                    }
                },
            }
        ]
        summary = report.summarize_rows(rows, source="test")
        assert summary["baseline_guard_stop_distribution"] == {"timeout_exceeded": 1}

    def test_stop_reason_not_in_guard_set_ignored(self) -> None:
        rows = [
            {
                "case_id": "x",
                "condition": "rlm",
                "patch_meta": {
                    "baseline_meta": {
                        "attempts": [
                            {
                                "stop_reason": "completed",  # not in _GUARD_STOP_REASONS
                            }
                        ]
                    }
                },
            }
        ]
        summary = report.summarize_rows(rows, source="test")
        assert summary["baseline_guard_stop_distribution"] == {}


class TestRenderTextSummary:
    """Cover lines 427-510: _render_text_summary and _format_distribution."""

    def _make_summary(self) -> dict:
        rows = [
            {
                "case_id": "case-a",
                "condition": "cot_repair",
                "success": False,
                "patch_meta": {
                    "finish_reason": "length",
                    "plan_patch_meta": {"finish_reason": "length"},
                },
            }
        ]
        return report.summarize_rows(rows, source="test.jsonl")

    def test_render_text_summary_returns_string(self) -> None:
        summary = self._make_summary()
        text = report._render_text_summary(summary)  # type: ignore[attr-defined]
        assert isinstance(text, str)
        assert "Source: test.jsonl" in text

    def test_render_text_summary_includes_key_sections(self) -> None:
        summary = self._make_summary()
        text = report._render_text_summary(summary)  # type: ignore[attr-defined]
        assert "Attempt finish_reason:" in text
        assert "Planner finish_reason:" in text
        assert "Teacher finish_reason:" in text
        assert "Learner finish_reason:" in text
        assert "Baseline attempt stop_reason:" in text
        assert "Baseline call finish_reason:" in text
        assert "Baseline call saturation:" in text
        assert "lambda-RLM decomposition:" in text
        assert "RLM outcomes:" in text
        assert "Prompt overflow rows:" in text
        assert "Lower-bound harmed cases:" in text

    def test_render_text_summary_includes_unknown_bucket_when_present(self) -> None:
        # Row that has missing planner telemetry (no plan_patch_meta despite cot_repair)
        rows = [
            {
                "case_id": "x",
                "condition": "cot_repair",
                "success": False,
                "patch_meta": {"finish_reason": "stop"},
                # no plan_patch_meta or retry_plan_then_patch_meta
            }
        ]
        summary = report.summarize_rows(rows, source="test.jsonl")
        text = report._render_text_summary(summary)  # type: ignore[attr-defined]
        assert "Unknown bucket:" in text
        assert "planner finish_reason" in text

    def test_render_text_summary_includes_baseline_guard_stops(self) -> None:
        rows = [
            {
                "case_id": "x",
                "condition": "rlm",
                "patch_meta": {
                    "baseline_meta": {"attempts": [{"guard_name": "iteration_budget_exhausted"}]}
                },
            }
        ]
        summary = report.summarize_rows(rows, source="test.jsonl")
        text = report._render_text_summary(summary)  # type: ignore[attr-defined]
        assert "Baseline guard stops:" in text


class TestFormatDistribution:
    """Cover lines 514-518: _format_distribution."""

    def test_empty_distribution_shows_none_reported(self) -> None:
        result = report._format_distribution({}, unknown=0)  # type: ignore[attr-defined]
        assert result == "none reported (unknown=0)"

    def test_non_dict_distribution_shows_none_reported(self) -> None:
        result = report._format_distribution(None, unknown=5)  # type: ignore[attr-defined]
        assert result == "none reported (unknown=5)"

    def test_populated_distribution_formats_correctly(self) -> None:
        result = report._format_distribution({"stop": 3, "length": 1}, unknown=2)  # type: ignore[attr-defined]
        assert "stop=3" in result
        assert "length=1" in result
        assert "unknown=2" in result


class TestMainTextFormat:
    """Cover line 541: main() text format output branch."""

    def test_main_text_format_produces_readable_output(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path = tmp_path / "results.jsonl"
        path.write_text(
            '{"case_id":"case-a","condition":"base_repair","success":false,"patch_meta":{"finish_reason":"length"}}\n',
            encoding="utf-8",
        )
        exit_code = report.main([str(path)])  # default format is "text"
        captured = capsys.readouterr()
        assert exit_code == 0
        assert "Source:" in captured.out
        assert "Attempt finish_reason:" in captured.out

    def test_main_multiple_paths_text_format(
        self, tmp_path: Path, capsys: pytest.CaptureFixture[str]
    ) -> None:
        path1 = tmp_path / "a.jsonl"
        path2 = tmp_path / "b.jsonl"
        path1.write_text(
            '{"case_id":"a","condition":"base_repair","success":false,"patch_meta":{"finish_reason":"stop"}}\n',
            encoding="utf-8",
        )
        path2.write_text(
            '{"case_id":"b","condition":"cot_repair","success":true,"patch_meta":{"finish_reason":"stop","plan_patch_meta":{"finish_reason":"stop"}}}\n',
            encoding="utf-8",
        )
        exit_code = report.main([str(path1), str(path2)])
        captured = capsys.readouterr()
        assert exit_code == 0
        # Multiple summaries are joined with double newlines
        assert captured.out.count("Source:") == 2
