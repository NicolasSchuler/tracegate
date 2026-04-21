"""Regression tests for eval token-usage synchronization semantics."""

from __future__ import annotations

from evals import eval_attempt_flow, run_eval


def test_sync_attempt_usage_metadata_preserves_exact_usage() -> None:
    attempt_metadata: dict[str, object] = {}
    patch_meta: dict[str, object] = {
        "usage": {
            "prompt_tokens": 11,
            "completion_tokens": 7,
            "total_tokens": 18,
            "output_tokens_details": {"reasoning_tokens": 5},
        }
    }

    run_eval._sync_attempt_usage_metadata(
        attempt_metadata=attempt_metadata,
        patch_meta=patch_meta,
        llm_call_count=3,
    )

    assert attempt_metadata == {
        "llm_call_count": 3,
        "token_usage_available": True,
        "tokens_prompt": 11,
        "tokens_completion": 7,
        "tokens_total": 18,
        "tokens_reasoning": 5,
        "reasoning_token_usage_available": True,
        "tokens_input": 11,
        "tokens_output": 7,
    }
    assert patch_meta["token_usage_available"] is True
    assert patch_meta["tokens_prompt"] == 11
    assert patch_meta["tokens_completion"] == 7
    assert patch_meta["tokens_total"] == 18
    assert patch_meta["tokens_reasoning"] == 5
    assert patch_meta["reasoning_token_usage_available"] is True
    assert patch_meta["tokens_input"] == 11
    assert patch_meta["tokens_output"] == 7


def test_sync_attempt_usage_metadata_zero_fills_when_usage_is_partial_only() -> None:
    attempt_metadata: dict[str, object] = {}
    patch_meta: dict[str, object] = {
        "baseline_metrics": {
            "usage_available": False,
            "usage_reported_call_count": 2,
            "usage_missing_call_count": 1,
            "reported_tokens_prompt": 20,
            "reported_tokens_completion": 6,
            "reported_tokens_total": 26,
        }
    }

    run_eval._sync_attempt_usage_metadata(
        attempt_metadata=attempt_metadata,
        patch_meta=patch_meta,
        llm_call_count=3,
    )

    assert attempt_metadata == {
        "llm_call_count": 3,
        "token_usage_available": False,
        "tokens_prompt": 0,
        "tokens_completion": 0,
        "tokens_total": 0,
        "tokens_reasoning": 0,
        "reasoning_token_usage_available": False,
        "tokens_input": 0,
        "tokens_output": 0,
    }
    assert patch_meta["token_usage_available"] is False
    assert patch_meta["tokens_prompt"] == 0
    assert patch_meta["tokens_completion"] == 0
    assert patch_meta["tokens_total"] == 0
    assert patch_meta["tokens_reasoning"] == 0
    assert patch_meta["reasoning_token_usage_available"] is False
    assert patch_meta["tokens_input"] == 0
    assert patch_meta["tokens_output"] == 0
    assert patch_meta["baseline_metrics"]["reported_tokens_total"] == 26


def test_sync_attempt_usage_metadata_uses_exact_baseline_usage_after_stale_false_is_cleared() -> (
    None
):
    attempt_metadata: dict[str, object] = {}
    patch_meta: dict[str, object] = {
        "token_usage_available": False,
        "baseline_metrics": {
            "usage_available": True,
            "usage_reported_call_count": 3,
            "usage_missing_call_count": 0,
            "reported_tokens_prompt": 20,
            "reported_tokens_completion": 6,
            "reported_tokens_total": 26,
        },
    }

    eval_attempt_flow._reset_patch_usage_sync_metadata(patch_meta)
    run_eval._sync_attempt_usage_metadata(
        attempt_metadata=attempt_metadata,
        patch_meta=patch_meta,
        llm_call_count=3,
    )

    assert attempt_metadata == {
        "llm_call_count": 3,
        "token_usage_available": True,
        "tokens_prompt": 20,
        "tokens_completion": 6,
        "tokens_total": 26,
        "tokens_reasoning": 0,
        "reasoning_token_usage_available": False,
        "tokens_input": 20,
        "tokens_output": 6,
    }
    assert patch_meta["token_usage_available"] is True
    assert patch_meta["tokens_prompt"] == 20
    assert patch_meta["tokens_completion"] == 6
    assert patch_meta["tokens_total"] == 26
    assert patch_meta["tokens_reasoning"] == 0
    assert patch_meta["reasoning_token_usage_available"] is False
    assert patch_meta["tokens_input"] == 20
    assert patch_meta["tokens_output"] == 6


def test_reset_patch_usage_sync_metadata_clears_stale_token_fields() -> None:
    patch_meta: dict[str, object] = {
        "llm_call_count": 3,
        "tokens_prompt": 20,
        "tokens_completion": 6,
        "tokens_total": 26,
        "tokens_reasoning": 9,
        "usage": {
            "prompt_tokens": 20,
            "completion_tokens": 6,
            "total_tokens": 26,
            "reasoning_tokens": 9,
        },
        "token_usage_available": False,
        "reasoning_token_usage_available": True,
        "tokens_input": 20,
        "tokens_output": 6,
        "baseline_metrics": {
            "usage_available": True,
            "reported_tokens_total": 26,
        },
        "propose_duration_sec": 1.2,
    }

    eval_attempt_flow._reset_patch_usage_sync_metadata(patch_meta)

    for key in (
        "llm_call_count",
        "tokens_prompt",
        "tokens_completion",
        "tokens_total",
        "tokens_reasoning",
        "usage",
        "token_usage_available",
        "reasoning_token_usage_available",
        "tokens_input",
        "tokens_output",
    ):
        assert key not in patch_meta
    assert patch_meta["baseline_metrics"] == {
        "usage_available": True,
        "reported_tokens_total": 26,
    }
    assert patch_meta["propose_duration_sec"] == 1.2


def test_sync_attempt_prompt_char_metadata_records_input_and_output_chars() -> None:
    attempt_metadata: dict[str, object] = {}
    patch_meta: dict[str, object] = {}

    run_eval._sync_attempt_prompt_char_metadata(
        attempt_metadata=attempt_metadata,
        patch_meta=patch_meta,
        prompt_chars_total=1234,
        completion_chars_total=321,
    )

    assert attempt_metadata["prompt_chars_total"] == 1234
    assert attempt_metadata["completion_chars_total"] == 321
    assert attempt_metadata["chars_input"] == 1234
    assert attempt_metadata["chars_output"] == 321
    assert attempt_metadata["chars_total"] == 1555
    assert patch_meta["prompt_chars_total"] == 1234
    assert patch_meta["completion_chars_total"] == 321
    assert patch_meta["chars_input"] == 1234
    assert patch_meta["chars_output"] == 321
    assert patch_meta["chars_total"] == 1555


def test_completion_chars_from_response_prefers_exact_meta_over_diff() -> None:
    assert (
        run_eval._completion_chars_from_response(
            diff="x" * 200,
            patch_meta={"completion_chars_total": 37},
        )
        == 37
    )


def test_patch_attempt_char_deltas_use_adapter_owned_chars_for_baselines() -> None:
    assert run_eval._patch_attempt_char_deltas(
        cycle_prompt="this prompt should not be counted",
        diff="diff text should not be counted either",
        patch_meta={
            "prompt_chars_total": 91,
            "completion_chars_total": 47,
        },
        is_baseline_patcher=True,
    ) == (91, 47)


def test_patch_attempt_char_deltas_do_not_fallback_to_cycle_prompt_for_baselines() -> None:
    assert run_eval._patch_attempt_char_deltas(
        cycle_prompt="outer cycle prompt",
        diff="diff text",
        patch_meta={},
        is_baseline_patcher=True,
    ) == (0, 0)


def test_token_usage_available_from_meta_prefers_explicit_flag() -> None:
    patch_meta = {
        "token_usage_available": False,
        "usage": {
            "prompt_tokens": 11,
            "completion_tokens": 7,
            "total_tokens": 18,
        },
    }

    assert run_eval._token_usage_available_from_meta(patch_meta) is False


def test_token_usage_available_from_meta_rejects_zeroed_available_flags() -> None:
    patch_meta = {
        "token_usage_available": True,
        "baseline_metrics": {
            "usage_available": True,
        },
    }

    assert run_eval._token_usage_available_from_meta(patch_meta) is False


def test_usage_token_counts_from_patch_meta_falls_back_to_reported_baseline_tokens() -> None:
    patch_meta = {
        "baseline_metrics": {
            "usage_available": True,
            "reported_tokens_prompt": 17,
            "reported_tokens_completion": 9,
            "reported_tokens_total": 26,
        }
    }

    assert run_eval._usage_token_counts_from_patch_meta(patch_meta) == (17, 9, 26)


def test_aggregate_attempt_llm_usage_ignores_partial_baseline_usage_payload() -> None:
    llm_calls, prompt_tokens, completion_tokens, total_tokens = (
        run_eval._aggregate_attempt_llm_usage(
            stage_a_meta=None,
            patch_meta={
                "usage": {
                    "prompt_tokens": 17,
                    "completion_tokens": 9,
                    "total_tokens": 26,
                },
                "baseline_metrics": {
                    "usage_available": False,
                    "usage_reported_call_count": 2,
                    "usage_missing_call_count": 1,
                    "reported_tokens_prompt": 17,
                    "reported_tokens_completion": 9,
                    "reported_tokens_total": 26,
                },
            },
            stage_a_default_llm_calls=0,
            patch_default_llm_calls=0,
        )
    )

    assert llm_calls == 0
    assert prompt_tokens == 0
    assert completion_tokens == 0
    assert total_tokens == 0


def test_record_token_usage_state_marks_timeout_without_usage_unavailable() -> None:
    attempt_meta = {
        "error": "timeout",
        "llm_call_count": 2,
        "baseline_metrics": {"usage_available": False},
    }

    state = run_eval._record_token_usage_state(
        attempt_meta=attempt_meta,
        patch_meta=None,
        prompt_tokens=0,
        total_tokens=0,
    )

    assert state == "unavailable"


def test_record_token_usage_state_marks_zero_call_rows_not_applicable() -> None:
    state = run_eval._record_token_usage_state(
        attempt_meta={},
        patch_meta=None,
        prompt_tokens=0,
        total_tokens=0,
    )

    assert state == "not_applicable"


def test_request_context_and_failure_helpers_cover_edge_branches() -> None:
    assert run_eval._request_context_failure_kind_from_meta(None) == ""
    assert (
        run_eval._request_context_failure_kind_from_meta(
            {"request_context_failure_kind": "backend_unavailable"}
        )
        == "backend_unavailable"
    )
    assert (
        run_eval._request_context_failure_kind_from_meta({"error": "request_budget_exceeded"})
        == "request_budget_exceeded"
    )
    assert (
        run_eval._request_context_failure_kind_from_meta({"error": "openai_compat_reasoning_only"})
        == "compat_reasoning_only"
    )
    assert (
        run_eval._request_context_failure_kind_from_meta(
            {"request_context_backend_unavailable": True}
        )
        == "backend_unavailable"
    )
    assert (
        run_eval._request_context_failure_kind_from_meta(
            {"error": "invalid_request_context_payload"}
        )
        == "invalid_payload"
    )
    assert (
        run_eval._request_context_failure_kind_from_meta({"error": "request_failed"}) == "transport"
    )
    assert (
        run_eval._request_context_failure_kind_from_meta({"error": "http_error", "status": 503})
        == "backend_unavailable"
    )
    assert (
        run_eval._request_context_failure_kind_from_meta({"error": "http_error", "status": 400})
        == "transport"
    )

    assert (
        run_eval._request_context_recovery_failure_kind_from_meta({"error": "request_failed"})
        == "transport"
    )
    assert (
        run_eval._request_context_recovery_failure_kind_from_meta(
            {"error": "http_error", "status": 429}
        )
        == "transport"
    )
    assert (
        run_eval._request_context_recovery_failure_kind_from_meta(
            {"error": "http_error", "status": 503}
        )
        == "backend_unavailable"
    )
    assert run_eval._request_context_retry_counts_from_meta(
        {
            "request_context_retry_count": "2",
            "request_context_transport_retry_count": 3,
        }
    ) == (2, 3)

    assert run_eval._adaptive_helper_outcome_for_attempt(
        helper_invoked=True,
        helper_materialized_context=True,
        helper_decision_reason="",
        stage_a_meta=None,
    ) == (run_eval.ADAPTIVE_HELPER_OUTCOME_MATERIALIZED, "")
    assert run_eval._adaptive_helper_outcome_for_attempt(
        helper_invoked=True,
        helper_materialized_context=False,
        helper_decision_reason="",
        stage_a_meta={"error": "request_budget_exceeded"},
    ) == (run_eval.ADAPTIVE_HELPER_OUTCOME_REQUEST_BUDGET_EXCEEDED, "request_budget_exceeded")
    assert run_eval._adaptive_helper_outcome_for_attempt(
        helper_invoked=True,
        helper_materialized_context=False,
        helper_decision_reason="llm_invalid_response_no_helper",
        stage_a_meta=None,
    ) == (run_eval.ADAPTIVE_HELPER_OUTCOME_INVALID_PAYLOAD, "invalid_payload")
    assert run_eval._adaptive_helper_outcome_for_attempt(
        helper_invoked=True,
        helper_materialized_context=False,
        helper_decision_reason="",
        stage_a_meta={"error": "request_failed"},
    ) == (run_eval.ADAPTIVE_HELPER_OUTCOME_TRANSPORT_FAILED, "transport")
    assert run_eval._adaptive_helper_outcome_for_attempt(
        helper_invoked=True,
        helper_materialized_context=False,
        helper_decision_reason="",
        stage_a_meta={"error": "http_error", "status": 503},
    ) == (run_eval.ADAPTIVE_HELPER_OUTCOME_UNAVAILABLE_BACKEND, "backend_unavailable")
    assert run_eval._adaptive_helper_outcome_for_attempt(
        helper_invoked=True,
        helper_materialized_context=False,
        helper_decision_reason="",
        stage_a_meta={"error": "openai_compat_reasoning_only"},
    ) == (run_eval.ADAPTIVE_HELPER_OUTCOME_COMPAT_REASONING_ONLY, "compat_reasoning_only")
    assert run_eval._adaptive_helper_outcome_for_attempt(
        helper_invoked=True,
        helper_materialized_context=False,
        helper_decision_reason="",
        stage_a_meta={"error": "invalid_request_context_payload"},
    ) == (run_eval.ADAPTIVE_HELPER_OUTCOME_INVALID_PAYLOAD, "invalid_payload")
    assert run_eval._adaptive_helper_outcome_for_attempt(
        helper_invoked=True,
        helper_materialized_context=False,
        helper_decision_reason="",
        stage_a_meta=None,
    ) == (run_eval.ADAPTIVE_HELPER_OUTCOME_SEMANTIC_NO_CONTEXT, "")
    assert run_eval._adaptive_helper_outcome_for_attempt(
        helper_invoked=False,
        helper_materialized_context=False,
        helper_decision_reason="request_context_unavailable",
        stage_a_meta=None,
    ) == (run_eval.ADAPTIVE_HELPER_OUTCOME_SKIPPED_UNSUPPORTED, "")
    assert run_eval._adaptive_helper_outcome_for_attempt(
        helper_invoked=False,
        helper_materialized_context=False,
        helper_decision_reason="helper_saturation_guard",
        stage_a_meta=None,
    ) == (run_eval.ADAPTIVE_HELPER_OUTCOME_SKIPPED_GUARDRAIL, "")
    assert run_eval._adaptive_helper_outcome_for_attempt(
        helper_invoked=False,
        helper_materialized_context=False,
        helper_decision_reason="",
        stage_a_meta=None,
    ) == (run_eval.ADAPTIVE_HELPER_OUTCOME_NOT_INVOKED, "")
    assert run_eval._adaptive_helper_outcome_for_attempt(
        helper_invoked=False,
        helper_materialized_context=False,
        helper_decision_reason="other_policy",
        stage_a_meta=None,
    ) == (run_eval.ADAPTIVE_HELPER_OUTCOME_SKIPPED_POLICY, "")


def test_usage_and_rendering_helpers_cover_edge_branches() -> None:
    assert run_eval._usage_token_counts_from_usage_payload(
        {
            "prompt_tokens": 4,
            "completion_tokens": 6,
            "total_tokens": "bad",
        }
    ) == (4, 6, 10)
    assert run_eval._usage_token_counts_from_usage_payload(
        {
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "input_tokens": 5,
            "output_tokens": 3,
        }
    ) == (5, 3, 8)
    assert run_eval._reasoning_token_counts_from_usage_payload(
        {"output_tokens_details": {"reasoning_tokens": 4}}
    ) == (4, True)
    assert run_eval._reasoning_token_counts_from_usage_payload(
        {"completion_tokens_details": {"reasoning_tokens": 0}}
    ) == (0, True)
    assert run_eval._reasoning_token_counts_from_usage_payload({"reasoning_tokens": "bad"}) == (
        0,
        False,
    )

    assert run_eval._token_usage_blocked_from_meta({"token_usage_available": False}) is True
    assert run_eval._token_usage_blocked_from_meta({"usage_available": False}) is True
    assert (
        run_eval._token_usage_blocked_from_meta({"baseline_metrics": {"usage_available": False}})
        is True
    )
    assert run_eval._token_usage_blocked_from_meta(None) is False

    assert run_eval._usage_token_counts_from_patch_meta(
        {
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 2,
                "total_tokens": 3,
            },
            "tokens_prompt": 7,
            "tokens_completion": 8,
            "tokens_total": 9,
        }
    ) == (7, 8, 9)
    assert run_eval._reasoning_token_counts_from_patch_meta(
        {
            "usage": {"output_tokens_details": {"reasoning_tokens": 6}},
        }
    ) == (6, True)

    assert run_eval._completion_chars_from_response(diff="abcd", patch_meta=None) == 4
    assert (
        run_eval._completion_chars_from_response(
            diff=None,
            patch_meta={"raw_response": "wxyz"},
        )
        == 4
    )
    assert (
        run_eval._completion_chars_from_response(
            diff=None,
            patch_meta={"response": "pq"},
        )
        == 2
    )
    assert (
        run_eval._completion_chars_from_response(
            diff=None,
            patch_meta={"patch_diff_chars": 9},
        )
        == 9
    )

    assert run_eval._prompt_chars_from_patch_meta({"response_input_chars_total": 12}) == 12
    assert run_eval._prompt_chars_from_patch_meta(None) == 0
    assert run_eval._baseline_metrics_from_patch_meta(None) is None
    assert run_eval._baseline_metric_float({"baseline_metrics": {"ratio": -2}}, "ratio") == 0.0
    assert run_eval._baseline_metric_float({"baseline_metrics": {"ratio": "x"}}, "ratio") is None
    assert run_eval._llm_call_count_from_meta({"llm_call_count": 4}) == 4
    assert run_eval._llm_call_count_from_meta(None, default=-3) == 0
    assert run_eval._usage_available_flag_from_meta({"usage_available": True}) is True
    assert (
        run_eval._usage_available_flag_from_meta({"baseline_metrics": {"usage_available": False}})
        is False
    )
    assert run_eval._usage_available_flag_from_meta(None) is None

    assert (
        run_eval._token_usage_available_from_meta(
            {
                "token_usage_available": True,
                "usage": {
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
                },
            }
        )
        is False
    )
    assert (
        run_eval._token_usage_available_from_meta(
            {
                "usage_available": True,
                "usage": {
                    "prompt_tokens": 2,
                    "completion_tokens": 1,
                    "total_tokens": 3,
                },
            }
        )
        is True
    )

    assert run_eval._timeout_error_from_meta({"baseline_meta": {"error": "timeout"}}) is True
    assert (
        run_eval._record_token_usage_state(
            attempt_meta={"llm_call_count": 1},
            patch_meta=None,
            prompt_tokens=1,
            total_tokens=0,
        )
        == "available"
    )
    assert (
        run_eval._format_progress_token_usage(
            prompt_tokens=4,
            completion_tokens=5,
            total_tokens=9,
            available_records=0,
            unavailable_records=2,
        )
        == "tokens=unavailable"
    )
    assert (
        run_eval._format_progress_token_usage(
            prompt_tokens=4,
            completion_tokens=5,
            total_tokens=9,
            available_records=1,
            unavailable_records=1,
        )
        == "tokens=prompt:4,completion:5,total:9,coverage=partial"
    )
    assert run_eval._aggregate_attempt_llm_usage(
        stage_a_meta={
            "llm_call_count": 2,
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 2,
                "total_tokens": 3,
            },
        },
        patch_meta={
            "llm_call_count": 1,
            "usage": {
                "prompt_tokens": 4,
                "completion_tokens": 5,
                "total_tokens": 9,
            },
        },
        stage_a_default_llm_calls=0,
        patch_default_llm_calls=0,
    ) == (3, 5, 7, 12)
    assert run_eval._prompt_chars_for_llm_calls(None, llm_call_count=3) == 0
    assert run_eval._prompt_chars_for_llm_calls("abc", llm_call_count=2) == 6
    assert run_eval._clip_patcher_meta_text("  hello  ", max_chars=10) == "hello"
    assert run_eval._clip_patcher_meta_text("abcdefghij", max_chars=5) == "ab..."

    compact_summary = run_eval._summarize_context_compaction_meta(
        {
            "llm_call_count": 1,
            "compact_context_duration_sec": 1.5,
            "status": 200,
            "usage": {
                "input_tokens": 3,
                "output_tokens": 4,
                "total_tokens": 7,
            },
            "error": "x" * 300,
            "body": "y" * 400,
        }
    )
    assert compact_summary["usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 4,
        "total_tokens": 7,
    }
    assert compact_summary["error"].endswith("...")
    assert compact_summary["body"].endswith("...")

    recovery_summary = run_eval._summarize_recovery_usage_meta(
        {
            "llm_call_count": 2,
            "usage": {
                "prompt_tokens": 1,
                "completion_tokens": 2,
                "total_tokens": 3,
            },
            "reason": "z" * 300,
        }
    )
    assert recovery_summary["usage"] == {
        "prompt_tokens": 1,
        "completion_tokens": 2,
        "total_tokens": 3,
    }
    assert recovery_summary["reason"].endswith("...")


def test_metadata_sync_helpers_cover_state_updates() -> None:
    attempt_meta: dict[str, object] = {}
    patch_meta: dict[str, object] = {}

    run_eval._apply_usage_totals_to_patch_meta(
        patch_meta=patch_meta,
        llm_call_count=2,
        tokens_prompt=3,
        tokens_completion=4,
        tokens_total=7,
        tokens_reasoning=5,
        reasoning_token_usage_available=True,
    )
    assert patch_meta["llm_call_count"] == 2
    assert patch_meta["usage"] == {
        "prompt_tokens": 3,
        "completion_tokens": 4,
        "total_tokens": 7,
        "reasoning_tokens": 5,
    }
    assert patch_meta["tokens_reasoning"] == 5
    assert patch_meta["reasoning_token_usage_available"] is True

    blocked_meta: dict[str, object] = {"usage_available": False}
    run_eval._apply_usage_totals_to_patch_meta(
        patch_meta=blocked_meta,
        llm_call_count=1,
        tokens_prompt=5,
        tokens_completion=6,
        tokens_total=11,
    )
    assert blocked_meta["llm_call_count"] == 1
    assert "usage" not in blocked_meta
    assert blocked_meta["tokens_total"] == 0
    assert blocked_meta["tokens_reasoning"] == 0
    assert blocked_meta["reasoning_token_usage_available"] is False

    run_eval._sync_syntax_preflight_metadata(
        attempt_metadata=attempt_meta,
        patch_meta=patch_meta,
        enabled=True,
        cycles_used=2,
        failures=1,
        last_error_type="SyntaxError",
        last_error_msg="bad syntax",
        events=[{"cycle": 1}],
    )
    assert patch_meta["syntax_preflight_enabled"] is True
    assert patch_meta["syntax_preflight_events"] == [{"cycle": 1}]
    assert attempt_meta["syntax_preflight_enabled"] is True

    run_eval._sync_patcher_request_failure_metadata(
        attempt_metadata=attempt_meta,
        patch_meta=patch_meta,
        failed=True,
        kind="transport",
        error="request_failed",
        status=503,
        exc="ConnectionError: refused",
        runner_forced_patch_skipped=True,
        runner_forced_patch_skip_reason="transport_error",
    )
    assert patch_meta["patcher_request_failed"] is True
    assert patch_meta["patcher_request_failure_kind"] == "transport"
    assert patch_meta["runner_forced_patch_skipped"] is True

    run_eval._sync_patch_sanitize_metadata(
        attempt_metadata=attempt_meta,
        patch_meta=patch_meta,
        enabled=True,
        passed=False,
        rejected=True,
        reasons=["bad path"],
        error_msg="sanitize failed",
    )
    assert patch_meta["patch_sanitize_rejected"] is True
    assert attempt_meta["patch_sanitize_error_msg"] == "sanitize failed"

    diff = "diff --git a/foo.py b/foo.py\n--- a/foo.py\n+++ b/foo.py\n@@ -1 +1 @@\n-a\n+b\n"
    run_eval._sync_patch_diff_metadata(patch_meta=patch_meta, diff=diff)
    assert patch_meta["patch_diff_chars"] == len(diff)
    assert patch_meta["patch_file_count"] == 1

    run_eval._sync_patch_verifier_metadata(
        attempt_metadata=attempt_meta,
        patch_meta=patch_meta,
        enabled=True,
        scope=run_eval.PATCH_VERIFIER_SCOPE_ALL,
        passed=False,
        score=1,
        reasons=["too small"],
        min_score=3,
    )
    assert patch_meta["patch_verifier_scope"] == run_eval.PATCH_VERIFIER_SCOPE_ALL
    assert patch_meta["patch_verifier_score"] == 1

    run_eval._sync_patch_verifier_recovery_metadata(
        attempt_metadata=attempt_meta,
        patch_meta=patch_meta,
        eligible=True,
        invoked=True,
        result=run_eval.PATCH_VERIFIER_RECOVERY_RESULT_PASSED_VERIFIER,
        prompt_chars_before=10,
        prompt_chars_after=5,
        duration_sec=1.5,
        terminal_note="recovered",
        score_before=1,
        reasons_before=["too small"],
        patch_files_before=["foo.py"],
        source_scope_matched_before=True,
        raw_meta={"detail": "ok"},
    )
    assert (
        patch_meta["patch_verifier_recovery_result"]
        == run_eval.PATCH_VERIFIER_RECOVERY_RESULT_PASSED_VERIFIER
    )
    assert attempt_meta["patch_verifier_recovery_raw_meta"] == {"detail": "ok"}

    retry_history_turns: list[str] = []
    run_eval._append_retry_history_turn_if_enabled(
        retry_context_mode="history",
        retry_history_turns=retry_history_turns,
        attempt=3,
        note="note",
        failure_signature="sig",
        patch_meta={
            "retry_summary": {
                "attempt": 3,
                "hypothesis_tested": "hypothesis",
                "evidence_used": "evidence",
                "outcome": "outcome",
                "new_constraint": "constraint",
                "failure_signature": "sig",
            }
        },
    )
    assert retry_history_turns and "Failure signature: sig" in retry_history_turns[0]

    run_eval._append_retry_history_turn_if_enabled(
        retry_context_mode="fresh",
        retry_history_turns=retry_history_turns,
        attempt=4,
        note="note",
        failure_signature="sig2",
        patch_meta={},
    )
    assert len(retry_history_turns) == 1
