"""Coverage-boost tests for evals/eval_attempt_flow.py — part 2: context and payload utilities.

Tests _coerce_context_request_payload, _filter_context_request_payload_for_action,
_request_context_failure_kind_from_meta, _request_context_recovery_failure_kind_from_meta,
_adaptive_helper_outcome_for_attempt, _llm_call_count_from_meta,
_aggregate_attempt_llm_usage, _append_retry_history_turn_if_enabled.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from evals import eval_attempt_flow

# ---------------------------------------------------------------------------
# _adaptive_helper_outcome_for_attempt
# ---------------------------------------------------------------------------


class TestAdaptiveHelperOutcome:
    def test_materialized(self) -> None:
        o, _ = eval_attempt_flow._adaptive_helper_outcome_for_attempt(
            helper_invoked=True,
            helper_materialized_context=True,
            helper_decision_reason="llm_requested_helper",
            stage_a_meta=None,
        )
        assert o == "materialized"

    def test_backend_unavailable(self) -> None:
        o, fk = eval_attempt_flow._adaptive_helper_outcome_for_attempt(
            helper_invoked=True,
            helper_materialized_context=False,
            helper_decision_reason="",
            stage_a_meta={"request_context_backend_unavailable": True},
        )
        assert o == "unavailable_backend" and fk == "backend_unavailable"

    def test_transport_failed(self) -> None:
        o, _fk = eval_attempt_flow._adaptive_helper_outcome_for_attempt(
            helper_invoked=True,
            helper_materialized_context=False,
            helper_decision_reason="",
            stage_a_meta={"error": "request_failed"},
        )
        assert o == "transport_failed"

    def test_request_budget_exceeded(self) -> None:
        o, _ = eval_attempt_flow._adaptive_helper_outcome_for_attempt(
            helper_invoked=True,
            helper_materialized_context=False,
            helper_decision_reason="",
            stage_a_meta={"error": "request_budget_exceeded"},
        )
        assert o == "request_budget_exceeded"

    def test_compat_reasoning_only(self) -> None:
        o, fk = eval_attempt_flow._adaptive_helper_outcome_for_attempt(
            helper_invoked=True,
            helper_materialized_context=False,
            helper_decision_reason="",
            stage_a_meta={"error": "openai_compat_reasoning_only"},
        )
        assert o == "compat_reasoning_only"
        assert fk == "compat_reasoning_only"

    def test_invalid_payload(self) -> None:
        o, _ = eval_attempt_flow._adaptive_helper_outcome_for_attempt(
            helper_invoked=True,
            helper_materialized_context=False,
            helper_decision_reason="llm_invalid_response_no_helper",
            stage_a_meta=None,
        )
        assert o == "invalid_payload"

    def test_semantic_no_context(self) -> None:
        o, _ = eval_attempt_flow._adaptive_helper_outcome_for_attempt(
            helper_invoked=True,
            helper_materialized_context=False,
            helper_decision_reason="llm_requested_helper",
            stage_a_meta={},
        )
        assert o == "semantic_no_context"

    def test_skipped_unsupported(self) -> None:
        o, _ = eval_attempt_flow._adaptive_helper_outcome_for_attempt(
            helper_invoked=False,
            helper_materialized_context=False,
            helper_decision_reason="request_context_unavailable",
            stage_a_meta=None,
        )
        assert o == "skipped_unsupported"

    def test_skipped_guardrail_pregate(self) -> None:
        o, _ = eval_attempt_flow._adaptive_helper_outcome_for_attempt(
            helper_invoked=False,
            helper_materialized_context=False,
            helper_decision_reason="informativeness_pregate_skip",
            stage_a_meta=None,
        )
        assert o == "skipped_guardrail"

    def test_skipped_guardrail_budget(self) -> None:
        o, _ = eval_attempt_flow._adaptive_helper_outcome_for_attempt(
            helper_invoked=False,
            helper_materialized_context=False,
            helper_decision_reason="helper_budget_exhausted",
            stage_a_meta=None,
        )
        assert o == "skipped_guardrail"

    def test_skipped_guardrail_saturation(self) -> None:
        o, _ = eval_attempt_flow._adaptive_helper_outcome_for_attempt(
            helper_invoked=False,
            helper_materialized_context=False,
            helper_decision_reason="helper_saturation_guard",
            stage_a_meta=None,
        )
        assert o == "skipped_guardrail"


class TestControllerRuntimeEvidenceInputs:
    def test_fixed_v1_uses_fallback_bundle_items(self, tmp_path: Path) -> None:
        workdir = tmp_path / "case"
        workdir.mkdir()
        (workdir / "buggy.py").write_text(
            "def buggy(value):\n    return value + 1\n",
            encoding="utf-8",
        )
        (workdir / "test_buggy.py").write_text(
            "from buggy import buggy\n\n\ndef test_buggy():\n    assert buggy(1) == 1\n",
            encoding="utf-8",
        )

        items, new_ids, reused_ids = eval_attempt_flow._controller_runtime_evidence_inputs(
            baseline_hybrid_guidance_mode="fixed_v1",
            seen_evidence_ids=set(),
            workdir=workdir,
            snapshot={
                "frames": [
                    {
                        "file_rel": "buggy.py",
                        "line": 2,
                        "function": "buggy",
                        "code": "return value + 1",
                        "locals": {"value": 1},
                    }
                ]
            },
            stdout="FAILED test_buggy.py::test_buggy - AssertionError",
            stderr="",
            max_total_chars=4000,
            file_index_lines=["buggy.py", "test_buggy.py"],
            combined_evidence_items=[
                {"id": "old", "type": "frame", "locator": "frame:0", "text": "old evidence"}
            ],
            new_evidence_ids=["old"],
            reused_evidence_ids=[],
        )

        assert items
        assert all(str(item.get("id") or "").strip() for item in items)
        assert new_ids == [str(item.get("id") or "") for item in items]
        assert reused_ids == []
        assert all(item.get("type") == "file" for item in items)
        assert any(str(item.get("path") or "") == "buggy.py" for item in items)

    def test_fixed_v1_marks_seen_fallback_bundle_items_as_reused(self, tmp_path: Path) -> None:
        workdir = tmp_path / "case"
        workdir.mkdir()
        (workdir / "buggy.py").write_text(
            "def buggy(value):\n    return value + 1\n",
            encoding="utf-8",
        )
        (workdir / "test_buggy.py").write_text(
            "from buggy import buggy\n\n\ndef test_buggy():\n    assert buggy(1) == 1\n",
            encoding="utf-8",
        )

        first_items, first_new_ids, first_reused_ids = (
            eval_attempt_flow._controller_runtime_evidence_inputs(
                baseline_hybrid_guidance_mode="fixed_v1",
                seen_evidence_ids=set(),
                workdir=workdir,
                snapshot={
                    "frames": [
                        {
                            "file_rel": "buggy.py",
                            "line": 2,
                            "function": "buggy",
                            "code": "return value + 1",
                            "locals": {"value": 1},
                        }
                    ]
                },
                stdout="FAILED test_buggy.py::test_buggy - AssertionError",
                stderr="",
                max_total_chars=4000,
                file_index_lines=["buggy.py", "test_buggy.py"],
                combined_evidence_items=[],
                new_evidence_ids=[],
                reused_evidence_ids=[],
            )
        )

        second_items, second_new_ids, second_reused_ids = (
            eval_attempt_flow._controller_runtime_evidence_inputs(
                baseline_hybrid_guidance_mode="fixed_v1",
                seen_evidence_ids=set(first_new_ids),
                workdir=workdir,
                snapshot={
                    "frames": [
                        {
                            "file_rel": "buggy.py",
                            "line": 2,
                            "function": "buggy",
                            "code": "return value + 1",
                            "locals": {"value": 1},
                        }
                    ]
                },
                stdout="FAILED test_buggy.py::test_buggy - AssertionError",
                stderr="",
                max_total_chars=4000,
                file_index_lines=["buggy.py", "test_buggy.py"],
                combined_evidence_items=[],
                new_evidence_ids=[],
                reused_evidence_ids=[],
            )
        )

        assert first_items
        assert first_reused_ids == []
        assert [str(item.get("id") or "") for item in second_items] == [
            str(item.get("id") or "") for item in first_items
        ]
        assert second_new_ids == []
        assert second_reused_ids == first_new_ids

    def test_not_invoked_empty(self) -> None:
        o, _ = eval_attempt_flow._adaptive_helper_outcome_for_attempt(
            helper_invoked=False,
            helper_materialized_context=False,
            helper_decision_reason="",
            stage_a_meta=None,
        )
        assert o == "not_invoked"

    def test_skipped_policy(self) -> None:
        o, _ = eval_attempt_flow._adaptive_helper_outcome_for_attempt(
            helper_invoked=False,
            helper_materialized_context=False,
            helper_decision_reason="online_policy_skip_helper",
            stage_a_meta=None,
        )
        assert o == "skipped_policy"


# ---------------------------------------------------------------------------
# _request_context_failure_kind_from_meta
# ---------------------------------------------------------------------------


class TestRequestContextFailureKind:
    def test_none_meta(self) -> None:
        assert eval_attempt_flow._request_context_failure_kind_from_meta(None) == ""

    def test_explicit_kind(self) -> None:
        assert (
            eval_attempt_flow._request_context_failure_kind_from_meta(
                {"request_context_failure_kind": "transport"}
            )
            == "transport"
        )

    def test_request_budget_exceeded(self) -> None:
        assert (
            eval_attempt_flow._request_context_failure_kind_from_meta(
                {"error": "request_budget_exceeded"}
            )
            == "request_budget_exceeded"
        )

    def test_backend_unavailable(self) -> None:
        assert (
            eval_attempt_flow._request_context_failure_kind_from_meta(
                {"request_context_backend_unavailable": True}
            )
            == "backend_unavailable"
        )

    def test_compat_reasoning_only(self) -> None:
        assert (
            eval_attempt_flow._request_context_failure_kind_from_meta(
                {"error": "openai_compat_reasoning_only"}
            )
            == "compat_reasoning_only"
        )

    def test_invalid_payload(self) -> None:
        assert (
            eval_attempt_flow._request_context_failure_kind_from_meta(
                {"error": "invalid_request_context_payload"}
            )
            == "invalid_payload"
        )

    def test_request_failed(self) -> None:
        assert (
            eval_attempt_flow._request_context_failure_kind_from_meta({"error": "request_failed"})
            == "transport"
        )

    def test_http_502(self) -> None:
        assert (
            eval_attempt_flow._request_context_failure_kind_from_meta(
                {"error": "http_error", "status": 502}
            )
            == "backend_unavailable"
        )

    def test_http_400(self) -> None:
        assert (
            eval_attempt_flow._request_context_failure_kind_from_meta(
                {"error": "http_error", "status": 400}
            )
            == "transport"
        )

    def test_empty(self) -> None:
        assert eval_attempt_flow._request_context_failure_kind_from_meta({}) == ""


# ---------------------------------------------------------------------------
# _request_context_recovery_failure_kind_from_meta
# ---------------------------------------------------------------------------


class TestRequestContextRecoveryFailureKind:
    def test_none(self) -> None:
        assert eval_attempt_flow._request_context_recovery_failure_kind_from_meta(None) == ""

    def test_request_failed(self) -> None:
        assert (
            eval_attempt_flow._request_context_recovery_failure_kind_from_meta(
                {"error": "request_failed"}
            )
            == "transport"
        )

    def test_http_503(self) -> None:
        assert (
            eval_attempt_flow._request_context_recovery_failure_kind_from_meta(
                {"error": "http_error", "status": 503}
            )
            == "backend_unavailable"
        )

    def test_http_429(self) -> None:
        assert (
            eval_attempt_flow._request_context_recovery_failure_kind_from_meta(
                {"error": "http_error", "status": 429}
            )
            == "transport"
        )

    def test_empty(self) -> None:
        assert eval_attempt_flow._request_context_recovery_failure_kind_from_meta({}) == ""


# ---------------------------------------------------------------------------
# _coerce_context_request_payload
# ---------------------------------------------------------------------------


class TestCoerceContextRequestPayload:
    def test_non_dict(self) -> None:
        assert eval_attempt_flow._coerce_context_request_payload("not_dict") is None

    def test_requests_non_list(self) -> None:
        assert eval_attempt_flow._coerce_context_request_payload({"requests": "bad"}) is None

    def test_file_request(self) -> None:
        r = eval_attempt_flow._coerce_context_request_payload(
            {"need_more_context": True, "requests": [{"type": "file", "path": "f.py"}]}
        )
        assert r is not None and len(r["requests"]) == 1

    def test_frame_needs_int_index(self) -> None:
        r = eval_attempt_flow._coerce_context_request_payload(
            {"requests": [{"type": "frame", "index": "bad"}]}
        )
        assert r is not None and len(r["requests"]) == 0

    def test_frame_rejects_bool(self) -> None:
        r = eval_attempt_flow._coerce_context_request_payload(
            {"requests": [{"type": "frame", "index": True}]}
        )
        assert r is not None and len(r["requests"]) == 0

    def test_local_var(self) -> None:
        r = eval_attempt_flow._coerce_context_request_payload(
            {"requests": [{"type": "local_var", "frame": 0, "name": "x"}]}
        )
        assert r is not None and len(r["requests"]) == 1

    def test_local_var_missing_name(self) -> None:
        r = eval_attempt_flow._coerce_context_request_payload(
            {"requests": [{"type": "local_var", "frame": 0}]}
        )
        assert r is not None and len(r["requests"]) == 0

    def test_optional_fields(self) -> None:
        r = eval_attempt_flow._coerce_context_request_payload(
            {
                "snapshot_justification": "need stack",
                "diagnosis": "off by one",
                "missing_fact": None,
                "sufficiency_rationale": "enough",
                "compact_context_reason": "too much",
            }
        )
        assert r is not None
        assert r["diagnosis"] == "off by one"
        assert r["missing_fact"] is None

    def test_missing_fact_string(self) -> None:
        r = eval_attempt_flow._coerce_context_request_payload({"missing_fact": "scope"})
        assert r is not None and r["missing_fact"] == "scope"

    def test_non_dict_entry_filtered(self) -> None:
        r = eval_attempt_flow._coerce_context_request_payload(
            {"requests": ["bad", {"type": "file", "path": "f.py"}]}
        )
        assert r is not None and len(r["requests"]) == 1

    def test_no_type_filtered(self) -> None:
        r = eval_attempt_flow._coerce_context_request_payload({"requests": [{"path": "f.py"}]})
        assert r is not None and len(r["requests"]) == 0


# ---------------------------------------------------------------------------
# _filter_context_request_payload_for_action
# ---------------------------------------------------------------------------


class TestFilterContextRequestPayloadForAction:
    def test_none_returns_none(self) -> None:
        assert (
            eval_attempt_flow._filter_context_request_payload_for_action(
                action="invoke_helper", request_payload=None
            )
            is None
        )

    def test_unrestricted_returns_as_is(self) -> None:
        p = {"need_more_context": True, "requests": [{"type": "file", "path": "f.py"}]}
        assert (
            eval_attempt_flow._filter_context_request_payload_for_action(
                action="invoke_helper", request_payload=p
            )
            is p
        )


# ---------------------------------------------------------------------------
# _llm_call_count_from_meta / _aggregate_attempt_llm_usage
# ---------------------------------------------------------------------------


class TestLlmUsage:
    def test_count_none(self) -> None:
        assert eval_attempt_flow._llm_call_count_from_meta(None) == 0

    def test_count_value(self) -> None:
        assert eval_attempt_flow._llm_call_count_from_meta({"llm_call_count": 3}) == 3

    def test_aggregate_both_none(self) -> None:
        c, _p, _co, _t = eval_attempt_flow._aggregate_attempt_llm_usage(
            stage_a_meta=None,
            patch_meta=None,
            stage_a_default_llm_calls=0,
            patch_default_llm_calls=0,
        )
        assert c == 0

    def test_aggregate_with_data(self) -> None:
        c, p, _co, _t = eval_attempt_flow._aggregate_attempt_llm_usage(
            stage_a_meta={
                "llm_call_count": 1,
                "tokens_prompt": 50,
                "tokens_completion": 30,
                "tokens_total": 80,
            },
            patch_meta={
                "llm_call_count": 2,
                "tokens_prompt": 100,
                "tokens_completion": 60,
                "tokens_total": 160,
            },
            stage_a_default_llm_calls=1,
            patch_default_llm_calls=1,
        )
        assert c == 3 and p == 150

    def test_aggregate_defaults(self) -> None:
        c, _, _, _ = eval_attempt_flow._aggregate_attempt_llm_usage(
            stage_a_meta={}, patch_meta={}, stage_a_default_llm_calls=1, patch_default_llm_calls=1
        )
        assert c == 2


# ---------------------------------------------------------------------------
# _append_retry_history_turn_if_enabled
# ---------------------------------------------------------------------------


class TestAppendRetryHistoryTurn:
    def test_history_mode(self) -> None:
        turns: list[str] = []
        eval_attempt_flow._append_retry_history_turn_if_enabled(
            retry_context_mode="history",
            retry_history_turns=turns,
            attempt=1,
            note="tests_still_failing",
            failure_signature="sig-1",
            patch_meta={"status": 200},
        )
        assert len(turns) == 1

    def test_non_history_skips(self) -> None:
        turns: list[str] = []
        eval_attempt_flow._append_retry_history_turn_if_enabled(
            retry_context_mode="delta",
            retry_history_turns=turns,
            attempt=1,
            note="tests_still_failing",
            failure_signature="sig-1",
            patch_meta={},
        )
        assert len(turns) == 0
