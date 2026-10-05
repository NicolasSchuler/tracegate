from __future__ import annotations

import pytest

from baselines.controller_packet import (
    AdaptiveControllerStateV2,
    ControllerPacketV1,
    EvidenceBundleV1,
    HistoryBundleV1,
    HistoryLessonV1,
    SolverHintsV1,
    VerifierReportV1,
    parse_controller_packet,
)
from evals.controller_summarizer import (
    ControllerStrategySummaryResult,
    _truncate,
    build_controller_strategy_summary,
)

# ---------------------------------------------------------------------------
# _truncate helper
# ---------------------------------------------------------------------------


class TestTruncate:
    def test_empty_string(self) -> None:
        assert _truncate("", max_chars=10) == ""

    def test_short_string_unchanged(self) -> None:
        assert _truncate("hello", max_chars=10) == "hello"

    def test_long_string_truncated_with_ellipsis(self) -> None:
        result = _truncate("hello world", max_chars=8)
        assert result.endswith("...")
        assert len(result) <= 8

    def test_max_chars_le_3_no_ellipsis(self) -> None:
        result = _truncate("hello world", max_chars=3)
        assert len(result) <= 3
        assert "..." not in result

    def test_whitespace_stripped(self) -> None:
        assert _truncate("  hi  ", max_chars=10) == "hi"


# ---------------------------------------------------------------------------
# build_controller_strategy_summary — full integration
# ---------------------------------------------------------------------------


def test_build_controller_strategy_summary_uses_crash_verifier_history_and_plan() -> None:
    packet = parse_controller_packet(
        {
            "schema_version": "v1",
            "case_id": "adaptive_case",
            "condition": "layered_adaptive_llm",
            "attempt": 2,
            "controller_state": {
                "helper_action_executed": "compact_then_patch",
                "helper_decision_reason": "policy_action_compact_then_patch",
            },
            "evidence_bundle": {
                "summary": {"crash_anchor": "buggy.py:4 in f"},
                "snapshot_reduction": {
                    "crash_frame_summary": "buggy.py:4 in f() -> return helper.value()"
                },
            },
            "history_bundle": {
                "lessons": [
                    {
                        "source": "rca_history",
                        "attempt": 1,
                        "lesson": "previous patch missed the crash file",
                    }
                ]
            },
            "verifier_report": {
                "status": "previous_verifier_rejected",
                "reasons": ["patch missed crash file"],
            },
            "solver_hints": {"retry_plan": "Touch buggy.py only and preserve the signature."},
        }
    )

    assert packet is not None
    result = build_controller_strategy_summary(packet)

    assert result.sources == [
        "crash_anchor",
        "verifier_report",
        "history_lessons",
        "retry_plan",
    ]
    assert "buggy.py:4 in f" in result.summary
    assert "patch missed crash file" in result.summary
    assert "Touch buggy.py only" in result.summary


# ---------------------------------------------------------------------------
# Empty / minimal inputs
# ---------------------------------------------------------------------------


def test_empty_input_no_clauses() -> None:
    """A packet with all empty/None fields produces an empty summary."""
    packet = parse_controller_packet(
        {
            "schema_version": "v1",
            "case_id": "c1",
            "condition": "base_repair",
            "attempt": 1,
            "controller_state": {},
            "evidence_bundle": {},
            "history_bundle": {},
            "verifier_report": {},
            "solver_hints": {},
        }
    )
    assert packet is not None
    result = build_controller_strategy_summary(packet)
    assert result.summary == ""
    assert result.sources == []


# ---------------------------------------------------------------------------
# Crash anchor from snapshot_reduction fallback (line 44-45)
# ---------------------------------------------------------------------------


def test_crash_anchor_from_snapshot_reduction_fallback() -> None:
    """When evidence summary crash_anchor is empty, fall back to snapshot_reduction."""
    packet = parse_controller_packet(
        {
            "schema_version": "v1",
            "case_id": "c1",
            "condition": "base_repair",
            "attempt": 1,
            "controller_state": {},
            "evidence_bundle": {
                "summary": {"crash_anchor": ""},
                "snapshot_reduction": {"crash_frame_summary": "file.py:10 in main"},
            },
            "history_bundle": {},
            "verifier_report": {},
            "solver_hints": {},
        }
    )
    assert packet is not None
    result = build_controller_strategy_summary(packet)
    assert "file.py:10 in main" in result.summary
    assert "crash_anchor" in result.sources


# ---------------------------------------------------------------------------
# Verifier with empty reasons (line 55 — fallback clause)
# ---------------------------------------------------------------------------


def test_verifier_with_no_reasons_uses_status() -> None:
    """When verifier has a status but no reasons, use generic constraint clause."""
    packet = parse_controller_packet(
        {
            "schema_version": "v1",
            "case_id": "c1",
            "condition": "base_repair",
            "attempt": 1,
            "controller_state": {},
            "evidence_bundle": {},
            "history_bundle": {},
            "verifier_report": {"status": "rejected", "reasons": []},
            "solver_hints": {},
        }
    )
    assert packet is not None
    result = build_controller_strategy_summary(packet)
    assert "Respect the verifier constraint: rejected." in result.summary
    assert "verifier_report" in result.sources


# ---------------------------------------------------------------------------
# History: previous_note fallback (lines 63-65)
# ---------------------------------------------------------------------------


def test_history_previous_note_fallback() -> None:
    """When no lessons but previous_note is set, use that."""
    packet = parse_controller_packet(
        {
            "schema_version": "v1",
            "case_id": "c1",
            "condition": "base_repair",
            "attempt": 2,
            "controller_state": {},
            "evidence_bundle": {},
            "history_bundle": {
                "lessons": [],
                "previous_note": "tests_still_failing",
            },
            "verifier_report": {},
            "solver_hints": {},
        }
    )
    assert packet is not None
    result = build_controller_strategy_summary(packet)
    assert "Previous attempt outcome: tests_still_failing." in result.summary
    assert "previous_note" in result.sources


# ---------------------------------------------------------------------------
# History with empty lesson text (line 60-61 branch: lesson is empty)
# ---------------------------------------------------------------------------


def test_history_lesson_with_empty_text_skips() -> None:
    """A lesson with blank text should not produce a history clause."""
    packet = parse_controller_packet(
        {
            "schema_version": "v1",
            "case_id": "c1",
            "condition": "base_repair",
            "attempt": 2,
            "controller_state": {},
            "evidence_bundle": {},
            "history_bundle": {
                "lessons": [{"source": "x", "attempt": 1, "lesson": "  "}],
                "previous_note": "",
            },
            "verifier_report": {},
            "solver_hints": {},
        }
    )
    assert packet is not None
    result = build_controller_strategy_summary(packet)
    assert "history_lessons" not in result.sources
    assert "previous_note" not in result.sources


# ---------------------------------------------------------------------------
# Branch coverage: [60→67] — lessons list is non-empty but lesson text is empty
# ---------------------------------------------------------------------------


def test_history_lesson_present_but_empty_text_bypasses_clause() -> None:
    """Branch [60→67]: lessons list is truthy but first lesson.lesson is empty.

    The parser normally filters out empty lessons, so we construct the packet
    directly to exercise this defensive branch.
    """
    # Build a minimal ControllerPacketV1 with a HistoryLessonV1 whose text is ""
    controller_state = AdaptiveControllerStateV2(
        schema_version="v2",
        helper_action_selected="",
        helper_action_executed="",
        helper_decision_reason="",
        helper_outcome="",
        helper_materialized_context=False,
        helper_calls_used=0,
        helper_budget_max_calls=0,
        no_progress_streak=0,
        retry_prompt_mode="",
        retry_context_mode="",
        context_compaction_requested=False,
        context_compaction_invoked=False,
        context_compaction_success=False,
        context_compaction_trigger="",
        prompt_budget={},
    )
    evidence_bundle = EvidenceBundleV1(
        schema_version="v1",
        case_id="c1",
        condition="base_repair",
        attempt=1,
        failure_signature="",
        selected_ids=[],
        new_ids=[],
        reused_ids=[],
        ast_context=None,
        snapshot_reduction=None,
        summary=None,
        items=[],
    )
    history_bundle = HistoryBundleV1(
        schema_version="v1",
        retry_context_mode="",
        attempt_turns=[],
        retry_history_text="",
        retry_history_turns_included=0,
        retry_history_truncated=False,
        previous_note="",
        previous_failure_signature="",
        previous_patch_verifier_rejected=False,
        previous_patch_verifier_reasons=[],
        lessons=[HistoryLessonV1(source="test", attempt=1, lesson="")],
        repo_history=None,
        persistent_session_id="",
        persistent_records_loaded=0,
    )
    verifier_report = VerifierReportV1(
        schema_version="v1",
        status="",
        scope="",
        reasons=[],
        previous_timed_out=False,
        previous_timeout_sec=None,
        syntax_cycle=0,
        syntax_max_cycles=0,
        last_error_type="",
        last_error_msg="",
        retry_feedback="",
    )
    solver_hints = SolverHintsV1(
        tool_routes=[],
        retry_plan="",
        recommended_focus=[],
        decomposition=[],
    )
    packet = ControllerPacketV1(
        schema_version="v1",
        case_id="c1",
        condition="base_repair",
        attempt=1,
        controller_state=controller_state,
        evidence_bundle=evidence_bundle,
        history_bundle=history_bundle,
        verifier_report=verifier_report,
        solver_hints=solver_hints,
    )
    result = build_controller_strategy_summary(packet)
    # lessons is non-empty but lesson text is empty → no clause added
    assert "history_lessons" not in result.sources
    assert "Carry forward" not in result.summary
