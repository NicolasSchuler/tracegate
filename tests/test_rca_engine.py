"""Tests for the RCA engine attempt-record builder and state transitions."""

from __future__ import annotations

from llmdebug.rca_engine import build_rca_attempt_record
from llmdebug.rca_types import RCAAttemptRequest


def test_build_rca_attempt_record_creates_initial_attempt() -> None:
    record = build_rca_attempt_record(
        previous=None,
        attempt_id="att-1",
        request=RCAAttemptRequest(
            session_id="sess-1",
            action="hypothesize",
            failure_signature="sig-1",
            evidence_refs=("exception:AssertionError",),
            hypothesis="off-by-one",
            experiment_plan="add boundary check test",
            gate_mode="soft",
            exploratory=False,
        ),
        now="2026-03-01T12:00:00Z",
    )
    assert record["attempt_id"] == "att-1"
    assert record["session_id"] == "sess-1"
    assert record["failure_signature"] == "sig-1"
    assert record["repeat_count"] == 1
    assert record["state"] == "hypothesized"
    assert record["created_at"] == "2026-03-01T12:00:00Z"
    assert record["updated_at"] == "2026-03-01T12:00:00Z"
    assert record["gate_feedback"]["allowed"] is True
    assert record["gate_feedback"]["mode"] == "soft"
    assert record["next_required"] == ["Write a minimal experiment or patch plan."]
    assert isinstance(record["allowed_actions"], list)
    assert record["allowed_actions"]


def test_build_rca_attempt_record_merges_previous_context_and_repeats() -> None:
    previous = {
        "attempt_id": "att-0",
        "session_id": "sess-1",
        "state": "hypothesized",
        "failure_signature": "sig-1",
        "repeat_count": 1,
        "evidence_refs": ["exception:AssertionError"],
        "hypothesis": "off-by-one",
        "hypothesis_delta": "new",
        "experiment_plan": "add boundary check test",
        "proposed_change_summary": None,
        "verification_result": None,
        "gate_feedback": {},
        "gate_warnings": [],
        "allowed_actions": ["plan", "verify"],
        "next_required": ["verification_result"],
        "exploratory": False,
        "transition_history": [],
        "created_at": "2026-03-01T11:59:00Z",
        "updated_at": "2026-03-01T11:59:00Z",
    }
    record = build_rca_attempt_record(
        previous=previous,
        attempt_id="att-2",
        request=RCAAttemptRequest(
            session_id="sess-1",
            action="plan",
            failure_signature="sig-1",
            evidence_refs=("crash_frame:buggy.py:12:f",),
            gate_mode="soft",
            exploratory=True,
        ),
        now="2026-03-01T12:01:00Z",
    )
    assert record["repeat_count"] == 2
    assert record["created_at"] == "2026-03-01T11:59:00Z"
    assert record["updated_at"] == "2026-03-01T12:01:00Z"
    assert record["exploratory"] is True
    assert record["experiment_plan"] == "add boundary check test"
    assert "exception:AssertionError" in record["evidence_refs"]
    assert "crash_frame:buggy.py:12:f" in record["evidence_refs"]
    assert record["transition_history"]


def test_build_rca_attempt_record_applies_budget_after_current_attempt() -> None:
    previous = {
        "attempt_id": "att-1",
        "session_id": "sess-1",
        "state": "changed",
        "failure_signature": "sig-1",
        "repeat_count": 1,
        "evidence_refs": ["crash_frame:buggy.py:12:f"],
        "hypothesis": "off-by-one",
        "hypothesis_delta": "unchanged",
        "experiment_plan": "add boundary check test",
        "proposed_change_summary": "try inclusive upper bound",
        "verification_result": None,
        "gate_feedback": {},
        "gate_warnings": [],
        "allowed_actions": ["verify", "change", "bind_evidence"],
        "next_required": ["Run verification checks and capture result."],
        "exploratory": False,
        "exploration_budget_remaining": 2,
        "transition_history": [],
        "created_at": "2026-03-01T11:59:00Z",
        "updated_at": "2026-03-01T11:59:00Z",
    }
    record = build_rca_attempt_record(
        previous=previous,
        attempt_id="att-2",
        request=RCAAttemptRequest(
            session_id="sess-1",
            action="change",
            failure_signature="sig-1",
            gate_mode="soft",
            exploratory=False,
        ),
        now="2026-03-01T12:01:00Z",
    )
    assert record["repeat_count"] == 2
    assert record["exploration_budget_remaining"] == 1
    assert record["gate_feedback"]["severity"] == "info"
    assert "loop_protection" not in record["gate_feedback"]["triggered_checks"]


def test_build_rca_attempt_record_strict_gate_retains_previous_state() -> None:
    previous = {
        "attempt_id": "att-0",
        "session_id": "sess-1",
        "state": "captured",
        "failure_signature": "sig-1",
        "repeat_count": 1,
        "evidence_refs": [],
        "hypothesis": "",
        "hypothesis_delta": "unknown",
        "experiment_plan": "",
        "proposed_change_summary": None,
        "verification_result": None,
        "gate_feedback": {},
        "gate_warnings": [],
        "allowed_actions": ["bind_evidence", "capture"],
        "next_required": ["Bind concrete evidence from traceback/frames."],
        "exploratory": False,
        "transition_history": [],
        "created_at": "2026-03-01T11:59:00Z",
        "updated_at": "2026-03-01T11:59:00Z",
    }
    record = build_rca_attempt_record(
        previous=previous,
        attempt_id="att-1",
        request=RCAAttemptRequest(
            session_id="sess-1",
            action="plan",
            failure_signature="sig-1",
            evidence_refs=(),
            hypothesis="",
            experiment_plan="",
            gate_mode="strict",
            exploratory=False,
        ),
        now="2026-03-01T12:01:00Z",
    )
    assert record["state"] == "captured"
    assert record["gate_feedback"]["allowed"] is False
    # RCA-06: "plan" isn't a transition from "captured" either.
    assert set(record["gate_feedback"]["triggered_checks"]) == {
        "evidence_required",
        "hypothesis_required",
        "illegal_transition",
    }


def test_rca_state_literal_is_defined_once() -> None:
    """RCA-22: gates and rca_transitions re-export the one definition in rca_types."""
    from llmdebug import gates, rca_transitions, rca_types

    assert gates.RCAState is rca_types.RCAState
    assert rca_transitions.RCAState is rca_types.RCAState
    assert rca_transitions.RCAAction is rca_types.RCAAction


def test_attempt_record_matches_its_typed_dict() -> None:
    """RCA-22: the engine's record and RCAAttemptRecord can't drift apart silently."""
    from llmdebug.rca_types import RCAAttemptRecord

    record = build_rca_attempt_record(
        previous=None,
        attempt_id="att-1",
        request=RCAAttemptRequest(session_id="s", action="capture", failure_signature="sig"),
        now="2026-03-01T12:00:00Z",
    )

    assert set(record) == set(RCAAttemptRecord.__annotations__)


def test_previous_values_are_named() -> None:
    from llmdebug.rca_transitions import resolve_previous_values

    values = resolve_previous_values(
        {"state": "planned", "hypothesis": "h", "evidence_refs": ["r"]}
    )

    assert values.state == "planned"
    assert values.hypothesis == "h"
    assert values.evidence_refs == ["r"]
    assert values.verification_result is None
    assert resolve_previous_values(None).state == "captured"
