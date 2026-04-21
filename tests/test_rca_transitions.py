"""Tests for RCA transition helpers."""

from __future__ import annotations

import pytest

from llmdebug.rca_transitions import (
    apply_stall_rule,
    build_transition_history,
    classify_hypothesis_delta,
    merge_evidence_refs,
    normalize_action,
)


def test_normalize_action_unknown_is_rejected() -> None:
    with pytest.raises(ValueError, match="Invalid RCA action"):
        normalize_action("NotARealAction")


def test_classify_hypothesis_delta_variants() -> None:
    assert classify_hypothesis_delta("", "") == "unknown"
    assert classify_hypothesis_delta("x", "x") == "unchanged"
    assert classify_hypothesis_delta("x", "x and y") == "refined"
    assert classify_hypothesis_delta("", "new") == "added"
    assert classify_hypothesis_delta("old", "") == "removed"
    assert classify_hypothesis_delta("old", "different") == "replaced"


def test_merge_evidence_refs_deduplicates_and_caps() -> None:
    merged = merge_evidence_refs(
        ["a", "b", "a", ""],
        ["b", "c", "d", "e"],
        max_items=4,
    )
    assert merged == ["a", "b", "c", "d"]


def test_apply_stall_rule_marks_state_when_no_progress() -> None:
    state, warnings = apply_stall_rule(
        target_state="verified",
        gate_warnings=[],
        same_signature=True,
        repeat_count=3,
        stall_threshold=3,
        hypothesis_advanced=False,
        plan_advanced=False,
        normalized_action="bind_evidence",
    )
    assert state == "stalled"
    assert any("stalled" in w.lower() for w in warnings)


def test_build_transition_history_truncates_to_last_32() -> None:
    previous_history = [
        {"at": f"t{i}", "action": "bind_evidence", "from": "captured", "to": "captured"}
        for i in range(40)
    ]
    history = build_transition_history(
        previous={"transition_history": previous_history},
        now="tnow",
        normalized_action="verify",
        previous_state="changed",
        target_state="verified",
        gate_allowed=True,
        gate_mode="soft",
    )

    assert len(history) == 32
    assert history[-1]["at"] == "tnow"
    assert history[-1]["action"] == "verify"
