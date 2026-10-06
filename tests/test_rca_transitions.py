"""Tests for RCA transition helpers."""

from __future__ import annotations

import pytest

from llmdebug.rca_transitions import (
    AttemptProgress,
    TransitionInputs,
    TransitionStep,
    apply_stall_rule,
    build_transition_history,
    classify_hypothesis_delta,
    derive_repair_streak,
    merge_evidence_refs,
    normalize_action,
    verification_passed,
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
    progress = AttemptProgress(
        same_signature=True,
        repeat_count=3,
        hypothesis_delta="unchanged",
        plan_advanced=False,
        hypothesis_advanced=False,
    )
    state, warnings = apply_stall_rule(
        TransitionInputs(
            action="bind_evidence", progress=progress, stall_threshold=3, repair_streak=3
        ),
        target_state="verified",
        gate_warnings=[],
    )
    assert state == "stalled"
    assert any("stalled" in w.lower() for w in warnings)


def test_build_transition_history_truncates_to_last_32() -> None:
    previous_history = [
        {"at": f"t{i}", "action": "bind_evidence", "from": "captured", "to": "captured"}
        for i in range(40)
    ]
    history = build_transition_history(
        {"transition_history": previous_history},
        now="tnow",
        step=TransitionStep(
            action="verify",
            from_state="changed",
            to_state="verified",
            gate_allowed=True,
            gate_mode="soft",
        ),
    )

    assert len(history) == 32
    assert history[-1]["at"] == "tnow"
    assert history[-1]["action"] == "verify"


def _progress(**overrides: object) -> AttemptProgress:
    base: dict[str, object] = {
        "same_signature": True,
        "repeat_count": 4,
        "hypothesis_delta": "unchanged",
        "plan_advanced": False,
        "hypothesis_advanced": False,
    }
    base.update(overrides)
    return AttemptProgress(**base)  # type: ignore[arg-type]


def test_derive_repair_streak_counts_only_unproductive_repairs() -> None:
    assert (
        derive_repair_streak(
            previous_streak=2,
            action="change",
            progress=_progress(),
            evidence_advanced=False,
            verification_is_passing=False,
        )
        == 3
    )
    # A plain workflow step neither trips nor clears loop protection.
    assert (
        derive_repair_streak(
            previous_streak=2,
            action="plan",
            progress=_progress(),
            evidence_advanced=False,
            verification_is_passing=False,
        )
        == 2
    )


@pytest.mark.parametrize(
    ("kwargs", "progress_overrides"),
    [
        ({}, {"same_signature": False}),
        ({}, {"hypothesis_advanced": True}),
        ({}, {"plan_advanced": True}),
        ({"evidence_advanced": True}, {}),
        ({"verification_is_passing": True}, {}),
    ],
)
def test_derive_repair_streak_resets_on_progress(
    kwargs: dict[str, bool], progress_overrides: dict[str, object]
) -> None:
    call: dict[str, bool] = {"evidence_advanced": False, "verification_is_passing": False}
    call.update(kwargs)
    assert (
        derive_repair_streak(
            previous_streak=5,
            action="change",
            progress=_progress(**progress_overrides),
            **call,
        )
        == 0
    )


def test_apply_stall_rule_skips_denied_actions() -> None:
    state, warnings = apply_stall_rule(
        TransitionInputs(
            action="verify",
            progress=_progress(),
            stall_threshold=2,
            repair_streak=5,
            gate_allowed=False,
        ),
        target_state="evidence_bound",
        gate_warnings=[],
    )
    assert state == "evidence_bound"
    assert warnings == []


def test_verification_passed_requires_a_pass_status() -> None:
    assert verification_passed({"status": "pass"}) is True
    assert verification_passed({"status": "fail"}) is False
    assert verification_passed(None) is False
    assert verification_passed("pass") is False
