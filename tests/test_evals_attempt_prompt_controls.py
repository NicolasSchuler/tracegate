"""Tests for Stage-B reveal-policy helpers in eval_attempt_flow."""

from __future__ import annotations

from evals.eval_attempt_flow import _resolve_stage_b_reveal_policy


def test_traceback_first_reveal_starts_with_snapshot_index_only() -> None:
    result = _resolve_stage_b_reveal_policy(
        mode="traceback_first",
        is_adaptive_condition=True,
        snapshot_available=True,
        stage_a_context="",
        stage_a_need_snapshot=False,
        failure_signature_repeat=False,
        no_progress_streak=0,
    )

    assert result["applied"] is True
    assert result["include_snapshot"] is False
    assert result["include_stage_a_context"] is False
    assert result["snapshot_lite_needed"] is True
    assert result["trigger"] == "traceback_only"


def test_traceback_first_reveal_promotes_snapshot_and_stage_a_context_on_triggers() -> None:
    result = _resolve_stage_b_reveal_policy(
        mode="traceback_first",
        is_adaptive_condition=True,
        snapshot_available=True,
        stage_a_context="--- buggy.py ---\nvalue = 1\n",
        stage_a_need_snapshot=True,
        failure_signature_repeat=True,
        no_progress_streak=2,
    )

    assert result["include_snapshot"] is True
    assert result["include_stage_a_context"] is True
    assert result["snapshot_lite_needed"] is False
    assert result["triggers"] == [
        "stage_a_context",
        "need_snapshot",
        "failure_signature_repeat",
        "no_progress_escalation",
    ]
