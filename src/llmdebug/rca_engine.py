"""Pure RCA domain engine.

This module computes RCA attempt records without performing persistence I/O.
"""

from __future__ import annotations

from typing import Any

from .gates import GateRequest, evaluate_gate, normalize_gate_mode
from .rca_transitions import (
    TransitionInputs,
    TransitionStep,
    allowed_actions_for_state,
    apply_stall_rule,
    build_transition_history,
    carry_forward_verification,
    default_next_required,
    derive_attempt_progress,
    derive_repair_streak,
    normalize_action,
    resolve_attempt_values,
    resolve_merged_evidence_refs,
    resolve_previous_values,
    resolve_target_state,
    verification_passed,
)
from .rca_types import RCAAttemptRecord, RCAAttemptRequest


def build_rca_attempt_record(
    *,
    previous: dict[str, Any] | None,
    attempt_id: str,
    request: RCAAttemptRequest,
    now: str,
) -> RCAAttemptRecord:
    """Build a complete RCA attempt record by advancing the state machine.

    This is the core pure-function entry point for the RCA domain engine.
    Given the previous attempt record (if any) and the requested action, it
    resolves carry-forward values, evaluates gate checks, computes the target
    state (including stall detection), and returns a fully-populated attempt
    dict ready for persistence.

    The function is side-effect-free: all I/O (storage, networking) is the
    caller's responsibility.

    Args:
        previous: The preceding attempt record dict, or ``None`` for the
            first attempt in a session.  Carry-forward fields (hypothesis,
            plan, evidence refs, etc.) are extracted from this record.
        attempt_id: Unique identifier for the new attempt.
        request: Normalized :class:`RCAAttemptRequest` carrying session
            identity, action, failure signature, evidence refs, optional
            hypothesis/plan/change summary, verification status, and gate
            settings for the current attempt.
        now: ISO-8601 timestamp string used as ``updated_at`` (and
            ``created_at`` for the first attempt).

    Returns:
        A dict representing the new attempt record with the following keys:

        - **attempt_id** / **session_id**: Echo of the input identifiers.
        - **state**: The resolved target :data:`RCAState` after this action.
        - **failure_signature** / **repeat_count**: Signature tracking.
        - **repair_streak**: Consecutive unproductive repair attempts on this
          signature; drives stall detection.
        - **evidence_refs**: Merged, deduplicated evidence references (at most
          16; earlier references win).
        - **hypothesis** / **hypothesis_delta**: Current hypothesis text
          and whether it changed from the previous attempt.
        - **experiment_plan** / **proposed_change_summary**: Current plan
          and change summary.
        - **verification_result**: Current verification outcome dict.
        - **gate_feedback**: Serialized :class:`GateFeedback` dict.
        - **gate_warnings**: List of warning strings (includes stall
          warnings when applicable).
        - **allowed_actions**: Actions valid from the new state.
        - **next_required**: Suggested next steps from the gate or state.
        - **exploratory** / **exploration_budget_remaining**: Exploration
          tracking.
        - **transition_history**: Chronological list of state transitions,
          capped at the 32 most recent.
        - **created_at** / **updated_at**: Timestamps.

    Raises:
        ValueError: If ``request.action`` is not a valid RCA action.
    """
    normalized_mode = normalize_gate_mode(request.gate_mode)
    normalized_action = normalize_action(request.action)
    (
        previous_state,
        previous_hypothesis,
        previous_plan,
        previous_refs,
        previous_verification,
        previous_change_summary,
    ) = resolve_previous_values(previous)

    stored_refs = resolve_merged_evidence_refs(previous_refs, [])
    merged_refs = resolve_merged_evidence_refs(previous_refs, list(request.evidence_refs))
    evidence_advanced = len(merged_refs) > len(stored_refs)
    hypothesis_value, plan_value, proposed_change_value = resolve_attempt_values(
        previous_hypothesis=previous_hypothesis,
        previous_plan=previous_plan,
        previous_change_summary=previous_change_summary,
        hypothesis=request.hypothesis,
        experiment_plan=request.experiment_plan,
        proposed_change_summary=request.proposed_change_summary,
    )
    progress = derive_attempt_progress(
        previous=previous,
        failure_signature=request.failure_signature,
        previous_hypothesis=previous_hypothesis,
        hypothesis_value=hypothesis_value,
        previous_plan=previous_plan,
        plan_value=plan_value,
    )
    same_signature, repeat_count, hypothesis_delta, _, hypothesis_advanced = progress
    verification_value = carry_forward_verification(
        previous_verification,
        request.verification_result,
        same_signature=same_signature,
        action=normalized_action,
    )

    # Evaluate the current attempt against the previously available budget,
    # then consume one unit if we repeated the same signature without advancing.
    budget_before_attempt = (
        int(previous.get("exploration_budget_remaining", request.exploration_budget))
        if previous
        else request.exploration_budget
    )
    if same_signature and not hypothesis_advanced:
        exploration_budget_remaining = max(budget_before_attempt - 1, 0)
    else:
        exploration_budget_remaining = request.exploration_budget

    if same_signature and not hypothesis_advanced:
        gate_budget_remaining = budget_before_attempt
    else:
        gate_budget_remaining = request.exploration_budget

    gate_payload = {
        "evidence_refs": merged_refs,
        "hypothesis": hypothesis_value,
        "experiment_plan": plan_value,
        "verification_result": verification_value,
        "exploratory": request.exploratory,
        "override_reason": request.override_reason or "",
        "hypothesis_advanced": hypothesis_advanced,
    }
    gate = evaluate_gate(
        GateRequest(
            action=normalized_action,
            payload=gate_payload,
            mode=normalized_mode,
            # A session's first attempt has no state to transition from.
            current_state=previous_state if previous else None,
            repeated_signature_count=repeat_count,
            exploration_budget=request.exploration_budget,
            exploration_budget_remaining=gate_budget_remaining,
        )
    )

    target_state = resolve_target_state(previous_state, normalized_action, gate.allowed)
    previous_repair_streak = int(previous.get("repair_streak") or 0) if previous else 0
    # A denied action never ran, so it neither consumes nor clears the repair streak.
    repair_streak = (
        derive_repair_streak(
            previous_streak=previous_repair_streak,
            action=normalized_action,
            progress=progress,
            evidence_advanced=evidence_advanced,
            verification_is_passing=verification_passed(verification_value),
        )
        if gate.allowed
        else previous_repair_streak
    )
    target_state, gate_warnings = apply_stall_rule(
        TransitionInputs(
            action=normalized_action,
            progress=progress,
            stall_threshold=request.stall_threshold,
            repair_streak=repair_streak,
            gate_allowed=gate.allowed,
        ),
        target_state=target_state,
        gate_warnings=list(gate.warnings),
    )
    transition_history = build_transition_history(
        previous,
        now=now,
        step=TransitionStep(
            action=normalized_action,
            from_state=previous_state,
            to_state=target_state,
            gate_allowed=gate.allowed,
            gate_mode=gate.mode,
        ),
    )
    created_at = str(previous.get("created_at") or now) if previous else now
    next_required = (
        list(gate.next_required) if gate.next_required else default_next_required(target_state)
    )

    return {
        "attempt_id": attempt_id,
        "session_id": request.session_id,
        "state": target_state,
        "failure_signature": request.failure_signature,
        "repeat_count": repeat_count,
        "repair_streak": repair_streak,
        "evidence_refs": merged_refs,
        "hypothesis": hypothesis_value or None,
        "hypothesis_delta": hypothesis_delta,
        "experiment_plan": plan_value or None,
        "proposed_change_summary": proposed_change_value or None,
        "verification_result": verification_value,
        "gate_feedback": gate.to_dict(),
        "gate_warnings": gate_warnings,
        "allowed_actions": allowed_actions_for_state(target_state),
        "next_required": next_required,
        "exploratory": request.exploratory,
        "exploration_budget_remaining": exploration_budget_remaining,
        "transition_history": transition_history,
        "created_at": created_at,
        "updated_at": now,
    }
