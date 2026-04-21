"""Pure RCA domain engine.

This module computes RCA attempt records without performing persistence I/O.
"""

from __future__ import annotations

from typing import Any

from .gates import evaluate_gate, normalize_gate_mode
from .rca_transitions import (
    allowed_actions_for_state,
    apply_stall_rule,
    build_transition_history,
    default_next_required,
    derive_attempt_progress,
    normalize_action,
    resolve_attempt_values,
    resolve_merged_evidence_refs,
    resolve_previous_values,
    resolve_target_state,
)
from .rca_types import RCAAttemptRequest


def build_rca_attempt_record(
    *,
    previous: dict[str, Any] | None,
    attempt_id: str,
    request: RCAAttemptRequest,
    now: str,
) -> dict[str, Any]:
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
        - **evidence_refs**: Merged list of all evidence references.
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
        - **transition_history**: Chronological list of state transitions.
        - **created_at** / **updated_at**: Timestamps.
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

    merged_refs = resolve_merged_evidence_refs(previous_refs, list(request.evidence_refs))
    (
        hypothesis_value,
        plan_value,
        verification_value,
        proposed_change_value,
    ) = resolve_attempt_values(
        previous_hypothesis=previous_hypothesis,
        previous_plan=previous_plan,
        previous_change_summary=previous_change_summary,
        previous_verification=previous_verification,
        hypothesis=request.hypothesis,
        experiment_plan=request.experiment_plan,
        proposed_change_summary=request.proposed_change_summary,
        verification_result=request.verification_result,
    )
    same_signature, repeat_count, hypothesis_delta, plan_advanced, hypothesis_advanced = (
        derive_attempt_progress(
            previous=previous,
            failure_signature=request.failure_signature,
            previous_hypothesis=previous_hypothesis,
            hypothesis_value=hypothesis_value,
            previous_plan=previous_plan,
            plan_value=plan_value,
        )
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
        mode=normalized_mode,
        current_state=previous_state,
        action=normalized_action,
        payload=gate_payload,
        repeated_signature_count=repeat_count,
        exploration_budget=request.exploration_budget,
        exploration_budget_remaining=gate_budget_remaining,
    )

    target_state = resolve_target_state(previous_state, normalized_action, gate.allowed)
    target_state, gate_warnings = apply_stall_rule(
        target_state=target_state,
        gate_warnings=list(gate.warnings),
        same_signature=same_signature,
        repeat_count=repeat_count,
        stall_threshold=request.stall_threshold,
        hypothesis_advanced=hypothesis_advanced,
        plan_advanced=plan_advanced,
        normalized_action=normalized_action,
    )
    transition_history = build_transition_history(
        previous=previous,
        now=now,
        normalized_action=normalized_action,
        previous_state=previous_state,
        target_state=target_state,
        gate_allowed=gate.allowed,
        gate_mode=gate.mode,
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
