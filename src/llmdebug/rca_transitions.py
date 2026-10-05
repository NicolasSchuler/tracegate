"""Pure RCA transition and state-evolution helpers."""

from __future__ import annotations

from typing import Any, Literal, cast, get_args

RCAState = Literal[
    "captured",
    "evidence_bound",
    "hypothesized",
    "planned",
    "changed",
    "verified",
    "resolved",
    "stalled",
]
RCAAction = Literal[
    "capture",
    "bind_evidence",
    "hypothesize",
    "plan",
    "change",
    "verify",
    "resolve",
]

DEFAULT_STATE: RCAState = "captured"
_VALID_STATES: frozenset[str] = frozenset(get_args(RCAState))
_VALID_ACTIONS: set[str] = {
    "capture",
    "bind_evidence",
    "hypothesize",
    "plan",
    "change",
    "verify",
    "resolve",
}
_STATE_TRANSITIONS: dict[str, RCAState] = {
    "capture": "captured",
    "bind_evidence": "evidence_bound",
    "hypothesize": "hypothesized",
    "plan": "planned",
    "change": "changed",
    "verify": "verified",
    "resolve": "resolved",
}


def normalize_action(action: str) -> RCAAction:
    """Normalize and validate an RCA action string."""
    lowered = action.strip().lower()
    if lowered in _VALID_ACTIONS:
        return cast(RCAAction, lowered)
    raise ValueError(
        f"Invalid RCA action {action!r}. Valid actions: {', '.join(sorted(_VALID_ACTIONS))}"
    )


def classify_hypothesis_delta(previous: str, current: str) -> str:
    """Classify how the current hypothesis changed vs previous."""
    prev = previous.strip()
    curr = current.strip()
    if not prev and not curr:
        return "unknown"
    if prev == curr:
        return "unchanged"
    if prev and curr and prev in curr:
        return "refined"
    if not prev and curr:
        return "added"
    if prev and not curr:
        return "removed"
    return "replaced"


def allowed_actions_for_state(state: str) -> list[str]:
    """Return sorted allowed action list for display/UI guidance."""
    matrix: dict[str, list[str]] = {
        "captured": ["bind_evidence", "capture"],
        "evidence_bound": ["hypothesize", "bind_evidence"],
        "hypothesized": ["plan", "bind_evidence", "hypothesize"],
        "planned": ["change", "plan", "bind_evidence"],
        "changed": ["verify", "change", "bind_evidence"],
        "verified": ["resolve", "bind_evidence", "plan"],
        "resolved": ["capture", "bind_evidence"],
        "stalled": ["bind_evidence", "hypothesize", "plan"],
    }
    return matrix.get(state, ["bind_evidence"])


def default_next_required(state: str) -> list[str]:
    """Return canonical coaching steps for the given state."""
    steps: dict[str, list[str]] = {
        "captured": ["Bind concrete evidence from traceback/frames."],
        "evidence_bound": ["State a falsifiable root-cause hypothesis."],
        "hypothesized": ["Write a minimal experiment or patch plan."],
        "planned": ["Apply one focused change tied to the plan."],
        "changed": ["Run verification checks and capture result."],
        "verified": ["Resolve if checks passed, else return to evidence."],
        "stalled": ["Gather new evidence or revise hypothesis before retry."],
    }
    return steps.get(state, [])


def merge_evidence_refs(
    previous: list[str],
    incoming: list[str],
    *,
    max_items: int = 16,
) -> list[str]:
    """Merge + deduplicate evidence refs while keeping stable order."""
    merged: list[str] = []
    seen: set[str] = set()
    for value in previous + incoming:
        if not value:
            continue
        item = value.strip()
        if not item or item in seen:
            continue
        seen.add(item)
        merged.append(item)
        if len(merged) >= max_items:
            break
    return merged


def resolve_previous_values(
    previous: dict[str, Any] | None,
) -> tuple[str, str, str, Any, Any, str]:
    """Extract previous attempt values with stable defaults."""
    previous_state = str(previous.get("state")) if previous else DEFAULT_STATE
    previous_hypothesis = str(previous.get("hypothesis") or "") if previous else ""
    previous_plan = str(previous.get("experiment_plan") or "") if previous else ""
    previous_refs: list[Any] = (
        cast(list[Any], previous.get("evidence_refs"))
        if isinstance(previous, dict) and previous.get("evidence_refs")
        else []
    )
    previous_verification = (
        previous.get("verification_result") if isinstance(previous, dict) else None
    )
    previous_change_summary = (
        str(previous.get("proposed_change_summary") or "") if isinstance(previous, dict) else ""
    )
    return (
        previous_state,
        previous_hypothesis,
        previous_plan,
        previous_refs,
        previous_verification,
        previous_change_summary,
    )


def resolve_merged_evidence_refs(
    previous_refs: Any,
    evidence_refs: list[str] | None,
) -> list[str]:
    """Merge previously stored evidence refs with incoming refs."""
    incoming_refs = evidence_refs if isinstance(evidence_refs, list) else []
    return merge_evidence_refs(
        [str(ref).strip() for ref in cast(list[Any], previous_refs)]
        if isinstance(previous_refs, list)
        else [],
        [str(ref).strip() for ref in incoming_refs],
    )


def resolve_attempt_values(
    *,
    previous_hypothesis: str,
    previous_plan: str,
    previous_change_summary: str,
    hypothesis: str | None,
    experiment_plan: str | None,
    proposed_change_summary: str | None,
) -> tuple[str, str, str]:
    """Resolve effective text values (input overrides previous)."""
    hypothesis_value = (hypothesis or "").strip() or previous_hypothesis
    plan_value = (experiment_plan or "").strip() or previous_plan
    proposed_change_value = (proposed_change_summary or "").strip() or previous_change_summary
    return hypothesis_value, plan_value, proposed_change_value


def carry_forward_verification(
    previous_verification: Any,
    verification_result: dict[str, Any] | None,
    *,
    same_signature: bool,
    action: RCAAction,
) -> Any:
    """Return the verification result that applies to this attempt.

    A new result always wins. The previous one describes the failure and the code it ran
    against, so it carries forward only while the failure signature is unchanged and no
    new change is being applied; otherwise a stale ``pass`` would satisfy the strict
    ``resolve`` gate for a different failure (RCA-01).
    """
    if verification_result is not None:
        return verification_result
    if same_signature and action != "change":
        return previous_verification
    return None


def derive_attempt_progress(
    *,
    previous: dict[str, Any] | None,
    failure_signature: str,
    previous_hypothesis: str,
    hypothesis_value: str,
    previous_plan: str,
    plan_value: str,
) -> tuple[bool, int, str, bool, bool]:
    """Compute repeat counts and progression signals for current attempt."""
    same_signature = (
        bool(previous) and str(previous.get("failure_signature") or "") == failure_signature
    )
    repeat_count = int(previous.get("repeat_count") or 1) + 1 if same_signature and previous else 1
    hypothesis_delta = classify_hypothesis_delta(previous_hypothesis, hypothesis_value)
    plan_advanced = bool(plan_value) and plan_value != previous_plan
    hypothesis_advanced = hypothesis_delta in {"added", "refined", "replaced"}
    return same_signature, repeat_count, hypothesis_delta, plan_advanced, hypothesis_advanced


def resolve_target_state(
    previous_state: str,
    normalized_action: RCAAction,
    gate_allowed: bool,
) -> RCAState:
    """Resolve next state according to transition matrix + gate decision."""
    target_state = _STATE_TRANSITIONS.get(normalized_action, "evidence_bound")
    if gate_allowed:
        return target_state
    # A denied action keeps any valid state, including "stalled", which no action targets
    # (RCA-04): resetting it to "captured" dropped the stall marker.
    return cast(RCAState, previous_state if previous_state in _VALID_STATES else DEFAULT_STATE)


def apply_stall_rule(
    *,
    target_state: RCAState,
    gate_warnings: list[str],
    same_signature: bool,
    repeat_count: int,
    stall_threshold: int,
    hypothesis_advanced: bool,
    plan_advanced: bool,
    normalized_action: RCAAction,
) -> tuple[RCAState, list[str]]:
    """Apply stall detection for repeated signatures without progression."""
    if (
        same_signature
        and repeat_count >= max(stall_threshold, 2)
        and not hypothesis_advanced
        and not plan_advanced
        and normalized_action in {"bind_evidence", "change", "verify"}
    ):
        gate_warnings.append(
            "Repeated failure signature without hypothesis/plan advancement; state marked stalled."
        )
        return "stalled", gate_warnings
    return target_state, gate_warnings


def build_transition_history(
    *,
    previous: dict[str, Any] | None,
    now: str,
    normalized_action: RCAAction,
    previous_state: str,
    target_state: RCAState,
    gate_allowed: bool,
    gate_mode: str,
) -> list[dict[str, Any]]:
    """Append one transition row while capping history length."""
    previous_history_raw = (
        previous.get("transition_history") if isinstance(previous, dict) else cast(list[Any], [])
    )
    transition_history: list[dict[str, Any]] = []
    if isinstance(previous_history_raw, list):
        transition_history = [
            cast(dict[str, Any], item)
            for item in cast(list[Any], previous_history_raw)
            if isinstance(item, dict)
        ]
    transition_history.append(
        {
            "at": now,
            "action": normalized_action,
            "from": previous_state,
            "to": target_state,
            "gate_allowed": gate_allowed,
            "gate_mode": gate_mode,
        }
    )
    if len(transition_history) > 32:
        transition_history = transition_history[-32:]
    return transition_history
