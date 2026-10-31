"""Pure RCA transition and state-evolution helpers."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, NamedTuple, cast, get_args

from .rca_types import RCAAction, RCAState

__all__ = ["RCAAction", "RCAState"]  # re-exported: defined once in rca_types (RCA-22)

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
# Actions that re-tread the same failure instead of describing new reasoning. Only these
# feed the repair streak that drives stall detection (RCA-06).
_REPAIR_LOOP_ACTIONS: frozenset[str] = frozenset({"bind_evidence", "change", "verify"})


def normalize_action(action: str) -> RCAAction:
    """Normalize and validate an RCA action string.

    Args:
        action: Action name in any case, optionally surrounded by whitespace.

    Returns:
        The stripped, lowercase action.

    Raises:
        ValueError: If the action is not one of the known RCA actions.
    """
    lowered = action.strip().lower()
    if lowered in _VALID_ACTIONS:
        return cast(RCAAction, lowered)
    raise ValueError(
        f"Invalid RCA action {action!r}. Valid actions: {', '.join(sorted(_VALID_ACTIONS))}"
    )


def classify_hypothesis_delta(previous: str, current: str) -> str:
    """Classify how the current hypothesis changed vs previous.

    Args:
        previous: Hypothesis from the previous attempt; empty when there was none.
        current: Effective hypothesis for the current attempt.

    Returns:
        After stripping both texts: ``unknown`` when both are empty, ``unchanged`` when
        they are equal, ``refined`` when the current text contains the previous one,
        ``added`` or ``removed`` when only one side is present, and ``replaced``
        otherwise.
    """
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
    """Return the allowed actions for *state*, primary next action first.

    The list is ordered by preference, not alphabetically. It drives display/UI
    guidance and the gate's ``illegal_transition`` check.

    Args:
        state: Current RCA state.

    Returns:
        The allowed actions; ``["bind_evidence"]`` for an unknown state.
    """
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
    """Return canonical coaching steps for the given state.

    Args:
        state: RCA state to coach from.

    Returns:
        Human-readable next steps; empty for ``resolved`` and unknown states.
    """
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
    """Merge + deduplicate evidence refs while keeping stable order.

    Args:
        previous: Already stored refs, which come first.
        incoming: New refs appended after *previous*.
        max_items: Maximum number of refs to keep; once it is reached, later refs are
            dropped.

    Returns:
        Stripped, non-empty, unique refs in first-seen order.
    """
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


class PreviousAttemptValues(NamedTuple):
    """Carry-forward inputs read from the previous attempt record.

    Attributes:
        state: RCA state of the previous attempt.
        hypothesis: Previous hypothesis text; empty when there was none.
        experiment_plan: Previous experiment plan; empty when there was none.
        evidence_refs: Previously stored evidence refs.
        verification_result: Previous verification result, or ``None``.
        change_summary: Previous proposed change summary; empty when there was none.
    """

    state: str
    hypothesis: str
    experiment_plan: str
    evidence_refs: list[Any]
    verification_result: Any
    change_summary: str


def resolve_previous_values(previous: dict[str, Any] | None) -> PreviousAttemptValues:
    """Extract previous attempt values with stable defaults.

    Args:
        previous: Previous attempt record, or ``None`` for a session's first attempt.

    Returns:
        The carry-forward values. Without a previous record the state is
        ``captured``, the text fields and evidence refs are empty, and the
        verification result is ``None``.
    """
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
    return PreviousAttemptValues(
        state=previous_state,
        hypothesis=previous_hypothesis,
        experiment_plan=previous_plan,
        evidence_refs=previous_refs,
        verification_result=previous_verification,
        change_summary=previous_change_summary,
    )


def resolve_merged_evidence_refs(
    previous_refs: Any,
    evidence_refs: list[str] | None,
) -> list[str]:
    """Merge previously stored evidence refs with incoming refs.

    Args:
        previous_refs: Refs stored on the previous record; any non-list value counts
            as no refs.
        evidence_refs: Incoming refs, or ``None`` for none.

    Returns:
        The merged refs, as produced by :func:`merge_evidence_refs`.
    """
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
    """Resolve effective text values (input overrides previous).

    Args:
        previous_hypothesis: Hypothesis carried from the previous attempt.
        previous_plan: Experiment plan carried from the previous attempt.
        previous_change_summary: Change summary carried from the previous attempt.
        hypothesis: New hypothesis; ``None`` or blank keeps *previous_hypothesis*.
        experiment_plan: New experiment plan; ``None`` or blank keeps
            *previous_plan*.
        proposed_change_summary: New change summary; ``None`` or blank keeps
            *previous_change_summary*.

    Returns:
        The effective hypothesis, experiment plan, and change summary, with new
        values stripped.
    """
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

    Args:
        previous_verification: Verification result stored on the previous attempt.
        verification_result: Verification result supplied with this attempt, if any.
        same_signature: Whether this attempt has the previous attempt's failure
            signature.
        action: Normalized action of this attempt.

    Returns:
        The new result when given; else the previous result when the signature is
        unchanged and the action is not ``change``; else ``None``.
    """
    if verification_result is not None:
        return verification_result
    if same_signature and action != "change":
        return previous_verification
    return None


def verification_passed(verification_result: Any) -> bool:
    """Return whether a verification result dict reports ``status == "pass"``.

    Args:
        verification_result: Verification result to inspect; non-dict values count as
            not passing.

    Returns:
        True only for a dict whose ``status`` is ``"pass"``.
    """
    if not isinstance(verification_result, dict):
        return False
    return str(cast(dict[str, Any], verification_result).get("status") or "") == "pass"


class AttemptProgress(NamedTuple):
    """Repeat and progression signals of the current attempt versus the previous one.

    Attributes:
        same_signature: Whether the failure signature matches the previous attempt's.
        repeat_count: Consecutive attempts with this signature, including this one.
        hypothesis_delta: Classification from :func:`classify_hypothesis_delta`.
        plan_advanced: Whether a non-empty plan differs from the previous plan.
        hypothesis_advanced: Whether the hypothesis was added, refined, or replaced.
    """

    same_signature: bool
    repeat_count: int
    hypothesis_delta: str
    plan_advanced: bool
    hypothesis_advanced: bool


@dataclass(frozen=True)
class TransitionInputs:
    """The requested action and how the attempt progressed (RCA-23).

    Attributes:
        action: Normalized action of the current attempt.
        progress: Repeat and progression signals versus the previous attempt.
        stall_threshold: Repair-streak length that marks the session stalled; values
            below 2 are treated as 2.
        repair_streak: Consecutive unproductive repair attempts, including this one.
        gate_allowed: Whether the gate let the action proceed.
    """

    action: RCAAction
    progress: AttemptProgress
    stall_threshold: int
    repair_streak: int = 0
    gate_allowed: bool = True


@dataclass(frozen=True)
class TransitionStep:
    """One row of ``transition_history`` (RCA-23).

    Attributes:
        action: Normalized action that was requested.
        from_state: State before the action.
        to_state: State after the gate decision and stall detection.
        gate_allowed: Whether the gate let the action proceed.
        gate_mode: Gate mode that evaluated the action.
    """

    action: RCAAction
    from_state: str
    to_state: RCAState
    gate_allowed: bool
    gate_mode: str


def derive_attempt_progress(
    *,
    previous: dict[str, Any] | None,
    failure_signature: str,
    previous_hypothesis: str,
    hypothesis_value: str,
    previous_plan: str,
    plan_value: str,
) -> AttemptProgress:
    """Compute repeat counts and progression signals for current attempt.

    Args:
        previous: Previous attempt record, or ``None`` for a session's first attempt.
        failure_signature: Failure signature of the current attempt.
        previous_hypothesis: Hypothesis carried from the previous attempt.
        hypothesis_value: Effective hypothesis of the current attempt.
        previous_plan: Experiment plan carried from the previous attempt.
        plan_value: Effective experiment plan of the current attempt.

    Returns:
        The signature match, repeat count, hypothesis delta, and whether the plan or
        hypothesis advanced.
    """
    same_signature = (
        bool(previous) and str(previous.get("failure_signature") or "") == failure_signature
    )
    repeat_count = int(previous.get("repeat_count") or 1) + 1 if same_signature and previous else 1
    hypothesis_delta = classify_hypothesis_delta(previous_hypothesis, hypothesis_value)
    plan_advanced = bool(plan_value) and plan_value != previous_plan
    hypothesis_advanced = hypothesis_delta in {"added", "refined", "replaced"}
    return AttemptProgress(
        same_signature=same_signature,
        repeat_count=repeat_count,
        hypothesis_delta=hypothesis_delta,
        plan_advanced=plan_advanced,
        hypothesis_advanced=hypothesis_advanced,
    )


def derive_repair_streak(
    *,
    previous_streak: int,
    action: RCAAction,
    progress: AttemptProgress,
    evidence_advanced: bool,
    verification_is_passing: bool,
) -> int:
    """Count consecutive unsuccessful repair attempts against one failure signature.

    Stall detection needs "how often did we re-try the same failure without bringing
    anything new", not "how many attempts share this signature": the plain
    ``repeat_count`` also counts the bookkeeping steps of the normal
    ``capture -> bind_evidence -> hypothesize -> plan -> change`` walk, so a first-time
    workflow used to be marked ``stalled`` at ``change`` (RCA-06).

    The streak resets to 0 whenever the attempt brings progress -- a different failure
    signature, an advanced hypothesis or plan, newly bound evidence, or a passing
    verification. ``bind_evidence``/``change``/``verify`` that bring none of those
    increment it by one; any other action carries it forward unchanged, so describing
    the same plan twice neither trips nor clears loop protection.

    Args:
        previous_streak: Repair streak stored on the previous attempt.
        action: Normalized action of the current attempt.
        progress: Repeat and progression signals of the current attempt.
        evidence_advanced: Whether this attempt bound evidence refs that were not
            stored yet.
        verification_is_passing: Whether the effective verification result passes.

    Returns:
        The repair streak after this attempt.
    """
    carried = max(previous_streak, 0)
    if not progress.same_signature:
        return 0
    if progress.hypothesis_advanced or progress.plan_advanced or evidence_advanced:
        return 0
    if verification_is_passing:
        return 0
    if action not in _REPAIR_LOOP_ACTIONS:
        return carried
    return carried + 1


def resolve_target_state(
    previous_state: str,
    normalized_action: RCAAction,
    gate_allowed: bool,
) -> RCAState:
    """Resolve next state according to transition matrix + gate decision.

    Args:
        previous_state: State before the action.
        normalized_action: Requested action, already normalized.
        gate_allowed: Whether the gate let the action proceed.

    Returns:
        The action's target state when allowed; otherwise *previous_state* when it is
        a valid state, else ``captured``.
    """
    target_state = _STATE_TRANSITIONS.get(normalized_action, "evidence_bound")
    if gate_allowed:
        return target_state
    # A denied action keeps any valid state, including "stalled", which no action targets
    # (RCA-04): resetting it to "captured" dropped the stall marker.
    return cast(RCAState, previous_state if previous_state in _VALID_STATES else DEFAULT_STATE)


def apply_stall_rule(
    inputs: TransitionInputs,
    *,
    target_state: RCAState,
    gate_warnings: list[str],
) -> tuple[RCAState, list[str]]:
    """Apply stall detection for repeated repair attempts without progression.

    A denied action did not happen, so it never marks the session stalled: evaluating
    the rule unconditionally let a rejected request change the persisted state and the
    allowed actions (RCA-06).

    Args:
        inputs: Requested action, progress signals, repair streak, stall threshold,
            and gate decision.
        target_state: State resolved from the transition matrix and gate decision.
        gate_warnings: Warnings collected so far; a stall warning is appended to this
            list in place.

    Returns:
        ``("stalled", gate_warnings)`` when the stall rule fires, else
        ``(target_state, gate_warnings)``.
    """
    progress = inputs.progress
    if (
        inputs.gate_allowed
        and progress.same_signature
        and inputs.repair_streak >= max(inputs.stall_threshold, 2)
        and inputs.action in _REPAIR_LOOP_ACTIONS
    ):
        gate_warnings.append(
            "Repeated failure signature without hypothesis/plan advancement; state marked stalled."
        )
        return "stalled", gate_warnings
    return target_state, gate_warnings


def build_transition_history(
    previous: dict[str, Any] | None,
    *,
    now: str,
    step: TransitionStep,
) -> list[dict[str, Any]]:
    """Append one transition row while capping history length.

    Args:
        previous: Previous attempt record whose ``transition_history`` is extended, or
            ``None`` to start a new history.
        now: Timestamp recorded on the new row.
        step: Action, state change, and gate outcome of the current attempt.

    Returns:
        A new history list with the row appended, keeping the 32 most recent rows.
    """
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
            "action": step.action,
            "from": step.from_state,
            "to": step.to_state,
            "gate_allowed": step.gate_allowed,
            "gate_mode": step.gate_mode,
        }
    )
    if len(transition_history) > 32:
        transition_history = transition_history[-32:]
    return transition_history
