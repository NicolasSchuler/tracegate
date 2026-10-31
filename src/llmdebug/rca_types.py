"""Shared RCA request types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, TypedDict

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


class RCAAttemptRecord(TypedDict):
    """One persisted RCA attempt, as built by ``rca_engine.build_rca_attempt_record``.

    Records read back from storage are plain dicts and may predate fields added here.

    Attributes:
        attempt_id: Unique identifier of this attempt.
        session_id: RCA session the attempt belongs to.
        state: RCA state after the requested action, the gate decision, and stall
            detection.
        failure_signature: Stable fingerprint of the failure the attempt addresses.
        repeat_count: Consecutive attempts in the session with this failure signature,
            including this one.
        repair_streak: Consecutive repair attempts on this signature that brought no
            progress; drives stall detection.
        evidence_refs: Merged, deduplicated evidence references (at most 16).
        hypothesis: Current root-cause hypothesis, carried forward when not restated.
        hypothesis_delta: How the hypothesis changed versus the previous attempt:
            ``unknown``, ``unchanged``, ``refined``, ``added``, ``removed``, or
            ``replaced``.
        experiment_plan: Current experiment plan, carried forward when not restated.
        proposed_change_summary: Current change summary, carried forward when not
            restated.
        verification_result: Verification outcome (``status`` and ``checks``) that
            applies to this attempt, or ``None``.
        gate_feedback: Serialized :class:`~llmdebug.gates.GateFeedback` for the
            requested action.
        gate_warnings: Gate warnings, plus a stall warning when this attempt marked the
            session stalled.
        allowed_actions: Actions suggested from the new state.
        next_required: Next steps from the gate, or the state's default coaching steps.
        exploratory: Whether the attempt was marked exploratory.
        exploration_budget_remaining: Exploration budget left after this attempt;
            each same-signature attempt without hypothesis progress consumes one unit.
        transition_history: Chronological state transitions, capped at the 32 most
            recent.
        created_at: ISO-8601 UTC timestamp of the session's first attempt.
        updated_at: ISO-8601 UTC timestamp of this attempt.
    """

    attempt_id: str
    session_id: str
    state: RCAState
    failure_signature: str
    repeat_count: int
    repair_streak: int
    evidence_refs: list[str]
    hypothesis: str | None
    hypothesis_delta: str
    experiment_plan: str | None
    proposed_change_summary: str | None
    verification_result: dict[str, Any] | None
    gate_feedback: dict[str, Any]
    gate_warnings: list[str]
    allowed_actions: list[str]
    next_required: list[str]
    exploratory: bool
    exploration_budget_remaining: int
    transition_history: list[dict[str, Any]]
    created_at: str
    updated_at: str


@dataclass(frozen=True)
class RCAAttemptRequest:
    """Normalized RCA attempt inputs shared across handlers and state engine.

    Attributes:
        session_id: RCA session to advance.
        action: Requested RCA action such as ``bind_evidence`` or ``verify``; the
            engine normalizes and validates it.
        failure_signature: Stable fingerprint of the current failure.
        evidence_refs: New evidence references to merge with the stored ones.
        hypothesis: Root-cause hypothesis; ``None`` or empty keeps the previous one.
        experiment_plan: Experiment plan; ``None`` or empty keeps the previous one.
        proposed_change_summary: Summary of the proposed change; ``None`` or empty
            keeps the previous one.
        verification_result: New verification outcome with ``status`` and ``checks``,
            or ``None`` to carry the previous result forward where allowed.
        gate_mode: Gate policy (``off``, ``soft``, or ``strict``); ``None`` or an
            unrecognized value means ``soft``.
        exploratory: Whether this step is exploratory; together with
            *override_reason* it lets a strict gate accept hard failures.
        override_reason: Justification for an exploratory strict-gate override.
        stall_threshold: Repair-streak length that marks the session stalled
            (at least 2 is enforced).
        exploration_budget: Same-signature attempts allowed before loop protection
            warns.
    """

    session_id: str
    action: str
    failure_signature: str
    evidence_refs: tuple[str, ...] = ()
    hypothesis: str | None = None
    experiment_plan: str | None = None
    proposed_change_summary: str | None = None
    verification_result: dict[str, Any] | None = None
    gate_mode: str | None = None
    exploratory: bool = False
    override_reason: str | None = None
    stall_threshold: int = 3
    exploration_budget: int = 2


@dataclass(frozen=True)
class RCASnapshotAdvanceRequest:
    """RCA advancement request where identity is derived from a snapshot.

    Attributes:
        out_dir: Snapshot directory that holds the RCA store.
        snapshot: Snapshot used to derive the session ID, failure signature, and
            evidence refs; ``None`` derives the signature from the fallback output.
        attempt: Action and attempt inputs; its ``session_id`` and
            ``failure_signature`` are replaced by the derived values.
        fallback_stdout: Captured stdout used for the failure signature when there is
            no snapshot.
        fallback_stderr: Captured stderr, preferred over stdout for that signature.
    """

    out_dir: str
    snapshot: dict[str, Any] | None
    attempt: RCAAttemptRequest
    fallback_stdout: str = ""
    fallback_stderr: str = ""
