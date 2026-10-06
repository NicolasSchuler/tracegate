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
    """Normalized RCA attempt inputs shared across handlers and state engine."""

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
    """RCA advancement request where identity is derived from a snapshot."""

    out_dir: str
    snapshot: dict[str, Any] | None
    attempt: RCAAttemptRequest
    fallback_stdout: str = ""
    fallback_stderr: str = ""
