"""Shared RCA request types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


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
