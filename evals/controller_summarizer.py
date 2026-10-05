"""Deterministic controller-owned summarization for hybrid adaptive runs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from baselines.controller_packet import ControllerPacketV1

_MAX_SUMMARY_CHARS = 220


def _truncate(text: str, *, max_chars: int) -> str:
    value = str(text or "").strip()
    if len(value) <= max(0, int(max_chars)):
        return value
    if max_chars <= 3:
        return value[: max(0, int(max_chars))]
    return value[: max(0, int(max_chars)) - 3].rstrip() + "..."


@dataclass(frozen=True)
class ControllerStrategySummaryResult:
    summary: str
    sources: list[str]


def build_controller_strategy_summary(
    controller_packet: ControllerPacketV1,
) -> ControllerStrategySummaryResult:
    """Build a bounded solver-facing strategy summary from typed controller state."""
    evidence_bundle = controller_packet.evidence_bundle
    history_bundle = controller_packet.history_bundle
    verifier_report = controller_packet.verifier_report
    solver_hints = controller_packet.solver_hints

    clauses: list[str] = []
    sources: list[str] = []

    crash_anchor = ""
    if evidence_bundle.summary is not None:
        crash_anchor = str(evidence_bundle.summary.crash_anchor or "").strip()
    if not crash_anchor and evidence_bundle.snapshot_reduction is not None:
        crash_anchor = str(evidence_bundle.snapshot_reduction.crash_frame_summary or "").strip()
    if crash_anchor:
        clauses.append(f"Localize the fix at {crash_anchor}.")
        sources.append("crash_anchor")

    if verifier_report.status and verifier_report.status != "none":
        verifier_reason = str(verifier_report.reasons[0] if verifier_report.reasons else "").strip()
        if verifier_reason:
            clauses.append(f"Avoid the previous verifier failure: {verifier_reason}.")
        else:
            clauses.append(f"Respect the verifier constraint: {verifier_report.status}.")
        sources.append("verifier_report")

    if history_bundle.lessons:
        lesson = str(history_bundle.lessons[0].lesson or "").strip()
        if lesson:
            clauses.append(f"Carry forward this lesson: {lesson}.")
            sources.append("history_lessons")
    elif history_bundle.previous_note:
        clauses.append(f"Previous attempt outcome: {history_bundle.previous_note}.")
        sources.append("previous_note")

    retry_plan = str(solver_hints.retry_plan or "").strip()
    if retry_plan:
        clauses.append(f"Execution plan: {retry_plan}.")
        sources.append("retry_plan")

    summary = _truncate(" ".join(clauses), max_chars=_MAX_SUMMARY_CHARS)
    return ControllerStrategySummaryResult(summary=summary, sources=sources)


__all__ = [
    "ControllerStrategySummaryResult",
    "build_controller_strategy_summary",
]
