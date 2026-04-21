"""Typed adaptive-controller sidecars for hybrid eval runs.

These helpers package attempt-scoped controller data for the shared eval
harness. The packet is additive: baseline-only runs can ignore it entirely,
while adaptive conditions and controller-backed baselines can consume the
structured metadata they need.
"""

from __future__ import annotations

import ast
import logging
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
from uuid import uuid4

from llmdebug._debug_session import coerce_crash_frame_index
from llmdebug.git_context import get_enhanced_git_context

from . import _json as json
from .adaptive_history import load_persistent_adaptive_history

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = "v1"
_MAX_EVIDENCE_SUMMARY_CHARS = 480
_MAX_RETRY_PLAN_CHARS = 2_000
_MAX_RETRY_FEEDBACK_CHARS = 1_200
_MAX_HISTORY_TURNS = 4
_MAX_REPO_COMMITS = 3
_MAX_LOCALS = 8
_MAX_AST_SNIPPET_LINES = 24
_MAX_SELECTED_EVIDENCE_ITEMS = 6
_MAX_EVIDENCE_FACTS = 5
_MAX_LOCAL_FACTS = 4
_MAX_HISTORY_LESSONS = 4
_MAX_HISTORY_LESSON_CHARS = 180
HYBRID_GUIDANCE_MODE_TUNED_V1 = "tuned_v1"
HYBRID_GUIDANCE_MODE_FAIR_V1 = "fair_v1"
HYBRID_GUIDANCE_MODE_FIXED_V1 = "fixed_v1"
HYBRID_GUIDANCE_MODES = frozenset(
    {
        HYBRID_GUIDANCE_MODE_FAIR_V1,
        HYBRID_GUIDANCE_MODE_FIXED_V1,
        HYBRID_GUIDANCE_MODE_TUNED_V1,
    }
)
ADAPTIVE_CONTROLLER_SIDECAR_FILENAMES: tuple[str, ...] = (
    "llmdebug_controller_packet.json",
    "llmdebug_evidence_bundle.json",
    "llmdebug_history_bundle.json",
    "llmdebug_verifier_report.json",
)


def _truncate(text: str, *, max_chars: int) -> str:
    value = str(text or "").strip()
    if len(value) <= max(0, int(max_chars)):
        return value
    if max_chars <= 3:
        return value[: max(0, int(max_chars))]
    return value[: max(0, int(max_chars)) - 3].rstrip() + "..."


@dataclass(frozen=True)
class ASTContextV1:
    """Compact AST view around the crash symbol used by controller prompts."""

    path: str
    crash_line: int | None
    symbol_name: str
    symbol_kind: str
    node_type: str
    line_span: tuple[int, int] | None
    snippet: str
    related_names: list[str]


@dataclass(frozen=True)
class SnapshotReductionV1:
    """Reduced snapshot slice chosen for controller-side prompt assembly."""

    selected_frame_count: int
    omitted_frame_count: int
    crash_frame_summary: str
    selected_frames: list[dict[str, Any]]
    selected_locals: list[dict[str, str]]


@dataclass(frozen=True)
class EvidenceItemV1:
    """One selected evidence item carried through controller sidecars."""

    id: str
    kind: str
    locator: str
    summary: str
    path: str | None
    line_span: tuple[int, int] | None
    score: float
    provenance: str
    raw_ref: str | None = None


@dataclass(frozen=True)
class EvidenceSummaryV1:
    """High-level summary of why the selected evidence set was chosen."""

    strategy: str
    crash_anchor: str
    top_facts: list[str]
    selected_local_names: list[str]
    omitted_item_count: int
    omitted_frame_count: int


@dataclass(frozen=True)
class EvidenceBundleV1:
    """Attempt-scoped evidence packet for controller and solver consumers.

    ``selected_ids`` is the full ordered evidence set for this attempt.
    ``new_ids`` is the subset first introduced on this attempt.
    ``reused_ids`` is the subset carried forward from prior attempts.
    """

    schema_version: str
    case_id: str
    condition: str
    attempt: int
    failure_signature: str
    selected_ids: list[str]
    new_ids: list[str]
    reused_ids: list[str]
    ast_context: ASTContextV1 | None
    snapshot_reduction: SnapshotReductionV1 | None
    summary: EvidenceSummaryV1 | None
    items: list[EvidenceItemV1]
    snapshot_full_text: str = ""


@dataclass(frozen=True)
class RepoCommitV1:
    """Compact commit summary included in repo-history context."""

    commit: str
    commit_full: str
    summary: str
    author: str
    author_time: str


@dataclass(frozen=True)
class RepoHistoryV1:
    """Crash-file git history hints attached to adaptive retry context."""

    repo_root: str
    head_branch: str
    head_commit: str
    crash_file_path: str
    crash_file_line: int | None
    crash_file_blame_summary: str
    crash_file_diffstat: dict[str, Any] | None
    recent_commits: list[RepoCommitV1]
    warnings: list[str]
    error: str = ""


@dataclass(frozen=True)
class HistoryLessonV1:
    """One condensed lesson extracted from prior attempts or RCA history."""

    source: str
    attempt: int | None
    lesson: str


@dataclass(frozen=True)
class HistoryBundleV1:
    """Retry-history sidecar for adaptive and controller-backed retries.

    ``retry_context_mode`` controls how much historical text is exposed to the
    patching model. The bundle also carries prior verifier state and optional
    persistent RCA-derived lessons.
    """

    schema_version: str
    retry_context_mode: str
    attempt_turns: list[str]
    retry_history_text: str
    retry_history_turns_included: int
    retry_history_truncated: bool
    previous_note: str
    previous_failure_signature: str
    previous_patch_verifier_rejected: bool
    previous_patch_verifier_reasons: list[str]
    lessons: list[HistoryLessonV1]
    repo_history: RepoHistoryV1 | None
    persistent_session_id: str
    persistent_records_loaded: int
    persistent_history_error: str = ""


@dataclass(frozen=True)
class VerifierReportV1:
    """Verifier and syntax-preflight outcome summary for the current attempt."""

    schema_version: str
    status: str
    scope: str
    reasons: list[str]
    previous_timed_out: bool
    previous_timeout_sec: int | None
    syntax_cycle: int
    syntax_max_cycles: int
    last_error_type: str
    last_error_msg: str
    retry_feedback: str


@dataclass(frozen=True)
class AdaptiveControllerStateV2:
    """Controller action and budgeting state emitted for one attempt."""

    schema_version: str
    helper_action_selected: str
    helper_action_executed: str
    helper_decision_reason: str
    helper_outcome: str
    helper_materialized_context: bool
    helper_calls_used: int
    helper_budget_max_calls: int
    no_progress_streak: int
    retry_prompt_mode: str
    retry_context_mode: str
    context_compaction_requested: bool
    context_compaction_invoked: bool
    context_compaction_success: bool
    context_compaction_trigger: str
    prompt_budget: dict[str, int]


@dataclass(frozen=True)
class SolverHintsV1:
    """Structured hints forwarded from controller logic to the patching model."""

    tool_routes: list[str]
    retry_plan: str
    recommended_focus: list[str]
    decomposition: list[dict[str, Any]]
    strategy_summary: str = ""


@dataclass(frozen=True)
class ControllerPacketV1:
    """Top-level adaptive controller sidecar consumed by downstream helpers."""

    schema_version: str
    case_id: str
    condition: str
    attempt: int
    controller_state: AdaptiveControllerStateV2
    evidence_bundle: EvidenceBundleV1
    history_bundle: HistoryBundleV1
    verifier_report: VerifierReportV1
    solver_hints: SolverHintsV1


@dataclass(frozen=True)
class RenderedControllerGuidance:
    """Rendered bounded controller guidance plus surface metadata."""

    text: str
    mode: str
    sections: tuple[str, ...]

    @property
    def char_count(self) -> int:
        return len(self.text)


def _normalize_hybrid_guidance_mode(mode: str | None) -> str:
    normalized = str(mode or "").strip() or HYBRID_GUIDANCE_MODE_TUNED_V1
    if normalized not in HYBRID_GUIDANCE_MODES:
        allowed = ", ".join(sorted(HYBRID_GUIDANCE_MODES))
        raise ValueError(f"invalid hybrid guidance mode {normalized!r}; expected one of: {allowed}")
    return normalized


def _render_tuned_controller_guidance_lines(
    controller_packet: ControllerPacketV1 | None,
    *,
    include_history: bool,
    include_retry_plan: bool,
) -> tuple[list[str], tuple[str, ...]]:
    if controller_packet is None:
        return [], ()
    lines: list[str] = []
    sections: list[str] = []
    state = controller_packet.controller_state
    if state.helper_action_executed or state.helper_outcome:
        lines.append(
            "controller_state: "
            f"action={state.helper_action_executed or 'none'} "
            f"outcome={state.helper_outcome or 'n/a'}"
        )
        sections.append("controller_state")
    evidence_summary = controller_packet.evidence_bundle.summary
    if evidence_summary is not None:
        if evidence_summary.crash_anchor:
            lines.append(f"crash_anchor: {evidence_summary.crash_anchor}")
            sections.append("crash_anchor")
        if evidence_summary.top_facts:
            lines.append("evidence_summary: " + " | ".join(evidence_summary.top_facts[:4]))
            sections.append("evidence_summary")
    verifier = controller_packet.verifier_report
    if verifier.status and verifier.status != "none":
        lines.append(f"verifier_status: {verifier.status}")
        sections.append("verifier_status")
    if verifier.reasons:
        rendered_reasons = ", ".join(
            str(item) for item in verifier.reasons[:3] if str(item).strip()
        )
        if rendered_reasons:
            lines.append("verifier_reasons: " + rendered_reasons)
            sections.append("verifier_reasons")
    if include_history:
        lessons = controller_packet.history_bundle.lessons
        if lessons:
            rendered_lessons = []
            for lesson in lessons[:3]:
                prefix = (
                    f"attempt {lesson.attempt}"
                    if isinstance(lesson.attempt, int) and lesson.attempt > 0
                    else lesson.source
                )
                rendered_lessons.append(f"{prefix}: {lesson.lesson}")
            if rendered_lessons:
                lines.append("history_lessons: " + " | ".join(rendered_lessons))
                sections.append("history_lessons")
        repo_history = controller_packet.history_bundle.repo_history
        if repo_history is not None and repo_history.crash_file_blame_summary:
            lines.append(f"blame_hint: {repo_history.crash_file_blame_summary}")
            sections.append("blame_hint")
    if include_retry_plan and controller_packet.solver_hints.retry_plan:
        lines.append(f"retry_plan: {controller_packet.solver_hints.retry_plan}")
        sections.append("retry_plan")
    if controller_packet.solver_hints.strategy_summary:
        lines.append(f"strategy_summary: {controller_packet.solver_hints.strategy_summary}")
        sections.append("strategy_summary")
    if controller_packet.solver_hints.decomposition:
        rendered_steps = []
        for step in controller_packet.solver_hints.decomposition[:3]:
            if not isinstance(step, dict):
                continue
            title = str(step.get("title") or "").strip()
            question = str(step.get("question") or "").strip()
            if title and question:
                rendered_steps.append(f"{title}: {question}")
        if rendered_steps:
            lines.append("decomposition: " + " | ".join(rendered_steps))
            sections.append("decomposition")
    return lines, tuple(sections)


def _join_guidance_sections_with_cap(
    rendered_sections: list[tuple[str, str]],
    *,
    max_chars: int,
) -> tuple[str, tuple[str, ...]]:
    if max_chars <= 0:
        return "", ()
    lines: list[str] = []
    included_sections: list[str] = []
    for section_name, section_text in rendered_sections:
        line = str(section_text or "").strip()
        if not line:
            continue
        separator = "\n" if lines else ""
        current_length = sum(len(item) for item in lines) + max(0, len(lines) - 1)
        remaining_chars = max_chars - current_length - len(separator)
        if remaining_chars <= 0:
            break
        if len(line) <= remaining_chars:
            lines.append(line)
            included_sections.append(section_name)
            continue
        truncated = _truncate(line, max_chars=remaining_chars)
        if truncated:
            lines.append(truncated)
            included_sections.append(section_name)
        break
    return "\n".join(lines), tuple(included_sections)


def _render_fair_controller_guidance(
    controller_packet: ControllerPacketV1 | None,
    *,
    include_history: bool,
    include_retry_plan: bool,
    max_chars: int,
) -> RenderedControllerGuidance:
    if controller_packet is None:
        return RenderedControllerGuidance(
            text="",
            mode=HYBRID_GUIDANCE_MODE_FAIR_V1,
            sections=(),
        )
    rendered_sections: list[tuple[str, str]] = []
    state = controller_packet.controller_state
    action = state.helper_action_executed or state.helper_action_selected or "none"
    outcome = state.helper_outcome or "n/a"
    if action != "none" or outcome != "n/a":
        rendered_sections.append(
            ("controller_state", f"controller_state: action={action} outcome={outcome}")
        )

    evidence_summary = controller_packet.evidence_bundle.summary
    if evidence_summary is not None and evidence_summary.crash_anchor:
        rendered_sections.append(
            (
                "crash_anchor",
                f"crash_anchor: {_truncate(evidence_summary.crash_anchor, max_chars=220)}",
            )
        )
    if evidence_summary is not None and evidence_summary.top_facts:
        top_facts = [
            _truncate(str(fact), max_chars=180)
            for fact in evidence_summary.top_facts[:4]
            if str(fact).strip()
        ]
        top_facts = [fact for fact in top_facts if fact]
        if top_facts:
            rendered_sections.append(
                ("evidence_summary", "evidence_summary: " + " | ".join(top_facts))
            )

    verifier = controller_packet.verifier_report
    verifier_status = str(verifier.status or "").strip()
    verifier_reasons = [
        _truncate(str(reason), max_chars=160)
        for reason in verifier.reasons[:3]
        if str(reason).strip()
    ]
    verifier_reasons = [reason for reason in verifier_reasons if reason]
    if verifier_status or verifier_reasons:
        verifier_bits = [f"status={verifier_status or 'none'}"]
        if verifier_reasons:
            verifier_bits.append("reasons=" + " | ".join(verifier_reasons))
        rendered_sections.append(("verifier", "verifier: " + " ".join(verifier_bits)))

    if include_history:
        lessons = controller_packet.history_bundle.lessons
        rendered_lessons = []
        for lesson in lessons[:3]:
            prefix = (
                f"attempt {lesson.attempt}"
                if isinstance(lesson.attempt, int) and lesson.attempt > 0
                else str(lesson.source or "history")
            )
            lesson_text = _truncate(str(lesson.lesson), max_chars=160)
            if lesson_text:
                rendered_lessons.append(f"{prefix}: {lesson_text}")
        if rendered_lessons:
            rendered_sections.append(
                ("history_lessons", "history_lessons: " + " | ".join(rendered_lessons))
            )

    if include_retry_plan and controller_packet.solver_hints.retry_plan:
        rendered_sections.append(
            (
                "retry_plan",
                "retry_plan: "
                + _truncate(controller_packet.solver_hints.retry_plan, max_chars=360),
            )
        )

    text, sections = _join_guidance_sections_with_cap(
        rendered_sections,
        max_chars=max_chars,
    )
    return RenderedControllerGuidance(
        text=text,
        mode=HYBRID_GUIDANCE_MODE_FAIR_V1,
        sections=sections,
    )


def _render_fixed_controller_guidance(
    controller_packet: ControllerPacketV1 | None,
    *,
    max_chars: int,
) -> RenderedControllerGuidance:
    if controller_packet is None:
        return RenderedControllerGuidance(
            text="",
            mode=HYBRID_GUIDANCE_MODE_FIXED_V1,
            sections=(),
        )

    rendered_sections: list[tuple[str, str]] = []
    evidence_bundle = controller_packet.evidence_bundle
    evidence_summary = evidence_bundle.summary
    if evidence_summary is not None and evidence_summary.crash_anchor:
        rendered_sections.append(
            (
                "crash_anchor",
                f"crash_anchor: {_truncate(evidence_summary.crash_anchor, max_chars=220)}",
            )
        )
    if evidence_summary is not None and evidence_summary.top_facts:
        top_facts = [
            _truncate(str(fact), max_chars=180)
            for fact in evidence_summary.top_facts[:4]
            if str(fact).strip()
        ]
        top_facts = [fact for fact in top_facts if fact]
        if top_facts:
            rendered_sections.append(
                ("evidence_summary", "evidence_summary: " + " | ".join(top_facts))
            )

    ast_context = evidence_bundle.ast_context
    if ast_context is not None:
        header = f"ast_focus: {ast_context.path}"
        if ast_context.line_span:
            header = f"{header} lines={ast_context.line_span}"
        if ast_context.symbol_name or ast_context.node_type:
            header = f"{header} symbol={ast_context.symbol_name or ast_context.node_type}"
        rendered_sections.append(("ast_focus", _truncate(header, max_chars=240)))
        snippet = _truncate(ast_context.snippet, max_chars=900)
        if snippet:
            rendered_sections.append(("ast_snippet", f"ast_snippet:\n{snippet}"))

    if evidence_bundle.items:
        rendered_items: list[str] = []
        for row in evidence_bundle.items[:4]:
            kind = row.kind or "evidence"
            locator = row.locator or row.path or ""
            summary = _truncate(str(row.summary), max_chars=180)
            if summary:
                rendered_items.append(
                    f"{kind} {locator}".strip() + (f": {summary}" if summary else "")
                )
        if rendered_items:
            rendered_sections.append(
                ("selected_evidence", "selected_evidence: " + " | ".join(rendered_items))
            )

    snapshot_full_text = str(evidence_bundle.snapshot_full_text or "").strip()
    if snapshot_full_text:
        rendered_sections.append(("snapshot_full", f"snapshot_full:\n{snapshot_full_text}"))
    else:
        snapshot_reduction = evidence_bundle.snapshot_reduction
        if snapshot_reduction is not None and snapshot_reduction.crash_frame_summary:
            rendered_sections.append(
                (
                    "snapshot_focus",
                    "snapshot_focus: "
                    + _truncate(snapshot_reduction.crash_frame_summary, max_chars=220),
                )
            )

    text, sections = _join_guidance_sections_with_cap(
        rendered_sections,
        max_chars=max_chars,
    )
    return RenderedControllerGuidance(
        text=text,
        mode=HYBRID_GUIDANCE_MODE_FIXED_V1,
        sections=sections,
    )


def render_hybrid_controller_guidance(
    controller_packet: ControllerPacketV1 | None,
    *,
    mode: str = HYBRID_GUIDANCE_MODE_TUNED_V1,
    include_history: bool = True,
    include_retry_plan: bool = True,
    max_chars: int = 2_000,
) -> RenderedControllerGuidance:
    """Render hybrid baseline guidance and return text plus audit metadata."""
    resolved_mode = _normalize_hybrid_guidance_mode(mode)
    if resolved_mode == HYBRID_GUIDANCE_MODE_FAIR_V1:
        return _render_fair_controller_guidance(
            controller_packet,
            include_history=include_history,
            include_retry_plan=include_retry_plan,
            max_chars=max_chars,
        )
    if resolved_mode == HYBRID_GUIDANCE_MODE_FIXED_V1:
        return _render_fixed_controller_guidance(
            controller_packet,
            max_chars=max_chars,
        )
    lines, sections = _render_tuned_controller_guidance_lines(
        controller_packet,
        include_history=include_history,
        include_retry_plan=include_retry_plan,
    )
    if not lines:
        return RenderedControllerGuidance(
            text="",
            mode=HYBRID_GUIDANCE_MODE_TUNED_V1,
            sections=(),
        )
    text, included_sections = _join_guidance_sections_with_cap(
        list(zip(sections, lines, strict=True)),
        max_chars=max_chars,
    )
    return RenderedControllerGuidance(
        text=text,
        mode=HYBRID_GUIDANCE_MODE_TUNED_V1,
        sections=included_sections,
    )


def hybrid_guidance_metadata(rendered: RenderedControllerGuidance) -> dict[str, Any]:
    """Return patch-metadata fields for one rendered hybrid guidance block."""
    return {
        "baseline_hybrid_guidance_mode": rendered.mode,
        "baseline_hybrid_guidance_chars": rendered.char_count,
        "baseline_hybrid_guidance_sections": list(rendered.sections),
    }


def render_compact_controller_guidance(
    controller_packet: ControllerPacketV1 | None,
    *,
    include_history: bool = True,
    include_retry_plan: bool = True,
    max_chars: int = 2_000,
) -> str:
    """Render a generic, bounded guidance block for baseline prompts."""
    return render_hybrid_controller_guidance(
        controller_packet,
        mode=HYBRID_GUIDANCE_MODE_TUNED_V1,
        include_history=include_history,
        include_retry_plan=include_retry_plan,
        max_chars=max_chars,
    ).text


def _coerce_line_span(value: Any) -> tuple[int, int] | None:
    if isinstance(value, (list, tuple)) and len(value) == 2:
        first, second = value
        if (
            isinstance(first, int)
            and not isinstance(first, bool)
            and isinstance(second, int)
            and not isinstance(second, bool)
        ):
            return (first, second)
    return None


def _parse_ast_context(value: Any) -> ASTContextV1 | None:
    if not isinstance(value, dict):
        return None
    return ASTContextV1(
        path=str(value.get("path") or ""),
        crash_line=_safe_int(value.get("crash_line")),
        symbol_name=str(value.get("symbol_name") or ""),
        symbol_kind=str(value.get("symbol_kind") or ""),
        node_type=str(value.get("node_type") or ""),
        line_span=_coerce_line_span(value.get("line_span")),
        snippet=str(value.get("snippet") or ""),
        related_names=[str(item) for item in value.get("related_names", []) if str(item).strip()],
    )


def _parse_snapshot_reduction(value: Any) -> SnapshotReductionV1 | None:
    if not isinstance(value, dict):
        return None
    selected_frames = value.get("selected_frames")
    selected_locals = value.get("selected_locals")
    return SnapshotReductionV1(
        selected_frame_count=max(0, int(value.get("selected_frame_count") or 0)),
        omitted_frame_count=max(0, int(value.get("omitted_frame_count") or 0)),
        crash_frame_summary=str(value.get("crash_frame_summary") or ""),
        selected_frames=selected_frames if isinstance(selected_frames, list) else [],
        selected_locals=selected_locals if isinstance(selected_locals, list) else [],
    )


def _parse_evidence_summary(value: Any) -> EvidenceSummaryV1 | None:
    if not isinstance(value, dict):
        return None
    return EvidenceSummaryV1(
        strategy=str(value.get("strategy") or ""),
        crash_anchor=str(value.get("crash_anchor") or ""),
        top_facts=[str(item) for item in value.get("top_facts", []) if str(item).strip()],
        selected_local_names=[
            str(item) for item in value.get("selected_local_names", []) if str(item).strip()
        ],
        omitted_item_count=max(0, int(value.get("omitted_item_count") or 0)),
        omitted_frame_count=max(0, int(value.get("omitted_frame_count") or 0)),
    )


def _parse_repo_history(value: Any) -> RepoHistoryV1 | None:
    if not isinstance(value, dict):
        return None
    head_value = value.get("head")
    head_mapping = head_value if isinstance(head_value, dict) else {}
    crash_file_value = value.get("crash_file")
    crash_file_mapping = crash_file_value if isinstance(crash_file_value, dict) else {}
    blame_value = crash_file_mapping.get("blame")
    blame_mapping = blame_value if isinstance(blame_value, dict) else {}
    recent_commits_raw = value.get("recent_commits")
    recent_commits: list[RepoCommitV1] = []
    if isinstance(recent_commits_raw, list):
        for row in recent_commits_raw:
            if not isinstance(row, dict):
                continue
            recent_commits.append(
                RepoCommitV1(
                    commit=str(row.get("commit") or ""),
                    commit_full=str(row.get("commit_full") or ""),
                    summary=str(row.get("summary") or row.get("subject") or ""),
                    author=str(row.get("author") or ""),
                    author_time=str(row.get("author_time") or ""),
                )
            )
    warnings_raw = value.get("warnings")
    if not isinstance(warnings_raw, list):
        warnings_raw = value.get("_git_warnings")
    warnings = (
        [str(item) for item in warnings_raw if str(item).strip()]
        if isinstance(warnings_raw, list)
        else []
    )
    diffstat_value = value.get("crash_file_diffstat")
    return RepoHistoryV1(
        repo_root=str(value.get("repo_root") or ""),
        head_branch=str(value.get("head_branch") or head_mapping.get("branch") or ""),
        head_commit=str(value.get("head_commit") or head_mapping.get("commit") or ""),
        crash_file_path=str(
            value.get("crash_file_path") or crash_file_mapping.get("file_rel") or ""
        ),
        crash_file_line=_safe_int(value.get("crash_file_line"))
        or _safe_int(crash_file_mapping.get("line")),
        crash_file_blame_summary=str(
            value.get("crash_file_blame_summary") or blame_mapping.get("summary") or ""
        ),
        crash_file_diffstat=diffstat_value if isinstance(diffstat_value, dict) else None,
        recent_commits=recent_commits[:_MAX_REPO_COMMITS],
        warnings=warnings[:6],
        error=str(value.get("error") or ""),
    )


def _parse_history_lessons(value: Any) -> list[HistoryLessonV1]:
    if not isinstance(value, list):
        return []
    lessons: list[HistoryLessonV1] = []
    for row in value:
        if not isinstance(row, dict):
            continue
        lesson = str(row.get("lesson") or "").strip()
        if not lesson:
            continue
        lessons.append(
            HistoryLessonV1(
                source=str(row.get("source") or ""),
                attempt=_safe_int(row.get("attempt")),
                lesson=lesson,
            )
        )
    return lessons


def _parse_evidence_items(value: Any) -> list[EvidenceItemV1]:
    if not isinstance(value, list):
        return []
    out: list[EvidenceItemV1] = []
    for row in value:
        if not isinstance(row, dict):
            continue
        evidence_id = str(row.get("id") or "").strip()
        if not evidence_id:
            continue
        score_raw = row.get("score")
        score = float(score_raw) if isinstance(score_raw, (int, float)) else 0.0
        out.append(
            EvidenceItemV1(
                id=evidence_id,
                kind=str(row.get("kind") or ""),
                locator=str(row.get("locator") or ""),
                summary=str(row.get("summary") or ""),
                path=str(row.get("path")) if isinstance(row.get("path"), str) else None,
                line_span=_coerce_line_span(row.get("line_span")),
                score=score,
                provenance=str(row.get("provenance") or ""),
                raw_ref=str(row.get("raw_ref")) if isinstance(row.get("raw_ref"), str) else None,
            )
        )
    return out


def parse_controller_packet(value: Any) -> ControllerPacketV1 | None:
    """Best-effort parser for controller packets loaded from JSON sidecars.

    If the packet's ``schema_version`` does not match ``_SCHEMA_VERSION``,
    a warning is logged but parsing continues on a best-effort basis.
    """
    if not isinstance(value, dict):
        return None

    # Warn on version mismatch but attempt best-effort parsing.
    packet_version = str(value.get("schema_version") or "").strip()
    if packet_version and packet_version != _SCHEMA_VERSION:
        logger.warning(
            "controller packet schema_version %r != expected %r; parsing continues best-effort",
            packet_version,
            _SCHEMA_VERSION,
        )

    state_value = value.get("controller_state")
    evidence_value = value.get("evidence_bundle")
    history_value = value.get("history_bundle")
    verifier_value = value.get("verifier_report")
    if not (
        isinstance(state_value, dict)
        and isinstance(evidence_value, dict)
        and isinstance(history_value, dict)
        and isinstance(verifier_value, dict)
    ):
        return None

    evidence_bundle = EvidenceBundleV1(
        schema_version=str(evidence_value.get("schema_version") or ""),
        case_id=str(evidence_value.get("case_id") or ""),
        condition=str(evidence_value.get("condition") or ""),
        attempt=max(0, int(evidence_value.get("attempt") or 0)),
        failure_signature=str(evidence_value.get("failure_signature") or ""),
        selected_ids=[
            str(item) for item in evidence_value.get("selected_ids", []) if str(item).strip()
        ],
        new_ids=[str(item) for item in evidence_value.get("new_ids", []) if str(item).strip()],
        reused_ids=[
            str(item) for item in evidence_value.get("reused_ids", []) if str(item).strip()
        ],
        ast_context=_parse_ast_context(evidence_value.get("ast_context")),
        snapshot_reduction=_parse_snapshot_reduction(evidence_value.get("snapshot_reduction")),
        summary=_parse_evidence_summary(evidence_value.get("summary")),
        items=_parse_evidence_items(evidence_value.get("items")),
        snapshot_full_text=str(evidence_value.get("snapshot_full_text") or ""),
    )
    history_bundle = HistoryBundleV1(
        schema_version=str(history_value.get("schema_version") or ""),
        retry_context_mode=str(history_value.get("retry_context_mode") or ""),
        attempt_turns=[
            str(item) for item in history_value.get("attempt_turns", []) if str(item).strip()
        ],
        retry_history_text=str(history_value.get("retry_history_text") or ""),
        retry_history_turns_included=max(
            0, int(history_value.get("retry_history_turns_included") or 0)
        ),
        retry_history_truncated=bool(history_value.get("retry_history_truncated")),
        previous_note=str(history_value.get("previous_note") or ""),
        previous_failure_signature=str(history_value.get("previous_failure_signature") or ""),
        previous_patch_verifier_rejected=bool(
            history_value.get("previous_patch_verifier_rejected")
        ),
        previous_patch_verifier_reasons=[
            str(item)
            for item in history_value.get("previous_patch_verifier_reasons", [])
            if str(item).strip()
        ],
        lessons=_parse_history_lessons(history_value.get("lessons")),
        repo_history=_parse_repo_history(history_value.get("repo_history")),
        persistent_session_id=str(history_value.get("persistent_session_id") or ""),
        persistent_records_loaded=max(0, int(history_value.get("persistent_records_loaded") or 0)),
        persistent_history_error=str(history_value.get("persistent_history_error") or ""),
    )
    verifier_report = VerifierReportV1(
        schema_version=str(verifier_value.get("schema_version") or ""),
        status=str(verifier_value.get("status") or ""),
        scope=str(verifier_value.get("scope") or ""),
        reasons=[str(item) for item in verifier_value.get("reasons", []) if str(item).strip()],
        previous_timed_out=bool(verifier_value.get("previous_timed_out")),
        previous_timeout_sec=_safe_int(verifier_value.get("previous_timeout_sec")),
        syntax_cycle=max(0, int(verifier_value.get("syntax_cycle") or 0)),
        syntax_max_cycles=max(0, int(verifier_value.get("syntax_max_cycles") or 0)),
        last_error_type=str(verifier_value.get("last_error_type") or ""),
        last_error_msg=str(verifier_value.get("last_error_msg") or ""),
        retry_feedback=str(verifier_value.get("retry_feedback") or ""),
    )
    controller_state = AdaptiveControllerStateV2(
        schema_version=str(state_value.get("schema_version") or ""),
        helper_action_selected=str(state_value.get("helper_action_selected") or ""),
        helper_action_executed=str(state_value.get("helper_action_executed") or ""),
        helper_decision_reason=str(state_value.get("helper_decision_reason") or ""),
        helper_outcome=str(state_value.get("helper_outcome") or ""),
        helper_materialized_context=bool(state_value.get("helper_materialized_context")),
        helper_calls_used=max(0, int(state_value.get("helper_calls_used") or 0)),
        helper_budget_max_calls=max(0, int(state_value.get("helper_budget_max_calls") or 0)),
        no_progress_streak=max(0, int(state_value.get("no_progress_streak") or 0)),
        retry_prompt_mode=str(state_value.get("retry_prompt_mode") or ""),
        retry_context_mode=str(state_value.get("retry_context_mode") or ""),
        context_compaction_requested=bool(state_value.get("context_compaction_requested")),
        context_compaction_invoked=bool(state_value.get("context_compaction_invoked")),
        context_compaction_success=bool(state_value.get("context_compaction_success")),
        context_compaction_trigger=str(state_value.get("context_compaction_trigger") or ""),
        prompt_budget={
            str(key): max(0, int(item))
            for key, item in state_value.get("prompt_budget", {}).items()
        }
        if isinstance(state_value.get("prompt_budget"), dict)
        else {},
    )
    solver_hints_value = value.get("solver_hints")
    solver_hints_mapping = solver_hints_value if isinstance(solver_hints_value, dict) else {}
    decomposition_raw = solver_hints_mapping.get("decomposition")
    decomposition: list[dict[str, Any]] = []
    if isinstance(decomposition_raw, list):
        for row in decomposition_raw:
            if not isinstance(row, dict):
                continue
            title = str(row.get("title") or "").strip()
            question = str(row.get("question") or "").strip()
            if not title or not question:
                continue
            supporting_routes_raw = row.get("supporting_routes")
            evidence_ids_raw = row.get("evidence_ids")
            decomposition.append(
                {
                    "title": title,
                    "question": question,
                    "supporting_routes": [
                        str(item) for item in supporting_routes_raw if str(item).strip()
                    ]
                    if isinstance(supporting_routes_raw, list)
                    else [],
                    "evidence_ids": [str(item) for item in evidence_ids_raw if str(item).strip()]
                    if isinstance(evidence_ids_raw, list)
                    else [],
                }
            )
    solver_hints = SolverHintsV1(
        tool_routes=[
            str(item) for item in solver_hints_mapping.get("tool_routes", []) if str(item).strip()
        ],
        retry_plan=str(solver_hints_mapping.get("retry_plan") or ""),
        recommended_focus=[
            str(item)
            for item in solver_hints_mapping.get("recommended_focus", [])
            if str(item).strip()
        ],
        decomposition=decomposition,
        strategy_summary=str(solver_hints_mapping.get("strategy_summary") or ""),
    )
    return ControllerPacketV1(
        schema_version=str(value.get("schema_version") or ""),
        case_id=str(value.get("case_id") or ""),
        condition=str(value.get("condition") or ""),
        attempt=max(0, int(value.get("attempt") or 0)),
        controller_state=controller_state,
        evidence_bundle=evidence_bundle,
        history_bundle=history_bundle,
        verifier_report=verifier_report,
        solver_hints=solver_hints,
    )


def _crash_frame(snapshot: dict[str, Any] | None) -> dict[str, Any] | None:
    if not isinstance(snapshot, dict):
        return None
    frames_raw = snapshot.get("frames")
    if not isinstance(frames_raw, list):
        return None
    if not frames_raw:
        return None
    crash_idx = coerce_crash_frame_index(snapshot)
    if 0 <= crash_idx < len(frames_raw):
        frame = frames_raw[crash_idx]
        if isinstance(frame, dict):
            return frame
    for frame in frames_raw:
        if isinstance(frame, dict):
            return frame
    return None


def _safe_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _ast_context_from_snapshot(
    *, root: Path, snapshot: dict[str, Any] | None
) -> ASTContextV1 | None:
    frame = _crash_frame(snapshot)
    if not isinstance(frame, dict):
        return None
    rel_path = str(frame.get("file_rel") or frame.get("file") or "").strip()
    line_no = _safe_int(frame.get("line"))
    if not rel_path:
        return None
    source_path = root / rel_path
    if not source_path.is_file():
        return None
    try:
        source = source_path.read_text(encoding="utf-8")
    except OSError:
        return None
    if not source.strip():
        return None
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return ASTContextV1(
            path=rel_path,
            crash_line=line_no,
            symbol_name="",
            symbol_kind="",
            node_type="SyntaxError",
            line_span=None,
            snippet=_truncate(source, max_chars=800),
            related_names=[],
        )

    line_target = line_no if isinstance(line_no, int) and line_no > 0 else 1
    enclosing_nodes: list[ast.AST] = []
    for node in ast.walk(tree):
        start = _safe_int(getattr(node, "lineno", None))
        end = _safe_int(getattr(node, "end_lineno", None))
        if start is None or end is None:
            continue
        if start <= line_target <= end:
            enclosing_nodes.append(node)

    if not enclosing_nodes:
        return None

    preferred_types = (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)
    target = min(
        enclosing_nodes,
        key=lambda node: (
            0 if isinstance(node, preferred_types) else 1,
            (_safe_int(getattr(node, "end_lineno", None)) or line_target)
            - (_safe_int(getattr(node, "lineno", None)) or line_target),
        ),
    )
    start = _safe_int(getattr(target, "lineno", None))
    end = _safe_int(getattr(target, "end_lineno", None))
    if start is None or end is None:
        return None

    lines = source.splitlines()
    snippet = "\n".join(lines[start - 1 : end][:_MAX_AST_SNIPPET_LINES]).rstrip()
    symbol_name = str(getattr(target, "name", "") or "")
    symbol_kind = ""
    if isinstance(target, ast.FunctionDef):
        symbol_kind = "function"
    elif isinstance(target, ast.AsyncFunctionDef):
        symbol_kind = "async_function"
    elif isinstance(target, ast.ClassDef):
        symbol_kind = "class"
    related_names = sorted(
        {
            node.id
            for node in ast.walk(target)
            if isinstance(node, ast.Name) and isinstance(node.id, str) and node.id.strip()
        }
    )[:8]
    return ASTContextV1(
        path=rel_path,
        crash_line=line_no,
        symbol_name=symbol_name,
        symbol_kind=symbol_kind,
        node_type=type(target).__name__,
        line_span=(start, end),
        snippet=snippet,
        related_names=related_names,
    )


def _snapshot_reduction_from_snapshot(
    snapshot: dict[str, Any] | None,
) -> SnapshotReductionV1 | None:
    if not isinstance(snapshot, dict):
        return None
    frames_raw = snapshot.get("frames")
    if not isinstance(frames_raw, list) or not frames_raw:
        return None
    frame_count = sum(1 for frame in frames_raw if isinstance(frame, dict))
    if frame_count == 0:
        return None
    crash = _crash_frame(snapshot)
    if crash is None:
        return None
    ordered_frames = [crash]
    ordered_frames.extend(
        frame for frame in frames_raw if isinstance(frame, dict) and frame is not crash
    )
    selected_frames = ordered_frames[: min(3, len(ordered_frames))]
    crash_file = str(crash.get("file_rel") or crash.get("file") or "?")
    crash_line = _safe_int(crash.get("line"))
    crash_func = str(crash.get("function") or "?")
    crash_code = _truncate(str(crash.get("code") or ""), max_chars=160)
    frame_summaries = [
        {
            "path": str(frame.get("file_rel") or frame.get("file") or "?"),
            "line": _safe_int(frame.get("line")),
            "function": str(frame.get("function") or "?"),
            "code": _truncate(str(frame.get("code") or ""), max_chars=160),
        }
        for frame in selected_frames
    ]
    locals_dict = crash.get("locals")
    locals_out: list[dict[str, str]] = []
    if isinstance(locals_dict, dict):
        for key, value in list(locals_dict.items())[:_MAX_LOCALS]:
            key_str = str(key)
            if key_str.startswith("@"):
                continue
            rendered = _truncate(json.dumps(value, ensure_ascii=False, default=str), max_chars=200)
            locals_out.append({"name": key_str, "value": rendered})
    crash_summary = f"{crash_file}:{crash_line or '?'} in {crash_func}()"
    if crash_code:
        crash_summary = f"{crash_summary} -> {crash_code}"
    return SnapshotReductionV1(
        selected_frame_count=len(selected_frames),
        omitted_frame_count=max(0, frame_count - len(selected_frames)),
        crash_frame_summary=crash_summary,
        selected_frames=frame_summaries,
        selected_locals=locals_out,
    )


def _summary_from_evidence_text(text: str) -> str:
    lines = [line.strip() for line in str(text or "").splitlines() if line.strip()]
    if lines and lines[0].startswith("[") and "]" in lines[0]:
        lines = lines[1:]
    return _truncate(" ".join(lines[:3]), max_chars=_MAX_EVIDENCE_SUMMARY_CHARS)


def _line_span_from_raw_item(item: dict[str, Any]) -> tuple[int, int] | None:
    line_value = item.get("line")
    if isinstance(line_value, int) and not isinstance(line_value, bool) and line_value > 0:
        return (line_value, line_value)
    locator = str(item.get("locator") or "")
    if ":" not in locator:
        return None
    tail = locator.rsplit(":", 1)[-1]
    if tail.isdigit():
        line_no = int(tail)
        return (line_no, line_no) if line_no > 0 else None
    return None


def _evidence_kind_priority(kind: str) -> int:
    normalized = str(kind or "").strip().lower()
    if normalized in {"frame", "trace_frame"}:
        return 0
    if normalized in {"source_span", "file", "file_only"}:
        return 1
    if normalized in {"search_hit", "search_only"}:
        return 2
    return 3


def _history_value_from_turn(turn: str, prefix: str) -> str:
    needle = f"{prefix}:"
    for raw_line in str(turn or "").splitlines():
        line = raw_line.strip()
        if line.startswith(needle):
            return line[len(needle) :].strip()
    return ""


def _history_attempt_from_turn(turn: str) -> int | None:
    lines = [line.strip() for line in str(turn or "").splitlines() if line.strip()]
    if not lines:
        return None
    first = lines[0]
    if not first.startswith("Attempt "):
        return None
    attempt_text = first[len("Attempt ") :].strip()
    return int(attempt_text) if attempt_text.isdigit() else None


def _evidence_items_from_raw(
    *,
    evidence_items: list[dict[str, Any]],
    new_ids: list[str],
    reused_ids: list[str],
) -> list[EvidenceItemV1]:
    new_id_set = {str(item) for item in new_ids}
    reused_id_set = {str(item) for item in reused_ids}
    raw_items: list[EvidenceItemV1] = []
    for raw in evidence_items:
        if not isinstance(raw, dict):
            continue
        evidence_id = str(raw.get("id") or "").strip()
        if not evidence_id:
            continue
        score = 0.5
        if evidence_id in new_id_set:
            score = 1.0
        elif evidence_id in reused_id_set:
            score = 0.7
        raw_items.append(
            EvidenceItemV1(
                id=evidence_id,
                kind=str(raw.get("type") or "evidence"),
                locator=str(raw.get("locator") or raw.get("path") or evidence_id),
                summary=_summary_from_evidence_text(str(raw.get("text") or "")),
                path=str(raw.get("path")) if isinstance(raw.get("path"), str) else None,
                line_span=_line_span_from_raw_item(raw),
                score=score,
                provenance="adaptive_helper",
                raw_ref=evidence_id,
            )
        )
    deduped: list[EvidenceItemV1] = []
    seen_keys: set[tuple[str, str, str]] = set()
    for item in sorted(
        raw_items,
        key=lambda row: (
            -float(row.score),
            _evidence_kind_priority(row.kind),
            len(row.summary or ""),
            row.id,
        ),
    ):
        identity = (item.kind, item.locator, item.summary)
        if identity in seen_keys:
            continue
        seen_keys.add(identity)
        deduped.append(item)
    return deduped[:_MAX_SELECTED_EVIDENCE_ITEMS]


def _build_evidence_summary(
    *,
    ast_context: ASTContextV1 | None,
    snapshot_reduction: SnapshotReductionV1 | None,
    items: list[EvidenceItemV1],
    total_item_count: int,
) -> EvidenceSummaryV1 | None:
    top_facts: list[str] = []
    crash_anchor = ""
    if ast_context is not None:
        crash_anchor = f"{ast_context.path}:{ast_context.crash_line or '?'}"
        if ast_context.symbol_name:
            crash_anchor = f"{crash_anchor} in {ast_context.symbol_name}"
        top_facts.append(
            _truncate(
                (
                    f"Crash is localized to {ast_context.symbol_kind or ast_context.node_type} "
                    f"{ast_context.symbol_name or ast_context.path}"
                ).strip(),
                max_chars=140,
            )
        )
    if snapshot_reduction is not None and snapshot_reduction.crash_frame_summary:
        if not crash_anchor:
            crash_anchor = snapshot_reduction.crash_frame_summary
        top_facts.append(_truncate(snapshot_reduction.crash_frame_summary, max_chars=180))
    local_names: list[str] = []
    if snapshot_reduction is not None:
        local_names = [
            str(row.get("name") or "")
            for row in snapshot_reduction.selected_locals
            if isinstance(row, dict) and str(row.get("name") or "").strip()
        ][:_MAX_LOCAL_FACTS]
        if local_names:
            top_facts.append(
                _truncate(
                    "Relevant locals: " + ", ".join(local_names),
                    max_chars=140,
                )
            )
    for item in items[:_MAX_EVIDENCE_FACTS]:
        if item.summary:
            label = item.kind or "evidence"
            locator = item.locator or item.path or item.id
            top_facts.append(_truncate(f"{label} {locator}: {item.summary}".strip(), max_chars=180))
    unique_facts: list[str] = []
    seen_facts: set[str] = set()
    for fact in top_facts:
        if not fact or fact in seen_facts:
            continue
        seen_facts.add(fact)
        unique_facts.append(fact)
    if not unique_facts and not crash_anchor:
        return None
    omitted_items = max(0, int(total_item_count) - len(items))
    omitted_frames = snapshot_reduction.omitted_frame_count if snapshot_reduction is not None else 0
    return EvidenceSummaryV1(
        strategy="deterministic_extract_v1",
        crash_anchor=_truncate(crash_anchor, max_chars=160),
        top_facts=unique_facts[:_MAX_EVIDENCE_FACTS],
        selected_local_names=local_names[:_MAX_LOCAL_FACTS],
        omitted_item_count=omitted_items,
        omitted_frame_count=max(0, int(omitted_frames)),
    )


def _build_history_lessons(
    *,
    retry_history_turns: list[str],
    previous_note: str,
    previous_patch_verifier_reasons: list[str],
    persistent_lessons: list[HistoryLessonV1],
) -> list[HistoryLessonV1]:
    lessons: list[HistoryLessonV1] = []
    for turn in reversed(retry_history_turns[-_MAX_HISTORY_TURNS:]):
        lesson = (
            _history_value_from_turn(turn, "New constraint")
            or _history_value_from_turn(turn, "Outcome")
            or _history_value_from_turn(turn, "Hypothesis tested")
        )
        if not lesson:
            lesson = _truncate(str(turn), max_chars=_MAX_HISTORY_LESSON_CHARS)
        if not lesson:
            continue
        lessons.append(
            HistoryLessonV1(
                source="attempt_history",
                attempt=_history_attempt_from_turn(turn),
                lesson=_truncate(lesson, max_chars=_MAX_HISTORY_LESSON_CHARS),
            )
        )
    if previous_patch_verifier_reasons:
        lessons.append(
            HistoryLessonV1(
                source="patch_verifier",
                attempt=None,
                lesson=_truncate(
                    "Verifier constraint: "
                    + ", ".join(str(item) for item in previous_patch_verifier_reasons[:3]),
                    max_chars=_MAX_HISTORY_LESSON_CHARS,
                ),
            )
        )
    elif str(previous_note or "").strip():
        lessons.append(
            HistoryLessonV1(
                source="previous_note",
                attempt=None,
                lesson=_truncate(str(previous_note), max_chars=_MAX_HISTORY_LESSON_CHARS),
            )
        )
    lessons.extend(persistent_lessons[:_MAX_HISTORY_LESSONS])
    deduped: list[HistoryLessonV1] = []
    seen_lessons: set[tuple[str, str]] = set()
    for lesson in lessons:
        identity = (lesson.source, lesson.lesson)
        if identity in seen_lessons:
            continue
        seen_lessons.add(identity)
        deduped.append(lesson)
    return deduped[:_MAX_HISTORY_LESSONS]


def _compact_repo_history(
    *,
    snapshot: dict[str, Any] | None,
    attempt_dir: Path,
) -> RepoHistoryV1 | None:
    if not isinstance(snapshot, dict):
        return None
    try:
        enhanced = get_enhanced_git_context(snapshot, out_dir=str(attempt_dir), max_commits=5)
    except Exception as exc:
        return RepoHistoryV1(
            repo_root="",
            head_branch="",
            head_commit="",
            crash_file_path="",
            crash_file_line=None,
            crash_file_blame_summary="",
            crash_file_diffstat=None,
            recent_commits=[],
            warnings=[],
            error=f"{type(exc).__name__}: {exc}",
        )
    if not isinstance(enhanced, dict):
        return None
    recent_commits_raw = enhanced.get("recent_commits")
    recent_commits = (
        [row for row in recent_commits_raw if isinstance(row, dict)][:_MAX_REPO_COMMITS]
        if isinstance(recent_commits_raw, list)
        else []
    )
    compacted: dict[str, Any] = {}
    for key in ("repo_root", "head", "crash_file", "crash_file_diffstat"):
        value = enhanced.get(key)
        if value is not None:
            compacted[key] = value
    if recent_commits:
        compacted["recent_commits"] = recent_commits
    warnings = enhanced.get("_git_warnings")
    if isinstance(warnings, list) and warnings:
        compacted["_git_warnings"] = [str(item) for item in warnings[:6]]
    return _parse_repo_history(compacted) if compacted else None


def _build_solver_decomposition(
    *,
    evidence_bundle: EvidenceBundleV1,
    history_lessons: list[HistoryLessonV1],
    repo_history: RepoHistoryV1 | None,
    verifier_report: VerifierReportV1,
    retry_plan_text: str,
) -> list[dict[str, Any]]:
    decomposition: list[dict[str, Any]] = []
    evidence_summary = evidence_bundle.summary
    crash_anchor = evidence_summary.crash_anchor if evidence_summary is not None else ""
    evidence_ids = [item.id for item in evidence_bundle.items[:4] if item.id]
    if (
        crash_anchor
        or evidence_bundle.ast_context is not None
        or evidence_bundle.snapshot_reduction is not None
    ):
        decomposition.append(
            {
                "title": "localize_fix",
                "question": _truncate(
                    (
                        f"Identify the minimal behavior change required at {crash_anchor}."
                        if crash_anchor
                        else "Identify the minimal behavior change required at the crash site."
                    ),
                    max_chars=180,
                ),
                "supporting_routes": [
                    route
                    for route in ("evidence_summary", "ast_context", "snapshot_reduction")
                    if (
                        (route == "evidence_summary" and evidence_summary is not None)
                        or (route == "ast_context" and evidence_bundle.ast_context is not None)
                        or (
                            route == "snapshot_reduction"
                            and evidence_bundle.snapshot_reduction is not None
                        )
                    )
                ],
                "evidence_ids": evidence_ids[:2],
            }
        )
    if history_lessons or repo_history is not None:
        decomposition.append(
            {
                "title": "preserve_constraints",
                "question": _truncate(
                    "Preserve the existing contract implied by recent failures, verifier feedback, and repository history.",
                    max_chars=180,
                ),
                "supporting_routes": [
                    route
                    for route in ("history_lessons", "repo_history")
                    if (route == "history_lessons" and history_lessons)
                    or (route == "repo_history" and repo_history is not None)
                ],
                "evidence_ids": evidence_ids[2:4],
            }
        )
    if verifier_report.status != "none" or retry_plan_text.strip():
        decomposition.append(
            {
                "title": "avoid_repeat_failures",
                "question": _truncate(
                    "Use verifier and retry feedback to avoid repeating prior invalid or irrelevant edits.",
                    max_chars=180,
                ),
                "supporting_routes": [
                    route
                    for route in ("retry_plan", "verifier_report")
                    if (route == "retry_plan" and retry_plan_text.strip())
                    or (route == "verifier_report" and verifier_report.status != "none")
                ],
                "evidence_ids": evidence_ids[:2],
            }
        )
    return decomposition[:3]


def build_adaptive_controller_sidecars(
    *,
    case_id: str,
    condition: str,
    attempt: int,
    root: Path,
    attempt_dir: Path,
    snapshot: dict[str, Any] | None,
    failure_signature: str,
    evidence_items: list[dict[str, Any]],
    new_evidence_ids: list[str],
    reused_evidence_ids: list[str],
    retry_prompt_mode: str,
    retry_context_mode: str,
    retry_history_text: str,
    retry_history_turns: list[str],
    retry_history_turns_included: int,
    retry_history_truncated: bool,
    previous_failure_signature: str,
    previous_note: str,
    previous_patch_verifier_rejected: bool,
    previous_patch_verifier_reasons: list[str],
    previous_patch_verifier_scope: str,
    previous_timed_out: bool,
    previous_timeout_sec: int | None,
    helper_action_selected: str,
    helper_action_executed: str,
    helper_decision_reason: str,
    helper_outcome: str,
    helper_materialized_context: bool,
    helper_calls_used: int,
    helper_budget_max_calls: int,
    no_progress_streak: int,
    context_compaction_requested: bool,
    context_compaction_invoked: bool,
    context_compaction_success: bool,
    context_compaction_trigger: str,
    prompt_budget: dict[str, int],
    retry_plan_text: str,
    syntax_cycle: int,
    syntax_max_cycles: int,
    syntax_last_error_type: str,
    syntax_last_error_msg: str,
    syntax_retry_feedback: str,
    external_decomposition: list[dict[str, Any]] | None = None,
    external_decomposition_source: str = "",
    external_strategy_summary: str = "",
) -> dict[str, dict[str, Any]]:
    """Build the controller sidecar payloads written for one attempt.

    The returned mapping includes the top-level controller packet plus the
    evidence, history, and verifier sub-packets, all keyed by their canonical
    sidecar filenames.
    """
    ast_context = _ast_context_from_snapshot(root=root, snapshot=snapshot)
    snapshot_reduction = _snapshot_reduction_from_snapshot(snapshot)
    snapshot_full_text = ""
    if isinstance(snapshot, dict):
        snapshot_full_text = json.dumps(snapshot, indent=2, ensure_ascii=False)
    raw_evidence_count = sum(1 for row in evidence_items if isinstance(row, dict))
    typed_items = _evidence_items_from_raw(
        evidence_items=evidence_items,
        new_ids=new_evidence_ids,
        reused_ids=reused_evidence_ids,
    )
    evidence_summary = _build_evidence_summary(
        ast_context=ast_context,
        snapshot_reduction=snapshot_reduction,
        items=typed_items,
        total_item_count=raw_evidence_count,
    )
    evidence_bundle = EvidenceBundleV1(
        schema_version=_SCHEMA_VERSION,
        case_id=case_id,
        condition=condition,
        attempt=attempt,
        failure_signature=failure_signature,
        selected_ids=[item.id for item in typed_items],
        new_ids=[str(item) for item in new_evidence_ids],
        reused_ids=[str(item) for item in reused_evidence_ids],
        ast_context=ast_context,
        snapshot_reduction=snapshot_reduction,
        summary=evidence_summary,
        items=typed_items,
        snapshot_full_text=snapshot_full_text,
    )
    persistent_history = load_persistent_adaptive_history(
        out_dir=attempt_dir.parent,
        snapshot=snapshot,
        fallback_session_id=f"{case_id}:{condition}",
        max_records=6,
        max_lessons=_MAX_HISTORY_LESSONS,
        max_chars=_MAX_HISTORY_LESSON_CHARS,
    )
    persistent_lessons = [
        HistoryLessonV1(
            source=lesson.source,
            attempt=lesson.attempt,
            lesson=lesson.lesson,
        )
        for lesson in persistent_history.lessons
    ]
    history_lessons = _build_history_lessons(
        retry_history_turns=retry_history_turns,
        previous_note=previous_note,
        previous_patch_verifier_reasons=previous_patch_verifier_reasons,
        persistent_lessons=persistent_lessons,
    )
    history_bundle = HistoryBundleV1(
        schema_version=_SCHEMA_VERSION,
        retry_context_mode=retry_context_mode,
        attempt_turns=[
            _truncate(str(turn), max_chars=600)
            for turn in retry_history_turns[-_MAX_HISTORY_TURNS:]
        ],
        retry_history_text=_truncate(retry_history_text, max_chars=4_000),
        retry_history_turns_included=max(0, int(retry_history_turns_included)),
        retry_history_truncated=bool(retry_history_truncated),
        previous_note=str(previous_note or ""),
        previous_failure_signature=str(previous_failure_signature or ""),
        previous_patch_verifier_rejected=bool(previous_patch_verifier_rejected),
        previous_patch_verifier_reasons=[str(item) for item in previous_patch_verifier_reasons[:8]],
        lessons=history_lessons,
        repo_history=_compact_repo_history(snapshot=snapshot, attempt_dir=attempt_dir),
        persistent_session_id=persistent_history.session_id,
        persistent_records_loaded=persistent_history.records_loaded,
        persistent_history_error=persistent_history.error,
    )
    verifier_status = "none"
    if previous_patch_verifier_rejected:
        verifier_status = "previous_verifier_rejected"
    if syntax_last_error_type or syntax_last_error_msg:
        verifier_status = "syntax_retry_feedback"
    verifier_report = VerifierReportV1(
        schema_version=_SCHEMA_VERSION,
        status=verifier_status,
        scope=str(previous_patch_verifier_scope or ""),
        reasons=[str(item) for item in previous_patch_verifier_reasons[:8]],
        previous_timed_out=bool(previous_timed_out),
        previous_timeout_sec=previous_timeout_sec,
        syntax_cycle=max(0, int(syntax_cycle)),
        syntax_max_cycles=max(0, int(syntax_max_cycles)),
        last_error_type=str(syntax_last_error_type or ""),
        last_error_msg=_truncate(str(syntax_last_error_msg or ""), max_chars=800),
        retry_feedback=_truncate(
            str(syntax_retry_feedback or ""), max_chars=_MAX_RETRY_FEEDBACK_CHARS
        ),
    )
    controller_state = AdaptiveControllerStateV2(
        schema_version="v2",
        helper_action_selected=str(helper_action_selected or ""),
        helper_action_executed=str(helper_action_executed or ""),
        helper_decision_reason=str(helper_decision_reason or ""),
        helper_outcome=str(helper_outcome or ""),
        helper_materialized_context=bool(helper_materialized_context),
        helper_calls_used=max(0, int(helper_calls_used)),
        helper_budget_max_calls=max(0, int(helper_budget_max_calls)),
        no_progress_streak=max(0, int(no_progress_streak)),
        retry_prompt_mode=str(retry_prompt_mode or ""),
        retry_context_mode=str(retry_context_mode or ""),
        context_compaction_requested=bool(context_compaction_requested),
        context_compaction_invoked=bool(context_compaction_invoked),
        context_compaction_success=bool(context_compaction_success),
        context_compaction_trigger=str(context_compaction_trigger or ""),
        prompt_budget={str(key): max(0, int(value)) for key, value in prompt_budget.items()},
    )
    tool_routes: list[str] = []
    if ast_context is not None:
        tool_routes.append("ast_context")
    if snapshot_reduction is not None:
        tool_routes.append("snapshot_reduction")
    if evidence_summary is not None:
        tool_routes.append("evidence_summary")
    if history_bundle.repo_history:
        tool_routes.append("repo_history")
    if history_lessons:
        tool_routes.append("history_lessons")
    if persistent_history.records_loaded > 0:
        tool_routes.append("persistent_history")
    if history_bundle.retry_history_text:
        tool_routes.append("retry_history")
    if retry_plan_text.strip():
        tool_routes.append("retry_plan")
    if external_strategy_summary.strip():
        tool_routes.append("strategy_summary")
    decomposition = list(external_decomposition or [])
    if not decomposition:
        decomposition = _build_solver_decomposition(
            evidence_bundle=evidence_bundle,
            history_lessons=history_lessons,
            repo_history=history_bundle.repo_history,
            verifier_report=verifier_report,
            retry_plan_text=retry_plan_text,
        )
    if decomposition:
        tool_routes.append(external_decomposition_source.strip() or "lambda_style_decomposition")
    controller_packet = ControllerPacketV1(
        schema_version=_SCHEMA_VERSION,
        case_id=case_id,
        condition=condition,
        attempt=attempt,
        controller_state=controller_state,
        evidence_bundle=evidence_bundle,
        history_bundle=history_bundle,
        verifier_report=verifier_report,
        solver_hints=SolverHintsV1(
            tool_routes=tool_routes,
            retry_plan=_truncate(retry_plan_text, max_chars=_MAX_RETRY_PLAN_CHARS),
            recommended_focus=[
                route
                for route in (
                    "evidence_summary",
                    "ast_context",
                    "snapshot_reduction",
                    "history_lessons",
                    "persistent_history",
                    "repo_history",
                    "retry_history",
                )
                if route in tool_routes
            ],
            decomposition=decomposition,
            strategy_summary=_truncate(
                str(external_strategy_summary or ""),
                max_chars=_MAX_RETRY_FEEDBACK_CHARS,
            ),
        ),
    )
    return {
        ADAPTIVE_CONTROLLER_SIDECAR_FILENAMES[0]: asdict(controller_packet),
        ADAPTIVE_CONTROLLER_SIDECAR_FILENAMES[1]: asdict(evidence_bundle),
        ADAPTIVE_CONTROLLER_SIDECAR_FILENAMES[2]: asdict(history_bundle),
        ADAPTIVE_CONTROLLER_SIDECAR_FILENAMES[3]: asdict(verifier_report),
    }


def remove_adaptive_controller_sidecars(*, context_dir: Path) -> list[Path]:
    """Delete any known controller sidecars from ``context_dir`` and return them."""
    removed: list[Path] = []
    for filename in ADAPTIVE_CONTROLLER_SIDECAR_FILENAMES:
        path = context_dir / filename
        if not path.exists():
            continue
        path.unlink()
        removed.append(path)
    return removed


def write_adaptive_controller_sidecars(
    *,
    context_dir: Path,
    sidecars: dict[str, dict[str, Any]],
) -> list[Path]:
    """Write sidecar JSON files, injecting ``_sidecar_schema_version``."""
    written: list[Path] = []
    for filename, payload in sidecars.items():
        enriched = {**payload, "_sidecar_schema_version": _SCHEMA_VERSION}
        path = context_dir / filename
        tmp_path = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        tmp_path.write_text(json.dumps(enriched, indent=2, ensure_ascii=False), encoding="utf-8")
        tmp_path.replace(path)
        written.append(path)
    return written
