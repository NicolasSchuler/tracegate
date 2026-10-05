"""Adaptive-controller packet contract shared by the eval harness and controller-backed baselines.

The harness builds packets (``evals.adaptive_controller_packet``); baselines parse them
and render hybrid guidance from them. Living in ``baselines`` keeps the dependency
one-way: ``evals`` imports ``baselines``, never the reverse.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


_SCHEMA_VERSION = "v1"


_MAX_REPO_COMMITS = 3


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


def _safe_int(value: Any) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None
