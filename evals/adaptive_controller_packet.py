"""Typed adaptive-controller sidecars for hybrid eval runs.

These helpers package attempt-scoped controller data for the shared eval
harness. The packet is additive: baseline-only runs can ignore it entirely,
while adaptive conditions and controller-backed baselines can consume the
structured metadata they need.
"""

from __future__ import annotations

import ast
from dataclasses import asdict
from pathlib import Path
from typing import Any
from uuid import uuid4

from baselines.controller_packet import (
    _MAX_REPO_COMMITS,
    _SCHEMA_VERSION,
    AdaptiveControllerStateV2,
    ASTContextV1,
    ControllerPacketV1,
    EvidenceBundleV1,
    EvidenceItemV1,
    EvidenceSummaryV1,
    HistoryBundleV1,
    HistoryLessonV1,
    RepoHistoryV1,
    SnapshotReductionV1,
    SolverHintsV1,
    VerifierReportV1,
    _parse_repo_history,
    _safe_int,
    _truncate,
)
from llmdebug._debug_session import coerce_crash_frame_index
from llmdebug.git_context import get_enhanced_git_context

from . import _json as json
from .adaptive_history import load_persistent_adaptive_history

_MAX_EVIDENCE_SUMMARY_CHARS = 480
_MAX_RETRY_PLAN_CHARS = 2_000
_MAX_RETRY_FEEDBACK_CHARS = 1_200
_MAX_HISTORY_TURNS = 4
_MAX_LOCALS = 8
_MAX_AST_SNIPPET_LINES = 24
_MAX_SELECTED_EVIDENCE_ITEMS = 6
_MAX_EVIDENCE_FACTS = 5
_MAX_LOCAL_FACTS = 4
_MAX_HISTORY_LESSONS = 4
_MAX_HISTORY_LESSON_CHARS = 180
ADAPTIVE_CONTROLLER_SIDECAR_FILENAMES: tuple[str, ...] = (
    "llmdebug_controller_packet.json",
    "llmdebug_evidence_bundle.json",
    "llmdebug_history_bundle.json",
    "llmdebug_verifier_report.json",
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
