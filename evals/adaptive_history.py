"""Persistent adaptive-history helpers for hybrid controller runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

_MAX_EVIDENCE_REFS = 12


def _truncate(text: str, *, max_chars: int) -> str:
    value = str(text or "").strip()
    if len(value) <= max(0, int(max_chars)):
        return value
    if max_chars <= 3:
        return value[: max(0, int(max_chars))]
    return value[: max(0, int(max_chars)) - 3].rstrip() + "..."


def _coerce_int(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    return None


@dataclass(frozen=True)
class PersistentHistoryLesson:
    source: str
    attempt: int | None
    lesson: str


@dataclass(frozen=True)
class PersistentHistoryBundle:
    session_id: str
    records_loaded: int
    lessons: list[PersistentHistoryLesson]
    error: str = ""


def _derive_session_id(
    *,
    snapshot: dict[str, Any] | None,
    out_dir: Path,
    fallback_session_id: str,
) -> str:
    from llmdebug.rca_state import derive_session_id

    return derive_session_id(
        snapshot,
        out_dir=str(out_dir),
        fallback=str(fallback_session_id or "adaptive_history"),
    )


def _dedupe_evidence_refs(
    new_ids: list[str],
    reused_ids: list[str],
) -> list[str]:
    ordered = [str(item).strip() for item in [*new_ids, *reused_ids]]
    deduped: list[str] = []
    seen: set[str] = set()
    for item in ordered:
        if not item or item in seen:
            continue
        seen.add(item)
        deduped.append(item)
        if len(deduped) >= _MAX_EVIDENCE_REFS:
            break
    return deduped


def persist_adaptive_attempt_history(
    *,
    out_dir: Path,
    snapshot: dict[str, Any] | None,
    fallback_session_id: str,
    record: dict[str, Any],
    patch_meta: dict[str, Any] | None,
    retry_plan_text: str,
) -> str:
    """Persist one adaptive attempt into the RCA store."""
    from llmdebug.rca_state import advance_rca_attempt
    from llmdebug.rca_types import RCAAttemptRequest

    out_dir.mkdir(parents=True, exist_ok=True)
    session_id = _derive_session_id(
        snapshot=snapshot,
        out_dir=out_dir,
        fallback_session_id=fallback_session_id,
    )
    record_meta = patch_meta if isinstance(patch_meta, dict) else {}
    attempt_metadata_raw = record.get("attempt_metadata")
    attempt_metadata = attempt_metadata_raw if isinstance(attempt_metadata_raw, dict) else {}
    retry_summary_raw = attempt_metadata.get("retry_summary")
    retry_summary = retry_summary_raw if isinstance(retry_summary_raw, dict) else {}
    note = str(record.get("note") or "").strip()
    success = bool(record.get("success"))
    hypothesis = str(retry_summary.get("hypothesis_tested") or "").strip()
    experiment_plan = str(retry_plan_text or retry_summary.get("new_constraint") or "").strip()
    proposed_change_summary = ""
    patch_files_raw = record_meta.get("patch_file_paths")
    if isinstance(patch_files_raw, list):
        patch_files = [str(item).strip() for item in patch_files_raw if str(item).strip()]
        if patch_files:
            proposed_change_summary = "Touched files: " + ", ".join(patch_files[:4])
    verification_result: dict[str, Any] = {
        "attempt": int(record.get("attempt") or 0),
        "status": note or ("success" if success else "unknown"),
        "success": success,
        "patch_verifier_rejected": bool(record_meta.get("patch_verifier_enabled"))
        and record_meta.get("patch_verifier_passed") is False,
        "timed_out": bool(record.get("timed_out")),
    }
    patch_verifier_reasons_raw = record_meta.get("patch_verifier_reasons")
    if isinstance(patch_verifier_reasons_raw, list):
        verification_result["patch_verifier_reasons"] = [
            str(item) for item in patch_verifier_reasons_raw[:4] if str(item).strip()
        ]
    gate_reason = record_meta.get("adaptive_stop_trigger_reason")
    if isinstance(gate_reason, str) and gate_reason.strip():
        verification_result["adaptive_stop_reason"] = gate_reason.strip()
    evidence_refs = _dedupe_evidence_refs(
        [str(item) for item in record_meta.get("evidence_ids_new", []) if str(item).strip()]
        if isinstance(record_meta.get("evidence_ids_new"), list)
        else [],
        [str(item) for item in record_meta.get("evidence_ids_reused", []) if str(item).strip()]
        if isinstance(record_meta.get("evidence_ids_reused"), list)
        else [],
    )
    failure_signature = str(
        record_meta.get("adaptive_postpatch_failure_signature")
        or retry_summary.get("failure_signature")
        or record.get("failure_signature")
        or ""
    ).strip()
    if not failure_signature:
        failure_signature = "unknown_failure"
    advance_rca_attempt(
        out_dir=str(out_dir),
        request=RCAAttemptRequest(
            session_id=session_id,
            action="resolve" if success else "verify",
            failure_signature=failure_signature,
            evidence_refs=tuple(evidence_refs),
            hypothesis=hypothesis or None,
            experiment_plan=experiment_plan or None,
            proposed_change_summary=proposed_change_summary or None,
            verification_result=verification_result,
            gate_mode="soft",
            exploratory=False,
        ),
    )
    return session_id


def _lesson_from_record(
    record: dict[str, Any], *, max_chars: int
) -> PersistentHistoryLesson | None:
    verification_raw = record.get("verification_result")
    verification = verification_raw if isinstance(verification_raw, dict) else {}
    status = str(verification.get("status") or "").strip()
    hypothesis = str(record.get("hypothesis") or "").strip()
    experiment_plan = str(record.get("experiment_plan") or "").strip()
    change_summary = str(record.get("proposed_change_summary") or "").strip()
    gate_warnings_raw = record.get("gate_warnings")
    gate_warnings = (
        [str(item).strip() for item in gate_warnings_raw if str(item).strip()]
        if isinstance(gate_warnings_raw, list)
        else []
    )
    detail = ""
    if status and verification.get("patch_verifier_rejected") is True:
        reasons_raw = verification.get("patch_verifier_reasons")
        reasons = (
            ", ".join(str(item) for item in reasons_raw[:3] if str(item).strip())
            if isinstance(reasons_raw, list)
            else ""
        )
        detail = f"{status}: {reasons or 'prior verifier rejection'}"
    elif status and hypothesis:
        detail = f"{status}: {hypothesis}"
    elif status and experiment_plan:
        detail = f"{status}: {experiment_plan}"
    elif hypothesis and experiment_plan:
        detail = f"hypothesis={hypothesis}; plan={experiment_plan}"
    elif hypothesis:
        detail = f"hypothesis={hypothesis}"
    elif experiment_plan:
        detail = f"plan={experiment_plan}"
    elif change_summary:
        detail = change_summary
    elif gate_warnings:
        detail = gate_warnings[0]
    elif status:
        detail = status
    if not detail:
        return None
    return PersistentHistoryLesson(
        source="rca_history",
        attempt=_coerce_int(verification.get("attempt")),
        lesson=_truncate(detail, max_chars=max_chars),
    )


def load_persistent_adaptive_history(
    *,
    out_dir: Path,
    snapshot: dict[str, Any] | None,
    fallback_session_id: str,
    max_records: int = 6,
    max_lessons: int = 4,
    max_chars: int = 180,
) -> PersistentHistoryBundle:
    """Load bounded persistent RCA history for one adaptive case session."""
    if max_records <= 0 or max_lessons <= 0:
        return PersistentHistoryBundle(session_id="", records_loaded=0, lessons=[])
    try:
        from llmdebug.rca_state import get_rca_history
    except Exception as exc:
        return PersistentHistoryBundle(
            session_id="",
            records_loaded=0,
            lessons=[],
            error=f"{type(exc).__name__}: {exc}",
        )
    try:
        session_id = _derive_session_id(
            snapshot=snapshot,
            out_dir=out_dir,
            fallback_session_id=fallback_session_id,
        )
        records = get_rca_history(
            out_dir=str(out_dir),
            session_id=session_id,
            limit=max(1, int(max_records)),
        )
    except Exception as exc:
        return PersistentHistoryBundle(
            session_id="",
            records_loaded=0,
            lessons=[],
            error=f"{type(exc).__name__}: {exc}",
        )
    lessons: list[PersistentHistoryLesson] = []
    seen_lessons: set[str] = set()
    for record in records:
        if not isinstance(record, dict):
            continue
        lesson = _lesson_from_record(record, max_chars=max_chars)
        if lesson is None or lesson.lesson in seen_lessons:
            continue
        seen_lessons.add(lesson.lesson)
        lessons.append(lesson)
        if len(lessons) >= max_lessons:
            break
    return PersistentHistoryBundle(
        session_id=session_id,
        records_loaded=len(records),
        lessons=lessons,
    )


__all__ = [
    "PersistentHistoryBundle",
    "PersistentHistoryLesson",
    "load_persistent_adaptive_history",
    "persist_adaptive_attempt_history",
]
