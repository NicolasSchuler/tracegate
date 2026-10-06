"""Adaptive progress signals, helper gating and the traceback informativeness pre-gate."""

from __future__ import annotations

from typing import Any

from evals.eval_patch_constants import TRUNCATED_RESPONSE_ERROR
from evals.eval_types import (
    _ASSERTION_VALUE_PATTERN,
    _DIAGNOSTIC_MESSAGE_PATTERNS,
    _EXCEPTION_TYPE_SPECIFICITY,
    _EXCEPTION_TYPE_SPECIFICITY_DEFAULT,
    _PYTEST_SUMMARY_COUNT_RE,
    _STDOUT_PYTEST_ERROR_RE,
    _STDOUT_TRACEBACK_ERROR_RE,
    ADAPTIVE_HELPER_OUTCOME_COMPAT_REASONING_ONLY,
    ADAPTIVE_HELPER_OUTCOME_INVALID_PAYLOAD,
    ADAPTIVE_HELPER_OUTCOME_MATERIALIZED,
    ADAPTIVE_HELPER_OUTCOME_NOT_INVOKED,
    ADAPTIVE_HELPER_OUTCOME_REQUEST_BUDGET_EXCEEDED,
    ADAPTIVE_HELPER_OUTCOME_SEMANTIC_NO_CONTEXT,
    ADAPTIVE_HELPER_OUTCOME_SKIPPED_GUARDRAIL,
    ADAPTIVE_HELPER_OUTCOME_SKIPPED_POLICY,
    ADAPTIVE_HELPER_OUTCOME_SKIPPED_UNSUPPORTED,
    ADAPTIVE_HELPER_OUTCOME_TRANSPORT_FAILED,
    ADAPTIVE_HELPER_OUTCOME_UNAVAILABLE_BACKEND,
    OPENAI_CONTEXT_WINDOW_RESERVE_TOKENS,
    AdaptiveProgressState,
    PreGateResult,
    RunResult,
)
from evals.native_stage_b_flow import openai_preflight_total_char_cap
from evals.record_keys import (
    KEY_ADAPTIVE_PROGRESS_STOP_NOTE,
    NOTE_FORCED_NOOP_PATCH,
    NOTE_NO_PATCH_PROPOSED,
    NOTE_PATCH_APPLY_FAILED,
    NOTE_PATCH_VERIFIER_REJECTED,
    NOTE_PATCHER_REQUEST_FAILED,
    NOTE_RETRY_GATE_NO_NEW_EVIDENCE,
    NOTE_SUCCESS,
    NOTE_SYNTAX_PREFLIGHT_FAILED,
    NOTE_TESTS_STILL_FAILING,
)
from evals.transport_failures import _request_context_failure_kind_from_meta


def _adaptive_helper_outcome_for_attempt(
    *,
    helper_invoked: bool,
    helper_materialized_context: bool,
    helper_decision_reason: str,
    stage_a_meta: dict[str, Any] | None,
) -> tuple[str, str]:
    decision_reason = str(helper_decision_reason or "").strip()
    failure_kind = _request_context_failure_kind_from_meta(stage_a_meta)
    if helper_invoked:
        if helper_materialized_context:
            return ADAPTIVE_HELPER_OUTCOME_MATERIALIZED, failure_kind
        if failure_kind == "compat_reasoning_only":
            return ADAPTIVE_HELPER_OUTCOME_COMPAT_REASONING_ONLY, failure_kind
        if failure_kind == "backend_unavailable":
            return ADAPTIVE_HELPER_OUTCOME_UNAVAILABLE_BACKEND, failure_kind
        if failure_kind == "transport":
            return ADAPTIVE_HELPER_OUTCOME_TRANSPORT_FAILED, failure_kind
        if failure_kind == "request_budget_exceeded":
            return ADAPTIVE_HELPER_OUTCOME_REQUEST_BUDGET_EXCEEDED, failure_kind
        if failure_kind == "invalid_payload" or decision_reason == "llm_invalid_response_no_helper":
            return ADAPTIVE_HELPER_OUTCOME_INVALID_PAYLOAD, failure_kind or "invalid_payload"
        return ADAPTIVE_HELPER_OUTCOME_SEMANTIC_NO_CONTEXT, failure_kind

    if decision_reason == "request_context_unavailable":
        return ADAPTIVE_HELPER_OUTCOME_SKIPPED_UNSUPPORTED, ""
    if decision_reason in {
        "informativeness_pregate_skip",
        "helper_budget_exhausted",
        "helper_saturation_guard",
    }:
        return ADAPTIVE_HELPER_OUTCOME_SKIPPED_GUARDRAIL, ""
    if decision_reason in {"not_adaptive_condition", ""}:
        return ADAPTIVE_HELPER_OUTCOME_NOT_INVOKED, ""
    return ADAPTIVE_HELPER_OUTCOME_SKIPPED_POLICY, ""


def _adaptive_gate_should_invoke_helper(
    *,
    policy: str,
    attempt: int,
    timed_out: bool,
    signature_repeated: bool,
    stagnation_streak: int,
    stagnation_window: int,
) -> tuple[bool, str]:
    """Decide whether the adaptive gate should invoke the helper.

    Returns (should_invoke, reason_token) based on the gate policy,
    attempt number, timeout state, and stagnation streak.
    """
    if timed_out:
        return True, "gate_timeout"
    if policy == "aggressive":
        return True, "gate_aggressive"

    normalized_window = max(1, int(stagnation_window))
    if policy == "conservative":
        required_repeats = max(normalized_window + 1, 2)
        if signature_repeated and stagnation_streak >= required_repeats:
            return True, "gate_conservative_repeated_signature"
        return False, "gate_conservative_waiting"

    # balanced (default)
    if attempt >= 2 and signature_repeated and stagnation_streak >= normalized_window:
        return True, "gate_balanced_repeated_signature"
    return False, "gate_balanced_waiting"


def _adaptive_progress_signal(
    *,
    note: str = "",
    success: bool = False,
    signature_repeated: bool,
    new_evidence_count: int,
    before_run: RunResult | None = None,
    after_run: RunResult | None = None,
    previous_failure_context: dict[str, Any] | None = None,
    patch_meta: dict[str, Any] | None = None,
) -> AdaptiveProgressState:
    """Compute a multi-signal progress assessment for an attempt.

    Evaluates failure-signature repetition, new-evidence count,
    pytest-failure reduction, pipeline-stage advancement, verifier
    score, and scope alignment to produce an ``AdaptiveProgressState``.
    """
    patch_meta_dict = patch_meta if isinstance(patch_meta, dict) else {}
    previous_context = (
        previous_failure_context if isinstance(previous_failure_context, dict) else {}
    )

    component_signature = 0.0 if signature_repeated else 1.0
    if bool(success):
        component_signature = 1.0
    component_new_evidence = 1.0 if int(new_evidence_count) > 0 else 0.0

    pytest_before_failed_total = _pytest_failed_total(before_run)
    pytest_after_failed_total = _pytest_failed_total(after_run)
    if bool(success) and pytest_after_failed_total is None:
        pytest_after_failed_total = 0

    component_pytest_fail_reduction = 0.0
    if (
        pytest_before_failed_total is not None
        and pytest_after_failed_total is not None
        and pytest_before_failed_total > 0
        and pytest_after_failed_total < pytest_before_failed_total
    ):
        component_pytest_fail_reduction = min(
            1.0,
            max(
                0.0,
                (pytest_before_failed_total - pytest_after_failed_total)
                / float(pytest_before_failed_total),
            ),
        )

    stage_before, stage_before_ord = _adaptive_pipeline_stage_before(
        previous_failure_context=previous_failure_context,
        current_run=before_run,
    )
    stage_after, stage_after_ord = _adaptive_pipeline_stage_for_outcome(
        note=note,
        success=success,
    )
    component_stage_advance = min(
        1.0,
        max(0.0, float(stage_after_ord - stage_before_ord) / 4.0),
    )

    prev_verifier_score = _adaptive_optional_int(previous_context.get("patch_verifier_score"))
    curr_verifier_score = _adaptive_optional_int(patch_meta_dict.get("patch_verifier_score"))
    component_verifier_score = 0.0
    if (
        prev_verifier_score is not None
        and curr_verifier_score is not None
        and curr_verifier_score > prev_verifier_score
    ):
        component_verifier_score = min(
            1.0,
            max(0.0, float(curr_verifier_score - prev_verifier_score) / 4.0),
        )

    prev_scope_matched = _adaptive_optional_bool(
        previous_context.get("patch_verifier_source_scope_matched")
    )
    curr_scope_matched = _adaptive_optional_bool(
        patch_meta_dict.get("patch_verifier_source_scope_matched")
    )
    component_scope_alignment = (
        1.0 if prev_scope_matched is False and curr_scope_matched is True else 0.0
    )

    contributions = {
        "failure_signature_changed": 0.20 * component_signature,
        "new_evidence_added": 0.10 * component_new_evidence,
        "pytest_failure_count_reduction": 0.25 * component_pytest_fail_reduction,
        "pipeline_stage_advance": 0.20 * component_stage_advance,
        "patch_verifier_score_improvement": 0.15 * component_verifier_score,
        "source_scope_alignment_improvement": 0.10 * component_scope_alignment,
    }
    score = min(1.0, max(0.0, sum(contributions.values())))
    reason_order = (
        "failure_signature_changed",
        "pytest_failure_count_reduction",
        "pipeline_stage_advance",
        "patch_verifier_score_improvement",
        "source_scope_alignment_improvement",
        "new_evidence_added",
    )
    reason = "no_progress_v2"
    best_contribution = 0.0
    for reason_key in reason_order:
        contribution = contributions.get(reason_key, 0.0)
        if contribution > best_contribution:
            best_contribution = contribution
            reason = reason_key

    return AdaptiveProgressState(
        made=bool(score > 0.0),
        reason=reason if score > 0.0 else "no_progress_v2",
        score=float(score),
        component_signature=float(component_signature),
        component_new_evidence=float(component_new_evidence),
        component_pytest_fail_reduction=float(component_pytest_fail_reduction),
        component_stage_advance=float(component_stage_advance),
        component_verifier_score=float(component_verifier_score),
        component_scope_alignment=float(component_scope_alignment),
        pytest_before_failed_total=pytest_before_failed_total,
        pytest_after_failed_total=pytest_after_failed_total,
        stage_before=stage_before,
        stage_after=stage_after,
        stage_before_ord=stage_before_ord,
        stage_after_ord=stage_after_ord,
    )


def _adaptive_optional_int(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _adaptive_optional_bool(value: Any) -> bool | None:
    if isinstance(value, bool):
        return value
    return None


def _pytest_failed_total(run: RunResult | None) -> int | None:
    if run is None:
        return None
    for text in (str(run.stdout or ""), str(run.stderr or "")):
        total = _pytest_failed_total_from_text(text)
        if total is not None:
            return total
    if not run.timed_out and int(run.returncode) == 0:
        return 0
    return None


def _pytest_failed_total_from_text(text: str) -> int | None:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    for line in reversed(lines):
        lower = line.lower()
        if "short test summary info" in lower:
            continue
        if "no tests ran" in lower:
            return 0
        if "=" not in line and " in " not in lower:
            continue
        if not any(token in lower for token in (" failed", " error", " errors", " passed")):
            continue
        total = 0
        matched = False
        for match in _PYTEST_SUMMARY_COUNT_RE.finditer(lower):
            matched = True
            total += int(match.group(1))
        if matched:
            return total
        if " passed" in lower:
            return 0
    return None


def _adaptive_pipeline_stage_for_outcome(*, note: str, success: bool) -> tuple[str, int]:
    if bool(success) or str(note or "") == NOTE_SUCCESS:
        return "success", 4
    note_norm = str(note or "").strip()
    if note_norm in {
        NOTE_PATCHER_REQUEST_FAILED,
        NOTE_NO_PATCH_PROPOSED,
        TRUNCATED_RESPONSE_ERROR,
        NOTE_RETRY_GATE_NO_NEW_EVIDENCE,
    }:
        return "request_or_no_patch", 0
    if note_norm in {NOTE_FORCED_NOOP_PATCH, NOTE_PATCH_APPLY_FAILED, NOTE_SYNTAX_PREFLIGHT_FAILED}:
        return "pretest_patch_failure", 1
    if note_norm == NOTE_PATCH_VERIFIER_REJECTED:
        return "verifier_rejected", 2
    return "tests_still_failing", 3


def _adaptive_pipeline_stage_before(
    *,
    previous_failure_context: dict[str, Any] | None,
    current_run: RunResult | None,
) -> tuple[str, int]:
    if isinstance(previous_failure_context, dict):
        previous_note = str(previous_failure_context.get("note") or "")
        if previous_note:
            return _adaptive_pipeline_stage_for_outcome(note=previous_note, success=False)
    if current_run is None:
        return "tests_still_failing", 3
    if current_run.timed_out:
        return "tests_still_failing", 3
    if int(current_run.returncode) == 0:
        return "success", 4
    return "tests_still_failing", 3


def _sync_adaptive_progress_metadata(
    *,
    attempt_metadata: dict[str, Any],
    patch_meta: dict[str, Any],
    progress_state: AdaptiveProgressState,
    no_progress_streak: int,
    recomputed_after_patch: bool = False,
) -> None:
    fields = {
        "adaptive_progress_made": bool(progress_state.made),
        "adaptive_progress_reason": str(progress_state.reason),
        "adaptive_progress_score": float(progress_state.score),
        "adaptive_progress_component_signature": float(progress_state.component_signature),
        "adaptive_progress_component_new_evidence": float(progress_state.component_new_evidence),
        "adaptive_progress_component_pytest_fail_reduction": float(
            progress_state.component_pytest_fail_reduction
        ),
        "adaptive_progress_component_stage_advance": float(progress_state.component_stage_advance),
        "adaptive_progress_component_verifier_score": float(
            progress_state.component_verifier_score
        ),
        "adaptive_progress_component_scope_alignment": float(
            progress_state.component_scope_alignment
        ),
        "adaptive_pytest_before_failed_total": progress_state.pytest_before_failed_total,
        "adaptive_pytest_after_failed_total": progress_state.pytest_after_failed_total,
        "adaptive_pipeline_stage_before": str(progress_state.stage_before),
        "adaptive_pipeline_stage_after": str(progress_state.stage_after),
        "adaptive_pipeline_stage_before_ord": int(progress_state.stage_before_ord),
        "adaptive_pipeline_stage_after_ord": int(progress_state.stage_after_ord),
        "adaptive_no_progress_streak": int(max(0, int(no_progress_streak))),
    }
    attempt_metadata.update(fields)
    patch_meta.update(fields)
    patch_meta["adaptive_progress_recomputed_after_patch"] = bool(recomputed_after_patch)


def _recompute_adaptive_progress_for_attempt_outcome(
    *,
    attempt_metadata: dict[str, Any],
    patch_meta: dict[str, Any],
    note: str,
    success: bool,
    signature_repeated: bool,
    new_evidence_count: int,
    before_run: RunResult | None,
    after_run: RunResult | None,
    previous_failure_context: dict[str, Any] | None,
    current_no_progress_streak: int,
    recomputed_after_patch: bool,
) -> tuple[AdaptiveProgressState, int]:
    progress_state = _adaptive_progress_signal(
        note=note,
        success=success,
        signature_repeated=signature_repeated,
        new_evidence_count=new_evidence_count,
        before_run=before_run,
        after_run=after_run,
        previous_failure_context=previous_failure_context,
        patch_meta=patch_meta,
    )
    next_no_progress_streak = 0 if progress_state.made else max(0, int(current_no_progress_streak))
    _sync_adaptive_progress_metadata(
        attempt_metadata=attempt_metadata,
        patch_meta=patch_meta,
        progress_state=progress_state,
        no_progress_streak=next_no_progress_streak,
        recomputed_after_patch=recomputed_after_patch,
    )
    return progress_state, next_no_progress_streak


def _adaptive_should_stop_for_no_progress(
    *,
    no_progress_streak: int,
    stop_window: int,
    helper_budget_exhausted: bool,
    helper_saturation_guarded: bool,
    helper_decision_reason: str,
) -> tuple[bool, str]:
    if int(no_progress_streak) < max(1, int(stop_window)):
        return False, "streak_below_window"
    if helper_budget_exhausted:
        return True, "helper_budget_exhausted"
    if helper_saturation_guarded:
        return True, "helper_saturation_guard"
    if helper_decision_reason in {
        "llm_declined_helper",
        "llm_invalid_response_no_helper",
        "llm_snapshot_request_denied_attempt1",
        "request_context_unavailable",
    }:
        return True, helper_decision_reason
    return False, "waiting_for_more_signal"


def _mark_adaptive_progress_stop(
    *,
    attempt_metadata: dict[str, Any],
    patch_meta: dict[str, Any],
    is_adaptive_condition: bool,
    adaptive_progress_stop_candidate: bool,
    adaptive_progress_stop_reason: str,
    note: str,
) -> bool:
    triggered = bool(
        is_adaptive_condition
        and adaptive_progress_stop_candidate
        and note
        in {
            "hard_stop_no_progress",
            NOTE_NO_PATCH_PROPOSED,
            TRUNCATED_RESPONSE_ERROR,
            NOTE_PATCHER_REQUEST_FAILED,
            NOTE_FORCED_NOOP_PATCH,
            NOTE_PATCH_VERIFIER_REJECTED,
            NOTE_SYNTAX_PREFLIGHT_FAILED,
            NOTE_PATCH_APPLY_FAILED,
            NOTE_TESTS_STILL_FAILING,
        }
    )
    attempt_metadata["adaptive_progress_stop_triggered"] = triggered
    patch_meta["adaptive_progress_stop_triggered"] = triggered
    patch_meta["adaptive_progress_stop_reason"] = adaptive_progress_stop_reason
    if triggered:
        patch_meta[KEY_ADAPTIVE_PROGRESS_STOP_NOTE] = note
    return triggered


def _apply_no_progress_token_cap_candidate(
    *,
    is_adaptive_condition: bool,
    adaptive_progress_made: bool,
    adaptive_no_progress_token_sum: int,
    attempt_tokens_total: int,
    adaptive_no_progress_token_cap: int,
    adaptive_progress_stop_candidate: bool,
    adaptive_progress_stop_reason: str,
) -> tuple[bool, str, int, bool]:
    next_sum = max(0, int(adaptive_no_progress_token_sum))
    if not is_adaptive_condition:
        return adaptive_progress_stop_candidate, adaptive_progress_stop_reason, next_sum, False
    if adaptive_progress_made:
        next_sum = 0
    else:
        next_sum += max(0, int(attempt_tokens_total))

    cap = max(0, int(adaptive_no_progress_token_cap))
    cap_hit = cap > 0 and next_sum >= cap
    if cap_hit:
        return True, "no_progress_token_cap", next_sum, True
    return adaptive_progress_stop_candidate, adaptive_progress_stop_reason, next_sum, False


def _extract_exception_info(
    *,
    snapshot: dict[str, Any] | None,
    stdout: str,
    stderr: str,
) -> tuple[str, str, int, bool]:
    """Extract (exception_type, message, frame_count, crash_in_user_code)."""
    if snapshot and isinstance(snapshot.get("exception"), dict):
        exc_info = snapshot["exception"]
        exc_type = str(exc_info.get("type", "Unknown"))
        exc_msg = str(exc_info.get("message", ""))
        frames = snapshot.get("frames")
        frame_count = len(frames) if isinstance(frames, list) else 0
        crash_in_user_code = False
        if isinstance(frames, list) and len(frames) > 0:
            crash_frame = frames[0]
            if isinstance(crash_frame, dict):
                crash_in_user_code = crash_frame.get("file_rel") is not None
        # Strip module prefix (e.g., "builtins.ValueError" -> "ValueError")
        if "." in exc_type:
            exc_type = exc_type.rsplit(".", 1)[-1]
        return exc_type, exc_msg, frame_count, crash_in_user_code

    # Fallback: parse from stdout/stderr
    for text in (stdout, stderr):
        if not text:
            continue
        m = _STDOUT_PYTEST_ERROR_RE.search(text)
        if m:
            exc_type = m.group(1).rsplit(".", 1)[-1]
            return exc_type, m.group(2).strip(), 0, False
        m = _STDOUT_TRACEBACK_ERROR_RE.search(text)
        if m:
            exc_type = m.group(1).rsplit(".", 1)[-1]
            return exc_type, m.group(2).strip(), 0, False

    return "Unknown", "", 0, False


def _traceback_informativeness_score(
    *,
    exception_type: str,
    exception_message: str,
    frame_count: int,
    crash_in_user_code: bool,
) -> PreGateResult:
    """Score traceback informativeness and return a PreGateResult with all features."""
    # Signal 1: Exception type specificity (weight 0.30)
    f_type_specificity = _EXCEPTION_TYPE_SPECIFICITY.get(
        exception_type, _EXCEPTION_TYPE_SPECIFICITY_DEFAULT
    )

    # Signal 2: Message contains diagnostic values (weight 0.25)
    f_message_diagnostic = 0.0
    if exception_message:
        for pattern in _DIAGNOSTIC_MESSAGE_PATTERNS:
            if pattern.search(exception_message):
                f_message_diagnostic = 1.0
                break
        if f_message_diagnostic == 0.0:
            f_message_diagnostic = 0.3  # Non-empty but generic
    # else: empty message → 0.0

    # Signal 3: Crash frame in user code (weight 0.15)
    f_crash_in_user_code = 1.0 if crash_in_user_code else 0.0

    # Signal 4: Assertion message quality (weight 0.15)
    if exception_type == "AssertionError":
        if exception_message and _ASSERTION_VALUE_PATTERN.search(exception_message):
            f_assertion_quality = 1.0
        elif exception_message:
            f_assertion_quality = 0.5
        else:
            f_assertion_quality = 0.0
    else:
        f_assertion_quality = 0.5  # Neutral for non-assertions

    # Signal 5: Frame count / depth (weight 0.15)
    # frame_count=0 means no structured data (stdout fallback) — treat as neutral.
    if frame_count == 0:
        f_frame_depth = 0.5
    elif frame_count <= 3:
        f_frame_depth = 1.0
    elif frame_count <= 6:
        f_frame_depth = 0.7
    elif frame_count <= 10:
        f_frame_depth = 0.4
    else:
        f_frame_depth = 0.2

    score = (
        0.30 * f_type_specificity
        + 0.25 * f_message_diagnostic
        + 0.15 * f_crash_in_user_code
        + 0.15 * f_assertion_quality
        + 0.15 * f_frame_depth
    )
    score = min(1.0, max(0.0, score))

    tier = (
        "A"
        if f_type_specificity >= 0.8
        else ("B" if f_type_specificity >= 0.5 else ("C" if f_type_specificity >= 0.3 else "D"))
    )
    reason = f"{exception_type}_tier{tier}"

    return PreGateResult(
        should_skip=False,  # Caller decides based on threshold
        score=score,
        exception_type=exception_type,
        reason=reason,
        f_type_specificity=f_type_specificity,
        f_message_diagnostic=f_message_diagnostic,
        f_crash_in_user_code=f_crash_in_user_code,
        f_assertion_quality=f_assertion_quality,
        f_frame_depth=f_frame_depth,
    )


def _traceback_informativeness_pre_gate(
    *,
    snapshot: dict[str, Any] | None,
    stdout: str,
    stderr: str,
    threshold: float,
) -> PreGateResult:
    """Return a PreGateResult with should_skip set based on threshold."""
    exc_type, exc_msg, frame_count, crash_in_user = _extract_exception_info(
        snapshot=snapshot,
        stdout=stdout,
        stderr=stderr,
    )
    result = _traceback_informativeness_score(
        exception_type=exc_type,
        exception_message=exc_msg,
        frame_count=frame_count,
        crash_in_user_code=crash_in_user,
    )
    return result._replace(should_skip=result.score >= threshold)


def _adaptive_prompt_budget_for_attempt(
    *,
    base_budget: dict[str, int],
    helper_materialized_context: bool,
    profile: str,
) -> dict[str, int]:
    out = dict(base_budget)
    if not helper_materialized_context or profile != "compact":
        return out

    def _cap(value: int, limit: int) -> int:
        if value <= 0:
            return value
        return min(value, limit)

    out["total_chars"] = _cap(int(out.get("total_chars", 0)), 120_000)
    out["retry_delta_chars"] = _cap(int(out.get("retry_delta_chars", 0)), 12_000)
    out["stdout_chars"] = _cap(int(out.get("stdout_chars", 0)), 16_000)
    out["snapshot_chars"] = _cap(int(out.get("snapshot_chars", 0)), 20_000)
    out["evidence_chars"] = _cap(int(out.get("evidence_chars", 0)), 32_000)
    out["retry_history_chars"] = _cap(int(out.get("retry_history_chars", 0)), 8_000)
    return out


def _openai_preflight_total_char_cap(
    *,
    context_window_tokens: int,
    max_tokens: int,
    reserve_tokens: int = OPENAI_CONTEXT_WINDOW_RESERVE_TOKENS,
) -> int:
    return openai_preflight_total_char_cap(
        context_window_tokens=context_window_tokens,
        max_tokens=max_tokens,
        reserve_tokens=reserve_tokens,
    )
