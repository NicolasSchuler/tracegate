"""Attempt and infrastructure-event records, retry-history turns and breakdowns."""

from __future__ import annotations

import sys
from typing import Any

from evals.attempt_usage import (
    _baseline_metric_float,
    _nonnegative_int,
    _usage_token_counts_from_patch_meta,
)
from evals.eval_prompts import _build_retry_summary, _render_retry_history_turn
from evals.eval_recording import _build_base_record, _build_model_metadata, _empty_attempt_metadata
from evals.eval_types import EVAL_RECORD_TYPE_INFRA_EVENT, CaseSpec, Condition, RunResult
from evals.pytest_runner import _build_failure_signature_for_attempt, _timeout_fields


def _build_infra_event_record(
    *,
    case: CaseSpec,
    condition: Condition,
    run_id: str,
    attempt: int,
    patcher_name: str,
    run_metadata: dict[str, Any],
    attempt_metadata: dict[str, Any] | None,
    patcher: Any,
    patch_meta: dict[str, Any],
    context_protocol: str,
    before: RunResult,
    cumulative_runtime_sec: float,
    note: str = "infra_transport_failure",
) -> dict[str, Any]:
    materialized_attempt_metadata = (
        dict(attempt_metadata)
        if isinstance(attempt_metadata, dict)
        else _empty_attempt_metadata(attempt=attempt)
    )
    materialized_attempt_metadata["attempt"] = attempt
    record = _build_base_record(
        case=case,
        condition=condition,
        run_id=run_id,
        attempt=attempt,
        patcher_name=patcher_name,
        run_metadata=run_metadata,
        attempt_metadata=materialized_attempt_metadata,
        model_metadata=_build_model_metadata(patcher=patcher, patch_meta=patch_meta),
        context_protocol=context_protocol,
    )
    record["record_type"] = EVAL_RECORD_TYPE_INFRA_EVENT
    record.update(
        {
            "success": False,
            "before_returncode": before.returncode,
            "before_duration_sec": before.duration_sec,
            "cumulative_runtime_sec": cumulative_runtime_sec,
            "attempt_wall_clock_sec": 0.0,
            "note": note,
            "patch_meta": dict(patch_meta),
        }
    )
    record.update(_timeout_fields(before))
    return record


def _append_retry_history_turn_if_enabled(
    *,
    retry_context_mode: str,
    retry_history_turns: list[str],
    attempt: int,
    note: str,
    failure_signature: str,
    patch_meta: dict[str, Any],
) -> None:
    if retry_context_mode != "history":
        return
    retry_history_turns.append(
        _render_retry_history_turn(
            attempt=attempt,
            note=note,
            failure_signature=failure_signature,
            patch_meta=patch_meta,
        )
    )


def _build_attempt_outcome_record(
    *,
    case: CaseSpec,
    condition: Condition,
    run_id: str,
    attempt: int,
    patcher_name: str,
    run_metadata: dict[str, Any],
    attempt_metadata: dict[str, Any],
    patcher: Any,
    patch_meta: dict[str, Any] | None,
    context_protocol: str,
    success: bool,
    note: str,
    before: RunResult,
    cumulative_runtime_sec: float,
    after: RunResult | None = None,
    attempt_wall_clock_sec: float | None = None,
) -> dict[str, Any]:
    patch_meta_dict = patch_meta if isinstance(patch_meta, dict) else None
    retry_summary = _build_retry_summary(
        attempt=attempt,
        note=note,
        failure_signature=(
            str(patch_meta_dict.get("adaptive_postpatch_failure_signature") or "")
            if patch_meta_dict is not None
            else ""
        ),
        patch_meta=patch_meta_dict,
    )
    if not str(retry_summary.get("failure_signature") or ""):
        retry_summary["failure_signature"] = _build_failure_signature_for_attempt(
            run=before,
            snapshot=None,
        )
    attempt_metadata["retry_summary"] = dict(retry_summary)
    if patch_meta_dict is not None:
        patch_meta_dict["retry_summary"] = dict(retry_summary)
    if (
        bool(attempt_metadata.get("patch_verifier_recovery_invoked"))
        or bool(attempt_metadata.get("patch_verifier_recovery_eligible"))
        or (
            patch_meta_dict is not None
            and (
                bool(patch_meta_dict.get("patch_verifier_recovery_invoked"))
                or bool(patch_meta_dict.get("patch_verifier_recovery_eligible"))
            )
        )
    ):
        attempt_metadata["patch_verifier_recovery_terminal_note"] = str(note or "")
        if patch_meta_dict is not None:
            patch_meta_dict["patch_verifier_recovery_terminal_note"] = str(note or "")
    record = _build_base_record(
        case=case,
        condition=condition,
        run_id=run_id,
        attempt=attempt,
        patcher_name=patcher_name,
        run_metadata=run_metadata,
        attempt_metadata=attempt_metadata,
        model_metadata=_build_model_metadata(patcher=patcher, patch_meta=patch_meta),
        context_protocol=context_protocol,
    )
    default_attempt_wall_clock_sec = float(before.duration_sec)
    if after is not None:
        default_attempt_wall_clock_sec += float(after.duration_sec)
    measured_attempt_wall_clock_sec = (
        float(attempt_wall_clock_sec)
        if isinstance(attempt_wall_clock_sec, (int, float))
        else default_attempt_wall_clock_sec
    )
    measured_attempt_wall_clock_sec = max(0.0, measured_attempt_wall_clock_sec)
    payload: dict[str, Any] = {
        "success": success,
        "before_returncode": before.returncode,
        "before_duration_sec": before.duration_sec,
        "cumulative_runtime_sec": cumulative_runtime_sec,
        "attempt_wall_clock_sec": measured_attempt_wall_clock_sec,
        "note": note,
    }
    if after is not None:
        payload["after_returncode"] = after.returncode
        payload["after_duration_sec"] = after.duration_sec
    if patch_meta_dict is not None:
        payload["patch_meta"] = patch_meta_dict
        payload["patch_touches_tests"] = bool(patch_meta_dict.get("patch_touches_tests"))
        payload["patch_test_file_count"] = _nonnegative_int(
            patch_meta_dict.get("patch_test_file_count")
        )
        payload["patch_file_count"] = _nonnegative_int(patch_meta_dict.get("patch_file_count"))
    record.update(payload)
    record.update(_timeout_fields(after if after is not None else before))
    return record


def _finalize_breakdowns(record: dict[str, Any]) -> None:
    """Populate timing_breakdown and token_breakdown from the final record.

    Extracts per-phase timing from patch_meta and record fields, and
    per-phase token counts from stage-A meta and patch-level usage.
    Called just before the record is appended to the results list.
    """
    attempt_meta = record.get("attempt_metadata")
    if not isinstance(attempt_meta, dict):
        return
    timing = attempt_meta.get("timing_breakdown")
    tokens = attempt_meta.get("token_breakdown")
    if not isinstance(timing, dict) or not isinstance(tokens, dict):
        return

    patch_meta = record.get("patch_meta")
    pm = patch_meta if isinstance(patch_meta, dict) else {}
    # ── Timing breakdown ────────────────────────────────────────────────
    # Stage-A LLM: from patcher meta request_context_duration_sec
    stage_a_meta = pm.get("stage_a_meta")
    stage_a_llm_sec = 0.0
    if isinstance(stage_a_meta, dict):
        stage_a_llm_sec = max(0.0, float(stage_a_meta.get("request_context_duration_sec") or 0.0))
    timing["stage_a_llm_sec"] = stage_a_llm_sec
    timing["adaptive_controller_prepare_sec"] = max(
        0.0,
        float(pm.get("adaptive_controller_prepare_sec") or 0.0),
    )
    timing["adaptive_controller_summarizer_sec"] = max(
        0.0,
        float(pm.get("adaptive_controller_summarizer_sec") or 0.0),
    )
    timing["adaptive_controller_sidecar_build_sec"] = max(
        0.0,
        float(pm.get("adaptive_controller_sidecar_build_sec") or 0.0),
    )
    timing["adaptive_controller_sidecar_write_sec"] = max(
        0.0,
        float(pm.get("adaptive_controller_sidecar_write_sec") or 0.0),
    )
    timing["baseline_case_conversion_sec"] = max(
        0.0,
        float(pm.get("baseline_case_conversion_sec") or 0.0),
    )
    timing["baseline_sidecar_load_sec"] = max(
        0.0,
        float(pm.get("baseline_sidecar_load_sec") or 0.0),
    )

    baseline_llm_sec = _baseline_metric_float(pm, "llm_duration_sec")
    timing["stage_b_llm_sec"] = (
        baseline_llm_sec
        if baseline_llm_sec is not None
        else max(0.0, float(pm.get("propose_duration_sec") or 0.0))
    )
    timing["baseline_internal_total_sec"] = (
        _baseline_metric_float(pm, "internal_total_duration_sec") or 0.0
    )
    timing["baseline_eval_sec"] = _baseline_metric_float(pm, "eval_duration_sec") or 0.0
    timing["baseline_orchestration_sec"] = (
        _baseline_metric_float(pm, "orchestration_duration_sec") or 0.0
    )

    context_compaction_invoked = bool(
        attempt_meta.get("context_compaction_invoked") or pm.get("context_compaction_invoked")
    )
    context_compaction_meta = (
        pm.get("context_compaction_meta")
        if isinstance(pm.get("context_compaction_meta"), dict)
        else None
    )

    # Compaction LLM: prefer the persisted nested metadata, with top-level fallback for old records
    compaction_duration_raw: Any
    if isinstance(context_compaction_meta, dict) and (
        "compact_context_duration_sec" in context_compaction_meta
    ):
        compaction_duration_raw = context_compaction_meta.get("compact_context_duration_sec")
    else:
        compaction_duration_raw = pm.get("compact_context_duration_sec")
    timing["context_compaction_llm_sec"] = max(0.0, float(compaction_duration_raw or 0.0))

    # Patch apply: apply_duration_sec
    timing["patch_apply_sec"] = max(0.0, float(pm.get("apply_duration_sec") or 0.0))

    # Test durations (from top-level record fields)
    timing["test_before_sec"] = max(0.0, float(record.get("before_duration_sec") or 0.0))
    timing["test_after_sec"] = max(0.0, float(record.get("after_duration_sec") or 0.0))

    # Recovery: sum context_recovery + no_diff_recovery + verifier_recovery durations
    recovery_llm_sec = max(0.0, float(pm.get("context_recovery_duration_sec") or 0.0))
    recovery_llm_sec += max(0.0, float(pm.get("no_diff_recovery_duration_sec") or 0.0))
    recovery_llm_sec += max(0.0, float(pm.get("patch_verifier_recovery_duration_sec") or 0.0))
    timing["recovery_llm_sec"] = recovery_llm_sec
    timing["recovery_total_sec"] = recovery_llm_sec

    # llmdebug_total_sec: snapshot_read + context_fulfillment (tool calls for llmdebug data)
    timing["llmdebug_total_sec"] = max(
        0.0,
        timing.get("snapshot_read_sec", 0.0) + timing.get("stage_a_context_fulfillment_sec", 0.0),
    )

    # llm_total_sec: Stage-B propose_duration_sec already includes recovery retries.
    timing["llm_total_sec"] = max(
        0.0,
        stage_a_llm_sec
        + timing.get("stage_b_llm_sec", 0.0)
        + timing.get("context_compaction_llm_sec", 0.0),
    )

    # overhead_sec: wall_clock - test_before - test_after - llm_total
    attempt_wall = max(0.0, float(record.get("attempt_wall_clock_sec") or 0.0))
    timing["overhead_sec"] = max(
        0.0,
        attempt_wall
        - timing.get("test_before_sec", 0.0)
        - timing.get("test_after_sec", 0.0)
        - timing.get("llm_total_sec", 0.0)
        - timing.get("baseline_eval_sec", 0.0)
        - timing.get("baseline_orchestration_sec", 0.0),
    )

    # ── Token breakdown ─────────────────────────────────────────────────
    # Stage-A tokens
    sa_p, sa_c, sa_t = _usage_token_counts_from_patch_meta(
        stage_a_meta if isinstance(stage_a_meta, dict) else None
    )
    tokens["stage_a_tokens_prompt"] = sa_p
    tokens["stage_a_tokens_completion"] = sa_c
    tokens["stage_a_tokens_total"] = sa_t

    # Compaction tokens. ATT-13: without compaction's own meta the tokens are unknown;
    # falling back to patch_meta reported the propose call's tokens as compaction tokens.
    cc_p = cc_c = cc_t = 0
    if context_compaction_invoked and isinstance(context_compaction_meta, dict):
        cc_p, cc_c, cc_t = _usage_token_counts_from_patch_meta(context_compaction_meta)
    tokens["compaction_tokens_prompt"] = cc_p
    tokens["compaction_tokens_completion"] = cc_c
    tokens["compaction_tokens_total"] = cc_t

    # Recovery tokens: request-failure, no-diff, and verifier-guided retries
    context_recovery_meta = pm.get("context_recovery_meta")
    if not isinstance(context_recovery_meta, dict):
        raw_context_recovery_meta = pm.get("context_recovery_raw_meta")
        context_recovery_meta = (
            raw_context_recovery_meta if isinstance(raw_context_recovery_meta, dict) else None
        )
    no_diff_recovery_meta = pm.get("no_diff_recovery_meta")
    if not isinstance(no_diff_recovery_meta, dict):
        raw_no_diff_recovery_meta = pm.get("no_diff_recovery_raw_meta")
        no_diff_recovery_meta = (
            raw_no_diff_recovery_meta if isinstance(raw_no_diff_recovery_meta, dict) else None
        )
    verifier_recovery_meta = (
        pm.get("patch_verifier_recovery_raw_meta")
        if isinstance(pm.get("patch_verifier_recovery_raw_meta"), dict)
        else None
    )
    recovery_prompt_tokens = recovery_completion_tokens = recovery_total_tokens = 0
    for recovery_meta in (
        context_recovery_meta,
        no_diff_recovery_meta,
        verifier_recovery_meta,
    ):
        rec_p, rec_c, rec_t = _usage_token_counts_from_patch_meta(
            recovery_meta if isinstance(recovery_meta, dict) else None
        )
        recovery_prompt_tokens += rec_p
        recovery_completion_tokens += rec_c
        recovery_total_tokens += rec_t
    tokens["recovery_tokens_prompt"] = recovery_prompt_tokens
    tokens["recovery_tokens_completion"] = recovery_completion_tokens
    tokens["recovery_tokens_total"] = recovery_total_tokens

    # Stage-B tokens: total attempt tokens minus stage-A minus compaction
    total_prompt = max(0, int(attempt_meta.get("tokens_prompt") or 0))
    total_completion = max(0, int(attempt_meta.get("tokens_completion") or 0))
    total_total = max(0, int(attempt_meta.get("tokens_total") or 0))
    stage_b_prompt = max(
        0,
        total_prompt
        - tokens.get("stage_a_tokens_prompt", 0)
        - tokens.get("compaction_tokens_prompt", 0)
        - tokens.get("recovery_tokens_prompt", 0),
    )
    stage_b_completion = max(
        0,
        total_completion
        - tokens.get("stage_a_tokens_completion", 0)
        - tokens.get("compaction_tokens_completion", 0)
        - tokens.get("recovery_tokens_completion", 0),
    )
    stage_b_total = max(
        0,
        total_total
        - tokens.get("stage_a_tokens_total", 0)
        - tokens.get("compaction_tokens_total", 0)
        - tokens.get("recovery_tokens_total", 0),
    )
    tokens["stage_b_tokens_prompt"] = stage_b_prompt
    tokens["stage_b_tokens_completion"] = stage_b_completion
    tokens["stage_b_tokens_total"] = stage_b_total

    # Tool response chars and snapshot chars
    tokens["tool_response_chars"] = max(0, int(attempt_meta.get("stage_a_chars_returned") or 0))


def _build_structured_previous_attempt_diff(
    previous_failure_context: dict[str, Any] | None,
    *,
    current_failure_context: dict[str, Any],
) -> dict[str, Any] | None:
    """Build a structured diff between consecutive attempt failures.

    Compares snapshots (when available) or falls back to text-based
    comparison of pytest output.
    """
    if not previous_failure_context:
        return None

    prev_attempt = int(previous_failure_context.get("attempt", 0))
    curr_attempt = int(current_failure_context.get("attempt", 0))
    prev_signature = str(previous_failure_context.get("failure_signature") or "")
    curr_signature = str(current_failure_context.get("failure_signature") or "")

    prev_snapshot = previous_failure_context.get("snapshot")
    curr_snapshot = current_failure_context.get("snapshot")
    if isinstance(prev_snapshot, dict) and isinstance(curr_snapshot, dict):
        try:
            from llmdebug.snapshot_diff import diff_snapshots, structured_diff_summary

            diff = diff_snapshots(prev_snapshot, curr_snapshot)
            structured = structured_diff_summary(diff)
            structured["attempt_prev"] = f"attempt_{prev_attempt:02d}"
            structured["attempt_curr"] = f"attempt_{curr_attempt:02d}"
            return structured
        except Exception as e:
            sys.stderr.write(
                f"llmdebug eval: structured diff failed, using text fallback: "
                f"{type(e).__name__}: {e}\n"
            )
            pass

    prev_output = (
        str(previous_failure_context.get("stderr") or "")
        + "\n"
        + str(previous_failure_context.get("stdout") or "")
    ).strip()
    curr_output = (
        str(current_failure_context.get("stderr") or "")
        + "\n"
        + str(current_failure_context.get("stdout") or "")
    ).strip()

    return {
        "attempt_prev": f"attempt_{prev_attempt:02d}",
        "attempt_curr": f"attempt_{curr_attempt:02d}",
        "failure_signature_same": prev_signature == curr_signature,
        "evidence_delta": {
            "new_frames": [],
            "dropped_frames": [],
            "message_changed": prev_output != curr_output,
        },
        "change_delta": {
            "touched_files": [],
            "touched_functions": [],
        },
        "hypothesis_delta": "unknown",
        "verification_delta": {
            "old_status": "unknown",
            "new_status": "unknown",
            "status_changed": False,
        },
    }
