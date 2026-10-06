"""Stage A of an adaptive attempt: context requests, request filtering and execution.

Moved out of the attempt loop in ``evals.eval_attempt_flow`` (X-01).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

from evals.adaptive_online_policy import (
    ACTION_COMPACT_THEN_PATCH,
    ACTION_FRAME_LOCALS_LITE,
    ACTION_RETRY_HELPER_STRICT_JSON,
    ACTION_SKIP_HELPER,
    ACTION_SNAPSHOT_FULL_LITE,
    ACTION_SNAPSHOT_LITE,
)
from evals.eval_evidence import (
    _coerce_context_request_payload,
    build_fallback_context_bundle,
    deny_fallback_for_restricted_action,
    execute_context_requests,
)
from evals.eval_prompts import (
    _format_frame_locals_section,
    _format_snapshot_full_section,
    _format_snapshot_index_section,
    render_stage_a_request_prompt,
)
from evals.eval_types import (
    HELPER_RESTRICTED_STAGE_A_ACTIONS,
    StageAContextData,
    _empty_stage_a_context,
    _filter_context_request_payload_for_action,
)
from evals.transport_failures import (
    _request_context_failure_kind_from_meta,
    _request_context_recovery_failure_kind_from_meta,
)


@dataclass(frozen=True)
class StageAInputs:
    """Read-only inputs of :func:`run_stage_a_requests`."""

    _run_openai_operation_with_transport_recovery: Any
    adaptive_budget_unlimited: Any
    adaptive_tep_controller: Any
    attempt: Any
    attempt_dir: Any
    cumulative_runtime_sec: Any
    current: Any
    default_context_protocol: Any
    effective_helper_budget_limit: Any
    evidence: Any
    file_index_lines: Any
    full_tep_enabled: Any
    helper_action_executed: Any
    helper_budget_exhausted: Any
    helper_calls_used: Any
    helper_saturation_guarded: Any
    is_adaptive_condition: Any
    is_adaptive_gated: Any
    is_adaptive_llm_discretion: Any
    patcher_for_attempt: Any
    pregate_result: Any
    request_context_fn: Any
    run_stage_a: Any
    snapshot: Any
    stage_a_max_chars: Any
    stage_a_max_requests: Any
    stage_a_meta: Any
    stage_a_raw_response: Any
    stage_b_snapshot: Any
    supports_context_request: Any
    workdir: Any


@dataclass
class StageAState:
    """Values :func:`run_stage_a_requests` may update, initialised by the attempt loop."""

    _stage_a_context_fulfillment_sec: Any
    helper_decision_reason: Any
    helper_invoked: Any
    helper_need_more_context: Any
    helper_should_consult: Any
    snapshot_lite_context: Any
    stage_a_compact_context_reason: Any
    stage_a_data: Any
    stage_a_need_compact_context: Any
    stage_a_need_snapshot: Any
    stage_a_prompt: Any
    stage_a_request_invoked: Any
    stage_a_request_payload: Any
    stage_a_snapshot_justification: Any


def run_stage_a_requests(inputs: StageAInputs, state: StageAState) -> StageAState:
    """Run Stage A for an attempt: ask for context, filter the requests, execute them.

    Layer 3 of the policy enforcement lives here: requests outside the selected
    action's allowed types are dropped before execution (ATT-03, ATT-04). Moved
    verbatim from the attempt loop (X-01)."""
    _run_openai_operation_with_transport_recovery = (
        inputs._run_openai_operation_with_transport_recovery
    )
    adaptive_budget_unlimited = inputs.adaptive_budget_unlimited
    adaptive_tep_controller = inputs.adaptive_tep_controller
    attempt = inputs.attempt
    attempt_dir = inputs.attempt_dir
    cumulative_runtime_sec = inputs.cumulative_runtime_sec
    current = inputs.current
    default_context_protocol = inputs.default_context_protocol
    effective_helper_budget_limit = inputs.effective_helper_budget_limit
    evidence = inputs.evidence
    file_index_lines = inputs.file_index_lines
    full_tep_enabled = inputs.full_tep_enabled
    helper_action_executed = inputs.helper_action_executed
    helper_budget_exhausted = inputs.helper_budget_exhausted
    helper_calls_used = inputs.helper_calls_used
    helper_saturation_guarded = inputs.helper_saturation_guarded
    is_adaptive_condition = inputs.is_adaptive_condition
    is_adaptive_gated = inputs.is_adaptive_gated
    is_adaptive_llm_discretion = inputs.is_adaptive_llm_discretion
    patcher_for_attempt = inputs.patcher_for_attempt
    pregate_result = inputs.pregate_result
    request_context_fn = inputs.request_context_fn
    run_stage_a = inputs.run_stage_a
    snapshot = inputs.snapshot
    stage_a_max_chars = inputs.stage_a_max_chars
    stage_a_max_requests = inputs.stage_a_max_requests
    stage_a_meta = inputs.stage_a_meta
    stage_a_raw_response = inputs.stage_a_raw_response
    stage_b_snapshot = inputs.stage_b_snapshot
    supports_context_request = inputs.supports_context_request
    workdir = inputs.workdir
    _stage_a_context_fulfillment_sec = state._stage_a_context_fulfillment_sec
    helper_decision_reason = state.helper_decision_reason
    helper_invoked = state.helper_invoked
    helper_need_more_context = state.helper_need_more_context
    helper_should_consult = state.helper_should_consult
    snapshot_lite_context = state.snapshot_lite_context
    stage_a_compact_context_reason = state.stage_a_compact_context_reason
    stage_a_data = state.stage_a_data
    stage_a_need_compact_context = state.stage_a_need_compact_context
    stage_a_need_snapshot = state.stage_a_need_snapshot
    stage_a_prompt = state.stage_a_prompt
    stage_a_request_invoked = state.stage_a_request_invoked
    stage_a_request_payload = state.stage_a_request_payload
    stage_a_snapshot_justification = state.stage_a_snapshot_justification
    if run_stage_a:
        stage_a_prompt = render_stage_a_request_prompt(
            evidence=evidence,
            snapshot=snapshot,
            helper_budget_total=(
                effective_helper_budget_limit
                if is_adaptive_condition and not adaptive_budget_unlimited
                else 0
            ),
            helper_budget_remaining=(
                max(0, effective_helper_budget_limit - helper_calls_used)
                if is_adaptive_condition and not adaptive_budget_unlimited
                else 0
            ),
            helper_action=helper_action_executed,
            tep_actor_overrides=(
                adaptive_tep_controller.stage_a_actor_overrides() if full_tep_enabled else None
            ),
        )
        if supports_context_request:
            stage_a_request_invoked = True
            if is_adaptive_condition:
                helper_invoked = True
            if helper_action_executed == ACTION_RETRY_HELPER_STRICT_JSON:
                context_result = _run_openai_operation_with_transport_recovery(
                    patcher_for_attempt=patcher_for_attempt,
                    operation_name="request_context",
                    invoke=lambda request_context_fn=request_context_fn,
                    stage_a_prompt=stage_a_prompt,
                    attempt_dir=attempt_dir,
                    workdir=workdir: (
                        request_context_fn(
                            prompt=stage_a_prompt,
                            context_dir=attempt_dir,
                            cwd=workdir,
                            strict_json_first=True,
                        )
                    ),
                    classify_failure_kind=_request_context_recovery_failure_kind_from_meta,
                    attempt=attempt,
                    attempt_metadata=None,
                    attempt_context_protocol=default_context_protocol,
                    before=current,
                    cumulative_runtime_sec_value=cumulative_runtime_sec,
                )
            else:
                context_result = _run_openai_operation_with_transport_recovery(
                    patcher_for_attempt=patcher_for_attempt,
                    operation_name="request_context",
                    invoke=lambda request_context_fn=request_context_fn,
                    stage_a_prompt=stage_a_prompt,
                    attempt_dir=attempt_dir,
                    workdir=workdir: (
                        request_context_fn(
                            prompt=stage_a_prompt,
                            context_dir=attempt_dir,
                            cwd=workdir,
                        )
                    ),
                    classify_failure_kind=_request_context_recovery_failure_kind_from_meta,
                    attempt=attempt,
                    attempt_metadata=None,
                    attempt_context_protocol=default_context_protocol,
                    before=current,
                    cumulative_runtime_sec_value=cumulative_runtime_sec,
                )
        else:
            context_result = None
        if context_result is not None:
            if hasattr(context_result, "meta") and isinstance(context_result.meta, dict):
                stage_a_meta.update(context_result.meta)
            request_candidate = (
                context_result.request_json if hasattr(context_result, "request_json") else None
            )
            request_payload = _coerce_context_request_payload(request_candidate)
            if request_payload is not None:
                stage_a_request_payload = _filter_context_request_payload_for_action(
                    action=helper_action_executed,
                    request_payload=request_payload,
                )
            if hasattr(context_result, "raw_response"):
                raw_candidate = context_result.raw_response
                if isinstance(raw_candidate, str):
                    stage_a_raw_response = raw_candidate

        if stage_a_request_payload is not None:
            stage_a_need_snapshot = bool(stage_a_request_payload.get("need_snapshot"))
            stage_a_snapshot_justification = str(
                stage_a_request_payload.get("snapshot_justification") or ""
            )
            stage_a_need_compact_context = bool(stage_a_request_payload.get("need_compact_context"))
            stage_a_compact_context_reason = str(
                stage_a_request_payload.get("compact_context_reason") or ""
            )
            _context_fulfillment_start = time.perf_counter()
            stage_a_data = execute_context_requests(
                request_payload=stage_a_request_payload,
                root=workdir,
                snapshot=snapshot,
                file_index=file_index_lines,
                stdout=current.stdout,
                stderr=current.stderr,
                max_requests=stage_a_max_requests,
                max_total_chars=stage_a_max_chars,
            )
            stage_a_data = deny_fallback_for_restricted_action(
                stage_a_data,
                restricted=helper_action_executed in HELPER_RESTRICTED_STAGE_A_ACTIONS,
            )
            _stage_a_context_fulfillment_sec = time.perf_counter() - _context_fulfillment_start
            if is_adaptive_condition:
                helper_need_more_context = bool(stage_a_request_payload.get("need_more_context"))
                if is_adaptive_llm_discretion:
                    helper_decision_reason = (
                        "llm_requested_helper"
                        if (helper_need_more_context or stage_a_need_snapshot)
                        else "llm_declined_helper"
                    )
                elif is_adaptive_gated and not helper_should_consult and helper_need_more_context:
                    helper_should_consult = True
                    helper_decision_reason = "gate_requested_helper"
        else:
            request_context_failure_kind = _request_context_failure_kind_from_meta(stage_a_meta)
            stage_a_failure_reason = {
                "transport": "helper_transport_failed",
                "backend_unavailable": "helper_unavailable_backend",
                "compat_reasoning_only": "helper_compat_reasoning_only",
                "request_budget_exceeded": "helper_request_budget_exceeded",
                "invalid_payload": "helper_invalid_payload",
            }.get(request_context_failure_kind, "invalid_or_missing_stage_a_response")
            if is_adaptive_llm_discretion:
                if request_context_failure_kind in {"", "invalid_payload"}:
                    helper_decision_reason = "llm_invalid_response_no_helper"
                stage_a_data = StageAContextData(
                    blocks=[],
                    text="",
                    used_fallback=False,
                    requests_count=0,
                    chars_returned=0,
                    request_success=False,
                    request_payload=None,
                    request_raw=stage_a_raw_response,
                    reason=stage_a_failure_reason,
                    evidence_items=[],
                )
            else:
                stage_a_data = deny_fallback_for_restricted_action(
                    build_fallback_context_bundle(
                        root=workdir,
                        snapshot=snapshot,
                        stdout=current.stdout,
                        stderr=current.stderr,
                        max_total_chars=stage_a_max_chars,
                        reason=stage_a_failure_reason,
                        file_index_lines=file_index_lines,
                        request_raw=stage_a_raw_response,
                    ),
                    restricted=helper_action_executed in HELPER_RESTRICTED_STAGE_A_ACTIONS,
                )

        if stage_a_data.request_payload is None:
            stage_a_data = StageAContextData(
                blocks=stage_a_data.blocks,
                text=stage_a_data.text,
                used_fallback=stage_a_data.used_fallback,
                requests_count=stage_a_data.requests_count,
                chars_returned=stage_a_data.chars_returned,
                request_success=stage_a_data.request_success,
                request_payload=stage_a_request_payload,
                request_raw=stage_a_raw_response,
                reason=stage_a_data.reason,
                evidence_items=stage_a_data.evidence_items,
            )
    else:
        if is_adaptive_llm_discretion:
            if pregate_result.should_skip:
                pass  # Keep "informativeness_pregate_skip" from earlier
            elif helper_budget_exhausted:
                helper_decision_reason = "helper_budget_exhausted"
            elif helper_saturation_guarded:
                helper_decision_reason = "helper_saturation_guard"
            elif helper_action_executed == ACTION_SNAPSHOT_LITE and snapshot is not None:
                snapshot_lite_context = _format_snapshot_index_section(
                    stage_b_snapshot if stage_b_snapshot is not None else snapshot
                )
                helper_decision_reason = "policy_action_snapshot_lite"
            elif helper_action_executed == ACTION_FRAME_LOCALS_LITE and snapshot is not None:
                snapshot_lite_context = _format_frame_locals_section(
                    stage_b_snapshot if stage_b_snapshot is not None else snapshot
                )
                helper_decision_reason = "policy_action_frame_locals_lite"
            elif helper_action_executed == ACTION_SNAPSHOT_FULL_LITE and snapshot is not None:
                snapshot_lite_context = _format_snapshot_full_section(
                    stage_b_snapshot if stage_b_snapshot is not None else snapshot
                )
                helper_decision_reason = "policy_action_snapshot_full_lite"
            elif helper_action_executed == ACTION_COMPACT_THEN_PATCH:
                helper_decision_reason = "policy_action_compact_then_patch"
                stage_a_need_compact_context = True
                stage_a_compact_context_reason = "policy_action"
            elif not supports_context_request:
                helper_decision_reason = "request_context_unavailable"
            elif helper_action_executed == ACTION_SKIP_HELPER:
                helper_decision_reason = "online_policy_skip_helper"
            stage_a_data = _empty_stage_a_context(reason=helper_decision_reason)
        elif is_adaptive_gated:
            stage_a_data = _empty_stage_a_context(reason=helper_decision_reason)
    return StageAState(
        _stage_a_context_fulfillment_sec=_stage_a_context_fulfillment_sec,
        helper_decision_reason=helper_decision_reason,
        helper_invoked=helper_invoked,
        helper_need_more_context=helper_need_more_context,
        helper_should_consult=helper_should_consult,
        snapshot_lite_context=snapshot_lite_context,
        stage_a_compact_context_reason=stage_a_compact_context_reason,
        stage_a_data=stage_a_data,
        stage_a_need_compact_context=stage_a_need_compact_context,
        stage_a_need_snapshot=stage_a_need_snapshot,
        stage_a_prompt=stage_a_prompt,
        stage_a_request_invoked=stage_a_request_invoked,
        stage_a_request_payload=stage_a_request_payload,
        stage_a_snapshot_justification=stage_a_snapshot_justification,
    )
