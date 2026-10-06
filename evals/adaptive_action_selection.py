"""Helper-action selection for adaptive attempts: the policy layer (layer 2).

Moved out of the attempt loop in ``evals.eval_attempt_flow`` (X-01). The caller
initialises :class:`HelperActionState` for the attempt and reads it back.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evals.adaptive_online_policy import (
    ACTION_COMPACT_THEN_PATCH,
    ACTION_FRAME_LOCALS_LITE,
    ACTION_INVOKE_HELPER,
    ACTION_SKIP_HELPER,
    ACTION_SNAPSHOT_FULL_LITE,
    ACTION_SNAPSHOT_LITE,
)
from evals.adaptive_progress import _adaptive_gate_should_invoke_helper
from evals.decision_model.runtime import (
    AdviceRequest,
    advise_for_attempt,
    binding_of,
    record_advice_state,
)
from evals.decision_model.state import DecisionStateInputs, redact_terms_for_unit
from evals.eval_patch_constants import TRUNCATED_RESPONSE_ERROR
from evals.eval_types import HELPER_STAGE_A_ACTIONS, _helper_action_availability


@dataclass(frozen=True)
class HelperActionInputs:
    """Read-only inputs of :func:`select_helper_action`."""

    _emit_progress: Any
    _progress_payload: Any
    adaptive_budget_topup_granted: Any
    adaptive_budget_unlimited: Any
    adaptive_informativeness_pregate: Any
    adaptive_lite_snapshot_mode: Any
    adaptive_online_policy_controller: Any
    adaptive_recovery_decision_reason: Any
    adaptive_recovery_forced_action: Any
    adaptive_recovery_override_reason: Any
    attempt: Any
    case: Any
    condition: Any
    context_compaction_calls_used: Any
    current: Any
    effective_context_compaction_max_calls: Any
    effective_helper_budget_limit: Any
    failure_signature_repeat: Any
    helper_action_default: Any
    helper_budget_exhausted: Any
    helper_calls_used: Any
    helper_saturation_guarded: Any
    is_adaptive_condition: Any
    is_adaptive_gated: Any
    is_adaptive_llm_discretion: Any
    k: Any
    last_helper_action_executed: Any
    last_helper_decision_reason: Any
    normalized_adaptive_gate_policy: Any
    normalized_adaptive_online_policy_mode: Any
    normalized_adaptive_reveal_mode: Any
    normalized_adaptive_stagnation_window: Any
    normalized_context_compaction_mode: Any
    online_policy_features_raw: Any
    pregate_result: Any
    previous_failure_context: Any
    retry_history_turns: Any
    run_id: Any
    snapshot: Any
    stagnation_streak: Any
    supports_context_compaction: Any
    supports_context_request: Any
    supports_strict_json_first: Any
    workdir: Any


@dataclass
class HelperActionState:
    """Values :func:`select_helper_action` may update, initialised by the attempt loop."""

    adaptive_force_helper_stage_a: Any
    adaptive_stagnation_override_applied: Any
    adaptive_stagnation_override_reason: Any
    helper_action_executed: Any
    helper_action_selected: Any
    helper_available_actions: Any
    helper_decision_reason: Any
    helper_should_consult: Any
    helper_unavailable_action_reasons: Any
    online_policy_algo_for_attempt: Any
    online_policy_decision: Any
    online_policy_eligible: Any
    online_policy_features_scaled: Any
    online_policy_mode_for_attempt: Any
    online_policy_override_reason: Any


def select_helper_action(inputs: HelperActionInputs, state: HelperActionState) -> HelperActionState:
    """Pick the helper action for an adaptive attempt (the policy layer, layer 2).

    Applies the hard constraints first (informativeness pre-gate, helper budget,
    saturation guard, recovery overrides), then the gated rule or the online policy,
    and records why. Moved verbatim from the attempt loop (X-01)."""
    _emit_progress = inputs._emit_progress
    _progress_payload = inputs._progress_payload
    adaptive_budget_topup_granted = inputs.adaptive_budget_topup_granted
    adaptive_budget_unlimited = inputs.adaptive_budget_unlimited
    adaptive_informativeness_pregate = inputs.adaptive_informativeness_pregate
    adaptive_lite_snapshot_mode = inputs.adaptive_lite_snapshot_mode
    adaptive_online_policy_controller = inputs.adaptive_online_policy_controller
    adaptive_recovery_decision_reason = inputs.adaptive_recovery_decision_reason
    adaptive_recovery_forced_action = inputs.adaptive_recovery_forced_action
    adaptive_recovery_override_reason = inputs.adaptive_recovery_override_reason
    attempt = inputs.attempt
    case = inputs.case
    condition = inputs.condition
    context_compaction_calls_used = inputs.context_compaction_calls_used
    current = inputs.current
    effective_context_compaction_max_calls = inputs.effective_context_compaction_max_calls
    effective_helper_budget_limit = inputs.effective_helper_budget_limit
    failure_signature_repeat = inputs.failure_signature_repeat
    helper_action_default = inputs.helper_action_default
    helper_budget_exhausted = inputs.helper_budget_exhausted
    helper_calls_used = inputs.helper_calls_used
    helper_saturation_guarded = inputs.helper_saturation_guarded
    is_adaptive_condition = inputs.is_adaptive_condition
    is_adaptive_gated = inputs.is_adaptive_gated
    is_adaptive_llm_discretion = inputs.is_adaptive_llm_discretion
    k = inputs.k
    last_helper_action_executed = inputs.last_helper_action_executed
    last_helper_decision_reason = inputs.last_helper_decision_reason
    normalized_adaptive_gate_policy = inputs.normalized_adaptive_gate_policy
    normalized_adaptive_online_policy_mode = inputs.normalized_adaptive_online_policy_mode
    normalized_adaptive_reveal_mode = inputs.normalized_adaptive_reveal_mode
    normalized_adaptive_stagnation_window = inputs.normalized_adaptive_stagnation_window
    normalized_context_compaction_mode = inputs.normalized_context_compaction_mode
    online_policy_features_raw = inputs.online_policy_features_raw
    pregate_result = inputs.pregate_result
    previous_failure_context = inputs.previous_failure_context
    retry_history_turns = inputs.retry_history_turns
    run_id = inputs.run_id
    snapshot = inputs.snapshot
    stagnation_streak = inputs.stagnation_streak
    supports_context_compaction = inputs.supports_context_compaction
    supports_context_request = inputs.supports_context_request
    supports_strict_json_first = inputs.supports_strict_json_first
    workdir = inputs.workdir
    adaptive_force_helper_stage_a = state.adaptive_force_helper_stage_a
    adaptive_stagnation_override_applied = state.adaptive_stagnation_override_applied
    adaptive_stagnation_override_reason = state.adaptive_stagnation_override_reason
    helper_action_executed = state.helper_action_executed
    helper_action_selected = state.helper_action_selected
    helper_available_actions = state.helper_available_actions
    helper_decision_reason = state.helper_decision_reason
    helper_should_consult = state.helper_should_consult
    helper_unavailable_action_reasons = state.helper_unavailable_action_reasons
    online_policy_algo_for_attempt = state.online_policy_algo_for_attempt
    online_policy_decision = state.online_policy_decision
    online_policy_eligible = state.online_policy_eligible
    online_policy_features_scaled = state.online_policy_features_scaled
    online_policy_mode_for_attempt = state.online_policy_mode_for_attempt
    online_policy_override_reason = state.online_policy_override_reason
    if is_adaptive_condition:
        last_helper_invalid_response = (
            str(last_helper_decision_reason).strip() == "llm_invalid_response_no_helper"
        )
        if pregate_result.should_skip:
            helper_decision_reason = "informativeness_pregate_skip"
            online_policy_override_reason = "constraint_informativeness_pregate_skip"
            helper_action_selected = ACTION_SKIP_HELPER
            helper_action_executed = ACTION_SKIP_HELPER
        elif helper_budget_exhausted:
            helper_decision_reason = "helper_budget_exhausted"
            online_policy_override_reason = "constraint_helper_budget_exhausted"
            helper_action_selected = ACTION_SKIP_HELPER
            helper_action_executed = ACTION_SKIP_HELPER
        elif helper_saturation_guarded:
            helper_decision_reason = "helper_saturation_guard"
            online_policy_override_reason = "constraint_helper_saturation_guard"
            helper_action_selected = ACTION_SKIP_HELPER
            helper_action_executed = ACTION_SKIP_HELPER
        elif adaptive_recovery_forced_action:
            helper_decision_reason = adaptive_recovery_decision_reason
            online_policy_override_reason = adaptive_recovery_override_reason
            helper_action_selected = adaptive_recovery_forced_action
            helper_action_executed = adaptive_recovery_forced_action
            helper_should_consult = True
        elif is_adaptive_gated:
            helper_should_consult, helper_decision_reason = _adaptive_gate_should_invoke_helper(
                policy=normalized_adaptive_gate_policy,
                attempt=attempt,
                timed_out=bool(current.timed_out),
                signature_repeated=failure_signature_repeat,
                stagnation_streak=stagnation_streak,
                stagnation_window=normalized_adaptive_stagnation_window,
            )
            helper_action_selected = (
                ACTION_INVOKE_HELPER if helper_should_consult else ACTION_SKIP_HELPER
            )
            helper_action_executed = helper_action_selected
        else:
            helper_decision_reason = "llm_discretion_pending"
            helper_available_actions, helper_unavailable_action_reasons = (
                _helper_action_availability(
                    supports_context_request=supports_context_request,
                    supports_strict_json_first=supports_strict_json_first,
                    snapshot_available=snapshot is not None,
                    context_compaction_mode=normalized_context_compaction_mode,
                    supports_context_compaction=supports_context_compaction,
                    context_compaction_calls_used=context_compaction_calls_used,
                    context_compaction_max_calls=effective_context_compaction_max_calls,
                    last_helper_invalid_response=last_helper_invalid_response,
                )
            )
            online_policy_eligible = bool(
                adaptive_online_policy_controller is not None
                and normalized_adaptive_online_policy_mode in {"shadow", "on"}
                and len(helper_available_actions) > 1
            )
            if online_policy_eligible and adaptive_online_policy_controller is not None:
                decide_kwargs: dict[str, Any] = {
                    "feature_values": online_policy_features_raw,
                    "default_action": helper_action_default,
                    "available_actions": helper_available_actions,
                    "unavailable_action_reasons": helper_unavailable_action_reasons,
                }
                # System-1 decision model: queried outside the controller lock, only when
                # the policy itself is consulted (layer-1 constraints already applied).
                decision_model_binding = binding_of(adaptive_online_policy_controller)
                decision_model_advice = None
                if decision_model_binding is not None:
                    _emit_progress(
                        "decision_model_query",
                        _progress_payload(
                            owner="harness",
                            phase_label="harness: decision model",
                            note="querying decision model",
                            attempt=int(attempt),
                            max_attempts=int(k),
                        ),
                    )
                    redact_paths, redact_tokens = redact_terms_for_unit(
                        workdirs=(str(Path(workdir).resolve()), str(workdir)),
                        case_id=case.case_id,
                        condition=condition.value,
                        run_id=str(run_id),
                    )
                    decision_model_advice = advise_for_attempt(
                        decision_model_binding,
                        AdviceRequest(
                            state_inputs=DecisionStateInputs(
                                stdout=current.stdout,
                                stderr=current.stderr,
                                timed_out=bool(current.timed_out),
                                snapshot=snapshot,
                                attempt=int(attempt),
                                max_attempts=int(k),
                                previous_failure_context=previous_failure_context,
                                retry_history_turns=tuple(retry_history_turns),
                                pregate=(
                                    pregate_result if adaptive_informativeness_pregate else None
                                ),
                                helper_calls_remaining=(
                                    None
                                    if adaptive_budget_unlimited
                                    else max(0, effective_helper_budget_limit - helper_calls_used)
                                ),
                                compaction_calls_remaining=(
                                    max(
                                        0,
                                        effective_context_compaction_max_calls
                                        - context_compaction_calls_used,
                                    )
                                    if normalized_context_compaction_mode != "off"
                                    else 0
                                ),
                                failure_signature_repeat=bool(failure_signature_repeat),
                                last_helper_action=str(last_helper_action_executed or ""),
                                reveal_mode=normalized_adaptive_reveal_mode,
                                redact_paths=redact_paths,
                                redact_tokens=redact_tokens,
                            ),
                            available_actions=tuple(helper_available_actions),
                            lite_snapshot_mode=adaptive_lite_snapshot_mode,
                        ),
                    )
                    decide_kwargs["advice"] = decision_model_advice
                online_policy_decision = adaptive_online_policy_controller.decide(
                    rng_key=f"helper|{case.case_id}|{condition.value}|{attempt}",
                    **decide_kwargs,
                )
                record_advice_state(
                    decision_model_binding,
                    decision=online_policy_decision,
                    advice=decision_model_advice,
                    context={
                        "run_id": str(run_id),
                        "case_id": case.case_id,
                        "condition": condition.value,
                        "attempt": int(attempt),
                    },
                )
                online_policy_features_scaled = dict(online_policy_decision.feature_values)
                online_policy_mode_for_attempt = online_policy_decision.mode
                online_policy_algo_for_attempt = online_policy_decision.algorithm
                helper_action_selected = str(
                    online_policy_decision.selected_action or helper_action_default
                )
                helper_action_executed = str(
                    online_policy_decision.applied_action or helper_action_default
                )
                if online_policy_decision.applied_action != online_policy_decision.selected_action:
                    online_policy_override_reason = "shadow_mode_baseline"
            elif normalized_adaptive_online_policy_mode == "off":
                online_policy_override_reason = "policy_off"
                helper_action_selected = helper_action_default
                helper_action_executed = helper_action_default
            elif adaptive_online_policy_controller is None:
                online_policy_override_reason = "policy_controller_unavailable"
                helper_action_selected = helper_action_default
                helper_action_executed = helper_action_default
            elif len(helper_available_actions) == 1:
                only_action = str(helper_available_actions[0])
                online_policy_override_reason = "constraint_single_feasible_action"
                helper_action_selected = only_action
                helper_action_executed = only_action
            else:
                online_policy_override_reason = "not_eligible"
                helper_action_selected = helper_action_default
                helper_action_executed = helper_action_default
            helper_should_consult = helper_action_executed in HELPER_STAGE_A_ACTIONS
            if helper_action_executed == ACTION_SKIP_HELPER:
                helper_decision_reason = "online_policy_skip_helper"
            elif helper_action_executed == ACTION_SNAPSHOT_LITE:
                helper_decision_reason = "policy_action_snapshot_lite"
            elif helper_action_executed == ACTION_FRAME_LOCALS_LITE:
                helper_decision_reason = "policy_action_frame_locals_lite"
            elif helper_action_executed == ACTION_SNAPSHOT_FULL_LITE:
                helper_decision_reason = "policy_action_snapshot_full_lite"
            elif helper_action_executed == ACTION_COMPACT_THEN_PATCH:
                helper_decision_reason = "policy_action_compact_then_patch"
        if adaptive_budget_topup_granted and not online_policy_override_reason:
            online_policy_override_reason = "constraint_helper_budget_topup"

        previous_note_for_override = (
            str(previous_failure_context.get("note") or "")
            if isinstance(previous_failure_context, dict)
            else ""
        )
        previous_stagnation_signal = previous_note_for_override in {
            "no_patch_proposed",
            TRUNCATED_RESPONSE_ERROR,
            "patcher_request_failed",
            "forced_noop_patch",
            "patch_verifier_rejected",
            "syntax_preflight_failed",
            "patch_apply_failed",
            "tests_still_failing",
            "retry_gate_no_new_evidence",
            "hard_stop_no_progress",
        }
        policy_skip_signal = bool(
            online_policy_decision is not None
            and str(online_policy_decision.applied_action or "") not in HELPER_STAGE_A_ACTIONS
        )
        historical_skip_signal = last_helper_decision_reason in {
            "online_policy_skip_helper",
            "llm_declined_helper",
            "llm_invalid_response_no_helper",
            "llm_snapshot_request_denied_attempt1",
            "policy_action_snapshot_lite",
            "policy_action_frame_locals_lite",
            "policy_action_snapshot_full_lite",
            "policy_action_compact_then_patch",
        }
        if (
            is_adaptive_llm_discretion
            and (not helper_should_consult)
            and failure_signature_repeat
            and stagnation_streak > 0
            and previous_stagnation_signal
            and (policy_skip_signal or historical_skip_signal)
            and (not pregate_result.should_skip)
            and supports_context_request
        ):
            adaptive_stagnation_override_reason = "repeated_no_progress_after_stagnation_note"
            if not helper_budget_exhausted and not helper_saturation_guarded:
                adaptive_stagnation_override_applied = True
                adaptive_force_helper_stage_a = True
                helper_action_executed = ACTION_INVOKE_HELPER
                helper_should_consult = True
                helper_decision_reason = "stagnation_override_force_helper"
                online_policy_override_reason = "constraint_stagnation_override_force_helper"
    return HelperActionState(
        adaptive_force_helper_stage_a=adaptive_force_helper_stage_a,
        adaptive_stagnation_override_applied=adaptive_stagnation_override_applied,
        adaptive_stagnation_override_reason=adaptive_stagnation_override_reason,
        helper_action_executed=helper_action_executed,
        helper_action_selected=helper_action_selected,
        helper_available_actions=helper_available_actions,
        helper_decision_reason=helper_decision_reason,
        helper_should_consult=helper_should_consult,
        helper_unavailable_action_reasons=helper_unavailable_action_reasons,
        online_policy_algo_for_attempt=online_policy_algo_for_attempt,
        online_policy_decision=online_policy_decision,
        online_policy_eligible=online_policy_eligible,
        online_policy_features_scaled=online_policy_features_scaled,
        online_policy_mode_for_attempt=online_policy_mode_for_attempt,
        online_policy_override_reason=online_policy_override_reason,
    )
