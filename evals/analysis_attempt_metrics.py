"""Per-row metrics read from attempt records: Stage A, retry gates, preflight, verifier, helper and policy telemetry."""

from __future__ import annotations

from collections import defaultdict
from typing import Any

from evals.analysis_numeric import _as_int, _mean
from evals.analysis_ope_support import (
    ADAPTIVE_BUDGET_FAMILIES,
    ADAPTIVE_BUDGET_REWARD_SCOPES,
    ADAPTIVE_ONLINE_ACTIONS,
    ADAPTIVE_ONLINE_REWARD_SCOPES,
    _metadata_with_fallback,
)


def _context_protocol_from_records(records: list[dict[str, Any]]) -> str:
    """Extract the context protocol label, defaulting to ``'monolithic'``."""
    for record in records:
        protocol = record.get("context_protocol")
        if isinstance(protocol, str) and protocol:
            return protocol
        attempt_meta = record.get("attempt_metadata")
        if isinstance(attempt_meta, dict):
            maybe = attempt_meta.get("context_protocol")
            if isinstance(maybe, str) and maybe:
                return maybe
    return "monolithic"


def _stage_b_prompt_chars_mean(records: list[dict[str, Any]]) -> float | None:
    """Mean stage-B prompt character count across attempt records."""
    values: list[float] = []
    for record in records:
        attempt_meta = record.get("attempt_metadata")
        if not isinstance(attempt_meta, dict):
            continue
        value = attempt_meta.get("stage_b_prompt_chars")
        if isinstance(value, (int, float)):
            values.append(float(value))
    return _mean(values)


def _stage_a_request_counts(records: list[dict[str, Any]]) -> tuple[int, int]:
    """Count stage-A request attempts and successes.

    Returns:
        Tuple of (total_attempts, total_successes).
    """
    attempts = 0
    successes = 0
    for record in records:
        attempt_meta = record.get("attempt_metadata")
        if not isinstance(attempt_meta, dict):
            continue
        success = attempt_meta.get("stage_a_request_success")
        if success is None:
            continue
        attempts += 1
        if bool(success):
            successes += 1
    return attempts, successes


def _retry_gate_counts(records: list[dict[str, Any]]) -> tuple[int, int]:
    """Count retry-gate evaluations and blocks across records.

    Returns:
        Tuple of (evaluated_attempts, blocked_attempts).
    """
    attempts = 0
    blocks = 0
    for record in records:
        patch_meta = record.get("patch_meta")
        if not isinstance(patch_meta, dict):
            continue
        # ATT-05: unscored blocks are infra events; the next scored attempt counts them.
        blocks_before = patch_meta.get("retry_gate_blocks_before")
        if isinstance(blocks_before, int) and blocks_before > 0:
            attempts += blocks_before
            blocks += blocks_before
        evaluated = patch_meta.get("retry_gate_evaluated")
        blocked = patch_meta.get("retry_gate_blocked")
        if isinstance(evaluated, bool):
            attempts += 1
            if bool(blocked):
                blocks += 1
            continue
        if blocked is None:
            continue
        attempts += 1
        if bool(blocked):
            blocks += 1
    return attempts, blocks


def _adaptive_budget_topup_attempt_metrics(records: list[dict[str, Any]]) -> dict[str, int]:
    """Count attempts where an adaptive budget top-up was granted."""
    attempts = 0
    for record in records:
        attempt = _as_int(record.get("attempt"))
        if attempt <= 0:
            continue
        attempt_meta = record.get("attempt_metadata")
        patch_meta = record.get("patch_meta")
        attempt_meta_dict = attempt_meta if isinstance(attempt_meta, dict) else {}
        patch_meta_dict = patch_meta if isinstance(patch_meta, dict) else {}
        granted = attempt_meta_dict.get(
            "adaptive_budget_topup_granted",
            patch_meta_dict.get("adaptive_budget_topup_granted"),
        )
        if bool(granted):
            attempts += 1
    return {"topup_attempts": attempts}


def _syntax_preflight_attempt_metrics(records: list[dict[str, Any]]) -> dict[str, int]:
    """Aggregate syntax-preflight cycle, failure, and enablement metrics."""
    enabled_attempts = 0
    cycles_sum = 0
    failures_sum = 0
    failed_attempts = 0
    for record in records:
        attempt = _as_int(record.get("attempt"))
        if attempt <= 0:
            continue
        note = str(record.get("note") or "")
        attempt_meta = record.get("attempt_metadata")
        patch_meta = record.get("patch_meta")
        attempt_meta_dict = attempt_meta if isinstance(attempt_meta, dict) else {}
        patch_meta_dict = patch_meta if isinstance(patch_meta, dict) else {}
        enabled = bool(
            attempt_meta_dict.get(
                "syntax_preflight_enabled",
                patch_meta_dict.get("syntax_preflight_enabled"),
            )
        )
        if enabled:
            enabled_attempts += 1
        cycles_value = attempt_meta_dict.get(
            "syntax_preflight_cycles_used",
            patch_meta_dict.get("syntax_preflight_cycles_used"),
        )
        failures_value = attempt_meta_dict.get(
            "syntax_preflight_failures",
            patch_meta_dict.get("syntax_preflight_failures"),
        )
        if isinstance(cycles_value, (int, float)):
            cycles_sum += int(cycles_value)
        if isinstance(failures_value, (int, float)):
            failures_sum += int(failures_value)
        if note == "syntax_preflight_failed":
            failed_attempts += 1
    return {
        "enabled_attempts": enabled_attempts,
        "cycles_sum": cycles_sum,
        "failures_sum": failures_sum,
        "failed_attempts": failed_attempts,
    }


def _patch_verifier_source_scope_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate patch-verifier source-scope and recovery metrics.

    Counts verifier invocations, source-scope matches, recovery
    eligibility, invocations, pass/fail results, and terminal notes.
    """
    verifier_attempts = 0
    source_scope_matches = 0
    recovery_eligible_attempts = 0
    recovery_invoked_attempts = 0
    recovery_passed_attempts = 0
    recovery_result_counts: dict[str, int] = defaultdict(int)
    recovery_terminal_note_counts: dict[str, int] = defaultdict(int)
    for record in records:
        attempt = _as_int(record.get("attempt"))
        if attempt <= 0:
            continue
        note = str(record.get("note") or "")
        attempt_meta = record.get("attempt_metadata")
        attempt_meta_dict = attempt_meta if isinstance(attempt_meta, dict) else {}
        patch_meta = record.get("patch_meta")
        patch_meta_dict = patch_meta if isinstance(patch_meta, dict) else {}
        if not patch_meta_dict:
            continue
        if not bool(patch_meta_dict.get("patch_verifier_enabled")):
            continue
        verifier_attempts += 1
        if bool(patch_meta_dict.get("patch_verifier_source_scope_matched")):
            source_scope_matches += 1
        eligible = bool(
            attempt_meta_dict.get(
                "patch_verifier_recovery_eligible",
                patch_meta_dict.get("patch_verifier_recovery_eligible"),
            )
        )
        invoked = bool(
            attempt_meta_dict.get(
                "patch_verifier_recovery_invoked",
                patch_meta_dict.get("patch_verifier_recovery_invoked"),
            )
        )
        result = str(
            attempt_meta_dict.get(
                "patch_verifier_recovery_result",
                patch_meta_dict.get("patch_verifier_recovery_result"),
            )
            or ""
        ).strip()
        if eligible:
            recovery_eligible_attempts += 1
        if invoked:
            recovery_invoked_attempts += 1
            if note:
                recovery_terminal_note_counts[note] += 1
        if invoked and result == "passed_verifier":
            recovery_passed_attempts += 1
        if (
            eligible
            or invoked
            or (patch_meta_dict.get("patch_verifier_passed") is False and result)
        ):
            recovery_result_counts[result or "not_eligible"] += 1
    return {
        "verifier_attempts": verifier_attempts,
        "source_scope_matches": source_scope_matches,
        "recovery_eligible_attempts": recovery_eligible_attempts,
        "recovery_invoked_attempts": recovery_invoked_attempts,
        "recovery_passed_attempts": recovery_passed_attempts,
        "recovery_result_counts": dict(sorted(recovery_result_counts.items())),
        "recovery_terminal_note_counts": dict(sorted(recovery_terminal_note_counts.items())),
    }


def _adaptive_helper_outcome(
    *,
    attempt_meta_dict: dict[str, Any],
    patch_meta_dict: dict[str, Any],
) -> str:
    """Classify the adaptive helper outcome for a single attempt.

    Returns one of: ``'materialized'``, ``'transport_failed'``,
    ``'unavailable_backend'``, ``'compat_reasoning_only'``,
    ``'request_budget_exceeded'``,
    ``'invalid_payload'``, ``'semantic_no_context'``,
    ``'skipped_unsupported'``, ``'skipped_guardrail'``,
    ``'skipped_policy'``, or ``'not_invoked'``.
    """
    explicit = attempt_meta_dict.get(
        "adaptive_helper_outcome",
        patch_meta_dict.get("adaptive_helper_outcome"),
    )
    if isinstance(explicit, str) and explicit.strip():
        normalized = explicit.strip()
        if normalized == "backend_unavailable":
            return "unavailable_backend"
        return normalized

    failure_kind_raw = attempt_meta_dict.get(
        "adaptive_helper_failure_kind",
        patch_meta_dict.get("adaptive_helper_failure_kind"),
    )
    failure_kind = str(failure_kind_raw or "").strip()

    helper_invoked_raw = attempt_meta_dict.get(
        "adaptive_helper_invoked",
        patch_meta_dict.get("adaptive_helper_invoked"),
    )
    helper_invoked = bool(helper_invoked_raw) if isinstance(helper_invoked_raw, bool) else False
    helper_reason = str(
        attempt_meta_dict.get(
            "adaptive_helper_decision_reason",
            patch_meta_dict.get("adaptive_helper_decision_reason"),
        )
        or ""
    ).strip()
    helper_materialized_raw = attempt_meta_dict.get(
        "adaptive_helper_materialized_context",
        patch_meta_dict.get("adaptive_helper_materialized_context"),
    )
    helper_materialized = (
        bool(helper_materialized_raw) if isinstance(helper_materialized_raw, bool) else False
    )
    if helper_invoked:
        if helper_materialized:
            return "materialized"
        if failure_kind == "transport":
            return "transport_failed"
        if failure_kind == "backend_unavailable":
            return "unavailable_backend"
        if failure_kind == "compat_reasoning_only":
            return "compat_reasoning_only"
        if failure_kind == "request_budget_exceeded":
            return "request_budget_exceeded"
        if failure_kind == "invalid_payload":
            return "invalid_payload"
        if helper_reason == "llm_invalid_response_no_helper":
            return "invalid_payload"
        return "semantic_no_context"
    if helper_reason == "request_context_unavailable":
        return "skipped_unsupported"
    if helper_reason in {
        "informativeness_pregate_skip",
        "helper_budget_exhausted",
        "helper_saturation_guard",
    }:
        return "skipped_guardrail"
    if helper_reason in {"not_adaptive_condition", ""}:
        return "not_invoked"
    return "skipped_policy"


def _helper_retry_count(
    *, attempt_meta_dict: dict[str, Any], patch_meta_dict: dict[str, Any]
) -> int:
    """Extract the helper retry count for a single attempt."""
    value = attempt_meta_dict.get(
        "adaptive_helper_retry_count",
        patch_meta_dict.get("adaptive_helper_retry_count"),
    )
    return _as_int(value)


def _helper_transport_retry_count(
    *,
    attempt_meta_dict: dict[str, Any],
    patch_meta_dict: dict[str, Any],
) -> int:
    """Extract the helper transport retry count for a single attempt."""
    value = attempt_meta_dict.get(
        "adaptive_helper_transport_retry_count",
        patch_meta_dict.get("adaptive_helper_transport_retry_count"),
    )
    return _as_int(value)


def _adaptive_helper_attempt_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate adaptive helper invocation and outcome metrics.

    Walks all attempt records to tally invocations, materialization,
    transport failures, retries, budget exhaustion, and saturation
    guard activations.
    """
    helper_invocation_attempts = 0
    helper_materialized_attempts = 0
    helper_low_yield_attempts = 0
    helper_transport_failure_attempts = 0
    helper_backend_unavailable_attempts = 0
    helper_compat_reasoning_only_attempts = 0
    helper_invalid_payload_attempts = 0
    helper_request_budget_exceeded_attempts = 0
    helper_retry_attempts = 0
    helper_transport_retry_attempts = 0
    helper_calls_used_final = 0
    helper_decision_attempts = 0
    helper_need_more_true_attempts = 0
    helper_need_more_false_attempts = 0
    helper_budget_exhausted_attempts = 0
    helper_saturation_guard_attempts = 0
    first_helper_attempt: int | None = None
    adaptive_mode = "none"

    for record in records:
        attempt = _as_int(record.get("attempt"))
        attempt_meta = record.get("attempt_metadata")
        patch_meta = record.get("patch_meta")
        attempt_meta_dict = attempt_meta if isinstance(attempt_meta, dict) else {}
        patch_meta_dict = patch_meta if isinstance(patch_meta, dict) else {}

        mode_raw = attempt_meta_dict.get("adaptive_mode", patch_meta_dict.get("adaptive_mode"))
        if isinstance(mode_raw, str) and mode_raw and adaptive_mode == "none":
            adaptive_mode = mode_raw
        if attempt > 0 and mode_raw in {"gated", "llm_discretion"}:
            helper_decision_attempts += 1

        helper_invoked_raw = attempt_meta_dict.get(
            "adaptive_helper_invoked", patch_meta_dict.get("adaptive_helper_invoked")
        )
        if isinstance(helper_invoked_raw, bool) and helper_invoked_raw:
            helper_invocation_attempts += 1
            if first_helper_attempt is None:
                first_helper_attempt = attempt
        helper_outcome = _adaptive_helper_outcome(
            attempt_meta_dict=attempt_meta_dict,
            patch_meta_dict=patch_meta_dict,
        )
        if helper_outcome == "materialized":
            helper_materialized_attempts += 1
        if helper_outcome == "semantic_no_context":
            helper_low_yield_attempts += 1
        if helper_outcome == "transport_failed":
            helper_transport_failure_attempts += 1
        if helper_outcome == "unavailable_backend":
            helper_backend_unavailable_attempts += 1
        if helper_outcome == "compat_reasoning_only":
            helper_compat_reasoning_only_attempts += 1
        if helper_outcome == "invalid_payload":
            helper_invalid_payload_attempts += 1
        if helper_outcome == "request_budget_exceeded":
            helper_request_budget_exceeded_attempts += 1

        helper_retry_attempts += _helper_retry_count(
            attempt_meta_dict=attempt_meta_dict,
            patch_meta_dict=patch_meta_dict,
        )
        helper_transport_retry_attempts += _helper_transport_retry_count(
            attempt_meta_dict=attempt_meta_dict,
            patch_meta_dict=patch_meta_dict,
        )

        helper_calls_after = attempt_meta_dict.get(
            "adaptive_helper_calls_used_after_attempt",
            patch_meta_dict.get("adaptive_helper_calls_used_after_attempt"),
        )
        if isinstance(helper_calls_after, (int, float)):
            helper_calls_used_final = max(helper_calls_used_final, int(helper_calls_after))

        need_more_context_raw = attempt_meta_dict.get(
            "adaptive_helper_need_more_context",
            patch_meta_dict.get("adaptive_helper_need_more_context"),
        )
        if isinstance(need_more_context_raw, bool):
            if need_more_context_raw:
                helper_need_more_true_attempts += 1
            else:
                helper_need_more_false_attempts += 1

        if bool(
            attempt_meta_dict.get(
                "adaptive_helper_budget_exhausted",
                patch_meta_dict.get("adaptive_helper_budget_exhausted"),
            )
        ):
            helper_budget_exhausted_attempts += 1
        if bool(
            attempt_meta_dict.get(
                "adaptive_helper_saturation_guarded",
                patch_meta_dict.get("adaptive_helper_saturation_guarded"),
            )
        ):
            helper_saturation_guard_attempts += 1

    return {
        "adaptive_mode": adaptive_mode,
        "helper_invocation_attempts": helper_invocation_attempts,
        "helper_materialized_attempts": helper_materialized_attempts,
        "helper_low_yield_attempts": helper_low_yield_attempts,
        "helper_transport_failure_attempts": helper_transport_failure_attempts,
        "helper_backend_unavailable_attempts": helper_backend_unavailable_attempts,
        "helper_compat_reasoning_only_attempts": helper_compat_reasoning_only_attempts,
        "helper_invalid_payload_attempts": helper_invalid_payload_attempts,
        "helper_request_budget_exceeded_attempts": helper_request_budget_exceeded_attempts,
        "helper_retry_attempts": helper_retry_attempts,
        "helper_transport_retry_attempts": helper_transport_retry_attempts,
        "helper_calls_used_final": helper_calls_used_final,
        "helper_decision_attempts": helper_decision_attempts,
        "helper_need_more_true_attempts": helper_need_more_true_attempts,
        "helper_need_more_false_attempts": helper_need_more_false_attempts,
        "helper_budget_exhausted_attempts": helper_budget_exhausted_attempts,
        "helper_saturation_guard_attempts": helper_saturation_guard_attempts,
        "first_helper_attempt": first_helper_attempt,
    }


def _adaptive_online_attempt_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate adaptive online-policy decision and reward metrics.

    Tracks per-action selection/application counts, overrides, policy
    updates, reward sums, propensity sums, local cost metrics, and
    delayed-rescue credit attribution.
    """
    adaptive_online_mode = "off"
    adaptive_online_policy_algo = ""
    decisions = 0
    selected_action_counts: dict[str, int] = defaultdict(int)
    applied_action_counts: dict[str, int] = defaultdict(int)
    behavior_action_counts: dict[str, int] = defaultdict(int)
    override_attempts = 0
    updates = 0
    reward_sum = 0.0
    reward_count = 0
    reward_values: list[float] = []
    propensity_sum = 0.0
    propensity_count = 0
    local_tokens_total_sum = 0.0
    local_tokens_total_count = 0
    local_wall_sec_sum = 0.0
    local_wall_sec_count = 0
    local_llm_call_count_sum = 0.0
    local_llm_call_count_count = 0
    reward_progress_score_sum = 0.0
    reward_progress_score_count = 0
    progress_reason_counts: dict[str, int] = defaultdict(int)
    delayed_rescue_bonus_sum = 0.0
    delayed_rescue_bonus_count = 0
    credited_rescues_plus_1 = 0
    credited_rescues_plus_2 = 0

    for record in records:
        attempt_meta = record.get("attempt_metadata")
        patch_meta = record.get("patch_meta")
        attempt_meta_dict = attempt_meta if isinstance(attempt_meta, dict) else {}
        patch_meta_dict = patch_meta if isinstance(patch_meta, dict) else {}

        mode_raw = attempt_meta_dict.get(
            "adaptive_online_policy_mode",
            patch_meta_dict.get("adaptive_online_policy_mode"),
        )
        if isinstance(mode_raw, str) and mode_raw and adaptive_online_mode == "off":
            adaptive_online_mode = mode_raw
        algo_raw = attempt_meta_dict.get(
            "adaptive_online_policy_algo",
            patch_meta_dict.get("adaptive_online_policy_algo"),
        )
        if isinstance(algo_raw, str) and algo_raw and not adaptive_online_policy_algo:
            adaptive_online_policy_algo = algo_raw

        decision_id = attempt_meta_dict.get(
            "adaptive_online_decision_id",
            patch_meta_dict.get("adaptive_online_decision_id"),
        )
        has_decision = isinstance(decision_id, str) and bool(decision_id.strip())
        reward_scope = _metadata_with_fallback(
            attempt_meta_dict,
            patch_meta_dict,
            key="adaptive_online_reward_scope",
        )
        has_local_scope = reward_scope in ADAPTIVE_ONLINE_REWARD_SCOPES
        if has_decision:
            decisions += 1

        selected_action = attempt_meta_dict.get(
            "adaptive_online_action_selected",
            patch_meta_dict.get("adaptive_online_action_selected"),
        )
        if isinstance(selected_action, str) and selected_action in ADAPTIVE_ONLINE_ACTIONS:
            selected_action_counts[selected_action] += 1
        applied_action = attempt_meta_dict.get(
            "adaptive_online_action_applied",
            patch_meta_dict.get("adaptive_online_action_applied"),
        )
        if isinstance(applied_action, str) and applied_action in ADAPTIVE_ONLINE_ACTIONS:
            applied_action_counts[applied_action] += 1
        behavior_action = attempt_meta_dict.get(
            "adaptive_online_behavior_action",
            patch_meta_dict.get("adaptive_online_behavior_action"),
        )
        if isinstance(behavior_action, str) and behavior_action in ADAPTIVE_ONLINE_ACTIONS:
            behavior_action_counts[behavior_action] += 1
        if (
            isinstance(selected_action, str)
            and isinstance(applied_action, str)
            and selected_action
            and applied_action
            and selected_action != "none"
            and applied_action != "none"
            and selected_action != applied_action
        ):
            override_attempts += 1

        updated_raw = attempt_meta_dict.get(
            "adaptive_online_updated",
            patch_meta_dict.get("adaptive_online_updated"),
        )
        if isinstance(updated_raw, bool) and updated_raw:
            updates += 1

        reward_raw = attempt_meta_dict.get(
            "adaptive_online_reward",
            patch_meta_dict.get("adaptive_online_reward"),
        )
        if has_decision and has_local_scope and isinstance(reward_raw, (int, float)):
            reward_value = float(reward_raw)
            reward_sum += reward_value
            reward_count += 1
            reward_values.append(reward_value)

        reward_progress_score_raw = _metadata_with_fallback(
            attempt_meta_dict,
            patch_meta_dict,
            key="adaptive_online_reward_progress_score",
        )
        if has_decision and has_local_scope and isinstance(reward_progress_score_raw, (int, float)):
            reward_progress_score_sum += float(reward_progress_score_raw)
            reward_progress_score_count += 1

        progress_reason_raw = _metadata_with_fallback(
            attempt_meta_dict,
            patch_meta_dict,
            key="adaptive_progress_reason",
        )
        if has_decision and has_local_scope and isinstance(progress_reason_raw, str):
            progress_reason = progress_reason_raw.strip()
            if progress_reason:
                progress_reason_counts[progress_reason] += 1

        propensity_raw = attempt_meta_dict.get(
            "adaptive_online_propensity",
            patch_meta_dict.get("adaptive_online_propensity"),
        )
        if has_decision and isinstance(propensity_raw, (int, float)):
            propensity_sum += float(propensity_raw)
            propensity_count += 1

        local_tokens_total_raw = _metadata_with_fallback(
            attempt_meta_dict,
            patch_meta_dict,
            key="adaptive_online_local_tokens_total",
        )
        if has_decision and has_local_scope and isinstance(local_tokens_total_raw, (int, float)):
            local_tokens_total_sum += float(local_tokens_total_raw)
            local_tokens_total_count += 1

        local_wall_sec_raw = _metadata_with_fallback(
            attempt_meta_dict,
            patch_meta_dict,
            key="adaptive_online_local_wall_sec",
        )
        if has_decision and has_local_scope and isinstance(local_wall_sec_raw, (int, float)):
            local_wall_sec_sum += float(local_wall_sec_raw)
            local_wall_sec_count += 1

        local_llm_call_count_raw = _metadata_with_fallback(
            attempt_meta_dict,
            patch_meta_dict,
            key="adaptive_online_local_llm_call_count",
        )
        if has_decision and has_local_scope and isinstance(local_llm_call_count_raw, (int, float)):
            local_llm_call_count_sum += float(local_llm_call_count_raw)
            local_llm_call_count_count += 1

        delayed_rescue_bonus_raw = _metadata_with_fallback(
            attempt_meta_dict,
            patch_meta_dict,
            key="adaptive_online_reward_delayed_rescue_bonus",
        )
        if has_decision and has_local_scope and isinstance(delayed_rescue_bonus_raw, (int, float)):
            delayed_rescue_bonus = float(delayed_rescue_bonus_raw)
            if delayed_rescue_bonus > 0.0:
                delayed_rescue_bonus_sum += delayed_rescue_bonus
                delayed_rescue_bonus_count += 1
                horizon_attempts = _as_int(
                    _metadata_with_fallback(
                        attempt_meta_dict,
                        patch_meta_dict,
                        key="adaptive_online_reward_credit_horizon_attempts",
                    )
                )
                if horizon_attempts == 1:
                    credited_rescues_plus_1 += 1
                elif horizon_attempts == 2:
                    credited_rescues_plus_2 += 1

    return {
        "adaptive_online_mode": adaptive_online_mode,
        "adaptive_online_policy_algo": adaptive_online_policy_algo,
        "decisions": decisions,
        "selected_action_counts": {
            action: int(selected_action_counts.get(action, 0)) for action in ADAPTIVE_ONLINE_ACTIONS
        },
        "applied_action_counts": {
            action: int(applied_action_counts.get(action, 0)) for action in ADAPTIVE_ONLINE_ACTIONS
        },
        "behavior_action_counts": {
            action: int(behavior_action_counts.get(action, 0)) for action in ADAPTIVE_ONLINE_ACTIONS
        },
        "selected_invoke_attempts": int(selected_action_counts.get("invoke_helper", 0)),
        "applied_invoke_attempts": int(applied_action_counts.get("invoke_helper", 0)),
        "override_attempts": override_attempts,
        "updates": updates,
        "reward_sum": reward_sum,
        "reward_count": reward_count,
        "reward_values": reward_values,
        "propensity_sum": propensity_sum,
        "propensity_count": propensity_count,
        "local_tokens_total_sum": local_tokens_total_sum,
        "local_tokens_total_count": local_tokens_total_count,
        "local_wall_sec_sum": local_wall_sec_sum,
        "local_wall_sec_count": local_wall_sec_count,
        "local_llm_call_count_sum": local_llm_call_count_sum,
        "local_llm_call_count_count": local_llm_call_count_count,
        "reward_progress_score_sum": reward_progress_score_sum,
        "reward_progress_score_count": reward_progress_score_count,
        "progress_reason_counts": dict(progress_reason_counts),
        "delayed_rescue_bonus_sum": delayed_rescue_bonus_sum,
        "delayed_rescue_bonus_count": delayed_rescue_bonus_count,
        "credited_rescues_plus_1": credited_rescues_plus_1,
        "credited_rescues_plus_2": credited_rescues_plus_2,
    }


def _adaptive_budget_policy_attempt_metrics(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate adaptive budget-policy decision and control metrics.

    Tracks budget increase selections, overrides, per-control-knob
    change counts, effective control values, reward/propensity sums,
    and applied multiplier statistics.
    """
    adaptive_budget_policy_mode = "off"
    adaptive_budget_policy_algo = ""
    adaptive_budget_policy_architecture = ""
    decisions = 0
    selected_increase_attempts = 0
    applied_increase_attempts = 0
    selected_time_attempts = 0
    selected_tokens_attempts = 0
    selected_context_attempts = 0
    applied_time_attempts = 0
    applied_tokens_attempts = 0
    applied_context_attempts = 0
    override_attempts = 0
    updates = 0
    reward_sum = 0.0
    reward_count = 0
    reward_values: list[float] = []
    propensity_sum = 0.0
    propensity_count = 0
    family_propensity_sum = 0.0
    family_propensity_count = 0
    request_budget_exceeded_attempts = 0
    applied_multiplier_sum = 0.0
    applied_multiplier_count = 0
    local_tokens_total_sum = 0.0
    local_tokens_total_count = 0
    local_wall_sec_sum = 0.0
    local_wall_sec_count = 0
    local_llm_call_count_sum = 0.0
    local_llm_call_count_count = 0
    reward_progress_score_sum = 0.0
    reward_progress_score_count = 0
    propose_budget_change_attempts = 0
    max_tokens_change_attempts = 0
    helper_allowance_change_attempts = 0
    stage_a_requests_change_attempts = 0
    stage_a_chars_change_attempts = 0
    retry_history_budget_change_attempts = 0
    compaction_allowance_change_attempts = 0
    response_mode_change_attempts = 0
    partial_override_attempts = 0
    effective_helper_max_calls_sum = 0.0
    effective_helper_max_calls_count = 0
    effective_stage_a_max_requests_sum = 0.0
    effective_stage_a_max_requests_count = 0
    effective_stage_a_max_chars_sum = 0.0
    effective_stage_a_max_chars_count = 0
    effective_retry_history_max_chars_sum = 0.0
    effective_retry_history_max_chars_count = 0
    effective_context_compaction_max_calls_sum = 0.0
    effective_context_compaction_max_calls_count = 0
    effective_openai_max_tokens_sum = 0.0
    effective_openai_max_tokens_count = 0

    for record in records:
        attempt_meta = record.get("attempt_metadata")
        patch_meta = record.get("patch_meta")
        attempt_meta_dict = attempt_meta if isinstance(attempt_meta, dict) else {}
        patch_meta_dict = patch_meta if isinstance(patch_meta, dict) else {}

        mode_raw = attempt_meta_dict.get(
            "adaptive_budget_policy_mode",
            patch_meta_dict.get("adaptive_budget_policy_mode"),
        )
        if isinstance(mode_raw, str) and mode_raw and adaptive_budget_policy_mode == "off":
            adaptive_budget_policy_mode = mode_raw
        algo_raw = attempt_meta_dict.get(
            "adaptive_budget_policy_algo",
            patch_meta_dict.get("adaptive_budget_policy_algo"),
        )
        if isinstance(algo_raw, str) and algo_raw and not adaptive_budget_policy_algo:
            adaptive_budget_policy_algo = algo_raw
        architecture_raw = attempt_meta_dict.get(
            "adaptive_budget_policy_architecture",
            patch_meta_dict.get("adaptive_budget_policy_architecture"),
        )
        if (
            isinstance(architecture_raw, str)
            and architecture_raw
            and not adaptive_budget_policy_architecture
        ):
            adaptive_budget_policy_architecture = architecture_raw

        decision_id = attempt_meta_dict.get(
            "adaptive_budget_decision_id",
            patch_meta_dict.get("adaptive_budget_decision_id"),
        )
        has_decision = isinstance(decision_id, str) and bool(decision_id.strip())
        reward_scope = _metadata_with_fallback(
            attempt_meta_dict,
            patch_meta_dict,
            key="adaptive_budget_reward_scope",
        )
        has_local_scope = reward_scope in ADAPTIVE_BUDGET_REWARD_SCOPES
        applied_controls_value = _metadata_with_fallback(
            attempt_meta_dict,
            patch_meta_dict,
            key="adaptive_budget_applied_control_values",
        )
        applied_controls = (
            applied_controls_value if isinstance(applied_controls_value, dict) else {}
        )
        if has_decision:
            decisions += 1

        selected_action = attempt_meta_dict.get(
            "adaptive_budget_action_selected",
            patch_meta_dict.get("adaptive_budget_action_selected"),
        )
        if selected_action == "increase_budget":
            selected_increase_attempts += 1
        selected_family = attempt_meta_dict.get(
            "adaptive_budget_family_selected",
            patch_meta_dict.get("adaptive_budget_family_selected"),
        )
        if has_decision and selected_family == "time":
            selected_time_attempts += 1
        if has_decision and selected_family == "tokens":
            selected_tokens_attempts += 1
        if has_decision and selected_family == "context":
            selected_context_attempts += 1
        applied_action = attempt_meta_dict.get(
            "adaptive_budget_action_applied",
            patch_meta_dict.get("adaptive_budget_action_applied"),
        )
        if applied_action == "increase_budget":
            applied_increase_attempts += 1
        applied_family = attempt_meta_dict.get(
            "adaptive_budget_family_applied",
            patch_meta_dict.get("adaptive_budget_family_applied"),
        )
        if has_decision and applied_family == "time":
            applied_time_attempts += 1
        if has_decision and applied_family == "tokens":
            applied_tokens_attempts += 1
        if has_decision and applied_family == "context":
            applied_context_attempts += 1
        family_decision_family = ""
        if isinstance(selected_family, str) and selected_family in ADAPTIVE_BUDGET_FAMILIES:
            family_decision_family = selected_family
        elif isinstance(applied_family, str) and applied_family in ADAPTIVE_BUDGET_FAMILIES:
            family_decision_family = applied_family
        if (
            isinstance(selected_action, str)
            and isinstance(applied_action, str)
            and selected_action
            and applied_action
            and selected_action != "none"
            and applied_action != "none"
            and selected_action != applied_action
        ):
            override_attempts += 1

        updated_raw = attempt_meta_dict.get(
            "adaptive_budget_updated",
            patch_meta_dict.get("adaptive_budget_updated"),
        )
        if isinstance(updated_raw, bool) and updated_raw:
            updates += 1

        reward_raw = attempt_meta_dict.get(
            "adaptive_budget_reward",
            patch_meta_dict.get("adaptive_budget_reward"),
        )
        if has_decision and has_local_scope and isinstance(reward_raw, (int, float)):
            reward_value = float(reward_raw)
            reward_sum += reward_value
            reward_count += 1
            reward_values.append(reward_value)

        reward_progress_score_raw = _metadata_with_fallback(
            attempt_meta_dict,
            patch_meta_dict,
            key="adaptive_budget_reward_progress_score",
        )
        if has_decision and has_local_scope and isinstance(reward_progress_score_raw, (int, float)):
            reward_progress_score_sum += float(reward_progress_score_raw)
            reward_progress_score_count += 1

        propensity_raw = attempt_meta_dict.get(
            "adaptive_budget_propensity",
            patch_meta_dict.get("adaptive_budget_propensity"),
        )
        if has_decision and isinstance(propensity_raw, (int, float)):
            propensity_sum += float(propensity_raw)
            propensity_count += 1
        family_propensity_raw = attempt_meta_dict.get(
            "adaptive_budget_family_propensity",
            patch_meta_dict.get("adaptive_budget_family_propensity"),
        )
        if (
            has_decision
            and family_decision_family
            and isinstance(family_propensity_raw, (int, float))
        ):
            family_propensity_sum += float(family_propensity_raw)
            family_propensity_count += 1

        request_budget_exceeded_raw = attempt_meta_dict.get(
            "adaptive_budget_request_budget_exceeded",
            patch_meta_dict.get("adaptive_budget_request_budget_exceeded"),
        )
        if (
            has_decision
            and isinstance(request_budget_exceeded_raw, bool)
            and request_budget_exceeded_raw
        ):
            request_budget_exceeded_attempts += 1

        applied_multiplier_raw = attempt_meta_dict.get(
            "adaptive_budget_applied_multiplier",
            patch_meta_dict.get("adaptive_budget_applied_multiplier"),
        )
        if has_decision and isinstance(applied_multiplier_raw, (int, float)):
            applied_multiplier_sum += float(applied_multiplier_raw)
            applied_multiplier_count += 1
        default_multiplier_raw = attempt_meta_dict.get(
            "adaptive_budget_default_multiplier",
            patch_meta_dict.get("adaptive_budget_default_multiplier"),
        )

        applied_controls_changed_value = _metadata_with_fallback(
            attempt_meta_dict,
            patch_meta_dict,
            key="adaptive_budget_applied_controls_changed",
        )
        if isinstance(applied_controls_changed_value, dict):
            applied_controls_changed = {
                str(key): bool(value) for key, value in applied_controls_changed_value.items()
            }
        else:
            legacy_propose_change = bool(
                isinstance(applied_multiplier_raw, (int, float))
                and isinstance(default_multiplier_raw, (int, float))
                and float(applied_multiplier_raw) > float(default_multiplier_raw)
            )
            applied_controls_changed = {
                "propose_budget": legacy_propose_change,
                "max_tokens": False,
                "helper_allowance": False,
                "stage_a_requests": False,
                "stage_a_chars": False,
                "retry_history_budget": False,
                "compaction_allowance": False,
                "response_mode": False,
            }
        if has_decision and bool(applied_controls_changed.get("propose_budget")):
            propose_budget_change_attempts += 1
        if has_decision and bool(applied_controls_changed.get("max_tokens")):
            max_tokens_change_attempts += 1
        if has_decision and bool(applied_controls_changed.get("helper_allowance")):
            helper_allowance_change_attempts += 1
        if has_decision and bool(applied_controls_changed.get("stage_a_requests")):
            stage_a_requests_change_attempts += 1
        if has_decision and bool(applied_controls_changed.get("stage_a_chars")):
            stage_a_chars_change_attempts += 1
        if has_decision and bool(applied_controls_changed.get("retry_history_budget")):
            retry_history_budget_change_attempts += 1
        if has_decision and bool(applied_controls_changed.get("compaction_allowance")):
            compaction_allowance_change_attempts += 1
        if has_decision and bool(applied_controls_changed.get("response_mode")):
            response_mode_change_attempts += 1

        partial_override_raw = _metadata_with_fallback(
            attempt_meta_dict,
            patch_meta_dict,
            key="adaptive_budget_partial_override",
        )
        if has_decision and bool(partial_override_raw):
            partial_override_attempts += 1

        local_tokens_total_raw = _metadata_with_fallback(
            attempt_meta_dict,
            patch_meta_dict,
            key="adaptive_budget_local_tokens_total",
        )
        if has_decision and has_local_scope and isinstance(local_tokens_total_raw, (int, float)):
            local_tokens_total_sum += float(local_tokens_total_raw)
            local_tokens_total_count += 1

        local_wall_sec_raw = _metadata_with_fallback(
            attempt_meta_dict,
            patch_meta_dict,
            key="adaptive_budget_local_wall_sec",
        )
        if has_decision and has_local_scope and isinstance(local_wall_sec_raw, (int, float)):
            local_wall_sec_sum += float(local_wall_sec_raw)
            local_wall_sec_count += 1

        local_llm_call_count_raw = _metadata_with_fallback(
            attempt_meta_dict,
            patch_meta_dict,
            key="adaptive_budget_local_llm_call_count",
        )
        if has_decision and has_local_scope and isinstance(local_llm_call_count_raw, (int, float)):
            local_llm_call_count_sum += float(local_llm_call_count_raw)
            local_llm_call_count_count += 1

        effective_helper_max_calls_raw = _metadata_with_fallback(
            attempt_meta_dict,
            patch_meta_dict,
            key="adaptive_budget_effective_helper_max_calls",
        )
        if effective_helper_max_calls_raw is None and "helper_max_calls" in applied_controls:
            effective_helper_max_calls_raw = applied_controls.get("helper_max_calls")
        if has_decision and isinstance(effective_helper_max_calls_raw, (int, float)):
            effective_helper_max_calls_sum += float(effective_helper_max_calls_raw)
            effective_helper_max_calls_count += 1

        effective_stage_a_max_requests_raw = _metadata_with_fallback(
            attempt_meta_dict,
            patch_meta_dict,
            key="adaptive_budget_effective_stage_a_max_requests",
        )
        if (
            effective_stage_a_max_requests_raw is None
            and "stage_a_max_requests" in applied_controls
        ):
            effective_stage_a_max_requests_raw = applied_controls.get("stage_a_max_requests")
        if has_decision and isinstance(effective_stage_a_max_requests_raw, (int, float)):
            effective_stage_a_max_requests_sum += float(effective_stage_a_max_requests_raw)
            effective_stage_a_max_requests_count += 1

        effective_stage_a_max_chars_raw = _metadata_with_fallback(
            attempt_meta_dict,
            patch_meta_dict,
            key="adaptive_budget_effective_stage_a_max_chars",
        )
        if effective_stage_a_max_chars_raw is None and "stage_a_max_chars" in applied_controls:
            effective_stage_a_max_chars_raw = applied_controls.get("stage_a_max_chars")
        if has_decision and isinstance(effective_stage_a_max_chars_raw, (int, float)):
            effective_stage_a_max_chars_sum += float(effective_stage_a_max_chars_raw)
            effective_stage_a_max_chars_count += 1

        effective_retry_history_max_chars_raw = _metadata_with_fallback(
            attempt_meta_dict,
            patch_meta_dict,
            key="adaptive_budget_effective_retry_history_max_chars",
        )
        if (
            effective_retry_history_max_chars_raw is None
            and "retry_history_max_chars" in applied_controls
        ):
            effective_retry_history_max_chars_raw = applied_controls.get("retry_history_max_chars")
        if has_decision and isinstance(effective_retry_history_max_chars_raw, (int, float)):
            effective_retry_history_max_chars_sum += float(effective_retry_history_max_chars_raw)
            effective_retry_history_max_chars_count += 1

        effective_context_compaction_max_calls_raw = _metadata_with_fallback(
            attempt_meta_dict,
            patch_meta_dict,
            key="adaptive_budget_effective_context_compaction_max_calls",
        )
        if (
            effective_context_compaction_max_calls_raw is None
            and "context_compaction_max_calls" in applied_controls
        ):
            effective_context_compaction_max_calls_raw = applied_controls.get(
                "context_compaction_max_calls"
            )
        if has_decision and isinstance(effective_context_compaction_max_calls_raw, (int, float)):
            effective_context_compaction_max_calls_sum += float(
                effective_context_compaction_max_calls_raw
            )
            effective_context_compaction_max_calls_count += 1

        effective_openai_max_tokens_raw = _metadata_with_fallback(
            attempt_meta_dict,
            patch_meta_dict,
            key="adaptive_budget_effective_openai_max_tokens",
        )
        if effective_openai_max_tokens_raw is None and "openai_max_tokens" in applied_controls:
            effective_openai_max_tokens_raw = applied_controls.get("openai_max_tokens")
        if has_decision and isinstance(effective_openai_max_tokens_raw, (int, float)):
            effective_openai_max_tokens_sum += float(effective_openai_max_tokens_raw)
            effective_openai_max_tokens_count += 1

    return {
        "adaptive_budget_policy_mode": adaptive_budget_policy_mode,
        "adaptive_budget_policy_algo": adaptive_budget_policy_algo,
        "adaptive_budget_policy_architecture": adaptive_budget_policy_architecture,
        "decisions": decisions,
        "selected_increase_attempts": selected_increase_attempts,
        "applied_increase_attempts": applied_increase_attempts,
        "selected_time_attempts": selected_time_attempts,
        "selected_tokens_attempts": selected_tokens_attempts,
        "selected_context_attempts": selected_context_attempts,
        "applied_time_attempts": applied_time_attempts,
        "applied_tokens_attempts": applied_tokens_attempts,
        "applied_context_attempts": applied_context_attempts,
        "override_attempts": override_attempts,
        "updates": updates,
        "reward_sum": reward_sum,
        "reward_count": reward_count,
        "reward_values": reward_values,
        "propensity_sum": propensity_sum,
        "propensity_count": propensity_count,
        "family_propensity_sum": family_propensity_sum,
        "family_propensity_count": family_propensity_count,
        "request_budget_exceeded_attempts": request_budget_exceeded_attempts,
        "applied_multiplier_sum": applied_multiplier_sum,
        "applied_multiplier_count": applied_multiplier_count,
        "local_tokens_total_sum": local_tokens_total_sum,
        "local_tokens_total_count": local_tokens_total_count,
        "local_wall_sec_sum": local_wall_sec_sum,
        "local_wall_sec_count": local_wall_sec_count,
        "local_llm_call_count_sum": local_llm_call_count_sum,
        "local_llm_call_count_count": local_llm_call_count_count,
        "reward_progress_score_sum": reward_progress_score_sum,
        "reward_progress_score_count": reward_progress_score_count,
        "propose_budget_change_attempts": propose_budget_change_attempts,
        "max_tokens_change_attempts": max_tokens_change_attempts,
        "helper_allowance_change_attempts": helper_allowance_change_attempts,
        "stage_a_requests_change_attempts": stage_a_requests_change_attempts,
        "stage_a_chars_change_attempts": stage_a_chars_change_attempts,
        "retry_history_budget_change_attempts": retry_history_budget_change_attempts,
        "compaction_allowance_change_attempts": compaction_allowance_change_attempts,
        "response_mode_change_attempts": response_mode_change_attempts,
        "partial_override_attempts": partial_override_attempts,
        "effective_helper_max_calls_sum": effective_helper_max_calls_sum,
        "effective_helper_max_calls_count": effective_helper_max_calls_count,
        "effective_stage_a_max_requests_sum": effective_stage_a_max_requests_sum,
        "effective_stage_a_max_requests_count": effective_stage_a_max_requests_count,
        "effective_stage_a_max_chars_sum": effective_stage_a_max_chars_sum,
        "effective_stage_a_max_chars_count": effective_stage_a_max_chars_count,
        "effective_retry_history_max_chars_sum": effective_retry_history_max_chars_sum,
        "effective_retry_history_max_chars_count": effective_retry_history_max_chars_count,
        "effective_context_compaction_max_calls_sum": effective_context_compaction_max_calls_sum,
        "effective_context_compaction_max_calls_count": effective_context_compaction_max_calls_count,
        "effective_openai_max_tokens_sum": effective_openai_max_tokens_sum,
        "effective_openai_max_tokens_count": effective_openai_max_tokens_count,
    }
