"""Per-unit and per-condition eval attempt flow helpers.

Extracted from ``evals.run_eval`` to keep the public facade module focused on
stable compatibility exports.
"""
# ruff: noqa: F821

from __future__ import annotations

import inspect
import shutil
import sys
import threading
import time
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from baselines.contracts import BASELINE_CONTRACT_VERSION
from evals import _json as json
from evals.adaptive_budget_online_policy import (
    ACTION_BUDGET_FAMILY_CONTEXT,
    ACTION_BUDGET_FAMILY_TIME,
    ACTION_BUDGET_FAMILY_TOKENS,
    ACTION_INCREASE_BUDGET,
    ACTION_KEEP_BUDGET,
    BUDGET_POLICY_ARCH_HIERARCHICAL_V1,
    BUDGET_POLICY_ARCH_TIER_BINARY,
    BUDGET_POLICY_ARCHITECTURES,
    BUDGET_POLICY_FAMILIES,
)
from evals.adaptive_controller_packet import (
    HYBRID_GUIDANCE_MODE_FIXED_V1,
    remove_adaptive_controller_sidecars,
)
from evals.adaptive_controller_runtime import (
    ControllerRuntimeInputs,
    prepare_controller_runtime,
    write_controller_sidecars_for_cycle,
)
from evals.adaptive_history import persist_adaptive_attempt_history
from evals.adaptive_online_policy import (
    ACTION_COMPACT_THEN_PATCH,
    ACTION_FRAME_LOCALS_LITE,
    ACTION_INVOKE_HELPER,
    ACTION_RETRY_HELPER_STRICT_JSON,
    ACTION_SKIP_HELPER,
    ACTION_SNAPSHOT_FULL_LITE,
    ACTION_SNAPSHOT_LITE,
)
from evals.adaptive_tep import (
    AdaptiveTepController,
    run_post_attempt_learning,
)
from evals.attempt_metadata import (
    AdaptiveAttemptMetadata,
    ContextCompactionMetadata,
    OpenAIAttemptMetadata,
    PromptBudgetMetadata,
    RetryAttemptMetadata,
    StageAAttemptMetadata,
    SyntaxPreflightMetadata,
    build_attempt_metadata_request,
)
from evals.attempt_verification import (
    apply_diff_with_optional_recovery,
    evaluate_patch_sanitize,
    evaluate_syntax_preflight,
    make_syntax_preflight_state,
)
from evals.eval_attempt_support import (
    _adaptive_gate_should_invoke_helper,
    _adaptive_helper_outcome_for_attempt,
    _adaptive_pipeline_stage_before,
    _adaptive_pipeline_stage_for_outcome,
    _adaptive_prompt_budget_for_attempt,
    _adaptive_should_stop_for_no_progress,
    _aggregate_attempt_llm_usage,
    _append_retry_history_turn_if_enabled,
    _apply_no_progress_token_cap_candidate,
    _apply_usage_totals_to_patch_meta,
    _build_attempt_outcome_record,
    _build_failure_signature_for_attempt,
    _build_infra_event_record,
    _build_structured_previous_attempt_diff,
    _classify_patcher_request_failure,
    _collect_file_index_lines,
    _combine_reasoning_usage_segments,
    _completion_chars_from_response,
    _effective_patch_verifier_scope,
    _enrich_failure_context,
    _ensure_deterministic_seed_ack,
    _finalize_breakdowns,
    _format_duration_precise,
    _format_progress_token_usage,
    _is_context_limit_http_error_message,
    _is_retryable_transport_http_status,
    _is_snapshot_condition,
    _is_transport_runtime_error,
    _llm_call_count_from_meta,
    _mark_adaptive_progress_stop,
    _measure_attempt_wall_clock_sec,
    _nonnegative_int,
    _normalized_progress_verbosity,
    _openai_preflight_total_char_cap,
    _patch_attempt_char_deltas,
    _plan_patch_with_snapshot_context,
    _progress_is_enabled,
    _progress_print,
    _prompt_chars_for_llm_calls,
    _propose_patch_with_snapshot_context,
    _reasoning_usage_segment_from_meta,
    _recompute_adaptive_progress_for_attempt_outcome,
    _record_token_usage_state,
    _remaining_attempt_propose_budget_sec,
    _request_context_failure_kind_from_meta,
    _request_context_recovery_failure_kind_from_meta,
    _request_context_retry_counts_from_meta,
    _resolve_run_condition_inputs,
    _result_meta_dict,
    _result_with_meta,
    _score_patch_relevance,
    _summarize_context_compaction_meta,
    _summarize_recovery_usage_meta,
    _sync_adaptive_progress_metadata,
    _sync_attempt_prompt_char_metadata,
    _sync_attempt_usage_metadata,
    _sync_patch_diff_metadata,
    _sync_patch_sanitize_metadata,
    _sync_patch_verifier_metadata,
    _sync_patch_verifier_recovery_metadata,
    _sync_patcher_request_failure_metadata,
    _sync_syntax_preflight_metadata,
    _timeout_fields,
    _traceback_informativeness_pre_gate,
    _transport_recovery_failure_kind_from_meta,
    _truncate_snapshot_for_artifact,
    _usage_token_counts_from_patch_meta,
    _write_run_outputs,
    apply_diff,
    collect_text_context,
    default_lambda_rlm_repo_dir,
    invoke_lambda_rlm_decomposition,
    make_patcher,
    read_latest_snapshot,
    run_one_condition,
    run_pytest,
)
from evals.eval_config import (
    _parse_adaptive_budget_policy_tiers,
    _parse_adaptive_online_action_score_biases,
)
from evals.eval_evidence import (
    _coerce_context_request_payload,
    _compute_evidence_fingerprint,
    _evidence_ids_from_items,
    _merge_evidence_items,
    _render_evidence_index,
    _render_evidence_items,
    _strip_eval_only_case_artifacts,
    build_fallback_context_bundle,
    execute_context_requests,
)
from evals.eval_patches import (
    _build_runner_forced_attempt_diff,
    _is_forced_noop_marker_diff,
    _render_patch_verifier_recovery_feedback,
)
from evals.eval_policy_budget import (
    AdaptiveBudgetMetadataUpdate,
    _finalize_adaptive_policy_attempt,
    _sync_adaptive_budget_policy_metadata,
)
from evals.eval_policy_features import (
    _adaptive_budget_control_changes,
    _adaptive_budget_control_profile_to_dict,
    _adaptive_budget_controls_changed,
    _adaptive_budget_family_availability,
    _adaptive_budget_feature_vector,
    _adaptive_budget_level_step_delta,
    _adaptive_budget_next_tier_index,
    _adaptive_budget_recorded_multiplier,
    _adaptive_budget_union_change_flags,
    _adaptive_online_feature_vector,
    _adaptive_policy_local_accounting,
    _apply_adaptive_budget_profile_to_patcher,
    _build_adaptive_budget_control_profile,
    _build_adaptive_budget_family_profile,
)
from evals.eval_policy_online import (
    AdaptiveOnlineMetadataUpdate,
    _adaptive_online_delayed_rescue_bonus,
    _resolve_pending_adaptive_online_credit,
    _sync_adaptive_online_metadata,
)
from evals.eval_prompts import (
    _apply_prompt_budget,
    _build_retry_history_section,
    _build_snapshot_prompt_text,
    _format_frame_locals_section,
    _format_snapshot_full_section,
    _format_snapshot_index_section,
    _is_generic_snapshot_justification,
    _render_retry_packet,
    prune_snapshot_for_prompt,
    render_stage_a_request_prompt,
)
from evals.eval_recording import (
    _build_base_record,
    _build_model_metadata,
    _default_run_metadata,
    _empty_attempt_metadata,
    _materialize_attempt_metadata_request,
)
from evals.eval_types import (
    ADAPTIVE_BUDGET_POLICY_ALGOS,
    ADAPTIVE_BUDGET_POLICY_MODES,
    ADAPTIVE_DELAYED_RESCUE_HORIZON_ATTEMPTS,
    ADAPTIVE_GATE_POLICIES,
    ADAPTIVE_HELPER_BUDGET_PROFILES,
    ADAPTIVE_HELPER_OUTCOME_COMPAT_REASONING_ONLY,
    ADAPTIVE_HELPER_OUTCOME_INVALID_PAYLOAD,
    ADAPTIVE_HELPER_OUTCOME_MATERIALIZED,
    ADAPTIVE_HELPER_OUTCOME_NOT_INVOKED,
    ADAPTIVE_HELPER_OUTCOME_REQUEST_BUDGET_EXCEEDED,
    ADAPTIVE_HELPER_OUTCOME_SEMANTIC_NO_CONTEXT,
    ADAPTIVE_HELPER_OUTCOME_TRANSPORT_FAILED,
    ADAPTIVE_HELPER_OUTCOME_UNAVAILABLE_BACKEND,
    ADAPTIVE_ONLINE_POLICY_ALGOS,
    ADAPTIVE_ONLINE_POLICY_MODES,
    ADAPTIVE_OPENAI_HELPER_RESPONSE_MODES,
    ADAPTIVE_REVEAL_MODES,
    CONTEXT_COMPACTION_MODES,
    CONTEXT_COMPACTION_SCOPE_TOKENS,
    CONTEXT_PROTOCOLS,
    CONTEXT_RECOVERY_FAILURE_KINDS,
    CONTEXT_RECOVERY_PROMPT_BUDGET,
    HELPER_STAGE_A_ACTIONS,
    OPENAI_CONTEXT_WINDOW_RESERVE_TOKENS,
    OPENAI_RESPONSE_MODE_AUTOSWITCH_MODES,
    PATCH_VERIFIER_RECOVERY_RESULT_NO_DIFF,
    PATCH_VERIFIER_RECOVERY_RESULT_NOT_ELIGIBLE,
    PATCH_VERIFIER_RECOVERY_RESULT_PASSED_VERIFIER,
    PATCH_VERIFIER_RECOVERY_RESULT_REQUEST_FAILED,
    PATCH_VERIFIER_RECOVERY_RESULT_SKIPPED_DEADLINE_EXHAUSTED,
    PATCH_VERIFIER_RECOVERY_RESULT_STILL_REJECTED,
    PATCH_VERIFIER_SCOPE_ADAPTIVE_ONLY,
    PATCH_VERIFIER_SCOPE_ALL,
    PATCH_VERIFIER_SCOPE_DISABLED,
    PROMPT_ORDER_MODES,
    RETRY_CONTEXT_MODES,
    RETRY_PROMPT_MODES,
    SNAPSHOT_FRAME_PRUNING_MODES,
    AdaptiveProgressState,
    CaseSpec,
    Condition,
    EvalUnitInputs,
    PendingAdaptiveOnlineCredit,
    PreGateResult,
    RunConditionInputs,
    RunResult,
    StageAContextData,
    UnitExecutionOutcome,
    _default_helper_action,
    _empty_stage_a_context,
    _filter_context_request_payload_for_action,
    _helper_action_availability,
)
from evals.native_openai_attempt_flow import (
    NativeOpenAIAttemptPromptRequest,
    assemble_native_openai_attempt_prompt,
)
from evals.native_openai_attempt_flow import (
    maybe_autoswitch_openai_response_mode as _maybe_autoswitch_openai_response_mode,
)
from evals.native_openai_attempt_flow import (
    sync_openai_response_mode_autoswitch_metadata as _sync_openai_response_mode_autoswitch_metadata,
)
from evals.native_stage_b_flow import StageBPromptInputs, build_stage_b_prompt_sections
from evals.patchers.base import Patcher
from evals.patchers.baseline_patcher import resolve_effective_baseline_model
from evals.policy_feature_scaling import normalize_feature_scaling_profile, scale_feature_mapping
from evals.stage_a_contract import StageARequestPayloadModel, normalize_stage_a_request_payload
from evals.token_counting import estimate_text_token_count
from evals.toon_llm_encoder import estimate_token_savings

# Preserve the historical eval_attempt_flow helper surface for downstream imports.
_COMPAT_HELPER_EXPORTS = (
    _adaptive_pipeline_stage_before,
    _adaptive_pipeline_stage_for_outcome,
    _is_context_limit_http_error_message,
    _is_retryable_transport_http_status,
    _is_transport_runtime_error,
)


def _decide_adaptive_helper_recovery_topup(
    *,
    is_adaptive_condition: bool,
    helper_budget_exhausted: bool,
    adaptive_budget_unlimited: bool,
    adaptive_budget_topup_calls_used: int,
    adaptive_budget_topup_max_calls: int,
    pregate_should_skip: bool,
    helper_saturation_guarded: bool,
    supports_context_request: bool,
    supports_strict_json_first: bool,
    previous_note: str,
    previous_helper_failure_kind: str,
    previous_progress_reason: str,
    previous_helper_need_more_context: bool | None,
    last_helper_need_more_context: bool | None,
    failure_signature_repeat: bool,
    adaptive_no_progress_streak: int,
) -> tuple[str, str, str, str]:
    if not (
        is_adaptive_condition
        and helper_budget_exhausted
        and (not adaptive_budget_unlimited)
        and adaptive_budget_topup_max_calls > 0
        and adaptive_budget_topup_calls_used < adaptive_budget_topup_max_calls
        and (not pregate_should_skip)
        and (not helper_saturation_guarded)
        and supports_context_request
    ):
        return "", "", "", ""
    if previous_helper_failure_kind == "invalid_payload" and supports_strict_json_first:
        return (
            "invalid_payload_recovery_once",
            ACTION_RETRY_HELPER_STRICT_JSON,
            "constraint_invalid_payload_force_strict_retry",
            "recovery_force_retry_helper_strict_json",
        )
    if previous_progress_reason == "no_progress_v2":
        return (
            "no_progress_recovery_once",
            ACTION_INVOKE_HELPER,
            "constraint_no_progress_force_helper",
            "recovery_force_invoke_helper",
        )
    if previous_note in {
        "patch_verifier_rejected",
        "syntax_preflight_failed",
        "retry_gate_no_new_evidence",
    }:
        return (
            "repair_failure_recovery_once",
            ACTION_INVOKE_HELPER,
            "constraint_repair_failure_force_helper",
            "recovery_force_invoke_helper",
        )
    if (
        (last_helper_need_more_context is True or previous_helper_need_more_context is True)
        and failure_signature_repeat
        and adaptive_no_progress_streak > 0
    ):
        return (
            "need_more_context_recovery_once",
            ACTION_INVOKE_HELPER,
            "constraint_need_more_context_force_helper",
            "recovery_force_invoke_helper",
        )
    return "", "", "", ""


_PATCH_USAGE_SYNC_META_KEYS = (
    "llm_call_count",
    "tokens_prompt",
    "tokens_completion",
    "tokens_total",
    "tokens_reasoning",
    "usage",
    "token_usage_available",
    "reasoning_token_usage_available",
    "tokens_input",
    "tokens_output",
)


def _reset_patch_usage_sync_metadata(patch_meta: dict[str, Any]) -> None:
    """Clear attempt-synced usage fields before merging fresh patch metadata."""
    for key in _PATCH_USAGE_SYNC_META_KEYS:
        patch_meta.pop(key, None)


def _tep_jaccard_similarity(text_a: str, text_b: str) -> float:
    """Word-level Jaccard similarity between two texts."""
    words_a = set(text_a.lower().split())
    words_b = set(text_b.lower().split())
    if not words_a and not words_b:
        return 1.0
    if not words_a or not words_b:
        return 0.0
    return len(words_a & words_b) / len(words_a | words_b)


def _build_full_tep_node_packets(
    *,
    note: str,
    success: bool,
    stage_a: dict[str, Any],
    context_compaction: dict[str, Any],
    retry_plan: dict[str, Any],
    patch: dict[str, Any],
) -> list[dict[str, str]]:
    """Build full-mode TEP packets for only the nodes that actually ran."""
    node_packets: list[dict[str, str]] = []

    if bool(stage_a.get("invoked")):
        stage_a_text = str(stage_a.get("text") or "")
        stage_a_output_summary = json.dumps(
            stage_a.get("request_payload") or {},
            ensure_ascii=False,
        )
        if stage_a_text:
            stage_a_output_summary += "\n" + stage_a_text
        node_packets.append(
            {
                "node_id": "stage_a_request",
                "input_summary": str(stage_a.get("input_summary") or ""),
                "output_summary": stage_a_output_summary,
                "objective_packet": json.dumps(
                    {
                        "attempt_note": note,
                        "success": success,
                        "helper_decision_reason": str(stage_a.get("helper_decision_reason") or ""),
                        "requests_count": int(stage_a.get("requests_count") or 0),
                        "request_success": bool(stage_a.get("request_success")),
                        "materialized_context": bool(stage_a.get("materialized_context")),
                    },
                    ensure_ascii=False,
                ),
                "note": note,
            }
        )

    if bool(context_compaction.get("invoked")):
        node_packets.append(
            {
                "node_id": "context_compaction",
                "input_summary": str(context_compaction.get("input_summary") or ""),
                "output_summary": str(context_compaction.get("output_summary") or ""),
                "objective_packet": json.dumps(
                    {
                        "attempt_note": note,
                        "success": success,
                        "compaction_requested": bool(context_compaction.get("requested")),
                        "compaction_invoked": bool(context_compaction.get("invoked")),
                        "compaction_success": bool(context_compaction.get("success")),
                        "compaction_trigger": str(context_compaction.get("trigger") or ""),
                        "compaction_ratio": float(context_compaction.get("ratio") or 0.0),
                    },
                    ensure_ascii=False,
                ),
                "note": note,
            }
        )

    if bool(retry_plan.get("invoked")):
        node_packets.append(
            {
                "node_id": "retry_planner",
                "input_summary": str(retry_plan.get("input_summary") or ""),
                "output_summary": str(retry_plan.get("output_summary") or ""),
                "objective_packet": json.dumps(
                    {
                        "attempt_note": note,
                        "success": success,
                        "plan_invoked": bool(retry_plan.get("invoked")),
                        "plan_succeeded": bool(retry_plan.get("succeeded")),
                        "plan_injected": bool(retry_plan.get("injected")),
                        "plan_error": str(retry_plan.get("error") or ""),
                    },
                    ensure_ascii=False,
                ),
                "note": note,
            }
        )

    node_packets.append(
        {
            "node_id": "patch_proposer",
            "input_summary": str(patch.get("input_summary") or ""),
            "output_summary": str(patch.get("output_summary") or note),
            "objective_packet": json.dumps(
                {
                    "attempt_note": note,
                    "success": success,
                    "patch_verifier_score": patch.get("patch_verifier_score"),
                    "patch_verifier_passed": patch.get("patch_verifier_passed"),
                    "syntax_preflight_cycles_used": patch.get("syntax_preflight_cycles_used"),
                    "postpatch_signature_repeat": patch.get("postpatch_signature_repeat"),
                },
                ensure_ascii=False,
            ),
            "note": note,
        }
    )

    return node_packets


def _update_aux_signals_from_attempt(
    *,
    online_kwargs: dict[str, Any],
    budget_kwargs: dict[str, Any],
    patch_meta: dict[str, Any],
    new_evidence_ids: list[str],
    reused_evidence_ids: list[str],
    last_context_compaction_ratio: float,
    pregate_should_skip: bool,
    helper_action_executed: str,
) -> None:
    """Populate auxiliary reward signals on finalize kwargs from attempt data."""
    # pytest fail reduction — from adaptive progress component.
    pytest_fail_reduction = max(
        0.0,
        min(1.0, float(patch_meta.get("adaptive_progress_component_pytest_fail_reduction", 0.0))),
    )
    # Evidence novelty — fraction of new evidence IDs.
    total_evidence = len(new_evidence_ids) + len(reused_evidence_ids)
    evidence_novelty = len(new_evidence_ids) / max(1, total_evidence) if total_evidence > 0 else 0.0
    # Compaction efficiency — ratio from last compaction pass.
    compaction_ratio = max(0.0, min(1.0, float(last_context_compaction_ratio)))
    # Informativeness alignment is based on the executed action, not the
    # decision reason. A skip only aligns when the pregate itself advised skip.
    helper_skipped = helper_action_executed == ACTION_SKIP_HELPER
    informativeness_alignment = (
        1.0
        if (pregate_should_skip and helper_skipped)
        or (not pregate_should_skip and not helper_skipped)
        else 0.0
    )

    online_kwargs["aux_pytest_fail_reduction"] = pytest_fail_reduction
    online_kwargs["aux_evidence_novelty_ratio"] = evidence_novelty
    online_kwargs["aux_compaction_ratio"] = compaction_ratio
    online_kwargs["aux_informativeness_alignment"] = informativeness_alignment

    budget_kwargs["aux_pytest_fail_reduction"] = pytest_fail_reduction
    budget_kwargs["aux_evidence_novelty_ratio"] = evidence_novelty


def _call_run_pytest_compat(
    cwd,
    python_args,
    condition,
    timeout_sec,
    output_format,
    *,
    progress_update=None,
    progress_phase_label: str = "",
):
    try:
        return run_pytest(
            cwd,
            python_args,
            condition,
            timeout_sec,
            output_format,
            progress_update=progress_update,
            progress_phase_label=progress_phase_label,
        )
    except TypeError as exc:
        message = str(exc)
        if "unexpected keyword argument" not in message or (
            "progress_update" not in message and "progress_phase_label" not in message
        ):
            raise
    return run_pytest(cwd, python_args, condition, timeout_sec, output_format)


def _normalize_prompt_order_mode(value: str) -> str:
    normalized = str(value or "current").strip().lower()
    return normalized if normalized in PROMPT_ORDER_MODES else "current"


def _normalize_adaptive_reveal_mode(value: str) -> str:
    normalized = str(value or "full").strip().lower()
    return normalized if normalized in ADAPTIVE_REVEAL_MODES else "full"


def _normalize_snapshot_frame_pruning_mode(value: str) -> str:
    normalized = str(value or "off").strip().lower()
    return normalized if normalized in SNAPSHOT_FRAME_PRUNING_MODES else "off"


def _baseline_hybrid_guidance_mode_for_patcher(patcher: Any) -> str:
    config = getattr(patcher, "config", None)
    extra = getattr(config, "extra", None)
    if not isinstance(extra, dict):
        return ""
    return str(extra.get("baseline_hybrid_guidance_mode") or "").strip()


def _controller_runtime_evidence_inputs(
    *,
    baseline_hybrid_guidance_mode: str,
    seen_evidence_ids: set[str],
    workdir: Path,
    snapshot: dict[str, Any] | None,
    stdout: str,
    stderr: str,
    max_total_chars: int,
    file_index_lines: list[str],
    combined_evidence_items: list[dict[str, Any]],
    new_evidence_ids: list[str],
    reused_evidence_ids: list[str],
) -> tuple[list[dict[str, Any]], list[str], list[str]]:
    if baseline_hybrid_guidance_mode != HYBRID_GUIDANCE_MODE_FIXED_V1:
        return (
            list(combined_evidence_items),
            list(new_evidence_ids),
            list(reused_evidence_ids),
        )
    fixed_bundle = build_fallback_context_bundle(
        root=workdir,
        snapshot=snapshot,
        stdout=stdout,
        stderr=stderr,
        max_total_chars=max_total_chars,
        reason="fixed_evidence_v1_bundle",
        file_index_lines=file_index_lines,
    )
    fixed_items = list(fixed_bundle.evidence_items)
    fixed_ids = _evidence_ids_from_items(fixed_items)
    fixed_new_ids = [
        evidence_id for evidence_id in fixed_ids if evidence_id not in seen_evidence_ids
    ]
    fixed_reused_ids = [
        evidence_id for evidence_id in fixed_ids if evidence_id in seen_evidence_ids
    ]
    return fixed_items, fixed_new_ids, fixed_reused_ids


def _token_estimate_model_name(patcher: Any) -> str:
    """Resolve the best available model identifier for optional token counting."""
    if patcher is None:
        return ""
    config = getattr(patcher, "config", None)
    for candidate in (
        getattr(patcher, "model", None),
        getattr(config, "model", None),
        getattr(config, "model_name", None),
    ):
        value = str(candidate or "").strip()
        if value:
            return value
    return ""


def _openai_preflight_prompt_cap_tokens(
    *, context_window_tokens: int, max_tokens: int
) -> int | None:
    """Return the preflight prompt token cap used by the char-cap fallback."""
    normalized_window = max(0, int(context_window_tokens))
    if normalized_window <= 0:
        return None
    usable_tokens = (
        normalized_window
        - max(0, int(max_tokens))
        - max(0, int(OPENAI_CONTEXT_WINDOW_RESERVE_TOKENS))
    )
    if usable_tokens <= 0:
        return 1
    return usable_tokens


def _resolve_stage_b_reveal_policy(
    *,
    mode: str,
    is_adaptive_condition: bool,
    snapshot_available: bool,
    stage_a_context: str,
    stage_a_need_snapshot: bool,
    failure_signature_repeat: bool,
    no_progress_streak: int,
) -> dict[str, Any]:
    """Decide whether Stage-B should stay traceback-first or reveal more evidence."""
    normalized_mode = _normalize_adaptive_reveal_mode(mode)
    stage_a_context_present = bool(str(stage_a_context or "").strip())
    if normalized_mode != "traceback_first" or not is_adaptive_condition:
        triggers = ["full_mode"]
        return {
            "mode": normalized_mode,
            "applied": False,
            "include_snapshot": bool(snapshot_available),
            "include_stage_a_context": stage_a_context_present,
            "snapshot_lite_needed": False,
            "trigger": triggers[0],
            "triggers": triggers,
        }

    reveal_triggers: list[str] = []
    if stage_a_context_present:
        reveal_triggers.append("stage_a_context")
    if stage_a_need_snapshot and snapshot_available:
        reveal_triggers.append("need_snapshot")
    if failure_signature_repeat:
        reveal_triggers.append("failure_signature_repeat")
    if no_progress_streak > 0:
        reveal_triggers.append("no_progress_escalation")

    include_snapshot = bool(
        snapshot_available
        and any(
            trigger in {"need_snapshot", "failure_signature_repeat", "no_progress_escalation"}
            for trigger in reveal_triggers
        )
    )
    include_stage_a_context = bool(
        stage_a_context_present
        and any(
            trigger in {"stage_a_context", "failure_signature_repeat", "no_progress_escalation"}
            for trigger in reveal_triggers
        )
    )
    effective_triggers = reveal_triggers or ["traceback_only"]
    return {
        "mode": normalized_mode,
        "applied": True,
        "include_snapshot": include_snapshot,
        "include_stage_a_context": include_stage_a_context,
        "snapshot_lite_needed": bool(snapshot_available and not include_snapshot),
        "trigger": "+".join(effective_triggers),
        "triggers": effective_triggers,
    }


def execute_eval_unit_impl(
    *,
    case: CaseSpec,
    condition: Condition,
    artifacts_root: Path,
    inputs: EvalUnitInputs,
) -> UnitExecutionOutcome:
    """Execute a single (case, condition) eval unit with *k* attempts.

    Copies the case directory into a workdir, runs baseline pytest,
    manages the propose-verify loop, handles adaptive gating/budget,
    and streams JSONL records.  Called from ``main`` via thread-pool.
    """
    run_inputs = inputs.run_condition
    patcher_args = inputs.patcher_args
    patcher_override = inputs.patcher_override
    keep_workdirs = bool(inputs.keep_workdirs)
    progress_verbosity = inputs.progress_verbosity
    progress_heartbeat_sec = float(inputs.progress_heartbeat_sec)
    unit_progress_callback = inputs.unit_progress_callback
    k = int(run_inputs.k)
    unit_record_sink = run_inputs.record_sink

    unit_started = time.monotonic()
    normalized_progress_verbosity = _normalized_progress_verbosity(progress_verbosity)
    normalized_progress_heartbeat_sec = max(0.0, float(progress_heartbeat_sec))
    workdir = artifacts_root / "workdirs" / case.case_id / condition.value
    context_dir_base = artifacts_root / "contexts" / case.case_id / condition.value

    runtime_error: str | None = None
    case_records: list[dict[str, Any]] | None = None
    observed_records_count = 0
    persisted_records_count = 0
    progress_state_lock = threading.Lock()
    progress_state: dict[str, Any] = {
        "phase": "initializing",
        "phase_started_monotonic": unit_started,
        "attempt": 0,
        "max_attempts": int(k),
        "syntax_cycle": 0,
        "syntax_max_cycles": 0,
        "last_note": "",
        "waiting_for": "",
        "external_operation": "",
        "external_provider": "",
        "owner": "",
        "source": "",
        "activity_excerpt": "",
        "attempts_completed": 0,
        "llm_calls_observed": 0,
        "eval_duration_sec_observed": 0.0,
        "observed_records_count": 0,
    }
    heartbeat_stop_event = threading.Event()
    heartbeat_thread: threading.Thread | None = None

    def _coerce_nonnegative_float(raw: Any) -> float:
        try:
            return max(0.0, float(raw))
        except (TypeError, ValueError):
            return 0.0

    def _snapshot_progress_state() -> dict[str, Any]:
        with progress_state_lock:
            return dict(progress_state)

    def _forward_unit_progress(event: str, payload: dict[str, Any] | None = None) -> None:
        if unit_progress_callback is None:
            return
        try:
            forwarded_payload = dict(payload) if isinstance(payload, dict) else None
            unit_progress_callback(case.case_id, condition.value, event, forwarded_payload)
        except Exception:
            pass

    def _update_progress_state(event: str, payload: dict[str, Any] | None = None) -> None:
        now = time.monotonic()
        with progress_state_lock:
            phase_label = ""
            if isinstance(payload, dict):
                phase_label = str(payload.get("phase_label") or "").strip()
            phase_value = phase_label or str(event or "running")
            phase_changed = str(progress_state.get("phase") or "") != phase_value
            progress_state["phase"] = phase_value
            if phase_changed:
                progress_state["phase_started_monotonic"] = now
                for state_key in (
                    "waiting_for",
                    "external_operation",
                    "external_provider",
                    "owner",
                    "source",
                    "activity_excerpt",
                ):
                    progress_state[state_key] = ""
            if isinstance(payload, dict):
                for key in ("attempt", "max_attempts", "syntax_cycle", "syntax_max_cycles"):
                    value = payload.get(key)
                    if isinstance(value, int):
                        progress_state[key] = int(value)
                for key in ("attempts_completed", "llm_calls_observed"):
                    value = payload.get(key)
                    if isinstance(value, int):
                        progress_state[key] = max(0, int(value))
                eval_duration_sec_observed = payload.get("eval_duration_sec_observed")
                if isinstance(eval_duration_sec_observed, (int, float)):
                    progress_state["eval_duration_sec_observed"] = max(
                        0.0, float(eval_duration_sec_observed)
                    )
                note_value = payload.get("note")
                if note_value is not None:
                    progress_state["last_note"] = str(note_value)
                for payload_key, state_key in (
                    ("waiting_for", "waiting_for"),
                    ("external_operation", "external_operation"),
                    ("external_provider", "external_provider"),
                    ("owner", "owner"),
                    ("source", "source"),
                    ("activity_excerpt", "activity_excerpt"),
                ):
                    if payload_key in payload:
                        progress_state[state_key] = str(payload.get(payload_key) or "")
        _forward_unit_progress(event, payload)

        if _progress_is_enabled(normalized_progress_verbosity, "trace"):
            snapshot = _snapshot_progress_state()
            attempt_value = int(snapshot.get("attempt") or 0)
            max_attempts_value = int(snapshot.get("max_attempts") or 0)
            attempt_fragment = (
                f" attempt={attempt_value}/{max_attempts_value}"
                if attempt_value > 0 and max_attempts_value > 0
                else (f" attempt={attempt_value}" if attempt_value > 0 else "")
            )
            syntax_cycle = int(snapshot.get("syntax_cycle") or 0)
            syntax_max_cycles = int(snapshot.get("syntax_max_cycles") or 0)
            syntax_fragment = (
                f" syntax_cycle={syntax_cycle}/{syntax_max_cycles}"
                if syntax_cycle > 0 and syntax_max_cycles > 0
                else ""
            )
            wait_for = str(snapshot.get("waiting_for") or "").strip()
            wait_fragment = f" waiting_for={wait_for}" if wait_for else ""
            owner = str(snapshot.get("owner") or "").strip()
            source = str(snapshot.get("source") or "").strip()
            owner_fragment = f" owner={owner}" if owner else ""
            source_fragment = f" source={source}" if source else ""
            _progress_print(
                verbosity=normalized_progress_verbosity,
                level="trace",
                message=(
                    f"[TRACE] {case.case_id}/{condition.value} "
                    f"phase={snapshot.get('phase', 'running')}{attempt_fragment}"
                    f"{syntax_fragment}{wait_fragment}{owner_fragment}{source_fragment}"
                ),
            )

    def _emit_progress_heartbeat() -> None:
        snapshot = _snapshot_progress_state()
        now = time.monotonic()
        unit_elapsed_sec = max(0.0, now - unit_started)
        unit_elapsed = _format_duration_precise(unit_elapsed_sec)
        phase_started = _coerce_nonnegative_float(
            snapshot.get("phase_started_monotonic", unit_started)
        )
        if phase_started <= 0.0:
            phase_started = unit_started
        phase_elapsed_sec = max(0.0, now - phase_started)
        phase_elapsed = _format_duration_precise(phase_elapsed_sec)
        attempt_value = int(snapshot.get("attempt") or 0)
        max_attempts_value = int(snapshot.get("max_attempts") or 0)
        attempt_fragment = (
            f" attempt={attempt_value}/{max_attempts_value}"
            if attempt_value > 0 and max_attempts_value > 0
            else (f" attempt={attempt_value}" if attempt_value > 0 else "")
        )
        note_text = str(snapshot.get("last_note") or "").strip()
        note_fragment = f" note={note_text}" if note_text else ""
        waiting_for = str(snapshot.get("waiting_for") or "").strip()
        external_operation = str(snapshot.get("external_operation") or "").strip()
        external_provider = str(snapshot.get("external_provider") or "").strip()
        owner = str(snapshot.get("owner") or "").strip()
        source = str(snapshot.get("source") or "").strip()
        activity_excerpt = str(snapshot.get("activity_excerpt") or "").strip()
        attempts_completed = int(snapshot.get("attempts_completed") or 0)
        llm_calls_observed = int(snapshot.get("llm_calls_observed") or 0)
        eval_duration_sec_observed = _coerce_nonnegative_float(
            snapshot.get("eval_duration_sec_observed")
        )
        wait_fragment = f" wait={waiting_for}" if waiting_for else ""
        owner_fragment = f" owner={owner}" if owner else ""
        source_fragment = f" source={source}" if source else ""
        provider_fragment = f" provider={external_provider}" if external_provider else ""
        operation_fragment = f" op={external_operation}" if external_operation else ""
        attempts_completed_fragment = (
            f" iters={attempts_completed}" if attempts_completed > 0 else ""
        )
        llm_calls_fragment = f" llm_calls={llm_calls_observed}" if llm_calls_observed > 0 else ""
        eval_fragment = (
            f" eval={eval_duration_sec_observed:.1f}s" if eval_duration_sec_observed > 0.0 else ""
        )
        activity_fragment = f" activity={activity_excerpt}" if activity_excerpt else ""
        records_observed = int(snapshot.get("observed_records_count") or 0)
        _forward_unit_progress(
            "heartbeat",
            {
                "attempt": attempt_value,
                "max_attempts": max_attempts_value,
                "note": note_text,
                "phase": str(snapshot.get("phase", "running")),
                "waiting_for": waiting_for,
                "external_operation": external_operation,
                "external_provider": external_provider,
                "owner": owner,
                "source": source,
                "activity_excerpt": activity_excerpt,
                "attempts_completed": attempts_completed,
                "llm_calls_observed": llm_calls_observed,
                "eval_duration_sec_observed": eval_duration_sec_observed,
                "records_observed": records_observed,
                "unit_elapsed_sec": unit_elapsed_sec,
                "phase_elapsed_sec": phase_elapsed_sec,
            },
        )
        _progress_print(
            verbosity=normalized_progress_verbosity,
            level="detailed",
            message=(
                f"[HEARTBEAT] {case.case_id}/{condition.value}{attempt_fragment} "
                f"phase={snapshot.get('phase', 'running')} "
                f"records={records_observed} "
                f"unit={unit_elapsed} phase_elapsed={phase_elapsed}"
                f"{owner_fragment}{source_fragment}{wait_fragment}{provider_fragment}"
                f"{operation_fragment}{attempts_completed_fragment}{llm_calls_fragment}"
                f"{eval_fragment}{note_fragment}{activity_fragment}"
            ),
        )

    def _on_progress_update(event: str, payload: dict[str, Any] | None = None) -> None:
        _update_progress_state(event, payload)

    def _on_record(record: dict[str, Any]) -> None:
        nonlocal observed_records_count
        nonlocal persisted_records_count
        with progress_state_lock:
            observed_records_count += 1
            progress_state["observed_records_count"] = int(observed_records_count)
        if unit_record_sink is not None:
            unit_record_sink(record)
            persisted_records_count += 1
        note = str(record.get("note") or "")
        attempt = int(record.get("attempt") or 0)
        _update_progress_state(
            "record_emitted",
            {
                "attempt": attempt,
                "max_attempts": int(k),
                "note": note,
            },
        )
        if _progress_is_enabled(normalized_progress_verbosity, "detailed"):
            patch_meta = record.get("patch_meta")
            attempt_meta = record.get("attempt_metadata")
            propose_sec = (
                _coerce_nonnegative_float(patch_meta.get("propose_duration_sec"))
                if isinstance(patch_meta, dict)
                else 0.0
            )
            prompt_chars_total = (
                _nonnegative_int(
                    attempt_meta.get("prompt_chars_total"),
                    default=_nonnegative_int(patch_meta.get("prompt_chars_total"), default=0)
                    if isinstance(patch_meta, dict)
                    else 0,
                )
                if isinstance(attempt_meta, dict)
                else _nonnegative_int(patch_meta.get("prompt_chars_total"), default=0)
                if isinstance(patch_meta, dict)
                else 0
            )
            completion_chars_total = (
                _nonnegative_int(
                    attempt_meta.get("completion_chars_total"),
                    default=_nonnegative_int(patch_meta.get("completion_chars_total"), default=0)
                    if isinstance(patch_meta, dict)
                    else _completion_chars_from_response(
                        patch_meta=patch_meta if isinstance(patch_meta, dict) else None
                    ),
                )
                if isinstance(attempt_meta, dict)
                else _nonnegative_int(patch_meta.get("completion_chars_total"), default=0)
                if isinstance(patch_meta, dict)
                else 0
            )
            if isinstance(attempt_meta, dict):
                prompt_tokens = _nonnegative_int(
                    attempt_meta.get("tokens_prompt"),
                    default=_usage_token_counts_from_patch_meta(patch_meta)[0]
                    if isinstance(patch_meta, dict)
                    else 0,
                )
                completion_tokens = _nonnegative_int(
                    attempt_meta.get("tokens_completion"),
                    default=_usage_token_counts_from_patch_meta(patch_meta)[1]
                    if isinstance(patch_meta, dict)
                    else 0,
                )
                total_tokens = _nonnegative_int(
                    attempt_meta.get("tokens_total"),
                    default=_usage_token_counts_from_patch_meta(patch_meta)[2]
                    if isinstance(patch_meta, dict)
                    else 0,
                )
            else:
                prompt_tokens, completion_tokens, total_tokens = (
                    _usage_token_counts_from_patch_meta(patch_meta)
                )
            test_after_sec = _coerce_nonnegative_float(record.get("after_duration_sec"))
            attempt_wall_sec = _coerce_nonnegative_float(record.get("attempt_wall_clock_sec"))
            status_label = "success" if bool(record.get("success")) else "failed"
            token_usage_state = _record_token_usage_state(
                attempt_meta=attempt_meta if isinstance(attempt_meta, dict) else None,
                patch_meta=patch_meta if isinstance(patch_meta, dict) else None,
                prompt_tokens=prompt_tokens,
                total_tokens=total_tokens,
            )
            token_usage_text = _format_progress_token_usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                available_records=1 if token_usage_state == "available" else 0,
                unavailable_records=1 if token_usage_state == "unavailable" else 0,
            )
            _progress_print(
                verbosity=normalized_progress_verbosity,
                level="detailed",
                message=(
                    f"[ATTEMPT] {case.case_id}/{condition.value} "
                    f"attempt={attempt}/{int(k)} status={status_label} note={note or 'none'} "
                    f"wall={_format_duration_precise(attempt_wall_sec)} "
                    f"llm={_format_duration_precise(propose_sec)} "
                    f"chars=in:{prompt_chars_total}ch,out:{completion_chars_total}ch "
                    f"{token_usage_text} "
                    f"test_after={_format_duration_precise(test_after_sec)}"
                ),
            )

    heartbeat_updates_enabled = unit_progress_callback is not None
    if normalized_progress_heartbeat_sec > 0.0 and (
        heartbeat_updates_enabled or _progress_is_enabled(normalized_progress_verbosity, "detailed")
    ):

        def _heartbeat_loop() -> None:
            while not heartbeat_stop_event.wait(normalized_progress_heartbeat_sec):
                _emit_progress_heartbeat()

        heartbeat_thread = threading.Thread(
            target=_heartbeat_loop,
            name=f"eval-heartbeat-{case.case_id}-{condition.value}",
            daemon=True,
        )
        heartbeat_thread.start()

    try:
        _update_progress_state("setup_workdir")
        workdir.parent.mkdir(parents=True, exist_ok=True)
        context_dir_base.mkdir(parents=True, exist_ok=True)

        if workdir.exists():
            shutil.rmtree(workdir)
        shutil.copytree(
            case.path,
            workdir,
            dirs_exist_ok=True,
            ignore=shutil.ignore_patterns("__pycache__", "*.pyc", "*.pyo"),
        )

        patcher = patcher_override if patcher_override is not None else make_patcher(patcher_args)
        _update_progress_state("run_one_condition")
        if hasattr(patcher, "set_progress_callback"):
            patcher.set_progress_callback(_on_progress_update)
        try:
            case_records = run_one_condition(
                case=case,
                condition=condition,
                workdir=workdir,
                context_dir_base=context_dir_base,
                **{
                    **run_inputs.__dict__,
                    "patcher": patcher,
                    "record_sink": _on_record,
                    "progress_update": _on_progress_update,
                },
            )
        finally:
            if hasattr(patcher, "set_progress_callback"):
                patcher.set_progress_callback(None)
        _update_progress_state(
            "unit_completed",
            {
                "note": "completed",
            },
        )
    except RuntimeError as exc:
        runtime_error = str(exc)
    except Exception as exc:
        runtime_error = f"{type(exc).__name__}: {exc}"
    finally:
        heartbeat_stop_event.set()
        if heartbeat_thread is not None:
            heartbeat_thread.join(timeout=0.25)
        if not keep_workdirs:
            shutil.rmtree(workdir, ignore_errors=True)

    return UnitExecutionOutcome(
        case=case,
        condition=condition,
        case_records=case_records,
        unit_duration_sec=time.monotonic() - unit_started,
        runtime_error=runtime_error,
        records_streamed=persisted_records_count > 0,
    )


def run_one_condition_impl(
    *,
    case: CaseSpec,
    condition: Condition,
    workdir: Path,
    context_dir_base: Path,
    inputs: RunConditionInputs | None = None,
    **legacy_kwargs: Any,
) -> list[dict[str, Any]]:
    """Execute up to *k* patch-verify attempts for a single (case, condition).

    Manages the full attempt loop: pytest baseline, context assembly,
    prompt rendering, patch proposal, diff application, re-verification,
    and adaptive policy updates.  Returns a list of JSONL-ready records.
    """
    resolved_inputs = _resolve_run_condition_inputs(inputs=inputs, legacy_kwargs=legacy_kwargs)
    patcher = resolved_inputs.patcher
    k = int(resolved_inputs.k)
    output_format = resolved_inputs.output_format
    run_id = resolved_inputs.run_id
    include_files_context = resolved_inputs.include_files_context
    max_context_chars = int(resolved_inputs.max_context_chars)
    case_fail_fast_on_request_budget_exceeded = bool(
        resolved_inputs.case_fail_fast_on_request_budget_exceeded
    )
    force_patch_attempt = bool(resolved_inputs.force_patch_attempt)
    native_baseline_mode = str(getattr(resolved_inputs, "native_baseline_mode", "") or "")
    context_protocol = resolved_inputs.context_protocol
    context_stage_a_max_requests = int(resolved_inputs.context_stage_a_max_requests)
    context_stage_a_max_chars = int(resolved_inputs.context_stage_a_max_chars)
    retry_prompt_mode = resolved_inputs.retry_prompt_mode
    retry_context_mode = resolved_inputs.retry_context_mode
    retry_history_max_chars = int(resolved_inputs.retry_history_max_chars)
    prompt_retry_reflection = bool(resolved_inputs.prompt_retry_reflection)
    prompt_adaptive_stage_b_cot = bool(resolved_inputs.prompt_adaptive_stage_b_cot)
    prompt_repair_method_scaffold = bool(resolved_inputs.prompt_repair_method_scaffold)
    prompt_retry_plan_then_patch = bool(resolved_inputs.prompt_retry_plan_then_patch)
    prompt_order_mode = resolved_inputs.prompt_order_mode
    adaptive_reveal_mode = resolved_inputs.adaptive_reveal_mode
    snapshot_frame_pruning_mode = resolved_inputs.snapshot_frame_pruning_mode
    pytest_output_tail_lines = int(resolved_inputs.pytest_output_tail_lines)
    snapshot_artifact_max_string_chars = int(resolved_inputs.snapshot_artifact_max_string_chars)
    prompt_budget_total_chars = int(resolved_inputs.prompt_budget_total_chars)
    prompt_budget_retry_delta_chars = int(resolved_inputs.prompt_budget_retry_delta_chars)
    prompt_budget_stdout_chars = int(resolved_inputs.prompt_budget_stdout_chars)
    prompt_budget_snapshot_chars = int(resolved_inputs.prompt_budget_snapshot_chars)
    prompt_budget_evidence_chars = int(resolved_inputs.prompt_budget_evidence_chars)
    prompt_budget_retry_history_chars = int(resolved_inputs.prompt_budget_retry_history_chars)
    context_compaction_mode = resolved_inputs.context_compaction_mode
    context_compaction_scope = resolved_inputs.context_compaction_scope
    context_compaction_max_input_chars = int(resolved_inputs.context_compaction_max_input_chars)
    context_compaction_target_chars = int(resolved_inputs.context_compaction_target_chars)
    context_compaction_max_calls = int(resolved_inputs.context_compaction_max_calls)
    openai_context_window_tokens = int(resolved_inputs.openai_context_window_tokens)
    openai_context_window_source = resolved_inputs.openai_context_window_source
    openai_context_window_probe_provider = resolved_inputs.openai_context_window_probe_provider
    openai_context_window_probe_error = resolved_inputs.openai_context_window_probe_error
    openai_response_mode_autoswitch = resolved_inputs.openai_response_mode_autoswitch
    adaptive_gate_policy = resolved_inputs.adaptive_gate_policy
    adaptive_max_helper_calls = int(resolved_inputs.adaptive_max_helper_calls)
    adaptive_budget_topup_max_calls = int(resolved_inputs.adaptive_budget_topup_max_calls)
    adaptive_stagnation_window = int(resolved_inputs.adaptive_stagnation_window)
    adaptive_saturation_threshold = float(resolved_inputs.adaptive_saturation_threshold)
    adaptive_no_progress_token_cap = int(resolved_inputs.adaptive_no_progress_token_cap)
    adaptive_helper_budget_profile = resolved_inputs.adaptive_helper_budget_profile
    adaptive_openai_helper_response_mode = resolved_inputs.adaptive_openai_helper_response_mode
    adaptive_controller_lambda_rlm = bool(resolved_inputs.adaptive_controller_lambda_rlm)
    adaptive_controller_lambda_rlm_repo_dir = (
        resolved_inputs.adaptive_controller_lambda_rlm_repo_dir
    )
    adaptive_controller_lambda_rlm_python = resolved_inputs.adaptive_controller_lambda_rlm_python
    adaptive_controller_lambda_rlm_context_window_chars = int(
        resolved_inputs.adaptive_controller_lambda_rlm_context_window_chars
    )
    adaptive_controller_lambda_rlm_timeout_sec = int(
        resolved_inputs.adaptive_controller_lambda_rlm_timeout_sec
    )
    adaptive_informativeness_pregate = bool(resolved_inputs.adaptive_informativeness_pregate)
    adaptive_informativeness_pregate_threshold = float(
        resolved_inputs.adaptive_informativeness_pregate_threshold
    )
    syntax_preflight_enabled = bool(resolved_inputs.syntax_preflight_enabled)
    syntax_preflight_max_cycles = int(resolved_inputs.syntax_preflight_max_cycles)
    syntax_preflight_fallthrough = bool(resolved_inputs.syntax_preflight_fallthrough)
    patch_sanitize_enabled = bool(resolved_inputs.patch_sanitize_enabled)
    patch_verifier_scope = resolved_inputs.patch_verifier_scope
    patch_verifier_min_score = int(resolved_inputs.patch_verifier_min_score)
    patch_verifier_infer_test_source_scope = bool(
        resolved_inputs.patch_verifier_infer_test_source_scope
    )
    adaptive_online_policy_controller = resolved_inputs.adaptive_online_policy_controller
    adaptive_online_policy_mode = resolved_inputs.adaptive_online_policy_mode
    adaptive_online_policy_algo = resolved_inputs.adaptive_online_policy_algo
    adaptive_online_feature_scaling_profile = (
        resolved_inputs.adaptive_online_feature_scaling_profile
    )
    adaptive_online_action_score_biases = resolved_inputs.adaptive_online_action_score_biases
    adaptive_budget_policy_controller = resolved_inputs.adaptive_budget_policy_controller
    adaptive_budget_policy_mode = resolved_inputs.adaptive_budget_policy_mode
    adaptive_budget_policy_algo = resolved_inputs.adaptive_budget_policy_algo
    adaptive_budget_policy_architecture = resolved_inputs.adaptive_budget_policy_architecture
    adaptive_budget_bwk_run_token_budget = int(resolved_inputs.adaptive_budget_bwk_run_token_budget)
    adaptive_budget_feature_scaling_profile = (
        resolved_inputs.adaptive_budget_feature_scaling_profile
    )
    adaptive_budget_policy_tiers = resolved_inputs.adaptive_budget_policy_tiers
    adaptive_budget_policy_max_ups_per_case = int(
        resolved_inputs.adaptive_budget_policy_max_ups_per_case
    )
    adaptive_budget_reward_rho_budget_exceeded = float(
        resolved_inputs.adaptive_budget_reward_rho_budget_exceeded
    )
    adaptive_reward_lambda_tokens = float(resolved_inputs.adaptive_reward_lambda_tokens)
    adaptive_reward_mu_latency = float(resolved_inputs.adaptive_reward_mu_latency)
    adaptive_reward_token_scale = float(resolved_inputs.adaptive_reward_token_scale)
    adaptive_reward_latency_scale = float(resolved_inputs.adaptive_reward_latency_scale)
    adaptive_reward_eta_progress = float(resolved_inputs.adaptive_reward_eta_progress)
    adaptive_reward_kappa_stagnation = float(resolved_inputs.adaptive_reward_kappa_stagnation)
    adaptive_reward_success_hint = float(resolved_inputs.adaptive_reward_success_hint)
    adaptive_reward_progress_scale = float(resolved_inputs.adaptive_reward_progress_scale)
    adaptive_reward_stagnation_scale = float(resolved_inputs.adaptive_reward_stagnation_scale)
    adaptive_reward_materialization_scale = float(
        resolved_inputs.adaptive_reward_materialization_scale
    )
    adaptive_toon_encoding = bool(resolved_inputs.adaptive_toon_encoding)
    adaptive_tep_mode = str(getattr(resolved_inputs, "adaptive_tep_mode", "off") or "off")
    adaptive_tep_state_path = str(getattr(resolved_inputs, "adaptive_tep_state_path", "") or "")
    adaptive_tep_init_state_path = str(
        getattr(resolved_inputs, "adaptive_tep_init_state_path", "") or ""
    )
    adaptive_tep_beta_init = float(getattr(resolved_inputs, "adaptive_tep_beta_init", 1.0) or 1.0)
    adaptive_tep_beta_decay = float(getattr(resolved_inputs, "adaptive_tep_beta_decay", 0.9) or 0.9)
    adaptive_tep_equilibrium_max_iters = int(
        getattr(resolved_inputs, "adaptive_tep_equilibrium_max_iters", 20) or 20
    )
    adaptive_tep_equilibrium_window = int(
        getattr(resolved_inputs, "adaptive_tep_equilibrium_window", 3) or 3
    )
    adaptive_tep_equilibrium_variance_threshold = float(
        getattr(resolved_inputs, "adaptive_tep_equilibrium_variance_threshold", 0.5) or 0.5
    )
    adaptive_tep_validation_buffer_size = int(
        getattr(resolved_inputs, "adaptive_tep_validation_buffer_size", 4) or 4
    )
    enable_tep = bool(resolved_inputs.enable_tep)
    tep_nudge_escalation = bool(resolved_inputs.tep_nudge_escalation)
    tep_equilibrium_check = bool(resolved_inputs.tep_equilibrium_check)
    tep_equilibrium_max_iterations = int(resolved_inputs.tep_equilibrium_max_iterations)
    tep_equilibrium_similarity_threshold = float(
        resolved_inputs.tep_equilibrium_similarity_threshold
    )
    enable_auxiliary_rewards = bool(resolved_inputs.enable_auxiliary_rewards)
    aux_pytest_reduction_weight = float(resolved_inputs.adaptive_aux_pytest_reduction_weight)
    aux_evidence_novelty_weight = float(resolved_inputs.adaptive_aux_evidence_novelty_weight)
    aux_compaction_efficiency_weight = float(
        resolved_inputs.adaptive_aux_compaction_efficiency_weight
    )
    aux_informativeness_alignment_weight = float(
        resolved_inputs.adaptive_aux_informativeness_alignment_weight
    )
    run_metadata = resolved_inputs.run_metadata
    record_prompt = resolved_inputs.record_prompt
    record_evidence = resolved_inputs.record_evidence
    deterministic = bool(resolved_inputs.deterministic)
    seed = resolved_inputs.seed
    transport_recovery_gate = resolved_inputs.transport_recovery_gate
    openai_transport_recovery = bool(resolved_inputs.openai_transport_recovery)
    record_sink = resolved_inputs.record_sink
    progress_update = resolved_inputs.progress_update

    records: list[dict[str, Any]] = []
    deferred_record_sink_queue: list[dict[str, Any]] = []
    deferred_record_sink_pending_ids: set[int] = set()
    scored_attempts_consumed = 0

    def _emit_progress(event: str, payload: dict[str, Any] | None = None) -> None:
        if progress_update is None:
            return
        try:
            progress_update(event, payload)
        except Exception:
            # Progress callbacks are best-effort and must not break eval execution.
            return

    def _progress_payload(
        *,
        owner: str,
        source: str = "phase",
        phase_label: str = "",
        note: str = "",
        **extra: Any,
    ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "owner": str(owner or ""),
            "source": str(source or ""),
        }
        if phase_label:
            payload["phase_label"] = phase_label
        if note:
            payload["note"] = note
        payload.update(extra)
        return payload

    def _flush_deferred_record_sink_queue() -> None:
        if record_sink is None:
            return
        while deferred_record_sink_queue:
            front_record = deferred_record_sink_queue[0]
            if id(front_record) in deferred_record_sink_pending_ids:
                break
            record_sink(front_record)
            deferred_record_sink_queue.pop(0)

    def _append_record(record: dict[str, Any], *, defer_sink: bool = False) -> None:
        _finalize_breakdowns(record)
        records.append(record)
        _emit_progress(
            "record_emitted",
            {
                "attempt": int(record.get("attempt") or 0),
                "note": str(record.get("note") or ""),
                "success": bool(record.get("success")),
            },
        )
        if record_sink is not None:
            if defer_sink or deferred_record_sink_queue:
                deferred_record_sink_queue.append(record)
                if defer_sink:
                    deferred_record_sink_pending_ids.add(id(record))
                _flush_deferred_record_sink_queue()
            else:
                record_sink(record)

    def _mark_scored_attempt_consumed() -> None:
        nonlocal scored_attempts_consumed
        scored_attempts_consumed += 1

    def _append_infra_event(
        *,
        attempt: int,
        patcher_for_attempt: Any,
        patch_meta: dict[str, Any],
        attempt_metadata: dict[str, Any] | None,
        attempt_context_protocol: str,
        before: RunResult,
        cumulative_runtime_sec_value: float,
    ) -> None:
        infra_record = _build_infra_event_record(
            case=case,
            condition=condition,
            run_id=run_id,
            attempt=attempt,
            patcher_name=patcher_name,
            run_metadata=effective_run_metadata,
            attempt_metadata=attempt_metadata,
            patcher=patcher_for_attempt,
            patch_meta=patch_meta,
            context_protocol=attempt_context_protocol,
            before=before,
            cumulative_runtime_sec=cumulative_runtime_sec_value,
        )
        _append_record(infra_record)

    def _run_openai_operation_with_transport_recovery(
        *,
        patcher_for_attempt: Any,
        operation_name: str,
        invoke: Callable[[], Any],
        classify_failure_kind: Callable[[dict[str, Any]], str],
        attempt: int,
        attempt_metadata: dict[str, Any] | None,
        attempt_context_protocol: str,
        before: RunResult,
        cumulative_runtime_sec_value: float,
    ) -> Any:
        patcher_name_value = str(getattr(patcher_for_attempt, "name", "") or "").strip()
        is_openai_patcher_for_attempt = patcher_name_value == "openai"
        supports_transport_probe = callable(
            getattr(patcher_for_attempt, "probe_server_ready", None)
        )
        if is_openai_patcher_for_attempt:
            provider = "openai_server"
            model = str(getattr(patcher_for_attempt, "model", "") or "").strip()
            operation_label = operation_name.replace("_", " ").strip() or "request"
            note = f"{operation_label}" + (f" [{model}]" if model else "")
            _emit_progress(
                "llm_request",
                _progress_payload(
                    owner="openai",
                    attempt=int(attempt),
                    max_attempts=int(k),
                    phase_label=f"llm: {operation_label}",
                    waiting_for=provider,
                    external_operation=operation_name,
                    external_provider=provider,
                    note=note,
                ),
            )
        if (
            not bool(openai_transport_recovery)
            or transport_recovery_gate is None
            or not supports_transport_probe
        ):
            return invoke()

        transport_recovery_gate.wait_if_open()
        result = invoke()
        result_meta = _result_meta_dict(result)
        failure_kind = str(classify_failure_kind(result_meta) or "").strip()
        if failure_kind not in {"transport", "backend_unavailable"}:
            return result

        recovery_outcome = transport_recovery_gate.recover(
            patcher=patcher_for_attempt,
            operation_name=operation_name,
        )
        infra_patch_meta = dict(result_meta)
        infra_patch_meta["transport_recovery_operation"] = operation_name
        infra_patch_meta["transport_recovery_failure_kind"] = failure_kind
        infra_patch_meta["transport_recovery_attempted"] = bool(recovery_outcome.get("attempted"))
        infra_patch_meta["transport_recovery_recovered"] = bool(recovery_outcome.get("recovered"))
        infra_patch_meta["transport_recovery_timed_out"] = bool(recovery_outcome.get("timed_out"))
        infra_patch_meta["transport_recovery_wait_sec"] = float(
            recovery_outcome.get("wait_sec") or 0.0
        )
        infra_patch_meta["transport_recovery_probe_attempts"] = list(
            recovery_outcome.get("probe_attempts") or []
        )
        if (
            patcher_name_value != "openai"
            and str(result_meta.get("error") or "") == "repair_failed"
        ):
            infra_patch_meta["infra_failure_kind"] = failure_kind
            infra_patch_meta["infra_resume_eligible"] = True
            infra_patch_meta["infra_source"] = "baseline_repair"
        probe_meta = recovery_outcome.get("probe_meta")
        if isinstance(probe_meta, dict) and probe_meta:
            infra_patch_meta["transport_recovery_probe_meta"] = dict(probe_meta)
        _append_infra_event(
            attempt=attempt,
            patcher_for_attempt=patcher_for_attempt,
            patch_meta=infra_patch_meta,
            attempt_metadata=attempt_metadata,
            attempt_context_protocol=attempt_context_protocol,
            before=before,
            cumulative_runtime_sec_value=cumulative_runtime_sec_value,
        )

        if not bool(recovery_outcome.get("recovered")):
            failed_meta = dict(result_meta)
            failed_meta.update(
                {
                    "transport_recovery_operation": operation_name,
                    "transport_recovery_attempted": True,
                    "transport_recovery_recovered": False,
                    "transport_recovery_timed_out": bool(recovery_outcome.get("timed_out")),
                    "transport_recovery_wait_sec": float(recovery_outcome.get("wait_sec") or 0.0),
                    "transport_recovery_probe_attempts": list(
                        recovery_outcome.get("probe_attempts") or []
                    ),
                }
            )
            if (
                patcher_name_value != "openai"
                and str(result_meta.get("error") or "") == "repair_failed"
            ):
                failed_meta["infra_failure_kind"] = failure_kind
                failed_meta["infra_resume_eligible"] = True
                failed_meta["infra_source"] = "baseline_repair"
            probe_meta = recovery_outcome.get("probe_meta")
            if isinstance(probe_meta, dict) and probe_meta:
                failed_meta["transport_recovery_probe_meta"] = dict(probe_meta)
            return _result_with_meta(result, failed_meta)

        transport_recovery_gate.wait_if_open()
        replay_result = invoke()
        replay_meta = _result_meta_dict(replay_result)
        replay_meta["transport_recovery_operation"] = operation_name
        replay_meta["transport_recovery_attempted"] = True
        replay_meta["transport_recovery_recovered"] = True
        replay_meta["transport_recovery_replayed"] = True
        replay_meta["transport_recovery_wait_sec"] = float(recovery_outcome.get("wait_sec") or 0.0)
        replay_meta["transport_recovery_probe_attempts"] = list(
            recovery_outcome.get("probe_attempts") or []
        )
        probe_meta = recovery_outcome.get("probe_meta")
        if isinstance(probe_meta, dict) and probe_meta:
            replay_meta["transport_recovery_probe_meta"] = dict(probe_meta)
        replay_failure_kind = str(classify_failure_kind(replay_meta) or "").strip()
        if (
            replay_failure_kind
            and patcher_name_value != "openai"
            and str(replay_meta.get("error") or "") == "repair_failed"
        ):
            replay_meta["infra_failure_kind"] = replay_failure_kind
            replay_meta["infra_resume_eligible"] = True
            replay_meta["infra_source"] = "baseline_repair"
            replay_meta["transport_recovery_replayed_failure_kind"] = replay_failure_kind
        return _result_with_meta(replay_result, replay_meta)

    normalized_context_protocol = (
        context_protocol if context_protocol in CONTEXT_PROTOCOLS else "monolithic"
    )
    is_adaptive_gated = condition == Condition.LAYERED_ADAPTIVE_GATED
    is_adaptive_llm_discretion = condition == Condition.LAYERED_ADAPTIVE_LLM
    is_adaptive_condition = is_adaptive_gated or is_adaptive_llm_discretion
    toon_encoding_requested = bool(adaptive_toon_encoding and is_adaptive_condition)
    pending_online_credits: list[PendingAdaptiveOnlineCredit] = []
    adaptive_history_fallback_session_id = f"{case.case_id}:{condition.value}"

    def _persist_adaptive_history_for_record(
        *,
        record: dict[str, Any],
        patch_meta: dict[str, Any],
        retry_plan_value: str,
        snapshot_for_history: dict[str, Any] | None,
    ) -> None:
        if not is_adaptive_condition:
            return
        session_id = ""
        error = ""
        try:
            session_id = persist_adaptive_attempt_history(
                out_dir=context_dir_base,
                snapshot=snapshot_for_history,
                fallback_session_id=adaptive_history_fallback_session_id,
                record=record,
                patch_meta=patch_meta,
                retry_plan_text=retry_plan_value,
            )
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
        attempt_metadata_raw = record.get("attempt_metadata")
        if isinstance(attempt_metadata_raw, dict):
            if session_id:
                attempt_metadata_raw["persistent_history_session_id"] = session_id
            if error:
                attempt_metadata_raw["persistent_history_error"] = error
        patch_meta_raw = record.get("patch_meta")
        if isinstance(patch_meta_raw, dict):
            if session_id:
                patch_meta_raw["persistent_history_session_id"] = session_id
            if error:
                patch_meta_raw["persistent_history_error"] = error
        if session_id:
            patch_meta["persistent_history_session_id"] = session_id
        if error:
            patch_meta["persistent_history_error"] = error

    def _capture_attempt_history(
        *,
        attempt: int,
        note: str,
        failure_signature: str,
        record: dict[str, Any],
        patch_meta: dict[str, Any],
        retry_plan_value: str,
        snapshot_for_history: dict[str, Any] | None,
    ) -> None:
        _append_retry_history_turn_if_enabled(
            retry_context_mode=effective_retry_context_mode,
            retry_history_turns=retry_history_turns,
            attempt=attempt,
            note=note,
            failure_signature=failure_signature,
            patch_meta=patch_meta,
        )
        _persist_adaptive_history_for_record(
            record=record,
            patch_meta=patch_meta,
            retry_plan_value=retry_plan_value,
            snapshot_for_history=snapshot_for_history,
        )

    def _queue_pending_online_credit(
        *,
        pending_credit: PendingAdaptiveOnlineCredit | None,
        record: dict[str, Any],
    ) -> None:
        if pending_credit is None:
            return
        pending_credit.record_ref = record
        pending_online_credits.append(pending_credit)

    def _resolve_expired_pending_online_credits(*, current_attempt: int) -> None:
        if not pending_online_credits:
            return
        remaining: list[PendingAdaptiveOnlineCredit] = []
        for pending in pending_online_credits:
            if (
                int(current_attempt) - int(pending.attempt)
                > ADAPTIVE_DELAYED_RESCUE_HORIZON_ATTEMPTS
            ):
                _resolve_pending_online_credit_and_flush(
                    pending=pending,
                    delayed_rescue_bonus=0.0,
                    resolved_attempt=0,
                    resolution_reason="local_only_horizon_expired",
                )
            else:
                remaining.append(pending)
        pending_online_credits[:] = remaining

    def _resolve_pending_online_credits_for_success(*, success_attempt: int) -> None:
        if not pending_online_credits:
            return
        credited_index: int | None = None
        for idx in range(len(pending_online_credits) - 1, -1, -1):
            pending = pending_online_credits[idx]
            horizon_attempts = int(success_attempt) - int(pending.attempt)
            if 0 < horizon_attempts <= ADAPTIVE_DELAYED_RESCUE_HORIZON_ATTEMPTS:
                credited_index = idx
                break
        for idx, pending in enumerate(pending_online_credits):
            if idx == credited_index:
                delayed_rescue_bonus = _adaptive_online_delayed_rescue_bonus(
                    origin_attempt=pending.attempt,
                    resolved_attempt=success_attempt,
                    failure_signature_repeat=pending.failure_signature_repeat,
                    case_difficulty=pending.case_difficulty,
                    case_snapshot_dependency=pending.case_snapshot_dependency,
                )
                _resolve_pending_online_credit_and_flush(
                    pending=pending,
                    delayed_rescue_bonus=delayed_rescue_bonus,
                    resolved_attempt=success_attempt,
                    resolution_reason="future_success_attempt",
                )
            else:
                _resolve_pending_online_credit_and_flush(
                    pending=pending,
                    delayed_rescue_bonus=0.0,
                    resolved_attempt=0,
                    resolution_reason=(
                        "local_only_superseded_by_newer_helper"
                        if credited_index is not None
                        else "local_only_case_ended"
                    ),
                )
        pending_online_credits.clear()

    def _resolve_remaining_pending_online_credits() -> None:
        if not pending_online_credits:
            return
        for pending in pending_online_credits:
            _resolve_pending_online_credit_and_flush(
                pending=pending,
                delayed_rescue_bonus=0.0,
                resolved_attempt=0,
                resolution_reason="local_only_case_ended",
            )
        pending_online_credits.clear()

    def _resolve_pending_online_credit_and_flush(
        *,
        pending: PendingAdaptiveOnlineCredit,
        delayed_rescue_bonus: float,
        resolved_attempt: int,
        resolution_reason: str,
    ) -> None:
        pending_record_ref = pending.record_ref
        _resolve_pending_adaptive_online_credit(
            pending=pending,
            adaptive_online_policy_controller=adaptive_online_policy_controller,
            delayed_rescue_bonus=delayed_rescue_bonus,
            resolved_attempt=resolved_attempt,
            resolution_reason=resolution_reason,
        )
        if isinstance(pending_record_ref, dict):
            deferred_record_sink_pending_ids.discard(id(pending_record_ref))
        _flush_deferred_record_sink_queue()

    def _refresh_adaptive_progress_for_note(
        *,
        note: str,
        success: bool,
        signature_repeated: bool,
        after_run: RunResult | None = None,
        recomputed_after_patch: bool = False,
    ) -> tuple[bool, str]:
        nonlocal adaptive_no_progress_streak
        nonlocal adaptive_no_progress_streak_current
        nonlocal adaptive_progress_made
        nonlocal adaptive_progress_reason
        nonlocal adaptive_progress_score
        nonlocal adaptive_progress_stop_candidate
        nonlocal adaptive_progress_stop_reason

        if not is_adaptive_condition:
            return adaptive_progress_stop_candidate, adaptive_progress_stop_reason

        progress_state, adaptive_no_progress_streak_current = (
            _recompute_adaptive_progress_for_attempt_outcome(
                attempt_metadata=attempt_metadata,
                patch_meta=patch_meta,
                note=note,
                success=success,
                signature_repeated=signature_repeated,
                new_evidence_count=len(bookkeeping_new_evidence_ids),
                before_run=current,
                after_run=after_run,
                previous_failure_context=previous_failure_context,
                current_no_progress_streak=adaptive_no_progress_streak_current,
                recomputed_after_patch=recomputed_after_patch,
            )
        )
        adaptive_progress_made = bool(progress_state.made)
        adaptive_progress_reason = str(progress_state.reason)
        adaptive_progress_score = float(progress_state.score)
        adaptive_no_progress_streak = (
            0 if adaptive_progress_made else adaptive_no_progress_streak_current
        )
        (
            adaptive_progress_stop_candidate,
            adaptive_progress_stop_reason,
        ) = _adaptive_should_stop_for_no_progress(
            no_progress_streak=adaptive_no_progress_streak_current,
            stop_window=adaptive_no_progress_stop_window,
            helper_budget_exhausted=helper_budget_exhausted,
            helper_saturation_guarded=helper_saturation_guarded,
            helper_decision_reason=helper_decision_reason,
        )
        return adaptive_progress_stop_candidate, adaptive_progress_stop_reason

    default_context_protocol = (
        "monolithic" if is_adaptive_condition else normalized_context_protocol
    )
    normalized_retry_prompt_mode = (
        retry_prompt_mode if retry_prompt_mode in RETRY_PROMPT_MODES else "auto"
    )
    if normalized_retry_prompt_mode == "auto":
        base_retry_prompt_mode = "full"
    else:
        base_retry_prompt_mode = normalized_retry_prompt_mode
    normalized_retry_context_mode = (
        retry_context_mode if retry_context_mode in RETRY_CONTEXT_MODES else "history"
    )
    effective_retry_context_mode = normalized_retry_context_mode
    base_retry_history_max_chars = max(0, int(retry_history_max_chars))
    effective_retry_history_max_chars = base_retry_history_max_chars
    normalized_pytest_output_tail_lines = max(0, int(pytest_output_tail_lines))
    normalized_snapshot_artifact_max_string_chars = max(0, int(snapshot_artifact_max_string_chars))
    base_stage_a_max_requests = max(1, int(context_stage_a_max_requests))
    base_stage_a_max_chars = max(1, int(context_stage_a_max_chars))
    stage_a_max_requests = base_stage_a_max_requests
    stage_a_max_chars = base_stage_a_max_chars
    prompt_budget = {
        "total_chars": max(0, int(prompt_budget_total_chars)),
        "retry_delta_chars": max(0, int(prompt_budget_retry_delta_chars)),
        "stdout_chars": max(0, int(prompt_budget_stdout_chars)),
        "snapshot_chars": max(0, int(prompt_budget_snapshot_chars)),
        "evidence_chars": max(0, int(prompt_budget_evidence_chars)),
        "retry_history_chars": max(0, int(prompt_budget_retry_history_chars)),
    }
    normalized_context_compaction_mode = (
        context_compaction_mode
        if context_compaction_mode in CONTEXT_COMPACTION_MODES
        else "auto_overflow"
    )
    normalized_context_compaction_scope = {
        token.strip().lower()
        for token in str(context_compaction_scope or "").split(",")
        if token.strip() and token.strip().lower() in CONTEXT_COMPACTION_SCOPE_TOKENS
    }
    if not normalized_context_compaction_scope:
        normalized_context_compaction_scope = {"snapshot", "pytest", "files"}
    normalized_context_compaction_max_input_chars = max(0, int(context_compaction_max_input_chars))
    normalized_context_compaction_target_chars = max(1, int(context_compaction_target_chars))
    base_context_compaction_max_calls = max(0, int(context_compaction_max_calls))
    normalized_openai_context_window_tokens = max(0, int(openai_context_window_tokens))
    normalized_openai_context_window_source = str(openai_context_window_source or "")
    normalized_openai_context_window_probe_provider = str(
        openai_context_window_probe_provider or ""
    )
    normalized_openai_context_window_probe_error = str(openai_context_window_probe_error or "")
    normalized_openai_response_mode_autoswitch = (
        openai_response_mode_autoswitch
        if openai_response_mode_autoswitch in OPENAI_RESPONSE_MODE_AUTOSWITCH_MODES
        else "off"
    )
    normalized_adaptive_gate_policy = (
        adaptive_gate_policy if adaptive_gate_policy in ADAPTIVE_GATE_POLICIES else "balanced"
    )
    normalized_adaptive_max_helper_calls = max(0, int(adaptive_max_helper_calls))
    normalized_adaptive_budget_topup_max_calls = max(0, int(adaptive_budget_topup_max_calls))
    normalized_adaptive_stagnation_window = max(1, int(adaptive_stagnation_window))
    adaptive_no_progress_stop_window = max(2, normalized_adaptive_stagnation_window + 1)
    normalized_adaptive_saturation_threshold = min(
        1.0, max(0.0, float(adaptive_saturation_threshold))
    )
    normalized_adaptive_no_progress_token_cap = max(0, int(adaptive_no_progress_token_cap))
    normalized_adaptive_helper_budget_profile = (
        adaptive_helper_budget_profile
        if adaptive_helper_budget_profile in ADAPTIVE_HELPER_BUDGET_PROFILES
        else "default"
    )
    normalized_adaptive_openai_helper_response_mode = (
        adaptive_openai_helper_response_mode
        if adaptive_openai_helper_response_mode in ADAPTIVE_OPENAI_HELPER_RESPONSE_MODES
        else "inherit"
    )
    adaptive_mode_label = (
        "gated"
        if is_adaptive_gated
        else ("llm_discretion" if is_adaptive_llm_discretion else "none")
    )
    adaptive_policy_label = normalized_adaptive_gate_policy if is_adaptive_condition else "disabled"
    normalized_adaptive_online_policy_mode = (
        adaptive_online_policy_mode
        if adaptive_online_policy_mode in ADAPTIVE_ONLINE_POLICY_MODES
        else "off"
    )
    normalized_adaptive_online_policy_algo = (
        adaptive_online_policy_algo
        if adaptive_online_policy_algo in ADAPTIVE_ONLINE_POLICY_ALGOS
        else "linucb"
    )
    normalized_adaptive_online_feature_scaling_profile = normalize_feature_scaling_profile(
        adaptive_online_feature_scaling_profile
    )
    normalized_adaptive_online_action_score_biases = _parse_adaptive_online_action_score_biases(
        adaptive_online_action_score_biases
    )
    normalized_adaptive_budget_policy_mode = (
        adaptive_budget_policy_mode
        if adaptive_budget_policy_mode in ADAPTIVE_BUDGET_POLICY_MODES
        else "on"
    )
    normalized_adaptive_budget_policy_algo = (
        adaptive_budget_policy_algo
        if adaptive_budget_policy_algo in ADAPTIVE_BUDGET_POLICY_ALGOS
        else "linucb"
    )
    normalized_adaptive_budget_policy_architecture = (
        adaptive_budget_policy_architecture
        if adaptive_budget_policy_architecture in BUDGET_POLICY_ARCHITECTURES
        else BUDGET_POLICY_ARCH_TIER_BINARY
    )
    normalized_adaptive_budget_feature_scaling_profile = normalize_feature_scaling_profile(
        adaptive_budget_feature_scaling_profile
    )
    try:
        normalized_adaptive_budget_policy_tiers = _parse_adaptive_budget_policy_tiers(
            adaptive_budget_policy_tiers
        )
    except (TypeError, ValueError):
        normalized_adaptive_budget_policy_tiers = [1.0, 1.5, 2.0]
    normalized_adaptive_budget_policy_max_ups_per_case = max(
        0, int(adaptive_budget_policy_max_ups_per_case)
    )
    normalized_adaptive_budget_reward_rho_budget_exceeded = max(
        0.0, float(adaptive_budget_reward_rho_budget_exceeded)
    )
    normalized_syntax_preflight_enabled = bool(syntax_preflight_enabled)
    normalized_syntax_preflight_max_cycles = max(1, int(syntax_preflight_max_cycles))
    normalized_syntax_preflight_fallthrough = bool(syntax_preflight_fallthrough)
    normalized_patch_sanitize_enabled = bool(patch_sanitize_enabled)
    normalized_patch_verifier_scope = (
        str(patch_verifier_scope)
        if str(patch_verifier_scope)
        in {
            PATCH_VERIFIER_SCOPE_DISABLED,
            PATCH_VERIFIER_SCOPE_ADAPTIVE_ONLY,
            PATCH_VERIFIER_SCOPE_ALL,
        }
        else PATCH_VERIFIER_SCOPE_ADAPTIVE_ONLY
    )
    normalized_patch_verifier_min_score = max(0, int(patch_verifier_min_score))
    normalized_patch_verifier_infer_test_source_scope = bool(patch_verifier_infer_test_source_scope)
    normalized_case_fail_fast_on_request_budget_exceeded = bool(
        case_fail_fast_on_request_budget_exceeded
    )
    normalized_adaptive_controller_lambda_rlm = bool(adaptive_controller_lambda_rlm)
    normalized_adaptive_controller_lambda_rlm_context_window_chars = max(
        1, int(adaptive_controller_lambda_rlm_context_window_chars)
    )
    normalized_adaptive_controller_lambda_rlm_timeout_sec = max(
        1, int(adaptive_controller_lambda_rlm_timeout_sec)
    )
    normalized_adaptive_controller_lambda_rlm_repo_dir = Path(
        str(adaptive_controller_lambda_rlm_repo_dir or default_lambda_rlm_repo_dir())
    ).expanduser()
    normalized_adaptive_controller_lambda_rlm_python = Path(
        str(adaptive_controller_lambda_rlm_python or sys.executable)
    ).expanduser()
    if adaptive_online_policy_controller is not None:
        normalized_adaptive_online_policy_mode = adaptive_online_policy_controller.mode
        normalized_adaptive_online_policy_algo = adaptive_online_policy_controller.algorithm
        normalized_adaptive_online_feature_scaling_profile = normalize_feature_scaling_profile(
            getattr(adaptive_online_policy_controller.config, "feature_scaling_profile", "v1")
        )
        normalized_adaptive_online_action_score_biases = {
            str(action): float(value)
            for action, value in getattr(
                adaptive_online_policy_controller.config,
                "action_score_biases",
                normalized_adaptive_online_action_score_biases,
            ).items()
        }
    if adaptive_budget_policy_controller is not None:
        normalized_adaptive_budget_policy_mode = adaptive_budget_policy_controller.mode
        normalized_adaptive_budget_policy_algo = adaptive_budget_policy_controller.algorithm
        normalized_adaptive_budget_policy_architecture = str(
            getattr(
                adaptive_budget_policy_controller.config,
                "architecture",
                BUDGET_POLICY_ARCH_TIER_BINARY,
            )
        )
        normalized_adaptive_budget_feature_scaling_profile = normalize_feature_scaling_profile(
            getattr(adaptive_budget_policy_controller.config, "feature_scaling_profile", "v1")
        )
    normalized_adaptive_budget_bwk_run_token_budget = max(
        0,
        int(
            getattr(
                getattr(adaptive_budget_policy_controller, "config", None),
                "bwk_run_token_budget",
                adaptive_budget_bwk_run_token_budget,
            )
            or 0
        ),
    )
    normalized_adaptive_reward_lambda_tokens = max(0.0, float(adaptive_reward_lambda_tokens))
    normalized_adaptive_reward_mu_latency = max(0.0, float(adaptive_reward_mu_latency))
    normalized_adaptive_reward_token_scale = max(1.0, float(adaptive_reward_token_scale))
    normalized_adaptive_reward_latency_scale = max(1.0, float(adaptive_reward_latency_scale))
    normalized_adaptive_reward_eta_progress = max(0.0, float(adaptive_reward_eta_progress))
    normalized_adaptive_reward_kappa_stagnation = max(0.0, float(adaptive_reward_kappa_stagnation))
    normalized_adaptive_reward_success_hint = max(0.0, float(adaptive_reward_success_hint))
    normalized_adaptive_reward_progress_scale = max(0.0, float(adaptive_reward_progress_scale))
    normalized_adaptive_reward_stagnation_scale = max(0.0, float(adaptive_reward_stagnation_scale))
    normalized_adaptive_reward_materialization_scale = max(
        0.0, float(adaptive_reward_materialization_scale)
    )
    effective_run_metadata = (
        dict(run_metadata)
        if isinstance(run_metadata, dict)
        else _default_run_metadata(
            run_id=run_id,
            patcher_name=str(getattr(patcher, "name", "unknown")),
        )
    )
    patcher_name = str(getattr(patcher, "name", "unknown"))
    is_openai_patcher = patcher_name == "openai"
    is_baseline_patcher = patcher_name == "baseline"
    if not is_baseline_patcher:
        try:
            from evals.patchers.baseline_patcher import BaselinePatcher
        except Exception:
            pass
        else:
            is_baseline_patcher = isinstance(patcher, BaselinePatcher)
    adaptive_solver_architecture = ""
    adaptive_native_solver_deprecated = False
    if is_adaptive_condition:
        if is_baseline_patcher:
            adaptive_solver_architecture = "hybrid_baseline_controller_v1"
        elif is_openai_patcher:
            adaptive_solver_architecture = "legacy_native_repair_loop"
            adaptive_native_solver_deprecated = True
        else:
            adaptive_solver_architecture = "adaptive_controller_custom_patcher"
    normalized_prompt_retry_reflection = bool(prompt_retry_reflection)
    normalized_native_baseline_mode = str(native_baseline_mode or "").strip().lower()
    cot_baseline_semantics = (
        normalized_native_baseline_mode == Condition.COT_REPAIR.value and is_adaptive_condition
    )
    normalized_prompt_adaptive_stage_b_cot = bool(prompt_adaptive_stage_b_cot)
    effective_prompt_adaptive_stage_b_cot = (
        normalized_prompt_adaptive_stage_b_cot or cot_baseline_semantics
    )
    prompt_adaptive_stage_b_cot_source = (
        "flag+native_baseline"
        if normalized_prompt_adaptive_stage_b_cot and cot_baseline_semantics
        else "native_baseline"
        if cot_baseline_semantics
        else "flag"
        if normalized_prompt_adaptive_stage_b_cot
        else "off"
    )
    normalized_prompt_repair_method_scaffold = bool(prompt_repair_method_scaffold)
    normalized_prompt_retry_plan_then_patch = bool(prompt_retry_plan_then_patch)
    normalized_prompt_order_mode = _normalize_prompt_order_mode(prompt_order_mode)
    normalized_adaptive_reveal_mode = _normalize_adaptive_reveal_mode(adaptive_reveal_mode)
    normalized_snapshot_frame_pruning_mode = _normalize_snapshot_frame_pruning_mode(
        snapshot_frame_pruning_mode
    )

    def _method_is_inherited_noop(method: Any, *, base_method: Callable[..., Any]) -> bool:
        return getattr(method, "__func__", method) is base_method

    def _patcher_supports_request_context(patcher_for_attempt: Any) -> bool:
        explicit_support = getattr(patcher_for_attempt, "supports_request_context", None)
        if callable(explicit_support):
            try:
                return bool(explicit_support())
            except Exception:
                return False
        request_context_fn_local = getattr(patcher_for_attempt, "request_context", None)
        if not callable(request_context_fn_local):
            return False
        return not _method_is_inherited_noop(
            request_context_fn_local,
            base_method=Patcher.request_context,
        )

    def _patcher_supports_context_compaction(patcher_for_attempt: Any) -> bool:
        explicit_support = getattr(patcher_for_attempt, "supports_context_compaction", None)
        if callable(explicit_support):
            try:
                return bool(explicit_support())
            except Exception:
                return False
        context_compaction_fn_local = getattr(patcher_for_attempt, "compact_context", None)
        if not callable(context_compaction_fn_local):
            return False
        return not _method_is_inherited_noop(
            context_compaction_fn_local,
            base_method=Patcher.compact_context,
        )

    def _patcher_supports_strict_json_first(patcher_for_attempt: Any) -> bool:
        explicit_support = getattr(patcher_for_attempt, "supports_strict_json_first", None)
        if callable(explicit_support):
            try:
                return bool(explicit_support())
            except Exception:
                return False
        if not _patcher_supports_request_context(patcher_for_attempt):
            return False
        request_context_fn_local = getattr(patcher_for_attempt, "request_context", None)
        if not callable(request_context_fn_local):
            return False
        try:
            signature = inspect.signature(request_context_fn_local)
        except (TypeError, ValueError):
            return False
        return any(
            parameter.kind == inspect.Parameter.VAR_KEYWORD
            for parameter in signature.parameters.values()
        ) or ("strict_json_first" in signature.parameters)

    request_context_fn = getattr(patcher, "request_context", None)
    supports_context_request = _patcher_supports_request_context(patcher)
    context_compaction_fn = getattr(patcher, "compact_context", None)
    supports_context_compaction = _patcher_supports_context_compaction(patcher)
    raw_budget_policy_base_propose_budget_sec = getattr(patcher, "max_propose_sec", 0.0)
    try:
        budget_policy_base_propose_budget_sec = max(
            0.0, float(raw_budget_policy_base_propose_budget_sec)
        )
    except (TypeError, ValueError):
        budget_policy_base_propose_budget_sec = 0.0
    raw_base_openai_max_tokens = getattr(patcher, "max_tokens", 0)
    try:
        base_openai_max_tokens = max(0, int(raw_base_openai_max_tokens))
    except (TypeError, ValueError):
        base_openai_max_tokens = 0
    base_openai_response_mode = str(getattr(patcher, "response_mode", "") or "")
    requested_adaptive_tep_mode = adaptive_tep_mode.strip().lower()
    full_tep_supported = bool(
        requested_adaptive_tep_mode == "full"
        and is_openai_patcher
        and callable(getattr(patcher, "generate_text", None))
        and callable(getattr(patcher, "generate_json", None))
    )
    adaptive_tep_controller: AdaptiveTepController | None = None
    if full_tep_supported:
        adaptive_tep_controller = AdaptiveTepController(
            state_path=adaptive_tep_state_path,
            init_state_path=adaptive_tep_init_state_path,
            beta_init=adaptive_tep_beta_init,
            beta_decay=adaptive_tep_beta_decay,
            equilibrium_max_iters=adaptive_tep_equilibrium_max_iters,
            equilibrium_window=adaptive_tep_equilibrium_window,
            equilibrium_variance_threshold=adaptive_tep_equilibrium_variance_threshold,
            validation_buffer_size=adaptive_tep_validation_buffer_size,
        )
    full_tep_enabled = adaptive_tep_controller is not None

    def _bind_full_tep_generation_overrides(
        *,
        patcher_for_attempt: Any,
    ) -> tuple[Any, Any, bool, bool, bool]:
        request_context_fn_local = getattr(patcher_for_attempt, "request_context", None)
        context_compaction_fn_local = getattr(patcher_for_attempt, "compact_context", None)
        if not full_tep_enabled or adaptive_tep_controller is None:
            return (
                request_context_fn_local,
                context_compaction_fn_local,
                _patcher_supports_request_context(patcher_for_attempt),
                _patcher_supports_context_compaction(patcher_for_attempt),
                _patcher_supports_strict_json_first(patcher_for_attempt),
            )
        if not callable(getattr(patcher_for_attempt, "generate_json", None)):
            return (
                request_context_fn_local,
                context_compaction_fn_local,
                _patcher_supports_request_context(patcher_for_attempt),
                _patcher_supports_context_compaction(patcher_for_attempt),
                _patcher_supports_strict_json_first(patcher_for_attempt),
            )

        context_compaction_schema = {
            "type": "object",
            "properties": {
                "compacted_context": {"type": "string"},
                "reduction_rationale": {"type": "string"},
            },
            "required": ["compacted_context"],
            "additionalProperties": False,
        }

        def _full_tep_request_context(
            *,
            prompt: str,
            context_dir: Path,
            cwd: Path,
            strict_json_first: bool = False,
        ) -> SimpleNamespace:
            result = patcher_for_attempt.generate_json(
                system_prompt=adaptive_tep_controller.stage_a_system_prompt(
                    strict_json=bool(strict_json_first)
                ),
                prompt=prompt,
                context_dir=context_dir,
                cwd=cwd,
                schema_name="stage_a_request",
                schema=StageARequestPayloadModel.model_json_schema(),
                temperature=adaptive_tep_controller.node("stage_a_request").temperature,
                max_tokens=768,
                operation_name="tep_stage_a_request",
            )
            payload = normalize_stage_a_request_payload(result.payload or {})
            return SimpleNamespace(
                request_json=payload,
                raw_response=result.raw_response,
                meta=result.meta,
            )

        def _full_tep_compact_context(
            *,
            prompt: str,
            context_text: str,
            context_dir: Path,
            cwd: Path,
            target_chars: int,
        ) -> SimpleNamespace:
            result = patcher_for_attempt.generate_json(
                system_prompt=adaptive_tep_controller.compaction_system_prompt(
                    target_chars=max(1, int(target_chars))
                ),
                prompt=(
                    prompt.rstrip()
                    + "\n\nContext to compact:\n"
                    + str(context_text or "").rstrip()
                    + "\n"
                ),
                context_dir=context_dir,
                cwd=cwd,
                schema_name="context_compaction",
                schema=context_compaction_schema,
                temperature=adaptive_tep_controller.node("context_compaction").temperature,
                max_tokens=1024,
                operation_name="tep_context_compaction",
            )
            payload = result.payload if isinstance(result.payload, dict) else {}
            compacted_text = str(payload.get("compacted_context") or "").strip() or None
            return SimpleNamespace(
                compacted_text=compacted_text,
                raw_response=result.raw_response,
                meta=result.meta,
            )

        return _full_tep_request_context, _full_tep_compact_context, True, True, True

    context_compaction_calls_used = 0
    last_context_compaction_requested = False
    last_context_compaction_invoked = False
    last_context_compaction_success = False
    last_context_compaction_ratio = 0.0
    last_context_compaction_trigger = ""

    _strip_eval_only_case_artifacts(workdir)
    llmdebug_dir = workdir / ".llmdebug"
    previous_failure_context: dict[str, Any] | None = None
    seen_evidence_ids: set[str] = set()
    retry_history_turns: list[str] = []
    runner_force_patch_attempt_enabled = bool(force_patch_attempt)
    helper_calls_used = 0
    helper_calls_without_context = 0
    adaptive_budget_topup_calls_used = 0
    last_helper_need_more_context: bool | None = None
    last_helper_decision_reason = ""
    last_helper_action_executed = ACTION_SKIP_HELPER
    stagnation_streak = 0
    adaptive_no_progress_streak = 0
    adaptive_no_progress_token_sum = 0
    adaptive_budget_policy_base_tier_index = next(
        (
            idx
            for idx, value in enumerate(normalized_adaptive_budget_policy_tiers)
            if abs(float(value) - 1.0) <= 1e-9
        ),
        0,
    )
    adaptive_budget_policy_current_tier_index = adaptive_budget_policy_base_tier_index
    adaptive_budget_time_level = adaptive_budget_policy_base_tier_index
    adaptive_budget_tokens_level = adaptive_budget_policy_base_tier_index
    adaptive_budget_context_level = adaptive_budget_policy_base_tier_index
    adaptive_budget_policy_ups_used = 0
    if normalized_adaptive_budget_policy_architecture == BUDGET_POLICY_ARCH_HIERARCHICAL_V1:
        base_budget_control_profile = _build_adaptive_budget_family_profile(
            time_level=adaptive_budget_time_level,
            tokens_level=adaptive_budget_tokens_level,
            context_level=adaptive_budget_context_level,
            base_tier_index=adaptive_budget_policy_base_tier_index,
            tier_multipliers=normalized_adaptive_budget_policy_tiers,
            base_propose_budget_sec=budget_policy_base_propose_budget_sec,
            base_openai_max_tokens=base_openai_max_tokens,
            base_helper_max_calls=normalized_adaptive_max_helper_calls,
            base_stage_a_max_requests=base_stage_a_max_requests,
            base_stage_a_max_chars=base_stage_a_max_chars,
            base_retry_history_max_chars=base_retry_history_max_chars,
            base_context_compaction_max_calls=base_context_compaction_max_calls,
            context_compaction_enabled=normalized_context_compaction_mode != "off",
            base_response_mode=base_openai_response_mode,
            is_openai_patcher=is_openai_patcher,
        )
    else:
        base_budget_control_profile = _build_adaptive_budget_control_profile(
            tier_index=adaptive_budget_policy_base_tier_index,
            base_tier_index=adaptive_budget_policy_base_tier_index,
            tier_multipliers=normalized_adaptive_budget_policy_tiers,
            base_propose_budget_sec=budget_policy_base_propose_budget_sec,
            base_openai_max_tokens=base_openai_max_tokens,
            base_helper_max_calls=normalized_adaptive_max_helper_calls,
            base_stage_a_max_requests=base_stage_a_max_requests,
            base_stage_a_max_chars=base_stage_a_max_chars,
            base_retry_history_max_chars=base_retry_history_max_chars,
            base_context_compaction_max_calls=base_context_compaction_max_calls,
            context_compaction_enabled=normalized_context_compaction_mode != "off",
            base_response_mode=base_openai_response_mode,
            is_openai_patcher=is_openai_patcher,
        )
    file_index_cache: str | None = None
    file_index_lines_cache: tuple[str, ...] | None = None
    monolithic_context_cache: str | None = None

    def _file_index_lines_for_attempt() -> tuple[str, ...]:
        nonlocal file_index_lines_cache, file_index_cache
        if file_index_lines_cache is None:
            file_index_lines_cache = tuple(_collect_file_index_lines(workdir))
            file_index_cache = "\n".join(file_index_lines_cache)
        return file_index_lines_cache

    def _file_index_text_for_attempt() -> str:
        nonlocal file_index_cache
        if file_index_cache is None:
            file_index_cache = "\n".join(_file_index_lines_for_attempt())
        return file_index_cache

    def _monolithic_context_for_attempt() -> str:
        nonlocal monolithic_context_cache
        if monolithic_context_cache is None:
            monolithic_context_cache = collect_text_context(
                workdir,
                max_total_chars=max_context_chars,
                file_index_lines=_file_index_lines_for_attempt(),
            )
        return monolithic_context_cache

    def _invalidate_file_catalog_cache() -> None:
        nonlocal file_index_cache, file_index_lines_cache, monolithic_context_cache
        file_index_cache = None
        file_index_lines_cache = None
        monolithic_context_cache = None

    # Initial run (attempt 0, no patch yet).
    shutil.rmtree(llmdebug_dir, ignore_errors=True)
    _emit_progress(
        "pytest_before_initial",
        _progress_payload(
            owner="pytest",
            phase_label="pytest: initial before",
            note="initial verification",
            attempt=0,
            max_attempts=int(k),
        ),
    )
    current = _call_run_pytest_compat(
        workdir,
        case.python_args,
        condition,
        case.timeout_sec,
        output_format,
        progress_update=_emit_progress,
        progress_phase_label="pytest: initial before",
    )
    cumulative_runtime_sec = current.duration_sec
    if current.returncode == 0:
        attempt_dir = context_dir_base / "attempt_00"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        _write_run_outputs(
            attempt_dir,
            current,
            prefix="pytest_before",
            tail_lines=normalized_pytest_output_tail_lines,
        )
        record = _build_base_record(
            case=case,
            condition=condition,
            run_id=run_id,
            attempt=0,
            patcher_name=patcher_name,
            run_metadata=effective_run_metadata,
            attempt_metadata={
                **_empty_attempt_metadata(attempt=0),
                "max_attempts": int(k),
                "attempt_ordinal_label": f"0/{int(k)}",
                "context_protocol": default_context_protocol,
                "retry_prompt_mode": base_retry_prompt_mode,
                "retry_context_mode": effective_retry_context_mode,
                "retry_history_budget_chars": int(prompt_budget.get("retry_history_chars", 0)),
                "openai_context_window_source": normalized_openai_context_window_source,
                "openai_context_window_probe_provider": normalized_openai_context_window_probe_provider,
                "openai_context_window_probe_error": normalized_openai_context_window_probe_error,
                "openai_response_mode_autoswitch_policy": normalized_openai_response_mode_autoswitch,
                "adaptive_mode": adaptive_mode_label,
                "adaptive_policy": adaptive_policy_label,
                "adaptive_helper_budget_max_calls": base_budget_control_profile.helper_max_calls,
                "adaptive_helper_budget_unlimited": (
                    base_budget_control_profile.helper_max_calls <= 0
                ),
                "adaptive_helper_budget_profile": normalized_adaptive_helper_budget_profile,
                "adaptive_budget_topup_granted": False,
                "adaptive_budget_topup_reason": "",
                "adaptive_online_policy_mode": normalized_adaptive_online_policy_mode,
                "adaptive_online_policy_algo": normalized_adaptive_online_policy_algo,
                "adaptive_online_reward_lambda_tokens": normalized_adaptive_reward_lambda_tokens,
                "adaptive_online_reward_mu_latency": normalized_adaptive_reward_mu_latency,
                "adaptive_online_reward_token_scale": normalized_adaptive_reward_token_scale,
                "adaptive_online_reward_latency_scale": normalized_adaptive_reward_latency_scale,
                "adaptive_budget_policy_mode": normalized_adaptive_budget_policy_mode,
                "adaptive_budget_policy_algo": normalized_adaptive_budget_policy_algo,
                "adaptive_budget_policy_architecture": (
                    normalized_adaptive_budget_policy_architecture
                ),
                "adaptive_budget_tiers": [
                    float(value) for value in normalized_adaptive_budget_policy_tiers
                ],
                "adaptive_budget_tier_index_before": adaptive_budget_policy_base_tier_index,
                "adaptive_budget_tier_index_after": adaptive_budget_policy_base_tier_index,
                "adaptive_budget_family_selected": "",
                "adaptive_budget_family_applied": "",
                "adaptive_budget_available_families": [],
                "adaptive_budget_unavailable_family_reasons": {},
                "adaptive_budget_selected_multiplier": float(
                    normalized_adaptive_budget_policy_tiers[adaptive_budget_policy_base_tier_index]
                ),
                "adaptive_budget_applied_multiplier": float(
                    normalized_adaptive_budget_policy_tiers[adaptive_budget_policy_base_tier_index]
                ),
                "adaptive_budget_default_multiplier": float(
                    normalized_adaptive_budget_policy_tiers[adaptive_budget_policy_base_tier_index]
                ),
                "adaptive_budget_max_ups_per_case": normalized_adaptive_budget_policy_max_ups_per_case,
                "adaptive_budget_ups_used": 0,
                "adaptive_budget_reward_rho_budget_exceeded": (
                    normalized_adaptive_budget_reward_rho_budget_exceeded
                ),
                "syntax_preflight_enabled": normalized_syntax_preflight_enabled,
                "syntax_preflight_fallthrough": normalized_syntax_preflight_fallthrough,
                "syntax_preflight_cycles_used": 0,
                "syntax_preflight_failures": 0,
                "syntax_preflight_last_error_type": "",
                "syntax_preflight_last_error_msg": "",
            },
            model_metadata=_build_model_metadata(patcher=patcher, patch_meta=None),
            context_protocol=default_context_protocol,
        )
        record.update(
            {
                "success": True,
                "before_returncode": current.returncode,
                "before_duration_sec": current.duration_sec,
                "cumulative_runtime_sec": cumulative_runtime_sec,
                "attempt_wall_clock_sec": float(current.duration_sec),
                "note": "no_patch_needed",
            }
        )
        record.update(_timeout_fields(current))
        _append_record(record)
        return records

    # ── Retry loop: up to k attempts per (case, condition) unit ──────────
    #
    # Each iteration generates a patch and re-runs tests.  Two flags at the
    # bottom of the loop control early exit:
    #   attempt_terminated_early  - this attempt ended without reaching test re-run
    #   should_break_after_attempt - also abandon remaining attempts (stop case)
    #
    # When attempt_terminated_early=True and should_break_after_attempt=False,
    # the loop `continue`s to the next attempt.  When both are True, it `break`s.
    #
    # Six triggers can stop the loop before all k attempts complete:
    #   1. request_budget_exceeded + case_fail_fast_on_request_budget_exceeded
    #      (non-adaptive; sets should_break_after_attempt via fail_fast_on_budget_exceeded)
    #   2. patch_verifier_rejected  (scope-dependent; should_break_after_attempt = adaptive_stop_triggered)
    #   3. syntax_preflight_failed - cycle exit  (adaptive only; same mechanism)
    #   4. patch_apply_failed  (adaptive only; same mechanism)
    #   5. syntax_preflight_failed - post-apply  (adaptive only; same mechanism)
    #   6. tests_still_failing + no-progress token cap hit  (adaptive only;
    #      breaks directly via end-of-loop `if adaptive_stop_triggered: break`,
    #      NOT through should_break_after_attempt)
    #
    # Triggers 2-5 require adaptive_stop_triggered=True (from
    # _mark_adaptive_progress_stop), which only fires when
    # is_adaptive_condition=True and no progress is detected.
    # Trigger 6 also requires adaptive_stop_triggered but uses a separate
    # break path at the bottom of the loop body.
    # ───────────────────────────────────────────────────────────────────────
    while scored_attempts_consumed < k:
        attempt = scored_attempts_consumed + 1
        _resolve_expired_pending_online_credits(current_attempt=attempt)
        attempt_start = time.perf_counter()
        _emit_progress(
            "attempt_start",
            _progress_payload(
                owner="harness",
                phase_label="harness: attempt start",
                note="attempt started",
                attempt=int(attempt),
                max_attempts=int(k),
            ),
        )
        attempt_dir = context_dir_base / f"attempt_{attempt:02d}"
        attempt_dir.mkdir(parents=True, exist_ok=True)
        if is_baseline_patcher:
            remove_adaptive_controller_sidecars(context_dir=attempt_dir)
            (attempt_dir / "baseline_case_context.json").write_text(
                json.dumps(
                    {
                        "schema_version": BASELINE_CONTRACT_VERSION,
                        "case_id": case.case_id,
                        "python_args": list(case.python_args),
                        "timeout_sec": int(case.timeout_sec),
                        "description": case.description,
                        "category": case.category,
                        "difficulty": case.difficulty,
                        "snapshot_dependency": case.snapshot_dependency,
                    }
                ),
                encoding="utf-8",
            )

        _write_run_outputs(
            attempt_dir,
            current,
            prefix="pytest_before",
            tail_lines=normalized_pytest_output_tail_lines,
        )

        snapshot = None
        snapshot_error = None
        stage_b_snapshot = None
        snapshot_full_text: str | None = None
        snapshot_prompt_text = ""
        snapshot_prompt_toon_active = False
        snapshot_prompt_token_savings: dict[str, Any] = {}
        snapshot_artifact_text: str | None = None
        snapshot_pruning_metadata: dict[str, Any] = {
            "mode": normalized_snapshot_frame_pruning_mode,
            "applied": False,
            "frames_before": 0,
            "frames_after": 0,
            "kept_frame_indices": [],
        }
        _snapshot_read_sec = 0.0
        if _is_snapshot_condition(condition):
            _emit_progress(
                "snapshot_read",
                _progress_payload(
                    owner="harness",
                    phase_label="harness: snapshot read",
                    note="reading snapshot",
                    attempt=int(attempt),
                    max_attempts=int(k),
                ),
            )
            _snapshot_read_start = time.perf_counter()
            snapshot, snapshot_error = read_latest_snapshot(llmdebug_dir)
            _snapshot_read_sec = time.perf_counter() - _snapshot_read_start
            if snapshot is not None:
                stage_b_snapshot = snapshot
                snapshot_full_text = json.dumps(snapshot, indent=2, ensure_ascii=False)
                snapshot_artifact = _truncate_snapshot_for_artifact(
                    snapshot,
                    max_string_chars=normalized_snapshot_artifact_max_string_chars,
                )
                snapshot_artifact_text = json.dumps(snapshot_artifact, indent=2, ensure_ascii=False)
                (attempt_dir / "snapshot.json").write_text(snapshot_artifact_text, encoding="utf-8")

        failure_signature = _build_failure_signature_for_attempt(run=current, snapshot=snapshot)
        previous_failure_signature = (
            previous_failure_context.get("failure_signature")
            if isinstance(previous_failure_context, dict)
            else None
        )
        failure_signature_repeat = bool(
            previous_failure_signature and previous_failure_signature == failure_signature
        )
        stagnation_streak = (stagnation_streak + 1) if failure_signature_repeat else 0
        budget_policy_family_levels_before = {
            ACTION_BUDGET_FAMILY_TIME: adaptive_budget_time_level,
            ACTION_BUDGET_FAMILY_TOKENS: adaptive_budget_tokens_level,
            ACTION_BUDGET_FAMILY_CONTEXT: adaptive_budget_context_level,
        }
        if normalized_adaptive_budget_policy_architecture == BUDGET_POLICY_ARCH_HIERARCHICAL_V1:
            budget_policy_tier_index_before = max(budget_policy_family_levels_before.values())
            budget_policy_default_profile = _build_adaptive_budget_family_profile(
                time_level=adaptive_budget_time_level,
                tokens_level=adaptive_budget_tokens_level,
                context_level=adaptive_budget_context_level,
                base_tier_index=adaptive_budget_policy_base_tier_index,
                tier_multipliers=normalized_adaptive_budget_policy_tiers,
                base_propose_budget_sec=budget_policy_base_propose_budget_sec,
                base_openai_max_tokens=base_openai_max_tokens,
                base_helper_max_calls=normalized_adaptive_max_helper_calls,
                base_stage_a_max_requests=base_stage_a_max_requests,
                base_stage_a_max_chars=base_stage_a_max_chars,
                base_retry_history_max_chars=base_retry_history_max_chars,
                base_context_compaction_max_calls=base_context_compaction_max_calls,
                context_compaction_enabled=normalized_context_compaction_mode != "off",
                base_response_mode=base_openai_response_mode,
                is_openai_patcher=is_openai_patcher,
            )
        else:
            budget_policy_tier_index_before = adaptive_budget_policy_current_tier_index
            budget_policy_default_profile = _build_adaptive_budget_control_profile(
                tier_index=budget_policy_tier_index_before,
                base_tier_index=adaptive_budget_policy_base_tier_index,
                tier_multipliers=normalized_adaptive_budget_policy_tiers,
                base_propose_budget_sec=budget_policy_base_propose_budget_sec,
                base_openai_max_tokens=base_openai_max_tokens,
                base_helper_max_calls=normalized_adaptive_max_helper_calls,
                base_stage_a_max_requests=base_stage_a_max_requests,
                base_stage_a_max_chars=base_stage_a_max_chars,
                base_retry_history_max_chars=base_retry_history_max_chars,
                base_context_compaction_max_calls=base_context_compaction_max_calls,
                context_compaction_enabled=normalized_context_compaction_mode != "off",
                base_response_mode=base_openai_response_mode,
                is_openai_patcher=is_openai_patcher,
            )
        budget_policy_selected_profile = budget_policy_default_profile
        budget_policy_applied_profile = budget_policy_default_profile
        budget_policy_default_controls = _adaptive_budget_control_profile_to_dict(
            budget_policy_default_profile
        )
        budget_policy_selected_controls = dict(budget_policy_default_controls)
        budget_policy_applied_controls = dict(budget_policy_default_controls)
        budget_policy_selected_controls_changed = _adaptive_budget_control_changes(
            current=budget_policy_default_profile,
            target=budget_policy_selected_profile,
        )
        budget_policy_applied_controls_changed = dict(budget_policy_selected_controls_changed)
        budget_policy_partial_override = False
        budget_policy_partial_override_errors: dict[str, str] = {}
        budget_policy_tier_index_after = budget_policy_tier_index_before
        budget_policy_selected_family = ""
        budget_policy_applied_family = ""
        budget_policy_available_families: tuple[str, ...] = ()
        budget_policy_unavailable_family_reasons: dict[str, str] = {}
        budget_policy_candidate_profiles: dict[str, AdaptiveBudgetControlProfile] = {}
        budget_policy_candidate_changes: dict[str, dict[str, bool]] = {}
        budget_policy_candidate_levels: dict[str, dict[str, int]] = {}
        budget_policy_selected_family_levels = dict(budget_policy_family_levels_before)
        budget_policy_applied_family_levels = dict(budget_policy_family_levels_before)
        budget_policy_default_propose_multiplier = float(
            budget_policy_default_profile.propose_budget_multiplier
        )
        if normalized_adaptive_budget_policy_architecture == BUDGET_POLICY_ARCH_HIERARCHICAL_V1:
            budget_policy_default_multiplier = _adaptive_budget_recorded_multiplier(
                family="",
                family_levels=budget_policy_family_levels_before,
                tier_multipliers=normalized_adaptive_budget_policy_tiers,
                fallback=budget_policy_default_propose_multiplier,
            )
        else:
            budget_policy_default_multiplier = budget_policy_default_propose_multiplier
        budget_policy_selected_multiplier = budget_policy_default_multiplier
        budget_policy_applied_multiplier = budget_policy_default_multiplier
        budget_policy_applied_propose_multiplier = budget_policy_default_propose_multiplier
        stage_a_max_requests = int(budget_policy_default_profile.stage_a_max_requests)
        stage_a_max_chars = int(budget_policy_default_profile.stage_a_max_chars)
        effective_retry_history_max_chars = int(
            budget_policy_default_profile.retry_history_max_chars
        )
        effective_context_compaction_max_calls = int(
            budget_policy_default_profile.context_compaction_max_calls
        )
        adaptive_budget_unlimited = budget_policy_default_profile.helper_max_calls <= 0
        effective_helper_budget_limit = (
            0
            if adaptive_budget_unlimited
            else max(
                0,
                int(budget_policy_default_profile.helper_max_calls)
                + adaptive_budget_topup_calls_used,
            )
        )
        helper_budget_exhausted = bool(
            is_adaptive_condition
            and (not adaptive_budget_unlimited)
            and helper_calls_used >= effective_helper_budget_limit
        )
        helper_saturation_guarded = False
        if (
            is_adaptive_condition
            and helper_calls_used >= max(2, normalized_adaptive_stagnation_window)
            and helper_calls_used > 0
        ):
            low_yield_ratio = helper_calls_without_context / float(helper_calls_used)
            helper_saturation_guarded = low_yield_ratio >= normalized_adaptive_saturation_threshold

        helper_decision_mode = adaptive_mode_label
        helper_should_consult = False
        helper_invoked = False
        helper_action_default = _default_helper_action()
        helper_action_selected = ACTION_SKIP_HELPER
        helper_action_executed = ACTION_SKIP_HELPER
        helper_action_fallback_reason = ""
        helper_materialized_context = False
        helper_outcome = ADAPTIVE_HELPER_OUTCOME_NOT_INVOKED
        helper_failure_kind = ""
        helper_retry_count = 0
        helper_transport_retry_count = 0
        helper_need_more_context: bool | None = None
        helper_decision_reason = "not_adaptive_condition"
        adaptive_context_requested = False
        adaptive_snapshot_included = False
        adaptive_snapshot_lite_included = False
        adaptive_snapshot_attempt1_denied = False
        adaptive_snapshot_attempt1_denied_reason = ""
        adaptive_budget_topup_granted = False
        adaptive_budget_topup_reason = ""
        patcher_for_attempt = patcher
        (
            request_context_fn,
            context_compaction_fn,
            supports_context_request,
            supports_context_compaction,
            supports_strict_json_first,
        ) = _bind_full_tep_generation_overrides(patcher_for_attempt=patcher_for_attempt)
        online_policy_decision: PolicyDecision | None = None
        online_policy_features_raw: dict[str, float] = {}
        online_policy_features_scaled: dict[str, float] = {}
        helper_available_actions: tuple[str, ...] = ()
        helper_unavailable_action_reasons: dict[str, str] = {}
        online_policy_eligible = False
        online_policy_override_reason = ""
        online_policy_mode_for_attempt = (
            normalized_adaptive_online_policy_mode if is_adaptive_llm_discretion else "off"
        )
        online_policy_algo_for_attempt = (
            normalized_adaptive_online_policy_algo if is_adaptive_llm_discretion else ""
        )
        budget_policy_decision: BudgetPolicyDecision | None = None
        budget_policy_features_raw: dict[str, float] = {}
        budget_policy_features_scaled: dict[str, float] = {}
        budget_policy_eligible = False
        budget_policy_override_reason = ""
        budget_policy_mode_for_attempt = (
            normalized_adaptive_budget_policy_mode if is_adaptive_condition else "off"
        )
        budget_policy_algo_for_attempt = (
            normalized_adaptive_budget_policy_algo if is_adaptive_condition else ""
        )
        adaptive_stagnation_override_applied = False
        adaptive_stagnation_override_reason = ""
        adaptive_force_helper_stage_a = False
        adaptive_recovery_forced_action = ""
        adaptive_recovery_override_reason = ""
        adaptive_recovery_decision_reason = ""

        # --- Pre-gate: traceback informativeness ---
        pregate_result = PreGateResult(
            should_skip=False,
            score=0.0,
            exception_type="",
            reason="",
            f_type_specificity=0.0,
            f_message_diagnostic=0.0,
            f_crash_in_user_code=0.0,
            f_assertion_quality=0.0,
            f_frame_depth=0.0,
        )
        if is_adaptive_condition and adaptive_informativeness_pregate:
            pregate_result = _traceback_informativeness_pre_gate(
                snapshot=snapshot,
                stdout=current.stdout,
                stderr=current.stderr,
                threshold=adaptive_informativeness_pregate_threshold,
            )
        if snapshot is not None:
            stage_b_snapshot, snapshot_pruning_metadata = prune_snapshot_for_prompt(
                snapshot,
                mode=normalized_snapshot_frame_pruning_mode,
                pregate_score=pregate_result.score if is_adaptive_condition else 0.0,
                failure_signature_repeat=failure_signature_repeat,
            )
            if stage_b_snapshot is not None:
                snapshot_prompt_token_savings = estimate_token_savings(
                    stage_b_snapshot,
                    use_toon=bool(toon_encoding_requested and is_openai_patcher),
                    use_compact_keys=True,
                    model=_token_estimate_model_name(patcher_for_attempt),
                )
            if toon_encoding_requested and is_openai_patcher and stage_b_snapshot is not None:
                snapshot_prompt_text = _build_snapshot_prompt_text(
                    stage_b_snapshot,
                    use_toon=True,
                    use_compact_keys=True,
                )
                snapshot_prompt_toon_active = bool(snapshot_prompt_text)

        if is_adaptive_condition:
            budget_policy_max_tier_index = max(0, len(normalized_adaptive_budget_policy_tiers) - 1)
            can_apply_more_ups = (
                normalized_adaptive_budget_policy_max_ups_per_case > 0
                and adaptive_budget_policy_ups_used
                < normalized_adaptive_budget_policy_max_ups_per_case
            )
            next_budget_tier_index = min(
                _adaptive_budget_next_tier_index(
                    current_tier_index=budget_policy_tier_index_before,
                    tier_multipliers=normalized_adaptive_budget_policy_tiers,
                ),
                budget_policy_max_tier_index,
            )
            time_delta_magnitude = 0.0
            tokens_delta_magnitude = 0.0
            context_delta_magnitude = 0.0
            if normalized_adaptive_budget_policy_architecture == BUDGET_POLICY_ARCH_HIERARCHICAL_V1:
                (
                    budget_policy_available_families,
                    budget_policy_unavailable_family_reasons,
                    budget_policy_candidate_profiles,
                    budget_policy_candidate_changes,
                    budget_policy_candidate_levels,
                ) = _adaptive_budget_family_availability(
                    current_profile=budget_policy_default_profile,
                    time_level=adaptive_budget_time_level,
                    tokens_level=adaptive_budget_tokens_level,
                    context_level=adaptive_budget_context_level,
                    max_tier_index=budget_policy_max_tier_index,
                    base_tier_index=adaptive_budget_policy_base_tier_index,
                    tier_multipliers=normalized_adaptive_budget_policy_tiers,
                    base_propose_budget_sec=budget_policy_base_propose_budget_sec,
                    base_openai_max_tokens=base_openai_max_tokens,
                    base_helper_max_calls=normalized_adaptive_max_helper_calls,
                    base_stage_a_max_requests=base_stage_a_max_requests,
                    base_stage_a_max_chars=base_stage_a_max_chars,
                    base_retry_history_max_chars=base_retry_history_max_chars,
                    base_context_compaction_max_calls=base_context_compaction_max_calls,
                    context_compaction_enabled=normalized_context_compaction_mode != "off",
                    base_response_mode=base_openai_response_mode,
                    is_openai_patcher=is_openai_patcher,
                )
                next_budget_tier_changes = _adaptive_budget_union_change_flags(
                    budget_policy_candidate_changes
                )
                next_budget_tier_changes_any = bool(budget_policy_available_families)
                can_increase_tier = bool(budget_policy_available_families)
                time_delta_magnitude = _adaptive_budget_level_step_delta(
                    current_level=adaptive_budget_time_level,
                    max_tier_index=budget_policy_max_tier_index,
                    tier_multipliers=normalized_adaptive_budget_policy_tiers,
                )
                tokens_delta_magnitude = _adaptive_budget_level_step_delta(
                    current_level=adaptive_budget_tokens_level,
                    max_tier_index=budget_policy_max_tier_index,
                    tier_multipliers=normalized_adaptive_budget_policy_tiers,
                )
                context_delta_magnitude = _adaptive_budget_level_step_delta(
                    current_level=adaptive_budget_context_level,
                    max_tier_index=budget_policy_max_tier_index,
                    tier_multipliers=normalized_adaptive_budget_policy_tiers,
                )
            else:
                can_increase_tier = (
                    adaptive_budget_policy_current_tier_index < budget_policy_max_tier_index
                )
                budget_policy_next_profile = _build_adaptive_budget_control_profile(
                    tier_index=next_budget_tier_index,
                    base_tier_index=adaptive_budget_policy_base_tier_index,
                    tier_multipliers=normalized_adaptive_budget_policy_tiers,
                    base_propose_budget_sec=budget_policy_base_propose_budget_sec,
                    base_openai_max_tokens=base_openai_max_tokens,
                    base_helper_max_calls=normalized_adaptive_max_helper_calls,
                    base_stage_a_max_requests=base_stage_a_max_requests,
                    base_stage_a_max_chars=base_stage_a_max_chars,
                    base_retry_history_max_chars=base_retry_history_max_chars,
                    base_context_compaction_max_calls=base_context_compaction_max_calls,
                    context_compaction_enabled=normalized_context_compaction_mode != "off",
                    base_response_mode=base_openai_response_mode,
                    is_openai_patcher=is_openai_patcher,
                )
                next_budget_tier_changes = _adaptive_budget_control_changes(
                    current=budget_policy_default_profile,
                    target=budget_policy_next_profile,
                )
                next_budget_tier_changes_any = _adaptive_budget_controls_changed(
                    next_budget_tier_changes
                )
                time_delta_magnitude = max(
                    0.0,
                    budget_policy_next_profile.propose_budget_multiplier
                    - budget_policy_default_profile.propose_budget_multiplier,
                )
                tokens_delta_magnitude = time_delta_magnitude
                context_delta_magnitude = time_delta_magnitude
            budget_policy_features_raw = _adaptive_budget_feature_vector(
                attempt=attempt,
                max_attempts=k,
                timed_out=bool(current.timed_out),
                failure_signature_repeat=failure_signature_repeat,
                stagnation_streak=stagnation_streak,
                helper_calls_used=helper_calls_used,
                helper_calls_without_context=helper_calls_without_context,
                adaptive_no_progress_streak=adaptive_no_progress_streak,
                adaptive_no_progress_token_sum=adaptive_no_progress_token_sum,
                helper_budget_exhausted=helper_budget_exhausted,
                helper_saturation_guarded=helper_saturation_guarded,
                supports_context_request=supports_context_request,
                snapshot_available=snapshot is not None,
                pregate_result=pregate_result,
                context_compaction_enabled=normalized_context_compaction_mode != "off",
                supports_context_compaction=supports_context_compaction,
                context_compaction_calls_used=context_compaction_calls_used,
                context_compaction_max_calls=effective_context_compaction_max_calls,
                last_context_compaction_requested=last_context_compaction_requested,
                last_context_compaction_invoked=last_context_compaction_invoked,
                last_context_compaction_success=last_context_compaction_success,
                last_context_compaction_ratio=last_context_compaction_ratio,
                last_context_compaction_trigger=last_context_compaction_trigger,
                current_tier_index=budget_policy_tier_index_before,
                max_tier_index=budget_policy_max_tier_index,
                ups_used=adaptive_budget_policy_ups_used,
                max_ups_per_case=normalized_adaptive_budget_policy_max_ups_per_case,
                base_propose_budget_sec=budget_policy_base_propose_budget_sec,
                next_tier_changes=next_budget_tier_changes,
                next_tier_multiplier_delta=max(
                    time_delta_magnitude,
                    tokens_delta_magnitude,
                    context_delta_magnitude,
                ),
                last_helper_need_more_context=last_helper_need_more_context,
                last_helper_decision_reason=last_helper_decision_reason,
                budget_architecture=normalized_adaptive_budget_policy_architecture,
                current_time_level=adaptive_budget_time_level,
                current_tokens_level=adaptive_budget_tokens_level,
                current_context_level=adaptive_budget_context_level,
                can_raise_time=(ACTION_BUDGET_FAMILY_TIME in budget_policy_available_families)
                if normalized_adaptive_budget_policy_architecture
                == BUDGET_POLICY_ARCH_HIERARCHICAL_V1
                else bool(next_budget_tier_changes.get("propose_budget")),
                can_raise_tokens=(ACTION_BUDGET_FAMILY_TOKENS in budget_policy_available_families)
                if normalized_adaptive_budget_policy_architecture
                == BUDGET_POLICY_ARCH_HIERARCHICAL_V1
                else bool(next_budget_tier_changes.get("max_tokens")),
                can_raise_context=(ACTION_BUDGET_FAMILY_CONTEXT in budget_policy_available_families)
                if normalized_adaptive_budget_policy_architecture
                == BUDGET_POLICY_ARCH_HIERARCHICAL_V1
                else bool(
                    next_budget_tier_changes.get("helper_allowance")
                    or next_budget_tier_changes.get("stage_a_requests")
                    or next_budget_tier_changes.get("stage_a_chars")
                    or next_budget_tier_changes.get("retry_history_budget")
                    or next_budget_tier_changes.get("compaction_allowance")
                ),
                time_delta_magnitude=time_delta_magnitude,
                tokens_delta_magnitude=tokens_delta_magnitude,
                context_delta_magnitude=context_delta_magnitude,
                toon_encoding_active=snapshot_prompt_toon_active,
            )
            budget_policy_features_scaled = scale_feature_mapping(
                budget_policy_features_raw,
                profile=normalized_adaptive_budget_feature_scaling_profile,
            )
            budget_policy_eligible = bool(
                normalized_adaptive_budget_policy_mode in {"shadow", "on"}
                and adaptive_budget_policy_controller is not None
                and can_apply_more_ups
                and next_budget_tier_changes_any
            )
            if budget_policy_eligible:
                budget_policy_decision = adaptive_budget_policy_controller.decide(
                    feature_values=budget_policy_features_raw,
                    default_action=ACTION_KEEP_BUDGET,
                    family_feature_values=budget_policy_features_raw,
                    available_families=budget_policy_available_families,
                    unavailable_family_reasons=budget_policy_unavailable_family_reasons,
                )
                budget_policy_features_scaled = dict(budget_policy_decision.feature_values)
                budget_policy_mode_for_attempt = budget_policy_decision.mode
                budget_policy_algo_for_attempt = budget_policy_decision.algorithm
                budget_policy_selected_family = str(budget_policy_decision.selected_family or "")
                budget_policy_applied_family = str(budget_policy_decision.applied_family or "")
                if (
                    normalized_adaptive_budget_policy_architecture
                    == BUDGET_POLICY_ARCH_HIERARCHICAL_V1
                ):
                    if (
                        budget_policy_decision.selected_action == ACTION_INCREASE_BUDGET
                        and budget_policy_selected_family in budget_policy_candidate_profiles
                    ):
                        budget_policy_selected_profile = budget_policy_candidate_profiles[
                            budget_policy_selected_family
                        ]
                        budget_policy_selected_controls = _adaptive_budget_control_profile_to_dict(
                            budget_policy_selected_profile
                        )
                        budget_policy_selected_controls_changed = dict(
                            budget_policy_candidate_changes[budget_policy_selected_family]
                        )
                        budget_policy_selected_family_levels = dict(
                            budget_policy_candidate_levels[budget_policy_selected_family]
                        )
                        budget_policy_selected_multiplier = _adaptive_budget_recorded_multiplier(
                            family=budget_policy_selected_family,
                            family_levels=budget_policy_selected_family_levels,
                            tier_multipliers=normalized_adaptive_budget_policy_tiers,
                            fallback=float(
                                budget_policy_selected_profile.propose_budget_multiplier
                            ),
                        )
                    if (
                        budget_policy_decision.applied_action == ACTION_INCREASE_BUDGET
                        and budget_policy_applied_family in budget_policy_candidate_profiles
                    ):
                        budget_policy_applied_profile = budget_policy_candidate_profiles[
                            budget_policy_applied_family
                        ]
                        budget_policy_applied_controls = _adaptive_budget_control_profile_to_dict(
                            budget_policy_applied_profile
                        )
                        budget_policy_applied_controls_changed = dict(
                            budget_policy_candidate_changes[budget_policy_applied_family]
                        )
                        budget_policy_applied_family_levels = dict(
                            budget_policy_candidate_levels[budget_policy_applied_family]
                        )
                        budget_policy_default_multiplier = _adaptive_budget_recorded_multiplier(
                            family=budget_policy_applied_family,
                            family_levels=budget_policy_family_levels_before,
                            tier_multipliers=normalized_adaptive_budget_policy_tiers,
                            fallback=budget_policy_default_propose_multiplier,
                        )
                        budget_policy_applied_multiplier = _adaptive_budget_recorded_multiplier(
                            family=budget_policy_applied_family,
                            family_levels=budget_policy_applied_family_levels,
                            tier_multipliers=normalized_adaptive_budget_policy_tiers,
                            fallback=float(budget_policy_applied_profile.propose_budget_multiplier),
                        )
                        budget_policy_tier_index_after = max(
                            budget_policy_applied_family_levels.values()
                        )
                    if (
                        (
                            budget_policy_decision.applied_action
                            != budget_policy_decision.selected_action
                        )
                        or budget_policy_applied_family != budget_policy_selected_family
                    ) and not budget_policy_override_reason:
                        budget_policy_override_reason = "shadow_mode_baseline"
                else:
                    if budget_policy_decision.selected_action == ACTION_INCREASE_BUDGET:
                        budget_policy_selected_profile = budget_policy_next_profile
                        budget_policy_selected_controls = _adaptive_budget_control_profile_to_dict(
                            budget_policy_selected_profile
                        )
                        budget_policy_selected_controls_changed = _adaptive_budget_control_changes(
                            current=budget_policy_default_profile,
                            target=budget_policy_selected_profile,
                        )
                        budget_policy_selected_multiplier = float(
                            budget_policy_selected_profile.propose_budget_multiplier
                        )
                    if budget_policy_decision.applied_action == ACTION_INCREASE_BUDGET:
                        budget_policy_tier_index_after = next_budget_tier_index
                        budget_policy_applied_profile = budget_policy_next_profile
                        budget_policy_applied_controls = _adaptive_budget_control_profile_to_dict(
                            budget_policy_applied_profile
                        )
                        budget_policy_applied_controls_changed = _adaptive_budget_control_changes(
                            current=budget_policy_default_profile,
                            target=budget_policy_applied_profile,
                        )
                        budget_policy_applied_multiplier = float(
                            budget_policy_applied_profile.propose_budget_multiplier
                        )
                    if (
                        budget_policy_decision.applied_action
                        != budget_policy_decision.selected_action
                        and not budget_policy_override_reason
                    ):
                        budget_policy_override_reason = "shadow_mode_baseline"
            else:
                if normalized_adaptive_budget_policy_mode == "off":
                    budget_policy_override_reason = "policy_off"
                elif adaptive_budget_policy_controller is None:
                    budget_policy_override_reason = "policy_controller_unavailable"
                elif normalized_adaptive_budget_policy_max_ups_per_case <= 0:
                    budget_policy_override_reason = "constraint_max_ups_disabled"
                elif (
                    adaptive_budget_policy_ups_used
                    >= normalized_adaptive_budget_policy_max_ups_per_case
                ):
                    budget_policy_override_reason = "constraint_max_ups_reached"
                elif (
                    normalized_adaptive_budget_policy_architecture
                    == BUDGET_POLICY_ARCH_HIERARCHICAL_V1
                ):
                    budget_policy_override_reason = "constraint_no_available_family"
                elif not can_increase_tier:
                    budget_policy_override_reason = "constraint_tier_ceiling"
                elif not next_budget_tier_changes_any:
                    budget_policy_override_reason = "constraint_next_tier_no_control_change"
                else:
                    budget_policy_override_reason = "not_eligible"

            should_apply_budget_change = bool(
                budget_policy_decision is not None
                and budget_policy_decision.applied_action == ACTION_INCREASE_BUDGET
                and (
                    (
                        normalized_adaptive_budget_policy_architecture
                        == BUDGET_POLICY_ARCH_HIERARCHICAL_V1
                        and budget_policy_applied_family in BUDGET_POLICY_FAMILIES
                    )
                    or budget_policy_tier_index_after > budget_policy_tier_index_before
                )
            )
            if should_apply_budget_change:
                (
                    patcher_for_attempt,
                    budget_policy_applied_profile,
                    budget_policy_partial_override_errors,
                ) = _apply_adaptive_budget_profile_to_patcher(
                    patcher=patcher_for_attempt,
                    applied_profile=budget_policy_applied_profile,
                    base_propose_budget_sec=budget_policy_base_propose_budget_sec,
                    base_openai_max_tokens=base_openai_max_tokens,
                    base_response_mode=base_openai_response_mode,
                    is_openai_patcher=is_openai_patcher,
                )
                budget_policy_partial_override = bool(budget_policy_partial_override_errors)
                budget_policy_applied_controls = _adaptive_budget_control_profile_to_dict(
                    budget_policy_applied_profile
                )
                budget_policy_applied_controls_changed = _adaptive_budget_control_changes(
                    current=budget_policy_default_profile,
                    target=budget_policy_applied_profile,
                )
                budget_policy_applied_propose_multiplier = float(
                    budget_policy_applied_profile.propose_budget_multiplier
                )
                if not _adaptive_budget_controls_changed(budget_policy_applied_controls_changed):
                    budget_policy_tier_index_after = budget_policy_tier_index_before
                    budget_policy_applied_profile = budget_policy_default_profile
                    budget_policy_applied_controls = dict(budget_policy_default_controls)
                    budget_policy_applied_controls_changed = _adaptive_budget_control_changes(
                        current=budget_policy_default_profile,
                        target=budget_policy_applied_profile,
                    )
                    if (
                        normalized_adaptive_budget_policy_architecture
                        == BUDGET_POLICY_ARCH_HIERARCHICAL_V1
                    ):
                        budget_policy_default_multiplier = _adaptive_budget_recorded_multiplier(
                            family="",
                            family_levels=budget_policy_family_levels_before,
                            tier_multipliers=normalized_adaptive_budget_policy_tiers,
                            fallback=budget_policy_default_propose_multiplier,
                        )
                    else:
                        budget_policy_default_multiplier = budget_policy_default_propose_multiplier
                    budget_policy_applied_multiplier = budget_policy_default_multiplier
                    budget_policy_applied_propose_multiplier = (
                        budget_policy_default_propose_multiplier
                    )
                    budget_policy_applied_family = ""
                    budget_policy_applied_family_levels = dict(budget_policy_family_levels_before)
                    if budget_policy_decision is not None:
                        budget_policy_decision = replace(
                            budget_policy_decision,
                            applied_action=ACTION_KEEP_BUDGET,
                            applied_family="",
                        )
                    if not budget_policy_override_reason:
                        budget_policy_override_reason = (
                            "constraint_budget_profile_no_effective_change"
                        )
                else:
                    if (
                        normalized_adaptive_budget_policy_architecture
                        == BUDGET_POLICY_ARCH_HIERARCHICAL_V1
                    ):
                        budget_policy_applied_multiplier = _adaptive_budget_recorded_multiplier(
                            family=budget_policy_applied_family,
                            family_levels=budget_policy_applied_family_levels,
                            tier_multipliers=normalized_adaptive_budget_policy_tiers,
                            fallback=budget_policy_applied_propose_multiplier,
                        )
                    else:
                        budget_policy_applied_multiplier = budget_policy_applied_propose_multiplier
                    if (
                        normalized_adaptive_budget_policy_architecture
                        == BUDGET_POLICY_ARCH_HIERARCHICAL_V1
                    ):
                        adaptive_budget_time_level = budget_policy_applied_family_levels[
                            ACTION_BUDGET_FAMILY_TIME
                        ]
                        adaptive_budget_tokens_level = budget_policy_applied_family_levels[
                            ACTION_BUDGET_FAMILY_TOKENS
                        ]
                        adaptive_budget_context_level = budget_policy_applied_family_levels[
                            ACTION_BUDGET_FAMILY_CONTEXT
                        ]
                        adaptive_budget_policy_current_tier_index = max(
                            adaptive_budget_time_level,
                            adaptive_budget_tokens_level,
                            adaptive_budget_context_level,
                        )
                    else:
                        adaptive_budget_policy_current_tier_index = budget_policy_tier_index_after
                    adaptive_budget_policy_ups_used += 1

        (
            request_context_fn,
            context_compaction_fn,
            supports_context_request,
            supports_context_compaction,
            supports_strict_json_first,
        ) = _bind_full_tep_generation_overrides(patcher_for_attempt=patcher_for_attempt)
        stage_a_max_requests = int(budget_policy_applied_profile.stage_a_max_requests)
        stage_a_max_chars = int(budget_policy_applied_profile.stage_a_max_chars)
        effective_retry_history_max_chars = int(
            budget_policy_applied_profile.retry_history_max_chars
        )
        effective_context_compaction_max_calls = int(
            budget_policy_applied_profile.context_compaction_max_calls
        )
        adaptive_budget_unlimited = budget_policy_applied_profile.helper_max_calls <= 0
        effective_helper_budget_limit = (
            0
            if adaptive_budget_unlimited
            else max(
                0,
                int(budget_policy_applied_profile.helper_max_calls)
                + adaptive_budget_topup_calls_used,
            )
        )
        helper_budget_exhausted = bool(
            is_adaptive_condition
            and (not adaptive_budget_unlimited)
            and helper_calls_used >= effective_helper_budget_limit
        )
        previous_note_for_recovery = (
            str(previous_failure_context.get("note") or "")
            if isinstance(previous_failure_context, dict)
            else ""
        )
        previous_helper_failure_kind = (
            str(previous_failure_context.get("adaptive_helper_failure_kind") or "")
            if isinstance(previous_failure_context, dict)
            else ""
        )
        previous_progress_reason = (
            str(previous_failure_context.get("adaptive_progress_reason") or "")
            if isinstance(previous_failure_context, dict)
            else ""
        )
        previous_helper_need_more_context_raw = (
            previous_failure_context.get("adaptive_helper_need_more_context")
            if isinstance(previous_failure_context, dict)
            else None
        )
        previous_helper_need_more_context = (
            previous_helper_need_more_context_raw
            if isinstance(previous_helper_need_more_context_raw, bool)
            else None
        )
        (
            recovery_topup_reason,
            recovery_forced_action,
            recovery_override_reason,
            recovery_decision_reason,
        ) = _decide_adaptive_helper_recovery_topup(
            is_adaptive_condition=is_adaptive_condition,
            helper_budget_exhausted=helper_budget_exhausted,
            adaptive_budget_unlimited=adaptive_budget_unlimited,
            adaptive_budget_topup_calls_used=adaptive_budget_topup_calls_used,
            adaptive_budget_topup_max_calls=normalized_adaptive_budget_topup_max_calls,
            pregate_should_skip=pregate_result.should_skip,
            helper_saturation_guarded=helper_saturation_guarded,
            supports_context_request=supports_context_request,
            supports_strict_json_first=supports_strict_json_first,
            previous_note=previous_note_for_recovery,
            previous_helper_failure_kind=previous_helper_failure_kind,
            previous_progress_reason=previous_progress_reason,
            previous_helper_need_more_context=previous_helper_need_more_context,
            last_helper_need_more_context=last_helper_need_more_context,
            failure_signature_repeat=failure_signature_repeat,
            adaptive_no_progress_streak=adaptive_no_progress_streak,
        )
        if recovery_topup_reason:
            adaptive_budget_topup_calls_used += 1
            adaptive_budget_topup_granted = True
            adaptive_budget_topup_reason = recovery_topup_reason
            helper_budget_exhausted = False
            effective_helper_budget_limit += 1
            adaptive_recovery_forced_action = recovery_forced_action
            adaptive_recovery_override_reason = recovery_override_reason
            adaptive_recovery_decision_reason = recovery_decision_reason

        if is_adaptive_llm_discretion:
            online_policy_features_raw = _adaptive_online_feature_vector(
                attempt=attempt,
                max_attempts=k,
                timed_out=bool(current.timed_out),
                failure_signature_repeat=failure_signature_repeat,
                stagnation_streak=stagnation_streak,
                helper_calls_used=helper_calls_used,
                helper_calls_without_context=helper_calls_without_context,
                adaptive_no_progress_streak=adaptive_no_progress_streak,
                helper_budget_exhausted=helper_budget_exhausted,
                helper_saturation_guarded=helper_saturation_guarded,
                supports_context_request=supports_context_request,
                snapshot_available=snapshot is not None,
                pregate_result=pregate_result,
                effective_helper_budget_limit=effective_helper_budget_limit,
                adaptive_budget_unlimited=adaptive_budget_unlimited,
                adaptive_budget_topup_calls_used=adaptive_budget_topup_calls_used,
                adaptive_budget_topup_max_calls=normalized_adaptive_budget_topup_max_calls,
                last_helper_need_more_context=last_helper_need_more_context,
                last_helper_decision_reason=last_helper_decision_reason,
                last_helper_action_executed=last_helper_action_executed,
                context_compaction_enabled=normalized_context_compaction_mode != "off",
                supports_context_compaction=supports_context_compaction,
                context_compaction_calls_used=context_compaction_calls_used,
                context_compaction_max_calls=effective_context_compaction_max_calls,
                last_context_compaction_requested=last_context_compaction_requested,
                last_context_compaction_invoked=last_context_compaction_invoked,
                last_context_compaction_success=last_context_compaction_success,
                last_context_compaction_ratio=last_context_compaction_ratio,
                last_context_compaction_trigger=last_context_compaction_trigger,
                case_difficulty=case.difficulty,
                case_snapshot_dependency=case.snapshot_dependency,
                case_bug_source=case.bug_source,
                feature_scaling_profile=normalized_adaptive_online_feature_scaling_profile,
            )
            online_policy_features_scaled = scale_feature_mapping(
                online_policy_features_raw,
                profile=normalized_adaptive_online_feature_scaling_profile,
            )

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
                    online_policy_decision = adaptive_online_policy_controller.decide(
                        feature_values=online_policy_features_raw,
                        default_action=helper_action_default,
                        available_actions=helper_available_actions,
                        unavailable_action_reasons=helper_unavailable_action_reasons,
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
                    if (
                        online_policy_decision.applied_action
                        != online_policy_decision.selected_action
                    ):
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

        current_failure_context: dict[str, Any] = {
            "attempt": attempt,
            "failure_signature": failure_signature,
            "stdout": current.stdout,
            "stderr": current.stderr,
            "snapshot": snapshot,
        }
        previous_attempt_diff = _build_structured_previous_attempt_diff(
            previous_failure_context,
            current_failure_context=current_failure_context,
        )
        previous_note = (
            str(previous_failure_context.get("note") or "")
            if isinstance(previous_failure_context, dict)
            else ""
        )
        previous_patch_verifier_rejected = bool(
            isinstance(previous_failure_context, dict)
            and previous_failure_context.get("patch_verifier_rejected")
        )
        previous_patch_verifier_reasons_raw = (
            previous_failure_context.get("patch_verifier_reasons")
            if isinstance(previous_failure_context, dict)
            else []
        )
        previous_patch_verifier_reasons = (
            [str(item) for item in previous_patch_verifier_reasons_raw]
            if isinstance(previous_patch_verifier_reasons_raw, list)
            else []
        )
        previous_patch_verifier_scope = (
            str(previous_failure_context.get("patch_verifier_scope") or "")
            if isinstance(previous_failure_context, dict)
            else ""
        )
        previous_timed_out = bool(
            isinstance(previous_failure_context, dict) and previous_failure_context.get("timed_out")
        )
        previous_timeout_sec_raw = (
            previous_failure_context.get("timeout_sec")
            if isinstance(previous_failure_context, dict)
            else None
        )
        previous_timeout_sec = (
            int(previous_timeout_sec_raw) if isinstance(previous_timeout_sec_raw, int) else None
        )

        file_index_lines = _file_index_lines_for_attempt()
        file_index = _file_index_text_for_attempt()
        attempt_context_protocol = default_context_protocol
        evidence = {
            "case_id": case.case_id,
            "condition": condition.value,
            "attempt": attempt,
            "repro": [sys.executable, *case.python_args],
            "stdout": current.stdout,
            "stderr": current.stderr,
            "failure_signature": failure_signature,
            "previous_failure_signature": previous_failure_signature,
            "failure_signature_repeat": failure_signature_repeat,
            "previous_attempt_diff": previous_attempt_diff,
            "timed_out": bool(current.timed_out),
            "timeout_sec": int(current.timeout_sec)
            if isinstance(current.timeout_sec, int)
            else None,
            "previous_note": previous_note,
            "previous_patch_verifier_rejected": previous_patch_verifier_rejected,
            "previous_patch_verifier_reasons": previous_patch_verifier_reasons,
            "previous_patch_verifier_scope": previous_patch_verifier_scope,
            "previous_timed_out": previous_timed_out,
            "previous_timeout_sec": previous_timeout_sec,
            "file_index": file_index,
            "files": "",
            "snapshot_available": snapshot is not None,
            "snapshot_error": snapshot_error,
        }
        snapshot_for_prompt = None
        stage_a_prompt: str | None = None
        stage_a_data = _empty_stage_a_context(reason="not_used")
        stage_a_meta: dict[str, Any] = {}
        stage_a_request_invoked = False
        stage_a_request_payload: ContextRequestPayload | None = None
        stage_a_raw_response: str | None = None
        stage_a_need_snapshot = False
        stage_a_snapshot_justification = ""
        stage_a_need_compact_context = False
        stage_a_compact_context_reason = ""
        snapshot_lite_context = ""
        adaptive_reveal_applied = False
        adaptive_reveal_trigger = "full_mode"
        adaptive_reveal_triggers: list[str] = ["full_mode"]
        context_action_materialized = False
        helper_local_wall_sec = 0.0
        _stage_a_context_fulfillment_sec = 0.0
        _context_compaction_total_sec = 0.0
        _prompt_assembly_sec = 0.0
        run_stage_a = False
        if is_adaptive_gated:
            run_stage_a = helper_should_consult
        elif is_adaptive_llm_discretion:
            can_run_stage_a = (
                not pregate_result.should_skip
                and not helper_budget_exhausted
                and not helper_saturation_guarded
                and supports_context_request
            )
            if adaptive_force_helper_stage_a:
                run_stage_a = can_run_stage_a
                if not run_stage_a and not adaptive_stagnation_override_reason:
                    adaptive_stagnation_override_reason = "stagnation_override_constraints_blocked"
            elif not can_run_stage_a:
                run_stage_a = False
            else:
                run_stage_a = helper_action_executed in HELPER_STAGE_A_ACTIONS
                if not run_stage_a and helper_action_executed == ACTION_SKIP_HELPER:
                    helper_decision_reason = "online_policy_skip_helper"
        helper_local_started_at = time.perf_counter() if run_stage_a else None
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
                stage_a_need_compact_context = bool(
                    stage_a_request_payload.get("need_compact_context")
                )
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
                _stage_a_context_fulfillment_sec = time.perf_counter() - _context_fulfillment_start
                if is_adaptive_condition:
                    helper_need_more_context = bool(
                        stage_a_request_payload.get("need_more_context")
                    )
                    if is_adaptive_llm_discretion:
                        helper_decision_reason = (
                            "llm_requested_helper"
                            if (helper_need_more_context or stage_a_need_snapshot)
                            else "llm_declined_helper"
                        )
                    elif (
                        is_adaptive_gated and not helper_should_consult and helper_need_more_context
                    ):
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
                    stage_a_data = build_fallback_context_bundle(
                        root=workdir,
                        snapshot=snapshot,
                        stdout=current.stdout,
                        stderr=current.stderr,
                        max_total_chars=stage_a_max_chars,
                        reason=stage_a_failure_reason,
                        file_index_lines=file_index_lines,
                        request_raw=stage_a_raw_response,
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
        if helper_local_started_at is not None:
            helper_local_wall_sec = time.perf_counter() - helper_local_started_at

        if is_adaptive_condition:
            adaptive_context_requested = stage_a_data.requests_count > 0

            # Attempt-1 snapshot restriction: require justification to include snapshot in Stage-B.
            if (
                attempt == 1
                and stage_a_need_snapshot
                and _is_generic_snapshot_justification(stage_a_snapshot_justification)
            ):
                adaptive_snapshot_attempt1_denied = True
                adaptive_snapshot_attempt1_denied_reason = "missing_or_generic_justification"
                stage_a_need_snapshot = False

            adaptive_snapshot_included = bool(stage_a_need_snapshot and snapshot is not None)
            helper_materialized_context = bool(
                adaptive_context_requested or adaptive_snapshot_included
            )
            context_action_materialized = bool(helper_materialized_context)
            helper_retry_count, helper_transport_retry_count = (
                _request_context_retry_counts_from_meta(stage_a_meta)
            )
            if helper_action_executed == ACTION_COMPACT_THEN_PATCH:
                helper_outcome = ADAPTIVE_HELPER_OUTCOME_SEMANTIC_NO_CONTEXT
                helper_failure_kind = ""
            else:
                helper_outcome, helper_failure_kind = _adaptive_helper_outcome_for_attempt(
                    helper_invoked=helper_invoked,
                    helper_materialized_context=helper_materialized_context,
                    helper_decision_reason=helper_decision_reason,
                    stage_a_meta=stage_a_meta,
                )
                helper_reason_overrides = {
                    ADAPTIVE_HELPER_OUTCOME_COMPAT_REASONING_ONLY: ("helper_compat_reasoning_only"),
                    ADAPTIVE_HELPER_OUTCOME_INVALID_PAYLOAD: "helper_invalid_payload",
                    ADAPTIVE_HELPER_OUTCOME_TRANSPORT_FAILED: "helper_transport_failed",
                    ADAPTIVE_HELPER_OUTCOME_UNAVAILABLE_BACKEND: "helper_unavailable_backend",
                    ADAPTIVE_HELPER_OUTCOME_REQUEST_BUDGET_EXCEEDED: (
                        "helper_request_budget_exceeded"
                    ),
                }
                helper_reason_override = helper_reason_overrides.get(helper_outcome)
                if helper_reason_override:
                    stage_a_data = replace(stage_a_data, reason=helper_reason_override)
            attempt_context_protocol = "staged" if helper_materialized_context else "monolithic"
            snapshot_for_prompt = stage_b_snapshot if adaptive_snapshot_included else None
            if (
                is_adaptive_llm_discretion
                and not helper_materialized_context
                and helper_outcome == ADAPTIVE_HELPER_OUTCOME_SEMANTIC_NO_CONTEXT
                and helper_decision_reason == "llm_requested_helper"
            ):
                helper_decision_reason = (
                    "llm_snapshot_request_denied_attempt1"
                    if adaptive_snapshot_attempt1_denied
                    else "llm_declined_helper"
                )
                helper_outcome = ADAPTIVE_HELPER_OUTCOME_SEMANTIC_NO_CONTEXT
            if helper_invoked:
                if helper_outcome not in {
                    ADAPTIVE_HELPER_OUTCOME_COMPAT_REASONING_ONLY,
                    ADAPTIVE_HELPER_OUTCOME_TRANSPORT_FAILED,
                    ADAPTIVE_HELPER_OUTCOME_UNAVAILABLE_BACKEND,
                }:
                    helper_calls_used += 1
                if helper_outcome == ADAPTIVE_HELPER_OUTCOME_SEMANTIC_NO_CONTEXT:
                    helper_calls_without_context += 1
            if isinstance(helper_need_more_context, bool):
                last_helper_need_more_context = helper_need_more_context
        else:
            attempt_context_protocol = "monolithic"
            snapshot_for_prompt = None
        if is_adaptive_condition:
            last_helper_decision_reason = str(helper_decision_reason)
            last_helper_action_executed = str(helper_action_executed or ACTION_SKIP_HELPER)
            if helper_outcome == ADAPTIVE_HELPER_OUTCOME_NOT_INVOKED:
                helper_outcome, helper_failure_kind = _adaptive_helper_outcome_for_attempt(
                    helper_invoked=helper_invoked,
                    helper_materialized_context=helper_materialized_context,
                    helper_decision_reason=helper_decision_reason,
                    stage_a_meta=stage_a_meta,
                )

        project_files_context = (
            _monolithic_context_for_attempt()
            if include_files_context and attempt_context_protocol == "monolithic"
            else ""
        )
        evidence["files"] = project_files_context
        evidence["snapshot_available"] = stage_b_snapshot is not None

        stage_a_default_llm_calls = 1 if (is_openai_patcher and stage_a_request_invoked) else 0
        (
            helper_local_llm_calls,
            helper_local_tokens_prompt,
            helper_local_tokens_completion,
            helper_local_tokens_total,
        ) = _aggregate_attempt_llm_usage(
            stage_a_meta=stage_a_meta if stage_a_request_invoked else None,
            patch_meta=None,
            stage_a_default_llm_calls=stage_a_default_llm_calls if stage_a_request_invoked else 0,
            patch_default_llm_calls=0,
        )
        helper_local_accounting = _adaptive_policy_local_accounting(
            llm_call_count=helper_local_llm_calls,
            tokens_prompt=helper_local_tokens_prompt,
            tokens_completion=helper_local_tokens_completion,
            tokens_total=helper_local_tokens_total,
            wall_sec=helper_local_wall_sec if stage_a_request_invoked else 0.0,
        )

        structured_seed_data = _empty_stage_a_context()
        structured_seed_active = is_adaptive_condition and helper_materialized_context
        if structured_seed_active:
            structured_seed_data = build_fallback_context_bundle(
                root=workdir,
                snapshot=snapshot_for_prompt,
                stdout=current.stdout,
                stderr=current.stderr,
                max_total_chars=stage_a_max_chars,
                reason="adaptive_helper_seed_bundle",
                file_index_lines=file_index_lines,
            )

        previous_seen_evidence_ids = set(seen_evidence_ids)
        combined_evidence_items = _merge_evidence_items(
            structured_seed_data.evidence_items if structured_seed_active else [],
            stage_a_data.evidence_items,
        )
        combined_evidence_ids = _evidence_ids_from_items(combined_evidence_items)
        new_evidence_items = [
            item
            for item in combined_evidence_items
            if str(item.get("id") or "") not in previous_seen_evidence_ids
        ]
        new_evidence_ids = _evidence_ids_from_items(new_evidence_items)
        reused_evidence_ids = [
            evidence_id
            for evidence_id in combined_evidence_ids
            if evidence_id in previous_seen_evidence_ids
        ]
        seen_evidence_ids.update(combined_evidence_ids)

        attempt_retry_prompt_mode = base_retry_prompt_mode
        is_delta_retry = attempt_retry_prompt_mode == "delta" and attempt > 1
        evidence_items_for_prompt = (
            new_evidence_items if is_delta_retry else combined_evidence_items
        )
        baseline_hybrid_guidance_mode = ""
        controller_evidence_items = list(combined_evidence_items)
        controller_new_evidence_ids = list(new_evidence_ids)
        controller_reused_evidence_ids = list(reused_evidence_ids)
        bookkeeping_new_evidence_ids = list(new_evidence_ids)
        bookkeeping_reused_evidence_ids = list(reused_evidence_ids)
        bookkeeping_evidence_items_for_prompt = list(evidence_items_for_prompt)
        if is_baseline_patcher and is_adaptive_condition:
            baseline_hybrid_guidance_mode = _baseline_hybrid_guidance_mode_for_patcher(
                patcher_for_attempt
            )
            (
                controller_evidence_items,
                controller_new_evidence_ids,
                controller_reused_evidence_ids,
            ) = _controller_runtime_evidence_inputs(
                baseline_hybrid_guidance_mode=baseline_hybrid_guidance_mode,
                seen_evidence_ids=previous_seen_evidence_ids,
                workdir=workdir,
                snapshot=snapshot,
                stdout=current.stdout,
                stderr=current.stderr,
                max_total_chars=stage_a_max_chars,
                file_index_lines=file_index_lines,
                combined_evidence_items=combined_evidence_items,
                new_evidence_ids=new_evidence_ids,
                reused_evidence_ids=reused_evidence_ids,
            )
            if baseline_hybrid_guidance_mode == HYBRID_GUIDANCE_MODE_FIXED_V1:
                bookkeeping_new_evidence_ids = list(controller_new_evidence_ids)
                bookkeeping_reused_evidence_ids = list(controller_reused_evidence_ids)
                controller_new_id_set = set(controller_new_evidence_ids)
                bookkeeping_evidence_items_for_prompt = (
                    [
                        item
                        for item in controller_evidence_items
                        if str(item.get("id") or "") in controller_new_id_set
                    ]
                    if is_delta_retry
                    else list(controller_evidence_items)
                )
                seen_evidence_ids.update(_evidence_ids_from_items(controller_evidence_items))
        stage_a_context_full = _render_evidence_items(combined_evidence_items)
        stage_a_context_text = (
            _render_evidence_items(new_evidence_items) if is_delta_retry else stage_a_context_full
        )
        stage_a_evidence_index_text = _render_evidence_index(evidence_items_for_prompt)
        retry_delta_chars = len(stage_a_context_text) if is_delta_retry else 0
        adaptive_progress_made = False
        adaptive_progress_reason = "not_adaptive_condition"
        adaptive_progress_score = 0.0
        adaptive_no_progress_streak_current = 0
        adaptive_progress_stop_candidate = False
        adaptive_progress_stop_reason = "not_adaptive_condition"

        retry_packet = ""
        if structured_seed_active or is_delta_retry:
            retry_packet = _render_retry_packet(
                failure_signature=failure_signature,
                previous_failure_signature=evidence.get("previous_failure_signature"),
                failure_signature_repeat=bool(evidence.get("failure_signature_repeat")),
                previous_attempt_diff=previous_attempt_diff,
                timed_out=bool(evidence.get("timed_out")),
                timeout_sec=(
                    int(evidence.get("timeout_sec"))
                    if isinstance(evidence.get("timeout_sec"), int)
                    else None
                ),
                previous_note=str(evidence.get("previous_note") or ""),
                previous_patch_verifier_rejected=bool(
                    evidence.get("previous_patch_verifier_rejected")
                ),
                previous_patch_verifier_reasons=(
                    list(evidence.get("previous_patch_verifier_reasons"))
                    if isinstance(evidence.get("previous_patch_verifier_reasons"), list)
                    else []
                ),
                previous_patch_verifier_scope=str(
                    evidence.get("previous_patch_verifier_scope") or ""
                ),
                toon_encoding_active=toon_encoding_requested,
            )
        evidence_fingerprint = _compute_evidence_fingerprint(
            failure_signature=failure_signature,
            stage_a_blocks=stage_a_data.blocks,
            stage_a_context=stage_a_context_full,
            structured_seed_blocks=structured_seed_data.blocks if structured_seed_active else [],
        )
        current_failure_context["evidence_fingerprint"] = evidence_fingerprint

        (
            retry_history_text,
            retry_history_chars,
            retry_history_turns_included,
            retry_history_truncated,
        ) = (
            _build_retry_history_section(
                prior_turns=retry_history_turns,
                max_chars=effective_retry_history_max_chars,
            )
            if effective_retry_context_mode == "history" and attempt > 1
            else ("", 0, 0, False)
        )
        attempt_prompt_budget = _adaptive_prompt_budget_for_attempt(
            base_budget=prompt_budget,
            helper_materialized_context=(is_adaptive_condition and helper_materialized_context),
            profile=normalized_adaptive_helper_budget_profile,
        )
        openai_max_tokens_for_attempt = 0
        openai_preflight_prompt_cap_chars: int | None = None
        openai_preflight_prompt_cap_tokens: int | None = None
        openai_preflight_trimmed = False
        if is_openai_patcher and normalized_openai_context_window_tokens > 0:
            openai_max_tokens_for_attempt = int(getattr(patcher_for_attempt, "max_tokens", 0) or 0)
            openai_preflight_prompt_cap_chars = _openai_preflight_total_char_cap(
                context_window_tokens=normalized_openai_context_window_tokens,
                max_tokens=openai_max_tokens_for_attempt,
            )
            openai_preflight_prompt_cap_tokens = _openai_preflight_prompt_cap_tokens(
                context_window_tokens=normalized_openai_context_window_tokens,
                max_tokens=openai_max_tokens_for_attempt,
            )
            previous_total_chars = int(attempt_prompt_budget.get("total_chars", 0))
            next_total_chars = (
                openai_preflight_prompt_cap_chars
                if previous_total_chars <= 0
                else min(previous_total_chars, openai_preflight_prompt_cap_chars)
            )
            openai_preflight_trimmed = (previous_total_chars <= 0 and next_total_chars > 0) or (
                previous_total_chars > 0 and next_total_chars < previous_total_chars
            )
            attempt_prompt_budget["total_chars"] = max(0, int(next_total_chars))
        include_stage_a_evidence_index = bool(
            stage_a_evidence_index_text and (is_delta_retry or not stage_a_context_text.strip())
        )
        stage_a_index_reason = (
            "delta_id_map"
            if (include_stage_a_evidence_index and is_delta_retry)
            else ("context_missing" if include_stage_a_evidence_index else "redundant_skipped")
        )
        reveal_policy = _resolve_stage_b_reveal_policy(
            mode=normalized_adaptive_reveal_mode,
            is_adaptive_condition=is_adaptive_condition,
            snapshot_available=snapshot_for_prompt is not None or stage_b_snapshot is not None,
            stage_a_context=stage_a_context_text,
            stage_a_need_snapshot=stage_a_need_snapshot,
            failure_signature_repeat=failure_signature_repeat,
            no_progress_streak=adaptive_no_progress_streak,
        )
        adaptive_reveal_applied = bool(reveal_policy["applied"])
        adaptive_reveal_trigger = str(reveal_policy["trigger"])
        adaptive_reveal_triggers = [
            str(trigger) for trigger in reveal_policy.get("triggers", []) if str(trigger)
        ]
        if (
            bool(reveal_policy["snapshot_lite_needed"])
            and not snapshot_lite_context
            and stage_b_snapshot is not None
        ):
            snapshot_lite_context = _format_snapshot_index_section(stage_b_snapshot)

        include_snapshot_in_prompt = bool(reveal_policy["include_snapshot"])
        stage_b_snapshot_for_prompt = stage_b_snapshot if include_snapshot_in_prompt else None
        stage_b_snapshot_prompt_text = (
            snapshot_prompt_text if stage_b_snapshot_for_prompt is not None else ""
        )
        include_pytest_output_in_prompt = True
        include_stage_a_context_in_prompt = bool(reveal_policy["include_stage_a_context"])
        include_project_files_in_prompt = attempt_context_protocol == "monolithic"
        compacted_context_for_prompt = ""
        context_compaction_requested = bool(stage_a_need_compact_context)
        context_compaction_request_reason = str(stage_a_compact_context_reason or "")
        policy_compact_then_patch_requested = bool(
            is_adaptive_llm_discretion and helper_action_executed == ACTION_COMPACT_THEN_PATCH
        )
        context_compaction_invoked = False
        context_compaction_success = False
        context_compaction_trigger = "not_needed"
        context_compaction_error = ""
        context_compaction_auto_overflow_triggered = False
        context_compaction_replaced_sections: list[str] = []
        context_compaction_source_sections: list[str] = []
        context_compaction_source_chars: dict[str, int] = {}
        context_compaction_input_chars = 0
        context_compaction_output_chars = 0
        context_compaction_prompt_text: str | None = None
        context_compaction_prompt_chars = 0
        context_compaction_meta: dict[str, Any] = {}

        def _build_stage_b_sections_for_attempt(
            *,
            include_stage_a_evidence_index_value: bool,
            include_project_files_value: bool,
            compacted_context_value: str,
            include_snapshot_value: bool,
            include_pytest_output_value: bool,
            include_stage_a_context_value: bool,
            retry_repair_plan_value: str = "",
            planning_mode: bool = False,
            evidence_value: dict[str, Any] = evidence,
            snapshot_value: dict[str, Any] | None = stage_b_snapshot_for_prompt,
            snapshot_prompt_text_value: str = stage_b_snapshot_prompt_text,
            stage_a_context_text_value: str = stage_a_context_text,
            stage_a_evidence_index_text_value: str = stage_a_evidence_index_text,
            retry_packet_value: str = retry_packet,
            retry_prompt_mode_value: str = attempt_retry_prompt_mode,
            retry_history_value: str = retry_history_text,
            attempt_value: int = attempt,
            known_evidence_refs_value: list[str] = reused_evidence_ids,
            new_evidence_ids_value: list[str] = new_evidence_ids,
            snapshot_lite_context_value: str = snapshot_lite_context,
            prompt_retry_reflection_value: bool = normalized_prompt_retry_reflection,
            prompt_adaptive_stage_b_cot_value: bool = effective_prompt_adaptive_stage_b_cot,
            prompt_repair_method_scaffold_value: bool = normalized_prompt_repair_method_scaffold,
            enable_tep_value: bool = False,
            stagnation_streak_value: int = 0,
        ) -> list[dict[str, Any]]:
            effective_snapshot_value = snapshot_value if include_snapshot_value else None
            effective_snapshot_prompt_text_value = (
                snapshot_prompt_text_value if effective_snapshot_value is not None else ""
            )
            return build_stage_b_prompt_sections(
                prompt_inputs=StageBPromptInputs(
                    evidence=evidence_value,
                    snapshot=effective_snapshot_value,
                    snapshot_prompt_text=effective_snapshot_prompt_text_value,
                    stage_a_context=stage_a_context_text_value,
                    stage_a_evidence_index=stage_a_evidence_index_text_value,
                    compacted_context=compacted_context_value,
                    include_snapshot=include_snapshot_value,
                    include_pytest_output=include_pytest_output_value,
                    include_stage_a_context=include_stage_a_context_value,
                    include_project_files=include_project_files_value,
                    retry_packet=retry_packet_value,
                    retry_history=retry_history_value,
                    retry_prompt_mode=retry_prompt_mode_value,
                    attempt=attempt_value,
                    known_evidence_refs=known_evidence_refs_value,
                    new_evidence_ids=new_evidence_ids_value,
                    snapshot_lite_context=snapshot_lite_context_value,
                    prompt_retry_reflection=prompt_retry_reflection_value,
                    prompt_adaptive_stage_b_cot=prompt_adaptive_stage_b_cot_value,
                    prompt_repair_method_scaffold=prompt_repair_method_scaffold_value,
                    prompt_order_mode=normalized_prompt_order_mode,
                    enable_tep=enable_tep_value,
                    stagnation_streak=stagnation_streak_value,
                    tep_patch_actor_overrides=(
                        adaptive_tep_controller.patch_actor_overrides()
                        if full_tep_enabled
                        else None
                    ),
                    tep_planner_actor_overrides=(
                        adaptive_tep_controller.planner_actor_overrides()
                        if full_tep_enabled
                        else None
                    ),
                ),
                include_stage_a_evidence_index=include_stage_a_evidence_index_value,
                retry_repair_plan=retry_repair_plan_value,
                planning_mode=planning_mode,
            )

        attempt_shared_propose_budget_sec = 0.0
        attempt_shared_propose_deadline_monotonic: float | None = None
        if is_openai_patcher:
            raw_attempt_budget_sec = getattr(patcher_for_attempt, "max_propose_sec", 0.0)
            try:
                attempt_shared_propose_budget_sec = max(0.0, float(raw_attempt_budget_sec))
            except (TypeError, ValueError):
                attempt_shared_propose_budget_sec = 0.0
            if attempt_shared_propose_budget_sec > 0:
                attempt_shared_propose_deadline_monotonic = (
                    time.monotonic() + attempt_shared_propose_budget_sec
                )
        plan_patch_enabled = (
            normalized_prompt_retry_plan_then_patch
            or condition is Condition.COT_REPAIR
            or cot_baseline_semantics
        ) and hasattr(patcher_for_attempt, "plan_patch")
        if full_tep_enabled and adaptive_tep_controller is not None:
            adaptive_tep_controller.prime_stage_b_defaults(
                planning_mode=False,
                sections=_build_stage_b_sections_for_attempt(
                    include_stage_a_evidence_index_value=include_stage_a_evidence_index,
                    include_project_files_value=include_project_files_in_prompt,
                    compacted_context_value=compacted_context_for_prompt,
                    include_snapshot_value=include_snapshot_in_prompt,
                    include_pytest_output_value=include_pytest_output_in_prompt,
                    include_stage_a_context_value=include_stage_a_context_in_prompt,
                    enable_tep_value=enable_tep and tep_nudge_escalation,
                    stagnation_streak_value=stagnation_streak,
                ),
            )
            if plan_patch_enabled:
                adaptive_tep_controller.prime_stage_b_defaults(
                    planning_mode=True,
                    sections=_build_stage_b_sections_for_attempt(
                        include_stage_a_evidence_index_value=include_stage_a_evidence_index,
                        include_project_files_value=include_project_files_in_prompt,
                        compacted_context_value=compacted_context_for_prompt,
                        include_snapshot_value=include_snapshot_in_prompt,
                        include_pytest_output_value=include_pytest_output_in_prompt,
                        include_stage_a_context_value=include_stage_a_context_in_prompt,
                        enable_tep_value=enable_tep and tep_nudge_escalation,
                        stagnation_streak_value=stagnation_streak,
                        planning_mode=True,
                    ),
                )

        def _run_retry_plan_patch(
            retry_plan_prompt_text: str,
            *,
            patcher_for_attempt_value=patcher_for_attempt,
            attempt_dir_value=attempt_dir,
            workdir_value=workdir,
            snapshot_full_text_value=snapshot_full_text,
            snapshot_artifact_text_value=snapshot_artifact_text,
            attempt_deadline_monotonic_value=attempt_shared_propose_deadline_monotonic,
            attempt_value=attempt,
            attempt_context_protocol_value=attempt_context_protocol,
            current_value=current,
            cumulative_runtime_sec_value=cumulative_runtime_sec,
        ) -> tuple[str | None, str | None, dict[str, Any]]:
            if full_tep_enabled and adaptive_tep_controller is not None:
                result = _run_openai_operation_with_transport_recovery(
                    patcher_for_attempt=patcher_for_attempt_value,
                    operation_name="plan_patch",
                    invoke=lambda: patcher_for_attempt_value.generate_text(
                        system_prompt=adaptive_tep_controller.planner_system_prompt(),
                        prompt=retry_plan_prompt_text,
                        context_dir=attempt_dir_value,
                        cwd=workdir_value,
                        temperature=adaptive_tep_controller.node("retry_planner").temperature,
                        max_tokens=768,
                        operation_name="tep_retry_planner",
                    ),
                    classify_failure_kind=_transport_recovery_failure_kind_from_meta,
                    attempt=attempt_value,
                    attempt_metadata=None,
                    attempt_context_protocol=attempt_context_protocol_value,
                    before=current_value,
                    cumulative_runtime_sec_value=cumulative_runtime_sec_value,
                )
                plan_meta = (
                    dict(result.meta)
                    if hasattr(result, "meta") and isinstance(result.meta, dict)
                    else {}
                )
                raw_response = getattr(result, "raw_response", None)
                text = getattr(result, "text", None)
                if isinstance(raw_response, str):
                    plan_meta.setdefault("raw_response", raw_response)
                return (
                    text if isinstance(text, str) and text.strip() else None,
                    raw_response if isinstance(raw_response, str) else None,
                    plan_meta,
                )
            result = _run_openai_operation_with_transport_recovery(
                patcher_for_attempt=patcher_for_attempt_value,
                operation_name="plan_patch",
                invoke=lambda: _plan_patch_with_snapshot_context(
                    patcher=patcher_for_attempt_value,
                    prompt=retry_plan_prompt_text,
                    context_dir=attempt_dir_value,
                    cwd=workdir_value,
                    snapshot_full_text=snapshot_full_text_value,
                    snapshot_artifact_text=snapshot_artifact_text_value,
                    attempt_deadline_monotonic=attempt_deadline_monotonic_value,
                ),
                classify_failure_kind=_transport_recovery_failure_kind_from_meta,
                attempt=attempt_value,
                attempt_metadata=None,
                attempt_context_protocol=attempt_context_protocol_value,
                before=current_value,
                cumulative_runtime_sec_value=cumulative_runtime_sec_value,
            )
            plan_meta = (
                dict(result.meta)
                if hasattr(result, "meta") and isinstance(result.meta, dict)
                else {}
            )
            raw_response = getattr(result, "raw_response", None)
            if isinstance(raw_response, str):
                plan_meta.setdefault("raw_response", raw_response)
            return (
                result.plan_text if hasattr(result, "plan_text") else None,
                raw_response if isinstance(raw_response, str) else None,
                plan_meta,
            )

        _emit_progress(
            "prompt_build",
            _progress_payload(
                owner="harness",
                phase_label="harness: prompt build",
                note="assembling prompt",
                attempt=int(attempt),
                max_attempts=int(k),
            ),
        )
        stage_b_prompt_inputs = StageBPromptInputs(
            evidence=evidence,
            snapshot=stage_b_snapshot_for_prompt,
            snapshot_prompt_text=stage_b_snapshot_prompt_text,
            stage_a_context=stage_a_context_text,
            stage_a_evidence_index=stage_a_evidence_index_text,
            compacted_context=compacted_context_for_prompt,
            include_snapshot=include_snapshot_in_prompt,
            include_pytest_output=include_pytest_output_in_prompt,
            include_stage_a_context=include_stage_a_context_in_prompt,
            include_project_files=include_project_files_in_prompt,
            retry_packet=retry_packet,
            retry_history=retry_history_text,
            retry_prompt_mode=attempt_retry_prompt_mode,
            attempt=attempt,
            known_evidence_refs=reused_evidence_ids,
            new_evidence_ids=new_evidence_ids,
            snapshot_lite_context=snapshot_lite_context,
            prompt_retry_reflection=normalized_prompt_retry_reflection,
            prompt_adaptive_stage_b_cot=effective_prompt_adaptive_stage_b_cot,
            prompt_repair_method_scaffold=normalized_prompt_repair_method_scaffold,
            prompt_order_mode=normalized_prompt_order_mode,
            enable_tep=enable_tep and tep_nudge_escalation,
            stagnation_streak=stagnation_streak,
            tep_patch_actor_overrides=(
                adaptive_tep_controller.patch_actor_overrides() if full_tep_enabled else None
            ),
            tep_planner_actor_overrides=(
                adaptive_tep_controller.planner_actor_overrides() if full_tep_enabled else None
            ),
        )
        plan_on_first_attempt = bool(
            plan_patch_enabled and (condition is Condition.COT_REPAIR or cot_baseline_semantics)
        )
        native_prompt_result = assemble_native_openai_attempt_prompt(
            request=NativeOpenAIAttemptPromptRequest(
                prompt_inputs=stage_b_prompt_inputs,
                attempt_prompt_budget=attempt_prompt_budget,
                is_delta_retry=is_delta_retry,
                include_stage_a_evidence_index=include_stage_a_evidence_index,
                stage_a_index_reason=stage_a_index_reason,
                project_files_context=project_files_context,
                current_stdout=current.stdout,
                current_stderr=current.stderr,
                context_compaction_requested=context_compaction_requested,
                context_compaction_request_reason=context_compaction_request_reason,
                policy_compact_then_patch_requested=policy_compact_then_patch_requested,
                context_compaction_mode=normalized_context_compaction_mode,
                context_compaction_scope=set(normalized_context_compaction_scope),
                supports_context_compaction=supports_context_compaction,
                context_compaction_calls_used=context_compaction_calls_used,
                context_compaction_max_calls=effective_context_compaction_max_calls,
                context_compaction_max_input_chars=normalized_context_compaction_max_input_chars,
                context_compaction_target_chars=normalized_context_compaction_target_chars,
                context_compaction_fn=context_compaction_fn,
                patcher_for_attempt=patcher_for_attempt,
                run_openai_operation_with_transport_recovery=_run_openai_operation_with_transport_recovery,
                classify_transport_recovery_failure_kind=_transport_recovery_failure_kind_from_meta,
                attempt_context_protocol=attempt_context_protocol,
                before=current,
                cumulative_runtime_sec=cumulative_runtime_sec,
                attempt_dir=attempt_dir,
                workdir=workdir,
                retry_plan_enabled=plan_patch_enabled,
                plan_on_first_attempt=plan_on_first_attempt,
                tep_equilibrium_check=tep_equilibrium_check,
                tep_equilibrium_max_iterations=tep_equilibrium_max_iterations,
                tep_equilibrium_similarity_threshold=tep_equilibrium_similarity_threshold,
                context_compaction_actor_overrides=(
                    adaptive_tep_controller.compaction_actor_overrides()
                    if full_tep_enabled
                    else None
                ),
                run_plan_patch=_run_retry_plan_patch,
            )
        )
        plan_patch_mode = (
            "cot_condition"
            if condition is Condition.COT_REPAIR
            else "cot_baseline"
            if cot_baseline_semantics
            else "retry_ablation"
            if normalized_prompt_retry_plan_then_patch
            else "off"
        )
        stage_b_prompt_inputs = native_prompt_result.prompt_inputs
        prompt = native_prompt_result.prompt
        prompt_budget_result = native_prompt_result.prompt_budget_result
        include_stage_a_evidence_index = native_prompt_result.include_stage_a_evidence_index
        stage_a_index_reason = native_prompt_result.stage_a_index_reason

        include_snapshot_in_prompt = stage_b_prompt_inputs.include_snapshot
        include_pytest_output_in_prompt = stage_b_prompt_inputs.include_pytest_output
        include_stage_a_context_in_prompt = stage_b_prompt_inputs.include_stage_a_context
        include_project_files_in_prompt = stage_b_prompt_inputs.include_project_files
        compacted_context_for_prompt = stage_b_prompt_inputs.compacted_context

        context_compaction_requested = native_prompt_result.context_compaction.requested
        context_compaction_request_reason = native_prompt_result.context_compaction.request_reason
        context_compaction_invoked = native_prompt_result.context_compaction.invoked
        context_compaction_success = native_prompt_result.context_compaction.success
        context_compaction_trigger = native_prompt_result.context_compaction.trigger
        context_compaction_error = native_prompt_result.context_compaction.error
        context_compaction_auto_overflow_triggered = (
            native_prompt_result.context_compaction.auto_overflow_triggered
        )
        context_compaction_replaced_sections = list(
            native_prompt_result.context_compaction.replaced_sections
        )
        context_compaction_source_sections = list(
            native_prompt_result.context_compaction.source_sections
        )
        context_compaction_source_chars = dict(native_prompt_result.context_compaction.source_chars)
        context_compaction_input_chars = native_prompt_result.context_compaction.input_chars
        context_compaction_output_chars = native_prompt_result.context_compaction.output_chars
        context_compaction_prompt_text = native_prompt_result.context_compaction.prompt_text
        context_compaction_prompt_chars = len(context_compaction_prompt_text or "")
        context_compaction_meta = dict(native_prompt_result.context_compaction.meta)
        context_compaction_calls_used = native_prompt_result.context_compaction.calls_used

        retry_plan_prompt_chars = 0
        retry_plan_text = native_prompt_result.retry_plan.retry_plan_text
        retry_plan_raw_response = native_prompt_result.retry_plan.retry_plan_raw_response
        retry_plan_prompt_text = native_prompt_result.retry_plan.retry_plan_prompt_text
        retry_plan_meta = dict(native_prompt_result.retry_plan.retry_plan_meta)
        retry_plan_duration_sec = native_prompt_result.retry_plan.retry_plan_duration_sec
        retry_plan_invoked = native_prompt_result.retry_plan.retry_plan_invoked
        retry_plan_succeeded = native_prompt_result.retry_plan.retry_plan_succeeded
        retry_plan_prompt_injected = native_prompt_result.retry_plan.retry_plan_prompt_injected
        retry_plan_error = native_prompt_result.retry_plan.retry_plan_error
        retry_plan_tep_equilibrium_iterations = (
            native_prompt_result.retry_plan.tep_equilibrium_iterations
        )
        retry_plan_tep_equilibrium_converged = (
            native_prompt_result.retry_plan.tep_equilibrium_converged
        )

        _prompt_assembly_sec = native_prompt_result.prompt_assembly_sec
        _context_compaction_total_sec = native_prompt_result.context_compaction_total_sec

        adaptive_snapshot_lite_included = bool("Compact snapshot index:\n" in prompt)
        prompt_token_estimate = estimate_text_token_count(
            prompt,
            model=_token_estimate_model_name(patcher_for_attempt),
        )
        prompt_tokens_estimate_pre_call = (
            int(prompt_token_estimate.get("token_count") or 0)
            if prompt_token_estimate.get("available")
            else None
        )
        openai_preflight_prompt_exceeds_token_cap_estimate = bool(
            isinstance(prompt_tokens_estimate_pre_call, int)
            and isinstance(openai_preflight_prompt_cap_tokens, int)
            and prompt_tokens_estimate_pre_call > openai_preflight_prompt_cap_tokens
        )
        if helper_action_executed in {
            ACTION_SNAPSHOT_LITE,
            ACTION_FRAME_LOCALS_LITE,
            ACTION_SNAPSHOT_FULL_LITE,
        }:
            context_action_materialized = adaptive_snapshot_lite_included
            helper_outcome = (
                ADAPTIVE_HELPER_OUTCOME_MATERIALIZED
                if adaptive_snapshot_lite_included
                else ADAPTIVE_HELPER_OUTCOME_SEMANTIC_NO_CONTEXT
            )
            helper_failure_kind = ""

        last_context_compaction_requested = context_compaction_requested
        last_context_compaction_invoked = context_compaction_invoked
        if context_compaction_invoked:
            last_context_compaction_success = context_compaction_success
            last_context_compaction_trigger = context_compaction_trigger
            if context_compaction_input_chars > 0 and context_compaction_output_chars > 0:
                last_context_compaction_ratio = min(
                    1.0,
                    context_compaction_output_chars / float(max(1, context_compaction_input_chars)),
                )
            else:
                last_context_compaction_ratio = 0.0
        elif context_compaction_requested:
            last_context_compaction_success = False
            last_context_compaction_trigger = context_compaction_trigger
        if policy_compact_then_patch_requested:
            context_action_materialized = bool(context_compaction_success)
            helper_outcome = (
                ADAPTIVE_HELPER_OUTCOME_MATERIALIZED
                if context_compaction_success
                else ADAPTIVE_HELPER_OUTCOME_SEMANTIC_NO_CONTEXT
            )
            helper_failure_kind = ""
            helper_local_accounting = _adaptive_policy_local_accounting(
                llm_call_count=_llm_call_count_from_meta(
                    context_compaction_meta if context_compaction_invoked else None,
                    default=1 if (is_openai_patcher and context_compaction_invoked) else 0,
                ),
                tokens_prompt=_usage_token_counts_from_patch_meta(
                    context_compaction_meta if context_compaction_invoked else None
                )[0],
                tokens_completion=_usage_token_counts_from_patch_meta(
                    context_compaction_meta if context_compaction_invoked else None
                )[1],
                tokens_total=_usage_token_counts_from_patch_meta(
                    context_compaction_meta if context_compaction_invoked else None
                )[2],
                wall_sec=float(context_compaction_meta.get("compact_context_duration_sec") or 0.0)
                if context_compaction_invoked
                else 0.0,
            )
        elif helper_action_executed in {
            ACTION_SKIP_HELPER,
            ACTION_SNAPSHOT_LITE,
            ACTION_FRAME_LOCALS_LITE,
            ACTION_SNAPSHOT_FULL_LITE,
        }:
            helper_local_accounting = _adaptive_policy_local_accounting()
        retry_history_budget_chars = int(attempt_prompt_budget.get("retry_history_chars", 0))
        retry_history_budget_applied = bool(
            retry_history_text
            and (
                retry_history_text not in prompt
                or (
                    retry_history_budget_chars > 0
                    and retry_history_chars > retry_history_budget_chars
                )
            )
        )
        stage_a_prompt_chars = _prompt_chars_for_llm_calls(
            stage_a_prompt,
            llm_call_count=_llm_call_count_from_meta(
                stage_a_meta if stage_a_request_invoked else None,
                default=stage_a_default_llm_calls if stage_a_request_invoked else 0,
            ),
        )
        context_compaction_prompt_chars = _prompt_chars_for_llm_calls(
            context_compaction_prompt_text,
            llm_call_count=_llm_call_count_from_meta(
                context_compaction_meta if context_compaction_invoked else None,
                default=1 if (is_openai_patcher and context_compaction_invoked) else 0,
            ),
        )
        retry_plan_prompt_chars = _prompt_chars_for_llm_calls(
            retry_plan_prompt_text,
            llm_call_count=_llm_call_count_from_meta(
                retry_plan_meta if retry_plan_invoked else None,
                default=1 if (is_openai_patcher and retry_plan_invoked) else 0,
            ),
        )
        retry_plan_llm_calls = 0
        retry_plan_tokens_prompt = 0
        retry_plan_tokens_completion = 0
        retry_plan_tokens_total = 0
        initial_prompt_chars_total = (
            stage_a_prompt_chars + context_compaction_prompt_chars + retry_plan_prompt_chars
        )
        initial_completion_chars_total = (
            _nonnegative_int(stage_a_data.chars_returned, default=0)
            + _nonnegative_int(context_compaction_output_chars, default=0)
            + max(0, len(retry_plan_text))
        )
        attempt_metadata = _materialize_attempt_metadata_request(
            request=build_attempt_metadata_request(
                attempt_dir=attempt_dir,
                attempt=attempt,
                max_attempts=k,
                evidence=evidence,
                prompt=prompt,
                record_prompt=record_prompt,
                record_evidence=record_evidence,
                context_protocol=attempt_context_protocol,
                llm_call_count=0,
                prompt_chars_total=initial_prompt_chars_total,
                completion_chars_total=initial_completion_chars_total,
                tokens_prompt=0,
                tokens_completion=0,
                tokens_total=0,
                evidence_ids_new_count=len(bookkeeping_new_evidence_ids),
                evidence_ids_reused_count=len(bookkeeping_reused_evidence_ids),
                evidence_ids_sent_count=len(
                    _evidence_ids_from_items(bookkeeping_evidence_items_for_prompt)
                ),
                stage_a=StageAAttemptMetadata(
                    stage_a_prompt=stage_a_prompt,
                    stage_a_raw_response=stage_a_data.request_raw,
                    stage_a_request_payload=stage_a_data.request_payload,
                    stage_a_prompt_chars=stage_a_prompt_chars,
                    stage_a_requests_count=stage_a_data.requests_count,
                    stage_a_chars_returned=stage_a_data.chars_returned,
                    stage_a_request_success=stage_a_data.request_success,
                    stage_a_index_included=bool(include_stage_a_evidence_index),
                    stage_a_index_reason=stage_a_index_reason,
                ),
                retry=RetryAttemptMetadata(
                    retry_prompt_mode=attempt_retry_prompt_mode,
                    retry_context_mode=effective_retry_context_mode,
                    retry_history_chars=retry_history_chars,
                    retry_history_truncated=retry_history_truncated,
                    retry_history_turns_included=retry_history_turns_included,
                    retry_history_budget_chars=retry_history_budget_chars,
                    retry_history_budget_applied=retry_history_budget_applied,
                    retry_delta_prompt=is_delta_retry,
                    retry_delta_chars=retry_delta_chars,
                ),
                adaptive=AdaptiveAttemptMetadata(
                    adaptive_mode=helper_decision_mode,
                    adaptive_policy=adaptive_policy_label,
                    adaptive_signature_repeat=failure_signature_repeat,
                    adaptive_stagnation_streak=stagnation_streak,
                    adaptive_helper_invoked=helper_invoked,
                    adaptive_helper_action_default=helper_action_default,
                    adaptive_helper_action_selected=helper_action_selected,
                    adaptive_helper_action_executed=helper_action_executed,
                    adaptive_helper_action_fallback_reason=helper_action_fallback_reason,
                    adaptive_helper_available_actions=list(helper_available_actions),
                    adaptive_helper_unavailable_action_reasons=dict(
                        helper_unavailable_action_reasons
                    ),
                    adaptive_helper_materialized_context=helper_materialized_context,
                    adaptive_helper_outcome=helper_outcome,
                    adaptive_helper_failure_kind=helper_failure_kind,
                    adaptive_helper_retry_count=helper_retry_count,
                    adaptive_helper_transport_retry_count=helper_transport_retry_count,
                    adaptive_helper_decision_reason=helper_decision_reason,
                    adaptive_stagnation_override_applied=adaptive_stagnation_override_applied,
                    adaptive_stagnation_override_reason=adaptive_stagnation_override_reason,
                    adaptive_helper_need_more_context=helper_need_more_context,
                    adaptive_helper_calls_used_before_attempt=max(
                        0,
                        helper_calls_used
                        - (1 if (is_adaptive_condition and helper_invoked) else 0),
                    ),
                    adaptive_helper_calls_used_after_attempt=helper_calls_used,
                    adaptive_helper_budget_max_calls=budget_policy_applied_profile.helper_max_calls,
                    adaptive_helper_budget_unlimited=adaptive_budget_unlimited,
                    adaptive_helper_budget_exhausted=helper_budget_exhausted,
                    adaptive_helper_saturation_guarded=helper_saturation_guarded,
                    adaptive_helper_budget_profile=normalized_adaptive_helper_budget_profile,
                    adaptive_budget_topup_granted=adaptive_budget_topup_granted,
                    adaptive_budget_topup_reason=adaptive_budget_topup_reason,
                    adaptive_context_requested=adaptive_context_requested,
                    adaptive_snapshot_included=adaptive_snapshot_included,
                    adaptive_snapshot_lite_included=adaptive_snapshot_lite_included,
                    adaptive_snapshot_attempt1_denied=adaptive_snapshot_attempt1_denied,
                    adaptive_snapshot_attempt1_denied_reason=(
                        adaptive_snapshot_attempt1_denied_reason
                    ),
                    adaptive_progress_made=adaptive_progress_made,
                    adaptive_progress_reason=adaptive_progress_reason,
                    adaptive_no_progress_streak=adaptive_no_progress_streak_current,
                    adaptive_progress_stop_candidate=adaptive_progress_stop_candidate,
                    adaptive_progress_stop_triggered=False,
                    adaptive_progress_stop_reason=adaptive_progress_stop_reason,
                    adaptive_informativeness_pregate_fired=pregate_result.should_skip,
                    adaptive_informativeness_pregate_score=pregate_result.score,
                    adaptive_informativeness_pregate_exception_type=pregate_result.exception_type,
                    adaptive_informativeness_pregate_reason=pregate_result.reason,
                    adaptive_informativeness_pregate_features={
                        "type_specificity": pregate_result.f_type_specificity,
                        "message_diagnostic": pregate_result.f_message_diagnostic,
                        "crash_in_user_code": pregate_result.f_crash_in_user_code,
                        "assertion_quality": pregate_result.f_assertion_quality,
                        "frame_depth": pregate_result.f_frame_depth,
                    },
                ),
                prompt_budget=PromptBudgetMetadata(
                    prompt_budget_used_chars=prompt_budget_result.used_chars,
                    prompt_budget_overflow_chars=prompt_budget_result.overflow_chars,
                    prompt_budget_dropped_sections=list(prompt_budget_result.dropped_sections),
                ),
                context_compaction=ContextCompactionMetadata(
                    context_compaction_mode=normalized_context_compaction_mode,
                    context_compaction_scope=sorted(normalized_context_compaction_scope),
                    context_compaction_supported=supports_context_compaction,
                    context_compaction_requested=context_compaction_requested,
                    context_compaction_request_reason=context_compaction_request_reason,
                    context_compaction_trigger=context_compaction_trigger,
                    context_compaction_auto_overflow_triggered=(
                        context_compaction_auto_overflow_triggered
                    ),
                    context_compaction_invoked=context_compaction_invoked,
                    context_compaction_success=context_compaction_success,
                    context_compaction_error=context_compaction_error,
                    context_compaction_calls_used=context_compaction_calls_used,
                    context_compaction_max_calls=effective_context_compaction_max_calls,
                    context_compaction_source_sections=context_compaction_source_sections,
                    context_compaction_source_chars=context_compaction_source_chars,
                    context_compaction_replaced_sections=context_compaction_replaced_sections,
                    context_compaction_input_chars=context_compaction_input_chars,
                    context_compaction_target_chars=normalized_context_compaction_target_chars,
                    context_compaction_output_chars=context_compaction_output_chars,
                    context_compaction_prompt_chars=context_compaction_prompt_chars,
                ),
                openai=OpenAIAttemptMetadata(
                    openai_preflight_prompt_cap_chars=openai_preflight_prompt_cap_chars,
                    openai_preflight_trimmed=openai_preflight_trimmed,
                    openai_context_window_source=normalized_openai_context_window_source,
                    openai_context_window_probe_provider=(
                        normalized_openai_context_window_probe_provider
                    ),
                    openai_context_window_probe_error=normalized_openai_context_window_probe_error,
                    openai_response_mode_autoswitch_policy=(
                        normalized_openai_response_mode_autoswitch
                    ),
                    openai_response_mode_autoswitched=False,
                    openai_response_mode_autoswitch_from="",
                    openai_response_mode_autoswitch_to="",
                    openai_response_mode_autoswitch_cycle=0,
                    openai_response_mode_autoswitch_reason="",
                ),
                syntax_preflight=SyntaxPreflightMetadata(
                    syntax_preflight_enabled=normalized_syntax_preflight_enabled,
                    syntax_preflight_cycles_used=0,
                    syntax_preflight_failures=0,
                    syntax_preflight_last_error_type="",
                    syntax_preflight_last_error_msg="",
                ),
            )
        )
        attempt_metadata["prompt_order_mode"] = normalized_prompt_order_mode
        attempt_metadata["adaptive_tep_mode_requested"] = requested_adaptive_tep_mode
        attempt_metadata["adaptive_tep_full_enabled"] = full_tep_enabled
        attempt_metadata["adaptive_tep_state_path"] = adaptive_tep_state_path
        attempt_metadata["adaptive_tep_init_state_path"] = adaptive_tep_init_state_path
        if adaptive_tep_controller is not None:
            attempt_metadata["adaptive_tep_beta"] = float(adaptive_tep_controller.state.beta)
        attempt_metadata["adaptive_reveal_mode"] = normalized_adaptive_reveal_mode
        attempt_metadata["adaptive_reveal_applied"] = adaptive_reveal_applied
        attempt_metadata["adaptive_reveal_trigger"] = adaptive_reveal_trigger
        attempt_metadata["adaptive_reveal_triggers"] = list(adaptive_reveal_triggers)
        attempt_metadata["snapshot_frame_pruning_mode"] = normalized_snapshot_frame_pruning_mode
        attempt_metadata["snapshot_frame_pruning_applied"] = bool(
            snapshot_pruning_metadata.get("applied")
        )
        attempt_metadata["adaptive_budget_bwk_run_token_budget_config"] = (
            normalized_adaptive_budget_bwk_run_token_budget
        )
        attempt_metadata["snapshot_frame_pruning_frames_before"] = int(
            snapshot_pruning_metadata.get("frames_before", 0) or 0
        )
        attempt_metadata["snapshot_frame_pruning_frames_after"] = int(
            snapshot_pruning_metadata.get("frames_after", 0) or 0
        )
        attempt_metadata["snapshot_frame_pruning_kept_frame_indices"] = list(
            snapshot_pruning_metadata.get("kept_frame_indices", [])
        )
        attempt_metadata["prompt_retry_reflection"] = normalized_prompt_retry_reflection
        attempt_metadata["prompt_adaptive_stage_b_cot"] = normalized_prompt_adaptive_stage_b_cot
        attempt_metadata["prompt_adaptive_stage_b_cot_effective"] = (
            effective_prompt_adaptive_stage_b_cot
        )
        attempt_metadata["prompt_adaptive_stage_b_cot_source"] = prompt_adaptive_stage_b_cot_source
        attempt_metadata["native_baseline_mode"] = normalized_native_baseline_mode
        attempt_metadata["prompt_repair_method_scaffold"] = normalized_prompt_repair_method_scaffold
        attempt_metadata["prompt_retry_plan_then_patch"] = normalized_prompt_retry_plan_then_patch
        attempt_metadata["plan_patch_enabled"] = plan_patch_enabled
        attempt_metadata["plan_patch_first_attempt_enabled"] = plan_on_first_attempt
        attempt_metadata["plan_patch_mode"] = plan_patch_mode
        attempt_metadata["plan_patch_invoked"] = retry_plan_invoked
        attempt_metadata["plan_patch_succeeded"] = retry_plan_succeeded
        attempt_metadata["plan_patch_prompt_chars"] = retry_plan_prompt_chars
        attempt_metadata["plan_patch_plan_chars"] = len(retry_plan_text)
        attempt_metadata["plan_patch_duration_sec"] = retry_plan_duration_sec
        attempt_metadata["plan_patch_prompt_injected"] = retry_plan_prompt_injected
        attempt_metadata["plan_patch_error"] = retry_plan_error
        attempt_metadata["plan_patch_tep_equilibrium_iterations"] = (
            retry_plan_tep_equilibrium_iterations
        )
        attempt_metadata["plan_patch_tep_equilibrium_converged"] = (
            retry_plan_tep_equilibrium_converged
        )
        attempt_metadata["retry_plan_then_patch_invoked"] = retry_plan_invoked
        attempt_metadata["retry_plan_then_patch_succeeded"] = retry_plan_succeeded
        attempt_metadata["retry_plan_then_patch_prompt_chars"] = retry_plan_prompt_chars
        attempt_metadata["retry_plan_then_patch_plan_chars"] = len(retry_plan_text)
        attempt_metadata["retry_plan_then_patch_duration_sec"] = retry_plan_duration_sec
        attempt_metadata["retry_plan_then_patch_prompt_injected"] = retry_plan_prompt_injected
        attempt_metadata["retry_plan_then_patch_error"] = retry_plan_error
        attempt_metadata["retry_plan_then_patch_tep_equilibrium_iterations"] = (
            retry_plan_tep_equilibrium_iterations
        )
        attempt_metadata["retry_plan_then_patch_tep_equilibrium_converged"] = (
            retry_plan_tep_equilibrium_converged
        )
        if adaptive_solver_architecture:
            attempt_metadata["adaptive_solver_architecture"] = adaptive_solver_architecture
            attempt_metadata["adaptive_native_solver_deprecated"] = (
                adaptive_native_solver_deprecated
            )
        attempt_metadata["adaptive_controller_lambda_rlm_enabled"] = (
            normalized_adaptive_controller_lambda_rlm
        )
        attempt_metadata["adaptive_controller_lambda_rlm_repo_dir"] = str(
            normalized_adaptive_controller_lambda_rlm_repo_dir
        )
        attempt_metadata["adaptive_controller_lambda_rlm_python"] = str(
            normalized_adaptive_controller_lambda_rlm_python
        )
        attempt_metadata["adaptive_controller_lambda_rlm_context_window_chars"] = int(
            normalized_adaptive_controller_lambda_rlm_context_window_chars
        )
        attempt_metadata["adaptive_controller_lambda_rlm_timeout_sec"] = int(
            normalized_adaptive_controller_lambda_rlm_timeout_sec
        )
        attempt_metadata["adaptive_controller_lambda_rlm_invoked"] = False
        attempt_metadata["adaptive_controller_lambda_rlm_success"] = False

        patch_meta: dict[str, Any] = {}
        patch_meta["context_protocol"] = attempt_context_protocol
        patch_meta["adaptive_tep_mode_requested"] = requested_adaptive_tep_mode
        patch_meta["adaptive_tep_full_enabled"] = full_tep_enabled
        patch_meta["adaptive_tep_state_path"] = adaptive_tep_state_path
        patch_meta["adaptive_tep_init_state_path"] = adaptive_tep_init_state_path
        if adaptive_tep_controller is not None:
            patch_meta["adaptive_tep_beta"] = float(adaptive_tep_controller.state.beta)
        patch_meta["stage_a_prompt_chars"] = stage_a_prompt_chars
        patch_meta["stage_b_prompt_chars"] = len(prompt)
        patch_meta["prompt_chars_total"] = initial_prompt_chars_total
        patch_meta["completion_chars_total"] = initial_completion_chars_total
        patch_meta["chars_input"] = initial_prompt_chars_total
        patch_meta["chars_output"] = initial_completion_chars_total
        patch_meta["chars_total"] = initial_prompt_chars_total + initial_completion_chars_total
        patch_meta["evidence_fingerprint"] = evidence_fingerprint
        patch_meta["prompt_retry_reflection"] = normalized_prompt_retry_reflection
        patch_meta["prompt_adaptive_stage_b_cot"] = normalized_prompt_adaptive_stage_b_cot
        patch_meta["prompt_adaptive_stage_b_cot_effective"] = effective_prompt_adaptive_stage_b_cot
        patch_meta["prompt_adaptive_stage_b_cot_source"] = prompt_adaptive_stage_b_cot_source
        patch_meta["native_baseline_mode"] = normalized_native_baseline_mode
        patch_meta["prompt_repair_method_scaffold"] = normalized_prompt_repair_method_scaffold
        patch_meta["prompt_retry_plan_then_patch"] = normalized_prompt_retry_plan_then_patch
        patch_meta["plan_patch_enabled"] = plan_patch_enabled
        patch_meta["plan_patch_first_attempt_enabled"] = plan_on_first_attempt
        patch_meta["plan_patch_mode"] = plan_patch_mode
        patch_meta["plan_patch_invoked"] = retry_plan_invoked
        patch_meta["plan_patch_succeeded"] = retry_plan_succeeded
        patch_meta["plan_patch_prompt_chars"] = retry_plan_prompt_chars
        patch_meta["plan_patch_plan_chars"] = len(retry_plan_text)
        patch_meta["plan_patch_duration_sec"] = retry_plan_duration_sec
        patch_meta["plan_patch_prompt_injected"] = retry_plan_prompt_injected
        patch_meta["plan_patch_error"] = retry_plan_error
        patch_meta["plan_patch_tep_equilibrium_iterations"] = retry_plan_tep_equilibrium_iterations
        patch_meta["plan_patch_tep_equilibrium_converged"] = retry_plan_tep_equilibrium_converged
        patch_meta["retry_plan_then_patch_invoked"] = retry_plan_invoked
        patch_meta["retry_plan_then_patch_succeeded"] = retry_plan_succeeded
        patch_meta["retry_plan_then_patch_prompt_chars"] = retry_plan_prompt_chars
        patch_meta["retry_plan_then_patch_plan_chars"] = len(retry_plan_text)
        patch_meta["retry_plan_then_patch_duration_sec"] = retry_plan_duration_sec
        patch_meta["retry_plan_then_patch_prompt_injected"] = retry_plan_prompt_injected
        patch_meta["retry_plan_then_patch_error"] = retry_plan_error
        patch_meta["retry_plan_then_patch_tep_equilibrium_iterations"] = (
            retry_plan_tep_equilibrium_iterations
        )
        patch_meta["retry_plan_then_patch_tep_equilibrium_converged"] = (
            retry_plan_tep_equilibrium_converged
        )
        if adaptive_solver_architecture:
            patch_meta["adaptive_solver_architecture"] = adaptive_solver_architecture
            patch_meta["adaptive_native_solver_deprecated"] = adaptive_native_solver_deprecated
        patch_meta["adaptive_controller_lambda_rlm_enabled"] = (
            normalized_adaptive_controller_lambda_rlm
        )
        patch_meta["adaptive_controller_lambda_rlm_repo_dir"] = str(
            normalized_adaptive_controller_lambda_rlm_repo_dir
        )
        patch_meta["adaptive_controller_lambda_rlm_python"] = str(
            normalized_adaptive_controller_lambda_rlm_python
        )
        patch_meta["adaptive_controller_lambda_rlm_context_window_chars"] = int(
            normalized_adaptive_controller_lambda_rlm_context_window_chars
        )
        patch_meta["adaptive_controller_lambda_rlm_timeout_sec"] = int(
            normalized_adaptive_controller_lambda_rlm_timeout_sec
        )
        patch_meta["adaptive_controller_lambda_rlm_invoked"] = False
        patch_meta["adaptive_controller_lambda_rlm_success"] = False
        if isinstance(retry_plan_raw_response, str):
            patch_meta["plan_patch_raw_response_chars"] = len(retry_plan_raw_response)
        else:
            patch_meta["plan_patch_raw_response_chars"] = 0
        if record_prompt == "full" and retry_plan_invoked:
            if isinstance(retry_plan_prompt_text, str) and retry_plan_prompt_text:
                plan_prompt_path = attempt_dir / "plan_prompt.txt"
                plan_prompt_path.write_text(retry_plan_prompt_text, encoding="utf-8")
                attempt_metadata["plan_patch_prompt_path"] = str(plan_prompt_path)
                patch_meta["plan_patch_prompt_path"] = str(plan_prompt_path)
            plan_response_text = (
                retry_plan_raw_response
                if isinstance(retry_plan_raw_response, str) and retry_plan_raw_response
                else retry_plan_text
            )
            if isinstance(plan_response_text, str) and plan_response_text:
                plan_response_path = attempt_dir / "plan_response.txt"
                plan_response_path.write_text(plan_response_text, encoding="utf-8")
                attempt_metadata["plan_patch_response_path"] = str(plan_response_path)
                patch_meta["plan_patch_response_path"] = str(plan_response_path)
        if retry_plan_meta:
            patch_meta["plan_patch_meta"] = retry_plan_meta
            patch_meta["retry_plan_then_patch_meta"] = retry_plan_meta
        patch_meta["retry_prompt_mode"] = attempt_retry_prompt_mode
        patch_meta["retry_context_mode"] = effective_retry_context_mode
        patch_meta["retry_history_chars"] = retry_history_chars
        patch_meta["retry_history_truncated"] = retry_history_truncated
        patch_meta["retry_history_turns_included"] = retry_history_turns_included
        patch_meta["retry_history_budget_chars"] = retry_history_budget_chars
        patch_meta["retry_history_budget_applied"] = retry_history_budget_applied
        patch_meta["retry_delta_prompt"] = is_delta_retry
        patch_meta["retry_delta_chars"] = retry_delta_chars
        patch_meta["evidence_ids_new_count"] = len(bookkeeping_new_evidence_ids)
        patch_meta["evidence_ids_reused_count"] = len(bookkeeping_reused_evidence_ids)
        patch_meta["evidence_ids_sent_count"] = len(
            _evidence_ids_from_items(bookkeeping_evidence_items_for_prompt)
        )
        patch_meta["evidence_ids_new"] = bookkeeping_new_evidence_ids[:20]
        patch_meta["evidence_ids_reused"] = bookkeeping_reused_evidence_ids[:20]
        patch_meta["prompt_budget_used_chars"] = prompt_budget_result.used_chars
        patch_meta["prompt_budget_overflow_chars"] = prompt_budget_result.overflow_chars
        patch_meta["prompt_budget_dropped_sections"] = list(prompt_budget_result.dropped_sections)
        patch_meta["stage_a_index_included"] = bool(include_stage_a_evidence_index)
        patch_meta["stage_a_index_reason"] = stage_a_index_reason
        patch_meta["context_compaction_mode"] = normalized_context_compaction_mode
        patch_meta["context_compaction_scope"] = sorted(normalized_context_compaction_scope)
        patch_meta["context_compaction_supported"] = supports_context_compaction
        patch_meta["context_compaction_requested"] = context_compaction_requested
        patch_meta["context_compaction_request_reason"] = context_compaction_request_reason
        patch_meta["context_compaction_trigger"] = context_compaction_trigger
        patch_meta["context_compaction_auto_overflow_triggered"] = (
            context_compaction_auto_overflow_triggered
        )
        patch_meta["context_compaction_invoked"] = context_compaction_invoked
        patch_meta["context_compaction_success"] = context_compaction_success
        patch_meta["context_compaction_error"] = context_compaction_error
        patch_meta["context_compaction_calls_used"] = context_compaction_calls_used
        patch_meta["context_compaction_max_calls"] = effective_context_compaction_max_calls
        patch_meta["context_compaction_source_sections"] = list(context_compaction_source_sections)
        patch_meta["context_compaction_source_chars"] = dict(context_compaction_source_chars)
        patch_meta["context_compaction_replaced_sections"] = list(
            context_compaction_replaced_sections
        )
        patch_meta["context_compaction_input_chars"] = context_compaction_input_chars
        patch_meta["context_compaction_target_chars"] = normalized_context_compaction_target_chars
        patch_meta["context_compaction_output_chars"] = context_compaction_output_chars
        patch_meta["context_compaction_prompt_chars"] = context_compaction_prompt_chars
        if context_compaction_meta:
            patch_meta["context_compaction_meta"] = _summarize_context_compaction_meta(
                context_compaction_meta
            )
        patch_meta["openai_preflight_prompt_cap_chars"] = openai_preflight_prompt_cap_chars
        patch_meta["openai_preflight_trimmed"] = openai_preflight_trimmed
        patch_meta["openai_context_window_source"] = normalized_openai_context_window_source
        patch_meta["openai_context_window_probe_provider"] = (
            normalized_openai_context_window_probe_provider
        )
        patch_meta["openai_context_window_probe_error"] = (
            normalized_openai_context_window_probe_error
        )
        token_estimate_metadata = {
            "prompt_token_estimate_available": bool(prompt_token_estimate.get("available")),
            "prompt_token_estimate_error": str(prompt_token_estimate.get("error") or ""),
            "prompt_token_estimate_model": str(prompt_token_estimate.get("model") or ""),
            "prompt_token_estimate_tokenizer": str(prompt_token_estimate.get("tokenizer") or ""),
            "prompt_tokens_estimate_pre_call": prompt_tokens_estimate_pre_call,
            "openai_preflight_prompt_cap_tokens": openai_preflight_prompt_cap_tokens,
            "openai_preflight_prompt_exceeds_token_cap_estimate": (
                openai_preflight_prompt_exceeds_token_cap_estimate
            ),
            "adaptive_toon_token_estimate_available": bool(
                snapshot_prompt_token_savings.get("token_estimate_available")
            ),
            "adaptive_toon_token_estimate_error": str(
                snapshot_prompt_token_savings.get("token_estimate_error") or ""
            ),
            "adaptive_toon_token_estimate_model": str(
                snapshot_prompt_token_savings.get("token_estimate_model") or ""
            ),
            "adaptive_toon_token_estimate_tokenizer": str(
                snapshot_prompt_token_savings.get("token_estimate_tokenizer") or ""
            ),
            "adaptive_toon_json_tokens_estimate": snapshot_prompt_token_savings.get(
                "json_tokens_estimate"
            ),
            "adaptive_toon_encoded_tokens_estimate": snapshot_prompt_token_savings.get(
                "encoded_tokens_estimate"
            ),
            "adaptive_toon_savings_tokens_estimate": snapshot_prompt_token_savings.get(
                "savings_tokens_estimate"
            ),
            "adaptive_toon_savings_tokens_frac": snapshot_prompt_token_savings.get(
                "savings_tokens_frac"
            ),
        }
        attempt_metadata.update(token_estimate_metadata)
        patch_meta.update(token_estimate_metadata)
        patch_meta["adaptive_mode"] = helper_decision_mode
        patch_meta["adaptive_policy"] = adaptive_policy_label
        patch_meta["adaptive_signature_repeat"] = failure_signature_repeat
        patch_meta["adaptive_stagnation_streak"] = stagnation_streak
        if enable_tep:
            patch_meta["tep_metadata"] = {
                "stagnation_streak": stagnation_streak,
                "failure_signature_repeat": failure_signature_repeat,
                "nudge_priority_applied": max(0, 3 - stagnation_streak)
                if failure_signature_repeat and stagnation_streak > 0
                else None,
                "diagnosis_refinement_injected": failure_signature_repeat
                and stagnation_streak >= 2,
                "equilibrium_check_enabled": tep_equilibrium_check,
                "equilibrium_iterations": retry_plan_tep_equilibrium_iterations,
                "equilibrium_converged": retry_plan_tep_equilibrium_converged,
            }
        patch_meta["adaptive_helper_invoked"] = helper_invoked
        patch_meta["adaptive_helper_action_default"] = helper_action_default
        patch_meta["adaptive_helper_action_selected"] = helper_action_selected
        patch_meta["adaptive_helper_action_executed"] = helper_action_executed
        patch_meta["adaptive_helper_action_fallback_reason"] = helper_action_fallback_reason
        patch_meta["adaptive_helper_available_actions"] = list(helper_available_actions)
        patch_meta["adaptive_helper_unavailable_action_reasons"] = dict(
            helper_unavailable_action_reasons
        )
        patch_meta["adaptive_helper_materialized_context"] = helper_materialized_context
        patch_meta["adaptive_helper_outcome"] = helper_outcome
        patch_meta["adaptive_helper_failure_kind"] = helper_failure_kind
        patch_meta["adaptive_helper_retry_count"] = helper_retry_count
        patch_meta["adaptive_helper_transport_retry_count"] = helper_transport_retry_count
        patch_meta["adaptive_helper_decision_reason"] = helper_decision_reason
        patch_meta["adaptive_stagnation_override_applied"] = adaptive_stagnation_override_applied
        patch_meta["adaptive_stagnation_override_reason"] = adaptive_stagnation_override_reason
        patch_meta["adaptive_helper_need_more_context"] = helper_need_more_context
        patch_meta["adaptive_helper_calls_used_before_attempt"] = max(
            0,
            helper_calls_used - (1 if (is_adaptive_condition and helper_invoked) else 0),
        )
        patch_meta["adaptive_helper_calls_used_after_attempt"] = helper_calls_used
        patch_meta["adaptive_helper_budget_max_calls"] = (
            budget_policy_applied_profile.helper_max_calls
        )
        patch_meta["adaptive_helper_budget_unlimited"] = adaptive_budget_unlimited
        patch_meta["adaptive_helper_budget_exhausted"] = helper_budget_exhausted
        patch_meta["adaptive_helper_saturation_guarded"] = helper_saturation_guarded
        patch_meta["adaptive_saturation_threshold"] = normalized_adaptive_saturation_threshold
        patch_meta["adaptive_helper_budget_profile"] = normalized_adaptive_helper_budget_profile
        patch_meta["prompt_order_mode"] = normalized_prompt_order_mode
        patch_meta["adaptive_reveal_mode"] = normalized_adaptive_reveal_mode
        patch_meta["adaptive_reveal_applied"] = adaptive_reveal_applied
        patch_meta["adaptive_reveal_trigger"] = adaptive_reveal_trigger
        patch_meta["adaptive_reveal_triggers"] = list(adaptive_reveal_triggers)
        patch_meta["snapshot_frame_pruning_mode"] = normalized_snapshot_frame_pruning_mode
        patch_meta["snapshot_frame_pruning_applied"] = bool(
            snapshot_pruning_metadata.get("applied")
        )
        patch_meta["adaptive_budget_bwk_run_token_budget_config"] = (
            normalized_adaptive_budget_bwk_run_token_budget
        )
        patch_meta["snapshot_frame_pruning_frames_before"] = int(
            snapshot_pruning_metadata.get("frames_before", 0) or 0
        )
        patch_meta["snapshot_frame_pruning_frames_after"] = int(
            snapshot_pruning_metadata.get("frames_after", 0) or 0
        )
        patch_meta["snapshot_frame_pruning_kept_frame_indices"] = list(
            snapshot_pruning_metadata.get("kept_frame_indices", [])
        )
        patch_meta["adaptive_budget_topup_granted"] = adaptive_budget_topup_granted
        patch_meta["adaptive_budget_topup_reason"] = adaptive_budget_topup_reason
        patch_meta["adaptive_informativeness_pregate_fired"] = pregate_result.should_skip
        patch_meta["adaptive_informativeness_pregate_score"] = pregate_result.score
        patch_meta["adaptive_informativeness_pregate_exception_type"] = (
            pregate_result.exception_type
        )
        patch_meta["adaptive_informativeness_pregate_reason"] = pregate_result.reason
        patch_meta["adaptive_informativeness_pregate_features"] = {
            "type_specificity": pregate_result.f_type_specificity,
            "message_diagnostic": pregate_result.f_message_diagnostic,
            "crash_in_user_code": pregate_result.f_crash_in_user_code,
            "assertion_quality": pregate_result.f_assertion_quality,
            "frame_depth": pregate_result.f_frame_depth,
        }
        patch_meta["adaptive_context_requested"] = adaptive_context_requested
        patch_meta["adaptive_snapshot_included"] = adaptive_snapshot_included
        patch_meta["adaptive_snapshot_lite_included"] = adaptive_snapshot_lite_included
        patch_meta["adaptive_snapshot_attempt1_denied"] = adaptive_snapshot_attempt1_denied
        patch_meta["adaptive_snapshot_attempt1_denied_reason"] = (
            adaptive_snapshot_attempt1_denied_reason
        )
        if is_adaptive_condition:
            pre_attempt_progress_note = (
                str(previous_failure_context.get("note") or "tests_still_failing")
                if isinstance(previous_failure_context, dict)
                else "tests_still_failing"
            )
            progress_state, adaptive_no_progress_streak_current = (
                _recompute_adaptive_progress_for_attempt_outcome(
                    attempt_metadata=attempt_metadata,
                    patch_meta=patch_meta,
                    note=pre_attempt_progress_note,
                    success=False,
                    signature_repeated=True,
                    new_evidence_count=len(bookkeeping_new_evidence_ids),
                    before_run=current,
                    after_run=None,
                    previous_failure_context=previous_failure_context,
                    current_no_progress_streak=adaptive_no_progress_streak + 1,
                    recomputed_after_patch=False,
                )
            )
            adaptive_progress_made = bool(progress_state.made)
            adaptive_progress_reason = str(progress_state.reason)
            adaptive_progress_score = float(progress_state.score)
            adaptive_no_progress_streak = (
                0 if adaptive_progress_made else adaptive_no_progress_streak_current
            )
            (
                adaptive_progress_stop_candidate,
                adaptive_progress_stop_reason,
            ) = _adaptive_should_stop_for_no_progress(
                no_progress_streak=adaptive_no_progress_streak_current,
                stop_window=adaptive_no_progress_stop_window,
                helper_budget_exhausted=helper_budget_exhausted,
                helper_saturation_guarded=helper_saturation_guarded,
                helper_decision_reason=helper_decision_reason,
            )
        else:
            _sync_adaptive_progress_metadata(
                attempt_metadata=attempt_metadata,
                patch_meta=patch_meta,
                progress_state=AdaptiveProgressState(
                    made=False,
                    reason="not_adaptive_condition",
                    score=0.0,
                    component_signature=0.0,
                    component_new_evidence=0.0,
                    component_pytest_fail_reduction=0.0,
                    component_stage_advance=0.0,
                    component_verifier_score=0.0,
                    component_scope_alignment=0.0,
                    pytest_before_failed_total=None,
                    pytest_after_failed_total=None,
                    stage_before="",
                    stage_after="",
                    stage_before_ord=0,
                    stage_after_ord=0,
                ),
                no_progress_streak=0,
                recomputed_after_patch=False,
            )
        patch_meta["adaptive_progress_made"] = adaptive_progress_made
        patch_meta["adaptive_progress_reason"] = adaptive_progress_reason
        patch_meta["adaptive_no_progress_streak"] = adaptive_no_progress_streak_current
        patch_meta["adaptive_progress_stop_candidate"] = adaptive_progress_stop_candidate
        patch_meta["adaptive_progress_stop_triggered"] = False
        patch_meta["adaptive_progress_stop_reason"] = adaptive_progress_stop_reason
        patch_meta["patch_file_count"] = 0
        patch_meta["patch_test_file_count"] = 0
        patch_meta["patch_non_test_file_count"] = 0
        patch_meta["patch_touches_tests"] = False
        patch_meta["patch_file_paths"] = []
        patch_meta["patch_test_file_paths"] = []
        patch_verifier_scope_for_attempt = _effective_patch_verifier_scope(
            configured_scope=normalized_patch_verifier_scope,
            is_adaptive_condition=is_adaptive_condition,
        )
        patch_meta["patch_verifier_enabled"] = False
        patch_meta["patch_verifier_scope"] = patch_verifier_scope_for_attempt
        patch_meta["patch_verifier_min_score"] = int(normalized_patch_verifier_min_score)
        attempt_metadata["patch_verifier_scope"] = patch_verifier_scope_for_attempt
        attempt_metadata["patch_verifier_min_score"] = int(normalized_patch_verifier_min_score)
        patch_meta["patch_verifier_passed"] = None
        patch_meta["patch_verifier_score"] = None
        patch_meta["patch_verifier_reasons"] = []
        patch_meta["patch_verifier_inferred_source_scope"] = []
        patch_meta["patch_verifier_source_scope_matched"] = False
        patch_meta["patch_verifier_recovery_eligible"] = False
        patch_meta["patch_verifier_recovery_invoked"] = False
        patch_meta["patch_verifier_recovery_result"] = PATCH_VERIFIER_RECOVERY_RESULT_NOT_ELIGIBLE
        patch_meta["patch_verifier_recovery_prompt_chars_before"] = 0
        patch_meta["patch_verifier_recovery_prompt_chars_after"] = 0
        patch_meta["patch_verifier_recovery_duration_sec"] = 0.0
        patch_meta["patch_verifier_recovery_terminal_note"] = ""
        patch_meta["patch_verifier_recovery_score_before"] = None
        patch_meta["patch_verifier_recovery_reasons_before"] = []
        patch_meta["patch_verifier_recovery_patch_files_before"] = []
        patch_meta["patch_verifier_recovery_source_scope_matched_before"] = False
        patch_meta["patch_verifier_recovery_raw_meta"] = None
        patch_meta["syntax_preflight_enabled"] = normalized_syntax_preflight_enabled
        patch_meta["syntax_preflight_cycles_used"] = 0
        patch_meta["syntax_preflight_failures"] = 0
        patch_meta["syntax_preflight_last_error_type"] = ""
        patch_meta["syntax_preflight_last_error_msg"] = ""
        patch_meta["syntax_preflight_events"] = []
        _sync_openai_response_mode_autoswitch_metadata(
            attempt_metadata=attempt_metadata,
            patch_meta=patch_meta,
            policy=normalized_openai_response_mode_autoswitch,
            switched=False,
            from_mode="",
            to_mode="",
            cycle=0,
            reason="",
        )
        _sync_patcher_request_failure_metadata(
            attempt_metadata=attempt_metadata,
            patch_meta=patch_meta,
            failed=False,
        )
        _sync_patch_sanitize_metadata(
            attempt_metadata=attempt_metadata,
            patch_meta=patch_meta,
            enabled=normalized_patch_sanitize_enabled,
            passed=False,
            rejected=False,
            reasons=[],
            error_msg="",
        )
        if attempt_context_protocol == "staged" or stage_a_request_invoked:
            patch_meta["stage_a_request_success"] = stage_a_data.request_success
            patch_meta["stage_a_used_fallback"] = stage_a_data.used_fallback
            patch_meta["stage_a_reason"] = stage_a_data.reason
            patch_meta["stage_a_requests_count"] = stage_a_data.requests_count
            patch_meta["stage_a_chars_returned"] = stage_a_data.chars_returned
            if stage_a_request_payload:
                patch_meta["stage_a_diagnosis"] = str(stage_a_request_payload.get("diagnosis", ""))
                patch_meta["stage_a_missing_fact"] = stage_a_request_payload.get("missing_fact")
                patch_meta["stage_a_need_snapshot"] = bool(
                    stage_a_request_payload.get("need_snapshot")
                )
                patch_meta["stage_a_snapshot_justification"] = str(
                    stage_a_request_payload.get("snapshot_justification") or ""
                )
                patch_meta["stage_a_need_compact_context"] = bool(
                    stage_a_request_payload.get("need_compact_context")
                )
                patch_meta["stage_a_compact_context_reason"] = str(
                    stage_a_request_payload.get("compact_context_reason") or ""
                )
            if stage_a_meta:
                patch_meta["stage_a_meta"] = stage_a_meta
        (
            attempt_llm_calls,
            attempt_tokens_prompt,
            attempt_tokens_completion,
            attempt_tokens_total,
        ) = _aggregate_attempt_llm_usage(
            stage_a_meta=stage_a_meta,
            patch_meta=None,
            stage_a_default_llm_calls=stage_a_default_llm_calls,
            patch_default_llm_calls=0,
        )
        if context_compaction_invoked:
            (
                compaction_llm_calls,
                compaction_tokens_prompt,
                compaction_tokens_completion,
                compaction_tokens_total,
            ) = _aggregate_attempt_llm_usage(
                stage_a_meta=None,
                patch_meta=context_compaction_meta if context_compaction_meta else None,
                stage_a_default_llm_calls=0,
                patch_default_llm_calls=1 if is_openai_patcher else 0,
            )
            attempt_llm_calls += compaction_llm_calls
            attempt_tokens_prompt += compaction_tokens_prompt
            attempt_tokens_completion += compaction_tokens_completion
            attempt_tokens_total += compaction_tokens_total
        if retry_plan_invoked:
            (
                retry_plan_llm_calls,
                retry_plan_tokens_prompt,
                retry_plan_tokens_completion,
                retry_plan_tokens_total,
            ) = _aggregate_attempt_llm_usage(
                stage_a_meta=None,
                patch_meta=retry_plan_meta if retry_plan_meta else None,
                stage_a_default_llm_calls=0,
                patch_default_llm_calls=1 if is_openai_patcher else 0,
            )
            attempt_llm_calls += retry_plan_llm_calls
            attempt_tokens_prompt += retry_plan_tokens_prompt
            attempt_tokens_completion += retry_plan_tokens_completion
            attempt_tokens_total += retry_plan_tokens_total
        attempt_tokens_reasoning, reasoning_token_usage_available = (
            _combine_reasoning_usage_segments(
                (
                    stage_a_meta if isinstance(stage_a_meta, dict) else None,
                    stage_a_default_llm_calls,
                ),
                (
                    context_compaction_meta if context_compaction_meta else None,
                    1 if (context_compaction_invoked and is_openai_patcher) else 0,
                ),
                (
                    retry_plan_meta if retry_plan_meta else None,
                    1 if (retry_plan_invoked and is_openai_patcher) else 0,
                ),
                (patch_meta, 1 if is_openai_patcher else 0),
            )
        )
        _apply_usage_totals_to_patch_meta(
            patch_meta=patch_meta,
            llm_call_count=attempt_llm_calls,
            tokens_prompt=attempt_tokens_prompt,
            tokens_completion=attempt_tokens_completion,
            tokens_total=attempt_tokens_total,
            tokens_reasoning=attempt_tokens_reasoning,
            reasoning_token_usage_available=reasoning_token_usage_available,
        )
        _sync_attempt_usage_metadata(
            attempt_metadata=attempt_metadata,
            patch_meta=patch_meta,
            llm_call_count=attempt_llm_calls,
        )
        _sync_attempt_prompt_char_metadata(
            attempt_metadata=attempt_metadata,
            patch_meta=patch_meta,
            prompt_chars_total=initial_prompt_chars_total,
            completion_chars_total=initial_completion_chars_total,
        )
        # Populate timing/token breakdown fields from local instrumentation.
        _tb = attempt_metadata.get("timing_breakdown")
        if isinstance(_tb, dict):
            _tb["snapshot_read_sec"] = max(0.0, _snapshot_read_sec)
            _tb["stage_a_context_fulfillment_sec"] = max(0.0, _stage_a_context_fulfillment_sec)
            _tb["stage_a_total_sec"] = max(0.0, helper_local_wall_sec)
            _tb["prompt_assembly_sec"] = max(0.0, _prompt_assembly_sec)
            _tb["context_compaction_total_sec"] = max(0.0, _context_compaction_total_sec)
        _tkb = attempt_metadata.get("token_breakdown")
        if isinstance(_tkb, dict):
            _tkb["snapshot_chars"] = len(snapshot_artifact_text or "")
        _sync_adaptive_online_metadata(
            update=AdaptiveOnlineMetadataUpdate(
                attempt_metadata=attempt_metadata,
                patch_meta=patch_meta,
                mode=online_policy_mode_for_attempt,
                algo=online_policy_algo_for_attempt,
                decision=online_policy_decision,
                eligible=online_policy_eligible,
                override_reason=online_policy_override_reason,
                feature_scaling_profile=normalized_adaptive_online_feature_scaling_profile,
                configured_action_score_biases=normalized_adaptive_online_action_score_biases,
                feature_values_raw=online_policy_features_raw,
                feature_values_scaled=online_policy_features_scaled,
                reward_lambda_tokens=normalized_adaptive_reward_lambda_tokens,
                reward_mu_latency=normalized_adaptive_reward_mu_latency,
                reward_token_scale=normalized_adaptive_reward_token_scale,
                reward_latency_scale=normalized_adaptive_reward_latency_scale,
                reward_eta_progress=normalized_adaptive_reward_eta_progress,
                reward_kappa_stagnation=normalized_adaptive_reward_kappa_stagnation,
                reward_success_hint_scale=normalized_adaptive_reward_success_hint,
                reward_progress_hint_scale=normalized_adaptive_reward_progress_scale,
                reward_stagnation_penalty_scale=normalized_adaptive_reward_stagnation_scale,
                reward_helper_materialization_bonus_scale=normalized_adaptive_reward_materialization_scale,
                local_accounting=helper_local_accounting,
                behavior_action=str(
                    attempt_metadata.get("adaptive_helper_action_executed") or "none"
                ),
                decay=(
                    float(getattr(adaptive_online_policy_controller.config, "decay", 1.0))
                    if adaptive_online_policy_controller is not None
                    else 1.0
                ),
            )
        )
        _sync_adaptive_budget_policy_metadata(
            update=AdaptiveBudgetMetadataUpdate(
                attempt_metadata=attempt_metadata,
                patch_meta=patch_meta,
                mode=budget_policy_mode_for_attempt,
                algo=budget_policy_algo_for_attempt,
                architecture=normalized_adaptive_budget_policy_architecture,
                decision=budget_policy_decision,
                eligible=budget_policy_eligible,
                override_reason=budget_policy_override_reason,
                feature_scaling_profile=normalized_adaptive_budget_feature_scaling_profile,
                feature_values_raw=budget_policy_features_raw,
                feature_values_scaled=budget_policy_features_scaled,
                tiers=normalized_adaptive_budget_policy_tiers,
                tier_index_before=budget_policy_tier_index_before,
                tier_index_after=budget_policy_tier_index_after,
                selected_multiplier=budget_policy_selected_multiplier,
                applied_multiplier=budget_policy_applied_multiplier,
                default_multiplier=budget_policy_default_multiplier,
                default_controls=budget_policy_default_controls,
                selected_controls=budget_policy_selected_controls,
                applied_controls=budget_policy_applied_controls,
                selected_controls_changed=budget_policy_selected_controls_changed,
                applied_controls_changed=budget_policy_applied_controls_changed,
                partial_override=budget_policy_partial_override,
                partial_override_errors=budget_policy_partial_override_errors,
                max_ups_per_case=normalized_adaptive_budget_policy_max_ups_per_case,
                ups_used=adaptive_budget_policy_ups_used,
                reward_lambda_tokens=normalized_adaptive_reward_lambda_tokens,
                reward_mu_latency=normalized_adaptive_reward_mu_latency,
                reward_token_scale=normalized_adaptive_reward_token_scale,
                reward_latency_scale=normalized_adaptive_reward_latency_scale,
                reward_rho_budget_exceeded=normalized_adaptive_budget_reward_rho_budget_exceeded,
                reward_eta_progress=normalized_adaptive_reward_eta_progress,
                reward_kappa_stagnation=normalized_adaptive_reward_kappa_stagnation,
                reward_success_hint_scale=normalized_adaptive_reward_success_hint,
                reward_progress_hint_scale=normalized_adaptive_reward_progress_scale,
                reward_stagnation_penalty_scale=normalized_adaptive_reward_stagnation_scale,
                local_accounting=_adaptive_policy_local_accounting(),
                decay=(
                    float(getattr(adaptive_budget_policy_controller.config, "decay", 1.0))
                    if adaptive_budget_policy_controller is not None
                    else 1.0
                ),
            )
        )
        online_finalize_kwargs = {
            "is_adaptive_llm_discretion": is_adaptive_llm_discretion,
            "online_policy_decision": online_policy_decision,
            "adaptive_online_policy_controller": adaptive_online_policy_controller,
            "run_id": run_id,
            "case_id": case.case_id,
            "condition_value": condition.value,
            "attempt": attempt,
            "attempt_metadata": attempt_metadata,
            "patch_meta": patch_meta,
            "online_policy_mode_for_attempt": online_policy_mode_for_attempt,
            "online_policy_algo_for_attempt": online_policy_algo_for_attempt,
            "online_policy_eligible": online_policy_eligible,
            "online_policy_override_reason": online_policy_override_reason,
            "online_policy_feature_scaling_profile": (
                normalized_adaptive_online_feature_scaling_profile
            ),
            "online_policy_features_raw": online_policy_features_raw,
            "online_policy_features_scaled": online_policy_features_scaled,
            "helper_decision_reason": helper_decision_reason,
            "behavior_action": str(
                attempt_metadata.get("adaptive_helper_action_executed") or "none"
            ),
            "normalized_adaptive_online_policy_mode": normalized_adaptive_online_policy_mode,
            "case_difficulty": case.difficulty,
            "case_snapshot_dependency": case.snapshot_dependency,
            "adaptive_progress_made": adaptive_progress_made,
            "adaptive_progress_score": adaptive_progress_score,
            "adaptive_no_progress_streak_current": adaptive_no_progress_streak_current,
            "reward_lambda_tokens": normalized_adaptive_reward_lambda_tokens,
            "reward_mu_latency": normalized_adaptive_reward_mu_latency,
            "reward_token_scale": normalized_adaptive_reward_token_scale,
            "reward_latency_scale": normalized_adaptive_reward_latency_scale,
            "reward_eta_progress": normalized_adaptive_reward_eta_progress,
            "reward_kappa_stagnation": normalized_adaptive_reward_kappa_stagnation,
            "reward_success_hint_scale": normalized_adaptive_reward_success_hint,
            "reward_progress_hint_scale": normalized_adaptive_reward_progress_scale,
            "reward_stagnation_penalty_scale": normalized_adaptive_reward_stagnation_scale,
            "reward_helper_materialization_bonus_scale": normalized_adaptive_reward_materialization_scale,
            "local_accounting": helper_local_accounting,
            "context_materialized": context_action_materialized,
            "helper_outcome": helper_outcome,
            "enable_auxiliary_rewards": enable_auxiliary_rewards,
            "aux_pytest_fail_reduction": 0.0,
            "aux_evidence_novelty_ratio": 0.0,
            "aux_compaction_ratio": 0.0,
            "aux_informativeness_alignment": 0.0,
            "aux_pytest_reduction_weight": aux_pytest_reduction_weight,
            "aux_evidence_novelty_weight": aux_evidence_novelty_weight,
            "aux_compaction_efficiency_weight": aux_compaction_efficiency_weight,
            "aux_informativeness_alignment_weight": aux_informativeness_alignment_weight,
        }
        budget_finalize_kwargs = {
            "is_adaptive_condition": is_adaptive_condition,
            "budget_policy_decision": budget_policy_decision,
            "adaptive_budget_policy_controller": adaptive_budget_policy_controller,
            "run_id": run_id,
            "case_id": case.case_id,
            "condition_value": condition.value,
            "attempt": attempt,
            "attempt_metadata": attempt_metadata,
            "patch_meta": patch_meta,
            "budget_policy_mode_for_attempt": budget_policy_mode_for_attempt,
            "budget_policy_algo_for_attempt": budget_policy_algo_for_attempt,
            "budget_policy_eligible": budget_policy_eligible,
            "budget_policy_override_reason": budget_policy_override_reason,
            "budget_policy_feature_scaling_profile": (
                normalized_adaptive_budget_feature_scaling_profile
            ),
            "budget_policy_features_raw": budget_policy_features_raw,
            "budget_policy_features_scaled": budget_policy_features_scaled,
            "normalized_adaptive_budget_policy_mode": normalized_adaptive_budget_policy_mode,
            "adaptive_progress_made": adaptive_progress_made,
            "adaptive_progress_score": adaptive_progress_score,
            "adaptive_no_progress_streak_current": adaptive_no_progress_streak_current,
            "reward_lambda_tokens": normalized_adaptive_reward_lambda_tokens,
            "reward_mu_latency": normalized_adaptive_reward_mu_latency,
            "reward_token_scale": normalized_adaptive_reward_token_scale,
            "reward_latency_scale": normalized_adaptive_reward_latency_scale,
            "reward_eta_progress": normalized_adaptive_reward_eta_progress,
            "reward_kappa_stagnation": normalized_adaptive_reward_kappa_stagnation,
            "reward_rho_budget_exceeded": normalized_adaptive_budget_reward_rho_budget_exceeded,
            "reward_success_hint_scale": normalized_adaptive_reward_success_hint,
            "reward_progress_hint_scale": normalized_adaptive_reward_progress_scale,
            "reward_stagnation_penalty_scale": normalized_adaptive_reward_stagnation_scale,
            "local_accounting": _adaptive_policy_local_accounting(),
            "tiers": normalized_adaptive_budget_policy_tiers,
            "tier_index_before": budget_policy_tier_index_before,
            "tier_index_after": budget_policy_tier_index_after,
            "selected_multiplier": budget_policy_selected_multiplier,
            "applied_multiplier": budget_policy_applied_multiplier,
            "default_multiplier": budget_policy_default_multiplier,
            "default_controls": budget_policy_default_controls,
            "selected_controls": budget_policy_selected_controls,
            "applied_controls": budget_policy_applied_controls,
            "selected_controls_changed": budget_policy_selected_controls_changed,
            "applied_controls_changed": budget_policy_applied_controls_changed,
            "partial_override": budget_policy_partial_override,
            "partial_override_errors": budget_policy_partial_override_errors,
            "max_ups_per_case": normalized_adaptive_budget_policy_max_ups_per_case,
            "ups_used": adaptive_budget_policy_ups_used,
            "enable_auxiliary_rewards": enable_auxiliary_rewards,
            "aux_pytest_fail_reduction": 0.0,
            "aux_evidence_novelty_ratio": 0.0,
            "aux_pytest_reduction_weight": aux_pytest_reduction_weight,
            "aux_evidence_novelty_weight": aux_evidence_novelty_weight,
        }

        if (
            is_openai_patcher
            and is_adaptive_condition
            and helper_materialized_context
            and normalized_adaptive_openai_helper_response_mode != "inherit"
        ):
            target_mode = normalized_adaptive_openai_helper_response_mode
            current_mode = str(getattr(patcher_for_attempt, "response_mode", "")).strip()
            if target_mode and target_mode != current_mode:
                try:
                    patcher_for_attempt = replace(patcher_for_attempt, response_mode=target_mode)
                    patch_meta["adaptive_openai_response_mode_override"] = target_mode
                except Exception as exc:
                    patch_meta["adaptive_openai_response_mode_override_error"] = (
                        f"{type(exc).__name__}: {exc}"
                    )
        if budget_policy_partial_override_errors.get("max_propose_sec"):
            patch_meta["adaptive_budget_propose_budget_override_error"] = (
                budget_policy_partial_override_errors["max_propose_sec"]
            )
        if is_openai_patcher:
            attempt_metadata["attempt_shared_propose_budget_base_sec"] = float(
                budget_policy_base_propose_budget_sec
            )
            patch_meta["attempt_shared_propose_budget_base_sec"] = float(
                budget_policy_base_propose_budget_sec
            )
            attempt_metadata["attempt_shared_propose_budget_multiplier"] = float(
                budget_policy_applied_propose_multiplier
            )
            patch_meta["attempt_shared_propose_budget_multiplier"] = float(
                budget_policy_applied_propose_multiplier
            )
        else:
            attempt_metadata["attempt_shared_propose_budget_base_sec"] = 0.0
            patch_meta["attempt_shared_propose_budget_base_sec"] = 0.0
            attempt_metadata["attempt_shared_propose_budget_multiplier"] = 1.0
            patch_meta["attempt_shared_propose_budget_multiplier"] = 1.0

        budget_finalize_kwargs["budget_policy_decision"] = budget_policy_decision
        budget_finalize_kwargs["budget_policy_override_reason"] = budget_policy_override_reason
        budget_finalize_kwargs["tier_index_after"] = budget_policy_tier_index_after
        budget_finalize_kwargs["applied_multiplier"] = budget_policy_applied_multiplier
        budget_finalize_kwargs["default_controls"] = budget_policy_default_controls
        budget_finalize_kwargs["selected_controls"] = budget_policy_selected_controls
        budget_finalize_kwargs["applied_controls"] = budget_policy_applied_controls
        budget_finalize_kwargs["selected_controls_changed"] = (
            budget_policy_selected_controls_changed
        )
        budget_finalize_kwargs["applied_controls_changed"] = budget_policy_applied_controls_changed
        budget_finalize_kwargs["partial_override"] = budget_policy_partial_override
        budget_finalize_kwargs["partial_override_errors"] = budget_policy_partial_override_errors
        budget_finalize_kwargs["ups_used"] = adaptive_budget_policy_ups_used

        # Retry gate: skip propose_patch when evidence fingerprint is unchanged.
        # Only evaluate when Stage-A actually produced evidence this attempt.
        if (
            is_adaptive_condition
            and helper_materialized_context
            and bool(evidence.get("failure_signature_repeat"))
        ):
            patch_meta["retry_gate_evaluated"] = True
            patch_meta["retry_gate_blocked"] = False
            patch_meta["retry_gate_reason"] = "new_evidence_present"
            prev_evidence_fingerprint = ""
            if isinstance(previous_failure_context, dict):
                prev_evidence_fingerprint = str(
                    previous_failure_context.get("evidence_fingerprint") or ""
                )
            if prev_evidence_fingerprint and prev_evidence_fingerprint == evidence_fingerprint:
                patch_meta["retry_gate_blocked"] = True
                patch_meta["retry_gate_reason"] = "same_signature_without_new_evidence"
                patch_meta["retry_gate_previous_evidence_fingerprint"] = prev_evidence_fingerprint
                attempt_wall_sec = _measure_attempt_wall_clock_sec(attempt_start)
                cumulative_runtime_sec += attempt_wall_sec
                pending_credit = _finalize_adaptive_policy_attempt(
                    note="retry_gate_no_new_evidence",
                    success=False,
                    attempt_wall_sec=attempt_wall_sec,
                    online_finalize_kwargs=online_finalize_kwargs,
                    budget_finalize_kwargs=budget_finalize_kwargs,
                )
                record = _build_attempt_outcome_record(
                    case=case,
                    condition=condition,
                    run_id=run_id,
                    attempt=attempt,
                    patcher_name=patcher_name,
                    run_metadata=effective_run_metadata,
                    attempt_metadata=attempt_metadata,
                    patcher=patcher_for_attempt,
                    patch_meta=patch_meta,
                    context_protocol=attempt_context_protocol,
                    success=False,
                    note="retry_gate_no_new_evidence",
                    before=current,
                    cumulative_runtime_sec=cumulative_runtime_sec,
                    attempt_wall_clock_sec=attempt_wall_sec,
                )
                _queue_pending_online_credit(pending_credit=pending_credit, record=record)
                _append_record(record, defer_sink=pending_credit is not None)
                _mark_scored_attempt_consumed()
                _capture_attempt_history(
                    attempt=attempt,
                    note="retry_gate_no_new_evidence",
                    failure_signature=failure_signature,
                    record=record,
                    patch_meta=patch_meta,
                    retry_plan_value=retry_plan_text,
                    snapshot_for_history=snapshot,
                )
                previous_failure_context = _enrich_failure_context(
                    base=current_failure_context,
                    note="retry_gate_no_new_evidence",
                    patch_meta=patch_meta,
                    run=current,
                )
                continue

        should_write_adaptive_baseline_sidecars = is_baseline_patcher and is_adaptive_condition
        syntax_state = make_syntax_preflight_state(
            enabled=normalized_syntax_preflight_enabled,
            max_cycles=(
                normalized_syntax_preflight_max_cycles if normalized_syntax_preflight_enabled else 1
            ),
        )
        syntax_preflight_cycles_used = syntax_state.cycles_used
        syntax_preflight_failures = syntax_state.failures
        syntax_preflight_last_error_type = syntax_state.last_error_type
        syntax_preflight_last_error_msg = syntax_state.last_error_msg
        syntax_preflight_events = syntax_state.events
        syntax_preflight_max_cycles_for_attempt = syntax_state.max_cycles
        syntax_retry_feedback = syntax_state.retry_feedback
        prepared_controller_runtime = None
        if should_write_adaptive_baseline_sidecars:
            prepared_controller_runtime = prepare_controller_runtime(
                inputs=ControllerRuntimeInputs(
                    case_id=case.case_id,
                    condition_value=condition.value,
                    attempt=attempt,
                    workdir=workdir,
                    attempt_dir=attempt_dir,
                    snapshot=snapshot,
                    failure_signature=failure_signature,
                    evidence_items=controller_evidence_items,
                    new_evidence_ids=controller_new_evidence_ids,
                    reused_evidence_ids=controller_reused_evidence_ids,
                    retry_prompt_mode=attempt_retry_prompt_mode,
                    retry_context_mode=effective_retry_context_mode,
                    retry_history_text=retry_history_text,
                    retry_history_turns=retry_history_turns,
                    retry_history_turns_included=retry_history_turns_included,
                    retry_history_truncated=retry_history_truncated,
                    previous_failure_signature=str(previous_failure_signature or ""),
                    previous_note=previous_note,
                    previous_patch_verifier_rejected=previous_patch_verifier_rejected,
                    previous_patch_verifier_reasons=previous_patch_verifier_reasons,
                    previous_patch_verifier_scope=previous_patch_verifier_scope,
                    previous_timed_out=previous_timed_out,
                    previous_timeout_sec=previous_timeout_sec,
                    helper_action_selected=helper_action_selected,
                    helper_action_executed=helper_action_executed,
                    helper_decision_reason=helper_decision_reason,
                    helper_outcome=helper_outcome,
                    helper_materialized_context=helper_materialized_context,
                    helper_calls_used=helper_calls_used,
                    helper_budget_max_calls=budget_policy_applied_profile.helper_max_calls,
                    no_progress_streak=adaptive_no_progress_streak_current,
                    context_compaction_requested=context_compaction_requested,
                    context_compaction_invoked=context_compaction_invoked,
                    context_compaction_success=context_compaction_success,
                    context_compaction_trigger=context_compaction_trigger,
                    prompt_budget=attempt_prompt_budget,
                    retry_plan_text=retry_plan_text,
                    lambda_rlm_enabled=normalized_adaptive_controller_lambda_rlm,
                    lambda_rlm_repo_dir=normalized_adaptive_controller_lambda_rlm_repo_dir,
                    lambda_rlm_python=normalized_adaptive_controller_lambda_rlm_python,
                    lambda_rlm_context_window_chars=(
                        normalized_adaptive_controller_lambda_rlm_context_window_chars
                    ),
                    lambda_rlm_timeout_sec=normalized_adaptive_controller_lambda_rlm_timeout_sec,
                    case_timeout_sec=case.timeout_sec,
                    patcher_name=str(
                        getattr(patcher, "name", "") or getattr(patcher_for_attempt, "name", "")
                    ),
                    configured_model=str(
                        getattr(getattr(patcher, "config", None), "model", "")
                        or getattr(getattr(patcher_for_attempt, "config", None), "model", "")
                    ).strip(),
                    configured_api_base_url=str(
                        getattr(getattr(patcher, "config", None), "api_base_url", "")
                        or getattr(getattr(patcher_for_attempt, "config", None), "api_base_url", "")
                    ),
                    configured_api_key=str(
                        getattr(getattr(patcher, "config", None), "api_key", "")
                        or getattr(getattr(patcher_for_attempt, "config", None), "api_key", "")
                    ),
                    lambda_rlm_invoke=invoke_lambda_rlm_decomposition,
                    resolve_effective_model=resolve_effective_baseline_model,
                )
            )
            attempt_metadata.update(prepared_controller_runtime.prepared_attempt_meta)
            patch_meta.update(prepared_controller_runtime.prepared_patch_meta)
        attempt_metadata["attempt_shared_propose_budget_sec"] = attempt_shared_propose_budget_sec
        patch_meta["attempt_shared_propose_budget_sec"] = attempt_shared_propose_budget_sec
        attempt_llm_calls_cumulative = _nonnegative_int(
            attempt_metadata.get("llm_call_count"), default=0
        )
        attempt_tokens_prompt_cumulative = _nonnegative_int(
            attempt_metadata.get("tokens_prompt"), default=0
        )
        attempt_tokens_completion_cumulative = _nonnegative_int(
            attempt_metadata.get("tokens_completion"), default=0
        )
        attempt_tokens_total_cumulative = _nonnegative_int(
            attempt_metadata.get("tokens_total"), default=0
        )
        attempt_tokens_reasoning_cumulative = _nonnegative_int(
            attempt_metadata.get("tokens_reasoning"), default=0
        )
        attempt_reasoning_token_usage_any = attempt_llm_calls_cumulative > 0
        attempt_reasoning_token_usage_all_available = (
            bool(attempt_metadata.get("reasoning_token_usage_available"))
            if attempt_reasoning_token_usage_any
            else True
        )
        attempt_prompt_chars_cumulative = _nonnegative_int(
            attempt_metadata.get("prompt_chars_total"), default=0
        )
        attempt_completion_chars_cumulative = _nonnegative_int(
            attempt_metadata.get("completion_chars_total"), default=0
        )
        budget_local_llm_calls = 0
        budget_local_tokens_prompt = 0
        budget_local_tokens_completion = 0
        budget_local_tokens_total = 0
        budget_local_wall_sec = 0.0
        lambda_rlm_call_count = _nonnegative_int(
            patch_meta.get("adaptive_controller_lambda_rlm_call_count"), default=0
        )
        lambda_rlm_prompt_tokens = _nonnegative_int(
            patch_meta.get("adaptive_controller_lambda_rlm_prompt_tokens"), default=0
        )
        lambda_rlm_completion_tokens = _nonnegative_int(
            patch_meta.get("adaptive_controller_lambda_rlm_completion_tokens"), default=0
        )
        lambda_rlm_total_tokens = _nonnegative_int(
            patch_meta.get("adaptive_controller_lambda_rlm_total_tokens"), default=0
        )
        lambda_rlm_duration_sec = max(
            0.0,
            float(patch_meta.get("adaptive_controller_lambda_rlm_duration_sec") or 0.0),
        )
        attempt_llm_calls_cumulative += lambda_rlm_call_count
        attempt_tokens_prompt_cumulative += lambda_rlm_prompt_tokens
        attempt_tokens_completion_cumulative += lambda_rlm_completion_tokens
        attempt_tokens_total_cumulative += lambda_rlm_total_tokens
        if retry_plan_invoked:
            budget_local_llm_calls += retry_plan_llm_calls
            budget_local_tokens_prompt += retry_plan_tokens_prompt
            budget_local_tokens_completion += retry_plan_tokens_completion
            budget_local_tokens_total += retry_plan_tokens_total
            budget_local_wall_sec += retry_plan_duration_sec
        if lambda_rlm_call_count > 0 or lambda_rlm_duration_sec > 0.0:
            budget_local_llm_calls += lambda_rlm_call_count
            budget_local_tokens_prompt += lambda_rlm_prompt_tokens
            budget_local_tokens_completion += lambda_rlm_completion_tokens
            budget_local_tokens_total += lambda_rlm_total_tokens
            budget_local_wall_sec += lambda_rlm_duration_sec
        if lambda_rlm_call_count > 0:
            attempt_reasoning_token_usage_any = True
            attempt_reasoning_token_usage_all_available = False
        diff_to_apply = ""
        context_recovery_invoked = False
        attempt_terminated_early = False
        should_break_after_attempt = False
        patch_applied_and_ready = False

        def _accumulate_timing_metadata_field(
            field: str,
            delta_sec: float,
            *targets: dict[str, Any],
        ) -> None:
            normalized_delta_sec = max(0.0, float(delta_sec))
            for target in targets:
                target[field] = max(0.0, float(target.get(field) or 0.0)) + normalized_delta_sec

        syntax_state.apply_to_metadata(attempt_metadata=attempt_metadata, patch_meta=patch_meta)
        for syntax_cycle in range(1, syntax_preflight_max_cycles_for_attempt + 1):
            syntax_state.cycles_used = syntax_cycle
            syntax_preflight_cycles_used = syntax_state.cycles_used
            syntax_preflight_failures = syntax_state.failures
            syntax_preflight_last_error_type = syntax_state.last_error_type
            syntax_preflight_last_error_msg = syntax_state.last_error_msg
            syntax_retry_feedback = syntax_state.retry_feedback
            if should_write_adaptive_baseline_sidecars and prepared_controller_runtime is not None:
                sidecar_write_result = write_controller_sidecars_for_cycle(
                    prepared=prepared_controller_runtime,
                    syntax_cycle=syntax_cycle,
                    syntax_max_cycles=syntax_preflight_max_cycles_for_attempt,
                    syntax_last_error_type=syntax_preflight_last_error_type,
                    syntax_last_error_msg=syntax_preflight_last_error_msg,
                    syntax_retry_feedback=syntax_retry_feedback,
                    context_dir=attempt_dir,
                )
                written_sidecar_names = list(sidecar_write_result.written_sidecar_names)
                sidecar_error = sidecar_write_result.sidecar_error
                sidecar_build_sec = max(0.0, float(sidecar_write_result.sidecar_build_sec))
                sidecar_write_sec = max(0.0, float(sidecar_write_result.sidecar_write_sec))
                attempt_metadata["adaptive_controller_packet_written"] = bool(written_sidecar_names)
                patch_meta["adaptive_controller_packet_written"] = bool(written_sidecar_names)
                attempt_metadata["adaptive_controller_sidecars"] = list(written_sidecar_names)
                patch_meta["adaptive_controller_sidecars"] = list(written_sidecar_names)
                _accumulate_timing_metadata_field(
                    "adaptive_controller_sidecar_build_sec",
                    sidecar_build_sec,
                    attempt_metadata,
                    patch_meta,
                )
                _accumulate_timing_metadata_field(
                    "adaptive_controller_sidecar_write_sec",
                    sidecar_write_sec,
                    attempt_metadata,
                    patch_meta,
                )
                if sidecar_error:
                    attempt_metadata["adaptive_controller_sidecar_error"] = sidecar_error
                    patch_meta["adaptive_controller_sidecar_error"] = sidecar_error
            _emit_progress(
                "attempt_propose_patch",
                _progress_payload(
                    owner="harness",
                    phase_label="harness: propose patch",
                    note="proposing patch",
                    attempt=int(attempt),
                    max_attempts=int(k),
                    syntax_cycle=int(syntax_cycle),
                    syntax_max_cycles=int(syntax_preflight_max_cycles_for_attempt),
                ),
            )
            if is_baseline_patcher:
                baseline_provider = str(getattr(patcher_for_attempt, "name", "") or "").strip()
                baseline_provider = baseline_provider or "baseline"
                _emit_progress(
                    "baseline_request",
                    _progress_payload(
                        owner="baseline",
                        attempt=int(attempt),
                        max_attempts=int(k),
                        syntax_cycle=int(syntax_cycle),
                        syntax_max_cycles=int(syntax_preflight_max_cycles_for_attempt),
                        phase_label=f"baseline: {baseline_provider} subprocess",
                        waiting_for=f"{baseline_provider}_subprocess",
                        external_operation="baseline_repair",
                        external_provider=baseline_provider,
                        note=f"{baseline_provider} subprocess",
                    ),
                )
            cycle_prompt = prompt + syntax_retry_feedback
            propose_invoke_count = 0

            def _rebuild_attempt_dir_for_transport_replay(
                *,
                replay_attempt_dir=attempt_dir,
                replay_current=current,
            ) -> None:
                shutil.rmtree(replay_attempt_dir, ignore_errors=True)
                replay_attempt_dir.mkdir(parents=True, exist_ok=True)
                if is_baseline_patcher:
                    remove_adaptive_controller_sidecars(context_dir=replay_attempt_dir)
                    (replay_attempt_dir / "baseline_case_context.json").write_text(
                        json.dumps(
                            {
                                "schema_version": BASELINE_CONTRACT_VERSION,
                                "case_id": case.case_id,
                                "python_args": list(case.python_args),
                                "timeout_sec": int(case.timeout_sec),
                                "description": case.description,
                                "category": case.category,
                                "difficulty": case.difficulty,
                                "snapshot_dependency": case.snapshot_dependency,
                            }
                        ),
                        encoding="utf-8",
                    )
                _write_run_outputs(
                    replay_attempt_dir,
                    replay_current,
                    prefix="pytest_before",
                    tail_lines=normalized_pytest_output_tail_lines,
                )

            def _invoke_propose_patch(
                *,
                replay_patcher=patcher_for_attempt,
                replay_prompt=cycle_prompt,
                replay_attempt_dir=attempt_dir,
                replay_snapshot_full_text=snapshot_full_text,
                replay_snapshot_artifact_text=snapshot_artifact_text,
                replay_deadline=attempt_shared_propose_deadline_monotonic,
            ) -> Any:
                nonlocal propose_invoke_count
                propose_invoke_count += 1
                if propose_invoke_count > 1:
                    _rebuild_attempt_dir_for_transport_replay()
                system_prompt_override = None
                if full_tep_enabled and adaptive_tep_controller is not None:
                    system_prompt_override = adaptive_tep_controller.patch_system_prompt(
                        response_mode=str(getattr(replay_patcher, "response_mode", "") or "edits")
                    )
                return _propose_patch_with_snapshot_context(
                    patcher=replay_patcher,
                    prompt=replay_prompt,
                    context_dir=replay_attempt_dir,
                    cwd=workdir,
                    snapshot_full_text=replay_snapshot_full_text,
                    snapshot_artifact_text=replay_snapshot_artifact_text,
                    system_prompt_override=system_prompt_override,
                    attempt_deadline_monotonic=replay_deadline,
                )

            patch_start = time.perf_counter()
            patch = _run_openai_operation_with_transport_recovery(
                patcher_for_attempt=patcher_for_attempt,
                operation_name="propose_patch",
                invoke=_invoke_propose_patch,
                classify_failure_kind=_transport_recovery_failure_kind_from_meta,
                attempt=attempt,
                attempt_metadata=attempt_metadata,
                attempt_context_protocol=attempt_context_protocol,
                before=current,
                cumulative_runtime_sec_value=cumulative_runtime_sec,
            )
            patch_duration_sec = time.perf_counter() - patch_start
            patch_usage_meta: dict[str, Any] | None = None
            _reset_patch_usage_sync_metadata(patch_meta)
            for key in (
                "patch_diff_xxh3_128",
                "patch_diff_chars",
                "patch_file_count",
                "patch_test_file_count",
                "patch_non_test_file_count",
                "patch_touches_tests",
                "patch_file_paths",
                "patch_test_file_paths",
                "patch_verifier_passed",
                "patch_verifier_score",
                "patch_verifier_reasons",
                "patch_verifier_patch_files",
                "patch_verifier_crash_file",
                "patch_verifier_crash_file_matched",
                "patch_verifier_crash_neighbor_matched",
                "patch_verifier_crash_neighbors",
                "patch_verifier_inferred_source_scope",
                "patch_verifier_source_scope_matched",
                "patch_verifier_failing_test_file",
                "patch_verifier_touches_test_file",
                "patch_verifier_touches_failing_test_file",
                "patch_verifier_substantive_change",
                "patch_verifier_recovery_eligible",
                "patch_verifier_recovery_invoked",
                "patch_verifier_recovery_result",
                "patch_verifier_recovery_prompt_chars_before",
                "patch_verifier_recovery_prompt_chars_after",
                "patch_verifier_recovery_duration_sec",
                "patch_verifier_recovery_terminal_note",
                "patch_verifier_recovery_score_before",
                "patch_verifier_recovery_reasons_before",
                "patch_verifier_recovery_patch_files_before",
                "patch_verifier_recovery_source_scope_matched_before",
                "patch_verifier_recovery_raw_meta",
                "apply_duration_sec",
                "initial_apply_error",
                "runner_forced_patch_attempt",
                "runner_forced_patch_target",
                "runner_forced_patch_reason",
                "runner_forced_patch_build_sec",
                "runner_forced_patch_recovery",
                "runner_forced_patch_recovery_build_sec",
                "runner_forced_patch_recovery_apply_sec",
                "runner_forced_patch_recovery_target",
                "runner_forced_patch_recovery_reason",
                "runner_forced_patch_recovery_touches_tests",
                "runner_forced_patch_recovery_file_count",
                "runner_forced_patch_recovery_test_file_count",
                "runner_forced_patch_recovery_file_paths",
                "runner_forced_patch_recovery_error",
                "runner_forced_patch_skipped",
                "runner_forced_patch_skip_reason",
                "runner_forced_patch_marker_only",
                "runner_forced_patch_disabled_after_noop",
                "context_recovery_invoked",
                "context_recovery_kind",
                "context_recovery_prompt_chars_before",
                "context_recovery_prompt_chars_after",
                "context_recovery_duration_sec",
                "context_recovery_retry_success",
                "context_recovery_meta",
                "context_recovery_raw_meta",
                "no_diff_recovery_invoked",
                "no_diff_recovery_prompt_chars_before",
                "no_diff_recovery_prompt_chars_after",
                "no_diff_recovery_duration_sec",
                "no_diff_recovery_retry_success",
                "no_diff_recovery_meta",
                "no_diff_recovery_raw_meta",
                "patcher_request_failed",
                "patcher_request_failure_kind",
                "patcher_request_failure_error",
                "patcher_request_failure_status",
                "patcher_request_failure_exc",
                "patch_sanitize_enabled",
                "patch_sanitize_passed",
                "patch_sanitize_rejected",
                "patch_sanitize_reasons",
                "patch_sanitize_error_msg",
                "adaptive_no_progress_token_sum",
                "adaptive_no_progress_token_cap",
                "adaptive_no_progress_token_cap_hit",
            ):
                patch_meta.pop(key, None)
            if isinstance(patch.meta, dict):
                patch_usage_meta = dict(patch.meta)
                baseline_case_conversion_sec_cumulative = max(
                    0.0,
                    float(patch_meta.get("baseline_case_conversion_sec") or 0.0),
                )
                baseline_sidecar_load_sec_cumulative = max(
                    0.0,
                    float(patch_meta.get("baseline_sidecar_load_sec") or 0.0),
                )
                patch_meta.update(patch_usage_meta)
                if "baseline_case_conversion_sec" in patch_usage_meta:
                    patch_meta["baseline_case_conversion_sec"] = (
                        baseline_case_conversion_sec_cumulative
                        + max(
                            0.0,
                            float(patch_usage_meta.get("baseline_case_conversion_sec") or 0.0),
                        )
                    )
                if "baseline_sidecar_load_sec" in patch_usage_meta:
                    patch_meta["baseline_sidecar_load_sec"] = (
                        baseline_sidecar_load_sec_cumulative
                        + max(
                            0.0,
                            float(patch_usage_meta.get("baseline_sidecar_load_sec") or 0.0),
                        )
                    )
            elif patch.meta is not None:
                patch_meta["raw_meta"] = patch.meta
            patch_meta["propose_duration_sec"] = patch_duration_sec
            patch_meta["syntax_preflight_cycle"] = syntax_cycle
            patch_meta["syntax_preflight_prompt_chars"] = len(cycle_prompt)
            _ensure_deterministic_seed_ack(
                deterministic=deterministic,
                seed=seed,
                patcher_name=patcher_name,
                patch_meta=patch_meta,
                case_id=case.case_id,
                condition=condition.value,
                attempt=attempt,
            )
            (
                cycle_llm_calls,
                cycle_tokens_prompt,
                cycle_tokens_completion,
                cycle_tokens_total,
            ) = _aggregate_attempt_llm_usage(
                stage_a_meta=None,
                patch_meta=patch_usage_meta,
                stage_a_default_llm_calls=0,
                patch_default_llm_calls=1 if is_openai_patcher else 0,
            )
            attempt_llm_calls_cumulative += cycle_llm_calls
            attempt_tokens_prompt_cumulative += cycle_tokens_prompt
            attempt_tokens_completion_cumulative += cycle_tokens_completion
            attempt_tokens_total_cumulative += cycle_tokens_total
            cycle_tokens_reasoning, cycle_reasoning_applicable, cycle_reasoning_available = (
                _reasoning_usage_segment_from_meta(
                    patch_usage_meta,
                    default_llm_calls=1 if is_openai_patcher else 0,
                )
            )
            if cycle_reasoning_applicable:
                attempt_reasoning_token_usage_any = True
                if cycle_reasoning_available:
                    attempt_tokens_reasoning_cumulative += cycle_tokens_reasoning
                else:
                    attempt_reasoning_token_usage_all_available = False
            patch_prompt_chars, patch_completion_chars = _patch_attempt_char_deltas(
                cycle_prompt=cycle_prompt,
                diff=patch.diff,
                patch_meta=patch_usage_meta,
                is_baseline_patcher=is_baseline_patcher,
            )
            attempt_prompt_chars_cumulative += patch_prompt_chars
            attempt_completion_chars_cumulative += patch_completion_chars
            budget_local_llm_calls += cycle_llm_calls
            budget_local_tokens_prompt += cycle_tokens_prompt
            budget_local_tokens_completion += cycle_tokens_completion
            budget_local_tokens_total += cycle_tokens_total
            budget_local_wall_sec += patch_duration_sec
            budget_finalize_kwargs["local_accounting"] = _adaptive_policy_local_accounting(
                llm_call_count=budget_local_llm_calls,
                tokens_prompt=budget_local_tokens_prompt,
                tokens_completion=budget_local_tokens_completion,
                tokens_total=budget_local_tokens_total,
                wall_sec=budget_local_wall_sec,
            )
            _apply_usage_totals_to_patch_meta(
                patch_meta=patch_meta,
                llm_call_count=attempt_llm_calls_cumulative,
                tokens_prompt=attempt_tokens_prompt_cumulative,
                tokens_completion=attempt_tokens_completion_cumulative,
                tokens_total=attempt_tokens_total_cumulative,
                tokens_reasoning=attempt_tokens_reasoning_cumulative,
                reasoning_token_usage_available=(
                    attempt_reasoning_token_usage_any
                    and attempt_reasoning_token_usage_all_available
                ),
            )
            _sync_attempt_usage_metadata(
                attempt_metadata=attempt_metadata,
                patch_meta=patch_meta,
                llm_call_count=attempt_llm_calls_cumulative,
            )
            _sync_attempt_prompt_char_metadata(
                attempt_metadata=attempt_metadata,
                patch_meta=patch_meta,
                prompt_chars_total=attempt_prompt_chars_cumulative,
                completion_chars_total=attempt_completion_chars_cumulative,
            )
            _sync_syntax_preflight_metadata(
                attempt_metadata=attempt_metadata,
                patch_meta=patch_meta,
                enabled=normalized_syntax_preflight_enabled,
                cycles_used=syntax_preflight_cycles_used,
                failures=syntax_preflight_failures,
                last_error_type=syntax_preflight_last_error_type,
                last_error_msg=syntax_preflight_last_error_msg,
                events=syntax_preflight_events,
            )
            _sync_patcher_request_failure_metadata(
                attempt_metadata=attempt_metadata,
                patch_meta=patch_meta,
                failed=False,
            )
            _sync_patch_sanitize_metadata(
                attempt_metadata=attempt_metadata,
                patch_meta=patch_meta,
                enabled=normalized_patch_sanitize_enabled,
                passed=False,
                rejected=False,
                reasons=[],
                error_msg="",
            )
            _sync_patch_verifier_recovery_metadata(
                attempt_metadata=attempt_metadata,
                patch_meta=patch_meta,
                eligible=False,
                invoked=False,
                result=PATCH_VERIFIER_RECOVERY_RESULT_NOT_ELIGIBLE,
            )
            if bool(patch_meta.get("infra_resume_eligible")):
                attempt_metadata["infra_failure_kind"] = str(
                    patch_meta.get("infra_failure_kind") or ""
                )
                attempt_metadata["infra_resume_eligible"] = True
                attempt_metadata["infra_source"] = str(patch_meta.get("infra_source") or "")
                attempt_terminated_early = True
                should_break_after_attempt = True
                break

            diff_to_apply = patch.diff
            request_failed, request_failure_kind = _classify_patcher_request_failure(patch_meta)
            request_failure_error = str(patch_meta.get("error") or "")
            request_failure_status = patch_meta.get("status")
            request_failure_status_int = (
                int(request_failure_status) if isinstance(request_failure_status, int) else None
            )
            request_failure_exc = str(patch_meta.get("exc") or "")
            if request_failed:
                _sync_patcher_request_failure_metadata(
                    attempt_metadata=attempt_metadata,
                    patch_meta=patch_meta,
                    failed=True,
                    kind=request_failure_kind,
                    error=request_failure_error,
                    status=request_failure_status_int,
                    exc=request_failure_exc,
                )
            if (
                diff_to_apply is None
                and request_failed
                and not context_recovery_invoked
                and request_failure_kind in CONTEXT_RECOVERY_FAILURE_KINDS
                and not (
                    normalized_case_fail_fast_on_request_budget_exceeded
                    and request_failure_error == "request_budget_exceeded"
                )
            ):
                context_recovery_invoked = True
                recovery_sections = _build_stage_b_sections_for_attempt(
                    include_stage_a_evidence_index_value=True,
                    include_project_files_value=False,
                    compacted_context_value=compacted_context_for_prompt,
                    include_snapshot_value=include_snapshot_in_prompt,
                    include_pytest_output_value=include_pytest_output_in_prompt,
                    include_stage_a_context_value=include_stage_a_context_in_prompt,
                    retry_repair_plan_value=retry_plan_text,
                    enable_tep_value=enable_tep and tep_nudge_escalation,
                    stagnation_streak_value=stagnation_streak,
                )
                recovery_budget_result = _apply_prompt_budget(
                    sections=recovery_sections,
                    budget=CONTEXT_RECOVERY_PROMPT_BUDGET,
                    is_delta_retry=is_delta_retry,
                )
                recovery_prompt = recovery_budget_result.prompt + syntax_retry_feedback
                recovery_patch_start = time.perf_counter()
                recovery_patch = _run_openai_operation_with_transport_recovery(
                    patcher_for_attempt=patcher_for_attempt,
                    operation_name="propose_patch_context_recovery",
                    invoke=lambda patcher_for_attempt=patcher_for_attempt,
                    recovery_prompt=recovery_prompt,
                    attempt_dir=attempt_dir,
                    workdir=workdir,
                    snapshot_full_text=snapshot_full_text,
                    snapshot_artifact_text=snapshot_artifact_text,
                    attempt_shared_propose_deadline_monotonic=attempt_shared_propose_deadline_monotonic: (
                        _propose_patch_with_snapshot_context(
                            patcher=patcher_for_attempt,
                            prompt=recovery_prompt,
                            context_dir=attempt_dir,
                            cwd=workdir,
                            snapshot_full_text=snapshot_full_text,
                            snapshot_artifact_text=snapshot_artifact_text,
                            attempt_deadline_monotonic=attempt_shared_propose_deadline_monotonic,
                        )
                    ),
                    classify_failure_kind=_transport_recovery_failure_kind_from_meta,
                    attempt=attempt,
                    attempt_metadata=attempt_metadata,
                    attempt_context_protocol=attempt_context_protocol,
                    before=current,
                    cumulative_runtime_sec_value=cumulative_runtime_sec,
                )
                recovery_patch_duration_sec = time.perf_counter() - recovery_patch_start

                recovery_usage_meta: dict[str, Any] | None = None
                for key in ("error", "status", "body", "exc"):
                    patch_meta.pop(key, None)
                if isinstance(recovery_patch.meta, dict):
                    recovery_usage_meta = dict(recovery_patch.meta)
                    patch_meta.update(recovery_usage_meta)
                    patch_meta["context_recovery_meta"] = _summarize_recovery_usage_meta(
                        recovery_usage_meta
                    )
                elif recovery_patch.meta is not None:
                    patch_meta["context_recovery_raw_meta"] = recovery_patch.meta
                patch_meta["propose_duration_sec"] = (
                    patch_duration_sec + recovery_patch_duration_sec
                )
                patch_meta["context_recovery_invoked"] = True
                patch_meta["context_recovery_kind"] = request_failure_kind
                patch_meta["context_recovery_prompt_chars_before"] = len(cycle_prompt)
                patch_meta["context_recovery_prompt_chars_after"] = len(recovery_prompt)
                patch_meta["context_recovery_duration_sec"] = recovery_patch_duration_sec
                (
                    recovery_llm_calls,
                    recovery_tokens_prompt,
                    recovery_tokens_completion,
                    recovery_tokens_total,
                ) = _aggregate_attempt_llm_usage(
                    stage_a_meta=None,
                    patch_meta=recovery_usage_meta,
                    stage_a_default_llm_calls=0,
                    patch_default_llm_calls=1 if is_openai_patcher else 0,
                )
                attempt_llm_calls_cumulative += recovery_llm_calls
                attempt_tokens_prompt_cumulative += recovery_tokens_prompt
                attempt_tokens_completion_cumulative += recovery_tokens_completion
                attempt_tokens_total_cumulative += recovery_tokens_total
                (
                    recovery_tokens_reasoning,
                    recovery_reasoning_applicable,
                    recovery_reasoning_available,
                ) = _reasoning_usage_segment_from_meta(
                    recovery_usage_meta,
                    default_llm_calls=1 if is_openai_patcher else 0,
                )
                if recovery_reasoning_applicable:
                    attempt_reasoning_token_usage_any = True
                    if recovery_reasoning_available:
                        attempt_tokens_reasoning_cumulative += recovery_tokens_reasoning
                    else:
                        attempt_reasoning_token_usage_all_available = False
                recovery_prompt_chars, recovery_completion_chars = _patch_attempt_char_deltas(
                    cycle_prompt=recovery_prompt,
                    diff=recovery_patch.diff,
                    patch_meta=recovery_usage_meta,
                    is_baseline_patcher=is_baseline_patcher,
                )
                attempt_prompt_chars_cumulative += recovery_prompt_chars
                attempt_completion_chars_cumulative += recovery_completion_chars
                budget_local_llm_calls += recovery_llm_calls
                budget_local_tokens_prompt += recovery_tokens_prompt
                budget_local_tokens_completion += recovery_tokens_completion
                budget_local_tokens_total += recovery_tokens_total
                budget_local_wall_sec += recovery_patch_duration_sec
                budget_finalize_kwargs["local_accounting"] = _adaptive_policy_local_accounting(
                    llm_call_count=budget_local_llm_calls,
                    tokens_prompt=budget_local_tokens_prompt,
                    tokens_completion=budget_local_tokens_completion,
                    tokens_total=budget_local_tokens_total,
                    wall_sec=budget_local_wall_sec,
                )
                _apply_usage_totals_to_patch_meta(
                    patch_meta=patch_meta,
                    llm_call_count=attempt_llm_calls_cumulative,
                    tokens_prompt=attempt_tokens_prompt_cumulative,
                    tokens_completion=attempt_tokens_completion_cumulative,
                    tokens_total=attempt_tokens_total_cumulative,
                    tokens_reasoning=attempt_tokens_reasoning_cumulative,
                    reasoning_token_usage_available=(
                        attempt_reasoning_token_usage_any
                        and attempt_reasoning_token_usage_all_available
                    ),
                )
                _sync_attempt_usage_metadata(
                    attempt_metadata=attempt_metadata,
                    patch_meta=patch_meta,
                    llm_call_count=attempt_llm_calls_cumulative,
                )
                _sync_attempt_prompt_char_metadata(
                    attempt_metadata=attempt_metadata,
                    patch_meta=patch_meta,
                    prompt_chars_total=attempt_prompt_chars_cumulative,
                    completion_chars_total=attempt_completion_chars_cumulative,
                )
                _ensure_deterministic_seed_ack(
                    deterministic=deterministic,
                    seed=seed,
                    patcher_name=patcher_name,
                    patch_meta=patch_meta,
                    case_id=case.case_id,
                    condition=condition.value,
                    attempt=attempt,
                )
                diff_to_apply = recovery_patch.diff
                patch_meta["context_recovery_retry_success"] = diff_to_apply is not None
                request_failed, request_failure_kind = _classify_patcher_request_failure(patch_meta)
                request_failure_error = str(patch_meta.get("error") or "")
                request_failure_status = patch_meta.get("status")
                request_failure_status_int = (
                    int(request_failure_status) if isinstance(request_failure_status, int) else None
                )
                request_failure_exc = str(patch_meta.get("exc") or "")
                _sync_patcher_request_failure_metadata(
                    attempt_metadata=attempt_metadata,
                    patch_meta=patch_meta,
                    failed=request_failed,
                    kind=request_failure_kind if request_failed else "",
                    error=request_failure_error if request_failed else "",
                    status=request_failure_status_int if request_failed else None,
                    exc=request_failure_exc if request_failed else "",
                )
            if (
                diff_to_apply is None
                and is_openai_patcher
                and (not request_failed)
                and str(patch_meta.get("error") or "").strip()
            ):
                no_diff_recovery_sections = _build_stage_b_sections_for_attempt(
                    include_stage_a_evidence_index_value=True,
                    include_project_files_value=False,
                    compacted_context_value=compacted_context_for_prompt,
                    include_snapshot_value=include_snapshot_in_prompt,
                    include_pytest_output_value=include_pytest_output_in_prompt,
                    include_stage_a_context_value=include_stage_a_context_in_prompt,
                    retry_repair_plan_value=retry_plan_text,
                    enable_tep_value=enable_tep and tep_nudge_escalation,
                    stagnation_streak_value=stagnation_streak,
                )
                no_diff_recovery_sections.append(
                    {
                        "key": "no_diff_recovery",
                        "priority": 3,
                        "text": (
                            "Retry directive: the previous response did not produce an applicable "
                            "patch. Return one concrete minimal diff that changes source behavior "
                            "in traceback-linked non-test files whenever possible.\n"
                        ),
                    }
                )
                no_diff_recovery_budget_result = _apply_prompt_budget(
                    sections=no_diff_recovery_sections,
                    budget=CONTEXT_RECOVERY_PROMPT_BUDGET,
                    is_delta_retry=is_delta_retry,
                )
                no_diff_recovery_prompt = (
                    no_diff_recovery_budget_result.prompt + syntax_retry_feedback
                )
                no_diff_recovery_start = time.perf_counter()
                no_diff_recovery_patch = _run_openai_operation_with_transport_recovery(
                    patcher_for_attempt=patcher_for_attempt,
                    operation_name="propose_patch_no_diff_recovery",
                    invoke=lambda patcher_for_attempt=patcher_for_attempt,
                    no_diff_recovery_prompt=no_diff_recovery_prompt,
                    attempt_dir=attempt_dir,
                    workdir=workdir,
                    snapshot_full_text=snapshot_full_text,
                    snapshot_artifact_text=snapshot_artifact_text,
                    attempt_shared_propose_deadline_monotonic=attempt_shared_propose_deadline_monotonic: (
                        _propose_patch_with_snapshot_context(
                            patcher=patcher_for_attempt,
                            prompt=no_diff_recovery_prompt,
                            context_dir=attempt_dir,
                            cwd=workdir,
                            snapshot_full_text=snapshot_full_text,
                            snapshot_artifact_text=snapshot_artifact_text,
                            attempt_deadline_monotonic=attempt_shared_propose_deadline_monotonic,
                        )
                    ),
                    classify_failure_kind=_transport_recovery_failure_kind_from_meta,
                    attempt=attempt,
                    attempt_metadata=attempt_metadata,
                    attempt_context_protocol=attempt_context_protocol,
                    before=current,
                    cumulative_runtime_sec_value=cumulative_runtime_sec,
                )
                no_diff_recovery_duration_sec = time.perf_counter() - no_diff_recovery_start

                no_diff_recovery_usage_meta: dict[str, Any] | None = None
                for key in ("error", "status", "body", "exc"):
                    patch_meta.pop(key, None)
                if isinstance(no_diff_recovery_patch.meta, dict):
                    no_diff_recovery_usage_meta = dict(no_diff_recovery_patch.meta)
                    patch_meta.update(no_diff_recovery_usage_meta)
                    patch_meta["no_diff_recovery_meta"] = _summarize_recovery_usage_meta(
                        no_diff_recovery_usage_meta
                    )
                elif no_diff_recovery_patch.meta is not None:
                    patch_meta["no_diff_recovery_raw_meta"] = no_diff_recovery_patch.meta
                existing_propose_duration = float(
                    patch_meta.get("propose_duration_sec", patch_duration_sec) or patch_duration_sec
                )
                patch_meta["propose_duration_sec"] = (
                    existing_propose_duration + no_diff_recovery_duration_sec
                )
                patch_meta["no_diff_recovery_invoked"] = True
                patch_meta["no_diff_recovery_prompt_chars_before"] = len(cycle_prompt)
                patch_meta["no_diff_recovery_prompt_chars_after"] = len(no_diff_recovery_prompt)
                patch_meta["no_diff_recovery_duration_sec"] = no_diff_recovery_duration_sec
                (
                    no_diff_recovery_llm_calls,
                    no_diff_recovery_tokens_prompt,
                    no_diff_recovery_tokens_completion,
                    no_diff_recovery_tokens_total,
                ) = _aggregate_attempt_llm_usage(
                    stage_a_meta=None,
                    patch_meta=no_diff_recovery_usage_meta,
                    stage_a_default_llm_calls=0,
                    patch_default_llm_calls=1 if is_openai_patcher else 0,
                )
                attempt_llm_calls_cumulative += no_diff_recovery_llm_calls
                attempt_tokens_prompt_cumulative += no_diff_recovery_tokens_prompt
                attempt_tokens_completion_cumulative += no_diff_recovery_tokens_completion
                attempt_tokens_total_cumulative += no_diff_recovery_tokens_total
                (
                    no_diff_recovery_tokens_reasoning,
                    no_diff_reasoning_applicable,
                    no_diff_reasoning_available,
                ) = _reasoning_usage_segment_from_meta(
                    no_diff_recovery_usage_meta,
                    default_llm_calls=1 if is_openai_patcher else 0,
                )
                if no_diff_reasoning_applicable:
                    attempt_reasoning_token_usage_any = True
                    if no_diff_reasoning_available:
                        attempt_tokens_reasoning_cumulative += no_diff_recovery_tokens_reasoning
                    else:
                        attempt_reasoning_token_usage_all_available = False
                no_diff_prompt_chars, no_diff_completion_chars = _patch_attempt_char_deltas(
                    cycle_prompt=no_diff_recovery_prompt,
                    diff=no_diff_recovery_patch.diff,
                    patch_meta=no_diff_recovery_usage_meta,
                    is_baseline_patcher=is_baseline_patcher,
                )
                attempt_prompt_chars_cumulative += no_diff_prompt_chars
                attempt_completion_chars_cumulative += no_diff_completion_chars
                budget_local_llm_calls += no_diff_recovery_llm_calls
                budget_local_tokens_prompt += no_diff_recovery_tokens_prompt
                budget_local_tokens_completion += no_diff_recovery_tokens_completion
                budget_local_tokens_total += no_diff_recovery_tokens_total
                budget_local_wall_sec += no_diff_recovery_duration_sec
                budget_finalize_kwargs["local_accounting"] = _adaptive_policy_local_accounting(
                    llm_call_count=budget_local_llm_calls,
                    tokens_prompt=budget_local_tokens_prompt,
                    tokens_completion=budget_local_tokens_completion,
                    tokens_total=budget_local_tokens_total,
                    wall_sec=budget_local_wall_sec,
                )
                _apply_usage_totals_to_patch_meta(
                    patch_meta=patch_meta,
                    llm_call_count=attempt_llm_calls_cumulative,
                    tokens_prompt=attempt_tokens_prompt_cumulative,
                    tokens_completion=attempt_tokens_completion_cumulative,
                    tokens_total=attempt_tokens_total_cumulative,
                    tokens_reasoning=attempt_tokens_reasoning_cumulative,
                    reasoning_token_usage_available=(
                        attempt_reasoning_token_usage_any
                        and attempt_reasoning_token_usage_all_available
                    ),
                )
                _sync_attempt_usage_metadata(
                    attempt_metadata=attempt_metadata,
                    patch_meta=patch_meta,
                    llm_call_count=attempt_llm_calls_cumulative,
                )
                _sync_attempt_prompt_char_metadata(
                    attempt_metadata=attempt_metadata,
                    patch_meta=patch_meta,
                    prompt_chars_total=attempt_prompt_chars_cumulative,
                    completion_chars_total=attempt_completion_chars_cumulative,
                )
                _ensure_deterministic_seed_ack(
                    deterministic=deterministic,
                    seed=seed,
                    patcher_name=patcher_name,
                    patch_meta=patch_meta,
                    case_id=case.case_id,
                    condition=condition.value,
                    attempt=attempt,
                )
                diff_to_apply = no_diff_recovery_patch.diff
                patch_meta["no_diff_recovery_retry_success"] = diff_to_apply is not None
                request_failed, request_failure_kind = _classify_patcher_request_failure(patch_meta)
                request_failure_error = str(patch_meta.get("error") or "")
                request_failure_status = patch_meta.get("status")
                request_failure_status_int = (
                    int(request_failure_status) if isinstance(request_failure_status, int) else None
                )
                request_failure_exc = str(patch_meta.get("exc") or "")
                _sync_patcher_request_failure_metadata(
                    attempt_metadata=attempt_metadata,
                    patch_meta=patch_meta,
                    failed=request_failed,
                    kind=request_failure_kind if request_failed else "",
                    error=request_failure_error if request_failed else "",
                    status=request_failure_status_int if request_failed else None,
                    exc=request_failure_exc if request_failed else "",
                )
            if diff_to_apply is None and runner_force_patch_attempt_enabled:
                if request_failed:
                    _sync_patcher_request_failure_metadata(
                        attempt_metadata=attempt_metadata,
                        patch_meta=patch_meta,
                        failed=True,
                        kind=request_failure_kind,
                        error=request_failure_error,
                        status=request_failure_status_int,
                        exc=request_failure_exc,
                        runner_forced_patch_skipped=True,
                        runner_forced_patch_skip_reason=(
                            "compatibility_failure"
                            if request_failure_kind == "compat_reasoning_only"
                            else "transport_or_context_failure"
                        ),
                    )
                else:
                    force_start = time.perf_counter()
                    forced = _build_runner_forced_attempt_diff(
                        cwd=workdir, file_index=file_index, attempt=attempt
                    )
                    patch_meta["runner_forced_patch_build_sec"] = time.perf_counter() - force_start
                    if forced is not None:
                        forced_target, diff_to_apply = forced
                        patch_meta["runner_forced_patch_attempt"] = True
                        patch_meta["runner_forced_patch_target"] = forced_target
                        patch_meta["runner_forced_patch_reason"] = "no_patch_proposed"
            no_diff_note_override = ""
            if diff_to_apply is not None and _is_forced_noop_marker_diff(diff_to_apply):
                patch_meta["runner_forced_patch_marker_only"] = True
                patch_meta["runner_forced_patch_disabled_after_noop"] = True
                attempt_metadata["runner_forced_patch_marker_only"] = True
                attempt_metadata["runner_forced_patch_disabled_after_noop"] = True
                runner_force_patch_attempt_enabled = False
                diff_to_apply = None
                no_diff_note_override = "forced_noop_patch"
            if diff_to_apply is None:
                no_diff_note = (
                    no_diff_note_override
                    if no_diff_note_override
                    else ("patcher_request_failed" if request_failed else "no_patch_proposed")
                )
                fail_fast_on_budget_exceeded = bool(
                    normalized_case_fail_fast_on_request_budget_exceeded
                    and request_failed
                    and request_failure_error == "request_budget_exceeded"
                )
                if fail_fast_on_budget_exceeded:
                    patch_meta["case_fail_fast_triggered"] = True
                    patch_meta["case_fail_fast_reason"] = "request_budget_exceeded"
                    attempt_metadata["case_fail_fast_triggered"] = True
                    attempt_metadata["case_fail_fast_reason"] = "request_budget_exceeded"
                _refresh_adaptive_progress_for_note(
                    note=no_diff_note,
                    success=False,
                    signature_repeated=True,
                )
                (
                    adaptive_progress_stop_candidate_for_note,
                    adaptive_progress_stop_reason_for_note,
                    adaptive_no_progress_token_sum,
                    adaptive_no_progress_token_cap_hit,
                ) = _apply_no_progress_token_cap_candidate(
                    is_adaptive_condition=is_adaptive_condition,
                    adaptive_progress_made=adaptive_progress_made,
                    adaptive_no_progress_token_sum=adaptive_no_progress_token_sum,
                    attempt_tokens_total=_nonnegative_int(
                        attempt_metadata.get("tokens_total"), default=0
                    ),
                    adaptive_no_progress_token_cap=normalized_adaptive_no_progress_token_cap,
                    adaptive_progress_stop_candidate=adaptive_progress_stop_candidate,
                    adaptive_progress_stop_reason=adaptive_progress_stop_reason,
                )
                patch_meta["adaptive_no_progress_token_sum"] = adaptive_no_progress_token_sum
                patch_meta["adaptive_no_progress_token_cap"] = (
                    normalized_adaptive_no_progress_token_cap
                )
                patch_meta["adaptive_no_progress_token_cap_hit"] = (
                    adaptive_no_progress_token_cap_hit
                )
                adaptive_stop_triggered = _mark_adaptive_progress_stop(
                    attempt_metadata=attempt_metadata,
                    patch_meta=patch_meta,
                    is_adaptive_condition=is_adaptive_condition,
                    adaptive_progress_stop_candidate=adaptive_progress_stop_candidate_for_note,
                    adaptive_progress_stop_reason=adaptive_progress_stop_reason_for_note,
                    note=no_diff_note,
                )
                _sync_syntax_preflight_metadata(
                    attempt_metadata=attempt_metadata,
                    patch_meta=patch_meta,
                    enabled=normalized_syntax_preflight_enabled,
                    cycles_used=syntax_preflight_cycles_used,
                    failures=syntax_preflight_failures,
                    last_error_type=syntax_preflight_last_error_type,
                    last_error_msg=syntax_preflight_last_error_msg,
                    events=syntax_preflight_events,
                )
                attempt_wall_sec = _measure_attempt_wall_clock_sec(attempt_start)
                cumulative_runtime_sec += attempt_wall_sec
                pending_credit = _finalize_adaptive_policy_attempt(
                    note=no_diff_note,
                    success=False,
                    attempt_wall_sec=attempt_wall_sec,
                    online_finalize_kwargs=online_finalize_kwargs,
                    budget_finalize_kwargs=budget_finalize_kwargs,
                )
                record = _build_attempt_outcome_record(
                    case=case,
                    condition=condition,
                    run_id=run_id,
                    attempt=attempt,
                    patcher_name=patcher_name,
                    run_metadata=effective_run_metadata,
                    attempt_metadata=attempt_metadata,
                    patcher=patcher_for_attempt,
                    patch_meta=patch_meta,
                    context_protocol=attempt_context_protocol,
                    success=False,
                    note=no_diff_note,
                    before=current,
                    cumulative_runtime_sec=cumulative_runtime_sec,
                    attempt_wall_clock_sec=attempt_wall_sec,
                )
                _queue_pending_online_credit(pending_credit=pending_credit, record=record)
                _append_record(record, defer_sink=pending_credit is not None)
                _mark_scored_attempt_consumed()
                _capture_attempt_history(
                    attempt=attempt,
                    note=no_diff_note,
                    failure_signature=failure_signature,
                    record=record,
                    patch_meta=patch_meta,
                    retry_plan_value=retry_plan_text,
                    snapshot_for_history=snapshot,
                )
                previous_failure_context = _enrich_failure_context(
                    base=current_failure_context,
                    note=no_diff_note,
                    patch_meta=patch_meta,
                    run=current,
                )
                attempt_terminated_early = True
                should_break_after_attempt = adaptive_stop_triggered or fail_fast_on_budget_exceeded
                break

            _sync_patch_diff_metadata(patch_meta=patch_meta, diff=diff_to_apply)
            patch_path = attempt_dir / "patch.diff"
            if syntax_cycle > 1:
                patch_path = attempt_dir / f"patch_cycle_{syntax_cycle:02d}.diff"
            patch_path.write_text(diff_to_apply, encoding="utf-8")

            patch_verifier_scope = patch_verifier_scope_for_attempt
            patch_verifier_enabled = patch_verifier_scope != PATCH_VERIFIER_SCOPE_DISABLED
            patch_verifier_passed: bool | None = None
            patch_verifier_score: int | None = None
            patch_verifier_reasons: list[str] = []
            patch_verifier_meta: dict[str, Any] = {}
            if patch_verifier_enabled:
                patch_verifier_passed, patch_verifier_meta = _score_patch_relevance(
                    diff=diff_to_apply,
                    cwd=workdir,
                    snapshot=snapshot,
                    stdout=current.stdout,
                    stderr=current.stderr,
                    min_score=normalized_patch_verifier_min_score,
                    infer_test_source_scope=normalized_patch_verifier_infer_test_source_scope,
                    file_index_lines=file_index_lines,
                )
                patch_meta.update(patch_verifier_meta)
                patch_verifier_score = int(patch_verifier_meta.get("patch_verifier_score", 0))
                raw_reasons = patch_verifier_meta.get("patch_verifier_reasons")
                if isinstance(raw_reasons, list):
                    patch_verifier_reasons = [str(item) for item in raw_reasons]
            _sync_patch_verifier_metadata(
                attempt_metadata=attempt_metadata,
                patch_meta=patch_meta,
                enabled=patch_verifier_enabled,
                scope=patch_verifier_scope,
                passed=patch_verifier_passed,
                score=patch_verifier_score,
                reasons=patch_verifier_reasons,
                min_score=normalized_patch_verifier_min_score,
            )
            if patch_verifier_enabled and patch_verifier_passed is False:
                initial_patch_verifier_score = patch_verifier_score
                initial_patch_verifier_reasons = list(patch_verifier_reasons)
                patch_verifier_patch_files_before_raw = patch_meta.get("patch_verifier_patch_files")
                patch_verifier_patch_files_before = (
                    [str(item) for item in patch_verifier_patch_files_before_raw]
                    if isinstance(patch_verifier_patch_files_before_raw, list)
                    else []
                )
                patch_verifier_scope_before = bool(
                    patch_meta.get("patch_verifier_source_scope_matched")
                )
                patch_verifier_recovery_eligible = bool(
                    patch_meta.get("patch_verifier_substantive_change")
                )
                _sync_patch_verifier_recovery_metadata(
                    attempt_metadata=attempt_metadata,
                    patch_meta=patch_meta,
                    eligible=patch_verifier_recovery_eligible,
                    invoked=False,
                    result=PATCH_VERIFIER_RECOVERY_RESULT_NOT_ELIGIBLE,
                    score_before=initial_patch_verifier_score,
                    reasons_before=initial_patch_verifier_reasons,
                    patch_files_before=patch_verifier_patch_files_before,
                    source_scope_matched_before=patch_verifier_scope_before,
                )
                if patch_verifier_recovery_eligible:
                    remaining_recovery_budget_sec = _remaining_attempt_propose_budget_sec(
                        attempt_shared_propose_deadline_monotonic
                    )
                    if (
                        remaining_recovery_budget_sec is not None
                        and remaining_recovery_budget_sec <= 0
                    ):
                        _sync_patch_verifier_recovery_metadata(
                            attempt_metadata=attempt_metadata,
                            patch_meta=patch_meta,
                            eligible=True,
                            invoked=False,
                            result=PATCH_VERIFIER_RECOVERY_RESULT_SKIPPED_DEADLINE_EXHAUSTED,
                            prompt_chars_before=len(cycle_prompt),
                            score_before=initial_patch_verifier_score,
                            reasons_before=initial_patch_verifier_reasons,
                            patch_files_before=patch_verifier_patch_files_before,
                            source_scope_matched_before=patch_verifier_scope_before,
                        )
                    else:
                        verifier_recovery_sections = _build_stage_b_sections_for_attempt(
                            include_stage_a_evidence_index_value=True,
                            include_project_files_value=False,
                            compacted_context_value=compacted_context_for_prompt,
                            include_snapshot_value=include_snapshot_in_prompt,
                            include_pytest_output_value=include_pytest_output_in_prompt,
                            include_stage_a_context_value=include_stage_a_context_in_prompt,
                            retry_repair_plan_value=retry_plan_text,
                            enable_tep_value=enable_tep and tep_nudge_escalation,
                            stagnation_streak_value=stagnation_streak,
                        )
                        inferred_source_scope_raw = patch_meta.get(
                            "patch_verifier_inferred_source_scope"
                        )
                        inferred_source_scope = (
                            [str(item) for item in inferred_source_scope_raw]
                            if isinstance(inferred_source_scope_raw, list)
                            else []
                        )
                        verifier_recovery_sections.append(
                            {
                                "key": "patch_verifier_recovery",
                                "priority": 3,
                                "text": _render_patch_verifier_recovery_feedback(
                                    score=patch_verifier_score,
                                    min_score=normalized_patch_verifier_min_score,
                                    reasons=patch_verifier_reasons,
                                    patch_files=patch_verifier_patch_files_before,
                                    crash_file=str(
                                        patch_meta.get("patch_verifier_crash_file") or ""
                                    ),
                                    inferred_source_scope=inferred_source_scope,
                                ),
                            }
                        )
                        verifier_recovery_budget_result = _apply_prompt_budget(
                            sections=verifier_recovery_sections,
                            budget=CONTEXT_RECOVERY_PROMPT_BUDGET,
                            is_delta_retry=is_delta_retry,
                        )
                        verifier_recovery_prompt = (
                            verifier_recovery_budget_result.prompt + syntax_retry_feedback
                        )
                        verifier_recovery_start = time.perf_counter()
                        verifier_recovery_patch = _run_openai_operation_with_transport_recovery(
                            patcher_for_attempt=patcher_for_attempt,
                            operation_name="propose_patch_verifier_recovery",
                            invoke=lambda patcher_for_attempt=patcher_for_attempt,
                            verifier_recovery_prompt=verifier_recovery_prompt,
                            attempt_dir=attempt_dir,
                            workdir=workdir,
                            snapshot_full_text=snapshot_full_text,
                            snapshot_artifact_text=snapshot_artifact_text,
                            attempt_shared_propose_deadline_monotonic=attempt_shared_propose_deadline_monotonic: (
                                _propose_patch_with_snapshot_context(
                                    patcher=patcher_for_attempt,
                                    prompt=verifier_recovery_prompt,
                                    context_dir=attempt_dir,
                                    cwd=workdir,
                                    snapshot_full_text=snapshot_full_text,
                                    snapshot_artifact_text=snapshot_artifact_text,
                                    attempt_deadline_monotonic=attempt_shared_propose_deadline_monotonic,
                                )
                            ),
                            classify_failure_kind=_transport_recovery_failure_kind_from_meta,
                            attempt=attempt,
                            attempt_metadata=attempt_metadata,
                            attempt_context_protocol=attempt_context_protocol,
                            before=current,
                            cumulative_runtime_sec_value=cumulative_runtime_sec,
                        )
                        verifier_recovery_duration_sec = (
                            time.perf_counter() - verifier_recovery_start
                        )

                        verifier_recovery_usage_meta: dict[str, Any] | None = None
                        verifier_recovery_raw_meta: Any = verifier_recovery_patch.meta
                        if isinstance(verifier_recovery_patch.meta, dict):
                            verifier_recovery_usage_meta = dict(verifier_recovery_patch.meta)
                            _ensure_deterministic_seed_ack(
                                deterministic=deterministic,
                                seed=seed,
                                patcher_name=patcher_name,
                                patch_meta=verifier_recovery_usage_meta,
                                case_id=case.case_id,
                                condition=condition.value,
                                attempt=attempt,
                            )
                            verifier_recovery_raw_meta = dict(verifier_recovery_usage_meta)
                        (
                            verifier_recovery_llm_calls,
                            verifier_recovery_tokens_prompt,
                            verifier_recovery_tokens_completion,
                            verifier_recovery_tokens_total,
                        ) = _aggregate_attempt_llm_usage(
                            stage_a_meta=None,
                            patch_meta=verifier_recovery_usage_meta,
                            stage_a_default_llm_calls=0,
                            patch_default_llm_calls=1 if is_openai_patcher else 0,
                        )
                        attempt_llm_calls_cumulative += verifier_recovery_llm_calls
                        attempt_tokens_prompt_cumulative += verifier_recovery_tokens_prompt
                        attempt_tokens_completion_cumulative += verifier_recovery_tokens_completion
                        attempt_tokens_total_cumulative += verifier_recovery_tokens_total
                        (
                            verifier_recovery_tokens_reasoning,
                            verifier_reasoning_applicable,
                            verifier_reasoning_available,
                        ) = _reasoning_usage_segment_from_meta(
                            verifier_recovery_usage_meta,
                            default_llm_calls=1 if is_openai_patcher else 0,
                        )
                        if verifier_reasoning_applicable:
                            attempt_reasoning_token_usage_any = True
                            if verifier_reasoning_available:
                                attempt_tokens_reasoning_cumulative += (
                                    verifier_recovery_tokens_reasoning
                                )
                            else:
                                attempt_reasoning_token_usage_all_available = False
                        verifier_prompt_chars, verifier_completion_chars = (
                            _patch_attempt_char_deltas(
                                cycle_prompt=verifier_recovery_prompt,
                                diff=verifier_recovery_patch.diff,
                                patch_meta=verifier_recovery_usage_meta,
                                is_baseline_patcher=is_baseline_patcher,
                            )
                        )
                        attempt_prompt_chars_cumulative += verifier_prompt_chars
                        attempt_completion_chars_cumulative += verifier_completion_chars
                        budget_local_llm_calls += verifier_recovery_llm_calls
                        budget_local_tokens_prompt += verifier_recovery_tokens_prompt
                        budget_local_tokens_completion += verifier_recovery_tokens_completion
                        budget_local_tokens_total += verifier_recovery_tokens_total
                        budget_local_wall_sec += verifier_recovery_duration_sec
                        budget_finalize_kwargs["local_accounting"] = (
                            _adaptive_policy_local_accounting(
                                llm_call_count=budget_local_llm_calls,
                                tokens_prompt=budget_local_tokens_prompt,
                                tokens_completion=budget_local_tokens_completion,
                                tokens_total=budget_local_tokens_total,
                                wall_sec=budget_local_wall_sec,
                            )
                        )
                        existing_propose_duration = float(
                            patch_meta.get("propose_duration_sec", patch_duration_sec)
                            or patch_duration_sec
                        )
                        patch_meta["propose_duration_sec"] = (
                            existing_propose_duration + verifier_recovery_duration_sec
                        )
                        verifier_recovery_request_failed, _ = _classify_patcher_request_failure(
                            verifier_recovery_usage_meta or {}
                        )
                        verifier_recovery_result = (
                            PATCH_VERIFIER_RECOVERY_RESULT_REQUEST_FAILED
                            if verifier_recovery_request_failed
                            else PATCH_VERIFIER_RECOVERY_RESULT_NO_DIFF
                        )
                        verifier_recovery_diff = verifier_recovery_patch.diff
                        if verifier_recovery_diff is not None:
                            patch_meta.update(verifier_recovery_usage_meta or {})
                            patch_meta["propose_duration_sec"] = (
                                existing_propose_duration + verifier_recovery_duration_sec
                            )
                            _sync_patch_diff_metadata(
                                patch_meta=patch_meta,
                                diff=verifier_recovery_diff,
                            )
                            patch_path.write_text(verifier_recovery_diff, encoding="utf-8")
                            patch_verifier_passed, patch_verifier_meta = _score_patch_relevance(
                                diff=verifier_recovery_diff,
                                cwd=workdir,
                                snapshot=snapshot,
                                stdout=current.stdout,
                                stderr=current.stderr,
                                min_score=normalized_patch_verifier_min_score,
                                infer_test_source_scope=(
                                    normalized_patch_verifier_infer_test_source_scope
                                ),
                                file_index_lines=file_index_lines,
                            )
                            patch_meta.update(patch_verifier_meta)
                            patch_verifier_score = int(
                                patch_verifier_meta.get("patch_verifier_score", 0)
                            )
                            raw_reasons = patch_verifier_meta.get("patch_verifier_reasons")
                            patch_verifier_reasons = (
                                [str(item) for item in raw_reasons]
                                if isinstance(raw_reasons, list)
                                else []
                            )
                            _sync_patch_verifier_metadata(
                                attempt_metadata=attempt_metadata,
                                patch_meta=patch_meta,
                                enabled=patch_verifier_enabled,
                                scope=patch_verifier_scope,
                                passed=patch_verifier_passed,
                                score=patch_verifier_score,
                                reasons=patch_verifier_reasons,
                                min_score=normalized_patch_verifier_min_score,
                            )
                            _sync_patcher_request_failure_metadata(
                                attempt_metadata=attempt_metadata,
                                patch_meta=patch_meta,
                                failed=False,
                            )
                            diff_to_apply = verifier_recovery_diff
                            verifier_recovery_result = (
                                PATCH_VERIFIER_RECOVERY_RESULT_PASSED_VERIFIER
                                if patch_verifier_passed
                                else PATCH_VERIFIER_RECOVERY_RESULT_STILL_REJECTED
                            )
                        _sync_patch_verifier_recovery_metadata(
                            attempt_metadata=attempt_metadata,
                            patch_meta=patch_meta,
                            eligible=True,
                            invoked=True,
                            result=verifier_recovery_result,
                            prompt_chars_before=len(cycle_prompt),
                            prompt_chars_after=len(verifier_recovery_prompt),
                            duration_sec=verifier_recovery_duration_sec,
                            score_before=initial_patch_verifier_score,
                            reasons_before=initial_patch_verifier_reasons,
                            patch_files_before=patch_verifier_patch_files_before,
                            source_scope_matched_before=patch_verifier_scope_before,
                            raw_meta=verifier_recovery_raw_meta,
                        )
                        _apply_usage_totals_to_patch_meta(
                            patch_meta=patch_meta,
                            llm_call_count=attempt_llm_calls_cumulative,
                            tokens_prompt=attempt_tokens_prompt_cumulative,
                            tokens_completion=attempt_tokens_completion_cumulative,
                            tokens_total=attempt_tokens_total_cumulative,
                            tokens_reasoning=attempt_tokens_reasoning_cumulative,
                            reasoning_token_usage_available=(
                                attempt_reasoning_token_usage_any
                                and attempt_reasoning_token_usage_all_available
                            ),
                        )
                        _sync_attempt_usage_metadata(
                            attempt_metadata=attempt_metadata,
                            patch_meta=patch_meta,
                            llm_call_count=attempt_llm_calls_cumulative,
                        )
                        _sync_attempt_prompt_char_metadata(
                            attempt_metadata=attempt_metadata,
                            patch_meta=patch_meta,
                            prompt_chars_total=attempt_prompt_chars_cumulative,
                            completion_chars_total=attempt_completion_chars_cumulative,
                        )
            _sync_syntax_preflight_metadata(
                attempt_metadata=attempt_metadata,
                patch_meta=patch_meta,
                enabled=normalized_syntax_preflight_enabled,
                cycles_used=syntax_preflight_cycles_used,
                failures=syntax_preflight_failures,
                last_error_type=syntax_preflight_last_error_type,
                last_error_msg=syntax_preflight_last_error_msg,
                events=syntax_preflight_events,
            )
            if patch_verifier_enabled and patch_verifier_passed is False:
                _refresh_adaptive_progress_for_note(
                    note="patch_verifier_rejected",
                    success=False,
                    signature_repeated=True,
                )
                (
                    adaptive_progress_stop_candidate_for_note,
                    adaptive_progress_stop_reason_for_note,
                    adaptive_no_progress_token_sum,
                    adaptive_no_progress_token_cap_hit,
                ) = _apply_no_progress_token_cap_candidate(
                    is_adaptive_condition=is_adaptive_condition,
                    adaptive_progress_made=adaptive_progress_made,
                    adaptive_no_progress_token_sum=adaptive_no_progress_token_sum,
                    attempt_tokens_total=_nonnegative_int(
                        attempt_metadata.get("tokens_total"), default=0
                    ),
                    adaptive_no_progress_token_cap=normalized_adaptive_no_progress_token_cap,
                    adaptive_progress_stop_candidate=adaptive_progress_stop_candidate,
                    adaptive_progress_stop_reason=adaptive_progress_stop_reason,
                )
                patch_meta["adaptive_no_progress_token_sum"] = adaptive_no_progress_token_sum
                patch_meta["adaptive_no_progress_token_cap"] = (
                    normalized_adaptive_no_progress_token_cap
                )
                patch_meta["adaptive_no_progress_token_cap_hit"] = (
                    adaptive_no_progress_token_cap_hit
                )
                adaptive_stop_triggered = _mark_adaptive_progress_stop(
                    attempt_metadata=attempt_metadata,
                    patch_meta=patch_meta,
                    is_adaptive_condition=is_adaptive_condition,
                    adaptive_progress_stop_candidate=adaptive_progress_stop_candidate_for_note,
                    adaptive_progress_stop_reason=adaptive_progress_stop_reason_for_note,
                    note="patch_verifier_rejected",
                )
                attempt_wall_sec = _measure_attempt_wall_clock_sec(attempt_start)
                cumulative_runtime_sec += attempt_wall_sec
                pending_credit = _finalize_adaptive_policy_attempt(
                    note="patch_verifier_rejected",
                    success=False,
                    attempt_wall_sec=attempt_wall_sec,
                    online_finalize_kwargs=online_finalize_kwargs,
                    budget_finalize_kwargs=budget_finalize_kwargs,
                )
                record = _build_attempt_outcome_record(
                    case=case,
                    condition=condition,
                    run_id=run_id,
                    attempt=attempt,
                    patcher_name=patcher_name,
                    run_metadata=effective_run_metadata,
                    attempt_metadata=attempt_metadata,
                    patcher=patcher_for_attempt,
                    patch_meta=patch_meta,
                    context_protocol=attempt_context_protocol,
                    success=False,
                    note="patch_verifier_rejected",
                    before=current,
                    cumulative_runtime_sec=cumulative_runtime_sec,
                    attempt_wall_clock_sec=attempt_wall_sec,
                )
                _queue_pending_online_credit(pending_credit=pending_credit, record=record)
                _append_record(record, defer_sink=pending_credit is not None)
                _mark_scored_attempt_consumed()
                _capture_attempt_history(
                    attempt=attempt,
                    note="patch_verifier_rejected",
                    failure_signature=failure_signature,
                    record=record,
                    patch_meta=patch_meta,
                    retry_plan_value=retry_plan_text,
                    snapshot_for_history=snapshot,
                )
                previous_failure_context = _enrich_failure_context(
                    base=current_failure_context,
                    note="patch_verifier_rejected",
                    patch_meta=patch_meta,
                    run=current,
                )
                attempt_terminated_early = True
                should_break_after_attempt = adaptive_stop_triggered
                break

            if normalized_patch_sanitize_enabled:
                sanitize_decision = evaluate_patch_sanitize(
                    state=syntax_state,
                    syntax_cycle=syntax_cycle,
                    diff_to_apply=diff_to_apply,
                    patch_sanitize_enabled=True,
                )
                sanitize_meta = dict(sanitize_decision.meta)
                sanitize_ok = not bool(sanitize_meta.get("patch_sanitize_rejected"))
                sanitize_reasons = [
                    str(item) for item in sanitize_meta.get("patch_sanitize_reasons", [])
                ]
                sanitize_error_msg = str(sanitize_meta.get("patch_sanitize_error_msg") or "")
                _sync_patch_sanitize_metadata(
                    attempt_metadata=attempt_metadata,
                    patch_meta=patch_meta,
                    enabled=True,
                    passed=bool(sanitize_ok),
                    rejected=not bool(sanitize_ok),
                    reasons=sanitize_reasons,
                    error_msg=sanitize_error_msg,
                )
                if not sanitize_ok:
                    syntax_preflight_failures = syntax_state.failures
                    syntax_preflight_last_error_type = syntax_state.last_error_type
                    syntax_preflight_last_error_msg = syntax_state.last_error_msg
                    syntax_retry_feedback = syntax_state.retry_feedback
                    syntax_state.apply_to_metadata(
                        attempt_metadata=attempt_metadata,
                        patch_meta=patch_meta,
                    )
                    if syntax_cycle < syntax_preflight_max_cycles_for_attempt:
                        if is_openai_patcher:
                            patcher_for_attempt, _ = _maybe_autoswitch_openai_response_mode(
                                patcher_for_attempt=patcher_for_attempt,
                                policy=normalized_openai_response_mode_autoswitch,
                                reason="patch_sanitize_rejected",
                                syntax_cycle=syntax_cycle,
                                patch_meta=patch_meta,
                                attempt_metadata=attempt_metadata,
                            )
                        continue

                    _refresh_adaptive_progress_for_note(
                        note="syntax_preflight_failed",
                        success=False,
                        signature_repeated=True,
                    )
                    (
                        adaptive_progress_stop_candidate_for_note,
                        adaptive_progress_stop_reason_for_note,
                        adaptive_no_progress_token_sum,
                        adaptive_no_progress_token_cap_hit,
                    ) = _apply_no_progress_token_cap_candidate(
                        is_adaptive_condition=is_adaptive_condition,
                        adaptive_progress_made=adaptive_progress_made,
                        adaptive_no_progress_token_sum=adaptive_no_progress_token_sum,
                        attempt_tokens_total=_nonnegative_int(
                            attempt_metadata.get("tokens_total"), default=0
                        ),
                        adaptive_no_progress_token_cap=normalized_adaptive_no_progress_token_cap,
                        adaptive_progress_stop_candidate=adaptive_progress_stop_candidate,
                        adaptive_progress_stop_reason=adaptive_progress_stop_reason,
                    )
                    patch_meta["adaptive_no_progress_token_sum"] = adaptive_no_progress_token_sum
                    patch_meta["adaptive_no_progress_token_cap"] = (
                        normalized_adaptive_no_progress_token_cap
                    )
                    patch_meta["adaptive_no_progress_token_cap_hit"] = (
                        adaptive_no_progress_token_cap_hit
                    )
                    adaptive_stop_triggered = _mark_adaptive_progress_stop(
                        attempt_metadata=attempt_metadata,
                        patch_meta=patch_meta,
                        is_adaptive_condition=is_adaptive_condition,
                        adaptive_progress_stop_candidate=adaptive_progress_stop_candidate_for_note,
                        adaptive_progress_stop_reason=adaptive_progress_stop_reason_for_note,
                        note="syntax_preflight_failed",
                    )
                    attempt_wall_sec = _measure_attempt_wall_clock_sec(attempt_start)
                    cumulative_runtime_sec += attempt_wall_sec
                    pending_credit = _finalize_adaptive_policy_attempt(
                        note="syntax_preflight_failed",
                        success=False,
                        attempt_wall_sec=attempt_wall_sec,
                        online_finalize_kwargs=online_finalize_kwargs,
                        budget_finalize_kwargs=budget_finalize_kwargs,
                    )
                    record = _build_attempt_outcome_record(
                        case=case,
                        condition=condition,
                        run_id=run_id,
                        attempt=attempt,
                        patcher_name=patcher_name,
                        run_metadata=effective_run_metadata,
                        attempt_metadata=attempt_metadata,
                        patcher=patcher_for_attempt,
                        patch_meta=patch_meta,
                        context_protocol=attempt_context_protocol,
                        success=False,
                        note="syntax_preflight_failed",
                        before=current,
                        cumulative_runtime_sec=cumulative_runtime_sec,
                        attempt_wall_clock_sec=attempt_wall_sec,
                    )
                    _queue_pending_online_credit(pending_credit=pending_credit, record=record)
                    _append_record(record, defer_sink=pending_credit is not None)
                    _mark_scored_attempt_consumed()
                    _capture_attempt_history(
                        attempt=attempt,
                        note="syntax_preflight_failed",
                        failure_signature=failure_signature,
                        record=record,
                        patch_meta=patch_meta,
                        retry_plan_value=retry_plan_text,
                        snapshot_for_history=snapshot,
                    )
                    previous_failure_context = _enrich_failure_context(
                        base=current_failure_context,
                        note="syntax_preflight_failed",
                        patch_meta=patch_meta,
                        run=current,
                    )
                    attempt_terminated_early = True
                    should_break_after_attempt = adaptive_stop_triggered
                    break

            _emit_progress(
                "patch_apply",
                _progress_payload(
                    owner="harness",
                    phase_label="harness: patch apply",
                    note="applying diff",
                    attempt=int(attempt),
                    max_attempts=int(k),
                ),
            )
            apply_start = time.perf_counter()
            patch_application = apply_diff_with_optional_recovery(
                workdir=workdir,
                attempt_dir=attempt_dir,
                diff_to_apply=diff_to_apply,
                file_index=file_index,
                attempt=attempt,
                runner_force_patch_attempt_enabled=runner_force_patch_attempt_enabled,
                patch_meta=patch_meta,
                apply_diff_fn=apply_diff,
            )
            patch_meta["apply_duration_sec"] = time.perf_counter() - apply_start
            file_backups = patch_application.file_backups
            applied_diff_for_preflight = patch_application.applied_diff_for_preflight
            apply_ok = patch_application.apply_ok
            apply_msg = patch_application.apply_msg
            if not apply_ok:
                _refresh_adaptive_progress_for_note(
                    note="patch_apply_failed",
                    success=False,
                    signature_repeated=True,
                )
                (
                    adaptive_progress_stop_candidate_for_note,
                    adaptive_progress_stop_reason_for_note,
                    adaptive_no_progress_token_sum,
                    adaptive_no_progress_token_cap_hit,
                ) = _apply_no_progress_token_cap_candidate(
                    is_adaptive_condition=is_adaptive_condition,
                    adaptive_progress_made=adaptive_progress_made,
                    adaptive_no_progress_token_sum=adaptive_no_progress_token_sum,
                    attempt_tokens_total=_nonnegative_int(
                        attempt_metadata.get("tokens_total"), default=0
                    ),
                    adaptive_no_progress_token_cap=normalized_adaptive_no_progress_token_cap,
                    adaptive_progress_stop_candidate=adaptive_progress_stop_candidate,
                    adaptive_progress_stop_reason=adaptive_progress_stop_reason,
                )
                patch_meta["adaptive_no_progress_token_sum"] = adaptive_no_progress_token_sum
                patch_meta["adaptive_no_progress_token_cap"] = (
                    normalized_adaptive_no_progress_token_cap
                )
                patch_meta["adaptive_no_progress_token_cap_hit"] = (
                    adaptive_no_progress_token_cap_hit
                )
                adaptive_stop_triggered = _mark_adaptive_progress_stop(
                    attempt_metadata=attempt_metadata,
                    patch_meta=patch_meta,
                    is_adaptive_condition=is_adaptive_condition,
                    adaptive_progress_stop_candidate=adaptive_progress_stop_candidate_for_note,
                    adaptive_progress_stop_reason=adaptive_progress_stop_reason_for_note,
                    note="patch_apply_failed",
                )
                syntax_state.apply_to_metadata(
                    attempt_metadata=attempt_metadata, patch_meta=patch_meta
                )
                (attempt_dir / "apply_error.txt").write_text(apply_msg, encoding="utf-8")
                attempt_wall_sec = _measure_attempt_wall_clock_sec(attempt_start)
                cumulative_runtime_sec += attempt_wall_sec
                pending_credit = _finalize_adaptive_policy_attempt(
                    note="patch_apply_failed",
                    success=False,
                    attempt_wall_sec=attempt_wall_sec,
                    online_finalize_kwargs=online_finalize_kwargs,
                    budget_finalize_kwargs=budget_finalize_kwargs,
                )
                record = _build_attempt_outcome_record(
                    case=case,
                    condition=condition,
                    run_id=run_id,
                    attempt=attempt,
                    patcher_name=patcher_name,
                    run_metadata=effective_run_metadata,
                    attempt_metadata=attempt_metadata,
                    patcher=patcher_for_attempt,
                    patch_meta=patch_meta,
                    context_protocol=attempt_context_protocol,
                    success=False,
                    note="patch_apply_failed",
                    before=current,
                    cumulative_runtime_sec=cumulative_runtime_sec,
                    attempt_wall_clock_sec=attempt_wall_sec,
                )
                _queue_pending_online_credit(pending_credit=pending_credit, record=record)
                _append_record(record, defer_sink=pending_credit is not None)
                _mark_scored_attempt_consumed()
                _capture_attempt_history(
                    attempt=attempt,
                    note="patch_apply_failed",
                    failure_signature=failure_signature,
                    record=record,
                    patch_meta=patch_meta,
                    retry_plan_value=retry_plan_text,
                    snapshot_for_history=snapshot,
                )
                previous_failure_context = _enrich_failure_context(
                    base=current_failure_context,
                    note="patch_apply_failed",
                    patch_meta=patch_meta,
                    run=current,
                )
                attempt_terminated_early = True
                should_break_after_attempt = adaptive_stop_triggered
                break

            _invalidate_file_catalog_cache()

            if normalized_syntax_preflight_enabled:
                syntax_decision = evaluate_syntax_preflight(
                    state=syntax_state,
                    syntax_cycle=syntax_cycle,
                    workdir=workdir,
                    diff=applied_diff_for_preflight,
                    file_backups=file_backups,
                    syntax_preflight_fallthrough=normalized_syntax_preflight_fallthrough,
                )
                syntax_preflight_failures = syntax_state.failures
                syntax_preflight_last_error_type = syntax_state.last_error_type
                syntax_preflight_last_error_msg = syntax_state.last_error_msg
                syntax_retry_feedback = syntax_state.retry_feedback
                if syntax_decision.status == "ready":
                    syntax_state.apply_to_metadata(
                        attempt_metadata=attempt_metadata,
                        patch_meta=patch_meta,
                    )
                    patch_applied_and_ready = True
                    break

                if syntax_decision.status == "retry":
                    syntax_state.apply_to_metadata(
                        attempt_metadata=attempt_metadata,
                        patch_meta=patch_meta,
                    )
                    if is_openai_patcher:
                        patcher_for_attempt, _ = _maybe_autoswitch_openai_response_mode(
                            patcher_for_attempt=patcher_for_attempt,
                            policy=normalized_openai_response_mode_autoswitch,
                            reason="syntax_preflight_failed",
                            syntax_cycle=syntax_cycle,
                            patch_meta=patch_meta,
                            attempt_metadata=attempt_metadata,
                        )
                    continue

                # Syntax preflight exhausted all cycles.
                if syntax_decision.status == "fallthrough":
                    # Fall through to test execution — let the test runner
                    # determine the outcome.  This normalizes the comparison
                    # with baselines that self-validate internally.
                    patch_meta["syntax_preflight_exhausted_fallthrough"] = True
                    attempt_metadata["syntax_preflight_exhausted_fallthrough"] = True
                    syntax_state.apply_to_metadata(
                        attempt_metadata=attempt_metadata,
                        patch_meta=patch_meta,
                    )
                    patch_applied_and_ready = True
                    break

                syntax_state.apply_to_metadata(
                    attempt_metadata=attempt_metadata, patch_meta=patch_meta
                )
                _refresh_adaptive_progress_for_note(
                    note="syntax_preflight_failed",
                    success=False,
                    signature_repeated=True,
                )
                (
                    adaptive_progress_stop_candidate_for_note,
                    adaptive_progress_stop_reason_for_note,
                    adaptive_no_progress_token_sum,
                    adaptive_no_progress_token_cap_hit,
                ) = _apply_no_progress_token_cap_candidate(
                    is_adaptive_condition=is_adaptive_condition,
                    adaptive_progress_made=adaptive_progress_made,
                    adaptive_no_progress_token_sum=adaptive_no_progress_token_sum,
                    attempt_tokens_total=_nonnegative_int(
                        attempt_metadata.get("tokens_total"), default=0
                    ),
                    adaptive_no_progress_token_cap=normalized_adaptive_no_progress_token_cap,
                    adaptive_progress_stop_candidate=adaptive_progress_stop_candidate,
                    adaptive_progress_stop_reason=adaptive_progress_stop_reason,
                )
                patch_meta["adaptive_no_progress_token_sum"] = adaptive_no_progress_token_sum
                patch_meta["adaptive_no_progress_token_cap"] = (
                    normalized_adaptive_no_progress_token_cap
                )
                patch_meta["adaptive_no_progress_token_cap_hit"] = (
                    adaptive_no_progress_token_cap_hit
                )
                adaptive_stop_triggered = _mark_adaptive_progress_stop(
                    attempt_metadata=attempt_metadata,
                    patch_meta=patch_meta,
                    is_adaptive_condition=is_adaptive_condition,
                    adaptive_progress_stop_candidate=adaptive_progress_stop_candidate_for_note,
                    adaptive_progress_stop_reason=adaptive_progress_stop_reason_for_note,
                    note="syntax_preflight_failed",
                )
                attempt_wall_sec = _measure_attempt_wall_clock_sec(attempt_start)
                cumulative_runtime_sec += attempt_wall_sec
                pending_credit = _finalize_adaptive_policy_attempt(
                    note="syntax_preflight_failed",
                    success=False,
                    attempt_wall_sec=attempt_wall_sec,
                    online_finalize_kwargs=online_finalize_kwargs,
                    budget_finalize_kwargs=budget_finalize_kwargs,
                )
                record = _build_attempt_outcome_record(
                    case=case,
                    condition=condition,
                    run_id=run_id,
                    attempt=attempt,
                    patcher_name=patcher_name,
                    run_metadata=effective_run_metadata,
                    attempt_metadata=attempt_metadata,
                    patcher=patcher_for_attempt,
                    patch_meta=patch_meta,
                    context_protocol=attempt_context_protocol,
                    success=False,
                    note="syntax_preflight_failed",
                    before=current,
                    cumulative_runtime_sec=cumulative_runtime_sec,
                    attempt_wall_clock_sec=attempt_wall_sec,
                )
                _queue_pending_online_credit(pending_credit=pending_credit, record=record)
                _append_record(record, defer_sink=pending_credit is not None)
                _mark_scored_attempt_consumed()
                _capture_attempt_history(
                    attempt=attempt,
                    note="syntax_preflight_failed",
                    failure_signature=failure_signature,
                    record=record,
                    patch_meta=patch_meta,
                    retry_plan_value=retry_plan_text,
                    snapshot_for_history=snapshot,
                )
                previous_failure_context = _enrich_failure_context(
                    base=current_failure_context,
                    note="syntax_preflight_failed",
                    patch_meta=patch_meta,
                    run=current,
                )
                attempt_terminated_early = True
                should_break_after_attempt = adaptive_stop_triggered
                break

            syntax_preflight_events.append(
                {
                    "cycle": syntax_cycle,
                    "passed": True,
                    "syntax_preflight_checked_files": [],
                    "syntax_preflight_checked_count": 0,
                    "syntax_preflight_error_file": "",
                    "syntax_preflight_error_type": "",
                    "syntax_preflight_error_msg": "",
                }
            )
            _sync_syntax_preflight_metadata(
                attempt_metadata=attempt_metadata,
                patch_meta=patch_meta,
                enabled=normalized_syntax_preflight_enabled,
                cycles_used=syntax_preflight_cycles_used,
                failures=syntax_preflight_failures,
                last_error_type=syntax_preflight_last_error_type,
                last_error_msg=syntax_preflight_last_error_msg,
                events=syntax_preflight_events,
            )
            patch_applied_and_ready = True
            break

        _sync_syntax_preflight_metadata(
            attempt_metadata=attempt_metadata,
            patch_meta=patch_meta,
            enabled=normalized_syntax_preflight_enabled,
            cycles_used=syntax_preflight_cycles_used,
            failures=syntax_preflight_failures,
            last_error_type=syntax_preflight_last_error_type,
            last_error_msg=syntax_preflight_last_error_msg,
            events=syntax_preflight_events,
        )
        if attempt_terminated_early:
            if should_break_after_attempt:
                break
            continue
        if not patch_applied_and_ready:
            continue

        # Re-run after patch.
        shutil.rmtree(llmdebug_dir, ignore_errors=True)
        _emit_progress(
            "attempt_pytest_after",
            _progress_payload(
                owner="pytest",
                phase_label="pytest: after patch",
                note="verifying patch",
                attempt=int(attempt),
                max_attempts=int(k),
            ),
        )
        after = _call_run_pytest_compat(
            workdir,
            case.python_args,
            condition,
            case.timeout_sec,
            output_format,
            progress_update=_emit_progress,
            progress_phase_label="pytest: after patch",
        )
        _write_run_outputs(
            attempt_dir,
            after,
            prefix="pytest_after",
            tail_lines=normalized_pytest_output_tail_lines,
        )
        success = after.returncode == 0
        note = "success" if success else "tests_still_failing"
        adaptive_progress_stop_candidate_for_note = adaptive_progress_stop_candidate
        adaptive_progress_stop_reason_for_note = adaptive_progress_stop_reason
        after_failure_signature = ""
        after_signature_repeated = False
        after_snapshot: dict[str, Any] | None = None
        if is_adaptive_condition and not success:
            after_snapshot, _ = read_latest_snapshot(llmdebug_dir)
            after_failure_signature = _build_failure_signature_for_attempt(
                run=after,
                snapshot=after_snapshot,
            )
            after_signature_repeated = bool(after_failure_signature == failure_signature)
        if is_adaptive_condition:
            adaptive_progress_stop_candidate_for_note, adaptive_progress_stop_reason_for_note = (
                _refresh_adaptive_progress_for_note(
                    note=note,
                    success=success,
                    signature_repeated=(False if success else after_signature_repeated),
                    after_run=after,
                    recomputed_after_patch=True,
                )
            )
            patch_meta["adaptive_postpatch_failure_signature"] = after_failure_signature
            patch_meta["adaptive_postpatch_signature_repeat"] = after_signature_repeated
        (
            adaptive_progress_stop_candidate_for_note,
            adaptive_progress_stop_reason_for_note,
            adaptive_no_progress_token_sum,
            adaptive_no_progress_token_cap_hit,
        ) = _apply_no_progress_token_cap_candidate(
            is_adaptive_condition=is_adaptive_condition,
            adaptive_progress_made=bool(attempt_metadata.get("adaptive_progress_made")),
            adaptive_no_progress_token_sum=adaptive_no_progress_token_sum,
            attempt_tokens_total=_nonnegative_int(attempt_metadata.get("tokens_total"), default=0),
            adaptive_no_progress_token_cap=normalized_adaptive_no_progress_token_cap,
            adaptive_progress_stop_candidate=adaptive_progress_stop_candidate_for_note,
            adaptive_progress_stop_reason=adaptive_progress_stop_reason_for_note,
        )
        patch_meta["adaptive_no_progress_token_sum"] = adaptive_no_progress_token_sum
        patch_meta["adaptive_no_progress_token_cap"] = normalized_adaptive_no_progress_token_cap
        patch_meta["adaptive_no_progress_token_cap_hit"] = adaptive_no_progress_token_cap_hit
        adaptive_stop_triggered = _mark_adaptive_progress_stop(
            attempt_metadata=attempt_metadata,
            patch_meta=patch_meta,
            is_adaptive_condition=is_adaptive_condition,
            adaptive_progress_stop_candidate=adaptive_progress_stop_candidate_for_note,
            adaptive_progress_stop_reason=adaptive_progress_stop_reason_for_note,
            note=note,
        )
        attempt_wall_sec = _measure_attempt_wall_clock_sec(attempt_start)
        cumulative_runtime_sec += attempt_wall_sec
        _update_aux_signals_from_attempt(
            online_kwargs=online_finalize_kwargs,
            budget_kwargs=budget_finalize_kwargs,
            patch_meta=patch_meta,
            new_evidence_ids=bookkeeping_new_evidence_ids,
            reused_evidence_ids=bookkeeping_reused_evidence_ids,
            last_context_compaction_ratio=last_context_compaction_ratio,
            pregate_should_skip=pregate_result.should_skip,
            helper_action_executed=helper_action_executed,
        )
        if full_tep_enabled and adaptive_tep_controller is not None:
            tep_node_packets = _build_full_tep_node_packets(
                note=note,
                success=success,
                stage_a={
                    "invoked": stage_a_request_invoked,
                    "input_summary": stage_a_prompt or "",
                    "request_payload": stage_a_request_payload,
                    "text": stage_a_data.text,
                    "helper_decision_reason": helper_decision_reason,
                    "requests_count": stage_a_data.requests_count,
                    "request_success": stage_a_data.request_success,
                    "materialized_context": helper_materialized_context,
                },
                context_compaction={
                    "requested": context_compaction_requested,
                    "invoked": context_compaction_invoked,
                    "success": context_compaction_success,
                    "trigger": context_compaction_trigger,
                    "ratio": last_context_compaction_ratio,
                    "input_summary": context_compaction_prompt_text or "",
                    "output_summary": compacted_context_for_prompt or context_compaction_error,
                },
                retry_plan={
                    "invoked": retry_plan_invoked,
                    "succeeded": retry_plan_succeeded,
                    "injected": retry_plan_prompt_injected,
                    "error": retry_plan_error,
                    "input_summary": retry_plan_prompt_text or "",
                    "output_summary": retry_plan_text or retry_plan_error,
                },
                patch={
                    "input_summary": prompt,
                    "output_summary": diff_to_apply or note,
                    "patch_verifier_score": patch_meta.get("patch_verifier_score"),
                    "patch_verifier_passed": patch_meta.get("patch_verifier_passed"),
                    "syntax_preflight_cycles_used": attempt_metadata.get(
                        "syntax_preflight_cycles_used"
                    ),
                    "postpatch_signature_repeat": patch_meta.get(
                        "adaptive_postpatch_signature_repeat"
                    ),
                },
            )
            adaptive_tep_learning = run_post_attempt_learning(
                controller=adaptive_tep_controller,
                patcher=patcher_for_attempt,
                context_dir=attempt_dir,
                cwd=workdir,
                case_id=case.case_id,
                node_packets=tep_node_packets,
            )
            attempt_metadata["adaptive_tep_full_learning"] = adaptive_tep_learning
            patch_meta["adaptive_tep_full_learning"] = adaptive_tep_learning
            patch_meta["adaptive_tep_beta"] = float(adaptive_tep_controller.state.beta)
        pending_credit = _finalize_adaptive_policy_attempt(
            note=note,
            success=success,
            attempt_wall_sec=attempt_wall_sec,
            online_finalize_kwargs=online_finalize_kwargs,
            budget_finalize_kwargs=budget_finalize_kwargs,
        )
        record = _build_attempt_outcome_record(
            case=case,
            condition=condition,
            run_id=run_id,
            attempt=attempt,
            patcher_name=patcher_name,
            run_metadata=effective_run_metadata,
            attempt_metadata=attempt_metadata,
            patcher=patcher_for_attempt,
            patch_meta=patch_meta,
            context_protocol=attempt_context_protocol,
            success=success,
            note=note,
            before=current,
            cumulative_runtime_sec=cumulative_runtime_sec,
            after=after,
            attempt_wall_clock_sec=attempt_wall_sec,
        )
        _queue_pending_online_credit(pending_credit=pending_credit, record=record)
        _append_record(record, defer_sink=pending_credit is not None)
        _mark_scored_attempt_consumed()
        _capture_attempt_history(
            attempt=attempt,
            note=note,
            failure_signature=failure_signature,
            record=record,
            patch_meta=patch_meta,
            retry_plan_value=retry_plan_text,
            snapshot_for_history=(
                after_snapshot if (is_adaptive_condition and not success) else snapshot
            ),
        )

        if success:
            _resolve_pending_online_credits_for_success(success_attempt=attempt)
            break
        if adaptive_stop_triggered:
            break

        # Continue: next attempt uses this failure as evidence.
        previous_failure_context = _enrich_failure_context(
            base=current_failure_context,
            note=note,
            patch_meta=patch_meta,
            run=after,
        )
        current = after

    _resolve_remaining_pending_online_credits()
    if adaptive_tep_controller is not None:
        try:
            adaptive_tep_controller.save()
        except Exception:
            pass
    return records
