"""Multi-condition repair benchmark harness for llmdebug experiments.

Orchestrates case discovery, condition execution across native llmdebug
conditions and external baselines, patch proposal via pluggable patchers,
pytest verification, and JSONL result recording. Supports local and
Docker-based (testcontainers) execution, TOML config files, resume from
partial runs, adaptive gating policies, online contextual-bandit policy
controllers, and budget-aware resource management.
"""
# ruff: noqa: F401

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import replace
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING, Any

if __package__ in {None, ""}:
    _BOOTSTRAP_REPO_ROOT = Path(__file__).resolve().parents[1]
    _BOOTSTRAP_SRC_ROOT = _BOOTSTRAP_REPO_ROOT / "src"
    for _candidate in (str(_BOOTSTRAP_REPO_ROOT), str(_BOOTSTRAP_SRC_ROOT)):
        if _candidate not in sys.path:
            sys.path.insert(0, _candidate)
    from evals import _json as json
else:
    from . import _json as json

try:
    import tomllib as _toml_loader
except ModuleNotFoundError:
    _toml_loader = None

from baselines.openai_compat import COMPAT_PROFILE_CHOICES
from baselines.subprocess_monitor import run_monitored_subprocess
from evals.adaptive_budget_online_policy import (
    ACTION_BUDGET_FAMILY_CONTEXT,
    ACTION_BUDGET_FAMILY_TIME,
    ACTION_BUDGET_FAMILY_TOKENS,
    ACTION_INCREASE_BUDGET,
    ACTION_KEEP_BUDGET,
    BUDGET_POLICY_ACTIONS,
    BUDGET_POLICY_ARCH_HIERARCHICAL_V1,
    BUDGET_POLICY_ARCH_TIER_BINARY,
    BUDGET_POLICY_ARCHITECTURES,
    BUDGET_POLICY_FAMILIES,
    AdaptiveBudgetPolicyController,
    BudgetPolicyConfig,
    BudgetPolicyDecision,
)
from evals.adaptive_online_policy import (
    ACTION_COMPACT_THEN_PATCH,
    ACTION_INVOKE_HELPER,
    ACTION_RETRY_HELPER_STRICT_JSON,
    ACTION_SKIP_HELPER,
    ACTION_SNAPSHOT_LITE,
    ONLINE_POLICY_ACTIONS,
    AdaptiveOnlinePolicyController,
    OnlinePolicyConfig,
    PolicyDecision,
)
from evals.eval_patch_constants import PATCH_APPLY_TIMEOUT_SEC, TRUNCATED_RESPONSE_ERROR
from evals.eval_prompts import (
    _apply_prompt_budget,
    _build_retry_history_section,
    _build_retry_summary,
    _format_snapshot_index_section,
    _is_generic_snapshot_justification,
    _render_retry_history_turn,
    _render_retry_packet,
    _should_invoke_context_compaction,
    render_stage_a_request_prompt,
)
from evals.policy_feature_scaling import (
    DEFAULT_POLICY_FEATURE_SCALING_PROFILE,
    normalize_feature_scaling_profile,
    scale_feature_mapping,
)
from evals.progress_output import RunProgressSummary, build_progress_palette

if TYPE_CHECKING:
    from evals.patchers.payload_types import ContextRequestPayload
else:
    ContextRequestPayload = dict[str, Any]  # type: ignore[misc,assignment]

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
if __package__ in {None, ""} and str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from evals.decision_model.config import (  # noqa: E402
    add_decision_model_arguments,
    add_policy_input_arguments,
)
from evals.eval_config import (  # noqa: E402
    _apply_toml_configs_to_args,
    _default_openai_api_key,
    _parse_adaptive_budget_policy_tiers,
    _parse_adaptive_online_action_score_biases,
    _write_resolved_run_config_lockfile,
    add_execution_arguments,
    build_execution_config_from_args,
    get_active_execution_config,
    resolve_openai_api_key_default,
    set_active_execution_config,
)
from evals.eval_containers import (  # noqa: E402
    _docker_client_from_env as _docker_client_from_env,
)
from evals.eval_containers import (  # noqa: E402  # noqa: E402
    _EvalContainerPool,
    check_testcontainers_runtime_available,
    cleanup_stale_eval_containers,
)
from evals.eval_policy_budget import (  # noqa: E402
    _adaptive_budget_behavior_fields,
    _finalize_adaptive_budget_policy_attempt,
    _finalize_adaptive_policy_attempt,
    _sync_adaptive_budget_policy_metadata,
    _sync_finalize_progress_kwargs_from_attempt_metadata,
)
from evals.eval_policy_features import (  # noqa: E402
    _adaptive_budget_control_changes,
    _adaptive_budget_control_profile_to_dict,
    _adaptive_budget_controls_changed,
    _adaptive_budget_family_availability,
    _adaptive_budget_feature_vector,
    _adaptive_budget_level_multiplier,
    _adaptive_budget_level_step_delta,
    _adaptive_budget_local_reward_components,
    _adaptive_budget_profile_for_family_step,
    _adaptive_budget_recorded_multiplier,
    _adaptive_budget_union_change_flags,
    _adaptive_online_feature_vector,
    _adaptive_online_local_reward_components,
    _adaptive_policy_local_accounting,
    _apply_adaptive_budget_profile_to_patcher,
    _build_adaptive_budget_control_profile,
    _build_adaptive_budget_family_profile,
)
from evals.eval_policy_online import (  # noqa: E402
    _adaptive_online_behavior_fields,
    _adaptive_online_delayed_rescue_bonus,
    _adaptive_online_reward_breakdown,
    _adaptive_online_should_defer_credit,
    _adaptive_online_update_suppression_reason,
    _finalize_adaptive_online_attempt,
    _record_adaptive_online_policy_outcome,
    _resolve_pending_adaptive_online_credit,
    _sync_adaptive_online_metadata,
    _sync_record_metadata_copies,
)
from evals.eval_recording import (  # noqa: E402
    DEFAULT_REPO_OPENAI_COMPAT_PROFILE,
    _baseline_openai_compat_settings_from_args,
    _build_base_record,
    _build_model_metadata,
    _build_patcher_config,
    _build_run_metadata,
    _default_run_metadata,
    _empty_attempt_metadata,
    _materialize_attempt_metadata,
    _native_openai_compat_settings_from_args,
    _resolve_llmdebug_snapshot_schema_version,
    _RunResultsLock,
    _SynchronizedJsonlWriter,
)
from evals.eval_types import (  # noqa: E402
    _ASSERTION_VALUE_PATTERN,
    _DIAGNOSTIC_MESSAGE_PATTERNS,
    _EXCEPTION_TYPE_SPECIFICITY,
    _EXCEPTION_TYPE_SPECIFICITY_DEFAULT,
    _PROGRESS_LEVEL_ORDER,
    _PROGRESS_PRINT_LOCK,
    _PYTEST_SUMMARY_COUNT_RE,
    _STDOUT_PYTEST_ERROR_RE,
    _STDOUT_TRACEBACK_ERROR_RE,
    ADAPTIVE_AUX_REWARD_COMPACTION_EFFICIENCY_WEIGHT_DEFAULT,
    ADAPTIVE_AUX_REWARD_EVIDENCE_NOVELTY_WEIGHT_DEFAULT,
    ADAPTIVE_AUX_REWARD_INFORMATIVENESS_ALIGNMENT_WEIGHT_DEFAULT,
    ADAPTIVE_AUX_REWARD_PYTEST_REDUCTION_WEIGHT_DEFAULT,
    ADAPTIVE_BUDGET_CONTROL_CHANGE_KEYS,
    ADAPTIVE_BUDGET_POLICY_ALGOS,
    ADAPTIVE_BUDGET_POLICY_MODES,
    ADAPTIVE_BUDGET_REWARD_RHO_BUDGET_EXCEEDED_DEFAULT,
    ADAPTIVE_BUDGET_REWARD_SCOPE,
    ADAPTIVE_DELAYED_RESCUE_HORIZON_ATTEMPTS,
    ADAPTIVE_FEATURE_SCALING_PROFILES,
    ADAPTIVE_GATE_POLICIES,
    ADAPTIVE_HELPER_BUDGET_PROFILES,
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
    ADAPTIVE_LOCAL_REWARD_DELAYED_RESCUE_BONUS_BASE,
    ADAPTIVE_LOCAL_REWARD_DELAYED_RESCUE_BONUS_MAX,
    ADAPTIVE_LOCAL_REWARD_DELAYED_RESCUE_CONTEXT_MULTIPLIER,
    ADAPTIVE_LOCAL_REWARD_DELAYED_RESCUE_HORIZON_TWO_FACTOR,
    ADAPTIVE_LOCAL_REWARD_HELPER_MATERIALIZATION_BONUS_SCALE,
    ADAPTIVE_LOCAL_REWARD_PROGRESS_HINT_SCALE,
    ADAPTIVE_LOCAL_REWARD_STAGNATION_PENALTY_SCALE,
    ADAPTIVE_LOCAL_REWARD_SUCCESS_HINT,
    ADAPTIVE_ONLINE_POLICY_ALGOS,
    ADAPTIVE_ONLINE_POLICY_MODES,
    ADAPTIVE_ONLINE_REWARD_SCOPE,
    ADAPTIVE_OPENAI_HELPER_RESPONSE_MODES,
    ADAPTIVE_REWARD_ETA_PROGRESS_DEFAULT,
    ADAPTIVE_REWARD_KAPPA_STAGNATION_DEFAULT,
    ADAPTIVE_REWARD_LAMBDA_TOKENS_DEFAULT,
    ADAPTIVE_REWARD_LATENCY_SCALE_DEFAULT,
    ADAPTIVE_REWARD_MATERIALIZATION_SCALE_DEFAULT,
    ADAPTIVE_REWARD_MU_LATENCY_DEFAULT,
    ADAPTIVE_REWARD_PROGRESS_SCALE_DEFAULT,
    ADAPTIVE_REWARD_STAGNATION_SCALE_DEFAULT,
    ADAPTIVE_REWARD_SUCCESS_HINT_DEFAULT,
    ADAPTIVE_REWARD_TOKEN_SCALE_DEFAULT,
    CONTEXT_COMPACTION_MODES,
    CONTEXT_COMPACTION_SCOPE_TOKENS,
    CONTEXT_PROTOCOLS,
    CONTEXT_RECOVERY_FAILURE_KINDS,
    CONTEXT_RECOVERY_PROMPT_BUDGET,
    DEFAULT_ADAPTIVE_BUDGET_ONLINE_ALPHA,
    DEFAULT_ADAPTIVE_BUDGET_ONLINE_DECAY,
    DEFAULT_ADAPTIVE_BUDGET_ONLINE_EPSILON,
    DEFAULT_ADAPTIVE_BUDGET_ONLINE_WARMUP_DECISIONS,
    DEFAULT_ADAPTIVE_BUDGET_POLICY_MAX_UPS_PER_CASE,
    DEFAULT_ADAPTIVE_BUDGET_POLICY_TIERS,
    DEFAULT_ADAPTIVE_BUDGET_TOPUP_MAX_CALLS,
    DEFAULT_ADAPTIVE_NO_PROGRESS_TOKEN_CAP,
    DEFAULT_ADAPTIVE_ONLINE_ACTION_SCORE_BIASES,
    DEFAULT_ADAPTIVE_ONLINE_ALPHA,
    DEFAULT_ADAPTIVE_ONLINE_DECAY,
    DEFAULT_ADAPTIVE_ONLINE_EPSILON,
    DEFAULT_ADAPTIVE_ONLINE_WARMUP_DECISIONS,
    DEFAULT_ADAPTIVE_POLICY_L2,
    DEFAULT_ADAPTIVE_POLICY_SAVE_EVERY,
    DEFAULT_CONTAINER_CLEANUP_STALE_GRACE_SEC,
    DEFAULT_EXECUTOR,
    DEFAULT_OPENAI_TRANSPORT_RECOVERY_ENABLED,
    DEFAULT_OPENAI_TRANSPORT_RECOVERY_MAX_WAIT_SEC,
    DEFAULT_OPENAI_TRANSPORT_RECOVERY_POLL_SEC,
    DEFAULT_PROGRESS_HEARTBEAT_SEC,
    DEFAULT_PYTEST_OUTPUT_TAIL_LINES,
    DEFAULT_SNAPSHOT_ARTIFACT_MAX_STRING_CHARS,
    DEFAULT_SYNTAX_PREFLIGHT_MAX_CYCLES,
    DEFAULT_TRANSPORT_FAIL_FAST_STREAK,
    EMPTY_ADAPTIVE_POLICY_LOCAL_ACCOUNTING,
    EVAL_CONTAINER_LABEL_KEY,
    EVAL_CONTAINER_LABEL_VALUE,
    EVAL_RECORD_SCHEMA_VERSION,
    EVAL_RECORD_TYPE_ATTEMPT,
    EVAL_RECORD_TYPE_INFRA_EVENT,
    HELPER_CONTEXT_MATERIALIZING_ACTIONS,
    HELPER_STAGE_A_ACTIONS,
    OPENAI_CONTEXT_WINDOW_RESERVE_TOKENS,
    OPENAI_RESPONSE_MODE_AUTOSWITCH_MODES,
    PATCH_VERIFIER_MIN_SCORE,
    PATCH_VERIFIER_RECOVERY_RESULT_NO_DIFF,
    PATCH_VERIFIER_RECOVERY_RESULT_NOT_ELIGIBLE,
    PATCH_VERIFIER_RECOVERY_RESULT_PASSED_VERIFIER,
    PATCH_VERIFIER_RECOVERY_RESULT_REQUEST_FAILED,
    PATCH_VERIFIER_RECOVERY_RESULT_SKIPPED_DEADLINE_EXHAUSTED,
    PATCH_VERIFIER_RECOVERY_RESULT_STILL_REJECTED,
    PATCH_VERIFIER_SCOPE_ADAPTIVE_ONLY,
    PATCH_VERIFIER_SCOPE_ALL,
    PATCH_VERIFIER_SCOPE_DISABLED,
    PATCHER_CONTEXT_LIMIT_ERROR_MARKERS,
    PERSISTED_PYTEST_OUTPUT_HARD_CHAR_CAP,
    PROGRESS_VERBOSITIES,
    PYTEST_TIMEOUT_RETURNCODE,
    RECORD_MODES,
    RETRY_CONTEXT_MODES,
    RETRY_PROMPT_MODES,
    RETRYABLE_HTTP_STATUS_CODES,
    RUN_EVAL_LOCKFILE_REDACTED,
    VALID_CATEGORIES,
    VALID_DIFFICULTIES,
    VALID_SNAPSHOT_DEPENDENCIES,
    AdaptiveBudgetControlProfile,
    AdaptiveOnlineRewardBreakdown,
    AdaptivePolicyLocalAccounting,
    AdaptiveProgressState,
    CaseSpec,
    Condition,
    ConfigResolutionState,
    EvalUnitInputs,
    ExecutionConfig,
    PendingAdaptiveOnlineCredit,
    PreGateResult,
    RunConditionInputs,
    RunResult,
    StageAContextData,
    UnitExecutionOutcome,
    _ActiveRunSettings,
    _default_helper_action,
    _empty_stage_a_context,
    _filter_context_request_payload_for_action,
    _helper_action_availability,
    _TrailingResultsTailIssue,
    baseline_condition_for_name,
    is_baseline_condition,
)
from evals.oracle_integrity import (  # noqa: E402
    clear_junit_reports,
    consume_junit_counts,
    junit_report_name,
)
from llmdebug._hashing import xxh64_hexdigest_text  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """CLI entry-point for the eval harness.

    Parses arguments, discovers cases, resumes partial runs, and
    dispatches (case, condition) units to a thread-pool executor.

    Returns:
        Exit code: 0 on success, 1 on partial failure, 2 on fatal error.
    """
    argv_tokens = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()
    args = parser.parse_args(argv_tokens)
    return run_main_flow(parser=parser, args=args, argv_tokens=argv_tokens, seams=_current_seams())


def _current_seams() -> EvalSeams:
    """This module's collaborators as they are now, monkeypatches included (X-02)."""
    return EvalSeams(
        repo_root=_REPO_ROOT,
        load_cases=load_cases,
        make_patcher=make_patcher,
        run_one_condition=run_one_condition,
        build_run_settings_fingerprint=_build_run_settings_fingerprint,
        write_resolved_run_config_lockfile=_write_resolved_run_config_lockfile,
        load_existing_run_state=_load_existing_run_state,
        repair_trailing_results_tail=_repair_trailing_results_tail,
        check_testcontainers_runtime_available=check_testcontainers_runtime_available,
        cleanup_stale_eval_containers=cleanup_stale_eval_containers,
        run_results_lock=_RunResultsLock,
    )


def _execute_eval_unit(
    *,
    case: CaseSpec,
    condition: Condition,
    artifacts_root: Path,
    inputs: EvalUnitInputs,
) -> UnitExecutionOutcome:
    return execute_eval_unit_impl(
        case=case,
        condition=condition,
        artifacts_root=artifacts_root,
        inputs=inputs if inputs.seams is not None else replace(inputs, seams=_current_seams()),
    )


from evals.adaptive_controller_packet import remove_adaptive_controller_sidecars  # noqa: E402
from evals.adaptive_controller_runtime import (  # noqa: E402
    ControllerRuntimeInputs,
    prepare_controller_runtime,
    write_controller_sidecars_for_cycle,
)
from evals.adaptive_history import persist_adaptive_attempt_history  # noqa: E402
from evals.eval_attempt_flow import execute_eval_unit_impl, run_one_condition_impl  # noqa: E402
from evals.eval_evidence import (  # noqa: E402
    _coerce_context_request_payload,
    _collect_file_index_lines,
    _compute_evidence_fingerprint,
    _crash_file_from_snapshot,
    _detect_failing_test_file,
    _evidence_ids_from_items,
    _extract_subprocess_script_neighbors,
    _import_neighbors,
    _merge_evidence_items,
    _render_evidence_index,
    _render_evidence_items,
    _strip_eval_only_case_artifacts,
    _xxh3_128_jsonlike,
    _xxh3_128_text,
    build_fallback_context_bundle,
    collect_text_context,
    execute_context_requests,
)
from evals.eval_main_flow import run_main_flow  # noqa: E402
from evals.eval_seams import EvalSeams  # noqa: E402
from evals.lambda_rlm_backend import (  # noqa: E402
    DEFAULT_LAMBDA_RLM_CONTEXT_WINDOW_CHARS,
    DEFAULT_LAMBDA_RLM_TIMEOUT_SEC,
    default_lambda_rlm_repo_dir,
    invoke_lambda_rlm_decomposition,
)
from evals.patchers.baseline_patcher import resolve_effective_baseline_model  # noqa: E402

# snapshot_chars set by the instrumentation in run_eval; keep if already set


def run_one_condition(
    *,
    case: CaseSpec,
    condition: Condition,
    workdir: Path,
    context_dir_base: Path,
    inputs: RunConditionInputs | None = None,
    **legacy_kwargs: Any,
) -> list[dict[str, Any]]:
    return run_one_condition_impl(
        case=case,
        condition=condition,
        workdir=workdir,
        context_dir_base=context_dir_base,
        inputs=inputs,
        **legacy_kwargs,
    )


from evals.adaptive_progress import (  # noqa: E402
    _adaptive_gate_should_invoke_helper,
    _adaptive_helper_outcome_for_attempt,
    _adaptive_optional_bool,
    _adaptive_optional_int,
    _adaptive_pipeline_stage_before,
    _adaptive_pipeline_stage_for_outcome,
    _adaptive_progress_signal,
    _adaptive_prompt_budget_for_attempt,
    _adaptive_should_stop_for_no_progress,
    _apply_no_progress_token_cap_candidate,
    _extract_exception_info,
    _mark_adaptive_progress_stop,
    _openai_preflight_total_char_cap,
    _pytest_failed_total,
    _pytest_failed_total_from_text,
    _recompute_adaptive_progress_for_attempt_outcome,
    _sync_adaptive_progress_metadata,
    _traceback_informativeness_pre_gate,
    _traceback_informativeness_score,
)
from evals.attempt_proposal import (  # noqa: E402
    _build_attempt_budget_exhausted_patch_result,
    _build_attempt_budget_exhausted_plan_result,
    _ensure_deterministic_seed_ack,
    _measure_attempt_wall_clock_sec,
    _patcher_with_shared_attempt_budget,
    _plan_patch_with_snapshot_context,
    _propose_patch_with_snapshot_context,
    _remaining_attempt_propose_budget_sec,
)
from evals.attempt_records import (  # noqa: E402
    _append_retry_history_turn_if_enabled,
    _build_attempt_outcome_record,
    _build_infra_event_record,
    _build_structured_previous_attempt_diff,
    _finalize_breakdowns,
)
from evals.attempt_usage import (  # noqa: E402
    _aggregate_attempt_llm_usage,
    _apply_usage_totals_to_patch_meta,
    _baseline_metric_float,
    _baseline_metrics_from_patch_meta,
    _clip_patcher_meta_text,
    _combine_reasoning_usage_segments,
    _completion_chars_from_response,
    _format_progress_token_usage,
    _llm_call_count_from_meta,
    _nonnegative_int,
    _optional_nonnegative_int,
    _patch_attempt_char_deltas,
    _prompt_chars_for_llm_calls,
    _prompt_chars_from_patch_meta,
    _reasoning_token_counts_from_patch_meta,
    _reasoning_token_counts_from_usage_payload,
    _reasoning_usage_segment_from_meta,
    _record_token_usage_state,
    _summarize_context_compaction_meta,
    _summarize_recovery_usage_meta,
    _sync_attempt_prompt_char_metadata,
    _sync_attempt_usage_metadata,
    _timeout_error_from_meta,
    _token_usage_available_from_meta,
    _token_usage_blocked_from_meta,
    _usage_available_flag_from_meta,
    _usage_payload_from_counts,
    _usage_token_counts_from_patch_meta,
    _usage_token_counts_from_usage_payload,
)
from evals.attempt_verification import (  # noqa: E402  # noqa: E402
    _sync_patch_diff_metadata,
    _sync_patch_sanitize_metadata,
    _sync_patch_verifier_metadata,
    _sync_patch_verifier_recovery_metadata,
    _sync_syntax_preflight_metadata,
    apply_diff_with_optional_recovery,
    evaluate_patch_sanitize,
    evaluate_syntax_preflight,
    make_syntax_preflight_state,
)
from evals.baseline_runtime import (  # noqa: E402
    _BASELINE_REPO_DIR_BY_NAME,
    _BASELINE_SHORTHAND_CHOICES,
    _NATIVE_BASELINE_MODES,
    _NATIVE_COT_BASELINE_ALLOWED_CONDITIONS,
    DEFAULT_BASELINE_RLM_COMPACTION_THRESHOLD_PCT,
    DEFAULT_BASELINE_RLM_MAX_DEPTH,
    DEFAULT_BASELINE_RLM_MAX_ERRORS,
    DEFAULT_BASELINE_RLM_MAX_ITERATIONS,
    DEFAULT_BASELINE_RLM_TIMEOUT_SEC,
    _absolute_symlink_portability_hint,
    _autodetect_baseline_python,
    _autodetect_lambda_rlm_python,
    _baseline_extra_from_args,
    _baseline_patch_inventory,
    _baseline_runtime_manifest_entry,
    _baseline_runtime_manifest_path,
    _default_baseline_repo_dir,
    _default_baseline_repo_relative_parts,
    _normalize_adaptive_controller_lambda_rlm_settings,
    _normalize_baseline_selection,
    _normalize_repo_local_path_to_current_root,
    _resolve_conditions_for_run,
    _resolve_external_baseline_runtime_requirements,
    _resolve_manifest_path_entry,
    _validate_external_baseline_runtime_requirements,
)
from evals.case_loading import (  # noqa: E402
    load_cases,
)
from evals.eval_cli_args import (  # noqa: E402
    build_parser,
)
from evals.eval_patches import (  # noqa: E402
    _build_runner_forced_attempt_diff,
    _compute_patch_path_metrics,
    _extract_candidate_patch_files,
    _is_forced_noop_marker_diff,
    _is_test_file_path,
    _patch_contains_substantive_change,
    _patch_mentions_crash_file,
    _patch_mentions_failing_test_context,
    _patch_mentions_scope_files,
    _render_patch_verifier_recovery_feedback,
    apply_diff,  # re-exported for importers/validators
)
from evals.native_openai_attempt_flow import (  # noqa: E402
    NativeOpenAIAttemptPromptRequest,
    assemble_native_openai_attempt_prompt,
)
from evals.native_openai_attempt_flow import (  # noqa: E402
    maybe_autoswitch_openai_response_mode as _maybe_autoswitch_openai_response_mode,
)
from evals.native_openai_attempt_flow import (  # noqa: E402
    sync_openai_response_mode_autoswitch_metadata as _sync_openai_response_mode_autoswitch_metadata,
)
from evals.native_stage_b_flow import (  # noqa: E402
    StageBPromptInputs,
    build_stage_b_prompt_sections,
    build_stage_b_prompt_with_budget,
    maybe_run_retry_plan_then_patch,
    openai_preflight_total_char_cap,
)
from evals.patch_relevance import (  # noqa: E402
    _effective_patch_verifier_scope,
    _infer_test_source_scope,
    _score_patch_relevance,
)
from evals.patcher_factory import (  # noqa: E402
    _resolve_openai_context_window_for_run,
    make_patcher,
)
from evals.progress_display import (  # noqa: E402
    _format_duration,
    _format_duration_precise,
    _normalized_progress_verbosity,
    _progress_is_enabled,
    _progress_print,
)
from evals.pytest_runner import (  # noqa: E402
    _base_pytest_env,
    _build_failure_signature_for_attempt,
    _build_pytest_args,
    _coerce_subprocess_output,
    _enrich_failure_context,
    _is_container_wait_timeout,
    _is_snapshot_condition,
    _run_pytest_local,
    _run_pytest_pooled,
    _run_pytest_testcontainers,
    _status_code_from_wait_result,
    _timeout_fields,
    _truncate_persisted_pytest_output,
    _truncate_snapshot_for_artifact,
    _truncate_snapshot_string_for_artifact,
    _write_run_outputs,
    read_latest_snapshot,
    run_pytest,
)
from evals.run_condition_builder import (  # noqa: E402
    _RUN_CONDITION_INPUT_FIELD_NAMES,
    _RUN_CONDITION_INPUT_REQUIRED_KEYS,
    _build_run_condition_inputs_from_legacy_kwargs,
    _resolve_run_condition_inputs,
)
from evals.run_state import (  # noqa: E402
    _build_run_settings_fingerprint,
    _load_existing_run_state,
    _repair_trailing_results_tail,
)
from evals.transport_failures import (  # noqa: E402
    _INFRA_FAILURE_TEXT_MARKERS,
    _INFRA_HTTP_STATUS_RE,
    _classify_patcher_request_failure,
    _filter_attempt_records,
    _is_attempt_record,
    _is_context_limit_http_error_message,
    _is_retryable_transport_http_status,
    _is_transport_runtime_error,
    _OpenAITransportRecoveryGate,
    _patcher_infra_failure_kind_from_meta,
    _record_is_infra_resume_eligible,
    _record_is_transport_request_failure,
    _record_type,
    _request_context_failure_kind_from_meta,
    _request_context_recovery_failure_kind_from_meta,
    _request_context_retry_counts_from_meta,
    _result_meta_dict,
    _result_with_meta,
    _single_line_error,
    _sync_patcher_request_failure_metadata,
    _transport_recovery_failure_kind_from_meta,
)

if __name__ == "__main__":
    raise SystemExit(main())
