"""Multi-condition repair benchmark harness for llmdebug experiments.

Orchestrates case discovery, condition execution across native llmdebug
conditions and external baselines, patch proposal via pluggable patchers,
pytest verification, and JSONL result recording. Supports local and
Docker-based (testcontainers) execution, TOML config files, resume from
partial runs, adaptive gating policies, online contextual-bandit policy
controllers, and budget-aware resource management.
"""

from __future__ import annotations

import shutil
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path
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

from evals.adaptive_budget_online_policy import (
    BUDGET_POLICY_ARCH_TIER_BINARY,
)
from evals.eval_patch_constants import PATCH_APPLY_TIMEOUT_SEC
from evals.eval_prompts import (
    _apply_prompt_budget,
    _build_retry_history_section,
    _is_generic_snapshot_justification,
    _should_invoke_context_compaction,
)
from evals.policy_feature_scaling import (
    DEFAULT_POLICY_FEATURE_SCALING_PROFILE,
)

if TYPE_CHECKING:
    from evals.patchers.payload_types import ContextRequestPayload
else:
    ContextRequestPayload = dict[str, Any]  # type: ignore[misc,assignment]

_REPO_ROOT = Path(__file__).resolve().parents[1]
_SRC_ROOT = _REPO_ROOT / "src"
if __package__ in {None, ""} and str(_SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(_SRC_ROOT))

from evals.eval_config import (  # noqa: E402
    _default_openai_api_key,
    _parse_adaptive_budget_policy_tiers,
    _write_resolved_run_config_lockfile,
    add_execution_arguments,
    build_execution_config_from_args,
    set_active_execution_config,
)
from evals.eval_containers import (  # noqa: E402
    _docker_client_from_env as _docker_client_from_env,
)
from evals.eval_containers import (  # noqa: E402  # noqa: E402
    check_testcontainers_runtime_available,
    cleanup_stale_eval_containers,
)
from evals.eval_policy_budget import (  # noqa: E402
    _finalize_adaptive_policy_attempt,
    _sync_adaptive_budget_policy_metadata,
)
from evals.eval_policy_features import (  # noqa: E402
    _adaptive_budget_control_changes,
    _adaptive_budget_local_reward_components,
    _adaptive_online_local_reward_components,
    _build_adaptive_budget_control_profile,
    _build_adaptive_budget_family_profile,
)
from evals.eval_policy_online import (  # noqa: E402
    _adaptive_online_delayed_rescue_bonus,
    _sync_adaptive_online_metadata,
)
from evals.eval_recording import (  # noqa: E402
    _build_model_metadata,
    _materialize_attempt_metadata,
    _RunResultsLock,
)
from evals.eval_types import (  # noqa: E402
    _EXCEPTION_TYPE_SPECIFICITY_DEFAULT,
    ADAPTIVE_BUDGET_REWARD_RHO_BUDGET_EXCEEDED_DEFAULT,
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
    ADAPTIVE_REWARD_ETA_PROGRESS_DEFAULT,
    ADAPTIVE_REWARD_KAPPA_STAGNATION_DEFAULT,
    ADAPTIVE_REWARD_LAMBDA_TOKENS_DEFAULT,
    ADAPTIVE_REWARD_LATENCY_SCALE_DEFAULT,
    ADAPTIVE_REWARD_MU_LATENCY_DEFAULT,
    ADAPTIVE_REWARD_TOKEN_SCALE_DEFAULT,
    DEFAULT_ADAPTIVE_BUDGET_POLICY_MAX_UPS_PER_CASE,
    DEFAULT_ADAPTIVE_BUDGET_POLICY_TIERS,
    DEFAULT_ADAPTIVE_BUDGET_TOPUP_MAX_CALLS,
    DEFAULT_ADAPTIVE_NO_PROGRESS_TOKEN_CAP,
    DEFAULT_ADAPTIVE_ONLINE_ACTION_SCORE_BIASES,
    DEFAULT_CONTAINER_CLEANUP_STALE_GRACE_SEC,
    DEFAULT_OPENAI_TRANSPORT_RECOVERY_ENABLED,
    DEFAULT_PYTEST_OUTPUT_TAIL_LINES,
    DEFAULT_SNAPSHOT_ARTIFACT_MAX_STRING_CHARS,
    DEFAULT_SYNTAX_PREFLIGHT_MAX_CYCLES,
    EVAL_CONTAINER_LABEL_KEY,
    EVAL_CONTAINER_LABEL_VALUE,
    EVAL_RECORD_SCHEMA_VERSION,
    EVAL_RECORD_TYPE_ATTEMPT,
    EVAL_RECORD_TYPE_INFRA_EVENT,
    PATCH_VERIFIER_MIN_SCORE,
    PATCH_VERIFIER_RECOVERY_RESULT_PASSED_VERIFIER,
    PATCH_VERIFIER_SCOPE_ADAPTIVE_ONLY,
    PATCH_VERIFIER_SCOPE_ALL,
    PATCH_VERIFIER_SCOPE_DISABLED,
    PYTEST_TIMEOUT_RETURNCODE,
    RUN_EVAL_LOCKFILE_REDACTED,
    AdaptiveBudgetControlProfile,
    CaseSpec,
    Condition,
    EvalUnitInputs,
    ExecutionConfig,
    RunConditionInputs,
    RunResult,
    UnitExecutionOutcome,
    _ActiveRunSettings,
)


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


from evals.eval_attempt_flow import execute_eval_unit_impl, run_one_condition_impl  # noqa: E402
from evals.eval_evidence import (  # noqa: E402
    _coerce_context_request_payload,
    _collect_file_index_lines,
    build_fallback_context_bundle,
    collect_text_context,
    execute_context_requests,
)
from evals.eval_main_flow import run_main_flow  # noqa: E402
from evals.eval_seams import EvalSeams  # noqa: E402
from evals.lambda_rlm_backend import (  # noqa: E402
    DEFAULT_LAMBDA_RLM_CONTEXT_WINDOW_CHARS,
    DEFAULT_LAMBDA_RLM_TIMEOUT_SEC,
)

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
    _pytest_failed_total_from_text,
    _traceback_informativeness_pre_gate,
    _traceback_informativeness_score,
)
from evals.attempt_proposal import (  # noqa: E402
    _build_attempt_budget_exhausted_patch_result,
    _build_attempt_budget_exhausted_plan_result,
    _ensure_deterministic_seed_ack,
    _patcher_with_shared_attempt_budget,
    _plan_patch_with_snapshot_context,
    _propose_patch_with_snapshot_context,
    _remaining_attempt_propose_budget_sec,
)
from evals.attempt_records import (  # noqa: E402
    _append_retry_history_turn_if_enabled,
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
    _completion_chars_from_response,
    _format_progress_token_usage,
    _llm_call_count_from_meta,
    _nonnegative_int,
    _patch_attempt_char_deltas,
    _prompt_chars_for_llm_calls,
    _prompt_chars_from_patch_meta,
    _reasoning_token_counts_from_patch_meta,
    _reasoning_token_counts_from_usage_payload,
    _record_token_usage_state,
    _summarize_context_compaction_meta,
    _summarize_recovery_usage_meta,
    _sync_attempt_prompt_char_metadata,
    _sync_attempt_usage_metadata,
    _timeout_error_from_meta,
    _token_usage_available_from_meta,
    _token_usage_blocked_from_meta,
    _usage_available_flag_from_meta,
    _usage_token_counts_from_patch_meta,
    _usage_token_counts_from_usage_payload,
)
from evals.attempt_verification import (  # noqa: E402  # noqa: E402
    _sync_patch_diff_metadata,
    _sync_patch_sanitize_metadata,
    _sync_patch_verifier_metadata,
    _sync_patch_verifier_recovery_metadata,
    _sync_syntax_preflight_metadata,
)
from evals.baseline_runtime import (  # noqa: E402
    DEFAULT_BASELINE_RLM_COMPACTION_THRESHOLD_PCT,
    DEFAULT_BASELINE_RLM_MAX_DEPTH,
    DEFAULT_BASELINE_RLM_MAX_ERRORS,
    DEFAULT_BASELINE_RLM_MAX_ITERATIONS,
    _autodetect_baseline_python,
    _baseline_extra_from_args,
    _baseline_patch_inventory,
    _normalize_adaptive_controller_lambda_rlm_settings,
    _normalize_baseline_selection,
    _resolve_conditions_for_run,
    _resolve_external_baseline_runtime_requirements,
    _validate_external_baseline_runtime_requirements,
)
from evals.case_loading import (  # noqa: E402
    load_cases,
)
from evals.eval_cli_args import (  # noqa: E402
    build_parser,
)
from evals.eval_patches import (  # noqa: E402
    _patch_mentions_scope_files,
    apply_diff,  # re-exported for importers/validators
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
    _build_run_condition_inputs_from_legacy_kwargs,
    _resolve_run_condition_inputs,
)
from evals.run_state import (  # noqa: E402
    _build_run_settings_fingerprint,
    _load_existing_run_state,
    _repair_trailing_results_tail,
)
from evals.transport_failures import (  # noqa: E402
    _classify_patcher_request_failure,
    _filter_attempt_records,
    _is_attempt_record,
    _is_context_limit_http_error_message,
    _is_retryable_transport_http_status,
    _is_transport_runtime_error,
    _OpenAITransportRecoveryGate,
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


# The CLI entrypoint and the names other modules and tests reach through it. Code moved
# to the modules that own it (X-02); keep this list to what is still used here.
__all__ = [
    "ADAPTIVE_BUDGET_REWARD_RHO_BUDGET_EXCEEDED_DEFAULT",
    "ADAPTIVE_HELPER_OUTCOME_COMPAT_REASONING_ONLY",
    "ADAPTIVE_HELPER_OUTCOME_INVALID_PAYLOAD",
    "ADAPTIVE_HELPER_OUTCOME_MATERIALIZED",
    "ADAPTIVE_HELPER_OUTCOME_NOT_INVOKED",
    "ADAPTIVE_HELPER_OUTCOME_REQUEST_BUDGET_EXCEEDED",
    "ADAPTIVE_HELPER_OUTCOME_SEMANTIC_NO_CONTEXT",
    "ADAPTIVE_HELPER_OUTCOME_SKIPPED_GUARDRAIL",
    "ADAPTIVE_HELPER_OUTCOME_SKIPPED_POLICY",
    "ADAPTIVE_HELPER_OUTCOME_SKIPPED_UNSUPPORTED",
    "ADAPTIVE_HELPER_OUTCOME_TRANSPORT_FAILED",
    "ADAPTIVE_HELPER_OUTCOME_UNAVAILABLE_BACKEND",
    "ADAPTIVE_REWARD_ETA_PROGRESS_DEFAULT",
    "ADAPTIVE_REWARD_KAPPA_STAGNATION_DEFAULT",
    "ADAPTIVE_REWARD_LAMBDA_TOKENS_DEFAULT",
    "ADAPTIVE_REWARD_LATENCY_SCALE_DEFAULT",
    "ADAPTIVE_REWARD_MU_LATENCY_DEFAULT",
    "ADAPTIVE_REWARD_TOKEN_SCALE_DEFAULT",
    "BUDGET_POLICY_ARCH_TIER_BINARY",
    "DEFAULT_ADAPTIVE_BUDGET_POLICY_MAX_UPS_PER_CASE",
    "DEFAULT_ADAPTIVE_BUDGET_POLICY_TIERS",
    "DEFAULT_ADAPTIVE_BUDGET_TOPUP_MAX_CALLS",
    "DEFAULT_ADAPTIVE_NO_PROGRESS_TOKEN_CAP",
    "DEFAULT_ADAPTIVE_ONLINE_ACTION_SCORE_BIASES",
    "DEFAULT_BASELINE_RLM_COMPACTION_THRESHOLD_PCT",
    "DEFAULT_BASELINE_RLM_MAX_DEPTH",
    "DEFAULT_BASELINE_RLM_MAX_ERRORS",
    "DEFAULT_BASELINE_RLM_MAX_ITERATIONS",
    "DEFAULT_CONTAINER_CLEANUP_STALE_GRACE_SEC",
    "DEFAULT_LAMBDA_RLM_CONTEXT_WINDOW_CHARS",
    "DEFAULT_LAMBDA_RLM_TIMEOUT_SEC",
    "DEFAULT_OPENAI_TRANSPORT_RECOVERY_ENABLED",
    "DEFAULT_POLICY_FEATURE_SCALING_PROFILE",
    "DEFAULT_PYTEST_OUTPUT_TAIL_LINES",
    "DEFAULT_SNAPSHOT_ARTIFACT_MAX_STRING_CHARS",
    "DEFAULT_SYNTAX_PREFLIGHT_MAX_CYCLES",
    "EVAL_CONTAINER_LABEL_KEY",
    "EVAL_CONTAINER_LABEL_VALUE",
    "EVAL_RECORD_SCHEMA_VERSION",
    "EVAL_RECORD_TYPE_ATTEMPT",
    "EVAL_RECORD_TYPE_INFRA_EVENT",
    "PATCH_APPLY_TIMEOUT_SEC",
    "PATCH_VERIFIER_MIN_SCORE",
    "PATCH_VERIFIER_RECOVERY_RESULT_PASSED_VERIFIER",
    "PATCH_VERIFIER_SCOPE_ADAPTIVE_ONLY",
    "PATCH_VERIFIER_SCOPE_ALL",
    "PATCH_VERIFIER_SCOPE_DISABLED",
    "PYTEST_TIMEOUT_RETURNCODE",
    "RUN_EVAL_LOCKFILE_REDACTED",
    "_EXCEPTION_TYPE_SPECIFICITY_DEFAULT",
    "_REPO_ROOT",
    "_SRC_ROOT",
    "AdaptiveBudgetControlProfile",
    "ExecutionConfig",
    "RunResult",
    "_ActiveRunSettings",
    "_OpenAITransportRecoveryGate",
    "_adaptive_budget_control_changes",
    "_adaptive_budget_local_reward_components",
    "_adaptive_gate_should_invoke_helper",
    "_adaptive_helper_outcome_for_attempt",
    "_adaptive_online_delayed_rescue_bonus",
    "_adaptive_online_local_reward_components",
    "_adaptive_optional_bool",
    "_adaptive_optional_int",
    "_adaptive_pipeline_stage_before",
    "_adaptive_pipeline_stage_for_outcome",
    "_adaptive_progress_signal",
    "_adaptive_prompt_budget_for_attempt",
    "_adaptive_should_stop_for_no_progress",
    "_aggregate_attempt_llm_usage",
    "_append_retry_history_turn_if_enabled",
    "_apply_no_progress_token_cap_candidate",
    "_apply_prompt_budget",
    "_apply_usage_totals_to_patch_meta",
    "_autodetect_baseline_python",
    "_base_pytest_env",
    "_baseline_extra_from_args",
    "_baseline_metric_float",
    "_baseline_metrics_from_patch_meta",
    "_baseline_patch_inventory",
    "_build_adaptive_budget_control_profile",
    "_build_adaptive_budget_family_profile",
    "_build_attempt_budget_exhausted_patch_result",
    "_build_attempt_budget_exhausted_plan_result",
    "_build_failure_signature_for_attempt",
    "_build_infra_event_record",
    "_build_model_metadata",
    "_build_pytest_args",
    "_build_retry_history_section",
    "_build_run_condition_inputs_from_legacy_kwargs",
    "_build_structured_previous_attempt_diff",
    "_classify_patcher_request_failure",
    "_clip_patcher_meta_text",
    "_coerce_context_request_payload",
    "_coerce_subprocess_output",
    "_collect_file_index_lines",
    "_completion_chars_from_response",
    "_default_openai_api_key",
    "_effective_patch_verifier_scope",
    "_enrich_failure_context",
    "_ensure_deterministic_seed_ack",
    "_execute_eval_unit",
    "_extract_exception_info",
    "_filter_attempt_records",
    "_finalize_adaptive_policy_attempt",
    "_finalize_breakdowns",
    "_format_duration",
    "_format_duration_precise",
    "_format_progress_token_usage",
    "_infer_test_source_scope",
    "_is_attempt_record",
    "_is_container_wait_timeout",
    "_is_context_limit_http_error_message",
    "_is_generic_snapshot_justification",
    "_is_retryable_transport_http_status",
    "_is_snapshot_condition",
    "_is_transport_runtime_error",
    "_llm_call_count_from_meta",
    "_mark_adaptive_progress_stop",
    "_materialize_attempt_metadata",
    "_nonnegative_int",
    "_normalize_adaptive_controller_lambda_rlm_settings",
    "_normalize_baseline_selection",
    "_normalized_progress_verbosity",
    "_openai_preflight_total_char_cap",
    "_parse_adaptive_budget_policy_tiers",
    "_patch_attempt_char_deltas",
    "_patch_mentions_scope_files",
    "_patcher_with_shared_attempt_budget",
    "_plan_patch_with_snapshot_context",
    "_progress_is_enabled",
    "_progress_print",
    "_prompt_chars_for_llm_calls",
    "_prompt_chars_from_patch_meta",
    "_propose_patch_with_snapshot_context",
    "_pytest_failed_total_from_text",
    "_reasoning_token_counts_from_patch_meta",
    "_reasoning_token_counts_from_usage_payload",
    "_record_is_transport_request_failure",
    "_record_token_usage_state",
    "_record_type",
    "_remaining_attempt_propose_budget_sec",
    "_request_context_failure_kind_from_meta",
    "_request_context_recovery_failure_kind_from_meta",
    "_request_context_retry_counts_from_meta",
    "_resolve_conditions_for_run",
    "_resolve_external_baseline_runtime_requirements",
    "_resolve_openai_context_window_for_run",
    "_resolve_run_condition_inputs",
    "_result_meta_dict",
    "_result_with_meta",
    "_run_pytest_local",
    "_run_pytest_pooled",
    "_run_pytest_testcontainers",
    "_score_patch_relevance",
    "_should_invoke_context_compaction",
    "_single_line_error",
    "_status_code_from_wait_result",
    "_summarize_context_compaction_meta",
    "_summarize_recovery_usage_meta",
    "_sync_adaptive_budget_policy_metadata",
    "_sync_adaptive_online_metadata",
    "_sync_attempt_prompt_char_metadata",
    "_sync_attempt_usage_metadata",
    "_sync_patch_diff_metadata",
    "_sync_patch_sanitize_metadata",
    "_sync_patch_verifier_metadata",
    "_sync_patch_verifier_recovery_metadata",
    "_sync_patcher_request_failure_metadata",
    "_sync_syntax_preflight_metadata",
    "_timeout_error_from_meta",
    "_timeout_fields",
    "_token_usage_available_from_meta",
    "_token_usage_blocked_from_meta",
    "_traceback_informativeness_pre_gate",
    "_traceback_informativeness_score",
    "_transport_recovery_failure_kind_from_meta",
    "_truncate_persisted_pytest_output",
    "_truncate_snapshot_for_artifact",
    "_truncate_snapshot_string_for_artifact",
    "_usage_available_flag_from_meta",
    "_usage_token_counts_from_patch_meta",
    "_usage_token_counts_from_usage_payload",
    "_validate_external_baseline_runtime_requirements",
    "_write_run_outputs",
    "add_execution_arguments",
    "apply_diff",
    "build_execution_config_from_args",
    "build_fallback_context_bundle",
    "collect_text_context",
    "execute_context_requests",
    "json",
    "main",
    "read_latest_snapshot",
    "run_one_condition",
    "run_pytest",
    "set_active_execution_config",
    "shutil",
    "subprocess",
    "time",
]
