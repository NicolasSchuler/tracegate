"""Owned support surface for extracted ``evals.run_eval`` attempt-flow modules."""
# ruff: noqa: F401

from __future__ import annotations

from typing import Any

from evals import run_eval as _legacy
from evals.run_eval import (
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
)


def _legacy_call(name: str, *args: Any, **kwargs: Any) -> Any:
    return getattr(_legacy, name)(*args, **kwargs)


def make_patcher(*args: Any, **kwargs: Any) -> Any:
    return _legacy_call("make_patcher", *args, **kwargs)


def read_latest_snapshot(*args: Any, **kwargs: Any) -> Any:
    return _legacy_call("read_latest_snapshot", *args, **kwargs)


def run_one_condition(*args: Any, **kwargs: Any) -> Any:
    return _legacy_call("run_one_condition", *args, **kwargs)


def run_pytest(*args: Any, **kwargs: Any) -> Any:
    return _legacy_call("run_pytest", *args, **kwargs)


def apply_diff(*args: Any, **kwargs: Any) -> Any:
    return _legacy_call("apply_diff", *args, **kwargs)


def _collect_file_index_lines(*args: Any, **kwargs: Any) -> Any:
    return _legacy_call("_collect_file_index_lines", *args, **kwargs)


def collect_text_context(*args: Any, **kwargs: Any) -> Any:
    return _legacy_call("collect_text_context", *args, **kwargs)


def default_lambda_rlm_repo_dir(*args: Any, **kwargs: Any) -> Any:
    return _legacy_call("default_lambda_rlm_repo_dir", *args, **kwargs)


def invoke_lambda_rlm_decomposition(*args: Any, **kwargs: Any) -> Any:
    return _legacy_call("invoke_lambda_rlm_decomposition", *args, **kwargs)


def _measure_attempt_wall_clock_sec(*args: Any, **kwargs: Any) -> Any:
    return _legacy_call("_measure_attempt_wall_clock_sec", *args, **kwargs)


def _remaining_attempt_propose_budget_sec(*args: Any, **kwargs: Any) -> Any:
    return _legacy_call("_remaining_attempt_propose_budget_sec", *args, **kwargs)
