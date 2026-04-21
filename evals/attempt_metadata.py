"""Typed builders for per-attempt eval metadata."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .adaptive_online_policy import ACTION_INVOKE_HELPER, ACTION_SKIP_HELPER
from .eval_types import ADAPTIVE_HELPER_OUTCOME_NOT_INVOKED, ContextRequestPayload


@dataclass(frozen=True)
class StageAAttemptMetadata:
    stage_a_prompt: str | None = None
    stage_a_raw_response: str | None = None
    stage_a_request_payload: ContextRequestPayload | None = None
    stage_a_prompt_chars: int = 0
    stage_a_requests_count: int = 0
    stage_a_chars_returned: int = 0
    stage_a_request_success: bool | None = None
    stage_a_index_included: bool = False
    stage_a_index_reason: str = ""


@dataclass(frozen=True)
class RetryAttemptMetadata:
    retry_prompt_mode: str = "full"
    retry_context_mode: str = "history"
    retry_history_chars: int = 0
    retry_history_truncated: bool = False
    retry_history_turns_included: int = 0
    retry_history_budget_chars: int = 0
    retry_history_budget_applied: bool = False
    retry_delta_prompt: bool = False
    retry_delta_chars: int = 0


@dataclass(frozen=True)
class AdaptiveAttemptMetadata:
    adaptive_mode: str = "none"
    adaptive_policy: str = "disabled"
    adaptive_signature_repeat: bool = False
    adaptive_stagnation_streak: int = 0
    adaptive_helper_invoked: bool = False
    adaptive_helper_action_default: str = ACTION_INVOKE_HELPER
    adaptive_helper_action_selected: str = ACTION_INVOKE_HELPER
    adaptive_helper_action_executed: str = ACTION_SKIP_HELPER
    adaptive_helper_action_fallback_reason: str = ""
    adaptive_helper_available_actions: list[str] = field(default_factory=list)
    adaptive_helper_unavailable_action_reasons: dict[str, str] = field(default_factory=dict)
    adaptive_helper_materialized_context: bool = False
    adaptive_helper_outcome: str = ADAPTIVE_HELPER_OUTCOME_NOT_INVOKED
    adaptive_helper_failure_kind: str = ""
    adaptive_helper_retry_count: int = 0
    adaptive_helper_transport_retry_count: int = 0
    adaptive_helper_decision_reason: str = ""
    adaptive_stagnation_override_applied: bool = False
    adaptive_stagnation_override_reason: str = ""
    adaptive_helper_need_more_context: bool | None = None
    adaptive_helper_calls_used_before_attempt: int = 0
    adaptive_helper_calls_used_after_attempt: int = 0
    adaptive_helper_budget_max_calls: int = 0
    adaptive_helper_budget_unlimited: bool = True
    adaptive_helper_budget_exhausted: bool = False
    adaptive_helper_saturation_guarded: bool = False
    adaptive_helper_budget_profile: str = "default"
    adaptive_budget_topup_granted: bool = False
    adaptive_budget_topup_reason: str = ""
    adaptive_context_requested: bool = False
    adaptive_snapshot_included: bool = False
    adaptive_snapshot_lite_included: bool = False
    adaptive_snapshot_attempt1_denied: bool = False
    adaptive_snapshot_attempt1_denied_reason: str = ""
    adaptive_progress_made: bool = False
    adaptive_progress_reason: str = ""
    adaptive_no_progress_streak: int = 0
    adaptive_progress_stop_candidate: bool = False
    adaptive_progress_stop_triggered: bool = False
    adaptive_progress_stop_reason: str = ""
    adaptive_informativeness_pregate_fired: bool = False
    adaptive_informativeness_pregate_score: float = 0.0
    adaptive_informativeness_pregate_exception_type: str = ""
    adaptive_informativeness_pregate_reason: str = ""
    adaptive_informativeness_pregate_features: dict[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class PromptBudgetMetadata:
    prompt_budget_used_chars: int | None = None
    prompt_budget_overflow_chars: int = 0
    prompt_budget_dropped_sections: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class ContextCompactionMetadata:
    context_compaction_mode: str = "off"
    context_compaction_scope: list[str] = field(default_factory=list)
    context_compaction_supported: bool = False
    context_compaction_requested: bool = False
    context_compaction_request_reason: str = ""
    context_compaction_trigger: str = ""
    context_compaction_auto_overflow_triggered: bool = False
    context_compaction_invoked: bool = False
    context_compaction_success: bool = False
    context_compaction_error: str = ""
    context_compaction_calls_used: int = 0
    context_compaction_max_calls: int = 0
    context_compaction_source_sections: list[str] = field(default_factory=list)
    context_compaction_source_chars: dict[str, int] = field(default_factory=dict)
    context_compaction_replaced_sections: list[str] = field(default_factory=list)
    context_compaction_input_chars: int = 0
    context_compaction_target_chars: int = 0
    context_compaction_output_chars: int = 0
    context_compaction_prompt_chars: int = 0


@dataclass(frozen=True)
class OpenAIAttemptMetadata:
    openai_preflight_prompt_cap_chars: int | None = None
    openai_preflight_trimmed: bool = False
    openai_context_window_source: str = ""
    openai_context_window_probe_provider: str = ""
    openai_context_window_probe_error: str = ""
    openai_response_mode_autoswitch_policy: str = "off"
    openai_response_mode_autoswitched: bool = False
    openai_response_mode_autoswitch_from: str = ""
    openai_response_mode_autoswitch_to: str = ""
    openai_response_mode_autoswitch_cycle: int = 0
    openai_response_mode_autoswitch_reason: str = ""


@dataclass(frozen=True)
class SyntaxPreflightMetadata:
    syntax_preflight_enabled: bool = False
    syntax_preflight_cycles_used: int = 0
    syntax_preflight_failures: int = 0
    syntax_preflight_last_error_type: str = ""
    syntax_preflight_last_error_msg: str = ""


@dataclass(frozen=True)
class AttemptMetadataRequest:
    attempt_dir: Path
    attempt: int
    max_attempts: int
    evidence: dict[str, Any]
    prompt: str
    record_prompt: str
    record_evidence: str
    context_protocol: str
    llm_call_count: int = 0
    prompt_chars_total: int = 0
    completion_chars_total: int = 0
    tokens_prompt: int = 0
    tokens_completion: int = 0
    tokens_total: int = 0
    evidence_ids_new_count: int = 0
    evidence_ids_reused_count: int = 0
    evidence_ids_sent_count: int = 0
    timing_breakdown: dict[str, float] | None = None
    token_breakdown: dict[str, int] | None = None
    stage_a: StageAAttemptMetadata = field(default_factory=StageAAttemptMetadata)
    retry: RetryAttemptMetadata = field(default_factory=RetryAttemptMetadata)
    adaptive: AdaptiveAttemptMetadata = field(default_factory=AdaptiveAttemptMetadata)
    prompt_budget: PromptBudgetMetadata = field(default_factory=PromptBudgetMetadata)
    context_compaction: ContextCompactionMetadata = field(default_factory=ContextCompactionMetadata)
    openai: OpenAIAttemptMetadata = field(default_factory=OpenAIAttemptMetadata)
    syntax_preflight: SyntaxPreflightMetadata = field(default_factory=SyntaxPreflightMetadata)


def build_attempt_metadata_request(
    *,
    attempt_dir: Path,
    attempt: int,
    max_attempts: int,
    evidence: dict[str, Any],
    prompt: str,
    record_prompt: str,
    record_evidence: str,
    context_protocol: str,
    llm_call_count: int = 0,
    prompt_chars_total: int = 0,
    completion_chars_total: int = 0,
    tokens_prompt: int = 0,
    tokens_completion: int = 0,
    tokens_total: int = 0,
    evidence_ids_new_count: int = 0,
    evidence_ids_reused_count: int = 0,
    evidence_ids_sent_count: int = 0,
    timing_breakdown: dict[str, float] | None = None,
    token_breakdown: dict[str, int] | None = None,
    stage_a: StageAAttemptMetadata | None = None,
    retry: RetryAttemptMetadata | None = None,
    adaptive: AdaptiveAttemptMetadata | None = None,
    prompt_budget: PromptBudgetMetadata | None = None,
    context_compaction: ContextCompactionMetadata | None = None,
    openai: OpenAIAttemptMetadata | None = None,
    syntax_preflight: SyntaxPreflightMetadata | None = None,
) -> AttemptMetadataRequest:
    """Build a typed per-attempt metadata request."""
    return AttemptMetadataRequest(
        attempt_dir=attempt_dir,
        attempt=attempt,
        max_attempts=max_attempts,
        evidence=dict(evidence),
        prompt=str(prompt),
        record_prompt=str(record_prompt),
        record_evidence=str(record_evidence),
        context_protocol=str(context_protocol),
        llm_call_count=llm_call_count,
        prompt_chars_total=prompt_chars_total,
        completion_chars_total=completion_chars_total,
        tokens_prompt=tokens_prompt,
        tokens_completion=tokens_completion,
        tokens_total=tokens_total,
        evidence_ids_new_count=evidence_ids_new_count,
        evidence_ids_reused_count=evidence_ids_reused_count,
        evidence_ids_sent_count=evidence_ids_sent_count,
        timing_breakdown=dict(timing_breakdown) if timing_breakdown is not None else None,
        token_breakdown=dict(token_breakdown) if token_breakdown is not None else None,
        stage_a=stage_a or StageAAttemptMetadata(),
        retry=retry or RetryAttemptMetadata(),
        adaptive=adaptive or AdaptiveAttemptMetadata(),
        prompt_budget=prompt_budget or PromptBudgetMetadata(),
        context_compaction=context_compaction or ContextCompactionMetadata(),
        openai=openai or OpenAIAttemptMetadata(),
        syntax_preflight=syntax_preflight or SyntaxPreflightMetadata(),
    )


def build_attempt_metadata_request_from_legacy_kwargs(
    *,
    attempt_dir: Path,
    attempt: int,
    max_attempts: int,
    evidence: dict[str, Any],
    prompt: str,
    record_prompt: str,
    record_evidence: str,
    context_protocol: str,
    stage_a_prompt: str | None = None,
    stage_a_raw_response: str | None = None,
    stage_a_request_payload: ContextRequestPayload | None = None,
    stage_a_prompt_chars: int = 0,
    stage_a_requests_count: int = 0,
    stage_a_chars_returned: int = 0,
    stage_a_request_success: bool | None = None,
    retry_prompt_mode: str = "full",
    retry_context_mode: str = "history",
    retry_history_chars: int = 0,
    retry_history_truncated: bool = False,
    retry_history_turns_included: int = 0,
    retry_history_budget_chars: int = 0,
    retry_history_budget_applied: bool = False,
    retry_delta_prompt: bool = False,
    retry_delta_chars: int = 0,
    adaptive_mode: str = "none",
    adaptive_policy: str = "disabled",
    adaptive_signature_repeat: bool = False,
    adaptive_stagnation_streak: int = 0,
    adaptive_helper_invoked: bool = False,
    adaptive_helper_action_default: str = ACTION_INVOKE_HELPER,
    adaptive_helper_action_selected: str = ACTION_INVOKE_HELPER,
    adaptive_helper_action_executed: str = ACTION_SKIP_HELPER,
    adaptive_helper_action_fallback_reason: str = "",
    adaptive_helper_available_actions: list[str] | None = None,
    adaptive_helper_unavailable_action_reasons: dict[str, str] | None = None,
    adaptive_helper_materialized_context: bool = False,
    adaptive_helper_outcome: str = ADAPTIVE_HELPER_OUTCOME_NOT_INVOKED,
    adaptive_helper_failure_kind: str = "",
    adaptive_helper_retry_count: int = 0,
    adaptive_helper_transport_retry_count: int = 0,
    adaptive_helper_decision_reason: str = "",
    adaptive_stagnation_override_applied: bool = False,
    adaptive_stagnation_override_reason: str = "",
    adaptive_helper_need_more_context: bool | None = None,
    adaptive_helper_calls_used_before_attempt: int = 0,
    adaptive_helper_calls_used_after_attempt: int = 0,
    adaptive_helper_budget_max_calls: int = 0,
    adaptive_helper_budget_unlimited: bool = True,
    adaptive_helper_budget_exhausted: bool = False,
    adaptive_helper_saturation_guarded: bool = False,
    adaptive_helper_budget_profile: str = "default",
    adaptive_budget_topup_granted: bool = False,
    adaptive_budget_topup_reason: str = "",
    adaptive_context_requested: bool = False,
    adaptive_snapshot_included: bool = False,
    adaptive_snapshot_lite_included: bool = False,
    adaptive_snapshot_attempt1_denied: bool = False,
    adaptive_snapshot_attempt1_denied_reason: str = "",
    adaptive_progress_made: bool = False,
    adaptive_progress_reason: str = "",
    adaptive_no_progress_streak: int = 0,
    adaptive_progress_stop_candidate: bool = False,
    adaptive_progress_stop_triggered: bool = False,
    adaptive_progress_stop_reason: str = "",
    llm_call_count: int = 0,
    prompt_chars_total: int = 0,
    completion_chars_total: int = 0,
    tokens_prompt: int = 0,
    tokens_completion: int = 0,
    tokens_total: int = 0,
    evidence_ids_new_count: int = 0,
    evidence_ids_reused_count: int = 0,
    evidence_ids_sent_count: int = 0,
    stage_a_index_included: bool = False,
    stage_a_index_reason: str = "",
    openai_preflight_prompt_cap_chars: int | None = None,
    openai_preflight_trimmed: bool = False,
    openai_context_window_source: str = "",
    openai_context_window_probe_provider: str = "",
    openai_context_window_probe_error: str = "",
    openai_response_mode_autoswitch_policy: str = "off",
    openai_response_mode_autoswitched: bool = False,
    openai_response_mode_autoswitch_from: str = "",
    openai_response_mode_autoswitch_to: str = "",
    openai_response_mode_autoswitch_cycle: int = 0,
    openai_response_mode_autoswitch_reason: str = "",
    prompt_budget_used_chars: int | None = None,
    prompt_budget_overflow_chars: int = 0,
    prompt_budget_dropped_sections: list[str] | None = None,
    context_compaction_mode: str = "off",
    context_compaction_scope: list[str] | None = None,
    context_compaction_supported: bool = False,
    context_compaction_requested: bool = False,
    context_compaction_request_reason: str = "",
    context_compaction_trigger: str = "",
    context_compaction_auto_overflow_triggered: bool = False,
    context_compaction_invoked: bool = False,
    context_compaction_success: bool = False,
    context_compaction_error: str = "",
    context_compaction_calls_used: int = 0,
    context_compaction_max_calls: int = 0,
    context_compaction_source_sections: list[str] | None = None,
    context_compaction_source_chars: dict[str, int] | None = None,
    context_compaction_replaced_sections: list[str] | None = None,
    context_compaction_input_chars: int = 0,
    context_compaction_target_chars: int = 0,
    context_compaction_output_chars: int = 0,
    context_compaction_prompt_chars: int = 0,
    adaptive_informativeness_pregate_fired: bool = False,
    adaptive_informativeness_pregate_score: float = 0.0,
    adaptive_informativeness_pregate_exception_type: str = "",
    adaptive_informativeness_pregate_reason: str = "",
    adaptive_informativeness_pregate_features: dict[str, float] | None = None,
    syntax_preflight_enabled: bool = False,
    syntax_preflight_cycles_used: int = 0,
    syntax_preflight_failures: int = 0,
    syntax_preflight_last_error_type: str = "",
    syntax_preflight_last_error_msg: str = "",
    timing_breakdown: dict[str, float] | None = None,
    token_breakdown: dict[str, int] | None = None,
) -> AttemptMetadataRequest:
    """Build a typed request from the legacy flat keyword adapter."""
    return build_attempt_metadata_request(
        attempt_dir=attempt_dir,
        attempt=attempt,
        max_attempts=max_attempts,
        evidence=evidence,
        prompt=prompt,
        record_prompt=record_prompt,
        record_evidence=record_evidence,
        context_protocol=context_protocol,
        llm_call_count=llm_call_count,
        prompt_chars_total=prompt_chars_total,
        completion_chars_total=completion_chars_total,
        tokens_prompt=tokens_prompt,
        tokens_completion=tokens_completion,
        tokens_total=tokens_total,
        evidence_ids_new_count=evidence_ids_new_count,
        evidence_ids_reused_count=evidence_ids_reused_count,
        evidence_ids_sent_count=evidence_ids_sent_count,
        timing_breakdown=timing_breakdown,
        token_breakdown=token_breakdown,
        stage_a=StageAAttemptMetadata(
            stage_a_prompt=stage_a_prompt,
            stage_a_raw_response=stage_a_raw_response,
            stage_a_request_payload=stage_a_request_payload,
            stage_a_prompt_chars=stage_a_prompt_chars,
            stage_a_requests_count=stage_a_requests_count,
            stage_a_chars_returned=stage_a_chars_returned,
            stage_a_request_success=stage_a_request_success,
            stage_a_index_included=stage_a_index_included,
            stage_a_index_reason=stage_a_index_reason,
        ),
        retry=RetryAttemptMetadata(
            retry_prompt_mode=retry_prompt_mode,
            retry_context_mode=retry_context_mode,
            retry_history_chars=retry_history_chars,
            retry_history_truncated=retry_history_truncated,
            retry_history_turns_included=retry_history_turns_included,
            retry_history_budget_chars=retry_history_budget_chars,
            retry_history_budget_applied=retry_history_budget_applied,
            retry_delta_prompt=retry_delta_prompt,
            retry_delta_chars=retry_delta_chars,
        ),
        adaptive=AdaptiveAttemptMetadata(
            adaptive_mode=adaptive_mode,
            adaptive_policy=adaptive_policy,
            adaptive_signature_repeat=adaptive_signature_repeat,
            adaptive_stagnation_streak=adaptive_stagnation_streak,
            adaptive_helper_invoked=adaptive_helper_invoked,
            adaptive_helper_action_default=adaptive_helper_action_default,
            adaptive_helper_action_selected=adaptive_helper_action_selected,
            adaptive_helper_action_executed=adaptive_helper_action_executed,
            adaptive_helper_action_fallback_reason=adaptive_helper_action_fallback_reason,
            adaptive_helper_available_actions=list(adaptive_helper_available_actions or []),
            adaptive_helper_unavailable_action_reasons=dict(
                adaptive_helper_unavailable_action_reasons or {}
            ),
            adaptive_helper_materialized_context=adaptive_helper_materialized_context,
            adaptive_helper_outcome=adaptive_helper_outcome,
            adaptive_helper_failure_kind=adaptive_helper_failure_kind,
            adaptive_helper_retry_count=adaptive_helper_retry_count,
            adaptive_helper_transport_retry_count=adaptive_helper_transport_retry_count,
            adaptive_helper_decision_reason=adaptive_helper_decision_reason,
            adaptive_stagnation_override_applied=adaptive_stagnation_override_applied,
            adaptive_stagnation_override_reason=adaptive_stagnation_override_reason,
            adaptive_helper_need_more_context=adaptive_helper_need_more_context,
            adaptive_helper_calls_used_before_attempt=adaptive_helper_calls_used_before_attempt,
            adaptive_helper_calls_used_after_attempt=adaptive_helper_calls_used_after_attempt,
            adaptive_helper_budget_max_calls=adaptive_helper_budget_max_calls,
            adaptive_helper_budget_unlimited=adaptive_helper_budget_unlimited,
            adaptive_helper_budget_exhausted=adaptive_helper_budget_exhausted,
            adaptive_helper_saturation_guarded=adaptive_helper_saturation_guarded,
            adaptive_helper_budget_profile=adaptive_helper_budget_profile,
            adaptive_budget_topup_granted=adaptive_budget_topup_granted,
            adaptive_budget_topup_reason=adaptive_budget_topup_reason,
            adaptive_context_requested=adaptive_context_requested,
            adaptive_snapshot_included=adaptive_snapshot_included,
            adaptive_snapshot_lite_included=adaptive_snapshot_lite_included,
            adaptive_snapshot_attempt1_denied=adaptive_snapshot_attempt1_denied,
            adaptive_snapshot_attempt1_denied_reason=adaptive_snapshot_attempt1_denied_reason,
            adaptive_progress_made=adaptive_progress_made,
            adaptive_progress_reason=adaptive_progress_reason,
            adaptive_no_progress_streak=adaptive_no_progress_streak,
            adaptive_progress_stop_candidate=adaptive_progress_stop_candidate,
            adaptive_progress_stop_triggered=adaptive_progress_stop_triggered,
            adaptive_progress_stop_reason=adaptive_progress_stop_reason,
            adaptive_informativeness_pregate_fired=adaptive_informativeness_pregate_fired,
            adaptive_informativeness_pregate_score=adaptive_informativeness_pregate_score,
            adaptive_informativeness_pregate_exception_type=(
                adaptive_informativeness_pregate_exception_type
            ),
            adaptive_informativeness_pregate_reason=adaptive_informativeness_pregate_reason,
            adaptive_informativeness_pregate_features=dict(
                adaptive_informativeness_pregate_features or {}
            ),
        ),
        prompt_budget=PromptBudgetMetadata(
            prompt_budget_used_chars=prompt_budget_used_chars,
            prompt_budget_overflow_chars=prompt_budget_overflow_chars,
            prompt_budget_dropped_sections=list(prompt_budget_dropped_sections or []),
        ),
        context_compaction=ContextCompactionMetadata(
            context_compaction_mode=context_compaction_mode,
            context_compaction_scope=list(context_compaction_scope or []),
            context_compaction_supported=context_compaction_supported,
            context_compaction_requested=context_compaction_requested,
            context_compaction_request_reason=context_compaction_request_reason,
            context_compaction_trigger=context_compaction_trigger,
            context_compaction_auto_overflow_triggered=context_compaction_auto_overflow_triggered,
            context_compaction_invoked=context_compaction_invoked,
            context_compaction_success=context_compaction_success,
            context_compaction_error=context_compaction_error,
            context_compaction_calls_used=context_compaction_calls_used,
            context_compaction_max_calls=context_compaction_max_calls,
            context_compaction_source_sections=list(context_compaction_source_sections or []),
            context_compaction_source_chars=dict(context_compaction_source_chars or {}),
            context_compaction_replaced_sections=list(context_compaction_replaced_sections or []),
            context_compaction_input_chars=context_compaction_input_chars,
            context_compaction_target_chars=context_compaction_target_chars,
            context_compaction_output_chars=context_compaction_output_chars,
            context_compaction_prompt_chars=context_compaction_prompt_chars,
        ),
        openai=OpenAIAttemptMetadata(
            openai_preflight_prompt_cap_chars=openai_preflight_prompt_cap_chars,
            openai_preflight_trimmed=openai_preflight_trimmed,
            openai_context_window_source=openai_context_window_source,
            openai_context_window_probe_provider=openai_context_window_probe_provider,
            openai_context_window_probe_error=openai_context_window_probe_error,
            openai_response_mode_autoswitch_policy=openai_response_mode_autoswitch_policy,
            openai_response_mode_autoswitched=openai_response_mode_autoswitched,
            openai_response_mode_autoswitch_from=openai_response_mode_autoswitch_from,
            openai_response_mode_autoswitch_to=openai_response_mode_autoswitch_to,
            openai_response_mode_autoswitch_cycle=openai_response_mode_autoswitch_cycle,
            openai_response_mode_autoswitch_reason=openai_response_mode_autoswitch_reason,
        ),
        syntax_preflight=SyntaxPreflightMetadata(
            syntax_preflight_enabled=syntax_preflight_enabled,
            syntax_preflight_cycles_used=syntax_preflight_cycles_used,
            syntax_preflight_failures=syntax_preflight_failures,
            syntax_preflight_last_error_type=syntax_preflight_last_error_type,
            syntax_preflight_last_error_msg=syntax_preflight_last_error_msg,
        ),
    )
