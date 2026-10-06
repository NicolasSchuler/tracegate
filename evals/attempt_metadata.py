"""Typed builders for per-attempt eval metadata."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from dataclasses import MISSING, dataclass, field, fields
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


_GROUPED_METADATA_TYPES: tuple[tuple[str, type[Any]], ...] = (
    ("stage_a", StageAAttemptMetadata),
    ("retry", RetryAttemptMetadata),
    ("adaptive", AdaptiveAttemptMetadata),
    ("prompt_budget", PromptBudgetMetadata),
    ("context_compaction", ContextCompactionMetadata),
    ("openai", OpenAIAttemptMetadata),
    ("syntax_preflight", SyntaxPreflightMetadata),
)
_GROUPED_REQUEST_FIELDS = frozenset(name for name, _metadata_type in _GROUPED_METADATA_TYPES)
_TOP_LEVEL_LEGACY_FIELDS = frozenset(
    field_info.name
    for field_info in fields(AttemptMetadataRequest)
    if field_info.name not in _GROUPED_REQUEST_FIELDS
)
_GROUPED_LEGACY_FIELDS = frozenset(
    field_info.name
    for _name, metadata_type in _GROUPED_METADATA_TYPES
    for field_info in fields(metadata_type)
)
_ALLOWED_LEGACY_FIELDS = _TOP_LEVEL_LEGACY_FIELDS | _GROUPED_LEGACY_FIELDS
_REQUIRED_LEGACY_FIELDS = frozenset(
    field_info.name
    for field_info in fields(AttemptMetadataRequest)
    if field_info.name in _TOP_LEVEL_LEGACY_FIELDS
    and field_info.default is MISSING
    and field_info.default_factory is MISSING
)
_LEGACY_LIST_FIELDS = frozenset(
    {
        "adaptive_helper_available_actions",
        "prompt_budget_dropped_sections",
        "context_compaction_scope",
        "context_compaction_source_sections",
        "context_compaction_replaced_sections",
    }
)
_LEGACY_DICT_FIELDS = frozenset(
    {
        "adaptive_helper_unavailable_action_reasons",
        "adaptive_informativeness_pregate_features",
        "context_compaction_source_chars",
    }
)


def _build_attempt_metadata_request_from_legacy_values(
    legacy_values: Mapping[str, Any],
) -> AttemptMetadataRequest:
    """Validate and group values accepted by the deprecated flat adapter."""
    unexpected = sorted(set(legacy_values) - _ALLOWED_LEGACY_FIELDS)
    if unexpected:
        if len(unexpected) == 1:
            raise TypeError(
                "build_attempt_metadata_request_from_legacy_kwargs() got an unexpected "
                f"keyword argument {unexpected[0]!r}"
            )
        rendered = ", ".join(repr(name) for name in unexpected)
        raise TypeError(
            "build_attempt_metadata_request_from_legacy_kwargs() got unexpected keyword "
            f"arguments: {rendered}"
        )

    missing = sorted(_REQUIRED_LEGACY_FIELDS - set(legacy_values))
    if missing:
        rendered = ", ".join(repr(name) for name in missing)
        noun = "argument" if len(missing) == 1 else "arguments"
        raise TypeError(
            "build_attempt_metadata_request_from_legacy_kwargs() missing required keyword "
            f"{noun}: {rendered}"
        )

    values = dict(legacy_values)
    for field_name in _LEGACY_LIST_FIELDS:
        if field_name in values:
            values[field_name] = list(values[field_name] or [])
    for field_name in _LEGACY_DICT_FIELDS:
        if field_name in values:
            values[field_name] = dict(values[field_name] or {})

    grouped_values: dict[str, Any] = {}
    for group_name, metadata_type in _GROUPED_METADATA_TYPES:
        group_values = {
            field_info.name: values.pop(field_info.name)
            for field_info in fields(metadata_type)
            if field_info.name in values
        }
        grouped_values[group_name] = metadata_type(**group_values)

    top_level_values = {
        field_name: values.pop(field_name)
        for field_name in _TOP_LEVEL_LEGACY_FIELDS
        if field_name in values
    }
    top_level_values["evidence"] = dict(top_level_values["evidence"])
    for field_name in ("prompt", "record_prompt", "record_evidence", "context_protocol"):
        top_level_values[field_name] = str(top_level_values[field_name])
    for field_name in ("timing_breakdown", "token_breakdown"):
        value = top_level_values.get(field_name)
        if value is not None:
            top_level_values[field_name] = dict(value)

    return AttemptMetadataRequest(**top_level_values, **grouped_values)


def build_attempt_metadata_request_from_legacy_kwargs(
    **legacy_kwargs: Any,
) -> AttemptMetadataRequest:
    """Build a grouped request through the deprecated flat keyword interface."""
    warnings.warn(
        (
            "build_attempt_metadata_request_from_legacy_kwargs(**kwargs) is deprecated; "
            "construct AttemptMetadataRequest directly"
        ),
        DeprecationWarning,
        stacklevel=2,
    )
    return _build_attempt_metadata_request_from_legacy_values(legacy_kwargs)
