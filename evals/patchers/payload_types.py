"""Shared TypedDict contracts for patcher input/output payloads.

These types define the structured data exchanged between patcher components
and their callers, covering LLM token usage, context-gathering requests, and
per-run diagnostic metadata.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, TypedDict, cast


class UsagePayload(TypedDict, total=False):
    """Token-usage statistics reported by an LLM API call.

    Attributes:
        prompt_tokens (int): Number of tokens consumed by the prompt.
        completion_tokens (int): Number of tokens produced in the completion.
        total_tokens (int): Combined prompt and completion token count.
        reasoning_tokens (int): Provider-reported reasoning/thinking token count
            when explicitly exposed. This is informational and treated as a
            subset of completion/output tokens rather than an additive metric.
    """

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    reasoning_tokens: int


class ContextRequestEntry(TypedDict, total=False):
    """A single item in a context-gathering request sent to the repository.

    Attributes:
        type (str): Category of context to retrieve
            (e.g. ``"file"``, ``"function"``, ``"variable"``).
        path (str): File path associated with the requested context.
        pattern (str): Search pattern used to locate the relevant code.
        index (int): Numeric index selecting a specific item within the result set.
        frame (int): Stack-frame index for context requests derived from tracebacks.
        name (str): Identifier name (variable, function, or class) being requested.
    """

    type: str
    path: str
    pattern: str
    index: int
    frame: int
    name: str


class ContextRequestPayload(TypedDict, total=False):
    """Full payload returned by the patcher when deciding what context it needs.

    Attributes:
        diagnosis (str): One-sentence root-cause hypothesis derived from the traceback.
        missing_fact (str | None): Specific piece of information that is still unknown,
            or ``None`` if no additional fact is needed.
        need_more_context (bool): Whether the patcher requires additional context
            before proposing a patch.
        need_snapshot (bool): Whether a full repository snapshot is requested.
        need_compact_context (bool): Whether a compact (summarised) context view
            is requested instead of raw file contents.
        requests (list[ContextRequestEntry]): Ordered list of individual context
            items to retrieve.
        snapshot_justification (str): Explanation of why a full snapshot is required.
        compact_context_reason (str): Explanation of why compact context is preferred.
        sufficiency_rationale (str): Explanation of why the current context is
            (or is not) sufficient to propose a patch.
    """

    diagnosis: str  # P0.1: one-sentence root-cause hypothesis from traceback
    missing_fact: str | None  # P0.1: specific missing fact, or null if none needed
    need_more_context: bool
    need_snapshot: bool
    need_compact_context: bool
    requests: list[ContextRequestEntry]
    snapshot_justification: str
    compact_context_reason: str
    sufficiency_rationale: str


class StructuredOutputAttempt(TypedDict, total=False):
    """One structured-output stage tried while proposing edits."""

    mode: str
    ok: bool
    error: str
    status: int | None
    terminal: bool
    parse_error: str
    edits_shape: str


class ProcessMeta(TypedDict, total=False):
    """The outcome of a patcher run: error, subprocess result and HTTP status."""

    error: str
    reason: str
    exc: str
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool
    timeout_sec: float
    status: int
    body: str
    path: str
    file: str
    attempt: int


class ModelRequestMeta(TypedDict, total=False):
    """The model call: endpoint, sampling settings, usage and API-compat seed."""

    base_url: str
    model: str
    response_mode: str
    temperature: float
    max_tokens: int
    max_propose_sec: float
    llm_call_count: int
    usage: UsagePayload
    finish_reason: str
    compat_profile_requested: str
    compat_profile_applied: str
    compat_message: str
    direct_output_requested: bool
    allow_reasoning_fallback: bool
    enable_thinking: bool
    content_present: bool
    empty_content_with_reasoning: bool
    used_reasoning_fallback: bool
    final_text_chars: int
    api_compat_seed_requested: int | None
    api_compat_seed_acknowledged: bool | None
    api_compat_seed_response: int | None


class TransportMeta(TypedDict, total=False):
    """HTTP transport: timeouts, retries and protocol version."""

    effective_timeout_sec: float
    remaining_budget_sec: float | None
    transport_impl: str
    transport_http2_requested: bool
    transport_http2_effective: bool
    transport_http2_fallback_used: bool
    transport_http_version: str
    transport_max_retries: int
    transport_retry_limit: int
    transport_retry_limit_effective: int
    transport_retry_backoff_base_sec: float
    transport_retry_backoff_max_sec: float
    transport_retry_count: int
    transport_retry_attempts: list[dict[str, Any]]
    transport_error: str
    retry_after_sec: float


class ResponseMeta(TypedDict, total=False):
    """What came back: raw text kept for diagnosis, parse errors and reasoning."""

    response: str
    raw_response: str
    payload: str
    parse_error: str
    reasoning_content_present: bool
    reasoning_content_chars: int
    reasoning_present: bool
    reasoning_chars: int
    structured_output_mode: str
    structured_output_attempts: list[StructuredOutputAttempt]
    edits_payload_shape: str


class RequestContextMeta(TypedDict, total=False):
    """Stage A: the context request and its retries."""

    request_context_duration_sec: float
    request_context_attempts: list[dict[str, Any]]
    request_context_retry_count: int
    request_context_transport_retry_count: int
    request_context_strict_retry_count: int
    request_context_strict_json: bool
    request_context_strict_json_first: bool
    request_context_response_format_fallback_used: bool
    request_context_outcome: str
    request_context_failure_kind: str
    request_context_backend_unavailable: bool
    request_context_effective_success: bool


class GenerationMeta(TypedDict, total=False):
    """Auxiliary generations: patch plan, free text, JSON and context compaction."""

    plan_patch_max_tokens_requested: int
    plan_patch_max_tokens: int
    plan_patch_duration_sec: float
    plan_patch_output_chars: int
    text_generation_operation: str
    text_generation_duration_sec: float
    text_generation_max_tokens: int
    json_generation_operation: str
    json_generation_duration_sec: float
    json_generation_max_tokens: int
    compact_context_duration_sec: float
    compact_context_target_chars: int
    compact_context_input_chars: int
    compact_context_output_chars: int
    compact_context_output_truncated: bool


class EditsMeta(TypedDict, total=False):
    """Edit and full-file response handling."""

    find_recovery_applied: bool
    find_recovery_mode: str
    find_recovery_modes: list[str]
    full_file_supported: bool
    full_file_support_reason: str
    full_file_target_path: str
    full_file_code_fence_stripped: bool
    full_file_output_chars: int


class FallbackMeta(TypedDict, total=False):
    """A fallback to another response mode, or a runner-forced patch."""

    fallback_from: str
    fallback_reason: str
    fallback_meta: PatcherMeta
    fallback_trigger: PatcherMeta
    edits_response_mode: str
    edits_response_attempts: list[StructuredOutputAttempt]
    forced_patch_attempt: bool
    forced_patch_target: str
    forced_patch_reason: str
    forced_patch_skipped: bool
    forced_patch_skip_reason: str
    forced_patch_skip_detail: str


class ProbeMeta(TypedDict, total=False):
    """Server-readiness and context-window probes."""

    server_ready: bool
    server_ready_probe_provider: str
    server_ready_probe_url: str
    server_ready_probe_attempts: list[dict[str, Any]]
    server_ready_probe_error: str
    server_ready_probe_duration_sec: float
    context_window_source: str
    context_window_probe_provider: str
    context_window_probe_url: str
    context_window_probe_attempts: list[dict[str, Any]]
    context_window_probe_error: str
    context_window_probe_duration_sec: float


class InfraMeta(TypedDict, total=False):
    """Infrastructure failures that make a unit eligible for resume."""

    infra_failure_kind: str
    infra_resume_eligible: bool
    infra_source: str


class PatcherMeta(
    ProcessMeta,
    ModelRequestMeta,
    TransportMeta,
    ResponseMeta,
    RequestContextMeta,
    GenerationMeta,
    EditsMeta,
    FallbackMeta,
    ProbeMeta,
    InfraMeta,
    total=False,
):
    """Diagnostic metadata emitted by a single patcher run (X-03).

    Composed per concern so each part names the keys one stage writes. Callers
    still add keys of their own (the harness merges patcher metadata into the
    attempt's ``patch_meta``), so readers should use ``.get``.
    """


def copy_patcher_meta(meta: Mapping[str, Any] | None) -> PatcherMeta:
    """A shallow copy of *meta* (empty for ``None``), typed as :class:`PatcherMeta`.

    Patcher metadata also arrives as plain dicts (the baseline compat helpers can't
    import this module), so the copy is where it takes on the declared type.
    """
    return cast(PatcherMeta, dict(meta or {}))


def as_patcher_meta(meta: dict[str, Any]) -> PatcherMeta:
    """*meta* itself (not a copy), typed as :class:`PatcherMeta`.

    For metadata the harness builds with keys of its own and keeps updating after it
    is attached to a result.
    """
    return cast(PatcherMeta, meta)


JSONDict = dict[str, Any]


__all__ = [
    "ContextRequestEntry",
    "ContextRequestPayload",
    "EditsMeta",
    "FallbackMeta",
    "GenerationMeta",
    "InfraMeta",
    "JSONDict",
    "ModelRequestMeta",
    "PatcherMeta",
    "ProbeMeta",
    "ProcessMeta",
    "RequestContextMeta",
    "ResponseMeta",
    "StructuredOutputAttempt",
    "TransportMeta",
    "UsagePayload",
    "as_patcher_meta",
    "copy_patcher_meta",
]
