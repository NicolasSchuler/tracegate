"""Shared TypedDict contracts for patcher input/output payloads.

These types define the structured data exchanged between patcher components
and their callers, covering LLM token usage, context-gathering requests, and
per-run diagnostic metadata.
"""

from __future__ import annotations

from typing import Any, TypedDict


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


class PatcherMeta(TypedDict, total=False):
    """Diagnostic metadata emitted by a single patcher run.

    Captures subprocess outcomes, LLM configuration, token usage, retry
    state, fallback transitions, and forced-patch override details so that
    every run can be fully reconstructed from its metadata alone.

    Attributes:
        error (str): Human-readable error message if the run failed.
        returncode (int): Process exit code from the patcher subprocess.
        stdout (str): Captured standard output from the patcher subprocess.
        stderr (str): Captured standard error from the patcher subprocess.
        timed_out (bool): Whether the run exceeded its time limit.
        timeout_sec (float): Time limit in seconds that was applied to the run.
        llm_call_count (int): Number of LLM API calls made during the run.
        usage (UsagePayload): Aggregated token-usage statistics for the run.
        status (int): HTTP status code from the last LLM API response.
        body (str): Raw response body from the last LLM API call.
        exc (str): Serialised exception string if an unexpected error was raised.
        attempt (int): Retry-attempt number (1-based) for this patcher invocation.
        base_url (str): Base URL of the LLM API endpoint used.
        model (str): Identifier of the LLM model used.
        response_mode (str): Sampling or response strategy requested from the LLM.
        temperature (float): Sampling temperature passed to the LLM.
        max_tokens (int): Maximum token budget allowed for the LLM response.
        max_propose_sec (float): Time limit in seconds for the patch-proposal phase.
        request_context_duration_sec (float): Wall-clock time spent gathering context.
        fallback_from (str): Strategy name from which execution fell back.
        fallback_reason (str): Explanation of why the fallback was triggered.
        forced_patch_attempt (bool): Whether a forced-patch override was attempted.
        forced_patch_target (str): Path or identifier targeted by the forced patch.
        forced_patch_reason (str): Reason the forced patch was applied.
        forced_patch_skipped (bool): Whether the forced-patch attempt was skipped.
        forced_patch_skip_reason (str): Short reason code for skipping the forced patch.
        forced_patch_skip_detail (str): Detailed explanation for skipping the forced patch.
        reason (str): General-purpose reason string describing the patcher outcome.
    """

    error: str
    returncode: int
    stdout: str
    stderr: str
    timed_out: bool
    timeout_sec: float
    llm_call_count: int
    usage: UsagePayload
    status: int
    body: str
    exc: str
    attempt: int
    base_url: str
    model: str
    response_mode: str
    temperature: float
    max_tokens: int
    max_propose_sec: float
    request_context_duration_sec: float
    fallback_from: str
    fallback_reason: str
    forced_patch_attempt: bool
    forced_patch_target: str
    forced_patch_reason: str
    forced_patch_skipped: bool
    forced_patch_skip_reason: str
    forced_patch_skip_detail: str
    reason: str
    infra_failure_kind: str
    infra_resume_eligible: bool
    infra_source: str


JSONDict = dict[str, Any]


__all__ = [
    "ContextRequestEntry",
    "ContextRequestPayload",
    "JSONDict",
    "PatcherMeta",
    "UsagePayload",
]
