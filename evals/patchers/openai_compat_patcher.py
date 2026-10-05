"""LLM-based patch generation using an OpenAI-compatible chat-completions API.

Implements the :class:`~evals.patchers.base.Patcher` protocol by routing every
patch-generation request to any server that speaks the OpenAI
``/v1/chat/completions`` API — vLLM, LM Studio, llama.cpp, OpenAI, and
compatible local or remote endpoints.

Three response modes are supported:

* ``"edits"`` *(default)* — the model returns a structured JSON payload
  ``{"edits": [{"path": ..., "find": ..., "replace": ...}]}`` that is applied
  as a series of find-replace operations directly against the source files.  The
  patcher derives a unified diff from the resulting file changes.  Three
  ``response_format`` variants are attempted in order (``json_schema`` →
  ``json_object`` → plain text) and, if none succeeds, the patcher falls back
  to ``"diff"`` mode.
* ``"diff"`` — the model returns a raw unified diff which the patcher extracts
  and normalises.  When no parseable diff is found the patcher falls back to
  ``"edits"`` mode.
* ``"full_file"`` — when the case has exactly one non-test Python source file,
  the model returns the full repaired contents of that file and the patcher
  derives a unified diff locally. Unsupported case shapes and unusable
  responses fall back to ``"diff"`` mode.

Optional multi-stage context-gathering is also supported:

* :meth:`~OpenAICompatPatcher.request_context` /
  :meth:`~OpenAICompatPatcher.arequest_context` — Stage A: ask the model which
  additional files, snapshots, or other artefacts it needs before generating a
  patch, and return the request as a structured :data:`ContextRequestPayload`.
* :meth:`~OpenAICompatPatcher.compact_context` /
  :meth:`~OpenAICompatPatcher.acompact_context` — compress a large accumulated
  context block down to a caller-supplied character budget.
* :meth:`~OpenAICompatPatcher.plan_patch` — ask the model for a concise
  natural-language repair plan that can be prepended to subsequent prompts to
  steer it away from repeated mistakes.

Both synchronous and ``async`` variants are provided for every public operation
so that the patcher can be embedded in both threaded and async eval runners
without blocking the event loop.

Example::

    patcher = OpenAICompatPatcher(
        base_url="http://127.0.0.1:1234",
        model="my-model",
        response_mode="edits",
    )
    result = patcher.propose_patch(prompt=..., context_dir=..., cwd=...)
    if result.diff:
        ...
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

from baselines.openai_compat import (
    normalize_compat_profile,
)
from evals.patchers.openai_diff_text import (  # noqa: F401
    _DIFF_META_PREFIXES,
    _HUNK_RE,
    _MARKDOWN_BULLET_RE,
    _clean_path_token,
    _extract_unified_diff,
    _fix_hunk_counts,
    _git_file_block,
    _is_file_header_pair,
    _iter_file_diffs,
    _looks_like_raw_unified_diff,
    _read_hunks,
    _strip_think,
)
from evals.patchers.openai_edits_apply import (  # noqa: F401
    _apply_find_replace_with_recovery,
    _apply_fuzzy_block_replacement,
    _build_forced_attempt_diff,
    _compile_whitespace_normalized_pattern,
    _extract_file_index_paths,
    _looks_like_edit_entry,
    _normalize_edits_payload,
    _normalize_full_file_response,
    _normalize_newlines,
    _read_edit_target,
    _record_find_recovery,
    _resolve_edits,
    _resolve_full_file_target,
    _ResolvedEdits,
    _with_line_endings,
)
from evals.patchers.openai_probes import (  # noqa: F401
    _coerce_positive_int,
    _entry_matches_model,
    _extract_lmstudio_instance_context_length,
    _extract_lmstudio_v0_context_window,
    _extract_lmstudio_v1_context_window,
    _extract_model_entries,
    _extract_openai_models_context_window,
    _http_get_json,
    _http_get_json_async,
    _http_probe_ready,
    _http_probe_ready_async,
    _probe_context_window_tokens,
    _probe_context_window_tokens_async,
    _probe_server_ready,
    _probe_server_ready_async,
)
from evals.patchers.openai_response import (  # noqa: F401
    _accumulate_llm_totals,
    _coerce_content_to_text,
    _coerce_payload_bool,
    _error_meta,
    _extract_content_part_text,
    _extract_first_fenced_code_block,
    _extract_json_object,
    _extract_response_seed,
    _get_first_choice_content,
    _get_first_choice_finish_reason,
    _get_first_choice_reasoning,
    _get_first_choice_reasoning_content,
    _is_truncated_response,
    _llm_call_count_from_meta,
    _meta_with_llm_totals,
    _nonnegative_int,
    _optional_nonnegative_int,
    _reasoning_tokens_from_payload,
    _safe_trunc,
    _usage_counts_from_payload,
    _usage_payload_from_counts,
)
from evals.patchers.openai_transport import (  # noqa: F401
    _ASYNC_HTTP_CLIENTS,
    _ASYNC_HTTP_CLIENTS_LOCK,
    _BACKEND_UNAVAILABLE_HTTP_STATUSES,
    _CHAT_COMPLETION_RETRYABLE_HTTP_STATUSES,
    _DEFAULT_CHAT_COMPLETION_RETRY_BACKOFF_BASE_SEC,
    _DEFAULT_CHAT_COMPLETION_RETRY_BACKOFF_MAX_SEC,
    _DEFAULT_CHAT_COMPLETION_TRANSPORT_MAX_RETRIES,
    _DEFAULT_HTTPX_KEEPALIVE_EXPIRY_SEC,
    _DEFAULT_HTTPX_MAX_CONNECTIONS,
    _DEFAULT_HTTPX_MAX_KEEPALIVE_CONNECTIONS,
    _HTTP2_PROTOCOL_ERRORS,
    _OPENAI_COMPAT_REASONING_ONLY_ERROR,
    _OPENAI_PROGRESS_BASE_PAYLOAD,
    _OPENAI_PROGRESS_CALLBACK,
    _RETRYABLE_HTTP_ERRORS,
    _THREAD_HTTP_CLIENTS,
    _apply_eval_openai_compat_request_profile,
    _apply_reasoning_fallback_to_response_payload,
    _base_url_from_chat_url,
    _chat_completion,
    _chat_completion_async,
    _chat_completion_retry_backoff_sec,
    _chat_http_client_key,
    _chat_http_target,
    _emit_openai_transport_progress,
    _eval_openai_compat_config,
    _get_async_http_client,
    _get_thread_http_client,
    _is_backend_unavailable_error,
    _is_context_exceeded_http_error,
    _is_http2_protocol_error,
    _is_retryable_chat_completion_error,
    _is_retryable_chat_completion_http_status,
    _merge_extra_body_into_payload,
    _normalize_base_url,
    _origin_from_base_url,
    _PersistentAsyncHTTPClient,
    _PersistentHTTPClient,
    _remaining_request_budget_sec,
    _should_use_proxy_transport,
    _with_openai_transport_progress,
    _with_openai_transport_progress_async,
)

from ..eval_patch_constants import TRUNCATED_RESPONSE_ERROR
from ..eval_patches import _build_diff_from_changes
from ..stage_a_contract import (
    STAGE_A_REQUEST_SCHEMA_COMPACT,
    normalize_stage_a_request_payload,
    stage_a_allowed_request_types_text,
)
from .base import (
    ContextCompactionResult,
    ContextRequestResult,
    Patcher,
    PatchPlanResult,
    PatchResult,
)
from .payload_types import (
    ContextRequestPayload,
    PatcherMeta,
    StructuredOutputAttempt,
    copy_patcher_meta,
)

OPENAI_API_KEY_ENV_FALLBACKS = ("Z_API", "OPENAI_API_KEY", "LLMDEBUG_EVAL_OPENAI_API_KEY")
_REQUEST_CONTEXT_MAX_TRANSPORT_REISSUES = 1
_REQUEST_CONTEXT_MAX_STRICT_JSON_RETRIES = 1
_REQUEST_CONTEXT_STRICT_JSON_MAX_TOKENS = 768
_PLAN_PATCH_MAX_TOKENS = 768
_DEFAULT_MAX_PROPOSE_SEC = 300.0


@dataclass(frozen=True)
class TextGenerationResult:
    text: str | None
    raw_response: str | None = None
    meta: PatcherMeta | None = None


@dataclass(frozen=True)
class JsonGenerationResult:
    payload: dict[str, Any] | None
    raw_response: str | None = None
    meta: PatcherMeta | None = None


PATCH_EDITS_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "required": ["edits"],
    "additionalProperties": False,
    "properties": {
        "edits": {
            "type": "array",
            "minItems": 1,
            "items": {
                "type": "object",
                "required": ["path", "find", "replace"],
                "additionalProperties": False,
                "properties": {
                    "path": {"type": "string", "minLength": 1},
                    "find": {"type": "string", "minLength": 1},
                    "replace": {"type": "string"},
                    "count": {"type": "integer", "minimum": 1},
                },
            },
        }
    },
}


def _first_nonempty_env(*keys: str) -> str:
    for key in keys:
        value = os.getenv(str(key), "")
        if isinstance(value, str) and value.strip():
            return value
    return ""


def _default_api_key() -> str:
    return _first_nonempty_env(*OPENAI_API_KEY_ENV_FALLBACKS) or "local"


def _request_context_system_prompt(*, strict_json: bool) -> str:
    base = (
        "Return only JSON for staged context requests. "
        "Use this schema exactly: "
        + STAGE_A_REQUEST_SCHEMA_COMPACT
        + " "
        + f"Allowed request types: {stage_a_allowed_request_types_text()}. "
        "Set need_compact_context=true only when compacted context is needed instead of raw bulk context. "
        "Use need_more_context=false with empty requests when evidence is sufficient. "
        "No prose and no markdown."
    )
    if not strict_json:
        return base
    return (
        base + " JSON only. Do not explain. Do not wrap the object in markdown. "
        "If unsure, return a valid object with need_more_context=false and requests=[]."
    )


def _plan_patch_system_prompt() -> str:
    return (
        "You are a software repair planner. "
        "Return only a concise plain-text repair plan with these headings exactly: "
        "Diagnosis, Why previous attempt failed, New constraint, Minimal edit plan, "
        "Verification plan. "
        "Do not return code, a diff, markdown fences, or any extra commentary."
    )


@dataclass(frozen=True)
class OpenAICompatPatcher(Patcher):
    """OpenAI API-compatible chat-completions patcher.

    Routes patch-generation and context-gathering calls to any server that
    implements the OpenAI ``/v1/chat/completions`` REST API.  Tested against
    vLLM, LM Studio, llama.cpp, and the upstream OpenAI service.

    The class is an immutable :func:`~dataclasses.dataclass`; all fields are
    set at construction time and serve as the patcher's configuration.

    Attributes:
        base_url: Base URL of the OpenAI-compatible API server, e.g.
            ``"http://127.0.0.1:1234"``.  A ``/v1`` suffix is appended
            automatically when the URL does not already end with a versioned
            path segment.
        model: Model identifier to pass in every ``chat/completions`` request,
            e.g. ``"mistral-7b-instruct"`` or ``"gpt-4o"``.
        api_key: Bearer token sent in the ``Authorization`` header.  When
            ``None`` the value is resolved from the environment variables
            ``Z_API``, ``OPENAI_API_KEY``, or
            ``LLMDEBUG_EVAL_OPENAI_API_KEY`` (first non-empty wins), falling
            back to the string ``"local"``.
        timeout_sec: Per-request HTTP timeout in seconds.  Defaults to
            ``120.0``.
        http2: Whether to attempt an HTTP/2 connection.  Falls back to HTTP/1.1
            automatically on protocol negotiation errors.  Defaults to ``True``.
        transport_max_retries: Maximum number of additional transport-layer
            retry attempts on transient errors (connection resets, timeouts,
            retryable HTTP status codes such as 429/503).  Defaults to
            ``_DEFAULT_CHAT_COMPLETION_TRANSPORT_MAX_RETRIES`` (6).
        transport_retry_backoff_base_sec: Base delay in seconds for exponential
            back-off between transport retries.  Defaults to ``1.0``.
        transport_retry_backoff_max_sec: Upper cap on the exponential back-off
            delay.  Defaults to ``8.0``.
        max_propose_sec: Wall-clock deadline (seconds) for the entire
            :meth:`propose_patch` / :meth:`apropose_patch` call, measured from
            the moment the method is entered.  A value of ``0`` disables the
            deadline.  Defaults to ``300.0``.
        temperature: Sampling temperature forwarded to the model.  Use ``0.0``
            for deterministic (greedy) decoding.  Defaults to ``0.0``.
        max_tokens: Maximum number of tokens to generate per request.  Defaults
            to ``2048``.
        plan_patch_max_tokens: Native ``cot_repair`` planner token cap used by
            :meth:`plan_patch`. Defaults to ``768``.
        compat_profile: OpenAI-compatible request-shaping profile to apply
            before each request. ``"qwen_openai_raw"`` sends
            ``chat_template_kwargs.enable_thinking=false``;
            ``"qwen_dashscope"`` sends top-level ``enable_thinking=false``;
            ``"lmstudio_model_yaml"`` and ``"off"`` do not inject request-side
            controls. ``"auto"`` preserves existing default behaviour.
        allow_reasoning_fallback: When ``True``, reasoning-only responses are
            normalized into downstream content instead of surfacing a
            compatibility error. Defaults to ``False``.
        enable_thinking: Whether to request server-side thinking/reasoning mode
            when the chosen compatibility profile supports it. Enabling this
            also auto-enables reasoning fallback. Defaults to ``False``.
        response_mode: Controls the patch-generation strategy.  ``"edits"``
            (default) requests a structured JSON edits payload and builds the
            diff locally; ``"diff"`` requests a raw unified diff from the model.
        force_patch_attempt: When ``True`` and the model produces no usable
            diff, a deterministic minimal patch (appending a marker comment to
            the first non-test ``.py`` file) is constructed as a last resort.
            Defaults to ``False``.
        seed: Optional integer seed forwarded to the model for reproducibility.
            Not all servers honour this field.  Defaults to ``None``.
        name: Symbolic name for this patcher instance, used in run logs.
            Defaults to ``"openai"``.
    """

    base_url: str
    model: str
    api_key: str | None = None
    timeout_sec: float = 120.0
    http2: bool = True
    transport_max_retries: int = _DEFAULT_CHAT_COMPLETION_TRANSPORT_MAX_RETRIES
    transport_retry_backoff_base_sec: float = _DEFAULT_CHAT_COMPLETION_RETRY_BACKOFF_BASE_SEC
    transport_retry_backoff_max_sec: float = _DEFAULT_CHAT_COMPLETION_RETRY_BACKOFF_MAX_SEC
    max_propose_sec: float = _DEFAULT_MAX_PROPOSE_SEC
    temperature: float = 0.0
    max_tokens: int = 2048
    plan_patch_max_tokens: int = _PLAN_PATCH_MAX_TOKENS
    compat_profile: str = "auto"
    allow_reasoning_fallback: bool = False
    enable_thinking: bool = False
    response_mode: str = "edits"  # "edits" | "diff" | "full_file"
    force_patch_attempt: bool = False
    seed: int | None = None
    name: str = "openai"

    def supports_request_context(self) -> bool:
        """OpenAI-compatible patchers implement Stage-A helper requests."""
        return True

    def supports_context_compaction(self) -> bool:
        """OpenAI-compatible patchers implement delegated context compaction."""
        return True

    def supports_strict_json_first(self) -> bool:
        """OpenAI-compatible patchers support strict-JSON-first helper retries."""
        return True

    @_with_openai_transport_progress(
        external_operation="request_context",
        phase_label="openai: request context",
    )
    def request_context(
        self,
        *,
        prompt: str,
        context_dir: Path,
        cwd: Path,
        strict_json_first: bool = False,
    ) -> ContextRequestResult | None:
        """Ask the model which additional context it needs before patching.

        Stage A of the context-gathering pipeline.  Sends *prompt* to the
        ``chat/completions`` endpoint with a system instruction that constrains
        the response to the :data:`~evals.stage_a_contract.STAGE_A_REQUEST_SCHEMA_COMPACT`
        JSON schema.  The parsed payload declares which files, runtime
        snapshots, or other artefacts should be fetched and injected into the
        prompt before the model attempts to generate a patch.

        The method retries automatically on transient transport failures.  When
        the model returns a response that cannot be parsed as a valid
        :data:`ContextRequestPayload`, one additional attempt is made using a
        shorter ``strict_json`` system prompt and a reduced token budget.

        Args:
            prompt: The assembled debugging/repair prompt for the current
                repair attempt.
            context_dir: Directory containing pre-cached context artefacts.
                Not used by this implementation; reserved for the protocol.
            cwd: Working directory of the case under repair.  Not used by this
                implementation; reserved for the protocol.
            strict_json_first: When ``True`` the very first attempt already
                uses the strict-JSON system prompt and a reduced token budget
                (``_REQUEST_CONTEXT_STRICT_JSON_MAX_TOKENS``).  Useful when
                the model is known to be reliable at structured output.

        Returns:
            A :class:`~evals.patchers.base.ContextRequestResult` containing:

            - ``request_json``: the parsed :data:`ContextRequestPayload` on
              success, or ``None`` on failure.
            - ``raw_response``: the raw model output string.
            - ``meta``: telemetry dict with timing, retry counts, token usage,
              outcome classification (``request_context_outcome``), and
              per-attempt details (``request_context_attempts``).
        """
        del cwd
        base_url = _normalize_base_url(self.base_url)
        url = f"{base_url}/chat/completions"
        api_key = self.api_key
        if api_key is None:
            api_key = _default_api_key()

        start = time.perf_counter()
        totals = (0, 0, 0, 0, 0, False, True)
        helper_attempts: list[dict[str, Any]] = []
        transport_reissues_used = 0
        strict_json_retries_used = 0
        strict_json = bool(strict_json_first)
        raw_response: str | None = None
        request_json: ContextRequestPayload | None = None
        meta: PatcherMeta = {}

        while True:
            raw, attempt_meta = _request_context_via_json(
                url=url,
                api_key=api_key,
                model=self.model,
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=(
                    max(1, min(int(self.max_tokens), _REQUEST_CONTEXT_STRICT_JSON_MAX_TOKENS))
                    if strict_json
                    else self.max_tokens
                ),
                compat_profile=self.compat_profile,
                allow_reasoning_fallback=self.allow_reasoning_fallback,
                enable_thinking=self.enable_thinking,
                timeout_sec=self.timeout_sec,
                seed=self.seed,
                http2=self.http2,
                strict_json=strict_json,
                transport_max_retries=self.transport_max_retries,
                transport_retry_backoff_base_sec=self.transport_retry_backoff_base_sec,
                transport_retry_backoff_max_sec=self.transport_retry_backoff_max_sec,
            )
            totals = _accumulate_llm_totals(totals=totals, meta=attempt_meta, default_calls=0)
            meta = cast(PatcherMeta, dict(attempt_meta or {}))
            request_json, raw_response = _normalize_context_request_payload(raw)
            failure_kind = _request_context_failure_kind(
                meta=meta,
                request_json=request_json,
                raw_response=raw_response,
            )
            outcome = _request_context_outcome(
                meta=meta,
                request_json=request_json,
                raw_response=raw_response,
            )
            helper_attempts.append(
                {
                    "attempt": len(helper_attempts) + 1,
                    "strict_json": bool(strict_json),
                    "ok": request_json is not None,
                    "outcome": outcome,
                    "failure_kind": failure_kind,
                    "error": str(meta.get("error") or ""),
                    "status": meta.get("status"),
                    "llm_call_count": _llm_call_count_from_meta(meta, default=0),
                    "transport_retry_count": _nonnegative_int(
                        meta.get("transport_retry_count"), default=0
                    ),
                    "response_format_fallback_used": bool(
                        meta.get("request_context_response_format_fallback_used")
                    ),
                }
            )
            if request_json is not None:
                break
            if _is_openai_compat_reasoning_only(meta):
                break
            if (
                failure_kind in {"transport", "backend_unavailable"}
                and transport_reissues_used < _REQUEST_CONTEXT_MAX_TRANSPORT_REISSUES
            ):
                transport_reissues_used += 1
                strict_json = False
                continue
            if (
                failure_kind == "invalid_payload"
                and strict_json_retries_used < _REQUEST_CONTEXT_MAX_STRICT_JSON_RETRIES
                and not strict_json
            ):
                strict_json_retries_used += 1
                strict_json = True
                continue
            break
        duration_sec = time.perf_counter() - start

        result_meta: PatcherMeta = copy_patcher_meta(meta)
        result_meta.setdefault("base_url", base_url)
        result_meta.setdefault("model", self.model)
        result_meta.setdefault("temperature", self.temperature)
        result_meta.setdefault("max_tokens", self.max_tokens)
        result_meta.setdefault(
            "compat_profile_requested", normalize_compat_profile(self.compat_profile)
        )
        result_meta.setdefault("response_mode", self.response_mode)
        result_meta.setdefault("api_compat_seed_requested", self.seed)
        result_meta.setdefault(
            "api_compat_seed_acknowledged", False if self.seed is not None else None
        )
        result_meta.setdefault("api_compat_seed_response", None)
        result_meta["request_context_duration_sec"] = duration_sec
        result_meta["request_context_attempts"] = helper_attempts
        result_meta["request_context_retry_count"] = max(0, len(helper_attempts) - 1)
        result_meta["request_context_transport_retry_count"] = transport_reissues_used
        result_meta["request_context_strict_retry_count"] = strict_json_retries_used
        result_meta["request_context_strict_json_first"] = bool(strict_json_first)
        failure_kind = _request_context_failure_kind(
            meta=result_meta,
            request_json=request_json,
            raw_response=raw_response,
        )
        outcome = _request_context_outcome(
            meta=result_meta,
            request_json=request_json,
            raw_response=raw_response,
        )
        result_meta["request_context_outcome"] = outcome
        result_meta["request_context_failure_kind"] = failure_kind
        result_meta["request_context_backend_unavailable"] = failure_kind == "backend_unavailable"
        result_meta["request_context_effective_success"] = request_json is not None
        result_meta = _meta_with_llm_totals(meta=result_meta, totals=totals)
        if request_json is None and raw_response is not None and "error" not in result_meta:
            result_meta["error"] = "invalid_request_context_payload"
        return ContextRequestResult(
            request_json=request_json,
            raw_response=raw_response,
            meta=result_meta,
        )

    @_with_openai_transport_progress(
        external_operation="plan_patch",
        phase_label="openai: plan patch",
    )
    def plan_patch(self, *, prompt: str, context_dir: Path, cwd: Path) -> PatchPlanResult | None:
        """Ask the model for a concise natural-language repair plan.

        Issues a single ``chat/completions`` request using a specialised system
        prompt that instructs the model to produce a plain-text plan structured
        around five headings: *Diagnosis*, *Why previous attempt failed*,
        *New constraint*, *Minimal edit plan*, and *Verification plan*.

        The returned plan text is intended to be prepended to the next
        :meth:`propose_patch` prompt so that the model is guided away from
        repeating the same mistake.  The token budget is capped at
        :attr:`plan_patch_max_tokens` regardless of :attr:`max_tokens`.

        Args:
            prompt: The repair/diagnostic prompt containing failure information
                from the previous attempt.
            context_dir: Directory containing pre-cached context artefacts.
                Not used by this implementation; reserved for the protocol.
            cwd: Working directory of the case under repair.  Not used by this
                implementation; reserved for the protocol.

        Returns:
            A :class:`~evals.patchers.base.PatchPlanResult` containing:

            - ``plan_text``: the stripped plan string on success, or ``None``
              when the model returns an empty response.
            - ``raw_response``: the raw model output string.
            - ``meta``: telemetry dict with timing, token usage, and the
              effective token cap used (``plan_patch_max_tokens``).
        """
        del cwd
        base_url = _normalize_base_url(self.base_url)
        url = f"{base_url}/chat/completions"
        api_key = self.api_key
        if api_key is None:
            api_key = _default_api_key()

        deadline_monotonic: float | None = None
        if self.max_propose_sec > 0:
            deadline_monotonic = time.monotonic() + self.max_propose_sec

        plan_max_tokens = max(1, min(int(self.max_tokens), int(self.plan_patch_max_tokens)))
        start = time.perf_counter()
        data, meta = _chat_completion(
            url=url,
            api_key=api_key,
            model=self.model,
            messages=[
                {"role": "system", "content": _plan_patch_system_prompt()},
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=plan_max_tokens,
            compat_profile=self.compat_profile,
            allow_reasoning_fallback=self.allow_reasoning_fallback,
            enable_thinking=self.enable_thinking,
            timeout_sec=self.timeout_sec,
            seed=self.seed,
            http2=self.http2,
            deadline_monotonic=deadline_monotonic,
            transport_max_retries=self.transport_max_retries,
            transport_retry_backoff_base_sec=self.transport_retry_backoff_base_sec,
            transport_retry_backoff_max_sec=self.transport_retry_backoff_max_sec,
        )
        duration_sec = time.perf_counter() - start
        result_meta: PatcherMeta = copy_patcher_meta(meta)
        result_meta.setdefault("base_url", base_url)
        result_meta.setdefault("model", self.model)
        result_meta.setdefault("temperature", self.temperature)
        result_meta.setdefault("timeout_sec", self.timeout_sec)
        result_meta.setdefault("transport_max_retries", self.transport_max_retries)
        result_meta.setdefault(
            "transport_retry_backoff_base_sec", self.transport_retry_backoff_base_sec
        )
        result_meta.setdefault(
            "transport_retry_backoff_max_sec", self.transport_retry_backoff_max_sec
        )
        result_meta.setdefault("max_propose_sec", self.max_propose_sec)
        result_meta.setdefault(
            "compat_profile_requested", normalize_compat_profile(self.compat_profile)
        )
        result_meta.setdefault("plan_patch_max_tokens_requested", int(self.plan_patch_max_tokens))
        result_meta.setdefault("api_compat_seed_requested", self.seed)
        result_meta.setdefault(
            "api_compat_seed_acknowledged", False if self.seed is not None else None
        )
        result_meta.setdefault("api_compat_seed_response", None)
        result_meta["plan_patch_duration_sec"] = duration_sec
        result_meta["plan_patch_max_tokens"] = plan_max_tokens

        raw_response = _get_first_choice_content(data) if isinstance(data, dict) else None
        plan_text = (
            raw_response.strip() if isinstance(raw_response, str) and raw_response.strip() else None
        )
        if plan_text is None and "error" not in result_meta:
            result_meta["error"] = "empty_plan_response"
        if plan_text is not None:
            result_meta["plan_patch_output_chars"] = len(plan_text)
        return PatchPlanResult(plan_text=plan_text, raw_response=raw_response, meta=result_meta)

    def generate_text(
        self,
        *,
        system_prompt: str,
        prompt: str,
        context_dir: Path,
        cwd: Path,
        temperature: float | None = None,
        max_tokens: int | None = None,
        operation_name: str = "generate_text",
    ) -> TextGenerationResult:
        del context_dir
        del cwd
        base_url = _normalize_base_url(self.base_url)
        url = f"{base_url}/chat/completions"
        api_key = self.api_key
        if api_key is None:
            api_key = _default_api_key()

        selected_temperature = self.temperature if temperature is None else float(temperature)
        selected_max_tokens = self.max_tokens if max_tokens is None else max(1, int(max_tokens))
        # LLM-11: the same max_propose_sec deadline as propose_patch and plan_patch, so
        # retries can't run a TEP or Stage-A call far past the attempt's budget.
        deadline_monotonic: float | None = None
        if self.max_propose_sec > 0:
            deadline_monotonic = time.monotonic() + self.max_propose_sec
        start = time.perf_counter()
        raw, meta = _chat_completion(
            url=url,
            api_key=api_key,
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=selected_temperature,
            max_tokens=selected_max_tokens,
            compat_profile=self.compat_profile,
            allow_reasoning_fallback=self.allow_reasoning_fallback,
            enable_thinking=self.enable_thinking,
            timeout_sec=self.timeout_sec,
            seed=self.seed,
            http2=self.http2,
            deadline_monotonic=deadline_monotonic,
            transport_max_retries=self.transport_max_retries,
            transport_retry_backoff_base_sec=self.transport_retry_backoff_base_sec,
            transport_retry_backoff_max_sec=self.transport_retry_backoff_max_sec,
        )
        duration_sec = time.perf_counter() - start
        result_meta: PatcherMeta = copy_patcher_meta(meta)
        result_meta.setdefault("base_url", base_url)
        result_meta.setdefault("model", self.model)
        result_meta.setdefault("temperature", selected_temperature)
        result_meta.setdefault("timeout_sec", self.timeout_sec)
        result_meta["text_generation_operation"] = operation_name
        result_meta["text_generation_duration_sec"] = duration_sec
        result_meta["text_generation_max_tokens"] = selected_max_tokens
        if raw is None:
            return TextGenerationResult(text=None, raw_response=None, meta=result_meta)
        content = _get_first_choice_content(raw)
        if content is None:
            result_meta.setdefault("error", "no_choice_content")
            return TextGenerationResult(text=None, raw_response=_safe_trunc(raw), meta=result_meta)
        return TextGenerationResult(text=content, raw_response=content, meta=result_meta)

    def generate_json(
        self,
        *,
        system_prompt: str,
        prompt: str,
        context_dir: Path,
        cwd: Path,
        schema_name: str,
        schema: dict[str, Any],
        temperature: float | None = None,
        max_tokens: int | None = None,
        operation_name: str = "generate_json",
    ) -> JsonGenerationResult:
        del context_dir
        del cwd
        base_url = _normalize_base_url(self.base_url)
        url = f"{base_url}/chat/completions"
        api_key = self.api_key
        if api_key is None:
            api_key = _default_api_key()

        selected_temperature = self.temperature if temperature is None else float(temperature)
        selected_max_tokens = self.max_tokens if max_tokens is None else max(1, int(max_tokens))
        # LLM-11: the same max_propose_sec deadline as propose_patch and plan_patch, so
        # retries can't run a TEP or Stage-A call far past the attempt's budget.
        deadline_monotonic: float | None = None
        if self.max_propose_sec > 0:
            deadline_monotonic = time.monotonic() + self.max_propose_sec
        start = time.perf_counter()
        totals = (0, 0, 0, 0, 0, False, True)
        raw, meta = _chat_completion(
            url=url,
            api_key=api_key,
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt},
            ],
            temperature=selected_temperature,
            max_tokens=selected_max_tokens,
            compat_profile=self.compat_profile,
            allow_reasoning_fallback=self.allow_reasoning_fallback,
            enable_thinking=self.enable_thinking,
            timeout_sec=self.timeout_sec,
            seed=self.seed,
            http2=self.http2,
            deadline_monotonic=deadline_monotonic,
            extra_payload={
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {"name": schema_name, "schema": schema},
                }
            },
            transport_max_retries=self.transport_max_retries,
            transport_retry_backoff_base_sec=self.transport_retry_backoff_base_sec,
            transport_retry_backoff_max_sec=self.transport_retry_backoff_max_sec,
        )
        totals = _accumulate_llm_totals(totals=totals, meta=meta, default_calls=1)
        if raw is None:
            err = str(meta.get("error", ""))
            body = str(meta.get("body", ""))
            if err == "http_error" and "response_format" in body:
                raw, meta = _chat_completion(
                    url=url,
                    api_key=api_key,
                    model=self.model,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=selected_temperature,
                    max_tokens=selected_max_tokens,
                    compat_profile=self.compat_profile,
                    allow_reasoning_fallback=self.allow_reasoning_fallback,
                    enable_thinking=self.enable_thinking,
                    timeout_sec=self.timeout_sec,
                    seed=self.seed,
                    http2=self.http2,
                    deadline_monotonic=deadline_monotonic,
                    extra_payload={"response_format": {"type": "json_object"}},
                    transport_max_retries=self.transport_max_retries,
                    transport_retry_backoff_base_sec=self.transport_retry_backoff_base_sec,
                    transport_retry_backoff_max_sec=self.transport_retry_backoff_max_sec,
                )
                totals = _accumulate_llm_totals(totals=totals, meta=meta, default_calls=1)
        duration_sec = time.perf_counter() - start
        result_meta: PatcherMeta = _meta_with_llm_totals(
            meta=cast(PatcherMeta, copy_patcher_meta(meta)),
            totals=totals,
        )
        result_meta.setdefault("base_url", base_url)
        result_meta.setdefault("model", self.model)
        result_meta.setdefault("temperature", selected_temperature)
        result_meta.setdefault("timeout_sec", self.timeout_sec)
        result_meta["json_generation_operation"] = operation_name
        result_meta["json_generation_duration_sec"] = duration_sec
        result_meta["json_generation_max_tokens"] = selected_max_tokens
        if raw is None:
            return JsonGenerationResult(payload=None, raw_response=None, meta=result_meta)
        content = _get_first_choice_content(raw)
        if content is None:
            result_meta.setdefault("error", "no_choice_content")
            return JsonGenerationResult(
                payload=None, raw_response=_safe_trunc(raw), meta=result_meta
            )
        if _is_truncated_response(raw):
            return JsonGenerationResult(
                payload=None,
                raw_response=content[:2000],
                meta=_error_meta(result_meta, TRUNCATED_RESPONSE_ERROR),
            )
        payload = _extract_json_object(content)
        if payload is None:
            result_meta.setdefault("error", "no_json_payload")
            return JsonGenerationResult(payload=None, raw_response=content[:2000], meta=result_meta)
        return JsonGenerationResult(payload=payload, raw_response=content, meta=result_meta)

    @_with_openai_transport_progress_async(
        external_operation="request_context",
        phase_label="openai: request context",
    )
    async def arequest_context(
        self,
        *,
        prompt: str,
        context_dir: Path,
        cwd: Path,
        strict_json_first: bool = False,
    ) -> ContextRequestResult | None:
        """Async variant of :meth:`request_context`.

        Ask the model which additional context it needs before patching,
        without blocking the running event loop.  Behaviour and return value
        are identical to :meth:`request_context`; see that method for full
        parameter and return-value documentation.

        Args:
            prompt: The assembled debugging/repair prompt for the current
                repair attempt.
            context_dir: Directory containing pre-cached context artefacts.
                Not used by this implementation; reserved for the protocol.
            cwd: Working directory of the case under repair.  Not used by this
                implementation; reserved for the protocol.
            strict_json_first: When ``True`` the very first attempt already
                uses the strict-JSON system prompt and a reduced token budget.

        Returns:
            A :class:`~evals.patchers.base.ContextRequestResult` — see
            :meth:`request_context` for field descriptions.
        """
        del context_dir
        del cwd
        base_url = _normalize_base_url(self.base_url)
        url = f"{base_url}/chat/completions"
        api_key = self.api_key
        if api_key is None:
            api_key = _default_api_key()

        start = time.perf_counter()
        totals = (0, 0, 0, 0, 0, False, True)
        helper_attempts: list[dict[str, Any]] = []
        transport_reissues_used = 0
        strict_json_retries_used = 0
        strict_json = bool(strict_json_first)
        raw_response: str | None = None
        request_json: ContextRequestPayload | None = None
        meta: PatcherMeta = {}

        while True:
            raw, attempt_meta = await _request_context_via_json_async(
                url=url,
                api_key=api_key,
                model=self.model,
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=(
                    max(1, min(int(self.max_tokens), _REQUEST_CONTEXT_STRICT_JSON_MAX_TOKENS))
                    if strict_json
                    else self.max_tokens
                ),
                compat_profile=self.compat_profile,
                allow_reasoning_fallback=self.allow_reasoning_fallback,
                enable_thinking=self.enable_thinking,
                timeout_sec=self.timeout_sec,
                seed=self.seed,
                http2=self.http2,
                strict_json=strict_json,
                transport_max_retries=self.transport_max_retries,
                transport_retry_backoff_base_sec=self.transport_retry_backoff_base_sec,
                transport_retry_backoff_max_sec=self.transport_retry_backoff_max_sec,
            )
            totals = _accumulate_llm_totals(totals=totals, meta=attempt_meta, default_calls=0)
            meta = cast(PatcherMeta, dict(attempt_meta or {}))
            request_json, raw_response = _normalize_context_request_payload(raw)
            failure_kind = _request_context_failure_kind(
                meta=meta,
                request_json=request_json,
                raw_response=raw_response,
            )
            outcome = _request_context_outcome(
                meta=meta,
                request_json=request_json,
                raw_response=raw_response,
            )
            helper_attempts.append(
                {
                    "attempt": len(helper_attempts) + 1,
                    "strict_json": bool(strict_json),
                    "ok": request_json is not None,
                    "outcome": outcome,
                    "failure_kind": failure_kind,
                    "error": str(meta.get("error") or ""),
                    "status": meta.get("status"),
                    "llm_call_count": _llm_call_count_from_meta(meta, default=0),
                    "transport_retry_count": _nonnegative_int(
                        meta.get("transport_retry_count"), default=0
                    ),
                    "response_format_fallback_used": bool(
                        meta.get("request_context_response_format_fallback_used")
                    ),
                }
            )
            if request_json is not None:
                break
            if _is_openai_compat_reasoning_only(meta):
                break
            if (
                failure_kind in {"transport", "backend_unavailable"}
                and transport_reissues_used < _REQUEST_CONTEXT_MAX_TRANSPORT_REISSUES
            ):
                transport_reissues_used += 1
                strict_json = False
                continue
            if (
                failure_kind == "invalid_payload"
                and strict_json_retries_used < _REQUEST_CONTEXT_MAX_STRICT_JSON_RETRIES
                and not strict_json
            ):
                strict_json_retries_used += 1
                strict_json = True
                continue
            break
        duration_sec = time.perf_counter() - start

        result_meta: PatcherMeta = copy_patcher_meta(meta)
        result_meta.setdefault("base_url", base_url)
        result_meta.setdefault("model", self.model)
        result_meta.setdefault("temperature", self.temperature)
        result_meta.setdefault("max_tokens", self.max_tokens)
        result_meta.setdefault(
            "compat_profile_requested", normalize_compat_profile(self.compat_profile)
        )
        result_meta.setdefault("response_mode", self.response_mode)
        result_meta.setdefault("api_compat_seed_requested", self.seed)
        result_meta.setdefault(
            "api_compat_seed_acknowledged", False if self.seed is not None else None
        )
        result_meta.setdefault("api_compat_seed_response", None)
        result_meta["request_context_duration_sec"] = duration_sec
        result_meta["request_context_attempts"] = helper_attempts
        result_meta["request_context_retry_count"] = max(0, len(helper_attempts) - 1)
        result_meta["request_context_transport_retry_count"] = transport_reissues_used
        result_meta["request_context_strict_retry_count"] = strict_json_retries_used
        result_meta["request_context_strict_json_first"] = bool(strict_json_first)
        failure_kind = _request_context_failure_kind(
            meta=result_meta,
            request_json=request_json,
            raw_response=raw_response,
        )
        outcome = _request_context_outcome(
            meta=result_meta,
            request_json=request_json,
            raw_response=raw_response,
        )
        result_meta["request_context_outcome"] = outcome
        result_meta["request_context_failure_kind"] = failure_kind
        result_meta["request_context_backend_unavailable"] = failure_kind == "backend_unavailable"
        result_meta["request_context_effective_success"] = request_json is not None
        result_meta = _meta_with_llm_totals(meta=result_meta, totals=totals)
        if request_json is None and raw_response is not None and "error" not in result_meta:
            result_meta["error"] = "invalid_request_context_payload"
        return ContextRequestResult(
            request_json=request_json,
            raw_response=raw_response,
            meta=result_meta,
        )

    @_with_openai_transport_progress(
        external_operation="compact_context",
        phase_label="openai: compact context",
    )
    def compact_context(
        self,
        *,
        prompt: str,
        context_text: str,
        context_dir: Path,
        cwd: Path,
        target_chars: int,
    ) -> ContextCompactionResult | None:
        """Ask the model to compress a context block to a target character budget.

        Sends *context_text* together with *prompt* to the ``chat/completions``
        endpoint.  The system instruction constrains the response to a JSON
        payload with a ``compacted_context`` field whose value must not exceed
        *target_chars* characters.  Critical diagnostic information (tracebacks,
        file paths, assertion messages, relevant state) is to be preserved;
        low-signal content is dropped.

        After the model responds the compacted text is hard-truncated to
        *target_chars* as a safety measure regardless of what the model reports.

        Falls back gracefully when the server does not support
        ``response_format: json_object`` by re-issuing the request without that
        constraint.

        Args:
            prompt: The repair prompt that provides task context for the
                compaction (used to anchor what is considered "critical").
            context_text: The full context string to be compressed.
            context_dir: Directory containing pre-cached context artefacts.
                Not used by this implementation; reserved for the protocol.
            cwd: Working directory of the case under repair.  Not used by this
                implementation; reserved for the protocol.
            target_chars: Maximum character length of the returned compacted
                text.  Must be a positive integer.

        Returns:
            A :class:`~evals.patchers.base.ContextCompactionResult` containing:

            - ``compacted_text``: the compressed and hard-truncated text on
              success, or ``None`` on failure.
            - ``raw_response``: the raw model output string.
            - ``meta``: telemetry dict with timing, input/output character
              counts, truncation flag, and token usage.
        """
        del cwd
        base_url = _normalize_base_url(self.base_url)
        url = f"{base_url}/chat/completions"
        api_key = self.api_key
        if api_key is None:
            api_key = _default_api_key()

        target_chars_safe = max(1, int(target_chars))
        start = time.perf_counter()
        compacted_text, raw_response, meta = _compact_context_via_json(
            url=url,
            api_key=api_key,
            model=self.model,
            prompt=prompt,
            context_text=context_text,
            target_chars=target_chars_safe,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            compat_profile=self.compat_profile,
            allow_reasoning_fallback=self.allow_reasoning_fallback,
            enable_thinking=self.enable_thinking,
            timeout_sec=self.timeout_sec,
            seed=self.seed,
            http2=self.http2,
            transport_max_retries=self.transport_max_retries,
            transport_retry_backoff_base_sec=self.transport_retry_backoff_base_sec,
            transport_retry_backoff_max_sec=self.transport_retry_backoff_max_sec,
        )
        duration_sec = time.perf_counter() - start

        result_meta: PatcherMeta = copy_patcher_meta(meta)
        result_meta.setdefault("base_url", base_url)
        result_meta.setdefault("model", self.model)
        result_meta.setdefault("temperature", self.temperature)
        result_meta.setdefault("max_tokens", self.max_tokens)
        result_meta.setdefault(
            "compat_profile_requested", normalize_compat_profile(self.compat_profile)
        )
        result_meta.setdefault("response_mode", self.response_mode)
        result_meta.setdefault("api_compat_seed_requested", self.seed)
        result_meta.setdefault(
            "api_compat_seed_acknowledged", False if self.seed is not None else None
        )
        result_meta.setdefault("api_compat_seed_response", None)
        result_meta["compact_context_duration_sec"] = duration_sec
        result_meta["compact_context_target_chars"] = target_chars_safe
        result_meta["compact_context_input_chars"] = len(context_text or "")

        if compacted_text is None:
            if raw_response is not None and "error" not in result_meta:
                result_meta["error"] = "invalid_context_compaction_payload"
            return ContextCompactionResult(
                compacted_text=None,
                raw_response=raw_response,
                meta=result_meta,
            )

        trimmed = compacted_text[:target_chars_safe]
        result_meta["compact_context_output_chars"] = len(compacted_text)
        result_meta["compact_context_output_truncated"] = len(trimmed) < len(compacted_text)
        return ContextCompactionResult(
            compacted_text=trimmed,
            raw_response=raw_response,
            meta=result_meta,
        )

    @_with_openai_transport_progress_async(
        external_operation="compact_context",
        phase_label="openai: compact context",
    )
    async def acompact_context(
        self,
        *,
        prompt: str,
        context_text: str,
        context_dir: Path,
        cwd: Path,
        target_chars: int,
    ) -> ContextCompactionResult | None:
        """Async variant of :meth:`compact_context`.

        Ask the model to compress a context block to a target character budget
        without blocking the running event loop.  Behaviour and return value
        are identical to :meth:`compact_context`; see that method for full
        parameter and return-value documentation.

        Args:
            prompt: The repair prompt providing task context for compaction.
            context_text: The full context string to be compressed.
            context_dir: Directory containing pre-cached context artefacts.
                Not used by this implementation; reserved for the protocol.
            cwd: Working directory of the case under repair.  Not used by this
                implementation; reserved for the protocol.
            target_chars: Maximum character length of the returned compacted
                text.

        Returns:
            A :class:`~evals.patchers.base.ContextCompactionResult` — see
            :meth:`compact_context` for field descriptions.
        """
        del cwd
        base_url = _normalize_base_url(self.base_url)
        url = f"{base_url}/chat/completions"
        api_key = self.api_key
        if api_key is None:
            api_key = _default_api_key()

        target_chars_safe = max(1, int(target_chars))
        start = time.perf_counter()
        compacted_text, raw_response, meta = await _compact_context_via_json_async(
            url=url,
            api_key=api_key,
            model=self.model,
            prompt=prompt,
            context_text=context_text,
            target_chars=target_chars_safe,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            compat_profile=self.compat_profile,
            allow_reasoning_fallback=self.allow_reasoning_fallback,
            enable_thinking=self.enable_thinking,
            timeout_sec=self.timeout_sec,
            seed=self.seed,
            http2=self.http2,
            transport_max_retries=self.transport_max_retries,
            transport_retry_backoff_base_sec=self.transport_retry_backoff_base_sec,
            transport_retry_backoff_max_sec=self.transport_retry_backoff_max_sec,
        )
        duration_sec = time.perf_counter() - start

        result_meta: PatcherMeta = copy_patcher_meta(meta)
        result_meta.setdefault("base_url", base_url)
        result_meta.setdefault("model", self.model)
        result_meta.setdefault("temperature", self.temperature)
        result_meta.setdefault("max_tokens", self.max_tokens)
        result_meta.setdefault(
            "compat_profile_requested", normalize_compat_profile(self.compat_profile)
        )
        result_meta.setdefault("response_mode", self.response_mode)
        result_meta.setdefault("api_compat_seed_requested", self.seed)
        result_meta.setdefault(
            "api_compat_seed_acknowledged", False if self.seed is not None else None
        )
        result_meta.setdefault("api_compat_seed_response", None)
        result_meta["compact_context_duration_sec"] = duration_sec
        result_meta["compact_context_target_chars"] = target_chars_safe
        result_meta["compact_context_input_chars"] = len(context_text or "")

        if compacted_text is None:
            if raw_response is not None and "error" not in result_meta:
                result_meta["error"] = "invalid_context_compaction_payload"
            return ContextCompactionResult(
                compacted_text=None,
                raw_response=raw_response,
                meta=result_meta,
            )

        trimmed = compacted_text[:target_chars_safe]
        result_meta["compact_context_output_chars"] = len(compacted_text)
        result_meta["compact_context_output_truncated"] = len(trimmed) < len(compacted_text)
        return ContextCompactionResult(
            compacted_text=trimmed,
            raw_response=raw_response,
            meta=result_meta,
        )

    @_with_openai_transport_progress(
        external_operation="propose_patch",
        phase_label="openai: propose patch",
    )
    def propose_patch(
        self,
        *,
        prompt: str,
        context_dir: Path,
        cwd: Path,
        system_prompt_override: str | None = None,
    ) -> PatchResult:
        """Generate a patch for the failing test case at *cwd*.

        Dispatches to one of three sub-strategies based on :attr:`response_mode`:

        * ``"edits"`` — calls :func:`_propose_patch_via_edits`, which asks the
          model for a JSON ``{"edits": [...]}`` payload and attempts to obtain a
          valid response across three ``response_format`` escalation stages:
          ``json_schema`` → ``json_object`` → plain text.  Once a valid payload
          is received the find-replace edits are applied to the source files
          under *cwd* and a unified diff is generated locally.  Falls back to
          the ``"diff"`` strategy automatically when no edits can be applied
          (e.g. ``find`` string not found).
        * ``"diff"`` — calls :func:`_propose_patch_via_diff`, which asks the
          model to emit a raw unified diff and makes up to two attempts with
          progressively stronger formatting instructions.  Falls back to the
          ``"edits"`` strategy when no parseable diff is found in the output.
        * ``"full_file"`` — calls :func:`_propose_patch_via_full_file`, which
          asks the model for the complete repaired contents of a single target
          source file and derives a unified diff locally. Unsupported case
          shapes and unusable responses fall back to the ``"diff"`` strategy.

        When :attr:`force_patch_attempt` is ``True`` and both strategies fail
        to produce a usable diff, a deterministic minimal patch (a marker
        comment appended to the first non-test ``.py`` file listed in the
        prompt's file index) is constructed as a last resort — unless the
        failure is attributed to a transport error, context-budget exhaustion,
        or context-length overflow.

        Args:
            prompt: The assembled repair prompt including file contents, pytest
                output, and any gathered context.
            context_dir: Directory containing pre-cached context artefacts.
                Not used by this implementation; reserved for the protocol.
            cwd: Working directory of the case under repair.  Used by the
                ``"edits"`` strategy to resolve relative file paths and to
                enforce that no edit escapes the case directory.

        Returns:
            A :class:`~evals.patchers.base.PatchResult` containing:

            - ``diff``: a unified diff string on success, or ``None`` when no
              patch could be generated.
            - ``meta``: telemetry dict including the response mode, token
              usage, transport retry details, structured-output attempt log,
              fallback chain metadata, and forced-attempt flags if applicable.
        """
        base_url = _normalize_base_url(self.base_url)
        url = f"{base_url}/chat/completions"

        api_key = self.api_key
        if api_key is None:
            api_key = _default_api_key()

        deadline_monotonic: float | None = None
        if self.max_propose_sec > 0:
            deadline_monotonic = time.monotonic() + self.max_propose_sec

        if self.response_mode == "edits":
            result = _propose_patch_via_edits(
                url=url,
                api_key=api_key,
                model=self.model,
                prompt=prompt,
                cwd=cwd,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                compat_profile=self.compat_profile,
                allow_reasoning_fallback=self.allow_reasoning_fallback,
                enable_thinking=self.enable_thinking,
                timeout_sec=self.timeout_sec,
                seed=self.seed,
                http2=self.http2,
                deadline_monotonic=deadline_monotonic,
                system_prompt_override=system_prompt_override,
                transport_max_retries=self.transport_max_retries,
                transport_retry_backoff_base_sec=self.transport_retry_backoff_base_sec,
                transport_retry_backoff_max_sec=self.transport_retry_backoff_max_sec,
            )
        elif self.response_mode == "full_file":
            result = _propose_patch_via_full_file(
                url=url,
                api_key=api_key,
                model=self.model,
                prompt=prompt,
                cwd=cwd,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                compat_profile=self.compat_profile,
                allow_reasoning_fallback=self.allow_reasoning_fallback,
                enable_thinking=self.enable_thinking,
                timeout_sec=self.timeout_sec,
                seed=self.seed,
                http2=self.http2,
                deadline_monotonic=deadline_monotonic,
                system_prompt_override=system_prompt_override,
                transport_max_retries=self.transport_max_retries,
                transport_retry_backoff_base_sec=self.transport_retry_backoff_base_sec,
                transport_retry_backoff_max_sec=self.transport_retry_backoff_max_sec,
            )
        else:
            diff_result = _propose_patch_via_diff(
                url=url,
                api_key=api_key,
                model=self.model,
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                compat_profile=self.compat_profile,
                allow_reasoning_fallback=self.allow_reasoning_fallback,
                enable_thinking=self.enable_thinking,
                timeout_sec=self.timeout_sec,
                seed=self.seed,
                http2=self.http2,
                deadline_monotonic=deadline_monotonic,
                system_prompt_override=system_prompt_override,
                transport_max_retries=self.transport_max_retries,
                transport_retry_backoff_base_sec=self.transport_retry_backoff_base_sec,
                transport_retry_backoff_max_sec=self.transport_retry_backoff_max_sec,
            )
            if (
                diff_result.diff is None
                and str((diff_result.meta or {}).get("error", "")) == "no_diff_found"
            ):
                edits_fallback = _propose_patch_via_edits(
                    url=url,
                    api_key=api_key,
                    model=self.model,
                    prompt=prompt,
                    cwd=cwd,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    compat_profile=self.compat_profile,
                    allow_reasoning_fallback=self.allow_reasoning_fallback,
                    enable_thinking=self.enable_thinking,
                    timeout_sec=self.timeout_sec,
                    seed=self.seed,
                    http2=self.http2,
                    deadline_monotonic=deadline_monotonic,
                    allow_diff_fallback=False,
                    system_prompt_override=system_prompt_override,
                    transport_max_retries=self.transport_max_retries,
                    transport_retry_backoff_base_sec=self.transport_retry_backoff_base_sec,
                    transport_retry_backoff_max_sec=self.transport_retry_backoff_max_sec,
                )
                totals = (0, 0, 0, 0, 0, False, True)
                diff_meta = copy_patcher_meta(diff_result.meta)
                edits_meta = copy_patcher_meta(edits_fallback.meta)
                totals = _accumulate_llm_totals(totals=totals, meta=diff_meta, default_calls=0)
                totals = _accumulate_llm_totals(totals=totals, meta=edits_meta, default_calls=0)
                merged_meta = copy_patcher_meta(edits_meta)
                merged_meta["fallback_from"] = "diff"
                merged_meta["fallback_reason"] = "no_diff_found"
                merged_meta["fallback_meta"] = diff_meta
                if merged_meta.get("error") is None:
                    merged_meta.pop("error", None)
                result = PatchResult(
                    diff=edits_fallback.diff,
                    meta=_meta_with_llm_totals(meta=merged_meta, totals=totals),
                )
            else:
                result = diff_result

        result_meta = copy_patcher_meta(result.meta)
        result_meta.setdefault("base_url", base_url)
        result_meta.setdefault("model", self.model)
        result_meta.setdefault("temperature", self.temperature)
        result_meta.setdefault("max_tokens", self.max_tokens)
        result_meta.setdefault(
            "compat_profile_requested", normalize_compat_profile(self.compat_profile)
        )
        result_meta.setdefault("timeout_sec", self.timeout_sec)
        result_meta.setdefault("transport_max_retries", self.transport_max_retries)
        result_meta.setdefault(
            "transport_retry_backoff_base_sec", self.transport_retry_backoff_base_sec
        )
        result_meta.setdefault(
            "transport_retry_backoff_max_sec", self.transport_retry_backoff_max_sec
        )
        result_meta.setdefault("max_propose_sec", self.max_propose_sec)
        result_meta.setdefault("response_mode", self.response_mode)
        result_meta.setdefault("api_compat_seed_requested", self.seed)
        result_meta.setdefault(
            "api_compat_seed_acknowledged", False if self.seed is not None else None
        )
        result_meta.setdefault("api_compat_seed_response", None)
        result = PatchResult(diff=result.diff, meta=result_meta)

        if result.diff is not None or not self.force_patch_attempt:
            return result

        skip_reason = _forced_patch_skip_reason(cast(PatcherMeta, result_meta))
        if skip_reason:
            meta = copy_patcher_meta(result.meta)
            meta["forced_patch_skipped"] = True
            meta["forced_patch_skip_reason"] = _forced_patch_skip_category(skip_reason)
            meta["forced_patch_skip_detail"] = skip_reason
            return PatchResult(diff=None, meta=meta)

        forced = _build_forced_attempt_diff(prompt=prompt, cwd=cwd)
        if forced is None:
            return result

        meta = copy_patcher_meta(result.meta)
        meta["forced_patch_attempt"] = True
        meta["forced_patch_target"] = forced[0]
        meta["forced_patch_reason"] = "no_model_diff"
        return PatchResult(diff=forced[1], meta=meta)

    @_with_openai_transport_progress_async(
        external_operation="propose_patch",
        phase_label="openai: propose patch",
    )
    async def apropose_patch(self, *, prompt: str, context_dir: Path, cwd: Path) -> PatchResult:
        """Async variant of :meth:`propose_patch`.

        Generate a patch for the failing test case at *cwd* without blocking
        the running event loop.  Behaviour, fallback logic, and return value
        are identical to :meth:`propose_patch`; see that method for full
        parameter and return-value documentation.

        Args:
            prompt: The assembled repair prompt including file contents, pytest
                output, and any gathered context.
            context_dir: Directory containing pre-cached context artefacts.
                Not used by this implementation; reserved for the protocol.
            cwd: Working directory of the case under repair.  Used to resolve
                and validate relative file paths in the ``"edits"`` strategy.

        Returns:
            A :class:`~evals.patchers.base.PatchResult` — see
            :meth:`propose_patch` for field descriptions.
        """
        base_url = _normalize_base_url(self.base_url)
        url = f"{base_url}/chat/completions"

        api_key = self.api_key
        if api_key is None:
            api_key = _default_api_key()

        deadline_monotonic: float | None = None
        if self.max_propose_sec > 0:
            deadline_monotonic = time.monotonic() + self.max_propose_sec

        if self.response_mode == "edits":
            result = await _propose_patch_via_edits_async(
                url=url,
                api_key=api_key,
                model=self.model,
                prompt=prompt,
                cwd=cwd,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                compat_profile=self.compat_profile,
                allow_reasoning_fallback=self.allow_reasoning_fallback,
                enable_thinking=self.enable_thinking,
                timeout_sec=self.timeout_sec,
                seed=self.seed,
                http2=self.http2,
                deadline_monotonic=deadline_monotonic,
                transport_max_retries=self.transport_max_retries,
                transport_retry_backoff_base_sec=self.transport_retry_backoff_base_sec,
                transport_retry_backoff_max_sec=self.transport_retry_backoff_max_sec,
            )
        elif self.response_mode == "full_file":
            result = await _propose_patch_via_full_file_async(
                url=url,
                api_key=api_key,
                model=self.model,
                prompt=prompt,
                cwd=cwd,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                compat_profile=self.compat_profile,
                allow_reasoning_fallback=self.allow_reasoning_fallback,
                enable_thinking=self.enable_thinking,
                timeout_sec=self.timeout_sec,
                seed=self.seed,
                http2=self.http2,
                deadline_monotonic=deadline_monotonic,
                transport_max_retries=self.transport_max_retries,
                transport_retry_backoff_base_sec=self.transport_retry_backoff_base_sec,
                transport_retry_backoff_max_sec=self.transport_retry_backoff_max_sec,
            )
        else:
            diff_result = await _propose_patch_via_diff_async(
                url=url,
                api_key=api_key,
                model=self.model,
                prompt=prompt,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                compat_profile=self.compat_profile,
                allow_reasoning_fallback=self.allow_reasoning_fallback,
                enable_thinking=self.enable_thinking,
                timeout_sec=self.timeout_sec,
                seed=self.seed,
                http2=self.http2,
                deadline_monotonic=deadline_monotonic,
                transport_max_retries=self.transport_max_retries,
                transport_retry_backoff_base_sec=self.transport_retry_backoff_base_sec,
                transport_retry_backoff_max_sec=self.transport_retry_backoff_max_sec,
            )
            if (
                diff_result.diff is None
                and str((diff_result.meta or {}).get("error", "")) == "no_diff_found"
            ):
                edits_fallback = await _propose_patch_via_edits_async(
                    url=url,
                    api_key=api_key,
                    model=self.model,
                    prompt=prompt,
                    cwd=cwd,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    compat_profile=self.compat_profile,
                    allow_reasoning_fallback=self.allow_reasoning_fallback,
                    enable_thinking=self.enable_thinking,
                    timeout_sec=self.timeout_sec,
                    seed=self.seed,
                    http2=self.http2,
                    deadline_monotonic=deadline_monotonic,
                    allow_diff_fallback=False,
                    transport_max_retries=self.transport_max_retries,
                    transport_retry_backoff_base_sec=self.transport_retry_backoff_base_sec,
                    transport_retry_backoff_max_sec=self.transport_retry_backoff_max_sec,
                )
                totals = (0, 0, 0, 0, 0, False, True)
                diff_meta = copy_patcher_meta(diff_result.meta)
                edits_meta = copy_patcher_meta(edits_fallback.meta)
                totals = _accumulate_llm_totals(totals=totals, meta=diff_meta, default_calls=0)
                totals = _accumulate_llm_totals(totals=totals, meta=edits_meta, default_calls=0)
                merged_meta = copy_patcher_meta(edits_meta)
                merged_meta["fallback_from"] = "diff"
                merged_meta["fallback_reason"] = "no_diff_found"
                merged_meta["fallback_meta"] = diff_meta
                if merged_meta.get("error") is None:
                    merged_meta.pop("error", None)
                result = PatchResult(
                    diff=edits_fallback.diff,
                    meta=_meta_with_llm_totals(meta=merged_meta, totals=totals),
                )
            else:
                result = diff_result

        result_meta = copy_patcher_meta(result.meta)
        result_meta.setdefault("base_url", base_url)
        result_meta.setdefault("model", self.model)
        result_meta.setdefault("temperature", self.temperature)
        result_meta.setdefault("max_tokens", self.max_tokens)
        result_meta.setdefault(
            "compat_profile_requested", normalize_compat_profile(self.compat_profile)
        )
        result_meta.setdefault("timeout_sec", self.timeout_sec)
        result_meta.setdefault("transport_max_retries", self.transport_max_retries)
        result_meta.setdefault(
            "transport_retry_backoff_base_sec", self.transport_retry_backoff_base_sec
        )
        result_meta.setdefault(
            "transport_retry_backoff_max_sec", self.transport_retry_backoff_max_sec
        )
        result_meta.setdefault("max_propose_sec", self.max_propose_sec)
        result_meta.setdefault("response_mode", self.response_mode)
        result_meta.setdefault("api_compat_seed_requested", self.seed)
        result_meta.setdefault(
            "api_compat_seed_acknowledged", False if self.seed is not None else None
        )
        result_meta.setdefault("api_compat_seed_response", None)
        result = PatchResult(diff=result.diff, meta=result_meta)

        if result.diff is not None or not self.force_patch_attempt:
            return result

        skip_reason = _forced_patch_skip_reason(cast(PatcherMeta, result_meta))
        if skip_reason:
            meta = copy_patcher_meta(result.meta)
            meta["forced_patch_skipped"] = True
            meta["forced_patch_skip_reason"] = _forced_patch_skip_category(skip_reason)
            meta["forced_patch_skip_detail"] = skip_reason
            return PatchResult(diff=None, meta=meta)

        forced = _build_forced_attempt_diff(prompt=prompt, cwd=cwd)
        if forced is None:
            return result

        meta = copy_patcher_meta(result.meta)
        meta["forced_patch_attempt"] = True
        meta["forced_patch_target"] = forced[0]
        meta["forced_patch_reason"] = "no_model_diff"
        return PatchResult(diff=forced[1], meta=meta)

    def probe_context_window_tokens(self) -> tuple[int | None, PatcherMeta]:
        """Probe the server for the model's context-window size in tokens.

        Queries up to three provider-specific model listing endpoints in order
        of preference, stopping at the first that returns a value for
        :attr:`model`:

        1. **LM Studio API v1** — ``{origin}/api/v1/models``
        2. **LM Studio API v0** — ``{origin}/api/v0/models``
        3. **OpenAI-compatible** — ``{base_url}/models``

        Each response is parsed with a provider-specific extractor that looks
        for ``context_length``, ``max_context_length``, or
        ``max_input_tokens`` fields on the matching model entry.  For LM Studio
        v1, per-instance ``context_length`` overrides the model-level field
        when the model is currently loaded.

        Returns:
            A 2-tuple ``(tokens, meta)`` where:

            - ``tokens`` is the context-window size in tokens as a positive
              ``int``, or ``None`` if no probe succeeded.
            - ``meta`` is a telemetry dict with keys such as
              ``context_window_probe_provider``, ``context_window_source``,
              ``context_window_probe_url``, ``context_window_probe_attempts``
              (per-probe error details), and ``context_window_probe_duration_sec``.
        """
        base_url = _normalize_base_url(self.base_url)
        api_key = self.api_key
        if api_key is None:
            api_key = _default_api_key()
        started = time.perf_counter()
        tokens, meta = _probe_context_window_tokens(
            base_url=base_url,
            model=self.model,
            api_key=api_key,
            timeout_sec=self.timeout_sec,
            http2=self.http2,
        )
        result_meta: PatcherMeta = copy_patcher_meta(meta)
        result_meta.setdefault("base_url", base_url)
        result_meta.setdefault("model", self.model)
        result_meta["context_window_probe_duration_sec"] = time.perf_counter() - started
        return tokens, result_meta

    async def aprobe_context_window_tokens(self) -> tuple[int | None, PatcherMeta]:
        """Async variant of :meth:`probe_context_window_tokens`.

        Probe the server for the model's context-window size in tokens without
        blocking the running event loop.  Behaviour and return value are
        identical to :meth:`probe_context_window_tokens`; see that method for
        full documentation of the probe order and return fields.

        Returns:
            A 2-tuple ``(tokens, meta)`` — see
            :meth:`probe_context_window_tokens` for field descriptions.
        """
        base_url = _normalize_base_url(self.base_url)
        api_key = self.api_key
        if api_key is None:
            api_key = _default_api_key()
        started = time.perf_counter()
        tokens, meta = await _probe_context_window_tokens_async(
            base_url=base_url,
            model=self.model,
            api_key=api_key,
            timeout_sec=self.timeout_sec,
            http2=self.http2,
        )
        result_meta: PatcherMeta = copy_patcher_meta(meta)
        result_meta.setdefault("base_url", base_url)
        result_meta.setdefault("model", self.model)
        result_meta["context_window_probe_duration_sec"] = time.perf_counter() - started
        return tokens, result_meta

    def probe_server_ready(self) -> tuple[bool, PatcherMeta]:
        """Probe whether the backend is ready to accept fresh OpenAI calls.

        Probe order:

        1. ``GET <origin>/health``
        2. ``GET <base_url>/models``

        Any ``2xx`` response counts as healthy regardless of body contents.
        """
        base_url = _normalize_base_url(self.base_url)
        api_key = self.api_key
        if api_key is None:
            api_key = _default_api_key()
        started = time.perf_counter()
        ready, meta = _probe_server_ready(
            base_url=base_url,
            api_key=api_key,
            timeout_sec=self.timeout_sec,
            http2=self.http2,
        )
        result_meta: PatcherMeta = copy_patcher_meta(meta)
        result_meta.setdefault("base_url", base_url)
        result_meta.setdefault("model", self.model)
        result_meta["server_ready_probe_duration_sec"] = time.perf_counter() - started
        return ready, result_meta

    async def aprobe_server_ready(self) -> tuple[bool, PatcherMeta]:
        """Async variant of :meth:`probe_server_ready`."""
        base_url = _normalize_base_url(self.base_url)
        api_key = self.api_key
        if api_key is None:
            api_key = _default_api_key()
        started = time.perf_counter()
        ready, meta = await _probe_server_ready_async(
            base_url=base_url,
            api_key=api_key,
            timeout_sec=self.timeout_sec,
            http2=self.http2,
        )
        result_meta: PatcherMeta = copy_patcher_meta(meta)
        result_meta.setdefault("base_url", base_url)
        result_meta.setdefault("model", self.model)
        result_meta["server_ready_probe_duration_sec"] = time.perf_counter() - started
        return ready, result_meta


def _request_context_via_json(
    *,
    url: str,
    api_key: str,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    compat_profile: str = "auto",
    allow_reasoning_fallback: bool = False,
    enable_thinking: bool = False,
    timeout_sec: float,
    seed: int | None,
    http2: bool,
    strict_json: bool = False,
    transport_max_retries: int = _DEFAULT_CHAT_COMPLETION_TRANSPORT_MAX_RETRIES,
    transport_retry_backoff_base_sec: float = _DEFAULT_CHAT_COMPLETION_RETRY_BACKOFF_BASE_SEC,
    transport_retry_backoff_max_sec: float = _DEFAULT_CHAT_COMPLETION_RETRY_BACKOFF_MAX_SEC,
) -> tuple[dict[str, Any] | None, PatcherMeta]:
    messages = [
        {
            "role": "system",
            "content": _request_context_system_prompt(strict_json=strict_json),
        },
        {"role": "user", "content": prompt},
    ]

    totals = (0, 0, 0, 0, 0, False, True)
    response_format_fallback_used = False
    raw, meta = _chat_completion(
        url=url,
        api_key=api_key,
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        compat_profile=compat_profile,
        allow_reasoning_fallback=allow_reasoning_fallback,
        enable_thinking=enable_thinking,
        timeout_sec=timeout_sec,
        seed=seed,
        http2=http2,
        extra_payload={"response_format": {"type": "json_object"}},
        transport_max_retries=transport_max_retries,
        transport_retry_backoff_base_sec=transport_retry_backoff_base_sec,
        transport_retry_backoff_max_sec=transport_retry_backoff_max_sec,
    )
    totals = _accumulate_llm_totals(totals=totals, meta=meta, default_calls=1)
    if raw is None:
        err = str(meta.get("error", ""))
        body = str(meta.get("body", ""))
        if err == "http_error" and "response_format" in body:
            response_format_fallback_used = True
            raw, meta = _chat_completion(
                url=url,
                api_key=api_key,
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                compat_profile=compat_profile,
                allow_reasoning_fallback=allow_reasoning_fallback,
                enable_thinking=enable_thinking,
                timeout_sec=timeout_sec,
                seed=seed,
                http2=http2,
                transport_max_retries=transport_max_retries,
                transport_retry_backoff_base_sec=transport_retry_backoff_base_sec,
                transport_retry_backoff_max_sec=transport_retry_backoff_max_sec,
            )
            totals = _accumulate_llm_totals(totals=totals, meta=meta, default_calls=1)
    meta = _meta_with_llm_totals(meta=cast(PatcherMeta, dict(meta)), totals=totals)
    meta["request_context_response_format_fallback_used"] = response_format_fallback_used
    meta["request_context_strict_json"] = bool(strict_json)
    return raw, meta


async def _request_context_via_json_async(
    *,
    url: str,
    api_key: str,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    compat_profile: str = "auto",
    allow_reasoning_fallback: bool = False,
    enable_thinking: bool = False,
    timeout_sec: float,
    seed: int | None,
    http2: bool,
    strict_json: bool = False,
    transport_max_retries: int = _DEFAULT_CHAT_COMPLETION_TRANSPORT_MAX_RETRIES,
    transport_retry_backoff_base_sec: float = _DEFAULT_CHAT_COMPLETION_RETRY_BACKOFF_BASE_SEC,
    transport_retry_backoff_max_sec: float = _DEFAULT_CHAT_COMPLETION_RETRY_BACKOFF_MAX_SEC,
) -> tuple[dict[str, Any] | None, PatcherMeta]:
    messages = [
        {
            "role": "system",
            "content": _request_context_system_prompt(strict_json=strict_json),
        },
        {"role": "user", "content": prompt},
    ]

    totals = (0, 0, 0, 0, 0, False, True)
    response_format_fallback_used = False
    raw, meta = await _chat_completion_async(
        url=url,
        api_key=api_key,
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        compat_profile=compat_profile,
        allow_reasoning_fallback=allow_reasoning_fallback,
        enable_thinking=enable_thinking,
        timeout_sec=timeout_sec,
        seed=seed,
        http2=http2,
        extra_payload={"response_format": {"type": "json_object"}},
        transport_max_retries=transport_max_retries,
        transport_retry_backoff_base_sec=transport_retry_backoff_base_sec,
        transport_retry_backoff_max_sec=transport_retry_backoff_max_sec,
    )
    totals = _accumulate_llm_totals(totals=totals, meta=meta, default_calls=1)
    if raw is None:
        err = str(meta.get("error", ""))
        body = str(meta.get("body", ""))
        if err == "http_error" and "response_format" in body:
            response_format_fallback_used = True
            raw, meta = await _chat_completion_async(
                url=url,
                api_key=api_key,
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                compat_profile=compat_profile,
                allow_reasoning_fallback=allow_reasoning_fallback,
                enable_thinking=enable_thinking,
                timeout_sec=timeout_sec,
                seed=seed,
                http2=http2,
                transport_max_retries=transport_max_retries,
                transport_retry_backoff_base_sec=transport_retry_backoff_base_sec,
                transport_retry_backoff_max_sec=transport_retry_backoff_max_sec,
            )
            totals = _accumulate_llm_totals(totals=totals, meta=meta, default_calls=1)
    meta = _meta_with_llm_totals(meta=cast(PatcherMeta, dict(meta)), totals=totals)
    meta["request_context_response_format_fallback_used"] = response_format_fallback_used
    meta["request_context_strict_json"] = bool(strict_json)
    return raw, meta


def _compact_context_via_json(
    *,
    url: str,
    api_key: str,
    model: str,
    prompt: str,
    context_text: str,
    target_chars: int,
    temperature: float,
    max_tokens: int,
    compat_profile: str = "auto",
    allow_reasoning_fallback: bool = False,
    enable_thinking: bool = False,
    timeout_sec: float,
    seed: int | None,
    http2: bool,
    transport_max_retries: int = _DEFAULT_CHAT_COMPLETION_TRANSPORT_MAX_RETRIES,
    transport_retry_backoff_base_sec: float = _DEFAULT_CHAT_COMPLETION_RETRY_BACKOFF_BASE_SEC,
    transport_retry_backoff_max_sec: float = _DEFAULT_CHAT_COMPLETION_RETRY_BACKOFF_MAX_SEC,
) -> tuple[str | None, str | None, PatcherMeta]:
    target_chars_safe = max(1, int(target_chars))
    messages = [
        {
            "role": "system",
            "content": (
                "Return only JSON for context compaction. "
                "Use this schema exactly: "
                '{"compacted_context":"...",'
                '"reduction_rationale":"..."} '
                f"The compacted_context must be <= {target_chars_safe} characters. "
                "Preserve critical traceback facts, file paths, assertions, and state clues. "
                "No prose and no markdown."
            ),
        },
        {
            "role": "user",
            "content": (
                "Task:\n"
                + prompt.rstrip()
                + "\n\n"
                + f"Target characters: {target_chars_safe}\n"
                + "Context to compact:\n"
                + context_text.rstrip()
                + "\n"
            ),
        },
    ]

    totals = (0, 0, 0, 0, 0, False, True)
    raw, meta = _chat_completion(
        url=url,
        api_key=api_key,
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        compat_profile=compat_profile,
        allow_reasoning_fallback=allow_reasoning_fallback,
        enable_thinking=enable_thinking,
        timeout_sec=timeout_sec,
        seed=seed,
        http2=http2,
        extra_payload={"response_format": {"type": "json_object"}},
        transport_max_retries=transport_max_retries,
        transport_retry_backoff_base_sec=transport_retry_backoff_base_sec,
        transport_retry_backoff_max_sec=transport_retry_backoff_max_sec,
    )
    totals = _accumulate_llm_totals(totals=totals, meta=meta, default_calls=1)
    if raw is None:
        err = str(meta.get("error", ""))
        body = str(meta.get("body", ""))
        if err == "http_error" and "response_format" in body:
            raw, meta = _chat_completion(
                url=url,
                api_key=api_key,
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                compat_profile=compat_profile,
                allow_reasoning_fallback=allow_reasoning_fallback,
                enable_thinking=enable_thinking,
                timeout_sec=timeout_sec,
                seed=seed,
                http2=http2,
                transport_max_retries=transport_max_retries,
                transport_retry_backoff_base_sec=transport_retry_backoff_base_sec,
                transport_retry_backoff_max_sec=transport_retry_backoff_max_sec,
            )
            totals = _accumulate_llm_totals(totals=totals, meta=meta, default_calls=1)

    meta = _meta_with_llm_totals(meta=cast(PatcherMeta, dict(meta)), totals=totals)
    compacted_context, raw_response = _normalize_context_compaction_payload(raw)
    return compacted_context, raw_response, meta


async def _compact_context_via_json_async(
    *,
    url: str,
    api_key: str,
    model: str,
    prompt: str,
    context_text: str,
    target_chars: int,
    temperature: float,
    max_tokens: int,
    compat_profile: str = "auto",
    allow_reasoning_fallback: bool = False,
    enable_thinking: bool = False,
    timeout_sec: float,
    seed: int | None,
    http2: bool,
    transport_max_retries: int = _DEFAULT_CHAT_COMPLETION_TRANSPORT_MAX_RETRIES,
    transport_retry_backoff_base_sec: float = _DEFAULT_CHAT_COMPLETION_RETRY_BACKOFF_BASE_SEC,
    transport_retry_backoff_max_sec: float = _DEFAULT_CHAT_COMPLETION_RETRY_BACKOFF_MAX_SEC,
) -> tuple[str | None, str | None, PatcherMeta]:
    target_chars_safe = max(1, int(target_chars))
    messages = [
        {
            "role": "system",
            "content": (
                "Return only JSON for context compaction. "
                "Use this schema exactly: "
                '{"compacted_context":"...",'
                '"reduction_rationale":"..."} '
                f"The compacted_context must be <= {target_chars_safe} characters. "
                "Preserve critical traceback facts, file paths, assertions, and state clues. "
                "No prose and no markdown."
            ),
        },
        {
            "role": "user",
            "content": (
                "Task:\n"
                + prompt.rstrip()
                + "\n\n"
                + f"Target characters: {target_chars_safe}\n"
                + "Context to compact:\n"
                + context_text.rstrip()
                + "\n"
            ),
        },
    ]

    totals = (0, 0, 0, 0, 0, False, True)
    raw, meta = await _chat_completion_async(
        url=url,
        api_key=api_key,
        model=model,
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        compat_profile=compat_profile,
        allow_reasoning_fallback=allow_reasoning_fallback,
        enable_thinking=enable_thinking,
        timeout_sec=timeout_sec,
        seed=seed,
        http2=http2,
        extra_payload={"response_format": {"type": "json_object"}},
        transport_max_retries=transport_max_retries,
        transport_retry_backoff_base_sec=transport_retry_backoff_base_sec,
        transport_retry_backoff_max_sec=transport_retry_backoff_max_sec,
    )
    totals = _accumulate_llm_totals(totals=totals, meta=meta, default_calls=1)
    if raw is None:
        err = str(meta.get("error", ""))
        body = str(meta.get("body", ""))
        if err == "http_error" and "response_format" in body:
            raw, meta = await _chat_completion_async(
                url=url,
                api_key=api_key,
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                compat_profile=compat_profile,
                allow_reasoning_fallback=allow_reasoning_fallback,
                enable_thinking=enable_thinking,
                timeout_sec=timeout_sec,
                seed=seed,
                http2=http2,
                transport_max_retries=transport_max_retries,
                transport_retry_backoff_base_sec=transport_retry_backoff_base_sec,
                transport_retry_backoff_max_sec=transport_retry_backoff_max_sec,
            )
            totals = _accumulate_llm_totals(totals=totals, meta=meta, default_calls=1)

    meta = _meta_with_llm_totals(meta=cast(PatcherMeta, dict(meta)), totals=totals)
    compacted_context, raw_response = _normalize_context_compaction_payload(raw)
    return compacted_context, raw_response, meta


def _normalize_context_compaction_payload(
    response: dict[str, Any] | None,
) -> tuple[str | None, str | None]:
    if response is None:
        return None, None

    content = _get_first_choice_content(response)
    if content is None:
        return None, _safe_trunc(response)
    if _is_truncated_response(response):
        return None, content
    payload = _extract_json_object(content)
    if payload is None:
        return None, content

    candidates = (
        payload.get("compacted_context"),
        payload.get("summary"),
        payload.get("context"),
        payload.get("text"),
    )
    for candidate in candidates:
        if isinstance(candidate, str) and candidate.strip():
            return candidate.strip(), content
    return None, content


def _normalize_context_request_payload(
    response: dict[str, Any] | None,
) -> tuple[ContextRequestPayload | None, str | None]:
    if response is None:
        return None, None

    content = _get_first_choice_content(response)
    if content is None:
        return None, _safe_trunc(response)
    if _is_truncated_response(response):
        return None, content
    raw_payload = _extract_json_object(content)
    if raw_payload is None:
        return None, content
    normalized = normalize_stage_a_request_payload(raw_payload)
    if normalized is None:
        return None, content
    return normalized, content


def _propose_patch_via_diff(
    *,
    url: str,
    api_key: str,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    compat_profile: str = "auto",
    allow_reasoning_fallback: bool = False,
    enable_thinking: bool = False,
    timeout_sec: float,
    seed: int | None,
    http2: bool,
    deadline_monotonic: float | None = None,
    system_prompt_override: str | None = None,
    transport_max_retries: int = _DEFAULT_CHAT_COMPLETION_TRANSPORT_MAX_RETRIES,
    transport_retry_backoff_base_sec: float = _DEFAULT_CHAT_COMPLETION_RETRY_BACKOFF_BASE_SEC,
    transport_retry_backoff_max_sec: float = _DEFAULT_CHAT_COMPLETION_RETRY_BACKOFF_MAX_SEC,
) -> PatchResult:
    base_messages = [
        {
            "role": "system",
            "content": system_prompt_override
            or (
                "You fix failing tests by returning a single unified diff. "
                "Return ONLY the diff; no explanations. Do not use Markdown fences."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    # Try twice: second attempt strengthens formatting instructions without assistant prefill.
    attempts: list[list[dict[str, str]]] = [
        base_messages,
        [
            {
                "role": "system",
                "content": system_prompt_override
                or (
                    "You fix failing tests by returning a single unified diff. "
                    "Return ONLY the diff; no explanations. Do not use Markdown fences. "
                    "Start your response with 'diff --git a/'."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    ]

    last_meta: PatcherMeta | None = None
    totals = (0, 0, 0, 0, 0, False, True)
    for idx, messages in enumerate(attempts, start=1):
        raw, meta = _chat_completion(
            url=url,
            api_key=api_key,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            compat_profile=compat_profile,
            allow_reasoning_fallback=allow_reasoning_fallback,
            enable_thinking=enable_thinking,
            timeout_sec=timeout_sec,
            seed=seed,
            http2=http2,
            deadline_monotonic=deadline_monotonic,
            transport_max_retries=transport_max_retries,
            transport_retry_backoff_base_sec=transport_retry_backoff_base_sec,
            transport_retry_backoff_max_sec=transport_retry_backoff_max_sec,
        )
        totals = _accumulate_llm_totals(totals=totals, meta=meta, default_calls=1)
        last_meta = _meta_with_llm_totals(meta={"attempt": idx, **meta}, totals=totals)
        if raw is None:
            if _is_openai_compat_reasoning_only(meta):
                break
            if _is_request_budget_exceeded(meta):
                break
            continue

        content = _get_first_choice_content(raw)
        if content is None:
            last_meta["error"] = "no_choice_content"
            last_meta["response"] = _safe_trunc(raw)
            continue
        if _is_truncated_response(raw):
            last_meta = _error_meta(
                last_meta, TRUNCATED_RESPONSE_ERROR, raw_response=content[:2000]
            )
            break

        diff = _extract_unified_diff(content)
        if diff is None:
            last_meta["error"] = "no_diff_found"
            last_meta["raw_response"] = content[:2000]
            continue

        fixed = _fix_hunk_counts(diff)
        return PatchResult(diff=fixed, meta=last_meta)

    if last_meta is None:
        last_meta = _meta_with_llm_totals(meta={"error": "unknown"}, totals=totals)
    return PatchResult(diff=None, meta=last_meta)


def _propose_patch_via_edits(
    *,
    url: str,
    api_key: str,
    model: str,
    prompt: str,
    cwd: Path,
    temperature: float,
    max_tokens: int,
    compat_profile: str = "auto",
    allow_reasoning_fallback: bool = False,
    enable_thinking: bool = False,
    timeout_sec: float,
    seed: int | None,
    http2: bool,
    deadline_monotonic: float | None = None,
    allow_diff_fallback: bool = True,
    system_prompt_override: str | None = None,
    transport_max_retries: int = _DEFAULT_CHAT_COMPLETION_TRANSPORT_MAX_RETRIES,
    transport_retry_backoff_base_sec: float = _DEFAULT_CHAT_COMPLETION_RETRY_BACKOFF_BASE_SEC,
    transport_retry_backoff_max_sec: float = _DEFAULT_CHAT_COMPLETION_RETRY_BACKOFF_MAX_SEC,
) -> PatchResult:
    messages = [
        {
            "role": "system",
            "content": system_prompt_override
            or (
                "Return only JSON with this schema: "
                '{"edits":[{"path":"relative/path.py","find":"exact old text","replace":"new text","count":1}]}. '
                "No prose. No markdown. "
                "Use only exact existing paths from the prompt's available file list. "
                "Never use placeholder paths like relative/path.py."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    totals = (0, 0, 0, 0, 0, False, True)

    def _with_totals(meta: PatcherMeta) -> PatcherMeta:
        return _meta_with_llm_totals(meta=meta, totals=totals)

    payload: dict[str, Any] | None = None
    meta: PatcherMeta = {}
    stages = [
        (
            "json_schema",
            {
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "patch_edits",
                        "schema": PATCH_EDITS_JSON_SCHEMA,
                    },
                }
            },
        ),
        ("json_object", {"response_format": {"type": "json_object"}}),
        ("plain", None),
    ]
    stage_attempts: list[StructuredOutputAttempt] = []
    fallback_reason: str | None = None
    fallback_raw_response: str | None = None
    fallback_payload: dict[str, Any] | None = None

    def _fallback_to_diff(
        *,
        reason: str,
        extra_error_meta: PatcherMeta | None = None,
    ) -> PatchResult:
        nonlocal totals
        if not allow_diff_fallback:
            error_meta = _error_meta(meta, reason)
            if extra_error_meta:
                error_meta.update(extra_error_meta)
            return PatchResult(diff=None, meta=_with_totals(error_meta))

        diff_fallback = _propose_patch_via_diff(
            url=url,
            api_key=api_key,
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            compat_profile=compat_profile,
            allow_reasoning_fallback=allow_reasoning_fallback,
            enable_thinking=enable_thinking,
            timeout_sec=timeout_sec,
            seed=seed,
            http2=http2,
            deadline_monotonic=deadline_monotonic,
            system_prompt_override=system_prompt_override,
            transport_max_retries=transport_max_retries,
            transport_retry_backoff_base_sec=transport_retry_backoff_base_sec,
            transport_retry_backoff_max_sec=transport_retry_backoff_max_sec,
        )
        if diff_fallback.diff is not None:
            fallback_meta = copy_patcher_meta(diff_fallback.meta)
            totals = _accumulate_llm_totals(totals=totals, meta=fallback_meta, default_calls=0)
            fallback_meta["fallback_from"] = "edits"
            fallback_meta["fallback_reason"] = reason
            fallback_meta["edits_response_mode"] = str(meta.get("structured_output_mode") or "")
            fallback_meta["edits_response_attempts"] = stage_attempts
            if extra_error_meta:
                fallback_meta["fallback_trigger"] = copy_patcher_meta(extra_error_meta)
            return PatchResult(diff=diff_fallback.diff, meta=_with_totals(fallback_meta))

        fallback_meta = copy_patcher_meta(diff_fallback.meta)
        totals = _accumulate_llm_totals(totals=totals, meta=fallback_meta, default_calls=0)
        error_meta = _error_meta(meta, reason, fallback_meta=fallback_meta)
        if extra_error_meta:
            error_meta.update(extra_error_meta)
        return PatchResult(diff=None, meta=_with_totals(error_meta))

    for stage_name, extra_payload in stages:
        raw, stage_meta = _chat_completion(
            url=url,
            api_key=api_key,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            compat_profile=compat_profile,
            allow_reasoning_fallback=allow_reasoning_fallback,
            enable_thinking=enable_thinking,
            timeout_sec=timeout_sec,
            seed=seed,
            http2=http2,
            extra_payload=extra_payload,
            deadline_monotonic=deadline_monotonic,
            transport_max_retries=transport_max_retries,
            transport_retry_backoff_base_sec=transport_retry_backoff_base_sec,
            transport_retry_backoff_max_sec=transport_retry_backoff_max_sec,
        )
        totals = _accumulate_llm_totals(totals=totals, meta=stage_meta, default_calls=1)
        meta = copy_patcher_meta(stage_meta)
        meta["structured_output_mode"] = stage_name
        attempt: StructuredOutputAttempt = {
            "mode": stage_name,
            "ok": raw is not None,
            "error": str(meta.get("error", "")),
            "status": meta.get("status"),
        }
        if raw is None:
            stage_attempts.append(attempt)
            # Don't spam retries when prompt already overflows available context.
            if _is_context_exceeded_http_error(meta) or _is_request_budget_exceeded(meta):
                attempt["terminal"] = True
                break
            if _is_openai_compat_reasoning_only(meta):
                attempt["terminal"] = True
                break
            continue

        raw_reasoning = _get_first_choice_reasoning_content(raw)
        if raw_reasoning is not None:
            meta["reasoning_content_present"] = bool(raw_reasoning.strip())
            meta["reasoning_content_chars"] = len(raw_reasoning)
        raw_reasoning_text = _get_first_choice_reasoning(raw)
        if raw_reasoning_text is not None:
            meta["reasoning_present"] = bool(raw_reasoning_text.strip())
            meta["reasoning_chars"] = len(raw_reasoning_text)

        content = _get_first_choice_content(raw)
        if content is None:
            attempt["parse_error"] = "no_choice_content"
            stage_attempts.append(attempt)
            fallback_reason = "no_json_payload"
            fallback_raw_response = _safe_trunc(raw)
            continue
        if _is_truncated_response(raw):
            attempt["parse_error"] = TRUNCATED_RESPONSE_ERROR
            attempt["terminal"] = True
            stage_attempts.append(attempt)
            meta["structured_output_attempts"] = stage_attempts
            return PatchResult(
                diff=None,
                meta=_with_totals(
                    _error_meta(meta, TRUNCATED_RESPONSE_ERROR, raw_response=content[:2000])
                ),
            )

        payload_candidate = _extract_json_object(content)
        if payload_candidate is None:
            attempt["parse_error"] = "no_json_payload"
            stage_attempts.append(attempt)
            fallback_reason = "no_json_payload"
            fallback_raw_response = content[:2000]
            continue

        edits, edits_shape = _normalize_edits_payload(payload_candidate)
        if edits is None:
            attempt["parse_error"] = "missing_edits"
            stage_attempts.append(attempt)
            fallback_reason = "missing_edits"
            fallback_payload = payload_candidate
            continue

        payload = dict(payload_candidate)
        payload["edits"] = edits
        if edits_shape is not None:
            attempt["edits_shape"] = edits_shape
            meta["edits_payload_shape"] = edits_shape
        stage_attempts.append(attempt)
        break

    if stage_attempts:
        meta["structured_output_attempts"] = stage_attempts

    if payload is None:
        # If prompt is already over context budget, a diff fallback call will also fail.
        if _is_context_exceeded_http_error(meta):
            return PatchResult(diff=None, meta=_with_totals(copy_patcher_meta(meta)))
        if _is_openai_compat_reasoning_only(meta):
            return PatchResult(diff=None, meta=_with_totals(copy_patcher_meta(meta)))
        if fallback_reason is None:
            return PatchResult(diff=None, meta=_with_totals(copy_patcher_meta(meta)))

        error_meta: PatcherMeta = {}
        if fallback_reason == "missing_edits":
            error_meta["payload"] = _safe_trunc(fallback_payload) if fallback_payload else "{}"
        else:
            error_meta["raw_response"] = fallback_raw_response or ""
        return _fallback_to_diff(reason=fallback_reason, extra_error_meta=error_meta)

    edits = payload.get("edits")
    if not isinstance(edits, list) or not edits:
        # Defensive guard: schema/validation above should prevent this.
        return PatchResult(diff=None, meta=_with_totals(copy_patcher_meta(meta)))

    resolved = _resolve_edits(edits, cwd=cwd)
    if resolved.failure == "file_read_error":
        return PatchResult(
            diff=None,
            meta=_with_totals(_error_meta(meta, resolved.failure, **resolved.failure_meta)),
        )
    if resolved.failure:
        # LLM-19: every payload-shape failure gets the diff fallback, not only find_not_found.
        return _fallback_to_diff(reason=resolved.failure, extra_error_meta=resolved.failure_meta)
    if not resolved.changed:
        return _fallback_to_diff(reason="no_effective_changes")

    _record_find_recovery(meta, resolved.recovery_modes)
    diff = _build_diff_from_changes(resolved.changed)
    return PatchResult(diff=diff, meta=_with_totals(copy_patcher_meta(meta)))


def _propose_patch_via_full_file(
    *,
    url: str,
    api_key: str,
    model: str,
    prompt: str,
    cwd: Path,
    temperature: float,
    max_tokens: int,
    compat_profile: str = "auto",
    allow_reasoning_fallback: bool = False,
    enable_thinking: bool = False,
    timeout_sec: float,
    seed: int | None,
    http2: bool,
    deadline_monotonic: float | None = None,
    system_prompt_override: str | None = None,
    transport_max_retries: int = _DEFAULT_CHAT_COMPLETION_TRANSPORT_MAX_RETRIES,
    transport_retry_backoff_base_sec: float = _DEFAULT_CHAT_COMPLETION_RETRY_BACKOFF_BASE_SEC,
    transport_retry_backoff_max_sec: float = _DEFAULT_CHAT_COMPLETION_RETRY_BACKOFF_MAX_SEC,
) -> PatchResult:
    totals = (0, 0, 0, 0, 0, False, True)
    target_rel, support_reason = _resolve_full_file_target(cwd=cwd)
    full_file_meta: PatcherMeta = {
        "full_file_target_path": target_rel or "",
        "full_file_supported": target_rel is not None,
        "full_file_support_reason": support_reason,
        "full_file_output_chars": 0,
        "full_file_code_fence_stripped": False,
    }

    def _with_totals(meta: PatcherMeta) -> PatcherMeta:
        return _meta_with_llm_totals(meta=meta, totals=totals)

    def _fallback_to_diff(
        *,
        reason: str,
        extra_error_meta: PatcherMeta | None = None,
    ) -> PatchResult:
        nonlocal totals
        diff_fallback = _propose_patch_via_diff(
            url=url,
            api_key=api_key,
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            compat_profile=compat_profile,
            allow_reasoning_fallback=allow_reasoning_fallback,
            enable_thinking=enable_thinking,
            timeout_sec=timeout_sec,
            seed=seed,
            http2=http2,
            deadline_monotonic=deadline_monotonic,
            system_prompt_override=system_prompt_override,
            transport_max_retries=transport_max_retries,
            transport_retry_backoff_base_sec=transport_retry_backoff_base_sec,
            transport_retry_backoff_max_sec=transport_retry_backoff_max_sec,
        )
        fallback_meta = copy_patcher_meta(diff_fallback.meta)
        totals = _accumulate_llm_totals(totals=totals, meta=fallback_meta, default_calls=0)
        if diff_fallback.diff is not None:
            fallback_meta.update(full_file_meta)
            fallback_meta["fallback_from"] = "full_file"
            fallback_meta["fallback_reason"] = reason
            if extra_error_meta:
                fallback_meta["fallback_trigger"] = copy_patcher_meta(extra_error_meta)
            return PatchResult(diff=diff_fallback.diff, meta=_with_totals(fallback_meta))

        error_meta: PatcherMeta = {
            **full_file_meta,
            "error": reason,
            "fallback_meta": fallback_meta,
        }
        if extra_error_meta:
            error_meta.update(extra_error_meta)
        return PatchResult(diff=None, meta=_with_totals(error_meta))

    if target_rel is None:
        return _fallback_to_diff(reason="unsupported_case_shape")

    target_path = (cwd / target_rel).resolve()
    try:
        target_path.relative_to(cwd.resolve())
    except Exception:
        return PatchResult(
            diff=None,
            meta=_with_totals(
                {
                    **full_file_meta,
                    "error": "path_outside_case",
                    "path": target_rel,
                }
            ),
        )
    try:
        before, before_crlf = _read_edit_target(target_path)
    except Exception as exc:
        return PatchResult(
            diff=None,
            meta=_with_totals(
                {
                    **full_file_meta,
                    "error": "file_read_error",
                    "path": target_rel,
                    "exc": f"{type(exc).__name__}: {exc}",
                }
            ),
        )

    attempts: list[list[dict[str, str]]] = [
        [
            {
                "role": "system",
                "content": system_prompt_override
                or (
                    f"You fix failing tests by returning the complete repaired contents of "
                    f"{target_rel}. Return ONLY the full contents of {target_rel}. "
                    "No prose. No markdown fences. Do not return a diff."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        [
            {
                "role": "system",
                "content": system_prompt_override
                or (
                    f"Return ONLY the complete repaired contents of {target_rel}. "
                    "Start directly with the file contents. "
                    "No prose. No markdown fences. Do not return a diff."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    ]

    last_error_meta: PatcherMeta | None = None
    for idx, messages in enumerate(attempts, start=1):
        raw, meta = _chat_completion(
            url=url,
            api_key=api_key,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            compat_profile=compat_profile,
            allow_reasoning_fallback=allow_reasoning_fallback,
            enable_thinking=enable_thinking,
            timeout_sec=timeout_sec,
            seed=seed,
            http2=http2,
            deadline_monotonic=deadline_monotonic,
            transport_max_retries=transport_max_retries,
            transport_retry_backoff_base_sec=transport_retry_backoff_base_sec,
            transport_retry_backoff_max_sec=transport_retry_backoff_max_sec,
        )
        totals = _accumulate_llm_totals(totals=totals, meta=meta, default_calls=1)
        attempt_meta: PatcherMeta = {
            **full_file_meta,
            "attempt": idx,
            **meta,
        }
        if raw is None:
            last_error_meta = attempt_meta
            if _is_openai_compat_reasoning_only(meta):
                break
            if _is_request_budget_exceeded(meta):
                break
            continue

        repaired, response_meta = _normalize_full_file_response(raw)
        attempt_meta.update(response_meta)
        full_file_meta["full_file_code_fence_stripped"] = bool(
            attempt_meta.get("full_file_code_fence_stripped")
        )
        full_file_meta["full_file_output_chars"] = _nonnegative_int(
            attempt_meta.get("full_file_output_chars"),
            default=0,
        )
        if repaired is None:
            last_error_meta = attempt_meta
            if str(attempt_meta.get("error") or "") in {
                "truncated_full_file_response",
                "returned_diff_instead_of_file",
            }:
                break
            continue
        repaired = _normalize_newlines(repaired)
        if repaired == before:
            return _fallback_to_diff(reason="no_effective_changes")
        diff = _build_diff_from_changes(
            [
                (
                    target_rel,
                    _with_line_endings(before, before_crlf),
                    _with_line_endings(repaired, before_crlf),
                )
            ]
        )
        if diff.strip():
            return PatchResult(diff=diff, meta=_with_totals(attempt_meta))
        return _fallback_to_diff(reason="no_effective_changes")

    fallback_reason = (
        str(last_error_meta.get("error") or "").strip() if last_error_meta is not None else ""
    )
    if not fallback_reason:
        fallback_reason = "empty_full_file_response"
    fallback_trigger: PatcherMeta | None = None
    if last_error_meta is not None and any(
        key in last_error_meta for key in ("raw_response", "attempt", "status", "finish_reason")
    ):
        fallback_trigger = copy_patcher_meta(
            {
                key: last_error_meta.get(key)
                for key in ("attempt", "status", "finish_reason", "error", "raw_response")
                if key in last_error_meta
            }
        )
    return _fallback_to_diff(reason=fallback_reason, extra_error_meta=fallback_trigger)


async def _propose_patch_via_diff_async(
    *,
    url: str,
    api_key: str,
    model: str,
    prompt: str,
    temperature: float,
    max_tokens: int,
    compat_profile: str = "auto",
    allow_reasoning_fallback: bool = False,
    enable_thinking: bool = False,
    timeout_sec: float,
    seed: int | None,
    http2: bool,
    deadline_monotonic: float | None = None,
    system_prompt_override: str | None = None,
    transport_max_retries: int = _DEFAULT_CHAT_COMPLETION_TRANSPORT_MAX_RETRIES,
    transport_retry_backoff_base_sec: float = _DEFAULT_CHAT_COMPLETION_RETRY_BACKOFF_BASE_SEC,
    transport_retry_backoff_max_sec: float = _DEFAULT_CHAT_COMPLETION_RETRY_BACKOFF_MAX_SEC,
) -> PatchResult:
    base_messages = [
        {
            "role": "system",
            "content": (
                system_prompt_override
                or "You fix failing tests by returning a single unified diff. "
                "Return ONLY the diff; no explanations. Do not use Markdown fences."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    attempts: list[list[dict[str, str]]] = [
        base_messages,
        [
            {
                "role": "system",
                "content": (
                    system_prompt_override
                    or "You fix failing tests by returning a single unified diff. "
                    "Return ONLY the diff; no explanations. Do not use Markdown fences. "
                    "Start your response with 'diff --git a/'."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    ]

    last_meta: PatcherMeta | None = None
    totals = (0, 0, 0, 0, 0, False, True)
    for idx, messages in enumerate(attempts, start=1):
        raw, meta = await _chat_completion_async(
            url=url,
            api_key=api_key,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            compat_profile=compat_profile,
            allow_reasoning_fallback=allow_reasoning_fallback,
            enable_thinking=enable_thinking,
            timeout_sec=timeout_sec,
            seed=seed,
            http2=http2,
            deadline_monotonic=deadline_monotonic,
            transport_max_retries=transport_max_retries,
            transport_retry_backoff_base_sec=transport_retry_backoff_base_sec,
            transport_retry_backoff_max_sec=transport_retry_backoff_max_sec,
        )
        totals = _accumulate_llm_totals(totals=totals, meta=meta, default_calls=1)
        last_meta = _meta_with_llm_totals(meta={"attempt": idx, **meta}, totals=totals)
        if raw is None:
            if _is_openai_compat_reasoning_only(meta):
                break
            if _is_request_budget_exceeded(meta):
                break
            continue

        content = _get_first_choice_content(raw)
        if content is None:
            last_meta["error"] = "no_choice_content"
            last_meta["response"] = _safe_trunc(raw)
            continue
        if _is_truncated_response(raw):
            last_meta = _error_meta(
                last_meta, TRUNCATED_RESPONSE_ERROR, raw_response=content[:2000]
            )
            break

        diff = _extract_unified_diff(content)
        if diff is None:
            last_meta["error"] = "no_diff_found"
            last_meta["raw_response"] = content[:2000]
            continue

        fixed = _fix_hunk_counts(diff)
        return PatchResult(diff=fixed, meta=last_meta)

    if last_meta is None:
        last_meta = _meta_with_llm_totals(meta={"error": "unknown"}, totals=totals)
    return PatchResult(diff=None, meta=last_meta)


async def _propose_patch_via_edits_async(
    *,
    url: str,
    api_key: str,
    model: str,
    prompt: str,
    cwd: Path,
    temperature: float,
    max_tokens: int,
    compat_profile: str = "auto",
    allow_reasoning_fallback: bool = False,
    enable_thinking: bool = False,
    timeout_sec: float,
    seed: int | None,
    http2: bool,
    deadline_monotonic: float | None = None,
    allow_diff_fallback: bool = True,
    system_prompt_override: str | None = None,
    transport_max_retries: int = _DEFAULT_CHAT_COMPLETION_TRANSPORT_MAX_RETRIES,
    transport_retry_backoff_base_sec: float = _DEFAULT_CHAT_COMPLETION_RETRY_BACKOFF_BASE_SEC,
    transport_retry_backoff_max_sec: float = _DEFAULT_CHAT_COMPLETION_RETRY_BACKOFF_MAX_SEC,
) -> PatchResult:
    messages = [
        {
            "role": "system",
            "content": (
                system_prompt_override
                or "Return only JSON with this schema: "
                '{"edits":[{"path":"relative/path.py","find":"exact old text","replace":"new text","count":1}]}. '
                "No prose. No markdown. "
                "Use only exact existing paths from the prompt's available file list. "
                "Never use placeholder paths like relative/path.py."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    totals = (0, 0, 0, 0, 0, False, True)

    def _with_totals(meta: PatcherMeta) -> PatcherMeta:
        return _meta_with_llm_totals(meta=meta, totals=totals)

    payload: dict[str, Any] | None = None
    meta: PatcherMeta = {}
    stages = [
        (
            "json_schema",
            {
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "patch_edits",
                        "schema": PATCH_EDITS_JSON_SCHEMA,
                    },
                }
            },
        ),
        ("json_object", {"response_format": {"type": "json_object"}}),
        ("plain", None),
    ]
    stage_attempts: list[StructuredOutputAttempt] = []
    fallback_reason: str | None = None
    fallback_raw_response: str | None = None
    fallback_payload: dict[str, Any] | None = None

    async def _fallback_to_diff(
        *,
        reason: str,
        extra_error_meta: PatcherMeta | None = None,
    ) -> PatchResult:
        nonlocal totals
        if not allow_diff_fallback:
            error_meta = _error_meta(meta, reason)
            if extra_error_meta:
                error_meta.update(extra_error_meta)
            return PatchResult(diff=None, meta=_with_totals(error_meta))

        diff_fallback = await _propose_patch_via_diff_async(
            url=url,
            api_key=api_key,
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            compat_profile=compat_profile,
            allow_reasoning_fallback=allow_reasoning_fallback,
            enable_thinking=enable_thinking,
            timeout_sec=timeout_sec,
            seed=seed,
            http2=http2,
            deadline_monotonic=deadline_monotonic,
            system_prompt_override=system_prompt_override,
            transport_max_retries=transport_max_retries,
            transport_retry_backoff_base_sec=transport_retry_backoff_base_sec,
            transport_retry_backoff_max_sec=transport_retry_backoff_max_sec,
        )
        if diff_fallback.diff is not None:
            fallback_meta = copy_patcher_meta(diff_fallback.meta)
            totals = _accumulate_llm_totals(totals=totals, meta=fallback_meta, default_calls=0)
            fallback_meta["fallback_from"] = "edits"
            fallback_meta["fallback_reason"] = reason
            fallback_meta["edits_response_mode"] = str(meta.get("structured_output_mode") or "")
            fallback_meta["edits_response_attempts"] = stage_attempts
            if extra_error_meta:
                fallback_meta["fallback_trigger"] = copy_patcher_meta(extra_error_meta)
            return PatchResult(diff=diff_fallback.diff, meta=_with_totals(fallback_meta))

        fallback_meta = copy_patcher_meta(diff_fallback.meta)
        totals = _accumulate_llm_totals(totals=totals, meta=fallback_meta, default_calls=0)
        error_meta = _error_meta(meta, reason, fallback_meta=fallback_meta)
        if extra_error_meta:
            error_meta.update(extra_error_meta)
        return PatchResult(diff=None, meta=_with_totals(error_meta))

    for stage_name, extra_payload in stages:
        raw, stage_meta = await _chat_completion_async(
            url=url,
            api_key=api_key,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            compat_profile=compat_profile,
            allow_reasoning_fallback=allow_reasoning_fallback,
            enable_thinking=enable_thinking,
            timeout_sec=timeout_sec,
            seed=seed,
            http2=http2,
            extra_payload=extra_payload,
            deadline_monotonic=deadline_monotonic,
            transport_max_retries=transport_max_retries,
            transport_retry_backoff_base_sec=transport_retry_backoff_base_sec,
            transport_retry_backoff_max_sec=transport_retry_backoff_max_sec,
        )
        totals = _accumulate_llm_totals(totals=totals, meta=stage_meta, default_calls=1)
        meta = copy_patcher_meta(stage_meta)
        meta["structured_output_mode"] = stage_name
        attempt: StructuredOutputAttempt = {
            "mode": stage_name,
            "ok": raw is not None,
            "error": str(meta.get("error", "")),
            "status": meta.get("status"),
        }
        if raw is None:
            stage_attempts.append(attempt)
            if _is_context_exceeded_http_error(meta) or _is_request_budget_exceeded(meta):
                attempt["terminal"] = True
                break
            if _is_openai_compat_reasoning_only(meta):
                attempt["terminal"] = True
                break
            continue

        raw_reasoning = _get_first_choice_reasoning_content(raw)
        if raw_reasoning is not None:
            meta["reasoning_content_present"] = bool(raw_reasoning.strip())
            meta["reasoning_content_chars"] = len(raw_reasoning)
        raw_reasoning_text = _get_first_choice_reasoning(raw)
        if raw_reasoning_text is not None:
            meta["reasoning_present"] = bool(raw_reasoning_text.strip())
            meta["reasoning_chars"] = len(raw_reasoning_text)

        content = _get_first_choice_content(raw)
        if content is None:
            attempt["parse_error"] = "no_choice_content"
            stage_attempts.append(attempt)
            fallback_reason = "no_json_payload"
            fallback_raw_response = _safe_trunc(raw)
            continue
        if _is_truncated_response(raw):
            attempt["parse_error"] = TRUNCATED_RESPONSE_ERROR
            attempt["terminal"] = True
            stage_attempts.append(attempt)
            meta["structured_output_attempts"] = stage_attempts
            return PatchResult(
                diff=None,
                meta=_with_totals(
                    _error_meta(meta, TRUNCATED_RESPONSE_ERROR, raw_response=content[:2000])
                ),
            )

        payload_candidate = _extract_json_object(content)
        if payload_candidate is None:
            attempt["parse_error"] = "no_json_payload"
            stage_attempts.append(attempt)
            fallback_reason = "no_json_payload"
            fallback_raw_response = content[:2000]
            continue

        edits, edits_shape = _normalize_edits_payload(payload_candidate)
        if edits is None:
            attempt["parse_error"] = "missing_edits"
            stage_attempts.append(attempt)
            fallback_reason = "missing_edits"
            fallback_payload = payload_candidate
            continue

        payload = dict(payload_candidate)
        payload["edits"] = edits
        if edits_shape is not None:
            attempt["edits_shape"] = edits_shape
            meta["edits_payload_shape"] = edits_shape
        stage_attempts.append(attempt)
        break

    if stage_attempts:
        meta["structured_output_attempts"] = stage_attempts

    if payload is None:
        if _is_context_exceeded_http_error(meta):
            return PatchResult(diff=None, meta=_with_totals(copy_patcher_meta(meta)))
        if _is_openai_compat_reasoning_only(meta):
            return PatchResult(diff=None, meta=_with_totals(copy_patcher_meta(meta)))
        if fallback_reason is None:
            return PatchResult(diff=None, meta=_with_totals(copy_patcher_meta(meta)))

        error_meta: PatcherMeta = {}
        if fallback_reason == "missing_edits":
            error_meta["payload"] = _safe_trunc(fallback_payload) if fallback_payload else "{}"
        else:
            error_meta["raw_response"] = fallback_raw_response or ""
        return await _fallback_to_diff(reason=fallback_reason, extra_error_meta=error_meta)

    edits = payload.get("edits")
    if not isinstance(edits, list) or not edits:
        return PatchResult(diff=None, meta=_with_totals(copy_patcher_meta(meta)))

    resolved = _resolve_edits(edits, cwd=cwd)
    if resolved.failure == "file_read_error":
        return PatchResult(
            diff=None,
            meta=_with_totals(_error_meta(meta, resolved.failure, **resolved.failure_meta)),
        )
    if resolved.failure:
        # LLM-19: every payload-shape failure gets the diff fallback, not only find_not_found.
        return await _fallback_to_diff(
            reason=resolved.failure, extra_error_meta=resolved.failure_meta
        )
    if not resolved.changed:
        return await _fallback_to_diff(reason="no_effective_changes")

    _record_find_recovery(meta, resolved.recovery_modes)
    diff = _build_diff_from_changes(resolved.changed)
    return PatchResult(diff=diff, meta=_with_totals(copy_patcher_meta(meta)))


async def _propose_patch_via_full_file_async(
    *,
    url: str,
    api_key: str,
    model: str,
    prompt: str,
    cwd: Path,
    temperature: float,
    max_tokens: int,
    compat_profile: str = "auto",
    allow_reasoning_fallback: bool = False,
    enable_thinking: bool = False,
    timeout_sec: float,
    seed: int | None,
    http2: bool,
    deadline_monotonic: float | None = None,
    system_prompt_override: str | None = None,
    transport_max_retries: int = _DEFAULT_CHAT_COMPLETION_TRANSPORT_MAX_RETRIES,
    transport_retry_backoff_base_sec: float = _DEFAULT_CHAT_COMPLETION_RETRY_BACKOFF_BASE_SEC,
    transport_retry_backoff_max_sec: float = _DEFAULT_CHAT_COMPLETION_RETRY_BACKOFF_MAX_SEC,
) -> PatchResult:
    totals = (0, 0, 0, 0, 0, False, True)
    target_rel, support_reason = _resolve_full_file_target(cwd=cwd)
    full_file_meta: PatcherMeta = {
        "full_file_target_path": target_rel or "",
        "full_file_supported": target_rel is not None,
        "full_file_support_reason": support_reason,
        "full_file_output_chars": 0,
        "full_file_code_fence_stripped": False,
    }

    def _with_totals(meta: PatcherMeta) -> PatcherMeta:
        return _meta_with_llm_totals(meta=meta, totals=totals)

    async def _fallback_to_diff(
        *,
        reason: str,
        extra_error_meta: PatcherMeta | None = None,
    ) -> PatchResult:
        nonlocal totals
        diff_fallback = await _propose_patch_via_diff_async(
            url=url,
            api_key=api_key,
            model=model,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
            compat_profile=compat_profile,
            allow_reasoning_fallback=allow_reasoning_fallback,
            enable_thinking=enable_thinking,
            timeout_sec=timeout_sec,
            seed=seed,
            http2=http2,
            deadline_monotonic=deadline_monotonic,
            system_prompt_override=system_prompt_override,
            transport_max_retries=transport_max_retries,
            transport_retry_backoff_base_sec=transport_retry_backoff_base_sec,
            transport_retry_backoff_max_sec=transport_retry_backoff_max_sec,
        )
        fallback_meta = copy_patcher_meta(diff_fallback.meta)
        totals = _accumulate_llm_totals(totals=totals, meta=fallback_meta, default_calls=0)
        if diff_fallback.diff is not None:
            fallback_meta.update(full_file_meta)
            fallback_meta["fallback_from"] = "full_file"
            fallback_meta["fallback_reason"] = reason
            if extra_error_meta:
                fallback_meta["fallback_trigger"] = copy_patcher_meta(extra_error_meta)
            return PatchResult(diff=diff_fallback.diff, meta=_with_totals(fallback_meta))

        error_meta: PatcherMeta = {
            **full_file_meta,
            "error": reason,
            "fallback_meta": fallback_meta,
        }
        if extra_error_meta:
            error_meta.update(extra_error_meta)
        return PatchResult(diff=None, meta=_with_totals(error_meta))

    if target_rel is None:
        return await _fallback_to_diff(reason="unsupported_case_shape")

    target_path = (cwd / target_rel).resolve()
    try:
        target_path.relative_to(cwd.resolve())
    except Exception:
        return PatchResult(
            diff=None,
            meta=_with_totals(
                {
                    **full_file_meta,
                    "error": "path_outside_case",
                    "path": target_rel,
                }
            ),
        )
    try:
        before, before_crlf = _read_edit_target(target_path)
    except Exception as exc:
        return PatchResult(
            diff=None,
            meta=_with_totals(
                {
                    **full_file_meta,
                    "error": "file_read_error",
                    "path": target_rel,
                    "exc": f"{type(exc).__name__}: {exc}",
                }
            ),
        )

    attempts: list[list[dict[str, str]]] = [
        [
            {
                "role": "system",
                "content": (
                    system_prompt_override
                    or f"You fix failing tests by returning the complete repaired contents of "
                    f"{target_rel}. Return ONLY the full contents of {target_rel}. "
                    "No prose. No markdown fences. Do not return a diff."
                ),
            },
            {"role": "user", "content": prompt},
        ],
        [
            {
                "role": "system",
                "content": (
                    system_prompt_override
                    or f"Return ONLY the complete repaired contents of {target_rel}. "
                    "Start directly with the file contents. "
                    "No prose. No markdown fences. Do not return a diff."
                ),
            },
            {"role": "user", "content": prompt},
        ],
    ]

    last_error_meta: PatcherMeta | None = None
    for idx, messages in enumerate(attempts, start=1):
        raw, meta = await _chat_completion_async(
            url=url,
            api_key=api_key,
            model=model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            compat_profile=compat_profile,
            allow_reasoning_fallback=allow_reasoning_fallback,
            enable_thinking=enable_thinking,
            timeout_sec=timeout_sec,
            seed=seed,
            http2=http2,
            deadline_monotonic=deadline_monotonic,
            transport_max_retries=transport_max_retries,
            transport_retry_backoff_base_sec=transport_retry_backoff_base_sec,
            transport_retry_backoff_max_sec=transport_retry_backoff_max_sec,
        )
        totals = _accumulate_llm_totals(totals=totals, meta=meta, default_calls=1)
        attempt_meta: PatcherMeta = {
            **full_file_meta,
            "attempt": idx,
            **meta,
        }
        if raw is None:
            last_error_meta = attempt_meta
            if _is_openai_compat_reasoning_only(meta):
                break
            if _is_request_budget_exceeded(meta):
                break
            continue

        repaired, response_meta = _normalize_full_file_response(raw)
        attempt_meta.update(response_meta)
        full_file_meta["full_file_code_fence_stripped"] = bool(
            attempt_meta.get("full_file_code_fence_stripped")
        )
        full_file_meta["full_file_output_chars"] = _nonnegative_int(
            attempt_meta.get("full_file_output_chars"),
            default=0,
        )
        if repaired is None:
            last_error_meta = attempt_meta
            if str(attempt_meta.get("error") or "") in {
                "truncated_full_file_response",
                "returned_diff_instead_of_file",
            }:
                break
            continue
        repaired = _normalize_newlines(repaired)
        if repaired == before:
            return await _fallback_to_diff(reason="no_effective_changes")
        diff = _build_diff_from_changes(
            [
                (
                    target_rel,
                    _with_line_endings(before, before_crlf),
                    _with_line_endings(repaired, before_crlf),
                )
            ]
        )
        if diff.strip():
            return PatchResult(diff=diff, meta=_with_totals(attempt_meta))
        return await _fallback_to_diff(reason="no_effective_changes")

    fallback_reason = (
        str(last_error_meta.get("error") or "").strip() if last_error_meta is not None else ""
    )
    if not fallback_reason:
        fallback_reason = "empty_full_file_response"
    fallback_trigger: PatcherMeta | None = None
    if last_error_meta is not None and any(
        key in last_error_meta for key in ("raw_response", "attempt", "status", "finish_reason")
    ):
        fallback_trigger = copy_patcher_meta(
            {
                key: last_error_meta.get(key)
                for key in ("attempt", "status", "finish_reason", "error", "raw_response")
                if key in last_error_meta
            }
        )
    return await _fallback_to_diff(reason=fallback_reason, extra_error_meta=fallback_trigger)


def _request_context_failure_kind(
    *,
    meta: PatcherMeta,
    request_json: ContextRequestPayload | None,
    raw_response: str | None,
) -> str:
    if request_json is not None:
        return ""
    error = str(meta.get("error", "")).strip()
    if error == "request_budget_exceeded":
        return "request_budget_exceeded"
    if error == _OPENAI_COMPAT_REASONING_ONLY_ERROR:
        return "compat_reasoning_only"
    if _is_backend_unavailable_error(meta):
        return "backend_unavailable"
    if _is_retryable_chat_completion_error(meta):
        return "transport"
    if raw_response is not None:
        return "invalid_payload"
    if error == "invalid_request_context_payload":
        return "invalid_payload"
    return ""


def _request_context_outcome(
    *,
    meta: PatcherMeta,
    request_json: ContextRequestPayload | None,
    raw_response: str | None,
) -> str:
    if request_json is not None:
        return "valid_payload"
    failure_kind = _request_context_failure_kind(
        meta=meta,
        request_json=request_json,
        raw_response=raw_response,
    )
    return {
        "transport": "transport_failed",
        "backend_unavailable": "unavailable_backend",
        "request_budget_exceeded": "request_budget_exceeded",
        "compat_reasoning_only": "compat_reasoning_only",
        "invalid_payload": "invalid_payload",
    }.get(failure_kind, "")


def _is_request_budget_exceeded(meta: PatcherMeta) -> bool:
    return str(meta.get("error", "")) == "request_budget_exceeded"


def _is_openai_compat_reasoning_only(meta: PatcherMeta | None) -> bool:
    if not isinstance(meta, dict):
        return False
    return str(meta.get("error", "")).strip() == _OPENAI_COMPAT_REASONING_ONLY_ERROR


def _forced_patch_skip_reason(meta: PatcherMeta) -> str:
    error = str(meta.get("error", "")).strip()
    if error in {"request_failed", "bad_request_url"}:
        return error
    if _is_request_budget_exceeded(meta):
        return "request_budget_exceeded"
    if error == _OPENAI_COMPAT_REASONING_ONLY_ERROR:
        return error
    if _is_context_exceeded_http_error(meta):
        return "context_limit"

    fallback_meta = meta.get("fallback_meta")
    if isinstance(fallback_meta, dict):
        nested = _forced_patch_skip_reason(cast(PatcherMeta, fallback_meta))
        if nested:
            return f"fallback_{nested}"
    return ""


def _forced_patch_skip_category(skip_reason: str) -> str:
    if skip_reason.endswith(_OPENAI_COMPAT_REASONING_ONLY_ERROR):
        return "compatibility_failure"
    return "transport_or_context_failure"
