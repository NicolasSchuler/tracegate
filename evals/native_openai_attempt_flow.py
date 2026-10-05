"""Legacy native OpenAI attempt orchestration helpers for the eval harness."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from evals.patchers.base import replace_patcher

from .eval_prompts import (
    _build_context_compaction_bundle,
    _should_invoke_context_compaction,
    render_context_compaction_prompt,
)
from .eval_types import PromptBudgetResult
from .native_stage_b_flow import (
    RetryPlanResult,
    StageBPromptInputs,
    build_stage_b_prompt_with_budget,
    maybe_run_retry_plan_then_patch,
)


@dataclass(frozen=True)
class ContextCompactionResult:
    requested: bool = False
    request_reason: str = ""
    invoked: bool = False
    success: bool = False
    trigger: str = "not_needed"
    error: str = ""
    auto_overflow_triggered: bool = False
    replaced_sections: list[str] = field(default_factory=list)
    source_sections: list[str] = field(default_factory=list)
    source_chars: dict[str, int] = field(default_factory=dict)
    input_chars: int = 0
    output_chars: int = 0
    prompt_text: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    calls_used: int = 0


@dataclass(frozen=True)
class NativeOpenAIAttemptPromptRequest:
    prompt_inputs: StageBPromptInputs
    attempt_prompt_budget: dict[str, int]
    is_delta_retry: bool
    include_stage_a_evidence_index: bool
    stage_a_index_reason: str
    project_files_context: str
    current_stdout: str
    current_stderr: str
    context_compaction_requested: bool
    context_compaction_request_reason: str
    policy_compact_then_patch_requested: bool
    context_compaction_mode: str
    context_compaction_scope: set[str]
    supports_context_compaction: bool
    context_compaction_calls_used: int
    context_compaction_max_calls: int
    context_compaction_max_input_chars: int
    context_compaction_target_chars: int
    context_compaction_fn: Callable[..., Any] | None
    patcher_for_attempt: Any
    run_openai_operation_with_transport_recovery: Callable[..., Any]
    classify_transport_recovery_failure_kind: Callable[[Any], str]
    attempt_context_protocol: str
    before: Any
    cumulative_runtime_sec: float
    attempt_dir: Path
    workdir: Path
    retry_plan_enabled: bool
    plan_on_first_attempt: bool = False
    tep_equilibrium_check: bool = False
    tep_equilibrium_max_iterations: int = 2
    tep_equilibrium_similarity_threshold: float = 0.85
    context_compaction_actor_overrides: dict[str, str] | None = None
    run_plan_patch: Callable[[str], tuple[str | None, str | None, dict[str, Any]]] | None = None


@dataclass(frozen=True)
class NativeOpenAIAttemptPromptResult:
    prompt_inputs: StageBPromptInputs
    prompt: str
    prompt_budget_result: PromptBudgetResult
    include_stage_a_evidence_index: bool
    stage_a_index_reason: str
    context_compaction: ContextCompactionResult
    retry_plan: RetryPlanResult
    prompt_assembly_sec: float
    context_compaction_total_sec: float


def _context_compaction_auto_overflow_triggered(
    *,
    prompt_budget_result: PromptBudgetResult,
) -> bool:
    return bool(
        prompt_budget_result.overflow_chars > 0
        or prompt_budget_result.section_status.get("snapshot") in {"truncated", "dropped"}
        or prompt_budget_result.section_status.get("pytest_output") in {"truncated", "dropped"}
        or prompt_budget_result.section_status.get("project_files") in {"truncated", "dropped"}
        or prompt_budget_result.section_status.get("stage_a_context") in {"truncated", "dropped"}
    )


def _replace_prompt_inputs_for_compaction(
    *,
    prompt_inputs: StageBPromptInputs,
    compacted_context: str,
    replace_snapshot: bool,
    replace_pytest_output: bool,
    replace_stage_a_context: bool,
    replace_project_files: bool,
) -> StageBPromptInputs:
    return replace(
        prompt_inputs,
        compacted_context=compacted_context,
        include_snapshot=not replace_snapshot,
        include_pytest_output=not replace_pytest_output,
        include_stage_a_context=not replace_stage_a_context,
        include_project_files=prompt_inputs.include_project_files and not replace_project_files,
    )


def assemble_native_openai_attempt_prompt(
    *,
    request: NativeOpenAIAttemptPromptRequest,
) -> NativeOpenAIAttemptPromptResult:
    prompt_assembly_start = time.perf_counter()
    prompt_inputs = request.prompt_inputs
    prompt_result = build_stage_b_prompt_with_budget(
        prompt_inputs=prompt_inputs,
        attempt_prompt_budget=request.attempt_prompt_budget,
        is_delta_retry=request.is_delta_retry,
        include_stage_a_evidence_index=request.include_stage_a_evidence_index,
        stage_a_index_reason=request.stage_a_index_reason,
    )
    include_stage_a_evidence_index = prompt_result.include_stage_a_evidence_index
    stage_a_index_reason = prompt_result.stage_a_index_reason

    context_compaction_bundle = _build_context_compaction_bundle(
        scope=request.context_compaction_scope,
        snapshot=prompt_inputs.snapshot,
        stdout=request.current_stdout,
        stderr=request.current_stderr,
        stage_a_context=prompt_inputs.stage_a_context,
        project_files_context=(
            request.project_files_context if prompt_inputs.include_project_files else ""
        ),
        max_input_chars=request.context_compaction_max_input_chars,
    )
    context_compaction_auto_overflow_triggered = _context_compaction_auto_overflow_triggered(
        prompt_budget_result=prompt_result.prompt_budget_result
    )
    context_compaction_input_chars = len(context_compaction_bundle.text)
    should_compact_context, context_compaction_trigger = _should_invoke_context_compaction(
        mode=request.context_compaction_mode,
        supports_context_compaction=request.supports_context_compaction,
        stage_a_requested=request.context_compaction_requested,
        policy_action_requested=request.policy_compact_then_patch_requested,
        auto_overflow_triggered=context_compaction_auto_overflow_triggered,
        calls_used=request.context_compaction_calls_used,
        max_calls=request.context_compaction_max_calls,
        bundle_input_chars=context_compaction_input_chars,
    )

    context_compaction_invoked = False
    context_compaction_success = False
    context_compaction_error = ""
    context_compaction_output_chars = 0
    context_compaction_prompt_text: str | None = None
    context_compaction_meta: dict[str, Any] = {}
    context_compaction_replaced_sections: list[str] = []
    context_compaction_calls_used = int(request.context_compaction_calls_used)
    context_compaction_total_sec = 0.0

    if should_compact_context:
        context_compaction_started = time.perf_counter()
        context_compaction_invoked = True
        context_compaction_calls_used += 1
        context_compaction_prompt_text = render_context_compaction_prompt(
            evidence=prompt_inputs.evidence,
            attempt=prompt_inputs.attempt,
            reason=request.context_compaction_request_reason,
            source_sections=list(context_compaction_bundle.source_sections),
            target_chars=request.context_compaction_target_chars,
            tep_actor_overrides=request.context_compaction_actor_overrides,
        )
        try:
            compaction_result = request.run_openai_operation_with_transport_recovery(
                patcher_for_attempt=request.patcher_for_attempt,
                operation_name="compact_context",
                invoke=lambda context_compaction_fn=request.context_compaction_fn,
                context_compaction_prompt_text=context_compaction_prompt_text,
                context_compaction_bundle=context_compaction_bundle,
                attempt_dir=request.attempt_dir,
                workdir=request.workdir,
                normalized_context_compaction_target_chars=request.context_compaction_target_chars: context_compaction_fn(  # pyright: ignore[reportOptionalCall]
                    prompt=context_compaction_prompt_text,
                    context_text=context_compaction_bundle.text,
                    context_dir=request.attempt_dir,
                    cwd=request.workdir,
                    target_chars=normalized_context_compaction_target_chars,
                ),
                classify_failure_kind=request.classify_transport_recovery_failure_kind,
                attempt=prompt_inputs.attempt,
                attempt_metadata=None,
                attempt_context_protocol=request.attempt_context_protocol,
                before=request.before,
                cumulative_runtime_sec_value=request.cumulative_runtime_sec,
            )
        except Exception as exc:
            compaction_result = None
            context_compaction_error = f"{type(exc).__name__}: {exc}"
        if compaction_result is not None:
            if hasattr(compaction_result, "meta") and isinstance(compaction_result.meta, dict):
                context_compaction_meta = dict(compaction_result.meta)
            compacted_text_candidate = (
                compaction_result.compacted_text
                if hasattr(compaction_result, "compacted_text")
                else None
            )
            if isinstance(compacted_text_candidate, str) and compacted_text_candidate.strip():
                compacted_context_for_prompt = compacted_text_candidate.strip()
                context_compaction_output_chars = len(compacted_context_for_prompt)
                context_compaction_success = True
                if context_compaction_bundle.replace_snapshot:
                    context_compaction_replaced_sections.append("snapshot")
                if context_compaction_bundle.replace_pytest_output:
                    context_compaction_replaced_sections.append("pytest_output")
                if context_compaction_bundle.replace_stage_a_context:
                    context_compaction_replaced_sections.append("stage_a_context")
                if context_compaction_bundle.replace_project_files:
                    context_compaction_replaced_sections.append("project_files")
                prompt_inputs = _replace_prompt_inputs_for_compaction(
                    prompt_inputs=prompt_inputs,
                    compacted_context=compacted_context_for_prompt,
                    replace_snapshot=bool(context_compaction_bundle.replace_snapshot),
                    replace_pytest_output=bool(context_compaction_bundle.replace_pytest_output),
                    replace_stage_a_context=bool(context_compaction_bundle.replace_stage_a_context),
                    replace_project_files=bool(context_compaction_bundle.replace_project_files),
                )
                prompt_result = build_stage_b_prompt_with_budget(
                    prompt_inputs=prompt_inputs,
                    attempt_prompt_budget=request.attempt_prompt_budget,
                    is_delta_retry=request.is_delta_retry,
                    include_stage_a_evidence_index=include_stage_a_evidence_index,
                    stage_a_index_reason=stage_a_index_reason,
                )
                include_stage_a_evidence_index = prompt_result.include_stage_a_evidence_index
                stage_a_index_reason = prompt_result.stage_a_index_reason
            elif not context_compaction_error:
                context_compaction_error = str(context_compaction_meta.get("error") or "")
        elif not context_compaction_error:
            context_compaction_error = "compact_context_unavailable"
        context_compaction_total_sec = time.perf_counter() - context_compaction_started

    retry_plan_started = time.perf_counter()

    def _call_retry_plan() -> RetryPlanResult:
        kwargs = {
            "enabled": request.retry_plan_enabled,
            "plan_on_first_attempt": request.plan_on_first_attempt,
            "attempt": prompt_inputs.attempt,
            "prompt_inputs": prompt_inputs,
            "attempt_prompt_budget": request.attempt_prompt_budget,
            "is_delta_retry": request.is_delta_retry,
            "include_stage_a_evidence_index": include_stage_a_evidence_index,
            "stage_a_index_reason": stage_a_index_reason,
            "tep_equilibrium_check": request.tep_equilibrium_check,
            "tep_equilibrium_max_iterations": request.tep_equilibrium_max_iterations,
            "tep_equilibrium_similarity_threshold": request.tep_equilibrium_similarity_threshold,
            "run_plan_patch": request.run_plan_patch
            if request.run_plan_patch is not None
            else lambda _prompt: (None, None, {}),
        }
        # ATT-22: no TypeError-message parsing; the callee takes every one of these.
        return maybe_run_retry_plan_then_patch(**kwargs)

    retry_plan = _call_retry_plan()
    retry_plan = replace(
        retry_plan,
        retry_plan_duration_sec=time.perf_counter() - retry_plan_started,
    )
    context_compaction = ContextCompactionResult(
        requested=request.context_compaction_requested,
        request_reason=request.context_compaction_request_reason,
        invoked=context_compaction_invoked,
        success=context_compaction_success,
        trigger=context_compaction_trigger,
        error=context_compaction_error,
        auto_overflow_triggered=context_compaction_auto_overflow_triggered,
        replaced_sections=context_compaction_replaced_sections,
        source_sections=list(context_compaction_bundle.source_sections),
        source_chars=dict(context_compaction_bundle.source_chars),
        input_chars=context_compaction_input_chars,
        output_chars=context_compaction_output_chars,
        prompt_text=context_compaction_prompt_text,
        meta=context_compaction_meta,
        calls_used=context_compaction_calls_used,
    )
    return NativeOpenAIAttemptPromptResult(
        prompt_inputs=prompt_inputs,
        prompt=retry_plan.prompt,
        prompt_budget_result=retry_plan.prompt_budget_result,
        include_stage_a_evidence_index=retry_plan.include_stage_a_evidence_index,
        stage_a_index_reason=retry_plan.stage_a_index_reason,
        context_compaction=context_compaction,
        retry_plan=retry_plan,
        prompt_assembly_sec=time.perf_counter() - prompt_assembly_start,
        context_compaction_total_sec=context_compaction_total_sec,
    )


def sync_openai_response_mode_autoswitch_metadata(
    *,
    attempt_metadata: dict[str, Any],
    patch_meta: dict[str, Any],
    policy: str,
    switched: bool,
    from_mode: str = "",
    to_mode: str = "",
    cycle: int = 0,
    reason: str = "",
) -> None:
    values: dict[str, Any] = {
        "openai_response_mode_autoswitch_policy": str(policy or "off"),
        "openai_response_mode_autoswitched": bool(switched),
        "openai_response_mode_autoswitch_from": str(from_mode or ""),
        "openai_response_mode_autoswitch_to": str(to_mode or ""),
        "openai_response_mode_autoswitch_cycle": int(cycle),
        "openai_response_mode_autoswitch_reason": str(reason or ""),
    }
    patch_meta.update(values)
    attempt_metadata.update(values)


def maybe_autoswitch_openai_response_mode(
    *,
    patcher_for_attempt: Any,
    policy: str,
    reason: str,
    syntax_cycle: int,
    patch_meta: dict[str, Any],
    attempt_metadata: dict[str, Any],
) -> tuple[Any, bool]:
    if policy != "retry_to_diff":
        return patcher_for_attempt, False
    current_mode = str(getattr(patcher_for_attempt, "response_mode", "")).strip()
    if current_mode != "edits":
        return patcher_for_attempt, False
    try:
        next_patcher = replace_patcher(patcher_for_attempt, response_mode="diff")
    except Exception as exc:
        patch_meta["openai_response_mode_autoswitch_error"] = f"{type(exc).__name__}: {exc}"
        return patcher_for_attempt, False
    sync_openai_response_mode_autoswitch_metadata(
        attempt_metadata=attempt_metadata,
        patch_meta=patch_meta,
        policy=policy,
        switched=True,
        from_mode="edits",
        to_mode="diff",
        cycle=syntax_cycle,
        reason=reason,
    )
    return next_patcher, True


__all__ = [
    "ContextCompactionResult",
    "NativeOpenAIAttemptPromptRequest",
    "NativeOpenAIAttemptPromptResult",
    "assemble_native_openai_attempt_prompt",
    "maybe_autoswitch_openai_response_mode",
    "sync_openai_response_mode_autoswitch_metadata",
]
