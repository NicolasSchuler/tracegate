"""Controller-owned adaptive runtime helpers for hybrid baseline runs."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from baselines.controller_packet import parse_controller_packet

from .adaptive_controller_packet import (
    build_adaptive_controller_sidecars,
    write_adaptive_controller_sidecars,
)
from .controller_summarizer import build_controller_strategy_summary
from .lambda_rlm_backend import invoke_lambda_rlm_decomposition
from .patchers.baseline_patcher import resolve_effective_baseline_model


@dataclass(frozen=True)
class ControllerRuntimeInputs:
    case_id: str
    condition_value: str
    attempt: int
    workdir: Path
    attempt_dir: Path
    snapshot: dict[str, Any] | None
    failure_signature: str
    evidence_items: list[dict[str, Any]]
    new_evidence_ids: list[str]
    reused_evidence_ids: list[str]
    retry_prompt_mode: str
    retry_context_mode: str
    retry_history_text: str
    retry_history_turns: list[str]
    retry_history_turns_included: int
    retry_history_truncated: bool
    previous_failure_signature: str
    previous_note: str
    previous_patch_verifier_rejected: bool
    previous_patch_verifier_reasons: list[str]
    previous_patch_verifier_scope: str
    previous_timed_out: bool
    previous_timeout_sec: int | None
    helper_action_selected: str
    helper_action_executed: str
    helper_decision_reason: str
    helper_outcome: str
    helper_materialized_context: bool
    helper_calls_used: int
    helper_budget_max_calls: int
    no_progress_streak: int
    context_compaction_requested: bool
    context_compaction_invoked: bool
    context_compaction_success: bool
    context_compaction_trigger: str
    prompt_budget: dict[str, int]
    retry_plan_text: str
    lambda_rlm_enabled: bool = False
    lambda_rlm_repo_dir: Path | None = None
    lambda_rlm_python: Path | None = None
    lambda_rlm_context_window_chars: int = 100_000
    lambda_rlm_timeout_sec: int = 120
    case_timeout_sec: int = 0
    patcher_name: str = ""
    configured_model: str = ""
    configured_api_base_url: str = ""
    configured_api_key: str = ""
    lambda_rlm_invoke: Callable[..., Any] | None = None
    resolve_effective_model: Callable[..., str] | None = None


@dataclass(frozen=True)
class PreparedControllerRuntime:
    inputs: ControllerRuntimeInputs
    controller_strategy_summary: str = ""
    controller_strategy_sources: list[str] = field(default_factory=list)
    lambda_rlm_decomposition: list[dict[str, Any]] = field(default_factory=list)
    lambda_rlm_strategy_summary: str = ""
    lambda_rlm_error: str = ""
    lambda_rlm_call_count: int = 0
    lambda_rlm_prompt_tokens: int = 0
    lambda_rlm_completion_tokens: int = 0
    lambda_rlm_total_tokens: int = 0
    lambda_rlm_duration_sec: float = 0.0
    lambda_rlm_root_model: str = ""
    lambda_rlm_usage_summary: dict[str, Any] = field(default_factory=dict)
    bootstrap_controller_packet_error: str = ""
    prepared_attempt_meta: dict[str, Any] = field(default_factory=dict)
    prepared_patch_meta: dict[str, Any] = field(default_factory=dict)

    @property
    def strategy_summary_for_packet(self) -> str:
        return self.lambda_rlm_strategy_summary or self.controller_strategy_summary


@dataclass(frozen=True)
class ControllerSidecarWriteResult:
    written_sidecar_names: list[str]
    sidecar_error: str = ""
    sidecar_build_sec: float = 0.0
    sidecar_write_sec: float = 0.0

    @property
    def wrote_sidecars(self) -> bool:
        return bool(self.written_sidecar_names)


def _normalized_prompt_budget(prompt_budget: dict[str, int]) -> dict[str, int]:
    return {str(key): max(0, int(value)) for key, value in prompt_budget.items()}


def _build_sidecars(
    *,
    inputs: ControllerRuntimeInputs,
    syntax_cycle: int,
    syntax_max_cycles: int,
    syntax_last_error_type: str,
    syntax_last_error_msg: str,
    syntax_retry_feedback: str,
    external_decomposition: list[dict[str, Any]] | None = None,
    external_strategy_summary: str = "",
) -> dict[str, dict[str, Any]]:
    return build_adaptive_controller_sidecars(
        case_id=inputs.case_id,
        condition=inputs.condition_value,
        attempt=inputs.attempt,
        root=inputs.workdir,
        attempt_dir=inputs.attempt_dir,
        snapshot=inputs.snapshot,
        failure_signature=inputs.failure_signature,
        evidence_items=inputs.evidence_items,
        new_evidence_ids=inputs.new_evidence_ids,
        reused_evidence_ids=inputs.reused_evidence_ids,
        retry_prompt_mode=inputs.retry_prompt_mode,
        retry_context_mode=inputs.retry_context_mode,
        retry_history_text=inputs.retry_history_text,
        retry_history_turns=inputs.retry_history_turns,
        retry_history_turns_included=inputs.retry_history_turns_included,
        retry_history_truncated=inputs.retry_history_truncated,
        previous_failure_signature=inputs.previous_failure_signature,
        previous_note=inputs.previous_note,
        previous_patch_verifier_rejected=inputs.previous_patch_verifier_rejected,
        previous_patch_verifier_reasons=inputs.previous_patch_verifier_reasons,
        previous_patch_verifier_scope=inputs.previous_patch_verifier_scope,
        previous_timed_out=inputs.previous_timed_out,
        previous_timeout_sec=inputs.previous_timeout_sec,
        helper_action_selected=inputs.helper_action_selected,
        helper_action_executed=inputs.helper_action_executed,
        helper_decision_reason=inputs.helper_decision_reason,
        helper_outcome=inputs.helper_outcome,
        helper_materialized_context=inputs.helper_materialized_context,
        helper_calls_used=inputs.helper_calls_used,
        helper_budget_max_calls=inputs.helper_budget_max_calls,
        no_progress_streak=inputs.no_progress_streak,
        context_compaction_requested=inputs.context_compaction_requested,
        context_compaction_invoked=inputs.context_compaction_invoked,
        context_compaction_success=inputs.context_compaction_success,
        context_compaction_trigger=inputs.context_compaction_trigger,
        prompt_budget=_normalized_prompt_budget(inputs.prompt_budget),
        retry_plan_text=inputs.retry_plan_text,
        syntax_cycle=syntax_cycle,
        syntax_max_cycles=syntax_max_cycles,
        syntax_last_error_type=syntax_last_error_type,
        syntax_last_error_msg=syntax_last_error_msg,
        syntax_retry_feedback=syntax_retry_feedback,
        external_decomposition=external_decomposition,
        external_decomposition_source="lambda_rlm_decomposition" if external_decomposition else "",
        external_strategy_summary=external_strategy_summary,
    )


def prepare_controller_runtime(*, inputs: ControllerRuntimeInputs) -> PreparedControllerRuntime:
    prepare_started = time.perf_counter()
    prepared_attempt_meta: dict[str, Any] = {"adaptive_controller_summarizer_invoked": True}
    prepared_patch_meta: dict[str, Any] = {"adaptive_controller_summarizer_invoked": True}

    controller_strategy_summary = ""
    controller_strategy_sources: list[str] = []
    bootstrap_controller_packet_error = ""
    controller_summarizer_sec = 0.0

    try:
        bootstrap_build_started = time.perf_counter()
        bootstrap_sidecars = _build_sidecars(
            inputs=inputs,
            syntax_cycle=1,
            syntax_max_cycles=1,
            syntax_last_error_type="",
            syntax_last_error_msg="",
            syntax_retry_feedback="",
        )
        bootstrap_sidecar_build_sec = time.perf_counter() - bootstrap_build_started
        prepared_attempt_meta["adaptive_controller_bootstrap_sidecar_build_sec"] = (
            bootstrap_sidecar_build_sec
        )
        prepared_patch_meta["adaptive_controller_bootstrap_sidecar_build_sec"] = (
            bootstrap_sidecar_build_sec
        )
        bootstrap_controller_packet = parse_controller_packet(
            bootstrap_sidecars["llmdebug_controller_packet.json"]
        )
        if bootstrap_controller_packet is None:
            bootstrap_controller_packet_error = "bootstrap controller packet parse failed"
        else:
            summarizer_started = time.perf_counter()
            summary_result = build_controller_strategy_summary(bootstrap_controller_packet)
            controller_summarizer_sec = time.perf_counter() - summarizer_started
            controller_strategy_summary = summary_result.summary
            controller_strategy_sources = list(summary_result.sources)
            prepared_attempt_meta["adaptive_controller_summarizer_summary"] = (
                controller_strategy_summary
            )
            prepared_patch_meta["adaptive_controller_summarizer_summary"] = (
                controller_strategy_summary
            )
            prepared_attempt_meta["adaptive_controller_summarizer_sources"] = (
                controller_strategy_sources
            )
            prepared_patch_meta["adaptive_controller_summarizer_sources"] = (
                controller_strategy_sources
            )
    except Exception as exc:
        bootstrap_controller_packet = None
        bootstrap_controller_packet_error = f"{type(exc).__name__}: {exc}"

    prepared_attempt_meta["adaptive_controller_summarizer_sec"] = controller_summarizer_sec
    prepared_patch_meta["adaptive_controller_summarizer_sec"] = controller_summarizer_sec
    if bootstrap_controller_packet_error:
        prepared_patch_meta["adaptive_controller_summarizer_error"] = (
            bootstrap_controller_packet_error
        )

    lambda_rlm_decomposition: list[dict[str, Any]] = []
    lambda_rlm_strategy_summary = ""
    lambda_rlm_error = ""
    lambda_rlm_call_count = 0
    lambda_rlm_prompt_tokens = 0
    lambda_rlm_completion_tokens = 0
    lambda_rlm_total_tokens = 0
    lambda_rlm_duration_sec = 0.0
    lambda_rlm_root_model = ""
    lambda_rlm_usage_summary: dict[str, Any] = {}

    prepared_attempt_meta["adaptive_controller_lambda_rlm_invoked"] = bool(
        inputs.lambda_rlm_enabled
    )
    prepared_patch_meta["adaptive_controller_lambda_rlm_invoked"] = bool(inputs.lambda_rlm_enabled)
    prepared_attempt_meta["adaptive_controller_lambda_rlm_success"] = False
    prepared_patch_meta["adaptive_controller_lambda_rlm_success"] = False

    if inputs.lambda_rlm_enabled:
        repo_dir = inputs.lambda_rlm_repo_dir or Path()
        python_executable = inputs.lambda_rlm_python or Path()
        prepared_patch_meta["adaptive_controller_lambda_rlm_repo_available"] = repo_dir.is_dir()
        prepared_patch_meta["adaptive_controller_lambda_rlm_python_available"] = (
            python_executable.is_file()
        )
        if not repo_dir.is_dir():
            lambda_rlm_error = f"lambda-RLM repo not found: {repo_dir}"
        elif not python_executable.is_file():
            lambda_rlm_error = f"lambda-RLM python executable not found: {python_executable}"
        elif bootstrap_controller_packet is None:
            lambda_rlm_error = (
                bootstrap_controller_packet_error
                or "lambda-RLM bootstrap controller packet parse failed"
            )
        else:
            try:
                configured_lambda_model = str(inputs.configured_model or "").strip()
                resolve_effective_model = (
                    inputs.resolve_effective_model or resolve_effective_baseline_model
                )
                lambda_model = configured_lambda_model or resolve_effective_model(
                    baseline_name=inputs.patcher_name,
                    configured_model=configured_lambda_model,
                )
                lambda_rlm_invoke = inputs.lambda_rlm_invoke or invoke_lambda_rlm_decomposition
                lambda_result = lambda_rlm_invoke(
                    controller_packet=bootstrap_controller_packet,
                    model=lambda_model,
                    base_url=inputs.configured_api_base_url,
                    api_key=inputs.configured_api_key,
                    repo_dir=repo_dir,
                    python_executable=python_executable,
                    context_window_chars=inputs.lambda_rlm_context_window_chars,
                    timeout_sec=min(
                        max(1, int(inputs.case_timeout_sec or 1)),
                        max(1, int(inputs.lambda_rlm_timeout_sec)),
                    ),
                )
                lambda_rlm_decomposition = list(lambda_result.steps)
                lambda_rlm_strategy_summary = str(lambda_result.summary or "")
                lambda_rlm_duration_sec = float(lambda_result.execution_time_sec)
                lambda_rlm_root_model = str(lambda_result.root_model or "")
                lambda_rlm_usage_summary = dict(lambda_result.usage_summary)
                model_usage_summaries_raw = lambda_result.usage_summary.get("model_usage_summaries")
                if isinstance(model_usage_summaries_raw, dict):
                    for usage_row in model_usage_summaries_raw.values():
                        if isinstance(usage_row, dict):
                            lambda_rlm_call_count += int(usage_row.get("total_calls") or 0)
                lambda_rlm_prompt_tokens = int(
                    lambda_result.usage_summary.get("total_input_tokens") or 0
                )
                lambda_rlm_completion_tokens = int(
                    lambda_result.usage_summary.get("total_output_tokens") or 0
                )
                lambda_rlm_total_tokens = lambda_rlm_prompt_tokens + lambda_rlm_completion_tokens
                success = bool(lambda_rlm_decomposition or lambda_rlm_strategy_summary)
                prepared_attempt_meta["adaptive_controller_lambda_rlm_success"] = success
                prepared_patch_meta["adaptive_controller_lambda_rlm_success"] = success
                prepared_attempt_meta["adaptive_controller_lambda_rlm_duration_sec"] = (
                    lambda_rlm_duration_sec
                )
                prepared_patch_meta["adaptive_controller_lambda_rlm_duration_sec"] = (
                    lambda_rlm_duration_sec
                )
                prepared_attempt_meta["adaptive_controller_lambda_rlm_root_model"] = (
                    lambda_rlm_root_model
                )
                prepared_patch_meta["adaptive_controller_lambda_rlm_root_model"] = (
                    lambda_rlm_root_model
                )
                prepared_attempt_meta["adaptive_controller_lambda_rlm_strategy_summary"] = (
                    lambda_rlm_strategy_summary
                )
                prepared_patch_meta["adaptive_controller_lambda_rlm_strategy_summary"] = (
                    lambda_rlm_strategy_summary
                )
                prepared_attempt_meta["adaptive_controller_lambda_rlm_step_count"] = len(
                    lambda_rlm_decomposition
                )
                prepared_patch_meta["adaptive_controller_lambda_rlm_step_count"] = len(
                    lambda_rlm_decomposition
                )
                prepared_attempt_meta["adaptive_controller_lambda_rlm_usage_summary"] = dict(
                    lambda_rlm_usage_summary
                )
                prepared_patch_meta["adaptive_controller_lambda_rlm_usage_summary"] = dict(
                    lambda_rlm_usage_summary
                )
                prepared_attempt_meta["adaptive_controller_lambda_rlm_call_count"] = (
                    lambda_rlm_call_count
                )
                prepared_patch_meta["adaptive_controller_lambda_rlm_call_count"] = (
                    lambda_rlm_call_count
                )
                prepared_patch_meta["adaptive_controller_lambda_rlm_prompt_tokens"] = (
                    lambda_rlm_prompt_tokens
                )
                prepared_patch_meta["adaptive_controller_lambda_rlm_completion_tokens"] = (
                    lambda_rlm_completion_tokens
                )
                prepared_patch_meta["adaptive_controller_lambda_rlm_total_tokens"] = (
                    lambda_rlm_total_tokens
                )
                if not success:
                    lambda_rlm_error = "lambda-RLM returned no usable decomposition"
            except Exception as exc:
                lambda_rlm_error = f"{type(exc).__name__}: {exc}"
        if lambda_rlm_error:
            prepared_attempt_meta["adaptive_controller_lambda_rlm_error"] = lambda_rlm_error
            prepared_patch_meta["adaptive_controller_lambda_rlm_error"] = lambda_rlm_error

    prepare_sec = time.perf_counter() - prepare_started
    prepared_attempt_meta["adaptive_controller_prepare_sec"] = prepare_sec
    prepared_patch_meta["adaptive_controller_prepare_sec"] = prepare_sec

    return PreparedControllerRuntime(
        inputs=inputs,
        controller_strategy_summary=controller_strategy_summary,
        controller_strategy_sources=controller_strategy_sources,
        lambda_rlm_decomposition=lambda_rlm_decomposition,
        lambda_rlm_strategy_summary=lambda_rlm_strategy_summary,
        lambda_rlm_error=lambda_rlm_error,
        lambda_rlm_call_count=lambda_rlm_call_count,
        lambda_rlm_prompt_tokens=lambda_rlm_prompt_tokens,
        lambda_rlm_completion_tokens=lambda_rlm_completion_tokens,
        lambda_rlm_total_tokens=lambda_rlm_total_tokens,
        lambda_rlm_duration_sec=lambda_rlm_duration_sec,
        lambda_rlm_root_model=lambda_rlm_root_model,
        lambda_rlm_usage_summary=lambda_rlm_usage_summary,
        bootstrap_controller_packet_error=bootstrap_controller_packet_error,
        prepared_attempt_meta=prepared_attempt_meta,
        prepared_patch_meta=prepared_patch_meta,
    )


def write_controller_sidecars_for_cycle(
    *,
    prepared: PreparedControllerRuntime,
    syntax_cycle: int,
    syntax_max_cycles: int,
    syntax_last_error_type: str,
    syntax_last_error_msg: str,
    syntax_retry_feedback: str,
    context_dir: Path,
) -> ControllerSidecarWriteResult:
    try:
        sidecar_build_started = time.perf_counter()
        sidecars = _build_sidecars(
            inputs=prepared.inputs,
            syntax_cycle=syntax_cycle,
            syntax_max_cycles=syntax_max_cycles,
            syntax_last_error_type=syntax_last_error_type,
            syntax_last_error_msg=syntax_last_error_msg,
            syntax_retry_feedback=syntax_retry_feedback,
            external_decomposition=prepared.lambda_rlm_decomposition,
            external_strategy_summary=prepared.strategy_summary_for_packet,
        )
        sidecar_build_sec = time.perf_counter() - sidecar_build_started
        sidecar_write_started = time.perf_counter()
        written_sidecars = write_adaptive_controller_sidecars(
            context_dir=context_dir,
            sidecars=sidecars,
        )
        sidecar_write_sec = time.perf_counter() - sidecar_write_started
    except Exception as exc:
        return ControllerSidecarWriteResult(
            written_sidecar_names=[],
            sidecar_error=f"{type(exc).__name__}: {exc}",
            sidecar_build_sec=0.0,
            sidecar_write_sec=0.0,
        )
    return ControllerSidecarWriteResult(
        written_sidecar_names=[path.name for path in written_sidecars],
        sidecar_build_sec=sidecar_build_sec,
        sidecar_write_sec=sidecar_write_sec,
    )


__all__ = [
    "ControllerRuntimeInputs",
    "ControllerSidecarWriteResult",
    "PreparedControllerRuntime",
    "prepare_controller_runtime",
    "write_controller_sidecars_for_cycle",
]
