"""Native Stage-B prompt flow helpers for the eval harness."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .eval_prompts import _apply_prompt_budget, _build_stage_b_prompt_sections
from .eval_types import (
    OPENAI_CONTEXT_WINDOW_CHARS_PER_TOKEN,
    OPENAI_CONTEXT_WINDOW_RESERVE_TOKENS,
    PromptBudgetResult,
)


@dataclass(frozen=True)
class StageBPromptInputs:
    """Inputs needed to assemble a Stage-B repair prompt.

    ``retry_prompt_mode`` controls retry-specific instructions. ``retry_context_mode``
    controls how much prior-attempt context is exposed. ``known_evidence_refs``
    and ``new_evidence_ids`` keep retry prompts aligned with the controller's
    evidence accounting.
    """

    evidence: dict[str, Any]
    snapshot: dict[str, Any] | None
    stage_a_context: str
    stage_a_evidence_index: str
    compacted_context: str
    include_snapshot: bool
    include_pytest_output: bool
    include_stage_a_context: bool
    include_project_files: bool
    retry_packet: str
    retry_history: str
    retry_prompt_mode: str
    attempt: int
    known_evidence_refs: list[str]
    new_evidence_ids: list[str]
    snapshot_prompt_text: str = ""
    snapshot_lite_context: str = ""
    prompt_retry_reflection: bool = False
    prompt_adaptive_stage_b_cot: bool = False
    prompt_repair_method_scaffold: bool = False
    prompt_order_mode: str = "current"
    enable_tep: bool = False
    stagnation_streak: int = 0
    tep_patch_actor_overrides: dict[str, str] | None = None
    tep_planner_actor_overrides: dict[str, str] | None = None


@dataclass(frozen=True)
class StageBPromptBuildResult:
    """Prompt text plus the budget decisions used to produce it."""

    prompt: str
    prompt_budget_result: PromptBudgetResult
    include_stage_a_evidence_index: bool
    stage_a_index_reason: str


@dataclass(frozen=True)
class RetryPlanResult:
    """Combined result of optional retry planning followed by patch prompting.

    When retry planning is enabled on later attempts, this records both the
    planning prompt outcome and whether the returned retry plan was actually
    injected into the final patch prompt after budgeting.
    """

    prompt: str
    prompt_budget_result: PromptBudgetResult
    include_stage_a_evidence_index: bool
    stage_a_index_reason: str
    retry_plan_text: str = ""
    retry_plan_raw_response: str | None = None
    retry_plan_prompt_text: str | None = None
    retry_plan_meta: dict[str, Any] = field(default_factory=dict)
    retry_plan_duration_sec: float = 0.0
    retry_plan_invoked: bool = False
    retry_plan_succeeded: bool = False
    retry_plan_prompt_injected: bool = False
    retry_plan_error: str = ""
    tep_equilibrium_iterations: int = 0
    tep_equilibrium_converged: bool = False


def openai_preflight_total_char_cap(
    *,
    context_window_tokens: int,
    max_tokens: int,
    reserve_tokens: int = OPENAI_CONTEXT_WINDOW_RESERVE_TOKENS,
) -> int:
    """Convert a token budget into a conservative character cap for prompt assembly."""
    normalized_window = max(0, int(context_window_tokens))
    if normalized_window <= 0:
        return 0
    usable_tokens = normalized_window - max(0, int(max_tokens)) - max(0, int(reserve_tokens))
    if usable_tokens <= 0:
        return 1
    return max(1, usable_tokens * OPENAI_CONTEXT_WINDOW_CHARS_PER_TOKEN)


def stage_a_index_fallback_needed(
    *,
    budget_result: PromptBudgetResult,
    include_index: bool,
    stage_a_evidence_index_text: str,
    compacted_context: str,
) -> bool:
    """Return whether the Stage-A evidence index should be force-included.

    The fallback only triggers when the index was initially omitted and prompt
    budgeting truncated or dropped the Stage-A context or compacted context.
    """
    if not stage_a_evidence_index_text or include_index:
        return False
    if budget_result.section_status.get("stage_a_context") in {"truncated", "dropped"}:
        return True
    return bool(
        compacted_context
        and budget_result.section_status.get("compacted_context") in {"truncated", "dropped"}
    )


def build_stage_b_prompt_sections(
    *,
    prompt_inputs: StageBPromptInputs,
    include_stage_a_evidence_index: bool,
    retry_repair_plan: str = "",
    planning_mode: bool = False,
) -> list[dict[str, Any]]:
    """Build raw Stage-B prompt sections before prompt-budget trimming."""
    return _build_stage_b_prompt_sections(
        evidence=prompt_inputs.evidence,
        snapshot=prompt_inputs.snapshot,
        snapshot_prompt_text=prompt_inputs.snapshot_prompt_text,
        stage_a_context=prompt_inputs.stage_a_context,
        stage_a_evidence_index=prompt_inputs.stage_a_evidence_index,
        include_stage_a_evidence_index=include_stage_a_evidence_index,
        compacted_context=prompt_inputs.compacted_context,
        include_snapshot=prompt_inputs.include_snapshot,
        include_pytest_output=prompt_inputs.include_pytest_output,
        include_stage_a_context=prompt_inputs.include_stage_a_context,
        include_project_files=prompt_inputs.include_project_files,
        retry_packet=prompt_inputs.retry_packet,
        retry_history=prompt_inputs.retry_history,
        retry_prompt_mode=prompt_inputs.retry_prompt_mode,
        attempt=prompt_inputs.attempt,
        known_evidence_refs=prompt_inputs.known_evidence_refs,
        new_evidence_ids=prompt_inputs.new_evidence_ids,
        snapshot_lite_context=prompt_inputs.snapshot_lite_context,
        prompt_retry_reflection=prompt_inputs.prompt_retry_reflection,
        prompt_adaptive_stage_b_cot=prompt_inputs.prompt_adaptive_stage_b_cot,
        prompt_repair_method_scaffold=prompt_inputs.prompt_repair_method_scaffold,
        retry_repair_plan=retry_repair_plan,
        planning_mode=planning_mode,
        prompt_order_mode=prompt_inputs.prompt_order_mode,
        enable_tep=prompt_inputs.enable_tep,
        stagnation_streak=prompt_inputs.stagnation_streak,
        tep_actor_overrides=(
            prompt_inputs.tep_planner_actor_overrides
            if planning_mode
            else prompt_inputs.tep_patch_actor_overrides
        ),
    )


def build_stage_b_prompt_with_budget(
    *,
    prompt_inputs: StageBPromptInputs,
    attempt_prompt_budget: dict[str, int],
    is_delta_retry: bool,
    include_stage_a_evidence_index: bool,
    stage_a_index_reason: str,
    retry_repair_plan: str = "",
    planning_mode: bool = False,
) -> StageBPromptBuildResult:
    """Assemble a Stage-B prompt and rerun with index fallback when needed."""
    prompt_sections = build_stage_b_prompt_sections(
        prompt_inputs=prompt_inputs,
        include_stage_a_evidence_index=include_stage_a_evidence_index,
        retry_repair_plan=retry_repair_plan,
        planning_mode=planning_mode,
    )
    prompt_budget_result = _apply_prompt_budget(
        sections=prompt_sections,
        budget=attempt_prompt_budget,
        is_delta_retry=is_delta_retry,
    )
    include_index = include_stage_a_evidence_index
    index_reason = stage_a_index_reason
    if stage_a_index_fallback_needed(
        budget_result=prompt_budget_result,
        include_index=include_index,
        stage_a_evidence_index_text=prompt_inputs.stage_a_evidence_index,
        compacted_context=prompt_inputs.compacted_context,
    ):
        prompt_sections = build_stage_b_prompt_sections(
            prompt_inputs=prompt_inputs,
            include_stage_a_evidence_index=True,
            retry_repair_plan=retry_repair_plan,
            planning_mode=planning_mode,
        )
        prompt_budget_result = _apply_prompt_budget(
            sections=prompt_sections,
            budget=attempt_prompt_budget,
            is_delta_retry=is_delta_retry,
        )
        include_index = True
        index_reason = "context_truncated_fallback"
    return StageBPromptBuildResult(
        prompt=prompt_budget_result.prompt,
        prompt_budget_result=prompt_budget_result,
        include_stage_a_evidence_index=include_index,
        stage_a_index_reason=index_reason,
    )


def maybe_run_retry_plan_then_patch(
    *,
    enabled: bool,
    plan_on_first_attempt: bool,
    attempt: int,
    prompt_inputs: StageBPromptInputs,
    attempt_prompt_budget: dict[str, int],
    is_delta_retry: bool,
    include_stage_a_evidence_index: bool,
    stage_a_index_reason: str,
    run_plan_patch: Callable[[str], tuple[str | None, str | None, dict[str, Any]]],
    duration_sec: float = 0.0,
    tep_equilibrium_check: bool = False,
    tep_equilibrium_max_iterations: int = 2,
    tep_equilibrium_similarity_threshold: float = 0.85,
) -> RetryPlanResult:
    """Optionally request a retry plan before producing the final patch prompt.

    By default, attempts ``<= 1`` skip planning entirely. Callers may opt into
    first-attempt planning via ``plan_on_first_attempt``. When planning runs,
    the helper first builds a planning-mode prompt, calls ``run_plan_patch``,
    then injects the returned plan into the final prompt if budgeting keeps the
    section.
    """
    base_prompt_result = build_stage_b_prompt_with_budget(
        prompt_inputs=prompt_inputs,
        attempt_prompt_budget=attempt_prompt_budget,
        is_delta_retry=is_delta_retry,
        include_stage_a_evidence_index=include_stage_a_evidence_index,
        stage_a_index_reason=stage_a_index_reason,
    )
    if not enabled or (attempt <= 1 and not plan_on_first_attempt):
        return RetryPlanResult(
            prompt=base_prompt_result.prompt,
            prompt_budget_result=base_prompt_result.prompt_budget_result,
            include_stage_a_evidence_index=base_prompt_result.include_stage_a_evidence_index,
            stage_a_index_reason=base_prompt_result.stage_a_index_reason,
        )

    planning_prompt_result = build_stage_b_prompt_with_budget(
        prompt_inputs=prompt_inputs,
        attempt_prompt_budget=attempt_prompt_budget,
        is_delta_retry=is_delta_retry,
        include_stage_a_evidence_index=include_stage_a_evidence_index,
        stage_a_index_reason=stage_a_index_reason,
        planning_mode=True,
    )
    raw_retry_plan_text, retry_plan_raw_response, retry_plan_meta = run_plan_patch(
        planning_prompt_result.prompt
    )
    retry_plan_text = str(raw_retry_plan_text or "").strip()
    retry_plan_succeeded = bool(retry_plan_text)

    # TEP equilibrium check: re-run planning until plan stabilises.
    tep_eq_iterations = 1
    tep_eq_converged = False
    if retry_plan_succeeded and tep_equilibrium_check and tep_equilibrium_max_iterations > 1:
        from evals.eval_attempt_flow import _tep_jaccard_similarity

        for _eq_round in range(tep_equilibrium_max_iterations - 1):
            prev_plan = retry_plan_text
            new_raw, _, _ = run_plan_patch(planning_prompt_result.prompt)
            new_plan = str(new_raw or "").strip()
            tep_eq_iterations += 1
            if not new_plan:
                break
            sim = _tep_jaccard_similarity(prev_plan, new_plan)
            retry_plan_text = new_plan
            if sim >= tep_equilibrium_similarity_threshold:
                tep_eq_converged = True
                break

    prompt_result = base_prompt_result
    retry_plan_prompt_injected = False
    if retry_plan_succeeded:
        prompt_result = build_stage_b_prompt_with_budget(
            prompt_inputs=prompt_inputs,
            attempt_prompt_budget=attempt_prompt_budget,
            is_delta_retry=is_delta_retry,
            include_stage_a_evidence_index=planning_prompt_result.include_stage_a_evidence_index,
            stage_a_index_reason=planning_prompt_result.stage_a_index_reason,
            retry_repair_plan=retry_plan_text,
        )
        retry_plan_prompt_injected = (
            prompt_result.prompt_budget_result.section_status.get("retry_repair_plan") != "dropped"
        )
    retry_plan_error = str(retry_plan_meta.get("error") or "")
    return RetryPlanResult(
        prompt=prompt_result.prompt,
        prompt_budget_result=prompt_result.prompt_budget_result,
        include_stage_a_evidence_index=prompt_result.include_stage_a_evidence_index,
        stage_a_index_reason=prompt_result.stage_a_index_reason,
        retry_plan_text=retry_plan_text,
        retry_plan_raw_response=retry_plan_raw_response,
        retry_plan_prompt_text=planning_prompt_result.prompt,
        retry_plan_meta=dict(retry_plan_meta),
        retry_plan_duration_sec=max(0.0, float(duration_sec)),
        retry_plan_invoked=True,
        retry_plan_succeeded=retry_plan_succeeded,
        retry_plan_prompt_injected=retry_plan_prompt_injected,
        retry_plan_error=retry_plan_error,
        tep_equilibrium_iterations=tep_eq_iterations,
        tep_equilibrium_converged=tep_eq_converged,
    )


__all__ = [
    "RetryPlanResult",
    "StageBPromptBuildResult",
    "StageBPromptInputs",
    "build_stage_b_prompt_sections",
    "build_stage_b_prompt_with_budget",
    "maybe_run_retry_plan_then_patch",
    "openai_preflight_total_char_cap",
    "stage_a_index_fallback_needed",
]
