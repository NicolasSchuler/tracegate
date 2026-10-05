from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest

from evals import native_openai_attempt_flow as flow
from evals.eval_types import PromptBudgetResult
from evals.native_stage_b_flow import RetryPlanResult, StageBPromptBuildResult, StageBPromptInputs


def _prompt_inputs() -> StageBPromptInputs:
    return StageBPromptInputs(
        evidence={"case_id": "case-1"},
        snapshot={"frames": [{"file": "buggy.py", "line": 4}]},
        stage_a_context="requested context",
        stage_a_evidence_index="[e1] buggy.py:4",
        compacted_context="",
        include_snapshot=True,
        include_pytest_output=True,
        include_stage_a_context=True,
        include_project_files=True,
        retry_packet="retry packet",
        retry_history="retry history",
        retry_prompt_mode="history",
        attempt=2,
        known_evidence_refs=["e0"],
        new_evidence_ids=["e1"],
        snapshot_lite_context="snapshot lite",
        prompt_retry_reflection=True,
        prompt_adaptive_stage_b_cot=False,
        prompt_repair_method_scaffold=True,
    )


def _request(
    tmp_path: Path, *, prompt_inputs: StageBPromptInputs
) -> flow.NativeOpenAIAttemptPromptRequest:
    return flow.NativeOpenAIAttemptPromptRequest(
        prompt_inputs=prompt_inputs,
        attempt_prompt_budget={"total_chars": 400},
        is_delta_retry=False,
        include_stage_a_evidence_index=False,
        stage_a_index_reason="redundant_skipped",
        project_files_context="buggy.py\nhelper.py",
        current_stdout="stdout",
        current_stderr="stderr",
        context_compaction_requested=True,
        context_compaction_request_reason="policy_action",
        policy_compact_then_patch_requested=True,
        context_compaction_mode="auto",
        context_compaction_scope={"snapshot", "pytest", "files", "stage_a_context"},
        supports_context_compaction=True,
        context_compaction_calls_used=0,
        context_compaction_max_calls=1,
        context_compaction_max_input_chars=500,
        context_compaction_target_chars=120,
        context_compaction_fn=lambda **_: SimpleNamespace(
            compacted_text="compacted summary",
            meta={"compact_context_duration_sec": 0.25},
        ),
        patcher_for_attempt=SimpleNamespace(response_mode="edits"),
        run_openai_operation_with_transport_recovery=lambda *, invoke, **__: invoke(),
        classify_transport_recovery_failure_kind=lambda _: "",
        attempt_context_protocol="staged",
        before=SimpleNamespace(stdout="stdout", stderr="stderr"),
        cumulative_runtime_sec=0.0,
        attempt_dir=tmp_path / "attempt",
        workdir=tmp_path / "workdir",
        retry_plan_enabled=True,
        plan_on_first_attempt=False,
        run_plan_patch=lambda prompt: (f"plan for {prompt}", prompt, {"plan_meta": True}),
    )


def test_assemble_native_openai_attempt_prompt_compacts_context_and_reuses_updated_prompt_inputs(
    tmp_path: Path,
    monkeypatch,
) -> None:
    prompt_inputs_seen: list[StageBPromptInputs] = []
    retry_plan_inputs: dict[str, StageBPromptInputs] = {}

    def _fake_build_stage_b_prompt_with_budget(  # type: ignore[no-untyped-def]
        *,
        prompt_inputs,
        attempt_prompt_budget,
        is_delta_retry,
        include_stage_a_evidence_index,
        stage_a_index_reason,
        retry_repair_plan="",
        planning_mode=False,
    ):
        del attempt_prompt_budget, is_delta_retry, retry_repair_plan, planning_mode
        prompt_inputs_seen.append(prompt_inputs)
        label = "compacted" if prompt_inputs.compacted_context else "base"
        return StageBPromptBuildResult(
            prompt=f"{label}-prompt",
            prompt_budget_result=PromptBudgetResult(
                prompt=f"{label}-prompt",
                used_chars=24,
                overflow_chars=12 if label == "base" else 0,
                dropped_sections=[],
                section_status={
                    "snapshot": "truncated" if label == "base" else "kept",
                    "stage_a_context": "truncated" if label == "base" else "kept",
                    "project_files": "truncated" if label == "base" else "kept",
                },
            ),
            include_stage_a_evidence_index=include_stage_a_evidence_index,
            stage_a_index_reason=stage_a_index_reason,
        )

    def _fake_retry_plan_then_patch(  # type: ignore[no-untyped-def]
        *,
        enabled,
        plan_on_first_attempt,
        attempt,
        prompt_inputs,
        attempt_prompt_budget,
        is_delta_retry,
        include_stage_a_evidence_index,
        stage_a_index_reason,
        run_plan_patch,
        tep_equilibrium_check,
        tep_equilibrium_max_iterations,
        tep_equilibrium_similarity_threshold,
    ):
        del enabled, plan_on_first_attempt, attempt, attempt_prompt_budget, is_delta_retry
        del run_plan_patch
        retry_plan_inputs["prompt_inputs"] = prompt_inputs
        return RetryPlanResult(
            prompt="final-prompt",
            prompt_budget_result=PromptBudgetResult(
                prompt="final-prompt",
                used_chars=30,
                overflow_chars=0,
                dropped_sections=[],
                section_status={"retry_repair_plan": "kept"},
            ),
            include_stage_a_evidence_index=include_stage_a_evidence_index,
            stage_a_index_reason=stage_a_index_reason,
            retry_plan_text="retry plan",
            retry_plan_raw_response="retry plan raw",
            retry_plan_prompt_text="retry prompt",
            retry_plan_meta={"plan_meta": True},
            retry_plan_invoked=True,
            retry_plan_succeeded=True,
            retry_plan_prompt_injected=True,
        )

    monkeypatch.setattr(
        flow, "build_stage_b_prompt_with_budget", _fake_build_stage_b_prompt_with_budget
    )
    monkeypatch.setattr(flow, "maybe_run_retry_plan_then_patch", _fake_retry_plan_then_patch)
    monkeypatch.setattr(
        flow,
        "_build_context_compaction_bundle",
        lambda **_: SimpleNamespace(
            text="snapshot\nstdout\ncontext",
            source_sections=["snapshot", "pytest_output", "stage_a_context", "project_files"],
            source_chars={"snapshot": 8, "pytest_output": 6, "stage_a_context": 7},
            replace_snapshot=True,
            replace_pytest_output=False,
            replace_stage_a_context=True,
            replace_project_files=True,
        ),
    )
    monkeypatch.setattr(flow, "_should_invoke_context_compaction", lambda **_: (True, "policy"))
    monkeypatch.setattr(flow, "render_context_compaction_prompt", lambda **_: "compact prompt")

    result = flow.assemble_native_openai_attempt_prompt(
        request=_request(tmp_path, prompt_inputs=_prompt_inputs())
    )

    assert len(prompt_inputs_seen) == 2
    assert result.prompt == "final-prompt"
    assert result.context_compaction.invoked is True
    assert result.context_compaction.success is True
    assert result.context_compaction.trigger == "policy"
    assert result.context_compaction.replaced_sections == [
        "snapshot",
        "stage_a_context",
        "project_files",
    ]
    assert result.context_compaction.calls_used == 1
    assert result.prompt_inputs.compacted_context == "compacted summary"
    assert result.prompt_inputs.include_snapshot is False
    assert result.prompt_inputs.include_pytest_output is True
    assert result.prompt_inputs.include_stage_a_context is False
    assert result.prompt_inputs.include_project_files is False
    assert retry_plan_inputs["prompt_inputs"] == result.prompt_inputs


def test_assemble_native_openai_attempt_prompt_forwards_tep_equilibrium_settings(
    tmp_path: Path,
    monkeypatch,
) -> None:
    captured_kwargs: dict[str, object] = {}

    def _fake_retry_plan_then_patch(**kwargs):  # type: ignore[no-untyped-def]
        captured_kwargs.update(kwargs)
        return RetryPlanResult(
            prompt="final-prompt",
            prompt_budget_result=PromptBudgetResult(
                prompt="final-prompt",
                used_chars=30,
                overflow_chars=0,
                dropped_sections=[],
                section_status={"retry_repair_plan": "kept"},
            ),
            include_stage_a_evidence_index=kwargs["include_stage_a_evidence_index"],
            stage_a_index_reason=kwargs["stage_a_index_reason"],
            retry_plan_text="retry plan",
            retry_plan_invoked=True,
            retry_plan_succeeded=True,
            retry_plan_prompt_injected=True,
            tep_equilibrium_iterations=2,
            tep_equilibrium_converged=True,
        )

    monkeypatch.setattr(flow, "maybe_run_retry_plan_then_patch", _fake_retry_plan_then_patch)

    request = _request(tmp_path, prompt_inputs=_prompt_inputs())
    request = flow.NativeOpenAIAttemptPromptRequest(
        **{
            **{f.name: getattr(request, f.name) for f in request.__dataclass_fields__.values()},
            "tep_equilibrium_check": True,
            "tep_equilibrium_max_iterations": 4,
            "tep_equilibrium_similarity_threshold": 0.92,
        }
    )

    result = flow.assemble_native_openai_attempt_prompt(request=request)

    assert captured_kwargs["tep_equilibrium_check"] is True
    assert captured_kwargs["tep_equilibrium_max_iterations"] == 4
    assert captured_kwargs["tep_equilibrium_similarity_threshold"] == pytest.approx(0.92)
    assert result.retry_plan.tep_equilibrium_iterations == 2
    assert result.retry_plan.tep_equilibrium_converged is True


@dataclass(frozen=True)
class _DummyPatcher:
    response_mode: str


def test_maybe_autoswitch_openai_response_mode_updates_metadata() -> None:
    attempt_metadata: dict[str, object] = {}
    patch_meta: dict[str, object] = {}

    next_patcher, switched = flow.maybe_autoswitch_openai_response_mode(
        patcher_for_attempt=_DummyPatcher(response_mode="edits"),
        policy="retry_to_diff",
        reason="syntax_preflight_failed",
        syntax_cycle=2,
        patch_meta=patch_meta,
        attempt_metadata=attempt_metadata,
    )

    assert switched is True
    assert next_patcher.response_mode == "diff"
    assert patch_meta["openai_response_mode_autoswitched"] is True
    assert patch_meta["openai_response_mode_autoswitch_from"] == "edits"
    assert patch_meta["openai_response_mode_autoswitch_to"] == "diff"
    assert patch_meta["openai_response_mode_autoswitch_cycle"] == 2
    assert patch_meta["openai_response_mode_autoswitch_reason"] == "syntax_preflight_failed"
    assert attempt_metadata["openai_response_mode_autoswitched"] is True


# ---------------------------------------------------------------------------
# maybe_autoswitch: policy != "retry_to_diff" → no switch (line 332)
# ---------------------------------------------------------------------------


def test_autoswitch_skips_when_policy_not_retry_to_diff() -> None:
    patcher = _DummyPatcher(response_mode="edits")
    result, switched = flow.maybe_autoswitch_openai_response_mode(
        patcher_for_attempt=patcher,
        policy="off",
        reason="syntax_preflight_failed",
        syntax_cycle=1,
        patch_meta={},
        attempt_metadata={},
    )
    assert switched is False
    assert result is patcher


# ---------------------------------------------------------------------------
# maybe_autoswitch: current_mode != "edits" → no switch (line 335)
# ---------------------------------------------------------------------------


def test_autoswitch_skips_when_mode_not_edits() -> None:
    patcher = _DummyPatcher(response_mode="diff")
    result, switched = flow.maybe_autoswitch_openai_response_mode(
        patcher_for_attempt=patcher,
        policy="retry_to_diff",
        reason="syntax_preflight_failed",
        syntax_cycle=1,
        patch_meta={},
        attempt_metadata={},
    )
    assert switched is False
    assert result is patcher


# ---------------------------------------------------------------------------
# maybe_autoswitch: replace() raises → error captured (lines 338-340)
# ---------------------------------------------------------------------------


def test_autoswitch_captures_replace_error() -> None:
    """When replace() raises, error is stored in patch_meta and no switch occurs."""

    # A plain object that doesn't support dataclasses.replace
    class _BadPatcher:
        response_mode = "edits"

    patch_meta: dict[str, object] = {}
    _result, switched = flow.maybe_autoswitch_openai_response_mode(
        patcher_for_attempt=_BadPatcher(),
        policy="retry_to_diff",
        reason="syntax_preflight_failed",
        syntax_cycle=1,
        patch_meta=patch_meta,
        attempt_metadata={},
    )
    assert switched is False
    assert "openai_response_mode_autoswitch_error" in patch_meta


# ---------------------------------------------------------------------------
# assemble: compaction raises exception (lines 208-210)
# ---------------------------------------------------------------------------


def test_assemble_handles_compaction_exception(tmp_path: Path, monkeypatch) -> None:
    """When run_openai_operation_with_transport_recovery raises, error is captured."""
    build_calls: list[StageBPromptInputs] = []

    def _fake_build(  # type: ignore[no-untyped-def]
        *,
        prompt_inputs,
        attempt_prompt_budget,
        is_delta_retry,
        include_stage_a_evidence_index,
        stage_a_index_reason,
        retry_repair_plan="",
        planning_mode=False,
    ):
        del attempt_prompt_budget, is_delta_retry, retry_repair_plan, planning_mode
        build_calls.append(prompt_inputs)
        return StageBPromptBuildResult(
            prompt="base-prompt",
            prompt_budget_result=PromptBudgetResult(
                prompt="base-prompt",
                used_chars=20,
                overflow_chars=10,
                dropped_sections=[],
                section_status={"snapshot": "truncated"},
            ),
            include_stage_a_evidence_index=include_stage_a_evidence_index,
            stage_a_index_reason=stage_a_index_reason,
        )

    def _fake_retry(  # type: ignore[no-untyped-def]
        *,
        enabled,
        plan_on_first_attempt,
        attempt,
        prompt_inputs,
        attempt_prompt_budget,
        is_delta_retry,
        include_stage_a_evidence_index,
        stage_a_index_reason,
        run_plan_patch,
        tep_equilibrium_check,
        tep_equilibrium_max_iterations,
        tep_equilibrium_similarity_threshold,
    ):
        del enabled, plan_on_first_attempt, attempt, prompt_inputs
        del attempt_prompt_budget, is_delta_retry, run_plan_patch
        return RetryPlanResult(
            prompt="final",
            prompt_budget_result=PromptBudgetResult(
                prompt="final",
                used_chars=20,
                overflow_chars=0,
                dropped_sections=[],
                section_status={},
            ),
            include_stage_a_evidence_index=include_stage_a_evidence_index,
            stage_a_index_reason=stage_a_index_reason,
        )

    monkeypatch.setattr(flow, "build_stage_b_prompt_with_budget", _fake_build)
    monkeypatch.setattr(flow, "maybe_run_retry_plan_then_patch", _fake_retry)
    monkeypatch.setattr(
        flow,
        "_build_context_compaction_bundle",
        lambda **_: SimpleNamespace(
            text="text",
            source_sections=["snapshot"],
            source_chars={"snapshot": 4},
            replace_snapshot=True,
            replace_pytest_output=False,
            replace_stage_a_context=False,
            replace_project_files=False,
        ),
    )
    monkeypatch.setattr(flow, "_should_invoke_context_compaction", lambda **_: (True, "auto"))
    monkeypatch.setattr(flow, "render_context_compaction_prompt", lambda **_: "prompt text")

    def _raise_transport(**_: object) -> None:
        raise RuntimeError("transport failure")

    req = _request(tmp_path, prompt_inputs=_prompt_inputs())
    req = flow.NativeOpenAIAttemptPromptRequest(
        **{
            **{f.name: getattr(req, f.name) for f in req.__dataclass_fields__.values()},
            "run_openai_operation_with_transport_recovery": _raise_transport,
        }
    )
    result = flow.assemble_native_openai_attempt_prompt(request=req)

    assert result.context_compaction.invoked is True
    assert result.context_compaction.success is False
    assert "RuntimeError" in result.context_compaction.error
    # Only called once because compaction failed → no re-build
    assert len(build_calls) == 1


# ---------------------------------------------------------------------------
# assemble: compaction returns None → "compact_context_unavailable" (lines 250-251)
# ---------------------------------------------------------------------------


def test_assemble_compaction_returns_none(tmp_path: Path, monkeypatch) -> None:
    """When compaction returns None, error is 'compact_context_unavailable'."""

    def _fake_build(  # type: ignore[no-untyped-def]
        *,
        prompt_inputs,
        attempt_prompt_budget,
        is_delta_retry,
        include_stage_a_evidence_index,
        stage_a_index_reason,
        retry_repair_plan="",
        planning_mode=False,
    ):
        del attempt_prompt_budget, is_delta_retry, retry_repair_plan, planning_mode
        return StageBPromptBuildResult(
            prompt="base-prompt",
            prompt_budget_result=PromptBudgetResult(
                prompt="base-prompt",
                used_chars=20,
                overflow_chars=10,
                dropped_sections=[],
                section_status={"snapshot": "truncated"},
            ),
            include_stage_a_evidence_index=include_stage_a_evidence_index,
            stage_a_index_reason=stage_a_index_reason,
        )

    def _fake_retry(  # type: ignore[no-untyped-def]
        *,
        enabled,
        plan_on_first_attempt,
        attempt,
        prompt_inputs,
        attempt_prompt_budget,
        is_delta_retry,
        include_stage_a_evidence_index,
        stage_a_index_reason,
        run_plan_patch,
        tep_equilibrium_check,
        tep_equilibrium_max_iterations,
        tep_equilibrium_similarity_threshold,
    ):
        del enabled, plan_on_first_attempt, attempt, prompt_inputs
        del attempt_prompt_budget, is_delta_retry, run_plan_patch
        return RetryPlanResult(
            prompt="final",
            prompt_budget_result=PromptBudgetResult(
                prompt="final",
                used_chars=20,
                overflow_chars=0,
                dropped_sections=[],
                section_status={},
            ),
            include_stage_a_evidence_index=include_stage_a_evidence_index,
            stage_a_index_reason=stage_a_index_reason,
        )

    monkeypatch.setattr(flow, "build_stage_b_prompt_with_budget", _fake_build)
    monkeypatch.setattr(flow, "maybe_run_retry_plan_then_patch", _fake_retry)
    monkeypatch.setattr(
        flow,
        "_build_context_compaction_bundle",
        lambda **_: SimpleNamespace(
            text="text",
            source_sections=["snapshot"],
            source_chars={"snapshot": 4},
            replace_snapshot=True,
            replace_pytest_output=False,
            replace_stage_a_context=False,
            replace_project_files=False,
        ),
    )
    monkeypatch.setattr(flow, "_should_invoke_context_compaction", lambda **_: (True, "auto"))
    monkeypatch.setattr(flow, "render_context_compaction_prompt", lambda **_: "prompt text")

    req = _request(tmp_path, prompt_inputs=_prompt_inputs())
    req = flow.NativeOpenAIAttemptPromptRequest(
        **{
            **{f.name: getattr(req, f.name) for f in req.__dataclass_fields__.values()},
            "run_openai_operation_with_transport_recovery": lambda *, invoke, **__: None,
        }
    )
    result = flow.assemble_native_openai_attempt_prompt(request=req)

    assert result.context_compaction.invoked is True
    assert result.context_compaction.success is False
    assert result.context_compaction.error == "compact_context_unavailable"


# ---------------------------------------------------------------------------
# assemble: compaction result with empty text → error from meta (lines 248-249)
# ---------------------------------------------------------------------------


def test_assemble_compaction_empty_text_uses_meta_error(tmp_path: Path, monkeypatch) -> None:
    """When compacted_text is empty and meta has an error, that error is used."""

    def _fake_build(  # type: ignore[no-untyped-def]
        *,
        prompt_inputs,
        attempt_prompt_budget,
        is_delta_retry,
        include_stage_a_evidence_index,
        stage_a_index_reason,
        retry_repair_plan="",
        planning_mode=False,
    ):
        del attempt_prompt_budget, is_delta_retry, retry_repair_plan, planning_mode
        return StageBPromptBuildResult(
            prompt="base-prompt",
            prompt_budget_result=PromptBudgetResult(
                prompt="base-prompt",
                used_chars=20,
                overflow_chars=10,
                dropped_sections=[],
                section_status={"snapshot": "truncated"},
            ),
            include_stage_a_evidence_index=include_stage_a_evidence_index,
            stage_a_index_reason=stage_a_index_reason,
        )

    def _fake_retry(  # type: ignore[no-untyped-def]
        *,
        enabled,
        plan_on_first_attempt,
        attempt,
        prompt_inputs,
        attempt_prompt_budget,
        is_delta_retry,
        include_stage_a_evidence_index,
        stage_a_index_reason,
        run_plan_patch,
        tep_equilibrium_check,
        tep_equilibrium_max_iterations,
        tep_equilibrium_similarity_threshold,
    ):
        del enabled, plan_on_first_attempt, attempt, prompt_inputs
        del attempt_prompt_budget, is_delta_retry, run_plan_patch
        return RetryPlanResult(
            prompt="final",
            prompt_budget_result=PromptBudgetResult(
                prompt="final",
                used_chars=20,
                overflow_chars=0,
                dropped_sections=[],
                section_status={},
            ),
            include_stage_a_evidence_index=include_stage_a_evidence_index,
            stage_a_index_reason=stage_a_index_reason,
        )

    monkeypatch.setattr(flow, "build_stage_b_prompt_with_budget", _fake_build)
    monkeypatch.setattr(flow, "maybe_run_retry_plan_then_patch", _fake_retry)
    monkeypatch.setattr(
        flow,
        "_build_context_compaction_bundle",
        lambda **_: SimpleNamespace(
            text="text",
            source_sections=["snapshot"],
            source_chars={"snapshot": 4},
            replace_snapshot=True,
            replace_pytest_output=False,
            replace_stage_a_context=False,
            replace_project_files=False,
        ),
    )
    monkeypatch.setattr(flow, "_should_invoke_context_compaction", lambda **_: (True, "auto"))
    monkeypatch.setattr(flow, "render_context_compaction_prompt", lambda **_: "prompt text")

    req = _request(tmp_path, prompt_inputs=_prompt_inputs())
    req = flow.NativeOpenAIAttemptPromptRequest(
        **{
            **{f.name: getattr(req, f.name) for f in req.__dataclass_fields__.values()},
            "run_openai_operation_with_transport_recovery": lambda *, invoke, **__: SimpleNamespace(
                compacted_text="",
                meta={"error": "context_too_large"},
            ),
        }
    )
    result = flow.assemble_native_openai_attempt_prompt(request=req)

    assert result.context_compaction.invoked is True
    assert result.context_compaction.success is False
    assert result.context_compaction.error == "context_too_large"


# ---------------------------------------------------------------------------
# assemble: compaction result without meta attr (line 212->214 branch)
# ---------------------------------------------------------------------------


def test_assemble_compaction_result_no_meta_attr(tmp_path: Path, monkeypatch) -> None:
    """When compaction_result has no .meta, context_compaction_meta stays empty."""

    def _fake_build(  # type: ignore[no-untyped-def]
        *,
        prompt_inputs,
        attempt_prompt_budget,
        is_delta_retry,
        include_stage_a_evidence_index,
        stage_a_index_reason,
        retry_repair_plan="",
        planning_mode=False,
    ):
        del attempt_prompt_budget, is_delta_retry, retry_repair_plan, planning_mode
        return StageBPromptBuildResult(
            prompt="base-prompt",
            prompt_budget_result=PromptBudgetResult(
                prompt="base-prompt",
                used_chars=20,
                overflow_chars=10,
                dropped_sections=[],
                section_status={"snapshot": "truncated"},
            ),
            include_stage_a_evidence_index=include_stage_a_evidence_index,
            stage_a_index_reason=stage_a_index_reason,
        )

    def _fake_retry(  # type: ignore[no-untyped-def]
        *,
        enabled,
        plan_on_first_attempt,
        attempt,
        prompt_inputs,
        attempt_prompt_budget,
        is_delta_retry,
        include_stage_a_evidence_index,
        stage_a_index_reason,
        run_plan_patch,
        tep_equilibrium_check,
        tep_equilibrium_max_iterations,
        tep_equilibrium_similarity_threshold,
    ):
        del enabled, plan_on_first_attempt, attempt, prompt_inputs
        del attempt_prompt_budget, is_delta_retry, run_plan_patch
        return RetryPlanResult(
            prompt="final",
            prompt_budget_result=PromptBudgetResult(
                prompt="final",
                used_chars=20,
                overflow_chars=0,
                dropped_sections=[],
                section_status={},
            ),
            include_stage_a_evidence_index=include_stage_a_evidence_index,
            stage_a_index_reason=stage_a_index_reason,
        )

    monkeypatch.setattr(flow, "build_stage_b_prompt_with_budget", _fake_build)
    monkeypatch.setattr(flow, "maybe_run_retry_plan_then_patch", _fake_retry)
    monkeypatch.setattr(
        flow,
        "_build_context_compaction_bundle",
        lambda **_: SimpleNamespace(
            text="text",
            source_sections=["snapshot"],
            source_chars={"snapshot": 4},
            replace_snapshot=False,
            replace_pytest_output=True,
            replace_stage_a_context=False,
            replace_project_files=False,
        ),
    )
    monkeypatch.setattr(flow, "_should_invoke_context_compaction", lambda **_: (True, "auto"))
    monkeypatch.setattr(flow, "render_context_compaction_prompt", lambda **_: "prompt text")

    req = _request(tmp_path, prompt_inputs=_prompt_inputs())
    # Return result with compacted_text but no .meta attribute
    req = flow.NativeOpenAIAttemptPromptRequest(
        **{
            **{f.name: getattr(req, f.name) for f in req.__dataclass_fields__.values()},
            "run_openai_operation_with_transport_recovery": lambda *, invoke, **__: SimpleNamespace(
                compacted_text="compacted output",
            ),
        }
    )
    result = flow.assemble_native_openai_attempt_prompt(request=req)

    assert result.context_compaction.invoked is True
    assert result.context_compaction.success is True
    assert result.context_compaction.meta == {}
    # replace_pytest_output=True, others False
    assert result.context_compaction.replaced_sections == ["pytest_output"]
