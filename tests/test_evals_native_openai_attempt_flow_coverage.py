"""Coverage-boost tests for evals/native_openai_attempt_flow.py.

Targets missing branch [248, 252]: when compaction result is not None but
compacted_text is empty/None AND context_compaction_error was already set
from a prior exception path.  This triggers the elif at line 248 to be
False, skipping to line 252.

Also targets the equivalent flow where compaction_result is returned but
has no usable compacted_text, exercising the empty-text error-from-meta path.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from evals import native_openai_attempt_flow as flow
from evals.eval_types import PromptBudgetResult
from evals.native_stage_b_flow import RetryPlanResult, StageBPromptInputs


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
    tmp_path: Path,
    *,
    prompt_inputs: StageBPromptInputs,
    compaction_fn_override=None,  # type: ignore[no-untyped-def]
    transport_recovery_override=None,  # type: ignore[no-untyped-def]
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
        context_compaction_fn=compaction_fn_override
        or (
            lambda **_: SimpleNamespace(
                compacted_text="compacted summary",
                meta={"compact_context_duration_sec": 0.25},
            )
        ),
        patcher_for_attempt=SimpleNamespace(response_mode="edits"),
        run_openai_operation_with_transport_recovery=transport_recovery_override
        or (lambda *, invoke, **__: invoke()),
        classify_transport_recovery_failure_kind=lambda _: "",
        attempt_context_protocol="staged",
        before=SimpleNamespace(stdout="stdout", stderr="stderr"),
        cumulative_runtime_sec=0.0,
        attempt_dir=tmp_path / "attempt",
        workdir=tmp_path / "workdir",
        retry_plan_enabled=False,
        run_plan_patch=lambda prompt: (None, {}),
    )


class TestCompactionEmptyTextWithErrorAlreadySet:
    """Branch [248, 252]: compaction_result not None, text empty, error already set.

    This occurs when the invocation itself raises an exception (setting
    context_compaction_error at line 210) but then compaction_result ends up
    None — that path goes to the ``elif`` at line 250.  The tricky [248, 252]
    branch requires the "result is not None, text is empty, error is truthy"
    path.  We simulate this by having the transport recovery wrapper raise.
    """

    def test_compaction_raises_then_result_none_with_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When transport recovery raises, compaction_result is None and error is set."""
        build_calls = []
        retry_calls = []

        def _fake_build(
            *,
            prompt_inputs,
            attempt_prompt_budget,
            is_delta_retry,
            include_stage_a_evidence_index,
            stage_a_index_reason,
        ):  # type: ignore[no-untyped-def]
            build_calls.append(prompt_inputs)
            return SimpleNamespace(
                prompt="test prompt",
                prompt_budget_result=PromptBudgetResult(
                    prompt="test prompt",
                    used_chars=50,
                    overflow_chars=0,
                    dropped_sections=[],
                    section_status={},
                ),
                include_stage_a_evidence_index=include_stage_a_evidence_index,
                stage_a_index_reason=stage_a_index_reason,
            )

        def _fake_retry(
            *,
            enabled,
            attempt,
            prompt_inputs,
            attempt_prompt_budget,
            is_delta_retry,
            include_stage_a_evidence_index,
            stage_a_index_reason,
            run_plan_patch,
            plan_on_first_attempt,
            tep_equilibrium_check,
            tep_equilibrium_max_iterations,
            tep_equilibrium_similarity_threshold,
        ):  # type: ignore[no-untyped-def]
            retry_calls.append(1)
            return RetryPlanResult(
                prompt="final prompt",
                prompt_budget_result=PromptBudgetResult(
                    prompt="final prompt",
                    used_chars=50,
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
                text="text content",
                source_sections=["snapshot"],
                source_chars={"snapshot": 12},
                replace_snapshot=False,
                replace_pytest_output=False,
                replace_stage_a_context=False,
                replace_project_files=False,
            ),
        )
        monkeypatch.setattr(flow, "_should_invoke_context_compaction", lambda **_: (True, "auto"))
        monkeypatch.setattr(flow, "render_context_compaction_prompt", lambda **_: "prompt text")

        def _raise_on_invoke(*, invoke, **__):  # type: ignore[no-untyped-def]
            raise RuntimeError("transport crash")

        req = _request(
            tmp_path, prompt_inputs=_prompt_inputs(), transport_recovery_override=_raise_on_invoke
        )
        result = flow.assemble_native_openai_attempt_prompt(request=req)

        # Compaction was invoked but failed
        assert result.context_compaction.invoked is True
        assert result.context_compaction.success is False
        assert "RuntimeError: transport crash" in result.context_compaction.error

    def test_compaction_returns_empty_text_error_from_meta(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Compaction result has empty compacted_text → error taken from meta."""

        def _fake_build(
            *,
            prompt_inputs,
            attempt_prompt_budget,
            is_delta_retry,
            include_stage_a_evidence_index,
            stage_a_index_reason,
        ):  # type: ignore[no-untyped-def]
            return SimpleNamespace(
                prompt="test prompt",
                prompt_budget_result=PromptBudgetResult(
                    prompt="test prompt",
                    used_chars=50,
                    overflow_chars=0,
                    dropped_sections=[],
                    section_status={},
                ),
                include_stage_a_evidence_index=include_stage_a_evidence_index,
                stage_a_index_reason=stage_a_index_reason,
            )

        def _fake_retry(
            *,
            enabled,
            attempt,
            prompt_inputs,
            attempt_prompt_budget,
            is_delta_retry,
            include_stage_a_evidence_index,
            stage_a_index_reason,
            run_plan_patch,
            plan_on_first_attempt,
            tep_equilibrium_check,
            tep_equilibrium_max_iterations,
            tep_equilibrium_similarity_threshold,
        ):  # type: ignore[no-untyped-def]
            return RetryPlanResult(
                prompt="final prompt",
                prompt_budget_result=PromptBudgetResult(
                    prompt="final prompt",
                    used_chars=50,
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
                text="text content",
                source_sections=["snapshot"],
                source_chars={"snapshot": 12},
                replace_snapshot=False,
                replace_pytest_output=False,
                replace_stage_a_context=False,
                replace_project_files=False,
            ),
        )
        monkeypatch.setattr(flow, "_should_invoke_context_compaction", lambda **_: (True, "auto"))
        monkeypatch.setattr(flow, "render_context_compaction_prompt", lambda **_: "prompt text")

        # Return result with empty compacted_text and error in meta
        def _return_empty_text(*, invoke, **__):  # type: ignore[no-untyped-def]
            return SimpleNamespace(
                compacted_text="   ",  # whitespace only → not valid
                meta={"error": "compaction produced nothing useful"},
            )

        req = _request(
            tmp_path,
            prompt_inputs=_prompt_inputs(),
            transport_recovery_override=_return_empty_text,
        )
        result = flow.assemble_native_openai_attempt_prompt(request=req)

        assert result.context_compaction.invoked is True
        assert result.context_compaction.success is False
        # Error extracted from meta since compacted_text was empty
        assert result.context_compaction.error == "compaction produced nothing useful"
