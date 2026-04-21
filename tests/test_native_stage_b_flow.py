from __future__ import annotations

from evals import native_stage_b_flow as flow
from evals.eval_prompts import _build_stage_b_prompt_sections
from evals.eval_types import PromptBudgetResult


def _prompt_inputs() -> flow.StageBPromptInputs:
    return flow.StageBPromptInputs(
        evidence={},
        snapshot=None,
        stage_a_context="requested context",
        stage_a_evidence_index="[e1] buggy.py:4",
        compacted_context="",
        include_snapshot=True,
        include_pytest_output=True,
        include_stage_a_context=True,
        include_project_files=False,
        retry_packet="",
        retry_history="",
        retry_prompt_mode="history",
        attempt=2,
        known_evidence_refs=[],
        new_evidence_ids=[],
    )


def test_openai_preflight_total_char_cap_preserves_guard() -> None:
    assert flow.openai_preflight_total_char_cap(context_window_tokens=0, max_tokens=512) == 0
    assert flow.openai_preflight_total_char_cap(context_window_tokens=1024, max_tokens=2048) == 1


def test_build_stage_b_prompt_with_budget_reintroduces_stage_a_index_on_truncation(
    monkeypatch,
) -> None:
    apply_calls = {"count": 0}

    monkeypatch.setattr(
        flow,
        "build_stage_b_prompt_sections",
        lambda **kwargs: [{"name": "stub", "text": str(kwargs["include_stage_a_evidence_index"])}],
    )

    def _fake_apply_prompt_budget(*, sections, budget, is_delta_retry):  # type: ignore[no-untyped-def]
        del budget, is_delta_retry
        apply_calls["count"] += 1
        include_index = sections[0]["text"] == "True"
        if apply_calls["count"] == 1:
            return PromptBudgetResult(
                prompt="without-index",
                used_chars=10,
                overflow_chars=4,
                dropped_sections=[],
                section_status={"stage_a_context": "truncated"},
            )
        return PromptBudgetResult(
            prompt="with-index" if include_index else "unexpected",
            used_chars=12,
            overflow_chars=0,
            dropped_sections=[],
            section_status={"stage_a_context": "kept"},
        )

    monkeypatch.setattr(flow, "_apply_prompt_budget", _fake_apply_prompt_budget)

    result = flow.build_stage_b_prompt_with_budget(
        prompt_inputs=_prompt_inputs(),
        attempt_prompt_budget={"total_chars": 200},
        is_delta_retry=True,
        include_stage_a_evidence_index=False,
        stage_a_index_reason="redundant_skipped",
    )

    assert apply_calls["count"] == 2
    assert result.prompt == "with-index"
    assert result.include_stage_a_evidence_index is True
    assert result.stage_a_index_reason == "context_truncated_fallback"


def test_maybe_run_retry_plan_then_patch_can_plan_on_attempt_one(monkeypatch) -> None:
    monkeypatch.setattr(
        flow,
        "build_stage_b_prompt_with_budget",
        lambda **kwargs: flow.StageBPromptBuildResult(
            prompt="plan-prompt" if kwargs.get("planning_mode") else "final-prompt",
            prompt_budget_result=PromptBudgetResult(
                prompt="plan-prompt" if kwargs.get("planning_mode") else "final-prompt",
                used_chars=20,
                overflow_chars=0,
                dropped_sections=[],
                section_status={"retry_repair_plan": "kept"},
            ),
            include_stage_a_evidence_index=kwargs["include_stage_a_evidence_index"],
            stage_a_index_reason=kwargs["stage_a_index_reason"],
        ),
    )

    result = flow.maybe_run_retry_plan_then_patch(
        enabled=True,
        plan_on_first_attempt=True,
        attempt=1,
        prompt_inputs=flow.StageBPromptInputs(
            **{**_prompt_inputs().__dict__, "attempt": 1},
        ),
        attempt_prompt_budget={"total_chars": 200},
        is_delta_retry=False,
        include_stage_a_evidence_index=False,
        stage_a_index_reason="redundant_skipped",
        run_plan_patch=lambda prompt: (
            "Diagnosis: fix\nMinimal edit plan: change return",
            prompt,
            {},
        ),
    )

    assert result.retry_plan_invoked is True
    assert result.retry_plan_succeeded is True
    assert result.retry_plan_prompt_text == "plan-prompt"
    assert result.retry_plan_raw_response == "plan-prompt"
    assert result.prompt == "final-prompt"


def test_maybe_run_retry_plan_then_patch_tep_equilibrium_converges(monkeypatch) -> None:
    monkeypatch.setattr(
        flow,
        "build_stage_b_prompt_with_budget",
        lambda **kwargs: flow.StageBPromptBuildResult(
            prompt="plan-prompt" if kwargs.get("planning_mode") else "final-prompt",
            prompt_budget_result=PromptBudgetResult(
                prompt="plan-prompt" if kwargs.get("planning_mode") else "final-prompt",
                used_chars=20,
                overflow_chars=0,
                dropped_sections=[],
                section_status={"retry_repair_plan": "kept"},
            ),
            include_stage_a_evidence_index=kwargs["include_stage_a_evidence_index"],
            stage_a_index_reason=kwargs["stage_a_index_reason"],
        ),
    )
    plans = iter(
        [
            ("plan alpha", "raw alpha", {}),
            ("plan alpha", "raw alpha again", {}),
        ]
    )

    result = flow.maybe_run_retry_plan_then_patch(
        enabled=True,
        plan_on_first_attempt=False,
        attempt=2,
        prompt_inputs=_prompt_inputs(),
        attempt_prompt_budget={"total_chars": 200},
        is_delta_retry=False,
        include_stage_a_evidence_index=False,
        stage_a_index_reason="redundant_skipped",
        tep_equilibrium_check=True,
        tep_equilibrium_max_iterations=3,
        tep_equilibrium_similarity_threshold=0.95,
        run_plan_patch=lambda prompt: next(plans),
    )

    assert result.retry_plan_succeeded is True
    assert result.tep_equilibrium_iterations == 2
    assert result.tep_equilibrium_converged is True
    assert result.retry_plan_text == "plan alpha"


def test_maybe_run_retry_plan_then_patch_tep_equilibrium_stops_at_max_iterations(
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        flow,
        "build_stage_b_prompt_with_budget",
        lambda **kwargs: flow.StageBPromptBuildResult(
            prompt="plan-prompt" if kwargs.get("planning_mode") else "final-prompt",
            prompt_budget_result=PromptBudgetResult(
                prompt="plan-prompt" if kwargs.get("planning_mode") else "final-prompt",
                used_chars=20,
                overflow_chars=0,
                dropped_sections=[],
                section_status={"retry_repair_plan": "kept"},
            ),
            include_stage_a_evidence_index=kwargs["include_stage_a_evidence_index"],
            stage_a_index_reason=kwargs["stage_a_index_reason"],
        ),
    )
    plans = iter(
        [
            ("plan alpha", "raw alpha", {}),
            ("plan beta", "raw beta", {}),
            ("plan gamma", "raw gamma", {}),
        ]
    )

    result = flow.maybe_run_retry_plan_then_patch(
        enabled=True,
        plan_on_first_attempt=False,
        attempt=2,
        prompt_inputs=_prompt_inputs(),
        attempt_prompt_budget={"total_chars": 200},
        is_delta_retry=False,
        include_stage_a_evidence_index=False,
        stage_a_index_reason="redundant_skipped",
        tep_equilibrium_check=True,
        tep_equilibrium_max_iterations=3,
        tep_equilibrium_similarity_threshold=0.99,
        run_plan_patch=lambda prompt: next(plans),
    )

    assert result.retry_plan_succeeded is True
    assert result.tep_equilibrium_iterations == 3
    assert result.tep_equilibrium_converged is False
    assert result.retry_plan_text == "plan gamma"


def test_native_stage_b_builder_matches_shared_prompt_ordering() -> None:
    prompt_inputs = flow.StageBPromptInputs(
        **{
            **_prompt_inputs().__dict__,
            "evidence": {
                "condition": "layered_adaptive_llm",
                "repro": ["python", "-m", "pytest", "-q"],
                "stdout": "traceback",
                "stderr": "",
                "failure_signature": "AssertionError:sig",
                "previous_failure_signature": "AssertionError:prev",
                "failure_signature_repeat": True,
                "file_index": "buggy.py",
                "files": "buggy.py",
            },
            "snapshot": {
                "exception": {"type": "IndexError", "message": "out of range"},
                "frames": [{"file_rel": "buggy.py", "line": 4, "function": "f"}],
            },
            "snapshot_lite_context": "Frame 0: buggy.py:4",
            "prompt_order_mode": "crash_first",
        }
    )

    native_sections = flow.build_stage_b_prompt_sections(
        prompt_inputs=prompt_inputs,
        include_stage_a_evidence_index=True,
    )
    shared_sections = _build_stage_b_prompt_sections(
        evidence=prompt_inputs.evidence,
        snapshot=prompt_inputs.snapshot,
        snapshot_prompt_text=prompt_inputs.snapshot_prompt_text,
        stage_a_context=prompt_inputs.stage_a_context,
        stage_a_evidence_index=prompt_inputs.stage_a_evidence_index,
        include_stage_a_evidence_index=True,
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
        prompt_order_mode=prompt_inputs.prompt_order_mode,
    )

    assert [section["key"] for section in native_sections] == [
        section["key"] for section in shared_sections
    ]


def test_native_stage_b_builder_applies_full_tep_actor_overrides() -> None:
    prompt_inputs = flow.StageBPromptInputs(
        **{
            **_prompt_inputs().__dict__,
            "evidence": {
                "condition": "layered_adaptive_llm",
                "repro": ["python", "-m", "pytest", "-q"],
                "stdout": "traceback",
                "stderr": "",
                "failure_signature": "AssertionError:sig",
                "previous_failure_signature": "AssertionError:prev",
                "failure_signature_repeat": True,
            },
            "prompt_retry_reflection": True,
            "prompt_repair_method_scaffold": True,
            "prompt_adaptive_stage_b_cot": True,
            "enable_tep": True,
            "stagnation_streak": 2,
            "tep_patch_actor_overrides": {
                "instructions": "Use the rewritten patch instructions.",
                "closing": "Return only the updated patch.",
            },
        }
    )

    sections = flow.build_stage_b_prompt_sections(
        prompt_inputs=prompt_inputs,
        include_stage_a_evidence_index=True,
    )
    sections_by_key = {section["key"]: section["text"] for section in sections}

    assert sections_by_key["instructions"] == "Use the rewritten patch instructions."
    assert sections_by_key["closing"] == "Return only the updated patch."
