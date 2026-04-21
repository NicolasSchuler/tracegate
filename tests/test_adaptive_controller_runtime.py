from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from evals import adaptive_controller_runtime as runtime


def _controller_inputs(tmp_path: Path) -> runtime.ControllerRuntimeInputs:
    root = tmp_path / "case"
    root.mkdir()
    (root / "buggy.py").write_text(
        "def buggy(value: int) -> int:\n    return value + 1\n",
        encoding="utf-8",
    )
    attempt_dir = tmp_path / "attempt_01"
    attempt_dir.mkdir()
    lambda_repo = tmp_path / "lambda_repo"
    lambda_repo.mkdir()
    lambda_python = tmp_path / "python"
    lambda_python.write_text("#!/usr/bin/env python\n", encoding="utf-8")
    return runtime.ControllerRuntimeInputs(
        case_id="case",
        condition_value="layered_adaptive_llm",
        attempt=1,
        workdir=root,
        attempt_dir=attempt_dir,
        snapshot={
            "frames": [
                {
                    "file_rel": "buggy.py",
                    "line": 2,
                    "function": "buggy",
                    "code": "return value + 1",
                    "locals": {"value": 1},
                }
            ]
        },
        failure_signature="sig-1",
        evidence_items=[],
        new_evidence_ids=[],
        reused_evidence_ids=[],
        retry_prompt_mode="delta",
        retry_context_mode="history",
        retry_history_text="Attempt 1: tests_still_failing",
        retry_history_turns=["Attempt 1: tests_still_failing"],
        retry_history_turns_included=1,
        retry_history_truncated=False,
        previous_failure_signature="",
        previous_note="",
        previous_patch_verifier_rejected=False,
        previous_patch_verifier_reasons=[],
        previous_patch_verifier_scope="adaptive_only",
        previous_timed_out=False,
        previous_timeout_sec=None,
        helper_action_selected="snapshot_lite",
        helper_action_executed="snapshot_lite",
        helper_decision_reason="policy_action_snapshot_lite",
        helper_outcome="materialized",
        helper_materialized_context=True,
        helper_calls_used=1,
        helper_budget_max_calls=2,
        no_progress_streak=0,
        context_compaction_requested=False,
        context_compaction_invoked=False,
        context_compaction_success=False,
        context_compaction_trigger="not_needed",
        prompt_budget={"total_chars": 1000},
        retry_plan_text="Focus on the return expression.",
        lambda_rlm_enabled=True,
        lambda_rlm_repo_dir=lambda_repo,
        lambda_rlm_python=lambda_python,
        lambda_rlm_context_window_chars=5000,
        lambda_rlm_timeout_sec=30,
        case_timeout_sec=60,
        patcher_name="openai",
        configured_model="gpt-test",
        configured_api_base_url="https://example.invalid",
        configured_api_key="test-key",
    )


def test_prepare_controller_runtime_collects_summary_and_lambda_rlm_usage(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(
        runtime,
        "build_controller_strategy_summary",
        lambda packet: SimpleNamespace(summary="deterministic summary", sources=["packet"]),
    )
    monkeypatch.setattr(
        runtime,
        "invoke_lambda_rlm_decomposition",
        lambda **kwargs: SimpleNamespace(
            steps=[{"kind": "inspect"}],
            summary="lambda summary",
            execution_time_sec=0.25,
            root_model=kwargs["model"],
            usage_summary={
                "total_input_tokens": 13,
                "total_output_tokens": 5,
                "model_usage_summaries": {"planner": {"total_calls": 2}},
            },
        ),
    )

    prepared = runtime.prepare_controller_runtime(inputs=_controller_inputs(tmp_path))

    assert prepared.controller_strategy_summary == "deterministic summary"
    assert prepared.prepared_patch_meta["adaptive_controller_summarizer_summary"] == (
        "deterministic summary"
    )
    assert prepared.lambda_rlm_strategy_summary == "lambda summary"
    assert prepared.lambda_rlm_call_count == 2
    assert prepared.lambda_rlm_total_tokens == 18
    assert prepared.prepared_attempt_meta["adaptive_controller_lambda_rlm_success"] is True
    assert prepared.strategy_summary_for_packet == "lambda summary"


def test_write_controller_sidecars_for_cycle_writes_expected_sidecars(tmp_path: Path) -> None:
    prepared = runtime.prepare_controller_runtime(inputs=_controller_inputs(tmp_path))

    result = runtime.write_controller_sidecars_for_cycle(
        prepared=prepared,
        syntax_cycle=1,
        syntax_max_cycles=2,
        syntax_last_error_type="",
        syntax_last_error_msg="",
        syntax_retry_feedback="",
        context_dir=prepared.inputs.attempt_dir,
    )

    assert result.sidecar_error == ""
    assert "llmdebug_controller_packet.json" in result.written_sidecar_names
    assert (prepared.inputs.attempt_dir / "llmdebug_controller_packet.json").is_file()
