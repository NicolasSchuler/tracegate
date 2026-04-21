"""Coverage-boost tests for evals/adaptive_controller_runtime.py.

Targets missing lines and branches to push coverage from 91% to 95%+:
- Line 104: wrote_sidecars property (empty list → False)
- Lines 192, 209-211, 214: bootstrap packet parse failure / exception
- Lines 244, 246, 248: lambda-RLM repo/python/packet missing
- Line 341: lambda-RLM no usable decomposition
- Lines 392-393: exception in write_controller_sidecars_for_cycle
- Branch [283, 282]: non-dict usage_row in model_usage_summaries
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from evals import adaptive_controller_runtime as runtime

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _base_inputs(tmp_path: Path, **overrides) -> runtime.ControllerRuntimeInputs:  # type: ignore[no-untyped-def]
    """Build a ControllerRuntimeInputs with sensible defaults."""
    root = tmp_path / "case"
    root.mkdir(exist_ok=True)
    (root / "buggy.py").write_text("def f(): return 1\n", encoding="utf-8")
    attempt_dir = tmp_path / "attempt_01"
    attempt_dir.mkdir(exist_ok=True)

    defaults = {
        "case_id": "case",
        "condition_value": "layered_adaptive_llm",
        "attempt": 1,
        "workdir": root,
        "attempt_dir": attempt_dir,
        "snapshot": {
            "frames": [
                {
                    "file_rel": "buggy.py",
                    "line": 1,
                    "function": "f",
                    "code": "return 1",
                    "locals": {},
                }
            ]
        },
        "failure_signature": "sig-1",
        "evidence_items": [],
        "new_evidence_ids": [],
        "reused_evidence_ids": [],
        "retry_prompt_mode": "delta",
        "retry_context_mode": "history",
        "retry_history_text": "",
        "retry_history_turns": [],
        "retry_history_turns_included": 0,
        "retry_history_truncated": False,
        "previous_failure_signature": "",
        "previous_note": "",
        "previous_patch_verifier_rejected": False,
        "previous_patch_verifier_reasons": [],
        "previous_patch_verifier_scope": "adaptive_only",
        "previous_timed_out": False,
        "previous_timeout_sec": None,
        "helper_action_selected": "skip_helper",
        "helper_action_executed": "skip_helper",
        "helper_decision_reason": "policy_off",
        "helper_outcome": "not_invoked",
        "helper_materialized_context": False,
        "helper_calls_used": 0,
        "helper_budget_max_calls": 2,
        "no_progress_streak": 0,
        "context_compaction_requested": False,
        "context_compaction_invoked": False,
        "context_compaction_success": False,
        "context_compaction_trigger": "not_needed",
        "prompt_budget": {"total_chars": 1000},
        "retry_plan_text": "",
        "lambda_rlm_enabled": False,
    }
    defaults.update(overrides)
    return runtime.ControllerRuntimeInputs(**defaults)


# ---------------------------------------------------------------------------
# ControllerSidecarWriteResult.wrote_sidecars property (line 104)
# ---------------------------------------------------------------------------


class TestControllerSidecarWriteResultProperty:
    def test_wrote_sidecars_false_for_empty(self) -> None:
        result = runtime.ControllerSidecarWriteResult(written_sidecar_names=[])
        assert result.wrote_sidecars is False

    def test_wrote_sidecars_true_for_nonempty(self) -> None:
        result = runtime.ControllerSidecarWriteResult(written_sidecar_names=["file.json"])
        assert result.wrote_sidecars is True


# ---------------------------------------------------------------------------
# bootstrap_controller_packet parse failure → line 192
# ---------------------------------------------------------------------------


class TestBootstrapControllerPacketParseFailed:
    def test_parse_returns_none_sets_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When parse_controller_packet returns None, the error should be set."""
        monkeypatch.setattr(runtime, "parse_controller_packet", lambda _: None)
        monkeypatch.setattr(
            runtime,
            "build_controller_strategy_summary",
            lambda packet: SimpleNamespace(summary="s", sources=["x"]),
        )

        inputs = _base_inputs(tmp_path, lambda_rlm_enabled=False)
        prepared = runtime.prepare_controller_runtime(inputs=inputs)

        assert (
            prepared.bootstrap_controller_packet_error == "bootstrap controller packet parse failed"
        )
        assert prepared.controller_strategy_summary == ""
        # Error recorded in patch meta
        assert "adaptive_controller_summarizer_error" in prepared.prepared_patch_meta


# ---------------------------------------------------------------------------
# bootstrap_controller_packet exception → lines 209-211, 214
# ---------------------------------------------------------------------------


class TestBootstrapControllerPacketException:
    def test_exception_during_bootstrap_sets_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """When _build_sidecars or parse raises, the error is captured."""

        def _raise_sidecars(**kwargs):  # type: ignore[no-untyped-def]
            raise ValueError("sidecar build failed")

        monkeypatch.setattr(runtime, "build_adaptive_controller_sidecars", _raise_sidecars)

        inputs = _base_inputs(tmp_path, lambda_rlm_enabled=False)
        prepared = runtime.prepare_controller_runtime(inputs=inputs)

        assert "ValueError: sidecar build failed" in prepared.bootstrap_controller_packet_error
        assert "adaptive_controller_summarizer_error" in prepared.prepared_patch_meta


# ---------------------------------------------------------------------------
# lambda_rlm_enabled + repo_dir not found → line 244
# ---------------------------------------------------------------------------


class TestLambdaRlmRepoNotFound:
    def test_repo_not_found_error(self, tmp_path: Path) -> None:
        nonexistent_repo = tmp_path / "no_such_repo"
        python_exec = tmp_path / "python"
        python_exec.write_text("#!/usr/bin/env python\n", encoding="utf-8")

        inputs = _base_inputs(
            tmp_path,
            lambda_rlm_enabled=True,
            lambda_rlm_repo_dir=nonexistent_repo,
            lambda_rlm_python=python_exec,
        )
        prepared = runtime.prepare_controller_runtime(inputs=inputs)

        assert "lambda-RLM repo not found" in prepared.lambda_rlm_error
        assert "adaptive_controller_lambda_rlm_error" in prepared.prepared_attempt_meta


# ---------------------------------------------------------------------------
# lambda_rlm_enabled + python not found → line 246
# ---------------------------------------------------------------------------


class TestLambdaRlmPythonNotFound:
    def test_python_not_found_error(self, tmp_path: Path) -> None:
        repo_dir = tmp_path / "lambda_repo"
        repo_dir.mkdir()
        nonexistent_python = tmp_path / "no_python_here"

        inputs = _base_inputs(
            tmp_path,
            lambda_rlm_enabled=True,
            lambda_rlm_repo_dir=repo_dir,
            lambda_rlm_python=nonexistent_python,
        )
        prepared = runtime.prepare_controller_runtime(inputs=inputs)

        assert "lambda-RLM python executable not found" in prepared.lambda_rlm_error


# ---------------------------------------------------------------------------
# lambda_rlm_enabled + bootstrap packet is None → line 248
# ---------------------------------------------------------------------------


class TestLambdaRlmBootstrapPacketNone:
    def test_bootstrap_packet_none_blocks_lambda_rlm(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(runtime, "parse_controller_packet", lambda _: None)

        repo_dir = tmp_path / "lambda_repo"
        repo_dir.mkdir()
        python_exec = tmp_path / "python"
        python_exec.write_text("#!/usr/bin/env python\n", encoding="utf-8")

        inputs = _base_inputs(
            tmp_path,
            lambda_rlm_enabled=True,
            lambda_rlm_repo_dir=repo_dir,
            lambda_rlm_python=python_exec,
        )
        prepared = runtime.prepare_controller_runtime(inputs=inputs)

        assert "bootstrap controller packet parse failed" in prepared.lambda_rlm_error


# ---------------------------------------------------------------------------
# lambda-RLM returns no usable decomposition → line 341
# ---------------------------------------------------------------------------


class TestLambdaRlmNoUsableDecomposition:
    def test_empty_decomposition_sets_error(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            runtime,
            "build_controller_strategy_summary",
            lambda packet: SimpleNamespace(summary="s", sources=["x"]),
        )

        repo_dir = tmp_path / "lambda_repo"
        repo_dir.mkdir()
        python_exec = tmp_path / "python"
        python_exec.write_text("#!/usr/bin/env python\n", encoding="utf-8")

        def _empty_lambda(**kwargs):  # type: ignore[no-untyped-def]
            return SimpleNamespace(
                steps=[],
                summary="",
                execution_time_sec=0.1,
                root_model="test",
                usage_summary={"total_input_tokens": 0, "total_output_tokens": 0},
            )

        inputs = _base_inputs(
            tmp_path,
            lambda_rlm_enabled=True,
            lambda_rlm_repo_dir=repo_dir,
            lambda_rlm_python=python_exec,
            lambda_rlm_invoke=_empty_lambda,
        )
        prepared = runtime.prepare_controller_runtime(inputs=inputs)

        assert prepared.lambda_rlm_error == "lambda-RLM returned no usable decomposition"
        assert prepared.prepared_attempt_meta["adaptive_controller_lambda_rlm_success"] is False


# ---------------------------------------------------------------------------
# Non-dict usage_row in model_usage_summaries → branch [283, 282]
# ---------------------------------------------------------------------------


class TestLambdaRlmNonDictUsageRow:
    def test_non_dict_usage_row_skipped(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(
            runtime,
            "build_controller_strategy_summary",
            lambda packet: SimpleNamespace(summary="s", sources=["x"]),
        )

        repo_dir = tmp_path / "lambda_repo"
        repo_dir.mkdir()
        python_exec = tmp_path / "python"
        python_exec.write_text("#!/usr/bin/env python\n", encoding="utf-8")

        def _lambda_with_bad_usage(**kwargs):  # type: ignore[no-untyped-def]
            return SimpleNamespace(
                steps=[{"kind": "inspect"}],
                summary="plan summary",
                execution_time_sec=0.2,
                root_model="test",
                usage_summary={
                    "total_input_tokens": 10,
                    "total_output_tokens": 5,
                    "model_usage_summaries": {
                        "good_model": {"total_calls": 3},
                        "bad_model": "not_a_dict",  # non-dict row → skipped
                        "also_bad": 42,
                    },
                },
            )

        inputs = _base_inputs(
            tmp_path,
            lambda_rlm_enabled=True,
            lambda_rlm_repo_dir=repo_dir,
            lambda_rlm_python=python_exec,
            lambda_rlm_invoke=_lambda_with_bad_usage,
        )
        prepared = runtime.prepare_controller_runtime(inputs=inputs)

        # Only the dict row contributes calls
        assert prepared.lambda_rlm_call_count == 3
        assert prepared.lambda_rlm_total_tokens == 15
        assert prepared.lambda_rlm_error == ""


# ---------------------------------------------------------------------------
# write_controller_sidecars_for_cycle exception → lines 392-393
# ---------------------------------------------------------------------------


class TestWriteSidecarsException:
    def test_exception_returns_error_result(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        inputs = _base_inputs(tmp_path, lambda_rlm_enabled=False)
        prepared = runtime.prepare_controller_runtime(inputs=inputs)

        # Make write_adaptive_controller_sidecars raise
        monkeypatch.setattr(
            runtime,
            "write_adaptive_controller_sidecars",
            lambda **kw: (_ for _ in ()).throw(OSError("disk full")),
        )

        result = runtime.write_controller_sidecars_for_cycle(
            prepared=prepared,
            syntax_cycle=1,
            syntax_max_cycles=1,
            syntax_last_error_type="",
            syntax_last_error_msg="",
            syntax_retry_feedback="",
            context_dir=tmp_path / "ctx",
        )

        assert result.written_sidecar_names == []
        assert "OSError: disk full" in result.sidecar_error
        assert result.wrote_sidecars is False
