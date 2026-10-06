"""Tests for P1.1 — Rule-based traceback informativeness pre-gate."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from evals import eval_attempt_flow, run_eval
from evals.patchers.base import ContextRequestResult, PatchResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_case(tmp_path: Path) -> run_eval.CaseSpec:
    case_dir = tmp_path / "case_src"
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "buggy.py").write_text(
        "def f() -> int:\n    return 1\n",
        encoding="utf-8",
    )
    (case_dir / "test_buggy.py").write_text(
        "from buggy import f\n\ndef test_f() -> None:\n    assert f() == 2\n",
        encoding="utf-8",
    )
    return run_eval.CaseSpec(
        case_id="pregate_case",
        path=case_dir,
        category="traceback_controls",
        difficulty="easy",
        snapshot_dependency="helpful",
        expected_exception="AssertionError",
        tags=[],
        python_args=["-m", "pytest", "-q"],
        timeout_sec=30,
    )


def _snapshot_payload(
    exc_type: str = "AssertionError",
    exc_message: str = "expected 2 got 1",
    frames: list[dict] | None = None,
) -> dict:
    if frames is None:
        frames = [
            {
                "file_rel": "buggy.py",
                "line": 2,
                "function": "f",
                "code": "return 1",
                "locals": {"x": 1},
            }
        ]
    return {
        "exception": {"type": exc_type, "message": exc_message},
        "frames": frames,
        "pytest": {"nodeid": "test_buggy.py::test_f", "repro": ["python", "-m", "pytest", "-q"]},
    }


@dataclass
class _PreGateCapturePatcher:
    """Patcher that records whether request_context was called."""

    name: str = "pregate_capture"
    request_calls: int = 0
    prompts: list[str] = field(default_factory=list)

    def request_context(self, *, prompt: str, context_dir, cwd) -> ContextRequestResult:
        del context_dir, cwd
        self.request_calls += 1
        return ContextRequestResult(
            request_json={
                "need_more_context": True,
                "requests": [{"type": "file", "path": "buggy.py"}],
                "sufficiency_rationale": "need code context",
            },
            raw_response='{"need_more_context":true}',
            meta={},
        )

    def propose_patch(self, *, prompt: str, context_dir, cwd) -> PatchResult:
        del context_dir, cwd
        self.prompts.append(prompt)
        return PatchResult(diff=None, meta={"reason": "no_diff"})


# ===========================================================================
# Unit tests: Scoring
# ===========================================================================


class TestTracebackInformativenessScore:
    def test_syntax_error_scores_high(self) -> None:
        result = run_eval._traceback_informativeness_score(
            exception_type="SyntaxError",
            exception_message="invalid syntax",
            frame_count=1,
            crash_in_user_code=True,
        )
        assert result.score >= 0.85

    def test_name_error_scores_high(self) -> None:
        result = run_eval._traceback_informativeness_score(
            exception_type="NameError",
            exception_message="name 'x' is not defined",
            frame_count=1,
            crash_in_user_code=True,
        )
        assert result.score >= 0.8

    def test_indentation_error_scores_high(self) -> None:
        result = run_eval._traceback_informativeness_score(
            exception_type="IndentationError",
            exception_message="unexpected indent",
            frame_count=1,
            crash_in_user_code=True,
        )
        assert result.score >= 0.85

    def test_assertion_error_bare_scores_low(self) -> None:
        result = run_eval._traceback_informativeness_score(
            exception_type="AssertionError",
            exception_message="",
            frame_count=1,
            crash_in_user_code=True,
        )
        # 0.30*0.1 + 0.25*0.0 + 0.15*1.0 + 0.15*0.0 + 0.15*1.0 = 0.33
        assert result.score <= 0.35

    def test_assertion_error_with_values_moderate(self) -> None:
        result = run_eval._traceback_informativeness_score(
            exception_type="AssertionError",
            exception_message="assert 1 == 2",
            frame_count=1,
            crash_in_user_code=True,
        )
        # AssertionError has low type specificity (0.1) — even with good
        # assertion quality, the score stays below the 0.7 threshold.
        assert result.score < 0.7
        assert result.f_assertion_quality == 1.0  # comparison values present

    def test_timeout_error_scores_very_low(self) -> None:
        result = run_eval._traceback_informativeness_score(
            exception_type="TimeoutError",
            exception_message="",
            frame_count=5,
            crash_in_user_code=False,
        )
        # 0.30*0.05 + 0.25*0.0 + 0.15*0.0 + 0.15*0.5 + 0.15*0.7 = 0.195
        assert result.score < 0.25

    def test_key_error_scores_moderate(self) -> None:
        result = run_eval._traceback_informativeness_score(
            exception_type="KeyError",
            exception_message="'missing'",
            frame_count=2,
            crash_in_user_code=True,
        )
        assert 0.5 <= result.score <= 0.8

    def test_unknown_exception_default(self) -> None:
        result = run_eval._traceback_informativeness_score(
            exception_type="CustomError",
            exception_message="something happened",
            frame_count=3,
            crash_in_user_code=True,
        )
        assert result.f_type_specificity == run_eval._EXCEPTION_TYPE_SPECIFICITY_DEFAULT

    def test_attribute_error_nonetype(self) -> None:
        result = run_eval._traceback_informativeness_score(
            exception_type="AttributeError",
            exception_message="'NoneType' object has no attribute 'x'",
            frame_count=2,
            crash_in_user_code=True,
        )
        assert 0.55 <= result.score <= 0.85

    def test_deep_stack_reduces_depth_score(self) -> None:
        result = run_eval._traceback_informativeness_score(
            exception_type="TypeError",
            exception_message="unsupported operand type(s)",
            frame_count=8,
            crash_in_user_code=True,
        )
        assert result.f_frame_depth == 0.4  # 7-10 range

    def test_zero_frame_count_gives_neutral_depth(self) -> None:
        result = run_eval._traceback_informativeness_score(
            exception_type="NameError",
            exception_message="name 'x' is not defined",
            frame_count=0,
            crash_in_user_code=False,
        )
        assert result.f_frame_depth == 0.5  # Unknown → neutral

    def test_features_are_populated(self) -> None:
        result = run_eval._traceback_informativeness_score(
            exception_type="SyntaxError",
            exception_message="invalid syntax",
            frame_count=1,
            crash_in_user_code=True,
        )
        assert result.f_type_specificity == 1.0
        assert result.f_message_diagnostic == 1.0
        assert result.f_crash_in_user_code == 1.0
        assert result.f_assertion_quality == 0.5  # neutral for non-assertion
        assert result.f_frame_depth == 1.0


# ===========================================================================
# Unit tests: Extraction
# ===========================================================================


class TestExtractExceptionInfo:
    def test_extract_from_snapshot(self) -> None:
        snapshot = _snapshot_payload("NameError", "name 'x' is not defined")
        exc_type, exc_msg, frame_count, crash_in_user = run_eval._extract_exception_info(
            snapshot=snapshot,
            stdout="",
            stderr="",
        )
        assert exc_type == "NameError"
        assert exc_msg == "name 'x' is not defined"
        assert frame_count == 1
        assert crash_in_user is True

    def test_extract_from_stdout_fallback(self) -> None:
        exc_type, exc_msg, _, _ = run_eval._extract_exception_info(
            snapshot=None,
            stdout="E   NameError: name 'x' is not defined\n",
            stderr="",
        )
        assert exc_type == "NameError"
        assert exc_msg == "name 'x' is not defined"

    def test_extract_from_stderr_fallback(self) -> None:
        exc_type, exc_msg, _, _ = run_eval._extract_exception_info(
            snapshot=None,
            stdout="",
            stderr="NameError: name 'x' is not defined\n",
        )
        assert exc_type == "NameError"
        assert exc_msg == "name 'x' is not defined"

    def test_extract_pytest_format(self) -> None:
        exc_type, exc_msg, _, _ = run_eval._extract_exception_info(
            snapshot=None,
            stdout="E   TypeError: unsupported operand type(s)\n",
            stderr="",
        )
        assert exc_type == "TypeError"
        assert "unsupported" in exc_msg

    def test_extract_returns_unknown_on_empty(self) -> None:
        exc_type, exc_msg, frame_count, crash_in_user = run_eval._extract_exception_info(
            snapshot=None,
            stdout="",
            stderr="",
        )
        assert exc_type == "Unknown"
        assert exc_msg == ""
        assert frame_count == 0
        assert crash_in_user is False

    def test_extract_prefers_snapshot(self) -> None:
        snapshot = _snapshot_payload("SyntaxError", "invalid syntax from snapshot")
        exc_type, exc_msg, _, _ = run_eval._extract_exception_info(
            snapshot=snapshot,
            stdout="E   NameError: name 'y' is not defined\n",
            stderr="",
        )
        assert exc_type == "SyntaxError"
        assert "snapshot" in exc_msg

    def test_extract_strips_module_prefix(self) -> None:
        snapshot = {
            "exception": {"type": "builtins.ValueError", "message": "bad value"},
            "frames": [],
        }
        exc_type, _, _, _ = run_eval._extract_exception_info(
            snapshot=snapshot,
            stdout="",
            stderr="",
        )
        assert exc_type == "ValueError"


# ===========================================================================
# Unit tests: Pre-gate decision
# ===========================================================================


class TestPreGateDecision:
    def test_pregate_skips_syntax_error(self) -> None:
        snapshot = _snapshot_payload("SyntaxError", "invalid syntax")
        result = run_eval._traceback_informativeness_pre_gate(
            snapshot=snapshot,
            stdout="",
            stderr="",
            threshold=0.7,
        )
        assert result.should_skip is True
        assert result.exception_type == "SyntaxError"

    def test_pregate_no_skip_assertion_error(self) -> None:
        snapshot = _snapshot_payload("AssertionError", "")
        result = run_eval._traceback_informativeness_pre_gate(
            snapshot=snapshot,
            stdout="",
            stderr="",
            threshold=0.7,
        )
        assert result.should_skip is False

    def test_pregate_respects_threshold(self) -> None:
        snapshot = _snapshot_payload("NameError", "name 'x' is not defined")
        result = run_eval._traceback_informativeness_pre_gate(
            snapshot=snapshot,
            stdout="",
            stderr="",
            threshold=0.95,
        )
        assert result.should_skip is False
        assert result.score < 0.95

    def test_pregate_returns_features(self) -> None:
        snapshot = _snapshot_payload("IndexError", "list index out of range")
        result = run_eval._traceback_informativeness_pre_gate(
            snapshot=snapshot,
            stdout="",
            stderr="",
            threshold=0.7,
        )
        assert result.f_type_specificity == 0.7
        assert result.f_message_diagnostic == 1.0


# ===========================================================================
# Integration tests
# ===========================================================================


class TestPreGateIntegration:
    def test_adaptive_pregate_skips_stage_a_for_syntax_error(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        case = _make_case(tmp_path)
        workdir = tmp_path / "workdir"
        context_dir = tmp_path / "contexts"
        context_dir.mkdir()
        shutil.copytree(case.path, workdir)
        patcher = _PreGateCapturePatcher()

        monkeypatch.setattr(
            eval_attempt_flow,
            "run_pytest",
            lambda *args, **kwargs: run_eval.RunResult(
                returncode=1,
                duration_sec=0.01,
                stdout="E   SyntaxError: invalid syntax\n",
                stderr="",
            ),
        )
        monkeypatch.setattr(
            eval_attempt_flow,
            "read_latest_snapshot",
            lambda *_args, **_kwargs: (
                _snapshot_payload("SyntaxError", "invalid syntax"),
                None,
            ),
        )

        records = run_eval.run_one_condition(
            case=case,
            condition=run_eval.Condition.LAYERED_ADAPTIVE_LLM,
            workdir=workdir,
            context_dir_base=context_dir,
            patcher=patcher,
            k=1,
            output_format="json_compact",
            run_id="run-pregate-syntax",
            include_files_context=False,
            force_patch_attempt=False,
            adaptive_informativeness_pregate=True,
            adaptive_informativeness_pregate_threshold=0.7,
        )

        assert len(records) >= 1
        assert patcher.request_calls == 0, "Stage-A should NOT have been called"
        meta = records[0]["attempt_metadata"]
        assert meta["adaptive_helper_decision_reason"] == "informativeness_pregate_skip"
        assert meta["adaptive_informativeness_pregate_fired"] is True
        assert meta["adaptive_informativeness_pregate_score"] >= 0.7

    def test_adaptive_pregate_disabled_allows_normal_flow(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        case = _make_case(tmp_path)
        workdir = tmp_path / "workdir"
        context_dir = tmp_path / "contexts"
        context_dir.mkdir()
        shutil.copytree(case.path, workdir)
        patcher = _PreGateCapturePatcher()

        monkeypatch.setattr(
            eval_attempt_flow,
            "run_pytest",
            lambda *args, **kwargs: run_eval.RunResult(
                returncode=1,
                duration_sec=0.01,
                stdout="E   SyntaxError: invalid syntax\n",
                stderr="",
            ),
        )
        monkeypatch.setattr(
            eval_attempt_flow,
            "read_latest_snapshot",
            lambda *_args, **_kwargs: (
                _snapshot_payload("SyntaxError", "invalid syntax"),
                None,
            ),
        )

        records = run_eval.run_one_condition(
            case=case,
            condition=run_eval.Condition.LAYERED_ADAPTIVE_LLM,
            workdir=workdir,
            context_dir_base=context_dir,
            patcher=patcher,
            k=1,
            output_format="json_compact",
            run_id="run-pregate-disabled",
            include_files_context=False,
            force_patch_attempt=False,
            adaptive_informativeness_pregate=False,  # disabled (default)
        )

        assert len(records) >= 1
        # When pregate is disabled, Stage-A should proceed normally
        assert patcher.request_calls >= 1, "Stage-A should have been called with pregate disabled"
        meta = records[0]["attempt_metadata"]
        assert meta["adaptive_informativeness_pregate_fired"] is False

    def test_adaptive_pregate_does_not_fire_for_assertion_error(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        case = _make_case(tmp_path)
        workdir = tmp_path / "workdir"
        context_dir = tmp_path / "contexts"
        context_dir.mkdir()
        shutil.copytree(case.path, workdir)
        patcher = _PreGateCapturePatcher()

        monkeypatch.setattr(
            eval_attempt_flow,
            "run_pytest",
            lambda *args, **kwargs: run_eval.RunResult(
                returncode=1,
                duration_sec=0.01,
                stdout="E   AssertionError: expected 2 got 1\n",
                stderr="",
            ),
        )
        monkeypatch.setattr(
            eval_attempt_flow,
            "read_latest_snapshot",
            lambda *_args, **_kwargs: (
                _snapshot_payload("AssertionError", "expected 2 got 1"),
                None,
            ),
        )

        records = run_eval.run_one_condition(
            case=case,
            condition=run_eval.Condition.LAYERED_ADAPTIVE_LLM,
            workdir=workdir,
            context_dir_base=context_dir,
            patcher=patcher,
            k=1,
            output_format="json_compact",
            run_id="run-pregate-assertion",
            include_files_context=False,
            force_patch_attempt=False,
            adaptive_informativeness_pregate=True,
            adaptive_informativeness_pregate_threshold=0.7,
        )

        assert len(records) >= 1
        # AssertionError scores low, so pregate should NOT fire and Stage-A should proceed
        assert patcher.request_calls >= 1, "Stage-A should be called for AssertionError"
        meta = records[0]["attempt_metadata"]
        assert meta["adaptive_informativeness_pregate_fired"] is False
        assert meta["adaptive_helper_decision_reason"] != "informativeness_pregate_skip"
