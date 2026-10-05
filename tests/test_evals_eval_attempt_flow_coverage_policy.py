"""Coverage-boost tests for evals/eval_attempt_flow.py — part 3: adaptive policy utilities.

Tests _adaptive_gate_should_invoke_helper, _adaptive_should_stop_for_no_progress,
_apply_no_progress_token_cap_candidate, _mark_adaptive_progress_stop,
_adaptive_pipeline_stage_for_outcome, _adaptive_pipeline_stage_before,
and integration test for BaselinePatcher import failure path.
"""

from __future__ import annotations

import builtins
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pytest

from evals import eval_attempt_flow


def _make_run_result(**overrides: Any) -> Any:
    from evals.run_eval import RunResult

    defaults = {
        "returncode": 1,
        "duration_sec": 0.1,
        "stdout": "FAILED",
        "stderr": "",
        "timed_out": False,
    }
    defaults.update(overrides)
    return RunResult(**defaults)


def _make_case(tmp_path: Path, case_id: str = "cov3") -> Any:
    from evals.run_eval import CaseSpec

    case_dir = tmp_path / "cases" / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    return CaseSpec(
        case_id=case_id,
        path=case_dir,
        category="unit_test",
        difficulty="easy",
        snapshot_dependency="none",
        expected_exception="AssertionError",
        tags=[],
        python_args=["-m", "pytest", "-q"],
        timeout_sec=10,
    )


# ---------------------------------------------------------------------------
# _adaptive_gate_should_invoke_helper
# ---------------------------------------------------------------------------


class TestAdaptiveGateShouldInvokeHelper:
    def test_timeout_invokes(self) -> None:
        s, r = eval_attempt_flow._adaptive_gate_should_invoke_helper(
            policy="balanced",
            attempt=1,
            timed_out=True,
            signature_repeated=False,
            stagnation_streak=0,
            stagnation_window=3,
        )
        assert s is True and r == "gate_timeout"

    def test_aggressive_invokes(self) -> None:
        s, r = eval_attempt_flow._adaptive_gate_should_invoke_helper(
            policy="aggressive",
            attempt=1,
            timed_out=False,
            signature_repeated=False,
            stagnation_streak=0,
            stagnation_window=3,
        )
        assert s is True and r == "gate_aggressive"

    def test_conservative_waiting(self) -> None:
        s, r = eval_attempt_flow._adaptive_gate_should_invoke_helper(
            policy="conservative",
            attempt=2,
            timed_out=False,
            signature_repeated=True,
            stagnation_streak=1,
            stagnation_window=3,
        )
        assert s is False and r == "gate_conservative_waiting"

    def test_conservative_invokes(self) -> None:
        s, r = eval_attempt_flow._adaptive_gate_should_invoke_helper(
            policy="conservative",
            attempt=5,
            timed_out=False,
            signature_repeated=True,
            stagnation_streak=5,
            stagnation_window=3,
        )
        assert s is True and r == "gate_conservative_repeated_signature"

    def test_balanced_waiting(self) -> None:
        s, r = eval_attempt_flow._adaptive_gate_should_invoke_helper(
            policy="balanced",
            attempt=1,
            timed_out=False,
            signature_repeated=False,
            stagnation_streak=0,
            stagnation_window=3,
        )
        assert s is False and r == "gate_balanced_waiting"

    def test_balanced_invokes(self) -> None:
        s, r = eval_attempt_flow._adaptive_gate_should_invoke_helper(
            policy="balanced",
            attempt=3,
            timed_out=False,
            signature_repeated=True,
            stagnation_streak=3,
            stagnation_window=3,
        )
        assert s is True and r == "gate_balanced_repeated_signature"


# ---------------------------------------------------------------------------
# _adaptive_should_stop_for_no_progress
# ---------------------------------------------------------------------------


class TestAdaptiveShouldStopForNoProgress:
    def test_below_window(self) -> None:
        s, r = eval_attempt_flow._adaptive_should_stop_for_no_progress(
            no_progress_streak=1,
            stop_window=5,
            helper_budget_exhausted=False,
            helper_saturation_guarded=False,
            helper_decision_reason="",
        )
        assert s is False and r == "streak_below_window"

    def test_budget_exhausted(self) -> None:
        s, r = eval_attempt_flow._adaptive_should_stop_for_no_progress(
            no_progress_streak=5,
            stop_window=3,
            helper_budget_exhausted=True,
            helper_saturation_guarded=False,
            helper_decision_reason="",
        )
        assert s is True and r == "helper_budget_exhausted"

    def test_saturation_guarded(self) -> None:
        s, r = eval_attempt_flow._adaptive_should_stop_for_no_progress(
            no_progress_streak=5,
            stop_window=3,
            helper_budget_exhausted=False,
            helper_saturation_guarded=True,
            helper_decision_reason="",
        )
        assert s is True and r == "helper_saturation_guard"

    def test_llm_declined(self) -> None:
        s, _r = eval_attempt_flow._adaptive_should_stop_for_no_progress(
            no_progress_streak=5,
            stop_window=3,
            helper_budget_exhausted=False,
            helper_saturation_guarded=False,
            helper_decision_reason="llm_declined_helper",
        )
        assert s is True

    def test_waiting(self) -> None:
        s, r = eval_attempt_flow._adaptive_should_stop_for_no_progress(
            no_progress_streak=5,
            stop_window=3,
            helper_budget_exhausted=False,
            helper_saturation_guarded=False,
            helper_decision_reason="other",
        )
        assert s is False and r == "waiting_for_more_signal"


# ---------------------------------------------------------------------------
# _apply_no_progress_token_cap_candidate
# ---------------------------------------------------------------------------


class TestApplyNoProgressTokenCapCandidate:
    def test_not_adaptive(self) -> None:
        s, _, _, h = eval_attempt_flow._apply_no_progress_token_cap_candidate(
            is_adaptive_condition=False,
            adaptive_progress_made=False,
            adaptive_no_progress_token_sum=100,
            attempt_tokens_total=50,
            adaptive_no_progress_token_cap=200,
            adaptive_progress_stop_candidate=False,
            adaptive_progress_stop_reason="",
        )
        assert s is False and h is False

    def test_progress_resets(self) -> None:
        _, _, ns, _ = eval_attempt_flow._apply_no_progress_token_cap_candidate(
            is_adaptive_condition=True,
            adaptive_progress_made=True,
            adaptive_no_progress_token_sum=100,
            attempt_tokens_total=50,
            adaptive_no_progress_token_cap=200,
            adaptive_progress_stop_candidate=False,
            adaptive_progress_stop_reason="",
        )
        assert ns == 0

    def test_cap_hit(self) -> None:
        s, r, _, h = eval_attempt_flow._apply_no_progress_token_cap_candidate(
            is_adaptive_condition=True,
            adaptive_progress_made=False,
            adaptive_no_progress_token_sum=150,
            attempt_tokens_total=100,
            adaptive_no_progress_token_cap=200,
            adaptive_progress_stop_candidate=False,
            adaptive_progress_stop_reason="",
        )
        assert s is True and r == "no_progress_token_cap" and h is True

    def test_below_cap(self) -> None:
        _, _, ns, h = eval_attempt_flow._apply_no_progress_token_cap_candidate(
            is_adaptive_condition=True,
            adaptive_progress_made=False,
            adaptive_no_progress_token_sum=50,
            attempt_tokens_total=50,
            adaptive_no_progress_token_cap=200,
            adaptive_progress_stop_candidate=False,
            adaptive_progress_stop_reason="",
        )
        assert ns == 100 and h is False


# ---------------------------------------------------------------------------
# _mark_adaptive_progress_stop
# ---------------------------------------------------------------------------


class TestMarkAdaptiveProgressStop:
    def test_triggered(self) -> None:
        am: dict[str, Any] = {}
        pm: dict[str, Any] = {}
        t = eval_attempt_flow._mark_adaptive_progress_stop(
            attempt_metadata=am,
            patch_meta=pm,
            is_adaptive_condition=True,
            adaptive_progress_stop_candidate=True,
            adaptive_progress_stop_reason="cap",
            note="tests_still_failing",
        )
        assert t is True and pm["adaptive_progress_stop_note"] == "tests_still_failing"

    def test_not_adaptive(self) -> None:
        am: dict[str, Any] = {}
        pm: dict[str, Any] = {}
        assert (
            eval_attempt_flow._mark_adaptive_progress_stop(
                attempt_metadata=am,
                patch_meta=pm,
                is_adaptive_condition=False,
                adaptive_progress_stop_candidate=True,
                adaptive_progress_stop_reason="cap",
                note="tests_still_failing",
            )
            is False
        )

    def test_not_candidate(self) -> None:
        am: dict[str, Any] = {}
        pm: dict[str, Any] = {}
        assert (
            eval_attempt_flow._mark_adaptive_progress_stop(
                attempt_metadata=am,
                patch_meta=pm,
                is_adaptive_condition=True,
                adaptive_progress_stop_candidate=False,
                adaptive_progress_stop_reason="",
                note="tests_still_failing",
            )
            is False
        )

    def test_success_note(self) -> None:
        am: dict[str, Any] = {}
        pm: dict[str, Any] = {}
        assert (
            eval_attempt_flow._mark_adaptive_progress_stop(
                attempt_metadata=am,
                patch_meta=pm,
                is_adaptive_condition=True,
                adaptive_progress_stop_candidate=True,
                adaptive_progress_stop_reason="cap",
                note="success",
            )
            is False
        )


# ---------------------------------------------------------------------------
# _adaptive_pipeline_stage_for_outcome / _adaptive_pipeline_stage_before
# ---------------------------------------------------------------------------


class TestAdaptivePipelineStages:
    def test_success(self) -> None:
        assert eval_attempt_flow._adaptive_pipeline_stage_for_outcome(
            note="success", success=True
        ) == ("success", 4)

    def test_no_patch(self) -> None:
        assert eval_attempt_flow._adaptive_pipeline_stage_for_outcome(
            note="no_patch_proposed", success=False
        ) == ("request_or_no_patch", 0)

    def test_patch_fail(self) -> None:
        assert eval_attempt_flow._adaptive_pipeline_stage_for_outcome(
            note="patch_apply_failed", success=False
        ) == ("pretest_patch_failure", 1)

    def test_verifier_rejected(self) -> None:
        assert eval_attempt_flow._adaptive_pipeline_stage_for_outcome(
            note="patch_verifier_rejected", success=False
        ) == ("verifier_rejected", 2)

    def test_tests_failing(self) -> None:
        assert eval_attempt_flow._adaptive_pipeline_stage_for_outcome(
            note="tests_still_failing", success=False
        ) == ("tests_still_failing", 3)

    def test_forced_noop(self) -> None:
        assert eval_attempt_flow._adaptive_pipeline_stage_for_outcome(
            note="forced_noop_patch", success=False
        ) == ("pretest_patch_failure", 1)

    def test_before_with_previous(self) -> None:
        s, _ = eval_attempt_flow._adaptive_pipeline_stage_before(
            previous_failure_context={"note": "no_patch_proposed"}, current_run=None
        )
        assert s == "request_or_no_patch"

    def test_before_no_run(self) -> None:
        s, _ = eval_attempt_flow._adaptive_pipeline_stage_before(
            previous_failure_context=None, current_run=None
        )
        assert s == "tests_still_failing"

    def test_before_timed_out(self) -> None:
        s, _ = eval_attempt_flow._adaptive_pipeline_stage_before(
            previous_failure_context=None, current_run=_make_run_result(timed_out=True)
        )
        assert s == "tests_still_failing"

    def test_before_success(self) -> None:
        s, _ = eval_attempt_flow._adaptive_pipeline_stage_before(
            previous_failure_context=None, current_run=_make_run_result(returncode=0)
        )
        assert s == "success"


# ---------------------------------------------------------------------------
# Integration: BaselinePatcher import failure (lines 1210-1211)
# ---------------------------------------------------------------------------


class TestBaselinePatcherImportFailure:
    def test_import_failure_falls_back(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        original_import = builtins.__import__

        def _patched_import(name: str, *args: Any, **kwargs: Any) -> Any:
            if name == "evals.patchers.baseline_patcher":
                raise ImportError("no baseline patcher")
            return original_import(name, *args, **kwargs)

        case = _make_case(tmp_path, "import_fail")
        workdir = tmp_path / "workdirs" / "import_fail" / "base_repair"
        workdir.mkdir(parents=True, exist_ok=True)
        context_dir = tmp_path / "contexts" / "import_fail" / "base_repair"
        context_dir.mkdir(parents=True, exist_ok=True)

        monkeypatch.setattr(eval_attempt_flow, "run_pytest", lambda *a, **kw: _make_run_result())

        stub_patcher = SimpleNamespace(
            name="custom_patcher",
            propose_patch=lambda *a, **kw: SimpleNamespace(
                diff="--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-pass\n+return 1\n",
                meta={"status": "ok"},
            ),
        )

        monkeypatch.setattr(builtins, "__import__", _patched_import)
        try:
            from evals.eval_types import RunConditionInputs
            from evals.run_eval import Condition

            result = eval_attempt_flow.run_one_condition_impl(
                case=case,
                condition=Condition.BASE_REPAIR,
                workdir=workdir,
                context_dir_base=context_dir,
                inputs=RunConditionInputs(
                    k=1,
                    output_format="diff",
                    run_id="test-run-id",
                    include_files_context=False,
                    force_patch_attempt=False,
                    run_metadata={"run_id": "test-run-id"},
                    patcher=stub_patcher,
                ),
            )
            assert isinstance(result, list)
        finally:
            monkeypatch.setattr(builtins, "__import__", original_import)
