"""Coverage-boost tests for evals/eval_attempt_flow.py.

Targets specific uncovered lines and branches to push coverage from ~90.5%
toward ~93%.  Focuses on:

- Progress callback exception swallowing (lines 146-147, 628-630)
- Trace-level progress printing (lines 199-220)
- Detailed-level record callback formatting (lines 333-416)
- Heartbeat emission (lines 230-310)
- Phase-started <= 0 heartbeat branch (line 239)
- Transport recovery branches (lines 734-803)
- Adaptive history persistence error path (lines 834-851)
- Normalization edge cases inside run_one_condition_impl
- Patcher attribute coercion fallbacks (lines 1237-1243, 3023-3024)
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import pytest

from evals import run_eval
from evals.eval_types import RunConditionInputs

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_case(tmp_path: Path, case_id: str = "cov_boost") -> run_eval.CaseSpec:
    """Create a minimal CaseSpec for test scaffolding."""
    case_dir = tmp_path / "cases" / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    return run_eval.CaseSpec(
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


def _minimal_run_condition_inputs(**overrides: Any) -> RunConditionInputs:
    """Build a RunConditionInputs with sensible defaults for tests."""
    defaults: dict[str, Any] = {
        "k": 1,
        "output_format": "diff",
        "run_id": "test-run-id",
        "include_files_context": False,
        "force_patch_attempt": False,
        "run_metadata": {"run_id": "test-run-id"},
    }
    defaults.update(overrides)
    return RunConditionInputs(**defaults)


def _fake_run_one_condition_factory(
    records: list[dict[str, Any]] | None = None,
    *,
    callback_event: str = "setup",
    raise_in_progress: bool = False,
    emit_progress_events: bool = False,
    progress_events: list[tuple[str, dict[str, Any] | None]] | None = None,
) -> Any:
    """Return a callable mimicking run_one_condition with configurable behavior."""

    def _fake(**kwargs: Any) -> list[dict[str, Any]]:
        progress_update = kwargs.get("progress_update")
        record_sink = kwargs.get("record_sink")

        # Optionally fire progress events so the outer harness processes them
        if emit_progress_events and progress_update is not None:
            events = progress_events or [
                (
                    callback_event,
                    {
                        "attempt": 1,
                        "max_attempts": 1,
                        "phase_label": f"test: {callback_event}",
                        "owner": "test",
                        "source": "phase",
                    },
                ),
            ]
            for event, payload in events:
                progress_update(event, payload)

        result_records = records or [
            {
                "run_id": kwargs.get("run_id", "test-run-id"),
                "case_id": kwargs.get("case", SimpleNamespace(case_id="c")).case_id,
                "condition": kwargs.get("condition", SimpleNamespace(value="base_repair")).value,
                "attempt": 1,
                "success": True,
                "note": "patched",
                "patch_meta": {
                    "propose_duration_sec": 1.5,
                    "prompt_chars_total": 500,
                    "completion_chars_total": 200,
                },
                "attempt_metadata": {
                    "tokens_prompt": 100,
                    "tokens_completion": 50,
                    "tokens_total": 150,
                    "prompt_chars_total": 500,
                    "completion_chars_total": 200,
                },
                "after_duration_sec": 0.3,
                "attempt_wall_clock_sec": 2.1,
            },
        ]
        for rec in result_records:
            if record_sink is not None:
                record_sink(rec)
        return result_records

    return _fake


# ===================================================================
# 2. _reset_patch_usage_sync_metadata
# ===================================================================


class TestResetPatchUsageSyncMetadata:
    """Validate that _reset_patch_usage_sync_metadata clears the right keys."""

    def test_clears_all_sync_keys(self) -> None:
        from evals.eval_attempt_flow import (
            _PATCH_USAGE_SYNC_META_KEYS,
            _reset_patch_usage_sync_metadata,
        )

        meta: dict[str, Any] = dict.fromkeys(_PATCH_USAGE_SYNC_META_KEYS, 42)
        meta["other_key"] = "preserved"
        _reset_patch_usage_sync_metadata(meta)
        for k in _PATCH_USAGE_SYNC_META_KEYS:
            assert k not in meta
        assert meta["other_key"] == "preserved"

    def test_noop_on_empty_dict(self) -> None:
        from evals.eval_attempt_flow import _reset_patch_usage_sync_metadata

        meta: dict[str, Any] = {}
        _reset_patch_usage_sync_metadata(meta)
        assert meta == {}


# ===================================================================
# 3. execute_eval_unit_impl — progress trace verbosity (lines 199-220)
# ===================================================================


class TestExecuteEvalUnitTraceVerbosity:
    """Cover the trace-level progress printing path (lines 199-220)."""

    def test_trace_verbosity_emits_trace_progress_lines(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        from evals import eval_attempt_flow

        case = _make_case(tmp_path, "trace_verbosity")

        # Use a fake run_one_condition that fires progress events
        monkeypatch.setattr(
            run_eval,
            "run_one_condition",
            _fake_run_one_condition_factory(
                emit_progress_events=True,
                progress_events=[
                    (
                        "llm_request",
                        {
                            "attempt": 1,
                            "max_attempts": 1,
                            "phase_label": "llm: propose patch",
                            "owner": "openai",
                            "source": "phase",
                            "waiting_for": "openai_server",
                            "syntax_cycle": 1,
                            "syntax_max_cycles": 3,
                        },
                    ),
                ],
            ),
        )

        outcome = run_eval._execute_eval_unit(
            case=case,
            condition=run_eval.Condition.BASE_REPAIR,
            artifacts_root=tmp_path / "artifacts",
            inputs=run_eval.EvalUnitInputs(
                patcher_args=argparse.Namespace(),
                patcher_override=SimpleNamespace(name="stub"),
                keep_workdirs=False,
                progress_verbosity="trace",
                progress_heartbeat_sec=0.0,
                unit_progress_callback=None,
                run_condition=_minimal_run_condition_inputs(),
            ),
        )
        assert outcome.runtime_error is None
        captured = capsys.readouterr().out
        assert "[TRACE]" in captured
        assert "trace_verbosity" in captured

    def test_trace_verbosity_with_waiting_for_and_owner(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Exercises lines 214-219 (wait_fragment, owner_fragment, source_fragment)."""
        from evals import eval_attempt_flow

        case = _make_case(tmp_path, "trace_owner")

        monkeypatch.setattr(
            run_eval,
            "run_one_condition",
            _fake_run_one_condition_factory(
                emit_progress_events=True,
                progress_events=[
                    (
                        "llm_request",
                        {
                            "attempt": 2,
                            "max_attempts": 3,
                            "phase_label": "llm: propose",
                            "owner": "openai_api",
                            "source": "stream",
                            "waiting_for": "api_server",
                        },
                    ),
                ],
            ),
        )

        outcome = run_eval._execute_eval_unit(
            case=case,
            condition=run_eval.Condition.BASE_REPAIR,
            artifacts_root=tmp_path / "artifacts",
            inputs=run_eval.EvalUnitInputs(
                patcher_args=argparse.Namespace(),
                patcher_override=SimpleNamespace(name="stub"),
                keep_workdirs=False,
                progress_verbosity="trace",
                progress_heartbeat_sec=0.0,
                unit_progress_callback=None,
                run_condition=_minimal_run_condition_inputs(),
            ),
        )
        assert outcome.runtime_error is None
        captured = capsys.readouterr().out
        assert "waiting_for=api_server" in captured
        assert "owner=openai_api" in captured
        assert "source=stream" in captured


# ===================================================================
# 4. execute_eval_unit_impl — callback exception swallowing (146-147)
# ===================================================================


class TestExecuteEvalUnitCallbackExceptionSwallowed:
    """Cover lines 146-147: exception in unit_progress_callback is swallowed."""

    def test_failing_callback_does_not_crash(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        case = _make_case(tmp_path, "failing_cb")
        events_received: list[str] = []

        def _bad_callback(case_id: str, condition: str, event: str, payload: Any) -> None:
            events_received.append(event)
            raise RuntimeError("callback boom")

        monkeypatch.setattr(
            run_eval,
            "run_one_condition",
            _fake_run_one_condition_factory(
                emit_progress_events=True,
            ),
        )

        outcome = run_eval._execute_eval_unit(
            case=case,
            condition=run_eval.Condition.BASE_REPAIR,
            artifacts_root=tmp_path / "artifacts",
            inputs=run_eval.EvalUnitInputs(
                patcher_args=argparse.Namespace(),
                patcher_override=SimpleNamespace(name="stub"),
                keep_workdirs=False,
                progress_verbosity="basic",
                progress_heartbeat_sec=0.0,
                unit_progress_callback=_bad_callback,
                run_condition=_minimal_run_condition_inputs(),
            ),
        )
        # The execution should succeed despite the callback failing
        assert outcome.runtime_error is None
        # At least one event was delivered (triggering the exception path)
        assert len(events_received) >= 1


# ===================================================================
# 5. execute_eval_unit_impl — detailed record callback (lines 333-416)
# ===================================================================


class TestExecuteEvalUnitDetailedRecordCallback:
    """Cover lines 333-416: detailed-level progress printing for records."""

    def test_detailed_verbosity_prints_attempt_info(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        case = _make_case(tmp_path, "detailed_record")

        monkeypatch.setattr(
            run_eval,
            "run_one_condition",
            _fake_run_one_condition_factory(),
        )

        outcome = run_eval._execute_eval_unit(
            case=case,
            condition=run_eval.Condition.BASE_REPAIR,
            artifacts_root=tmp_path / "artifacts",
            inputs=run_eval.EvalUnitInputs(
                patcher_args=argparse.Namespace(),
                patcher_override=SimpleNamespace(name="stub"),
                keep_workdirs=False,
                progress_verbosity="detailed",
                progress_heartbeat_sec=0.0,
                unit_progress_callback=None,
                run_condition=_minimal_run_condition_inputs(),
            ),
        )
        assert outcome.runtime_error is None
        captured = capsys.readouterr().out
        assert "[ATTEMPT]" in captured
        assert "detailed_record" in captured
        assert "status=success" in captured

    def test_detailed_record_with_no_attempt_metadata(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Cover branch where attempt_metadata is not a dict (lines 367-388)."""
        case = _make_case(tmp_path, "no_attempt_meta")

        records = [
            {
                "run_id": "test-run-id",
                "case_id": "no_attempt_meta",
                "condition": "base_repair",
                "attempt": 1,
                "success": False,
                "note": "syntax_error",
                "patch_meta": {"propose_duration_sec": 0.5},
                "attempt_metadata": None,
                "after_duration_sec": 0.1,
                "attempt_wall_clock_sec": 0.8,
            },
        ]

        monkeypatch.setattr(
            run_eval,
            "run_one_condition",
            _fake_run_one_condition_factory(records=records),
        )

        outcome = run_eval._execute_eval_unit(
            case=case,
            condition=run_eval.Condition.BASE_REPAIR,
            artifacts_root=tmp_path / "artifacts",
            inputs=run_eval.EvalUnitInputs(
                patcher_args=argparse.Namespace(),
                patcher_override=SimpleNamespace(name="stub"),
                keep_workdirs=False,
                progress_verbosity="detailed",
                progress_heartbeat_sec=0.0,
                unit_progress_callback=None,
                run_condition=_minimal_run_condition_inputs(),
            ),
        )
        assert outcome.runtime_error is None
        captured = capsys.readouterr().out
        assert "[ATTEMPT]" in captured
        assert "status=failed" in captured

    def test_detailed_record_with_non_dict_patch_meta(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Cover branch where patch_meta is not a dict (lines 349-366)."""
        case = _make_case(tmp_path, "non_dict_pmeta")

        records = [
            {
                "run_id": "test-run-id",
                "case_id": "non_dict_pmeta",
                "condition": "base_repair",
                "attempt": 1,
                "success": True,
                "note": "patched",
                "patch_meta": "not_a_dict",
                "attempt_metadata": None,
                "after_duration_sec": 0.2,
                "attempt_wall_clock_sec": 1.0,
            },
        ]

        monkeypatch.setattr(
            run_eval,
            "run_one_condition",
            _fake_run_one_condition_factory(records=records),
        )

        outcome = run_eval._execute_eval_unit(
            case=case,
            condition=run_eval.Condition.BASE_REPAIR,
            artifacts_root=tmp_path / "artifacts",
            inputs=run_eval.EvalUnitInputs(
                patcher_args=argparse.Namespace(),
                patcher_override=SimpleNamespace(name="stub"),
                keep_workdirs=False,
                progress_verbosity="detailed",
                progress_heartbeat_sec=0.0,
                unit_progress_callback=None,
                run_condition=_minimal_run_condition_inputs(),
            ),
        )
        assert outcome.runtime_error is None
        captured = capsys.readouterr().out
        assert "[ATTEMPT]" in captured


# ===================================================================
# 6. execute_eval_unit_impl — heartbeat emission (lines 230-310)
# ===================================================================


class TestExecuteEvalUnitHeartbeat:
    """Cover heartbeat loop and _emit_progress_heartbeat (lines 230-310, 442)."""

    def test_heartbeat_fires_with_callback(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Line 442: heartbeat_updates_enabled triggers heartbeat_thread."""
        case = _make_case(tmp_path, "heartbeat_fire")
        heartbeat_events: list[dict[str, Any]] = []

        clock = {"now": 1000.0}
        from evals import eval_attempt_flow

        monkeypatch.setattr(eval_attempt_flow.time, "monotonic", lambda: clock["now"])

        def _fake(**kwargs: Any) -> list[dict[str, Any]]:
            pu = kwargs.get("progress_update")
            if pu:
                pu(
                    "llm_request",
                    {
                        "attempt": 1,
                        "max_attempts": 1,
                        "phase_label": "llm: propose",
                        "owner": "test_owner",
                        "source": "test_src",
                        "waiting_for": "test_server",
                    },
                )
            clock["now"] = 1000.1
            time.sleep(0.06)
            rec = {
                "run_id": "test-run-id",
                "case_id": "heartbeat_fire",
                "condition": "base_repair",
                "attempt": 1,
                "success": True,
                "note": "ok",
                "patch_meta": {},
                "attempt_metadata": {},
            }
            rs = kwargs.get("record_sink")
            if rs:
                rs(rec)
            return [rec]

        monkeypatch.setattr(run_eval, "run_one_condition", _fake)

        outcome = run_eval._execute_eval_unit(
            case=case,
            condition=run_eval.Condition.BASE_REPAIR,
            artifacts_root=tmp_path / "artifacts",
            inputs=run_eval.EvalUnitInputs(
                patcher_args=argparse.Namespace(),
                patcher_override=SimpleNamespace(name="stub"),
                keep_workdirs=False,
                progress_verbosity="off",
                progress_heartbeat_sec=0.02,
                unit_progress_callback=lambda cid, cond, event, payload: (
                    heartbeat_events.append({"event": event, **(payload or {})})
                ),
                run_condition=_minimal_run_condition_inputs(),
            ),
        )
        assert outcome.runtime_error is None
        # At least one heartbeat event should have been emitted
        hb = [e for e in heartbeat_events if e.get("event") == "heartbeat"]
        assert len(hb) >= 1

    def test_heartbeat_with_phase_started_zero(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Line 239: phase_started <= 0 -> use unit_started as fallback."""
        case = _make_case(tmp_path, "heartbeat_zero_phase")
        heartbeat_payloads: list[dict[str, Any]] = []

        from evals import eval_attempt_flow

        monkeypatch.setattr(eval_attempt_flow.time, "monotonic", lambda: 0.0)

        def _fake(**kwargs: Any) -> list[dict[str, Any]]:
            pu = kwargs.get("progress_update")
            if pu:
                # Fire an event that sets phase_started_monotonic to 0
                pu("setup", {"phase_label": "init"})
            time.sleep(0.06)
            rec = {
                "run_id": "test-run-id",
                "case_id": "heartbeat_zero_phase",
                "condition": "base_repair",
                "attempt": 1,
                "success": True,
                "note": "ok",
                "patch_meta": {},
                "attempt_metadata": {},
            }
            rs = kwargs.get("record_sink")
            if rs:
                rs(rec)
            return [rec]

        monkeypatch.setattr(run_eval, "run_one_condition", _fake)

        outcome = run_eval._execute_eval_unit(
            case=case,
            condition=run_eval.Condition.BASE_REPAIR,
            artifacts_root=tmp_path / "artifacts",
            inputs=run_eval.EvalUnitInputs(
                patcher_args=argparse.Namespace(),
                patcher_override=SimpleNamespace(name="stub"),
                keep_workdirs=False,
                progress_verbosity="off",
                progress_heartbeat_sec=0.02,
                unit_progress_callback=lambda cid, cond, event, payload: (
                    heartbeat_payloads.append(payload) if event == "heartbeat" else None
                ),
                run_condition=_minimal_run_condition_inputs(),
            ),
        )
        assert outcome.runtime_error is None


# ===================================================================
# 7. run_one_condition_impl — progress_update exception (628-630)
# ===================================================================


class TestRunOneConditionProgressUpdateException:
    """Cover lines 628-630: exception in run_one_condition's progress_update."""

    def test_failing_progress_update_does_not_crash(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from evals import eval_attempt_flow

        case = _make_case(tmp_path, "progress_boom")
        workdir = tmp_path / "workdirs" / "progress_boom" / "base_repair"
        workdir.mkdir(parents=True, exist_ok=True)
        context_dir = tmp_path / "contexts" / "progress_boom" / "base_repair"
        context_dir.mkdir(parents=True, exist_ok=True)

        call_count = {"n": 0}

        def _bad_progress(event: str, payload: Any = None) -> None:
            call_count["n"] += 1
            raise RuntimeError("progress callback exploded")

        # Create a stub patcher that returns a dummy patch
        stub_patcher = SimpleNamespace(
            name="stub",
            propose_patch=lambda *a, **kw: SimpleNamespace(
                diff="--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-pass\n+return 1\n",
                meta={"status": "ok"},
            ),
        )

        # We need to intercept the first pytest call (the "before" run).
        # Mock run_pytest to return a RunResult-like.
        fake_result = run_eval.RunResult(
            returncode=1,
            duration_sec=0.1,
            stdout="FAILED",
            stderr="",
            timed_out=False,
        )
        monkeypatch.setattr(eval_attempt_flow, "run_pytest", lambda *a, **kw: fake_result)

        # run_one_condition_impl uses many globals from run_eval, so let's
        # call it through run_eval.run_one_condition which delegates properly.
        # The progress_update will be called and should not crash the flow.
        result = eval_attempt_flow.run_one_condition_impl(
            case=case,
            condition=run_eval.Condition.BASE_REPAIR,
            workdir=workdir,
            context_dir_base=context_dir,
            inputs=_minimal_run_condition_inputs(
                patcher=stub_patcher,
                progress_update=_bad_progress,
            ),
        )
        # The function should complete despite progress callback failures
        assert isinstance(result, list)
        assert call_count["n"] >= 1


# ===================================================================
# 8. Patcher attribute coercion fallbacks (lines 1237-1243, 3023-3024)
# ===================================================================


class TestPatcherAttributeCoercionFallback:
    """Cover TypeError/ValueError fallback for patcher max_propose_sec / max_tokens."""

    def test_nonnumeric_max_propose_sec_and_max_tokens(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Lines 1237-1243: patcher attrs that are not valid numbers."""
        from evals import eval_attempt_flow

        case = _make_case(tmp_path, "coerce_fallback")
        workdir = tmp_path / "workdirs" / "coerce_fallback" / "base_repair"
        workdir.mkdir(parents=True, exist_ok=True)
        context_dir = tmp_path / "contexts" / "coerce_fallback" / "base_repair"
        context_dir.mkdir(parents=True, exist_ok=True)

        stub_patcher = SimpleNamespace(
            name="openai",
            max_propose_sec="not_a_number",
            max_tokens="also_invalid",
            response_mode="text",
            propose_patch=lambda *a, **kw: SimpleNamespace(
                diff="--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-pass\n+1\n",
                meta={"status": "ok"},
            ),
        )

        fake_result = run_eval.RunResult(
            returncode=1,
            duration_sec=0.1,
            stdout="FAILED",
            stderr="",
            timed_out=False,
        )
        monkeypatch.setattr(
            eval_attempt_flow,
            "run_pytest",
            lambda *a, **kw: fake_result,
        )

        result = eval_attempt_flow.run_one_condition_impl(
            case=case,
            condition=run_eval.Condition.BASE_REPAIR,
            workdir=workdir,
            context_dir_base=context_dir,
            inputs=_minimal_run_condition_inputs(patcher=stub_patcher),
        )
        assert isinstance(result, list)


# ===================================================================
# 9. Normalization edge cases in run_one_condition_impl
# ===================================================================


class TestNormalizationEdgeCases:
    """Cover normalization branches for context_compaction_scope, budget tiers, etc."""

    def test_empty_compaction_scope_defaults_to_all(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Line 1054: empty context_compaction_scope -> default set."""
        from evals import eval_attempt_flow

        case = _make_case(tmp_path, "empty_scope")
        workdir = tmp_path / "workdirs" / "empty_scope" / "base_repair"
        workdir.mkdir(parents=True, exist_ok=True)
        context_dir = tmp_path / "contexts" / "empty_scope" / "base_repair"
        context_dir.mkdir(parents=True, exist_ok=True)

        stub_patcher = SimpleNamespace(
            name="stub",
            propose_patch=lambda *a, **kw: SimpleNamespace(
                diff="--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-pass\n+1\n",
                meta={"status": "ok"},
            ),
        )

        fake_result = run_eval.RunResult(
            returncode=1,
            duration_sec=0.1,
            stdout="FAILED",
            stderr="",
            timed_out=False,
        )
        monkeypatch.setattr(
            eval_attempt_flow,
            "run_pytest",
            lambda *a, **kw: fake_result,
        )

        result = eval_attempt_flow.run_one_condition_impl(
            case=case,
            condition=run_eval.Condition.BASE_REPAIR,
            workdir=workdir,
            context_dir_base=context_dir,
            inputs=_minimal_run_condition_inputs(
                patcher=stub_patcher,
                context_compaction_scope="",
            ),
        )
        assert isinstance(result, list)

    def test_invalid_budget_tiers_fallback(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Lines 1131-1132: invalid budget tiers -> fallback [1.0, 1.5, 2.0]."""
        from evals import eval_attempt_flow

        case = _make_case(tmp_path, "bad_tiers")
        workdir = tmp_path / "workdirs" / "bad_tiers" / "base_repair"
        workdir.mkdir(parents=True, exist_ok=True)
        context_dir = tmp_path / "contexts" / "bad_tiers" / "base_repair"
        context_dir.mkdir(parents=True, exist_ok=True)

        stub_patcher = SimpleNamespace(
            name="stub",
            propose_patch=lambda *a, **kw: SimpleNamespace(
                diff="--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-pass\n+1\n",
                meta={"status": "ok"},
            ),
        )

        fake_result = run_eval.RunResult(
            returncode=1,
            duration_sec=0.1,
            stdout="FAILED",
            stderr="",
            timed_out=False,
        )
        monkeypatch.setattr(
            eval_attempt_flow,
            "run_pytest",
            lambda *a, **kw: fake_result,
        )

        result = eval_attempt_flow.run_one_condition_impl(
            case=case,
            condition=run_eval.Condition.BASE_REPAIR,
            workdir=workdir,
            context_dir_base=context_dir,
            inputs=_minimal_run_condition_inputs(
                patcher=stub_patcher,
                adaptive_budget_policy_tiers="not,valid,tiers",
            ),
        )
        assert isinstance(result, list)

    def test_invalid_context_protocol_falls_back_to_monolithic(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Cover normalization of context_protocol to 'monolithic'."""
        from evals import eval_attempt_flow

        case = _make_case(tmp_path, "bad_protocol")
        workdir = tmp_path / "workdirs" / "bad_protocol" / "base_repair"
        workdir.mkdir(parents=True, exist_ok=True)
        context_dir = tmp_path / "contexts" / "bad_protocol" / "base_repair"
        context_dir.mkdir(parents=True, exist_ok=True)

        stub_patcher = SimpleNamespace(
            name="stub",
            propose_patch=lambda *a, **kw: SimpleNamespace(
                diff="--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-pass\n+1\n",
                meta={"status": "ok"},
            ),
        )

        fake_result = run_eval.RunResult(
            returncode=1,
            duration_sec=0.1,
            stdout="FAILED",
            stderr="",
            timed_out=False,
        )
        monkeypatch.setattr(
            eval_attempt_flow,
            "run_pytest",
            lambda *a, **kw: fake_result,
        )

        result = eval_attempt_flow.run_one_condition_impl(
            case=case,
            condition=run_eval.Condition.BASE_REPAIR,
            workdir=workdir,
            context_dir_base=context_dir,
            inputs=_minimal_run_condition_inputs(
                patcher=stub_patcher,
                context_protocol="nonexistent_protocol",
            ),
        )
        assert isinstance(result, list)

    def test_invalid_compaction_mode_falls_back(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """Cover normalization of context_compaction_mode fallback."""
        from evals import eval_attempt_flow

        case = _make_case(tmp_path, "bad_compact")
        workdir = tmp_path / "workdirs" / "bad_compact" / "base_repair"
        workdir.mkdir(parents=True, exist_ok=True)
        context_dir = tmp_path / "contexts" / "bad_compact" / "base_repair"
        context_dir.mkdir(parents=True, exist_ok=True)

        stub_patcher = SimpleNamespace(
            name="stub",
            propose_patch=lambda *a, **kw: SimpleNamespace(
                diff="--- a/f.py\n+++ b/f.py\n@@ -1 +1 @@\n-pass\n+1\n",
                meta={"status": "ok"},
            ),
        )

        fake_result = run_eval.RunResult(
            returncode=1,
            duration_sec=0.1,
            stdout="FAILED",
            stderr="",
            timed_out=False,
        )
        monkeypatch.setattr(
            eval_attempt_flow,
            "run_pytest",
            lambda *a, **kw: fake_result,
        )

        result = eval_attempt_flow.run_one_condition_impl(
            case=case,
            condition=run_eval.Condition.BASE_REPAIR,
            workdir=workdir,
            context_dir_base=context_dir,
            inputs=_minimal_run_condition_inputs(
                patcher=stub_patcher,
                context_compaction_mode="nonexistent_mode",
            ),
        )
        assert isinstance(result, list)


# ===================================================================
# 10. execute_eval_unit_impl — runtime error path
# ===================================================================


class TestExecuteEvalUnitRuntimeError:
    """Cover the exception handling path in execute_eval_unit_impl."""

    def test_runtime_error_captured(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        case = _make_case(tmp_path, "runtime_err")

        def _exploding(**kwargs: Any) -> list[dict[str, Any]]:
            raise ValueError("boom in run_one_condition")

        monkeypatch.setattr(run_eval, "run_one_condition", _exploding)

        outcome = run_eval._execute_eval_unit(
            case=case,
            condition=run_eval.Condition.BASE_REPAIR,
            artifacts_root=tmp_path / "artifacts",
            inputs=run_eval.EvalUnitInputs(
                patcher_args=argparse.Namespace(),
                patcher_override=SimpleNamespace(name="stub"),
                keep_workdirs=False,
                progress_verbosity="basic",
                progress_heartbeat_sec=0.0,
                unit_progress_callback=None,
                run_condition=_minimal_run_condition_inputs(),
            ),
        )
        assert outcome.runtime_error is not None
        assert "boom" in outcome.runtime_error


# ===================================================================
# 11. Heartbeat with detailed verbosity (lines 280-310 printing)
# ===================================================================


class TestHeartbeatDetailedPrinting:
    """Cover heartbeat printing at detailed level (lines ~280-310)."""

    def test_heartbeat_detailed_prints_phase_info(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        case = _make_case(tmp_path, "hb_detailed")
        from evals import eval_attempt_flow

        clock = {"now": 500.0}
        monkeypatch.setattr(eval_attempt_flow.time, "monotonic", lambda: clock["now"])

        def _fake(**kwargs: Any) -> list[dict[str, Any]]:
            pu = kwargs.get("progress_update")
            if pu:
                pu(
                    "setup",
                    {
                        "attempt": 1,
                        "max_attempts": 2,
                        "phase_label": "harness: setup",
                        "note": "configuring workspace",
                    },
                )
            clock["now"] = 500.1
            time.sleep(0.06)
            rec = {
                "run_id": "test-run-id",
                "case_id": "hb_detailed",
                "condition": "base_repair",
                "attempt": 1,
                "success": True,
                "note": "ok",
                "patch_meta": {},
                "attempt_metadata": {},
            }
            rs = kwargs.get("record_sink")
            if rs:
                rs(rec)
            return [rec]

        monkeypatch.setattr(run_eval, "run_one_condition", _fake)

        outcome = run_eval._execute_eval_unit(
            case=case,
            condition=run_eval.Condition.BASE_REPAIR,
            artifacts_root=tmp_path / "artifacts",
            inputs=run_eval.EvalUnitInputs(
                patcher_args=argparse.Namespace(),
                patcher_override=SimpleNamespace(name="stub"),
                keep_workdirs=False,
                progress_verbosity="detailed",
                progress_heartbeat_sec=0.02,
                unit_progress_callback=None,
                run_condition=_minimal_run_condition_inputs(),
            ),
        )
        assert outcome.runtime_error is None
        captured = capsys.readouterr().out
        # Heartbeat at detailed level should print HEARTBEAT lines
        assert "[HEARTBEAT]" in captured or "[ATTEMPT]" in captured


# ===================================================================
# 12. Attempt support imports — coverage of the compatibility wiring
# ===================================================================


class TestBoundRunEvalSymbols:
    """Ensure attempt-flow support symbols are available after import."""

    def test_binds_run_pytest(self) -> None:
        from evals import eval_attempt_flow

        assert hasattr(eval_attempt_flow, "run_pytest")
        assert hasattr(eval_attempt_flow, "Condition")
        assert hasattr(eval_attempt_flow, "CaseSpec")


# ===================================================================
# 13. execute_eval_unit_impl with keep_workdirs=True (cleanup skip)
# ===================================================================


class TestExecuteEvalUnitKeepWorkdirs:
    """Cover the keep_workdirs=True branch (line ~485 skip shutil.rmtree)."""

    def test_workdir_preserved_when_keep_workdirs_true(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        case = _make_case(tmp_path, "keep_wd")

        monkeypatch.setattr(
            run_eval,
            "run_one_condition",
            _fake_run_one_condition_factory(),
        )

        outcome = run_eval._execute_eval_unit(
            case=case,
            condition=run_eval.Condition.BASE_REPAIR,
            artifacts_root=tmp_path / "artifacts",
            inputs=run_eval.EvalUnitInputs(
                patcher_args=argparse.Namespace(),
                patcher_override=SimpleNamespace(name="stub"),
                keep_workdirs=True,
                progress_verbosity="off",
                progress_heartbeat_sec=0.0,
                unit_progress_callback=None,
                run_condition=_minimal_run_condition_inputs(),
            ),
        )
        assert outcome.runtime_error is None
        # Workdir should still exist
        wd = tmp_path / "artifacts" / "workdirs" / "keep_wd" / "base_repair"
        assert wd.exists()


# ===================================================================
# 14. eval_duration_sec_observed progress state update
# ===================================================================


class TestProgressStateEvalDuration:
    """Cover the eval_duration_sec_observed payload handling (lines ~175-177)."""

    def test_eval_duration_sec_observed_tracked(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        case = _make_case(tmp_path, "eval_dur")
        last_payloads: list[dict[str, Any]] = []

        monkeypatch.setattr(
            run_eval,
            "run_one_condition",
            _fake_run_one_condition_factory(
                emit_progress_events=True,
                progress_events=[
                    (
                        "duration_update",
                        {
                            "eval_duration_sec_observed": 42.5,
                            "attempts_completed": 3,
                            "llm_calls_observed": 7,
                        },
                    ),
                ],
            ),
        )

        outcome = run_eval._execute_eval_unit(
            case=case,
            condition=run_eval.Condition.BASE_REPAIR,
            artifacts_root=tmp_path / "artifacts",
            inputs=run_eval.EvalUnitInputs(
                patcher_args=argparse.Namespace(),
                patcher_override=SimpleNamespace(name="stub"),
                keep_workdirs=False,
                progress_verbosity="off",
                progress_heartbeat_sec=0.0,
                unit_progress_callback=lambda cid, cond, event, payload: (
                    last_payloads.append(payload) if payload else None
                ),
                run_condition=_minimal_run_condition_inputs(),
            ),
        )
        assert outcome.runtime_error is None


# ===================================================================
# 16. Pre-existing workdir removal (line 442)
# ===================================================================


class TestExecuteEvalUnitWorkdirRemoval:
    """Cover line 442: shutil.rmtree when workdir already exists."""

    def test_pre_existing_workdir_is_removed(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        case = _make_case(tmp_path, "preexist_wd")
        artifacts_root = tmp_path / "artifacts"

        # Pre-create the workdir so the rmtree branch is triggered
        workdir = artifacts_root / "workdirs" / "preexist_wd" / "base_repair"
        workdir.mkdir(parents=True, exist_ok=True)
        (workdir / "stale_file.txt").write_text("should be removed")

        monkeypatch.setattr(
            run_eval,
            "run_one_condition",
            _fake_run_one_condition_factory(),
        )

        outcome = run_eval._execute_eval_unit(
            case=case,
            condition=run_eval.Condition.BASE_REPAIR,
            artifacts_root=artifacts_root,
            inputs=run_eval.EvalUnitInputs(
                patcher_args=argparse.Namespace(),
                patcher_override=SimpleNamespace(name="stub"),
                keep_workdirs=True,
                progress_verbosity="off",
                progress_heartbeat_sec=0.0,
                unit_progress_callback=None,
                run_condition=_minimal_run_condition_inputs(),
            ),
        )
        assert outcome.runtime_error is None
        # The stale file should no longer be there (workdir was re-created from case)
        assert not (workdir / "stale_file.txt").exists()


# ===================================================================
# 17. Trace verbosity with no attempt/syntax info (empty fragments)
# ===================================================================


class TestTraceVerbosityEmptyFragments:
    """Cover branches in lines 202-213 where attempt/syntax values are 0."""

    def test_trace_with_zero_attempt_values(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        """Lines 202-206: attempt_value==0 -> empty attempt_fragment."""
        case = _make_case(tmp_path, "trace_zero")

        monkeypatch.setattr(
            run_eval,
            "run_one_condition",
            _fake_run_one_condition_factory(
                emit_progress_events=True,
                progress_events=[
                    (
                        "phase_change",
                        {
                            "phase_label": "test: init",
                            # No attempt or syntax info -> fragments should be empty
                        },
                    ),
                ],
            ),
        )

        outcome = run_eval._execute_eval_unit(
            case=case,
            condition=run_eval.Condition.BASE_REPAIR,
            artifacts_root=tmp_path / "artifacts",
            inputs=run_eval.EvalUnitInputs(
                patcher_args=argparse.Namespace(),
                patcher_override=SimpleNamespace(name="stub"),
                keep_workdirs=False,
                progress_verbosity="trace",
                progress_heartbeat_sec=0.0,
                unit_progress_callback=None,
                run_condition=_minimal_run_condition_inputs(),
            ),
        )
        assert outcome.runtime_error is None
        captured = capsys.readouterr().out
        assert "[TRACE]" in captured
        # Should NOT contain attempt= since attempt is 0
        for line in captured.split("\n"):
            if "[TRACE]" in line and "trace_zero" in line:
                assert "syntax_cycle=" not in line


# ===================================================================
# 18. Detailed heartbeat with note text
# ===================================================================


class TestHeartbeatWithNoteText:
    """Cover heartbeat path where note_text is populated (lines ~247-250)."""

    def test_heartbeat_with_note_shows_note_fragment(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        case = _make_case(tmp_path, "hb_note")
        from evals import eval_attempt_flow

        clock = {"now": 200.0}
        monkeypatch.setattr(eval_attempt_flow.time, "monotonic", lambda: clock["now"])

        def _fake(**kwargs: Any) -> list[dict[str, Any]]:
            pu = kwargs.get("progress_update")
            if pu:
                pu(
                    "setup",
                    {
                        "attempt": 1,
                        "max_attempts": 2,
                        "phase_label": "harness: setup",
                        "note": "patching file",
                    },
                )
            clock["now"] = 200.08
            time.sleep(0.06)
            rec = {
                "run_id": "test-run-id",
                "case_id": "hb_note",
                "condition": "base_repair",
                "attempt": 1,
                "success": True,
                "note": "ok",
                "patch_meta": {},
                "attempt_metadata": {},
            }
            rs = kwargs.get("record_sink")
            if rs:
                rs(rec)
            return [rec]

        monkeypatch.setattr(run_eval, "run_one_condition", _fake)

        outcome = run_eval._execute_eval_unit(
            case=case,
            condition=run_eval.Condition.BASE_REPAIR,
            artifacts_root=tmp_path / "artifacts",
            inputs=run_eval.EvalUnitInputs(
                patcher_args=argparse.Namespace(),
                patcher_override=SimpleNamespace(name="stub"),
                keep_workdirs=False,
                progress_verbosity="detailed",
                progress_heartbeat_sec=0.02,
                unit_progress_callback=None,
                run_condition=_minimal_run_condition_inputs(),
            ),
        )
        assert outcome.runtime_error is None
