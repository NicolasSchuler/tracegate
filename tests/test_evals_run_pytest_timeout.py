"""Tests for pytest timeout handling and structured result reporting in eval runs."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from evals import run_eval
from evals.patchers.base import PatchResult
from evals.subprocess_monitor import MonitoredSubprocessResult


def test_run_pytest_timeout_returns_structured_result(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(
        run_eval,
        "run_monitored_subprocess",
        lambda **_kwargs: MonitoredSubprocessResult(
            returncode=-9,
            duration_sec=1.0,
            stdout="partial stdout",
            stderr="partial stderr",
            timed_out=True,
            timeout_sec=1,
        ),
    )

    result = run_eval.run_pytest(
        cwd=tmp_path,
        python_args=["-m", "pytest", "-q"],
        condition=run_eval.Condition.BASE_REPAIR,
        timeout_sec=1,
        output_format="json_compact",
        execution=run_eval.ExecutionConfig(executor="local"),
    )

    assert result.returncode == run_eval.PYTEST_TIMEOUT_RETURNCODE
    assert result.timed_out is True
    assert result.timeout_sec == 1
    assert "partial stdout" in result.stdout
    assert "pytest timed out after 1s" in result.stderr


def test_run_pytest_monitored_local_preserves_output_and_emits_progress(
    tmp_path, monkeypatch
) -> None:
    emitted: list[tuple[str, dict[str, object] | None]] = []

    def _fake_monitored_run(**kwargs) -> MonitoredSubprocessResult:
        progress_callback = kwargs["progress_callback"]
        progress_base_payload = dict(kwargs["progress_base_payload"])
        progress_callback(
            "pytest_activity",
            {
                **progress_base_payload,
                "source": "stdout",
                "activity_excerpt": "test_buggy.py::test_f",
            },
        )
        progress_callback(
            "pytest_activity",
            {
                **progress_base_payload,
                "source": "stderr",
                "activity_excerpt": "AssertionError: expected 2 got 1",
            },
        )
        return MonitoredSubprocessResult(
            returncode=1,
            duration_sec=0.25,
            stdout="test_buggy.py::test_f\n",
            stderr="AssertionError: expected 2 got 1\n",
        )

    monkeypatch.setattr(run_eval, "run_monitored_subprocess", _fake_monitored_run)

    result = run_eval.run_pytest(
        cwd=tmp_path,
        python_args=["-m", "pytest", "-q"],
        condition=run_eval.Condition.BASE_REPAIR,
        timeout_sec=2,
        output_format="json_compact",
        execution=run_eval.ExecutionConfig(executor="local"),
        progress_update=lambda event, payload: emitted.append((event, payload)),
        progress_phase_label="pytest: after patch",
    )

    assert result.returncode == 1
    assert result.timed_out is False
    assert result.timeout_sec is None
    assert result.duration_sec == pytest.approx(0.25)
    assert result.stdout == "test_buggy.py::test_f\n"
    assert result.stderr == "AssertionError: expected 2 got 1\n"
    assert emitted == [
        (
            "pytest_activity",
            {
                "owner": "pytest",
                "phase_label": "pytest: after patch",
                "source": "stdout",
                "activity_excerpt": "test_buggy.py::test_f",
            },
        ),
        (
            "pytest_activity",
            {
                "owner": "pytest",
                "phase_label": "pytest: after patch",
                "source": "stderr",
                "activity_excerpt": "AssertionError: expected 2 got 1",
            },
        ),
    ]


@dataclass(frozen=True)
class _NoDiffPatcher:
    name: str = "no_diff"

    def propose_patch(self, *, prompt: str, context_dir, cwd) -> PatchResult:
        return PatchResult(diff=None, meta={"reason": "no_diff"})


def test_run_one_condition_records_timeout_metadata(tmp_path, monkeypatch) -> None:
    case_dir = tmp_path / "case_src"
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "buggy.py").write_text("def f() -> int:\n    return 1\n", encoding="utf-8")
    (case_dir / "test_buggy.py").write_text(
        "from buggy import f\n\n\ndef test_f() -> None:\n    assert f() == 2\n",
        encoding="utf-8",
    )
    case = run_eval.CaseSpec(
        case_id="timeout_case",
        path=case_dir,
        category="traceback_controls",
        difficulty="easy",
        snapshot_dependency="none",
        expected_exception="AssertionError",
        tags=[],
        python_args=["-m", "pytest", "-q"],
        timeout_sec=1,
        description=None,
    )

    def _fake_run_pytest(
        cwd,
        python_args,
        condition,
        timeout_sec,
        output_format,
        **_kwargs,
    ):
        return run_eval.RunResult(
            returncode=run_eval.PYTEST_TIMEOUT_RETURNCODE,
            duration_sec=0.01,
            stdout="",
            stderr="pytest timed out after 1s",
            timed_out=True,
            timeout_sec=1,
        )

    monkeypatch.setattr(run_eval, "run_pytest", _fake_run_pytest)

    records = run_eval.run_one_condition(
        case=case,
        condition=run_eval.Condition.BASE_REPAIR,
        workdir=tmp_path / "workdir",
        context_dir_base=tmp_path / "context",
        patcher=_NoDiffPatcher(),
        k=1,
        output_format="json_compact",
        run_id="run-timeout",
        include_files_context=False,
        force_patch_attempt=False,
        context_protocol="staged",
    )

    assert len(records) == 1
    record = records[0]
    assert record["note"] == "no_patch_proposed"
    assert record["timed_out"] is True
    assert record["timeout_sec"] == 1
    assert record["context_protocol"] == "monolithic"
    assert isinstance(record.get("attempt_wall_clock_sec"), (int, float))
    assert float(record["attempt_wall_clock_sec"]) >= 0.0


def test_run_one_condition_initial_success_records_attempt_wall_clock(
    tmp_path, monkeypatch
) -> None:
    case_dir = tmp_path / "case_src_success"
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "buggy.py").write_text("def f() -> int:\n    return 2\n", encoding="utf-8")
    (case_dir / "test_buggy.py").write_text(
        "from buggy import f\n\n\ndef test_f() -> None:\n    assert f() == 2\n",
        encoding="utf-8",
    )
    case = run_eval.CaseSpec(
        case_id="success_case",
        path=case_dir,
        category="traceback_controls",
        difficulty="easy",
        snapshot_dependency="none",
        expected_exception="AssertionError",
        tags=[],
        python_args=["-m", "pytest", "-q"],
        timeout_sec=1,
        description=None,
    )

    def _fake_run_pytest(
        cwd,
        python_args,
        condition,
        timeout_sec,
        output_format,
        **_kwargs,
    ):
        return run_eval.RunResult(
            returncode=0,
            duration_sec=0.25,
            stdout="",
            stderr="",
            timed_out=False,
            timeout_sec=None,
        )

    monkeypatch.setattr(run_eval, "run_pytest", _fake_run_pytest)

    records = run_eval.run_one_condition(
        case=case,
        condition=run_eval.Condition.BASE_REPAIR,
        workdir=tmp_path / "workdir_success",
        context_dir_base=tmp_path / "context_success",
        patcher=_NoDiffPatcher(),
        k=1,
        output_format="json_compact",
        run_id="run-success",
        include_files_context=False,
        force_patch_attempt=False,
        context_protocol="staged",
    )

    assert len(records) == 1
    record = records[0]
    assert record["note"] == "no_patch_needed"
    assert record["attempt"] == 0
    assert record["attempt_wall_clock_sec"] == pytest.approx(0.25)
