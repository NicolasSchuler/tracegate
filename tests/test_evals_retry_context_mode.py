"""Tests for retry context-mode behaviour across successive eval attempts."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from evals import run_eval
from evals.patchers.base import PatchResult


def _make_case(tmp_path: Path, *, case_id: str) -> run_eval.CaseSpec:
    case_dir = tmp_path / f"{case_id}_src"
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "buggy.py").write_text(
        "def f() -> int:\n    return 1\n",
        encoding="utf-8",
    )
    (case_dir / "test_buggy.py").write_text(
        "from buggy import f\n\n\ndef test_f() -> None:\n    assert f() == 2\n",
        encoding="utf-8",
    )
    return run_eval.CaseSpec(
        case_id=case_id,
        path=case_dir,
        category="traceback_controls",
        difficulty="easy",
        snapshot_dependency="none",
        expected_exception="AssertionError",
        tags=[],
        python_args=["-m", "pytest", "-q"],
        timeout_sec=30,
    )


@dataclass
class _CapturePatcher:
    name: str = "capture"
    prompts: list[str] = field(default_factory=list)

    def propose_patch(self, *, prompt: str, context_dir, cwd) -> PatchResult:
        del context_dir, cwd
        self.prompts.append(prompt)
        return PatchResult(diff=None, meta={"reason": "no_diff"})


def _failing_pytest(*_args, **_kwargs) -> run_eval.RunResult:
    return run_eval.RunResult(
        returncode=1,
        duration_sec=0.01,
        stdout="E   AssertionError: expected 2 got 1\n",
        stderr="",
    )


def test_build_retry_history_section_prefers_newest_turns() -> None:
    turns = [
        (
            "Attempt 1\nHypothesis tested: first try\nEvidence used: base_repair\n"
            "Outcome: no_patch_proposed\nNew constraint: no diff"
        ),
        (
            "Attempt 2\nHypothesis tested: second try\nEvidence used: base_repair\n"
            "Outcome: patch_apply_failed\nNew constraint: apply failed"
        ),
        (
            "Attempt 3\nHypothesis tested: third try\nEvidence used: stage_a_requests=1\n"
            "Outcome: tests_still_failing\nNew constraint: signature repeated"
        ),
    ]
    budget = len(turns[2]) + 2 + len(turns[1])
    history, chars, turns_included, truncated = run_eval._build_retry_history_section(
        prior_turns=turns,
        max_chars=budget,
    )
    assert history.startswith("Attempt 3")
    assert "Attempt 2" in history
    assert "Attempt 1" not in history
    assert chars == len(history)
    assert turns_included == 2
    assert truncated is True


def test_retry_context_mode_history_includes_previous_attempt_summary(
    tmp_path: Path, monkeypatch
) -> None:
    case = _make_case(tmp_path, case_id="history_case")
    workdir = tmp_path / "workdir_history"
    context_dir = tmp_path / "context_history"
    shutil.copytree(case.path, workdir)
    patcher = _CapturePatcher()

    monkeypatch.setattr(run_eval, "run_pytest", _failing_pytest)

    records = run_eval.run_one_condition(
        case=case,
        condition=run_eval.Condition.BASE_REPAIR,
        workdir=workdir,
        context_dir_base=context_dir,
        patcher=patcher,
        k=2,
        output_format="json_compact",
        run_id="history-run",
        include_files_context=False,
        force_patch_attempt=False,
        retry_context_mode="history",
        retry_history_max_chars=4_000,
    )

    assert len(records) == 2
    assert len(patcher.prompts) == 2
    assert "Retry history from previous attempts:" not in patcher.prompts[0]
    assert "Retry history from previous attempts:" in patcher.prompts[1]
    assert "Attempt 1" in patcher.prompts[1]
    assert "Hypothesis tested:" in patcher.prompts[1]
    assert "Evidence used:" in patcher.prompts[1]
    assert "Outcome: no_patch_proposed" in patcher.prompts[1]

    second_attempt_metadata = records[1]["attempt_metadata"]
    assert second_attempt_metadata["retry_context_mode"] == "history"
    assert int(second_attempt_metadata["retry_history_chars"] or 0) > 0
    assert int(second_attempt_metadata["retry_history_turns_included"] or 0) >= 1


def test_retry_context_mode_fresh_excludes_previous_attempt_summary(
    tmp_path: Path, monkeypatch
) -> None:
    case = _make_case(tmp_path, case_id="fresh_case")
    workdir = tmp_path / "workdir_fresh"
    context_dir = tmp_path / "context_fresh"
    shutil.copytree(case.path, workdir)
    patcher = _CapturePatcher()

    monkeypatch.setattr(run_eval, "run_pytest", _failing_pytest)

    records = run_eval.run_one_condition(
        case=case,
        condition=run_eval.Condition.BASE_REPAIR,
        workdir=workdir,
        context_dir_base=context_dir,
        patcher=patcher,
        k=2,
        output_format="json_compact",
        run_id="fresh-run",
        include_files_context=False,
        force_patch_attempt=False,
        retry_context_mode="fresh",
        retry_history_max_chars=4_000,
    )

    assert len(records) == 2
    assert len(patcher.prompts) == 2
    assert "Retry history from previous attempts:" not in patcher.prompts[0]
    assert "Retry history from previous attempts:" not in patcher.prompts[1]

    second_attempt_metadata = records[1]["attempt_metadata"]
    assert second_attempt_metadata["retry_context_mode"] == "fresh"
    assert int(second_attempt_metadata["retry_history_chars"] or 0) == 0
    assert int(second_attempt_metadata["retry_history_turns_included"] or 0) == 0
