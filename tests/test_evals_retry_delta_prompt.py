"""Tests for delta-prompt generation on eval retry attempts."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from evals import eval_attempt_flow, run_eval
from evals.native_stage_b_flow import OpenAIPromptBudgetError
from evals.patchers.base import ContextRequestResult, PatchResult


def _make_case(tmp_path: Path) -> run_eval.CaseSpec:
    case_dir = tmp_path / "case_src"
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "buggy.py").write_text(
        "import helper\n\n\ndef f() -> int:\n    return helper.value()\n",
        encoding="utf-8",
    )
    (case_dir / "helper.py").write_text("def value() -> int:\n    return 1\n", encoding="utf-8")
    (case_dir / "test_buggy.py").write_text(
        "from buggy import f\n\n\ndef test_f() -> None:\n    assert f() == 2\n",
        encoding="utf-8",
    )
    (case_dir / "notes.md").write_text("x" * 6000 + "\n", encoding="utf-8")
    return run_eval.CaseSpec(
        case_id="retry_delta_case",
        path=case_dir,
        category="traceback_controls",
        difficulty="easy",
        snapshot_dependency="helpful",
        expected_exception="AssertionError",
        tags=[],
        python_args=["-m", "pytest", "-q"],
        timeout_sec=30,
    )


def _snapshot_payload() -> dict:
    return {
        "exception": {
            "type": "AssertionError",
            "message": "expected 2 got 1",
        },
        "frames": [
            {
                "file_rel": "buggy.py",
                "line": 4,
                "function": "f",
                "code": "return helper.value()",
                "locals": {"x": 1},
            }
        ],
        "pytest": {
            "nodeid": "test_buggy.py::test_f",
            "repro": ["python", "-m", "pytest", "-q"],
        },
    }


@dataclass
class _DeltaCapturePatcher:
    name: str = "delta_capture"
    request_calls: int = 0
    prompts: list[str] = field(default_factory=list)

    def request_context(self, *, prompt: str, context_dir, cwd) -> ContextRequestResult:
        del prompt, context_dir, cwd
        self.request_calls += 1
        if self.request_calls == 1:
            requests = [{"type": "file", "path": "buggy.py"}]
        else:
            requests = [{"type": "search", "path": "buggy.py", "pattern": "helper.value"}]
        return ContextRequestResult(
            request_json={
                "need_more_context": True,
                "requests": requests,
                "sufficiency_rationale": "need source",
            },
            raw_response='{"need_more_context":true}',
            meta={},
        )

    def propose_patch(self, *, prompt: str, context_dir, cwd) -> PatchResult:
        del context_dir, cwd
        self.prompts.append(prompt)
        return PatchResult(diff=None, meta={"reason": "no_diff"})


@dataclass
class _BudgetPatcher:
    name: str = "budget_capture"
    prompts: list[str] = field(default_factory=list)

    def propose_patch(self, *, prompt: str, context_dir, cwd) -> PatchResult:
        del context_dir, cwd
        self.prompts.append(prompt)
        return PatchResult(diff=None, meta={"reason": "no_diff"})


@dataclass
class _OpenAIPreflightCapturePatcher:
    name: str = "openai"
    max_tokens: int = 2048
    prompts: list[str] = field(default_factory=list)

    def propose_patch(self, *, prompt: str, context_dir, cwd) -> PatchResult:
        del context_dir, cwd
        self.prompts.append(prompt)
        return PatchResult(diff=None, meta={"reason": "no_diff", "max_tokens": self.max_tokens})


def test_structured_retry_uses_delta_prompt_and_evidence_references(tmp_path, monkeypatch) -> None:
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir"
    context_dir = tmp_path / "context"
    shutil.copytree(case.path, workdir)

    patcher = _DeltaCapturePatcher()

    monkeypatch.setattr(
        eval_attempt_flow,
        "run_pytest",
        lambda *args, **kwargs: run_eval.RunResult(
            returncode=1,
            duration_sec=0.01,
            stdout="tests/test_buggy.py::test_f\nE   AssertionError: expected 2 got 1\n",
            stderr="",
        ),
    )
    monkeypatch.setattr(
        eval_attempt_flow,
        "read_latest_snapshot",
        lambda *_args, **_kwargs: (_snapshot_payload(), None),
    )

    records = run_eval.run_one_condition(
        case=case,
        condition=run_eval.Condition.LAYERED_ADAPTIVE_GATED,
        workdir=workdir,
        context_dir_base=context_dir,
        patcher=patcher,
        k=2,
        output_format="json_compact",
        run_id="run-retry-delta",
        include_files_context=False,
        force_patch_attempt=False,
        context_protocol="staged",
        adaptive_gate_policy="aggressive",
        adaptive_max_helper_calls=2,
        retry_prompt_mode="delta",
    )

    assert len(records) == 2
    assert len(patcher.prompts) == 2

    first_prompt, second_prompt = patcher.prompts
    assert "Pytest output (stdout):" in first_prompt
    assert "Pytest output (stdout):" not in second_prompt
    assert "Known evidence references:" in second_prompt
    assert "New evidence IDs in this retry:" in second_prompt
    assert "Stage-A requested context:" in second_prompt
    assert "Stage-A evidence index:" in second_prompt

    second_meta = records[1]["attempt_metadata"]
    assert second_meta["retry_delta_prompt"] is True
    assert second_meta["evidence_ids_new_count"] >= 1
    assert second_meta["evidence_ids_reused_count"] >= 1
    first_new_ids = list(records[0]["patch_meta"].get("evidence_ids_new") or [])
    second_new_ids = list(records[1]["patch_meta"].get("evidence_ids_new") or [])
    assert first_new_ids
    assert second_new_ids
    stage_a_index_section = second_prompt.split("Stage-A evidence index:\n", 1)[1].split("\n\n", 1)[
        0
    ]
    assert second_new_ids[0] in stage_a_index_section
    assert first_new_ids[0] not in stage_a_index_section


def test_prompt_budget_caps_prompt_size(tmp_path, monkeypatch) -> None:
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir_budget"
    context_dir = tmp_path / "context_budget"
    shutil.copytree(case.path, workdir)

    patcher = _BudgetPatcher()

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

    records = run_eval.run_one_condition(
        case=case,
        condition=run_eval.Condition.BASE_REPAIR,
        workdir=workdir,
        context_dir_base=context_dir,
        patcher=patcher,
        k=1,
        output_format="json_compact",
        run_id="run-budget",
        include_files_context=True,
        max_context_chars=200_000,
        force_patch_attempt=False,
        context_protocol="monolithic",
        prompt_budget_total_chars=800,
        prompt_budget_stdout_chars=120,
        prompt_budget_snapshot_chars=120,
        prompt_budget_evidence_chars=180,
    )

    assert len(records) == 1
    assert len(patcher.prompts) == 1
    assert len(patcher.prompts[0]) <= 800

    attempt_meta = records[0]["attempt_metadata"]
    assert int(attempt_meta["prompt_budget_used_chars"] or 0) <= 800
    assert int(attempt_meta["prompt_budget_overflow_chars"] or 0) == 0
    assert int(attempt_meta["stage_b_prompt_chars"] or 0) <= 800


def test_prompt_budget_disabled_reports_zero_overflow() -> None:
    result = run_eval._apply_prompt_budget(
        sections=[{"key": "instructions", "text": "abc"}],
        budget={"total_chars": 0},
        is_delta_retry=False,
    )
    assert result.prompt == "abc\n"
    assert result.used_chars == len(result.prompt)
    assert result.overflow_chars == 0


def test_openai_preflight_context_window_caps_prompt(tmp_path, monkeypatch) -> None:
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir_openai_preflight"
    context_dir = tmp_path / "context_openai_preflight"
    shutil.copytree(case.path, workdir)
    patcher = _OpenAIPreflightCapturePatcher()

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

    context_window_tokens = 4500  # 1,428 usable prompt tokens after max_tokens and reserve
    expected_cap = run_eval._openai_preflight_total_char_cap(
        context_window_tokens=context_window_tokens,
        max_tokens=patcher.max_tokens,
    )

    records = run_eval.run_one_condition(
        case=case,
        condition=run_eval.Condition.BASE_REPAIR,
        workdir=workdir,
        context_dir_base=context_dir,
        patcher=patcher,
        k=1,
        output_format="json_compact",
        run_id="run-openai-preflight",
        include_files_context=True,
        max_context_chars=200_000,
        force_patch_attempt=False,
        context_protocol="monolithic",
        openai_context_window_tokens=context_window_tokens,
        prompt_budget_total_chars=180_000,
    )

    assert len(records) == 1
    assert len(patcher.prompts) == 1
    assert expected_cap > 0
    assert len(patcher.prompts[0]) <= expected_cap

    attempt_meta = records[0]["attempt_metadata"]
    assert int(attempt_meta["openai_preflight_prompt_cap_chars"] or 0) == expected_cap
    assert attempt_meta["openai_preflight_trimmed"] is True
    assert int(attempt_meta["stage_b_prompt_chars"] or 0) <= expected_cap

    patch_meta = records[0]["patch_meta"]
    assert int(patch_meta["openai_preflight_prompt_cap_chars"] or 0) == expected_cap
    assert patch_meta["openai_preflight_trimmed"] is True


def test_openai_preflight_degenerate_context_window_raises_instead_of_one_char_prompt(
    tmp_path, monkeypatch
) -> None:
    """LLM-06: a window smaller than max_tokens + reserve used to send a 1-character prompt."""
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir_openai_preflight_min_cap"
    context_dir = tmp_path / "context_openai_preflight_min_cap"
    shutil.copytree(case.path, workdir)
    patcher = _OpenAIPreflightCapturePatcher()

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

    with pytest.raises(OpenAIPromptBudgetError):
        run_eval.run_one_condition(
            case=case,
            condition=run_eval.Condition.BASE_REPAIR,
            workdir=workdir,
            context_dir_base=context_dir,
            patcher=patcher,
            k=1,
            output_format="json_compact",
            run_id="run-openai-preflight-min-cap",
            include_files_context=True,
            max_context_chars=200_000,
            force_patch_attempt=False,
            context_protocol="monolithic",
            openai_context_window_tokens=2000,
            prompt_budget_total_chars=180_000,
        )

    assert patcher.prompts == []
