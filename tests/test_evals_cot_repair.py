"""Tests for the COT_REPAIR condition: chain-of-thought enhanced base repair."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path

from evals import eval_attempt_flow, run_eval
from evals.eval_types import Condition
from evals.patchers.base import ContextRequestResult, PatchPlanResult, PatchResult

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_case(tmp_path: Path) -> run_eval.CaseSpec:
    case_dir = tmp_path / "case_src"
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "buggy.py").write_text(
        "import helper\n\n\ndef f() -> int:\n    return helper.value()\n",
        encoding="utf-8",
    )
    (case_dir / "helper.py").write_text(
        "def value() -> int:\n    return 1\n",
        encoding="utf-8",
    )
    (case_dir / "test_buggy.py").write_text(
        "from buggy import f\n\n\ndef test_f() -> None:\n    assert f() == 2\n",
        encoding="utf-8",
    )
    return run_eval.CaseSpec(
        case_id="cot_case",
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
        "exception": {"type": "AssertionError", "message": "expected 2 got 1"},
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
class _CapturePromptPatcher:
    """Patcher that records propose_patch prompts and exposes request_context."""

    name: str = "cot_capture"
    prompts: list[str] = field(default_factory=list)
    plan_prompts: list[str] = field(default_factory=list)
    propose_calls: int = 0
    plan_calls: int = 0
    request_context_calls: int = 0

    def request_context(self, *, prompt: str, context_dir: Path, cwd: Path) -> ContextRequestResult:
        del prompt, context_dir, cwd
        self.request_context_calls += 1
        return ContextRequestResult(
            request_json={
                "need_more_context": True,
                "requests": [{"type": "file", "path": "buggy.py"}],
                "sufficiency_rationale": "need source",
            },
            raw_response='{"need_more_context":true}',
            meta={},
        )

    def propose_patch(self, *, prompt: str, context_dir: Path, cwd: Path) -> PatchResult:
        del context_dir, cwd
        self.propose_calls += 1
        self.prompts.append(prompt)
        return PatchResult(diff=None, meta={"reason": "no_diff"})

    def plan_patch(self, *, prompt: str, context_dir: Path, cwd: Path) -> PatchPlanResult:
        del context_dir, cwd
        self.plan_calls += 1
        self.plan_prompts.append(prompt)
        return PatchPlanResult(
            plan_text=(
                "Diagnosis: helper.value returns the wrong value.\n"
                "Why previous attempt failed: none.\n"
                "New constraint: edit helper.py only.\n"
                "Minimal edit plan: return 2 instead of 1.\n"
                "Verification plan: rerun pytest."
            ),
            raw_response=(
                "Diagnosis: helper.value returns the wrong value.\n"
                "Why previous attempt failed: none.\n"
                "New constraint: edit helper.py only.\n"
                "Minimal edit plan: return 2 instead of 1.\n"
                "Verification plan: rerun pytest."
            ),
            meta={"note": "plan"},
        )


def _stub_pytest(monkeypatch) -> None:  # type: ignore[no-untyped-def]
    monkeypatch.setattr(
        eval_attempt_flow,
        "run_pytest",
        lambda *args, **kwargs: run_eval.RunResult(
            returncode=1,
            duration_sec=0.01,
            stdout=("tests/test_buggy.py::test_f\nE   AssertionError: expected 2 got 1\n"),
            stderr="",
        ),
    )
    monkeypatch.setattr(
        eval_attempt_flow,
        "read_latest_snapshot",
        lambda *_args, **_kwargs: (_snapshot_payload(), None),
    )


def _first_attempt_metadata(
    records: list[dict[str, object]],
) -> tuple[dict[str, object], dict[str, object]]:
    attempt_records = [record for record in records if int(record.get("attempt", 0)) >= 1]
    assert attempt_records
    record = attempt_records[0]
    attempt_metadata = record.get("attempt_metadata") or {}
    patch_meta = record.get("patch_meta") or {}
    assert isinstance(attempt_metadata, dict)
    assert isinstance(patch_meta, dict)
    return attempt_metadata, patch_meta


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_cot_repair_condition_value() -> None:
    """Verify the enum member has the expected string value."""
    assert Condition.COT_REPAIR.value == "cot_repair"


def test_cot_repair_prompt_includes_reasoning_preamble(
    tmp_path: Path,
    monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    """Run with COT_REPAIR and verify the CoT reasoning preamble appears."""
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir_cot_prompt"
    context_dir = tmp_path / "context_cot_prompt"
    shutil.copytree(case.path, workdir)
    _stub_pytest(monkeypatch)

    patcher = _CapturePromptPatcher()

    run_eval.run_one_condition(
        case=case,
        condition=Condition.COT_REPAIR,
        workdir=workdir,
        context_dir_base=context_dir,
        patcher=patcher,
        k=1,
        output_format="json_compact",
        run_id="run-cot-prompt",
        include_files_context=True,
        force_patch_attempt=False,
    )

    assert patcher.propose_calls == 1
    prompt = patcher.prompts[0]
    # The CoT preamble must include the step-by-step reasoning instructions
    assert "step by step" in prompt
    assert "root cause" in prompt.lower()
    assert "Write out your reasoning" in prompt


def test_cot_repair_invokes_plan_then_patch_on_attempt_one(
    tmp_path: Path,
    monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir_cot_plan"
    context_dir = tmp_path / "context_cot_plan"
    shutil.copytree(case.path, workdir)
    _stub_pytest(monkeypatch)

    patcher = _CapturePromptPatcher()

    records = run_eval.run_one_condition(
        case=case,
        condition=Condition.COT_REPAIR,
        workdir=workdir,
        context_dir_base=context_dir,
        patcher=patcher,
        k=1,
        output_format="json_compact",
        run_id="run-cot-plan",
        include_files_context=True,
        force_patch_attempt=False,
    )

    assert patcher.plan_calls == 1
    assert patcher.propose_calls == 1
    assert "Structured repair plan from planning step:" in patcher.prompts[0]

    meta, patch_meta = _first_attempt_metadata(records)
    assert meta.get("plan_patch_invoked") is True
    assert meta.get("plan_patch_succeeded") is True
    assert meta.get("plan_patch_mode") == "cot_condition"
    assert meta.get("prompt_adaptive_stage_b_cot") is False
    assert meta.get("prompt_adaptive_stage_b_cot_effective") is False
    assert meta.get("prompt_adaptive_stage_b_cot_source") == "off"
    assert meta.get("native_baseline_mode") == ""
    assert patch_meta.get("plan_patch_invoked") is True
    assert patch_meta.get("plan_patch_succeeded") is True
    assert patch_meta.get("plan_patch_mode") == "cot_condition"
    assert patch_meta.get("native_baseline_mode") == ""


def test_cot_repair_skips_stage_a(
    tmp_path: Path,
    monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    """COT_REPAIR must NOT invoke Stage-A (request_context)."""
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir_cot_skip_a"
    context_dir = tmp_path / "context_cot_skip_a"
    shutil.copytree(case.path, workdir)
    _stub_pytest(monkeypatch)

    patcher = _CapturePromptPatcher()

    run_eval.run_one_condition(
        case=case,
        condition=Condition.COT_REPAIR,
        workdir=workdir,
        context_dir_base=context_dir,
        patcher=patcher,
        k=1,
        output_format="json_compact",
        run_id="run-cot-skip-a",
        include_files_context=True,
        force_patch_attempt=False,
    )

    # Stage-A (request_context) should never be called for cot_repair
    assert patcher.request_context_calls == 0
    # But propose_patch must still run
    assert patcher.propose_calls == 1


def test_cot_repair_is_not_snapshot_condition() -> None:
    """cot_repair should not be classified as a snapshot condition."""
    assert run_eval._is_snapshot_condition(Condition.COT_REPAIR) is False


def test_cot_repair_no_structured_seed(
    tmp_path: Path,
    monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    """COT_REPAIR records must not contain structured_seed/adaptive indicators."""
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir_cot_no_seed"
    context_dir = tmp_path / "context_cot_no_seed"
    shutil.copytree(case.path, workdir)
    _stub_pytest(monkeypatch)

    patcher = _CapturePromptPatcher()

    records = run_eval.run_one_condition(
        case=case,
        condition=Condition.COT_REPAIR,
        workdir=workdir,
        context_dir_base=context_dir,
        patcher=patcher,
        k=1,
        output_format="json_compact",
        run_id="run-cot-no-seed",
        include_files_context=True,
        force_patch_attempt=False,
    )

    assert len(records) >= 1
    # Check the first real attempt record (attempt >= 1)
    attempt_records = [r for r in records if int(r.get("attempt", 0)) >= 1]
    assert len(attempt_records) >= 1
    record = attempt_records[0]
    meta = record.get("attempt_metadata") or {}
    # structured_seed is only active for adaptive conditions — not cot_repair
    prompt_text = patcher.prompts[0] if patcher.prompts else ""
    assert "Retry packet (delta-first, structured):" not in prompt_text
    # Adaptive mode should be off for non-adaptive conditions
    assert meta.get("adaptive_mode") == "none"
    assert meta.get("adaptive_policy") == "disabled"


def test_adaptive_cot_baseline_invokes_planner_and_injects_plan_on_attempt_one(
    tmp_path: Path,
    monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir_adaptive_cot_baseline"
    context_dir = tmp_path / "context_adaptive_cot_baseline"
    shutil.copytree(case.path, workdir)
    _stub_pytest(monkeypatch)

    patcher = _CapturePromptPatcher()

    records = run_eval.run_one_condition(
        case=case,
        condition=Condition.LAYERED_ADAPTIVE_LLM,
        workdir=workdir,
        context_dir_base=context_dir,
        patcher=patcher,
        k=1,
        output_format="json_compact",
        run_id="run-adaptive-cot-baseline",
        include_files_context=True,
        force_patch_attempt=False,
        adaptive_max_helper_calls=1,
        native_baseline_mode="cot_repair",
    )

    assert patcher.plan_calls == 1
    assert patcher.propose_calls == 1
    assert "Structured repair plan from planning step:" in patcher.prompts[0]

    meta, patch_meta = _first_attempt_metadata(records)
    assert meta.get("plan_patch_invoked") is True
    assert meta.get("plan_patch_succeeded") is True
    assert meta.get("plan_patch_first_attempt_enabled") is True
    assert meta.get("plan_patch_mode") == "cot_baseline"
    assert meta.get("prompt_adaptive_stage_b_cot") is False
    assert meta.get("prompt_adaptive_stage_b_cot_effective") is True
    assert meta.get("prompt_adaptive_stage_b_cot_source") == "native_baseline"
    assert meta.get("native_baseline_mode") == "cot_repair"
    assert patch_meta.get("plan_patch_mode") == "cot_baseline"
    assert patch_meta.get("prompt_adaptive_stage_b_cot") is False
    assert patch_meta.get("prompt_adaptive_stage_b_cot_effective") is True
    assert patch_meta.get("prompt_adaptive_stage_b_cot_source") == "native_baseline"
    assert patch_meta.get("native_baseline_mode") == "cot_repair"


def test_plain_layered_adaptive_llm_does_not_invoke_attempt_one_planner(
    tmp_path: Path,
    monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir_plain_adaptive"
    context_dir = tmp_path / "context_plain_adaptive"
    shutil.copytree(case.path, workdir)
    _stub_pytest(monkeypatch)

    patcher = _CapturePromptPatcher()

    records = run_eval.run_one_condition(
        case=case,
        condition=Condition.LAYERED_ADAPTIVE_LLM,
        workdir=workdir,
        context_dir_base=context_dir,
        patcher=patcher,
        k=1,
        output_format="json_compact",
        run_id="run-plain-adaptive",
        include_files_context=True,
        force_patch_attempt=False,
        adaptive_max_helper_calls=1,
    )

    assert patcher.plan_calls == 0
    assert patcher.propose_calls == 1

    meta, patch_meta = _first_attempt_metadata(records)
    assert meta.get("plan_patch_invoked") is False
    assert meta.get("plan_patch_first_attempt_enabled") is False
    assert meta.get("plan_patch_mode") == "off"
    assert meta.get("prompt_adaptive_stage_b_cot") is False
    assert meta.get("prompt_adaptive_stage_b_cot_effective") is False
    assert meta.get("prompt_adaptive_stage_b_cot_source") == "off"
    assert meta.get("native_baseline_mode") == ""
    assert patch_meta.get("plan_patch_mode") == "off"
    assert patch_meta.get("prompt_adaptive_stage_b_cot_effective") is False


def test_prompt_only_adaptive_stage_b_cot_does_not_enable_cot_baseline_semantics(
    tmp_path: Path,
    monkeypatch,  # type: ignore[no-untyped-def]
) -> None:
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir_prompt_only_adaptive_cot"
    context_dir = tmp_path / "context_prompt_only_adaptive_cot"
    shutil.copytree(case.path, workdir)
    _stub_pytest(monkeypatch)

    patcher = _CapturePromptPatcher()

    records = run_eval.run_one_condition(
        case=case,
        condition=Condition.LAYERED_ADAPTIVE_LLM,
        workdir=workdir,
        context_dir_base=context_dir,
        patcher=patcher,
        k=1,
        output_format="json_compact",
        run_id="run-prompt-only-adaptive-cot",
        include_files_context=True,
        force_patch_attempt=False,
        adaptive_max_helper_calls=1,
        prompt_adaptive_stage_b_cot=True,
    )

    assert patcher.plan_calls == 0
    assert patcher.propose_calls == 1
    assert "Before proposing a fix, think through the problem step by step:" in patcher.prompts[0]

    meta, patch_meta = _first_attempt_metadata(records)
    assert meta.get("plan_patch_invoked") is False
    assert meta.get("plan_patch_mode") == "off"
    assert meta.get("prompt_adaptive_stage_b_cot") is True
    assert meta.get("prompt_adaptive_stage_b_cot_effective") is True
    assert meta.get("prompt_adaptive_stage_b_cot_source") == "flag"
    assert meta.get("native_baseline_mode") == ""
    assert patch_meta.get("plan_patch_mode") == "off"
    assert patch_meta.get("prompt_adaptive_stage_b_cot") is True
    assert patch_meta.get("prompt_adaptive_stage_b_cot_effective") is True
    assert patch_meta.get("prompt_adaptive_stage_b_cot_source") == "flag"
