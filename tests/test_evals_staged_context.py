"""Tests for staged context delivery (two-stage prompt flow) in eval runs."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from typing import Any, cast

import pytest

from evals import run_eval
from evals.patchers.base import ContextRequestResult, PatchPlanResult, PatchResult


def _make_case(tmp_path) -> run_eval.CaseSpec:
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
    return run_eval.CaseSpec(
        case_id="staged_case",
        path=case_dir,
        category="traceback_controls",
        difficulty="easy",
        snapshot_dependency="none",
        expected_exception="AssertionError",
        tags=[],
        python_args=["-m", "pytest", "-q"],
        timeout_sec=30,
    )


def test_execute_context_requests_valid_payload(tmp_path) -> None:
    case = _make_case(tmp_path)
    snapshot = {
        "frames": [
            {
                "file_rel": "buggy.py",
                "line": 4,
                "function": "f",
                "code": "return helper.value()",
                "locals": {"x": 1},
            }
        ]
    }
    payload = {
        "need_more_context": True,
        "requests": [
            {"type": "file", "path": "buggy.py"},
            {"type": "search", "path": "buggy.py", "pattern": "helper.value"},
            {"type": "frame", "index": 0},
        ],
        "sufficiency_rationale": "Need file and frame details.",
    }
    result = run_eval.execute_context_requests(
        request_payload=payload,
        root=case.path,
        snapshot=snapshot,
        file_index="buggy.py\nhelper.py\ntest_buggy.py",
        stdout="",
        stderr="",
        max_requests=6,
        max_total_chars=50_000,
    )

    assert result.used_fallback is False
    assert result.request_success is True
    assert result.requests_count == 3
    assert "buggy.py" in result.text
    assert "frame[0]" in result.text


def test_execute_context_requests_local_var_request(tmp_path) -> None:
    case = _make_case(tmp_path)
    snapshot = {
        "frames": [
            {
                "file_rel": "buggy.py",
                "line": 4,
                "function": "f",
                "code": "return helper.value()",
                "locals": {"user_payload": {"timeout": None, "retries": 2}},
            }
        ]
    }
    payload = {
        "need_more_context": True,
        "requests": [{"type": "local_var", "frame": 0, "name": "user_payload"}],
        "sufficiency_rationale": "Need the suspicious local only.",
    }
    result = run_eval.execute_context_requests(
        request_payload=payload,
        root=case.path,
        snapshot=snapshot,
        file_index="buggy.py\nhelper.py\ntest_buggy.py",
        stdout="",
        stderr="",
        max_requests=6,
        max_total_chars=50_000,
    )

    assert result.used_fallback is False
    assert result.request_success is True
    assert result.requests_count == 1
    assert "local_var" in result.text
    assert "user_payload" in result.text


def test_coerce_context_request_payload_preserves_frame_index_zero() -> None:
    payload = run_eval._coerce_context_request_payload(
        {
            "need_more_context": True,
            "requests": [{"type": "frame", "index": 0}],
            "sufficiency_rationale": "Need frame details.",
        }
    )
    assert payload is not None
    requests = payload.get("requests")
    assert isinstance(requests, list)
    assert requests == [{"type": "frame", "index": 0}]


def test_coerce_context_request_payload_preserves_local_var_request() -> None:
    payload = run_eval._coerce_context_request_payload(
        {
            "need_more_context": True,
            "requests": [{"type": "local_var", "frame": 0, "name": "user_payload"}],
            "sufficiency_rationale": "Need exact local.",
        }
    )
    assert payload is not None
    requests = payload.get("requests")
    assert isinstance(requests, list)
    assert requests == [{"type": "local_var", "frame": 0, "name": "user_payload"}]


def test_execute_context_requests_string_false_need_more_context_is_false(tmp_path) -> None:
    case = _make_case(tmp_path)
    payload = cast(
        Any,
        {
            "need_more_context": "false",
            "requests": [{"type": "file", "path": "buggy.py"}],
        },
    )
    result = run_eval.execute_context_requests(
        request_payload=payload,
        root=case.path,
        snapshot=None,
        file_index="buggy.py\nhelper.py\ntest_buggy.py",
        stdout="",
        stderr="",
        max_requests=6,
        max_total_chars=50_000,
    )
    assert result.used_fallback is False
    assert result.request_success is True
    assert result.reason == "sufficient_context"
    assert result.requests_count == 0
    assert result.text == ""


def test_execute_context_requests_path_whitelist_enforced(tmp_path) -> None:
    case = _make_case(tmp_path)
    payload = {
        "need_more_context": True,
        "requests": [{"type": "file", "path": "outside.py"}],
    }
    result = run_eval.execute_context_requests(
        request_payload=payload,
        root=case.path,
        snapshot=None,
        file_index="buggy.py\nhelper.py\ntest_buggy.py",
        stdout="",
        stderr="",
        max_requests=6,
        max_total_chars=50_000,
    )

    assert result.used_fallback is True
    assert result.request_success is False
    assert result.reason == "no_valid_stage_a_requests"


def test_execute_context_requests_accepts_dot_prefixed_path_variant(tmp_path) -> None:
    case = _make_case(tmp_path)
    payload = {
        "need_more_context": True,
        "requests": [{"type": "file", "path": "./buggy.py"}],
    }
    result = run_eval.execute_context_requests(
        request_payload=payload,
        root=case.path,
        snapshot=None,
        file_index="buggy.py\nhelper.py\ntest_buggy.py",
        stdout="",
        stderr="",
        max_requests=6,
        max_total_chars=50_000,
    )

    assert result.used_fallback is False
    assert result.request_success is True
    assert result.requests_count == 1
    assert "--- buggy.py ---" in result.text


def test_execute_context_requests_normalizes_separator_mismatch(tmp_path) -> None:
    case = _make_case(tmp_path)
    nested = case.path / "pkg"
    nested.mkdir(parents=True, exist_ok=True)
    (nested / "service.py").write_text("def value() -> int:\n    return 3\n", encoding="utf-8")
    payload = {
        "need_more_context": True,
        "requests": [{"type": "file", "path": "pkg/service.py"}],
    }
    result = run_eval.execute_context_requests(
        request_payload=payload,
        root=case.path,
        snapshot=None,
        file_index="pkg\\service.py",
        stdout="",
        stderr="",
        max_requests=6,
        max_total_chars=50_000,
    )

    assert result.used_fallback is False
    assert result.request_success is True
    assert result.requests_count == 1
    assert "--- pkg/service.py ---" in result.text


def test_execute_context_requests_caps_request_count(tmp_path) -> None:
    case = _make_case(tmp_path)
    payload = {
        "need_more_context": True,
        "requests": [
            {"type": "file", "path": "buggy.py"},
            {"type": "file", "path": "helper.py"},
            {"type": "file", "path": "test_buggy.py"},
        ],
    }
    result = run_eval.execute_context_requests(
        request_payload=payload,
        root=case.path,
        snapshot=None,
        file_index="buggy.py\nhelper.py\ntest_buggy.py",
        stdout="",
        stderr="",
        max_requests=2,
        max_total_chars=50_000,
    )

    assert result.used_fallback is False
    assert result.requests_count == 2


def test_execute_context_requests_search_is_bounded(tmp_path) -> None:
    case = _make_case(tmp_path)
    repeated = "\n".join(f"value = helper.value()  # {i}" for i in range(40))
    (case.path / "buggy.py").write_text(repeated + "\n", encoding="utf-8")

    payload = {
        "need_more_context": True,
        "requests": [{"type": "search", "path": "buggy.py", "pattern": "helper.value"}],
    }
    result = run_eval.execute_context_requests(
        request_payload=payload,
        root=case.path,
        snapshot=None,
        file_index="buggy.py",
        stdout="",
        stderr="",
        max_requests=6,
        max_total_chars=50_000,
    )

    match_lines = [line for line in result.text.splitlines() if line[:1].isdigit()]
    assert len(match_lines) <= 20


def test_execute_context_requests_file_read_is_bounded(tmp_path) -> None:
    case = _make_case(tmp_path)
    very_large = "X" * 200_000
    (case.path / "large.txt").write_text(very_large, encoding="utf-8")

    payload = {
        "need_more_context": True,
        "requests": [{"type": "file", "path": "large.txt"}],
    }
    result = run_eval.execute_context_requests(
        request_payload=payload,
        root=case.path,
        snapshot=None,
        file_index=("buggy.py", "helper.py", "test_buggy.py", "large.txt"),
        stdout="",
        stderr="",
        max_requests=6,
        max_total_chars=80_000,
    )

    assert result.used_fallback is False
    assert result.request_success is True
    assert result.requests_count == 1
    assert "...[TRUNC]" in result.text
    assert len(result.text) < 45_000


@dataclass
class _RetryNoDiffPatcher:
    name: str = "retry_no_diff"
    request_context_calls: int = 0

    def request_context(self, *, prompt: str, context_dir, cwd) -> ContextRequestResult:
        del prompt
        del context_dir
        del cwd
        self.request_context_calls += 1
        return ContextRequestResult(
            request_json={
                "need_more_context": False,
                "requests": [],
                "sufficiency_rationale": "ok",
            },
            raw_response='{"need_more_context":false,"requests":[]}',
            meta={},
        )

    def propose_patch(self, *, prompt: str, context_dir, cwd) -> PatchResult:
        del prompt
        del context_dir
        del cwd
        return PatchResult(diff=None, meta={"reason": "no_diff"})


@dataclass
class _CapturePromptNoContextPatcher:
    name: str = "capture_prompt"
    prompts: list[str] = field(default_factory=list)

    def propose_patch(self, *, prompt: str, context_dir, cwd) -> PatchResult:
        del context_dir
        del cwd
        self.prompts.append(prompt)
        return PatchResult(diff=None, meta={"reason": "no_diff"})


@dataclass
class _PlanThenPatchPatcher:
    name: str = "openai"
    plan_prompts: list[str] | None = None
    propose_prompts: list[str] | None = None

    def __post_init__(self) -> None:
        if self.plan_prompts is None:
            self.plan_prompts = []
        if self.propose_prompts is None:
            self.propose_prompts = []

    def plan_patch(self, *, prompt: str, context_dir, cwd) -> PatchPlanResult:
        del context_dir
        del cwd
        assert self.plan_prompts is not None
        self.plan_prompts.append(prompt)
        return PatchPlanResult(
            plan_text=(
                "Diagnosis: helper.value returns the wrong value\n"
                "Why previous attempt failed: no source behavior changed\n"
                "New constraint: edit helper.py, not the test\n"
                "Minimal edit plan: change helper.value to return 2\n"
                "Verification plan: rerun pytest -q"
            ),
            meta={},
        )

    def propose_patch(self, *, prompt: str, context_dir, cwd) -> PatchResult:
        del context_dir
        del cwd
        assert self.propose_prompts is not None
        self.propose_prompts.append(prompt)
        return PatchResult(diff=None, meta={"reason": "no_diff"})


def test_run_one_condition_non_adaptive_staged_request_is_forced_to_monolithic(
    tmp_path, monkeypatch
) -> None:
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir"
    context_dir = tmp_path / "context"
    shutil.copytree(case.path, workdir)
    patcher = _RetryNoDiffPatcher()

    def _fake_run_pytest(cwd, python_args, condition, timeout_sec, output_format):
        del cwd, python_args, condition, timeout_sec, output_format
        return run_eval.RunResult(
            returncode=1,
            duration_sec=0.01,
            stdout="E   AssertionError: expected 2 got 1\n",
            stderr="",
        )

    monkeypatch.setattr(run_eval, "run_pytest", _fake_run_pytest)

    records = run_eval.run_one_condition(
        case=case,
        condition=run_eval.Condition.BASE_REPAIR,
        workdir=workdir,
        context_dir_base=context_dir,
        patcher=patcher,
        k=2,
        output_format="json_compact",
        run_id="run-stage-retry",
        include_files_context=False,
        force_patch_attempt=False,
        context_protocol="staged",
    )

    assert len(records) == 2
    assert all(record["context_protocol"] == "monolithic" for record in records)
    assert patcher.request_context_calls == 0
    assert records[1]["attempt"] == 2


@pytest.mark.parametrize(
    "condition", [run_eval.Condition.BASE_REPAIR, run_eval.Condition.COT_REPAIR]
)
def test_non_adaptive_native_prompt_keeps_pytest_but_excludes_snapshot_and_stage_a(
    tmp_path,
    monkeypatch,
    condition: run_eval.Condition,
) -> None:
    case = _make_case(tmp_path)
    workdir = tmp_path / f"workdir_{condition.value}"
    context_dir = tmp_path / f"context_{condition.value}"
    shutil.copytree(case.path, workdir)
    patcher = _CapturePromptNoContextPatcher()

    monkeypatch.setattr(
        run_eval,
        "run_pytest",
        lambda *args, **kwargs: run_eval.RunResult(
            returncode=1,
            duration_sec=0.01,
            stdout="tests/test_buggy.py::test_f\nE   AssertionError: expected 2 got 1\n",
            stderr="",
        ),
    )
    monkeypatch.setattr(
        run_eval,
        "read_latest_snapshot",
        lambda *_args, **_kwargs: (
            {
                "exception": {"type": "AssertionError", "message": "expected 2 got 1"},
                "frames": [{"file_rel": "buggy.py", "line": 4, "function": "f"}],
            },
            None,
        ),
    )

    records = run_eval.run_one_condition(
        case=case,
        condition=condition,
        workdir=workdir,
        context_dir_base=context_dir,
        patcher=patcher,
        k=1,
        output_format="json_compact",
        run_id=f"run-no-snapshot-{condition.value}",
        include_files_context=True,
        force_patch_attempt=False,
        context_protocol="staged",
    )

    assert len(records) == 1
    assert records[0]["context_protocol"] == "monolithic"
    assert len(patcher.prompts) == 1
    prompt = patcher.prompts[0]
    assert "Pytest output (stdout):" in prompt
    assert "AssertionError: expected 2 got 1" in prompt
    assert "Project files:" in prompt
    assert "=== llmdebug Debug Snapshot ===" not in prompt
    assert "=== llmdebug Snapshot Index ===" not in prompt
    assert "Stage-A requested context:" not in prompt
    assert "Stage-A evidence index:" not in prompt


def test_run_one_condition_retry_plan_then_patch_injects_plan_on_retries(
    tmp_path, monkeypatch
) -> None:
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir_plan_then_patch"
    context_dir = tmp_path / "context_plan_then_patch"
    shutil.copytree(case.path, workdir)

    def _fake_run_pytest(cwd, python_args, condition, timeout_sec, output_format):
        del cwd, python_args, condition, timeout_sec, output_format
        return run_eval.RunResult(
            returncode=1,
            duration_sec=0.01,
            stdout="E   AssertionError: expected 2 got 1\n",
            stderr="",
        )

    patcher = _PlanThenPatchPatcher()
    monkeypatch.setattr(run_eval, "run_pytest", _fake_run_pytest)

    records = run_eval.run_one_condition(
        case=case,
        condition=run_eval.Condition.BASE_REPAIR,
        workdir=workdir,
        context_dir_base=context_dir,
        patcher=patcher,
        k=2,
        output_format="json_compact",
        run_id="run-plan-then-patch",
        include_files_context=False,
        force_patch_attempt=False,
        context_protocol="monolithic",
        prompt_retry_plan_then_patch=True,
    )

    assert len(records) == 2
    assert patcher.plan_prompts is not None
    assert patcher.propose_prompts is not None
    assert len(patcher.plan_prompts) == 1
    assert len(patcher.propose_prompts) == 2
    assert "RETURN ONLY THE REPAIR PLAN" in patcher.plan_prompts[0]
    assert "Structured repair plan from planning step:" not in patcher.propose_prompts[0]
    assert "Structured repair plan from planning step:" in patcher.propose_prompts[1]
    assert "Diagnosis: helper.value returns the wrong value" in patcher.propose_prompts[1]
    assert records[1]["patch_meta"]["retry_plan_then_patch_invoked"] is True
    assert records[1]["patch_meta"]["retry_plan_then_patch_succeeded"] is True


def test_run_one_condition_reuses_file_catalog_when_fs_is_unchanged(tmp_path, monkeypatch) -> None:
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir_cache_hit"
    context_dir = tmp_path / "context_cache_hit"
    shutil.copytree(case.path, workdir)

    calls = {"index": 0, "context": 0}
    real_collect_index = run_eval._collect_file_index_lines
    real_collect_context = run_eval.collect_text_context

    def _counted_collect_index(root):
        calls["index"] += 1
        return real_collect_index(root)

    def _counted_collect_context(root, *, max_total_chars=200_000, file_index_lines=None):
        calls["context"] += 1
        return real_collect_context(
            root,
            max_total_chars=max_total_chars,
            file_index_lines=file_index_lines,
        )

    def _fake_run_pytest(cwd, python_args, condition, timeout_sec, output_format):
        del cwd, python_args, condition, timeout_sec, output_format
        return run_eval.RunResult(
            returncode=1,
            duration_sec=0.01,
            stdout="E   AssertionError: expected 2 got 1\n",
            stderr="",
        )

    monkeypatch.setattr(run_eval, "_collect_file_index_lines", _counted_collect_index)
    monkeypatch.setattr(run_eval, "collect_text_context", _counted_collect_context)
    monkeypatch.setattr(run_eval, "run_pytest", _fake_run_pytest)

    records = run_eval.run_one_condition(
        case=case,
        condition=run_eval.Condition.BASE_REPAIR,
        workdir=workdir,
        context_dir_base=context_dir,
        patcher=_RetryNoDiffPatcher(),
        k=3,
        output_format="json_compact",
        run_id="run-cache-hit",
        include_files_context=True,
        force_patch_attempt=False,
        context_protocol="monolithic",
    )

    assert len(records) == 3
    assert calls["index"] == 1
    assert calls["context"] == 1


@dataclass(frozen=True)
class _AlwaysDiffPatcher:
    name: str = "always_diff"

    def propose_patch(self, *, prompt: str, context_dir, cwd) -> PatchResult:
        del prompt
        del context_dir
        del cwd
        diff = (
            "diff --git a/buggy.py b/buggy.py\n"
            "--- a/buggy.py\n"
            "+++ b/buggy.py\n"
            "@@ -1,5 +1,5 @@\n"
            " import helper\n"
            "\n"
            "\n"
            " def f() -> int:\n"
            "-    return helper.value()\n"
            "+    return helper.value()\n"
        )
        return PatchResult(diff=diff, meta={})


def test_run_one_condition_invalidates_cached_file_catalog_after_apply_success(
    tmp_path, monkeypatch
) -> None:
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir_cache_invalidate"
    context_dir = tmp_path / "context_cache_invalidate"
    shutil.copytree(case.path, workdir)

    calls = {"index": 0}
    real_collect_index = run_eval._collect_file_index_lines

    def _counted_collect_index(root):
        calls["index"] += 1
        return real_collect_index(root)

    def _fake_run_pytest(cwd, python_args, condition, timeout_sec, output_format):
        del cwd, python_args, condition, timeout_sec, output_format
        return run_eval.RunResult(
            returncode=1,
            duration_sec=0.01,
            stdout="E   AssertionError: expected 2 got 1\n",
            stderr="",
        )

    monkeypatch.setattr(run_eval, "_collect_file_index_lines", _counted_collect_index)
    monkeypatch.setattr(run_eval, "run_pytest", _fake_run_pytest)
    monkeypatch.setattr(run_eval, "apply_diff", lambda cwd, diff: (True, "ok"))

    records = run_eval.run_one_condition(
        case=case,
        condition=run_eval.Condition.BASE_REPAIR,
        workdir=workdir,
        context_dir_base=context_dir,
        patcher=_AlwaysDiffPatcher(),
        k=2,
        output_format="json_compact",
        run_id="run-cache-invalidate",
        include_files_context=False,
        force_patch_attempt=False,
        context_protocol="monolithic",
    )

    assert len(records) == 2
    assert calls["index"] == 2
