"""Tests for forced-patch policy fallback when the patcher produces no diff."""

from __future__ import annotations

import shutil
from dataclasses import dataclass

import pytest

from evals import eval_attempt_flow, run_eval
from evals.patchers.base import ContextCompactionResult, ContextRequestResult, PatchResult


@dataclass(frozen=True)
class _NoDiffPatcher:
    name: str = "no_diff"

    def propose_patch(self, *, prompt: str, context_dir, cwd) -> PatchResult:
        return PatchResult(diff=None, meta={"reason": "no_diff"})


@dataclass(frozen=True)
class _RequestFailedNoDiffPatcher:
    name: str = "request_failed_no_diff"

    def propose_patch(self, *, prompt: str, context_dir, cwd) -> PatchResult:
        del prompt, context_dir, cwd
        return PatchResult(
            diff=None,
            meta={"error": "request_failed", "exc": "TimeoutError: timed out patcher"},
        )


@dataclass(frozen=True)
class _ContextLimitNoDiffPatcher:
    name: str = "context_limit_no_diff"

    def propose_patch(self, *, prompt: str, context_dir, cwd) -> PatchResult:
        del prompt, context_dir, cwd
        return PatchResult(
            diff=None,
            meta={
                "error": "http_error",
                "status": 400,
                "body": "Requested prompt is greater than the context length.",
            },
        )


@dataclass(frozen=True)
class _RetryableHttpNoDiffPatcher:
    name: str = "retryable_http_no_diff"

    def propose_patch(self, *, prompt: str, context_dir, cwd) -> PatchResult:
        del prompt, context_dir, cwd
        return PatchResult(
            diff=None,
            meta={
                "error": "http_error",
                "status": 503,
                "body": "backend temporarily unavailable",
            },
        )


@dataclass(frozen=True)
class _CompatReasoningOnlyNoDiffPatcher:
    name: str = "compat_reasoning_only_no_diff"

    def propose_patch(self, *, prompt: str, context_dir, cwd) -> PatchResult:
        del prompt, context_dir, cwd
        return PatchResult(
            diff=None,
            meta={
                "error": "openai_compat_reasoning_only",
                "compatibility_reason": "openai_compat_reasoning_only",
            },
        )


@dataclass(frozen=True)
class _FallbackCompatReasoningOnlyNoDiffPatcher:
    name: str = "fallback_compat_reasoning_only_no_diff"

    def propose_patch(self, *, prompt: str, context_dir, cwd) -> PatchResult:
        del prompt, context_dir, cwd
        return PatchResult(
            diff=None,
            meta={
                "error": "no_json_payload",
                "fallback_meta": {
                    "error": "openai_compat_reasoning_only",
                    "compatibility_reason": "openai_compat_reasoning_only",
                },
            },
        )


@dataclass
class _AlwaysRequestBudgetExceededPatcher:
    name: str = "request_budget_exhausted"
    calls: int = 0

    def propose_patch(self, *, prompt: str, context_dir, cwd) -> PatchResult:
        del prompt, context_dir, cwd
        self.calls += 1
        return PatchResult(
            diff=None,
            meta={
                "error": "request_budget_exceeded",
                "remaining_budget_sec": 0.0,
                "llm_call_count": 0,
                "tokens_prompt": 0,
                "tokens_completion": 0,
                "tokens_total": 0,
            },
        )


@dataclass(frozen=True)
class _BadDiffPatcher:
    name: str = "bad_diff"

    def propose_patch(self, *, prompt: str, context_dir, cwd) -> PatchResult:
        diff = (
            "--- a/buggy.py\n"
            "+++ b/buggy.py\n"
            "@@ -99,1 +99,1 @@\n"
            "-missing_line = 1\n"
            "+missing_line = 2\n"
        )
        return PatchResult(diff=diff, meta={"reason": "bad_diff"})


@dataclass(frozen=True)
class _MalformedStageAPatcher:
    name: str = "malformed_stage_a"

    def request_context(self, *, prompt: str, context_dir, cwd) -> ContextRequestResult:
        del prompt
        del context_dir
        del cwd
        return ContextRequestResult(
            request_json=None,
            raw_response="not-json",
            meta={"error": "bad_json"},
        )

    def propose_patch(self, *, prompt: str, context_dir, cwd) -> PatchResult:
        return PatchResult(diff=None, meta={"reason": "no_diff"})


@dataclass(frozen=True)
class _TestFileDiffPatcher:
    name: str = "test_file_diff"

    def propose_patch(self, *, prompt: str, context_dir, cwd) -> PatchResult:
        diff = (
            "--- a/test_buggy.py\n"
            "+++ b/test_buggy.py\n"
            "@@ -3,3 +3,3 @@\n"
            " \n"
            " def test_f() -> None:\n"
            "-    assert f() == 2\n"
            "+    assert f() == 1\n"
        )
        return PatchResult(diff=diff, meta={"reason": "test_file_edit"})


@dataclass
class _CompactingStageAPatcher:
    name: str = "compacting_stage_a"
    compact_calls: int = 0
    last_patch_prompt: str = ""

    def request_context(self, *, prompt: str, context_dir, cwd) -> ContextRequestResult:
        del prompt, context_dir, cwd
        return ContextRequestResult(
            request_json={
                "need_more_context": True,
                "need_snapshot": True,
                "need_compact_context": True,
                "snapshot_justification": "need frame locals to compare actual vs expected values",
                "compact_context_reason": "snapshot has too much noise",
                "requests": [{"type": "file", "path": "buggy.py"}],
            },
            raw_response='{"need_more_context":true}',
            meta={},
        )

    def compact_context(
        self,
        *,
        prompt: str,
        context_text: str,
        context_dir,
        cwd,
        target_chars: int,
    ) -> ContextCompactionResult:
        del context_dir, cwd
        self.compact_calls += 1
        assert "snapshot has too much noise" in prompt
        assert "Pytest output" in context_text
        compacted = "Root cause hints: buggy.f returns 1 while test expects 2."
        return ContextCompactionResult(
            compacted_text=compacted[:target_chars],
            raw_response='{"compacted_context":"ok"}',
            meta={"llm_call_count": 1, "usage": {"prompt_tokens": 5, "completion_tokens": 3}},
        )

    def propose_patch(self, *, prompt: str, context_dir, cwd) -> PatchResult:
        del context_dir, cwd
        self.last_patch_prompt = prompt
        return PatchResult(diff=None, meta={"reason": "no_diff"})


@dataclass
class _ContextRecoveryDiffPatcher:
    name: str = "context_recovery_diff"
    calls: int = 0

    def propose_patch(self, *, prompt: str, context_dir, cwd) -> PatchResult:
        del prompt, context_dir, cwd
        self.calls += 1
        if self.calls == 1:
            return PatchResult(
                diff=None,
                meta={
                    "error": "http_error",
                    "status": 400,
                    "body": "Requested prompt is greater than the context length.",
                },
            )
        return PatchResult(
            diff=(
                "diff --git a/buggy.py b/buggy.py\n"
                "--- a/buggy.py\n"
                "+++ b/buggy.py\n"
                "@@ -1,2 +1,2 @@\n"
                " def f() -> int:\n"
                "-    return 1\n"
                "+    return 2\n"
            ),
            meta={
                "reason": "recovered_diff",
                "llm_call_count": 1,
                "usage": {"prompt_tokens": 7, "completion_tokens": 3, "total_tokens": 10},
            },
        )


@dataclass
class _OpenAINoDiffRecoveryPatcher:
    name: str = "openai"
    calls: int = 0

    def propose_patch(self, *, prompt: str, context_dir, cwd) -> PatchResult:
        del prompt, context_dir, cwd
        self.calls += 1
        if self.calls == 1:
            return PatchResult(
                diff=None,
                meta={
                    "error": "no_json_payload",
                    "reason": "invalid_response",
                },
            )
        return PatchResult(
            diff=(
                "diff --git a/buggy.py b/buggy.py\n"
                "--- a/buggy.py\n"
                "+++ b/buggy.py\n"
                "@@ -1,2 +1,2 @@\n"
                " def f() -> int:\n"
                "-    return 1\n"
                "+    return 2\n"
            ),
            meta={
                "reason": "recovered_diff",
                "llm_call_count": 1,
                "usage": {"prompt_tokens": 9, "completion_tokens": 4, "total_tokens": 13},
            },
        )


@pytest.mark.parametrize(
    ("meta", "expected"),
    [
        ({"error": "request_failed"}, "transport"),
        ({"error": "http_error", "status": 503}, "backend_unavailable"),
        ({"error": "http_error", "status": 500}, "transport"),
        ({"error": "request_budget_exceeded"}, ""),
        ({"error": "bad_request_url"}, ""),
    ],
)
def test_transport_recovery_failure_kind_excludes_local_failures(meta, expected) -> None:
    assert run_eval._transport_recovery_failure_kind_from_meta(meta) == expected


@pytest.mark.parametrize(
    ("meta", "expected"),
    [
        ({"error": "request_failed"}, "transport"),
        ({"error": "http_error", "status": 503}, "backend_unavailable"),
        ({"error": "http_error", "status": 400}, ""),
        ({"error": "http_error", "status": 401}, ""),
        ({"error": "http_error", "status": 404}, ""),
        ({"error": "request_budget_exceeded"}, ""),
    ],
)
def test_request_context_recovery_failure_kind_only_retries_transportish_errors(
    meta,
    expected,
) -> None:
    assert run_eval._request_context_recovery_failure_kind_from_meta(meta) == expected


@dataclass(frozen=True)
class _MarkerOnlyDiffPatcher:
    name: str = "marker_only_diff"

    def propose_patch(self, *, prompt: str, context_dir, cwd) -> PatchResult:
        del prompt, context_dir, cwd
        return PatchResult(
            diff=(
                "diff --git a/buggy.py b/buggy.py\n"
                "--- a/buggy.py\n"
                "+++ b/buggy.py\n"
                "@@ -1,2 +1,3 @@\n"
                " def f() -> int:\n"
                "     return 1\n"
                "+# llmdebug-force-attempt-1\n"
            ),
            meta={"reason": "marker_only"},
        )


def _make_case(tmp_path) -> run_eval.CaseSpec:
    case_dir = tmp_path / "case_src"
    case_dir.mkdir(parents=True, exist_ok=True)
    (case_dir / "buggy.py").write_text("def f() -> int:\n    return 1\n", encoding="utf-8")
    (case_dir / "test_buggy.py").write_text(
        "from buggy import f\n\n\ndef test_f() -> None:\n    assert f() == 2\n",
        encoding="utf-8",
    )
    return run_eval.CaseSpec(
        case_id="force_patch_case",
        path=case_dir,
        category="traceback_controls",
        difficulty="easy",
        snapshot_dependency="none",
        expected_exception="AssertionError",
        tags=[],
        python_args=["-m", "pytest", "-q"],
        timeout_sec=30,
        description="force-patch policy test",
    )


def test_runner_force_patch_attempt_applies_when_no_diff(tmp_path, monkeypatch) -> None:
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir"
    context_dir = tmp_path / "context"
    shutil.copytree(case.path, workdir)

    def _fake_run_pytest(cwd, python_args, condition, timeout_sec, output_format, **_kwargs):
        return run_eval.RunResult(
            returncode=1,
            duration_sec=0.01,
            stdout="E   AssertionError: expected 2 got 1\n",
            stderr="",
        )

    monkeypatch.setattr(eval_attempt_flow, "run_pytest", _fake_run_pytest)
    monkeypatch.setattr(eval_attempt_flow, "apply_diff", lambda cwd, diff: (True, ""))

    records = run_eval.run_one_condition(
        case=case,
        condition=run_eval.Condition.BASE_REPAIR,
        workdir=workdir,
        context_dir_base=context_dir,
        patcher=_NoDiffPatcher(),
        k=1,
        output_format="json_compact",
        run_id="run",
        include_files_context=False,
        force_patch_attempt=True,
    )

    assert len(records) == 1
    record = records[0]
    assert record["note"] == "forced_noop_patch"
    assert record["patch_meta"]["runner_forced_patch_attempt"] is True
    assert record["patch_meta"]["runner_forced_patch_target"] == "buggy.py"
    assert record["patch_meta"]["runner_forced_patch_marker_only"] is True
    assert record["patch_meta"]["runner_forced_patch_disabled_after_noop"] is True


def test_runner_no_force_patch_attempt_preserves_no_patch_proposed(tmp_path, monkeypatch) -> None:
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir_no_force"
    context_dir = tmp_path / "context_no_force"
    shutil.copytree(case.path, workdir)

    def _fake_run_pytest(cwd, python_args, condition, timeout_sec, output_format, **_kwargs):
        return run_eval.RunResult(
            returncode=1,
            duration_sec=0.01,
            stdout="E   AssertionError: expected 2 got 1\n",
            stderr="",
        )

    apply_called = {"count": 0}

    def _fake_apply_diff(cwd, diff):
        apply_called["count"] += 1
        return True, ""

    monkeypatch.setattr(eval_attempt_flow, "run_pytest", _fake_run_pytest)
    monkeypatch.setattr(eval_attempt_flow, "apply_diff", _fake_apply_diff)

    records = run_eval.run_one_condition(
        case=case,
        condition=run_eval.Condition.BASE_REPAIR,
        workdir=workdir,
        context_dir_base=context_dir,
        patcher=_NoDiffPatcher(),
        k=1,
        output_format="json_compact",
        run_id="run",
        include_files_context=False,
        force_patch_attempt=False,
    )

    assert len(records) == 1
    record = records[0]
    assert record["note"] == "no_patch_proposed"
    assert apply_called["count"] == 0


def test_runner_force_patch_attempt_skips_on_request_failure(tmp_path, monkeypatch) -> None:
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir_request_failed_skip"
    context_dir = tmp_path / "context_request_failed_skip"
    shutil.copytree(case.path, workdir)

    monkeypatch.setattr(
        eval_attempt_flow,
        "run_pytest",
        lambda cwd,
        python_args,
        condition,
        timeout_sec,
        output_format,
        **_kwargs: run_eval.RunResult(
            returncode=1,
            duration_sec=0.01,
            stdout="E   AssertionError: expected 2 got 1\n",
            stderr="",
        ),
    )
    apply_called = {"count": 0}

    def _fake_apply_diff(cwd, diff):
        del cwd, diff
        apply_called["count"] += 1
        return True, ""

    monkeypatch.setattr(eval_attempt_flow, "apply_diff", _fake_apply_diff)

    records = run_eval.run_one_condition(
        case=case,
        condition=run_eval.Condition.BASE_REPAIR,
        workdir=workdir,
        context_dir_base=context_dir,
        patcher=_RequestFailedNoDiffPatcher(),
        k=1,
        output_format="json_compact",
        run_id="run-request-failed-skip",
        include_files_context=False,
        force_patch_attempt=True,
    )

    assert len(records) == 1
    record = records[0]
    assert record["note"] == "patcher_request_failed"
    assert apply_called["count"] == 0
    assert record["patch_meta"]["patcher_request_failed"] is True
    assert record["patch_meta"]["patcher_request_failure_kind"] == "transport"
    assert record["patch_meta"]["runner_forced_patch_skipped"] is True
    assert record["patch_meta"]["runner_forced_patch_skip_reason"] == "transport_or_context_failure"
    assert "runner_forced_patch_attempt" not in record["patch_meta"]


def test_runner_force_patch_attempt_skips_on_context_limit_failure(tmp_path, monkeypatch) -> None:
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir_context_limit_skip"
    context_dir = tmp_path / "context_context_limit_skip"
    shutil.copytree(case.path, workdir)

    monkeypatch.setattr(
        eval_attempt_flow,
        "run_pytest",
        lambda cwd,
        python_args,
        condition,
        timeout_sec,
        output_format,
        **_kwargs: run_eval.RunResult(
            returncode=1,
            duration_sec=0.01,
            stdout="E   AssertionError: expected 2 got 1\n",
            stderr="",
        ),
    )
    apply_called = {"count": 0}

    def _fake_apply_diff(cwd, diff):
        del cwd, diff
        apply_called["count"] += 1
        return True, ""

    monkeypatch.setattr(eval_attempt_flow, "apply_diff", _fake_apply_diff)

    records = run_eval.run_one_condition(
        case=case,
        condition=run_eval.Condition.BASE_REPAIR,
        workdir=workdir,
        context_dir_base=context_dir,
        patcher=_ContextLimitNoDiffPatcher(),
        k=1,
        output_format="json_compact",
        run_id="run-context-limit-skip",
        include_files_context=False,
        force_patch_attempt=True,
    )

    assert len(records) == 1
    record = records[0]
    assert record["note"] == "patcher_request_failed"
    assert apply_called["count"] == 0
    assert record["patch_meta"]["patcher_request_failed"] is True
    assert record["patch_meta"]["patcher_request_failure_kind"] == "context_limit"
    assert record["patch_meta"]["runner_forced_patch_skipped"] is True
    assert record["patch_meta"]["runner_forced_patch_skip_reason"] == "transport_or_context_failure"
    assert "runner_forced_patch_attempt" not in record["patch_meta"]


def test_runner_force_patch_attempt_skips_on_retryable_http_transport_failure(
    tmp_path, monkeypatch
) -> None:
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir_retryable_http_skip"
    context_dir = tmp_path / "context_retryable_http_skip"
    shutil.copytree(case.path, workdir)

    monkeypatch.setattr(
        eval_attempt_flow,
        "run_pytest",
        lambda cwd,
        python_args,
        condition,
        timeout_sec,
        output_format,
        **_kwargs: run_eval.RunResult(
            returncode=1,
            duration_sec=0.01,
            stdout="E   AssertionError: expected 2 got 1\n",
            stderr="",
        ),
    )
    apply_called = {"count": 0}

    def _fake_apply_diff(cwd, diff):
        del cwd, diff
        apply_called["count"] += 1
        return True, ""

    monkeypatch.setattr(eval_attempt_flow, "apply_diff", _fake_apply_diff)

    records = run_eval.run_one_condition(
        case=case,
        condition=run_eval.Condition.BASE_REPAIR,
        workdir=workdir,
        context_dir_base=context_dir,
        patcher=_RetryableHttpNoDiffPatcher(),
        k=1,
        output_format="json_compact",
        run_id="run-retryable-http-skip",
        include_files_context=False,
        force_patch_attempt=True,
    )

    assert len(records) == 1
    record = records[0]
    assert record["note"] == "patcher_request_failed"
    assert apply_called["count"] == 0
    assert record["patch_meta"]["patcher_request_failed"] is True
    assert record["patch_meta"]["patcher_request_failure_kind"] == "transport"
    assert record["patch_meta"]["runner_forced_patch_skipped"] is True
    assert record["patch_meta"]["runner_forced_patch_skip_reason"] == "transport_or_context_failure"
    assert "runner_forced_patch_attempt" not in record["patch_meta"]


def test_runner_force_patch_attempt_skips_on_compat_reasoning_only_failure(
    tmp_path, monkeypatch
) -> None:
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir_compat_reasoning_only_skip"
    context_dir = tmp_path / "context_compat_reasoning_only_skip"
    shutil.copytree(case.path, workdir)

    monkeypatch.setattr(
        eval_attempt_flow,
        "run_pytest",
        lambda cwd,
        python_args,
        condition,
        timeout_sec,
        output_format,
        **_kwargs: run_eval.RunResult(
            returncode=1,
            duration_sec=0.01,
            stdout="E   AssertionError: expected 2 got 1\n",
            stderr="",
        ),
    )
    apply_called = {"count": 0}

    def _fake_apply_diff(cwd, diff):
        del cwd, diff
        apply_called["count"] += 1
        return True, ""

    monkeypatch.setattr(eval_attempt_flow, "apply_diff", _fake_apply_diff)

    records = run_eval.run_one_condition(
        case=case,
        condition=run_eval.Condition.BASE_REPAIR,
        workdir=workdir,
        context_dir_base=context_dir,
        patcher=_CompatReasoningOnlyNoDiffPatcher(),
        k=1,
        output_format="json_compact",
        run_id="run-compat-reasoning-only-skip",
        include_files_context=False,
        force_patch_attempt=True,
    )

    assert len(records) == 1
    record = records[0]
    assert record["note"] == "patcher_request_failed"
    assert apply_called["count"] == 0
    assert record["patch_meta"]["patcher_request_failed"] is True
    assert record["patch_meta"]["patcher_request_failure_kind"] == "compat_reasoning_only"
    assert record["patch_meta"]["runner_forced_patch_skipped"] is True
    assert record["patch_meta"]["runner_forced_patch_skip_reason"] == "compatibility_failure"
    assert "runner_forced_patch_attempt" not in record["patch_meta"]


def test_runner_force_patch_attempt_skips_on_fallback_compat_reasoning_only_failure(
    tmp_path, monkeypatch
) -> None:
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir_fallback_compat_reasoning_only_skip"
    context_dir = tmp_path / "context_fallback_compat_reasoning_only_skip"
    shutil.copytree(case.path, workdir)

    monkeypatch.setattr(
        eval_attempt_flow,
        "run_pytest",
        lambda cwd,
        python_args,
        condition,
        timeout_sec,
        output_format,
        **_kwargs: run_eval.RunResult(
            returncode=1,
            duration_sec=0.01,
            stdout="E   AssertionError: expected 2 got 1\n",
            stderr="",
        ),
    )
    apply_called = {"count": 0}

    def _fake_apply_diff(cwd, diff):
        del cwd, diff
        apply_called["count"] += 1
        return True, ""

    monkeypatch.setattr(eval_attempt_flow, "apply_diff", _fake_apply_diff)

    records = run_eval.run_one_condition(
        case=case,
        condition=run_eval.Condition.BASE_REPAIR,
        workdir=workdir,
        context_dir_base=context_dir,
        patcher=_FallbackCompatReasoningOnlyNoDiffPatcher(),
        k=1,
        output_format="json_compact",
        run_id="run-fallback-compat-reasoning-only-skip",
        include_files_context=False,
        force_patch_attempt=True,
    )

    assert len(records) == 1
    record = records[0]
    assert record["note"] == "patcher_request_failed"
    assert apply_called["count"] == 0
    assert record["patch_meta"]["patcher_request_failed"] is True
    assert record["patch_meta"]["patcher_request_failure_kind"] == "compat_reasoning_only"
    assert record["patch_meta"]["runner_forced_patch_skipped"] is True
    assert record["patch_meta"]["runner_forced_patch_skip_reason"] == "compatibility_failure"
    assert "runner_forced_patch_attempt" not in record["patch_meta"]


def test_request_budget_exceeded_retries_by_default(tmp_path, monkeypatch) -> None:
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir_request_budget_retry"
    context_dir = tmp_path / "context_request_budget_retry"
    shutil.copytree(case.path, workdir)

    monkeypatch.setattr(
        eval_attempt_flow,
        "run_pytest",
        lambda cwd,
        python_args,
        condition,
        timeout_sec,
        output_format,
        **_kwargs: run_eval.RunResult(
            returncode=1,
            duration_sec=0.01,
            stdout="E   AssertionError: expected 2 got 1\n",
            stderr="",
        ),
    )

    apply_called = {"count": 0}

    def _fake_apply_diff(cwd, diff):
        del cwd, diff
        apply_called["count"] += 1
        return True, ""

    patcher = _AlwaysRequestBudgetExceededPatcher()
    monkeypatch.setattr(eval_attempt_flow, "apply_diff", _fake_apply_diff)

    records = run_eval.run_one_condition(
        case=case,
        condition=run_eval.Condition.BASE_REPAIR,
        workdir=workdir,
        context_dir_base=context_dir,
        patcher=patcher,
        k=2,
        output_format="json_compact",
        run_id="run-budget-retry-default",
        include_files_context=False,
        force_patch_attempt=True,
        case_fail_fast_on_request_budget_exceeded=False,
    )

    assert [record["note"] for record in records] == [
        "patcher_request_failed",
        "patcher_request_failed",
    ]
    assert apply_called["count"] == 0
    assert patcher.calls >= 2


def test_request_budget_exceeded_case_fail_fast_stops_additional_attempts(
    tmp_path, monkeypatch
) -> None:
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir_request_budget_fail_fast"
    context_dir = tmp_path / "context_request_budget_fail_fast"
    shutil.copytree(case.path, workdir)

    monkeypatch.setattr(
        eval_attempt_flow,
        "run_pytest",
        lambda cwd,
        python_args,
        condition,
        timeout_sec,
        output_format,
        **_kwargs: run_eval.RunResult(
            returncode=1,
            duration_sec=0.01,
            stdout="E   AssertionError: expected 2 got 1\n",
            stderr="",
        ),
    )
    apply_called = {"count": 0}

    def _fake_apply_diff(cwd, diff):
        del cwd, diff
        apply_called["count"] += 1
        return True, ""

    patcher = _AlwaysRequestBudgetExceededPatcher()
    monkeypatch.setattr(eval_attempt_flow, "apply_diff", _fake_apply_diff)

    records = run_eval.run_one_condition(
        case=case,
        condition=run_eval.Condition.BASE_REPAIR,
        workdir=workdir,
        context_dir_base=context_dir,
        patcher=patcher,
        k=2,
        output_format="json_compact",
        run_id="run-budget-fail-fast",
        include_files_context=False,
        force_patch_attempt=True,
        case_fail_fast_on_request_budget_exceeded=True,
    )

    assert len(records) == 1
    record = records[0]
    assert record["note"] == "patcher_request_failed"
    assert record["patch_meta"]["patcher_request_failure_error"] == "request_budget_exceeded"
    assert record["patch_meta"]["case_fail_fast_triggered"] is True
    assert record["patch_meta"]["case_fail_fast_reason"] == "request_budget_exceeded"
    assert apply_called["count"] == 0
    assert patcher.calls >= 1


def test_runner_force_patch_recovers_after_apply_failure(tmp_path, monkeypatch) -> None:
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir_recovery"
    context_dir = tmp_path / "context_recovery"
    shutil.copytree(case.path, workdir)

    def _fake_run_pytest(cwd, python_args, condition, timeout_sec, output_format, **_kwargs):
        return run_eval.RunResult(
            returncode=1,
            duration_sec=0.01,
            stdout="E   AssertionError: expected 2 got 1\n",
            stderr="",
        )

    apply_attempts = {"count": 0}

    def _fake_apply_diff(cwd, diff):
        apply_attempts["count"] += 1
        if "llmdebug-force-attempt-1" in diff:
            return True, ""
        return False, "simulated hunk failure"

    monkeypatch.setattr(eval_attempt_flow, "run_pytest", _fake_run_pytest)
    monkeypatch.setattr(eval_attempt_flow, "apply_diff", _fake_apply_diff)

    records = run_eval.run_one_condition(
        case=case,
        condition=run_eval.Condition.BASE_REPAIR,
        workdir=workdir,
        context_dir_base=context_dir,
        patcher=_BadDiffPatcher(),
        k=1,
        output_format="json_compact",
        run_id="run",
        include_files_context=False,
        force_patch_attempt=True,
    )

    assert len(records) == 1
    record = records[0]
    # ATT-07 / D8: a runner marker is not the model's patch, so the attempt stays an
    # apply failure, the metadata describes the applied marker diff and the marker is
    # reverted afterwards.
    assert record["note"] == "patch_apply_failed"
    assert record["success"] is False
    assert record["patch_meta"]["runner_forced_patch_recovery"] is True
    assert record["patch_meta"]["runner_forced_patch_recovery_target"] == "buggy.py"
    assert record["patch_meta"]["runner_forced_patch_recovery_reverted"] is True
    assert record["patch_meta"]["runner_forced_patch_recovery_tests_passed"] is False
    assert (
        record["patch_meta"]["model_patch_diff_xxh3_128"]
        != (record["patch_meta"]["patch_diff_xxh3_128"])
    )
    assert "llmdebug-force-attempt" not in (workdir / "buggy.py").read_text(encoding="utf-8")
    assert apply_attempts["count"] == 2

    recovery_diff = (context_dir / "attempt_01" / "patch_recovery.diff").read_text(encoding="utf-8")
    assert "llmdebug-force-attempt-1" in recovery_diff


def test_runner_no_force_patch_keeps_apply_failure(tmp_path, monkeypatch) -> None:
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir_recovery_off"
    context_dir = tmp_path / "context_recovery_off"
    shutil.copytree(case.path, workdir)

    def _fake_run_pytest(cwd, python_args, condition, timeout_sec, output_format, **_kwargs):
        return run_eval.RunResult(
            returncode=1,
            duration_sec=0.01,
            stdout="E   AssertionError: expected 2 got 1\n",
            stderr="",
        )

    monkeypatch.setattr(eval_attempt_flow, "run_pytest", _fake_run_pytest)
    monkeypatch.setattr(
        eval_attempt_flow, "apply_diff", lambda cwd, diff: (False, "simulated hunk failure")
    )

    records = run_eval.run_one_condition(
        case=case,
        condition=run_eval.Condition.BASE_REPAIR,
        workdir=workdir,
        context_dir_base=context_dir,
        patcher=_BadDiffPatcher(),
        k=1,
        output_format="json_compact",
        run_id="run",
        include_files_context=False,
        force_patch_attempt=False,
    )

    assert len(records) == 1
    record = records[0]
    assert record["note"] == "patch_apply_failed"
    assert "runner_forced_patch_recovery" not in record["patch_meta"]


def test_non_adaptive_staged_context_ignores_stage_a_request_path(tmp_path, monkeypatch) -> None:
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir_stage_a_fallback"
    context_dir = tmp_path / "context_stage_a_fallback"
    shutil.copytree(case.path, workdir)

    def _fake_run_pytest(cwd, python_args, condition, timeout_sec, output_format, **_kwargs):
        return run_eval.RunResult(
            returncode=1,
            duration_sec=0.01,
            stdout="E   AssertionError: expected 2 got 1\n",
            stderr="",
        )

    monkeypatch.setattr(eval_attempt_flow, "run_pytest", _fake_run_pytest)
    monkeypatch.setattr(eval_attempt_flow, "apply_diff", lambda cwd, diff: (True, ""))

    records = run_eval.run_one_condition(
        case=case,
        condition=run_eval.Condition.BASE_REPAIR,
        workdir=workdir,
        context_dir_base=context_dir,
        patcher=_MalformedStageAPatcher(),
        k=1,
        output_format="json_compact",
        run_id="run-stage-a-fallback",
        include_files_context=False,
        force_patch_attempt=False,
        context_protocol="staged",
    )

    assert len(records) == 1
    record = records[0]
    assert record["attempt_metadata"]["stage_a_request_success"] is None
    assert record["attempt_metadata"]["stage_a_requests_count"] == 0
    assert record["patch_meta"]["context_protocol"] == "monolithic"
    assert "stage_a_request_success" not in record["patch_meta"]
    assert "stage_a_used_fallback" not in record["patch_meta"]


def test_patch_metrics_detect_test_file_edits(tmp_path, monkeypatch) -> None:
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir_test_metric"
    context_dir = tmp_path / "context_test_metric"
    shutil.copytree(case.path, workdir)

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
    monkeypatch.setattr(eval_attempt_flow, "apply_diff", lambda cwd, diff: (True, ""))

    records = run_eval.run_one_condition(
        case=case,
        condition=run_eval.Condition.BASE_REPAIR,
        workdir=workdir,
        context_dir_base=context_dir,
        patcher=_TestFileDiffPatcher(),
        k=1,
        output_format="json_compact",
        run_id="run-test-metric",
        include_files_context=False,
        force_patch_attempt=False,
    )

    assert len(records) == 1
    record = records[0]
    patch_meta = record["patch_meta"]
    assert patch_meta["patch_touches_tests"] is True
    assert int(patch_meta["patch_test_file_count"]) == 1
    assert int(patch_meta["patch_file_count"]) >= 1
    assert "test_buggy.py" in patch_meta["patch_test_file_paths"]


def test_context_compaction_stage_a_request_replaces_raw_sections(tmp_path, monkeypatch) -> None:
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir_context_compaction"
    context_dir = tmp_path / "context_context_compaction"
    shutil.copytree(case.path, workdir)
    patcher = _CompactingStageAPatcher()

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
    monkeypatch.setattr(eval_attempt_flow, "apply_diff", lambda cwd, diff: (True, ""))
    monkeypatch.setattr(
        eval_attempt_flow,
        "read_latest_snapshot",
        lambda *_args, **_kwargs: (
            {
                "exception": {"type": "AssertionError", "message": "expected 2 got 1"},
                "frames": [
                    {
                        "file_rel": "buggy.py",
                        "line": 2,
                        "function": "f",
                        "code": "return 1",
                        "locals": {},
                        "locals_meta": {},
                    }
                ],
            },
            None,
        ),
    )

    records = run_eval.run_one_condition(
        case=case,
        condition=run_eval.Condition.LAYERED_ADAPTIVE_GATED,
        workdir=workdir,
        context_dir_base=context_dir,
        patcher=patcher,
        k=1,
        output_format="json_compact",
        run_id="run-context-compaction",
        include_files_context=False,
        force_patch_attempt=False,
        context_protocol="staged",
        context_compaction_mode="auto_overflow",
        context_compaction_scope="snapshot,pytest,files",
        adaptive_gate_policy="aggressive",
    )

    assert len(records) == 1
    assert patcher.compact_calls == 1
    assert "Compacted context summary:" in patcher.last_patch_prompt
    assert "=== llmdebug Debug Snapshot ===" not in patcher.last_patch_prompt
    record = records[0]
    assert record["patch_meta"]["context_compaction_invoked"] is True
    assert record["patch_meta"]["context_compaction_success"] is True
    assert record["patch_meta"]["context_compaction_trigger"] == "stage_a_requested"
    assert "snapshot" in record["patch_meta"]["context_compaction_replaced_sections"]


def test_context_recovery_retries_once_with_compact_budget(tmp_path, monkeypatch) -> None:
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir_context_recovery_retry"
    context_dir = tmp_path / "context_context_recovery_retry"
    shutil.copytree(case.path, workdir)
    patcher = _ContextRecoveryDiffPatcher()

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
    monkeypatch.setattr(eval_attempt_flow, "apply_diff", lambda cwd, diff: (True, ""))

    records = run_eval.run_one_condition(
        case=case,
        condition=run_eval.Condition.BASE_REPAIR,
        workdir=workdir,
        context_dir_base=context_dir,
        patcher=patcher,
        k=1,
        output_format="json_compact",
        run_id="run-context-recovery-retry",
        include_files_context=False,
        force_patch_attempt=False,
    )

    assert len(records) == 1
    assert patcher.calls == 2
    record = records[0]
    assert record["note"] == "tests_still_failing"
    assert record["patch_meta"]["context_recovery_invoked"] is True
    assert record["patch_meta"]["context_recovery_retry_success"] is True
    assert record["patch_meta"]["context_recovery_meta"]["usage"]["total_tokens"] == 10
    tokens = record["attempt_metadata"]["token_breakdown"]
    assert tokens["recovery_tokens_total"] == 10
    assert tokens["stage_b_tokens_total"] == 0


def test_openai_no_diff_recovery_retries_once_before_forced_patch(tmp_path, monkeypatch) -> None:
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir_openai_no_diff_recovery"
    context_dir = tmp_path / "context_openai_no_diff_recovery"
    shutil.copytree(case.path, workdir)
    patcher = _OpenAINoDiffRecoveryPatcher()

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
    monkeypatch.setattr(eval_attempt_flow, "apply_diff", lambda cwd, diff: (True, ""))

    records = run_eval.run_one_condition(
        case=case,
        condition=run_eval.Condition.BASE_REPAIR,
        workdir=workdir,
        context_dir_base=context_dir,
        patcher=patcher,
        k=1,
        output_format="json_compact",
        run_id="run-openai-no-diff-recovery",
        include_files_context=False,
        force_patch_attempt=False,
    )

    assert len(records) == 1
    assert patcher.calls == 2
    record = records[0]
    assert record["note"] == "tests_still_failing"
    assert record["patch_meta"]["no_diff_recovery_invoked"] is True
    assert record["patch_meta"]["no_diff_recovery_retry_success"] is True
    assert record["patch_meta"]["no_diff_recovery_meta"]["usage"]["total_tokens"] == 13
    tokens = record["attempt_metadata"]["token_breakdown"]
    assert tokens["recovery_tokens_total"] == 13
    assert tokens["stage_b_tokens_total"] == 0
    assert "runner_forced_patch_attempt" not in record["patch_meta"]


def test_marker_only_diff_is_treated_as_forced_noop(tmp_path, monkeypatch) -> None:
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir_marker_noop"
    context_dir = tmp_path / "context_marker_noop"
    shutil.copytree(case.path, workdir)

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
    monkeypatch.setattr(eval_attempt_flow, "apply_diff", lambda cwd, diff: (True, ""))

    records = run_eval.run_one_condition(
        case=case,
        condition=run_eval.Condition.BASE_REPAIR,
        workdir=workdir,
        context_dir_base=context_dir,
        patcher=_MarkerOnlyDiffPatcher(),
        k=1,
        output_format="json_compact",
        run_id="run-marker-noop",
        include_files_context=False,
        force_patch_attempt=True,
    )

    assert len(records) == 1
    record = records[0]
    assert record["note"] == "forced_noop_patch"
    assert record["patch_meta"]["runner_forced_patch_marker_only"] is True
    assert record["patch_meta"]["runner_forced_patch_disabled_after_noop"] is True
