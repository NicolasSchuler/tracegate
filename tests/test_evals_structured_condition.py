"""Tests for structured condition handling and context-request flow in eval runs."""

from __future__ import annotations

import shutil
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace

import pytest

from evals import eval_attempt_flow, pytest_runner, run_eval
from evals.patchers.base import ContextRequestResult, PatchResult
from evals.run_context import RunContext
from llmdebug import _json as json


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
    (case_dir / "notes.md").write_text("unrelated large context marker\n", encoding="utf-8")
    return run_eval.CaseSpec(
        case_id="structured_case",
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
class _SnapshotContextFilePatcher:
    name: str = "snapshot_context_file"
    observed_message_lengths: list[int] = field(default_factory=list)
    observed_marker_present: list[bool] = field(default_factory=list)

    def propose_patch(self, *, prompt: str, context_dir, cwd) -> PatchResult:
        del prompt, cwd
        snapshot_payload = json.loads((context_dir / "snapshot.json").read_text(encoding="utf-8"))
        exception_payload = snapshot_payload.get("exception") or {}
        if not isinstance(exception_payload, dict):
            exception_payload = {}
        message = str(exception_payload.get("message") or "")
        self.observed_message_lengths.append(len(message))
        self.observed_marker_present.append("snapshot string truncated" in message)
        return PatchResult(diff=None, meta={"reason": "no_diff"})


@dataclass
class _CapturePromptPatcher:
    name: str = "capture_prompt"
    prompts: list[str] = field(default_factory=list)
    propose_calls: int = 0

    def request_context(self, *, prompt: str, context_dir, cwd) -> ContextRequestResult:
        del prompt, context_dir, cwd
        return ContextRequestResult(
            request_json={
                "need_more_context": True,
                "requests": [
                    {"type": "file", "path": "buggy.py"},
                    {"type": "file", "path": "test_buggy.py"},
                    {"type": "file", "path": "helper.py"},
                ],
                "sufficiency_rationale": "need source files",
            },
            raw_response='{"need_more_context":true}',
            meta={},
        )

    def propose_patch(self, *, prompt: str, context_dir, cwd) -> PatchResult:
        del context_dir, cwd
        self.propose_calls += 1
        self.prompts.append(prompt)
        return PatchResult(diff=None, meta={"reason": "no_diff"})


def test_structured_condition_uses_seed_bundle_without_project_dump(tmp_path, monkeypatch) -> None:
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir"
    context_dir = tmp_path / "context"
    shutil.copytree(case.path, workdir)

    patcher = _CapturePromptPatcher()

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
        k=1,
        output_format="json_compact",
        run_id="run-structured-seed",
        include_files_context=True,
        force_patch_attempt=False,
        context_protocol="staged",
        adaptive_gate_policy="aggressive",
    )

    assert len(records) == 1
    assert patcher.propose_calls == 1
    prompt = patcher.prompts[0]
    assert "Retry packet (delta-first, structured):" in prompt
    assert "Stage-A requested context:" in prompt
    assert "--- buggy.py ---" in prompt
    assert "--- test_buggy.py ---" in prompt
    assert "--- helper.py ---" in prompt
    assert "Project files:" not in prompt
    assert "unrelated large context marker" not in prompt


def test_snapshot_artifact_cap_preserves_full_snapshot_for_patcher_context_file(
    tmp_path,
    monkeypatch,
) -> None:
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir_snapshot_artifact_cap"
    context_dir = tmp_path / "context_snapshot_artifact_cap"
    shutil.copytree(case.path, workdir)

    patcher = _SnapshotContextFilePatcher()
    huge_message = "X" * 1000
    snapshot_payload = _snapshot_payload()
    snapshot_payload["exception"]["message"] = huge_message

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
        lambda *_args, **_kwargs: (snapshot_payload, None),
    )

    records = run_eval.run_one_condition(
        case=case,
        condition=run_eval.Condition.LAYERED_ADAPTIVE_GATED,
        workdir=workdir,
        context_dir_base=context_dir,
        patcher=patcher,
        k=1,
        output_format="json_compact",
        run_id="run-snapshot-artifact-cap",
        include_files_context=True,
        force_patch_attempt=False,
        snapshot_artifact_max_string_chars=80,
    )

    assert len(records) == 1
    assert patcher.observed_message_lengths == [len(huge_message)]
    assert patcher.observed_marker_present == [False]

    persisted_snapshot = json.loads(
        (context_dir / "attempt_01" / "snapshot.json").read_text("utf-8")
    )
    persisted_exception = persisted_snapshot.get("exception") or {}
    persisted_message = str(persisted_exception.get("message") or "")
    assert len(persisted_message) <= 80
    assert "snapshot string truncated" in persisted_message


@dataclass
class _SameEvidencePatcher:
    name: str = "same_evidence"
    propose_calls: int = 0

    def request_context(self, *, prompt: str, context_dir, cwd) -> ContextRequestResult:
        del prompt, context_dir, cwd
        return ContextRequestResult(
            request_json={
                "need_more_context": True,
                "requests": [{"type": "file", "path": "buggy.py"}],
                "sufficiency_rationale": "need buggy file",
            },
            raw_response='{"need_more_context":true,"requests":[{"type":"file","path":"buggy.py"}]}',
            meta={},
        )

    def propose_patch(self, *, prompt: str, context_dir, cwd) -> PatchResult:
        del prompt, context_dir, cwd
        self.propose_calls += 1
        return PatchResult(diff=None, meta={"reason": "no_diff"})


def test_structured_condition_retry_gate_blocks_same_signature_without_new_evidence(
    tmp_path,
    monkeypatch,
) -> None:
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir_retry_block"
    context_dir = tmp_path / "context_retry_block"
    shutil.copytree(case.path, workdir)
    patcher = _SameEvidencePatcher()

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
    emitted_records: list[dict[str, object]] = []

    records = run_eval.run_one_condition(
        case=case,
        condition=run_eval.Condition.LAYERED_ADAPTIVE_GATED,
        workdir=workdir,
        context_dir_base=context_dir,
        patcher=patcher,
        k=2,
        output_format="json_compact",
        run_id="run-structured-gate",
        include_files_context=False,
        force_patch_attempt=False,
        context_protocol="staged",
        adaptive_gate_policy="aggressive",
        adaptive_max_helper_calls=2,
        record_sink=emitted_records.append,
    )

    # ATT-05 / D9: the block is an unscored event; attempt 2 is retried once and proposes.
    assert len(records) == 3
    assert len(emitted_records) == 3
    assert patcher.propose_calls == 2
    assert records[1]["note"] == "retry_gate_no_new_evidence"
    assert records[1]["record_type"] == "infra_event"
    assert records[1]["patch_meta"]["retry_gate_blocked"] is True
    assert int(records[1]["attempt_metadata"]["retry_history_chars"] or 0) > 0
    assert emitted_records[1]["note"] == "retry_gate_no_new_evidence"
    assert emitted_records[1]["attempt"] == 2
    assert records[2]["attempt"] == 2
    assert records[2]["patch_meta"]["retry_gate_reason"] == "block_cap_reached"
    assert records[2]["patch_meta"]["retry_gate_blocks_before"] == 1


def test_structured_condition_retry_gate_preserves_measured_attempt_wall_clock(
    tmp_path,
    monkeypatch,
) -> None:
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir_retry_wall_clock"
    context_dir = tmp_path / "context_retry_wall_clock"
    shutil.copytree(case.path, workdir)
    patcher = _SameEvidencePatcher()
    measured_attempt_wall_secs = iter([0.2, 1.25, 0.3])  # attempt 1, gate block, attempt 2

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
    monkeypatch.setattr(
        eval_attempt_flow,
        "_measure_attempt_wall_clock_sec",
        lambda _attempt_start: next(measured_attempt_wall_secs),
    )

    records = run_eval.run_one_condition(
        case=case,
        condition=run_eval.Condition.LAYERED_ADAPTIVE_GATED,
        workdir=workdir,
        context_dir_base=context_dir,
        patcher=patcher,
        k=2,
        output_format="json_compact",
        run_id="run-structured-gate-wall-clock",
        include_files_context=False,
        force_patch_attempt=False,
        context_protocol="staged",
        adaptive_gate_policy="aggressive",
        adaptive_max_helper_calls=2,
    )

    assert len(records) == 3  # ATT-05 / D9: attempt 1, unscored gate block, attempt 2
    assert records[1]["note"] == "retry_gate_no_new_evidence"
    assert records[1]["attempt_wall_clock_sec"] == 1.25
    assert records[1]["cumulative_runtime_sec"] == records[0]["cumulative_runtime_sec"] + 1.25
    timing = records[1]["attempt_metadata"]["timing_breakdown"]
    assert timing["test_before_sec"] == pytest.approx(0.01)
    assert timing["overhead_sec"] == pytest.approx(1.24)


@dataclass
class _ChangingEvidencePatcher:
    name: str = "changing_evidence"
    propose_calls: int = 0
    request_calls: int = 0

    def request_context(self, *, prompt: str, context_dir, cwd) -> ContextRequestResult:
        del prompt, context_dir, cwd
        self.request_calls += 1
        path = "buggy.py" if self.request_calls == 1 else "helper.py"
        return ContextRequestResult(
            request_json={
                "need_more_context": True,
                "requests": [{"type": "file", "path": path}],
                "sufficiency_rationale": "need a related file",
            },
            raw_response='{"need_more_context":true,"requests":[{"type":"file","path":"dynamic"}]}',
            meta={},
        )

    def propose_patch(self, *, prompt: str, context_dir, cwd) -> PatchResult:
        del prompt, context_dir, cwd
        self.propose_calls += 1
        return PatchResult(diff=None, meta={"reason": "no_diff"})


def test_structured_condition_retry_gate_allows_retry_when_evidence_changes(
    tmp_path,
    monkeypatch,
) -> None:
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir_retry_allow"
    context_dir = tmp_path / "context_retry_allow"
    shutil.copytree(case.path, workdir)
    patcher = _ChangingEvidencePatcher()

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
        run_id="run-structured-gate-allow",
        include_files_context=False,
        force_patch_attempt=False,
        context_protocol="staged",
        adaptive_gate_policy="aggressive",
        adaptive_max_helper_calls=2,
    )

    assert len(records) == 2
    assert patcher.propose_calls == 2
    assert records[1]["note"] == "no_patch_proposed"
    assert records[1]["patch_meta"]["retry_gate_evaluated"] is True
    assert records[1]["patch_meta"]["retry_gate_blocked"] is False
    assert int(records[1]["attempt_metadata"]["retry_history_chars"] or 0) > 0


@dataclass
class _ChangingEvidenceCapturePatcher:
    name: str = "changing_evidence_capture"
    propose_calls: int = 0
    request_calls: int = 0
    prompts: list[str] = field(default_factory=list)

    def request_context(self, *, prompt: str, context_dir, cwd) -> ContextRequestResult:
        del prompt, context_dir, cwd
        self.request_calls += 1
        path = "buggy.py" if self.request_calls == 1 else "helper.py"
        return ContextRequestResult(
            request_json={
                "need_more_context": True,
                "requests": [{"type": "file", "path": path}],
                "sufficiency_rationale": "need a related file",
            },
            raw_response='{"need_more_context":true,"requests":[{"type":"file","path":"dynamic"}]}',
            meta={},
        )

    def propose_patch(self, *, prompt: str, context_dir, cwd) -> PatchResult:
        del context_dir, cwd
        self.propose_calls += 1
        self.prompts.append(prompt)
        return PatchResult(diff=None, meta={"reason": "no_diff"})


def test_structured_history_mode_applies_retry_history_budget(tmp_path, monkeypatch) -> None:
    case = _make_case(tmp_path)
    workdir = tmp_path / "workdir_retry_history_budget"
    context_dir = tmp_path / "context_retry_history_budget"
    shutil.copytree(case.path, workdir)
    patcher = _ChangingEvidenceCapturePatcher()

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
        run_id="run-structured-history-budget",
        include_files_context=False,
        force_patch_attempt=False,
        context_protocol="staged",
        adaptive_gate_policy="aggressive",
        adaptive_max_helper_calls=2,
        retry_context_mode="history",
        retry_history_max_chars=4_000,
        prompt_budget_retry_history_chars=80,
    )

    assert len(records) == 2
    assert len(patcher.prompts) == 2
    assert "Retry history from previous attempts:" in patcher.prompts[1]
    assert "[truncated]" in patcher.prompts[1]

    second_meta = records[1]["attempt_metadata"]
    assert int(second_meta["retry_history_chars"] or 0) > 0
    assert int(second_meta["retry_history_budget_chars"] or 0) == 80
    assert second_meta["retry_history_budget_applied"] is True


def test_run_pytest_enables_plugin_for_structured_snapshot_condition(tmp_path, monkeypatch) -> None:
    captured: dict[str, object] = {}

    def _fake_run_monitored_subprocess(*, cmd, **kwargs):
        del kwargs
        captured["args"] = cmd
        return SimpleNamespace(
            returncode=1,
            duration_sec=0.01,
            stdout="",
            stderr="",
            timed_out=False,
            timeout_sec=None,
        )

    monkeypatch.setattr(pytest_runner, "run_monitored_subprocess", _fake_run_monitored_subprocess)
    _ = run_eval.run_pytest(
        cwd=tmp_path,
        python_args=["-m", "pytest", "-q"],
        condition=run_eval.Condition.LAYERED_ADAPTIVE_GATED,
        timeout_sec=1,
        output_format="json_compact",
        run_context=RunContext(execution=run_eval.ExecutionConfig(executor="local")),
    )

    args = list(captured.get("args") or [])
    assert "-p" in args
    assert "llmdebug.pytest_plugin" in args
