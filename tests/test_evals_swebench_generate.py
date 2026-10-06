"""Tests for SWE-bench dataset loading, row filtering, and generation logic."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import cast

import pytest

from evals.patchers.base import PatchResult
from evals.patchers.openai_compat_patcher import OpenAICompatPatcher
from evals.swebench.generate import (
    ProposalOutcome,
    _load_dataset_rows,
    _propose_prediction,
    _select_instance_ids,
)
from evals.swebench.prompt_policy import PromptInputs
from evals.swebench.types import PredictionRecord


def _install_fake_datasets(monkeypatch, rows) -> None:
    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_dataset = lambda _dataset_name, split: rows[split]
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)


def test_load_dataset_rows_returns_unique_instance_rows(monkeypatch) -> None:
    _install_fake_datasets(
        monkeypatch,
        {
            "test": [
                {"instance_id": "i-1", "repo": "demo/a"},
                {"instance_id": "i-2", "repo": "demo/b"},
            ]
        },
    )

    rows = _load_dataset_rows(dataset_name="demo", split="test")

    assert sorted(rows) == ["i-1", "i-2"]
    assert rows["i-1"]["repo"] == "demo/a"
    assert rows["i-2"]["repo"] == "demo/b"


def test_load_dataset_rows_rejects_duplicate_instance_ids(monkeypatch) -> None:
    _install_fake_datasets(
        monkeypatch,
        {
            "test": [
                {"instance_id": "dupe-1", "repo": "demo/a"},
                {"instance_id": "dupe-1", "repo": "demo/b"},
            ]
        },
    )

    with pytest.raises(RuntimeError, match="Duplicate instance_id values found"):
        _load_dataset_rows(dataset_name="demo", split="test")


# ---------------------------------------------------------------------------
# _select_instance_ids tests
# ---------------------------------------------------------------------------


def _make_rows_by_id(*ids: str) -> dict[str, dict[str, str]]:
    return {iid: {"instance_id": iid} for iid in ids}


def test_select_instance_ids_returns_sorted_keys_by_default() -> None:
    rows_by_id = _make_rows_by_id("z-1", "a-2", "m-3")

    result = _select_instance_ids(
        rows_by_id=rows_by_id,
        instance_ids_file=None,
        max_instances=0,
    )

    assert result == ["a-2", "m-3", "z-1"]


def test_select_instance_ids_respects_max_instances() -> None:
    rows_by_id = _make_rows_by_id("e-5", "d-4", "c-3", "b-2", "a-1")

    result = _select_instance_ids(
        rows_by_id=rows_by_id,
        instance_ids_file=None,
        max_instances=2,
    )

    assert result == ["a-1", "b-2"]


def test_select_instance_ids_reads_from_file(tmp_path: Path) -> None:
    rows_by_id = _make_rows_by_id("x-1", "y-2", "z-3")
    ids_file = tmp_path / "ids.txt"
    ids_file.write_text("z-3\ny-2\n", encoding="utf-8")

    result = _select_instance_ids(
        rows_by_id=rows_by_id,
        instance_ids_file=ids_file,
        max_instances=0,
    )

    assert result == ["z-3", "y-2"]


def test_select_instance_ids_rejects_unknown_ids_in_file(tmp_path: Path) -> None:
    rows_by_id = _make_rows_by_id("a-1", "b-2")
    ids_file = tmp_path / "ids.txt"
    ids_file.write_text("a-1\nnot-in-dataset\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="instance_ids_file contains ids not present in dataset"):
        _select_instance_ids(
            rows_by_id=rows_by_id,
            instance_ids_file=ids_file,
            max_instances=0,
        )


# ---------------------------------------------------------------------------
# _propose_prediction tests
# ---------------------------------------------------------------------------

_SAMPLE_ROW: dict[str, object] = {
    "instance_id": "inst-1",
    "repo": "demo/repo",
    "base_commit": "abc",
    "problem_statement": "Fix bug.",
    "hints_text": "",
    "FAIL_TO_PASS": None,
    "PASS_TO_PASS": None,
}


class _StubPatcher:
    """Minimal stand-in for :class:`OpenAICompatPatcher`."""

    def __init__(self, diff: str | None = None, error: Exception | None = None) -> None:
        self._diff = diff
        self._error = error

    def propose_patch(self, *, prompt: str, context_dir: Path, cwd: Path) -> PatchResult:
        if self._error is not None:
            raise self._error
        return PatchResult(diff=self._diff, meta=None)


def test_propose_prediction_returns_record_on_success(tmp_path: Path) -> None:
    patcher = cast(OpenAICompatPatcher, _StubPatcher(diff="diff --git a/f b/f"))

    outcome = _propose_prediction(
        instance_id="inst-1",
        row=dict(_SAMPLE_ROW),
        condition="baseline",
        prompt_inputs=PromptInputs(snapshot_dir=None),
        patcher=patcher,
        model_label="test-model",
        workspace_root=tmp_path,
    )

    assert isinstance(outcome, ProposalOutcome)
    assert outcome.is_empty is False
    assert outcome.request_error is None
    assert outcome.record == PredictionRecord(
        instance_id="inst-1",
        model_patch="diff --git a/f b/f",
        model_name_or_path="test-model",
    )


def test_propose_prediction_marks_empty_patch(tmp_path: Path) -> None:
    patcher = cast(OpenAICompatPatcher, _StubPatcher(diff=""))

    outcome = _propose_prediction(
        instance_id="inst-1",
        row=dict(_SAMPLE_ROW),
        condition="baseline",
        prompt_inputs=PromptInputs(snapshot_dir=None),
        patcher=patcher,
        model_label="test-model",
        workspace_root=tmp_path,
    )

    assert outcome.is_empty is True
    assert outcome.record.model_patch == ""
    assert outcome.request_error is None


def test_propose_prediction_captures_patcher_exception(tmp_path: Path) -> None:
    patcher = cast(
        OpenAICompatPatcher,
        _StubPatcher(error=RuntimeError("connection lost")),
    )

    outcome = _propose_prediction(
        instance_id="inst-1",
        row=dict(_SAMPLE_ROW),
        condition="baseline",
        prompt_inputs=PromptInputs(snapshot_dir=None),
        patcher=patcher,
        model_label="test-model",
        workspace_root=tmp_path,
    )

    assert outcome.is_empty is True
    assert outcome.record.model_patch == ""
    assert outcome.request_error is not None
    assert "RuntimeError" in outcome.request_error
    assert "connection lost" in outcome.request_error


# ---------------------------------------------------------------------------
# generate_lite_predictions validation tests (lines 76-85)
# ---------------------------------------------------------------------------


def test_generate_lite_rejects_negative_max_instances(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("evals.swebench.generate.validate_run_id", lambda x: x)
    monkeypatch.setattr(
        "evals.swebench.generate.parse_paired_conditions",
        lambda _c: ["baseline", "llmdebug"],
    )
    with pytest.raises(ValueError, match="max_instances must be >= 0"):
        from evals.swebench.generate import generate_lite_predictions

        generate_lite_predictions(
            run_id="r",
            conditions="baseline,llmdebug",
            max_instances=-1,
            openai_base_url="http://x",
            openai_model="m",
            openai_api_key=None,
            root=tmp_path,
        )


def test_generate_lite_rejects_jobs_less_than_one(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("evals.swebench.generate.validate_run_id", lambda x: x)
    monkeypatch.setattr(
        "evals.swebench.generate.parse_paired_conditions",
        lambda _c: ["baseline", "llmdebug"],
    )
    with pytest.raises(ValueError, match="jobs must be >= 1"):
        from evals.swebench.generate import generate_lite_predictions

        generate_lite_predictions(
            run_id="r",
            conditions="baseline,llmdebug",
            jobs=0,
            openai_base_url="http://x",
            openai_model="m",
            openai_api_key=None,
            root=tmp_path,
        )


def test_generate_lite_rejects_negative_max_request_errors(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("evals.swebench.generate.validate_run_id", lambda x: x)
    monkeypatch.setattr(
        "evals.swebench.generate.parse_paired_conditions",
        lambda _c: ["baseline", "llmdebug"],
    )
    with pytest.raises(ValueError, match="max_request_errors must be >= 0"):
        from evals.swebench.generate import generate_lite_predictions

        generate_lite_predictions(
            run_id="r",
            conditions="baseline,llmdebug",
            max_request_errors=-1,
            openai_base_url="http://x",
            openai_model="m",
            openai_api_key=None,
            root=tmp_path,
        )


def test_generate_lite_rejects_bad_error_rate(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("evals.swebench.generate.validate_run_id", lambda x: x)
    monkeypatch.setattr(
        "evals.swebench.generate.parse_paired_conditions",
        lambda _c: ["baseline", "llmdebug"],
    )
    with pytest.raises(ValueError, match="max_request_error_rate must be between 0 and 1"):
        from evals.swebench.generate import generate_lite_predictions

        generate_lite_predictions(
            run_id="r",
            conditions="baseline,llmdebug",
            max_request_error_rate=1.5,
            openai_base_url="http://x",
            openai_model="m",
            openai_api_key=None,
            root=tmp_path,
        )


def test_generate_lite_rejects_min_requests_below_one(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setattr("evals.swebench.generate.validate_run_id", lambda x: x)
    monkeypatch.setattr(
        "evals.swebench.generate.parse_paired_conditions",
        lambda _c: ["baseline", "llmdebug"],
    )
    with pytest.raises(
        ValueError,
        match="min_requests_for_error_rate_check must be >= 1",
    ):
        from evals.swebench.generate import generate_lite_predictions

        generate_lite_predictions(
            run_id="r",
            conditions="baseline,llmdebug",
            min_requests_for_error_rate_check=0,
            openai_base_url="http://x",
            openai_model="m",
            openai_api_key=None,
            root=tmp_path,
        )


# ---------------------------------------------------------------------------
# _load_dataset_rows edge cases (lines 182-183, 194, 211)
# ---------------------------------------------------------------------------


def test_load_dataset_rows_raises_on_missing_datasets_package(
    monkeypatch,
) -> None:
    """ImportError from ``datasets`` → RuntimeError."""
    monkeypatch.setitem(sys.modules, "datasets", None)
    with pytest.raises(RuntimeError, match="datasets is required"):
        _load_dataset_rows(dataset_name="demo", split="test")


def test_load_dataset_rows_skips_empty_instance_id(monkeypatch) -> None:
    """Rows with blank instance_id are silently skipped."""
    _install_fake_datasets(
        monkeypatch,
        {
            "test": [
                {"instance_id": "", "repo": "demo/a"},
                {"instance_id": "  ", "repo": "demo/b"},
                {"instance_id": "ok-1", "repo": "demo/c"},
            ]
        },
    )
    rows = _load_dataset_rows(dataset_name="demo", split="test")
    assert list(rows.keys()) == ["ok-1"]


def test_load_dataset_rows_raises_when_all_rows_empty(monkeypatch) -> None:
    """All rows have empty instance_id → RuntimeError (line 211)."""
    _install_fake_datasets(
        monkeypatch,
        {"test": [{"instance_id": "", "repo": "demo/a"}]},
    )
    with pytest.raises(RuntimeError, match="No rows with instance_id"):
        _load_dataset_rows(dataset_name="demo", split="test")


# ---------------------------------------------------------------------------
# _select_instance_ids — blank/duplicate lines in file (line 229)
# ---------------------------------------------------------------------------


def test_select_instance_ids_skips_blank_and_duplicate_lines(
    tmp_path: Path,
) -> None:
    rows_by_id = _make_rows_by_id("a-1", "b-2", "c-3")
    ids_file = tmp_path / "ids.txt"
    ids_file.write_text("\na-1\n  \nb-2\na-1\n\n", encoding="utf-8")
    result = _select_instance_ids(
        rows_by_id=rows_by_id,
        instance_ids_file=ids_file,
        max_instances=0,
    )
    assert result == ["a-1", "b-2"]


# ---------------------------------------------------------------------------
# _build_prompt — pass_to_pass section (line 632)
# ---------------------------------------------------------------------------


def test_build_prompt_includes_pass_to_pass_section() -> None:
    from evals.swebench.generate import _build_prompt

    row: dict[str, object] = {
        "instance_id": "inst-1",
        "repo": "demo/repo",
        "base_commit": "abc",
        "problem_statement": "Fix bug.",
        "hints_text": "",
        "FAIL_TO_PASS": None,
        "PASS_TO_PASS": "test_something",
    }
    prompt = _build_prompt(
        row=row, condition="baseline", prompt_inputs=PromptInputs(include_test_ids=True)
    )
    assert "Pass-to-pass tests:" in prompt
    assert "test_something" in prompt


# ---------------------------------------------------------------------------
# _resolve_git_commit edge cases (lines 682-695)
# ---------------------------------------------------------------------------


def test_resolve_git_commit_returns_unknown_on_oserror(monkeypatch) -> None:
    from evals.swebench.generate import _resolve_git_commit

    def _raise_oserror(*_args, **_kwargs):
        raise OSError("no git")

    monkeypatch.setattr("evals.swebench.generate.subprocess.run", _raise_oserror)
    assert _resolve_git_commit() == "unknown"


def test_resolve_git_commit_returns_unknown_on_nonzero_exit(
    monkeypatch,
) -> None:
    from evals.swebench.generate import _resolve_git_commit

    class _FakeProc:
        returncode = 128
        stdout = ""
        stderr = "fatal: not a repo"

    monkeypatch.setattr(
        "evals.swebench.generate.subprocess.run",
        lambda *a, **kw: _FakeProc(),
    )
    assert _resolve_git_commit() == "unknown"


def test_resolve_git_commit_returns_unknown_on_empty_stdout(
    monkeypatch,
) -> None:
    from evals.swebench.generate import _resolve_git_commit

    class _FakeProc:
        returncode = 0
        stdout = "  \n"
        stderr = ""

    monkeypatch.setattr(
        "evals.swebench.generate.subprocess.run",
        lambda *a, **kw: _FakeProc(),
    )
    assert _resolve_git_commit() == "unknown"


# ---------------------------------------------------------------------------
# _propose_prediction_async edge cases (lines 480-486)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_propose_prediction_async_captures_exception(
    tmp_path: Path,
) -> None:
    """Async patcher exception → empty patch + request_error."""
    from evals.swebench.generate import _propose_prediction_async

    class _ErrorPatcher:
        async def apropose_patch(self, *, prompt, context_dir, cwd):
            raise RuntimeError("async boom")

    patcher = cast(OpenAICompatPatcher, _ErrorPatcher())
    outcome = await _propose_prediction_async(
        instance_id="inst-1",
        row=dict(_SAMPLE_ROW),
        condition="baseline",
        prompt_inputs=PromptInputs(snapshot_dir=None),
        patcher=patcher,
        model_label="test-model",
        workspace_root=tmp_path,
    )
    assert outcome.is_empty is True
    assert outcome.record.model_patch == ""
    assert outcome.request_error is not None
    assert "RuntimeError" in outcome.request_error


# ---------------------------------------------------------------------------
# _generate_for_condition — single-job counts (lines 314, 316)
# ---------------------------------------------------------------------------


def test_generate_for_condition_single_job_counts(monkeypatch, tmp_path: Path) -> None:
    """Single-job loop tracks is_empty and request_error counts."""
    from evals.swebench.generate import _generate_for_condition

    call_index = 0

    class _AlternatingPatcher:
        def propose_patch(self, *, prompt, context_dir, cwd):
            nonlocal call_index
            call_index += 1
            if call_index % 2 == 0:
                raise RuntimeError("flaky")
            return PatchResult(diff="diff --git a/f b/f", meta=None)

    patcher = cast(OpenAICompatPatcher, _AlternatingPatcher())
    rows_by_id = {
        "a": dict(_SAMPLE_ROW, instance_id="a"),
        "b": dict(_SAMPLE_ROW, instance_id="b"),
    }
    run_dir = tmp_path / "run"
    (run_dir / "baseline").mkdir(parents=True, exist_ok=True)

    summary = _generate_for_condition(
        run_dir=run_dir,
        condition="baseline",
        selected_ids=["a", "b"],
        rows_by_id=rows_by_id,
        patcher=patcher,
        model_label="test",
        resume=False,
        prompt_inputs=PromptInputs(snapshot_dir=None),
        jobs=1,
        openai_async=False,
        max_request_errors=25,
        max_request_error_rate=0.5,
        min_requests_for_error_rate_check=10,
    )
    assert summary.generated == 2
    assert summary.empty_patch_count == 1
    assert summary.request_error_count == 1


# ---------------------------------------------------------------------------
# _generate_for_condition_async — counts (lines 540, 542)
# and exception cancellation (lines 565-569)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_generate_for_condition_async_counts(
    tmp_path: Path,
) -> None:
    """Async loop tracks is_empty and request_error counts."""
    from evals.swebench.generate import _generate_for_condition_async

    call_index = 0

    class _AsyncAlternatingPatcher:
        async def apropose_patch(self, *, prompt, context_dir, cwd):
            nonlocal call_index
            call_index += 1
            if call_index % 2 == 0:
                raise RuntimeError("async flaky")
            return PatchResult(diff="diff --git a/f b/f", meta=None)

    patcher = cast(OpenAICompatPatcher, _AsyncAlternatingPatcher())
    rows_by_id = {
        "a": dict(_SAMPLE_ROW, instance_id="a"),
        "b": dict(_SAMPLE_ROW, instance_id="b"),
    }
    run_dir = tmp_path / "run"
    (run_dir / "baseline").mkdir(parents=True, exist_ok=True)
    pred_path = run_dir / "baseline" / "predictions.jsonl"
    existing_map: dict[str, PredictionRecord] = {}
    ws = tmp_path / "ws"
    ws.mkdir()

    gen, empty, errors = await _generate_for_condition_async(
        condition="baseline",
        pending_ids=["a", "b"],
        selected_ids=["a", "b"],
        rows_by_id=rows_by_id,
        existing_map=existing_map,
        prompt_inputs=PromptInputs(snapshot_dir=None),
        patcher=patcher,
        model_label="test",
        workspace_root=ws,
        generated=0,
        resumed=0,
        empty_patch_count=0,
        request_error_count=0,
        jobs=1,
        max_request_errors=25,
        max_request_error_rate=0.5,
        min_requests_for_error_rate_check=10,
        prediction_path_value=pred_path,
    )
    assert gen == 2
    assert empty == 1
    assert errors == 1


@pytest.mark.asyncio
async def test_generate_for_condition_async_cancels_on_error(
    tmp_path: Path,
) -> None:
    """Exception in async loop cancels remaining tasks (lines 565-569)."""
    from evals.swebench.generate import _generate_for_condition_async

    class _BoomPatcher:
        async def apropose_patch(self, *, prompt, context_dir, cwd):
            raise RuntimeError("total failure")

    patcher = cast(OpenAICompatPatcher, _BoomPatcher())
    rows_by_id = {f"i-{i}": dict(_SAMPLE_ROW, instance_id=f"i-{i}") for i in range(5)}
    ids = list(rows_by_id)
    run_dir = tmp_path / "run"
    (run_dir / "baseline").mkdir(parents=True, exist_ok=True)
    pred_path = run_dir / "baseline" / "predictions.jsonl"
    ws = tmp_path / "ws"
    ws.mkdir()

    with pytest.raises(RuntimeError, match=r"total failure|all patch requests|error"):
        await _generate_for_condition_async(
            condition="baseline",
            pending_ids=ids,
            selected_ids=ids,
            rows_by_id=rows_by_id,
            existing_map={},
            prompt_inputs=PromptInputs(snapshot_dir=None),
            patcher=patcher,
            model_label="test",
            workspace_root=ws,
            generated=0,
            resumed=0,
            empty_patch_count=0,
            request_error_count=0,
            jobs=2,
            max_request_errors=0,
            max_request_error_rate=0.0,
            min_requests_for_error_rate_check=1,
            prediction_path_value=pred_path,
        )
