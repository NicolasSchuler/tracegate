"""Tests for SWE-bench Pro dataset loading, row filtering, and generation logic."""

from __future__ import annotations

import sys
import types
from pathlib import Path
from typing import Any, cast

import pytest

from evals.patchers.base import PatchResult
from evals.patchers.openai_compat_patcher import OpenAICompatPatcher
from evals.swebench.types import PredictionRecord
from evals.swebench_pro.generate import (
    ProposalOutcome,
    _load_dataset_rows,
    _propose_prediction,
    _select_instance_ids,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _install_fake_datasets(monkeypatch, rows) -> None:
    fake_datasets = types.ModuleType("datasets")
    fake_datasets.load_dataset = lambda _dataset_name, split: rows[split]
    monkeypatch.setitem(sys.modules, "datasets", fake_datasets)


def _make_rows_by_id(*ids: str) -> dict[str, dict[str, str]]:
    return {iid: {"instance_id": iid} for iid in ids}


_SAMPLE_ROW: dict[str, object] = {
    "instance_id": "pro-inst-1",
    "repo": "demo/pro-repo",
    "base_commit": "def456",
    "problem_statement": "Fix Pro bug.",
    "repo_language": "Python",
    "requirements": "numpy>=1.0",
    "interface": "def solve(x): ...",
    "fail_to_pass": None,
    "pass_to_pass": None,
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


# ---------------------------------------------------------------------------
# _load_dataset_rows tests
# ---------------------------------------------------------------------------


def test_load_dataset_rows_returns_unique_instance_rows(monkeypatch) -> None:
    _install_fake_datasets(
        monkeypatch,
        {
            "test": [
                {"instance_id": "pro-1", "repo": "demo/a"},
                {"instance_id": "pro-2", "repo": "demo/b"},
            ]
        },
    )

    rows = _load_dataset_rows(dataset_name="demo-pro", split="test")

    assert sorted(rows) == ["pro-1", "pro-2"]
    assert rows["pro-1"]["repo"] == "demo/a"
    assert rows["pro-2"]["repo"] == "demo/b"


def test_load_dataset_rows_rejects_duplicate_instance_ids(monkeypatch) -> None:
    _install_fake_datasets(
        monkeypatch,
        {
            "test": [
                {"instance_id": "dupe-pro", "repo": "demo/a"},
                {"instance_id": "dupe-pro", "repo": "demo/b"},
            ]
        },
    )

    with pytest.raises(RuntimeError, match="Duplicate instance_id values found"):
        _load_dataset_rows(dataset_name="demo-pro", split="test")


# ---------------------------------------------------------------------------
# _select_instance_ids tests
# ---------------------------------------------------------------------------


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


def test_propose_prediction_returns_record_on_success(tmp_path: Path) -> None:
    patcher = cast(OpenAICompatPatcher, _StubPatcher(diff="diff --git a/f b/f"))

    outcome = _propose_prediction(
        instance_id="pro-inst-1",
        row=dict(_SAMPLE_ROW),
        condition="baseline",
        snapshot_dir=None,
        patcher=patcher,
        model_label="test-pro-model",
        workspace_root=tmp_path,
    )

    assert isinstance(outcome, ProposalOutcome)
    assert outcome.is_empty is False
    assert outcome.request_error is None
    assert outcome.record == PredictionRecord(
        instance_id="pro-inst-1",
        model_patch="diff --git a/f b/f",
        model_name_or_path="test-pro-model",
    )


def test_propose_prediction_marks_empty_patch(tmp_path: Path) -> None:
    patcher = cast(OpenAICompatPatcher, _StubPatcher(diff=""))

    outcome = _propose_prediction(
        instance_id="pro-inst-1",
        row=dict(_SAMPLE_ROW),
        condition="baseline",
        snapshot_dir=None,
        patcher=patcher,
        model_label="test-pro-model",
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
        instance_id="pro-inst-1",
        row=dict(_SAMPLE_ROW),
        condition="baseline",
        snapshot_dir=None,
        patcher=patcher,
        model_label="test-pro-model",
        workspace_root=tmp_path,
    )

    assert outcome.is_empty is True
    assert outcome.record.model_patch == ""
    assert outcome.request_error is not None
    assert "RuntimeError" in outcome.request_error
    assert "connection lost" in outcome.request_error


def test_propose_prediction_uses_build_prompt_pro(tmp_path: Path) -> None:
    """Verify _propose_prediction invokes _build_prompt_pro (not _build_prompt)."""
    captured_prompts: list[str] = []

    class _CapturePatcher:
        def propose_patch(self, *, prompt: str, context_dir: Path, cwd: Path) -> PatchResult:
            captured_prompts.append(prompt)
            return PatchResult(diff="diff --git a/f b/f", meta=None)

    patcher = cast(OpenAICompatPatcher, _CapturePatcher())

    _propose_prediction(
        instance_id="pro-inst-1",
        row=dict(_SAMPLE_ROW),
        condition="baseline",
        snapshot_dir=None,
        patcher=patcher,
        model_label="test-pro-model",
        workspace_root=tmp_path,
    )

    assert len(captured_prompts) == 1
    prompt = captured_prompts[0]
    # Pro-specific fields should appear in the prompt
    assert "demo/pro-repo" in prompt
    assert "pro-inst-1" in prompt
