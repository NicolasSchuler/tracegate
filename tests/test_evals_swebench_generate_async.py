"""Tests for async SWE-bench prediction generation and proposal outcomes."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from evals.patchers.base import PatchResult
from evals.patchers.openai_compat_patcher import OpenAICompatPatcher
from evals.swebench.generate import (
    ProposalOutcome,
    _propose_prediction_async,
    generate_lite_predictions,
)
from evals.swebench.prompt_policy import PromptInputs
from evals.swebench.types import PredictionRecord


@pytest.mark.asyncio
async def test_propose_prediction_async_uses_async_patcher(tmp_path: Path) -> None:
    class _StubAsyncPatcher:
        async def apropose_patch(self, *, prompt: str, context_dir: Path, cwd: Path) -> PatchResult:
            del prompt, context_dir, cwd
            return PatchResult(
                diff=(
                    "diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n@@ -1 +1 @@\n-old\n+new\n"
                ),
                meta=None,
            )

    outcome = await _propose_prediction_async(
        instance_id="instance-1",
        row={
            "instance_id": "instance-1",
            "repo": "demo/repo",
            "base_commit": "abc123",
            "problem_statement": "Fix failing tests.",
        },
        condition="baseline",
        prompt_inputs=PromptInputs(snapshot_dir=None),
        patcher=cast(OpenAICompatPatcher, _StubAsyncPatcher()),
        model_label="model::baseline",
        workspace_root=tmp_path,
    )

    assert outcome.request_error is None
    assert outcome.is_empty is False
    assert outcome.record.instance_id == "instance-1"
    assert "diff --git a/a.py b/a.py" in outcome.record.model_patch


def test_generate_lite_predictions_prefers_async_path_when_enabled(
    monkeypatch, tmp_path: Path
) -> None:
    rows_by_id = {
        "instance-1": {
            "instance_id": "instance-1",
            "repo": "demo/repo",
            "base_commit": "abc123",
            "problem_statement": "Fix failing tests.",
        }
    }
    async_calls = {"count": 0}

    monkeypatch.setattr(
        "evals.swebench.generate._load_dataset_rows",
        lambda **kwargs: rows_by_id,
    )
    monkeypatch.setattr("evals.swebench.generate._resolve_git_commit", lambda: "deadbeef")

    async def _fake_async(**kwargs) -> ProposalOutcome:
        async_calls["count"] += 1
        instance_id = str(kwargs["instance_id"])
        model_label = str(kwargs["model_label"])
        return ProposalOutcome(
            record=PredictionRecord(
                instance_id=instance_id,
                model_patch="diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n",
                model_name_or_path=model_label,
            ),
            is_empty=False,
            request_error=None,
        )

    def _unexpected_sync(**kwargs):
        raise AssertionError(f"sync proposal path should be bypassed in async mode: {kwargs}")

    monkeypatch.setattr("evals.swebench.generate._propose_prediction_async", _fake_async)
    monkeypatch.setattr("evals.swebench.generate._propose_prediction", _unexpected_sync)

    summary = generate_lite_predictions(
        run_id="async-demo-run",
        conditions="baseline,llmdebug",
        max_instances=1,
        openai_base_url="http://127.0.0.1:1234/v1",
        openai_model="dummy-model",
        openai_api_key=None,
        openai_async=True,
        jobs=1,
        root=tmp_path,
        prompt_inputs=PromptInputs(snapshot_dir=tmp_path),
    )

    assert summary.instance_count == 1
    assert async_calls["count"] == 2


def test_generate_lite_records_prompt_policy_and_snapshot_coverage(  # DAT-01..03
    monkeypatch, tmp_path: Path
) -> None:
    import json

    rows_by_id = {
        iid: {
            "instance_id": iid,
            "repo": "demo/repo",
            "base_commit": "abc",
            "problem_statement": "x",
        }
        for iid in ("instance-1", "instance-2")
    }
    monkeypatch.setattr("evals.swebench.generate._load_dataset_rows", lambda **kwargs: rows_by_id)
    monkeypatch.setattr("evals.swebench.generate._resolve_git_commit", lambda: "deadbeef")

    async def _fake_async(**kwargs) -> ProposalOutcome:
        return ProposalOutcome(
            record=PredictionRecord(
                instance_id=str(kwargs["instance_id"]),
                model_patch="diff --git a/a.py b/a.py\n--- a/a.py\n+++ b/a.py\n",
                model_name_or_path=str(kwargs["model_label"]),
            ),
            is_empty=False,
        )

    monkeypatch.setattr("evals.swebench.generate._propose_prediction_async", _fake_async)
    snapshots = tmp_path / "snapshots"
    snapshots.mkdir()
    (snapshots / "instance-1.txt").write_text("frame locals", encoding="utf-8")

    summary = generate_lite_predictions(
        run_id="policy-run",
        conditions="baseline,llmdebug",
        openai_base_url="http://127.0.0.1:1234/v1",
        openai_model="dummy-model",
        openai_api_key=None,
        openai_async=True,
        root=tmp_path,
        prompt_inputs=PromptInputs(snapshot_dir=snapshots),
    )

    metadata = json.loads((summary.run_dir / "run_metadata.json").read_text(encoding="utf-8"))
    assert metadata["prompt_policy"] == {
        "include_hints": False,
        "include_test_ids": False,
        "snapshot_dir": str(snapshots),
    }
    assert metadata["snapshot_coverage"]["snapshot_hits"] == 1
    assert metadata["snapshot_coverage"]["snapshot_misses"] == 1
    assert metadata["snapshot_coverage"]["snapshot_missing_instance_ids"] == ["instance-2"]
