"""Tests for async SWE-bench Pro prediction generation and proposal outcomes."""

from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest

from evals.patchers.base import PatchResult
from evals.patchers.openai_compat_patcher import OpenAICompatPatcher
from evals.swebench.types import PredictionRecord
from evals.swebench_pro.generate import (
    ProposalOutcome,
    _propose_prediction_async,
    generate_pro_predictions,
)


@pytest.mark.asyncio
async def test_propose_prediction_async_uses_async_patcher(tmp_path: Path) -> None:
    """The async proposal path should invoke ``apropose_patch`` on the patcher."""

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
        instance_id="pro-instance-1",
        row={
            "instance_id": "pro-instance-1",
            "repo": "demo/pro-repo",
            "base_commit": "def456",
            "problem_statement": "Fix Pro bug.",
        },
        condition="baseline",
        snapshot_dir=None,
        patcher=cast(OpenAICompatPatcher, _StubAsyncPatcher()),
        model_label="model::baseline",
        workspace_root=tmp_path,
    )

    assert outcome.request_error is None
    assert outcome.is_empty is False
    assert outcome.record.instance_id == "pro-instance-1"
    assert "diff --git a/a.py b/a.py" in outcome.record.model_patch


@pytest.mark.asyncio
async def test_propose_prediction_async_captures_error(tmp_path: Path) -> None:
    """Errors in the async patcher should be captured, not raised."""

    class _ErrorAsyncPatcher:
        async def apropose_patch(self, *, prompt: str, context_dir: Path, cwd: Path) -> PatchResult:
            raise RuntimeError("LLM service unavailable")

    outcome = await _propose_prediction_async(
        instance_id="pro-instance-err",
        row={
            "instance_id": "pro-instance-err",
            "repo": "demo/pro-repo",
            "base_commit": "abc123",
            "problem_statement": "Some Pro problem.",
        },
        condition="baseline",
        snapshot_dir=None,
        patcher=cast(OpenAICompatPatcher, _ErrorAsyncPatcher()),
        model_label="model::baseline",
        workspace_root=tmp_path,
    )

    assert outcome.is_empty is True
    assert outcome.record.model_patch == ""
    assert outcome.request_error is not None
    assert "RuntimeError" in outcome.request_error
    assert "LLM service unavailable" in outcome.request_error


def test_generate_pro_predictions_prefers_async_path_when_enabled(
    monkeypatch, tmp_path: Path
) -> None:
    """When ``openai_async=True``, the top-level entry-point must use the async proposal path."""
    rows_by_id = {
        "pro-instance-1": {
            "instance_id": "pro-instance-1",
            "repo": "demo/pro-repo",
            "base_commit": "def456",
            "problem_statement": "Fix Pro bug.",
        }
    }
    async_calls = {"count": 0}

    monkeypatch.setattr(
        "evals.swebench_pro.generate._load_dataset_rows",
        lambda **kwargs: rows_by_id,
    )
    monkeypatch.setattr("evals.swebench_pro.generate._resolve_git_commit", lambda: "deadbeef")

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

    monkeypatch.setattr("evals.swebench_pro.generate._propose_prediction_async", _fake_async)
    monkeypatch.setattr("evals.swebench_pro.generate._propose_prediction", _unexpected_sync)

    summary = generate_pro_predictions(
        run_id="async-pro-demo-run",
        conditions="baseline,llmdebug",
        max_instances=1,
        openai_base_url="http://127.0.0.1:1234/v1",
        openai_model="dummy-model",
        openai_api_key=None,
        openai_async=True,
        jobs=1,
        root=tmp_path,
    )

    assert summary.instance_count == 1
    # Two conditions (baseline + llmdebug) x 1 instance = 2 async calls
    assert async_calls["count"] == 2
