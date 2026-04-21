"""Tests for shared time-budget accounting across eval retry attempts."""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from evals import run_eval
from evals.patchers.base import PatchResult


@dataclass(frozen=True)
class _FakeOpenAIPatcher:
    name: str = "openai"
    max_propose_sec: float = 180.0
    response_mode: str = "edits"
    model: str = "fake-model"
    base_url: str = "http://127.0.0.1:12000/v1"

    def propose_patch(self, *, prompt: str, context_dir: Path, cwd: Path) -> PatchResult:
        del prompt
        del context_dir
        del cwd
        return PatchResult(
            diff=None,
            meta={
                "llm_call_count": 0,
                "tokens_prompt": 0,
                "tokens_completion": 0,
                "tokens_total": 0,
                "probe_max_propose_sec": self.max_propose_sec,
            },
        )


@dataclass(frozen=True)
class _FailIfCalledOpenAIPatcher(_FakeOpenAIPatcher):
    def propose_patch(self, *, prompt: str, context_dir: Path, cwd: Path) -> PatchResult:
        del prompt
        del context_dir
        del cwd
        raise AssertionError(
            "propose_patch should not be called when shared attempt budget is exhausted"
        )


def test_propose_patch_with_snapshot_context_clamps_openai_budget_to_attempt_deadline(
    tmp_path,
) -> None:
    patcher = _FakeOpenAIPatcher(max_propose_sec=180.0)
    deadline = time.monotonic() + 0.25

    result = run_eval._propose_patch_with_snapshot_context(
        patcher=patcher,
        prompt="irrelevant",
        context_dir=tmp_path,
        cwd=tmp_path,
        snapshot_full_text=None,
        snapshot_artifact_text=None,
        attempt_deadline_monotonic=deadline,
    )

    assert result.diff is None
    assert isinstance(result.meta, dict)
    assert result.meta.get("attempt_shared_propose_deadline_enabled") is True
    assert result.meta.get("attempt_shared_propose_budget_clamped") is True
    assert isinstance(result.meta.get("attempt_shared_propose_call_budget_sec"), float)
    assert 0.0 < float(result.meta["attempt_shared_propose_call_budget_sec"]) <= 0.3
    assert float(result.meta.get("probe_max_propose_sec", 0.0)) <= 0.3
    assert float(result.meta.get("probe_max_propose_sec", 0.0)) < 180.0


def test_propose_patch_with_snapshot_context_short_circuits_when_attempt_budget_exhausted(
    tmp_path,
) -> None:
    patcher = _FailIfCalledOpenAIPatcher(max_propose_sec=180.0)

    result = run_eval._propose_patch_with_snapshot_context(
        patcher=patcher,
        prompt="irrelevant",
        context_dir=tmp_path,
        cwd=tmp_path,
        snapshot_full_text=None,
        snapshot_artifact_text=None,
        attempt_deadline_monotonic=time.monotonic() - 1.0,
    )

    assert result.diff is None
    assert isinstance(result.meta, dict)
    assert result.meta.get("error") == "request_budget_exceeded"
    assert result.meta.get("llm_call_count") == 0
    assert result.meta.get("tokens_total") == 0
    assert result.meta.get("attempt_shared_propose_budget_exhausted") is True
