"""Integration tests for the ThreadPoolExecutor path (jobs > 1) in _generate_for_condition()."""

from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, cast

import pytest

from evals.patchers.base import PatchResult
from evals.patchers.openai_compat_patcher import OpenAICompatPatcher
from evals.swebench.prompt_policy import PromptInputs
from evals.swebench.types import CONDITION_BASELINE
from evals.swebench_pro.generate import _generate_for_condition
from evals.swebench_pro.io import ensure_run_layout, read_predictions


def _make_rows(instance_ids: list[str]) -> dict[str, dict[str, Any]]:
    """Build a minimal rows_by_id mapping for the given instance ids."""
    return {
        iid: {
            "instance_id": iid,
            "repo": "demo/pro-repo",
            "base_commit": "def456",
            "problem_statement": f"Fix Pro issue in {iid}.",
        }
        for iid in instance_ids
    }


# ------------------------------------------------------------------
# Test 1: All predictions produced successfully via threaded path
# ------------------------------------------------------------------


def test_generate_for_condition_threaded_produces_all_predictions(tmp_path: Path) -> None:
    """The threaded path (jobs > 1) must produce one prediction per instance."""
    instance_ids = [f"pro-inst-{i}" for i in range(1, 7)]
    rows_by_id = _make_rows(instance_ids)

    run_dir = tmp_path / "evals" / "swebench_pro_runs" / "threaded-run"
    ensure_run_layout(run_dir, conditions=[CONDITION_BASELINE])

    class _StubPatcher:
        def propose_patch(self, *, prompt: str, context_dir: Path, cwd: Path) -> PatchResult:
            # Extract instance_id from the prompt for a unique diff per instance.
            for iid in instance_ids:
                if iid in prompt:
                    return PatchResult(
                        diff=f"diff --git a/f.py b/f.py\n+{iid}\n",
                        meta=None,
                    )
            return PatchResult(diff="diff --git a/f.py b/f.py\n+unknown\n", meta=None)

    summary = _generate_for_condition(
        run_dir=run_dir,
        condition=CONDITION_BASELINE,
        selected_ids=instance_ids,
        rows_by_id=rows_by_id,
        patcher=cast(OpenAICompatPatcher, _StubPatcher()),
        model_label="stub-pro-model::baseline",
        resume=False,
        prompt_inputs=PromptInputs(snapshot_dir=None),
        jobs=4,
        openai_async=False,
        max_request_errors=25,
        max_request_error_rate=0.5,
        min_requests_for_error_rate_check=10,
    )

    assert summary.generated == 6
    assert summary.resumed == 0
    assert summary.total_instances == 6
    assert summary.empty_patch_count == 0
    assert summary.request_error_count == 0

    records = read_predictions(summary.predictions_path)
    assert len(records) == 6
    assert {r.instance_id for r in records} == set(instance_ids)
    for record in records:
        assert record.model_patch, f"empty model_patch for {record.instance_id}"


# ------------------------------------------------------------------
# Test 2: Mixed successes and errors via threaded path
# ------------------------------------------------------------------


def test_generate_for_condition_threaded_handles_mixed_errors(tmp_path: Path) -> None:
    """One failing instance must not prevent the other instances from succeeding."""
    instance_ids = [f"pro-inst-{i}" for i in range(1, 7)]
    rows_by_id = _make_rows(instance_ids)

    run_dir = tmp_path / "evals" / "swebench_pro_runs" / "threaded-mixed"
    ensure_run_layout(run_dir, conditions=[CONDITION_BASELINE])

    class _MixedPatcher:
        def propose_patch(self, *, prompt: str, context_dir: Path, cwd: Path) -> PatchResult:
            if "pro-inst-4" in prompt:
                raise RuntimeError("LLM timeout")
            return PatchResult(diff="diff --git a/f.py b/f.py\n+fix\n", meta=None)

    summary = _generate_for_condition(
        run_dir=run_dir,
        condition=CONDITION_BASELINE,
        selected_ids=instance_ids,
        rows_by_id=rows_by_id,
        patcher=cast(OpenAICompatPatcher, _MixedPatcher()),
        model_label="stub-pro-model::baseline",
        resume=False,
        prompt_inputs=PromptInputs(snapshot_dir=None),
        jobs=3,
        openai_async=False,
        max_request_errors=25,
        max_request_error_rate=0.5,
        min_requests_for_error_rate_check=10,
    )

    assert summary.generated == 6
    assert summary.request_error_count == 1

    # Errors still produce empty-patch records in the predictions file.
    records = read_predictions(summary.predictions_path)
    assert len(records) == 6
    assert {r.instance_id for r in records} == set(instance_ids)

    # pro-inst-4 should have an empty patch due to the error.
    by_id = {r.instance_id: r for r in records}
    assert by_id["pro-inst-4"].model_patch == ""
    # The other instances should have non-empty patches.
    for iid in instance_ids:
        if iid != "pro-inst-4":
            assert by_id[iid].model_patch, f"expected non-empty patch for {iid}"


def test_generate_for_condition_threaded_stops_submitting_after_health_abort(
    tmp_path: Path,
) -> None:
    instance_ids = [f"pro-inst-{i}" for i in range(1, 6)]
    rows_by_id = _make_rows(instance_ids)

    run_dir = tmp_path / "evals" / "swebench_pro_runs" / "threaded-abort"
    ensure_run_layout(run_dir, conditions=[CONDITION_BASELINE])

    release_event = threading.Event()
    timer = threading.Timer(0.1, release_event.set)
    started_ids: list[str] = []
    started_lock = threading.Lock()
    timer.start()

    class _AbortPatcher:
        def propose_patch(self, *, prompt: str, context_dir: Path, cwd: Path) -> PatchResult:
            instance_id = next(iid for iid in instance_ids if iid in prompt)
            with started_lock:
                started_ids.append(instance_id)
                attempt_number = len(started_ids)
            if attempt_number == 1:
                raise RuntimeError("LLM timeout")
            release_event.wait(timeout=1.0)
            return PatchResult(diff="diff --git a/f.py b/f.py\n+fix\n", meta=None)

    try:
        with pytest.raises(RuntimeError, match="exceeded threshold"):
            _generate_for_condition(
                run_dir=run_dir,
                condition=CONDITION_BASELINE,
                selected_ids=instance_ids,
                rows_by_id=rows_by_id,
                patcher=cast(OpenAICompatPatcher, _AbortPatcher()),
                model_label="stub-pro-model::baseline",
                resume=False,
                prompt_inputs=PromptInputs(snapshot_dir=None),
                jobs=2,
                openai_async=False,
                max_request_errors=0,
                max_request_error_rate=1.0,
                min_requests_for_error_rate_check=10,
            )
    finally:
        release_event.set()
        timer.cancel()

    assert len(started_ids) == 2
