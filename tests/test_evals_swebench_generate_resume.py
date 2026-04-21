"""Integration tests for the resume logic in _generate_for_condition()."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

from evals.patchers.base import PatchResult
from evals.patchers.openai_compat_patcher import OpenAICompatPatcher
from evals.swebench.generate import _generate_for_condition
from evals.swebench.io import (
    ensure_run_layout,
    prediction_path,
    read_predictions,
    write_predictions,
)
from evals.swebench.types import CONDITION_BASELINE, PredictionRecord


def _make_rows(instance_ids: list[str]) -> dict[str, dict[str, Any]]:
    """Build a minimal rows_by_id mapping for the given instance ids."""
    return {
        iid: {
            "instance_id": iid,
            "repo": "demo/repo",
            "base_commit": "abc123",
            "problem_statement": f"Fix issue in {iid}.",
        }
        for iid in instance_ids
    }


class _StubPatcher:
    """Patcher stub that returns a unique diff per instance."""

    def propose_patch(self, *, prompt: str, context_dir: Path, cwd: Path) -> PatchResult:
        return PatchResult(
            diff="diff --git a/f.py b/f.py\n+new-generation\n",
            meta=None,
        )


_HEALTH_KWARGS: dict[str, Any] = {
    "max_request_errors": 25,
    "max_request_error_rate": 0.5,
    "min_requests_for_error_rate_check": 10,
}


# ------------------------------------------------------------------
# Test 1: Resume skips existing predictions
# ------------------------------------------------------------------


def test_generate_for_condition_resumes_from_existing_predictions(tmp_path: Path) -> None:
    instance_ids = [f"inst-{i}" for i in range(1, 6)]
    rows_by_id = _make_rows(instance_ids)

    run_dir = tmp_path / "evals" / "swebench_runs" / "resume-run"
    ensure_run_layout(run_dir, conditions=[CONDITION_BASELINE])

    # Pre-populate predictions for inst-1 and inst-2.
    pre_existing = [
        PredictionRecord(
            instance_id="inst-1",
            model_patch="diff --git a/f.py b/f.py\n+pre-existing-1\n",
            model_name_or_path="original-model",
        ),
        PredictionRecord(
            instance_id="inst-2",
            model_patch="diff --git a/f.py b/f.py\n+pre-existing-2\n",
            model_name_or_path="original-model",
        ),
    ]
    pred_path = prediction_path(run_dir, CONDITION_BASELINE)
    write_predictions(pred_path, pre_existing)

    summary = _generate_for_condition(
        run_dir=run_dir,
        condition=CONDITION_BASELINE,
        selected_ids=instance_ids,
        rows_by_id=rows_by_id,
        patcher=cast(OpenAICompatPatcher, _StubPatcher()),
        model_label="stub-model::baseline",
        resume=True,
        snapshot_dir=None,
        jobs=1,
        openai_async=False,
        **_HEALTH_KWARGS,
    )

    assert summary.resumed == 2
    assert summary.generated == 3
    assert summary.total_instances == 5

    records = read_predictions(summary.predictions_path)
    assert len(records) == 5
    assert {r.instance_id for r in records} == set(instance_ids)

    # Verify pre-existing records are preserved with original patches.
    by_id = {r.instance_id: r for r in records}
    assert "pre-existing-1" in by_id["inst-1"].model_patch
    assert "pre-existing-2" in by_id["inst-2"].model_patch

    # Newly generated records should have the stub's patch content.
    for iid in ("inst-3", "inst-4", "inst-5"):
        assert "new-generation" in by_id[iid].model_patch


# ------------------------------------------------------------------
# Test 2: Resume with no existing file generates everything
# ------------------------------------------------------------------


def test_generate_for_condition_resume_no_existing_file(tmp_path: Path) -> None:
    instance_ids = [f"inst-{i}" for i in range(1, 4)]
    rows_by_id = _make_rows(instance_ids)

    run_dir = tmp_path / "evals" / "swebench_runs" / "resume-no-file"
    ensure_run_layout(run_dir, conditions=[CONDITION_BASELINE])

    # Do NOT create the predictions file — resume should still work.
    summary = _generate_for_condition(
        run_dir=run_dir,
        condition=CONDITION_BASELINE,
        selected_ids=instance_ids,
        rows_by_id=rows_by_id,
        patcher=cast(OpenAICompatPatcher, _StubPatcher()),
        model_label="stub-model::baseline",
        resume=True,
        snapshot_dir=None,
        jobs=1,
        openai_async=False,
        **_HEALTH_KWARGS,
    )

    assert summary.resumed == 0
    assert summary.generated == 3
    assert summary.total_instances == 3

    records = read_predictions(summary.predictions_path)
    assert len(records) == 3


# ------------------------------------------------------------------
# Test 3: Resume when all instances already done
# ------------------------------------------------------------------


def test_generate_for_condition_resume_all_already_done(tmp_path: Path) -> None:
    instance_ids = [f"inst-{i}" for i in range(1, 4)]
    rows_by_id = _make_rows(instance_ids)

    run_dir = tmp_path / "evals" / "swebench_runs" / "resume-all-done"
    ensure_run_layout(run_dir, conditions=[CONDITION_BASELINE])

    # Pre-populate all instances.
    all_records = [
        PredictionRecord(
            instance_id=iid,
            model_patch=f"diff --git a/f.py b/f.py\n+already-done-{iid}\n",
            model_name_or_path="original-model",
        )
        for iid in instance_ids
    ]
    pred_path = prediction_path(run_dir, CONDITION_BASELINE)
    write_predictions(pred_path, all_records)

    summary = _generate_for_condition(
        run_dir=run_dir,
        condition=CONDITION_BASELINE,
        selected_ids=instance_ids,
        rows_by_id=rows_by_id,
        patcher=cast(OpenAICompatPatcher, _StubPatcher()),
        model_label="stub-model::baseline",
        resume=True,
        snapshot_dir=None,
        jobs=1,
        openai_async=False,
        **_HEALTH_KWARGS,
    )

    assert summary.resumed == 3
    assert summary.generated == 0
    assert summary.total_instances == 3

    records = read_predictions(summary.predictions_path)
    assert len(records) == 3

    # All records should retain their original patches.
    by_id = {r.instance_id: r for r in records}
    for iid in instance_ids:
        assert f"already-done-{iid}" in by_id[iid].model_patch
