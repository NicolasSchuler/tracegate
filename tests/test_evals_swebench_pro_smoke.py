"""Tests for SWE-bench Pro scoring smoke-test subset preparation.

Mirrors the structure of ``test_evals_swebench_smoke.py`` — exercises subset
creation, instance counting, and error handling for the Pro smoke module.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from evals.swebench.types import PredictionRecord
from evals.swebench_pro.io import write_instance_ids, write_predictions
from evals.swebench_pro.smoke import prepare_scoring_smoke_subset

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _records(ids: list[str], model_tag: str) -> list[PredictionRecord]:
    return [
        PredictionRecord(
            instance_id=item,
            model_patch="diff --git a/a.py b/a.py\n",
            model_name_or_path=model_tag,
        )
        for item in ids
    ]


def _setup_source_run(
    tmp_path: Path,
    ids: list[str],
    *,
    dirname: str = "source-run",
) -> Path:
    """Create a valid source run directory with paired predictions."""
    source = tmp_path / dirname
    (source / "baseline").mkdir(parents=True, exist_ok=True)
    (source / "llmdebug").mkdir(parents=True, exist_ok=True)
    write_predictions(
        source / "baseline" / "predictions.jsonl",
        _records(ids, "baseline-model"),
    )
    write_predictions(
        source / "llmdebug" / "predictions.jsonl",
        _records(ids, "llmdebug-model"),
    )
    write_instance_ids(source / "instance_ids.txt", ids)
    return source


# ---------------------------------------------------------------------------
# Happy path
# ---------------------------------------------------------------------------


class TestPrepareScoringSmoke:
    def test_writes_paired_subset(self, tmp_path: Path) -> None:
        ids = ["inst-1", "inst-2", "inst-3"]
        source = _setup_source_run(tmp_path, ids)
        output = tmp_path / "smoke-run"

        summary = prepare_scoring_smoke_subset(
            source_run_dir=source,
            output_run_dir=output,
            max_instances=2,
            conditions="baseline,llmdebug",
        )

        assert summary.instance_count == 2
        assert summary.conditions == ("baseline", "llmdebug")
        assert (output / "baseline" / "predictions.jsonl").exists()
        assert (output / "llmdebug" / "predictions.jsonl").exists()
        assert (output / "COPY_INSTRUCTIONS.txt").exists()
        assert (output / "run_index.json").exists()
        assert (output / "run_metadata.json").exists()
        assert (output / "instance_ids.txt").read_text(encoding="utf-8").splitlines() == [
            "inst-1",
            "inst-2",
        ]

    def test_single_instance(self, tmp_path: Path) -> None:
        ids = ["inst-1", "inst-2", "inst-3"]
        source = _setup_source_run(tmp_path, ids)
        output = tmp_path / "smoke-1"

        summary = prepare_scoring_smoke_subset(
            source_run_dir=source,
            output_run_dir=output,
            max_instances=1,
            conditions="baseline,llmdebug",
        )

        assert summary.instance_count == 1
        assert (output / "instance_ids.txt").read_text(encoding="utf-8").splitlines() == [
            "inst-1",
        ]

    def test_max_instances_exceeds_available(self, tmp_path: Path) -> None:
        """When max_instances > available instances, all are selected."""
        ids = ["inst-1", "inst-2"]
        source = _setup_source_run(tmp_path, ids)
        output = tmp_path / "smoke-big"

        summary = prepare_scoring_smoke_subset(
            source_run_dir=source,
            output_run_dir=output,
            max_instances=100,
            conditions="baseline,llmdebug",
        )

        assert summary.instance_count == 2

    def test_output_has_run_metadata(self, tmp_path: Path) -> None:
        ids = ["inst-1"]
        source = _setup_source_run(tmp_path, ids)
        output = tmp_path / "smoke-meta"

        prepare_scoring_smoke_subset(
            source_run_dir=source,
            output_run_dir=output,
            max_instances=1,
            conditions="baseline,llmdebug",
        )

        metadata_text = (output / "run_metadata.json").read_text(encoding="utf-8")
        assert "swebench_pro_score_smoke_subset_v1" in metadata_text
        assert str(source) in metadata_text

    def test_summary_has_correct_source_dir(self, tmp_path: Path) -> None:
        ids = ["inst-1"]
        source = _setup_source_run(tmp_path, ids)
        output = tmp_path / "smoke-src"

        summary = prepare_scoring_smoke_subset(
            source_run_dir=source,
            output_run_dir=output,
            max_instances=1,
            conditions="baseline,llmdebug",
        )

        assert summary.source_run_dir == source
        assert summary.output_run_dir == output

    def test_empty_output_dir_is_accepted(self, tmp_path: Path) -> None:
        """An existing but empty output directory should not raise."""
        ids = ["inst-1"]
        source = _setup_source_run(tmp_path, ids)
        output = tmp_path / "smoke-empty"
        output.mkdir(parents=True, exist_ok=True)

        summary = prepare_scoring_smoke_subset(
            source_run_dir=source,
            output_run_dir=output,
            max_instances=1,
            conditions="baseline,llmdebug",
        )

        assert summary.instance_count == 1


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------


class TestPrepareScokeErrors:
    def test_fails_when_output_not_empty(self, tmp_path: Path) -> None:
        ids = ["inst-1", "inst-2"]
        source = _setup_source_run(tmp_path, ids)

        output = tmp_path / "smoke-run"
        output.mkdir(parents=True, exist_ok=True)
        (output / "keep.txt").write_text("x", encoding="utf-8")

        with pytest.raises(RuntimeError, match="already exists and is not empty"):
            prepare_scoring_smoke_subset(
                source_run_dir=source,
                output_run_dir=output,
                max_instances=1,
                conditions="baseline,llmdebug",
            )

    def test_fails_when_source_does_not_exist(self, tmp_path: Path) -> None:
        source = tmp_path / "nonexistent"
        output = tmp_path / "smoke-run"

        with pytest.raises(RuntimeError, match="does not exist"):
            prepare_scoring_smoke_subset(
                source_run_dir=source,
                output_run_dir=output,
                max_instances=1,
                conditions="baseline,llmdebug",
            )

    def test_fails_when_max_instances_less_than_one(self, tmp_path: Path) -> None:
        ids = ["inst-1"]
        source = _setup_source_run(tmp_path, ids)
        output = tmp_path / "smoke-zero"

        with pytest.raises(ValueError, match="max_instances must be >= 1"):
            prepare_scoring_smoke_subset(
                source_run_dir=source,
                output_run_dir=output,
                max_instances=0,
                conditions="baseline,llmdebug",
            )

    def test_fails_with_single_condition(self, tmp_path: Path) -> None:
        ids = ["inst-1"]
        source = _setup_source_run(tmp_path, ids)
        output = tmp_path / "smoke-single"

        with pytest.raises(ValueError, match="conditions must include exactly baseline,llmdebug"):
            prepare_scoring_smoke_subset(
                source_run_dir=source,
                output_run_dir=output,
                max_instances=1,
                conditions="baseline",
            )

    def test_fails_when_source_missing_condition_dir(self, tmp_path: Path) -> None:
        """Source has baseline but not llmdebug predictions."""
        source = tmp_path / "partial-run"
        (source / "baseline").mkdir(parents=True, exist_ok=True)
        write_predictions(
            source / "baseline" / "predictions.jsonl",
            _records(["inst-1"], "baseline-model"),
        )
        write_instance_ids(source / "instance_ids.txt", ["inst-1"])
        output = tmp_path / "smoke-partial"

        with pytest.raises((RuntimeError, ValueError)):
            prepare_scoring_smoke_subset(
                source_run_dir=source,
                output_run_dir=output,
                max_instances=1,
                conditions="baseline,llmdebug",
            )
